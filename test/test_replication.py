"""Replicating finalized canonical FileDB products into named HSDS sources (#94).

The invariant under test throughout: a FileDB product existing on disk never
implies it reached HSDS, and one stage's replication never disturbs another's.
"""

import json
from pathlib import Path

import h5py
import pytest
from omas import ODS

from vaft.database import replication
from vaft.database.filedb import FileDB
from vaft.database.replication import (
    ProductNotEligibleError,
    ReplicationRecord,
    StageNotReplicableError,
    is_reusable,
    replicate_stage,
)
from vaft.database.staging import external_h5_links, merge_master_links


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _product_ods() -> ODS:
    """A diagnostics-shaped product: owned IDS plus one the stage does not own."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["magnetics.ids_properties.comment"] = "owned by diagnostics"
    ods["barometry.ids_properties.comment"] = "owned by diagnostics"
    ods["equilibrium.ids_properties.comment"] = "owned by EFIT, not by diagnostics"
    return ods


@pytest.fixture
def staged(tmp_path):
    """A FileDB with one completed diagnostics product for shot 39915."""
    from vaft.omas import save as save_local

    db = FileDB(tmp_path)
    product = db.omas_product("diagnostics", shot=39915)
    product.parent.mkdir(parents=True, exist_ok=True)
    save_local(_product_ods(), product)

    manifest = db.omas_manifest("diagnostics", shot=39915)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"stage": "diagnostics", "status": "success"}))
    return db


def _patch_remote(monkeypatch, *, sent):
    """Stand in for HSDS: capture what a save would have written."""
    monkeypatch.setattr(replication, "_fetch_remote_master", lambda *a, **k: None)
    monkeypatch.setattr(replication, "merge_remote_master", lambda *a, **k: ())
    monkeypatch.setattr(
        replication, "_round_trip", lambda ods, **kwargs: {"passed": True}
    )
    monkeypatch.setattr(
        "vaft.database.utils.require_source_exists", lambda source: None
    )

    def fake_save(ods, shot, *, source=None, occurrence=None, **kwargs):
        sent.append({"shot": shot, "source": source, "ids": sorted(ods.keys())})
        return f"hdf5://{source}/{shot}/"

    monkeypatch.setattr("vaft.database.save", fake_save)


# --------------------------------------------------------------------------- #
# what may be replicated at all
# --------------------------------------------------------------------------- #


def test_a_stage_with_no_destination_is_refused(tmp_path):
    with pytest.raises(StageNotReplicableError, match="not replicated to HSDS"):
        replicate_stage("static", 39915, filedb=FileDB(tmp_path))


def test_ideal_gpec_replication_is_refused_and_names_its_issue(tmp_path):
    with pytest.raises(StageNotReplicableError, match=r"#95"):
        replicate_stage("gpec_ideal", 39915, filedb=FileDB(tmp_path))


def test_a_missing_manifest_is_refused_rather_than_assumed_complete(tmp_path):
    db = FileDB(tmp_path)
    product = db.omas_product("diagnostics", shot=39915)
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_text("{}")

    with pytest.raises(ProductNotEligibleError, match="No stage manifest"):
        replicate_stage("diagnostics", 39915, filedb=db)


@pytest.mark.parametrize("status", ["skipped", "blocked", "failed"])
def test_an_unfinished_stage_has_nothing_to_replicate(staged, status):
    manifest = staged.omas_manifest("diagnostics", shot=39915)
    manifest.write_text(json.dumps({"stage": "diagnostics", "status": status}))

    with pytest.raises(ProductNotEligibleError, match="nothing to replicate"):
        replicate_stage("diagnostics", 39915, filedb=staged)


def test_a_manifest_with_no_status_is_refused(staged):
    manifest = staged.omas_manifest("diagnostics", shot=39915)
    manifest.write_text(json.dumps({"stage": "diagnostics"}))

    with pytest.raises(ProductNotEligibleError, match="records no status"):
        replicate_stage("diagnostics", 39915, filedb=staged)


# --------------------------------------------------------------------------- #
# owned-subtree semantics
# --------------------------------------------------------------------------- #


def test_only_the_ids_the_stage_owns_are_sent(staged, monkeypatch):
    sent = []
    _patch_remote(monkeypatch, sent=sent)

    record = replicate_stage("diagnostics", 39915, filedb=staged)

    assert sent[0]["source"] == "main"
    # equilibrium is in the product but belongs to EFIT; it must not travel.
    assert "equilibrium" not in sent[0]["ids"]
    assert set(record.ids) == {"magnetics", "barometry"}
    # Provenance rides along so the replica knows which shot it describes.
    assert "dataset_description" in sent[0]["ids"]


def test_a_product_carrying_none_of_its_owned_ids_is_refused(tmp_path, monkeypatch):
    from vaft.omas import save as save_local

    db = FileDB(tmp_path)
    product = db.omas_product("eddy", shot=39915)
    product.parent.mkdir(parents=True, exist_ok=True)
    ods = ODS(consistency_check=False)
    ods["magnetics.ids_properties.comment"] = "carried through, not owned"
    save_local(ods, product)
    manifest = db.omas_manifest("eddy", shot=39915)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"status": "success"}))

    _patch_remote(monkeypatch, sent=[])
    with pytest.raises(ProductNotEligibleError, match="none of the IDS this stage owns"):
        replicate_stage("eddy", 39915, filedb=db)


def test_the_record_names_the_source_actually_written(staged, monkeypatch):
    _patch_remote(monkeypatch, sent=[])

    record = replicate_stage("diagnostics", 39915, filedb=staged)

    assert record.source == "main"
    assert record.remote_uri == "hdf5://main/39915/"
    written = json.loads(
        staged.omas_replication_record("diagnostics", shot=39915).read_text()
    )
    assert written["source"] == "main"


# --------------------------------------------------------------------------- #
# local completion vs. remote replication
# --------------------------------------------------------------------------- #


def test_a_finished_product_is_not_replicated_until_it_is(staged):
    """The two states have separate homes and neither implies the other."""
    assert staged.omas_product("diagnostics", shot=39915).exists()
    assert staged.omas_manifest("diagnostics", shot=39915).exists()
    assert not staged.omas_replication_record("diagnostics", shot=39915).exists()


def test_repeating_a_replication_reuses_the_record(staged, monkeypatch):
    sent = []
    _patch_remote(monkeypatch, sent=sent)

    first = replicate_stage("diagnostics", 39915, filedb=staged)
    second = replicate_stage("diagnostics", 39915, filedb=staged)

    assert len(sent) == 1, "an unchanged product must not be re-sent"
    assert second == first
    assert second.validated


def test_a_rebuilt_product_is_replicated_again(staged, monkeypatch):
    from vaft.omas import save as save_local

    sent = []
    _patch_remote(monkeypatch, sent=sent)
    replicate_stage("diagnostics", 39915, filedb=staged)

    # Same path, different bytes: the recorded hash no longer describes it.
    changed = _product_ods()
    changed["magnetics.ids_properties.comment"] = "reprocessed"
    save_local(changed, staged.omas_product("diagnostics", shot=39915))

    replicate_stage("diagnostics", 39915, filedb=staged)

    assert len(sent) == 2


def test_a_record_alone_does_not_make_a_replication_reusable():
    record = ReplicationRecord(
        stage="diagnostics", shot=39915, source="main",
        remote_uri="hdf5://main/39915/", ids=("magnetics",), occurrence=0,
        product_sha256="abc", state="validated", attempts=1, started_at="now",
    )

    assert is_reusable(record, product_sha256="abc", require_validation=True)
    # A different product at the same path is a different product.
    assert not is_reusable(record, product_sha256="def", require_validation=True)
    # Sent but never checked does not satisfy a contract that wants validation.
    from dataclasses import replace

    sent_only = replace(record, state="replicated")
    assert not is_reusable(sent_only, product_sha256="abc", require_validation=True)
    assert is_reusable(sent_only, product_sha256="abc", require_validation=False)


def test_a_failed_replication_is_recorded_and_raises(staged, monkeypatch):
    _patch_remote(monkeypatch, sent=[])

    def explode(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("vaft.database.save", explode)

    with pytest.raises(replication.ReplicationError, match="connection reset"):
        replicate_stage("diagnostics", 39915, filedb=staged, attempts=1)

    record = replication.read_record(
        staged.omas_replication_record("diagnostics", shot=39915)
    )
    assert record.state == "failed"
    assert not record.replicated
    # A failed replication must not look reusable on the next run.
    assert not is_reusable(
        record, product_sha256=record.product_sha256, require_validation=False
    )


def test_a_transient_failure_is_retried(staged, monkeypatch):
    _patch_remote(monkeypatch, sent=[])
    calls = []

    def flaky(ods, shot, *, source=None, occurrence=None, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return f"hdf5://{source}/{shot}/"

    monkeypatch.setattr("vaft.database.save", flaky)

    record = replicate_stage(
        "diagnostics", 39915, filedb=staged, attempts=3, retry_delay=0
    )

    assert len(calls) == 2
    assert record.validated


# --------------------------------------------------------------------------- #
# the master merge: one stage must not hide another
# --------------------------------------------------------------------------- #


def _master(path: Path, links) -> Path:
    with h5py.File(path, "w") as handle:
        for name in links:
            handle[name] = h5py.ExternalLink(f"{name}.h5", f"/{name}")
    return path


def test_merging_keeps_every_stage_visible_to_the_eager_reader(tmp_path):
    """stage_imas_shot resolves a shot's contents from the master's links.

    A stage that uploads a master describing only its own IDS would make every
    other stage's IDS disappear from the shot, even though the files remain.
    """
    previous = _master(tmp_path / "previous.h5", ["magnetics", "pf_active"])
    current = _master(tmp_path / "current.h5", ["equilibrium"])

    added = merge_master_links(
        current,
        previous,
        present_files=["magnetics.h5", "pf_active.h5", "equilibrium.h5"],
    )

    assert added == ("magnetics", "pf_active")
    assert external_h5_links(current) == [
        "equilibrium.h5",
        "magnetics.h5",
        "pf_active.h5",
    ]


def test_merging_drops_links_whose_file_is_gone(tmp_path):
    previous = _master(tmp_path / "previous.h5", ["magnetics", "removed"])
    current = _master(tmp_path / "current.h5", ["equilibrium"])

    merge_master_links(current, previous, present_files=["magnetics.h5", "equilibrium.h5"])

    assert "removed.h5" not in external_h5_links(current)


def test_the_stage_that_just_wrote_wins_a_name_collision(tmp_path):
    previous = _master(tmp_path / "previous.h5", ["equilibrium"])
    current = tmp_path / "current.h5"
    with h5py.File(current, "w") as handle:
        handle["equilibrium"] = h5py.ExternalLink("equilibrium.h5", "/refreshed")

    added = merge_master_links(current, previous, present_files=["equilibrium.h5"])

    assert added == ()
    with h5py.File(current, "r") as handle:
        assert handle.get("equilibrium", getlink=True).path == "/refreshed"


def test_a_first_write_has_no_previous_master_to_merge(tmp_path):
    current = _master(tmp_path / "current.h5", ["equilibrium"])

    assert merge_master_links(current, None, present_files=["equilibrium.h5"]) == ()
    assert external_h5_links(current) == ["equilibrium.h5"]


# --------------------------------------------------------------------------- #
# the master must survive a retry and a flaky fetch
# --------------------------------------------------------------------------- #


def test_the_previous_master_is_captured_once_not_per_attempt(staged, monkeypatch):
    """A retry must merge against the state before the first attempt.

    Re-reading the remote master between attempts would merge against the
    master this stage's own failed attempt left behind, dropping every other
    stage's IDS from the shot for good.
    """
    monkeypatch.setattr(
        "vaft.database.utils.require_source_exists", lambda source: None
    )
    monkeypatch.setattr(
        replication, "_round_trip", lambda ods, **kwargs: {"passed": True}
    )
    fetches = []
    monkeypatch.setattr(
        replication,
        "_fetch_remote_master",
        lambda source, shot, target: fetches.append(target) or None,
    )
    monkeypatch.setattr(replication, "merge_remote_master", lambda *a, **k: ())

    calls = []

    def flaky(ods, shot, *, source=None, occurrence=None, **kwargs):
        calls.append(source)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return f"hdf5://{source}/{shot}/"

    monkeypatch.setattr("vaft.database.save", flaky)

    replicate_stage("diagnostics", 39915, filedb=staged, attempts=3, retry_delay=0)

    assert len(calls) == 3
    assert len(fetches) == 1, "the pre-write master must be read exactly once"


def test_an_unfetchable_existing_master_is_an_error_not_an_absence(monkeypatch):
    """Reading a fetch failure as "no master" would skip the merge silently."""
    monkeypatch.setattr(replication, "_remote_entries", lambda s, shot: ("master.h5",))

    def failed_get(uri, target):
        return target  # leaves nothing behind

    monkeypatch.setattr("vaft.database.transport.run_hsget", failed_get)

    with pytest.raises(replication.ReplicationError, match="could not be materialized"):
        replication._fetch_remote_master("main", 39915, Path("/nonexistent/master.h5"))


def test_a_shot_with_no_folder_yet_simply_has_no_master(monkeypatch):
    monkeypatch.setattr(replication, "_remote_entries", lambda s, shot: ())

    assert replication._fetch_remote_master("main", 39915, Path("/x/master.h5")) is None


def test_derived_images_are_not_mistaken_for_stage_ids():
    from vaft.database.h5image import derived_filename

    entries = ("equilibrium.h5", derived_filename("equilibrium.h5"), "master.h5")

    assert replication._remote_canonical_files(entries) == ("equilibrium.h5",)


# --------------------------------------------------------------------------- #
# a source is the union of what its stages sent, not one cumulative product
# --------------------------------------------------------------------------- #


def _stage_product(db, stage, shot, ods, status="success"):
    from vaft.omas import save as save_local

    product = db.omas_product(stage, shot=shot)
    product.parent.mkdir(parents=True, exist_ok=True)
    save_local(ods, product)
    manifest = db.omas_manifest(stage, shot=shot)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"stage": stage, "status": status}))


def test_a_vacuum_shot_reaches_main_without_an_equilibrium(tmp_path, monkeypatch):
    """A shot that never reached EFIT still replicates what it did produce.

    Requiring a cumulative post-EFIT product would mean only shots with a
    successful reconstruction exist in `main` at all.
    """
    db = FileDB(tmp_path)
    diagnostics = ODS(consistency_check=False)
    diagnostics["magnetics.ids_properties.comment"] = "vacuum shot"
    _stage_product(db, "diagnostics", 45000, diagnostics)

    eddy = ODS(consistency_check=False)
    eddy["magnetics.ids_properties.comment"] = "carried through"
    eddy["pf_passive.ids_properties.comment"] = "computed"
    _stage_product(db, "eddy", 45000, eddy)

    # EFIT was skipped: its product exists, but the manifest says so.
    _stage_product(db, "efit", 45000, ODS(consistency_check=False), status="skipped")

    sent = []
    _patch_remote(monkeypatch, sent=sent)

    replicate_stage("diagnostics", 45000, filedb=db)
    replicate_stage("eddy", 45000, filedb=db)
    with pytest.raises(ProductNotEligibleError):
        replicate_stage("efit", 45000, filedb=db)

    replicated = {ids for call in sent for ids in call["ids"]}
    assert {"magnetics", "pf_passive"} <= replicated
    assert "equilibrium" not in replicated
    assert {call["source"] for call in sent} == {"main"}


def test_eddy_does_not_overwrite_what_diagnostics_wrote(tmp_path, monkeypatch):
    db = FileDB(tmp_path)
    eddy = ODS(consistency_check=False)
    eddy["magnetics.ids_properties.comment"] = "carried through from diagnostics"
    eddy["pf_passive.ids_properties.comment"] = "computed here"
    _stage_product(db, "eddy", 45000, eddy)

    sent = []
    _patch_remote(monkeypatch, sent=sent)
    replicate_stage("eddy", 45000, filedb=db)

    assert "magnetics" not in sent[0]["ids"]
    assert "pf_passive" in sent[0]["ids"]


def test_the_refinement_and_the_baseline_land_in_different_sources(tmp_path, monkeypatch):
    """main must not acquire a CHEASE-refined equilibrium."""
    db = FileDB(tmp_path)
    for stage in ("efit", "chease"):
        ods = ODS(consistency_check=False)
        ods["equilibrium.ids_properties.comment"] = stage
        _stage_product(db, stage, 39915, ods)

    sent = []
    _patch_remote(monkeypatch, sent=sent)
    baseline = replicate_stage("efit", 39915, filedb=db)
    refined = replicate_stage("chease", 39915, filedb=db)

    assert baseline.source == "main"
    assert refined.source == "chease-mhd-stability"
    assert baseline.ids == refined.ids == ("equilibrium",)


def test_a_failed_stability_replication_leaves_the_baseline_alone(tmp_path, monkeypatch):
    db = FileDB(tmp_path)
    for stage in ("efit", "mhd_linear"):
        ods = ODS(consistency_check=False)
        ods["equilibrium.ids_properties.comment" if stage == "efit"
            else "mhd_linear.ids_properties.comment"] = stage
        _stage_product(db, stage, 39915, ods)

    _patch_remote(monkeypatch, sent=[])
    baseline = replicate_stage("efit", 39915, filedb=db)

    def only_stability_fails(ods, shot, *, source=None, **kwargs):
        if source == "chease-mhd-stability":
            raise RuntimeError("stability replication failed")
        return f"hdf5://{source}/{shot}/"

    monkeypatch.setattr("vaft.database.save", only_stability_fails)
    with pytest.raises(replication.ReplicationError):
        replicate_stage("mhd_linear", 39915, filedb=db, attempts=1)

    # The EFIT record is untouched: separate stages, separate records.
    assert replication.read_record(
        db.omas_replication_record("efit", shot=39915)
    ) == baseline
    assert (
        replication.read_record(
            db.omas_replication_record("mhd_linear", shot=39915)
        ).state
        == "failed"
    )


def test_a_missing_hsds_namespace_fails_with_the_provisioning_fix(staged, monkeypatch):
    """Nothing creates a top-level namespace, and nothing falls back to public."""
    from vaft.database.sources import MissingSourceError

    sent = []
    _patch_remote(monkeypatch, sent=sent)

    def unprovisioned(source):
        raise MissingSourceError(source, "domain not found")

    monkeypatch.setattr("vaft.database.utils.require_source_exists", unprovisioned)

    with pytest.raises(MissingSourceError, match="hstouch"):
        replicate_stage("diagnostics", 39915, filedb=staged)

    assert sent == [], "nothing is sent to a namespace that does not exist"


def test_per_code_and_mode_stability_detail_survives_replication(tmp_path, monkeypatch):
    """DCON, RDCON and STRIDE share one publication lineage but keep their identity."""
    from vaft.omas import load as load_local, save as save_local

    db = FileDB(tmp_path)
    ods = ODS(consistency_check=False)
    ods["mhd_linear.time"] = [0.3, 0.31]
    for index, (code, mode, status) in enumerate(
        [("dcon", 1, "success"), ("rdcon", 2, "no_output"), ("stride", 1, "failed")]
    ):
        base = f"mhd_linear.code.parameters.cases.{index}"
        ods[f"{base}.code"] = code
        ods[f"{base}.toroidal_mode"] = mode
        ods[f"{base}.status"] = status
    _stage_product(db, "mhd_linear", 39915, ods)

    captured = {}

    def capture(projected, shot, *, source=None, occurrence=None, **kwargs):
        captured["ods"] = projected
        return f"hdf5://{source}/{shot}/"

    _patch_remote(monkeypatch, sent=[])
    monkeypatch.setattr("vaft.database.save", capture)

    record = replicate_stage("mhd_linear", 39915, filedb=db)

    assert record.source == "chease-mhd-stability"
    replicated = captured["ods"]
    for index, code in enumerate(["dcon", "rdcon", "stride"]):
        base = f"mhd_linear.code.parameters.cases.{index}"
        assert replicated[f"{base}.code"] == code
    # A failed case travels as a failed case, not as an absence.
    assert replicated["mhd_linear.code.parameters.cases.2.status"] == "failed"


def test_a_landed_replica_that_fails_its_check_is_still_recorded(staged, monkeypatch):
    """The bytes are on the server; only the comparison failed.

    Recording nothing would send the next run back through the upload instead of
    straight to the check, and would lose the fact that the data is there.
    """
    _patch_remote(monkeypatch, sent=[])

    def mismatch(ods, **kwargs):
        raise replication.RoundTripValidationError("psi differs")

    monkeypatch.setattr(replication, "_round_trip", mismatch)

    with pytest.raises(replication.RoundTripValidationError):
        replicate_stage("diagnostics", 39915, filedb=staged)

    record = replication.read_record(
        staged.omas_replication_record("diagnostics", shot=39915)
    )
    assert record.state == "replicated"
    assert record.replicated and not record.validated
    assert "psi differs" in record.error
    # A run that demands validation still has work to do; one that does not, does not.
    assert not is_reusable(
        record, product_sha256=record.product_sha256, require_validation=True
    )
    assert is_reusable(
        record, product_sha256=record.product_sha256, require_validation=False
    )


def test_the_record_counts_the_attempt_that_actually_worked(staged, monkeypatch):
    _patch_remote(monkeypatch, sent=[])
    calls = []

    def flaky(ods, shot, *, source=None, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return f"hdf5://{source}/{shot}/"

    monkeypatch.setattr("vaft.database.save", flaky)

    record = replicate_stage(
        "diagnostics", 39915, filedb=staged, attempts=5, retry_delay=0
    )

    assert record.attempts == 2, "not the configured ceiling"
