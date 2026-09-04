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


# --------------------------------------------------------------------------- #
# round-trip validation vs. Access Layer write stamps
# --------------------------------------------------------------------------- #


def test_access_layer_stamps_are_recognised_but_dd_version_is_not():
    """`version_put` mixes writer identity with a real compatibility signal.

    `access_layer`/`access_layer_language` say which library wrote the replica.
    `data_dictionary` says which DD it conforms to -- a genuine difference worth
    failing on, so it must not be swept up by the same filter.
    """
    f = replication._is_write_provenance
    assert f("pf_passive.ids_properties.version_put.access_layer")
    assert f("pf_passive.ids_properties.version_put.access_layer_language")
    assert not f("pf_passive.ids_properties.version_put.data_dictionary")
    assert not f("pf_passive.current")
    assert not f("equilibrium.time_slice.0.profiles_2d.0.psi")


def _round_trip_against(monkeypatch, replica, sent_stamps=None):
    """Run _round_trip with a supplied replica, returning its summary.

    `sent` mirrors what replicate_stage actually passes: the object *after*
    save_ods, which the Access Layer has already stamped in memory. The
    comparison runs with scope="reference", so a path absent from `sent` is
    never compared at all -- which is why the stamps have to be here for this
    to reproduce the production case.
    """
    monkeypatch.setattr("vaft.database.load", lambda *a, **k: replica)
    sent = ODS(consistency_check=False)
    sent["pf_passive.ids_properties.comment"] = "eddy currents"
    sent["pf_passive.loop.0.name"] = "PF1"
    for leaf, value in (sent_stamps or {}).items():
        sent[f"pf_passive.ids_properties.version_put.{leaf}"] = value
    return sent, replication._round_trip(
        sent, shot=39915, source="main", ids=("pf_passive",), occurrence=0
    )


def test_a_replica_stamped_by_the_access_layer_still_validates(monkeypatch):
    """The source is OMAS JSON and has no Access Layer, so the replica always
    carries stamps the source structurally cannot have. Before this was
    excluded, no replication could ever reach `validated`."""
    replica = ODS(consistency_check=False)
    replica["pf_passive.ids_properties.comment"] = "eddy currents"
    replica["pf_passive.loop.0.name"] = "PF1"
    replica["pf_passive.ids_properties.version_put.access_layer"] = "5.7.2"
    replica["pf_passive.ids_properties.version_put.access_layer_language"] = "IMAS-Python 2.3.0"

    # The writer stamped its own identity into the object it sent.
    _sent, summary = _round_trip_against(
        monkeypatch,
        replica,
        sent_stamps={"access_layer": "5.6.0", "access_layer_language": "IMAS-Python 2.1.0"},
    )

    assert summary["passed"] is True
    assert summary["write_provenance_excluded"] == 2


def test_a_real_difference_still_fails_the_round_trip(monkeypatch):
    """Excluding the stamps must not blunt the check it exists to perform."""
    replica = ODS(consistency_check=False)
    replica["pf_passive.ids_properties.comment"] = "eddy currents"
    replica["pf_passive.loop.0.name"] = "PF9-WRONG"
    replica["pf_passive.ids_properties.version_put.access_layer"] = "5.7.2"

    with pytest.raises(replication.RoundTripValidationError, match="loop.0.name"):
        _round_trip_against(monkeypatch, replica, sent_stamps={"access_layer": "5.6.0"})


def test_a_data_dictionary_mismatch_still_fails(monkeypatch):
    """The DD version is a compatibility signal, not writer identity."""
    replica = ODS(consistency_check=False)
    replica["pf_passive.ids_properties.comment"] = "eddy currents"
    replica["pf_passive.loop.0.name"] = "PF1"
    replica["pf_passive.ids_properties.version_put.data_dictionary"] = "9.9.9"

    with pytest.raises(replication.RoundTripValidationError, match="data_dictionary"):
        _round_trip_against(monkeypatch, replica, sent_stamps={"data_dictionary": "3.41.0"})


# --------------------------------------------------------------------------- #
# an unavailable channel is empty locally and absent remotely
# --------------------------------------------------------------------------- #


def _entry(kind, path="magnetics.b_field_pol_probe.0.field.data", reference_shape=None):
    from vaft.omas.comparison import ComparisonEntry, DifferenceKind, ParityClassification

    return ComparisonEntry(
        path=path,
        classification=ParityClassification.REGRESSION,
        kind=kind,
        message="",
        reference_shape=reference_shape,
        candidate_shape=None,
        reference_type="ndarray",
        candidate_type=None,
        rtol=None,
        atol=None,
        max_abs_error=None,
        max_rel_error=None,
        policy_rule=None,
        policy_note=None,
    )


def test_an_empty_local_array_absent_from_the_replica_is_not_a_regression():
    """A disabled coil or unrecorded field code arrives as a zero-length array.

    IMAS stores no node for one, so the replica lacks the path. Both sides say
    "no data"; only the representation differs.
    """
    from vaft.omas.comparison import DifferenceKind

    assert replication._is_empty_left_unwritten(
        _entry(DifferenceKind.MISSING_CANDIDATE, reference_shape=(0,))
    )


def test_real_data_missing_from_the_replica_still_fails():
    """The narrowness is the point: a populated path that did not arrive is
    data loss, which is exactly what this check exists to catch."""
    from vaft.omas.comparison import DifferenceKind

    assert not replication._is_empty_left_unwritten(
        _entry(DifferenceKind.MISSING_CANDIDATE, reference_shape=(4096,))
    )
    # A value difference is never explained away by emptiness.
    assert not replication._is_empty_left_unwritten(
        _entry(DifferenceKind.NUMERICAL, reference_shape=(0,))
    )
    # No shape information means no grounds to excuse it.
    assert not replication._is_empty_left_unwritten(
        _entry(DifferenceKind.MISSING_CANDIDATE, reference_shape=None)
    )


def test_a_replica_missing_only_empty_channels_validates(monkeypatch):
    """End to end: the local product carries an unavailable channel, the
    replica does not, and the round trip still passes."""
    import numpy as np

    replica = ODS(consistency_check=False)
    replica["pf_passive.ids_properties.comment"] = "eddy currents"
    replica["pf_passive.loop.0.name"] = "PF1"

    monkeypatch.setattr("vaft.database.load", lambda *a, **k: replica)
    sent = ODS(consistency_check=False)
    sent["pf_passive.ids_properties.comment"] = "eddy currents"
    sent["pf_passive.loop.0.name"] = "PF1"
    sent["pf_passive.loop.0.current.data"] = np.array([])  # unavailable channel

    summary = replication._round_trip(
        sent, shot=39915, source="main", ids=("pf_passive",), occurrence=0
    )

    assert summary["passed"] is True
    assert summary["empty_arrays_unwritten"] >= 1


# --------------------------------------------------------------------------- #
# a busy server is not a missing namespace
# --------------------------------------------------------------------------- #


def test_a_genuine_404_is_reported_as_a_missing_namespace(monkeypatch):
    from vaft.database import utils
    from vaft.database.sources import MissingSourceError

    def absent(path, mode=None):
        raise OSError(404, "Not Found")

    monkeypatch.setattr(utils.h5pyd, "Folder", absent)
    with pytest.raises(MissingSourceError, match="hstouch"):
        utils.require_source_exists("main")


def test_a_transient_failure_is_raised_as_itself(monkeypatch):
    """Under concurrent load HSDS answers some probes with anything but 404.

    Reporting those as "the namespace does not exist" sends an operator to
    provision a folder that is already there -- which happened across 1298
    replications -- and, because a missing namespace is not retryable, it also
    killed each shot outright instead of retrying a blip.
    """
    from vaft.database import utils
    from vaft.database.sources import MissingSourceError

    def busy(path, mode=None):
        raise OSError(503, "Service Unavailable")

    monkeypatch.setattr(utils.h5pyd, "Folder", busy)
    with pytest.raises(OSError) as excinfo:
        utils.require_source_exists("main")
    assert not isinstance(excinfo.value, MissingSourceError)
    assert excinfo.value.args[0] == 503


def test_a_transient_probe_failure_is_retried(staged, monkeypatch):
    from vaft.database import replication as R

    calls = []

    def flaky(source):
        calls.append(source)
        if len(calls) == 1:
            raise OSError(503, "Service Unavailable")

    monkeypatch.setattr("vaft.database.utils.require_source_exists", flaky)
    monkeypatch.setattr(R, "_fetch_remote_master", lambda *a, **k: None)
    monkeypatch.setattr(R, "merge_remote_master", lambda *a, **k: ())
    monkeypatch.setattr(R, "_round_trip", lambda ods, **kw: {"passed": True})
    monkeypatch.setattr("vaft.database.save", lambda *a, **k: "uri")

    record = replicate_stage(
        "diagnostics", 39915, filedb=staged, attempts=3, retry_delay=0
    )

    assert len(calls) == 2
    assert record.validated


def test_a_missing_namespace_is_not_retried(staged, monkeypatch):
    """Provisioning is an operator action; three attempts cannot fix it."""
    from vaft.database import replication as R
    from vaft.database.sources import MissingSourceError

    calls = []

    def absent(source):
        calls.append(source)
        raise MissingSourceError(source, "domain not found")

    fetches = []

    def fetch(source, shot, dest):
        fetches.append(source)
        return None

    monkeypatch.setattr("vaft.database.utils.require_source_exists", absent)
    monkeypatch.setattr(R, "_fetch_remote_master", fetch)
    monkeypatch.setattr(R, "merge_remote_master", lambda *a, **k: ())
    monkeypatch.setattr("vaft.database.save", lambda *a, **k: "uri")

    with pytest.raises(MissingSourceError):
        replicate_stage("diagnostics", 39915, filedb=staged, attempts=3, retry_delay=0)

    assert len(calls) == 1
    # The probe comes first: an unprovisioned namespace never costs a fetch.
    assert fetches == []
