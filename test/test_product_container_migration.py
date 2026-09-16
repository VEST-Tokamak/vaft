"""Migrating stored stage products onto their declared shape (#813).

Modelled on `test_filedb_grammar_relocation.py`, because `migrate-products` is
modelled on `relocate` -- with the difference that matters asserted here: this
one opens and rewrites files, so the tests cover what an interrupted run leaves
behind and what the verification actually proves, neither of which `relocate`
could get wrong.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from omas import ODS, save_omas_json

import vaft.database.product_migration as product_migration
import vaft.omas as vomas
from vaft.cli.filedb import main as filedb_main
from vaft.database.filedb import FileDB
from vaft.database.product_migration import (
    MIGRATIONS,
    ProductMigrationError,
    audit_product_containers,
    migrate_product_containers,
    sweep_superseded_products,
)
from vaft.database.sources import STAGE_REPLICATION

SHOT = 39915
TIME = np.linspace(0.0, 0.01, 40)


def _diagnostics_ods() -> ODS:
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = SHOT
    ods["pf_active.time"] = TIME
    ods["magnetics.ip.0.data"] = np.ones_like(TIME)
    return ods


def _over_carrying_eddy_ods() -> ODS:
    """An eddy product in the shape the old builder wrote: its own plus more."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = SHOT
    ods["magnetics.ip.0.data"] = np.ones_like(TIME)
    ods["pf_active.time"] = TIME
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.1, 0.2])
    ods["pf_passive.time"] = TIME
    ods["pf_passive.loop.0.current"] = np.arange(40, dtype=float)
    return ods


def _write_legacy(root: Path, stage: str, shot: int, ods: ODS) -> Path:
    output = root / "omas" / stage / str(shot) / "output"
    output.mkdir(parents=True, exist_ok=True)
    product = output / f"{stage}.json"
    save_omas_json(ods, str(product))

    metadata = root / "omas" / stage / str(shot) / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "manifest.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "shot": shot,
                "status": "success",
                "input": {"diagnostics_sha256": "deadbeef"},
                "output": {"name": f"{stage}.json", "sha256": "stale"},
            }
        ),
        encoding="utf-8",
    )
    return product


@pytest.fixture
def tree(tmp_path) -> Path:
    _write_legacy(tmp_path, "diagnostics", SHOT, _diagnostics_ods())
    _write_legacy(tmp_path, "eddy", SHOT, _over_carrying_eddy_ods())
    return tmp_path


def test_the_plan_names_every_product_that_has_to_move(tree):
    report = audit_product_containers(tree)

    assert {item.stage for item in report.pending} == {"diagnostics", "eddy"}
    assert {item.verify for item in report.pending} == {"bytes", "semantic"}
    assert report.settled == () and report.conflicting == ()
    assert report.safe_to_apply


def test_a_dry_run_changes_nothing(tree):
    before = {
        path: path.read_bytes() for path in sorted(tree.rglob("*")) if path.is_file()
    }

    audit_product_containers(tree)
    migrate_product_containers(tree, apply=False)

    after = {
        path: path.read_bytes() for path in sorted(tree.rglob("*")) if path.is_file()
    }
    assert after == before


def test_directory_sync_is_skipped_on_windows(monkeypatch, tmp_path):
    """Windows cannot open a directory descriptor for ``fsync``."""
    monkeypatch.setattr(product_migration, "IS_WINDOWS", True)

    def must_not_open(*_args, **_kwargs):
        pytest.fail("Windows must not try to open a directory for fsync")

    monkeypatch.setattr(product_migration.os, "open", must_not_open)
    product_migration._fsync_directory(tmp_path)


def test_a_migrated_container_decompresses_to_the_original_bytes(tree):
    """The (a) half is lossless, checked as bytes rather than as an ODS.

    Comparing two ODS objects would prove less and cost more: the claim is that
    the stored `.json` is exactly what the new writer gzips, so the strongest
    available check is that the gzip member equals the file byte for byte.
    """
    original = (tree / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json").read_bytes()

    migrate_product_containers(tree, stages=["diagnostics"], apply=True)

    migrated = tree / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json.gz"
    assert gzip.decompress(migrated.read_bytes()) == original


def test_a_migrated_container_is_byte_identical_to_a_freshly_written_product(tree, tmp_path):
    """Migrating and re-running the pipeline must not produce different files.

    `vaft.omas.save` gzips with `mtime=0`, and the migration uses the same
    parameters, so a product's `output.sha256` means the same thing whichever
    way it got there. Without this a migrated tree and a rebuilt one would
    disagree on every hash for no reason anyone could see.
    """
    migrate_product_containers(tree, stages=["diagnostics"], apply=True)
    migrated = tree / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json.gz"

    fresh = tmp_path / "fresh.json.gz"
    vomas.save(_diagnostics_ods(), fresh)

    assert migrated.read_bytes() == fresh.read_bytes()


def test_a_migrated_eddy_product_is_exactly_what_the_stage_owns(tree):
    migrate_product_containers(tree, stages=["eddy"], apply=True)

    product = vomas.load(tree / "omas/eddy" / str(SHOT) / "output/eddy.json.gz")
    top_level = {key.split(".")[0] for key in product.keys()}
    assert top_level == {"dataset_description"} | set(STAGE_REPLICATION["eddy"].ids)
    assert np.array_equal(
        np.asarray(product["pf_passive.loop.0.current"]), np.arange(40, dtype=float)
    )


def test_the_migrated_product_resolves_through_the_canonical_resolver(tree):
    """The point of the move: the resolver asks for a file that now exists."""
    migrate_product_containers(tree, apply=True)

    db = FileDB(tree)
    for stage in ("diagnostics", "eddy"):
        assert db.omas_product(stage, shot=SHOT).is_file()


def test_running_it_twice_is_a_no_op(tree):
    migrate_product_containers(tree, apply=True)
    migrated = tree / "omas/eddy" / str(SHOT) / "output/eddy.json.gz"
    first = migrated.read_bytes()
    stamp = migrated.stat().st_mtime_ns

    second = migrate_product_containers(tree, apply=True)

    assert second.pending == ()
    assert migrated.read_bytes() == first
    assert migrated.stat().st_mtime_ns == stamp, (
        "an already-projected eddy product was rewritten; the discriminator "
        "cannot tell a migrated product from a pending one"
    )


def test_a_half_migrated_tree_resumes_rather_than_starting_over(tree):
    migrate_product_containers(tree, stages=["diagnostics"], apply=True)
    done = tree / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json.gz"
    stamp = done.stat().st_mtime_ns

    report = audit_product_containers(tree)
    assert [item.stage for item in report.pending] == ["eddy"]
    assert report.migrated == (str(done),)

    migrate_product_containers(tree, apply=True)
    assert done.stat().st_mtime_ns == stamp


def test_an_orphan_temporary_is_named_and_then_cleared(tree):
    """What a process killed mid-write leaves, and how the next run sees it."""
    output = tree / "omas/eddy" / str(SHOT) / "output"
    stray = output / ".eddy.migrating.4242.json.gz"
    stray.write_bytes(b"half a product")

    report = audit_product_containers(tree)
    assert report.orphan_temporaries == (str(stray),)
    # The real product is still pending: a temporary is not a product.
    assert any(item.stage == "eddy" for item in report.pending)

    migrate_product_containers(tree, apply=True)
    assert not stray.exists()
    assert (output / "eddy.json.gz").is_file()


def test_an_original_newer_than_its_replacement_stops_the_whole_run(tree):
    """Something is still writing the old name; which one is authoritative is
    not a question a container rewriter can answer."""
    import os

    migrate_product_containers(tree, apply=True)
    stale = tree / "omas/eddy" / str(SHOT) / "output/eddy.json"
    future = stale.stat().st_mtime + 10_000
    os.utime(stale, (future, future))

    report = audit_product_containers(tree)
    assert report.conflicting == (str(stale),)
    assert not report.safe_to_apply

    # A second, non-conflicting product must not be touched by the refusal.
    other = _write_legacy(tree, "efit", 41524, _diagnostics_ods())
    with pytest.raises(ProductMigrationError, match="newer"):
        migrate_product_containers(tree, apply=True)
    assert other.is_file()
    assert not other.with_suffix(".json.gz").exists()


def test_the_manifest_records_the_migration_and_leaves_input_alone(tree):
    """`output` describes the file; `input` describes the run that produced it.

    Rewriting `input` would assert the physics was computed from a file that
    did not exist when it ran. The dangling join is closed from the other side,
    by `migration.previous_output.sha256`.
    """
    original = (tree / "omas/eddy" / str(SHOT) / "output/eddy.json").read_bytes()
    import hashlib

    previous = hashlib.sha256(original).hexdigest()

    migrate_product_containers(tree, apply=True)

    manifest = json.loads(
        (tree / "omas/eddy" / str(SHOT) / "metadata/manifest.json").read_text()
    )
    assert manifest["input"] == {"diagnostics_sha256": "deadbeef"}
    assert manifest["output"]["name"] == "eddy.json.gz"
    assert manifest["output"]["sha256"] != "stale"
    assert manifest["migration"]["previous_output"]["sha256"] == previous
    assert set(manifest["migration"]["projection"]["dropped_ids"]) == {
        "magnetics",
        "pf_active",
        "wall",
    }
    assert manifest["migration"]["verified"] == "semantic"


def test_the_sweep_refuses_a_plan_it_has_not_been_shown_clean(tree):
    unmigrated = audit_product_containers(tree)
    with pytest.raises(ProductMigrationError, match="not been migrated"):
        sweep_superseded_products(unmigrated, apply=False)

    migrate_product_containers(tree, apply=True)
    clean = audit_product_containers(tree, verify_shape=True)
    result = sweep_superseded_products(clean, apply=False)
    assert result["summary"]["removed"] == 2
    # A dry run deletes nothing.
    assert (tree / "omas/eddy" / str(SHOT) / "output/eddy.json").is_file()


def test_the_sweep_refuses_an_empty_report(tmp_path):
    """An empty plan and an unexamined tree look identical from here."""
    (tmp_path / "omas").mkdir()
    empty = audit_product_containers(tmp_path)
    with pytest.raises(ProductMigrationError, match="empty report"):
        sweep_superseded_products(empty, apply=True)


def test_the_sweep_keeps_an_eddy_original_whose_shot_has_no_diagnostics(tree):
    """That file is the only local copy of the shot's magnetics.

    Every other dropped IDS is recoverable from the shot's diagnostics product
    and the era's static product -- but that recoverability is a precondition
    of the deletion, not a property of it.
    """
    _write_legacy(tree, "eddy", 44444, _over_carrying_eddy_ods())
    migrate_product_containers(tree, apply=True)

    report = audit_product_containers(tree, verify_shape=True)
    result = sweep_superseded_products(report, apply=True)

    orphaned = tree / "omas/eddy/44444/output/eddy.json"
    assert orphaned.is_file(), "the only copy of shot 44444's magnetics was deleted"
    assert any("44444" in entry for entry in result["deletion_blocked"])
    # The shot that *is* recoverable was swept.
    assert not (tree / "omas/eddy" / str(SHOT) / "output/eddy.json").exists()


def test_verify_shape_finds_a_product_that_kept_what_it_does_not_own(tree):
    """The discriminator is the file name; --verify-shape makes it a fact.

    Without this, "a `.json.gz` eddy product has been projected" rests on an
    inventory of which writers exist, which is weaker than reading the file.
    """
    migrate_product_containers(tree, apply=True)
    # Put an unprojected product under the migrated name, as a writer on the
    # old code with the new constant would.
    vomas.save(
        _over_carrying_eddy_ods(),
        tree / "omas/eddy" / str(SHOT) / "output/eddy.json.gz",
    )

    quiet = audit_product_containers(tree, verify_shape=False)
    assert quiet.failures == ()

    checked = audit_product_containers(tree, verify_shape=True)
    assert any("does not own" in entry for entry in checked.failures)
    assert not checked.safe_to_apply
    with pytest.raises(ProductMigrationError):
        sweep_superseded_products(checked, apply=True)


def test_the_cli_dry_runs_by_default_and_reports_conflicts_in_its_exit_code(tree, capsys):
    assert filedb_main(["migrate-products", str(tree)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["summary"]["pending"] == 2
    assert not (tree / "omas/eddy" / str(SHOT) / "output/eddy.json.gz").exists()

    assert filedb_main(["migrate-products", str(tree), "--apply"]) == 0
    capsys.readouterr()

    import os

    stale = tree / "omas/eddy" / str(SHOT) / "output/eddy.json"
    future = stale.stat().st_mtime + 10_000
    os.utime(stale, (future, future))

    assert filedb_main(["migrate-products", str(tree)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["conflicting"] == 1


def test_an_unreadable_root_is_named_rather_than_silently_empty(tmp_path):
    with pytest.raises(ProductMigrationError, match="not a directory"):
        audit_product_containers(tmp_path / "nowhere")


def test_the_hdf5_stages_are_not_in_the_migration_at_all():
    """They declared `.h5` from the start (#599); there is no `.json` to move."""
    stages = {row.stage for row in MIGRATIONS}
    assert stages.isdisjoint(
        {"soft_x_rays", "camera_visible", "camera_visible_fluctuation"}
    )
    assert {"diagnostics", "eddy", "efit", "chease", "static"} <= stages
    # eddy is the only stage whose *contents* change, and what it keeps is read
    # from the registry rather than restated here.
    projections = {row.stage: row.keep_ids for row in MIGRATIONS if row.keep_ids}
    assert projections == {"eddy": STAGE_REPLICATION["eddy"].ids}


def test_an_unknown_stage_is_refused_rather_than_silently_matching_nothing(tree):
    with pytest.raises(ProductMigrationError, match="No product migration"):
        audit_product_containers(tree, stages=["not_a_stage"])


def test_verify_shape_catches_a_container_only_product_that_no_longer_decodes(tree):
    """The sweep's evidence must cover the stages holding most of the data.

    A container-only product has no shape to check, so the temptation is to
    check nothing -- and then `--verify-shape` says "clean" for every stage but
    eddy, which is 13 of 14 and includes the largest. What it can check is that
    the container still decodes, which is what separates "the original is
    redundant" from "the original is the only copy left".
    """
    migrate_product_containers(tree, apply=True)
    migrated = tree / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json.gz"
    original = tree / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json"

    # Truncate it the way a bad restore or a half-copy would.
    migrated.write_bytes(migrated.read_bytes()[: len(migrated.read_bytes()) // 2])

    quiet = audit_product_containers(tree, verify_shape=False)
    assert quiet.failures == ()

    checked = audit_product_containers(tree, verify_shape=True)
    assert any("could not be decompressed" in entry for entry in checked.failures)
    assert not checked.safe_to_apply

    with pytest.raises(ProductMigrationError):
        sweep_superseded_products(checked, apply=True)
    assert original.is_file(), "the only readable copy was deleted"


def test_the_audit_reports_what_the_sweep_will_refuse(tree):
    """A gate an operator can read before running the step that refuses.

    `deletion_blocked` is in the report's schema, so it has to be measured
    there: a field that is always empty reads as "nothing will be refused" on
    exactly the tree where something will be.
    """
    _write_legacy(tree, "eddy", 44444, _over_carrying_eddy_ods())
    migrate_product_containers(tree, apply=True)

    report = audit_product_containers(tree, verify_shape=True)
    assert any("44444" in entry for entry in report.deletion_blocked)
    assert report.to_dict()["summary"]["deletion_blocked"] == 1

    # And the shot that is recoverable is not reported as blocked.
    assert not any(f"/{SHOT}/" in entry for entry in report.deletion_blocked)


def test_a_projected_product_is_decoded_once(tree, monkeypatch):
    """Loading the source is what this migration's cost is made of.

    The eddy half is ~5300 products whose load dominates each one, so a second
    decode to recompute what the first already held is close to a second full
    run. Counted rather than trusted: the source is opened once per product.
    """
    import vaft.omas as vomas

    source = tree / "omas/eddy" / str(SHOT) / "output/eddy.json"
    loads: list[str] = []
    real_load = vomas.load

    def _counting_load(path, *args, **kwargs):
        loads.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(vomas, "load", _counting_load)

    migrate_product_containers(tree, stages=["eddy"], apply=True)

    assert loads.count(str(source)) == 1, loads
# --------------------------------------------------------------------------- #
# The join the migration leaves dangling
# --------------------------------------------------------------------------- #
def _real_hash_tree(root: Path) -> tuple[Path, Path, Path]:
    """A tree whose eddy manifest records the diagnostics hash a run would.

    The fixture above writes `"deadbeef"`, which never matches and so cannot
    show a migration *breaking* a match that held. These do.
    """
    from vaft.database.replication import sha256_file

    diagnostics = _write_legacy(root, "diagnostics", SHOT, _diagnostics_ods())
    eddy = _write_legacy(root, "eddy", SHOT, _over_carrying_eddy_ods())
    manifest_path = root / "omas" / "eddy" / str(SHOT) / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"] = {"diagnostics_sha256": sha256_file(diagnostics)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return diagnostics, eddy, manifest_path


def test_composition_survives_the_migration_it_invalidates(tmp_path):
    """Re-encoding diagnostics must not break every composition on the tree.

    `migrate-products` leaves the eddy manifest's `input` alone on purpose, so
    after the diagnostics product is re-containered its recorded hash names a
    file that no longer exists. `compose_stage_products` is called at
    `strict=True` by the pipeline itself -- `generate_constraints_ods` and
    `generate_stage_plots` -- so without the superseded-hash fallback the
    migration would stop EFIT on every shot at once.
    """
    from vaft.database.composition import compose_stage_products

    diagnostics, eddy, manifest_path = _real_hash_tree(tmp_path)
    compose_stage_products(
        diagnostics=diagnostics, eddy=eddy, eddy_manifest=manifest_path
    )  # holds before

    migrate_product_containers(tmp_path, apply=True)

    compose_stage_products(
        diagnostics=tmp_path / "omas/diagnostics" / str(SHOT) / "output/diagnostics.json.gz",
        eddy=tmp_path / "omas/eddy" / str(SHOT) / "output/eddy.json.gz",
        eddy_manifest=manifest_path,
    )


def test_the_fallback_accepts_only_the_file_that_was_superseded(tmp_path):
    """The relaxation is one hash, not the end of the check.

    A migrated tree still has to refuse an eddy product paired with diagnostics
    it was not computed from -- otherwise the fallback would have replaced a
    guard with nothing, which is worse than the failure it was added to prevent.
    """
    from vaft.database.composition import StageCompositionError, compose_stage_products

    _, _, manifest_path = _real_hash_tree(tmp_path)
    migrate_product_containers(tmp_path, apply=True)

    # Same grids, so this gets past the grid check and reaches the hash check --
    # only the bytes differ, which is exactly what the hash is there to notice.
    other = _diagnostics_ods()
    other["magnetics.ip.0.data"] = np.full_like(TIME, 2.0)
    foreign = tmp_path / "foreign.json.gz"
    vomas.save(other, foreign)

    with pytest.raises(StageCompositionError, match="different diagnostics file"):
        compose_stage_products(
            diagnostics=foreign,
            eddy=tmp_path / "omas/eddy" / str(SHOT) / "output/eddy.json.gz",
            eddy_manifest=manifest_path,
        )
