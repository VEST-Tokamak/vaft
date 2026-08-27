from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaft.database.filedb import (
    ArtifactClass,
    FileDB,
    FileDBConfigError,
    FileDBPathError,
    GPECCode,
    audit_legacy_filedb,
)


def test_complete_canonical_directory_grammar(tmp_path):
    db = FileDB(tmp_path / "FileDB")
    root = tmp_path / "FileDB"

    assert db.raw(39915) == root / "raw/39915"
    assert (
        db.legacy("thomson-scattering", 22027)
        == root / "legacy/thomson-scattering/22027"
    )
    assert (
        db.omas("static", machine_version="vest-2019") == root / "omas/static/vest-2019"
    )
    for stage in ("diagnostics", "eddy", "efit", "chease"):
        assert db.omas(stage, shot=39915) == root / f"omas/{stage}/39915"
    assert db.efit(39915) == root / "efit/39915"
    assert db.chease(39915) == root / "chease/39915"
    assert db.pipeline("preflight", artifact="metadata") == root / "pipeline/preflight/metadata"
    for code in ("dcon", "rdcon", "stride", "ideal-gpec"):
        assert db.gpec(code, 39915, 1) == root / f"gpec/{code}/39915/n=1"

    assert not root.exists(), "path resolution must not materialize directories"


def test_raw_shot_directory_is_flat_and_rejects_artifact_subdirectories(tmp_path):
    db = FileDB(tmp_path / "FileDB")

    assert db.raw(39915) == tmp_path / "FileDB/raw/39915"
    with pytest.raises(FileDBPathError, match="artifact is not valid"):
        db.resolve("raw", shot=39915, artifact="output")


@pytest.mark.parametrize(
    "artifact",
    ["input", "output", "log", "plot", "config", "work", "metadata"],
)
def test_every_artifact_class_is_supported_without_materialization(tmp_path, artifact):
    db = FileDB(tmp_path / "FileDB")

    path = db.omas("efit", shot=39915, artifact=artifact)

    assert path == tmp_path / "FileDB/omas/efit/39915" / artifact
    assert not path.exists()


@pytest.mark.parametrize("code", list(GPECCode))
@pytest.mark.parametrize("mode", [1, 2, 6])
@pytest.mark.parametrize(
    "artifact",
    ["input", "output", "log", "plot", "config", "work", "metadata"],
)
def test_gpec_supports_every_code_mode_and_artifact_class_without_materialization(
    tmp_path, code, mode, artifact
):
    """DCON/RDCON/STRIDE need distinct modes per code (unlike legacy's shared
    scan list) and the full artifact-class set to store input namelists,
    solver output, logs, and run metadata for each (code, mode) cell."""
    db = FileDB(tmp_path / "FileDB")
    root = tmp_path / "FileDB"

    path = db.gpec(code, 39915, mode, artifact=artifact)

    assert path == root / f"gpec/{code.value}/39915/n={mode}/{artifact}"
    assert not path.exists()


def test_gpec_multiple_modes_of_the_same_code_do_not_collide(tmp_path):
    db = FileDB(tmp_path / "FileDB")

    paths = {db.gpec(GPECCode.RDCON, 39915, mode) for mode in (1, 2, 3, 4, 5, 6)}

    assert len(paths) == 6


def test_enum_arguments_and_same_shot_resolve_without_collisions(tmp_path):
    db = FileDB(tmp_path)
    paths = {
        db.raw(39915),
        db.legacy("thomson", 39915),
        db.omas("diagnostics", shot=39915),
        db.omas("eddy", shot=39915),
        db.omas("efit", shot=39915),
        db.omas("chease", shot=39915),
        db.efit(39915),
        db.chease(39915),
        db.gpec(GPECCode.DCON, 39915, 1, artifact=ArtifactClass.OUTPUT),
    }

    assert len(paths) == 9


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (("imas",), {"shot": 39915}, "Invalid domain"),
        (("main",), {"shot": 39915}, "Invalid domain"),
        (("raw",), {"shot": 0}, "positive integer"),
        (("raw",), {"shot": "39915.0"}, "positive integer"),
        (("legacy",), {"subdomain": "../bad", "shot": 39915}, "safe path component"),
        (("omas",), {"subdomain": "baseline", "shot": 39915}, "OMAS subdomain"),
        (
            ("omas",),
            {"subdomain": "static", "shot": 39915, "machine_version": "v1"},
            "shot is not valid",
        ),
        (
            ("omas",),
            {"subdomain": "static", "machine_version": None},
            "machine_version",
        ),
        (("gpec",), {"code": "gpec", "shot": 39915, "mode": 1}, "GPEC code"),
        (("gpec",), {"code": "dcon", "shot": 39915, "mode": 0}, "toroidal mode"),
        (("pipeline",), {"subdomain": "preflight", "shot": 39915}, "shot is not valid"),
        (("efit",), {"shot": 39915, "artifact": "result"}, "artifact class"),
    ],
)
def test_invalid_path_requests_fail_actionably(tmp_path, args, kwargs, message):
    with pytest.raises(FileDBPathError, match=message):
        FileDB(tmp_path).resolve(*args, **kwargs)


def test_runtime_root_precedence_and_expansion(tmp_path):
    explicit = FileDB.from_config(
        {"filedb": {"root": "${DEPLOY_ROOT}/FileDB"}},
        environment={
            "DEPLOY_ROOT": str(tmp_path / "explicit"),
            "VAFT_FILEDB_DIR": "/ignored",
        },
    )
    canonical = FileDB.from_config(
        environment={"VAFT_FILEDB_DIR": str(tmp_path / "environment")}
    )

    assert explicit.root == tmp_path / "explicit/FileDB"
    assert canonical.root == tmp_path / "environment"
    with pytest.raises(FileDBConfigError, match="VAFT_FILEDB_DIR"):
        FileDB.from_config(environment={})
    with pytest.raises(FileDBConfigError, match="DEPLOY_ROOT"):
        FileDB.from_config(
            {"filedb": {"root": "${DEPLOY_ROOT}/FileDB"}}, environment={}
        )


def test_legacy_resolution_is_explicit_and_read_only(tmp_path):
    artifact = tmp_path / "39915/omas/39915_efit.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("reference")
    db = FileDB(tmp_path)

    resolved = db.resolve_legacy_readonly(
        39915, "omas", "39915_efit.json", require_exists=True
    )

    assert resolved.path == artifact
    assert resolved.exists
    assert resolved.read_only
    assert resolved.layout == "legacy-shot-first"
    with pytest.raises(FileDBPathError, match="Invalid legacy area"):
        db.resolve_legacy_readonly(39915, "public")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        db.resolve_legacy_readonly(39915, "omas", "missing.json", require_exists=True)


def test_legacy_stability_resolution_accepts_only_positive_mode_directories(tmp_path):
    db = FileDB(tmp_path)

    resolved = db.resolve_legacy_readonly(
        39915, "linear_stability", "0.319", "dcon", "nn=1", "result.dat"
    )

    assert resolved.path == (
        tmp_path / "39915/linear_stability/0.319/dcon/nn=1/result.dat"
    )
    with pytest.raises(FileDBPathError, match="positive integer"):
        db.resolve_legacy_readonly(39915, "linear_stability", "dcon", "nn=0")


def _tree_snapshot(root: Path) -> list[tuple[str, str, bytes | None]]:
    result = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result.append((relative, "symlink", None))
        elif path.is_dir():
            result.append((relative, "directory", None))
        else:
            result.append((relative, "file", path.read_bytes()))
    return result


def test_migration_audit_detects_all_risks_without_writes(tmp_path):
    legacy_root = tmp_path / "public"
    target_root = tmp_path / "new-FileDB"
    shot = legacy_root / "39915"
    (shot / "diagnostics").mkdir(parents=True)
    (shot / "omas").mkdir()
    (shot / "efit/gfile").mkdir(parents=True)
    (shot / "efit/output").mkdir()
    (shot / "linear_stability/0.319/dcon/nn=1").mkdir(parents=True)
    (shot / "diagnostics/vest_39915_daq_raw.json.gz").write_bytes(b"same")
    (shot / "omas/39915_diagnostics.json").write_bytes(b"same")
    (shot / "omas/39915_combined.json").write_text("obsolete")
    (shot / "efit/gfile/g039915.00319").write_text("collision-a")
    (shot / "efit/output/g039915.00319").write_text("collision-b")
    (shot / "linear_stability/0.319/dcon/nn=1/result.dat").write_text("stable")
    (shot / "diagnostics/raw-link").symlink_to(
        shot / "diagnostics/vest_39915_daq_raw.json.gz"
    )
    before = _tree_snapshot(legacy_root)

    report = audit_legacy_filedb(legacy_root, target_root=target_root)

    assert _tree_snapshot(legacy_root) == before
    assert not target_root.exists()
    assert report.symlinks == ("39915/diagnostics/raw-link",)
    assert len(report.duplicates) == 1
    assert set(report.duplicates[0].sources) == {
        "39915/diagnostics/vest_39915_daq_raw.json.gz",
        "39915/omas/39915_diagnostics.json",
    }
    assert len(report.collisions) == 1
    assert set(report.collisions[0].sources) == {
        "39915/efit/gfile/g039915.00319",
        "39915/efit/output/g039915.00319",
    }
    assert {entry.source for entry in report.unmapped} == {
        "39915/omas/39915_combined.json"
    }
    assert {item.product for item in report.missing_products} == {
        "eddy_ods",
        "efit_ods",
        "chease_ods",
    }
    stability = next(
        entry for entry in report.entries if entry.source.endswith("result.dat")
    )
    assert stability.proposed_target == str(
        target_root / "gpec/dcon/39915/n=1/work/0.319/result.dat"
    )
    payload = report.to_dict()
    assert payload["dry_run"] is True
    assert (
        json.loads(json.dumps(payload, allow_nan=False))["summary"]["collisions"] == 1
    )


def test_migration_audit_detects_preexisting_target_collisions(tmp_path):
    legacy_root = tmp_path / "public"
    target_root = tmp_path / "new-FileDB"
    source_omas = legacy_root / "39915/omas"
    source_omas.mkdir(parents=True)
    (source_omas / "39915_efit.json").write_text("new efit")
    (source_omas / "39915_eddy.json").write_text("new eddy")

    existing_file = target_root / "omas/efit/39915/output/39915_efit.json"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("existing efit")
    broken_symlink = target_root / "omas/eddy/39915/output/39915_eddy.json"
    broken_symlink.parent.mkdir(parents=True)
    broken_symlink.symlink_to(target_root / "missing-eddy.json")
    before = _tree_snapshot(target_root)

    report = audit_legacy_filedb(legacy_root, target_root=target_root)

    assert _tree_snapshot(target_root) == before
    assert len(report.collisions) == 2
    collisions = {item.proposed_target: item for item in report.collisions}
    file_collision = collisions[str(existing_file)]
    assert file_collision.sources == ("39915/omas/39915_efit.json",)
    assert file_collision.existing_target is True
    assert file_collision.existing_target_kind == "file"
    symlink_collision = collisions[str(broken_symlink)]
    assert symlink_collision.sources == ("39915/omas/39915_eddy.json",)
    assert symlink_collision.existing_target is True
    assert symlink_collision.existing_target_kind == "symlink"


def test_workflow_main_uses_only_the_canonical_resolver():
    workflow = Path(__file__).parents[1] / "workflow/main/Snakefile"
    text = workflow.read_text()

    assert "from vaft.database.filedb import FileDB" in text
    assert "/srv/vest.filedb/public" not in text
    assert "imas/baseline" not in text
    assert "omas/baseline" not in text
