"""Workflow coverage for EFIT artifact alignment and ODS dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
)
SCRIPT = WORKFLOW / "run_efit_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("run_efit_reconstruction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_artifact_manifest_aligns_nc_mfile_and_retains_afile_hash(tmp_path):
    for subdir, filename in (
        ("kfile", "k039915.00319"),
        ("gfile", "g039915.00319"),
        ("mfile", "m039915.00319.nc"),
        ("afile", "a039915.00319"),
    ):
        directory = tmp_path / subdir
        directory.mkdir()
        (directory / filename).write_bytes(filename.encode())

    payload = MODULE._artifact_payload(tmp_path, 39915, "completed")

    case = payload["cases"]["039915.00319"]
    assert case["disposition"] == "collected"
    assert set(case) == {"kfile", "gfile", "mfile", "afile", "disposition"}
    assert all(case[kind]["sha256"] for kind in ("kfile", "gfile", "mfile", "afile"))


def test_final_ods_has_explicit_inputs_and_chease_keeps_text_manifest():
    snakefile = (WORKFLOW / "Snakefile").read_text(encoding="utf-8")
    final_rule = snakefile.split("rule generate_efit_ods:", 1)[1].split(
        "rule run_chease:", 1
    )[0]
    for dependency in (
        'constraints=PATHS.shot_pattern("constraints_ods")',
        'kfiles=PATHS.shot_pattern("kfile_manifest")',
        'artifacts=PATHS.shot_pattern("efit_artifact_manifest")',
        'status=PATHS.shot_pattern("efit_status")',
    ):
        assert dependency in final_rule

    chease_rule = snakefile.split("rule run_chease:", 1)[1]
    assert 'gfiles=PATHS.shot_pattern("gfile_manifest")' in chease_rule

