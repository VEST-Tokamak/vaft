"""Parity tests for pipeline 1's path helper.

`shot_first` must reproduce the literal paths pipeline 1 has always used, so
its output stays diffable against `/srv/vest.filedb/public`. `filedb` must
match `vaft.database.filedb.FileDB`'s canonical grammar exactly, with no path
reconstructed by hand.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vaft.database.filedb import ArtifactClass, FileDB, GPECCode


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
    / "paths.py"
)
SPEC = importlib.util.spec_from_file_location("pipeline1_paths", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

BASE_DIR = "/srv/vest.filedb/public"
SHOT = 48226
VERSION = "vest-43017-45957-pf1906"


def test_shot_first_reproduces_the_legacy_literal_paths():
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.SHOT_FIRST)

    assert paths.raw_dump(SHOT) == f"{BASE_DIR}/{SHOT}/diagnostics/vest_{SHOT}_daq_raw.json.gz"
    assert paths.diagnostics_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_diagnostics.json"
    assert paths.eddy_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_eddy.json"
    assert paths.constraints_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_constraints.json"
    assert paths.kfile_manifest(SHOT) == f"{BASE_DIR}/{SHOT}/efit/kfile/kfiles_generated.txt"
    assert paths.gfile_manifest(SHOT) == f"{BASE_DIR}/{SHOT}/efit/gfile/gfiles_generated.txt"
    assert paths.efit_status(SHOT) == f"{BASE_DIR}/{SHOT}/efit/efit_status.txt"
    assert paths.efit_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_efit.json"
    assert paths.chease_refined(SHOT) == f"{BASE_DIR}/{SHOT}/chease/refined_gfiles_generated.txt"
    assert paths.chease_status(SHOT) == f"{BASE_DIR}/{SHOT}/chease/chease_status.txt"
    assert paths.chease_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_chease.json"
    assert paths.gpec_workdir(SHOT) == f"{BASE_DIR}/{SHOT}/linear_stability"
    assert paths.mhd_linear_ods(SHOT) == f"{BASE_DIR}/{SHOT}/linear_stability/mhd_linear.json"
    assert paths.mhd_linear_manifest(SHOT) == f"{BASE_DIR}/{SHOT}/linear_stability/mhd_linear_manifest.json"
    assert paths.preflight_eligible() == f"{BASE_DIR}/preflight/eligible_shots.json"
    assert paths.preflight_excluded() == f"{BASE_DIR}/preflight/excluded_shots.json"

    # New in this phase: mirrors the legacy `static_file_dir`.
    assert paths.static_ods(VERSION) == f"{BASE_DIR}/static/{VERSION}/static.json"


def test_filedb_layout_matches_the_canonical_resolver():
    filedb = FileDB(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    assert paths.raw_dump(SHOT) == str(
        filedb.raw(SHOT, artifact="output") / f"vest_{SHOT}_daq_raw.json.gz"
    )
    assert paths.diagnostics_ods(SHOT) == str(
        filedb.omas("diagnostics", shot=SHOT, artifact="output") / "diagnostics.json"
    )
    assert paths.eddy_ods(SHOT) == str(
        filedb.omas("eddy", shot=SHOT, artifact="output") / "eddy.json"
    )
    # Issue #77: EFIT constraints live under omas/efit/{shot}/work.
    assert paths.constraints_ods(SHOT) == str(
        filedb.omas("efit", shot=SHOT, artifact="work") / "constraints.json"
    )
    assert paths.efit_ods(SHOT) == str(
        filedb.omas("efit", shot=SHOT, artifact="output") / "efit.json"
    )
    assert paths.chease_ods(SHOT) == str(
        filedb.omas("chease", shot=SHOT, artifact="output") / "chease.json"
    )
    assert paths.kfile_manifest(SHOT) == str(
        filedb.efit(SHOT, artifact="input") / "kfiles_generated.txt"
    )
    assert paths.gfile_manifest(SHOT) == str(
        filedb.efit(SHOT, artifact="output") / "gfiles_generated.txt"
    )
    assert paths.chease_refined(SHOT) == str(
        filedb.chease(SHOT, artifact="output") / "refined_gfiles_generated.txt"
    )
    assert paths.static_ods(VERSION) == str(
        filedb.omas("static", machine_version=VERSION, artifact="output") / "static.json"
    )
    assert paths.static_manifest(VERSION) == str(
        filedb.omas("static", machine_version=VERSION, artifact="metadata") / "manifest.json"
    )


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        MODULE.PipelinePaths(BASE_DIR, "not-a-real-layout")


@pytest.mark.parametrize("layout", [MODULE.SHOT_FIRST, MODULE.FILEDB])
def test_shot_pattern_produces_a_snakemake_wildcard(layout):
    paths = MODULE.PipelinePaths(BASE_DIR, layout)
    pattern = paths.shot_pattern("diagnostics_ods")
    assert "{shot}" in pattern
    assert str(MODULE._SHOT_SENTINEL) not in pattern


@pytest.mark.parametrize("layout", [MODULE.SHOT_FIRST, MODULE.FILEDB])
def test_version_pattern_produces_a_snakemake_wildcard(layout):
    paths = MODULE.PipelinePaths(BASE_DIR, layout)
    pattern = paths.version_pattern("static_ods")
    assert "{machine_version}" in pattern
    assert MODULE._VERSION_SENTINEL not in pattern


def test_filedb_gpec_workdirs_and_mhd_linear_products_are_canonical():
    filedb = FileDB(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)
    assert paths.gpec_workdir(SHOT, "dcon", 1) == str(filedb.gpec("dcon", SHOT, 1, artifact="work"))
    assert paths.mhd_linear_ods(SHOT) == str(filedb.omas("mhd_linear", shot=SHOT, artifact="output") / "mhd_linear.json")


def test_gpec_module_paths_are_shot_first_literals():
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.SHOT_FIRST)

    assert paths.gpec_module_status(SHOT, "dcon", 1) == f"{BASE_DIR}/{SHOT}/linear_stability/dcon/n=1/status.txt"
    assert paths.gpec_module_manifest(SHOT, "rdcon", 2) == f"{BASE_DIR}/{SHOT}/linear_stability/rdcon/n=2/run.json"


def test_gpec_module_paths_match_filedb_for_every_code_and_several_modes():
    filedb = FileDB(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    for code in GPECCode:
        for mode in (1, 2, 3):
            assert paths.gpec_module_status(SHOT, code.value, mode) == str(
                filedb.gpec(code, SHOT, mode, artifact="metadata") / "status.txt"
            )
            assert paths.gpec_module_manifest(SHOT, code.value, mode) == str(
                filedb.gpec(code, SHOT, mode, artifact="output") / "run.json"
            )


def test_gpec_module_path_translates_the_ideal_gpec_alias():
    """`vaft.code.gpec`'s module key "gpec" maps onto FileDB's `GPECCode.IDEAL_GPEC`
    ("ideal-gpec"), not a literal `GPECCode("gpec")`, which does not exist."""
    filedb = FileDB(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    assert paths.gpec_module_status(SHOT, "gpec", 1) == str(
        filedb.gpec(GPECCode.IDEAL_GPEC, SHOT, 1, artifact="metadata") / "status.txt"
    )


@pytest.mark.parametrize("layout", [MODULE.SHOT_FIRST, MODULE.FILEDB])
def test_gpec_module_pattern_produces_shot_code_and_mode_wildcards(layout):
    paths = MODULE.PipelinePaths(BASE_DIR, layout)

    status_pattern = paths.gpec_module_pattern("gpec_module_status")
    manifest_pattern = paths.gpec_module_pattern("gpec_module_manifest")

    for pattern in (status_pattern, manifest_pattern):
        assert "{shot}" in pattern
        assert "{code}" in pattern
        assert "{mode}" in pattern
        assert str(MODULE._SHOT_SENTINEL) not in pattern
        assert str(MODULE._MODE_SENTINEL) not in pattern
        assert MODULE._CODE_SENTINEL not in pattern
    assert status_pattern.endswith("status.txt")
    assert manifest_pattern.endswith("run.json")


def test_gpec_module_pattern_does_not_corrupt_an_unrelated_dcon_substring():
    """The `{code}` substitution used to be a blind `str.replace("dcon", ...)`,
    which would also rewrite any unrelated "dcon" substring elsewhere in the
    resolved path (e.g. inside the base dir itself). It must only ever swap
    the whole path segment produced for the `code` argument."""
    base_dir = "/srv/vest.filedb/mrdcon-archive"
    paths = MODULE.PipelinePaths(base_dir, MODULE.SHOT_FIRST)

    status_pattern = paths.gpec_module_pattern("gpec_module_status")

    assert status_pattern.startswith("/srv/vest.filedb/mrdcon-archive/")
    assert "{code}" in status_pattern
    assert status_pattern.count("{code}") == 1


def test_gpec_module_status_round_trips_every_artifact_class_without_materialization():
    filedb = FileDB(BASE_DIR)
    for artifact in ArtifactClass:
        # Just confirm FileDB itself accepts every artifact class for gpec --
        # paths.py only ever asks for "metadata"/"output", this locks in that
        # the underlying resolver supports the full set pipeline.py could use.
        assert filedb.gpec(GPECCode.DCON, SHOT, 1, artifact=artifact)
