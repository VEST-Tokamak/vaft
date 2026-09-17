"""Parity tests for pipeline 1's path helper.

`shot_first` must reproduce the literal paths pipeline 1 has always used, so
its output stays diffable against `/srv/vest.filedb/public`. `filedb` must
match `vaft.database.filedb.FileDB`'s canonical grammar exactly, with no path
reconstructed by hand.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path, PurePosixPath

import pytest

from vaft.database.filedb import ArtifactClass, FileDB, StabilityProduct


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
FAMILY = "magnetic"
REFINEMENT = "chease"
SHOT = 48226
VERSION = "vest-43017-45957-pf1906"


def _rule_filedb(root: str) -> FileDB:
    """Return FileDB expectations serialized in Snakemake's slash grammar."""
    filedb = FileDB(root)
    filedb.root = PurePosixPath(filedb.root.as_posix())
    return filedb


def test_shot_first_reproduces_the_legacy_literal_paths():
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.SHOT_FIRST)

    assert (
        paths.raw_dump(SHOT)
        == f"{BASE_DIR}/{SHOT}/diagnostics/vest_{SHOT}_daq_raw.json.gz"
    )
    assert (
        paths.diagnostics_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_diagnostics.json"
    )
    assert paths.eddy_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_eddy.json"
    assert (
        paths.constraints_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_constraints.json"
    )
    assert (
        paths.kfile_manifest(SHOT)
        == f"{BASE_DIR}/{SHOT}/efit/kfile/kfiles_generated.txt"
    )
    assert (
        paths.gfile_manifest(SHOT)
        == f"{BASE_DIR}/{SHOT}/efit/gfile/gfiles_generated.txt"
    )
    assert paths.efit_status(SHOT) == f"{BASE_DIR}/{SHOT}/efit/efit_status.txt"
    assert (
        paths.efit_artifact_manifest(SHOT)
        == f"{BASE_DIR}/{SHOT}/efit/artifact_manifest.json"
    )
    assert paths.efit_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_efit.json"
    assert (
        paths.chease_refined(SHOT)
        == f"{BASE_DIR}/{SHOT}/chease/refined_gfiles_generated.txt"
    )
    assert paths.chease_status(SHOT) == f"{BASE_DIR}/{SHOT}/chease/chease_status.txt"
    assert paths.chease_ods(SHOT) == f"{BASE_DIR}/{SHOT}/omas/{SHOT}_chease.json"
    assert paths.gpec_workdir(SHOT) == f"{BASE_DIR}/{SHOT}/linear_stability"
    assert (
        paths.mhd_linear_ods(SHOT)
        == f"{BASE_DIR}/{SHOT}/linear_stability/mhd_linear.json"
    )
    assert (
        paths.mhd_linear_manifest(SHOT)
        == f"{BASE_DIR}/{SHOT}/linear_stability/mhd_linear_manifest.json"
    )
    assert paths.preflight_eligible() == f"{BASE_DIR}/preflight/eligible_shots.json"
    assert paths.preflight_excluded() == f"{BASE_DIR}/preflight/excluded_shots.json"

    # New in this phase: mirrors the legacy `static_file_dir`.
    assert paths.static_ods(VERSION) == f"{BASE_DIR}/static/{VERSION}/static.json"


def test_filedb_layout_matches_the_canonical_resolver():
    filedb = _rule_filedb(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    assert paths.raw_dump(SHOT) == str(
        filedb.raw(SHOT) / f"vest_{SHOT}_daq_raw.json.gz"
    )
    assert paths.raw_manifest(SHOT) == str(
        filedb.raw(SHOT) / f"vest_{SHOT}_daq_manifest.json"
    )
    # Asked of the resolver rather than spelled out: the container is declared
    # in one place (`OMAS_PRODUCT_SUFFIX` / `OMAS_PRODUCT_SUFFIXES`), and a
    # literal here would have to be edited every time it moves -- which is how
    # a fourth spelling of a product name gets into circulation.
    assert paths.diagnostics_ods(SHOT) == str(filedb.omas_product("diagnostics", shot=SHOT))
    assert paths.eddy_ods(SHOT) == str(filedb.omas_product("eddy", shot=SHOT))
    # Issue #77: EFIT constraints live under omas/efit/{shot}/work.
    assert paths.constraints_ods(SHOT) == str(
        filedb.omas("efit", shot=SHOT, family=FAMILY, artifact="work") / "constraints.json"
    )
    assert paths.efit_ods(SHOT) == str(
        filedb.omas_product("efit", shot=SHOT, family=FAMILY)
    )
    # Issue #139: validation plots are a canonical `plot/` artifact, resolved
    # beside the stage output they validate.
    assert paths.stage_plot(SHOT, "efit", "equilibrium_overview_verification.png") == str(
        filedb.omas("efit", shot=SHOT, family=FAMILY, artifact="plot")
        / "equilibrium_overview_verification.png"
    )
    assert paths.chease_ods(SHOT) == str(
        filedb.omas_product("chease", shot=SHOT, family=FAMILY)
    )
    assert paths.kfile_manifest(SHOT) == str(
        filedb.efit(SHOT, family=FAMILY, artifact="input") / "kfiles_generated.txt"
    )
    assert paths.gfile_manifest(SHOT) == str(
        filedb.efit(SHOT, family=FAMILY, artifact="output") / "gfiles_generated.txt"
    )
    assert paths.efit_artifact_manifest(SHOT) == str(
        filedb.efit(SHOT, family=FAMILY, artifact="metadata") / "artifact_manifest.json"
    )
    assert paths.chease_refined(SHOT) == str(
        filedb.chease(SHOT, family=FAMILY, artifact="output") / "refined_gfiles_generated.txt"
    )
    assert paths.static_ods(VERSION) == str(
        filedb.omas_product("static", machine_version=VERSION)
    )
    assert paths.static_manifest(VERSION) == str(
        filedb.omas("static", machine_version=VERSION, artifact="metadata")
        / "manifest.json"
    )
    # One mhd_linear per stability product, not one per shot: the toroidal_mode
    # AOS is a dense (time, n_tor) grid, so two products sharing a stage product
    # would overwrite each other at the same (time_slice, position).
    assert paths.mhd_linear_ods(SHOT, "dcon") == str(
        filedb.omas_product(
            "mhd_linear", shot=SHOT, family=FAMILY, refinement=REFINEMENT,
            product="dcon-peeling",
        )
    )
    assert paths.mhd_linear_manifest(SHOT, "rdcon") == str(
        filedb.omas(
            "mhd_linear", shot=SHOT, family=FAMILY, refinement=REFINEMENT,
            product="rdcon", artifact="metadata",
        ) / "manifest.json"
    )
    assert paths.mhd_linear_ods(SHOT, "dcon") != paths.mhd_linear_ods(SHOT, "rdcon")
    assert paths.preflight_eligible() == str(
        filedb.pipeline("preflight", artifact="metadata") / "eligible_shots.json"
    )
    assert paths.preflight_excluded() == str(
        filedb.pipeline("preflight", artifact="metadata") / "excluded_shots.json"
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


def test_filedb_gpec_workdir_is_one_canonical_cell_per_code_and_mode():
    filedb = _rule_filedb(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    assert paths.gpec_workdir(SHOT, "dcon", 1) == str(
        filedb.gpec("dcon-peeling", SHOT, 1, family=FAMILY, refinement=REFINEMENT, artifact="work")
    )
    assert paths.gpec_workdir(SHOT, "gpec", 2) == str(
        filedb.gpec(StabilityProduct.IDEAL_GPEC, SHOT, 2, family=FAMILY, refinement=REFINEMENT, artifact="work")
    )


def test_filedb_paths_never_leak_into_the_historical_shot_first_tree():
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)
    products = [
        paths.raw_dump(SHOT),
        paths.diagnostics_ods(SHOT),
        paths.efit_ods(SHOT),
        paths.chease_ods(SHOT),
        paths.gpec_workdir(SHOT, "dcon", 1),
        paths.mhd_linear_ods(SHOT, "dcon"),
        # A stability log is the record of one product's run, so it lands
        # beside that product rather than in a shot-wide directory.
        paths.log(SHOT, "run_gpec_suite", "dcon"),
        paths.log(SHOT, "build_mhd_linear", "rdcon"),
        paths.preflight_eligible(),
    ]
    banned = (
        f"/{SHOT}/omas/", f"/{SHOT}/efit/", f"/{SHOT}/chease/",
        f"/{SHOT}/linear_stability/", f"/{SHOT}/logs/",
    )
    assert all(not any(token in path for token in banned) for path in products)


def test_gpec_module_paths_are_shot_first_literals():
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.SHOT_FIRST)

    assert (
        paths.gpec_module_status(SHOT, "dcon", 1)
        == f"{BASE_DIR}/{SHOT}/linear_stability/dcon/n=1/status.txt"
    )
    assert (
        paths.gpec_module_manifest(SHOT, "rdcon", 2)
        == f"{BASE_DIR}/{SHOT}/linear_stability/rdcon/n=2/run.json"
    )


def test_gpec_module_paths_match_filedb_for_every_code_and_several_modes():
    filedb = _rule_filedb(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    # `DCON_LEGACY` ("dcon") is excluded: it is a product only a migrated
    # pre-canonical tree carries, and as an *input* here "dcon" is the solver
    # module, which `stability_product` maps to the product its shipped edge
    # configuration produces. The pipeline never files anything under it.
    for product in StabilityProduct:
        if product is StabilityProduct.DCON_LEGACY:
            continue
        for mode in (1, 2, 3):
            assert paths.gpec_module_status(SHOT, product.value, mode) == str(
                filedb.gpec(
                    product, SHOT, mode,
                    family=FAMILY, refinement=REFINEMENT, artifact="metadata",
                ) / "status.txt"
            )
            assert paths.gpec_module_manifest(SHOT, product.value, mode) == str(
                filedb.gpec(
                    product, SHOT, mode,
                    family=FAMILY, refinement=REFINEMENT, artifact="output",
                ) / "run.json"
            )


def test_the_shot_first_tree_keeps_the_module_spelling_for_a_product():
    """The legacy layout has no place to record an edge treatment.

    The Snakemake wildcard carries the *product*, because it has to tell
    `dcon-peeling` from `dcon-kink`. Interpolating that straight into the
    shot-first tree would write `linear_stability/dcon-peeling/`, where the
    reference output this layout exists to stay diffable against has
    `linear_stability/dcon/` -- and would re-solve every DCON cell of an
    existing tree, because the old directory no longer matches.
    """
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.SHOT_FIRST)

    legacy = f"{BASE_DIR}/{SHOT}/linear_stability/dcon/n=1/status.txt"
    assert paths.gpec_module_status(SHOT, "dcon", 1) == legacy
    assert paths.gpec_module_status(SHOT, "dcon-peeling", 1) == legacy
    # Both DCON products collapse onto one legacy directory. That is the
    # layout's limitation, not a defect -- a run needing both branches needs
    # `layout: filedb`, where they are separate identities.
    assert paths.gpec_module_status(SHOT, "dcon-kink", 1) == legacy
    assert paths.gpec_module_manifest(SHOT, "ideal-gpec", 2) == (
        f"{BASE_DIR}/{SHOT}/linear_stability/gpec/n=2/run.json"
    )


def test_an_explicit_edge_treatment_decides_the_product():
    """So a finished cell can be checked against the product it was filed under.

    `stability_product(cell_product, edge_treatment=output.edge_treatment)` has
    to be able to disagree with `cell_product`; a version that returned the
    input unchanged would make that check pass for every cell, including one
    whose artifact says it truncated at the dW peak.
    """
    assert MODULE.stability_product("dcon") == "dcon-peeling"
    assert MODULE.stability_product("dcon-kink") == "dcon-kink"

    assert (
        MODULE.stability_product("dcon-peeling", edge_treatment="peak_dw_truncated")
        == "dcon-kink"
    )
    assert (
        MODULE.stability_product("dcon-kink", edge_treatment="full_edge")
        == "dcon-peeling"
    )
    with pytest.raises(ValueError, match="edge treatment"):
        MODULE.stability_product("dcon", edge_treatment="sideways")


def test_the_module_and_product_translations_are_inverse():
    """A product added on one side and not the other routes a cell to the wrong
    solver, so the two maps are held to each other."""
    for module in ("rdcon", "stride", "gpec"):
        assert MODULE.solver_module(MODULE.stability_product(module)) == module
    for product in ("dcon-peeling", "dcon-kink"):
        assert MODULE.solver_module(product) == "dcon"
    # Idempotent in both directions: a call site takes whichever it is handed.
    assert MODULE.solver_module("rdcon") == "rdcon"
    assert MODULE.stability_product("ideal-gpec") == "ideal-gpec"


def test_gpec_module_path_translates_the_ideal_gpec_alias():
    """A solver module is translated to the product its output is filed as.

    `vaft.code.gpec`'s module key "gpec" maps onto `StabilityProduct.IDEAL_GPEC`
    ("ideal-gpec"), not a literal `StabilityProduct("gpec")`, which does not
    exist; and "dcon" maps onto the product its shipped edge configuration
    actually produces.
    """
    filedb = _rule_filedb(BASE_DIR)
    paths = MODULE.PipelinePaths(BASE_DIR, MODULE.FILEDB)

    assert paths.gpec_module_status(SHOT, "dcon", 1) == str(
        filedb.gpec(
            StabilityProduct.DCON_PEELING, SHOT, 1,
            family=FAMILY, refinement=REFINEMENT, artifact="metadata",
        ) / "status.txt"
    )
    assert paths.gpec_module_status(SHOT, "gpec", 1) == str(
        filedb.gpec(StabilityProduct.IDEAL_GPEC, SHOT, 1, family=FAMILY, refinement=REFINEMENT, artifact="metadata") / "status.txt"
    )


@pytest.mark.parametrize("layout", [MODULE.SHOT_FIRST, MODULE.FILEDB])
def test_gpec_module_pattern_produces_shot_product_and_mode_wildcards(layout):
    paths = MODULE.PipelinePaths(BASE_DIR, layout)

    status_pattern = paths.gpec_module_pattern("gpec_module_status")
    manifest_pattern = paths.gpec_module_pattern("gpec_module_manifest")

    for pattern in (status_pattern, manifest_pattern):
        assert "{shot}" in pattern
        assert "{product}" in pattern
        assert "{mode}" in pattern
        assert str(MODULE._SHOT_SENTINEL) not in pattern
        assert str(MODULE._MODE_SENTINEL) not in pattern
        assert MODULE._CODE_SENTINEL not in pattern
    assert status_pattern.endswith("status.txt")
    assert manifest_pattern.endswith("run.json")


def test_gpec_module_pattern_does_not_corrupt_an_unrelated_dcon_substring():
    """The `{product}` substitution used to be a blind `str.replace("dcon", ...)`,
    which would also rewrite any unrelated "dcon" substring elsewhere in the
    resolved path (e.g. inside the base dir itself). It must only ever swap
    the whole path segment produced for the `code` argument."""
    base_dir = "/srv/vest.filedb/mrdcon-archive"
    paths = MODULE.PipelinePaths(base_dir, MODULE.SHOT_FIRST)

    status_pattern = paths.gpec_module_pattern("gpec_module_status")

    assert status_pattern.startswith("/srv/vest.filedb/mrdcon-archive/")
    assert "{product}" in status_pattern
    assert status_pattern.count("{product}") == 1


def test_the_product_wildcard_survives_a_layout_that_rewrites_it():
    """The segment is located by diffing two resolutions, not by matching back.

    `shot_first` files a product under its solver module, so the sentinel that
    went in is not the segment that comes out. Matching the sentinel back
    produced a pattern with no `{product}` in it at all, which Snakemake would
    then treat as a single concrete path -- one rule for every shot and mode.
    """
    for layout in (MODULE.SHOT_FIRST, MODULE.FILEDB):
        paths = MODULE.PipelinePaths(BASE_DIR, layout)
        for product in ("gpec_module_status", "gpec_module_manifest"):
            pattern = paths.gpec_module_pattern(product)
            assert pattern.count("{product}") == 1, (layout, product, pattern)
            assert "{shot}" in pattern and "{mode}" in pattern
            assert "dcon" not in pattern, (layout, product, pattern)


def test_gpec_module_status_round_trips_every_artifact_class_without_materialization():
    filedb = _rule_filedb(BASE_DIR)
    for artifact in ArtifactClass:
        # Just confirm FileDB itself accepts every artifact class for gpec --
        # paths.py only ever asks for "metadata"/"output", this locks in that
        # the underlying resolver supports the full set pipeline.py could use.
        assert filedb.gpec(
                StabilityProduct.DCON_PEELING, SHOT, 1,
                family=FAMILY, refinement=REFINEMENT, artifact=artifact,
            )


@pytest.mark.skipif(
    os.name != "nt",
    reason=r"only Windows reads C:\... as a path; on POSIX a backslash is an "
    "ordinary filename character and is deliberately left alone",
)
@pytest.mark.parametrize("layout", [MODULE.SHOT_FIRST, MODULE.FILEDB])
def test_windows_base_is_serialized_for_snakemake(layout):
    paths = MODULE.PipelinePaths(r"C:\vaft data\filedb", layout)

    assert paths.raw_dump(SHOT).startswith("C:/vaft data/filedb/")
    assert "\\" not in paths.shot_pattern("diagnostics_ods")


def test_a_posix_root_keeps_a_backslash_that_is_part_of_a_name():
    """The separator rewrite must not touch a legal POSIX filename.

    A backslash is an ordinary character on Linux, which is where this
    pipeline actually runs, so folding it into a separator there would
    silently retarget the whole tree.
    """
    paths = MODULE.PipelinePaths("/srv/vest.filedb/archive\\2026", MODULE.SHOT_FIRST)
    expected = (
        "/srv/vest.filedb/archive/2026"
        if os.name == "nt"
        else "/srv/vest.filedb/archive\\2026"
    )

    assert paths.raw_dump(SHOT).startswith(expected)
