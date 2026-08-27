"""Canonical post-generation validation plots for the FileDB pipeline (issue #139)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from vaft.database.filedb import FileDB
from vaft.plot.registry import canonical_names
from vaft.validation import (
    STAGE_VALIDATION_PLOTS,
    ValidationPlot,
    raw_acquisition_qa_model,
    render_stage_plots,
    stage_plot_filenames,
    stages,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "workflow" / "automatic_pipeline_1_routine_data_processing"
sys.path.insert(0, str(WORKFLOW_DIR))

from paths import FILEDB, SHOT_FIRST, PipelinePaths  # noqa: E402

BASE_DIR = "/srv/vest.filedb/public"
SHOT = 41234
MACHINE_VERSION = "vest-45967-plus-pf2507"
REQUIRED_RAW_FIELDS = (1, 12, 25, 59, 109)


@pytest.fixture(scope="module")
def sample_ods():
    from vaft.omas.sample import sample_ods as load

    return load()


def _raw_payload():
    rng = np.random.default_rng(20260827)
    payload = {
        "shot": SHOT,
        "fields": {
            str(code): {
                "type": "fast" if code % 2 else "slow",
                "data": (rng.normal(size=64) * code).tolist(),
            }
            for code in (*REQUIRED_RAW_FIELDS, 7, 8)
        },
    }
    payload["fields"]["8"]["data"] = [0.0] * 64
    payload["fields"]["7"]["data"] = []
    payload["field_quality"] = {"7": "empty", "8": "all_zero"}
    return payload


# --- the declared contract -------------------------------------------------

def test_every_declared_ods_plot_is_a_registered_canonical_renderer():
    registered = set(canonical_names())
    for stage, entries in STAGE_VALIDATION_PLOTS.items():
        for entry in entries:
            if entry.kind != "ods":
                continue
            assert entry.plot in registered, (stage, entry.plot)


def test_filenames_are_unique_and_deterministic_within_a_stage():
    for stage in stages():
        names = stage_plot_filenames(stage)
        assert len(names) == len(set(names)), stage
        assert all(name.endswith(".png") for name in names), stage
        required = stage_plot_filenames(stage, required_only=True)
        assert required, f"{stage} declares no required validation plot"
        assert list(required) == [name for name in names if name in set(required)]


def test_the_declared_stage_set():
    assert set(stages()) == {"diagnostics", "efit", "raw", "static"}


def test_unknown_kind_and_empty_plot_name_are_refused():
    with pytest.raises(ValueError, match="unknown validation-plot kind"):
        ValidationPlot("magnetics_time_ip", kind="hdf5")
    with pytest.raises(ValueError, match="must name a renderer"):
        ValidationPlot("")


# --- canonical FileDB placement --------------------------------------------

def test_plot_paths_resolve_to_the_canonical_plot_artifact():
    paths = PipelinePaths(BASE_DIR, FILEDB)
    filedb = FileDB(BASE_DIR)

    assert paths.stage_plot(SHOT, "diagnostics", "magnetics_overview.png") == str(
        filedb.omas("diagnostics", shot=SHOT, artifact="plot") / "magnetics_overview.png"
    )
    assert paths.stage_plot_manifest(SHOT, "diagnostics") == str(
        filedb.omas("diagnostics", shot=SHOT, artifact="metadata") / "plot_manifest.json"
    )
    assert paths.static_plot(MACHINE_VERSION, "machine_geometry_poloidal.png") == str(
        filedb.omas("static", machine_version=MACHINE_VERSION, artifact="plot")
        / "machine_geometry_poloidal.png"
    )
    assert paths.raw_plot(SHOT, "raw_overview_acquisition.png") == str(
        filedb.raw(SHOT) / "plot" / "raw_overview_acquisition.png"
    )
    assert paths.code_plot_dir(SHOT, "chease") == str(filedb.chease(SHOT, artifact="plot"))
    assert paths.chease_plot_manifest(SHOT) == str(
        filedb.chease(SHOT, artifact="plot") / "plot_refined_gfiles_generated.txt"
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda paths: paths.stage_plot(SHOT, "diagnostics", "a.png"),
        lambda paths: paths.stage_plot_manifest(SHOT, "diagnostics"),
        lambda paths: paths.static_plot(MACHINE_VERSION, "a.png"),
        lambda paths: paths.raw_plot(SHOT, "a.png"),
        lambda paths: paths.code_plot(SHOT, "chease", "a.png"),
        lambda paths: paths.chease_plot_manifest(SHOT),
    ],
)
def test_plot_paths_have_no_legacy_shot_first_equivalent(call):
    paths = PipelinePaths(BASE_DIR, SHOT_FIRST)
    with pytest.raises(ValueError, match="canonical FileDB artifact"):
        call(paths)


def test_wildcard_patterns_keep_the_snakemake_wildcards():
    paths = PipelinePaths(BASE_DIR, FILEDB)
    assert paths.shot_pattern("stage_plot", "diagnostics", "magnetics_overview.png") == (
        f"{BASE_DIR}/omas/diagnostics/{{shot}}/plot/magnetics_overview.png"
    )
    assert paths.version_pattern("static_plot", "machine_geometry_poloidal.png") == (
        f"{BASE_DIR}/omas/static/{{machine_version}}/plot/machine_geometry_poloidal.png"
    )


# --- execution -------------------------------------------------------------

def test_stage_writes_exactly_the_declared_files(tmp_path, sample_ods):
    directory = tmp_path / "plot"
    manifest = render_stage_plots("diagnostics", sample_ods, directory)

    generated = {row["file"] for row in manifest["plots"] if row["status"] == "generated"}
    assert generated == {path.name for path in directory.iterdir()}
    assert generated <= set(stage_plot_filenames("diagnostics"))
    assert set(stage_plot_filenames("diagnostics", required_only=True)) <= generated
    for path in directory.iterdir():
        assert path.stat().st_size > 1000, path


def test_manifest_records_the_persisted_files_and_what_else_is_available(
    tmp_path, sample_ods
):
    manifest = render_stage_plots("diagnostics", sample_ods, tmp_path / "plot")
    assert manifest["stage"] == "diagnostics"
    for row in manifest["plots"]:
        if row["status"] != "generated":
            continue
        target = tmp_path / "plot" / row["file"]
        assert row["bytes"] == target.stat().st_size
        assert len(row["sha256"]) == 64
    assert "magnetics_time_ip" in manifest["available"]
    assert len(manifest["available"]) > len(manifest["plots"])


def test_an_absent_optional_ids_is_reported_and_does_not_fail_the_stage(
    tmp_path, sample_ods
):
    manifest = render_stage_plots("diagnostics", sample_ods, tmp_path / "plot")
    skipped = {row["name"]: row for row in manifest["plots"] if row["status"] == "skipped"}
    # Shot 39915 carries no interferometer.
    assert "interferometer_overview" in skipped
    assert "does not carry the data" in skipped["interferometer_overview"]["reason"]
    assert not (tmp_path / "plot" / "interferometer_overview.png").exists()


def test_a_required_plot_without_data_is_an_actionable_failure(tmp_path):
    from omas import ODS

    with pytest.raises(ValueError, match="required validation plot"):
        render_stage_plots("diagnostics", ODS(consistency_check=False), tmp_path / "plot")


def test_rendering_is_deterministic_for_the_same_input(tmp_path, sample_ods):
    first = render_stage_plots("diagnostics", sample_ods, tmp_path / "a")
    second = render_stage_plots("diagnostics", sample_ods, tmp_path / "b")
    digests = {
        row["name"]: row["sha256"] for row in first["plots"] if row["status"] == "generated"
    }
    for row in second["plots"]:
        if row["status"] == "generated":
            assert row["sha256"] == digests[row["name"]], row["name"]


def test_unknown_stage_names_the_declared_stages():
    with pytest.raises(KeyError, match="declared stages are"):
        render_stage_plots("no_such_stage", None, "/tmp")


# --- raw acquisition QA ----------------------------------------------------

def test_raw_qa_summarizes_coverage_quality_and_mandatory_signals():
    model = raw_acquisition_qa_model(_raw_payload(), required_fields=REQUIRED_RAW_FIELDS)
    titles = [getattr(panel, "title", "") for panel in model.models]
    assert any("Sample coverage" in title for title in titles)
    assert any("Zero / non-finite" in title for title in titles)
    # Representative mandatory signals, not one panel per raw field.
    signal_panels = [title for title in titles if title.startswith("Field ")]
    assert 0 < len(signal_panels) <= 4
    assert "7 fields, 2 flagged" in model.suptitle

    fractions = model.models[1]
    zero = dict(zip(fractions.series[0].x.tolist(), fractions.series[0].y.tolist()))
    non_finite = dict(zip(fractions.series[1].x.tolist(), fractions.series[1].y.tolist()))
    assert zero[8.0] == 1.0  # flatlined channel
    assert zero[7.0] == 1.0 and non_finite[7.0] == 1.0  # empty channel
    assert zero[1.0] == 0.0


def test_raw_stage_writes_one_compact_overview(tmp_path):
    directory = tmp_path / "plot"
    manifest = render_stage_plots(
        "raw",
        _raw_payload(),
        directory,
        shot=SHOT,
        required_fields=REQUIRED_RAW_FIELDS,
    )
    assert [path.name for path in directory.iterdir()] == ["raw_overview_acquisition.png"]
    assert manifest["plots"][0]["status"] == "generated"
    assert "available" not in manifest


def test_raw_payload_without_fields_is_reported():
    with pytest.raises(ValueError, match="no 'fields' mapping"):
        raw_acquisition_qa_model({"shot": SHOT})


# --- Snakemake tracking ----------------------------------------------------

def test_snakefile_declares_required_plots_as_real_outputs():
    source = (WORKFLOW_DIR / "Snakefile").read_text(encoding="utf-8")
    # Outputs are expanded from the registry, so a newly declared required plot
    # cannot become an untracked side effect of an existing rule.
    for rule, helper in (
        ("rule plot_raw:", "plots=raw_plot_outputs()"),
        ("rule plot_static:", "plots=static_plot_outputs()"),
        ("rule plot_diagnostics:", 'plots=stage_plot_outputs("diagnostics")'),
        ("rule plot_efit:", 'plots=stage_plot_outputs("efit")'),
    ):
        start = source.index(rule)
        block = source[start : source.index("log:", start)]
        assert helper in block, rule
    assert "chease_plot_manifest" in source


def test_stage_plot_outputs_cover_every_required_plot():
    paths = PipelinePaths(BASE_DIR, FILEDB)
    for stage in ("diagnostics", "efit"):
        required = stage_plot_filenames(stage, required_only=True)
        patterns = [paths.shot_pattern("stage_plot", stage, name) for name in required]
        assert len(patterns) == len(required)
        assert all("{shot}" in pattern and "/plot/" in pattern for pattern in patterns)


def test_generate_stage_plots_writes_the_manifest_the_metadata_references(
    tmp_path, sample_ods
):
    from omas import save_omas_json

    source = tmp_path / "diagnostics.json"
    save_omas_json(sample_ods, str(source))
    metadata = tmp_path / "metadata" / "plot_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(WORKFLOW_DIR / "generate_stage_plots.py"),
            "--stage", "diagnostics",
            "--shot", str(SHOT),
            "--input", str(source),
            "--output-dir", str(tmp_path / "plot"),
            "--metadata", str(metadata),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(metadata.read_text())
    persisted = {path.name for path in (tmp_path / "plot").iterdir()}
    assert {
        row["file"] for row in payload["plots"] if row["status"] == "generated"
    } == persisted
    assert payload["shot"] == SHOT
    # The manifest is tied to the exact product it validated.
    assert payload["input"]["name"] == source.name
    assert len(payload["input"]["sha256"]) == 64
