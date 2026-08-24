from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

import vaft
from vaft.omas import (
    DifferenceKind,
    ParityClassification,
    compare_ods,
    load_reference_manifest,
    load_tolerance_policy,
    verify_reference_artifacts,
    write_comparison_reports,
)


REFERENCE_DIR = Path(__file__).parent / "data" / "vest_reference"
MANIFEST = REFERENCE_DIR / "manifest.yaml"
TOLERANCES = REFERENCE_DIR / "tolerances.yaml"
FIXTURE = REFERENCE_DIR / "shot-39915-compact.json.gz"


def test_exact_and_within_tolerance_results_are_distinct():
    reference = {
        "magnetics.time": np.array([0.1, 0.2]),
        "magnetics.ip.0.data": np.array([0.0, 10_000.0]),
    }
    candidate = {
        "magnetics.time": np.array([0.1, 0.2 + 5.0e-10]),
        "magnetics.ip.0.data": np.array([0.0, 10_000.0]),
    }

    result = compare_ods(reference, candidate)

    assert result.passed
    assert result.classification is ParityClassification.ACCEPTABLE
    by_path = {entry.path: entry for entry in result.entries}
    assert by_path["magnetics.ip.0.data"].classification is ParityClassification.EXACT
    assert by_path["magnetics.time"].classification is ParityClassification.ACCEPTABLE
    assert by_path["magnetics.time"].kind is DifferenceKind.TIME
    assert by_path["magnetics.time"].max_abs_error == pytest.approx(5.0e-10)


def test_structural_metadata_time_and_numerical_failures_are_separate():
    reference = {
        "dataset_description.data_entry.machine": "VEST",
        "magnetics.time": np.array([0.1, 0.2]),
        "magnetics.ip.0.data": np.array([1.0, 2.0]),
        "tf.r0": 0.4,
    }
    candidate = {
        "dataset_description.data_entry.machine": "OTHER",
        "magnetics.time": np.array([0.1, 0.2, 0.3]),
        "magnetics.ip.0.data": np.array([1.0, 3.0]),
        "wall.description_2d.0.type.name": "limiter",
    }

    result = compare_ods(reference, candidate)
    by_path = {entry.path: entry for entry in result.entries}

    assert not result.passed
    assert by_path["dataset_description.data_entry.machine"].kind is DifferenceKind.METADATA
    assert by_path["magnetics.time"].kind is DifferenceKind.TIME
    assert by_path["magnetics.ip.0.data"].kind is DifferenceKind.NUMERICAL
    assert by_path["tf.r0"].kind is DifferenceKind.MISSING_CANDIDATE
    assert by_path["wall.description_2d.0.type.name"].kind is DifferenceKind.MISSING_REFERENCE
    summary = result.summary()["kinds"]
    assert summary["metadata"] == 1
    assert summary["time"] == 1
    assert summary["structure.missing_candidate"] == 1
    assert summary["structure.missing_reference"] == 1
    assert summary["numerical"] == 1


def test_policy_classifies_declared_improvements_and_unavailable_reference():
    reference = {"dataset_description.data_entry.pulse": 39915}
    candidate = {
        "dataset_description.data_entry.pulse": 39915,
        "dataset_description.process.0.name": "workflow/main",
        "mhd_linear.time": np.array([0.319]),
    }

    result = compare_ods(reference, candidate, policy=TOLERANCES)
    by_path = {entry.path: entry for entry in result.entries}

    assert result.passed
    assert result.classification is ParityClassification.INTENTIONAL
    assert (
        by_path["dataset_description.process.0.name"].classification
        is ParityClassification.INTENTIONAL
    )
    assert (
        by_path["mhd_linear.time"].classification
        is ParityClassification.UNAVAILABLE
    )
    assert by_path["mhd_linear.time"].policy_note


def test_path_filters_limit_the_comparison_surface():
    reference = {"magnetics.time": [0.1], "tf.r0": 0.4}
    candidate = {"magnetics.time": [0.1], "tf.r0": 0.5}

    result = compare_ods(reference, candidate, paths=["magnetics.*"])

    assert result.passed
    assert [entry.path for entry in result.entries] == ["magnetics.time"]


def test_reference_scope_ignores_candidate_paths_outside_compact_fixture():
    reference = {"magnetics.time": [0.1]}
    candidate = {"magnetics.time": [0.1], "tf.r0": 0.4}

    result = compare_ods(reference, candidate, scope="reference")

    assert result.passed
    assert result.scope == "reference"
    assert [entry.path for entry in result.entries] == ["magnetics.time"]


def test_unmatched_path_filter_does_not_silently_pass():
    with pytest.raises(ValueError, match="No ODS paths"):
        compare_ods(
            {"magnetics.time": [0.1]},
            {"magnetics.time": [0.1]},
            paths=["equilibrium.*"],
        )


def test_json_and_markdown_reports_are_machine_and_human_readable(tmp_path):
    result = compare_ods(
        {"magnetics.ip.0.data": [1.0]},
        {"magnetics.ip.0.data": [2.0]},
        reference_label="legacy",
        candidate_label="main",
    )
    json_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "comparison.md"

    written = write_comparison_reports(
        result, json_path=json_path, markdown_path=markdown_path
    )

    assert written == (json_path, markdown_path)
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["passed"] is False
    assert payload["entries"][0]["kind"] == "numerical"
    markdown = markdown_path.read_text()
    assert "Status: **FAIL**" in markdown
    assert "magnetics.ip.0.data" in markdown


def test_integer_and_nonfinite_differences_produce_strict_json(tmp_path):
    result = compare_ods(
        {
            "magnetics.ip.0.data": np.array([1, 2]),
            "tf.b_field_tor_vacuum_r.data": np.array([np.nan]),
        },
        {
            "magnetics.ip.0.data": np.array([1, 3]),
            "tf.b_field_tor_vacuum_r.data": np.array([1.0]),
        },
    )
    report = tmp_path / "comparison.json"

    write_comparison_reports(result, json_path=report)
    payload = json.loads(report.read_text())
    by_path = {entry["path"]: entry for entry in payload["entries"]}

    assert by_path["magnetics.ip.0.data"]["max_abs_error"] == 1.0
    assert by_path["tf.b_field_tor_vacuum_r.data"]["max_abs_error"] is None


def test_scalar_numeric_differences_are_reported():
    result = compare_ods({"tf.r0": 0.4}, {"tf.r0": np.float64(0.41)})

    assert not result.passed
    entry = result.entries[0]
    assert entry.path == "tf.r0"
    assert entry.max_abs_error == pytest.approx(0.01)
    assert entry.max_rel_error == pytest.approx(0.025)


def test_zero_reference_error_is_json_serializable(tmp_path):
    result = compare_ods(
        {"equilibrium.time_slice.0.global_quantities.ip": 0.0},
        {"equilibrium.time_slice.0.global_quantities.ip": 1.0},
    )
    report = tmp_path / "zero-reference.json"

    write_comparison_reports(result, json_path=report)
    entry = json.loads(report.read_text())["entries"][0]

    assert entry["max_abs_error"] == 1.0
    assert entry["max_rel_error"] is None


def test_reference_manifest_and_repository_artifacts_verify_offline():
    manifest = load_reference_manifest(MANIFEST)

    assert manifest["reference_id"] == "vest-legacy-2026-08-19-v1"
    assert set(manifest["shots"]) == {
        "39915",
        "43016",
        "43017",
        "45966",
        "45967",
        "48224",
        "48226",
        "48233",
        "22027",
    }
    verification = verify_reference_artifacts(MANIFEST)
    assert {item.artifact_id for item in verification} == {
        "repository.config_snapshot",
        "repository.tolerances",
        "repository.shot_39915_compact",
    }
    assert all(item.valid for item in verification)


def test_shot_39915_compact_fixture_compares_without_server_access():
    reference = vaft.omas.load(FIXTURE)

    result = compare_ods(
        reference,
        reference,
        policy=TOLERANCES,
        reference_label="legacy-39915-compact",
        candidate_label="round-trip",
    )

    assert len(reference.flat()) == 123
    assert reference["dataset_description.data_entry.pulse"] == 39915
    np.testing.assert_allclose(
        reference["equilibrium.time"],
        np.array([0.316, 0.317, 0.318, 0.319, 0.320, 0.322, 0.325, 0.327]),
    )
    assert result.passed
    assert result.classification is ParityClassification.EXACT


def test_tolerance_policy_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        load_tolerance_policy(
            {
                "schema_version": 1,
                "defaults": {"numeric": {"atol": -1.0}},
            }
        )


def test_omas_json_gzip_save_is_deterministic_and_loadable(tmp_path):
    source = ODS(consistency_check=False)
    source["dataset_description.data_entry.pulse"] = 39915
    source["magnetics.time"] = np.array([0.1, 0.2])
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    vaft.omas.save(source, first)
    vaft.omas.save(source, second)
    restored = vaft.omas.load(first, imas_version="3.41.0")

    assert first.read_bytes() == second.read_bytes()
    assert compare_ods(source, restored).passed
