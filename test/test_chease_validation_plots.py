"""CHEASE refinement validation (issue #172).

CHEASE refines an EFIT equilibrium at fixed boundary; the refined ODS carries
only the *refined* result (`GEQDSK.to_omas()` writes just `equilibrium`), so
the refinement-vs-input comparison `vaft.code.chease.comparison_metrics`
already computes is embedded onto `equilibrium.code.parameters` by
`generate_chease_ods.py`, and read back from there by both the stage metrics
and the `chease_overview_refinement_summary` figure. `chease_overview_profile_validity`
needs none of that -- it reads the refined profiles directly.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

import vaft.omas as vomas
from vaft.validation import (
    STAGE_METRICS,
    STAGE_PRECONDITIONS,
    render_stage_plots,
    stage_plot_filenames,
)

COMPARISON_METRICS = {
    "0": {
        "q_rms_rel": 0.01,
        "pressure_rms_rel": 0.02,
        "pprime_rms_rel": 0.03,
        "ffprim_rms_rel": 0.04,
        "psi_axis_abs_diff": 1e-4,
        "psi_boundary_abs_diff": 2e-4,
        "boundary_r_rms": 1e-3,
        "boundary_z_rms": 2e-3,
        "boundary_rz_rms": 3e-3,
        "boundary_points": 64.0,
        "current_abs_diff": 10.0,
        "current_rel_diff": 0.001,
    },
    "1": {
        "q_rms_rel": 0.05,
        "pressure_rms_rel": 0.06,
        "pprime_rms_rel": 0.07,
        "ffprim_rms_rel": 0.08,
        "psi_axis_abs_diff": 3e-4,
        "psi_boundary_abs_diff": 4e-4,
        "boundary_r_rms": 4e-3,
        "boundary_z_rms": 5e-3,
        "boundary_rz_rms": 6e-3,
        "boundary_points": 64.0,
        "current_abs_diff": 20.0,
        "current_rel_diff": 0.002,
    },
}

RECORDS_SUMMARY = [
    {"input": "g041234.00300", "status": "completed"},
    {"input": "g041234.00310", "status": "completed"},
]


def _chease_ods(*, monotonic_slice1=True, positive_pressure_slice1=True):
    """A refined-ODS fixture carrying exactly what the chease stage validates."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    ods["equilibrium.code.name"] = "chease"
    ods["equilibrium.code.library.0.name"] = "chease"
    ods["equilibrium.code.parameters"] = json.dumps(
        {"comparison_metrics": COMPARISON_METRICS, "records_summary": RECORDS_SUMMARY}
    )
    times = np.array([0.300, 0.310])
    ods["equilibrium.time"] = times
    for index, (q0, q95, monotonic, positive) in enumerate(
        (
            (1.05, 3.2, True, True),
            (1.10, 3.5, monotonic_slice1, positive_pressure_slice1),
        )
    ):
        root = f"equilibrium.time_slice.{index}"
        ods[f"{root}.time"] = float(times[index])
        q = np.linspace(q0, q95, 33)
        if not monotonic:
            q[16] = q[0] - 1.0  # a mid-profile dip breaks monotonicity
        pressure = np.linspace(2.0e4, 0.0, 33)
        if not positive:
            pressure[-1] = -50.0
        ods[f"{root}.profiles_1d.q"] = q
        ods[f"{root}.profiles_1d.pressure"] = pressure
        ods[f"{root}.global_quantities.q_axis"] = float(q0)
        ods[f"{root}.global_quantities.q_95"] = float(q95)
    return ods


def _empty_chease_ods():
    ods = ODS(consistency_check=False)
    ods["equilibrium.ids_properties.comment"] = "CHEASE output unavailable: skipped"
    ods["equilibrium.code.name"] = "chease"
    ods["equilibrium.code.library.0.name"] = "chease"
    ods["equilibrium.code.parameters"] = json.dumps(
        {
            "comparison_metrics": {},
            "records_summary": [{"input": "g041234.00300", "status": "missing_input"}],
        }
    )
    return ods


# --- metrics -----------------------------------------------------------------


def test_metrics_surface_the_embedded_comparison_and_physics_flags():
    metrics = STAGE_METRICS["chease"](_chease_ods())
    assert metrics["time_slice_count"] == 2
    assert metrics["slices"]["0"]["comparison"] == COMPARISON_METRICS["0"]
    assert metrics["slices"]["0"]["q0"] == pytest.approx(1.05)
    assert metrics["slices"]["0"]["q95"] == pytest.approx(3.2)
    assert metrics["slices"]["0"]["q_monotonic"] is True
    assert metrics["slices"]["0"]["pressure_positive"] is True
    assert metrics["records_summary"] == RECORDS_SUMMARY


def test_a_non_monotonic_q_profile_is_flagged():
    metrics = STAGE_METRICS["chease"](_chease_ods(monotonic_slice1=False))
    assert metrics["slices"]["0"]["q_monotonic"] is True
    assert metrics["slices"]["1"]["q_monotonic"] is False


def test_negative_pressure_is_flagged():
    metrics = STAGE_METRICS["chease"](_chease_ods(positive_pressure_slice1=False))
    assert metrics["slices"]["0"]["pressure_positive"] is True
    assert metrics["slices"]["1"]["pressure_positive"] is False


# --- precondition --------------------------------------------------------------


def test_a_real_chease_run_passes_the_precondition():
    assert STAGE_PRECONDITIONS["chease"](_chease_ods()) is None


def test_a_skipped_or_failed_run_is_an_empty_product():
    reason = STAGE_PRECONDITIONS["chease"](_empty_chease_ods())
    assert "no refined equilibrium time slice" in reason


# --- rendering -----------------------------------------------------------------


def test_stage_writes_both_required_summaries(tmp_path):
    manifest = render_stage_plots("chease", _chease_ods(), tmp_path / "plot")
    generated = {row["file"] for row in manifest["plots"] if row["status"] == "generated"}
    assert generated == set(stage_plot_filenames("chease", required_only=True))
    assert "metrics" in manifest
    assert manifest["metrics"]["time_slice_count"] == 2


def test_the_empty_product_skips_every_plot_but_keeps_the_metrics(tmp_path):
    manifest = render_stage_plots("chease", _empty_chease_ods(), tmp_path / "plot")
    assert manifest["status"] == "empty"
    assert {row["status"] for row in manifest["plots"]} == {"skipped"}
    assert not list((tmp_path / "plot").iterdir())
    assert manifest["metrics"]["records_summary"] == [
        {"input": "g041234.00300", "status": "missing_input"}
    ]


def test_adapters_render_from_the_refined_ods():
    ods = _chease_ods()
    figure, _ = vomas.plot_chease_overview_refinement_summary(ods)
    assert figure is not None
    figure, _ = vomas.plot_chease_overview_profile_validity(ods)
    assert figure is not None


def test_a_non_chease_equilibrium_ods_does_not_offer_the_chease_plots():
    """`equilibrium.code.name` alone is not exclusive to CHEASE.

    `vaft.data.vfit` also sets it (to "VFIT"), and the registry's
    availability check only tests path presence, not value -- so gating on
    `code.name` alone would falsely offer these plots for any equilibrium
    ODS that happens to set it. `code.library.0.name` is written only by
    `generate_chease_ods.py`.
    """
    vfit_ods = ODS(consistency_check=False)
    vfit_ods["equilibrium.code.name"] = "VFIT"
    vfit_ods["equilibrium.time"] = np.array([0.300])
    vfit_ods["equilibrium.time_slice.0.time"] = 0.300
    vfit_ods["equilibrium.time_slice.0.profiles_1d.q"] = np.linspace(1.0, 3.0, 33)

    offered = {row["name"] for row in vomas.available_plots(vfit_ods)}
    assert "chease_overview_refinement_summary" not in offered
    assert "chease_overview_profile_validity" not in offered


def test_refinement_summary_requires_embedded_comparison_metrics():
    ods = ODS(consistency_check=False)
    ods["equilibrium.code.name"] = "chease"
    ods["equilibrium.time"] = np.array([0.300])
    ods["equilibrium.time_slice.0.time"] = 0.300
    with pytest.raises(ValueError, match="comparison_metrics"):
        vomas.plot_chease_overview_refinement_summary(ods)
