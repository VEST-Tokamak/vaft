"""Linear MHD stability validation plots (issue #139).

The plot set is deliberately minimal: `n_tor` and DCON's `energy_perturbed`
drive the energy figure, while the run-coverage figure is built from the stage
manifest rather than the ODS. RDCON/STRIDE's full per-surface Delta-prime has
no `mhd_linear` slot and survives in the manifest, so it is reported as a
metric rather than invented into a figure.

Note the `mhd_linear` IDS is laid out as a dense (time, n_tor) grid: every
requested mode has an entry in every time slice, so "this shot produced
nothing" is a statement about payloads, not about entry counts -- see the
padding tests at the end. There is no packaged mhd_linear product, so these
build their own ODSs the way test_machine_mapping_mhd_linear.py does.
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
    STAGE_PRECONDITIONS,
    render_stage_plots,
    stage_plot_filenames,
)


def _mhd_linear_ods(*, energies=None, times=(0.316, 0.317, 0.318)):
    """``toroidal_mode`` entries whose array position is not the mode number."""
    energies = energies or {1: [-0.4, -0.5, -0.6], 2: [0.2, 0.15, 0.1]}
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    ods["mhd_linear.ids_properties.homogeneous_time"] = 1
    ods["mhd_linear.time"] = list(times)
    for slice_index in range(len(times)):
        # Written in descending mode order, so a recipe keying off array
        # position instead of n_tor would silently mislabel every trace.
        for position, n_tor in enumerate(sorted(energies, reverse=True)):
            base = f"mhd_linear.time_slice.{slice_index}.toroidal_mode.{position}"
            ods[f"{base}.n_tor"] = n_tor
            ods[f"{base}.energy_perturbed"] = energies[n_tor][slice_index]
    return ods


def _manifest(tmp_path, *, failed_cell=True):
    payload = {
        "schema_version": 1,
        "stage": "mhd_linear",
        "shot": 41234,
        "modules_modes": {
            "t=316/dcon/n=1": {
                "status": "success",
                "modes": {"1": {"module": "dcon", "variable": "W_t_eigenvalue", "value": -0.4}},
            },
            "t=316/rdcon/n=1": {
                "status": "success",
                "modes": {"1": {"module": "rdcon", "variable": "Delta_prime", "value": 1.23}},
            },
            "t=316/stride/n=2": {"status": "failed", "reason": "solver timeout"}
            if failed_cell
            else {"status": "success", "modes": {}},
        },
    }
    path = tmp_path / "mhd_linear_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- the figure --------------------------------------------------------------

def test_traces_are_grouped_by_n_tor_not_by_array_position():
    figure, axes = vomas.plot_mhd_linear_time_energy_perturbed(_mhd_linear_ods())
    labels = [line.get_label() for line in axes.lines]
    assert labels == ["n=1", "n=2"]
    by_label = {line.get_label(): line.get_ydata() for line in axes.lines}
    assert list(by_label["n=1"]) == [-0.4, -0.5, -0.6]
    assert list(by_label["n=2"]) == [0.2, 0.15, 0.1]
    assert list(axes.lines[0].get_xdata()) == [0.316, 0.317, 0.318]


def test_a_mode_present_in_only_some_slices_keeps_its_own_time_base():
    ods = _mhd_linear_ods(energies={1: [-0.4, -0.5, -0.6]})
    # A second mode mapped for the last slice only.
    base = "mhd_linear.time_slice.2.toroidal_mode.1"
    ods[f"{base}.n_tor"] = 3
    ods[f"{base}.energy_perturbed"] = -0.9
    figure, axes = vomas.plot_mhd_linear_time_energy_perturbed(ods)
    by_label = {line.get_label(): line for line in axes.lines}
    assert list(by_label["n=3"].get_xdata()) == [0.318]
    assert list(by_label["n=1"].get_xdata()) == [0.316, 0.317, 0.318]


def test_an_ods_with_modes_but_no_energy_is_reported():
    ods = _mhd_linear_ods()
    for slice_index in range(3):
        for position in range(2):
            del ods[
                f"mhd_linear.time_slice.{slice_index}.toroidal_mode.{position}"
                ".energy_perturbed"
            ]
    with pytest.raises(ValueError, match="only DCON writes"):
        vomas.plot_mhd_linear_time_energy_perturbed(ods)


# --- empty products ----------------------------------------------------------

@pytest.mark.parametrize(
    "builder, reason",
    [
        (lambda: ODS(consistency_check=False), "no mhd_linear time slice"),
        (
            lambda: _strip_modes(_mhd_linear_ods()),
            "no toroidal mode",
        ),
    ],
)
def test_a_shot_the_gpec_suite_mapped_nothing_for_skips_cleanly(
    tmp_path, builder, reason
):
    ods = builder()
    assert reason in STAGE_PRECONDITIONS["mhd_linear"](ods)
    manifest = render_stage_plots("mhd_linear", ods, tmp_path / "plot")
    assert manifest["status"] == "empty"
    assert {row["status"] for row in manifest["plots"]} == {"skipped"}
    assert not list((tmp_path / "plot").iterdir())


def _strip_modes(ods):
    for slice_index in range(len(ods["mhd_linear.time_slice"])):
        del ods[f"mhd_linear.time_slice.{slice_index}.toroidal_mode"]
    return ods


# --- metrics -----------------------------------------------------------------

def test_delta_prime_and_solver_status_reach_the_metrics_from_the_manifest(tmp_path):
    manifest = render_stage_plots(
        "mhd_linear",
        _mhd_linear_ods(),
        tmp_path / "plot",
        shot=41234,
        stage_manifest=_manifest(tmp_path),
    )
    assert [path.name for path in (tmp_path / "plot").iterdir()] == list(
        stage_plot_filenames("mhd_linear")
    )
    metrics = manifest["metrics"]
    assert metrics["time_slice_count"] == 3
    assert sorted(metrics["modes"]) == ["1", "2"]

    runs = metrics["solver_runs"]
    # Delta-prime has no IDS slot, so the manifest is its only home.
    assert runs["t=316/rdcon/n=1"]["modes"]["1"]["variable"] == "Delta_prime"
    assert runs["t=316/rdcon/n=1"]["modes"]["1"]["value"] == 1.23
    # A failed solver cell is reported, not fatal.
    assert runs["t=316/stride/n=2"]["status"] == "failed"
    assert runs["t=316/stride/n=2"]["reason"] == "solver timeout"
    assert metrics["solver_status_counts"] == {"failed": 1, "success": 2}


def test_metrics_work_without_a_manifest(tmp_path):
    manifest = render_stage_plots(
        "mhd_linear", _mhd_linear_ods(), tmp_path / "plot", shot=41234
    )
    metrics = manifest["metrics"]
    assert "solver_runs" not in metrics
    assert metrics["modes"]["1"][0]["energy_perturbed"] == pytest.approx(-0.4)
    # Issue #173 phase 1's coverage plot needs the stage manifest; without one
    # it degrades to skipped rather than failing the whole stage.
    coverage = next(row for row in manifest["plots"] if row["name"] == "mhd_linear_run_coverage")
    assert coverage["status"] == "skipped"
    assert "stage_manifest" in coverage["reason"]


def test_the_figure_is_written_and_non_empty(tmp_path):
    render_stage_plots(
        "mhd_linear", _mhd_linear_ods(), tmp_path / "plot", shot=41234
    )
    target = tmp_path / "plot" / "stability_energy_perturbed.png"
    assert target.stat().st_size > 10_000


# --- issue #173 phase 1: run coverage -----------------------------------------

def test_run_coverage_plot_is_written_from_the_manifest_independent_of_the_ods(tmp_path):
    """The coverage plot is built entirely from `modules_modes`; an ODS with no
    mapped mode still gets it, since coverage is about which runs were
    *attempted*, not about what reached the IDS."""
    manifest = render_stage_plots(
        "mhd_linear",
        ODS(consistency_check=False),  # deliberately empty -- precondition would
        tmp_path / "plot",             # normally skip everything...
        shot=41234,
        stage_manifest=_manifest(tmp_path, failed_cell=True),
    )
    # ...and it does: the stage-level empty precondition applies uniformly to
    # every declared plot, coverage included, so this is still "skipped".
    assert manifest["status"] == "empty"
    coverage = next(row for row in manifest["plots"] if row["name"] == "mhd_linear_run_coverage")
    assert coverage["status"] == "skipped"


def test_run_coverage_plot_renders_when_the_ods_is_non_empty(tmp_path):
    manifest = render_stage_plots(
        "mhd_linear",
        _mhd_linear_ods(),
        tmp_path / "plot",
        shot=41234,
        stage_manifest=_manifest(tmp_path, failed_cell=True),
    )
    coverage = next(row for row in manifest["plots"] if row["name"] == "mhd_linear_run_coverage")
    assert coverage["status"] == "generated"
    target = tmp_path / "plot" / "stability_run_coverage.png"
    assert target.stat().st_size > 5_000


def test_run_coverage_model_groups_by_module_and_status():
    from vaft.validation import mhd_linear_run_coverage_model

    manifest = json.loads(_manifest_payload(failed_cell=True))
    model = mhd_linear_run_coverage_model(manifest)
    titles = {panel.title.split(" — ")[0] for panel in model.models}
    assert titles == {"dcon", "rdcon", "stride"}
    stride_panel = next(panel for panel in model.models if panel.title.startswith("stride"))
    assert stride_panel.series[0].label == "failed"


def test_run_coverage_model_raises_on_an_empty_manifest():
    from vaft.validation import mhd_linear_run_coverage_model

    with pytest.raises(ValueError, match="modules_modes"):
        mhd_linear_run_coverage_model({"modules_modes": {}})


def _manifest_payload(*, failed_cell: bool) -> str:
    payload = {
        "schema_version": 1,
        "stage": "mhd_linear",
        "shot": 41234,
        "modules_modes": {
            "t=316/dcon/n=1": {"status": "success", "modes": {}},
            "t=316/rdcon/n=1": {"status": "success", "modes": {}},
            "t=316/stride/n=2": {"status": "failed", "reason": "solver timeout"}
            if failed_cell
            else {"status": "success", "modes": {}},
        },
    }
    return json.dumps(payload)


def test_a_dense_grid_of_padding_only_is_reported_as_an_empty_product(tmp_path):
    """Under the dense (time, n_tor) layout a shot no solver produced anything
    for still has `toroidal_mode` entries, so emptiness is a question about
    payloads rather than entry counts -- the stage must still skip cleanly
    instead of trying to plot padding."""
    ods = ODS(consistency_check=False)
    ods["mhd_linear.ids_properties.homogeneous_time"] = 1
    ods["mhd_linear.time"] = [0.316, 0.317]
    for time_slice in range(2):
        for position, n_tor in enumerate((1, 2)):
            ods[f"mhd_linear.time_slice.{time_slice}.toroidal_mode.{position}.n_tor"] = n_tor

    reason = STAGE_PRECONDITIONS["mhd_linear"](ods)
    assert reason is not None and "padding only" in reason

    manifest = render_stage_plots("mhd_linear", ods, tmp_path / "plot", shot=41234)
    assert manifest["status"] == "empty"
    assert {row["status"] for row in manifest["plots"]} == {"skipped"}
    assert not list((tmp_path / "plot").iterdir())


def test_one_solved_cell_in_a_dense_grid_is_not_an_empty_product(tmp_path):
    ods = ODS(consistency_check=False)
    ods["mhd_linear.ids_properties.homogeneous_time"] = 1
    ods["mhd_linear.time"] = [0.316]
    for position, n_tor in enumerate((1, 2)):
        ods[f"mhd_linear.time_slice.0.toroidal_mode.{position}.n_tor"] = n_tor
    ods["mhd_linear.time_slice.0.toroidal_mode.0.energy_perturbed"] = -0.4

    assert STAGE_PRECONDITIONS["mhd_linear"](ods) is None
