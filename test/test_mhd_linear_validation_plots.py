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
from vaft.database.production_qa import render_stage_plots, stage_plot_filenames
from vaft.validation.stage_evidence import STAGE_PRECONDITIONS


def _add_eigenfunction(ods, base, *, n_tor, harmonics=(-3.0, -2.0, -1.0), n_psi=16):
    """The `(psi, m)` subtree `vaft.machine_mapping.mhd_linear` maps for a DCON run.

    Only cells whose DCON run also produced `solutions.bin` carry one, so this
    is applied per cell rather than to every entry in the grid.
    """
    psi_n = np.linspace(0.05, 0.98, n_psi)
    m = np.asarray(harmonics, dtype=float)
    # A peak per harmonic at its own rational surface, so the drawn curves are
    # distinguishable and the dominant one is unambiguous.
    envelope = np.exp(-((psi_n[:, None] - np.linspace(0.3, 0.8, m.size)[None, :]) ** 2) / 0.01)
    amplitude = envelope * np.linspace(1.0, 3.0, m.size)[None, :]
    ods[f"{base}.plasma.grid_type.index"] = -1
    ods[f"{base}.plasma.grid_type.name"] = "inverse_psi_hamada_fourier"
    ods[f"{base}.plasma.grid.dim1"] = psi_n
    ods[f"{base}.plasma.grid.dim2"] = m
    ods[f"{base}.plasma.displacement_perpendicular.real"] = amplitude
    ods[f"{base}.plasma.displacement_perpendicular.imaginary"] = np.zeros_like(amplitude)
    q = 1.0 + 7.0 * psi_n**2
    singular_factor = m[None, :] - n_tor * q[:, None]
    ods[f"{base}.plasma.b_field_perturbed.coordinate1.real"] = np.zeros_like(amplitude)
    ods[f"{base}.plasma.b_field_perturbed.coordinate1.imaginary"] = singular_factor * amplitude
    ods[f"{base}.m_pol_dominant"] = float(m[-1])
    return ods


def _mhd_linear_ods(*, energies=None, times=(0.316, 0.317, 0.318), eigenfunction=True):
    """``toroidal_mode`` entries whose array position is not the mode number."""
    energies = energies or {1: [-0.4, -0.5, -0.6], 2: [0.2, 0.15, 0.1]}
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    ods["mhd_linear.ids_properties.homogeneous_time"] = 1
    ods["mhd_linear.time"] = list(times)
    ods["mhd_linear.code.parameters"] = '<parameters><eigenfunction_grid radial_stride="4"/></parameters>'
    for slice_index in range(len(times)):
        # Written in descending mode order, so a recipe keying off array
        # position instead of n_tor would silently mislabel every trace.
        for position, n_tor in enumerate(sorted(energies, reverse=True)):
            base = f"mhd_linear.time_slice.{slice_index}.toroidal_mode.{position}"
            ods[f"{base}.n_tor"] = n_tor
            ods[f"{base}.energy_perturbed"] = energies[n_tor][slice_index]
            if eigenfunction:
                _add_eigenfunction(ods, base, n_tor=n_tor)
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


# --- the eigenfunction figures -----------------------------------------------

def test_the_eigenfunction_defaults_to_the_least_stable_mapped_cell():
    """A shot holds one eigenfunction per (time, n_tor); the figure shows one.

    The default is the most negative `energy_perturbed` because that is the
    case a reader opens this figure to look at, and it is derived from the IDS
    rather than from array order.
    """
    ods = _mhd_linear_ods()

    figure, axes = vomas.plot_mhd_linear_profile_displacement(ods)

    # n=1 at the last slice is the most negative (-0.6) of the six cells.
    assert "n=1" in axes.get_title()
    assert "-0.6" in axes.get_title()


def test_an_explicit_cell_overrides_the_least_stable_default():
    ods = _mhd_linear_ods()

    figure, axes = vomas.plot_mhd_linear_profile_displacement(ods, time_slice=0, n_tor=2)

    assert "n=2" in axes.get_title()
    assert "0.2" in axes.get_title()


def test_harmonics_are_labelled_by_true_poloidal_mode_number():
    """`m=-3`, never `ipert=1`: the block index is not the physical harmonic."""
    ods = _mhd_linear_ods()

    figure, axes = vomas.plot_mhd_linear_profile_displacement(ods)

    assert [line.get_label() for line in axes.lines] == ["m=-3", "m=-2", "m=-1"]


def test_amplitudes_are_peak_normalized_and_the_axis_says_so():
    """DCON's eigenvector normalization is arbitrary, so an absolute axis lies.

    Normalizing keeps what the normalization cannot change -- the shape and the
    relative harmonic content -- and drops the number that would mean nothing.
    """
    ods = _mhd_linear_ods()

    figure, axes = vomas.plot_mhd_linear_profile_displacement(ods)

    peak = max(float(np.nanmax(line.get_ydata())) for line in axes.lines)
    assert peak == pytest.approx(1.0)
    assert "peak" in axes.get_ylabel()

    # Rescaling every harmonic by one global factor leaves the figure identical.
    rescaled = _mhd_linear_ods()
    for slice_index in range(3):
        for position in range(2):
            base = f"mhd_linear.time_slice.{slice_index}.toroidal_mode.{position}"
            rescaled[f"{base}.plasma.displacement_perpendicular.real"] = (
                rescaled[f"{base}.plasma.displacement_perpendicular.real"] * 1e6
            )
    _, rescaled_axes = vomas.plot_mhd_linear_profile_displacement(rescaled)
    for original, scaled in zip(axes.lines, rescaled_axes.lines):
        np.testing.assert_allclose(original.get_ydata(), scaled.get_ydata())


def test_the_drawn_harmonics_are_capped_and_the_title_says_how_many_were_dropped():
    ods = _mhd_linear_ods()

    figure, axes = vomas.plot_mhd_linear_profile_displacement(ods, max_harmonics=2)

    assert len(axes.lines) == 2
    assert "1 weaker harmonics omitted" in axes.get_title()


def test_the_title_reports_the_radial_stride_the_mapper_applied():
    """What reaches the IDS is a strided view, and the figure must not imply otherwise."""
    figure, axes = vomas.plot_mhd_linear_profile_displacement(_mhd_linear_ods())

    assert "every 4th radial sample" in axes.get_title()


def test_the_perturbed_field_carries_the_singular_factor_the_displacement_does_not():
    """The two panels differ by exactly |m - nq|, which is why both are drawn.

    `b = i(m - nq) xi` (match/ideal.f:372), so the field is the displacement
    reweighted towards each harmonic's resonant surface. Both figures are peak-
    normalized independently, so the ratio recovers that factor up to one
    constant -- and it being *constant* is the claim.
    """
    ods = _mhd_linear_ods()

    _, displacement_axes = vomas.plot_mhd_linear_profile_displacement(ods)
    _, field_axes = vomas.plot_mhd_linear_profile_b_field_perturbed(ods)

    displacement = {line.get_label(): line for line in displacement_axes.lines}["m=-2"]
    field = {line.get_label(): line for line in field_axes.lines}["m=-2"]

    psi_n = np.asarray(displacement.get_xdata())
    singular_factor = np.abs(-2.0 - 1 * (1.0 + 7.0 * psi_n**2))
    usable = np.asarray(displacement.get_ydata()) > 1e-6
    ratio = (
        np.asarray(field.get_ydata())[usable]
        / np.asarray(displacement.get_ydata())[usable]
        / singular_factor[usable]
    )

    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)


def test_an_ods_with_modes_but_no_eigenfunction_is_reported():
    ods = _mhd_linear_ods(eigenfunction=False)

    with pytest.raises(ValueError, match="companion `match`"):
        vomas.plot_mhd_linear_profile_displacement(ods)


def test_the_stage_skips_the_eigenfunction_figure_when_no_cell_carries_one(tmp_path):
    """No `match` is a normal installation, not a stage failure."""
    manifest = render_stage_plots(
        "mhd_linear",
        _mhd_linear_ods(eigenfunction=False),
        tmp_path / "plot",
        shot=41234,
        stage_manifest=_manifest(tmp_path),
    )

    rows = {row["name"]: row for row in manifest["plots"]}
    assert rows["mhd_linear_overview_eigenfunction"]["status"] == "skipped"
    assert rows["mhd_linear_time_energy_perturbed"]["status"] == "generated"
    assert not (tmp_path / "plot" / "stability_eigenfunction.png").exists()


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
    # Directory order is unspecified (hash order on ext4, sorted on APFS), so
    # compare the sets of names, not the order the filesystem hands them out.
    assert sorted(path.name for path in (tmp_path / "plot").iterdir()) == sorted(
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
    from vaft.database.production_qa import mhd_linear_run_coverage_model

    manifest = json.loads(_manifest_payload(failed_cell=True))
    model = mhd_linear_run_coverage_model(manifest)
    titles = {panel.title.split(" — ")[0] for panel in model.models}
    assert titles == {"dcon", "rdcon", "stride"}
    stride_panel = next(panel for panel in model.models if panel.title.startswith("stride"))
    assert stride_panel.series[0].label == "failed"


def test_run_coverage_model_raises_on_an_empty_manifest():
    from vaft.database.production_qa import mhd_linear_run_coverage_model

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
