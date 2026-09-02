"""EFIT submitted-constraint and reconstruction-residual validation (issue #139).

Stage 5 validates what was handed to EFIT; stage 6 validates what came back.
Both read the same product, because `collect_efit_outputs` merges the submitted
constraints into the equilibrium before overlaying the k-file and m-file parses.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

import vaft.omas as vomas
from vaft.omas._plot_recipes import (
    CONSTRAINT_STATES,
    _constraint_state,
    _constraint_table,
    _slice_times,
)
from vaft.database.production_qa import render_stage_plots, stage_plot_filenames
from vaft.validation.stage_evidence import STAGE_PRECONDITIONS

FAMILIES = ("bpol_probe", "flux_loop", "pf_current")


@pytest.fixture(scope="module")
def efit_ods():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    for slice_index in range(len(ods["equilibrium.time_slice"])):
        root = f"equilibrium.time_slice.{slice_index}"
        for family in FAMILIES:
            family_root = f"{root}.constraints.{family}"
            for index in range(len(ods.get(family_root, []))):
                base = f"{family_root}.{index}"
                measured = float(ods.get(f"{base}.measured", float("nan")))
                if not np.isfinite(measured):
                    measured = 1.0e-3 * (index + 1)
                ods[f"{base}.measured"] = measured
                ods[f"{base}.reconstructed"] = (
                    measured if family == "pf_current" else measured * 1.05
                )
                ods[f"{base}.weight"] = (
                    0.0 if family == "flux_loop" and index == 0 else 1.0
                )
                ods[f"{base}.exact"] = int(family == "pf_current")
        ip = float(ods[f"{root}.global_quantities.ip"])
        ods[f"{root}.constraints.ip.measured"] = ip
        ods[f"{root}.constraints.ip.reconstructed"] = ip
        ods[f"{root}.constraints.ip.weight"] = 1.0
        ods[f"{root}.convergence.grad_shafranov_deviation_value"] = 1.0e-3
        ods[f"{root}.convergence.iterations_n"] = 11
    return ods


def _constraint_ods(*, slices=2, channels=4, drop_from_slice=None):
    """An equilibrium carrying only what the constraint validation reads."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    times = np.linspace(0.30, 0.30 + 0.001 * (slices - 1), slices)
    ods["equilibrium.time"] = times
    for slice_index in range(slices):
        root = f"equilibrium.time_slice.{slice_index}"
        ods[f"{root}.time"] = float(times[slice_index])
        ods[f"{root}.convergence.grad_shafranov_deviation_value"] = 1e-3 * (slice_index + 1)
        ods[f"{root}.convergence.iterations_n"] = 10 + slice_index
        for family in FAMILIES:
            for index in range(channels):
                base = f"{root}.constraints.{family}.{index}"
                measured = 1e-2 * (index + 1)
                dropped = (
                    drop_from_slice is not None
                    and slice_index >= drop_from_slice
                    and index == 0
                )
                ods[f"{base}.measured"] = 0.0 if dropped else measured
                ods[f"{base}.reconstructed"] = 0.0 if dropped else measured * 1.05
                ods[f"{base}.weight"] = 0.0 if dropped else 1.0
                ods[f"{base}.source"] = f"{family}_ch{index}"
        ods[f"{root}.constraints.ip.measured"] = 7.0e4
        ods[f"{root}.constraints.ip.reconstructed"] = 7.1e4
        ods[f"{root}.constraints.ip.weight"] = 1.0
    return ods


# --- channel state classification -------------------------------------------

def test_channel_state_matches_what_the_constraint_builder_writes():
    # generate_constraints_ods zeroes both measured and weight for an absent
    # channel; kfile.py zeroes weight alone for one outside the fitted families.
    assert _constraint_state(0.0, 0.0) == "missing"
    assert _constraint_state(1.2e-2, 0.0) == "disabled"
    assert _constraint_state(1.2e-2, 1.0) == "enabled"
    assert _constraint_state(1.2e-2, float("nan")) == "enabled"


def test_constraint_table_keeps_every_channel_including_the_dead_ones():
    ods = _constraint_ods(drop_from_slice=0)
    table = _constraint_table(
        ods, time_slice=0, family="bpol_probe", is_array=True
    )
    assert len(table.state) == 4, "a missing channel must still occupy its index"
    assert table.state[0] == "missing"
    assert set(table.state[1:]) == {"enabled"}
    assert table.source[0] == "bpol_probe_ch0"
    assert table.count("missing") == 1
    assert table.mask("enabled").sum() == 3
    assert set(CONSTRAINT_STATES) == {"enabled", "disabled", "missing"}


def test_residual_is_measured_minus_reconstructed(efit_ods):
    table = _constraint_table(
        efit_ods, time_slice=0, family="bpol_probe", is_array=True
    )
    assert np.allclose(table.residual, table.measured - table.reconstructed)
    assert table.measured.size == len(efit_ods["equilibrium.time_slice.0.constraints.bpol_probe"])


# --- the figures -------------------------------------------------------------

def test_submitted_constraints_separate_enabled_disabled_and_missing():
    figure, axes = vomas.plot_equilibrium_overview_constraints(
        _constraint_ods(drop_from_slice=0)
    )
    titles = [ax.get_title() for ax in np.ravel(axes) if ax.get_visible()]
    assert any("3/4 fitted" in title for title in titles)
    legends = " ".join(
        text.get_text()
        for ax in np.ravel(axes)
        if ax.get_legend() is not None
        for text in ax.get_legend().get_texts()
    )
    assert "enabled (3)" in legends and "missing (1)" in legends


def test_coverage_shows_a_step_when_a_channel_is_dropped_between_slices():
    ods = _constraint_ods(slices=4, drop_from_slice=2)
    figure, axes = vomas.plot_equilibrium_overview_constraint_coverage(ods)
    panel = np.ravel(axes)[0]
    enabled = next(
        line for line in panel.lines if line.get_label().startswith("enabled")
    )
    counts = enabled.get_ydata()
    assert list(counts) == [4.0, 4.0, 3.0, 3.0], "the drop must be visible as a step"


def test_coverage_is_flat_when_the_channel_set_is_consistent():
    figure, axes = vomas.plot_equilibrium_overview_constraint_coverage(
        _constraint_ods(slices=4)
    )
    panel = np.ravel(axes)[0]
    enabled = next(
        line for line in panel.lines if line.get_label().startswith("enabled")
    )
    assert len(set(enabled.get_ydata())) == 1


def test_residual_figure_shows_convergence_beside_the_residuals(efit_ods):
    figure, axes = vomas.plot_equilibrium_overview_residuals(efit_ods)
    titles = [ax.get_title() for ax in np.ravel(axes) if ax.get_visible()]
    assert any("measured − reconstructed" in title for title in titles)
    assert any("Residual RMS by family" in title for title in titles)
    assert any("Grad-Shafranov deviation" in title for title in titles)
    assert any("Iterations" in title for title in titles)
    # PF currents are fitted exactly, so they are named rather than drawn on a
    # log axis that cannot show zero.
    assert any("fitted exactly" in title for title in titles)


def test_every_efit_figure_renders_for_the_complete_fixture(efit_ods, tmp_path):
    from vaft.plot import save_figure

    for name in (
        "equilibrium_overview_constraints",
        "equilibrium_overview_constraint_coverage",
        "equilibrium_overview_residuals",
    ):
        figure, _axes = getattr(vomas, f"plot_{name}")(efit_ods)
        target = tmp_path / f"{name}.png"
        save_figure(figure, target, dpi=100)
        assert target.stat().st_size > 10_000, name


def test_the_shared_table_refactor_leaves_the_verification_plot_unchanged(efit_ods):
    # _verification_constraint_panel was rebuilt on _constraint_table; it must
    # still drop channels without both a finite measured and reconstructed value.
    figure, axes = vomas.plot_equilibrium_overview_verification(efit_ods, time_slice=0)
    assert axes.shape == (2, 2)
    assert "relative RMS error" in axes[0, 0].get_title()
    assert len(axes[0, 0].lines) >= 2


# --- empty products ----------------------------------------------------------

def test_an_efit_ods_without_slices_is_an_empty_product_not_a_failure(tmp_path):
    # This is what generate_efit_ods writes when EFIT fails or is disabled.
    minimal = ODS(consistency_check=False)
    minimal["dataset_description.data_entry.pulse"] = 41234
    minimal["equilibrium.ids_properties.comment"] = "EFIT output unavailable: skipped"

    assert STAGE_PRECONDITIONS["efit"](minimal)
    manifest = render_stage_plots("efit", minimal, tmp_path / "plot")
    assert manifest["status"] == "empty"
    assert "no accepted equilibrium time slice" in manifest["reason"]
    assert {row["status"] for row in manifest["plots"]} == {"skipped"}
    assert not list((tmp_path / "plot").iterdir())
    # Metrics still run for an empty product: an empty stage is when its
    # diagnostics matter most.
    assert manifest["metrics"]["slice_count"] == 0


def test_a_required_plot_with_unexpectedly_absent_data_still_raises(tmp_path):
    # An ODS that passes the precondition but carries no constraints is a bug,
    # not a legitimately empty product, and must not be skipped quietly.
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = [0.30]
    ods["equilibrium.time_slice.0.time"] = 0.30
    assert STAGE_PRECONDITIONS["efit"](ods) is None
    with pytest.raises(ValueError, match="required validation plot"):
        render_stage_plots("efit", ods, tmp_path / "plot")


# --- the stage as the workflow runs it ---------------------------------------

def test_efit_stage_writes_its_figures_and_metrics(tmp_path, efit_ods):
    directory = tmp_path / "plot"
    manifest = render_stage_plots("efit", efit_ods, directory, shot=39915)

    assert set(stage_plot_filenames("efit", required_only=True)) <= {
        path.name for path in directory.iterdir()
    }
    metrics = manifest["metrics"]
    assert metrics["slice_count"] == len(efit_ods["equilibrium.time_slice"])

    first = metrics["slices"][0]["families"]["bpol_probe"]
    table = _constraint_table(
        efit_ods, time_slice=0, family="bpol_probe", is_array=True
    )
    fitted = table.mask("enabled") & np.isfinite(table.residual)
    assert first["residual_rms"] == pytest.approx(
        float(np.sqrt(np.mean(table.residual[fitted] ** 2)))
    )
    assert first["enabled"] + first["disabled"] + first["missing"] == len(table.state)
    # The flux-loop family of shot 39915 carries one disabled channel.
    assert metrics["slices"][0]["families"]["flux_loop"]["disabled"] == 1
    assert np.isfinite(metrics["slices"][0]["grad_shafranov_deviation"])


def test_slice_times_fall_back_to_the_slice_index_when_time_is_absent():
    ods = ODS(consistency_check=False)
    ods["equilibrium.time_slice.0.constraints.ip.measured"] = 1.0
    ods["equilibrium.time_slice.1.constraints.ip.measured"] = 2.0
    assert list(_slice_times(ods)) == [0.0, 1.0]


# --- review regressions -----------------------------------------------------

def test_error_bars_stay_on_their_own_channels_when_one_is_dropped():
    """A channel dropped for a non-finite value must drop its error bar too.

    `_state_series` filters non-finite values, so truncating the uncertainty
    array from the end shifted every later bar onto the wrong channel.
    """
    ods = _constraint_ods(channels=5)
    root = "equilibrium.time_slice.0.constraints.bpol_probe"
    for index in range(5):
        # Distinct, increasing uncertainties make a misalignment visible.
        ods[f"{root}.{index}.measured_error_upper"] = 1.0e-3 * (index + 1)
    # Channel 1 is enabled but carries no usable measurement.
    ods[f"{root}.1.measured"] = float("nan")

    figure, axes = vomas.plot_equilibrium_overview_constraints(ods)
    panel = np.ravel(axes)[0]
    container = next(
        (child for child in panel.containers if hasattr(child, "has_yerr")), None
    )
    assert container is not None, "the enabled trace must carry error bars"

    table = _constraint_table(ods, time_slice=0, family="bpol_probe", is_array=True)
    kept = np.flatnonzero(table.mask("enabled") & np.isfinite(table.measured))
    assert list(kept) == [0, 2, 3, 4], "channel 1 must be the dropped one"

    _line, _caps, bars = container
    segments = bars[0].get_segments()
    assert len(segments) == len(kept)
    # Each bar's half-height must be that channel's own uncertainty, scaled to
    # the panel's display units -- not its neighbour's.
    scale = 1e3
    for segment, index in zip(segments, kept):
        half = abs(segment[1][1] - segment[0][1]) / 2.0
        assert half == pytest.approx(table.uncertainty[index] * scale, rel=1e-6), index


def test_the_convergence_figure_survives_a_missing_m_file(efit_ods):
    """One absent optional artifact must not fail the whole EFIT stage.

    `convergence` is written by the m-file mapper and the verdict by the a-file.
    Requiring either would take the stage down over an input the figure can do
    without, when it can still draw iterations and self-consistency.
    """
    ods = efit_ods.copy()
    for index in range(len(ods["equilibrium.time_slice"])):
        del ods[f"equilibrium.time_slice.{index}.convergence"]

    assert "equilibrium_overview_convergence" in {
        row["name"] for row in vomas.available_plots(ods)
    }
    figure, axes = vomas.plot_equilibrium_overview_convergence(ods)
    titles = [ax.get_title() for ax in np.ravel(axes) if ax.get_visible()]
    assert any("EFIT outputs against each other" in title for title in titles)


def test_the_convergence_figure_still_fails_when_nothing_is_available():
    # The contract holds: a required plot that genuinely cannot be produced is
    # an actionable failure, not a silent gap.
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = [0.30]
    ods["equilibrium.time_slice.0.time"] = 0.30
    with pytest.raises(ValueError, match="no convergence information"):
        vomas.plot_equilibrium_overview_convergence(ods)
