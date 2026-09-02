"""Synthetic / reconstructed overlays on diagnostic time plots (issue #261 §9).

The full diagnostic waveform stays the primary signal; an equilibrium's
prediction of it is drawn as markers at the slices that hold one, in the same
unit, in the same panel, named by its role.  No packaged shot stores a finite
``reconstructed`` constraint value, so the fixture fills some in.
"""

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
import vaft.omas
from vaft.omas._plot_recipes import SYNTHETIC_CONSTRAINTS, build_model, normalize_entries


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def bare():
    return _load("samples/39915/omas.json.gz")


@pytest.fixture(scope="module")
def reconstructed():
    """39915 with a reconstruction written for two loops and Ip on slices 2-5."""
    ods = _load("samples/39915/omas.json.gz")
    for t in range(2, 6):
        for j in range(2):
            base = f"equilibrium.time_slice.{t}.constraints.flux_loop.{j}"
            ods[f"{base}.reconstructed"] = float(ods[f"{base}.measured"]) * 1.05
        base = f"equilibrium.time_slice.{t}.constraints.ip"
        ods[f"{base}.reconstructed"] = float(ods[f"{base}.measured"]) * 0.98
    return ods


def _by_role(model):
    return {role: [s for s in model.series if s.role == role] for role in ("", "reconstruction", "constraint")}


def test_the_waveform_stays_primary_and_the_prediction_is_markers_only(reconstructed):
    model = build_model(
        "flux_loop_time_flux", normalize_entries(reconstructed), selection=[0, 1], synthetic="equilibrium"
    )
    roles = _by_role(model)
    assert len(roles[""]) == 2 and len(roles["reconstruction"]) == 2 and not roles["constraint"]
    for measured, synthetic in zip(roles[""], roles["reconstruction"]):
        assert synthetic.channel == measured.channel and synthetic.entry == measured.entry
        assert synthetic.style["linestyle"] == "none" and synthetic.style["marker"] == "o"
        assert synthetic.x.size == 4  # slices 2..5 only: a reconstruction exists where the solver wrote one
        assert measured.x.size > 100


def test_the_prediction_is_in_the_measurement_unit(reconstructed):
    model = build_model(
        "flux_loop_time_flux", normalize_entries(reconstructed), selection=[0], synthetic="equilibrium"
    )
    measured, synthetic = _by_role(model)[""][0], _by_role(model)["reconstruction"][0]
    assert model.y_unit == "mWb"
    at_slices = np.interp(synthetic.x, measured.x, measured.y)
    # The fixture wrote 1.05 x the measured constraint, which equals the waveform at that time.
    assert np.allclose(synthetic.y, at_slices * 1.05, rtol=0.02)


def test_channels_are_matched_by_source_identifier_not_position(reconstructed):
    # Only loops 0 and 1 carry a reconstruction; a selection of loop 5 alone gets none.
    model = build_model(
        "flux_loop_time_flux", normalize_entries(reconstructed), selection=[5], synthetic="equilibrium"
    )
    assert not _by_role(model)["reconstruction"]


def test_both_adds_the_constraint_the_solver_was_given(reconstructed):
    model = build_model(
        "flux_loop_time_flux", normalize_entries(reconstructed), selection=[0], synthetic="both"
    )
    roles = _by_role(model)
    assert len(roles["constraint"]) == 1 and roles["constraint"][0].style["marker"] == "x"
    assert roles["constraint"][0].x.size == 9  # measured exists on every slice


def test_a_shot_without_a_reconstruction_draws_nothing_extra_and_does_not_fail(bare):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(bare, selection=[0, 1], synthetic="equilibrium")
    assert [line.get_label() for line in axes.lines] == ["[0] (59.2 cm, 68.5 cm)", "[1] (79.2 cm, 46.0 cm)"]
    plt.close(figure)


def test_the_default_is_no_overlay(reconstructed):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(reconstructed, selection=[0])
    assert len(axes.lines) == 1
    plt.close(figure)


def test_legends_name_the_role(reconstructed):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(reconstructed, selection=[0], synthetic="equilibrium")
    assert [t.get_text() for t in axes.get_legend().get_texts()] == [
        "[0] (59.2 cm, 68.5 cm)", "[0] (59.2 cm, 68.5 cm) (reconstruction)"
    ]
    plt.close(figure)
    figure, axes = vaft.omas.plot_plasma_current_time(reconstructed, synthetic="equilibrium")
    # A scalar plot has no channel; the title carries the shot, the legend the roles.
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ["measured", "reconstruction"]
    assert "#39915" in axes.get_title()
    plt.close(figure)


def test_the_overlay_shares_its_channel_panel_in_subplots(reconstructed):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        reconstructed, selection=[0, 1], synthetic="equilibrium", layout="subplots"
    )
    assert axes.shape == (2, 1)
    for panel in axes.ravel():
        labels = [line.get_label() for line in panel.lines]
        assert len(labels) == 2 and labels[1] == labels[0] + " (reconstruction)"
    plt.close(figure)


@pytest.mark.parametrize("name", sorted(SYNTHETIC_CONSTRAINTS))
def test_every_supported_plot_accepts_the_option(reconstructed, name):
    model = build_model(name, normalize_entries(reconstructed), synthetic="equilibrium")
    assert model is not None


def test_unsupported_plots_and_modes_are_refused(bare):
    with pytest.raises(ValueError, match="synthetic overlay is unsupported for 'mirnov_time_voltage'"):
        vaft.omas.plot_mirnov_time_voltage(bare, synthetic="equilibrium")
    with pytest.raises(ValueError, match="synthetic must be one of equilibrium, both"):
        vaft.omas.plot_flux_loop_time_flux(bare, synthetic="efit")


def test_discovery_advertises_the_capability_and_its_availability(bare, reconstructed):
    registry = vaft.omas.available_plots(query="flux loop")
    assert registry.find("flux_loop_time_flux").synthetic == {"overlay": "equilibrium"}
    assert "synthetic overlay: equilibrium" in str(registry)
    assert not vaft.omas.available_plots(query="mirnov").find("mirnov_time_voltage").synthetic
    without = vaft.omas.available_plots(bare, query="flux loop").find("flux_loop_time_flux")
    assert without.synthetic["available"] is False
    assert "synthetic overlay: equilibrium (supported, unavailable in this ODS)" in str(
        vaft.omas.available_plots(bare, query="flux loop")
    )
    with_it = vaft.omas.available_plots(reconstructed, query="ip").find("plasma_current_time")
    assert with_it.synthetic["available"] is True
