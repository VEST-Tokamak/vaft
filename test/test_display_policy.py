"""Scientific display policy contract (issue #256, phase C).

Unit, scale, and notation are separate concepts resolved through one table:
changing the display unit always rescales the data and relabels the axis
together, unsupported units fail loudly instead of silently keeping factor 1,
and titles/channel labels follow the canonical grammar.  Policy: ``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft.omas
from vaft.plot import display


# ---------------------------------------------------------------------------
# Table-level contract
# ---------------------------------------------------------------------------

def test_every_quantity_maps_canonical_to_one_and_default_is_allowed():
    for quantity in display.QUANTITIES.values():
        assert quantity.units[quantity.canonical_unit] == 1.0, quantity.name
        assert quantity.default in quantity.units, quantity.name


def test_resolution_defaults_follow_the_ratified_table():
    assert display.resolve_display("A").unit == "kA"
    assert display.resolve_display("A").scale == pytest.approx(1e-3)
    assert display.resolve_display("Wb").unit == "mWb"
    assert display.resolve_display("T").unit == "mT"
    assert display.resolve_display("eV").unit == "eV"
    assert display.resolve_display("J").unit == "J"
    assert display.resolve_display("W").unit == "kW"
    assert display.resolve_display("m^-3").unit == "10^18 m^-3"
    assert display.resolve_display("m^-3").scale == pytest.approx(1e-18)
    assert display.resolve_display("s").unit == "s"


def test_subject_defaults_override_the_quantity_default():
    # mT for probes, T for the TF field; Torr + scientific for barometry.
    assert display.resolve_display("T", subject="b_field_probe").unit == "mT"
    assert display.resolve_display("T", subject="tf_coil").unit == "T"
    barometry = display.resolve_display("Pa", subject="barometry")
    assert barometry.unit == "Torr"
    assert barometry.notation == "scientific"
    assert display.resolve_display("Pa", subject="equilibrium").unit == "Pa"


def test_explicit_unit_always_wins_and_scales():
    spec = display.resolve_display("A", unit="MA", subject="plasma_current")
    assert (spec.unit, spec.scale) == ("MA", pytest.approx(1e-6))
    # The old bug class: voltage relabeled to mV must actually scale by 1e3.
    spec = display.resolve_display("V", unit="mV")
    assert (spec.unit, spec.scale) == ("mV", pytest.approx(1e3))


def test_unsupported_units_raise_naming_the_alternatives():
    with pytest.raises(ValueError, match="kA"):
        display.resolve_display("A", unit="furlongs")
    # No silent factor-1 fallback for a typo either.
    with pytest.raises(ValueError, match="supported units"):
        display.resolve_display("V", unit="kw")
    # Pass-through quantities accept only their canonical unit.
    assert display.resolve_display("a.u.").scale == 1.0
    with pytest.raises(ValueError, match="no display conversions"):
        display.resolve_display("a.u.", unit="mV")


def test_torr_conversion_is_numeric_not_just_a_label():
    spec = display.resolve_display("Pa", unit="Torr")
    assert spec.scale == pytest.approx(1.0 / 133.322368)


def test_auto_unit_uses_median_magnitude_and_is_deterministic():
    data = np.array([2.0e5, 1.5e5, 1.8e5])  # amperes -> kA window
    spec = display.resolve_display("A", unit="auto", data=data)
    assert spec.unit == "kA"
    spec = display.resolve_display("A", unit="auto", data=data * 1e2)
    assert spec.unit == "MA"
    # Same data, same answer.
    again = display.resolve_display("A", unit="auto", data=data)
    assert again.unit == "kA"
    with pytest.raises(ValueError, match="requires the plotted data"):
        display.resolve_display("A", unit="auto")


def test_titles_and_channel_labels_follow_the_grammar():
    assert (
        display.figure_title("Flux loop", "mWb", shot=39915)
        == "Flux loop [mWb] #39915"
    )
    assert (
        display.figure_title("Flux loop", "mWb", shot=39915, coordinates=True)
        == "Flux loop [mWb] #39915 — (R [m], Z [m])"
    )
    assert (
        display.figure_title("Electron density", "10^18 m^-3", shot=39915, time_s=0.3)
        == "Electron density [10^18 m^-3] #39915 @ 0.3 s"
    )
    assert display.subject_display_name("b_field_probe") == "B field probe"
    assert display.channel_label(0, 0.08, 0.42) == "[0] (0.08, 0.42)"
    assert display.channel_label(3) == "[3]"


# ---------------------------------------------------------------------------
# Adapter-level contract (label and data always change together)
# ---------------------------------------------------------------------------

@pytest.fixture()
def ip_ods():
    ods = omas.ODS()
    ods["magnetics.ip.0.time"] = np.array([0.0, 0.1, 0.2])
    ods["magnetics.ip.0.data"] = np.array([0.0, 1.0e5, 2.0e5])
    return ods


def test_plasma_current_defaults_to_ka_with_canonical_title(ip_ods):
    figure, axes = vaft.omas.plot_plasma_current_time(ip_ods, label=["39915"])
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.0, 100.0, 200.0])
    assert "[kA]" in axes.get_ylabel()
    # The heading is the recipe's own title, so siblings that share a display
    # unit stay distinguishable ("MHD Stored Energy" vs "Magnetic Stored Energy").
    assert axes.get_title() == "Plasma Current [kA] #39915"
    plt.close(figure)


def test_unit_override_rescales_and_relabels_together(ip_ods):
    figure, axes = vaft.omas.plot_plasma_current_time(ip_ods, yunit="MA")
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.0, 0.1, 0.2])
    assert "[MA]" in axes.get_ylabel()
    plt.close(figure)


def test_unsupported_adapter_unit_fails_loudly(ip_ods):
    with pytest.raises(ValueError, match="supported units"):
        vaft.omas.plot_plasma_current_time(ip_ods, yunit="furlongs")


def test_time_axis_resolves_through_the_same_policy(ip_ods):
    figure, axes = vaft.omas.plot_plasma_current_time(ip_ods, xunit="ms")
    np.testing.assert_allclose(axes.lines[0].get_xdata(), [0.0, 100.0, 200.0])
    assert "[ms]" in axes.get_xlabel()
    plt.close(figure)


def test_profile_scaling_matches_its_label():
    # Regression: _build_profile_1d used to relabel without rescaling.
    ods = omas.ODS()
    ods["equilibrium.time_slice.0.profiles_1d.psi_norm"] = np.linspace(0, 1, 4)
    ods["equilibrium.time_slice.0.profiles_1d.pressure"] = np.array(
        [400.0, 300.0, 200.0, 0.0]
    )
    figure, axes = vaft.omas.plot_equilibrium_profile_pressure(
        ods, coordinate="psi_norm", yunit="kPa"
    )
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.4, 0.3, 0.2, 0.0])
    assert "[kPa]" in axes.get_ylabel()
    plt.close(figure)


def test_interferometer_scaled_axis_comes_from_the_table():
    ods = omas.ODS()
    ods["interferometer.channel.0.n_e_line.time"] = np.array([0.0, 0.1])
    ods["interferometer.channel.0.n_e_line.data"] = np.array([1.0e18, 2.0e18])
    figure, axes = vaft.omas.plot_interferometer_time_n_e_line(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0, 2.0])
    assert "[10^18 m^-2]" in axes.get_ylabel()
    plt.close(figure)


def test_beta_members_keep_their_own_dimensionless_conventions():
    """Related quantities are not interchangeable: beta_t is a percent, beta_p is not."""
    ods = omas.ODS()
    ods["equilibrium.time"] = np.array([0.1, 0.2])
    for index, (beta_t, beta_p) in enumerate(((0.025, 0.8), (0.030, 0.9))):
        slice_path = f"equilibrium.time_slice.{index}.global_quantities"
        ods[f"{slice_path}.beta_tor"] = beta_t
        ods[f"{slice_path}.beta_pol"] = beta_p

    figure, axes = vaft.omas.plot_equilibrium_time_beta_t(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [2.5, 3.0])
    assert axes.get_ylabel() == "Toroidal Beta [%]"
    plt.close(figure)

    figure, axes = vaft.omas.plot_equilibrium_time_beta_p(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.8, 0.9])
    assert axes.get_ylabel() == "Poloidal Beta"
    plt.close(figure)

    # The percent convention is the only one on offer for that quantity.
    with pytest.raises(ValueError, match="displayed as"):
        vaft.omas.plot_equilibrium_time_beta_t(ods, yunit="kA")


def test_panel_members_keep_the_short_recipe_title(ip_ods):
    """A composite carries the shot in its suptitle, not on every panel."""
    figure, axes = vaft.omas.plot_current_overview(ip_ods, label=["39915"])
    titles = [panel.get_title() for panel in np.asarray(axes).ravel() if panel.get_visible()]
    assert titles == ["Plasma Current"], titles

    # The same plot standalone is decorated with unit and shot.
    standalone_figure, standalone = vaft.omas.plot_plasma_current_time(
        ip_ods, label=["39915"]
    )
    assert standalone.get_title() == "Plasma Current [kA] #39915"
    plt.close(figure)
    plt.close(standalone_figure)


def test_multi_shot_titles_leave_shot_identity_to_the_legend(ip_ods):
    figure, axes = vaft.omas.plot_plasma_current_time(
        [ip_ods, ip_ods], label=["39915", "39916"]
    )
    assert "#" not in axes.get_title()
    labels = [line.get_label() for line in axes.lines]
    assert labels == ["39915", "39916"]
    plt.close(figure)
