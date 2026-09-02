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
    # Exponent units are typeset in titles too; the ASCII input is unchanged.
    assert (
        display.figure_title("Electron density", "10^18 m^-3", shot=39915, time_s=0.3)
        == "Electron density [10$^{18}$ m$^{-3}$] #39915 @ 0.3 s"
    )
    assert display.subject_display_name("b_field_probe") == "B field probe"
    # Positions are stored in metres and shown in centimetres, inline.
    assert display.channel_label(0, 0.08, 0.42) == "[0] (8.0 cm, 42.0 cm)"
    assert display.channel_label(3) == "[3]"
    assert display.channel_label(3, float("nan"), 0.1) == "[3]"


# ---------------------------------------------------------------------------
# Adapter-level contract (label and data always change together)
# ---------------------------------------------------------------------------

@pytest.fixture()
def ip_ods():
    ods = omas.ODS()
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["magnetics.ip.0.time"] = np.array([0.0, 0.1, 0.2])
    ods["magnetics.ip.0.data"] = np.array([0.0, 1.0e5, 2.0e5])
    return ods


def test_plasma_current_defaults_to_ka_with_canonical_title(ip_ods):
    figure, axes = vaft.omas.plot_plasma_current_time(ip_ods)
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
    assert "[10$^{18}$ m$^{-2}$]" in axes.get_ylabel()
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
    figure, axes = vaft.omas.plot_current_overview(ip_ods)
    titles = [panel.get_title() for panel in np.asarray(axes).ravel() if panel.get_visible()]
    assert titles == ["Plasma Current"], titles
    # The identity the panels no longer repeat lives on the suptitle.
    assert figure._suptitle.get_text().endswith("#39915")

    # The same plot standalone is decorated with unit and shot.
    standalone_figure, standalone = vaft.omas.plot_plasma_current_time(ip_ods)
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


def test_an_ods_without_a_pulse_names_no_shot():
    """A container-key fallback must not be printed as a fabricated shot."""
    ods = omas.ODS()
    ods["magnetics.ip.0.time"] = np.array([0.0, 0.1])
    ods["magnetics.ip.0.data"] = np.array([0.0, 1.0e5])
    figure, axes = vaft.omas.plot_plasma_current_time(ods)
    assert axes.get_title() == "Plasma Current [kA]"
    assert "#" not in axes.get_title()
    plt.close(figure)


# ---------------------------------------------------------------------------
# Legend and channel-label policy (issue #256 sections 8 and 9, visual review)
# ---------------------------------------------------------------------------

def _loops():
    from vaft.data import sample

    return vaft.omas.load(sample(39915, "omas"))


def test_a_lone_trace_draws_no_legend(ip_ods):
    """The title already says what the one trace is."""
    figure, axes = vaft.omas.plot_plasma_current_time(ip_ods)
    assert axes.get_legend() is None
    plt.close(figure)


def test_a_single_entry_labels_channels_without_the_shot():
    """The shot is in the title, so repeating it on every entry says nothing."""
    figure, axes = vaft.omas.plot_flux_loop_time_flux(_loops(), selection=[0, 5])
    labels = [line.get_label() for line in axes.lines]
    assert labels == ["[0] (59.2 cm, 68.5 cm)", "[5] (9.1 cm, 4.0 cm)"]
    assert not any("39915" in label for label in labels)
    plt.close(figure)


def test_several_entries_of_one_channel_put_the_shot_in_the_entry():
    ods = _loops()
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        [ods, ods], label=["39915", "39916"], selection=[5]
    )
    assert [line.get_label() for line in axes.lines] == ["39915", "39916"]
    # The channel is stated once, as the legend title, not on every entry.
    assert axes.get_legend().get_title().get_text() == "[5] (9.1 cm, 4.0 cm)"
    plt.close(figure)


def test_several_entries_of_several_channels_spell_both():
    ods = _loops()
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        [ods, ods], label=["39915", "39916"], selection=[0, 5]
    )
    assert axes.lines[0].get_label() == "39915 · [0] (59.2 cm, 68.5 cm)"
    assert axes.lines[-1].get_label() == "39916 · [5] (9.1 cm, 4.0 cm)"
    plt.close(figure)


def test_a_legend_past_the_threshold_becomes_a_count():
    from vaft.plot.style import LEGEND_MAX_ENTRIES

    figure, axes = vaft.omas.plot_flux_loop_time_flux(_loops())  # 11 loops
    assert len(axes.lines) > LEGEND_MAX_ENTRIES
    assert axes.get_legend() is None
    assert any(text.get_text().endswith("traces") for text in axes.texts)
    plt.close(figure)
    # At or below the threshold the legend is drawn.
    figure, axes = vaft.omas.plot_flux_loop_time_flux(_loops(), selection="inboard")
    assert len(axes.lines) == 7 and axes.get_legend() is not None
    plt.close(figure)


def test_legend_overrides_beat_the_policy(ip_ods):
    figure, axes = vaft.omas.plot_plasma_current_time(ip_ods, legend=True)
    assert axes.get_legend() is not None
    plt.close(figure)
    figure, axes = vaft.omas.plot_flux_loop_time_flux(_loops(), selection="inboard", legend=False)
    assert axes.get_legend() is None
    plt.close(figure)


def test_a_channel_without_geometry_keeps_its_identifier():
    ods = omas.ODS()
    ods["magnetics.time"] = np.linspace(0.0, 0.1, 4)
    for index in range(2):
        ods[f"magnetics.flux_loop.{index}.name"] = f"FL0{index}"
        ods[f"magnetics.flux_loop.{index}.flux.data"] = np.ones(4)
    figure, axes = vaft.omas.plot_flux_loop_time_flux(ods)
    assert [line.get_label() for line in axes.lines] == ["FL00", "FL01"]
    plt.close(figure)


def test_a_single_sample_series_is_visible():
    """A one-slice ODS used to draw a line through one point: nothing."""
    ods = omas.ODS()
    ods["equilibrium.time"] = np.array([0.3])
    ods["equilibrium.time_slice.0.global_quantities.beta_pol"] = 0.8
    figure, axes = vaft.omas.plot_equilibrium_time_beta_p(ods)
    assert len(axes.lines[0].get_xdata()) == 1
    assert axes.lines[0].get_marker() not in ("None", "", None)
    plt.close(figure)


def test_unit_markup_is_presentation_only():
    # The ASCII spelling stays the key and the accepted input...
    assert display.resolve_display("m^-3").unit == "10^18 m^-3"
    assert display.resolve_display("A/m^2", unit="MA/m^2").unit == "MA/m^2"
    # ...and only the rendered label carries mathtext.
    assert display.unit_markup("10^18 m^-3") == "10$^{18}$ m$^{-3}$"
    assert display.unit_markup("A/m^2") == "A/m$^{2}$"
    assert display.unit_markup("kA") == "kA"
    from vaft.plot.style import axis_label
    assert axis_label("n_e", "10^18 m^-3") == "n_e [10$^{18}$ m$^{-3}$]"


def test_normalized_coordinates_are_typeset():
    ods = _loops()
    figure, axes = vaft.omas.plot_equilibrium_profile_pressure(ods, coordinate="rho_tor_norm")
    assert r"$\rho_N$" in axes.get_xlabel()
    plt.close(figure)
    figure, axes = vaft.omas.plot_equilibrium_profile_pressure(ods, coordinate="psi_norm")
    assert r"$\psi_N$" in axes.get_xlabel()
    plt.close(figure)


def test_empty_subject_and_quantity_are_never_rendered():
    """Titles and labels come from recipe text; registry fields never leak."""
    from vaft.plot import registry

    ods = _loops()
    for name in ("plasma_current_time", "flux_loop_time_flux", "equilibrium_profile_q"):
        spec = registry.get_spec(name)
        figure, axes = getattr(vaft.omas, f"plot_{name}")(ods)
        rendered = " ".join([axes.get_title(), axes.get_xlabel(), axes.get_ylabel()]
                            + [line.get_label() for line in axes.lines])
        assert "subject=" not in rendered and "quantity=" not in rendered
        assert "''" not in rendered and "None" not in rendered, (name, rendered)
        plt.close(figure)


def test_drawing_into_the_same_axes_twice_replaces_the_legend_decision():
    """A caller-supplied axes is the composable path; the policy must converge."""
    from vaft.plot.style import LEGEND_MAX_ENTRIES

    ods = _loops()
    figure, axes = plt.subplots()
    vaft.omas.plot_flux_loop_time_flux(ods, ax=axes, selection="inboard")   # 7 -> legend
    assert axes.get_legend() is not None
    vaft.omas.plot_flux_loop_time_flux(ods, ax=axes, selection="outboard")  # 11 -> note
    assert axes.get_legend() is None, "the stale legend must go when the count passes the threshold"
    notes = [t.get_text() for t in axes.texts if t.get_text().endswith("traces")]
    assert notes == ["11 traces"], notes
    vaft.omas.plot_flux_loop_time_flux(ods, ax=axes, selection="outboard")  # 15 -> one note, not two
    notes = [t.get_text() for t in axes.texts if t.get_text().endswith("traces")]
    assert notes == ["15 traces"], notes
    assert len(axes.lines) > LEGEND_MAX_ENTRIES
    plt.close(figure)


def test_a_trace_without_a_channel_never_borrows_a_legend_title():
    from vaft.plot.models import Series
    from vaft.plot.style import trace_labels

    x = np.arange(3.0)
    with_channel = Series(x=x, y=x, entry="shot1", channel="[0] (1.0 cm, 2.0 cm)")
    without = Series(x=x, y=x, entry="shot2", channel="")
    labels, title = trace_labels([with_channel, without])
    assert title is None, "no shared channel exists, so no legend title"
    assert labels == ["shot1 · [0] (1.0 cm, 2.0 cm)", "shot2"]
    # When every trace does share the one channel, the title is stated once.
    shared = Series(x=x, y=x, entry="shot2", channel="[0] (1.0 cm, 2.0 cm)")
    assert trace_labels([with_channel, shared]) == (["shot1", "shot2"], "[0] (1.0 cm, 2.0 cm)")


def test_unit_markup_keeps_a_whole_decimal_exponent():
    assert display.unit_markup("m^2.5") == "m$^{2.5}$"
    assert display.unit_markup("10^18 m^-3") == "10$^{18}$ m$^{-3}$"
