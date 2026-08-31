"""The ``vaft.omas`` plot adapters (issue #63) against the packaged sample shot."""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODC, ODS

import vaft.omas as vomas
from vaft.data import sample
from vaft.plot import registry

logging.getLogger("vaft.omas.process_wrapper").setLevel(logging.WARNING)


@pytest.fixture(scope="module")
def sample_ods():
    return vomas.load(sample(39915, "omas"))


def _line_data(axes):
    return [line.get_xydata() for line in axes.lines]


def test_every_canonical_plot_has_an_adapter():
    for name in registry.canonical_names():
        adapter = getattr(vomas, f"plot_{name}", None)
        assert callable(adapter), name
        assert adapter.__doc__ and name in adapter.__doc__


def test_available_plots_filters_by_what_the_object_actually_holds(sample_ods):
    everything = {row["name"] for row in vomas.available_plots()}
    for_shot = {row["name"] for row in vomas.available_plots(sample_ods)}

    assert for_shot < everything
    assert "plasma_current_time" in for_shot
    # 39915 carries no Thomson scattering, so its plots must not be offered.
    assert "thomson_scattering_time_electron_temperature" not in for_shot


def test_every_offered_plot_actually_renders(sample_ods):
    failures = []
    for row in vomas.available_plots(sample_ods):
        try:
            figure, _ = getattr(vomas, f"plot_{row['name']}")(sample_ods)
            plt.close(figure)
        except Exception as exc:  # pragma: no cover - reported below
            failures.append(f"{row['name']}: {type(exc).__name__}: {exc}")
    assert not failures, failures


def test_a_spectrogram_skips_channels_that_carry_no_waveform(sample_ods):
    """A declared-but-unacquired channel must not decide the default.

    The 39915 array declares 76 B-pol probes, and probe 0 has geometry only.
    Availability accepts the array because other probes hold a waveform, so
    the default channel has to be one of those rather than a hardcoded 0.
    """
    assert "magnetics.b_field_pol_probe.0.voltage.data" not in sample_ods

    figure, _ = vomas.plot_mirnov_spectrogram(sample_ods)
    plt.close(figure)

    # An explicitly requested empty channel is still an error, not a silent
    # substitution of a different probe's signal.
    with pytest.raises(ValueError, match="voltage.data is not available"):
        vomas.plot_mirnov_spectrogram(sample_ods, channel=0)


def test_adapters_default_to_no_display(sample_ods, monkeypatch):
    monkeypatch.setattr(
        plt, "show", lambda *a, **k: pytest.fail("adapter displayed implicitly")
    )
    figure, _ = vomas.plot_plasma_current_time(sample_ods)
    plt.close(figure)


def test_ods_odc_and_list_inputs_produce_the_same_artists(sample_ods):
    odc = ODC()
    odc["a"] = sample_ods

    figures = []
    results = []
    for source in (sample_ods, odc, [sample_ods]):
        figure, axes = vomas.plot_plasma_current_time(source)
        figures.append(figure)
        results.append(_line_data(axes))

    reference = results[0]
    assert len(reference) == 1
    for other in results[1:]:
        assert len(other) == len(reference)
        np.testing.assert_allclose(other[0], reference[0])
    for figure in figures:
        plt.close(figure)


def test_list_inputs_get_deterministic_labels_and_ordering(sample_ods):
    figure, axes = vomas.plot_plasma_current_time([sample_ods, sample_ods], label="key")
    assert [line.get_label() for line in axes.lines] == ["0", "1"]
    plt.close(figure)

    figure, axes = vomas.plot_plasma_current_time(
        [sample_ods, sample_ods], label=["first", "second"]
    )
    assert [line.get_label() for line in axes.lines] == ["first", "second"]
    plt.close(figure)


def test_pulse_labels_are_used_by_default(sample_ods):
    figure, axes = vomas.plot_plasma_current_time(sample_ods)
    assert axes.lines[0].get_label() == "39915"
    plt.close(figure)


def test_mismatched_explicit_labels_are_reported(sample_ods):
    with pytest.raises(ValueError, match="labels for"):
        vomas.plot_plasma_current_time([sample_ods, sample_ods], label=["only-one"])


def test_adapters_render_into_caller_supplied_axes(sample_ods):
    figure, target = plt.subplots()
    before = set(plt.get_fignums())
    returned_figure, returned_axes = vomas.plot_plasma_current_time(sample_ods, ax=target)
    assert returned_figure is figure
    assert returned_axes is target
    assert set(plt.get_fignums()) == before
    plt.close(figure)


def test_unsupported_input_types_are_reported():
    with pytest.raises(TypeError, match="omas ODS"):
        vomas.plot_plasma_current_time(42)


def test_missing_data_produces_an_actionable_error(sample_ods):
    empty = ODS(consistency_check=False)
    with pytest.raises(ValueError, match="required|not available|are present"):
        vomas.plot_equilibrium_field_psi(empty)


def test_tf_field_divides_by_reference_radius(sample_ods):
    # Regression: tf.b_field_tor_vacuum_r.data is B_t * R [T*m], not B_t [T];
    # the adapter must divide by tf.r0 to recover the field itself.
    ods = ODS(consistency_check=False)
    ods["tf.time"] = np.array([0.0, 0.1, 0.2])
    ods["tf.b_field_tor_vacuum_r.data"] = np.array([0.4, 0.4, 0.4])
    ods["tf.r0"] = 0.4

    figure, axes = vomas.plot_tf_coil_time_b_t(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0, 1.0, 1.0])
    plt.close(figure)

    figure, axes = vomas.plot_tf_coil_time_b_t_vacuum_r(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.4, 0.4, 0.4])
    plt.close(figure)


def test_tf_field_tolerates_a_missing_reference_radius():
    ods = ODS(consistency_check=False)
    ods["tf.time"] = np.array([0.0, 0.1])
    ods["tf.b_field_tor_vacuum_r.data"] = np.array([0.4, 0.4])

    figure, axes = vomas.plot_tf_coil_time_b_t(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.4, 0.4])
    plt.close(figure)


def test_power_balance_computes_the_real_terms_not_just_its_inputs(sample_ods):
    # EFIT-only data has no volume/core-profile basis for the derived balance.
    offered = {row["name"] for row in vomas.available_plots(sample_ods)}
    assert "summary_time_power_balance" not in offered


def test_machine_topview_includes_pellet_geometry():
    # Regression: the builder's own error message named "pellets" as a
    # supported IDS but never actually read one.
    ods = ODS(consistency_check=False)
    ods["pellets.time_slice.0.pellet.0.path_geometry.first_point.r"] = 0.9
    ods["pellets.time_slice.0.pellet.0.path_geometry.first_point.phi"] = 0.5

    figure, axes = vomas.plot_machine_geometry_topview(ods)
    assert any("Pellet" in line.get_label() for line in axes.lines)
    plt.close(figure)


def test_partial_composites_drop_the_absent_panels(sample_ods):
    offered = {row["name"] for row in vomas.available_plots(sample_ods)}
    assert "equilibrium_time_beta" not in offered

    empty = ODS(consistency_check=False)
    with pytest.raises(ValueError, match="none of the panels"):
        vomas.plot_equilibrium_time_beta(empty)


def test_channel_line_plots_ignore_scalar_placeholder_channels():
    """A populated IDS may contain scalar NaN placeholders after valid rows."""
    ods = ODS(consistency_check=False)
    time = np.array([0.0, 0.1, 0.2])
    ods["magnetics.ip.0.time"] = time
    ods["magnetics.ip.0.data"] = np.array([1.0, 2.0, 3.0])
    ods["magnetics.b_field_pol_probe.0.field.time"] = time
    ods["magnetics.b_field_pol_probe.0.field.data"] = np.array([0.1, 0.2, 0.3])
    ods["magnetics.b_field_pol_probe.1.field.time"] = np.nan
    ods["magnetics.b_field_pol_probe.1.field.data"] = np.nan

    figure, axes = vomas.plot_b_field_probe_time_field(ods)
    assert len(axes.lines) == 1
    plt.close(figure)

    figure, axes = vomas.plot_magnetics_overview(ods)
    assert axes.size == 2
    plt.close(figure)


class TestPlotMethods:
    def teardown_method(self):
        vomas.disable_plot_methods()

    def test_importing_vaft_does_not_mutate_ods(self):
        import importlib

        importlib.import_module("vaft")
        assert not hasattr(ODS, "plot_plasma_current_time")

    def test_registration_is_explicit_and_idempotent(self, sample_ods):
        first = vomas.enable_plot_methods()
        assert "plot_plasma_current_time" in first
        assert vomas.enable_plot_methods() == first

        figure, axes = sample_ods.plot_plasma_current_time()
        assert len(axes.lines) == 1
        plt.close(figure)

    def test_canonical_names_avoid_omas_native_methods(self):
        # OMAS ships its own plot_* methods; the canonical grammar must not
        # collide with any of them.
        native = {name for name in dir(ODS) if name.startswith("plot_")}
        ours = {f"plot_{name}" for name in registry.canonical_names()}
        assert not native & ours, sorted(native & ours)

    def test_collisions_are_refused_unless_overwrite_is_requested(self):
        name = f"plot_{registry.canonical_names()[0]}"
        setattr(ODS, name, lambda self: "pre-existing")
        try:
            with pytest.raises(RuntimeError, match="refusing to replace"):
                vomas.enable_plot_methods()
            assert ODS().__getattribute__(name)() == "pre-existing"
            vomas.enable_plot_methods(overwrite=True)
            assert name in getattr(ODS, "_vaft_plot_methods")
        finally:
            try:
                delattr(ODS, name)
            except AttributeError:
                pass


class TestOverlayMethods:
    """``enable_overlay_methods`` gives OMAS' own overlays the ax/show contract."""

    def teardown_method(self):
        vomas.disable_overlay_methods()

    def test_importing_vaft_does_not_wrap_omas_overlays(self):
        import importlib

        importlib.import_module("vaft")
        assert not getattr(ODS.plot_wall_overlay, "_vaft_overlay_wrapper", False)

    def test_registration_is_idempotent(self):
        first = vomas.enable_overlay_methods()
        assert "plot_wall_overlay" in first
        assert "plot_pf_active_overlay" in first
        wrapper = ODS.plot_wall_overlay
        assert vomas.enable_overlay_methods() == first
        # A second call must not wrap the wrapper.
        assert ODS.plot_wall_overlay is wrapper

    def test_disable_restores_the_original_omas_callables(self):
        originals = {
            name: getattr(ODS, name)
            for name in (
                "plot_wall_overlay",
                "plot_pf_active_overlay",
                "plot_magnetics_overlay",
            )
        }
        vomas.enable_overlay_methods()
        assert getattr(ODS, "plot_wall_overlay") is not originals["plot_wall_overlay"]
        vomas.disable_overlay_methods()
        for name, original in originals.items():
            assert getattr(ODS, name) is original, name

    def test_omitted_ax_gives_each_overlay_its_own_figure(self, sample_ods):
        vomas.enable_overlay_methods()
        before = set(plt.get_fignums())
        sample_ods.plot_wall_overlay(color="lightgray")
        sample_ods.plot_pf_active_overlay(edgecolor="red")
        created = set(plt.get_fignums()) - before
        try:
            assert len(created) == 2
        finally:
            for number in created:
                plt.close(number)

    def test_caller_supplied_ax_creates_nothing_and_is_never_closed(self, sample_ods):
        vomas.enable_overlay_methods()
        figure, axes = plt.subplots()
        try:
            existing = set(plt.get_fignums())
            sample_ods.plot_wall_overlay(ax=axes, color="lightgray")
            sample_ods.plot_pf_active_overlay(ax=axes, edgecolor="red")
            assert set(plt.get_fignums()) == existing
            assert figure.number in plt.get_fignums()
        finally:
            plt.close(figure)

    def test_overlays_default_to_no_display(self, monkeypatch, sample_ods):
        vomas.enable_overlay_methods()
        monkeypatch.setattr(plt, "show", lambda *a, **k: pytest.fail("show() called"))
        figure, _ = plt.subplots()
        try:
            sample_ods.plot_wall_overlay(ax=figure.axes[0])
        finally:
            plt.close(figure)

    def test_vaft_canonical_adapters_are_never_wrapped(self):
        """``plot_camera_visible_image_efit_overlay`` matches the name pattern.

        It is one of ours and already honors the ax/show contract, so wrapping
        it would resolve the axes twice, drop the renderer's figsize and call
        finalize twice. Neither opt-in order may wrap it.
        """
        canonical = {f"plot_{name}" for name in registry.canonical_names()}
        assert any(name.endswith("_overlay") for name in canonical), (
            "this test is only meaningful while a canonical plot ends in _overlay"
        )

        for plot_methods_first in (True, False):
            if plot_methods_first:
                vomas.enable_plot_methods()
                wrapped = set(vomas.enable_overlay_methods())
            else:
                vomas.enable_overlay_methods()
                vomas.enable_plot_methods()
                wrapped = set(getattr(ODS, "_vaft_overlay_methods", frozenset()))
            try:
                assert not wrapped & canonical, sorted(wrapped & canonical)
                adapter = ODS.plot_camera_visible_image_efit_overlay
                assert not getattr(adapter, "_vaft_overlay_wrapper", False)
            finally:
                vomas.disable_overlay_methods()
                vomas.disable_plot_methods()

    def test_twodim_geometry_all_still_composes_onto_one_figure(self, sample_ods):
        """The wrapper must not scatter an existing composition across figures.

        `vaft.plot.twodim.twodim_geometry_all` draws ten overlays that belong on
        one axes; before this it leaned on pyplot's current axes, which the
        wrapper's "omitted ax means a new figure" rule would have turned into
        ten separate figures.
        """
        from vaft.plot.twodim import twodim_geometry_all

        plt.close("all")
        twodim_geometry_all(sample_ods)
        unwrapped = len(plt.get_fignums())
        plt.close("all")

        vomas.enable_overlay_methods()
        twodim_geometry_all(sample_ods)
        wrapped = len(plt.get_fignums())
        plt.close("all")

        assert unwrapped == 1
        assert wrapped == 1

    def test_the_aggregate_plot_overlay_dispatcher_is_not_wrapped(self, sample_ods):
        """`plot_overlay` matches the name pattern but is a dispatcher.

        Its `return_overlay_list=True` path draws nothing, so wrapping it would
        leak one blank figure per query.
        """
        wrapped = vomas.enable_overlay_methods()
        assert "plot_overlay" not in wrapped

        plt.close("all")
        sample_ods.plot_overlay(return_overlay_list=True)
        try:
            assert plt.get_fignums() == []
        finally:
            plt.close("all")

    def test_ax_may_be_passed_positionally(self, sample_ods):
        """OMAS declares `ax` positional-or-keyword, so `plot_x_overlay(ax)` is legal."""
        vomas.enable_overlay_methods()
        figure, axes = plt.subplots()
        try:
            before = set(plt.get_fignums())
            sample_ods.plot_wall_overlay(axes, color="lightgray")
            assert set(plt.get_fignums()) == before
            assert axes.get_children()
        finally:
            plt.close(figure)

    def test_overwrite_unwraps_instead_of_nesting(self):
        """A nested wrapper would leave ODS permanently patched after disable."""
        pristine = ODS.plot_wall_overlay
        vomas.enable_overlay_methods()
        # Clearing the bookkeeping while the wrappers stay installed is what
        # reaches the overwrite path.
        ODS._vaft_overlay_methods = frozenset()
        with pytest.raises(RuntimeError, match="refusing to re-wrap"):
            vomas.enable_overlay_methods()
        vomas.enable_overlay_methods(overwrite=True)

        inner = getattr(ODS.plot_wall_overlay, "__wrapped__", None)
        assert not getattr(inner, "_vaft_overlay_wrapper", False), "wrapper was nested"

        vomas.disable_overlay_methods()
        assert ODS.plot_wall_overlay is pristine

    def test_wrapping_covers_the_overlays_vaft_plot_twodim_relies_on(self):
        wrapped = set(vomas.enable_overlay_methods())
        # vaft/plot/twodim.py calls these OMAS overlays directly; an OMAS
        # release that renames them must fail here rather than silently.
        required = {
            "plot_bolometer_overlay",
            "plot_charge_exchange_overlay",
            "plot_gas_injection_overlay",
            "plot_interferometer_overlay",
            "plot_langmuir_probes_overlay",
            "plot_magnetics_overlay",
            "plot_pf_active_overlay",
            "plot_position_control_overlay",
            "plot_thomson_scattering_overlay",
            "plot_wall_overlay",
        }
        assert required <= wrapped, sorted(required - wrapped)
