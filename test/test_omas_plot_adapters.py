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
    assert "magnetics_time_ip" in for_shot
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

    figure, _ = vomas.plot_magnetics_spectrogram_mirnov(sample_ods)
    plt.close(figure)

    # An explicitly requested empty channel is still an error, not a silent
    # substitution of a different probe's signal.
    with pytest.raises(ValueError, match="voltage.data is not available"):
        vomas.plot_magnetics_spectrogram_mirnov(sample_ods, channel=0)


def test_adapters_default_to_no_display(sample_ods, monkeypatch):
    monkeypatch.setattr(
        plt, "show", lambda *a, **k: pytest.fail("adapter displayed implicitly")
    )
    figure, _ = vomas.plot_magnetics_time_ip(sample_ods)
    plt.close(figure)


def test_ods_odc_and_list_inputs_produce_the_same_artists(sample_ods):
    odc = ODC()
    odc["a"] = sample_ods

    figures = []
    results = []
    for source in (sample_ods, odc, [sample_ods]):
        figure, axes = vomas.plot_magnetics_time_ip(source)
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
    figure, axes = vomas.plot_magnetics_time_ip([sample_ods, sample_ods], label="key")
    assert [line.get_label() for line in axes.lines] == ["0", "1"]
    plt.close(figure)

    figure, axes = vomas.plot_magnetics_time_ip(
        [sample_ods, sample_ods], label=["first", "second"]
    )
    assert [line.get_label() for line in axes.lines] == ["first", "second"]
    plt.close(figure)


def test_pulse_labels_are_used_by_default(sample_ods):
    figure, axes = vomas.plot_magnetics_time_ip(sample_ods)
    assert axes.lines[0].get_label() == "39915"
    plt.close(figure)


def test_mismatched_explicit_labels_are_reported(sample_ods):
    with pytest.raises(ValueError, match="labels for"):
        vomas.plot_magnetics_time_ip([sample_ods, sample_ods], label=["only-one"])


def test_adapters_render_into_caller_supplied_axes(sample_ods):
    figure, target = plt.subplots()
    before = set(plt.get_fignums())
    returned_figure, returned_axes = vomas.plot_magnetics_time_ip(sample_ods, ax=target)
    assert returned_figure is figure
    assert returned_axes is target
    assert set(plt.get_fignums()) == before
    plt.close(figure)


def test_unsupported_input_types_are_reported():
    with pytest.raises(TypeError, match="omas ODS"):
        vomas.plot_magnetics_time_ip(42)


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

    figure, axes = vomas.plot_tf_time_b_field_tor(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0, 1.0, 1.0])
    plt.close(figure)

    figure, axes = vomas.plot_tf_time_b_field_tor_vacuum_r(ods)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.4, 0.4, 0.4])
    plt.close(figure)


def test_tf_field_tolerates_a_missing_reference_radius():
    ods = ODS(consistency_check=False)
    ods["tf.time"] = np.array([0.0, 0.1])
    ods["tf.b_field_tor_vacuum_r.data"] = np.array([0.4, 0.4])

    figure, axes = vomas.plot_tf_time_b_field_tor(ods)
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
    assert "summary_time_beta" not in offered

    empty = ODS(consistency_check=False)
    with pytest.raises(ValueError, match="none of the panels"):
        vomas.plot_summary_time_beta(empty)


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

    figure, axes = vomas.plot_magnetics_time_b_field_pol_probe_field(ods)
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
        assert not hasattr(ODS, "plot_magnetics_time_ip")

    def test_registration_is_explicit_and_idempotent(self, sample_ods):
        first = vomas.enable_plot_methods()
        assert "plot_magnetics_time_ip" in first
        assert vomas.enable_plot_methods() == first

        figure, axes = sample_ods.plot_magnetics_time_ip()
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
