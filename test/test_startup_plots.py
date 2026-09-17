"""The startup tutorial's plots (issue #888).

ECH launched power and Rogowski currents as line plots, every spectral line at
once, the vacuum startup proxies against time (several shots overlaid), the
midplane cut of the vacuum map with its empirical thresholds, the ECR line on
the 2-D map, and vacuum field lines over a camera frame.
"""

import copy
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

import vaft
import vaft.omas as vomas
from vaft.formula.startup import (
    LLOYD_FIGURE_OF_MERIT_ECH_V_PER_M,
    LLOYD_FIGURE_OF_MERIT_OHMIC_V_PER_M,
    electron_cyclotron_resonance_radius,
)
from vaft.plot.backend import recipes
from vaft.plot.models import LineSeries, Panels

COARSE = 21


@pytest.fixture(scope="module")
def sample():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vomas.sample_ods()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _names(ods):
    return {row.name for row in vomas.available_plots(ods)}


# ---------------------------------------------------------------------------
# ec_launchers_time_power
# ---------------------------------------------------------------------------

def _ec_ods() -> ODS:
    rng = np.random.default_rng(0)
    time = np.linspace(0.25, 0.35, 1001)
    power = np.where((time > 0.28) & (time < 0.32), 6.0e3, 0.0) + 50.0 * rng.standard_normal(time.size)
    power[[300, 400, 500]] += 5.0e4  # spikes
    power[600:610] = np.nan  # detector out of range
    ods = ODS(consistency_check=False)
    ods["ec_launchers.beam.0.name"] = "VEST 2.45 GHz magnetron"
    ods["ec_launchers.beam.0.identifier"] = "ech_2p45ghz"
    ods["ec_launchers.beam.0.power_launched.data"] = power
    ods["ec_launchers.beam.0.power_launched.time"] = time
    return ods


def test_the_ec_power_plot_is_offered_only_with_ec_launchers():
    assert "ec_launchers_time_power" not in _names(_rogowski_ods())
    assert "ec_launchers_time_power" in _names(_ec_ods())
    assert "rogowski_coil_time_current" not in _names(_ec_ods())


def test_the_ec_power_is_one_trace_per_beam_named_by_the_beam():
    model = vomas.extract_ec_launchers_time_power(_ec_ods())
    assert isinstance(model, LineSeries)
    (trace,) = model.series
    assert "VEST 2.45 GHz magnetron" in trace.label
    assert model.y_unit == "kW"
    assert np.nanmax(trace.y) > 40.0  # the spikes, in kW


def test_the_ec_power_smooths_away_its_spikes():
    model = vomas.extract_ec_launchers_time_power(_ec_ods(), smooth=2e-3)
    (trace,) = model.series
    assert np.nanmax(trace.y) < 7.0
    assert np.isnan(np.asarray(trace.y)[600:610]).all()
    assert "median 2 ms" in model.title
    figure, _ = vomas.plot_ec_launchers_time_power(_ec_ods(), smooth=2e-3)
    plt.close(figure)


# ---------------------------------------------------------------------------
# rogowski_coil_time_current
# ---------------------------------------------------------------------------

def _rogowski_ods() -> ODS:
    time = np.linspace(0.25, 0.35, 501)
    ods = ODS(consistency_check=False)
    for index, (name, scale) in enumerate((("plasma current", 8e4), ("diamagnetic TF", 2e3))):
        ods[f"magnetics.rogowski_coil.{index}.name"] = name
        ods[f"magnetics.rogowski_coil.{index}.identifier"] = f"rogowski_coil:{index}"
        ods[f"magnetics.rogowski_coil.{index}.current.data"] = scale * np.sin(np.pi * (time - 0.25) / 0.1)
        ods[f"magnetics.rogowski_coil.{index}.current.time"] = time
    return ods


def test_the_rogowski_plot_draws_each_coil_under_its_name(sample):
    assert "rogowski_coil_time_current" in _names(_rogowski_ods())
    model = vomas.extract_rogowski_coil_time_current(_rogowski_ods())
    labels = [trace.label for trace in model.series]
    assert any("plasma current" in label for label in labels)
    assert any("diamagnetic TF" in label for label in labels)
    figure, axes = vomas.plot_rogowski_coil_time_current(_rogowski_ods())
    assert len(axes.lines) == 2
    plt.close(figure)


# ---------------------------------------------------------------------------
# spectrometer_uv_time_intensity(emission="all")
# ---------------------------------------------------------------------------

def test_every_processed_line_of_every_channel_is_drawn(sample):
    model = vomas.extract_spectrometer_uv_time_intensity(sample, emission="all")
    labels = [trace.label for trace in model.series]
    assert len(labels) == 9
    for line in ("H-alpha_6563", "OI_7770", "H-beta_4861", "H-gamma_4340", "CII_4267",
                 "CIII_1909", "OII_3726", "OV_629"):
        assert any(label.endswith(line) for label in labels), line
    assert sum(label.endswith("H-alpha_6563") for label in labels) == 2
    assert len(set(labels)) == 9, "the two H-alpha channels must stay distinguishable"


# ---------------------------------------------------------------------------
# startup_proxies_time
# ---------------------------------------------------------------------------

def _shifted(ods, dt):
    """A copy of ``ods`` with every stored time base moved by ``dt``."""
    moved = copy.deepcopy(ods)
    for path, value in ods.flat().items():
        if path.split(".")[-1] == "time":
            moved[path] = np.asarray(value, dtype=float) + dt
    return moved


@pytest.fixture(scope="module")
def proxies(sample):
    return vomas.extract_startup_proxies_time(sample)


def test_three_stacked_panels_in_display_units(proxies):
    assert isinstance(proxies, Panels) and len(proxies.models) == 3
    b_z, v_loop, decay = proxies.models
    assert b_z.y_unit in ("G", "mT", "T")
    assert v_loop.y_unit == "V"
    assert decay.y_unit == ""
    assert decay.y_limits == recipes.STARTUP_DECAY_INDEX_LIMITS


def test_the_default_window_runs_from_oh_onset_to_ip_peak(sample, proxies):
    from vaft.omas.discharge_timing import oh_coil_onset
    from vaft.omas.plasma_features import ip_peak

    start, stop = float(oh_coil_onset(sample).time), float(ip_peak(sample).time)
    trace = proxies.models[0].series[0]
    assert trace.x.min() >= start and trace.x.max() <= stop
    assert proxies.models[0].x_limits == (start, stop)


def test_the_values_are_the_helper_s(sample, proxies):
    from vaft.omas.process_wrapper import compute_startup_proxies_ods

    expected = compute_startup_proxies_ods(recipes._isolated_copy(sample, recipes._STARTUP_ROOTS))
    trace = proxies.models[1].series[0]
    index = np.searchsorted(expected["time"], trace.x)
    np.testing.assert_allclose(trace.y, expected["v_loop"][index])


def test_the_breakdown_onset_is_marked_and_the_stable_band_drawn(sample, proxies):
    onset = vomas.find_breakdown_onset(sample)
    for panel in proxies.models:
        verticals = [s for s in panel.series if s.x.size == 2 and s.x[0] == s.x[1]]
        assert verticals and verticals[0].x[0] == pytest.approx(onset)
        assert verticals[0].style["linestyle"] == "--"
    levels = {float(s.y[0]) for s in proxies.models[2].series if s.y.size == 2 and s.y[0] == s.y[1]}
    assert {0.0, 1.5} <= levels
    assert "time" not in sample["pf_passive"], "the eddy solve landed in the caller's ODS"


def test_markers_can_be_turned_off(sample):
    model = vomas.extract_startup_proxies_time(sample, markers=False, rz=(0.45, 0.0))
    assert all(s.x.size > 2 for s in model.models[0].series)
    assert "R = 0.45 m" in model.suptitle


def test_several_entries_overlay_with_their_own_onsets_in_their_own_colours(sample):
    later = _shifted(sample, 2e-3)
    model = vomas.extract_startup_proxies_time([sample, later], label=["first", "later"])
    top = model.models[0]
    traces = [s for s in top.series if s.x.size > 2]
    markers = [s for s in top.series if s.x.size == 2 and s.x[0] == s.x[1]]
    assert [t.entry for t in traces] == ["first", "later"]
    assert traces[0].style["color"] != traces[1].style["color"]
    assert len(markers) == 2
    assert markers[1].x[0] - markers[0].x[0] == pytest.approx(2e-3, abs=1e-4)
    assert [m.style["color"] for m in markers] == [t.style["color"] for t in traces]
    figure, axes = vomas.plot_startup_proxies_time([sample, later], label=["first", "later"])
    assert len(axes) == 3
    plt.close(figure)


def test_a_malformed_point_is_refused(sample):
    with pytest.raises(ValueError, match="rz"):
        vomas.extract_startup_proxies_time(sample, rz=(0.4,))


# ---------------------------------------------------------------------------
# vacuum_field_midplane
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["v_loop", "b_z", "breakdown", "lloyd_margin"])
def test_each_midplane_field_is_a_radial_profile(sample, field):
    # The fill pressure is pinned for the margin: this is about the profile's
    # shape, and a prefill low enough that A p L < 1 everywhere has no margin.
    pinned = {"p_Pa": 2.7e-3} if field == "lloyd_margin" else {}
    model = vomas.extract_vacuum_field_midplane(sample, field=field, resolution=COARSE, **pinned)
    assert isinstance(model, LineSeries)
    profile = model.series[0]
    assert np.isfinite(profile.y).any() and np.isnan(profile.y).any()  # masked outside the limiter
    assert model.x_unit == "m"
    ecr = [s for s in model.series if "GHz ECR" in s.label]
    assert len(ecr) == 1


def test_the_breakdown_cut_carries_both_empirical_thresholds(sample):
    model = vomas.extract_vacuum_field_midplane(sample, field="breakdown", resolution=COARSE)
    levels = {s.label: float(s.y[0]) for s in model.series if s.y.size == 2 and s.y[0] == s.y[1]}
    assert levels["Ohmic threshold (1000 V/m)"] == LLOYD_FIGURE_OF_MERIT_OHMIC_V_PER_M
    assert levels["ECH-assisted threshold (100 V/m)"] == LLOYD_FIGURE_OF_MERIT_ECH_V_PER_M
    assert model.log_y


def test_the_ecr_line_is_the_formula_at_the_drawn_instant(sample):
    from vaft.omas.process_wrapper import _vacuum_toroidal_product

    model = vomas.extract_vacuum_field_midplane(sample, field="b_z", resolution=COARSE)
    (line,) = [s for s in model.series if "GHz ECR" in s.label]
    instant = recipes._vacuum_psi_time(sample)
    base = np.asarray(sample["pf_active.time"], dtype=float)
    instant = float(base[np.argmin(np.abs(base - instant))])
    expected = electron_cyclotron_resonance_radius(_vacuum_toroidal_product(sample, instant), 2.45e9)
    assert line.x[0] == pytest.approx(expected)
    none = vomas.extract_vacuum_field_midplane(sample, field="b_z", resolution=COARSE, ec_frequency_Hz=None)
    assert not [s for s in none.series if "ECR" in s.label]


def test_the_margin_takes_a_fill_pressure(sample):
    low = vomas.extract_vacuum_field_midplane(sample, field="lloyd_margin", resolution=COARSE, p_Pa=1e-3)
    assert "p = 1.00 mPa" in low.title


def test_an_unknown_midplane_field_names_the_vocabulary(sample):
    with pytest.raises(ValueError, match="v_loop"):
        vomas.extract_vacuum_field_midplane(sample, field="psi")


def test_the_midplane_cut_steps_with_a_slider_and_a_field_choice(sample):
    result = vomas.plot_vacuum_field_midplane(
        sample, interactive=True, interaction_backend="none", resolution=COARSE,
    )
    controls = {control.name: control for control in result.controls}
    slider = controls["time_index"]
    assert slider.kind == "range"
    assert controls["field"].options == recipes.VACUUM_MIDPLANE_FIELD_NAMES
    base = np.asarray(sample["pf_active.time"], dtype=float)
    assert f"{base[slider.default] * 1e3:.1f} ms" in result.axes.get_title()
    result.state.set("time_index", slider.default + 25)
    assert f"{base[slider.default + 25] * 1e3:.1f} ms" in result.axes.get_title()
    result.state.set("field", "b_z")
    assert "Vertical field" in result.axes.get_title()
    plt.close(result.figure)


def test_fields_without_their_inputs_are_not_offered(sample):
    from vaft.plot.backend.discovery import describe_one

    stripped = recipes._isolated_copy(sample, ("pf_active", "pf_passive", "wall", "equilibrium"))
    record = describe_one("vacuum_field_midplane", [("no tf", stripped)])
    assert set(record.fields["options"]) == {"v_loop", "b_z"}


# ---------------------------------------------------------------------------
# plot_vacuum_field: the ECR line and p_Pa=
# ---------------------------------------------------------------------------

def test_the_2d_map_draws_the_ecr_line_only_when_asked(sample):
    plain = vomas.extract_vacuum_field(sample, field="psi", resolution=COARSE)
    assert not [layer for layer in plain.overlays if "ECR" in layer.label]
    marked = vomas.extract_vacuum_field(sample, field="psi", resolution=COARSE, ec_frequency_Hz=2.45e9)
    lines = [layer for layer in marked.overlays if "ECR" in layer.label and layer.kind == "polyline"]
    assert len(lines) == 1 and lines[0].r[0] == lines[0].r[1]


def test_the_ecr_line_follows_the_slider(sample):
    result = vomas.plot_vacuum_field(
        sample, field="psi", interactive=True, interaction_backend="none",
        resolution=COARSE, ec_frequency_Hz=2.45e9,
    )

    def ecr_radius():
        lines = [line for line in result.axes.lines if "GHz ECR" in line.get_label()]
        assert len(lines) == 1
        return float(lines[0].get_xdata()[0])

    before = ecr_radius()
    default = result.state["time_index"]
    result.state.set("time_index", default - 400)
    assert ecr_radius() != pytest.approx(before)
    plt.close(result.figure)


def test_p_pa_is_an_accepted_option_of_the_vacuum_map(sample):
    """Regression: the builder took p_Pa= but the option schema refused it."""
    figure, _ = vomas.plot_vacuum_field(sample, field="lloyd_margin", p_Pa=3e-3, resolution=COARSE)
    plt.close(figure)


# ---------------------------------------------------------------------------
# vacuum field lines over a camera frame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def camera_sample(sample):
    import matplotlib.image as mpimg

    from vaft.machine_mapping.camera_visible import (
        vfit_camera_visible_dynamic,
        vfit_camera_visible_static,
    )

    ods = copy.deepcopy(sample)
    frames = vaft.data.sample_camera_visible_frame_paths(39915)[:4]
    images = []
    for _, path in frames:
        image = mpimg.imread(path)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        images.append((image * 255).astype(int))
    vfit_camera_visible_static(ods, lines_n=images[0].shape[0], columns_n=images[0].shape[1])
    vfit_camera_visible_dynamic(ods, images=images, times_s=[time for time, _ in frames])
    return ods


def _vacuum_lines(axes):
    return [line for line in axes.lines if line.get_label().startswith("vacuum field line")]


def test_vacuum_field_lines_land_on_the_frame(camera_sample):
    onset = vomas.find_breakdown_onset(camera_sample)
    figure, axes = vomas.plot_camera_visible_image_vacuum_field_line(
        camera_sample, shot=39915, time=onset, ec_frequency_Hz=2.45e9, max_turns=1.0,
    )
    lines = _vacuum_lines(axes)
    assert len(lines) == 2  # (0.4, 0) and the ECR radius
    height, width = axes.get_images()[0].get_array().shape
    for line in lines:
        xy = line.get_xydata()
        finite = xy[np.isfinite(xy).all(axis=1)]
        assert finite.shape[0] >= 2
        assert (finite[:, 0] >= 0).all() and (finite[:, 0] < width).all()
        assert (finite[:, 1] >= 0).all() and (finite[:, 1] < height).all()
    plt.close(figure)


def test_the_overlay_is_also_a_name_on_the_general_camera_plot(camera_sample):
    model = vomas.extract_camera_visible_image(
        camera_sample, shot=39915, frame_index=1, overlay="vacuum_field_line",
        seeds=[(0.45, 0.0)], max_turns=0.5,
    )
    labels = [layer.label for layer in model.overlays]
    assert labels == ["vacuum field line R=0.450 m, Z=0.000 m"]
    assert "equilibrium" not in model.title


def test_the_overlay_never_reads_an_equilibrium(camera_sample):
    without = copy.deepcopy(camera_sample)
    del without["equilibrium"]
    with_eq = vomas.extract_camera_visible_image(
        camera_sample, shot=39915, frame_index=1, overlay="vacuum_field_line", max_turns=0.5,
    )
    no_eq = vomas.extract_camera_visible_image(
        without, shot=39915, frame_index=1, overlay="vacuum_field_line", max_turns=0.5,
    )
    np.testing.assert_allclose(with_eq.overlays[0].r, no_eq.overlays[0].r, equal_nan=True)


def test_the_overlay_without_a_toroidal_field_says_so():
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["camera_visible.channel.0.name"] = "Fast Camera"
    ods["camera_visible.channel.0.detector.0.lines_n"] = 8
    ods["camera_visible.channel.0.detector.0.columns_n"] = 8
    for index in range(2):
        ods[f"camera_visible.channel.0.detector.0.frame.{index}.image_raw"] = np.zeros((8, 8), dtype=int)
        ods[f"camera_visible.channel.0.detector.0.frame.{index}.time"] = 0.3 + 1e-3 * index
    with pytest.raises(ValueError, match="tf.b_field_tor_vacuum_r"):
        vomas.extract_camera_visible_image(ods, frame_index=0, overlay="vacuum_field_line")
