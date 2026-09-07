"""Issue #485: the toroidal phase of a fluctuation band, read around the torus.

The measurement is a phase per probe at one instant; the fit is a line of
slope ``-n`` through those points.  The plot draws the measurement always
and the fit only when asked, states how many *distinct* toroidal positions
stood behind the number, and refuses to advertise itself at all on an input
whose array cannot support the fit -- a requirement no path can express.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model, missing_required_path

NAME = "mirnov_spatial_phase"
FIT_KWARGS = dict(
    frequencies=[8_000.0], candidate_n=range(0, 5), window_size=512,
    preprocess=False, time=0.020,
)


def _phase_ods():
    """Four probes at 0/120/180/240 degrees carrying one n=2 band."""
    from omas import ODS

    sample_rate = 100_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    angles = np.deg2rad([0.0, 120.0, 180.0, 240.0])
    ods = ODS()
    for index, angle in enumerate(angles):
        data = np.sin(2.0 * np.pi * 8_000.0 * time + (0.2 - 2 * angle))
        probe = f"magnetics.b_field_pol_probe.{index}"
        ods[f"{probe}.name"] = f"TOR{index}"
        ods[f"{probe}.toroidal_angle"] = float(angle)
        ods[f"{probe}.voltage.time"] = time
        ods[f"{probe}.voltage.data"] = data
    ods["dataset_description.data_entry.pulse"] = 99999
    return ods


@pytest.fixture(scope="module")
def phase_ods():
    return _phase_ods()


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


def _model(ods, **options):
    return build_model(NAME, normalize_entries(ods), **{**FIT_KWARGS, **options})


# ---------------------------------------------------------------------------
# what it draws
# ---------------------------------------------------------------------------

def test_the_measured_phases_sit_on_the_stored_angles_and_recover_the_mode(phase_ods):
    model = _model(phase_ods)
    measured = [s for s in model.series if s.role != "fit"]
    assert len(measured) == 1, "one band was asked for"
    band = measured[0]
    np.testing.assert_allclose(np.sort(band.x), [0.0, 120.0, 180.0, 240.0], atol=1e-6)
    assert band.label.endswith("n=2"), band.label
    assert band.label.startswith("8.0 kHz")
    assert band.style["linestyle"] == "none" and band.style["marker"] == "o"
    assert np.all(np.abs(band.y) <= 180.0 + 1e-9)


def test_the_fit_is_drawn_only_when_asked(phase_ods):
    points_only = _model(phase_ods, show_fit=False)
    assert len(points_only.series) == 1
    assert all(s.role != "fit" for s in points_only.series)

    with_fit = _model(phase_ods, show_fit=True)
    fits = [s for s in with_fit.series if s.role == "fit"]
    assert len(fits) == 1
    assert fits[0].style["linestyle"] == "--"
    measured = [s for s in with_fit.series if s.role != "fit"][0]
    assert fits[0].channel == measured.channel
    assert fits[0].style["color"] == measured.style["color"], "a band and its fit share a colour"
    assert fits[0].x.size > 100, "the fitted line is drawn on a dense angle grid"


def test_the_fitted_line_breaks_where_the_phase_wraps(phase_ods):
    fit = next(s for s in _model(phase_ods, show_fit=True).series if s.role == "fit")
    breaks = np.isnan(fit.y)
    assert breaks.any(), "a wrapped phase must not be joined across +-180 deg"
    # A break is a gap in the line, so both coordinates carry it.
    np.testing.assert_array_equal(breaks, np.isnan(fit.x))
    segments = np.split(fit.y, np.flatnonzero(breaks))
    for segment in segments:
        finite = segment[~np.isnan(segment)]
        if finite.size > 1:
            assert np.max(np.abs(np.diff(finite))) <= 180.0


def test_the_title_says_how_many_positions_stood_behind_the_number(phase_ods):
    title = _model(phase_ods).title
    assert "4 toroidal positions" in title
    assert "ms" in title and "Toroidal mode phase" in title


def test_the_axes_are_the_torus_and_a_phase(phase_ods):
    model = _model(phase_ods)
    assert model.coordinate_label == "Toroidal angle phi [deg]"
    assert model.y_unit == "deg" and model.x_limits == (0.0, 360.0)


def test_time_snaps_to_a_stored_sample(phase_ods):
    early = _model(phase_ods, time=0.005).title
    late = _model(phase_ods, time=0.030).title
    assert early != late
    assert "5." in early.split("t = ")[1] and "30." in late.split("t = ")[1]


# ---------------------------------------------------------------------------
# availability: a data condition, not a path
# ---------------------------------------------------------------------------

def test_one_toroidal_position_cannot_support_a_fit():
    """A requirement no path expresses: the probes must be spread around phi."""
    stacked = _phase_ods()
    for index in range(4):
        stacked[f"magnetics.b_field_pol_probe.{index}.toroidal_angle"] = 0.0
    reason = missing_required_path(stacked, NAME)
    assert reason and "distinct toroidal angles" in reason, reason
    assert NAME not in {record.name for record in vaft.omas.available_plots(stacked)}
    with pytest.raises(ValueError, match="toroidal"):
        vaft.omas.plot_mirnov_spatial_phase(stacked, show=False)


def test_the_packaged_shot_says_how_thin_its_array_is(sample):
    """Two positions do fit, and the title must not hide what that means."""
    from vaft.plot.backend.recipes import _toroidal_phase_channels

    _, angles = _toroidal_phase_channels(sample)
    assert np.unique(np.round(np.degrees(angles), 3)).size == 2
    assert missing_required_path(sample, NAME) is None
    model = build_model(NAME, normalize_entries(sample), time=0.30, window_size=512)
    assert "2 toroidal positions" in model.title


def test_the_predicate_passes_an_array_that_can(phase_ods):
    assert missing_required_path(phase_ods, NAME) is None
    assert NAME in {record.name for record in vaft.omas.available_plots(phase_ods)}


# ---------------------------------------------------------------------------
# discovery, controls and the adapter surfaces
# ---------------------------------------------------------------------------

def test_discovery_names_the_analysis_and_its_parameters(phase_ods):
    record = next(r for r in vaft.omas.available_plots(phase_ods, detail=True) if r.name == NAME)
    assert record.analysis["default"] == "wrapped n fit"
    assert "show_fit" in record.analysis["methods"]["wrapped n fit"]
    assert record.view == "spatial" and record.subject == "mirnov"


def test_the_fit_is_a_toggle_the_control_layer_offers(phase_ods):
    from vaft.plot.controls import controls_for
    from vaft.plot.navigation import ControlState

    record = next(r for r in vaft.omas.available_plots(phase_ods, detail=True) if r.name == NAME)
    toggle = next(c for c in controls_for(record) if c.name == "show_fit")
    assert toggle.kind == "toggle" and toggle.default is True
    state = ControlState(controls_for(record))
    state.set("show_fit", False)
    assert state.as_options()["show_fit"] is False


def test_the_legacy_name_now_points_here():
    from vaft.plot._migration import DEPRECATED

    assert DEPRECATED["toroidal_phase_mode_fit"] == NAME
    # The n(f)/coherence view is a different measurement and keeps its target.
    assert DEPRECATED["toroidal_mode_spectrum"] == "mirnov_spectrogram"


def test_both_renderers_draw_it(phase_ods):
    fig, ax = vaft.omas.plot_mirnov_spatial_phase(phase_ods, show=False, **FIT_KWARGS)
    assert len(ax.lines) == 2, "one measured band and its fit"
    plt.close(fig)
    figure = vaft.omas.plot_mirnov_spatial_phase(
        phase_ods, show=False, backend="plotly", **FIT_KWARGS
    )
    assert len(figure.data) == 2


def test_omas_and_imas_agree(sample):
    """The same measurement whichever data model carries the probes."""
    imas = pytest.importorskip("imas")
    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    options = dict(time=0.30, window_size=512)
    from vaft.imas.entries import normalize_entries as imas_entries

    left = build_model(NAME, normalize_entries(sample), **options)
    right = build_model(NAME, imas_entries(entry), **options)
    assert len(left.series) == len(right.series)
    for a, b in zip(left.series, right.series):
        assert a.label == b.label
        np.testing.assert_allclose(a.x, b.x, equal_nan=True)
        np.testing.assert_allclose(a.y, b.y, equal_nan=True)


# ---------------------------------------------------------------------------
# what the numbers are allowed to claim
# ---------------------------------------------------------------------------

def test_a_number_two_positions_cannot_settle_is_labelled_as_a_family(sample):
    """Angles that are all multiples of one spacing alias n by 360/spacing."""
    from vaft.plot.backend.recipes import _toroidal_alias_period

    assert _toroidal_alias_period(np.array([0.0, 240.0])) == 3
    assert _toroidal_alias_period(np.array([0.0, 120.0, 180.0, 240.0])) == 6
    assert _toroidal_alias_period(np.array([0.0])) is None

    model = build_model(NAME, normalize_entries(sample), time=0.30, window_size=512)
    # 39915 sits at 0 and 240 degrees, so its default candidates -6..6 hold
    # five members of one family and the label must not pick one silently.
    assert all("(mod 3)" in s.label for s in model.series), [s.label for s in model.series]


def test_a_number_the_array_can_settle_carries_no_family(phase_ods):
    """Four positions with candidates 0..4 leave no alias in the set."""
    label = _model(phase_ods).series[0].label
    assert label.endswith("n=2") and "mod" not in label


def test_the_title_reports_the_probes_behind_the_positions(sample):
    title = build_model(NAME, normalize_entries(sample), time=0.30, window_size=512).title
    assert "2 toroidal positions" in title
    assert "probes at" in title
    # Probes on another timebase are excluded, and saying so is part of the count.
    assert "left out" in title


def test_probes_a_hair_apart_are_one_position():
    """Stored angles carry rounding noise; a microdegree is not a baseline."""
    from vaft.plot.backend.recipes import _distinct_toroidal_angles

    noisy = _phase_ods()
    for index in range(4):
        noisy[f"magnetics.b_field_pol_probe.{index}.toroidal_angle"] = 1e-9 * index
    _, angles = __import__("vaft").plot.backend.recipes._toroidal_phase_channels(noisy)
    assert _distinct_toroidal_angles(angles).size == 1
    assert missing_required_path(noisy, NAME) is not None


# ---------------------------------------------------------------------------
# what it refuses, and how it says so
# ---------------------------------------------------------------------------

def test_probes_on_another_timebase_are_answered_before_the_plot_is_offered(phase_ods):
    """Availability and buildability must agree about the timebase."""
    import copy as _copy

    mixed = _copy.deepcopy(phase_ods)
    for index in (1, 2, 3):
        probe = f"magnetics.b_field_pol_probe.{index}"
        mixed[f"{probe}.voltage.time"] = np.asarray(mixed[f"{probe}.voltage.time"])[:100]
        mixed[f"{probe}.voltage.data"] = np.asarray(mixed[f"{probe}.voltage.data"])[:100]
    reason = missing_required_path(mixed, NAME)
    assert reason and "timebase" in reason, reason
    assert NAME not in {record.name for record in vaft.omas.available_plots(mixed)}


def test_a_failing_predicate_is_reported_as_a_failure(phase_ods):
    """A broken check is not the same fact as an unsupported input."""
    import dataclasses

    from vaft.plot.backend import recipes

    def explode(_ods):
        raise RuntimeError("connection lost")

    broken = dataclasses.replace(recipes.RECIPES[NAME], available=explode)
    reason = recipes._unmet_data_condition(broken, NAME, phase_ods)
    assert "RuntimeError" in reason and "connection lost" in reason


def test_named_channels_are_honoured_and_unknown_ones_refused(phase_ods):
    model = _model(phase_ods, channels=[0, 2])
    measured = [s for s in model.series if s.role != "fit"][0]
    np.testing.assert_allclose(np.sort(measured.x), [0.0, 180.0], atol=1e-6)
    assert "2 probes at 2 toroidal positions" in model.title
    with pytest.raises(ValueError, match="carry no toroidal angle"):
        _model(phase_ods, channels=[0, 99])


def test_each_band_is_paired_with_its_own_fit_by_colour(sample):
    """Two bands, so a fit taking the wrong band's colour would show."""
    model = build_model(NAME, normalize_entries(sample), time=0.30, window_size=512)
    measured = [s for s in model.series if s.role != "fit"]
    fits = [s for s in model.series if s.role == "fit"]
    assert len(measured) == 2 and len(fits) == 2
    assert measured[0].style["color"] != measured[1].style["color"]
    for band, fit in zip(measured, fits):
        assert fit.style["color"] == band.style["color"]
        assert fit.channel == band.channel
