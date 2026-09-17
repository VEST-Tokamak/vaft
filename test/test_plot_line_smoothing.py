"""``smooth=``: a NaN-aware rolling median on every line plot (issue #888).

The ECH launched power is forward minus reflected from two noisy detectors,
so it is read through a median; the option is generic, on every LineRecipe
plot and every line member of a composite.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

import vaft.omas as vomas
from vaft.plot.backend import recipes

DT = 1e-4


def _flux_loop_ods(values: np.ndarray) -> ODS:
    ods = ODS(consistency_check=False)
    ods["magnetics.flux_loop.0.name"] = "FL0"
    ods["magnetics.flux_loop.0.voltage.data"] = values
    ods["magnetics.flux_loop.0.voltage.time"] = np.arange(values.size) * DT
    return ods


def _signal():
    rng = np.random.default_rng(3)
    values = np.sin(np.linspace(0.0, 3.0, 400)) + 0.01 * rng.standard_normal(400)
    values[[50, 180, 300]] += 25.0  # spikes
    values[220:230] = np.nan  # a detector out of range
    return values


def test_spikes_are_removed_and_gaps_kept():
    values = _signal()
    model = vomas.extract_flux_loop_time_voltage(_flux_loop_ods(values), smooth=10 * DT)
    (trace,) = model.series
    y = np.asarray(trace.y)
    assert np.nanmax(np.abs(y)) < 1.5, "a spike survived the median"
    assert np.isnan(y[220:230]).all(), "the gap was filled from its neighbours"
    assert np.isfinite(y[:220]).all() and np.isfinite(y[230:]).all()
    assert "(median 1 ms)" in model.title


def test_a_window_shorter_than_the_sample_spacing_is_a_no_op():
    values = _signal()
    raw = vomas.extract_flux_loop_time_voltage(_flux_loop_ods(values))
    same = vomas.extract_flux_loop_time_voltage(_flux_loop_ods(values), smooth=DT / 4)
    np.testing.assert_array_equal(np.asarray(raw.series[0].y), np.asarray(same.series[0].y))


@pytest.mark.parametrize("bad", [0.0, -1e-3, float("nan"), "1ms", True])
def test_invalid_windows_are_refused(bad):
    with pytest.raises(ValueError, match="smooth"):
        vomas.extract_flux_loop_time_voltage(_flux_loop_ods(_signal()), smooth=bad)


def test_the_window_needs_the_time_abscissa():
    with pytest.raises(ValueError, match="time abscissa"):
        vomas.extract_flux_loop_time_voltage(_flux_loop_ods(_signal()), smooth=1e-3, x="index")


def test_the_rolling_median_keeps_the_sample_count_and_ends():
    x = np.arange(20) * 0.1
    y = np.arange(20, dtype=float)
    smoothed = recipes._rolling_median(x, y, 0.3)
    assert smoothed.shape == y.shape
    np.testing.assert_allclose(smoothed[5:15], y[5:15])  # a ramp is its own median


def test_the_plot_draws_the_smoothed_trace():
    values = _signal()
    figure, axes = vomas.plot_flux_loop_time_voltage(_flux_loop_ods(values), smooth=1e-3)
    drawn = axes.lines[0].get_ydata()
    assert np.nanmax(np.abs(drawn)) < 1.5
    plt.close(figure)


def test_a_composite_hands_the_window_to_its_line_members():
    sample = vomas.sample_ods()
    raw = vomas.extract_diagnostics_overview(sample, members=["spectrometer_uv_time_intensity"])
    smoothed = vomas.extract_diagnostics_overview(sample, members=["spectrometer_uv_time_intensity"], smooth=2e-3)
    a, b = np.asarray(raw.models[0].series[0].y), np.asarray(smoothed.models[0].series[0].y)
    assert a.shape == b.shape and not np.allclose(a, b, equal_nan=True)
