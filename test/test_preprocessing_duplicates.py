"""Small preprocessing duplicates and their canonical owners (#2162 B3.2-B3.7).

Each duplicate is held to a frozen copy of what it computed on develop at
75b154e31 (after #2135), so the check does not depend on the code it checks:

* the Mirnov band-pass design (``process.magnetics._firwin_order`` + ``firwin``)
  -> :func:`~vaft.process.signal_processing.fir_filter_coefficients`;
* the interferometer-spectrogram Butterworth in the plot recipes
  -> :func:`~vaft.process.signal_processing.butterworth_highpass`;
* the diamagnetic Rogowski ``integrate.cumulative_trapezoid`` calls
  -> :func:`vaft.compat.cumtrapz_compat`.

Two candidates are *kept*, and the tests at the end say why:

* ``vaft.code.efit.legacy.smooth`` is not equivalent to
  :func:`~vaft.process.signal_processing.smooth`: a NaN stays local in the
  loop version and taints the rest of the record in the cumulative-sum one;
* ``vaft.plot.soft_x_rays._baseline_correct`` subtracts a ``nanmean`` over a
  closed time range with no fallback, which is not
  :func:`~vaft.process.soft_x_rays.sxr_baseline_correction`'s tail mean with a
  last-1000-sample fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy import signal as scipy_signal

from vaft.compat import cumtrapz_compat
from vaft.process import magnetics as process_magnetics

def _rng():
    """A fresh generator per test, so xdist scheduling cannot change the data."""
    return np.random.default_rng(2162)


# ---------------------------------------------------------------------------
# Mirnov band-pass design
# ---------------------------------------------------------------------------

def _frozen_firwin_order(sample_rate):
    order = int(float(sample_rate) * 1e-3)
    return order + 1 if order % 2 == 0 else order


def _frozen_mirnov_preprocess_signal(
    data, *, sample_rate=250_000.0, high_pass_cutoff=2_000.0, low_pass_cutoff=90_000.0,
    amplifier_gain=1.0, filter_order=None,
):
    from vaft.process.signal_processing import apply_fir_filter

    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return values.copy()
    if amplifier_gain == 0:
        raise ValueError("amplifier_gain must be non-zero.")
    processed = values - float(np.nanmean(values))
    order = int(filter_order) if filter_order is not None else _frozen_firwin_order(sample_rate)
    nyquist = 0.5 * float(sample_rate)
    for cutoff, pass_zero in ((high_pass_cutoff, False), (low_pass_cutoff, True)):
        if cutoff is None:
            continue
        cutoff = float(cutoff)
        if cutoff <= 0 or cutoff >= nyquist:
            continue
        taps = scipy_signal.firwin(order, cutoff, fs=sample_rate, pass_zero=pass_zero, window="hann")
        processed = apply_fir_filter(taps, processed, short_record="causal")
    return processed / float(amplifier_gain)


@pytest.mark.parametrize("sample_rate", [25_000.0, 250_000.0, 1_000_000.0, 2_000_000.0])
@pytest.mark.parametrize(("kind", "pass_zero"), [("highpass", False), ("lowpass", True)])
def test_mirnov_derived_design_is_fir_filter_coefficients(sample_rate, kind, pass_zero):
    from vaft.process.signal_processing import fir_filter_coefficients

    cutoff = 0.1 * sample_rate
    frozen = scipy_signal.firwin(
        _frozen_firwin_order(sample_rate), cutoff, fs=sample_rate, pass_zero=pass_zero, window="hann"
    )
    canonical = fir_filter_coefficients(cutoff, sample_rate, kind=kind, window="hann")
    np.testing.assert_array_equal(canonical, frozen)


@pytest.mark.parametrize(
    ("sample_rate", "kwargs", "length"),
    [
        (250_000.0, {}, 20_000),
        (1_000_000.0, {"amplifier_gain": 0.004529}, 20_000),
        (250_000.0, {"high_pass_cutoff": None}, 20_000),
        (250_000.0, {"filter_order": 101}, 20_000),
        # Short record: the causal fallback of apply_fir_filter.
        (250_000.0, {}, 400),
        # A band edge at or past Nyquist is skipped, not raised.
        (25_000.0, {}, 5_000),
    ],
)
def test_mirnov_preprocess_signal_matches_frozen_reference(sample_rate, kwargs, length):
    data = 0.3 + _rng().standard_normal(length)
    expected = _frozen_mirnov_preprocess_signal(data, sample_rate=sample_rate, **kwargs)
    actual = process_magnetics.mirnov_preprocess_signal(data, sample_rate=sample_rate, **kwargs)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Interferometer-spectrogram high-pass
# ---------------------------------------------------------------------------

def _frozen_recipe_highpass(signal_values, sample_rate, highpass_cutoff=200.0):
    nyquist = sample_rate / 2.0
    fluctuation = signal_values
    if signal_values.size > 32 and 0.0 < highpass_cutoff < nyquist:
        b, a = scipy_signal.butter(4, highpass_cutoff / nyquist, btype="highpass")
        fluctuation = scipy_signal.filtfilt(b, a, signal_values)
    return fluctuation


def _interferometer_ods(time, values):
    from omas import ODS

    ods = ODS()
    ods["interferometer.channel.0.n_e_line.data"] = values
    ods["interferometer.channel.0.n_e_line.time"] = time
    ods["interferometer.time"] = time
    return ods


@pytest.mark.parametrize(
    ("length", "options"),
    [(4_000, {}), (4_000, {"highpass_cutoff": 1_500.0}), (20, {}), (4_000, {"highpass_cutoff": 0.0})],
)
def test_interferometer_spectrogram_fluctuation_matches_frozen_reference(monkeypatch, length, options):
    from vaft.plot.backend import recipes

    time = np.arange(length) * 1e-5  # 100 kHz
    values = 1e19 * (1.0 + 0.5 * time / time[-1]) + 1e17 * np.sin(2 * np.pi * 5e3 * time)
    values = values + 1e16 * _rng().standard_normal(length)
    captured = {}

    def capture(time_, values_, **kwargs):
        captured["fluctuation"] = np.asarray(values_)
        raise _Captured

    monkeypatch.setattr(recipes, "_spectrogram_result", capture)
    with pytest.raises(_Captured):
        recipes._build_interferometer_spectrogram(_interferometer_ods(time, values), **options)
    steps = np.diff(time)
    sample_rate = 1.0 / float(np.median(steps[steps > 0]))  # as the recipe derives it
    expected = _frozen_recipe_highpass(values, sample_rate, **options)
    np.testing.assert_array_equal(captured["fluctuation"], expected)


class _Captured(Exception):
    pass


# ---------------------------------------------------------------------------
# Rogowski / diamagnetic integration
# ---------------------------------------------------------------------------

def test_cumtrapz_compat_is_cumulative_trapezoid_on_the_diamagnetic_call_shape():
    time = np.cumsum(np.full(25_000, 4e-5))
    raw = np.sin(2 * np.pi * 300 * time) + 0.01 * _rng().standard_normal(time.size)
    expected = integrate.cumulative_trapezoid(raw, time, initial=0.0)
    np.testing.assert_array_equal(cumtrapz_compat(raw, x=time, initial=0.0), expected)
    twice = integrate.cumulative_trapezoid(expected, time, initial=0.0)
    np.testing.assert_array_equal(cumtrapz_compat(expected, x=time, initial=0.0), twice)


# ---------------------------------------------------------------------------
# Kept, with the reason
# ---------------------------------------------------------------------------

def test_legacy_smooth_is_kept_because_its_nan_handling_differs():
    from vaft.code.efit.legacy import smooth as legacy_smooth
    from vaft.process.signal_processing import smooth

    finite = 1.0e5 + 1.0e3 * _rng().standard_normal(20_000)
    # On finite data the two agree to summation-order round-off only: a
    # per-window sum against a difference of two cumulative sums, each about
    # N * 1e5 = 2e9 here, so the bound is eps * 2e9 / 1e5 ~ 5e-12 relative.
    np.testing.assert_allclose(legacy_smooth(finite, 10), smooth(finite, 10), rtol=1e-11, atol=0.0)

    gappy = finite.copy()
    gappy[100] = np.nan
    # The loop version blanks only the windows that contain the NaN ...
    assert int(np.isnan(legacy_smooth(gappy, 10)).sum()) == 9
    # ... which a cumulative-sum version cannot do, so replacing it would
    # change the deprecated H-alpha detector on any record with a gap.
    assert int(np.isnan(smooth(gappy, 10)).sum()) > 9


def test_plot_sxr_baseline_is_a_closed_range_nanmean_without_fallback():
    from vaft.plot.soft_x_rays import _baseline_correct

    time = np.linspace(0.0, 1.0, 101)
    signal_ = 2.0 + time
    signal_[5] = np.nan
    corrected = _baseline_correct(time, signal_, (0.0, 0.1))
    # Mean over t in [0, 0.1] inclusive, NaN skipped: (2.00 + ... + 2.10 - 2.05) / 10.
    expected_offset = np.nanmean(signal_[:11])
    np.testing.assert_allclose(corrected, signal_ - expected_offset, rtol=0, atol=1e-15)
    # A range with no sample leaves the signal alone (no tail fallback).
    np.testing.assert_array_equal(_baseline_correct(time, signal_, (2.0, 3.0)), signal_)


# ---------------------------------------------------------------------------
# The new canonical high-pass, independently of the recipe
# ---------------------------------------------------------------------------

def test_butterworth_highpass_is_the_frozen_recipe_filter():
    from vaft.process.signal_processing import butterworth_highpass

    values = np.cumsum(_rng().standard_normal(5_000))
    np.testing.assert_array_equal(
        butterworth_highpass(values, 200.0, 1e5, order=4, zero_phase=True),
        _frozen_recipe_highpass(values, 1e5),
    )


def test_butterworth_highpass_magnitude_and_zero_phase():
    from vaft.process.signal_processing import butterworth_highpass

    fs, cutoff = 1e5, 1e3
    time = np.arange(40_000) / fs
    centre = slice(10_000, 30_000)  # away from filtfilt's edge transients
    # A constant is removed entirely.
    assert np.max(np.abs(butterworth_highpass(np.full(time.size, 3.0), cutoff, fs)[centre])) < 1e-9
    # A tone a decade above the cutoff passes with the two-pass gain |H|^2 of a
    # bilinear-transform Butterworth, 1 / (1 + (tan(pi fc/fs) / tan(pi f/fs))^(2n)),
    # and no phase shift.
    for order in (2, 4):
        f = 10 * cutoff
        tone = np.sin(2 * np.pi * f * time)
        out = butterworth_highpass(tone, cutoff, fs, order=order)
        ratio = np.tan(np.pi * cutoff / fs) / np.tan(np.pi * f / fs)
        expected_gain = 1.0 / (1.0 + ratio ** (2 * order))
        fit = np.linalg.lstsq(
            np.column_stack((np.sin(2 * np.pi * f * time[centre]), np.cos(2 * np.pi * f * time[centre]))),
            out[centre], rcond=None,
        )[0]
        assert abs(np.hypot(*fit) - expected_gain) < 1e-6
        assert abs(np.arctan2(fit[1], fit[0])) < 1e-6
    # At the cutoff the two-pass response is -6 dB.
    tone = np.sin(2 * np.pi * cutoff * time)
    out = butterworth_highpass(tone, cutoff, fs, order=2)
    assert abs(np.max(np.abs(out[centre])) - 0.5) < 1e-3


def test_butterworth_highpass_rejects_a_cutoff_outside_the_band():
    from vaft.process.signal_processing import butterworth_highpass

    with pytest.raises(ValueError, match="cutoff must lie"):
        butterworth_highpass(np.zeros(100), 6e4, 1e5)


def test_mirnov_refuses_an_even_filter_order():
    with pytest.raises(ValueError, match="filter_order must be odd"):
        process_magnetics.mirnov_preprocess_signal(np.ones(5_000), filter_order=250)


def test_no_private_mirnov_designer_remains():
    assert not hasattr(process_magnetics, "_firwin_order")
