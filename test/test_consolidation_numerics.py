"""Analytic references for the scientific preprocessing primitives (issue #2166, Scope B).

Every expected value here comes from closed-form mathematics, never from
calling the function under test on another input: the DC gain of a normalized
low-pass, the antisymmetry of a zero-phase step response, the frequency an
undersampled tone folds to, the Dirichlet kernel of a moving average,
``integral cos = sin``, a cubic spline's exactness on a cubic.  These tests pin
the *current* API so the consolidation (#2147, B3 #2162) has an independent
yardstick: a migrated implementation either still satisfies the mathematics or
the diff says which identity it broke.

The FIR section (``apply_fir_filter``, the VEST probe and IMPA low passes,
``fir_filter_coefficients``) is written against develop *after* #2135 made the
probe low pass zero-phase, so it tests the intended behaviour.  Its reference
designs are the window-method formula written out here (Hamming-windowed sinc,
normalized to unit DC gain), not ``scipy.signal.firwin``.

Tolerances are stated next to each assertion with the reason for their size.
Where a tolerance is "float round-off" it is a few hundred ulp of the signal
scale; where it is set by the filter it is derived from the window's
documented passband ripple / stopband attenuation, not from an observed run.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vaft.compat import cumtrapz_compat
from vaft.process.magnetics import (
    DegenerateBaselineWindowError,
    VestMagneticsProcessingConfig,
    _linear_baseline,
    vest_flux_loop_flux_from_voltage,
)
from vaft.process.impa import impa_lowpass
from vaft.process.magnetics import b_field_pol_probe_field, vest_probe_lowpass_taps
from vaft.process.signal_processing import (
    ResamplingError,
    apply_fir_filter,
    fir_filter_coefficients,
    SignalRepairError,
    anti_alias_filter,
    describe_time_grid,
    detect_clipped_samples,
    repair_clipped_interval,
    resample_to_time,
    smooth,
    subtract_baseline,
)

# VEST acquisition rates (vaft.database.raw FAST_DT / SLOW_DT): restated here
# as literals on purpose, so the test does not inherit a changed constant.
FAST_RATE = 250_000.0
SLOW_RATE = 25_000.0

# Hamming-window FIR: peak passband ripple ~0.0194 dB (|1 - gain| <= 2.3e-3)
# and minimum stopband attenuation ~53 dB (gain <= 2.3e-3) for a single pass
# (Oppenheim & Schafer, window-method table).  filtfilt applies the filter
# twice, squaring the magnitude response.
HAMMING_PASSBAND_DEVIATION_FILTFILT = 2 * 2.3e-3   # (1 +- d)^2 ~ 1 +- 2d
HAMMING_STOPBAND_GAIN_FILTFILT = (2.3e-3) ** 2     # ~ -106 dB


def _uniform_time(rate: float, duration: float, t0: float = 0.0) -> np.ndarray:
    n = int(round(duration * rate))
    return t0 + np.arange(n) / rate


def _tone_amplitude_phase(t: np.ndarray, y: np.ndarray, frequency: float) -> tuple[float, float]:
    """Least-squares amplitude and phase of ``A cos(2 pi f t + phi)`` in ``y``.

    Independent of the code under test: a 2-column linear regression.
    """
    design = np.column_stack(
        [np.cos(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t), np.ones_like(t)]
    )
    (a, b, _), *_ = np.linalg.lstsq(design, y, rcond=None)
    # y = a cos + b sin = A cos(wt + phi) with A cos phi = a, -A sin phi = b
    return float(math.hypot(a, b)), float(math.atan2(-b, a))


def _interior(n: int, margin: int) -> slice:
    return slice(margin, n - margin)


# ---------------------------------------------------------------------------
# describe_time_grid
# ---------------------------------------------------------------------------

def test_a_uniform_grid_reports_its_constructed_spacing():
    t = 0.1 + np.arange(1000) * 4e-6
    grid = describe_time_grid(t)
    assert grid.n == 1000
    assert grid.t0 == pytest.approx(0.1, abs=0)
    # arange * dt accumulates no error beyond one rounding per sample.
    assert grid.dt == pytest.approx(4e-6, rel=1e-9)
    assert grid.sample_rate == pytest.approx(250_000.0, rel=1e-9)
    assert grid.uniform and grid.strictly_increasing


def test_a_linspace_timebase_is_uniform_at_its_true_spacing():
    """Shots after 42190 store linspace(0, span, n): dt = span / (n - 1), not 4e-6."""
    n = 250_001
    t = np.linspace(0.0, 1.0, n)
    grid = describe_time_grid(t)
    assert grid.dt == pytest.approx(1.0 / (n - 1), rel=1e-9)
    assert grid.uniform


def test_an_irregular_grid_reports_the_jitter_it_was_built_with():
    steps = np.full(99, 1e-5)
    steps[40] = 1.02e-5  # one 2 % long interval
    t = np.concatenate(([0.0], np.cumsum(steps)))
    grid = describe_time_grid(t)
    # median step is the nominal 1e-5; jitter is |1.02e-5 - 1e-5| / 1e-5.
    assert grid.dt == pytest.approx(1e-5, rel=1e-9)
    assert grid.max_relative_jitter == pytest.approx(0.02, rel=1e-6)
    assert not grid.uniform
    assert grid.strictly_increasing


def test_a_non_monotone_grid_is_not_strictly_increasing():
    t = np.array([0.0, 1.0, 2.0, 1.5, 3.0])
    assert not describe_time_grid(t).strictly_increasing


def test_degenerate_grids_report_undefined_spacing():
    empty = describe_time_grid([])
    assert empty.n == 0 and math.isnan(empty.dt)
    single = describe_time_grid([0.3])
    assert single.n == 1 and math.isnan(single.dt) and math.isnan(single.sample_rate)


# ---------------------------------------------------------------------------
# anti_alias_filter: impulse, step, DC -- the zero-phase identities
# ---------------------------------------------------------------------------

CUTOFF = 0.8 * SLOW_RATE / 2  # resample_to_time's default for a 25 kHz target
STOPBAND = SLOW_RATE / 2


def test_the_impulse_response_is_symmetric_about_the_impulse():
    """Zero phase <=> real, even impulse response: no group delay at any frequency."""
    n = 8001
    centre = n // 2
    x = np.zeros(n)
    x[centre] = 1.0
    y = anti_alias_filter(x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND)
    k = np.arange(1, 1500)
    # Symmetry is exact in exact arithmetic; round-off of a few-hundred-tap
    # double convolution is ~1e-16 per tap.
    np.testing.assert_allclose(y[centre + k], y[centre - k], rtol=0, atol=1e-14)
    # Centroid of the response = the impulse instant (group delay 0 samples).
    idx = np.arange(n)
    centroid = float(np.sum(idx * y) / np.sum(y))
    assert centroid == pytest.approx(centre, abs=1e-9)


def test_the_dc_gain_is_exactly_one():
    """firwin normalizes the low-pass to unit DC gain; filtfilt squares it: 1."""
    n = 8001
    x = np.zeros(n)
    x[n // 2] = 1.0
    y = anti_alias_filter(x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND)
    assert float(np.sum(y)) == pytest.approx(1.0, abs=1e-12)

    constant = np.full(n, 3.25)
    out = anti_alias_filter(constant, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND)
    # filtfilt's odd-extension pad of a constant is the same constant, so even
    # the edges are exact up to round-off.
    np.testing.assert_allclose(out, 3.25, rtol=0, atol=1e-12)


def test_a_linear_ramp_passes_unchanged_including_the_edges():
    """An even impulse response with unit DC gain maps a + b n to itself; the
    odd-extension pad of a line is the same line."""
    t = _uniform_time(FAST_RATE, 0.02)
    ramp = 2.0 - 40.0 * t
    out = anti_alias_filter(ramp, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND)
    np.testing.assert_allclose(out, ramp, rtol=0, atol=1e-11)


def test_the_step_response_crosses_half_height_at_the_step_without_delay():
    """For an even response g with sum 1, y[c-1+k] + y[c-k] = 1 for a unit step at c."""
    n = 8000
    c = n // 2
    x = np.zeros(n)
    x[c:] = 1.0
    y = anti_alias_filter(x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND)
    k = np.arange(1, 1500)
    np.testing.assert_allclose(y[c - 1 + k] + y[c - k], 1.0, rtol=0, atol=1e-12)
    # Far from the step the response has settled to the input.
    assert y[c - 1500] == pytest.approx(0.0, abs=1e-6)
    assert y[c + 1500] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("frequency", "expected_gain", "tolerance"),
    [
        # passband: Hamming ripple, applied twice
        (2_000.0, 1.0, HAMMING_PASSBAND_DEVIATION_FILTFILT),
        # at its cutoff a window-method low-pass sits at 1/2 +- delta per pass
        # (the window smears the ideal step symmetrically), so filtfilt gives
        # (1/2 +- delta)^2 = 1/4 +- delta; 2 delta is the stated tolerance.
        (CUTOFF, 0.25, HAMMING_PASSBAND_DEVIATION_FILTFILT),
        # stopband (past the 12.5 kHz edge): attenuation of two passes.  The
        # factor 10 is margin for the 3.3 fs / df tap rule only approximating
        # the Hamming design table at this transition width.
        (20_000.0, 0.0, 10 * HAMMING_STOPBAND_GAIN_FILTFILT),
    ],
)
def test_a_sinusoid_is_scaled_by_the_design_response_with_zero_phase(frequency, expected_gain, tolerance):
    t = _uniform_time(FAST_RATE, 0.04)
    phase0 = 0.7
    x = np.cos(2 * np.pi * frequency * t + phase0)
    y = anti_alias_filter(x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND)
    keep = _interior(t.size, 2000)  # beyond the edge transient (filter ~340 taps)
    amplitude, phase = _tone_amplitude_phase(t[keep], y[keep], frequency)
    assert amplitude == pytest.approx(expected_gain, abs=tolerance)
    if expected_gain > 0.1:
        # Zero phase: the fitted phase is the input's; 1e-6 rad is ~1e-10 s
        # at 2 kHz, far below any physical timing requirement, far above
        # round-off.
        assert phase == pytest.approx(phase0, abs=1e-6)


def test_an_out_of_band_cutoff_is_refused():
    with pytest.raises(ResamplingError):
        anti_alias_filter(np.zeros(5000), source_rate=FAST_RATE, cutoff_hz=FAST_RATE)
    with pytest.raises(ResamplingError):
        anti_alias_filter(np.zeros(5000), source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=CUTOFF / 2)


def test_a_short_record_is_returned_unfiltered_and_marked():
    """Below 3 * numtaps + 1 samples filtfilt cannot run; the fallback is a
    passthrough (no causal substitute, so no phase delay) that says so."""
    x = np.cos(2 * np.pi * 20_000.0 * _uniform_time(FAST_RATE, 0.0008))  # 200 samples
    with pytest.warns(RuntimeWarning, match="shorter than"):
        y, mask = anti_alias_filter(
            x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND, return_mask=True
        )
    assert np.array_equal(y, x)
    assert not mask.any()


def test_nan_segments_are_filtered_independently_and_stay_nan():
    t = _uniform_time(FAST_RATE, 0.04)
    x = 1.5 + 0.0 * t
    gap = slice(4000, 4100)
    x[gap] = np.nan
    x[8000:8010] = np.nan  # a second gap; every finite run stays long enough to filter
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # every finite run here is long enough
        y, mask = anti_alias_filter(
            x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND, return_mask=True
        )
    assert np.isnan(y[gap]).all() and np.isnan(y[8000:8010]).all()
    finite = np.isfinite(x)
    # A constant run stays the constant (unit DC gain, constant pad) right up
    # to the gap: no NaN smear, no edge ringing.
    np.testing.assert_allclose(y[finite], 1.5, rtol=0, atol=1e-12)
    assert np.array_equal(mask, finite)


def test_a_short_finite_run_between_gaps_is_left_alone_and_unmarked():
    x = np.ones(6000)
    x[3000] = np.nan
    x[3051] = np.nan  # a 50-sample finite run, far shorter than 3 * numtaps + 1
    x[3001:3051] = 7.0
    with pytest.warns(RuntimeWarning, match="finite run"):
        y, mask = anti_alias_filter(
            x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, stopband_hz=STOPBAND, return_mask=True
        )
    assert np.array_equal(y[3001:3051], x[3001:3051])
    assert not mask[3001:3051].any()
    assert mask[:3000].all() and mask[3052:].all()


def test_nan_policy_error_refuses_a_non_finite_sample():
    x = np.ones(5000)
    x[10] = np.nan
    with pytest.raises(ResamplingError):
        anti_alias_filter(x, source_rate=FAST_RATE, cutoff_hz=CUTOFF, nan_policy="error")


# ---------------------------------------------------------------------------
# resample_to_time: identity, interpolation, decimation, aliasing
# ---------------------------------------------------------------------------

def test_same_grid_resampling_is_the_identity():
    t = _uniform_time(FAST_RATE, 0.01)
    x = np.sin(2 * np.pi * 3_000.0 * t) + 0.1 * t
    assert np.array_equal(resample_to_time(t, x, t), x)


def test_upsampling_a_line_is_exact_and_unfiltered():
    """Linear interpolation is exact on a line; a rate *increase* needs no filter."""
    source = _uniform_time(SLOW_RATE, 0.01)
    target = _uniform_time(FAST_RATE, 0.0099)
    line = 0.5 + 12.0 * source
    out = resample_to_time(source, line, target)
    np.testing.assert_allclose(out, 0.5 + 12.0 * target, rtol=0, atol=1e-13)


def test_decimation_preserves_an_in_band_tone_with_zero_phase():
    source = _uniform_time(FAST_RATE, 0.04)
    target = source[::10]  # aligned 25 kHz grid: no interpolation error
    phase0 = -1.1
    x = np.cos(2 * np.pi * 2_000.0 * source + phase0)
    y = resample_to_time(source, x, target)
    keep = _interior(target.size, 200)
    amplitude, phase = _tone_amplitude_phase(target[keep], y[keep], 2_000.0)
    assert amplitude == pytest.approx(1.0, abs=HAMMING_PASSBAND_DEVIATION_FILTFILT)
    assert phase == pytest.approx(phase0, abs=1e-6)


def _alias_amplitude(source_rate: float, target_rate: float, tone: float, **kwargs) -> tuple[float, float]:
    """Amplitude a tone above the target Nyquist leaves at its alias frequency."""
    source = _uniform_time(source_rate, 0.08)
    step = int(round(source_rate / target_rate))
    target = source[::step]
    x = np.cos(2 * np.pi * tone * source)
    y = resample_to_time(source, x, target, **kwargs)
    # Sampling cos(2 pi f t) at f_s folds f to |f - round(f / f_s) f_s|.
    alias = abs(tone - round(tone / target_rate) * target_rate)
    keep = _interior(target.size, 200)
    amplitude, _ = _tone_amplitude_phase(target[keep], y[keep], alias)
    return amplitude, alias


def test_bare_interpolation_folds_an_above_nyquist_tone_at_full_amplitude():
    """The negative control for the next test: without the filter a 20 kHz tone
    sampled at 25 kHz *is* a 5 kHz tone of amplitude 1 -- exactly, by
    trigonometry, since the target instants are source samples."""
    amplitude, alias = _alias_amplitude(FAST_RATE, SLOW_RATE, 20_000.0, anti_alias=False)
    assert alias == pytest.approx(5_000.0)
    assert amplitude == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("source_rate", [FAST_RATE, 125_000.0])
def test_decimation_suppresses_the_alias_of_an_above_nyquist_tone(source_rate):
    """With the default filter the folded 5 kHz content is at the stopband
    level of two Hamming passes, and an 8 kHz tone (inside the 10 kHz passband)
    survives at unit gain.  The second source rate checks that the filter is
    designed from the *measured* timebase: a filter sized for an assumed
    250 kHz would put its real cutoff at 5 kHz on 125 kHz data -- the alias
    check alone would not see that (it suppresses even harder), the in-band
    tone does."""
    amplitude, _ = _alias_amplitude(source_rate, SLOW_RATE, 20_000.0)
    assert amplitude < 10 * HAMMING_STOPBAND_GAIN_FILTFILT

    source = _uniform_time(source_rate, 0.08)
    target = source[:: int(round(source_rate / SLOW_RATE))]
    y = resample_to_time(source, np.cos(2 * np.pi * 8_000.0 * source), target)
    keep = _interior(target.size, 200)
    in_band, _ = _tone_amplitude_phase(target[keep], y[keep], 8_000.0)
    # The window method centres its transition (width ~ stopband - cutoff =
    # 2.5 kHz) on the -6 dB cutoff, so the passband runs to ~8.75 kHz and
    # 8 kHz carries only the passband ripple; a 5 kHz-cutoff filter gives ~1e-6.
    assert in_band == pytest.approx(1.0, abs=HAMMING_PASSBAND_DEVIATION_FILTFILT)


def test_a_chirp_keeps_its_in_band_part_and_loses_its_out_of_band_part():
    """Linear chirp 1 -> 40 kHz over 0.4 s.  The sweep moves ~130 Hz across the
    filter's ~1.4 ms length, so the response is quasi-stationary: the in-band
    stretch is the chirp itself, the stretch above the target Nyquist is gone."""
    source = _uniform_time(FAST_RATE, 0.4)
    f0, f1, duration = 1_000.0, 40_000.0, 0.4
    rate = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * source + 0.5 * rate * source**2)
    x = np.cos(phase)
    target = source[::10]
    y = resample_to_time(source, x, target)
    instantaneous = f0 + rate * target
    expected = np.cos(2 * np.pi * (f0 * target + 0.5 * rate * target**2))

    in_band = (instantaneous < 5_000.0) & (target > 0.005)
    # passband ripple of two Hamming passes, plus the quasi-stationary error
    # of a ~1.4 ms filter on a 97.5 kHz/s sweep: < 1e-2 combined.
    assert np.max(np.abs(y[in_band] - expected[in_band])) < 1e-2

    out_of_band = (instantaneous > 15_000.0) & (target < duration - 0.005)
    # beyond 15 kHz the response is in the stopband: two Hamming passes, with
    # the same factor-10 margin as the stationary stopband case.
    assert float(np.sqrt(np.mean(y[out_of_band] ** 2))) < 10 * HAMMING_STOPBAND_GAIN_FILTFILT


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#2166 finding: a record shorter than 3*numtaps+1 is decimated with no "
        "anti-alias filter (documented passthrough + RuntimeWarning + mask), so an "
        "above-Nyquist tone folds in at full amplitude. The analytic requirement "
        "'downsampling introduces no aliased content' does not hold for short records."
    ),
)
def test_a_short_record_is_decimated_without_aliasing():
    source = _uniform_time(FAST_RATE, 0.0032)  # 800 samples < 3 * ~340 + 1
    target = source[::10]
    x = np.cos(2 * np.pi * 20_000.0 * source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = resample_to_time(source, x, target)
    amplitude, _ = _tone_amplitude_phase(target, y, 5_000.0)
    assert amplitude < 0.1


def test_the_filter_mask_marks_a_short_record_as_not_band_limited():
    source = _uniform_time(FAST_RATE, 0.0032)
    x = np.cos(2 * np.pi * 20_000.0 * source)
    with pytest.warns(RuntimeWarning):
        _, mask = resample_to_time(source, x, source[::10], return_filter_mask=True)
    assert not mask.any()


def test_an_irregular_source_refuses_to_be_anti_aliased_but_interpolates_on_request():
    rng = np.random.default_rng(2166)
    steps = 4e-6 * (1.0 + 0.05 * rng.uniform(-1, 1, 20_000))
    source = np.concatenate(([0.0], np.cumsum(steps)))
    target = np.arange(0.0, source[-1], 4e-5)
    line = 3.0 - 2.0 * source
    with pytest.raises(ResamplingError, match="not uniformly sampled"):
        resample_to_time(source, line, target)
    out = resample_to_time(source, line, target, anti_alias=False)
    np.testing.assert_allclose(out, 3.0 - 2.0 * target, rtol=0, atol=1e-12)


def test_extrapolation_policies_do_what_they_name():
    source = np.linspace(0.0, 1.0, 11)
    values = 2.0 * source
    target = np.array([-0.5, 0.5, 1.5])
    np.testing.assert_allclose(
        resample_to_time(source, values, target, anti_alias=False), [0.0, 1.0, 2.0]
    )
    nan_out = resample_to_time(source, values, target, anti_alias=False, extrapolate="nan")
    assert math.isnan(nan_out[0]) and nan_out[1] == pytest.approx(1.0) and math.isnan(nan_out[2])
    with pytest.raises(ResamplingError):
        resample_to_time(source, values, target, anti_alias=False, extrapolate="error")


def test_an_unsorted_source_is_refused_by_default():
    with pytest.raises(ResamplingError, match="strictly increasing"):
        resample_to_time([0.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.5])


def test_a_shuffled_target_gets_the_sorted_targets_values():
    """Metamorphic: evaluating at a permutation of the instants permutes the
    result; the cutoff must come from the sorted spacing."""
    source = _uniform_time(FAST_RATE, 0.02)
    x = np.cos(2 * np.pi * 2_000.0 * source) + np.cos(2 * np.pi * 20_000.0 * source)
    target = source[::10]
    order = np.random.default_rng(7).permutation(target.size)
    sorted_out = resample_to_time(source, x, target)
    shuffled_out = resample_to_time(source, x, target[order])
    np.testing.assert_array_equal(shuffled_out, sorted_out[order])


# ---------------------------------------------------------------------------
# smooth: a centred moving average
# ---------------------------------------------------------------------------

def test_smooth_preserves_constants_and_lines_everywhere():
    """Every window, including the shrunken edge windows, is centred on its
    sample, and a centred mean of a line is the line."""
    x = np.arange(101, dtype=float)
    line = 4.0 - 0.25 * x
    np.testing.assert_allclose(smooth(line, 9), line, rtol=0, atol=1e-12)
    np.testing.assert_allclose(smooth(np.full(50, -2.0), 7), -2.0, rtol=0, atol=1e-14)


def test_smooth_of_an_impulse_is_a_centred_boxcar():
    x = np.zeros(101)
    x[50] = 1.0
    y = smooth(x, 9)
    expected = np.zeros(101)
    expected[46:55] = 1.0 / 9.0  # 9 samples centred on 50: no delay
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-15)


def test_smooth_scales_a_sinusoid_by_the_dirichlet_kernel_with_zero_phase():
    span, n = 7, 4000
    cycles_per_sample = 0.03
    k = np.arange(n)
    x = np.cos(2 * np.pi * cycles_per_sample * k + 0.4)
    y = smooth(x, span)
    # Mean of span samples of a unit cosine: sin(pi f L) / (L sin(pi f)).
    gain = math.sin(math.pi * cycles_per_sample * span) / (span * math.sin(math.pi * cycles_per_sample))
    keep = slice(span, n - span)
    np.testing.assert_allclose(y[keep], gain * x[keep], rtol=0, atol=1e-12)


def test_smooth_reduces_an_even_span_by_one():
    x = np.zeros(21)
    x[10] = 1.0
    expected = np.zeros(21)
    expected[9:12] = 1.0 / 3.0
    np.testing.assert_allclose(smooth(x, 4), expected, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _pulse_on_drift(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drift = 0.3 - 1.7 * t
    pulse = np.where((t > 0.3) & (t < 0.6), np.sin(np.pi * (t - 0.3) / 0.3) ** 2, 0.0)
    baseline_idx = np.flatnonzero((t < 0.25) | (t > 0.65))
    return drift, pulse, baseline_idx


def test_subtract_baseline_removes_a_linear_drift_exactly():
    t = np.linspace(0.0, 1.0, 2001)
    drift, pulse, idx = _pulse_on_drift(t)
    corrected, baseline = subtract_baseline(t, drift + pulse, idx, fitting_opt="linear")
    # Noise-free least squares of a line on a line: exact to round-off of a
    # 2-parameter fit on O(1) values.
    np.testing.assert_allclose(baseline, drift, rtol=0, atol=1e-9)
    np.testing.assert_allclose(corrected, pulse, rtol=0, atol=1e-9)


def test_subtract_baseline_quadratic_removes_a_quadratic_drift_exactly():
    t = np.linspace(0.0, 1.0, 2001)
    drift = 0.1 + 0.5 * t - 0.8 * t**2
    _, pulse, idx = _pulse_on_drift(t)
    corrected, _ = subtract_baseline(t, drift + pulse, idx, fitting_opt="quadratic")
    np.testing.assert_allclose(corrected, pulse, rtol=0, atol=1e-8)


def test_the_magnetics_linear_baseline_recovers_the_line_and_ignores_bad_samples():
    t = np.linspace(0.0, 1.0, 1001)
    line = -0.02 + 0.5 * t
    values = line + np.where(t > 0.5, 1.0, 0.0)
    values[3] = np.nan  # a non-finite sample inside the window is skipped
    indices = np.concatenate((np.arange(0, 200), [5000, -1]))  # out-of-range ignored
    baseline = _linear_baseline(t, values, indices)
    np.testing.assert_allclose(baseline, line, rtol=0, atol=1e-12)


def test_the_magnetics_linear_baseline_refuses_a_degenerate_window():
    t = np.linspace(0.0, 1.0, 11)
    values = t.copy()
    values[1] = np.nan
    with pytest.raises(DegenerateBaselineWindowError):
        _linear_baseline(t, values, np.array([0, 1]))  # one valid sample
    with pytest.warns(UserWarning, match="zero baseline"):
        zero = _linear_baseline(t, values, np.array([0, 1]), allow_zero_fallback=True)
    assert np.array_equal(zero, np.zeros_like(t))


# ---------------------------------------------------------------------------
# Integration: sign, initial condition, accuracy
# ---------------------------------------------------------------------------

def test_cumulative_trapezoid_integrates_cos_to_sin_from_zero():
    omega = 2 * np.pi * 50.0
    h = 1e-5
    t = np.arange(0.0, 0.1, h)
    integral = cumtrapz_compat(np.cos(omega * t), x=t, initial=0.0)
    assert integral[0] == 0.0  # the initial condition is exactly zero
    # Composite trapezoid error bound: T h^2 max|f''| / 12 = T h^2 omega^2 / 12.
    bound = t[-1] * h**2 * omega**2 / 12.0
    np.testing.assert_allclose(integral, np.sin(omega * t) / omega, rtol=0, atol=bound)


def test_cumulative_trapezoid_honours_a_nonuniform_spacing():
    t = np.sort(np.random.default_rng(3).uniform(0.0, 1.0, 400))
    t[0] = 0.0
    # The trapezoid rule is exact for a linear integrand on any grid:
    # integral_0^t 2 s ds = t^2.  A left-Riemann sum would be off by
    # sum(h_i^2) here, which a constant integrand could not reveal.
    np.testing.assert_allclose(cumtrapz_compat(2.0 * t, x=t, initial=0.0), t**2, rtol=0, atol=1e-12)


def test_flux_loop_flux_carries_faradays_minus_sign_and_a_zero_start():
    """V = -dPhi/dt.  A loop voltage V0 switched on at t1 after a quiet baseline
    is a flux -(V0 / 2 pi)(t - t1) per radian after t1 and zero before."""
    n = 2000
    t = np.arange(n) * 4e-6
    t1_index = 1000
    v0 = 0.8
    offset = 0.05  # a DC voltage offset integrates to a linear flux drift ...
    voltage = offset + np.where(np.arange(n) >= t1_index, v0, 0.0)
    config = VestMagneticsProcessingConfig(daq_mode="native_daq", flux_baseline_samples=900)
    flux = vest_flux_loop_flux_from_voltage(t, voltage, flux_loop_number=1, config=config)
    t1 = t[t1_index]
    expected = np.where(t >= t1, -(v0 / (2 * np.pi)) * (t - t1), 0.0)
    # The trapezoid straddling the switch-on contributes h * v0 / 2, i.e. the
    # analytic ramp is offset by -(v0 / 2pi)(h / 2) after t1.
    expected[t1_index:] -= (v0 / (2 * np.pi)) * (4e-6 / 2)
    # ... which the linear baseline fitted over samples 0..899 removes exactly,
    # everywhere: the drift -(offset / 2pi) t is a line on the whole record.
    np.testing.assert_allclose(flux, expected, rtol=0, atol=1e-15)
    assert flux[-1] < 0.0  # positive loop voltage -> decreasing flux


# ---------------------------------------------------------------------------
# Clipping detection and repair
# ---------------------------------------------------------------------------

def test_clip_detection_is_the_union_of_strict_tolerance_bands():
    data = np.array([-5.0, -4.9995, -4.99, 0.0, 4.9998, 5.0 * 32767 / 32768, 4.0])
    mask = detect_clipped_samples(data, clip_values=[-5.0, 5.0 * 32767 / 32768], tolerance=1e-3)
    assert mask.tolist() == [True, True, False, False, True, True, False]
    # The band is open: a sample exactly one tolerance away is not saturated.
    assert not detect_clipped_samples([1.5], clip_values=1.0, tolerance=0.5)[0]


def test_clip_detection_refuses_meaningless_rails():
    with pytest.raises(SignalRepairError):
        detect_clipped_samples([0.0], clip_values=[], tolerance=1e-3)
    with pytest.raises(SignalRepairError):
        detect_clipped_samples([0.0], clip_values=1.0, tolerance=0.0)


def test_repairing_a_clipped_cubic_recovers_the_cubic_exactly():
    """A not-a-knot cubic spline reproduces any cubic polynomial, so a cubic
    clipped at a rail is reconstructed exactly; unclipped samples are untouched."""
    t = np.linspace(-0.9, 0.9, 401)
    # T3(t) + 0.2: a local maximum of 1.2 at t = -0.5, and 0.416 / -0.016 at
    # the ends, so the clipped stretch is interior.
    truth = 4.0 * t**3 - 3.0 * t + 0.2
    rail = 1.0
    clipped = np.minimum(truth, rail)
    repaired, mask = repair_clipped_interval(t, clipped, clip_value=rail, tolerance=1e-9, return_mask=True)
    assert np.array_equal(mask, truth >= rail - 1e-9)
    np.testing.assert_allclose(repaired, truth, rtol=0, atol=1e-10)
    assert np.array_equal(repaired[~mask], clipped[~mask])
    # The mask is what keeps a reconstructed sample from passing as measured.
    assert mask.sum() > 10


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("touches_start", "start or end of the record"),
        ("all_saturated", "Every sample is saturated"),
        ("non_finite", "non-finite"),
        ("too_little_support", "at least 4 unsaturated"),
    ],
)
def test_repair_refuses_to_extrapolate_or_invent(case, message):
    t = np.linspace(0.0, 1.0, 50)
    data = np.sin(2 * np.pi * t)
    if case == "touches_start":
        data[:3] = 1.0
    elif case == "all_saturated":
        data[:] = 1.0
    elif case == "non_finite":
        data[10] = np.nan
    else:
        data[:] = 1.0
        data[[20, 21, 22]] = 0.0  # three unsaturated samples < min_support = 4
    with pytest.raises(SignalRepairError, match=message):
        repair_clipped_interval(t, data, clip_value=1.0, tolerance=1e-6)


# ---------------------------------------------------------------------------
# FIR: apply_fir_filter, the VEST designs, and the probe low pass (post-#2135)
# ---------------------------------------------------------------------------

def _hamming_lowpass(numtaps: int, cutoff: float, fs: float) -> np.ndarray:
    """Window-method low pass written out: Hamming-windowed ideal sinc, unit DC gain.

    h[n] = w[n] * (2 fc / fs) sinc(2 fc / fs (n - M)),  M = (N - 1) / 2,
    w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1)).  Independent of scipy.signal.
    """
    n = np.arange(numtaps)
    centre = (numtaps - 1) / 2.0
    ideal = (2.0 * cutoff / fs) * np.sinc(2.0 * cutoff / fs * (n - centre))
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (numtaps - 1))
    taps = ideal * window
    return taps / taps.sum()


def _dtft(taps: np.ndarray, frequency: float, fs: float) -> complex:
    """H(f) = sum_k h_k exp(-i 2 pi f k / fs), evaluated directly."""
    k = np.arange(taps.size)
    return complex(np.sum(taps * np.exp(-2j * np.pi * frequency * k / fs)))


PROBE_TAPS = 251  # vest.yaml / VestMagneticsProcessingConfig: 251 taps, 2.5 kHz, 250 kHz
PROBE_CUTOFF = 2_500.0
IMPA_TAPS = 26    # vaft.process.impa._FIR_TAPS: the donor's order-round(fs * 1e-3) design
IMPA_RATE = 25_000.0
IMPA_CUTOFF = 250.0  # vest.yaml impa.processing.signal_lowpass_hz


def test_the_vest_probe_design_is_the_251_tap_hamming_window_low_pass():
    taps = vest_probe_lowpass_taps()
    assert taps.size == PROBE_TAPS
    # Same formula evaluated two ways: agreement to round-off.
    np.testing.assert_allclose(taps, _hamming_lowpass(PROBE_TAPS, PROBE_CUTOFF, FAST_RATE), rtol=0, atol=1e-15)


def test_the_impa_low_pass_is_the_26_tap_hamming_design_applied_forward_and_back():
    """filtfilt's impulse response in the interior is h * reversed(h): the
    autocorrelation of the taps, centred on the impulse (even length or not)."""
    h = _hamming_lowpass(IMPA_TAPS, IMPA_CUTOFF, IMPA_RATE)
    n = 2001
    centre = n // 2
    x = np.zeros(n)
    x[centre] = 1.0
    y = impa_lowpass(x, IMPA_CUTOFF, IMPA_RATE)
    autocorrelation = np.correlate(h, h, mode="full")  # length 2N - 1, centred at N - 1
    lags = np.arange(-(IMPA_TAPS - 1), IMPA_TAPS)
    np.testing.assert_allclose(y[centre + lags], autocorrelation, rtol=0, atol=1e-14)
    outside = np.ones(n, dtype=bool)
    outside[centre + lags] = False
    assert np.max(np.abs(y[outside])) < 1e-14


@pytest.mark.parametrize(
    ("numtaps", "cutoff", "fs"),
    [(PROBE_TAPS, PROBE_CUTOFF, FAST_RATE), (IMPA_TAPS, IMPA_CUTOFF, IMPA_RATE)],
)
def test_the_hann_default_design_is_not_equivalent_to_the_vest_hamming_designs(numtaps, cutoff, fs):
    """``fir_filter_coefficients``'s default (Hann, int(fs * 1e-3) forced odd) is
    a different filter from either VEST Hamming design: a different length at
    25 kHz (25 vs 26), a different window at 250 kHz, and a measurably different
    response.  Swapping one for the other is a processing change, not a refactor."""
    hamming = _hamming_lowpass(numtaps, cutoff, fs)
    hann = fir_filter_coefficients(cutoff, fs)
    # Lengths: Hann int(fs * 1e-3) forced odd -- 251 at 250 kHz (same as the
    # probe design), 25 at 25 kHz (one fewer than the donor's 26).
    assert (hann.size, hamming.size) == {FAST_RATE: (251, 251), IMPA_RATE: (25, 26)}[fs]
    # Two-pass magnitude over the passband (0 .. cutoff / 2, where 2 delta is
    # the matching ripple bound) and the transition + stopband.
    freqs = np.concatenate((np.linspace(0.0, 0.5 * cutoff, 100), np.linspace(cutoff, min(10 * cutoff, 0.49 * fs), 400)))
    gain_hamming = np.array([abs(_dtft(hamming, f, fs)) ** 2 for f in freqs])
    gain_hann = np.array([abs(_dtft(hann, f, fs)) ** 2 for f in freqs])
    # Equivalence would need the two to agree to within the Hamming design's
    # own two-pass ripple (2 delta = 4.6e-3) everywhere; they do not.  As a
    # control for the criterion, Hamming 251 vs Hamming 249 taps differ by
    # < 1e-3 (cold review), so it separates "same design" from "different".
    assert np.max(np.abs(gain_hamming - gain_hann)) > HAMMING_PASSBAND_DEVIATION_FILTFILT


def test_apply_fir_filter_is_zero_phase_with_unit_dc_gain():
    taps = vest_probe_lowpass_taps()
    n = 4001
    centre = n // 2
    x = np.zeros(n)
    x[centre] = 1.0
    y = apply_fir_filter(taps, x)
    k = np.arange(1, 600)
    np.testing.assert_allclose(y[centre + k], y[centre - k], rtol=0, atol=1e-15)
    idx = np.arange(n)
    assert float(np.sum(idx * y) / np.sum(y)) == pytest.approx(centre, abs=1e-9)  # group delay 0
    assert float(np.sum(y)) == pytest.approx(1.0, abs=1e-12)  # (sum h)^2 = 1


@pytest.mark.parametrize("frequency", [500.0, PROBE_CUTOFF, 6_000.0])
def test_apply_fir_filter_scales_a_tone_by_the_squared_design_magnitude(frequency):
    """Forward-backward: gain |H(f)|^2 from the taps' own DTFT, phase exactly 0."""
    taps = vest_probe_lowpass_taps()
    t = _uniform_time(FAST_RATE, 0.05)
    x = np.cos(2 * np.pi * frequency * t + 0.3)
    y = apply_fir_filter(taps, x)
    keep = _interior(t.size, 1500)  # past the ~250-sample-per-pass edge transients
    amplitude, phase = _tone_amplitude_phase(t[keep], y[keep], frequency)
    expected = abs(_dtft(taps, frequency, FAST_RATE)) ** 2
    # Least squares over >= 20 cycles of the interior; error is round-off.
    assert amplitude == pytest.approx(expected, abs=1e-9)
    if expected > 1e-3:
        assert phase == pytest.approx(0.3, abs=1e-6)


def test_a_short_record_is_refused_by_default():
    taps = vest_probe_lowpass_taps()
    with pytest.raises(ValueError, match=f"at least {3 * PROBE_TAPS + 1} samples"):
        apply_fir_filter(taps, np.ones(3 * PROBE_TAPS))
    # The boundary itself is accepted.
    assert apply_fir_filter(taps, np.ones(3 * PROBE_TAPS + 1)).shape == (3 * PROBE_TAPS + 1,)


def test_the_causal_fallback_is_a_forward_convolution_with_the_full_group_delay():
    """short_record='causal' is lfilter: y = (h * x)[:n], a pure delay of (N - 1) / 2
    samples on a linear-phase design -- 125 samples, 0.5 ms, for the probe filter."""
    taps = vest_probe_lowpass_taps()
    n = 600  # < 3 * 251 + 1
    t = np.arange(n) / FAST_RATE
    x = np.cos(2 * np.pi * 500.0 * t)
    y = apply_fir_filter(taps, x, short_record="causal")
    np.testing.assert_allclose(y, np.convolve(taps, x)[:n], rtol=0, atol=1e-13)
    # Past the start-up transient the output is the input delayed by 125
    # samples and scaled by |H(500 Hz)|.
    delay = (PROBE_TAPS - 1) // 2
    gain = abs(_dtft(taps, 500.0, FAST_RATE))
    steady = slice(PROBE_TAPS, n)
    delayed = gain * np.cos(2 * np.pi * 500.0 * (t[steady] - delay / FAST_RATE))
    np.testing.assert_allclose(y[steady], delayed, rtol=0, atol=1e-12)


def test_a_single_tap_is_its_gain_applied_twice():
    x = np.linspace(-1.0, 2.0, 7)
    np.testing.assert_array_equal(apply_fir_filter([0.5], x), x * 0.25)
    np.testing.assert_array_equal(apply_fir_filter(np.array([-3.0]), x), x * 9.0)


def test_unknown_short_record_policy_is_refused():
    with pytest.raises(ValueError, match="short_record"):
        apply_fir_filter([0.5, 0.5], np.ones(10), short_record="pad")


def test_the_probe_low_pass_moves_no_feature_in_time():
    """End to end through b_field_pol_probe_field: a probe voltage V = -dB/dt of
    a Gaussian field pulse (sigma 2 ms, spectrum e^{-(2 pi f sigma)^2 / 2} below
    1e-30 at the 2.5 kHz cutoff) comes back as B itself -- same peak instant,
    same centroid.  The pre-#2135 causal lfilter delayed it by 125 samples."""
    t = _uniform_time(FAST_RATE, 0.06)
    t0, sigma, amplitude = 0.03, 2e-3, 0.05
    field_true = amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    voltage = -field_true * (-(t - t0) / sigma**2)  # V = -dB/dt, analytic
    baseline = np.arange(2000)  # 8 ms of quiet lead-in: B < 1e-20 there
    *_, field, _ = b_field_pol_probe_field(t, voltage, np.array([1.0]), vest_probe_lowpass_taps(),
                                           baseline_indices=baseline)
    field = field[:, 0]
    # A loose upper bound (two-pass ripple times the amplitude; the actual
    # error is ~6e-6), still 30x below what a 0.5 ms shift of this pulse
    # produces (~7.6e-3).  The sharp checks are the centroid and the peak.
    np.testing.assert_allclose(field, field_true, rtol=0, atol=HAMMING_PASSBAND_DEVIATION_FILTFILT * amplitude)
    centroid = lambda y: float(np.sum(t * y) / np.sum(y))  # noqa: E731
    # A causal filter moves the centroid by 0.5 ms; this bound is 1/1000 of a sample.
    assert centroid(field) == pytest.approx(centroid(field_true), abs=1e-3 / FAST_RATE)
    assert int(np.argmax(field)) == int(np.argmax(field_true))


def test_the_probe_low_pass_refuses_a_record_too_short_for_zero_phase():
    """No silent causal fallback in the probe path: that is the route back to the
    0.5 ms delay #2135 removed."""
    n = 3 * PROBE_TAPS  # one short of the 3N + 1 filtfilt needs
    t = np.arange(n) / FAST_RATE
    with pytest.raises(ValueError, match="zero-phase"):
        b_field_pol_probe_field(t, np.zeros(n), np.array([1.0]), vest_probe_lowpass_taps(),
                                baseline_indices=np.arange(100))
