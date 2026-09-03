"""Anti-aliased resampling: rejection, in-band fidelity, and timing (issue #425).

Ground truth is analytic.  VEST acquires at 250 kHz (``FAST_DT``) and 25 kHz
(``SLOW_DT``) while every processed time grid is 25 kHz, so the case under test
throughout is a 10x decimation: a tone at 40 kHz folds to
``|40 - 2 * 25| = 10 kHz``, squarely inside the band a reader would trust.

The no-op tests matter as much as the rejection test.  ``resample_to_time`` is
adopted at call sites where the rates are already equal, and those sites must
not move a single stored value -- so equality with ``np.interp`` there is
asserted exactly, not approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.signal_processing import (
    ResamplingError,
    anti_alias_filter,
    describe_time_grid,
    resample_to_time,
)

FS_FAST = 250_000.0  # VEST fast DAQ [Hz]
FS_SLOW = 25_000.0  # every processed policy grid [Hz]
SPAN = 0.1  # seconds, the fast DAQ's record length


def _fast_grid() -> np.ndarray:
    return np.arange(0.0, SPAN, 1.0 / FS_FAST)


def _slow_grid() -> np.ndarray:
    return np.arange(0.0, SPAN, 1.0 / FS_SLOW)


def _tone(time: np.ndarray, frequency: float, phase: float = 0.0) -> np.ndarray:
    return np.sin(2.0 * np.pi * frequency * time + phase)


def _amplitude_at(values: np.ndarray, frequency: float, fs: float) -> float:
    """Single-sided amplitude of ``values`` in the bin nearest ``frequency``."""
    spectrum = np.abs(np.fft.rfft(values)) * 2.0 / values.size
    freqs = np.fft.rfftfreq(values.size, 1.0 / fs)
    return float(spectrum[np.argmin(np.abs(freqs - frequency))])


class TestDescribeTimeGrid:
    def test_arange_grid_is_uniform_with_the_exact_step(self):
        grid = describe_time_grid(_slow_grid())
        assert grid.uniform
        assert grid.strictly_increasing
        assert grid.dt == pytest.approx(1.0 / FS_SLOW, rel=1e-9)
        assert grid.sample_rate == pytest.approx(FS_SLOW, rel=1e-9)

    def test_linspace_grid_of_late_shots_is_still_uniform(self):
        # Shots after 42190 store linspace(0, span, n), so dt is span/(n-1)
        # rather than the nominal 4e-6.  A tight tolerance would reject it.
        grid = describe_time_grid(np.linspace(0.0, SPAN, 25_000))
        assert grid.uniform
        assert grid.dt == pytest.approx(SPAN / 24_999)
        assert grid.dt != 1.0 / FS_FAST

    def test_jittered_grid_is_not_uniform(self):
        rng = np.random.default_rng(0)
        time = _slow_grid()
        time = time + rng.normal(0.0, 0.3 / FS_SLOW, size=time.size)
        time = np.sort(time)
        assert not describe_time_grid(time).uniform

    def test_single_sample_grid_has_no_defined_spacing(self):
        grid = describe_time_grid([0.3])
        assert grid.n == 1
        assert grid.uniform
        assert np.isnan(grid.dt)
        assert np.isnan(grid.sample_rate)

    def test_non_monotonic_grid_is_reported(self):
        assert not describe_time_grid([0.0, 2.0, 1.0]).strictly_increasing
        assert not describe_time_grid([0.0, 1.0, 1.0]).strictly_increasing


class TestAntiAliasRejection:
    def test_bare_interpolation_folds_a_40khz_tone_into_the_band(self):
        # This is the defect, asserted so the fix cannot be quietly undone.
        source_time = _fast_grid()
        target_time = _slow_grid()
        aliased = np.interp(target_time, source_time, _tone(source_time, 40_000.0))
        assert _amplitude_at(aliased, 10_000.0, FS_SLOW) > 0.5

    def test_resample_to_time_rejects_the_same_tone(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        clean = resample_to_time(source_time, _tone(source_time, 40_000.0), target_time)
        # 40 dB is the acceptance bar; a 331-tap Hamming design gives far more.
        assert _amplitude_at(clean, 10_000.0, FS_SLOW) < 1e-2
        assert np.abs(clean).max() < 1e-2

    def test_rejection_holds_across_the_folding_frequencies(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        for frequency in (26_000.0, 40_000.0, 51_000.0, 74_000.0, 99_000.0):
            clean = resample_to_time(
                source_time, _tone(source_time, frequency), target_time
            )
            assert np.abs(clean).max() < 1e-2, frequency


class TestInBandPreservation:
    def test_amplitude_and_waveform_survive(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        resampled = resample_to_time(source_time, _tone(source_time, 2_000.0), target_time)
        assert _amplitude_at(resampled, 2_000.0, FS_SLOW) == pytest.approx(1.0, rel=1e-2)
        expected = _tone(target_time, 2_000.0)
        interior = slice(400, -400)  # clear of the filtfilt edge transients
        assert np.abs(resampled[interior] - expected[interior]).max() < 1e-2

    def test_phase_is_unshifted(self):
        # The property filtfilt buys and a causal lfilter would not: the
        # cross-correlation against the analytic tone peaks at lag zero exactly.
        source_time = _fast_grid()
        target_time = _slow_grid()
        resampled = resample_to_time(source_time, _tone(source_time, 2_000.0), target_time)
        interior = slice(400, -400)
        a = resampled[interior] - resampled[interior].mean()
        b = _tone(target_time, 2_000.0)[interior]
        correlation = np.correlate(a, b, mode="full")
        assert np.argmax(correlation) - (b.size - 1) == 0


class TestTimingPreservation:
    def test_step_edge_moves_by_at_most_one_target_sample(self):
        # A causal lfilter of the same design would shift this by
        # numtaps//2 / fs = 165 / 250e3 = 0.66 ms, about 16 target samples.
        source_time = _fast_grid()
        target_time = _slow_grid()
        step = (source_time >= 0.05).astype(float)
        resampled = resample_to_time(source_time, step, target_time)
        crossing = target_time[np.argmax(resampled >= 0.5)]
        assert abs(crossing - 0.05) <= 1.0 / FS_SLOW

    def test_gaussian_peak_stays_put(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        pulse = np.exp(-(((source_time - 0.05) / 2e-4) ** 2))
        resampled = resample_to_time(source_time, pulse, target_time)
        assert abs(target_time[np.argmax(resampled)] - 0.05) <= 1.0 / FS_SLOW


class TestNoOpGuarantees:
    def test_already_bandlimited_source_is_left_alone(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        signal = _tone(source_time, 1_000.0)
        filtered = resample_to_time(source_time, signal, target_time)
        plain = np.interp(target_time, source_time, signal)
        interior = slice(400, -400)
        # The residual is the FIR's passband ripple at 1 kHz, not lost signal.
        assert np.abs(filtered[interior] - plain[interior]).max() < 5e-3

    def test_equal_rate_alignment_is_bit_for_bit_interp(self):
        source_time = _slow_grid()
        signal = _tone(source_time, 3_000.0)
        for offset in (0.0, 1e-6, 0.5 / FS_SLOW):
            target_time = source_time + offset
            assert np.array_equal(
                resample_to_time(source_time, signal, target_time),
                np.interp(target_time, source_time, signal),
            )

    def test_upsampling_is_bit_for_bit_interp(self):
        source_time = _slow_grid()
        target_time = _fast_grid()
        signal = _tone(source_time, 3_000.0)
        assert np.array_equal(
            resample_to_time(source_time, signal, target_time),
            np.interp(target_time, source_time, signal),
        )

    def test_opting_out_is_bit_for_bit_interp_even_on_a_10x_decimation(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        signal = _tone(source_time, 40_000.0)
        assert np.array_equal(
            resample_to_time(source_time, signal, target_time, anti_alias=False),
            np.interp(target_time, source_time, signal),
        )

    def test_forcing_the_filter_overrides_the_ratio_test(self):
        source_time = _slow_grid()
        target_time = source_time
        signal = _tone(source_time, 9_000.0)
        forced = resample_to_time(
            source_time, signal, target_time, anti_alias=True, cutoff_hz=2_000.0
        )
        assert not np.array_equal(forced, signal)
        assert np.abs(forced[400:-400]).max() < 1e-2


class TestEdgeCases:
    def test_empty_target_returns_empty(self):
        source_time = _slow_grid()
        out = resample_to_time(source_time, _tone(source_time, 100.0), np.array([]))
        assert out.shape == (0,)

    def test_empty_source_raises(self):
        with pytest.raises(ResamplingError):
            resample_to_time(np.array([]), np.array([]), _slow_grid())

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ResamplingError):
            resample_to_time(_slow_grid(), np.zeros(3), _slow_grid())

    def test_single_sample_source_fills_a_constant(self):
        out = resample_to_time([0.3], [2.5], _slow_grid())
        assert np.array_equal(out, np.full(_slow_grid().size, 2.5))

    def test_single_sample_target_warns_and_interpolates(self):
        source_time = _fast_grid()
        signal = _tone(source_time, 1_000.0)
        with pytest.warns(RuntimeWarning, match="single sample"):
            out = resample_to_time(source_time, signal, [0.05])
        assert out.shape == (1,)
        assert out[0] == pytest.approx(np.interp(0.05, source_time, signal))

    def test_single_sample_target_cannot_place_a_forced_cutoff(self):
        source_time = _fast_grid()
        with pytest.raises(ResamplingError, match="cutoff_hz"):
            resample_to_time(source_time, _tone(source_time, 1e3), [0.05], anti_alias=True)

    def test_targets_past_the_source_are_clamped_like_interp(self):
        source_time = _slow_grid()
        signal = _tone(source_time, 1_000.0)
        target_time = np.arange(0.0, 0.15, 1.0 / FS_SLOW)
        out = resample_to_time(source_time, signal, target_time)
        overhang = target_time > source_time[-1]
        assert np.allclose(out[overhang], signal[-1])

    def test_extrapolate_nan_marks_the_overhang(self):
        source_time = _slow_grid()
        signal = _tone(source_time, 1_000.0)
        target_time = np.arange(0.0, 0.15, 1.0 / FS_SLOW)
        out = resample_to_time(source_time, signal, target_time, extrapolate="nan")
        assert np.isnan(out[target_time > source_time[-1]]).all()

    def test_extrapolate_error_refuses_to_fabricate_a_tail(self):
        source_time = _slow_grid()
        with pytest.raises(ResamplingError, match="fabricate"):
            resample_to_time(
                source_time,
                _tone(source_time, 1e3),
                np.arange(0.0, 0.15, 1.0 / FS_SLOW),
                extrapolate="error",
            )

    def test_unsorted_source_raises_rather_than_returning_nonsense(self):
        time = np.array([0.0, 2.0, 1.0, 3.0])
        with pytest.raises(ResamplingError, match="strictly increasing"):
            resample_to_time(time, np.arange(4.0), np.array([0.5, 1.5]))

    def test_duplicate_timestamps_raise(self):
        time = np.array([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ResamplingError):
            resample_to_time(time, np.arange(4.0), np.array([0.5, 1.5]))

    def test_on_unsorted_sort_recovers(self):
        time = np.array([0.0, 2.0, 1.0, 2.0])
        values = np.array([0.0, 2.0, 1.0, 2.0])
        with pytest.warns(RuntimeWarning, match="sorting"):
            out = resample_to_time(
                time, values, np.array([0.5, 1.5]), on_unsorted="sort"
            )
        assert np.allclose(out, [0.5, 1.5])

    def test_short_source_falls_back_to_plain_interpolation(self):
        source_time = np.arange(0.0, 400e-6, 1.0 / FS_FAST)  # 100 samples
        signal = _tone(source_time, 40_000.0)
        target_time = np.arange(0.0, 400e-6, 1.0 / FS_SLOW)  # 10 samples
        with pytest.warns(RuntimeWarning, match="shorter than"):
            out = resample_to_time(source_time, signal, target_time)
        assert np.array_equal(out, np.interp(target_time, source_time, signal))

    def test_nan_gaps_stay_local_and_rejection_still_holds(self):
        source_time = _fast_grid()
        signal = _tone(source_time, 40_000.0)
        signal[12_000:12_010] = np.nan
        target_time = _slow_grid()
        out = resample_to_time(source_time, signal, target_time)
        gap = (target_time >= source_time[12_000]) & (target_time <= source_time[12_009])
        assert np.isnan(out[gap]).all()
        early = out[400:1_000]
        assert np.isfinite(early).all()
        assert np.abs(early).max() < 1e-2

    def test_nan_policy_error_refuses(self):
        source_time = _fast_grid()
        signal = _tone(source_time, 40_000.0)
        signal[100] = np.nan
        with pytest.raises(ResamplingError):
            resample_to_time(source_time, signal, _slow_grid(), nan_policy="error")

    def test_two_dimensional_input_is_resampled_per_row(self):
        source_time = _fast_grid()
        target_time = _slow_grid()
        stacked = np.vstack([_tone(source_time, 40_000.0), _tone(source_time, 2_000.0)])
        out = resample_to_time(source_time, stacked, target_time)
        assert out.shape == (2, target_time.size)
        assert np.abs(out[0]).max() < 1e-2
        assert _amplitude_at(out[1], 2_000.0, FS_SLOW) == pytest.approx(1.0, rel=1e-2)

    def test_invalid_options_are_rejected(self):
        source_time = _slow_grid()
        signal = _tone(source_time, 100.0)
        for kwargs in (
            {"anti_alias": "maybe"},
            {"extrapolate": "wrap"},
            {"on_unsorted": "shrug"},
        ):
            with pytest.raises(ValueError):
                resample_to_time(source_time, signal, source_time, **kwargs)


class TestAntiAliasFilterDirectly:
    def test_cutoff_must_sit_below_the_source_nyquist(self):
        with pytest.raises(ResamplingError):
            anti_alias_filter(np.zeros(10_000), source_rate=FS_SLOW, cutoff_hz=20_000.0)

    def test_transition_band_is_sized_against_the_stopband_not_the_source(self):
        # Sizing the taps against the source Nyquist would give a 9-tap filter
        # with ~25 dB of rejection at 40 kHz instead of the 300+ taps needed.
        source_time = _fast_grid()
        filtered = anti_alias_filter(
            _tone(source_time, 40_000.0),
            source_rate=FS_FAST,
            cutoff_hz=10_000.0,
            stopband_hz=12_500.0,
        )
        assert np.abs(filtered).max() < 1e-3
