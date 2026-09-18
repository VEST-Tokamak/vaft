"""Cross-spectrum, coherence and ridge tracking between two fluctuation records (#1005).

The numbers are checked against signals whose answer is known by construction:
a lag of ``dt`` between two tones is a phase of ``-2 pi f dt`` at that tone,
independent noise has a coherence below the 95 % significance level, and a
linear chirp is recovered bin by bin.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.fluctuation import (
    CrossSpectrum,
    FrequencyTrack,
    compute_spectrogram,
    cross_spectrum,
    track_dominant_frequency,
)


def _tone_pair(*, fs=200_000.0, f0=10_000.0, lag_s=12.5e-6, duration=0.02, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration * fs)) / fs
    x = np.sin(2 * np.pi * f0 * time) + noise * rng.standard_normal(time.size)
    y = 0.5 * np.sin(2 * np.pi * f0 * (time - lag_s)) + noise * rng.standard_normal(time.size)
    return time, x, y


class TestCrossSpectrum:
    def test_the_phase_at_the_tone_is_the_lag(self):
        f0, lag = 10_000.0, 12.5e-6  # an eighth of a period: -45 degrees
        time, x, y = _tone_pair(f0=f0, lag_s=lag)
        result = cross_spectrum(time, x, time, y, nperseg=400)
        assert isinstance(result, CrossSpectrum)
        peak = int(np.argmin(np.abs(result.frequency - f0)))
        assert result.frequency[peak] == pytest.approx(f0)
        assert result.phase[peak] == pytest.approx(-2 * np.pi * f0 * lag, abs=0.02)
        assert result.coherence[peak] > 0.99
        assert result.coherence[peak] > result.significance_95

    def test_the_phase_sign_is_y_relative_to_x(self):
        """Swapping the arguments flips the phase: it is arg(S_xy) = phi_y - phi_x."""
        time, x, y = _tone_pair()
        forward = cross_spectrum(time, x, time, y, nperseg=400)
        backward = cross_spectrum(time, y, time, x, nperseg=400)
        peak = int(np.argmax(forward.coherence))
        assert backward.phase[peak] == pytest.approx(-forward.phase[peak], abs=1e-9)
        np.testing.assert_allclose(backward.coherence, forward.coherence, atol=1e-12)

    def test_independent_noise_stays_below_significance(self):
        rng = np.random.default_rng(3)
        fs = 100_000.0
        time = np.arange(40_000) / fs
        x, y = rng.standard_normal(time.size), rng.standard_normal(time.size)
        result = cross_spectrum(time, x, time, y, nperseg=256)
        # 50 % overlap on 40000 samples of 256: (40000 - 128) // 128 segments.
        assert result.n_segments == (40_000 - 128) // 128
        assert result.significance_95 == pytest.approx(
            1 - 0.05 ** (1 / (result.n_segments - 1))
        )
        # A 95 % level: about 5 % of independent bins may exceed it, no more.
        exceed = np.mean(result.coherence > result.significance_95)
        assert exceed < 0.10
        assert np.median(result.coherence) < result.significance_95

    def test_coherence_and_csd_match_scipy(self):
        from scipy import signal

        time, x, y = _tone_pair(noise=0.5)
        result = cross_spectrum(time, x, time, y, nperseg=256)
        f, cxy = signal.coherence(x, y, fs=200_000.0, nperseg=256)
        _, pxy = signal.csd(x, y, fs=200_000.0, nperseg=256)
        np.testing.assert_allclose(result.frequency, f)
        np.testing.assert_allclose(result.coherence, cxy, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.csd, pxy, rtol=1e-10, atol=1e-20)
        np.testing.assert_allclose(result.phase, np.angle(pxy))

    def test_mismatched_rates_meet_on_the_slower_grid_over_the_overlap(self):
        f0, lag = 5_000.0, 25e-6
        fast_fs, slow_fs = 1_000_000.0, 250_000.0
        t_fast = 0.300 + np.arange(40_000) / fast_fs            # 0.300 .. 0.340 s
        t_slow = 0.310 + np.arange(10_000) / slow_fs            # 0.310 .. 0.350 s
        x = np.sin(2 * np.pi * f0 * t_fast)
        y = np.sin(2 * np.pi * f0 * (t_slow - lag))
        result = cross_spectrum(t_fast, x, t_slow, y, nperseg=500)
        assert result.sample_rate == pytest.approx(slow_fs)
        assert result.resampled == ("x",)
        low, high = result.time_range
        assert low == pytest.approx(0.310)
        assert high == pytest.approx(t_fast[-1], abs=1 / slow_fs)
        peak = int(np.argmin(np.abs(result.frequency - f0)))
        assert result.phase[peak] == pytest.approx(-2 * np.pi * f0 * lag, abs=0.03)
        assert result.coherence[peak] > 0.99

    def test_an_explicit_common_time_is_used_as_given(self):
        time, x, y = _tone_pair()
        grid = time[100:3000:2]
        result = cross_spectrum(time, x, time, y, common_time=grid, nperseg=200)
        assert result.sample_rate == pytest.approx(100_000.0)
        assert result.time_range == (pytest.approx(grid[0]), pytest.approx(grid[-1]))
        assert set(result.resampled) == {"x", "y"}

    def test_records_that_do_not_overlap_are_refused(self):
        t1 = np.arange(1000) / 1e5
        t2 = t1 + 1.0
        with pytest.raises(ValueError, match="do not overlap"):
            cross_spectrum(t1, np.ones_like(t1), t2, np.ones_like(t2))

    def test_too_few_segments_for_a_significance_level_is_nan_not_a_number(self):
        time, x, y = _tone_pair(duration=0.002)  # 400 samples
        result = cross_spectrum(time, x, time, y, nperseg=400)
        assert result.n_segments == 1
        assert np.isnan(result.significance_95)


def _chirp(*, fs=200_000.0, duration=0.04, f_start=10_000.0, f_end=7_000.0, silent=None):
    time = np.arange(int(duration * fs)) / fs
    rate = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * time + 0.5 * rate * time**2)
    signal = np.sin(phase)
    if silent is not None:
        signal[(time >= silent[0]) & (time < silent[1])] = 0.0
    truth = f_start + rate * time
    return time, signal, truth


class TestTrackDominantFrequency:
    def test_a_chirp_is_recovered_within_one_bin(self):
        time, signal, truth = _chirp()
        spectrogram = compute_spectrogram(time, signal, nperseg=400)
        track = track_dominant_frequency(spectrogram, search_range=(3_000.0, 20_000.0), max_jump=1_500.0)
        assert isinstance(track, FrequencyTrack)
        bin_width = spectrogram.frequency[1] - spectrogram.frequency[0]
        expected = np.interp(track.time, time, truth)
        assert np.all(np.isfinite(track.frequency))
        assert np.max(np.abs(track.frequency - expected)) <= bin_width
        t, f, p = track  # unpacks as (time, frequency, power)
        assert p.shape == f.shape == t.shape

    def test_a_silent_stretch_returns_nan(self):
        time, signal, _ = _chirp(silent=(0.015, 0.025))
        spectrogram = compute_spectrogram(time, signal, nperseg=400)
        track = track_dominant_frequency(spectrogram, search_range=(3_000.0, 20_000.0), max_jump=1_500.0)
        half = 200 / 200_000.0  # half a window: windows touching the edge are mixed
        inside = (track.time > 0.015 + half) & (track.time < 0.025 - half)
        outside = (track.time < 0.015 - half) | (track.time > 0.025 + half)
        assert inside.any() and np.all(np.isnan(track.frequency[inside]))
        assert np.all(np.isnan(track.power[inside]))
        assert np.all(np.isfinite(track.frequency[outside]))

    def test_the_continuity_limit_keeps_a_weaker_ridge_against_a_distant_burst(self):
        """The burst is the loudest pixel of the map, so a tracker anchored on it would lose the ridge."""
        frequency = np.arange(0.0, 20_000.0, 500.0)
        magnitude = np.full((frequency.size, 5), 0.01)
        ridge = [8_000.0, 8_500.0, 9_000.0, 9_500.0, 10_000.0]
        for column, f in enumerate(ridge):
            magnitude[np.searchsorted(frequency, f), column] = 1.0
        magnitude[np.searchsorted(frequency, 16_000.0), 2] = 1.5  # a louder, distant burst
        unconstrained = track_dominant_frequency(
            (np.arange(5.0), frequency, magnitude), search_range=(1_000.0, 19_000.0)
        )
        constrained = track_dominant_frequency(
            (np.arange(5.0), frequency, magnitude), search_range=(1_000.0, 19_000.0), max_jump=1_000.0
        )
        assert unconstrained.frequency[2] == pytest.approx(16_000.0)
        np.testing.assert_allclose(constrained.frequency, ridge)

    def test_the_floor_is_relative_to_the_map(self):
        frequency = np.arange(0.0, 10_000.0, 1_000.0)
        magnitude = np.zeros((frequency.size, 3))
        magnitude[5, 0] = 1.0
        magnitude[5, 1] = 0.02
        magnitude[5, 2] = 0.5
        track = track_dominant_frequency(
            (np.arange(3.0), frequency, magnitude), search_range=(1_000.0, 9_000.0), floor_ratio=0.1
        )
        assert track.frequency[0] == pytest.approx(5_000.0)
        assert np.isnan(track.frequency[1])
        assert track.frequency[2] == pytest.approx(5_000.0)
        assert track.power[2] == pytest.approx(0.5)

    def test_no_bin_in_range_is_all_nan(self):
        frequency = np.arange(0.0, 1_000.0, 100.0)
        track = track_dominant_frequency(
            (np.arange(2.0), frequency, np.ones((frequency.size, 2))), search_range=(5_000.0, 6_000.0)
        )
        assert np.all(np.isnan(track.frequency))

    def test_the_camera_reference_tracker_delegates_and_agrees(self):
        from vaft.process.camera_fluctuation import track_reference_frequency

        time, signal, _ = _chirp()
        spectrogram = compute_spectrogram(time, signal, nperseg=400)
        shared = track_dominant_frequency(spectrogram, search_range=(3_000.0, 15_000.0), floor_ratio=0.0)
        np.testing.assert_allclose(track_reference_frequency(spectrogram), shared.frequency)
