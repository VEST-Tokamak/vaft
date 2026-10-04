"""Time-resolved cross spectra from signals with known frequency and phase."""

import numpy as np
import pytest
from scipy import signal

from vaft.process.fluctuation import cross_spectrogram


def _tone_pair(fs=40_000, duration=0.2, frequency=2_000, phase=0.6):
    time = 0.3 + np.arange(round(fs * duration)) / fs
    rng = np.random.default_rng(1611)
    x = np.sin(2 * np.pi * frequency * time) + 0.1 * rng.normal(size=time.size)
    y = np.sin(2 * np.pi * frequency * time + phase) + 0.1 * rng.normal(size=time.size)
    return time, x, y


def test_known_frequency_phase_and_one_sided_welch_density():
    time, x, y = _tone_pair()
    result = cross_spectrogram(time, x, time, y, nperseg=200)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.frequency[peak] == pytest.approx(2_000)
    assert np.median(result.coherence[peak]) > 0.95
    assert np.median(result.phase[peak]) == pytest.approx(0.6, abs=0.04)
    assert result.time[0] == pytest.approx(time[0] + 399.5 / 40_000)
    assert result.frequency_resolution_hz == pytest.approx(200)
    assert result.segments_per_window == 4
    assert result.window_step_samples == 400
    _, expected = signal.csd(x[:800], y[:800], fs=40_000, nperseg=200, noverlap=0)
    np.testing.assert_allclose(result.csd[:, 0], expected[:result.frequency.size])
    assert result.bandwidth_source == ("nyquist_only", "nyquist_only")


def test_independent_noise_has_finite_coherence_below_one():
    rng = np.random.default_rng(9)
    time = np.arange(16_000) / 40_000
    result = cross_spectrogram(time, rng.normal(size=time.size),
                               time, rng.normal(size=time.size))
    assert np.nanmedian(result.coherence) < 0.4
    assert np.nanmedian(result.coherence) > 0.01
    assert np.nanmax(result.coherence) <= 1


def test_phase_reverses_when_inputs_are_swapped():
    time, x, y = _tone_pair()
    forward = cross_spectrogram(time, x, time, y, nperseg=200)
    reverse = cross_spectrogram(time, y, time, x, nperseg=200)
    np.testing.assert_allclose(forward.csd, np.conj(reverse.csd))
    np.testing.assert_allclose(np.angle(np.exp(1j * (forward.phase + reverse.phase))), 0,
                               atol=1e-12)


def test_rate_reduction_records_filter_and_rejects_alias_band():
    fast = 0.3 + np.arange(40_000) / 200_000
    slow = 0.31 + np.arange(8_000) / 40_000
    # The 38 kHz fast-only tone would alias to 2 kHz without filtering.
    x = np.sin(2 * np.pi * 2_000 * fast) + np.sin(2 * np.pi * 38_000 * fast)
    y = np.sin(2 * np.pi * 2_000 * slow + 0.4)
    result = cross_spectrogram(fast, x, slow, y, nperseg=200,
                               bandwidth_y_hz=10_000)
    assert result.sample_rate == pytest.approx(40_000)
    assert result.common_bandwidth_hz == pytest.approx(10_000)
    assert result.frequency[-1] == pytest.approx(10_000)
    assert result.resampling[0].operation == "anti_alias_resample"
    assert result.resampling[0].anti_alias_cutoff_hz == pytest.approx(16_000)
    assert result.resampling[1].operation == "crop"
    assert result.bandwidth_source == ("nyquist_only", "declared")
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert np.median(result.coherence[peak]) > 0.95
    assert np.median(result.phase[peak]) == pytest.approx(0.4, abs=0.05)
    pure = cross_spectrogram(fast, np.sin(2 * np.pi * 2_000 * fast),
                             slow, y, nperseg=200, bandwidth_y_hz=10_000)
    assert np.median(result.psd_x[peak]) == pytest.approx(
        np.median(pure.psd_x[peak]), rel=0.03
    )
    sampling_only = cross_spectrogram(fast, x, slow, y, nperseg=200)
    assert sampling_only.common_bandwidth_hz == pytest.approx(14_000)
    with pytest.raises(ValueError, match="above common usable bandwidth"):
        cross_spectrogram(fast, x, slow, y, nperseg=200,
                          frequency_range=(0, 18_000))


def test_short_nonoverlapping_and_outside_grid_are_rejected():
    time, x, y = _tone_pair(duration=0.01)
    with pytest.raises(ValueError, match="are required"):
        cross_spectrogram(time, x, time, y)
    with pytest.raises(ValueError, match="do not overlap"):
        cross_spectrogram(time, x, time + 1, y, nperseg=50)
    with pytest.raises(ValueError, match="within both"):
        cross_spectrogram(time, x, time, y, nperseg=50,
                          common_time=time - 1 / 40_000)


def test_zero_power_has_undefined_coherence_and_phase():
    time = np.arange(2048) / 40_000
    result = cross_spectrogram(time, np.zeros_like(time), time,
                               np.sin(2 * np.pi * 2_000 * time))
    assert np.isnan(result.coherence).all()
    assert np.isnan(result.phase).all()
