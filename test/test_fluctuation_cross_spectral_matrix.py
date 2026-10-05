"""Multi-record cross spectra with known phase, bandwidth and missing data."""

import numpy as np
import pytest
from scipy import signal

from vaft.process.fluctuation import FluctuationRecord, cross_spectral_matrix


def _record(name, time, values, *, band=None, units="a.u."):
    return FluctuationRecord(name, time, values, units, band)


def test_one_sided_raw_matrix_matches_scipy_and_phase_convention():
    fs = 40_000
    time = 0.3 + np.arange(8_000) / fs
    rng = np.random.default_rng(1611)
    x = np.sin(2 * np.pi * 2_000 * time) + 0.05 * rng.normal(size=time.size)
    y = 3 * np.sin(2 * np.pi * 2_000 * time + 0.5) + 0.05 * rng.normal(size=time.size)
    result = cross_spectral_matrix([_record("x", time, x, units="T"),
                                    _record("y", time, y, units="W")], nperseg=200)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    _, expected = signal.csd(x[:800], y[:800], fs=fs, nperseg=200, noverlap=0)
    assert result.raw_csd[peak, 0, 1, 0] == pytest.approx(expected[peak])
    assert np.angle(result.raw_csd[peak, 0, 1, 0]) == pytest.approx(0.5, abs=0.03)
    np.testing.assert_allclose(result.raw_csd[peak, 0],
                               result.raw_csd[peak, 0].conj().T)
    np.testing.assert_allclose(np.diag(result.matrix[peak, 0]), 1)
    assert result.units == ("T", "W")
    assert result.normalization == "psd"
    assert result.shared_segments[peak, 0] == 4


def test_independent_noise_and_two_modes_do_not_make_rank_one_matrix():
    rng = np.random.default_rng(7)
    fs = 40_000
    time = np.arange(16_000) / fs
    x = np.sin(2 * np.pi * 2_000 * time) + 0.3 * rng.normal(size=time.size)
    y = np.sin(2 * np.pi * 2_000 * time + 0.3) + 0.3 * rng.normal(size=time.size)
    z = np.sin(2 * np.pi * 5_000 * time) + 0.3 * rng.normal(size=time.size)
    result = cross_spectral_matrix([_record("x", time, x), _record("y", time, y),
                                    _record("z", time, z)], nperseg=200)
    at_2k = int(np.argmin(abs(result.frequency - 2_000)))
    at_5k = int(np.argmin(abs(result.frequency - 5_000)))
    assert abs(np.median(result.matrix[at_2k, :, 0, 1])) > 0.8
    assert abs(np.median(result.matrix[at_2k, :, 0, 2])) < 0.8
    assert abs(np.median(result.matrix[at_5k, :, 0, 2])) < 0.8
    for frequency in (at_2k, at_5k):
        for column in range(result.time.size):
            assert np.linalg.eigvalsh(result.matrix[frequency, column]).min() > -1e-12


def test_frequency_mask_excludes_slow_record_without_zero_filling():
    fast = np.arange(24_000) / 120_000
    slow = np.arange(8_000) / 40_000
    x = np.sin(2 * np.pi * 2_000 * fast) + np.sin(2 * np.pi * 30_000 * fast)
    y = np.sin(2 * np.pi * 2_000 * slow)
    result = cross_spectral_matrix([_record("fast", fast, x),
                                    _record("slow", slow, y, band=10_000)], nperseg=240)
    low = int(np.argmin(abs(result.frequency - 2_000)))
    high = int(np.argmin(abs(result.frequency - 30_000)))
    assert result.sample_rate == pytest.approx(120_000)
    assert result.resampling[1].operation == "interpolate"
    assert result.valid[low, :, :].all()
    assert result.valid[high, :, 0].all()
    assert not result.valid[high, :, 1].any()
    assert np.isnan(result.matrix[high, :, 1, :]).all()
    assert np.isnan(result.raw_csd[high, :, :, 1]).all()


def test_missing_record_is_unavailable_while_others_remain():
    time = np.arange(8_000) / 40_000
    x = np.sin(2 * np.pi * 2_000 * time)
    missing = np.full(time.size, np.nan)
    result = cross_spectral_matrix([_record("x", time, x),
                                    _record("missing", time, missing),
                                    _record("copy", time, x)], nperseg=200)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.valid[peak, :, 0].all()
    assert not result.valid[peak, :, 1].any()
    assert result.valid[peak, :, 2].all()
    assert np.isnan(result.matrix[peak, :, 1, :]).all()
    assert np.nanmedian(abs(result.matrix[peak, :, 0, 2])) == pytest.approx(1)


def test_explicit_normalization_scales_and_invalid_options():
    fs = 40_000
    time = np.arange(8_000) / fs
    x = np.sin(2 * np.pi * 2_000 * time)
    records = [_record("x", time, x), _record("double", time, 2 * x)]
    raw = cross_spectral_matrix(records, nperseg=200, normalization="none")
    user = cross_spectral_matrix(records, nperseg=200, normalization="user",
                                 user_scales={"x": 1, "double": 2})
    variance = cross_spectral_matrix(records, nperseg=200, normalization="variance")
    peak = int(np.argmin(abs(raw.frequency - 2_000)))
    assert user.matrix[peak, 0, 1, 1] == pytest.approx(raw.matrix[peak, 0, 1, 1] / 4)
    assert variance.normalization_scales[1] == pytest.approx(2 * variance.normalization_scales[0])
    with pytest.raises(ValueError, match="exactly one"):
        cross_spectral_matrix(records, normalization="user", user_scales={"x": 1})
    with pytest.raises(ValueError, match="unique"):
        cross_spectral_matrix([records[0], records[0]])


def test_requested_rate_reduction_filters_fast_only_alias_and_records_cutoff():
    fast = np.arange(24_000) / 120_000
    slow = np.arange(8_000) / 40_000
    low = np.sin(2 * np.pi * 2_000 * fast)
    mixed = low + np.sin(2 * np.pi * 38_000 * fast)
    y = np.sin(2 * np.pi * 2_000 * slow)
    inputs = [_record("fast", fast, mixed), _record("slow", slow, y)]
    result = cross_spectral_matrix(inputs, sample_rate=40_000, nperseg=200)
    reference = cross_spectral_matrix([_record("fast", fast, low), inputs[1]],
                                      sample_rate=40_000, nperseg=200)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.resampling[0].operation == "anti_alias_resample"
    assert result.resampling[0].anti_alias_cutoff_hz == pytest.approx(16_000)
    assert result.resampling[0].usable_bandwidth_hz == pytest.approx(14_000)
    assert result.frequency[-1] == pytest.approx(20_000)
    above_fast_band = int(np.argmin(abs(result.frequency - 18_000)))
    assert not result.valid[above_fast_band, :, 0].any()
    assert result.valid[above_fast_band, :, 1].all()
    assert np.nanmedian(result.raw_csd[peak, :, 0, 0].real) == pytest.approx(
        np.nanmedian(reference.raw_csd[peak, :, 0, 0].real), rel=0.03
    )


def test_missing_time_block_excludes_only_affected_windows():
    time = np.arange(8_000) / 40_000
    x = np.sin(2 * np.pi * 2_000 * time)
    gapped = x.copy()
    gapped[3_000:4_000] = np.nan
    result = cross_spectral_matrix([_record("x", time, x),
                                    _record("gapped", time, gapped)], nperseg=200)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.valid[peak, 0].all()
    assert not result.valid[peak, 8, 1]
    assert result.valid[peak, -1].all()


def test_disjoint_partial_records_do_not_erase_fully_observed_pair():
    time = np.arange(800) / 40_000
    tone = np.sin(2 * np.pi * 2_000 * time)
    first_half = tone.copy()
    first_half[400:800] = np.nan
    second_half = tone.copy()
    second_half[:400] = np.nan
    result = cross_spectral_matrix([
        _record("x", time, tone), _record("y", time, tone),
        _record("first", time, first_half), _record("second", time, second_half),
    ], nperseg=100)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.valid[peak, 0, 0]
    assert result.valid[peak, 0, 1]
    assert np.isfinite(result.raw_csd[peak, 0, 0, 1])
    assert result.shared_segments[peak, 0] >= 2
    assert result.valid[peak, 0, 2] != result.valid[peak, 0, 3]


def test_exactly_one_outer_window_keeps_last_sample():
    time = np.arange(400) / 100_000
    signal = np.sin(2 * np.pi * 2_000 * time)
    result = cross_spectral_matrix([_record("x", time, signal)],
                                   nperseg=100, segments_per_window=4)
    assert result.time.size == 1
    assert result.time_range[1] == pytest.approx(time[-1])


def test_zero_power_keeps_raw_zero_but_masks_undefined_psd_normalization():
    time = np.arange(800) / 40_000
    tone = np.sin(2 * np.pi * 2_000 * time)
    result = cross_spectral_matrix([_record("off", time, np.zeros_like(time)),
                                    _record("tone", time, tone)], nperseg=100)
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.raw_valid[peak, 0].all()
    assert result.raw_csd[peak, 0, 0, 0] == 0
    assert result.raw_csd[peak, 0, 0, 1] == 0
    assert not result.valid[peak, 0, 0]
    assert result.valid[peak, 0, 1]
    assert np.isnan(result.matrix[peak, 0, 0, :]).all()


def test_variance_normalization_masks_missing_record_without_losing_pair():
    time = np.arange(800) / 40_000
    tone = np.sin(2 * np.pi * 2_000 * time)
    result = cross_spectral_matrix([
        _record("x", time, tone), _record("missing", time, np.full(time.size, np.nan)),
        _record("y", time, tone),
    ], nperseg=100, normalization="variance")
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert result.valid[peak, 0, 0]
    assert not result.valid[peak, 0, 1]
    assert result.valid[peak, 0, 2]
    assert np.isnan(result.normalization_scales[1])
    assert np.isfinite(result.matrix[peak, 0, 0, 2])


def test_nonoverlap_and_short_record_fail_explicitly():
    time = np.arange(100) / 40_000
    with pytest.raises(ValueError, match="required"):
        cross_spectral_matrix([_record("x", time, np.ones_like(time))])
    with pytest.raises(ValueError, match="do not overlap"):
        cross_spectral_matrix([_record("x", time, np.ones_like(time)),
                               _record("y", time + 1, np.ones_like(time))])
