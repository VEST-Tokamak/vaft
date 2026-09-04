"""Synthetic coverage for the ported VEST SXR Viewer analysis routines.

Oracle: ``vest_sxr_viewer.py`` v5 (module-level science functions, lines
159-433).  During the port every routine here was verified bit-identical
against that oracle on shared random input -- band signals, the full optional
vacuum-reference workflow, the Be/Al Te chain (values, NaN masks, relative
fluctuation), Hilbert phase, and the ranked mode-number table.  These tests
re-check the same behaviours against analytic ground truth so they need no
external files.
"""

import numpy as np
import pytest

from vaft.process.signal_processing import (
    butterworth_bandpass,
    butterworth_lowpass,
    detrend_moving_average,
)
from vaft.process.soft_x_rays import (
    hilbert_instantaneous_phase,
    load_te_ratio_calibration,
    rank_toroidal_mode_numbers,
    sxr_band_signals,
    sxr_baseline_correction,
    sxr_electron_temperature,
    sxr_subtract_vacuum_reference,
)

FS = 125e6 / 128.0  # 455xx-campaign SXR sampling frequency [Hz]


def sxr_time(n: int = 20000) -> np.ndarray:
    return np.arange(n, dtype=float) / FS + 0.285


class TestConditioningHelpers:
    def test_lowpass_passes_slow_and_rejects_fast(self):
        t = sxr_time()
        slow = np.sin(2 * np.pi * 1_000.0 * t)
        fast = np.sin(2 * np.pi * 200_000.0 * t)

        out = butterworth_lowpass(slow + fast, 10_000.0, FS, order=4)

        # The causal filter delays the slow tone too, so compare against its
        # own response to the slow part: the difference is the leaked fast tone.
        slow_only = butterworth_lowpass(slow, 10_000.0, FS, order=4)
        tail = slice(2000, None)  # skip the settle-in
        assert np.std(out[tail] - slow_only[tail]) < 0.01 * np.std(fast)

    def test_bandpass_selects_only_its_band(self):
        t = sxr_time()
        in_band = np.sin(2 * np.pi * 3_500.0 * t)
        below = np.sin(2 * np.pi * 500.0 * t)
        above = np.sin(2 * np.pi * 30_000.0 * t)

        out = butterworth_bandpass(in_band + below + above, 3e3, 4e3, FS)

        interior = slice(3000, -3000)  # keep clear of filtfilt edge transients
        assert np.corrcoef(out[interior], in_band[interior])[0, 1] > 0.99
        assert np.std(out[interior]) == pytest.approx(
            np.std(in_band[interior]), rel=0.05
        )

    def test_band_edges_are_validated(self):
        with pytest.raises(ValueError, match="band edges"):
            butterworth_bandpass(np.zeros(64), 4e3, 3e3, FS)
        with pytest.raises(ValueError, match="cutoff"):
            butterworth_lowpass(np.zeros(64), FS, FS)

    def test_detrend_matches_pandas_centered_rolling_mean(self):
        import pandas as pd

        x = np.random.default_rng(0).standard_normal(500)
        for window in (2, 3, 4, 7, 300, 499, 600):
            expected = x - pd.Series(x).rolling(
                window=window, center=True, min_periods=1
            ).mean().to_numpy()
            np.testing.assert_allclose(detrend_moving_average(x, window), expected)

    def test_detrend_removes_a_slow_ramp(self):
        t = sxr_time(4096)
        tone = np.sin(2 * np.pi * 50_000.0 * t)
        ramp = np.linspace(0.0, 5.0, t.size)

        out = detrend_moving_average(tone + ramp, 301)

        interior = slice(400, -400)
        assert np.max(np.abs(out[interior] - tone[interior])) < 0.05


class TestBaselineAndReference:
    def test_baseline_zeroes_a_known_per_channel_offset(self):
        t = sxr_time()
        offsets = np.array([[0.5], [-1.2], [3.0]])
        data = np.zeros((3, t.size)) + offsets

        corrected = sxr_baseline_correction(t, data, baseline_start=0.300)

        np.testing.assert_allclose(corrected, 0.0, atol=1e-12)

    def test_baseline_falls_back_to_the_record_tail(self):
        t = sxr_time(4000)
        data = np.ones((2, t.size))

        corrected = sxr_baseline_correction(t, data, baseline_start=99.0)

        np.testing.assert_allclose(corrected, 0.0, atol=1e-12)

    def test_reference_subtraction_removes_common_mode_and_keeps_the_tone(self):
        t = sxr_time()
        # PF pickup is slow compared with the reference low-pass; the causal
        # filter's group delay is negligible at 100 Hz but real at kHz scales.
        pf_noise = 0.4 * np.sin(2 * np.pi * 100.0 * t)
        tone = 0.2 * np.sin(2 * np.pi * 20_000.0 * t)   # plasma-only fluctuation
        plasma = np.tile(pf_noise + tone, (2, 1))
        vacuum = np.tile(pf_noise, (2, 1))

        corrected = sxr_subtract_vacuum_reference(
            plasma, vacuum, cutoff=3_000.0, fs=FS, order=2
        )

        tail = slice(4000, None)
        residual = corrected[0, tail] - tone[tail]
        assert np.std(residual) < 0.1 * np.std(pf_noise)
        assert np.corrcoef(corrected[0, tail], tone[tail])[0, 1] > 0.99

    def test_reference_is_a_standalone_step_only(self):
        # The main routines must not accept a reference: PF-noise removal is
        # always the caller's explicit pre-step.
        import inspect

        for function in (sxr_band_signals, sxr_electron_temperature):
            assert "reference" not in inspect.signature(function).parameters

    def test_records_are_truncated_to_the_shorter_one(self):
        result = sxr_subtract_vacuum_reference(
            np.zeros((2, 1000)), np.zeros((2, 800)), cutoff=1e3, fs=FS
        )
        assert result.shape == (2, 800)


class TestBandSignals:
    def test_recovers_a_tone_in_band_and_rejects_one_outside(self):
        t = sxr_time()
        tone = 0.3 * np.sin(2 * np.pi * 3_500.0 * t)
        other = 0.5 * np.sin(2 * np.pi * 20_000.0 * t)
        data = np.tile(tone + other + 0.7, (2, 1))

        result = sxr_band_signals(
            t, data, baseline_start=0.320,
            bands={"low": (3e3, 4e3), "high": (18e3, 22e3)},
            fs=FS, time_range=(0.290, 0.300),
        )

        window = (t >= 0.290) & (t <= 0.300)
        # Filtering runs on the windowed segment (viewer parity), so keep the
        # comparison clear of the filtfilt edge transients at both ends.
        interior = slice(2000, -2000)
        assert np.corrcoef(
            result.bands["low"][0][interior], tone[window][interior]
        )[0, 1] > 0.99
        assert np.corrcoef(
            result.bands["high"][0][interior], other[window][interior]
        )[0, 1] > 0.99
        assert np.std(result.bands["low"][0][interior]) == pytest.approx(
            np.std(tone), rel=0.05
        )

    def test_dead_channels_zero_the_bands_but_keep_raw(self):
        t = sxr_time(8192)
        data = np.tile(np.sin(2 * np.pi * 3_500.0 * t), (3, 1))

        result = sxr_band_signals(
            t, data, baseline_start=0.291,
            bands={"b": (3e3, 4e3)}, fs=FS, dead_channels=[1],
        )

        np.testing.assert_array_equal(result.bands["b"][1], 0.0)
        assert np.std(result.raw[1]) > 0.1          # raw stays visible
        assert np.std(result.bands["b"][0]) > 0.1   # neighbours untouched

    def test_baseline_none_means_already_corrected(self):
        t = sxr_time(8192)
        data = np.tile(np.sin(2 * np.pi * 3_500.0 * t), (1, 1)) + 2.0

        result = sxr_band_signals(
            t, data, baseline_start=None, bands={}, fs=FS
        )

        # No baseline was subtracted: the offset survives in raw.
        assert result.raw.mean() == pytest.approx(2.0, abs=0.01)

    def test_empty_time_range_raises(self):
        t = sxr_time(1000)
        with pytest.raises(ValueError, match="selects no samples"):
            sxr_band_signals(t, np.zeros((1, t.size)), baseline_start=None,
                             bands={}, fs=FS, time_range=(9.0, 10.0))


class TestElectronTemperature:
    @staticmethod
    def linear_calibration(ratio):
        return 10.0 + 50.0 * np.asarray(ratio)

    def test_recovers_te_from_a_known_signal_ratio(self):
        t = sxr_time()
        al = np.full(t.size, 1.0)
        be = np.full(t.size, 0.5 * 1.07)  # ratio 0.5 after the Al gain

        result = sxr_electron_temperature(
            t, np.vstack([be, al]), [(0, 1)],
            calibration=self.linear_calibration,
            baseline_start=None, fs=FS,
            al_threshold=0.10, time_range=(0.295, 0.300),
        )

        # Steady signals pass the causal low-pass unchanged after settle-in.
        assert np.nanmedian(result.te) == pytest.approx(10.0 + 50.0 * 0.5, rel=1e-3)

    def test_al_below_threshold_is_masked_as_nan(self):
        t = sxr_time()
        al = np.full(t.size, 1.0)
        al[t > 0.295] = 0.01          # drops below threshold mid-window
        be = np.full(t.size, 0.4)

        result = sxr_electron_temperature(
            t, np.vstack([be, al]), [(0, 1)],
            calibration=self.linear_calibration,
            baseline_start=None, fs=FS, al_threshold=0.10,
            time_range=(0.290, 0.300),
        )

        early = result.time < 0.294
        late = result.time > 0.296
        assert np.all(np.isfinite(result.te[0][early]))
        assert np.all(np.isnan(result.te[0][late]))

    def test_constant_te_has_near_zero_relative_fluctuation(self):
        t = sxr_time()
        result = sxr_electron_temperature(
            t, np.vstack([np.full(t.size, 0.535), np.full(t.size, 1.0)]), [(0, 1)],
            calibration=self.linear_calibration,
            baseline_start=None, fs=FS, time_range=(0.295, 0.300),
        )
        assert np.nanmax(np.abs(result.rel_fluctuation)) < 0.5  # percent

    def test_packaged_calibration_table_round_trips(self):
        interpolator, te, ratio = load_te_ratio_calibration()

        assert te.size >= 2 and te.size == ratio.size
        # The interpolator must reproduce the table's own points.
        np.testing.assert_allclose(interpolator(ratio), te, rtol=1e-12)
        # And Te rises with the Be/Al ratio over the tabulated range.
        assert np.all(np.diff(te[np.argsort(ratio)]) > 0)

    def test_calibration_table_requires_te_and_ratio_columns(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'te' and 'ratio'"):
            load_te_ratio_calibration(bad)


class TestToroidalModeRanking:
    THETA_A, THETA_B = 0.0, 120.0  # VEST 4 o'clock / 12 o'clock ports

    def test_recovers_an_imposed_mode_number(self):
        n_true = 1
        phase_a = 37.0
        # phase(theta) = phase_a - n * (theta - theta_a), wrapped.
        phase_b = ((phase_a - n_true * (self.THETA_B - self.THETA_A)) + 180.0) % 360.0 - 180.0

        ranked = rank_toroidal_mode_numbers(
            self.THETA_A, phase_a, self.THETA_B, phase_b, n_max=6
        )

        assert ranked[0].n == n_true
        assert ranked[0].residual_deg == pytest.approx(0.0, abs=1e-9)

    def test_reports_the_120_degree_aliasing_degeneracy(self):
        # With observations 120 deg apart, n and n +/- 3 are indistinguishable.
        phase_b = ((37.0 - 1 * 120.0) + 180.0) % 360.0 - 180.0
        ranked = rank_toroidal_mode_numbers(0.0, 37.0, 120.0, phase_b, n_max=6)

        best = ranked[0]
        degenerate = {c.n for c in ranked
                      if abs(abs(c.residual_deg) - abs(best.residual_deg)) < 1e-6}
        assert {1, -2, 4}.issubset(degenerate)   # 1, 1-3, 1+3
        # |n| smallest is ranked first among the degenerates.
        assert best.n == 1

    def test_equal_angles_raise(self):
        with pytest.raises(ValueError, match="single location"):
            rank_toroidal_mode_numbers(120.0, 10.0, 120.0, 30.0, 4)
        with pytest.raises(ValueError, match="single location"):
            rank_toroidal_mode_numbers(0.0, 10.0, 360.0, 30.0, 4)

    def test_hilbert_phase_of_a_pure_tone(self):
        t = sxr_time(8192)
        frequency = 3_500.0
        phase0 = np.deg2rad(40.0)
        signal = np.cos(2 * np.pi * frequency * t + phase0)

        t_eval = float(t[4000])
        phase_deg, envelope, index, _, _ = hilbert_instantaneous_phase(
            signal, t, t_eval
        )

        expected = np.rad2deg(
            np.angle(np.exp(1j * (2 * np.pi * frequency * t_eval + phase0)))
        )
        assert index == 4000
        assert phase_deg == pytest.approx(expected, abs=1.0)
        assert envelope == pytest.approx(1.0, abs=0.02)

    def test_end_to_end_mode_number_from_two_synthetic_signals(self):
        """Full chain: band-pass -> Hilbert phase at two ports -> ranked n."""
        t = sxr_time()
        frequency, n_true = 3_500.0, 1
        make = lambda theta_deg: np.cos(
            2 * np.pi * frequency * t - np.deg2rad(n_true * theta_deg)
        )
        t_eval = 0.295

        phases = []
        for theta in (self.THETA_A, self.THETA_B):
            result = sxr_band_signals(
                t, make(theta)[np.newaxis, :], baseline_start=None,
                bands={"mode": (3e3, 4e3)}, fs=FS, time_range=(0.290, 0.300),
            )
            phase_deg, *_ = hilbert_instantaneous_phase(
                result.bands["mode"][0], result.time, t_eval
            )
            phases.append(phase_deg)

        ranked = rank_toroidal_mode_numbers(
            self.THETA_A, phases[0], self.THETA_B, phases[1], n_max=6
        )
        best = ranked[0]
        degenerate = {c.n for c in ranked
                      if abs(abs(c.residual_deg) - abs(best.residual_deg)) < 1.0}
        assert n_true in degenerate
        assert best.n == n_true  # |n| tiebreak puts the true small mode first

    def test_an_aliased_true_mode_is_reported_through_its_representative(self):
        # A true n=2 wave is indistinguishable from n=-1 with ports 120 deg
        # apart; the ranking surfaces the minimal-|n| representative but the
        # true mode sits in the same equal-residual degenerate set.
        phase_b = float((37.0 - 2 * 120.0 + 180.0) % 360.0 - 180.0)
        ranked = rank_toroidal_mode_numbers(0.0, 37.0, 120.0, phase_b, n_max=6)

        best = ranked[0]
        degenerate = {c.n for c in ranked
                      if abs(abs(c.residual_deg) - abs(best.residual_deg)) < 1e-6}
        assert best.n == -1
        assert 2 in degenerate
