"""Synthetic coverage for the diagnostic-independent spectral analysis layer.

Every signal here is generated in-process, so the numerical claims -- recovered
peaks, slopes, breaks and band powers -- are checked against a known truth rather
than against a stored fixture.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from vaft.process.fluctuation import (
    MIN_FIT_POINTS,
    NONUNIFORM_TOLERANCE,
    FluctuationSpectrum,
    analyze_fluctuation_spectrum,
    compute_band_power,
    compute_psd,
    compute_spectrogram,
    find_spectral_break,
    fit_power_law_spectrum,
)

SAMPLE_RATE = 250_000.0
N_SAMPLES = 2**18


def uniform_time(n: int = N_SAMPLES, fs: float = SAMPLE_RATE) -> np.ndarray:
    return np.arange(n, dtype=float) / fs


def power_law_signal(alpha: float, *, n: int = N_SAMPLES, fs: float = SAMPLE_RATE,
                     seed: int = 1) -> np.ndarray:
    """A time series whose PSD follows ``f**alpha`` by construction.

    Amplitudes are shaped as ``f**(alpha/2)`` because the PSD is the squared
    magnitude; phases are random so the result is broadband, not periodic.
    """
    rng = np.random.default_rng(seed)
    frequency = np.fft.rfftfreq(n, 1.0 / fs)
    amplitude = np.zeros_like(frequency)
    amplitude[1:] = frequency[1:] ** (alpha / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, frequency.size)
    return np.fft.irfft(amplitude * np.exp(1j * phase), n)


def broken_power_law_signal(alpha_low: float, alpha_high: float, break_frequency: float,
                            *, n: int = N_SAMPLES, fs: float = SAMPLE_RATE,
                            seed: int = 7) -> np.ndarray:
    """A time series with two power-law regimes joined continuously at the break."""
    rng = np.random.default_rng(seed)
    frequency = np.fft.rfftfreq(n, 1.0 / fs)
    amplitude = np.zeros_like(frequency)
    low = (frequency > 0) & (frequency <= break_frequency)
    high = frequency > break_frequency
    amplitude[low] = frequency[low] ** (alpha_low / 2.0)
    # Continuity at the break fixes the high-frequency prefactor.
    amplitude[high] = break_frequency ** ((alpha_low - alpha_high) / 2.0) * frequency[
        high
    ] ** (alpha_high / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, frequency.size)
    return np.fft.irfft(amplitude * np.exp(1j * phase), n)


class TestPsdEstimation:
    def test_recovers_a_known_spectral_peak(self):
        time = uniform_time()
        rng = np.random.default_rng(0)
        signal = np.sin(2 * np.pi * 12_000.0 * time) + 0.01 * rng.standard_normal(time.size)

        spectrum = compute_psd(time, signal, nperseg=8192)

        peak = spectrum.frequency[np.argmax(spectrum.psd)]
        resolution = spectrum.sample_rate / 8192
        assert abs(peak - 12_000.0) < 2 * resolution

    def test_integrated_psd_matches_signal_variance(self):
        # Parseval: for a zero-mean signal, the integral of the one-sided PSD
        # over frequency is the variance.
        time = uniform_time()
        signal = 2.0 * np.random.default_rng(3).standard_normal(time.size)

        spectrum = compute_psd(time, signal, nperseg=8192, detrend=False)

        integrated = np.trapezoid(spectrum.psd, spectrum.frequency)
        assert integrated == pytest.approx(np.var(signal), rel=0.02)

    def test_sample_rate_is_derived_from_the_time_axis(self):
        spectrum = compute_psd(uniform_time(n=4096), power_law_signal(-2.0, n=4096))
        assert spectrum.sample_rate == pytest.approx(SAMPLE_RATE)
        assert spectrum.method == "welch"

    def test_explicit_sample_rate_overrides_the_time_axis(self):
        time = uniform_time(n=4096)
        signal = power_law_signal(-2.0, n=4096)

        spectrum = compute_psd(time, signal, sample_rate=1000.0)

        assert spectrum.sample_rate == pytest.approx(1000.0)
        assert spectrum.frequency.max() == pytest.approx(500.0)

    def test_frequency_axis_is_deterministic(self):
        time = uniform_time(n=8192)
        signal = power_law_signal(-2.0, n=8192)

        first = compute_psd(time, signal, nperseg=1024)
        second = compute_psd(time, signal, nperseg=1024)

        np.testing.assert_array_equal(first.frequency, second.frequency)
        np.testing.assert_array_equal(first.psd, second.psd)


class TestTimeAxisValidation:
    def test_rejects_a_decreasing_time_axis(self):
        time = uniform_time(n=4096)
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_psd(time[::-1], power_law_signal(-2.0, n=4096))

    def test_rejects_repeated_timestamps(self):
        time = uniform_time(n=4096)
        time[10] = time[9]
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_psd(time, power_law_signal(-2.0, n=4096))

    def test_rejects_materially_nonuniform_sampling(self):
        # The whole point: never silently assume uniform spacing.
        time = np.sort(np.random.default_rng(1).uniform(0.0, 0.04, 4096))
        with pytest.raises(ValueError, match="materially nonuniform"):
            compute_psd(time, power_law_signal(-2.0, n=4096))

    def test_tolerates_jitter_below_the_documented_tolerance(self):
        time = uniform_time(n=4096)
        jitter = 0.1 * NONUNIFORM_TOLERANCE / SAMPLE_RATE
        time = time + np.linspace(0.0, jitter, time.size)

        assert compute_psd(time, power_law_signal(-2.0, n=4096)).psd.size > 0

    def test_rejects_mismatched_lengths(self):
        time = uniform_time(n=4096)
        with pytest.raises(ValueError, match="equal length"):
            compute_psd(time, power_law_signal(-2.0, n=4096)[:-1])


class TestPowerLawFitting:
    @pytest.mark.parametrize("alpha", [-1.0, -2.0, -3.0])
    def test_recovers_a_known_spectral_index(self, alpha):
        time = uniform_time()
        spectrum = compute_psd(time, power_law_signal(alpha), nperseg=8192)

        fit = fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=(1e3, 5e4))

        assert fit.alpha == pytest.approx(alpha, abs=0.05)
        assert fit.r_squared > 0.99
        assert fit.n_points > MIN_FIT_POINTS
        assert fit.frequency_range == (1e3, 5e4)
        assert fit.residuals.size == fit.n_points
        assert fit.stderr < 0.01

    def test_reports_fit_quality_for_a_poor_fit(self):
        # White noise is flat, so a power-law model explains almost nothing.
        time = uniform_time()
        noise = np.random.default_rng(11).standard_normal(time.size)
        spectrum = compute_psd(time, noise, nperseg=8192)

        fit = fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=(1e3, 5e4))

        assert fit.alpha == pytest.approx(0.0, abs=0.05)
        assert fit.r_squared < 0.1

    def test_raises_when_the_range_holds_too_few_points(self):
        time = uniform_time(n=4096)
        spectrum = compute_psd(time, power_law_signal(-2.0, n=4096), nperseg=256)

        with pytest.raises(ValueError, match="insufficient fit range"):
            fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=(1000.0, 1001.0))

    def test_rejects_an_inverted_range(self):
        time = uniform_time(n=4096)
        spectrum = compute_psd(time, power_law_signal(-2.0, n=4096))
        with pytest.raises(ValueError, match="increasing"):
            fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=(5e4, 1e3))


class TestDerivativeTransferFunction:
    def test_differentiation_shifts_the_spectral_index_by_two(self):
        """A pickup coil measures dB/dt, so its PSD slope is alpha + 2, not alpha.

        This is the guard against reading a Mirnov voltage spectral index as a
        magnetic-field spectral index.
        """
        time = uniform_time()
        field = power_law_signal(-3.0)
        derivative = np.gradient(field, time)

        field_spectrum = compute_psd(time, field, nperseg=8192)
        derivative_spectrum = compute_psd(time, derivative, nperseg=8192)

        # Stay well below Nyquist: the discrete-gradient transfer function rolls
        # off from the ideal 2*pi*f as the band edge is approached.
        band = (1e3, 2e4)
        field_fit = fit_power_law_spectrum(
            field_spectrum.frequency, field_spectrum.psd, f_range=band
        )
        derivative_fit = fit_power_law_spectrum(
            derivative_spectrum.frequency, derivative_spectrum.psd, f_range=band
        )

        assert derivative_fit.alpha - field_fit.alpha == pytest.approx(2.0, abs=0.1)

    def test_compute_psd_does_not_integrate_derivative_signals(self):
        # compute_psd must analyse exactly what it is handed; if it silently
        # integrated dB/dt the two spectra below would agree.
        time = uniform_time(n=2**16)
        field = power_law_signal(-3.0, n=2**16)
        derivative = np.gradient(field, time)

        field_psd = compute_psd(time, field, nperseg=4096).psd
        derivative_psd = compute_psd(time, derivative, nperseg=4096).psd

        # atol=0: these PSDs are ~1e-16, so an absolute tolerance would call
        # any two of them equal.
        assert not np.allclose(field_psd, derivative_psd, rtol=1e-6, atol=0.0)


class TestSpectralBreak:
    def test_imposed_mode_fits_either_side_of_a_caller_boundary(self):
        time = uniform_time()
        signal = broken_power_law_signal(-1.0, -3.0, 8_000.0)
        spectrum = compute_psd(time, signal, nperseg=8192)

        result = find_spectral_break(
            spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4),
            break_frequency=8_000.0,
        )

        assert result.mode == "imposed"
        assert result.break_frequency == pytest.approx(8_000.0)
        assert result.alpha_low == pytest.approx(-1.0, abs=0.1)
        assert result.alpha_high == pytest.approx(-3.0, abs=0.1)
        assert result.r_squared > 0.95

    def test_search_mode_recovers_the_break_from_the_data(self):
        time = uniform_time()
        signal = broken_power_law_signal(-1.0, -3.0, 8_000.0)
        spectrum = compute_psd(time, signal, nperseg=8192)

        result = find_spectral_break(
            spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4),
            search_range=(2e3, 3e4),
        )

        assert result.mode == "search"
        assert result.break_frequency == pytest.approx(8_000.0, rel=0.2)
        assert result.alpha_low == pytest.approx(-1.0, abs=0.15)
        assert result.alpha_high == pytest.approx(-3.0, abs=0.15)

    @pytest.mark.parametrize("true_break", [5.0e3, 8.0e3, 2.0e4])
    def test_search_does_not_pin_to_the_edge_of_the_search_range(self, true_break):
        """Regression on the scoring criterion.

        Scoring candidates by point-weighted R^2 biased the answer toward small
        break frequencies -- the low segment holds few points there, so the
        score was dominated by the high segment and pinned to the range edge.
        Total squared residual is comparable across candidates and does not.
        """
        time = uniform_time()
        signal = broken_power_law_signal(-1.0, -3.0, true_break)
        spectrum = compute_psd(time, signal, nperseg=8192)
        search_range = (2.0e3, 4.0e4)

        result = find_spectral_break(
            spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4),
            search_range=search_range,
        )

        assert result.break_frequency == pytest.approx(true_break, rel=0.1)
        assert result.break_frequency > search_range[0] * 1.05
        assert result.break_frequency < search_range[1] * 0.95

    def test_the_two_modes_are_distinct_and_never_inferred(self):
        time = uniform_time(n=2**16)
        spectrum = compute_psd(time, broken_power_law_signal(-1.0, -3.0, 8_000.0, n=2**16))

        with pytest.raises(ValueError, match="exactly one of"):
            find_spectral_break(spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4))
        with pytest.raises(ValueError, match="exactly one of"):
            find_spectral_break(
                spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4),
                break_frequency=8e3, search_range=(2e3, 3e4),
            )

    def test_imposed_break_must_lie_inside_the_fit_range(self):
        time = uniform_time(n=2**16)
        spectrum = compute_psd(time, broken_power_law_signal(-1.0, -3.0, 8_000.0, n=2**16))
        with pytest.raises(ValueError, match="strictly inside"):
            find_spectral_break(
                spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4), break_frequency=9e4
            )

    def test_search_range_must_lie_inside_the_fit_range(self):
        time = uniform_time(n=2**16)
        spectrum = compute_psd(time, broken_power_law_signal(-1.0, -3.0, 8_000.0, n=2**16))
        with pytest.raises(ValueError, match="must lie within"):
            find_spectral_break(
                spectrum.frequency, spectrum.psd, fit_range=(500.0, 6e4),
                search_range=(100.0, 3e4),
            )


class TestBandPower:
    def test_integrates_a_flat_spectrum_analytically(self):
        # A constant PSD integrates to height * width over any band.
        frequency = np.linspace(0.0, 1000.0, 1001)
        psd = np.full_like(frequency, 2.0)

        powers = compute_band_power(frequency, psd, {"a": (100.0, 300.0)})

        assert powers["a"] == pytest.approx(2.0 * 200.0)

    def test_band_edges_are_inclusive(self):
        frequency = np.linspace(0.0, 100.0, 101)
        psd = np.ones_like(frequency)

        powers = compute_band_power(frequency, psd, {"edge": (10.0, 20.0)})

        # Closed interval [10, 20] spans exactly 10 Hz of a unit-height PSD.
        assert powers["edge"] == pytest.approx(10.0)

    def test_reports_caller_defined_ratios(self):
        frequency = np.linspace(0.0, 1000.0, 1001)
        psd = np.full_like(frequency, 3.0)

        powers = compute_band_power(
            frequency, psd,
            {"low": (0.0, 100.0), "high": (100.0, 300.0)},
            ratios={"high_over_low": ("high", "low")},
        )

        assert powers["high_over_low"] == pytest.approx(2.0)

    def test_a_band_with_too_few_samples_integrates_to_zero(self):
        frequency = np.linspace(0.0, 1000.0, 11)  # 100 Hz spacing
        psd = np.ones_like(frequency)

        assert compute_band_power(frequency, psd, {"thin": (101.0, 199.0)})["thin"] == 0.0

    def test_unknown_ratio_band_is_reported(self):
        frequency = np.linspace(0.0, 100.0, 101)
        psd = np.ones_like(frequency)
        with pytest.raises(KeyError, match="unknown band"):
            compute_band_power(
                frequency, psd, {"a": (1.0, 10.0)}, ratios={"r": ("a", "missing")}
            )

    def test_zero_denominator_yields_nan(self):
        frequency = np.linspace(0.0, 1000.0, 1001)
        psd = np.zeros_like(frequency)

        powers = compute_band_power(
            frequency, psd, {"a": (10.0, 20.0), "b": (30.0, 40.0)},
            ratios={"r": ("a", "b")},
        )

        assert np.isnan(powers["r"])


class TestSpectrogram:
    def test_tracks_a_linear_chirp(self):
        from scipy.signal import chirp

        time = uniform_time()
        signal = chirp(time, f0=5e3, f1=5e4, t1=time[-1], method="linear")

        result = compute_spectrogram(time, signal, nperseg=4096, overlap=0.5)

        peak = result.frequency[np.argmax(result.magnitude, axis=0)]
        assert peak[0] == pytest.approx(5e3, rel=0.15)
        assert peak[-1] == pytest.approx(5e4, rel=0.05)
        assert np.all(np.diff(peak) >= -result.frequency[1])  # monotonically rising

    def test_time_axis_is_on_the_callers_timebase(self):
        offset = 0.26
        time = uniform_time(n=2**15) + offset
        signal = power_law_signal(-2.0, n=2**15)

        result = compute_spectrogram(time, signal, nperseg=2048)

        assert result.time[0] >= offset
        assert result.time[-1] <= time[-1]

    def test_window_duration_matches_an_equivalent_sample_count(self):
        time = uniform_time(n=2**15)
        signal = power_law_signal(-2.0, n=2**15)

        by_samples = compute_spectrogram(time, signal, nperseg=2500)
        by_seconds = compute_spectrogram(time, signal, window_duration=2500 / SAMPLE_RATE)

        np.testing.assert_allclose(by_samples.frequency, by_seconds.frequency)
        np.testing.assert_allclose(by_samples.magnitude, by_seconds.magnitude)

    def test_a_signal_shorter_than_one_window_yields_an_empty_result(self):
        time = uniform_time(n=100)
        signal = np.random.default_rng(0).standard_normal(100)

        result = compute_spectrogram(time, signal, nperseg=4096)

        assert result.time.size == 0
        assert result.frequency.size == 2049
        assert result.magnitude.shape == (2049, 0)

    def test_coordinates_are_deterministic(self):
        time = uniform_time(n=2**15)
        signal = power_law_signal(-2.0, n=2**15)

        first = compute_spectrogram(time, signal, nperseg=2048)
        second = compute_spectrogram(time, signal, nperseg=2048)

        np.testing.assert_array_equal(first.time, second.time)
        np.testing.assert_array_equal(first.frequency, second.frequency)
        np.testing.assert_array_equal(first.magnitude, second.magnitude)

    def test_rejects_both_window_specifications(self):
        time = uniform_time(n=4096)
        with pytest.raises(ValueError, match="not both"):
            compute_spectrogram(time, power_law_signal(-2.0, n=4096),
                                nperseg=256, window_duration=0.01)

    def test_rejects_an_out_of_range_overlap(self):
        time = uniform_time(n=4096)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            compute_spectrogram(time, power_law_signal(-2.0, n=4096),
                                nperseg=256, overlap=1.0)


class TestDiagnosticIndependence:
    def test_identical_samples_give_identical_results_whatever_the_diagnostic(self):
        """The API has no diagnostic parameter, so provenance cannot change results."""
        time = uniform_time(n=2**16)
        samples = power_law_signal(-2.0, n=2**16)

        # The same array, imagined as a Mirnov voltage, an SXR power and an
        # interferometer line density.
        results = [compute_psd(time, samples, nperseg=4096) for _ in range(3)]

        for other in results[1:]:
            np.testing.assert_array_equal(results[0].psd, other.psd)
            np.testing.assert_array_equal(results[0].frequency, other.frequency)

    def test_units_metadata_is_caller_supplied_and_never_inferred(self):
        time = uniform_time(n=4096)
        spectrum = compute_psd(time, power_law_signal(-2.0, n=4096), units="T**2/Hz")
        assert spectrum.units == "T**2/Hz"


class TestComposition:
    def test_analyze_runs_every_requested_stage(self):
        time = uniform_time()
        signal = broken_power_law_signal(-1.0, -3.0, 8_000.0)

        result = analyze_fluctuation_spectrum(
            time, signal,
            nperseg=8192,
            fit_ranges=[(1e3, 6e3), (1.2e4, 6e4)],
            bands={"low": (1e3, 6e3), "high": (1.2e4, 6e4)},
            ratios={"high_over_low": ("high", "low")},
            break_frequency=8_000.0,
            break_fit_range=(500.0, 6e4),
        )

        assert isinstance(result, FluctuationSpectrum)
        assert len(result.fits) == 2
        assert result.fits[0].alpha == pytest.approx(-1.0, abs=0.15)
        assert result.fits[1].alpha == pytest.approx(-3.0, abs=0.15)
        assert result.spectral_break.mode == "imposed"
        assert set(result.band_power) == {"low", "high", "high_over_low"}

    def test_analyze_without_options_is_just_a_psd(self):
        time = uniform_time(n=2**16)
        result = analyze_fluctuation_spectrum(time, power_law_signal(-2.0, n=2**16))

        assert result.fits == ()
        assert result.spectral_break is None
        assert result.band_power == {}

    def test_break_analysis_requires_an_explicit_fit_range(self):
        time = uniform_time(n=2**16)
        with pytest.raises(ValueError, match="break_fit_range"):
            analyze_fluctuation_spectrum(
                time, power_law_signal(-2.0, n=2**16), break_frequency=8e3
            )


class TestNoPhysicalInterpretation:
    """The library must ship no spectral-slope constants of its own."""

    def test_module_source_holds_no_slope_constants(self):
        source = Path(__file__).resolve().parents[1] / "vaft" / "process" / "fluctuation.py"
        text = source.read_text(encoding="utf-8")

        forbidden = re.compile(r"-\s*(5\s*/\s*3|8\s*/\s*3|1\.6+7?|2\.6+7?)\b")
        assert not forbidden.search(text), (
            "vaft/process/fluctuation.py must not hard-code reference spectral slopes; "
            "they are caller-supplied at the plotting layer."
        )
        assert "kolmogorov" not in text.lower()

    def test_results_carry_no_regime_labels(self):
        time = uniform_time(n=2**16)
        spectrum = compute_psd(time, power_law_signal(-2.0, n=2**16))
        fit = fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=(1e3, 5e4))

        # A fit reports numbers; naming the regime is the caller's job.
        assert not hasattr(fit, "regime")
        assert not hasattr(fit, "classification")
