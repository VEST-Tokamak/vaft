from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from ipywidgets import IntSlider, interact
from scipy import signal
from scipy.signal import coherence, csd, find_peaks, savgol_filter

from vaft.compat import cumtrapz_compat
from vaft.database import raw as raw_db
from vaft.process import define_baseline, subtract_baseline

# Naming convention for function name: {diagnostics_name}_{processing_quantity}


@dataclass(frozen=True)
class VestMagneticsProcessingConfig:
    """Default VEST magnetics processing settings used by legacy `vfit_equilibrium_magnetics`.

    The values intentionally preserve the long-running VEST EFIT input workflow
    while making the knobs explicit for reproducibility and parameter scans.
    Shots 41446--41451 and shots from 41660 onward use indices 6500--9000
    with a 5000-sample probe baseline. Other shots use indices 6000--8500
    with an 8500-sample probe baseline. These are legacy acquisition-era
    policies, not automatic signal-quality decisions.
    """

    time_start: float = 0.0
    time_end: float = 0.99996
    sample_count: int = 25_000
    fast_sample_rate: float = 250_000.0
    lowpass_cutoff: float = 2_500.0
    lowpass_taps: int = 251
    default_index_start: int = 6000
    default_index_end: int = 8500
    default_probe_baseline_end: int = 8500
    late_shot_min: int = 41660
    transient_shot_min: int = 41446
    transient_shot_max: int = 41451
    late_index_start: int = 6500
    late_index_end: int = 9000
    late_probe_baseline_end: int = 5000
    flux_baseline_first_start: int = 3499
    flux_baseline_first_end: int = 5000
    flux_baseline_second_start: int = 11999
    flux_baseline_second_end: int = 15000
    flux_baseline_late_start: int = 5999
    flux_baseline_late_end: int = 7000
    flux_baseline_late_loop_numbers: tuple[int, ...] = (9, 10, 11)
    calibration_mode: str = "divide"
    flux_output_per_radian: bool = True

    def timebase(self) -> np.ndarray:
        return np.linspace(self.time_start, self.time_end, self.sample_count)

    def window_for_shot(self, shot: int) -> tuple[int, int, int]:
        if self.transient_shot_min <= shot <= self.transient_shot_max or shot >= self.late_shot_min:
            return self.late_index_start, self.late_index_end, self.late_probe_baseline_end
        return self.default_index_start, self.default_index_end, self.default_probe_baseline_end


DEFAULT_VEST_MAGNETICS_PROCESSING = VestMagneticsProcessingConfig()


@dataclass(frozen=True)
class MirnovSpectrogramResult:
    """Time-frequency result for one Mirnov waveform."""

    time: np.ndarray
    frequency: np.ndarray
    magnitude: np.ndarray


@dataclass(frozen=True)
class ToroidalModeResult:
    """Cross-phase toroidal mode-number result."""

    frequency: np.ndarray
    n: np.ndarray
    power: np.ndarray
    phase: np.ndarray
    spectrum_frequency: np.ndarray
    cross_power: np.ndarray
    peak_indices: np.ndarray
    n_raw: np.ndarray
    n_rounded: np.ndarray
    coherence: np.ndarray


@dataclass(frozen=True)
class ToroidalPhaseModeFit:
    """Wrapped toroidal phase fit for one fluctuation frequency."""

    frequency: float
    n: int
    intercept: float
    rms_error: float
    phase: np.ndarray
    fitted_phase: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class ToroidalPhaseFitResult:
    """Mode-line fits from a selected time slice."""

    time: float
    toroidal_angle: np.ndarray
    modes: tuple[ToroidalPhaseModeFit, ...]
    candidate_n: np.ndarray


def _calibrated_signal(values: np.ndarray, calibration: float, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "divide":
        return values / calibration
    if mode == "multiply":
        return values * calibration
    raise ValueError(f"Unsupported VEST magnetics calibration mode: {mode}")


def _linear_baseline(time_axis: np.ndarray, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    valid = indices[(indices >= 0) & (indices < values.size)]
    if valid.size < 2:
        return np.zeros(values.size, dtype=float)
    return np.polyval(np.polyfit(time_axis[valid], values[valid], 1), time_axis)


def vest_magnetics_time_window(shot: int, config: VestMagneticsProcessingConfig | None = None) -> np.ndarray:
    """Return the VEST MD output time window for a shot."""
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    index_start, index_end, _ = cfg.window_for_shot(int(shot))
    return cfg.timebase()[index_start : index_end + 1]


def vest_b_field_pol_probe_legacy(
    time: np.ndarray,
    raw: np.ndarray,
    calibration: float,
    *,
    shot: int,
    config: VestMagneticsProcessingConfig | None = None,
) -> np.ndarray:
    """Process one VEST poloidal B-field probe using the legacy EFIT workflow."""
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    _, _, baseline_end = cfg.window_for_shot(int(shot))
    time = np.asarray(time, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if time.size <= 1 or raw.size <= 1:
        raise ValueError("VEST poloidal-field processing requires at least two samples")

    lowpass = signal.firwin(
        cfg.lowpass_taps,
        cfg.lowpass_cutoff,
        pass_zero="lowpass",
        fs=cfg.fast_sample_rate,
    )
    filtered = signal.lfilter(lowpass, 1, raw)
    calibrated = _calibrated_signal(filtered, float(calibration), cfg.calibration_mode)
    integrated = -cumtrapz_compat(calibrated, x=time, initial=0)
    baseline = _linear_baseline(time, integrated, np.arange(min(baseline_end, integrated.size)))
    return integrated - baseline


def vest_flux_loop_legacy(
    time: np.ndarray,
    raw: np.ndarray,
    calibration: float,
    *,
    flux_loop_number: int,
    config: VestMagneticsProcessingConfig | None = None,
) -> np.ndarray:
    """Process one VEST flux loop using the legacy EFIT workflow."""
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    time = np.asarray(time, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if time.size <= 1 or raw.size <= 1:
        raise ValueError("VEST flux-loop processing requires at least two samples")

    calibrated = _calibrated_signal(raw, float(calibration), cfg.calibration_mode)
    integrated = -cumtrapz_compat(calibrated, x=time, initial=0)
    if cfg.flux_output_per_radian:
        integrated = integrated / (2 * np.pi)

    if int(flux_loop_number) in cfg.flux_baseline_late_loop_numbers:
        baseline_indices = np.arange(cfg.flux_baseline_late_start, min(cfg.flux_baseline_late_end, integrated.size))
    else:
        first = np.arange(cfg.flux_baseline_first_start, min(cfg.flux_baseline_first_end, integrated.size))
        second = np.arange(cfg.flux_baseline_second_start, min(cfg.flux_baseline_second_end, integrated.size))
        baseline_indices = np.concatenate((first, second))
    baseline = _linear_baseline(time, integrated, baseline_indices)
    return integrated - baseline


def vest_equilibrium_magnetics_signals(
    shot: int,
    channels: Sequence[dict],
    loader: Callable[[int, int], tuple[np.ndarray, np.ndarray] | None],
    *,
    indices: Sequence[int] | None = None,
    config: VestMagneticsProcessingConfig | None = None,
    allow_missing: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Process VEST MD channels into flux-loop and B-probe waveforms."""
    cfg = config or DEFAULT_VEST_MAGNETICS_PROCESSING
    channel_rows = list(channels)
    if indices is not None:
        channel_rows = [channel_rows[int(index)] for index in indices]

    magnetics_time = vest_magnetics_time_window(shot, cfg)
    data_flux_loops: list[np.ndarray] = []
    data_probes: list[np.ndarray] = []
    flux_loop_counter = 0

    for channel in channel_rows:
        field_code = int(channel["field_code"])
        calibration = float(channel["calibration"])
        kind = str(channel["kind"])
        loaded = loader(int(shot), field_code)

        if kind == "flux_loop":
            flux_loop_counter += 1

        try:
            source_time, source_data = raw_db.require_signal(
                loaded,
                shot=shot,
                field=field_code,
                signal_name=str(channel.get("name", kind)),
            )
        except raw_db.RawSignalUnavailableError:
            if not allow_missing:
                raise
            missing = np.array([], dtype=float)
            if kind == "b_field_pol_probe":
                data_probes.append(missing)
            else:
                data_flux_loops.append(missing)
            continue
        if kind == "b_field_pol_probe":
            processed_full = vest_b_field_pol_probe_legacy(source_time, source_data, calibration, shot=shot, config=cfg)
            processed = np.interp(magnetics_time, source_time, processed_full)
        else:
            processed_full = vest_flux_loop_legacy(
                source_time,
                source_data,
                calibration,
                flux_loop_number=flux_loop_counter,
                config=cfg,
            )
            processed = np.interp(magnetics_time, source_time, processed_full)

        if kind == "b_field_pol_probe":
            data_probes.append(processed)
        else:
            data_flux_loops.append(processed)

    return magnetics_time, data_flux_loops, data_probes


def _firwin_order(sample_rate: float) -> int:
    order = int(float(sample_rate) * 1e-3)
    return order + 1 if order % 2 == 0 else order


def mirnov_preprocess_signal(
    data: np.ndarray,
    *,
    sample_rate: float = 250_000.0,
    high_pass_cutoff: float | None = 2_000.0,
    low_pass_cutoff: float | None = 90_000.0,
    amplifier_gain: float = 1.0,
    filter_order: int | None = None,
) -> np.ndarray:
    """Apply the MATLAB ``vest_osc`` Mirnov preprocessing chain."""
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return values.copy()
    if amplifier_gain == 0:
        raise ValueError("amplifier_gain must be non-zero.")

    processed = values - float(np.nanmean(values))
    order = int(filter_order) if filter_order is not None else _firwin_order(sample_rate)
    nyquist = 0.5 * float(sample_rate)

    for cutoff, pass_zero in ((high_pass_cutoff, False), (low_pass_cutoff, True)):
        if cutoff is None:
            continue
        cutoff = float(cutoff)
        if cutoff <= 0 or cutoff >= nyquist:
            continue
        taps = signal.firwin(order, cutoff, fs=sample_rate, pass_zero=pass_zero, window="hann")
        if processed.size > 3 * taps.size:
            processed = signal.filtfilt(taps, 1.0, processed)
        else:
            processed = signal.lfilter(taps, 1.0, processed)

    return processed / float(amplifier_gain)


def _sample_rate_from_time(time: np.ndarray, default: float = 250_000.0) -> float:
    if time.size < 2:
        return float(default)
    dt = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        return float(default)
    return 1.0 / dt


def mirnov_spectrogram(
    time: np.ndarray,
    data: np.ndarray,
    *,
    sample_rate: float | None = None,
    window_size: int = 500,
    time_resolution: int = 1,
    time_range: tuple[float, float] | None = None,
) -> MirnovSpectrogramResult:
    """Compute the manual Hann-window FFT spectrogram used by ``vest_mirnov``."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(data, dtype=float)
    if time.size != values.size:
        raise ValueError("time and data must have the same length.")
    if time.size == 0:
        return MirnovSpectrogramResult(np.array([]), np.array([]), np.empty((0, 0)))

    window_size = int(window_size)
    if window_size <= 1 or window_size % 2:
        raise ValueError("window_size must be an even integer greater than 1.")
    step = max(1, int(time_resolution))
    half_window = window_size // 2

    if time_range is None:
        first = half_window - 1
        last = values.size - half_window - 1
    else:
        first = int(np.searchsorted(time, float(time_range[0]), side="left"))
        last = int(np.searchsorted(time, float(time_range[1]), side="left"))
    centers = np.arange(first, last + 1, step, dtype=int)
    centers = centers[(centers - half_window + 1 >= 0) & (centers + half_window + 1 <= values.size)]
    if centers.size == 0:
        frequencies = np.fft.rfftfreq(window_size, d=1.0 / (sample_rate or _sample_rate_from_time(time)))
        return MirnovSpectrogramResult(np.array([]), frequencies, np.empty((frequencies.size, 0)))

    windows = np.empty((window_size, centers.size), dtype=float)
    for column, center in enumerate(centers):
        windows[:, column] = values[center - half_window + 1 : center + half_window + 1]

    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time)
    tapered = windows * signal.windows.hann(window_size)[:, np.newaxis]
    spectrum = np.fft.rfft(tapered, axis=0)
    magnitude = 2.0 * np.abs(spectrum / window_size)
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / fs)
    return MirnovSpectrogramResult(time[centers], frequencies, magnitude)


def toroidal_mode_analysis(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    *,
    sample_rate: float = 250_000.0,
    phase_geometry: float = np.pi / 6,
    peak_threshold: float = 0.1,
    sensor_count: int = 4,
    nperseg: int | None = None,
) -> ToroidalModeResult:
    """Estimate toroidal mode number from two toroidally separated Mirnov traces."""
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    if a.size != b.size:
        raise ValueError("signal_a and signal_b must have the same length.")
    if a.size < 2:
        empty = np.array([])
        return ToroidalModeResult(empty, empty, empty, empty, empty, empty, empty.astype(int), empty, empty, empty)
    if phase_geometry == 0:
        raise ValueError("phase_geometry must be non-zero.")

    segment = min(a.size, int(nperseg) if nperseg is not None else 256)
    frequencies, cross_power = csd(a, b, fs=sample_rate, nperseg=segment)
    _, coherence_values = coherence(a, b, fs=sample_rate, nperseg=segment)
    phase = np.angle(cross_power)
    n_raw = phase / float(phase_geometry)
    n_rounded = np.round(n_raw)

    power_abs = np.abs(cross_power)
    if power_abs.size == 0 or float(np.max(power_abs)) == 0.0:
        peak_indices = np.array([], dtype=int)
    else:
        peaks, _ = find_peaks(power_abs, height=float(peak_threshold) * float(np.max(power_abs)))
        coherence_threshold = np.tanh(1.96 / np.sqrt(max(1.0, 2.0 * float(sensor_count) - 2.0)))
        peak_indices = peaks[coherence_values[peaks] > coherence_threshold]

    total_power = float(np.sum(power_abs))
    relative_power = power_abs[peak_indices] / total_power if total_power > 0 else np.zeros(peak_indices.size)
    return ToroidalModeResult(
        frequency=frequencies[peak_indices],
        n=n_rounded[peak_indices],
        power=relative_power,
        phase=phase[peak_indices],
        spectrum_frequency=frequencies,
        cross_power=cross_power,
        peak_indices=peak_indices,
        n_raw=n_raw,
        n_rounded=n_rounded,
        coherence=coherence_values,
    )


def _wrap_phase_radians(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _fit_wrapped_toroidal_n(
    toroidal_angle: np.ndarray,
    phase: np.ndarray,
    candidate_n: np.ndarray,
) -> tuple[int, float, float, np.ndarray]:
    best_n = 0
    best_intercept = 0.0
    best_error = np.inf
    best_fit = np.zeros_like(phase)
    for n_value in candidate_n:
        residual_offset = phase + float(n_value) * toroidal_angle
        intercept = float(np.angle(np.mean(np.exp(1j * residual_offset))))
        fitted = _wrap_phase_radians(intercept - float(n_value) * toroidal_angle)
        residual = _wrap_phase_radians(phase - fitted)
        rms_error = float(np.sqrt(np.mean(residual**2)))
        if rms_error < best_error:
            best_n = int(n_value)
            best_intercept = intercept
            best_error = rms_error
            best_fit = fitted
    return best_n, best_intercept, best_error, best_fit


def toroidal_phase_fit_at_time(
    time: np.ndarray,
    signals: np.ndarray,
    toroidal_angle: np.ndarray,
    *,
    center_time: float,
    sample_rate: float | None = None,
    window_size: int = 500,
    frequencies: Sequence[float] | None = None,
    num_modes: int = 2,
    candidate_n: Sequence[int] = tuple(range(-6, 7)),
    peak_threshold: float = 0.1,
) -> ToroidalPhaseFitResult:
    """Fit wrapped toroidal ``n`` mode lines at one selected time.

    ``signals`` must have shape ``(n_channels, n_time)``. If ``frequencies`` is
    omitted, dominant frequencies are selected from the channel-averaged FFT
    magnitude in the selected window.
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(signals, dtype=float)
    angles = np.asarray(toroidal_angle, dtype=float)
    candidates = np.asarray(tuple(candidate_n), dtype=int)

    if data.ndim != 2:
        raise ValueError("signals must have shape (n_channels, n_time).")
    if data.shape[0] != angles.size:
        raise ValueError("toroidal_angle length must match the number of signal channels.")
    if data.shape[1] != time.size:
        raise ValueError("signals time dimension must match time length.")
    if data.shape[0] < 2:
        raise ValueError("At least two toroidal channels are required.")
    if candidates.size == 0:
        raise ValueError("candidate_n must contain at least one integer.")

    window_size = int(window_size)
    if window_size <= 1 or window_size % 2:
        raise ValueError("window_size must be an even integer greater than 1.")
    if time.size < window_size:
        raise ValueError("time array is shorter than window_size.")

    center_index = int(np.argmin(np.abs(time - float(center_time))))
    half_window = window_size // 2
    start = center_index - half_window
    stop = start + window_size
    if start < 0:
        start = 0
        stop = window_size
    if stop > time.size:
        stop = time.size
        start = stop - window_size

    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time)
    window = signal.windows.hann(window_size)
    windowed = data[:, start:stop] * window[np.newaxis, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    spectrum_frequency = np.fft.rfftfreq(window_size, d=1.0 / fs)
    magnitude = np.abs(spectrum)

    if frequencies is None:
        average_magnitude = np.mean(magnitude, axis=0)
        average_magnitude[0] = 0.0
        if np.max(average_magnitude) <= 0:
            selected_indices = np.array([], dtype=int)
        else:
            peaks, props = find_peaks(average_magnitude, height=float(peak_threshold) * float(np.max(average_magnitude)))
            if peaks.size == 0:
                selected_indices = np.array([int(np.argmax(average_magnitude))], dtype=int)
            else:
                order = np.argsort(props["peak_heights"])[::-1]
                selected_indices = peaks[order[: max(1, int(num_modes))]]
    else:
        selected_indices = np.array(
            [int(np.argmin(np.abs(spectrum_frequency - float(freq)))) for freq in frequencies],
            dtype=int,
        )

    modes: list[ToroidalPhaseModeFit] = []
    for frequency_index in selected_indices:
        complex_values = spectrum[:, frequency_index]
        phase = _wrap_phase_radians(np.angle(complex_values))
        amplitude = np.abs(complex_values)
        best_n, intercept, rms_error, fitted = _fit_wrapped_toroidal_n(angles, phase, candidates)
        modes.append(
            ToroidalPhaseModeFit(
                frequency=float(spectrum_frequency[frequency_index]),
                n=best_n,
                intercept=intercept,
                rms_error=rms_error,
                phase=phase,
                fitted_phase=fitted,
                amplitude=amplitude,
            )
        )

    modes.sort(key=lambda item: float(np.mean(item.amplitude)), reverse=True)
    return ToroidalPhaseFitResult(
        time=float(time[center_index]),
        toroidal_angle=angles,
        modes=tuple(modes),
        candidate_n=candidates,
    )

def rogowski_coil_ip(
    time,
    rogowski_raw,
    flux_loop_raw,
    flux_loop_gain=11,
    effective_vessel_res=5.8e-4,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    smooth_window=10
):
    """
    Compute the plasma current from Rogowski coil and flux loop signals.

    The function:
    1. Defines baseline indices near baseline_onset and baseline_offset.
    2. Fits a baseline (using baseline_type) and subtracts it from both rogowski_raw and flux_loop_raw.
    3. Converts the flux_loop_raw to a "reference current" by multiplying by flux_loop_gain.
    4. Optionally smooths the flux loop reference with a Savitzky-Golay filter (smooth_window).
    5. Subtracts flux_loop_ref from rogowski_raw to get the final Ip.
    6. If the resulting signal is predominantly negative, flips its sign.

    Parameters
    ----------
    time : np.ndarray
        Time array for the signals.
    rogowski_raw : np.ndarray
        Raw Rogowski coil data array (plasma current sensor).
    flux_loop_raw : np.ndarray
        Raw flux loop array (used as a reference).
    flux_loop_gain : float, optional
        Gain factor/multiplier for flux loop data. Default is 11.
    effective_vessel_res : float, optional
        Effective vessel resistance or mutual inductance factor. (Currently not used in baseline.)
    baseline_onset : float, optional
        Time (in seconds) at which signals begin to deviate; used for baseline definition.
    baseline_offset : float, optional
        Time (in seconds) at which signals return to baseline; used for baseline definition.
    baseline_type : {'linear','quadratic','spline','exp'}, optional
        Type of baseline fitting to use. Default 'linear'.
    baseline_onset_window : int, optional
        Number of samples before baseline_onset to include in the baseline fit. Default 500.
    baseline_offset_window : int, optional
        Number of samples after baseline_offset to include in the baseline fit. Default 100.
    smooth_window : int, optional
        Window size for Savitzky-Golay smoothing of flux_loop reference. Default 10 (must be odd).

    Returns
    -------
    time : np.ndarray
        Same time array as input.
    ip : np.ndarray
        Final processed plasma current signal.
    """
    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)

    # Define baseline indices
    baseline_indices_rogowski = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )
    baseline_indices_flux = baseline_indices_rogowski  # same region, typically

    # Subtract baseline from rogowski
    rogowski_corr, rogowski_baseline = subtract_baseline(
        time, rogowski_raw, baseline_indices_rogowski, fitting_opt=baseline_type
    )

    # Subtract baseline from flux loop
    flux_corr, flux_baseline = subtract_baseline(
        time, flux_loop_raw, baseline_indices_flux, fitting_opt=baseline_type
    )

    # Convert flux loop signal to current reference
    # For example: flux_ref = flux_corr * (flux_loop_gain / mutual_inductance)
    # We'll do the simplest version: flux_corr * flux_loop_gain
    flux_ref = flux_corr * flux_loop_gain

    # Smooth the flux loop reference (Savitzky-Golay) if smooth_window > 2
    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1

    flux_ref_smooth = savgol_filter(flux_ref, smooth_window, polyorder=1)

    # Final plasma current
    ip = rogowski_corr - flux_ref_smooth

    # If absolute negative peak is larger than the positive peak, invert
    if abs(np.min(ip)) > abs(np.max(ip)):
        ip = -ip

    return time, ip


def b_field_pol_probe_field(
    time,
    raw,
    gain,
    lowpass_param,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    plot_opt=False,
):
    """
    Process B-field poloidal probe data with gain applied first.
    """
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    m, n = raw.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match the number of signals (columns in raw).")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples (rows in raw).")

    # Apply gain at the start
    raw = raw * gain

    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)

    baseline_indices = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )

    # Apply low-pass filter
    filtered_raw = signal.lfilter(lowpass_param, [1.0], raw, axis=0)

    # Integrate to get flux (negative sign if your system defines it so)
    integrated_flux = -cumtrapz_compat(filtered_raw, x=time, initial=0, axis=0)

    # Subtract baseline for each column
    field = np.empty_like(integrated_flux)
    baselines = np.empty_like(integrated_flux)
    for i in range(n):
        flux_corrected, baseline = subtract_baseline(
            time, integrated_flux[:, i], baseline_indices, fitting_opt=baseline_type
        )
        field[:, i] = flux_corrected
        baselines[:, i] = baseline

    if plot_opt:
        def interactive_plot(index):
            return _plot_signal_processing_panels(
                time,
                title=f"B-field Signal Processing: Index {index}\n"
                      f"Baseline: {baseline_type}",
                raw_traces=(
                    ("Raw (gain applied)", raw[:, index]),
                    ("Filtered Signal", filtered_raw[:, index]),
                ),
                corrected_traces=(
                    ("Integrated Signal", integrated_flux[:, index]),
                    ("Baseline", baselines[:, index]),
                    ("Baseline-Corrected Signal", field[:, index]),
                ),
            )

        interact(interactive_plot, index=IntSlider(min=0, max=n-1, step=1, value=0))

    return raw, filtered_raw, integrated_flux, field, baselines

def flux_loop_flux(
    time,
    raw,
    gain,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    plot_opt=False,
):
    """
    Process flux loop data for multiple signals.

    Steps:
    1. Integrate the raw data (dividing by gain) to obtain flux, 
       including a negative sign and 1/(2*pi) factor if desired.
    2. Remove baseline offsets using define_baseline + subtract_baseline.

    Parameters
    ----------
    time : np.ndarray, shape [m]
        Time array for the flux loop signals.
    raw : np.ndarray, shape [m x n]
        Measured raw data from multiple flux loops. Each column is a separate signal.
    gain : np.ndarray, shape [n]
        Gain factor for each flux loop signal. Must match number of columns in `raw`.
    baseline_onset : float
        Time in seconds to define the start of the baseline region.
    baseline_offset : float
        Time in seconds to define the end of the baseline region.
    baseline_type : {'linear','quadratic','spline','exp'}
        Type of baseline fitting to use.
    baseline_onset_window : int
        Number of samples before baseline_onset to include in the baseline fit.
    baseline_offset_window : int
        Number of samples after baseline_offset to include in the baseline fit.
    plot_opt : bool
        Whether to plot the results interactively.

    Returns
    -------
    time : np.ndarray, shape [m]
        Same as input.
    processed_data : np.ndarray, shape [m x n]
        Integrated, baseline-corrected flux data for each loop.
    baselines : np.ndarray, shape [m x n]
        Baseline values for each signal.
    """
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    m, n = raw.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match number of signals.")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples.")

    # Apply gain at the start
    raw = raw * gain

    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)
    baseline_indices = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )

    # Integrate flux loop data for each signal
    # - sign if that is convention, also / (2*pi)
    integrated_data = -cumtrapz_compat(raw, x=time, initial=0, axis=0) / (2 * np.pi)

    # Remove offset for each signal
    processed_data = np.empty_like(integrated_data)
    baselines = np.empty_like(integrated_data)
    for i in range(n):
        flux_corrected, baseline = subtract_baseline(
            time, integrated_data[:, i], baseline_indices, fitting_opt=baseline_type
        )
        processed_data[:, i] = flux_corrected
        baselines[:, i] = baseline

    if plot_opt:
        def interactive_plot(index):
            return _plot_signal_processing_panels(
                time,
                title=f"Flux Loop Signal Processing: Index {index}\n"
                      f"Baseline: {baseline_type}",
                raw_traces=(("Raw (gain applied)", raw[:, index]),),
                corrected_traces=(
                    ("Integrated Signal", integrated_data[:, index]),
                    ("Baseline", baselines[:, index]),
                    ("Baseline-Corrected Signal", processed_data[:, index]),
                ),
            )

        interact(interactive_plot, index=IntSlider(min=0, max=n-1, step=1, value=0))

    return time, processed_data, baselines


# def toroidal_mode_analysis(
#     time_vector, 
#     signal_matrix, 
#     toroidal_angles, 
#     time_points, 
#     window_size=1000, 
#     thres_peak=0.1, 
#     plot_opt=False,
#     nperseg=256,
#     coherence_q=4
# ):
#     """
#     Compute coherence, phase, toroidal mode number, and relative power using the first signal as reference.
    
#     Parameters
#     ----------
#     time_vector : np.ndarray
#         Time axis vector (e.g., 0~1s, 250kHz sampling -> length 250000)
#     signal_matrix : np.ndarray
#         2D array of shape (num_signals x num_samples).
#         Each row represents a different probe(channel), each column represents a time sample.
#     toroidal_angles : np.ndarray
#         Toroidal angles (in radians) corresponding to each probe(row). Length num_signals.
#     time_points : list or np.ndarray
#         Time indices at which to perform analysis (e.g., [1000, 2000, 3000, ...])
#     window_size : int
#         Window size determining how many samples to analyze around each time_point.
#         (default 1000 -> ±500 points)
#     thres_peak : float
#         Minimum height ratio for peak detection relative to maximum spectrum value (default 0.1).
#     plot_opt : bool
#         If True, displays a simple phase plot with slider.
#     nperseg : int
#         nperseg value to use for csd, coherence calculations (default 256).
#     coherence_q : int
#         q value used for coherence threshold calculation. (default 4)
#         Generalizes the original tanh(1.96 / sqrt(2*q-2)) formula.
    
#     Returns
#     -------
#     results : dict
#         {
#           "time": [t1, t2, ...],             # Actual analysis times (seconds)
#           "coherence": [...],               # Array of coherence values for valid peaks for [num_signals-1] channels at each time_point
#           "phase": [...],                   # Array of phase values
#           "mode_number": [...],             # Array of mode numbers
#           "frequencies": [...],             # Array of peak frequencies
#           "power": [...]                    # Array of relative peak powers
#         }
#     """

#     num_signals, num_samples = signal_matrix.shape
#     if len(toroidal_angles) != num_signals:
#         raise ValueError("The number of toroidal angles must match the number of signals.")
    
#     # 샘플링 주파수(Hz)
#     f_sample = 1.0 / np.mean(np.diff(time_vector))
    
#     # 코히런스 임계값(원본 코드 아이디어)
#     coherence_threshold = np.tanh(1.96 / np.sqrt(2 * coherence_q - 2))
    
#     results = {
#         "time": [],
#         "coherence": [],
#         "phase": [],
#         "mode_number": [],
#         "frequencies": [],
#         "power": []
#     }
    
#     all_time_results = []  # 플롯에서 슬라이더로 접근 가능하도록 저장
    
#     half_win = window_size // 2
    
#     for t_idx in time_points:
#         # 창 범위 확인
#         if t_idx < half_win or t_idx >= num_samples - half_win:
#             continue
        
#         window_start = t_idx - half_win
#         window_end   = t_idx + half_win
        
#         ref_signal = signal_matrix[0, window_start:window_end]
#         ref_angle  = toroidal_angles[0]
        
#         time_results = {
#             "coherence": [],
#             "phase": [],
#             "mode_number": [],
#             "frequencies": [],
#             "power": []
#         }
        
#         # 각 프로브(i=1~num_signals-1)에 대해
#         for i in range(1, num_signals):
#             signal_i = signal_matrix[i, window_start:window_end]
            
#             # Cross-spectral density
#             f, pxy = csd(ref_signal, signal_i, fs=f_sample, nperseg=nperseg)
#             magnitude = np.abs(pxy)
            
#             # Coherence
#             _, cxy = coherence(ref_signal, signal_i, fs=f_sample, nperseg=nperseg)
            
#             # 피크 찾기: 크기가 thres_peak * max(magnitude) 이상인 피크
#             peaks, peak_props = find_peaks(
#                 magnitude, 
#                 height=thres_peak * np.max(magnitude)
#             )
#             # 크기 기준 내림차순 정렬
#             peak_heights = peak_props["peak_heights"]
#             desc_order = np.argsort(peak_heights)[::-1]
#             peaks = peaks[desc_order]
            
#             # 코히런스 필터
#             valid_peaks = []
#             for pk in peaks:
#                 if cxy[pk] > coherence_threshold:
#                     valid_peaks.append(pk)
#             valid_peaks = np.array(valid_peaks, dtype=int)
            
#             if len(valid_peaks) > 0:
#                 # 위상(각 유효 피크에서)
#                 phase_vals = np.angle(pxy[valid_peaks])
                
#                 # 모드 번호: (phase / Δphi)
#                 delta_phi = toroidal_angles[i] - ref_angle
#                 n_raw = phase_vals / delta_phi
#                 n_rounded = np.round(n_raw).astype(int)
                
#                 # 상대 파워: 각 피크의 |pxy| / 전체 스펙트럼 |pxy| 합
#                 total_power = np.sum(magnitude)
#                 power_vals  = magnitude[valid_peaks] / total_power
                
#                 time_results["coherence"].append(cxy[valid_peaks])
#                 time_results["phase"].append(phase_vals)
#                 time_results["mode_number"].append(n_rounded)
#                 time_results["frequencies"].append(f[valid_peaks])
#                 time_results["power"].append(power_vals)
#             else:
#                 # 유효 피크가 없으면 빈 배열 저장
#                 time_results["coherence"].append(np.array([]))
#                 time_results["phase"].append(np.array([]))
#                 time_results["mode_number"].append(np.array([]))
#                 time_results["frequencies"].append(np.array([]))
#                 time_results["power"].append(np.array([]))
        
#         # 전체 결과에 추가
#         results["time"].append(time_vector[t_idx])
#         results["coherence"].append(time_results["coherence"])
#         results["phase"].append(time_results["phase"])
#         results["mode_number"].append(time_results["mode_number"])
#         results["frequencies"].append(time_results["frequencies"])
#         results["power"].append(time_results["power"])
        
#         all_time_results.append(time_results)
    
#     if plot_opt:
#         fig, ax = plt.subplots()
#         plt.subplots_adjust(bottom=0.25)

#         def update_plot(idx):
#             ax.clear()
#             time_idx = time_points[idx]
#             time_result = all_time_results[idx]

#             # Plot reference point
#             ax.scatter([toroidal_angles[0]], [0], marker='o', color='red',
#                        label='Reference (0°)', s=100)

#             # Plot phase differences relative to reference
#             for i in range(num_signals - 1):
#                 phases = time_result["phase"][i]
#                 if len(phases) > 0:
#                     # 여러 피크가 있을 수 있으나 여기서는 평균값만 예시로 표시
#                     ax.scatter([toroidal_angles[i+1]], [np.mean(phases)], 
#                                marker='o', label=f'Probe {i+1}')

#             # 대략적 모드번호 피팅 예시 (평균 모드 사용)
#             if any(len(mn) > 0 for mn in time_result["mode_number"]):
#                 valid_modes = [np.mean(mn) for mn in time_result["mode_number"] if len(mn) > 0]
#                 if valid_modes:
#                     mean_mode = np.mean(valid_modes)
#                     theta = np.linspace(0, 2*np.pi, 100)
#                     ax.plot(theta, mean_mode * theta, 'r--', label=f'n={mean_mode:.1f}')

#             ax.set_title(f'Time: {time_vector[time_idx]:.5f} s')
#             ax.set_xlabel('Toroidal Angle (rad)')
#             ax.set_ylabel('Phase Difference (rad)')
#             ax.set_ylim(-np.pi, np.pi)
#             ax.set_xlim(0, 2*np.pi)
#             ax.legend()
#             ax.grid(True)
#             plt.draw()

#         ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
#         slider = Slider(ax_slider, "Time Index", 0, len(time_points) - 1, 
#                         valinit=0, valstep=1)
#         slider.on_changed(lambda val: update_plot(int(val)))
#         update_plot(0)
#         plt.show()

#     return results


def _plot_signal_processing_panels(time, *, title, raw_traces, corrected_traces):
    """Render the raw/corrected signal-processing panels through ``vaft.plot``.

    Processing owns the numerics; rendering is delegated so no Matplotlib code
    lives in this namespace (issue #63).
    """
    from vaft.plot import LineSeries, Panels, Series, render_panels

    def _panel(traces, y_label, panel_title=""):
        return LineSeries(
            series=tuple(
                Series(x=time, y=values, label=label, style={"alpha": 0.7})
                for label, values in traces
            ),
            x_label="Time", x_unit="s", y_label=y_label, title=panel_title,
        )

    return render_panels(
        Panels(
            models=(
                _panel(raw_traces, "Signal", title),
                _panel(corrected_traces, "Flux"),
            ),
            share_x=True,
        ),
        figsize=(10, 8),
        show=True,
    )
