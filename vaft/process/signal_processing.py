import warnings
from collections.abc import Sequence

import numpy as np
import scipy.signal as scipy_signal
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit


__all__ = [
    "SignalRepairError",
    "butterworth_bandpass",
    "butterworth_lowpass",
    "detect_active_window",
    "detect_clipped_samples",
    "detrend_moving_average",
    "define_baseline",
    "exp_baseline",
    "is_signal_active",
    "line_average_density",
    "linear_baseline",
    "process_signal",
    "quadratic_baseline",
    "repair_clipped_interval",
    "signal_on_offset",
    "smooth",
    "subtract_baseline",
    "vest_coil_current_noise_reduction",
    # Direct compatibility imports remain available for one deprecation cycle.
    "VEST_CoilCurrentNoiseReduction",
    "signal_onoffset",
    "vfit_signal_start_end",
    "vfit_signal_startend",
]


class SignalRepairError(ValueError):
    """Raised when a clipped waveform cannot be defensibly reconstructed.

    Reconstructing a saturated interval is an interpolation, so it is only
    valid where surrounding unsaturated samples actually constrain it.
    Raising beats returning a fabricated waveform that looks like data.
    """


def detect_clipped_samples(data, *, clip_values, tolerance: float) -> np.ndarray:
    """Return a boolean mask of samples sitting at an acquisition limit.

    ``clip_values`` is a single signed level or a sequence of them, and the
    mask is their union: a sample is saturated when it lies within
    ``tolerance`` of *any* supplied level. Real acquisition hardware rails on
    both sides and rarely symmetrically -- VEST's diamagnetic Rogowski channel
    is a signed 16-bit ADC over +/-5 V, so its rails are exactly ``-5.0`` and
    ``5 * 32767 / 32768`` (see `vest.yaml`, issue #285).

    Detecting every level in one pass is not a convenience. Where a waveform
    oscillates hard enough to hit both rails within a few samples, repairing
    one rail at a time would fit the reconstruction through samples still
    pinned at the other one.
    """
    values = np.asarray(data, dtype=float)
    levels = np.atleast_1d(np.asarray(clip_values, dtype=float)).reshape(-1)
    if levels.size == 0:
        raise SignalRepairError("`clip_values` must contain at least one acquisition limit.")
    if not np.all(np.isfinite(levels)):
        raise SignalRepairError(f"`clip_values` must be finite, got {clip_values!r}.")
    width = float(tolerance)
    if not np.isfinite(width) or width <= 0.0:
        raise SignalRepairError(f"`tolerance` must be a positive finite width, got {tolerance!r}.")

    saturated = np.zeros(values.shape, dtype=bool)
    for level in levels:
        saturated |= np.abs(values - level) < width
    return saturated


def repair_clipped_interval(
    time,
    data,
    *,
    clip_value: float | Sequence[float],
    tolerance: float,
    min_support: int = 4,
    return_mask: bool = False,
):
    """Reconstruct samples saturated at an acquisition limit by interpolation.

    Samples within ``tolerance`` of ``clip_value`` are treated as saturated
    and replaced by a cubic spline fitted to the remaining samples on the
    physical ``time`` axis. Every unsaturated sample is preserved exactly.

    ``clip_value`` may be a single signed level or a sequence of levels, in
    which case saturation is their union (see `detect_clipped_samples`).

    This is deliberately machine-independent: callers supply the limit and
    tolerance (VEST's PF6 acquisition clips near -5000 A, see `vest.yaml`).

    With ``return_mask=True`` the saturation mask is returned alongside the
    repaired waveform, so a caller can report which samples it reconstructed
    instead of handing downstream consumers an unmarked mixture.

    Raises:
        SignalRepairError: if the inputs contain non-finite values, the whole
            waveform is saturated, fewer than ``min_support`` unsaturated
            samples remain, or the saturated interval reaches either end of
            the record (which would require extrapolation, not interpolation).
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(data, dtype=float)
    if time.shape != values.shape:
        raise SignalRepairError(
            f"time and data must have the same shape, got {time.shape} and {values.shape}"
        )
    if values.ndim != 1:
        raise SignalRepairError("`data` must be one-dimensional.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise SignalRepairError(
            "Cannot repair a waveform containing non-finite samples; "
            "clean or mask the signal before requesting saturation repair."
        )

    saturated = detect_clipped_samples(values, clip_values=clip_value, tolerance=tolerance)
    if not saturated.any():
        return (values.copy(), saturated) if return_mask else values.copy()
    if saturated.all():
        raise SignalRepairError(
            f"Every sample is saturated at {clip_value}; there is no unsaturated "
            "support to interpolate from, so no waveform can be reconstructed."
        )

    support = np.flatnonzero(~saturated)
    if support.size < int(min_support):
        raise SignalRepairError(
            f"Cubic reconstruction needs at least {int(min_support)} unsaturated "
            f"samples, found {support.size}."
        )

    clipped = np.flatnonzero(saturated)
    if clipped[0] < support[0] or clipped[-1] > support[-1]:
        raise SignalRepairError(
            "The saturated interval reaches the start or end of the record, so "
            "reconstructing it would extrapolate beyond the measured support "
            "rather than interpolate between it."
        )

    spline = CubicSpline(time[support], values[support])
    repaired = values.copy()
    repaired[clipped] = spline(time[clipped])
    return (repaired, saturated) if return_mask else repaired


def line_average_density(n_e_line, path_length_m: float) -> np.ndarray:
    """Return line-average electron density given an explicit chord length.

    ``n_e_line`` is a line-integrated density (m^-2); dividing by the
    diagnostic's known path length gives a line-average density (m^-3). No
    calibration or geometry is inferred here -- ``path_length_m`` must be
    supplied by the caller.
    """
    if path_length_m <= 0:
        raise ValueError(f"path_length_m must be positive, got {path_length_m}")
    return np.asarray(n_e_line, dtype=float) / float(path_length_m)


def smooth(array, span: int) -> np.ndarray:
    """Apply MATLAB-like moving-average smoothing with edge tapering."""
    values = np.asarray(array, dtype=float)
    if values.ndim != 1:
        raise ValueError("`array` must be one-dimensional.")
    if values.size == 0 or span <= 1:
        return values.copy()
    if span % 2 == 0:
        span -= 1
    span = min(span, values.size if values.size % 2 == 1 else values.size - 1)
    if span < 1:
        return values.copy()

    count = values.size
    out = np.zeros(count, dtype=float)
    half = (span - 1) // 2

    # O(N) vectorized implementation using cumulative sums.
    cumsum = np.zeros(count + 1, dtype=float)
    cumsum[1:] = np.cumsum(values)

    # Interior: full-width windows of size `span`.
    interior = np.arange(half, count - half)
    if interior.size:
        out[interior] = (cumsum[interior + half + 1] - cumsum[interior - half]) / span

    # Edge tapering: left and right sides use progressively narrower windows.
    if half > 0:
        left_idx = np.arange(half)
        widths = 2 * left_idx + 1
        out[left_idx] = (cumsum[widths] - cumsum[0]) / widths
        right_idx = count - 1 - left_idx
        out[right_idx] = (cumsum[count] - cumsum[count - widths]) / widths

    return out


def butterworth_lowpass(data, cutoff: float, fs: float, order: int = 2,
                        *, zero_phase: bool = False) -> np.ndarray:
    """Butterworth low-pass filter along the last axis.

    ``zero_phase=False`` applies a causal ``lfilter`` -- the convention of the
    validated VEST SXR viewer, whose low-passed signals feed ratio and reference
    arithmetic where matched group delay between channels matters more than zero
    phase.  Pass ``zero_phase=True`` for a forward-backward ``filtfilt``.
    """
    values = np.asarray(data, dtype=float)
    nyquist = 0.5 * float(fs)
    if not 0.0 < float(cutoff) < nyquist:
        raise ValueError(
            f"cutoff must lie in (0, {nyquist:g}) Hz for fs={fs:g}; got {cutoff!r}"
        )
    b, a = scipy_signal.butter(int(order), float(cutoff) / nyquist, btype="low")
    if zero_phase:
        return scipy_signal.filtfilt(b, a, values, axis=-1)
    return scipy_signal.lfilter(b, a, values, axis=-1)


def butterworth_bandpass(data, low: float, high: float, fs: float, order: int = 2,
                         *, zero_phase: bool = True) -> np.ndarray:
    """Butterworth band-pass filter along the last axis (zero-phase by default)."""
    values = np.asarray(data, dtype=float)
    nyquist = 0.5 * float(fs)
    if not 0.0 < float(low) < float(high) < nyquist:
        raise ValueError(
            f"band edges must satisfy 0 < low < high < {nyquist:g} Hz for fs={fs:g}; "
            f"got ({low!r}, {high!r})"
        )
    b, a = scipy_signal.butter(
        int(order), [float(low) / nyquist, float(high) / nyquist], btype="band"
    )
    if zero_phase:
        return scipy_signal.filtfilt(b, a, values, axis=-1)
    return scipy_signal.lfilter(b, a, values, axis=-1)


def detrend_moving_average(data, window_samples: int) -> np.ndarray:
    """Subtract a centered moving-average trend along the last axis.

    The trend is a centered rolling mean with ``min_periods=1`` semantics: edge
    windows shrink rather than producing NaNs, so the output has the input's
    length.  Matches ``pandas.Series.rolling(window, center=True,
    min_periods=1).mean()``, the convention of the validated VEST SXR viewer.
    """
    values = np.asarray(data, dtype=float)
    window = int(window_samples)
    if window <= 1 or values.shape[-1] == 0:
        return values - values  # zero trend removal, preserving shape/dtype

    length = values.shape[-1]
    cumsum = np.zeros(values.shape[:-1] + (length + 1,), dtype=float)
    np.cumsum(values, axis=-1, out=cumsum[..., 1:])
    # A centered pandas window of size w at index i spans
    # [i - w//2, i + (w-1)//2], clipped to the record.
    index = np.arange(length)
    start = np.clip(index - window // 2, 0, length)
    stop = np.clip(index + (window - 1) // 2 + 1, 0, length)
    trend = (cumsum[..., stop] - cumsum[..., start]) / (stop - start)
    return values - trend


def vest_coil_current_noise_reduction(data) -> np.ndarray:
    """Suppress point spikes in coil current traces."""
    values = np.asarray(data, dtype=float)
    if values.size < 3:
        return values.copy()

    smoothed = values.copy()
    for index_i in range(2, values.size - 1):
        local_ref = abs((values[index_i + 1] + values[index_i - 1]) / 2.0)
        diff = abs(values[index_i]) - local_ref
        if diff > 0.001:
            smoothed[index_i] = smoothed[index_i - 1]
    return smoothed


def detect_active_window(time, signal, threshold: float = 0.01) -> tuple[float, float]:
    """Return the active time window that contains a signal's main peak.

    This operation is machine-independent.  VEST source selection and
    calibration belong in :mod:`vaft.machine_mapping`; callers should pass the
    resulting physical signal to this processing function.
    """
    time_values = np.asarray(time, dtype=float)
    data_values = np.asarray(signal, dtype=float)
    if time_values.ndim != 1 or data_values.ndim != 1:
        raise ValueError("`time` and `signal` must be one-dimensional.")
    if time_values.size != data_values.size:
        raise ValueError("`time` and `signal` must have the same length.")
    if time_values.size == 0:
        raise ValueError("`time` and `signal` must not be empty.")

    peak_index = int(np.argmax(data_values))
    if data_values[peak_index] < threshold:
        return float(time_values[0]), float(time_values[-1])

    start_index = peak_index
    while start_index > 0 and data_values[start_index - 1] >= threshold:
        start_index -= 1

    end_index = peak_index
    while end_index + 1 < data_values.size and data_values[end_index + 1] >= threshold:
        end_index += 1

    return float(time_values[start_index]), float(time_values[end_index])


def vfit_signal_start_end(time, data, threshold: float = 0.01) -> tuple[float, float]:
    """Deprecated compatibility wrapper for :func:`detect_active_window`."""
    warnings.warn(
        "vfit_signal_start_end() is deprecated; use detect_active_window().",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_active_window(time, data, threshold=threshold)


def process_signal(time, data, options=None):
    """Legacy conditioning wrapper kept in the process layer."""
    if options is None:
        options = {}

    time = np.asarray(time).reshape(-1)
    data = np.asarray(data).reshape(-1)

    if "time_range" in options:
        tstart, tend = options["time_range"]
        mask = (time >= tstart) & (time <= tend)
        time = time[mask]
        data = data[mask]

    if options.get("resample", False):
        dt = float(options.get("dt", 4e-5))
        if time.size == 0:
            return time, data
        new_time = np.arange(time[0], time[-1], dt)
        data = np.interp(new_time, time, data)
        time = new_time

    if "filter_params" in options:
        fp = options["filter_params"] or {}
        filter_type = fp.get("type", "lowpass")
        cutoff = fp.get("cutoff", 1000)
        order = int(fp.get("order", 4))

        if time.size < 2:
            return time, data

        fs = 1.0 / (time[1] - time[0])
        nyquist = fs / 2.0

        if filter_type == "bandpass":
            if not (isinstance(cutoff, (list, tuple, np.ndarray)) and len(cutoff) == 2):
                raise ValueError(
                    "'cutoff' must be a 2-element sequence [low, high] for 'bandpass' filter."
                )
            low, high = float(cutoff[0]), float(cutoff[1])
            if not (0 < low < high < nyquist):
                raise ValueError(
                    f"For 'bandpass' filter, cutoff must satisfy "
                    f"0 < low ({low}) < high ({high}) < fs/2 ({nyquist})."
                )
            b, a = scipy_signal.butter(order, [low, high], btype="band", fs=fs)
        elif filter_type in ("lowpass", "highpass"):
            cutoff_val = float(
                cutoff[0] if isinstance(cutoff, (list, tuple, np.ndarray)) else cutoff
            )
            if not (0 < cutoff_val < nyquist):
                raise ValueError(
                    f"'cutoff' ({cutoff_val}) must be between 0 and fs/2 ({nyquist}) "
                    f"for '{filter_type}' filter."
                )
            btype = "low" if filter_type == "lowpass" else "high"
            b, a = scipy_signal.butter(order, cutoff_val, btype=btype, fs=fs)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        data = scipy_signal.filtfilt(b, a, data)

    return time, data

def define_baseline(time, onset_time, onset_window, offset_time=None, offset_window=None):
    """
    Define a baseline window from the signal using onset and optional offset TIMES.

    Internally, we convert the specified times (onset_time, offset_time) to
    indices via np.searchsorted. The 'onset_window'/'offset_window' parameters
    remain as an integer number of samples to be included before or after
    the onset or offset indices.

    Parameters
    ----------
    time : numpy.ndarray
        The time array corresponding to the signal.
    onset_time : float
        The time at which the signal begins to deviate from baseline.
    onset_window : int
        The number of points (samples) to include in the baseline window before the onset index.
    offset_time : float, optional
        The time at which the signal returns to baseline. If None, no offset region is used.
    offset_window : int, optional
        The number of points (samples) to include in the baseline window after the offset index.
        If None, no offset region is used.

    Returns
    -------
    numpy.ndarray
        Indices of the baseline window values.
    """
    baseline_indices = []

    # Convert onset_time -> onset_idx
    onset_idx = np.searchsorted(time, onset_time)

    # Add onset window
    if onset_window > 0:
        start_idx = max(0, onset_idx - onset_window)
        baseline_indices.extend(range(start_idx, onset_idx))

    # Convert offset_time -> offset_idx if provided
    if offset_time is not None and offset_window is not None:
        offset_idx = np.searchsorted(time, offset_time)
        end_idx = min(len(time), offset_idx + offset_window)
        baseline_indices.extend(range(offset_idx, end_idx))

    return np.array(baseline_indices)

def linear_baseline(x, a, b):
    """Linear model for baseline fitting: y = a * x + b"""
    return a * x + b

def quadratic_baseline(x, a, b, c):
    """Quadratic model for baseline fitting: y = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c

def exp_baseline(x, a, b, c):
    """Exponential model for baseline fitting: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def subtract_baseline(time, signal, baseline_indices, fitting_opt='linear'):
    """
    Fit the baseline and subtract it from the signal.

    Parameters:
        time (numpy.ndarray): The time array corresponding to the signal.
        signal (numpy.ndarray): The input signal array.
        baseline_indices (numpy.ndarray): Indices specifying the baseline window.
        fitting_opt (str): The fitting option ('linear', 'quadratic', 'spline', 'exp').

    Returns:
        numpy.ndarray: The signal with the baseline subtracted.
        numpy.ndarray: The fitted baseline values.
    """
    x_baseline = time[baseline_indices]
    y_baseline = signal[baseline_indices]

    if fitting_opt == 'linear':
        popt, _ = curve_fit(linear_baseline, x_baseline, y_baseline)
        fitted_baseline = linear_baseline(time, *popt)
    elif fitting_opt == 'quadratic':
        popt, _ = curve_fit(quadratic_baseline, x_baseline, y_baseline)
        fitted_baseline = quadratic_baseline(time, *popt)
    elif fitting_opt == 'spline':
        spline = UnivariateSpline(x_baseline, y_baseline, s=0)
        fitted_baseline = spline(time)
    elif fitting_opt == 'exp':
        popt, _ = curve_fit(exp_baseline, x_baseline, y_baseline, maxfev=10000)
        fitted_baseline = exp_baseline(time, *popt)
    else:
        raise ValueError("Unsupported fitting option. Choose from 'linear', 'quadratic', 'spline', 'exp'.")

    corrected_signal = signal - fitted_baseline
    return corrected_signal, fitted_baseline

def signal_on_offset(time, data, smooth_window=5, threshold=0.01, verbose=False):
    if verbose:
        print("threshold for signal detection:", threshold)
    # Smooth the data
    smoothed = scipy_signal.savgol_filter(data, smooth_window, 3)
    return detect_active_window(time, smoothed, threshold=threshold)


def VEST_CoilCurrentNoiseReduction(data):  # noqa: N802
    """Deprecated compatibility wrapper for the snake-case function."""
    warnings.warn(
        "VEST_CoilCurrentNoiseReduction() is deprecated; use "
        "vest_coil_current_noise_reduction().",
        DeprecationWarning,
        stacklevel=2,
    )
    return vest_coil_current_noise_reduction(data)


def vfit_signal_startend(time, data, threshold: float = 0.01):
    """Deprecated compatibility wrapper for :func:`detect_active_window`."""
    warnings.warn(
        "vfit_signal_startend() is deprecated; use detect_active_window().",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_active_window(time, data, threshold=threshold)


def signal_onoffset(time, data, smooth_window=5, threshold=0.01, verbose=False):
    """Deprecated compatibility wrapper for :func:`signal_on_offset`."""
    warnings.warn(
        "signal_onoffset() is deprecated; use signal_on_offset().",
        DeprecationWarning,
        stacklevel=2,
    )
    return signal_on_offset(
        time,
        data,
        smooth_window=smooth_window,
        threshold=threshold,
        verbose=verbose,
    )

def is_signal_active(
    data,
    var_ratio_thresh=1e-2,
    change_ratio_thresh=1e-2,
    verbose=False,
):
    """
    Determines whether the given data represents an active signal
    using scale-invariant (relative) thresholds.

    Parameters:
        data (array-like): The signal data to analyze.
        var_ratio_thresh (float): Variance threshold relative to signal scale.
        change_ratio_thresh (float): Mean |Δx| threshold relative to signal scale.
        verbose (bool): If True, print debug information.

    Returns:
        bool: True if the signal is active, False otherwise.
    """
    data = np.asarray(data)

    if data.size < 2:
        return False

    variance = np.var(data)
    mean_abs_change = np.mean(np.abs(np.diff(data)))

    # Scale definitions (always positive, scale-aware)
    scale_var = np.var(data) + 1e-12
    scale_change = np.mean(np.abs(data)) + 1e-12

    var_ratio = variance / scale_var
    change_ratio = mean_abs_change / scale_change

    if verbose:
        print(f"Variance ratio: {var_ratio:.3e} (thresh={var_ratio_thresh:.3e})")
        print(f"Mean |Δx| ratio: {change_ratio:.3e} (thresh={change_ratio_thresh:.3e})")

    if var_ratio < var_ratio_thresh and change_ratio < change_ratio_thresh:
        return False
    return True
