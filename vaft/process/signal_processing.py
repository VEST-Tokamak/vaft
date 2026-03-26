import numpy as np
from numpy import ndarray
from typing import List, Dict, Any, Tuple
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


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

    for i in range(half):
        width = 2 * i + 1
        out[i] = np.mean(values[:width])
        out[count - 1 - i] = np.mean(values[count - width : count])

    for i in range(half, count - half):
        out[i] = np.mean(values[i - half : i + half + 1])

    return out


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


def vfit_signal_start_end(time, data, threshold: float = 0.01) -> tuple[float, float]:
    """Detect an active signal window containing the main peak."""
    time_values = np.asarray(time, dtype=float)
    data_values = np.asarray(data, dtype=float)
    if time_values.ndim != 1 or data_values.ndim != 1:
        raise ValueError("`time` and `data` must be one-dimensional.")
    if time_values.size != data_values.size:
        raise ValueError("`time` and `data` must have the same length.")
    if time_values.size == 0:
        raise ValueError("`time` and `data` must not be empty.")

    peak_index = int(np.argmax(data_values))
    start_index = -1
    end_index = -1

    for idx, value in enumerate(data_values):
        if value >= threshold:
            if start_index == -1:
                start_index = idx
        else:
            end_index = idx - 1
            if start_index != -1 and start_index < peak_index < end_index:
                break
            start_index = -1

    if start_index == -1:
        start_index = 0
    if end_index == -1:
        end_index = time_values.size - 1

    return float(time_values[start_index]), float(time_values[end_index])


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
        if filter_type == "lowpass":
            b, a = signal.butter(order, cutoff, btype="low", fs=fs)
        elif filter_type == "highpass":
            b, a = signal.butter(order, cutoff, btype="high", fs=fs)
        elif filter_type == "bandpass":
            b, a = signal.butter(order, cutoff, btype="band", fs=fs)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        data = signal.filtfilt(b, a, data)

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
    data=signal.savgol_filter(data, smooth_window, 3)

    # Find the onset and offset of a signal (e.g. Halpha signal)
    nbt=len(time)

    # index of maximum value
    indxm=min(range(len(data)), key=lambda i: abs(data[i]-max(data)))
    indxs=-1
    indxe=-1
    # We are looking for windows that constain continue data above threshold.
    # The window we are looking for, must contain the maximum value
    for i in range(nbt):
        if data[i]>= threshold:
            if indxs==-1:
                indxs=i # start of the window
        else:
            indxe=i-1 # end of the window
            if indxs < indxm and indxm < indxe:
                break # if the windiw contains the maximum, we stop
            indxs=-1
    onset = time[indxs]
    offset = time[indxe]
    return onset, offset


VEST_CoilCurrentNoiseReduction = vest_coil_current_noise_reduction
vfit_signal_startend = vfit_signal_start_end
signal_onoffset = signal_on_offset

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
