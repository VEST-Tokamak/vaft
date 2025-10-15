import numpy as np
from numpy import ndarray
from typing import List, Dict, Any, Tuple
from omas import *
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import numpy as np

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

def signal_onoffset(time,data,smooth_window=5, threshold=0.01):
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
    onset=time[indxs]
    offset=time[indxe]
    return onset,offset

def is_signal_active(data, threshold=None, verbose=False):
    """
    Determines whether the given data represents an active signal.

    Parameters:
        data (array-like): The signal data to analyze.
        threshold (float or None): If None, use adaptive threshold based on data scale.
        verbose (bool): If True, print debug information.

    Returns:
        bool: True if the signal is active, False otherwise.
    """
    data = np.array(data)
    variance = np.var(data)
    mean_abs_change = np.mean(np.abs(np.diff(data)))

    if threshold is None:
        # Adaptive thresholds based on data scale
        mean_val = np.mean(data)
        var_threshold = 0.001 * mean_val**2
        change_threshold = 0.001 * mean_val
    else:
        var_threshold = threshold
        change_threshold = threshold

    if verbose:
        print(f"Variance: {variance:.3e}, Threshold: {var_threshold:.3e}")
        print(f"Mean |Δx|: {mean_abs_change:.3e}, Threshold: {change_threshold:.3e}")

    if variance < var_threshold and mean_abs_change < change_threshold:
        return False
    return True
