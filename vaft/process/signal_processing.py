"""Conditioning of one waveform on a plain time axis.

Everything here takes NumPy arrays and returns NumPy arrays.  Nothing reads
an ODS, nothing knows what the waveform is, and nothing embeds a VEST
constant except where the function's name says so (``vest_*``) and its
``Applicability`` section says why.  The VEST source selection, calibration
and processing windows that decide *which* waveform to condition, and with
what numbers, live in :mod:`vaft.machine_mapping` and ``vest.yaml``.

Four kinds of operation:

* **Filtering** -- :func:`smooth`, :func:`butterworth_lowpass`,
  :func:`butterworth_bandpass`, :func:`detrend_moving_average`,
  :func:`anti_alias_filter`, :func:`vest_coil_current_noise_reduction`.
* **Baseline** -- :func:`define_baseline` selects the quiet samples,
  :func:`subtract_baseline` fits one of :func:`linear_baseline`,
  :func:`quadratic_baseline`, :func:`exp_baseline` or a spline through them
  and removes it.
* **Saturation repair** -- :func:`detect_clipped_samples`,
  :func:`repair_clipped_interval`.
* **Timebase and activity** -- :func:`describe_time_grid`,
  :func:`resample_to_time`, :func:`detect_active_window`,
  :func:`signal_on_offset`, :func:`is_signal_active`,
  :func:`infer_signal_orientation`, :func:`line_average_density`, and the
  legacy :func:`process_signal` wrapper.

Notation
--------
t        : time                                   [s]
dt       : sample spacing                         [s]
f_s      : sample rate, ``1 / dt``                [Hz]
f_N      : Nyquist frequency, ``f_s / 2``         [Hz]
f_c      : filter cutoff                          [Hz]
x        : the waveform, in whatever unit it has  [any]

Conventions
-----------
Two filtering conventions coexist and each function says which it uses.
*Causal* (``scipy.signal.lfilter``) delays every feature by the filter's
group delay and is the convention of the validated VEST SXR viewer, where
matched delay between channels matters more than zero phase.  *Zero-phase*
(``scipy.signal.filtfilt``, forward then backward) moves nothing in time
and is mandatory wherever an onset time is read off the result.
:func:`butterworth_lowpass` defaults to causal, everything else to
zero-phase.

Provenance
----------
.. [VFIT] The legacy VEST equilibrium workflow in MATLAB
   (``VFIT_VEST-Equilibrium-Code``), from which ``smooth``,
   ``vest_coil_current_noise_reduction``, ``detect_active_window`` and the
   baseline routines were ported, names and defaults preserved.
.. [SXR] The validated VEST soft X-ray viewer, whose filtering and
   detrending conventions ``butterworth_lowpass`` and
   ``detrend_moving_average`` reproduce (issue #131).
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import dataclasses

import numpy as np
import scipy.signal as scipy_signal
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit


__all__ = [
    "ResamplingError",
    "SignalOrientation",
    "SignalRepairError",
    "TimeGrid",
    "anti_alias_filter",
    "butterworth_bandpass",
    "butterworth_lowpass",
    "describe_time_grid",
    "detect_active_window",
    "infer_signal_orientation",
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
    "resample_to_time",
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
    ``tolerance`` of *any* supplied level.  Real acquisition hardware rails on
    both sides and rarely symmetrically -- VEST's diamagnetic Rogowski channel
    is a signed 16-bit ADC over +/-5 V, so its rails are exactly ``-5.0`` and
    ``5 * 32767 / 32768`` [1]_.

    Detecting every level in one pass is not a convenience.  Where a waveform
    oscillates hard enough to hit both rails within a few samples, repairing
    one rail at a time would fit the reconstruction through samples still
    pinned at the other one.

    Parameters
    ----------
    data : array_like
        The waveform, raw or calibrated; the rails are in its units [any].
    clip_values : float or sequence of float
        The acquisition limit or limits, signed, in the units of ``data`` [any].
    tolerance : float
        Half-width of the band around each limit inside which a sample counts
        as saturated [any].

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` where the sample is saturated; same shape as
        ``data`` [-].

    Raises
    ------
    SignalRepairError
        ``clip_values`` is empty or non-finite, or ``tolerance`` is not a
        positive finite width.

    Assumptions
    -----------
    Saturation shows as a sample *at* the rail, not as a distorted sample near
    it.  A digitizer that soft-clips will not be detected by a narrow
    tolerance.

    Applicability
    -------------
    Machine-independent.  The caller supplies the rails and tolerance; this
    function infers nothing about the hardware.  The VEST values are read from
    ``vest.yaml`` by :mod:`vaft.machine_mapping`, never here.

    Provenance
    ----------
    .. [1] ``vest.yaml``, the ``diamagnetic_flux`` processing block, where the
       asymmetric 16-bit rails are derived; issue #285.
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
    physical ``time`` axis.  Every unsaturated sample is preserved exactly.

    ``clip_value`` may be a single signed level or a sequence of levels, in
    which case saturation is their union (see :func:`detect_clipped_samples`).

    With ``return_mask=True`` the saturation mask is returned alongside the
    repaired waveform, so a caller can report which samples it reconstructed
    instead of handing downstream consumers an unmarked mixture.

    Parameters
    ----------
    time : array_like
        Sample times, one-dimensional, same length as ``data`` [s].
    data : array_like
        The waveform with saturated samples, one-dimensional [any].
    clip_value : float or sequence of float
        The acquisition limit or limits, signed, in the units of ``data`` [any].
    tolerance : float
        Half-width of the band around each limit inside which a sample counts
        as saturated [any].
    min_support : int, optional
        Fewest unsaturated samples the spline may be fitted through [-].
    return_mask : bool, optional
        Also return the saturation mask [-].

    Returns
    -------
    np.ndarray or tuple of (np.ndarray, np.ndarray)
        The repaired waveform, and with ``return_mask=True`` also the boolean
        saturation mask [any].

    Raises
    ------
    SignalRepairError
        The inputs differ in shape or are not one-dimensional; any sample is
        non-finite; the whole waveform is saturated; fewer than ``min_support``
        unsaturated samples remain; or the saturated interval reaches either
        end of the record, which would require extrapolation.

    Processing steps
    ----------------
    1. Mask the saturated samples with :func:`detect_clipped_samples`.
    2. Refuse if the mask covers everything, leaves fewer than ``min_support``
       samples, or touches either end of the record.
    3. Fit ``scipy.interpolate.CubicSpline`` through the unsaturated samples
       against ``time``.
    4. Evaluate it at the saturated instants and write those samples back.

    Input semantics
    ---------------
    Raw or calibrated -- the rails must be expressed in the same units as
    ``data``.  Unrepaired: a waveform that has already been filtered has
    smeared its rails and cannot be repaired here.

    Output semantics
    ----------------
    Repaired.  The reconstructed samples are interpolated, not measured; the
    mask says which they are.

    Defaults
    --------
    ``min_support = 4`` is a numerical convenience: a cubic spline needs four
    points to be determined at all, and this is the floor, not a quality bar.

    Assumptions
    -----------
    The waveform is smooth on the scale of the saturated interval, so a cubic
    through its neighbours is a defensible estimate of what was lost.  A
    saturated interval that hides a genuine fast feature is reconstructed as
    if the feature were absent.

    Applicability
    -------------
    Machine-independent.  Callers supply the limit and tolerance.  On VEST the
    PF6 supply clips near -5000 A and the repair is enabled per acquisition
    era in ``vest.yaml``, because PF6's gain flips sign at shot 38110 and
    before that a -5000 A sample is ordinary data [1]_.

    Limitations
    -----------
    Interpolation only: a saturated run at the start or end of the record is
    refused rather than extrapolated.  Reconstructing more than a few samples
    in a row grows less defensible with every sample; there is no cap on that
    here, so the caller should look at the mask.

    Provenance
    ----------
    .. [1] ``vest.yaml``, ``pf_active`` processing ``saturation_repair`` and the
       per-era policy comment; the donor ``vest_pf.m`` applied the -5000 A
       repair unconditionally.  Issues #195 and #285.
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

    Parameters
    ----------
    n_e_line : array_like
        Line-integrated electron density along the diagnostic chord [m^-2].
    path_length_m : float
        Length of the chord through the plasma [m].

    Returns
    -------
    np.ndarray
        Line-average electron density, ``n_e_line / path_length_m`` [m^-3].

    Raises
    ------
    ValueError
        ``path_length_m`` is not positive.

    Convention
    ----------
    Line-*integrated* in, line-*averaged* out.  An interferometer measures the
    former; the latter is what confinement scalings and Greenwald fractions
    want.  No calibration or geometry is inferred here -- the chord length is
    the caller's.

    Applicability
    -------------
    Machine-independent.  The chord length of a VEST interferometer channel
    comes from :mod:`vaft.machine_mapping`.
    """
    if path_length_m <= 0:
        raise ValueError(f"path_length_m must be positive, got {path_length_m}")
    return np.asarray(n_e_line, dtype=float) / float(path_length_m)


def smooth(array, span: int) -> np.ndarray:
    """Apply MATLAB-like moving-average smoothing with edge tapering.

    A centred moving average of odd width ``span``.  At each end the window
    shrinks symmetrically -- widths 1, 3, 5, ... -- so the output has the
    input's length and no sample is padded, which is what MATLAB's ``smooth``
    does and what the ported VEST workflows expect.

    Parameters
    ----------
    array : array_like
        The waveform, one-dimensional [any].
    span : int
        Window width in samples; an even value is reduced by one, and a value
        wider than the record is clipped to it [-].

    Returns
    -------
    np.ndarray
        The smoothed waveform, same length as ``array`` [any].

    Raises
    ------
    ValueError
        ``array`` is not one-dimensional.

    Convention
    ----------
    Zero-phase: the window is centred, so features are not shifted in time.
    Edge samples are averaged over progressively narrower windows rather than
    over a padded record, which biases the ends less than reflection padding
    would but leaves them noisier than the interior.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] MATLAB ``smooth(y, span)`` with the default ``'moving'`` method;
       ported from the legacy VEST workflow [VFIT]_ with its semantics
       preserved so that results match the donor sample for sample.
    """
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

    Parameters
    ----------
    data : array_like
        The waveform or a stack of them; the time axis is the last [any].
    cutoff : float
        Cutoff frequency, strictly between 0 and the Nyquist frequency [Hz].
    fs : float
        Sample rate [Hz].
    order : int, optional
        Filter order [-].
    zero_phase : bool, optional
        ``True`` for forward-backward ``filtfilt``; ``False`` for a causal
        ``lfilter`` [-].

    Returns
    -------
    np.ndarray
        The filtered waveform, same shape as ``data`` [any].

    Raises
    ------
    ValueError
        ``cutoff`` is not inside ``(0, fs / 2)``.

    Defaults
    --------
    ``order = 2`` and ``zero_phase = False`` are validated-workflow defaults:
    the values the VEST SXR viewer uses [SXR]_, kept so that
    :mod:`vaft.process.soft_x_rays` reproduces it.

    Convention
    ----------
    **Causal by default.**  With ``zero_phase=False`` every feature is delayed
    by the filter's group delay, identically on every channel, which is what
    ratio and reference arithmetic between SXR channels needs.  Pass
    ``zero_phase=True`` wherever a time is going to be read off the result --
    an onset, a peak -- because a causal filter moves it.

    Applicability
    -------------
    Machine-independent.  The defaults are VEST SXR practice but embed no VEST
    constant.

    Limitations
    -----------
    ``filtfilt`` needs a record longer than three times the filter's padding
    length and raises from SciPy otherwise; nothing here catches that.
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
    """Butterworth band-pass filter along the last axis.

    Parameters
    ----------
    data : array_like
        The waveform or a stack of them; the time axis is the last [any].
    low : float
        Lower band edge [Hz].
    high : float
        Upper band edge, above ``low`` and below the Nyquist frequency [Hz].
    fs : float
        Sample rate [Hz].
    order : int, optional
        Filter order [-].
    zero_phase : bool, optional
        ``True`` for forward-backward ``filtfilt``; ``False`` for a causal
        ``lfilter`` [-].

    Returns
    -------
    np.ndarray
        The filtered waveform, same shape as ``data`` [any].

    Raises
    ------
    ValueError
        The band edges do not satisfy ``0 < low < high < fs / 2``.

    Defaults
    --------
    ``order = 2`` is a validated-workflow default shared with
    :func:`butterworth_lowpass`.  ``zero_phase = True`` is the opposite of the
    low-pass default because band-passed signals feed spectral and
    mode-number analysis, where phase across channels is the measurement.

    Convention
    ----------
    Zero-phase by default; see :func:`butterworth_lowpass` for the causal
    alternative and when each is right.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    As :func:`butterworth_lowpass`: a record too short for ``filtfilt`` raises
    from SciPy.
    """
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

    Parameters
    ----------
    data : array_like
        The waveform or a stack of them; the time axis is the last [any].
    window_samples : int
        Width of the trend window in samples; ``1`` or less removes nothing [-].

    Returns
    -------
    np.ndarray
        ``data`` minus its rolling mean, same shape [any].

    Convention
    ----------
    The trend is a centred rolling mean with ``min_periods=1`` semantics: the
    window at index ``i`` spans ``[i - w//2, i + (w-1)//2]`` clipped to the
    record, so edge windows shrink rather than yield NaN and the output has the
    input's length.  This is exactly ``pandas.Series.rolling(window,
    center=True, min_periods=1).mean()``, the convention of the validated VEST
    SXR viewer [SXR]_, reimplemented with cumulative sums so that
    :mod:`vaft.process.soft_x_rays` does not need pandas for it.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A centred window is zero-phase but the shrinking edge windows bias the
    first and last ``w/2`` samples toward the local level.  The trend is a
    plain mean: one large excursion inside the window pulls the trend toward
    it and appears, inverted, in the neighbours.
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
    """Suppress point spikes in coil current traces.

    Walks the record and, wherever a sample's magnitude exceeds the mean of
    its two neighbours' magnitudes by more than a fixed margin, replaces it
    with the previous sample.  A port of the donor routine, kept sample for
    sample so that the legacy PF-current preprocessing reproduces.

    Parameters
    ----------
    data : array_like
        Coil current, one-dimensional; records shorter than three samples are
        returned unchanged [A].

    Returns
    -------
    np.ndarray
        The de-spiked current, same length [A].

    Processing steps
    ----------------
    1. For each interior sample (the first two and the last are never
       touched), compute ``|x[i]| - |(x[i+1] + x[i-1]) / 2|``.
    2. Where that exceeds ``0.001``, overwrite ``x[i]`` with the already
       processed ``x[i-1]``.

    Defaults
    --------
    The ``0.001`` margin is a legacy compatibility value from the donor
    ``VEST_CoilCurrentNoiseReduction.m`` [1]_.  It is absolute, in the units of
    ``data``, and its derivation is not recorded.  On a trace in amperes it
    flags any excursion above a milliamp, so in practice every spike; on a
    trace in kiloamperes it would let real noise through.

    Convention
    ----------
    Causal: a spike is replaced by its predecessor, so a genuine step is
    delayed by one sample at its leading edge.

    Applicability
    -------------
    VEST-specific.  Written for the VEST PF coil current traces as digitized by
    the legacy DAQ; the margin has no meaning for another machine's units or
    noise floor.

    Limitations
    -----------
    Only a single-sample spike is caught: two consecutive bad samples pass,
    because the second is compared against the first.  Sign is ignored in the
    test but preserved in the replacement.

    Provenance
    ----------
    .. [1] ``Function/VEST_CoilCurrentNoiseReduction.m`` in the legacy VEST
       workflow [VFIT]_; loop bounds and margin preserved.
    """
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

    Finds the largest sample and walks outward from it in both directions
    while the signal stays at or above ``threshold``; the window is the time
    of the first and last sample reached.  If the peak itself is below
    ``threshold`` the whole record is returned.

    Parameters
    ----------
    time : array_like
        Sample times, one-dimensional [s].
    signal : array_like
        The waveform, one-dimensional, same length as ``time``; already the
        physical quantity, not a raw digitizer voltage [any].
    threshold : float, optional
        Level at or above which a sample counts as active, in the units of
        ``signal`` [any].

    Returns
    -------
    tuple of (float, float)
        ``(start, end)`` times of the active window [s].

    Raises
    ------
    ValueError
        The inputs are not one-dimensional, differ in length, or are empty.

    Defaults
    --------
    ``threshold = 0.01`` is a legacy compatibility value from the donor
    ``vfit_signal_startend``, which was applied to an H-alpha trace
    normalised to its own minimum.  It is absolute, so a caller with a
    differently scaled signal must pass its own.

    Assumptions
    -----------
    The signal has one dominant positive excursion and is at or above
    ``threshold`` throughout it.  A dip below the threshold inside the pulse
    ends the window early.

    Applicability
    -------------
    Machine-independent.  VEST source selection and calibration belong in
    :mod:`vaft.machine_mapping`; the resulting physical signal is what is
    passed here.  For a plasma onset with evidence and a verdict, use
    :mod:`vaft.process.onset` instead.

    Limitations
    -----------
    The threshold is absolute, not relative to the peak, and the walk is not
    robust to noise: a single sample below ``threshold`` stops it.

    Provenance
    ----------
    .. [1] ``vfit_signal_startend``, the function nested in
       ``Function/vest_Halpha_tstart_tend.m`` of the legacy VEST workflow
       [VFIT]_; kept here under the deprecated name ``vfit_signal_start_end``.
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


# ---------------------------------------------------------------------------
# Rate changes: interpolation versus downsampling
# ---------------------------------------------------------------------------
#
# VEST acquires on two DAQ rates -- ``FAST_DT = 4e-6`` (250 kHz) and
# ``SLOW_DT = 4e-5`` (25 kHz), see :mod:`vaft.database.raw` -- while every
# processed time grid declared in ``vaft/machine_mapping/vest.yaml`` is 4e-5.
# A fast channel written onto a policy grid is therefore a 10x decimation, and
# ``np.interp`` alone folds everything above the 12.5 kHz target Nyquist back
# into the stored band.  ``resample_to_time`` exists so that the safe thing is
# what you get by default and the unsafe thing has to be asked for by name.


class ResamplingError(ValueError):
    """Raised when a rate change cannot be performed without fabricating signal.

    Separate from a plain ``ValueError`` because the recoveries differ: an
    unsorted source timebase means the loader is broken, not that the caller
    passed a bad option.
    """


@dataclass(frozen=True)
class TimeGrid:
    """Description of a timebase, as measured rather than as declared.

    ``dt`` is the *median* spacing, which is robust to the single ragged
    interval a concatenated acquisition can leave behind.  ``dt`` and
    ``sample_rate`` are ``nan`` for a single-sample grid, where neither is
    defined.
    """

    n: int
    t0: float
    dt: float
    sample_rate: float
    uniform: bool
    strictly_increasing: bool
    max_relative_jitter: float


def describe_time_grid(time, *, rtol: float = 1e-3) -> TimeGrid:
    """Measure a timebase's spacing, uniformity and monotonicity.

    Parameters
    ----------
    time : array_like
        Sample times, flattened [s].
    rtol : float, optional
        Largest ``|dt_i - median(dt)| / median(dt)`` for which the grid is
        still reported uniform [-].

    Returns
    -------
    TimeGrid
        ``n``, ``t0``, the median ``dt``, ``sample_rate``, whether the grid is
        ``uniform`` within ``rtol`` and ``strictly_increasing``, and the
        ``max_relative_jitter`` actually measured [any].

    Defaults
    --------
    ``rtol = 1e-3`` is a numerical convenience sized to an acquisition-era
    fact: VEST shots after 42190 store their timebase as ``linspace(0, span,
    n)``, so a nominally 4e-6 s grid is really ``span / (n - 1)`` =
    4.00016e-6 s.  That is uniform for every purpose here and a tight
    tolerance would only reject it.

    Convention
    ----------
    ``dt`` is the *median* spacing, robust to the single ragged interval a
    concatenated acquisition can leave behind.  ``dt`` and ``sample_rate``
    are ``nan`` for a single-sample grid, where neither is defined; an empty
    grid reports ``n = 0`` and ``nan`` throughout.

    Applicability
    -------------
    Machine-independent.
    """
    values = np.asarray(time, dtype=float).reshape(-1)
    n = int(values.size)
    if n == 0:
        return TimeGrid(0, float("nan"), float("nan"), float("nan"), False, False, float("nan"))
    t0 = float(values[0])
    if n == 1:
        return TimeGrid(1, t0, float("nan"), float("nan"), True, True, float("nan"))

    steps = np.diff(values)
    dt = float(np.median(steps))
    strictly_increasing = bool(np.all(steps > 0.0))
    if dt == 0.0 or not np.isfinite(dt):
        return TimeGrid(n, t0, dt, float("nan"), False, strictly_increasing, float("inf"))
    jitter = float(np.max(np.abs(steps - dt)) / abs(dt))
    return TimeGrid(
        n=n,
        t0=t0,
        dt=dt,
        sample_rate=1.0 / dt,
        uniform=bool(jitter <= float(rtol)),
        strictly_increasing=strictly_increasing,
        max_relative_jitter=jitter,
    )


def _anti_alias_numtaps(source_rate: float, cutoff_hz: float, stopband_hz: float) -> int:
    """Hamming-window tap estimate for the transition band ``cutoff -> stopband``.

    ``numtaps ~ 3.3 * fs / transition`` is the standard Hamming rule.  The
    transition must be measured against the frequency the stopband has to start
    at -- the *target* Nyquist -- not against the source Nyquist, which is ten
    times higher for a VEST fast channel and would size the filter at nine taps
    instead of the three hundred the job needs.  Forced odd so the FIR is Type I
    with an integer group delay, which keeps the ``filtfilt`` result symmetric.
    """
    transition = max(float(stopband_hz) - float(cutoff_hz), float(cutoff_hz) * 1e-3)
    numtaps = int(math.ceil(3.3 * float(source_rate) / transition))
    return max(numtaps + 1 - (numtaps % 2), 3)


def _filtfilt_min_length(numtaps: int) -> int:
    """Shortest record ``scipy.signal.filtfilt`` accepts for an FIR of this length."""
    return 3 * (int(numtaps) - 1) + 1


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` spans of consecutive ``True`` in ``mask``."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def anti_alias_filter(
    values,
    *,
    source_rate: float,
    cutoff_hz: float,
    stopband_hz: float | None = None,
    numtaps: int | None = None,
    axis: int = -1,
    nan_policy: str = "segment",
) -> np.ndarray:
    """Zero-phase FIR low-pass applied on the *source* grid, ahead of a rate cut.

    This is the half of a downsample that ``np.interp`` cannot do for you.
    Filtering after the rate has already been reduced is too late: whatever
    folded is now indistinguishable from real in-band signal.

    Parameters
    ----------
    values : array_like
        The waveform or a stack of them; the time axis is ``axis`` [any].
    source_rate : float
        Sample rate of ``values`` [Hz].
    cutoff_hz : float
        Passband edge, strictly inside ``(0, source_rate / 2)`` [Hz].
    stopband_hz : float or None, optional
        Frequency by which the response must be down -- for a downsample, the
        target Nyquist.  Sets the transition band and so the tap count [Hz].
    numtaps : int or None, optional
        Override the designed tap count; forced odd [-].
    axis : int, optional
        The time axis of ``values`` [-].
    nan_policy : {"segment", "error", "ignore"}, optional
        ``"segment"`` filters each maximal finite run on its own; ``"error"``
        refuses non-finite samples; ``"ignore"`` hands them to ``filtfilt``,
        which smears them across the record [-].

    Returns
    -------
    np.ndarray
        The filtered waveform, same shape as ``values`` [any].

    Raises
    ------
    ResamplingError
        ``cutoff_hz`` or ``stopband_hz`` outside their bands, or a non-finite
        sample under ``nan_policy="error"``.
    ValueError
        Unknown ``nan_policy``.

    Processing steps
    ----------------
    1. Size the filter: ``numtaps ~ 3.3 * source_rate / (stopband - cutoff)``,
       the Hamming-window rule [1]_, measured against the *target* Nyquist --
       not the source Nyquist, which is ten times higher for a VEST fast
       channel and would size the filter at nine taps instead of the three
       hundred the job needs.  Forced odd so the FIR is Type I with integer
       group delay.
    2. Design it with ``scipy.signal.firwin`` (Hamming window, low-pass).
    3. Apply ``scipy.signal.filtfilt`` to each finite run long enough for it;
       shorter runs are passed through with a ``RuntimeWarning``.

    Defaults
    --------
    ``stopband_hz = 1.25 * cutoff_hz`` is a numerical convenience consistent
    with :func:`resample_to_time`'s default 0.8-of-Nyquist cutoff: it puts
    the stopband exactly at the target Nyquist.  ``nan_policy = "segment"`` is
    a validated-workflow default -- one NaN must not blank a whole record.

    Convention
    ----------
    Zero-phase, and load-bearing: filterscope intensity feeds onset detection,
    where a causal filter would move the onset by the group delay.  ``firwin``
    + ``filtfilt`` rather than a Butterworth or ``decimate``'s Chebyshev,
    because it is the idiom the VEST mappers already use
    (``machine_mapping/tf.py``, ``machine_mapping/pf_active.py``,
    ``process/magnetics.py``).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``filtfilt`` needs ``3 * (numtaps - 1) + 1`` samples; a record or finite
    run shorter than that is returned unfiltered, with a warning, rather than
    padded.  The filter assumes evenly spaced samples; on an irregular grid the
    design rate is a fiction, which :func:`resample_to_time` refuses for you.

    Provenance
    ----------
    .. [1] Hamming-window FIR design rule, ``N ~ 3.3 f_s / Delta f``, e.g.
       Oppenheim & Schafer, *Discrete-Time Signal Processing*, window method.
    .. [2] ``vaft/machine_mapping/tf.py`` and ``pf_active.py``, the mappers
       whose established filtering idiom this generalizes.
    """
    data = np.asarray(values, dtype=float)
    rate = float(source_rate)
    cutoff = float(cutoff_hz)
    nyquist = 0.5 * rate
    if not 0.0 < cutoff < nyquist:
        raise ResamplingError(
            f"anti-alias cutoff must lie in (0, {nyquist:g}) Hz for a "
            f"{rate:g} Hz source; got {cutoff!r}"
        )
    stopband = float(stopband_hz) if stopband_hz is not None else 1.25 * cutoff
    if not cutoff < stopband <= nyquist:
        raise ResamplingError(
            f"anti-alias stopband must lie in ({cutoff:g}, {nyquist:g}] Hz; got {stopband!r}"
        )
    taps_count = int(numtaps) if numtaps is not None else _anti_alias_numtaps(rate, cutoff, stopband)
    if taps_count % 2 == 0:
        taps_count += 1

    if nan_policy not in ("segment", "error", "ignore"):
        raise ValueError(f"nan_policy must be 'segment', 'error' or 'ignore'; got {nan_policy!r}")

    moved = np.moveaxis(data, axis, -1)
    length = moved.shape[-1]
    minimum = _filtfilt_min_length(taps_count)

    finite = np.isfinite(moved)
    if nan_policy == "error" and not finite.all():
        raise ResamplingError("anti-alias filtering requires finite samples (nan_policy='error')")

    taps = scipy_signal.firwin(taps_count, cutoff, pass_zero="lowpass", fs=rate)

    if nan_policy == "ignore" or finite.all():
        if length < minimum:
            warnings.warn(
                f"record of {length} samples is shorter than the {minimum} required to "
                f"anti-alias filter with {taps_count} taps; leaving it unfiltered",
                RuntimeWarning,
                stacklevel=3,
            )
            return data
        return np.moveaxis(scipy_signal.filtfilt(taps, 1.0, moved, axis=-1), -1, axis)

    out = moved.copy()
    flat = out.reshape(-1, length)
    flat_finite = finite.reshape(-1, length)
    warned = False
    for row, row_finite in zip(flat, flat_finite):
        for start, stop in _finite_runs(row_finite):
            if stop - start < minimum:
                if not warned:
                    warnings.warn(
                        f"a finite run of {stop - start} samples is shorter than the "
                        f"{minimum} required to anti-alias filter with {taps_count} taps; "
                        "leaving it unfiltered",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    warned = True
                continue
            row[start:stop] = scipy_signal.filtfilt(taps, 1.0, row[start:stop])
    return np.moveaxis(flat.reshape(moved.shape), -1, axis)


def _interp_along_last_axis(target_time, source_time, values, *, extrapolate):
    """``np.interp`` broadcast over the leading axes, with an end-fill policy."""
    length = values.shape[-1]
    flat = values.reshape(-1, length)
    out = np.empty((flat.shape[0], target_time.size), dtype=float)
    left = right = None
    if extrapolate == "nan":
        left = right = float("nan")
    for index, row in enumerate(flat):
        out[index] = np.interp(target_time, source_time, row, left=left, right=right)
    return out.reshape(values.shape[:-1] + (target_time.size,))


def resample_to_time(
    source_time,
    values,
    target_time,
    *,
    anti_alias: bool | str = "auto",
    cutoff_fraction: float = 0.8,
    cutoff_hz: float | None = None,
    numtaps: int | None = None,
    min_ratio: float = 1.05,
    extrapolate: str = "clamp",
    nan_policy: str = "segment",
    on_unsorted: str = "error",
    axis: int = -1,
) -> np.ndarray:
    """Project a signal onto ``target_time``, low-passing first if that is a rate cut.

    Use this instead of a bare ``np.interp`` anywhere a diagnostic is written
    onto a common time grid.  Interpolation and downsampling look identical in
    source but are not the same operation: evaluating a signal at new instants
    is always fine, whereas *reducing* the sample rate discards the information
    needed to distinguish a high frequency from its alias, so the content above
    the new Nyquist has to be removed while it still can be.

    Parameters
    ----------
    source_time : array_like
        Sample times of ``values``, flattened; must be strictly increasing
        under the default ``on_unsorted`` [s].
    values : array_like
        The waveform or a stack of them; the time axis is ``axis`` [any].
    target_time : array_like
        Instants to evaluate at; need not be sorted [s].
    anti_alias : bool or "auto", optional
        ``"auto"`` filters only for a genuine rate reduction; ``True`` always;
        ``False`` never [-].
    cutoff_fraction : float, optional
        Anti-alias cutoff as a fraction of the *target* Nyquist [-].
    cutoff_hz : float or None, optional
        Explicit cutoff, overriding ``cutoff_fraction`` [Hz].
    numtaps : int or None, optional
        Passed to :func:`anti_alias_filter` [-].
    min_ratio : float, optional
        ``dt_target / dt_source`` at or below which ``"auto"`` does not
        filter [-].
    extrapolate : {"clamp", "nan", "error"}, optional
        What a target sample outside the source span gets [-].
    nan_policy : str, optional
        Passed to :func:`anti_alias_filter` [-].
    on_unsorted : {"error", "sort"}, optional
        Refuse an unsorted source, or sort it and collapse duplicates [-].
    axis : int, optional
        The time axis of ``values`` [-].

    Returns
    -------
    np.ndarray
        ``values`` on ``target_time``; the time axis has ``len(target_time)``
        samples [any].

    Raises
    ------
    ResamplingError
        Empty source; unsorted or duplicated source times under the default
        policy; a target outside the source span under ``extrapolate="error"``;
        a forced filter whose cutoff exceeds the source Nyquist; or a filter
        requested on a non-uniform source grid.
    ValueError
        An option outside its allowed values.

    Processing steps
    ----------------
    1. Validate options; sort and de-duplicate the source if asked.
    2. Measure both grids with :func:`describe_time_grid`.
    3. Decide whether to filter: ``anti_alias`` as given, or under ``"auto"``
       whether ``dt_target / dt_source > min_ratio``.
    4. If filtering, refuse a non-uniform source, place the cutoff at
       ``cutoff_fraction`` of the target Nyquist (or ``cutoff_hz``), put the
       stopband at the target Nyquist, and run :func:`anti_alias_filter` on
       the source grid.
    5. ``np.interp`` onto ``target_time`` with the ``extrapolate`` policy.

    Output semantics
    ----------------
    Interpolated.  When no filter runs, the result is **bit-for-bit**
    ``np.interp`` -- deliberately, so equal-rate call sites can adopt this
    primitive without moving a stored value.  When one does, the result is
    band-limited to the target Nyquist and then interpolated.

    Defaults
    --------
    ``anti_alias = "auto"`` and ``min_ratio = 1.05`` are validated-workflow
    defaults: the safe thing by default, the unsafe thing by name.
    ``cutoff_fraction = 0.8`` is a numerical convenience -- ``firwin`` sits at
    -6 dB at its cutoff and needs a transition band, so a passband edge below
    Nyquist puts everything that can fold into the stopband.  ``extrapolate =
    "clamp"`` matches ``np.interp``.  ``on_unsorted = "error"`` because
    ``np.interp`` returns silent nonsense for an unsorted ``x``.

    Convention
    ----------
    The cutoff is placed against the *target* grid's spacing measured in time
    order, even when ``target_time`` is unsorted; the shuffled array would
    report a median ``dt`` tens of times too large and design a filter that
    eats the signal.  An explicit ``cutoff_hz`` above the target Nyquist is
    honoured as a deliberate choice to keep more band and gets
    :func:`anti_alias_filter`'s own transition instead.

    Assumptions
    -----------
    ``values`` is a band-limited physical signal.  A validity mask or any
    other logical series is not, and must be resampled with
    ``anti_alias=False``.

    Applicability
    -------------
    Machine-independent.  The motivating case is VEST, which acquires at
    ``FAST_DT = 4e-6`` s and ``SLOW_DT = 4e-5`` s while every processed grid
    in ``vest.yaml`` is 4e-5 s, so a fast channel written onto a policy grid
    is a 10x decimation [1]_.

    Limitations
    -----------
    A single-sample target has no Nyquist: ``"auto"`` interpolates with a
    warning and ``True`` raises unless ``cutoff_hz`` is given.  A non-uniform
    source grid cannot be anti-aliased by a single-rate FIR and is refused;
    resample onto a uniform grid first, or pass ``anti_alias=False`` to accept
    a bare interpolation knowingly.

    Provenance
    ----------
    .. [1] :mod:`vaft.database.raw` ``FAST_DT`` / ``SLOW_DT``, and the
       processed time grids declared in ``vaft/machine_mapping/vest.yaml``.
    """
    if anti_alias not in (True, False, "auto"):
        raise ValueError(f"anti_alias must be True, False or 'auto'; got {anti_alias!r}")
    if extrapolate not in ("clamp", "nan", "error"):
        raise ValueError(f"extrapolate must be 'clamp', 'nan' or 'error'; got {extrapolate!r}")
    if on_unsorted not in ("error", "sort"):
        raise ValueError(f"on_unsorted must be 'error' or 'sort'; got {on_unsorted!r}")

    times = np.asarray(source_time, dtype=float).reshape(-1)
    targets = np.asarray(target_time, dtype=float).reshape(-1)
    data = np.asarray(values, dtype=float)

    if times.size == 0:
        raise ResamplingError("cannot resample from an empty source timebase")
    moved = np.moveaxis(data, axis, -1) if data.ndim > 1 else data.reshape(-1)
    moved = np.atleast_1d(moved)
    if moved.shape[-1] != times.size:
        raise ResamplingError(
            f"source_time has {times.size} samples but values has {moved.shape[-1]} "
            "along the time axis"
        )

    if targets.size == 0:
        empty = np.empty(moved.shape[:-1] + (0,), dtype=float)
        return np.moveaxis(empty, -1, axis) if data.ndim > 1 else empty.reshape(0)

    source_grid = describe_time_grid(times)
    if not source_grid.strictly_increasing:
        if on_unsorted == "error":
            raise ResamplingError(
                "source_time must be strictly increasing; np.interp returns silent "
                "nonsense otherwise. Pass on_unsorted='sort' to sort and collapse "
                "duplicates instead."
            )
        warnings.warn(
            "source_time was not strictly increasing; sorting and collapsing duplicates",
            RuntimeWarning,
            stacklevel=2,
        )
        order = np.argsort(times, kind="stable")
        times = times[order]
        moved = moved[..., order]
        keep = np.concatenate(([True], np.diff(times) > 0.0))
        times = times[keep]
        moved = moved[..., keep]
        source_grid = describe_time_grid(times)

    target_sorted = bool(targets.size < 2 or np.all(np.diff(targets) >= 0.0))
    target_low = float(targets[0] if target_sorted else targets.min())
    target_high = float(targets[-1] if target_sorted else targets.max())
    if extrapolate == "error" and (target_low < times[0] or target_high > times[-1]):
        raise ResamplingError(
            f"target_time spans [{target_low:g}, {target_high:g}] s, outside the source's "
            f"[{times[0]:g}, {times[-1]:g}] s; np.interp would clamp and fabricate a "
            "constant tail (extrapolate='error')"
        )

    if times.size == 1:
        filled = np.repeat(moved, targets.size, axis=-1)
        return np.moveaxis(filled, -1, axis) if data.ndim > 1 else filled.reshape(-1)

    # np.interp evaluates at arbitrary instants, so target_time is allowed to be
    # unsorted -- but the *spacing* statistics that place the cutoff are only
    # meaningful in time order.  Measuring the shuffled array would report a
    # median dt tens of times too large and design a filter that eats the signal.
    target_grid = describe_time_grid(targets if target_sorted else np.sort(targets))
    should_filter = bool(anti_alias is True)
    if anti_alias == "auto":
        if targets.size < 2:
            warnings.warn(
                "target_time has a single sample, so no target Nyquist is defined; "
                "interpolating without an anti-alias filter",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            ratio = target_grid.dt / source_grid.dt
            should_filter = bool(np.isfinite(ratio) and ratio > float(min_ratio))

    if should_filter and not source_grid.uniform:
        # firwin/filtfilt assume evenly spaced samples.  On an irregular grid the
        # design rate is a fiction and the filter neither rejects what it should
        # nor preserves what it should, so say so rather than return a number
        # that looks filtered.  Resample onto a uniform grid first, or pass
        # anti_alias=False to accept a bare interpolation knowingly.
        raise ResamplingError(
            f"source_time is not uniformly sampled (spacing varies by "
            f"{source_grid.max_relative_jitter:.3g} of the median), so a "
            "linear-phase FIR designed at a single rate cannot anti-alias it. "
            "Resample onto a uniform grid first, or pass anti_alias=False to "
            "interpolate without the guarantee."
        )

    if should_filter:
        if cutoff_hz is not None:
            cutoff = float(cutoff_hz)
        elif targets.size < 2 or not np.isfinite(target_grid.dt):
            raise ResamplingError(
                "anti_alias=True needs a target Nyquist to place the cutoff, but "
                "target_time has fewer than two samples; pass cutoff_hz explicitly"
            )
        else:
            cutoff = float(cutoff_fraction) * (0.5 / target_grid.dt)
        # The stopband has to start at the target Nyquist: that is the frequency
        # above which content folds instead of being discarded.  An explicit
        # cutoff_hz above that Nyquist is a deliberate choice to keep more band,
        # and it gets anti_alias_filter's own 1.25x transition instead -- forcing
        # the stopband below the cutoff would collapse the transition band to
        # nothing and silently price the filter out of the record's length.
        stopband = None
        if targets.size >= 2 and np.isfinite(target_grid.dt):
            target_nyquist = 0.5 / abs(target_grid.dt)
            if cutoff < target_nyquist:
                stopband = min(target_nyquist, 0.5 * source_grid.sample_rate)
        moved = anti_alias_filter(
            moved,
            source_rate=source_grid.sample_rate,
            cutoff_hz=cutoff,
            stopband_hz=stopband,
            numtaps=numtaps,
            axis=-1,
            nan_policy=nan_policy,
        )

    out = _interp_along_last_axis(targets, times, moved, extrapolate=extrapolate)
    return np.moveaxis(out, -1, axis) if data.ndim > 1 else out.reshape(-1)


def process_signal(time, data, options=None):
    """Legacy conditioning wrapper: crop, resample, then filter.

    Parameters
    ----------
    time : array_like
        Sample times, flattened [s].
    data : array_like
        The waveform, flattened [any].
    options : dict or None, optional
        ``time_range`` ``(t_start, t_end)`` to crop to; ``resample`` with
        ``dt`` to write onto a uniform grid; ``filter_params`` with ``type``
        (``"lowpass"``, ``"highpass"``, ``"bandpass"``), ``cutoff`` and
        ``order`` [-].

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The conditioned ``(time, data)`` [any].

    Raises
    ------
    ValueError
        A ``filter_params`` cutoff outside ``(0, f_s / 2)``, a malformed
        band-pass cutoff, or an unknown filter type.

    Processing steps
    ----------------
    1. Crop to ``time_range``.
    2. Resample onto ``arange(t[0], t[-1], dt)`` through
       :func:`resample_to_time`, so a rate reduction anti-aliases on the input
       grid first.
    3. Apply a zero-phase Butterworth from ``filter_params``.  This is a
       *shaping* filter and its ``cutoff`` is interpreted against the
       **output** grid's sample rate, not the input's.

    Defaults
    --------
    ``dt = 4e-5`` s is the VEST slow-DAQ spacing, a machine-specific setting.
    ``cutoff = 1000`` Hz and ``order = 4`` are legacy compatibility values
    whose origin is not recorded.

    Convention
    ----------
    Zero-phase (``filtfilt``) for the shaping filter.  The sample rate for
    filter design is taken from the first two samples, not the median.

    Applicability
    -------------
    VEST-specific.  The defaults embed the VEST acquisition grid, and the
    option dictionary is the legacy call convention kept for existing
    workflows; new code should call :func:`resample_to_time` and
    :func:`butterworth_lowpass` directly.

    Limitations
    -----------
    Kept for compatibility; its future is decided by the module-ownership
    audit tracked in #263.
    """
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
        data = resample_to_time(time, data, new_time)
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
    """Select the baseline samples around a signal's onset and optional offset.

    Converts ``onset_time`` (and ``offset_time``) to indices with
    ``np.searchsorted`` and returns the ``onset_window`` samples immediately
    before the onset, plus the ``offset_window`` samples immediately after the
    offset when both are given.  The result is what :func:`subtract_baseline`
    fits through.

    Parameters
    ----------
    time : np.ndarray
        Sample times, sorted [s].
    onset_time : float
        Time at which the signal begins to deviate from baseline [s].
    onset_window : int
        Number of samples before the onset to include [-].
    offset_time : float or None, optional
        Time at which the signal returns to baseline [s].
    offset_window : int or None, optional
        Number of samples after the offset to include; both offset arguments
        must be given for the offset region to be used [-].

    Returns
    -------
    np.ndarray
        Indices into ``time`` of the baseline samples, onset region first [-].

    Assumptions
    -----------
    The signal is genuinely quiet in the selected windows.  A window that
    reaches back into a previous pulse, or forward into pickup from a coil
    ramp, is fitted as if it were baseline.

    Applicability
    -------------
    Machine-independent.  The onset and offset times, and the window widths,
    are the caller's; on VEST they come from the magnetics processing window
    in ``vest.yaml`` via :mod:`vaft.process.magnetics`.

    Limitations
    -----------
    Windows are clipped to the record silently, so a window wider than the
    pre-onset record yields fewer samples than asked, with no warning.

    Provenance
    ----------
    .. [1] The legacy VEST EFIT magnetics chain [VFIT]_, where the baseline is
       taken before the discharge and after it and removed by a fit.
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
    """Linear model for baseline fitting: ``y = a * x + b``.

    Parameters
    ----------
    x : array_like
        Abscissa, normally time [any].
    a : float
        Slope [any].
    b : float
        Intercept [any].

    Returns
    -------
    np.ndarray
        ``a * x + b`` [any].

    Applicability
    -------------
    Machine-independent.  A fit model for :func:`subtract_baseline`.
    """
    return a * x + b

def quadratic_baseline(x, a, b, c):
    """Quadratic model for baseline fitting: ``y = a * x^2 + b * x + c``.

    Parameters
    ----------
    x : array_like
        Abscissa, normally time [any].
    a : float
        Quadratic coefficient [any].
    b : float
        Linear coefficient [any].
    c : float
        Constant [any].

    Returns
    -------
    np.ndarray
        ``a * x**2 + b * x + c`` [any].

    Applicability
    -------------
    Machine-independent.  A fit model for :func:`subtract_baseline`.
    """
    return a * x**2 + b * x + c

def exp_baseline(x, a, b, c):
    """Exponential model for baseline fitting: ``y = a * exp(b * x) + c``.

    Parameters
    ----------
    x : array_like
        Abscissa, normally time [any].
    a : float
        Amplitude [any].
    b : float
        Rate; the sign decides growth or decay [any].
    c : float
        Offset [any].

    Returns
    -------
    np.ndarray
        ``a * exp(b * x) + c`` [any].

    Applicability
    -------------
    Machine-independent.  A fit model for :func:`subtract_baseline`.
    """
    return a * np.exp(b * x) + c

def subtract_baseline(time, signal, baseline_indices, fitting_opt='linear'):
    """Fit a baseline through the selected samples and subtract it everywhere.

    Parameters
    ----------
    time : np.ndarray
        Sample times [s].
    signal : np.ndarray
        The waveform, same length as ``time`` [any].
    baseline_indices : np.ndarray
        Indices of the samples the baseline is fitted through, normally from
        :func:`define_baseline` [-].
    fitting_opt : {"linear", "quadratic", "spline", "exp"}, optional
        The baseline model [-].

    Returns
    -------
    corrected : np.ndarray
        ``signal`` minus the fitted baseline [any].
    baseline : np.ndarray
        The fitted baseline evaluated on the whole ``time`` axis [any].

    Raises
    ------
    ValueError
        Unknown ``fitting_opt``.

    Processing steps
    ----------------
    1. Take ``time`` and ``signal`` at ``baseline_indices``.
    2. Fit the chosen model through them: ``scipy.optimize.curve_fit`` for
       the parametric models, ``scipy.interpolate.UnivariateSpline(s=0)`` --
       an interpolating spline -- for ``"spline"``.
    3. Evaluate the model on the full ``time`` axis and subtract.

    Output semantics
    ----------------
    Baseline-subtracted.  The second return is the baseline itself, so the
    caller can plot or store what was removed.

    Defaults
    --------
    ``fitting_opt = "linear"`` is a legacy compatibility value: the donor
    workflow removed a linear drift, which is the right model for an
    integrator's offset.  The exponential fit uses ``maxfev = 10000``, a
    numerical convenience.

    Assumptions
    -----------
    The samples at ``baseline_indices`` contain no signal.  The parametric
    models assume the drift has that form over the *whole* record, including
    the pulse, where it is extrapolated rather than measured.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``"spline"`` interpolates the baseline samples exactly, so noise in them
    is reproduced in the baseline; with the onset and offset regions far
    apart it is a straight line between them in practice.  ``"exp"`` can fail
    to converge and raises from SciPy.  Nothing checks that
    ``baseline_indices`` is non-empty.

    Provenance
    ----------
    .. [1] The legacy VEST EFIT magnetics chain [VFIT]_, where an integrated
       probe signal's drift is removed by a fit through the pre- and
       post-discharge windows.
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
    """Smooth a waveform, then return the window in which it is above a threshold.

    Parameters
    ----------
    time : array_like
        Sample times, one-dimensional [s].
    data : array_like
        The waveform, one-dimensional, same length as ``time`` [any].
    smooth_window : int, optional
        Savitzky-Golay window length in samples; must be odd and larger than
        the polynomial order, 3 [-].
    threshold : float, optional
        Level at or above which the smoothed signal counts as active, in the
        units of ``data`` [any].
    verbose : bool, optional
        Print the threshold [-].

    Returns
    -------
    tuple of (float, float)
        ``(onset, offset)`` times [s].

    Processing steps
    ----------------
    1. Smooth with ``scipy.signal.savgol_filter(data, smooth_window, 3)``.
    2. Hand the smoothed waveform to :func:`detect_active_window`.

    Defaults
    --------
    ``smooth_window = 5`` and the cubic polynomial are legacy compatibility
    values from the ported workflow, unrecorded beyond that.  ``threshold =
    0.01`` is inherited from :func:`detect_active_window`.

    Convention
    ----------
    Zero-phase: a Savitzky-Golay filter is a centred polynomial fit and moves
    nothing in time, so the returned onset is not delayed by the smoothing.

    Applicability
    -------------
    Machine-independent.  Callers in :mod:`vaft.omas.general` and
    :mod:`vaft.database` pass ``threshold=0.05`` for VEST H-alpha; the default
    is not a VEST policy.  For a plasma onset with evidence and a verdict, use
    :mod:`vaft.process.onset`.

    Limitations
    -----------
    Those of :func:`detect_active_window`: an absolute threshold and a walk
    that stops at the first sample below it.
    """

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
    """Decide whether a trace carries a signal or is flat.

    Parameters
    ----------
    data : array_like
        The waveform; fewer than two samples is never active [any].
    var_ratio_thresh : float, optional
        Threshold on the variance ratio described below [-].
    change_ratio_thresh : float, optional
        Threshold on the mean absolute sample-to-sample change divided by the
        mean absolute level [-].
    verbose : bool, optional
        Print both ratios [-].

    Returns
    -------
    bool
        ``True`` unless *both* ratios fall below their thresholds [-].

    Defaults
    --------
    Both thresholds at ``1e-2`` are legacy compatibility values; no derivation
    is recorded.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The variance test is a tautology: the variance is divided by itself plus
    ``1e-12``, so the ratio is 1 for any trace whose variance exceeds about
    1e-10, and the ``var_ratio_thresh`` branch can only fire for a trace that
    is constant to twelve decimal places.  The decision is therefore made by
    ``change_ratio`` alone, and a trace flat to one part in a million is
    reported active.  Tracked in #463; this docstring describes the code as it
    is, not as it was meant to be.
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


@dataclasses.dataclass(frozen=True)
class SignalOrientation:
    """The dominant sign of a signal over its representative region (#307).

    ``multiplier`` is ``+1`` or ``-1`` when ``resolved``, else ``+1`` with
    ``reason`` saying why nothing could be decided: a caller that wants an
    intuitive display multiplies the *whole* signal by it once and never
    rectifies samples.  ``statistic`` is the median of the region and
    ``count`` how many samples it held.
    """

    multiplier: int
    statistic: float
    count: int
    resolved: bool
    reason: str = ""


def infer_signal_orientation(
    signal,
    *,
    mask=None,
    active_fraction: float = 0.1,
    min_samples: int = 8,
    agreement: float = 0.6,
) -> SignalOrientation:
    """Infer the dominant sign of ``signal`` from its active region.

    The region is ``mask`` when given, otherwise the samples whose magnitude
    reaches ``active_fraction`` of the signal's peak magnitude -- so the long
    zero stretches before and after a discharge do not vote.  The verdict is
    the sign of the region's median, resolved only when at least
    ``agreement`` of the region shares it and the region holds at least
    ``min_samples`` finite samples; otherwise it is unresolved and the
    multiplier is ``+1`` (canonical), never a guess.  ``abs`` is never applied
    to the data and no sample is flipped on its own.

    Parameters
    ----------
    signal : array_like
        The waveform, one-dimensional; non-finite samples are ignored [any].
    mask : array_like of bool or None, optional
        The region to vote over, instead of the magnitude rule [-].
    active_fraction : float, optional
        Fraction of the peak magnitude a sample must reach to be in the region
        [-].
    min_samples : int, optional
        Fewest finite samples the region may hold for a verdict [-].
    agreement : float, optional
        Smallest fraction of the region that must share the median's sign [-].

    Returns
    -------
    SignalOrientation
        ``multiplier`` (+1 or -1 when ``resolved``, else +1), the median
        ``statistic``, the region's ``count``, and a ``reason`` when
        unresolved [-].

    Raises
    ------
    ValueError
        ``signal`` is not one-dimensional.

    Defaults
    --------
    ``active_fraction = 0.1``, ``min_samples = 8`` and ``agreement = 0.6`` are
    empirical estimates fixed when #307 was implemented; #307 sets the rule
    (median of an active region, never ``abs``, never per-sample) but records
    no derivation for the three numbers.

    Convention
    ----------
    The multiplier is meant to be applied to the *whole* signal once, for
    display or for a sign-convention check.  It never rectifies samples.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A signal that is genuinely bipolar over its active region -- a Mirnov
    fluctuation, a flux loop through a reversal -- is correctly left
    unresolved, which the caller must handle rather than assume.

    Provenance
    ----------
    .. [1] Issue #307, which established the rule and its defaults.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim > 1:
        raise ValueError(f"signal must be one-dimensional; got shape {values.shape}")
    values = values.ravel()
    finite = np.isfinite(values)
    if mask is not None:
        region = finite & np.asarray(mask, dtype=bool).ravel()
    else:
        peak = float(np.max(np.abs(values[finite]))) if finite.any() else 0.0
        region = finite & (np.abs(values) >= active_fraction * peak) if peak > 0 else finite & False
    chosen = values[region]
    if chosen.size < min_samples:
        return SignalOrientation(1, float("nan"), int(chosen.size), False,
                                 f"only {chosen.size} active samples (need {min_samples})")
    statistic = float(np.median(chosen))
    if statistic == 0.0:
        return SignalOrientation(1, statistic, int(chosen.size), False, "median is zero")
    sign = 1 if statistic > 0 else -1
    share = float(np.mean(np.sign(chosen) == sign))
    if share < agreement:
        return SignalOrientation(1, statistic, int(chosen.size), False,
                                 f"only {share:.0%} of the active samples share the median's sign")
    return SignalOrientation(sign, statistic, int(chosen.size), True)
