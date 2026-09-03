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

    ``rtol`` is deliberately loose (0.1%).  VEST shots after 42190 store their
    timebase as ``linspace(0, span, n)``, so a nominally 4e-6 grid is really
    ``span / (n - 1)`` = 4.00016e-6; that is uniform for every purpose here and
    a tight tolerance would only reject it.
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

    ``firwin`` + ``filtfilt`` rather than a Butterworth or ``decimate``'s
    Chebyshev.  It is the idiom the VEST mappers already use
    (``machine_mapping/tf.py``, ``machine_mapping/pf_active.py``,
    ``process/magnetics.py``), and zero phase is load-bearing here -- filterscope
    intensity feeds onset detection, where a causal ``lfilter`` would move the
    onset by the filter's group delay.

    ``nan_policy="segment"`` filters each maximal finite run independently;
    ``filtfilt`` would otherwise smear a single NaN across the whole record.
    Runs too short for the filter are passed through with a ``RuntimeWarning``.

    ``stopband_hz`` is the frequency by which the response must be down -- for a
    downsample, the target Nyquist.  It sets the transition band and so the tap
    count; it defaults to ``1.25 * cutoff_hz``, the value consistent with
    :func:`resample_to_time`'s default 0.8-of-Nyquist cutoff.
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

    The default, ``anti_alias="auto"``, measures both grids and decides:

    * ``dt_target / dt_source <= min_ratio`` -- alignment, an upsample, or a
      rate change too small to matter.  No filter is designed and the result is
      **bit-for-bit** ``np.interp``.  This exactness is deliberate: it is what
      lets equal-rate call sites adopt the primitive without moving a single
      stored value.
    * otherwise -- a genuine rate reduction.  A zero-phase FIR low-pass runs on
      the source grid before interpolating.

    ``anti_alias=True`` demands the filter regardless of the measured ratio, and
    ``anti_alias=False`` is the escape hatch for the cases where filtering is
    wrong -- a validity mask, say, which is logical rather than bandlimited.
    Opting out is exact ``np.interp`` too, so the choice is visible in the diff
    rather than hidden in a numerical difference.

    Parameters
    ----------
    source_time, values, target_time
        ``values`` may carry leading axes; the time axis is ``axis``.
    cutoff_fraction
        Anti-alias cutoff as a fraction of the *target* Nyquist (default 0.8).
        ``firwin`` sits at -6 dB at its cutoff and needs a transition band, so
        a passband edge below Nyquist puts everything that can fold into the
        stopband.  ``cutoff_hz`` overrides this outright.
    min_ratio
        Rate ratio below which ``"auto"`` performs no filtering at all.
    extrapolate
        ``"clamp"`` (default, matching ``np.interp``), ``"nan"``, or ``"error"``
        for target samples outside the source's span.
    nan_policy
        Passed to :func:`anti_alias_filter`.
    on_unsorted
        ``"error"`` (default) or ``"sort"``.  ``np.interp`` returns silent
        nonsense for an unsorted ``x``, so this is checked rather than assumed.

    Raises
    ------
    ResamplingError
        Empty source, unsorted or duplicated source times under the default
        policy, a target outside the source span under ``extrapolate="error"``,
        or a forced filter whose cutoff exceeds the source Nyquist.
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
    """Legacy conditioning wrapper kept in the process layer.

    Order of operations is crop, resample, filter.  The resample goes through
    :func:`resample_to_time`, so reducing the sample rate anti-aliases on the
    input grid first; the ``filter_params`` stage that follows is a *shaping*
    filter and its ``cutoff`` is interpreted against the **output** grid's
    sample rate, not the input's.
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
    multiplier is ``+1`` (canonical), never a guess.  ``abs`` is never
    applied to the data and no sample is flipped on its own.
    """
    values = np.asarray(signal, dtype=float).ravel()
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
