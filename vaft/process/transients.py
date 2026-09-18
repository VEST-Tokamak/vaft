"""Measurements of a plasma-current transient on plain arrays.

A current quench and the brief rise of current that can precede it are the
two features of a plasma-current record a transient-event analysis starts
from.  Everything here measures them -- times, rates, amplitudes -- on a
``(time, current)`` pair and knows nothing about the machine that produced it.

Notation
--------
t          : time                                                  [s]
I_p        : plasma current, as stored                             [A]
I_ref      : reference current the fractions are taken of          [A]
t_80, t_20 : last 80 % and first 20 % crossings of the quench       [s]
dI/dt      : rate of change of the current magnitude             [A/s]

Conventions
-----------
**Numbers, not names.**  A quench found here is a fall of the current
magnitude through two fractions of a reference; a spike is a short positive
excursion above the local trend.  Neither is called an internal reconnection
event or a disruption: that is a classification, and it needs evidence -- the
magnetic fluctuation, the radiation, the position -- that a current record
alone does not carry.

**The sign of the stored current is removed.**  Every quantity is measured on
``polarity * I_p``, where ``polarity`` is the sign of the current at its
largest magnitude, and the polarity is returned, so a machine that stores a
negative current gets the same numbers as one that stores a positive one.

**Absence is an answer.**  A record with no quench, or no spike above its own
noise, returns a result whose measured fields are ``None`` and whose
``reason`` says why, never the record bounds or its loudest noise sample.

Provenance
----------
.. [ITPA2007] T. C. Hender *et al.*, "Chapter 3: MHD stability, operational
   limits and disruptions", *Nucl. Fusion* **47** (2007) S128, section 5.1:
   the 80 %-20 % current-quench time and its extrapolation by 1/0.6.
.. [Jung2022] E. C. Jung *et al.*, "Observation of MHD-correlated blobs during
   internal reconnection events in VEST", *Nucl. Fusion* **62** (2022) 126029.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .onset import MAD_TO_SIGMA, median_smooth

__all__ = [
    "CurrentQuench",
    "CurrentSpike",
    "current_quench",
    "current_spike",
]

#: The two fractions of the reference current the quench is timed between.
QUENCH_HIGH_FRACTION = 0.8
QUENCH_LOW_FRACTION = 0.2

#: How far before the quench a spike is looked for, and the trend it stands above.
SPIKE_LOOKBACK_S = 2e-3
SPIKE_TREND_S = 0.5e-3
#: Robust standard deviations of the residual a spike must exceed.
SPIKE_N_SIGMA = 5.0


@dataclass(frozen=True)
class CurrentQuench:
    """The timing and rate of one current quench, or why none was measured."""

    reference_current: float | None
    reference_time: float | None
    time_80: float | None
    time_20: float | None
    duration_80_20: float | None
    extrapolated_quench_time: float | None
    didt_min: float | None
    time_didt_min: float | None
    polarity: int
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class CurrentSpike:
    """A positive excursion of the current magnitude above its local trend."""

    time: float | None
    amplitude: float | None
    relative_amplitude: float | None
    trend_current: float | None
    noise_sigma: float | None
    before: float | None
    polarity: int
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.reason is None


def _record(time, current) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(current, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError(f"time and current must have equal length; got {t.size} and {y.size}")
    if t.size < 2:
        raise ValueError("a current record needs at least two samples")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    return t, y


def _polarity(current: np.ndarray) -> int:
    finite = np.where(np.isfinite(current), current, 0.0)
    peak = finite[int(np.argmax(np.abs(finite)))]
    return -1 if peak < 0 else 1


def _crossing(t: np.ndarray, y: np.ndarray, left: int, level: float) -> float:
    """Linear-interpolation time where ``y`` crosses ``level`` between ``left`` and ``left + 1``."""
    y0, y1 = y[left], y[left + 1]
    if y1 == y0:
        return float(t[left])
    fraction = (level - y0) / (y1 - y0)
    return float(t[left] + np.clip(fraction, 0.0, 1.0) * (t[left + 1] - t[left]))


def _kernel(t: np.ndarray, duration: float) -> int:
    samples = int(round(float(duration) / float(np.median(np.diff(t)))))
    return max(3, samples + (1 - samples % 2))


def current_quench(
    time,
    ip,
    *,
    window: Sequence[float] | None = None,
    reference_current: float | None = None,
    smoothing_s: float | None = None,
) -> CurrentQuench:
    """Time a current quench between 80 % and 20 % of a reference current, and its fastest fall.

    Parameters
    ----------
    time : array_like
        Strictly increasing sample times [s].
    ip : array_like
        Plasma current, either sign [A].
    window : sequence of float, optional
        ``(start, stop)`` the search is restricted to [s].
    reference_current : float, optional
        Current the fractions are taken of; default the largest magnitude in the
        window [A].
    smoothing_s : float, optional
        Width of a quadratic Savitzky-Golay window the rate is taken with; ``None``
        takes centred differences of the record as it is [s].

    Returns
    -------
    CurrentQuench
        The reference current in amperes, the 80 % and 20 % crossing times, their
        separation and its extrapolation to a full quench in seconds, the most
        negative rate of change of the current magnitude in amperes per second
        and when it occurred, the polarity, and ``reason`` when nothing was
        measured [-].

    Processing steps
    ----------------
    1. Restrict to the window and flip the record to positive polarity.
    2. Take the reference current, and the time of the largest magnitude.
    3. Find the first sample after that time below 20 % of the reference, and
       the last sample before it at or above 80 %; interpolate both crossings.
       With no sample at or above 80 % in between -- a ``reference_current``
       above the peak divided by 0.8 -- stop with a reason.
    4. Differentiate the magnitude -- centred differences, or a quadratic
       Savitzky-Golay derivative over ``smoothing_s`` -- and take its minimum
       between the two crossings.

    Convention
    ----------
    **Measured on the magnitude.** ``didt_min`` is the most negative rate of the
    polarity-corrected current, so a falling current reads negative whichever
    sign the machine stores. The rate is searched between the two crossings
    only, so the sharp fall at the end of a spike before the 80 % crossing is
    not mistaken for the quench rate.

    ``extrapolated_quench_time`` is ``duration_80_20 / 0.6``, the ITPA convention
    for the time a linear quench would take from 100 % to zero.

    Defaults
    --------
    The 80 % and 20 % fractions are the conventional ITPA disruption-database
    values. With no ``reference_current`` the largest magnitude in the window is
    used, a numerical convenience.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The default reference is the largest magnitude, so a spike on the flat top
    raises it and moves both crossings slightly earlier; pass
    ``reference_current`` to use a plateau value instead. Only the first fall
    through 20 % after the largest magnitude is timed. A spike *inside* the
    quench still contributes the fall that ends it, which can be several times
    the mean quench rate; ``time_didt_min`` says where the minimum was, and
    :func:`current_spike` with ``before=`` set past it measures the spike. The raw
    derivative of a noisy record is dominated by the noise; ``smoothing_s``
    exists for that, and assumes uniform sampling (it uses the median step).

    Provenance
    ----------
    .. [1] [ITPA2007]_: the 80 %-20 % current-quench time.
    .. [2] A. Savitzky and M. J. E. Golay, *Anal. Chem.* **36** (1964) 1627, as
       implemented by :func:`scipy.signal.savgol_filter`: the smoothed rate.
    """
    t, y = _record(time, ip)
    polarity = _polarity(y)
    empty = dict(
        reference_current=None, reference_time=None, time_80=None, time_20=None,
        duration_80_20=None, extrapolated_quench_time=None, didt_min=None,
        time_didt_min=None, polarity=polarity,
    )
    if window is not None:
        keep = (t >= float(window[0])) & (t <= float(window[1]))
        t, y = t[keep], y[keep]
        if t.size < 2:
            return CurrentQuench(**empty, reason="fewer than two samples inside the window")
    magnitude = polarity * y
    magnitude = np.where(np.isfinite(magnitude), magnitude, np.nan)
    if not np.any(np.nan_to_num(magnitude) > 0):
        return CurrentQuench(**empty, reason="no current: the record never rises above zero")

    peak_index = int(np.nanargmax(magnitude))
    reference = float(magnitude[peak_index]) if reference_current is None else float(abs(reference_current))
    high, low = QUENCH_HIGH_FRACTION * reference, QUENCH_LOW_FRACTION * reference

    after = np.nonzero(magnitude[peak_index:] < low)[0]
    if after.size == 0:
        return CurrentQuench(
            **{**empty, "reference_current": reference, "reference_time": float(t[peak_index])},
            reason="no quench: the current never falls below 20 % of the reference after its peak",
        )
    index_20 = peak_index + int(after[0])
    above = np.nonzero(magnitude[peak_index:index_20] >= high)[0]
    if above.size == 0:
        # The record never holds 80 % of the reference between its peak and the
        # fall: a reference above peak / 0.8 has no 80 % crossing to time, and one
        # above peak / 0.2 leaves index_20 at the peak itself.
        return CurrentQuench(
            **{**empty, "reference_current": reference, "reference_time": float(t[peak_index])},
            reason=(
                "no 80 % crossing: the current never reaches 80 % of the reference "
                f"({high:.6g} A) before it falls below 20 %"
            ),
        )
    index_80 = peak_index + int(above[-1])
    # above[-1] < index_20 - peak_index, so both crossings have a right neighbour
    # and index_20 - 1 >= index_80 >= 0.
    time_80 = _crossing(t, magnitude, index_80, high)
    time_20 = _crossing(t, magnitude, index_20 - 1, low)

    if smoothing_s is None:
        rate = np.gradient(magnitude, t)
    else:
        from scipy.signal import savgol_filter

        window_samples = min(_kernel(t, smoothing_s), magnitude.size - (1 - magnitude.size % 2))
        rate = savgol_filter(
            np.nan_to_num(magnitude), window_samples, 2, deriv=1,
            delta=float(np.median(np.diff(t))),
        )
    segment = slice(index_80, index_20 + 1)
    local = int(np.nanargmin(rate[segment]))
    duration = time_20 - time_80
    return CurrentQuench(
        reference_current=reference,
        reference_time=float(t[peak_index]),
        time_80=time_80,
        time_20=time_20,
        duration_80_20=float(duration),
        extrapolated_quench_time=float(duration / (QUENCH_HIGH_FRACTION - QUENCH_LOW_FRACTION)),
        didt_min=float(rate[segment][local]),
        time_didt_min=float(t[index_80 + local]),
        polarity=polarity,
    )


def current_spike(
    time,
    ip,
    *,
    before: float | None = None,
    lookback_s: float = SPIKE_LOOKBACK_S,
    trend_s: float = SPIKE_TREND_S,
    n_sigma: float = SPIKE_N_SIGMA,
) -> CurrentSpike:
    """Measure a short positive rise of the current magnitude above its trend, before a quench.

    Parameters
    ----------
    time : array_like
        Strictly increasing sample times [s].
    ip : array_like
        Plasma current, either sign [A].
    before : float, optional
        End of the search; default the quench's 80 % crossing from
        :func:`current_quench` [s].
    lookback_s : float, optional
        Length of the search window ending at ``before`` [s].
    trend_s : float, optional
        Width of the centred median filter that defines the trend [s].
    n_sigma : float, optional
        Robust standard deviations of the residual a spike must exceed [-].

    Returns
    -------
    CurrentSpike
        The time of the largest excursion in seconds, its height above the trend
        in amperes and as a fraction of the trend current, the trend current and
        the noise's robust sigma in amperes, the search end, the polarity, and
        ``reason`` when nothing qualified [-].

    Processing steps
    ----------------
    1. Flip the record to positive polarity.
    2. Take ``before`` as given, or the 80 % crossing of the quench; with neither,
       stop with a reason.
    3. Median-filter the magnitude over ``trend_s`` and subtract it: an excursion
       shorter than half that width survives in the residual, the trend does not.
    4. Within ``[before - lookback_s, before]`` take the largest residual above
       the residual's median, and the noise as the robust sigma of the first
       differences over the same samples divided by the square root of two.
    5. Accept it only above ``n_sigma`` robust sigmas.

    Convention
    ----------
    **Positive on the magnitude.** A spike is a rise of the polarity-corrected
    current, so on a machine that stores a negative current it is a more negative
    excursion of the stored value. Its amplitude is reported positive.

    Defaults
    --------
    ``lookback_s = 2 ms`` and ``trend_s = 0.5 ms`` are empirical, a numerical
    convenience sized for sub-millisecond excursions on a current that changes over
    milliseconds, as in a small tokamak; a longer discharge wants both scaled.
    ``n_sigma = 5`` is a numerical convenience that keeps the largest of a few
    hundred Gaussian residual samples below the threshold.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    One spike: the largest excursion in the window. An excursion longer than
    half of ``trend_s`` is absorbed into the trend and not reported. Pickup from a
    coil switching at the same time looks identical in a current record; nothing
    here can tell them apart.

    Provenance
    ----------
    .. [1] [Jung2022]_: the positive plasma-current spike observed with the
       internal reconnection events of VEST; measured here as a residual above a
       median trend, which asserts nothing about its cause.
    """
    t, y = _record(time, ip)
    polarity = _polarity(y)
    magnitude = polarity * y
    empty = dict(
        time=None, amplitude=None, relative_amplitude=None, trend_current=None,
        noise_sigma=None, polarity=polarity,
    )
    if before is None:
        quench = current_quench(t, y)
        if not quench.found:
            return CurrentSpike(
                **empty, before=None,
                reason=f"no before= and no quench to search before ({quench.reason})",
            )
        before = quench.time_80
    before = float(before)

    trend = median_smooth(magnitude, _kernel(t, trend_s))
    residual = magnitude - trend
    inside = (t >= before - float(lookback_s)) & (t <= before)
    if np.count_nonzero(inside) < 3:
        return CurrentSpike(**empty, before=before, reason="fewer than three samples in the search window")
    window_residual = residual[inside]
    median = float(np.nanmedian(window_residual))
    # The noise is measured on first differences, which a trend cannot inflate:
    # the residual of a running median shrinks wherever the current falls
    # faster than its noise, so its own spread under-reports the noise there.
    steps = np.diff(magnitude[inside])
    sigma = float(MAD_TO_SIGMA * np.nanmedian(np.abs(steps - np.nanmedian(steps))) / np.sqrt(2.0))
    local = int(np.nanargmax(window_residual))
    height = float(window_residual[local] - median)
    index = int(np.nonzero(inside)[0][local])
    if not height > float(n_sigma) * sigma:
        return CurrentSpike(
            **{**empty, "noise_sigma": sigma}, before=before,
            reason=(
                f"no excursion above {n_sigma:g} robust sigma of the residual noise "
                f"({height:.4g} A against {float(n_sigma) * sigma:.4g} A)"
            ),
        )
    trend_current = float(trend[index])
    return CurrentSpike(
        time=float(t[index]),
        amplitude=height,
        relative_amplitude=height / trend_current if trend_current else float("nan"),
        trend_current=trend_current,
        noise_sigma=sigma,
        before=before,
        polarity=polarity,
    )
