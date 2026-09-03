"""Onset and active-window primitives on plain arrays (issue #409).

Every detector here answers one question about one waveform -- *when does it
become active, and on what evidence* -- and returns an :class:`OnsetRecord`.
Nothing here knows what the waveform is.  The diagnostic preprocessing, the
choice of which signal is authoritative for a *plasma* onset and the verdicts
built on top live in ``vaft.omas.plasma_onset`` and ``vaft.validation``.

Three ideas, kept separable because a 205-shot study of the VEST raw database
showed each does a different job:

* **threshold** -- ``baseline + max(fraction * peak, sigma * robust_sigma)``.
  The fraction-of-peak term makes the boundary independent of a channel's
  noise floor (two H-alpha channels with a ten-fold noise difference agree to
  0.2 ms); the sigma term keeps a weak record from being thresholded at its own
  noise.
* **persistence** -- the signal must stay above threshold for ``hold_s``.
  0.5 ms removes the coil-firing pickup spikes from plasma-current records
  (1 false onset in 84 shots, from 51) and most isolated optical spikes.
* **morphology** -- width, prominence and integral of the accepted run.  Width
  is the one that matters on that corpus: the residual false onsets after
  persistence are 0.5-1 ms flashes on failed breakdowns and the noisy burst an
  ECH switch-on induces in every channel; ``min_width_s = 1 ms`` removes them
  (1 in 51 non-plasma shots, and that one is genuine light).

For a pulse-shaped record :func:`principal_pulse_onset` walks back from the
global maximum, so an isolated excursion is unreachable by construction; its
one weakness is a record with no pulse, where the maximum *is* the pickup --
:func:`pickup_scale` measures the record's own impulsive excursions and a
floor on the pulse amplitude relative to them refuses that case.

Two contracts every consumer can rely on: ``OnsetRecord.time`` is always a
sample of the input grid (or ``None``), which is what lets a half-open window
``[start, onset)`` exclude exactly the samples from the onset onward; and *no
onset is never the whole record* -- a detector that finds nothing says so with
``time=None`` and a flag, never by returning the record bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.signal as scipy_signal

from .signal_processing import butterworth_lowpass

__all__ = [
    "OnsetRecord",
    "RunFeatures",
    "excess_threshold",
    "isolated_excursions",
    "median_smooth",
    "pickup_scale",
    "principal_pulse_onset",
    "principal_pulse_window",
    "robust_baseline",
    "run_features",
    "sustained_excess_onset",
    "zero_phase_lowpass",
]

#: Gaussian-consistent scale of the median absolute deviation.
MAD_TO_SIGMA = 1.4826

#: How many rejected runs a record keeps in full; the count is always kept.
MAX_REJECTED_RUNS = 32


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunFeatures:
    """Temporal-shape features of one contiguous run above threshold."""

    start_time: float
    end_time: float
    width_s: float
    samples: int
    peak: float
    peak_time: float
    prominence: float
    integral: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "width_s": self.width_s,
            "samples": self.samples,
            "peak": self.peak,
            "peak_time": self.peak_time,
            "prominence": self.prominence,
            "integral": self.integral,
        }


@dataclass(frozen=True)
class OnsetRecord:
    """What one detector concluded about one waveform.

    ``time`` is a sample of the input time grid, or ``None`` when no onset
    qualifies.  ``evidence`` carries the numbers the decision was made with;
    ``rejected`` the runs that crossed the threshold but failed persistence or
    morphology, each with the reason -- the early faint light a consumer may
    want to know about is in there, not lost.
    """

    time: float | None
    index: int | None
    method: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    rejected: tuple[tuple[float, str, RunFeatures], ...] = ()
    accepted: RunFeatures | None = None

    @property
    def found(self) -> bool:
        return self.time is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "index": self.index,
            "method": self.method,
            "evidence": dict(self.evidence),
            "flags": list(self.flags),
            "accepted": None if self.accepted is None else self.accepted.as_dict(),
            "rejected": [
                {"time": t, "reason": why, **feats.as_dict()} for t, why, feats in self.rejected
            ],
        }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _as_arrays(time, values) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError(f"time and values differ in length: {t.size} vs {y.size}")
    if t.size < 2:
        raise ValueError("at least two samples are needed")
    return t, y


def _fill_non_finite(y: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(y)
    if not bad.any():
        return y
    out = y.copy()
    good = np.flatnonzero(~bad)
    if good.size == 0:
        out[:] = 0.0
        return out
    out[bad] = np.interp(np.flatnonzero(bad), good, y[good])
    return out


def robust_baseline(values, reference_mask=None) -> tuple[float, float]:
    """Median and robust sigma (``1.4826 * MAD``) over the reference samples.

    Non-finite samples are ignored.  Returns ``(nan, nan)`` when fewer than two
    finite reference samples exist, so a caller can flag the reference rather
    than threshold against garbage.  The MAD is computed here rather than
    through ``vaft.formula`` so this module stays a scipy-only import.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    if reference_mask is not None:
        y = y[np.asarray(reference_mask, dtype=bool).reshape(-1)]
    y = y[np.isfinite(y)]
    if y.size < 2:
        return float("nan"), float("nan")
    median = float(np.median(y))
    return median, float(MAD_TO_SIGMA * np.median(np.abs(y - median)))


def median_smooth(values, kernel_samples: int) -> np.ndarray:
    """Median filter that removes excursions shorter than half the kernel.

    Non-finite samples are linearly interpolated first.  ``kernel_samples``
    is forced odd; ``1`` returns the (finite-filled) input.
    """
    y = _fill_non_finite(np.asarray(values, dtype=float).reshape(-1))
    k = int(kernel_samples)
    if k <= 1:
        return y
    if k % 2 == 0:
        k += 1
    if k >= y.size:
        k = y.size if y.size % 2 else y.size - 1
        if k <= 1:
            return y
    return scipy_signal.medfilt(y, kernel_size=k)


def zero_phase_lowpass(values, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Forward-backward Butterworth low-pass: no group delay, so an onset
    read from the output is not shifted from the input's.

    A causal filter of the same order moved the plasma-current onset by
    +0.1 to +0.9 ms on VEST records; this one by less than a sample.
    """
    y = _fill_non_finite(np.asarray(values, dtype=float).reshape(-1))
    return np.asarray(butterworth_lowpass(y, float(cutoff_hz), float(fs), int(order), zero_phase=True))


# ---------------------------------------------------------------------------
# Threshold and run features
# ---------------------------------------------------------------------------


def excess_threshold(
    values,
    reference_mask,
    *,
    fraction: float,
    sigma: float,
    search_mask=None,
) -> tuple[float, float, float, float]:
    """``(baseline, robust_sigma, peak, threshold)``.

    ``peak`` is the largest excess over the baseline within ``search_mask``
    (the whole record when ``None``); ``threshold`` is
    ``baseline + max(fraction * peak, sigma * robust_sigma)``.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    baseline, spread = robust_baseline(y, reference_mask)
    if not np.isfinite(baseline):
        return baseline, spread, float("nan"), float("nan")
    region = y if search_mask is None else y[np.asarray(search_mask, dtype=bool).reshape(-1)]
    region = region[np.isfinite(region)]
    peak = float(region.max() - baseline) if region.size else float("nan")
    spread_term = float(sigma) * spread if np.isfinite(spread) else 0.0
    threshold = baseline + max(float(fraction) * max(peak, 0.0), spread_term)
    return baseline, spread, peak, float(threshold)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` index intervals where ``mask`` is True."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def run_features(time, values, baseline: float, start: int, stop: int) -> RunFeatures:
    """Width, peak, prominence and integral of ``values[start:stop] - baseline``."""
    t, y = _as_arrays(time, values)
    seg = y[start:stop] - baseline
    dt = float(np.median(np.diff(t)))
    i_max = start + int(np.argmax(seg))
    excess = y - baseline
    if 0 < i_max < y.size - 1:
        prominence = float(scipy_signal.peak_prominences(excess, [i_max])[0][0])
    else:
        prominence = float(seg.max())
    return RunFeatures(
        start_time=float(t[start]),
        end_time=float(t[stop - 1]),
        width_s=float((stop - start) * dt),
        samples=int(stop - start),
        peak=float(seg.max()),
        peak_time=float(t[i_max]),
        prominence=prominence,
        integral=float(np.sum(seg) * dt),
    )


def isolated_excursions(values, threshold: float, max_run_samples: int) -> int:
    """How many runs above ``threshold`` are shorter than ``max_run_samples``."""
    y = np.asarray(values, dtype=float).reshape(-1)
    return sum(1 for a, b in _runs(y > threshold) if (b - a) < int(max_run_samples))


def pickup_scale(values, baseline: float, robust_sigma: float, dt: float,
                 *, impulse_max_s: float = 2.0e-3, sigma: float = 6.0) -> float:
    """The record's own impulsive-excursion amplitude.

    The largest absolute excess over the baseline among runs above
    ``sigma * robust_sigma`` that last less than ``impulse_max_s`` -- on a
    plasma-current record, the coil-firing pickup measured on that very shot.
    ``0.0`` when there is none.  The bar is six robust sigmas rather than the
    detector's five: white noise does not reach six sigma in a record of a few
    thousand samples, so the scale measures pickup, not the noise tail.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    if not (np.isfinite(baseline) and np.isfinite(robust_sigma) and robust_sigma > 0):
        return 0.0
    excess = np.abs(y - baseline)
    max_samples = max(1, int(round(float(impulse_max_s) / float(dt))))
    peaks = [excess[a:b].max() for a, b in _runs(excess > sigma * robust_sigma) if (b - a) < max_samples]
    return float(max(peaks)) if peaks else 0.0


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _reference(t: np.ndarray, reference_mask, reference_fraction: float) -> np.ndarray:
    if reference_mask is not None:
        return np.asarray(reference_mask, dtype=bool).reshape(-1)
    n = max(2, int(round(float(reference_fraction) * t.size)))
    mask = np.zeros(t.size, dtype=bool)
    mask[:n] = True
    return mask


def _degenerate(t, y, method, baseline, spread, peak, threshold, ref_mask) -> OnsetRecord | None:
    """The cases where thresholding would be meaningless, as flagged records."""
    flags: list[str] = []
    if int(np.count_nonzero(ref_mask)) < 2:
        flags.append("reference_too_short")
    if not np.isfinite(baseline) or not np.isfinite(spread):
        flags.append("reference_not_finite")
    elif spread <= 0.0:
        flags.append("reference_flat")
    if not np.isfinite(peak):
        flags.append("no_finite_samples")
    if not flags:
        return None
    return OnsetRecord(
        time=None, index=None, method=method,
        evidence={"baseline_median": baseline, "robust_sigma": spread, "peak": peak, "threshold": threshold},
        flags=("no_onset", *flags),
    )


def sustained_excess_onset(
    time,
    values,
    *,
    fraction: float = 0.02,
    sigma: float = 5.0,
    hold_s: float = 5.0e-4,
    min_width_s: float = 0.0,
    min_prominence_sigma: float = 0.0,
    min_integral_fraction: float = 0.0,
    reference_mask=None,
    reference_fraction: float = 0.2,
    search_mask=None,
    prefilter_samples: int = 1,
) -> OnsetRecord:
    """First run above the threshold that persists and has the right shape.

    The threshold is :func:`excess_threshold`; a run must last at least
    ``hold_s``, be at least ``min_width_s`` wide, rise at least
    ``min_prominence_sigma`` robust sigmas above its surroundings and carry at
    least ``min_integral_fraction`` of the record's total positive excess.
    Runs that fail are kept in ``rejected`` with the reason.

    ``prefilter_samples`` applies :func:`median_smooth` first, which is how an
    optical channel's isolated spikes are removed before they are counted.
    The returned time is the first sample of the accepted run.
    """
    t, raw = _as_arrays(time, values)
    y = median_smooth(raw, prefilter_samples) if prefilter_samples > 1 else _fill_non_finite(raw)
    ref = _reference(t, reference_mask, reference_fraction)
    baseline, spread, peak, threshold = excess_threshold(
        y, ref, fraction=fraction, sigma=sigma, search_mask=search_mask
    )
    method = "sustained_excess"
    degenerate = _degenerate(t, y, method, baseline, spread, peak, threshold, ref)
    if degenerate is not None:
        return degenerate
    dt = float(np.median(np.diff(t)))
    hold = max(1, int(round(float(hold_s) / dt)))
    above = y > threshold
    if search_mask is not None:
        above &= np.asarray(search_mask, dtype=bool).reshape(-1)
    total = float(np.sum(np.clip(y - baseline, 0.0, None)) * dt)
    rejected: list[tuple[float, str, RunFeatures]] = []
    n_rejected = 0
    evidence: dict[str, Any] = {
        "baseline_median": baseline, "robust_sigma": spread, "peak": peak, "threshold": threshold,
        "fraction": float(fraction), "sigma": float(sigma), "hold_samples": hold,
        "min_width_s": float(min_width_s), "min_prominence_sigma": float(min_prominence_sigma),
        "min_integral_fraction": float(min_integral_fraction), "prefilter_samples": int(prefilter_samples),
    }
    for start, stop in _runs(above):
        feats = run_features(t, y, baseline, start, stop)
        why = None
        if stop - start < hold:
            why = "persistence"
        elif feats.width_s < float(min_width_s):
            why = "width"
        elif spread > 0 and feats.prominence < float(min_prominence_sigma) * spread:
            why = "prominence"
        elif feats.integral < float(min_integral_fraction) * total:
            why = "integral"
        if why is not None:
            n_rejected += 1
            if len(rejected) < MAX_REJECTED_RUNS:
                rejected.append((float(t[start]), why, feats))
            continue
        flags: list[str] = []
        if start == 0:
            flags.append("onset_at_record_start")
        if ref[start]:
            flags.append("reference_contaminated")
        evidence["isolated_excursions_before_onset"] = n_rejected
        evidence["n_rejected"] = n_rejected
        return OnsetRecord(
            time=float(t[start]), index=int(start), method=method, evidence=evidence,
            flags=tuple(flags), rejected=tuple(rejected), accepted=feats,
        )
    evidence["n_rejected"] = n_rejected
    flags = ["no_onset"]
    if np.isfinite(peak) and np.isfinite(spread) and peak < float(sigma) * spread:
        flags.append("peak_below_noise")
    return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                       flags=tuple(flags), rejected=tuple(rejected))


def principal_pulse_onset(
    time,
    values,
    *,
    fraction: float = 0.02,
    sigma: float = 5.0,
    reference_mask=None,
    reference_fraction: float = 0.2,
    search_mask=None,
    cutoff_hz: float | None = None,
    fs: float | None = None,
    order: int = 4,
    pickup_floor: float = 3.0,
    impulse_max_s: float = 2.0e-3,
) -> OnsetRecord:
    """Onset of the pulse that contains the global maximum.

    Optionally low-passes the record first (zero-phase), then walks back from
    the maximum while the excess stays above the threshold; the first sample
    of that connected run is the onset.  An excursion not connected to the
    maximum can never be chosen.

    A record with no pulse has its maximum *at* the pickup, so two floors
    guard the answer: the peak must exceed ``sigma`` robust sigmas
    (``peak_below_noise``) and ``pickup_floor`` times the record's own
    :func:`pickup_scale` (``peak_below_pickup_floor``).
    """
    t, raw = _as_arrays(time, values)
    y = _fill_non_finite(raw)
    dt = float(np.median(np.diff(t)))
    if cutoff_hz is not None:
        y = zero_phase_lowpass(y, float(cutoff_hz), float(fs) if fs else 1.0 / dt, order)
    ref = _reference(t, reference_mask, reference_fraction)
    baseline, spread, peak, threshold = excess_threshold(
        y, ref, fraction=fraction, sigma=sigma, search_mask=search_mask
    )
    method = "principal_pulse"
    degenerate = _degenerate(t, y, method, baseline, spread, peak, threshold, ref)
    if degenerate is not None:
        return degenerate
    evidence: dict[str, Any] = {
        "baseline_median": baseline, "robust_sigma": spread, "peak": peak, "threshold": threshold,
        "fraction": float(fraction), "sigma": float(sigma), "cutoff_hz": cutoff_hz,
        "pickup_floor": float(pickup_floor),
    }
    if peak < float(sigma) * spread:
        evidence["pickup_scale"] = pickup_scale(y, baseline, spread, dt, impulse_max_s=impulse_max_s)
        return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                           flags=("no_onset", "peak_below_noise"))
    region = y if search_mask is None else np.where(np.asarray(search_mask, dtype=bool).reshape(-1), y, -np.inf)
    i_peak = int(np.argmax(region))
    start = i_peak
    while start > 0 and y[start - 1] > threshold:
        start -= 1
    stop = i_peak + 1
    while stop < y.size and y[stop] > threshold:
        stop += 1
    feats = run_features(t, y, baseline, start, stop)
    # A pulse is never as brief as a pickup impulse: a principal run narrower
    # than ``impulse_max_s`` is the pickup itself (a record with no pulse has
    # its maximum at a coil-firing spike), whatever its amplitude.
    if feats.width_s < float(impulse_max_s):
        evidence["principal_width_s"] = feats.width_s
        return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                           flags=("no_onset", "principal_run_impulsive"), accepted=feats)
    # Pickup is what the record does outside the pulse: the pulse's own noisy
    # threshold crossings during its rise are not excursions, so the scale is
    # measured beyond one impulse length on either side of the run.
    margin = max(1, int(round(float(impulse_max_s) / dt)))
    outside = np.r_[y[: max(0, start - margin)], y[min(y.size, stop + margin):]]
    scale = pickup_scale(outside, baseline, spread, dt, impulse_max_s=impulse_max_s)
    evidence["pickup_scale"] = scale
    if pickup_floor and scale > 0.0 and peak < float(pickup_floor) * scale:
        return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                           flags=("no_onset", "peak_below_pickup_floor"), accepted=feats)
    evidence.update({"peak_time": float(t[i_peak]), "peak_index": i_peak,
                     "isolated_excursions_before_onset": isolated_excursions(
                         y[:start], threshold, max(1, int(round(impulse_max_s / dt))))})
    flags: list[str] = []
    if start == 0:
        flags.append("onset_at_record_start")
    if ref[start]:
        flags.append("reference_contaminated")
    return OnsetRecord(time=float(t[start]), index=int(start), method=method,
                       evidence=evidence, flags=tuple(flags), accepted=feats)


def principal_pulse_window(time, values, **kwargs) -> tuple[OnsetRecord, OnsetRecord]:
    """``(onset, offset)`` of the principal pulse; the offset is the last
    sample of the connected run, as its own record with ``method
    "principal_pulse_offset"``.
    """
    onset = principal_pulse_onset(time, values, **kwargs)
    if not onset.found or onset.accepted is None:
        return onset, OnsetRecord(time=None, index=None, method="principal_pulse_offset",
                                  evidence=dict(onset.evidence), flags=onset.flags)
    t, _ = _as_arrays(time, values)
    end_index = int(np.searchsorted(t, onset.accepted.end_time))
    offset = OnsetRecord(time=float(onset.accepted.end_time), index=end_index,
                         method="principal_pulse_offset", evidence=dict(onset.evidence),
                         flags=onset.flags, accepted=onset.accepted)
    return onset, offset
