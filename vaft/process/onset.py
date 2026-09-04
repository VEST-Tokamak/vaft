"""Onset and active-window primitives on plain arrays (issue #409).

Every detector here answers one question about one waveform -- *when does it
become active, and on what evidence* -- and returns an :class:`OnsetRecord`.
Nothing here knows what the waveform is.  The diagnostic preprocessing, the
choice of which signal is authoritative for a *plasma* onset and the verdicts
built on top live in ``vaft.omas.plasma_timing`` and ``vaft.validation``.

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
    "PulseWindow",
    "RunFeatures",
    "active_window",
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


@dataclass(frozen=True)
class PulseWindow:
    """When a waveform is active: its onset, its offset, and the segments between.

    ``segments`` are the runs above threshold that passed persistence and
    morphology, in time order, after dips shorter than ``gap_s`` were bridged
    and re-emergences within ``post_quiet_s`` of a segment's end were merged
    into it.  The window is the envelope ``[onset, offset]`` of the segments a
    caller asked for -- all of them, or only the one holding the global
    maximum.  ``offset.time`` is the last sample of the pulse: the last sample
    above the end threshold, or, when the window says ``offset_from_collapse``,
    the last sample of the quench that ended it (the tail after it stays
    above the threshold).  Either way it is a grid sample, so a half-open
    consumer window is ``[onset.time, t[offset.index + 1])``.

    A window is never assumed: a pulse still active at the last sample is
    reported with ``offset_at_record_end``, one already active at the first
    with ``onset_at_record_start``, and more than one segment with
    ``multiple_segments`` so nobody averages across a gap unknowingly.
    """

    onset: OnsetRecord
    offset: OnsetRecord
    segments: tuple[RunFeatures, ...] = ()
    flags: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.onset.found and self.offset.found

    @property
    def start(self) -> float | None:
        return self.onset.time

    @property
    def end(self) -> float | None:
        return self.offset.time

    @property
    def duration_s(self) -> float | None:
        if not self.found:
            return None
        return float(self.offset.time - self.onset.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "duration_s": self.duration_s,
            "flags": list(self.flags),
            "segments": [s.as_dict() for s in self.segments],
            "onset": self.onset.as_dict(),
            "offset": self.offset.as_dict(),
            "evidence": dict(self.evidence),
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
    # anti-alias: not a resample -- the same grid in and out; this only fills
    # the non-finite samples from their finite neighbours.
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
    padlen = 3 * (int(order) + 1)  # filtfilt's default for a b/a filter of this order
    if y.size <= padlen:
        raise ValueError(
            f"a zero-phase low-pass of order {order} needs more than {padlen} samples; "
            f"the record has {y.size}"
        )
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


def _extend_forward(above: np.ndarray, start: int, quiet: int) -> int:
    """First index after ``start`` that is not part of the active run, where a
    run resuming within ``quiet`` samples of a stop continues the same run."""
    stop = start
    while stop < above.size and above[stop]:
        stop += 1
    while stop < above.size:
        nxt = stop
        while nxt < above.size and not above[nxt]:
            nxt += 1
        if nxt >= above.size or nxt - stop > quiet:
            break
        stop = nxt
        while stop < above.size and above[stop]:
            stop += 1
    return stop


def _bridged(mask: np.ndarray, bridge_samples: int) -> np.ndarray:
    """``mask`` with False gaps of at most ``bridge_samples`` filled in.

    A single noise sample below the threshold during a rise must not split
    the run: the fragment before it would fail persistence and the onset
    would move past the dip.
    """
    if bridge_samples <= 0:
        return mask
    out = mask.copy()
    for start, stop in _runs(~mask):
        if stop - start <= bridge_samples and start > 0 and stop < mask.size:
            out[start:stop] = True
    return out


def _brief_run(t: np.ndarray, y: np.ndarray, baseline: float, start: int, stop: int) -> RunFeatures:
    """Features of a run that failed persistence: cheap, prominence not evaluated."""
    seg = y[start:stop] - baseline
    dt = float(np.median(np.diff(t)))
    i_max = start + int(np.argmax(seg))
    return RunFeatures(start_time=float(t[start]), end_time=float(t[stop - 1]),
                       width_s=float((stop - start) * dt), samples=int(stop - start),
                       peak=float(seg.max()), peak_time=float(t[i_max]),
                       prominence=float("nan"), integral=float(np.sum(seg) * dt))


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


def _lowpass_skipped(t: np.ndarray, cutoff_hz, fs) -> bool:
    """Whether a requested low-pass cannot be designed on this grid.

    A rule tuned on a fast grid names a cutoff in hertz; on a grid whose
    Nyquist frequency is at or below that cutoff the filter has nothing to
    remove and cannot be designed, so the record is used as it is and the
    detector says ``lowpass_skipped`` -- on every verdict, found or not.
    """
    if cutoff_hz is None or t.size < 2:
        return False
    rate = float(fs) if fs else 1.0 / float(np.median(np.diff(t)))
    return float(cutoff_hz) >= 0.5 * rate


def _lowpassed(y: np.ndarray, cutoff_hz, fs, dt: float, order: int) -> np.ndarray:
    """The zero-phase low-pass, or the record itself when there is no cutoff."""
    if cutoff_hz is None:
        return y
    return zero_phase_lowpass(y, float(cutoff_hz), float(fs) if fs else 1.0 / dt, order)


def _too_short(t: np.ndarray, cutoff_hz, order: int) -> bool:
    """Whether a record has too few samples to be judged at all.

    Two samples cannot carry a threshold, and a zero-phase filter that will
    run needs more than its padding length; either way the answer is *no
    evidence*, flagged ``record_too_short``, not an exception a consumer has
    to guess at.  ``cutoff_hz`` is ``None`` here when the filter is skipped.
    """
    if t.size < 2:
        return True
    return cutoff_hz is not None and t.size <= 3 * (int(order) + 1)


def _with_flag(result, flag: str):
    """``result`` (an :class:`OnsetRecord` or :class:`PulseWindow`) with ``flag`` added."""
    if isinstance(result, PulseWindow):
        return PulseWindow(
            onset=_with_flag(result.onset, flag),
            offset=_with_flag(result.offset, flag),
            segments=result.segments,
            flags=tuple(dict.fromkeys((*result.flags, flag))),
            evidence=result.evidence,
        )
    return OnsetRecord(
        time=result.time, index=result.index, method=result.method, evidence=result.evidence,
        flags=tuple(dict.fromkeys((*result.flags, flag))), rejected=result.rejected,
        accepted=result.accepted,
    )


def _reference(t: np.ndarray, reference_mask, reference_fraction: float) -> np.ndarray:
    if reference_mask is not None:
        return np.asarray(reference_mask, dtype=bool).reshape(-1)
    n = max(2, int(round(float(reference_fraction) * t.size)))
    mask = np.zeros(t.size, dtype=bool)
    mask[:n] = True
    return mask


def _reference_shifted(y: np.ndarray, ref_mask: np.ndarray, spread: float, sigma: float) -> bool:
    """Whether the reference stretch itself changes level.

    Compares the medians of its first and last quarters: a step inside the
    reference makes the baseline the wrong level *and* hides the peak, so the
    threshold is meaningless even though every number is finite.
    """
    ref = y[ref_mask]
    ref = ref[np.isfinite(ref)]
    if ref.size < 8:
        return False
    quarter = ref.size // 4
    first, last = float(np.median(ref[:quarter])), float(np.median(ref[-quarter:]))
    if not np.isfinite(spread) or spread <= 0.0:
        return first != last
    return abs(last - first) > float(sigma) * spread


def _settle_reference(y: np.ndarray, ref_mask: np.ndarray, sigma: float) -> tuple[np.ndarray, tuple[str, ...]]:
    """The reference to threshold against, repaired when it is contaminated.

    A transient inside the reference (a pre-ionization pulse, a coil firing
    earlier than expected) shifts its level; the earliest quarter, before the
    contamination, is still a baseline.  Returns the mask to use and the
    ``reference_contaminated`` flag when that repair was made.
    """
    baseline, spread = robust_baseline(y, ref_mask)
    if not _reference_shifted(y, ref_mask, spread, sigma):
        return ref_mask, ()
    idx = np.flatnonzero(ref_mask)
    quarter = idx[: max(2, idx.size // 4)]
    repaired = np.zeros_like(ref_mask)
    repaired[quarter] = True
    return repaired, ("reference_contaminated",)


def _degenerate(t, y, method, baseline, spread, peak, threshold, ref_mask, sigma) -> OnsetRecord | None:
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
    bridge_samples: int = 2,
) -> OnsetRecord:
    """First run above the threshold that persists and has the right shape.

    The threshold is :func:`excess_threshold`; a run must last at least
    ``hold_s``, be at least ``min_width_s`` wide, rise at least
    ``min_prominence_sigma`` robust sigmas above its surroundings and carry at
    least ``min_integral_fraction`` of the record's total positive excess.
    Runs that fail are kept in ``rejected`` with the reason.

    ``prefilter_samples`` applies :func:`median_smooth` first, which is how an
    optical channel's isolated spikes are removed before they are counted.
    Gaps of at most ``bridge_samples`` below the threshold do not split a run.
    The returned time is the first sample of the accepted run.
    """
    t, raw = _as_arrays(time, values)
    y = median_smooth(raw, prefilter_samples) if prefilter_samples > 1 else _fill_non_finite(raw)
    ref, ref_flags = _settle_reference(y, _reference(t, reference_mask, reference_fraction), sigma)
    baseline, spread, peak, threshold = excess_threshold(
        y, ref, fraction=fraction, sigma=sigma, search_mask=search_mask
    )
    method = "sustained_excess"
    degenerate = _degenerate(t, y, method, baseline, spread, peak, threshold, ref, sigma)
    if degenerate is not None:
        return degenerate
    dt = float(np.median(np.diff(t)))
    hold = max(1, int(round(float(hold_s) / dt)))
    above = _bridged(y > threshold, int(bridge_samples))
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
        "bridge_samples": int(bridge_samples),
    }
    for start, stop in _runs(above):
        why = None
        if stop - start < hold:
            # persistence needs no shape features; the prominence they carry is
            # an O(n) evaluation per run and most runs on a noisy record end here
            why = "persistence"
            feats = _brief_run(t, y, baseline, start, stop)
        else:
            feats = run_features(t, y, baseline, start, stop)
        if why is None and feats.width_s < float(min_width_s):
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
        flags: list[str] = list(ref_flags)
        if start == 0:
            flags.append("onset_at_record_start")
        if ref[start]:
            flags.append("onset_inside_reference")
        evidence["isolated_excursions_before_onset"] = n_rejected
        evidence["n_rejected"] = n_rejected
        return OnsetRecord(
            time=float(t[start]), index=int(start), method=method, evidence=evidence,
            flags=tuple(flags), rejected=tuple(rejected), accepted=feats,
        )
    evidence["n_rejected"] = n_rejected
    flags = ["no_onset", *ref_flags]
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
    bridge_samples: int = 2,
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
    skipped = _lowpass_skipped(t, cutoff_hz, fs)
    if skipped:
        cutoff_hz = None
    if _too_short(t, cutoff_hz, order):
        record = OnsetRecord(time=None, index=None, method="principal_pulse",
                             evidence={"n_samples": int(t.size)}, flags=("no_onset", "record_too_short"))
        return _with_flag(record, "lowpass_skipped") if skipped else record
    record = _principal_pulse_onset(
        t, raw, fraction=fraction, sigma=sigma, reference_mask=reference_mask,
        reference_fraction=reference_fraction, search_mask=search_mask, cutoff_hz=cutoff_hz,
        fs=fs, order=order, pickup_floor=pickup_floor, impulse_max_s=impulse_max_s,
        bridge_samples=bridge_samples,
    )
    return _with_flag(record, "lowpass_skipped") if skipped else record


def _principal_pulse_onset(
    t, raw, *, fraction, sigma, reference_mask, reference_fraction, search_mask, cutoff_hz, fs,
    order, pickup_floor, impulse_max_s, bridge_samples,
) -> OnsetRecord:
    y = _fill_non_finite(raw)
    dt = float(np.median(np.diff(t)))
    y = _lowpassed(y, cutoff_hz, fs, dt, order)
    ref, ref_flags = _settle_reference(y, _reference(t, reference_mask, reference_fraction), sigma)
    baseline, spread, peak, threshold = excess_threshold(
        y, ref, fraction=fraction, sigma=sigma, search_mask=search_mask
    )
    method = "principal_pulse"
    degenerate = _degenerate(t, y, method, baseline, spread, peak, threshold, ref, sigma)
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
    if search_mask is not None:
        sel = np.asarray(search_mask, dtype=bool).reshape(-1)
        if not sel.any():
            return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                               flags=("no_onset", "search_mask_empty"))
        region = np.where(sel, y, -np.inf)
    else:
        region = y
    i_peak = int(np.argmax(region))
    above = _bridged(y > threshold, int(bridge_samples))
    start = i_peak
    while start > 0 and above[start - 1]:
        start -= 1
    stop = i_peak + 1
    while stop < y.size and above[stop]:
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
    flags: list[str] = list(ref_flags)
    if start == 0:
        flags.append("onset_at_record_start")
    if ref[start]:
        flags.append("onset_inside_reference")
    return OnsetRecord(time=float(t[start]), index=int(start), method=method,
                       evidence=evidence, flags=tuple(flags), accepted=feats)


def active_window(
    time,
    values,
    *,
    fraction: float = 0.02,
    sigma: float = 5.0,
    hold_s: float = 5.0e-4,
    min_width_s: float = 0.0,
    min_prominence_sigma: float = 0.0,
    min_integral_fraction: float = 0.0,
    gap_s: float = 1.0e-3,
    post_quiet_s: float = 2.0e-3,
    principal_only: bool = False,
    pickup_floor: float = 0.0,
    impulse_max_s: float = 2.0e-3,
    reference_mask=None,
    reference_fraction: float = 0.2,
    trailing_fraction: float = 0.1,
    trailing_max_fraction: float = 0.1,
    end_fraction: float | None = None,
    collapse_fallback: bool = True,
    collapse_rate_fraction: float = 0.10,
    collapse_min_drop: float = 0.5,
    search_mask=None,
    prefilter_samples: int = 1,
    cutoff_hz: float | None = None,
    fs: float | None = None,
    order: int = 4,
) -> PulseWindow:
    """The window over which a waveform is active, onset to offset.

    Threshold as :func:`excess_threshold`; runs above it with dips shorter
    than ``gap_s`` bridged; each run must pass persistence (``hold_s``) and
    morphology (``min_width_s``, ``min_prominence_sigma``,
    ``min_integral_fraction``) to be a segment; a segment beginning within
    ``post_quiet_s`` of the previous one's end is merged into it, so a brief
    quiet moment does not end the window while a real gap does.

    ``principal_only`` keeps just the segment holding the global maximum
    (the plasma-current pulse rather than any earlier light), guarded by
    the impulsive-run and ``pickup_floor`` rules of
    :func:`principal_pulse_onset`; otherwise the window is the envelope of
    every segment and ``multiple_segments`` says when there is more than one.

    The baseline after a pulse need not be the one before it -- a
    plasma-current record settles a few percent of its peak above zero once
    the plasma is gone -- so the *offset* is judged against a trailing
    reference: the last ``trailing_fraction`` of the record, when that stretch
    is quiet (spread within three times the leading one) and its level lies
    below ``trailing_max_fraction`` of the peak.  Otherwise the record is
    still active at its end and the window says ``offset_at_record_end``.

    ``end_fraction`` is the fraction of the peak the signal must fall below,
    above that trailing level, for the window to end (``fraction`` when
    ``None``).  A plasma-current record needs a higher one than its onset:
    after a termination the Rogowski keeps reading the induced vessel
    current, a few to ten percent of the peak decaying over tens of
    milliseconds, which no level between it and the trailing baseline
    separates from plasma; the collapse does, and 10 % of the peak is below
    every plasma and above every such tail on the VEST corpus -- except a
    disruption, whose vessel-current tail can sit at a third of the peak for
    a hundred milliseconds.  When the signal never falls below the end
    threshold, ``collapse_fallback`` ends the window at the end of the
    **last** steep fall after the peak (falling at ``collapse_rate_fraction``
    of the steepest rate or faster, and removing at least ``collapse_min_drop``
    of the current present before it, as a quench does) and says
    ``offset_from_collapse``.  The
    last fall, not the steepest: a plasma survives a mid-pulse drop and
    terminates later.  On the VEST corpus this puts the plasma-current
    offset within 2 ms of the light's on every discharge, level or collapse.
    """
    t, raw = _as_arrays(time, values)
    method = "active_window"
    skipped = _lowpass_skipped(t, cutoff_hz, fs)
    if skipped:
        cutoff_hz = None
    if _too_short(t, cutoff_hz, order):
        none = OnsetRecord(time=None, index=None, method=method, evidence={"n_samples": int(t.size)},
                           flags=("no_onset", "record_too_short"))
        window = PulseWindow(onset=none, offset=none, flags=none.flags, evidence=none.evidence)
        return _with_flag(window, "lowpass_skipped") if skipped else window
    window = _active_window(
        t, raw, method, fraction=fraction, sigma=sigma, hold_s=hold_s, min_width_s=min_width_s,
        min_prominence_sigma=min_prominence_sigma, min_integral_fraction=min_integral_fraction,
        gap_s=gap_s, post_quiet_s=post_quiet_s, principal_only=principal_only,
        pickup_floor=pickup_floor, impulse_max_s=impulse_max_s, reference_mask=reference_mask,
        reference_fraction=reference_fraction, trailing_fraction=trailing_fraction,
        trailing_max_fraction=trailing_max_fraction, end_fraction=end_fraction,
        collapse_fallback=collapse_fallback, collapse_rate_fraction=collapse_rate_fraction,
        collapse_min_drop=collapse_min_drop, search_mask=search_mask,
        prefilter_samples=prefilter_samples, cutoff_hz=cutoff_hz, fs=fs, order=order,
    )
    return _with_flag(window, "lowpass_skipped") if skipped else window


def _active_window(
    t, raw, method, *, fraction, sigma, hold_s, min_width_s, min_prominence_sigma,
    min_integral_fraction, gap_s, post_quiet_s, principal_only, pickup_floor, impulse_max_s,
    reference_mask, reference_fraction, trailing_fraction, trailing_max_fraction, end_fraction,
    collapse_fallback, collapse_rate_fraction, collapse_min_drop, search_mask, prefilter_samples,
    cutoff_hz, fs, order,
) -> PulseWindow:
    y = median_smooth(raw, prefilter_samples) if prefilter_samples > 1 else _fill_non_finite(raw)
    dt = float(np.median(np.diff(t)))
    y = _lowpassed(y, cutoff_hz, fs, dt, order)
    ref, ref_flags = _settle_reference(y, _reference(t, reference_mask, reference_fraction), sigma)
    baseline, spread, peak, threshold = excess_threshold(
        y, ref, fraction=fraction, sigma=sigma, search_mask=search_mask
    )
    evidence: dict[str, Any] = {
        "baseline_median": baseline, "robust_sigma": spread, "peak": peak, "threshold": threshold,
        "fraction": float(fraction), "sigma": float(sigma), "hold_s": float(hold_s),
        "gap_s": float(gap_s), "post_quiet_s": float(post_quiet_s), "cutoff_hz": cutoff_hz,
        "principal_only": bool(principal_only), "prefilter_samples": int(prefilter_samples),
    }

    def empty(flags: tuple[str, ...]) -> PulseWindow:
        none = OnsetRecord(time=None, index=None, method=method, evidence=evidence, flags=flags)
        return PulseWindow(onset=none, offset=none, flags=flags, evidence=evidence)

    degenerate = _degenerate(t, y, method, baseline, spread, peak, threshold, ref, sigma)
    if degenerate is not None:
        return empty(degenerate.flags)
    if np.isfinite(peak) and peak < float(sigma) * spread:
        return empty(("no_onset", "peak_below_noise"))

    hold = max(1, int(round(float(hold_s) / dt)))
    gap = max(0, int(round(float(gap_s) / dt)))
    quiet = max(0, int(round(float(post_quiet_s) / dt)))
    above = _bridged(y > threshold, gap)
    if search_mask is not None:
        above &= np.asarray(search_mask, dtype=bool).reshape(-1)
    total = float(np.sum(np.clip(y - baseline, 0.0, None)) * dt)

    # qualifying runs -> segments
    kept: list[tuple[int, int]] = []
    rejected: list[tuple[float, str, RunFeatures]] = []
    for start, stop in _runs(above):
        if stop - start < hold:
            if len(rejected) < MAX_REJECTED_RUNS:
                rejected.append((float(t[start]), "persistence", _brief_run(t, y, baseline, start, stop)))
            continue
        feats = run_features(t, y, baseline, start, stop)
        why = None
        if feats.width_s < float(min_width_s):
            why = "width"
        elif spread > 0 and feats.prominence < float(min_prominence_sigma) * spread:
            why = "prominence"
        elif feats.integral < float(min_integral_fraction) * total:
            why = "integral"
        if why is not None:
            if len(rejected) < MAX_REJECTED_RUNS:
                rejected.append((float(t[start]), why, feats))
            continue
        kept.append((start, stop))
    evidence["n_rejected"] = len(rejected)
    if not kept:
        return empty(("no_onset",))

    # merge segments separated by less than the post-quiet time
    merged: list[tuple[int, int]] = [kept[0]]
    for start, stop in kept[1:]:
        if start - merged[-1][1] <= quiet:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))

    if principal_only:
        region = y if search_mask is None else np.where(np.asarray(search_mask, dtype=bool).reshape(-1), y, -np.inf)
        i_peak = int(np.argmax(region))
        chosen = [seg for seg in merged if seg[0] <= i_peak < seg[1]]
        if not chosen:
            # the maximum lies in a run that did not qualify (an impulse)
            return empty(("no_onset", "principal_run_impulsive"))
        first, last = chosen[0]
        peak_feats = run_features(t, y, baseline, first, last)
        if peak_feats.width_s < float(impulse_max_s):
            return empty(("no_onset", "principal_run_impulsive"))
        margin = max(1, int(round(float(impulse_max_s) / dt)))
        outside = np.r_[y[: max(0, first - margin)], y[min(y.size, last + margin):]]
        scale = pickup_scale(outside, baseline, spread, dt, impulse_max_s=impulse_max_s)
        evidence["pickup_scale"] = scale
        if pickup_floor and scale > 0.0 and peak < float(pickup_floor) * scale:
            return empty(("no_onset", "peak_below_pickup_floor"))
        segments = [chosen[0]]
    else:
        segments = merged

    first, last = segments[0][0], segments[-1][1]
    flags: list[str] = list(ref_flags)
    # Offset against the trailing reference, when there is a quiet one.
    n_trail = max(2, int(round(float(trailing_fraction) * y.size)))
    trail = np.zeros(y.size, dtype=bool)
    trail[-n_trail:] = True
    trail_baseline, trail_spread = robust_baseline(y, trail)
    trail_quiet = bool(
        np.isfinite(trail_baseline) and np.isfinite(trail_spread)
        and (trail_spread <= 3.0 * spread if spread > 0 else trail_spread == 0.0)
        and (trail_baseline - baseline) < float(trailing_max_fraction) * peak
    )
    evidence["trailing_baseline"] = trail_baseline
    evidence["trailing_sigma"] = trail_spread
    evidence["trailing_quiet"] = bool(trail_quiet)
    end_frac = float(fraction if end_fraction is None else end_fraction)
    if trail_quiet:
        end_threshold = trail_baseline + max(end_frac * peak, float(sigma) * trail_spread)
    elif end_frac > float(fraction):
        end_threshold = baseline + max(end_frac * peak, float(sigma) * spread)
    else:
        end_threshold = None
    if end_threshold is not None:
        above_end = _bridged(y > end_threshold, gap)
        seg0, seg1 = segments[-1]
        i_peak_last = seg0 + int(np.argmax(y[seg0:seg1]))
        stop = _extend_forward(above_end, i_peak_last, quiet)
        if stop < y.size:
            last = stop
            segments[-1] = (seg0, last)
            evidence["offset_threshold"] = float(end_threshold)
    # The last steep fall after the peak, always recorded; used when the level
    # never comes down (a disruption's vessel-current tail).
    seg_start = segments[-1][0]
    i_peak_seg = seg_start + int(np.argmax(y[seg_start:max(seg_start + 1, segments[-1][1])]))
    dy = np.gradient(y, dt)
    steepest = float(dy[i_peak_seg:].min()) if i_peak_seg < y.size else 0.0
    collapse_end: int | None = None
    if steepest < 0.0:
        steep = dy < float(collapse_rate_fraction) * steepest
        steep[:i_peak_seg] = False
        falls = [(a, b) for a, b in _runs(steep)
                 if (y[a] - y[min(b, y.size - 1)]) >= float(collapse_min_drop) * max(y[a] - baseline, 0.0)]
        if falls:
            collapse_end = int(min(falls[-1][1], y.size - 1))
            evidence["collapse_time"] = float(t[collapse_end])
    if last >= y.size and collapse_fallback and collapse_end is not None and collapse_end > first:
        last = collapse_end + 1
        segments[-1] = (segments[-1][0], last)
        flags.append("offset_from_collapse")
    if first == 0:
        flags.append("onset_at_record_start")
    if last >= y.size:
        flags.append("offset_at_record_end")
    if len(segments) > 1:
        flags.append("multiple_segments")
    if ref[first]:
        flags.append("onset_inside_reference")
    feats_all = tuple(run_features(t, y, baseline, a, b) for a, b in segments)
    onset = OnsetRecord(time=float(t[first]), index=int(first), method=method, evidence=evidence,
                        flags=tuple(flags), rejected=tuple(rejected), accepted=feats_all[0])
    offset = OnsetRecord(time=float(t[last - 1]), index=int(last - 1), method=method + "_offset",
                         evidence=evidence, flags=tuple(flags), accepted=feats_all[-1])
    evidence["n_segments"] = len(segments)
    return PulseWindow(onset=onset, offset=offset, segments=feats_all, flags=tuple(flags), evidence=evidence)


def principal_pulse_window(time, values, **kwargs) -> tuple[OnsetRecord, OnsetRecord]:
    """``(onset, offset)`` of the pulse holding the global maximum.

    One :func:`active_window` call with ``principal_only=True`` and the
    principal defaults (``pickup_floor`` 3, no persistence: the pulse is the
    run holding the maximum); the onset is the run's first sample, exactly
    what :func:`principal_pulse_onset` returns, and the offset carries the
    window's flags -- ``offset_at_record_end``, ``offset_from_collapse``.
    """
    kwargs.setdefault("pickup_floor", 3.0)
    kwargs.setdefault("hold_s", 0.0)
    window = active_window(time, values, principal_only=True, **kwargs)
    if not window.found:
        none = OnsetRecord(time=None, index=None, method="principal_pulse", evidence=dict(window.evidence),
                           flags=window.flags)
        return none, OnsetRecord(time=None, index=None, method="principal_pulse_offset",
                                 evidence=dict(window.evidence), flags=window.flags)
    onset = OnsetRecord(time=window.onset.time, index=window.onset.index, method="principal_pulse",
                        evidence=dict(window.evidence), flags=window.onset.flags,
                        rejected=window.onset.rejected, accepted=window.onset.accepted)
    offset = OnsetRecord(time=window.offset.time, index=window.offset.index,
                         method="principal_pulse_offset", evidence=dict(window.evidence),
                         flags=window.flags, accepted=window.offset.accepted)
    return onset, offset
