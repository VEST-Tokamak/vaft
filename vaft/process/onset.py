"""Onset and active-window primitives on plain arrays (issue #409).

Every detector here answers one question about one waveform -- *when does it
become active, and on what evidence* -- and returns an :class:`OnsetRecord`.
Nothing here knows what the waveform is.  The diagnostic preprocessing, the
choice of which signal is authoritative for a *plasma* onset and the verdicts
built on top live in ``vaft.omas.plasma_timing`` and ``vaft.validation``.

Three ideas, kept separable because a study of the VEST raw database showed
each does a different job.  The study is ``workflow/plasma_onset/scan_corpus.py``
and its table ``test/data/onset_corpus.json`` (its ``summary`` carries the
counts); the figures quoted below are from the hand-reviewed first pass that
motivated the rules:

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
    "PeakRecord",
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
    "robust_peak",
    "run_features",
    "sustained_excess_onset",
    "zero_crossing_after_excursion",
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
            "rejected": _rejected_as_dicts(self.rejected),
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

    The MAD is computed here rather than through ``vaft.formula`` so this module
    stays a scipy-only import.

    Parameters
    ----------
    values : array_like
        The waveform [any].
    reference_mask : array_like of bool or None, optional
        Samples to measure over; ``None`` uses all of them [-].

    Returns
    -------
    baseline : float
        Median of the finite reference samples, ``nan`` when fewer than two
        exist [any].
    robust_sigma : float
        ``1.4826`` times the median absolute deviation about that median, on the
        same samples, ``nan`` under the same condition [any].

    Defaults
    --------
    The MAD-to-sigma factor ``1.4826`` (``MAD_TO_SIGMA``) is a physical constant
    of the Gaussian distribution, not a tuning value: it makes the returned
    spread equal the standard deviation for Gaussian noise.

    Assumptions
    -----------
    The reference samples are quiet.  Nothing here checks that; a caller that
    cannot guarantee it asks a detector, which settles the reference itself and
    flags ``reference_contaminated``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``(nan, nan)`` on fewer than two finite reference samples, so a caller can
    flag the reference rather than threshold against garbage.  A reference of
    identical samples gives ``robust_sigma == 0``; a threshold built on it
    degenerates to the fraction-of-peak term alone.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
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

    The prefilter every optical detector runs first: an isolated digitizer spike
    is not evidence of light, and a median of an odd kernel removes any excursion
    narrower than half of it without moving the edges of the ones it keeps.

    Parameters
    ----------
    values : array_like
        The waveform; non-finite samples are linearly interpolated first [any].
    kernel_samples : int
        Median kernel width in samples, forced odd; ``1`` or less returns the
        finite-filled input unchanged [-].

    Returns
    -------
    numpy.ndarray
        The filtered waveform, same length and grid as the input [any].

    Defaults
    --------
    None; the kernel is the caller's.  The VEST widths are a machine-specific
    setting [vest.yaml]: five samples on the slow optical and current
    channels, one (no filtering) where the rule wants the raw record.

    Convention
    ----------
    Zero-phase by construction: a median filter of an odd kernel introduces no
    group delay, so an onset read off the output is not shifted from the input's.
    An even ``kernel_samples`` is incremented rather than applied, because an even
    median averages two order statistics and does shift edges.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A kernel at or above the record length is clipped to the longest odd kernel
    that fits.  A flat top wider than the kernel survives, which is what lets
    :func:`robust_peak` still see a railed digitizer.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
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
    """Forward-backward Butterworth low-pass: no group delay.

    An onset read off the output is not shifted from the input's.  A causal filter
    of the same order moved the plasma-current onset by +0.1 to +0.9 ms on VEST
    records; this one by less than a sample.

    Parameters
    ----------
    values : array_like
        The waveform; non-finite samples are linearly interpolated first [any].
    cutoff_hz : float
        Low-pass cutoff [Hz].
    fs : float
        Sample rate of the record [Hz].
    order : int, optional
        Butterworth order; the filter is applied twice, so the effective
        roll-off is twice this [-].

    Returns
    -------
    numpy.ndarray
        The filtered waveform, same length and grid as the input [any].

    Raises
    ------
    ValueError
        When the record has no more samples than ``filtfilt``'s padding length,
        ``3 * (order + 1)``.  The detectors call :func:`_too_short` first and
        report ``record_too_short`` instead of raising.

    Defaults
    --------
    ``order = 4`` is a numerical convenience: the same order the VEST soft X-ray
    viewer and ``vaft.process.signal_processing`` use, steep enough to matter and
    short enough to pad on a few-thousand-sample record.

    Convention
    ----------
    Zero-phase (``scipy.signal.filtfilt``, forward then backward), which is
    mandatory wherever an onset time is read off the result and is the opposite
    choice from ``vaft.process.signal_processing.butterworth_lowpass``, whose
    default is causal to match the validated viewer.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A cutoff at or above the record's Nyquist frequency cannot be designed; the
    detectors detect that case themselves and report ``lowpass_skipped`` rather
    than filtering.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
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
    """Baseline, spread, peak and activity threshold of one waveform.

    The threshold is ``baseline + max(fraction * peak, sigma * robust_sigma)``.
    The fraction-of-peak term makes the boundary independent of a channel's noise
    floor -- two VEST H-alpha channels with a ten-fold noise difference agree to
    0.2 ms on it -- and the sigma term keeps a weak record from being thresholded
    at its own noise.

    Parameters
    ----------
    values : array_like
        The waveform [any].
    reference_mask : array_like of bool or None
        Samples that define the quiet reference, passed to
        :func:`robust_baseline` [-].
    fraction : float
        Fraction of the peak excess the threshold sits at [-].
    sigma : float
        Number of robust sigmas the threshold sits at [-].
    search_mask : array_like of bool or None, optional
        Samples the detector may look at; the peak that scales the threshold is
        taken inside it too.  ``None`` is the whole record [-].

    Returns
    -------
    baseline : float
        Median of the reference samples [any].
    robust_sigma : float
        Robust sigma of the reference samples [any].
    peak : float
        Largest excess over the baseline inside the search mask, ``nan`` when the
        baseline is not finite or the mask selects nothing [any].
    threshold : float
        The activity threshold, in the units of the record, not of the excess
        [any].

    Defaults
    --------
    None; both terms are the caller's.  The VEST values are a machine-specific
    setting [vest.yaml].

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A negative peak (every sample below the baseline inside the mask) contributes
    nothing: the fraction term is clipped at zero and the threshold falls back to
    the sigma term.  Detectors treat a non-finite baseline or threshold as *no
    evidence* rather than thresholding on it.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    .. [vest.yaml] The VEST values for these rules are policy, not defaults:
       ``vaft/machine_mapping/vest.yaml`` carries them in ``plasma_timing``,
       ``discharge_timing`` and ``plasma_features``, resolved by
       ``vaft.machine_mapping.utils`` and passed in by ``vaft.omas.plasma_timing``,
       ``vaft.omas.discharge_timing`` and ``vaft.omas.plasma_features``.
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


def _run_around(above: np.ndarray, i: int) -> tuple[int, int]:
    """The maximal ``[start, stop)`` stretch of True samples in ``above`` that holds ``i``."""
    start = i
    while start > 0 and above[start - 1]:
        start -= 1
    stop = i + 1
    while stop < above.size and above[stop]:
        stop += 1
    return start, stop


def _rejected_as_dicts(rejected) -> list[dict[str, Any]]:
    return [{"time": t, "reason": why, **feats.as_dict()} for t, why, feats in rejected]


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
    """Width, peak, prominence and integral of one run above a baseline.

    The temporal shape of a candidate run, on which the morphology rules of the
    detectors are stated.  Prominence comes from ``scipy.signal.peak_prominences``
    on the whole excess record, so it measures how far the run rises above its
    surroundings, not above the baseline.

    Parameters
    ----------
    time : array_like
        Time grid of the record; must be as long as ``values`` [s].
    values : array_like
        The waveform [any].
    baseline : float
        Level the excess is measured from [any].
    start, stop : int
        Half-open index bounds of the run, ``values[start:stop]`` [-].

    Returns
    -------
    RunFeatures
        ``start_time`` and ``end_time`` (first and last sample of the run) [s],
        ``width_s`` (``samples * dt``) [s], ``samples`` [-], ``peak`` and
        ``prominence`` (excess over the baseline) [any], ``peak_time`` [s] and
        ``integral`` (excess integrated over the run) [any*s].

    Convention
    ----------
    ``end_time`` is the run's last sample, not the first sample after it, so a
    run of one sample has ``start_time == end_time`` and ``width_s == dt``.  The
    sample spacing is the median of ``diff(time)``, so a record with a few
    irregular samples still gets a representative width.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A run touching either end of the record has no surroundings on that side;
    ``prominence`` then falls back to the run's own peak excess.  Nothing checks
    that ``start`` and ``stop`` bound a run that is actually above the baseline.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    """
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
    """How many runs above a threshold are shorter than a given length.

    The count of spikes a record carries: the detectors report it as
    ``isolated_excursions_before_onset`` so a consumer can see how noisy the
    stretch before an onset was without re-reading the waveform.

    Parameters
    ----------
    values : array_like
        The waveform [any].
    threshold : float
        Level a sample must exceed to be part of a run [any].
    max_run_samples : int
        Runs strictly shorter than this many samples are counted [-].

    Returns
    -------
    int
        Number of such runs [-].

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Purely a count: which runs, and where, is not reported.  The detectors keep
    the runs they rejected, with reasons, in ``OnsetRecord.rejected``.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    return sum(1 for a, b in _runs(y > threshold) if (b - a) < int(max_run_samples))


def pickup_scale(values, baseline: float, robust_sigma: float, dt: float,
                 *, impulse_max_s: float = 2.0e-3, sigma: float = 6.0) -> float:
    """The record's own impulsive-excursion amplitude.

    The largest absolute excess over the baseline among runs above
    ``sigma * robust_sigma`` that last less than ``impulse_max_s`` -- on a
    plasma-current record, the coil-firing pickup measured on that very shot.
    :func:`principal_pulse_onset` compares a candidate pulse against it, which is
    what refuses a record whose maximum *is* the pickup.

    Parameters
    ----------
    values : array_like
        The waveform [any].
    baseline : float
        Level the excess is measured from [any].
    robust_sigma : float
        Robust spread of the reference, as returned by
        :func:`robust_baseline` [any].
    dt : float
        Sample spacing, used to turn ``impulse_max_s`` into samples [s].
    impulse_max_s : float, optional
        Longest run still counted as an impulse [s].
    sigma : float, optional
        Number of robust sigmas a sample must clear to be part of a run [-].

    Returns
    -------
    float
        The largest such excursion, or ``0.0`` when there is none or the
        baseline or spread is not usable [any].

    Defaults
    --------
    ``sigma = 6`` is empirical and deliberately above the detectors' five: white
    noise does not reach six sigma in a record of a few thousand samples, so the
    scale measures pickup rather than the noise tail.  ``impulse_max_s = 2 ms``
    is empirical too, the longest coil-firing transient seen on VEST
    plasma-current records [corpus].

    Convention
    ----------
    Absolute: a pickup transient of either sign counts, because the induced
    excursion's polarity depends on the coil and the winding, not on the plasma.

    Applicability
    -------------
    Machine-independent.  Both defaults were fixed on VEST records.

    Limitations
    -----------
    The scale is measured wherever the caller points it.  Inside a pulse the
    pulse's own noisy threshold crossings would be counted, so
    :func:`principal_pulse_onset` measures it outside the principal run, one
    impulse length clear of both edges.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
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

    The general-purpose detector: it makes no assumption that the record is
    pulse-shaped, so it is the one to use where the first activity matters and a
    later, larger excursion does not.

    Processing steps
    ----------------
    1. Median-filter the record over ``prefilter_samples`` (an optical channel's
       isolated spikes are removed before they can be counted).
    2. Settle the reference stretch and measure baseline and spread on it.
    3. Threshold as :func:`excess_threshold`; refuse the record outright when the
       baseline, spread or peak is degenerate.
    4. Bridge gaps of at most ``bridge_samples`` below the threshold, so a single
       noise sample during a rise does not split the run.
    5. Walk the runs in time order; the first that passes persistence
       (``hold_s``) and morphology (``min_width_s``, ``min_prominence_sigma``,
       ``min_integral_fraction``) is the onset, at its first sample.  Every run
       that fails is kept in ``rejected`` with the reason.

    Parameters
    ----------
    time : array_like
        Time grid of the record [s].
    values : array_like
        The waveform [any].
    fraction : float, optional
        Fraction-of-peak term of the threshold [-].
    sigma : float, optional
        Robust-sigma term of the threshold [-].
    hold_s : float, optional
        How long a run must stay above the threshold [s].
    min_width_s : float, optional
        Shortest accepted run; ``0`` disables the test [s].
    min_prominence_sigma : float, optional
        How far a run must rise above its surroundings, in robust sigmas; ``0``
        disables the test [-].
    min_integral_fraction : float, optional
        Fraction of the record's total positive excess a run must carry; ``0``
        disables the test [-].
    reference_mask : array_like of bool or None, optional
        Samples that define the quiet reference; ``None`` selects the leading
        ``reference_fraction`` of the record [-].
    reference_fraction : float, optional
        Fraction of the record used as the reference when no mask is given [-].
    search_mask : array_like of bool or None, optional
        Samples the detector may look at; the peak that scales the threshold is
        taken inside it too.  ``None`` is the whole record [-].
    prefilter_samples : int, optional
        Median-filter kernel applied first; ``1`` means no filtering [-].
    bridge_samples : int, optional
        Longest dip below the threshold that does not split a run [-].

    Returns
    -------
    OnsetRecord
        ``time`` is the accepted run's first sample of the input grid, or
        ``None`` with ``no_onset`` and a reason [s].

        ``evidence`` carries the baseline, spread, peak, threshold and the rule
        values; ``rejected`` the runs that crossed the threshold and failed,
        each with its reason.

    Defaults
    --------
    The signature values are numerical convenience -- a working detector for a
    caller with no policy -- and are documented as rules, not as numbers to copy.
    The VEST values are policy [vest.yaml]; the reasoning behind their
    magnitudes, and the false-onset counts that fixed them, is in the module
    docstring and in the corpus table [corpus].

    Convention
    ----------
    The onset is the run's *first* sample, so a half-open consumer window
    ``[start, onset)`` excludes exactly the samples from the onset onward.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    No pulse shape is assumed, so an early isolated excursion that passes both
    persistence and morphology *is* the onset.  Where the record is known to be
    pulse-shaped and the principal pulse is what is wanted,
    :func:`principal_pulse_onset` is unreachable by such an excursion by
    construction.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    .. [vest.yaml] The VEST values for these rules are policy, not defaults:
       ``vaft/machine_mapping/vest.yaml`` carries them in ``plasma_timing``,
       ``discharge_timing`` and ``plasma_features``, resolved by
       ``vaft.machine_mapping.utils`` and passed in by ``vaft.omas.plasma_timing``,
       ``vaft.omas.discharge_timing`` and ``vaft.omas.plasma_features``.
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

    An excursion not connected to the maximum can never be chosen, which is what
    makes this the detector for a record known to be pulse-shaped.  A record with
    no pulse has its maximum *at* the pickup, and two floors guard that case.

    Processing steps
    ----------------
    1. Low-pass the record (zero-phase) when a ``cutoff_hz`` is given and the
       grid can carry it; otherwise flag ``lowpass_skipped``.
    2. Settle the reference, measure baseline and spread, threshold as
       :func:`excess_threshold`.
    3. Refuse when the peak does not clear ``sigma`` robust sigmas
       (``peak_below_noise``).
    4. Walk back and forward from the maximum inside ``search_mask`` while the
       excess stays above the threshold; that connected run is the pulse.
    5. Refuse a run narrower than ``impulse_max_s`` (``principal_run_impulsive``)
       and a peak below ``pickup_floor`` times :func:`pickup_scale` measured
       outside the run (``peak_below_pickup_floor``); otherwise the onset is the
       run's first sample.

    Parameters
    ----------
    time : array_like
        Time grid of the record [s].
    values : array_like
        The waveform [any].
    fraction : float, optional
        Fraction-of-peak term of the threshold [-].
    sigma : float, optional
        Robust-sigma term of the threshold [-].
    reference_mask : array_like of bool or None, optional
        Samples that define the quiet reference; ``None`` selects the leading
        ``reference_fraction`` of the record [-].
    reference_fraction : float, optional
        Fraction of the record used as the reference when no mask is given [-].
    search_mask : array_like of bool or None, optional
        Samples the detector may look at; the peak that scales the threshold is
        taken inside it too.  ``None`` is the whole record [-].
    cutoff_hz : float or None, optional
        Zero-phase low-pass cutoff; ``None`` leaves the record unfiltered [Hz].
    fs : float or None, optional
        Sample rate; inferred from ``time`` when ``None`` [Hz].
    order : int, optional
        Butterworth order of that low-pass [-].
    pickup_floor : float, optional
        How many times the record's own impulsive scale the pulse must exceed;
        ``0`` disables the test [-].
    impulse_max_s : float, optional
        Longest run still treated as an impulse, both for the principal-run test
        and for :func:`pickup_scale` [s].
    bridge_samples : int, optional
        Longest dip below the threshold that does not split the run [-].

    Returns
    -------
    OnsetRecord
        ``time`` is the first sample of the principal run, or ``None`` with
        ``no_onset`` and one of ``peak_below_noise``, ``principal_run_impulsive``,
        ``peak_below_pickup_floor``, ``search_mask_empty`` or
        ``record_too_short`` [s].

        ``evidence`` carries the threshold terms, the peak's time and index and
        the measured ``pickup_scale``.

    Defaults
    --------
    Numerical convenience, as in :func:`sustained_excess_onset`; ``pickup_floor``
    and ``impulse_max_s`` are empirical, fixed on VEST plasma-current records
    [corpus].  The VEST values are policy [vest.yaml].

    Convention
    ----------
    The onset is the run's first sample.  The low-pass is zero-phase, so the
    onset is not shifted by the filtering.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Only for a record whose activity is one pulse: on a record with two comparable
    pulses the answer is the onset of whichever holds the maximum, with no flag
    saying the other exists.  :func:`active_window` reports that case as
    ``multiple_segments``.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    .. [vest.yaml] The VEST values for these rules are policy, not defaults:
       ``vaft/machine_mapping/vest.yaml`` carries them in ``plasma_timing``,
       ``discharge_timing`` and ``plasma_features``, resolved by
       ``vaft.machine_mapping.utils`` and passed in by ``vaft.omas.plasma_timing``,
       ``vaft.omas.discharge_timing`` and ``vaft.omas.plasma_features``.
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
    start, stop = _run_around(above, i_peak)
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

    The detector behind every plasma window in VAFT: what the onset detectors do
    for the start, plus the harder half, which is where a pulse *ends*.

    Processing steps
    ----------------
    1. Median-filter over ``prefilter_samples``, low-pass (zero-phase) when a
       ``cutoff_hz`` is given and the grid can carry it.
    2. Threshold as :func:`excess_threshold` against the leading reference.
    3. Bridge dips shorter than ``gap_s``; keep the runs that pass persistence
       (``hold_s``) and morphology as segments; merge a segment that begins
       within ``post_quiet_s`` of the previous one's end into it, so a brief
       quiet moment does not end the window while a real gap does.
    4. With ``principal_only``, keep only the segment holding the global maximum,
       guarded by the impulsive-run and ``pickup_floor`` rules of
       :func:`principal_pulse_onset`; otherwise the window is the envelope of
       every segment and ``multiple_segments`` says when there is more than one.
    5. Judge the offset against a *trailing* reference -- the last
       ``trailing_fraction`` of the record, used only when it is quiet (spread
       within three times the leading one) and sits below
       ``trailing_max_fraction`` of the peak -- at ``end_fraction`` of the peak
       above it.  Where that end threshold is never crossed and
       ``collapse_fallback`` is on, end the window at the end of the *last* steep
       fall after the peak (``collapse_rate_fraction``, ``collapse_min_drop``)
       and flag ``offset_from_collapse``.

    The baseline after a pulse need not be the one before it: a VEST
    plasma-current record settles a few percent of its peak above zero once the
    plasma is gone, and after a termination the Rogowski keeps reading the induced
    vessel current -- a few to ten percent of the peak, decaying over tens of
    milliseconds -- which no level between it and the trailing baseline separates
    from plasma.  The collapse does.  The last fall, not the steepest: a plasma
    survives a mid-pulse drop and terminates later.

    Parameters
    ----------
    time : array_like
        Time grid of the record [s].
    values : array_like
        The waveform [any].
    fraction : float, optional
        Fraction-of-peak term of the onset threshold [-].
    sigma : float, optional
        Robust-sigma term of the onset threshold [-].
    hold_s : float, optional
        How long a run must stay above the threshold to be a segment [s].
    min_width_s : float, optional
        Shortest accepted segment; ``0`` disables the test [s].
    min_prominence_sigma : float, optional
        How far a segment must rise above its surroundings, in robust sigmas;
        ``0`` disables the test [-].
    min_integral_fraction : float, optional
        Fraction of the record's total positive excess a segment must carry;
        ``0`` disables the test [-].
    gap_s : float, optional
        Longest dip below the threshold that does not split a segment [s].
    post_quiet_s : float, optional
        A segment beginning within this of the previous one's end is merged into
        it [s].
    principal_only : bool, optional
        Keep only the segment holding the global maximum [-].
    pickup_floor : float, optional
        Principal-segment floor relative to :func:`pickup_scale`; ``0`` disables
        the test [-].
    impulse_max_s : float, optional
        Longest run still treated as an impulse [s].
    reference_mask : array_like of bool or None, optional
        Samples that define the quiet reference; ``None`` selects the leading
        ``reference_fraction`` of the record [-].
    reference_fraction : float, optional
        Fraction of the record used as the reference when no mask is given [-].
    trailing_fraction : float, optional
        Fraction of the record at its end used as the trailing reference [-].
    trailing_max_fraction : float, optional
        How far above the leading baseline that trailing level may sit, as a
        fraction of the peak, for it to be used at all [-].
    end_fraction : float or None, optional
        Fraction of the peak the signal must fall below, above the trailing
        level, for the window to end; ``fraction`` when ``None`` [-].
    collapse_fallback : bool, optional
        Whether to end a window that never falls below the end threshold at the
        last steep fall [-].
    collapse_rate_fraction : float, optional
        How steep that fall must be, as a fraction of the steepest fall in the
        record [-].
    collapse_min_drop : float, optional
        Fraction of the level present before the fall that it must remove [-].
    search_mask : array_like of bool or None, optional
        Samples the detector may look at; the peak that scales the threshold is
        taken inside it too.  ``None`` is the whole record [-].
    prefilter_samples : int, optional
        Median-filter kernel applied first; ``1`` means no filtering [-].
    cutoff_hz : float or None, optional
        Zero-phase low-pass cutoff; ``None`` leaves the record unfiltered [Hz].
    fs : float or None, optional
        Sample rate; inferred from ``time`` when ``None`` [Hz].
    order : int, optional
        Butterworth order of that low-pass [-].

    Returns
    -------
    PulseWindow
        ``onset`` and ``offset`` records, both on samples of the input grid, the
        accepted ``segments`` in time order, and the flags that say what the
        window is not: ``offset_at_record_end`` (still active at the last
        sample), ``onset_at_record_start``, ``multiple_segments``,
        ``offset_from_collapse``, ``trailing_segment_dropped``,
        ``offset_threshold_above_peak`` [s].

    Defaults
    --------
    The signature values are numerical convenience; ``trailing_fraction``,
    ``trailing_max_fraction``, ``collapse_rate_fraction`` and
    ``collapse_min_drop`` are empirical, fixed on the hand-reviewed VEST
    discharges and then checked against the whole corpus [corpus].  The VEST
    values a pipeline actually runs with are policy [vest.yaml].

    Convention
    ----------
    ``offset.time`` is the *last* sample of the pulse -- the last sample above the
    end threshold, or, with ``offset_from_collapse``, the last sample of the
    quench that ended it -- so a half-open consumer window is
    ``[onset.time, time[offset.index + 1])``.  A window is never assumed: nothing
    here reports the record bounds when it found no pulse.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The end rule is the weak one.  A disruption's vessel-current tail can sit at a
    third of the peak for a hundred milliseconds, above any end threshold that
    still separates plasma from a normal termination, and is then ended by the
    collapse fallback rather than by the threshold.  Thirteen shots of the corpus
    reach the ``offset_threshold_above_peak`` guard, where the trailing level and
    the peak leave no usable end threshold; they are flagged, not repaired,
    tracked in issue #409.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    .. [vest.yaml] The VEST values for these rules are policy, not defaults:
       ``vaft/machine_mapping/vest.yaml`` carries them in ``plasma_timing``,
       ``discharge_timing`` and ``plasma_features``, resolved by
       ``vaft.machine_mapping.utils`` and passed in by ``vaft.omas.plasma_timing``,
       ``vaft.omas.discharge_timing`` and ``vaft.omas.plasma_features``.
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
    # The trailing form of the end threshold must lie below the pulse itself:
    # a quiet-looking tail whose noise band reaches the peak (a faint pulse
    # before a noisier stretch, 33 corpus shots) cannot bound the offset, and
    # the leading baseline judges it instead.
    end_threshold = None
    if trail_quiet:
        trailing = trail_baseline + max(end_frac * peak, float(sigma) * trail_spread)
        if trailing < baseline + peak:
            end_threshold = trailing
    if end_threshold is None and end_frac > float(fraction):
        end_threshold = baseline + max(end_frac * peak, float(sigma) * spread)
    if end_threshold is not None:
        above_end = _bridged(y > end_threshold, gap)
        # A trailing segment whose own peak never reaches the end threshold is
        # a re-emergence the end rule does not count as the pulse: drop it and
        # judge the one before.
        while len(segments) > 1 and float(y[segments[-1][0]:segments[-1][1]].max()) <= end_threshold:
            segments.pop()
            flags.append("trailing_segment_dropped")
        last = segments[-1][1]
        seg0, seg1 = segments[-1]
        i_peak_last = seg0 + int(np.argmax(y[seg0:seg1]))
        stop = _extend_forward(above_end, i_peak_last, quiet)
        if stop <= i_peak_last:
            # Unreachable with the threshold below the peak; the guard against a
            # segment ending before its own peak (an empty run) stays.
            flags.append("offset_threshold_above_peak")
        elif stop < y.size:
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

    One :func:`active_window` call with ``principal_only=True`` and the principal
    defaults, for callers that want the pair of records rather than the window
    object.  Every other keyword argument is forwarded to :func:`active_window`
    unchanged.

    Parameters
    ----------
    time : array_like
        Time grid of the record [s].
    values : array_like
        The waveform [any].

    Returns
    -------
    onset : OnsetRecord
        The principal run's first sample -- exactly what
        :func:`principal_pulse_onset` returns -- or ``None`` with the window's
        flags [s].
    offset : OnsetRecord
        The principal run's last sample, carrying the window's flags, so
        ``offset_at_record_end`` and ``offset_from_collapse`` reach a caller that
        never sees the :class:`PulseWindow` [s].

    Defaults
    --------
    ``pickup_floor = 3`` and ``hold_s = 0`` are set here unless the caller
    overrides them: empirical, and the pair that makes this agree with
    :func:`principal_pulse_onset` -- the pulse is the run holding the maximum, so
    persistence is redundant and the pickup floor does the refusing.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Both records carry the *window's* evidence rather than an onset detector's, so
    ``evidence`` is keyed as :func:`active_window` keys it.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
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


# ---------------------------------------------------------------------------
# Zero crossing after an anchored excursion
# ---------------------------------------------------------------------------


def zero_crossing_after_excursion(
    time,
    values,
    *,
    anchor_time: float,
    fraction: float = 0.05,
    sigma: float = 5.0,
    hold_s: float = 5.0e-4,
    anchor_tolerance_s: float = 5.0e-4,
    approach_fraction: float = 0.10,
    approach_hysteresis: float = 2.0,
    reference_mask=None,
    reference_fraction: float = 0.2,
    search_mask=None,
    prefilter_samples: int = 1,
    bridge_samples: int = 2,
) -> OnsetRecord:
    """First sample after an anchored excursion where the record changes sign.

    Written for a drive waveform: the loop voltage crosses zero once the ohmic
    swing that drove it is over, and *that* crossing -- not the record's loudest
    moment -- is the event.  Anchoring is what makes a later, larger pulse
    irrelevant: the excursion is judged where the drive starts.

    Processing steps
    ----------------
    1. Median-filter over ``prefilter_samples``, settle the reference, measure
       baseline and spread.
    2. Threshold ``|values - baseline|`` at
       ``max(fraction * peak, sigma * robust_sigma)``, the peak taken inside
       ``search_mask``.
    3. Take the first sustained run (``hold_s``) that starts within
       ``anchor_tolerance_s`` of ``anchor_time`` or contains the anchor sample;
       with none, return ``no_excursion_at_anchor``.
    4. From that run's extremum, walk forward to the first sample whose
       baseline-relative sign is opposite to the extremum's -- strictly, so a
       sample exactly at the baseline is not a crossing.
    5. Watch the decay in between: when the deviation drops below
       ``approach_fraction`` of the extremum and climbs back above
       ``approach_hysteresis`` times that level before crossing, flag
       ``approached_without_crossing`` and record the minimum and its time.

    Parameters
    ----------
    time : array_like
        Time grid of the record [s].
    values : array_like
        The waveform [any].
    anchor_time : float
        Where the excursion is expected to start, usually another detector's
        onset [s].
    fraction : float, optional
        Fraction-of-peak term of the excursion threshold [-].
    sigma : float, optional
        Robust-sigma term of the excursion threshold [-].
    hold_s : float, optional
        How long the excursion must stay above that threshold [s].
    anchor_tolerance_s : float, optional
        How far the run's start may lie from the anchor when it does not contain
        it [s].
    approach_fraction : float, optional
        Fraction of the extremum the decay must reach to count as an approach
        [-].
    approach_hysteresis : float, optional
        Multiple of that level the record must climb back above for the approach
        to be reported as one that did not cross; the hysteresis keeps noise
        around the level from counting as a re-rise [-].
    reference_mask : array_like of bool or None, optional
        Samples that define the quiet reference; ``None`` selects the leading
        ``reference_fraction`` of the record [-].
    reference_fraction : float, optional
        Fraction of the record used as the reference when no mask is given [-].
    search_mask : array_like of bool or None, optional
        Samples the detector may look at; the peak that scales the threshold is
        taken inside it too.  ``None`` is the whole record [-].
    prefilter_samples : int, optional
        Median-filter kernel applied first; ``1`` means no filtering [-].
    bridge_samples : int, optional
        Longest dip below the threshold that does not split the excursion [-].

    Returns
    -------
    OnsetRecord
        ``time`` is the first sample past the crossing, or ``None`` with
        ``no_excursion_at_anchor`` or ``no_zero_crossing`` [s].

        ``evidence`` carries the signed extremum (baseline-relative), its time,
        the run bounds, the run's start relative to the anchor, and the approach
        minimum when there was one.

    Defaults
    --------
    The signature values are numerical convenience; the VEST loop-voltage values -- the
    fraction, the hold, the anchor tolerance and both approach numbers -- are
    policy in the ``discharge_timing.vloop`` block [vest.yaml].

    Convention
    ----------
    The sign is measured relative to the *baseline*, not to zero, so a record
    with an offset crosses where it returns through its own quiet level.  The
    crossing must be strict: a sample exactly at the baseline is not one.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    One crossing only, the first after the anchored excursion.  A record that
    decays towards the baseline and turns back without reaching it is reported as
    ``approached_without_crossing`` *and* ``no_zero_crossing`` -- the flag is
    evidence for a consumer, not an event.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    .. [vest.yaml] The VEST values for these rules are policy, not defaults:
       ``vaft/machine_mapping/vest.yaml`` carries them in ``plasma_timing``,
       ``discharge_timing`` and ``plasma_features``, resolved by
       ``vaft.machine_mapping.utils`` and passed in by ``vaft.omas.plasma_timing``,
       ``vaft.omas.discharge_timing`` and ``vaft.omas.plasma_features``.
    """
    t, raw = _as_arrays(time, values)
    y = median_smooth(raw, prefilter_samples) if prefilter_samples > 1 else _fill_non_finite(raw)
    ref, ref_flags = _settle_reference(y, _reference(t, reference_mask, reference_fraction), sigma)
    baseline, spread = robust_baseline(y, ref)
    method = "excursion_zero_crossing"
    search = (np.ones(t.size, dtype=bool) if search_mask is None
              else np.asarray(search_mask, dtype=bool).reshape(-1))
    excess = np.abs(y - baseline) if np.isfinite(baseline) else np.full(t.size, np.nan)
    region = excess[search]
    region = region[np.isfinite(region)]
    peak = float(region.max()) if region.size else float("nan")
    spread_term = float(sigma) * spread if np.isfinite(spread) else 0.0
    threshold = max(float(fraction) * max(peak, 0.0), spread_term) if np.isfinite(peak) else float("nan")
    degenerate = _degenerate(t, y, method, baseline, spread, peak, threshold, ref, sigma)
    if degenerate is not None:
        return degenerate
    dt = float(np.median(np.diff(t)))
    hold = max(1, int(round(float(hold_s) / dt)))
    above = _bridged(excess > threshold, int(bridge_samples)) & search
    anchor = float(anchor_time)
    tol = float(anchor_tolerance_s)
    evidence: dict[str, Any] = {
        "baseline_median": baseline, "robust_sigma": spread, "peak": peak, "threshold": threshold,
        "fraction": float(fraction), "sigma": float(sigma), "hold_samples": hold,
        "anchor_time": anchor, "anchor_tolerance_s": tol, "approach_fraction": float(approach_fraction),
        "approach_hysteresis": float(approach_hysteresis),
        "prefilter_samples": int(prefilter_samples), "bridge_samples": int(bridge_samples),
    }
    rejected: list[tuple[float, str, RunFeatures]] = []
    excursion = None
    for start, stop in _runs(above):
        t_start = float(t[start])
        contains_anchor = t_start <= anchor <= float(t[stop - 1])
        if not contains_anchor and abs(t_start - anchor) > tol:
            why = "not_at_anchor"
        elif stop - start < hold:
            why = "persistence"
        else:
            excursion = (start, stop)
            break
        if len(rejected) < MAX_REJECTED_RUNS:
            rejected.append((t_start, why, _brief_run(t, excess, 0.0, start, stop)))
    evidence["n_rejected"] = len(rejected)
    if excursion is None:
        return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                           flags=("no_onset", "no_excursion_at_anchor", *ref_flags),
                           rejected=tuple(rejected))
    start, stop = excursion
    i_ext = start + int(np.argmax(excess[start:stop]))
    extremum = float(y[i_ext] - baseline)
    sign = 1.0 if extremum > 0 else -1.0
    evidence.update({
        "run_start": float(t[start]), "run_end": float(t[stop - 1]),
        "run_start_minus_anchor": float(t[start]) - anchor,
        "extremum": extremum, "extremum_time": float(t[i_ext]), "extremum_index": int(i_ext),
    })
    accepted = run_features(t, excess, 0.0, start, stop)
    deviation = (y - baseline) * sign  # positive along the excursion
    later = np.flatnonzero(deviation[i_ext + 1:] < 0.0)
    k = int(i_ext + 1 + later[0]) if later.size else None
    flags: list[str] = list(ref_flags)
    decay = deviation[i_ext:(k if k is not None else t.size)]
    level = float(approach_fraction) * abs(extremum)
    below = np.flatnonzero(decay < level)
    evidence["approach_min"] = None
    evidence["approach_time"] = None
    if below.size:
        first = int(below[0])
        rise = np.flatnonzero(decay[first:] >= float(approach_hysteresis) * level)
        if rise.size:
            segment = decay[first:first + int(rise[0])]
            i_min = first + int(np.argmin(segment))
            evidence["approach_min"] = float(sign * decay[i_min])
            evidence["approach_time"] = float(t[i_ext + i_min])
            flags.append("approached_without_crossing")
    if k is None:
        return OnsetRecord(time=None, index=None, method=method, evidence=evidence,
                           flags=("no_onset", "no_zero_crossing", *flags),
                           rejected=tuple(rejected), accepted=accepted)
    return OnsetRecord(time=float(t[k]), index=k, method=method, evidence=evidence,
                       flags=tuple(flags), rejected=tuple(rejected), accepted=accepted)


# ---------------------------------------------------------------------------
# Representative peak
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeakRecord:
    """What :func:`robust_peak` concluded about one waveform.

    ``value`` is the *smoothed* sample at the peak (the raw one is in the
    evidence -- that is what makes the peak spike-resistant), ``time`` a
    sample of the input grid or ``None`` when nothing qualifies, ``excess``
    the value relative to the baseline median.  ``rejected`` holds the
    louder candidates that were refused as impulsive, each with its run
    features, so a consumer can see what the record's loudest sample was.
    """

    value: float | None
    time: float | None
    index: int | None
    excess: float | None
    method: str = "robust_peak"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    rejected: tuple[tuple[float, str, RunFeatures], ...] = ()
    accepted: RunFeatures | None = None

    @property
    def found(self) -> bool:
        return self.time is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "time": self.time,
            "index": self.index,
            "excess": self.excess,
            "method": self.method,
            "evidence": dict(self.evidence),
            "flags": list(self.flags),
            "accepted": None if self.accepted is None else self.accepted.as_dict(),
            "rejected": _rejected_as_dicts(self.rejected),
        }


PEAK_POLARITIES = ("positive", "negative", "absolute")


def _no_peak(reason: str, evidence: Mapping[str, Any], *extra: str,
             rejected=(), accepted=None) -> PeakRecord:
    return PeakRecord(value=None, time=None, index=None, excess=None, evidence=dict(evidence),
                     flags=tuple(dict.fromkeys(("no_peak", reason, *extra))),
                     rejected=tuple(rejected), accepted=accepted)


def robust_peak(
    time,
    values,
    *,
    polarity: str = "positive",
    prefilter_samples: int = 5,
    cutoff_hz: float | None = None,
    fs: float | None = None,
    order: int = 4,
    sigma: float = 5.0,
    level_fraction: float = 0.5,
    min_width_s: float = 1.0e-3,
    max_candidates: int = 8,
    edge_samples: int = 1,
    impulse_max_s: float = 2.0e-3,
    reference_mask=None,
    reference_fraction: float = 0.2,
    search_mask=None,
    bridge_samples: int = 2,
) -> PeakRecord:
    """The representative peak of a record: its largest sustained excursion, not its loudest sample.

    The measurement behind every peak value VAFT reports -- the plasma current, a
    line's brightness, the diamagnetic flux -- for records where the loudest
    sample is routinely a coil-firing impulse or a digitizer glitch.

    Processing steps
    ----------------
    1. Median-filter over ``prefilter_samples`` and, when a ``cutoff_hz`` is
       given and the grid can carry it, zero-phase low-pass.
    2. Settle the reference and take the baseline as its median.
    3. Orient the record by ``polarity``: ``positive`` scores the excess over the
       baseline, ``negative`` the deficit, ``absolute`` the magnitude.
    4. Take the largest score inside ``search_mask`` as a candidate; the run
       holding it at ``level_fraction`` of its height must last ``min_width_s``.
       A narrower run is a spike: it is refused (``rejected``, reason
       ``impulsive``), masked whole -- shoulders included, bounded by
       ``impulse_max_s`` and by the baseline crossing, so it cannot come back --
       and the next candidate is tried, up to ``max_candidates`` times.
    5. Report the accepted run's extreme sample, with the raw record's own
       maximum kept in the evidence.

    Parameters
    ----------
    time : array_like
        Time grid of the record [s].
    values : array_like
        The waveform [any].
    polarity : {'positive', 'negative', 'absolute'}, optional
        What counts as up [-].
    prefilter_samples : int, optional
        Median-filter kernel applied first; ``1`` means no filtering [-].
    cutoff_hz : float or None, optional
        Zero-phase low-pass cutoff; ``None`` leaves the record unfiltered [Hz].
    fs : float or None, optional
        Sample rate; inferred from ``time`` when ``None`` [Hz].
    order : int, optional
        Butterworth order of that low-pass [-].
    sigma : float, optional
        How many robust sigmas the excursion must clear [-].
    level_fraction : float, optional
        Height at which the run holding a candidate is measured, as a fraction of
        the candidate's own excursion [-].
    min_width_s : float, optional
        How long that run must last for the candidate to be a peak rather than a
        spike [s].
    max_candidates : int, optional
        How many candidates may be refused before the record is given up on;
        must be at least 1 [-].
    edge_samples : int, optional
        How close to the edge of the search stretch a peak may lie before
        ``peak_at_window_edge`` is flagged [-].
    impulse_max_s : float, optional
        Longest run treated as an impulse, for both the refusal mask and the
        reported :func:`pickup_scale` [s].
    reference_mask : array_like of bool or None, optional
        Samples that define the quiet reference; ``None`` selects the leading
        ``reference_fraction`` of the record [-].
    reference_fraction : float, optional
        Fraction of the record used as the reference when no mask is given [-].
    search_mask : array_like of bool or None, optional
        Samples the detector may look at; the peak that scales the threshold is
        taken inside it too.  ``None`` is the whole record [-].
    bridge_samples : int, optional
        Longest dip below a candidate's level that does not split its run [-].

    Returns
    -------
    PeakRecord
        ``value`` is the smoothed record's extreme sample in the accepted run,
        and ``time`` the sample of the input grid it sits on -- or ``None`` with
        ``no_peak`` and one reason [any].

        ``evidence`` carries the baseline, the robust sigma, the rule values,
        ``raw_value``, ``raw_max`` and ``raw_max_time``, the measured
        ``pickup_scale`` and the number of refused candidates.

    Raises
    ------
    ValueError
        When ``polarity`` is not one of the three, or ``max_candidates`` is below
        one.

    Defaults
    --------
    The signature values are numerical convenience.  The VEST values are policy
    in the ``plasma_features`` block [vest.yaml], per signal: the plasma current,
    H-alpha, each impurity line and the diamagnetic flux carry their own
    ``prefilter``, ``sigma``, ``level_fraction`` and ``min_width_s``, and an
    H-alpha value is never reused for an impurity line.

    Convention
    ----------
    ``polarity`` decides the sign of the answer: with ``negative`` or
    ``absolute`` the reported ``value`` is still the record's own signed sample,
    not the score, so a VEST diamagnetic flux -- stored negative-going -- comes
    back negative.  ``value`` is the *smoothed* sample; the raw one is in the
    evidence.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Flags on a found peak say when the answer is not the loudest sample:
    ``peak_is_spike`` (the raw maximum lay in a refused run),
    ``raw_max_outside_run`` (it lies outside the accepted run for any other
    reason), ``peak_at_window_edge`` (the excursion may continue past the search
    stretch), ``peak_plateau`` (the raw record holds its extreme value for three
    samples or more -- a railed digitizer), ``reference_flat`` (the reference has
    no spread; a peak is still reported, because an integrated record can have an
    exactly zero lead).  A record whose every excursion is impulsive returns
    ``all_candidates_impulsive`` rather than a value.

    Provenance
    ----------
    .. [409] Issue #409, which introduced these primitives and fixed their rules
       against the VEST raw database.
    .. [corpus] ``workflow/plasma_onset/scan_corpus.py`` and the table it writes,
       ``test/data/onset_corpus.json``: the scan of VEST shots 39900-41700 the
       thresholds, hold times and widths were judged on.
    .. [vest.yaml] The VEST values for these rules are policy, not defaults:
       ``vaft/machine_mapping/vest.yaml`` carries them in ``plasma_timing``,
       ``discharge_timing`` and ``plasma_features``, resolved by
       ``vaft.machine_mapping.utils`` and passed in by ``vaft.omas.plasma_timing``,
       ``vaft.omas.discharge_timing`` and ``vaft.omas.plasma_features``.
    """
    if polarity not in PEAK_POLARITIES:
        raise ValueError(f"polarity must be one of {PEAK_POLARITIES}, not {polarity!r}")
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be at least 1")
    t, raw = _as_arrays(time, values)
    skipped = _lowpass_skipped(t, cutoff_hz, fs)
    effective_cutoff = None if skipped else cutoff_hz
    base_flags: tuple[str, ...] = ("lowpass_skipped",) if skipped else ()
    evidence: dict[str, Any] = {
        "polarity": polarity, "prefilter_samples": int(prefilter_samples), "cutoff_hz": cutoff_hz,
        "sigma": float(sigma), "level_fraction": float(level_fraction),
        "min_width_s": float(min_width_s), "edge_samples": int(edge_samples),
    }
    if _too_short(t, effective_cutoff, order):
        return _no_peak("record_too_short", evidence, *base_flags)
    dt = float(np.median(np.diff(t)))
    y = median_smooth(raw, prefilter_samples) if prefilter_samples > 1 else _fill_non_finite(raw)
    y = _lowpassed(y, effective_cutoff, fs, dt, order)
    ref, ref_flags = _settle_reference(y, _reference(t, reference_mask, reference_fraction), sigma)
    baseline, spread = robust_baseline(y, ref)
    evidence.update({"baseline_median": baseline, "robust_sigma": spread})
    if not np.isfinite(baseline) or not np.isfinite(spread):
        return _no_peak("reference_not_finite", evidence, *base_flags, *ref_flags)
    flags: list[str] = [*base_flags, *ref_flags]
    if spread <= 0.0:
        flags.append("reference_flat")
    deviation = y - baseline

    def oriented(dev: np.ndarray) -> np.ndarray:
        if polarity == "negative":
            return -dev
        if polarity == "absolute":
            return np.abs(dev)
        return dev

    score = oriented(deviation)
    if search_mask is not None:
        sel = np.asarray(search_mask, dtype=bool).reshape(-1)
        if sel.size != t.size:
            raise ValueError(f"search_mask has {sel.size} samples but the record has {t.size}")
        if not sel.any():
            return _no_peak("search_mask_empty", evidence, *flags)
    else:
        sel = np.ones(t.size, dtype=bool)
    score = np.where(sel, score, -np.inf)
    raw_score = np.where(sel, oriented(raw - baseline), -np.inf)
    raw_score = np.where(np.isfinite(raw_score), raw_score, -np.inf)
    i_raw = int(np.argmax(raw_score))
    if not np.isfinite(raw_score[i_raw]):
        return _no_peak("no_finite_samples", evidence, *flags)
    # the raw extreme on the chosen side: the loudest raw sample, whatever the polarity
    evidence.update({"raw_max": float(raw[i_raw]), "raw_max_time": float(t[i_raw])})
    peak_score = float(score.max())
    if peak_score <= 0.0:
        return _no_peak("record_flat", evidence, *flags)
    if spread > 0.0 and peak_score < float(sigma) * spread:
        return _no_peak("peak_below_noise", evidence, *flags)
    rejected: list[tuple[float, str, RunFeatures]] = []
    remaining = score.copy()
    selected = np.flatnonzero(sel)
    edge_first, edge_last = int(selected[0]), int(selected[-1])
    for _ in range(int(max_candidates)):
        i = int(np.argmax(remaining))
        if not np.isfinite(remaining[i]) or remaining[i] <= 0.0:
            break
        # the run is judged on what is still in play, so a refused excursion's
        # shoulders cannot rejoin a later candidate's run
        above = _bridged(remaining > float(level_fraction) * remaining[i], int(bridge_samples)) & sel
        start, stop = _run_around(above, i)
        feats = run_features(t, remaining, 0.0, start, stop)
        if feats.width_s < float(min_width_s):
            if len(rejected) < MAX_REJECTED_RUNS:
                rejected.append((float(t[i]), "impulsive", feats))
            # refuse the whole impulse: its run, widened by one impulse length on
            # each side (an impulse is that brief by definition) but never past
            # the baseline crossing -- so its shoulders cannot rejoin a later
            # candidate, and a genuine excursion beyond it is untouched
            reach = max(1, int(round(float(impulse_max_s) / dt)))
            base_start, base_stop = _run_around(remaining > 0.0, i)
            remaining[max(base_start, start - reach):min(base_stop, stop + reach)] = -np.inf
            continue
        margin = max(1, int(round(float(impulse_max_s) / dt)))
        outside = np.r_[y[: max(0, start - margin)], y[min(y.size, stop + margin):]]
        evidence.update({
            "raw_value": float(raw[i]),
            "pickup_scale": pickup_scale(outside, baseline, spread, dt, impulse_max_s=impulse_max_s),
            "n_rejected": len(rejected),
            "run_start": float(t[start]), "run_end": float(t[stop - 1]),
        })
        # A rail holds the raw extreme for consecutive samples; judged on the
        # raw record inside the run, since the median filter repeats values too.
        run_raw = raw_score[start:stop]
        j = int(np.argmax(run_raw))
        plateau = 1
        while j - plateau >= 0 and run_raw[j - plateau] == run_raw[j]:
            plateau += 1
        after = 1
        while j + after < run_raw.size and run_raw[j + after] == run_raw[j]:
            after += 1
        evidence["plateau_samples"] = plateau + after - 1
        if rejected and any(r_start <= float(t[i_raw]) <= r_end
                            for r_start, r_end in ((f.start_time, f.end_time) for _, _, f in rejected)):
            flags.append("peak_is_spike")
        if not (start <= i_raw < stop):
            flags.append("raw_max_outside_run")
        if i - edge_first < int(edge_samples) or edge_last - i < int(edge_samples):
            flags.append("peak_at_window_edge")
        if evidence["plateau_samples"] >= 3:
            flags.append("peak_plateau")
        return PeakRecord(value=float(y[i]), time=float(t[i]), index=i, excess=float(deviation[i]),
                         evidence=evidence, flags=tuple(dict.fromkeys(flags)),
                         rejected=tuple(rejected), accepted=feats)
    evidence["n_rejected"] = len(rejected)
    return _no_peak("all_candidates_impulsive", evidence, *flags, rejected=rejected)
