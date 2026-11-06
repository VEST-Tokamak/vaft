"""Declared acquisition settings checked against the acquired timebase.

A configured sample rate is a claim about the digitizer, not a measurement.
Filters, Nyquist limits, resampling and index windows are all designed from
the declared rate, so this module compares it with what the record's own
timestamps say and reports one of four outcomes:

``consistent``
    the observed rate agrees with the declared one within ``rtol`` and the
    sampling is regular;
``deviation``
    the record is usable but differs from the declaration -- a different
    rate, or irregular sampling (gaps, jitter) -- and the processing is
    designed from the declared rate regardless;
``invalid``
    the declaration or the record cannot support the processing: a
    non-positive or non-finite declared rate, a timebase that is not strictly
    increasing, or a filter cutoff at or above the Nyquist frequency of the
    *observed* rate;
``not_verifiable``
    too few finite timestamps to measure a rate.

The check never replaces the declared rate with the observed one: it reports.
Only ``fatal`` -- a cutoff the acquired data cannot carry -- is meant to stop
a transformation; everything else is recorded and the processing proceeds
unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "ACQUISITION_STATUSES",
    "AcquisitionInconsistencyError",
    "SampleRateCheck",
    "check_declared_sample_rate",
    "summarize_sample_rate_checks",
]

ACQUISITION_STATUSES = ("consistent", "deviation", "invalid", "not_verifiable")


@dataclass(frozen=True)
class SampleRateCheck:
    """Outcome of :func:`check_declared_sample_rate`.

    ``observed_hz`` is ``1 / median(diff(time))``; ``relative_deviation`` is
    ``|observed - declared| / declared``; ``irregularity`` is the largest
    ``|diff - median| / median`` over the record. ``fatal`` is true only
    when a cutoff is at or above the observed Nyquist frequency.
    """

    status: str
    declared_hz: float | None
    observed_hz: float | None
    relative_deviation: float | None
    irregularity: float | None
    samples: int
    cutoff_hz: float | None
    fatal: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AcquisitionInconsistencyError(ValueError):
    """A filter was asked for a cutoff the acquired record cannot carry.

    Raised by the callers of :func:`check_declared_sample_rate` when its result
    is ``fatal``. Stage builders treat it like an unavailable signal: the
    affected component is left out with the reason, its siblings are kept.
    """


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def check_declared_sample_rate(
    time: Any,
    declared: Any,
    *,
    rtol: float = 1.0e-3,
    irregular_rtol: float = 1.0e-2,
    cutoff_hz: Any = None,
) -> SampleRateCheck:
    """Compare a declared sample rate with the timebase of an acquired record.

    Parameters
    ----------
    time : array_like
        The record's timestamps [s].
    declared : float
        The configured sample rate the processing is designed from [Hz].
    rtol : float, optional
        Relative tolerance on the rate for ``consistent`` [-].
    irregular_rtol : float, optional
        Largest relative departure of one sample interval from the median
        before the sampling counts as irregular [-].
    cutoff_hz : float or sequence of float, optional
        Filter cutoff(s) designed on ``declared``; one at or above the
        observed Nyquist frequency makes the outcome ``invalid`` and ``fatal``
        [Hz].

    Returns
    -------
    SampleRateCheck
        Status, declared and observed rates, relative deviation, irregularity
        and whether the cutoff is fatal [-].

    Processing steps
    ----------------
    1. Keep the finite timestamps; fewer than two is ``not_verifiable``.
    2. Observed rate ``1 / median`` of the positive intervals, so a few gaps or
       duplicated samples do not move it; irregularity
       ``max|diff - median| / median``.
    3. A cutoff at or above half the observed rate is ``invalid`` and fatal --
       tested even on a non-increasing timebase, which is otherwise
       ``invalid`` but not fatal.
    4. Otherwise ``deviation`` when the rate or the regularity is outside
       tolerance, else ``consistent``.

    Applicability
    -------------
    Machine-independent.  Any sampled record with a declared rate.

    Limitations
    -----------
    Report-only: the declared rate is never replaced by the observed one. The
    observed rate of a record with fewer than two finite timestamps cannot be
    measured (``not_verifiable``).

    Provenance
    ----------
    .. [1] VAFT issue #2159 (declared versus observed acquisition) and #432
       (IMPA sample-rate consistency); no external source.
    """
    declared_hz = _finite_or_none(declared)
    raw_cutoffs = [] if cutoff_hz is None else np.atleast_1d(np.asarray(cutoff_hz, dtype=float)).tolist()
    bad_cutoff = any(not np.isfinite(value) or value <= 0.0 for value in raw_cutoffs)
    cutoffs = [float(value) for value in raw_cutoffs if np.isfinite(value)]
    highest_cutoff = max(cutoffs) if cutoffs else None
    t = np.asarray(time, dtype=float).reshape(-1)
    t = t[np.isfinite(t)]

    def result(status, reason, *, observed=None, deviation=None, irregularity=None, fatal=False):
        return SampleRateCheck(
            status=status,
            declared_hz=declared_hz,
            observed_hz=observed,
            relative_deviation=deviation,
            irregularity=irregularity,
            samples=int(t.size),
            cutoff_hz=highest_cutoff,
            fatal=bool(fatal),
            reason=reason,
        )

    if declared_hz is None or declared_hz <= 0.0:
        return result("invalid", f"declared sample rate {declared!r} is not a positive finite number")
    if bad_cutoff:
        return result("invalid", f"filter cutoff {cutoff_hz!r} is not a positive finite frequency")
    if t.size < 2:
        return result("not_verifiable", f"{t.size} finite timestamps; a rate needs at least two")
    steps = np.diff(t)
    monotonic = bool(np.all(steps > 0.0))
    positive = steps[steps > 0.0]
    if positive.size == 0:
        return result("invalid", "the timebase is not strictly increasing")
    median = float(np.median(positive))
    observed = 1.0 / median
    deviation = abs(observed - declared_hz) / declared_hz
    irregularity = float(np.max(np.abs(steps - median)) / median)
    measured = {"observed": observed, "deviation": deviation, "irregularity": irregularity}
    if highest_cutoff is not None and highest_cutoff >= 0.5 * observed:
        return result(
            "invalid",
            f"cutoff {highest_cutoff:g} Hz is at or above the observed Nyquist frequency "
            f"{0.5 * observed:g} Hz (observed rate {observed:g} Hz, declared {declared_hz:g} Hz)",
            fatal=True,
            **measured,
        )
    if not monotonic:
        return result("invalid", "the timebase is not strictly increasing", **measured)
    reasons = []
    if deviation > rtol:
        reasons.append(f"observed rate {observed:g} Hz differs from the declared {declared_hz:g} Hz")
    if irregularity > irregular_rtol:
        reasons.append(f"irregular sampling: an interval departs {irregularity:.3g} from the median")
    if reasons:
        return result("deviation", "; ".join(reasons), **measured)
    return result("consistent", "observed rate agrees with the declaration", **measured)


def summarize_sample_rate_checks(checks: Mapping[str, SampleRateCheck]) -> dict[str, Any]:
    """A compact, JSON-ready record of per-channel checks for a manifest.

    Parameters
    ----------
    checks : mapping of str to SampleRateCheck
        One check per channel, keyed by channel name [-].

    Returns
    -------
    dict
        ``status`` (the worst outcome: ``invalid`` > ``deviation`` >
        ``consistent`` > ``not_verifiable``), the declared rates, the
        observed-rate range, the count per status, and the full record of every
        channel that is not ``consistent`` [-].

    Applicability
    -------------
    Machine-independent.  Any set of :func:`check_declared_sample_rate` results.

    Provenance
    ----------
    .. [1] VAFT issue #2159; no external source.
    """
    counts = {status: 0 for status in ACQUISITION_STATUSES}
    observed = []
    flagged = {}
    declared = set()
    for name, check in checks.items():
        counts[check.status] += 1
        if check.declared_hz is not None:
            declared.add(check.declared_hz)
        if check.observed_hz is not None:
            observed.append(check.observed_hz)
        if check.status != "consistent":
            flagged[str(name)] = check.as_dict()
    if counts["invalid"]:
        overall = "invalid"
    elif counts["deviation"]:
        overall = "deviation"
    elif counts["consistent"]:
        overall = "consistent"
    else:
        overall = "not_verifiable"
    return {
        "status": overall,
        "declared_hz": sorted(declared),
        "observed_hz_range": [min(observed), max(observed)] if observed else None,
        "counts": counts,
        "channels": flagged,
    }
