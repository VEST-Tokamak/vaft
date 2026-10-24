"""Declared acquisition settings checked against the acquired timebase (#2159).

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
from typing import Any

import numpy as np

__all__ = [
    "ACQUISITION_STATUSES",
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


def _finite_or_none(value: Any) -> float | None:
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
        Relative tolerance for ``consistent`` [-].
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
    """
    declared_hz = _finite_or_none(declared)
    cutoffs = (
        []
        if cutoff_hz is None
        else [float(c) for c in np.atleast_1d(np.asarray(cutoff_hz, dtype=float)).tolist()]
    )
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
    if t.size < 2:
        return result("not_verifiable", f"{t.size} finite timestamps; a rate needs at least two")
    steps = np.diff(t)
    if np.any(steps <= 0.0):
        return result("invalid", "the timebase is not strictly increasing")
    median = float(np.median(steps))
    observed = 1.0 / median
    deviation = abs(observed - declared_hz) / declared_hz
    irregularity = float(np.max(np.abs(steps - median)) / median)
    if highest_cutoff is not None and highest_cutoff >= 0.5 * observed:
        return result(
            "invalid",
            f"cutoff {highest_cutoff:g} Hz is at or above the observed Nyquist frequency "
            f"{0.5 * observed:g} Hz (observed rate {observed:g} Hz, declared {declared_hz:g} Hz)",
            observed=observed,
            deviation=deviation,
            irregularity=irregularity,
            fatal=True,
        )
    reasons = []
    if deviation > rtol:
        reasons.append(f"observed rate {observed:g} Hz differs from the declared {declared_hz:g} Hz")
    if irregularity > irregular_rtol:
        reasons.append(f"irregular sampling: an interval departs {irregularity:.3g} from the median")
    if reasons:
        return result(
            "deviation", "; ".join(reasons), observed=observed, deviation=deviation, irregularity=irregularity
        )
    return result(
        "consistent",
        "observed rate agrees with the declaration",
        observed=observed,
        deviation=deviation,
        irregularity=irregularity,
    )


def summarize_sample_rate_checks(checks: dict[str, SampleRateCheck]) -> dict[str, Any]:
    """A compact, JSON-ready record of per-channel checks for a manifest.

    Counts per status, the observed-rate range, and the full record of every
    channel that is not ``consistent`` (consistent ones are only counted).
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
