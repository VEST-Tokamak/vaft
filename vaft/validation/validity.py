"""What a datum's validity nodes mean, stated once (issue #424).

The Data Dictionary gives a datum two validity nodes: a scalar ``validity``
and a per-sample ``validity_timed``.  VAFT writes the scalar as the *worst
state reached* over the record (:func:`aggregate_validity`), so a channel that
holds its last value after the diagnostics window reads ``-2`` there while
every earlier sample is a measurement.  Three consumers independently read
that scalar as "the channel is unusable" before #410 caught them, and the rule
that should have stopped them lived only in a docstring.

This module is that rule, as code:

1. When ``validity_timed`` is present it is authoritative.  The scalar is
   ignored for usability; it remains what it is, a summary for readers that
   want one.
2. When only the scalar is present it is broadcast over the record.
3. When neither is present the datum is *unassessed*.  Missing is not zero
   (#253): the usable mask is the consumer's ``default`` and nothing is
   condemned.
4. ``min_validity`` is the consumer's floor on the DD code.  A datum is
   condemned when it was assessed and no sample reaches the floor -- never
   because the scalar is below it.

Nothing here reads an ODS.  A :class:`ValidityRecord` is what an accessor
extracted, uninterpreted; :mod:`vaft.validation.imas` builds one from an OMAS
tree or a plain mapping, and a native-IMAS accessor can build the same record
from an IDS handle and reuse every function below unchanged.  The module
imports only NumPy so that the interpretation can never depend on which
access layer asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "VALIDITY_CERTIFIED",
    "VALIDITY_INVALID",
    "VALIDITY_MEANINGS",
    "VALIDITY_SUSPECT",
    "VALIDITY_VALID",
    "ValidityRecord",
    "aggregate_validity",
    "codes_at",
    "is_condemned",
    "is_usable",
    "record_from_mask",
    "usable_fraction",
    "usable_mask",
]

#: Data Dictionary validity codes, in descending order of confidence.  The
#: codes are severity-monotone, which is what makes ``min()`` the natural
#: "worst state wins" aggregation in :func:`aggregate_validity`.
VALIDITY_CERTIFIED = 1
VALIDITY_VALID = 0
VALIDITY_SUSPECT = -1
VALIDITY_INVALID = -2

#: Human-readable meanings, for manifests and error messages.  Codes below -2
#: are code-specific by the DD's own definition and are reported verbatim.
VALIDITY_MEANINGS: Mapping[int, str] = {
    VALIDITY_CERTIFIED: "valid and certified",
    VALIDITY_VALID: "valid from automated processing",
    VALIDITY_SUSPECT: "problem identified in processing; verification requested",
    VALIDITY_INVALID: "invalid; should not be used",
}


@dataclass(frozen=True, eq=False)
class ValidityRecord:
    """What a datum's validity nodes say -- extracted, not yet interpreted.

    ``scalar`` is ``<base>.validity`` or ``None`` when the node is absent;
    ``timed`` is ``<base>.validity_timed`` on the datum's own grid or
    ``None``; ``time`` is the grid ``timed`` is coordinated on, or ``None``
    when the accessor could not resolve one.  ``time`` is needed even when
    ``timed`` is absent: a scalar broadcast over the record has to know how
    long the record is.
    """

    scalar: int | None
    timed: np.ndarray | None
    time: np.ndarray | None

    def __post_init__(self) -> None:
        if self.scalar is not None:
            object.__setattr__(self, "scalar", int(self.scalar))
        if self.timed is not None:
            object.__setattr__(self, "timed", np.asarray(self.timed, dtype=int).reshape(-1))
        if self.time is not None:
            object.__setattr__(self, "time", np.asarray(self.time, dtype=float).reshape(-1))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValidityRecord):
            return NotImplemented
        return (
            self.scalar == other.scalar
            and _same(self.timed, other.timed)
            and _same(self.time, other.time)
        )

    __hash__ = None  # type: ignore[assignment]  # arrays are not hashable

    @property
    def assessed(self) -> bool:
        """Whether anything has said what this datum's validity is."""
        return self.scalar is not None or self.timed is not None


def _same(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.shape == right.shape and bool(np.array_equal(left, right))


def aggregate_validity(timed: Iterable[int]) -> int:
    """The scalar validity summarizing a time-resolved one: **worst state wins**.

    The Data Dictionary codes descend monotonically in confidence
    (1 > 0 > -1 > -2), so the minimum is the most severe state the channel
    reached.  A channel that saturates halfway through therefore reads
    ``-2`` at the scalar level even though it was good earlier -- which is why
    ``validity_timed`` stays authoritative for consumers that work slice by
    slice, and why this scalar must never be used to discard a whole channel
    when the timed field is available (:func:`is_condemned` is the reading
    that honours this).

    An empty input aggregates to :data:`VALIDITY_INVALID`: a waveform with no
    samples carries no usable datum.  That is distinct from an *absent*
    ``validity_timed``, which an accessor reports as ``None``.
    """
    values = np.asarray(list(timed), dtype=int).reshape(-1)
    if values.size == 0:
        return VALIDITY_INVALID
    return int(values.min())


def _query(times: Any) -> np.ndarray:
    return np.asarray(times, dtype=float).reshape(-1)


def _record_size(record: ValidityRecord) -> int:
    return 1 if record.time is None else int(record.time.size)


def codes_at(record: ValidityRecord, *, times: Any | None = None) -> np.ndarray | None:
    """The record's validity code per sample.

    Without ``times`` the codes come back on the record's own grid.  With
    ``times`` they are resampled to those instants by nearest sample, which is
    what a consumer working at reconstruction time slices needs: validity is a
    per-sample state, not something to interpolate between.

    ``timed`` wins wherever it exists; otherwise the scalar is broadcast.
    ``None`` when the record declares neither.  A timed field whose grid the
    accessor could not resolve, or whose length disagrees with the grid, is
    reported at its own aggregate on every requested instant: that says what
    the channel was at its worst rather than pretending to a per-slice answer.
    """
    if record.timed is None:
        if record.scalar is None:
            return None
        size = _record_size(record) if times is None else _query(times).size
        return np.full(size, record.scalar, dtype=int)

    timed = record.timed
    if times is None:
        return timed
    grid = record.time
    query = _query(times)
    if grid is None or grid.size != timed.size:
        return np.full(query.size, aggregate_validity(timed), dtype=int)
    index = np.clip(np.searchsorted(grid, query), 0, grid.size - 1)
    left = np.clip(index - 1, 0, grid.size - 1)
    take_left = np.abs(grid[left] - query) <= np.abs(grid[index] - query)
    return timed[np.where(take_left, left, index)]


def usable_mask(
    record: ValidityRecord,
    *,
    times: Any | None = None,
    min_validity: int = VALIDITY_VALID,
    default: bool = True,
) -> np.ndarray:
    """Which samples a consumer accepting ``min_validity`` may use.

    ``min_validity`` is a floor on the DD code, so the default accepts
    ``valid`` (0) and ``certified`` (1) while rejecting ``suspect`` (-1) and
    ``invalid`` (-2).

    ``default`` decides what an unassessed record means.  It is ``True``
    deliberately: absence of an assessment is not a rejection, and it keeps
    every consumer byte-identical on data produced before the quality layer
    existed.
    """
    codes = codes_at(record, times=times)
    if codes is None:
        size = _record_size(record) if times is None else _query(times).size
        return np.full(size, bool(default))
    return codes >= int(min_validity)


def usable_fraction(
    record: ValidityRecord,
    *,
    times: Any | None = None,
    window: Any | None = None,
    min_validity: int = VALIDITY_VALID,
    default: bool = True,
) -> float:
    """Fraction of the samples in ``window`` the record declares usable.

    A metric, not a verdict: it says how much of the record survives
    ``min_validity``, and the consumer decides what fraction it requires.  That
    threshold is policy and belongs to the consumer, not here (#253 §7).

    ``window`` is a boolean selector on the same grid as the mask.  ``nan``
    when it selects no samples.
    """
    mask = usable_mask(record, times=times, min_validity=min_validity, default=default)
    if window is not None:
        selector = np.asarray(window, dtype=bool).reshape(-1)
        if selector.size != mask.size:
            raise ValueError(
                f"window has {selector.size} samples but the record has {mask.size}"
            )
        mask = mask[selector]
    if mask.size == 0:
        return float("nan")
    return float(mask.mean())


def is_usable(
    record: ValidityRecord,
    *,
    times: Any | None = None,
    min_validity: int = VALIDITY_VALID,
    default: bool = True,
) -> bool:
    """Whether any sample at all -- or any of ``times`` -- reaches the floor.

    The timed field wins; a scalar alone is broadcast; an unassessed record is
    ``default``.
    """
    return bool(usable_mask(record, times=times, min_validity=min_validity, default=default).any())


def is_condemned(record: ValidityRecord, *, min_validity: int = VALIDITY_VALID) -> bool:
    """Assessed, and no usable sample anywhere.

    ``False`` when nothing has assessed the datum: an unassessed record is not
    a rejected one.  Never ``scalar < min_validity``: with a timed field
    present the scalar is its minimum, so that reading would condemn every
    channel that was good early and bad late.
    """
    if not record.assessed:
        return False
    return not bool(usable_mask(record, min_validity=min_validity, default=True).any())


def record_from_mask(scalar: int | None, mask: Any | None) -> ValidityRecord:
    """A record for a consumer that holds a boolean usable-sample mask.

    Plotting extracts ``(scalar, mask)`` rather than the raw codes.  A ``True``
    sample is recorded as :data:`VALIDITY_VALID` and a ``False`` one as
    :data:`VALIDITY_INVALID`, so the same interpretation functions apply; the
    grid is unknown, which only matters for resampling, which such a consumer
    does not do.
    """
    if mask is None:
        return ValidityRecord(scalar=scalar, timed=None, time=None)
    accepted = np.asarray(mask, dtype=bool).reshape(-1)
    return ValidityRecord(
        scalar=scalar,
        timed=np.where(accepted, VALIDITY_VALID, VALIDITY_INVALID),
        time=None,
    )
