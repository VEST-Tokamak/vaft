"""The validity interpretation layer, stated once and tested once (issue #424).

Pure: no ODS, no samples.  A :class:`ValidityRecord` is what an accessor
extracted; these tests pin what it means.  The case that broke three
consumers -- a scalar ``-2`` beside a timed field with usable samples -- leads.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaft.validation.validity import (
    VALIDITY_CERTIFIED,
    VALIDITY_INVALID,
    VALIDITY_SUSPECT,
    VALIDITY_VALID,
    ValidityRecord,
    aggregate_validity,
    codes_at,
    is_condemned,
    is_usable,
    record_from_mask,
    usable_fraction,
    usable_mask,
)

GRID = np.array([0.0, 1.0, 2.0, 3.0])
HELD_TAIL = np.array([0, 0, -2, -2])


def _record(scalar=None, timed=None, time=None) -> ValidityRecord:
    return ValidityRecord(scalar=scalar, timed=timed, time=time)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_timed_with_usable_samples_beats_a_condemning_scalar():
    """The held-tail case: the scalar is the worst state reached, the timed
    field says the channel was good first."""
    record = _record(scalar=VALIDITY_INVALID, timed=HELD_TAIL, time=GRID)
    assert is_usable(record)
    assert not is_condemned(record)
    assert usable_mask(record).tolist() == [True, True, False, False]
    assert usable_fraction(record) == 0.5


def test_timed_all_below_the_floor_is_condemned():
    record = _record(scalar=VALIDITY_INVALID, timed=np.full(4, VALIDITY_INVALID), time=GRID)
    assert is_condemned(record)
    assert not is_usable(record)


def test_an_empty_timed_field_is_condemned():
    """A waveform with no samples carries no usable datum, which is the
    aggregate's own empty rule."""
    assert aggregate_validity([]) == VALIDITY_INVALID
    record = _record(scalar=None, timed=np.zeros(0, dtype=int), time=np.zeros(0))
    assert is_condemned(record)
    assert not is_usable(record)


def test_a_scalar_alone_is_broadcast():
    assert is_condemned(_record(scalar=VALIDITY_INVALID))
    assert not is_usable(_record(scalar=VALIDITY_INVALID))
    assert not is_condemned(_record(scalar=VALIDITY_VALID))
    assert is_usable(_record(scalar=VALIDITY_VALID))


def test_nothing_assessed_is_the_default_and_never_condemned():
    record = _record()
    assert not record.assessed
    assert not is_condemned(record)
    assert is_usable(record)
    assert not is_usable(record, default=False)
    assert usable_mask(record, times=[0.5, 1.5, 2.5]).tolist() == [True, True, True]
    assert usable_mask(record, times=[0.5], default=False).tolist() == [False]


def test_the_floor_is_the_consumers():
    suspect = _record(timed=np.array([VALIDITY_SUSPECT, VALIDITY_SUSPECT]), time=GRID[:2])
    assert is_condemned(suspect)
    assert not is_condemned(suspect, min_validity=VALIDITY_SUSPECT)
    valid = _record(timed=np.array([VALIDITY_VALID, VALIDITY_VALID]), time=GRID[:2])
    assert not is_condemned(valid)
    assert is_condemned(valid, min_validity=VALIDITY_CERTIFIED)


# ---------------------------------------------------------------------------
# Resampling and sizing, unchanged from the accessor's former behaviour
# ---------------------------------------------------------------------------

def test_codes_at_resamples_by_nearest_sample():
    record = _record(timed=HELD_TAIL, time=GRID)
    assert codes_at(record).tolist() == HELD_TAIL.tolist()
    # 0.4 -> 0; 1.6 -> 2; 2.5 ties and takes the left sample; 9 clips to the end
    assert codes_at(record, times=[0.4, 1.6, 2.5, 9.0]).tolist() == [0, -2, -2, -2]


def test_codes_at_falls_back_to_the_aggregate_when_the_grid_is_unresolvable():
    without_grid = _record(timed=HELD_TAIL, time=None)
    assert codes_at(without_grid, times=[0.0, 1.0, 2.0]).tolist() == [-2, -2, -2]
    mismatched = _record(timed=HELD_TAIL, time=GRID[:3])
    assert codes_at(mismatched, times=[0.0]).tolist() == [-2]
    # On its own grid the timed field is returned as stored, whatever the grid.
    assert codes_at(without_grid).tolist() == HELD_TAIL.tolist()


def test_a_scalar_broadcast_is_sized_by_the_record_grid():
    assert codes_at(_record(scalar=0)).shape == (1,)
    assert codes_at(_record(scalar=0, time=np.arange(5.0))).shape == (5,)
    assert codes_at(_record(scalar=0), times=[0.0, 0.1]).shape == (2,)
    assert codes_at(_record()) is None


def test_usable_fraction_is_a_metric_over_a_window():
    record = _record(timed=HELD_TAIL, time=GRID)
    assert usable_fraction(record, window=[True, True, False, False]) == 1.0
    assert usable_fraction(record, window=[False, False, True, True]) == 0.0
    assert math.isnan(usable_fraction(record, window=[False] * 4))
    with pytest.raises(ValueError, match="window has 3 samples"):
        usable_fraction(record, window=[True, True, True])


# ---------------------------------------------------------------------------
# Consumers that hold a boolean mask rather than codes
# ---------------------------------------------------------------------------

def test_record_from_mask_covers_the_four_series_cases():
    partly = record_from_mask(VALIDITY_INVALID, [True, True, False, False])
    nothing = record_from_mask(VALIDITY_INVALID, [False] * 4)
    bare = record_from_mask(VALIDITY_INVALID, None)
    unassessed = record_from_mask(None, None)
    assert not is_condemned(partly)
    assert is_condemned(nothing)
    assert is_condemned(bare)
    assert not is_condemned(unassessed)
    assert partly.timed.tolist() == [VALIDITY_VALID, VALIDITY_VALID, VALIDITY_INVALID, VALIDITY_INVALID]


def test_a_stored_mask_is_authoritative_even_against_a_clean_scalar():
    """Inconsistent data: the writer always sets the scalar to the timed
    minimum, so a clean scalar over an all-false mask cannot come from it.
    The rule still applies -- the timed field decides."""
    assert is_condemned(record_from_mask(VALIDITY_VALID, [False, False]))
    assert is_condemned(record_from_mask(None, [False, False]))


def test_the_record_normalizes_its_fields():
    record = ValidityRecord(scalar=np.int64(-1), timed=[[0, 1]], time=[[0.0, 1.0]])
    assert record.scalar == -1 and isinstance(record.scalar, int)
    assert record.timed.dtype.kind == "i" and record.timed.shape == (2,)
    assert record.time.shape == (2,)
