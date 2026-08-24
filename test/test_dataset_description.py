"""Unit tests for dataset_description pulse-timestamp wiring (issue #126).

`dataset_description.pulse_time_begin` is a real IMAS field that was never
populated by any builder in this module. `date_from_shot`/`list_shots` in
`vaft.database.raw` are the authoritative SQL source for a shot's real
acquisition timestamp, so `dataset_description_from_raw_database` now uses
them opportunistically -- these tests fake `vaft.database.raw.date_from_shot`
so no live SQL connection is needed.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from omas import ODS

from vaft.machine_mapping.dataset_description import (
    dataset_description,
    dataset_description_from_raw_database,
    vfit_dataset_description,
)


def test_vfit_dataset_description_without_pulse_datetime_leaves_it_unset():
    ods = ODS()
    vfit_dataset_description(ods, shot=39915, run=1)

    assert ods["dataset_description.data_entry.pulse"] == 39915
    assert "dataset_description.pulse_time_begin" not in ods


def test_vfit_dataset_description_stores_pulse_datetime_as_iso8601():
    ods = ODS()
    when = datetime(2026, 5, 1, 8, 30, 0)

    vfit_dataset_description(ods, shot=39915, run=1, pulse_datetime=when)

    assert ods["dataset_description.pulse_time_begin"] == when.isoformat()


def test_dataset_description_forwards_pulse_datetime_from_options():
    ods = ODS()
    when = datetime(2026, 5, 1, 8, 30, 0)

    dataset_description(ods, 39915, {"pulse_datetime": when})

    assert ods["dataset_description.pulse_time_begin"] == when.isoformat()


def test_from_raw_database_fetches_pulse_datetime_when_sql_available(monkeypatch):
    when = datetime(2026, 5, 1, 8, 30, 0)

    def fake_date_from_shot(shot):
        assert shot == 39915
        return "2026-05-01", when

    import vaft.database.raw as raw

    monkeypatch.setattr(raw, "date_from_shot", fake_date_from_shot)

    ods = ODS()
    dataset_description_from_raw_database(ods, 39915)

    assert ods["dataset_description.pulse_time_begin"] == when.isoformat()


def test_from_raw_database_respects_an_explicit_pulse_datetime(monkeypatch):
    """An explicit pulse_datetime in options must not be overwritten by SQL."""
    explicit = datetime(2000, 1, 1, 0, 0, 0)

    def unexpected_sql_call(shot):
        pytest.fail("date_from_shot must not be called when pulse_datetime is already given")

    import vaft.database.raw as raw

    monkeypatch.setattr(raw, "date_from_shot", unexpected_sql_call)

    ods = ODS()
    dataset_description_from_raw_database(ods, 39915, {"pulse_datetime": explicit})

    assert ods["dataset_description.pulse_time_begin"] == explicit.isoformat()


def test_from_raw_database_degrades_gracefully_when_sql_unavailable(monkeypatch):
    """A database error must not prevent dataset_description from being built."""

    def failing_date_from_shot(shot):
        raise ConnectionError("simulated: SQL server unreachable")

    import vaft.database.raw as raw

    monkeypatch.setattr(raw, "date_from_shot", failing_date_from_shot)

    ods = ODS()
    dataset_description_from_raw_database(ods, 39915)

    assert ods["dataset_description.data_entry.pulse"] == 39915
    assert "dataset_description.pulse_time_begin" not in ods


def test_from_raw_database_shot_not_found_leaves_pulse_time_unset(monkeypatch):
    def missing_shot(shot):
        return None, None

    import vaft.database.raw as raw

    monkeypatch.setattr(raw, "date_from_shot", missing_shot)

    ods = ODS()
    dataset_description_from_raw_database(ods, 999999)

    assert "dataset_description.pulse_time_begin" not in ods
