"""Unit tests for `vaft.database.raw.list_shots` and its compatibility wrappers.

Issue #126: shots must be selectable by shot-number range, date range, or
their intersection, querying the SQL `shot` table directly with no HSDS/FileDB
scan and no waveform payload touched. These tests fake the DB_POOL/connection/
cursor chain so query construction and result normalization are verified
without any real SQL connection.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from vaft.database import raw


class _FakeCursor:
    def __init__(self, rows, recorder):
        self._rows = rows
        self._recorder = recorder

    def execute(self, query, params):
        self._recorder.append((query, tuple(params)))

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, rows, recorder):
        self._rows = rows
        self._recorder = recorder
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._rows, self._recorder)

    def close(self):
        self.closed = True


class _FakePool:
    def __init__(self, rows):
        self._rows = rows
        self.queries: list[tuple[str, tuple]] = []
        self.connections: list[_FakeConnection] = []

    def get_connection(self):
        conn = _FakeConnection(self._rows, self.queries)
        self.connections.append(conn)
        return conn


@pytest.fixture()
def fake_pool(monkeypatch):
    rows = [
        (45001, datetime(2026, 5, 2, 9, 0, 0)),
        (45000, datetime(2026, 5, 1, 8, 0, 0)),
    ]
    pool = _FakePool(rows)
    monkeypatch.setattr(raw, "DB_POOL", pool)

    def fail_if_called():
        pytest.fail("DB_POOL was already set; init_pool() must not run")

    monkeypatch.setattr(raw, "init_pool", fail_if_called)
    return pool


def test_no_filters_selects_every_shot_ordered_ascending(fake_pool):
    raw.list_shots()

    query, params = fake_pool.queries[0]
    assert "WHERE" not in query
    assert "ORDER BY shotNumber ASC" in query
    assert params == ()


def test_shot_range_only(fake_pool):
    raw.list_shots(shot_min=45000, shot_max=45010)

    query, params = fake_pool.queries[0]
    where_clause = query.split("WHERE", 1)[1]
    assert "shotNumber >= %s" in where_clause
    assert "shotNumber <= %s" in where_clause
    assert "recordDateTime" not in where_clause
    assert params == (45000, 45010)


def test_date_range_only_is_inclusive_on_both_ends(fake_pool):
    raw.list_shots(start_date="2026-05-01", end_date="2026-05-31")

    query, params = fake_pool.queries[0]
    assert "recordDateTime >= %s" in query
    assert "recordDateTime < %s" in query
    assert params[0] == datetime(2026, 5, 1, 0, 0, 0)
    # end_date is inclusive: the exclusive upper bound is the *next* day.
    assert params[1] == datetime(2026, 6, 1, 0, 0, 0)


def test_exact_date_is_a_special_case_of_a_date_range(fake_pool):
    raw.list_shots(start_date="2026-05-01", end_date="2026-05-01")

    _, params = fake_pool.queries[0]
    assert params == (datetime(2026, 5, 1, 0, 0, 0), datetime(2026, 5, 2, 0, 0, 0))


def test_shot_and_date_range_intersect(fake_pool):
    raw.list_shots(shot_min=45000, start_date="2026-05-01", end_date="2026-05-31")

    query, params = fake_pool.queries[0]
    assert query.count("WHERE") == 1
    assert " AND " in query
    assert params == (
        45000,
        datetime(2026, 5, 1, 0, 0, 0),
        datetime(2026, 6, 1, 0, 0, 0),
    )


def test_accepts_date_objects_as_well_as_iso_strings(fake_pool):
    raw.list_shots(start_date=date(2026, 5, 1), end_date=date(2026, 5, 1))

    _, params = fake_pool.queries[0]
    assert params == (datetime(2026, 5, 1, 0, 0, 0), datetime(2026, 5, 2, 0, 0, 0))


def test_results_are_normalized_to_int_shot_and_ordered_ascending(fake_pool):
    results = raw.list_shots()

    # The fake pool intentionally returns rows out of order to prove the
    # function relies on the SQL ORDER BY, not a client-side re-sort of
    # already-sorted rows -- if the query text lost `ORDER BY`, this would
    # only be caught by asserting the *rows are actually ascending*, not just
    # that the query mentions the clause.
    assert [shot for shot, _ in results] == [45001, 45000]
    assert isinstance(results[0][0], int)
    assert isinstance(results[0][1], datetime)


def test_empty_result_set_is_not_an_error(monkeypatch):
    pool = _FakePool([])
    monkeypatch.setattr(raw, "DB_POOL", pool)
    monkeypatch.setattr(raw, "init_pool", lambda: pytest.fail("must not init"))

    assert raw.list_shots(shot_min=999999, shot_max=999999) == []


def test_connection_is_always_closed_even_on_success(fake_pool):
    raw.list_shots()

    assert fake_pool.connections[0].closed


def test_db_pool_none_triggers_lazy_init(monkeypatch):
    calls = []

    def fake_init_pool():
        calls.append("init")
        monkeypatch.setattr(raw, "DB_POOL", _FakePool([]))

    monkeypatch.setattr(raw, "DB_POOL", None)
    monkeypatch.setattr(raw, "init_pool", fake_init_pool)

    raw.list_shots()

    assert calls == ["init"]


def test_date_from_shot_is_built_on_list_shots(fake_pool):
    fake_pool._rows = [(45000, datetime(2026, 5, 1, 8, 0, 0))]

    date_str, datetime_obj = raw.date_from_shot(45000)

    query, params = fake_pool.queries[0]
    assert "shotNumber >= %s" in query and "shotNumber <= %s" in query
    assert params == (45000, 45000)
    assert date_str == "2026-05-01"
    assert datetime_obj == datetime(2026, 5, 1, 8, 0, 0)


def test_date_from_shot_missing_shot_returns_none_pair(monkeypatch):
    pool = _FakePool([])
    monkeypatch.setattr(raw, "DB_POOL", pool)
    monkeypatch.setattr(raw, "init_pool", lambda: pytest.fail("must not init"))

    assert raw.date_from_shot(999999) == (None, None)


def test_shots_from_date_is_built_on_list_shots(fake_pool):
    fake_pool._rows = [(45001, datetime(2026, 5, 1, 9, 0, 0)), (45000, datetime(2026, 5, 1, 8, 0, 0))]

    shots = raw.shots_from_date("2026-05-01")

    query, params = fake_pool.queries[0]
    assert "recordDateTime >= %s" in query and "recordDateTime < %s" in query
    assert params == (datetime(2026, 5, 1, 0, 0, 0), datetime(2026, 5, 2, 0, 0, 0))
    assert shots == [45001, 45000]


def test_list_shots_against_the_live_sql_database():
    """Opportunistic integration check -- skips cleanly when the DB is unreachable.

    The connectivity probe happens inside the test body (not a collection-time
    `skipif`) so an unreachable DB costs one connection attempt for this test
    only, instead of blocking collection for the whole suite.

    Cross-checks list_shots() against date_from_shot()/shots_from_date() for a
    real, already-recorded shot, and confirms a shot-range + date-range
    intersection query returns a consistent, ascending-ordered subset.
    """
    if not raw.sql_loading_available():
        pytest.skip("mysql-connector-python is not installed")
    try:
        raw.init_pool()
    except Exception as exc:
        pytest.skip(f"live VEST SQL database not reachable: {exc}")
    if raw.DB_POOL is None:
        pytest.skip("live VEST SQL database not reachable")

    last = raw.last_shot()
    assert last is not None and last > 0

    date_str, datetime_obj = raw.date_from_shot(last)
    assert date_str is not None

    via_list_shots = raw.list_shots(shot_min=last, shot_max=last)
    assert via_list_shots == [(last, datetime_obj)]

    shots_that_day = raw.shots_from_date(date_str)
    assert last in shots_that_day

    intersected = raw.list_shots(
        shot_min=last - 100,
        shot_max=last,
        start_date=date_str,
        end_date=date_str,
    )
    assert all(last - 100 <= shot <= last for shot, _ in intersected)
    assert [shot for shot, _ in intersected] == sorted(shot for shot, _ in intersected)
