"""dump_all_raw_signals_for_shot() must capture pulse_datetime (issue #126)
and flag fields that are all-zero/all-NaN/empty rather than real data.

SQL mode resolves pulse_datetime from date_from_shot(); archive mode carries
forward whatever pulse_datetime the source archive already has, with no new
SQL call. Both paths are exercised without any real SQL connection.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime

import numpy as np
import pytest

from vaft.database import raw


def _read_dump(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_sql_mode_writes_pulse_datetime_from_date_from_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "get_all_field_codes_for_shot", lambda shot, max_retries=3: [13])
    monkeypatch.setattr(
        raw, "load_raw", lambda shot, fcode, max_retries=3, daq_type=0, sample_opt=False: (
            np.array([0.0, 1e-4, 2e-4]),
            np.array([1.0, 2.0, 3.0]),
        )
    )
    monkeypatch.setattr(raw, "date_from_shot", lambda shot: ("2026-05-01", datetime(2026, 5, 1, 8, 30, 0)))

    output = tmp_path / "raw.json.gz"
    assert raw.dump_all_raw_signals_for_shot(shot=39915, output_path=str(output))

    payload = _read_dump(output)
    assert payload["pulse_datetime"] == "2026-05-01T08:30:00"


def test_sql_mode_dump_still_succeeds_when_pulse_datetime_lookup_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "get_all_field_codes_for_shot", lambda shot, max_retries=3: [13])
    monkeypatch.setattr(
        raw, "load_raw", lambda shot, fcode, max_retries=3, daq_type=0, sample_opt=False: (
            np.array([0.0, 1e-4, 2e-4]),
            np.array([1.0, 2.0, 3.0]),
        )
    )

    def failing_date_from_shot(shot):
        raise ConnectionError("simulated: SQL server unreachable")

    monkeypatch.setattr(raw, "date_from_shot", failing_date_from_shot)

    output = tmp_path / "raw.json.gz"
    assert raw.dump_all_raw_signals_for_shot(shot=39915, output_path=str(output))

    payload = _read_dump(output)
    assert "pulse_datetime" not in payload
    assert payload["fields"]  # the dump itself is not blocked by the SQL failure


def test_archive_mode_carries_forward_the_source_archives_pulse_datetime(tmp_path, monkeypatch):
    def unexpected_sql_call(shot):
        pytest.fail("archive mode must not call date_from_shot")

    monkeypatch.setattr(raw, "date_from_shot", unexpected_sql_call)

    source = tmp_path / "shot_39915.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "shot": 39915,
                "pulse_datetime": "2026-05-01T08:30:00",
                "fields": {"13": {"type": "slow", "data": [1.0, 2.0, 3.0]}},
            },
            handle,
        )

    output = tmp_path / "redumped.json.gz"
    assert raw.dump_all_raw_signals_for_shot(
        shot=39915,
        output_path=str(output),
        sample_opt=str(tmp_path / "shot_{shot}.json.gz"),
    )

    payload = _read_dump(output)
    assert payload["pulse_datetime"] == "2026-05-01T08:30:00"


def test_archive_mode_without_a_source_pulse_datetime_omits_the_field(tmp_path, monkeypatch):
    def unexpected_sql_call(shot):
        pytest.fail("archive mode must not call date_from_shot")

    monkeypatch.setattr(raw, "date_from_shot", unexpected_sql_call)

    source = tmp_path / "shot_39915.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        json.dump(
            {"shot": 39915, "fields": {"13": {"type": "slow", "data": [1.0, 2.0, 3.0]}}},
            handle,
        )

    output = tmp_path / "redumped.json.gz"
    assert raw.dump_all_raw_signals_for_shot(
        shot=39915,
        output_path=str(output),
        sample_opt=str(tmp_path / "shot_{shot}.json.gz"),
    )

    payload = _read_dump(output)
    assert "pulse_datetime" not in payload


def test_flagged_field_quality_classifies_dead_and_empty_data():
    assert raw._flagged_field_quality(np.array([1.0, -0.5, 0.0, 2.3])) is None  # real signal
    assert raw._flagged_field_quality(np.array([0.0, 0.0, 0.0])) == "all_zero"
    assert raw._flagged_field_quality(np.array([np.nan, np.nan])) == "all_nan"
    assert raw._flagged_field_quality(np.array([])) == "empty"
    # Finite-but-zero mixed with NaN is still a dead channel, not "ok".
    assert raw._flagged_field_quality(np.array([np.nan, 0.0, 0.0])) == "all_zero"


def test_dump_flags_dead_channels_without_dropping_their_data(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "get_all_field_codes_for_shot", lambda shot, max_retries=3: [1, 2, 3])
    monkeypatch.setattr(raw, "date_from_shot", lambda shot: (None, None))

    fake_series = {
        1: np.array([1.0, -0.5, 2.3]),  # real
        2: np.array([0.0, 0.0, 0.0]),  # dead: all-zero
        3: np.array([np.nan, np.nan, np.nan]),  # dead: all-NaN
    }

    def fake_load_raw(shot, fcode, max_retries=3, daq_type=0, sample_opt=False):
        return np.array([0.0, 1e-4, 2e-4]), fake_series[fcode]

    monkeypatch.setattr(raw, "load_raw", fake_load_raw)

    output = tmp_path / "raw.json.gz"
    assert raw.dump_all_raw_signals_for_shot(shot=39915, output_path=str(output))

    payload = _read_dump(output)
    # The flagged data is still present -- marked, not deleted.
    assert payload["fields"]["2"]["data"] == [0.0, 0.0, 0.0]
    assert payload["field_quality"] == {"2": "all_zero", "3": "all_nan"}
    assert "1" not in payload["field_quality"]


def test_archive_redump_keeps_and_flags_empty_fields(tmp_path, monkeypatch):
    def unexpected_sql_call(shot):
        pytest.fail("archive mode must not call date_from_shot")

    monkeypatch.setattr(raw, "date_from_shot", unexpected_sql_call)
    source = tmp_path / "shot_39915.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "shot": 39915,
                "fields": {
                    "1": {"type": "slow", "data": []},
                    "2": {"type": "fast", "data": [1.0, 2.0, 3.0]},
                },
            },
            handle,
        )

    output = tmp_path / "redumped.json.gz"
    assert raw.dump_all_raw_signals_for_shot(
        shot=39915,
        output_path=str(output),
        sample_opt=str(tmp_path / "shot_{shot}.json.gz"),
    )

    payload = _read_dump(output)
    assert payload["fields"]["1"] == {"type": "slow", "data": []}
    assert payload["fields"]["2"]["data"] == [1.0, 2.0, 3.0]
    assert payload["field_quality"] == {"1": "empty"}


def test_dump_omits_field_quality_key_when_nothing_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "get_all_field_codes_for_shot", lambda shot, max_retries=3: [1])
    monkeypatch.setattr(raw, "date_from_shot", lambda shot: (None, None))
    monkeypatch.setattr(
        raw, "load_raw", lambda shot, fcode, max_retries=3, daq_type=0, sample_opt=False: (
            np.array([0.0, 1e-4, 2e-4]),
            np.array([1.0, -0.5, 2.3]),
        )
    )

    output = tmp_path / "raw.json.gz"
    assert raw.dump_all_raw_signals_for_shot(shot=39915, output_path=str(output))

    assert "field_quality" not in _read_dump(output)
