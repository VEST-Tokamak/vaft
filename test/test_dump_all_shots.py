"""Unit tests for the dump_all_shots.py backfill driver.

Discovery (list_shots) and per-shot export (dump_shot) are monkeypatched so
these test the driver's own logic -- skip-existing, dry-run, failure
isolation, and summary writing -- without any real SQL connection.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
    / "dump_all_shots.py"
)
SPEC = importlib.util.spec_from_file_location("dump_all_shots", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _run(monkeypatch, tmp_path, argv, shots, dump_side_effect=None):
    monkeypatch.setattr(MODULE, "list_shots", lambda **kwargs: shots)
    calls = []

    def fake_dump_shot(shot, output, metadata):
        calls.append((shot, output, metadata))
        if dump_side_effect is not None:
            dump_side_effect(shot)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("stub")

    monkeypatch.setattr(MODULE, "dump_shot", fake_dump_shot)
    monkeypatch.setattr("sys.argv", ["dump_all_shots.py", "--filedb-root", str(tmp_path), *argv])
    rc = MODULE.main()
    return rc, calls


def test_discovers_and_dumps_every_shot_into_filedb_layout(monkeypatch, tmp_path):
    shots = [(45000, datetime(2026, 5, 1, 8, 0, 0)), (45001, datetime(2026, 5, 2, 9, 0, 0))]

    rc, calls = _run(monkeypatch, tmp_path, [], shots)

    assert rc == 0
    assert [shot for shot, _, _ in calls] == [45000, 45001]
    output_45000 = calls[0][1]
    assert output_45000 == tmp_path / "raw" / "45000" / "vest_45000_daq_raw.json.gz"


def test_dry_run_does_not_call_dump_shot(monkeypatch, tmp_path):
    shots = [(45000, datetime(2026, 5, 1, 8, 0, 0))]

    rc, calls = _run(monkeypatch, tmp_path, ["--dry-run"], shots)

    assert rc == 0
    assert calls == []


def test_skip_existing_skips_shots_whose_dump_is_already_on_disk(monkeypatch, tmp_path):
    shots = [(45000, datetime(2026, 5, 1, 8, 0, 0)), (45001, datetime(2026, 5, 2, 9, 0, 0))]
    existing = tmp_path / "raw" / "45000" / "vest_45000_daq_raw.json.gz"
    existing.parent.mkdir(parents=True)
    existing.write_text("already here")

    rc, calls = _run(monkeypatch, tmp_path, ["--skip-existing"], shots)

    assert rc == 0
    assert [shot for shot, _, _ in calls] == [45001]


def test_one_failed_shot_does_not_abort_the_rest(monkeypatch, tmp_path):
    shots = [
        (45000, datetime(2026, 5, 1, 8, 0, 0)),
        (45001, datetime(2026, 5, 2, 9, 0, 0)),
        (45002, datetime(2026, 5, 3, 10, 0, 0)),
    ]

    def side_effect(shot):
        if shot == 45001:
            raise RuntimeError("simulated SQL failure")

    rc, calls = _run(monkeypatch, tmp_path, [], shots, dump_side_effect=side_effect)

    assert rc == 1  # nonzero when anything failed, so a backfill's exit code is checkable
    assert [shot for shot, _, _ in calls] == [45000, 45001, 45002]  # all attempted


def test_summary_records_completed_skipped_and_failed(monkeypatch, tmp_path):
    shots = [
        (45000, datetime(2026, 5, 1, 8, 0, 0)),
        (45001, datetime(2026, 5, 2, 9, 0, 0)),
    ]
    existing = tmp_path / "raw" / "45000" / "vest_45000_daq_raw.json.gz"
    existing.parent.mkdir(parents=True)
    existing.write_text("already here")

    summary_path = tmp_path / "summary.json"
    _run(monkeypatch, tmp_path, ["--skip-existing", "--summary", str(summary_path)], shots)

    summary = json.loads(summary_path.read_text())
    assert summary["discovered"] == 2
    assert summary["skipped"] == [45000]
    assert summary["completed"] == [45001]
    assert summary["failed"] == []


def test_limit_caps_the_number_of_shots_processed(monkeypatch, tmp_path):
    shots = [(45000 + i, datetime(2026, 5, 1, 8, 0, 0)) for i in range(5)]

    rc, calls = _run(monkeypatch, tmp_path, ["--limit", "2"], shots)

    assert [shot for shot, _, _ in calls] == [45000, 45001]


def test_shot_and_date_filters_are_forwarded_to_list_shots(monkeypatch, tmp_path):
    captured = {}

    def fake_list_shots(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(MODULE, "list_shots", fake_list_shots)
    monkeypatch.setattr(MODULE, "dump_shot", lambda shot, output, metadata: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "dump_all_shots.py",
            "--filedb-root",
            str(tmp_path),
            "--shot-min",
            "45000",
            "--shot-max",
            "46000",
            "--start-date",
            "2026-05-01",
            "--end-date",
            "2026-05-31",
        ],
    )

    MODULE.main()

    assert captured == {
        "shot_min": 45000,
        "shot_max": 46000,
        "start_date": "2026-05-01",
        "end_date": "2026-05-31",
    }
