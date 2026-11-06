import gzip
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
    / "validate_raw_dumps.py"
)
SPEC = importlib.util.spec_from_file_location("validate_raw_dumps", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _dump(path: Path, shot: int, fields: dict[int, list[float]]) -> None:
    payload = {
        "shot": shot,
        "fields": {str(code): {"data": values, "type": "slow"} for code, values in fields.items()},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_preflight_excludes_only_readable_archives_missing_required_signals(tmp_path):
    valid = tmp_path / "valid.json.gz"
    incomplete = tmp_path / "incomplete.json.gz"
    _dump(valid, 1, {1: [1.0, 2.0], 12: [1.0, 2.0], 109: [1.0, 2.0]})
    _dump(incomplete, 2, {1: [1.0, 2.0], 12: [1.0, 2.0], 109: [1.0]})

    eligible, excluded = MODULE.validate_raw_dumps([incomplete, valid], [1, 12, 109], min_samples=2)

    assert eligible == [1]
    assert excluded == [
        {
            "shot": 2,
            "reason": "missing_required_raw_signal",
            "missing_field_codes": [109],
            "raw_dump": str(incomplete),
        }
    ]


def test_preflight_fails_for_corrupt_or_incomplete_archive_export(tmp_path):
    corrupt = tmp_path / "corrupt.json.gz"
    corrupt.write_text("not a gzip archive", encoding="utf-8")

    with pytest.raises(MODULE.RawDumpValidationError, match="Cannot read raw dump"):
        MODULE.validate_raw_dumps([corrupt], [1])

    mismatch = tmp_path / "vest_3_daq_raw.json.gz"
    _dump(mismatch, 4, {1: [1.0, 2.0]})
    with pytest.raises(MODULE.RawDumpValidationError, match="named for shot 3 but contains shot 4"):
        MODULE.validate_raw_dumps([mismatch], [1])


@pytest.fixture()
def importable_module(monkeypatch):
    # A process pool pickles the worker function by module name, so the child
    # (forkserver/spawn) must be able to import the script the way `python
    # validate_raw_dumps.py` makes it importable.
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.setitem(__import__("sys").modules, "validate_raw_dumps", MODULE)
    return MODULE


def test_parallel_preflight_matches_the_serial_verdict(tmp_path, importable_module):
    paths = []
    for shot in range(1, 13):
        path = tmp_path / f"vest_{shot}_daq_raw.json.gz"
        short = [1.0] if shot % 4 == 0 else [1.0, 2.0]
        _dump(path, shot, {1: [1.0, 2.0], 12: [1.0, 2.0], 109: short})
        paths.append(path)

    serial = importable_module.validate_raw_dumps(paths[::-1], [1, 12, 109], min_samples=2)
    parallel = importable_module.validate_raw_dumps(paths[::-1], [1, 12, 109], min_samples=2, jobs=4)

    assert parallel == serial
    assert sorted(serial[0]) == [1, 2, 3, 5, 6, 7, 9, 10, 11]
    assert [entry["shot"] for entry in serial[1]] == [12, 4, 8]  # path order, as serially


def test_parallel_preflight_raises_the_error_the_serial_run_meets_first(tmp_path, importable_module):
    paths = []
    for shot in range(1, 7):
        path = tmp_path / f"vest_{shot}_daq_raw.json.gz"
        _dump(path, shot, {1: [1.0, 2.0]})
        paths.append(path)
    (tmp_path / "vest_3_daq_raw.json.gz").write_text("not a gzip archive", encoding="utf-8")
    (tmp_path / "vest_5_daq_raw.json.gz").write_text("not a gzip archive", encoding="utf-8")

    with pytest.raises(importable_module.RawDumpValidationError, match="vest_3_daq_raw"):
        importable_module.validate_raw_dumps(paths, [1], jobs=3)


def test_parallel_preflight_keeps_the_serial_error_order_for_any_read_failure(tmp_path, importable_module):
    # A truncated gzip raises EOFError, not RawDumpValidationError; the earlier
    # name mismatch must still be what both runs report.
    paths = []
    for shot in range(1, 7):
        path = tmp_path / f"vest_{shot}_daq_raw.json.gz"
        _dump(path, 99 if shot == 3 else shot, {1: [1.0, 2.0]})
        paths.append(path)
    truncated = tmp_path / "vest_6_daq_raw.json.gz"
    truncated.write_bytes(truncated.read_bytes()[:20])

    for jobs in (1, 3):
        with pytest.raises(importable_module.RawDumpValidationError, match="named for shot 3 but contains shot 99"):
            importable_module.validate_raw_dumps(paths, [1], jobs=jobs)
