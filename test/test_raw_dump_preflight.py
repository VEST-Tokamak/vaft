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
    _dump(valid, 1, {1: [1.0, 2.0], 13: [1.0, 2.0]})
    _dump(incomplete, 2, {1: [1.0, 2.0], 13: [1.0]})

    eligible, excluded = MODULE.validate_raw_dumps([incomplete, valid], [1, 13], min_samples=2)

    assert eligible == [1]
    assert excluded == [
        {
            "shot": 2,
            "reason": "missing_required_raw_signal",
            "missing_field_codes": [13],
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
