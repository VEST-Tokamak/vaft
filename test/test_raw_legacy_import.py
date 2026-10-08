from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from vaft.cli import raw_legacy_import as legacy
from vaft.database import raw as raw_db


def _write_csv(path, time, data):
    path.write_text("".join(f"{t:.6g},{v:.6g}\n" for t, v in zip(time, data)), encoding="ascii")


@pytest.fixture
def legacy_tree(tmp_path):
    root = tmp_path / "VEST Database"
    shot = root / "20000"
    shot.mkdir(parents=True)
    slow_t = np.arange(100) * 4e-5
    _write_csv(shot / "1.csv", slow_t, np.sin(slow_t * 1e3))
    fast_t = np.arange(100) * 4e-6
    _write_csv(shot / "109.csv", fast_t, np.linspace(0.0, 1.0, 100))
    _write_csv(shot / "12.csv", slow_t, np.zeros(100))  # dead channel
    _write_csv(shot / "25.csv", np.array([0.1, 0.1, 0.2]), np.array([1.0, 2.0, 3.0]))
    (shot / "59.csv").write_text("o\n", encoding="ascii")
    (shot / "20000_remark.txt").write_text("PF2 test", encoding="utf-8")
    (shot / "Thumbs.db").write_bytes(b"x")
    (root / "20001").mkdir()  # not yet transferred
    (root / "shotList3.csv").write_text(
        "shotCode,shotNumber,recordDateTime\n20000,20000,2018-06-01 12:34\n", encoding="ascii"
    )
    return root


def test_import_writes_canonical_archive_loadable_by_load_raw(legacy_tree, tmp_path):
    filedb = tmp_path / "FileDB"
    assert legacy.main(["--legacy-root", str(legacy_tree), "--filedb-root", str(filedb)]) == 0

    out_dir = filedb / "raw" / "20000"
    archive = out_dir / "vest_20000_daq_raw.json.gz"
    manifest = json.loads((out_dir / "vest_20000_daq_manifest.json").read_text())
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["shot"] == 20000
    assert payload["pulse_datetime"] == "2018-06-01T12:34:00"
    assert sorted(payload["fields"], key=int) == ["1", "12", "109"]
    assert payload["fields"]["1"]["type"] == "slow"
    assert payload["fields"]["109"]["type"] == "fast"
    assert payload["field_quality"] == {"12": "all_zero"}

    time, data = raw_db.load_raw(20000, 109, sample_opt=str(archive))
    assert time[0] == 0.0
    assert time[1] == pytest.approx(4e-6)
    assert data[-1] == pytest.approx(1.0)

    assert manifest["source"]["kind"] == "legacy-csv"
    assert manifest["inventory"]["field_codes"] == [1, 12, 109]
    assert manifest["quality_summary"]["all_zero"] == ["12"]
    assert manifest["legacy"]["skipped_fields"] == {"25": "nonuniform", "59": "malformed"}
    assert manifest["legacy"]["remark"] == "PF2 test"
    assert manifest["legacy"]["ignored_files"] == ["Thumbs.db"]
    assert manifest["output"]["name"] == archive.name
    assert not (filedb / "raw" / "20001").exists()


def test_trigger_policy_offsets_fast_records_only(legacy_tree, tmp_path):
    payload, info = legacy.convert_shot(
        20000, legacy_tree / "20000", pulse_datetime=None, timebase_policy="trigger"
    )
    offset = raw_db._daq_trigger_time_correction(20000)
    assert info["fast_time_offset"] == offset
    assert payload["fields"]["109"]["t0"] == pytest.approx(offset)
    assert payload["fields"]["1"]["t0"] == 0.0


def test_existing_products_are_not_overwritten(legacy_tree, tmp_path):
    filedb = tmp_path / "FileDB"
    out_dir = filedb / "raw" / "20000"
    out_dir.mkdir(parents=True)
    existing = out_dir / "vest_20000_daq_raw.json.gz"
    existing.write_bytes(b"production")

    args = ["--legacy-root", str(legacy_tree), "--filedb-root", str(filedb), "--shots", "20000"]
    assert legacy.main(args) == 0
    assert existing.read_bytes() == b"production"

    assert legacy.main([*args, "--force"]) == 0
    with gzip.open(existing, "rt", encoding="utf-8") as handle:
        assert json.load(handle)["shot"] == 20000


def test_legacy_field_entry_rejects_nonuniform_time():
    # dt = 2e-5 end to end, but the third sample sits a full interval early.
    time = np.array([0.0, 1e-5, 2e-5, 6e-5])
    with pytest.raises(legacy.LegacyFieldError) as error:
        legacy.legacy_field_entry(time, np.ones(4))
    assert error.value.reason == "nonuniform"
