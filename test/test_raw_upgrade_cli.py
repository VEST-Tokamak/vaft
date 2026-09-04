"""The raw-upgrade CLI: batch resilience, manifests, byte reproducibility."""

import gzip
import json

import numpy as np
import pytest

from vaft.cli import raw_upgrade
from vaft.database import raw
from vaft.omas.vest_upstream import _write_raw_payload, sha256_file


def _write_dump(root, shot, payload, *, legacy_writer=False):
    shot_dir = root / "raw" / str(shot)
    shot_dir.mkdir(parents=True)
    path = shot_dir / f"vest_{shot}_daq_raw.json.gz"
    if legacy_writer:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    else:
        _write_raw_payload(path, payload)
    return path


def _legacy_payload(shot, n_fast=2000):
    return {
        "shot": shot,
        "fields": {
            "275": {"type": "fast", "data": [0.1] * n_fast},
            "1": {"type": "slow", "data": [0.2] * 100},
        },
    }


def test_one_corrupt_dump_does_not_abort_the_batch(tmp_path, capsys):
    good_before = _write_dump(tmp_path, 43100, _legacy_payload(43100))
    bad = _write_dump(tmp_path, 43101, _legacy_payload(43101))
    bad.write_bytes(b"not gzip at all")
    good_after = _write_dump(tmp_path, 43102, _legacy_payload(43102))

    code = raw_upgrade.main(["--filedb-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 1                       # the failure is reported...
    assert "shot 43101: FAILED" in out
    assert "upgraded 2 dump(s)" in out     # ...but the rest of the batch ran
    for path in (good_before, good_after):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            fields = json.load(handle)["fields"]
        assert all("t0" in entry and "dt" in entry for entry in fields.values())


def test_existing_manifest_keeps_its_provenance_and_gets_a_fresh_sha(tmp_path):
    shot = 43100
    dump = _write_dump(tmp_path, shot, _legacy_payload(shot))
    manifest_path = dump.parent / f"vest_{shot}_daq_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "stage": "raw",
        "shot": shot,
        "status": "success",
        "source": {"kind": "vest-sql", "name": dump.name},
        "custom_note": "written by the original backfill",
        "output": {"name": dump.name, "sha256": "stale"},
    }), encoding="utf-8")

    assert raw_upgrade.main(["--filedb-root", str(tmp_path)]) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["kind"] == "vest-sql"          # not downgraded
    assert manifest["custom_note"] == "written by the original backfill"
    assert manifest["output"]["sha256"] == sha256_file(dump)  # refreshed


def test_missing_manifest_gets_the_reduced_existing_filedb_form(tmp_path):
    shot = 43100
    dump = _write_dump(tmp_path, shot, _legacy_payload(shot))

    assert raw_upgrade.main(["--filedb-root", str(tmp_path)]) == 0

    manifest = json.loads(
        (dump.parent / f"vest_{shot}_daq_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["kind"] == "existing-filedb"
    assert manifest["output"]["sha256"] == sha256_file(dump)


def test_upgraded_archive_is_byte_identical_to_a_fresh_canonical_write(tmp_path):
    """The upgrade must use the canonical serializer (sorted, compact, mtime=0)."""
    shot = 43100
    payload = _legacy_payload(shot)
    dump = _write_dump(tmp_path, shot, dict(payload), legacy_writer=True)

    assert raw_upgrade.main(["--filedb-root", str(tmp_path)]) == 0

    expected_payload = json.loads(json.dumps(payload))
    raw.upgrade_archive_timebase(expected_payload)
    # Same basename: the canonical writer embeds it in the gzip header.
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    expected = expected_dir / dump.name
    _write_raw_payload(expected, expected_payload)
    assert dump.read_bytes() == expected.read_bytes()


def test_upgrade_is_a_noop_second_time(tmp_path, capsys):
    _write_dump(tmp_path, 43100, _legacy_payload(43100))
    assert raw_upgrade.main(["--filedb-root", str(tmp_path)]) == 0
    first = (tmp_path / "raw" / "43100" / "vest_43100_daq_raw.json.gz").read_bytes()

    assert raw_upgrade.main(["--filedb-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "upgraded 0 dump(s)" in out
    assert (tmp_path / "raw" / "43100" / "vest_43100_daq_raw.json.gz").read_bytes() == first


def test_dry_run_writes_nothing(tmp_path):
    dump = _write_dump(tmp_path, 43100, _legacy_payload(43100), legacy_writer=True)
    before = dump.read_bytes()

    assert raw_upgrade.main(["--filedb-root", str(tmp_path), "--dry-run"]) == 0

    assert dump.read_bytes() == before
    assert not (dump.parent / "vest_43100_daq_manifest.json").exists()
