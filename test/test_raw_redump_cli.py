from __future__ import annotations

import gzip
import json

from vaft.cli import raw_redump


def _fake_archive_raw_source(*, shot, output, source, max_retries):
    assert max_retries == 1
    assert source is None
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump({"shot": shot, "fields": {"1": {"data": [1.0, 2.0]}}}, handle)
    return {
        "shot": shot,
        "output": {"name": output.name, "sha256": "test-hash"},
    }


def test_raw_redump_writes_flat_serial_filedb_products(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_redump, "archive_raw_source", _fake_archive_raw_source)
    monkeypatch.setattr(raw_redump.time, "sleep", lambda _: None)

    result = raw_redump.main(
        [
            "--filedb-root",
            str(tmp_path / "FileDB"),
            "--shots",
            "48223",
            "48224",
            "--attempts",
            "1",
            "--inter-shot-delay",
            "0",
        ]
    )

    assert result == 0
    for shot in (48223, 48224):
        directory = tmp_path / "FileDB/raw" / str(shot)
        dump = directory / f"vest_{shot}_daq_raw.json.gz"
        manifest = directory / f"vest_{shot}_daq_manifest.json"
        assert dump.is_file()
        assert manifest.is_file()
        assert json.loads(manifest.read_text())["output"]["name"] == dump.name
        assert not (directory / "output").exists()
        assert not (directory / "metadata").exists()


def test_raw_redump_adds_a_manifest_to_a_valid_existing_dump(tmp_path, monkeypatch):
    root = tmp_path / "FileDB"
    dump = root / "raw/48223/vest_48223_daq_raw.json.gz"
    dump.parent.mkdir(parents=True)
    with gzip.open(dump, "wt", encoding="utf-8") as handle:
        json.dump({"shot": 48223, "fields": {}}, handle)

    monkeypatch.setattr(
        raw_redump,
        "archive_raw_source",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not export")),
    )
    assert raw_redump.main(
        ["--filedb-root", str(root), "--shots", "48223", "--inter-shot-delay", "0"]
    ) == 0
    manifest = root / "raw/48223/vest_48223_daq_manifest.json"
    assert json.loads(manifest.read_text())["source"]["kind"] == "existing-filedb"


def test_raw_redump_uses_the_fixed_default_range_and_allows_an_override():
    assert raw_redump._selected_shots(None, None) == list(range(29350, 48824))
    assert raw_redump._selected_shots(None, [48223, 48224]) == [48223, 48224]
    assert raw_redump._selected_shots([48224, 48223, 48224], None) == [48224, 48223]
