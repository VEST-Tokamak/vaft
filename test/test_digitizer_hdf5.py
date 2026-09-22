"""Lossless HDF5 containers for SXR digitizer CSVs, and the sxr-pack CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from vaft.cli import sxr_pack
from vaft.database import digitizer_hdf5 as dh
from vaft.machine_mapping.soft_x_rays import (
    _discover_digitizer_files,
    load_digitizer_csv,
    soft_x_rays_from_digitizer_csv,
)

PACKAGED = Path(__file__).resolve().parents[1] / "vaft" / "data" / "legacy"


def _values(channels: int = 4, samples: int = 257, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mixed magnitudes so repr() produces both plain and exponent notation.
    return rng.normal(size=(channels, samples)) * np.logspace(-9, 0, channels)[:, None]


def _write_sample_v3(path: Path, data: np.ndarray, terminator: str = "\r\n") -> bytes:
    payload = dh.rebuild_csv_bytes(data, terminator)
    path.write_bytes(payload)
    return payload


@pytest.mark.parametrize("terminator", ["\r\n", "\n"])
def test_pack_is_bit_exact_and_regenerates_the_csv(tmp_path, terminator):
    data = _values()
    csv = tmp_path / "digitizer_22577_12345.csv"
    original = _write_sample_v3(csv, data, terminator)

    result = dh.pack_digitizer_csv(csv)

    assert result.target == tmp_path / "digitizer_22577_12345.h5"
    assert result.shape == data.shape
    assert csv.read_bytes() == original  # the source is never touched
    loaded = dh.load_digitizer_hdf5(result.target)
    assert loaded.dtype == np.float64
    assert np.array_equal(loaded.view(np.uint64), data.view(np.uint64))
    attrs = dh.read_attributes(result.target)
    assert (attrs["shot"], attrs["daq_label"], attrs["line_terminator"]) == (12345, "22577", terminator)

    restored = dh.restore_csv(result.target, tmp_path / "restored.csv")
    assert restored.read_bytes() == original


def test_container_is_chunked_per_channel_and_compressed(tmp_path):
    csv = tmp_path / "digitizer_17592_12345.csv"
    _write_sample_v3(csv, _values(channels=3, samples=1000))
    target = dh.pack_digitizer_csv(csv).target
    with h5py.File(target) as handle:
        dataset = handle["data"]
        assert dataset.chunks == (1, 1000)
        assert dataset.compression == "gzip" and dataset.shuffle


def test_non_repr_text_is_refused_and_leaves_nothing_behind(tmp_path):
    csv = tmp_path / "digitizer_17592_12345.csv"
    np.savetxt(csv, _values(), delimiter=",")  # %.18e, not the sample_v3 format
    with pytest.raises(dh.DigitizerPackError, match="byte for byte"):
        dh.pack_digitizer_csv(csv)
    assert sorted(p.name for p in tmp_path.iterdir()) == [csv.name]


def test_ragged_csv_is_refused(tmp_path):
    csv = tmp_path / "digitizer_17592_12345.csv"
    csv.write_bytes(b"1.0,2.0,3.0\r\n4.0,5.0\r\n")
    with pytest.raises(dh.DigitizerPackError):
        dh.pack_digitizer_csv(csv)
    assert not dh.container_path(csv).exists()


def test_tampered_container_fails_verification(tmp_path):
    csv = tmp_path / "digitizer_17592_12345.csv"
    _write_sample_v3(csv, _values())
    target = dh.pack_digitizer_csv(csv).target
    with h5py.File(target, "r+") as handle:
        handle["data"][0, 0] = handle["data"][0, 0] + 1.0
    with pytest.raises(dh.DigitizerPackError):
        dh.verify_container(target)


def test_existing_container_is_not_overwritten(tmp_path):
    csv = tmp_path / "digitizer_17592_12345.csv"
    _write_sample_v3(csv, _values())
    dh.pack_digitizer_csv(csv)
    with pytest.raises(FileExistsError):
        dh.pack_digitizer_csv(csv)


def test_mapper_prefers_the_container_and_reads_identical_values(tmp_path):
    data = _values(channels=40, samples=64, seed=3)
    shot_dir = tmp_path / "12345"
    shot_dir.mkdir()
    csv = shot_dir / "digitizer_17592_12345.csv"
    _write_sample_v3(csv, data)
    from_csv = soft_x_rays_from_digitizer_csv(12345, data_root=tmp_path, time_reference="archive")

    target = dh.pack_digitizer_csv(csv).target
    assert _discover_digitizer_files(12345, tmp_path) == [("17592", target)]
    from_h5 = soft_x_rays_from_digitizer_csv(12345, data_root=tmp_path, time_reference="archive")

    assert np.array_equal(load_digitizer_csv(csv), load_digitizer_csv(target))
    for index in range(40):
        path = f"soft_x_rays.channel.{index}.brightness.data"
        assert np.array_equal(from_csv[path], from_h5[path])
    assert str(target) in from_h5["soft_x_rays.ids_properties.source"]

    csv.unlink()  # the container alone is a complete source
    alone = soft_x_rays_from_digitizer_csv(12345, data_root=tmp_path, time_reference="archive")
    assert np.array_equal(alone["soft_x_rays.channel.0.brightness.data"],
                          from_csv["soft_x_rays.channel.0.brightness.data"])


def test_digitizer_file_argument_accepts_a_container(tmp_path):
    csv = tmp_path / "digitizer_22577_12345.csv"
    _write_sample_v3(csv, _values(channels=64, samples=32))
    target = dh.pack_digitizer_csv(csv).target
    ods = soft_x_rays_from_digitizer_csv(12345, digitizer_file=target, time_reference="archive")
    assert len(ods["soft_x_rays.channel"]) == 64


@pytest.mark.parametrize("daq", ["17592", "22577"])
def test_real_campaign_record_round_trips_byte_for_byte(tmp_path, daq):
    source = PACKAGED / f"digitizer_{daq}_45531.csv"
    if not source.exists():
        pytest.skip("packaged SXR sample is a repository-only file")
    csv = tmp_path / source.name
    csv.write_bytes(source.read_bytes())
    result = dh.pack_digitizer_csv(csv)
    assert result.ratio < 0.5
    assert dh.verify_container(result.target, expected_sha256=result.source_sha256)
    assert np.array_equal(load_digitizer_csv(source), load_digitizer_csv(result.target))


def _archive(tmp_path: Path) -> Path:
    root = tmp_path / "soft_x_rays"
    for shot in (100, 101):
        (root / str(shot)).mkdir(parents=True)
        (root / str(shot) / "provenance.json").write_text(json.dumps({"shot": shot}))
    _write_sample_v3(root / "100" / "digitizer_17592_100.csv", _values(seed=1))
    _write_sample_v3(root / "100" / "digitizer_22577_100.csv", _values(seed=2))
    np.savetxt(root / "101" / "digitizer_17592_101.csv", _values(), delimiter=",")
    (root / "_geometry").mkdir()
    return root


def test_cli_packs_deletes_only_verified_csvs_and_records_provenance(tmp_path, capsys):
    root = _archive(tmp_path)
    digest = dh._sha256_bytes((root / "100" / "digitizer_17592_100.csv").read_bytes())

    assert sxr_pack.main(["--root", str(root), "--delete-csv"]) == 1  # one refusal

    assert sorted(p.name for p in (root / "100").iterdir()) == [
        "digitizer_17592_100.h5", "digitizer_22577_100.h5", "provenance.json"]
    assert (root / "101" / "digitizer_17592_101.csv").exists()
    assert not (root / "101" / "digitizer_17592_101.h5").exists()
    provenance = json.loads((root / "100" / "provenance.json").read_text())
    assert provenance["shot"] == 100
    assert provenance["containers"]["digitizer_17592_100.csv"]["source_sha256"] == digest
    log = [json.loads(line) for line in (root / "_sxr_pack_log.jsonl").read_text().splitlines()]
    assert sorted(r["status"] for r in log) == ["failed", "packed", "packed"]
    assert "FAILED 101" in capsys.readouterr().err


def test_cli_resumes_by_verifying_an_existing_container(tmp_path):
    root = _archive(tmp_path)
    csv = root / "100" / "digitizer_17592_100.csv"
    dh.pack_digitizer_csv(csv)  # a run that died before deleting the CSV

    sxr_pack.main(["--root", str(root), "--first-shot", "100", "--last-shot", "100",
                   "--delete-csv"])

    log = [json.loads(line) for line in (root / "_sxr_pack_log.jsonl").read_text().splitlines()]
    assert {r["csv"]: r["status"] for r in log} == {
        "digitizer_17592_100.csv": "verified_existing",
        "digitizer_22577_100.csv": "packed",
    }
    assert not csv.exists()


def test_cli_refuses_to_delete_a_csv_its_container_does_not_match(tmp_path):
    root = _archive(tmp_path)
    csv = root / "100" / "digitizer_17592_100.csv"
    dh.pack_digitizer_csv(csv)
    _write_sample_v3(csv, _values(seed=99))  # the CSV changed after packing

    sxr_pack.main(["--root", str(root), "--last-shot", "100", "--delete-csv"])

    assert csv.exists()
    log = [json.loads(line) for line in (root / "_sxr_pack_log.jsonl").read_text().splitlines()]
    assert {r["csv"]: r["status"] for r in log}["digitizer_17592_100.csv"] == "failed"


def test_mapper_reads_a_csv_rewritten_after_its_container(tmp_path):
    import os

    csv = tmp_path / "digitizer_17592_12345.csv"
    _write_sample_v3(csv, _values(channels=40, samples=16, seed=1))
    target = dh.pack_digitizer_csv(csv).target
    fresh = _values(channels=40, samples=16, seed=2)
    _write_sample_v3(csv, fresh)
    stamp = target.stat().st_mtime
    os.utime(csv, (stamp + 10, stamp + 10))

    with pytest.warns(RuntimeWarning, match="newer than its container"):
        found = _discover_digitizer_files(12345, tmp_path)
    assert found == [("17592", csv)]
    assert np.array_equal(load_digitizer_csv(found[0][1]), fresh.T)


def test_cli_parallel_run_matches_serial(tmp_path):
    serial, parallel = _archive(tmp_path / "a"), _archive(tmp_path / "b")
    assert sxr_pack.main(["--root", str(serial)]) == 1
    assert sxr_pack.main(["--root", str(parallel), "--jobs", "2"]) == 1
    for root in (serial, parallel):
        assert (root / "100" / "digitizer_17592_100.h5").exists()
        assert (root / "100" / "digitizer_22577_100.h5").exists()
        assert not (root / "101" / "digitizer_17592_101.h5").exists()
    assert np.array_equal(
        dh.load_digitizer_hdf5(serial / "100" / "digitizer_22577_100.h5"),
        dh.load_digitizer_hdf5(parallel / "100" / "digitizer_22577_100.h5"),
    )


def test_cli_dry_run_writes_nothing(tmp_path):
    root = _archive(tmp_path)
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert sxr_pack.main(["--root", str(root), "--dry-run"]) == 0
    after = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert after == before
