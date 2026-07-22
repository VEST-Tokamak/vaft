"""Unit tests for selective eager HSDS staging and its disk cache."""

from __future__ import annotations

from pathlib import Path
import shutil

import h5py
import pytest

from vaft.database import staging


def _make_remote_shot(root: Path) -> None:
    for name in ("equilibrium.h5", "magnetics.h5", "dataset_description.h5"):
        with h5py.File(root / name, "w") as handle:
            handle.attrs["HDF5_BACKEND_VERSION"] = "1.0"
            handle.create_dataset("payload", data=[1, 2, 3])
    with h5py.File(root / "master.h5", "w") as master:
        master["equilibrium"] = h5py.ExternalLink("equilibrium.h5", "/equilibrium")
        master["magnetics"] = h5py.ExternalLink("magnetics.h5", "/magnetics")
        master["dataset_description"] = h5py.ExternalLink(
            "dataset_description.h5", "/dataset_description"
        )
        master["ordinary_metadata"] = h5py.SoftLink("/")


@pytest.fixture
def fake_hsds(monkeypatch, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    _make_remote_shot(remote)
    calls: list[str] = []
    revisions = {name: "revision-1" for name in ("master.h5", "equilibrium.h5", "magnetics.h5", "dataset_description.h5")}

    def fake_hsget(uri: str, target: Path) -> Path:
        name = uri.rsplit("/", 1)[-1]
        calls.append(name)
        shutil.copy2(remote / name, target)
        return target

    def remote_info(_self, uri: str):
        name = uri.rsplit("/", 1)[-1]
        return {
            "remote_uri": uri,
            "last_modified": revisions[name],
            "backend_version": "1.0",
        }

    monkeypatch.setattr(staging, "run_hsget", fake_hsget)
    monkeypatch.setattr(staging, "ensure_imas_hdf5_userblock", lambda *_: None)
    monkeypatch.setattr(staging.HSDSDomainCache, "_endpoint", staticmethod(lambda: "test-endpoint"))
    monkeypatch.setattr(staging.HSDSDomainCache, "_remote_info", remote_info)
    return remote, calls, revisions


def _external_names(master: Path) -> set[str]:
    with h5py.File(master, "r") as handle:
        return {
            Path(link.filename).name
            for name in handle
            if isinstance((link := handle.get(name, getlink=True)), h5py.ExternalLink)
        }


def test_selective_staging_only_materializes_requested_ids_and_partial_master(fake_hsds, tmp_path):
    _remote, calls, _revisions = fake_hsds
    output = tmp_path / "stage"

    plan = staging.stage_imas_shot(
        "public", 123, output, requested_ids=("equilibrium",), cache="off"
    )

    assert calls == ["master.h5", "dataset_description.h5", "equilibrium.h5"]
    assert plan["files"] == ["master.h5", "dataset_description.h5", "equilibrium.h5"]
    assert _external_names(output / "master.h5") == {"equilibrium.h5", "dataset_description.h5"}
    assert not (output / "magnetics.h5").exists()


def test_full_staging_retains_historical_all_linked_behavior(fake_hsds, tmp_path):
    _remote, calls, _revisions = fake_hsds
    output = tmp_path / "stage"

    staging.stage_imas_shot("public", 123, output, requested_ids=None, cache="off")

    assert calls == ["master.h5", "dataset_description.h5", "equilibrium.h5", "magnetics.h5"]
    assert _external_names(output / "master.h5") == {
        "equilibrium.h5", "magnetics.h5", "dataset_description.h5"
    }


def test_cache_hit_avoids_hsget_and_revision_change_redownloads(fake_hsds, tmp_path):
    _remote, calls, revisions = fake_hsds
    cache = tmp_path / "cache"

    staging.stage_imas_shot("public", 123, tmp_path / "cold", requested_ids=("equilibrium",), cache=cache)
    assert calls == ["master.h5", "dataset_description.h5", "equilibrium.h5"]

    calls.clear()
    warm = staging.stage_imas_shot(
        "public", 123, tmp_path / "warm", requested_ids=("equilibrium",), cache=cache
    )
    assert calls == []
    assert all(warm["cache_hits"].values())

    revisions["equilibrium.h5"] = "revision-2"
    calls.clear()
    staging.stage_imas_shot("public", 123, tmp_path / "changed", requested_ids=("equilibrium",), cache=cache)
    assert calls == ["equilibrium.h5"]


def test_corrupt_cached_domain_is_not_reused(fake_hsds, tmp_path):
    _remote, calls, _revisions = fake_hsds
    cache = tmp_path / "cache"
    staging.stage_imas_shot("public", 123, tmp_path / "cold", requested_ids=("equilibrium",), cache=cache)
    cached = next(cache.rglob("equilibrium.h5"))
    cached.write_bytes(b"not an hdf5 file")

    calls.clear()
    staging.stage_imas_shot("public", 123, tmp_path / "corrupt", requested_ids=("equilibrium",), cache=cache)
    assert calls == ["equilibrium.h5"]


def test_missing_ids_and_invalid_paths_have_actionable_errors(fake_hsds, tmp_path):
    _remote, _calls, _revisions = fake_hsds
    with pytest.raises(FileNotFoundError, match="Available IDS:.*equilibrium"):
        staging.stage_imas_shot("public", 123, tmp_path / "stage", requested_ids=("not_saved",), cache="off")
    with pytest.raises(ValueError, match="non-empty strings"):
        staging.requested_ids_from_paths([None])
