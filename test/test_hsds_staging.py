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
    revisions = {
        name: "revision-1"
        for name in (
            "master.h5",
            "equilibrium.h5",
            "magnetics.h5",
            "dataset_description.h5",
        )
    }

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
    monkeypatch.setattr(
        staging.HSDSDomainCache, "_endpoint", staticmethod(lambda: "test-endpoint")
    )
    monkeypatch.setattr(staging.HSDSDomainCache, "_remote_info", remote_info)
    monkeypatch.setattr(
        staging,
        "materialize_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging.H5ImageUnavailableError("no derived fixture")
        ),
    )
    return remote, calls, revisions


def _external_names(master: Path) -> set[str]:
    with h5py.File(master, "r") as handle:
        return {
            Path(link.filename).name
            for name in handle
            if isinstance((link := handle.get(name, getlink=True)), h5py.ExternalLink)
        }


def test_selective_staging_only_materializes_requested_ids_and_partial_master(
    fake_hsds, tmp_path
):
    _remote, calls, _revisions = fake_hsds
    output = tmp_path / "stage"

    plan = staging.stage_imas_shot(
        "public", 123, output, requested_ids=("equilibrium",), cache="off"
    )

    assert calls[0] == "master.h5"  # IDS domains follow concurrently (#1870)
    assert sorted(calls[1:]) == ["dataset_description.h5", "equilibrium.h5"]
    assert plan["files"] == ["master.h5", "dataset_description.h5", "equilibrium.h5"]
    assert _external_names(output / "master.h5") == {
        "equilibrium.h5",
        "dataset_description.h5",
    }
    assert not (output / "magnetics.h5").exists()


def test_full_staging_retains_historical_all_linked_behavior(fake_hsds, tmp_path):
    _remote, calls, _revisions = fake_hsds
    output = tmp_path / "stage"

    staging.stage_imas_shot("public", 123, output, requested_ids=None, cache="off")

    assert calls[0] == "master.h5"  # IDS domains follow concurrently (#1870)
    assert sorted(calls[1:]) == ["dataset_description.h5", "equilibrium.h5", "magnetics.h5"]
    assert _external_names(output / "master.h5") == {
        "equilibrium.h5",
        "magnetics.h5",
        "dataset_description.h5",
    }


def test_cache_hit_avoids_hsget_and_revision_change_redownloads(fake_hsds, tmp_path):
    _remote, calls, revisions = fake_hsds
    cache = tmp_path / "cache"

    staging.stage_imas_shot(
        "public", 123, tmp_path / "cold", requested_ids=("equilibrium",), cache=cache
    )
    # The master first; its IDS domains are fetched concurrently (#1870).
    assert calls[0] == "master.h5"
    assert sorted(calls[1:]) == ["dataset_description.h5", "equilibrium.h5"]

    calls.clear()
    warm = staging.stage_imas_shot(
        "public", 123, tmp_path / "warm", requested_ids=("equilibrium",), cache=cache
    )
    assert calls == []
    assert all(warm["cache_hits"].values())

    revisions["equilibrium.h5"] = "revision-2"
    calls.clear()
    staging.stage_imas_shot(
        "public", 123, tmp_path / "changed", requested_ids=("equilibrium",), cache=cache
    )
    assert calls == ["equilibrium.h5"]


def test_corrupt_cached_domain_is_not_reused(fake_hsds, tmp_path):
    _remote, calls, _revisions = fake_hsds
    cache = tmp_path / "cache"
    staging.stage_imas_shot(
        "public", 123, tmp_path / "cold", requested_ids=("equilibrium",), cache=cache
    )
    cached = next(cache.rglob("equilibrium.h5"))
    cached.write_bytes(b"not an hdf5 file")

    calls.clear()
    staging.stage_imas_shot(
        "public", 123, tmp_path / "corrupt", requested_ids=("equilibrium",), cache=cache
    )
    assert calls == ["equilibrium.h5"]


def test_missing_ids_and_invalid_paths_have_actionable_errors(fake_hsds, tmp_path):
    _remote, _calls, _revisions = fake_hsds
    with pytest.raises(FileNotFoundError, match="Available IDS:.*equilibrium"):
        staging.stage_imas_shot(
            "public", 123, tmp_path / "stage", requested_ids=("not_saved",), cache="off"
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        staging.requested_ids_from_paths([None])


def test_invalid_transport_is_rejected_before_cache_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        staging.HSDSDomainCache,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("cache must not be constructed"),
    )
    with pytest.raises(ValueError, match="transport"):
        staging.stage_imas_shot(
            "public",
            123,
            tmp_path / "stage",
            requested_ids=None,
            transport="invalid",
        )


def test_requested_ids_generator_is_reported_after_staging(fake_hsds, tmp_path):
    plan = staging.stage_imas_shot(
        "public",
        123,
        tmp_path / "generator",
        requested_ids=(name for name in ("equilibrium",)),
        cache="off",
    )
    assert plan["selected_ids"] == ["equilibrium"]


def test_h5image_transport_avoids_hsget(fake_hsds, monkeypatch, tmp_path):
    remote, calls, _revisions = fake_hsds

    def restore(_directory, _shot, filename, target, *, source_info):
        assert source_info["last_modified"] == "revision-1"
        shutil.copy2(remote / filename, target)
        return {"transport": "h5image", "logical_bytes": target.stat().st_size}

    monkeypatch.setattr(staging, "materialize_image", restore)
    plan = staging.stage_imas_shot(
        "public",
        123,
        tmp_path / "stage-h5image",
        requested_ids=("equilibrium",),
        cache="off",
        transport="h5image",
    )

    assert calls == []
    assert set(plan["transports"].values()) == {"h5image"}


def test_strict_h5image_transport_does_not_fallback(fake_hsds, tmp_path):
    _remote, calls, _revisions = fake_hsds
    with pytest.raises(staging.H5ImageUnavailableError, match="no derived fixture"):
        staging.stage_imas_shot(
            "public",
            123,
            tmp_path / "strict-missing",
            requested_ids=("equilibrium",),
            cache="off",
            transport="h5image",
        )
    assert calls == []


def test_domains_are_fetched_concurrently_and_all_reach_the_cache_manifest(fake_hsds, monkeypatch, tmp_path):
    """#1870: the IDS domains were fetched one after another -- on a warm cache
    eleven revision probes in a row, on a cold one every hsget in turn. They now
    overlap; the shared manifest still records every domain, so the next load is
    all cache hits."""
    import threading
    import time

    _remote, calls, _revisions = fake_hsds
    original = staging.run_hsget
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def slow_hsget(uri, target):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        try:
            return original(uri, target)
        finally:
            with lock:
                state["now"] -= 1

    monkeypatch.setattr(staging, "run_hsget", slow_hsget)
    cache = tmp_path / "cache"
    first = staging.stage_imas_shot("public", 123, tmp_path / "a", requested_ids=None, cache=cache)
    assert state["peak"] > 1
    assert sorted(first["cache_hits"]) == sorted(["master.h5", "equilibrium.h5", "magnetics.h5", "dataset_description.h5"])

    calls.clear()
    second = staging.stage_imas_shot("public", 123, tmp_path / "b", requested_ids=None, cache=cache)
    assert calls == [] and all(second["cache_hits"].values())


def test_one_domain_worker_restores_the_sequential_fetch(fake_hsds, monkeypatch, tmp_path):
    monkeypatch.setenv("VAFT_HSDS_DOMAIN_WORKERS", "1")
    _remote, calls, _revisions = fake_hsds
    plan = staging.stage_imas_shot("public", 123, tmp_path / "s", requested_ids=None, cache="off")
    assert calls[0] == "master.h5" and sorted(calls[1:]) == ["dataset_description.h5", "equilibrium.h5", "magnetics.h5"]
    assert plan["files"][0] == "master.h5"
    assert staging.domain_workers() == 1
