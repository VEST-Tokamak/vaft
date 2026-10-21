"""Documentation media kept outside git, pinned by SHA-256 (#2117).

No network: the release is stood in for by a ``file://`` directory through the
lock's ``base_url``.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "scripts" / "media.py"
BUILD = ROOT / "docs" / "build.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def media():
    return _load("vaft_docs_media_test", MEDIA)


@pytest.fixture(scope="module")
def build():
    return _load("vaft_docs_build_media_test", BUILD)


@pytest.fixture
def release(tmp_path):
    """A docs tree with a lock, and a 'release' directory serving the assets."""
    docs, served = tmp_path / "docs", tmp_path / "release"
    docs.mkdir()
    served.mkdir()
    files = {}
    for path, data in {
        "assets/demo/a.gif": b"GIF89a-demo",
        "assets/demo/a.gif.json": b'{"t":[0,1]}\n',
    }.items():
        sha256 = hashlib.sha256(data).hexdigest()
        asset = f"{sha256[:16]}-{Path(path).name}"
        (served / asset).write_bytes(data)
        files[path] = {"sha256": sha256, "bytes": len(data), "asset": asset}
    lock = {
        "schema_version": 1,
        "repository": "example/repo",
        "release": "docs-media",
        "base_url": served.as_uri() + "/{asset}",
        "include": ["assets/demo/*.gif", "assets/demo/*.gif.json"],
        "files": files,
    }
    (docs / "media.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return docs, served, lock


def test_the_committed_lock_is_well_formed(media):
    lock = media.load_lock(ROOT / "docs")
    assert lock is not None and lock["release"] == "docs-media" and "base_url" not in lock
    for path, entry in lock["files"].items():
        assert any(Path(path).match(p) for p in lock["include"]), path
        assert entry["asset"] == media.asset_name(entry["sha256"], path)
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0


def test_fetch_places_every_locked_file_verified(media, release, tmp_path):
    docs, _, lock = release
    placed = media.materialize(docs, tmp_path / "cache")
    assert sorted(placed) == sorted(lock["files"])
    assert media.verify(docs) == []
    assert media.materialize(docs, tmp_path / "cache") == [], "present files are left alone"


def test_a_digest_mismatch_fails_and_places_nothing(media, release, tmp_path):
    docs, served, lock = release
    asset = lock["files"]["assets/demo/a.gif"]["asset"]
    (served / asset).write_bytes(b"GIF89a-tampered")
    with pytest.raises(media.MediaError, match="sha256 mismatch"):
        media.materialize(docs, tmp_path / "cache")
    assert not (docs / "assets/demo/a.gif").exists()
    assert not any((tmp_path / "cache").glob(lock["files"]["assets/demo/a.gif"]["sha256"]))


def test_offline_with_a_cold_cache_names_the_files_and_the_fix(media, release, tmp_path):
    docs, _, lock = release
    with pytest.raises(media.MediaError) as failure:
        media.materialize(docs, tmp_path / "cold", offline=True)
    message = str(failure.value)
    assert all(path in message for path in lock["files"])
    assert "python docs/scripts/media.py fetch" in message


def test_offline_with_a_warm_cache_needs_no_network(media, release, tmp_path):
    docs, served, _ = release
    media.materialize(docs, tmp_path / "cache")
    for path in docs.rglob("*.gif*"):
        path.unlink()
    for asset in served.iterdir():
        asset.unlink()  # the "release" is gone; only the cache can supply them
    assert len(media.materialize(docs, tmp_path / "cache", offline=True)) == 2


def test_an_unreachable_release_is_a_clear_error(media, release, tmp_path):
    docs, served, _ = release
    for asset in served.iterdir():
        asset.unlink()
    with pytest.raises(media.MediaError, match="cannot fetch"):
        media.materialize(docs, tmp_path / "cache")


def test_a_tree_without_a_lock_has_no_media(media, tmp_path):
    assert media.load_lock(tmp_path) is None
    assert media.materialize(tmp_path) == []


def test_relock_pins_exactly_the_included_files(media, release):
    docs, _, lock = release
    media.materialize(docs, docs.parent / "cache")
    (docs / "assets/demo/a.gif").write_bytes(b"GIF89a-regenerated")
    (docs / "assets/demo/a.png").write_bytes(b"not included")
    relocked = media.relock(docs)
    assert set(relocked["files"]) == set(lock["files"])
    entry = relocked["files"]["assets/demo/a.gif"]
    assert entry["sha256"] == hashlib.sha256(b"GIF89a-regenerated").hexdigest()
    assert entry["asset"].endswith("-a.gif") and entry["asset"] != lock["files"]["assets/demo/a.gif"]["asset"]


def test_the_build_turns_a_media_failure_into_a_build_error(build, release, tmp_path):
    docs, served, _ = release
    track = build.Track(spec=build.TRACKS["development"], ref="x", commit="b" * 40,
                        commit_date="2026-01-01", root=docs.parent)
    for asset in served.iterdir():
        asset.unlink()
    with pytest.raises(build.BuildError, match="development: documentation media"):
        build.materialize_media(track, tmp_path / "cache")


def test_composed_validation_checks_locked_media(build, release, tmp_path):
    docs, _, lock = release
    track = build.Track(spec=build.TRACKS["development"], ref="x", commit="b" * 40,
                        commit_date="2026-01-01", root=docs.parent)
    build.materialize_media(track, tmp_path / "cache")
    tree = tmp_path / "composed" / "develop"
    for path in lock["files"]:
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_bytes((docs / path).read_bytes())
    (tree / "assets/demo/a.gif").write_bytes(b"GIF89a-other")
    (tree / "assets/demo/a.gif.json").unlink()
    with pytest.raises(build.BuildError) as failure:
        build.validate_composed(tmp_path / "composed", [track])
    assert "locked media assets/demo/a.gif does not match its sha256" in str(failure.value)
    assert "locked media assets/demo/a.gif.json is not in the composed tree" in str(failure.value)
