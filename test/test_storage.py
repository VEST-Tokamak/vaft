"""Shared cache root and namespaces (#2317): precedence, compatibility, ownership."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import threading

import pytest

from vaft import storage
from vaft.cli import cache as cache_cli
from vaft.data.public import _fetch
from vaft.database import hsds_cache


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in (
        "VAFT_CACHE_DIR",
        "VAFT_PUBLIC_DATA_DIR",
        "VAFT_FILEDB_DIR",
        "XDG_CACHE_HOME",
        "LOCALAPPDATA",
        "VAFT_HSDS_CACHE_MAX",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def _run(*argv):
    out = io.StringIO()
    code = cache_cli.main(list(argv), out=out)
    return code, out.getvalue()


# -- precedence --------------------------------------------------------------
def test_hsds_precedence_is_unchanged(isolated_home, tmp_path, monkeypatch):
    assert hsds_cache.cache_base() == isolated_home / ".cache" / "vaft" / "hsds"
    monkeypatch.setenv("VAFT_CACHE_DIR", str(tmp_path / "vol"))
    assert hsds_cache.cache_base() == tmp_path / "vol" / "hsds"
    assert hsds_cache.cache_base(tmp_path / "mine") == tmp_path / "mine"
    monkeypatch.setenv("VAFT_CACHE_DIR", "")
    assert hsds_cache.cache_base() == isolated_home / ".cache" / "vaft" / "hsds"


def test_public_precedence(isolated_home, tmp_path, monkeypatch):
    platform_default = storage.default_cache_root() / "public"
    assert _fetch.cache_dir() == platform_default
    monkeypatch.setenv("VAFT_CACHE_DIR", str(tmp_path / "vol"))
    assert _fetch.cache_dir() == tmp_path / "vol" / "public"
    monkeypatch.setenv("VAFT_PUBLIC_DATA_DIR", str(tmp_path / "pub"))
    assert _fetch.cache_dir() == tmp_path / "pub"
    assert _fetch.cache_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_existing_public_cache_is_not_relocated(isolated_home, tmp_path, monkeypatch):
    legacy = storage.default_cache_root() / "public"
    legacy.mkdir(parents=True)
    monkeypatch.setenv("VAFT_CACHE_DIR", str(tmp_path / "vol"))
    # The legacy cache keeps being used until the shared location exists.
    assert _fetch.cache_dir() == legacy
    (tmp_path / "vol" / "public").mkdir(parents=True)
    assert _fetch.cache_dir() == tmp_path / "vol" / "public"


def test_new_namespaces_use_the_shared_root(isolated_home, tmp_path, monkeypatch):
    root = storage.default_cache_root()
    assert storage.namespace_path("references/samples") == root / "references" / "samples"
    monkeypatch.setenv("VAFT_CACHE_DIR", str(tmp_path / "vol"))
    assert storage.namespace_path("generated/efund") == tmp_path / "vol" / "generated" / "efund"
    assert storage.namespace_path("generated/efund", tmp_path / "x") == tmp_path / "x"
    assert storage.cache_path("references", "samples") == tmp_path / "vol" / "references" / "samples"


@pytest.mark.parametrize(
    ("system", "platform", "env", "expected_parts"),
    [
        ("posix", "linux", {}, (".cache", "vaft")),
        ("posix", "linux", {"XDG_CACHE_HOME": "XDG"}, ("XDG", "vaft")),
        ("posix", "darwin", {}, ("Library", "Caches", "vaft")),
        ("nt", "win32", {}, ("AppData", "Local", "vaft")),
        ("nt", "win32", {"LOCALAPPDATA": "LAD"}, ("LAD", "vaft")),
    ],
)
def test_platform_roots(isolated_home, tmp_path, monkeypatch, system, platform, env, expected_parts):
    for key, value in env.items():
        monkeypatch.setenv(key, str(tmp_path / value))
    root = storage._platform_cache_root(system, platform, isolated_home)
    base = tmp_path if env else isolated_home
    assert root == base.joinpath(*expected_parts)


def test_empty_xdg_is_unset(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    root = storage._platform_cache_root("posix", "linux", isolated_home)
    assert root == isolated_home / ".cache" / "vaft"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "C:", "x\0y"])
def test_cache_path_rejects_traversal(bad):
    with pytest.raises(ValueError):
        storage.cache_path("references", bad)


def test_resolution_has_no_side_effects(isolated_home, tmp_path, monkeypatch):
    monkeypatch.setenv("VAFT_CACHE_DIR", str(tmp_path / "vol"))
    for name in storage.NAMESPACES:
        storage.namespace_path(name)
        storage.namespace_report(name)
    hsds_cache.cache_base()
    _fetch.cache_dir()
    assert not (tmp_path / "vol").exists()
    assert not (isolated_home / ".cache").exists()


def test_unknown_namespace():
    with pytest.raises(ValueError, match="unknown cache namespace"):
        storage.namespace_path("everything")


# -- entries and ownership --------------------------------------------------
def _publish(name, key, content=b"x", **kwargs):
    def populate(directory: Path) -> None:
        (directory / "table.bin").write_bytes(content)

    return storage.publish_entry(name, key, populate, record={"sha256": "abc"}, **kwargs)


def test_publish_is_atomic_and_listed(tmp_path):
    root = tmp_path / "gen"
    path = _publish("generated/efund", "id1", override=root)
    assert path == root / "id1"
    assert json.loads((path / storage.ENTRY_RECORD).read_text())["sha256"] == "abc"
    assert [e.key for e in storage.entries("generated/efund", root)] == ["id1"]
    assert storage.entry_path("generated/efund", "id1", root) == path


def test_failed_publish_leaves_nothing(tmp_path):
    root = tmp_path / "gen"

    def populate(directory: Path) -> None:
        (directory / "half.bin").write_bytes(b"partial")
        raise RuntimeError("verification failed")

    with pytest.raises(RuntimeError):
        storage.publish_entry("generated/efund", "id1", populate, override=root)
    assert storage.entry_path("generated/efund", "id1", root) is None
    assert [p.name for p in root.iterdir() if not p.name.startswith(".vaft") and p.name != ".lock"] == []


def test_concurrent_publish_keeps_one_complete_entry(tmp_path):
    root = tmp_path / "gen"
    results = []

    def worker(tag: bytes) -> None:
        results.append(_publish("generated/efund", "same", content=tag, override=root))

    threads = [threading.Thread(target=worker, args=(bytes([i]),)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 4  # every thread returned; none raised
    assert {str(p) for p in results} == {str(root / "same")}
    assert len(storage.entries("generated/efund", root)) == 1


def test_removal_requires_marker_and_skips_unknown(tmp_path):
    root = tmp_path / "gen"
    root.mkdir()
    (root / "user_file.txt").write_text("mine")
    with pytest.raises(ValueError, match="not removing anything"):
        storage.remove_entries("generated/efund", everything=True, override=root)
    _publish("generated/efund", "id1", override=root)
    result = storage.remove_entries("generated/efund", everything=True, override=root)
    assert result.removed == ["id1"]
    assert result.skipped == ["user_file.txt"]
    assert (root / "user_file.txt").read_text() == "mine"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_symlinked_entry_is_never_followed(tmp_path):
    root = tmp_path / "gen"
    _publish("generated/efund", "real", override=root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / storage.ENTRY_RECORD).write_text("{}")
    (outside / "precious.dat").write_text("keep")
    (root / "link").symlink_to(outside, target_is_directory=True)
    result = storage.remove_entries("generated/efund", everything=True, override=root)
    assert result.removed == ["real"]
    assert "link" in result.skipped
    assert (outside / "precious.dat").read_text() == "keep"


def test_dry_run_and_older_than(tmp_path):
    root = tmp_path / "gen"
    old = _publish("generated/efund", "old", override=root)
    _publish("generated/efund", "new", override=root)
    os.utime(old / storage.ENTRY_RECORD, (1, 1))
    preview = storage.remove_entries("generated/efund", older_than=3600, dry_run=True, override=root)
    assert preview.removed == ["old"] and old.exists()
    storage.remove_entries("generated/efund", older_than=3600, override=root)
    assert [e.key for e in storage.entries("generated/efund", root)] == ["new"]


def test_removal_needs_a_selection(tmp_path):
    with pytest.raises(ValueError, match="say what to remove"):
        storage.remove_entries("generated/efund", override=tmp_path)


def test_filedb_inside_namespace_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "gen"
    _publish("generated/efund", "id1", override=root)
    monkeypatch.setenv("VAFT_FILEDB_DIR", str(root / "id1"))
    with pytest.raises(ValueError, match="FileDB root"):
        storage.remove_entries("generated/efund", everything=True, override=root)


def test_hsds_is_not_managed_here():
    with pytest.raises(ValueError, match="hsds_cache"):
        storage.entries("hsds")


# -- CLI ---------------------------------------------------------------------
def test_cli_default_info_is_unchanged(isolated_home):
    code, text = _run("info", "--json")
    report = json.loads(text)
    assert code == 0
    assert "namespaces" not in report
    assert report["location"] == str(isolated_home / ".cache" / "vaft" / "hsds")


def test_cli_info_all_lists_every_namespace(isolated_home):
    code, text = _run("info", "--all", "--json")
    names = [r["namespace"] for r in json.loads(text)["namespaces"]]
    assert code == 0
    assert names == list(storage.NAMESPACES)


def test_cli_generated_prune_and_clear(isolated_home, tmp_path, monkeypatch):
    monkeypatch.setenv("VAFT_CACHE_DIR", str(tmp_path / "vol"))
    _publish("generated/efund", "id1")
    assert _run("prune", "--type", "generated")[0] == 2  # no cap: needs --older-than
    assert _run("prune", "--type", "generated", "--max", "1G")[0] == 2
    assert _run("clear", "--type", "generated")[0] == 2  # needs --yes or --entry
    code, text = _run("list", "--type", "generated")
    assert code == 0 and "id1" in text
    code, text = _run("clear", "--type", "generated", "--entry", "id1", "--dry-run")
    assert "Would remove 1" in text
    assert storage.entry_path("generated/efund", "id1") is not None
    code, text = _run("clear", "--type", "generated", "--yes")
    assert code == 0 and storage.entry_path("generated/efund", "id1") is None


def test_cli_all_is_never_a_removal_scope(isolated_home):
    with pytest.raises(SystemExit):
        _run("clear", "--type", "all", "--yes")
    with pytest.raises(SystemExit):
        _run("prune", "--type", "all", "--older-than", "1d")


def test_cli_hsds_only_flags_are_refused_elsewhere(isolated_home):
    assert _run("list", "--type", "public", "--shot", "39915")[0] == 2
    assert _run("clear", "--entry", "x")[0] == 2


def test_public_removal_only_touches_verified_registry_files(isolated_home, tmp_path, monkeypatch):
    public = tmp_path / "pub"
    public.mkdir()
    monkeypatch.setenv("VAFT_PUBLIC_DATA_DIR", str(public))
    key, source = next(iter(sorted(_fetch.SOURCES.items())))
    (public / source.filename).write_bytes(b"not the real bytes")
    (public / "pr08").mkdir()
    (public / "pr08" / "user.dat").write_text("keep")
    code, text = _run("clear", "--type", "public", "--yes")
    assert code == 0
    assert "checksum mismatch" in text
    assert (public / source.filename).exists()  # mismatching file left for inspection
    monkeypatch.setattr(_fetch, "sha256_of", lambda path: source.sha256.lower())
    code, text = _run("clear", "--type", "public", "--entry", key)
    assert code == 0 and not (public / source.filename).exists()
    assert (public / "pr08" / "user.dat").read_text() == "keep"
