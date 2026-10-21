"""The local HSDS domain cache is bounded and manageable (#1759, #964).

Offline: HSDS is replaced by a fake that writes small HDF5 files, and every
cache lives under ``tmp_path``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import h5py
import numpy as np
import pytest

from vaft.cli import cache as cache_cli
from vaft.database import hsds_cache, staging

KiB = 1024


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No real cache, config file or environment setting leaks into a test."""
    monkeypatch.delenv(hsds_cache.CACHE_MAX_ENV, raising=False)
    monkeypatch.delenv(hsds_cache.CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # never the real cache
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    hsds_cache._reported.clear()


# -- sizes and precedence ----------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (0, 0),
        ("0", 0),
        (123, 123),
        ("500M", 500 * 1024**2),
        ("20G", 20 * 1024**3),
        ("1GiB", 1024**3),
        ("1.5 gb", int(1.5 * 1024**3)),
        ("64k", 64 * 1024),
        ("none", None),
        ("Unlimited", None),
    ],
)
def test_sizes_parse(text, expected):
    assert hsds_cache.parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "lots", "-5", "5X", "5iB", True])
def test_bad_sizes_are_refused(text):
    with pytest.raises(ValueError):
        hsds_cache.parse_size(text)


def test_ages_parse():
    assert hsds_cache.parse_age("30d") == 30 * 86400
    assert hsds_cache.parse_age("12h") == 12 * 3600
    with pytest.raises(ValueError):
        hsds_cache.parse_age("soon")


def test_the_default_cap_is_one_gibibyte():
    limit = hsds_cache.resolve_cache_max()
    assert (limit.bytes, limit.source) == (1024**3, "default")


def test_argument_beats_environment_beats_config_beats_default(monkeypatch):
    path = hsds_cache.set_cache_max("2G")
    assert path == hsds_cache.config_path()
    limit = hsds_cache.resolve_cache_max()
    assert limit.bytes == 2 * 1024**3 and str(path) in limit.source

    monkeypatch.setenv(hsds_cache.CACHE_MAX_ENV, "500M")
    limit = hsds_cache.resolve_cache_max()
    assert (limit.bytes, limit.source) == (500 * 1024**2, "$VAFT_HSDS_CACHE_MAX")

    limit = hsds_cache.resolve_cache_max("unlimited")
    assert (limit.bytes, limit.source) == (None, "argument")


def test_a_bad_environment_value_names_the_variable(monkeypatch):
    monkeypatch.setenv(hsds_cache.CACHE_MAX_ENV, "plenty")
    with pytest.raises(ValueError, match="VAFT_HSDS_CACHE_MAX"):
        hsds_cache.resolve_cache_max()


def test_set_limit_keeps_the_rest_of_the_config_file():
    path = hsds_cache.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        '# my settings\n[plot]\nstyle = "paper"\n\n[hsds]\ncache_max = "1G"  # old\nother = 3\n',
        encoding="utf-8",
    )
    hsds_cache.set_cache_max("300M")
    text = path.read_text(encoding="utf-8")
    assert "# my settings" in text and 'style = "paper"' in text and "other = 3" in text
    assert 'cache_max = "300M"' in text and '"1G"' not in text
    assert hsds_cache.resolve_cache_max().bytes == 300 * 1024**2
    assert hsds_cache._scan_config(text) == "300M"  # the Python 3.10 fallback reader


def test_set_limit_adds_the_table_when_missing():
    hsds_cache.set_cache_max(0)
    assert hsds_cache.resolve_cache_max().bytes == 0


@pytest.mark.parametrize(
    "layout",
    [
        'hsds.endpoint = "x"\n',
        'hsds = { endpoint = "x" }\n',
        '[hsds]\n"cache_max" = 5\n',
        '[hsds]\nnote = """\ncache_max = 1\n"""\n',
    ],
)
def test_set_limit_refuses_a_layout_it_cannot_edit(layout):
    pytest.importorskip("tomllib")
    path = hsds_cache.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(layout, encoding="utf-8")
    with pytest.raises(ValueError, match="edit the file by hand"):
        hsds_cache.set_cache_max("2G")
    assert path.read_text(encoding="utf-8") == layout


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks and modes")
def test_set_limit_writes_through_a_symlink_and_keeps_the_mode(tmp_path):
    real = tmp_path / "dotfiles" / "vaft.toml"
    real.parent.mkdir()
    real.write_text("[plot]\n", encoding="utf-8")
    real.chmod(0o600)
    link = hsds_cache.config_path()
    link.parent.mkdir(parents=True)
    link.symlink_to(real)
    hsds_cache.set_cache_max("2G")
    assert link.is_symlink() and "cache_max" in real.read_text(encoding="utf-8")
    assert real.stat().st_mode & 0o777 == 0o600


def test_a_broken_setting_warns_and_loads_with_the_default(monkeypatch, caplog):
    monkeypatch.setenv(hsds_cache.CACHE_MAX_ENV, "plenty")
    with caplog.at_level("WARNING", logger="vaft.database.hsds_cache"):
        assert staging.HSDSDomainCache._resolve_cap(None) == hsds_cache.DEFAULT_CACHE_MAX
    assert "VAFT_HSDS_CACHE_MAX" in caplog.text
    with pytest.raises(ValueError):  # the caller's own argument is still checked
        staging.HSDSDomainCache._resolve_cap("plenty")


def test_set_limit_refuses_a_bad_value_without_writing():
    with pytest.raises(ValueError):
        hsds_cache.set_cache_max("big")
    assert not hsds_cache.config_path().exists()


def test_the_cache_location_is_relocatable(monkeypatch, tmp_path):
    # XDG_CACHE_HOME is set by the fixture and deliberately not followed: the
    # default never moved before the cap, and an existing cache must stay seen.
    assert hsds_cache.cache_base() == tmp_path / "home" / ".cache" / "vaft" / "hsds"
    monkeypatch.setenv(hsds_cache.CACHE_DIR_ENV, str(tmp_path / "big-volume"))
    assert hsds_cache.cache_base() == tmp_path / "big-volume" / "hsds"
    assert hsds_cache.cache_base(tmp_path / "mine") == tmp_path / "mine"


# -- a fake cache on disk ----------------------------------------------------
ENDPOINT = "0123456789abcdef"


def _domain(base: Path, source: str, shot: int, name: str, size: int, age: float) -> Path:
    """Lay out one domain exactly as HSDSDomainCache does (marker, manifest record)."""
    hsds_cache.mark_owned(base)
    shot_dir = base / ENDPOINT / source / str(shot)
    shot_dir.mkdir(parents=True, exist_ok=True)
    manifest = shot_dir / "manifest.json"
    payload = (
        json.loads(manifest.read_text(encoding="utf-8"))
        if manifest.exists()
        else {"schema_version": 1, "domains": {}}
    )
    payload["domains"][f"{name}.h5"] = {"local_size": size}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    path = shot_dir / f"{name}.h5"
    path.write_bytes(b"x" * size)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def test_prune_evicts_least_recently_used_first(tmp_path):
    base = tmp_path / "cache"
    oldest = _domain(base, "main", 1, "equilibrium", 40 * KiB, age=300)
    middle = _domain(base, "main", 2, "magnetics", 40 * KiB, age=200)
    newest = _domain(base, "public", 3, "master", 40 * KiB, age=100)

    result = hsds_cache.prune("90K", cache=base)

    assert [entry.path for entry in result.removed] == [oldest]
    assert not oldest.exists() and middle.exists() and newest.exists()
    assert not oldest.parent.exists()  # the emptied shot directory goes, manifest included
    assert result.remaining_bytes == 80 * KiB


def test_prune_defaults_to_the_resolved_cap(tmp_path, monkeypatch):
    base = tmp_path / "cache"
    for shot in range(4):
        _domain(base, "main", shot, "master", 30 * KiB, age=100 - shot)
    monkeypatch.setenv(hsds_cache.CACHE_MAX_ENV, "64K")
    hsds_cache.prune(cache=base)
    assert sorted(entry.shot for entry in hsds_cache.entries(base)) == [2, 3]


def test_prune_older_than_removes_stale_domains_only(tmp_path):
    base = tmp_path / "cache"
    stale = _domain(base, "main", 1, "master", KiB, age=40 * 86400)
    fresh = _domain(base, "main", 2, "master", KiB, age=60)
    hsds_cache.prune("none", older_than="30d", cache=base)
    assert not stale.exists() and fresh.exists()


def test_clear_narrows_by_shot_and_source(tmp_path):
    base = tmp_path / "cache"
    a = _domain(base, "main", 1, "master", KiB, age=10)
    b = _domain(base, "main", 2, "master", KiB, age=10)
    c = _domain(base, "public", 1, "master", KiB, age=10)
    hsds_cache.clear(base, shot=1, source="main")
    assert not a.exists() and b.exists() and c.exists()
    hsds_cache.clear(base)
    assert hsds_cache.entries(base) == []


def test_cleanup_never_touches_files_it_does_not_own(tmp_path):
    base = tmp_path / "cache"
    _domain(base, "main", 1, "master", KiB, age=10)
    keep = []
    # no manifest; a foreign manifest in a non-endpoint directory; an unlisted
    # .h5 beside a VAFT manifest; a non-.h5 file in a VAFT shot directory
    for relative in ("notes/data.h5", "runs/a/12345/data.h5", f"{ENDPOINT}/main/1/mine.h5",
                     f"{ENDPOINT}/main/1/README"):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
        keep.append(path)
    (base / "runs/a/12345/manifest.json").write_text(
        '{"schema_version": 1, "domains": {"data.h5": {}}}', encoding="utf-8"
    )

    hsds_cache.clear(base)
    hsds_cache.prune(0, cache=base)

    assert all(path.exists() for path in keep)
    assert hsds_cache.entries(base) == []


def test_an_unmarked_custom_base_is_never_cleaned(tmp_path):
    base = tmp_path / "my-runs"
    _domain(base, "main", 1, "master", KiB, age=10)
    (base / hsds_cache.MARKER_NAME).unlink()
    with pytest.raises(ValueError, match="not marked"):
        hsds_cache.clear(base)
    with pytest.raises(ValueError, match="not marked"):
        hsds_cache.prune(0, cache=base)
    assert hsds_cache.enforce(base, 0) is None
    assert len(hsds_cache.entries(base)) == 1


def test_the_default_base_needs_no_marker(tmp_path):
    base = hsds_cache.cache_base()
    _domain(base, "main", 1, "master", KiB, age=10)
    (base / hsds_cache.MARKER_NAME).unlink()
    assert len(hsds_cache.clear().removed) == 1


def test_a_domain_that_cannot_be_removed_is_skipped(tmp_path, monkeypatch, caplog):
    base = tmp_path / "cache"
    stuck = _domain(base, "main", 1, "master", 40 * KiB, age=300)
    other = _domain(base, "main", 2, "master", 40 * KiB, age=200)
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == stuck:
            raise PermissionError(13, "in use")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with caplog.at_level("WARNING", logger="vaft.database.hsds_cache"):
        result = hsds_cache.prune(0, cache=base)
    assert [entry.path for entry in result.removed] == [other]
    assert stuck.exists() and "could not remove" in caplog.text


def test_stale_write_debris_is_swept(tmp_path):
    base = tmp_path / "cache"
    domain = _domain(base, "main", 1, "master", KiB, age=10)
    stale = domain.parent / ".master.h5.abc.part"
    fresh = domain.parent / ".master.h5.def.part"
    for path in (stale, fresh):
        path.write_bytes(b"half")
    old = time.time() - 2 * hsds_cache.STALE_PART_SECONDS
    os.utime(stale, (old, old))
    hsds_cache.prune("none", cache=base)
    assert not stale.exists() and fresh.exists()


def test_info_reports_size_cap_and_its_source(tmp_path, monkeypatch):
    base = tmp_path / "cache"
    _domain(base, "main", 1, "master", 3 * KiB, age=10)
    _domain(base, "public", 2, "magnetics", 5 * KiB, age=10)
    monkeypatch.setenv(hsds_cache.CACHE_MAX_ENV, "20G")
    report = hsds_cache.info(base)
    assert report["used_bytes"] == 8 * KiB
    assert (report["cap_bytes"], report["cap_source"]) == (20 * 1024**3, "$VAFT_HSDS_CACHE_MAX")
    assert (report["domains"], report["shots"], report["sources"]) == (2, 2, ["main", "public"])


# -- HSDSDomainCache honours the cap ---------------------------------------------
@pytest.fixture
def fake_hsds(monkeypatch):
    """Serve ``hdf5://<dir>/<shot>/<name>.h5`` with a fixed size per name."""
    sizes: dict[str, int] = {}
    fetched: list[str] = []

    def remote_info(self, remote_uri):
        return {"remote_uri": remote_uri, "last_modified": "rev-1", "backend_version": "1"}

    def hsget(remote_uri, target):
        fetched.append(remote_uri)
        name = remote_uri.rsplit("/", 1)[-1][:-3]
        with h5py.File(target, "w") as handle:
            handle["payload"] = np.zeros(sizes.get(name, KiB) // 8)

    monkeypatch.setattr(staging.HSDSDomainCache, "_endpoint", staticmethod(lambda: "fake"))
    monkeypatch.setattr(staging.HSDSDomainCache, "_remote_info", remote_info)
    monkeypatch.setattr(staging, "run_hsget", hsget)
    monkeypatch.setattr(staging, "ensure_imas_hdf5_userblock", lambda *a, **k: None)
    return sizes, fetched


def _fetch(base, shot, name, tmp_path, cache_max=None):
    store = staging.HSDSDomainCache("main", shot, base, cache_max)
    _, hit, _, _ = store.materialize(f"{name}.h5", tmp_path / "stage" / str(shot), transport="canonical")
    return hit


def test_the_cache_stays_under_its_cap_evicting_the_oldest(fake_hsds, tmp_path):
    sizes, _ = fake_hsds
    sizes.update(a=40 * KiB, b=40 * KiB, c=40 * KiB)
    base = tmp_path / "cache"
    for shot, name in ((1, "a"), (2, "b"), (3, "c")):
        assert _fetch(base, shot, name, tmp_path, cache_max="100K") is False
        assert sum(e.size for e in hsds_cache.entries(base)) <= 100 * KiB
    assert [entry.shot for entry in hsds_cache.entries(base)] == [2, 3]


def test_a_hit_counts_as_a_use(fake_hsds, tmp_path):
    sizes, fetched = fake_hsds
    sizes.update(a=40 * KiB, b=40 * KiB, c=40 * KiB)
    base = tmp_path / "cache"
    _fetch(base, 1, "a", tmp_path, "100K")
    _fetch(base, 2, "b", tmp_path, "100K")
    old = time.time() - 3600
    for entry in hsds_cache.entries(base):
        os.utime(entry.path, (old, old))
    assert _fetch(base, 1, "a", tmp_path, "100K") is True  # shot 1 is now the newest
    _fetch(base, 3, "c", tmp_path, "100K")
    assert sorted(entry.shot for entry in hsds_cache.entries(base)) == [1, 3]
    assert len(fetched) == 3


def test_a_domain_larger_than_the_cap_is_read_but_not_cached(fake_hsds, tmp_path, caplog):
    sizes, _ = fake_hsds
    sizes["big"] = 200 * KiB
    base = tmp_path / "cache"
    with caplog.at_level("WARNING", logger="vaft.database.hsds_cache"):
        assert _fetch(base, 1, "big", tmp_path, "100K") is False
        assert _fetch(base, 1, "big", tmp_path, "100K") is False
    assert (tmp_path / "stage" / "1" / "big.h5").is_file()
    assert hsds_cache.entries(base) == []
    assert sum("larger than the HSDS cache cap" in r.message for r in caplog.records) == 1


def test_a_zero_cap_disables_the_cache(fake_hsds, tmp_path):
    base = tmp_path / "cache"
    store = staging.HSDSDomainCache("main", 1, base, 0)
    assert store.enabled is False
    _fetch(base, 1, "a", tmp_path, 0)
    assert not base.exists()


def test_an_evicted_domain_is_fetched_again(fake_hsds, tmp_path):
    sizes, fetched = fake_hsds
    base = tmp_path / "cache"
    assert _fetch(base, 1, "a", tmp_path, "none") is False
    assert _fetch(base, 1, "a", tmp_path, "none") is True
    hsds_cache.clear(base)
    assert _fetch(base, 1, "a", tmp_path, "none") is False
    assert len(fetched) == 2


@pytest.mark.skipif(sys.platform == "win32", reason="the cache lock is POSIX-only")
def test_concurrent_writers_keep_the_cache_consistent(tmp_path):
    """Two processes filling one capped cache leave it under the cap and valid."""
    base = tmp_path / "cache"
    script = tmp_path / "writer.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            import h5py
            import numpy as np

            from vaft.database import staging

            base, worker = Path(sys.argv[1]), int(sys.argv[2])
            staging.HSDSDomainCache._endpoint = staticmethod(lambda: "fake")
            staging.HSDSDomainCache._remote_info = lambda self, uri: {
                "remote_uri": uri, "last_modified": "rev-1", "backend_version": "1"}

            def hsget(uri, target):
                with h5py.File(target, "w") as handle:
                    handle["payload"] = np.zeros(20 * 1024 // 8)

            staging.run_hsget = hsget
            staging.ensure_imas_hdf5_userblock = lambda *a, **k: None
            for shot in range(12):
                for name in ("shared", f"w{worker}"):  # one domain both write
                    store = staging.HSDSDomainCache("main", shot % 4, base, "100K")
                    store.materialize(f"{name}.h5", base.parent / f"stage{worker}" / str(shot),
                                      transport="canonical")
            """
        ),
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": str(Path(staging.__file__).parents[2])}
    workers = [
        subprocess.Popen([sys.executable, str(script), str(base), str(worker)], env=environment)
        for worker in (1, 2)
    ]
    assert [worker.wait(timeout=300) for worker in workers] == [0, 0]

    items = hsds_cache.entries(base)
    assert 0 < sum(entry.size for entry in items) <= 100 * KiB
    for entry in items:
        with h5py.File(entry.path, "r"):  # never a half-written domain
            pass
        record = json.loads((entry.path.parent / "manifest.json").read_text(encoding="utf-8"))
        assert record["domains"][entry.filename]["local_size"] == entry.size
    assert not list(base.rglob("*.part"))


def test_a_failed_cache_write_does_not_fail_the_load(fake_hsds, tmp_path, monkeypatch, caplog):
    def broken(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(hsds_cache, "copy_in", broken)
    with caplog.at_level("WARNING", logger="vaft.database.staging"):
        assert _fetch(tmp_path / "cache", 1, "a", tmp_path, "none") is False
    assert (tmp_path / "stage" / "1" / "a.h5").is_file()
    assert "could not cache" in caplog.text


def test_two_loaders_of_one_shot_keep_each_others_records(fake_hsds, tmp_path):
    base = tmp_path / "cache"
    first = staging.HSDSDomainCache("main", 1, base, "none")
    second = staging.HSDSDomainCache("main", 1, base, "none")
    first.materialize("a.h5", tmp_path / "s1", transport="canonical")
    second.materialize("b.h5", tmp_path / "s2", transport="canonical")
    assert sorted(entry.filename for entry in hsds_cache.entries(base)) == ["a.h5", "b.h5"]


# -- the CLI ----------------------------------------------------------------------
def _cli(*argv):
    out = io.StringIO()
    code = cache_cli.main(list(argv), out=out)
    return code, out.getvalue()


def test_cli_info_names_the_winning_setting(tmp_path, monkeypatch):
    base = tmp_path / "cache"
    _domain(base, "main", 1, "master", 2 * KiB, age=10)
    code, text = _cli("info", "--cache", str(base))
    assert code == 0 and "1 GiB (from default)" in text and "1 domain(s) of 1 shot(s)" in text
    monkeypatch.setenv(hsds_cache.CACHE_MAX_ENV, "5G")
    assert "5 GiB (from $VAFT_HSDS_CACHE_MAX)" in _cli("info", "--cache", str(base))[1]


def test_cli_set_limit_then_prune(tmp_path):
    base = tmp_path / "cache"
    for shot in range(3):
        _domain(base, "main", shot, "master", 30 * KiB, age=100 - shot)
    code, text = _cli("set-limit", "40K")
    assert code == 0 and str(hsds_cache.config_path()) in text
    assert "(from " + str(hsds_cache.config_path()) in _cli("info", "--cache", str(base))[1]
    code, text = _cli("prune", "--cache", str(base))
    assert code == 0 and "Evicted 2 domain(s)" in text
    assert [entry.shot for entry in hsds_cache.entries(base)] == [2]


def test_cli_list_and_clear(tmp_path, capsys):
    base = tmp_path / "cache"
    _domain(base, "main", 39915, "equilibrium", KiB, age=10)
    _domain(base, "main", 40600, "magnetics", KiB, age=10)
    code, text = _cli("list", "--cache", str(base), "--shot", "39915")
    assert code == 0 and "equilibrium" in text and "magnetics" not in text

    assert _cli("clear", "--cache", str(base))[0] == 2  # everything needs --yes
    assert "--yes" in capsys.readouterr().err
    assert len(hsds_cache.entries(base)) == 2
    assert _cli("clear", "--cache", str(base), "--shot", "39915")[0] == 0
    assert [entry.shot for entry in hsds_cache.entries(base)] == [40600]
    assert _cli("clear", "--cache", str(base), "--yes")[0] == 0
    assert hsds_cache.entries(base) == []


def test_cli_refuses_a_bad_size(capsys):
    assert _cli("set-limit", "huge")[0] == 2
    assert "not a cache size" in capsys.readouterr().err


def test_vaft_cache_is_a_registered_command():
    from vaft.cli import _main

    assert _main._COMMANDS["cache"][0] == ".cache"
