"""Size cap and housekeeping for the local HSDS domain cache (#1759, #964).

``vaft.database.load(..., cache="auto")`` keeps every validated HSDS domain it
downloads (:class:`vaft.database.staging.HSDSDomainCache`). Cached domains are
regenerable from HSDS, so the cache is disposable: this module bounds it and
lets users inspect and clean it.

Layout, under the cache base::

    <base>/<endpoint hash>/<source>/<shot>/{master.h5, <ids>.h5, manifest.json}

The base is ``cache=<path>`` when given, otherwise ``$VAFT_CACHE_DIR/hsds``,
``$XDG_CACHE_HOME/vaft/hsds`` or ``~/.cache/vaft/hsds``, in that order.

**Cap.** Least-recently-used domains are evicted once the cache exceeds its
cap, after every write. The cap is resolved, highest first, from

1. the ``cache_max`` argument of ``vaft.database.load`` (and friends),
2. the ``VAFT_HSDS_CACHE_MAX`` environment variable,
3. ``hsds.cache_max`` in the user config file
   (``$XDG_CONFIG_HOME/vaft/config.toml``, else ``~/.config/vaft/config.toml``;
   ``vaft cache set-limit`` writes it),
4. the built-in default, 1 GiB.

Sizes are bytes or a number with a binary suffix (``500M``, ``20G``, ``1GiB``;
``1G`` is 1024**3 bytes). ``0`` disables caching; ``none`` or ``unlimited``
removes the cap. A single domain larger than the cap is read without being
cached.

**Recency.** A domain's last use is its file modification time, which VAFT
sets itself on every cache hit and write; file access times are not used
(``noatime`` mounts never update them). Nothing else in the cache reads the
modification time: validity is decided by the manifest's remote revision and
size.

**Concurrency.** Readers and writers of cache files hold a shared lock on
``<base>/.lock``; eviction, pruning and clearing hold it exclusively, so a
domain is never removed while another process is copying it in or out. The
lock is ``fcntl.flock`` and therefore POSIX-only; on Windows it is a no-op.

Only VAFT-owned files are touched: domain files (``*.h5``), their
``manifest.json`` and the directories that held them, inside the cache base.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_DIR_ENV",
    "CACHE_MAX_ENV",
    "DEFAULT_CACHE_MAX",
    "CacheEntry",
    "CacheLimit",
    "cache_base",
    "clear",
    "config_path",
    "entries",
    "format_size",
    "info",
    "parse_age",
    "parse_size",
    "prune",
    "resolve_cache_max",
    "set_cache_max",
]

#: Built-in cap: 1 GiB.
DEFAULT_CACHE_MAX = 1024**3
CACHE_MAX_ENV = "VAFT_HSDS_CACHE_MAX"
CACHE_DIR_ENV = "VAFT_CACHE_DIR"
CONFIG_SECTION = "hsds"
CONFIG_KEY = "cache_max"
LOCK_NAME = ".lock"
MANIFEST_NAME = "manifest.json"

_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)
_AGE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
_AGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_UNLIMITED = {"none", "unlimited", "inf", "infinity"}


# -- sizes -----------------------------------------------------------------
def parse_size(value: int | str | None) -> int | None:
    """Bytes for a cap; ``None`` means unlimited.

    Accepts an integer (bytes), a numeric string with an optional binary
    suffix (``K``, ``M``, ``G``, ``T``, optionally followed by ``B`` or
    ``iB``), or ``none``/``unlimited``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"not a cache size: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"cache size must not be negative: {value!r}")
        return value
    text = str(value).strip()
    if text.lower() in _UNLIMITED:
        return None
    match = _SIZE.match(text)
    if match is None:
        raise ValueError(
            f"not a cache size: {value!r} (use bytes or e.g. 500M, 20G; "
            "0 disables the cache, 'none' removes the cap)"
        )
    number, unit = match.groups()
    return int(float(number) * _UNITS[unit.upper()])


def format_size(size: int | None) -> str:
    """``1073741824`` -> ``"1 GiB"``; ``None`` -> ``"unlimited"``."""
    if size is None:
        return "unlimited"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" or value == int(value) else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB" if value != int(value) else f"{value:.0f} TiB"


def parse_age(value: str | float | int) -> float:
    """Seconds for ``30d``, ``12h``, ``90m``, ``45s``, ``2w`` or a number of seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = _AGE.match(str(value))
    if match is None:
        raise ValueError(f"not an age: {value!r} (use e.g. 30d, 12h, 90m)")
    number, unit = match.groups()
    return float(number) * _AGE_UNITS[unit.lower()]


# -- configuration ---------------------------------------------------------
def cache_base(cache: str | os.PathLike = "auto") -> Path:
    """The cache base directory for ``cache`` ("auto" or a path)."""
    if cache != "auto":
        return Path(cache).expanduser()
    if os.environ.get(CACHE_DIR_ENV):
        return Path(os.environ[CACHE_DIR_ENV]).expanduser() / "hsds"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "vaft" / "hsds"
    return Path.home() / ".cache" / "vaft" / "hsds"


def config_path() -> Path:
    """The user config file holding ``hsds.cache_max``."""
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".config"
    return base / "vaft" / "config.toml"


def _read_config_value(path: Path) -> str | int | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"cannot read {path}: {error}") from error
        section = data.get(CONFIG_SECTION)
        return section.get(CONFIG_KEY) if isinstance(section, dict) else None
    # No TOML parser (3.10 without tomli): read the one key we write.
    return _scan_config(text)


def _scan_config(text: str) -> str | int | None:
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
        elif section == CONFIG_SECTION and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key == CONFIG_KEY:
                value = value.strip("\"'")
                return int(value) if value.isdigit() else value
    return None


@dataclass(frozen=True)
class CacheLimit:
    """A resolved cap: bytes (``None`` = unlimited) and where it came from."""

    bytes: int | None
    source: str

    def __str__(self) -> str:
        return f"{format_size(self.bytes)} ({self.source})"


def resolve_cache_max(cache_max: int | str | None = None) -> CacheLimit:
    """Resolve the cap: argument > ``$VAFT_HSDS_CACHE_MAX`` > config file > 1 GiB.

    ``cache_max=None`` means "not given", so the next source is consulted; pass
    ``"none"`` or ``"unlimited"`` to remove the cap from the argument.
    """
    if cache_max is not None:
        return CacheLimit(parse_size(cache_max), "argument")
    env = os.environ.get(CACHE_MAX_ENV)
    if env is not None and env.strip():
        try:
            return CacheLimit(parse_size(env), f"${CACHE_MAX_ENV}")
        except ValueError as error:
            raise ValueError(f"${CACHE_MAX_ENV}: {error}") from None
    path = config_path()
    value = _read_config_value(path)
    if value is not None:
        try:
            return CacheLimit(parse_size(value), f"{path} [{CONFIG_SECTION}] {CONFIG_KEY}")
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
    return CacheLimit(DEFAULT_CACHE_MAX, "default")


def set_cache_max(value: int | str, path: Path | None = None) -> Path:
    """Persist ``hsds.cache_max`` in the user config file; returns the file.

    Other sections, keys and comments are kept: only the ``cache_max`` line of
    the ``[hsds]`` table is replaced or added.
    """
    parse_size(value)  # refuse a bad value before writing anything
    path = config_path() if path is None else Path(path)
    text = str(value).strip() if not isinstance(value, int) else str(value)
    line = f"{CONFIG_KEY} = {text}" if text.isdigit() else f'{CONFIG_KEY} = "{text}"'
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    section = None
    start = None
    replaced = False
    for index, raw in enumerate(lines):
        stripped = raw.split("#", 1)[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section == CONFIG_SECTION and not replaced:
                break
            section = stripped[1:-1].strip()
            if section == CONFIG_SECTION:
                start = index
        elif section == CONFIG_SECTION and "=" in stripped:
            if stripped.split("=", 1)[0].strip() == CONFIG_KEY:
                lines[index] = line
                replaced = True
                break
    if not replaced:
        if start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines += [f"[{CONFIG_SECTION}]", line]
        else:
            lines.insert(start + 1, line)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


# -- locking ---------------------------------------------------------------
@contextmanager
def cache_lock(base: Path, *, exclusive: bool) -> Iterator[None]:
    """Hold ``<base>/.lock`` shared (copying files) or exclusive (removing them)."""
    try:
        import fcntl
    except ImportError:  # Windows: no advisory lock (see the module docstring)
        yield
        return
    base.mkdir(parents=True, exist_ok=True)
    with open(base / LOCK_NAME, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# -- inspection ------------------------------------------------------------
@dataclass(frozen=True)
class CacheEntry:
    """One cached domain file."""

    path: Path
    endpoint: str
    source: str
    shot: int
    filename: str
    size: int
    last_used: float


def _shot_dirs(base: Path) -> Iterator[tuple[Path, str, str, int]]:
    """Yield ``(shot_dir, endpoint, source, shot)`` for every cached shot."""
    if not base.is_dir():
        return
    for manifest in base.rglob(MANIFEST_NAME):
        shot_dir = manifest.parent
        parts = shot_dir.relative_to(base).parts
        if len(parts) < 3 or not parts[-1].isdigit():
            continue  # not a VAFT cache shot directory; never touched
        yield shot_dir, parts[0], "/".join(parts[1:-1]), int(parts[-1])


def entries(
    cache: str | os.PathLike = "auto",
    *,
    shot: int | None = None,
    source: str | None = None,
) -> list[CacheEntry]:
    """Every cached domain, oldest use first, optionally for one shot/source."""
    base = cache_base(cache)
    found: list[CacheEntry] = []
    for shot_dir, endpoint, name, number in _shot_dirs(base):
        if shot is not None and number != int(shot):
            continue
        if source is not None and name != source.strip("/"):
            continue
        for path in shot_dir.glob("*.h5"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            found.append(
                CacheEntry(path, endpoint, name, number, path.name, stat.st_size, stat.st_mtime)
            )
    found.sort(key=lambda entry: (entry.last_used, str(entry.path)))
    return found


def info(cache: str | os.PathLike = "auto", *, cache_max: int | str | None = None) -> dict:
    """Location, used size, cap (and where it came from) and counts."""
    items = entries(cache)
    limit = resolve_cache_max(cache_max)
    return {
        "location": str(cache_base(cache)),
        "used_bytes": sum(entry.size for entry in items),
        "cap_bytes": limit.bytes,
        "cap_source": limit.source,
        "domains": len(items),
        "shots": len({(e.endpoint, e.source, e.shot) for e in items}),
        "sources": sorted({e.source for e in items}),
        "endpoints": len({e.endpoint for e in items}),
    }


# -- removal ---------------------------------------------------------------
def _remove(entry: CacheEntry) -> bool:
    try:
        entry.path.unlink()
    except FileNotFoundError:
        return False
    _tidy(entry.path.parent)
    return True


def _tidy(shot_dir: Path) -> None:
    """Drop a shot directory once its last domain is gone (manifest included)."""
    if any(shot_dir.glob("*.h5")):
        return
    manifest = shot_dir / MANIFEST_NAME
    try:
        manifest.unlink()
    except FileNotFoundError:
        pass
    for pattern in ("manifest.tmp", "manifest.json.*.tmp", "*.h5.part"):
        for leftover in shot_dir.glob(pattern):  # an interrupted write's debris
            leftover.unlink(missing_ok=True)
    try:
        shot_dir.rmdir()
    except OSError:
        pass  # something else lives here; leave it


@dataclass(frozen=True)
class PruneResult:
    removed: tuple[CacheEntry, ...]
    freed_bytes: int
    remaining_bytes: int


def _prune_locked(
    base: Path,
    max_bytes: int | None,
    older_than: float | None,
    now: float,
) -> PruneResult:
    items = entries(base)
    total = sum(entry.size for entry in items)
    removed: list[CacheEntry] = []
    for entry in items:  # oldest use first
        too_old = older_than is not None and now - entry.last_used > older_than
        too_big = max_bytes is not None and total > max_bytes
        if not (too_old or too_big):
            if older_than is None:
                break  # sorted by age: nothing later is older or needed
            continue
        if _remove(entry):
            removed.append(entry)
            total -= entry.size
    return PruneResult(tuple(removed), sum(e.size for e in removed), total)


def prune(
    max_bytes: int | str | None = None,
    *,
    older_than: str | float | None = None,
    cache: str | os.PathLike = "auto",
    cache_max: int | str | None = None,
) -> PruneResult:
    """Evict least-recently-used domains until the cache fits.

    ``max_bytes`` defaults to the resolved cap (:func:`resolve_cache_max` with
    ``cache_max``). ``older_than`` (e.g. ``"30d"``) also removes every domain
    not used for that long.
    """
    limit = (
        resolve_cache_max(cache_max).bytes if max_bytes is None else parse_size(max_bytes)
    )
    age = None if older_than is None else parse_age(older_than)
    base = cache_base(cache)
    if not base.is_dir():
        return PruneResult((), 0, 0)
    with cache_lock(base, exclusive=True):
        return _prune_locked(base, limit, age, time.time())


def clear(
    cache: str | os.PathLike = "auto",
    *,
    shot: int | None = None,
    source: str | None = None,
) -> PruneResult:
    """Remove cached domains: all of them, or those of one shot and/or source."""
    base = cache_base(cache)
    if not base.is_dir():
        return PruneResult((), 0, 0)
    with cache_lock(base, exclusive=True):
        removed = tuple(e for e in entries(base, shot=shot, source=source) if _remove(e))
        remaining = sum(e.size for e in entries(base))
    return PruneResult(removed, sum(e.size for e in removed), remaining)


# -- used by HSDSDomainCache -----------------------------------------------
_too_large_reported: set[str] = set()


def enforce(base: Path, max_bytes: int | None) -> PruneResult | None:
    """Evict down to ``max_bytes`` after a write (no-op when unlimited)."""
    if max_bytes is None or not base.is_dir():
        return None
    with cache_lock(base, exclusive=True):
        result = _prune_locked(base, max_bytes, None, time.time())
    if result.removed:
        logger.info(
            "HSDS cache over its %s cap: evicted %d domain(s), %s",
            format_size(max_bytes), len(result.removed), format_size(result.freed_bytes),
        )
    return result


def report_too_large(uri: str, size: int, max_bytes: int) -> None:
    """Log once per domain that it is read without being cached."""
    if uri in _too_large_reported:
        return
    _too_large_reported.add(uri)
    logger.warning(
        "%s (%s) is larger than the HSDS cache cap (%s); read without caching",
        uri, format_size(size), format_size(max_bytes),
    )


def touch(path: Path) -> None:
    """Mark ``path`` as just used (its modification time is the LRU stamp)."""
    try:
        os.utime(path)
    except OSError:
        pass


def copy_in(source: Path, destination: Path) -> None:
    """Write a domain into the cache atomically (never a half-written file)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    shutil.copyfile(source, partial)
    partial.replace(destination)
