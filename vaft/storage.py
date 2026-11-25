"""Shared local cache root and namespaces (#2317).

VAFT keeps three kinds of local data, with different lifecycles:

* **FileDB** (``VAFT_FILEDB_DIR``): persistent, authoritative pipeline
  products. Nothing here ever removes FileDB content.
* **Caches** under one root, ``VAFT_CACHE_DIR``: disposable copies that can be
  fetched or generated again. Each kind lives in its own *namespace*.
* **User-managed artifacts**, such as the MCP artifact root
  (``VAFT_ARTIFACT_DIR``). That variable is an *access* boundary for the MCP
  server, not a storage location, and this module never derives one from the
  other.

Namespaces
----------

=========================  ==================================  =====================
namespace                  holds                               eviction
=========================  ==================================  =====================
``hsds``                   validated HSDS domain copies        LRU cap (1 GiB)
``public``                 verified public-database downloads  explicit only
``references/samples``     verified Release copies of samples  explicit only
``references/efit-baseline`` the single EFIT baseline table    explicit only
``generated/efund``        EFUND tables keyed by identity      explicit only
=========================  ==================================  =====================

Where a namespace lives, highest precedence first:

* ``hsds``, unchanged from before this module (:mod:`vaft.database.hsds_cache`):
  an explicit ``cache=`` path, then ``$VAFT_CACHE_DIR/hsds``, then
  ``~/.cache/vaft/hsds`` on every platform.
* ``public``: an explicit path, then ``$VAFT_PUBLIC_DATA_DIR``, then
  ``$VAFT_CACHE_DIR/public``, then the platform default
  (``~/Library/Caches/vaft/public`` on macOS, ``%LOCALAPPDATA%/vaft/public`` on
  Windows, ``$XDG_CACHE_HOME/vaft/public`` or ``~/.cache/vaft/public``
  elsewhere). If ``$VAFT_CACHE_DIR/public`` does not exist yet but the platform
  default does, the platform default keeps being used, so setting
  ``VAFT_CACHE_DIR`` neither relocates nor re-downloads an existing public
  cache. Moving it is the user's explicit choice.
* Every other namespace: an explicit path, then ``$VAFT_CACHE_DIR/<namespace>``,
  then ``<platform cache root>/<namespace>``.

An empty environment variable counts as unset. Paths are expanded (``~``) but
not resolved, and resolving a path never creates a directory or touches the
network: only :func:`ensure_namespace` and :func:`publish_entry` write.

Ownership
---------

Only content VAFT wrote is ever removed. For the namespaces managed here
(everything but ``hsds``, which keeps its own rules), a namespace directory is
VAFT's only when it carries the :data:`NAMESPACE_MARKER` file naming that
namespace. An *entry* is a sub-directory holding an :data:`ENTRY_RECORD`, and it
counts only if it is a real directory directly inside the namespace directory,
not a symbolic link. Anything else (stray files, half-written ``.part``
directories younger than an hour, symbolic links) is reported, never removed.
Removal also refuses a namespace that contains, or is, the FileDB root.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterator

__all__ = [
    "CACHE_DIR_ENV",
    "ENTRY_RECORD",
    "FILEDB_DIR_ENV",
    "NAMESPACES",
    "NAMESPACE_MARKER",
    "PUBLIC_DATA_ENV",
    "Entry",
    "Namespace",
    "RemovalResult",
    "cache_path",
    "cache_root",
    "default_cache_root",
    "ensure_namespace",
    "entries",
    "entry_path",
    "file_lock",
    "namespace",
    "namespace_path",
    "namespace_report",
    "publish_entry",
    "remove_entries",
    "touch_entry",
]

CACHE_DIR_ENV = "VAFT_CACHE_DIR"
PUBLIC_DATA_ENV = "VAFT_PUBLIC_DATA_DIR"
FILEDB_DIR_ENV = "VAFT_FILEDB_DIR"
NAMESPACE_MARKER = ".vaft-cache-namespace"
ENTRY_RECORD = ".vaft-entry.json"
LOCK_NAME = ".lock"
#: How long a Windows lock holder waits for another process.
LOCK_WAIT_S = 60.0
#: A ``.part`` directory older than this is an interrupted write.
STALE_PART_SECONDS = 3600.0


@dataclass(frozen=True)
class Namespace:
    """One cache namespace: where it lives and who owns its content."""

    name: str
    parts: tuple[str, ...]
    owner: str
    holds: str
    eviction: str  # "lru-cap" or "explicit"
    override_env: str | None = None


NAMESPACES: dict[str, Namespace] = {
    ns.name: ns
    for ns in (
        Namespace(
            "hsds", ("hsds",), "vaft.database.hsds_cache",
            "validated copies of HSDS domains", "lru-cap",
        ),
        Namespace(
            "public", ("public",), "vaft.data.public",
            "verified downloads of registered public databases", "explicit",
            override_env=PUBLIC_DATA_ENV,
        ),
        Namespace(
            "references/samples", ("references", "samples"), "reference-sample resolver (#2329)",
            "verified copies of published reference-sample Release assets", "explicit",
        ),
        Namespace(
            "references/efit-baseline", ("references", "efit-baseline"), "EFIT table resolver (#2330)",
            "the single published EFIT Green-table baseline", "explicit",
        ),
        Namespace(
            "generated/efund", ("generated", "efund"), "EFIT table resolver (#2330)",
            "EFUND Green tables keyed by calculation identity (not interchangeable with the baseline)",
            "explicit",
        ),
    )
}


# -- paths -----------------------------------------------------------------
def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _platform_cache_root(system: str, platform: str, home: Path) -> Path:
    if system == "nt":
        local = _env("LOCALAPPDATA")
        root = Path(local) if local else home / "AppData" / "Local"
    elif platform == "darwin":
        root = home / "Library" / "Caches"
    else:
        xdg = _env("XDG_CACHE_HOME")
        root = Path(xdg).expanduser() if xdg else home / ".cache"
    return root / "vaft"


def default_cache_root() -> Path:
    """The platform cache root used when ``VAFT_CACHE_DIR`` is unset.

    ``%LOCALAPPDATA%/vaft`` on Windows, ``~/Library/Caches/vaft`` on macOS,
    ``$XDG_CACHE_HOME/vaft`` (else ``~/.cache/vaft``) elsewhere: the roots the
    public-data cache has always used.
    """
    return _platform_cache_root(os.name, sys.platform, Path.home())


def cache_root() -> Path:
    """``$VAFT_CACHE_DIR`` when set, else :func:`default_cache_root`."""
    configured = _env(CACHE_DIR_ENV)
    return Path(configured).expanduser() if configured else default_cache_root()


def _check_parts(parts: tuple[str, ...]) -> None:
    if not parts:
        raise ValueError("a cache path needs at least one component")
    for part in parts:
        if (
            not isinstance(part, str)
            or part in ("", ".", "..")
            or "/" in part
            or "\\" in part
            or "\0" in part
            or (len(part) >= 2 and part[1] == ":")
        ):
            raise ValueError(f"not a single cache path component: {part!r}")


def cache_path(*parts: str, root: str | os.PathLike[str] | None = None) -> Path:
    """``<root>/<parts...>`` under the cache root, refusing traversal.

    Each part must be one path component: no separators, no ``.``/``..``, no
    drive. Nothing is created.
    """
    _check_parts(parts)
    base = Path(root).expanduser() if root is not None else cache_root()
    return base.joinpath(*parts)


def namespace(name: str) -> Namespace:
    """The registered :class:`Namespace` called ``name``."""
    try:
        return NAMESPACES[name]
    except KeyError:
        raise ValueError(
            f"unknown cache namespace {name!r}; known: {', '.join(NAMESPACES)}"
        ) from None


def _hsds_path() -> Path:
    configured = _env(CACHE_DIR_ENV)
    if configured:
        return Path(configured).expanduser() / "hsds"
    return Path.home() / ".cache" / "vaft" / "hsds"


def namespace_path(name: str, override: str | os.PathLike[str] | None = None) -> Path:
    """Directory of namespace ``name`` (see the module docstring for precedence)."""
    ns = namespace(name)
    if override is not None:
        return Path(override).expanduser()
    if ns.name == "hsds":
        return _hsds_path()
    if ns.override_env and _env(ns.override_env):
        return Path(_env(ns.override_env)).expanduser()  # type: ignore[arg-type]
    configured = _env(CACHE_DIR_ENV)
    if configured:
        shared = Path(configured).expanduser().joinpath(*ns.parts)
        if ns.name == "public":
            legacy = default_cache_root().joinpath(*ns.parts)
            if not shared.exists() and legacy.is_dir():
                return legacy
        return shared
    return default_cache_root().joinpath(*ns.parts)


# -- locking ---------------------------------------------------------------
@contextmanager
def file_lock(directory: Path, *, exclusive: bool, name: str = LOCK_NAME) -> Iterator[None]:
    """Hold ``<directory>/<name>`` shared (reading, writing an entry) or exclusive (removing).

    ``fcntl.flock`` on POSIX. On Windows ``msvcrt.locking`` locks one byte; it
    has no shared mode, so every holder there is exclusive, and a held lock is
    waited for in short retries.
    """
    directory.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        with open(directory / name, "a+b") as handle:
            handle.seek(0)
            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    with open(directory / name, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# -- ownership and entries -------------------------------------------------
def _managed(name: str) -> Namespace:
    ns = namespace(name)
    if ns.name == "hsds":
        raise ValueError("the hsds namespace is managed by vaft.database.hsds_cache")
    return ns


def ensure_namespace(name: str, override: str | os.PathLike[str] | None = None) -> Path:
    """Create the namespace directory and its ownership marker; return the directory."""
    ns = _managed(name)
    directory = namespace_path(name, override)
    marker = directory / NAMESPACE_MARKER
    if not marker.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        # Written through a temporary and renamed, so a concurrent reader never
        # sees a half-written marker and mistakes the namespace for foreign.
        handle, name = tempfile.mkstemp(dir=directory, prefix=f"{NAMESPACE_MARKER}.", suffix=".part")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"namespace": ns.name, "owner": ns.owner}, indent=2) + "\n")
            os.replace(name, marker)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
    return directory


def _owned(directory: Path, name: str) -> bool:
    try:
        record = json.loads((directory / NAMESPACE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get("namespace") == name


@dataclass(frozen=True)
class Entry:
    """One VAFT-written entry of a managed namespace."""

    key: str
    path: Path
    size: int
    last_used: float
    record: dict[str, Any] = field(default_factory=dict)


def _tree_size(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for filename in files:
            try:
                total += os.lstat(os.path.join(root, filename)).st_size
            except OSError:
                pass
    return total


def _is_entry_dir(directory: Path, child: Path) -> bool:
    if child.is_symlink() or not child.is_dir() or child.name.startswith("."):
        return False
    if child.parent != directory:
        return False
    return (child / ENTRY_RECORD).is_file() and not (child / ENTRY_RECORD).is_symlink()


def _read_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads((path / ENTRY_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def entries(name: str, override: str | os.PathLike[str] | None = None) -> list[Entry]:
    """VAFT-written entries of a managed namespace, least recently used first.

    An unmarked namespace directory has no entries: nothing in it is VAFT's.
    """
    _managed(name)
    directory = namespace_path(name, override)
    if not directory.is_dir() or not _owned(directory, name):
        return []
    found = []
    for child in sorted(directory.iterdir()):
        if _is_entry_dir(directory, child):
            found.append(
                Entry(
                    key=child.name,
                    path=child,
                    size=_tree_size(child),
                    last_used=(child / ENTRY_RECORD).stat().st_mtime,
                    record=_read_record(child),
                )
            )
    return sorted(found, key=lambda e: (e.last_used, e.key))


def entry_path(name: str, key: str, override: str | os.PathLike[str] | None = None) -> Path | None:
    """The complete entry ``key`` of namespace ``name``, or ``None``.

    A directory without its record (an interrupted or foreign write) is not an
    entry.
    """
    _managed(name)
    _check_parts((key,))
    directory = namespace_path(name, override)
    candidate = directory / key
    if _owned(directory, name) and _is_entry_dir(directory, candidate):
        return candidate
    return None


def touch_entry(path: Path) -> None:
    """Mark an entry as just used (its record's modification time)."""
    try:
        os.utime(path / ENTRY_RECORD)
    except OSError:
        pass


def publish_entry(
    name: str,
    key: str,
    populate: Callable[[Path], None],
    *,
    record: dict[str, Any] | None = None,
    override: str | os.PathLike[str] | None = None,
) -> Path:
    """Build entry ``key`` atomically and return its path.

    ``populate(directory)`` writes the entry's files into a private temporary
    directory and verifies them (raising aborts the publish; nothing is left
    behind). The record is written last, and the directory is renamed into
    place, so a reader never sees a half-written entry. If another process
    published ``key`` first, its entry wins and this one is discarded. The
    scientific verification belongs to the caller; this only guarantees that
    what lands is complete.
    """
    _check_parts((key,))
    directory = ensure_namespace(name, override)
    with file_lock(directory, exclusive=False):
        existing = entry_path(name, key, override)
        if existing is not None:
            return existing
        partial = Path(tempfile.mkdtemp(dir=directory, prefix=f".{key}.", suffix=".part"))
        try:
            populate(partial)
            payload = {"key": key, "namespace": name, "created": time.time(), **(record or {})}
            (partial / ENTRY_RECORD).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            try:
                partial.rename(directory / key)
            except OSError:
                winner = entry_path(name, key, override)
                if winner is None:
                    raise
                shutil.rmtree(partial, ignore_errors=True)
                return winner
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return directory / key


@dataclass(frozen=True)
class RemovalResult:
    removed: list[str]
    freed_bytes: int
    remaining_bytes: int
    skipped: list[str]
    dry_run: bool


def _filedb_conflict(directory: Path) -> str | None:
    configured = _env(FILEDB_DIR_ENV)
    if not configured:
        return None
    try:
        filedb = Path(configured).expanduser().resolve()
        target = directory.resolve()
    except OSError:
        return None
    if target == filedb or filedb.is_relative_to(target):
        return f"{directory} contains the FileDB root {filedb}; refusing to remove anything there"
    return None


def remove_entries(
    name: str,
    *,
    keys: list[str] | None = None,
    older_than: float | None = None,
    everything: bool = False,
    dry_run: bool = False,
    override: str | os.PathLike[str] | None = None,
) -> RemovalResult:
    """Remove VAFT-written entries of a managed namespace.

    Select with ``keys``, ``older_than`` (seconds since last use) or
    ``everything=True``; with none of them nothing is removed. Only entries
    (see the module docstring) are candidates; unknown files are listed in
    ``skipped``. ``dry_run`` reports without removing.
    """
    _managed(name)
    if keys is None and older_than is None and not everything:
        raise ValueError("say what to remove: keys, older_than, or everything=True")
    directory = namespace_path(name, override)
    if not directory.is_dir():
        return RemovalResult([], 0, 0, [], dry_run)
    if not _owned(directory, name):
        raise ValueError(f"{directory} has no {NAMESPACE_MARKER} for {name!r}; not removing anything there")
    conflict = _filedb_conflict(directory)
    if conflict:
        raise ValueError(conflict)
    if keys is not None:
        for key in keys:
            _check_parts((key,))
    removed: list[str] = []
    freed = 0
    with file_lock(directory, exclusive=True):
        now = time.time()
        current = entries(name, override)
        skipped = sorted(
            child.name
            for child in directory.iterdir()
            if child.name not in {NAMESPACE_MARKER, LOCK_NAME}
            and not _is_entry_dir(directory, child)
            and not (child.name.endswith(".part") and child.name.startswith("."))
        )
        for entry in current:
            chosen = (
                everything
                or (keys is not None and entry.key in keys)
                or (older_than is not None and now - entry.last_used >= older_than)
            )
            if not chosen:
                continue
            removed.append(entry.key)
            freed += entry.size
            if not dry_run:
                shutil.rmtree(entry.path)
        if not dry_run:
            for child in directory.iterdir():
                if (
                    child.name.startswith(".")
                    and child.name.endswith(".part")
                    and child.is_dir()
                    and not child.is_symlink()
                    and now - child.stat().st_mtime > STALE_PART_SECONDS
                ):
                    shutil.rmtree(child, ignore_errors=True)
        remaining = sum(e.size for e in current if e.key not in removed)
    return RemovalResult(removed, freed, remaining, skipped, dry_run)


def namespace_report(name: str, override: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Location, ownership, size and entry count of one namespace (read-only)."""
    ns = namespace(name)
    directory = namespace_path(name, override)
    report: dict[str, Any] = {
        "namespace": ns.name,
        "location": str(directory),
        "exists": directory.is_dir(),
        "holds": ns.holds,
        "owner": ns.owner,
        "eviction": ns.eviction,
    }
    if ns.name == "hsds":
        return report
    owned = directory.is_dir() and _owned(directory, name)
    found = entries(name, override) if owned else []
    report.update(
        {
            "marked": owned,
            "entries": len(found),
            "used_bytes": sum(e.size for e in found),
        }
    )
    return report
