"""Selective, cached materialization of IMAS HDF5 domains from HSDS.

The native IMAS and eager OMAS readers require ordinary local HDF5 files.  This
module keeps that requirement while avoiding a complete-shot download when a
caller has asked for specific top-level IDSs.  It deliberately caches files,
not mutable IMAS/OMAS Python objects.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import logging
import shutil
import threading
import time
from typing import Any, Iterable, Literal

import h5py

try:
    import h5pyd
except ImportError:  # pragma: no cover - public guard covers this path
    h5pyd = None

from . import hsds_cache
from .transport import run_hsget
from .h5image import H5ImageUnavailableError, materialize_image
from .utils import _require_h5pyd, ensure_imas_hdf5_userblock

logger = logging.getLogger(__name__)

__all__ = [
    "HSDSDomainCache",
    "create_partial_master",
    "external_h5_links",
    "merge_master_links",
    "requested_ids_from_paths",
    "stage_imas_shot",
]


def external_h5_links(master_path: Path) -> list[str]:
    """Return filenames referenced by top-level external links in ``master``."""
    filenames: list[str] = []
    with h5py.File(master_path, "r") as master:
        for name in master:
            link = master.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink) and link.filename.endswith(".h5"):
                filenames.append(Path(link.filename).name)
    return sorted(set(filenames))


def merge_master_links(
    current_master: Path,
    previous_master: Path | None,
    *,
    present_files: Iterable[str],
) -> tuple[str, ...]:
    """Fold a previous master's IDS links back into a freshly written one.

    A shot folder is written by several stages, each owning a different part of
    the IMAS tree. Each write produces a ``master.h5`` describing only what that
    stage wrote, and uploading it as-is would make every other stage's IDS
    invisible: :func:`stage_imas_shot` resolves what a shot contains from the
    master's external links, not from the folder listing.

    So the master is merged rather than replaced. Links are carried over from
    ``previous_master`` -- never fabricated -- so the internal target path of
    each link stays whatever the IMAS writer chose. A carried link is kept only
    while its file is still ``present_files``; the current master always wins a
    name collision, because that stage just rewrote it.

    Returns the link names added from the previous master.
    """
    if previous_master is None or not Path(previous_master).exists():
        return ()
    present = {Path(name).name for name in present_files}
    added: list[str] = []
    with h5py.File(current_master, "r+") as current:
        existing = set(current)
        with h5py.File(previous_master, "r") as previous:
            for name in previous:
                if name in existing:
                    continue
                link = previous.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink):
                    continue
                if Path(link.filename).name not in present:
                    continue
                current[name] = h5py.ExternalLink(link.filename, link.path)
                added.append(name)
    return tuple(sorted(added))


def requested_ids_from_paths(paths: Iterable[Any] | None) -> tuple[str, ...] | None:
    """Infer requested top-level IDS names from OMAS/IMAS paths.

    ``None`` retains the historical complete-shot behavior.  The parser is
    intentionally conservative: an unfamiliar path is passed through to OMAS
    but does not silently broaden the download plan.
    """
    if paths is None:
        return None
    values: list[str] = []
    for path in paths:
        if not isinstance(path, str) or not path:
            raise ValueError(f"paths must contain non-empty strings; got {path!r}")
        top_level = path.split(".", 1)[0].split("[", 1)[0]
        if top_level:
            values.append(top_level)
    return tuple(dict.fromkeys(values))


def create_partial_master(
    source_master: Path,
    destination_master: Path,
    requested_ids: Iterable[str],
) -> tuple[str, ...]:
    """Copy ``source_master`` and retain only requested IDS external links.

    ``dataset_description`` is retained when it is linked because IMAS readers
    use its data-entry metadata.  The source (including any cached master) is
    never changed.
    """
    requested_files = {f"{name.removesuffix('.h5')}.h5" for name in requested_ids}
    requested_files.add("dataset_description.h5")
    destination_master.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_master, destination_master)
    retained: list[str] = []
    with h5py.File(destination_master, "r+") as master:
        for name in list(master):
            link = master.get(name, getlink=True)
            if not isinstance(link, h5py.ExternalLink) or not link.filename.endswith(
                ".h5"
            ):
                continue
            filename = Path(link.filename).name
            if filename not in requested_files:
                del master[name]
            else:
                retained.append(filename)
    return tuple(sorted(set(retained)))


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class HSDSDomainCache:
    """Persistent validated cache for HSDS HDF5 images of one shot.

    The cache is bounded: after every write, least-recently-used domains are
    evicted down to the cap ``cache_max`` resolves to (see
    :mod:`vaft.database.hsds_cache`; default 1 GiB). A cap of ``0`` disables
    the cache like ``cache="off"``.
    """

    def __init__(
        self,
        directory: str,
        shot: int,
        cache: str | Path = "auto",
        cache_max: int | str | None = None,
    ) -> None:
        self.directory = directory.strip("/")
        self.shot = int(shot)
        self.enabled = cache != "off"
        self.root: Path | None = None
        self.base: Path | None = None
        self.max_bytes: int | None = None
        self._manifest: dict[str, Any] = {"schema_version": 1, "domains": {}}
        if self.enabled:
            self.max_bytes = self._resolve_cap(cache_max)
            self.enabled = self.max_bytes != 0
        if self.enabled:
            self.base = hsds_cache.cache_base(cache)
            self.root = self._cache_root(cache)
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                hsds_cache.mark_owned(self.base)
            except OSError as error:  # a read-only or foreign base still caches
                hsds_cache.report_once(
                    f"marker:{self.base}",
                    "cannot mark %s as a VAFT cache (%s); nothing is evicted there",
                    self.base, error,
                )
            self._load_manifest()

    @staticmethod
    def _resolve_cap(cache_max: int | str | None) -> int | None:
        """The cap; a bad *argument* raises, a bad environment or config file warns.

        An unreadable ``$VAFT_HSDS_CACHE_MAX`` or config file is not the
        caller's mistake and must not fail every load: the default applies.
        """
        if cache_max is not None:
            return hsds_cache.resolve_cache_max(cache_max).bytes
        try:
            return hsds_cache.resolve_cache_max().bytes
        except ValueError as error:
            hsds_cache.report_once(
                f"config:{error}",
                "HSDS cache cap setting ignored (%s); using the %s default",
                error, hsds_cache.format_size(hsds_cache.DEFAULT_CACHE_MAX),
            )
            return hsds_cache.DEFAULT_CACHE_MAX

    @staticmethod
    def _endpoint() -> str:
        _require_h5pyd()
        try:
            return str(h5pyd.getServerInfo().get("endpoint") or "default")
        except Exception:
            # A failed revision probe below will force hsget, which provides the
            # actionable transport exception.  Do not reuse a cache under an
            # endpoint we could not identify.
            return "unavailable"

    def _cache_root(self, cache: str | Path) -> Path:
        endpoint_hash = hashlib.sha256(self._endpoint().encode()).hexdigest()[:16]
        base = hsds_cache.cache_base(cache)
        return base / endpoint_hash / self.directory / str(self.shot)

    @property
    def manifest_path(self) -> Path:
        assert self.root is not None
        return self.root / "manifest.json"

    def _load_manifest(self) -> None:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("domains"), dict):
                self._manifest = payload
        except FileNotFoundError:
            # Gone (eviction removed the shot's last domain): so are its records.
            self._manifest = {"schema_version": 1, "domains": {}}
        except (OSError, ValueError):
            pass

    def _write_manifest(self) -> None:
        if not self.enabled:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Per process and thread: two loaders of one shot must not share a temporary.
        temporary = self.manifest_path.with_name(
            f"manifest.json.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n"
        , encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _remote_info(self, remote_uri: str) -> dict[str, str | None]:
        _require_h5pyd()
        with h5pyd.File(remote_uri, "r") as remote:
            return {
                "remote_uri": remote_uri,
                "last_modified": _stringify(getattr(remote, "modified", None)),
                "backend_version": _stringify(remote.attrs.get("HDF5_BACKEND_VERSION")),
            }

    @staticmethod
    def _valid_local(
        path: Path, record: dict[str, Any], remote: dict[str, str | None]
    ) -> bool:
        # A cache must not become authoritative merely because a server omitted
        # revision metadata. In that case materialize a fresh domain instead.
        if remote.get("last_modified") is None:
            return False
        if (
            not path.is_file()
            or int(record.get("local_size", -1)) != path.stat().st_size
        ):
            return False
        if any(
            record.get(key) != remote.get(key)
            for key in ("remote_uri", "last_modified", "backend_version")
        ):
            return False
        try:
            with h5py.File(path, "r"):
                pass
        except OSError:
            return False
        return True

    def materialize(
        self,
        filename: str,
        target_dir: Path,
        *,
        transport: Literal["auto", "canonical", "h5image"] = "auto",
    ) -> tuple[Path, bool, str, dict[str, Any]]:
        """Put one domain file in ``target_dir`` and report its transport."""
        if not filename.endswith(".h5") or Path(filename).name != filename:
            raise ValueError(f"Expected a simple HDF5 filename, got {filename!r}")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        remote_uri = f"hdf5://{self.directory}/{self.shot}/{filename}"
        remote: dict[str, str | None] | None = None
        try:
            remote = self._remote_info(remote_uri)
        except Exception:
            # Required behavior: an unavailable revision probe never authorizes
            # a cache hit.  hsget below is allowed to surface its detailed error.
            pass

        if self.enabled and remote is not None:
            assert self.root is not None and self.base is not None
            cached = self.root / filename
            record = self._manifest["domains"].get(filename, {})
            with hsds_cache.cache_lock(self.base, exclusive=False):
                hit = self._valid_local(cached, record, remote)
                if hit:
                    shutil.copy2(cached, target)
                    hsds_cache.touch(cached)  # the LRU stamp
            if hit:
                return (
                    target,
                    True,
                    "local-cache",
                    {"logical_bytes": target.stat().st_size},
                )

        transfer: dict[str, Any] = {}
        used_transport = "canonical"
        if transport in {"auto", "h5image"}:
            if remote is None or remote.get("last_modified") is None:
                if transport == "h5image":
                    raise H5ImageUnavailableError(
                        f"Cannot validate a derived image without canonical revision metadata: {remote_uri}"
                    )
            else:
                try:
                    transfer = materialize_image(
                        self.directory,
                        self.shot,
                        filename,
                        target,
                        source_info=remote,
                    )
                    used_transport = "h5image"
                except H5ImageUnavailableError:
                    if transport == "h5image":
                        raise
        if used_transport == "canonical":
            run_hsget(remote_uri, target)
            ensure_imas_hdf5_userblock(target, target_dir)
            transfer = {
                "transport": "canonical",
                "remote_uri": remote_uri,
                "logical_bytes": target.stat().st_size,
            }
        if self.enabled:
            self._store(filename, target, remote_uri, remote, used_transport)
        return target, False, used_transport, transfer

    def _store(
        self,
        filename: str,
        target: Path,
        remote_uri: str,
        remote: dict[str, str | None] | None,
        used_transport: str,
    ) -> None:
        """Copy a fresh domain into the cache, then evict down to the cap."""
        assert self.root is not None and self.base is not None
        size = target.stat().st_size
        if self.max_bytes is not None and size > self.max_bytes:
            hsds_cache.report_too_large(remote_uri, size, self.max_bytes)
            return
        if remote is None:
            try:
                remote = self._remote_info(remote_uri)
            except Exception:
                remote = {
                    "remote_uri": remote_uri,
                    "last_modified": None,
                    "backend_version": None,
                }
        cached = self.root / filename
        # The cache is best-effort: the domain is already in ``target``, so a
        # failed copy-in or eviction is logged and never fails the load.
        try:
            with hsds_cache.cache_lock(self.base, exclusive=False):
                hsds_cache.copy_in(target, cached)
                hsds_cache.touch(cached)  # a just-written domain is the most recent
            # The manifest is read-modify-written exclusively: another loader of
            # this shot may have recorded domains since this one started, and
            # two shared holders would each drop the other's record. Until the
            # record lands the domain is unlisted, so eviction cannot see it.
            with hsds_cache.cache_lock(self.base, exclusive=True):
                self._load_manifest()
                self._manifest["domains"][filename] = {
                    **remote,
                    "local_size": cached.stat().st_size,
                    "transport": used_transport,
                }
                self._write_manifest()
            hsds_cache.enforce(self.base, self.max_bytes)
        except OSError as error:
            logger.warning("could not cache %s: %s", remote_uri, error)


def stage_imas_shot(
    directory: str,
    shot: int,
    staging_dir: Path,
    *,
    requested_ids: Iterable[str] | None,
    cache: str | Path = "auto",
    transport: Literal["auto", "canonical", "h5image"] = "auto",
    cache_max: int | str | None = None,
) -> dict[str, Any]:
    """Materialize a complete or IDS-selective local IMAS HDF5 shot.

    The returned dictionary is deliberately small and JSON-friendly so callers
    and benchmarks can report the download plan and cache state.
    """
    if transport not in {"auto", "canonical", "h5image"}:
        raise ValueError("transport must be 'auto', 'canonical', or 'h5image'")
    selected_ids = (
        None if requested_ids is None else tuple(dict.fromkeys(requested_ids))
    )
    cache_store = HSDSDomainCache(directory, shot, cache, cache_max)
    started = time.perf_counter()
    master, master_hit, master_transport, master_transfer = cache_store.materialize(
        "master.h5", staging_dir, transport=transport
    )
    master_fetch_seconds = time.perf_counter() - started
    started = time.perf_counter()
    linked = external_h5_links(master)
    domain_resolution_seconds = time.perf_counter() - started
    partial_master_seconds = 0.0
    if selected_ids is None:
        selected = linked
    else:
        selected_names = tuple(
            name.removesuffix(".h5") for name in selected_ids
        )
        requested_files = {f"{name}.h5" for name in selected_names}
        missing = sorted(requested_files - set(linked))
        if missing:
            available = ", ".join(path[:-3] for path in linked) or "(none)"
            raise FileNotFoundError(
                f"IDS not stored for shot {shot} in '{directory}': "
                f"{', '.join(path[:-3] for path in missing)}. Available IDS: {available}."
            )
        started = time.perf_counter()
        partial = staging_dir / "master.partial.h5"
        retained = create_partial_master(master, partial, selected_names)
        master.unlink()
        partial.replace(master)
        selected = list(retained)
        partial_master_seconds = time.perf_counter() - started

    hits = {"master.h5": master_hit}
    transports = {"master.h5": master_transport}
    transfers = {"master.h5": master_transfer}
    domain_fetch_seconds: dict[str, float] = {}
    for filename in selected:
        started = time.perf_counter()
        _, hits[filename], transports[filename], transfers[filename] = (
            cache_store.materialize(filename, staging_dir, transport=transport)
        )
        domain_fetch_seconds[filename] = time.perf_counter() - started
    return {
        "files": ["master.h5", *selected],
        "cache_hits": hits,
        "transports": transports,
        "transfers": transfers,
        "selected_ids": None if selected_ids is None else list(selected_ids),
        "timings": {
            "master_fetch_seconds": master_fetch_seconds,
            "domain_resolution_seconds": domain_resolution_seconds,
            "partial_master_seconds": partial_master_seconds,
            "domain_fetch_seconds": domain_fetch_seconds,
        },
    }
