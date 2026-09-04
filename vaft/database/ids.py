"""
IMAS HDF5 (ids image) HSDS interface module.

This module provides `save` and `load` APIs similar to `vaft.database.ods`,
but operates on IMAS HDF5 image files directly (e.g. `imas.h5`) without OMAS
conversion.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Optional, Union
import imas

from ..compat import temporary_directory
from .utils import (
    _require_h5pyd,
    ensure_imas_hdf5_userblock,
    is_connect,
    require_source_exists,
)
from .transport import run_hsget, run_hsload, verify_uploaded_image
from .h5image import publish_image
from .sources import resolve as resolve_source
from .staging import external_h5_links, stage_imas_shot


def _download_remote_image(remote_uri: str, out_path: Path) -> Path:
    """Download HSDS domain to a local HDF5 file via hsget."""
    run_hsget(remote_uri, out_path)
    ensure_imas_hdf5_userblock(out_path, out_path.parent)
    return out_path


def _external_h5_links(master_path: Path) -> list[str]:
    """Return HDF5 filenames linked by an IMAS master.h5 file."""
    return external_h5_links(master_path)


def _ids_top_level_name(ids_obj):
    # 1) IMAS IDS
    name = getattr(getattr(ids_obj, "metadata", None), "name", None)
    if name:
        return name

    # 2) fallback: use class name as IDS name (e.g. equilibrium, pf_active)
    cls = type(ids_obj).__name__
    return cls


def save(
    ids: imas.ids_toplevel.IDSToplevel,
    shot: int,
    source: Optional[str] = None,
    dd_version: Optional[str] = None,
    *,
    directory: Optional[str] = None,
    derived_cache: Literal["auto", "none", "imas-images", "omas", "both"] = "auto",
) -> Optional[str]:
    """
    Save one native IMAS IDS to a remote HSDS namespace.

    Workflow for env="server":
    1) Save IDS to local shot directory via imas.DBEntry (auto-generates master.h5)
    2) Upload IDS file ({ids_name}.h5) to HSDS
    3) Upload master.h5 aggregator to HSDS
    4) Clean up local staging

    Args:
        ids: IMAS IDS object to save.
        shot: shot number.
        source: Named HSDS source to publish into. Defaults to ``main``; the
            read-only legacy ``public`` source is refused.
        dd_version: IMAS DD version.
        directory: Deprecated alias for ``source``.

    A native IDS carries no ``dataset_description``, so unlike
    :func:`vaft.database.ods.save_ods` this path records no source provenance
    inside the payload; the namespace is the URI it is written to.
    Returns:
        HSDS URI string for the IDS file when uploaded, otherwise local path.
    """
    logging.getLogger().setLevel(logging.WARNING)

    source = resolve_source(source, directory=directory, writable=True)
    ids_name = _ids_top_level_name(ids)
    filename = f"{ids_name}.h5"
    if derived_cache not in {"auto", "none", "imas-images", "omas", "both"}:
        raise ValueError(
            "derived_cache must be 'auto', 'none', 'imas-images', 'omas', or 'both'"
        )
    if derived_cache in {"omas", "both"}:
        raise ValueError(
            "native IDS save supports only 'auto', 'none', or 'imas-images' derived_cache"
        )
    derived_mode = "imas-images" if derived_cache == "auto" else derived_cache

    _require_h5pyd()

    if not is_connect():
        raise ConnectionError("Connection to HSDS server failed")
    require_source_exists(source)

    # Not tempfile.TemporaryDirectory: imas_core's HDF5 backend keeps IDS files
    # open after DBEntry.close() returns, so on Windows cleanup would raise
    # WinError 32 and fail a transfer that already succeeded.
    with temporary_directory(prefix="hsds_tmp_") as _staging_dir:
        print(f"[INFO] Local staging directory: {_staging_dir.absolute()}")

        # Save IDS to local shot directory. This auto-generates master.h5.
        with imas.DBEntry(
            "imas:hdf5?path=" + str(_staging_dir), "w", dd_version=dd_version
        ) as dbentry:
            dbentry.put(ids)
        print(f"[INFO] Saved {filename} to local: {_staging_dir / filename}")

        # Upload to explicitly requested HSDS namespace.
        ids_remote_uri = f"hdf5://{source}/{shot}/{filename}"
        run_hsload(_staging_dir / filename, ids_remote_uri)
        verify_uploaded_image(_staging_dir / filename, ids_remote_uri)
        master_remote_uri = f"hdf5://{source}/{shot}/master.h5"
        run_hsload(_staging_dir / "master.h5", master_remote_uri)
        verify_uploaded_image(_staging_dir / "master.h5", master_remote_uri)
        if derived_mode == "imas-images":
            time.sleep(8.0)
            for local_path in (_staging_dir / filename, _staging_dir / "master.h5"):
                try:
                    result = publish_image(
                        local_path,
                        source,
                        int(shot),
                        imas_version=dd_version,
                    )
                    print(f"[INFO] Published derived IMAS image: {result['uri']}")
                except Exception as exc:
                    logging.warning(
                        "Could not publish derived IMAS image for shot %s (%s): %s",
                        shot,
                        local_path.name,
                        exc,
                    )
        return ids_remote_uri


def load(
    shot: int,
    ids_name: Union[str, list[str]],
    source: Optional[str] = None,
    occurrence: int | dict[str, int] = 0,
    dd_version: Optional[str] = None,
    local_dir: Optional[str] = None,
    cache: str | Path = "auto",
    transport: Literal["auto", "canonical", "h5image"] = "auto",
    *,
    directory: Optional[str] = None,
) -> Union[object, dict[str, object]]:
    """
    Load IDS object(s) from HSDS as native IMAS objects.

    Workflow:
    1) Download requested IDS files (equilibrium.h5, pf_active.h5, etc.) from HSDS
    2) Download master.h5 which aggregates all IDS information
    3) Open with `imas.DBEntry("imas:hdf5?path=...","r")`
    4) Return `dbentry.get(ids_name, occurrence)` result(s)

    Args:
        shot: shot number.
        ids_name: IDS name (str) or list of IDS names to load.
        source: Named HSDS source holding the shot. Defaults to ``main``.
        occurrence: IDS occurrence index.
        dd_version: IMAS DD version passed to DBEntry.
        local_dir: optional local staging directory. If omitted, temp dir is used.
        cache: ``"auto"`` (default) stores validated domains locally, ``"off"``
            always downloads, or a path selects a cache base directory.
        directory: Deprecated alias for ``source``.
    Returns:
        Native IMAS IDS object, or dict of IDS objects for list input.
    """
    logging.getLogger().setLevel(logging.WARNING)

    source = resolve_source(source, directory=directory)

    try:
        import imas
    except ImportError as exc:
        raise ImportError(
            "imas package is required to load native IDS objects"
        ) from exc
    if not hasattr(imas, "DBEntry"):
        raise RuntimeError("IMAS AL5 DBEntry is required for load_imas")

    # Use a real temp directory when no local_dir is provided so we never
    # accidentally delete unrelated user data.  nullcontext just passes the
    # caller-supplied directory through without any cleanup.
    staging_ctx = (
        temporary_directory(prefix="hsds_tmp_")
        if local_dir is None
        else nullcontext(local_dir)
    )

    with staging_ctx as staging_base:
        shot_dir = Path(staging_base) / str(int(shot))
        shot_dir.mkdir(parents=True, exist_ok=True)

        # Normalize ids_name to list
        ids_list = [ids_name] if isinstance(ids_name, str) else ids_name
        print(f"[INFO] Creating local staging directory: {shot_dir.absolute()}")

        plan = stage_imas_shot(
            directory=source,
            shot=shot,
            staging_dir=shot_dir,
            requested_ids=ids_list,
            cache=cache,
            transport=transport,
        )
        print(
            "[INFO] Materialized IMAS files: "
            + ", ".join(plan["files"])
            + f" (cache hits: {sum(plan['cache_hits'].values())}/{len(plan['cache_hits'])})"
        )

        # Open and load from local files
        uri = "imas:hdf5?path=" + str(shot_dir)
        print(f"[INFO] Loading from: {shot_dir.absolute()}")
        try:
            with imas.DBEntry(uri, "r", dd_version=dd_version) as dbentry:
                if isinstance(ids_name, str):
                    value = (
                        occurrence.get(ids_name, occurrence.get("*", 0))
                        if isinstance(occurrence, dict)
                        else occurrence
                    )
                    return dbentry.get(ids_name, value)
                return {
                    name: dbentry.get(
                        name,
                        occurrence.get(name, occurrence.get("*", 0))
                        if isinstance(occurrence, dict)
                        else occurrence,
                    )
                    for name in ids_name
                }
        except Exception as exc:
            if "different major version" in str(exc):
                # This is a DD-version contract error, not an incomplete
                # partial master. Preserve IMAS' actionable original message.
                raise
            raise RuntimeError(
                "IMAS could not open the selective partial master for "
                f"shot {shot}; required external links are dataset_description "
                f"and: {', '.join(ids_list)}. Check the stored master.h5 links. "
                f"Original IMAS error: {exc}"
            ) from exc
