"""
IMAS HDF5 (ids image) HSDS interface module.

This module provides `save` and `load` APIs similar to `vaft.database.ods`,
but operates on IMAS HDF5 image files directly (e.g. `imas.h5`) without OMAS
conversion.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import h5py
import requests
import imas
import os

try:
    import h5pyd
except ImportError:
    h5pyd = None  # optional: pip install h5pyd==0.20.0 --no-deps

_H5PYD_MSG = (
    "h5pyd is required for HSDS support. Install with: pip install h5pyd==0.20.0 --no-deps"
)


def _require_h5pyd() -> None:
    if h5pyd is None:
        raise ImportError(_H5PYD_MSG)


def is_connect() -> bool:
    """Return True if HSDS connection is ready."""
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)
    try:
        return h5pyd.getServerInfo().get("state") == "READY"
    except requests.exceptions.ConnectTimeout:
        return False


def _download_remote_image(remote_uri: str, out_path: Path) -> None:
    """Download HSDS domain to a local HDF5 file via hsget."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["hsget", remote_uri, str(out_path)]
    subprocess.run(command, capture_output=False, text=True, check=True)
    return out_path

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
    env: str = "server",
    path: Optional[Union[str, Path]] = None,
    dd_version: Optional[str] = None,
) -> Optional[str]:
    """
    Save a local IMAS HDF5 image file (e.g. `imas.h5`) to HSDS.

    Args:
        local_file: local input HDF5 file path.
        shot: shot number used for default remote filename.
        filename: remote filename under HSDS root. Default: `{shot}/imas.h5`.
        env: `server` to upload HSDS, `local` to only validate local file.
    Returns:
        HSDS URI string when uploaded, otherwise local path.
    """
    logging.getLogger().setLevel(logging.WARNING)

    ids_name = _ids_top_level_name(ids)
    filename = f"{ids_name}.h5"

    if env == "local":
        if path is None:
            path = f'~/public/imasdb/VEST/3/{str(shot)}/1'
        with imas.DBEntry("imas:hdf5?path="+str(path), "w") as dbentry:
            dbentry.put(ids)
    elif env == "server":
        _ref_dir = f"hsds_tmp_{int(shot)}"
        os.makedirs(_ref_dir, exist_ok=True)
        with imas.DBEntry("imas:hdf5?path=" + _ref_dir, "w") as dbentry:
            dbentry.put(ids)

    _require_h5pyd()

    if not is_connect():
        raise ConnectionError("Connection to HSDS server failed")

    username = (
        "public"
        if h5pyd.getServerInfo()["username"] == "admin"
        else h5pyd.getServerInfo()["username"]
    )
    remote_uri = f"hdf5://{username}/{shot}/{filename}"
    # if os.path.exists(f"hdf5://{username}/{shot}/") is False:
    #     command = ['hstouch', f"hdf5://{username}/{shot}/"]
    #     subprocess.run(command, capture_output=False, text=True, check=True)
    command = ["hsload", str(_ref_dir)+"/"+filename, remote_uri]
    subprocess.run(command, capture_output=False, text=True, check=True)
    return remote_uri


def load(
    shot: int,
    ids_name: Union[str, list[str]],
    directory: str = "public",
    filename: Optional[str] = None,
    occurrence: int = 0,
    dd_version: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Union[object, dict[str, object]]:
    """
    Load IDS object(s) from HSDS as native IMAS objects.

    Workflow:
    1) Download HSDS h5image to local `master.h5` using hsget
    2) Open with `imas.DBEntry("imas:hdf5?path=...","r")`
    3) Return `dbentry.get(ids_name, occurrence)` result(s)

    Args:
        shot: shot number used for default remote filename.
        ids_name: IDS name or list of IDS names to load.
        directory: HSDS directory/user (e.g. `public`).
        filename: remote filename. Default: `{shot}/imas.h5`.
        occurrence: IDS occurrence index.
        dd_version: IMAS DD version passed to DBEntry.
        local_dir: optional local staging directory. If omitted, temp dir is used.
    Returns:
        Native IMAS IDS object, or dict of IDS objects for list input.
    """
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)

    try:
        import imas
    except ImportError as exc:
        raise ImportError("imas package is required to load native IDS objects") from exc
    if not hasattr(imas, "DBEntry"):
        raise RuntimeError("IMAS AL5 DBEntry is required for load_imas")

    if filename is None:
        filename = f"{int(shot)}/imas.h5"
    remote_uri = f"hdf5://{directory}/{filename}"

    cleanup_stage = local_dir is None
    stage_dir = Path(local_dir) if local_dir is not None else Path(tempfile.mkdtemp(prefix=f"imas_{int(shot)}_"))

    try:
        master_h5 = stage_dir / "equilibrium.h5"
        output_path = _download_remote_image(remote_uri, master_h5)
        print(f"Downloaded HSDS file to local path: {output_path}")

        uri = "imas:hdf5?path=" + str(stage_dir)
        with imas.DBEntry(uri, "r", dd_version=dd_version) as dbentry:
            if isinstance(ids_name, str):
                return dbentry.get(ids_name, occurrence)
            return {name: dbentry.get(name, occurrence) for name in ids_name}
    finally:
        if cleanup_stage:
            shutil.rmtree(stage_dir, ignore_errors=True)
