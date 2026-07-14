"""
IMAS HDF5 (ids image) HSDS interface module.

This module provides `save` and `load` APIs similar to `vaft.database.ods`,
but operates on IMAS HDF5 image files directly (e.g. `imas.h5`) without OMAS
conversion.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union
import h5py
import imas

try:
    import h5pyd
except ImportError:
    h5pyd = None  # optional: pip install h5pyd==0.20.0 --no-deps

from .utils import _require_h5pyd, ensure_imas_hdf5_userblock, is_connect


def _download_remote_image(remote_uri: str, out_path: Path) -> Path:
    """Download HSDS domain to a local HDF5 file via hsget."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["hsget", remote_uri, str(out_path)]
    subprocess.run(command, capture_output=False, text=True, check=True)
    ensure_imas_hdf5_userblock(out_path, out_path.parent)
    return out_path


def _external_h5_links(master_path: Path) -> list[str]:
    """Return HDF5 filenames linked by an IMAS master.h5 file."""
    filenames = []
    with h5py.File(master_path, "r") as master:
        for name in master:
            link = master.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink) and link.filename.endswith(".h5"):
                filenames.append(Path(link.filename).name)
    return sorted(set(filenames))

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
    Save a local IMAS HDF5 image file to HSDS.

    Workflow for env="server":
    1) Save IDS to local shot directory via imas.DBEntry (auto-generates master.h5)
    2) Upload IDS file ({ids_name}.h5) to HSDS
    3) Upload master.h5 aggregator to HSDS
    4) Clean up local staging

    Args:
        ids: IMAS IDS object to save.
        shot: shot number.
        env: `server` to upload HSDS, `local` to only validate local file.
        path: local path for `env="local"`. Default: `~/public/imasdb/VEST/3/{shot}/1`.
        dd_version: IMAS DD version.
    Returns:
        HSDS URI string for the IDS file when uploaded, otherwise local path.
    """
    logging.getLogger().setLevel(logging.WARNING)

    ids_name = _ids_top_level_name(ids)
    filename = f"{ids_name}.h5"

    if env == "local":
        if path is None:
            path = f'~/public/imasdb/VEST/3/{str(shot)}/1'
        expanded_path = Path(path).expanduser()
        expanded_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Local storage path: {expanded_path.absolute()}")
        with imas.DBEntry("imas:hdf5?path="+str(expanded_path), "w", dd_version=dd_version) as dbentry:
            dbentry.put(ids)
        print(f"[INFO] Saved {filename} to: {expanded_path / filename}")
        print(f"[INFO] Saved master.h5 to: {expanded_path / 'master.h5'}")
        return str(expanded_path)
    elif env == "server":
        _require_h5pyd()

        if not is_connect():
            raise ConnectionError("Connection to HSDS server failed")

        with tempfile.TemporaryDirectory(prefix="hsds_tmp_") as tmp_dir:
            _staging_dir = Path(tmp_dir)
            print(f"[INFO] Local staging directory: {_staging_dir.absolute()}")

            # Save IDS to local shot directory
            # This auto-generates master.h5 along with {ids_name}.h5
            with imas.DBEntry("imas:hdf5?path=" + str(_staging_dir), "w", dd_version=dd_version) as dbentry:
                dbentry.put(ids)
            print(f"[INFO] Saved {filename} to local: {_staging_dir / filename}")
            print(f"[INFO] Saved master.h5 to local: {_staging_dir / 'master.h5'}")

            # Upload to HSDS
            username = (
                "public"
                if h5pyd.getServerInfo()["username"] == "admin"
                else h5pyd.getServerInfo()["username"]
            )

            # Upload IDS-specific file
            ids_remote_uri = f"hdf5://{username}/{shot}/{filename}"
            command = ["hsload", str(_staging_dir / filename), ids_remote_uri]
            subprocess.run(command, capture_output=False, text=True, check=True)
            print(f"[INFO] Uploaded {filename} to {ids_remote_uri}")

            # Upload master.h5 (aggregator file)
            master_remote_uri = f"hdf5://{username}/{shot}/master.h5"
            command = ["hsload", str(_staging_dir / "master.h5"), master_remote_uri]
            subprocess.run(command, capture_output=False, text=True, check=True)
            print(f"[INFO] Uploaded master.h5 to {master_remote_uri}")

            return ids_remote_uri


def load(
    shot: int,
    ids_name: Union[str, list[str]],
    directory: str = "public",
    occurrence: int = 0,
    dd_version: Optional[str] = None,
    local_dir: Optional[str] = None,
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
        directory: HSDS directory/user (e.g. `public`).
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

    # Use a real temp directory when no local_dir is provided so we never
    # accidentally delete unrelated user data.  nullcontext just passes the
    # caller-supplied directory through without any cleanup.
    staging_ctx = (
        tempfile.TemporaryDirectory(prefix="hsds_tmp_")
        if local_dir is None
        else nullcontext(local_dir)
    )

    with staging_ctx as staging_base:
        shot_dir = Path(staging_base) / str(int(shot))
        shot_dir.mkdir(parents=True, exist_ok=True)

        # Normalize ids_name to list
        ids_list = [ids_name] if isinstance(ids_name, str) else ids_name
        print(f"[INFO] Creating local staging directory: {shot_dir.absolute()}")

        # Download master.h5 first. IMAS-Core expects every external file linked
        # by master.h5 to be present when opening the data entry, even when the
        # caller only asks for one IDS.
        master_remote_uri = f"hdf5://{directory}/{shot}/master.h5"
        master_local = shot_dir / "master.h5"
        print(f"[INFO] Downloading master.h5 from {master_remote_uri}...")
        _download_remote_image(master_remote_uri, master_local)
        print(f"[INFO] Saved to: {master_local.absolute()}")

        linked_files = _external_h5_links(master_local)
        requested_files = [f"{ids}.h5" for ids in ids_list]
        for filename in sorted(set(linked_files + requested_files)):
            local_file = shot_dir / filename
            if local_file.exists():
                continue
            remote_uri = f"hdf5://{directory}/{shot}/{filename}"
            print(f"[INFO] Downloading {filename} from {remote_uri}...")
            _download_remote_image(remote_uri, local_file)
            print(f"[INFO] Saved to: {local_file.absolute()}")

        # Open and load from local files
        uri = "imas:hdf5?path=" + str(shot_dir)
        print(f"[INFO] Loading from: {shot_dir.absolute()}")
        with imas.DBEntry(uri, "r", dd_version=dd_version) as dbentry:
            if isinstance(ids_name, str):
                return dbentry.get(ids_name, occurrence)
            return {name: dbentry.get(name, occurrence) for name in ids_name}
