"""
Common utilities and helper functions for VEST database operations.

This module provides shared functionality used by multiple database interfaces
(ODS, IDS, raw) such as shot listing and file existence checking.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, List, Union

import h5py
import h5pyd
import numpy as np
import pandas as pd
import requests
import urllib3

# Must match the path the corrective updaters write to
# (workflow/automatic_pipeline_2_corrective_data_update/*.py), otherwise
# exist_shot() reads a stale copy and never reflects reprocessing.
PROCESSED_H5_PATH = "hdf5://public/processed_shots.h5"


def _require_h5pyd() -> None:
    """Compatibility guard for HSDS modules; h5pyd is a core dependency."""


def ensure_imas_hdf5_userblock(path: Union[str, Path], entry_dir: Union[str, Path]) -> None:
    """Repair an IMAS HDF5 image downloaded from HSDS so IMAS-Core can open it.

    HSDS ``hsget`` recreates the HDF5 data model but mangles two things IMAS-Core
    needs:

    1. The ``HDF5_BACKEND_VERSION`` root attribute is rebuilt as a short ASCII
       fixed-string; IMAS-Core's C reader expects the 20-byte UTF-8 fixed-string
       its writer produces and otherwise fails with
       ``Unable to read attribute: HDF5_BACKEND_VERSION``. We rewrite it to the
       expected datatype.
    2. The 1024-byte userblock IMAS stores the data-entry path in is dropped. We
       prefix the existing HDF5 bytes to restore it (preserving the internal
       layout).

    Both repairs are idempotent.
    """
    file_path = Path(path)

    # (1) normalize the HDF5_BACKEND_VERSION attribute datatype
    try:
        with h5py.File(file_path, "r+") as handle:
            if "HDF5_BACKEND_VERSION" not in handle.attrs:
                return
            type_id = handle.attrs.get_id("HDF5_BACKEND_VERSION").get_type()
            if type_id.get_size() != 20 or type_id.get_cset() != h5py.h5t.CSET_UTF8:
                value = (
                    np.asarray(handle.attrs["HDF5_BACKEND_VERSION"]).tobytes().rstrip(b"\x00")
                    or b"1.0"
                )
                del handle.attrs["HDF5_BACKEND_VERSION"]
                handle.attrs.create(
                    "HDF5_BACKEND_VERSION",
                    value,
                    dtype=h5py.string_dtype(encoding="utf-8", length=20),
                )
            userblock_size = handle.userblock_size
    except OSError:
        return

    # (2) restore the userblock if HSDS stripped it
    if userblock_size >= 1024:
        return

    entry_path = str(Path(entry_dir).resolve()).encode()[:1023] + b"\0"
    userblock = entry_path.ljust(1024, b"\0")
    tmp_path = file_path.with_name(file_path.name + ".userblock_tmp")

    with open(tmp_path, "wb") as dst:
        dst.write(userblock)
        with open(file_path, "rb") as src:
            shutil.copyfileobj(src, dst)
    tmp_path.replace(file_path)


def is_connect() -> bool:
    """
    Check if the user is connected to the HSDS server.

    Returns:
        bool: True if connected and server is READY, False otherwise.
    """
    logging.getLogger().setLevel(logging.WARNING)
    try:
        return h5pyd.getServerInfo().get("state") == "READY"
    except requests.exceptions.ConnectTimeout:
        return False


def _get_public_folders(sort: int = 0) -> List[str]:
    """Get numeric folder names (shot numbers) from ``public`` with optional sorting."""
    try:
        folder = list(h5pyd.Folder("/public/"))
        folder_list = [item for item in folder if not item.endswith('.h5') and item.isdigit()]

        if sort == 0:
            pass
        elif sort == 1:
            folder_list = sorted(folder_list, key=int)
        elif sort == -1:
            folder_list = sorted(folder_list, key=int, reverse=True)
        else:
            print(f"[WARNING] Invalid sort value: {sort}. Using default (0, no sorting)")

        print(folder_list)
        return folder_list
    except urllib3.exceptions.MaxRetryError:
        print("Connection error")
        return []


def _get_folder_contents(folder_name: str, sort: int = -1) -> List[str]:
    """Get all contents from custom folder with sorting."""
    try:
        folder = list(h5pyd.Folder("/" + folder_name + "/"))

        if sort == 0:
            file_list = list(folder)
        elif sort == 1:
            file_list = sorted(folder)
        elif sort == -1:
            file_list = sorted(folder, reverse=True)
        else:
            print(f"[WARNING] Invalid sort value: {sort}. Using default (-1, descending)")
            file_list = sorted(folder, reverse=True)

        print(file_list)
        return file_list
    except urllib3.exceptions.MaxRetryError:
        print("Connection error")
        return []


def exist_shot(
    username: Optional[str] = None,
    shot: Optional[int] = None,
    data_filter: Optional[str] = None,
    sort: int = -1,
) -> Union[List[str], bool, pd.DataFrame, None]:
    """Return a list of shot names or processed-diagnostic data from HSDS.

    Supports multiple filter options and folder-specific behaviors:
    - None (default): Standard ODS/IDS shots from ``public`` directory
    - 'ts' or 'thomson_scattering': Thomson scattering processed shots from processed_shots.h5
    - 'cx' or 'charge_exchange': Charge exchange (IDS/CES) processed shots from processed_shots.h5
    - 'cp' or 'core_profiles': Kinetic core_profiles built shots from processed_shots.h5

    Note: the 'ts'/'cx'/'cp' tables are read from the corrective-pipeline registry
    (``processed_shots.h5``), which logs what the updater *attempted*. A row does
    not guarantee the shot ODS actually carries that diagnostic — check the
    ``Status`` column ('invalid' means the run found nothing usable). Each updater
    writes its own registry group (``shots`` / ``cx_shots`` / ``cp_shots``), so the
    three listings are independent and never overwrite one another.

    Args:
        username (str, optional): The folder to access.
            Defaults to 'public'. Options: 'public' or other folder names.
        shot (int, optional): The specific shot number to search for. Only used with ODS filter.
        data_filter (str, optional): Filter type - None for ODS, 'ts'/'thomson_scattering'
            for Thomson scattering, 'cx'/'charge_exchange' for charge exchange,
            'cp'/'core_profiles' for kinetic core_profiles.
        sort (int, optional): Sort order for shot listings.
            - 1: Ascending (oldest first)
            - -1: Descending (newest first)
            - 0: No sorting

    Returns:
        Union[List[str], bool, pd.DataFrame, None]:
            - For ODS (data_filter=None):
              - List of folder/file names if no shot parameter
              - True if specific shot exists
              - False if specific shot does not exist
              - Empty list on connection error
            - For Thomson Scattering / Charge Exchange / core_profiles:
              - pd.DataFrame with columns: Index, Shot Number, Last Processed, Status
              - None on error or no matching data
    """
    logging.getLogger().setLevel(logging.WARNING)

    _filter = data_filter.lower().strip() if isinstance(data_filter, str) else data_filter

    # Handle Thomson Scattering filter
    if _filter in ('ts', 'thomson_scattering', 'thomson scattering'):
        return _exist_shot_ts(sort=sort != 0)

    # Handle Charge Exchange (IDS / CES) filter
    if _filter in ('cx', 'charge_exchange', 'charge exchange', 'ces', 'ids'):
        return _exist_shot_cx(sort=sort != 0)

    # Handle kinetic core_profiles filter
    if _filter in ('cp', 'core_profiles', 'core_profile', 'kinetic',
                   'kinetic_core_profile'):
        return _exist_shot_cp(sort=sort != 0)

    # Default folder
    if username is None:
        username = 'public'

    # Check for specific shot
    if shot is not None:
        try:
            folder = list(h5pyd.Folder("/" + username + "/"))
            for file in folder:
                if file.split(".")[0] == str(shot):
                    print(file)
                    return True
            print("File does not exist")
            return False
        except urllib3.exceptions.MaxRetryError:
            print("Connection error")
            return False

    # List contents based on folder
    if username == 'public':
        return _get_public_folders(sort=sort)
    return _get_folder_contents(username, sort=sort)


# HDF5 groups inside processed_shots.h5, one per diagnostic. Each updater owns
# its own group so Thomson and charge_exchange records never collide (both are
# keyed by bare shot number). Keep these in sync with the corrective updaters in
# workflow/automatic_pipeline_2_corrective_data_update/.
TS_REGISTRY_GROUP = "shots"       # update_thomson_scattering_and_core_profile.py
CX_REGISTRY_GROUP = "cx_shots"    # update_charge_exchange.py
CP_REGISTRY_GROUP = "cp_shots"    # update_core_profile.py


def _exist_shot_ts(sort: bool = True) -> Union[pd.DataFrame, None]:
    """
    Retrieve all processed Thomson scattering shots from h5pyd in a formatted table.

    Behavior:
    - Reads the ``shots`` group (each shotnumber as subgroup) from processed_shots.h5.
    - Extracts timestamp and status for each shot.
    - Returns DataFrame with columns: Index, Shot Number, Last Processed, Status.

    Args:
        sort (bool, optional): Whether to sort shots by shot number. Defaults to True.

    Returns:
        Union[pd.DataFrame, None]: DataFrame of TS shots or None on error.
    """
    return _read_processed_registry(
        sort=sort, group=TS_REGISTRY_GROUP, label="Thomson Scattering"
    )


def _exist_shot_cx(sort: bool = True) -> Union[pd.DataFrame, None]:
    """
    Retrieve processed charge_exchange (IDS/CES) shots from h5pyd in a table.

    Identical in shape and behavior to :func:`_exist_shot_ts` — it just reads the
    charge_exchange registry group (:data:`CX_REGISTRY_GROUP`) instead of the
    Thomson one, so the two listings never collide.

    Args:
        sort (bool, optional): Whether to sort shots by shot number. Defaults to True.

    Returns:
        Union[pd.DataFrame, None]: DataFrame of charge_exchange shots or None.
    """
    return _read_processed_registry(
        sort=sort, group=CX_REGISTRY_GROUP, label="Charge Exchange"
    )


def _exist_shot_cp(sort: bool = True) -> Union[pd.DataFrame, None]:
    """
    Retrieve shots with kinetic core_profiles built, in a table.

    Identical in shape and behavior to :func:`_exist_shot_ts`, reading the
    core_profiles registry group (:data:`CP_REGISTRY_GROUP`). Statuses recorded
    by ``update_core_profile.py``: ``kinetic_core_profile`` (real Ti/Vtor slice),
    ``core_profile`` (electron-only slices), ``thomson_only``, ``invalid``.

    Args:
        sort (bool, optional): Whether to sort shots by shot number. Defaults to True.

    Returns:
        Union[pd.DataFrame, None]: DataFrame of core_profiles shots or None.
    """
    return _read_processed_registry(
        sort=sort, group=CP_REGISTRY_GROUP, label="Core Profiles"
    )


def _read_processed_registry(
    sort: bool = True,
    group: str = TS_REGISTRY_GROUP,
    label: str = "Processed",
) -> Union[pd.DataFrame, None]:
    """
    Read one diagnostic's shot registry group from ``processed_shots.h5``.

    Args:
        sort: Whether to sort shots by shot number.
        group: HDF5 group name to read (e.g. ``shots`` for Thomson,
            ``cx_shots`` for charge_exchange).
        label: Human-readable diagnostic name for printed output.

    Returns:
        DataFrame with columns Index, Shot Number, Last Processed, Status, or
        None on error / when the group is missing or empty.
    """
    logging.getLogger().setLevel(logging.WARNING)

    try:
        with h5pyd.File(PROCESSED_H5_PATH, "r") as f:
            if group not in f:
                print(f"[INFO] No '{group}' group found in processed_shots.h5.")
                return None

            g = f[group]
            if len(g.keys()) == 0:
                print("[INFO] No processed shots recorded yet.")
                return None

            shots, timestamps, status = [], [], []

            # Get keys with optional sorting
            keys = sorted(g.keys(), key=lambda x: int(x)) if sort else list(g.keys())

            for key in keys:
                try:
                    ts = g[key]["timestamp"][...]
                    st = g[key]["status"][...]

                    # Handle numpy array with object dtype containing bytes
                    if isinstance(ts, np.ndarray) and ts.dtype == object:
                        ts = ts.item().decode('utf-8')

                    if isinstance(st, np.ndarray) and st.dtype == object:
                        st = st.item().decode('utf-8')

                except Exception as e:
                    print(f"[WARNING] Error processing shot {key}: {e}")
                    ts, st = "N/A", "unknown"

                shots.append(int(key))
                timestamps.append(ts)
                status.append(st)

        if not shots:
            print(f"[INFO] No {label} shots recorded yet.")
            return None

        df = pd.DataFrame({
            "Index": range(1, len(shots) + 1),
            "Shot Number": shots,
            "Last Processed": timestamps,
            "Status": status
        })

        print(f"Available {label} Shots:\n")
        print(df.to_string(index=False))

        return df

    except Exception as e:
        print(f"[ERROR] Failed to read processed shots: {e}")
        return None
