"""
Common utilities and helper functions for VEST database operations.

This module provides shared functionality used by multiple database interfaces
(ODS, IDS, raw) such as shot listing and file existence checking.
"""

import logging
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import requests
import urllib3

try:
    import h5pyd
except ImportError:
    h5pyd = None  # optional: pip install h5pyd==0.20.0 --no-deps

_H5PYD_MSG = (
    "h5pyd is required for HSDS support. Install with: pip install h5pyd==0.20.0 --no-deps"
)

PROCESSED_H5_PATH = "hdf5://public_omas/processed_shots.h5"


def _require_h5pyd() -> None:
    """Ensure h5pyd is available."""
    if h5pyd is None:
        raise ImportError(_H5PYD_MSG)


def is_connect() -> bool:
    """
    Check if the user is connected to the HSDS server.
    
    Returns:
        bool: True if connected and server is READY, False otherwise.
    """
    _require_h5pyd()
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
    """Return a list of shot names or Thomson scattering data from HSDS.

    Supports multiple filter options and folder-specific behaviors:
    - None (default): Standard ODS/IDS shots from ``public`` directory
    - 'ts' or 'thomson_scattering': Thomson scattering processed shots from processed_shots.h5

    Args:
        username (str, optional): The folder to access. 
            Defaults to 'public'. Options: 'public' or other folder names.
        shot (int, optional): The specific shot number to search for. Only used with ODS filter.
        data_filter (str, optional): Filter type - None for ODS, 'ts' or 'thomson_scattering' for Thomson scattering.
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
            - For Thomson Scattering (data_filter='ts' or 'thomson_scattering'):
              - pd.DataFrame with columns: Index, Shot Number, Last Processed, Status
              - None on error or no data
    """
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)

    # Handle Thomson Scattering filter
    if data_filter in ('ts', 'thomson_scattering', 'thomson scattering'):
        ts_sort = sort != 0
        return _exist_shot_ts(sort=ts_sort)
    
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


def _exist_shot_ts(sort: bool = True) -> Union[pd.DataFrame, None]:
    """
    Retrieve all processed Thomson scattering shots from h5pyd in a formatted table.

    Behavior:
    - Reads 'shots' group (each shotnumber as subgroup) from processed_shots.h5.
    - Extracts timestamp and status for each shot.
    - Returns DataFrame with columns: Index, Shot Number, Last Processed, Status.

    Args:
        sort (bool, optional): Whether to sort shots by shot number. Defaults to True.

    Returns:
        Union[pd.DataFrame, None]: DataFrame of TS shots or None on error.
    """
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)

    try:
        with h5pyd.File(PROCESSED_H5_PATH, "r") as f:
            if "shots" not in f:
                print("[INFO] No 'shots' group found in processed_shots.h5.")
                return None

            g = f["shots"]
            if len(g.keys()) == 0:
                print("[INFO] No processed shots recorded yet.")
                return None

            shots, timestamps, status = [], [], []
            
            # Get keys with optional sorting
            keys = sorted(g.keys(), key=lambda x: int(x)) if sort else list(g.keys())
            
            for key in keys:
                try:
                    shots.append(int(key))
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
                    
                timestamps.append(ts)
                status.append(st)

        df = pd.DataFrame({
            "Index": range(1, len(shots) + 1),
            "Shot Number": shots,
            "Last Processed": timestamps,
            "Status": status
        })

        print("Available Thomson Scattering Shots:\n")
        print(df.to_string(index=False))

        return df

    except Exception as e:
        print(f"[ERROR] Failed to read processed shots: {e}")
        return None
