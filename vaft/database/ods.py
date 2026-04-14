
"""
OMAS HSDS Database Interface Module

This module provides functions for interacting with ODS data stored in HDF5 format,
both locally and on a remote server. It handles operations such as saving, loading,
and checking the existence of ODS files.

Key Features:
- Server connection management
- File existence checking
- ODS data saving (local and server)
- ODS data loading
- HDF5 to ODS conversion utilities

Modification History:
2025-04-30, HS Yun:
    - Added type hints for all functions
    - Improved code formatting and linting
    - Updated docstrings to Google style format
    - Renamed functions to follow snake_case convention
    - Improved string formatting using f-strings
    - Added proper import organization
    - Enhanced error handling and parameter validation
"""

import requests
import urllib3
try:
    import h5pyd
except ImportError:
    h5pyd = None  # optional: pip install h5pyd==0.20.0 --no-deps
import h5py
import omas
import subprocess
import numpy as np
from typing import Optional, Union, List
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import logging
import pandas as pd
from datetime import datetime

from .utils import _require_h5pyd, exist_shot, is_connect

_H5PYD_MSG = (
    "h5pyd is required for HSDS support. Install with: pip install h5pyd==0.20.0 --no-deps"
)

PROCESSED_H5_PATH = "hdf5://public_omas/processed_shots.h5"


def save_ods(
    ods: omas.ODS,
    shot: int,
    filename: Optional[str] = None,
    env: str = 'server'
) -> None:
    """
    Function to save an ODS (Open Data Structure) file either locally or to an HDF5 server.

    This function handles saving the ODS data to an HDF5 file, either on the local file system or
    remotely to an HDF5 server using the `hsload` command. If no filename is provided, the function 
    defaults to using the shot number with an `.h5` extension.

    Parameters:
    ods (omas.ODS): The Open Data Structure (ODS) object that needs to be saved.
    shot (int): The shot number associated with the ODS data, used to generate the filename if not provided.
    filename (str, optional): The name of the file to save the ODS data. Defaults to `None`, which generates a name based on the shot number.
    env (str, optional): The environment where the file will be saved. It can be either 'server' for server upload or 'local' for local storage. Defaults to 'server'.

    Returns:
    None: The function doesn't return any specific value but prints information about the saving process.
    """
    logging.getLogger().setLevel(logging.WARNING)

    if filename is None:
        filename = f"{shot}.h5"

    if env == 'local':
        omas.save_omas_h5(ods, filename)
        return

    _require_h5pyd()
    if not is_connect():
        print('Error: Connection to the server failed')
        return
    
    username = 'public_omas' if h5pyd.getServerInfo()['username'] == 'admin' else h5pyd.getServerInfo()['username']
    file_path = f"hdf5://{username}/{filename}"
    omas.save_omas_h5(ods, filename)

    command = ['hsload', '--h5image', filename, file_path]
    result = subprocess.run(command, capture_output=False, text=True)
    subprocess.run(['rm', filename], capture_output=False, text=True)


def convert_dataset(ods: omas.ODS, data: Union[h5py.Dataset, h5py.Group]) -> None:
    """
    Recursive utility function to map HDF5 structure to ODS

    :param ods: input ODS to be populated

    :param data: HDF5 dataset of group
    """
    keys = data.keys()
    try:
        keys = sorted(list(map(int, keys)))
    except ValueError:
        pass
    for oitem in keys:
        item = str(oitem)
        if item.endswith('_error_upper'):
            continue
        if isinstance(data[item], h5py.Dataset):
            if item + '_error_upper' in data:
                if isinstance(data[item][()], (float, np.floating)):
                    ods.setraw(item, ufloat(data[item][()], data[item + '_error_upper'][()]))
                else:
                    ods.setraw(item, uarray(data[item][()], data[item + '_error_upper'][()]))
            else:
                ods.setraw(item, data[item][()])
        elif isinstance(data[item], h5py.Group):
            convert_dataset(ods.setraw(oitem, ods.same_init_ods()), data[item])

def load_ods(shot: Union[int, List[int]], directory: str = 'public_omas') -> Union[omas.ODS, List[omas.ODS]]:
    """
    Load ODS data from HDF5 file(s).
        
    Parameters:
        shot (Union[int, List[int]]): Shot number(s) to load
        directory (str, optional): Directory path. Defaults to public
    
    Returns:
        Union[omas.ODS, List[omas.ODS]]: ODS object or list of ODS objects
    """
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)

    if isinstance(shot, list):
        ods_list = []
        for s in shot:
            ods = _load_one_shot(int(s), directory)
            print("Successfully loaded ODS data for shot:", s)
            ods_list.append(ods)
        print("Successfully loaded a list of ODS data")
        return ods_list

    else:
        s = int(shot)
        ods = _load_one_shot(s, directory)
        print("Successfully loaded ODS data for shot:", s)
        return ods
    
def _load_one_shot(shot: int, directory: str) -> omas.ODS:
    logging.getLogger().setLevel(logging.WARNING)

    filename = f'hdf5://{directory}/{shot}.h5'
    print(f"Attempting to load ODS data from: {filename}")

    # 1) 기본 로드 먼저 시도
    ods = omas.ODS()
    try:
        with h5py.File(h5pyd.H5Image(filename)) as data:
            convert_dataset(ods, data)
        return ods

    # 2) 배열 인덱스 튀는 케이스만 dynamic으로 재시도
    except IndexError as e:
        # 원인 메시지에 time_slice[...] 같은 게 들어오면 거의 이 케이스
        ods = omas.ODS()
        with omas.omas_environment(ods, dynamic_path_creation='dynamic_array_structures'):
            with h5py.File(h5pyd.H5Image(filename)) as data:
                convert_dataset(ods, data)
        return ods