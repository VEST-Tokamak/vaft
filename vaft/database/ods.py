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
import h5pyd
import h5py
import omas
import subprocess
import numpy as np
from typing import Optional, Union, List
from uncertainties import ufloat
from uncertainties.unumpy import uarray


def is_connect() -> bool:
    """
    Check if the user is connected to the server.
    
    input: None
    output: True if the user is connected to the server, False otherwise.

    """
    try:
        if h5pyd.getServerInfo()['state']=='READY':
            username = h5pyd.getServerInfo()['username']
            return True 
        else:
            return False
    except requests.exceptions.ConnectTimeout:
        return False
    

from typing import Optional, List


def exist_file(username: Optional[str] = None, shot: Optional[int] = None) -> List[int]:
    """Return a list of shot numbers from matching files in the specified directory.

    If the 'username' or 'shot' parameter is provided, the function will return shot numbers with a matching prefix.
    If 'username' or 'shot' is not provided, the function will return all shot numbers in the user's folder.
    
    Args:
        username (str, optional): The username to access the corresponding folder.
        shot (int, optional): The shot number to search for.

    Returns:
        list[int]: List of shot numbers. Empty list if no matches found or connection error.
    """
    if username is None:
        username = h5pyd.getServerInfo()['username']
    
    try:
        folder = list(h5pyd.Folder("/" + username + "/"))
        shot_numbers = []
        
        if shot is not None:
            file_list = list(folder)
            for file in file_list:
                if file.split("_")[0] == str(shot):
                    shot_num = int(file.split(".")[0])
                    shot_numbers.append(shot_num)
                    print(shot_num)

            if not shot_numbers:
                print("No matching shots found")
        else:
            for file in folder:
                try:
                    shot_num = int(file.split(".")[0])
                    shot_numbers.append(shot_num)
                except ValueError:
                    continue
            print(f"Total number of shots: {len(shot_numbers)}")
            print("Shot numbers:", sorted(shot_numbers))
            
        return sorted(shot_numbers)
    except urllib3.exceptions.MaxRetryError:
        print("Connection error")
        return []

def save(
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

    if filename is None:
        filename = f"{shot}.h5"

    if env == 'local':
        omas.save_omas_h5(ods, filename)

    if not is_connect():
        print('Error: Connection to the server failed')
        return
    
    username = 'public' if h5pyd.getServerInfo()['username'] == 'admin' else h5pyd.getServerInfo()['username']
    file_path = f"hdf5://{username}/{filename}"
    omas.save_omas_h5(ods, filename)

    command = ['hsload', '--h5image', filename, file_path]
    result = subprocess.run(command, capture_output=True, text=True)

    subprocess.run(['rm', filename], capture_output=True, text=True)

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

def load(shot: Union[int, List[int]], directory: str = 'public') -> Union[omas.ODS, List[omas.ODS]]:
    """
    Load ODS data from HDF5 file(s).
        
    Parameters:
        shot (Union[int, List[int]]): Shot number(s) to load
        directory (str, optional): Directory path. Defaults to public
    
    Returns:
        Union[omas.ODS, List[omas.ODS]]: ODS object or list of ODS objects
    """

    if isinstance(shot, list):
        shot_list = shot
        ods_list = []
        for shot in shot_list:
            ods = omas.ODS()
            filename = f'hdf5://{directory}/{shot}.h5'
            with h5py.File(h5pyd.H5Image(filename)) as data:
                convert_dataset(ods, data)
            print("Successfully loaded ODS data for shot:", shot)
            ods_list.append(ods)
        print("Successfully loaded a list of ODS data")
        return ods_list

    else:
        ods = omas.ODS()
        shot_list = [int(shot)]
        filename = f'hdf5://{directory}/{shot}.h5'
        with h5py.File(h5pyd.H5Image(filename)) as data:
            convert_dataset(ods, data)
        print("Successfully loaded ODS data for shot:", shot)
        return ods