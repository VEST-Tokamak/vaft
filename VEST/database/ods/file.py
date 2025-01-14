import h5pyd, h5py
import omas
import requests
import subprocess
from .default import is_connect
import numpy
from uncertainties import ufloat
from uncertainties.unumpy import uarray


def save(ods, shot, filename=None, env='server'):
    """
    Function to save an ODS (Open Data Structure) file either locally or to an HDF5 server.

    This function handles saving the ODS data to an HDF5 file, either on the local file system or
    remotely to an HDF5 server using the `hsload` command. If no filename is provided, the function 
    defaults to using the shot number with an `.h5` extension.

    Parameters:
    ods (object): The Open Data Structure (ODS) object that needs to be saved.
    shot (int): The shot number associated with the ODS data, used to generate the filename if not provided.
    filename (str, optional): The name of the file to save the ODS data. Defaults to `None`, which generates a name based on the shot number.
    env (str, optional): The environment where the file will be saved. It can be either 'server' for server upload or 'local' for local storage. Defaults to 'server'.

    Returns:
    None: The function doesn't return any specific value but prints information about the saving process.
    """

    # If no filename is provided, use the shot number to create a default file name
    if filename is None:
        filename = str(shot) + '.h5'

    # If the environment is 'local', save the file locally using OMAS
    if env == 'local':
        omas.save_omas_h5(ods, filename)

    # Test the connection to the server, and exit if the connection fails
    if is_connect() != True:
        print('Error: Connection to the server failed')
        return
    
    # Get the current username from the HDF5 server and construct the file path
    username = 'public' if h5pyd.getServerInfo()['username'] == 'admin' else h5pyd.getServerInfo()['username']
    file_path = "hdf5://{}/{}".format(username, filename)
    omas.save_omas_h5(ods, filename)

    command = ['hsload', '--h5image', filename, file_path]
    result = subprocess.run(command, capture_output=True, text=True)

    subprocess.run(['rm', filename], capture_output=True, text=True)

def convertDataset(ods, data):
    """
    Recursive utility function to map HDF5 structure to ODS

    :param ods: input ODS to be populated

    :param data: HDF5 dataset of group
    """
    import h5py

    keys = data.keys()
    print(keys)
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
                if isinstance(data[item][()], (float, numpy.floating)):
                    ods.setraw(item, ufloat(data[item][()], data[item + '_error_upper'][()]))
                else:
                    ods.setraw(item, uarray(data[item][()], data[item + '_error_upper'][()]))
            else:
                ods.setraw(item, data[item][()])
        elif isinstance(data[item], h5py.Group):
            convertDataset(ods.setraw(oitem, ods.same_init_ods()), data[item])

def load(ods, shot, directory=None):
    if directory is None:
        filename = f'hdf5://public/{shot}.h5'
    else:
        filename = f'hdf5://{directory}/{shot}.h5'
    with h5py.File(h5pyd.H5Image(filename)) as data:
        convertDataset(ods, data)