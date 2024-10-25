import h5pyd, h5py
import omas
import requests
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError
import numpy
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import subprocess


def is_server_ready():
    try:
        if h5pyd.getServerInfo()['state']=='READY':
            return True
        else:
            return False
    except requests.exceptions.ConnectTimeout:
        return False
    
    
def exist_file(username, shot=None):
    try:
        Folder = list(h5pyd.Folder("/{username}/"))
        if shot is not None:
            file_list = list(Folder)
            print(lambda x: x.split("_")[0] == str(shot), file_list)
        else:
            print(list(Folder))
    except MaxRetryError:
        return False
    return True


def create_file_path(shot, username, new=False):
    folder_path = "/public/"
    if username is not None:
        folder_path = "/" + username +"/"

    Folder = list(h5pyd.Folder(folder_path))
    if run is None:
        run = max([int(x.split("_")[1].split(".")[0]) for x in Folder if x.split("_")[0] == str(shot)])
        if new:
            run += 1

    return folder_path  + str(shot) + "_" + str(run) + ".h5"



def convertDataset(ods, data):
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
                if isinstance(data[item][()], (float, numpy.floating)):
                    ods.setraw(item, ufloat(data[item][()], data[item + '_error_upper'][()]))
                else:
                    ods.setraw(item, uarray(data[item][()], data[item + '_error_upper'][()]))
            else:
                ods.setraw(item, data[item][()])
        elif isinstance(data[item], h5py.Group):
            convertDataset(ods.setraw(oitem, ods.same_init_ods()), data[item])



def add_static_data(ods):
    with h5py.File('./static_data_v1.h5', 'r') as data:
        convertDataset(ods, data)


def load_shot(shot, username = None, run = None):
    file_path = create_file_path(shot, username, run, new=False)

    print("Loading file: " + file_path)

    ods = omas.ODS()
    omas.load_omas_h5(file_path, hsds=True)

    if ods['dataset_description.data_entry.run'] != 1:
        print(f"Warning: The static data for vest is updated.")
        print(f"Please run the following command to update the static data: ")
        print(f"pip update vest")
        return None
    
    add_static_data(ods)

    return ods


def load_shot_data(shot, username = None, run = None, path = None):
    folder_path = "/public/"
    if username is not None:
        folder_path = "/" + username +"/"

    Folder = list(h5pyd.Folder(folder_path))
    if run is None:
        run = max([int(x.split("_")[1].split(".")[0]) for x in Folder if x.split("_")[0] == str(shot)])
    file_path = folder_path  + str(shot) + "_" + str(run) + ".h5"

    print("Loading file: " + file_path)

    ods = omas.ODS()

    omas.load_omas_h5(file_path, ods, hsds=True)

    if ods['dataset_description.data_entry.run'] != 1:
        print(f"Warning: The static data for vest is updated.")
        print(f"Please run the following command to update the static data: ")
        print(f"pip update vest")
        return None

    add_static_data(ods)
    return ods

def delete(shot, username, run):
    file_path = create_file_path(shot, username, run, new=False)

    try:
        result = subprocess.run(
            ["hsrm",  file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"File {file_path} has been deleted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to delete file {file_path}. Error: {e.stderr.decode()}")


def save_local(ods, filename):
    omas.save_omas_h5(ods, filename)

def save_server(ods, shot, username, run=None):
    if username is None:
        username = h5pyd.getServerInfo()['username']
    if run is not None:
         new = True
    file_path = create_file_path(shot, username, run, new)
    file= h5pyd.File(file_path, 'a')
    file_owner = file.owner

    if file_owner!=username:
        print(f"Error: You are not the owner of the file {file_path}.")
        file.close()
        return

    print("Saving file: " + file_path)