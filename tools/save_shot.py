import omas
import h5pyd
from tools.path_tools import create_file_path

# Not implemented yet

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
    