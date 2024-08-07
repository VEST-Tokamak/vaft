import h5pyd

def create_file_path(shot, username, run, new=False):
    folder_path = "/public/"
    if username is not None:
        folder_path = "/" + username +"/"

    Folder = list(h5pyd.Folder(folder_path))
    if run is None:
        run = max([int(x.split("_")[1].split(".")[0]) for x in Folder if x.split("_")[0] == str(shot)])
        if new:
            run += 1

    return folder_path  + str(shot) + "_" + str(run) + ".h5"

