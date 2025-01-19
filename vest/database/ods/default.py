import h5pyd
import requests
import urllib3

def is_connect():
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
    

def exist_file(username=h5pyd.getServerInfo()['username'], shot=None):
    """
    Check if the file exists.

    If the 'username' or 'shot' parameter is provided, the function will check if a file with a matching prefix exists.
    If 'username' or 'shot' is not provided, the function will list all files in the user's folder.
    
    Parameters:
        username (str): The username to access the corresponding folder.
        shot (int, optional): The shot number to search for (default is None).
    
    Returns:
        bool: True if the file or folder exists, False if it does not or if a connection error occurs.
    """
    try:
        Folder = list(h5pyd.Folder("/" + username + "/"))
        if shot is not None:
            file_list = list(Folder)
            file_exist = False  

            for file in file_list:
                if file.split("_")[0] == str(shot):
                    file_exist = True
                    print(file)
                    return True
            
            if not file_exist:
                print("File does not exist")
                return False
        else:
            print(list(Folder))
    except urllib3.exceptions.MaxRetryError:
        return False
    return True
