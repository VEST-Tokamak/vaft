import h5pyd, h5py
import omas
import requests
import subprocess


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
    
