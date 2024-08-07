import h5pyd
import requests
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError

def is_server_ready():
    try:
        if h5pyd.getServerInfo()['state']=='READY':
            return True
        else:
            return False
    except requests.exceptions.ConnectTimeout:
        return False