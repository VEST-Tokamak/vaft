import vest
from vest.process import *
from vest.plot import *
from vest.machine_mapping import *
from vest.formula import *
from vest.omas import *
from vest.code import *
from vest.database import *
from vest.database.file import *
from vest.database.raw import *
# from vest.database.ods import * 
# Traceback (most recent call last):
#   File "/Users/yun/Library/CloudStorage/OneDrive-Personal/Documents/Research/Code/vest/test/test_import.py", line 11, in <module>
#     from vest.database.ods import *
#   File "/Users/yun/Library/CloudStorage/OneDrive-Personal/Documents/Research/Code/vest/vest/database/ods/__init__.py", line 1, in <module>
#     from .default import *
#   File "/Users/yun/Library/CloudStorage/OneDrive-Personal/Documents/Research/Code/vest/vest/database/ods/default.py", line 23, in <module>
#     def exist_file(username=h5pyd.getServerInfo()['username'], shot=None):
#   File "/Users/yun/miniforge3/envs/fusion/lib/python3.8/site-packages/h5pyd/_hl/serverinfo.py", line 37, in getServerInfo
#     http_conn = HttpConn(
#   File "/Users/yun/miniforge3/envs/fusion/lib/python3.8/site-packages/h5pyd/_hl/httpconn.py", line 205, in __init__

print(f"vest version: {vest.__version__}")