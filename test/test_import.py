import vaft
from vaft.process import *
from vaft.plot import *
from vaft.machine_mapping import *
from vaft.formula import *
from vaft.omas import *
from vaft.code import *
from vaft.database import *
from vaft.database.file import *
from vaft.database.raw import *
# from vaft.database.ods import * 
# Traceback (most recent call last):
#   File "/Users/yun/Library/CloudStorage/OneDrive-Personal/Documents/Research/Code/vaft/test/test_import.py", line 11, in <module>
#     from vaft.database.ods import *
#   File "/Users/yun/Library/CloudStorage/OneDrive-Personal/Documents/Research/Code/vaft/vaft/database/ods/__init__.py", line 1, in <module>
#     from .default import *
#   File "/Users/yun/Library/CloudStorage/OneDrive-Personal/Documents/Research/Code/vaft/vaft/database/ods/default.py", line 23, in <module>
#     def exist_file(username=h5pyd.getServerInfo()['username'], shot=None):
#   File "/Users/yun/miniforge3/envs/fusion/lib/python3.8/site-packages/h5pyd/_hl/serverinfo.py", line 37, in getServerInfo
#     http_conn = HttpConn(
#   File "/Users/yun/miniforge3/envs/fusion/lib/python3.8/site-packages/h5pyd/_hl/httpconn.py", line 205, in __init__

print(f"vaft version: {vaft.__version__}")