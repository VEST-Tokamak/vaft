import vaft
import pkg_resources
from omas import *

# load sample ods from file
data_path = pkg_resources.resource_filename('vaft', 'data/41514.h5')
ods = load_omas_h5(data_path, consistency_check=False)

# change time convention

