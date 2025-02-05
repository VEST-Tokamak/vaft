# in this file, we test the all available plot submodule functions
# we will use the sample ods file in the data folder
from omas import *
import vest
import pkg_resources

data_path = pkg_resources.resource_filename('vest', 'data/39915.json')
# load sample ods from file
ods = ODS()
ods = ods.load(data_path, consistency_check=False)

# plot pf_active
vest.plot.time_pf_active_all_current(ods)

