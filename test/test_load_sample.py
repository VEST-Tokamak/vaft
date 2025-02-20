import vest
import pkg_resources
from omas import *

# load sample ods from file
data_path = pkg_resources.resource_filename('vest', 'data/39915.json')

# load the ods file
ods = ODS()
ods = ods.load(data_path, consistency_check=False)

# print the ods keys
print(ods.keys())

# print the ods['equilibrium'] keys
print(ods['equilibrium'].keys())

