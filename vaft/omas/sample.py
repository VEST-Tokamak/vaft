from omas import *
from vaft.data.resources import data_path, require_repository_sample

def sample_ods():
    # load the sample ods file in the package data folder
    sample_path = data_path("omas/39915.json")

    # load the ods file
    ods = ODS()
    ods = ods.load(str(sample_path), consistency_check=False)
    return ods

def sample_odc():
    # load the sample odc file in the package data folder
    data_1 = "omas/39915.json"
    data_2 = "omas/41524.json"
    data_3 = "omas/41672.json"

    root = data_path()

    paths = [require_repository_sample(root / data) for data in (data_1, data_2, data_3)]

    # load the ods files
    ods1 = ODS()
    ods1 = ods1.load(str(paths[0]), consistency_check=False)
    ods2 = ODS()
    ods2 = ods2.load(str(paths[1]), consistency_check=False)
    ods3 = ODS()
    ods3 = ods3.load(str(paths[2]), consistency_check=False)

    # make the odc file
    odc = ODC()
    odc['0'] = ods1
    odc['1'] = ods2
    odc['2'] = ods3

    return odc

def sample_gfile():
    """Load the historical packaged sample g-file as a VAFT GEQDSK object."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00317")
