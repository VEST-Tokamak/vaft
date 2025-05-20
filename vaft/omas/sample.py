from omas import *
import pkg_resources
from omfit_classes.omfit_eqdsk import OMFITeqdsk

def sample_ods():
    # load the sample ods file in the package data folder
    data_path = pkg_resources.resource_filename('vaft', 'data/39915.json')

    # load the ods file
    ods = ODS()
    ods = ods.load(data_path, consistency_check=False)
    return ods

def sample_odc():
    # load the sample odc file in the package data folder
    data_1 = '39915.json'
    data_2 = '41524.json'
    data_3 = '41672.json'

    data_path = pkg_resources.resource_filename('vaft', 'data/')

    # load the ods files
    ods1 = ODS()
    ods1 = ods1.load(data_path + data_1, consistency_check=False)
    ods2 = ODS()
    ods2 = ods2.load(data_path + data_2, consistency_check=False)
    ods3 = ODS()
    ods3 = ods3.load(data_path + data_3, consistency_check=False)

    # make the odc file
    odc = ODC()
    odc['0'] = ods1
    odc['1'] = ods2
    odc['2'] = ods3

    return odc

def sample_gfile():
    # load the sample gfile file in the package data folder
    data_path = pkg_resources.resource_filename('vaft', 'data/g039915.00317')

    # load the gfile file
    gfile = OMFITeqdsk(data_path)
    return gfile