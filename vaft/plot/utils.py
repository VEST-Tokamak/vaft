from omas import *

def get_from_path(obj, path):
    """'a.b.c' 형태의 path로 dict/ODS에서 값 추출"""
    for key in path.split('.'):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
        if obj is None:
            return None
    return obj

def odc_or_ods_check(odc_or_ods):
    """
    Check input type and initialize ODC if necessary.
    
    Parameters:
    odc_or_ods (ODC or ODS): Input object to check.
    
    Returns:
    ODC: Initialized ODC object.
    
    Raises:
    TypeError: If input is not of type ODS or ODC.
    """
    if isinstance(odc_or_ods, ODC):
        return odc_or_ods
    elif isinstance(odc_or_ods, ODS):
        odc = ODC()
        odc['0'] = odc_or_ods
        return odc
    else:
        raise TypeError("Input must be of type ODS or ODC")


def extract_labels_from_odc(odc, opt = 'shot'):
    """
    Extract list from ODC object. 
    
    Parameters:
    odc (ODC): ODC object to extract labels from.
    opt (str): The option for the list. Can be 'shot'/'pulse' or 'key'
    Returns:
    list: List of labels extracted from ODC.
    """
    labels = []
    for key in odc.keys():
        if opt == 'key':
            labels.append(key)
        elif opt == 'shot' or opt == 'pulse':
            try:
                data_entry = odc[key].get('dataset_description.data_entry', {})
                labels.append(data_entry.get('pulse'))
            except:
                print(f"Key {key} does not have a dataset_description.data_entry.")
                labels.append(key)
        elif opt == 'run':
            try:
                data_entry = odc[key].get('dataset_description.data_entry', {})
                labels.append(data_entry.get('run'))
            except:
                print(f"Key {key} does not have a dataset_description.data_entry.")
                labels.append(key)
        else:
            print(f"Invalid option: {opt}, using key as label.")
            labels.append(key)
    return labels

