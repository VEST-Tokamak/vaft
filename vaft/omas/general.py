import vaft
from omas import *
import numpy as np
import matplotlib.pyplot as plt
import re

# ----------------------------------------------------------------------
# Find information from ODS
# ----------------------------------------------------------------------
def find_shotnumber(ods):
    """Find the shot number from the ODS."""
    return ods['dataset_description.data_entry.pulse']

def find_shotclass(ods,plot_opt=0):
    """
    !!! Obsolete function:
    it is not successfully find the shot class. Need to be improved.

    Find shot class from ODS
    # 'Plasma': Plasma discharge is stable and Halpha is detected
    # 'Vacuum': Vacuum Test Shot
    # 'Breakdown failure': Try to discharge but failed (BD Test Shot)

    """
    # check existence of barometry and spectrometer
    if 'barometry' not in ods:
        print('Barometry not found in ODS')
        return
    if 'spectrometer_uv' not in ods:
        print('Spectrometer not found in ODS')
        return

    # Check status: Vacuum, BD failure, Plasma
    time_pres=ods['barometry.gauge.0.pressure.time'] # time
    data_pres=ods['barometry.gauge.0.pressure.data'] # pressure
    data_alpha=ods['spectrometer_uv.channel.0.processed_line.0.intensity.data'] # Halpha
    time_alpha=ods['spectrometer_uv.time'] # Halpha

    if vaft.process.is_signal_active(data_alpha, 0.01):
        status = 'Plasma'
    else:
        if not vaft.process.is_signal_active(data_pres): # no pressure?
            status='Vacuum'
        else:
            status='BD failure'
    return status

def find_chamber_boundary(ods):
    """Find the chamber boundary from the ODS."""
    return ods['wall.description_2d.0.limiter.unit.0.outline.r'], ods['wall.description_2d.0.limiter.unit.0.outline.z']

def find_breakdown_onset(ods):
    time=ods.time('spectrometer_uv')
    data=ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
    (onset, offset) = vaft.process.signal_onoffset(time, data)
    return onset

def find_breakdown_onset(ods):
    """Find the onset of the breakdown using spectrometer_uv signal."""
    return _find_signal_onset(ods, 'spectrometer_uv', 'spectrometer_uv.channel.0.processed_line.0.intensity.data')

def find_vloop_onset(ods):
    """Find the onset of the loop voltage signal (maximum of flux loop signal)."""
    time = ods.time('magnetics')
    flux = ods['magnetics.flux_loop.0.flux.data']
    return time[np.argmax(flux)]

def find_ip_onset(ods):
    """Find the onset of the plasma current signal."""
    return _find_signal_onset(ods, 'magnetics', 'magnetics.ip.0.data')

def find_pf_active_onset(ods):
    """Find the onset for each pf_active channel current signal."""
    time = ods.time('pf_active')
    onsets = []
    for i in range(len(ods['pf_active.channel'])):
        current = ods[f'pf_active.channel.{i}.current.data']
        (onset, offset) = vaft.process.signal_onoffset(time, current)
        onset_all.append(onset)
        onset_list.append(onset)
    return onset_all, onset_list

def shift_time(ods, time_shift):
    for path in ods.paths():
        if 'time' in path:
            path_string = '.'.join(map(str, path))
            ods[path_string] += time_shift # shift the time by time_shift (if time_shift is negative, the time is shifted to the left)
        if 'onset' in path:
            path_string = '.'.join(map(str, path))
            ods[path_string] += time_shift
        if 'offset' in path:
            path_string = '.'.join(map(str, path))
            ods[path_string] += time_shift

def change_time_convention(ods, convention = 'vloop'):
    # covention list: 'daq', 'vloop', 'ip', 'breakdown'
    if 'summary.code.parameters' not in ods:
        ods['summary.code.parameters'] = CodeParameters()
        ods['summary.code.parameters.time_convention'] = 'daq'
        vloop_onset = find_vloop_onset(ods)
        ods['summary.code.parameters.vloop_onset'] = vloop_onset
        ip_onset = find_ip_onset(ods)
        ods['summary.code.parameters.ip_onset'] = ip_onset
        breakdown_onset = find_breakdown_onset(ods)
        ods['summary.code.parameters.breakdown_onset'] = breakdown_onset

    orgianl_convention = ods['summary.code.parameters.time_convention']

    # calculate the time shift
    if orgianl_convention == 'daq':
        if convention == 'vloop':
            time_shift = - vloop_onset
        elif convention == 'ip':
            time_shift = - ip_onset
        elif convention == 'breakdown':
            time_shift = - breakdown_onset
    elif orgianl_convention == 'vloop':
        if convention == 'daq':
            time_shift = vloop_onset
        elif convention == 'ip':
            time_shift = vloop_onset - ip_onset
        elif convention == 'breakdown':
            time_shift = vloop_onset - breakdown_onset
    elif orgianl_convention == 'ip':
        if convention == 'daq':
            time_shift = ip_onset
        elif convention == 'vloop':
            time_shift = ip_onset - vloop_onset
        elif convention == 'breakdown':
            time_shift = ip_onset - breakdown_onset
    elif orgianl_convention == 'breakdown':
        if convention == 'daq':
            time_shift = breakdown_onset
        elif convention == 'vloop':
            time_shift = breakdown_onset - vloop_onset
        elif convention == 'ip':
            time_shift = breakdown_onset - ip_onset
    # Print the time shift
    print(f'Time shift from {orgianl_convention} to {convention} is {time_shift}')

    # shift the time
    shift_time(ods, time_shift)

def print_info(ods, key_name=None):
  
  key_list=[]
  for key in ods.keys():
    key_list.append(key)
  
  if (key_name == None):
  
    print("{:<20} : {}".format(" Machine_name", ods['dataset_description.data_entry.machine']))
    print("{:<20} : {}".format(" Shot_number", ods['dataset_description.data_entry.pulse']))
    print("{:<20} : {}".format(" Operation_type", ods['dataset_description.data_entry.pulse_type']))
    print("{:<20} : {}".format(" Run", ods['dataset_description.data_entry.run']))
    print("{:<20} : {}".format(" User_name", ods['dataset_description.data_entry.user']))

    print(" {:<20} : {}".format("KEY", "VALUES"), '\n')
    for key in ods.keys():
        print(" {:<20}".format(key), ':', ','.join(ods[key].keys()))

  elif key_name in key_list:
    print("\n Number of",key_name," Data set \n")
    for key in ods[key_name]:
        if key=="time" or key=="ids_properties":
            continue
        print("  {:<17} : {}".format(key , len(ods[key_name][key])))
        
  else:
    print("key_name value Error!")
    



# def check_thompson(ods):
#     if 'thomson_scattering' not in ods.keys():
#         return False
    
#     if 'time' not in ods['thomson_scattering'].keys():
#         return False
    
#     if 'ne.data' not in ods['thomson_scattering'].keys():
#         return False
    
#     if 'te.data' not in ods['thomson_scattering'].keys():
#         return False
    
#     return True

# def check_equilibrium(ods):
#     if 'equilibrium' not in ods.keys():
#         return False
    
#     if 'time' not in ods['equilibrium'].keys():
#         return False
    
#     return True
