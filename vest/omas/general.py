import vest
from omas import *
import numpy as np

def find_breakdown_onset(ods):
    time=ods.time('spectrometer_uv')
    data=ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
    (onset, offset) = vest.process.signal_onoffset(time, data)
    return onset

def find_vloop_onset(ods):
    # find the onset of loop voltage signal (same as maximum of flux loop flux signal)
            # ax.plot(ods['magnetics.time'], ods[f'magnetics.b_field_pol_probe.{i}.field.data'])
    time=ods.time('magnetics')
    flux=ods['magnetics.flux_loop.0.flux.data']
    # find the maximum time of the flux loop signal
    onset = time[np.argmax(flux)]
    return onset

def find_ip_onset(ods):
    # find the onset of the plasma current signal
    time=ods.time('magnetics')
    current=ods['magnetics.ip.0.data']
    # find the onset of the plasma current signal
    (onset, offset) = vest.process.signal_onoffset(time, current)
    return onset

def find_pf_active_onset(ods):
    # find the first onset of the pf active current signal and onsets for each channel
    time=ods.time('pf_active')
    onset_all = []
    onset_list = []
    for i in range(len(ods['pf_active.channel'])):
        current = ods[f'pf_active.channel.{i}.current.data']
        (onset, offset) = vest.process.signal_onoffset(time, current)
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
