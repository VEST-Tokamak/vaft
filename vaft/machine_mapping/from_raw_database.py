import numpy as np
from numpy import ndarray
from typing import List, Dict, Any, Tuple
from omas import *
import vaft.machine_mapping
import vaft.machine_mapping.from_file
from vaft.process import compute_br_bz_phi
import math
import yaml

# naming convention: machine_mapping.<diagnostic_name>_from_raw_database(ods, shotnumber)


# def raw_database_info(file, shot, key):
#     """
#     Retrieve data for all channels of a system from a YAML file in dictionary format.

#     Args:
#         file (str): Path to the YAML file.
#         shot (int): Shot number to retrieve the data for.
#         key (str): Key to retrieve the data for (e.g., 'tf').

#     Returns:
#         dict: A dictionary containing labels, fields, and gains indexed by channel.
#               Example: {'labels': {'0': 'TF Coil 1', ...},
#                         'fields': {'0': 1, ...},
#                         'gains': {'0': -30000, ...}}
#     """
#     # Load the YAML file
#     with open(file, 'r') as f:
#         data = yaml.safe_load(f)

#     # Initialize result dictionary
#     result = {'labels': {}, 'fields': {}, 'gains': {}}

#     # Sort shot numbers and iterate to find relevant data
#     for current_shot in sorted(data.keys(), key=int):
#         if int(current_shot) > shot:
#             break  # Stop if the current shot exceeds the target shot

#         if key in data[current_shot]:  # Check if the key exists for the current shot
#             for index, item in data[current_shot][key].items():
#                 index = str(index)  # Ensure index is a string for dictionary keys
#                 if "label" in item:
#                     result['labels'][index] = item["label"]
#                 if "field" in item:
#                     result['fields'][index] = item["field"]
#                 if "gain" in item:
#                     result['gains'][index] = item["gain"]

#     # Verify completeness of the result
#     for index in result['labels'].keys():
#         if (index not in result['fields'] or index not in result['gains']):
#             raise ValueError(f"Incomplete data for shot {shot} and key '{key}' in channel {index}.")

#     return result


# """
# Coils
# """
# def tf_from_raw_database(ods,shot):
#     """
#     Process TF data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.tf_static(ods)

#     info = raw_database_info(file = 'raw_database.yaml', shot, 'tf'):
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']
#     r0 = info['r0']['0']
#     turns = info['turns']['0']

#     (time, data) = vaft.database.raw.load(shot,field)
#     (time,current,BtR)=vaft.process.tf(time,data,gain,r0,turns)

#     TF=ods['tf']
#     TF['ids_properties.comment'] = 'TF data from VEST raw database'
#     TF['ids_properties.homogeneous_time'] = 1
#     TF['r0']=r0
#     TF['b_field_tor_vacuum_r.data']=BtR
#     TF['time']=time
#     TF['coil.0.current.data']=current
#     TF['r0']=r0



# def barometry_from_raw_database(
#     ods: Dict[str, Any],
#     shot: int,
# ) -> None:
#     """
#     Load pressure gauge data into ODS['barometry'].

#     :param ods: ODS structure.
#     :param shot: Shot number.
#     :param tstart: Start time.
#     :param tend: End time.
#     :param dt: Time step.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.barometry_static(ods)

#     info = raw_database_info(file = 'raw_database_info.yaml', shot, 'barometry')
#     label = info['labels']['0']
#     field = info['fields']['0']

#     time, torr = vaft.database.raw.load(shot, field)

#     # Convert to Pa
#     Pa = torr * 133.3223684211 # 1 torr = 133.3223684211 Pa

#     # Store data
#     ods['barometry.gauge.0.pressure.time'] = time
#     ods['barometry.gauge.0.pressure.data'] = Pa

# """
# Mangetics
# """

# def flux_loop_from_raw_database(ods, shot):
#     """
#     Process flux loop data from the VEST raw database.
#     """
#     # Update static ODS structure

#     info = raw_database_info(file = 'raw_database.yaml', shot, 'flux_loop')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']

#     (time, data) = vaft.database.raw.load(shot,field)
#     (time, BtR, BtZ, BtPhi) = vaft.process.flux_loop(time,data,gain)

#     FL=ods['flux_loop']
#     FL['ids_properties.comment'] = 'Flux loop data from VEST raw database'
#     FL['ids_properties.homogeneous_time'] = 1
#     FL['time']=time
#     FL['b_field_tor_vacuum_r.data']=BtR
#     FL['b_field_tor_vacuum_z.data']=BtZ
#     FL['b_field_tor_vacuum_phi.data']=BtPhi

# def b_field_pol_probe_from_raw_database(ods, shot):
#     """
#     Process B-field poloidal probe data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.b_field_pol_probe_static(ods)

#     info = raw_database_info(file = 'raw_database_info.yaml', shot, 'b_field_pol_probe')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']

#     # setting
#     lowpass_param = 0.01
#     baseline_onset, _ = vaft.omas.general.find_pf_active_onset(ods)
#     baseline_offset = 0.28
#     baseline_type = 'linear'
#     baseline_onset_window = 500
#     baseline_offset_window = 100
#     plot_opt = False

#     (time, data) = vaft.database.raw.load(shot,field)
#     (time, BtR, BtZ, BtPhi) = vaft.process.b_field_pol_probe_field(
#         time, data, gain, lowpass_param, baseline_onset, baseline_offset, baseline_type, baseline_onset_window, baseline_offset_window, plot_opt)
#     BP=ods['b_field_pol_probe']
#     BP['ids_properties.comment'] = 'B-field poloidal probe data from VEST raw database'
#     BP['ids_properties.homogeneous_time'] = 1
#     BP['time']=time
#     BP['b_field_tor_vacuum_r.data']=BtR
#     BP['b_field_tor_vacuum_z.data']=BtZ
#     BP['b_field_tor_vacuum_phi.data']=BtPhi

# def rogowski_coil_and_ip_from_raw_database(ods, shot):
#     """
#     Process Rogowski coil and Ip data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.rogowski_coil_and_ip_static(ods)

#     info = raw_database_info(file = 'raw_database_info.yaml', shot, 'rogowski_coil')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']
#     fl_field = info['fl_field']['0']
#     effective_res = info['effective_res']['0']

#     (rogowski_time, rogowski_raw) = vaft.database.raw.load(shot,field)
#     (time_fl, fl_law) = vaft.database.raw.load(shot,fl_field)

#     (time, ip) = vaft.process.ip(time,rogowski_raw,gain,fl_law,effective_res)

#     RC=ods['rogowski_coil']
#     RC['ids_properties.comment'] = 'Rogowski coil and Ip data from VEST raw database'
#     RC['ids_properties.homogeneous_time'] = 1
#     RC['time']=time
#     RC['current.data']=Ip

# def diamagnetic_flux_from_raw_database(ods, shot):

# # def internal_magnetic_probe_array_dynamic(ods, shot):


# def magnetics_from_raw_database(ods, shot):
#     """
#     Process magnetics data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.magnetics_static(ods)

#     # Load raw data from database, post-process, and map routinely available magnetic diagnostics to ODS

    