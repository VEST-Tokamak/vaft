import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from uncertainties import unumpy
from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

# # load data from file and store in ods

# # naming convention: machine_mapping.<diagnostic_name>_from_file(ods, shotnumber)

# # =============================================================================
# # Thomson Scattering
# # =============================================================================

def vfit_thomson_scattering_static(ods):
    """
    Set up static properties for Thomson scattering in the ODS object for VEST tokamak.

    This function populates the 'thomson_scattering' section of the given ODS object
    with static (time-independent) data such as positions and names of the channels.

    Parameters:
        ods (ODS): The OMAS data structure to populate.
    """
    ods['thomson_scattering.ids_properties.homogeneous_time'] = 1

    r_positions = [0.475, 0.425, 0.37, 0.31, 0.255]
    z_positions = [0, 0, 0, 0, 0]
    names = [
        'Polychrometer 1R1',
        'Polychrometer 2R2',
        'Polychrometer 3R3',
        'Polychrometer 4R4',
        'Polychrometer 5R5',
    ]

    for i in range(5):
        ods[f'thomson_scattering.channel.{i}.position.r'] = r_positions[i]
        ods[f'thomson_scattering.channel.{i}.position.z'] = z_positions[i]
        ods[f'thomson_scattering.channel.{i}.name'] = names[i]

def vfit_thomson_scattering_dynamic(ods, shotnumber, base_path=None):
    """
    Load dynamic Thomson scattering data from a .mat file into the ODS object for VEST tokamak.

    This function reads electron temperature and density data from a MATLAB .mat file
    for a given shot number and populates the 'thomson_scattering' section of the ODS.

    Parameters:
        ods (ODS): The OMAS data structure to populate.
        shotnumber (int): The shot number to load data for.
        base_path (str, optional): The base directory containing the data files.
            Defaults to the current working directory.
    """
    if base_path is None:
        base_path = os.getcwd()

    # find the file such that NeTe_Shot{shotnumber}*.mat
    pattern = f'NeTe_Shot{shotnumber}*.mat'
    for file in os.listdir(base_path):
        if pattern in file:
            filename = file
            print(f'Found file: {filename}')
            break
    filename_v10 = f'NeTe_Shot{shotnumber}_v10.mat'
    filename_v9_rev = f'NeTe_Shot{shotnumber}_v9_rev.mat'

    if os.path.exists(filename_v10):
        filename = filename_v10
        version = 'v10'
    elif os.path.exists(filename_v9_rev):
        filename = filename_v9_rev
        version = 'v9_rev'
    else :
        raise FileNotFoundError(f"No file matching {pattern} found in {base_path}")
 
    mat_data = loadmat(os.path.join(base_path, filename))

    if 'dataset_description.data_entry.pulse' not in ods:
        ods['dataset_description.data_entry.pulse'] = shotnumber

    ods['thomson_scattering.time'] = mat_data['time_TS'][0] / 1e3  # Convert from ms to s

    for i in range(1, 6):  # Channels are numbered from 1 to 5
        channel_index = i - 1  # Indices in ods start from 0
        te_key = f'poly{i}R{i}_Te'
        te_sigma_key = f'poly{i}R{i}_sigmaTe'
        ne_key = f'poly{i}R{i}_Ne'
        ne_sigma_key = f'poly{i}R{i}_sigmaNe'

        # ods[f'thomson_scattering.channel.{channel_index}.t_e.data'] = unumpy.uarray(
        #     mat_data[te_key][0], mat_data[te_sigma_key][0]
        # )
        # ods[f'thomson_scattering.channel.{channel_index}.n_e.data'] = unumpy.uarray(
        #     mat_data[ne_key][0], mat_data[ne_sigma_key][0]
        # )
        ods[f'thomson_scattering.channel.{channel_index}.t_e.data'] = unumpy.uarray(
            mat_data[te_key][0],abs(mat_data[te_sigma_key][0])
        )
        ods[f'thomson_scattering.channel.{channel_index}.n_e.data'] = unumpy.uarray(
            mat_data[ne_key][0],abs(mat_data[ne_sigma_key][0])
        )
        # ad-hoc Ne, Te sigma as absolute value


# def thomson_scattering_from_file(ods: Dict[str, Any], shotnumber: int) -> None:
#     """
#     Load Thomson scattering data into ODS['thomson_scattering'].

#     :param ods: ODS structure.
#     :param shotnumber: Shot number.
#     """
#     ts = ods['thomson_scattering']
#     ts['ids_properties.homogeneous_time'] = 1

#     # Example geometry
#     for i, rpos in enumerate([0.475, 0.425, 0.37, 0.31, 0.255]):
#         ts[f'channel.{i}.position.r'] = rpos
#         ts[f'channel.{i}.position.z'] = 0.0
#         ts[f'channel.{i}.name'] = f'Polychrometer {i + 1}'

#     # Example data
#     time_db = np.linspace(0.2, 0.4, 50)
#     ne_db = 1e19 * np.ones((5, 50))  # 5 channels
#     te_db = 50.0 * np.ones((5, 50))

#     ts['time'] = time_db
#     for i in range(5):
#         ts[f'channel.{i}.t_e.data'] = te_db[i]
#         ts[f'channel.{i}.n_e.data'] = ne_db[i]


# # =============================================================================
# # Ion Doppler Spectroscopy
# # =============================================================================

# def ion_doppler_spectroscopy_from_file(
#     ods: Dict[str, Any],
#     shotnumber: int,
#     options: str = 'single'
# ) -> None:
#     """
#     Load ion Doppler spectroscopy data into ODS['charge_exchange'].

#     :param ods: ODS structure.
#     :param shotnumber: Shot number.
#     :param options: 'single' or 'profile'.
#     """
#     ods['charge_exchange.ids_properties.homogeneous_time'] = 1
#     if options == 'single':
#         print("read_doppler_single(ods, shotnumber) stub")
#     elif options == 'profile':
#         print("read_doppler_profile(ods, shotnumber) stub")


# # =============================================================================
# # Fast Camera
# # =============================================================================

# def vfit_fastcamera_from_file(ods: Dict[str, Any], shotnumber: int) -> None:
#     """
#     Load fast camera frames from local .bmp for ODS['camera_visible'].

#     :param ods: ODS structure.
#     :param shotnumber: Shot number.
#     """
#     vfit_camera_visible(ods, shotnumber)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Refactored EFIT Workflow for VEST Data

# This script automates VEST diagnostic data retrieval (poloidal/toroidal fields,
# flux loops, etc.), computes eddy currents, generates EFIT constraints (k-files),
# and merges EFIT results back into ODS for further analysis and plotting.
# """



# # =============================================================================
# # Equilibrium from External Analysis (Element/Profiles)
# # =============================================================================

# def vfit_element_analysis_from_file(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Populate ODS equilibrium from external element analysis mat file.

#     :param ods: ODS containing 'equilibrium'.
#     :param shot: Shot number.
#     """
#     print("Reading element analysis data from .mat (stub).")
#     # mat = read_mat_from_ElementAnalysis(shot)
#     # parse mat and store in ods accordingly
#     print("vfit_equilibrium_from_element_analysis done (stub).")


# def vfit_from_profile_fitting_from_file(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Populate ODS equilibrium from external profile fitting mat file.

#     :param ods: ODS containing 'equilibrium'.
#     :param shot: Shot number.
#     """
#     print("Reading profile fitting data from .mat (stub).")
#     # mat = read_mat_from_ProfileFitting(shot)
#     # parse mat and store in ods
#     print("vfit_equilibrium_from_profile_fitting done (stub).")
