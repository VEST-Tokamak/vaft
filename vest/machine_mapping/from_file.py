# # load data from file and store in ods

# # naming convention: machine_mapping.<diagnostic_name>_from_file(ods, shotnumber)

# # =============================================================================
# # Thomson Scattering
# # =============================================================================

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
