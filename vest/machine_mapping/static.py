# import math
# import numpy as np
# import scipy.io
# from typing import Dict, Any

# # Assuming that the import from '.' needs to be specified
# from .database import raw_database_info  # Replace with the actual module


# def dataset_description(ods: Dict[str, Any], shot: int, run: int) -> None:
#     """
#     Populate dataset_description in ODS.

#     :param ods: ODS with 'dataset_description'.
#     :param shot: Shot number.
#     :param run: Run number.
#     """
#     dd = ods['dataset_description']
#     dd['ids_properties.comment'] = 'VEST OMAS-ODS Data based on vestpy code'
#     dd['ids_properties.homogeneous_time'] = 2
#     dd['data_entry.machine'] = 'VEST'
#     dd['data_entry.pulse'] = shot
#     dd['data_entry.pulse_type'] = 'pulse'
#     dd['data_entry.run'] = run
#     dd['data_entry.user'] = os.environ.get('USER', 'unknown')




# def pf_active_static(ods, shot, file: str = 'static_data_info.yaml') -> None:
#     """
#     Retrieve static PF active data from the database.

#     :param ods: Operational data structure.
#     :param shot: Shot number.
#     :param file: YAML file containing static data information.
#     """
#     info = raw_database_info(file=file, shot=shot, key='pf_active')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']
#     # Add further processing as needed
#     print(f"Static PF Active Data: Label={label}, Field={field}, Gain={gain}")


# def pf_active_dynamic(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Retrieve dynamic PF active data from the database.

#     :param ods: Operational data structure.
#     :param shot: Shot number.
#     """
#     # Implementation needed
#     print("pf_active_dynamic is not yet implemented.")


# def vfit_pf_active(
#     ods: Dict[str, Any],
#     shot: int,
#     tstart: float,
#     tend: float,
#     dt: float
# ) -> None:
#     """
#     Generate PF coil geometry from a .mat file and attach coil currents from DB.

#     :param ods: ODS containing `pf_active`.
#     :param shot: VEST shot number.
#     :param tstart: Start time for sampling PF currents.
#     :param tend: End time for sampling PF currents.
#     :param dt: Time step for PF sampling.
#     """
#     pf = ods.get('pf_active', {})
#     pf['ids_properties.comment'] = 'PF config from vfit_pf_active'
#     pf['ids_properties.homogeneous_time'] = 1

#     nbcoil = 10
#     nb_elt = [158, 100, 16, 16, 24, 24, 48, 48, 48, 48]

#     # Example reading from DB:
#     # (time_db, data_db) = vest_loadn(shot, 'PF1 Current')  # or vest_load(...)
#     time_db = np.linspace(0, 1, 500)
#     data_db = np.sin(2 * np.pi * time_db) * 1000.0  # Placeholder data

#     if dt > 0:
#         tstart = max(tstart, time_db[0])
#         tend = min(tend, time_db[-1])
#         time_1 = np.arange(tstart, tend, dt)
#     else:
#         time_1 = time_db

#     pf['time'] = time_1.tolist()

#     # Define material properties
#     r_coil = 1.68e-8  # Copper resistivity in Ohm-meter

#     # Define coil geometry parameters
#     # rpf: Represents the effective length factor of each coil (related to radius or perimeter)
#     # apf: Represents the cross-sectional area of each coil
#     rpf = [0.053, 0.104, 0.29, 0.57, 0.71, 0.71, 0.71, 0.71, 0.93, 0.93]  # Effective length factors for each coil
#     apf = [
#         0.04128, 0.0304, 0.000812, 0.000812, 0.001218, 0.001218, 0.0027216,
#         0.0027216, 0.0027216, 0.0027216
#     ]  # Cross-sectional areas for each coil

#     # Loop through all coils and assign properties
#     for i in range(nbcoil):
#         # Set coil name and identifier
#         pf[f'coil.{i}.name'] = f'PF{i + 1}'
#         pf[f'coil.{i}.identifier'] = f'PF{i + 1}'

#         # Calculate resistance using the formula R = ρ * (L / A)
#         # L = 2 * pi * rpf (effective length factor)
#         # A = apf (cross-sectional area)
#         pf[f'coil.{i}.resistance'] = 2.0 * math.pi * r_coil * rpf[i] / apf[i]


#     # Load geometry from .mat
#     try:
#         data2 = scipy.io.loadmat(
#             '../Geometry/VEST_DiscretizedCoilGeometry_Full_ver_1906.mat'
#         )
#         coil_array = data2.get('coil')
#         if coil_array is None:
#             raise ValueError("Key 'coil' not found in the .mat file.")
#     except FileNotFoundError:
#         print("Geometry file not found.")
#         return
#     except Exception as e:
#         print(f"Error loading geometry: {e}")
#         return

#     idx = 0
#     for k in range(nbcoil):
#         n_belt = nb_elt[k]
#         for i in range(n_belt):
#             try:
#                 coil_data = coil_array[idx]
#                 pf[f'coil.{k}.element.{i}.turns_with_sign'] = coil_data[5]
#                 pf[f'coil.{k}.element.{i}.geometry.geometry_type'] = 2  # Assuming type 2
#                 pf[f'coil.{k}.element.{i}.geometry.rectangle.r'] = float(coil_data[0])
#                 pf[f'coil.{k}.element.{i}.geometry.rectangle.z'] = float(coil_data[1])
#                 dr_val = float(coil_data[2])
#                 dz_val = float(coil_data[3])
#                 pf[f'coil.{k}.element.{i}.geometry.rectangle.width'] = dr_val
#                 pf[f'coil.{k}.element.{i}.geometry.rectangle.height'] = dz_val
#                 pf[f'coil.{k}.element.{i}.area'] = dr_val * dz_val
#                 idx += 1
#             except IndexError:
#                 print(f"Index {idx} out of range for coil_array.")
#                 break
#             except Exception as e:
#                 print(f"Error processing coil element {k}.{i}: {e}")
#                 continue

#     # Load coil current from vest DB
#     # (time_pf, pf_data) = vfit_pf(shot)  # Example function to get all PF currents
#     # We'll just create placeholders:
#     pf_data = np.cos(2 * np.pi * time_db) * 1200.0
#     # Interpolate
#     pf['coil.0.current.data'] = np.interp(time_1, time_db, pf_data).tolist()

#     ods['pf_active'] = pf
#     print("vfit_pf_active completed successfully.")
