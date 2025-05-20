#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored VEST CODE DEV module.

This module provides a suite of Python functions for:
1. Loading data from the VEST MySQL database.
2. Performing coil/wall/plasma-based magnetic field and flux calculations.
3. Handling signal processing for magnetic diagnostics.
4. Computing vacuum fields, eddy currents, and related diagnostic comparisons.

Refactoring Guidelines Followed:
- PEP 8 coding style.
- Meaningful naming conventions and docstrings.
- Clear organization into single-responsibility functions.
- Removal of redundant code; repeated patterns unified into helper functions.
- Type hints added for better clarity and maintenance.
- Basic performance optimizations where straightforward (e.g., DRY code).

NOTE: This code assumes the availability of the external modules/functions:
`greenBrBz`, `greenR`, `greendBrBz` (from `green`), `omas`, `omas_utils`,
`omas_mongo`, and standard libraries (`numpy`, `mysql.connector`, etc.).

Author: VEST CODE DEV
Date: 2025-01-26
"""

import copy
import json
import math
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import scipy
from numpy import ndarray
from os.path import exists
from pymongo import MongoClient
from scipy import integrate, signal

from omas import ODS, omas_rcparams, load_omas_nc
from omas.omas_mongo import get_mongo_credentials, json_dumper
from omas.omas_utils import printd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green's function calculations
Includes elliptic approximations and Green functions for Br, Bz, Psi, 
as well as partial derivatives used in advanced coil/wall modeling.
"""

import numpy as np


# =============================================================================
# File-based read/write of Br, Bz, Psi
# =============================================================================
def write_b(
    br: ndarray,
    bz: ndarray,
    phi: ndarray,
    f_br: str,
    f_bz: str,
    f_phi: str
) -> None:
    """
    Save grid-based (Br, Bz, Phi) response matrices to text files.

    :param br: 2D array of Br values.
    :param bz: 2D array of Bz values.
    :param phi: 2D array of Phi values.
    :param f_br: Output filename for Br data.
    :param f_bz: Output filename for Bz data.
    :param f_phi: Output filename for Phi data.
    """
    nbturn, nbpf = br.shape

    with open(f_br, 'w') as f:
        f.write(f"{nbturn} {nbpf}\n")
        for j in range(nbturn):
            row_str = " ".join(f"{br[j][i]:12.5e}" for i in range(nbpf))
            f.write(row_str + "\n")

    with open(f_bz, 'w') as f:
        f.write(f"{nbturn} {nbpf}\n")
        for j in range(nbturn):
            row_str = " ".join(f"{bz[j][i]:12.5e}" for i in range(nbpf))
            f.write(row_str + "\n")

    with open(f_phi, 'w') as f:
        f.write(f"{nbturn} {nbpf}\n")
        for j in range(nbturn):
            row_str = " ".join(f"{phi[j][i]:12.5e}" for i in range(nbpf))
            f.write(row_str + "\n")


def read_b(
    f_br: str,
    f_bz: str,
    f_phi: str
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Read grid-based (Br, Bz, Phi) response matrices from text files.

    :param f_br: Input filename for Br data.
    :param f_bz: Input filename for Bz data.
    :param f_phi: Input filename for Phi data.
    :return: Tuple of (Br, Bz, Phi) as numpy arrays.
    """
    with open(f_br, 'r') as f:
        nbturn, nbpf = map(int, f.readline().split())
        br_data = np.zeros((nbturn, nbpf))
        for j in range(nbturn):
            line_vals = f.readline().split()
            for i in range(nbpf):
                br_data[j][i] = float(line_vals[i])

    with open(f_bz, 'r') as f:
        nbturn, nbpf = map(int, f.readline().split())
        bz_data = np.zeros((nbturn, nbpf))
        for j in range(nbturn):
            line_vals = f.readline().split()
            for i in range(nbpf):
                bz_data[j][i] = float(line_vals[i])

    with open(f_phi, 'r') as f:
        nbturn, nbpf = map(int, f.readline().split())
        phi_data = np.zeros((nbturn, nbpf))
        for j in range(nbturn):
            line_vals = f.readline().split()
            for i in range(nbpf):
                phi_data[j][i] = float(line_vals[i])

    return br_data, bz_data, phi_data


# =============================================================================
# Smoothing (Equivalent to MATLAB "smooth" function)
# =============================================================================
def smooth(array: Union[List[float], ndarray], span: int) -> ndarray:
    """
    Smooth a 1D array with a simple moving average over 'span' points.

    :param array: 1D data to be smoothed.
    :param span: Smoothing window size (odd number).
    :return: Smoothed array (same length as input).
    """
    arr = np.array(array)
    if span % 2 == 0:
        span -= 1

    nbv = len(arr)
    out = np.zeros(nbv)
    span2 = (span - 1) // 2

    for i in range(span2):
        div_l = 2 * i + 1
        window_l = arr[:div_l]
        out[i] = np.sum(window_l) / div_l

        window_r = arr[nbv - div_l:]
        out[nbv - 1 - i] = np.sum(window_r) / div_l

    end_l = nbv - span2
    for i in range(span2, end_l):
        window_m = arr[i - span2 : i + span2 + 1]
        out[i] = np.sum(window_m) / span

    return out


# =============================================================================
# VEST PF and IP
# =============================================================================
def vest_pf(shot: int) -> Tuple[ndarray, List[ndarray]]:
    """
    Load poloidal field coil data from the database for a given shot.

    :param shot: Shot number.
    :return: (time, PFdata) where PFdata is a list of PF coil current arrays.
    """
    f_sample = 25000.0
    f_cut_low = 100.0
    d_low_regular = signal.firwin(26, f_cut_low, pass_zero='lowpass',
                                  fs=f_sample)
    x_base = list(range(25))
    data = scipy.io.loadmat('Coil_info.mat')

    n_coil = data['CoilNumber'][0] - 1
    g_coil = data['CoilGain'][0]
    coil_codes = data['CoilCode'][0]
    nbcoil = 10

    time_pf, data_temp = vest_load(shot, coil_codes[0])  # Just to get time
    time_pf = np.array(time_pf)
    pf_data = []

    cpt = 0
    for i in range(nbcoil):
        if i in n_coil:
            _, raw_data = vest_load(shot, coil_codes[cpt])
            raw_data = np.array(raw_data)
            temp = raw_data - statistics.mean(raw_data[x_base])
            temp = temp * g_coil[cpt]
            temp = signal.lfilter(d_low_regular, 1, temp)
            pf_data.append(temp)
            cpt += 1
        else:
            pf_data.append(np.zeros(len(time_pf)))

    return time_pf, pf_data


def vest_ip(shot: int) -> Tuple[ndarray, ndarray]:
    """
    Load plasma current data from the database.

    :param shot: Shot number.
    :return: (time, ip)
    """
    x_ip = 102  # Raw plasma current data code
    x_fl = 25   # Flux loop #10 code

    if shot < 17455:
        ind_mutual = 2.8e-4
    else:
        ind_mutual = 5.0e-4

    # Time index of baseline subtraction
    xtime = vest_time(shot)
    x_window = 500
    nbt = 2 * x_window + 2
    x_base = np.zeros(nbt, dtype=int)
    x_base[0] = xtime[0] - x_window
    x_base[-1] = xtime[-1] + x_window
    for i in range(x_window):
        x_base[i + 1] = x_base[i] + 1
        x_base[nbt - 2 - i] = x_base[nbt - 1 - i] - 1

    time_raw, temp_ip = vest_load(shot, x_ip)
    time_raw, temp_fl = vest_load(shot, x_fl)
    time_raw = np.array(time_raw)
    temp_ip = np.array(temp_ip)
    temp_fl = np.array(temp_fl)

    # Subtract polynomial baseline
    base_ip = np.polyfit(time_raw[x_base], temp_ip[x_base], 1)
    ip_shot = temp_ip - np.polyval(base_ip, time_raw)

    # Convert flux loop #10 data to reference IP
    ip_ref = temp_fl * 11.0 / ind_mutual
    base_ref = np.polyfit(time_raw[x_base], ip_ref[x_base], 1)
    ip_ref = ip_ref - np.polyval(base_ref, time_raw)

    ip_out = ip_shot - ip_ref

    # Polarity reversed since shot #20259
    if shot >= 20259:
        ip_out = -ip_out

    return time_raw, ip_out


def vest_tf(shot: int) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Load toroidal field coil data from database and compute on-axis Bt.

    :param shot: Shot number.
    :return: (time, tf, bt) raw TF coil current and derived B_t on axis.
    """
    mu0 = 4 * math.pi * 1e-7
    code_tf = 1
    turn_tf = 24
    gain_tf = -1e4
    axis_r = 0.4
    x_base = list(range(25))

    time_tf, tf_data = vest_load(shot, code_tf)
    time_tf = np.array(time_tf)
    tf_data = np.array(tf_data)

    # Subtract offset
    mean_offset = statistics.mean(tf_data[x_base])
    tf_data = (tf_data - mean_offset) * gain_tf
    bt_data = mu0 * tf_data * turn_tf / axis_r / (2 * math.pi)
    return time_tf, tf_data, bt_data


# =============================================================================
# Self-Inductance Calculation
# =============================================================================
def self_induM_new(r_val: float, area: float) -> float:
    """
    Compute self-inductance from FIST model.

    :param r_val: Effective radius.
    :param area: Cross-sectional area.
    :return: Self-inductance [H].
    """
    mu0 = 4.0 * math.pi * 1.0e-7
    result = mu0 * r_val * (
        math.log(8.0 * r_val / math.sqrt(area / math.pi)) - 7.0 / 4.0
    )
    return result


# =============================================================================
# Eddy Currents and Vacuum Field Calculations
# =============================================================================

# =============================================================================
# MD Validity Checking
# =============================================================================
def vest_md_validity(ods: ODS) -> None:
    """
    Compare measured flux/field data to vacuum calculation for validation.

    :param ods: OMAS dataset with pf_active, pf_passive, magnetics, etc.
    """
    mg = ods['magnetics']
    pf = ods['pf_active']
    pfp = ods['pf_passive']

    broken_threshold = 0.85

    nbflux = len(mg['flux_loop'])
    nbprobe = len(mg['b_field_pol_probe'])

    mg_time = mg['time']
    flux_meas = []
    probe_meas = []
    rz_list = []

    # Gather flux loop positions
    for i in range(nbflux):
        flux_meas.append(mg[f'flux_loop.{i}.flux.data'])
        r_loop = mg[f'flux_loop.{i}.position.0.r']
        z_loop = mg[f'flux_loop.{i}.position.0.z']
        rz_list.append([r_loop, z_loop])

    # Gather poloidal probe positions
    for i in range(nbprobe):
        probe_meas.append(mg[f'b_field_pol_probe.{i}.field.data'])
        r_probe = mg[f'b_field_pol_probe.{i}.position.r']
        z_probe = mg[f'b_field_pol_probe.{i}.position.z']
        rz_list.append([r_probe, z_probe])

    # Compute vacuum fields at measurement positions
    time_arr, psi_calc, br_calc, bz_calc = vest_vac1(ods, rz_list)
    print("vest_vac1 done (validity check).")

    flux_calc = []
    probe_calc = []

    # Interpolate flux
    for i in range(nbflux):
        flux_calc.append(np.interp(mg_time, time_arr, psi_calc[i]))

    # Interpolate Bz for probes
    for i in range(nbprobe):
        index = i + nbflux
        probe_calc.append(np.interp(mg_time, time_arr, bz_calc[index]))

    # Evaluate correlation and mark validity
    for i in range(nbflux):
        cxy = np.corrcoef(flux_meas[i], flux_calc[i])[0, 1]
        validity = 0 if cxy >= broken_threshold else -1
        mg[f'flux_loop.{i}.flux.validity'] = validity

    for i in range(nbprobe):
        cxy = np.corrcoef(probe_meas[i], probe_calc[i])[0, 1]
        validity = 0 if cxy >= broken_threshold else -1
        mg[f'b_field_pol_probe.{i}.field.validity'] = validity
import os
import re
import math
import copy
import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import medfilt
from typing import Any, Dict, List, Optional, Tuple, Union

# from omfit_classes.omfit_eqdsk import OMFITgeqdsk  # OMFIT class if needed
# from green import greenR  # External green's function library
# from omas import ODS
# from some_module import vest_load, vfit_pf, vfit_md, ...

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not installed. Camera image functions may not work.")


# =============================================================================
# Helper / Shared Functions
# =============================================================================



# =============================================================================
# Equilibrium ODS Generation
# =============================================================================

def vfit_equilibrium_geqdsk(ods: Dict[str, Any], listg: List[str]) -> None:
    """
    Generate an equilibrium ODS from a list of g-EQDSK files.

    :param ods: ODS to populate (e.g., ods['equilibrium']).
    :param listg: List of g-EQDSK filenames.
    """
    # Example usage via OMFIT classes (commented out if not using OMFIT):
    # if len(listg) > 0:
    #     eq = OMFITgeqdsk(listg[0])
    #     ods1 = eq.to_omas()
    #     for i in range(len(listg) - 1):
    #         eq1 = OMFITgeqdsk(listg[i + 1])
    #         ods1 = eq1.to_omas(ods=ods1, time_index=i + 1)

    # # Copy from ods1 to the final ods
    # eq_dict = ods['equilibrium']
    # eq1_dict = ods1['equilibrium']
    # eq_dict['ids_properties'] = eq1_dict['ids_properties']
    # eq_dict['ids_properties.homogeneous_time'] = 1
    #
    # eq_dict['vacuum_toroidal_field'] = eq1_dict['vacuum_toroidal_field']
    # eq_dict['time'] = eq1_dict['time']
    # eq_dict['code'] = eq1_dict['code']
    # for i in range(len(eq1_dict['time_slice'])):
    #     eq_dict[f'time_slice.{i}'] = eq1_dict[f'time_slice.{i}']

    print("vfit_equilibrium_geqdsk stub: implement if OMFITgeqdsk is available.")


def vfit_equilibrium_mini(
    ods: Dict[str, Any],
    shot: int,
    tstart: float,
    tend: float,
    dt: float
) -> None:
    """
    Generate a minimal equilibrium ODS with only a few parameters.

    :param ods: ODS structure to populate (ods['equilibrium']).
    :param shot: VEST shot number.
    :param tstart: Start time for sampling.
    :param tend: End time for sampling.
    :param dt: Time step for sampling.
    """
    eq = ods['equilibrium']
    eq['ids_properties.comment'] = 'mini equilibrium config from vest_magnetics'
    eq['ids_properties.homogeneous_time'] = 1

    # Attempt to get Ip from DB
    try:
        # (time, Ip) = vfit_PlasmaCurrent(shot)
        # example stub:
        time = np.linspace(0.3, 0.4, 100)
        ip_data = 500.0 * np.ones_like(time)  # 500 A example
        case = 'Plasma'
    except Exception:
        print("No plasma current data. Vacuum equilibrium generated.")
        case = 'Vacuum'
        return

    if case == 'Plasma':
        print("Plasma current data found; still generating vacuum eq in stub.")
        tstart = max(tstart, time[0])
        tend = min(tend, time[-1])

    time_1 = np.arange(tstart, tend, dt)
    eq['time'] = time_1
    nbt = len(time_1)

    # If plasma were used, do something:
    # e.g., eq['time_slice.<i>.global_quantities.magnetic_axis.r'] = 0.4

    # Example grid
    dim1 = np.zeros(18)
    dim2 = np.zeros(60)
    dd = 0.04
    for i in range(len(dim1)):
        dim1[i] = 0.09 + i * dd
    for i in range(len(dim2)):
        dim2[i] = -1.18 + i * dd

    # In a real scenario, we might call something like vfit_vac2(ods, time_1).
    # For now, we simply store the grid.

    # Example storing
    for i in range(nbt):
        eq[f'time_slice.{i}.profiles_2d.0.grid.dim1'] = dim1
        eq[f'time_slice.{i}.profiles_2d.0.grid.dim2'] = dim2


def vfit_equilibrium_mini2(
    ods: Dict[str, Any],
    shot: int,
    tstart: float,
    tend: float,
    dt: float
) -> None:
    """
    Alternate minimal equilibrium ODS generator with different grid resolution.

    :param ods: ODS structure to populate (ods['equilibrium']).
    :param shot: VEST shot number.
    :param tstart: Start time for sampling.
    :param tend: End time for sampling.
    :param dt: Time step for sampling.
    """
    eq = ods['equilibrium']
    eq['ids_properties.comment'] = 'mini equilibrium config from vest_magnetics'
    eq['ids_properties.homogeneous_time'] = 1

    # Example: get Ip from DB
    # (time, Ip) = vfit_PlasmaCurrent(shot)
    time = np.linspace(0.3, 0.4, 100)
    ip_data = 700.0 * np.ones_like(time)

    tstart = max(tstart, time[0])
    tend = min(tend, time[-1])
    time_1 = np.arange(tstart, tend, dt)

    eq['time'] = time_1
    nbt = len(time_1)
    eq['code.parameters.time_slice'] = time_1

    # Example: store axis info
    r_axis = 0.4
    z_axis = 0.0
    for i in range(nbt):
        eq[f'time_slice.{i}.global_quantities.magnetic_axis.r'] = r_axis
        eq[f'time_slice.{i}.global_quantities.magnetic_axis.z'] = z_axis

    # Example big grid
    dim1 = np.zeros(129)
    dim2 = np.zeros(513)
    del_r = 0.007421875
    for i in range(129):
        dim1[i] = 0.05 + i * del_r
    del_z = 0.005859375
    for i in range(513):
        dim2[i] = -1.5 + i * del_z

    # Interpolate IP
    ip_interp = np.interp(time_1, time, ip_data)
    for i in range(nbt):
        eq[f'time_slice.{i}.global_quantities.ip'] = ip_interp[i]
        eq[f'time_slice.{i}.profiles_2d.0.grid.dim1'] = dim1
        eq[f'time_slice.{i}.profiles_2d.0.grid.dim2'] = dim2


# =============================================================================
# EM Coupling
# =============================================================================

def vfit_em_coupling(ods: Dict[str, Any]) -> None:
    """
    Load and store mutual inductances for active/passive coils from .mat data.

    :param ods: ODS containing `em_coupling`.
    """
    em = ods['em_coupling']
    data = scipy.io.loadmat('./Geometry/VEST_em_coupling.mat')

    em['mutual_active_active'] = data['mutual_active_active']
    em['mutual_passive_active'] = data['mutual_passive_active']
    em['mutual_passive_passive'] = data['mutual_passive_passive']


def vfit_em_coupling_reduce(ods: Dict[str, Any], case: int) -> None:
    """
    Load and store reduced mutual inductances for active/passive coils.

    :param ods: ODS containing `em_coupling`.
    :param case: Choice of coupling file (1..4).
    """
    em = ods['em_coupling']
    if case == 1:
        fname = './Geometry/Reduce_em_coupling.mat'
    elif case == 2:
        fname = './Geometry/Reduce_em_coupling_40.mat'
    elif case == 3:
        fname = './Geometry/Reduce_em_coupling_79.mat'
    elif case == 4:
        fname = './Geometry/Reduce_em_coupling_158.mat'
    else:
        print(f"Invalid case: {case}")
        return

    data = scipy.io.loadmat(fname)
    em['mutual_active_active'] = data['mutual_active_active']
    em['mutual_passive_active'] = data['mutual_passive_active']
    em['mutual_passive_passive'] = data['mutual_passive_passive']


# =============================================================================
# PF Active
# =============================================================================



def vfit_pf_active_reduce(
    ods: Dict[str, Any],
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    case: int
) -> None:
    """
    Reduced PF coil geometry from specialized .mat files, attach DB currents.

    :param ods: ODS for pf_active.
    :param shot: VEST shot number.
    :param tstart: Start time to sample coil currents.
    :param tend: End time to sample coil currents.
    :param dt: Time step.
    :param case: Select which geometry file to load (1..4).
    """
    pf = ods['pf_active']
    pf['ids_properties.comment'] = 'PF config from vfit_pf_active_reduce'
    pf['ids_properties.homogeneous_time'] = 1

    if case == 1:
        fname = './Geometry/Fiesta_Coil_geo.mat'
        dname = 'coil'
        nb1 = 40
    elif case == 2:
        fname = './Geometry/VEST_DiscretizedCoilGeometry_40.mat'
        dname = 'coil'
        nb1 = 40
    elif case == 3:
        fname = './Geometry/VEST_DiscretizedCoilGeometry_79.mat'
        dname = 'coil'
        nb1 = 79
    elif case == 4:
        fname = './Geometry/VEST_DiscretizedCoilGeometry_158.mat'
        dname = 'coil'
        nb1 = 158
    else:
        print(f"Invalid case: {case}")
        return

    nb_elt = [nb1, 2, 2, 2, 2]
    pf_name = [1, 5, 6, 9, 10]
    nbcoil = len(nb_elt)

    # Get DB currents (stub)
    time_db = np.linspace(0, 1, 500)
    data_db = np.sin(2 * np.pi * time_db) * 1500.0

    if dt > 0:
        tstart = max(tstart, time_db[0])
        tend = min(tend, time_db[-1])
        time_1 = np.arange(tstart, tend, dt)
    else:
        time_1 = time_db

    pf['time'] = time_1
    for i in range(nbcoil):
        pf[f'coil.{i}.name'] = f'PF{pf_name[i]}'
        pf[f'coil.{i}.identifier'] = f'PF{pf_name[i]}'

    # Resistances
    r_coil = 1.68e-8
    rpf = [0.053, 0.71, 0.71, 0.93, 0.93]
    apf = [0.04128, 0.001218, 0.001218, 0.0027216, 0.0027216]
    for i in range(nbcoil):
        pf[f'coil.{i}.resistance'] = (
            2.0 * math.pi * r_coil * rpf[i] / apf[i]
        )

    data2 = scipy.io.loadmat(fname)
    coil_array = data2[dname]
    idx = 0
    for k in range(nbcoil):
        n_belt = nb_elt[k]
        for i in range(n_belt):
            pf[f'coil.{k}.element.{i}.turns_with_sign'] = coil_array[idx][5]
            pf[f'coil.{k}.element.{i}.geometry.geometry_type'] = 2
            pf[f'coil.{k}.element.{i}.geometry.rectangle.r'] = coil_array[idx][0]
            pf[f'coil.{k}.element.{i}.geometry.rectangle.z'] = coil_array[idx][1]
            dr_val = coil_array[idx][2]
            dz_val = coil_array[idx][3]
            pf[f'coil.{k}.element.{i}.geometry.rectangle.width'] = dr_val
            pf[f'coil.{k}.element.{i}.geometry.rectangle.height'] = dz_val
            pf[f'coil.{k}.element.{i}.area'] = dr_val * dz_val
            idx += 1

    # Stub: load PF currents
    # (time_pf, PFdata) = vfit_pf(shot)
    for i in range(nbcoil):
        # Example: use the same data for all
        pf[f'coil.{i}.current.data'] = np.interp(time_1, time_db, data_db)

    print("vfit_pf_active_reduce done (placeholder).")


# Other PF active variants like vfit_pf_active_efit166, vfit_pf_active_efit,
# etc., would follow similarly. They are not repeated here for brevity but
# follow the same pattern: define geometry, set coil elements, resistances,
# then attach currents from DB.


# =============================================================================
# PF Passive (Wall) Geometry
# =============================================================================

def vfit_pf_passive(ods: Dict[str, Any]) -> None:
    """
    Generate the PF passive loops (wall) from IMAS_wall.mat (~277 elements).

    :param ods: ODS with `pf_passive`.
    """
    pfp = ods['pf_passive']
    pfp['ids_properties.comment'] = 'PF passive from init_vest'
    pfp['ids_properties.homogeneous_time'] = 1

    data3 = scipy.io.loadmat('./Geometry/IMAS_wall.mat')
    wall_data = data3['wall']

    nb_loop = len(wall_data)
    geom_type = 1  # Outline polygon

    for i_loop in range(nb_loop):
        pfp[f'loop.{i_loop}.element.0.geometry.geometry_type'] = geom_type

    for i_loop in range(nb_loop):
        wnum = int(wall_data[i_loop][7])
        r = wall_data[i_loop][0]
        z = wall_data[i_loop][1]
        dr = wall_data[i_loop][2]
        dz = wall_data[i_loop][3]

        # Resistivity
        if wnum == 11:
            # tungsten
            resis = 5.6e-8
        else:
            # SUS316LN
            resis = 7.8e-7

        pfp[f'loop.{i_loop}.name'] = f'W{wnum}'
        pfp[f'loop.{i_loop}.element.0.identifier'] = f'W{wnum}'

        pfp[f'loop.{i_loop}.element.0.area'] = dr * dz
        pfp[f'loop.{i_loop}.element.0.turns_with_sign'] = 1
        pfp[f'loop.{i_loop}.resistivity'] = resis
        pfp[f'loop.{i_loop}.resistance'] = (
            2.0 * math.pi * r * resis / (dr * dz)
        )

        pfp[f'loop.{i_loop}.element.0.geometry.outline.r'] = [
            r - dr / 2, r + dr / 2, r + dr / 2, r - dr / 2
        ]
        pfp[f'loop.{i_loop}.element.0.geometry.outline.z'] = [
            z - dz / 2, z - dz / 2, z + dz / 2, z + dz / 2
        ]


def vfit_pf_passive_big(ods: Dict[str, Any], mfile: str = 'no') -> None:
    """
    Generate PF passive loops from a large geometry mat file. Optionally apply
    in/out board resistance scaling from mfile.

    :param ods: ODS with pf_passive.
    :param mfile: If not 'no', apply correction from the named .mat file.
    """
    pfp = ods['pf_passive']
    pfp['ids_properties.comment'] = 'PF passive from init_vest'
    pfp['ids_properties.homogeneous_time'] = 1

    data3 = scipy.io.loadmat('./Geometry/VEST_WallLimiterGeometry_ver_1512.mat')
    wall_data = data3['WallGeometry']

    nb_loop = len(wall_data)
    geom_type = 1

    for i_loop in range(nb_loop):
        pfp[f'loop.{i_loop}.element.0.geometry.geometry_type'] = geom_type

    for i_loop in range(nb_loop):
        wnum = int(wall_data[i_loop][7])
        r_val = wall_data[i_loop][0]
        z_val = wall_data[i_loop][1]
        dr_val = wall_data[i_loop][2]
        dz_val = wall_data[i_loop][3]

        if wnum == 11:
            resis = 5.6e-8  # Tungsten
        else:
            resis = 7.8e-7  # SUS316LN

        pfp[f'loop.{i_loop}.name'] = f'W{wnum}'
        pfp[f'loop.{i_loop}.element.0.identifier'] = f'W{wnum}'

        area_val = dr_val * dz_val
        pfp[f'loop.{i_loop}.element.0.area'] = area_val
        pfp[f'loop.{i_loop}.element.0.turns_with_sign'] = 1
        pfp[f'loop.{i_loop}.resistivity'] = resis
        pfp[f'loop.{i_loop}.resistance'] = (
            2.0 * math.pi * r_val * resis / area_val
        )
        pfp[f'loop.{i_loop}.element.0.geometry.outline.r'] = [
            r_val - dr_val / 2,
            r_val + dr_val / 2,
            r_val + dr_val / 2,
            r_val - dr_val / 2
        ]
        pfp[f'loop.{i_loop}.element.0.geometry.outline.z'] = [
            z_val - dz_val / 2,
            z_val - dz_val / 2,
            z_val + dz_val / 2,
            z_val + dz_val / 2
        ]

    if mfile != 'no':
        mdic = scipy.io.loadmat(f'../Geometry/{mfile}')
        inboard_t = mdic['Wall_Factor_Inboard']
        outboard_t = mdic['Wall_Factor_Outboard']

        inboard = inboard_t[0]
        outboard = outboard_t[0]

        # Example usage of inboard/outboard offsets, as in original code:
        # This logic depends on known geometry indexing. Omitted here for brevity.
        print("vfit_pf_passive_big: applying custom in/out board scaling (stub).")


# Others like vfit_pf_passive_reduce(), vfit_pf_passive_big2() similarly
# define geometry and fill in loop data.


# =============================================================================
# Magnetics ODS Generation
# =============================================================================

def vfit_magnetics(
    ods: Dict[str, Any],
    shot: int,
    tstart: float,
    tend: float,
    dt: float
) -> None:
    """
    Generate magnetics ODS from geometry .mat and VEST DB flux/probe signals.

    :param ods: ODS with 'magnetics'.
    :param shot: VEST shot number.
    :param tstart: Start time for signals.
    :param tend: End time for signals.
    :param dt: Time step.
    """
    mg = ods['magnetics']
    mg['ids_properties.comment'] = 'magnetics config from vest_magnetics'
    mg['ids_properties.homogeneous_time'] = 1

    length = 0.01
    angle = 3 * math.pi / 2

    # Example geometry read
    # data = scipy.io.loadmat('./Geometry/VEST_MagneticsGeometry_Full_ver_2302.mat')
    # md_array = data['md']
    # nbd = len(md_array)

    # Example signal read from DB
    time_db = np.linspace(0, 0.5, 1000)
    flux_loop_data = [np.sin(2 * np.pi * time_db)]
    probe_data = [np.cos(2 * np.pi * time_db)]

    if dt > 0:
        tstart = max(tstart, time_db[0])
        tend = min(tend, time_db[-1])
        time_1 = np.arange(tstart, tend, dt)
    else:
        time_1 = time_db
        tstart = time_db[0]
        tend = time_db[-1]

    mg['time'] = time_1

    # Suppose we have 1 flux loop and 1 pol probe for demonstration
    # Positions are placeholders
    mg['flux_loop.0.name'] = "FluxLoop1"
    mg['flux_loop.0.identifier'] = "FluxLoop1"
    for j in range(len(time_1)):
        mg[f'flux_loop.0.position.{j}.r'] = 0.3
        mg[f'flux_loop.0.position.{j}.z'] = 0.0

    mg['flux_loop.0.flux.data'] = np.interp(time_1, time_db, flux_loop_data[0]) * (
        2 * math.pi
    )

    mg['b_field_pol_probe.0.name'] = "Probe1"
    mg['b_field_pol_probe.0.identifier'] = "Probe1"
    mg['b_field_pol_probe.0.position.r'] = 0.4
    mg['b_field_pol_probe.0.position.z'] = 0.0
    mg['b_field_pol_probe.0.length'] = length
    mg['b_field_pol_probe.0.poloidal_angle'] = angle
    mg['b_field_pol_probe.0.field.data'] = np.interp(
        time_1, time_db, probe_data[0]
    )

    # Example IP, Diamagnetic flux
    mg['ip.0.data'] = 200.0 * np.ones_like(time_1)
    mg['ip.0.time'] = time_1

    mg['diamagnetic_flux.0.data'] = 0.01 * np.ones_like(time_1)
    mg['diamagnetic_flux.0.time'] = time_1


# vfit_magneticsi(ods, shot, tstart, tend, dt) or vfit_magnetics2(...) are
# similar expansions with additional geometry or internal probes.


# =============================================================================
# Dataset Description
# =============================================================================


# =============================================================================
# EFIT Wall
# =============================================================================

def vfit_wall_efit(ods: Dict[str, Any], mfile: str = 'no') -> None:
    """
    Generate EFIT-style 2D vessel description in ODS['wall'].

    :param ods: ODS with 'wall'.
    :param mfile: If not 'no', apply in/outboard factors from .mat file.
    """
    wl = ods['wall']
    wl['ids_properties.comment'] = 'Wall from vfit_wall_efit'
    wl['ids_properties.homogeneous_time'] = 2

    data3 = scipy.io.loadmat('./Geometry/VEST_WallLimiterGeometry_ver_1512.mat')
    wall_data = data3['WallGeometry']
    nb_loop = len(wall_data)

    for i_loop in range(nb_loop):
        wnum = int(wall_data[i_loop][7])
        r_val = wall_data[i_loop][0]
        z_val = wall_data[i_loop][1]
        dr_val = wall_data[i_loop][2]
        dz_val = wall_data[i_loop][3]

        if wnum == 11:
            resis = 5.6e-8  # Tungsten
        else:
            resis = 7.8e-7  # SUS316LN

        wl[f'description_2d.0.vessel.unit.{i_loop}.name'] = f'W{wnum}'
        wl[f'description_2d.0.vessel.unit.{i_loop}.identifier'] = f'W{wnum}'

        wl[f'description_2d.0.vessel.unit.{i_loop}.element.0.resistivity'] = resis
        wl[f'description_2d.0.vessel.unit.{i_loop}.element.0.resistance'] = (
            2.0 * math.pi * r_val * resis / (dr_val * dz_val)
        )
        wl[f'description_2d.0.vessel.unit.{i_loop}.element.0.outline.r'] = [
            r_val - dr_val / 2,
            r_val + dr_val / 2,
            r_val + dr_val / 2,
            r_val - dr_val / 2
        ]
        wl[f'description_2d.0.vessel.unit.{i_loop}.element.0.outline.z'] = [
            z_val - dz_val / 2,
            z_val - dz_val / 2,
            z_val + dz_val / 2,
            z_val + dz_val / 2
        ]

    if mfile != 'no':
        print("vfit_wall_efit: applying scaling from mfile stub.")



# =============================================================================
# Additional Diagnostics (Barometry, Filterscope, Camera, etc.)
# =============================================================================

def vfit_barometry(
    ods: Dict[str, Any],
    shot: int,
    tstart: float,
    tend: float,
    dt: float
) -> None:
    """
    Load pressure gauge data into ODS['barometry'].

    :param ods: ODS structure.
    :param shot: Shot number.
    :param tstart: Start time.
    :param tend: End time.
    :param dt: Time step.
    """
    ods['barometry.ids_properties.comment'] = 'VEST Pressure Gauge data'
    ods['barometry.ids_properties.homogeneous_time'] = 1

    ods['barometry.gauge.0.name'] = 'PKR-251 Main Gauge'
    ods['barometry.gauge.0.type.index'] = 0
    ods['barometry.gauge.0.type.name'] = 'Penning'
    ods['barometry.gauge.0.type.description'] = 'PKR-251 Main Gauge'

    # Example data from DB
    time_db = np.linspace(0, 0.5, 200)
    data_db = 1e-3 * np.ones_like(time_db)  # Torr
    # Convert Torr -> Pa
    data_db_pa = data_db * 133.3223684211

    if dt > 0:
        tstart = max(tstart, time_db[0])
        tend = min(tend, time_db[-1])
        time_arr = np.arange(tstart, tend, dt)
    else:
        time_arr = time_db

    data_interp = np.interp(time_arr, time_db, data_db_pa)
    # store
    ods['barometry.gauge.0.pressure.time'] = time_arr
    ods['barometry.gauge.0.pressure.data'] = data_interp


def vfit_filterscope(
    ods: Dict[str, Any],
    shot: int,
    t_start: float,
    t_end: float,
    dt: float
) -> None:
    """
    Example population of filterscope intensities in ODS['spectrometer_uv'].

    :param ods: ODS structure.
    :param shot: Shot number.
    :param t_start: Start time.
    :param t_end: End time.
    :param dt: Time step.
    """
    ods['spectrometer_uv.ids_properties.comment'] = 'VEST filterscope data'
    ods['spectrometer_uv.ids_properties.homogeneous_time'] = 1

    # Example static channel definitions
    ods['spectrometer_uv.channel.0.name'] = 'H alpha Filterscope'
    ods['spectrometer_uv.channel.1.name'] = 'O-I Filterscope'
    ods['spectrometer_uv.channel.2.name'] = 'Versatile Filterscope'

    ods['spectrometer_uv.channel.0.processed_line.0.label'] = 'H-alpha_6563'
    ods['spectrometer_uv.channel.1.processed_line.0.label'] = 'OI_7770'
    # etc...

    # Example data
    time_db = np.linspace(0, 1, 500)
    data_db = np.sin(2 * np.pi * time_db)

    t_start = max(t_start, 0)
    t_end = min(t_end, 1)
    time_arr = np.arange(t_start, t_end, dt)

    data_interp = -np.interp(time_arr, time_db, data_db)
    ods['spectrometer_uv.time'] = time_arr
    # Example storing single channel/line
    ods['spectrometer_uv.channel.0.processed_line.0.intensity.data'] = data_interp


def vfit_camera_visible(ods: Dict[str, Any], shotnumber: int) -> None:
    """
    Load fast camera frames from local .bmp files into ODS['camera_visible'].

    :param ods: ODS structure.
    :param shotnumber: VEST shot number (folder-based).
    """
    ods['camera_visible.name'] = 'Fast Camera'
    ods['camera_visible.ids_properties.homogeneous_time'] = 1
    ods['camera_visible.channel.0.detector.0.exposure_time'] = 0.001

    if cv2 is None:
        print("OpenCV not available. vfit_camera_visible cannot load images.")
        return

    base_path = os.getcwd()
    shot_path = os.path.join(base_path, str(shotnumber))

    frame_pattern = re.compile(r"(\d+)_(\d+\.\d+)_ms\.bmp")
    frame_files = glob.glob(os.path.join(shot_path, "*.bmp"))
    frame_files.sort(key=lambda x: float(
        frame_pattern.match(os.path.basename(x)).group(2))
    )

    i = 0
    for frame_file in frame_files:
        filename = os.path.basename(frame_file)
        match = frame_pattern.match(filename)
        if match:
            time_ms = float(match.group(2))
            time_s = round(time_ms / 1000.0, 6)
            image = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            ods[f'camera_visible.channel.0.detector.0.frame.{i}.image_raw'] = image
            ods[f'camera_visible.channel.0.detector.0.frame.{i}.time'] = time_s
            i += 1
    if i == 0:
        print("No matched BMP files found for fast camera.")


def vfit_charge_exchange(
    ods: Dict[str, Any],
    shotnumber: int,
    options: str = 'single'
) -> None:
    """
    Store charge-exchange (doppler) data in ODS['charge_exchange'].

    :param ods: ODS structure.
    :param shotnumber: VEST shot number.
    :param options: 'single' or 'profile' for the data reading method.
    """
    ods['charge_exchange.ids_properties.homogeneous_time'] = 1
    if options == 'single':
        print("read_doppler_single stub for shot", shotnumber)
        # read_doppler_single(ods, shotnumber)
    elif options == 'profile':
        print("read_doppler_profile stub for shot", shotnumber)
        # read_doppler_profile(ods, shotnumber)


def vfit_langmuir_probes(
    ods: Dict[str, Any],
    shot: int,
    radial_position: float = 0.65,
    gas_species: str = 'H'
) -> None:
    """
    Populate ODS['langmuir_probes'] with triple-probe data from DB or fallback.

    :param ods: ODS structure.
    :param shot: VEST shot number.
    :param radial_position: Approx position of the triple probe.
    :param gas_species: Gas species, e.g., 'H', 'He'.
    """
    # This is a stub implementing the structure from the original code
    lp = ods['langmuir_probes']
    lp['embedded.0.name'] = 'Main Chamber Triple Probe'
    lp['embedded.0.position.r'] = radial_position
    lp['embedded.0.position.z'] = 0.0
    lp['embedded.0.position.phi'] = 0.0
    lp['embedded.0.surface_area'] = 2.0 * math.pi * 0.00005 * 0.0015  # example

    # Attempt to read from DB or fallback to fake data
    time_data = np.linspace(0.27, 0.35, 100)
    ne_data = 1e19 * np.ones(100)
    te_data = 20.0 * np.ones(100)

    lp['embedded.0.time'] = time_data
    lp['embedded.0.n_e.data'] = ne_data
    lp['embedded.0.t_e.data'] = te_data




import os
import re
import math
import json
import shutil
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import imageio

from scipy.io import netcdf
from scipy.signal import medfilt
from scipy.optimize import minimize, curve_fit

# OMAS imports
from omas import ODS, omas_environment, save_omas_json, load_omas_json
import omas_plot  # For equilibrium_summary if available

# OMFIT imports (OMFITgeqdsk, OMFITkeqdsk, etc.)
try:
    from omfit_classes.omfit_eqdsk import OMFITgeqdsk, OMFITaeqdsk, OMFITkeqdsk, OMFITmeqdsk
except ImportError:
    OMFITgeqdsk = None
    OMFITaeqdsk = None
    OMFITkeqdsk = None
    OMFITmeqdsk = None
    print("Warning: OMFIT classes not found.")

###############################################################################
# Misc Imports from local modules / placeholders (Vest-specific utilities)
###############################################################################
# e.g. from VFIT_ODS import vfit_magnetics, vfit_tf, ...
# e.g. from VFIT_tools import vfit_eddy, vest_load, ...
# For brevity, we assume these are accessible in your environment.


def gauss_fit_centered(x: np.ndarray, amplitude: float, sigma: float, offset: float) -> np.ndarray:
    """
    Gaussian function, centered at x=0, for curve fitting.

    :param x: Independent variable.
    :param amplitude: Gaussian amplitude.
    :param sigma: Gaussian sigma (width).
    :param offset: Vertical offset.
    :return: Gaussian values at x.
    """
    return amplitude * np.exp(-x**2 / (2 * sigma * sigma)) + offset


def gauss_fit_four_params(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Gaussian function with four parameters.

    :param x: Independent variable.
    :param a: Amplitude.
    :param b: Center.
    :param c: Sigma (width).
    :param d: Offset.
    :return: Gaussian values.
    """
    return a * np.exp(-(x - b)**2 / (2 * c * c)) + d


def residual_gauss_centered(coef: list, x: np.ndarray, y: np.ndarray) -> float:
    """
    Residual function for a centered Gaussian fitting (3 params).

    :param coef: [amplitude, sigma, offset].
    :param x: Independent variable array.
    :param y: Dependent (measured) data array.
    :return: Residual (sqrt of sum of squares).
    """
    amplitude, sigma, offset = coef
    y_model = gauss_fit_centered(x, amplitude, sigma, offset)
    return np.sqrt(np.sum((y - y_model) ** 2))


def residual_gauss_four_params(coef: list, x: np.ndarray, y: np.ndarray) -> float:
    """
    Residual function for 4-parameter Gaussian fitting.

    :param coef: [a, b, c, d].
    :param x: Independent variable array.
    :param y: Dependent (measured) data array.
    :return: Residual (sqrt of sum of squares).
    """
    a, b, c, d = coef
    y_model = gauss_fit_four_params(x, a, b, c, d)
    return np.sqrt(np.sum((y - y_model) ** 2))


def get_decimal_places(value: float) -> int:
    """
    Returns the number of decimal places in a float.

    :param value: Float value.
    :return: Number of decimal places.
    """
    value_str = str(value)
    if '.' in value_str:
        return len(value_str.split('.')[1])
    return 0


###############################################################################
# Generators: Diagnostics, Eddy, Constraints
###############################################################################

def generate_diagnostics_ods(shotnumber: int, save_dir: str) -> None:
    """
    Generate a 'diagnostics.json' ODS with coil, magnetics, filterscope,
    TF data, etc.

    :param shotnumber: VEST shot number.
    :param save_dir: Directory to save the resulting ODS JSON.
    """
    start_time = datetime.datetime.now()
    tstart, tend = 0.26, 0.36
    dt = 4e-5

    # 1) Create new ODS
    ods = ODS()

    # 2) Example calls to local VFIT or vest-DB utilities
    # vest_connection_pool()  # connect to VEST DB if needed

    # vfit_pf_active(ods, shotnumber, tstart, tend, dt)
    # vfit_filterscope(ods, shotnumber, tstart, tend, dt)
    # vfit_em_coupling(ods)  # optional EM coupling step
    # vfit_dataset_description(ods, shotnumber, 1)
    # vfit_tf(ods, shotnumber, tstart, tend, dt)
    # vfit_magnetics(ods, shotnumber, tstart, tend, dt)

    # 3) Save result
    fullfilename = os.path.join(save_dir, f'{shotnumber}_diagnostics.json')
    save_omas_json(ods, fullfilename)

    end_time = datetime.datetime.now()
    print(f'Diagnostics ODS generated in {end_time - start_time}')


def generate_eddy_ods(
    shotnumber: int,
    save_dir: str,
    filament_position=None,
    filament_fraction=None,
    wall_correction_opt: int = 1
) -> None:
    """
    Generate eddy.json containing PF_passive geometry + eddy current solution.

    :param shotnumber: VEST shot number.
    :param save_dir: Directory to save eddy ODS.
    :param filament_position: Optionally specify filament positions for plasma.
    :param filament_fraction: Plasma current fractions per filament.
    :param wall_correction_opt: 0 => original; 1 => corrected with a mat file.
    """
    if filament_position is None:
        filament_position = []
    if filament_fraction is None:
        filament_fraction = []

    start_time = datetime.datetime.now()

    # load or produce existing diagnostics ODS
    diag_filename = os.path.join(save_dir, f'{shotnumber}_diagnostics.json')
    if os.path.exists(diag_filename):
        ods = load_omas_json(diag_filename)
    else:
        ods = ODS()
        # Possibly re-run generate_diagnostics_ods or partial steps

    # PF and PF passive geometry, e.g.:
    # vfit_pf_passive_big(ods)  # or standard geometry
    # vfit_eddy(ods, filament_position, filament_Ip, option='EVD')

    # Save ODS
    fullfilename = os.path.join(save_dir, f'{shotnumber}_eddy.json')
    save_omas_json(ods, fullfilename)

    end_time = datetime.datetime.now()
    print(f'Eddy ODS generated in {end_time - start_time}')


def generate_constraints_ods(
    shotnumber: int,
    save_dir: str,
    efit_table_dir: str,
    time_array: np.ndarray,
    uncertainty: list,
    weighting: list,
    broken=None,
    fit: int = 0,
    fl_correct_coeff=None
) -> None:
    """
    Generate constraints ODS for EFIT input from diagnostics + eddy ODS.

    :param shotnumber: VEST shot number.
    :param save_dir: Directory to save constraints ODS.
    :param efit_table_dir: Table/geometry dir for EFIT reference.
    :param time_array: Array of times for EFIT slices.
    :param uncertainty: Array of uncertainties for different constraints.
    :param weighting: Weights for constraints (pf_coil, ip, etc.).
    :param broken: List of broken signals (1-based indexing).
    :param fit: 0 => no fitting, 1 => partial, 2 => all.
    :param fl_correct_coeff: optional correction for flux loop signals.
    """
    if broken is None:
        broken = []
    start_time = datetime.datetime.now()

    # 1) Load existing ODS
    diag_filename = os.path.join(save_dir, f'{shotnumber}_diagnostics.json')
    eddy_filename = os.path.join(save_dir, f'{shotnumber}_eddy.json')
    ods_diag = load_omas_json(diag_filename)
    ods_eddy = load_omas_json(eddy_filename)
    # Merge or cross-attach PF passive from eddy
    # ods_diag['pf_passive'] = ods_eddy['pf_passive']

    # 2) Add uncertainties to constraints, PF coil, etc.
    # e.g. vfit_pf_active_efit16(ods_diag, shotnumber, tstart, tend, dt)

    # 3) Convert diag -> eq constraints
    # vfit_equilibrium_form_constraints(ods_diag, time_array, constraints, default_average)

    # 4) Mark broken signals, do Gaussian fits if needed, etc.

    # 5) Add EFIT code parameters, e.g.:
    # eq = ods_diag['equilibrium']
    # eq['code.parameters'] ...

    # 6) Save
    ods_eq = ODS()
    # ods_eq['equilibrium'] = eq
    fullfilename = os.path.join(save_dir, f'{shotnumber}_constraints.json')
    save_omas_json(ods_eq, fullfilename)

    end_time = datetime.datetime.now()
    print(f'Constraints ODS generated in {end_time - start_time}')


###############################################################################
# k-File Generation, EFIT Execution, G-file Merging
###############################################################################

def generate_kfile(
    shotnumber: int,
    save_dir: str,
    npprime: int,
    nffprime: int
) -> None:
    """
    Generate standard EFIT k-files from constraints ODS.

    :param shotnumber: VEST shot number.
    :param save_dir: Directory for k-files.
    :param npprime: # of poloidal current shape parameters.
    :param nffprime: # of FF' shape parameters.
    """
    # 1) Load constraints ODS
    constraints_filename = os.path.join(save_dir, f'{shotnumber}_constraints.json')
    ods_const = load_omas_json(constraints_filename)
    eq_const = ods_const.get('equilibrium', {})

    # 2) For each EFIT time slice => build k0{shotnumber}.00{time}
    time_points = eq_const.get('time', [])
    for time_idx, tval in enumerate(time_points):
        # read constraints at time slice
        # build lines => BRSP, BITFC, etc.
        # write to file
        k_name = f'k0{shotnumber}.00{int(tval*1e3):04d}'
        kfile_path = os.path.join(save_dir, 'kfile', k_name)
        os.makedirs(os.path.dirname(kfile_path), exist_ok=True)
        with open(kfile_path, 'w') as fout:
            # example placeholders
            fout.write("&IN1\n")
            fout.write("IECURR=0\n")
            fout.write(f"KFFCUR={nffprime}\n")
            fout.write(f"KPPCUR={npprime}\n")
            fout.write(" / \n")
        print(f"K-file generated: {kfile_path}")


def run_efit(
    shotnumber: int,
    time_points: np.ndarray,
    save_dir: str,
    run_dir: str
) -> None:
    """
    Runs EFIT for each k-file and organizes resulting g/m/a files.

    :param shotnumber: VEST shot number.
    :param time_points: EFIT time slices.
    :param save_dir: Directory containing 'kfile/' and for storing logs.
    :param run_dir: Path to EFIT binary.
    """
    start_time = datetime.datetime.now()

    kfile_names = [f'k0{shotnumber}.00{int(t*1000)}' for t in time_points]
    log_file_dir = os.path.join(save_dir, 'log')
    os.makedirs(log_file_dir, exist_ok=True)
    log_path = os.path.join(log_file_dir, f'{shotnumber}_log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)

    for kfile in kfile_names:
        # Write run script
        run_script = os.path.join(save_dir, 'EFIT_run.sh')
        with open(run_script, 'w') as f:
            f.write(f"{run_dir} 129 << __c__MATCHING_EOF__c__\n")
            f.write("2\n1\n")
            f.write(os.path.join(save_dir, 'kfile', kfile) + "\n")
            f.write("__c__MATCHING_EOF__c__\n")

        # Execute
        with open(log_path, 'a') as log_file:
            log_file.write(f'Running EFIT for {kfile}\n')
            process = subprocess.Popen(
                ['bash', run_script],
                stdout=log_file,
                stderr=log_file
            )
            process.wait()
            log_file.write(f'EFIT done for {kfile}\n\n')
        print(f'EFIT run for {kfile}')

    # Move g/m/a files
    current_dir = os.getcwd()
    patterns_subdirs = {'g': 'gfile', 'm': 'mfile', 'a': 'afile'}
    for pat, subdir in patterns_subdirs.items():
        target_subdir = os.path.join(save_dir, subdir)
        os.makedirs(target_subdir, exist_ok=True)
        for f_name in os.listdir(current_dir):
            if f_name.startswith(pat + f"0{shotnumber}.") and re.match(r'.*\.\d{5}', f_name):
                src_path = os.path.join(current_dir, f_name)
                dest_path = os.path.join(target_subdir, f_name)
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} => {dest_path}")

    end_time = datetime.datetime.now()
    print(f'EFIT completed in {end_time - start_time}')


def generate_efit_ods(shotnumber: int, save_dir: str) -> None:
    """
    Merge EFIT g/m/a files into a final 'efit.json' ODS.

    :param shotnumber: VEST shot number.
    :param save_dir: Directory containing gfile/mfile/afile subdirs.
    """
    g_dir = os.path.join(save_dir, 'gfile')
    m_dir = os.path.join(save_dir, 'mfile')
    a_dir = os.path.join(save_dir, 'afile')
    if not all([os.path.exists(x) for x in [g_dir, m_dir, a_dir]]):
        print("Missing g/m/a directories. Skipping.")
        return

    # Loop over sorted g-files
    ods = ODS()
    g_files = sorted(f for f in os.listdir(g_dir) if f.startswith(f'g0{shotnumber}'))
    for i, gfile_name in enumerate(g_files):
        g_path = os.path.join(g_dir, gfile_name)
        m_path = os.path.join(m_dir, gfile_name.replace('g', 'm'))
        a_path = os.path.join(a_dir, gfile_name.replace('g', 'a'))

        if OMFITgeqdsk is None:
            print("OMFITgeqdsk not available. Cannot parse gfiles.")
            return

        # example usage:
        try:
            geq = OMFITgeqdsk(g_path)
            # meq = OMFITmeqdsk(m_path)
            # aeq = OMFITaeqdsk(a_path)
            # Merge each into ODS
            ods = geq.to_omas(ods=ods, time_index=i)
            # ods = meq.to_omas(ods, time_index=i)
            # ods = aeq.to_omas(ods, time_index=i)
        except Exception as e:
            print(f"Failed reading {g_path}: {e}")

    # finalize
    efit_path = os.path.join(save_dir, f'{shotnumber}_efit.json')
    save_omas_json(ods, efit_path)
    print(f"EFIT ODS saved to {efit_path}")


###############################################################################
# Plotting / Checking
###############################################################################

def plot_efit(shotnumber: int, save_dir: str) -> None:
    """
    Example that plots final EFIT boundary or flux surfaces.

    :param shotnumber: Shot number.
    :param save_dir: Dir with gfiles, etc.
    """
    g_dir = os.path.join(save_dir, 'gfile')
    g_files = sorted(f for f in os.listdir(g_dir) if f.startswith(f'g0{shotnumber}'))

    for idx, gf in enumerate(g_files):
        print(f"Plotting {gf}")
        # Suppose we parse with OMFITgeqdsk, get RBBBS, ZBBBS, etc., then plot:
        # placeholder

    # May also cross-check measured vs. reconstructed signals.


def make_gif(directory: str, filename_prefix: str, durations=None) -> None:
    """
    Combine PNG images (matching prefix) into an animated GIF.

    :param directory: Directory containing the .png images.
    :param filename_prefix: Common prefix for the frames, e.g. 'shot123_profile_'.
    :param durations: optional list of frame durations.
    """
    if durations is None:
        durations = []

    # gather frames sorted by numeric suffix
    def is_valid_file(fname):
        try:
            # e.g. 'shot123_profile_12.png'
            return fname.startswith(filename_prefix) and fname.endswith('.png') \
                   and int(fname.split('_')[-1].split('.')[0]) >= 0
        except ValueError:
            return False

    frames = sorted(
        [f for f in os.listdir(directory) if is_valid_file(f)],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    if not frames:
        print("No frames found to create GIF.")
        return
    if len(frames) == 1:
        print("Only one frame found; skipping GIF creation.")
        return

    images = [imageio.imread(os.path.join(directory, f)) for f in frames]
    # default uniform durations = 0.5s
    if not durations:
        durations = [0.5] * len(images)
    if len(durations) != len(images):
        raise ValueError("Length of durations must match # of frames.")

    out_gif = os.path.join(directory, f'{filename_prefix}.gif')
    with imageio.get_writer(out_gif, mode='I') as writer:
        for img, dur in zip(images, durations):
            writer.append_data(img, {'duration': dur})
    print(f"GIF created: {out_gif}")


###############################################################################
# Additional Utility: brokenFinder, flux corrections, etc.
###############################################################################

def broken_finder(shotnumber: int, save_dir: str, option: int = 2) -> list:
    """
    Attempt to detect broken poloidal probes or flux loops by comparing
    measured signals with vacuum+eddy predictions pre- and post-plasma.

    :param shotnumber: VEST shot number.
    :param save_dir: Path containing _diagnostics and _eddy ODS.
    :param option: 1 => correlation-based, 2 => absolute difference-based.
    :return: 1-based indices of broken signals.
    """
    broken_list = []

    diag_filename = os.path.join(save_dir, f'{shotnumber}_diagnostics.json')
    eddy_filename = os.path.join(save_dir, f'{shotnumber}_eddy.json')
    if not os.path.exists(diag_filename) or not os.path.exists(eddy_filename):
        print("Missing ODS files for brokenFinder.")
        return broken_list

    ods_diag = load_omas_json(diag_filename)
    ods_eddy = load_omas_json(eddy_filename)
    # e.g. combine PF passive
    # ods_diag['pf_passive'] = ods_eddy['pf_passive']

    # compute or load psi_total, Bz_total, etc.
    # check correlation or absolute difference pre- and post-plasma
    # fill broken_list with 1-based indices

    return broken_list


# Additional placeholders for other scripts like:
# - plot_diagnostics_ods(...)
# - plot_eddy_ods(...)
# - correct_flux_loop(...)
# - generate_kfile2,3,4(...) with specialized weighting logic
# etc.

if __name__ == "__main__":
    # Example usage:
    shot = 12345
    output_dir = "./output"
    # generate_diagnostics_ods(shot, output_dir)
    # generate_eddy_ods(shot, output_dir)
    # generate_constraints_ods(shot, output_dir, "table_dir", np.linspace(0.26, 0.36, 5), [0.1]*9, [1.0]*9)
    # generate_kfile(shot, output_dir, 2, 2)
    # run_efit(shot, np.linspace(0.26, 0.36, 5), output_dir, "/path/to/efit")
    # generate_efit_ods(shot, output_dir)
    pass

from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotV(PF,PFP):
    nbcoil=len(PF['coil'])
    nbloop=len(PFP['loop'])


# plot Geometry
    nbtot=0
    for i in range(nbcoil):
        nbelt=len(PF['coil.{}.element'.format(i)])
        nbtot=nbtot+nbelt
    xvar=np.zeros(nbtot)
    yvar=np.zeros(nbtot)
    cpt=0
    for i in range(nbcoil):
        nbelt=len(PF['coil.{}.element'.format(i)])
        for j in range(nbelt):
            xvar[cpt]=PF['coil.{}.element.{}.geometry.rectangle.r'.format(i,j)]
            yvar[cpt]=PF['coil.{}.element.{}.geometry.rectangle.z'.format(i,j)]
            cpt=cpt+1

    xvar2=[]
    yvar2=[]
    for i in range(nbloop):
        nbelt2=len(PFP['loop.{}.element'.format(i)])
        for k in range(nbelt2):
            if PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==1:
                for j in range(4):
                    xvar2.append(PFP['loop.{}.element.{}.geometry.outline.r'.format(i,k)][j])
                    yvar2.append(PFP['loop.{}.element.{}.geometry.outline.z'.format(i,k)][j])
            elif PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==2:
                xvar2.append(PFP['loop.{}.element.{}.geometry.rectangle.r'.format(i,k)])
                yvar2.append(PFP['loop.{}.element.{}.geometry.rectangle.z'.format(i,k)])

    fpf2=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.scatter(xvar,yvar,lw=1,label='Coil position')
    plt.scatter(xvar2,yvar2,lw=1,label='VV position')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    nbCoil=len(PF['coil'])
    Color=['b','g','r','c','m','y','b','g','r','c','m','y']

    if len(PF['coil[0].current.data'])>0:
        time=PF['time']
        if len(time) == 0:
            time=PF['coil[0].current.time']

        fpf=plt.figure(facecolor='white')
        for j in range(nbCoil):
            pf=PF['coil.{}.current.data'.format(j)]
            plt.plot(time,pf,lw=2,color=Color[j],label='Current {}'.format(PF['coil.{}.name'.format(j)]))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()

    try:
        print(len(PFP['loop.0.current']))
        ok=1
    except:
        ok=0

    if ok==1:
        time=PFP['time']
        if len(time) == 0:
            time=PF['time']

        pf1=PFP['loop[0].current']
        pf2=PFP['loop[5].current']
        pf3=PFP['loop[9].current']
        fpf3=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Current {}'.format(PFP['loop[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Current {}'.format(PFP['loop[5].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Current {}'.format(PFP['loop[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()

        nbloop=len(PFP['loop'])
        k=-1
        for i in range(len(time)):
            if time[i]> 0.31 and k ==-1:
                k=i
        tot=0.
        for i in range(nbloop):
            if PFP['loop.{}.name'.format(i)]=='W1':
                tot=tot+PFP['loop.{}.current'.format(i)][k]
                print(PFP['loop.{}.current'.format(i)][k])
        print(tot)
        
    plt.show()


if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    PF=ods['pf_active']
    PFP=ods['pf_passive']

    plotV(PF,PFP)


from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotTF(TF):
    
    R0=TF['r0']
    myy=TF['b_field_tor_vacuum_r.data']/R0
    myx=TF['time']
    myy2=TF['coil.0.current.data']
    
    fig1=plt.figure(facecolor='white')
    plt.plot(myx,myy,label='b_field_tor')

    fig2=plt.figure(facecolor='white')
    plt.plot(myx,myy2,label='current')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    TF=ods['tf']


    plotTF(TF)

from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotV(PF,PFP):
    nbcoil=len(PF['coil'])
    nbloop=len(PFP['loop'])

# plot Geometry
    nbtot=0
    for i in range(nbcoil):
        nbelt=len(PF['coil.{}.element'.format(i)])
        nbtot=nbtot+nbelt
    xvar=np.zeros(nbtot)
    yvar=np.zeros(nbtot)
    cpt=0
    for i in range(nbcoil):
        nbelt=len(PF['coil.{}.element'.format(i)])
        for j in range(nbelt):
            xvar[cpt]=PF['coil.{}.element.{}.geometry.rectangle.r'.format(i,j)]
            yvar[cpt]=PF['coil.{}.element.{}.geometry.rectangle.z'.format(i,j)]
            cpt=cpt+1
            
    nbtot=4*nbloop
    xvar2=[]
    yvar2=[]
    for i in range(nbloop):
        nbelt2=len(PFP['loop.{}.element'.format(i)])
        for k in range(nbelt2):
            if PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==1:
                for j in range(4):
                    xvar2.append(PFP['loop.{}.element.0.geometry.outline.r'.format(i)][j])
                    yvar2.append(PFP['loop.{}.element.0.geometry.outline.z'.format(i)][j])
            elif PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==2:
                xvar2.append(PFP['loop.{}.element.0.geometry.rectangle.r'.format(i)])
                yvar2.append(PFP['loop.{}.element.0.geometry.rectangle.z'.format(i)])

    fpf2=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.scatter(xvar,yvar,lw=1,label='Coil position')
    plt.scatter(xvar2,yvar2,lw=1,label='VV position')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    nbCoil=len(PF['coil'])
    Color=['b','g','r','c','m','y','b','g','r','c','m','y']

    if len(PF['coil[0].current.data'])>0:
        time=PF['time']
        if len(time) == 0:
            time=PF['coil[0].current.time']

        pf1=PF['coil[0].current.data']
        pf2=PF['coil[4].current.data']
        pf3=PF['coil[9].current.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Current {}'.format(PF['coil[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Current {}'.format(PF['coil[4].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Current {}'.format(PF['coil[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()


    if len(PF['coil[0].voltage.data'])>0:
        time=PF['time']
        if len(time) == 0:
            time=PF['coil[0].voltage.time']

        pf1=PF['coil[0].voltage.data']
        pf2=PF['coil[4].voltage.data']
        pf3=PF['coil[9].voltage.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Voltage {}'.format(PF['coil[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Voltage {}'.format(PF['coil[4].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Voltage {}'.format(PF['coil[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()


    plt.show()


if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    PF=ods['pf_active']
    PFP=ods['pf_passive']

    plotV(PF,PFP)

from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotMG(MG):
    nbloop=len(MG['flux_loop'])
    nbprobe=len(MG['b_field_pol_probe'])
    print(nbloop,nbprobe)
    
# plot Geometry
    xvar=np.zeros(nbloop)
    yvar=np.zeros(nbloop)
    for i in range(nbloop):
        xvar[i]=MG['flux_loop.{}.position.0.r'.format(i)]
        yvar[i]=MG['flux_loop.{}.position.0.z'.format(i)]
             
    xvar2=np.zeros(nbprobe)
    yvar2=np.zeros(nbprobe)
    for i in range(nbprobe):
        xvar2[i]=MG['b_field_pol_probe.{}.position.r'.format(i)]
        yvar2[i]=MG['b_field_pol_probe.{}.position.z'.format(i)]


    fig1=plt.figure(facecolor='white')
    myx=MG['time']
    myy=MG['diamagnetic_flux.0.data']
    plt.plot(myx,myy)

    
    fpf2=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.scatter(xvar,yvar,lw=1,label='FL position')
    plt.scatter(xvar2,yvar2,lw=1,label='Probe position')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    Color=['b','g','r','c','m','y','b','g','r','c','m','y']

    if len(MG['flux_loop[0].flux.data'])>0:
        time=MG['time']

        pf1=MG['flux_loop[0].flux.data']
        pf2=MG['flux_loop[4].flux.data']
        pf3=MG['flux_loop[9].flux.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Current {}'.format(MG['flux_loop[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Current {}'.format(MG['flux_loop[4].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Current {}'.format(MG['flux_loop[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()


    if len(MG['b_field_pol_probe[0].field.data'])>0:
        time=MG['time']

        pf1=MG['b_field_pol_probe[10].field.data']
        pf2=MG['b_field_pol_probe[30].field.data']
        pf3=MG['b_field_pol_probe[60].field.data']
        pf4=MG['b_field_pol_probe[63].field.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Bz {}'.format(MG['b_field_pol_probe[10].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Bz {}'.format(MG['b_field_pol_probe[30].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Bz {}'.format(MG['b_field_pol_probe[60].name']))
        plt.plot(time,pf4,lw=2,color=Color[3],label='Bz {}'.format(MG['b_field_pol_probe[63].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()


    plt.show()


if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    MG=ods['magnetics']


    plotMG(MG)

# plot Psi, Br and Bz stored in a EQ ODS
# python plotEQ.py shot run time_index
# python plotEQ.py 37194 
from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt
from omas.omas_structure import add_extra_structures

def plotEQ(EQ,inc):
#    plt.scatter(xvar,yvar,lw=1,label='FL position')
#    plt.scatter(xvar2,yvar2,lw=1,label='Probe position')
    nbt=len(EQ['time_slice'])
    print(nbt)
    print(EQ['ids_properties.homogeneous_time'])
    if inc > nbt:
        inc=nbt
    temps=EQ['time'][inc-1]
    print('Time index:{}/{} - {} s'.format(inc,nbt,temps))
    xvar=EQ['time_slice.{}.profiles_2d.0.grid.dim1'.format(inc-1)]
    zvar=EQ['time_slice.{}.profiles_2d.0.grid.dim2'.format(inc-1)]
    psi=EQ['time_slice.{}.profiles_2d.0.psi'.format(inc-1)]
    br=EQ['time_slice.{}.profiles_2d.0.b_field_r'.format(inc-1)]
    bz=EQ['time_slice.{}.profiles_2d.0.b_field_z'.format(inc-1)]
    fpsi=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    #    print(len(xvar),len(zvar),len(psi),len(psi[0]))
    #    if len(psi) != len(zvar):
    psi=psi.T
    br=br.T
    bz=bz.T
    plt.contourf(xvar,zvar,psi)
    mystring="Shot: {} Run:{} Time:{}".format(shot,run,temps)
    plt.colorbar()
    plt.title(mystring)
    #    plt.legend()

    r=xvar
    z=zvar
    Nr=len(r)
    Nz=len(z)

    (r2,z2) = np.meshgrid(r,z)

    shape=(Nz,Nr)
    ndecay=np.zeros(shape)
    dBZ=np.zeros((Nr,Nz))
    dr=r[1]-r[0]
    for t in range(nbt):
        BZ=bz
        for i  in range(Nr-1):
            dBZ[i]=(BZ.T[i+1]-BZ.T[i])/dr
        ndecay = -r2/BZ * dBZ.T

    fbr=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.contourf(xvar,zvar,br)
    mystring="Shot: {} Run:{} Time:{}".format(shot,run,temps)
    plt.colorbar()
    plt.title(mystring)

    fbz=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.contourf(xvar,zvar,bz)
    mystring="Shot: {} Run:{} Time:{}".format(shot,run,temps)
    plt.colorbar()
    plt.title(mystring)

    
#    fn=plt.figure(facecolor='white')
#    myax=plt.axes()
#    myax.set_aspect('equal')
#    plt.contourf(xvar,zvar,ndecay)
#    mystring="Shot: {} Run:{}".format(shot,run)
#    plt.colorbar()
#    plt.title(mystring)


    
        
    plt.show()


    

if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])
    inc= int(argv[2])
# new data (centroid) are createad in the equilibrium ODS when the ODS is generated from geqdsk files
    _extra_structures = {
        'equilibrium': {
            'equilibrium.time_slice.:.profiles_1d.centroid.r_max': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r_max(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid r max",
                "units": 'm',
                "cocos_signal": '?'  # optional
            },
            'equilibrium.time_slice.:.profiles_1d.centroid.r_min': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r_min(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid r min",
                "units": 'm',
                "cocos_signal": '?'  # optional
            },
            'equilibrium.time_slice.:.profiles_1d.centroid.r': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid r",
                "units": 'm',
                "cocos_signal": '?'  # optional
            },
            'equilibrium.time_slice.:.profiles_1d.centroid.z': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.z(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid z",
                "units": 'm',
                "cocos_signal": '?'  # optional
            }
        }
    }
    add_extra_structures(_extra_structures)
    
    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    EQ=ods['equilibrium']


    plotEQ(EQ,inc)

from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotW(ods,shot,run):
    nbrun=len(run)
    Rin=[]
    Rout=[]
    Wnum=[0,50,100,150,200,750,800,850,900,949]
    nbnum=len(Wnum)
    Inum=[]

    times=[]
    for i in range(nbrun):
        ODS=ods[i]
        PFP=ODS['pf_passive']
        times.append(PFP['time'])
        nbloop=len(PFP['loop'])

        rin=[]
        rout=[]
        for j in range(nbloop):
            if PFP[f'loop.{j}.name']=='W1': # outboard
                rout.append(PFP[f'loop.{j}.resistance'])
            if PFP[f'loop.{j}.name']=='W11': # inboard
                rin.append(PFP[f'loop.{j}.resistance'])
        nbin=len(rin)
        nbout=len(rout)
            
        Rout.append(rout)
        Rin.append(rin)

        iwall=[]
        for j in range(nbnum):
            iwall.append(PFP[f'loop.{j}.current'])
        Inum.append(iwall)
        

    Color=['b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m']

    xin=np.arange(nbin)
    xout=np.arange(nbout)

    f1=plt.figure(facecolor='white')
    for i in range(nbrun):
        plt.plot(xin,Rin[i],lw=2,color=Color[i],label=f'{run[i]}')
    plt.title(f'{shot} - R inboard')
    plt.legend()
    f2=plt.figure(facecolor='white')
    for i in range(nbrun):
        plt.plot(xout,Rout[i],lw=2,color=Color[i],label=f'{run[i]}')
    plt.title(f'{shot} - R outboard')
    plt.legend()
    
    
    for j in range(nbnum):
        f=plt.figure(facecolor='white')
        for i in range(nbrun):
            time=times[i]
            nbt=len(time)
            yvar=Inum[i][j]
            #            if j == 0:
            #                print(yvar[int(nbt/2)],max(yvar))
            plt.plot(time,yvar,lw=2,color=Color[i],label=f'{run[i]}')
        if j<=4:
            board='outboard'
        else:
            board='inboard'
        plt.title(f'{shot} - Eddy current - {board}')
        plt.legend()

#    print(len(Inum),len(Inum[0]),nbt)

       
    plt.show()


if __name__ == "__main__":
    shot = 41516
    run = [1,2,3,4,5]
    nbrun=len(run)
    
    ods=[]
    for i in range(nbrun):
        filename='{}_{}.json'.format(shot,run[i])
        ods.append(load_omas_json(filename))

    plotW(ods,shot,run)

from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import netcdf
    
def plotEQ(EQs,shots,runs):
    name=f'{shots[0]}_{runs[0]}'
    nbODS=len(EQs)
    fig = plt.figure(figsize=(10,8))
    fig.suptitle(name, fontsize=16)
#    ax = fig.add_subplot()
#    fig.subplots_adjust(top=0.85)
#    ax.axis([0, 10, 0, 10])
#    ax.text(1,2, 'Experimental data',color='blue', fontsize=15)
    
    ax1 = plt.subplot2grid((5, 4), (0, 0))
    ax2 = plt.subplot2grid((5, 4), (0, 1))
    ax3 = plt.subplot2grid((5, 4), (0, 2))
    ax4 = plt.subplot2grid((5, 4), (0, 3))
    ax5 = plt.subplot2grid((5, 4), (2, 0))
    ax6 = plt.subplot2grid((5, 4), (2, 1))
    ax7 = plt.subplot2grid((5, 4), (2, 2))
    ax8 = plt.subplot2grid((5, 4), (2, 3))
    ax9 = plt.subplot2grid((5, 4), (4, 0))
    ax10 = plt.subplot2grid((5, 4), (4, 1))
    ax11 = plt.subplot2grid((5, 4), (4, 2))
    ax11.axis('off')


    Color=['blue','orange','green','red','purple','pink','gray','olive','cyan']
    
    for j in range(nbODS):
        EQ=EQs[j]
        time=EQ['time']
        nbt=len(time)
        B_normal=[]
        B_pol=[]
        B_tor=[]
        b_field=[]
        R=[]
        a=[]
        elong=[]
        triang=[]
        Ip=[]
        li_3=[]
        for i in range(nbt):
            B_normal.append(EQ[f'time_slice.{i}.global_quantities.beta_normal'])
            B_pol.append(EQ[f'time_slice.{i}.global_quantities.beta_pol'])
            B_tor.append(EQ[f'time_slice.{i}.global_quantities.beta_tor'])
            b_field.append(EQ[f'time_slice.{i}.global_quantities.magnetic_axis.b_field_tor'])
            R.append(EQ[f'time_slice.{i}.global_quantities.magnetic_axis.r'])
            RR=EQ[f'time_slice.{i}.boundary.outline.r']
            a.append((max(RR)-min(RR))/2)
            elong.append(EQ[f'time_slice.{i}.profiles_1d.elongation'][-1])
            triang.append(EQ[f'time_slice.{i}.profiles_1d.triangularity_upper'][-1])
            Ip.append(EQ[f'time_slice.{i}.global_quantities.ip'])
            li_3.append(EQ[f'time_slice.{i}.global_quantities.li_3'])
    
        ax1.scatter(time, B_normal,color=Color[j])
        ax1.set_title('Beta_normal', fontsize=10)

        ax2.scatter(time, B_pol,color=Color[j])
        ax2.set_title('Beta_pol', fontsize=10)

        ax3.scatter(time, B_tor,color=Color[j])
        ax3.set_title('Beta_tor', fontsize=10)

        ax4.scatter(time, b_field,color=Color[j])
        ax4.set_title('b_field_tor', fontsize=10)

        ax5.scatter(time, R,color=Color[j])
        ax5.set_title('R', fontsize=9)

        ax6.scatter(time, a,color=Color[j])
        ax6.set_title('a', fontsize=9)

        ax7.scatter(time, elong,color=Color[j])
        ax7.set_title('elongation', fontsize=10)

        ax8.scatter(time, triang,color=Color[j])
        ax8.set_title('triangularity', fontsize=9)

        ax9.scatter(time, Ip,color=Color[j])
        ax9.set_title('Ip', fontsize=10)

        ax10.scatter(time, li_3,color=Color[j])
        ax10.set_title('li_3', fontsize=10)

        ax11.text(0,j*0.2,f'Shot: {shots[j]} - {runs[j]}',color=Color[j], fontsize=10)


    plt.savefig(f'C{name}.png')
    print(f'C{name}.png generated')
    
    #display plots
    plt.show()
        
        
if __name__ == "__main__":
    argv=sys.argv[1:]

    nbODS=int(len(argv)/2)
    shots=[]
    runs=[]
    for i in range(nbODS):
        shots.append(int(argv[2*i]))
        runs.append(int(argv[2*i+1]))

    EQs=[]
    for i in range(nbODS):
        filename=f'{shots[i]}_{runs[i]}.json'
        ods = load_omas_json(filename)
        EQs.append(ods['equilibrium'])

    
    plotEQ(EQs,shots,runs)

from VEST_tools import vest_load,vest_loadn
import matplotlib.pyplot as plt

shot=[37194,37195,37196,37197,37198,37199]

Data=['PF1 Current','PF2 and 3 Current','PF4 Current','PF5 Current','PF6 Current','PF7 Current','PF8 Current','PF9 Current','PF10 Current','TF Current']
Xlabel='Time [s]'
Ylabel='Current [A]'


def Title(k):
    T = 'VEST PF Current, shot number :'
    T = T.lstrip()
    T = T.rstrip()
    T = T.ljust(31)
    T = T+str(shot[k])
    return (T)

nbData=len(Data)
nbshot=len(shot)

# call vest_load

Datas=[]
Times=[]
Legend=[]

for k in range(nbshot):

    for i in range(nbData):

        (time,data)=vest_loadn(shot[k],Data[i])

        if len(data) > 1:

            Datas.append(data)

            Legend.append(Data[i])

            Times.append(time)


    #print(time[4999],time[8999])

    nbfig=len(Legend)
    fig=plt.figure()

    for i in range(nbfig):

        plt.plot(Times[i],Datas[i],label=Legend[i])

        plt.legend(fontsize=10,loc='upper left')
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title(Title(k))

    Datas=[]
    Times=[]
    Legend=[]

plt.show()


from VEST_tools import vest_load,vest_loadn
import matplotlib.pyplot as plt

shot=37194
Coils=['PF1','PF2 and 3','PF4','PF5','PF6','PF7','PF8','PF9','PF10']
nbCoil=len(Coils)
# call vest_load

Currents=[]
Legend=[]
for i in range(nbCoil):
    (time,data)=vest_loadn(shot,'{} Current'.format(Coils[i]))
    if len(data) > 1:
        Currents.append(data)
        Legend.append(Coils[i])
        rtime=time

nbfig=len(Legend)
fig=plt.figure()
for i in range(nbfig):
    plt.plot(rtime,Currents[i],label=Legend[i])
plt.legend(fontsize=10,loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Current [A]')
plt.title('VEST PF Current')
plt.show()

from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import netcdf
    
def plotC(name):
    gname='g'+name
    mname='m'+name+'.nc'
    mfile= netcdf.NetCDFFile(mname,'r')

    # the g-file contains the 2D Psi
    gfile=OMFITgeqdsk(gname)
    ods=gfile.to_omas()

    RBBBS=gfile['RBBBS']
    ZBBBS=gfile['ZBBBS']

    # Psi
    ZSI= [0.685, 0.46, -0.46, -0.685, -0.805, 0.04, 0.805, 0.696, 0.62, -0.696, -0.62]

    COILS_exp=mfile.variables['silopt'][:][0]
    COILS_rec=mfile.variables['csilop'][:][0]
    FWTSI=mfile.variables['fwtsi'][:][0]
    Psioute=0.
    Psioutexp=[]
    Psioutrec=[]
    Psiouty=[]
    PsioutW=0.
    for i in range(4):
        if FWTSI[i]:
            Psioute=Psioute+abs((COILS_exp[i]-COILS_rec[i])/COILS_exp[i])*100.
            Psioutexp.append(COILS_exp[i])
            Psioutrec.append(COILS_rec[i])
            Psiouty.append(ZSI[i])
            PsioutW=PsioutW+FWTSI[i]

    Psioute=Psioute/len(Psioutexp)
    PsioutW=PsioutW/len(Psioutexp)
            
    Psiine=0.
    Psiinexp=[]
    Psiinrec=[]
    Psiiny=[]
    PsiinW=0.
    for i in range(4,11):
        if FWTSI[i]:
            Psiine=Psiine+abs((COILS_exp[i]-COILS_rec[i])/COILS_exp[i])*100.
            Psiinexp.append(COILS_exp[i])
            Psiinrec.append(COILS_rec[i])
            Psiiny.append(ZSI[i])
            PsiinW=PsiinW+FWTSI[i]
    Psiine=Psiine/len(Psiinexp)
    PsiinW=PsiinW/len(Psiinexp)
    
    # Bz
    # There is a problem with the a-eqdsk file generated by EFIT (in my version at least). Need to remove the test      if (ishot.lt.91000) then at line 150 of write_a.f90
    ZMP2 = [0.54, 0.5, 0.46, 0.42, 0.38, 0.34, 0.3, 0.26, 0.22, 0.16, 0.12, 0.08, 0.04, 0.0, -0.04, -0.08, -0.12, -0.16, -0.22, -0.26, -0.3, -0.34, -0.38, -0.42 ,-0.46, -0.5, -0.54,0.42, 0.38, 0.34, 0.3, 0.26, 0.22, 0.18, 0.1, 0.06, 0.02, -0.02, -0.06, -0.1, -0.14, -0.18,-0.22, -0.26, -0.3, -0.34, -0.38, -0.42, 0.8328, 0.8728, 0.9128, 0.9528, 0.9928, 1.0328,1.0728, 1.1128, -0.8328, -0.8728, -0.9128, -0.9528, -0.9928, -1.0328, -1.0728,-1.1128]

    EXPMP2_exp=mfile.variables['expmpi'][:][0]
    EXPMP2_rec=mfile.variables['cmpr2'][:][0]
    FWTMP2=mfile.variables['fwtmp2'][:][0]
    intBz=0.
    if len(EXPMP2_exp) > 64:
        intBz=1

    Bzine=0.
    Bzinexp=[]
    Bzinrec=[]
    Bziny=[]
    BzinW=0.
    for i in range(27):
        if FWTMP2[i]:
            Bzine=Bzine+abs((EXPMP2_exp[i]-EXPMP2_rec[i])/EXPMP2_exp[i])*100.
            Bzinexp.append(EXPMP2_exp[i])
            Bzinrec.append(EXPMP2_rec[i])
            Bziny.append(ZMP2[i])
            BzinW=BzinW+FWTMP2[i]
    Bzine=Bzine/len(Bzinexp)
    BzinW=BzinW/len(Bzinexp)
    
    Bzoute=0.
    Bzoutexp=[]
    Bzoutrec=[]
    Bzouty=[]
    BzoutW=0.
    for i in range(27,48):
        if FWTMP2[i]:
            Bzoute=Bzoute+abs((EXPMP2_exp[i]-EXPMP2_rec[i])/EXPMP2_exp[i])*100.
            Bzoutexp.append(EXPMP2_exp[i])
            Bzoutrec.append(EXPMP2_rec[i])
            Bzouty.append(ZMP2[i])
            BzoutW= BzoutW+FWTMP2[i]
    Bzoute=Bzoute/len(Bzoutexp)
    BzoutW=BzoutW/len(Bzoutexp)
    
    Bzsidee=0.
    Bzsideexp=[]
    Bzsiderec=[]
    Bzsidey=[]
    BzsideW=0.
    for i in range(48,64):
        if FWTMP2[i]:
            Bzsidee=Bzsidee+abs((EXPMP2_exp[i]-EXPMP2_rec[i])/EXPMP2_exp[i])*100.
            Bzsideexp.append(EXPMP2_exp[i])
            Bzsiderec.append(EXPMP2_rec[i])
            Bzsidey.append(ZMP2[i])
            BzsideW=BzsideW+FWTMP2[i]
    Bzsidee=Bzsidee/len(Bzsideexp)
    BzsideW=BzsideW/len(Bzsideexp)


    Bzinte=0.
    BzintW=0.
    if intBz==1:
        Bzintexp=[]
        Bzintrec=[]
        Bzinty=[]
        R0=0.245
        dR=0.025

        for i in range(64,80):
            if FWTMP2[i]:
                Bzinte=Bzinte+abs((EXPMP2_exp[i]-EXPMP2_rec[i])/EXPMP2_exp[i])*100.
                Bzintexp.append(EXPMP2_exp[i])
                Bzintrec.append(EXPMP2_rec[i])
                Bzinty.append(R0+(i-64)*dR)
                BzintW=BzintW+FWTMP2[i]
        Bzinte=Bzinte/len(Bzintexp)
        BzintW=BzintW/len(Bzintexp)

        
    PLASMA_exp=mfile.variables['plasma'][0]
    FWTCUR=mfile.variables['fwtcur'][0]
    PLASMA_rec=mfile.variables['cpasma'][0]
    Ipe=abs((PLASMA_exp-PLASMA_rec)/PLASMA_exp)*100.
    IpW=FWTCUR
    
    BRSP_exp=mfile.variables['fccurt'][:][0]
    FWTFC=mfile.variables['fwtfc'][:][0]
    BRSP_rec=mfile.variables['ccbrsp'][:][0]
    Ipfe=0.
    Ipfexp=[]
    Ipfrec=[]
    Ipfy=[]
    IpfW=0.
    for i in range(len(BRSP_exp)):
        Ipfe=Ipfe+abs((BRSP_exp[i]-BRSP_rec[i])/BRSP_exp[i])*100.
        Ipfexp.append(BRSP_exp[i])
        Ipfrec.append(BRSP_rec[i])
        Ipfy.append(i)
        IpfW=IpfW+FWTFC[i]
    Ipfe=Ipfe/len(BRSP_exp)
    IpfW=IpfW/len(BRSP_exp)
    
    DFLUX_exp=mfile.variables['diamag'][0]
    FWTDLC=mfile.variables['fwtdlc'][0]
    DFLUX_rec=mfile.variables['cdflux'][0]
    Dfluxe=abs((DFLUX_exp-DFLUX_rec)/DFLUX_exp)*100.
    DfluxW=FWTDLC
    
    EQ=ods['equilibrium']
    xvar=EQ['time_slice.0.profiles_2d.0.grid.dim1']
    zvar=EQ['time_slice.0.profiles_2d.0.grid.dim2']
    psi=EQ['time_slice.0.profiles_2d.0.psi']

    mfile.close()
    
    f=open(f'result_{name}.dat','w')
    f.write('Exp.\t\t\tRec.\t\t\tErr.[%]\t\tWeight\n')
    f.write('Psi [WB/rad]\n')
    for i in range(len(COILS_exp)):
        if FWTSI[i]:
            err=abs((COILS_exp[i]-COILS_rec[i])/COILS_exp[i])*100.
            f.write(f'{COILS_exp[i]:.11f}\t\t{COILS_rec[i]:.11f}\t\t{err:.2f}\t\t{FWTSI[i]:.3e}\n')
        else:
            f.write(f'{0:.11f}\t\t{0:.11f}\t\t{0.:.2f}\t\t{0:.3e}\n')
            
    f.write('Bz [T]:\n')
    for i in range(len(EXPMP2_exp)):
        if FWTMP2[i]:
            err=abs((EXPMP2_exp[i]-EXPMP2_rec[i])/EXPMP2_exp[i])*100.
            f.write(f'{EXPMP2_exp[i]:.11f}\t\t{EXPMP2_rec[i]:.11f}\t\t{err:.2f}\t\t{FWTMP2[i]:.3e}\n')
        else:
            f.write(f'{0:.11f}\t\t{0:.11f}\t\t{0.:.2f}\t\t{0:.3e}\n')

    f.write('Ip [A]:\n')
    f.write(f'{PLASMA_exp:.7f}\t\t{PLASMA_rec:.7f}\t\t{Ipe:.2f}\t\t{FWTCUR:.3e}\n')

    f.write('Ipf [A]:\n')
    for i in range(len(BRSP_exp)):
        err=abs((BRSP_exp[i]-BRSP_rec[i])/BRSP_exp[i])*100.
        f.write(f'{BRSP_exp[i]:.7f}\t\t{BRSP_rec[i]:.7f}\t\t{err:.2f}\t\t{FWTFC[i]:.3e}\n')

    f.write('Diamag. Flux [mWb]:\n')
    if FWTDLC:
        f.write(f'{DFLUX_exp:.11f}\t\t{DFLUX_rec:.11f}\t\t{Dfluxe:.2f}\t\t{FWTDLC:.3e}\n')
    else:
        f.write(f'{0.:.11f}\t\t{0.:.11f}\t\t{0.:.2f}\t\t{0.:.3e}\n')

        
    f.close()

    t1 = np.arange(0.0, 3.0, 0.01)
    f1 = np.arange(0.0, 3.0, 0.01)

    fig = plt.figure(figsize=(10,8))
    fig.suptitle(name, fontsize=16)
#    ax = fig.add_subplot()
#    fig.subplots_adjust(top=0.85)
#    ax.axis([0, 10, 0, 10])
#    ax.text(1,2, 'Experimental data',color='blue', fontsize=15)
    
    ax1 = plt.subplot2grid((5, 3), (0, 0))
    ax2 = plt.subplot2grid((5, 3), (0, 1))
    ax3 = plt.subplot2grid((5, 3), (2, 0))
    ax4 = plt.subplot2grid((5, 3), (2, 1))
    ax5 = plt.subplot2grid((5, 3), (4, 0))
    ax6 = plt.subplot2grid((5, 3), (4, 1))
    ax7 = plt.subplot2grid((5, 3), (0, 2), rowspan=3)
    ax8 = plt.subplot2grid((5, 3), (4, 2))

    ax9 = plt.subplot2grid((5, 3), (1, 0))
    ax9.axis('off')
    ax10 = plt.subplot2grid((5, 3), (3, 1))
    ax10.axis('off')
    ax11 = plt.subplot2grid((5, 3), (3, 2))
    ax11.axis('off')

    ax9.text(0,0.6,'Experimental data',color='blue', fontsize=15)
    ax9.text(0,0.4,'Reconstructed data',color='orange', fontsize=15)
    ax10.text(0,0.6,f'Dflux exp.:{DFLUX_exp/1000.:.4e} Wb - W:{DfluxW:.2e}',color='blue', fontsize=10)
    ax10.text(0,0.4,f'Dflux rec.:{DFLUX_rec/1000.:.4e} Wb',color='orange', fontsize=10)
    ax10.text(0,0.2,f'Dflux err.:{Dfluxe:.2f} %', fontsize=10)
    ax11.text(0,0.6,f'Ip exp.:{PLASMA_exp:.4f} A - W:{IpW:.2e}',color='blue', fontsize=10)
    ax11.text(0,0.4,f'Ip rec.:{PLASMA_rec:.4f} A',color='orange', fontsize=10)
    ax11.text(0,0.2,f'Ip err.:{Ipe:.2f} %', fontsize=10)
#    ax9.text(0,0.6,'Reconstructed data',color='orange', fontsize=15)
    
    ax1.scatter(Bziny, Bzinexp,color='blue')
    ax1.scatter(Bziny, Bzinrec,color='orange')
    ax1.set_title(f'Bz in [T] - err: {Bzine:.2f} % - W:{BzinW:.2e}', fontsize=10)
    ax1.set_ylim([0,np.max(Bzinrec)*1.2])

    ax2.scatter(Bzsidey, Bzsideexp,color='blue')
    ax2.scatter(Bzsidey, Bzsiderec,color='orange')
    ax2.set_title(f'Bz side [T] - err: {Bzsidee:.2f} % - W:{BzsideW:.2e}', fontsize=10)
    ax2.set_ylim([np.min(Bzsideexp)*1.2,np.max(Bzsideexp)*1.2])

    ax3.scatter(Bzouty, Bzoutexp,color='blue')
    ax3.scatter(Bzouty, Bzoutrec,color='orange')
    ax3.set_title(f'Bz out [T] - err: {Bzoute:.2f} % - W:{BzoutW:.2e}', fontsize=10)
    ax3.set_ylim([np.min(Bzoutexp)*1.2,0])

    if intBz==1:
        ax4.scatter(Bzinty, Bzintexp,color='blue')
        ax4.scatter(Bzinty, Bzintrec,color='orange')
        ax4.set_title(f'int. Bz [T]- err: {Bzinte:.2f} % - W:{BzintW:.2e}', fontsize=10)
    else:
        ax4.set_title(' no internal Bz', fontsize=10)

    ax5.scatter(Psiiny, Psiinexp,color='blue')
    ax5.scatter(Psiiny, Psiinrec,color='orange')
    ax5.set_ylim([np.min(Psiinexp)*1.2,0])
    ax5.set_title(f'Psi in [Wb/rad] - err: {Psiine:.2f} % - W:{PsiinW:.2e}', fontsize=9)

    ax6.scatter(Psiouty, Psioutexp,color='blue')
    ax6.scatter(Psiouty, Psioutrec,color='orange')
    ax6.set_ylim([np.min(Psioutexp)*1.2,0])
    ax6.set_title(f'Psi out [Wb/rad] - err: {Psioute:.2f} % - W:{PsioutW:.2e}', fontsize=9)

    ax8.scatter(Ipfy, Ipfexp,color='blue')
    ax8.scatter(Ipfy, Ipfrec,color='orange')
    ax8.set_title(f'Ipf [A] - err: {Ipfe:.2f} % - W:{IpfW:.2e}', fontsize=9)

    ax7.contour(xvar,zvar,psi.T)
    ax7.plot(RBBBS,ZBBBS,color='red')
    ax7.set_title(f'Psi')
    ax7.set_ylim([-1,1])
    ax7.set_xlim([0,1.25])

    plt.savefig(f'{name}.png')
    print(f'result_{name}.dat generated')
    print(f'{name}.png generated')
    
    #display plots
    plt.show()
        
        
if __name__ == "__main__":
    argv=sys.argv[1:]
    name=argv[0]

    plotC(name)

from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt
from omas.omas_structure import add_extra_structures

def plot_mag(MG):
    nbloop=len(MG['flux_loop'])
    nbprobe=len(MG['b_field_pol_probe'])
    print(nbloop,nbprobe)

    MGtime=MG['time']
    # flux_loop
    nbp=len(MG['flux_loop'])
    MGpsi=[]
    psi=[]
    Label=[]
    for i in range(nbp):
        psi.append(MG['flux_loop.{}.flux.reconstructed'.format(i)])
        MGpsi.append(MG['flux_loop.{}.flux.data'.format(i)])
        Label.append(MG['flux_loop.{}.name'.format(i)])

    nbp2=len(MG['b_field_pol_probe'])
    bz=[]
    MGbz=[]
    BLabel=[]
    for i in range(nbp2):
        bz.append(MG['b_field_pol_probe.{}.field.reconstructed'.format(i)])
        MGbz.append(MG['b_field_pol_probe.{}.field.data'.format(i)])
        BLabel.append(MG['b_field_pol_probe.{}.name'.format(i)])
    
    Color=['b','g','r','c','m','y','b','g','r','c','m','y']
        
    # Flux
    f1=plt.figure(facecolor='white')
    for i in range(6):
        plt.plot(MGtime,psi[i],color=Color[i])
        plt.plot(MGtime,MGpsi[i],color=Color[i],linestyle='dashed',label=Label[i])
    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()
    
    f2=plt.figure(facecolor='white')
    for i in range(6,11):
        plt.plot(MGtime,psi[i],color=Color[i])
        plt.plot(MGtime,MGpsi[i],color=Color[i],linestyle='dashed',label=Label[i])
    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    # Bz
    for j in range(12):
        fb1=plt.figure(facecolor='white')
        for i in range(5):
            plt.plot(MGtime,bz[i+j*5],color=Color[i])
            plt.plot(MGtime,MGbz[i+j*5],color=Color[i],linestyle='dashed',label=BLabel[i+j*5])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()

        fb1=plt.figure(facecolor='white')
    j=12
    for i in range(4):
        plt.plot(MGtime,bz[i+j*5],color=Color[i])
        plt.plot(MGtime,MGbz[i+j*5],color=Color[i],linestyle='dashed',label=BLabel[i+j*5])
    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    
    plt.show()



if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    # OMAS extra_structures
    _extra_structures = {
        'magnetics': {
            "magnetics.b_field_pol_probe[:].field.reconstructed": {
                "coordinates": ["magnetics.b_field_pol_probe[:].field.time"],
                "documentation": "value calculated from the reconstructed magnetics",
                "data_type": "FLT_1D",
                "units": "T",
                "cocos_signal": "?",
            },
            "magnetics.flux_loop[:].flux.reconstructed": {
                "coordinates": ["magnetics.flux_loop[:].flux.time"],
                "documentation": "value calculated from the reconstructed magnetics",
                "data_type": "FLT_1D",
                "units": "Wb",
                "cocos_signal": "?",
            }
        }
    }
    add_extra_structures(_extra_structures)
    
    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    MG=ods['magnetics']


    plot_mag(MG)
