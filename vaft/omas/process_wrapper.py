from typing import List, Tuple, Dict, Any
from numpy import ndarray
import numpy as np
from omas import *
<<<<<<< Updated upstream
from vaft.process import compute_br_bz_phi, compute_response_matrix, compute_response_vector, compute_impedance_matrices, solve_eddy_currents, compute_vacuum_fields_1d

def calc_grid_ods(ods: Dict[str, Any], xvar: List[float], zvar: List[float]) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Wrapper function for calc_grid to work with OMAS data structure.

    :param ods: OMAS data structure containing PF coil and loop data.
    :param xvar: List of x (radial) coordinates.
    :param zvar: List of z (vertical) coordinates.
    :return: Tuple of (Br, Bz, Phi) matrices.
    """
    pf = ods['pf_active']
    pfp = ods['pf_passive']

    coil_turns = [
        [pf[f'coil.{i}.element.{j}.turns_with_sign'] for j in range(len(pf[f'coil.{i}.element']))]
        for i in range(len(pf['coil']))
    ]
    coil_r = [
        [pf[f'coil.{i}.element.{j}.geometry.rectangle.r'] for j in range(len(pf[f'coil.{i}.element']))]
        for i in range(len(pf['coil']))
    ]
    coil_z = [
        [pf[f'coil.{i}.element.{j}.geometry.rectangle.z'] for j in range(len(pf[f'coil.{i}.element']))]
        for i in range(len(pf['coil']))
    ]

    loop_geometry_type = [
        pfp[f'loop.{i}.element[0].geometry.geometry_type'] for i in range(len(pfp['loop']))
    ]
    loop_outline_r = [
        pfp[f'loop.{i}.element[0].geometry.outline.r'] if loop_geometry_type[i] == 1 else []
        for i in range(len(pfp['loop']))
    ]
    loop_outline_z = [
        pfp[f'loop.{i}.element[0].geometry.outline.z'] if loop_geometry_type[i] == 1 else []
        for i in range(len(pfp['loop']))
    ]
    loop_rectangle_r = [
        pfp[f'loop.{i}.element[0].geometry.rectangle.r'] if loop_geometry_type[i] == 2 else 0.0
        for i in range(len(pfp['loop']))
    ]
    loop_rectangle_z = [
        pfp[f'loop.{i}.element[0].geometry.rectangle.z'] if loop_geometry_type[i] == 2 else 0.0
        for i in range(len(pfp['loop']))
    ]

    return calc_grid(
        xvar, zvar, coil_turns, coil_r, coil_z,
        loop_geometry_type, loop_outline_r, loop_outline_z,
        loop_rectangle_r, loop_rectangle_z
    )
=======
from vaft.process import compute_br_bz_phi, compute_response_matrix, compute_impedance_matrices, solve_eddy_currents, compute_vacuum_fields_1d
import logging
import vaft.process
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for geometry types
DT_SUB = 1e-6  # Time step for eddy current calculation
GEOMETRY_TYPE_POLYGON = 1
GEOMETRY_TYPE_RECTANGLE = 2

def compute_grid_ods(ods: Dict[str, Any], xvar: List[float], zvar: List[float]) -> Tuple[ndarray, ndarray, ndarray]:
    """Compute magnetic field components (Br, Bz, Phi) on a grid using OMAS data structure.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        xvar: Radial coordinates
        zvar: Vertical coordinates

    Returns:
        Tuple of (Br, Bz, Phi) matrices

    Raises:
        KeyError: If required ODS data is missing
    """
    try:
        pf = ods['pf_active']
        pfp = ods['pf_passive']

        # Extract coil data
        coil_turns = np.array([
            [pf[f'coil.{i}.element.{j}.turns_with_sign'] for j in range(len(pf[f'coil.{i}.element']))]
            for i in range(len(pf['coil']))
        ])
        coil_r = np.array([
            [pf[f'coil.{i}.element.{j}.geometry.rectangle.r'] for j in range(len(pf[f'coil.{i}.element']))]
            for i in range(len(pf['coil']))
        ])
        coil_z = np.array([
            [pf[f'coil.{i}.element.{j}.geometry.rectangle.z'] for j in range(len(pf[f'coil.{i}.element']))]
            for i in range(len(pf['coil']))
        ])

        # Extract loop data
        loop_geometry_type = np.array([
            pfp[f'loop.{i}.element[0].geometry.geometry_type'] for i in range(len(pfp['loop']))
        ])
        loop_outline_r = np.array([
            pfp[f'loop.{i}.element[0].geometry.outline.r'] if loop_geometry_type[i] == GEOMETRY_TYPE_POLYGON else []
            for i in range(len(pfp['loop']))
        ])
        loop_outline_z = np.array([
            pfp[f'loop.{i}.element[0].geometry.outline.z'] if loop_geometry_type[i] == GEOMETRY_TYPE_POLYGON else []
            for i in range(len(pfp['loop']))
        ])
        loop_rectangle_r = np.array([
            pfp[f'loop.{i}.element[0].geometry.rectangle.r'] if loop_geometry_type[i] == GEOMETRY_TYPE_RECTANGLE else 0.0
            for i in range(len(pfp['loop']))
        ])
        loop_rectangle_z = np.array([
            pfp[f'loop.{i}.element[0].geometry.rectangle.z'] if loop_geometry_type[i] == GEOMETRY_TYPE_RECTANGLE else 0.0
            for i in range(len(pfp['loop']))
        ])

        return calc_grid(
            xvar, zvar, coil_turns, coil_r, coil_z,
            loop_geometry_type, loop_outline_r, loop_outline_z,
            loop_rectangle_r, loop_rectangle_z
        )
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise
>>>>>>> Stashed changes

def compute_point_response_ods(
    ods: ODS,
    rz: List[List[float]],
    plasma: List[List[float]] = None
) -> Tuple[ndarray, ndarray, ndarray]:
    """Compute magnetic response (Psi, Bz, Br) at observation points.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        rz: List of [r, z] observation points
        e.g) rz = [(4,0)]
        plasma: Optional list of [r, z] plasma element points
        e.g) plasma = [[0.4, 0.0]]
    
    Returns:
        Tuple of (Psi, Bz, Br) arrays with shape (len(rz), nb_coil+nb_loop+nb_plasma)
<<<<<<< Updated upstream
    """
    # Extract coil data from ODS
    coil_data = []
    for ii in range(len(ods['pf_active']['coil'])):
        elements = []
        for jj in range(len(ods['pf_active'][f'coil.{ii}.element'])):
            elements.append({
                'turns': ods['pf_active'][f'coil.{ii}.element.{jj}.turns_with_sign'],
                'r': ods['pf_active'][f'coil.{ii}.element.{jj}.geometry.rectangle.r'],
                'z': ods['pf_active'][f'coil.{ii}.element.{jj}.geometry.rectangle.z']
            })
        coil_data.append({'elements': elements})

    # Extract passive loop data from ODS
    passive_loop_data = []
    for ii in range(len(ods['pf_passive']['loop'])):
        loop = ods['pf_passive'][f'loop.{ii}.element[0].geometry']
        loop_data = {'geometry_type': loop['geometry_type']}
        
        if loop_data['geometry_type'] == 1:
            loop_data.update({
                'outline_r': loop['outline.r'],
                'outline_z': loop['outline.z']
            })
        else:
            loop_data.update({
                'rectangle_r': loop['rectangle.r'],
                'rectangle_z': loop['rectangle.z']
            })
        passive_loop_data.append(loop_data)

    return cal_response_vector(
        coil_data=coil_data,
        passive_loop_data=passive_loop_data,
        plasma_points=plasma,
        observation_points=rz
    )
=======

    Raises:
        KeyError: If required ODS data is missing
    """
    # Process observation points
    if isinstance(rz, tuple) and len(rz) == 2 and all(isinstance(x, (float, int)) for x in rz):
        rz = [rz]

    try:
        # Extract coil data
        coil_data = []
        for ii in range(len(ods['pf_active']['coil'])):
            elements = []
            for jj in range(len(ods['pf_active'][f'coil.{ii}.element'])):
                elements.append({
                    'turns': ods['pf_active'][f'coil.{ii}.element.{jj}.turns_with_sign'],
                    'r': ods['pf_active'][f'coil.{ii}.element.{jj}.geometry.rectangle.r'],
                    'z': ods['pf_active'][f'coil.{ii}.element.{jj}.geometry.rectangle.z']
                })
            coil_data.append({'elements': elements})

        # Extract passive loop data
        passive_loop_data = []
        for ii in range(len(ods['pf_passive']['loop'])):
            loop = ods['pf_passive'][f'loop.{ii}.element[0].geometry']
            loop_data = {'geometry_type': loop['geometry_type']}
            
            if loop_data['geometry_type'] == GEOMETRY_TYPE_POLYGON:
                loop_data.update({
                    'outline_r': loop['outline.r'],
                    'outline_z': loop['outline.z']
                })
            else:
                loop_data.update({
                    'rectangle_r': loop['rectangle.r'],
                    'rectangle_z': loop['rectangle.z']
                })
            passive_loop_data.append(loop_data)

        # Process plasma points
        if plasma is None:
            plasma_points = []
        elif isinstance(plasma, (list, tuple)) and len(plasma) == 2 and all(isinstance(x, (float, int)) for x in plasma):
            plasma_points = [plasma]
        elif isinstance(plasma, (list, tuple)) and len(plasma) > 0 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in plasma):
            plasma_points = plasma
        else:
            raise ValueError("plasma must be None, a single [r, z] point, or a list of [r, z] points")

        Psi_matrix, Bz_matrix, Br_matrix = compute_response_matrix(
            coil_data=coil_data,
            passive_loop_data=passive_loop_data,
            plasma_points=plasma_points,
            observation_points=rz
        )

        return Psi_matrix, Bz_matrix, Br_matrix
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise
>>>>>>> Stashed changes

def compute_grid_response_ods(
    ods: ODS,
    plasma: List[List[float]] = None
) -> ndarray:
    """Compute Green's function response matrix (Psi) on equilibrium 2D grid.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        plasma: Optional list of [r, z] plasma element points
    
    Returns:
<<<<<<< Updated upstream
        ndarray: 2D response matrix mapping coil/wall/plasma -> grid
    """
    pf = ods['pf_active']
    pfp = ods['pf_passive']
    eq = ods['equilibrium']

    nbcoil = len(pf['coil'])
    nbloop = len(pfp['loop'])
    nbplas = len(plasma)

    # Pull out grid
    r_vals = eq['time_slice.0.profiles_2d.0.grid.dim1']
    z_vals = eq['time_slice.0.profiles_2d.0.grid.dim2']
    nr = len(r_vals)
    nz = len(z_vals)

    cpsi = np.zeros((nr * nz, nbcoil + nbloop + nbplas))

    idx = 0
    for jr, rv in enumerate(r_vals):
        print(f"{jr+1}/{nr}")
        for iz, zv in enumerate(z_vals):
            # From coils
            for ii in range(nbcoil):
                nbelti = len(pf[f'coil.{ii}.element'])
                sum_phi = 0.0
                for jj in range(nbelti):
                    nbturns = pf[f'coil.{ii}.element.{jj}.turns_with_sign']
                    r2 = pf[f'coil.{ii}.element.{jj}.geometry.rectangle.r']
                    z2 = pf[f'coil.{ii}.element.{jj}.geometry.rectangle.z']
                    _, _, phi_val = compute_br_bz_phi(rv, zv, r2, z2)
                    sum_phi += phi_val * nbturns
                cpsi[idx][ii] = sum_phi

            # From passive loops
            for ii in range(nbloop):
                if pfp[f'loop.{ii}.element[0].geometry.geometry_type'] == 1:
                    nbelti = len(
                        pfp[f'loop.{ii}.element[0].geometry.outline.r']
                    )
                    r2 = sum(
                        pfp[f'loop.{ii}.element[0].geometry.outline.r']
                    ) / (nbelti - 1)
                    z2 = sum(
                        pfp[f'loop.{ii}.element[0].geometry.outline.z']
                    ) / (nbelti - 1)
                else:
                    r2 = pfp[f'loop.{ii}.element[0].geometry.rectangle.r']
                    z2 = pfp[f'loop.{ii}.element[0].geometry.rectangle.z']

                _, _, phi_val = compute_br_bz_phi(rv, zv, r2, z2)
                cpsi[idx][nbcoil + ii] = phi_val

            # From plasma
            for ii in range(nbplas):
                r2, z2 = plasma[ii]
                _, _, phi_val = compute_br_bz_phi(rv, zv, r2, z2)
                cpsi[idx][nbcoil + nbloop + ii] = phi_val

            idx += 1

    return cpsi

def compute_impedance_matrices_ods(ods, plasma: List[Tuple[float, float]]):
    """
    ODS-facing function to build or retrieve R, L, M (resistance, inductance, mutual).
    Reads ODS, calls `compute_impedance_matrices()`, and stores results in ODS.
    """
    pf = ods["pf_active"]
    pfp = ods["pf_passive"]
    em = ods["em_coupling"]

    nbcoil = len(pf["coil"])
    nbloop = len(pfp["loop"])
    loop_res = np.zeros(nbloop)

    # Resistances
    for i_loop in range(nbloop):
        loop_res[i_loop] = pfp[f"loop.{i_loop}.resistance"]

    # M among loops
    mutual_pp = em["mutual_passive_passive"]  # shape (nbloop, nbloop)

    # M with coil (and possibly plasma)
    mutual_pa = em["mutual_passive_active"]   # shape (nbloop, nbcoil)

    # Prepare loop geometry info
    # Example: for each loop, we compute average R,Z, plus the "coef" logic from original code
    passive_loop_geometry = []
    for i_loop in range(nbloop):
        loop_name = pfp[f"loop.{i_loop}.name"]
        # Example logic for geometry
        geom_type = pfp[f"loop.{i_loop}.element.0.geometry.geometry_type"]
        if geom_type == 1:
            # polygon with 4 corners
            r_list = pfp[f"loop.{i_loop}.element.0.geometry.outline.r"]
            z_list = pfp[f"loop.{i_loop}.element.0.geometry.outline.z"]
            r_avg = sum(r_list) / len(r_list)
            z_avg = sum(z_list) / len(z_list)
        else:
            # rectangle
            r_avg = pfp[f"loop.{i_loop}.element.0.geometry.rectangle.r"]
            z_avg = pfp[f"loop.{i_loop}.element.0.geometry.rectangle.z"]

        coef = 1.0 if loop_name == "W11" else 1.04
        passive_loop_geometry.append((loop_name, r_avg, z_avg, coef))

    # Coil geometry (list of lists)
    coil_geometry = []
    for i_coil in range(nbcoil):
        n_elem = len(pf[f"coil.{i_coil}.element"])
        c_geom = []
        for j_el in range(n_elem):
            turns = pf[f"coil.{i_coil}.element.{j_el}.turns_with_sign"]
            rc = pf[f"coil.{i_coil}.element.{j_el}.geometry.rectangle.r"]
            zc = pf[f"coil.{i_coil}.element.{j_el}.geometry.rectangle.z"]
            c_geom.append((rc, zc, turns))
        coil_geometry.append(c_geom)

    # Call the *core* function
    R_mat, L_mat, M_mat = compute_impedance_matrices(
        loop_res,
        passive_loop_geometry,
        coil_geometry,
        mutual_pp,
        mutual_pa,
        plasma
    )

    # Optionally store R_mat, L_mat, M_mat back into ODS, or return them
    # Example direct store:
    pfp["R_mat"] = R_mat
    pfp["L_mat"] = L_mat
    pfp["M_mat"] = M_mat

    return R_mat, L_mat, M_mat

def compute_eddy_currents(ods, plasma: List[Tuple[float, float]], ip: List[np.ndarray]) -> None:
    """
    ODS-facing function that uses the precomputed or newly computed impedance
    matrices, then solves the eddy currents in the passive loops. Writes solution to ODS.
    """
    pf = ods["pf_active"]
    pfp = ods["pf_passive"]

    nbcoil = len(pf["coil"])
    nbloop = len(pfp["loop"])
    nbplas = len(plasma)
    time_arr = pf["time"]
    nbt = len(time_arr)

    # 1) Acquire R, L, M
    try:
        R_mat = pfp["R_mat"]
        L_mat = pfp["L_mat"]
        M_mat = pfp["M_mat"]
    except KeyError:
        # If not found, compute on the fly
        R_mat, L_mat, M_mat = compute_impedance_matrices(ods, plasma)

    # 2) Build coil+plasma current vs time array
    # shape => (n_times, nbcoil+nbplas)
    coil_plasma_currents = []
    for i_coil in range(nbcoil):
        coil_plasma_currents.append(pf[f"coil.{i_coil}.current.data"])
    for i_p in range(nbplas):
        coil_plasma_currents.append(ip[i_p])

    coil_plasma_currents = np.array(coil_plasma_currents).T  # shape => (n_times, nbcoil+nbplas)

    # 3) Solve eddy currents
    I_loop = solve_eddy_currents(
        R_mat, 
        L_mat, 
        M_mat, 
        coil_plasma_currents, 
        time_arr,
        dt_sub=1e-6
    )

    # 4) Store results back in ODS
    pfp["time"] = time_arr
    for i_loop in range(nbloop):
        pfp[f"loop.{i_loop}.current"] = I_loop[:, i_loop]

def compute_vacuum_fields_1d(ods, rz: List[Tuple[float, float]]):
    """
    ODS-facing function to compute vacuum fields at 1D points (rz),
    ignoring plasma (or after eddy current solution).
    """
    pf = ods["pf_active"]
    pfp = ods["pf_passive"]
    nbcoil = len(pf["coil"])
    nbloop = len(pfp["loop"])
    time_arr = pf["time"]
    nbt = len(time_arr)

    # 1) Ensure eddy currents are computed
    #    (which calls compute_impedance_matrices if needed)
    # For vacuum, pass an empty plasma list or ip list:
    compute_eddy_currents(ods, plasma=[], ip=[])

    # 2) Build coil+loop response arrays (psi_c, br_c, bz_c).
    #    Suppose you have a function that computes the "response vectors"
    #    for each point in `rz`. This was done in your original `vest_rspv1` call.
    #    We'll just pretend we have them in ODS or we compute them now:
    psi_c = ods.get("psi_c", None)  
    br_c  = ods.get("br_c", None)
    bz_c  = ods.get("bz_c", None)
    # shape => e.g. (n_points, nb_coil+nb_loop)

    # If not present, you can fill them by calling your geometry-based function:
    # psi_c, br_c, bz_c = precompute_vacuum_responses(rz, coil_geometry, loop_geometry)

    # 3) Collect coil + loop currents
    coil_loop_curr = np.zeros((nbt, nbcoil + nbloop))
    for t in range(nbt):
        for i_coil in range(nbcoil):
            coil_loop_curr[t, i_coil] = pf[f"coil.{i_coil}.current.data"][t]
        for i_loop in range(nbloop):
            coil_loop_curr[t, nbcoil + i_loop] = pfp[f"loop.{i_loop}.current"][t]

    # 4) Call the pure function
    psi_out, br_out, bz_out = compute_vacuum_fields_1d(
        coil_loop_curr,
        psi_c,
        br_c,
        bz_c
    )

    # 5) Store in ODS or return
    # For example, store as arrays in `ods`
    ods["vac_fields_1d"] = {
        "time": time_arr,
        "rz_points": rz,
        "psi": psi_out,  # shape (n_times, n_points)
        "br": br_out,
        "bz": bz_out
    }

    return time_arr, psi_out, br_out, bz_out
=======
        Response matrix for Psi calculation

    Raises:
        KeyError: If required ODS data is missing
    """
    try:
        pf = ods['pf_active']
        pfp = ods['pf_passive']
        eq = ods['equilibrium']

        nbcoil = len(pf['coil'])
        nbloop = len(pfp['loop'])

        # Process plasma points
        if plasma is None:
            nbplas = 0
            plasma_points_arr = np.empty((0, 2))
        elif isinstance(plasma, (list, tuple)) and len(plasma) == 2 and all(isinstance(x, (float, int)) for x in plasma):
            nbplas = 1
            plasma_points_arr = np.array([plasma])
        elif isinstance(plasma, (list, tuple)) and len(plasma) > 0 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in plasma):
            nbplas = len(plasma)
            plasma_points_arr = np.array(plasma)
        else:
            raise ValueError("plasma must be None, a single [r, z] point, or a list of [r, z] points")

        # Get grid coordinates
        r_vals = eq['time_slice.0.profiles_2d.0.grid.dim1']
        z_vals = eq['time_slice.0.profiles_2d.0.grid.dim2']
        
        R_obs, Z_obs = np.meshgrid(r_vals, z_vals)
        R_obs_flat = R_obs.flatten()
        Z_obs_flat = Z_obs.flatten()
        n_grid_points = len(R_obs_flat)

        cpsi = np.zeros((n_grid_points, nbcoil + nbloop + nbplas))
        
        logger.info("Computing grid response (Psi)...")

        # Compute coil contributions
        logger.info(f"Processing {nbcoil} active coils...")
        temp_coil_psi_sum = np.zeros((n_grid_points, nbcoil))
        for ii in range(nbcoil):
            current_coil_total_phi = np.zeros(n_grid_points)
            for jj in range(len(pf[f'coil.{ii}.element'])):
                nbturns = pf[f'coil.{ii}.element.{jj}.turns_with_sign']
                r2_coil = pf[f'coil.{ii}.element.{jj}.geometry.rectangle.r']
                z2_coil = pf[f'coil.{ii}.element.{jj}.geometry.rectangle.z']
                
                _, _, phi_grid_flat_for_element = compute_br_bz_phi(R_obs_flat, Z_obs_flat, r2_coil, z2_coil)
                current_coil_total_phi += phi_grid_flat_for_element * nbturns
            temp_coil_psi_sum[:, ii] = current_coil_total_phi
        cpsi[:, :nbcoil] = temp_coil_psi_sum
        logger.info("Coil contributions complete.")

        # Compute loop contributions
        logger.info(f"Processing {nbloop} passive loops...")
        for ii in range(nbloop):
            if pfp[f'loop.{ii}.element[0].geometry.geometry_type'] == GEOMETRY_TYPE_POLYGON:
                r2_loop = np.mean(pfp[f'loop.{ii}.element[0].geometry.outline.r'])
                z2_loop = np.mean(pfp[f'loop.{ii}.element[0].geometry.outline.z'])
            else:
                r2_loop = pfp[f'loop.{ii}.element[0].geometry.rectangle.r']
                z2_loop = pfp[f'loop.{ii}.element[0].geometry.rectangle.z']
            
            _, _, phi_grid_flat = compute_br_bz_phi(R_obs_flat, Z_obs_flat, r2_loop, z2_loop)
            cpsi[:, nbcoil + ii] = phi_grid_flat
        logger.info("Loop contributions complete.")

        # Compute plasma contributions if any
        if nbplas > 0:
            logger.info(f"Processing {nbplas} plasma points...")
            for ii_plas in range(nbplas):
                r2_plas, z2_plas = plasma_points_arr[ii_plas]
                _, _, phi_grid_flat = compute_br_bz_phi(R_obs_flat, Z_obs_flat, r2_plas, z2_plas)
                cpsi[:, nbcoil + nbloop + ii_plas] = phi_grid_flat
            logger.info("Plasma contributions complete.")
        
        logger.info("Grid response computation complete.")
        return cpsi

    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise
    except Exception as e: 
        logger.error(f"Error during computation: {e}")
        raise

def compute_impedance_matrices_ods(
    ods: ODS,
    plasma: List[Tuple[float, float]]
) -> Tuple[ndarray, ndarray, ndarray]:
    """Compute R, L, M matrices for eddy current calculations.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        plasma: List of (r, z) tuples for plasma elements

    Returns:
        Tuple of (R_mat, L_mat, M_mat) arrays

    Raises:
        KeyError: If required ODS data is missing
    """
    try:
        pf = ods["pf_active"]
        pfp = ods["pf_passive"]
        em = ods["em_coupling"]

        nbcoil = len(pf["coil"])
        nbloop = len(pfp["loop"])

        # Extract loop resistances
        loop_res = np.array([pfp[f"loop.{i_loop}.resistance"] for i_loop in range(nbloop)])

        # Get mutual inductances
        mutual_pp = em["mutual_passive_passive"]
        mutual_pa = em["mutual_passive_active"]

        # Extract loop geometries
        passive_loop_geometry = []
        for i_loop in range(nbloop):
            loop_name = pfp[f"loop.{i_loop}.name"]
            geom_type = pfp[f"loop.{i_loop}.element.0.geometry.geometry_type"]
            
            if geom_type == GEOMETRY_TYPE_POLYGON:
                r_list = pfp[f"loop.{i_loop}.element.0.geometry.outline.r"]
                z_list = pfp[f"loop.{i_loop}.element.0.geometry.outline.z"]
                r_avg = np.mean(r_list)
                z_avg = np.mean(z_list)
            else:
                r_avg = pfp[f"loop.{i_loop}.element.0.geometry.rectangle.r"]
                z_avg = pfp[f"loop.{i_loop}.element.0.geometry.rectangle.z"]

            coef = 1.0 if loop_name == "W11" else 1.04
            passive_loop_geometry.append((loop_name, r_avg, z_avg, coef))

        # Extract coil geometries
        coil_geometry = []
        for i_coil in range(nbcoil):
            n_elem = len(pf[f"coil.{i_coil}.element"])
            c_geom = []
            for j_el in range(n_elem):
                turns = pf[f"coil.{i_coil}.element.{j_el}.turns_with_sign"]
                rc = pf[f"coil.{i_coil}.element.{j_el}.geometry.rectangle.r"]
                zc = pf[f"coil.{i_coil}.element.{j_el}.geometry.rectangle.z"]
                c_geom.append((rc, zc, turns))
            coil_geometry.append(c_geom)

        # Compute impedance matrices
        R_mat, L_mat, M_mat = compute_impedance_matrices(
            loop_res,
            passive_loop_geometry,
            coil_geometry,
            mutual_pp,
            mutual_pa,
            plasma
        )

        # Store results in ODS
        pfp["R_mat"] = R_mat
        pfp["L_mat"] = L_mat
        pfp["M_mat"] = M_mat

        return R_mat, L_mat, M_mat
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

def compute_eddy_currents(
    ods: ODS,
    plasma: List[Tuple[float, float]],
    ip: List[ndarray]
) -> None:
    """Solve eddy currents in passive loops using precomputed impedance matrices.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        plasma: List of (r, z) tuples for plasma elements
        ip: List of plasma current arrays

    Raises:
        KeyError: If required ODS data is missing
    """
    try:
        pf = ods["pf_active"]
        pfp = ods["pf_passive"]

        nbcoil = len(pf["coil"])
        nbloop = len(pfp["loop"])
        nbplas = len(plasma)
        time_arr = pf["time"]
        nbt = len(time_arr)

        # Get or compute impedance matrices
        R_mat, L_mat, M_mat = compute_impedance_matrices_ods(ods, plasma)

        # Construct current array
        coil_plasma_currents = np.array([
            pf[f"coil.{i_coil}.current.data"] for i_coil in range(nbcoil)
        ] + [ip[i_p] for i_p in range(nbplas)]).T

        # Solve eddy currents
        I_loop = solve_eddy_currents(
            R_mat, 
            L_mat, 
            M_mat, 
            coil_plasma_currents, 
            time_arr,
            dt_sub=DT_SUB
        )

        # Store results
        pfp["time"] = time_arr
        for i_loop in range(nbloop):
            pfp[f"loop.{i_loop}.current"] = I_loop[:, i_loop]
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

def compute_point_vacuum_fields_ods(
    ods: ODS,
    rz: List[Tuple[float, float]] = [(0.4, 0.0)],
    plot_opt: bool = False,
    mode: str = 'vacuum'
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Compute vacuum fields at observation points.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        rz: List of (r, z) observation points (default: [(0.4, 0.0)])
        plot_opt: Whether to plot results
        mode: Which contributions to include in calculation:
            - 'vacuum': Include both PF active and passive contributions (default)
            - 'pf_active': Include only PF active coil contributions
            - 'pf_passive': Include only PF passive loop contributions
    
    Returns:
        Tuple of (time_arr, psi_out, br_out, bz_out) arrays

    Raises:
        KeyError: If required ODS data is missing
        ValueError: If invalid mode is specified
    """
    try:
        if mode not in ['vacuum', 'pf_active', 'pf_passive']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: vacuum, pf_active, pf_passive")

        pf = ods["pf_active"]
        pfp = ods["pf_passive"]
        nbcoil = len(pf["coil"])
        nbloop = len(pfp["loop"])
        time_arr = pf["time"]
        nbt = len(time_arr)

        # Compute response matrix
        psi_c, br_c, bz_c = compute_point_response_ods(ods, rz, plasma=None)
        
        # Verify response matrix shapes
        expected_sources = nbcoil + nbloop
        if psi_c.shape[1] != expected_sources or \
           br_c.shape[1] != expected_sources or \
           bz_c.shape[1] != expected_sources:
            raise RuntimeError(f"Response matrix shape mismatch. Expected: {expected_sources}, Got: {psi_c.shape[1]}")

        # Construct current array based on mode
        coil_loop_curr = np.zeros((nbt, nbcoil + nbloop))
        for t in range(nbt):
            if mode in ['vacuum', 'pf_active']:
                coil_loop_curr[t, :nbcoil] = [pf[f"coil.{i_coil}.current.data"][t] for i_coil in range(nbcoil)]
            if mode in ['vacuum', 'pf_passive']:
                coil_loop_curr[t, nbcoil:] = [pfp[f"loop.{i_loop}.current"][t] for i_loop in range(nbloop)]

        # Compute vacuum fields
        psi_out, br_out, bz_out = vaft.process.compute_vacuum_fields_1d(
            coil_loop_curr,
            psi_c,
            br_c,
            bz_c
        )

        # Plot if requested
        if plot_opt:
            n_points = psi_out.shape[1] if psi_out.ndim == 2 else 1
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            if n_points == 1:
                psi_plot = psi_out[:, 0] if psi_out.ndim == 2 else psi_out
                br_plot = br_out[:, 0] if br_out.ndim == 2 else br_out
                bz_plot = bz_out[:, 0] if bz_out.ndim == 2 else bz_out
                label = f"(r={rz[0][0]:.3f}, z={rz[0][1]:.3f})" if isinstance(rz[0], (list, tuple)) else str(rz[0])
                axs[0].plot(time_arr, psi_plot, label=label)
                axs[1].plot(time_arr, br_plot, label=label)
                axs[2].plot(time_arr, bz_plot, label=label)
            else:
                for i in range(n_points):
                    psi_plot = psi_out[:, i]
                    br_plot = br_out[:, i]
                    bz_plot = bz_out[:, i]
                    label = f"(r={rz[i][0]:.3f}, z={rz[i][1]:.3f})" if isinstance(rz[i], (list, tuple)) else str(rz[i])
                    axs[0].plot(time_arr, psi_plot, label=label)
                    axs[1].plot(time_arr, br_plot, label=label)
                    axs[2].plot(time_arr, bz_plot, label=label)
            axs[0].set_ylabel("ψ_out")
            axs[1].set_ylabel("B_r")
            axs[2].set_ylabel("B_z")
            axs[2].set_xlabel("Time [s]")
            axs[0].set_title(f"Vacuum Field Quantities at Each Time Step (Mode: {mode})")
            for ax in axs:
                ax.legend()
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
        return time_arr, psi_out, br_out, bz_out
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise
>>>>>>> Stashed changes

def compute_null_ods(ods, time):
    """Compute poloidal flux (psi) on grid at given time using coil and eddy currents.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        time: Time point for computation
    
    Returns:
        Tuple of (psi_reshaped, R_mesh, Z_mesh)
    """
    cpsi = compute_grid_response_ods(ods)
    time_eddy = ods['pf_passive']['time']
    time_idx = np.argmin(np.abs(time_eddy - time))
    coil_current = np.array([ods['pf_active'][f'coil.{i}.current.data'][time_idx] for i in range(len(ods['pf_active']['coil']))])
    eddy_current = np.array([ods['pf_passive'][f'loop.{i}.current'][time_idx] for i in range(len(ods['pf_passive']['loop']))])
    
    currents_combined = np.concatenate((coil_current, eddy_current))
    psi_flat = np.dot(cpsi, currents_combined)
    
    rgrid = ods['equilibrium']['time_slice.0.profiles_2d.0.grid.dim1']
    zgrid = ods['equilibrium']['time_slice.0.profiles_2d.0.grid.dim2']
    
    R_mesh, Z_mesh = np.meshgrid(rgrid, zgrid)
    psi_reshaped = psi_flat.reshape(len(zgrid), len(rgrid))

    return psi_reshaped, R_mesh, Z_mesh
