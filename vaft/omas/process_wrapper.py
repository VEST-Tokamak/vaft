from typing import List, Tuple, Dict, Any, Optional
from numpy import ndarray
import numpy as np
from omas import *
from vaft.process import compute_br_bz_phi, compute_response_matrix, compute_response_vector, compute_impedance_matrices, solve_eddy_currents, compute_vacuum_fields_1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DT_SUB = 1e-6  # Time step size for eddy current calculation
GEOMETRY_TYPE_POLYGON = 1
GEOMETRY_TYPE_RECTANGLE = 2

def calc_grid_ods(ods: Dict[str, Any], xvar: List[float], zvar: List[float]) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Wrapper function for calc_grid to work with OMAS data structure.

    Args:
        ods: OMAS data structure containing PF coil and loop data
        xvar: List of x (radial) coordinates
        zvar: List of z (vertical) coordinates

    Returns:
        Tuple of (Br, Bz, Phi) matrices

    Raises:
        KeyError: If required data is missing from ODS
    """
    try:
        pf = ods['pf_active']
        pfp = ods['pf_passive']

        # Vectorized coil data extraction
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

        # Vectorized loop data extraction
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

def cal_response_vector_ods(
    ods: ODS,
    plasma: List[List[float]],
    rz: List[List[float]]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    ODS wrapper for computing response matrix (Psi, Bz, Br).

    Args:
        ods: OMAS data structure containing `pf_active` & `pf_passive`
        plasma: List of [r, z] points for plasma elements (if any)
        rz: List of [r, z] observation points
    
    Returns:
        Tuple of (Psi, Bz, Br) arrays with shape (len(rz), nb_coil+nb_loop+nb_plasma)

    Raises:
        KeyError: If required data is missing from ODS
    """
    try:
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

        return cal_response_vector(
            coil_data=coil_data,
            passive_loop_data=passive_loop_data,
            plasma_points=plasma,
            observation_points=rz
        )
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

def compute_response_matrix_ods(
    ods: ODS,
    plasma: List[List[float]]
) -> ndarray:
    """
    Compute Green's function table (coil/wall -> 2D grid).
    If plasma is present, it's appended. Typically for vacuum.

    Args:
        ods: OMAS data structure containing equilibrium and PF coil data
        plasma: List of [r, z] for plasma elements
    
    Returns:
        ndarray: 2D response matrix mapping coil/wall/plasma -> grid

    Raises:
        KeyError: If required data is missing from ODS
    """
    try:
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

        # Vectorized computation for each grid point
        for jr, rv in enumerate(r_vals):
            logger.info(f"Processing radial point {jr+1}/{nr}")
            for iz, zv in enumerate(z_vals):
                idx = jr * nz + iz
                
                # Vectorized coil contribution
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

                # Vectorized loop contribution
                for ii in range(nbloop):
                    if pfp[f'loop.{ii}.element[0].geometry.geometry_type'] == GEOMETRY_TYPE_POLYGON:
                        nbelti = len(pfp[f'loop.{ii}.element[0].geometry.outline.r'])
                        r2 = np.mean(pfp[f'loop.{ii}.element[0].geometry.outline.r'])
                        z2 = np.mean(pfp[f'loop.{ii}.element[0].geometry.outline.z'])
                    else:
                        r2 = pfp[f'loop.{ii}.element[0].geometry.rectangle.r']
                        z2 = pfp[f'loop.{ii}.element[0].geometry.rectangle.z']

                    _, _, phi_val = compute_br_bz_phi(rv, zv, r2, z2)
                    cpsi[idx][nbcoil + ii] = phi_val

                # Vectorized plasma contribution
                for ii in range(nbplas):
                    r2, z2 = plasma[ii]
                    _, _, phi_val = compute_br_bz_phi(rv, zv, r2, z2)
                    cpsi[idx][nbcoil + nbloop + ii] = phi_val

        return cpsi
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

def compute_impedance_matrices_ods(
    ods: ODS,
    plasma: List[Tuple[float, float]]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    ODS-facing function to build or retrieve R, L, M (resistance, inductance, mutual).
    Reads ODS, calls `compute_impedance_matrices()`, and stores results in ODS.

    Args:
        ods: OMAS data structure containing PF coil and loop data
        plasma: List of (r, z) tuples for plasma elements

    Returns:
        Tuple of (R_mat, L_mat, M_mat) arrays

    Raises:
        KeyError: If required data is missing from ODS
    """
    try:
        pf = ods["pf_active"]
        pfp = ods["pf_passive"]
        em = ods["em_coupling"]

        nbcoil = len(pf["coil"])
        nbloop = len(pfp["loop"])
        loop_res = np.zeros(nbloop)

        # Vectorized resistance extraction
        loop_res = np.array([pfp[f"loop.{i_loop}.resistance"] for i_loop in range(nbloop)])

        # Mutual inductances
        mutual_pp = em["mutual_passive_passive"]  # shape (nbloop, nbloop)
        mutual_pa = em["mutual_passive_active"]   # shape (nbloop, nbcoil)

        # Vectorized loop geometry extraction
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

        # Vectorized coil geometry extraction
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
    """
    ODS-facing function that uses the precomputed or newly computed impedance
    matrices, then solves the eddy currents in the passive loops. Writes solution to ODS.

    Args:
        ods: OMAS data structure containing PF coil and loop data
        plasma: List of (r, z) tuples for plasma elements
        ip: List of plasma current arrays

    Raises:
        KeyError: If required data is missing from ODS
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
        try:
            R_mat = pfp["R_mat"]
            L_mat = pfp["L_mat"]
            M_mat = pfp["M_mat"]
        except KeyError:
            logger.info("Impedance matrices not found in ODS, computing on the fly")
            R_mat, L_mat, M_mat = compute_impedance_matrices_ods(ods, plasma)

        # Vectorized current array construction
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

        # Store results in ODS
        pfp["time"] = time_arr
        for i_loop in range(nbloop):
            pfp[f"loop.{i_loop}.current"] = I_loop[:, i_loop]
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

def compute_vacuum_fields_1d(
    ods: ODS,
    rz: List[Tuple[float, float]],
    plot_opt: bool = False
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    ODS-facing function to compute vacuum fields at 1D points (rz),
    ignoring plasma (or after eddy current solution).

    Args:
        ods: OMAS data structure containing PF coil and loop data
        rz: List of (r, z) tuples for observation points
        plot_opt: Whether to plot the results

    Returns:
        Tuple of (time_arr, psi_out, br_out, bz_out) arrays

    Raises:
        KeyError: If required data is missing from ODS
    """
    try:
        pf = ods["pf_active"]
        pfp = ods["pf_passive"]
        nbcoil = len(pf["coil"])
        nbloop = len(pfp["loop"])
        time_arr = pf["time"]
        nbt = len(time_arr)

        # Compute eddy currents
        compute_eddy_currents(ods, plasma=[], ip=[])

        # Get response vectors
        psi_c = ods.get("psi_c")
        br_c = ods.get("br_c")
        bz_c = ods.get("bz_c")

        if any(x is None for x in [psi_c, br_c, bz_c]):
            logger.warning("Response vectors not found in ODS, computing on the fly")
            psi_c, br_c, bz_c = cal_response_vector_ods(ods, [], rz)

        # Vectorized current array construction
        coil_loop_curr = np.zeros((nbt, nbcoil + nbloop))
        for t in range(nbt):
            coil_loop_curr[t, :nbcoil] = [pf[f"coil.{i_coil}.current.data"][t] for i_coil in range(nbcoil)]
            coil_loop_curr[t, nbcoil:] = [pfp[f"loop.{i_loop}.current"][t] for i_loop in range(nbloop)]

        # Compute vacuum fields
        psi_out, br_out, bz_out = compute_vacuum_fields_1d(
            coil_loop_curr,
            psi_c,
            br_c,
            bz_c
        )

        # Plot if requested
        if plot_opt and len(rz) == 1:
            import matplotlib.pyplot as plt
            vloop = np.gradient(psi_out, time_arr)

            fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
            fig.suptitle("Vacuum Field Quantities at Each Time Step")

            axs[0, 0].plot(time_arr, psi_out)
            axs[0, 0].set_ylabel("ψ_out")

            axs[0, 1].plot(time_arr, vloop)
            axs[0, 1].set_ylabel("V_loop")

            axs[1, 0].plot(time_arr, br_out)
            axs[1, 0].set_ylabel("B_r")
            axs[1, 0].set_xlabel("Time [s]")

            axs[1, 1].plot(time_arr, bz_out)
            axs[1, 1].set_ylabel("B_z")
            axs[1, 1].set_xlabel("Time [s]")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        return time_arr, psi_out, br_out, bz_out
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

