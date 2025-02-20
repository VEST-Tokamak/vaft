from typing import List, Tuple, Dict, Any
from numpy import ndarray
import numpy as np
from omas import *
from vest.process import compute_br_bz_phi, compute_response_matrix, compute_response_vector, compute_impedance_matrices, solve_eddy_currents, compute_vacuum_fields_1d

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

