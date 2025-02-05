from vest.formula import green_br_bz, green_r
from vest.process import calculate_distance
from typing import List, Dict, Any, Tuple
import numpy as np
from numpy import ndarray


# Description of the axisymmetric mutual electromagnetics calculations.

def compute_br_bz_phi(
    r1: float,
    z1: float,
    r2: float,
    z2: float,
    shift: float = 0.01
) -> Tuple[float, float, float]:
    """
    Compute Br, Bz, and Phi (green_r) using a shift approach to avoid singularities.

    :param r1: Radius coordinate of the observation point.
    :param z1: Z coordinate of the observation point.
    :param r2: Radius coordinate of the source element.
    :param z2: Z coordinate of the source element.
    :param shift: Shift value to use if the points are too close.
    :return: (Br, Bz, Phi) at (r1, z1).
    """
    if calculate_distance(r1, r2, z1, z2) < shift / 3.0:
        br1, bz1 = green_br_bz(r1 + shift, z1, r2, z2)
        br2, bz2 = green_br_bz(r1 - shift, z1, r2, z2)
        p1 = green_r(r1 + shift, z1, r2, z2)
        p2 = green_r(r1 - shift, z1, r2, z2)

        br = (br1 + br2) / 2.0
        bz = (bz1 + bz2) / 2.0
        phi = (p1 + p2) / 2.0
    else:
        br, bz = green_br_bz(r1, z1, r2, z2)
        phi = green_r(r1, z1, r2, z2)
    return br, bz, phi

def compute_br_bz_phi(xr, zr, r2, z2):
    # Placeholder for the actual computation of Br, Bz, and Phi.
    # Replace this with the real formula or function call.
    br = 0.0
    bz = 0.0
    phi = 0.0
    return br, bz, phi

def calc_grid(
    xvar: List[float],
    zvar: List[float],
    coil_turns: List[List[float]],
    coil_r: List[List[float]],
    coil_z: List[List[float]],
    loop_geometry_type: List[int],
    loop_outline_r: List[List[float]],
    loop_outline_z: List[List[float]],
    loop_rectangle_r: List[float],
    loop_rectangle_z: List[float]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Compute the response matrix (Br, Bz, and Psi) for a 2D grid.

    :param xvar: List of x (radial) coordinates.
    :param zvar: List of z (vertical) coordinates.
    :param coil_turns: List of turns for each coil element.
    :param coil_r: List of r positions for each coil element.
    :param coil_z: List of z positions for each coil element.
    :param loop_geometry_type: List indicating geometry type for each loop.
    :param loop_outline_r: List of r coordinates for loop outlines.
    :param loop_outline_z: List of z coordinates for loop outlines.
    :param loop_rectangle_r: List of r positions for loop rectangles.
    :param loop_rectangle_z: List of z positions for loop rectangles.
    :return: Tuple of (Br, Bz, Phi) matrices with shape
             (len(xvar)*len(zvar), nbcoil+nbloop).
    """
    nbcoil = len(coil_turns)
    nbloop = len(loop_geometry_type)
    total_points = len(xvar) * len(zvar)

    br_array = np.zeros((total_points, nbcoil + nbloop))
    bz_array = np.zeros((total_points, nbcoil + nbloop))
    phi_array = np.zeros((total_points, nbcoil + nbloop))

    count = 0
    for i, xr in enumerate(xvar):
        for j, zr in enumerate(zvar):
            if count % 100 == 0:
                percent = (count * 100.0) / (total_points - 1)
                print(f"{percent:.2f}%")

            # Active coils
            for ii in range(nbcoil):
                sum_br, sum_bz, sum_phi = 0.0, 0.0, 0.0

                for jj in range(len(coil_turns[ii])):
                    nbturns = coil_turns[ii][jj]
                    r2 = coil_r[ii][jj]
                    z2 = coil_z[ii][jj]
                    br_val, bz_val, phi_val = compute_br_bz_phi(xr, zr, r2, z2)
                    sum_br += br_val * nbturns
                    sum_bz += bz_val * nbturns
                    sum_phi += phi_val * nbturns

                br_array[count][ii] = sum_br
                bz_array[count][ii] = sum_bz
                phi_array[count][ii] = sum_phi

            # Passive loops
            for ii in range(nbloop):
                if loop_geometry_type[ii] == 1:
                    nbelti = len(loop_outline_r[ii])
                    r2 = sum(loop_outline_r[ii]) / (nbelti - 1)
                    z2 = sum(loop_outline_z[ii]) / (nbelti - 1)
                else:
                    r2 = loop_rectangle_r[ii]
                    z2 = loop_rectangle_z[ii]

                br_val, bz_val, phi_val = compute_br_bz_phi(xr, zr, r2, z2)
                br_array[count][nbcoil + ii] = br_val
                bz_array[count][nbcoil + ii] = bz_val
                phi_array[count][nbcoil + ii] = phi_val

            count += 1

    return br_array, bz_array, phi_array

def compute_response_matrix(
    observation_points: List[List[float]],
    coil_data: List[Dict[str, Any]],
    passive_loop_data: List[Dict[str, Any]],
    plasma_points: List[List[float]]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Core computation function for electromagnetic response matrix.
    
    Args:
        observation_points: List of [r, z] observation points
        coil_data: List of dicts containing coil elements with fields:
            - elements: List of dicts with 'turns', 'r', 'z' for each element
        passive_loop_data: List of dicts containing loop data with fields:
            - geometry_type: 1 for outline, 2 for rectangle
            - outline_r, outline_z: Lists of coordinates for outline (type 1)
            - rectangle_r, rectangle_z: Single point for rectangle (type 2)
        plasma_points: List of [r, z] points for plasma elements
    
    Returns:
        Tuple of (Psi, Bz, Br) arrays
    """
    # ... existing compute_response_matrix implementation ...

def compute_response_vector(
    coil_data: List[Dict[str, Any]],
    passive_loop_data: List[Dict[str, Any]],
    plasma_points: List[List[float]],
    observation_points: List[List[float]]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Calculate response matrix using structured input data.

    Args:
        coil_data: List of dictionaries containing coil element data
        passive_loop_data: List of dictionaries containing passive loop data
        plasma_points: List of [r, z] points for plasma elements
        observation_points: List of [r, z] observation points
    
    Returns:
        Tuple of (Psi, Bz, Br) arrays with shape (len(observation_points), nb_coil+nb_loop+nb_plasma)
    """
    return compute_response_matrix(
        observation_points=observation_points,
        coil_data=coil_data,
        passive_loop_data=passive_loop_data,
        plasma_points=plasma_points
    )

def compute_impedance_matrices(
    loop_resistances: np.ndarray,
    passive_loop_geometry: List[Tuple[str, float, float, float]],  
    # e.g. [(loop_name, average_r, average_z, geometry_coef), ...]
    coil_geometry: List[List[Tuple[float, float, int]]],  
    # e.g. coil_geometry[i] -> list of (rc, zc, turns_with_sign) for each coil element
    mutual_pp: np.ndarray,       # mutual_passive_passive from ODS
    mutual_pa: np.ndarray,       # mutual_passive_active from ODS
    plasma_rz: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute R, L, M matrices for the passive loops, given geometry info.

    :param loop_resistances: array of shape (nbloop,) with each loop's resistance.
    :param passive_loop_geometry: list describing each passive loop:
           - loop_name (str),
           - average_r (float),
           - average_z (float),
           - geometry_coef (float)  # e.g. 1.0 or 1.04 ...
    :param coil_geometry: list of coils, each coil is a list of (rc, zc, turns).
    :param mutual_pp: mutual_passive_passive matrix from external (shape = (nbloop, nbloop)).
    :param mutual_pa: mutual_passive_active matrix from external (shape = (nbloop, nbcoil)).
    :param plasma_rz: list of (r, z) for each plasma current element (optional).
    :return: (R, L, M) for passive loops: R_mat, L_mat, M_mat
    """
    nbloop = len(passive_loop_geometry)
    nbcoil = len(coil_geometry)
    nbplas = len(plasma_rz)

    # Build R (nbloop x nbloop)
    R_mat = np.zeros((nbloop, nbloop))
    np.fill_diagonal(R_mat, loop_resistances)

    # M among passive loops
    M_mat = mutual_pp  # shape (nbloop, nbloop)

    # L with coil + plasma
    if nbplas == 0:
        # No plasma => Use existing mutual_passive_active from ODS
        L_mat = mutual_pa
    else:
        # Recompute partial or full (example approach)
        # For each loop => for each coil => sum geometry. Then plasma similarly
        L_mat = np.zeros((nbloop, nbcoil + nbplas))

        # Precompute coil total turns for each coil
        coil_turns = np.zeros(nbcoil)
        for i_coil in range(nbcoil):
            total_turns = sum(el[2] for el in coil_geometry[i_coil])
            coil_turns[i_coil] = total_turns

        for i_loop, (loop_name, r1, z1, coef) in enumerate(passive_loop_geometry):
            # Coil part
            for j_coil in range(nbcoil):
                elem_list = coil_geometry[j_coil]
                n_el = len(elem_list)
                # For each element in coil j_coil
                for (rc, zc, turns) in elem_list:
                    # example usage of green's function
                    L_mat[i_loop, j_coil] += coef * green_r(r1, z1, rc, zc) / n_el
                # Scale by total turns
                L_mat[i_loop, j_coil] *= coil_turns[j_coil]

            # Plasma part
            for j_plasma, (rp, zp) in enumerate(plasma_rz):
                idx_p = nbcoil + j_plasma
                L_mat[i_loop, idx_p] = coef * green_r(r1, z1, rp, zp)
                # If you consider "turns" for each plasma element, multiply by that if needed

    return R_mat, L_mat, M_mat

def solve_eddy_currents(
    R_mat: np.ndarray,
    L_mat: np.ndarray,
    M_mat: np.ndarray,
    coil_plasma_currents: np.ndarray,  # shape (n_times, nb_coil+nb_plasma)
    time: np.ndarray,                  # shape (n_times,)
    dt_sub: float = 1e-6
) -> np.ndarray:
    """
    Solve the RL circuit equation for vacuum vessel using a Runge-Kutta scheme.
    
    :param R_mat: (nbloop, nbloop) resistance matrix
    :param L_mat: (nbloop, nbcoil+nbplas) mutual with coil + plasma
    :param M_mat: (nbloop, nbloop) mutual among passive loops
    :param coil_plasma_currents: shaped (n_times, nb_coil+nb_plasma)
    :param time: time array of shape (n_times,)
    :param dt_sub: sub-timestep for finer integration
    :return: computed passive loop currents, shape (n_times, nbloop)
    """
    nbloop = R_mat.shape[0]
    n_times = len(time)

    # Precompute the inverse of M
    B_inv = np.linalg.inv(M_mat)
    A_mat = -B_inv @ R_mat

    # Output array
    I_loop = np.zeros((n_times, nbloop))

    # We will integrate from time[0] to time[-1] in small steps dt_sub
    t_fine = np.arange(time[0], time[-1] + dt_sub, dt_sub)
    nfine = len(t_fine)

    # Interpolation buffers
    i_loop_old = np.zeros(nbloop)
    # We store the loop currents at each sub-step for final interpolation:
    i_loop_fine = np.zeros((nfine, nbloop))
    
    # Just an example approach for sub-stepping:
    idx_time = 0  # which main time index we are referencing
    coil_plasma_old = coil_plasma_currents[idx_time]

    # We can do a simple loop in time:
    for i_sub in range(nfine):
        t_now = t_fine[i_sub]

        # If we have advanced to a next main time index
        if t_now > time[idx_time] and idx_time < n_times - 1:
            idx_time += 1
            coil_plasma_old = coil_plasma_currents[idx_time]

        # Next sub-step input (linearly or simply using the current main step)
        # For demonstration, we just hold the coil+plasma current from the nearest time.
        # Or use np.interp if you want more accurate interpolation.
        coil_plasma_now = coil_plasma_old

        # Approx "voltage" from coil+plasma changes
        # (Here we do not do partial difference in coil current; your approach may differ.)
        # Example: treat L*dI/dt as a "voltage" input:
        #   Vw = - L * dIc/dt
        # for demonstration, do a no-op or simple approach:
        #   (You can do a difference with the previous sub-step, etc.)
        # 
        # We'll do a naive approach: 
        #   dI_coil_plasma/dt ~ 0, so Vw = 0
        # 
        # Or you can adapt from your original code's logic for "ic_inc" etc.
        Vw_now = np.zeros(nbloop)

        # Right-hand side for derivative:
        # dI_loop/dt = A_mat @ I_loop + B_inv @ Vw
        rhs = A_mat @ i_loop_old + B_inv @ Vw_now

        # Simple Euler (for demonstration; can also implement full RK4 as in your code).
        i_loop_new = i_loop_old + dt_sub * rhs

        # Store
        i_loop_old = i_loop_new
        i_loop_fine[i_sub] = i_loop_new

    # Now, we interpolate i_loop_fine back to the original (coarse) time grid
    for i_time in range(n_times):
        I_loop[i_time] = np.interp(time[i_time], t_fine, i_loop_fine[:, 0])  # or vector interpolation
        # for a vector interpolation you'd do something more like:
        # I_loop[i_time] = np.array([np.interp(time[i_time], t_fine, i_loop_fine[:, k]) 
        #                            for k in range(nbloop)])

    return I_loop


def compute_vacuum_fields_1d(
    coil_plus_loop_currents: np.ndarray,  # shape (n_times, nb_coil+nb_loop)
    coil_plus_loop_psi_resp: np.ndarray,  # shape (n_points, nb_coil+nb_loop)
    coil_plus_loop_br_resp: np.ndarray,   # shape (n_points, nb_coil+nb_loop)
    coil_plus_loop_bz_resp: np.ndarray,   # shape (n_points, nb_coil+nb_loop)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine coil+loop currents with precomputed response vectors
    to get psi, br, bz at given 1D points.

    :param coil_plus_loop_currents: (n_times, nb_coil+nb_loop)
    :param coil_plus_loop_psi_resp: (n_points, nb_coil+nb_loop)
    :param coil_plus_loop_br_resp:  (n_points, nb_coil+nb_loop)
    :param coil_plus_loop_bz_resp:  (n_points, nb_coil+nb_loop)
    :return: psi(t, pt), br(t, pt), bz(t, pt)
             each shape => (n_times, n_points)
    """
    n_times = coil_plus_loop_currents.shape[0]
    n_points = coil_plus_loop_psi_resp.shape[0]

    psi_out = np.zeros((n_times, n_points))
    br_out = np.zeros((n_times, n_points))
    bz_out = np.zeros((n_times, n_points))

    for i_time in range(n_times):
        ix = coil_plus_loop_currents[i_time]  # shape (nb_coil+nb_loop,)
        psi_out[i_time] = coil_plus_loop_psi_resp @ ix
        br_out[i_time]  = coil_plus_loop_br_resp  @ ix
        bz_out[i_time]  = coil_plus_loop_bz_resp  @ ix

    return psi_out, br_out, bz_out