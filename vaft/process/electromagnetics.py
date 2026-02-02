from vaft.formula import green_br_bz, green_r, calculate_distance
from typing import List, Dict, Any, Tuple
import numpy as np
from numpy import ndarray
# from scipy.linalg import expm # 행렬 지수 함수 - EVD 방법으로 대체

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not found. Falling back to slower Python execution for solve_eddy_currents. Install Numba for performance.")

# Description of the axisymmetric mutual electromagnetics calculations.
def compute_br_bz_phi(
    r_obs: np.ndarray,
    z_obs: np.ndarray,
    r_src: float,
    z_src: float,
    shift: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Br, Bz, and Phi using a shift approach to avoid singularities.
    Vectorized for observer points (r_obs, z_obs).

    :param r_obs: Array of radius coordinates of the observation points.
    :param z_obs: Array of Z coordinates of the observation points.
    :param r_src: Radius coordinate of the source element.
    :param z_src: Z coordinate of the source element.
    :param shift: Shift value to use if the points are too close.
    :return: (Br, Bz, Phi) arrays at (r_obs, z_obs).
    """
    distances = calculate_distance(r_obs, r_src, z_obs, z_src)
    
    condition = distances < (shift / 3.0)

    br1_shifted, bz1_shifted = green_br_bz(r_obs + shift, z_obs, r_src, z_src)
    br2_shifted, bz2_shifted = green_br_bz(r_obs - shift, z_obs, r_src, z_src)
    phi1_shifted = green_r(r_obs + shift, z_obs, r_src, z_src)
    phi2_shifted = green_r(r_obs - shift, z_obs, r_src, z_src)

    br_direct, bz_direct = green_br_bz(r_obs, z_obs, r_src, z_src)
    phi_direct = green_r(r_obs, z_obs, r_src, z_src)

    br_final = np.where(condition, (br1_shifted + br2_shifted) / 2.0, br_direct)
    bz_final = np.where(condition, (bz1_shifted + bz2_shifted) / 2.0, bz_direct)
    phi_final = np.where(condition, (phi1_shifted + phi2_shifted) / 2.0, phi_direct)
    
    return br_final, bz_final, phi_final

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
    plasma_points: List[List[float]] = None
    ) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Compute the Green's function response matrix (Psi, Bz, Br) at arbitrary observation points (not a fixed R,Z grid).

    Args:
        observation_points: List of [r, z] observation points (arbitrary, e.g., sensor/diagnostic locations)
        coil_data: List of dicts containing coil elements with fields:
            - elements: List of dicts with 'turns', 'r', 'z' for each element
        passive_loop_data: List of dicts containing loop data with fields:
            - geometry_type: 1 for outline, 2 for rectangle
            - outline_r, outline_z: Lists of coordinates for outline (type 1)
            - rectangle_r, rectangle_z: Single point for rectangle (type 2)
        plasma_points: List of [r, z] points for plasma elements (can be None, a single [r,z] point, or a list)
    
    Returns:
        Tuple of (Psi, Bz, Br) arrays, each of shape (len(observation_points), nb_coil+nb_loop+nb_plasma):
            - Psi_matrix: Magnetic flux response
            - Bz_matrix: Bz field response
            - Br_matrix: Br field response

    Note:
        This function is for general (r, z) points, not for a regular R,Z grid. For grid-based response, use a dedicated grid function.
    """
    nb_obs = len(observation_points)
    nb_coil = len(coil_data)
    nb_loop = len(passive_loop_data)
    
    # Handle plasma argument robustly (now also accepts empty list as 'no plasma')
    if plasma_points is None or (isinstance(plasma_points, (list, tuple)) and len(plasma_points) == 0):
        nb_plas = 0
        actual_plasma_points = []
    elif isinstance(plasma_points, (list, tuple)) and len(plasma_points) == 2 and all(isinstance(x, (float, int)) for x in plasma_points):
        nb_plas = 1
        actual_plasma_points = [plasma_points]
    elif isinstance(plasma_points, (list, tuple)) and len(plasma_points) > 0 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in plasma_points):
        nb_plas = len(plasma_points)
        actual_plasma_points = plasma_points
    else:
        raise ValueError("plasma_points must be None, a single [r, z] point, or a list of [r, z] points")

    total_sources = nb_coil + nb_loop + nb_plas

    Psi_matrix = np.zeros((nb_obs, total_sources))
    Bz_matrix = np.zeros((nb_obs, total_sources))
    Br_matrix = np.zeros((nb_obs, total_sources))

    for i_obs, (r1, z1) in enumerate(observation_points):
        # Active Coils contribution
        for i_c, coil in enumerate(coil_data):
            sum_psi_coil = 0.0
            sum_bz_coil = 0.0
            sum_br_coil = 0.0
            for element in coil['elements']:
                r2_c, z2_c, turns_c = element['r'], element['z'], element['turns']
                br, bz, psi = compute_br_bz_phi(r1, z1, r2_c, z2_c) # Uses the corrected version
                sum_psi_coil += psi * turns_c
                sum_bz_coil += bz * turns_c
                sum_br_coil += br * turns_c
            Psi_matrix[i_obs, i_c] = sum_psi_coil
            Bz_matrix[i_obs, i_c] = sum_bz_coil
            Br_matrix[i_obs, i_c] = sum_br_coil

        # Passive Loops contribution
        for i_l, loop in enumerate(passive_loop_data):
            if loop['geometry_type'] == 1: # Polygon (Outline)
                r2_l = np.mean(loop['outline_r'])
                z2_l = np.mean(loop['outline_z'])
            else: # Rectangle
                r2_l = loop['rectangle_r']
                z2_l = loop['rectangle_z']
            
            br, bz, psi = compute_br_bz_phi(r1, z1, r2_l, z2_l)
            Psi_matrix[i_obs, nb_coil + i_l] = psi
            Bz_matrix[i_obs, nb_coil + i_l] = bz
            Br_matrix[i_obs, nb_coil + i_l] = br

        # Plasma Elements contribution
        for i_p, (r2_p, z2_p) in enumerate(actual_plasma_points):
            br, bz, psi = compute_br_bz_phi(r1, z1, r2_p, z2_p)
            Psi_matrix[i_obs, nb_coil + nb_loop + i_p] = psi
            Bz_matrix[i_obs, nb_coil + nb_loop + i_p] = bz
            Br_matrix[i_obs, nb_coil + nb_loop + i_p] = br
            
    return Psi_matrix, Bz_matrix, Br_matrix

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

# _solve_eddy_currents_original = solve_eddy_currents # Keep a reference to the original, just in case

def solve_eddy_currents(
    R_mat: np.ndarray,    # (nbloop, nbloop)
    L_mat: np.ndarray,    # (nbloop, nbcoil+nbplas)
    M_mat: np.ndarray,    # (nbloop, nbloop)
    coil_plasma_currents: np.ndarray,  # (n_times, nbcoil+nbplas)
    time: np.ndarray,     # (n_times,)
    dt_sub: float = 5e-5
    ) -> np.ndarray:
    """
    Solve the RL circuit equation for vacuum vessel using EVD method.
    Optimized by pre-calculating active current derivatives.
    Maintains structural similarities to vfit_eddy for matrix setup.
    """
    nbloop = R_mat.shape[0]
    n_times_original = len(time)
    n_active_sources = coil_plasma_currents.shape[1]

    if nbloop == 0:
        return np.zeros((n_times_original, 0))
    if n_times_original == 0:
        return np.zeros((0, nbloop))

    # Matrix setup (similar to vfit_eddy, using direct inv with pseudo-inverse fallback)
    try:
        B_inv_M = np.linalg.inv(M_mat)
    except np.linalg.LinAlgError:
        # print(f"Error inverting M_mat: {e}. Using pseudo-inverse as fallback.")
        try:
            B_inv_M = np.linalg.pinv(M_mat)
        except np.linalg.LinAlgError as e_pinv:
            print(f"Pseudo-inverse of M_mat also failed: {e_pinv}. Aborting.")
            return np.full((n_times_original, nbloop), np.nan)

    A_sys = -B_inv_M @ R_mat

    try:
        C_R_inv = np.linalg.inv(R_mat)
    except np.linalg.LinAlgError:
        # print(f"Error inverting R_mat: {e}. Using pseudo-inverse as fallback.")
        try:
            C_R_inv = np.linalg.pinv(R_mat)
        except np.linalg.LinAlgError as e_pinv:
            print(f"Pseudo-inverse of R_mat also failed: {e_pinv}. Aborting.")
            return np.full((n_times_original, nbloop), np.nan)
            
    if np.any(np.isnan(B_inv_M)) or np.any(np.isinf(B_inv_M)) or \
       np.any(np.isnan(C_R_inv)) or np.any(np.isinf(C_R_inv)):
        print("Error: Inverse of M_mat or R_mat contains NaN/Inf after fallback. Aborting.")
        return np.full((n_times_original, nbloop), np.nan)

    # Eigenvalue decomposition of A_sys
    # print("Performing eigenvalue decomposition of A_sys...")
    try:
        eigenvalues_w, E_vec = np.linalg.eig(A_sys)
        E_inv = np.linalg.inv(E_vec)
    except np.linalg.LinAlgError as e:
        print(f"Error during eigenvalue decomposition or E_vec inversion: {e}. Aborting.")
        return np.full((n_times_original, nbloop), np.nan)
    # print("Eigenvalue decomposition and E_vec inversion successful.")

    # Calculate RLR = E @ diag(exp(w*dt)) @ Einv (state transition matrix)
    F_diag_exp = np.diag(np.exp(eigenvalues_w * dt_sub))
    RLR_mat = E_vec @ F_diag_exp @ E_inv
    
    if np.isrealobj(A_sys) and not np.isrealobj(RLR_mat):
        # print("RLR_mat was complex, taking real part. Max imaginary part: ", np.max(np.abs(np.imag(RLR_mat))))
        RLR_mat = np.real(RLR_mat)

    if np.any(np.isnan(RLR_mat)) or np.any(np.isinf(RLR_mat)):
        print("Error: RLR_mat contains NaN/Inf. Aborting.")
        return np.full((n_times_original, nbloop), np.nan)

    # Fine time grid
    if n_times_original == 0: 
        return np.zeros((0, nbloop))
    if time.size == 1 or np.isclose(time[0], time[-1]):
        t_fine = np.array([time[0]])
    elif time[0] > time[-1]: 
        t_fine = np.array([]) 
    else:
        t_fine = np.arange(time[0], time[-1], dt_sub) 
        if t_fine.size == 0 and not np.isclose(time[0],time[-1]): 
             t_fine = np.array([time[0]])

    n_fine_steps = len(t_fine)
    if n_fine_steps == 0:
        return np.zeros((n_times_original, nbloop)) 

    # --- Pre-calculate interpolated active currents and their derivatives ---
    coil_plasma_currents_fine = np.zeros((n_fine_steps, n_active_sources))
    if n_active_sources > 0:
        if n_times_original > 1:
            for i_src in range(n_active_sources):
                coil_plasma_currents_fine[:, i_src] = np.interp(t_fine, time, coil_plasma_currents[:, i_src])
        elif n_times_original == 1 and n_fine_steps > 0: 
            for i_src in range(n_active_sources):
                coil_plasma_currents_fine[:, i_src] = coil_plasma_currents[0, i_src]

    d_coil_plasma_dt_fine = np.zeros_like(coil_plasma_currents_fine)
    if n_fine_steps > 1 and n_active_sources > 0:
        diff_currents = np.diff(coil_plasma_currents_fine, axis=0)
        d_coil_plasma_dt_fine[:-1, :] = diff_currents / dt_sub
        d_coil_plasma_dt_fine[-1, :] = d_coil_plasma_dt_fine[-2, :] 
    elif n_fine_steps == 1 and n_active_sources > 0: 
        d_coil_plasma_dt_fine[:, :] = 0.0 
    # --- End of pre-calculation ---

    i_loop_old = np.zeros(nbloop) 
    i_loop_fine_out = np.zeros((n_fine_steps, nbloop))
    if n_fine_steps > 0:
        i_loop_fine_out[0, :] = i_loop_old 

    # print("Starting EVD time integration loop (Optimized - pre-calculated derivatives)...")
    
    for i_sub in range(n_fine_steps - 1): 
        # if i_sub % 10000 == 0: # Progress indicator can be re-enabled if needed for long runs
        #     print(f"EVD loop: iteration {i_sub} / {n_fine_steps - 1}")

        current_dIc_dt = d_coil_plasma_dt_fine[i_sub, :]
        
        Vw_source_term = -L_mat @ current_dIc_dt
        I_particular = C_R_inv @ Vw_source_term
        
        i_loop_new = I_particular + RLR_mat @ (i_loop_old - I_particular)
            
        i_loop_old = i_loop_new.copy()
        i_loop_fine_out[i_sub+1, :] = i_loop_new 

    # Interpolate results back to original time grid
    I_loop_final = np.zeros((n_times_original, nbloop))
    if n_times_original > 0: 
        if n_fine_steps == 0: 
            pass 
        elif n_fine_steps == 1: 
            for i_l in range(nbloop):
                I_loop_final[:, i_l] = i_loop_fine_out[0, i_l] 
        else: 
            for i_l in range(nbloop):
                if time.size > 1 and np.isclose(t_fine[0], t_fine[-1]) and t_fine.size > 1:
                    I_loop_final[:,i_l] = i_loop_fine_out[0,i_l]
                else:
                    I_loop_final[:, i_l] = np.interp(time, t_fine, i_loop_fine_out[:, i_l])
    
    # print("EVD method (Optimized) finished.")
    return I_loop_final

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