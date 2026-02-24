from typing import List, Tuple, Dict, Any, Optional
from numpy import ndarray
import numpy as np
from omas import *
from vaft.process import (
    compute_br_bz_phi,
    compute_response_matrix,
    compute_impedance_matrices,
    solve_eddy_currents,
    compute_vacuum_fields_1d,
    time_derivative,
    psi_to_RZ,
    volume_average,
    poloidal_field_at_boundary,
    calculate_average_boundary_poloidal_field,
    shafranov_integrals,
    calculate_reconstructed_diamagnetic_flux,
    calculate_diamagnetism,
)
from vaft.formula import (
    spitzer_resistivity_from_T_e_Z_eff_ln_Lambda,
    approximated_diamagnetism_from_B_pa_B_tv_R0_delta_phi,
    virial_beta_p_from_S_alpha_mu,
    virial_li_from_S_alpha_mu,
    kinetic_energy_from_beta_p_B_pa_V_p,
    magnetic_energy_from_li_B_pa_V_p,
)
from vaft.omas import find_matching_time_indices
from vaft.omas.update import update_equilibrium_boundary
from scipy.interpolate import interp1d
import logging
import vaft.process
import matplotlib.pyplot as plt
from matplotlib.path import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback diamagnetic flux [Wb] when magnetics.diamagnetic_flux is not available
default_delta_phi = 2.0 * np.pi

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

def compute_point_response_ods(
    ods: ODS,
    rz: List[List[float]],
    plasma: List[List[float]] = None
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

def compute_grid_response_ods(
    ods: ODS,
    plasma: List[List[float]] = None
    ) -> ndarray:
    """Compute Green's function response matrix (Psi) on equilibrium 2D grid.
    
    Args:
        ods: OMAS data structure with PF coil and loop data
        plasma: Optional list of [r, z] plasma element points
    
    Returns:
        ndarray: 2D response matrix mapping coil/wall/plasma -> grid

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

def compute_core_profile_psi(
    ods: ODS,
    option: str = 'n_e',
    time_slice: Optional[int] = None
    ) -> Tuple[ndarray, ndarray, ndarray]:
    """Compute core profile in psi_norm coordinate system.
    
    Args:
        ods: OMAS data structure
        option: Profile option ('n_e', 't_e', 'n_i', 't_i')
        time_slice: Time slice index (None = use first available)
    
    Returns:
        Tuple of (psi_norm, profile_1d, time_value)
    """
    from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi
    
    # Basic availability checks
    if 'core_profiles.profiles_1d' not in ods:
        raise KeyError("core_profiles.profiles_1d not found in ODS")
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")
    
    # Determine time slice
    if time_slice is None:
        cp_idx = 0
    else:
        cp_idx = time_slice if time_slice < len(ods['core_profiles.profiles_1d']) else 0
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    
    # Get time
    if 'time' in cp_ts:
        cp_time = float(cp_ts['time'])
    elif 'core_profiles.time' in ods and cp_idx < len(ods['core_profiles.time']):
        cp_time = float(ods['core_profiles.time'][cp_idx])
    else:
        cp_time = float(cp_idx)
    
    # Find matching equilibrium time slice
    equil_times = []
    for idx in range(len(ods['equilibrium.time_slice'])):
        eq_ts = ods['equilibrium.time_slice'][idx]
        if 'time' in eq_ts:
            equil_times.append(float(eq_ts['time']))
        elif 'equilibrium.time' in ods and idx < len(ods['equilibrium.time']):
            equil_times.append(float(ods['equilibrium.time'][idx]))
        else:
            equil_times.append(float(idx))
    
    equil_times = np.asarray(equil_times)
    equil_idx = np.argmin(np.abs(equil_times - cp_time))
    eq_ts = ods['equilibrium.time_slice'][equil_idx]
    
    # Get core profile data
    grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
    if 'rho_tor_norm' not in grid:
        raise KeyError(f"rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}]")
    
    rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
    
    # Get profile data based on option
    if option == 'n_e':
        if 'electrons.density' not in cp_ts:
            raise KeyError(f"electrons.density missing in core_profiles.profiles_1d[{cp_idx}]")
        profile_1d_rho = np.asarray(cp_ts['electrons.density'], float)
    elif option == 't_e':
        if 'electrons.temperature' not in cp_ts:
            raise KeyError(f"electrons.temperature missing in core_profiles.profiles_1d[{cp_idx}]")
        profile_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
    elif option == 'n_i':
        if 'ion' not in cp_ts or len(cp_ts['ion']) == 0:
            raise KeyError(f"ion array missing in core_profiles.profiles_1d[{cp_idx}]")
        # Sum all ion densities
        profile_1d_rho = np.zeros_like(rho_tor_norm_cp)
        for ion_ts in cp_ts['ion']:
            # Handle case where cp_ts['ion'] is a dictionary (OMAS arrays are often dicts)
            # In that case, iterating over it gives keys (int), not values
            if isinstance(ion_ts, (int, np.integer)):
                ion_ts = cp_ts['ion'][ion_ts]
            if 'density' in ion_ts:
                profile_1d_rho += np.asarray(ion_ts['density'], float)
    elif option == 't_i':
        if 'ion' not in cp_ts or len(cp_ts['ion']) == 0:
            raise KeyError(f"ion array missing in core_profiles.profiles_1d[{cp_idx}]")
        # Density-weighted ion temperature
        n_i_total = np.zeros_like(rho_tor_norm_cp)
        nT_i_total = np.zeros_like(rho_tor_norm_cp)
        for ion_ts in cp_ts['ion']:
            # Handle case where cp_ts['ion'] is a dictionary (OMAS arrays are often dicts)
            # In that case, iterating over it gives keys (int), not values
            if isinstance(ion_ts, (int, np.integer)):
                ion_ts = cp_ts['ion'][ion_ts]
            if 'density' in ion_ts and 'temperature' in ion_ts:
                n_i = np.asarray(ion_ts['density'], float)
                T_i = np.asarray(ion_ts['temperature'], float)
                n_i_total += n_i
                nT_i_total += n_i * T_i
        profile_1d_rho = nT_i_total / n_i_total if np.any(n_i_total > 0) else np.zeros_like(rho_tor_norm_cp)
    else:
        raise ValueError(f"Invalid option: {option}. Must be one of: 'n_e', 't_e', 'n_i', 't_i'")
    
    # Get equilibrium profiles_1d for coordinate conversion
    eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
        if 'psi_norm' not in eq_profiles_1d:
            raise KeyError(f"Failed to create psi_norm for equilibrium.time_slice[{equil_idx}]")
    
    if 'rho_tor_norm' not in eq_profiles_1d:
        raise KeyError(f"rho_tor_norm missing in equilibrium.profiles_1d for time_slice[{equil_idx}]")
    
    rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
    psi_norm_eq = np.asarray(eq_profiles_1d['psi_norm'], float)
    
    # Ensure monotonicity
    if not np.all(np.diff(rho_tor_norm_eq) > 0):
        sort_idx = np.argsort(rho_tor_norm_eq)
        rho_tor_norm_eq_sorted = rho_tor_norm_eq[sort_idx]
        psi_norm_eq_sorted = psi_norm_eq[sort_idx]
        unique_mask = np.concatenate(([True], np.diff(rho_tor_norm_eq_sorted) > 1e-10))
        rho_tor_norm_eq_sorted = rho_tor_norm_eq_sorted[unique_mask]
        psi_norm_eq_sorted = psi_norm_eq_sorted[unique_mask]
    else:
        rho_tor_norm_eq_sorted = rho_tor_norm_eq
        psi_norm_eq_sorted = psi_norm_eq
    
    # Use equilibrium psi_norm grid as target coordinate system
    psiN_1d = psi_norm_eq_sorted
    
    # Create inverse mapping: psi_norm -> rho_tor_norm
    interp_psi_to_rho = interp1d(psi_norm_eq_sorted, rho_tor_norm_eq_sorted,
                                 kind='linear',
                                 bounds_error=False,
                                 fill_value=(rho_tor_norm_eq_sorted[0], rho_tor_norm_eq_sorted[-1]))
    rho_tor_norm_at_psiN = interp_psi_to_rho(psiN_1d)
    
    # Interpolate profile to psi_norm coordinate
    interp_func = interp1d(rho_tor_norm_cp, profile_1d_rho,
                          kind='linear',
                          bounds_error=False,
                          fill_value=(profile_1d_rho[0], profile_1d_rho[-1]))
    profile_1d = interp_func(rho_tor_norm_at_psiN)
    
    return psiN_1d, profile_1d, cp_time

def compute_core_profile_2d(
    ods: ODS,
    option: str = 'n_e',
    time_slice: Optional[int] = None
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, float]:
    """Compute core profile in 2D (R,Z) coordinate system.
    
    Args:
        ods: OMAS data structure
        option: Profile option ('n_e', 't_e', 'n_i', 't_i')
        time_slice: Time slice index (None = use first available)
    
    Returns:
        Tuple of (profile_RZ, R_grid, Z_grid, psiN_RZ, time_value)
    """
    from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi
    
    # Basic availability checks
    if 'core_profiles.profiles_1d' not in ods:
        raise KeyError("core_profiles.profiles_1d not found in ODS")
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")
    
    # Determine time slice
    if time_slice is None:
        cp_idx = 0
    else:
        cp_idx = time_slice if time_slice < len(ods['core_profiles.profiles_1d']) else 0
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    
    # Get time
    if 'time' in cp_ts:
        cp_time = float(cp_ts['time'])
    elif 'core_profiles.time' in ods and cp_idx < len(ods['core_profiles.time']):
        cp_time = float(ods['core_profiles.time'][cp_idx])
    else:
        cp_time = float(cp_idx)
    
    # Find matching equilibrium time slice
    equil_times = []
    for idx in range(len(ods['equilibrium.time_slice'])):
        eq_ts = ods['equilibrium.time_slice'][idx]
        if 'time' in eq_ts:
            equil_times.append(float(eq_ts['time']))
        elif 'equilibrium.time' in ods and idx < len(ods['equilibrium.time']):
            equil_times.append(float(ods['equilibrium.time'][idx]))
        else:
            equil_times.append(float(idx))
    
    equil_times = np.asarray(equil_times)
    equil_idx = np.argmin(np.abs(equil_times - cp_time))
    eq_ts = ods['equilibrium.time_slice'][equil_idx]
    
    # Get core profile data
    grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
    if 'rho_tor_norm' not in grid:
        raise KeyError(f"rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}]")
    
    rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
    
    # Get profile data based on option
    if option == 'n_e':
        if 'electrons.density' not in cp_ts:
            raise KeyError(f"electrons.density missing in core_profiles.profiles_1d[{cp_idx}]")
        profile_1d_rho = np.asarray(cp_ts['electrons.density'], float)
    elif option == 't_e':
        if 'electrons.temperature' not in cp_ts:
            raise KeyError(f"electrons.temperature missing in core_profiles.profiles_1d[{cp_idx}]")
        profile_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
    elif option == 'n_i':
        if 'ion' not in cp_ts or len(cp_ts['ion']) == 0:
            raise KeyError(f"ion array missing in core_profiles.profiles_1d[{cp_idx}]")
        # Sum all ion densities
        profile_1d_rho = np.zeros_like(rho_tor_norm_cp)
        for ion_ts in cp_ts['ion']:
            # Handle case where cp_ts['ion'] is a dictionary (OMAS arrays are often dicts)
            # In that case, iterating over it gives keys (int), not values
            if isinstance(ion_ts, (int, np.integer)):
                ion_ts = cp_ts['ion'][ion_ts]
            if 'density' in ion_ts:
                profile_1d_rho += np.asarray(ion_ts['density'], float)
    elif option == 't_i':
        if 'ion' not in cp_ts or len(cp_ts['ion']) == 0:
            raise KeyError(f"ion array missing in core_profiles.profiles_1d[{cp_idx}]")
        # Density-weighted ion temperature
        n_i_total = np.zeros_like(rho_tor_norm_cp)
        nT_i_total = np.zeros_like(rho_tor_norm_cp)
        for ion_ts in cp_ts['ion']:
            # Handle case where cp_ts['ion'] is a dictionary (OMAS arrays are often dicts)
            # In that case, iterating over it gives keys (int), not values
            if isinstance(ion_ts, (int, np.integer)):
                ion_ts = cp_ts['ion'][ion_ts]
            if 'density' in ion_ts and 'temperature' in ion_ts:
                n_i = np.asarray(ion_ts['density'], float)
                T_i = np.asarray(ion_ts['temperature'], float)
                n_i_total += n_i
                nT_i_total += n_i * T_i
        profile_1d_rho = nT_i_total / n_i_total if np.any(n_i_total > 0) else np.zeros_like(rho_tor_norm_cp)
    else:
        raise ValueError(f"Invalid option: {option}. Must be one of: 'n_e', 't_e', 'n_i', 't_i'")
    
    # Get equilibrium profiles_1d for coordinate conversion
    eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
        if 'psi_norm' not in eq_profiles_1d:
            raise KeyError(f"Failed to create psi_norm for equilibrium.time_slice[{equil_idx}]")
    
    if 'rho_tor_norm' not in eq_profiles_1d:
        raise KeyError(f"rho_tor_norm missing in equilibrium.profiles_1d for time_slice[{equil_idx}]")
    
    rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
    psi_norm_eq = np.asarray(eq_profiles_1d['psi_norm'], float)
    
    # Ensure monotonicity
    if not np.all(np.diff(rho_tor_norm_eq) > 0):
        sort_idx = np.argsort(rho_tor_norm_eq)
        rho_tor_norm_eq_sorted = rho_tor_norm_eq[sort_idx]
        psi_norm_eq_sorted = psi_norm_eq[sort_idx]
        unique_mask = np.concatenate(([True], np.diff(rho_tor_norm_eq_sorted) > 1e-10))
        rho_tor_norm_eq_sorted = rho_tor_norm_eq_sorted[unique_mask]
        psi_norm_eq_sorted = psi_norm_eq_sorted[unique_mask]
    else:
        rho_tor_norm_eq_sorted = rho_tor_norm_eq
        psi_norm_eq_sorted = psi_norm_eq
    
    # Use equilibrium psi_norm grid as target coordinate system
    psiN_1d = psi_norm_eq_sorted
    
    # Create inverse mapping: psi_norm -> rho_tor_norm
    interp_psi_to_rho = interp1d(psi_norm_eq_sorted, rho_tor_norm_eq_sorted,
                                 kind='linear',
                                 bounds_error=False,
                                 fill_value=(rho_tor_norm_eq_sorted[0], rho_tor_norm_eq_sorted[-1]))
    rho_tor_norm_at_psiN = interp_psi_to_rho(psiN_1d)
    
    # Interpolate profile to psi_norm coordinate
    interp_func = interp1d(rho_tor_norm_cp, profile_1d_rho,
                          kind='linear',
                          bounds_error=False,
                          fill_value=(profile_1d_rho[0], profile_1d_rho[-1]))
    profile_1d = interp_func(rho_tor_norm_at_psiN)
    
    # Get equilibrium 2D grid and ψ(R,Z)
    R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
    Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
    psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
    
    # Ensure psi_RZ shape convention is (len(R), len(Z))
    # When R_grid and Z_grid have the same length, use physical properties:
    # R_grid: major radius, typically all positive (R >= 0)
    # Z_grid: vertical position, typically has negative values (up-down symmetry)
    if psi_RZ.shape != (len(R_grid), len(Z_grid)):
        if psi_RZ.shape == (len(Z_grid), len(R_grid)):
            psi_RZ = psi_RZ.T
        else:
            raise ValueError(
                f"Unexpected psi_RZ shape {psi_RZ.shape}; expected {(len(R_grid), len(Z_grid))} or {(len(Z_grid), len(R_grid))}"
            )
    
    psi_axis = float(eq_ts['global_quantities.psi_axis'])
    psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
    
    # Map to 2D (R,Z)
    profile_RZ, psiN_RZ = psi_to_RZ(psiN_1d, profile_1d, psi_RZ, psi_axis, psi_lcfs)
    
    return profile_RZ, R_grid, Z_grid, psiN_RZ, cp_time

def compute_magnetic_energy(ods: ODS, time_slice: Optional[int] = None) -> float:
    """Compute magnetic energy from ODS.
    
    Args:
        ods: OMAS data structure
        time_slice: Time slice index (None = use first available)
    
    Returns:
        float: Magnetic energy [J]
    """
    from vaft.formula.constants import MU0

    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")

    eq_idx = 0 if time_slice is None else int(time_slice)
    if eq_idx >= len(ods['equilibrium.time_slice']):
        raise IndexError(f"time_slice {eq_idx} is out of bounds for equilibrium.time_slice")

    eq_ts = ods['equilibrium.time_slice'][eq_idx]

    # Required equilibrium 2D psi grid
    try:
        R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
        Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
        psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
        psi_axis = float(eq_ts['global_quantities.psi_axis'])
        psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
    except KeyError as e:
        raise KeyError(f"Missing equilibrium keys for magnetic energy: {e}")

    # Reference toroidal field (B_phi) & reference radius for F = R * B_phi
    # Prefer equilibrium global_quantities (produced by EFIT mapping in this repo)
    try:
        B0 = float(eq_ts['global_quantities.b0'])  # [T]
    except Exception:
        # Fallback to vacuum_toroidal_field.b0 if present (time-dependent array)
        if 'equilibrium.vacuum_toroidal_field.b0' in ods:
            b0_arr = np.asarray(ods['equilibrium.vacuum_toroidal_field.b0'], float)
            # If time array exists, use closest by index; else take first
            B0 = float(b0_arr[eq_idx]) if b0_arr.size > eq_idx else float(b0_arr.flat[0])
        else:
            raise KeyError("Missing reference toroidal field: equilibrium.time_slice[*].global_quantities.b0")

    try:
        R0 = float(eq_ts['global_quantities.major_radius'])  # [m]
    except Exception:
        if 'equilibrium.vacuum_toroidal_field.r0' in ods:
            R0 = float(np.asarray(ods['equilibrium.vacuum_toroidal_field.r0'], float).flat[0])
        else:
            raise KeyError("Missing reference radius for toroidal field (major_radius or vacuum_toroidal_field.r0)")

    # Ip is not needed for B from psi, but user requested to load it (sanity / completeness)
    Ip = None
    try:
        Ip = float(eq_ts['global_quantities.ip'])
    except Exception:
        pass

    # Ensure psi_RZ shape convention is (len(R), len(Z))
    # When R_grid and Z_grid have the same length, use physical properties:
    # R_grid: major radius, typically all positive (R >= 0)
    # Z_grid: vertical position, typically has negative values (up-down symmetry)
    if psi_RZ.shape != (len(R_grid), len(Z_grid)):
        if psi_RZ.shape == (len(Z_grid), len(R_grid)):
            psi_RZ = psi_RZ.T
        else:
            raise ValueError(
                f"Unexpected psi_RZ shape {psi_RZ.shape}; expected {(len(R_grid), len(Z_grid))} or {(len(Z_grid), len(R_grid))}"
            )

    # Gradients: dpsi/dR and dpsi/dZ on (R,Z) grid
    dpsi_dR, dpsi_dZ = np.gradient(psi_RZ, R_grid, Z_grid, edge_order=2)

    # Build mesh for B components
    Rm, Zm = np.meshgrid(R_grid, Z_grid, indexing="ij")
    Rm_safe = np.where(Rm == 0.0, np.nan, Rm)

    # B field from poloidal flux psi
    # B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR
    B_R = -(1.0 / Rm_safe) * dpsi_dZ
    B_Z = (1.0 / Rm_safe) * dpsi_dR

    # Toroidal field: B_phi = F(psi) / R.
    # Here we approximate F as constant using reference point: F ≈ B0 * R0
    F_ref = B0 * R0
    B_PHI = F_ref / Rm_safe

    # Total B^2
    B2 = B_R**2 + B_Z**2 + B_PHI**2

    # Normalize psi for plasma mask via volume_average()
    psiN_RZ = (psi_RZ - psi_axis) / (psi_lcfs - psi_axis)

    # Magnetic energy density and volume integral (use provided volume_average)
    w_mag_RZ = B2 / (2.0 * MU0)  # [J/m^3]
    w_avg, V = volume_average(w_mag_RZ, psiN_RZ, R_grid, Z_grid)
    W_B = float(w_avg * V)  # [J]

    # Store computed fields back into ODS (optional but useful for diagnostics/plotting)
    eq_ts['profiles_2d.0.b_field_r'] = B_R
    eq_ts['profiles_2d.0.b_field_z'] = B_Z
    eq_ts['profiles_2d.0.b_field_tor'] = B_PHI

    return W_B


def compute_virial_equilibrium_quantities_ods(
    ods: ODS,
    time_slice: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Shafranov/virial equilibrium quantities from an arbitrary equilibrium.

    Uses equilibrium 2D psi, boundary outline, and
    vacuum toroidal field to compute:
    - Shafranov integrals S1, S2, S3 and alpha
    - Average boundary poloidal field B_pa
    - Approximated diamagnetism μ̂_i, then virial beta_p and l_i
    - Kinetic and magnetic energies W_kin, W_mag

    Args:
        ods: OMAS data structure with equilibrium.time_slice (with boundary.outline,
             profiles_2d.0.psi and grid; optional profiles_2d.0.b_field_r/z and
             global_quantities.b0, major_radius or magnetic_axis).
             For μ̂_i, delta_phi uses measured diamagnetic flux when present:
             magnetics.diamagnetic_flux.0.data interpolated at equilibrium time
        time_slice: Time slice index (None = all slices).

    Returns:
        Dict mapping time_slice index -> dict of computed quantities (s_1, s_2, s_3,
        alpha, B_pa, beta_p, li, W_mag, W_kin, V_p, mui_hat).

    Raises:
        KeyError: If required equilibrium or boundary data is missing.
    """
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        raise KeyError("equilibrium.time_slice not found in ODS")

    # Measured diamagnetic flux (magnetics.diamagnetic_flux) interpolated at each equilibrium time
    delta_phi_interp = None
    if "magnetics.diamagnetic_flux.0.data" in ods and "magnetics.time" in ods and len(ods["magnetics.diamagnetic_flux"]) > 0:
        t_mag = np.asarray(ods["magnetics.time"], float)
        flux_mag = np.asarray(ods["magnetics.diamagnetic_flux.0.data"], float)
        if t_mag.size >= 2 and flux_mag.size == t_mag.size:
            delta_phi_interp = interp1d(
                t_mag, flux_mag,
                kind="linear",
                bounds_error=False,
                fill_value=(flux_mag[0], flux_mag[-1]),
            )
    else:
        raise KeyError("Missing magnetics.diamagnetic_flux.0.data or magnetics.time")

    slices_to_process = (
        list(range(len(ods["equilibrium.time_slice"])))
        if time_slice is None
        else [int(time_slice)]
    )
    if time_slice is not None and (
        time_slice < 0 or time_slice >= len(ods["equilibrium.time_slice"])
    ):
        raise IndexError(
            f"time_slice {time_slice} is out of bounds for equilibrium.time_slice"
        )
    # Ensure boundary outline is existed
    if "boundary.geometric_axis" not in ods["equilibrium.time_slice"][0]:
        update_equilibrium_boundary(ods)

    out = {}
    for eq_idx in slices_to_process:
        eq_ts = ods["equilibrium.time_slice"][eq_idx]
        try:
            R_grid_1d = np.asarray(eq_ts["profiles_2d.0.grid.dim1"], float)
            Z_grid_1d = np.asarray(eq_ts["profiles_2d.0.grid.dim2"], float)
            psi_RZ = np.asarray(eq_ts["profiles_2d.0.psi"], float)
        except KeyError as e:
            raise KeyError(f"Missing equilibrium 2D grid/psi for time_slice {eq_idx}: {e}") from e

        # Ensure psi_RZ shape (nR, nZ) so RectBivariateSpline and shafranov mask align with grid
        nR, nZ = len(R_grid_1d), len(Z_grid_1d)
        if psi_RZ.shape != (nR, nZ):
            if psi_RZ.shape == (nZ, nR):
                psi_RZ = psi_RZ.T
            else:
                raise ValueError(
                    f"psi shape {psi_RZ.shape} does not match grid (nR={nR}, nZ={nZ})"
                )

        # Boundary outline
        R_bdry = np.asarray(eq_ts["boundary.outline.r"], float)
        Z_bdry = np.asarray(eq_ts["boundary.outline.z"], float)
        if R_bdry.size == 0 or Z_bdry.size == 0:
            logger.warning(
                "Time slice %s: empty boundary.outline, skipping virial computation", eq_idx
            )
            out[eq_idx] = {
                "s_1": np.nan, "s_2": np.nan, "s_3": np.nan, "alpha": np.nan,
                "B_pa": np.nan, "beta_p": np.nan, "li": np.nan,
                "W_mag": np.nan, "W_kin": np.nan, "V_p": np.nan, "mui_hat": np.nan,
            }
            nans = [k for k in ("s_1", "s_2", "s_3", "alpha", "B_pa", "beta_p", "li", "W_mag", "W_kin")
                     if not np.isfinite(np.asarray(out[eq_idx][k], float))]
            if nans:
                logger.warning("Time slice %s: virial quantities are NaN or non-finite: %s", eq_idx, nans)
            continue

        B_p_bdry, _, _ = poloidal_field_at_boundary(
            R_grid_1d, Z_grid_1d, psi_RZ, R_bdry, Z_bdry
        )
        B_pa = float(calculate_average_boundary_poloidal_field(R_bdry, Z_bdry, B_p_bdry))

        use_ods_bfield = (
            "profiles_2d.0.b_field_r" in eq_ts and "profiles_2d.0.b_field_z" in eq_ts
        )
        if use_ods_bfield:
            B_R_grid = np.asarray(eq_ts["profiles_2d.0.b_field_r"], float)
            B_Z_grid = np.asarray(eq_ts["profiles_2d.0.b_field_z"], float)
            if B_R_grid.shape == (nZ, nR):
                B_R_grid = B_R_grid.T
                B_Z_grid = B_Z_grid.T
            elif B_R_grid.shape != (nR, nZ):
                use_ods_bfield = False
        if not use_ods_bfield:
            dpsi_dR, dpsi_dZ = np.gradient(psi_RZ, R_grid_1d, Z_grid_1d, edge_order=2)
            Rm, Zm = np.meshgrid(R_grid_1d, Z_grid_1d, indexing="ij")
            Rm_safe = np.where(Rm == 0.0, np.nan, Rm)
            B_R_grid = -(1.0 / Rm_safe) * dpsi_dZ
            B_Z_grid = (1.0 / Rm_safe) * dpsi_dR


        R_mesh, Z_mesh = np.meshgrid(R_grid_1d, Z_grid_1d, indexing="ij")

        S1, S2, S3, alpha = shafranov_integrals(
            R_bdry, Z_bdry, B_p_bdry,
            R_mesh, Z_mesh, B_R_grid, B_Z_grid)
        S1, S2, S3, alpha = float(S1), float(S2), float(S3), float(alpha)

        # Geometric axis
        R_0 = float(eq_ts["boundary.geometric_axis.r"])
        Z_0 = float(eq_ts["boundary.geometric_axis.z"])
        R_bdry_c = np.append(R_bdry, R_bdry[0]) if (R_bdry[0] != R_bdry[-1] or Z_bdry[0] != Z_bdry[-1]) else R_bdry
        Z_bdry_c = np.append(Z_bdry, Z_bdry[0]) if (R_bdry[0] != R_bdry[-1] or Z_bdry[0] != Z_bdry[-1]) else Z_bdry
        dR_b = np.diff(R_bdry_c)
        dZ_b = np.diff(Z_bdry_c)
        R_mid_b = 0.5 * (R_bdry_c[:-1] + R_bdry_c[1:])
        V_p = float(np.abs(-np.sum(np.pi * (R_mid_b**2) * dZ_b)))

        # delta_phi: measured diamagnetic flux at this time, or 2*pi if not available
        t_eq = float(eq_ts["time"])
        delta_phi = abs(float(delta_phi_interp(t_eq))) if delta_phi_interp is not None else default_delta_phi

        # Vacuum toroidal field at geometric axis from magnetic_axis
        B_t_axis = float(eq_ts['global_quantities.magnetic_axis.b_field_tor'])
        R_axis = float(eq_ts['global_quantities.magnetic_axis.r'])
        B_tv = abs(B_t_axis * R_axis / R_0) # [T]

        mui_hat = np.nan
        if np.isfinite(B_pa) and B_pa > 0 and np.isfinite(V_p) and V_p > 0 and np.isfinite(B_tv):
            mui_hat = float(
                approximated_diamagnetism_from_B_pa_B_tv_R0_delta_phi(
                    B_pa, B_tv, R_0, delta_phi, V_p
                )
            )

        den_beta = 3.0 * (alpha - 1.0) + 1.0
        den_li = 3.0 * alpha - 2.0
        if np.isfinite(mui_hat) and abs(den_beta) > 1e-12:
            beta_p = float(virial_beta_p_from_S_alpha_mu(S1, S2, S3, alpha, mui_hat))
        else:
            beta_p = np.nan
        if np.isfinite(mui_hat) and abs(den_li) > 1e-12:
            li = float(virial_li_from_S_alpha_mu(S1, S2, S3, alpha, mui_hat))
        else:
            li = np.nan

        if np.isfinite(beta_p) and np.isfinite(B_pa) and np.isfinite(V_p):
            W_kin = float(kinetic_energy_from_beta_p_B_pa_V_p(beta_p, B_pa, V_p))
        else:
            W_kin = np.nan
        if np.isfinite(li) and np.isfinite(B_pa) and np.isfinite(V_p):
            W_mag = float(magnetic_energy_from_li_B_pa_V_p(li, B_pa, V_p))
        else:
            W_mag = np.nan

        out[eq_idx] = {
            "s_1": S1, "s_2": S2, "s_3": S3, "alpha": alpha,
            "B_pa": B_pa, "beta_p": beta_p, "li": li,
            "W_mag": W_mag, "W_kin": W_kin, "V_p": V_p, "mui_hat": mui_hat,
        }
        nans = [k for k in ("s_1", "s_2", "s_3", "alpha", "B_pa", "beta_p", "li", "W_mag", "W_kin")
                 if not np.isfinite(np.asarray(out[eq_idx][k], float))]
        if nans:
            logger.warning("Time slice %s: virial quantities are NaN or non-finite: %s", eq_idx, nans)
    return out


def compute_reconstructed_diamagnetic_flux(ods, time_index=0):
    """
    Compute reconstructed diamagnetic flux (CDFLUX) from ODS.

    Loads equilibrium data from ODS and calls
    :func:`vaft.process.equilibrium.calculate_reconstructed_diamagnetic_flux`
    with physical quantities only. Formula: Phi_dia = Integral_surf
    (B_phi_plasma - B_phi_vacuum) dA [Wb]. Returns negative for diamagnetic plasma.
    """
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        raise KeyError("equilibrium.time_slice not found in ODS")
    if time_index >= len(ods["equilibrium.time_slice"]):
        raise IndexError(
            f"time_index {time_index} is out of range for equilibrium.time_slice"
        )

    eq_slice = ods["equilibrium.time_slice"][time_index]

    def _ensure_rz_shape(arr: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, float)
        if arr.shape == (len(R), len(Z)):
            return arr
        if arr.shape == (len(Z), len(R)):
            return arr.T
        raise ValueError(
            f"Unexpected 2D shape {arr.shape}, expected ({len(R)}, {len(Z)}) or transposed"
        )

    R_grid = np.asarray(eq_slice["profiles_2d.0.grid.dim1"], float)
    Z_grid = np.asarray(eq_slice["profiles_2d.0.grid.dim2"], float)
    psi_RZ = _ensure_rz_shape(
        np.asarray(eq_slice["profiles_2d.0.psi"], float), R_grid, Z_grid
    )

    psi_axis = float(eq_slice.get("global_quantities.psi_axis", np.nan))
    psi_lcfs = float(eq_slice.get("global_quantities.psi_boundary", np.nan))
    if not np.isfinite(psi_axis) or not np.isfinite(psi_lcfs) or psi_lcfs == psi_axis:
        psi_axis = float(np.nanmin(psi_RZ))
        psi_lcfs = float(np.nanmax(psi_RZ))

    f_1d = np.asarray(eq_slice["profiles_1d.f"], float)
    if "profiles_1d.psi" in eq_slice:
        psi_1d = np.asarray(eq_slice["profiles_1d.psi"], float)
        psiN_1d = (psi_1d - psi_axis) / (psi_lcfs - psi_axis)
        idx = np.argsort(psi_1d)
        f_vac_val = float(np.interp(psi_lcfs, psi_1d[idx], f_1d[idx]))
    elif "profiles_1d.psi_norm" in eq_slice:
        psiN_1d = np.asarray(eq_slice["profiles_1d.psi_norm"], float)
        f_vac_val = float(np.interp(1.0, psiN_1d, f_1d))
    else:
        raise KeyError("Need profiles_1d.psi or profiles_1d.psi_norm for F profile")

    if psiN_1d.size != f_1d.size:
        raise ValueError("profiles_1d F and psi/psi_norm must have the same length")

    return calculate_reconstructed_diamagnetic_flux(
        R_grid, Z_grid, psi_RZ, psi_axis, psi_lcfs, psiN_1d, f_1d, f_vac_val
    )


def compute_diamagnetism(ods, time_index=0):
    """
    Compute diamagnetism μ_i from ODS using the volume-integral definition.

    μ_i = (1 / (B_pa² Ω)) ∫_Ω (B_tv² - B_t²) dV

    Loads equilibrium data, B_pa (average boundary poloidal field), V_p (plasma volume),
    and calls :func:`vaft.process.equilibrium.calculate_diamagnetism`.
    """
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        raise KeyError("equilibrium.time_slice not found in ODS")
    if time_index >= len(ods["equilibrium.time_slice"]):
        raise IndexError(
            f"time_index {time_index} is out of range for equilibrium.time_slice"
        )

    eq_slice = ods["equilibrium.time_slice"][time_index]

    def _ensure_rz_shape(arr: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, float)
        if arr.shape == (len(R), len(Z)):
            return arr
        if arr.shape == (len(Z), len(R)):
            return arr.T
        raise ValueError(
            f"Unexpected 2D shape {arr.shape}, expected ({len(R)}, {len(Z)}) or transposed"
        )

    R_grid = np.asarray(eq_slice["profiles_2d.0.grid.dim1"], float)
    Z_grid = np.asarray(eq_slice["profiles_2d.0.grid.dim2"], float)
    psi_RZ = _ensure_rz_shape(
        np.asarray(eq_slice["profiles_2d.0.psi"], float), R_grid, Z_grid
    )

    psi_axis = float(eq_slice.get("global_quantities.psi_axis", np.nan))
    psi_lcfs = float(eq_slice.get("global_quantities.psi_boundary", np.nan))
    if not np.isfinite(psi_axis) or not np.isfinite(psi_lcfs) or psi_lcfs == psi_axis:
        psi_axis = float(np.nanmin(psi_RZ))
        psi_lcfs = float(np.nanmax(psi_RZ))

    f_1d = np.asarray(eq_slice["profiles_1d.f"], float)
    if "profiles_1d.psi" in eq_slice:
        psi_1d = np.asarray(eq_slice["profiles_1d.psi"], float)
        psiN_1d = (psi_1d - psi_axis) / (psi_lcfs - psi_axis)
        idx = np.argsort(psi_1d)
        psi_1d_s = psi_1d[idx]
        f_1d_s = f_1d[idx]
        f_at_lcfs = float(np.interp(psi_lcfs, psi_1d_s, f_1d_s))
        # F_vac is defined at LCFS only (vacuum reference). If μ_i comes out with
        # unexpected sign, check F profile sign convention (F = R*B_φ) and that
        # psi_norm/psi ordering (axis vs boundary) matches the equilibrium.
        f_vac_val = f_at_lcfs
    elif "profiles_1d.psi_norm" in eq_slice:
        psiN_1d = np.asarray(eq_slice["profiles_1d.psi_norm"], float)
        # OMAS: psi_norm 0 = axis, 1 = LCFS → F_vac = F at psi_norm=1
        f_vac_val = float(np.interp(1.0, psiN_1d, f_1d))
    else:
        raise KeyError("Need profiles_1d.psi or profiles_1d.psi_norm for F profile")

    if psiN_1d.size != f_1d.size:
        raise ValueError("profiles_1d F and psi/psi_norm must have the same length")

    R_bdry = np.asarray(eq_slice["boundary.outline.r"], float)
    Z_bdry = np.asarray(eq_slice["boundary.outline.z"], float)
    B_p_bdry, _, _ = poloidal_field_at_boundary(
        R_grid, Z_grid, psi_RZ, R_bdry, Z_bdry
    )
    B_pa = float(calculate_average_boundary_poloidal_field(R_bdry, Z_bdry, B_p_bdry))

    V_p = None
    if "profiles_1d.volume" in eq_slice:
        vol = np.asarray(eq_slice["profiles_1d.volume"], float)
        if vol.size >= 1 and np.isfinite(vol).any():
            V_p = float(np.nanmean(vol))
    if V_p is None or V_p <= 0:
        R_bc = np.append(R_bdry, R_bdry[0]) if (R_bdry[0] != R_bdry[-1] or Z_bdry[0] != Z_bdry[-1]) else R_bdry
        Z_bc = np.append(Z_bdry, Z_bdry[0]) if (R_bdry[0] != R_bdry[-1] or Z_bdry[0] != Z_bdry[-1]) else Z_bdry
        dR_b = np.diff(R_bc)
        dZ_b = np.diff(Z_bc)
        R_mid_b = 0.5 * (R_bc[:-1] + R_bc[1:])
        V_p = float(np.abs(-np.sum(np.pi * (R_mid_b**2) * dZ_b)))

    return calculate_diamagnetism(
        R_grid, Z_grid, psi_RZ, psi_axis, psi_lcfs,
        psiN_1d, f_1d, f_vac_val, B_pa, V_p=V_p
    )


def compute_ohmic_heating_power_from_core_profiles(ods: ODS, time_slice: Optional[int] = None, 
                                                    Z_eff: float = 2.0, ln_Lambda: float = 17.0) -> float:
    """
    Compute ohmic heating power from core profiles.
    
    Calculates P_Ω,diss = ∫_V η J_φ² dV where:
    - η is Spitzer resistivity calculated from T_e
    - J_φ is toroidal current density from equilibrium
    - Integration is over plasma volume
    
    Args:
        ods: OMAS data structure
        time_slice: Time slice index for core profile (None = use first available)
        Z_eff: Effective charge (default: 2.0)
        ln_Lambda: Coulomb logarithm (default: 17.0)
    
    Returns:
        P_ohm: Ohmic heating power [W]
    
    Raises:
        KeyError: If required data is missing
        ValueError: If plasma volume is zero
    """
    from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi, update_equilibrium_profiles_2d_j_tor
    
    # Find matching time indices between core_profiles and equilibrium
    cp_idx, equil_idx, time = find_matching_time_indices(ods, time_slice)
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    eq_ts = ods['equilibrium.time_slice'][equil_idx]
    
    # Get core profile data: T_e and rho_tor_norm
    grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
    if 'rho_tor_norm' not in grid:
        raise KeyError(f"rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}]")
    
    rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
    
    if 'electrons.temperature' not in cp_ts:
        raise KeyError(f"electrons.temperature missing in core_profiles.profiles_1d[{cp_idx}]")
    T_e_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
    
    # Get equilibrium profiles_1d for coordinate conversion
    eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
        if 'psi_norm' not in eq_profiles_1d:
            raise KeyError(f"Failed to create psi_norm for equilibrium.time_slice[{equil_idx}]")
    
    if 'rho_tor_norm' not in eq_profiles_1d:
        raise KeyError(f"rho_tor_norm missing in equilibrium.profiles_1d for time_slice[{equil_idx}]")
    
    rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
    psi_norm_eq = np.asarray(eq_profiles_1d['psi_norm'], float)
    
    # Ensure monotonicity
    if not np.all(np.diff(rho_tor_norm_eq) > 0):
        sort_idx = np.argsort(rho_tor_norm_eq)
        rho_tor_norm_eq_sorted = rho_tor_norm_eq[sort_idx]
        psi_norm_eq_sorted = psi_norm_eq[sort_idx]
        unique_mask = np.concatenate(([True], np.diff(rho_tor_norm_eq_sorted) > 1e-10))
        rho_tor_norm_eq_sorted = rho_tor_norm_eq_sorted[unique_mask]
        psi_norm_eq_sorted = psi_norm_eq_sorted[unique_mask]
    else:
        rho_tor_norm_eq_sorted = rho_tor_norm_eq
        psi_norm_eq_sorted = psi_norm_eq
    
    # Use equilibrium psi_norm grid as target coordinate system
    psiN_1d = psi_norm_eq_sorted
    
    # Create inverse mapping: psi_norm -> rho_tor_norm
    interp_psi_to_rho = interp1d(psi_norm_eq_sorted, rho_tor_norm_eq_sorted,
                                 kind='linear',
                                 bounds_error=False,
                                 fill_value=(rho_tor_norm_eq_sorted[0], rho_tor_norm_eq_sorted[-1]))
    rho_tor_norm_at_psiN = interp_psi_to_rho(psiN_1d)
    
    # Interpolate T_e to psi_norm coordinate
    interp_func = interp1d(rho_tor_norm_cp, T_e_1d_rho,
                          kind='linear',
                          bounds_error=False,
                          fill_value=(T_e_1d_rho[0], T_e_1d_rho[-1]))
    T_e_1d = interp_func(rho_tor_norm_at_psiN)
    
    # Get equilibrium 2D grid and ψ(R,Z)
    R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
    Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
    psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
    
    # Ensure psi_RZ shape convention is (len(R), len(Z))
    # When R_grid and Z_grid have the same length, use physical properties:
    # R_grid: major radius, typically all positive (R >= 0)
    # Z_grid: vertical position, typically has negative values (up-down symmetry)
    if psi_RZ.shape != (len(R_grid), len(Z_grid)):
        if psi_RZ.shape == (len(Z_grid), len(R_grid)):
            psi_RZ = psi_RZ.T
        else:
            raise ValueError(
                f"Unexpected psi_RZ shape {psi_RZ.shape}; expected {(len(R_grid), len(Z_grid))} or {(len(Z_grid), len(R_grid))}"
            )
    
    psi_axis = float(eq_ts['global_quantities.psi_axis'])
    psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
    
    # Map T_e to 2D (R,Z)
    T_e_RZ, psiN_RZ = psi_to_RZ(psiN_1d, T_e_1d, psi_RZ, psi_axis, psi_lcfs)
    
    # Calculate Spitzer resistivity 2D profile
    # Handle zero/negative temperatures (outside plasma)
    T_e_RZ_safe = np.where(T_e_RZ > 0, T_e_RZ, np.nan)
    eta_RZ = np.zeros_like(T_e_RZ)
    valid_mask = ~np.isnan(T_e_RZ_safe)
    eta_RZ[valid_mask] = spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(
        T_e_RZ_safe[valid_mask], Z_eff=Z_eff, ln_Lambda=ln_Lambda
    )
    
    # Get J_tor from equilibrium profiles_2d
    J_phi_RZ = None
    for key in ['profiles_2d.0.j_tor']:
        if key in eq_ts:
            try:
                J_phi_RZ = np.asarray(eq_ts[key], float)
                # Ensure shape matches (len(R), len(Z))
                if J_phi_RZ.shape != (len(R_grid), len(Z_grid)):
                    if J_phi_RZ.shape == (len(Z_grid), len(R_grid)):
                        J_phi_RZ = J_phi_RZ.T
                    else:
                        raise ValueError(f"J_phi shape {J_phi_RZ.shape} doesn't match grid")
                break
            except Exception as e:
                logger.warning(f"Found {key} but could not use it: {e}")
    
    # If j_tor not found, try to build it from 1D profile
    if J_phi_RZ is None:
        try:
            update_equilibrium_profiles_2d_j_tor(ods, time_slice=equil_idx)
            # Try again after update
            if 'profiles_2d.0.j_tor' in eq_ts:
                J_phi_RZ = np.asarray(eq_ts['profiles_2d.0.j_tor'], float)
                # Ensure shape matches (len(R), len(Z))
                if J_phi_RZ.shape != (len(R_grid), len(Z_grid)):
                    if J_phi_RZ.shape == (len(Z_grid), len(R_grid)):
                        J_phi_RZ = J_phi_RZ.T
                    else:
                        raise ValueError(f"J_phi shape {J_phi_RZ.shape} doesn't match grid")
        except Exception as e:
            logger.warning(f"Could not build 2D j_tor from 1D profile: {e}")
    
    if J_phi_RZ is None:
        raise KeyError(f"Toroidal current density (j_tor/jtor/j) not found in equilibrium.time_slice[{equil_idx}].profiles_2d.0 and could not be built from profiles_1d.j_tor")
    
    # Calculate eta * J_phi^2 2D profile
    eta_J2_RZ = eta_RZ * (J_phi_RZ ** 2)
    
    # Compute volume integral: P_ohm = ∫_V η J_φ² dV
    # Using volume_average: returns (average, volume), so integral = average * volume
    p_avg, V = volume_average(eta_J2_RZ, psiN_RZ, R_grid, Z_grid)
    P_ohm = float(p_avg * V)  # [W]
    
    return P_ohm

def compute_volume_averaged_pressure(ods: ODS, time_slice: Optional[int] = None, option: str = 'equilibrium') -> np.ndarray:
    """
    Compute volume-averaged pressure for equilibrium time slices.
    
    Two options available:
    - 'equilibrium': Uses profiles_1d.psi_norm and profiles_1d.pressure from equilibrium
    - 'core_profiles': Computes pressure from core_profiles as p = 2 * n_e * T_e * e
    
    Args:
        ods: OMAS data structure
        time_slice: If None, compute for all time slices. If int, compute for specific slice.
        option: 'equilibrium' or 'core_profiles' (default: 'equilibrium')
    
    Returns:
        np.ndarray: Volume-averaged pressure array (length = number of time slices processed)
    """
    from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi
    
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")
    
    def _ensure_rz_shape(arr: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Ensure 2D array is shaped as (len(R), len(Z)) to match indexing='ij' mesh."""
        arr = np.asarray(arr)
        if arr.shape == (len(R), len(Z)):
            return arr
        if arr.shape == (len(Z), len(R)):
            return arr.T
        raise ValueError(f"Unexpected 2D array shape {arr.shape}, expected {(len(R), len(Z))} or {(len(Z), len(R))}")
    
    # Determine which time slices to process
    if time_slice is None:
        time_slices = list(range(len(ods['equilibrium.time_slice'])))
    else:
        time_slices = [int(time_slice)]
        if time_slice >= len(ods['equilibrium.time_slice']):
            raise IndexError(f"time_slice {time_slice} is out of bounds")
    
    pressure_vol_avg_list = []
    
    for eq_idx in time_slices:
        eq_ts = ods['equilibrium.time_slice'][eq_idx]
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=eq_idx)
        try:
            # Load 2D grid + psi
            R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
            Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
            psi_RZ = _ensure_rz_shape(np.asarray(eq_ts['profiles_2d.0.psi'], float), R_grid, Z_grid)
            
            # Get psi normalization constants
            psi_axis = float(eq_ts.get('global_quantities.psi_axis', np.nan))
            psi_lcfs = float(eq_ts.get('global_quantities.psi_boundary', np.nan))
            if not np.isfinite(psi_axis) or not np.isfinite(psi_lcfs) or psi_lcfs == psi_axis:
                # Fallback: normalize by min/max of psi_RZ
                psi_axis = float(np.nanmin(psi_RZ))
                psi_lcfs = float(np.nanmax(psi_RZ))
            
            if option == 'equilibrium':
                # Extract 1D profiles: psi_norm and pressure from equilibrium
                psi_norm_1d = np.asarray(eq_ts['profiles_1d.psi_norm'], float)
                p_1d = np.asarray(eq_ts['profiles_1d.pressure'], float)
                
                # Check that arrays have same length
                if len(psi_norm_1d) != len(p_1d):
                    logger.warning(f"Time slice {eq_idx}: psi_norm and pressure have different lengths, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
            elif option == 'core_profiles':
                # Compute pressure from core_profiles: p = 2 * n_e * T_e * e
                if 'core_profiles.profiles_1d' not in ods or len(ods['core_profiles.profiles_1d']) == 0:
                    logger.warning(f"Time slice {eq_idx}: core_profiles.profiles_1d not found, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
                # Find matching core profile time slice using find_matching_time_indices
                # Iterate through core profiles to find one that matches this equilibrium index
                cp_idx = None
                for cp_idx_candidate in range(len(ods['core_profiles.profiles_1d'])):
                    try:
                        cp_idx_found, equil_idx_found, _ = find_matching_time_indices(ods, time_slice=cp_idx_candidate)
                        if equil_idx_found == eq_idx:
                            cp_idx = cp_idx_found
                            break
                    except (KeyError, ValueError):
                        # Continue searching if this core profile doesn't match
                        continue
                
                if cp_idx is None:
                    logger.warning(f"Time slice {eq_idx}: No matching core profile time slice found, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
                cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
                
                # Get core profile grid
                grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
                if 'rho_tor_norm' not in grid:
                    logger.warning(f"Time slice {eq_idx}: rho_tor_norm grid missing in core_profiles, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
                rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
                
                # Get n_e and T_e
                if 'electrons.density' not in cp_ts:
                    logger.warning(f"Time slice {eq_idx}: electrons.density missing in core_profiles, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                n_e_1d_rho = np.asarray(cp_ts['electrons.density'], float)
                
                if 'electrons.temperature' not in cp_ts:
                    logger.warning(f"Time slice {eq_idx}: electrons.temperature missing in core_profiles, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                T_e_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
                
                # Check array lengths
                if len(rho_tor_norm_cp) != len(n_e_1d_rho) or len(rho_tor_norm_cp) != len(T_e_1d_rho):
                    logger.warning(f"Time slice {eq_idx}: Array length mismatch in core_profiles, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
                # Get equilibrium profiles_1d for coordinate conversion
                eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
                
                # Ensure equilibrium has psi_norm
                if 'psi_norm' not in eq_profiles_1d:
                    update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=eq_idx)
                    eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
                    if 'psi_norm' not in eq_profiles_1d:
                        logger.warning(f"Time slice {eq_idx}: Failed to create psi_norm, skipping")
                        pressure_vol_avg_list.append(np.nan)
                        continue
                
                if 'rho_tor_norm' not in eq_profiles_1d:
                    logger.warning(f"Time slice {eq_idx}: rho_tor_norm missing in equilibrium.profiles_1d, skipping")
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
                rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
                psi_norm_eq = np.asarray(eq_profiles_1d['psi_norm'], float)
                
                # Ensure monotonicity
                if not np.all(np.diff(rho_tor_norm_eq) > 0):
                    sort_idx = np.argsort(rho_tor_norm_eq)
                    rho_tor_norm_eq_sorted = rho_tor_norm_eq[sort_idx]
                    psi_norm_eq_sorted = psi_norm_eq[sort_idx]
                    unique_mask = np.concatenate(([True], np.diff(rho_tor_norm_eq_sorted) > 1e-10))
                    rho_tor_norm_eq_sorted = rho_tor_norm_eq_sorted[unique_mask]
                    psi_norm_eq_sorted = psi_norm_eq_sorted[unique_mask]
                else:
                    rho_tor_norm_eq_sorted = rho_tor_norm_eq
                    psi_norm_eq_sorted = psi_norm_eq
                
                # Use equilibrium psi_norm grid as target coordinate system
                psiN_1d = psi_norm_eq_sorted
                
                # Create inverse mapping: psi_norm -> rho_tor_norm
                interp_psi_to_rho = interp1d(psi_norm_eq_sorted, rho_tor_norm_eq_sorted,
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=(rho_tor_norm_eq_sorted[0], rho_tor_norm_eq_sorted[-1]))
                rho_tor_norm_at_psiN = interp_psi_to_rho(psiN_1d)
                
                # Interpolate n_e and T_e to psi_norm coordinate
                interp_n_e = interp1d(rho_tor_norm_cp, n_e_1d_rho,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value=(n_e_1d_rho[0], n_e_1d_rho[-1]))
                n_e_1d = interp_n_e(rho_tor_norm_at_psiN)
                
                interp_T_e = interp1d(rho_tor_norm_cp, T_e_1d_rho,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value=(T_e_1d_rho[0], T_e_1d_rho[-1]))
                T_e_1d = interp_T_e(rho_tor_norm_at_psiN)
                
                # Calculate pressure: p = 2 * n_e * T_e * e
                # T_e is in eV, e = 1.602176634e-19 C (elementary charge)
                QE = 1.602176634e-19  # elementary charge [C]
                p_1d = 2.0 * n_e_1d * T_e_1d * QE  # [Pa]
                
                # Use psi_norm_1d from equilibrium
                psi_norm_1d = psiN_1d
                
            else:
                raise ValueError(f"Invalid option: {option}. Must be 'equilibrium' or 'core_profiles'")
            
            # Build 2D pressure map using psi_to_RZ
            p_RZ, psiN_RZ = psi_to_RZ(psi_norm_1d, p_1d, psi_RZ, psi_axis, psi_lcfs)
            
            # Compute volume average
            p_avg, _ = volume_average(p_RZ, psiN_RZ, R_grid, Z_grid)
            pressure_vol_avg_list.append(float(p_avg))
            
        except Exception as e:
            logger.warning(f"Time slice {eq_idx}: Could not compute volume-averaged pressure: {e}")
            pressure_vol_avg_list.append(np.nan)
    
    return np.asarray(pressure_vol_avg_list, float)
