from typing import List, Tuple, Dict, Any, Optional
from numpy import ndarray
import numpy as np
from omas import *
from pathlib import Path

from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
from vaft.process import (
    compute_br_bz_phi,
    compute_response_matrix,
    compute_impedance_matrices,
    solve_eddy_currents,
    compute_vacuum_fields_1d,
    time_derivative,
    psi_to_rz,
    volume_average,
    poloidal_field_at_boundary,
    calculate_average_boundary_poloidal_field,
    shafranov_integrals,
    efit_virial_volume_integrals,
    computed_diamagnetism_from_phi,
    fractional_cell_weights_from_boundary,
    calculate_reconstructed_diamagnetic_flux,
    calculate_diamagnetism,
    prepare_boundary_for_shafranov,
    extract_flux_surface_contours,
    make_equilibrium_field_interpolator,
    project_points,
    sweep_toroidal,
    toroidal_ring,
    trace_field_line,
    trajectory_world_points,
)
from vaft.formula import (
    spitzer_resistivity_from_T_e_Z_eff_ln_Lambda,
    virial_bongard_from_S_alpha_mu,
    virial_lao_from_S_alpha_mu_rt,
    virial_beta_pd_from_S_mu_rt,
    kinetic_energy_from_beta_p_B_pa_V_p,
    magnetic_energy_from_li_B_pa_V_p,
)
from vaft.omas import find_matching_time_indices
from vaft.omas.update import update_equilibrium_boundary
from scipy.interpolate import interp1d
import logging
import vaft.process
from matplotlib.path import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEOMETRY_TYPE_POLYGON = 1
GEOMETRY_TYPE_RECTANGLE = 2
DT_SUB = 5e-5
#: Observation-source separation [m] below which the exact response path
#: warns: the clamped elliptic integrals return finite artifacts there.
COINCIDENT_SOURCE_TOL = 1e-6

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

def compute_point_response_matrices_ods(
    ods: ODS,
    rz: List[List[float]],
    plasma_points: Optional[List[List[float]]] = None,
    components: Tuple[str, ...] = ("psi", "bz", "br"),
    ) -> Tuple[ndarray, ...]:
    """Vectorized, exact-elliptic (Psi, Bz, Br) response matrices from an ODS.

    Fast alternative to :func:`compute_point_response_ods` (issue #239):
    delegates to ``vaft.process.electromagnetics.compute_point_response_matrices``
    (scipy-exact Green's functions, full NumPy broadcasting) instead of the
    per-point Python loops over ``compute_br_bz_phi``. Column ordering matches
    :func:`compute_point_response_ods`: ``[coils..., loops..., plasma...]``,
    with coil columns summed over discretized elements weighted by
    ``turns_with_sign`` and passive loops reduced to their outline centroid
    (or rectangle centre).

    Two deliberate differences from the legacy path: exact elliptic integrals
    instead of the polynomial approximation (~1e-6 relative), and no 1 cm
    shift-averaging near sources. An observation point coincident with a
    source does NOT diverge or raise — the exact Green's functions clamp the
    elliptic parameter, so it returns finite but PHYSICALLY MEANINGLESS
    values; a UserWarning is emitted when any observation point sits within
    ``COINCIDENT_SOURCE_TOL`` of a source. Keep observation points off the
    source locations.

    :param ods: OMAS data structure with ``pf_active`` and ``pf_passive``
    :param rz: observation points, sequence of [r, z] pairs (shape (n, 2))
    :param plasma_points: optional plasma filament points, same shape rules
    :param components: which matrices to compute, a subset of
        ("psi", "bz", "br") in the desired order; "psi" alone skips the
        field-component elliptic passes (~3x cheaper)
    :return: matrices matching *components* (default (Psi, Bz, Br)),
        each (n_points, nbcoil + nbloop + nbplas)
    """
    import warnings

    from vaft.process.electromagnetics import compute_point_response_matrices

    rz = np.atleast_2d(np.asarray(rz, dtype=float))
    if rz.ndim != 2 or rz.shape[1] != 2:
        raise ValueError(f"rz must have shape (n, 2) of [r, z] pairs; got {rz.shape}")
    obs_r, obs_z = rz[:, 0], rz[:, 1]

    try:
        pf = ods["pf_active"]
        pfp = ods["pf_passive"]
        nbcoil = len(pf["coil"])
        nbloop = len(pfp["loop"])

        src_r, src_z, turns, groups = [], [], [], []
        for ii in range(nbcoil):
            for jj in range(len(pf[f"coil.{ii}.element"])):
                src_r.append(float(pf[f"coil.{ii}.element.{jj}.geometry.rectangle.r"]))
                src_z.append(float(pf[f"coil.{ii}.element.{jj}.geometry.rectangle.z"]))
                turns.append(float(pf[f"coil.{ii}.element.{jj}.turns_with_sign"]))
                groups.append(ii)
        for ii in range(nbloop):
            geometry = pfp[f"loop.{ii}.element[0].geometry"]
            if geometry["geometry_type"] == GEOMETRY_TYPE_POLYGON:
                src_r.append(float(np.mean(geometry["outline.r"])))
                src_z.append(float(np.mean(geometry["outline.z"])))
            else:
                src_r.append(float(geometry["rectangle.r"]))
                src_z.append(float(geometry["rectangle.z"]))
            turns.append(1.0)
            groups.append(nbcoil + ii)
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

    n_groups = nbcoil + nbloop
    if plasma_points is not None and len(plasma_points) > 0:
        plasma = np.atleast_2d(np.asarray(plasma_points, dtype=float))
        if plasma.ndim != 2 or plasma.shape[1] != 2:
            raise ValueError(
                f"plasma_points must have shape (n, 2) of [r, z] pairs; "
                f"got {plasma.shape}"
            )
        for r_p, z_p in plasma:
            src_r.append(float(r_p))
            src_z.append(float(z_p))
            turns.append(1.0)
            groups.append(n_groups)
            n_groups += 1

    src_r = np.asarray(src_r)
    src_z = np.asarray(src_z)
    sep2 = (obs_r[:, None] - src_r[None, :]) ** 2 + (obs_z[:, None] - src_z[None, :]) ** 2
    n_coincident = int(np.count_nonzero(np.min(sep2, axis=1) < COINCIDENT_SOURCE_TOL**2))
    if n_coincident:
        warnings.warn(
            f"{n_coincident} observation point(s) coincide with a source "
            f"(within {COINCIDENT_SOURCE_TOL:g} m); the exact Green's "
            "functions return finite but physically meaningless values there",
            stacklevel=2,
        )

    return compute_point_response_matrices(
        obs_r,
        obs_z,
        src_r,
        src_z,
        turns=np.asarray(turns),
        groups=np.asarray(groups, dtype=int),
        n_groups=n_groups,
        components=components,
    )

def ensure_em_coupling(ods: ODS) -> None:
    """Populate ``em_coupling`` from the packaged asset when it is absent.

    The coupling matrices are a function of the shot's PF geometry version, not
    a measurement, so the compact samples ship the pf_active/pf_passive geometry
    and leave the matrices to be reconstructed. Rebuild them here rather than
    making every caller materialize them first.

    A partially populated ``em_coupling`` still needs reconstruction: the sample
    that shipped before the geometry-only change carried
    ``mutual_passive_active`` without ``mutual_passive_passive``, so gating on
    any single matrix lets such an ODS through to a bare ``KeyError`` in
    :func:`compute_impedance_matrices_ods`. Require every matrix that function
    reads, and leave a caller-supplied pair untouched.
    """
    existing = ods["em_coupling"] if "em_coupling" in ods else None
    if existing is not None and all(
        np.size(existing.get(matrix, []))
        for matrix in ("mutual_passive_passive", "mutual_passive_active")
    ):
        return

    from vaft.machine_mapping.em_coupling import em_coupling as _map_em_coupling

    shot = None
    try:
        shot = int(ods["dataset_description.data_entry.pulse"])
    except (KeyError, ValueError, TypeError):
        pass
    _map_em_coupling(ods, shot=shot)




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
        ensure_em_coupling(ods)
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
        if np.shape(mutual_pa) != (nbloop, nbcoil):
            raise ValueError(
                "em_coupling.mutual_passive_active must have shape "
                f"({nbloop}, {nbcoil}), got {np.shape(mutual_pa)}"
            )

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

        # Compute impedance matrices
        R_mat, L_mat, M_mat = compute_impedance_matrices(
            loop_res,
            passive_loop_geometry,
            None,
            mutual_pp,
            mutual_pa,
            plasma
        )

        return R_mat, L_mat, M_mat
    except KeyError as e:
        logger.error(f"Missing required data in ODS: {e}")
        raise

def compute_eddy_currents(
    ods: ODS,
    plasma: List[Tuple[float, float]],
    ip: List[ndarray],
    dt_sub: float = DT_SUB,
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
            dt_sub=dt_sub
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
        # compute_point_response_ods returns (Psi, Bz, Br) -- Bz before Br.
        # Unpacking it as (psi, br, bz) swapped the two field components for
        # every caller of this wrapper.
        psi_c, bz_c, br_c = compute_point_response_ods(ods, rz, plasma=None)
        
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

        # Plot if requested.  Rendering is delegated to vaft.plot; this
        # namespace only shapes the data into the view model (issue #63).
        if plot_opt:
            _plot_vacuum_field_quantities(time_arr, psi_out, br_out, bz_out, rz, mode)
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
    if 'time' not in ods['pf_passive']:
        # Geometry-only samples carry no passive-loop waveform; solve it from
        # the active-coil currents instead of requiring it to be stored.
        compute_eddy_currents(ods, plasma=[], ip=[])
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
    grid = cp_ts['grid'] if 'grid' in cp_ts else (ods['core_profiles.grid'] if 'core_profiles.grid' in ods else ODS())
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
    eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
        eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
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
    grid = cp_ts['grid'] if 'grid' in cp_ts else (ods['core_profiles.grid'] if 'core_profiles.grid' in ods else ODS())
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
    eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
        eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
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
    profile_RZ, psiN_RZ = psi_to_rz(psiN_1d, profile_1d, psi_RZ, psi_axis, psi_lcfs)
    
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
        # B_pol comes from grad(psi)/R, so psi must be in Wb/rad regardless of
        # the ODS storage convention (Wb per the IMAS DD since issue #236).
        _psi_factor = ods_psi_to_wb_per_radian_factor(ods, eq_idx)
        psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float) * _psi_factor
        psi_axis = float(eq_ts['global_quantities.psi_axis']) * _psi_factor
        psi_lcfs = float(eq_ts['global_quantities.psi_boundary']) * _psi_factor
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
    # Always refresh boundary-derived geometry before virial calculations.
    try:
        update_equilibrium_boundary(
            ods,
            time_slice=None if time_slice is None else slices_to_process,
        )
    except Exception as exc:
        logger.warning("update_equilibrium_boundary failed before virial: %s", exc)

    out = {}
    for eq_idx in slices_to_process:
        eq_ts = ods["equilibrium.time_slice"][eq_idx]
        try:
            R_grid_1d = np.asarray(eq_ts["profiles_2d.0.grid.dim1"], float)
            Z_grid_1d = np.asarray(eq_ts["profiles_2d.0.grid.dim2"], float)
            # Convert to Wb/rad: the Shafranov/virial integrals build B_pol
            # from grad(psi)/R (issue #236).
            _psi_factor = ods_psi_to_wb_per_radian_factor(ods, eq_idx)
            psi_RZ = np.asarray(eq_ts["profiles_2d.0.psi"], float) * _psi_factor
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

        # Boundary outline (normalized to CCW + uniform 256-point arc-length sampling).
        R_bdry_raw = np.asarray(eq_ts["boundary.outline.r"], float)
        Z_bdry_raw = np.asarray(eq_ts["boundary.outline.z"], float)
        R_bdry, Z_bdry = prepare_boundary_for_shafranov(
            R_bdry_raw,
            Z_bdry_raw,
            n_points=256,
            enforce_ccw=True,
        )
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


        # Axis geometry: validate per-slice and fallback to boundary geometry when missing/invalid.
        R_0 = np.nan
        Z_0 = np.nan
        if "boundary.geometric_axis.r" in eq_ts:
            try:
                R_0 = float(eq_ts["boundary.geometric_axis.r"])
            except Exception:
                R_0 = np.nan
        if "boundary.geometric_axis.z" in eq_ts:
            try:
                Z_0 = float(eq_ts["boundary.geometric_axis.z"])
            except Exception:
                Z_0 = np.nan
        if not np.isfinite(R_0) or not np.isfinite(Z_0):
            R_0 = 0.5 * (float(np.min(R_bdry)) + float(np.max(R_bdry)))
            Z_0 = 0.5 * (float(np.min(Z_bdry)) + float(np.max(Z_bdry)))
            logger.warning(
                "Time slice %s: invalid boundary.geometric_axis; fallback to boundary center (R0=%.6f, Z0=%.6f).",
                eq_idx,
                R_0,
                Z_0,
            )
            try:
                eq_ts["boundary.geometric_axis.r"] = float(R_0)
                eq_ts["boundary.geometric_axis.z"] = float(Z_0)
            except Exception:
                logger.debug(
                    "Time slice %s: failed to write fallback boundary.geometric_axis into ODS.",
                    eq_idx,
                )

        # Build common (R,Z) mesh and internal cell-fraction weights.
        R_mesh, Z_mesh = np.meshgrid(R_grid_1d, Z_grid_1d, indexing="ij")
        cell_weights = fractional_cell_weights_from_boundary(
            R_mesh,
            Z_mesh,
            R_bdry,
            Z_bdry,
            samples_per_axis=5,
        )

        # Psi normalization for profile mapping (same Wb/rad frame as psi_RZ)
        psi_axis = float(eq_ts["global_quantities.psi_axis"]) * _psi_factor if "global_quantities.psi_axis" in eq_ts else np.nan
        psi_lcfs = float(eq_ts["global_quantities.psi_boundary"]) * _psi_factor if "global_quantities.psi_boundary" in eq_ts else np.nan
        if (not np.isfinite(psi_axis)) or (not np.isfinite(psi_lcfs)) or psi_lcfs == psi_axis:
            psi_axis = float(np.nanmin(psi_RZ))
            psi_lcfs = float(np.nanmax(psi_RZ))
        psiN_RZ = (psi_RZ - psi_axis) / (psi_lcfs - psi_axis)

        # F profile (R*B_phi)
        f_1d = np.asarray(eq_ts["profiles_1d.f"], float) if "profiles_1d.f" in eq_ts else np.asarray([], float)
        psiN_1d = None
        if f_1d.size:
            if "profiles_1d.psi" in eq_ts:
                psi_1d = np.asarray(eq_ts["profiles_1d.psi"], float) * _psi_factor
                if psi_1d.size == f_1d.size and psi_lcfs != psi_axis:
                    psiN_1d = (psi_1d - psi_axis) / (psi_lcfs - psi_axis)
            elif "profiles_1d.psi_norm" in eq_ts:
                psiN_arr = np.asarray(eq_ts["profiles_1d.psi_norm"], float)
                if psiN_arr.size == f_1d.size:
                    psiN_1d = psiN_arr

        # Non-rotating branch: p_tot = p(psi)
        p_2d = np.zeros_like(psi_RZ, dtype=float)
        p_boundary = 0.0
        if "profiles_1d.pressure" in eq_ts:
            p_1d = np.asarray(eq_ts["profiles_1d.pressure"], float)
            if psiN_1d is not None and p_1d.size == psiN_1d.size:
                p_2d, _ = psi_to_rz(psiN_1d, p_1d, psi_RZ, psi_axis, psi_lcfs)
                order_p = np.argsort(psiN_1d)
                p_boundary = float(np.interp(1.0, psiN_1d[order_p], p_1d[order_p]))

        R_safe = np.where(R_mesh == 0.0, np.nan, R_mesh)
        F_2d = np.full_like(psi_RZ, np.nan, dtype=float)
        F_boundary = np.nan
        if psiN_1d is not None:
            order_f = np.argsort(psiN_1d)
            psiN_sorted = psiN_1d[order_f]
            f_sorted = f_1d[order_f]
            psiN_clip = np.clip(psiN_RZ, psiN_sorted[0], psiN_sorted[-1])
            F_2d = np.interp(psiN_clip.ravel(), psiN_sorted, f_sorted).reshape(psi_RZ.shape)
            F_boundary = float(np.interp(1.0, psiN_sorted, f_sorted))
        else:
            # Fallback: use magnetic-axis toroidal field as constant F
            B_t_axis = float(eq_ts["global_quantities.magnetic_axis.b_field_tor"]) if "global_quantities.magnetic_axis.b_field_tor" in eq_ts else np.nan
            R_axis = float(eq_ts["global_quantities.magnetic_axis.r"]) if "global_quantities.magnetic_axis.r" in eq_ts else np.nan
            if np.isfinite(B_t_axis) and np.isfinite(R_axis):
                F_boundary = B_t_axis * R_axis
                F_2d = np.full_like(psi_RZ, F_boundary, dtype=float)

        B_phi_grid = np.where(np.isfinite(F_2d), F_2d / R_safe, np.nan)
        B_phi_vac_grid = np.where(np.isfinite(F_boundary), F_boundary / R_safe, np.nan)

        # Volume terms (alpha, RT, Phi_dia_comp, volume)
        vol_terms = efit_virial_volume_integrals(
            R_mesh,
            Z_mesh,
            R_bdry,
            Z_bdry,
            B_R_grid,
            B_Z_grid,
            p_tot_grid=p_2d,
            B_phi_grid=B_phi_grid,
            B_phi_vac_grid=B_phi_vac_grid,
            F_grid=F_2d,
            F_boundary=F_boundary if np.isfinite(F_boundary) else None,
            cell_weights=cell_weights,
        )
        alpha = float(vol_terms["alpha"]) if np.isfinite(vol_terms["alpha"]) else np.nan
        RT = float(vol_terms["rt"]) if np.isfinite(vol_terms["rt"]) else np.nan
        phi_dia_comp = float(vol_terms["phi_dia_comp"]) if np.isfinite(vol_terms["phi_dia_comp"]) else np.nan
        V_p = float(vol_terms["volume"]) if np.isfinite(vol_terms["volume"]) else np.nan

        # Boundary terms (S1,S2,S3), using same B_ref = B_pa and weighted volume
        S1, S2, S3, _alpha_check = shafranov_integrals(
            R_bdry,
            Z_bdry,
            B_p_bdry,
            R_mesh,
            Z_mesh,
            B_R_grid,
            B_Z_grid,
            R_0=R_0,
            Z_0=Z_0,
            p_boundary=p_boundary,
            B_ref=B_pa,
            cell_weights=cell_weights,
            volume=V_p if np.isfinite(V_p) and V_p > 0 else None,
        )
        S1, S2, S3 = float(S1), float(S2), float(S3)
        if not np.isfinite(alpha):
            alpha = float(_alpha_check)

        B_t0 = np.nan
        if "equilibrium.vacuum_toroidal_field.b0" in ods:
            B_t0 = float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).flat[0])
        elif "global_quantities.magnetic_axis.b_field_tor" in eq_ts:
            B_t0 = float(eq_ts["global_quantities.magnetic_axis.b_field_tor"])

        mui = np.nan
        if np.isfinite(phi_dia_comp) and np.isfinite(B_t0) and np.isfinite(V_p) and np.isfinite(B_pa):
            if V_p > 0.0 and B_pa > 0.0:
                mui = computed_diamagnetism_from_phi(phi_dia_comp, B_t0, R_0, V_p, B_pa)

        RT_over_R0 = np.nan
        if np.isfinite(RT) and R_0 != 0.0:
            rt_candidate = RT / R_0
            # Additional sanity guard on RT/R0 to avoid closure blow-up when RT denominator is near-singular.
            if np.isfinite(rt_candidate) and abs(rt_candidate) <= 10.0:
                RT_over_R0 = rt_candidate
            else:
                logger.warning(
                    "Time slice %s: RT/R0 is abnormal (%.6g); skipping Lao closure for this slice.",
                    eq_idx,
                    rt_candidate,
                )

        beta_p_lao = np.nan
        li_lao = np.nan
        beta_p_bongard = np.nan
        li_bongard = np.nan
        beta_pd_vir = np.nan
        if np.isfinite(alpha) and np.isfinite(mui) and np.isfinite(RT_over_R0):
            try:
                beta_p_lao, li_lao = virial_lao_from_S_alpha_mu_rt(
                    S1, S2, S3, alpha, mui, RT_over_R0
                )
                beta_pd_vir = virial_beta_pd_from_S_mu_rt(S1, S2, mui, RT_over_R0)
            except ValueError:
                beta_p_lao, li_lao, beta_pd_vir = np.nan, np.nan, np.nan
        if np.isfinite(alpha) and np.isfinite(mui):
            try:
                beta_p_bongard, li_bongard = virial_bongard_from_S_alpha_mu(
                    S1, S2, S3, alpha, mui
                )
            except ValueError:
                beta_p_bongard, li_bongard = np.nan, np.nan

        # Keep backward-compatible top-level beta_p/li mapped to Lao version.
        beta_p = float(beta_p_lao) if np.isfinite(beta_p_lao) else np.nan
        li = float(li_lao) if np.isfinite(li_lao) else np.nan

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
            "W_mag": W_mag, "W_kin": W_kin, "V_p": V_p,
            "mui_hat": float(mui) if np.isfinite(mui) else np.nan,  # backward-compatible key
            "mui": float(mui) if np.isfinite(mui) else np.nan,
            "rt": RT,
            "phi_dia_comp": phi_dia_comp,
            "beta_p_vir": float(beta_p_lao) if np.isfinite(beta_p_lao) else np.nan,
            "li_vir": float(li_lao) if np.isfinite(li_lao) else np.nan,
            "beta_pd_vir": float(beta_pd_vir) if np.isfinite(beta_pd_vir) else np.nan,
            "virial_lao": {
                "beta_p": float(beta_p_lao) if np.isfinite(beta_p_lao) else np.nan,
                "li": float(li_lao) if np.isfinite(li_lao) else np.nan,
            },
            "virial_bongard": {
                "beta_p": float(beta_p_bongard) if np.isfinite(beta_p_bongard) else np.nan,
                "li": float(li_bongard) if np.isfinite(li_bongard) else np.nan,
            },
            "beta_p_vir_lao": float(beta_p_lao) if np.isfinite(beta_p_lao) else np.nan,
            "li_vir_lao": float(li_lao) if np.isfinite(li_lao) else np.nan,
            "beta_p_vir_bongard": float(beta_p_bongard) if np.isfinite(beta_p_bongard) else np.nan,
            "li_vir_bongard": float(li_bongard) if np.isfinite(li_bongard) else np.nan,
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

    psi_axis = float(eq_slice["global_quantities.psi_axis"]) if "global_quantities.psi_axis" in eq_slice else np.nan
    psi_lcfs = float(eq_slice["global_quantities.psi_boundary"]) if "global_quantities.psi_boundary" in eq_slice else np.nan
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


def compute_diamagnetic_flux_measured_vs_computed(
    ods: ODS,
    time_slice: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compare measured vs reconstructed diamagnetic flux at equilibrium time slices.

    - measured: magnetics.diamagnetic_flux interpolated at equilibrium time
    - computed: reconstructed diamagnetic flux from equilibrium fields
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

    if delta_phi_interp is None:
        raise ValueError("Unable to build measured diamagnetic-flux interpolator.")

    slices_to_process = (
        list(range(len(ods["equilibrium.time_slice"])))
        if time_slice is None
        else [int(time_slice)]
    )

    out: Dict[int, Dict[str, float]] = {}
    for eq_idx in slices_to_process:
        if eq_idx < 0 or eq_idx >= len(ods["equilibrium.time_slice"]):
            raise IndexError(f"time_slice {eq_idx} is out of bounds for equilibrium.time_slice")

        eq_ts = ods["equilibrium.time_slice"][eq_idx]
        t_eq = float(eq_ts["time"]) if "time" in eq_ts else np.nan
        if (not np.isfinite(float(t_eq))) and "equilibrium.time" in ods and eq_idx < len(ods["equilibrium.time"]):
            t_eq = float(ods["equilibrium.time"][eq_idx])
        t_eq = float(t_eq)
        if not np.isfinite(t_eq):
            raise ValueError(f"Missing finite equilibrium time for time_slice {eq_idx}")

        measured = float(delta_phi_interp(t_eq))
        computed = float(compute_reconstructed_diamagnetic_flux(ods, time_index=eq_idx))
        difference = computed - measured
        relative_error = np.nan if measured == 0.0 else difference / measured

        out[eq_idx] = {
            "time": t_eq,
            "measured": measured,
            "computed": computed,
            "difference": float(difference),
            "relative_error": float(relative_error) if np.isfinite(relative_error) else np.nan,
        }

    return out


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
    # poloidal_field_at_boundary below needs psi in Wb/rad (issue #236).
    _psi_factor = ods_psi_to_wb_per_radian_factor(eq_slice)
    psi_RZ = _ensure_rz_shape(
        np.asarray(eq_slice["profiles_2d.0.psi"], float), R_grid, Z_grid
    ) * _psi_factor

    psi_axis = float(eq_slice["global_quantities.psi_axis"]) * _psi_factor if "global_quantities.psi_axis" in eq_slice else np.nan
    psi_lcfs = float(eq_slice["global_quantities.psi_boundary"]) * _psi_factor if "global_quantities.psi_boundary" in eq_slice else np.nan
    if not np.isfinite(psi_axis) or not np.isfinite(psi_lcfs) or psi_lcfs == psi_axis:
        psi_axis = float(np.nanmin(psi_RZ))
        psi_lcfs = float(np.nanmax(psi_RZ))

    f_1d = np.asarray(eq_slice["profiles_1d.f"], float)
    if "profiles_1d.psi" in eq_slice:
        psi_1d = np.asarray(eq_slice["profiles_1d.psi"], float) * _psi_factor
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
    grid = cp_ts['grid'] if 'grid' in cp_ts else (ods['core_profiles.grid'] if 'core_profiles.grid' in ods else ODS())
    if 'rho_tor_norm' not in grid:
        raise KeyError(f"rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}]")
    
    rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
    
    if 'electrons.temperature' not in cp_ts:
        raise KeyError(f"electrons.temperature missing in core_profiles.profiles_1d[{cp_idx}]")
    T_e_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
    
    # Get equilibrium profiles_1d for coordinate conversion
    eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
        eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
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
    T_e_RZ, psiN_RZ = psi_to_rz(psiN_1d, T_e_1d, psi_RZ, psi_axis, psi_lcfs)
    
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

    if option == 'core_profiles':
        if 'core_profiles.profiles_1d' not in ods or len(ods['core_profiles.profiles_1d']) == 0:
            logger.warning(
                "core_profiles.profiles_1d not found; returning NaN volume-averaged pressure "
                "for %d equilibrium slices.",
                len(time_slices),
            )
            return np.full(len(time_slices), np.nan, dtype=float)
    
    pressure_vol_avg_list = []
    no_matching_cp_slices = []
    
    for eq_idx in time_slices:
        eq_ts = ods['equilibrium.time_slice'][eq_idx]
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=eq_idx)
        try:
            # Load 2D grid + psi
            R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
            Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
            psi_RZ = _ensure_rz_shape(np.asarray(eq_ts['profiles_2d.0.psi'], float), R_grid, Z_grid)
            
            # Get psi normalization constants
            psi_axis = float(eq_ts['global_quantities.psi_axis']) if 'global_quantities.psi_axis' in eq_ts else np.nan
            psi_lcfs = float(eq_ts['global_quantities.psi_boundary']) if 'global_quantities.psi_boundary' in eq_ts else np.nan
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
                    no_matching_cp_slices.append(eq_idx)
                    pressure_vol_avg_list.append(np.nan)
                    continue
                
                cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
                
                # Get core profile grid
                grid = cp_ts['grid'] if 'grid' in cp_ts else (ods['core_profiles.grid'] if 'core_profiles.grid' in ods else ODS())
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
                eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
                
                # Ensure equilibrium has psi_norm
                if 'psi_norm' not in eq_profiles_1d:
                    update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=eq_idx)
                    eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
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
            p_RZ, psiN_RZ = psi_to_rz(psi_norm_1d, p_1d, psi_RZ, psi_axis, psi_lcfs)
            
            # Compute volume average
            p_avg, _ = volume_average(p_RZ, psiN_RZ, R_grid, Z_grid)
            pressure_vol_avg_list.append(float(p_avg))
            
        except Exception as e:
            logger.warning(f"Time slice {eq_idx}: Could not compute volume-averaged pressure: {e}")
            pressure_vol_avg_list.append(np.nan)
    
    if no_matching_cp_slices:
        preview = no_matching_cp_slices[:10]
        suffix = "..." if len(no_matching_cp_slices) > 10 else ""
        logger.warning(
            "No matching core profile time slice found for %d/%d equilibrium slices; "
            "skipping those slices (indices: %s%s)",
            len(no_matching_cp_slices),
            len(time_slices),
            preview,
            suffix,
        )

    return np.asarray(pressure_vol_avg_list, float)


# =====================================================================

# FAST-camera EFIT overlay: pinhole projection of equilibrium/wall geometry
# into camera pixel space. Read-only: computes and returns projected pixel
# coordinates, never writes them back into the ods.
# =====================================================================


_CAMERA_VISIBLE_CALIBRATED_SHOTS = (34764, 39915, 47518)
_DEFAULT_FLUX_SURFACE_LEVELS = (0.25, 0.5, 0.75, 0.95)


def _load_camera_intrinsics(intrinsics_path: str | Path | None = None) -> dict:
    """Load the shared VEST FAST-camera intrinsics (fx, fy, cx, cy, distortion)."""
    import json

    from vaft.machine_mapping import resolve_geometry_asset

    if intrinsics_path is not None:
        path = Path(intrinsics_path).expanduser()
    else:
        path = resolve_geometry_asset("camera_visible/intrinsics.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    camera_matrix = np.array(
        [[data["fx"], 0.0, data["cx"]], [0.0, data["fy"], data["cy"]], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    dist_coeffs = np.array(
        [data["k1"], data["k2"], data["p1"], data["p2"], data["k3"]], dtype=float
    )
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": tuple(data["image_size"]),
    }


def _load_camera_pose(shot: int, pose_path: str | Path | None = None) -> dict:
    """Load a calibrated FAST-camera pose (rvec, tvec) for one shot."""
    import json

    from vaft.machine_mapping import resolve_geometry_asset

    if pose_path is not None:
        path = Path(pose_path).expanduser()
    else:
        try:
            path = resolve_geometry_asset(f"camera_visible/pose_{int(shot)}.json")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No calibrated FAST-camera pose is packaged for shot {shot}. "
                f"Calibrated shots: {_CAMERA_VISIBLE_CALIBRATED_SHOTS}. "
                "Provide pose_path to use an external pose file."
            ) from exc
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "rvec": np.array(data["rvec"], dtype=float),
        "tvec": np.array(data["tvec"], dtype=float),
        "convention": data.get("convention"),
    }


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def _resolve_camera_frame(
    ods: Any,
    *,
    channel: int,
    detector: int,
    frame_index: int | None,
    frame_time: float | None,
) -> tuple[int, float, tuple[int, ...]]:
    """Resolve a camera_visible frame index/time by explicit index or nearest time."""
    frame_prefix = f"camera_visible.channel.{channel}.detector.{detector}.frame"
    n_frames = len(ods[frame_prefix])
    frame_times = np.asarray(
        [float(ods[f"{frame_prefix}.{i}.time"]) for i in range(n_frames)], dtype=float
    )
    if frame_index is not None and frame_time is not None:
        raise ValueError("Specify at most one of frame_index or frame_time.")
    if frame_index is not None:
        resolved_frame_index = int(frame_index)
    elif frame_time is not None:
        resolved_frame_index = _nearest_index(frame_times, frame_time)
    else:
        resolved_frame_index = 0
    resolved_frame_time = float(frame_times[resolved_frame_index])
    image_shape = np.asarray(ods[f"{frame_prefix}.{resolved_frame_index}.image_raw"]).shape
    return resolved_frame_index, resolved_frame_time, image_shape


def _resolve_equilibrium_time_slice(ods: Any, time: float) -> tuple[int, float, Any]:
    """Resolve the equilibrium time slice nearest to ``time``."""
    equilibrium_times = np.asarray(ods["equilibrium.time"], dtype=float)
    equilibrium_time_index = _nearest_index(equilibrium_times, time)
    equilibrium_time = float(equilibrium_times[equilibrium_time_index])
    time_slice = ods[f"equilibrium.time_slice.{equilibrium_time_index}"]
    return equilibrium_time_index, equilibrium_time, time_slice


def compute_camera_visible_efit_overlay(
    ods: Any,
    shot: int,
    *,
    channel: int = 0,
    detector: int = 0,
    frame_index: int | None = None,
    frame_time: float | None = None,
    theta_deg_range: tuple[float, float] = (-90.0, 90.0),
    n_theta: int = 181,
    flux_surface_levels: tuple[float, ...] = _DEFAULT_FLUX_SURFACE_LEVELS,
    pose_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Project equilibrium/wall geometry into FAST-camera pixel space for one frame.

    Reads ``camera_visible`` (frame selection), ``equilibrium`` (LCFS, magnetic
    axis, psi grid, nearest time slice), and ``wall`` (limiter outline) from
    ``ods``; forward-projects each through the calibrated pinhole camera model
    for ``shot`` (see :mod:`vaft.process.camera_geometry`). Nothing is written
    back into ``ods`` -- this returns plain arrays only.

    Select the camera frame by ``frame_index`` or nearest ``frame_time``
    (defaults to the first frame if neither is given). Flux surfaces are
    derived from the equilibrium's 2D psi grid at the requested normalized-psi
    ``flux_surface_levels`` (not the LCFS-only ``boundary.outline``, which is
    used directly as the LCFS overlay).
    """
    intrinsics = _load_camera_intrinsics(intrinsics_path)
    pose = _load_camera_pose(shot, pose_path)
    camera_matrix = intrinsics["camera_matrix"]
    dist_coeffs = intrinsics["dist_coeffs"]
    rvec = pose["rvec"]
    tvec = pose["tvec"]

    resolved_frame_index, resolved_frame_time, image_shape = _resolve_camera_frame(
        ods, channel=channel, detector=detector, frame_index=frame_index, frame_time=frame_time
    )
    equilibrium_time_index, equilibrium_time, time_slice = _resolve_equilibrium_time_slice(
        ods, resolved_frame_time
    )

    theta_rad = np.deg2rad(np.linspace(theta_deg_range[0], theta_deg_range[1], n_theta))

    def _project_rz(r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        world_cm = sweep_toroidal(r_m, z_m, theta_rad)
        pixel_uv, valid_mask = project_points(world_cm, rvec, tvec, camera_matrix, dist_coeffs)
        return pixel_uv[valid_mask]

    wall_r = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
    wall_z = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
    wall_uv = _project_rz(wall_r, wall_z)

    lcfs_r = np.asarray(time_slice["boundary.outline.r"], dtype=float)
    lcfs_z = np.asarray(time_slice["boundary.outline.z"], dtype=float)
    lcfs_uv = _project_rz(lcfs_r, lcfs_z)

    mag_r = float(time_slice["global_quantities.magnetic_axis.r"])
    mag_z = float(time_slice["global_quantities.magnetic_axis.z"])
    mag_axis_world_cm = toroidal_ring(mag_r, mag_z, theta_rad)
    mag_axis_uv_all, mag_axis_valid = project_points(
        mag_axis_world_cm, rvec, tvec, camera_matrix, dist_coeffs
    )
    magnetic_axis_uv = mag_axis_uv_all[mag_axis_valid]

    flux_surfaces_uv: dict[float, np.ndarray] = {}
    if flux_surface_levels:
        R_grid = np.asarray(time_slice["profiles_2d.0.grid.dim1"], dtype=float)
        Z_grid = np.asarray(time_slice["profiles_2d.0.grid.dim2"], dtype=float)
        psi_grid = _read_psi_grid(time_slice, R_grid, Z_grid)
        psi_axis = float(time_slice["global_quantities.psi_axis"])
        psi_boundary = float(time_slice["global_quantities.psi_boundary"])
        contours = extract_flux_surface_contours(
            psi_grid, R_grid, Z_grid, psi_axis, psi_boundary, flux_surface_levels
        )
        for level, segments in contours.items():
            projected_segments = [_project_rz(r_pts, z_pts) for r_pts, z_pts in segments]
            projected_segments = [seg for seg in projected_segments if seg.size > 0]
            flux_surfaces_uv[level] = (
                np.concatenate(projected_segments, axis=0)
                if projected_segments
                else np.empty((0, 2))
            )

    return {
        "frame_index": resolved_frame_index,
        "frame_time": resolved_frame_time,
        "equilibrium_time_index": equilibrium_time_index,
        "equilibrium_time": equilibrium_time,
        "image_shape": image_shape,
        "wall_uv": wall_uv,
        "lcfs_uv": lcfs_uv,
        "magnetic_axis_uv": magnetic_axis_uv,
        "flux_surfaces_uv": flux_surfaces_uv,
    }


# =====================================================================

# Magnetic field-line tracing from an equilibrium time slice, and its
# projection onto FAST-camera pixel space. Read-only, like the EFIT overlay
# above: computes and returns arrays, never writes back into ods.
# =====================================================================



def _read_psi_grid(time_slice: Any, R_grid: np.ndarray, Z_grid: np.ndarray) -> np.ndarray:
    """Read ``profiles_2d.0.psi``, asserting the ``(len(R), len(Z))`` orientation.

    ``vaft.data.eqdsk.to_omas`` (the only writer this pipeline uses) reliably
    stores this as ``(nw, nh) = (R.size, Z.size)`` -- see its
    ``prof2d["psi"] = PSIRZ.reshape(nw, nh)``. A shape-equality guess at
    whether to transpose is genuinely ambiguous for a square grid (EFIT's
    common 129x129/65x65 default), and silently transposing a
    correctly-oriented square array corrupts it without raising: verified
    against this exact 129x129 grid, psi_N at the magnetic axis reads
    ~0.0000 untransposed vs. 1.49 (badly wrong) if transposed. Raise instead
    of guessing.
    """
    psi_grid = np.asarray(time_slice["profiles_2d.0.psi"], dtype=float)
    expected_shape = (R_grid.size, Z_grid.size)
    if psi_grid.shape != expected_shape:
        raise ValueError(
            f"profiles_2d.0.psi has shape {psi_grid.shape}, expected {expected_shape} "
            "= (len(grid.dim1), len(grid.dim2)) per vaft.data.eqdsk.to_omas's convention."
        )
    return psi_grid


def _equilibrium_field_slice_data(time_slice: Any) -> dict[str, np.ndarray]:
    """Read psi grid + F(psi) profile data needed to build a field interpolator."""
    R_grid = np.asarray(time_slice["profiles_2d.0.grid.dim1"], dtype=float)
    Z_grid = np.asarray(time_slice["profiles_2d.0.grid.dim2"], dtype=float)
    psi_grid = _read_psi_grid(time_slice, R_grid, Z_grid)
    # The field interpolator forms B_R/B_Z from grad(psi)/R and expects
    # Wb/rad (issue #236); psi_1d must stay in the same frame for F(psi).
    _psi_factor = ods_psi_to_wb_per_radian_factor(time_slice)
    return {
        "psi_grid": psi_grid * _psi_factor,
        "R_grid": R_grid,
        "Z_grid": Z_grid,
        "psi_1d": np.asarray(time_slice["profiles_1d.psi"], dtype=float) * _psi_factor,
        "f_1d": np.asarray(time_slice["profiles_1d.f"], dtype=float),
    }


def compute_field_line_trace(
    ods: Any,
    *,
    r0: float,
    z0: float,
    phi0: float = 0.0,
    time: float | None = None,
    time_index: int | None = None,
    dphi_deg: float = 1.0,
    max_length_m: float = 50.0,
    direction: str = "forward",
    use_wall_boundary: bool = True,
) -> dict[str, Any]:
    """Trace a magnetic field line from ``(r0, z0, phi0)`` using an equilibrium time slice.

    Reads the psi grid and ``F(psi) = R*B_phi`` profile from
    ``ods['equilibrium.time_slice.N']`` (nearest to ``time``, or ``time_index``
    directly, or the first slice if neither is given) and integrates
    ``dR/dphi = R*B_R/B_phi``, ``dZ/dphi = R*B_Z/B_phi`` with fixed-step RK4
    (see :func:`vaft.process.equilibrium.trace_field_line` for the full
    integration/termination contract). If ``use_wall_boundary`` and
    ``ods['wall...outline']`` is present, the trace also terminates on
    leaving the limiter polygon. Returns plain arrays; nothing is written
    back into ``ods``.
    """
    if time_index is not None and time is not None:
        raise ValueError("Specify at most one of time_index or time.")
    if time_index is not None:
        equilibrium_time_index = int(time_index)
        equilibrium_time = float(ods["equilibrium.time"][equilibrium_time_index])
        time_slice = ods[f"equilibrium.time_slice.{equilibrium_time_index}"]
    elif time is not None:
        equilibrium_time_index, equilibrium_time, time_slice = _resolve_equilibrium_time_slice(ods, time)
    else:
        equilibrium_time_index, equilibrium_time, time_slice = _resolve_equilibrium_time_slice(
            ods, float(ods["equilibrium.time"][0])
        )

    field_data = _equilibrium_field_slice_data(time_slice)
    b_field = make_equilibrium_field_interpolator(
        field_data["R_grid"], field_data["Z_grid"], field_data["psi_grid"],
        field_data["psi_1d"], field_data["f_1d"],
    )

    wall_r = wall_z = None
    if use_wall_boundary:
        try:
            wall_r = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
            wall_z = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
        except Exception:
            wall_r = wall_z = None

    r_bounds = (float(field_data["R_grid"].min()), float(field_data["R_grid"].max()))
    z_bounds = (float(field_data["Z_grid"].min()), float(field_data["Z_grid"].max()))

    trace = trace_field_line(
        r0, z0, phi0, b_field,
        dphi=np.deg2rad(dphi_deg),
        max_length_m=max_length_m,
        direction=direction,
        wall_r=wall_r, wall_z=wall_z,
        r_bounds=r_bounds, z_bounds=z_bounds,
    )

    return {
        "equilibrium_time_index": equilibrium_time_index,
        "equilibrium_time": equilibrium_time,
        "start_point": {"r0": float(r0), "z0": float(z0), "phi0": float(phi0)},
        "dphi_deg": float(dphi_deg),
        "max_length_m": float(max_length_m),
        "direction": direction,
        **trace,
    }


def compute_camera_visible_field_line_overlay(
    ods: Any,
    shot: int,
    *,
    r0: float,
    z0: float,
    phi0: float = 0.0,
    channel: int = 0,
    detector: int = 0,
    frame_index: int | None = None,
    frame_time: float | None = None,
    dphi_deg: float = 1.0,
    max_length_m: float = 50.0,
    direction: str = "forward",
    use_wall_boundary: bool = True,
    pose_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Project a traced magnetic field line into FAST-camera pixel space for one frame.

    Combines :func:`compute_field_line_trace` (equilibrium time slice nearest
    the resolved camera frame's time) with the calibrated pinhole projection
    used by :func:`compute_camera_visible_efit_overlay`. Nothing is written
    back into ``ods``.
    """
    intrinsics = _load_camera_intrinsics(intrinsics_path)
    pose = _load_camera_pose(shot, pose_path)

    resolved_frame_index, resolved_frame_time, image_shape = _resolve_camera_frame(
        ods, channel=channel, detector=detector, frame_index=frame_index, frame_time=frame_time
    )

    trace = compute_field_line_trace(
        ods, r0=r0, z0=z0, phi0=phi0, time=resolved_frame_time,
        dphi_deg=dphi_deg, max_length_m=max_length_m, direction=direction,
        use_wall_boundary=use_wall_boundary,
    )

    world_cm = trajectory_world_points(trace["R"], trace["Z"], trace["phi"])
    pixel_uv, valid_mask = project_points(
        world_cm, pose["rvec"], pose["tvec"], intrinsics["camera_matrix"], intrinsics["dist_coeffs"]
    )
    # Compacting with pixel_uv[valid_mask] would discard invalid samples'
    # positions in the trajectory, so a renderer drawing the remaining points
    # as one connected polyline would join two visible runs across a gap
    # (behind the camera / outside the distortion guard) with a fabricated
    # straight segment. Keep the full-length array and mark invalid samples
    # as NaN instead -- matplotlib breaks a plotted line at NaN, so the
    # discontinuity is preserved without any renderer-side changes.
    field_line_uv = pixel_uv.copy()
    field_line_uv[~valid_mask] = np.nan

    return {
        "frame_index": resolved_frame_index,
        "frame_time": resolved_frame_time,
        "equilibrium_time_index": trace["equilibrium_time_index"],
        "equilibrium_time": trace["equilibrium_time"],
        "image_shape": image_shape,
        "field_line_uv": field_line_uv,
        "field_line_valid": valid_mask,
        "trace": trace,
    }


def _point_label(rz, index: int) -> str:
    point = rz[index]
    if isinstance(point, (list, tuple)):
        return f"(r={point[0]:.3f}, z={point[1]:.3f})"
    return str(point)


def _plot_vacuum_field_quantities(time_arr, psi_out, br_out, bz_out, rz, mode):
    """Render the vacuum-field diagnostic panels through ``vaft.plot``."""
    from vaft.plot import LineSeries, Panels, Series, render_panels

    n_points = psi_out.shape[1] if psi_out.ndim == 2 else 1
    panels = []
    for values, label, unit in (
        (psi_out, "psi_out", "Wb"),
        (br_out, "B_r", "T"),
        (bz_out, "B_z", "T"),
    ):
        traces = []
        for index in range(n_points):
            column = values[:, index] if values.ndim == 2 else values
            traces.append(
                Series(x=time_arr, y=column, label=_point_label(rz, index))
            )
        panels.append(
            LineSeries(
                series=tuple(traces), x_label="Time", x_unit="s",
                y_label=label, y_unit=unit,
            )
        )
    return render_panels(
        Panels(
            models=tuple(panels),
            suptitle=f"Vacuum Field Quantities at Each Time Step (Mode: {mode})",
        ),
        figsize=(10, 8),
    )
