import vaft
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from uncertainties import unumpy
from omas import *
from vaft.formula import fit_profile


def equilibrium_mapping_thomson_scattering(ods, geq):
    """
    Map Thomson scattering positions to normalized poloidal flux coordinates (rho).

    This function finds the closest flux surface for each Thomson scattering measurement point
    and maps it to the corresponding normalized poloidal flux value (rho) in the equilibrium data.

    Parameters:
        ods (ODS): The OMAS data structure containing Thomson scattering positions.
        geq (dict): The equilibrium data containing flux surfaces and levels.

    Returns:
        numpy.ndarray: An array of mapped rho positions for each Thomson scattering point.
    """
    # Extract Thomson scattering positions
    r_t = ods['thomson_scattering.channel.:.position.r']
    z_t = ods['thomson_scattering.channel.:.position.z']

    # Extract flux surface levels (normalized poloidal flux values)
    flux_levels = geq['fluxSurfaces']['levels']

    # Initialize list to store mapped rho positions
    mapped_rho_position = []

    # For each Thomson scattering point, find the closest flux surface
    for r_dot, z_dot in zip(r_t, z_t):
        min_dist = float('inf')
        closest_rho = None

        # Iterate over all flux surfaces
        for i in range(len(geq['fluxSurfaces']['flux'])):
            R = geq['fluxSurfaces']['flux'][i]['R']
            Z = geq['fluxSurfaces']['flux'][i]['Z']

            # Compute distances to all points on this flux surface
            dists = np.sqrt((R - r_dot) ** 2 + (Z - z_dot) ** 2)
            min_flux_dist = np.min(dists)

            if min_flux_dist < min_dist:
                min_dist = min_flux_dist
                closest_rho = flux_levels[i]  # Normalized poloidal flux value

        mapped_rho_position.append(closest_rho)

    # Convert to numpy array
    mapped_rho_position = np.array(mapped_rho_position)

    # Ensure rho values are within [0, 1]
    mapped_rho_position = np.clip(mapped_rho_position, 0, 1)

    return mapped_rho_position

def profile_fitting_thomson_scattering(
    ods,
    time_ms,
    mapped_rho_position,
    Te_order=3,
    Ne_order=3,
    uncertainty_option=1,
    rho_points=100,
    fitting_function_te='polynomial',
    fitting_function_ne='polynomial',
    ):
    """
    Fit Thomson scattering Te and ne profiles with selectable 1D methods.

    Supported modes:
    - fitting_function_te, fitting_function_ne ∈ {'gp', 'polynomial', 'exponential', 'linear', 'core_poly_edge_exp'}

    Behavior:
    - Extracts Thomson data (Te, ne) at the requested time.
    - Fits Te(ρ) and ne(ρ) on ρ_tor_norm ∈ [0, 1] using the selected model for each.
    - If uncertainty_option == 1 and per-channel uncertainties are available, they are used as weights.
    - Edge behavior:
      - Polynomial/exponential basis includes a (1 - ρ) factor so the fitted profile tends toward 0 at ρ=1.
      - core_poly_edge_exp blends a core polynomial with an edge exponential using tanh(ρ) transition.
    - Returns callable fit functions for evaluating Te and ne on arbitrary ρ in [0, 1],
      plus the fitted mean profiles on a uniform evaluation grid.

    ## Arguments:
    - ods: OMAS data structure
    - time_ms: time in milliseconds
    - mapped_rho_position: list/array of mapped rho_tor_norm positions for Thomson channels
    - Te_order, Ne_order: polynomial order (used for polynomial/exponential/core_poly_edge_exp)
    - uncertainty_option: use uncertainties when fitting (1 = enabled)
    - rho_points: number of evaluation points on ρ in [0, 1]
    - fitting_function_te, fitting_function_ne: fit method selection for Te and ne

    ## Returns:
    - n_e_function, T_e_function: callable fit functions
    - coeffs_ne, coeffs_te: fit coefficients (None for GP/linear)
    - n_e_rho, T_e_rho: fitted profiles evaluated on rho_eval

    ## Example:
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='gp', fitting_function_ne='gp')
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='polynomial', fitting_function_ne='exponential')
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='linear', fitting_function_ne='linear')
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='core_poly_edge_exp', fitting_function_ne='core_poly_edge_exp')
    """
    # --- Extract Thomson data ---
    time_index = np.where(ods['thomson_scattering.time'] == time_ms / 1e3)[0][0]
    num_channels = len(ods['thomson_scattering.channel'])
    t_e, n_e, t_e_std, n_e_std = [], [], [], []

    for i in range(num_channels):
        ch = ods['thomson_scattering.channel'][i]
        t_e.append(ch['t_e.data'][time_index])
        n_e.append(ch['n_e.data'][time_index])
        t_e_std.append(ch['t_e.data_error_upper'][time_index])
        n_e_std.append(ch['n_e.data_error_upper'][time_index])

    t_e = np.array(t_e, dtype=float)
    n_e = np.array(n_e, dtype=float)
    t_e_std = np.clip(np.array(t_e_std, dtype=float), 1e-6, None)
    n_e_std = np.clip(np.array(n_e_std, dtype=float), 1e-6, None)
    rho = np.clip(np.array(mapped_rho_position, dtype=float).reshape(-1, 1), 0, 1)

    # density normalization
    n_e_scale = 1e18
    n_e_norm = n_e / n_e_scale
    n_e_std_norm = n_e_std / n_e_scale
    rho_eval = np.linspace(0, 1, rho_points)

    # --- Te / Ne FITS ---
    te_anchor_strength = None
    te_anchor = None
    if te_anchor_strength is not None:
        te_anchor = (np.array([1.0]), np.array([0.0]), np.array([te_anchor_strength]))

    T_e_rho, T_e_std, T_e_function_raw, coeffs_te = fit_profile(
        rho,
        t_e,
        t_e_std,
        rho_eval,
        order=Te_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_te,
        gp_anchor=te_anchor,
    )

    def T_e_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0, 1)
        return np.maximum(T_e_function_raw(x), 0.0)

    ne_anchor = None
    if fitting_function_ne.lower() == 'gp':
        ne_typ = np.nanmedian(n_e_norm[n_e_norm > 0]) if np.any(n_e_norm > 0) else 1.0
        ne_anchor_sigma_norm = max(0.01 * ne_typ, 1e-4)
        ne_anchor = (np.array([1.0]), np.array([0.0]), np.array([ne_anchor_sigma_norm]))

    n_e_rho_norm, n_e_std_norm_fit, n_e_function_raw, coeffs_ne = fit_profile(
        rho,
        n_e_norm,
        n_e_std_norm,
        rho_eval,
        order=Ne_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_ne,
        gp_anchor=ne_anchor,
    )

    n_e_rho = np.maximum(n_e_rho_norm, 0.0) * n_e_scale
    n_e_std = n_e_std_norm_fit * n_e_scale

    def n_e_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0, 1)
        y_norm = n_e_function_raw(x)
        return np.maximum(y_norm, 0.0) * n_e_scale

    return n_e_function, T_e_function, coeffs_ne, coeffs_te, n_e_rho, T_e_rho

def core_profiles(ods, time_ms, mapped_rho_position, n_e_function, T_e_function, tol_ms=0.1):
    """
    Construct and store core_profiles.profiles_1d for given Thomson scattering data.

    If a profile already exists for the same time (within tol_ms), it is replaced.

    ## Arguments:
    - ods: OMAS data structure (mutable)
    - time_ms: time in milliseconds
    - mapped_rho_position: list of mapped rho_tor_norm positions for Thomson channels
    - n_e_function, T_e_function: callable fit functions for ne, Te
    - tol_ms: tolerance in milliseconds for duplicate time detection

    ## Returns:
    - Updated ods with replaced or appended core_profiles.profiles_1d entry.
    """
    num_channels = len(ods['thomson_scattering.channel'])
    n_e_meas, T_e_meas = [], []
    t_idx = np.where(ods['thomson_scattering.time'] == time_ms / 1e3)[0][0]

    for i in range(num_channels):
        n_e_meas.append(ods[f'thomson_scattering.channel.{i}.n_e.data'][t_idx])
        T_e_meas.append(ods[f'thomson_scattering.channel.{i}.t_e.data'][t_idx])

    n_e_meas = np.array(n_e_meas)
    T_e_meas = np.array(T_e_meas)

    rho_meas = np.array(mapped_rho_position)
    rho_fit = np.linspace(0, 1, 100)

    n_e_recon = n_e_function(rho_fit)
    T_e_recon = T_e_function(rho_fit)

    # --- check for duplicate time entries ---
    existing_times = []
    if 'core_profiles.profiles_1d' in ods:
        n_profiles = len(ods['core_profiles.profiles_1d'])
        for i in range(n_profiles):
            try:
                t_existing = ods[f'core_profiles.profiles_1d.{i}.time']
                if abs(t_existing * 1000 - time_ms) < tol_ms:
                    existing_times.append(i)
            except Exception:
                continue

    # --- remove duplicates before writing ---
    for i in sorted(existing_times, reverse=True):
        ods.pop(f'core_profiles.profiles_1d.{i}')
        print(f"[INFO] Removed duplicate core_profile at {time_ms:.3f} ms (index {i})")

    # --- Determine next available index after removal ---
    next_idx = len(ods['core_profiles.profiles_1d']) if 'core_profiles.profiles_1d' in ods else 0
    base = f'core_profiles.profiles_1d.{next_idx}'

    ods[f'{base}.time'] = time_ms / 1000
    ods[f'{base}.grid.rho_tor_norm'] = rho_fit.tolist()

    # We assume all electrons are thermal electrons (because VEST is ohmically heated plasma)
    ods[f'{base}.electrons.density_thermal'] = n_e_recon.tolist()
    ods[f'{base}.electrons.density'] = n_e_recon.tolist()
    ods[f'{base}.electrons.temperature'] = T_e_recon.tolist()

    ods[f'{base}.ion.0.label'] = 'H+'
    ods[f'{base}.ion.0.density_thermal'] = n_e_recon.tolist()
    ods[f'{base}.ion.0.density'] = n_e_recon.tolist()
    ods[f'{base}.ion.0.temperature'] = T_e_recon.tolist()

    fit_base_n = f'{base}.electrons.density_fit'
    fit_base_t = f'{base}.electrons.temperature_fit'

    ods[f'{fit_base_n}.rho_tor_norm'] = rho_meas.tolist()
    ods[f'{fit_base_n}.measured'] = n_e_meas.tolist()
    ods[f'{fit_base_n}.reconstructed'] = n_e_recon.tolist()

    ods[f'{fit_base_t}.rho_tor_norm'] = rho_meas.tolist()
    ods[f'{fit_base_t}.measured'] = T_e_meas.tolist()
    ods[f'{fit_base_t}.reconstructed'] = T_e_recon.tolist()

    print(f"[UPDATED] core_profile at {time_ms:.3f} ms (index {next_idx})")
    return ods

def core_profiles_from_eq(
    ods,
    Te0_eV,
    rho_fit=None,
    tol_ms=0.1,
    eq_time_index=0,
    ):
    """
    Build synthetic core_profiles.profiles_1d from equilibrium pressure with:
        P(Pa) = 2 * ne(m^-3) * Te(eV) * e(J/eV)

    Assumptions:
      - same shape: ne = ne0*g, Te = Te0*g
      - g(rho) = sqrt(P(rho)/P(0))
      - Ti = Te is implicitly absorbed in factor 2
      - No Zeff / impurity modeling

    Reads:
      rho_src = ods['equilibrium.time_slice.<idx>.profiles_1d.rho_tor_norm']
      p_src   = ods['equilibrium.time_slice.<idx>.profiles_1d.pressure']   # Pa

    Writes:
      ods['core_profiles.profiles_1d.<next_idx>.*']
    """
    e_J_per_eV = 1.602176634e-19

    time_ms = ods['equilibrium.time'][eq_time_index] * 1e3

    if rho_fit is None:
        rho_fit = np.linspace(0.0, 1.0, 100)
    else:
        rho_fit = np.asarray(rho_fit, dtype=float)

    rho_src_path = f"equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm"
    p_src_path   = f"equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure"

    rho_src = np.asarray(ods[rho_src_path], dtype=float)
    p_src   = np.asarray(ods[p_src_path], dtype=float)  # Pa

    if rho_src.ndim != 1 or p_src.ndim != 1:
        raise ValueError("Expected 1D rho_tor_norm and 1D pressure at the selected time_slice.")

    order = np.argsort(rho_src)
    rho_src = rho_src[order]
    p_src = p_src[order]

    P_fit = np.interp(rho_fit, rho_src, p_src)

    if np.any(~np.isfinite(P_fit)):
        raise ValueError("Pressure profile contains NaN/Inf.")
    if P_fit[0] <= 0:
        raise ValueError("Pressure at rho=0 must be > 0 to define sqrt shape.")

    g = np.sqrt(np.clip(P_fit / P_fit[0], 0.0, None))

    Te = Te0_eV * g  # eV

    # ---- FIX: include eV->J ----
    ne0 = P_fit[0] / (2.0 * Te0_eV * e_J_per_eV)  # m^-3
    ne = ne0 * g

    if np.any(ne < 0) or np.any(Te < 0):
        raise ValueError("Generated ne/Te has negative values (check pressure and Te0_eV).")

    # ---- remove duplicates ----
    existing_idxs = []
    if "core_profiles.profiles_1d" in ods:
        for i in range(len(ods["core_profiles.profiles_1d"])):
            try:
                t_existing = ods[f"core_profiles.profiles_1d.{i}.time"]  # s
                if abs(t_existing * 1000.0 - time_ms) < tol_ms:
                    existing_idxs.append(i)
            except Exception:
                continue

    for i in sorted(existing_idxs, reverse=True):
        ods.pop(f"core_profiles.profiles_1d.{i}")
        print(f"[INFO] Removed duplicate core_profile at {time_ms:.3f} ms (index {i})")

    next_idx = len(ods["core_profiles.profiles_1d"]) if "core_profiles.profiles_1d" in ods else 0
    base = f"core_profiles.profiles_1d.{next_idx}"

    ods[f"{base}.time"] = time_ms / 1000.0
    ods[f"{base}.grid.rho_tor_norm"] = rho_fit.tolist()

    ods[f"{base}.electrons.density_thermal"] = ne.tolist()
    ods[f"{base}.electrons.density"] = ne.tolist()
    ods[f"{base}.electrons.temperature"] = Te.tolist()

    ods[f"{base}.ion.0.label"] = "H+"
    ods[f"{base}.ion.0.density_thermal"] = ne.tolist()
    ods[f"{base}.ion.0.density"] = ne.tolist()
    ods[f"{base}.ion.0.temperature"] = Te.tolist()

    print(f"[UPDATED] core_profile from eq pressure (Pa) at {time_ms:.3f} ms "
          f"(index {next_idx}), eq_time_slice={eq_time_index}")
    return ods

def core_profiles_from_eq_ratio(
    ods,
    C_ne_over_Te,   # density / temperature ratio
    rho_fit=None,
    tol_ms=0.1,
    eq_time_index=0,
    ):
    """
    Build synthetic core_profiles from equilibrium pressure using:
        n_e = C * sqrt(f)
        T_e = sqrt(f)
        P(Pa) = 2 * n_e * T_e * e

    where f(rho) = P(rho) / P(0)

    Parameters
    ----------
    C_ne_over_Te : float
        Density / temperature ratio (m^-3 / eV)
    """

    e_J = 1.602176634e-19
    time_ms = ods['equilibrium.time'][eq_time_index] * 1e3

    if rho_fit is None:
        rho_fit = np.linspace(0, 1, 100)

    rho_src = np.asarray(
        ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm']
    )
    p_src = np.asarray(
        ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure']
    )

    order = np.argsort(rho_src)
    rho_src, p_src = rho_src[order], p_src[order]

    P_fit = np.interp(rho_fit, rho_src, p_src)

    if P_fit[0] <= 0:
        raise ValueError("Invalid pressure profile")

    # --- shape ---
    f = P_fit / P_fit[0]

    # --- build profiles ---
    Te = np.sqrt(f)                 # eV (relative)
    ne = C_ne_over_Te * Te          # m^-3

    # --- scale to absolute pressure ---
    scale = P_fit[0] / (2 * C_ne_over_Te * e_J)
    Te *= np.sqrt(scale)
    ne *= np.sqrt(scale)

    # --- remove duplicates ---
    existing = []
    if 'core_profiles.profiles_1d' in ods:
        for i in range(len(ods['core_profiles.profiles_1d'])):
            t = ods[f'core_profiles.profiles_1d.{i}.time']
            if abs(t * 1000 - time_ms) < tol_ms:
                existing.append(i)
    for i in reversed(existing):
        ods.pop(f'core_profiles.profiles_1d.{i}')

    next_idx = len(ods['core_profiles.profiles_1d'])
    base = f'core_profiles.profiles_1d.{next_idx}'

    ods[f'{base}.time'] = time_ms / 1000
    ods[f'{base}.grid.rho_tor_norm'] = rho_fit.tolist()

    ods[f'{base}.electrons.density'] = ne.tolist()
    ods[f'{base}.electrons.density_thermal'] = ne.tolist()
    ods[f'{base}.electrons.temperature'] = Te.tolist()

    ods[f'{base}.ion.0.label'] = 'H+'
    ods[f'{base}.ion.0.density'] = ne.tolist()
    ods[f'{base}.ion.0.density_thermal'] = ne.tolist()
    ods[f'{base}.ion.0.temperature'] = Te.tolist()

    print(f"[UPDATED] core_profile from eq (ratio-fixed) at {time_ms:.2f} ms")
    return ods

def export_electron_profile_txt(
    n_e_function,
    T_e_function,
    n_e_coeff,
    T_e_coeff,
    rho_points=100,
    filename='electron_profiles.txt',
    ):
    """
    Export the fitted electron temperature and density profiles to a text file.

    This function evaluates the fitted electron temperature and density profiles at
    a specified number of rho points and exports the results to a text file.

    Parameters:
        n_e_function (callable): Function to compute fitted electron density at any rho.
        T_e_function (callable): Function to compute fitted electron temperature at any rho.
        n_e_coeff (numpy.ndarray): Coefficients of the fitted Ne function.
        T_e_coeff (numpy.ndarray): Coefficients of the fitted Te function.
        rho_points (int, optional): Number of rho points to evaluate the fitted profiles.
        filename (str, optional): The name of the text file to save the profiles to.
    """
    rho_eval = np.linspace(0, 1, rho_points)
    n_e_rho = n_e_function(rho_eval)
    T_e_rho = T_e_function(rho_eval)

    with open(filename, 'w') as f:
        f.write('psi_N, T_e [eV], n_e [m-3]\n')
        for rho, T_e, n_e in zip(rho_eval, T_e_rho, n_e_rho):
            f.write(f'{rho}, {T_e}, {n_e}\n')
