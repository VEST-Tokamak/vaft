import vaft
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from uncertainties import unumpy
from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import curve_fit


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

    """
    Fit Thomson scattering Te and ne profiles using GP, polynomial, or exponential models.

    ## Modes:
    - fitting_function_te, fitting_function_ne ∈ {'gp', 'polynomial', 'exponential'}
    - Uses uncertainties as weights (if available)
    - Returns callable functions and profile mean/std

    ## Returns:
        n_e_function, T_e_function, rho_eval, n_e_mean, T_e_mean, n_e_std, T_e_std
    """
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

    t_e = np.array(t_e)
    n_e = np.array(n_e)
    t_e_std = np.clip(np.array(t_e_std), 1e-6, None)
    n_e_std = np.clip(np.array(n_e_std), 1e-6, None)
    rho = np.clip(np.array(mapped_rho_position).reshape(-1, 1), 0, 1)

    n_e_scale = 1e18
    n_e_norm = n_e / n_e_scale
    n_e_std_norm = n_e_std / n_e_scale
    rho_eval = np.linspace(0, 1, rho_points)

    # --- Define fitting functions ---
    def make_fit_function(mode):
        if mode == 'polynomial':
            def func(x, *coeffs):
                s = sum([coeffs[k] * x**k for k in range(len(coeffs))])
                return (1 - x) * s
        elif mode == 'exponential':
            def func(x, *coeffs):
                s = sum([coeffs[k] * x**k for k in range(len(coeffs))])
                return (1 - x) * np.exp(s)
        else:
            raise ValueError(f"Invalid fitting function: {mode}")
        return func

    # ==========================================================
    # ---------------------- Te FIT -----------------------------
    # ==========================================================
    if fitting_function_te.lower() == 'gp':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.3, length_scale_bounds=(0.05, 5.0))
        gp_Te = GaussianProcessRegressor(kernel=kernel, alpha=t_e_std**2, normalize_y=True, n_restarts_optimizer=5)
        gp_Te.fit(rho, t_e)
        T_e_rho, T_e_std = gp_Te.predict(rho_eval[:, None], return_std=True)

        def T_e_function(rho_input):
            rho_input = np.clip(rho_input, 0, 1).reshape(-1, 1)
            return np.maximum(gp_Te.predict(rho_input), 0)

        coeffs_te = None

    else:
        fitting_function_Te = make_fit_function(fitting_function_te)
        p0 = np.ones(Te_order) * 0.1
        if uncertainty_option == 1:
            coeffs_te, _ = curve_fit(
                fitting_function_Te, rho.ravel(), t_e, sigma=t_e_std, absolute_sigma=True, p0=p0
            )
        else:
            coeffs_te, _ = curve_fit(fitting_function_Te, rho.ravel(), t_e, p0=p0)
        T_e_rho = fitting_function_Te(rho_eval, *coeffs_te)
        T_e_std = np.zeros_like(T_e_rho)

        def T_e_function(rho_input):
            rho_input = np.clip(rho_input, 0, 1)
            return fitting_function_Te(rho_input, *coeffs_te)

    # ==========================================================
    # ---------------------- Ne FIT -----------------------------
    # ==========================================================
    if fitting_function_ne.lower() == 'gp':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.3, length_scale_bounds=(0.05, 5.0))
        gp_Ne = GaussianProcessRegressor(kernel=kernel, alpha=n_e_std_norm**2, normalize_y=True, n_restarts_optimizer=5)
        gp_Ne.fit(rho, n_e_norm)
        n_e_rho_norm, n_e_std_norm = gp_Ne.predict(rho_eval[:, None], return_std=True)
        n_e_rho = n_e_rho_norm * n_e_scale
        n_e_std = n_e_std_norm * n_e_scale

        def n_e_function(rho_input):
            rho_input = np.clip(rho_input, 0, 1).reshape(-1, 1)
            return np.maximum(gp_Ne.predict(rho_input) * n_e_scale, 0)

        coeffs_ne = None

    else:
        fitting_function_Ne = make_fit_function(fitting_function_ne)
        p0 = np.ones(Ne_order) * 0.1
        if uncertainty_option == 1:
            coeffs_ne, _ = curve_fit(
                fitting_function_Ne, rho.ravel(), n_e_norm, sigma=n_e_std_norm, absolute_sigma=True, p0=p0
            )
        else:
            coeffs_ne, _ = curve_fit(fitting_function_Ne, rho.ravel(), n_e_norm, p0=p0)
        n_e_rho_norm = fitting_function_Ne(rho_eval, *coeffs_ne)
        n_e_rho = n_e_rho_norm * n_e_scale
        n_e_std = np.zeros_like(n_e_rho)

        def n_e_function(rho_input):
            rho_input = np.clip(rho_input, 0, 1)
            n_e_norm_out = fitting_function_Ne(rho_input, *coeffs_ne)
            return n_e_norm_out * n_e_scale

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

    ods[f'{base}.electrons.density_thermal'] = n_e_recon.tolist()
    ods[f'{base}.electrons.temperature'] = T_e_recon.tolist()

    ods[f'{base}.ion.0.label'] = 'D+'
    ods[f'{base}.ion.0.density_thermal'] = n_e_recon.tolist()
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

