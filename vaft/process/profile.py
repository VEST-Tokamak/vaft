import vaft
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from uncertainties import unumpy
from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

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
    Te_order,
    Ne_order,
    uncertainty_option=1,
    rho_points=100,
    fitting_function_opt='polynomial',
):
    """
    Fit electron temperature and density profiles from Thomson scattering data.

    This function fits the electron temperature (Te) and electron density (Ne) profiles
    using the specified fitting function (polynomial or exponential) at a given time.
    The fitted profiles are functions of normalized poloidal flux (rho).

    Parameters:
        ods (ODS): The OMAS data structure containing Thomson scattering data.
        time_ms (float): The time (in milliseconds) at which to extract data and fit profiles.
        mapped_rho_position (array-like): The mapped rho positions for each Thomson scattering point.
        Te_order (int): The order of the polynomial/exponential function for Te fitting.
        Ne_order (int): The order of the polynomial/exponential function for Ne fitting.
        uncertainty_option (int, optional): If 1, include uncertainties in the fit;
            else, ignore uncertainties.
        rho_points (int, optional): Number of rho points to evaluate the fitted profiles.
        fitting_function_opt (str, optional): Type of fitting function ('polynomial' or 'exponential').

    Returns:
        tuple: A tuple containing:
            - n_e_function (callable): Function to compute fitted electron density at any rho.
            - T_e_function (callable): Function to compute fitted electron temperature at any rho.
            - coeffs_ne (numpy.ndarray): Coefficients of the fitted Ne function.
            - coeffs_te (numpy.ndarray): Coefficients of the fitted Te function.
            - n_e_rho (numpy.ndarray): Fitted Ne values at evaluated rho points.
            - T_e_rho (numpy.ndarray): Fitted Te values at evaluated rho points.
    """
    from scipy.optimize import curve_fit

    # Extract data and uncertainties
    t_e = []
    n_e = []
    t_e_std = []
    n_e_std = []

    # Find time index for the specific time
    time_index = np.where(ods['thomson_scattering.time'] == (time_ms) / 1e3)[0][0]

    num_channels = len(ods['thomson_scattering.channel'])
    for i in range(num_channels):
        channel = ods['thomson_scattering.channel'][i]
        t_e_data = channel['t_e.data'][time_index]
        n_e_data = channel['n_e.data'][time_index]
        t_e_error = channel['t_e.data_error_upper'][time_index]
        n_e_error = channel['n_e.data_error_upper'][time_index]
        t_e.append(t_e_data)
        n_e.append(n_e_data)
        t_e_std.append(t_e_error)
        n_e_std.append(n_e_error)

    # Convert to numpy arrays
    t_e = np.array(t_e)
    n_e = np.array(n_e)
    t_e_std = np.array(t_e_std)
    n_e_std = np.array(n_e_std)
    rho = np.array(mapped_rho_position)

    # Ensure rho is within [0, 1]
    rho = np.clip(rho, 0, 1)

    # Normalize n_e data and uncertainties if necessary
    n_e_scale = 1e18
    n_e_norm = n_e / n_e_scale
    n_e_std_norm = n_e_std / n_e_scale

    # Define the fitting function with boundary condition at rho=1
    if fitting_function_opt == 'polynomial':
        def fitting_function(x, *coeffs):
            s = sum([coeffs[k] * x ** k for k in range(len(coeffs))])
            return (1 - x) * s

    elif fitting_function_opt == 'exponential':
        def fitting_function(x, *coeffs):
            s = sum([coeffs[k] * x ** k for k in range(len(coeffs))])
            return (1 - x) * np.exp(s)

    else:
        raise ValueError(f"Invalid fitting_function_opt: {fitting_function_opt}")

    # Prepare rho points for evaluating the profiles
    rho_eval = np.linspace(0, 1, rho_points)

    # Fit T_e profile
    initial_guess_te = np.ones(Te_order) * 0.1  # Start with small non-zero values
    if uncertainty_option == 1:  # Consider uncertainties in the fit
        coeffs_te, cov_te = curve_fit(
            fitting_function,
            rho,
            t_e,
            sigma=t_e_std,
            absolute_sigma=True,
            p0=initial_guess_te,
        )
    else:  # Do not consider uncertainties in the fit
        coeffs_te, cov_te = curve_fit(
            fitting_function,
            rho,
            t_e,
            p0=initial_guess_te,
        )
    T_e_rho = fitting_function(rho_eval, *coeffs_te)

    # Create T_e_function
    def T_e_function(rho_input):
        rho_input = np.clip(rho_input, 0, 1)
        return fitting_function(rho_input, *coeffs_te)

    # Fit n_e profile (on normalized data)
    initial_guess_ne = np.ones(Ne_order) * 0.1
    if uncertainty_option == 1:
        coeffs_ne, cov_ne = curve_fit(
            fitting_function,
            rho,
            n_e_norm,
            sigma=n_e_std_norm,
            absolute_sigma=True,
            p0=initial_guess_ne,
        )
    else:
        coeffs_ne, cov_ne = curve_fit(
            fitting_function,
            rho,
            n_e_norm,
            p0=initial_guess_ne,
        )
    n_e_rho_norm = fitting_function(rho_eval, *coeffs_ne)

    # Scale back the fitted n_e profile to original units
    n_e_rho = n_e_rho_norm * n_e_scale

    # Create n_e_function
    def n_e_function(rho_input):
        rho_input = np.clip(rho_input, 0, 1)
        n_e_norm_output = fitting_function(rho_input, *coeffs_ne)
        return n_e_norm_output * n_e_scale

    return n_e_function, T_e_function, coeffs_ne, coeffs_te, n_e_rho, T_e_rho

def plot_electron_profile_with_thomson(
    ods,
    time_ms,
    mapped_rho_position,
    n_e_function,
    T_e_function,
    n_e_coeff,
    T_e_coeff,
    save_opt = 1
):
    """
    Plot the core electron temperature and density profiles along with Thomson scattering data.

    This function plots the fitted electron temperature and density profiles as functions
    of normalized poloidal flux (rho), and overlays the measured Thomson scattering data
    with error bars.

    Parameters:
        ods (ODS): The OMAS data structure containing Thomson scattering data.
        time_ms (float): The time (in milliseconds) at which to extract data for plotting.
        mapped_rho_position (array-like): The mapped rho positions for each Thomson scattering point.
        n_e_function (callable): Function to compute fitted electron density at any rho.
        T_e_function (callable): Function to compute fitted electron temperature at any rho.
        n_e_coeff (numpy.ndarray): Coefficients of the fitted Ne function (unused in this function).
        T_e_coeff (numpy.ndarray): Coefficients of the fitted Te function (unused in this function).
    """
    # Extract data at specific time
    t_e = []
    n_e = []
    t_e_data_error_upper = []
    n_e_data_error_upper = []

    # Extract shotnumber
    shotnumber = ods['dataset_description.data_entry.pulse']

    time_index = np.where(ods['thomson_scattering.time'] == time_ms / 1e3)[0][0]

    num_channels = len(ods['thomson_scattering.channel'])
    for i in range(num_channels):
        t_e.append(ods['thomson_scattering.channel'][i]['t_e.data'][time_index])
        n_e.append(ods['thomson_scattering.channel'][i]['n_e.data'][time_index])
        t_e_data_error_upper.append(
            ods['thomson_scattering.channel'][i]['t_e.data_error_upper'][time_index]
        )
        n_e_data_error_upper.append(
            ods['thomson_scattering.channel'][i]['n_e.data_error_upper'][time_index]
        )

    # Convert to numpy arrays for consistency
    t_e = np.array(t_e)
    n_e = np.array(n_e)
    t_e_data_error_upper = np.array(t_e_data_error_upper)
    n_e_data_error_upper = np.array(n_e_data_error_upper)

    # Plot the core profiles
    rho_eval = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    axs[0].errorbar(
        mapped_rho_position,
        t_e,
        yerr=t_e_data_error_upper,
        fmt='o',
        label='Measured Te with uncertainty',
    )
    axs[0].plot(rho_eval, T_e_function(rho_eval), label='Fitted Te')
    axs[0].set_xlabel(r'$\psi_N$')
    axs[0].set_ylabel(r'$T_e$ (eV)')
    axs[0].set_title(f'shot = {shotnumber}, time = {time_ms} ms\nElectron Temperature Profile')
    axs[0].legend()

    axs[1].errorbar(
        mapped_rho_position,
        n_e,
        yerr=n_e_data_error_upper,
        fmt='o',
        label='Measured Ne with uncertainty',
    )
    axs[1].plot(rho_eval, n_e_function(rho_eval), label='Fitted Ne')
    axs[1].set_xlabel(r'$\psi_N$')
    axs[1].set_ylabel(r'$n_e$ (m$^{-3}$)')
    axs[1].set_title('Electron Density Profile')
    axs[1].legend()

    plt.tight_layout()

    if save_opt == 1:
        plt.savefig(f'thomson_scattering_profiles_{shotnumber}_{time_ms}.png')
    plt.show()

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


def plot_pressure_profile_with_geqdsk(shot, time_ms, OMFITgeq, n_e_function, T_e_function, geqdsk, save_opt = 1):
    """
    Plot the core electron temperature and density profiles along with the pressure profile from GEQDSK.

    This function plots the fitted electron temperature and density profiles as functions
    of normalized poloidal flux (rho), and overlays the pressure profile from GEQDSK.

    Parameters:
        n_e_function (callable): Function to compute fitted electron density at any rho.
        T_e_function (callable): Function to compute fitted electron temperature at any rho.
        geqdsk (dict): The GEQDSK equilibrium data containing pressure profiles.
    """
    # Extract pressure profile from GEQDSK
    psi = np.zeros(len(geqdsk['fluxSurfaces']['flux']))
    psi_N = np.zeros(len(geqdsk['fluxSurfaces']['flux']))
    pressure = np.zeros(len(geqdsk['fluxSurfaces']['flux']))

    for i in range(len(OMFITgeq['fluxSurfaces']['flux'])):
        psi[i] = OMFITgeq['fluxSurfaces']['flux'][i]['psi']
        pressure[i] = OMFITgeq['fluxSurfaces']['flux'][i]['p']
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])

    # Evaluate the fitted profiles at the same rho points
    T_e_rho = T_e_function(psi_N)
    n_e_rho = n_e_function(psi_N)

    # Calculate electron pressure from fitted profiles
    p_e_rho = n_e_rho * T_e_rho * 1.602e-19  # Convert to Pa

    # Plot the core profiles
    plt.figure(figsize=(10, 6))
    plt.plot(psi_N, p_e_rho, label='Fitted $p_e$ from TS')
    plt.plot(psi_N, pressure, label='EFIT $p$')
    plt.xlabel(r'$\psi_N$')
    plt.ylabel(r'$p$ (Pa)')
    plt.title('Electron Pressure Profile vs EFIT Total Pressure Profile, shot = {}, time = {} ms'.format(shot, time_ms))
    plt.legend()

    if save_opt == 1:
        plt.savefig(f'pressure_profile_comparison_{shot}_{time_ms}.png')
