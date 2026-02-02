from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d
from omas import *
import logging

logger = logging.getLogger(__name__)
from vaft.omas.process_wrapper import compute_volume_averaged_pressure
from vaft.process import compute_response_matrix
from vaft.process.equilibrium import psi_to_RZ, volume_average
from vaft.formula import magnetic_shear, ballooning_alpha_from_p_B_R
from vaft.formula.equilibrium import (
    loss_power_from_p_heat_dWdt_p_rad,
    heating_power_from_p_ohm_p_aux,
    ohmic_heating_power_from_I_p_V_res,
    loop_voltage_from_total_flux,
    inductive_voltage_from_dW_magdt_I_p,
    bremsstrahlung_power_density_from_Z_eff_n_e_T_e,
    bremsstrahlung_power_density_from_T_e_p_Z_eff,
    cyclotron_radiation_power_from_z_eff_n_e_t_e,
    line_radiation_power_from_z_eff_n_e_t_e,
    radiation_power_from_p_brem_p_cyc_p_line,
    stored_energy_from_p_V,
    confinement_time_from_engineering_parameters,
    confinement_time_from_P_loss_W_th,
    inverse_aspect_ratio_from_a_R,
    confinement_factor_from_tau_E_exp_tau_E_IPB89y2,
)

def compute_magnetic_shear(ods, time_slice: slice) -> ndarray:
    """
    Compute magnetic shear from ODS
    """
    r = ods['equilibrium']['time_slice'][time_slice]['profiles_1d']['r']
    q = ods['equilibrium']['time_slice'][time_slice]['profiles_1d']['q']
    return magnetic_shear(r, q)

# Obsolete function
# def compute_ballooning_alpha(ods, time_slice: slice) -> ndarray:
#     """
#     Compute ballooning alpha from ODS
#     """
#     V = ods['equilibrium']['profiles_1d']['V']
#     R = ods['equilibrium']['profiles_1d']['R']
#     p = ods['equilibrium']['profiles_1d']['p']
#     psi = ods['equilibrium']['profiles_1d']['psi']
#     return ballooning_alpha_from_p_B(V, R, p, psi)

# def compute_power_balance

# returen
# P_heat, P_ohmic, P_loss_total, P_loss_boundary, dWdt, P_rad 
# (Ignore NBI related parameters such as P_aux, P_CX, P_orbit)

def compute_tau_E_engineering_parameters(ods, time_slice: int, 
                                         Z_eff: float = 2.0, M: float = 1.0) -> Dict[str, float]:
    """
    Compute engineering parameters required for confinement time scaling law.
    
    Uses compute_power_balance and volume_average functions to get required parameters.
    
    Parameters
    ----------
    ods : ODS
        OMAS data structure
    time_slice : int
        Time slice index for equilibrium
    Z_eff : float, optional
        Effective charge number, by default 2.0 (not used in calculation, kept for compatibility)
    M : float, optional
        Average ion mass [amu], by default 1.0
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing engineering parameters:
        - I_p: Plasma current [A]
        - B_t: Toroidal magnetic field [T]
        - P_loss: Loss power [W]
        - n_e: Volume-averaged electron density [m^-3]
        - R: Major radius [m]
        - epsilon: Inverse aspect ratio [-]
        - kappa: Elongation [-]
        - M: Average ion mass [amu]
    """
    eq_ts = ods['equilibrium.time_slice'][time_slice]
    
    # update boundary
    from vaft.omas.update import update_equilibrium_boundary
    update_equilibrium_boundary(ods)

    # Get engineering parameters from equilibrium global_quantities
    I_p = abs(float(eq_ts['global_quantities.ip']))  # [A]

    # Get toroidal magnetic field at vessel ref position
    R_ref = 0.4 # [m] VEST reference
    B_t_axis = float(eq_ts['global_quantities.magnetic_axis.b_field_tor'])
    R_axis = float(eq_ts['global_quantities.magnetic_axis.r'])
    B_t = abs(B_t_axis * R_axis / R_ref) # [T]

    
    R = float(eq_ts['boundary.geometric_axis.r'])  # [m]
    a = float(eq_ts['boundary.minor_radius'])  # [m]
    kappa = float(eq_ts['boundary.elongation'])  # [-]
    epsilon = inverse_aspect_ratio_from_a_R(a, R)  # [-]

    # Compute P_loss from compute_power_balance
    power_balance = compute_power_balance(ods)
    # Find the index corresponding to the requested time_slice
    eq_time = float(eq_ts.get('time', time_slice))
    time_array = np.asarray(power_balance['time'])
    time_idx = int(np.argmin(np.abs(time_array - eq_time)))
    P_loss = float(power_balance['P_loss'][time_idx])  # [W]
    
    # Compute volume-averaged electron density
    # Find matching core profile time slice
    cp_idx = None
    if 'core_profiles.profiles_1d' in ods:
        eq_time = float(eq_ts.get('time', time_slice))
        min_time_diff = float('inf')
        for idx in range(len(ods['core_profiles.profiles_1d'])):
            cp_ts = ods['core_profiles.profiles_1d'][idx]
            cp_time = float(cp_ts.get('time', idx))
            time_diff = abs(cp_time - eq_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                cp_idx = idx
    
    if cp_idx is None:
        raise ValueError(f"No matching core profile found for equilibrium time_slice[{time_slice}]")
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    
    if 'electrons.density' not in cp_ts:
        raise ValueError(f"electrons.density not found in core_profiles.profiles_1d[{cp_idx}]")
    
    n_e_profile = np.asarray(cp_ts['electrons.density'], float)  # [m^-3]
    
    # Compute volume average of n_e using proper coordinate mapping
    # Get equilibrium 2D grid
    R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
    Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
    psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
    psi_axis = float(eq_ts['global_quantities.psi_axis'])
    psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
    
    # Get equilibrium profiles_1d for coordinate conversion
    eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=time_slice)
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
    
    if 'psi_norm' in eq_profiles_1d:
        psi_norm_1d = np.asarray(eq_profiles_1d['psi_norm'], float)
        
        # Get core profile coordinate (rho_tor_norm)
        grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
        if 'rho_tor_norm' in grid:
            rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
            
            # Get equilibrium rho_tor_norm for mapping
            if 'rho_tor_norm' in eq_profiles_1d:
                rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
                
                # Interpolate n_e from core profile rho_tor_norm to equilibrium psi_norm
                interp_func = interp1d(rho_tor_norm_cp, n_e_profile,
                                     kind='linear',
                                     bounds_error=False,
                                     fill_value=(n_e_profile[0], n_e_profile[-1]))
                n_e_psi_norm = interp_func(rho_tor_norm_eq)
                
                # Map to 2D grid and compute volume average
                n_e_RZ, psiN_RZ = psi_to_RZ(psi_norm_1d, n_e_psi_norm, psi_RZ, psi_axis, psi_lcfs)
                n_e_vol_avg, _ = volume_average(n_e_RZ, psiN_RZ, R_grid, Z_grid)  # [m^-3]
                # Ensure scalar
                if isinstance(n_e_vol_avg, np.ndarray):
                    n_e_vol_avg = float(n_e_vol_avg[0] if len(n_e_vol_avg) > 0 else n_e_vol_avg)
                else:
                    n_e_vol_avg = float(n_e_vol_avg)
            else:
                # Fallback: use simple average
                n_e_vol_avg = float(np.mean(n_e_profile))  # [m^-3]
        else:
            # Fallback: use simple average
            n_e_vol_avg = float(np.mean(n_e_profile))  # [m^-3]
    else:
        # Fallback: use simple average
        n_e_vol_avg = float(np.mean(n_e_profile))  # [m^-3]
    
    # Validate input parameters before computing
    if not np.isfinite(I_p) or I_p <= 0:
        raise ValueError(f"Invalid I_p: {I_p} for time_slice[{time_slice}]")
    if not np.isfinite(B_t) or B_t <= 0:
        raise ValueError(f"Invalid B_t: {B_t} for time_slice[{time_slice}]. "
                        f"Check equilibrium data at time={eq_time}")
    if not np.isfinite(P_loss) or P_loss <= 0:
        raise ValueError(f"Invalid P_loss: {P_loss} for time_slice[{time_slice}] at time={eq_time}. "
                        f"Check power_balance data. time_idx={time_idx}, "
                        f"power_balance['P_loss'] shape={np.asarray(power_balance['P_loss']).shape}")
    if not np.isfinite(n_e_vol_avg) or n_e_vol_avg <= 0:
        raise ValueError(f"Invalid n_e_vol_avg: {n_e_vol_avg}")
    if not np.isfinite(M) or M <= 0:
        raise ValueError(f"Invalid M: {M}")
    if not np.isfinite(R) or R <= 0:
        raise ValueError(f"Invalid R: {R}")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"Invalid epsilon: {epsilon}")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"Invalid kappa: {kappa}")
    
    return {
        'I_p': I_p,
        'B_t': B_t,
        'P_loss': P_loss,
        'n_e': n_e_vol_avg,
        'R': R,
        'epsilon': epsilon,
        'kappa': kappa,
        'M': M,
    }


def compute_tau_E_scaling(ods, time_slice: int, scaling: str = "IBP98y2", 
                          Z_eff: float = 2.0, M: float = 1.0) -> float:
    """
    Compute confinement time from scaling law using engineering parameters.
    
    Uses compute_tau_E_engineering_parameters to get required parameters and
    applies the scaling law.
    
    Parameters
    ----------
    ods : ODS
        OMAS data structure
    time_slice : int
        Time slice index for equilibrium
    scaling : str, optional
        Scaling law name, by default "IBP98y2"
    Z_eff : float, optional
        Effective charge number, by default 2.0
    M : float, optional
        Average ion mass [amu], by default 1.0
    
    Returns
    -------
    float
        Confinement time from scaling law [s]
    """
    # Get engineering parameters
    eng_params = compute_tau_E_engineering_parameters(ods, time_slice, Z_eff=Z_eff, M=M)
    
    # Compute confinement time from scaling law
    tau_E = confinement_time_from_engineering_parameters(
        I_p=eng_params['I_p'], 
        B_t=eng_params['B_t'], 
        P_loss=eng_params['P_loss'], 
        n_e=eng_params['n_e'],
        M=eng_params['M'], 
        R=eng_params['R'], 
        epsilon=eng_params['epsilon'], 
        kappa=eng_params['kappa'], 
        scaling=scaling
    )  # [s]
    
    # Handle complex numbers (should not happen with valid inputs, but handle gracefully)
    if isinstance(tau_E, complex):
        if tau_E.imag != 0:
            raise ValueError(f"Confinement time calculation resulted in complex number: {tau_E}. Check input parameters.")
        tau_E = tau_E.real
    
    # Ensure return value is scalar
    if isinstance(tau_E, np.ndarray):
        tau_E = float(tau_E[0] if len(tau_E) > 0 else tau_E)
    elif not isinstance(tau_E, (int, float, np.number)):
        raise TypeError(f"tau_E is not a numeric type: {type(tau_E)}, value: {tau_E}")
    else:
        tau_E = float(tau_E)
    
    # Check for invalid results
    if not np.isfinite(tau_E) or tau_E <= 0:
        raise ValueError(f"Invalid confinement time result: {tau_E}. Inputs: I_p={eng_params['I_p']}, B_t={eng_params['B_t']}, P_loss={eng_params['P_loss']}, n_e={eng_params['n_e']}, M={eng_params['M']}, R={eng_params['R']}, epsilon={eng_params['epsilon']}, kappa={eng_params['kappa']}, scaling={scaling}")
    
    return tau_E

def compute_tau_E_exp(ods, time_slice: int, Z_eff: float = 2.0) -> float:
    """
    Compute experimental confinement time from stored energy and loss power.
    
    Formula: τ_E,exp = W_th / P_loss
    
    Parameters
    ----------
    ods : ODS
        OMAS data structure
    time_slice : int
        Time slice index for equilibrium
    Z_eff : float, optional
        Effective charge number, by default 2.0
    
    Returns
    -------
    float
        Experimental confinement time [s]
    """
    eq_ts = ods['equilibrium.time_slice'][time_slice]
    
    # Get stored thermal energy W_th using compute_volume_averaged_pressure (core_profiles option)
    from vaft.omas.update import update_equilibrium_global_quantities_volume
    
    # Ensure volume is computed
    try:
        update_equilibrium_global_quantities_volume(ods, time_slice=time_slice)
    except Exception as e:
        logger.warning(f"Could not update volume: {e}")
    
    # Compute volume-averaged pressure from core_profiles
    try:
        p_vol_avg = compute_volume_averaged_pressure(ods, time_slice=time_slice, option='core_profiles')
        # p_vol_avg is an array, get the value for this time_slice
        if isinstance(p_vol_avg, np.ndarray):
            if len(p_vol_avg) == 1:
                p_vol_avg_val = float(p_vol_avg[0])
            else:
                p_vol_avg_val = float(p_vol_avg[time_slice])
        else:
            p_vol_avg_val = float(p_vol_avg)
        
        # Get plasma volume
        volume = float(eq_ts.get('global_quantities.volume'))
        
        # Calculate thermal energy: W_th = p_vol_average * 3/2 * volume
        W_th = p_vol_avg_val * (3.0 / 2.0) * volume  # [J]
    except Exception as e:
        raise ValueError(f"Could not determine stored thermal energy for time_slice[{time_slice}]: {e}")
    
    # Compute P_loss from compute_power_balance
    power_balance = compute_power_balance(ods)
    # Find the index corresponding to the requested time_slice
    eq_time = float(eq_ts.get('time', time_slice))
    time_array = np.asarray(power_balance['time'])
    time_idx = int(np.argmin(np.abs(time_array - eq_time)))
    P_loss = float(power_balance['P_loss'][time_idx])  # [W]
    
    # Validate inputs
    if not np.isfinite(P_loss) or P_loss <= 0:
        raise ValueError(f"Invalid P_loss: {P_loss}")
    if not np.isfinite(W_th) or W_th <= 0:
        raise ValueError(f"Invalid W_th: {W_th}")
    
    # Compute experimental confinement time
    tau_E_exp = confinement_time_from_P_loss_W_th(P_loss, W_th)  # [s]
    
    # Handle complex numbers
    if isinstance(tau_E_exp, complex):
        if tau_E_exp.imag != 0:
            raise ValueError(f"Confinement time calculation resulted in complex number: {tau_E_exp}. Check input parameters.")
        tau_E_exp = tau_E_exp.real
    
    # Ensure return value is scalar
    if isinstance(tau_E_exp, np.ndarray):
        tau_E_exp = float(tau_E_exp[0] if len(tau_E_exp) > 0 else tau_E_exp)
    elif not isinstance(tau_E_exp, (int, float, np.number)):
        raise TypeError(f"tau_E_exp is not a numeric type: {type(tau_E_exp)}, value: {tau_E_exp}")
    else:
        tau_E_exp = float(tau_E_exp)
    
    # Check for invalid results
    if not np.isfinite(tau_E_exp) or tau_E_exp <= 0:
        raise ValueError(f"Invalid experimental confinement time result: {tau_E_exp}. Inputs: P_loss={P_loss}, W_th={W_th}")
    
    return tau_E_exp

def compute_voltage_consumption(
    ods: ODS,
    time_slice: Optional[int] = None
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Compute loop / inductive / resistive voltage time series using:

    - V_loop = d/dt ( 2π ψ_boundary )
    - V_ind  = (dW_mag/dt) / I_p
    - V_res  = V_loop - V_ind

    Notes
    -----
    - Uses magnetic energy W_mag computed from 2D psi via
      `vaft.omas.process_wrapper.compute_magnetic_energy`.
    - If `time_slice` is provided, returns arrays of length 1.

    Returns
    -------
    time : ndarray
    V_loop : ndarray
    V_ind : ndarray
    V_res : ndarray
    """
    from vaft.omas.process_wrapper import compute_magnetic_energy

    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")

    n_eq = len(ods['equilibrium.time_slice'])

    if time_slice is None:
        idxs = list(range(n_eq))
    else:
        idx = int(time_slice)
        if idx < 0 or idx >= n_eq:
            raise IndexError(f"time_slice {idx} is out of bounds for equilibrium.time_slice")
        idxs = [idx]

    # Time, psi_boundary, Ip arrays
    t = np.zeros(len(idxs), dtype=float)
    psi_boundary = np.zeros(len(idxs), dtype=float)
    Ip = np.zeros(len(idxs), dtype=float)

    for k, i in enumerate(idxs):
        ts = ods['equilibrium.time_slice'][i]
        t[k] = float(ts.get('time', i))
        psi_boundary[k] = float(ts['global_quantities.psi_boundary']) - float(ts['global_quantities.psi_axis'])
        Ip[k] = float(ts['global_quantities.ip'])

    # V_loop from total flux (2π psi_boundary)
    if len(idxs) == 1:
        # With a single point we cannot differentiate; use stored value if present else 0
        ts = ods['equilibrium.time_slice'][idxs[0]]
        V_loop = np.array([float(ts.get('global_quantities.v_loop', 0.0))], dtype=float)
    else:
        V_loop = loop_voltage_from_total_flux(t, psi_boundary)

    # Magnetic energy time series
    W_mag = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        W_mag[k] = float(compute_magnetic_energy(ods, time_slice=i))

    # dW/dt using time_derivative function
    from vaft.process.numerical import time_derivative
    if len(idxs) == 1:
        dWdt = np.array([0.0], dtype=float)
    else:
        dWdt = time_derivative(t, W_mag)

    # V_ind and V_res
    with np.errstate(divide='ignore', invalid='ignore'):
        V_ind = np.where(Ip != 0.0, inductive_voltage_from_dW_magdt_I_p(dWdt, Ip), 0.0)
    V_res = V_loop - V_ind

    return t, V_loop, V_ind, V_res

def compute_bremsstrahlung_power(
    ods: ODS,
    time_slice: Optional[int] = None,
    Z_eff: float = 2.0
    ) -> Tuple[float, float]:
    """
    Compute bremsstrahlung power from core profiles using two methods.
    
    Calculates:
    - P_B_pressure = ∫_V S_B(pressure) dV using pressure-based formula
    - P_B_electron = ∫_V S_B(n_e, T_e) dV using electron density-based formula
    
    Args:
        ods: OMAS data structure
        time_slice: Time slice index for core profile (None = use first available)
        Z_eff: Effective charge (default: 2.0)
    
    Returns:
        Tuple of (P_B_pressure, P_B_electron) in [W]
    
    Raises:
        KeyError: If required data is missing
        ValueError: If plasma volume is zero
    """
    from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi
    from vaft.process.equilibrium import psi_to_RZ, volume_average
    from scipy.interpolate import interp1d
    
    # Basic availability checks
    if 'core_profiles.profiles_1d' not in ods:
        raise KeyError("core_profiles.profiles_1d not found in ODS")
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")
    
    # Determine time slice for core profile
    if time_slice is None:
        cp_idx = 0
    else:
        cp_idx = time_slice if time_slice < len(ods['core_profiles.profiles_1d']) else 0
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    
    # Get core profile time
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
    
    # Get core profile data: T_e, n_e, and rho_tor_norm
    grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
    if 'rho_tor_norm' not in grid:
        raise KeyError(f"rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}]")
    
    rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)
    
    if 'electrons.temperature' not in cp_ts:
        raise KeyError(f"electrons.temperature missing in core_profiles.profiles_1d[{cp_idx}]")
    T_e_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
    
    if 'electrons.density' not in cp_ts:
        raise KeyError(f"electrons.density missing in core_profiles.profiles_1d[{cp_idx}]")
    n_e_1d_rho = np.asarray(cp_ts['electrons.density'], float)
    
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
    
    # Interpolate T_e and n_e to psi_norm coordinate
    interp_T_e = interp1d(rho_tor_norm_cp, T_e_1d_rho,
                          kind='linear',
                          bounds_error=False,
                          fill_value=(T_e_1d_rho[0], T_e_1d_rho[-1]))
    T_e_1d = interp_T_e(rho_tor_norm_at_psiN)
    
    interp_n_e = interp1d(rho_tor_norm_cp, n_e_1d_rho,
                          kind='linear',
                          bounds_error=False,
                          fill_value=(n_e_1d_rho[0], n_e_1d_rho[-1]))
    n_e_1d = interp_n_e(rho_tor_norm_at_psiN)
    
    # Get equilibrium 2D grid and ψ(R,Z)
    R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
    Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
    psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
    psi_axis = float(eq_ts['global_quantities.psi_axis'])
    psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
    
    # Map T_e and n_e to 2D (R,Z)
    T_e_RZ, psiN_RZ = psi_to_RZ(psiN_1d, T_e_1d, psi_RZ, psi_axis, psi_lcfs)
    n_e_RZ, _ = psi_to_RZ(psiN_1d, n_e_1d, psi_RZ, psi_axis, psi_lcfs)
    
    # Get pressure from equilibrium profiles_1d if available, otherwise compute from n_e * T_e
    # pressure_1d = None
    # if 'pressure' in eq_profiles_1d:
    #     pressure_1d_eq = np.asarray(eq_profiles_1d['pressure'], float)
    #     # Interpolate pressure to psi_norm coordinate
    #     interp_p = interp1d(rho_tor_norm_eq_sorted, pressure_1d_eq,
    #                        kind='linear',
    #                        bounds_error=False,
    #                        fill_value=(pressure_1d_eq[0], pressure_1d_eq[-1]))
    #     pressure_1d = interp_p(rho_tor_norm_at_psiN)
    
    # If pressure not available, compute from n_e * T_e (convert eV to J)
    # if pressure_1d is None:
    # p = n_e * T_e, where T_e is in eV, convert to Pa: p = 2 * n_e * T_e * e (elementary charge)
    QE = 1.602176634e-19  # elementary charge [C]
    pressure_1d = n_e_1d * T_e_1d * QE * 2  # [Pa] 
    
    # Map pressure to 2D (R,Z)
    pressure_RZ, _ = psi_to_RZ(psiN_1d, pressure_1d, psi_RZ, psi_axis, psi_lcfs)
    
    # Calculate bremsstrahlung power density using pressure-based formula
    # Handle zero/negative values (outside plasma)
    T_e_RZ_safe = np.where(T_e_RZ > 0, T_e_RZ, np.nan)
    pressure_RZ_safe = np.where(pressure_RZ > 0, pressure_RZ, np.nan)
    valid_mask_p = ~np.isnan(T_e_RZ_safe) & ~np.isnan(pressure_RZ_safe)
    
    S_B_pressure_RZ = np.zeros_like(T_e_RZ)
    S_B_pressure_RZ[valid_mask_p] = bremsstrahlung_power_density_from_T_e_p_Z_eff(
        T_e_RZ_safe[valid_mask_p],
        pressure_RZ_safe[valid_mask_p],
        Z_eff=Z_eff
    )
    
    # Calculate bremsstrahlung power density using electron density-based formula
    n_e_RZ_safe = np.where(n_e_RZ > 0, n_e_RZ, np.nan)
    valid_mask_e = ~np.isnan(T_e_RZ_safe) & ~np.isnan(n_e_RZ_safe)
    
    S_B_electron_RZ = np.zeros_like(T_e_RZ)
    S_B_electron_RZ[valid_mask_e] = bremsstrahlung_power_density_from_Z_eff_n_e_T_e(
        n_e_RZ_safe[valid_mask_e],
        T_e_RZ_safe[valid_mask_e],
        Z_eff=Z_eff
    )
    
    # Compute volume integrals: P_B = ∫_V S_B dV
    # Using volume_average: returns (average, volume), so integral = average * volume
    p_avg_pressure, V = volume_average(S_B_pressure_RZ, psiN_RZ, R_grid, Z_grid)
    P_B_pressure = float(p_avg_pressure * V)  # [W]
    
    p_avg_electron, _ = volume_average(S_B_electron_RZ, psiN_RZ, R_grid, Z_grid)
    P_B_electron = float(p_avg_electron * V)  # [W]
    
    return P_B_pressure, P_B_electron

def compute_power_balance(
    ods: ODS
    ) -> Dict[str, ndarray]:
    """
    Compute power balance time series assuming P_aux = 0 and P_rad = 0:

    - P_ohm_flux = I_p * V_res (flux-based calculation)
    - P_ohm_diss = ∫_V η J_φ² dV (dissipation-based calculation from core profiles)
    - P_heat = P_ohm_diss + P_aux = P_ohm_diss
    - P_loss = P_heat - dW/dt - P_rad = P_ohm_diss - dW/dt

    Returns a dict of arrays: time, V_loop, V_ind, V_res, P_ohm_flux, P_ohm_diss, 
    P_aux, P_heat, P_rad, dWdt, P_loss.
    
    Note: Requires multiple time slices to compute dW/dt. If only one time slice
    is available, dW/dt will be set to zero.
    """
    from vaft.omas.process_wrapper import compute_magnetic_energy, compute_ohmic_heating_power_from_core_profiles

    t, V_loop, V_ind, V_res = compute_voltage_consumption(ods, time_slice=None)

    # Process all time slices
    idxs = list(range(len(ods['equilibrium.time_slice'])))

    Ip = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        Ip[k] = float(ods['equilibrium.time_slice'][i]['global_quantities.ip'])

    # Ohmic power from resistive voltage (flux-based)
    P_ohm_flux = ohmic_heating_power_from_I_p_V_res(Ip, V_res)

    # Ohmic power from dissipation (core profile-based)
    P_ohm_diss = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        # Try to find corresponding core profile time slice
        # Use equilibrium time slice index as a proxy for core profile time slice
        P_ohm_diss[k] = float(compute_ohmic_heating_power_from_core_profiles(ods, time_slice=i))

    # Assume no auxiliary heating and no radiation (as requested)
    P_aux = np.zeros_like(P_ohm_diss)
    P_rad = np.zeros_like(P_ohm_diss)
    # Use P_ohm_diss for P_heat calculation
    P_heat = heating_power_from_p_ohm_p_aux(P_ohm_diss, P_aux)

    # Thermal energy (W_th) & dW/dt from core_profiles volume-averaged pressure
    from vaft.omas.update import update_equilibrium_global_quantities_volume
    
    # Ensure volume is computed for all time slices
    update_equilibrium_global_quantities_volume(ods, time_slice=None)
    
    # Compute volume-averaged pressure from core_profiles for all time slices
    p_vol_avg = compute_volume_averaged_pressure(ods, time_slice=None, option='core_profiles')
    
    # Get time array
    t = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        eq_ts = ods['equilibrium.time_slice'][i]
        t[k] = float(eq_ts.get('time', i))
    
    # Calculate W_th = p_vol_average * 2/3 * volume for each time slice
    W_th = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        eq_ts = ods['equilibrium.time_slice'][i]
        volume = float(eq_ts['global_quantities.volume'])
        # W_th = p_vol_average * 2/3 * volume
        W_th[k] = p_vol_avg[k] * (2.0 / 3.0) * volume
    
    # Calculate dW/dt (requires multiple time slices)
    from vaft.process.numerical import time_derivative
    if len(idxs) == 1:
        dWdt = np.zeros_like(W_th)
        logger.warning("Only one time slice available, dW/dt set to zero")
    else:
        dWdt = time_derivative(t, W_th)

    # Use P_ohm_diss-based P_heat for P_loss calculation
    P_loss = loss_power_from_p_heat_dWdt_p_rad(P_heat, dWdt, P_rad)

    return {
        'time': t,
        'V_loop': V_loop,
        'V_ind': V_ind,
        'V_res': V_res,
        'P_ohm_flux': P_ohm_flux,
        'P_ohm_diss': P_ohm_diss,
        'P_aux': P_aux,
        'P_heat': P_heat,
        'P_rad': P_rad,
        'dWdt': dWdt,
        'P_loss': P_loss,
    }

def compute_confiment_time_paramters(
    ods: ODS,
    time_slice: int,
    Z_eff: float = 2.0,
    M: float = 1.0,
    ) -> Tuple[float, float, float, float, float]:
    """
    Compute confinement-time parameters and H-factor at a given time slice.

    Returns
    -------
    tau_IPB89 : float
        Here computed using the available "IBP98y2" scaling (repository provides IBP98y2, not IPB89y2).
    tau_H98y2 : float
        Confinement time from "H98y2" scaling.
    tau_NSTX : float
        Confinement time from "NSTX" scaling.
    H_factor : float
        H = tau_exp / tau_IPB89 (uses confinement_factor_from_tau_E_exp_tau_E_IPB89y2).
    tau_exp : float
        Experimental confinement time tau_exp = W_th / P_loss.
    """
    # Scaling-law confinement times
    tau_IPB89 = compute_tau_E_scaling(ods, time_slice, scaling="IPB89", Z_eff=Z_eff, M=M)
    tau_H98y2 = compute_tau_E_scaling(ods, time_slice, scaling="H98y2", Z_eff=Z_eff, M=M)
    tau_NSTX = compute_tau_E_scaling(ods, time_slice, scaling="NSTX", Z_eff=Z_eff, M=M)

    # Experimental confinement time
    tau_exp = compute_tau_E_exp(ods, time_slice, Z_eff=Z_eff)

    # H-factor (as defined in formula/equilibrium.py)
    H_factor = confinement_factor_from_tau_E_exp_tau_E_IPB89y2(tau_exp, tau_IPB89)
    
    # Ensure all return values are scalars
    if isinstance(tau_IPB89, np.ndarray):
        tau_IPB89 = float(tau_IPB89[0] if len(tau_IPB89) > 0 else tau_IPB89)
    else:
        tau_IPB89 = float(tau_IPB89)
    
    if isinstance(tau_H98y2, np.ndarray):
        tau_H98y2 = float(tau_H98y2[0] if len(tau_H98y2) > 0 else tau_H98y2)
    else:
        tau_H98y2 = float(tau_H98y2)
    
    if isinstance(tau_NSTX, np.ndarray):
        tau_NSTX = float(tau_NSTX[0] if len(tau_NSTX) > 0 else tau_NSTX)
    else:
        tau_NSTX = float(tau_NSTX)
    
    if isinstance(H_factor, np.ndarray):
        H_factor = float(H_factor[0] if len(H_factor) > 0 else H_factor)
    else:
        H_factor = float(H_factor)
    
    if isinstance(tau_exp, np.ndarray):
        tau_exp = float(tau_exp[0] if len(tau_exp) > 0 else tau_exp)
    else:
        tau_exp = float(tau_exp)

    return tau_IPB89, tau_H98y2, tau_NSTX, H_factor, tau_exp