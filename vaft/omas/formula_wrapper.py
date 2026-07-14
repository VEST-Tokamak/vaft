from typing import List, Tuple, Dict, Any, Optional
import re
import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d
from omas import *
import logging

logger = logging.getLogger(__name__)
from vaft.omas.process_wrapper import compute_volume_averaged_pressure
from vaft.process import compute_response_matrix
from vaft.process.equilibrium import psi_to_rz, volume_average
from vaft.formula import magnetic_shear, ballooning_alpha_from_p_B_R
from vaft.formula.equilibrium import (
    loss_power_from_p_heat_dWdt_p_rad,
    heating_power_from_p_ohm_p_aux,
    ohmic_heating_power_from_I_p_V_res,
    loop_voltage_from_total_flux,
    inductive_voltage_from_dW_magdt_I_p,
    bremsstrahlung_power_density_from_Z_eff_n_e_T_e,
    bremsstrahlung_power_density_from_T_e_p_Z_eff,
    stored_energy_from_p_V,
    confinement_time_from_engineering_parameters,
    confinement_time_from_P_loss_W_th,
    inverse_aspect_ratio_from_a_R,
    confinement_factor_ITER89P,
    cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e,
)
from vaft.formula.constants import _SCALING_COEFS

_POWER_BALANCE_CACHE: Dict[int, Dict[str, ndarray]] = {}
_POWER_BALANCE_CACHE_SIGNATURE: Dict[int, Tuple[int, int]] = {}


def _power_balance_signature(ods: ODS) -> Tuple[int, int]:
    """Create a lightweight signature for per-ODS power-balance caching."""
    n_eq = len(ods['equilibrium.time_slice']) if 'equilibrium.time_slice' in ods else 0
    n_cp = len(ods['core_profiles.profiles_1d']) if 'core_profiles.profiles_1d' in ods else 0
    return int(n_eq), int(n_cp)


def _get_cached_power_balance(ods: ODS) -> Dict[str, ndarray]:
    """
    Return cached power-balance result for this ODS when possible.

    This avoids recomputing the same equilibrium/core-profile time-matching path
    repeatedly within per-slice loops, which also suppresses duplicated warnings.
    """
    key = id(ods)
    sig = _power_balance_signature(ods)
    if key in _POWER_BALANCE_CACHE and _POWER_BALANCE_CACHE_SIGNATURE.get(key) == sig:
        return _POWER_BALANCE_CACHE[key]

    pb = compute_power_balance(ods)
    _POWER_BALANCE_CACHE[key] = pb
    _POWER_BALANCE_CACHE_SIGNATURE[key] = sig
    return pb

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
        - n_e: Line-averaged electron density at z=0 [m^-3]
        - n_e_line_avg: Line-averaged electron density at z=0 [m^-3]
        - n_e_vol_avg: Volume-averaged electron density [m^-3]
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
    power_balance = _get_cached_power_balance(ods)
    # Restrict to exact common times between equilibrium and core_profiles.
    eq_time = float(eq_ts['time']) if 'time' in eq_ts else float(time_slice)
    time_array = np.asarray(power_balance['time'], dtype=float)
    matched_idx = np.where(np.isclose(time_array, eq_time, rtol=0.0, atol=1e-9))[0]
    if len(matched_idx) == 0:
        raise ValueError(
            f"time_slice[{time_slice}] at time={eq_time} is not in the common "
            f"equilibrium/core_profiles time set. Available common times: {time_array.tolist()}"
        )
    time_idx = int(matched_idx[0])
    P_loss = float(power_balance['P_loss'][time_idx])  # [W]
    
    # Compute volume-averaged electron density
    # Use only core_profile slices with exact shared time.
    cp_idx = None
    if 'core_profiles.profiles_1d' in ods:
        eq_time = float(eq_ts['time']) if 'time' in eq_ts else float(time_slice)
        for idx in range(len(ods['core_profiles.profiles_1d'])):
            cp_ts = ods['core_profiles.profiles_1d'][idx]
            cp_time = float(cp_ts['time']) if 'time' in cp_ts else float(idx)
            if np.isclose(cp_time, eq_time, rtol=0.0, atol=1e-9):
                cp_idx = idx
                break
    
    if cp_idx is None:
        raise ValueError(f"No matching core profile found for equilibrium time_slice[{time_slice}]")
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    
    if 'electrons.density' not in cp_ts:
        raise ValueError(f"electrons.density not found in core_profiles.profiles_1d[{cp_idx}]")
    
    n_e_profile = np.asarray(cp_ts['electrons.density'], float)  # [m^-3]
    
    # Compute line-average (z=0) and volume-average of n_e using proper coordinate mapping
    # Get equilibrium 2D grid
    R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
    Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
    psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
    psi_axis = float(eq_ts['global_quantities.psi_axis'])
    psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
    
    # Get equilibrium profiles_1d for coordinate conversion
    eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
    
    # Ensure equilibrium has psi_norm
    if 'psi_norm' not in eq_profiles_1d:
        from vaft.omas.update import update_equilibrium_profiles_1d_normalized_psi
        update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=time_slice)
        eq_profiles_1d = eq_ts['profiles_1d'] if 'profiles_1d' in eq_ts else ODS()
    
    if 'psi_norm' not in eq_profiles_1d:
        raise ValueError(
            f"psi_norm not available in equilibrium.profiles_1d for time_slice[{time_slice}]. "
            "Cannot compute mapped line-averaged density at z=0."
        )
    psi_norm_1d = np.asarray(eq_profiles_1d['psi_norm'], float)

    # Get core profile coordinate (rho_tor_norm)
    grid = cp_ts['grid'] if 'grid' in cp_ts else (ods['core_profiles.grid'] if 'core_profiles.grid' in ods else ODS())
    if 'rho_tor_norm' not in grid:
        raise ValueError(
            f"rho_tor_norm not found in core_profiles grid for time_slice[{time_slice}]. "
            "Cannot map kinetic profile to equilibrium grid."
        )
    rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)

    # Get equilibrium rho_tor_norm for mapping
    if 'rho_tor_norm' not in eq_profiles_1d:
        raise ValueError(
            f"rho_tor_norm not found in equilibrium.profiles_1d for time_slice[{time_slice}]. "
            "Cannot map kinetic profile to equilibrium grid."
        )
    rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)

    # Interpolate n_e from core profile rho_tor_norm to equilibrium psi_norm
    interp_func = interp1d(
        rho_tor_norm_cp,
        n_e_profile,
        kind='linear',
        bounds_error=False,
        fill_value=(n_e_profile[0], n_e_profile[-1]),
    )
    n_e_psi_norm = interp_func(rho_tor_norm_eq)

    # Map to 2D grid and compute averages
    n_e_RZ, psiN_RZ = psi_to_rz(psi_norm_1d, n_e_psi_norm, psi_RZ, psi_axis, psi_lcfs)
    n_e_vol_avg, _ = volume_average(n_e_RZ, psiN_RZ, R_grid, Z_grid)  # [m^-3]
    # Ensure scalar
    if isinstance(n_e_vol_avg, np.ndarray):
        n_e_vol_avg = float(n_e_vol_avg[0] if len(n_e_vol_avg) > 0 else n_e_vol_avg)
    else:
        n_e_vol_avg = float(n_e_vol_avg)

    # Line-average at geometric midplane z=0 using in-plasma points.
    z0_idx = int(np.argmin(np.abs(Z_grid)))
    n_e_RZ_arr = np.asarray(n_e_RZ, dtype=float)
    psiN_RZ_arr = np.asarray(psiN_RZ, dtype=float)
    if not (n_e_RZ_arr.ndim == 2 and psiN_RZ_arr.ndim == 2):
        raise ValueError(
            f"Mapped kinetic profile must be 2D. Got n_e_RZ.ndim={n_e_RZ_arr.ndim}, "
            f"psiN_RZ.ndim={psiN_RZ_arr.ndim} for time_slice[{time_slice}]."
        )

    if n_e_RZ_arr.shape == (len(R_grid), len(Z_grid)):
        n_mid = n_e_RZ_arr[:, z0_idx]
        psi_mid = psiN_RZ_arr[:, z0_idx]
        R_mid = R_grid
    elif n_e_RZ_arr.shape == (len(Z_grid), len(R_grid)):
        n_mid = n_e_RZ_arr[z0_idx, :]
        psi_mid = psiN_RZ_arr[z0_idx, :]
        R_mid = R_grid
    else:
        raise ValueError(
            f"Unexpected mapped profile shape n_e_RZ={n_e_RZ_arr.shape} for "
            f"(len(R_grid), len(Z_grid))=({len(R_grid)}, {len(Z_grid)}). "
            f"Cannot compute z=0 line-average for time_slice[{time_slice}]."
        )

    in_plasma = (
        np.isfinite(n_mid)
        & np.isfinite(psi_mid)
        & (n_mid > 0.0)
        & (psi_mid >= 0.0)
        & (psi_mid <= 1.0)
    )
    if np.sum(in_plasma) < 2:
        raise ValueError(
            f"Insufficient valid in-plasma midplane points (count={int(np.sum(in_plasma))}) "
            f"to compute line-averaged density at z=0 for time_slice[{time_slice}]."
        )

    R_sel = np.asarray(R_mid[in_plasma], dtype=float)
    n_sel = np.asarray(n_mid[in_plasma], dtype=float)
    order = np.argsort(R_sel)
    R_sorted = R_sel[order]
    n_sorted = n_sel[order]
    chord_len = float(R_sorted[-1] - R_sorted[0])
    if chord_len <= 0.0:
        raise ValueError(
            f"Invalid midplane chord length ({chord_len}) for time_slice[{time_slice}]. "
            "Cannot compute line-averaged density."
        )
    n_e_line_avg = float(np.trapezoid(n_sorted, R_sorted) / chord_len)
    
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
    if not np.isfinite(n_e_line_avg) or n_e_line_avg <= 0:
        raise ValueError(f"Invalid n_e_line_avg: {n_e_line_avg}")
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
        'n_e': n_e_line_avg,
        'n_e_line_avg': n_e_line_avg,
        'n_e_vol_avg': n_e_vol_avg,
        'R': R,
        'epsilon': epsilon,
        'kappa': kappa,
        'M': M,
    }


def _density_input_for_scaling(
    eng_params: Dict[str, float],
    scaling: str,
) -> Tuple[float, str]:
    """
    Select the density value/definition pair expected by the requested scaling.

    Returns
    -------
    Tuple[float, str]
        (density_value_m^-3, density_definition_label)
    """
    if scaling not in _SCALING_COEFS:
        raise ValueError(f"Unknown scaling '{scaling}'. Available: {list(_SCALING_COEFS.keys())}")

    target_density_def = str(_SCALING_COEFS[scaling].get("density_definition", "line_avg")).strip().lower()
    if target_density_def in {"volume_avg", "volume-average", "volume_averaged", "volume-averaged", "volume"}:
        if "n_e_vol_avg" in eng_params:
            return float(eng_params["n_e_vol_avg"]), "volume_avg"
        raise ValueError(
            f"Scaling '{scaling}' requires volume-averaged density, "
            "but 'n_e_vol_avg' is missing in engineering parameters."
        )

    # Default: line-averaged density
    if "n_e_line_avg" in eng_params:
        return float(eng_params["n_e_line_avg"]), "line_avg"
    if "n_e" in eng_params:
        # Backward compatibility with older engineering-parameter payloads
        return float(eng_params["n_e"]), "line_avg"
    raise ValueError(
        f"Scaling '{scaling}' requires line-averaged density, "
        "but neither 'n_e_line_avg' nor 'n_e' is available."
    )


def _compute_tau_E_scaling_from_eng_params(
    eng_params: Dict[str, float],
    scaling: str,
) -> float:
    """Compute tau_E from a precomputed engineering-parameter dictionary."""
    n_e_value, n_e_definition = _density_input_for_scaling(eng_params, scaling)
    tau_E = confinement_time_from_engineering_parameters(
        I_p=eng_params['I_p'],
        B_t=eng_params['B_t'],
        P_loss=eng_params['P_loss'],
        n_e=n_e_value,
        M=eng_params['M'],
        R=eng_params['R'],
        epsilon=eng_params['epsilon'],
        kappa=eng_params['kappa'],
        scaling=scaling,
        input_density_definition=n_e_definition,
    )  # [s]

    # Handle complex numbers (should not happen with valid inputs, but handle gracefully)
    if isinstance(tau_E, complex):
        if tau_E.imag != 0:
            raise ValueError(
                f"Confinement time calculation resulted in complex number: {tau_E}. Check input parameters."
            )
        tau_E = tau_E.real

    # Ensure return value is scalar
    if isinstance(tau_E, np.ndarray):
        tau_E = float(tau_E[0] if len(tau_E) > 0 else tau_E)
    elif not isinstance(tau_E, (int, float, np.number)):
        raise TypeError(f"tau_E is not a numeric type: {type(tau_E)}, value: {tau_E}")
    else:
        tau_E = float(tau_E)

    if not np.isfinite(tau_E) or tau_E <= 0:
        raise ValueError(f"Invalid confinement time result: {tau_E}")
    return tau_E


def compute_tau_E_scaling(ods, time_slice: int, scaling: str = "IBP98y2", 
                          Z_eff: float = 2.0, M: float = 1.0,
                          eng_params: Optional[Dict[str, float]] = None) -> float:
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
        Confinement time from scaling law [s].
        Returns NaN when required engineering-parameter mapping fails and the
        time slice is skipped.
    """
    # Get engineering parameters once unless precomputed values are supplied.
    if eng_params is None:
        try:
            eng_params = compute_tau_E_engineering_parameters(ods, time_slice, Z_eff=Z_eff, M=M)
        except ValueError as err:
            logger.warning(
                f"Skipping time_slice[{time_slice}] for scaling '{scaling}': "
                f"failed to compute engineering parameters ({err})"
            )
            return float("nan")

    try:
        tau_E = _compute_tau_E_scaling_from_eng_params(eng_params, scaling=scaling)
    except Exception as err:
        logger.warning(
            f"Skipping time_slice[{time_slice}] for scaling '{scaling}': "
            f"failed scaling evaluation ({err}). "
            f"Inputs: I_p={eng_params.get('I_p')}, B_t={eng_params.get('B_t')}, "
            f"P_loss={eng_params.get('P_loss')}, n_e_line_avg={eng_params.get('n_e_line_avg')}, "
            f"n_e_vol_avg={eng_params.get('n_e_vol_avg')}, M={eng_params.get('M')}, "
            f"R={eng_params.get('R')}, epsilon={eng_params.get('epsilon')}, "
            f"kappa={eng_params.get('kappa')}"
        )
        return float("nan")

    return float(tau_E)

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
        volume = float(eq_ts['global_quantities.volume']) if 'global_quantities.volume' in eq_ts else np.nan
        
        # Calculate thermal energy: W_th = p_vol_average * 3/2 * volume
        W_th = p_vol_avg_val * (3.0 / 2.0) * volume  # [J]
    except Exception as e:
        raise ValueError(f"Could not determine stored thermal energy for time_slice[{time_slice}]: {e}")
    
    # Compute P_loss from compute_power_balance
    power_balance = _get_cached_power_balance(ods)
    # Find the index corresponding to the requested time_slice
    eq_time = float(eq_ts['time']) if 'time' in eq_ts else float(time_slice)
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
        t[k] = float(ts['time']) if 'time' in ts else float(i)
        psi_boundary[k] = float(ts['global_quantities.psi_boundary']) - float(ts['global_quantities.psi_axis'])
        Ip[k] = float(ts['global_quantities.ip'])

    # V_loop from total flux (2π psi_boundary)
    if len(idxs) == 1:
        # With a single point we cannot differentiate; use stored value if present else 0
        ts = ods['equilibrium.time_slice'][idxs[0]]
        V_loop_val = float(ts['global_quantities.v_loop']) if 'global_quantities.v_loop' in ts else 0.0
        V_loop = np.array([V_loop_val], dtype=float)
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
    from vaft.process.equilibrium import psi_to_rz, volume_average
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
    grid = cp_ts['grid'] if 'grid' in cp_ts else (ods['core_profiles.grid'] if 'core_profiles.grid' in ods else ODS())
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
    T_e_RZ, psiN_RZ = psi_to_rz(psiN_1d, T_e_1d, psi_RZ, psi_axis, psi_lcfs)
    n_e_RZ, _ = psi_to_rz(psiN_1d, n_e_1d, psi_RZ, psi_axis, psi_lcfs)
    
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
    pressure_RZ, _ = psi_to_rz(psiN_1d, pressure_1d, psi_RZ, psi_axis, psi_lcfs)
    
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

_DEFAULT_LINE_RADIATION_SPECIES = ("C", "O")
_DEFAULT_IMPURITY_FRACTIONS = {"C": 1.0e-2, "O": 1.0e-2}
_DEFAULT_TIME_MATCH_ATOL = 1.0e-6
_ATOMIC_NUMBERS = {
    "H": 1,
    "D": 1,
    "T": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "Ne": 10,
    "Ar": 18,
    "Kr": 36,
    "Xe": 54,
    "W": 74,
}


def _compute_time_match_atol(time_array: ndarray, base_atol: float = _DEFAULT_TIME_MATCH_ATOL) -> float:
    """
    Compute a robust absolute tolerance for matching time slices.

    The tolerance is bounded below by ``base_atol`` and, when possible,
    adapted to the native spacing of the provided time array.
    """
    arr = np.asarray(time_array, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float(base_atol)

    diffs = np.diff(np.unique(np.sort(arr)))
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        return float(base_atol)

    return float(max(base_atol, 0.25 * float(np.min(positive))))


def _find_time_match_index(time_array: ndarray, target_time: float) -> Optional[int]:
    """Return index of the nearest matching time slice, or ``None`` if no close match exists."""
    arr = np.asarray(time_array, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(target_time):
        return None

    atol = _compute_time_match_atol(arr)
    close_idx = np.where(np.isclose(arr, float(target_time), rtol=0.0, atol=atol))[0]
    if close_idx.size == 0:
        return None

    # Pick the closest candidate when multiple slices are within tolerance.
    idx = int(close_idx[np.argmin(np.abs(arr[close_idx] - float(target_time)))])
    return idx


def _normalize_atomic_symbol(label: Any) -> Optional[str]:
    """Normalize an impurity label to atomic symbol form (e.g. 'carbon6+' -> 'C')."""
    if label is None:
        return None
    if isinstance(label, bytes):
        text = label.decode("utf-8", errors="ignore")
    else:
        text = str(label)
    text = text.strip()
    if not text:
        return None

    match = re.match(r"([A-Za-z]+)", text)
    if not match:
        return None
    token = match.group(1)

    # Prefer one-letter symbols first to avoid mapping names like "carbon" to "Ca".
    cand1 = token[0].upper()
    if cand1 in _ATOMIC_NUMBERS:
        return cand1

    if len(token) >= 2:
        cand2 = token[:2].capitalize()
        if cand2 in _ATOMIC_NUMBERS:
            return cand2

    return None


def _sanitize_rho_grid(rho: ndarray) -> Optional[ndarray]:
    """Return sorted, unique, finite rho grid (or None if not enough points)."""
    rho_arr = np.asarray(rho, dtype=float).reshape(-1)
    rho_arr = rho_arr[np.isfinite(rho_arr)]
    if rho_arr.size < 2:
        return None
    rho_arr = np.unique(np.sort(rho_arr))
    return rho_arr if rho_arr.size >= 2 else None


def _interp_profile_to_target(
    rho_src: ndarray,
    profile_src: ndarray,
    rho_target: ndarray,
) -> Optional[ndarray]:
    """Interpolate 1D profile from source rho grid to target rho grid."""
    rho_src_arr = np.asarray(rho_src, dtype=float).reshape(-1)
    prof_src_arr = np.asarray(profile_src, dtype=float).reshape(-1)
    rho_tgt_arr = np.asarray(rho_target, dtype=float).reshape(-1)

    if rho_src_arr.size != prof_src_arr.size or rho_src_arr.size < 2 or rho_tgt_arr.size == 0:
        return None

    finite_src = np.isfinite(rho_src_arr) & np.isfinite(prof_src_arr)
    if np.count_nonzero(finite_src) < 2:
        return None

    rho_src_arr = rho_src_arr[finite_src]
    prof_src_arr = prof_src_arr[finite_src]
    sort_idx = np.argsort(rho_src_arr)
    rho_src_arr = rho_src_arr[sort_idx]
    prof_src_arr = prof_src_arr[sort_idx]

    unique_mask = np.concatenate(([True], np.diff(rho_src_arr) > 1e-10))
    rho_src_arr = rho_src_arr[unique_mask]
    prof_src_arr = prof_src_arr[unique_mask]
    if rho_src_arr.size < 2:
        return None

    out = np.full(rho_tgt_arr.shape, np.nan, dtype=float)
    finite_tgt = np.isfinite(rho_tgt_arr)
    if not np.any(finite_tgt):
        return out

    interp = interp1d(
        rho_src_arr,
        prof_src_arr,
        kind="linear",
        bounds_error=False,
        fill_value=(prof_src_arr[0], prof_src_arr[-1]),
    )
    out[finite_tgt] = interp(rho_tgt_arr[finite_tgt])
    return out


def _infer_impurity_fraction_from_zeff(Z_eff: Optional[float], species: str) -> Optional[float]:
    """
    Infer n_imp/n_e from Z_eff for a single-impurity, hydrogenic-main-ion model:
    Z_eff = 1 + (n_imp / n_e) * Z * (Z - 1).
    """
    if Z_eff is None or not np.isfinite(Z_eff):
        return None
    Z = _ATOMIC_NUMBERS.get(species)
    if Z is None or Z <= 1:
        return None
    fraction = (float(Z_eff) - 1.0) / float(Z * (Z - 1))
    return max(float(fraction), 0.0)


def _impurity_fraction_profile_from_core_profiles(
    cp_ts: ODS,
    rho_cp: ndarray,
    rho_target: ndarray,
    n_e_target: ndarray,
    species: str,
) -> Optional[ndarray]:
    """Get n_imp/n_e profile from core_profiles ion densities when available."""
    if "ion" not in cp_ts:
        return None

    for ion_idx in range(len(cp_ts["ion"])):
        ion_ts = cp_ts["ion"][ion_idx]
        ion_symbol = _normalize_atomic_symbol(ion_ts["label"] if "label" in ion_ts else None)
        if ion_symbol != species:
            continue

        density_key = None
        if "density" in ion_ts:
            density_key = "density"
        elif "density_thermal" in ion_ts:
            density_key = "density_thermal"
        if density_key is None:
            continue

        n_imp_cp = np.asarray(ion_ts[density_key], dtype=float)
        n_imp_target = _interp_profile_to_target(rho_cp, n_imp_cp, rho_target)
        if n_imp_target is None:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(n_e_target > 0.0, n_imp_target / n_e_target, 0.0)
        frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(frac, 0.0, None)

    return None


def _integrate_emissivity_profile(
    emissivity_profile: ndarray,
    volume_profile: Optional[ndarray],
    total_volume: float,
) -> float:
    """Integrate emissivity [W/m^3] over volume to obtain power [W]."""
    from vaft.compat import trapz_compat

    emiss = np.asarray(emissivity_profile, dtype=float)
    finite_emiss = np.isfinite(emiss)
    if not np.any(finite_emiss):
        return 0.0

    if volume_profile is not None:
        vol = np.asarray(volume_profile, dtype=float)
        if vol.shape == emiss.shape:
            finite = finite_emiss & np.isfinite(vol)
            if np.count_nonzero(finite) >= 2:
                vol_f = vol[finite]
                em_f = emiss[finite]
                order = np.argsort(vol_f)
                vol_f = vol_f[order]
                em_f = em_f[order]
                # Integrate using cumulative volume coordinate where available.
                if np.ptp(vol_f) > 0.0:
                    return float(trapz_compat(em_f, x=vol_f))

    if np.isfinite(total_volume) and total_volume > 0.0:
        return float(np.nanmean(emiss[finite_emiss]) * total_volume)
    return 0.0


def _compute_aurora_line_radiation_power_series(
    ods: ODS,
    eq_indices: List[int],
    eq_times: ndarray,
    volume_series: ndarray,
    line_radiation_species: Optional[List[str]] = None,
    impurity_fractions: Optional[Dict[str, float]] = None,
    Z_eff: Optional[float] = None,
) -> ndarray:
    """
    Estimate line-radiation power series using Aurora cooling factors.

    Model:
        P_rad_line = ∫ n_e * n_imp * Lz_line(Te, ne) dV

    Notes:
    - Priority per species: ODS ion profile > impurity_fractions > Z_eff estimate
      (single-species only) > 0.
    - Uses equilibrium-profile volume integration when profiles_1d.volume is available.
    - Falls back to volume-averaged emissivity * total volume otherwise.
    - If impurity profiles are unavailable in ODS, scalar impurity_fractions are used.
    - Aurora/ADAS lookup failures are handled by skipping the failing species.
    """
    P_rad_line = np.zeros(len(eq_indices), dtype=float)

    if "core_profiles.profiles_1d" not in ods:
        logger.warning("core_profiles missing; Aurora line radiation set to zero.")
        return P_rad_line

    try:
        import aurora  # aurorafusion package
    except Exception as exc:
        logger.warning(
            "Aurora is not available (%s). Setting line radiation to zero for this call.",
            exc,
        )
        return P_rad_line

    cp_profiles = ods["core_profiles.profiles_1d"]
    cp_times = np.asarray(
        [float(cp_profiles[j]["time"]) if "time" in cp_profiles[j] else float(j) for j in range(len(cp_profiles))],
        dtype=float,
    )

    normalized_fraction_map: Dict[str, float] = {}
    if impurity_fractions is None:
        normalized_fraction_map = dict(_DEFAULT_IMPURITY_FRACTIONS)
    elif impurity_fractions:
        for key, value in impurity_fractions.items():
            species = _normalize_atomic_symbol(key)
            if species is None:
                continue
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-numeric impurity fraction for %s: %r", key, value)
                continue
            normalized_fraction_map[species] = max(fraction, 0.0)

    if line_radiation_species is not None:
        species_list = []
        for s in line_radiation_species:
            symbol = _normalize_atomic_symbol(s)
            if symbol and symbol not in species_list:
                species_list.append(symbol)
    else:
        species_list = list(_DEFAULT_LINE_RADIATION_SPECIES)

    if not species_list:
        return P_rad_line

    warned_zero_fraction = set()
    warned_lookup_failure = set()
    warned_shape_failure = set()

    for k, eq_idx in enumerate(eq_indices):
        eq_time = float(eq_times[k])
        cp_idx = _find_time_match_index(cp_times, eq_time)
        if cp_idx is None:
            continue

        cp_ts = cp_profiles[int(cp_idx)]
        eq_ts = ods["equilibrium.time_slice"][eq_idx]

        grid = cp_ts["grid"] if "grid" in cp_ts else (ods["core_profiles.grid"] if "core_profiles.grid" in ods else ODS())
        if "rho_tor_norm" not in grid:
            continue
        rho_cp = np.asarray(grid["rho_tor_norm"], dtype=float)
        if "electrons.density" not in cp_ts or "electrons.temperature" not in cp_ts:
            continue

        n_e_cp = np.asarray(cp_ts["electrons.density"], dtype=float)
        T_e_cp = np.asarray(cp_ts["electrons.temperature"], dtype=float)

        eq_profiles_1d = eq_ts["profiles_1d"] if "profiles_1d" in eq_ts else ODS()
        rho_eq = None
        if "rho_tor_norm" in eq_profiles_1d:
            rho_eq = _sanitize_rho_grid(eq_profiles_1d["rho_tor_norm"])
        rho_target = rho_eq if rho_eq is not None else _sanitize_rho_grid(rho_cp)
        if rho_target is None:
            continue

        n_e = _interp_profile_to_target(rho_cp, n_e_cp, rho_target)
        T_e = _interp_profile_to_target(rho_cp, T_e_cp, rho_target)
        if n_e is None or T_e is None:
            continue

        finite_kin = np.isfinite(n_e) & np.isfinite(T_e) & (n_e > 0.0) & (T_e > 0.0)
        if not np.any(finite_kin):
            continue
        valid_idx = np.where(finite_kin)[0]

        volume_profile = None
        if "volume" in eq_profiles_1d and "rho_tor_norm" in eq_profiles_1d:
            volume_profile = _interp_profile_to_target(
                np.asarray(eq_profiles_1d["rho_tor_norm"], dtype=float),
                np.asarray(eq_profiles_1d["volume"], dtype=float),
                rho_target,
            )

        total_volume = float(volume_series[k]) if k < len(volume_series) else np.nan
        p_line_slice = 0.0

        for species in species_list:
            # Prefer explicit/ODS impurity density fractions. If unavailable and only
            # one species is requested, estimate scalar n_imp/n_e from Z_eff.
            frac_profile = _impurity_fraction_profile_from_core_profiles(
                cp_ts=cp_ts,
                rho_cp=rho_cp,
                rho_target=rho_target,
                n_e_target=n_e,
                species=species,
            )

            if frac_profile is None:
                frac_scalar = normalized_fraction_map.get(species)
                if frac_scalar is None and len(species_list) == 1:
                    frac_scalar = _infer_impurity_fraction_from_zeff(Z_eff, species)
                if frac_scalar is None:
                    frac_scalar = 0.0
                frac_profile = np.full_like(n_e, frac_scalar, dtype=float)

            frac_profile = np.where(np.isfinite(frac_profile), np.clip(frac_profile, 0.0, None), 0.0)
            if not np.any(frac_profile > 0.0):
                if species not in warned_zero_fraction:
                    logger.warning(
                        "No impurity fraction available for %s; line radiation from this species is zero.",
                        species,
                    )
                    warned_zero_fraction.add(species)
                continue

            try:
                # Aurora can fail with KeyError('ccd') when ne contains zeros because
                # its internal n0/ne handling creates NaN; evaluate only on strictly
                # positive, finite kinetic points.
                ne_valid_cm3 = np.asarray(n_e[valid_idx], dtype=float) * 1e-6
                Te_valid_eV = np.asarray(T_e[valid_idx], dtype=float)
                line_coeff, _ = aurora.radiation.get_cooling_factors(
                    imp=species,
                    ne_cm3=ne_valid_cm3,
                    Te_eV=Te_valid_eV,
                    plot=False,
                )
            except Exception as exc:
                if species not in warned_lookup_failure:
                    logger.warning(
                        "Aurora/ADAS cooling-factor lookup failed for species=%s "
                        "(first at eq_idx=%d): %s. Skipping this species.",
                        species,
                        eq_idx,
                        exc,
                    )
                    warned_lookup_failure.add(species)
                continue

            line_coeff_raw = np.asarray(line_coeff, dtype=float).reshape(-1)
            line_coeff = np.zeros_like(n_e, dtype=float)
            if line_coeff_raw.size == valid_idx.size:
                line_coeff[valid_idx] = np.clip(line_coeff_raw, 0.0, None)
            elif line_coeff_raw.size == 1:
                line_coeff[valid_idx] = max(float(line_coeff_raw[0]), 0.0)
            elif line_coeff_raw.size == n_e.size:
                line_coeff = np.clip(line_coeff_raw, 0.0, None)
            else:
                if species not in warned_shape_failure:
                    logger.warning(
                        "Unexpected Aurora cooling-factor shape for species=%s "
                        "(first at eq_idx=%d): %d (expected %d valid or %d full). "
                        "Skipping this species.",
                        species,
                        eq_idx,
                        line_coeff_raw.size,
                        valid_idx.size,
                        n_e.size,
                    )
                    warned_shape_failure.add(species)
                continue

            n_imp = frac_profile * n_e
            emissivity = np.where(
                finite_kin,
                np.clip(line_coeff, 0.0, None) * np.clip(n_e, 0.0, None) * np.clip(n_imp, 0.0, None),
                np.nan,
            )
            p_line_slice += _integrate_emissivity_profile(emissivity, volume_profile, total_volume)

        P_rad_line[k] = max(float(p_line_slice), 0.0)

    return P_rad_line


def _estimate_reference_bt_from_eq_time_slice(eq_ts: ODS, r_ref: float = 0.4) -> float:
    """
    Estimate toroidal field [T] at reference major radius using magnetic-axis values.

    Uses B_t(R_ref) ≈ B_t(axis) * R_axis / R_ref.
    Returns NaN if required data is missing.
    """
    try:
        b_axis = float(eq_ts["global_quantities.magnetic_axis.b_field_tor"])
        r_axis = float(eq_ts["global_quantities.magnetic_axis.r"])
        if not np.isfinite(b_axis) or not np.isfinite(r_axis) or not np.isfinite(r_ref) or r_ref <= 0.0:
            return float("nan")
        return abs(b_axis * r_axis / r_ref)
    except Exception:
        return float("nan")


def _compute_sync_radiation_power_series(
    ods: ODS,
    eq_indices: List[int],
    eq_times: ndarray,
    volume_series: ndarray,
) -> ndarray:
    """
    Estimate cyclotron/synchrotron power series using a practical scaling law:

        p_sync ~ C_sync * n_e * B_t^2 * T_e

    and integrate over volume to obtain total power [W].
    """
    P_sync = np.zeros(len(eq_indices), dtype=float)

    if "core_profiles.profiles_1d" not in ods:
        logger.warning("core_profiles missing; synchrotron scaling power set to zero.")
        return P_sync

    cp_profiles = ods["core_profiles.profiles_1d"]
    cp_times = np.asarray(
        [
            float(cp_profiles[j]["time"]) if "time" in cp_profiles[j] else float(j)
            for j in range(len(cp_profiles))
        ],
        dtype=float,
    )

    for k, eq_idx in enumerate(eq_indices):
        eq_time = float(eq_times[k])
        cp_idx = _find_time_match_index(cp_times, eq_time)
        if cp_idx is None:
            continue

        cp_ts = cp_profiles[int(cp_idx)]
        eq_ts = ods["equilibrium.time_slice"][eq_idx]

        grid = cp_ts["grid"] if "grid" in cp_ts else (ods["core_profiles.grid"] if "core_profiles.grid" in ods else ODS())
        if "rho_tor_norm" not in grid or "electrons.density" not in cp_ts or "electrons.temperature" not in cp_ts:
            continue

        rho_cp = np.asarray(grid["rho_tor_norm"], dtype=float)
        n_e_cp = np.asarray(cp_ts["electrons.density"], dtype=float)
        T_e_cp = np.asarray(cp_ts["electrons.temperature"], dtype=float)

        eq_profiles_1d = eq_ts["profiles_1d"] if "profiles_1d" in eq_ts else ODS()
        rho_eq = None
        if "rho_tor_norm" in eq_profiles_1d:
            rho_eq = _sanitize_rho_grid(eq_profiles_1d["rho_tor_norm"])
        rho_target = rho_eq if rho_eq is not None else _sanitize_rho_grid(rho_cp)
        if rho_target is None:
            continue

        n_e = _interp_profile_to_target(rho_cp, n_e_cp, rho_target)
        T_e = _interp_profile_to_target(rho_cp, T_e_cp, rho_target)
        if n_e is None or T_e is None:
            continue

        b_t_ref = _estimate_reference_bt_from_eq_time_slice(eq_ts)
        if not np.isfinite(b_t_ref) or b_t_ref <= 0.0:
            continue

        finite = np.isfinite(n_e) & np.isfinite(T_e) & (n_e > 0.0) & (T_e > 0.0)
        if not np.any(finite):
            continue

        sync_emissivity = np.where(
            finite,
            np.clip(
                cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e(
                    n_e_m3=np.asarray(n_e, dtype=float),
                    B_t_T=float(b_t_ref),
                    T_e_eV=np.asarray(T_e, dtype=float),
                ),
                0.0,
                None,
            ),
            np.nan,
        )

        volume_profile = None
        if "volume" in eq_profiles_1d and "rho_tor_norm" in eq_profiles_1d:
            volume_profile = _interp_profile_to_target(
                np.asarray(eq_profiles_1d["rho_tor_norm"], dtype=float),
                np.asarray(eq_profiles_1d["volume"], dtype=float),
                rho_target,
            )

        total_volume = float(volume_series[k]) if k < len(volume_series) else np.nan
        P_sync[k] = max(
            _integrate_emissivity_profile(sync_emissivity, volume_profile, total_volume),
            0.0,
        )

    return P_sync


def _compute_bremsstrahlung_power_series(
    ods: ODS,
    eq_indices: List[int],
    eq_times: ndarray,
    Z_eff: Optional[float] = 2.0,
) -> ndarray:
    """
    Compute bremsstrahlung radiation power [W] time series.

    Uses ``compute_bremsstrahlung_power`` (electron-density-based branch) with
    equilibrium/core_profiles time matching.
    """
    P_Br = np.zeros(len(eq_indices), dtype=float)

    if "core_profiles.profiles_1d" not in ods:
        logger.warning("core_profiles missing; bremsstrahlung power set to zero.")
        return P_Br

    cp_profiles = ods["core_profiles.profiles_1d"]
    cp_times = np.asarray(
        [
            float(cp_profiles[j]["time"]) if "time" in cp_profiles[j] else float(j)
            for j in range(len(cp_profiles))
        ],
        dtype=float,
    )

    failed_slices = []
    for k, _eq_idx in enumerate(eq_indices):
        cp_idx = _find_time_match_index(cp_times, float(eq_times[k]))
        if cp_idx is None:
            continue
        try:
            _, p_br_e = compute_bremsstrahlung_power(ods, time_slice=int(cp_idx), Z_eff=float(Z_eff or 2.0))
            P_Br[k] = max(float(p_br_e), 0.0)
        except Exception:
            failed_slices.append(int(_eq_idx))

    if failed_slices:
        preview = failed_slices[:10]
        suffix = "..." if len(failed_slices) > 10 else ""
        logger.warning(
            "Could not compute bremsstrahlung power for %d equilibrium slices; setting zero "
            "(failed slices: %s%s)",
            len(failed_slices),
            preview,
            suffix,
        )

    return P_Br


def compute_power_balance(
    ods: ODS,
    include_line_radiation: bool = True,
    line_radiation_species: Optional[List[str]] = None,
    impurity_fractions: Optional[Dict[str, float]] = None,
    Z_eff: Optional[float] = 2.0,
    ) -> Dict[str, ndarray]:
    """
    Compute power balance time series with optional Aurora-based line radiation:

    - P_ohm_flux = I_p * V_res (flux-based calculation)
    - P_ohm_diss = ∫_V η J_φ² dV (dissipation-based calculation from core profiles)
    - P_heat = P_ohm_diss + P_aux = P_ohm_diss
    - P_loss = P_heat - dW/dt - P_rad

    Aurora line-radiation model:
    - Uses line cooling coefficients from aurora.radiation.get_cooling_factors.
    - Computes P_rad_line = ∫ n_e * n_imp * Lz_line(Te, ne) dV.
    - Uses profile integration with equilibrium profiles_1d.volume when available;
      otherwise falls back to volume-averaged emissivity times total plasma volume.
    - Uses impurity density profiles from ODS if present, else impurity_fractions,
      else (single-species case only) Z_eff-based scalar estimate, else zero.
    - Priority per species: ODS ion profile > impurity_fractions > Z_eff (single species) > 0.
    - Default species/fractions: C and O with n_imp/n_e = 0.01 each.
    - If Aurora is unavailable, line radiation is set to zero with a warning.
    - ADAS/cooling-factor lookup failures skip only failing species with warnings.
    - Total radiation used in balance is:
      P_rad = P_rad_line + P_sync + P_Br
      where P_sync is a cyclotron/synchrotron scaling estimate and P_Br is
      the profile-integrated bremsstrahlung estimate.

    Returns a dict of arrays:
    time, V_loop, V_ind, V_res, P_ohm_flux, P_ohm_diss, P_aux, P_heat,
    P_rad_line, P_Br, P_sync, P_rad, P_trans, P_loss_total, dWdt, P_loss.
    Here, P_loss keeps historical semantics (transport-like loss term
    P_heat - dWdt - P_rad), while P_loss_total = P_rad + P_trans.
    
    Note: Requires multiple time slices to compute dW/dt. If only one time slice
    is available, dW/dt will be set to zero.
    """
    from vaft.omas.process_wrapper import compute_ohmic_heating_power_from_core_profiles

    _, V_loop, V_ind, V_res = compute_voltage_consumption(ods, time_slice=None)

    # Process all time slices
    idxs = list(range(len(ods['equilibrium.time_slice'])))
    t = np.asarray(
        [
            float(ods['equilibrium.time_slice'][i]['time'])
            if 'time' in ods['equilibrium.time_slice'][i]
            else float(i)
            for i in idxs
        ],
        dtype=float,
    )

    Ip = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        Ip[k] = float(ods['equilibrium.time_slice'][i]['global_quantities.ip'])

    # Ohmic power from resistive voltage (flux-based)
    P_ohm_flux = ohmic_heating_power_from_I_p_V_res(Ip, V_res)

    # Ohmic power from dissipation (core profile-based)
    P_ohm_diss = np.full(len(idxs), np.nan, dtype=float)
    failed_ohmic_slices = []
    missing_cp_match_slices = []
    cp_times = np.asarray([], dtype=float)
    if "core_profiles.profiles_1d" in ods and len(ods["core_profiles.profiles_1d"]) > 0:
        cp_times = np.asarray(
            [
                float(ods["core_profiles.profiles_1d"][j]["time"])
                if "time" in ods["core_profiles.profiles_1d"][j]
                else float(j)
                for j in range(len(ods["core_profiles.profiles_1d"]))
            ],
            dtype=float,
        )
    else:
        logger.warning(
            "core_profiles.profiles_1d not found; P_ohm_diss cannot be computed from "
            "dissipation and will be NaN."
        )

    for k, i in enumerate(idxs):
        cp_idx = _find_time_match_index(cp_times, t[k]) if cp_times.size > 0 else None
        if cp_idx is None:
            missing_cp_match_slices.append(i)
            continue
        try:
            P_ohm_diss[k] = float(compute_ohmic_heating_power_from_core_profiles(ods, time_slice=int(cp_idx)))
        except Exception:
            failed_ohmic_slices.append(i)

    if missing_cp_match_slices and cp_times.size > 0:
        preview = missing_cp_match_slices[:10]
        suffix = "..." if len(missing_cp_match_slices) > 10 else ""
        logger.warning(
            "No matching core_profiles slice found for %d/%d equilibrium slices; "
            "setting P_ohm_diss to NaN there (slices: %s%s).",
            len(missing_cp_match_slices),
            len(idxs),
            preview,
            suffix,
        )

    if failed_ohmic_slices:
        preview = failed_ohmic_slices[:10]
        suffix = "..." if len(failed_ohmic_slices) > 10 else ""
        logger.warning(
            "Could not compute P_ohm_diss for %d/%d equilibrium slices; setting NaN "
            "(failed slices: %s%s)",
            len(failed_ohmic_slices),
            len(idxs),
            preview,
            suffix,
        )

    # Assume no auxiliary heating
    P_aux = np.zeros_like(P_ohm_diss)
    P_heat = heating_power_from_p_ohm_p_aux(P_ohm_diss, P_aux)

    # Thermal energy (W_th) & dW/dt from core_profiles volume-averaged pressure
    from vaft.omas.update import update_equilibrium_global_quantities_volume
    
    # Ensure volume is computed for all time slices
    update_equilibrium_global_quantities_volume(ods, time_slice=None)
    
    # Compute volume-averaged pressure from core_profiles for all time slices
    p_vol_avg = compute_volume_averaged_pressure(ods, time_slice=None, option='core_profiles')
    
    # Calculate W_th = p_vol_average * 2/3 * volume for each time slice
    W_th = np.zeros(len(idxs), dtype=float)
    volume_series = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        eq_ts = ods['equilibrium.time_slice'][i]
        volume = float(eq_ts['global_quantities.volume'])
        volume_series[k] = volume
        # W_th = p_vol_average * 2/3 * volume
        W_th[k] = p_vol_avg[k] * (2.0 / 3.0) * volume
    
    # Calculate dW/dt robustly on finite W_th points only.
    # This prevents NaNs at trailing slices from contaminating earlier finite slices.
    from vaft.process.numerical import time_derivative
    if len(idxs) == 1:
        dWdt = np.zeros_like(W_th)
        logger.warning("Only one time slice available, dW/dt set to zero")
    else:
        finite_mask = np.isfinite(t) & np.isfinite(W_th)
        finite_idx = np.where(finite_mask)[0]
        dWdt = np.full_like(W_th, np.nan, dtype=float)
        if len(finite_idx) < 2:
            logger.warning("Insufficient finite W_th points for dW/dt; leaving dWdt as NaN")
        else:
            dWdt_finite = time_derivative(t[finite_idx], W_th[finite_idx])
            dWdt[finite_idx] = dWdt_finite

    line_rad_requested = bool(include_line_radiation)
    if line_rad_requested:
        P_rad_line = _compute_aurora_line_radiation_power_series(
            ods=ods,
            eq_indices=idxs,
            eq_times=t,
            volume_series=volume_series,
            line_radiation_species=line_radiation_species,
            impurity_fractions=impurity_fractions,
            Z_eff=Z_eff,
        )
    else:
        P_rad_line = np.zeros_like(P_ohm_diss)

    P_sync = _compute_sync_radiation_power_series(
        ods=ods,
        eq_indices=idxs,
        eq_times=t,
        volume_series=volume_series,
    )

    P_Br = _compute_bremsstrahlung_power_series(
        ods=ods,
        eq_indices=idxs,
        eq_times=t,
        Z_eff=Z_eff,
    )

    # Total radiation definition used for power balance:
    #   P_rad = P_rad_line + P_sync + P_Br
    P_rad = np.asarray(P_rad_line, dtype=float) + np.asarray(P_sync, dtype=float) + np.asarray(P_Br, dtype=float)

    P_loss = loss_power_from_p_heat_dWdt_p_rad(P_heat, dWdt, P_rad)
    P_trans = np.asarray(P_loss, dtype=float)
    P_loss_total = np.asarray(P_rad, dtype=float) + P_trans

    # Keep only times shared by both equilibrium and core_profiles when available.
    if cp_times.size > 0:
        common_mask = np.asarray([_find_time_match_index(cp_times, tt) is not None for tt in t], dtype=bool)
        if not np.any(common_mask):
            logger.warning(
                "No equilibrium/core_profiles time matches found within tolerance; "
                "returning full equilibrium time grid."
            )
            common_mask = np.ones_like(t, dtype=bool)
    else:
        common_mask = np.ones_like(t, dtype=bool)

    return {
        'time': t[common_mask],
        'V_loop': V_loop[common_mask],
        'V_ind': V_ind[common_mask],
        'V_res': V_res[common_mask],
        'P_ohm_flux': P_ohm_flux[common_mask],
        'P_ohm_diss': P_ohm_diss[common_mask],
        'P_aux': P_aux[common_mask],
        'P_heat': P_heat[common_mask],
        'P_rad_line': P_rad_line[common_mask],
        'P_Br': P_Br[common_mask],
        'P_rad': P_rad[common_mask],
        'P_sync': P_sync[common_mask],
        'P_trans': P_trans[common_mask],
        'P_loss_total': P_loss_total[common_mask],
        'dWdt': dWdt[common_mask],
        'P_loss': P_loss[common_mask],
    }

def compute_confiment_time_paramters(
    ods: ODS,
    time_slice: int,
    Z_eff: float = 2.0,
    M: float = 1.0,
    ) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute confinement-time parameters and H-factor at a given time slice.

    Returns
    -------
    tau_ITER89P : float
        Confinement time from ``ITER89P`` scaling.
    tau_H98y2 : float
        Confinement time from ``H98y2`` scaling.
    tau_NSTX2006H : float
        Confinement time from ``NSTX2006H`` scaling.
    tau_NSTX2006L : float
        Confinement time from ``NSTX2006L`` scaling.
    tau_Kurskiev2022 : float
        Confinement time from ``Kurskiev2022`` (ST multi-machine) scaling.
    H_factor : float
        H = tau_exp / tau_ITER89P (uses confinement_factor_ITER89P).
    tau_exp : float
        Experimental confinement time tau_exp = W_th / P_loss.

    Notes
    -----
    This helper is skip-tolerant for per-slice failures: when a required
    sub-calculation fails, the corresponding output is returned as NaN and a
    warning is logged instead of raising.
    """
    # Compute engineering parameters once per slice and reuse for all scaling laws.
    try:
        eng_params = compute_tau_E_engineering_parameters(ods, time_slice, Z_eff=Z_eff, M=M)
    except Exception as err:
        logger.warning(
            f"Skipping scaling-law confinement times for time_slice[{time_slice}]: "
            f"failed to compute engineering parameters ({err})"
        )
        eng_params = None

    # Scaling-law confinement times
    tau_ITER89P = compute_tau_E_scaling(
        ods,
        time_slice,
        scaling="ITER89P",
        Z_eff=Z_eff,
        M=M,
        eng_params=eng_params,
    ) if eng_params is not None else float("nan")
    tau_H98y2 = compute_tau_E_scaling(
        ods,
        time_slice,
        scaling="H98y2",
        Z_eff=Z_eff,
        M=M,
        eng_params=eng_params,
    ) if eng_params is not None else float("nan")
    tau_NSTX2006H = compute_tau_E_scaling(
        ods,
        time_slice,
        scaling="NSTX2006H",
        Z_eff=Z_eff,
        M=M,
        eng_params=eng_params,
    ) if eng_params is not None else float("nan")
    tau_NSTX2006L = compute_tau_E_scaling(
        ods,
        time_slice,
        scaling="NSTX2006L",
        Z_eff=Z_eff,
        M=M,
        eng_params=eng_params,
    ) if eng_params is not None else float("nan")
    tau_Kurskiev2022 = compute_tau_E_scaling(
        ods,
        time_slice,
        scaling="Kurskiev2022",
        Z_eff=Z_eff,
        M=M,
        eng_params=eng_params,
    ) if eng_params is not None else float("nan")

    # Experimental confinement time
    try:
        tau_exp = compute_tau_E_exp(ods, time_slice, Z_eff=Z_eff)
    except Exception as err:
        logger.warning(
            f"Skipping tau_exp for time_slice[{time_slice}]: "
            f"failed to compute experimental confinement time ({err})"
        )
        tau_exp = float("nan")

    # H-factor (as defined in formula/equilibrium.py)
    if np.isfinite(tau_exp) and np.isfinite(tau_ITER89P) and tau_ITER89P > 0:
        H_factor = confinement_factor_ITER89P(tau_exp, tau_ITER89P)
    else:
        H_factor = float("nan")
    
    # Ensure all return values are scalars
    if isinstance(tau_ITER89P, np.ndarray):
        tau_ITER89P = float(tau_ITER89P[0] if len(tau_ITER89P) > 0 else tau_ITER89P)
    else:
        tau_ITER89P = float(tau_ITER89P)
    
    if isinstance(tau_H98y2, np.ndarray):
        tau_H98y2 = float(tau_H98y2[0] if len(tau_H98y2) > 0 else tau_H98y2)
    else:
        tau_H98y2 = float(tau_H98y2)
    
    if isinstance(tau_NSTX2006H, np.ndarray):
        tau_NSTX2006H = float(tau_NSTX2006H[0] if len(tau_NSTX2006H) > 0 else tau_NSTX2006H)
    else:
        tau_NSTX2006H = float(tau_NSTX2006H)

    if isinstance(tau_NSTX2006L, np.ndarray):
        tau_NSTX2006L = float(tau_NSTX2006L[0] if len(tau_NSTX2006L) > 0 else tau_NSTX2006L)
    else:
        tau_NSTX2006L = float(tau_NSTX2006L)

    if isinstance(tau_Kurskiev2022, np.ndarray):
        tau_Kurskiev2022 = float(tau_Kurskiev2022[0] if len(tau_Kurskiev2022) > 0 else tau_Kurskiev2022)
    else:
        tau_Kurskiev2022 = float(tau_Kurskiev2022)
    
    if isinstance(H_factor, np.ndarray):
        H_factor = float(H_factor[0] if len(H_factor) > 0 else H_factor)
    else:
        H_factor = float(H_factor)
    
    if isinstance(tau_exp, np.ndarray):
        tau_exp = float(tau_exp[0] if len(tau_exp) > 0 else tau_exp)
    else:
        tau_exp = float(tau_exp)

    return tau_ITER89P, tau_H98y2, tau_NSTX2006H, tau_NSTX2006L, tau_Kurskiev2022, H_factor, tau_exp
