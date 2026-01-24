"""
Utility functions for plasma physics calculations.

This module provides common utility functions and fitting utilities used throughout
the formula module.
"""

import numpy as np
from typing import Union, Tuple, List, Dict
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# ------------------------------------------------------------------
# Basic Utilities
# ------------------------------------------------------------------

def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate gradient with proper handling of array dimensions.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
        
    Returns
    -------
    np.ndarray
        Gradient dy/dx
    """
    return np.gradient(y, x)


def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate definite integral using trapezoidal rule.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
        
    Returns
    -------
    float
        Definite integral ∫y dx
    """
    return np.trapz(y, x)


def normalize_profile(x: Union[float, np.ndarray],
                     x_axis: float,
                     x_boundary: float) -> Union[float, np.ndarray]:
    """
    Normalize a profile to range [0,1].
    
    Parameters
    ----------
    x : Union[float, np.ndarray]
        Values to normalize
    x_axis : float
        Value at axis
    x_boundary : float
        Value at boundary
        
    Returns
    -------
    Union[float, np.ndarray]
        Normalized values: (x - x_axis) / (x_boundary - x_axis)
    """
    return (x - x_axis) / (x_boundary - x_axis)


def calculate_peaking_factor(central: float,
                           volume_avg: float) -> float:
    """
    Calculate peaking factor PF = X(0) / ⟨X⟩.
    
    Parameters
    ----------
    central : float
        Central value
    volume_avg : float
        Volume-averaged value
        
    Returns
    -------
    float
        Peaking factor
    """
    return central / volume_avg


def calculate_volume_weighted_average(x: np.ndarray,
                                    V: np.ndarray) -> float:
    """
    Calculate volume-weighted average of a profile.
    
    Parameters
    ----------
    x : np.ndarray
        Profile values
    V : np.ndarray
        Volume elements
    
    Returns
    -------
    float
        Volume-weighted average
    """
    return np.sum(x * V) / np.sum(V)


def calculate_poloidal_flux(R: np.ndarray,
                          B_theta: np.ndarray,
                          l: np.ndarray,
                          psi_axis: float = 0.0) -> float:
    """
    Calculate poloidal flux from line integral of R*B_theta.
    
    Parameters
    ----------
    R : np.ndarray
        Major radius values
    B_theta : np.ndarray
        Poloidal magnetic field values
    l : np.ndarray
        Path length values
    psi_axis : float, optional
        Poloidal flux at magnetic axis, by default 0.0
        
    Returns
    -------
    float
        Poloidal flux value
    """
    return trapz_integral(l, R * B_theta) + psi_axis


def calculate_toroidal_flux(B_phi: np.ndarray,
                          dA: np.ndarray) -> float:
    """
    Calculate toroidal flux through surface.
    
    Parameters
    ----------
    B_phi : np.ndarray
        Toroidal magnetic field values
    dA : np.ndarray
        Area elements
    
    Returns
    -------
    float
        Toroidal flux value
    """
    return np.sum(B_phi * dA)


# ------------------------------------------------------------------
# Fitting Utilities
# ------------------------------------------------------------------

def make_fit_function(mode):
    """
    Build a 1D parametric fit function for polynomial or exponential modes.

    Supported modes:
    - 'polynomial': (1 - x) * poly(x)
    - 'exponential': (1 - x) * exp(poly(x))

    Parameters
    ----------
    mode : str
        Fitting mode: 'polynomial' or 'exponential'

    Returns
    -------
    callable
        Fit function f(x, *coeffs)
    """
    mode = mode.lower()
    if mode == 'polynomial':
        # (1-x)*poly(x) -> enforces value -> 0 at x=1
        def func(x, *coeffs):
            x = np.asarray(x, dtype=float)
            s = 0.0
            for k in range(len(coeffs)):
                s = s + coeffs[k] * x**k
            return (1.0 - x) * s
    elif mode == 'exponential':
        # (1-x)*exp(poly(x)) -> goes to 0 at x=1, stays positive
        def func(x, *coeffs):
            x = np.asarray(x, dtype=float)
            s = 0.0
            for k in range(len(coeffs)):
                s = s + coeffs[k] * x**k
            return (1.0 - x) * np.exp(s)
    else:
        raise ValueError(f"Invalid fitting function: {mode}")
    return func


def _core_poly_edge_exp_model(x, x0, w, *coeffs):
    """
    Core polynomial + edge exponential blend with tanh transition.
    """
    if w == 0:
        w = 1e-6
    z = (x - x0) / w
    core_order = max(len(coeffs) - 2, 1)
    core_coeffs = coeffs[:core_order]
    edge_offset = coeffs[core_order]
    edge_amp = coeffs[core_order + 1]

    core = 0.0
    for k, c in enumerate(core_coeffs):
        core = core + c * z**k

    edge = edge_offset + edge_amp * np.exp(-z)
    blend = 0.5 * (1.0 - np.tanh(z)) * core + 0.5 * (1.0 + np.tanh(z)) * edge
    return blend


def _initial_core_poly_edge_exp_guess(x, y, order):
    """
    Initial guess helper for core_poly_edge_exp fit.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size < 3:
        return [0.9, 0.05] + [1.0] * order + [y[-1] if y.size else 0.0, 0.0]

    dy = np.gradient(y, x)
    x0 = x[np.argmax(np.abs(dy))]
    w = 0.05 * (np.max(x) - np.min(x) + 1e-6)

    core_mask = x <= x0
    if np.count_nonzero(core_mask) >= order:
        core_coeffs = np.polyfit((x[core_mask] - x0) / w, y[core_mask], order - 1)[::-1]
    else:
        core_coeffs = np.ones(order, dtype=float)

    edge_offset = y[-1]
    edge_amp = y[-1] - y[0] if y.size > 1 else 0.0
    return [x0, w] + list(core_coeffs) + [edge_offset, edge_amp]


def fit_profile(
    x,
    y,
    y_std,
    x_eval,
    order=3,
    uncertainty_option=1,
    fitting_function='polynomial',
    gp_kernel=None,
    gp_anchor=None,
    n_restarts_optimizer=5,
):
    """
    Fit a 1D profile with selectable methods.

    Supported modes:
    - fitting_function ∈ {'gp', 'polynomial', 'exponential', 'linear', 'core_poly_edge_exp'}

    Behavior:
    - GP uses sklearn GaussianProcessRegressor; optional anchor points supported.
    - linear uses 1D interpolation.
    - core_poly_edge_exp blends core polynomial with edge exponential via tanh transition.
    - polynomial/exponential include a (1 - x) factor for edge roll-off.

    Parameters
    ----------
    x : array-like
        Data x values (1D)
    y : array-like
        Data y values (1D)
    y_std : array-like, optional
        Per-point uncertainty
    x_eval : array-like
        Evaluation grid (1D)
    order : int, optional
        Polynomial order (for polynomial/exponential/core_poly_edge_exp), by default 3
    uncertainty_option : int, optional
        Use y_std as weights if 1, by default 1
    fitting_function : str, optional
        Fit method selection, by default 'polynomial'
    gp_kernel : sklearn kernel, optional
        Optional sklearn kernel (GP only)
    gp_anchor : tuple, optional
        Optional tuple (x_anchor, y_anchor, y_std_anchor) for GP
    n_restarts_optimizer : int, optional
        GP hyperparameter restarts, by default 5

    Returns
    -------
    tuple
        Tuple of:
        - y_eval : fitted values on x_eval
        - y_std_eval : fitted std (nonzero for GP; zeros otherwise)
        - fit_function : callable f(x) for arbitrary x
        - coeffs : fitted coefficients (None for GP/linear)

    Examples
    --------
    >>> fit_profile(x, y, y_std, x_eval, fitting_function='gp')
    >>> fit_profile(x, y, y_std, x_eval, order=3, fitting_function='polynomial')
    >>> fit_profile(x, y, y_std, x_eval, fitting_function='linear')
    >>> fit_profile(x, y, y_std, x_eval, order=3, fitting_function='core_poly_edge_exp')
    """
    if fitting_function.lower() == 'gp':
        kernel = gp_kernel or (C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.3, length_scale_bounds=(0.05, 5.0)))

        if gp_anchor is not None:
            x_anchor, y_anchor, y_std_anchor = gp_anchor
            x_gp = np.append(x.ravel(), np.ravel(x_anchor))
            y_gp = np.append(y, np.ravel(y_anchor))
            y_std_gp = np.append(y_std, np.ravel(y_std_anchor))
        else:
            x_gp = x.ravel()
            y_gp = y
            y_std_gp = y_std

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=y_std_gp**2,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        gp.fit(x_gp[:, None], y_gp)

        y_eval, y_std_eval = gp.predict(x_eval[:, None], return_std=True)

        def fit_function(x_input):
            x_arr = np.asarray(x_input, float).reshape(-1, 1)
            return gp.predict(x_arr)

        coeffs = None
        return y_eval, y_std_eval, fit_function, coeffs

    if fitting_function.lower() == 'linear':
        def fit_function(x_input):
            x_arr = np.asarray(x_input, float)
            return np.interp(x_arr, x.ravel(), y)

        y_eval = fit_function(x_eval)
        y_std_eval = np.zeros_like(y_eval)
        coeffs = None
        return y_eval, y_std_eval, fit_function, coeffs
    

        # --- NEW: sqrt-based model ---
    # Idea: fit f(x) to y^2 (enforces y ~ sqrt(f)), then return sqrt(f) as the profile.
    # This is a "strong assumption" shape: y must share the same underlying shape in squared space.
    if fitting_function.lower() in {'sqrt', 'sqrt_poly', 'sqrt_polynomial', 'sqrt_exponential', 'sqrt_exp'}:
        # choose which base function to fit in the squared-space
        ff = fitting_function.lower()

        if ff in {'sqrt', 'sqrt_poly', 'sqrt_polynomial'}:
            base_function = 'polynomial'
        elif ff in {'sqrt_exponential', 'sqrt_exp'}:
            base_function = 'exponential'
        else:
            base_function = 'polynomial'  # fallback

        # positivity handling: fit y^2
        y_pos = np.maximum(y, 0.0)
        y2 = y_pos**2

        # uncertainty propagation for y^2: sigma_{y^2} ≈ 2*y*sigma_y
        if y_std is not None:
            y2_std = 2.0 * np.maximum(y_pos, 0.0) * np.asarray(y_std, float)
            y2_std = np.clip(y2_std, 1e-12, None)
        else:
            y2_std = None

        # reuse existing machinery by fitting in squared space with a normal fitting function
        y2_eval, y2_std_eval, f2_function, coeffs2 = fit_profile(
            x=x,
            y=y2,
            y_std=y2_std,
            x_eval=x_eval,
            order=order,
            uncertainty_option=uncertainty_option,
            fitting_function=base_function,
            gp_kernel=gp_kernel,
            gp_anchor=None,  # anchor in squared space would need special handling; keep simple
            n_restarts_optimizer=n_restarts_optimizer,
        )

        # back to y-space
        y_eval = np.sqrt(np.maximum(y2_eval, 0.0))
        y_std_eval = np.zeros_like(y_eval)  # keep API consistent; could be refined if needed

        def fit_function(x_input):
            x_arr = np.asarray(x_input, float)
            y2_pred = f2_function(x_arr)
            return np.sqrt(np.maximum(y2_pred, 0.0))

        # coeffs: return the underlying squared-space coeffs so you can debug/compare
        coeffs = coeffs2
        return y_eval, y_std_eval, fit_function, coeffs


    if fitting_function.lower() in {'core_poly_edge_exp', 'core_poly_edge_exponential'}:
        p0 = _initial_core_poly_edge_exp_guess(x, y, order)

        if uncertainty_option == 1 and y_std is not None:
            coeffs, _ = curve_fit(
                _core_poly_edge_exp_model, x.ravel(), y, sigma=y_std, absolute_sigma=True, p0=p0, maxfev=20000
            )
        else:
            coeffs, _ = curve_fit(_core_poly_edge_exp_model, x.ravel(), y, p0=p0, maxfev=20000)

        def fit_function(x_input):
            x_arr = np.asarray(x_input, float)
            return _core_poly_edge_exp_model(x_arr, *coeffs)

        y_eval = fit_function(x_eval)
        y_std_eval = np.zeros_like(y_eval)
        return y_eval, y_std_eval, fit_function, coeffs

    func = make_fit_function(fitting_function)
    p0 = np.ones(order, dtype=float) * 0.1

    if uncertainty_option == 1 and y_std is not None:
        coeffs, _ = curve_fit(func, x.ravel(), y, sigma=y_std, absolute_sigma=True, p0=p0)
    else:
        coeffs, _ = curve_fit(func, x.ravel(), y, p0=p0)

    y_eval = func(x_eval, *coeffs)
    y_std_eval = np.zeros_like(y_eval)

    def fit_function(x_input):
        x_arr = np.asarray(x_input, float)
        return func(x_arr, *coeffs)

    return y_eval, y_std_eval, fit_function, coeffs
