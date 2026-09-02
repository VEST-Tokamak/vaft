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
from vaft.compat import trapz_compat


# ------------------------------------------------------------------
# Basic Utilities
# ------------------------------------------------------------------

def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Derivative $dy/dx$ on a sampled 1-D profile.

    $$\frac{dy}{dx}\Big|_i \approx \frac{y_{i+1} - y_{i-1}}{x_{i+1} - x_{i-1}}$$

    (second-order central difference on a possibly non-uniform grid).

    Parameters
    ----------
    x : np.ndarray
        Independent variable, monotonic [any].
    y : np.ndarray
        Dependent variable, same length as ``x`` [any].

    Returns
    -------
    np.ndarray
        $dy/dx$ at every sample, in units of ``y`` per unit ``x`` [any].

    Numerical notes
    ---------------
    Wraps ``numpy.gradient(y, x)``: second-order accurate in the interior,
    first-order one-sided at the two end points, and noise-amplifying (a
    relative noise $\delta$ on ``y`` becomes $\delta/\Delta x$ on the
    derivative).  Needs at least two samples; only the first axis of a 2-D
    ``y`` is differentiated.
    """
    return np.gradient(y, x)


def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    r"""Definite integral $\int y\,dx$ by the trapezoidal rule.

    $$\int y\,dx \approx \sum_i \frac{y_i + y_{i+1}}{2}\,(x_{i+1} - x_i)$$

    Parameters
    ----------
    x : np.ndarray
        Sample abscissae, monotonic [any].
    y : np.ndarray
        Integrand at the samples [any].

    Returns
    -------
    float
        Integral over the sampled range, in units of ``y`` times ``x`` [any].

    Numerical notes
    ---------------
    ``numpy.trapezoid`` through :func:`vaft.compat.trapz_compat`; second-order
    accurate in the spacing, exact for piecewise-linear integrands, and
    sign-reversed for a decreasing ``x``.  A same-named helper in
    :mod:`vaft.formula.green` shadows this one on the package namespace
    (``vaft.formula.trapz_integral`` is the Green's-function copy).
    """
    return trapz_compat(y, x=x)


def normalize_profile(x: Union[float, np.ndarray],
                     x_axis: float,
                     x_boundary: float) -> Union[float, np.ndarray]:
    r"""Linear normalisation of a profile between its axis and boundary values.

    $$x_N = \frac{x - x_{\mathrm{axis}}}{x_{\mathrm{boundary}} - x_{\mathrm{axis}}}$$

    Parameters
    ----------
    x : float or np.ndarray
        Values to normalise [any].
    x_axis : float
        Value mapped to 0 [any].
    x_boundary : float
        Value mapped to 1 [any].

    Returns
    -------
    float or np.ndarray
        Normalised values, 0 at the axis value and 1 at the boundary value [-].

    Limitations
    -----------
    No guard against ``x_boundary == x_axis``: a degenerate equilibrium with
    equal axis and boundary flux, which packaged VEST samples do contain,
    returns ``inf``/``nan``.  Tracked in #357.
    """
    return (x - x_axis) / (x_boundary - x_axis)


def calculate_peaking_factor(central: float,
                           volume_avg: float) -> float:
    r"""Peaking factor, central value over volume average.

    $$\mathrm{PF} = \frac{X(0)}{\langle X\rangle}$$

    Parameters
    ----------
    central : float
        Value on the magnetic axis [any].
    volume_avg : float
        Volume average of the same quantity, same unit [any].

    Returns
    -------
    float
        Peaking factor [-].

    Limitations
    -----------
    No guard against a zero volume average.  Tracked in #357.
    """
    return central / volume_avg


def calculate_volume_weighted_average(x: np.ndarray,
                                    V: np.ndarray) -> float:
    r"""Volume-weighted average of a sampled profile.

    $$\langle X\rangle = \frac{\sum_i X_i\,V_i}{\sum_i V_i}$$

    Parameters
    ----------
    x : np.ndarray
        Profile values at each cell [any].
    V : np.ndarray
        Volume of each cell, same shape [m^3].

    Returns
    -------
    float
        Volume-weighted average in the unit of ``x`` [any].

    Limitations
    -----------
    No guard against a zero total volume.  Tracked in #357.
    """
    return np.sum(x * V) / np.sum(V)


def calculate_poloidal_flux(R: np.ndarray,
                          B_theta: np.ndarray,
                          l: np.ndarray,
                          psi_axis: float = 0.0) -> float:
    r"""Poloidal flux per radian from a line integral of $RB_\theta$.

    $$\psi(l) = \int_0^{l} R\,B_\theta\,dl' + \psi_a$$

    Parameters
    ----------
    R : np.ndarray
        Major radius along the path [m].
    B_theta : np.ndarray
        Poloidal field normal to the path [T].
    l : np.ndarray
        Path coordinate, monotonic [m].
    psi_axis : float, optional
        Offset added to the integral; default 0 [Wb/rad].

    Returns
    -------
    float
        Flux at the end of the path [Wb/rad].

    Convention
    ----------
    Flux per radian (COCOS 1-8 storage); multiply by $2\pi$ for the IMAS
    full-weber flux.  Sign follows ``B_theta`` and the direction of ``l``.
    The physics wrapper is :func:`vaft.formula.equilibrium.psi_from_RBtheta`.

    Numerical notes
    ---------------
    Trapezoidal rule over the whole path; returns the end value only.
    """
    return trapz_integral(l, R * B_theta) + psi_axis


def calculate_toroidal_flux(B_phi: np.ndarray,
                          dA: np.ndarray) -> float:
    r"""Toroidal flux as a sum of $B_\varphi$ over area elements.

    $$\Phi = \sum_i B_{\varphi,i}\,\Delta A_i$$

    Parameters
    ----------
    B_phi : np.ndarray
        Toroidal field on the area elements [T].
    dA : np.ndarray
        Area of each element, same shape [m^2].

    Returns
    -------
    float
        Toroidal flux [Wb].

    Convention
    ----------
    Full weber (toroidal flux has no per-radian form); sign of $B_\varphi$.
    The physics wrapper is :func:`vaft.formula.equilibrium.phi_from_Bphi`.

    Numerical notes
    ---------------
    A Riemann sum with caller-supplied areas, not a quadrature rule; first
    order in the cell size.  Tracked in #358.
    """
    return np.sum(B_phi * dA)


# ------------------------------------------------------------------
# Fitting Utilities
# ------------------------------------------------------------------

def make_fit_function(mode):
    r"""Build a 1-D parametric model $f(x; c_0, c_1, \dots)$ for profile fitting.

    $$\begin{aligned}
      \text{polynomial:}\ & (1-x)\,\textstyle\sum_k c_kx^k &
      \text{free\_polynomial:}\ & \textstyle\sum_k c_kx^k \\
      \text{exponential:}\ & (1-x)\exp\big(\textstyle\sum_k c_kx^k\big) &
      \text{free\_exponential:}\ & \exp\big(\textstyle\sum_k c_kx^k\big)
    \end{aligned}$$

    Parameters
    ----------
    mode : str
        Model name, case-insensitive [str].
        One of ``'polynomial'``, ``'free_polynomial'``, ``'exponential'``,
        ``'free_exponential'`` (a few aliases are accepted).

    Returns
    -------
    callable
        ``f(x, *coeffs)`` evaluating the model, in the unit of the data [any].

    Raises
    ------
    ValueError
        For an unknown mode.

    Assumptions
    -----------
    ``x`` is a normalised radius on $[0, 1]$: the $(1 - x)$ factor forces the
    constrained modes to zero at $x = 1$; the exponential modes are strictly
    positive and decay monotonically when the polynomial is decreasing.
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
    elif mode in {'free_polynomial', 'polynomial_unconstrained', 'unconstrained_polynomial'}:
        # plain poly(x) -> no boundary constraint at x=1
        def func(x, *coeffs):
            x = np.asarray(x, dtype=float)
            s = 0.0
            for k in range(len(coeffs)):
                s = s + coeffs[k] * x**k
            return s
    elif mode == 'exponential':
        # (1-x)*exp(poly(x)) -> goes to 0 at x=1, stays positive
        def func(x, *coeffs):
            x = np.asarray(x, dtype=float)
            s = 0.0
            for k in range(len(coeffs)):
                s = s + coeffs[k] * x**k
            return (1.0 - x) * np.exp(s)
    elif mode in {'free_exponential', 'exp_free', 'exponential_unconstrained'}:
        # exp(poly(x)) -> always > 0, NO edge-zero; monotonic decay if poly decreasing.
        # Edge value exp(poly(1)) stays small-but-finite (never exactly 0, never rises if c1<0).
        def func(x, *coeffs):
            x = np.asarray(x, dtype=float)
            s = 0.0
            for k in range(len(coeffs)):
                s = s + coeffs[k] * x**k
            return np.exp(s)
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
    r"""Fit a 1-D profile with a selectable model and evaluate it on a grid.

    Least-squares (``scipy.optimize.curve_fit``) for the parametric modes,
    Gaussian-process regression (``sklearn``) for ``'gp'``, linear
    interpolation for ``'linear'``, a core-polynomial/edge-exponential blend
    with a $\tanh$ transition for ``'core_poly_edge_exp'``, and square-root
    modes that fit $y^2$ and return $\sqrt{f}$.

    Parameters
    ----------
    x : array-like
        Data abscissae, 1-D [any].
    y : array-like
        Data values [any].
    y_std : array-like or None
        Per-point uncertainty, same unit as ``y``; ``None`` for unweighted [any].
    x_eval : array-like
        Evaluation grid [any].
    order : int, optional
        Number of polynomial coefficients (degree ``order - 1``); default 3 [-].
    uncertainty_option : int, optional
        1 (default) weights by ``y_std`` when given; 0 ignores it [-].
    fitting_function : str, optional
        Model name, default ``'polynomial'`` [str].
        One of ``'gp'``, ``'polynomial'``, ``'free_polynomial'``,
        ``'exponential'``, ``'free_exponential'``, ``'linear'``,
        ``'core_poly_edge_exp'``, ``'sqrt'``, ``'sqrt_exponential'``.
    gp_kernel : sklearn kernel or None, optional
        Kernel for the GP mode; default constant times RBF [n/a].
    gp_anchor : tuple or None, optional
        ``(x_anchor, y_anchor, y_std_anchor)`` extra points for the GP [n/a].
    n_restarts_optimizer : int, optional
        GP hyperparameter restarts; default 5 [-].

    Returns
    -------
    y_eval : np.ndarray
        Fitted values on ``x_eval`` [any].
    y_std_eval : np.ndarray
        Fitted uncertainty; non-zero for the GP mode only [any].
    fit_function : callable
        ``f(x)`` evaluating the fit at arbitrary ``x`` [n/a].
    coeffs : np.ndarray or None
        Fitted coefficients; ``None`` for the GP and linear modes [any].

    Raises
    ------
    ValueError
        Fewer than two valid points after masking, or an unknown mode.
    RuntimeError
        When ``curve_fit`` does not converge.

    Assumptions
    -----------
    ``x`` is a normalised radius on $[0, 1]$ for the constrained modes (they
    force zero at $x = 1$); uncertainties are one-sigma and independent.

    Limitations
    -----------
    Non-finite points and non-positive ``y_std`` are dropped with a warning.
    The ``'linear'`` mode uses ``numpy.interp``, which holds the end values
    constant outside the data range (silent constant extrapolation; tracked in
    #359).  The square-root modes clip negative data to zero before squaring,
    biasing the fit where the data cross zero, and report zero uncertainty.
    A fit that returns its initial guess unchanged is reported by a warning,
    not an exception.

    Numerical notes
    ---------------
    The initial guess is scaled to ``max|y|`` so raw densities in m^-3 do not
    stall the optimiser; ``maxfev=20000``.

    Examples
    --------
    >>> fit_profile(x, y, y_std, x_eval, fitting_function='gp')
    >>> fit_profile(x, y, y_std, x_eval, order=3, fitting_function='polynomial')
    >>> fit_profile(x, y, y_std, x_eval, fitting_function='linear')
    >>> fit_profile(x, y, y_std, x_eval, order=3, fitting_function='core_poly_edge_exp')
    """
    import warnings

    # --- input sanitization: accept lists, mask non-finite / non-positive-sigma points ---
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if y_std is not None:
        valid &= np.isfinite(y_std) & (y_std > 0)
    if not np.all(valid):
        warnings.warn(
            f"fit_profile: dropped {int(np.sum(~valid))} invalid data point(s) "
            "(non-finite value or non-positive sigma)"
        )
        x, y = x[valid], y[valid]
        if y_std is not None:
            y_std = y_std[valid]
    if x.size < 2:
        raise ValueError(
            "fit_profile requires at least 2 valid data points after masking"
        )

    if fitting_function.lower() == 'gp':
        kernel = gp_kernel or (C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.3, length_scale_bounds=(0.05, 5.0)))

        if gp_anchor is not None:
            x_anchor, y_anchor, y_std_anchor = gp_anchor
            x_gp = np.append(x.ravel(), np.ravel(x_anchor))
            y_gp = np.append(y, np.ravel(y_anchor))
            y_std_base = (
                y_std
                if y_std is not None
                else np.full(x.size, 1e-5 * max(1.0, float(np.max(np.abs(y)))))
            )
            y_std_gp = np.append(y_std_base, np.ravel(y_std_anchor))
        else:
            x_gp = x.ravel()
            y_gp = y
            y_std_gp = y_std

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=y_std_gp**2 if y_std_gp is not None else 1e-10,
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
        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]

        def fit_function(x_input):
            x_arr = np.asarray(x_input, float)
            return np.interp(x_arr, x_sorted, y_sorted)

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

    # Scale-aware initial guess: p0=0.1 makes curve_fit silently return the
    # initial guess for large-magnitude data (e.g. raw ne in m^-3).
    y_scale = float(np.max(np.abs(y)))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        y_scale = 1.0
    exp_modes = {
        'exponential', 'free_exponential', 'exp_free', 'exponential_unconstrained'
    }
    p0 = np.full(order, 0.1, dtype=float)
    if fitting_function.lower() in exp_modes:
        p0[:] = 0.0
        p0[0] = np.log(y_scale)
    else:
        p0[0] = y_scale

    try:
        if uncertainty_option == 1 and y_std is not None:
            coeffs, _ = curve_fit(
                func, x.ravel(), y, sigma=y_std, absolute_sigma=True, p0=p0,
                maxfev=20000,
            )
        else:
            coeffs, _ = curve_fit(func, x.ravel(), y, p0=p0, maxfev=20000)
    except RuntimeError as exc:
        raise RuntimeError(
            f"fit_profile: curve_fit failed to converge for mode "
            f"'{fitting_function}' (order={order}, {x.size} points, "
            f"max|y|={y_scale:.3g}): {exc}"
        ) from exc
    if np.allclose(coeffs, p0):
        warnings.warn(
            f"fit_profile: '{fitting_function}' fit returned its initial guess "
            "unchanged — the optimizer likely failed; inspect the data scale."
        )

    y_eval = func(x_eval, *coeffs)
    y_std_eval = np.zeros_like(y_eval)

    def fit_function(x_input):
        x_arr = np.asarray(x_input, float)
        return func(x_arr, *coeffs)

    return y_eval, y_std_eval, fit_function, coeffs
