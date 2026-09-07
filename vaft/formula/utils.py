"""
Utility functions for plasma physics calculations.

This module provides common utility functions and fitting utilities used throughout
the formula module.
"""

import warnings

import numpy as np
from typing import Union, Tuple, List, Dict
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import curve_fit, minimize
from vaft.compat import trapz_compat


#: What ``from vaft.formula.utils import *`` binds, and therefore what
#: reaches ``vaft.formula.__all__``. Profile fitting and small numerical helpers.
#: Declared so the package stops re-exporting this module's own imports --
#: ``np``, ``warnings``, ``Union``, ``curve_fit`` -- as though they were
#: formulas (#368).
__all__ = [
    "calculate_peaking_factor",
    "calculate_poloidal_flux",
    "calculate_toroidal_flux",
    "calculate_volume_weighted_average",
    "fit_profile",
    "gp_fit",
    "gradient",
    "make_fit_function",
    "normalize_profile",
    "trapz_integral",
]


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




# ---------------------------------------------------------------------------
# Gaussian-process regression (issue #426)
#
# Written against scipy so that scikit-learn is an optional dependency rather
# than a hard one: it was imported at module scope for this single fitting
# mode, and every module that needed `gradient` paid the whole import.
# ---------------------------------------------------------------------------

#: Kernel hyperparameter bounds, matching the sklearn kernel this replaced
#: (``ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.3, (0.05, 5.0))``).
_GP_CONSTANT_BOUNDS = (1e-3, 1e3)
_GP_LENGTH_SCALE_BOUNDS = (0.05, 5.0)
_GP_JITTER = 1e-10


def _gp_kernel(xa: np.ndarray, xb: np.ndarray, constant: float, length_scale: float) -> np.ndarray:
    """Squared-exponential covariance, ``constant * exp(-d^2 / 2 l^2)``."""
    distance = xa.reshape(-1, 1) - xb.reshape(1, -1)
    return constant * np.exp(-0.5 * (distance / length_scale) ** 2)


def _gp_posterior(x, y, noise_variance, x_eval, constant, length_scale):
    """Posterior mean and standard deviation at ``x_eval``, hyperparameters fixed.

    Cholesky rather than an explicit inverse: the covariance is symmetric
    positive definite by construction, and ``cho_solve`` is both faster and
    better conditioned than forming ``K^-1``.
    """
    gram = _gp_kernel(x, x, constant, length_scale)
    gram[np.diag_indices_from(gram)] += noise_variance + _GP_JITTER
    factor = cho_factor(gram, lower=True)
    weights = cho_solve(factor, y)

    cross = _gp_kernel(x_eval, x, constant, length_scale)
    mean = cross @ weights

    solved = cho_solve(factor, cross.T)
    variance = constant - np.einsum("ij,ji->i", cross, solved)
    return mean, np.sqrt(np.clip(variance, 0.0, None))


def _gp_negative_log_marginal_likelihood(theta, x, y, noise_variance):
    """-log p(y | x) for log-hyperparameters ``theta = (log c, log l)``."""
    constant, length_scale = np.exp(theta)
    gram = _gp_kernel(x, x, constant, length_scale)
    gram[np.diag_indices_from(gram)] += noise_variance + _GP_JITTER
    try:
        factor = cho_factor(gram, lower=True)
    except np.linalg.LinAlgError:
        return np.inf
    weights = cho_solve(factor, y)
    log_determinant = 2.0 * np.sum(np.log(np.diag(factor[0])))
    return 0.5 * (y @ weights + log_determinant + y.size * np.log(2.0 * np.pi))


def gp_fit(x, y, y_std, x_eval, *, n_restarts_optimizer: int = 5, random_state: int = 0):
    """Fit a squared-exponential Gaussian process and evaluate it on a grid.

    Parameters
    ----------
    x : array-like
        Data abscissae, 1-D [any].
    y : array-like
        Data values [any].
    y_std : array-like or None
        Per-point one-sigma uncertainty, entering as the diagonal noise;
        ``None`` uses a numerical jitter [any].
    x_eval : array-like
        Evaluation grid [any].
    n_restarts_optimizer : int, optional
        Extra random restarts of the marginal-likelihood optimisation, on top of
        the one from the default hyperparameters; default 5 [-].
    random_state : int, optional
        Seed for the restart draws, so a fit is reproducible; default 0 [-].

    Returns
    -------
    mean : np.ndarray
        Posterior mean on ``x_eval`` [any].
    std : np.ndarray
        Posterior standard deviation on ``x_eval`` [any].

    Numerical notes
    ---------------
    ``y`` is standardised before fitting and the posterior is mapped back,
    which is what makes one set of hyperparameter bounds work for data in
    m^-3 and in keV alike.  The marginal likelihood is optimised with
    ``scipy.optimize.minimize`` (L-BFGS-B) over the log-hyperparameters.

    References
    ----------
    .. [1] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for
           Machine Learning*, MIT Press (2006), Algorithm 2.1 and Eq. (5.8).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)

    # Standardise, as sklearn's normalize_y=True does: the hyperparameter
    # bounds are then scale-free.
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y))
    if not np.isfinite(y_scale) or y_scale == 0.0:
        y_scale = 1.0
    y_normalised = (y - y_mean) / y_scale

    if y_std is None:
        noise_variance = np.full(x.size, _GP_JITTER)
    else:
        noise_variance = (np.asarray(y_std, dtype=float).reshape(-1) / y_scale) ** 2

    bounds = [np.log(_GP_CONSTANT_BOUNDS), np.log(_GP_LENGTH_SCALE_BOUNDS)]
    starts = [np.array([np.log(1.0), np.log(0.3)])]
    rng = np.random.default_rng(random_state)
    for _ in range(max(0, int(n_restarts_optimizer))):
        starts.append(np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1])]))

    best_theta, best_value = starts[0], np.inf
    for start in starts:
        result = minimize(
            _gp_negative_log_marginal_likelihood,
            start,
            args=(x, y_normalised, noise_variance),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_theta, best_value = result.x, float(result.fun)

    constant, length_scale = np.exp(best_theta)
    mean, std = _gp_posterior(
        x, y_normalised, noise_variance, x_eval, constant, length_scale
    )
    return mean * y_scale + y_mean, std * y_scale


def _guarded_ratio(numerator, denominator, *, what: str, because: str):
    """Divide, or warn and return NaN when the denominator vanishes.

    NaN rather than an exception because these ratios sit under plotting and
    summary paths, where one degenerate slice should blank a point rather than
    take down the figure; a warning rather than a silent NaN because the
    degenerate case is real -- packaged VEST samples contain a slice whose axis
    and boundary flux are equal -- and it used to propagate as ``inf`` with
    nothing to say where it started.

    Mirrors ``vaft.formula.equilibrium._virial_ratio``, which makes the same
    trade for the Shafranov closures but is called in tight loops where the
    warning would be noise.
    """
    denominator = np.asarray(denominator, dtype=float)
    degenerate = ~np.isfinite(denominator) | (denominator == 0.0)
    if np.any(degenerate):
        warnings.warn(
            f"{what}: {because} is zero or non-finite, so the ratio is undefined; "
            "returning nan",
            RuntimeWarning,
            stacklevel=3,
        )
        safe = np.where(degenerate, np.nan, denominator)
        return np.asarray(numerator, dtype=float) / safe
    return numerator / denominator


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

    Numerical notes
    ---------------
    A degenerate profile whose axis and boundary values are equal -- which
    packaged VEST samples do contain -- warns and returns ``nan`` rather than
    propagating ``inf``.
    """
    return _guarded_ratio(
        x - x_axis,
        x_boundary - x_axis,
        what="normalize_profile",
        because="x_boundary - x_axis",
    )


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

    Numerical notes
    ---------------
    A zero volume average warns and returns ``nan``.
    """
    return _guarded_ratio(
        central, volume_avg, what="calculate_peaking_factor", because="volume_avg"
    )


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

    Numerical notes
    ---------------
    A zero total volume warns and returns ``nan``.
    """
    return _guarded_ratio(
        np.sum(x * V),
        np.sum(V),
        what="calculate_volume_weighted_average",
        because="sum(V)",
    )


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
        One of ``'gp'`` (scipy), ``'gp_sklearn'``, ``'polynomial'``, ``'free_polynomial'``,
        ``'exponential'``, ``'free_exponential'``, ``'linear'``,
        ``'core_poly_edge_exp'``, ``'sqrt'``, ``'sqrt_exponential'``.
    gp_kernel : sklearn kernel or None, optional
        Kernel for ``'gp_sklearn'`` only; ignored, with a warning, by the scipy
        ``'gp'`` mode; default constant times RBF [n/a].
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

    if fitting_function.lower() in {'gp', 'gp_sklearn'}:
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

        if fitting_function.lower() == 'gp_sklearn':
            # Imported here, not at module scope: this branch is the only thing
            # in vaft.formula that needs scikit-learn, and importing it eagerly
            # made it a hard dependency of the whole package (#426).
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "fitting_function='gp_sklearn' needs scikit-learn, which is an "
                    "optional dependency: pip install 'vaft[sklearn]', or use "
                    "fitting_function='gp' for the scipy implementation."
                ) from exc

            kernel = gp_kernel or (
                C(1.0, _GP_CONSTANT_BOUNDS) * RBF(0.3, _GP_LENGTH_SCALE_BOUNDS)
            )
            # normalize_y standardizes the target but leaves alpha alone, so the
            # noise has to be handed over already in normalized units -- passing
            # the raw variance understates it by var(y), which is what this code
            # used to do.
            scale = float(np.std(y_gp)) or 1.0
            alpha = (np.asarray(y_std_gp, float) / scale) ** 2 if y_std_gp is not None else _GP_JITTER
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=n_restarts_optimizer,
            )
            gp.fit(x_gp[:, None], y_gp)
            y_eval, y_std_eval = gp.predict(x_eval[:, None], return_std=True)

            def fit_function(x_input):
                x_arr = np.asarray(x_input, float).reshape(-1, 1)
                return gp.predict(x_arr)

            return y_eval, y_std_eval, fit_function, None

        if gp_kernel is not None:
            warnings.warn(
                "gp_kernel describes a scikit-learn kernel object and is ignored "
                "by the scipy implementation; pass fitting_function='gp_sklearn' "
                "to use it.",
                RuntimeWarning,
                stacklevel=2,
            )

        y_eval, y_std_eval = gp_fit(
            x_gp, y_gp, y_std_gp, x_eval, n_restarts_optimizer=n_restarts_optimizer
        )

        def fit_function(x_input):
            x_arr = np.asarray(x_input, float).reshape(-1)
            mean, _ = gp_fit(
                x_gp, y_gp, y_std_gp, x_arr, n_restarts_optimizer=n_restarts_optimizer
            )
            return mean

        return y_eval, y_std_eval, fit_function, None

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
