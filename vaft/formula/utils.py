"""
Utility functions for plasma physics calculations.

This module provides common utility functions and fitting utilities used throughout
the formula module.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from typing import Union, Tuple, List, Dict, NamedTuple
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import curve_fit, minimize
from vaft.compat import trapz_compat


#: What ``from vaft.formula.utils import *`` binds, and therefore what
#: reaches ``vaft.formula.__all__``. Profile fitting and small numerical helpers.
#: Declared so the package stops re-exporting this module's own imports --
#: ``np``, ``warnings``, ``Union``, ``curve_fit`` -- as though they were
#: formulas.
__all__ = [
    "calculate_peaking_factor",
    "calculate_poloidal_flux",
    "calculate_toroidal_flux",
    "calculate_volume_weighted_average",
    "eped_tanh_bounds",
    "fit_eped_tanh",
    "fit_profile",
    "gp_fit",
    "gradient",
    "make_fit_function",
    "normalize_profile",
    "normalized_gradient_scale_length",
    "tanh_pedestal_geometry",
    "trapz_integral",
]


# ------------------------------------------------------------------
# Basic Utilities
# ------------------------------------------------------------------

def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Derivative $dy/dx$ along the first axis of a sampled profile or map.

    $$\frac{dy}{dx}\Big|_i \approx \frac{y_{i+1} - y_{i-1}}{x_{i+1} - x_{i-1}}$$

    (second-order central difference on a possibly non-uniform grid).

    Parameters
    ----------
    x : np.ndarray
        Independent variable, monotonic [any].
    y : np.ndarray
        Dependent variable whose first axis has the length of ``x``: a 1-D
        profile or an N-D map indexed ``[x, ...]`` [any].

    Returns
    -------
    np.ndarray
        $dy/dx$ at every sample, of the shape of ``y``, in units of ``y`` per
        unit ``x`` [any].

    Numerical notes
    ---------------
    Wraps ``numpy.gradient(y, x, axis=0)``: second-order accurate in the
    interior, first-order one-sided at the two end points, and
    noise-amplifying (a relative noise $\delta$ on ``y`` becomes
    $\delta/\Delta x$ on the derivative).  Needs at least two samples; only
    the first axis of an N-D ``y`` is differentiated, each column
    independently, so the result equals stacking the 1-D derivatives.
    """
    return np.gradient(y, x, axis=0)


def normalized_gradient_scale_length(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    r"""Normalised inverse gradient scale length $a/L_y$ of a sampled profile.

    $$\frac{a}{L_y} = -\frac{a}{y}\,\frac{dy}{dx}$$

    Parameters
    ----------
    x : np.ndarray
        Radial coordinate, 1-D and monotonic, e.g. minor radius $r$ [m].
    y : np.ndarray
        Profile sampled on ``x`` along its last axis, e.g. $T_e$ or $n_e$;
        shape ``(..., len(x))`` [any].
    a : float
        Normalising length, in the unit of ``x``; for a gyrokinetic input the
        minor radius [m].

    Returns
    -------
    np.ndarray
        $a/L_y$ at every sample, of the shape of ``y``; ``nan`` where ``y``
        is zero or non-finite, with a ``RuntimeWarning`` [-].

    Convention
    ----------
    Positive for a profile that falls with ``x`` -- the usual peaked core
    profile -- so a larger number is a steeper drive.  ``x`` and ``a`` must
    share a unit and ``x`` must be the radius $a$ normalises: on $\rho_N$ or
    $\sqrt{\psi_N}$ pass ``a = 1`` and the result is $-d\ln y/d\rho$, which
    differs from $a/L_y$ by the Jacobian $d\rho/dr$.  The same holds for
    $a/L_{T_e}$, $a/L_{n_e}$ and $a/L_{p}$: only the profile changes.

    Numerical notes
    ---------------
    The derivative is ``numpy.gradient(y, x, axis=-1)``, identical to
    :func:`gradient` for a 1-D profile: second-order central differences in
    the interior, first-order one-sided at the two ends.  Differentiation
    amplifies noise, and dividing by ``y`` amplifies it again where the
    profile is small -- the edge -- so smooth or fit the profile first.

    Reduction
    ---------
    input: profile_1d
    output: profile_1d
    kind: differential
    locality: flux_surface_local
    role: closure_input

    References
    ----------
    .. [1] G. M. Staebler, J. E. Kinsey and R. E. Waltz, Phys. Plasmas 14
           (2007) 055909 (the $a/L_T$, $a/L_n$ drive inputs of TGLF).

    See Also
    --------
    gradient
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    derivative = np.gradient(y, x, axis=-1)
    return _guarded_ratio(
        -float(a) * derivative,
        y,
        what="normalized_gradient_scale_length",
        because="the profile value y",
    )


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
# Gaussian-process regression
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


def _target_scale(values) -> float:
    """The standard deviation to normalise a target by, or 1.0 when it has none.

    Both Gaussian-process paths normalise by this, so both reject a
    non-finite scale the same way; ``float(np.std(...)) or 1.0`` does not,
    because NaN is truthy.
    """
    scale = float(np.std(np.asarray(values, dtype=float)))
    return scale if np.isfinite(scale) and scale != 0.0 else 1.0


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


class _GPState(NamedTuple):
    """One trained Gaussian process: the data it saw and the fit it settled on.

    Split out of :func:`gp_fit` so a caller that evaluates the same fit many
    times -- :func:`fit_profile` hands back a callable that does exactly that --
    pays the hyperparameter search once instead of once per evaluation.
    """

    x: np.ndarray
    y_normalised: np.ndarray
    noise_variance: np.ndarray
    constant: float
    length_scale: float
    y_mean: float
    y_scale: float


def _gp_train(x, y, y_std, *, n_restarts_optimizer: int = 5, random_state: int = 0) -> "_GPState":
    """Standardise, optimise the marginal likelihood, and return the fit."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"gp_fit: x and y must be the same length, got {x.size} and {y.size}")

    # Checked here rather than left to scipy: `cho_solve` validates its input
    # and raises "array must not contain infs or NaNs" from three frames down,
    # naming neither the argument nor the caller. `fit_profile` masks its
    # primary data but appends `gp_anchor` points afterwards, so a non-finite
    # anchor reaches this untouched.
    for name, values in (("x", x), ("y", y)):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"gp_fit: {name} contains non-finite values, which the covariance "
                "solve cannot accept; mask them before fitting (a gp_anchor point "
                "is appended after fit_profile's own masking)."
            )

    # Standardise, as sklearn's normalize_y=True does: the hyperparameter
    # bounds are then scale-free.
    y_mean = float(np.mean(y))
    y_scale = _target_scale(y)
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

    # No restart produced a finite marginal likelihood, so there are no
    # hyperparameters to evaluate at. Say which stage gave up rather than
    # letting the posterior raise the same LinAlgError one frame later.
    if not np.isfinite(best_value):
        raise ValueError(
            "gp_fit: no restart produced a finite marginal likelihood, so no "
            "hyperparameters were found. The covariance is singular for this "
            "data -- check for duplicated abscissae, or for a supplied y_std "
            "that is zero where the data repeats."
        )

    constant, length_scale = np.exp(best_theta)
    return _GPState(
        x=x,
        y_normalised=y_normalised,
        noise_variance=noise_variance,
        constant=float(constant),
        length_scale=float(length_scale),
        y_mean=y_mean,
        y_scale=y_scale,
    )


def _gp_evaluate(state: "_GPState", x_eval):
    """Posterior mean and standard deviation of a trained fit, in data units."""
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    mean, std = _gp_posterior(
        state.x, state.y_normalised, state.noise_variance, x_eval,
        state.constant, state.length_scale,
    )
    return mean * state.y_scale + state.y_mean, std * state.y_scale


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
    state = _gp_train(
        x, y, y_std,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    return _gp_evaluate(state, x_eval)


def _guarded_ratio(numerator, denominator, *, what: str, because: str):
    """Divide, or warn and return NaN when the denominator vanishes.

    NaN rather than an exception because these ratios sit under plotting and
    summary paths, where one degenerate slice should blank a point rather than
    take down the figure; a warning rather than a silent NaN because the
    degenerate case is real -- packaged VEST samples contain a slice whose axis
    and boundary flux are equal -- and it used to propagate as ``inf`` with
    nothing to say where it started.

    Mirrors ``vaft.formula.virial._virial_ratio``, which makes the same
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

    Reduction
    ---------
    input: scalar_0d
    output: scalar_0d
    kind: dimensionless_normalization
    locality: global
    role: profile_descriptor
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


def _rectilinear_axis(coord: np.ndarray, axis: int, name: str, shape: tuple) -> np.ndarray:
    """1-D axis of a grid given as a 1-D axis or an ``ij``-indexed 2-D mesh."""
    coord = np.asarray(coord, dtype=float)
    if coord.ndim == 1:
        if coord.shape[0] != shape[axis]:
            raise ValueError(
                f"{name} has {coord.shape[0]} points but B_phi has {shape[axis]} along axis {axis}"
            )
        line = coord
    elif coord.ndim == 2:
        if coord.shape != shape:
            raise ValueError(f"{name} mesh has shape {coord.shape}, B_phi has {shape}")
        line = coord[:, 0] if axis == 0 else coord[0, :]
    else:
        raise ValueError(f"{name} must be a 1-D axis or a 2-D mesh, got ndim={coord.ndim}")
    if line.size < 2 or not np.all(np.isfinite(line)):
        raise ValueError(f"{name} needs at least two finite points")
    step = np.diff(line)
    if not (np.all(step > 0) or np.all(step < 0)):
        raise ValueError(f"{name} must be strictly monotonic")
    if coord.ndim == 2:
        expected = line[:, None] if axis == 0 else line[None, :]
        # A mesh read through float32 differs from its first row/column by a
        # few float32 ulps; allow that, and never more than a sliver of a cell.
        atol = max(1e-6 * np.min(np.abs(step)), 4 * np.finfo(np.float32).eps * np.max(np.abs(line)))
        if not np.allclose(coord, np.broadcast_to(expected, shape), rtol=0.0, atol=atol):
            raise ValueError(
                f"{name} is not a rectilinear 'ij' mesh (it varies along the other axis)"
            )
    return line


def _toroidal_flux(B_phi, R, Z, mask, dA, stacklevel: int) -> float:
    """Shared body of :func:`calculate_toroidal_flux` and ``phi_from_Bphi``."""
    if dA is None and Z is None and R is not None:
        if np.ndim(B_phi) == 2 and np.ndim(R) != 0:
            # A 2-D map with a second argument and no Z is ambiguous: a grid
            # call that forgot Z, or the old area map.  Summing B_phi * R would
            # be silently wrong, so refuse rather than guess.
            raise TypeError(
                "toroidal flux over a 2-D map needs both R and Z; for the deprecated "
                "area-element sum pass dA= by keyword"
            )
        # Legacy positional call ``f(B_phi, dA)`` on samples, or with a scalar
        # cell area (which can never be an R axis): the bare sum.
        dA, R = R, None
    if dA is not None:
        if R is not None or Z is not None or mask is not None:
            raise TypeError("pass either the grid (R, Z[, mask]) or the legacy dA, not both")
        warnings.warn(
            "toroidal flux from caller-supplied area elements dA is deprecated (#358): it is a bare "
            "Riemann sum whose accuracy is the caller's grid. Pass the (R, Z) grid and an optional "
            "mask instead; the function then integrates by the trapezoidal rule.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return float(np.sum(np.asarray(B_phi, dtype=float) * np.asarray(dA, dtype=float)))
    if R is None or Z is None:
        raise TypeError("toroidal flux needs the grid: pass both R and Z")
    B_phi = np.asarray(B_phi, dtype=float)
    if B_phi.ndim != 2:
        raise ValueError(f"B_phi must be a 2-D map indexed [R, Z], got ndim={B_phi.ndim}")
    r_line = _rectilinear_axis(R, 0, "R", B_phi.shape)
    z_line = _rectilinear_axis(Z, 1, "Z", B_phi.shape)
    integrand = B_phi
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != B_phi.shape:
            raise ValueError(f"mask has shape {mask.shape}, B_phi has {B_phi.shape}")
        if mask.dtype.kind not in "biu":
            # A float mask turns nan into True; insist on the boolean it means.
            raise ValueError(f"mask must be boolean, got dtype {mask.dtype}")
        integrand = np.where(mask.astype(bool), B_phi, 0.0)
    flux = trapz_compat(trapz_compat(integrand, x=z_line, axis=1), x=r_line)
    # Integrate over the area, whichever way the axes are stored.
    return float(flux * np.sign(r_line[-1] - r_line[0]) * np.sign(z_line[-1] - z_line[0]))


def calculate_toroidal_flux(B_phi: np.ndarray,
                            R: np.ndarray = None,
                            Z: np.ndarray = None,
                            mask: np.ndarray = None,
                            *,
                            dA: np.ndarray = None) -> float:
    r"""Toroidal flux $\Phi = \int B_\varphi\,dR\,dZ$ over a rectilinear $(R, Z)$ grid.

    $$\Phi = \iint_{S} B_\varphi(R, Z)\,dR\,dZ$$

    by the 2-D trapezoidal rule, over the whole grid or over the cells a
    ``mask`` selects.

    Parameters
    ----------
    B_phi : np.ndarray
        Toroidal field on the grid, indexed ``[R, Z]`` (IMAS ``profiles_2d``
        order) [T].
    R : np.ndarray, optional
        Major radius: a 1-D strictly monotonic axis of length ``B_phi.shape[0]``,
        or an ``ij``-indexed 2-D mesh of the shape of ``B_phi`` [m].
    Z : np.ndarray, optional
        Height: a 1-D strictly monotonic axis of length ``B_phi.shape[1]``, or an
        ``ij``-indexed 2-D mesh of the shape of ``B_phi`` [m].
    mask : np.ndarray, optional
        Boolean map of the shape of ``B_phi``; ``True`` marks the samples inside
        the cross-section.  ``psi_N <= 1`` alone also selects the private-flux
        region below a diverted X-point; combine it with an inside-the-LCFS
        test there.  Default: the whole grid [-].
    dA : np.ndarray, optional
        Deprecated.  Caller-supplied area elements of the old bare sum
        $\sum_i B_{\varphi,i}\,\Delta A_i$; also taken from a two-argument call
        ``calculate_toroidal_flux(B_phi, dA)`` when ``B_phi`` is not 2-D or ``dA``
        is a scalar cell area (a 2-D area map must pass ``dA=`` by keyword).  Emits a ``DeprecationWarning`` [m^2].

    Returns
    -------
    float
        Toroidal flux through the selected area [Wb].

    Raises
    ------
    ValueError
        If the grid is not rectilinear, an axis is not strictly monotonic, a
        shape does not match ``B_phi``, or ``mask`` is not boolean.
    TypeError
        If neither the grid nor ``dA`` is given, both are, or a 2-D map is
        given a second argument without ``Z``.

    Convention
    ----------
    Full weber (toroidal flux has no per-radian form); sign of $B_\varphi$
    along the COCOS $\varphi$ direction, so a reversed-field discharge flips it
    as surely as a COCOS conversion does.  The area is counted positive
    whichever direction the axes are stored in, so reversing ``R`` or ``Z``
    does not flip the sign.  The physics wrapper is
    :func:`vaft.formula.equilibrium.phi_from_Bphi`.

    Numerical notes
    ---------------
    Trapezoidal rule along ``Z`` then ``R`` through
    :func:`vaft.compat.trapz_compat`: second order in the spacing over the full
    grid, exact for a field bilinear in $(R, Z)$.  With a ``mask`` the integrand
    is set to zero outside it, so the staircase boundary makes the error first
    order in the cell size; refine the grid, not the mask, to converge.  The
    deprecated ``dA`` path is the old Riemann sum, first order and only as good
    as the caller's areas.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.4 (toroidal flux and the safety factor).
    """
    return _toroidal_flux(B_phi, R, Z, mask, dA, stacklevel=3)


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


def _eped_tanh_model(x, f0, f_ped, f_sep, x_ped, width, alpha, beta):
    """EPED-style profile: a tanh pedestal plus a core shape anchored at ``f0``.

    The pedestal is ``f_sep + (f_ped - f_sep)/2 * (1 - tanh((x - x_ped)/width))``.
    Inside ``x_ped`` a ``(1 - (x/x_ped)^alpha)^beta`` term lifts the curve from
    the tanh's own value at the axis to ``f0``, so the core is shaped
    independently of the pedestal and the two meet continuously at ``x_ped``.

    Unlike the modified tanh it replaces, the core correction is *gated* to
    ``x <= x_ped``: a term that multiplies the slope across the whole domain
    diverges at large ``|x - x_ped|``, which is why the legacy fit had to bound
    its slope parameter to keep the divergence off-screen.
    """
    x = np.asarray(x, dtype=float)
    if not width > 0:
        # A non-positive width inverts the pedestal -- the curve would rise
        # toward the edge -- silently.  The bounds keep the fitter away from
        # it; this keeps a direct caller away from it too.
        raise ValueError(f"width must be positive, got {width!r}")
    if not x_ped > 0:
        raise ValueError(f"x_ped must be positive, got {x_ped!r}")
    pedestal = f_sep + 0.5 * (f_ped - f_sep) * (1.0 - np.tanh((x - x_ped) / width))
    axis_pedestal = f_sep + 0.5 * (f_ped - f_sep) * (1.0 + np.tanh(x_ped / width))

    core = np.zeros_like(x)
    inside = x <= x_ped
    if np.any(inside) and x_ped > 0:
        scaled = np.clip(x[inside] / x_ped, 0.0, 1.0)
        core[inside] = np.power(np.clip(1.0 - np.power(scaled, alpha), 0.0, None), beta)
    return pedestal + core * (f0 - axis_pedestal)


def _initial_eped_tanh_guess(x, y):
    """``(p0, lower, upper)`` for :func:`_eped_tanh_model`.

    The percentile seeds and the ``x_ped in [0.75, 0.999]`` /
    ``width in [0.01, 0.3]`` box come from the pedestal-fitting study this
    model is ported from; they keep the fit inside the region where a pedestal
    is physically meaningful rather than letting it chase a core feature.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if not y.size or not np.any(np.isfinite(y)):
        raise ValueError("eped_tanh needs at least one finite sample to seed its fit")
    # Scaled by the data's magnitude, not by its maximum: a profile that is
    # everywhere negative -- a rotation profile, say -- has its *smallest*
    # magnitude at the maximum, so seeding from that would box the amplitudes
    # far inside the data and the fit could never reach it.
    scale = 5.0 * max(float(np.nanmax(np.abs(y))), 1e-12)
    axis, pedestal, separatrix = (
        float(np.nanpercentile(y, level)) for level in (100.0, 80.0, 5.0)
    )
    if abs(float(np.nanmin(y))) > abs(float(np.nanmax(y))):
        # Decreasing-and-negative: the axis is the most negative sample and
        # the separatrix the least, so the percentiles run the other way.
        axis, pedestal, separatrix = (
            float(np.nanpercentile(y, level)) for level in (0.0, 20.0, 95.0)
        )
    p0 = [axis, pedestal, separatrix, 0.94, 0.04, 1.5, 2.0]
    # alpha and beta stay at or above 1: below it the gated core term has an
    # infinite slope at x_ped (beta) or at the magnetic axis (alpha), which no
    # flux-function profile has and which makes any gradient a caller takes
    # from the fit depend on their grid spacing.
    lower = [-scale, -scale, -scale, 0.75, 0.01, 1.0, 1.0]
    upper = [scale, scale, scale, 0.999, 0.3, 8.0, 8.0]
    p0 = [min(max(value, low), high) for value, low, high in zip(p0, lower, upper)]
    return p0, lower, upper


def eped_tanh_bounds(x, y):
    """Parameter box the ``eped_tanh`` fit uses for this data.

    Public so that a caller checking whether a fitted parameter came back
    resting on a bound compares against the same box the fit used, rather than
    re-deriving one that can drift out of step with it.  The amplitude bounds
    scale with the data's magnitude; the shape bounds are fixed.

    Parameters
    ----------
    x : array_like
        Radial coordinate; used only for its length [-].
    y : array_like
        Profile samples, whose magnitude sets the amplitude bounds [any].

    Returns
    -------
    tuple of list
        ``(lower, upper)``, each seven entries ordered as the model's
        arguments ``f0, f_ped, f_sep, x_ped, width, alpha, beta``.  The
        amplitudes are ``+-5 max|y|`` [any]; ``x_ped`` is in ``[0.75, 0.999]``
        and ``width`` in ``[0.01, 0.3]`` [-]; ``alpha`` and ``beta`` are in
        ``[1, 8]`` [-].

    Notes
    -----
    ``alpha`` and ``beta`` stay at or above one because below it the gated
    core term has an infinite slope -- at ``x_ped`` for ``beta``, at the
    magnetic axis for ``alpha`` -- which no flux-function profile has, and
    which would make any gradient taken from the fit depend on the caller's
    grid spacing.

    References
    ----------
    .. [EPED] The seven-parameter form and the ``x_ped``/``width`` box follow
       the pedestal-fitting study ported in migration decision D-05;
       :func:`vaft.process.profile.pedestal_top` is the consumer.
    """
    _, lower, upper = _initial_eped_tanh_guess(x, y)
    return lower, upper


#: The ``eped_tanh`` coefficients by name, in the order :func:`fit_profile`
#: returns and serializes them.  The order is the serialized contract and is
#: never changed; the names only label it.
EPED_TANH_PARAMETERS = ("f0", "f_ped", "f_sep", "x_ped", "width", "alpha", "beta")

#: The unit of each :data:`EPED_TANH_PARAMETERS` entry: ``"y"`` is the unit of
#: the fitted samples, ``"x"`` the radial coordinate's, ``"-"`` dimensionless.
EPED_TANH_PARAMETER_UNITS = ("y", "y", "y", "x", "x", "-", "-")

#: The four coefficients that describe the pedestal itself (``f_ped``,
#: ``f_sep``, ``x_ped``, ``width``); ``f0``, ``alpha`` and ``beta`` only shape
#: the core and are routinely unconstrained by an edge-window fit.
EPED_TANH_PEDESTAL_INDICES = (1, 2, 3, 4)

#: Fraction of a parameter's box within which a fitted value counts as resting
#: on the bound -- the same criterion :func:`vaft.process.profile.pedestal_top`
#: has applied since it was written.
_EPED_BOUND_TOLERANCE = 1e-3

#: Extra ``(x_ped, width)`` seeds tried after the legacy seed, in order.  The
#: data-driven seed (the steepest descent of the samples) is inserted first.
_EPED_EXTRA_SEEDS = ((0.97, 0.02), (0.88, 0.06), (0.92, 0.12))

#: Relative improvement in the objective another start must achieve before it
#: replaces the legacy seed's optimum; a tie keeps the legacy coefficients
#: bit-for-bit.
_EPED_IMPROVEMENT = 1e-6


@dataclass(frozen=True, eq=False)
class EpedTanhFit:
    """Structured result of :func:`fit_eped_tanh`: coefficients, covariance and fit evidence.

    ``coefficients`` keep the legacy order :data:`EPED_TANH_PARAMETERS`, so
    ``fit.coefficients`` is exactly what ``fit_profile(..., 'eped_tanh')``
    returns.  ``covariance`` is ``curve_fit``'s parameter covariance -- with
    ``absolute_sigma=True`` when the fit was weighted by ``y_std``, scaled by
    the reduced chi-square otherwise -- and is all ``nan`` when it could not
    be estimated: unavailable uncertainty is never reported as zero.

    ``width`` is the *native* tanh scale ``w`` of
    ``f_sep + (f_ped - f_sep)/2 [1 - tanh((x - x_ped)/w)]``.  The canonical
    Groebner full width is ``full_width = 2 w``, with ``top = x_ped - w`` and
    ``foot = x_ped + w`` (:func:`tanh_pedestal_geometry`).
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    lower: tuple
    upper: tuple
    success: bool
    message: str
    nfev: int
    chi_squared: float
    n_points: int
    weighted: bool
    n_starts: int
    n_converged: int
    competing_minima: int
    start_messages: tuple = ()

    parameters = EPED_TANH_PARAMETERS
    units = EPED_TANH_PARAMETER_UNITS

    def __call__(self, x):
        return _eped_tanh_model(np.asarray(x, float), *self.coefficients)

    @property
    def degrees_of_freedom(self) -> int:
        """``n_points - 7`` [-]."""
        return int(self.n_points - len(EPED_TANH_PARAMETERS))

    @property
    def reduced_chi_squared(self) -> float:
        """``chi_squared / degrees_of_freedom``; ``nan`` without freedom [-]."""
        dof = self.degrees_of_freedom
        return float(self.chi_squared / dof) if dof > 0 else float("nan")

    @property
    def values(self) -> dict:
        """Coefficients by name [y, x or -]."""
        return dict(zip(EPED_TANH_PARAMETERS, (float(v) for v in self.coefficients)))

    @property
    def covariance_available(self) -> bool:
        """Whether every covariance entry is finite."""
        return bool(np.all(np.isfinite(self.covariance)))

    @property
    def std(self) -> np.ndarray:
        """One-sigma parameter uncertainties; ``nan`` where unavailable."""
        diag = np.diag(self.covariance).astype(float)
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(diag) & (diag >= 0.0), np.sqrt(np.abs(diag)), np.nan)

    @property
    def correlation(self) -> np.ndarray:
        """Parameter correlation matrix; ``nan`` where a variance is zero or unavailable."""
        sigma = self.std
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.covariance / np.outer(sigma, sigma)

    @property
    def bound_saturated(self) -> tuple:
        """Names of the coefficients resting on a bound of the parameter box."""
        out = []
        for name, value, low, high in zip(EPED_TANH_PARAMETERS, self.coefficients, self.lower, self.upper):
            span = high - low
            if min(value - low, high - value) <= _EPED_BOUND_TOLERANCE * span:
                out.append(name)
        return tuple(out)

    @property
    def pedestal_max_correlation(self) -> float:
        """Largest ``|rho|`` between two pedestal coefficients (``f_ped, f_sep, x_ped, width``) [-]."""
        block = self.correlation[np.ix_(EPED_TANH_PEDESTAL_INDICES, EPED_TANH_PEDESTAL_INDICES)]
        off = block[~np.eye(len(EPED_TANH_PEDESTAL_INDICES), dtype=bool)]
        return float(np.max(np.abs(off))) if np.all(np.isfinite(off)) else float("nan")

    @property
    def pedestal_condition_number(self) -> float:
        """Condition number of the pedestal coefficients' correlation block [-]."""
        block = self.correlation[np.ix_(EPED_TANH_PEDESTAL_INDICES, EPED_TANH_PEDESTAL_INDICES)]
        if not np.all(np.isfinite(block)):
            return float("nan")
        return float(np.linalg.cond(block))

    def jacobian(self, function, step_fraction=1e-5):
        """Central-difference Jacobian of ``function(coefficients)`` against the coefficients.

        ``function`` maps a coefficient vector to an output vector.  Steps are
        ``step_fraction * max(|theta_j|, sigma_j, 1e-3 * box_j)`` so a narrow
        ``width`` is not stepped across its own size.
        """
        from vaft.formula.sensitivity import finite_difference_jacobian

        theta = np.asarray(self.coefficients, float)
        box = np.asarray(self.upper, float) - np.asarray(self.lower, float)
        sigma = np.nan_to_num(self.std, nan=0.0)
        step = step_fraction * np.maximum.reduce([np.abs(theta), sigma, 1e-3 * box])
        # x_ped and width must stay positive for the model to evaluate.
        step[3] = min(step[3], 0.5 * theta[3])
        step[4] = min(step[4], 0.5 * theta[4])
        return finite_difference_jacobian(lambda t: np.atleast_1d(function(t)), theta, step=step)

    def propagate(self, function):
        """Linearized covariance ``J C J^T`` of ``function(coefficients)``; ``nan`` without a covariance."""
        from vaft.formula.sensitivity import linear_covariance_propagation

        jac = self.jacobian(function)
        if not self.covariance_available:
            n = jac.shape[0]
            return np.full((n, n), np.nan)
        return linear_covariance_propagation(jac, self.covariance)

    def profile_std(self, x):
        """One-sigma uncertainty of the fitted curve at ``x``, ``sqrt(diag(J C J^T))``; ``nan`` without a covariance."""
        x = np.asarray(x, float)
        flat = x.reshape(-1)
        cov = self.propagate(lambda t: _eped_tanh_model(flat, *t))
        var = np.diag(cov)
        with np.errstate(invalid="ignore"):
            out = np.where(np.isfinite(var), np.sqrt(np.clip(var, 0.0, None)), np.nan)
        return out.reshape(x.shape)


def _eped_steepest_seed(x, y):
    """``x_ped`` seed at the steepest descent of the sorted samples, inside the model's box."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    keep = np.concatenate(([True], np.diff(xs) > 0))
    xs, ys = xs[keep], ys[keep]
    if xs.size < 3:
        return None
    slope = np.gradient(ys, xs)
    # A positive profile drops toward the edge (negative slope); an
    # all-negative one -- a rotation, say -- rises toward zero.
    toward_edge = 1.0 if abs(float(np.nanmin(y))) <= abs(float(np.nanmax(y))) else -1.0
    inside = (xs >= 0.75) & (xs <= 0.999)
    if not np.any(inside):
        return None
    candidate = xs[inside][np.argmax(-toward_edge * slope[inside])]
    return float(candidate)


def fit_eped_tanh(x, y, y_std=None, *, uncertainty_option=1, n_starts=4):
    r"""Fit the seven-parameter EPED-style tanh profile and keep its covariance and fit evidence.

    $$f(x) = f_\mathrm{sep} + \frac{f_\mathrm{ped} - f_\mathrm{sep}}{2}\left[1 - \tanh\frac{x - x_\mathrm{ped}}{w}\right] + C(x)\,\bigl(f_0 - f_\mathrm{axis}\bigr)$$

    with the gated core shape $C(x) = (1 - (x/x_\mathrm{ped})^\alpha)^\beta$
    for $x \le x_\mathrm{ped}$ and zero outside it.  This is the model behind
    ``fit_profile(..., fitting_function='eped_tanh')``; that call returns
    ``fit_eped_tanh(...).coefficients`` unchanged in meaning and order.

    Parameters
    ----------
    x : array_like
        Radial coordinate of the samples, normalized so the LCFS is at one [-].
    y : array_like
        Profile samples [any].
    y_std : array_like, optional
        One-sigma uncertainty per sample; ``None`` fits unweighted [any].
    uncertainty_option : int, optional
        1 (default) weights by ``y_std`` with ``absolute_sigma=True``; 0
        ignores it, as in :func:`fit_profile` [-].
    n_starts : int, optional
        Number of starting points: the legacy seed first, then a data-driven
        seed at the samples' steepest descent and fixed ``(x_ped, w)``
        alternatives; default 4 [-].

    Returns
    -------
    EpedTanhFit
        Coefficients ``(f0, f_ped, f_sep, x_ped, width, alpha, beta)``, their
        covariance (``nan`` when unavailable), the parameter box, the
        optimizer's status, chi-square and multi-start record [y, x or -].

    Raises
    ------
    ValueError
        Non-finite samples, non-positive ``y_std``, or mismatched lengths.
    RuntimeError
        No starting point converged.

    Convention
    ----------
    ``width`` is the native tanh scale $w$.  The canonical Groebner full width
    is $\Delta = 2w$, pedestal top $x_\mathrm{ped} - \Delta/2$ and foot
    $x_\mathrm{ped} + \Delta/2$ (:func:`tanh_pedestal_geometry`), the
    convention of :func:`vaft.formula.equilibrium.modified_tanh_profile`.
    ``f_ped`` and ``f_sep`` are the tanh's *asymptotic* levels, not the
    profile's value at the top, foot or separatrix.

    Numerical notes
    ---------------
    Each start is a bounded ``scipy.optimize.curve_fit`` (trust-region
    reflective, ``max_nfev = 20000``) in the box of :func:`eped_tanh_bounds`.
    The legacy seed's optimum is kept unless another start lowers the
    objective by more than a relative ``1e-6``, so a fit the single-start
    code already solved returns bit-identical coefficients.  Starts that end
    within one unit of chi-square (weighted) or one percent (unweighted) of
    the best but with ``x_ped`` more than one ``w`` away are counted in
    ``competing_minima``: the data do not single out one pedestal.

    Uncertainty propagation
    -----------------------
    ``EpedTanhFit.profile_std`` and ``EpedTanhFit.propagate`` linearize any
    derived quantity $g(\theta)$ as $\sigma_g^2 = J_g C_\theta J_g^\mathsf{T}$
    with a central-difference Jacobian; with a weakly identified fit the
    linear covariance can be unreliable, which the correlation and condition
    diagnostics on the result report.

    References
    ----------
    .. [1] R. J. Groebner and T. N. Carlstrom, *Critical edge parameters for
           H-mode transition in DIII-D*, Plasma Phys. Control. Fusion 40, 673
           (1998).
    .. [2] P. B. Snyder et al., *A first-principles predictive model of the
           pedestal height and width*, Nucl. Fusion 51, 103016 (2011), for the
           EPED-style tanh profile.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length, got {x.size} and {y.size}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("fit_eped_tanh needs finite x and y; mask non-finite samples first")
    weighted = uncertainty_option == 1 and y_std is not None
    sigma = None
    if weighted:
        sigma = np.asarray(y_std, dtype=float).reshape(-1)
        if sigma.shape != y.shape:
            raise ValueError(f"y_std has {sigma.size} samples but y has {y.size}")
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("fit_eped_tanh needs finite, positive y_std; mask invalid sigmas first")
    p0, lower, upper = _initial_eped_tanh_guess(x, y)

    seeds = [list(p0)]
    extra = []
    steep = _eped_steepest_seed(x, y)
    if steep is not None:
        extra.append((steep, 0.03))
    extra.extend(_EPED_EXTRA_SEEDS)
    for x_ped, width in extra:
        if len(seeds) >= max(int(n_starts), 1):
            break
        seed = list(p0)
        seed[3] = min(max(x_ped, lower[3]), upper[3])
        seed[4] = min(max(width, lower[4]), upper[4])
        if any(np.allclose(seed, other) for other in seeds):
            continue
        seeds.append(seed)

    from scipy.optimize import OptimizeWarning

    results = []
    messages = []
    for seed in seeds:
        kwargs = {"p0": seed, "bounds": (lower, upper), "max_nfev": 20000, "full_output": True}
        if weighted:
            kwargs.update(sigma=sigma, absolute_sigma=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            try:
                coeffs, pcov, info, message, status = curve_fit(_eped_tanh_model, x, y, **kwargs)
            except (RuntimeError, ValueError) as error:
                messages.append(f"failed: {error}")
                results.append(None)
                continue
        notes = [str(w.message) for w in caught if issubclass(w.category, OptimizeWarning)]
        cost = float(np.sum(np.asarray(info["fvec"], float) ** 2))
        messages.append(f"{message}" + (f" ({'; '.join(notes)})" if notes else ""))
        results.append((np.asarray(coeffs, float), np.asarray(pcov, float), int(info["nfev"]),
                        cost, str(message), int(status) > 0))

    converged = [i for i, r in enumerate(results) if r is not None and r[5]]
    if not converged:
        raise RuntimeError(
            "eped_tanh: no starting point converged: " + " | ".join(messages)
        )
    best = converged[0]
    for i in converged[1:]:
        if results[i][3] < results[best][3] * (1.0 - _EPED_IMPROVEMENT):
            best = i
    coeffs, pcov, nfev, cost, message, success = results[best]
    tolerance = 1.0 if weighted else 0.01 * cost
    competing = sum(
        1 for i in converged
        if i != best and results[i][3] - cost <= tolerance
        and abs(results[i][0][3] - coeffs[3]) > coeffs[4]
    )
    if not np.all(np.isfinite(pcov)):
        pcov = np.full((len(EPED_TANH_PARAMETERS),) * 2, np.nan)
    return EpedTanhFit(
        coefficients=coeffs,
        covariance=pcov,
        lower=tuple(float(v) for v in lower),
        upper=tuple(float(v) for v in upper),
        success=bool(success),
        message=message,
        nfev=nfev,
        chi_squared=cost,
        n_points=int(x.size),
        weighted=bool(weighted),
        n_starts=len(seeds),
        n_converged=len(converged),
        competing_minima=int(competing),
        start_messages=tuple(messages),
    )


def tanh_pedestal_geometry(center, native_width, center_width_covariance=None):
    r"""Pedestal top, foot and full width of a tanh step, with their linearized uncertainties.

    $$\Delta = 2w,\qquad x_\mathrm{top} = x_c - \frac{\Delta}{2},\qquad x_\mathrm{foot} = x_c + \frac{\Delta}{2}$$

    $$\sigma_\mathrm{top}^2 = \sigma_c^2 + \frac{\sigma_\Delta^2}{4} - \mathrm{Cov}(x_c, \Delta),\qquad \sigma_\mathrm{foot}^2 = \sigma_c^2 + \frac{\sigma_\Delta^2}{4} + \mathrm{Cov}(x_c, \Delta)$$

    for a step written ``(1 - tanh((x - x_c)/w))/2`` -- the native scale $w$
    of the ``eped_tanh`` model -- or equivalently ``tanh(2(x_c - x)/Delta)``
    in Groebner's full-width form.

    Parameters
    ----------
    center : float
        Symmetry point $x_c$ of the step, the centre of the steep-gradient region [-].
    native_width : float
        Native tanh scale $w > 0$ [-].
    center_width_covariance : array_like, optional
        ``2 x 2`` covariance of ``(center, native_width)``; ``None`` returns
        ``nan`` uncertainties rather than zero [-].

    Returns
    -------
    top : float
        Pedestal top (knee), $x_c - w$ [-].
    foot : float
        Pedestal foot, $x_c + w$ [-].
    full_width : float
        Full width $\Delta = 2w$ [-].
    sigma_top : float
        One-sigma uncertainty of ``top``; ``nan`` without a covariance [-].
    sigma_foot : float
        One-sigma uncertainty of ``foot``; ``nan`` without a covariance [-].
    sigma_full_width : float
        One-sigma uncertainty of ``full_width``; ``nan`` without a covariance [-].

    Raises
    ------
    ValueError
        ``native_width`` is not positive, or the covariance is not ``2 x 2``.

    Convention
    ----------
    Groebner and Carlstrom's fit $A\tanh(2(X_\mathrm{sym} - X)/W) + B$ puts the
    knee at $X_\mathrm{sym} - W/2$ with the full width $W$; the same step in
    the native scale has $w = W/2$, so the knee is one native width inside
    the centre.  This is the convention of
    :func:`vaft.formula.equilibrium.modified_tanh_profile` and of
    ``BarrierStep.knee`` / ``BarrierStep.foot``.  The older
    ``PedestalTop.inner_edge``, ``x_c - w/2``, is a quarter of the full width
    inside the centre and is *not* this top.

    Uncertainty propagation
    -----------------------
    Exact for this linear map: $\sigma_\Delta^2 = 4\sigma_w^2$ and
    $\mathrm{Cov}(x_c, \Delta) = 2\,\mathrm{Cov}(x_c, w)$.

    References
    ----------
    .. [1] R. J. Groebner and T. N. Carlstrom, *Critical edge parameters for
           H-mode transition in DIII-D*, Plasma Phys. Control. Fusion 40, 673
           (1998), Fig. 1.
    """
    center = float(center)
    native_width = float(native_width)
    if not native_width > 0.0:
        raise ValueError(f"native_width must be positive, got {native_width!r}")
    full_width = 2.0 * native_width
    top = center - 0.5 * full_width
    foot = center + 0.5 * full_width
    if center_width_covariance is None:
        nan = float("nan")
        return top, foot, full_width, nan, nan, nan
    cov = np.asarray(center_width_covariance, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"center_width_covariance must be 2 x 2, got shape {cov.shape}")
    var_c, var_delta, cov_c_delta = cov[0, 0], 4.0 * cov[1, 1], 2.0 * cov[0, 1]
    with np.errstate(invalid="ignore"):
        sigma_top = float(np.sqrt(var_c + var_delta / 4.0 - cov_c_delta)) if np.isfinite(cov).all() else float("nan")
        sigma_foot = float(np.sqrt(var_c + var_delta / 4.0 + cov_c_delta)) if np.isfinite(cov).all() else float("nan")
        sigma_delta = float(np.sqrt(var_delta)) if np.isfinite(var_delta) else float("nan")
    return top, foot, full_width, sigma_top, sigma_foot, sigma_delta


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
    *,
    random_state=0,
    extrapolate=False,
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
        ``'core_poly_edge_exp'``, ``'eped_tanh'``, ``'sqrt'``,
        ``'sqrt_exponential'``.  ``'eped_tanh'`` fits the seven-parameter
        pedestal model ``(f0, f_ped, f_sep, x_ped, width, alpha, beta)`` and is
        what :func:`vaft.process.profile.pedestal_top` uses.
    gp_kernel : sklearn kernel or None, optional
        Kernel for ``'gp_sklearn'`` only; ignored, with a warning, by the scipy
        ``'gp'`` mode; default constant times RBF [n/a].
    gp_anchor : tuple or None, optional
        ``(x_anchor, y_anchor, y_std_anchor)`` extra points for the GP [n/a].
    n_restarts_optimizer : int, optional
        GP hyperparameter restarts; default 5 [-].
    random_state : int, optional
        Seed for the restart starting points of both GP backends; default 0.
        Without it the ``'gp_sklearn'`` backend would draw its restarts from
        numpy's global ``RandomState`` and the fit would depend on what ran
        before it in the process [-].
    extrapolate : bool, optional
        ``'linear'`` mode only: ``False`` (default) returns ``nan`` (value and
        uncertainty) outside ``[min(x), max(x)]``; ``True`` holds the end values
        constant there, the ``numpy.interp`` clamp.  The parametric and GP
        modes evaluate their model anywhere and do not read it [-].

    Returns
    -------
    y_eval : np.ndarray
        Fitted values on ``x_eval``; ``nan`` outside the data range in the
        ``'linear'`` mode unless ``extrapolate=True`` [any].
    y_std_eval : np.ndarray
        Fitted uncertainty: the GP posterior standard deviation for the GP
        modes; for ``'eped_tanh'`` the linearized parameter-covariance
        uncertainty $\sqrt{\mathrm{diag}(J C J^\mathsf{T})}$, ``nan`` where the
        covariance is unavailable; zero (no uncertainty computed) for the other
        parametric modes; in the ``'linear'`` mode ``nan`` wherever ``y_eval``
        is [any].
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
    The ``'linear'`` mode does not extrapolate unless asked: outside the data
    range it returns ``nan`` by default, and with ``extrapolate=True`` the
    constant end value, reported with zero uncertainty.  The square-root
    modes clip negative data to zero before squaring, biasing the fit where the
    data cross zero; they warn with the number of clipped points, and report
    zero uncertainty.
    A fit that returns its initial guess unchanged is reported by a warning,
    not an exception.  The ``'eped_tanh'`` mode is :func:`fit_eped_tanh`: its
    covariance, status, bound saturation and multi-start record are on
    ``fit_function.eped_fit``; before #2335 its second return was a zero
    placeholder, which a caller must not read as zero uncertainty.

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
            # made it a hard dependency of the whole package.
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
            scale = _target_scale(y_gp)
            alpha = (np.asarray(y_std_gp, float) / scale) ** 2 if y_std_gp is not None else _GP_JITTER
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=n_restarts_optimizer,
                # Without this scikit-learn seeds its restarts from numpy's
                # global RandomState, so the chosen local optimum of the
                # multimodal marginal likelihood depended on whatever ran
                # earlier in the process (cold review 0.8.0: Windows-only
                # disagreement with the scipy backend, which seeds itself).
                random_state=random_state,
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

        # Trained once. The returned callable re-evaluates that fit rather than
        # re-running the hyperparameter search, which is what the sklearn branch
        # gets for free by handing back a fitted estimator's predict.
        gp_state = _gp_train(
            x_gp, y_gp, y_std_gp,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        y_eval, y_std_eval = _gp_evaluate(gp_state, x_eval)

        def fit_function(x_input):
            return _gp_evaluate(gp_state, np.asarray(x_input, float).reshape(-1))[0]

        return y_eval, y_std_eval, fit_function, None

    if fitting_function.lower() == 'linear':
        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]
        x_lo, x_hi = x_sorted[0], x_sorted[-1]

        def fit_function(x_input):
            x_arr = np.asarray(x_input, float)
            if extrapolate:
                return np.interp(x_arr, x_sorted, y_sorted)
            # numpy.interp holds the end values outside the data; a profile
            # past the last channel is missing, not flat.
            return np.interp(x_arr, x_sorted, y_sorted, left=np.nan, right=np.nan)

        y_eval = fit_function(x_eval)
        y_std_eval = np.where(np.isnan(y_eval), np.nan, 0.0)
        coeffs = None
        if not extrapolate and np.any((x_eval < x_lo) | (x_eval > x_hi)):
            warnings.warn(
                f"fit_profile: linear mode returns nan at "
                f"{int(np.sum((x_eval < x_lo) | (x_eval > x_hi)))} evaluation point(s) outside "
                f"the data range [{x_lo:g}, {x_hi:g}]; pass extrapolate=True to hold the end values",
                stacklevel=2,
            )
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
        n_clipped = int(np.sum(y < 0.0))
        if n_clipped:
            warnings.warn(
                f"fit_profile: clipped {n_clipped} negative data point(s) to zero "
                f"before squaring ({fitting_function!r} mode)",
                stacklevel=2,
            )
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
            random_state=random_state,
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


    if fitting_function.lower() in {'eped_tanh', 'eped'}:
        # fit_eped_tanh keeps what curve_fit used to have discarded here: the
        # covariance, the optimizer status and a multi-start record.  The
        # four-tuple is unchanged in shape and coefficient order; the second
        # entry is now the linearized fit uncertainty sqrt(diag(J C J^T)),
        # nan where the covariance could not be estimated -- never the zero
        # placeholder it used to be.  The structured result rides on the
        # returned callable as ``fit_function.eped_fit``.
        eped_fit = fit_eped_tanh(x, y, y_std, uncertainty_option=uncertainty_option)
        coeffs = eped_fit.coefficients

        def fit_function(x_input):
            return _eped_tanh_model(np.asarray(x_input, float), *coeffs)

        fit_function.eped_fit = eped_fit
        y_eval = fit_function(x_eval)
        return y_eval, eped_fit.profile_std(x_eval), fit_function, coeffs

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
