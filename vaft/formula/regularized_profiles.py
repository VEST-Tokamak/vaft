"""Positive, regular flux-function profiles; independent of machine and ODS.

The coordinate is psi_N = rho_pol**2, never rho_tor.  Finite dF/dpsi_N
therefore implies dP/drho_pol(0)=0, without constraining dF/dpsi_N(0).
The legacy ``utils.fit_profile`` interface and all its modes are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.special import expit

from .utils import fit_profile

API_VERSION = "regularized-flux-profile-v1"

__all__ = [
    "API_VERSION", "RegularizedProfileConfig", "RegularizedProfileFit",
    "profile_basis", "evaluate_regularized_profile", "fit_regularized_profile",
]


@dataclass(frozen=True)
class RegularizedProfileConfig:
    """Serializable numerical assumptions, not measurement uncertainties.

    ``scale`` has the unit of y. ``floor`` is in y/scale units. Curvature
    is penalized in u=psi_N/domain_max, using 32-point Gauss-Legendre
    quadrature separately on each knot interval. No monotonicity is imposed.
    """
    basis: str = "bernstein2"
    scale: float = 1.0
    domain_max: float = 1.0
    huber_delta: float = 1.0
    regularization: float = 1e-4
    floor: float = 1e-10
    max_nfev: int = 2000
    tolerance: float = 1e-9
    condition_limit: float = 1e10
    knot_fractions: tuple[float, float] = (0.7, 0.9)

    def __post_init__(self):
        if self.basis not in ("bernstein2", "bernstein3", "bspline6"):
            raise ValueError("unsupported basis")
        positive = (self.scale, self.domain_max, self.huber_delta,
                    self.tolerance, self.condition_limit)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("scales, domain and tolerances must be finite positive")
        if (not np.isfinite(self.floor) or self.floor < 0 or
                not np.isfinite(self.regularization) or self.regularization < 0):
            raise ValueError("floor and regularization must be finite nonnegative")
        if (isinstance(self.max_nfev, (bool, np.bool_)) or
                not isinstance(self.max_nfev, (int, np.integer)) or self.max_nfev < 1):
            raise ValueError("max_nfev must be a positive integer")
        k = tuple(self.knot_fractions)
        if len(k) != 2 or not 0 < k[0] < k[1] < 1:
            raise ValueError("two strictly ordered interior knot fractions required")
        object.__setattr__(self, "knot_fractions", k)

    @property
    def coefficient_count(self):
        return {"bernstein2": 3, "bernstein3": 4, "bspline6": 6}[self.basis]


def profile_basis(psi_n, config, derivative=0):
    r"""Evaluate the positive profile basis or its flux-coordinate derivatives.

    $$B_j(u)={n\choose j}u^j(1-u)^{n-j},\qquad u=\psi_N/d$$

    The spline alternative uses six clamped cubic B-splines [1]_. Ordering is
    ascending Bernstein index or spline index; derivatives are in psi_N.

    Parameters
    ----------
    psi_n : array_like
        Finite normalized poloidal flux coordinates [-].
    config : RegularizedProfileConfig
        Basis, domain and knot declaration; scale is unused here [-].
    derivative : int, optional
        Derivative order, zero, one or two [-].

    Returns
    -------
    basis : ndarray
        Coordinates by ordered basis coefficients, with flux derivatives [-].

    Limitations
    -----------
    Extrapolated basis values are mathematical evaluations only. Positive
    partition-of-unity controls guarantee positivity only on the declared domain.

    References
    ----------
    .. [1] C. de Boor, A Practical Guide to Splines, revised edition (2001),
       Chapters IX-X, B-spline representation and differentiation.
    """
    x = np.asarray(psi_n, dtype=float)
    if not np.isfinite(x).all() or derivative not in (0, 1, 2):
        raise ValueError("finite coordinates and derivative 0, 1 or 2 required")
    u = x / config.domain_max
    if config.basis == "bspline6":
        knots = [0.] * 4 + list(config.knot_fractions) + [1.] * 4
        b = BSpline(knots, np.eye(6), 3, extrapolate=True)
        return b(u, nu=derivative) / config.domain_max**derivative
    n = config.coefficient_count - 1
    # Polynomial coefficients give exact derivatives including both endpoints.
    from numpy.polynomial import Polynomial
    cols = []
    for j in range(n + 1):
        p = comb(n, j) * Polynomial([0., 1.])**j * Polynomial([1., -1.])**(n-j)
        cols.append(p.deriv(derivative)(u) / config.domain_max**derivative)
    return np.stack(cols, axis=-1)


def evaluate_regularized_profile(psi_n, parameters, config, *, derivative=0,
                                 allow_extrapolation=False):
    r"""Evaluate a positive regular flux profile or its flux derivative.

    $$F(x)=s\sum_j B_j(x/d)\,[f+\log(1+\exp(\theta_j))]$$

    Parameter order and basis [1]_ are shared with the offline fit. Finite
    F' implies dF(rho_pol squared)/drho_pol=0 on axis without fixing F'(0).

    Parameters
    ----------
    psi_n : array_like
        Finite normalized poloidal flux coordinates [-].
    parameters : array_like
        Finite unconstrained softplus parameters in basis order [-].
    config : RegularizedProfileConfig
        Basis, floor and domain declaration, with physical scale s [-].
    derivative : int, optional
        Flux derivative order, zero, one or two [-].
    allow_extrapolation : bool, optional
        Enable diagnostic evaluations outside the positivity domain [-].

    Returns
    -------
    values : ndarray
        Profile or derivative in the physical unit y of config.scale [y].

    Limitations
    -----------
    Outside the declared domain the default is NaN. Explicit extrapolation
    carries no positivity guarantee. Numerical overflow is not clipped.

    References
    ----------
    .. [1] C. de Boor, A Practical Guide to Splines, revised edition (2001),
       Chapters IX-X, B-spline representation and differentiation.
    """
    x = np.asarray(psi_n, float)
    theta = np.asarray(parameters, float)
    if theta.shape != (config.coefficient_count,) or not np.isfinite(theta).all():
        raise ValueError("finite parameter vector of the declared size required")
    out = config.scale * (profile_basis(x, config, derivative) @
                          (config.floor + np.logaddexp(0., theta)))
    if not allow_extrapolation:
        out = np.where((x >= 0) & (x <= config.domain_max), out, np.nan)
    return out


@dataclass
class RegularizedProfileFit:
    """A conditional fitted profile, never a measured uncertainty estimate."""
    config: RegularizedProfileConfig
    parameters: np.ndarray
    status: str
    reason: str | None
    diagnostics: dict
    version: str = API_VERSION

    def evaluate(self, psi_n, **kwargs):
        if self.status != "available":
            raise ValueError("cannot evaluate a failed fit")
        return evaluate_regularized_profile(psi_n, self.parameters, self.config, **kwargs)

    def to_dict(self):
        return {"version": self.version, "config": asdict(self.config),
                "parameters": self.parameters.tolist(), "status": self.status,
                "reason": self.reason, "diagnostics": self.diagnostics}


def fit_regularized_profile(psi_n, values, *, config=None):
    r"""Fit independent samples with Huber residuals and quadratic curvature.

    $$J=\frac{1}{N}\sum_i H_\delta((F(x_i)-y_i)/s)
    +\frac{\lambda}{2}\int_0^1 [d^2(F/s)/du^2]^2\,du$$

    The Huber term [1]_ is quadratic within delta and linear outside it; the
    curvature term remains separately quadratic in u=psi_N/domain_max.
    Reuses VAFT's legacy free-polynomial fit for initialization and SciPy's
    least-squares kernel. No boundary anchor or post-fit clipping is applied.

    Parameters
    ----------
    psi_n : array_like
        One-dimensional finite samples within the declared flux domain [-].
    values : array_like
        Independent quantity samples in the physical unit of config.scale [y].
    config : RegularizedProfileConfig or None, optional
        Serialized numerical objective, basis and convergence assumptions [-].

    Returns
    -------
    fit : RegularizedProfileFit
        Parameters, status and diagnostics; completion is not accuracy proof [-].

    Limitations
    -----------
    Sigma weighting is not implemented. Scale and robust weights are numerical
    assumptions, never diagnostic sigma. Invalid inputs are rejected, not
    dropped. Callers own independent masks, branch grouping and support.

    References
    ----------
    .. [1] P. J. Huber, Robust Estimation of a Location Parameter,
       Annals of Mathematical Statistics 35 (1964), 73-101.
    """
    cfg = config or RegularizedProfileConfig()
    x, y = np.asarray(psi_n, float), np.asarray(values, float)
    if (x.ndim != 1 or y.shape != x.shape or not len(x) or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(x < 0) or np.any(x > cfg.domain_max)):
        raise ValueError("finite 1-D samples in the declared domain required")
    with np.errstate(over="raise", invalid="raise"):
        target = y / cfg.scale
    if not np.isfinite(target).all() or np.max(np.abs(target)) > 1e12:
        raise ValueError("numerical scale unsuitable for sample magnitude")
    b = profile_basis(x, cfg)
    rank = int(np.linalg.matrix_rank(b))
    condition = float(np.linalg.cond(b))
    diag = {"n_samples": len(x), "n_distinct": len(np.unique(x)),
            "data_rank": rank, "data_condition": condition,
            "sigma": None, "uncertainty": "unknown; numerical scale is not sigma",
            "observed_span": [float(x.min()), float(x.max())],
            "center_observed": bool(np.any(x == 0)),
            "max_observation_gap": float(np.diff(np.unique(x)).max()) if len(np.unique(x)) > 1 else None}
    theta = np.zeros(cfg.coefficient_count)
    def fail(reason):
        return RegularizedProfileFit(cfg, theta, "failed", reason, diag)
    if len(np.unique(x)) <= cfg.coefficient_count or rank < cfg.coefficient_count:
        return fail("insufficient distinct observations / data rank")
    if not np.isfinite(condition) or condition > cfg.condition_limit:
        return fail("ill-conditioned data design")
    # Reuse the existing unanchored polynomial fitter without its zero uncertainty.
    _, _, initial, _ = fit_profile(x, target, None, x, order=2,
                                  uncertainty_option=0, fitting_function="free_polynomial")
    c0 = np.maximum(initial(np.linspace(x.min(), x.max(), cfg.coefficient_count)),
                    max(cfg.floor * 10, 1e-6))
    theta = c0 + np.log(-np.expm1(-c0))  # stable inverse softplus
    nodes, weights = np.polynomial.legendre.leggauss(32)
    cuts = [0., *cfg.knot_fractions, 1.] if cfg.basis == "bspline6" else [0., 1.]
    qx, qw = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        qx.extend((lo + (nodes + 1) * (hi-lo)/2) * cfg.domain_max)
        qw.extend(weights * (hi-lo)/2)
    curvature = profile_basis(qx, cfg, 2) * cfg.domain_max**2
    penalty = np.sqrt(cfg.regularization * np.asarray(qw))[:, None] * curvature
    root_n = np.sqrt(len(x))
    def components(t):
        c = cfg.floor + np.logaddexp(0., t)
        r = b @ c - target
        a = np.abs(r)
        outside = a > cfg.huber_delta
        h = r.copy()
        h[outside] = np.sign(r[outside]) * np.sqrt(
            2*cfg.huber_delta*a[outside] - cfg.huber_delta**2)
        factor = np.ones(len(x))
        factor[outside] = cfg.huber_delta / np.abs(h[outside])
        return c, r, h, factor
    def residual(t):
        c, _, h, _ = components(t)
        return np.r_[h/root_n, penalty @ c]
    def jacobian(t):
        _, _, _, factor = components(t)
        return np.vstack((factor[:, None]*b/root_n, penalty)) * expit(t)
    try:
        opt = least_squares(residual, theta, jac=jacobian, max_nfev=cfg.max_nfev,
                            ftol=cfg.tolerance, xtol=cfg.tolerance,
                            gtol=cfg.tolerance, method="trf", loss="linear")
        theta = opt.x
        c, r, h, _ = components(theta)
        robust = np.minimum(1., cfg.huber_delta / np.maximum(np.abs(r), np.finfo(float).tiny))
        diag.update({"residual": (r*cfg.scale).tolist(),
                     "normalized_residual": r.tolist(), "robust_weights": robust.tolist(),
                     "downweighted_count": int(np.sum(robust < 1)),
                     "residual_rms": float(np.sqrt(np.mean(r*r))*cfg.scale),
                     "data_huber_mean": float(np.mean(h*h)/2),
                     "curvature_penalty": float(np.sum((penalty@c)**2)/2),
                     "weighted_data_condition": float(np.linalg.cond(np.sqrt(robust)[:, None]*b)),
                     "optimizer_status": int(opt.status), "message": opt.message,
                     "nfev": opt.nfev, "optimality": float(opt.optimality),
                     "converged": bool(opt.success), "coefficients_scaled": c.tolist()})
        if not opt.success or not np.isfinite(opt.fun).all():
            return fail("optimizer did not converge")
        if np.linalg.matrix_rank(opt.jac[:len(x)]) < cfg.coefficient_count:
            return fail("saturated / rank-deficient fitted data Jacobian")
        return RegularizedProfileFit(cfg, theta, "available", None, diag)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return fail(str(exc))
