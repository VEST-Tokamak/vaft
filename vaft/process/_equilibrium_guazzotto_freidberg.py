"""Guazzotto-Freidberg analytic equilibria, Parts 1 (#1148) and 2 (#1149).

L. Guazzotto and J. P. Freidberg, *Simple, general, realistic, robust,
analytic tokamak equilibria. Part 1. Limiter and divertor tokamaks*,
J. Plasma Phys. 87, 905870303 (2021), doi:10.1017/S002237782100009X, and
*Part 2. Pedestals and flow*, J. Plasma Phys. 87, 905870305 (2021),
doi:10.1017/S0022377821000118.  Equation numbers refer to Part 1 unless
marked "Part 2".  :mod:`vaft.process.equilibrium` is the public import location.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.constants import mu_0 as MU0
from scipy.optimize import brentq, minimize_scalar, root

from vaft.data.equilibrium import Contour, EquilibriumData, GuazzottoFreidbergEquilibrium

GF_TOPOLOGIES = ("limited", "double_null", "lower_single_null")

#: Basis of Eqs. (3.2) and (5.8): (separation index n, radial function, vertical function).
#: ``S_1`` vanishes identically because ``k_1 = 0``, and ``sin(h_4 y) = 0`` because
#: ``h_4 = 0``, which is how 16 candidate terms become 7 or 12.
_SYMMETRIC_BASIS = ((1, "C", "cos"), (2, "C", "cos"), (2, "S", "cos"), (3, "C", "cos"),
                    (3, "S", "cos"), (4, "C", "cos"), (4, "S", "cos"))
_ASYMMETRIC_BASIS = _SYMMETRIC_BASIS + ((1, "C", "sin"), (2, "C", "sin"), (2, "S", "sin"),
                                        (3, "C", "sin"), (3, "S", "sin"))

_KINDS = ("psi", "x", "y", "xx", "yy", "xy")


def _separation_constants(eps: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (3.3): h_n and k_n, with k_n^2 = alpha^2 - h_n^2/(1 + eps^2)."""
    scale = np.sqrt(1.0 + eps**2)*alpha
    h = np.array([scale, np.sqrt(35/36)*scale, np.sqrt(13/49)*scale, 0.0])
    k = np.array([0.0, alpha/6.0, 6.0*alpha/7.0, alpha])
    return h, k


def _flow_factor_exponent(gamma: float | None) -> float:
    """``gamma/(gamma - 1)``, the exponent of the flow factor in Eqs. (5.6) and (6.1)."""
    return 1.0 if gamma is None or np.isinf(gamma) else gamma/(gamma - 1)


def _source_polynomial(eps: float, nu: float, mach: float = 0.0, gamma: float | None = None) -> tuple[float, float, float]:
    """``(s1, s2, s3)`` with ``nu G(x) = s1 x + s2 x**2 + s3 x**3``, Eqs. (5.6)-(5.8).

    Without flow ``G = eps_hat x`` (Part 1).  With flow, ``gamma = inf`` and
    ``gamma = 2`` make ``G`` a quadratic and a cubic; ``s_j = eps_hat**j nu_j``
    so that ``lambda_j**2 = alpha**2 s_j`` of Eq. (5.10).
    """
    eps_hat = 2*eps/(1 + eps**2)
    if mach == 0:
        return (eps_hat*nu, 0.0, 0.0)
    zeta = mach**2*(1 + eps**2)/(1 + mach**2*eps**2)
    if gamma is None or np.isinf(gamma):
        nus = ((1 + zeta)*nu, zeta*nu, 0.0)
    else:
        nus = ((1 + 2*zeta)*nu, zeta*(2 + zeta)*nu, zeta**2*nu)
    return tuple(float(eps_hat**(j + 1)*v) for j, v in enumerate(nus))


def _series(k: float, lam: tuple[float, float, float], eps_hat: float, sine: bool, terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Eqs. (A2)-(A3) of Part 1, generalized to (B3) of Part 2: power-series coefficients of X_n.

    ``lam`` holds ``(lambda_1**2, lambda_2**2, lambda_3**2)`` of Eq. (5.10);
    Part 1 is ``(alpha**2 eps_hat nu, 0, 0)``.
    """
    l1, l2, l3 = lam
    a = np.zeros(terms); b = np.zeros(terms)
    if sine:
        b[0] = 1.0
    else:
        a[0] = 1.0
    for m in range(3, terms):
        am = eps_hat*(m-1)*(m-2)*a[m-1] + (l1 - eps_hat*k*k)*a[m-3] + 2*(m-1)*k*b[m-1] + 2*eps_hat*(m-2)*k*b[m-2]
        bm = eps_hat*(m-1)*(m-2)*b[m-1] + (l1 - eps_hat*k*k)*b[m-3] - 2*(m-1)*k*a[m-1] - 2*eps_hat*(m-2)*k*a[m-2]
        if m >= 4:
            am += l2*a[m-4]; bm += l2*b[m-4]
        if m >= 5:
            am += l3*a[m-5]; bm += l3*b[m-5]
        a[m] = -am/(m*(m-1)); b[m] = -bm/(m*(m-1))
    return a, b


def _radial(a: np.ndarray, b: np.ndarray, k: float, lam: tuple[float, float, float], eps_hat: float,
            x: np.ndarray) -> tuple[np.ndarray, ...]:
    """X, X', X'' by Eq. (A4); X'' from the ODE (5.10) itself."""
    m = np.arange(a.size)
    xm = x[..., None]**m
    dxm = np.where(m > 0, m*x[..., None]**np.maximum(m - 1, 0), 0.0)
    cos, sin = np.cos(k*x)[..., None], np.sin(k*x)[..., None]
    X = np.sum((a*cos + b*sin)*xm, axis=-1)
    Xp = np.sum((-k*a*xm + b*dxm)*sin + (k*b*xm + a*dxm)*cos, axis=-1)
    Xpp = -(k*k + lam[0]*x + lam[1]*x**2 + lam[2]*x**3)/(1.0 + eps_hat*x)*X
    return X, Xp, Xpp


def _basis_values(topology: str, eps: float, source: tuple[float, float, float], alpha: float, x: Any, y: Any,
                  terms: int) -> list[dict[str, np.ndarray]]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    eps_hat = 2*eps/(1 + eps**2); lam = tuple(alpha**2*s for s in source)
    h, k = _separation_constants(eps, alpha)
    radial: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}
    out = []
    for n, kind, vertical in (_ASYMMETRIC_BASIS if topology == "lower_single_null" else _SYMMETRIC_BASIS):
        if (n, kind) not in radial:
            a, b = _series(k[n-1], lam, eps_hat, kind == "S", terms)
            radial[(n, kind)] = _radial(a, b, k[n-1], lam, eps_hat, x)
        X, Xp, Xpp = radial[(n, kind)]
        hy = h[n-1]*y
        if vertical == "cos":
            Y, Yp, Ypp = np.cos(hy), -h[n-1]*np.sin(hy), -h[n-1]**2*np.cos(hy)
        else:
            Y, Yp, Ypp = np.sin(hy), h[n-1]*np.cos(hy), -h[n-1]**2*np.sin(hy)
        out.append({"psi": Y*X, "x": Y*Xp, "y": Yp*X, "xx": Y*Xpp, "yy": Ypp*X, "xy": Yp*Xp})
    return out


def _x_point_geometry(eps: float, kappa_x: float, delta_x: float) -> tuple[float, float, float, float]:
    """Eqs. (4.3), (4.5), (4.7): xi, x_X and the two-ellipse midplane curvatures."""
    root_term = np.sqrt(1 - delta_x**2)
    xi = root_term/(kappa_x - root_term)
    x_x = delta_x + eps/2*(1 - delta_x**2)
    lam1 = (1 - eps)*(1 - delta_x)*(1 + xi)/kappa_x**2
    lam2 = (1 + eps)*(1 + delta_x)*(1 + xi)/kappa_x**2
    return xi, x_x, lam1, lam2


def _constraints(topology: str, eps: float, kappa: float | None, delta: float | None,
                 kappa_x: float | None, delta_x: float | None) -> list[tuple[tuple[float, float], tuple[tuple[str, float], ...]]]:
    """The seven (Eqs. 3.5, 3.8 or 4.6) or twelve (Eqs. 5.3-5.6) matching conditions."""
    rows: list[tuple[tuple[float, float], tuple[tuple[str, float], ...]]] = []
    inner, outer = (-1.0, 0.0), (1.0, 0.0)
    if topology in ("limited", "lower_single_null"):
        dh = np.arcsin(delta)
        x_d = delta + eps/2*(1 - delta**2)
        top = (-x_d, kappa)
        l1 = (1 - eps)*(1 - dh)**2/kappa**2
        l2 = (1 + eps)*(1 + dh)**2/kappa**2
        l3 = kappa/((1 - eps*delta)**2*(1 - delta**2))
    if topology in ("double_null", "lower_single_null"):
        _, x_x, m1, m2 = _x_point_geometry(eps, kappa_x, delta_x)
    if topology == "limited":
        rows = [(inner, (("psi", 1.0),)), (outer, (("psi", 1.0),)), (top, (("psi", 1.0),)), (top, (("x", 1.0),)),
                (inner, (("yy", 1.0), ("x", l1))), (outer, (("yy", 1.0), ("x", -l2))), (top, (("xx", 1.0), ("y", -l3)))]
    elif topology == "double_null":
        xp = (-x_x, kappa_x)
        rows = [(inner, (("psi", 1.0),)), (outer, (("psi", 1.0),)), (xp, (("psi", 1.0),)), (xp, (("x", 1.0),)),
                (inner, (("yy", 1.0), ("x", m1))), (outer, (("yy", 1.0), ("x", -m2))), (xp, (("y", 1.0),))]
    else:
        # Eq. (5.4 h, i) prints the X-point as (-delta_X, -kappa_X); the X-point
        # of (5.3 d) is (-x_X, -kappa_X), and only that reproduces Table 4.
        xp = (-x_x, -kappa_x)
        a1, a2 = 0.5*(l1 + m1), 0.5*(l2 + m2)
        rows = [(inner, (("psi", 1.0),)), (outer, (("psi", 1.0),)), (top, (("psi", 1.0),)), (xp, (("psi", 1.0),)),
                (inner, (("y", 1.0),)), (outer, (("y", 1.0),)), (top, (("x", 1.0),)), (xp, (("x", 1.0),)), (xp, (("y", 1.0),)),
                (inner, (("yy", 1.0), ("x", a1))), (outer, (("yy", 1.0), ("x", -a2))), (top, (("xx", 1.0), ("y", -l3)))]
    return rows


def _matrix(topology: str, eps: float, source, alpha: float, rows, terms: int) -> np.ndarray:
    points = np.array([point for point, _ in rows])
    columns = _basis_values(topology, eps, source, alpha, points[:, 0], points[:, 1], terms)
    return np.array([[sum(weight*column[kind][i] for kind, weight in combo) for column in columns]
                     for i, (_, combo) in enumerate(rows)])


def _eigen_error(topology: str, eps: float, source, alpha: float, rows, terms: int) -> tuple[float, np.ndarray, float]:
    """Eq. (3.11): c_1 = 1, the first N-1 conditions solved, the last one's normalized miss."""
    A = _matrix(topology, eps, source, alpha, rows, terms)
    n = A.shape[0]
    sub = A[:n-1, 1:]
    rest = np.linalg.solve(sub, -A[:n-1, 0])
    terms_last = np.r_[A[n-1, 0], A[n-1, 1:]*rest]
    denominator = np.sum(np.abs(terms_last))
    error = float((np.sum(terms_last)/denominator)**2) if denominator > 0 else 1.0
    return error, np.r_[1.0, rest], float(np.linalg.cond(sub))


def solve_guazzotto_freidberg(
    topology: str = "limited", *, inverse_aspect_ratio: float, nu: float,
    elongation: float | None = None, triangularity: float | None = None,
    x_point_elongation: float | None = None, x_point_triangularity: float | None = None,
    alpha_range: tuple[float, float] = (0.5, 6.0), series_terms: int = 250,
    current_pedestal: float = 0.0, pressure_pedestal: float = 0.0, bootstrap_fraction: float = 0.0,
    mach_number: float = 0.0, adiabatic_index: float | None = None, allow_current_reversal: bool = False,
) -> GuazzottoFreidbergEquilibrium:
    """Solve the Guazzotto-Freidberg analytic equilibrium for a shape and a beta parameter.

    An exact solution of the Grad-Shafranov equation whose pressure and
    ``F**2`` are quadratic in the flux, so that -- unlike Solov'ev -- the
    pressure, its gradient and the toroidal current all vanish at the plasma
    surface.  The shape is imposed through the flux, slope and curvature at
    the midplane and top points (or an X-point), and the eigenvalue ``alpha``
    that makes a non-trivial solution exist is found by scanning.

    Parameters
    ----------
    topology : str, optional
        ``"limited"`` (smooth, up-down symmetric, Miller model surface),
        ``"double_null"`` or ``"lower_single_null"`` [-].
    inverse_aspect_ratio : float
        ``eps = a/R0``, in (0, 1) [-].
    nu : float
        The source-balance parameter ``mu0 p0/(mu0 p0 + B0 dB/(1 + eps**2))``,
        approximately the poloidal beta; 0 is force free [-].
    elongation : float, optional
        Elongation of the smooth (upper) boundary; required for limited and
        single-null [-].
    triangularity : float, optional
        Triangularity of the smooth (upper) boundary, magnitude below one [-].
    x_point_elongation : float, optional
        ``kappa_X``, height of the X-point over the minor radius; required for
        diverted topologies [-].
    x_point_triangularity : float, optional
        ``delta_X``, triangularity at the X-point, magnitude below one [-].
    alpha_range : tuple of float, optional
        Interval scanned for the lowest eigenvalue [-].
    series_terms : int, optional
        Terms kept in the radial power series [-].
    current_pedestal : float, optional
        ``f_J``, the edge-to-axis ratio of the toroidal current density
        (Part 2, Eq. 4.5), in [0, 1) [-].
    pressure_pedestal : float, optional
        ``f_P = p(surf)/p(axis)`` (Part 2, Eq. 3.1), in [0, 1); it leaves the
        flux unchanged and enters the plasma parameters [-].
    bootstrap_fraction : float, optional
        ``f_B = 1 - I/I_hat``, the edge-localized bootstrap current carried
        as a surface current (Part 2, Eq. 3.5), in [0, 1); plasma
        parameters only [-].
    mach_number : float, optional
        ``M0``, the toroidal flow Mach number on the axis (Part 2, Eq. 5.6) [-].
    adiabatic_index : float or None, optional
        ``gamma``, 2 (adiabatic) or ``inf`` (incompressible); required with
        flow, the two cases whose source is a polynomial (Part 2, Eq. 5.8) [-].
    allow_current_reversal : bool, optional
        Accept ``nu`` above ``nu_max``, where the inboard edge current
        reverses, and mark the result ``status = "current_reversal"`` [-].

    Returns
    -------
    GuazzottoFreidbergEquilibrium
        The eigenvalue, the coefficients normalized so psi is one on the
        magnetic axis, the separation constants, the axis in (x, y), the
        eigen-condition residual and conditioning, and the inputs [-].

    Raises
    ------
    ValueError
        An unknown topology or missing shape input; invalid geometry,
        including ``kappa_X <= 2 sqrt(1 - delta_X**2)`` (Eq. 4.4); ``nu``
        beyond ``nu_max`` (``1/eps_hat``, or ``-1/G(-1)`` with flow), where
        the inboard current reverses, unless *allow_current_reversal*; a
        pedestal or bootstrap fraction outside [0, 1); flow without
        ``adiabatic_index`` 2 or inf; or no eigenvalue in *alpha_range*, or
        no pedestal root below it (no physical root).

    Processing steps
    ----------------
    1. Fix the separation constants by Eq. (3.3) and build the radial
       functions by the power series of Appendix A.
    2. For a trial ``alpha``, set ``c_1 = 1`` and solve all but the last
       matching condition; the last one's normalized miss is Eq. (3.11).
    3. Scan *alpha_range* for the lowest minimum of that error, refine it,
       and accept it only if the error is at round-off level.
    4. Locate the magnetic axis as the maximum of psi and normalize psi to
       one there.
    5. With a current pedestal (Part 2, Eqs. 4.8-4.11), solve the
       inhomogeneous system ``psi_J = f_J/(1 - f_J)`` on the surface
       instead, and find the ``alpha`` below the pedestal-free eigenvalue at
       which ``psi_J(axis) = 1/(1 - f_J)``.

    Defaults
    --------
    ``series_terms = 250`` follows the paper, which keeps 150 routinely and
    needs about 220 at ``eps = 0.75``; ``alpha_range`` brackets every case
    the paper publishes (1.88 to 2.38).  Both are numerical conveniences.

    Convention
    ----------
    Normalized coordinates of Eq. (2.8): ``R = R0 sqrt(1 + eps**2 + 2 eps x)``,
    ``Z = a y``, ``psi = Psi/Psi_0`` with ``psi = 1`` on the axis and ``0`` on
    the surface; ``eps_hat = 2 eps/(1 + eps**2)``.  With flow the source is
    ``alpha**2 (1 + nu G(x))``, ``G = (1 + eps_hat x)(1 + 2 M0**2 eps x/(1 +
    M0**2 eps**2))**(gamma/(gamma - 1)) - 1`` (Part 2, Eq. 5.6).  The lower single null
    places its X-point at ``(-x_X, -kappa_X)`` with ``x_X = delta_X + (eps/2)
    (1 - delta_X**2)``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The paper's Eq. (5.4 h, i) evaluate the single-null X-point conditions at
    ``(-delta_X, -kappa_X)``; this uses the X-point itself, ``(-x_X,
    -kappa_X)``, which is the reading that reproduces the published
    eigenvalues.  Only the lowest eigenvalue is sought; the model surface is
    matched at three or four points only.  With a current pedestal the
    midplane curvature conditions keep their pedestal-free values, so the
    X-point angle is no longer the model's right angle (Part 2, Section 4.3).
    Part 2's Table 2 cases C and D keep ``nu = 1.04`` above the flow-reduced
    ``nu_max = 1.025``; they need *allow_current_reversal*.

    Provenance
    ----------
    .. [1] L. Guazzotto and J. P. Freidberg, J. Plasma Phys. 87, 905870303
       (2021), doi:10.1017/S002237782100009X: Eqs. (2.8)-(2.9), (3.2)-(3.11),
       (4.3)-(4.7), (5.3)-(5.8) and Appendix A.
    .. [2] L. Guazzotto and J. P. Freidberg, J. Plasma Phys. 87, 905870305
       (2021), doi:10.1017/S0022377821000118: Eqs. (3.1)-(3.5), (4.1)-(4.11),
       (5.6)-(5.14) and Appendix B.
    """
    if topology not in GF_TOPOLOGIES:
        raise ValueError(f"topology must be one of {GF_TOPOLOGIES}, got {topology!r}")
    eps = float(inverse_aspect_ratio)
    if not 0 < eps < 1:
        raise ValueError("invalid geometry: inverse_aspect_ratio must lie in (0, 1)")
    if nu < 0:
        raise ValueError("nu must be non-negative")
    f_j, f_p, f_b = float(current_pedestal), float(pressure_pedestal), float(bootstrap_fraction)
    for name, value in (("current_pedestal", f_j), ("pressure_pedestal", f_p), ("bootstrap_fraction", f_b)):
        if not 0 <= value < 1:
            raise ValueError(f"{name} must lie in [0, 1), got {value}")
    mach = float(mach_number)
    if mach < 0:
        raise ValueError("mach_number must be non-negative")
    if mach > 0 and not (adiabatic_index is not None and (np.isinf(adiabatic_index) or adiabatic_index == 2)):
        raise ValueError("toroidal flow needs adiabatic_index 2 (adiabatic) or inf (incompressible), "
                         "the two cases with a closed-form G(x) (Eq. 5.8)")
    gamma = None if mach == 0 else float(adiabatic_index)
    nu_max = _nu_max(eps, mach, gamma)
    reversed_current = nu > nu_max
    if reversed_current and not allow_current_reversal:
        bound = "-1/G(-1)" if mach > 0 else "1/eps_hat"
        raise ValueError(f"nonphysical current reversal: nu={nu} exceeds nu_max = {bound} = {nu_max:.4g} "
                         f"({'Part 2, Eq. 5.7' if mach > 0 else 'Eq. 2.10'})")
    if topology in ("limited", "lower_single_null"):
        if elongation is None or triangularity is None:
            raise ValueError(f"{topology} needs elongation and triangularity")
        if elongation <= 0 or abs(triangularity) >= 1:
            raise ValueError("invalid geometry: elongation must be positive and abs(triangularity) below one")
    if topology in ("double_null", "lower_single_null"):
        if x_point_elongation is None or x_point_triangularity is None:
            raise ValueError(f"{topology} needs x_point_elongation and x_point_triangularity")
        if abs(x_point_triangularity) >= 1 or x_point_elongation <= 2*np.sqrt(1 - x_point_triangularity**2):
            raise ValueError("invalid geometry: the two-ellipse X-point model needs "
                             "kappa_X > 2 sqrt(1 - delta_X**2) and abs(delta_X) < 1 (Eq. 4.4)")
    rows = _constraints(topology, eps, elongation, triangularity, x_point_elongation, x_point_triangularity)
    terms = int(series_terms)
    source = _source_polynomial(eps, nu, mach, gamma)
    grid = np.linspace(*alpha_range, 1101)
    errors = np.array([_eigen_error(topology, eps, source, a, rows, terms)[0] for a in grid])
    alpha = None
    for i in range(1, grid.size - 1):
        if errors[i] <= errors[i-1] and errors[i] <= errors[i+1] and errors[i] < 1e-3:
            refined = minimize_scalar(lambda a: _eigen_error(topology, eps, source, a, rows, terms)[0],
                                      bracket=(grid[i-1], grid[i], grid[i+1]), tol=1e-12)
            if refined.fun < 1e-14:
                alpha = float(refined.x)
                break
    if alpha is None:
        raise ValueError(f"no physical root: no eigenvalue alpha in {alpha_range} satisfies the matching conditions")
    if f_j == 0:
        residual, coefficients, condition = _eigen_error(topology, eps, source, alpha, rows, terms)
        axis, at_axis = _find_axis(topology, eps, source, alpha, coefficients, terms)
        coefficients = coefficients/float(at_axis["psi"])
    else:
        # Eqs. (4.8)-(4.11): psi_J = f_J/(1 - f_J) on the surface makes the
        # system inhomogeneous, and alpha is fixed by psi_J(axis) = 1/(1 - f_J)
        # instead.  psi_J(axis) rises from about f_J/(1 - f_J) at small alpha
        # to infinity at the pedestal-free eigenvalue, so the root lies below it.
        target = 1/(1 - f_j)

        def mismatch(a: float) -> float:
            try:
                u = _pedestal_coefficients(topology, eps, source, a, rows, terms, f_j/(1 - f_j))
                _, values = _find_axis(topology, eps, source, a, u, terms)
            except (ValueError, np.linalg.LinAlgError):
                return float("nan")
            return float(values["psi"]) - target

        low = alpha_range[0]
        # psi_J(axis) -> +inf as alpha -> alpha0 from below, so the last
        # bracket ends just under alpha0; the gap to the root shrinks with f_J,
        # and a geometric tail keeps a small pedestal's root inside a bracket.
        tail = alpha - (alpha - low)*np.geomspace(1/120, 1e-10, 25)
        trial = np.unique(np.r_[np.linspace(low, alpha, 121)[:-1], tail])
        signs = np.array([mismatch(a) for a in trial])
        crossing = [i for i in range(trial.size - 1)
                    if np.isfinite(signs[i]) and np.isfinite(signs[i+1]) and signs[i] < 0 <= signs[i+1]]
        if not crossing:
            raise ValueError(f"no physical root: psi_J(axis) = 1/(1 - f_J) is not reached for alpha in "
                             f"[{low}, {alpha:.4g}) (Eq. 4.11)")
        i = crossing[0]
        alpha = float(brentq(mismatch, trial[i], trial[i+1], xtol=1e-13))
        coefficients = _pedestal_coefficients(topology, eps, source, alpha, rows, terms, f_j/(1 - f_j))
        axis, at_axis = _find_axis(topology, eps, source, alpha, coefficients, terms)
        residual = float(((float(at_axis["psi"]) - target)/(abs(float(at_axis["psi"])) + target))**2)
        condition = float(np.linalg.cond(_matrix(topology, eps, source, alpha, rows, terms)))
    h, k = _separation_constants(eps, alpha)
    return GuazzottoFreidbergEquilibrium(
        topology=topology, inverse_aspect_ratio=eps, nu=float(nu), alpha=alpha, coefficients=coefficients,
        separation_h=h, separation_k=k, magnetic_axis=axis, elongation=elongation, triangularity=triangularity,
        x_point_elongation=x_point_elongation, x_point_triangularity=x_point_triangularity,
        eigen_residual=float(residual), condition_number=condition, series_terms=terms,
        metadata={"reference": "Guazzotto & Freidberg, J. Plasma Phys. 87, 905870303 (2021)"
                               + ("" if (f_j, f_p, f_b, mach) == (0, 0, 0, 0) else
                                  "; Part 2, J. Plasma Phys. 87, 905870305 (2021)")},
        current_pedestal=f_j, pressure_pedestal=f_p, bootstrap_fraction=f_b, mach_number=mach,
        adiabatic_index=gamma, status="current_reversal" if reversed_current else "converged",
    )


def _nu_max(eps: float, mach: float, gamma: float | None) -> float:
    """Eq. (5.7): ``nu_max = -1/G(-1)``; ``1/eps_hat`` without flow (Part 1, Eq. 2.10)."""
    eps_hat = 2*eps/(1 + eps**2)
    c = 2*mach**2*eps/(1 + mach**2*eps**2)
    g_inner = (1 - eps_hat)*(1 - c)**_flow_factor_exponent(gamma) - 1
    return float(-1/g_inner)


def _pedestal_coefficients(topology, eps, source, alpha, rows, terms, offset) -> np.ndarray:
    """Eq. (4.10): the inhomogeneous matching system, ``psi_J = offset`` at the surface points."""
    A = _matrix(topology, eps, source, alpha, rows, terms)
    v = np.array([offset if combo == (("psi", 1.0),) else 0.0 for _, combo in rows])
    return np.linalg.solve(A, v)


def _find_axis(topology, eps, source, alpha, coefficients, terms):
    """The magnetic axis, an extremum of psi near the geometric centre, and the fields there."""
    def gradient(point: np.ndarray) -> list[float]:
        values = _evaluate(topology, eps, source, alpha, coefficients, point[0], point[1], terms)
        return [float(values["x"]), float(values["y"])]

    solved = root(gradient, [0.0, 0.0])
    axis = (float(solved.x[0]), float(solved.x[1]))
    at_axis = _evaluate(topology, eps, source, alpha, coefficients, *axis, terms)
    if not solved.success or abs(axis[0]) > 1 or float(at_axis["xx"]*at_axis["yy"] - at_axis["xy"]**2) <= 0:
        raise ValueError("no magnetic axis (extremum of psi) was found near the geometric centre")
    return axis, at_axis


def _model_source(model: GuazzottoFreidbergEquilibrium) -> tuple[float, float, float]:
    return _source_polynomial(model.inverse_aspect_ratio, model.nu, model.mach_number, model.adiabatic_index)


def _flux_offset(model: GuazzottoFreidbergEquilibrium) -> float:
    """``f_J/(1 - f_J)``: ``psi = psi_J - offset`` (Eq. 4.7); zero without a current pedestal."""
    return model.current_pedestal/(1 - model.current_pedestal)


def _psi_on_grid(model: GuazzottoFreidbergEquilibrium, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """psi on the tensor grid ``xs`` x ``ys``, indexed (x, y), using separability.

    Each basis term is X_n(x) Y_n(y), so the radial series is evaluated once
    per x value instead of once per grid point.
    """
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    eps, alpha = model.inverse_aspect_ratio, model.alpha
    eps_hat = 2*eps/(1 + eps**2); lam = tuple(alpha**2*s for s in _model_source(model))
    h, k = model.separation_h, model.separation_k
    basis = _ASYMMETRIC_BASIS if model.topology == "lower_single_null" else _SYMMETRIC_BASIS
    psi = np.zeros((xs.size, ys.size))
    radial: dict[tuple[int, str], np.ndarray] = {}
    for c, (n, kind, vertical) in zip(model.coefficients, basis):
        if (n, kind) not in radial:
            a, b = _series(k[n-1], lam, eps_hat, kind == "S", model.series_terms)
            radial[(n, kind)] = _radial(a, b, k[n-1], lam, eps_hat, xs)[0]
        Y = np.cos(h[n-1]*ys) if vertical == "cos" else np.sin(h[n-1]*ys)
        psi += c*np.outer(radial[(n, kind)], Y)
    return psi - _flux_offset(model)


def _evaluate(topology, eps, source, alpha, coefficients, x, y, terms) -> dict[str, np.ndarray]:
    """psi_J (psi without a current pedestal) and its derivatives."""
    columns = _basis_values(topology, eps, source, alpha, x, y, terms)
    return {kind: sum(c*column[kind] for c, column in zip(coefficients, columns)) for kind in _KINDS}


def evaluate_guazzotto_freidberg(model: GuazzottoFreidbergEquilibrium, x: Any, y: Any) -> Mapping[str, np.ndarray]:
    """Normalized flux and its analytic derivatives of a Guazzotto-Freidberg equilibrium.

    Parameters
    ----------
    model : GuazzottoFreidbergEquilibrium
        A solved equilibrium [-].
    x : array_like
        Normalized radial coordinate, ``-1`` inboard and ``1`` outboard on the
        midplane [-].
    y : array_like
        Normalized height ``Z/a`` [-].

    Returns
    -------
    Mapping of str to np.ndarray
        ``psi`` and its derivatives ``psi_x``, ``psi_y``, ``psi_xx``,
        ``psi_yy``, ``psi_xy``; ``psi_j``, the shifted flux ``psi + f_J/(1 -
        f_J)`` of Part 2 (equal to psi without a current pedestal); and
        ``gs_residual``, the Grad-Shafranov equation evaluated as ``(1 +
        eps_hat x) psi_xx + psi_yy/(1 + eps**2) + alpha**2 (1 + nu G(x))
        psi_J``, zero to round-off (Eq. 2.9; Part 2, Eq. 5.14) [-].

    Convention
    ----------
    The normalized coordinates and flux of Eq. (2.8); physical fields follow
    from :func:`guazzotto_freidberg_to_equilibrium`.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Guazzotto and Freidberg (2021), Eqs. (2.9), (3.2) and (A4); Part 2,
       Eqs. (5.12)-(5.14).
    """
    values = _evaluate(model.topology, model.inverse_aspect_ratio, _model_source(model), model.alpha,
                       model.coefficients, x, y, model.series_terms)
    eps = model.inverse_aspect_ratio; eps_hat = 2*eps/(1 + eps**2)
    xx = np.asarray(x, dtype=float)
    s1, s2, s3 = _model_source(model)
    residual = ((1 + eps_hat*xx)*values["xx"] + values["yy"]/(1 + eps**2)
                + model.alpha**2*(1 + s1*xx + s2*xx**2 + s3*xx**3)*values["psi"])
    return {"psi": values["psi"] - _flux_offset(model), "psi_j": values["psi"], "psi_x": values["x"], "psi_y": values["y"], "psi_xx": values["xx"],
            "psi_yy": values["yy"], "psi_xy": values["xy"], "gs_residual": residual}


def _axis_curvature(model: GuazzottoFreidbergEquilibrium) -> tuple[float, float]:
    """``r`` and ``psi_xx psi_yy`` on the magnetic axis, the pieces of Eq. (6.12)."""
    eps = model.inverse_aspect_ratio
    at_axis = evaluate_guazzotto_freidberg(model, *model.magnetic_axis)
    return 1 + eps**2 + 2*eps*model.magnetic_axis[0], float(at_axis["psi_xx"]*at_axis["psi_yy"])


def _pedestal_factors(model: GuazzottoFreidbergEquilibrium) -> tuple[float, float]:
    """``(1 + f_J)/(1 - f_J)`` and ``(1 - f_P)(1 + M0**2 eps**2)**(gamma/(gamma - 1))`` of Part 2, Eq. (6.1)."""
    eps = model.inverse_aspect_ratio
    current = (1 + model.current_pedestal)/(1 - model.current_pedestal)
    pressure = (1 - model.pressure_pedestal)*(1 + model.mach_number**2*eps**2)**_flow_factor_exponent(model.adiabatic_index)
    return current, pressure


def _beta_ratio_from_q0(model: GuazzottoFreidbergEquilibrium, q0: float) -> float:
    """Part 1 Eq. (6.12) / Part 2 Eq. (6.1), inverted, as ``beta0/nu``, which stays finite as nu -> 0."""
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    current, pressure = _pedestal_factors(model)
    r_axis, curvature = _axis_curvature(model)
    denominator = q0**2*r_axis**2*curvature - current*(1 + eps**2)*(1 - nu)*eps**2*alpha**2
    if denominator <= 0:
        raise ValueError(f"q0={q0} is not reachable for this equilibrium: Eq. (6.12) gives a non-positive beta0")
    return float(current*eps**2*alpha**2/(pressure*denominator))


def _q0_from_beta_ratio(model: GuazzottoFreidbergEquilibrium, ratio: float) -> float:
    """The same relation forward: q0 from ``beta0/nu``."""
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    current, pressure = _pedestal_factors(model)
    r_axis, curvature = _axis_curvature(model)
    q0_squared = (current*eps**2*alpha**2/(pressure*ratio) + current*(1 + eps**2)*(1 - nu)*eps**2*alpha**2)
    return float(np.sqrt(q0_squared/(r_axis**2*curvature)))


def _plasma_region(model: GuazzottoFreidbergEquilibrium, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """A fine (x, y) grid, psi on it, and the closed psi = 0 surface around the axis."""
    import matplotlib.path as mpath
    from contourpy import contour_generator

    height = max(v for v in (model.elongation, model.x_point_elongation) if v is not None)*1.03
    x_edge = min(1.03, 0.999*(1 + model.inverse_aspect_ratio**2)/(2*model.inverse_aspect_ratio))
    xs = np.linspace(-x_edge, x_edge, int(resolution))
    ys = np.linspace(-height, height, int(resolution*height))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    psi = _psi_on_grid(model, xs, ys)
    lines = contour_generator(xs, ys, psi.T).lines(1e-6)
    closing = [line for line in lines if mpath.Path(line).contains_point(model.magnetic_axis)]
    if not closing:
        raise ValueError("the psi = 0 surface does not close around the magnetic axis on the evaluation grid")
    surface = max(closing, key=len)
    inside = mpath.Path(surface).contains_points(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    return xs, ys, psi, inside, surface


def guazzotto_freidberg_parameters(
    model: GuazzottoFreidbergEquilibrium, *, q0: float | None = 1.0, beta0: float | None = None,
    resolution: int = 500,
) -> Mapping[str, float]:
    """The dimensionless plasma parameters of Section 6 for a Guazzotto-Freidberg equilibrium.

    The flux surfaces fix ``alpha``; one more number -- the on-axis toroidal
    beta, or equivalently the safety factor on axis -- fixes the pressure and
    the diamagnetism, and with them every global quantity.

    Parameters
    ----------
    model : GuazzottoFreidbergEquilibrium
        A solved equilibrium [-].
    q0 : float or None, optional
        Safety factor on the magnetic axis, converted to ``beta0`` by
        Eq. (6.12); when *beta0* is given, the q0 it implies is reported instead [-].
    beta0 : float or None, optional
        Toroidal beta on axis, ``2 mu0 p0/B0**2`` [-].
    resolution : int, optional
        Grid points across ``-1 <= x <= 1`` for the plasma integrals [-].

    Returns
    -------
    Mapping of str to float
        ``alpha``, ``beta0``, ``q0``, ``delta_b_over_b0`` (Eq. 6.4),
        ``beta_t`` (6.6), ``beta_p`` (6.7), ``li`` (6.9), ``q_star`` (6.10)
        with the elongation it used as ``kappa_q_star``, and ``q95`` from
        Eq. (6.11) on the ``psi = 0.05`` surface; and from Part 2,
        ``beta0_hat`` (6.4), ``surface_field_ratio`` ``Delta_B = B_hat0/B0``
        (6.3), ``total_current_ratio`` ``I_hat/I`` (6.9), and the core current
        ``mu0 I/(a B0)`` as ``core_current`` (area integral of J_phi) and
        ``core_current_line_integral`` (closed integral of b_P on the
        surface), which agree by Ampere's law [-].

    Raises
    ------
    ValueError
        Neither *q0* nor *beta0*, a *q0* that Eq. (6.12) cannot reach, or a
        surface that does not close on the grid.

    Processing steps
    ----------------
    1. Take *beta0*, or obtain it from *q0* by Eq. (6.12) (Part 2, Eq. 6.1).
    2. Integrate ``psi_J``, ``psi_J**2`` and their ``(1 + nu G)/(1 + eps_hat x)``
       and ``(1 + G)/(1 + eps_hat x)`` weighted forms over the plasma in
       ``(x, y)``, where the volume element is uniform.
    3. Iterate the exterior field ratio ``Delta_B`` until the surface current
       carries the bootstrap fraction (Part 2, Eqs. 3.9 and 6.3).
    4. Evaluate Eqs. (6.4)-(6.10) (Part 2, 6.4-6.11), and q95 as the line
       integral ``F/(2 pi) closed_integral dl/(R**2 B_p)`` on the ``psi = 0.05``
       surface.

    Defaults
    --------
    ``q0 = 1`` is the paper's choice for Table 4, a conventional value.
    ``resolution = 500`` is a numerical convenience giving about three
    significant figures, the precision the paper publishes.

    Convention
    ----------
    ``beta_p`` is the paper's volume-averaged form, Eq. (6.7), not the IMAS
    DD definition.  For a diverted equilibrium ``q_star`` uses ``kappa95``,
    the height of the ``psi = 0.05`` surface over the full plasma width
    ``2a`` -- the reading that reproduces the published values -- and for a
    limited one the input elongation.  With ``nu = 0`` (force free) ``beta0``
    is zero and only *q0* can fix the normalization.  ``beta_t`` is
    normalized to the exterior vacuum field ``B_hat0`` (Part 2, Eq. 6.7
    prints the ratio ``B_hat0**2/B0**2`` inverted), and ``li`` weights the
    current with ``1 + nu G`` where Part 2, Eq. (6.10) prints ``1 + nu
    eps_hat x``; the two readings coincide without flow.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Integrals are on a Cartesian grid masked by the surface, so they
    carry an error of order the cell size.  Part 2's Table 2 is reproduced
    to 0.5 % in ``beta_p`` and 1.1 % in ``q_star``.  The surface-current
    jump uses Eq. (3.9), whose toroidal-field term carries ``R0**2/R**2``;
    Eq. (6.2) prints it without, a 0.2 % difference in q*.

    Provenance
    ----------
    .. [1] Guazzotto and Freidberg (2021), Section 6, Eqs. (6.3)-(6.12).
    .. [2] Guazzotto and Freidberg (2021), Part 2, doi:10.1017/S0022377821000118,
       Eqs. (3.6)-(3.10) and Section 6, Eqs. (6.1)-(6.12).
    """
    parameters, _ = _parameters_and_current(model, q0=q0, beta0=beta0, resolution=resolution)
    return parameters


def _surface_field_ratio(model, surface, psi_a: float, surface_pressure) -> tuple[float, float]:
    """Part 2 Eqs. (3.9) and (6.3): the exterior field ratio Delta_B that gives the bootstrap fraction f_B.

    Pressure balance across the surface current gives ``b_hat_P**2 = b_P**2
    + 2 mu0 p_surf/B0**2 + (1 - Delta_B**2) R0**2/R**2``.  Eq. (6.2) prints
    the last term without ``R0**2/R**2``; the toroidal field jump
    ``B_phi**2 - B_hat_phi**2 = (1 - Delta_B**2) B0**2 R0**2/R**2`` of
    Eqs. (3.6)-(3.9) carries it, and the two change q* by 0.2 % on Table 2.
    Returns ``(Delta_B, closed_integral b_P dl)``, the latter in units of ``B0 a``.
    """
    eps = model.inverse_aspect_ratio
    xc, yc = surface[:, 0], surface[:, 1]
    if not np.allclose(surface[0], surface[-1]):
        xc, yc = np.r_[xc, xc[0]], np.r_[yc, yc[0]]
    xm, ym = 0.5*(xc[1:] + xc[:-1]), 0.5*(yc[1:] + yc[:-1])
    r = 1 + eps**2 + 2*eps*xm
    dl = np.hypot(np.diff(np.sqrt(1 + eps**2 + 2*eps*xc)), eps*np.diff(yc))/eps          # units of a
    field = evaluate_guazzotto_freidberg(model, xm, ym)
    b_p2 = psi_a**2*(field["psi_x"]**2 + field["psi_y"]**2/r)
    inside = float(np.sum(np.sqrt(b_p2)*dl))
    geometric = 1/r
    base = b_p2 + surface_pressure(xm)
    f_b, f_p = model.bootstrap_fraction, model.pressure_pedestal
    if f_b == 0 and f_p == 0:
        return 1.0, inside
    target = inside/(1 - f_b)

    def excess(delta: float) -> float:
        return float(np.sum(np.sqrt(np.maximum(base + (1 - delta**2)*geometric, 0.0))*dl)) - target

    upper = float(np.sqrt(1 + np.min(base/geometric)))
    if excess(upper) > 0 or excess(1e-9) < 0:
        raise ValueError("no exterior field ratio Delta_B reproduces the requested bootstrap fraction (Eq. 6.3)")
    return float(brentq(excess, 1e-9, upper, xtol=1e-12)), inside


def _parameters_and_current(model, *, q0, beta0, resolution):
    """Section 6 parameters (Part 1 and Part 2) and ``mu0 I/(a B0)``, the core current."""
    import matplotlib.path as mpath
    from contourpy import contour_generator

    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    if beta0 is None:
        if q0 is None:
            raise ValueError("give q0 or beta0")
        ratio = _beta_ratio_from_q0(model, float(q0))      # beta0/nu
        q0_value = float(q0)
    else:
        if nu == 0:
            raise ValueError("a force-free equilibrium (nu = 0) has beta0 = 0; give q0 instead")
        ratio = float(beta0)/nu
        q0_value = _q0_from_beta_ratio(model, ratio)
    beta_axis = nu*ratio
    f_j, f_p, f_b = model.current_pedestal, model.pressure_pedestal, model.bootstrap_fraction
    current, pressure = _pedestal_factors(model)
    exponent = _flow_factor_exponent(model.adiabatic_index)
    flow_c = 2*model.mach_number**2*eps/(1 + model.mach_number**2*eps**2)
    psi_a = float(np.sqrt(pressure/current*ratio)/alpha)                 # Psi_0/(a R0 B0), Eq. (6.1)
    eps_hat = 2*eps/(1 + eps**2)
    s1, s2, s3 = _model_source(model)
    xs, ys, psi, inside, surface = _plasma_region(model, resolution)
    psi_j = psi + _flux_offset(model)
    X = xs[:, None]*np.ones((1, ys.size))
    cell = (xs[1] - xs[0])*(ys[1] - ys[0])
    source_weight = (1 + s1*X + s2*X**2 + s3*X**3)/(1 + eps_hat*X)     # (1 + nu G)/(1 + eps_hat x)
    flow_weight = (1 + flow_c*X)**exponent                                # (1 + G)/(1 + eps_hat x)
    edge = (f_p - f_j**2)/(1 - f_j)**2
    area = np.sum(inside)*cell
    i_source = np.sum(source_weight*psi_j*inside)*cell
    i_source2 = np.sum(source_weight*psi_j**2*inside)*cell
    i_pressure = np.sum(flow_weight*((1 - f_p)*psi_j**2 + edge)*inside)*cell
    delta_b = ratio*(1 + eps**2)*pressure*(1 - nu)/2                   # Eq. (6.5) with beta0 = nu*ratio
    i_core = psi_a*alpha**2*i_source                                     # mu0 I/(a B0), Ampere on Eq. (6.1)

    def surface_pressure(x):
        return beta_axis*f_p*(1 + model.mach_number**2*eps**2 + 2*model.mach_number**2*eps*x)**exponent

    delta_field, line_current = _surface_field_ratio(model, surface, psi_a, surface_pressure)
    current_ratio = 1/(1 - f_b)                                          # I_hat/I, Eq. (6.9)
    lines = contour_generator(xs, ys, psi.T).lines(0.05)
    closing = [line for line in lines if mpath.Path(line).contains_point(model.magnetic_axis)]
    if not closing:
        raise ValueError("the psi = 0.05 surface does not close around the magnetic axis on the evaluation grid")
    surface95 = max(closing, key=len)
    xc, yc = surface95[:, 0], surface95[:, 1]
    if not np.allclose(surface95[0], surface95[-1]):
        xc, yc = np.r_[xc, xc[0]], np.r_[yc, yc[0]]
    R = np.sqrt(1 + eps**2 + 2*eps*xc); Z = eps*yc                       # in units of R0
    # The height of the 95 % surface over the full plasma width 2a: the
    # published divertor q* values are reproduced with this, to 0.3 %.
    kappa95 = float(np.ptp(Z)/(2*eps))
    kappa_q = model.elongation if model.topology == "limited" else kappa95
    q_star = np.pi*eps*(1 + kappa_q**2)*delta_field/(current_ratio*i_core)   # Eq. (6.11)
    psi0 = eps*psi_a                                                     # in units of B0 R0**2
    field = evaluate_guazzotto_freidberg(model, 0.5*(xc[1:] + xc[:-1]), 0.5*(yc[1:] + yc[:-1]))
    r_mid = 0.5*(R[1:] + R[:-1])
    b_pol = psi0/eps*np.sqrt(field["psi_x"]**2 + field["psi_y"]**2/r_mid**2)
    a2 = delta_b*2*f_j/(1 + f_j)                                         # A_2 of Eqs. (4.3), (4.5)
    f95 = np.sqrt(1 + 2*delta_b*0.05**2 + 2*a2*(0.05 - 0.05**2))
    q95 = f95/(2*np.pi)*np.sum(np.hypot(np.diff(R), np.diff(Z))/(r_mid**2*b_pol))
    x_axis = model.magnetic_axis[0]
    parameters = {
        "alpha": alpha, "beta0": float(beta_axis), "q0": q0_value, "delta_b_over_b0": float(delta_b),
        "beta0_hat": float(beta_axis/delta_field**2*(1 + model.mach_number**2*eps**2
                                                     + 2*model.mach_number**2*eps*x_axis)**exponent),
        "beta_t": float(beta_axis*pressure/(1 - f_p)*np.sum(flow_weight*((1 - f_p)*psi_j**2/current
                        + (f_p - f_j**2)/(1 - f_j**2))*inside)*cell/area/delta_field**2),
        "beta_p": float(nu*i_pressure/((1 - f_p)*i_source2)),
        "li": float(4*np.pi/alpha**2/current_ratio**2*i_source2/i_source**2),
        "q_star": float(q_star), "kappa_q_star": float(kappa_q), "q95": float(q95),
        "surface_field_ratio": float(delta_field), "total_current_ratio": float(current_ratio),
        "core_current_line_integral": float(line_current), "core_current": float(i_core),
    }
    return parameters, float(i_core)


def guazzotto_freidberg_to_equilibrium(
    model: GuazzottoFreidbergEquilibrium, *, major_radius: float, toroidal_field: float,
    q0: float | None = 1.0, beta0: float | None = None, resolution: int = 129, convention: int = 11,
) -> EquilibriumData:
    """Export a Guazzotto-Freidberg equilibrium as a gridded record in physical units.

    Parameters
    ----------
    model : GuazzottoFreidbergEquilibrium
        A solved equilibrium [-].
    major_radius : float
        ``R0``, positive [m].
    toroidal_field : float
        Vacuum field ``B0`` at ``R0``, positive [T].
    q0 : float or None, optional
        Safety factor on axis, fixing ``beta0`` by Eq. (6.12) [-].
    beta0 : float or None, optional
        Toroidal beta on axis, taking precedence over *q0* [-].
    resolution : int, optional
        Grid points across the major radius [-].
    convention : int, optional
        COCOS index to export in, 1 to 18 [-].

    Returns
    -------
    EquilibriumData
        psi, axis and boundary flux, LCFS, pressure, F, p', FF' (psi = 0 on
        the boundary), the core plasma current and the vacuum field inside
        the surface, with ``metadata`` carrying the model and the Section 6
        parameters [-].

    Raises
    ------
    ValueError
        Invalid physical inputs or COCOS index, an unreachable *q0*, a
        boundary that does not close on the grid, or toroidal flow, whose
        pressure is not a flux function.

    Processing steps
    ----------------
    1. Fix ``beta0`` and from it ``p0`` (6.3), ``dB/B0`` (6.4) and the axis
       flux ``Psi_0`` (6.5).
    2. Evaluate ``psi`` on an R-Z grid through ``x = (R**2/R0**2 - 1 - eps**2)/(2 eps)``.
    3. Build ``p = p0 (f_P + (1 - f_P) psi**2 + A1 (psi - psi**2))`` and
       ``F**2 = R0**2 B0**2 (1 + 2 (dB/B0) psi**2 + 2 A2 (psi - psi**2))``
       (Part 2, Eqs. 4.1, 4.3, 4.5; ``A1 = A2 = 0`` without pedestals),
       their flux derivatives, and ``I_p`` from Ampere's law on J_phi.
    4. Store in COCOS 11 and convert when another convention is asked for.

    Defaults
    --------
    ``q0 = 1`` as in the paper's Table 4, a conventional value;
    ``resolution = 129`` is a numerical convenience.

    Convention
    ----------
    Built in COCOS 11 for a positive current: the stored flux is
    ``-2 pi Psi_0 psi``, a minimum on the axis rising to zero at the
    boundary, as :func:`solovev_example` stores it.  ``p'`` and ``FF'`` are per
    weber of that stored flux.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``q`` is not filled; compute it with :func:`calculate_q_profile_from_psi`
    or take q95 from :func:`guazzotto_freidberg_parameters`.  The surface
    currents of Part 2 are not on the grid: ``ip`` is the core current ``I``
    and ``bt0`` the interior ``B0``; ``I_hat/I`` and ``B_hat0/B0`` are in
    ``metadata["parameters"]``.

    Provenance
    ----------
    .. [1] Guazzotto and Freidberg (2021), Eqs. (2.2), (2.8), (6.3)-(6.8);
       Part 2, Eqs. (4.1)-(4.6) and (6.5)-(6.6).
    """
    from vaft.process._equilibrium_parametric import _closed_boundary_contour, _detect_convention, convert_cocos

    if major_radius <= 0 or toroidal_field <= 0:
        raise ValueError("major_radius and toroidal_field must be positive")
    if convention not in range(1, 19) or convention in (9, 10):
        raise ValueError("convention must be a COCOS index in the range 1..18 (excluding 9 and 10)")
    if model.mach_number > 0:
        raise ValueError("an equilibrium with toroidal flow has a pressure that is not a flux function "
                         "(Part 2, Eq. 6.1); the gridded record cannot hold it")
    parameters, i_core = _parameters_and_current(model, q0=q0, beta0=beta0, resolution=400)
    beta_axis = parameters["beta0"]
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    r0, b0 = float(major_radius), float(toroidal_field)
    minor = eps*r0
    p0 = beta_axis*b0**2/(2*MU0)
    delta_b = parameters["delta_b_over_b0"]
    ratio = beta_axis/nu if nu > 0 else _beta_ratio_from_q0(model, parameters["q0"])
    current, pressure_factor = _pedestal_factors(model)
    psi0 = eps*b0*r0**2/alpha*np.sqrt(ratio*pressure_factor/current)   # Wb/rad, Part 1 (6.5), Part 2 (6.6)
    height = max(v for v in (model.elongation, model.x_point_elongation) if v is not None)*minor
    # The radial series converges only for |x| < 1/eps_hat (R = 0 is a
    # singular point), so the grid stops short of that.
    x_edge = min(1.15, 0.999*(1 + eps**2)/(2*eps))
    r_min = r0*np.sqrt(max(1 + eps**2 - 2*eps*x_edge, 1e-6)); r_max = r0*np.sqrt(1 + eps**2 + 2*eps*x_edge)
    r = np.linspace(r_min, r_max, int(resolution))
    z = np.linspace(-1.2*height, 1.2*height, int(np.ceil(resolution*2.4*height/(r_max - r_min))) | 1)
    psi_n = _psi_on_grid(model, (r**2/r0**2 - 1 - eps**2)/(2*eps), z/minor)
    axis_x, axis_y = model.magnetic_axis
    axis = (r0*np.sqrt(1 + eps**2 + 2*eps*axis_x), minor*axis_y)
    stored = -2*np.pi*psi0
    temp = EquilibriumData(r=r, z=z, psi=psi_n, psi_axis=1.0, psi_boundary=0.0, magnetic_axis=axis)
    levels = (0.9999, 0.9995, 0.999) if model.topology != "limited" else (1.0, 0.9995, 0.999, 0.995)
    lcfs, lcfs_level = _closed_boundary_contour(temp, axis, levels)
    if lcfs is None:
        raise ValueError("the boundary does not close around the magnetic axis on this grid; raise resolution")
    psi_1d_n = np.linspace(1.0, 0.0, max(65, min(r.size, z.size)))
    # Part 2, Eqs. (4.1), (4.3), (4.5): Solov'ev-like terms linear in psi
    # carry the pressure and current pedestals; A1 = A2 = 0 without them.
    f_p, f_j = model.pressure_pedestal, model.current_pedestal
    a1 = 2*(1 - f_p)*f_j/(1 + f_j)
    a2 = delta_b*2*f_j/(1 + f_j)
    x = psi_1d_n
    pressure = p0*(f_p + (1 - f_p)*x**2 + a1*(x - x**2))
    f = r0*b0*np.sqrt(1 + 2*delta_b*x**2 + 2*a2*(x - x**2))
    pprime = p0*(2*(1 - f_p)*x + a1*(1 - 2*x))/stored
    ffprime = (r0*b0)**2*(2*delta_b*x + a2*(1 - 2*x))/stored
    ip = float(eps*b0*r0*i_core/MU0)                                     # core current, Ampere's law
    conv_11 = _detect_convention(explicit=11, bt0=b0, ip=ip, q=None, psi_1d=stored*psi_1d_n,
                                 source="analytic Guazzotto-Freidberg")
    eq_11 = EquilibriumData(
        r=r, z=z, psi=stored*psi_n, psi_axis=stored, psi_boundary=0.0, magnetic_axis=axis, lcfs=lcfs,
        psi_1d=stored*psi_1d_n, pressure=pressure, f=f, q=None, pprime=pprime, ffprime=ffprime,
        ip=ip, bt0=b0, r0=r0, time=None, convention=conv_11,
        metadata={"source_type": "guazzotto_freidberg", "model": model, "parameters": dict(parameters),
                  "p0": p0, "psi0_per_radian": psi0, "lcfs_psi_n": lcfs_level},
    )
    if convention == 11:
        return eq_11
    return convert_cocos(eq_11, convention)


__all__ = [
    "GF_TOPOLOGIES", "evaluate_guazzotto_freidberg", "guazzotto_freidberg_parameters",
    "guazzotto_freidberg_to_equilibrium", "solve_guazzotto_freidberg",
]
