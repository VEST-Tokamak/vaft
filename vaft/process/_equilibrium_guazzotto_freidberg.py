"""Guazzotto-Freidberg analytic equilibria, Part 1 (#1148).

L. Guazzotto and J. P. Freidberg, *Simple, general, realistic, robust,
analytic tokamak equilibria. Part 1. Limiter and divertor tokamaks*,
J. Plasma Phys. 87, 905870303 (2021), doi:10.1017/S002237782100009X.
Equation numbers below refer to that paper.  :mod:`vaft.process.equilibrium`
is the public import location.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.constants import mu_0 as MU0
from scipy.optimize import minimize_scalar, root

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


def _series(k: float, lam2: float, eps_hat: float, sine: bool, terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Eqs. (A2)-(A3): power-series coefficients of the cosine- or sine-like X_n."""
    a = np.zeros(terms); b = np.zeros(terms)
    if sine:
        b[0] = 1.0
    else:
        a[0] = 1.0
    for m in range(3, terms):
        am = eps_hat*(m-1)*(m-2)*a[m-1] + (lam2 - eps_hat*k*k)*a[m-3] + 2*(m-1)*k*b[m-1] + 2*eps_hat*(m-2)*k*b[m-2]
        bm = eps_hat*(m-1)*(m-2)*b[m-1] + (lam2 - eps_hat*k*k)*b[m-3] - 2*(m-1)*k*a[m-1] - 2*eps_hat*(m-2)*k*a[m-2]
        a[m] = -am/(m*(m-1)); b[m] = -bm/(m*(m-1))
    return a, b


def _radial(a: np.ndarray, b: np.ndarray, k: float, lam2: float, eps_hat: float, x: np.ndarray) -> tuple[np.ndarray, ...]:
    """X, X', X'' by Eq. (A4); X'' from the ODE (A1) itself."""
    m = np.arange(a.size)
    xm = x[..., None]**m
    dxm = np.where(m > 0, m*x[..., None]**np.maximum(m - 1, 0), 0.0)
    cos, sin = np.cos(k*x)[..., None], np.sin(k*x)[..., None]
    X = np.sum((a*cos + b*sin)*xm, axis=-1)
    Xp = np.sum((-k*a*xm + b*dxm)*sin + (k*b*xm + a*dxm)*cos, axis=-1)
    Xpp = -(k*k + lam2*x)/(1.0 + eps_hat*x)*X
    return X, Xp, Xpp


def _basis_values(topology: str, eps: float, nu: float, alpha: float, x: Any, y: Any, terms: int) -> list[dict[str, np.ndarray]]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    eps_hat = 2*eps/(1 + eps**2); lam2 = alpha**2*eps_hat*nu
    h, k = _separation_constants(eps, alpha)
    radial: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}
    out = []
    for n, kind, vertical in (_ASYMMETRIC_BASIS if topology == "lower_single_null" else _SYMMETRIC_BASIS):
        if (n, kind) not in radial:
            a, b = _series(k[n-1], lam2, eps_hat, kind == "S", terms)
            radial[(n, kind)] = _radial(a, b, k[n-1], lam2, eps_hat, x)
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


def _matrix(topology: str, eps: float, nu: float, alpha: float, rows, terms: int) -> np.ndarray:
    points = np.array([point for point, _ in rows])
    columns = _basis_values(topology, eps, nu, alpha, points[:, 0], points[:, 1], terms)
    return np.array([[sum(weight*column[kind][i] for kind, weight in combo) for column in columns]
                     for i, (_, combo) in enumerate(rows)])


def _eigen_error(topology: str, eps: float, nu: float, alpha: float, rows, terms: int) -> tuple[float, np.ndarray, float]:
    """Eq. (3.11): c_1 = 1, the first N-1 conditions solved, the last one's normalized miss."""
    A = _matrix(topology, eps, nu, alpha, rows, terms)
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
        beyond ``1/eps_hat``, where the inboard current reverses; or no
        eigenvalue in *alpha_range* (no physical root).

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

    Defaults
    --------
    ``series_terms = 250`` follows the paper, which keeps 150 routinely and
    needs about 220 at ``eps = 0.75``; ``alpha_range`` brackets every case
    the paper publishes (1.88 to 2.38).  Both are numerical conveniences.

    Convention
    ----------
    Normalized coordinates of Eq. (2.8): ``R = R0 sqrt(1 + eps**2 + 2 eps x)``,
    ``Z = a y``, ``psi = Psi/Psi_0`` with ``psi = 1`` on the axis and ``0`` on
    the surface; ``eps_hat = 2 eps/(1 + eps**2)``.  The lower single null
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
    matched at three or four points only.

    Provenance
    ----------
    .. [1] L. Guazzotto and J. P. Freidberg, J. Plasma Phys. 87, 905870303
       (2021), doi:10.1017/S002237782100009X: Eqs. (2.8)-(2.9), (3.2)-(3.11),
       (4.3)-(4.7), (5.3)-(5.8) and Appendix A.
    """
    if topology not in GF_TOPOLOGIES:
        raise ValueError(f"topology must be one of {GF_TOPOLOGIES}, got {topology!r}")
    eps = float(inverse_aspect_ratio)
    if not 0 < eps < 1:
        raise ValueError("invalid geometry: inverse_aspect_ratio must lie in (0, 1)")
    if nu < 0:
        raise ValueError("nu must be non-negative")
    eps_hat = 2*eps/(1 + eps**2)
    if nu > 1/eps_hat:
        raise ValueError(f"nonphysical current reversal: nu={nu} exceeds nu_max = 1/eps_hat = {1/eps_hat:.4g} (Eq. 2.10)")
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
    grid = np.linspace(*alpha_range, 1101)
    errors = np.array([_eigen_error(topology, eps, nu, a, rows, terms)[0] for a in grid])
    alpha = None
    for i in range(1, grid.size - 1):
        if errors[i] <= errors[i-1] and errors[i] <= errors[i+1] and errors[i] < 1e-3:
            refined = minimize_scalar(lambda a: _eigen_error(topology, eps, nu, a, rows, terms)[0],
                                      bracket=(grid[i-1], grid[i], grid[i+1]), tol=1e-12)
            if refined.fun < 1e-14:
                alpha = float(refined.x)
                break
    if alpha is None:
        raise ValueError(f"no physical root: no eigenvalue alpha in {alpha_range} satisfies the matching conditions")
    residual, coefficients, condition = _eigen_error(topology, eps, nu, alpha, rows, terms)
    h, k = _separation_constants(eps, alpha)

    def gradient(point: np.ndarray) -> list[float]:
        values = _evaluate(topology, eps, nu, alpha, coefficients, point[0], point[1], terms)
        return [float(values["x"]), float(values["y"])]

    solved = root(gradient, [0.0, 0.0])
    axis = (float(solved.x[0]), float(solved.x[1]))
    at_axis = _evaluate(topology, eps, nu, alpha, coefficients, *axis, terms)
    if not solved.success or float(at_axis["xx"]*at_axis["yy"] - at_axis["xy"]**2) <= 0:
        raise ValueError("no magnetic axis (extremum of psi) was found near the geometric centre")
    coefficients = coefficients/float(at_axis["psi"])
    return GuazzottoFreidbergEquilibrium(
        topology=topology, inverse_aspect_ratio=eps, nu=float(nu), alpha=alpha, coefficients=coefficients,
        separation_h=h, separation_k=k, magnetic_axis=axis, elongation=elongation, triangularity=triangularity,
        x_point_elongation=x_point_elongation, x_point_triangularity=x_point_triangularity,
        eigen_residual=float(residual), condition_number=condition, series_terms=terms,
        metadata={"reference": "Guazzotto & Freidberg, J. Plasma Phys. 87, 905870303 (2021)"},
    )


def _psi_on_grid(model: GuazzottoFreidbergEquilibrium, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """psi on the tensor grid ``xs`` x ``ys``, indexed (x, y), using separability.

    Each basis term is X_n(x) Y_n(y), so the radial series is evaluated once
    per x value instead of once per grid point.
    """
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    eps_hat = 2*eps/(1 + eps**2); lam2 = alpha**2*eps_hat*nu
    h, k = model.separation_h, model.separation_k
    basis = _ASYMMETRIC_BASIS if model.topology == "lower_single_null" else _SYMMETRIC_BASIS
    psi = np.zeros((xs.size, ys.size))
    radial: dict[tuple[int, str], np.ndarray] = {}
    for c, (n, kind, vertical) in zip(model.coefficients, basis):
        if (n, kind) not in radial:
            a, b = _series(k[n-1], lam2, eps_hat, kind == "S", model.series_terms)
            radial[(n, kind)] = _radial(a, b, k[n-1], lam2, eps_hat, xs)[0]
        Y = np.cos(h[n-1]*ys) if vertical == "cos" else np.sin(h[n-1]*ys)
        psi += c*np.outer(radial[(n, kind)], Y)
    return psi


def _evaluate(topology, eps, nu, alpha, coefficients, x, y, terms) -> dict[str, np.ndarray]:
    columns = _basis_values(topology, eps, nu, alpha, x, y, terms)
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
        ``psi_yy``, ``psi_xy``, and ``gs_residual``, the Grad-Shafranov
        equation (2.9) evaluated as ``(1 + eps_hat x) psi_xx + psi_yy/(1 +
        eps**2) + alpha**2 (1 + eps_hat nu x) psi``, zero to round-off [-].

    Convention
    ----------
    The normalized coordinates and flux of Eq. (2.8); physical fields follow
    from :func:`guazzotto_freidberg_to_equilibrium`.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Guazzotto and Freidberg (2021), Eqs. (2.9), (3.2) and (A4).
    """
    values = _evaluate(model.topology, model.inverse_aspect_ratio, model.nu, model.alpha, model.coefficients,
                       x, y, model.series_terms)
    eps = model.inverse_aspect_ratio; eps_hat = 2*eps/(1 + eps**2)
    xx = np.asarray(x, dtype=float)
    residual = ((1 + eps_hat*xx)*values["xx"] + values["yy"]/(1 + eps**2)
                + model.alpha**2*(1 + eps_hat*model.nu*xx)*values["psi"])
    return {"psi": values["psi"], "psi_x": values["x"], "psi_y": values["y"], "psi_xx": values["xx"],
            "psi_yy": values["yy"], "psi_xy": values["xy"], "gs_residual": residual}


def _axis_curvature(model: GuazzottoFreidbergEquilibrium) -> tuple[float, float]:
    """``r`` and ``psi_xx psi_yy`` on the magnetic axis, the pieces of Eq. (6.12)."""
    eps = model.inverse_aspect_ratio
    at_axis = evaluate_guazzotto_freidberg(model, *model.magnetic_axis)
    return 1 + eps**2 + 2*eps*model.magnetic_axis[0], float(at_axis["psi_xx"]*at_axis["psi_yy"])


def _beta_ratio_from_q0(model: GuazzottoFreidbergEquilibrium, q0: float) -> float:
    """Eq. (6.12), inverted, as ``beta0/nu``, which stays finite as nu -> 0."""
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    r_axis, curvature = _axis_curvature(model)
    denominator = q0**2*r_axis**2*curvature - (1 + eps**2)*(1 - nu)*eps**2*alpha**2
    if denominator <= 0:
        raise ValueError(f"q0={q0} is not reachable for this equilibrium: Eq. (6.12) gives a non-positive beta0")
    return float(eps**2*alpha**2/denominator)


def _q0_from_beta_ratio(model: GuazzottoFreidbergEquilibrium, ratio: float) -> float:
    """Eq. (6.12), forward: q0 = F/(R0 B0) (nu/beta0)**0.5 eps alpha/(r (psi_xx psi_yy)**0.5) on the axis."""
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    r_axis, curvature = _axis_curvature(model)
    delta_b = ratio*(1 + eps**2)*(1 - nu)/2
    return float(np.sqrt(1 + 2*delta_b)/np.sqrt(ratio)*eps*alpha/(r_axis*np.sqrt(curvature)))


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
        Eq. (6.11) on the ``psi = 0.05`` surface [-].

    Raises
    ------
    ValueError
        Neither *q0* nor *beta0*, a *q0* that Eq. (6.12) cannot reach, or a
        surface that does not close on the grid.

    Processing steps
    ----------------
    1. Take *beta0*, or obtain it from *q0* by Eq. (6.12).
    2. Integrate ``psi``, ``psi**2`` and their ``(1 + nu eps_hat x)/(1 + eps_hat x)``
       weighted forms over the plasma in ``(x, y)``, where the volume element
       is uniform.
    3. Evaluate Eqs. (6.4)-(6.10), and q95 as the line integral
       ``F/(2 pi) closed_integral dl/(R**2 B_p)`` on the ``psi = 0.05`` surface.

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
    is zero and only *q0* can fix the normalization.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Integrals are on a Cartesian grid masked by the surface, so they
    carry an error of order the cell size.

    Provenance
    ----------
    .. [1] Guazzotto and Freidberg (2021), Section 6, Eqs. (6.3)-(6.12).
    """
    parameters, _ = _parameters_and_current(model, q0=q0, beta0=beta0, resolution=resolution)
    return parameters


def _parameters_and_current(model, *, q0, beta0, resolution):
    """Section 6 parameters and the weighted current integral of Eq. (6.8)."""
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
    eps_hat = 2*eps/(1 + eps**2)
    xs, ys, psi, inside, _ = _plasma_region(model, resolution)
    X = xs[:, None]*np.ones((1, ys.size))
    cell = (xs[1] - xs[0])*(ys[1] - ys[0])
    weight = (1 + nu*eps_hat*X)/(1 + eps_hat*X)
    area = np.sum(inside)*cell
    i2 = np.sum(psi**2*inside)*cell
    iw = np.sum(weight*psi*inside)*cell
    iw2 = np.sum(weight*psi**2*inside)*cell
    delta_b = ratio*(1 + eps**2)*(1 - nu)/2                            # Eq. (6.4) with beta0 = nu*ratio
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
    q_star = np.pi*eps/alpha/np.sqrt(ratio)*(1 + kappa_q**2)/iw
    psi0 = eps/alpha*np.sqrt(ratio)                                      # in units of B0 R0**2
    field = evaluate_guazzotto_freidberg(model, 0.5*(xc[1:] + xc[:-1]), 0.5*(yc[1:] + yc[:-1]))
    r_mid = 0.5*(R[1:] + R[:-1])
    b_pol = psi0/eps*np.sqrt(field["psi_x"]**2 + field["psi_y"]**2/r_mid**2)
    f95 = np.sqrt(1 + 2*delta_b*0.05**2)
    q95 = f95/(2*np.pi)*np.sum(np.hypot(np.diff(R), np.diff(Z))/(r_mid**2*b_pol))
    parameters = {
        "alpha": alpha, "beta0": float(beta_axis), "q0": q0_value, "delta_b_over_b0": float(delta_b),
        "beta_t": float(beta_axis*i2/area), "beta_p": float(nu*i2/iw2), "li": float(4*np.pi/alpha**2*iw2/iw**2),
        "q_star": float(q_star), "kappa_q_star": float(kappa_q), "q95": float(q95),
    }
    return parameters, float(iw)


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
        the boundary), the plasma current of Eq. (6.8) and the vacuum field,
        with ``metadata`` carrying the model and the Section 6 parameters [-].

    Raises
    ------
    ValueError
        Invalid physical inputs or COCOS index, an unreachable *q0*, or a
        boundary that does not close on the grid.

    Processing steps
    ----------------
    1. Fix ``beta0`` and from it ``p0`` (6.3), ``dB/B0`` (6.4) and the axis
       flux ``Psi_0`` (6.5).
    2. Evaluate ``psi`` on an R-Z grid through ``x = (R**2/R0**2 - 1 - eps**2)/(2 eps)``.
    3. Build ``p = p0 psi**2`` and ``F**2 = R0**2 B0**2 (1 + 2 (dB/B0) psi**2)``,
       their flux derivatives, and ``I_p`` from Eq. (6.8).
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
    or take q95 from :func:`guazzotto_freidberg_parameters`.

    Provenance
    ----------
    .. [1] Guazzotto and Freidberg (2021), Eqs. (2.2), (2.8), (6.3)-(6.8).
    """
    from vaft.process._equilibrium_parametric import _closed_boundary_contour, _detect_convention, convert_cocos

    if major_radius <= 0 or toroidal_field <= 0:
        raise ValueError("major_radius and toroidal_field must be positive")
    if convention not in range(1, 19) or convention in (9, 10):
        raise ValueError("convention must be a COCOS index in the range 1..18 (excluding 9 and 10)")
    parameters, iw = _parameters_and_current(model, q0=q0, beta0=beta0, resolution=400)
    beta_axis = parameters["beta0"]
    eps, nu, alpha = model.inverse_aspect_ratio, model.nu, model.alpha
    r0, b0 = float(major_radius), float(toroidal_field)
    minor = eps*r0
    p0 = beta_axis*b0**2/(2*MU0)
    delta_b = parameters["delta_b_over_b0"]
    ratio = beta_axis/nu if nu > 0 else _beta_ratio_from_q0(model, parameters["q0"])
    psi0 = eps*b0*r0**2/alpha*np.sqrt(ratio)                             # Wb/rad, Eq. (6.5)
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
    pressure = p0*psi_1d_n**2
    f = r0*b0*np.sqrt(1 + 2*delta_b*psi_1d_n**2)
    pprime = 2*p0*psi_1d_n/stored
    ffprime = (r0*b0)**2*2*delta_b*psi_1d_n/stored
    ip = float(eps*b0*r0*alpha*np.sqrt(ratio)*iw/MU0)                    # Eq. (6.8)
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
