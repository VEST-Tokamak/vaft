"""Compact representations of existing equilibria: Solov'ev fit and MXH-Chebyshev (#1166).

Inverse representations -- equilibrium in, a few coefficients and their
fidelity out -- kept apart from the generators (``solve_solovev_constraints``,
``solve_guazzotto_freidberg``, CHEASE synthesis) that go the other way.
:mod:`vaft.process.equilibrium` is the public import location.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.constants import mu_0 as MU0
from scipy.interpolate import RectBivariateSpline

from vaft.data.equilibrium import (
    SOLOVEV_BASIS_SIZES,
    Contour,
    DerivationProvenance,
    MXHChebyshevRepresentation,
    SolovevEquilibrium,
    SolovevFit,
)

#: Fit states, in the issue's contract: no result; a result the model cannot
#: represent at all; a valid result outside the tolerance; an accepted one.
#: A metric that could not be evaluated is absent (or None), never False.
FIT_STATUSES = ("failed", "not_representable", "poor_fidelity", "accepted")


def _per_radian_factor(eq: Any) -> float:
    per_radian = None if eq.convention is None else eq.convention.psi_per_radian
    if per_radian is None:
        raise ValueError("the record's flux unit (Wb or Wb/rad) is not identified; declare the convention")
    return 1.0 if per_radian else 2*np.pi


def _symmetric_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """RMS and maximum of the two-way nearest distances between point sets."""
    from scipy.spatial import cKDTree

    d1 = cKDTree(second).query(first)[0]
    d2 = cKDTree(first).query(second)[0]
    both = np.r_[d1, d2]
    return float(np.sqrt(np.mean(both**2))), float(np.max(both))


def _dense(contour: Contour, count: int = 2048) -> np.ndarray:
    from vaft.process._equilibrium_parametric import _resample_contour

    return _resample_contour(contour, count).points


# --- Solov'ev fit -----------------------------------------------------------------


def fit_solovev(
    equilibrium: Any, *, basis: str = "cerfon_freidberg", psi_n_max: float = 1.0, tolerance: float = 0.02,
) -> SolovevFit:
    """Represent an equilibrium by the Solov'ev solution closest to its flux map.

    The inverse of :func:`solve_solovev_constraints`: instead of shaping a
    Solov'ev model to requested boundary points, find the constant ``p'`` and
    ``FF'`` and the homogeneous coefficients whose flux best matches a given
    equilibrium inside its boundary.  The flux is linear in all of them, so the
    fit is one linear least squares; how far the result is from the
    equilibrium is reported, not assumed.

    Parameters
    ----------
    equilibrium : EquilibriumData, ODS, GEQDSK or path
        Adapted through :func:`as_equilibrium`; needs psi, its axis and
        boundary values, an LCFS and a declared flux unit [-].
    basis : str, optional
        Homogeneous basis of :class:`~vaft.data.equilibrium.SolovevEquilibrium`:
        ``"classic"``, ``"cerfon_freidberg_even"`` or ``"cerfon_freidberg"`` [-].
    psi_n_max : float, optional
        Outermost normalized flux included in the fit [-].
    tolerance : float, optional
        Largest RMS flux error, and boundary error in minor radii, at which
        the fit is accepted [-].

    Returns
    -------
    SolovevFit
        The Solov'ev model (per-radian flux, ``rref`` the geometric centre),
        the fidelity metrics -- RMS and maximum flux error over the flux span,
        RMS ``|grad psi|`` error, magnetic-axis displacement, boundary RMS and
        maximum distance, and whether the topology matches (None when it
        could not be classified) -- the status (``accepted``,
        ``poor_fidelity``, ``not_representable`` when the fitted flux has no
        closed boundary, ``failed``) and reason, and the provenance [-].

    Raises
    ------
    ValueError
        An unknown basis, or a record without psi, an LCFS or a declared flux
        unit.

    Processing steps
    ----------------
    1. Convert the flux to per radian and select the grid points inside the
       LCFS with ``psi_N <= psi_n_max``.
    2. Solve for the basis coefficients, ``p'`` and ``FF'`` by least squares
       on the flux there.
    3. Evaluate the fitted model on the grid and measure the flux, field,
       axis, boundary and topology errors against the equilibrium.

    Defaults
    --------
    ``basis = "cerfon_freidberg"`` (twelve terms, up-down asymmetry allowed)
    and ``tolerance = 0.02`` are numerical conveniences: two percent of the
    flux span is roughly where a Solov'ev stand-in stops being useful for
    shape and field studies.

    Convention
    ----------
    The fit is done in per-radian flux, so the returned ``pprime`` and
    ``ffprime`` are per weber per radian whatever the input convention.  The
    model carries the record's orientation, which the metrics do not depend
    on; exporting it with :func:`solovev_to_equilibrium` needs the record's
    COCOS as the target convention, not the default.  Errors are relative to the flux span
    ``|psi_boundary - psi_axis|`` and to the minor radius.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A Solov'ev model has constant sources, so a reconstruction with peaked or
    hollow profiles fits only approximately, and the fitted ``p'`` and ``FF'``
    are effective values, not the equilibrium's.  A poor fit is returned with
    status ``poor_fidelity``, not rejected.

    Provenance
    ----------
    .. [1] Solov'ev (1968) and Cerfon and Freidberg, Phys. Plasmas 17, 032502
       (2010), for the model; the fit is the linear inverse of #1166.
    """
    from vaft.process._equilibrium_parametric import (
        _solovev_basis_derivative, _solovev_particular_derivative, as_equilibrium,
        derive_boundary_representation, evaluate_solovev, solovev_to_equilibrium,
    )
    from vaft.process.equilibrium import fractional_cell_weights_from_boundary

    if basis not in SOLOVEV_BASIS_SIZES:
        raise ValueError(f"basis must be one of {tuple(SOLOVEV_BASIS_SIZES)}, got {basis!r}")
    eq = as_equilibrium(equilibrium)
    if eq.psi is None or eq.lcfs is None or eq.psi_axis is None or eq.psi_boundary is None:
        raise ValueError("fit_solovev needs psi, psi_axis, psi_boundary and an LCFS")
    factor = _per_radian_factor(eq)
    r = np.asarray(eq.r, dtype=float); z = np.asarray(eq.z, dtype=float)
    psi = np.asarray(eq.psi, dtype=float)/factor
    psi_axis, psi_boundary = eq.psi_axis/factor, eq.psi_boundary/factor
    span = psi_boundary - psi_axis
    rm, zm = np.meshgrid(r, z, indexing="ij")
    psi_n = (psi - psi_axis)/span
    weights = fractional_cell_weights_from_boundary(r, z, eq.lcfs.r, eq.lcfs.z)
    use = (weights > 0.5) & (psi_n <= psi_n_max)
    rref = float(0.5*(np.max(eq.lcfs.r) + np.min(eq.lcfs.r)))
    unit = SolovevEquilibrium(np.zeros(SOLOVEV_BASIS_SIZES[basis]), 1.0, 0.0, rref, basis=basis)
    unit_ff = SolovevEquilibrium(np.zeros(SOLOVEV_BASIS_SIZES[basis]), 0.0, 1.0, rref, basis=basis)
    R, Z = rm[use], zm[use]
    columns = np.column_stack([
        *_solovev_basis_derivative(basis, rref, R, Z, "psi"),
        _solovev_particular_derivative(unit, R, Z, "psi"),
        _solovev_particular_derivative(unit_ff, R, Z, "psi"),
    ])
    scale = np.linalg.norm(columns, axis=0); scale[scale == 0] = 1.0
    solution, _, rank, _ = np.linalg.lstsq(columns/scale, psi[use], rcond=None)
    solution = solution/scale
    provenance = DerivationProvenance("linear least squares of the Solov'ev flux inside the LCFS",
                                      source_type=str(eq.metadata.get("source_type", "native")),
                                      source_fields=("psi", "lcfs"), convention=eq.convention,
                                      tolerances={"tolerance": tolerance, "psi_n_max": psi_n_max},
                                      notes=(f"basis={basis}", f"points={int(use.sum())}"))
    if rank < columns.shape[1]:
        return SolovevFit(None, {}, "failed", f"the fit is rank deficient ({rank}/{columns.shape[1]})", provenance)
    n = SOLOVEV_BASIS_SIZES[basis]
    f_edge = float(np.asarray(eq.f)[-1]) if eq.f is not None else 1.0
    pressure_edge = float(np.asarray(eq.pressure)[-1]) if eq.pressure is not None else 0.0
    model = SolovevEquilibrium(solution[:n], float(solution[n]), float(solution[n+1]), rref,
                               psi_boundary=psi_boundary, pressure_boundary=pressure_edge,
                               f_boundary=f_edge, basis=basis)  # f_sign follows f_edge (#1307)
    model_values = evaluate_solovev(model, rm, zm, cocos=11)
    model_psi = model_values["psi"]
    error = (model_psi - psi)[use]/abs(span)
    grad_eq = np.hypot(*np.gradient(psi, r, z, edge_order=2))
    grad_fit = np.hypot(model_values["dpsi_dr"], model_values["dpsi_dz"])
    bp_scale = float(np.sqrt(np.mean(grad_eq[use]**2)))
    metrics: dict[str, float | bool] = {
        "psi_rms_error": float(np.sqrt(np.mean(error**2))),
        "psi_max_error": float(np.max(np.abs(error))),
        "grad_psi_rms_error": float(np.sqrt(np.mean((grad_fit - grad_eq)[use]**2))/bp_scale) if bp_scale > 0 else float("nan"),
    }
    minor = 0.5*float(np.ptp(eq.lcfs.r))
    try:
        # The record's own wall, so limited-vs-diverted is classified alike on both.
        fitted = solovev_to_equilibrium(model, r, z, limiter=eq.limiter, convention=11)
    except ValueError as exc:
        state = "not_representable" if "closed contour" in str(exc) else "failed"
        return SolovevFit(model, metrics, state, f"the fitted model cannot be exported on this grid: {exc}", provenance)
    if eq.magnetic_axis is not None and fitted.magnetic_axis is not None:
        metrics["axis_displacement"] = float(np.hypot(*(np.subtract(fitted.magnetic_axis, eq.magnetic_axis))))/minor
    rms, maximum = _symmetric_distance(_dense(fitted.lcfs), _dense(eq.lcfs))
    metrics["boundary_rms_error"] = rms/minor
    metrics["boundary_max_error"] = maximum/minor
    try:
        metrics["topology_match"] = (derive_boundary_representation(fitted).topology
                                     == derive_boundary_representation(eq).topology)
    except Exception:  # noqa: BLE001 -- not evaluated, which is not a mismatch
        metrics["topology_match"] = None
    fitted_ok = (metrics["psi_rms_error"] <= tolerance and metrics["boundary_rms_error"] <= tolerance
                 and metrics.get("axis_displacement", 0.0) <= 5*tolerance
                 and metrics["topology_match"] is not False)
    status = "accepted" if fitted_ok else "poor_fidelity"
    reason = None if fitted_ok else (
        f"flux RMS {metrics['psi_rms_error']:.3g}, boundary RMS {metrics['boundary_rms_error']:.3g} and axis "
        f"shift {metrics.get('axis_displacement', float('nan')):.3g} (minor radii), topology match "
        f"{metrics['topology_match']}, against tolerance {tolerance:g}")
    return SolovevFit(model, metrics, status, reason, provenance)


# --- MXH-Chebyshev ----------------------------------------------------------------


def _mxh_surface(contour: Contour, harmonics: int) -> dict[str, Any]:
    """MXH parameters of one closed surface from Eqs. (6)-(8) of Xie and Li.

    ``R = R_c + r cos(theta_bar)``, ``Z = Z_c + kappa r sin(theta)`` with
    ``theta_bar = theta + c0 + sum c_m cos(m theta) + s_m sin(m theta)``.
    """
    points = _dense(contour, 1024)
    R, Z = points[:, 0], points[:, 1]
    r_c, r = 0.5*(R.max() + R.min()), 0.5*(R.max() - R.min())
    z_c = 0.5*(Z.max() + Z.min())
    kappa = 0.5*(Z.max() - Z.min())/r
    sin_theta = np.clip((Z - z_c)/(kappa*r), -1, 1)
    cos_bar = np.clip((R - r_c)/r, -1, 1)
    # Walk the surface counter-clockwise from its top; theta runs pi/2 -> 5 pi/2.
    area = 0.5*np.sum(R*np.roll(Z, -1) - np.roll(R, -1)*Z)
    if area < 0:
        R, Z, sin_theta, cos_bar = R[::-1], Z[::-1], sin_theta[::-1], cos_bar[::-1]
    start = int(np.argmax(Z))
    R, Z, sin_theta, cos_bar = (np.roll(a, -start) for a in (R, Z, sin_theta, cos_bar))
    bottom = int(np.argmin(Z))
    theta = np.empty_like(Z)
    theta[:bottom+1] = np.pi - np.arcsin(sin_theta[:bottom+1])          # inboard, top to bottom
    theta[bottom+1:] = 2*np.pi + np.arcsin(sin_theta[bottom+1:])        # outboard, bottom to top
    # theta_bar rises monotonically with theta around the surface, so its branch
    # follows R's extrema, not its distance to theta: arccos from the top to
    # the innermost point, 2 pi - arccos to the outermost, 2 pi + arccos after.
    # Choosing by distance to theta picks the mirror branch near the midplane
    # of a tilted surface.
    base = np.arccos(cos_bar)
    inner = int(np.argmin(R))
    outer = inner + int(np.argmax(R[inner:]))
    theta_bar = np.empty_like(base)
    theta_bar[:inner+1] = base[:inner+1]
    theta_bar[inner+1:outer+1] = 2*np.pi - base[inner+1:outer+1]
    theta_bar[outer+1:] = 2*np.pi + base[outer+1:]
    m = np.arange(1, harmonics + 1)
    design = np.column_stack([np.ones_like(theta), *(f(k*theta) for k in m for f in (np.cos, np.sin))])
    coefficients, *_ = np.linalg.lstsq(design, theta_bar - theta, rcond=None)
    return {"r_c": r_c, "z_c": z_c, "r": r, "kappa": kappa, "c0": coefficients[0],
            "c": coefficients[1::2], "s": coefficients[2::2]}


def _radial_basis(rho: np.ndarray, order: int) -> np.ndarray:
    """Eq. (10): u_l(rho) = (1 - rho**2) T_l(2 rho**2 - 1), l = 0 .. order."""
    x = 2*rho**2 - 1
    return np.column_stack([(1 - rho**2)*np.polynomial.chebyshev.Chebyshev.basis(l)(x) for l in range(order + 1)])


_PROFILE_NAMES = ("h", "v", "kappa", "a", "c0")


def _profile_names(harmonics: int) -> tuple[str, ...]:
    return _PROFILE_NAMES + tuple(f"c{m}" for m in range(1, harmonics + 1)) + tuple(f"s{m}" for m in range(1, harmonics + 1))


def evaluate_mxh_chebyshev(representation: MXHChebyshevRepresentation, rho: Any, theta: Any) -> tuple[np.ndarray, np.ndarray]:
    """Flux-surface points of an MXH-Chebyshev representation.

    Parameters
    ----------
    representation : MXHChebyshevRepresentation
        A fitted representation [-].
    rho : array_like
        ``sqrt(psi_N)``, in (0, 1] [-].
    theta : array_like
        Geometric poloidal angle of Eq. (7), broadcast against *rho* [rad].

    Returns
    -------
    tuple of np.ndarray
        ``(R, Z)`` [m].

    Convention
    ----------
    Eqs. (6)-(11) of Xie and Li: ``R = R0 + h + rho a cos(theta_bar)``,
    ``Z = Z0 + v + kappa rho a sin(theta)``, every profile ``f(rho) = f_edge +
    sum_l f_l (1 - rho**2) T_l(2 rho**2 - 1)``.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] H. Xie and Y. Li, arXiv:2601.02942 (2026), Eqs. (6)-(11).
    """
    rho, theta = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(theta, dtype=float))
    basis = _radial_basis(rho.ravel(), representation.radial_order)

    def profile(name: str) -> np.ndarray:
        edge, coefficients = representation.profiles[name]
        return (edge + basis @ np.asarray(coefficients)).reshape(rho.shape)

    theta_bar = theta + profile("c0")
    for m in range(1, representation.harmonics + 1):
        theta_bar = theta_bar + profile(f"c{m}")*np.cos(m*theta) + profile(f"s{m}")*np.sin(m*theta)
    radius = rho*profile("a")
    return (representation.r0 + profile("h") + radius*np.cos(theta_bar),
            representation.z0 + profile("v") + profile("kappa")*radius*np.sin(theta))


def fit_mxh_chebyshev(
    equilibrium: Any, *, harmonics: int = 2, radial_order: int = 4, surfaces: int = 24,
    tolerance: float = 0.01, boundary_tolerance: float = 0.02,
) -> MXHChebyshevRepresentation:
    """Represent an equilibrium's flux surfaces by MXH shapes with Chebyshev radial profiles.

    Every surface ``rho = sqrt(psi_N)`` is an MXH shape -- centre, minor radius,
    elongation and a harmonic distortion of the angle -- and each of those
    numbers is a shifted-Chebyshev series in ``rho`` anchored to its LCFS
    value.  The coefficient vector is a compact, grid-independent stand-in for
    the flux map, and the fit reports how faithful it is.

    Parameters
    ----------
    equilibrium : EquilibriumData, ODS, GEQDSK or path
        Adapted through :func:`as_equilibrium`; needs psi and an LCFS [-].
    harmonics : int, optional
        MXH harmonics ``M`` of the angle distortion, at least one [-].
    radial_order : int, optional
        Chebyshev order ``L`` of every radial profile [-].
    surfaces : int, optional
        Traced surfaces the profiles are fitted to [-].
    tolerance : float, optional
        Largest RMS normalized-flux error at which the fit is accepted [-].
    boundary_tolerance : float, optional
        Largest RMS boundary error, in minor radii, at which it is accepted [-].

    Returns
    -------
    MXHChebyshevRepresentation
        ``R0``, ``Z0``, the orders, every profile's edge value and Chebyshev
        coefficients, the parameter count, fidelity metrics (RMS and maximum
        ``psi_N`` error of the represented surfaces, boundary error in minor
        radii, axis displacement, per-surface MXH shape error), the status and
        reason, and the provenance [-].

    Raises
    ------
    ValueError
        *harmonics* below one, a negative *radial_order*, fewer surfaces than
        the radial order needs, or no psi map or LCFS.

    Processing steps
    ----------------
    1. Trace surfaces at ``rho`` from 0.15 to 0.97 and take the LCFS as ``rho = 1``.
    2. Extract each surface's MXH parameters: centre and minor radius from its
       radial extent, elongation from its height, the geometric angle from
       Eq. (7), the distorted angle from Eq. (6), and their difference as a
       harmonic series (Eq. 8).
    3. Fit every parameter's radial profile as ``f_edge + sum_l f_l u_l(rho)``
       with ``u_l`` of Eq. (10), the edge value fixed to the LCFS.
    4. Evaluate the representation on a (rho, theta) grid and measure
       ``psi_N(R, Z) - rho**2`` on the equilibrium's own flux map.

    Defaults
    --------
    ``harmonics = 2`` and ``radial_order = 4`` give 9 x 5 = 45 parameters,
    inside the paper's "fewer than 100" for high fidelity; ``tolerance =
    0.01`` matches its 1e-2 accuracy level and ``boundary_tolerance = 0.02``
    leaves room for a diverted corner.  All are numerical conveniences.

    Convention
    ----------
    The radial coordinate is ``rho_psi = sqrt(psi_N)`` (Eq. 4), so the flux
    is ``psi_axis + (psi_boundary - psi_axis) rho**2`` exactly and the
    geometric profiles carry the shape; ``R0`` and ``Z0`` are the LCFS centre,
    so ``h`` and ``v`` vanish there.  Purely geometric: no COCOS enters.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Xie and Li use this basis as the unknown of a Grad-Shafranov solver; here
    it only represents an existing equilibrium, so no force balance is
    imposed and the source profiles are not fitted.  A diverted LCFS has a
    corner an MXH shape of low order cannot follow; the boundary error says
    so.  The profiles are anchored at ``rho = 1`` only, so the axis position
    is an extrapolation, reported as a metric.

    Provenance
    ----------
    .. [1] H. Xie and Y. Li, *What is the minimum number of parameters
       required to represent solutions of the Grad-Shafranov equation?*,
       arXiv:2601.02942 (2026), Eqs. (4), (6)-(11).
    """
    from vaft.process._equilibrium_parametric import _contour_at_level, as_equilibrium

    harmonics, radial_order = int(harmonics), int(radial_order)
    if harmonics < 1 or radial_order < 0:
        raise ValueError("harmonics must be at least 1 and radial_order non-negative")
    if surfaces < radial_order + 2:
        raise ValueError("need at least radial_order + 2 surfaces")
    eq = as_equilibrium(equilibrium)
    if eq.psi is None or eq.lcfs is None or eq.psi_axis is None or eq.psi_boundary is None:
        raise ValueError("fit_mxh_chebyshev needs psi, psi_axis, psi_boundary and an LCFS")
    rho_levels = np.linspace(0.15, 0.97, int(surfaces) - 1)
    rows = []
    for rho in rho_levels:
        contour = _contour_at_level(eq, float(rho**2))
        if contour is not None and contour.closed:
            rows.append((rho, _mxh_surface(contour, harmonics)))
    edge = _mxh_surface(eq.lcfs, harmonics)
    rows.append((1.0, edge))
    rho = np.array([row[0] for row in rows])
    if rho.size - 1 < radial_order + 1:
        return MXHChebyshevRepresentation(
            r0=float(edge["r_c"]), z0=float(edge["z_c"]), harmonics=harmonics, radial_order=radial_order,
            profiles={}, parameter_count=0, metrics={}, status="failed",
            reason=f"only {rho.size - 1} interior surfaces closed; the radial order needs {radial_order + 1}",
            provenance=DerivationProvenance("MXH-Chebyshev fit", source_fields=("psi", "lcfs")))
    r0, z0 = edge["r_c"], edge["z_c"]
    samples = {
        "h": np.array([p["r_c"] - r0 for _, p in rows]), "v": np.array([p["z_c"] - z0 for _, p in rows]),
        "kappa": np.array([p["kappa"] for _, p in rows]), "a": np.array([p["r"]/rh for rh, p in rows]),
        "c0": np.array([p["c0"] for _, p in rows]),
    }
    for m in range(1, harmonics + 1):
        samples[f"c{m}"] = np.array([p["c"][m-1] for _, p in rows])
        samples[f"s{m}"] = np.array([p["s"][m-1] for _, p in rows])
    basis = _radial_basis(rho, radial_order)
    profiles = {}
    for name in _profile_names(harmonics):
        edge_value = float(samples[name][-1])
        coefficients, *_ = np.linalg.lstsq(basis[:-1], samples[name][:-1] - edge_value, rcond=None)
        profiles[name] = (edge_value, tuple(float(c) for c in coefficients))
    provenance = DerivationProvenance(
        "MXH shapes per surface with shifted-Chebyshev radial profiles (Xie & Li 2026)",
        source_type=str(eq.metadata.get("source_type", "native")), source_fields=("psi", "lcfs"),
        radial_coordinate="rho_psi", convention=eq.convention,
        tolerances={"tolerance": tolerance}, notes=(f"surfaces={rho.size}",),
    )
    representation = MXHChebyshevRepresentation(
        r0=float(r0), z0=float(z0), harmonics=harmonics, radial_order=radial_order, profiles=profiles,
        parameter_count=len(profiles)*(radial_order + 1), metrics={}, status="accepted", reason=None,
        provenance=provenance,
    )
    # Fidelity on the equilibrium's own flux map.
    spline = RectBivariateSpline(eq.r, eq.z, (np.asarray(eq.psi) - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis))
    test_rho = np.linspace(0.2, 0.95, 16)[:, None]; test_theta = np.linspace(0, 2*np.pi, 128, endpoint=False)[None, :]
    R, Z = evaluate_mxh_chebyshev(representation, test_rho, test_theta)
    psi_error = spline.ev(R, Z) - test_rho**2
    minor = 0.5*float(np.ptp(eq.lcfs.r))
    boundary = np.column_stack(evaluate_mxh_chebyshev(representation, np.ones(512), np.linspace(0, 2*np.pi, 512, endpoint=False)))
    boundary_rms, boundary_max = _symmetric_distance(boundary, _dense(eq.lcfs))
    shape_errors = []
    for rh, p in rows[:-1]:
        th = np.linspace(0, 2*np.pi, 256, endpoint=False)
        bar = th + p["c0"] + sum(p["c"][m-1]*np.cos(m*th) + p["s"][m-1]*np.sin(m*th) for m in range(1, harmonics + 1))
        pts = np.column_stack((p["r_c"] + p["r"]*np.cos(bar), p["z_c"] + p["kappa"]*p["r"]*np.sin(th)))
        traced = _contour_at_level(eq, float(rh**2))
        shape_errors.append(_symmetric_distance(pts, _dense(traced))[0]/minor)
    metrics: dict[str, float] = {
        "psi_n_rms_error": float(np.sqrt(np.mean(psi_error**2))),
        "psi_n_max_error": float(np.max(np.abs(psi_error))),
        "boundary_rms_error": boundary_rms/minor, "boundary_max_error": boundary_max/minor,
        "surface_shape_max_rms_error": float(np.max(shape_errors)) if shape_errors else float("nan"),
    }
    if eq.magnetic_axis is not None:
        axis = evaluate_mxh_chebyshev(representation, np.array([1e-6]), np.array([0.0]))
        metrics["axis_displacement"] = float(np.hypot(axis[0][0] - eq.magnetic_axis[0], axis[1][0] - eq.magnetic_axis[1]))/minor
    accepted = metrics["psi_n_rms_error"] <= tolerance and metrics["boundary_rms_error"] <= boundary_tolerance
    return MXHChebyshevRepresentation(
        r0=representation.r0, z0=representation.z0, harmonics=harmonics, radial_order=radial_order,
        profiles=profiles, parameter_count=representation.parameter_count, metrics=metrics,
        status="accepted" if accepted else "poor_fidelity",
        reason=None if accepted else (f"psi_N RMS error {metrics['psi_n_rms_error']:.3g} (tolerance {tolerance:g}), "
                                      f"boundary RMS {metrics['boundary_rms_error']:.3g} (tolerance {boundary_tolerance:g})"),
        provenance=provenance,
    )


__all__ = ["FIT_STATUSES", "evaluate_mxh_chebyshev", "fit_mxh_chebyshev", "fit_solovev"]
