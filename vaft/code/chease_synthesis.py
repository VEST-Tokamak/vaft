"""Fixed-boundary CHEASE equilibria synthesized from 0D descriptors (#120).

A fixed-boundary Grad-Shafranov solution is fixed by the boundary, the two
source functions and one normalization.  So the specification here is exactly
that and nothing more:

* **boundary** -- ``R0, a, kappa, delta, Z0`` (and optional squareness), turned
  into a Miller LCFS;
* **sources** -- the *shapes* of ``p'(psi_N)`` and ``FF'(psi_N)`` as
  generalized-parabolic kernels, plus ``pressure_fraction``, the share of the
  on-axis Grad-Shafranov source carried by the pressure term;
* **normalization** -- the one quantity CHEASE can impose: the plasma current
  (``NCSCAL = 2``) *or* the safety factor at ``psi_N = 0.95`` (``NCSCAL = 1``).

Everything else -- q0, the other of Ip / q95, beta_p, beta_N, l_i, the
Shafranov shift -- is an *achieved* output, measured on the solved equilibrium
and reported next to the request, never imposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.constants import mu_0 as MU0

from .chease import CHEASEConfig, CHEASEResult, refine_equilibrium

#: CHEASE's normalization modes, by the target they impose.
SUPPORTED_TARGETS = {"plasma_current": 2, "q95": 1}

#: Result states.
STATUSES = ("success", "invalid_geometry", "invalid_source_model", "unsupported_target",
            "non_converged", "invalid_equilibrium", "constraint_not_reached", "boundary_mismatch")


@dataclass(frozen=True)
class ZeroDimensionalEquilibriumSpec:
    """A fixed-boundary equilibrium request in 0D descriptors and explicit source shapes.

    ``p'(psi_N) ~ -pressure_fraction * (1 - psi_N**pressure_alpha)**pressure_beta
    / (mu0 R0**2)`` and ``FF'(psi_N) ~ -(1 - pressure_fraction) * (1 -
    psi_N**current_alpha)**current_beta``, so ``pressure_fraction`` is the share
    of the on-axis source ``mu0 R0**2 p' + FF'`` carried by the pressure.  Their
    common amplitude is not an input: CHEASE sets it to meet the target.
    """

    major_radius: float
    minor_radius: float
    elongation: float
    triangularity: float
    toroidal_field: float
    plasma_current: float
    vertical_position: float = 0.0
    squareness: float = 0.0
    pressure_fraction: float = 0.5
    pressure_alpha: float = 1.0
    pressure_beta: float = 2.0
    current_alpha: float = 1.0
    current_beta: float = 1.0


@dataclass
class SyntheticEquilibriumResult:
    """Requested vs achieved: the outcome of :func:`synthesize_equilibrium_from_0d`."""

    status: str
    reason: str | None
    spec: ZeroDimensionalEquilibriumSpec
    target: str
    requested: Mapping[str, float]
    achieved: Mapping[str, float] = field(default_factory=dict)
    residuals: Mapping[str, float] = field(default_factory=dict)
    boundary: Any = None
    source_profiles: Mapping[str, np.ndarray] = field(default_factory=dict)
    chease: Optional[CHEASEResult] = None
    refined_geqdsk: Optional[Path] = None
    refined_ods: Any = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


def _validate(spec: ZeroDimensionalEquilibriumSpec) -> tuple[str, str] | None:
    if not (spec.major_radius > 0 and 0 < spec.minor_radius < spec.major_radius):
        return "invalid_geometry", "requires major_radius > minor_radius > 0"
    if spec.elongation <= 0 or abs(spec.triangularity) >= 1 or abs(spec.squareness) >= 0.5:
        return "invalid_geometry", "requires elongation > 0, |triangularity| < 1 and |squareness| < 0.5"
    if spec.toroidal_field == 0 or spec.plasma_current == 0:
        return "invalid_geometry", "toroidal_field and plasma_current must be non-zero"
    if not 0 <= spec.pressure_fraction < 1:
        return "invalid_source_model", "pressure_fraction must lie in [0, 1): FF' may not vanish identically"
    for name in ("pressure_alpha", "pressure_beta", "current_alpha", "current_beta"):
        if not getattr(spec, name) > 0:
            return "invalid_source_model", f"{name} must be positive"
    return None


def construct_boundary(spec: ZeroDimensionalEquilibriumSpec, points: int = 256):
    """The Miller LCFS of a specification, counter-clockwise, as a closed contour."""
    from vaft.data.equilibrium import Contour, MillerSurface
    from vaft.process.equilibrium import evaluate_miller

    theta = np.linspace(0.0, 2*np.pi, int(points), endpoint=False)
    r, z = evaluate_miller(MillerSurface(spec.minor_radius, spec.major_radius, spec.vertical_position,
                                         spec.elongation, spec.triangularity, spec.squareness), theta)
    return Contour(r, z, True)


def _source_shapes(spec: ZeroDimensionalEquilibriumSpec, psi_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from vaft.formula.equilibrium import generalized_parabolic_profile

    pp = spec.pressure_fraction*generalized_parabolic_profile(psi_n, alpha=spec.pressure_alpha, beta=spec.pressure_beta)
    ff = (1 - spec.pressure_fraction)*generalized_parabolic_profile(psi_n, alpha=spec.current_alpha, beta=spec.current_beta)
    return pp, ff


def build_input_equilibrium(spec: ZeroDimensionalEquilibriumSpec, *, q95: float | None = None, resolution: int = 129):
    """A COCOS 11 ``EquilibriumData`` carrying the boundary and source shapes for CHEASE.

    Only the boundary, ``p'``, ``FF'``, ``F`` at the edge, the current and --
    for a q95 target -- ``q(0.95)`` reach CHEASE; the flux map is a smooth
    placeholder of the right sign and span, since CHEASE solves its own.
    """
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process._equilibrium_parametric import _detect_convention

    boundary = construct_boundary(spec)
    r0, a, kappa, z0 = spec.major_radius, spec.minor_radius, spec.elongation, spec.vertical_position
    r = np.linspace(r0 - 1.4*a, r0 + 1.4*a, int(resolution))
    z = np.linspace(z0 - 1.4*kappa*a, z0 + 1.4*kappa*a, int(resolution))
    rm, zm = np.meshgrid(r, z, indexing="ij")
    # A rough scale for the flux span: B_p ~ mu0 Ip / (2 pi a sqrt((1+kappa^2)/2)), psi ~ R0 a B_p / 2.
    b_pol = MU0*abs(spec.plasma_current)/(2*np.pi*a*np.sqrt((1 + kappa**2)/2))
    span = 2*np.pi*r0*a*b_pol/2                                          # Wb, COCOS 11
    sign = np.sign(spec.plasma_current)
    psi_axis = -sign*span
    rho2 = ((rm - r0)/a)**2 + ((zm - z0)/(kappa*a))**2
    psi = psi_axis*(1 - rho2)
    psi_1d_n = np.linspace(0.0, 1.0, r.size)
    pp_shape, ff_shape = _source_shapes(spec, psi_1d_n)
    # The Grad-Shafranov source per radian is A * shape with A chosen so the
    # placeholder current is near the request: j ~ A <shape> / (mu0 R0).
    # CHEASE rescales p' and FF' together to meet its target, so only their
    # ratio (pressure_fraction), their signs and their shapes survive.
    area = np.pi*a**2*kappa
    drive = float(np.mean(pp_shape + ff_shape)) or 1.0
    amplitude = MU0*abs(spec.plasma_current)*r0/(area*drive)
    # COCOS 11, stored per weber (per radian / 2 pi); negative for a positive current.
    pprime = -sign*amplitude*pp_shape/(MU0*r0**2)/(2*np.pi)
    ffprime = -sign*amplitude*ff_shape/(2*np.pi)
    dpsi = 0.0 - psi_axis
    psi_1d = psi_axis + psi_1d_n*dpsi
    pressure = -np.array([np.trapezoid(pprime[i:], psi_1d[i:]) for i in range(r.size)])
    f_edge = spec.toroidal_field*r0
    f2 = f_edge**2 - 2*np.array([np.trapezoid(ffprime[i:], psi_1d[i:]) for i in range(r.size)])
    if np.any(f2 <= 0):
        raise ValueError("the placeholder F^2 turns negative; raise toroidal_field")
    f = np.sign(f_edge)*np.sqrt(f2)
    # A quadratic q with q(0.95) = q95: the adapter samples q at the constraint
    # surface with a cubic interpolation, which reproduces a quadratic exactly.
    q_edge95 = q95 if q95 is not None else 3.0
    # COCOS 11 has sigma_rho_theta_phi = +1, so q carries the sign of Ip*Bt.
    q = np.sign(spec.plasma_current*spec.toroidal_field)*(1.0 + (q_edge95 - 1.0)*(psi_1d_n/0.95)**2)
    convention = _detect_convention(explicit=11, bt0=spec.toroidal_field, ip=spec.plasma_current, q=q,
                                    psi_1d=psi_1d, source="0D synthesis input")
    return EquilibriumData(
        r=r, z=z, psi=psi, psi_axis=psi_axis, psi_boundary=0.0, magnetic_axis=(r0, z0), lcfs=boundary,
        psi_1d=psi_1d, pressure=np.maximum(pressure, 0.0), f=f, q=q, pprime=pprime, ffprime=ffprime,
        ip=float(spec.plasma_current), bt0=float(spec.toroidal_field), r0=r0, convention=convention,
        metadata={"source_type": "0d_synthesis_input"},
    )


def _native_output(result: CHEASEResult) -> Path | None:
    candidate = Path(result.workdir)/"EQDSK_COCOS_02.OUT"
    return candidate if candidate.exists() else None


def synthesize_equilibrium_from_0d(
    spec: ZeroDimensionalEquilibriumSpec, *, target: str = "plasma_current", q95: float | None = None,
    config: CHEASEConfig | None = None, current_tolerance: float = 0.01, q95_tolerance: float = 0.02,
    boundary_tolerance: float = 0.02,
) -> SyntheticEquilibriumResult:
    """Solve a fixed-boundary CHEASE equilibrium from 0D descriptors and source shapes.

    The boundary, the source shapes and one normalization are handed to
    CHEASE; everything the solve then implies -- q0, the other of Ip or q95,
    betas, l_i, the axis shift -- is measured on the result and returned
    beside the request.  Invalid input and unmet targets come back as
    explicit states, not as a plausible-looking equilibrium.

    Parameters
    ----------
    spec : ZeroDimensionalEquilibriumSpec
        Geometry, field, current and source shapes [-].
    target : str, optional
        What CHEASE normalizes to: ``"plasma_current"`` (``NCSCAL = 2``, the
        spec's current) or ``"q95"`` (``NCSCAL = 1``, the safety factor at
        ``psi_N = 0.95``) [-].
    q95 : float, optional
        The q95 to impose when *target* is ``"q95"`` [-].
    config : CHEASEConfig, optional
        Run settings; the solve-shape fields (boundary taken as given,
        scaling mode, q surface) are set here and override it [-].
    current_tolerance : float, optional
        Largest relative plasma-current miss accepted for a current target [-].
    q95_tolerance : float, optional
        Largest relative q95 miss accepted for a q95 target [-].
    boundary_tolerance : float, optional
        Largest miss of the solved shape against the request: relative for the
        elongation and the radii, absolute for the triangularities, and in
        minor radii for the vertical position [-].

    Returns
    -------
    SyntheticEquilibriumResult
        Status and reason (every problem found, the first one's state), the
        request, the achieved descriptors -- magnitudes, in CHEASE's
        orientation, with the refined output's signs checked against the
        request -- their residuals, the constructed boundary, the source
        shapes, the CHEASE run, and the refined g-file and ODS.  Without a
        *config*, CHEASE runs in a fresh temporary directory [-].
    """
    requested = {
        "major_radius": spec.major_radius, "minor_radius": spec.minor_radius, "elongation": spec.elongation,
        "triangularity": spec.triangularity, "vertical_position": spec.vertical_position,
        "toroidal_field": spec.toroidal_field, "plasma_current": spec.plasma_current,
    }
    if target not in SUPPORTED_TARGETS:
        return SyntheticEquilibriumResult(
            "unsupported_target",
            f"CHEASE can normalize to {sorted(SUPPORTED_TARGETS)} only; {target!r} would be an output, not a control",
            spec, target, requested)
    if target == "q95":
        if q95 is None or not q95 > 1:
            return SyntheticEquilibriumResult("unsupported_target", "a q95 target needs a q95 above one", spec, target, requested)
        requested["q95"] = float(q95)
    problem = _validate(spec)
    if problem is not None:
        return SyntheticEquilibriumResult(problem[0], problem[1], spec, target, requested)
    try:
        equilibrium = build_input_equilibrium(spec, q95=q95)
    except ValueError as error:
        return SyntheticEquilibriumResult("invalid_source_model", str(error), spec, target, requested)
    from vaft.data.eqdsk import from_equilibrium, read_geqdsk
    from vaft.process.equilibrium import as_equilibrium, contour_shape_parameters, derive_global_descriptors

    psi_n = np.linspace(0.0, 1.0, 101)
    pp_shape, ff_shape = _source_shapes(spec, psi_n)
    settings = dict(vars(config)) if config is not None else {}
    if config is None or Path(config.workdir) == Path("."):
        import tempfile

        # CHEASE writes a dozen files; never into the caller's working directory by default.
        settings["workdir"] = Path(tempfile.mkdtemp(prefix="vaft-0d-chease-"))
    settings.update(target_psin=1.0, ncscal=SUPPORTED_TARGETS[target],
                    q_constraint_psi_norm=0.95 if target == "q95" else None,
                    edge_zero=False, preserve_boundary_limiter=False)
    run = refine_equilibrium(from_equilibrium(equilibrium), CHEASEConfig(**settings))
    result = SyntheticEquilibriumResult(
        "success", None, spec, target, requested, boundary=equilibrium.lcfs,
        source_profiles={"psi_n": psi_n, "pprime_shape": pp_shape, "ffprime_shape": ff_shape},
        chease=run, refined_geqdsk=run.refined_geqdsk, refined_ods=run.refined_ods,
    )
    native = _native_output(run)
    if not run.ok or native is None:
        result.status, result.reason = "non_converged", f"CHEASE returned {run.returncode}; no EQDSK_COCOS_02.OUT"
        return result
    # The native file carries CHEASE's own sign pattern (COCOS 2, both signs
    # forced positive), so magnitudes are read there and signs are checked on
    # the refined output, which is returned in the request's orientation.
    solved = as_equilibrium(read_geqdsk(native))
    descriptors = derive_global_descriptors(solved).values
    achieved = {name: float(descriptors[name].value) for name in (
        "ip", "q0", "q95", "bt0", "major_radius", "minor_radius", "elongation", "geometric_center_z",
        "beta_t", "beta_n", "beta_p_boundary_average", "li_virial", "shafranov_shift",
        "magnetic_axis_r", "magnetic_axis_z", "volume") if name in descriptors and descriptors[name].available}
    for name in ("ip", "q0", "q95", "bt0"):
        if name in achieved:
            achieved[name] = abs(achieved[name])
    if "beta_p_boundary_average" in achieved:
        achieved["beta_p"] = achieved.pop("beta_p_boundary_average")
    if solved.lcfs is not None:
        # One estimator for the achieved triangularity and its residual.
        shape = contour_shape_parameters(solved.lcfs.r, solved.lcfs.z)
        achieved["triangularity_upper"] = float(shape["triangularity_upper"])
        achieved["triangularity_lower"] = float(shape["triangularity_lower"])
    residuals = {}
    if target == "plasma_current" and "ip" in achieved:
        residuals["plasma_current"] = achieved["ip"]/abs(spec.plasma_current) - 1
    if target == "q95" and "q95" in achieved:
        residuals["q95"] = achieved["q95"]/q95 - 1
    if "bt0" in achieved:
        residuals["toroidal_field"] = achieved["bt0"]/abs(spec.toroidal_field) - 1
    for name, want in (("elongation", spec.elongation), ("minor_radius", spec.minor_radius),
                       ("major_radius", spec.major_radius)):
        if name in achieved:
            residuals[name] = achieved[name]/want - 1
    # Absolute, not relative: a triangularity or a vertical position can be zero.
    for name in ("triangularity_upper", "triangularity_lower"):
        if name in achieved:
            residuals[name] = achieved[name] - spec.triangularity
    if "geometric_center_z" in achieved:
        residuals["vertical_position"] = (achieved["geometric_center_z"] - spec.vertical_position)/spec.minor_radius
    result.achieved, result.residuals = achieved, residuals

    problems: list[tuple[str, str]] = []
    axis_inside = (solved.lcfs is not None and solved.magnetic_axis is not None
                   and _inside(solved.lcfs, solved.magnetic_axis))
    sign_ok = _signs_match(run.refined_ods, spec)
    if not axis_inside:
        problems.append(("invalid_equilibrium", "the magnetic axis is not inside the solved boundary"))
    if sign_ok is False:
        problems.append(("invalid_equilibrium", "the refined output's Ip/Bt signs differ from the request"))
    if target == "plasma_current" and abs(residuals.get("plasma_current", np.inf)) > current_tolerance:
        problems.append(("constraint_not_reached", f"plasma current off by {residuals.get('plasma_current'):.3g}"))
    if target == "q95" and abs(residuals.get("q95", np.inf)) > q95_tolerance:
        problems.append(("constraint_not_reached", f"q95 off by {residuals.get('q95'):.3g}"))
    off = [n for n in ("elongation", "minor_radius", "major_radius", "triangularity_upper",
                       "triangularity_lower", "vertical_position") if abs(residuals.get(n, 0.0)) > boundary_tolerance]
    if off:
        problems.append(("boundary_mismatch", "the solved boundary departs from the request in " + ", ".join(off)))
    if problems:
        result.status = problems[0][0]
        result.reason = "; ".join(reason for _, reason in problems)
    return result


def _inside(contour: Any, point: tuple[float, float]) -> bool:
    from matplotlib.path import Path as MplPath

    return bool(MplPath(np.column_stack((contour.r, contour.z))).contains_point(point))


def _signs_match(ods: Any, spec: ZeroDimensionalEquilibriumSpec) -> bool | None:
    """Do the refined output's Ip and B0 carry the request's signs?  None when unreadable."""
    try:
        ip = float(ods["equilibrium.time_slice.0.global_quantities.ip"])
        b0 = float(np.ravel(ods["equilibrium.vacuum_toroidal_field.b0"])[0])
    except Exception:
        return None
    return bool(np.sign(ip) == np.sign(spec.plasma_current) and np.sign(b0) == np.sign(spec.toroidal_field))


__all__ = [
    "STATUSES", "SUPPORTED_TARGETS", "SyntheticEquilibriumResult", "ZeroDimensionalEquilibriumSpec",
    "build_input_equilibrium", "construct_boundary", "synthesize_equilibrium_from_0d",
]
