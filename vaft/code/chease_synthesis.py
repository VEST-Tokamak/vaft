"""Fixed-boundary CHEASE equilibria synthesized from 0D descriptors (#120).

A fixed-boundary Grad-Shafranov solution is fixed by the boundary, the two
source functions and one normalization.  So the specification here is exactly
that and nothing more:

* **boundary** -- ``R0, a, kappa, delta, Z0`` (and optional squareness), turned
  into a Miller LCFS;
* **sources** -- the *shapes* of ``p'(psi_N)`` and ``FF'(psi_N)`` as
  generalized-parabolic kernels, plus ``pressure_fraction``, the share of the
  on-axis Grad-Shafranov source carried by the pressure term; or, for either
  source, a #1045 barrier profile (edge pedestal, ITB, or both) whose analytic
  ``psi_N`` derivative is the shape (#1166 scope B);
* **normalization** -- the one quantity CHEASE can impose: the plasma current
  (``NCSCAL = 2``) *or* the safety factor at ``psi_N = 0.95`` (``NCSCAL = 1``).

Everything else -- q0, the other of Ip / q95, beta_p, beta_N, l_i, the
Shafranov shift -- is an *achieved* output, measured on the solved equilibrium
and reported next to the request, never imposed.  A steep pedestal ``p'`` on
a fixed boundary is likewise a *source prescription*: nothing here predicts a
pedestal, its width or its stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
from scipy.constants import mu_0 as MU0

from .chease import CHEASEConfig, CHEASEResult, refine_equilibrium

if TYPE_CHECKING:
    from vaft.data.analytic_plasma_state import AnalyticPlasmaState, AnalyticProfile

#: CHEASE's normalization modes, by the target they impose.
SUPPORTED_TARGETS = {"plasma_current": 2, "q95": 1}

#: Result states.
STATUSES = ("success", "invalid_geometry", "invalid_source_model", "unsupported_target",
            "non_converged", "invalid_equilibrium", "constraint_not_reached", "boundary_mismatch",
            "timeout")

#: The generalized-parabolic exponents' defaults; a barrier profile replaces
#: them, so a non-default value next to one is a contradiction.
_DEFAULT_PRESSURE_EXPONENTS = (1.0, 2.0)
_DEFAULT_CURRENT_EXPONENTS = (1.0, 1.0)

#: Samples on which a barrier profile's gradient is checked for sign.
_PROFILE_CHECK_GRID = np.linspace(0.0, 1.0, 20001)

#: Profile resolution of the CHEASE input when a barrier profile is a source:
#: CHEASE resamples it linearly, and 257 points keep a 0.05-wide pedestal's
#: p' peak within about 1 % of the analytic one.
_PROFILE_RESOLUTION = 257

#: Smallest net axis-to-separatrix drop of a barrier profile, relative to its
#: level ``max(|g(0)|, |g(1)|)``: the shape is ``-g'/drop``, so a drop below
#: this is a near-flat profile whose shape is set by 1/drop amplification.
_MIN_RELATIVE_DROP = 1e-3

#: Barrier p' peak checks, used when the pressure profile has barrier steps.
_PEAK_GRID = np.linspace(0.0, 1.0, 20001)


@dataclass(frozen=True)
class ZeroDimensionalEquilibriumSpec:
    """A fixed-boundary equilibrium request in 0D descriptors and explicit source shapes.

    ``p'(psi_N) ~ -pressure_fraction * (1 - psi_N**pressure_alpha)**pressure_beta
    / (mu0 R0**2)`` and ``FF'(psi_N) ~ -(1 - pressure_fraction) * (1 -
    psi_N**current_alpha)**current_beta``, so ``pressure_fraction`` is the share
    of the on-axis source ``mu0 R0**2 p' + FF'`` carried by the pressure.  Their
    common amplitude is not an input: CHEASE sets it to meet the target.

    Barrier-profile sources (#1166 scope B).  ``pressure_profile`` -- an
    :class:`~vaft.data.AnalyticProfile` (e.g. from
    :func:`vaft.process.profile.compose_analytic_profile`) or an
    :class:`~vaft.data.AnalyticPlasmaState`, whose ``p_total`` is used --
    replaces the generalized-parabolic ``p'`` kernel by ``-dp/dpsi_N``, the
    profile's analytic derivative.  ``current_profile`` does the same for
    ``FF'``: it is a profile ``g(psi_N)`` in the role of ``F**2/2`` above its
    edge value, and the ``FF'`` shape is ``-dg/dpsi_N``, so a pedestal step in
    ``g`` is an edge-current bump of the tanh step's ``sech**2`` shape,
    centred on the step and of the step's full width.  When either profile is
    given, both shapes are scaled to unit integral over ``psi_N`` before the
    split, and ``pressure_fraction`` is the pressure's share of the
    ``psi_N``-*integrated* source ``int (mu0 R0**2 p' + FF') dpsi_N`` rather
    than of the on-axis one (a profile's gradient may vanish on axis).

    **This changes what ``pressure_fraction`` means even for the source that
    stays generalized-parabolic**: with only a ``current_profile``, the
    unchanged pressure kernel is split on the integrated basis too, so at
    ``pressure_fraction = 0.5`` and the default kernels the *on-axis* pressure
    share is 0.667, not 0.5.  Compare a profile case with a kernel-only case
    through the achieved descriptors, not through ``pressure_fraction``.

    A barrier profile must not rise outward anywhere (the pressure, because
    the CHEASE input carries ``p'`` of one sign; ``g``, because a rising
    ``g`` is a reversed current), and its net drop must be at least 1e-3 of
    its level.  Only
    the shape survives: the profile's unit, absolute level and separatrix
    value are dropped, since CHEASE rescales the amplitude to the target and
    a constant pressure does not enter the Grad-Shafranov equation.  Leaving
    both unset reproduces the generalized-parabolic model exactly.
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
    pressure_profile: "AnalyticProfile | AnalyticPlasmaState | None" = None
    current_profile: "AnalyticProfile | None" = None


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
    achieved_profiles: Mapping[str, np.ndarray] = field(default_factory=dict)

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
    return _validate_profiles(spec)


def _validate_profiles(spec: ZeroDimensionalEquilibriumSpec) -> tuple[str, str] | None:
    from vaft.data.analytic_plasma_state import AnalyticPlasmaState, AnalyticProfile

    pressure, current = spec.pressure_profile, spec.current_profile
    if pressure is not None:
        if not isinstance(pressure, (AnalyticProfile, AnalyticPlasmaState)):
            return ("invalid_source_model", "pressure_profile must be an AnalyticProfile or an "
                    f"AnalyticPlasmaState, got {type(pressure).__name__}")
        if spec.pressure_fraction == 0:
            return ("invalid_source_model", "a pressure_profile with pressure_fraction = 0 would be "
                    "discarded; give a positive pressure_fraction or drop the profile")
        if (spec.pressure_alpha, spec.pressure_beta) != _DEFAULT_PRESSURE_EXPONENTS:
            return ("invalid_source_model", "both a generalized-parabolic pressure shape (pressure_alpha, "
                    "pressure_beta) and a pressure_profile were given; the profile replaces the kernel, "
                    "so give one or the other")
        problem = _monotone_drop("pressure_profile", pressure, "the CHEASE input carries p' of one sign "
                                 "only, so the pressure must not increase outward")
        if problem is not None:
            return problem
    if current is not None:
        if not isinstance(current, AnalyticProfile):
            return ("invalid_source_model", "current_profile must be an AnalyticProfile, got "
                    f"{type(current).__name__}")
        if (spec.current_alpha, spec.current_beta) != _DEFAULT_CURRENT_EXPONENTS:
            return ("invalid_source_model", "both a generalized-parabolic current shape (current_alpha, "
                    "current_beta) and a current_profile were given; the profile replaces the kernel, "
                    "so give one or the other")
        problem = _monotone_drop("current_profile", current, "FF' = -dg/dpsi_N would change sign, a "
                                 "reversed current the CHEASE input passes through unchecked")
        if problem is not None:
            return problem
    return None


def _monotone_drop(name: str, profile: Any, why: str) -> tuple[str, str] | None:
    """Refuse a profile that rises outward anywhere, or whose net drop is too small to normalize by."""
    drive = -np.asarray(_profile_value(profile, _PROFILE_CHECK_GRID, derivative=True))
    ends = np.asarray(_profile_value(profile, np.array([0.0, 1.0])))
    if not (np.all(np.isfinite(drive)) and np.all(np.isfinite(ends))) or not drive.max() > 0:
        return "invalid_source_model", f"{name} does not fall outward anywhere, so it drives no source"
    if drive.min() < -1e-9*drive.max():
        where = float(_PROFILE_CHECK_GRID[np.argmin(drive)])
        return "invalid_source_model", f"{name} rises outward near psi_N = {where:.3f}; {why}"
    drop, level = float(ends[0] - ends[1]), float(np.max(np.abs(ends)))
    if drop < _MIN_RELATIVE_DROP*level:
        return ("invalid_source_model", f"{name} falls by only {drop:.3g} from axis to separatrix, below "
                f"{_MIN_RELATIVE_DROP:g} of its level {level:.3g}; its shape -g'/drop would be conditioned "
                "by the 1/drop amplification rather than by the profile")
    return None


def _profile_value(profile: Any, psi_n: np.ndarray, *, derivative: bool = False) -> np.ndarray:
    """A barrier profile, or its ``psi_N`` derivative; ``p_total`` for a plasma state."""
    from vaft.data.analytic_plasma_state import AnalyticPlasmaState
    from vaft.process.profile import evaluate_analytic_profile, evaluate_plasma_state

    if isinstance(profile, AnalyticPlasmaState):
        return np.asarray(evaluate_plasma_state(
            profile, psi_n, "dp_total_dpsi_norm" if derivative else "p_total"), dtype=float)
    return np.asarray(evaluate_analytic_profile(profile, psi_n, derivative=derivative), dtype=float)


def _has_barriers(profile: Any) -> bool:
    """Does a pressure profile (or any channel of a plasma state) carry a pedestal or ITB step?"""
    channels = profile.profiles.values() if hasattr(profile, "profiles") else (profile,)
    return any(channel.barriers for channel in channels)


def _uses_profiles(spec: ZeroDimensionalEquilibriumSpec) -> bool:
    return spec.pressure_profile is not None or spec.current_profile is not None


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

    if not _uses_profiles(spec):
        pp = spec.pressure_fraction*generalized_parabolic_profile(psi_n, alpha=spec.pressure_alpha, beta=spec.pressure_beta)
        ff = (1 - spec.pressure_fraction)*generalized_parabolic_profile(psi_n, alpha=spec.current_alpha, beta=spec.current_beta)
        return pp, ff
    pp = _unit_shape(spec.pressure_profile, spec.pressure_alpha, spec.pressure_beta, psi_n)
    ff = _unit_shape(spec.current_profile, spec.current_alpha, spec.current_beta, psi_n)
    return spec.pressure_fraction*pp, (1 - spec.pressure_fraction)*ff


def _unit_shape(profile: Any, alpha: float, beta: float, psi_n: np.ndarray) -> np.ndarray:
    """A source shape of unit integral over ``psi_N``: ``-g'/(g(0) - g(1))`` or the scaled kernel."""
    from scipy.special import beta as beta_function

    from vaft.formula.equilibrium import generalized_parabolic_profile

    if profile is None:
        # int_0^1 (1 - x**a)**b dx = B(1/a, b + 1)/a.
        integral = beta_function(1.0/alpha, beta + 1.0)/alpha
        return generalized_parabolic_profile(psi_n, alpha=alpha, beta=beta)/integral
    ends = _profile_value(profile, np.array([0.0, 1.0]))
    return -_profile_value(profile, psi_n, derivative=True)/float(ends[0] - ends[1])


def requested_pressure_shape(spec: ZeroDimensionalEquilibriumSpec, psi_n) -> tuple[np.ndarray, np.ndarray]:
    """The pressure a specification asks for, normalized: ``(p - p(1))/(p(0) - p(1))`` and its ``psi_N`` gradient.

    The amplitude of ``p`` is not an input -- CHEASE sets it -- so this is the
    requested pressure *shape*, one on axis and zero at the boundary, to be
    compared with the achieved one in
    :attr:`SyntheticEquilibriumResult.achieved_profiles`.

    Parameters
    ----------
    spec : ZeroDimensionalEquilibriumSpec
        The request [-].
    psi_n : array_like
        Normalized poloidal flux in ``[0, 1]`` [-].

    Returns
    -------
    tuple of np.ndarray
        The normalized pressure and ``d/dpsi_N`` of it (negative for a
        pressure falling outward), on *psi_n* [-].
    """
    x = np.asarray(psi_n, dtype=float)
    if spec.pressure_profile is not None:
        ends = _profile_value(spec.pressure_profile, np.array([0.0, 1.0]))
        drop = float(ends[0] - ends[1])
        return ((_profile_value(spec.pressure_profile, x) - ends[1])/drop,
                _profile_value(spec.pressure_profile, x, derivative=True)/drop)
    from scipy.integrate import cumulative_trapezoid

    fine = np.linspace(0.0, 1.0, 20001)
    shape = _unit_shape(None, spec.pressure_alpha, spec.pressure_beta, fine)
    outward = cumulative_trapezoid(shape[::-1], dx=fine[1], initial=0.0)[::-1]   # int_x^1 shape
    return np.interp(x, fine, outward/outward[0]), -_unit_shape(None, spec.pressure_alpha, spec.pressure_beta, x)


def build_input_equilibrium(spec: ZeroDimensionalEquilibriumSpec, *, q95: float | None = None, resolution: int = 129):
    """A COCOS 11 ``EquilibriumData`` carrying the boundary and source shapes for CHEASE.

    Only the boundary, ``p'``, ``FF'``, ``F`` at the edge, the current and --
    for a q95 target -- ``q(0.95)`` reach CHEASE; the flux map is a smooth
    placeholder of the right sign and span, since CHEASE solves its own.
    With a barrier-profile source, ``p'`` and ``FF'`` are that profile's
    analytic ``psi_N`` derivative sampled on the ``resolution`` points, so
    choose enough points to resolve its narrowest layer.
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
    boundary_tolerance: float = 0.02, pressure_shape_tolerance: float = 0.02,
    pprime_peak_tolerance: float = 0.1, pprime_peak_location_tolerance: float = 0.01,
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
    pressure_shape_tolerance : float, optional
        Largest miss of the achieved normalized pressure
        ``(p - p(1))/(p(0) - p(1))`` against the requested one, anywhere in
        ``psi_N``; enforced only when the spec carries a ``pressure_profile``,
        whose shape is then part of the request.  The miss is always reported
        as ``residuals["pressure_shape"]`` [-].
    pprime_peak_tolerance : float, optional
        Largest relative miss of the steepest achieved ``|p'|`` against the
        steepest requested one (both normalized as above), reported as
        ``residuals["pprime_peak"]``; enforced when the ``pressure_profile``
        has barrier steps, whose gradient the pressure-shape residual cannot
        see [-].
    pprime_peak_location_tolerance : float, optional
        Largest offset in ``psi_N`` of that steepest achieved gradient from the
        requested one, reported as ``residuals["pprime_peak_location"]`` and
        enforced with *pprime_peak_tolerance* [-].

    Returns
    -------
    SyntheticEquilibriumResult
        Status and reason (every problem found, the first one's state), the
        request, the achieved descriptors -- magnitudes, in CHEASE's
        orientation, with the refined output's signs checked against the
        request -- their residuals, the constructed boundary, the source
        shapes and the requested normalized pressure (``pressure_norm``,
        ``pprime_norm``), the achieved profiles on the solved ``psi_N``
        (``pressure_norm``, ``pprime_norm``, ``|q|``), the CHEASE run, and the
        refined g-file and ODS.  Without a *config*, CHEASE runs in a fresh
        temporary directory [-].
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
        equilibrium = build_input_equilibrium(
            spec, q95=q95, **({"resolution": _PROFILE_RESOLUTION} if _uses_profiles(spec) else {}))
    except ValueError as error:
        return SyntheticEquilibriumResult("invalid_source_model", str(error), spec, target, requested)
    from vaft.data.eqdsk import from_equilibrium, read_geqdsk
    from vaft.process.equilibrium import as_equilibrium, contour_shape_parameters, derive_global_descriptors

    psi_n = np.linspace(0.0, 1.0, 101)
    pp_shape, ff_shape = _source_shapes(spec, psi_n)
    pressure_norm, pprime_norm = requested_pressure_shape(spec, psi_n)
    settings = dict(vars(config)) if config is not None else {}
    if config is None or Path(config.workdir) == Path("."):
        import tempfile

        # CHEASE writes a dozen files; never into the caller's working directory by default.
        settings["workdir"] = Path(tempfile.mkdtemp(prefix="vaft-0d-chease-"))
    settings.update(target_psin=1.0, ncscal=SUPPORTED_TARGETS[target],
                    q_constraint_psi_norm=0.95 if target == "q95" else None,
                    edge_zero=False, preserve_boundary_limiter=False)
    try:
        run = refine_equilibrium(from_equilibrium(equilibrium), CHEASEConfig(**settings))
    except FileNotFoundError:
        from .chease import find_chease_executable

        if find_chease_executable(CHEASEConfig(**settings)) is not None:
            raise
        return SyntheticEquilibriumResult(
            "non_converged", "no CHEASE executable (set CHEASE, CHEASEHOME or CHEASE_EXEC_DIR); nothing was solved",
            spec, target, requested, boundary=equilibrium.lcfs)
    result = SyntheticEquilibriumResult(
        "success", None, spec, target, requested, boundary=equilibrium.lcfs,
        source_profiles={"psi_n": psi_n, "pprime_shape": pp_shape, "ffprime_shape": ff_shape,
                         "pressure_norm": pressure_norm, "pprime_norm": pprime_norm},
        chease=run, refined_geqdsk=run.refined_geqdsk, refined_ods=run.refined_ods,
    )
    if run.timed_out:
        # Stopped by a time limit (#1016): not a solve that failed to converge.
        reason = (run.stderr or "").strip().splitlines()[-1:] or [run.runtime_status]
        result.status, result.reason = "timeout", reason[0]
        return result
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
    profiles = _achieved_profiles(solved)
    if profiles:
        result.achieved_profiles = profiles
        residuals["pressure_shape"] = float(np.max(np.abs(
            np.interp(psi_n, profiles["psi_n"], profiles["pressure_norm"]) - pressure_norm)))
        _, fine_pprime = requested_pressure_shape(spec, _PEAK_GRID)
        want, got = int(np.argmin(fine_pprime)), int(np.argmin(profiles["pprime_norm"]))
        residuals["pprime_peak"] = float(profiles["pprime_norm"][got]/fine_pprime[want] - 1)
        residuals["pprime_peak_location"] = float(profiles["psi_n"][got] - _PEAK_GRID[want])
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
    if spec.pressure_profile is not None and not profiles:
        problems.append(("invalid_equilibrium", "the achieved pressure profile could not be read from the solved "
                         "equilibrium (missing arrays, or no pressure drop), so the requested pressure_profile "
                         "cannot be checked"))
    elif spec.pressure_profile is not None:
        if not residuals["pressure_shape"] <= pressure_shape_tolerance:
            problems.append(("constraint_not_reached", "the achieved normalized pressure departs from the "
                             f"requested profile by {residuals['pressure_shape']:.3g}"))
        if _has_barriers(spec.pressure_profile) and not (
                abs(residuals["pprime_peak"]) <= pprime_peak_tolerance
                and abs(residuals["pprime_peak_location"]) <= pprime_peak_location_tolerance):
            problems.append(("constraint_not_reached", "the barrier's steepest p' is not reproduced: peak off by "
                             f"{residuals['pprime_peak']:.3g} (relative), at psi_N offset "
                             f"{residuals['pprime_peak_location']:.3g}; the solve does not resolve the barrier"))
    if problems:
        result.status = problems[0][0]
        result.reason = "; ".join(reason for _, reason in problems)
    return result


def _achieved_profiles(solved: Any) -> dict[str, np.ndarray]:
    """Normalized pressure, its gradient and |q| on the solved ``psi_N``; empty when unreadable."""
    try:
        psi = np.asarray(solved.psi_1d, dtype=float)
        pressure = np.asarray(solved.pressure, dtype=float)
        pprime = np.asarray(solved.pprime, dtype=float)
        q = np.abs(np.asarray(solved.q, dtype=float))
    except (AttributeError, TypeError):
        return {}
    span = float(solved.psi_boundary - solved.psi_axis)
    drop = float(pressure[0] - pressure[-1])
    if psi.size < 2 or span == 0 or drop == 0:
        return {}
    return {"psi_n": (psi - solved.psi_axis)/span, "pressure_norm": (pressure - pressure[-1])/drop,
            "pprime_norm": pprime*span/drop, "q": q}


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
    "build_input_equilibrium", "construct_boundary", "requested_pressure_shape", "synthesize_equilibrium_from_0d",
]
