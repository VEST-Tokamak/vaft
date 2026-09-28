"""Synthetic kinetic profiles from a magnetic equilibrium, fidelity Levels 0-3 (#122).

``equilibrium + assumptions -> n_e, T_e, T_i, n_i, n_impurity`` as one
deterministic transformation.  The equilibrium supplies the coordinate, the
pressure ``p_eq(psi)`` and the geometry the averages need; the assumptions
are the records of :mod:`vaft.data.synthetic_kinetic_profiles`.  The profile
kernels are the #1045 composition of the #552 kernels, reused unchanged, and
the #1045 presets pass through :func:`spec_from_plasma_state` as special
cases.  Nothing here is fitted, measured or transport-predicted, and nothing
changes the equilibrium.  :mod:`vaft.process.profile` is the public import
location.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

from vaft.data.analytic_plasma_state import AnalyticPlasmaState, AnalyticProfile
from vaft.data.synthetic_kinetic_profiles import (
    ION_SPECIES,
    Composition,
    GradientProfile,
    IonSpecies,
    PressureClosureReport,
    ProfileSpec,
    ScalarTarget,
    SqrtPressureSplit,
    SyntheticKineticProfiles,
    SyntheticKineticSpec,
    SyntheticProfileError,
    TabulatedProfile,
    TargetResidual,
    TemperatureAssumption,
)
from vaft.formula.atomic import impurity_fraction_from_effective_charge
from vaft.formula.constants import QE

from ._analytic_plasma_state import compose_analytic_profile, evaluate_analytic_profile

__all__ = [
    "generate_synthetic_kinetic_profiles",
    "spec_from_plasma_state",
    "write_synthetic_core_profiles",
]

#: Default output grid: 101 points uniform in rho_pol_norm = sqrt(psi_norm).
_DEFAULT_POINTS = 101

#: Samples along the line-average chord.
_CHORD_POINTS = 801

#: Range of the core peaking exponent searched when a peaking factor is requested.
_BETA_RANGE = (1.0, 40.0)

#: Tolerances of the reported constraints, relative.
_TARGET_RTOL = 1e-9
_PEAKING_RTOL = 1e-8
_QN_RTOL = 1e-12
_ZEFF_ATOL = 1e-10


def _fail(status: str, message: str):
    raise SyntheticProfileError(status, message)


# --- equilibrium geometry -------------------------------------------------------


def _check_equilibrium(eq, needs_pressure: bool) -> None:
    for name in ("r", "z", "psi", "psi_1d"):
        if getattr(eq, name) is None:
            _fail("invalid_equilibrium", f"the equilibrium carries no {name}")
    a, b = eq.psi_axis, eq.psi_boundary
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) or a == b:
        _fail("invalid_equilibrium", "psi_axis and psi_boundary must be finite and distinct")
    if eq.lcfs is None or np.asarray(eq.lcfs.r).size < 3:
        _fail("invalid_equilibrium", "an LCFS outline is required for the volume and line averages")
    if eq.magnetic_axis is None:
        _fail("invalid_equilibrium", "a magnetic axis is required for the line-average chord")
    if needs_pressure:
        p = eq.pressure
        if p is None or p.size != eq.psi_1d.size or not np.all(np.isfinite(p)):
            _fail("invalid_equilibrium", "a finite pressure profile on psi_1d is required by this constraint")
        if p.min() < -1e-9 * max(p.max(), 1e-300) or p.max() <= 0.0:
            _fail("invalid_equilibrium",
                  f"the equilibrium pressure must be non-negative with a positive maximum (min {p.min():.4g} Pa)")


def _psi_cocos11(eq, grid):
    """Flux on the grid in Wb, COCOS 11, when every surviving convention candidate agrees."""
    from vaft.process._equilibrium_parametric import convert_cocos

    conv = eq.convention
    candidates = (conv.cocos,) if conv.cocos is not None else tuple(conv.candidates)
    if not candidates:
        return None, "the source COCOS is unknown, so psi cannot be stated in Wb, COCOS 11"
    ends = []
    for index in candidates:
        try:
            converted = convert_cocos(dataclasses.replace(eq, convention=dataclasses.replace(conv, cocos=index)), 11)
        except Exception as exc:  # noqa: BLE001 - an unconvertible candidate is reported, not fatal
            return None, f"COCOS {index} -> 11 failed: {exc}"
        ends.append((float(converted.psi_axis), float(converted.psi_boundary)))
    first = np.array(ends[0])
    if not all(np.allclose(np.array(e), first, rtol=1e-12, atol=0.0) for e in ends[1:]):
        return None, f"COCOS candidates {candidates} disagree on psi in COCOS 11; declare the source convention"
    note = (f"COCOS {candidates[0]} -> 11" if len(candidates) == 1
            else f"COCOS candidates {candidates} -> 11, identical in psi")
    return first[0] + grid * (first[1] - first[0]), note


def _chord(eq):
    """The horizontal chord through the magnetic axis between its two LCFS crossings."""
    r_axis, z_axis = (float(v) for v in eq.magnetic_axis)
    r, z = np.asarray(eq.lcfs.r, float), np.asarray(eq.lcfs.z, float)
    if r[0] != r[-1] or z[0] != z[-1]:
        r, z = np.r_[r, r[0]], np.r_[z, z[0]]
    crossings = []
    for i in range(r.size - 1):
        z0, z1 = z[i] - z_axis, z[i + 1] - z_axis
        if z0 == 0.0:
            crossings.append(r[i])
        elif z0 * z1 < 0.0:
            crossings.append(r[i] + (r[i + 1] - r[i]) * z0 / (z0 - z1))
    inner = [c for c in crossings if c < r_axis]
    outer = [c for c in crossings if c > r_axis]
    if not inner or not outer:
        _fail("coordinate_mapping_failed", "the horizontal chord through the axis does not cross the LCFS twice")
    return max(inner), min(outer), z_axis


def _geometry(eq, grid) -> dict[str, Any]:
    from scipy.interpolate import RectBivariateSpline

    from vaft.process._equilibrium_parametric import derive_radial_coordinates
    from vaft.process.equilibrium import flux_surface_quantities

    span = eq.psi_boundary - eq.psi_axis
    psi_n_1d = (eq.psi_1d - eq.psi_axis) / span
    order = np.argsort(psi_n_1d)
    x1d = psi_n_1d[order]
    if np.any(np.diff(x1d) <= 0.0):
        _fail("coordinate_mapping_failed", "the equilibrium's normalized flux is not strictly monotonic")
    geom: dict[str, Any] = {}
    p_eq = np.full(grid.shape, np.nan)
    if eq.pressure is not None and eq.pressure.size == x1d.size:
        p_eq = np.interp(grid, x1d, eq.pressure[order])
    geom["p_eq"] = p_eq

    coordinates = derive_radial_coordinates(eq)
    rho_tor = coordinates["rho_tor_n"]
    if rho_tor.value is None:
        geom["rho_tor_norm"], geom["rho_tor_note"] = None, f"unavailable: {rho_tor.reason}"
    else:
        geom["rho_tor_norm"] = np.interp(grid, x1d, np.asarray(rho_tor.value, float)[order])
        geom["rho_tor_note"] = "sqrt(Phi/Phi_boundary), Phi = int q dpsi, from the equilibrium's q"

    surfaces = flux_surface_quantities(eq.psi, eq.r, eq.z, eq.psi_axis, eq.psi_boundary, grid,
                                       axis_rz=tuple(eq.magnetic_axis),
                                       boundary=(np.asarray(eq.lcfs.r), np.asarray(eq.lcfs.z)))
    volume = np.asarray(surfaces["volume"], float)
    if not np.all(np.isfinite(volume)) or np.any(np.diff(volume) <= 0.0) or volume[-1] <= 0.0:
        _fail("coordinate_mapping_failed", "the traced flux-surface volume is not a positive increasing profile")
    geom["volume"] = volume

    r_in, r_out, z_axis = _chord(eq)
    r_chord = np.linspace(r_in, r_out, _CHORD_POINTS)
    psi_chord = RectBivariateSpline(eq.r, eq.z, eq.psi)(r_chord, np.full_like(r_chord, z_axis), grid=False)
    geom["chord_psi_norm"] = np.clip((psi_chord - eq.psi_axis) / span, 0.0, 1.0)
    geom["chord_r"] = r_chord
    geom["chord"] = (float(r_in), float(r_out), float(z_axis))
    geom["minor_radius"] = 0.5 * float(np.ptp(np.asarray(eq.lcfs.r, float)))
    geom["ip"] = None if eq.ip is None else float(eq.ip)
    geom["psi"], geom["psi_note"] = _psi_cocos11(eq, grid)
    return geom


def _volume_integral(values, volume) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(volume)))


def _measure(kind: str, values: np.ndarray, grid, geom, channel: str) -> float:
    if kind == "axis":
        return float(values[0])
    if kind == "separatrix":
        return float(values[-1])
    if kind == "volume_average":
        return _volume_integral(values, geom["volume"]) / float(geom["volume"][-1])
    line = float(np.trapezoid(np.interp(geom["chord_psi_norm"], grid, values), geom["chord_r"])
                 / (geom["chord_r"][-1] - geom["chord_r"][0]))
    if kind == "line_average":
        return line
    if channel != "n_e":
        _fail("invalid_normalization", f"a Greenwald fraction normalizes n_e, not {channel}")
    if geom["ip"] is None or geom["ip"] == 0.0:
        _fail("invalid_normalization", "a Greenwald fraction needs the plasma current, which the equilibrium lacks")
    from vaft.formula.stability import greenwald_density

    n_g = 1e19 * greenwald_density(abs(geom["ip"]) / 1e6, geom["minor_radius"])
    return line / n_g


def _definition(kind: str, geom) -> str:
    r_in, r_out, z = geom["chord"]
    chord = f"int f dR / (R_out - R_in) on Z = {z:.4g} m, R in [{r_in:.4g}, {r_out:.4g}] m (axis chord)"
    return {
        "axis": "f at psi_norm = 0",
        "separatrix": "f at psi_norm = 1",
        "line_average": chord,
        "volume_average": "int f dV / V, V(psi) the traced flux-surface volume",
        "greenwald_fraction": (f"line average ({chord}) / n_G, n_G = I_p/(pi a^2) [1e20 m^-3, MA, m], "
                               f"a = {geom['minor_radius']:.4g} m half the LCFS radial extent"),
        "peaking_factor": "f(psi_norm = 0) / volume average",
    }[kind]


# --- channels --------------------------------------------------------------------


def _coordinate(name: str, grid, geom):
    if name == "psi_norm":
        return grid
    if name == "rho_pol_norm":
        return np.sqrt(grid)
    if geom["rho_tor_norm"] is None:
        _fail("coordinate_mapping_failed", f"rho_tor_norm is {geom['rho_tor_note']}; declare another coordinate")
    return geom["rho_tor_norm"]


def _evaluate(shape, grid, geom) -> np.ndarray:
    if isinstance(shape, AnalyticProfile):
        return np.asarray(evaluate_analytic_profile(shape, grid), dtype=float)
    x = _coordinate(shape.coordinate, grid, geom)
    if isinstance(shape, GradientProfile):
        integral = shape.integral(x)
        return shape.boundary_value * np.exp(float(shape.integral(shape.boundary_position)) - integral)
    lo, hi = shape.x[0], shape.x[-1]
    if shape.extrapolation == "refuse" and (x.min() < lo - 1e-9 or x.max() > hi + 1e-9):
        _fail("coordinate_mapping_failed",
              f"the tabulated profile spans {shape.coordinate} [{lo:.4g}, {hi:.4g}] but the grid needs "
              f"[{x.min():.4g}, {x.max():.4g}]; extend it or declare extrapolation='hold'")
    xc = np.clip(x, lo, hi)
    if shape.interpolation == "linear":
        return np.interp(xc, shape.x, shape.values)
    from scipy.interpolate import PchipInterpolator

    return np.asarray(PchipInterpolator(shape.x, shape.values)(xc), dtype=float)


def _recompose(profile: AnalyticProfile, *, scale: float = 1.0, core_beta: float | None = None) -> AnalyticProfile:
    """The same #1045 composition with every value scaled and, optionally, a new core exponent."""
    kwargs = dict(axis_value=scale * profile.axis_value, separatrix_value=scale * profile.separatrix_value,
                  core_alpha=profile.core_alpha,
                  core_beta=profile.core_beta if core_beta is None else core_beta, unit=profile.unit)
    if profile.pedestal is not None:
        kwargs.update(pedestal_top_value=scale * profile.pedestal_top_value,
                      pedestal_position=profile.pedestal.position, pedestal_width=profile.pedestal.width)
    if profile.itb is not None:
        kwargs.update(itb_height=scale * profile.itb.height, itb_position=profile.itb.position,
                      itb_width=profile.itb.width)
    return compose_analytic_profile(profile.quantity, **kwargs)


def _resolve_channel(name: str, spec: ProfileSpec, grid, geom, targets: list, notes: dict) -> np.ndarray:
    shape = spec.shape
    resolved: dict[str, Any] = {"shape": type(shape).__name__}
    if spec.peaking_factor is not None:
        from scipy.optimize import brentq

        def peaking(beta):
            values = _evaluate(_recompose(shape, core_beta=beta), grid, geom)
            return values[0] / _measure("volume_average", values, grid, geom, name)

        def miss(beta):
            try:
                return peaking(beta) - spec.peaking_factor
            except ValueError:
                return np.nan

        ends = [miss(b) for b in _BETA_RANGE]
        if np.all(np.isfinite(ends)) and ends[0] * ends[1] < 0.0:
            beta = brentq(miss, *_BETA_RANGE, xtol=1e-12, rtol=1e-12)
            status = "brentq"
        else:
            finite = [(abs(m), b) for m, b in zip(ends, _BETA_RANGE) if np.isfinite(m)]
            beta = min(finite)[1] if finite else shape.core_beta
            status = "not_reached"
        shape = _recompose(shape, core_beta=beta)
        resolved["core_beta"] = float(beta)
        targets.append(("peaking_factor", name, spec.peaking_factor, status))
    values = _evaluate(shape, grid, geom)
    scale = 1.0
    if spec.target is not None:
        current = _measure(spec.target.kind, values, grid, geom, name)
        if not np.isfinite(current) or current <= 0.0:
            _fail("invalid_normalization",
                  f"{name} has {spec.target.kind} {current!r} before scaling; it cannot be scaled to a positive target")
        scale = spec.target.value / current
        values = values * scale
        targets.append((spec.target.kind, name, spec.target.value, "linear"))
    resolved["scale"] = float(scale)
    if isinstance(shape, AnalyticProfile):
        resolved["profile"] = _recompose(shape, scale=scale) if scale != 1.0 else shape
    elif isinstance(shape, GradientProfile):
        resolved["profile"] = dataclasses.replace(shape, boundary_value=shape.boundary_value * scale)
    else:
        resolved["profile"] = shape
    notes[name] = resolved
    return values


def _level(spec: ProfileSpec) -> int:
    if isinstance(spec.shape, GradientProfile):
        return 3
    return 2 if (spec.target is not None or spec.peaking_factor is not None) else 1


def _sqrt_pressure_split(P, *, te_axis=None, ne_over_te=None, k=2.0):
    """Level 0: ``n_e`` and ``T_e`` in the shape ``sqrt(P/P(0))`` with ``P = k e n_e T_e``.

    The operations and their order are those of the legacy helpers, so with
    ``k = 2`` the result is bit-identical to
    :func:`vaft.process.profile.core_profiles_from_eq` (``te_axis``) and
    :func:`~vaft.process.profile.core_profiles_from_eq_ratio` (``ne_over_te``).
    """
    e_J = 1.602176634e-19
    if te_axis is not None:
        g = np.sqrt(np.clip(P / P[0], 0.0, None))
        Te = te_axis * g
        ne0 = P[0] / (k * te_axis * e_J)
        return ne0 * g, Te
    f = P / P[0]
    Te = np.sqrt(f)
    ne = ne_over_te * Te
    scale = P[0] / (k * ne_over_te * e_J)
    Te *= np.sqrt(scale)
    ne *= np.sqrt(scale)
    return ne, Te


def _ratio(temperature: TemperatureAssumption, grid, geom) -> np.ndarray:
    value = temperature.ti_over_te
    if isinstance(value, TabulatedProfile):
        return _evaluate(value, grid, geom)
    return np.full(grid.shape, float(value))


def _safe_divide(numerator, denominator):
    out = np.full(np.shape(numerator), np.nan)
    positive = denominator > 0.0
    np.divide(numerator, denominator, out=out, where=positive)
    out[~positive & (numerator == 0.0)] = 0.0
    return out


# --- the generator ----------------------------------------------------------------


_HELD_SOLVED = {
    ("equilibrium", "temperature"): (("n_e", "composition", "T_i route"), ("T_e(psi)", "T_i")),
    ("equilibrium", "density"): (("T_e", "composition", "T_i route"), ("n_e(psi)", "n_i", "n_impurity")),
    ("equilibrium", "sqrt_split"): (("shape sqrt(p/p(0)) for n_e and T_e", "Level 0 amplitude", "composition",
                                     "T_i route"), ("n_e", "T_e")),
    ("thermal_energy", "temperature_amplitude"): (("n_e", "T_e shape", "T_i shape or route", "composition"),
                                                  ("one amplitude of T_e and T_i",)),
    ("thermal_energy", "density_amplitude"): (("n_e shape", "T_e", "T_i route", "composition"),
                                              ("one amplitude of n_e",)),
    ("kinetic", None): (("n_e", "T_e", "T_i route", "composition"), ()),
}


def _check_spec(spec: SyntheticKineticSpec) -> None:
    mode, closure, t = spec.pressure_constraint, spec.closure, spec.temperature
    if (closure == "sqrt_split") != (spec.sqrt_split is not None):
        _fail("invalid_normalization", "sqrt_split amplitudes belong to closure='sqrt_split' and it needs them")
    need = {"kinetic": ("n_e", "T_e"), "thermal_energy": ("n_e", "T_e")}.get(mode, ())
    if mode == "equilibrium":
        need = {"temperature": ("n_e",), "density": ("T_e",), "sqrt_split": ()}[closure]
        solved = {"temperature": ("T_e",), "density": ("n_e",), "sqrt_split": ("n_e", "T_e")}[closure]
        for name in solved:
            if getattr(spec, name) is not None:
                _fail("invalid_profile_model",
                      f"closure={closure!r} solves {name} so that p_kin = p_eq; a prescribed {name} "
                      "contradicts it -- drop it, or keep it with pressure_constraint='kinetic' and read "
                      "the reported mismatch")
        if closure == "sqrt_split" and (t.route == "profile" or isinstance(t.ti_over_te, TabulatedProfile)):
            _fail("invalid_profile_model",
                  "the Level 0 sqrt split needs a constant p/(e n_e T_e): a scalar ti_over_te or an "
                  "electron_pressure_fraction, not a T_i profile")
    for name in need:
        if getattr(spec, name) is None:
            _fail("invalid_profile_model", f"pressure_constraint={mode!r}, closure={closure!r} needs a {name} spec")
    if closure == "temperature_amplitude":
        for name, channel in (("T_e", spec.T_e), ("T_i", t.T_i)):
            if channel is not None and channel.target is not None:
                _fail("invalid_normalization",
                      f"the thermal-energy constraint scales {name}; a {channel.target.kind} target on it "
                      "fixes the same amplitude twice")
    if closure == "density_amplitude" and spec.n_e.target is not None:
        _fail("invalid_normalization",
              f"the thermal-energy constraint scales n_e; a {spec.n_e.target.kind} target fixes it twice")


def generate_synthetic_kinetic_profiles(
    equilibrium,
    spec: SyntheticKineticSpec,
    *,
    time: float | None = None,
    time_index: int = 0,
    pressure_floor: float = 1e-3,
    pressure_tolerance: float = 1e-6,
) -> SyntheticKineticProfiles:
    r"""Generate ``n_e``, ``T_e``, ``T_i`` and the ion densities from an equilibrium and declared assumptions.

    Parameters
    ----------
    equilibrium : GEQDSK, ODS, IMAS IDS, EquilibriumData or path
        The magnetic equilibrium, adapted through
        :func:`vaft.process.equilibrium.as_equilibrium`; read, never changed [-].
    spec : SyntheticKineticSpec
        Shapes, normalizations, temperature route, composition and pressure
        constraint; see :class:`vaft.data.SyntheticKineticSpec` [-].
    time : float, optional
        Time recorded on the result; the equilibrium's own time when omitted [s].
    time_index : int, optional
        Time slice of a multi-slice source [-].
    pressure_floor : float, optional
        Fraction of ``max(p_eq)`` above which the pressure residual is judged [-].
    pressure_tolerance : float, optional
        Largest relative local pressure residual a locally closed state may
        carry [-].

    Returns
    -------
    SyntheticKineticProfiles
        The profiles on the equilibrium's ``psi_norm`` grid with ``rho_pol_norm``,
        ``rho_tor_norm``, ``psi`` [Wb] and the enclosed volume; the species; the
        pressure-closure report; every requested target beside its recomputed
        value; validation, resolved parameters, provenance and a status [-].

    Raises
    ------
    SyntheticProfileError
        A refused input, with ``status`` ``invalid_equilibrium``,
        ``invalid_profile_model``, ``invalid_gradient_model``,
        ``invalid_normalization``, ``invalid_composition`` or
        ``coordinate_mapping_failed``: a missing equilibrium quantity, a
        channel the closure solves also prescribed, a target fixing an
        amplitude twice, a Greenwald fraction on a temperature, a tabulated
        profile that does not cover the grid, an unavailable coordinate.

    Processing steps
    ----------------
    1. Adapt the equilibrium and build the grid (101 points uniform in
       ``rho_pol_norm`` unless the spec gives ``psi_norm``); map ``p_eq``,
       ``rho_tor_norm`` (from ``q``) and ``psi`` (Wb, COCOS 11) onto it; trace
       the flux-surface volume; lay the axis chord.
    2. Resolve each prescribed channel: shape (analytic #1045 composition,
       tabulated, or integrated ``a/L``), then a peaking factor through the
       core exponent, then one amplitude target by linear scaling.
    3. Apply the pressure constraint: solve the closure's variable locally
       (``equilibrium``), one amplitude (``thermal_energy``), or nothing
       (``kinetic``).
    4. Build the ion densities from quasi-neutrality and ``Z_eff``, the
       pressures from ``p = e n T``.
    5. Recompute every target, ``Z_eff``, quasi-neutrality and the pressure
       residual from the final arrays; set the status.

    Input semantics
    ---------------
    Reconstructed or analytic: a magnetic equilibrium, and assumptions.

    Output semantics
    ----------------
    Synthetic: assumption-driven kinetic profiles consistent with the
    requested constraints, never measured, fitted or transport-predicted.

    Convention
    ----------
    The grid is the source equilibrium's normalized poloidal flux
    ``psi_norm = (psi - psi_axis)/(psi_boundary - psi_axis)``, independent of
    COCOS and of Wb against Wb/rad; ``rho_pol_norm = sqrt(psi_norm)``;
    ``rho_tor_norm`` is derived from ``q`` and never assumed equal to
    ``rho_pol_norm``.  ``psi`` is reported in Wb, COCOS 11 (the IMAS DD and
    :data:`vaft.data.cocos.VAFT_INTERNAL_COCOS`), only when every
    convention candidate of the source gives the same value.  ``p = e n T``
    in Pa with ``T`` in eV; the main ion is hydrogenic and every ion is at
    ``T_i``.  The line average is along the horizontal chord through the
    magnetic axis; the volume average uses the traced ``V(psi)``.

    Assumptions
    -----------
    Thermal particles only (no fast-ion pressure); uniform local ``Z_eff``;
    one fully stripped impurity charge state.

    Applicability
    -------------
    Machine-independent.  Any equilibrium with a flux map, an LCFS outline, a
    magnetic axis and a 1-D flux grid; ``p_eq`` is needed for the
    ``equilibrium`` and ``thermal_energy`` constraints, ``q`` for
    ``rho_tor_norm``, ``I_p`` for a Greenwald fraction.

    Limitations
    -----------
    Levels 0-3 only: no empirical closure, no pedestal model (EPED) and no
    transport solver.  The equilibrium is not updated to the kinetic pressure
    (#123).  A local pressure closure divides by the held variable, so a held
    profile that vanishes where ``p_eq`` does not fails the closure rather
    than being clipped.

    Provenance
    ----------
    .. [122] VAFT issue #122: the fidelity ladder, the closure modes and the
       residuals reported.
    .. [1045] :func:`vaft.process.profile.compose_analytic_profile` and
       :func:`~vaft.process.profile.evaluate_analytic_profile`, the shape
       kernels reused unchanged.
    .. [W11] J. Wesson, *Tokamaks*, 4th ed. (2011), Ch. 2, quasi-neutrality and
       the effective charge (via
       :func:`vaft.formula.atomic.impurity_fraction_from_effective_charge`).
    .. [G88] M. Greenwald et al., Nucl. Fusion 28, 2199 (1988), through
       :func:`vaft.formula.stability.greenwald_density`.
    """
    from vaft.process._equilibrium_parametric import as_equilibrium

    if not isinstance(spec, SyntheticKineticSpec):
        _fail("invalid_profile_model", "spec must be a SyntheticKineticSpec")
    _check_spec(spec)
    mode, closure = spec.pressure_constraint, spec.closure
    try:
        eq = as_equilibrium(equilibrium, time_index=time_index)
    except SyntheticProfileError:
        raise
    except Exception as exc:  # noqa: BLE001 - any unreadable source is the same refusal
        _fail("invalid_equilibrium", f"cannot adapt the equilibrium: {exc}")
    _check_equilibrium(eq, needs_pressure=mode != "kinetic")
    grid = spec.psi_norm if spec.psi_norm is not None else np.linspace(0.0, 1.0, _DEFAULT_POINTS) ** 2
    geom = _geometry(eq, grid)
    p_eq = geom["p_eq"]

    comp, temp = spec.composition, spec.temperature
    charge = comp.impurity_charge or 0.0
    fraction = impurity_fraction_from_effective_charge(comp.z_eff, charge) if comp.impurity else 0.0
    phi = 1.0 - (charge - 1.0) * fraction if comp.impurity else 1.0  # (n_i + n_I)/n_e

    targets: list = []
    notes: dict[str, Any] = {}
    levels: dict[str, int] = {}
    channel_notes: dict[str, str] = {}

    def channel(name):
        levels[name] = _level(getattr(spec, name) if name != "T_i" else temp.T_i)
        return _resolve_channel(name, getattr(spec, name) if name != "T_i" else temp.T_i, grid, geom,
                                targets, notes)

    def describe(name, prescribed):
        s = getattr(spec, name) if name != "T_i" else temp.T_i
        if not prescribed:
            return "solved locally by the pressure closure p_kin = p_eq"
        text = {AnalyticProfile: "analytic core/pedestal/ITB composition (#1045)",
                TabulatedProfile: f"tabulated in {getattr(s.shape, 'coordinate', '')}",
                GradientProfile: f"integrated prescribed a/L in {getattr(s.shape, 'coordinate', '')} "
                                 "(assumed, not transport-predicted)"}[type(s.shape)]
        if s.peaking_factor is not None:
            text += f", peaking {s.peaking_factor:g}"
        if s.target is not None:
            text += f", scaled to {s.target.kind} = {s.target.value:g}"
        return text

    ratio = _ratio(temp, grid, geom) if temp.route == "ratio" else None
    f_e = temp.electron_pressure_fraction

    def ions_from_te(T_e):
        if temp.route == "ratio":
            return ratio * T_e
        return (1.0 - f_e) / (f_e * phi) * T_e

    pressure_kin_scale = None
    if closure == "sqrt_split":
        if not p_eq[0] > 0.0:
            _fail("invalid_equilibrium", "the axis pressure must be positive to define the sqrt(p/p(0)) shape")
        k = 1.0 + phi * float(ratio[0]) if temp.route == "ratio" else 1.0 / f_e
        n_e, T_e = _sqrt_pressure_split(p_eq, te_axis=spec.sqrt_split.te_axis,
                                        ne_over_te=spec.sqrt_split.ne_over_te, k=k)
        T_i = ions_from_te(T_e)
        levels["n_e"] = levels["T_e"] = 0
        channel_notes["n_e"] = channel_notes["T_e"] = "Level 0: sqrt(p_eq/p_eq(0)) shape"
    else:
        n_e = channel("n_e") if spec.n_e is not None else None
        T_e = channel("T_e") if spec.T_e is not None else None
        T_i = channel("T_i") if temp.route == "profile" else None
        if mode == "equilibrium" and closure == "temperature":
            if temp.route == "ratio":
                T_e = _safe_divide(p_eq, QE * n_e * (1.0 + phi * ratio))
            elif temp.route == "partition":
                T_e = _safe_divide(f_e * p_eq, QE * n_e)
            else:
                T_e = _safe_divide(p_eq / QE - phi * n_e * T_i, n_e)
        elif mode == "equilibrium" and closure == "density":
            if temp.route == "partition":
                n_e = _safe_divide(f_e * p_eq, QE * T_e)
            else:
                T_i_here = ratio * T_e if temp.route == "ratio" else T_i
                n_e = _safe_divide(p_eq, QE * (T_e + phi * T_i_here))
        if T_i is None:
            T_i = ions_from_te(T_e)
        if mode == "thermal_energy":
            p_trial = QE * n_e * (T_e + phi * T_i)
            w_kin, w_eq = _volume_integral(p_trial, geom["volume"]), _volume_integral(p_eq, geom["volume"])
            if not w_kin > 0.0:
                _fail("invalid_profile_model", "the prescribed profiles carry no thermal energy to scale")
            pressure_kin_scale = w_eq / w_kin
            if closure == "temperature_amplitude":
                T_e, T_i = T_e * pressure_kin_scale, T_i * pressure_kin_scale
            else:
                n_e = n_e * pressure_kin_scale
    for name in ("n_e", "T_e", "T_i"):
        channel_notes.setdefault(name, describe(name, name in notes))
    if temp.route == "ratio":
        channel_notes["T_i"] = (f"T_i = (T_i/T_e) T_e, ratio {'tabulated' if isinstance(temp.ti_over_te, TabulatedProfile) else f'{float(ratio[0]):g}'}"
                                f" ({temp.source})")
        levels["T_i"] = 2
    elif temp.route == "partition":
        channel_notes["T_i"] = f"from p_e/(p_e + p_i) = {f_e:g} held locally ({temp.source})"
        levels["T_i"] = 2
    if pressure_kin_scale is not None:
        target_name = "T_e" if closure == "temperature_amplitude" else "n_e"
        channel_notes[target_name] += f"; amplitude x{pressure_kin_scale:.6g} for W_kin = W_eq"

    n_i = n_e * (1.0 - charge * fraction) if comp.impurity else n_e.copy()
    n_imp = n_e * fraction if comp.impurity else np.zeros_like(n_e)
    p_e = QE * n_e * T_e
    p_i = QE * (n_i + n_imp) * T_i
    p_total = p_e + p_i
    with np.errstate(divide="ignore", invalid="ignore"):
        z_eff = np.where(n_e > 0.0, (n_i + charge**2 * n_imp) / np.where(n_e > 0.0, n_e, 1.0), np.nan)

    # --- residuals ------------------------------------------------------------
    finite = all(np.all(np.isfinite(a)) for a in (n_e, T_e, T_i, n_i, n_imp, p_total))
    interior = grid < 1.0
    positive = {name: bool(np.all(a[interior] > 0.0) and np.all(a >= 0.0)) if finite else False
                for name, a in (("n_e", n_e), ("T_e", T_e), ("T_i", T_i))}
    ions_ok = bool(finite and np.all(n_i >= 0.0) and np.all(n_imp >= 0.0))
    qn = n_e - (n_i + charge * n_imp)
    qn_rel = float(np.max(np.abs(qn)) / np.max(np.abs(n_e))) if finite and np.max(np.abs(n_e)) > 0 else np.nan
    zeff_dev = np.nan_to_num(z_eff - comp.z_eff, nan=0.0)
    zeff_worst = float(z_eff[np.nanargmax(np.abs(z_eff - comp.z_eff))]) if np.any(np.isfinite(z_eff)) else np.nan

    records = []
    for kind, name, requested, status in targets:
        values = {"n_e": n_e, "T_e": T_e, "T_i": T_i}[name]
        if kind == "peaking_factor":
            achieved = float(values[0] / _measure("volume_average", values, grid, geom, name))
            tol = _PEAKING_RTOL
        else:
            achieved = _measure(kind, values, grid, geom, name)
            tol = _TARGET_RTOL
        rel = abs(achieved - requested) / abs(requested)
        records.append(TargetResidual(f"{name}.{kind}", _definition(kind, geom), float(requested), float(achieved),
                                      float(achieved - requested), float(rel), status, tol,
                                      bool(status != "not_reached" and rel <= tol)))
    if temp.route == "ratio" and not isinstance(temp.ti_over_te, TabulatedProfile):
        r = _safe_divide(T_i, T_e)
        dev = np.where(T_e > 0.0, np.abs(r - ratio), 0.0)
        worst = float(r[np.argmax(dev)])
        records.append(TargetResidual("T_i/T_e", "T_i/T_e wherever T_e > 0 (worst point)", float(ratio[0]), worst,
                                      worst - float(ratio[0]), abs(worst - float(ratio[0])) / float(ratio[0]),
                                      "local", 1e-12, bool(np.max(dev) <= 1e-12 * float(ratio[0]))))
    if temp.route == "partition":
        fe_prof = _safe_divide(p_e, p_total)
        dev = np.where(p_total > 0.0, np.abs(fe_prof - f_e), 0.0)
        worst = float(fe_prof[np.argmax(dev)])
        records.append(TargetResidual("p_e/p_total", "electron pressure fraction wherever p > 0 (worst point)",
                                      f_e, worst, worst - f_e, abs(worst - f_e) / f_e, "local", 1e-12,
                                      bool(np.max(dev) <= 1e-12)))
    records.append(TargetResidual("z_eff", "local (n_i + Z_I^2 n_I)/n_e, uniform target (worst point)",
                                  comp.z_eff, zeff_worst, zeff_worst - comp.z_eff,
                                  abs(zeff_worst - comp.z_eff) / comp.z_eff, "local", _ZEFF_ATOL,
                                  bool(np.max(np.abs(zeff_dev)) <= _ZEFF_ATOL)))
    records.append(TargetResidual("quasineutrality", "max|n_e - n_i - Z_I n_I| / max(n_e)", 0.0, qn_rel, qn_rel,
                                  qn_rel, "local", _QN_RTOL, bool(qn_rel <= _QN_RTOL)))

    has_peq = bool(np.all(np.isfinite(p_eq)))
    if has_peq and finite:
        scale = float(np.max(p_eq))
        absolute = p_total - p_eq
        relative = absolute / scale
        valid = p_eq >= pressure_floor * scale
        max_rel = float(np.max(np.abs(relative[valid])))
        rms_rel = float(np.sqrt(np.mean(relative[valid] ** 2)))
        w_eq = 1.5 * _volume_integral(p_eq, geom["volume"])
        w_kin = 1.5 * _volume_integral(p_total, geom["volume"])
        w_rel = (w_kin - w_eq) / w_eq
    else:
        absolute = relative = None
        max_rel = rms_rel = w_eq = w_rel = float("nan")
        w_kin = 1.5 * _volume_integral(p_total, geom["volume"]) if finite else float("nan")
    held, solved = _HELD_SOLVED[(mode, closure)]
    locally = bool(np.isfinite(max_rel) and max_rel <= pressure_tolerance)
    report = PressureClosureReport(mode, closure, held, solved, absolute, relative, max_rel, rms_rel, w_eq, w_kin,
                                   w_rel, pressure_floor, pressure_tolerance, locally)
    if mode == "thermal_energy":
        records.append(TargetResidual("thermal_energy", "W = 3/2 int p dV over the traced volume", w_eq, w_kin,
                                      w_kin - w_eq, abs(w_rel), "linear", 1e-9, bool(abs(w_rel) <= 1e-9)))
    if mode == "equilibrium":
        records.append(TargetResidual("pressure_closure", "max |p_kin - p_eq| / max(p_eq) over p_eq >= floor",
                                      0.0, max_rel, max_rel, max_rel, "local", pressure_tolerance, locally))

    # --- status ------------------------------------------------------------------
    solved_names = {"temperature": ("T_e", "T_i"), "density": ("n_e",), "sqrt_split": ("n_e", "T_e")}.get(closure, ())
    unmet = [t.name for t in records if not t.met and t.name not in ("z_eff", "quasineutrality",
                                                                     "pressure_closure")]
    if not finite:
        status, message = "numerically_suspect", "non-finite values in the generated profiles"
        bad_closure = [n for n in solved_names if not positive.get(n, True)]
        if mode == "equilibrium" and bad_closure:
            status = "pressure_closure_failed"
            message = (f"closure {closure!r} gives a non-finite {bad_closure[0]}: the held profiles vanish "
                       "where p_eq does not")
    elif not all(positive.values()) or not ions_ok:
        negative = [n for n, ok in positive.items() if not ok] or ["an ion density"]
        if mode == "equilibrium" and any(n in solved_names for n in negative):
            status = "pressure_closure_failed"
            message = (f"closure {closure!r} would need a non-positive {negative[0]} to reach p_kin = p_eq; "
                       "the held assumptions are incompatible with the equilibrium pressure")
        else:
            status, message = "numerically_suspect", f"{negative[0]} is not positive inside the plasma"
    elif not records[[t.name for t in records].index("quasineutrality")].met:
        status, message = "quasineutrality_failed", f"quasi-neutrality residual {qn_rel:.3g}"
    elif not records[[t.name for t in records].index("z_eff")].met:
        status, message = "zeff_constraint_failed", f"Z_eff reaches {zeff_worst:.12g}, not {comp.z_eff:g}"
    elif mode == "equilibrium" and not locally:
        status, message = "pressure_closure_failed", f"local pressure residual {max_rel:.3g} > {pressure_tolerance:g}"
    elif unmet:
        status, message = "constraint_not_reached", f"not met: {', '.join(unmet)}"
    else:
        status = "success"
        if mode == "kinetic":
            message = ("kinetic profiles as specified; the source equilibrium is "
                       + ("pressure-consistent within tolerance" if locally else
                          f"NOT pressure-consistent (max |p_kin - p_eq|/max p_eq = {max_rel:.3g})"
                          if has_peq else "unchecked (no p_eq)"))
        elif mode == "thermal_energy":
            message = ("W_kin = W_eq; locally " + ("consistent" if locally else
                       f"NOT pressure-consistent (max relative residual {max_rel:.3g})"))
        else:
            message = f"p_kin = p_eq locally (max relative residual {max_rel:.3g})"

    validation = {
        "finite": finite,
        "positive": positive,
        "ion_densities_nonnegative": ions_ok,
        "quasineutrality_max_relative": qn_rel,
        "z_eff_max_abs_deviation": float(np.max(np.abs(zeff_dev))),
        "edge_values": {n: float(a[-1]) for n, a in (("n_e", n_e), ("T_e", T_e), ("T_i", T_i))},
        "monotonic_decreasing": {n: bool(np.all(np.diff(a) <= 0.0)) for n, a in
                                 (("n_e", n_e), ("T_e", T_e), ("T_i", T_i))},
        "axis_a_over_L_rho_pol": {n: float(-np.gradient(np.log(np.maximum(a, 1e-300)), np.sqrt(grid))[0])
                                  if finite and a[0] > 0 else float("nan")
                                  for n, a in (("n_e", n_e), ("T_e", T_e), ("T_i", T_i))},
    }
    species = [IonSpecies(comp.main_ion, 1.0, ION_SPECIES[comp.main_ion][1], "main_ion")]
    if comp.impurity:
        species.append(IonSpecies(comp.impurity, charge, ION_SPECIES[comp.impurity][1], "impurity"))
    level = max(levels.values()) if levels else 0
    if closure == "sqrt_split":
        level = 0
    t_value = time if time is not None else eq.time
    conv = eq.convention
    provenance = {
        "generator": "vaft.process.profile.generate_synthetic_kinetic_profiles",
        "issue": 122,
        "profile_basis": "assumed",
        "transport_predicted": False,
        "fidelity_level": int(level),
        "label": spec.label,
        "equilibrium_source": str(eq.metadata.get("source_type", type(equilibrium).__name__)),
        "equilibrium_cocos": conv.cocos if conv.cocos is not None else list(conv.candidates),
        "equilibrium_modified": False,
        "coordinate": "psi_norm of the source equilibrium",
        "grid": "spec.psi_norm" if spec.psi_norm is not None else f"{_DEFAULT_POINTS} points uniform in rho_pol_norm",
        "rho_tor_norm": geom["rho_tor_note"],
        "psi": geom["psi_note"],
        "line_average": _definition("line_average", geom),
        "volume_average": _definition("volume_average", geom),
        "pressure_constraint": mode,
        "closure": closure,
        "held": list(held),
        "solved": list(solved),
        "channels": channel_notes,
        "temperature_route": temp.route,
        "temperature_source": temp.source,
        "composition": {"main_ion": comp.main_ion, "impurity": comp.impurity, "impurity_charge": comp.impurity_charge,
                        "z_eff": comp.z_eff, "z_eff_definition": "local, uniform in radius",
                        "impurity_fraction_n_I_over_n_e": float(fraction)},
        "time": None if t_value is None else float(t_value),
        "time_source": "argument" if time is not None else ("equilibrium" if eq.time is not None else "none"),
    }
    if spec.sqrt_split is not None:
        provenance["sqrt_split"] = {"te_axis": spec.sqrt_split.te_axis, "ne_over_te": spec.sqrt_split.ne_over_te}
    return SyntheticKineticProfiles(
        label=spec.label, status=status, message=message, fidelity_level=int(level), psi_norm=grid,
        rho_pol_norm=np.sqrt(grid), rho_tor_norm=geom["rho_tor_norm"], psi=geom["psi"], volume=geom["volume"],
        n_e=n_e, n_i=n_i, n_impurity=n_imp, T_e=T_e, T_i=T_i, z_eff=z_eff, p_e=p_e, p_i=p_i, p_total=p_total,
        p_eq=p_eq, species=tuple(species), pressure=report, targets=tuple(records), validation=validation,
        resolved=notes, spec=spec, provenance=provenance, time=None if t_value is None else float(t_value),
    )


# --- presets and output -------------------------------------------------------------


def spec_from_plasma_state(state: AnalyticPlasmaState, *, main_ion: str = "H",
                           label: str | None = None) -> SyntheticKineticSpec:
    r"""The generator spec that reproduces a #1045 analytic plasma state on any equilibrium.

    Parameters
    ----------
    state : AnalyticPlasmaState
        From a preset (:func:`vaft.process.profile.analytic_hmode_state`, ...)
        or :func:`vaft.process.profile.compose_plasma_state` [-].
    main_ion : str, optional
        Hydrogenic main-ion label, ``"H"``, ``"D"`` or ``"T"``; the state
        itself does not name one [-].
    label : str, optional
        Label of the spec; the state's label when omitted [-].

    Returns
    -------
    SyntheticKineticSpec
        ``pressure_constraint="kinetic"``, the state's three channels as
        unnormalized :class:`~vaft.data.analytic_plasma_state.AnalyticProfile`
        shapes, ``T_i`` as an independent profile, the state's composition, and
        its ``psi_norm`` grid when that runs from 0 to 1 [-].

    Raises
    ------
    SyntheticProfileError
        The state's impurity charge matches no species of
        :data:`vaft.data.ION_SPECIES`, or the main ion is not hydrogenic.

    Convention
    ----------
    The channels stay in ``psi_norm`` exactly as the state defines them, so
    the generator evaluates the same analytic functions: a preset is the
    Level 1 special case of the generator with no normalization and no
    pressure constraint.  The impurity is the element whose nuclear charge
    equals ``state.impurity_charge`` (carbon for the default 6).

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1045] VAFT issue #1045, whose presets become special cases of the #122
       generator's inputs.
    """
    if not isinstance(state, AnalyticPlasmaState):
        _fail("invalid_profile_model", "state must be an AnalyticPlasmaState")
    from vaft.data.synthetic_kinetic_profiles import _species_for_state

    impurity = _species_for_state(state)
    composition = Composition(main_ion=main_ion, impurity=impurity, z_eff=state.z_eff,
                              impurity_charge=state.impurity_charge if impurity else None)
    grid = state.psi_norm if state.psi_norm[0] == 0.0 and state.psi_norm[-1] == 1.0 else None
    return SyntheticKineticSpec(
        temperature=TemperatureAssumption(T_i=ProfileSpec(state.profiles["T_i"]), source=f"#1045 preset {state.label!r}"),
        composition=composition, n_e=ProfileSpec(state.profiles["n_e"]), T_e=ProfileSpec(state.profiles["T_e"]),
        pressure_constraint="kinetic", psi_norm=grid, label=label or state.label,
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_synthetic_core_profiles(result: SyntheticKineticProfiles, ods=None, *, time: float | None = None,
                                  time_tolerance: float = 1e-6):
    r"""Store a successful synthetic kinetic state as one ``core_profiles.profiles_1d`` slice.

    Parameters
    ----------
    result : SyntheticKineticProfiles
        From :func:`generate_synthetic_kinetic_profiles`; must have status
        ``"success"`` [-].
    ods : ODS, optional
        Mutated in place; a new ODS when omitted [-].
    time : float, optional
        Slice time; the result's time when omitted [s].
    time_tolerance : float, optional
        An existing slice within this of *time* is replaced [s].

    Returns
    -------
    ODS
        The ODS with the slice written [-].

    Raises
    ------
    SyntheticProfileError
        A result whose status is not ``"success"`` (``status`` carried over),
        no time on either the result or the call, or a ``core_profiles.code.parameters``
        written by another producer.

    Input semantics
    ---------------
    Synthetic: a generated, validated kinetic state.

    Output semantics
    ----------------
    Stored: the same arrays in standard ``core_profiles`` paths, labelled
    synthetic and assumption-driven in ``ids_properties.comment`` and
    ``code.parameters``.

    Convention
    ----------
    ``grid.rho_pol_norm`` always; ``grid.rho_tor_norm`` only when it was derived
    from the equilibrium's ``q``; ``grid.psi`` in Wb, COCOS 11, only when the
    source convention fixes it -- a coordinate that is not known is left out,
    never substituted.  ``grid.volume`` in m^3.  Electrons and each ion
    carry ``density``, ``density_thermal``, ``temperature`` and
    ``pressure_thermal``; ion 0 is the main ion, ion 1 the impurity, with
    ``z_ion``, ``element[0].z_n`` and ``element[0].a``.  The slice also carries
    ``zeff``, ``pressure_thermal``, ``pressure_ion_total`` and
    ``t_i_average``.  ``code.parameters`` is a JSON object keyed by slice time,
    which survives an IMAS round trip as a string.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    No rotation, fast particles or radiation; nothing in ``equilibrium`` is
    touched.

    Provenance
    ----------
    .. [DD] IMAS Data Dictionary ``core_profiles`` (COCOS 11), the paths written.
    .. [122] VAFT issue #122: synthetic profiles never labelled as measured
       or transport-predicted.
    """
    import vaft

    if not isinstance(result, SyntheticKineticProfiles):
        _fail("invalid_profile_model", "result must be a SyntheticKineticProfiles")
    if not result.ok:
        _fail(result.status, f"refusing to store a {result.status!r} result: {result.message}")
    t = time if time is not None else result.time
    if t is None:
        _fail("invalid_equilibrium", "no time on the result or the call; pass time= (seconds)")
    t = float(t)
    if ods is None:
        from omas import ODS

        ods = ODS()
    existing = {}
    path = "core_profiles.code.parameters"
    if path in ods:
        try:
            existing = json.loads(str(ods[path]))
            if not isinstance(existing, dict) or existing.get("producer") != "vaft synthetic kinetic profiles (#122)":
                raise ValueError
        except ValueError:
            _fail("invalid_profile_model",
                  "core_profiles.code.parameters was written by another producer; store synthetic profiles "
                  "in their own ODS rather than mixing them with measured slices")
    index = None
    count = len(ods["core_profiles.profiles_1d"]) if "core_profiles.profiles_1d" in ods else 0
    for i in range(count):
        key = f"core_profiles.profiles_1d.{i}.time"
        if key in ods and abs(float(ods[key]) - t) <= time_tolerance:
            from omas import ODS

            index = i
            ods[f"core_profiles.profiles_1d.{i}"] = ODS()  # replaced whole, in place: no stale leaf survives
            break
    base = f"core_profiles.profiles_1d.{count if index is None else index}"
    ods[f"{base}.time"] = t
    ods[f"{base}.grid.rho_pol_norm"] = np.asarray(result.rho_pol_norm)
    if result.rho_tor_norm is not None:
        ods[f"{base}.grid.rho_tor_norm"] = np.asarray(result.rho_tor_norm)
    if result.psi is not None:
        ods[f"{base}.grid.psi"] = np.asarray(result.psi)
    ods[f"{base}.grid.volume"] = np.asarray(result.volume)
    for leaf in ("density", "density_thermal"):
        ods[f"{base}.electrons.{leaf}"] = np.asarray(result.n_e)
    ods[f"{base}.electrons.temperature"] = np.asarray(result.T_e)
    ods[f"{base}.electrons.pressure_thermal"] = np.asarray(result.p_e)
    densities = (result.n_i, result.n_impurity)
    for i, species in enumerate(result.species):
        ion = f"{base}.ion.{i}"
        ods[f"{ion}.label"] = species.label if species.role == "main_ion" else f"{species.label}{species.z:g}+"
        ods[f"{ion}.z_ion"] = float(species.z)
        ods[f"{ion}.element.0.z_n"] = float(ION_SPECIES[species.label][0])
        ods[f"{ion}.element.0.a"] = float(species.a)
        ods[f"{ion}.element.0.atoms_n"] = 1
        for leaf in ("density", "density_thermal"):
            ods[f"{ion}.{leaf}"] = np.asarray(densities[i])
        ods[f"{ion}.temperature"] = np.asarray(result.T_i)
        ods[f"{ion}.pressure_thermal"] = QE * np.asarray(densities[i]) * np.asarray(result.T_i)
    ods[f"{base}.zeff"] = np.asarray(result.z_eff)
    ods[f"{base}.pressure_thermal"] = np.asarray(result.p_total)
    ods[f"{base}.pressure_ion_total"] = np.asarray(result.p_i)
    ods[f"{base}.t_i_average"] = np.asarray(result.T_i)

    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.ids_properties.comment"] = (
        "Synthetic, assumption-driven kinetic profiles generated from a magnetic equilibrium "
        "(vaft #122); not measured, not fitted, not transport-predicted.")
    ods["core_profiles.code.name"] = "vaft.process.profile.generate_synthetic_kinetic_profiles"
    ods["core_profiles.code.version"] = str(getattr(vaft, "__version__", "unknown"))
    slices = dict(existing.get("slices", {}))
    record = dict(result.provenance)
    record.update(status=result.status, fidelity_level=result.fidelity_level,
                  pressure_max_relative_residual=result.pressure.max_relative_residual,
                  targets={t_.name: {"requested": t_.requested, "achieved": t_.achieved, "met": t_.met}
                           for t_ in result.targets})
    slices[f"{t:.9f}"] = _jsonable(record)
    ods[path] = json.dumps({"producer": "vaft synthetic kinetic profiles (#122)", "slices": slices}, sort_keys=True)
    n = len(ods["core_profiles.profiles_1d"])
    ods["core_profiles.time"] = np.asarray([float(ods[f"core_profiles.profiles_1d.{i}.time"]) for i in range(n)])
    return ods
