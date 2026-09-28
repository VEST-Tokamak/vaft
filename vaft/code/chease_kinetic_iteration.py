"""Self-consistent equilibrium / kinetic-profile iteration through CHEASE (#123).

Composes three existing pieces and adds no physics of its own:

1. **kinetic profiles** -- :func:`vaft.process.profile.generate_synthetic_kinetic_profiles`
   (#122) applied to the current equilibrium with one unchanging
   :class:`~vaft.data.SyntheticKineticSpec`, so every shape is regenerated on
   the new coordinates rather than re-interpolated from an earlier iteration;
2. **a CHEASE pressure source** -- ``PRES`` and ``PPRIME = dp/dpsi`` built from
   ``p_kin`` by a monotone (PCHIP) interpolant and its analytic derivative
   (:func:`pressure_source_from_kinetic`), and ``FF'`` from an explicit
   :class:`~vaft.data.CurrentPolicy` (:func:`current_source_for_update`);
3. **a fixed-boundary solve** -- :func:`vaft.code.chease.refine_equilibrium`
   with ``target_psin = 1`` (the ``EXPEQ`` boundary is the initial ``RBBBS``,
   the #887 rule) and the policy's ``NCSCAL``.  A zero return code is not
   acceptance: every update is validated (:func:`validate_chease_update`).

The pressure is then re-extracted from the *solved* equilibrium, compared
with the kinetic pressure regenerated on it, and the loop repeats until the
declared :class:`~vaft.data.ConvergenceCriteria` hold, a detector fires, or
something fails -- each an explicit status with the whole history kept.

Why this converges: CHEASE rescales ``p'`` and ``FF'`` together by one factor
``lambda`` to meet the normalization, so the solved pressure is ``lambda
(Delta psi_{n+1}/Delta psi_n) p_kin``.  Carrying the solved ``FF'`` amplitude
into the next input drives ``lambda -> 1``; the contraction factor is roughly
the pressure-driven share of the current, a few tenths for VEST, so a handful
of iterations reach the interpolation floor.

The result is an **assumption-driven self-consistent state**: the kinetic
profiles are declared, not measured or transport-predicted, and the
equilibrium is re-solved for their pressure, not reconstructed.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from vaft.data.equilibrium_kinetic_iteration import (
    ConvergenceCriteria,
    CurrentPolicy,
    CurrentSource,
    EquilibriumKineticSpec,
    IterationState,
    PressureSource,
    SelfConsistentState,
)
from vaft.data.synthetic_kinetic_profiles import SyntheticProfileError
from vaft.process.profile import generate_synthetic_kinetic_profiles, write_synthetic_core_profiles

from .chease import CHEASEConfig, CHEASEResult, _copy_geqdsk, _coerce_geqdsk, refine_equilibrium

__all__ = [
    "PressureSourceError",
    "build_consistent_state",
    "current_source_for_update",
    "pressure_source_from_kinetic",
    "validate_chease_update",
]

#: Method recorded for the pressure source.
PRESSURE_METHOD = (
    "PCHIP (Fritsch-Carlson monotone cubic Hermite) interpolant of p_kin over psi_N on the kinetic grid; "
    "PRES = interpolant on the g-file's uniform psi_N grid; PPRIME = analytic derivative of the "
    "interpolant / (psi_boundary - psi_axis) of the current equilibrium, in the g-file's own psi units "
    "and sign; no smoothing, no clipping; p' must not change sign (the adapter writes -|p'|)")

#: The CHEASE settings the iteration fixes, whatever the caller's config says.
_FIXED_SETTINGS = {
    "target_psin": "1.0: the EXPEQ boundary is RBBBS itself, the initial boundary restored every solve (#887)",
    "edge_zero": "False: p' is built one-signed here, so no edge sample is zeroed behind the source's back",
    "preserve_boundary_limiter": "True: RBBBS/limiter restored from the input, i.e. the initial boundary",
    "output_cocos": "input: the refined g-file keeps the source orientation",
}

#: Rational q values whose surfaces are located on every state.
_RATIONAL_Q = (1.0, 1.5, 2.0, 3.0, 4.0)

#: Relative tolerance on |F_edge| = R0 |B0| between the initial equilibrium and every update.
_F_EDGE_RTOL = 1e-3


class PressureSourceError(ValueError):
    """``p_kin`` cannot be handed to CHEASE as a pressure source (non-finite, negative or non-monotone)."""


# --- sources -----------------------------------------------------------------------


def _uniform_psi_norm(geqdsk) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(geqdsk["NW"]))


def pressure_source_from_kinetic(kinetic_psi_norm, p_kin, geqdsk, *, previous=None,
                                 relaxation: float = 1.0) -> PressureSource:
    """Build the ``PRES``/``PPRIME`` pair CHEASE reads from a kinetic pressure.

    Parameters
    ----------
    kinetic_psi_norm : array_like
        Kinetic grid, strictly increasing from 0 to 1 [-].
    p_kin : array_like
        Kinetic pressure on that grid [Pa].
    geqdsk : GEQDSK
        The equilibrium the derivative is taken on: its ``NW`` fixes the output
        grid and ``SIBRY - SIMAG`` the flux span [-].
    previous : array_like, optional
        The pressure used in the previous update on the same kinetic grid, the
        base of under-relaxation; ``None`` means no relaxation base [Pa].
    relaxation : float, optional
        ``alpha`` of ``p = previous + alpha (p_kin - previous)`` [-].

    Returns
    -------
    PressureSource
        The raw and relaxed pressure, ``PRES`` and ``PPRIME`` on the g-file
        grid, the method and a trapezoidal round-trip check [-].

    Raises
    ------
    PressureSourceError
        A non-finite or negative pressure, or one that rises outward anywhere:
        the adapter writes ``-|p'|`` into ``EXPEQ``, so a sign change would be
        silently folded rather than solved.
    """
    from scipy.interpolate import PchipInterpolator

    x = np.asarray(kinetic_psi_norm, dtype=float)
    raw = np.asarray(p_kin, dtype=float)
    if x.ndim != 1 or x.shape != raw.shape or x.size < 3 or x[0] != 0.0 or x[-1] != 1.0 or np.any(np.diff(x) <= 0):
        raise PressureSourceError("the kinetic grid must run strictly from psi_N = 0 to 1 with the pressure on it")
    used = raw
    if previous is not None and relaxation != 1.0:
        base = np.asarray(previous, dtype=float)
        if base.shape != raw.shape or not np.all(np.isfinite(base)):
            raise PressureSourceError("the relaxation base is not a finite pressure on the kinetic grid")
        used = base + float(relaxation) * (raw - base)
    if not np.all(np.isfinite(used)):
        raise PressureSourceError("the kinetic pressure is not finite")
    scale = float(np.max(np.abs(used)))
    if scale <= 0.0 or used.min() < -1e-12 * scale:
        raise PressureSourceError(f"the kinetic pressure must be non-negative with a positive maximum "
                                  f"(min {used.min():.4g} Pa)")
    rises = np.flatnonzero(np.diff(used) > 1e-12 * scale)
    if rises.size:
        where = float(x[rises[0]])
        raise PressureSourceError(
            f"the kinetic pressure rises outward near psi_N = {where:.3f}; CHEASE's EXPEQ carries p' of one "
            "sign (the adapter writes -|p'|), so a hollow pressure cannot be handed to it faithfully")
    # anti-alias: not time-domain -- the kinetic pressure over normalized poloidal flux.
    interpolant = PchipInterpolator(x, used, extrapolate=False)
    grid = _uniform_psi_norm(geqdsk)
    pressure = np.asarray(interpolant(grid), dtype=float)
    dp = np.asarray(interpolant.derivative()(grid), dtype=float)
    dp = np.minimum(dp, 0.0)  # a monotone PCHIP has dp <= 0 up to rounding; this removes -0.0/+eps only
    span = float(geqdsk["SIBRY"]) - float(geqdsk["SIMAG"])
    if span == 0.0 or not np.isfinite(span):
        raise PressureSourceError("the equilibrium's flux span SIBRY - SIMAG is zero or not finite")
    pprime = dp / span
    from scipy.integrate import cumulative_trapezoid

    # p(x) = p(1) - int_x^1 p' dpsi_N; integrating along the reversed grid gives -int_x^1 directly.
    rebuilt = pressure[-1] + cumulative_trapezoid(dp[::-1], grid[::-1], initial=0.0)[::-1]
    roundtrip = float(np.max(np.abs(rebuilt - pressure)) / scale)
    return PressureSource(
        kinetic_psi_norm=x, raw_pressure=raw, relaxed_pressure=used, relaxation=float(relaxation),
        psi_norm=grid, pressure=pressure, dpressure_dpsi_norm=dp, pprime=pprime, psi_span=span,
        method=PRESSURE_METHOD, roundtrip_max_relative=roundtrip)


def _ffprime_shape(policy: CurrentPolicy, initial, grid: np.ndarray) -> tuple[np.ndarray, str]:
    """The normalized ``FF'`` shape the policy holds, on *grid* (unit max |shape|)."""
    if policy.kind == "analytic_ffprime_shape":
        from vaft.process.profile import evaluate_analytic_profile

        shape = -np.asarray(evaluate_analytic_profile(policy.current_profile, grid, derivative=True), dtype=float)
        method = ("FF' = A * (-dg/dpsi_N) of the declared analytic current profile g (#1166 scope B), "
                  "evaluated analytically on the g-file grid")
    else:
        source = np.asarray(initial["FFPRIM"], dtype=float)
        # anti-alias: not time-domain -- the initial FF' over normalized poloidal flux.
        shape = np.interp(grid, np.linspace(0.0, 1.0, source.size), source)
        method = ("FF' = A * (the initial equilibrium's FFPRIM, normalized), re-sampled from that one array "
                  "every update")
    peak = float(np.max(np.abs(shape)))
    if not np.all(np.isfinite(shape)) or peak <= 0.0:
        raise ValueError("the declared FF' shape is zero or not finite")
    return shape / peak, method


def current_source_for_update(policy: CurrentPolicy, initial, previous) -> CurrentSource:
    """Build the ``FF'`` of the next CHEASE input from the declared current policy.

    Parameters
    ----------
    policy : CurrentPolicy
        What is held [-].
    initial : GEQDSK
        The initial equilibrium, whose ``FFPRIM`` is the held shape under
        ``preserve_ffprime_shape`` [-].
    previous : GEQDSK
        The latest accepted equilibrium, whose ``FFPRIM`` gives the amplitude
        and whose ``NW`` the grid [-].

    Returns
    -------
    CurrentSource
        The shape, the amplitude ``A = <FF'_prev, s>/<s, s>`` (least squares
        on the grid, so a sign-changing ``FF'`` keeps its projection) and
        ``FF' = A s`` [T^2 m^2 per g-file psi unit].

    Raises
    ------
    ValueError
        The shape is degenerate, or the previous ``FF'`` has no component
        along it.
    """
    grid = _uniform_psi_norm(previous)
    shape, method = _ffprime_shape(policy, initial, grid)
    prev = np.asarray(previous["FFPRIM"], dtype=float)
    # anti-alias: not time-domain -- the previous FF' over normalized poloidal flux.
    prev = np.interp(grid, np.linspace(0.0, 1.0, prev.size), prev)
    amplitude = float(np.dot(prev, shape) / np.dot(shape, shape))
    if not np.isfinite(amplitude) or amplitude == 0.0:
        raise ValueError("the previous FF' has no component along the declared shape; its amplitude is undefined")
    return CurrentSource(psi_norm=grid, shape=shape, amplitude=amplitude, ffprime=amplitude * shape,
                         method=method + "; A = least-squares projection of the previous solve's FF' on the shape")


# --- validation ----------------------------------------------------------------------


def _closed(r, z) -> np.ndarray:
    rz = np.column_stack([np.asarray(r, float).ravel(), np.asarray(z, float).ravel()])
    if np.linalg.norm(rz[0] - rz[-1]) > 1e-12:
        rz = np.vstack([rz, rz[0]])
    return rz


def _boundary_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance of each point to the closed polyline *polygon* [m]."""
    a, b = polygon[:-1], polygon[1:]
    ab = b - a
    length2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(rel * ab[None], axis=2) / length2[None], 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(points[:, None, :] - nearest, axis=2), axis=1)


def _q_at(geqdsk, psi_norm: float) -> float:
    """``|q|`` at *psi_norm* by the cubic the adapter samples ``QSPEC`` with [-]."""
    from scipy.interpolate import interp1d

    q = np.abs(np.asarray(geqdsk["QPSI"], dtype=float))
    # anti-alias: not time-domain -- q over normalized poloidal flux, as the adapter samples QSPEC.
    return float(interp1d(np.linspace(0.0, 1.0, q.size), q, kind="cubic")(psi_norm))


def _check(value, tolerance, ok) -> dict[str, Any]:
    return {"value": None if value is None else float(value), "tolerance": tolerance, "ok": bool(ok)}


def validate_chease_update(run: CHEASEResult, *, initial, policy: CurrentPolicy, target: float,
                           criteria: ConvergenceCriteria) -> tuple[str, str, dict[str, Any]]:
    """Accept or refuse one CHEASE update: solver, geometry, normalization and sign checks.

    Parameters
    ----------
    run : CHEASEResult
        From :func:`vaft.code.chease.refine_equilibrium` [-].
    initial : GEQDSK
        The initial equilibrium: its orientation and ``|F_edge|`` are held [-].
    policy : CurrentPolicy
        Names the held normalization [-].
    target : float
        The held ``|I_p|`` [A] or ``q95`` [-].
    criteria : ConvergenceCriteria
        ``normalization_rtol`` and ``boundary_rtol`` [-].

    Returns
    -------
    tuple
        ``(status, message, checks)``: status ``"accepted"``, ``"chease_failed"``
        (no solution: a non-zero exit, a timeout, no ``EQDSK_COCOS_02.OUT``)
        or ``"validation_failed"`` (a solution that fails a check); ``checks``
        maps each check to its value, tolerance and verdict [-].

    Checks
    ------
    ``finite`` (``PSIRZ``, ``PRES``, ``PPRIME``, ``FFPRIM``, ``QPSI``, ``FPOL``),
    ``pressure_nonnegative``, ``axis_inside`` the solved boundary,
    ``boundary_preserved`` (the largest distance of CHEASE's own solved
    ``RBBBS`` from the ``EXPEQ`` boundary, in half radial extents),
    ``normalization`` (the held ``I_p`` or ``q95``, relative), ``orientation``
    (``I_p`` and ``B0`` signs of the refined file equal the initial's) and
    ``vacuum_field`` (``|F_edge|`` held).
    """
    from vaft.data.eqdsk import read_geqdsk

    checks: dict[str, Any] = {"returncode": run.returncode, "runtime_status": run.runtime_status,
                              "workdir": str(run.workdir), "comparison": dict(run.comparison)}
    native = Path(run.workdir) / "EQDSK_COCOS_02.OUT"
    if run.timed_out:
        reason = (run.stderr or "").strip().splitlines()[-1:] or [run.runtime_status]
        return "chease_failed", f"CHEASE stopped: {reason[0]}", checks
    if not run.ok or run.refined_geqdsk is None or not native.exists():
        return "chease_failed", (f"CHEASE returned {run.returncode} without a solution "
                                 f"(EQDSK_COCOS_02.OUT {'present' if native.exists() else 'missing'}); "
                                 f"see {Path(run.workdir) / 'chease.log'}"), checks
    try:
        refined = read_geqdsk(run.refined_geqdsk)
        solved = read_geqdsk(native)
    except Exception as error:  # noqa: BLE001 - an unreadable solution is a validation failure, named
        return "validation_failed", f"the solved g-file cannot be read: {error}", checks

    arrays = {key: np.asarray(refined[key], dtype=float) for key in ("PSIRZ", "PRES", "PPRIME", "FFPRIM", "QPSI", "FPOL")}
    finite = all(np.all(np.isfinite(a)) for a in arrays.values())
    checks["finite"] = _check(None, None, finite)
    pres = arrays["PRES"]
    checks["pressure_nonnegative"] = _check(pres.min(), 0.0, finite and pres.min() >= -1e-9 * max(pres.max(), 1e-300))

    solved_rz = _closed(solved["RBBBS"], solved["ZBBBS"])
    from matplotlib.path import Path as MplPath

    axis = (float(solved["RMAXIS"]), float(solved["ZMAXIS"]))
    checks["axis_inside"] = _check(None, None, MplPath(solved_rz).contains_point(axis))
    if run.materialized is not None:
        requested = _closed(run.materialized.boundary[:, 0], run.materialized.boundary[:, 1])
        half_width = 0.5 * float(np.ptp(requested[:, 0]))
        worst = float(np.max(_boundary_distance(solved_rz[:-1], requested)) / half_width)
        checks["boundary_preserved"] = _check(worst, criteria.boundary_rtol, worst <= criteria.boundary_rtol)
    else:
        checks["boundary_preserved"] = _check(None, criteria.boundary_rtol, False)

    if policy.normalization == "plasma_current":
        achieved = abs(float(refined["CURRENT"]))
    else:
        achieved = _q_at(refined, 0.95)
    miss = abs(achieved - target) / abs(target)
    checks["normalization"] = _check(miss, criteria.normalization_rtol, miss <= criteria.normalization_rtol)
    checks["normalization"]["achieved"], checks["normalization"]["target"] = achieved, float(target)
    same = (np.sign(float(refined["CURRENT"])) == np.sign(float(initial["CURRENT"]))
            and np.sign(float(refined["BCENTR"])) == np.sign(float(initial["BCENTR"])))
    checks["orientation"] = _check(None, None, same)
    f0 = abs(float(np.asarray(initial["FPOL"], dtype=float)[-1]))
    f1 = abs(float(arrays["FPOL"][-1]))
    f_miss = abs(f1 - f0) / f0 if f0 else np.inf
    checks["vacuum_field"] = _check(f_miss, _F_EDGE_RTOL, f_miss <= _F_EDGE_RTOL)

    failed = [name for name, value in checks.items() if isinstance(value, dict) and "ok" in value and not value["ok"]]
    if failed:
        detail = ", ".join(f"{name} ({checks[name]['value']:.3g} > {checks[name]['tolerance']:g})"
                           if checks[name]["value"] is not None and checks[name]["tolerance"] is not None
                           else name for name in failed)
        return "validation_failed", f"the CHEASE update fails: {detail}", checks
    return "accepted", "CHEASE update accepted", checks


# --- states ----------------------------------------------------------------------------


def _scalars(geqdsk) -> dict[str, float]:
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    names = {"ip": "ip", "q0": "q0", "q95": "q95", "li": "li_virial", "beta_p": "beta_p_boundary_average",
             "beta_t": "beta_t", "beta_n": "beta_n", "magnetic_axis_r": "magnetic_axis_r",
             "magnetic_axis_z": "magnetic_axis_z", "shafranov_shift": "shafranov_shift"}
    out = {name: float("nan") for name in names}
    try:
        values = derive_global_descriptors(as_equilibrium(geqdsk)).values
    except Exception:  # noqa: BLE001 - a descriptor that cannot be derived is NaN, and fails any criterion on it
        values = {}
    for name, key in names.items():
        item = values.get(key)
        if item is not None and item.available and np.ndim(item.value) == 0:
            out[name] = float(item.value)
    for name in ("ip", "q0", "q95"):
        out[name] = abs(out[name])
    out["qmin"] = float(np.min(np.abs(np.asarray(geqdsk["QPSI"], dtype=float))))
    return out


def _rational_surfaces(geqdsk) -> dict[str, float]:
    q = np.abs(np.asarray(geqdsk["QPSI"], dtype=float))
    x = np.linspace(0.0, 1.0, q.size)
    found = {}
    for value in _RATIONAL_Q:
        crossing = np.flatnonzero((q[:-1] - value) * (q[1:] - value) <= 0.0)
        if crossing.size:
            i = int(crossing[0])
            dq = q[i + 1] - q[i]
            found[f"q={value:g}"] = float(x[i] + (value - q[i]) / dq * (x[i + 1] - x[i])) if dq else float(x[i])
    return found


def _relative_change(new, old) -> float:
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    scale = float(np.max(np.abs(new)))
    return float(np.max(np.abs(new - old)) / scale) if scale > 0 else float("nan")


def _metrics(kin, p_eq, q, rho_tor, scalars, previous: IterationState | None, mode: str) -> dict[str, float]:
    p_kin = np.asarray(kin.p_total, dtype=float)
    metrics: dict[str, float] = {
        "closure_max_relative": float(kin.pressure.max_relative_residual),
        "closure_edge_max_relative": float(kin.pressure.edge_max_relative_residual),
        "thermal_energy_relative": float(kin.pressure.thermal_energy_relative_difference),
        "w_kin": float(kin.pressure.thermal_energy_kin),
    }
    if np.all(np.isfinite(p_eq)):
        diff = (p_kin - p_eq) / float(np.max(p_kin))
        metrics["pressure_max_relative"] = float(np.max(np.abs(diff)))
        metrics["pressure_rms_relative"] = float(np.sqrt(np.mean(diff * diff)))
    else:
        metrics["pressure_max_relative"] = metrics["pressure_rms_relative"] = float("nan")
    if mode == "equilibrium_pressure":
        # The closure is exact on its declared region; the carried edge is reported separately.
        metrics["pressure_max_relative"] = metrics["closure_max_relative"]
    if previous is None or previous.kinetic is None:
        return metrics
    old = previous.kinetic
    metrics["p_kin_change"] = _relative_change(p_kin, old.p_total)
    metrics["p_eq_change"] = _relative_change(p_eq, previous.p_eq)
    for name in ("n_e", "T_e", "T_i", "n_i"):
        metrics[f"{name}_change"] = _relative_change(getattr(kin, name), getattr(old, name))
    metrics["q_change"] = _relative_change(q, previous.q)
    if rho_tor is not None and previous.rho_tor_norm is not None:
        metrics["rho_tor_change"] = float(np.max(np.abs(np.asarray(rho_tor) - previous.rho_tor_norm)))
    else:
        metrics["rho_tor_change"] = float("nan")
    for name in ("q0", "q95", "li", "beta_p"):
        new, before = scalars[name], previous.scalars.get(name, np.nan)
        metrics[f"{name}_change"] = abs(new - before) / abs(new) if new else float("nan")
    w_old = previous.metrics.get("w_eq", np.nan)
    metrics["w_th_change"] = abs(scalars["w_th"] - w_old) / abs(scalars["w_th"]) if scalars["w_th"] else float("nan")
    metrics["axis_r_change"] = abs(scalars["magnetic_axis_r"] - previous.scalars.get("magnetic_axis_r", np.nan))
    metrics["axis_z_change"] = abs(scalars["magnetic_axis_z"] - previous.scalars.get("magnetic_axis_z", np.nan))
    return metrics


def _generate(geqdsk, spec: EquilibriumKineticSpec, time):
    """#122 on one equilibrium: ``(result, None)`` or ``(result_or_None, failure message)``."""
    try:
        kin = generate_synthetic_kinetic_profiles(geqdsk, spec.kinetic, time=time)
    except SyntheticProfileError as error:
        return None, f"#122 refused the kinetic spec on this equilibrium ({error.status}): {error}"
    if not kin.ok:
        return kin, f"#122 kinetic state not valid ({kin.status}): {kin.message}"
    return kin, None


def _state(index, geqdsk, path, spec, time, previous, *, criteria: ConvergenceCriteria | None = None,
           **extra) -> IterationState:
    """Regenerate the kinetic state on *geqdsk* and measure it against the equilibrium and *previous*."""
    kin, failure = _generate(geqdsk, spec, time)
    scalars = _scalars(geqdsk)
    base = dict(index=index, equilibrium=geqdsk, geqdsk_path=path, scalars=scalars,
                rational_surfaces=_rational_surfaces(geqdsk), **extra)
    if failure is not None:
        validation = dict(base.pop("validation", {}) or {})
        validation["kinetic"] = {"status": None if kin is None else kin.status, "message": failure}
        return IterationState(status="kinetic_generation_failed", message=failure, kinetic=kin,
                              validation=validation, **base)
    q_file = np.abs(np.asarray(geqdsk["QPSI"], dtype=float))
    # anti-alias: not time-domain -- |q| over normalized poloidal flux onto the kinetic grid.
    q = np.interp(kin.psi_norm, np.linspace(0.0, 1.0, q_file.size), q_file)
    p_eq = np.asarray(kin.p_eq, dtype=float)
    scalars["w_th"] = float(kin.pressure.thermal_energy_eq)
    metrics = _metrics(kin, p_eq, q, kin.rho_tor_norm, scalars, previous, spec.closure)
    metrics["w_eq"] = scalars["w_th"]
    if criteria is not None and previous is not None:
        metrics.update({f"met_{name}": float(ok) for name, ok in _criteria_met_metrics(metrics, criteria).items()})
    validation = dict(base.pop("validation", {}) or {})
    validation["kinetic"] = {"status": kin.status, "message": kin.message,
                             "targets": {t.name: {"requested": t.requested, "achieved": t.achieved, "met": t.met}
                                         for t in kin.targets},
                             "checks": dict(kin.validation)}
    status = "initial" if index == 0 else "accepted"
    return IterationState(status=status, message=kin.message if index == 0 else "accepted", kinetic=kin,
                          psi_norm=kin.psi_norm, p_eq=p_eq, p_kin=kin.p_total, q=q,
                          rho_tor_norm=kin.rho_tor_norm, metrics=metrics, validation=validation, **base)


# --- decisions ---------------------------------------------------------------------------

_CRITERIA_METRICS = {
    "pressure_rtol": ("pressure_max_relative",),
    "profile_rtol": ("p_kin_change", "p_eq_change", "n_e_change", "T_e_change", "T_i_change", "n_i_change"),
    "q_rtol": ("q_change",),
    "coordinate_atol": ("rho_tor_change",),
    "scalar_rtol": ("q0_change", "q95_change", "li_change", "beta_p_change", "w_th_change"),
    "axis_atol": ("axis_r_change", "axis_z_change"),
}


def _criteria_met_metrics(metrics: Mapping[str, float], criteria: ConvergenceCriteria) -> dict[str, bool]:
    """Each enabled criterion's verdict on a metrics record; a NaN metric never meets its tolerance."""
    met = {}
    for name, tolerance in criteria.enabled().items():
        values = [metrics.get(metric, np.nan) for metric in _CRITERIA_METRICS[name]]
        met[name] = bool(all(np.isfinite(v) and v <= tolerance for v in values))
    return met


def _criteria_met(state: IterationState, criteria: ConvergenceCriteria) -> dict[str, bool]:
    """Each enabled criterion's verdict on *state*."""
    return _criteria_met_metrics(state.metrics, criteria)


def _decide(states: list[IterationState], criteria: ConvergenceCriteria) -> tuple[str | None, str]:
    """The iteration's status after the latest accepted state, or ``None`` to continue."""
    latest = states[-1]
    met = _criteria_met(latest, criteria)
    if all(met.values()):
        return "converged", (f"every enabled criterion met after {latest.index} update(s): max|p_kin - p_eq|/max "
                             f"p_kin = {latest.metrics['pressure_max_relative']:.3g}")
    r = np.array([s.metrics.get("pressure_max_relative", np.nan) for s in states])
    s = np.array([s.metrics.get("thermal_energy_relative", np.nan) for s in states])
    n = len(states) - 1
    window = int(criteria.divergence_window)
    if n >= window and np.all(np.diff(r[-window - 1:]) > 0) and r[-1] > r[0]:
        return "diverged", (f"the pressure residual grew in each of the last {window} updates, to {r[-1]:.3g} "
                            f"(initial {r[0]:.3g})")
    w = int(criteria.oscillation_window)
    if len(s) >= w:
        tail = s[-w:]
        alternating = np.all(np.isfinite(tail)) and np.all(tail[1:] * tail[:-1] < 0)
        if alternating and abs(tail[-1]) >= criteria.oscillation_ratio * abs(tail[-3]):
            return "oscillating", (f"the thermal-energy residual alternates in sign over {w} states without "
                                   f"decaying (|s_n|/|s_n-2| = {abs(tail[-1]) / abs(tail[-3]):.3g})")
    k = int(criteria.stagnation_window)
    if n >= k and np.isfinite(r[-1]) and r[-1] >= criteria.stagnation_ratio * r[-1 - k]:
        return "stagnated", (f"the pressure residual fell by less than {1 - criteria.stagnation_ratio:.0%} over "
                             f"{k} updates ({r[-1 - k]:.3g} -> {r[-1]:.3g}); unmet: "
                             + ", ".join(name for name, ok in met.items() if not ok))
    return None, ""


# --- the workflow --------------------------------------------------------------------------


def _coerce_initial(equilibrium):
    """A GEQDSK from a path, GEQDSK, mapping, ODS or a #120 ``SyntheticEquilibriumResult``."""
    refined = getattr(equilibrium, "refined_geqdsk", None)
    if hasattr(equilibrium, "spec") and hasattr(equilibrium, "achieved_profiles"):
        if not equilibrium.ok or refined is None:
            raise ValueError(f"the #120 synthesis did not succeed ({equilibrium.status}); nothing to iterate from")
        return _coerce_geqdsk(refined), f"#120 synthesized equilibrium ({refined})"
    return _coerce_geqdsk(equilibrium), str(getattr(equilibrium, "source", "") or type(equilibrium).__name__)


def _solve_config(config: CHEASEConfig | None, policy: CurrentPolicy, workdir: Path) -> CHEASEConfig:
    base = config or CHEASEConfig(create_plot=False)
    return replace(base, workdir=workdir, target_psin=1.0, ncscal=policy.ncscal,
                   q_constraint_psi_norm=0.95 if policy.normalization == "q95" else None,
                   edge_zero=False, preserve_boundary_limiter=True, output_cocos="input")


def _write_ods(final: IterationState, spec: EquilibriumKineticSpec, status: str, time, policy_record) -> Any:
    ods = final.equilibrium.to_omas()
    t = time if time is not None else final.kinetic.time
    if t is not None:
        ods["equilibrium.time"] = np.asarray([float(t)])
        ods["equilibrium.time_slice.0.time"] = float(t)
    write_synthetic_core_profiles(final.kinetic, ods, time=t)
    ods["equilibrium.ids_properties.comment"] = (
        "Assumption-driven self-consistent equilibrium/kinetic state (vaft #123): "
        + ("the given equilibrium, decomposed into synthetic kinetic profiles" if spec.closure == "equilibrium_pressure"
           else "a CHEASE fixed-boundary equilibrium re-solved for the pressure of declared synthetic kinetic "
                "profiles")
        + "; not transport-predicted, not reconstructed from measurements.")
    ods["equilibrium.code.name"] = "vaft.code.chease_kinetic_iteration"
    ods["equilibrium.code.parameters"] = json.dumps(
        {"producer": "vaft self-consistent equilibrium-kinetic iteration (#123)", "status": status,
         "closure": spec.closure, "iteration": final.index, "policy": policy_record,
         "geqdsk": None if final.geqdsk_path is None else str(final.geqdsk_path)},
        sort_keys=True, default=str)
    return ods


def build_consistent_state(equilibrium, spec: EquilibriumKineticSpec, *, config: CHEASEConfig | None = None,
                           workdir: str | Path | None = None, time: float | None = None,
                           resume: SelfConsistentState | None = None) -> SelfConsistentState:
    """Iterate an equilibrium and synthetic kinetic profiles until they are mutually consistent.

    Parameters
    ----------
    equilibrium : GEQDSK, path, ODS or SyntheticEquilibriumResult
        The initial equilibrium; a #120 result contributes its refined g-file.
        Ignored when *resume* is given [-].
    spec : EquilibriumKineticSpec
        Closure mode, #122 kinetic spec, current policy and convergence
        criteria [-].
    config : CHEASEConfig, optional
        CHEASE numerics (mesh, output box, executable, timeout).  The
        iteration fixes ``target_psin = 1``, ``ncscal`` and
        ``q_constraint_psi_norm`` from the policy, ``edge_zero = False``,
        ``preserve_boundary_limiter = True`` and ``output_cocos = "input"``,
        and records that it did [-].
    workdir : path, optional
        Root of the per-update CHEASE directories ``iteration_NN``; a fresh
        temporary directory when omitted [-].
    time : float, optional
        Time stamped on the kinetic profiles and the output ODS [s].
    resume : SelfConsistentState, optional
        Continue a previous run from its last accepted state, keeping its
        history and relaxation base; ``max_iterations`` counts the whole
        history [-].

    Returns
    -------
    SelfConsistentState
        Status, every state (initial, accepted and the failed one that
        stopped the loop), the final accepted state, the ``equilibrium`` +
        ``core_profiles`` ODS of that state, the recorded policy and the
        provenance [-].

    Raises
    ------
    ValueError
        *spec* is not an :class:`~vaft.data.EquilibriumKineticSpec`, or a
        #120 result that did not succeed.

    Notes
    -----
    ``equilibrium_pressure`` generates the kinetic state once on the given
    equilibrium and never calls CHEASE.  ``kinetic_pressure`` repeats: build
    ``PRES``/``PPRIME`` from ``p_kin`` of the latest state and ``FF'`` from the
    policy, write the held ``I_p`` as the input ``CURRENT`` (``q95`` is read by
    the adapter from the latest solved ``q``), solve, validate, regenerate the
    kinetic state on the solution from the same spec, measure, decide.  A
    missing CHEASE executable is a ``chease_failed`` state, not an exception.
    """
    if not isinstance(spec, EquilibriumKineticSpec):
        raise ValueError("spec must be an EquilibriumKineticSpec")
    criteria, policy = spec.convergence, spec.current_policy
    if policy.current_profile is not None:
        from .chease_synthesis import _monotone_drop

        problem = _monotone_drop("current_profile", policy.current_profile,
                                 "FF' = -dg/dpsi_N would change sign, a reversed current")
        if problem is not None:
            raise ValueError(f"{problem[0]}: {problem[1]}")
    if resume is not None:
        initial_state = resume.initial
        initial = initial_state.equilibrium
        source_label = resume.provenance.get("initial_equilibrium", "resumed")
    else:
        initial, source_label = _coerce_initial(equilibrium)
        initial_state = _state(0, _copy_geqdsk(initial), None, spec, time, None)
    if policy.normalization == "plasma_current":
        target = abs(float(initial["CURRENT"]))
    else:
        target = _q_at(initial, 0.95)
    policy_record = {"kind": policy.kind, "normalization": policy.normalization, "ncscal": policy.ncscal,
                     "held_target": target, "held_target_unit": "A" if policy.normalization == "plasma_current" else "-",
                     "current_profile": None if policy.current_profile is None else repr(policy.current_profile),
                     **policy.describe()}
    root = Path(workdir) if workdir is not None else (
        Path(resume.provenance["workdir"]) if resume is not None and resume.provenance.get("workdir")
        else Path(tempfile.mkdtemp(prefix="vaft-kinetic-iteration-")))
    provenance = {
        "workflow": "vaft.code.chease_kinetic_iteration.build_consistent_state",
        "issue": 123,
        "state_basis": "assumption-driven self-consistent",
        "transport_predicted": False,
        "reconstructed": False,
        "closure": spec.closure,
        "kinetic_generator": "vaft.process.profile.generate_synthetic_kinetic_profiles (#122), same spec every state",
        "kinetic_pressure_constraint": spec.kinetic.pressure_constraint,
        "initial_equilibrium": source_label,
        "pressure_source_method": PRESSURE_METHOD,
        "chease_fixed_settings": dict(_FIXED_SETTINGS),
        "relaxation": criteria.relaxation,
        "tolerances": criteria.enabled(),
        "max_iterations": int(criteria.max_iterations),
        "workdir": str(root),
        "resumed_from_iteration": None if resume is None else (resume.final.index if resume.final else None),
    }

    def finish(status, message, states, final):
        ods = None
        if final is not None and final.kinetic is not None and final.kinetic.ok:
            try:
                ods = _write_ods(final, spec, status, time, policy_record)
            except Exception as error:  # noqa: BLE001 - recorded, the history is still returned
                provenance["ods_error"] = f"{type(error).__name__}: {error}"
        return SelfConsistentState(spec=spec, status=status, message=message, initial=states[0],
                                   iterations=tuple(states[1:]), final=final, ods=ods, policy=policy_record,
                                   provenance=provenance)

    states: list[IterationState] = [initial_state] + (list(resume.iterations) if resume is not None else [])
    if not initial_state.accepted:
        return finish("kinetic_generation_failed", initial_state.message, states, None)

    if spec.closure == "equilibrium_pressure":
        report = initial_state.kinetic.pressure
        message = (f"the given equilibrium's pressure is decomposed by the #122 closure {spec.kinetic.closure!r}: "
                   f"max|p_kin - p_eq|/max p_eq = {report.max_relative_residual:.3g} up to psi_N = "
                   f"{report.closure_exact_up_to_psi_n:.4g}; no CHEASE update is requested")
        return finish("converged", message, states, initial_state)

    accepted = [s for s in states if s.accepted]
    last = accepted[-1]
    if resume is not None and states[-1] is not last:
        # A resumed run restarts from the last accepted state, not from the failure.
        states = [s for s in states if s.accepted]
    base_pressure = (last.pressure_source.relaxed_pressure if last.pressure_source is not None
                     else initial_state.p_eq)
    solve_root = root
    solve_root.mkdir(parents=True, exist_ok=True)
    status, message = None, ""
    while status is None:
        index = len(states)
        if index > int(criteria.max_iterations):
            status = "max_iterations"
            message = (f"{criteria.max_iterations} updates without meeting every criterion; unmet: "
                       + ", ".join(k for k, ok in _criteria_met(last, criteria).items() if not ok))
            break
        workdir_n = solve_root / f"iteration_{index:02d}"
        previous_g = last.equilibrium
        try:
            psrc = pressure_source_from_kinetic(last.kinetic.psi_norm, last.kinetic.p_total, previous_g,
                                                previous=base_pressure, relaxation=criteria.relaxation)
        except PressureSourceError as error:
            states.append(IterationState(index=index, status="invalid_pressure_source", message=str(error)))
            status, message = "invalid_pressure_source", str(error)
            break
        try:
            csrc = current_source_for_update(policy, initial, previous_g)
        except ValueError as error:
            states.append(IterationState(index=index, status="invalid_pressure_source", pressure_source=psrc,
                                         message=f"FF' source: {error}"))
            status, message = "invalid_pressure_source", f"FF' source: {error}"
            break
        modified = _copy_geqdsk(previous_g)
        modified["PRES"] = psrc.pressure.copy()
        modified["PPRIME"] = psrc.pprime.copy()
        modified["FFPRIM"] = csrc.ffprime.copy()
        if policy.normalization == "plasma_current":
            modified["CURRENT"] = float(np.sign(float(initial["CURRENT"])) * target)
        solve = _solve_config(config, policy, workdir_n)
        chease_inputs = {"workdir": str(workdir_n), "ncscal": solve.ncscal, "target_psin": solve.target_psin,
                         "q_constraint_psi_norm": solve.q_constraint_psi_norm, "edge_zero": solve.edge_zero,
                         "mesh": solve.resolved_mesh, "nw": int(solve.nw),
                         "current_input": float(modified["CURRENT"])}
        common = dict(pressure_source=psrc, current_source=csrc, chease_workdir=workdir_n,
                      chease_inputs=chease_inputs)
        try:
            run = refine_equilibrium(modified, solve)
        except FileNotFoundError as error:
            from .chease import find_chease_executable

            reason = ("no CHEASE executable (set CHEASE, CHEASEHOME or CHEASE_EXEC_DIR); nothing was solved"
                      if find_chease_executable(solve) is None else f"{error}")
            states.append(IterationState(index=index, status="chease_failed", message=reason, **common))
            status, message = "chease_failed", reason
            break
        except Exception as error:  # noqa: BLE001 - the adapter's failure, named and kept with the history
            reason = f"CHEASE adapter raised {type(error).__name__}: {error}"
            states.append(IterationState(index=index, status="chease_failed", message=reason, **common))
            status, message = "chease_failed", reason
            break
        if run.materialized is not None:
            chease_inputs["qspec"] = float(run.materialized.qspec)
            chease_inputs["expeq_edge_modified_samples"] = int(np.count_nonzero(run.materialized.edge_modified))
        verdict, why, checks = validate_chease_update(run, initial=initial, policy=policy, target=target,
                                                      criteria=criteria)
        if verdict != "accepted":
            states.append(IterationState(index=index, status=verdict, message=why, validation={"chease": checks},
                                         geqdsk_path=run.refined_geqdsk, **common))
            status, message = verdict, why
            break
        from vaft.data.eqdsk import read_geqdsk

        solved = read_geqdsk(run.refined_geqdsk)
        state = _state(index, solved, Path(run.refined_geqdsk), spec, time, last, criteria=criteria,
                       validation={"chease": checks}, **common)
        states.append(state)
        if not state.accepted:
            status, message = "kinetic_generation_failed", state.message
            break
        last, base_pressure = state, psrc.relaxed_pressure
        status, message = _decide([s for s in states if s.accepted], criteria)
    final = next((s for s in reversed(states) if s.accepted), None)
    return finish(status, message, states, final)
