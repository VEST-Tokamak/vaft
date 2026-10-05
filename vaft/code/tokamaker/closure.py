"""Fitted PF currents held fixed in a genuine free-boundary forward solve."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from vaft.data.equilibrium import EquilibriumData
from vaft.process.equilibrium import (
    as_equilibrium, convert_cocos, derive_boundary_representation,
    derive_global_descriptors, fit_free_boundary_coils,
)
from .bridge import fit_free_boundary_coils_vfixed
from .config import TokaMakerConfig, TokaMakerResult
from .inputs import prepare_tokamaker_inputs
from .runner import _json_safe, run_tokamaker


@dataclass(frozen=True)
class FixedToFreeResult:
    """Initial inverse fit, forward solution and independent closure diagnostics."""
    fit_backend: str
    fit: Any
    forward: TokaMakerResult
    initial_currents_A: Mapping[str, float]
    realized_currents_A: Mapping[str, float]
    current_max_change_A: float | None
    comparison: Mapping[str, Any]
    status: str
    refined: TokaMakerResult | None = None
    refined_comparison: Mapping[str, Any] = field(default_factory=dict)
    refined_currents_A: Mapping[str, float] = field(default_factory=dict)
    refinement_status: str | None = None
    refinement_diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _distances(points, contour):
    a = contour
    ab = np.roll(a, -1, axis=0) - a
    length2 = np.sum(ab**2, axis=1)
    ap = points[:, None] - a[None]
    t = np.clip(np.divide(np.sum(ap * ab, axis=2), length2,
                          out=np.zeros(ap.shape[:2]), where=length2 > 0), 0, 1)
    return np.min(np.linalg.norm(ap - t[..., None] * ab, axis=2), axis=1)


def compare_equilibria(target: EquilibriumData, solved: EquilibriumData) -> dict[str, Any]:
    """Compare canonical fields using the same definitions on both records.

    LCFS distances are bidirectional vertex-to-segment distances [m], axis
    displacement is Euclidean [m], and active X-points use optimal one-to-one
    matching with unmatched counts retained. Global descriptors retain their
    definition and unavailability reasons. Neither inverse solver is used.
    """
    target, solved = convert_cocos(target, 11), convert_cocos(solved, 11)
    result: dict[str, Any] = {}
    if target.lcfs is None or solved.lcfs is None:
        result['lcfs'] = {'available': False, 'reason': 'both LCFS contours required'}
    else:
        a, b = target.lcfs.points, solved.lcfs.points
        if len(a) < 3 or len(b) < 3:
            raise ValueError('closed LCFS comparison needs three vertices')
        da, db = _distances(a, b), _distances(b, a)
        result['lcfs'] = {'available': True,
                          'rms_distance_m': float(np.sqrt((np.mean(da**2) + np.mean(db**2))/2)),
                          'max_distance_m': float(max(da.max(), db.max()))}
    result['axis_displacement_m'] = (None if target.magnetic_axis is None or solved.magnetic_axis is None
                                     else float(np.linalg.norm(np.subtract(target.magnetic_axis, solved.magnetic_axis))))
    boundaries = [derive_boundary_representation(eq) for eq in (target, solved)]
    result['topology'] = {'target': boundaries[0].topology.value,
                          'solved': boundaries[1].topology.value,
                          'target_reason': boundaries[0].reason, 'solved_reason': boundaries[1].reason}
    xs = [np.array([[x.r, x.z] for x in b.x_points if x.active]).reshape(-1, 2) for b in boundaries]
    displacement = []
    if len(xs[0]) and len(xs[1]):
        distances = np.linalg.norm(xs[0][:, None] - xs[1][None], axis=2)
        i, j = linear_sum_assignment(distances)
        displacement = distances[i, j].tolist()
    result['x_points'] = {'target_active_m': xs[0].tolist(), 'solved_active_m': xs[1].tolist(),
                          'matched_displacements_m': displacement,
                          'unmatched_target': len(xs[0]) - len(displacement),
                          'unmatched_solved': len(xs[1]) - len(displacement)}
    descriptors = [derive_global_descriptors(eq) for eq in (target, solved)]
    globals_ = {}
    for name in ('beta_p_boundary_average', 'li_virial', 'q95',
                 'thermal_energy', 'pressure_integral', 'volume'):
        entries = [d.values.get(name) for d in descriptors]
        values = [float(e.value) if e is not None and e.available and np.isfinite(e.value) else None for e in entries]
        globals_[name] = {'target': values[0], 'solved': values[1],
                         'difference': None if None in values else values[1] - values[0],
                         'definition': entries[0].definition if entries[0] is not None else None,
                         'target_reason': entries[0].reason if entries[0] is not None else 'unavailable',
                         'solved_reason': entries[1].reason if entries[1] is not None else 'unavailable'}
    # Ip is authoritative, rather than inferred from profile or Green sources.
    globals_['Ip_A'] = {'target': target.ip, 'solved': solved.ip,
                        'difference': None if target.ip is None or solved.ip is None else solved.ip - target.ip}
    result['globals'] = globals_
    return result


def fixed_to_free(
    target: EquilibriumData, machine: Any, workdir: str | Path, *,
    fit_backend: str = 'direct', config: TokaMakerConfig | None = None,
    fit_options: Mapping[str, Any] | None = None,
    refine_shape: bool = False, refinement_options: Mapping[str, Any] | None = None,
) -> FixedToFreeResult:
    """Fit PF currents, hold them fixed and solve a true free-boundary equilibrium.

    Both routes use canonical source tables by default. The direct route
    integrates the target plasma; the vfixed route obtains its vacuum target
    solely from a native fixed-boundary solve. Only coil responses are shared.
    ``workdir`` must be new; inverse, forward and comparison artefacts are
    isolated there. Solver convergence is reported separately from fit
    acceptance and closure accuracy. Shape constraints are absent in the
    initial solve. With ``refine_shape=True``, a separate native optimization
    starts from fitted currents, requires complete finite bounds and preserves
    the initial result. ``refinement_options`` controls isoflux samples, explicit
    saddle points, native constraint weights and a scaled deviation penalty.
    The refined result and status are separate; VSC is never applied.

    The adapter meshes bounding rectangles per coil half. This approximation
    is recorded in the manifest; compare against the inverse response from
    individual winding elements before treating closure errors as source errors.
    """
    if fit_backend not in ('direct', 'tokamaker_vfixed'):
        raise ValueError('fit_backend must be direct or tokamaker_vfixed')
    eq = convert_cocos(target, 11)
    cfg = config or TokaMakerConfig(profile_mode='equilibrium')
    if cfg.vsc_coil or cfg.split_coils or cfg.exclude_coils or cfg.vessel_currents:
        raise ValueError('fixed-current closure requires unchanged PF circuits and no VSC/vessel drive')
    from .profiles import profiles_for_config
    profiles_for_config(cfg, eq)  # validate sources before creating output
    if (eq.lcfs is None or not eq.lcfs.closed or eq.magnetic_axis is None
            or eq.r0 is None or eq.bt0 is None or eq.ip is None
            or not np.isfinite(eq.ip) or eq.ip <= 0):
        raise ValueError('closed target LCFS, axis, r0, bt0 and positive canonical Ip required')
    output = Path(workdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = dict(fit_options or {})
    if fit_backend == 'direct':
        fit = fit_free_boundary_coils(eq, machine, **options)
    else:
        fit = fit_free_boundary_coils_vfixed(eq, machine, output / 'fixed', config=cfg, **options)
    currents = dict(fit.currents_A)
    refinement = None
    if refine_shape:
        from .refinement import prepare_shape_refinement
        refine_options = dict(refinement_options or {})
        refine_options.setdefault('current_bounds', options.get('current_bounds', {}))
        refinement = prepare_shape_refinement(eq, currents, **refine_options)
    points = eq.lcfs.points
    radius = np.ptp(points[:, 0]) / 2
    cfg = replace(cfg, workdir=output / 'free', profile_equilibrium=eq,
                  coil_currents=currents, init_equilibrium=eq, eqdsk_cocos=7, shot=cfg.shot if cfg.shot is not None else 0,
                  time=cfg.time if cfg.time is not None else 0.,
                  ip=cfg.ip if cfg.ip is not None else eq.ip,
                  f0=cfg.f0 if cfg.f0 is not None or cfg.bt0 is not None else eq.r0 * eq.bt0,
                  init_r0=float((points[:, 0].max() + points[:, 0].min())/2),
                  init_z0=float((points[:, 1].max() + points[:, 1].min())/2),
                  init_a0=radius, init_kappa=np.ptp(points[:, 1])/(2*radius),
                  init_delta=float(((points[:, 0].max() + points[:, 0].min())/2 -
                                    (points[np.argmax(points[:, 1]), 0] + points[np.argmin(points[:, 1]), 0])/2)/radius))
    inputs = prepare_tokamaker_inputs(machine, cfg)
    forward = run_tokamaker(inputs, cfg)
    realized = dict(forward.scalars.get('coil_currents_A', {})) if forward.ok else {}
    delta = max(abs(realized[n] - currents[n]) for n in currents) if set(realized) == set(currents) else None
    comparison = {}
    status = 'solver_failed'
    if forward.ok:
        status = 'currents_changed' if delta is None or delta > 1e-6 else 'converged'
        try:
            solved = as_equilibrium(forward.gfile, convention=cfg.eqdsk_cocos)
            comparison = compare_equilibria(eq, solved)
        except Exception as exc:
            comparison = {'error': str(exc)}
            status = 'comparison_failed'
    refined = None
    refined_comparison = {}
    refined_currents = {}
    refinement_status = None
    refinement_diagnostics = {}
    if refine_shape:
        refine_cfg = replace(cfg, workdir=output / 'refined', mesh_file=inputs.mesh_file)
        if forward.ok and 'error' not in comparison:
            refine_cfg = replace(refine_cfg, init_equilibrium=solved)
        refined_inputs = prepare_tokamaker_inputs(machine, refine_cfg)
        refined = run_tokamaker(refined_inputs, refine_cfg, refinement=refinement)
        refinement_status = 'solver_failed'
        if refined.ok:
            refined_currents = dict(refined.scalars.get('coil_currents_A', {}))
            bounded = set(refined_currents) == set(currents) and all(
                refinement.current_bounds_A[n][0] - 1e-6 <= value <= refinement.current_bounds_A[n][1] + 1e-6
                for n, value in refined_currents.items())
            refinement_status = 'converged' if bounded else 'bounds_violated'
            if set(refined_currents) == set(currents):
                changes = {n: refined_currents[n] - currents[n] for n in currents}
                refinement_diagnostics = {
                    'current_changes_A': changes,
                    'max_current_change_A': max(abs(v) for v in changes.values()),
                    'scaled_current_deviation_norm': float(np.linalg.norm(list(changes.values()))/refinement.current_scale_A),
                    'active_bounds': [n for n, value in refined_currents.items() if any(
                        np.isclose(value, bound, rtol=0, atol=1e-6) for bound in refinement.current_bounds_A[n])],
                    'bounds_complete': True, 'currents_within_bounds': bool(bounded),
                    'regularization': refinement.regularization, 'current_scale_A': refinement.current_scale_A,
                }
            try:
                refined_eq = as_equilibrium(refined.gfile, convention=refine_cfg.eqdsk_cocos)
                refined_comparison = compare_equilibria(eq, refined_eq)
            except Exception as exc:
                refined_comparison = {'error': str(exc)}
                refinement_status = 'comparison_failed'
    manifest = {'fit_backend': fit_backend, 'fit_status': fit.status, 'fit_accepted': fit.accepted,
                'fit': fit.__dict__, 'initial_currents_A': currents, 'realized_currents_A': realized,
                'current_max_change_A': delta, 'status': status, 'forward_error': forward.error,
                'comparison': comparison, 'free_boundary': True, 'refine_shape': refine_shape,
                'refinement_status': refinement_status, 'refined_comparison': refined_comparison,
                'refined_currents_A': refined_currents, 'refinement_diagnostics': refinement_diagnostics,
                'refined_error': refined.error if refined else None,
                'coil_mesh_model': 'bounding rectangle per winding half',
                'profile_mode': cfg.profile_mode}
    (output / 'closure.json').write_text(json.dumps(_json_safe(manifest), indent=2), encoding='utf-8')
    return FixedToFreeResult(fit_backend, fit, forward, currents, realized, delta, comparison, status,
                             refined, refined_comparison, refined_currents, refinement_status, refinement_diagnostics)


__all__ = ['FixedToFreeResult', 'compare_equilibria', 'fixed_to_free']
