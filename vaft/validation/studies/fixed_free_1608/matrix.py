"""Server-side analytic fixed-to-free validation matrix using public VAFT APIs.

Run from the repository root with an installed OFT and a new output directory.
Each case writes its full native inputs, fit, forward and verification artefacts.
Failures remain in summary.json and do not abort later cases.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from omas import ODS
from vaft.code.tokamaker import TokaMakerConfig, fixed_to_free
from vaft.data.equilibrium import Contour
from vaft.process.equilibrium import (
    derive_boundary_representation, solovev_example, solve_guazzotto_freidberg,
    guazzotto_freidberg_to_equilibrium,
)


def _gate_inputs(record):
    """Frozen-current status and comparison from a summary.json or measured_matrix.json record.

    summary.json records are flat (``verification_status``,
    ``verified_comparison``); measured_matrix.json condenses the same solve
    into a nested ``verified`` block with the gate quantities at its top
    level. Both shapes yield the same gate inputs.
    """
    if 'verified' in record and 'verified_comparison' not in record:
        verified = record.get('verified') or {}
        comparison = {
            'lcfs': {'max_distance_m': verified.get('lcfs_max_m')},
            'axis_displacement_m': verified.get('axis_displacement_m'),
            'native_boundary': {
                'topology': verified.get('native_topology'),
                'matched_displacements_m': verified.get('x_point_displacements_m'),
                'unmatched_target': verified.get('unmatched_target_x_points'),
                'unmatched_solved': verified.get('unmatched_solved_x_points'),
            },
            'globals': {
                'q95_field_integral': verified.get('q95_field_integral'),
                'Ip_A': {'difference': verified.get('Ip_difference_A')},
            },
        }
        return verified.get('status'), comparison
    return record.get('verification_status'), record.get('verified_comparison') or {}


def acceptance_failures(record):
    """Evaluate the documented coarse-grid frozen-current benchmark gates."""
    failures = []
    if record.get('error'):
        return [record['error']]
    status, comparison = _gate_inputs(record)
    if status != 'converged':
        failures.append('frozen-current solve did not converge')
    lcfs = comparison.get('lcfs') or {}
    native = comparison.get('native_boundary') or {}
    globals_ = comparison.get('globals') or {}
    checks = (
        ('LCFS max > 12 mm', lcfs.get('max_distance_m'), .012),
        ('axis displacement > 2 mm', comparison.get('axis_displacement_m'), .002),
        ('active X displacement > 0.5 mm',
         max(native.get('matched_displacements_m') or [0.]), .0005),
        ('absolute q95 difference > 0.06',
         (globals_.get('q95_field_integral') or {}).get('difference'), .06),
        ('absolute Ip difference > 15 A',
         (globals_.get('Ip_A') or {}).get('difference'), 15.),
    )
    for label, value, limit in checks:
        if value is None or not np.isfinite(value) or abs(value) > limit:
            failures.append(label)
    if native.get('topology') != record.get('topology'):
        failures.append('native X-point topology mismatch')
    if native.get('unmatched_target') or native.get('unmatched_solved'):
        failures.append('native active X-point count mismatch')
    return failures


def target_case(family: str, topology: str, *, pedestal: float = 0.,
                nu: float = .5, resolution: int = 65):
    """Construct the named analytic target with the same VEST-scale R0/B0."""
    if family == 'solovev':
        eq = solovev_example('single_null' if topology == 'lower_single_null' else topology,
                             resolution=resolution)
    elif family in ('guazzotto_freidberg', 'guazzotto_pedestal'):
        options = dict(inverse_aspect_ratio=.33, nu=nu, elongation=1.6,
                       triangularity=.3, current_pedestal=pedestal)
        if topology != 'limited':
            options.update(x_point_elongation=2., x_point_triangularity=.5)
        model = solve_guazzotto_freidberg(topology, **options)
        eq = guazzotto_freidberg_to_equilibrium(
            model, major_radius=.4, toroidal_field=.1, resolution=resolution)
    else:
        raise ValueError(f'unknown family {family}')
    return eq


def machine_case(eq, topology):
    """Twelve independent one-turn PF circuits and a physical limiter."""
    machine = ODS(consistency_check=False)
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
        machine[f'pf_active.coil.{i}.name'] = f'PF{i+1}'
        prefix = f'pf_active.coil.{i}.element.0'
        machine[f'{prefix}.turns_with_sign'] = 1.
        for key, value in {'r': .4+.36*np.cos(angle), 'z': .65*np.sin(angle),
                           'width': .02, 'height': .02}.items():
            machine[f'{prefix}.geometry.rectangle.{key}'] = value
    if topology == 'limited':
        left = float(eq.lcfs.r.min())
        wall = ([left, .65, .65, left], [-.5, -.5, .5, .5])
    else:
        left = max(.03, float(eq.lcfs.r.min())-.04)
        top = max(.55, float(np.abs(eq.lcfs.z).max())+.06)
        wall = ([left, .65, .65, left], [-top, -top, top, top])
    if eq.limiter is None:
        eq = replace(eq, limiter=Contour(np.asarray(wall[0]), np.asarray(wall[1]), True))
    return eq, machine, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--families', nargs='+', default=['solovev', 'guazzotto_freidberg'])
    parser.add_argument('--topologies', nargs='+', default=['limited', 'lower_single_null', 'double_null'])
    parser.add_argument('--backends', nargs='+', default=['direct', 'tokamaker_vfixed'])
    parser.add_argument('--current-pedestal', type=float, default=.1)
    parser.add_argument('--nu', type=float, default=.5)
    parser.add_argument('--resolution', type=int, default=65)
    parser.add_argument('--dx', type=float, default=.04)
    parser.add_argument('--no-refinement', action='store_true')
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=False)
    records = []
    for family in args.families:
        for topology in args.topologies:
            try:
                target = target_case(family, topology,
                                     pedestal=args.current_pedestal if family == 'guazzotto_pedestal' else 0.,
                                     nu=args.nu, resolution=args.resolution)
                target, machine, wall = machine_case(target, topology)
                target_error = None
            except Exception as exc:
                target_error = f'{type(exc).__name__}: {exc}'
            for backend in args.backends:
                name = f'{family}_{topology}_{backend}'
                record = {'family': family, 'topology': topology, 'backend': backend,
                          'dx_plasma_m': args.dx, 'target_grid': args.resolution}
                try:
                    if target_error:
                        raise ValueError(f'target construction failed: {target_error}')
                    cfg = TokaMakerConfig(profile_mode='equilibrium', dx_plasma=args.dx,
                                          dx_vacuum=.08, dx_coil=.02, limiter=wall,
                                          lim_zmax=None, neck_limiter_zmax=None, nthreads=2)
                    fit_options = {'regularization': 1e-5,
                                   'current_bounds': {f'PF{i+1}': (-2e5, 2e5) for i in range(12)}}
                    if backend == 'direct':
                        x_points = [(x.r, x.z) for x in derive_boundary_representation(target).x_points if x.active]
                        fit_options.update(method='flux_normal', boundary_samples=64, x_points=x_points)
                    result = fixed_to_free(target, machine, args.workdir/name,
                                           fit_backend=backend, config=cfg,
                                           fit_options=fit_options,
                                           refine_shape=not args.no_refinement,
                                           verify_refinement=not args.no_refinement)
                    record.update(fit_status=result.fit.status, fit_accepted=result.fit.accepted,
                                  rms_relative_flux=result.fit.rms_relative_flux,
                                  max_relative_flux=result.fit.max_relative_flux,
                                  condition_number=result.fit.condition_number,
                                  initial_status=result.status, initial_comparison=result.comparison,
                                  refinement_status=result.refinement_status,
                                  refined_comparison=result.refined_comparison,
                                  verification_status=result.verification_status,
                                  verified_comparison=result.verified_comparison,
                                  refined_current_max_change_A=result.refinement_diagnostics.get('max_current_change_A'),
                                  initial_error=result.forward.error,
                                  refined_error=result.refined.error if result.refined else None,
                                  verification_error=result.verified_free.error if result.verified_free else None)
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                if not args.no_refinement:
                    record['acceptance_failures'] = acceptance_failures(record)
                    record['accepted'] = not record['acceptance_failures']
                records.append(record)
                (args.workdir/'summary.json').write_text(json.dumps(records, indent=2))
                print(json.dumps({key: record.get(key) for key in (
                    'family', 'topology', 'backend', 'fit_status', 'initial_status',
                    'refinement_status', 'verification_status', 'error')}), flush=True)
    if any(record.get('error') or record.get('acceptance_failures') for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
