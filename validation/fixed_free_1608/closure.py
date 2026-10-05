"""Public-API native free-boundary closure benchmark; repeated runs on servers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from omas import ODS
from vaft.code.tokamaker import TokaMakerConfig, fixed_to_free
from vaft.process.equilibrium import (
    solovev_example, solve_guazzotto_freidberg, guazzotto_freidberg_to_equilibrium,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--families', nargs='+', default=['solovev', 'guazzotto_freidberg'])
    parser.add_argument('--backends', nargs='+', default=['direct', 'tokamaker_vfixed'])
    parser.add_argument('--refine-shape', action='store_true')
    parser.add_argument('--dx', type=float, default=.04)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=False)
    records = []
    for family in args.families:
        if family == 'solovev':
            eq = solovev_example(resolution=65)
        else:
            eq = guazzotto_freidberg_to_equilibrium(solve_guazzotto_freidberg(
                'limited', inverse_aspect_ratio=.33, nu=.5, elongation=1.6,
                triangularity=.3), major_radius=.4, toroidal_field=.1, resolution=65)
        # Twelve independent one-turn rectangular PF circuits. The limiter is
        # distinct from the target LCFS and touches only its inboard midplane.
        machine = ODS(consistency_check=False)
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
            machine[f'pf_active.coil.{i}.name'] = f'PF{i+1}'
            prefix = f'pf_active.coil.{i}.element.0'
            machine[f'{prefix}.turns_with_sign'] = 1.
            for key, value in {'r': .4 + .36*np.cos(angle), 'z': .65*np.sin(angle),
                               'width': .02, 'height': .02}.items():
                machine[f'{prefix}.geometry.rectangle.{key}'] = value
        inboard = float(eq.lcfs.r.min())
        wall = ([inboard, .65, .65, inboard], [-.4, -.4, .4, .4])
        bounds = {f'PF{i+1}': (-2e5, 2e5) for i in range(12)}
        for backend in args.backends:
            config = TokaMakerConfig(profile_mode='equilibrium', dx_plasma=args.dx,
                                    dx_vacuum=.08, dx_coil=.02, limiter=wall,
                                    lim_zmax=None, neck_limiter_zmax=None, nthreads=2)
            options = {'regularization': 1e-5, 'current_bounds': bounds}
            if backend == 'direct':
                options.update(method='flux_normal', boundary_samples=64)
            result = fixed_to_free(eq, machine, args.workdir/f'{family}_{backend}',
                                   fit_backend=backend, config=config, fit_options=options,
                                   refine_shape=args.refine_shape)
            record = {'family': family, 'backend': backend, 'status': result.status,
                      'fit_status': result.fit.status, 'rms_relative_flux': result.fit.rms_relative_flux,
                      'max_relative_flux': result.fit.max_relative_flux,
                      'current_max_change_A': result.current_max_change_A,
                      'forward_error': result.forward.error, 'comparison': result.comparison,
                      'refinement_status': result.refinement_status,
                      'refined_currents_A': result.refined_currents_A,
                      'refined_error': result.refined.error if result.refined else None,
                      'refined_comparison': result.refined_comparison}
            records.append(record)
            (args.workdir/'summary.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
