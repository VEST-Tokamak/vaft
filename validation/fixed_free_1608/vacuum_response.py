"""Independently compare a native unit PF coil to full-Wb exact Green flux."""
import argparse
import json
import numpy as np
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import eval_green
from vaft.formula.green import green_psi_exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True)
    args = parser.parse_args()
    solver = TokaMaker(OFT_env(nthreads=2))
    pts, cells, reg, coils, cond = load_gs_mesh(args.mesh)
    try:
        solver.setup_mesh(pts, cells, reg)
        solver.setup_regions(cond_dict=cond, coil_dict=coils)
        solver.setup(order=2, F0=.04)
        solver.set_coil_currents({'PF1': 1.})
        vacuum = solver.vac_solve()
        if hasattr(vacuum, 'get_field_eval'):
            field = vacuum.get_field_eval('psi')
        else:
            solver.set_psi(vacuum)
            field = solver.get_field_eval('psi')
        points = np.array([[.4, 0.], [.45, .02]])
        native = np.array([float(np.asarray(field.eval(p)).ravel()[0]) for p in points])
        # PF1 in closure.py is a one-turn 0.02m square centred at (.76,0).
        nodes, weights = np.polynomial.legendre.leggauss(5)
        exact = sum(wr*wz/4*green_psi_exact(points[:, 0], points[:, 1], .76+.01*r, .01*z)
                    for r, wr in zip(nodes, weights) for z, wz in zip(nodes, weights))
        result = {'points_m': points.tolist(), 'native_FEM_psi_per_rad': native.tolist(),
                  'exact_full_Wb': exact.tolist(),
                  'relative_FEM_error': (2*np.pi*native/exact - 1).tolist(),
                  'mathematical_eval_green': eval_green(points, np.array([.76, 0.])).tolist()}
        print(json.dumps(result))
        np.testing.assert_allclose(2*np.pi*native, exact, rtol=.01)
    finally:
        solver.reset()


if __name__ == '__main__':
    main()
