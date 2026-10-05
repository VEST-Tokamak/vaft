"""Optional native shape optimization, kept separate from initial PF fitting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from vaft.process.equilibrium import derive_boundary_representation


@dataclass(frozen=True)
class ShapeRefinement:
    """Native isoflux/saddle constraints and scaled current deviation penalty.

    Points are R/Z metres, currents and bounds are physical amperes. Native
    OFT constraint weights are separate from the initial inverse-fit weights.
    Regularization penalizes deviation from initial fitted currents, rather
    than replacing the initial fit. Complete finite bounds are mandatory.
    """
    isoflux_points_m: np.ndarray
    saddle_points_m: np.ndarray
    reference_currents_A: Mapping[str, float]
    current_bounds_A: Mapping[str, tuple[float, float]]
    regularization: float = 1e-5
    current_scale_A: float = 1000.
    isoflux_weight: float = 1.
    saddle_weight: float = 1.

    def __post_init__(self):
        for name, minimum in (('isoflux_points_m', 4), ('saddle_points_m', 0)):
            points = np.asarray(getattr(self, name), dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum
                    or not np.isfinite(points).all() or np.any(points[:, 0] <= 0)):
                raise ValueError(f'{name} must be finite positive-R (n,2) points')
            object.__setattr__(self, name, points.copy())
        values = [self.regularization, self.current_scale_A, self.isoflux_weight, self.saddle_weight]
        if not np.isfinite(values).all() or np.any(np.asarray(values) <= 0):
            raise ValueError('refinement controls must be finite positive')
        if not self.reference_currents_A or set(self.reference_currents_A) != set(self.current_bounds_A):
            raise ValueError('complete nonempty PF current bounds required')
        for name, current in self.reference_currents_A.items():
            bounds = self.current_bounds_A[name]
            if (len(bounds) != 2 or not np.isfinite([current, *bounds]).all()
                    or bounds[0] >= bounds[1] or not bounds[0] <= current <= bounds[1]):
                raise ValueError(f'initial current and bounds invalid for {name}')


def prepare_shape_refinement(target, currents, *, current_bounds, boundary_samples=64,
                             x_points=None, regularization=1e-5, current_scale_A=1000.,
                             isoflux_weight=1., saddle_weight=1.) -> ShapeRefinement:
    """Resample LCFS by arc length and validate physical optimizer controls."""
    if boundary_samples < 4 or int(boundary_samples) != boundary_samples:
        raise ValueError('boundary_samples must be an integer >=4')
    values = np.array([regularization, current_scale_A, isoflux_weight, saddle_weight])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('refinement penalty, current scale and weights must be positive finite')
    if set(current_bounds) != set(currents):
        raise ValueError('refinement requires complete bounds for the initial PF circuits')
    for name, bounds in current_bounds.items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f'finite lower < upper bounds required for {name}')
        if not np.isfinite(currents[name]) or not bounds[0] <= currents[name] <= bounds[1]:
            raise ValueError(f'initial current must lie inside refinement bounds for {name}')
    if target.lcfs is None or not target.lcfs.closed:
        raise ValueError('closed target LCFS required for isoflux refinement')
    p = target.lcfs.points
    if np.array_equal(p[0], p[-1]):
        p = p[:-1]
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    keep = lengths > 0
    p = p[keep]
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    if len(p) < 3 or not np.isfinite(p).all() or np.any(p[:, 0] <= 0) or lengths.sum() == 0:
        raise ValueError('valid nondegenerate positive-R LCFS required')
    distances = np.r_[0., np.cumsum(lengths)]
    samples = np.linspace(0, distances[-1], int(boundary_samples), endpoint=False)
    closed = np.vstack([p, p[0]])
    boundary = np.column_stack([np.interp(samples, distances, closed[:, k]) for k in range(2)])
    if x_points is None:
        representation = derive_boundary_representation(target)
        x_points = [(x.r, x.z) for x in representation.x_points if x.active]
    saddles = np.asarray(x_points, dtype=float).reshape(-1, 2)
    if not np.isfinite(saddles).all() or np.any(saddles[:, 0] <= 0):
        raise ValueError('finite positive-R saddle points required')
    return ShapeRefinement(boundary, saddles, dict(currents), dict(current_bounds),
                           regularization, current_scale_A, isoflux_weight, saddle_weight)


def apply_shape_refinement(solver, refinement: ShapeRefinement) -> dict[str, Any]:
    """Apply installed native API set_isoflux/set_saddles and physical bounds."""
    solver.set_isoflux(refinement.isoflux_points_m,
                       weights=np.full(len(refinement.isoflux_points_m), refinement.isoflux_weight))
    if len(refinement.saddle_points_m):
        solver.set_saddles(refinement.saddle_points_m,
                          weights=np.full(len(refinement.saddle_points_m), refinement.saddle_weight))
    solver.set_coil_bounds(dict(refinement.current_bounds_A))
    terms = [solver.coil_reg_term({name: 1/refinement.current_scale_A},
                                 target=current/refinement.current_scale_A,
                                 weight=refinement.regularization)
             for name, current in refinement.reference_currents_A.items()]
    solver.set_coil_reg(reg_terms=terms)
    return {**refinement.__dict__, 'native_methods': ['set_isoflux', 'set_saddles', 'set_coil_bounds', 'set_coil_reg'],
            'penalty_definition': 'OFT weight on (I-I_initial)/current_scale_A'}


__all__ = ['ShapeRefinement', 'prepare_shape_refinement']
