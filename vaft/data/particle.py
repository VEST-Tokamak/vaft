"""SI-valued static fields and particle trajectories, independent of drawing backends.

Cartesian components are ordered x, y, z. Field Jacobians use J[i,j] = dB_i/dx_j.
Callables and field objects are runtime inputs; metadata is the serializable
provenance record, not a substitute for saving the original field source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Any
from types import MappingProxyType

import numpy as np

__all__ = [
    "ParticleField",
    "ParticleTrajectory",
    "ParticleComparison",
    "FieldEvaluationError",
]


class FieldEvaluationError(ValueError):
    """A trajectory reached a field boundary, null, or invalid field sample."""

    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise FieldEvaluationError(
            "nonfinite_field", f"{name} must be finite with shape {shape}"
        )
    return result


@dataclass(frozen=True)
class ParticleField:
    """Static Cartesian field with explicit domain and optional derivative/potential.

    E and potential must be supplied together; E = -grad(Phi) is checked at
    samples when derivatives are requested. Omitting both means E=0, Phi=0.
    Vector potential, when supplied, must use a documented gauge. For numerical
    derivatives, derivative_step in metres is mandatory; field sampling never
    extrapolates outside contains(x). Analytic Jacobians are preferred.
    """

    B: Callable
    E: Callable | None = None
    potential: Callable | None = None
    jacobian: Callable | None = None
    vector_potential: Callable | None = None
    contains: Callable | None = None
    derivative_step: float | None = None
    name: str = "static Cartesian field"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    magnetic_floor: float = 1e-14

    def __post_init__(self):
        if (self.E is None) != (self.potential is None):
            raise ValueError(
                "static E and its electrostatic potential must be supplied together"
            )
        if not callable(self.B):
            raise TypeError("B must be callable")
        if not np.isfinite(self.magnetic_floor) or self.magnetic_floor <= 0:
            raise ValueError("magnetic_floor must be positive and finite")
        if self.derivative_step is not None and (
            not np.isfinite(self.derivative_step) or self.derivative_step <= 0
        ):
            raise ValueError("derivative_step must be positive and finite")
        if self.vector_potential is not None and not self.provenance.get("gauge"):
            raise ValueError(
                "vector_potential requires a documented gauge in provenance"
            )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        for name in ("E", "potential", "jacobian", "vector_potential", "contains"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable")

    def check_position(self, x):
        x = _array(x, (3,), "position")
        if self.contains is not None and not self.contains(x):
            raise FieldEvaluationError("outside_domain", f"outside {self.name} domain")
        return x

    def magnetic(self, x):
        x = self.check_position(x)
        value = _array(self.B(x), (3,), "B")
        if np.linalg.norm(value) <= self.magnetic_floor:
            raise FieldEvaluationError(
                "magnetic_null", "magnetic field reached the configured floor"
            )
        return value

    def electric(self, x):
        x = self.check_position(x)
        return np.zeros(3) if self.E is None else _array(self.E(x), (3,), "E")

    def phi(self, x):
        x = self.check_position(x)
        value = 0.0 if self.potential is None else float(self.potential(x))
        if not np.isfinite(value):
            raise FieldEvaluationError(
                "nonfinite_field", "nonfinite electrostatic potential"
            )
        return value

    def _difference(self, function, x, h):
        columns = []
        for axis in np.eye(3):
            plus, minus = (
                self.check_position(x + h * axis),
                self.check_position(x - h * axis),
            )
            columns.append(
                (np.asarray(function(plus)) - np.asarray(function(minus))) / (2 * h)
            )
        return np.stack(columns, axis=-1)

    def derivatives(self, x):
        """Return magnitude gradient, curl(b), divergence(B), and derivative audit."""
        x = self.check_position(x)
        B = self.magnetic(x)
        strength = np.linalg.norm(B)
        if self.jacobian is not None:
            J = _array(self.jacobian(x), (3, 3), "B Jacobian")
            audit = {"method": "analytic", "relative_difference": 0.0}
        else:
            h = self.derivative_step
            if h is None:
                raise ValueError(
                    "GC requires a Jacobian or an explicit derivative_step in metres"
                )
            coarse = self._difference(self.magnetic, x, h)
            J = _array(self._difference(self.magnetic, x, h / 2), (3, 3), "B Jacobian")
            error = np.linalg.norm(J - coarse)
            scale = max(np.linalg.norm(J), strength * 1e-8 / h)
            if error > 0.05 * scale:
                raise FieldEvaluationError(
                    "derivative_unresolved",
                    "field derivative did not converge at h and h/2",
                )
            audit = {
                "method": "central_difference",
                "step_m": h,
                "relative_difference": float(error / scale),
            }
        b = B / strength
        grad = b @ J
        db = (J - np.outer(b, grad)) / strength
        curl = np.array([db[2, 1] - db[1, 2], db[0, 2] - db[2, 0], db[1, 0] - db[0, 1]])
        if self.E is not None:
            # Potential differences require a length scale even with analytic B.
            h = self.derivative_step
            if h is None:
                raise ValueError(
                    "electrostatic consistency checks require derivative_step"
                )
            grad_phi = self._difference(self.phi, x, h / 2)
            E = self.electric(x)
            if np.linalg.norm(E + grad_phi) > 1e-5 * max(
                np.linalg.norm(E), np.linalg.norm(grad_phi), 1e-12
            ):
                raise FieldEvaluationError(
                    "inconsistent_electric_field", "E is inconsistent with -grad(Phi)"
                )
        return grad, curl, float(np.trace(J)), audit

    def A(self, x):
        x = self.check_position(x)
        if self.vector_potential is None:
            raise ValueError(
                "canonical momentum requires vector_potential in a documented gauge"
            )
        return _array(self.vector_potential(x), (3,), "vector potential")

    def metadata(self):
        return {
            "name": self.name,
            "coordinates": "right-handed Cartesian x,y,z",
            "units": "SI",
            "static": True,
            "derivative_step_m": self.derivative_step,
            "magnetic_floor_T": self.magnetic_floor,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ParticleTrajectory:
    """Validated states on native solver sample times, with explicit termination.

    Full-orbit velocity is simultaneous with position. A GC record stores
    p_parallel and a model-constant mu, without inventing a full gyrovelocity.
    One initial state is retained even for immediate boundary termination.
    """

    time: np.ndarray
    position: np.ndarray
    charge: float
    mass: float
    field: ParticleField
    kind: str
    velocity: np.ndarray | None = None
    p_parallel: np.ndarray | None = None
    mu: float | None = None
    termination: str = "completed"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        t = np.asarray(self.time, float)
        if (
            t.ndim != 1
            or len(t) == 0
            or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError(
                "time must be a nonempty finite strictly increasing vector"
            )
        if (
            not np.isfinite(self.charge)
            or self.charge == 0
            or not np.isfinite(self.mass)
            or self.mass <= 0
        ):
            raise ValueError("charge must be nonzero and mass positive, both finite")
        arrays = {"time": t, "position": _array(self.position, (len(t), 3), "position")}
        if self.kind == "full_orbit":
            arrays["velocity"] = _array(self.velocity, (len(t), 3), "velocity")
        elif self.kind == "guiding_center":
            arrays["p_parallel"] = _array(self.p_parallel, (len(t),), "p_parallel")
            if self.mu is None or not np.isfinite(self.mu) or self.mu < 0:
                raise ValueError("GC requires a finite nonnegative mu")
        else:
            raise ValueError("kind must be full_orbit or guiding_center")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for name, value in arrays.items():
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def provenance_record(self):
        """Return JSON-compatible particle, initial-state and solver provenance."""
        initial = {
            "time_s": float(self.time[0]),
            "position_m": self.position[0].tolist(),
        }
        if self.kind == "full_orbit":
            initial["velocity_m_per_s"] = self.velocity[0].tolist()
        else:
            initial.update(
                p_parallel_kg_m_per_s=float(self.p_parallel[0]),
                mu_J_per_T=float(self.mu),
            )
        return {
            "charge_C": float(self.charge),
            "mass_kg": float(self.mass),
            "kind": self.kind,
            "initial_state": initial,
            "termination": self.termination,
            "field": self.field.metadata(),
            "solver": dict(self.metadata),
        }

    def sample(self, time):
        """Linearly interpolate states for presentation, with no extrapolation."""
        requested = np.atleast_1d(np.asarray(time, float))
        if requested.ndim != 1 or not np.all(np.isfinite(requested)):
            raise ValueError("display time must be a finite one-dimensional coordinate")
        if np.any(requested < self.time[0]) or np.any(requested > self.time[-1]):
            raise ValueError("display time is outside the computed trajectory")
        values = {"time": requested, "native_time": self.time}
        for name in ("position", "velocity", "p_parallel"):
            data = getattr(self, name)
            if data is None:
                continue
            if data.ndim == 1:
                values[name] = np.interp(requested, self.time, data)
            else:
                values[name] = np.column_stack(
                    [
                        np.interp(requested, self.time, data[:, j])
                        for j in range(data.shape[1])
                    ]
                )
        return values


@dataclass(frozen=True)
class ParticleComparison:
    """Full and GC runs paired on physical times, with ordering and error evidence."""

    full: ParticleTrajectory
    guiding_center: ParticleTrajectory
    time: np.ndarray
    full_center: np.ndarray
    gc_position: np.ndarray
    separation: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (
            self.full.kind != "full_orbit"
            or self.guiding_center.kind != "guiding_center"
        ):
            raise ValueError("comparison requires full-orbit and GC trajectories")
        if (
            self.full.field is not self.guiding_center.field
            or self.full.charge != self.guiding_center.charge
            or self.full.mass != self.guiding_center.mass
        ):
            raise ValueError("comparison requires the same field and particle")
        t = np.asarray(self.time, float)
        if (
            t.ndim != 1
            or len(t) == 0
            or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError("comparison time must be finite and strictly increasing")
        lo = max(self.full.time[0], self.guiding_center.time[0])
        hi = min(self.full.time[-1], self.guiding_center.time[-1])
        if t[0] < lo or t[-1] > hi:
            raise ValueError("comparison time exceeds the common trajectory interval")
        for name, shape in (
            ("time", (len(t),)),
            ("full_center", (len(t), 3)),
            ("gc_position", (len(t), 3)),
            ("separation", (len(t),)),
        ):
            value = _array(getattr(self, name), shape, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.separation < 0):
            raise ValueError("separation must be nonnegative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
