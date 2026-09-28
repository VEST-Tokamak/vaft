"""Flux-surface geometry of an arbitrary equilibrium, for equilibrium-aware diagrams (#1209).

One private adapter so every phenomenon diagram draws on the same surfaces,
normals and straight-field-line angle rather than rebuilding them::

    EquilibriumData (vaft.data.equilibrium)
        -> vaft.process.equilibrium.straight_field_line_map   (theta*, psi_norm, surfaces)
        -> vaft.process.equilibrium.calculate_q_profile_from_psi  (q on a rho grid)
        -> surfaces R(theta), Z(theta), theta*(theta), unit normals, q(rho)

The radial label is $\\rho = \\sqrt{\\psi_N}$. The mode phase uses $\\theta^*$
(PEST, straight field lines); the drawing uses the real $(R, Z)$ of the
surface -- the two are never identified. With no equilibrium given, the
default is the exact Solov'ev (Cerfon--Freidberg) equilibrium of
``vaft.process.equilibrium.solovev_example`` with $A = 0$ (pure pressure
source), whose $q$ rises from about 0.8 on the axis to 3 at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

#: the default Solov'ev equilibrium: a pure-pressure source (A = 0) puts q = 1 inside the plasma
_DEFAULT = {"a_parameter": 0.0}
TOPOLOGIES = ("limited", "single_null")


@dataclass(frozen=True)
class Surface:
    """One flux surface on a uniform geometric-angle grid."""

    rho: float
    R: np.ndarray
    Z: np.ndarray
    theta: np.ndarray  # geometric angle about the magnetic axis
    theta_star: np.ndarray  # PEST straight-field-line angle
    normal_R: np.ndarray  # unit normal, pointing to larger rho
    normal_Z: np.ndarray
    q: float


class EquilibriumGeometry:
    """Surfaces, normals, $\\theta^*$ and $q(\\rho)$ of one ``EquilibriumData``."""

    def __init__(self, equilibrium, *, n_theta: int = 256):
        from vaft.process.equilibrium import calculate_q_profile_from_psi, straight_field_line_map

        eq = equilibrium
        self.equilibrium = eq
        self.n_theta = int(n_theta)
        self.sfl = straight_field_line_map(eq.psi, eq.r, eq.z, eq.psi_axis, eq.psi_boundary, eq.magnetic_axis)
        self.axis = tuple(float(v) for v in self.sfl.magnetic_axis)
        rho = np.linspace(0.05, 0.98, 32)
        cocos = getattr(getattr(eq, "convention", None), "cocos", 11) or 11
        q = calculate_q_profile_from_psi(eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary,
                                         rho**2, axis_rz=eq.magnetic_axis, boundary=(eq.lcfs.r, eq.lcfs.z),
                                         cocos=cocos)
        self.rho_q = rho
        self.q_profile = np.abs(np.asarray(q, dtype=float))
        # the axis value by quadratic extrapolation in rho (q is even in rho near the axis)
        self.q0 = float(np.polyval(np.polyfit(rho[:6] ** 2, self.q_profile[:6], 1), 0.0))

    def q(self, rho) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        grid = np.concatenate([[0.0], self.rho_q])
        values = np.concatenate([[self.q0], self.q_profile])
        return np.interp(rho, grid, values)

    def rho_at_q(self, q_value: float) -> Optional[float]:
        """The innermost $\\rho$ where $q$ crosses ``q_value``, or None."""
        grid = np.concatenate([[0.0], self.rho_q])
        values = np.concatenate([[self.q0], self.q_profile]) - q_value
        idx = np.flatnonzero(np.sign(values[1:]) != np.sign(values[:-1]))
        if not idx.size:
            return None
        i = int(idx[0])
        return float(grid[i] + (grid[i + 1] - grid[i]) * values[i] / (values[i] - values[i + 1]))

    def normal(self, R, Z, h: float = 1e-5):
        """Unit normal to the flux surfaces, pointing to larger $\\psi_N$."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        dR = (self.sfl.psi_norm(R + h, Z) - self.sfl.psi_norm(R - h, Z)) / (2 * h)
        dZ = (self.sfl.psi_norm(R, Z + h) - self.sfl.psi_norm(R, Z - h)) / (2 * h)
        g = np.hypot(dR, dZ)
        return dR / g, dZ / g

    @lru_cache(maxsize=64)
    def surface(self, rho: float) -> Surface:
        rho = float(rho)
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (0, 1), not {rho!r}")
        s = self.sfl.surface(rho * rho, n_theta=self.n_theta)
        nR, nZ = self.normal(s["r"], s["z"])
        return Surface(rho, np.asarray(s["r"]), np.asarray(s["z"]), np.asarray(s["theta"]),
                       np.asarray(s["theta_star"]), nR, nZ, float(self.q(rho)))

    def point(self, rho, theta_star):
        """$(R, Z)$ on surface ``rho`` at straight-field-line angle ``theta_star``."""
        s = self.surface(float(rho))
        order = np.argsort(s.theta_star)
        ts = np.concatenate([s.theta_star[order] - 2 * np.pi, s.theta_star[order], s.theta_star[order] + 2 * np.pi])
        R = np.tile(s.R[order], 3)
        Z = np.tile(s.Z[order], 3)
        t = np.mod(np.asarray(theta_star, dtype=float), 2 * np.pi)
        return np.interp(t, ts, R), np.interp(t, ts, Z)

    @lru_cache(maxsize=1)
    def _rz_table(self):
        from scipy.interpolate import RegularGridInterpolator

        # rho = 0 is the magnetic axis itself, so the map stays regular through the centre
        rho = np.concatenate([[0.0], np.linspace(0.03, 0.97, 48)])
        ts = np.linspace(0.0, 2 * np.pi, 257)
        R = np.empty((rho.size, ts.size))
        Z = np.empty_like(R)
        R[0], Z[0] = self.axis
        for i, r in enumerate(rho[1:], start=1):
            R[i], Z[i] = self.point(r, ts)
        return (RegularGridInterpolator((rho, ts), R, method="linear"),
                RegularGridInterpolator((rho, ts), Z, method="linear"))

    def to_rz(self, rho, theta_star):
        """$(R, Z)$ at arbitrary $(\\rho, \\theta^*)$, by interpolation between tabulated surfaces."""
        rho = np.clip(np.asarray(rho, dtype=float), 0.0, 0.97)
        t = np.mod(np.asarray(theta_star, dtype=float), 2 * np.pi)
        fR, fZ = self._rz_table()
        pts = np.stack([rho, t], -1)
        return fR(pts), fZ(pts)

    @property
    def minor_radius(self) -> float:
        """Half the midplane width of the last closed surface [m]."""
        r = np.asarray(self.equilibrium.lcfs.r, dtype=float)
        return 0.5 * float(r.max() - r.min())


@lru_cache(maxsize=None)
def default_equilibrium(topology: str = "limited") -> EquilibriumGeometry:
    """The exact Solov'ev equilibrium of ``solovev_example``, $A = 0$."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, not {topology!r}")
    from vaft.process.equilibrium import solovev_example

    return EquilibriumGeometry(solovev_example(topology, **_DEFAULT))


def equilibrium_geometry(equilibrium=None) -> EquilibriumGeometry:
    """``None`` (the default Solov'ev), an ``EquilibriumData``, or an ``EquilibriumGeometry``."""
    if equilibrium is None:
        return default_equilibrium()
    if isinstance(equilibrium, EquilibriumGeometry):
        return equilibrium
    from vaft.data.equilibrium import EquilibriumData

    if isinstance(equilibrium, EquilibriumData):
        return EquilibriumGeometry(equilibrium)
    raise TypeError("equilibrium must be None, an EquilibriumData (vaft.data.equilibrium) or an EquilibriumGeometry, "
                    f"not {type(equilibrium).__name__}")
