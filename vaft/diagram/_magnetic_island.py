"""Magnetic-island schematic: one physical model, three projections.

Every projection is drawn from one :class:`IslandModel`, and the model asks
:mod:`vaft.formula.stability` for the physics -- the helical phase, the
pendulum flux function and the separatrix -- rather than restating it. What
lives here is only what a picture needs: sampling, the circular (or
elliptic) flux-surface embedding, the camera and the labels.

Lengths are normalised to the plasma minor radius ``a``: the LCFS is
``r = 1``, the rational surface ``r = r_s`` and the magnetic axis sits at
``R = R0 = aspect_ratio``. A point ``(r, theta, phi)`` is placed on the
Miller-shaped surface::

    R = R0 + r cos(theta + arcsin(delta(r)) sin(theta)),   Z = kappa r sin(theta)
    X = R cos(phi),                                         Y = R sin(phi)

with ``kappa = elongation`` on every surface and ``delta(r) = triangularity * r``,
so the shaping grows from a circle at the axis to the requested D at the
LCFS and the surfaces stay nested. ``theta`` runs from the outboard midplane
towards the top and ``phi`` counter-clockwise seen from above -- the
convention :func:`vaft.formula.stability.helical_phase` documents.

``theta`` is only the parametrisation angle. The helical phase is evaluated
in the straight-field-line (PEST) angle ``theta*`` of each surface, from
:func:`vaft.formula.equilibrium.straight_field_line_angle`: field lines are
straight in ``(theta*, phi)``, so O- and X-points evenly spaced in ``theta*``
spread out on the low-field side, as they do in a real torus -- even for
circular surfaces. The surfaces are prescribed (no Shafranov shift), so
``theta*`` is exact for them rather than for a Grad-Shafranov equilibrium.

``r`` is a flux-surface label; a width in ``r`` is a physical distance only
on the outboard midplane (``theta = 0``), where the embedding reduces to
``R = R0 + r``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import numpy as np

from vaft.formula.equilibrium import straight_field_line_angle
from vaft.formula.stability import (
    helical_phase,
    island_pendulum_hamiltonian,
    island_separatrix_half_width,
)

from ._projection import CAMERA_AZIMUTH, CAMERA_ELEVATION, THREE_D_SCALE  # noqa: F401  (re-exported)
from ._projection import camera as _camera
from ._projection import project as _project
from ._projection import split as _split
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

PROJECTIONS = ("poloidal", "top", "3d")

#: centimetres per minor radius in each projection
POLOIDAL_SCALE = 4.0
TOP_SCALE = 1.3
#: island flux contours, as fractions of the way from the O-point to the separatrix
ISLAND_LEVELS = (0.2, 0.45, 0.72)
#: the passing surface drawn on each side, as a multiple of the separatrix level
PASSING_LEVEL = 2.2

_N_THETA = 241
#: |delta| at the LCFS; the Miller map with delta(r) = delta r folds (its
#: Jacobian changes sign) from |delta| of about 0.93
MAX_TRIANGULARITY = 0.9
#: grid on which theta*(r, theta) is tabulated and interpolated
_SFL_R = np.linspace(0.0025, 1.0, 400)
_SFL_THETA = np.linspace(0.0, 2.0 * np.pi, 1025)


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IslandModel:
    """The single parameter set every projection of an island is drawn from."""

    m: int
    n: int
    width: float
    phase: float
    r_s: float
    aspect_ratio: float
    elongation: float
    triangularity: float = 0.0

    # --- straight-field-line angle ------------------------------------------

    def jacobian(self, r, theta):
        """Signed Jacobian of ``(r, theta, phi)``, ``R d(R, Z)/d(r, theta)``."""
        h = 1e-6
        d_r = (self.section(r + h, theta) - self.section(r - h, theta)) / (2 * h)
        d_t = (self.section(r, theta + h) - self.section(r, theta - h)) / (2 * h)
        R = self.major_radius + self.section(r, theta)[..., 0]
        return R * (d_r[..., 0] * d_t[..., 1] - d_r[..., 1] * d_t[..., 0])

    @cached_property
    def _sfl_table(self) -> np.ndarray:
        """``theta*(r, theta) - theta`` on the ``(_SFL_R, _SFL_THETA)`` grid (periodic in theta)."""
        rows = []
        for r in _SFL_R:
            R = self.major_radius + self.section(r, _SFL_THETA)[:, 0]
            star = straight_field_line_angle(_SFL_THETA, self.jacobian(r, _SFL_THETA), R)
            rows.append(star - _SFL_THETA)
        return np.array(rows)

    def _sfl_offset(self, r, angle, table) -> np.ndarray:
        """Bilinear lookup of a periodic offset table at ``(r, angle)``."""
        r, angle = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(angle, dtype=float))
        wrapped = np.mod(angle, 2.0 * np.pi)
        step = _SFL_THETA[1] - _SFL_THETA[0]
        j = np.clip((wrapped / step).astype(int), 0, len(_SFL_THETA) - 2)
        fj = wrapped / step - j
        ri = np.clip(r, _SFL_R[0], _SFL_R[-1])
        i = np.clip(np.searchsorted(_SFL_R, ri) - 1, 0, len(_SFL_R) - 2)
        fi = (ri - _SFL_R[i]) / (_SFL_R[i + 1] - _SFL_R[i])
        lower = table[i, j] * (1 - fj) + table[i, j + 1] * fj
        upper = table[i + 1, j] * (1 - fj) + table[i + 1, j + 1] * fj
        return lower * (1 - fi) + upper * fi

    @cached_property
    def _sfl_inverse_table(self) -> np.ndarray:
        """``theta - theta*`` tabulated against an even ``theta*`` grid, per ``r``."""
        rows = []
        for offset in self._sfl_table:
            star = _SFL_THETA + offset
            rows.append(np.interp(_SFL_THETA, star, _SFL_THETA) - _SFL_THETA)
        return np.array(rows)

    def theta_star(self, r, theta) -> np.ndarray:
        """Straight-field-line angle of the point ``(r, theta)``."""
        return np.asarray(theta, dtype=float) + self._sfl_offset(r, theta, self._sfl_table)

    def theta_from_star(self, r, theta_star) -> np.ndarray:
        """The parametrisation angle whose straight-field-line angle is ``theta_star``."""
        return np.asarray(theta_star, dtype=float) + self._sfl_offset(r, theta_star, self._sfl_inverse_table)

    # --- helical structure -------------------------------------------------

    def xi(self, theta, phi, r=None):
        """Helical phase at ``(r, theta, phi)`` (``r`` defaults to ``r_s``), from the formula layer."""
        r = self.r_s if r is None else r
        return helical_phase(self.theta_star(r, theta), phi, self.m, self.n, self.phase)

    def theta_at(self, xi, phi, branch: int = 0, r=None):
        """The parametrisation angle where the helical phase is ``xi`` on section ``phi``.

        ``branch`` selects which of the ``m`` solutions (``xi + 2 pi branch``);
        ``r`` (default ``r_s``) is the surface the angle is taken on, since
        ``theta*`` differs from surface to surface.
        """
        r = self.r_s if r is None else r
        star = (np.asarray(xi, dtype=float) + 2.0 * np.pi * branch
                + self.n * np.asarray(phi, dtype=float) + self.phase) / self.m
        return self.theta_from_star(r, star)

    def o_theta(self, phi=0.0) -> np.ndarray:
        """Parametrisation angles of the ``m`` O-points on the section at ``phi``."""
        return np.array([self.theta_at(0.0, phi, k) for k in range(self.m)])

    def x_theta(self, phi=0.0) -> np.ndarray:
        """Parametrisation angles of the ``m`` X-points on the section at ``phi``."""
        return np.array([self.theta_at(np.pi, phi, k) for k in range(self.m)])

    # --- flux function -----------------------------------------------------

    @property
    def o_level(self) -> float:
        return float(island_pendulum_hamiltonian(0.0, 0.0, self.width))

    @property
    def separatrix_level(self) -> float:
        return float(island_pendulum_hamiltonian(0.0, np.pi, self.width))

    def level_offset(self, level, xi):
        """``|x|`` where the flux function equals ``level`` at phase ``xi``.

        ``H(x, xi) - H(0, xi)`` is ``x**2 / 2`` for this model, so the level
        set is inverted through the formula itself; phases the level never
        reaches return ``nan``.
        """
        gap = level - island_pendulum_hamiltonian(0.0, xi, self.width)
        with np.errstate(invalid="ignore"):
            return np.where(gap >= 0.0, np.sqrt(2.0 * np.maximum(gap, 0.0)), np.nan)

    def separatrix_offset(self, xi):
        return island_separatrix_half_width(xi, self.width)

    # --- embedding ---------------------------------------------------------

    @property
    def major_radius(self) -> float:
        return float(self.aspect_ratio)

    def section(self, r, theta) -> np.ndarray:
        """``(R - R0, Z)`` of poloidal-section points, shape ``(..., 2)``.

        Miller shaping with ``delta(r) = triangularity * r``; a circle when
        both shaping parameters are at their defaults.
        """
        r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
        shift = np.arcsin(self.triangularity * r) * np.sin(theta)
        return np.stack([r * np.cos(theta + shift), self.elongation * r * np.sin(theta)], axis=-1)

    def cartesian(self, r, theta, phi) -> np.ndarray:
        """``(X, Y, Z)`` of torus points, shape ``(..., 3)``."""
        r, theta, phi = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (r, theta, phi)))
        sec = self.section(r, theta)
        R = self.major_radius + sec[..., 0]
        return np.stack([R * np.cos(phi), R * np.sin(phi), sec[..., 1]], axis=-1)

    def contains(self, r, R, Z) -> np.ndarray:
        """Whether ``(R, Z)`` lies strictly inside the surface of label ``r``.

        At height ``Z`` the surface is crossed at ``theta_1 = arcsin(Z / (kappa r))``
        and ``pi - theta_1``; the interior is the ``R`` interval between them.
        """
        R, Z = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(Z, dtype=float))
        s = Z / (self.elongation * r)
        inside = np.abs(s) < 1.0
        s = np.clip(s, -1.0, 1.0)
        theta1 = np.arcsin(s)
        a = np.arcsin(self.triangularity * r) * s
        R1 = self.major_radius + r * np.cos(theta1 + a)
        R2 = self.major_radius + r * np.cos(np.pi - theta1 + a)
        return inside & (R > np.minimum(R1, R2)) & (R < np.maximum(R1, R2))

    def locus(self, kind: str, phi) -> Tuple[np.ndarray, np.ndarray]:
        """``(theta, phi)`` along the O (``xi = 0``) or X (``xi = pi``) helix.

        One branch followed for ``m`` toroidal turns traces the whole closed
        helix when ``gcd(m, n) = 1``.
        """
        xi0 = {"o": 0.0, "x": np.pi}[kind]
        phi = np.asarray(phi, dtype=float)
        return self.theta_at(xi0, phi), phi


def _validate(m, n, width, phase, projection, r_s, aspect_ratio, elongation,
              triangularity=0.0) -> IslandModel:
    for name, value in (("m", m), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer mode number, not {value!r}")
    if math.gcd(int(m), int(n)) != 1:
        raise ValueError(f"m/n = {m}/{n} is not in lowest terms; use {m // math.gcd(m, n)}/{n // math.gcd(m, n)}")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, not {projection!r}")
    r_s = float(r_s)
    if not 0.0 < r_s < 1.0:
        raise ValueError(f"r_s is the rational-surface radius in units of the minor radius and must lie in (0, 1), not {r_s!r}")
    width = float(width)
    limit = 2.0 * min(r_s, 1.0 - r_s)
    if not 0.0 < width < limit:
        raise ValueError(
            f"width is the full island width in units of the minor radius and must lie in (0, {limit:.4g}) "
            f"so the island stays inside the plasma and off the axis at r_s = {r_s:g}; got {width!r}"
        )
    if not math.isfinite(float(phase)):
        raise ValueError(f"phase must be finite, not {phase!r}")
    if not float(aspect_ratio) > 1.0:
        raise ValueError(f"aspect_ratio must exceed 1, not {aspect_ratio!r}")
    if not float(elongation) > 0.0:
        raise ValueError(f"elongation must be positive, not {elongation!r}")
    if not abs(float(triangularity)) <= MAX_TRIANGULARITY:
        raise ValueError(
            f"triangularity must lie in [-{MAX_TRIANGULARITY}, {MAX_TRIANGULARITY}]; beyond about 0.93 the "
            f"delta(r) = delta r surfaces fold over and stop being nested, not {triangularity!r}"
        )
    return IslandModel(int(m), int(n), width, float(phase), r_s, float(aspect_ratio), float(elongation),
                       float(triangularity))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _coef(k: int, symbol: str) -> str:
    return symbol if k == 1 else f"{k}{symbol}"


def _q_text(model: IslandModel) -> str:
    return f"q={model.m}" if model.n == 1 else f"q={model.m}/{model.n}"


def _xi_text(model: IslandModel) -> str:
    return "\\xi=" + _coef(model.m, "\\theta^*") + "-" + _coef(model.n, "\\phi") + "-\\phi_0"


def _phase_text(model: IslandModel) -> str:
    return f"\\phi_0={model.phase:g}"


def _title(model: IslandModel, y: float, subtitle: str) -> List:
    return [
        Label((0.0, y), f"$m/n={model.m}/{model.n}$ magnetic island", "title", role="title"),
        Label((0.0, y - 0.8), subtitle, "subtitle", role="title"),
    ]


def _leader(text: str, text_at, target, anchor: str, role: str, style: str = "label") -> List:
    tx, ty = text_at
    gap = 0.08 if anchor == "west" else -0.08
    return [
        Arrow((tx - gap, ty), tuple(target), "leader", role=role),
        Label((tx, ty), text, style, anchor=anchor, role=role),
    ]


# ---------------------------------------------------------------------------
# Poloidal cross-section
# ---------------------------------------------------------------------------


def _closed_contour(model: IslandModel, level: float, branch: int, n: int = 121) -> np.ndarray:
    """``(r, theta)`` of the closed island contour at ``level`` around one O-point."""
    ratio = np.clip((level - model.o_level) / (model.separatrix_level - model.o_level), 0.0, 1.0)
    # The level set -A cos(xi) <= level reaches |xi| <= arccos(-level / A).
    xi_max = float(np.arccos(np.clip(-level / model.separatrix_level, -1.0, 1.0))) if ratio < 1 else np.pi
    t = np.linspace(0.0, 2.0 * np.pi, n)
    xi = xi_max * np.cos(t)  # clusters samples at the turning points
    x = np.sign(np.sin(t)) * np.nan_to_num(model.level_offset(level, xi))
    return np.stack([model.r_s + x, model.theta_at(xi, 0.0, branch, r=model.r_s + x)], axis=-1)


def _separatrix_lobe(model: IslandModel, branch: int, n: int = 181) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    xi = np.pi * np.cos(t)
    x = np.sign(np.sin(t)) * model.separatrix_offset(xi)
    return np.stack([model.r_s + x, model.theta_at(xi, 0.0, branch, r=model.r_s + x)], axis=-1)


def _poloidal_geometry(model: IslandModel, *, show_rational_surface=True, show_separatrix=True,
                       show_o_points=True, show_x_points=True) -> List:
    theta = np.linspace(0.0, 2.0 * np.pi, _N_THETA)
    sec = model.section
    items: List = []

    items.append(Polyline.of(sec(1.0, theta), "lcfs", role="lcfs", closed=True))
    for r in np.linspace(0.14, 0.94, 7):
        if abs(r - model.r_s) > 0.9 * model.width:
            items.append(Polyline.of(sec(r, theta), "surface", role="surface", closed=True))

    xi = np.linspace(0.0, 2.0 * np.pi * model.m, 120 * model.m + 1)
    passing = model.level_offset(PASSING_LEVEL * model.separatrix_level, xi)
    for sign in (1.0, -1.0):
        r = model.r_s + sign * passing
        if r.min() <= 0.02 or r.max() >= 0.98:
            continue  # a wide island leaves no room for this surface inside the plasma
        items.append(Polyline.of(sec(r, model.theta_at(xi, 0.0, 0, r=r)), "passing", role="passing", closed=True))

    if show_rational_surface:
        items.append(Polyline.of(sec(model.r_s, theta), "rational", role="rational", closed=True))

    for branch in range(model.m):
        for frac in ISLAND_LEVELS:
            level = model.o_level + frac * (model.separatrix_level - model.o_level)
            rt = _closed_contour(model, level, branch)
            items.append(Polyline.of(sec(rt[:, 0], rt[:, 1]), "island", role="island", closed=True))
    if show_separatrix:
        for branch in range(model.m):
            rt = _separatrix_lobe(model, branch)
            items.append(Polyline.of(sec(rt[:, 0], rt[:, 1]), "separatrix", role="separatrix", closed=True))

    if show_o_points:
        for th in model.o_theta(0.0):
            items.append(Marker(tuple(sec(model.r_s, th)), "o", "opoint", role="o_point"))
    if show_x_points:
        for th in model.x_theta(0.0):
            items.append(Marker(tuple(sec(model.r_s, th)), "x", "xpoint", role="x_point"))
    return items


def _poloidal(model: IslandModel, *, labels=True, **show) -> Scene:
    items = _poloidal_geometry(model, **show)
    sec = model.section
    r_in, r_out = model.r_s - model.width / 2, model.r_s + model.width / 2
    inner = sec(r_in, model.theta_at(0.0, 0.0, 0, r=r_in))
    outer = sec(r_out, model.theta_at(0.0, 0.0, 0, r=r_out))
    items.append(Arrow(tuple(inner), tuple(outer), "width arrow", role="width"))

    if labels:
        along = np.asarray(outer) - np.asarray(inner)
        along /= np.linalg.norm(along)
        # past the outer arrowhead, clear of the island's own contours
        items.append(Label(tuple(np.asarray(outer) + 0.05 * along), "$w$", "label",
                           anchor="west" if along[0] >= 0 else "east", role="width"))

    scene = Scene(tuple(items)).transformed(POLOIDAL_SCALE)
    if not labels:
        return scene

    S, k = POLOIDAL_SCALE, model.elongation
    col = 1.22 * S
    top = k * S
    ann: List = []
    ann += _title(model, top + 1.55,
                  f"$\\mathcal{{H}}=\\tfrac12x^2-(w/4)^2\\cos\\xi$, "
                  f"${_xi_text(model)}$; "
                  f"section at $\\phi=0$, ${_phase_text(model)}$")

    def at(r, th):
        return S * sec(r, th)

    lcfs_t = at(1.0, math.pi / 4)
    ann += _leader("LCFS", (col, max(lcfs_t[1], 0.0) + 0.9), lcfs_t, "west", "label")
    q_theta = float(model.theta_at(np.pi / 2, 0.0, 0))
    q_t = at(model.r_s, q_theta)
    side = 1 if q_t[0] >= 0 else -1
    ann += _leader(f"${_q_text(model)}$ rational surface (unperturbed)", (side * col, q_t[1] + 0.35), q_t,
                   "west" if side > 0 else "east", "label")
    o_t = at(model.r_s, float(model.o_theta(0.0)[-1]))
    side = 1 if o_t[0] >= 0 else -1
    ann += _leader("O-point ($\\xi=0$)", (side * col, o_t[1] - 0.6), o_t, "west" if side > 0 else "east", "label")
    th_x = float(model.x_theta(0.0)[0])
    x_t = at(model.r_s, th_x)
    side = 1 if x_t[0] > 1e-9 else -1
    ann += _leader("X-point ($\\xi=\\pi$)", (side * col, x_t[1] + 0.25), x_t, "west" if side > 0 else "east", "label")

    ax0 = (0.9 * S, -top - 0.55)
    ann += [
        Arrow(ax0, (ax0[0] + 0.9, ax0[1]), "axis", role="axes"),
        Label((ax0[0] + 0.95, ax0[1]), "$R$", anchor="west", role="axes"),
        Arrow(ax0, (ax0[0], ax0[1] + 0.9), "axis", role="axes"),
        Label((ax0[0], ax0[1] + 0.95), "$Z$", anchor="south", role="axes"),
        Label((0.0, -top - 1.1),
              f"$m={model.m}$ islands per poloidal section; "
              "O- and X-points alternate in helical phase", "note", role="note"),
    ]
    return scene + Scene(tuple(ann))


# ---------------------------------------------------------------------------
# Top view (projection onto the midplane)
# ---------------------------------------------------------------------------


def _circle(radius: float, n: int = _N_THETA) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([radius * np.cos(t), radius * np.sin(t)], axis=-1)


def _top_locus(model: IslandModel, kind: str) -> np.ndarray:
    """Midplane projection ``(X, Y)`` of a helix, in minor radii."""
    phi = np.linspace(0.0, 2.0 * np.pi * model.m, 240 * model.m + 1)
    theta, phi = model.locus(kind, phi)
    return model.cartesian(model.r_s, theta, phi)[:, :2]


def _top(model: IslandModel, *, labels=True, show_rational_surface=True, show_o_points=True,
         show_x_points=True, **_ignored) -> Scene:
    R0, S = model.major_radius, TOP_SCALE
    items: List = [
        Polyline.of(_circle(R0 + 1.0), "machine fill", role="plasma_fill", closed=True),
        Polyline.of(_circle(R0 - 1.0), "hole", role="plasma_fill", closed=True),
        Polyline.of(_circle(R0 + 1.0), "lcfs", role="lcfs", closed=True),
        Polyline.of(_circle(R0 - 1.0), "lcfs", role="lcfs", closed=True),
        Polyline.of(_circle(R0), "surface", role="axis", closed=True),
    ]
    if show_rational_surface:
        for r in (R0 + model.r_s, R0 - model.r_s):
            items.append(Polyline.of(_circle(r), "rational", role="rational", closed=True))
    items.append(Polyline.of([(R0 - 1.15, 0.0), (R0 + 1.15, 0.0)], "cut", role="cut"))
    items.append(Polyline.of(_top_locus(model, "x"), "x locus", role="x_locus"))
    items.append(Polyline.of(_top_locus(model, "o"), "o locus", role="o_locus"))
    if show_o_points:
        for th in model.o_theta(0.0):
            items.append(Marker(tuple(model.cartesian(model.r_s, th, 0.0)[:2]), "o", "opoint", role="o_point"))
    if show_x_points:
        for th in model.x_theta(0.0):
            items.append(Marker(tuple(model.cartesian(model.r_s, th, 0.0)[:2]), "x", "xpoint", role="x_point"))
    scene = Scene(tuple(items)).transformed(S)
    if not labels:
        return scene

    outer = (R0 + 1.0) * S
    col = outer + 0.5
    ann: List = []
    ann += _title(model, outer + 1.3,
                  f"helical phase ${_xi_text(model)}$, "
                  f"${_phase_text(model)}$; projection onto the $(R,\\phi)$ plane")

    def ring(radius, angle):
        return (S * radius * math.cos(angle), S * radius * math.sin(angle))

    ann += _leader("LCFS", (col, 0.3 * outer), ring(R0 + 1.0, math.radians(20)), "west", "label")
    if show_rational_surface:
        ann += _leader(f"${_q_text(model)}$ surface bounds $R_0\\pm r_s$", (col, 0.62 * outer),
                       ring(R0 + model.r_s, math.radians(55)), "west", "label")
    ann += _leader("magnetic axis $R=R_0$", (col, -0.45 * outer), ring(R0, math.radians(-22)), "west", "label")

    def locus_target(kind, angle):
        theta, _ = model.locus(kind, angle + 2.0 * np.pi * np.arange(model.m))
        radii = R0 + model.section(model.r_s, theta)[:, 0]
        return ring(float(radii.max()), angle)

    ann += _leader("O-point helical locus ($\\xi=0$)", (-col, 0.55 * outer),
                   locus_target("o", math.radians(140)), "east", "o_locus")
    ann += _leader("X-point helical locus ($\\xi=\\pi$)", (-col, -0.5 * outer),
                   locus_target("x", math.radians(215)), "east", "x_locus")
    ann += _leader("poloidal cut $\\phi=0$", (col, -0.12 * outer), (S * (R0 + 1.15), 0.0), "west", "cut")

    arc = 0.62 * (R0 - 1.0) * S
    t = np.radians(np.linspace(35.0, 105.0, 61))
    ann += [
        Polyline.of(np.stack([arc * np.cos(t), arc * np.sin(t)], axis=-1), "axis", role="phi_arrow"),
        Label((0.2 * arc, 1.15 * arc), "$\\phi$", role="phi_arrow"),
        Arrow((0.0, 0.0), (0.8, 0.0), "axis", role="axes"),
        Label((0.4, 0.05), "$R$", anchor="south", role="axes"),
        Marker((0.0, 0.0), "o", "opoint", role="machine_axis"),
        Label((0.0, -0.12), "machine axis", anchor="north", role="machine_axis"),
        Label((0.0, -outer - 0.75),
              "Top view suppresses $Z$: crossings of the projected loci are not reconnection points.\\\\"
              "The poloidal cross-section shows the separatrix and the width $w$.", "note", role="note"),
    ]
    return scene + Scene(tuple(ann))


# ---------------------------------------------------------------------------
# 3-D view
# ---------------------------------------------------------------------------


def _visible(model: IslandModel, r: float, theta, phi) -> np.ndarray:
    """Whether surface points ``(r, theta, phi)`` face the camera and are not occluded.

    The occluder is the surface of that same label (the drawn surface is
    treated as opaque for depth cues). The outward normal comes from the
    embedding's tangents, and a ray marched towards the viewer hides the
    point if it re-enters the torus.
    """
    view, _, _ = _camera()
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    points = model.cartesian(r, theta, phi)
    h = 1e-5
    d_theta = model.cartesian(r, theta + h, phi) - model.cartesian(r, theta - h, phi)
    d_phi = model.cartesian(r, theta, phi + h) - model.cartesian(r, theta, phi - h)
    normal = np.cross(d_theta, d_phi)
    axis = np.stack([model.major_radius * np.cos(phi), model.major_radius * np.sin(phi), np.zeros_like(phi)], axis=-1)
    normal *= np.sign(np.sum(normal * (points - axis), axis=-1))[..., None]
    facing = normal @ view > 0.0
    t = np.linspace(0.02, 2.5 * (model.major_radius + 1.0), 500)
    ray = points[:, None, :] + t[None, :, None] * view[None, None, :]
    occluded = model.contains(r, np.hypot(ray[..., 0], ray[..., 1]), ray[..., 2]).any(axis=1)
    return facing & ~occluded


def _three_d(model: IslandModel, *, labels=True, show_rational_surface=True, show_separatrix=True,
             show_o_points=True, show_x_points=True) -> Scene:
    r_s = model.r_s
    items: List = []

    if show_rational_surface:
        theta = np.linspace(0.0, 2.0 * np.pi, 97)
        for phi0 in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            pts = model.cartesian(r_s, theta, phi0)
            items += _split(_project(pts), _visible(model, r_s, theta, phi0), "mesh", "mesh hidden", "mesh")
        phi = np.linspace(0.0, 2.0 * np.pi, 193)
        for th0 in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            pts = model.cartesian(r_s, th0, phi)
            items += _split(_project(pts), _visible(model, r_s, th0, phi), "mesh", "mesh hidden", "mesh")

    phi = np.linspace(0.0, 2.0 * np.pi * model.m, 300 * model.m + 1)
    loci = {}
    for kind in ("x", "o"):
        theta, ph = model.locus(kind, phi)
        pts = model.cartesian(r_s, theta, ph)
        vis = _visible(model, r_s, theta, ph)
        loci[kind] = (_project(pts), vis)
        items += _split(loci[kind][0], vis, f"{kind} locus", f"{kind} locus hidden", f"{kind}_locus")

    if show_o_points:
        for th in model.o_theta(0.0):
            items.append(Marker(tuple(_project(model.cartesian(r_s, th, 0.0))), "o", "opoint", role="o_point"))
    if show_x_points:
        for th in model.x_theta(0.0):
            items.append(Marker(tuple(_project(model.cartesian(r_s, th, 0.0))), "x", "xpoint", role="x_point"))

    scene = Scene(tuple(items))
    xs = [p[0] for it in scene.items if isinstance(it, Polyline) for p in it.points]
    ys = [p[1] for it in scene.items if isinstance(it, Polyline) for p in it.points]
    left, bottom, top = min(xs), min(ys), max(ys)
    if not labels:
        return scene

    ann: List = []
    ann += _title(model, top + 2.6,
                  f"$m={model.m}$ poloidal and $n={model.n}$ toroidal periodicity; "
                  f"${_xi_text(model)}$, "
                  f"${_phase_text(model)}$")

    def pick(kind, anchor_xy):
        xy, vis = loci[kind]
        cand = np.where(vis)[0]
        best = cand[np.argmin(np.hypot(*(xy[cand] - np.asarray(anchor_xy)).T))]
        return tuple(xy[best])

    col = left - 0.7
    ann += _leader("O-point locus ($\\xi=0$)", (col, 0.35 * top), pick("o", (left + 1.2, 0.35 * top)),
                   "east", "o_locus")
    ann += _leader("X-point locus ($\\xi=\\pi$)", (col, 0.35 * bottom), pick("x", (left + 1.2, 0.35 * bottom)),
                   "east", "x_locus")
    if show_rational_surface:
        target = _project(model.cartesian(r_s, math.pi / 2, CAMERA_AZIMUTH + 0.8))
        ann += _leader(f"${_q_text(model)}$ rational surface", (target[0] + 0.3, top + 0.45),
                       target, "south west", "label")

    # toroidal direction: an arc in the midplane, outside the torus, facing the camera
    R_arc = model.major_radius + 1.35
    ph = np.linspace(CAMERA_AZIMUTH - 0.35, CAMERA_AZIMUTH + 0.35, 41)
    arc = _project(np.stack([R_arc * np.cos(ph), R_arc * np.sin(ph), np.full_like(ph, -0.7 * model.elongation)], axis=-1))
    ann += [
        Polyline.of(arc, "axis", role="phi_arrow"),
        Label(tuple(arc[len(arc) // 2] + np.array([0.0, -0.3])), "toroidal direction $\\phi$",
              anchor="north", role="phi_arrow"),
    ]
    return scene + Scene(tuple(ann))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_BUILDERS = {"poloidal": _poloidal, "top": _top, "3d": _three_d}


def magnetic_island(
    m: int = 2,
    n: int = 1,
    width: float = 0.24,
    phase: float = 0.0,
    projection: str = "poloidal",
    *,
    r_s: float = 0.64,
    aspect_ratio: float = 3.2,
    elongation: float = 1.0,
    triangularity: float = 0.0,
    show_rational_surface: bool = True,
    show_separatrix: bool = True,
    show_o_points: bool = True,
    show_x_points: bool = True,
    labels: bool = True,
) -> Diagram:
    r"""Schematic of an $m/n$ magnetic island in one of three projections.

    All projections are drawn from the same model -- the helical phase
    $\xi = m\theta^* - n\phi - \phi_0$ in each surface's straight-field-line
    angle $\theta^*$ (:mod:`vaft.formula.equilibrium`) and the pendulum flux function
    $\mathcal{H} = \tfrac12 x^2 - (w/4)^2\cos\xi$ of
    :mod:`vaft.formula.stability` -- so O-points ($\xi = 0$), X-points
    ($\xi = \pi$) and the width agree between them.

    Parameters
    ----------
    m, n : int
        Poloidal and toroidal mode numbers, positive and coprime.
    width : float
        Full island width at the O-point, in units of the minor radius.
    phase : float
        Helical phase offset $\phi_0$ [rad]. The O-points on the
        $\phi = 0$ section sit at $\theta^* = (\phi_0 + 2\pi k)/m$.
    projection : {"poloidal", "top", "3d"}
        Poloidal cross-section at $\phi = 0$; midplane projection onto
        $(R, \phi)$; or an orthographic 3-D view of the O and X helices.
    r_s : float
        Rational-surface radius in units of the minor radius.
    aspect_ratio : float
        $R_0 / a$ of the torus the schematic uses.
    elongation : float
        Vertical elongation $\kappa$, the same on every flux surface.
    triangularity : float
        Triangularity $\delta$ at the LCFS, in $[-0.9, 0.9]$. Surface $r$ has
        $\delta(r) = \delta\,r$ (Miller shaping), so the D shape relaxes to a
        circle towards the axis. ``width`` is a physical distance only on the
        outboard midplane.
    show_rational_surface, show_separatrix, show_o_points, show_x_points : bool
        Toggle the corresponding elements.
    labels : bool
        Title, leaders and explanatory notes.

    Returns
    -------
    Diagram
        TikZ source immediately; SVG (and inline Jupyter display) when first
        requested, which needs ``latex`` and ``dvisvgm``.
    """
    model = _validate(m, n, width, phase, projection, r_s, aspect_ratio, elongation, triangularity)
    scene = _BUILDERS[projection](
        model,
        labels=labels,
        show_rational_surface=show_rational_surface,
        show_separatrix=show_separatrix,
        show_o_points=show_o_points,
        show_x_points=show_x_points,
    )
    return Diagram(f"magnetic_island_{projection}", scene, model=model)
