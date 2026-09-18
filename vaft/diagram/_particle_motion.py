"""Single-particle motion diagrams: gyration and guiding-centre drifts.

Every orbit here is integrated from the Lorentz force by
:func:`vaft.formula.particle.boris_orbit`, and every drift arrow is the
corresponding drift formula of the same module -- so a figure cannot show
a drift the integrated orbit does not make. Units are normalised
(|q| = 1, B = 1, m_e = 1); the ion-to-electron mass ratio is reduced (4 by
default) so both orbits are visible, and each figure says so.

Every diagram's ``Diagram.model`` is a :class:`ParticleFigure` holding the
orbits and drift vectors, so tests check the physics without the drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from vaft.formula.particle import (
    boris_orbit,
    curvature_drift_velocity,
    exb_drift_velocity,
    grad_b_drift_velocity,
    gyrofrequency,
)

from ._projection import camera, project
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

_STEPS_PER_PERIOD = 120


@dataclass(eq=False)
class ParticleFigure:
    """What a particle-motion diagram shows, in normalised units."""

    orbits: Dict[str, np.ndarray] = field(default_factory=dict)
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)


def _uniform(vec):
    vec = np.asarray(vec, dtype=float)
    return lambda x: vec


def _orbit(q, m, x0, v0, E_field, B_field, B_scale, n_periods, steps=_STEPS_PER_PERIOD):
    period = 2.0 * math.pi / abs(float(gyrofrequency(q, m, B_scale)))
    dt = period / steps
    x, v = boris_orbit(q, m, x0, v0, E_field, B_field, dt, int(round(n_periods * steps)))
    return x, v, dt


def _title(text: str, subtitle: str, top: float, center: float = 0.0) -> List:
    return [
        Label((center, top), text, "title", role="title"),
        Label((center, top - 0.8), subtitle, "subtitle", role="title"),
    ]


def _validate_mass_ratio(mass_ratio: float) -> float:
    mass_ratio = float(mass_ratio)
    if not mass_ratio >= 1.0:
        raise ValueError(f"mass_ratio is m_i / m_e and must be at least 1, not {mass_ratio!r}")
    return mass_ratio


# ---------------------------------------------------------------------------
# E x B drift
# ---------------------------------------------------------------------------


def _guiding_centre(q, m, x, v, B):
    """Guiding centre of a particle at ``x`` moving at ``v`` (drift frame) in uniform ``B``."""
    B = np.asarray(B, dtype=float)
    return np.asarray(x, dtype=float) - m / (q * (B @ B)) * np.cross(B, v)


def exb_drift(*, mass_ratio: float = 4.0, labels: bool = True) -> Diagram:
    r"""E-cross-B drift of an ion and an electron in uniform crossed fields.

    $\mathbf{B} = B\hat z$ (out of the page) and $\mathbf{E} = E\hat x$. Both
    particles have the same kinetic energy in their guiding-centre frames, so
    the ion's orbit is $\sqrt{m_i/m_e}$ times larger; both guiding centres
    move at the same ``exb_drift_velocity`` -- along $-\hat y$ -- so no
    current flows.
    """
    mass_ratio = _validate_mass_ratio(mass_ratio)
    E, B = np.array([0.2, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    v_E = exb_drift_velocity(E, B)
    duration = 2.0 * math.pi * mass_ratio * 3.0  # three ion gyroperiods
    figure = ParticleFigure(parameters={"mass_ratio": mass_ratio, "E": float(E[0]), "B": float(B[2]),
                                        "duration": duration})
    rho_i = mass_ratio ** 0.5  # equal energy: u_i = u_e / sqrt(mass_ratio)
    gap = 2.2 * rho_i + 2.5
    for name, q, m, x_start in (("ion", 1.0, mass_ratio, 0.0), ("electron", -1.0, 1.0, gap)):
        u = 1.0 / math.sqrt(m)  # speed in the drift frame
        # start at the top of the orbit: gyration velocity along +x for an ion, -x for an electron
        u_vec = np.array([u * np.sign(q), 0.0, 0.0])
        x0 = np.array([x_start, 0.0, 0.0])
        x, _, dt = _orbit(q, m, x0, v_E + u_vec, _uniform(E), _uniform(B), 1.0, duration / (2 * math.pi * m))
        figure.orbits[name] = x
        gc0 = _guiding_centre(q, m, x0, u_vec, B)
        figure.vectors[f"{name}_guiding_centre"] = gc0
        figure.vectors[f"{name}_drift"] = v_E
        figure.parameters[f"{name}_dt"] = dt

    S = 0.45  # cm per unit length
    items: List = []
    for name, style in (("ion", "orbit ion"), ("electron", "orbit electron")):
        x = figure.orbits[name]
        items.append(Polyline.of(S * x[:, :2], style, role=f"{name}_orbit"))
        gc0 = figure.vectors[f"{name}_guiding_centre"]
        items.append(Arrow(tuple(S * gc0[:2]), tuple(S * (gc0 + v_E * duration)[:2]), "drift",
                           role=f"{name}_guiding_centre"))
        items.append(Label(tuple(S * x[0, :2]), "$+$" if name == "ion" else "$-$", "charge", role=f"{name}_start"))

    xs = np.concatenate([o[:, 0] for o in figure.orbits.values()])
    ys = np.concatenate([o[:, 1] for o in figure.orbits.values()])
    right, top, bottom = S * xs.max() + 1.3, S * ys.max(), S * ys.min()
    center = 0.5 * S * (xs.min() + xs.max()) + 1.2
    if labels:
        items += [
            Label((right + 0.25, top - 0.4), "$\\odot$", "legend symbol", role="legend"),
            Label((right + 0.8, top - 0.4), "$\\mathbf{B}$ ($+z$)", "label", anchor="west", role="legend"),
            Arrow((right, top - 1.5), (right + 1.2, top - 1.5), "field arrow", role="legend"),
            Label((right + 1.4, top - 1.5), "$\\mathbf{E}$ ($+x$)", "label", anchor="west", role="legend"),
            Label((right + 0.9, top - 2.6), "$\\mathbf{v}_E$ (guiding centre)", "small label", anchor="west",
                  role="legend"),
            Arrow((right, top - 2.6), (right + 0.8, top - 2.6), "drift", role="legend"),
            Arrow((right + 0.1, bottom + 0.4), (right + 1.1, bottom + 0.4), "axis", role="axes"),
            Label((right + 1.15, bottom + 0.4), "$x$", anchor="west", role="axes"),
            Arrow((right + 0.1, bottom + 0.4), (right + 0.1, bottom + 1.4), "axis", role="axes"),
            Label((right + 0.1, bottom + 1.45), "$y$", anchor="south", role="axes"),
        ]
        items += _title("$\\mathbf{E}\\times\\mathbf{B}$ drift",
                        "$\\mathbf{v}_E = \\mathbf{E}\\times\\mathbf{B}/B^2$: the same for ions and electrons, "
                        "so no current", top + 1.8, center)
        items.append(Label((center, bottom - 0.8),
                           f"orbits integrated from the Lorentz force at equal energy; "
                           f"$m_i/m_e = {mass_ratio:g}$ for visibility", "note", role="note"))
    return Diagram("exb_drift", Scene(tuple(items)), model=figure)


# ---------------------------------------------------------------------------
# Curvature and grad-B drift in a curved field
# ---------------------------------------------------------------------------


def _toroidal_field(R0: float, B0: float):
    def B_field(x):
        R = math.hypot(x[0], x[1])
        return B0 * R0 / R * np.array([-x[1] / R, x[0] / R, 0.0])
    return B_field


def curvature_drift(*, labels: bool = True) -> Diagram:
    r"""An ion spiralling along a curved field line and drifting off it.

    The field is the vacuum toroidal field $\mathbf{B} = B_0 R_0/R\,\hat\phi$,
    whose lines are circles about the $z$ axis: curved, and weaker outward.
    The drift is ``grad_b_drift_velocity`` plus ``curvature_drift_velocity``
    at the starting point -- along $+\hat z$ for an ion. The dashed guiding
    centre follows the field line displaced at that velocity, and the
    integrated orbit spirals around it.
    """
    R0, B0, q, m = 6.0, 4.0, 1.0, 1.0
    v_par, v_perp = 1.0, 2.0
    B_field = _toroidal_field(R0, B0)
    b0 = B_field(np.array([R0, 0.0, 0.0]))
    v_d = (grad_b_drift_velocity(q, m, v_perp, b0, [-B0 / R0, 0.0, 0.0])
           + curvature_drift_velocity(q, m, v_par, b0, [R0, 0.0, 0.0]))
    # start one Larmor radius from the guiding centre on the field line, x = gc + (m/qB^2) B x u
    gc_start = np.array([R0, 0.0, 0.0])
    u = np.array([v_perp, 0.0, 0.0])
    x0 = gc_start + m / (q * B0 ** 2) * np.cross(b0, u)
    v0 = np.array([0.0, v_par, 0.0]) + u + v_d
    arc = 0.5 * math.pi
    duration = arc * R0 / v_par
    x, _, dt = _orbit(q, m, x0, v0, lambda p: np.zeros(3), B_field, B0, duration * q * B0 / (2 * math.pi * m))
    figure = ParticleFigure(parameters={"R0": R0, "B0": B0, "v_par": v_par, "v_perp": v_perp, "duration": duration})
    figure.orbits["ion"] = x
    figure.vectors["drift"] = v_d
    t = np.linspace(0.0, duration, 181)
    phi_gc = v_par * t / R0
    gc = np.stack([R0 * np.cos(phi_gc), R0 * np.sin(phi_gc), v_d[2] * t], axis=-1)
    figure.orbits["guiding_centre"] = gc

    S = 0.85
    phi = np.linspace(-0.1, arc + 0.1, 181)
    line = np.stack([R0 * np.cos(phi), R0 * np.sin(phi), np.zeros_like(phi)], axis=-1)
    mid = gc[len(gc) // 2]
    axis_foot = np.array([0.0, 0.0, mid[2]])
    items: List = [
        Polyline.of(project([[0, 0, 0], [R0 + 1.5, 0, 0]], S), "frame axis", role="axes"),
        Polyline.of(project([[0, 0, 0], [0, R0 + 1.5, 0]], S), "frame axis", role="axes"),
        Polyline.of(project([[0, 0, 0], [0, 0, 3.5]], S), "frame axis", role="axes"),
        Polyline.of(project(line, S), "field line", role="field_line"),
        Arrow(tuple(project(line[-8], S)), tuple(project(line[-1], S)), "field line arrow", role="field_line"),
        Polyline.of(project(gc, S), "guiding centre", role="guiding_centre"),
        Polyline.of(project(x, S), "orbit ion", role="ion_orbit"),
        Arrow(tuple(project(axis_foot, S)), tuple(project(mid, S)), "vector", role="radius"),
        Arrow(tuple(project(mid, S)), tuple(project(mid + 2.0 * v_d / np.linalg.norm(v_d), S)), "drift",
              role="drift"),
    ]
    if labels:
        end = project(line[-1], S)
        tip = project(mid + 2.0 * v_d / np.linalg.norm(v_d), S)
        gc_label_at = project(gc[int(0.8 * len(gc))], S)
        items += [
            Label(tuple(0.5 * (project(axis_foot, S) + project(mid, S)) + np.array([0.0, 0.3])),
                  "$\\mathbf{R}_c$", "label", role="radius"),
            Label(tuple(end + np.array([-0.2, 0.3])), "$\\mathbf{B}$", "label", anchor="east", role="field_line"),
            Label(tuple(tip + np.array([0.25, 0.0])), "$\\mathbf{v}_{\\nabla B} + \\mathbf{v}_R$", "label",
                  anchor="west", role="drift"),
            Arrow((gc_label_at[0] - 2.2, gc_label_at[1] + 1.3), tuple(gc_label_at), "leader", role="guiding_centre"),
            Label((gc_label_at[0] - 2.25, gc_label_at[1] + 1.3), "guiding centre", "small label", anchor="east",
                  role="guiding_centre"),
            Label(tuple(project(x[0], S) + np.array([0.35, -0.1])), "$+$", "charge", anchor="west", role="ion_start"),
            Label(tuple(project([0, 0, 3.5], S) + np.array([0, 0.1])), "$z$", anchor="south", role="axes"),
            Label(tuple(project([R0 + 1.5, 0, 0], S)), "$x$", anchor="north", role="axes"),
            Label(tuple(project([0, R0 + 1.5, 0], S)), "$y$", anchor="west", role="axes"),
        ]
        xy = np.concatenate([project(x, S), project(line, S)])
        center = 0.5 * (xy[:, 0].min() + xy[:, 0].max())
        items += _title("Curvature and $\\nabla B$ drift",
                        "$\\mathbf{B} = B_0R_0/R\\,\\hat\\phi$: field lines curve and weaken outward, "
                        "so the ion drifts along $+z$", xy[:, 1].max() + 2.2, center)
        items.append(Label((center, xy[:, 1].min() - 0.9),
                           "orbit integrated from the Lorentz force; drift arrow and guiding centre from the "
                           "drift formulas", "note", role="note"))
    return Diagram("curvature_drift", Scene(tuple(items)), model=figure)


# ---------------------------------------------------------------------------
# Magnetization current
# ---------------------------------------------------------------------------


def magnetization_current(*, labels: bool = True) -> Diagram:
    r"""Gyrating ions in a bounded region: interior currents cancel, the edge carries one.

    $\mathbf{B}$ points into the page. Ions with guiding centres filling a
    square gyrate (orbits integrated from the Lorentz force); where two
    neighbouring orbits overlap their currents cancel, but at the edge of the
    region nothing cancels them, leaving the magnetization current
    $\mathbf{J}_M = \nabla\times\mathbf{M}$. Its direction is computed from
    the orbits -- binning their current -- and is diamagnetic.
    """
    q, m, B0, rho = 1.0, 1.0, 1.0, 1.0
    B = np.array([0.0, 0.0, -B0])
    side, spacing = 8.0, 1.6
    centres = np.arange(0.5 * spacing, side, spacing)
    figure = ParticleFigure(parameters={"side": side, "rho": rho, "spacing": spacing})
    orbits = {}
    for i, cx in enumerate(centres):
        for j, cy in enumerate(centres):
            # start on the orbit's +x side; an ion in B = -z B0 turns counter-clockwise
            x0 = np.array([cx + rho, cy, 0.0])
            v0 = np.array([0.0, rho * B0 * abs(q) / m, 0.0])
            x, v, dt = _orbit(q, m, x0, v0, lambda p: np.zeros(3), _uniform(B), B0, 1.0)
            orbits[(i, j)] = (x[:-1], v[1:])
    figure.orbits.update({f"orbit_{i}_{j}": o[0] for (i, j), o in orbits.items()})

    # current density J_y(x) and J_x(y), binned from every orbit sample
    samples = np.concatenate([o[0] for o in orbits.values()])
    velocities = np.concatenate([o[1] for o in orbits.values()])
    edges = np.linspace(-rho, side + rho, 41)
    Jy, _ = np.histogram(samples[:, 0], bins=edges, weights=q * velocities[:, 1])
    Jx, _ = np.histogram(samples[:, 1], bins=edges, weights=q * velocities[:, 0])
    figure.vectors["bin_edges"] = edges
    figure.vectors["J_y_of_x"] = Jy
    figure.vectors["J_x_of_y"] = Jx
    # net edge current: right edge (x > side - rho) and top edge (y > side - rho)
    right = Jy[edges[:-1] >= side - rho].sum()
    top = Jx[edges[:-1] >= side - rho].sum()
    figure.parameters["right_edge_current"] = float(right)
    figure.parameters["top_edge_current"] = float(top)

    S = 0.55
    items: List = [Polyline.of(S * np.array([[0, 0], [side, 0], [side, side], [0, side]]), "region box",
                               role="region", closed=True)]
    shown = [(0, 0), (len(centres) - 1, len(centres) - 1), (2, 1), (1, 3)] + \
        [(i, len(centres) - 1) for i in range(len(centres))] + [(len(centres) - 1, j) for j in range(len(centres))] + \
        [(i, 0) for i in range(len(centres))] + [(0, j) for j in range(len(centres))]
    for key in dict.fromkeys(shown):
        x = orbits[key][0]
        items.append(Polyline.of(S * x[:, :2], "orbit ion", role="orbit"))
        k = len(x) // 3
        items.append(Arrow(tuple(S * x[k - 3, :2]), tuple(S * x[k, :2]), "orbit tip", role="orbit"))
    # the net current along each edge, drawn just outside it in the computed sense
    L, off = S * side, 0.55
    sy, sx = np.sign(right), np.sign(top)
    items += [
        Arrow((L + off, 0.5 * L - 1.2 * sy), (L + off, 0.5 * L + 1.2 * sy), "current", role="edge_current"),
        Arrow((-off, 0.5 * L + 1.2 * sy), (-off, 0.5 * L - 1.2 * sy), "current", role="edge_current"),
        Arrow((0.5 * L - 1.2 * sx, L + off), (0.5 * L + 1.2 * sx, L + off), "current", role="edge_current"),
        Arrow((0.5 * L + 1.2 * sx, -off), (0.5 * L - 1.2 * sx, -off), "current", role="edge_current"),
    ]
    if labels:
        items += [
            Label((L + 1.3, 0.3 * L), "$\\otimes\\ \\mathbf{B}$", "label", anchor="west", role="legend"),
            Label((L + off + 0.2, 0.5 * L + 1.4), "$\\mathbf{J}_M$", "label", anchor="west", role="edge_current"),
        ]
        items += _title("Magnetization current",
                        "gyro-currents cancel inside; at the edge $\\mathbf{J}_M = \\nabla\\times\\mathbf{M}$ "
                        "survives, and it is diamagnetic", L + 2.4, 0.5 * L)
        items.append(Label((0.5 * L, -1.4), "ion orbits integrated from the Lorentz force; "
                           "edge arrows from the binned orbit current", "note", role="note"))
    return Diagram("magnetization_current", Scene(tuple(items)), model=figure)


# ---------------------------------------------------------------------------
# Toroidal drift: charge separation and outward E x B
# ---------------------------------------------------------------------------


def toroidal_drift(*, aspect_ratio: float = 2.2, labels: bool = True) -> Diagram:
    r"""Why a purely toroidal field cannot confine a plasma.

    In $\mathbf{B} = B_0R_0/R\,\hat\phi$ the grad-B and curvature drifts
    (``grad_b_drift_velocity`` + ``curvature_drift_velocity``) move ions up
    and electrons down. The separated charge sets up a vertical
    $\mathbf{E}$, and ``exb_drift_velocity`` of that field is outward for both
    species: the plasma is pushed out of the torus. All four directions are
    computed at the drawn cross-section.
    """
    from ._magnetic_island import _validate as _island_model

    if not aspect_ratio > 1.3:
        raise ValueError(f"aspect_ratio must exceed 1.3, not {aspect_ratio!r}")
    R0, a, B0 = float(aspect_ratio), 1.0, 1.0
    view, u, v = camera()
    phi_right = math.atan2(u[1], u[0])  # the section whose major-radius direction is screen-right
    Rhat = np.array([math.cos(phi_right), math.sin(phi_right), 0.0])
    zhat = np.array([0.0, 0.0, 1.0])
    point = R0 * Rhat
    b = B0 * np.array([-math.sin(phi_right), math.cos(phi_right), 0.0])
    drifts = {}
    for name, q, m in (("ion", 1.0, 4.0), ("electron", -1.0, 1.0)):
        drifts[name] = (grad_b_drift_velocity(q, m, 1.0, b, -B0 / R0 * Rhat)
                        + curvature_drift_velocity(q, m, 1.0, b, R0 * Rhat))
    # the ions pile up where they drift to: E points from their layer to the electrons'
    E_dir = -np.sign(drifts["ion"][2]) * zhat
    v_E = exb_drift_velocity(E_dir, b)
    figure = ParticleFigure(parameters={"aspect_ratio": aspect_ratio, "phi_section": phi_right})
    figure.vectors.update({"ion_drift": drifts["ion"], "electron_drift": drifts["electron"],
                           "E": E_dir, "v_E": v_E, "B": b, "R_hat": Rhat})

    S = 1.1
    items: List = []
    # torus silhouette: surface points whose normal is perpendicular to the view, drawn where not occluded
    torus = _island_model(2, 1, 0.1, 0.0, "3d", 0.5, R0, 1.0)
    phi = np.linspace(0.0, 2 * math.pi, 721)
    c = np.cos(phi) * view[0] + np.sin(phi) * view[1]
    for branch in (0.0, math.pi):
        theta = np.arctan2(-c, view[2]) + branch
        pts = torus.cartesian(a, theta, phi)
        nudged = pts + 0.02 * view
        ray = nudged[:, None, :] + np.linspace(0.02, 2.5 * (R0 + a), 400)[None, :, None] * view[None, None, :]
        visible = ~torus.contains(a, np.hypot(ray[..., 0], ray[..., 1]), ray[..., 2]).any(axis=1)
        for run in np.split(np.arange(len(phi)), np.where(np.diff(visible.astype(int)) != 0)[0] + 1):
            if visible[run[0]] and len(run) > 1:
                items.append(Polyline.of(project(pts[run], S), "torus rim", role="torus"))
    # the two cross-sections that face the viewer, as cut faces
    th = np.linspace(0.0, 2 * math.pi, 181)
    for ph in (phi_right + math.pi, phi_right):
        rh = np.array([math.cos(ph), math.sin(ph), 0.0])
        section = (R0 + a * np.cos(th))[:, None] * rh + (a * np.sin(th))[:, None] * zhat
        items.append(Polyline.of(project(section, S), "section fill", role="section", closed=True))

    # screen directions of R-hat and z-hat at the right section
    centre = project(point, S)
    eR = project(point + Rhat, S) - centre
    ez = project(point + zhat, S) - centre

    def at(x, y):
        return tuple(centre + x * eR + y * ez)

    sign_ion = float(np.sign(drifts["ion"][2]))
    for ang in np.radians([58, 90, 122]):
        items.append(Label(at(0.72 * math.cos(ang), sign_ion * 0.72 * math.sin(ang)), "$+$", "charge small",
                           role="ion_layer"))
        items.append(Label(at(0.72 * math.cos(ang), -sign_ion * 0.72 * math.sin(ang)), "$-$", "charge small",
                           role="electron_layer"))
    items += [
        Arrow(at(-0.45, -0.4 * E_dir[2]), at(-0.45, 0.4 * E_dir[2]), "field vector", role="E"),
        Arrow(at(0.2, 0.05 * sign_ion), at(0.2, 0.45 * sign_ion), "drift ion", role="ion_drift"),
        Arrow(at(0.2, -0.05 * sign_ion), at(0.2, -0.45 * sign_ion), "drift electron", role="electron_drift"),
    ]
    vE_screen = project(v_E / np.linalg.norm(v_E), S) - project(np.zeros(3), S)
    vE_screen /= np.linalg.norm(vE_screen)
    start = np.asarray(at(1.12, 0.0))
    items.append(Arrow(tuple(start), tuple(start + 1.4 * vE_screen), "exb", role="v_E"))

    # toroidal field arrow along +phi on the front top of the torus, and the z axis with the phi sense
    ph_front = math.atan2(view[1], view[0])
    arc = np.linspace(ph_front - 0.7, ph_front - 0.1, 40)
    arc_pts = np.stack([R0 * np.cos(arc), R0 * np.sin(arc), np.full_like(arc, a)], axis=-1)
    tangent = np.array([-math.sin(arc[0]), math.cos(arc[0]), 0.0])
    if np.dot(arc_pts[-1] - arc_pts[0], tangent) < 0:  # draw along +phi, the direction of B
        arc_pts = arc_pts[::-1]
    items.append(Arrow(tuple(project(arc_pts[-4], S)), tuple(project(arc_pts[-1], S)), "field arrow",
                       role="B_toroidal"))
    items.append(Polyline.of(project(arc_pts, S), "field line", role="B_toroidal"))
    items.append(Arrow(tuple(project(-1.6 * a * zhat, S)), tuple(project(2.1 * a * zhat, S)), "frame axis arrow",
                       role="axes"))
    ring = np.linspace(0.3, 2 * math.pi - 0.3, 60)
    ring_pts = np.stack([0.45 * np.cos(ring), 0.45 * np.sin(ring), np.full_like(ring, 1.7 * a)], axis=-1)
    items.append(Polyline.of(project(ring_pts, S), "frame axis", role="axes"))
    items.append(Arrow(tuple(project(ring_pts[-3], S)), tuple(project(ring_pts[-1], S)), "frame axis arrow",
                       role="axes"))
    if labels:
        items += [
            Label(tuple(start + 1.4 * vE_screen + np.array([0.15, 0.0])),
                  "$\\mathbf{v}_E = \\dfrac{\\mathbf{E}\\times\\mathbf{B}}{B^2}$", "label", anchor="west", role="v_E"),
            Label(at(-0.55, 0.0), "$\\mathbf{E}$", "label", anchor="east", role="E"),
            Arrow(tuple(np.asarray(at(0.2, 0.45 * sign_ion)) + np.array([1.9, 0.9 * sign_ion])),
                  at(0.28, 0.4 * sign_ion), "leader", role="ion_drift"),
            Label(tuple(np.asarray(at(0.2, 0.45 * sign_ion)) + np.array([1.95, 0.9 * sign_ion])), "ion drift",
                  "label", anchor="west", role="ion_drift"),
            Arrow(tuple(np.asarray(at(0.2, -0.45 * sign_ion)) + np.array([1.9, -0.9 * sign_ion])),
                  at(0.28, -0.4 * sign_ion), "leader", role="electron_drift"),
            Label(tuple(np.asarray(at(0.2, -0.45 * sign_ion)) + np.array([1.95, -0.9 * sign_ion])), "electron drift",
                  "label", anchor="west", role="electron_drift"),
            Label(tuple(project(arc_pts[-1], S) + np.array([0.1, 0.25])), "$\\mathbf{B}_\\mathrm{toroidal}$",
                  "label", anchor="south", role="B_toroidal"),
            Label(tuple(project(2.1 * a * zhat, S) + np.array([0, 0.1])), "$z$", anchor="south", role="axes"),
            Label(tuple(project(ring_pts[len(ring_pts) // 2], S) + np.array([0.0, -0.1])), "$\\phi$",
                  anchor="north", role="axes"),
        ]
        drawn = np.concatenate([np.asarray(it.points) for it in items if isinstance(it, Polyline)])
        top = max(float(project(2.1 * a * zhat, S)[1]), drawn[:, 1].max()) + 1.5
        items += _title("Toroidal drift and charge separation",
                        "$\\nabla B$ and curvature drifts separate charge; the resulting "
                        "$\\mathbf{E}\\times\\mathbf{B}$ pushes the plasma outward", top, 0.0)
        bottom = min(float(project(-1.6 * a * zhat, S)[1]), drawn[:, 1].min()) - 0.6
        items.append(Label((0.0, bottom), "drift directions from the formulas at the drawn cross-section: "
                           "a purely toroidal field cannot confine", "note", role="note"))
    return Diagram("toroidal_drift", Scene(tuple(items)), model=figure)
