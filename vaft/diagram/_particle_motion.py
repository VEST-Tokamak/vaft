"""Single-particle motion diagrams: gyration and guiding-centre drifts.

Every orbit here is integrated from the Lorentz force by
:func:`vaft.formula.particle.boris_orbit`, and every drift arrow is the
corresponding drift formula of the same module -- so a figure cannot show
a drift the integrated orbit does not make. Units are normalised
(|q| = 1, m_e = 1, fields of order one); the ion-to-electron mass ratio is
reduced (4 by default) so both orbits are visible, and each figure says so.

Every diagram's ``Diagram.model`` is a :class:`ParticleFigure` holding the
orbits and drift vectors, so tests check the physics without the drawing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from vaft.formula.equilibrium import vacuum_toroidal_field
from vaft.formula.particle import (
    boris_orbit,
    curvature_drift_velocity,
    exb_drift_velocity,
    grad_b_drift_velocity,
    gyrofrequency,
    larmor_radius,
)

from ._projection import camera, project
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

_STEPS_PER_PERIOD = 120
#: equal-energy E x B orbits loop only while u_i = 1/sqrt(m_i) exceeds v_E = 0.2
_MAX_MASS_RATIO = 20.0
#: guiding-centre spacing of the lattice whose current is binned, well below rho = 1
_LATTICE_SPACING = 0.25
#: parallel speed given to the uniform-field orbits: it does not change their
#: perpendicular motion, and turns them into helices in the 3-D views
_EXB_V_PAR = 0.25
_MAGNETIZATION_V_PAR = 0.25


def _check_projection(projection: str, allowed) -> str:
    if projection not in allowed:
        raise ValueError(f"projection must be one of {tuple(allowed)}, not {projection!r}")
    return projection


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
    if not 1.0 <= mass_ratio <= _MAX_MASS_RATIO:
        raise ValueError(
            f"mass_ratio is m_i / m_e and must lie in [1, {_MAX_MASS_RATIO:g}]: above that the ion's gyration "
            f"speed falls below v_E and its orbit stops looping; got {mass_ratio!r}"
        )
    return mass_ratio


# ---------------------------------------------------------------------------
# Equations shown on the figures
# ---------------------------------------------------------------------------

#: the formulas each family draws, shown on the figure as their docstrings define them
_EQUATIONS = {
    "exb_drift": (boris_orbit, exb_drift_velocity, larmor_radius),
    "curvature_drift": (boris_orbit, grad_b_drift_velocity, curvature_drift_velocity),
    "magnetization_current": (boris_orbit, gyrofrequency, larmor_radius),
    "toroidal_drift": (grad_b_drift_velocity, curvature_drift_velocity, exb_drift_velocity),
}
_DISPLAY_EQUATION = re.compile(r"\$\$(.+?)\$\$", re.S)
_EQUATION_LINE_HEIGHT = 0.95  # cm per displayed equation in the box


def formula_equation(function) -> str:
    """The defining equation of a ``vaft.formula`` function, from its docstring.

    The figures show exactly this text, so an equation on a diagram cannot
    drift from the one the formula documents and implements.
    """
    match = _DISPLAY_EQUATION.search(function.__doc__ or "")
    if match is None:
        raise ValueError(f"{function.__name__} documents no $$...$$ equation")
    return " ".join(match.group(1).split())


def _with_equations(scene: Scene, family: str) -> Scene:
    """Put the family's equations in a box where the note was, and the note below it."""
    notes = [it for it in scene.items if isinstance(it, Label) and it.role == "note"]
    if not notes:
        return scene
    note = notes[0]
    lines = [formula_equation(fn) for fn in _EQUATIONS[family]]
    text = "\\\\[3pt]".join(f"$\\displaystyle {line}$" for line in lines)
    height = _EQUATION_LINE_HEIGHT * len(lines) + 0.3
    x, y = note.at
    box = Label((x, y + 0.2), text, "formula box", anchor="north", role="equations")
    moved = Label((x, y + 0.2 - height - 0.35), note.text, note.style, note.anchor, note.role)
    items = tuple(moved if it is note else it for it in scene.items) + (box,)
    return Scene(items)


def _diagram(name: str, scene: Scene, figure: "ParticleFigure", labels: bool) -> Diagram:
    family = next((key for key in _EQUATIONS if name == key or name.startswith(key + "_")), None)
    if family is None:
        raise KeyError(f"no equation set registered for diagram {name!r}")
    if labels:
        scene = _with_equations(scene, family)
    return Diagram(name, scene, model=figure)


# ---------------------------------------------------------------------------
# E x B drift
# ---------------------------------------------------------------------------


def _guiding_centre(q, m, x, v, B):
    """Guiding centre of a particle at ``x`` moving at ``v`` (drift frame) in uniform ``B``."""
    B = np.asarray(B, dtype=float)
    return np.asarray(x, dtype=float) - m / (q * (B @ B)) * np.cross(B, v)


def exb_drift(*, mass_ratio: float = 4.0, projection: str = "perpendicular", labels: bool = True) -> Diagram:
    r"""E-cross-B drift of an ion and an electron in uniform crossed fields.

    $\mathbf{B} = B\hat z$ (out of the page) and $\mathbf{E} = E\hat x$. Both
    particles have the same kinetic energy in their guiding-centre frames, so
    the ion's orbit is $\sqrt{m_i/m_e}$ times larger; both guiding centres
    move at the same ``exb_drift_velocity`` -- along $-\hat y$ -- so no
    current flows.

    ``projection="perpendicular"`` is the plane normal to $\mathbf{B}$;
    ``"3d"`` shows the same orbits with a small parallel velocity, as helices
    along $\mathbf{B}$ drifting sideways.
    """
    _check_projection(projection, ("perpendicular", "3d"))
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
        v_par = np.array([0.0, 0.0, _EXB_V_PAR])
        x, _, dt = _orbit(q, m, x0, v_E + u_vec + v_par, _uniform(E), _uniform(B), 1.0,
                          duration / (2 * math.pi * m))
        figure.orbits[name] = x
        gc0 = _guiding_centre(q, m, x0, u_vec, B)
        figure.vectors[f"{name}_guiding_centre"] = gc0
        figure.vectors[f"{name}_drift"] = v_E
        figure.parameters[f"{name}_dt"] = dt
    figure.parameters["v_par"] = _EXB_V_PAR
    if projection == "3d":
        return _diagram("exb_drift_3d", _exb_3d(figure, labels), figure, labels)

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
    return _diagram("exb_drift", Scene(tuple(items)), figure, labels)


# ---------------------------------------------------------------------------
# Curvature and grad-B drift in a curved field
# ---------------------------------------------------------------------------


def _toroidal_field(R0: float, B0: float):
    """The vacuum toroidal field as a vector field, from ``vacuum_toroidal_field``."""
    def B_field(x):
        R = math.hypot(x[0], x[1])
        return float(vacuum_toroidal_field(B0, R0, R)) * np.array([-x[1] / R, x[0] / R, 0.0])
    return B_field


def _toroidal_field_gradient(R0: float, B0: float, R: float) -> float:
    """d|B|/dR of ``vacuum_toroidal_field`` at ``R``, by central difference."""
    h = 1e-6 * R
    return float(vacuum_toroidal_field(B0, R0, R + h) - vacuum_toroidal_field(B0, R0, R - h)) / (2 * h)


def curvature_drift(*, projection: str = "3d", labels: bool = True) -> Diagram:
    r"""An ion spiralling along a curved field line and drifting off it.

    The field is the vacuum toroidal field $\mathbf{B} = B_0 R_0/R\,\hat\phi$,
    whose lines are circles about the $z$ axis: curved, and weaker outward.
    The drift is ``grad_b_drift_velocity`` plus ``curvature_drift_velocity``
    at the starting point -- along $+\hat z$ for an ion. The dashed guiding
    centre follows the field line displaced at that velocity, and the
    integrated orbit spirals around it.

    ``projection="poloidal"`` looks along the field line (the $(R, z)$
    plane): the gyration is a circle climbing at the drift velocity.
    ``"top"`` looks down the $z$ axis at the curved line and the orbit
    wrapped around it.
    """
    _check_projection(projection, ("3d", "poloidal", "top"))
    R0, B0, q, m = 6.0, 4.0, 1.0, 1.0
    v_par, v_perp = 1.0, 2.0
    B_field = _toroidal_field(R0, B0)
    b0 = B_field(np.array([R0, 0.0, 0.0]))
    v_d = (grad_b_drift_velocity(q, m, v_perp, b0, [_toroidal_field_gradient(R0, B0, R0), 0.0, 0.0])
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
    if projection == "poloidal":
        return _diagram("curvature_drift_poloidal", _curvature_poloidal(figure, labels), figure, labels)
    if projection == "top":
        return _diagram("curvature_drift_top", _curvature_top(figure, labels), figure, labels)

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
            Arrow((gc_label_at[0] + 2.0, gc_label_at[1] - 1.4), tuple(gc_label_at), "leader", role="guiding_centre"),
            Label((gc_label_at[0] + 2.05, gc_label_at[1] - 1.4), "guiding centre", "small label", anchor="west",
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
    return _diagram("curvature_drift", Scene(tuple(items)), figure, labels)


# ---------------------------------------------------------------------------
# Magnetization current
# ---------------------------------------------------------------------------


def magnetization_current(*, projection: str = "perpendicular", labels: bool = True) -> Diagram:
    r"""Gyrating ions in a bounded region: interior currents cancel, the edge carries one.

    $\mathbf{B}$ points into the page. Ions with guiding centres filling a
    square gyrate (orbits integrated from the Lorentz force). With guiding
    centres much closer than a Larmor radius, the currents of overlapping
    orbits cancel point by point inside, but at the edge of the region nothing
    cancels them, leaving the magnetization current
    $\mathbf{J}_M = \nabla\times\mathbf{M}$. The current is computed by
    binning the orbits of such a fine lattice -- the drawn orbits are a sparse
    sample of it -- and is diamagnetic.

    ``projection="perpendicular"`` is the plane normal to $\mathbf{B}$;
    ``"3d"`` shows the same orbits with a small parallel velocity, as helical
    columns along $\mathbf{B}$ with the edge current wrapped around them.
    """
    _check_projection(projection, ("perpendicular", "3d"))
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
            v0 = np.array([0.0, rho * B0 * abs(q) / m, -_MAGNETIZATION_V_PAR])  # along B = -z
            x, v, dt = _orbit(q, m, x0, v0, lambda p: np.zeros(3), _uniform(B), B0, 3.0)
            # the perpendicular view and the current use one gyroperiod; the 3-D view all three
            orbits[(i, j)] = (x[:_STEPS_PER_PERIOD], v[1:_STEPS_PER_PERIOD + 1])
            figure.orbits[f"helix_{i}_{j}"] = x
    figure.orbits.update({f"orbit_{i}_{j}": o[0] for (i, j), o in orbits.items()})

    # The current: guiding centres filling the square at a spacing much finer than rho. In a
    # uniform field every orbit is the same Boris orbit translated, so one integration serves the
    # lattice; bins one lattice spacing wide then average the current over a lattice period.
    x1, v1, _ = _orbit(q, m, np.array([rho, 0.0, 0.0]), np.array([0.0, rho * B0 * abs(q) / m, 0.0]),
                       lambda p: np.zeros(3), _uniform(B), B0, 1.0)
    shape_x, shape_v = x1[:_STEPS_PER_PERIOD], v1[1:_STEPS_PER_PERIOD + 1]
    d = _LATTICE_SPACING
    lattice = np.arange(0.5 * d, side, d)
    rows = len(lattice)
    n_pad = int(math.ceil(rho / d)) + 1
    edges = lattice[0] - 0.5 * d + d * np.arange(-n_pad, rows + n_pad + 1)
    Jy, _ = np.histogram((lattice[:, None] + shape_x[None, :, 0]).ravel(), bins=edges,
                         weights=np.tile(q * shape_v[:, 1], rows) * rows)
    Jx, _ = np.histogram((lattice[:, None] + shape_x[None, :, 1]).ravel(), bins=edges,
                         weights=np.tile(q * shape_v[:, 0], rows) * rows)
    figure.parameters["lattice_spacing"] = d
    figure.vectors["bin_edges"] = edges
    figure.vectors["J_y_of_x"] = Jy
    figure.vectors["J_x_of_y"] = Jx
    # net edge current: right edge (x > side - rho) and top edge (y > side - rho)
    right = Jy[edges[:-1] >= side - rho].sum()
    top = Jx[edges[:-1] >= side - rho].sum()
    figure.parameters["right_edge_current"] = float(right)
    figure.parameters["top_edge_current"] = float(top)
    figure.parameters["n_centres"] = len(centres)
    if projection == "3d":
        return _diagram("magnetization_current_3d", _magnetization_3d(figure, labels), figure, labels)

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
    return _diagram("magnetization_current", Scene(tuple(items)), figure, labels)


# ---------------------------------------------------------------------------
# Toroidal drift: charge separation and outward E x B
# ---------------------------------------------------------------------------


def toroidal_drift(*, aspect_ratio: float = 2.2, projection: str = "3d", labels: bool = True) -> Diagram:
    r"""Why a purely toroidal field cannot confine a plasma.

    In $\mathbf{B} = B_0R_0/R\,\hat\phi$ the grad-B and curvature drifts
    (``grad_b_drift_velocity`` + ``curvature_drift_velocity``) move ions up
    and electrons down. The separated charge sets up a vertical
    $\mathbf{E}$, and ``exb_drift_velocity`` of that field is outward for both
    species: the plasma is pushed out of the torus. All four directions are
    computed at the drawn cross-section.

    ``projection="poloidal"`` is that cross-section in the $(R, z)$ plane --
    the textbook picture -- and ``"top"`` looks down on the circular field
    lines and the inward $\nabla B$, with the vertical drifts out of the page.
    """
    from ._magnetic_island import _validate as _island_model

    _check_projection(projection, ("3d", "poloidal", "top"))
    if not (math.isfinite(aspect_ratio) and aspect_ratio > 1.3):
        raise ValueError(f"aspect_ratio must be finite and exceed 1.3, not {aspect_ratio!r}")
    R0, a, B0 = float(aspect_ratio), 1.0, 1.0
    view, u, v = camera()
    phi_right = math.atan2(u[1], u[0])  # the section whose major-radius direction is screen-right
    Rhat = np.array([math.cos(phi_right), math.sin(phi_right), 0.0])
    zhat = np.array([0.0, 0.0, 1.0])
    point = R0 * Rhat
    b = float(vacuum_toroidal_field(B0, R0, R0)) * np.array([-math.sin(phi_right), math.cos(phi_right), 0.0])
    drifts = {}
    for name, q, m in (("ion", 1.0, 4.0), ("electron", -1.0, 1.0)):
        drifts[name] = (grad_b_drift_velocity(q, m, 1.0, b, _toroidal_field_gradient(R0, B0, R0) * Rhat)
                        + curvature_drift_velocity(q, m, 1.0, b, R0 * Rhat))
    # the ions pile up where they drift to: E points from their layer to the electrons'
    E_dir = -np.sign(drifts["ion"][2]) * zhat
    v_E = exb_drift_velocity(E_dir, b)
    figure = ParticleFigure(parameters={"aspect_ratio": aspect_ratio, "phi_section": phi_right})
    figure.vectors.update({"ion_drift": drifts["ion"], "electron_drift": drifts["electron"],
                           "E": E_dir, "v_E": v_E, "B": b, "R_hat": Rhat})
    if projection == "poloidal":
        return _diagram("toroidal_drift_poloidal", _toroidal_poloidal(figure, labels), figure, labels)
    if projection == "top":
        return _diagram("toroidal_drift_top", _toroidal_top(figure, labels), figure, labels)

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
    return _diagram("toroidal_drift", Scene(tuple(items)), figure, labels)



# ---------------------------------------------------------------------------
# Additional projections
# ---------------------------------------------------------------------------


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _exb_3d(figure: ParticleFigure, labels: bool) -> Scene:
    """The E x B orbits as helices along B (vertical), drifting along -y."""
    S = 0.42
    duration, v_par = figure.parameters["duration"], figure.parameters["v_par"]
    items: List = []
    for name, style in (("ion", "orbit ion"), ("electron", "orbit electron")):
        x = figure.orbits[name]
        items.append(Polyline.of(project(x, S), style, role=f"{name}_orbit"))
        gc0 = figure.vectors[f"{name}_guiding_centre"]
        gc1 = gc0 + (figure.vectors[f"{name}_drift"] + np.array([0.0, 0.0, v_par])) * duration
        items.append(Arrow(tuple(project(gc0, S)), tuple(project(gc1, S)), "drift", role=f"{name}_guiding_centre"))
        items.append(Label(tuple(project(x[0], S)), "$+$" if name == "ion" else "$-$", "charge", role=f"{name}_start"))
    pts = np.concatenate([project(o, S) for o in figure.orbits.values()])
    corner = np.array([figure.orbits["electron"][:, 0].max() + 4.0, 4.0, 0.0])
    origin = project(corner, S)
    items += [
        Arrow(tuple(origin), tuple(project(corner + [0, 0, 3.0], S)), "field vector", role="legend_B"),
        Arrow(tuple(origin), tuple(project(corner + [3.0, 0, 0], S)), "field arrow", role="legend_E"),
        Arrow(tuple(origin), tuple(project(corner + _unit(figure.vectors["ion_drift"]) * 3.0, S)), "drift",
              role="legend_vE"),
    ]
    if labels:
        items += [
            Label(tuple(project(corner + [0, 0, 3.2], S)), "$\\mathbf{B}$", "label", anchor="south", role="legend_B"),
            Label(tuple(project(corner + [3.2, 0, 0], S)), "$\\mathbf{E}$", "label", anchor="west", role="legend_E"),
            Label(tuple(project(corner + _unit(figure.vectors["ion_drift"]) * 3.3, S)), "$\\mathbf{v}_E$", "label",
                  anchor="north", role="legend_vE"),
        ]
        center = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
        items += _title("$\\mathbf{E}\\times\\mathbf{B}$ drift",
                        "helices along $\\mathbf{B}$ whose guiding centres drift at $\\mathbf{v}_E$, "
                        "the same for both species", pts[:, 1].max() + 2.0, center)
        items.append(Label((center, pts[:, 1].min() - 0.8),
                           f"the same orbits as the perpendicular view, with $v_\\parallel = {figure.parameters['v_par']:g}$; "
                           f"$m_i/m_e = {figure.parameters['mass_ratio']:g}$", "note", role="note"))
    return Scene(tuple(items))


def _curvature_poloidal(figure: ParticleFigure, labels: bool) -> Scene:
    """The (R - R0, z) plane: the gyration circle climbing at the drift velocity."""
    S = 2.2
    R0 = figure.parameters["R0"]
    x, gc, v_d = figure.orbits["ion"], figure.orbits["guiding_centre"], figure.vectors["drift"]
    Rz = np.stack([np.hypot(x[:, 0], x[:, 1]) - R0, x[:, 2]], axis=-1)
    gRz = np.stack([np.hypot(gc[:, 0], gc[:, 1]) - R0, gc[:, 2]], axis=-1)
    items: List = [
        Polyline.of(S * Rz, "orbit ion", role="ion_orbit"),
        Polyline.of(S * gRz, "guiding centre", role="guiding_centre"),
        # the drift at the start, where R-hat is +x: its (R, z) components are (v_x, v_z)
        Arrow(tuple(S * gRz[0]), tuple(S * gRz[0] + 2.0 * _unit([v_d[0], v_d[2]])), "drift", role="drift"),
        Label(tuple(S * gRz[0]), "$\\otimes$", "legend symbol", role="field_line"),
    ]
    ext = S * np.concatenate([Rz, gRz])
    left, right, bottom, top = ext[:, 0].min(), ext[:, 0].max(), ext[:, 1].min(), ext[:, 1].max()
    gradB = np.array([-1.0, 0.0])  # |B| falls with R: its gradient points to smaller R
    base = np.array([right + 2.6, bottom + 1.0])
    items += [
        Arrow(tuple(base), tuple(base + 1.2 * gradB), "vector", role="grad_B"),
        Arrow(tuple(base + np.array([0.0, 1.1])), tuple(base + np.array([1.2, 1.1])), "vector", role="R_c"),
        Arrow((left - 0.6, bottom - 0.4), (left + 0.6, bottom - 0.4), "axis", role="axes"),
        Arrow((left - 0.6, bottom - 0.4), (left - 0.6, bottom + 0.8), "axis", role="axes"),
    ]
    if labels:
        items += [
            Label(tuple(base + 0.6 * gradB + np.array([0.0, 0.1])), "$\\nabla B$", "label", anchor="south",
                  role="grad_B"),
            Label(tuple(base + np.array([1.3, 1.1])), "$\\mathbf{R}_c$", "label", anchor="west", role="R_c"),
            Label(tuple(S * gRz[0] + 2.0 * _unit([v_d[0], v_d[2]]) + np.array([0.15, 0.0])),
                  "$\\mathbf{v}_{\\nabla B} + \\mathbf{v}_R$", "label",
                  anchor="west", role="drift"),
            Label(tuple(S * gRz[0] + np.array([0.35, -0.3])), "$\\mathbf{B}$ (into page)", "small label",
                  anchor="west", role="field_line"),
            Label((left + 0.65, bottom - 0.4), "$R$", anchor="west", role="axes"),
            Label((left - 0.6, bottom + 0.85), "$z$", anchor="south", role="axes"),
        ]
        center = 0.5 * (left + right + 1.4)
        items += _title("Curvature and $\\nabla B$ drift",
                        "looking along the field line: the gyration climbs at "
                        "$\\mathbf{v}_{\\nabla B} + \\mathbf{v}_R$", top + 1.9, center)
        items.append(Label((center, bottom - 1.2), "the same orbit as the 3-D view, in the $(R, z)$ plane",
                           "note", role="note"))
    return Scene(tuple(items))


def _curvature_top(figure: ParticleFigure, labels: bool) -> Scene:
    """Looking down z: the curved field line and the orbit wrapped around it."""
    S = 0.85
    R0 = figure.parameters["R0"]
    x = figure.orbits["ion"]
    phi = np.linspace(-0.1, 0.5 * math.pi + 0.1, 181)
    line = np.stack([R0 * np.cos(phi), R0 * np.sin(phi)], axis=-1)
    items: List = [
        Polyline.of(S * line, "field line", role="field_line"),
        Arrow(tuple(S * line[-8]), tuple(S * line[-1]), "field line arrow", role="field_line"),
        Polyline.of(S * x[:, :2], "orbit ion", role="ion_orbit"),
        Label((0.0, 0.0), "$\\odot$", "legend symbol", role="axis"),
    ]
    mid = S * line[len(line) // 2]
    inward = -_unit(mid)
    items += [
        Arrow(tuple(mid - 1.4 * inward), tuple(mid - 0.2 * inward), "vector", role="grad_B"),
        Arrow((0.0, 0.0), tuple(mid + 0.0 * inward), "vector", role="radius"),
    ]
    if labels:
        items += [
            Label(tuple(mid - 1.5 * inward), "$\\nabla B$", "label", anchor="south west", role="grad_B"),
            Label(tuple(0.5 * mid + np.array([0.2, -0.3])), "$\\mathbf{R}_c$", "label", role="radius"),
            Label((0.25, -0.25), "$z$ axis", "small label", anchor="north west", role="axis"),
            Label(tuple(S * line[-1] + np.array([-0.2, 0.3])), "$\\mathbf{B}$", "label", anchor="east",
                  role="field_line"),
        ]
        ext = S * np.concatenate([x[:, :2], line])
        center = 0.5 * (ext[:, 0].min() + ext[:, 0].max())
        items += _title("Curvature and $\\nabla B$ drift",
                        "looking down $z$: the field line curves about the axis and $|\\mathbf{B}|$ "
                        "grows towards it; the drift is out of the page", ext[:, 1].max() + 1.9, center)
        items.append(Label((center, -1.0), "the same orbit as the 3-D view, projected onto the midplane",
                           "note", role="note"))
    return Scene(tuple(items))


def _magnetization_3d(figure: ParticleFigure, labels: bool) -> Scene:
    """The edge orbits as helical columns along B (pointing down), with the edge current around them."""
    S = 0.55
    side = figure.parameters["side"]
    n = int(figure.parameters["n_centres"])
    edge = [(i, j) for i in range(n) for j in range(n) if i in (0, n - 1) or j in (0, n - 1)]
    items: List = []
    for key in edge:
        items.append(Polyline.of(project(figure.orbits[f"helix_{key[0]}_{key[1]}"], S), "orbit ion", role="orbit"))
    z_top, z_bot = 0.0, float(min(o[:, 2].min() for k, o in figure.orbits.items() if k.startswith("helix")))
    for z in (z_top, z_bot):
        box = np.array([[0, 0, z], [side, 0, z], [side, side, z], [0, side, z]])
        items.append(Polyline.of(project(box, S), "region box", role="region", closed=True))
    for cx, cy in ((0, 0), (side, 0), (side, side), (0, side)):
        items.append(Polyline.of(project([[cx, cy, z_top], [cx, cy, z_bot]], S), "region box", role="region"))
    # the edge current, as a loop at mid-height in the computed sense
    zc = 0.5 * (z_top + z_bot)
    sense = np.sign(figure.parameters["right_edge_current"])  # +: counter-clockwise seen from +z
    t = np.linspace(0.0, 2 * math.pi, 5)[:-1] + math.pi / 4
    corners = side / 2 + (side / 2 + 0.9) * math.sqrt(2) * np.stack([np.cos(t), np.sin(t)], axis=-1)
    corners = np.clip(corners, -0.9, side + 0.9)
    loop = np.concatenate([corners, corners[:1]])
    if sense < 0:
        loop = loop[::-1]
    for a, b in zip(loop[:-1], loop[1:]):
        pa = np.array([a[0], a[1], zc])
        pb = np.array([b[0], b[1], zc])
        items.append(Arrow(tuple(project(pa + 0.2 * (pb - pa), S)), tuple(project(pa + 0.8 * (pb - pa), S)),
                           "current", role="edge_current"))
    top_mid = np.array([side / 2, side / 2, 1.5])
    items.append(Arrow(tuple(project(top_mid, S)), tuple(project(top_mid + [0, 0, -3.0], S)), "field vector",
                       role="legend_B"))
    if labels:
        pts = np.concatenate([it.points for it in items if isinstance(it, Polyline)])
        center = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
        items += [
            Label(tuple(project(top_mid, S) + np.array([0.0, 0.15])), "$\\mathbf{B}$", "label", anchor="south",
                  role="legend_B"),
            Label(tuple(project(np.array([side + 0.9, side / 2, zc]), S) + np.array([0.3, 0.0])), "$\\mathbf{J}_M$",
                  "label", anchor="west", role="edge_current"),
        ]
        items += _title("Magnetization current",
                        "gyrating ions stream along $\\mathbf{B}$; around the edge their currents add up to "
                        "$\\mathbf{J}_M$", np.max(np.asarray(pts)[:, 1]) + 1.9, center)
        items.append(Label((center, np.min(np.asarray(pts)[:, 1]) - 0.8),
                           "edge orbits of the perpendicular view with a small parallel velocity; "
                           "loop sense from the binned orbit current", "note", role="note"))
    return Scene(tuple(items))


def _toroidal_poloidal(figure: ParticleFigure, labels: bool) -> Scene:
    """The textbook cross-section: (R, z) plane, B along phi (into the page)."""
    S, a = 2.2, 1.0
    v = figure.vectors
    Rhat, zhat = v["R_hat"], np.array([0.0, 0.0, 1.0])

    def rz(vec):  # a 3-D vector's (R, z) components
        return np.array([np.dot(vec, Rhat), np.dot(vec, zhat)])

    th = np.linspace(0.0, 2 * math.pi, 181)
    items: List = [Polyline.of(S * a * np.stack([np.cos(th), np.sin(th)], axis=-1), "section fill",
                               role="section", closed=True)]
    # B = +phi: with R to the right and z up, +phi points into the page
    phi_into_page = float(np.dot(np.cross(Rhat, zhat), v["B"])) < 0
    items.append(Label((-1.25 * S, 0.95 * S), "$\\otimes$" if phi_into_page else "$\\odot$", "legend symbol",
                       role="B"))
    up = float(np.sign(rz(v["ion_drift"])[1]))
    for ang in np.radians([58, 90, 122]):
        items.append(Label(tuple(0.72 * S * np.array([math.cos(ang), up * math.sin(ang)])), "$+$", "charge small",
                           role="ion_layer"))
        items.append(Label(tuple(0.72 * S * np.array([math.cos(ang), -up * math.sin(ang)])), "$-$", "charge small",
                           role="electron_layer"))
    E = _unit(rz(v["E"]))
    ion, ele, vE = _unit(rz(v["ion_drift"])), _unit(rz(v["electron_drift"])), _unit(rz(v["v_E"]))
    items += [
        Arrow(tuple(S * (np.array([-0.3, 0.0]) - 0.4 * E)), tuple(S * (np.array([-0.3, 0.0]) + 0.4 * E)),
              "field vector", role="E"),
        Arrow(tuple(S * (np.array([0.25, 0.05]) * [1, up])), tuple(S * (np.array([0.25, 0.0]) + 0.45 * ion)),
              "drift ion", role="ion_drift"),
        Arrow(tuple(S * (np.array([0.25, -0.05]) * [1, up])), tuple(S * (np.array([0.25, 0.0]) + 0.45 * ele)),
              "drift electron", role="electron_drift"),
        Arrow(tuple(S * 1.1 * vE), tuple(S * 1.1 * vE + 1.4 * vE), "exb", role="v_E"),
        Arrow((-S - 1.0, -S - 0.3), (-S + 0.1, -S - 0.3), "axis", role="axes"),
        Arrow((-S - 1.0, -S - 0.3), (-S - 1.0, -S + 0.8), "axis", role="axes"),
        Arrow(tuple(S * np.array([-1.35, 0.0])), tuple(S * np.array([-1.35, 0.0]) + 0.9 * np.array([-1.0, 0.0])),
              "vector", role="grad_B"),
    ]
    if labels:
        tip = S * 1.1 * vE + 1.4 * vE
        items += [
            Label(tuple(tip + np.array([0.15, 0.0])), "$\\mathbf{v}_E = \\dfrac{\\mathbf{E}\\times\\mathbf{B}}{B^2}$",
                  "label", anchor="west", role="v_E"),
            Label(tuple(S * np.array([-0.3, 0.0]) + np.array([-0.2, 0.0])), "$\\mathbf{E}$", "label", anchor="east",
                  role="E"),
            Label(tuple(S * (np.array([0.32, 0.0]) + 0.35 * ion)), "ion drift", "small label", anchor="west",
                  role="ion_drift"),
            Label(tuple(S * (np.array([0.32, 0.0]) + 0.35 * ele)), "electron drift", "small label", anchor="west",
                  role="electron_drift"),
            Label((-1.25 * S + 0.35, 0.95 * S), "$\\mathbf{B}_\\mathrm{toroidal}$", "label", anchor="west",
                  role="B"),
            Label(tuple(S * np.array([-1.35, 0.0]) + np.array([-1.0, 0.25])), "$\\nabla B$", "label", anchor="south",
                  role="grad_B"),
            Label((-S + 0.15, -S - 0.3), "$R$", anchor="west", role="axes"),
            Label((-S - 1.0, -S + 0.85), "$z$", anchor="south", role="axes"),
        ]
        items += _title("Toroidal drift and charge separation",
                        "the poloidal cross-section: drifts separate charge, and $\\mathbf{E}\\times\\mathbf{B}$ "
                        "is outward", S + 1.9, 0.6)
        items.append(Label((0.6, -S - 1.2), "directions from the drift formulas at this cross-section",
                           "note", role="note"))
    return Scene(tuple(items))


def _toroidal_top(figure: ParticleFigure, labels: bool) -> Scene:
    """Looking down z: circular field lines, the inward grad B, the vertical drifts out of the page."""
    R0, a = figure.parameters["aspect_ratio"], 1.0
    S = 1.2
    phi = np.linspace(0.0, 2 * math.pi, 241)
    items: List = []
    for R in (R0 - a, R0 + a):
        items.append(Polyline.of(S * R * np.stack([np.cos(phi), np.sin(phi)], axis=-1), "torus rim", role="torus",
                                 closed=True))
    v = figure.vectors
    sense = 1.0 if float(np.dot(np.cross(v["R_hat"], v["B"]), [0, 0, 1.0])) > 0 else -1.0  # counter-clockwise?
    for k, R in enumerate((R0 - 0.5 * a, R0, R0 + 0.5 * a)):
        start = 0.3 + 0.9 * k
        arc = np.linspace(start, start + 2 * math.pi - 0.6, 181) * sense
        items.append(Polyline.of(S * R * np.stack([np.cos(arc), np.sin(arc)], axis=-1), "field line",
                                 role="field_line"))
        items.append(Arrow(tuple(S * R * np.array([math.cos(arc[-6]), math.sin(arc[-6])])),
                           tuple(S * R * np.array([math.cos(arc[-1]), math.sin(arc[-1])])), "field line arrow",
                           role="field_line"))
    probe = S * R0 * np.array([0.0, 1.0])
    items += [
        Arrow(tuple(probe + np.array([0.0, 1.6])), tuple(probe + np.array([0.0, 0.4])), "vector", role="grad_B"),
        Label(tuple(S * np.array([R0, 0.0]) + np.array([0.0, 0.0])),
              "$\\odot$" if v["ion_drift"][2] > 0 else "$\\otimes$", "legend symbol", role="ion_drift"),
        Label((0.0, 0.0), "$\\odot$", "legend symbol", role="axis"),
    ]
    if labels:
        outer = S * (R0 + a)
        items += [
            Label(tuple(probe + np.array([0.15, 1.6])), "$\\nabla B$", "label", anchor="west", role="grad_B"),
            Arrow((outer + 0.9, -0.9), tuple(S * np.array([R0, 0.0]) + np.array([0.2, -0.2])), "leader",
                  role="ion_drift"),
            Label((outer + 0.95, -0.9), "\\begin{tabular}{l}ion drift out of the page,\\\\electrons into it\\end{tabular}",
                  "small label", anchor="north west", role="ion_drift"),
            Label(tuple((outer + 0.25) * np.array([math.cos(0.55), math.sin(0.55)])),
                  "$\\mathbf{B}_\\mathrm{toroidal}$", "label", anchor="south west", role="field_line"),
            Label((0.25, -0.25), "$z$", "small label", anchor="north west", role="axis"),
        ]
        items += _title("Toroidal drift and charge separation",
                        "looking down $z$: field lines are circles and $|\\mathbf{B}| \\propto 1/R$ grows inward",
                        outer + 1.8, 0.0)
        items.append(Label((0.0, -outer - 0.8), "the vertical drifts point out of this plane; see the poloidal view",
                           "note", role="note"))
    return Scene(tuple(items))
