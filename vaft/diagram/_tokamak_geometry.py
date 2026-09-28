"""Tokamak toroidal geometry and flux coordinates (#1073).

The parent geometry the cylindrical and slab reductions of #1062 start from:
the torus and its angles, nested flux surfaces with and without the
Shafranov shift, shaping, the $1/R$ toroidal field, the safety factor as a
winding number, flux coordinates and the straight-field-line angle.

Every surface is ``vaft.formula.miller_surface``; the shift is
``shafranov_shift_from_r_a_R0_beta_p_li``; the field is
``vacuum_toroidal_field``; $\\theta^*$ is ``straight_field_line_angle`` through
the island model's tabulation. Shapes and parameters are schematic.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from vaft.formula.equilibrium import (
    miller_surface,
    shafranov_shift_from_r_a_R0_beta_p_li,
    vacuum_toroidal_field,
)

from ._chart import CHART_WIDTH, Chart, render_chart
from ._projection import camera, project, split
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: drawn torus and cross-section: major and minor radius [m] and centimetres per metre
_R0, _A = 3.0, 1.0
_CM = 2.4
_TORUS_PROJECTIONS = ("3d", "poloidal")
_SHAPES = ("circular", "shifted")
#: schematic poloidal beta and internal inductance of the shifted equilibrium
_BETA_P, _L_I = 0.8, 1.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _xy(R, Z, R_ref=_R0):
    """Cross-section coordinates [cm] of ``(R, Z)``, centred on ``R_ref``."""
    return np.stack([(np.asarray(R) - R_ref) * _CM, np.asarray(Z) * _CM], axis=-1)


def _surface(r, kappa=1.0, delta=0.0, shift=0.0, n=241) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    R, Z = miller_surface(r, theta, _R0, kappa, delta, shift=shift)
    return _xy(R, Z)


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _below(items) -> float:
    """Half a centimetre under the lowest drawn point."""
    ys = [np.asarray(it.points)[:, 1].min() for it in items if hasattr(it, "points")]
    return float(min(ys)) - 0.5


#: centimetres per metre in the 3-D torus views
_SCALE_3D = 1.6


def _p3(points) -> np.ndarray:
    return project(points, scale=_SCALE_3D)


def _torus(r, theta, phi, R0=_R0):
    R = R0 + r * np.cos(theta)
    return np.stack([R * np.cos(phi), R * np.sin(phi), r * np.sin(theta)], axis=-1)


def _torus_normal(theta, phi):
    return np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], axis=-1)


def _visible(normal, points=None, r=_A) -> np.ndarray:
    """Facing the camera and, given the points, not hidden behind the torus of minor radius ``r``."""
    view, _, _ = camera()
    facing = normal @ view > 0
    if points is None:
        return facing
    t = np.linspace(0.02, 2.5 * (_R0 + r), 400)
    ray = points[:, None, :] + t[None, :, None] * view[None, None, :]
    inside = (np.hypot(ray[..., 0], ray[..., 1]) - _R0) ** 2 + ray[..., 2] ** 2 < (r * (1 - 1e-3)) ** 2
    return facing & ~inside.any(axis=1)


def _camera_cut() -> float:
    """The toroidal angle whose poloidal plane faces the camera (its normal, phi-hat, points at it)."""
    view, _, _ = camera()
    return math.atan2(view[1], view[0]) - 0.5 * math.pi


def _torus_wireframe(r=_A, rings=12) -> List:
    items: List = []
    t = np.linspace(0.0, 2.0 * math.pi, 121)
    for ph in np.linspace(0.0, 2.0 * math.pi, rings + 1)[:-1]:
        phi = np.full_like(t, ph)
        pts = _torus(r, t, phi)
        items += split(_p3(pts), _visible(_torus_normal(t, phi), pts, r), "surface", "mesh hidden", "surface")
    p = np.linspace(0.0, 2.0 * math.pi, 241)
    for th in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        theta = np.full_like(p, th)
        pts = _torus(r, theta, p)
        items += split(_p3(pts), _visible(_torus_normal(theta, p), pts, r), "surface", "mesh hidden", "surface")
    return items


# ---------------------------------------------------------------------------
# the torus and its angles
# ---------------------------------------------------------------------------


def tokamak_torus(projection: str = "3d", *, labels: bool = True) -> Diagram:
    r"""The tokamak torus: major radius $R_0$, minor radius $a$, toroidal $\phi$ and poloidal $\theta$.

    ``"3d"`` draws the torus with its symmetry axis, the magnetic axis and a
    poloidal cross-section; ``"poloidal"`` draws that cross-section in
    $(R, Z)$ with $R = R_0 + r\cos\theta$, $Z = r\sin\theta$. $\phi$ is
    counter-clockwise seen from above and $\theta$ runs from the outboard
    midplane towards the top, as in ``helical_phase``.
    """
    if projection not in _TORUS_PROJECTIONS:
        raise ValueError(f"projection must be one of {_TORUS_PROJECTIONS}, not {projection!r}")
    labels = _check_labels(labels)
    items: List = []
    if projection == "3d":
        phi_cut = _camera_cut()
        items += _torus_wireframe()
        t = np.linspace(0.0, 2.0 * math.pi, 121)
        section = _torus(_A, t, np.full_like(t, phi_cut))
        items.append(Polyline.of(_p3(section), "section fill", role="cross_section", closed=True))
        p = np.linspace(0.0, 2.0 * math.pi, 241)
        axis_ring = _torus(0.0, p, p)
        items += split(_p3(axis_ring), _visible(np.stack([np.cos(p), np.sin(p), 0 * p], -1)), "rational",
                       "mesh hidden", "magnetic_axis")
        items.append(Arrow(tuple(_p3(np.array([0.0, 0.0, -1.8]))), tuple(_p3(np.array([0.0, 0.0, 2.0]))),
                           "frame axis arrow", role="symmetry_axis"))
        centre = np.array([_R0 * math.cos(phi_cut), _R0 * math.sin(phi_cut), 0.0])
        items.append(Arrow(tuple(_p3(np.zeros(3))), tuple(_p3(centre)), "width arrow", role="major_radius",
                           both=True))
        rim = _torus(_A, 0.0, phi_cut)
        items.append(Arrow(tuple(_p3(centre)), tuple(_p3(rim)), "width arrow", role="minor_radius", both=True))
        # the toroidal angle, measured on the midplane from x
        arc = np.linspace(0.0, 0.9, 40)
        phi_arc = np.stack([1.3 * np.cos(arc), 1.3 * np.sin(arc), 0 * arc], -1)
        items.append(Polyline.of(_p3(phi_arc), "angle arc", role="toroidal_angle"))
        # the poloidal angle, in the cut
        th = np.linspace(0.0, 1.0, 30)
        theta_arc = _torus(0.45 * _A, th, np.full_like(th, phi_cut))
        items.append(Polyline.of(_p3(theta_arc), "angle arc", role="poloidal_angle"))
        if labels:
            items += [
                Label(tuple(_p3(0.5 * centre) + np.array([0.0, 0.2])), "$R_0$", "label", anchor="south",
                      role="major_radius"),
                Label(tuple(_p3(0.5 * (centre + rim)) + np.array([0.0, -0.15])), "$a$", "label", anchor="north",
                      role="minor_radius"),
                Label(tuple(_p3(np.array([1.55 * math.cos(0.45), 1.55 * math.sin(0.45), 0.0]))), "$\\phi$",
                      "label", anchor="center", role="toroidal_angle"),
                Label(tuple(_p3(_torus(0.75 * _A, 0.5, phi_cut))), "$\\theta$", "label", anchor="center",
                      role="poloidal_angle"),
                Label(tuple(_p3(np.array([0.0, 0.0, 2.05]))), "$Z$", "label", anchor="south",
                      role="symmetry_axis"),
                Label(tuple(_p3(_torus(0.0, 0.0, phi_cut + math.pi)) + np.array([0.0, 0.15])), "magnetic axis",
                      "small label", anchor="south", role="magnetic_axis"),
                Label(tuple(_p3(_torus(_A, -0.5 * math.pi, phi_cut + 0.5 * math.pi)) + np.array([0.0, -0.15])),
                      "surface $r = a$", "small label", anchor="north", role="flux_surface"),
                _note("$R = R_0 + r\\cos\\theta$, $Z = r\\sin\\theta$; $\\phi$ counter-clockwise from above",
                      float(_p3(np.zeros(3))[0]), _below(items)),
            ]
    else:
        boundary = _surface(_A)
        items += [
            Polyline.of(boundary, "lcfs", role="boundary", closed=True),
            Arrow((-_R0 * _CM - 0.3, 0.0), (1.6 * _A * _CM, 0.0), "chart axis", role="axes"),
            Arrow((-_R0 * _CM, -1.4 * _A * _CM), (-_R0 * _CM, 1.4 * _A * _CM), "chart axis", role="symmetry_axis"),
            Arrow((-_R0 * _CM, -0.9 * _A * _CM), (0.0, -0.9 * _A * _CM), "width arrow", role="major_radius", both=True),
            Marker((0.0, 0.0), "o", "opoint", role="magnetic_axis"),
        ]
        th = 0.9
        tip = (_A * _CM * math.cos(th), _A * _CM * math.sin(th))
        items += [Arrow((0.0, 0.0), tip, "vector", role="minor_radius"),
                  Polyline.of(np.stack([0.55 * np.cos(np.linspace(0, th, 30)),
                                        0.55 * np.sin(np.linspace(0, th, 30))], -1), "angle arc",
                              role="poloidal_angle")]
        if labels:
            items += [
                Label((-0.5 * _R0 * _CM, -0.9 * _A * _CM - 0.12), "$R_0$", "label", anchor="north",
                      role="major_radius"),
                Label((0.5 * tip[0] - 0.12, 0.5 * tip[1] + 0.05), "$r$", "label", anchor="south east",
                      role="minor_radius"),
                Label((0.8 * math.cos(th / 2), 0.8 * math.sin(th / 2)), "$\\theta$", "label", anchor="west",
                      role="poloidal_angle"),
                Label((1.6 * _A * _CM + 0.15, 0.0), "$R$", "label", anchor="west", role="axes"),
                Label((-_R0 * _CM, 1.4 * _A * _CM + 0.1), "$Z$", "label", anchor="south", role="symmetry_axis"),
                Label((-_R0 * _CM + 0.15, 1.2 * _A * _CM), "symmetry axis", "small label", anchor="west",
                      role="symmetry_axis"),
                _note("$R = R_0 + r\\cos\\theta$, $Z = r\\sin\\theta$; $\\theta$ from the outboard midplane",
                      -0.3 * _R0 * _CM, -1.6 * _A * _CM),
            ]
    model = {"projection": projection, "R0": _R0, "a": _A}
    return Diagram(f"tokamak_torus_{projection}", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# nested flux surfaces, with and without the Shafranov shift
# ---------------------------------------------------------------------------


def flux_surfaces(shape: str = "circular", *, labels: bool = True) -> Diagram:
    r"""Nested circular flux surfaces, concentric or Shafranov-shifted.

    ``"circular"``: $R = R_0 + r\cos\theta$, $Z = r\sin\theta$, the magnetic
    axis on the geometric axis. ``"shifted"``: each surface's centre moves
    out by ``shafranov_shift_from_r_a_R0_beta_p_li`` (schematic
    $\beta_p = 0.8$, $l_i = 1$), zero at the boundary and largest on the
    magnetic axis, which now sits outside the geometric axis.
    """
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {_SHAPES}, not {shape!r}")
    labels = _check_labels(labels)
    radii = _A * np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    shifts = (shafranov_shift_from_r_a_R0_beta_p_li(radii, _A, _R0, _BETA_P, _L_I) if shape == "shifted"
              else np.zeros_like(radii))
    axis_shift = (shafranov_shift_from_r_a_R0_beta_p_li(0.0, _A, _R0, _BETA_P, _L_I) if shape == "shifted" else 0.0)
    items: List = []
    for r, d in zip(radii, shifts):
        items.append(Polyline.of(_surface(r, shift=d), "lcfs" if r == _A else "surface", role="flux_surface",
                                 closed=True))
    geometric = (0.0, 0.0)
    magnetic = (axis_shift * _CM, 0.0)
    items += [Marker(magnetic, "o", "opoint", role="magnetic_axis"),
              Marker(geometric, "x", "xpoint", role="geometric_axis")]
    if labels:
        top = _A * _CM
        if shape == "shifted":
            below = -top - 0.35
            items += [
                Polyline.of([magnetic, (magnetic[0] + 0.9, below)], "leader line", role="magnetic_axis"),
                Label((magnetic[0] + 0.9, below - 0.05), "magnetic axis", "small label", anchor="north west",
                      role="magnetic_axis"),
                Polyline.of([geometric, (geometric[0] - 0.9, below)], "leader line", role="geometric_axis"),
                Label((geometric[0] - 0.9, below - 0.05), "geometric axis", "small label", anchor="north east",
                      role="geometric_axis"),
            ]
        else:
            th = 0.7
            tip = (0.8 * _A * _CM * math.cos(th), 0.8 * _A * _CM * math.sin(th))
            items += [
                Polyline.of([(0.0, 0.0), (-0.9, -top - 0.35)], "leader line", role="magnetic_axis"),
                Label((-0.9, -top - 0.4), "magnetic axis $=$ geometric axis, at $R_0$", "small label",
                      anchor="north", role="magnetic_axis"),
                Arrow((0.0, 0.0), tip, "vector", role="minor_radius"),
                Label((0.5 * tip[0] - 0.1, 0.5 * tip[1] + 0.1), "$r$", "small label", anchor="south east",
                      role="minor_radius"),
                Polyline.of(np.stack([0.5 * np.cos(np.linspace(0, th, 20)), 0.5 * np.sin(np.linspace(0, th, 20))], -1),
                            "angle arc", role="poloidal_angle"),
                Label((0.65 * math.cos(th / 2), 0.65 * math.sin(th / 2)), "$\\theta$", "small label",
                      anchor="west", role="poloidal_angle"),
                Label((-0.72 * _A * _CM, 0.72 * _A * _CM), "$\\psi = $ const", "small label", anchor="south east",
                      role="flux_surface"),
            ]
        if shape == "shifted":
            items += [Arrow((0.0, -0.9), (magnetic[0], -0.9), "width arrow", role="shift", both=True),
                      Label((0.5 * magnetic[0], -1.0), "$\\Delta(0)$", "small label", anchor="north", role="shift"),
                      _note("$R = R_0 + \\Delta(r) + r\\cos\\theta$: the shift grows inward, crowding the low-field side",
                            0.0, -top - 1.1)]
        else:
            items.append(_note("$R = R_0 + r\\cos\\theta$: concentric, the magnetic axis on the geometric axis",
                               0.0, -top - 1.1))
    model = {"shape": shape, "radii": radii, "shifts": shifts, "axis_shift": axis_shift,
             "R0": _R0, "a": _A, "beta_p": _BETA_P, "l_i": _L_I}
    return Diagram(f"flux_surfaces_{shape}", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# shaping
# ---------------------------------------------------------------------------

_FAMILY = (("circular", 1.0, 0.0), ("elongated", 1.7, 0.0), ("triangular", 1.0, 0.5),
           ("elongated and triangular", 1.7, 0.5))


def shaping_family(*, labels: bool = True) -> Diagram:
    r"""Elongation $\kappa$ and triangularity $\delta$ on one boundary family.

    Four cross-sections from ``miller_surface``: circular, $\kappa = 1.7$,
    $\delta = 0.5$, and both, each with nested surfaces at
    $\delta(r) = \delta\,r/a$ and constant $\kappa$. The top of a surface
    sits at $R_0 - \delta r$: positive triangularity makes the D.
    """
    labels = _check_labels(labels)
    items: List = []
    gap = 2.9 * _A * _CM / 1.1
    boundaries = {}
    for i, (name, kappa, delta) in enumerate(_FAMILY):
        x0 = i * gap
        for r in _A * np.array([0.35, 0.7, 1.0]):
            pts = _surface(r, kappa, delta * r / _A) * 0.9 + np.array([x0, 0.0])
            items.append(Polyline.of(pts, "lcfs" if r == _A else "surface", role=f"shape:{name}", closed=True))
            if r == _A:
                boundaries[name] = pts
        items.append(Marker((x0, 0.0), "o", "opoint", role=f"shape:{name}"))
        if labels:
            items.append(Label((x0, -1.95 * _A * _CM * 0.9), f"{name}\\\\ $\\kappa = {kappa:g}$, $\\delta = {delta:g}$",
                               "small label,align=center", anchor="north", role=f"shape:{name}"))
    if labels:
        items.append(_note("$R = R_0 + r\\cos(\\theta + \\arcsin\\delta\\,\\sin\\theta)$, $Z = \\kappa r\\sin\\theta$",
                           1.5 * gap, -2.9 * _A * _CM * 0.9))
    model = {"family": _FAMILY, "boundaries": boundaries}
    return Diagram("shaping_family", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# high-field side, low-field side
# ---------------------------------------------------------------------------


def hfs_lfs_field(*, labels: bool = True) -> Diagram:
    r"""$B_\phi \propto 1/R$: the high-field side is inboard, the low-field side outboard.

    The cross-section above shares its $R$ axis with $B_\phi(R)$ from
    ``vacuum_toroidal_field`` below: across the plasma the field falls from
    $B_0R_0/(R_0 - a)$ inboard to $B_0R_0/(R_0 + a)$ outboard.
    """
    labels = _check_labels(labels)
    B0 = 1.0
    R = np.linspace(_R0 - 1.5 * _A, _R0 + 1.5 * _A, 201)
    B = vacuum_toroidal_field(B0, _R0, R)
    chart = Chart(x_range=(float(R[0]), float(R[-1])), y_range=(0.0, 1.15 * float(B.max())))
    chart.curves["B_phi"] = np.stack([R, B], axis=-1)
    chart.curves["plasma_in"] = np.array([[_R0 - _A, 0.0], [_R0 - _A, 1.1 * B.max()]])
    chart.curves["plasma_out"] = np.array([[_R0 + _A, 0.0], [_R0 + _A, 1.1 * B.max()]])
    chart.parameters.update({"B_hfs": float(vacuum_toroidal_field(B0, _R0, _R0 - _A)),
                             "B_lfs": float(vacuum_toroidal_field(B0, _R0, _R0 + _A)), "B0": B0})
    ticks = (_R0 - _A, _R0, _R0 + _A)
    scene = render_chart(
        chart, x_label="$R$", y_label="$B_\\phi/B_0$",
        curve_styles={"plasma_in": "rational", "plasma_out": "rational", "B_phi": "boundary"},
        region_text={}, x_ticks=ticks, x_tick_text=("$R_0 - a$", "$R_0$", "$R_0 + a$"),
        y_ticks=(1.0,), y_tick_text=("$1$",),
    )
    # the cross-section, on the chart's R scale, above it
    to_cm = lambda RR, ZZ: np.stack([chart.to_cm(np.stack([RR, 0 * RR], -1))[..., 0],  # noqa: E731
                                     np.asarray(ZZ) * CHART_WIDTH / (R[-1] - R[0]) + 9.6], axis=-1)
    theta = np.linspace(0.0, 2.0 * math.pi, 241)
    Rb, Zb = miller_surface(_A, theta, _R0, 1.0, 0.0)
    items: List = [Polyline.of(to_cm(Rb, Zb), "lcfs", role="boundary", closed=True)]
    for r in (0.35, 0.7):
        Rs, Zs = miller_surface(r * _A, theta, _R0, 1.0, 0.0)
        items.append(Polyline.of(to_cm(Rs, Zs), "surface", role="flux_surface", closed=True))
    if labels:
        hfs = to_cm(np.array([_R0 - _A]), np.array([0.0]))[0]
        lfs = to_cm(np.array([_R0 + _A]), np.array([0.0]))[0]
        items += [
            Label((float(hfs[0]) - 0.2, float(hfs[1])), "HFS\\\\ strong $B_\\phi$", "label,align=center",
                  anchor="east", role="hfs"),
            Label((float(lfs[0]) + 0.2, float(lfs[1])), "LFS\\\\ weak $B_\\phi$", "label,align=center",
                  anchor="west", role="lfs"),
            Label((CHART_WIDTH / 2, -1.25), "$B_\\phi = B_0R_0/R$: stronger inboard, weaker outboard", "note",
                  anchor="north", role="note"),
        ]
    return Diagram("hfs_lfs_field", scene + Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# safety factor as a winding number
# ---------------------------------------------------------------------------


def safety_factor_winding(q: int = 3, *, labels: bool = True) -> Diagram:
    r"""$q$ is the number of toroidal turns a field line makes per poloidal turn.

    A straight field line $\theta = \phi/q$ on a circular surface, followed
    for one poloidal transit, crosses a cross-section facing the camera $q$ times; the
    crossings are numbered in order. $q$ is a positive integer from 1 to 5 here.
    """
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or not 1 <= q <= 5:
        raise ValueError(f"q must be an integer from 1 to 5, not {q!r}")
    labels = _check_labels(labels)
    phi_cut = _camera_cut()
    items: List = _torus_wireframe()
    t = np.linspace(0.0, 2.0 * math.pi, 121)
    items.append(Polyline.of(_p3(_torus(_A, t, np.full_like(t, phi_cut))), "cut", role="cross_section",
                             closed=True))
    phi = np.linspace(phi_cut, phi_cut + 2.0 * math.pi * q, 1200 * q + 1)
    theta = (phi - phi_cut) / q
    line = _torus(_A, theta, phi)
    items += split(_p3(line), _visible(_torus_normal(theta, phi), line), "field line", "field line hidden",
                   "field_line")
    crossings = []
    for k in range(q + 1):
        point = _torus(_A, 2.0 * math.pi * k / q, phi_cut)
        crossings.append(point)
        if k < q:
            items.append(Marker(tuple(_p3(point)), "o", "opoint", role="crossing"))
            if labels:
                outward = _p3(point) - _p3(np.array([_R0 * math.cos(phi_cut), _R0 * math.sin(phi_cut), 0.0]))
                outward = outward / (np.linalg.norm(outward) or 1.0)
                items.append(Label(tuple(_p3(point) + 0.5 * outward), f"${k}$" if k else "$0$",
                                   "small label", anchor="center", role="crossing"))
    if labels:
        items.append(_note(f"$q = {q}$: {q} toroidal turn{'s' if q > 1 else ''} per poloidal turn; "
                           "the crossings of one cross-section are numbered", float(_p3(np.zeros(3))[0]), _below(items)))
    model = {"q": q, "field_line": line, "crossings": np.array(crossings), "phi_cut": phi_cut}
    return Diagram("safety_factor_winding", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# flux coordinates and the straight-field-line angle
# ---------------------------------------------------------------------------

#: the shaped surfaces of the coordinate figures: the reference island's D shape
_KAPPA, _DELTA, _ASPECT = 1.7, 0.4, 3.2


def _shaped_model():
    """The island model's surfaces, whose theta* table is ``straight_field_line_angle``'s."""
    from ._magnetic_island import _validate

    return _validate(3, 2, 0.16, 0.0, "poloidal", 0.55, _ASPECT, _KAPPA, _DELTA)


def flux_coordinates(*, labels: bool = True) -> Diagram:
    r"""Flux coordinates $(\psi, \theta, \phi)$ on a shaped cross-section.

    $\psi$ labels the nested surfaces, $\theta$ runs around each from the
    outboard midplane, and $\phi$ is the ignorable toroidal angle: with $R$
    to the right and $Z$ up, $+\phi$ points into the page. A coordinate line
    of constant $\theta$ crosses every surface.
    """
    labels = _check_labels(labels)
    model = _shaped_model()
    scale = _CM * 1.6
    items: List = []
    radii = np.array([0.25, 0.5, 0.75, 1.0])
    theta = np.linspace(0.0, 2.0 * math.pi, 241)
    for r in radii:
        pts = model.section(r, theta) * scale
        items.append(Polyline.of(pts, "lcfs" if r == 1.0 else "surface", role="flux_surface", closed=True))
    th0 = 0.8
    line = np.array([model.section(r, th0) for r in np.linspace(0.0, 1.0, 41)]) * scale
    items += [Polyline.of(line, "rational", role="theta_line"), Marker((0.0, 0.0), "o", "opoint", role="magnetic_axis"),
              Label((-0.35 * scale, 0.0), "$\\otimes$", "label", anchor="center", role="toroidal_angle")]
    arc = np.array([model.section(0.25, t) for t in np.linspace(0.0, th0, 20)]) * scale
    items.append(Polyline.of(arc, "angle arc", role="poloidal_angle"))
    if labels:
        for r in radii[1:]:
            p = model.section(r, math.pi * 1.25) * scale
            items.append(Label(tuple(p + np.array([-0.08, -0.08])), f"$\\psi_{{{int(r * 4)}}}$", "small label",
                               anchor="north east", role="flux_surface"))
        mid = model.section(0.32, th0 / 2) * scale
        items += [
            Label(tuple(mid + np.array([0.1, 0.0])), "$\\theta$", "label", anchor="west", role="poloidal_angle"),
            Label(tuple(line[-1] + np.array([0.1, 0.1])), "$\\theta = $ const", "small label", anchor="south west",
                  role="theta_line"),
            Label((-0.35 * scale, -0.3), "$+\\phi$", "small label", anchor="north", role="toroidal_angle"),
            _note("$\\psi$ = const: a flux surface; $\\theta$ around it; $\\phi$ into the page ($R$, $\\phi$, $Z$ right-handed)",
                  0.0, -_KAPPA * scale - 0.4),
        ]
    return Diagram("flux_coordinates", Scene(tuple(items)), model={"radii": radii, "theta_line": th0})


def poloidal_angle_comparison(*, labels: bool = True) -> Diagram:
    r"""The geometric poloidal angle and the straight-field-line angle $\theta^*$ are different coordinates.

    On the shaped surfaces, dashed rays are equal steps of the geometric
    angle measured from the magnetic axis; solid curves are equal steps of
    the PEST $\theta^*$ of ``straight_field_line_angle``, along which a field
    line advances uniformly. They coincide only on circular surfaces at
    infinite aspect ratio; here $\theta^*$ lines crowd towards the inboard side.
    """
    labels = _check_labels(labels)
    model = _shaped_model()
    scale = _CM * 1.6
    items: List = []
    theta = np.linspace(0.0, 2.0 * math.pi, 241)
    for r in (0.5, 1.0):
        items.append(Polyline.of(model.section(r, theta) * scale, "lcfs" if r == 1.0 else "surface",
                                 role="flux_surface", closed=True))
    r_grid = np.linspace(0.02, 1.0, 50)
    star_lines = {}
    for k in range(12):
        target = 2.0 * math.pi * k / 12
        # geometric ray: straight from the axis to where the boundary has this polar angle
        boundary = model.section(1.0, theta)
        polar = np.mod(np.arctan2(boundary[:, 1], boundary[:, 0]), 2.0 * math.pi)
        j = int(np.argmin(np.abs(np.angle(np.exp(1j * (polar - target))))))
        items.append(Polyline.of([(0.0, 0.0), tuple(boundary[j] * scale)], "approx", role="geometric_angle"))
        pts = np.array([model.section(r, model.theta_from_star(r, target)) for r in r_grid])
        star_lines[k] = pts
        items.append(Polyline.of(pts * scale, "field line", role="straight_field_line_angle"))
    items.append(Marker((0.0, 0.0), "o", "opoint", role="magnetic_axis"))
    if labels:
        items += [
            Label((1.15 * scale, 0.9 * scale), "solid: equal steps of $\\theta^*$", "small label", anchor="west",
                  role="straight_field_line_angle"),
            Label((1.15 * scale, 0.7 * scale), "dashed: equal steps of the geometric angle", "small label",
                  anchor="west", role="geometric_angle"),
            _note(f"$\\kappa = {_KAPPA:g}$, $\\delta = {_DELTA:g}$, $R_0/a = {_ASPECT:g}$; "
                  "$\\theta^*$ from straight\\_field\\_line\\_angle (PEST)", 0.0, -_KAPPA * scale - 0.4),
        ]
    return Diagram("poloidal_angle_comparison", Scene(tuple(items)),
                   model={"theta_star_lines": star_lines, "island_model": model})


def unwrapped_flux_surface(q: float = 2.5, *, labels: bool = True) -> Diagram:
    r"""On an unwrapped flux surface a field line is straight in $(\theta^*, \phi)$: $d\phi/d\theta^* = q$.

    The solid lines are one field line of safety factor $q$ in
    straight-field-line coordinates, wrapped at $\phi = 2\pi$. The dashed
    curve is the same field line against the shaped surface's
    parametrisation angle $\theta$ (the $r = 0.55$ surface of the reference
    D shape): straight in a straight-field-line angle, not in $\theta$.
    """
    try:
        q = float(q)
    except (TypeError, ValueError):
        raise ValueError(f"q must be a number, not {q!r}") from None
    if not (math.isfinite(q) and 0.5 <= q <= 6.0):
        raise ValueError(f"q must lie in [0.5, 6] for a readable figure, not {q!r}")
    labels = _check_labels(labels)
    model = _shaped_model()
    two_pi = 2.0 * math.pi
    chart = Chart(x_range=(0.0, two_pi), y_range=(0.0, two_pi))
    theta_star = np.linspace(0.0, two_pi, 1201)
    phi = q * theta_star
    wraps = np.floor(phi / two_pi).astype(int)
    theta_geo = model.theta_from_star(0.55, theta_star)
    for k in np.unique(wraps):
        sel = wraps == k
        if sel.sum() < 2:
            continue
        chart.curves[f"straight_{k}"] = np.stack([theta_star[sel], phi[sel] - two_pi * k], axis=-1)
        # the same points against the parametrisation angle
        chart.curves[f"parametrisation_{k}"] = np.stack([theta_geo[sel], phi[sel] - two_pi * k], axis=-1)
    chart.parameters.update({"q": q})
    ticks = (0.0, math.pi, two_pi)
    tick_text = ("$0$", "$\\pi$", "$2\\pi$")
    styles = {name: ("field line" if name.startswith("straight") else "approx") for name in chart.curves}
    scene = render_chart(chart, x_label="$\\theta^*$ (solid), $\\theta$ (dashed)", y_label="$\\phi$",
                         curve_styles=styles, region_text={}, x_ticks=ticks, y_ticks=ticks,
                         x_tick_text=tick_text, y_tick_text=tick_text)
    items = [Polyline(it.points, it.style,
                      "field_line" if it.role.startswith("straight") else "parametrisation_angle", it.closed)
             if isinstance(it, Polyline) and it.role.startswith(("straight", "parametrisation")) else it
             for it in scene.items]
    if labels:
        items.append(Label((CHART_WIDTH / 2, -1.45),
                           f"$d\\phi/d\\theta^* = q = {q:g}$: not straight in the parametrisation angle $\\theta$",
                           "note", anchor="north", role="note"))
    return Diagram("unwrapped_flux_surface", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# field-line pitch
# ---------------------------------------------------------------------------


def field_line_pitch(q: float = 1.0, *, labels: bool = True) -> Diagram:
    r"""A field line on a toroidal surface and its field $\mathbf B = B_\phi\hat{\boldsymbol\phi} + B_\theta\hat{\boldsymbol\theta}$.

    The line $\theta = \phi/q$ is followed for one toroidal transit; at one
    point the toroidal and poloidal components are drawn with
    $B_\theta/B_\phi = r/(qR)$, the ratio that keeps $\mathbf B$ tangent to
    the line, so their sum is the line's direction. Small $q$ makes the pitch
    visible; the default is $q = 1$.
    """
    try:
        q = float(q)
    except (TypeError, ValueError):
        raise ValueError(f"q must be a number, not {q!r}") from None
    if not (math.isfinite(q) and 0.5 <= q <= 5.0):
        raise ValueError(f"q must lie in [0.5, 5], not {q!r}")
    labels = _check_labels(labels)
    phi_cut = _camera_cut()
    items: List = _torus_wireframe()
    phi = np.linspace(phi_cut, phi_cut + 2.0 * math.pi, 1201)
    theta = 0.35 + (phi - phi_cut) / q
    line = _torus(_A, theta, phi)
    items += split(_p3(line), _visible(_torus_normal(theta, phi), line), "field line", "field line hidden",
                   "field_line")
    # the point: a visible stretch a little way along the line
    normal = _torus_normal(theta, phi)
    visible = _visible(normal, line)
    view, _, _ = camera()
    facing = np.where(visible, normal @ view, -np.inf)
    facing[:20] = facing[-20:] = -np.inf  # keep the tangent's neighbours on the line
    j = int(np.argmax(facing))  # the most camera-facing visible point
    P, th, ph = line[j], theta[j], phi[j]
    R = _R0 + _A * math.cos(th)
    phi_hat = np.array([-math.sin(ph), math.cos(ph), 0.0])
    theta_hat = np.array([-math.sin(th) * math.cos(ph), -math.sin(th) * math.sin(ph), math.cos(th)])
    B_phi = 1.2
    B_theta = B_phi * _A / (q * R)
    B = B_phi * phi_hat + B_theta * theta_hat
    tangent = (line[j + 1] - line[j - 1]) / np.linalg.norm(line[j + 1] - line[j - 1])
    items += [
        Arrow(tuple(_p3(P)), tuple(_p3(P + B_phi * phi_hat)), "drift ion", role="B_phi"),
        Arrow(tuple(_p3(P)), tuple(_p3(P + B_theta * theta_hat)), "drift", role="B_theta"),
        Arrow(tuple(_p3(P)), tuple(_p3(P + B)), "field vector", role="B"),
        Marker(tuple(_p3(P)), "o", "opoint", role="B"),
    ]
    if labels:
        items += [
            Label(tuple(_p3(P + 1.08 * B_phi * phi_hat)), "$B_\\phi\\hat{\\boldsymbol\\phi}$", "small label",
                  anchor="west", role="B_phi"),
            Label(tuple(_p3(P + 1.3 * B_theta * theta_hat) + np.array([0.0, 0.1])),
                  "$B_\\theta\\hat{\\boldsymbol\\theta}$", "small label", anchor="south", role="B_theta"),
            Label(tuple(_p3(P + 1.05 * B) + np.array([0.1, 0.1])), "$\\mathbf B$", "label", anchor="south west",
                  role="B"),
            _note(f"$B_\\theta/B_\\phi = r/(qR)$ keeps $\\mathbf B$ along the line; $q = {q:g}$, one toroidal transit",
                  float(_p3(np.zeros(3))[0]), _below(items)),
        ]
    model = {"q": q, "field_line": line, "point": P, "B": B, "tangent": tangent, "B_theta_over_B_phi": B_theta / B_phi}
    return Diagram("field_line_pitch", Scene(tuple(items)), model=model)
