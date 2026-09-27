"""Geometric approximations: which geometry, which ordering, and how modes map (#1062).

``geometry_ordering_map``
    slab, cylindrical and toroidal geometry on one axis, local / large aspect
    ratio / global ordering on the other, and the reductions between them,
    each arrow naming what it keeps or drops;
``field_line_geometry``
    the same field-line structure as a torus, a straightened cylinder and a
    local sheared slab (``geometry=``), so what survives each reduction shows;
``mode_number_mapping``
    $(m, n) \\to (k_y, k_z)$: $k_\\parallel(r)$ of an $m/n$ harmonic in a
    cylinder, its local sheared-slab tangent, and $q(r_s) = m/n$ as
    $k_\\parallel = 0$.

The physics is :mod:`vaft.formula.geometry`; profiles and radii are schematic.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from vaft.formula.geometry import (
    cylindrical_parallel_wavenumber,
    cylindrical_safety_factor_from_r_B,
    local_slab_from_cylinder,
    sheared_slab_field,
    sheared_slab_parallel_wavenumber,
)

from ._chart import CHART_WIDTH, Chart, render_chart
from ._concept import band, box, connector
from ._equations import formula_equation
from ._projection import camera, project, split
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene
from ._tearing import _schematic_q

GEOMETRIES = ("toroidal", "cylindrical", "slab")


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


# ---------------------------------------------------------------------------
# geometry x ordering
# ---------------------------------------------------------------------------

_COLUMNS = {"slab": 0.0, "cylindrical": 7.2, "toroidal": 15.6}
_ROWS = {"global": 0.0, "large_aspect_ratio": -3.2, "local": -7.2}


def _anchor_from(direction) -> str:
    dx, dy = direction
    vertical = "south" if dy > 0.38 else ("north" if dy < -0.38 else "")
    horizontal = "west" if dx > 0.38 else ("east" if dx < -0.38 else "")
    return " ".join(p for p in (vertical, horizontal) if p) or "center"


def geometry_ordering_map(*, labels: bool = True) -> Diagram:
    r"""Geometry and ordering are two axes, not one hierarchy.

    Columns are geometries (slab, cylindrical, toroidal), bands are orderings
    (global; large aspect ratio $\epsilon = a/R_0 \ll 1$; local). Each arrow is
    a reduction and names what it keeps or discards: the cylindrical tokamak
    is the $O(1)$ limit of a large-aspect-ratio torus, a local slab can come
    from the torus directly, and a curved slab keeps a $1/R_0$ curvature.
    """
    labels = _check_labels(labels)
    W = 5.0
    items: List = []
    left, right = -3.2, 18.8
    items += band(left, right, -1.0, 1.0, "global", role="ordering:global")
    items += band(left, right, -4.3, -2.1, "large aspect ratio, $\\epsilon = a/R_0 \\ll 1$",
                  role="ordering:large_aspect_ratio")
    items += band(left, right, -9.4, -4.8, "local, $|x| \\ll r$", role="ordering:local")
    for name, x in _COLUMNS.items():
        items.append(Label((x, 1.35), name, "subtitle", anchor="south", role=f"geometry:{name}"))
    col, row = _COLUMNS, _ROWS
    boxes = {
        "general_torus": box(col["toroidal"], row["global"], W, 1.1, "general axisymmetric torus",
                             role="representation:general_torus"),
        "screw_pinch": box(col["cylindrical"], row["global"], W, 1.1, "screw pinch",
                           role="representation:screw_pinch"),
        "lar_torus": box(col["toroidal"], row["large_aspect_ratio"], W, 1.1, "large-aspect-ratio torus",
                         role="representation:lar_torus"),
        "cyl_tokamak": box(col["cylindrical"], row["large_aspect_ratio"], W, 1.1, "cylindrical tokamak",
                           role="representation:cylindrical_tokamak"),
        "straight_slab": box(col["slab"], -5.9, W, 0.95, "straight slab", role="representation:straight_slab"),
        "sheared_slab": box(col["slab"], -7.2, W, 0.95, "sheared slab", role="representation:sheared_slab"),
        "curved_slab": box(col["slab"], -8.5, W, 0.95, "curved slab", role="representation:curved_slab"),
    }
    for b in boxes.values():
        items += list(b.items)
    # (start, end, label, position along the arrow, side: +1 left of travel, -1 right)
    paths = [
        ("general_torus", "lar_torus", "$\\epsilon \\ll 1$, keep $O(\\epsilon)$", 0.5, -1),
        ("lar_torus", "cyl_tokamak", "keep $O(1)$:\\\\ $R \\simeq R_0$, $z = R_0\\phi$", 0.5, -1),
        ("cyl_tokamak", "sheared_slab", "expand near $r_s$: keep $q_s$, $\\hat s$;\\\\ drop the radial profile",
         0.45, -1),
    ]
    for start, end, text, t, side in paths:
        arrow = connector(boxes[start], boxes[end], role=f"path:{start}->{end}")
        items.append(arrow)
        if labels:
            a0, a1 = np.array(arrow.start), np.array(arrow.end)
            d = (a1 - a0) / np.linalg.norm(a1 - a0)
            normal = side * np.array([-d[1], d[0]])
            at = a0 + t * (a1 - a0) + 0.2 * normal
            items.append(Label(tuple(at), text, "small label,align=center", anchor=_anchor_from(normal),
                               role=f"path:{start}->{end}"))
    # the torus reaches the local slabs directly -- locality, not aspect ratio, defines them.
    # The route runs down the right margin and along the local band, clear of every box.
    lane = right - 0.35
    general = boxes["general_torus"]
    items.append(Polyline.of([(general.x + 0.5 * general.width, general.y), (lane, general.y),
                              (lane, boxes["curved_slab"].y)], "connector line",
                             role="path:general_torus->local"))
    for name, text in (("sheared_slab", "local expansion of the torus: no cylinder, no $\\epsilon \\ll 1$ needed"),
                       ("curved_slab", "local, keep the torus curvature $\\kappa \\simeq 1/R_0$")):
        target = boxes[name]
        end = (target.x + 0.5 * target.width + 0.08, target.y)
        items.append(Arrow((lane, target.y), end, "connector", role=f"path:general_torus->{name}"))
        if labels:
            items.append(Label((0.5 * (lane + end[0]), target.y + 0.12), text, "small label", anchor="south",
                               role=f"path:general_torus->{name}"))
    paths += [("general_torus", "sheared_slab", "", 0, 0), ("general_torus", "curved_slab", "", 0, 0)]
    if labels:
        items.append(Label((0.5 * (left + right), -9.8),
                           "Geometry (columns) and ordering (bands) are independent; each arrow names what it keeps",
                           "note", anchor="north", role="note"))
    model = {"columns": dict(_COLUMNS), "boxes": {k: (b.x, b.y) for k, b in boxes.items()},
             "paths": tuple((a, b) for a, b, *_ in paths)}
    return Diagram("geometry_ordering_map", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# field-line geometry
# ---------------------------------------------------------------------------

#: the surface drawn in each geometry: minor radius, major radius, safety factor
_R0, _A, _Q = 2.2, 1.0, 1.5


def _torus_point(r, theta, phi, R0=_R0):
    R = R0 + r * np.cos(theta)
    return np.stack([R * np.cos(phi), R * np.sin(phi), r * np.sin(theta)], axis=-1)


def _visible(points, normals) -> np.ndarray:
    view, _, _ = camera()
    return normals @ view > 0


def _field_line_torus(labels: bool) -> Dict:
    theta0 = 0.0
    phi = np.linspace(0.0, 2.0 * math.pi, 601)
    theta = theta0 + phi / _Q  # one toroidal transit of a straight field line on a circular surface
    line = _torus_point(_A, theta, phi)
    normal = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], axis=-1)
    items: List = []
    for ph in np.linspace(0.0, 2.0 * math.pi, 9)[:-1]:
        t = np.linspace(0.0, 2.0 * math.pi, 121)
        ring = _torus_point(_A, t, np.full_like(t, ph))
        n_ring = np.stack([np.cos(t) * np.cos(ph), np.cos(t) * np.sin(ph), np.sin(t)], axis=-1)
        items += split(project(ring), _visible(ring, n_ring), "surface", "mesh hidden", "surface")
    for th in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        p = np.linspace(0.0, 2.0 * math.pi, 241)
        rim = _torus_point(_A, np.full_like(p, th), p)
        n_rim = np.stack([np.cos(th) * np.cos(p), np.cos(th) * np.sin(p), np.full_like(p, np.sin(th))], axis=-1)
        items += split(project(rim), _visible(rim, n_rim), "surface", "mesh hidden", "surface")
    items += split(project(line), _visible(line, normal), "field line", "mesh hidden", "field_line")
    return {"items": items, "line": line, "pitch": 1.0 / _Q,
            "caption": f"torus, one transit: $\\theta = \\phi/q$ with $q = {_Q:g}$; $1/R$ and toroidal curvature kept"}


def _field_line_cylinder(labels: bool) -> Dict:
    L = 2.0 * math.pi * _R0
    B_z = 1.0
    B_theta = _A * B_z / (_Q * _R0)
    q = cylindrical_safety_factor_from_r_B(_A, B_theta, B_z, _R0)
    z = np.linspace(0.0, L, 1201)
    theta = z / (q * _R0)
    # the cylinder axis lies along x on screen-friendly axes: (z, r cos, r sin)
    line = np.stack([z - L / 2, _A * np.cos(theta), _A * np.sin(theta)], axis=-1)
    normal = np.stack([np.zeros_like(z), np.cos(theta), np.sin(theta)], axis=-1)
    items: List = []
    for zz in np.linspace(0.0, L, 7):
        t = np.linspace(0.0, 2.0 * math.pi, 121)
        ring = np.stack([np.full_like(t, zz - L / 2), _A * np.cos(t), _A * np.sin(t)], axis=-1)
        n_ring = np.stack([np.zeros_like(t), np.cos(t), np.sin(t)], axis=-1)
        items += split(project(ring), _visible(ring, n_ring), "surface", "mesh hidden", "surface")
    for th in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        gen = np.stack([z - L / 2, np.full_like(z, _A * math.cos(th)), np.full_like(z, _A * math.sin(th))], axis=-1)
        n_gen = np.broadcast_to([0.0, math.cos(th), math.sin(th)], gen.shape)
        items += split(project(gen), _visible(gen, n_gen), "surface", "mesh hidden", "surface")
    items += split(project(line), _visible(line, normal), "field line", "mesh hidden", "field_line")
    return {"items": items, "line": line, "pitch": 1.0 / q,
            "caption": f"cylinder, one period $2\\pi R_0$: same $q = {q:g}$; no $1/R$, no toroidal curvature"}


def _field_line_slab(labels: bool) -> Dict:
    L_s, B0 = -3.0, 1.0
    xs = (-1.3, 0.0, 1.3)
    length, width = 6.0, 3.0
    items: List = []
    lines = {}
    for x in xs:
        corners = np.array([[x, -width / 2, -length / 2], [x, width / 2, -length / 2],
                            [x, width / 2, length / 2], [x, -width / 2, length / 2]])
        # slab axes on screen: (y, z, x) -> draw z along the page's long direction
        items.append(Polyline.of(project(corners[:, [2, 1, 0]]), "surface", role="surface", closed=True))
        B = sheared_slab_field(x, B0, L_s)
        direction = B / np.linalg.norm(B)
        for y0 in (-0.4 * width / 2, 0.4 * width / 2):
            t = np.linspace(-length / 2, length / 2, 2)
            pts = np.stack([np.full_like(t, x), y0 + direction[1] / direction[2] * t, t], axis=-1)
            inside = np.abs(pts[:, 1]) <= width / 2 + 1e-9
            pts = pts if inside.all() else _clip_to_width(pts, width)
            items.append(Polyline.of(project(pts[:, [2, 1, 0]]), "field line", role="field_line"))
            lines.setdefault(x, []).append(pts)
    if labels:
        for x in xs:
            edge = project(np.array([[length / 2, width / 2, x]]))[0]
            text = "$x = 0$" if x == 0 else ("$x > 0$" if x > 0 else "$x < 0$")
            items.append(Label((float(edge[0]) + 0.15, float(edge[1])), text, "small label", anchor="west",
                               role="surface"))
    return {"items": items, "line": np.vstack([p for v in lines.values() for p in v]),
            "pitch": {x: float(sheared_slab_field(x, B0, L_s)[1] / B0) for x in xs},
            "caption": "sheared slab: $\\mathbf B = B_0(\\hat z + (x/L_s)\\,\\hat y)$; tilt grows with $x$, no field-line curvature"}


def _clip_to_width(pts: np.ndarray, width: float) -> np.ndarray:
    y0, y1 = pts[0, 1], pts[-1, 1]
    t0, t1 = pts[0, 2], pts[-1, 2]
    slope = (y1 - y0) / (t1 - t0)
    lo = max(t0, t0 + ((-width / 2 if slope > 0 else width / 2) - y0) / slope) if slope else t0
    hi = min(t1, t0 + ((width / 2 if slope > 0 else -width / 2) - y0) / slope) if slope else t1
    t = np.array([lo, hi])
    return np.stack([np.full_like(t, pts[0, 0]), y0 + slope * (t - t0), t], axis=-1)


def field_line_geometry(geometry: str = "toroidal", *, labels: bool = True) -> Diagram:
    r"""The same field-line structure in toroidal, cylindrical and local sheared-slab geometry.

    ``"toroidal"``: a straight field line $\theta = \phi/q$ on a circular
    surface of a torus. ``"cylindrical"``: the torus straightened at $R_0$,
    $z = R_0\phi$, with the same $q$ from ``cylindrical_safety_factor_from_r_B``.
    ``"slab"``: three planes of a sheared slab, field lines along
    ``sheared_slab_field``, their tilt growing with $x$. Each drops what its
    reduction discards: $1/R$ and curvature, then the global radius.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, not {geometry!r}")
    labels = _check_labels(labels)
    built = {"toroidal": _field_line_torus, "cylindrical": _field_line_cylinder,
             "slab": _field_line_slab}[geometry](labels)
    items: List = list(built["items"])
    if labels:
        xy = np.vstack([np.asarray(it.points) for it in items if hasattr(it, "points")])
        items.append(Label((float(xy[:, 0].mean()), float(xy[:, 1].min()) - 0.6), built["caption"], "note",
                           anchor="north", role="note"))
    model = {"geometry": geometry, "field_line": built["line"], "pitch": built["pitch"]}
    return Diagram(f"field_line_geometry_{geometry}", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# mode-number mapping
# ---------------------------------------------------------------------------


def mode_number_mapping(m: int = 2, n: int = 1, *, labels: bool = True) -> Diagram:
    r"""$(m, n) \to (k_y, k_z)$: the rational surface is where $k_\parallel = 0$.

    $k_\parallel(r)$ of the $m/n$ harmonic in a cylinder with a schematic
    monotonic $q(r)$ (``cylindrical_parallel_wavenumber``) crosses zero at
    $q(r_s) = m/n$; the field-aligned local slab of ``local_slab_from_cylinder``
    is its tangent there, $k_\parallel \simeq k_z + k_y x/L_s$ with $x = r - r_s$
    and $k_z = 0$.
    """
    for name, value in (("m", m), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer mode number, not {value!r}")
    labels = _check_labels(labels)
    R0, a = 3.0, 1.0
    q_s = m / n
    q0, qa = _schematic_q(q_s)
    r = np.linspace(0.02, 1.0, 241) * a
    q = q0 + (qa - q0) * (r / a) ** 2
    r_s = a * math.sqrt((q_s - q0) / (qa - q0))
    s_hat = 2.0 * (qa - q0) * (r_s / a) ** 2 / q_s
    k_par = cylindrical_parallel_wavenumber(m, n, q, R0)
    k_y, L_s, k_z = local_slab_from_cylinder(m, n, r_s, R0, q_s, s_hat)
    span = float(np.max(np.abs(k_par)))
    chart = Chart(x_range=(0.0, 1.12 * a), y_range=(-1.15 * span, 1.15 * span))
    # the tangent, cut to the chart: at most a quarter radius each side, never past the axis
    slope = k_y / L_s
    reach = min(0.25 * a, r_s - 0.02 * a, 1.05 * span / abs(slope))
    x = np.array([-reach, 0.0, min(0.25 * a, 1.08 * a - r_s, 1.05 * span / abs(slope))])
    tangent = np.stack([r_s + x, sheared_slab_parallel_wavenumber(x, k_y, L_s, k_z)], axis=-1)
    chart.curves.update({"k_par": np.stack([r, k_par], axis=-1), "local_slab": tangent,
                         "zero": np.array([[0.0, 0.0], [1.08 * a, 0.0]])})
    chart.points["resonance"] = (r_s, 0.0)
    chart.parameters.update({"m": m, "n": n, "r_s": r_s, "q_s": q_s, "s_hat": s_hat, "R0": R0,
                             "k_y": k_y, "k_z": k_z, "L_s": L_s})
    scene = render_chart(
        chart, x_label="$r$", y_label="$k_\\parallel$",
        curve_styles={"zero": "approx", "k_par": "outer solution", "local_slab": "slope plus"},
        region_text={}, x_ticks=(r_s, a), x_tick_text=("$r_s$", "$a$"),
    )
    items: List = [Marker(tuple(chart.to_cm(chart.points["resonance"])), "o", "opoint", role="resonance")]
    if labels:
        curves = [chart.to_cm(chart.curves[k]) for k in ("k_par", "local_slab")]
        top = chart.to_cm(np.array([1.1 * a, 1.1 * span]))
        items.append(Label(tuple(top), f"$q(r_s) = m/n = {m}/{n} \\;\\Longleftrightarrow\\; k_\\parallel(r_s) = 0$",
                           "label", anchor="north east", role="resonance"))
        # outside r_s both the curve and the tangent are negative: the upper right stays empty
        under = chart.to_cm(np.array([1.1 * a, 0.8 * span]))
        items.append(Label(tuple(under), "dashed: local sheared slab, $k_z + k_y x/L_s$", "small label",
                           anchor="north east", role="local_slab"))
        items += [
            Label((CHART_WIDTH / 2, -1.4), f"$\\displaystyle {formula_equation(local_slab_from_cylinder)}$",
                  "formula box", anchor="north", role="equations"),
            Label((CHART_WIDTH / 2, -2.75), "Schematic $q(r)$; field-aligned slab at $r_s$, "
                  "$e^{i(m\\theta - n\\phi)}$", "note", anchor="north", role="note"),
        ]
    return Diagram("mode_number_mapping", scene + Scene(tuple(items)), model=chart)
