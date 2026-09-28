"""Cylindrical (screw-pinch) geometry: q profile, rational surfaces, mode shapes, matching (#1072).

``current_to_q_profile``
    $j(r) \\to B_\\theta(r) \\to q(r)$ for the peaked current $j \\propto (1 - x^2)^\\nu$;
``cylindrical_rational_surfaces``
    the rational surfaces $q(r_s) = m/n$ of one profile for a fixed $n$;
``cylindrical_mode_morphology``
    what the poloidal mode number means: $m = 0$ sausage, $m = 1$ kink,
    $m = 2, 3$ higher helical distortions;
``internal_external_kink``
    displacement confined inside $q = 1$ versus reaching the boundary;
``plasma_vacuum_wall``
    one harmonic across plasma, vacuum and an ideal wall;
``cylindrical_tearing_outer``
    the outer tearing solution of one surface on the cylinder, its $\\Delta'$,
    and the inner layer that ``slab_parity`` describes.

The profile is ``peaked_current_safety_factor`` (with ``cylindrical_poloidal_field``
and ``cylindrical_safety_factor_from_r_B``); the screw-pinch field line itself is
``field_line_geometry("cylindrical")``. Displacements and eigenfunctions are schematic.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.geometry import peaked_current_safety_factor
from vaft.formula.stability import delta_prime_from_outer_derivatives

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Label, Marker, Polyline, Scene

_NU, _QA = 1.0, 3.5


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _root(f, lo: float, hi: float) -> float:
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (f(lo) < 0) == (f(mid) < 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def current_to_q_profile(nu: float = _NU, q_a: float = _QA, *, labels: bool = True) -> Diagram:
    r"""From the current profile to the safety factor: $j(r) \to B_\theta(r) \to q(r)$.

    Three panels over $x = r/a$: the current $j \propto (1 - x^2)^\nu$, the
    enclosed-current field $B_\theta \propto [1 - (1 - x^2)^{\nu+1}]/x$ (Ampère),
    and $q(x)$ from ``peaked_current_safety_factor``, which rises from
    $q_a/(\nu + 1)$ on axis: a more peaked current lowers $q_0$ and steepens
    the shear.
    """
    nu, q_a = float(nu), float(q_a)
    if not (0.0 <= nu <= 6.0 and 1.0 <= q_a <= 8.0):
        raise ValueError(f"nu must lie in [0, 6] and q_a in [1, 8], not {nu!r} and {q_a!r}")
    labels = _check_labels(labels)
    x = np.linspace(0.0, 1.0, 201)
    j = (1.0 - x * x) ** nu
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(x > 0, (1.0 - (1.0 - x * x) ** (nu + 1.0)) / np.where(x > 0, x, 1.0), 0.0)
    q = peaked_current_safety_factor(x, q_a, nu)
    panels = [("j", "$j/j_0$", j / j.max()), ("B_theta", "$B_\\theta/B_\\theta(a)$", b / b[-1]),
              ("q", "$q$", q)]
    items: List = []
    charts = {}
    for i, (key, label, y) in enumerate(panels):
        top = 1.15 * float(np.max(y))
        chart = Chart(x_range=(0.0, 1.05), y_range=(0.0, top))
        chart.curves[key] = np.stack([x, y], -1)
        charts[key] = chart
        sub = render_chart(chart, x_label="$r/a$", y_label=label, curve_styles={key: "boundary"}, region_text={},
                           x_ticks=(0.0, 1.0), y_ticks=((1.0,) if key != "q" else (q[0], q_a)),
                           y_tick_text=(("$1$",) if key != "q" else (f"${q[0]:.2g}$", f"${q_a:g}$")))
        items += list(sub.transformed(scale=0.55, offset=(i * 7.0, 0.0)).items)
    if labels:
        for i, text in enumerate(("current", "Ampère: enclosed current", "safety factor")):
            items.append(Label((i * 7.0 + 0.55 * 0.5 * CHART_WIDTH, 0.55 * CHART_HEIGHT + 0.5), text, "subtitle",
                               anchor="south", role="title"))
        items += [
            Label((10.0, -1.4), f"$\\displaystyle {formula_equation(peaked_current_safety_factor)}$", "formula box",
                  anchor="north", role="equations"),
            _note(f"$j \\propto (1 - x^2)^{{{nu:g}}}$, $q_a = {q_a:g}$: $q_0 = q_a/(\\nu + 1) = {q_a / (nu + 1):.2g}$",
                  10.0, -2.8),
        ]
    model = {"nu": nu, "q_a": q_a, "x": x, "j": j, "B_theta": b, "q": q}
    return Diagram("current_to_q_profile", Scene(tuple(items)), model=model)


def cylindrical_rational_surfaces(n: int = 1, *, labels: bool = True) -> Diagram:
    r"""For one $n$, every $m$ with $q_0 < m/n < q_a$ has its rational surface $q(r_s) = m/n$.

    $q(r)$ of ``peaked_current_safety_factor`` ($\nu = 1$, $q_a = 3.5$) with
    the levels $m/n$ and their surfaces: the harmonic label $(m, n)$, its
    resonant radius, and several surfaces for one toroidal mode number.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 4:
        raise ValueError(f"n must be an integer from 1 to 4, not {n!r}")
    labels = _check_labels(labels)
    x = np.linspace(0.0, 1.0, 401)
    q = peaked_current_safety_factor(x, _QA, _NU)
    q0 = float(q[0])
    ms = [m for m in range(1, 40) if q0 < m / n < _QA and math.gcd(m, n) == 1]
    surfaces = {m: _root(lambda xx, m=m: float(peaked_current_safety_factor(xx, _QA, _NU)) - m / n, 0.0, 1.0)
                for m in ms}
    chart = Chart(x_range=(0.0, 1.08), y_range=(0.0, 1.12 * _QA))
    chart.curves["q"] = np.stack([x, q], -1)
    for m, rs in surfaces.items():
        chart.curves[f"level_{m}"] = np.array([[0.0, m / n], [rs, m / n]])
        chart.curves[f"drop_{m}"] = np.array([[rs, 0.0], [rs, m / n]])
    chart.parameters.update({"n": n, "surfaces": surfaces, "q0": q0})
    styles = {"q": "boundary", **{f"level_{m}": "approx" for m in ms}, **{f"drop_{m}": "rational" for m in ms}}
    scene = render_chart(chart, x_label="$r/a$", y_label="$q$", curve_styles=styles, region_text={},
                         x_ticks=(1.0,), x_tick_text=("$a$",), y_ticks=tuple(m / n for m in ms),
                         y_tick_text=tuple(f"${m}/{n}$" for m in ms))
    items: List = list(scene.items)
    for m, rs in surfaces.items():
        items.append(Marker(tuple(chart.to_cm(np.array([rs, m / n]))), "o", "opoint", role=f"surface:{m}"))
        if labels:
            items.append(Label(tuple(chart.to_cm(np.array([rs, 0.0])) + np.array([0.0, 0.25])), f"$r_{{{m},{n}}}$",
                               "small label", anchor="south west", role=f"surface:{m}"))
    if labels:
        items.append(_note(f"$n = {n}$: one rational surface per $m$ with $q_0 < m/n < q_a$; $\\nu = {_NU:g}$, "
                           f"$q_a = {_QA:g}$", CHART_WIDTH / 2, -1.45))
    return Diagram("cylindrical_rational_surfaces", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# mode shapes
# ---------------------------------------------------------------------------


def cylindrical_mode_morphology(*, labels: bool = True) -> Diagram:
    r"""The poloidal mode number as a shape: sausage, kink, and higher helical distortions.

    $m = 0$ from the side, the radius modulated along $z$; $m = 1, 2, 3$ as
    cross-sections of the boundary $r = a[1 + \epsilon\cos m\theta]$ ($m = 1$
    drawn as the rigid shift it is to first order). Dashed: the unperturbed
    column. Amplitudes are exaggerated.
    """
    labels = _check_labels(labels)
    a, eps = 1.2, 0.18
    items: List = []
    z = np.linspace(0.0, 5.0, 201)
    r0 = a * (1.0 + eps * np.cos(2.0 * math.pi * z / 2.5))
    for sign in (1, -1):
        items += [Polyline.of(np.stack([z, sign * r0], -1), "lcfs", role="mode:0"),
                  Polyline.of(np.stack([z, np.full_like(z, sign * a)], -1), "approx", role="unperturbed")]
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    shapes = {}
    for k, m in enumerate((1, 2, 3)):
        cx = 7.8 + 3.4 * k
        if m == 1:
            pts = np.stack([cx + eps * 1.6 * a + a * np.cos(t), a * np.sin(t)], -1)
        else:
            r = a * (1.0 + eps * np.cos(m * t))
            pts = np.stack([cx + r * np.cos(t), r * np.sin(t)], -1)
        shapes[m] = pts
        items += [Polyline.of(pts, "lcfs", role=f"mode:{m}", closed=True),
                  Polyline.of(np.stack([cx + a * np.cos(t), a * np.sin(t)], -1), "approx", role="unperturbed",
                              closed=True),
                  Marker((cx, 0.0), "x", "xpoint", role="axis")]
    if labels:
        for text, x in (("$m = 0$: sausage (side view)", 2.5), ("$m = 1$: kink", 7.8), ("$m = 2$", 11.2),
                        ("$m = 3$", 14.6)):
            items.append(Label((x, -a - 0.6), text, "small label", anchor="north", role="title"))
        items.append(_note("Boundary $r = a[1 + \\epsilon\\cos m\\theta]$; $m = 1$ is a rigid shift to first order; "
                           "dashed: unperturbed", 8.0, -a - 1.3))
    return Diagram("cylindrical_mode_morphology", Scene(tuple(items)), model={"shapes": shapes, "sausage": r0})


def internal_external_kink(*, labels: bool = True) -> Diagram:
    r"""Internal kink: displacement inside $q = 1$; external kink: displacement reaching the boundary.

    Left, schematic radial displacements: the $m = 1$ internal kink is a
    rigid shift of the core inside the $q = 1$ radius $r_1$ of a profile with
    $q_0 < 1$ (``peaked_current_safety_factor``, $\nu = 2$, $q_a = 2.5$); an
    $m = 2$ external kink grows as $r^{m-1}$ up to the edge, where it moves
    the boundary. Right, the matching cross-sections. The Kruskal--Shafranov
    $q_a \lesssim m/n$ is the rough heuristic, not an exact boundary.
    """
    labels = _check_labels(labels)
    nu, q_a = 2.0, 2.5
    r1 = _root(lambda xx: float(peaked_current_safety_factor(xx, q_a, nu)) - 1.0, 0.0, 1.0)
    x = np.linspace(0.0, 1.0, 401)
    internal = 0.5 * (1.0 - np.tanh((x - r1) / 0.02))
    external = x ** (2 - 1)
    chart = Chart(x_range=(0.0, 1.08), y_range=(0.0, 1.15))
    chart.curves.update({"internal": np.stack([x, internal], -1), "external": np.stack([x, external], -1),
                         "q1": np.array([[r1, 0.0], [r1, 1.1]])})
    chart.parameters.update({"r1": r1, "nu": nu, "q_a": q_a})
    scene = render_chart(chart, x_label="$r/a$", y_label="$\\xi_r/\\xi_\\mathrm{max}$",
                         curve_styles={"q1": "rational", "internal": "orbit ion", "external": "orbit electron"},
                         region_text={}, x_ticks=(r1, 1.0), x_tick_text=("$r_1$", "$a$"))
    items: List = list(scene.items)
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    R = 1.4
    for k, (name, cx) in enumerate((("internal", CHART_WIDTH + 2.5), ("external", CHART_WIDTH + 6.2))):
        cy = 0.5 * CHART_HEIGHT
        items.append(Polyline.of(np.stack([cx + R * np.cos(t), cy + R * np.sin(t)], -1), "approx",
                                 role="unperturbed", closed=True))
        if name == "internal":
            items += [Polyline.of(np.stack([cx + R * np.cos(t), cy + R * np.sin(t)], -1), "lcfs",
                                  role="boundary:internal", closed=True),
                      Polyline.of(np.stack([cx + 0.25 + R * r1 * np.cos(t), cy + R * r1 * np.sin(t)], -1),
                                  "orbit ion", role="core:internal", closed=True)]
        else:
            r = R * (1.0 + 0.15 * np.cos(2 * t))
            items.append(Polyline.of(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], -1), "lcfs",
                                     role="boundary:external", closed=True))
        if labels:
            items.append(Label((cx, cy - R - 0.35), f"{name} kink", "small label", anchor="north", role="title"))
    if labels:
        items += [
            Label((CHART_WIDTH + 0.5, CHART_HEIGHT), "light: $m = 1$ internal, inside $q = 1$", "small label",
                  anchor="south west", role="internal"),
            Label((CHART_WIDTH + 0.5, CHART_HEIGHT + 0.45), "dark: $m = 2$ external, $\\propto r^{m-1}$ to the edge",
                  "small label", anchor="south west", role="external"),
            _note("Schematic displacements; Kruskal--Shafranov, roughly $q_a < m/n$, is a heuristic, not a boundary",
                  0.5 * (CHART_WIDTH + 7.0), -1.45),
        ]
    return Diagram("internal_external_kink", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def plasma_vacuum_wall(m: int = 2, *, labels: bool = True) -> Diagram:
    r"""One harmonic across plasma, vacuum and an ideal conducting wall.

    Schematic $\psi_m(r)$: regular inside the plasma ($\propto r^m$), the vacuum
    solution $Ar^m + Br^{-m}$ outside matched to it at $r = a$, and
    $\psi_m(b) = 0$ on an ideal wall at $r = b = 1.6a$; dashed, the same with
    no wall ($\propto r^{-m}$). The problems of external kinks, resistive wall
    modes and tearing outer regions all match such pieces.
    """
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or not 1 <= m <= 5:
        raise ValueError(f"m must be an integer from 1 to 5, not {m!r}")
    labels = _check_labels(labels)
    a, b = 1.0, 1.6
    r_in = np.linspace(0.0, a, 201)
    r_out = np.linspace(a, b, 121)
    r_free = np.linspace(a, 2.2, 161)
    inside = (r_in / a) ** m
    A, B = np.linalg.solve([[a ** m, a ** -m], [b ** m, b ** -m]], [1.0, 0.0])
    wall = A * r_out ** m + B * r_out ** -m
    free = (r_free / a) ** -m
    chart = Chart(x_range=(0.0, 2.3), y_range=(0.0, 1.2))
    chart.curves.update({"plasma": np.stack([r_in, inside], -1), "vacuum_wall": np.stack([r_out, wall], -1),
                         "vacuum_free": np.stack([r_free, free], -1)})
    chart.parameters.update({"m": m, "a": a, "b": b, "A": A, "B": B})
    scene = render_chart(chart, x_label="$r$", y_label="$\\psi_m$",
                         curve_styles={"vacuum_free": "approx", "plasma": "boundary", "vacuum_wall": "boundary"},
                         region_text={}, x_ticks=(a, b), x_tick_text=("$a$", "$b$"))
    items: List = []
    x_a = float(chart.to_cm(np.array([a, 0.0]))[0])
    x_b = float(chart.to_cm(np.array([b, 0.0]))[0])
    items += [Polyline.of([(0.0, 0.0), (x_a, 0.0), (x_a, CHART_HEIGHT), (0.0, CHART_HEIGHT)], "concept band",
                          role="region:plasma", closed=True),
              Polyline.of([(x_b, 0.0), (x_b + 0.15, 0.0), (x_b + 0.15, CHART_HEIGHT), (x_b, CHART_HEIGHT)], "section fill",
                          role="region:wall", closed=True)]
    items += list(scene.items)
    if labels:
        items += [
            Label((0.5 * x_a, CHART_HEIGHT - 0.1), "plasma", "small label", anchor="north", role="region:plasma"),
            Label((0.5 * (x_a + x_b), CHART_HEIGHT - 0.1), "vacuum", "small label", anchor="north",
                  role="region:vacuum"),
            Label((x_b + 0.25, CHART_HEIGHT - 0.1), "ideal wall", "small label", anchor="north west",
                  role="region:wall"),
            _note(f"Schematic $m = {m}$; dashed: no wall, $\\psi_m \\propto r^{{-m}}$", CHART_WIDTH / 2, -1.45),
        ]
    return Diagram("plasma_vacuum_wall", Scene(tuple(items)), model=chart)


def cylindrical_tearing_outer(m: int = 2, n: int = 1, *, labels: bool = True) -> Diagram:
    r"""The outer tearing solution of one rational surface on the cylinder, and where the inner layer takes over.

    $r_s$ from $q(r_s) = m/n$ (``peaked_current_safety_factor``, $\nu = 1$,
    $q_a = 3.5$); schematic outer solutions $\propto r^m$ near the axis and
    vanishing at an ideal wall $b = 1.3a$, meeting at $\psi_m(r_s)$ with a
    slope jump, whose $\Delta'$ is ``delta_prime_from_outer_derivatives`` of
    the drawn slopes. The shaded layer at $r_s$ is the non-ideal region of
    ``slab_parity``; $\Delta'$ belongs to the outer, ideal cylindrical problem.
    """
    for name, v in (("m", m), ("n", n)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0:
            raise ValueError(f"{name} must be a positive integer mode number, not {v!r}")
    q0 = _QA / (_NU + 1.0)
    if not q0 < m / n < _QA:
        raise ValueError(f"m/n = {m}/{n} has no rational surface in this profile (q from {q0:g} to {_QA:g})")
    labels = _check_labels(labels)
    rs = _root(lambda xx: float(peaked_current_safety_factor(xx, _QA, _NU)) - m / n, 0.0, 1.0)
    b = 1.3
    # left: psi = c1 r^m + c2 r^(m+1), psi(rs) = 1, slope s_minus; right: quadratic in (b - r), psi(b) = 0
    s_minus, s_plus = -0.6 / rs, 1.2 / (b - rs)
    c = np.linalg.solve([[rs ** m, rs ** (m + 1)], [m * rs ** (m - 1), (m + 1) * rs ** m]], [1.0, s_minus])
    u = b - rs
    d = np.linalg.solve([[u, u * u], [1.0, 2.0 * u]], [1.0, -s_plus])
    r_left = np.linspace(0.0, rs, 201)
    r_right = np.linspace(rs, b, 201)
    left = c[0] * r_left ** m + c[1] * r_left ** (m + 1)
    right = d[0] * (b - r_right) + d[1] * (b - r_right) ** 2
    value = delta_prime_from_outer_derivatives(1.0, s_minus, s_plus)
    top = 1.15 * float(max(left.max(), right.max()))
    chart = Chart(x_range=(0.0, 1.4), y_range=(0.0, top))
    chart.curves.update({"outer_left": np.stack([r_left, left], -1), "outer_right": np.stack([r_right, right], -1)})
    chart.parameters.update({"m": m, "n": n, "r_s": rs, "b": b, "dpsi_dr_minus": s_minus, "dpsi_dr_plus": s_plus,
                             "delta_prime": value})
    scene = render_chart(chart, x_label="$r/a$", y_label="$\\psi_m$",
                         curve_styles={"outer_left": "outer solution", "outer_right": "outer solution"},
                         region_text={}, x_ticks=(rs, 1.0, b), x_tick_text=("$r_s$", "$a$", "$b$"))
    xs = float(chart.to_cm(np.array([rs, 0.0]))[0])
    half = 0.25
    items: List = [Polyline.of([(xs - half, 0.0), (xs + half, 0.0), (xs + half, CHART_HEIGHT), (xs - half, CHART_HEIGHT)],
                               "layer", role="inner_layer", closed=True)]
    items += list(scene.items)
    x_b = float(chart.to_cm(np.array([b, 0.0]))[0])
    items.append(Polyline.of([(x_b, 0.0), (x_b + 0.15, 0.0), (x_b + 0.15, CHART_HEIGHT), (x_b, CHART_HEIGHT)],
                             "section fill", role="wall", closed=True))
    if labels:
        items += [
            Label((xs, CHART_HEIGHT + 0.1), "inner layer $\\to$ slab\\_parity", "small label", anchor="south",
                  role="inner_layer"),
            Label((0.3 * xs, CHART_HEIGHT - 0.1), f"outer, ideal\\\\ $\\Delta' {'>' if value > 0 else '<'} 0$",
                  "small label,align=center", anchor="north", role="delta_prime"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(delta_prime_from_outer_derivatives)}$",
                  "formula box", anchor="north", role="equations"),
            _note(f"$m/n = {m}/{n}$ at $r_s = {rs:.2f}a$; schematic outer solutions, ideal wall at $b = {b:g}a$",
                  CHART_WIDTH / 2, -2.9),
        ]
    return Diagram("cylindrical_tearing_outer", Scene(tuple(items)), model=chart)
