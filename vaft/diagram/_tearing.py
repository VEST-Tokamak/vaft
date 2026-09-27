"""Tearing physics upstream of the island: rational surface, Delta-prime, layer matching (#1039).

Three single-concept diagrams, each independent of any equilibrium, shot or
solver output:

``rational_surface``
    where a helical perturbation resonates with the field-line pitch,
    $q(r_s) = m/n$, on a schematic monotonic $q(r)$;
``delta_prime``
    what the tearing stability index measures: the jump in the logarithmic
    derivative of the outer solution across $r_s$, evaluated by
    :func:`vaft.formula.stability.delta_prime_from_outer_derivatives`;
``tearing_layer_matching``
    why the ideal outer regions and the thin non-ideal layer are solved
    separately and matched at the layer edges.

The outer solutions are schematic: quadratics in $r$ that vanish on the axis
and at the edge and meet at $\\tilde\\psi(r_s)$ with prescribed slopes. Only the
sign of $\\Delta'$ carries meaning; no outer equation is solved and no layer
width is physical. The island these lead to is ``magnetic_island``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from vaft.formula.stability import delta_prime_from_outer_derivatives

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: rational-surface radius and the flux there, in units of the minor radius and of psi_s
_R_S = 0.5
_PSI_S = 1.0
#: |d psi / dr| of each outer branch at r_s for a non-zero index [psi_s / a]
_SLOPE = 1.5
_SIGNS = {"positive": 1, "zero": 0, "negative": -1}
#: half-width of the drawn inner layer [a]; schematic, not a physical width
_LAYER_HALF_WIDTH = 0.07
#: half-length of the drawn tangent segments [a]
_TANGENT = 0.2


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _with_box(scene: Scene, text: str, y: float) -> Tuple[Scene, float]:
    """A formula box centred under the chart at ``y``; returns the scene and the box's lower edge."""
    box = Label((CHART_WIDTH / 2, y), text, "formula box", anchor="north", role="equations")
    return scene + Scene((box,)), y - 1.25


# ---------------------------------------------------------------------------
# rational surface
# ---------------------------------------------------------------------------


def _schematic_q(q_s: float) -> Tuple[float, float]:
    """``(q0, qa)`` of the parabolic $q = q_0 + (q_a - q_0) r^2$ that crosses ``q_s`` inside the plasma."""
    return min(1.0, 0.6 * q_s), max(3.5, 1.4 * q_s)


#: size of the "q(r_s) = m/n" label, generously estimated [cm]
LEVEL_LABEL_SIZE = (2.4, 0.6)


def label_box(at, anchor: str, size=LEVEL_LABEL_SIZE) -> Tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` of a label of ``size`` placed at ``at`` with a compass ``anchor``."""
    w, h = size
    x, y = at
    x0 = x - w if "east" in anchor else (x if "west" in anchor else x - w / 2)
    y0 = y - h if "north" in anchor else (y if "south" in anchor else y - h / 2)
    return x0, y0, x0 + w, y0 + h


def _level_label_place(chart: Chart):
    """Where the level label clears the q curve: above-left, below-right or above-right of the crossing."""
    q_s, r_s = chart.parameters["q_s"], chart.parameters["r_s"]
    lx, ly = (float(v) for v in chart.to_cm(np.array([r_s, q_s])))
    right = float(chart.to_cm(np.array([1.08, q_s]))[0])
    curve = chart.to_cm(chart.curves["q_profile"])
    for at, anchor in (((0.25, ly + 0.15), "south west"), ((lx + 0.35, ly - 0.15), "north west"),
                       ((right, ly + 0.15), "south east")):
        x0, y0, x1, y1 = label_box(at, anchor)
        inside = (curve[:, 0] > x0) & (curve[:, 0] < x1) & (curve[:, 1] > y0) & (curve[:, 1] < y1)
        clear_of_ticks = y0 > 0.3 or anchor.startswith("south")  # a label below the level must clear the x ticks
        if clear_of_ticks and x1 < CHART_WIDTH + 0.6 and not inside.any():
            return at, anchor
    raise RuntimeError(f"no clear place for the q(r_s) label at m/n = {q_s:g}")


def rational_surface(m: int = 2, n: int = 1, *, labels: bool = True) -> Diagram:
    r"""Where a helical perturbation resonates: the rational surface $q(r_s) = m/n$.

    A schematic monotonic $q(r) = q_0 + (q_a - q_0)(r/a)^2$ -- not an
    equilibrium profile -- crosses the level $m/n$ exactly once, at $r_s$,
    where the field-line pitch matches the helicity of the $m/n$ perturbation.
    $m$ and $n$ are positive integers in lowest terms, as in ``magnetic_island``.
    """
    for name, value in (("m", m), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer mode number, not {value!r}")
    if math.gcd(int(m), int(n)) != 1:
        raise ValueError(f"m/n = {m}/{n} is not in lowest terms; use {m // math.gcd(m, n)}/{n // math.gcd(m, n)}")
    labels = _check_labels(labels)
    q_s = m / n
    q0, qa = _schematic_q(q_s)
    r_s = math.sqrt((q_s - q0) / (qa - q0))
    r = np.linspace(0.0, 1.0, 201)
    q = q0 + (qa - q0) * r**2
    chart = Chart(x_range=(0.0, 1.12), y_range=(0.0, 1.15 * qa))
    chart.curves["q_profile"] = np.stack([r, q], axis=-1)
    chart.curves["rational_level"] = np.array([[0.0, q_s], [1.08, q_s]])
    chart.curves["rational_surface"] = np.array([[r_s, 0.0], [r_s, q_s]])
    chart.points["crossing"] = (r_s, q_s)
    chart.parameters.update({"m": m, "n": n, "q_s": q_s, "q0": q0, "qa": qa, "r_s": r_s})
    # the y tick sits at q_s / (1.15 q_a); keep the axis title clear of it
    tick_height = q_s / chart.y_range[1]
    scene = render_chart(
        chart, x_label="$r$", y_label="$q$",
        curve_styles={"q_profile": "outer solution", "rational_level": "approx", "rational_surface": "rational"},
        region_text={},
        note="Schematic monotonic $q(r)$; not an equilibrium profile" if labels else "",
        x_ticks=(r_s, 1.0), x_tick_text=("$r_s$", "$a$"), y_ticks=(q_s,), y_tick_text=(f"${m}/{n}$",),
        y_label_at=0.25 if tick_height > 0.45 else 0.62,
    )
    crossing = Marker(tuple(chart.to_cm(chart.points["crossing"])), "o", "opoint", role="rational_surface")
    items = [crossing]
    if labels:
        at, anchor = _level_label_place(chart)
        items.append(Label(at, "$q(r_s) = m/n$", "label", anchor=anchor, role="rational_level"))
        # q < q_a * 0.3 + q_0 over the left third, so the top-left corner stays empty
        items.append(Label((0.25, CHART_HEIGHT - 0.1), "field-line pitch $=$ helicity", "label",
                           anchor="north west", role="region_pitch"))
        chart.parameters["level_label_anchor"] = anchor
    return Diagram("rational_surface", scene + Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# Delta-prime
# ---------------------------------------------------------------------------


def _outer_branches(sign: str, n: int = 121) -> Dict[str, np.ndarray]:
    """The two schematic outer solutions and their slopes at $r_s$.

    Left: $\\tilde\\psi = A r + B r^2$ with $\\tilde\\psi(0) = 0$; right:
    $\\tilde\\psi = C(a - r) + D(a - r)^2$ with $\\tilde\\psi(a) = 0$. Both reach
    $\\psi_s$ at $r_s$, with slopes $\\mp k\\,\\cdot$``_SLOPE`` for the sign $k$.
    """
    if sign not in _SIGNS:
        raise ValueError(f"sign must be one of {tuple(_SIGNS)}, not {sign!r}")
    k = _SIGNS[sign]
    slope_minus, slope_plus = -k * _SLOPE, k * _SLOPE
    A, B = np.linalg.solve([[_R_S, _R_S**2], [1.0, 2.0 * _R_S]], [_PSI_S, slope_minus])
    u_s = 1.0 - _R_S
    C, D = np.linalg.solve([[u_s, u_s**2], [1.0, 2.0 * u_s]], [_PSI_S, -slope_plus])
    r_left = np.linspace(0.0, _R_S, n)
    r_right = np.linspace(_R_S, 1.0, n)
    left = np.stack([r_left, A * r_left + B * r_left**2], axis=-1)
    right = np.stack([r_right, C * (1.0 - r_right) + D * (1.0 - r_right) ** 2], axis=-1)
    if left[:, 1].min() < -1e-12 or right[:, 1].min() < -1e-12:
        raise RuntimeError("the schematic outer solution went negative; _SLOPE is too large")
    return {"outer_left": left, "outer_right": right,
            "slopes": np.array([slope_minus, slope_plus]),
            "left_coefficients": np.array([A, B]), "right_coefficients": np.array([C, D])}


def _tangent(slope: float, half: float = _TANGENT) -> np.ndarray:
    r = np.array([_R_S - half, _R_S + half])
    return np.stack([r, _PSI_S + slope * (r - _R_S)], axis=-1)


def delta_prime(sign: str = "positive", *, labels: bool = True) -> Diagram:
    r"""What $\Delta'$ measures: the slope jump of the outer solution across $r_s$.

    Two schematic outer solutions -- one vanishing on the axis, one at the
    edge -- meet at $\tilde\psi(r_s)$ with different slopes; the dashed
    tangents show them. ``sign`` (``"positive"``, ``"zero"`` or
    ``"negative"``) sets the sign of the jump, and the index is evaluated by
    ``delta_prime_from_outer_derivatives`` from the drawn slopes. Only the
    sign is meaningful: this is not a stability calculation.
    """
    if not isinstance(sign, str):
        raise ValueError(f"sign must be one of {tuple(_SIGNS)}, not {sign!r}")
    branches = _outer_branches(sign)
    labels = _check_labels(labels)
    slope_minus, slope_plus = branches["slopes"]
    value = delta_prime_from_outer_derivatives(_PSI_S, slope_minus, slope_plus)
    chart = Chart(x_range=(0.0, 1.12), y_range=(0.0, 1.6))
    chart.curves.update({"outer_left": branches["outer_left"], "outer_right": branches["outer_right"],
                         "slope_left": _tangent(slope_minus), "slope_right": _tangent(slope_plus),
                         "rational_surface": np.array([[_R_S, 0.0], [_R_S, 1.3]])})
    chart.points["psi_s"] = (_R_S, _PSI_S)
    relation = {"positive": ">", "zero": "=", "negative": "<"}[sign]
    chart.parameters.update({"sign": sign, "r_s": _R_S, "psi_s": _PSI_S, "dpsi_dr_minus": slope_minus,
                             "dpsi_dr_plus": slope_plus, "delta_prime": value})
    scene = render_chart(
        chart, x_label="$r$", y_label="$\\tilde\\psi$",
        curve_styles={"rational_surface": "rational", "slope_left": "slope minus", "slope_right": "slope plus",
                      "outer_left": "outer solution", "outer_right": "outer solution"},
        region_text={},
        x_ticks=(_R_S, 1.0), x_tick_text=("$r_s$", "$a$"),
    )
    items = [Marker(tuple(chart.to_cm(chart.points["psi_s"])), "o", "opoint", role="psi_s")]
    if labels:
        top = chart.to_cm(np.array([_R_S, 1.46]))
        items.append(Label(tuple(top), f"$\\Delta' {relation} 0$", "region", role="delta_prime"))
        for key, text, default in (("slope_left", "$\\tilde\\psi'(r_s^-)$", 0),
                                   ("slope_right", "$\\tilde\\psi'(r_s^+)$", 1)):
            ends = chart.curves[key]
            # the upper end is clear of the outer curve, which lies under both tangents' tops
            index = default if ends[0, 1] == ends[1, 1] else int(np.argmax(ends[:, 1]))
            end = chart.to_cm(ends[index])
            dx, anchor = (-0.05, "east") if index == 0 else (0.05, "west")
            items.append(Label((float(end[0]) + dx, float(end[1])), text, "label", anchor=anchor, role=key))
    scene = scene + Scene(tuple(items))
    if labels:
        scene, below = _with_box(scene, f"$\\displaystyle {formula_equation(delta_prime_from_outer_derivatives)}$",
                                 -1.3)
        scene = scene + Scene((Label((CHART_WIDTH / 2, below - 0.45),
                                     "Schematic outer solutions; only the sign of $\\Delta'$ is meaningful",
                                     "note", anchor="north", role="note"),))
    return Diagram("delta_prime", scene, model=chart)


# ---------------------------------------------------------------------------
# layer matching
# ---------------------------------------------------------------------------


def _hermite(x0, y0, d0, x1, y1, d1, n: int = 41) -> np.ndarray:
    """The cubic through ``(x0, y0)`` and ``(x1, y1)`` with slopes ``d0`` and ``d1``."""
    t = np.linspace(0.0, 1.0, n)
    h = x1 - x0
    y = ((2 * t**3 - 3 * t**2 + 1) * y0 + (t**3 - 2 * t**2 + t) * h * d0
         + (-2 * t**3 + 3 * t**2) * y1 + (t**3 - t**2) * h * d1)
    return np.stack([x0 + t * h, y], axis=-1)


def tearing_layer_matching(*, labels: bool = True) -> Diagram:
    r"""Why tearing theory splits the plasma into ideal outer regions and a thin inner layer.

    The outer solutions of ``delta_prime`` (the positive case) hold outside a
    layer of schematic half-width around $r_s$; inside, a non-ideal layer
    solution joins them, matching value and slope at both edges. The outer
    problem supplies $\Delta'$, the layer problem its own jump, and tearing
    stability follows from equating the two. No layer physics is modelled.
    """
    labels = _check_labels(labels)
    branches = _outer_branches("positive")
    lo, hi = _R_S - _LAYER_HALF_WIDTH, _R_S + _LAYER_HALF_WIDTH
    (A, B), (C, D) = branches["left_coefficients"], branches["right_coefficients"]
    left = branches["outer_left"][branches["outer_left"][:, 0] <= lo + 1e-12]
    right = branches["outer_right"][branches["outer_right"][:, 0] >= hi - 1e-12]
    left = np.vstack([left, [[lo, A * lo + B * lo**2]]]) if left[-1, 0] < lo else left
    right = np.vstack([[[hi, C * (1 - hi) + D * (1 - hi) ** 2]], right]) if right[0, 0] > hi else right
    inner = _hermite(lo, A * lo + B * lo**2, A + 2 * B * lo,
                     hi, C * (1 - hi) + D * (1 - hi) ** 2, -(C + 2 * D * (1 - hi)))
    y_top = 1.75
    chart = Chart(x_range=(0.0, 1.12), y_range=(0.0, 2.0))
    chart.curves.update({
        "outer_left": left, "outer_right": right, "inner_solution": inner,
        "inner_layer": np.array([[lo, 0.0], [hi, 0.0], [hi, y_top], [lo, y_top]]),
        "rational_surface": np.array([[_R_S, 0.0], [_R_S, y_top]]),
    })
    chart.labels.update({"outer_left": (0.2, 1.5), "outer_right": (0.83, 1.5)})
    chart.parameters.update({"r_s": _R_S, "layer_half_width": _LAYER_HALF_WIDTH, "layer_edges": (lo, hi)})
    region = {"outer_left": "outer region\\\\ ideal MHD", "outer_right": "outer region\\\\ ideal MHD"}
    scene = render_chart(
        chart, x_label="$r$", y_label="$\\tilde\\psi$",
        curve_styles={"rational_surface": "rational", "outer_left": "outer solution",
                      "outer_right": "outer solution", "inner_solution": "inner solution"},
        region_text=region if labels else {},
        x_ticks=(_R_S, 1.0), x_tick_text=("$r_s$", "$a$"),
    )
    band = Polyline.of(chart.to_cm(chart.curves["inner_layer"]), "layer", role="inner_layer", closed=True)
    items = [band]
    # the matching arrows run from each outer region to the layer edge it matches
    edge_y = 1.2
    for role, x_from, x_edge in (("matching_left", 0.3, lo), ("matching_right", 0.73, hi)):
        start = chart.to_cm(np.array([x_from, edge_y]))
        end = chart.to_cm(np.array([x_edge, edge_y]))
        items.append(Arrow(tuple(start), tuple(end), "match", role=role))
    if labels:
        top = chart.to_cm(np.array([_R_S, y_top]))
        items.append(Label((float(top[0]), float(top[1]) + 0.1),
                           "inner layer: non-ideal", "label", anchor="south", role="inner_layer"))
        items.append(Label((CHART_WIDTH / 2, -1.3),
                           "matching: $\\Delta'_{\\mathrm{outer}} = \\Delta_{\\mathrm{layer}}$",
                           "formula box", anchor="north", role="equations"))
        items.append(Label((CHART_WIDTH / 2, -2.4), "Schematic layer width; no layer physics is modelled",
                           "note", anchor="north", role="note"))
    # the band sits under the curves
    scene = Scene((band,) + scene.items + tuple(items[1:]))
    return Diagram("tearing_layer_matching", scene, model=chart)
