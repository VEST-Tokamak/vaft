"""Current-profile and safety-factor topology: shapes, l_i, q landmarks, rational surfaces (#1604).

``current_profile_shapes``
    peaked, broad and hollow current at the same $I_p$:
    $j(r) \\to I(r) \\to B_\\theta(r) \\to l_i$;
``q_profile_landmarks``
    where $q_0$, $q_{\\min}$, $q_{95} = q(\\psi_N = 0.95)$ and the edge $q_a$ sit;
``q_profile_topologies``
    monotonic, weak-shear and reversed-shear $q$ from those three currents,
    with the sign of the magnetic shear;
``rational_surface_topology``
    one $m/n$ crossed once by a monotonic $q$ and twice by a reversed-shear
    $q$: a double-resonant configuration, which is not by itself a double
    tearing mode.

All four are one reduced model: a straight cylinder at large aspect ratio
whose current is one of three shapes, normalized to the same $I_p$. The
chain is ``cylindrical_enclosed_current``, ``cylindrical_poloidal_field``
(Ampère), ``cylindrical_safety_factor_from_r_B``,
``cylindrical_poloidal_flux`` (for $\\psi_N$),
``cylindrical_internal_inductance`` and ``shear_from_r_q``; the crossings
are ``vaft.process.equilibrium.find_rational_surfaces``, the routine that
finds them on a reconstructed $q$. These are concept figures, not
equilibria and not stability results: ``rational_surface`` keeps the
single-crossing definition, ``current_to_q_profile`` the analytic peaked
family, and ``li_qa`` the $l_i$-$q$ operating spaces.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from vaft.formula.equilibrium import shear_from_r_q
from vaft.formula.geometry import (
    cylindrical_enclosed_current,
    cylindrical_internal_inductance,
    cylindrical_poloidal_field,
    cylindrical_poloidal_flux,
    cylindrical_safety_factor_from_r_B,
)

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the three current shapes, unnormalized, over x = r/a
_CURRENTS = {
    "peaked": lambda x: (1.0 - x * x) ** 2,
    "broad": lambda x: 1.0 - x ** 4,
    "hollow": lambda x: (0.1 + x * x) * (1.0 - x * x) ** 1.5,
}
_CURRENT_TEXT = {"peaked": "peaked", "broad": "broad", "hollow": "hollow / off-axis"}
#: the q topology each current shape gives in this model
_TOPOLOGY = {"monotonic": "peaked", "weak_shear": "broad", "reversed_shear": "hollow"}
_TOPOLOGY_TEXT = {"monotonic": "monotonic", "weak_shear": "weak shear", "reversed_shear": "reversed shear"}
_STYLE = {"peaked": "inner solution", "broad": "boundary", "hollow": "slope minus"}
#: edge q of the canonical profiles
_Q_A = 3.5
#: |s| below which the shear is drawn as weak
_WEAK_SHEAR = 0.1
#: q_a / (m/n) that puts the level m/n inside each topology, crossed once or twice
_Q_A_PER_LEVEL = {"monotonic": 1.6, "weak_shear": 1.25, "reversed_shear": 1.15}
_PANEL = 0.55


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _check_name(name, allowed: Sequence[str], what: str) -> str:
    if not isinstance(name, str) or name not in allowed:
        raise ValueError(f"{what} must be one of {tuple(allowed)}, not {name!r}")
    return name


def _check_names(names, allowed: Sequence[str], what: str) -> Tuple[str, ...]:
    if isinstance(names, str):
        names = (names,)
    names = tuple(names)
    bad = [n for n in names if n not in allowed]
    if not names or bad or len(set(names)) != len(names):
        raise ValueError(f"{what} must be distinct names from {tuple(allowed)}, not {names!r}")
    return names


def _cylinder(shape: str, q_a: float = _Q_A, points: int = 801) -> Dict:
    """The reduced cylinder of one current shape: normalized $j$, $I$, $B_\\theta$, $q$, $\\psi_N$, $s$, $l_i$.

    Units are $a = 1$, $I_p = 1$, $\\bar j = I_p/\\pi a^2 = 1/\\pi$; ``j`` is
    returned as $j/\\bar j$ and ``B_theta`` as $B_\\theta/B_\\theta(a)$. $B_z/R_0$
    is fixed by $q(a) = q_a$, and $q(0)$ is the limit $q_a\\bar j/j(0)$.
    """
    x = np.linspace(0.0, 1.0, points)
    j = _CURRENTS[shape](x)
    current = cylindrical_enclosed_current(x, j)
    j, current = j / current[-1], current / current[-1]  # I_p = 1
    b = np.zeros_like(x)
    b[1:] = cylindrical_poloidal_field(x[1:], current[1:])
    b = b / b[-1]
    q = np.empty_like(x)
    q[1:] = cylindrical_safety_factor_from_r_B(x[1:], b[1:], q_a, 1.0)  # B_z/R0 = q_a B_theta(a)/a
    q[0] = q_a / (math.pi * j[0])
    psi = cylindrical_poloidal_flux(x, b, 1.0)
    i_min = int(np.argmin(q))
    return {
        "shape": shape, "q_a": float(q_a), "x": x, "j": math.pi * j, "I": current, "B_theta": b, "q": q,
        "psi_n": psi / psi[-1], "s": shear_from_r_q(x, q),
        "l_i": cylindrical_internal_inductance(x, b),
        "q0": float(q[0]), "q_min": float(q[i_min]), "x_min": float(x[i_min]),
    }


def _x_at_psi_n(model: Dict, psi_n: float) -> float:
    return float(np.interp(psi_n, model["psi_n"], model["x"]))


def _q_at_psi_n(model: Dict, psi_n: float) -> float:
    return float(np.interp(psi_n, model["psi_n"], model["q"]))


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _panel(chart: Chart, offset: Tuple[float, float], scale: float = _PANEL, **kwargs) -> List:
    return list(render_chart(chart, **kwargs).transformed(scale=scale, offset=offset).items)


def _to_panel(chart: Chart, xy, offset: Tuple[float, float], scale: float = _PANEL) -> Tuple[float, float]:
    p = chart.to_cm(np.asarray(xy, dtype=float)) * scale + np.asarray(offset)
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------------------
# current-profile shapes
# ---------------------------------------------------------------------------


def current_profile_shapes(profiles: Sequence[str] = ("peaked", "broad", "hollow"), *,
                           labels: bool = True) -> Diagram:
    r"""Current-profile shapes at the same $I_p$: $j(r) \to I(r) \to B_\theta(r) \to l_i$.

    Peaked $j \propto (1 - x^2)^2$, broad $1 - x^4$ and hollow (off-axis)
    $(0.1 + x^2)(1 - x^2)^{3/2}$ over $x = r/a$, each normalized to the same
    total current, so they differ only in where the current is enclosed:
    ``cylindrical_enclosed_current`` gives $I(r)$, Ampère
    (``cylindrical_poloidal_field``) $B_\theta(r)$, and
    ``cylindrical_internal_inductance`` $l_i$. The shapes are of the current
    density; $l_i$ is one number for the whole $I(r)$ and is not itself
    peaked or hollow. A more centrally enclosed current tends to a larger
    $l_i$, but different profiles can share one $l_i$. Reduced cylindrical
    model; the $l_i$-$q$ operating spaces are ``li_qa``.
    """
    profiles = _check_names(profiles, tuple(_CURRENTS), "profiles")
    labels = _check_labels(labels)
    models = {p: _cylinder(p) for p in profiles}
    x = models[profiles[0]]["x"]
    panels = (("j", "$j/\\bar j$", "current density", None),
              ("I", "$I/I_p$", "enclosed current", cylindrical_enclosed_current),
              ("B_theta", "$B_\\theta/B_\\theta(a)$", "poloidal field (Ampère)", cylindrical_poloidal_field))
    step = 7.6
    items: List = []
    charts: Dict[str, Chart] = {}
    for k, (key, y_label, title, formula) in enumerate(panels):
        top = 1.15 * max(float(np.max(m[key])) for m in models.values())
        chart = Chart(x_range=(0.0, 1.05), y_range=(0.0, top))
        for p, m in models.items():
            chart.curves[p] = np.stack([x, m[key]], -1)
        charts[key] = chart
        items += _panel(chart, (k * step, 0.0), x_label="$r/a$", y_label=y_label,
                        curve_styles={p: _STYLE[p] for p in profiles}, region_text={}, x_ticks=(0.0, 1.0),
                        y_ticks=(1.0,), y_tick_text=("$1$",))
        if labels:
            items.append(Label((k * step + _PANEL * 0.5 * CHART_WIDTH, _PANEL * CHART_HEIGHT + 0.5), title,
                               "subtitle", anchor="south", role="title"))
            if formula is not None:
                items.append(Label((k * step + _PANEL * 0.5 * CHART_WIDTH, -1.4),
                                   f"$\\displaystyle {formula_equation(formula)}$", "formula box", anchor="north",
                                   role="equations"))
        if k:
            y = _PANEL * CHART_HEIGHT * 0.55
            items.append(Arrow(((k - 1) * step + _PANEL * CHART_WIDTH + 0.5, y), (k * step - 1.45, y), "connector",
                               role="chain"))
    # the l_i column: one value per profile, keyed by its line
    x0 = 3 * step - 1.0
    y_top = _PANEL * CHART_HEIGHT - 0.2
    items.append(Arrow((2 * step + _PANEL * CHART_WIDTH + 0.55, _PANEL * CHART_HEIGHT * 0.55),
                       (x0 - 0.25, _PANEL * CHART_HEIGHT * 0.55), "connector", role="chain"))
    for i, p in enumerate(profiles):
        y = y_top - 0.75 * i
        items.append(Polyline.of([(x0, y), (x0 + 0.9, y)], _STYLE[p], role=f"legend:{p}"))
        if labels:
            items.append(Label((x0 + 1.1, y), f"{_CURRENT_TEXT[p]}: $l_i = {models[p]['l_i']:.2f}$", "small label",
                               anchor="west", role=f"l_i:{p}"))
    if labels:
        items += [
            Label((x0 + 1.6, _PANEL * CHART_HEIGHT + 0.5), "internal inductance", "subtitle", anchor="south",
                  role="title"),
            Label((x0 + 1.6, -1.4), f"$\\displaystyle {formula_equation(cylindrical_internal_inductance)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Same $I_p$: the shapes differ only in where the current is enclosed. Peaked, broad and hollow "
                  "describe $j(r)$;", 12.5, -3.0),
            _note("$l_i$ is one number for the whole profile, and different profiles can share one $l_i$.",
                  12.5, -3.6),
            _note("Reduced cylindrical model, $\\bar j = I_p/\\pi a^2$; the $l_i$--$q$ operating spaces are "
                  "\\texttt{li\\_qa}.", 12.5, -4.2),
        ]
    model = {"profiles": models, "charts": charts}
    return Diagram("current_profile_shapes", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# q landmarks
# ---------------------------------------------------------------------------


def q_profile_landmarks(profile: str = "monotonic", *, labels: bool = True) -> Diagram:
    r"""Where $q_0$, $q_\min$, $q_{95}$ and the edge $q_a$ sit on one safety-factor profile.

    $q(r/a)$ of the reduced cylinder (``profile="monotonic"``: the peaked
    current; ``"reversed_shear"``: the hollow one), with a second axis giving
    the normalized poloidal flux $\psi_N$ (``cylindrical_poloidal_flux``) at
    each radius. $q_0$ is on axis; $q_\min$ is the minimum, equal to $q_0$
    only for a monotonic $q$; $q_{95} = q(\psi_N = 0.95)$, which is not at
    $r/a = 0.95$; $q_a$ is the boundary value of a cylinder or limited
    plasma -- in a diverted equilibrium $q \to \infty$ at the separatrix,
    which is why $q_{95}$ is quoted. In this cylinder $q_a = q_\mathrm{cyl}$
    by construction; in a shaped, finite-aspect-ratio torus $q_{95}$ and the
    equilibrium edge $q$ differ from $q_\mathrm{cyl}$ and $q^*$.
    """
    profile = _check_name(profile, ("monotonic", "reversed_shear"), "profile")
    labels = _check_labels(labels)
    m = _cylinder(_TOPOLOGY[profile])
    x, q = m["x"], m["q"]
    x95, q95 = _x_at_psi_n(m, 0.95), _q_at_psi_n(m, 0.95)
    top = 1.18 * max(float(q.max()), m["q_a"])
    chart = Chart(x_range=(0.0, 1.1), y_range=(0.0, top))
    chart.curves["q"] = np.stack([x, q], -1)
    chart.curves["drop_95"] = np.array([[x95, 0.0], [x95, q95]])
    chart.points.update({"q0": (0.0, m["q0"]), "q_min": (m["x_min"], m["q_min"]), "q95": (x95, q95),
                         "q_a": (1.0, m["q_a"])})
    chart.parameters.update({"profile": profile, "x95": x95, "q95": q95, "q0": m["q0"], "q_min": m["q_min"],
                             "x_min": m["x_min"], "q_a": m["q_a"]})
    scene = render_chart(chart, x_label="$r/a$", y_label="$q$", curve_styles={"q": "boundary", "drop_95": "approx"},
                         region_text={}, x_ticks=(0.0, 0.5, 1.0), y_ticks=tuple(range(1, int(top) + 1)))
    items: List = list(scene.items)
    # the psi_N axis under the r/a ticks: where each flux fraction is in radius
    y_axis = -1.7
    items.append(Polyline.of([(0.0, y_axis), tuple(chart.to_cm(np.array([1.0, 0.0])) + [0.0, y_axis])],
                             "tick", role="psi_n_axis"))
    for value in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0):
        cx = float(chart.to_cm(np.array([_x_at_psi_n(m, value), 0.0]))[0])
        items.append(Polyline.of([(cx, y_axis), (cx, y_axis - 0.12)], "tick", role="psi_n_axis"))
        if labels and value != 1.0:
            items.append(Label((cx, y_axis - 0.18), f"${value:g}$", "ticklabel", anchor="north",
                               role="psi_n_axis"))
    for name in ("q0", "q95", "q_a") + (("q_min",) if profile == "reversed_shear" else ()):
        items.append(Marker(tuple(chart.to_cm(chart.points[name])), "o", "opoint", role=name))
    if labels:
        right = float(chart.to_cm(np.array([1.0, 0.0]))[0])
        items.append(Label((right + 0.2, y_axis), "$\\psi_N$", "ticklabel", anchor="west", role="psi_n_axis"))
        p0 = chart.to_cm(chart.points["q0"])
        q0_text = ("$q_0 = q_{\\min}$" if profile == "monotonic" else "$q_0$") + ", on axis"
        items.append(Label((p0[0] + 0.25, p0[1] + 0.3), q0_text, "label", anchor="south west", role="q0"))
        if profile == "reversed_shear":
            pm = chart.to_cm(chart.points["q_min"])
            items.append(Label((pm[0], pm[1] - 0.3), "$q_{\\min}$", "label", anchor="north", role="q_min"))
        p95 = chart.to_cm(chart.points["q95"])
        items.append(Label((p95[0] - 0.2, p95[1] + 0.3), "$q_{95} = q(\\psi_N = 0.95)$", "label",
                           anchor="south east", role="q95"))
        pa = chart.to_cm(chart.points["q_a"])
        items.append(Label((pa[0] + 0.2, pa[1] + 0.1), "$q_a$", "label", anchor="south west", role="q_a"))
        items += [
            _note(f"$\\psi_N = 0.95$ lies at $r/a = {x95:.2f}$, not $0.95$: $q_{{95}} = {q95:.2f}$, "
                  f"$q_a = {m['q_a']:g}$" + (f", $q_{{\\min}} = {m['q_min']:.2f}$" if profile == "reversed_shear"
                                             else ""), CHART_WIDTH / 2, -2.55),
            _note("$q_a$: the boundary of a cylinder or limited plasma. In a diverted equilibrium "
                  "$q \\to \\infty$ at the separatrix,", CHART_WIDTH / 2, -3.2),
            _note("so $q_{95}$ is quoted instead. Here $q_a = q_\\mathrm{cyl}$; in a shaped torus $q_{95}$ and "
                  "edge $q$ differ from $q_\\mathrm{cyl}$ and $q^*$.", CHART_WIDTH / 2, -3.75),
        ]
    return Diagram("q_profile_landmarks", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# q topologies
# ---------------------------------------------------------------------------


def _shear_runs(x: np.ndarray, s: np.ndarray) -> List[Tuple[str, float, float]]:
    """``(sign, x0, x1)`` runs of the shear: ``"+"``, ``"0"`` (|s| < the weak threshold) or ``"-"``."""
    sign = np.where(s > _WEAK_SHEAR, "+", np.where(s < -_WEAK_SHEAR, "-", "0"))
    runs, start = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or sign[i] != sign[start]:
            runs.append((str(sign[start]), float(x[start]), float(x[min(i, len(x) - 1)])))
            start = i
    return runs


_SHEAR_FILL = {"+": "region plasma", "0": "concept band", "-": "layer"}
_SHEAR_TEXT = {"+": "$s > 0$", "0": "$s \\approx 0$", "-": "$s < 0$"}


def q_profile_topologies(profiles: Sequence[str] = ("monotonic", "weak_shear", "reversed_shear"), *,
                         labels: bool = True) -> Diagram:
    r"""Monotonic, weak-shear and reversed-shear $q$, each from its current shape, with the sign of the shear.

    Columns of the reduced cylinder at one $I_p$ and $q_a$: the peaked
    current gives a monotonic $q$ with $q_\min = q_0$; the broad current a
    core where $q$ is nearly flat ($|s| < 0.1$, drawn as $s \approx 0$);
    the hollow current a $q$ whose minimum is off axis, with $s < 0$ inside
    $q_\min$. $s = (r/q)\,dq/dr$ is ``shear_from_r_q``. This is a profile
    topology, not a stability diagram; $q$ depends on the enclosed current
    and the geometry, and not every hollow current reverses the shear.
    """
    profiles = _check_names(profiles, tuple(_TOPOLOGY), "profiles")
    labels = _check_labels(labels)
    models = {p: _cylinder(_TOPOLOGY[p]) for p in profiles}
    q_top = 1.12 * max(float(m["q"].max()) for m in models.values())
    j_top = 1.15 * max(float(m["j"].max()) for m in models.values())
    step = 7.0
    j_base = _PANEL * CHART_HEIGHT + 2.3
    items: List = []
    runs_of: Dict[str, List] = {}
    for k, p in enumerate(profiles):
        m = models[p]
        x0 = k * step
        jc = Chart(x_range=(0.0, 1.05), y_range=(0.0, j_top))
        jc.curves["j"] = np.stack([m["x"], m["j"]], -1)
        items += _panel(jc, (x0, j_base), scale=0.4, x_label="", y_label="$j/\\bar j$",
                        curve_styles={"j": _STYLE[m["shape"]]}, region_text={})
        qc = Chart(x_range=(0.0, 1.05), y_range=(0.0, q_top))
        qc.curves["q"] = np.stack([m["x"], m["q"]], -1)
        runs = _shear_runs(m["x"], m["s"])
        runs_of[p] = runs
        for sign, a, b in runs:
            (xa, _), (xb, _) = _to_panel(qc, (a, 0.0), (x0, 0.0)), _to_panel(qc, (b, 0.0), (x0, 0.0))
            items.append(Polyline.of([(xa, 0.0), (xb, 0.0), (xb, _PANEL * CHART_HEIGHT), (xa, _PANEL * CHART_HEIGHT)],
                                     _SHEAR_FILL[sign], role=f"shear:{sign}", closed=True))
            if labels and xb - xa > 0.7:
                items.append(Label((0.5 * (xa + xb), _PANEL * CHART_HEIGHT - 0.05), _SHEAR_TEXT[sign],
                                   "small label", anchor="north", role=f"shear:{sign}"))
        items += _panel(qc, (x0, 0.0), x_label="$r/a$", y_label="$q$", curve_styles={"q": "boundary"},
                        region_text={}, x_ticks=(0.0, 1.0), y_ticks=tuple(range(1, int(q_top) + 1)))
        at = _to_panel(qc, (m["x_min"], m["q_min"]), (x0, 0.0))
        items.append(Marker(at, "o", "opoint", role="q_min"))
        if labels:
            text = "$q_{\\min} = q_0$" if m["x_min"] < 0.02 else "$q_{\\min}$"
            items.append(Label((at[0] + 0.15, at[1] - 0.12), text, "small label", anchor="north west", role="q_min"))
            items.append(Label((x0 + _PANEL * 0.5 * CHART_WIDTH, j_base + 0.4 * CHART_HEIGHT + 0.35),
                               f"{_CURRENT_TEXT[m['shape']]} current", "subtitle", anchor="south", role="title"))
            items.append(Arrow((x0 + _PANEL * 0.5 * CHART_WIDTH, j_base - 0.25),
                               (x0 + _PANEL * 0.5 * CHART_WIDTH, _PANEL * CHART_HEIGHT + 1.05), "connector",
                               role="chain"))
            items.append(Label((x0 + _PANEL * 0.5 * CHART_WIDTH, _PANEL * CHART_HEIGHT + 0.45),
                               f"{_TOPOLOGY_TEXT[p]} $q$", "subtitle", anchor="south", role="title"))
    if labels:
        mid = 0.5 * ((len(profiles) - 1) * step + _PANEL * CHART_WIDTH)
        items += [
            Label((mid, -1.4), f"$\\displaystyle {formula_equation(shear_from_r_q)}$", "formula box",
                  anchor="north", role="equations"),
            _note(f"Reduced cylindrical model at one $I_p$ and $q_a = {_Q_A:g}$; $|s| < {_WEAK_SHEAR:g}$ is drawn as "
                  "$s \\approx 0$. A profile topology, not a stability result:", mid, -2.75),
            _note("$q$ follows the enclosed current and the geometry, and not every hollow current reverses the "
                  "shear.", mid, -3.35),
        ]
    model = {"profiles": models, "shear_runs": runs_of}
    return Diagram("q_profile_topologies", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# rational-surface topology
# ---------------------------------------------------------------------------


def rational_surface_topology(profile: str = "reversed_shear", m: int = 2, n: int = 1, *,
                              labels: bool = True) -> Diagram:
    r"""How one $m/n$ meets each $q$ topology: once on a monotonic $q$, twice on a reversed-shear $q$.

    Left, $q(r/a)$ of the reduced cylinder scaled so the level $m/n$ lies
    inside it, with every crossing that
    ``vaft.process.equilibrium.find_rational_surfaces`` returns; right, the
    same surfaces in a circular cross-section. For ``"reversed_shear"``,
    $q(r_1) = q(r_2) = m/n$ with $r_1 < r_{\min} < r_2$: two rational
    surfaces of the same helicity, a *double-resonant configuration*. It is
    not a double tearing mode: that is the instability in which tearing
    layers on the two surfaces couple, and only a stability calculation can
    say whether it grows. ``rational_surface`` keeps the single-crossing
    definition.
    """
    profile = _check_name(profile, tuple(_TOPOLOGY), "profile")
    for name, value in (("m", m), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer mode number, not {value!r}")
    if math.gcd(int(m), int(n)) != 1:
        raise ValueError(f"m/n = {m}/{n} is not in lowest terms; use {m // math.gcd(m, n)}/{n // math.gcd(m, n)}")
    labels = _check_labels(labels)
    from vaft.process.equilibrium import find_rational_surfaces

    level = m / n
    model = _cylinder(_TOPOLOGY[profile], q_a=_Q_A_PER_LEVEL[profile] * level, points=2001)
    x, q = model["x"], model["q"]
    found = find_rational_surfaces(x, q, n, m_range=(m, m))
    crossings = [float(v) for v in found["psi_n_rational"]]  # here the coordinate is r/a
    top = 1.15 * float(q.max())
    chart = Chart(x_range=(0.0, 1.1), y_range=(0.0, top))
    chart.curves["q"] = np.stack([x, q], -1)
    chart.curves["level"] = np.array([[0.0, level], [1.06, level]])
    for i, rs in enumerate(crossings):
        chart.curves[f"drop_{i}"] = np.array([[rs, 0.0], [rs, level]])
    chart.parameters.update({"profile": profile, "m": m, "n": n, "level": level, "crossings": crossings,
                             "q_a": model["q_a"], "q0": model["q0"], "q_min": model["q_min"],
                             "x_min": model["x_min"]})
    double = len(crossings) == 2
    names = ["$r_1$", "$r_2$"] if double else ["$r_s$"] * len(crossings)
    styles = {"q": "boundary", "level": "approx", **{f"drop_{i}": "rational" for i in range(len(crossings))}}
    scene = render_chart(chart, x_label="$r/a$", y_label="$q$", curve_styles=styles, region_text={},
                         x_ticks=tuple(crossings) + (1.0,), x_tick_text=tuple(names) + ("$a$",),
                         y_ticks=(level,), y_tick_text=(f"${m}/{n}$",))
    items: List = list(scene.items)
    for i, rs in enumerate(crossings):
        items.append(Marker(tuple(chart.to_cm(np.array([rs, level]))), "o", "opoint", role=f"surface:{i}"))
    if profile == "reversed_shear":
        items.append(Marker(tuple(chart.to_cm(np.array([model["x_min"], model["q_min"]]))), "x", "xpoint",
                            role="q_min"))
    # the cross-section: boundary, the q_min surface and each rational surface
    cx, cy, radius = CHART_WIDTH + 3.6, 0.5 * CHART_HEIGHT, 2.6
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    ring = lambda r: np.stack([cx + radius * r * np.cos(t), cy + radius * r * np.sin(t)], -1)  # noqa: E731
    items.append(Polyline.of(ring(1.0), "lcfs", role="boundary", closed=True))
    if profile == "reversed_shear":
        items.append(Polyline.of(ring(model["x_min"]), "approx", role="q_min", closed=True))
    for i, rs in enumerate(crossings):
        items.append(Polyline.of(ring(rs), "separatrix", role=f"surface:{i}", closed=True))
    items.append(Marker((cx, cy), "x", "xpoint", role="axis"))
    if labels:
        at = chart.to_cm(np.array([0.02, level]))
        items.append(Label((at[0], at[1] + 0.1), f"$q = {m}/{n}$", "small label", anchor="south west",
                           role="level"))
        if profile == "reversed_shear":
            pm = chart.to_cm(np.array([model["x_min"], model["q_min"]]))
            items.append(Label((pm[0], pm[1] - 0.25), "$q_{\\min}$", "small label", anchor="north", role="q_min"))
        side = ("inner", "outer") if double else ("",) * len(crossings)
        for i, rs in enumerate(crossings):
            angle = math.radians(35.0 - 70.0 * i) if double else math.radians(35.0)
            px, py = cx + radius * rs * math.cos(angle), cy + radius * rs * math.sin(angle)
            text = f"{side[i]} {names[i]}".strip() if double else names[i]
            items.append(Label((cx + radius * 1.08 * math.cos(angle), cy + radius * 1.08 * math.sin(angle)),
                               text, "small label", anchor="west", role=f"surface:{i}"))
            items.append(Polyline.of([(px, py), (cx + radius * 1.06 * math.cos(angle),
                                                 cy + radius * 1.06 * math.sin(angle))],
                                     "leader line", role=f"surface:{i}"))
        if profile == "reversed_shear":
            items.append(Label((cx, cy - radius * model["x_min"] + 0.08), "$q_{\\min}$", "small label",
                               anchor="south", role="q_min"))
        items.append(Label((cx, cy + radius + 0.25), "cross-section, same surfaces", "subtitle", anchor="south",
                           role="title"))
        if double:
            items.append(_note(f"$q(r_1) = q(r_2) = {m}/{n}$, $r_1 < r_{{\\min}} < r_2$: two rational surfaces with the "
                               "same $m/n$ -- a double-resonant configuration", 0.5 * (CHART_WIDTH + 6.2), -1.45))
            chain = ("reversed-shear $q$", f"same ${m}/{n}$ crossed twice", "double-resonant configuration",
                     "tearing layers that may couple", "\\mbox{double tearing} mode")
            y, width, gap = -3.5, 3.1, 0.4
            x_start = 0.5 * (CHART_WIDTH + 6.2) - 0.5 * (5 * width + 4 * gap)
            boxes = [box(x_start + width / 2 + i * (width + gap), y, width, 1.2, text, latex=True,
                         role=f"chain:{i}") for i, text in enumerate(chain)]
            for b in boxes:
                items += list(b.items)
            for a, b in zip(boxes, boxes[1:]):
                items.append(connector(a, b, role="chain"))
            items.append(_note("The $q$ profile shows the resonances only; whether the layers couple and grow needs "
                               "a tearing-stability calculation.", 0.5 * (CHART_WIDTH + 6.2), -4.35))
        else:
            items.append(_note(f"{_TOPOLOGY_TEXT[profile].capitalize()} $q$: one rational surface $q(r_s) = {m}/{n}$; "
                               f"$q_a = {model['q_a']:.2f}$. Reduced cylindrical model", 0.5 * (CHART_WIDTH + 6.2),
                               -1.45))
    return Diagram("rational_surface_topology", Scene(tuple(items)), model=chart)
