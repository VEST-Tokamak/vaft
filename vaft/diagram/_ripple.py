"""Toroidicity and TF ripple, one concept per diagram (#1070).

``trapped_and_passing_orbits``
    the $1/R$ mirror splits guiding-centre orbits into passing and trapped
    (banana) classes, by $\\mu$ and energy conservation;
``toroidal_field_ripple``
    $N_\\mathrm{TF}$ discrete coils corrugate $B(\\phi)$ around the smooth field;
``ripple_well_formation``
    along a field line the ripple makes local wells where $\\alpha^* < 1$;
``stochastic_ripple_orbit``
    ripple kicks at successive banana tips decorrelate above the
    Goldston--White--Boozer threshold.

$B_\\phi \\propto 1/R$ itself is ``hfs_lfs_field``. The physics is
:mod:`vaft.formula.ripple`, ``vacuum_toroidal_field`` and
``parallel_speed_from_mu``; amplitudes and widths are schematic.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.equilibrium import vacuum_toroidal_field
from vaft.formula.particle import parallel_speed_from_mu
from vaft.formula.ripple import gwb_stochastic_threshold, ripple_well_parameter, toroidal_ripple_field

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

_R0, _A, _CM = 3.0, 1.0, 2.4


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


# ---------------------------------------------------------------------------
# trapped and passing orbits
# ---------------------------------------------------------------------------

#: surface of the orbits, and a schematic banana width in units of the minor radius
_R_ORBIT, _WIDTH = 0.55, 0.6


def _guiding_centre_orbit(pitch: float) -> dict:
    """One bounce (trapped) or transit (passing) of a guiding centre starting outboard.

    $v_\\parallel(\\theta)$ follows from $\\mu$ and energy with $B(\\theta) = B_0R_0/(R_0 + r\\cos\\theta)$;
    the radial excursion is proportional to $v_\\parallel$, the conserved canonical
    toroidal momentum at large aspect ratio ($\\Delta r = q\\,\\Delta v_\\parallel/(\\epsilon\\Omega)$),
    scaled to a schematic width.
    """
    theta = np.linspace(-math.pi, math.pi, 4001)
    B = vacuum_toroidal_field(1.0, _R0, _R0 + _R_ORBIT * np.cos(theta))
    B_ref = float(vacuum_toroidal_field(1.0, _R0, _R0 + _R_ORBIT))
    v_par = parallel_speed_from_mu(1.0, pitch, B_ref, B)
    reach = np.isfinite(v_par)
    trapped = not reach.all()
    if trapped:
        th = theta[reach]
        up, down = v_par[reach], -v_par[reach]
        theta_path = np.concatenate([th, th[::-1]])
        v_path = np.concatenate([up, down[::-1]])
    else:
        theta_path, v_path = theta, v_par
    r = _R_ORBIT + _WIDTH * _A * (v_path - pitch)
    R = _R0 + r * np.cos(theta_path)
    Z = r * np.sin(theta_path)
    bounce = float(theta[reach].max()) if trapped else None
    return {"trapped": trapped, "R": R, "Z": Z, "theta": theta_path, "v_par": v_path, "bounce_theta": bounce}


def trapped_and_passing_orbits(*, labels: bool = True) -> Diagram:
    r"""The $1/R$ mirror: passing orbits go round, trapped ones bounce as bananas.

    With $\mu = mv_\perp^2/(2B)$ and $E = \tfrac12 mv_\parallel^2 + \mu B$ conserved
    (``parallel_speed_from_mu``) in $B \propto 1/R$ (``vacuum_toroidal_field``),
    a particle launched outboard with small pitch reflects before the inboard
    side; one with large pitch passes. The guiding centre's radial excursion,
    proportional to $v_\parallel$, turns the trapped path into a banana; its
    width is drawn schematically.
    """
    labels = _check_labels(labels)
    epsilon = _R_ORBIT / _R0
    boundary_pitch = math.sqrt(2 * epsilon / (1 + epsilon))  # B_max/B_min = (1 + eps)/(1 - eps)
    orbits = {"trapped": _guiding_centre_orbit(0.5 * boundary_pitch),
              "passing": _guiding_centre_orbit(1.15 * boundary_pitch)}
    theta = np.linspace(0.0, 2.0 * math.pi, 241)
    items: List = [
        Polyline.of(np.stack([_A * np.cos(theta), _A * np.sin(theta)], -1) * _CM, "lcfs", role="boundary", closed=True),
        Polyline.of(np.stack([_R_ORBIT * np.cos(theta), _R_ORBIT * np.sin(theta)], -1) * _CM, "rational",
                    role="flux_surface", closed=True),
    ]
    for name, style in (("passing", "orbit electron"), ("trapped", "orbit ion")):
        o = orbits[name]
        items.append(Polyline.of(np.stack([o["R"] - _R0, o["Z"]], -1) * _CM, style, role=f"orbit_{name}",
                                 closed=not o["trapped"]))
    # the bounce points are the banana's own tips, where v_par = 0
    t = orbits["trapped"]
    for i in (int(np.argmax(t["theta"])), int(np.argmin(t["theta"]))):
        items.append(Marker(((t["R"][i] - _R0) * _CM, t["Z"][i] * _CM), "o", "opoint", role="bounce_point"))
    if labels:
        top = _A * _CM
        items += [
            Label((-top - 0.2, 0.0), "HFS\\\\ strong $B$", "small label,align=center", anchor="east", role="hfs"),
            Label((top + 0.2, 0.0), "LFS\\\\ weak $B$", "small label,align=center", anchor="west", role="lfs"),
            Polyline.of([((t["R"][int(np.argmax(t["theta"]))] - _R0) * _CM, t["Z"][int(np.argmax(t["theta"]))] * _CM),
                         (0.75 * top, 1.05 * top)], "leader line", role="orbit_trapped"),
            Label((0.75 * top + 0.05, 1.05 * top), "trapped (banana), light", "small label", anchor="south west",
                  role="orbit_trapped"),
            Polyline.of([((orbits["passing"]["R"].min() - _R0) * _CM, -0.1),
                         (-0.75 * top, -1.05 * top)], "leader line", role="orbit_passing"),
            Label((-0.75 * top - 0.05, -1.05 * top), "passing, dark", "small label", anchor="north east",
                  role="orbit_passing"),
            Label((0.0, -top - 0.75),
                  "$\\mu = \\dfrac{mv_\\perp^2}{2B}$, $\\ E = \\tfrac12 mv_\\parallel^2 + \\mu B$ conserved; "
                  "trapped if $|v_\\parallel/v| < \\sqrt{2\\epsilon/(1 + \\epsilon)}$ outboard",
                  "formula box", anchor="north", role="equations"),
            _note("Guiding-centre paths; the banana width is exaggerated", 0.0, -top - 1.8),
        ]
    model = {"orbits": orbits, "epsilon": epsilon, "boundary_pitch": boundary_pitch}
    return Diagram("trapped_and_passing_orbits", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# TF ripple
# ---------------------------------------------------------------------------


def toroidal_field_ripple(n_tf: int = 16, *, labels: bool = True) -> Diagram:
    r"""$N_\mathrm{TF}$ discrete coils corrugate $B(\phi)$ around the smooth $1/R$ field.

    Left, the torus from above with the coils' legs; right, the outboard
    midplane field of ``toroidal_ripple_field`` over a quarter turn: smooth
    $1 - \epsilon$ and rippled, maximal under each coil and minimal between.
    The ripple amplitude is exaggerated.
    """
    if isinstance(n_tf, bool) or not isinstance(n_tf, (int, np.integer)) or not 4 <= n_tf <= 36:
        raise ValueError(f"n_tf must be an integer from 4 to 36, not {n_tf!r}")
    labels = _check_labels(labels)
    epsilon, delta = 0.3, 0.02
    # top view
    scale = 0.75
    items: List = []
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    for rr, style in ((_R0 - _A, "lcfs"), (_R0 + _A, "lcfs"), (_R0, "rational")):
        items.append(Polyline.of(np.stack([rr * np.cos(t), rr * np.sin(t)], -1) * scale, style, role="plasma",
                                 closed=True))
    coils = 2.0 * math.pi * np.arange(n_tf) / n_tf
    for c in coils:
        for r0, r1 in ((_R0 - _A - 0.55, _R0 - _A - 0.2), (_R0 + _A + 0.2, _R0 + _A + 0.9)):
            items.append(Polyline.of([(r0 * scale * math.cos(c), r0 * scale * math.sin(c)),
                                      (r1 * scale * math.cos(c), r1 * scale * math.sin(c))], "machine", role="coil"))
    # the field along phi on the outboard midplane, a quarter turn
    offset = (_R0 + _A + 1.6) * scale + 0.6
    phi = np.linspace(0.0, 0.5 * math.pi, 801)
    B = toroidal_ripple_field(1.0, epsilon, 0.0, delta, n_tf, phi, phase=math.pi)  # maxima under coils at phi_k
    chart = Chart(x_range=(0.0, 0.5 * math.pi), y_range=(0.62, 0.75))
    chart.curves["rippled"] = np.stack([phi, B], -1)
    chart.curves["smooth"] = np.array([[0.0, 1 - epsilon], [0.5 * math.pi, 1 - epsilon]])
    ticks = tuple(c for c in coils if c <= 0.5 * math.pi + 1e-9)
    scene = render_chart(chart, x_label="$\\phi$", y_label="$B/B_0$",
                         curve_styles={"smooth": "approx", "rippled": "boundary"}, region_text={},
                         x_ticks=ticks, x_tick_text=tuple("" for _ in ticks))
    items += list(scene.transformed(scale=0.6, offset=(offset, -0.3 * CHART_HEIGHT)).items)
    if labels:
        items += [
            Label((0.0, 0.0), f"$N_\\mathrm{{TF}} = {n_tf}$", "label", anchor="center", role="coil"),
            Label((offset + 0.3 * CHART_WIDTH, -0.3 * CHART_HEIGHT - 0.35), "ticks under the axis: coil positions",
                  "small label", anchor="north", role="coil"),
            Label((offset + 0.3 * CHART_WIDTH, -0.3 * CHART_HEIGHT - 1.3),
                  f"$\\displaystyle {formula_equation(toroidal_ripple_field)}$", "formula box", anchor="north",
                  role="equations"),
            _note("Outboard midplane; ripple amplitude exaggerated", offset + 0.3 * CHART_WIDTH,
                  -0.3 * CHART_HEIGHT - 2.5),
        ]
    model = {"n_tf": n_tf, "coils": coils, "phi": phi, "B": B, "epsilon": epsilon, "delta": delta}
    return Diagram("toroidal_field_ripple", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# ripple wells
# ---------------------------------------------------------------------------


def ripple_well_formation(*, labels: bool = True) -> Diagram:
    r"""Along a field line the ripple makes local wells where $\alpha^* < 1$.

    $B$ along $\phi = q\theta$ from ``toroidal_ripple_field`` rides on the
    smooth toroidal slope; where ``ripple_well_parameter`` is below 1 --
    near the midplanes, where that slope vanishes -- the corrugation has
    local minima that can trap particles.
    """
    labels = _check_labels(labels)
    eps, q, n, delta = 0.25, 2.0, 16, 0.006
    theta = np.linspace(0.0, math.pi, 4001)
    B = toroidal_ripple_field(1.0, eps, theta, delta, n, q * theta)
    # alpha* is first order; for this multiplicative field wells form where alpha* < 1 - eps cos(theta)
    alpha = ripple_well_parameter(eps, theta, q, delta, n) / (1.0 - eps * np.cos(theta))
    minima = np.flatnonzero((B[1:-1] < B[:-2]) & (B[1:-1] < B[2:])) + 1
    chart = Chart(x_range=(0.0, math.pi), y_range=(0.7, 1.3))
    chart.curves["B"] = np.stack([theta, B], -1)
    chart.curves["smooth"] = np.stack([theta, 1.0 - eps * np.cos(theta)], -1)
    chart.parameters.update({"epsilon": eps, "q": q, "n_tf": n, "delta": delta})
    # the alpha* < 1 bands, shaded under the curve
    inside = alpha < 1.0
    edges = np.flatnonzero(np.diff(inside.astype(int)))
    starts = [0] + list(edges + 1) if inside[0] else list(edges + 1)
    bands = []
    for s0 in starts:
        if not inside[s0]:
            continue
        s1 = s0
        while s1 + 1 < len(inside) and inside[s1 + 1]:
            s1 += 1
        bands.append((float(theta[s0]), float(theta[s1])))
    scene = render_chart(chart, x_label="$\\theta$ along the field line", y_label="$B/B_0$",
                         curve_styles={"smooth": "approx", "B": "boundary"}, region_text={},
                         x_ticks=(0.0, 0.5 * math.pi, math.pi), x_tick_text=("$0$", "$\\pi/2$", "$\\pi$"))
    items: List = []
    for a0, a1 in bands:
        x0, x1 = chart.to_cm(np.array([[a0, 0.7], [a1, 0.7]]))[:, 0]
        items.append(Polyline.of([(x0, 0.0), (x1, 0.0), (x1, CHART_HEIGHT), (x0, CHART_HEIGHT)], "layer",
                                 role="ripple_well_region", closed=True))
    items += list(scene.items)
    for i in minima:
        items.append(Marker(tuple(chart.to_cm(np.array([theta[i], B[i]]))), "o", "opoint", role="ripple_well"))
    if labels:
        items += [
            Label((0.25, CHART_HEIGHT - 0.1), "shaded: $\\alpha^* < 1 - \\epsilon\\cos\\theta$, local wells (dots)",
                  "small label",
                  anchor="north west", role="ripple_well_region"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(ripple_well_parameter)}$",
                  "formula box", anchor="north", role="equations"),
            _note(f"$\\epsilon = {eps:g}$, $q = {q:g}$, $N_\\mathrm{{TF}} = {n}$, $\\delta = {delta:g}$; "
                  "$\\phi = q\\theta$ along the line", CHART_WIDTH / 2, -2.75),
        ]
    model = chart
    chart.parameters["bands"] = bands
    chart.parameters["minima"] = theta[minima]
    return Diagram("ripple_well_formation", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# stochastic ripple transport
# ---------------------------------------------------------------------------


def _tip_map(K: float, steps: int, x0: float = 0.4, p0: float = 0.3) -> np.ndarray:
    """The banana-tip map in standard form: $p_{k+1} = p_k + K\\sin x_k$, $x_{k+1} = x_k + p_{k+1}$.

    $x = N_\\mathrm{TF}\\phi_\\mathrm{tip}$ is the tip's ripple phase and $p$ its
    radial position scaled by the precession shear; each kick is the ripple's
    radial step, each advance the precession between bounces.
    """
    x, p = x0, p0
    out = [p]
    for _ in range(steps):
        p = p + K * math.sin(x)
        x = x + p
        out.append(p)
    return np.array(out)


def stochastic_ripple_orbit(*, labels: bool = True) -> Diagram:
    r"""Ripple kicks at successive banana tips decorrelate above the GWB threshold.

    Each bounce the ripple shifts the banana tip radially by a step $\propto \delta$
    whose sign depends on the tip's ripple phase; the tip's toroidal
    precession then depends on its new radius. Written as a map of the tips it
    is the standard map, $p_{k+1} = p_k + K\sin x_k$, $x_{k+1} = x_k + p_{k+1}$,
    with $K$ playing $\delta/\delta_\mathrm{GWB}$ (``gwb_stochasticity_parameter``;
    the order-one factor is convention). Below $K \approx 1$ the tip oscillates
    about its start; above it random-walks out.
    """
    labels = _check_labels(labels)
    steps = 300
    regular = _tip_map(0.3, steps)
    stochastic = _tip_map(2.5, steps)
    k = np.arange(steps + 1, dtype=float)
    span = float(max(np.abs(regular).max(), np.abs(stochastic).max()))
    chart = Chart(x_range=(0.0, float(steps)), y_range=(-1.1 * span, 1.1 * span))
    chart.curves["regular"] = np.stack([k, regular], -1)
    chart.curves["stochastic"] = np.stack([k, stochastic], -1)
    chart.parameters.update({"K_regular": 0.3, "K_stochastic": 2.5})
    scene = render_chart(chart, x_label="bounce number $k$", y_label="tip radius $p_k$",
                         curve_styles={"stochastic": "orbit electron", "regular": "orbit ion"}, region_text={},
                         x_ticks=(0.0, 100.0, 200.0, 300.0))
    items: List = []
    if labels:
        items += [
            Label((CHART_WIDTH - 0.2, CHART_HEIGHT - 0.1), "dark, $K = 2.5 > 1$: stochastic, tips random-walk",
                  "small label", anchor="north east", role="stochastic"),
            Label((CHART_WIDTH - 0.2, CHART_HEIGHT - 0.6), "light, $K = 0.3 < 1$: regular, tips oscillate",
                  "small label", anchor="north east", role="regular"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(gwb_stochastic_threshold)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Banana-tip map in standard form, $K \\sim \\delta/\\delta_\\mathrm{GWB}$; schematic, not a loss rate",
                  CHART_WIDTH / 2, -2.8),
        ]
    return Diagram("stochastic_ripple_orbit", scene + Scene(tuple(items)), model=chart)
