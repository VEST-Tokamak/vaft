"""Clebsch labels, field-aligned and ballooning representations (#1075).

``clebsch_field_line_label``
    a field line is the intersection of $\\psi$ = const and $\\alpha$ = const;
    lines of different $\\alpha = \\phi - q\\theta$ on one surface, and
    $\\mathbf B \\propto \\nabla\\psi\\times\\nabla\\alpha$ at a point;
``ballooning_curvature_drive``
    good and bad curvature along the extended angle, its $2\\pi$ covering
    periods, and what $\\theta_0$ does;
``ballooning_newcomb_test``
    Newcomb's test on the extended angle for stable, unstable and
    second-stable $(s, \\alpha)$;
``ballooning_harmonic_envelope``
    many coupled poloidal harmonics $m \\approx nq$ whose amplitudes are the
    Fourier transform of an envelope along the field line (a stated model);
``ballooning_workflow``
    from global harmonics to the infinite-$n$ equation on each surface, and
    how that differs from a finite-$n$ global calculation.

The physics is :mod:`vaft.formula.stability` (``field_line_label``,
``s_alpha_curvature_drive``, ``s_alpha_ballooning_solution``) on the
circular $s$-$\\alpha$ model; the torus is ``_tokamak_geometry``'s.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List

import numpy as np

from vaft.formula.stability import (
    field_line_label,
    s_alpha_ballooning_solution,
    s_alpha_curvature_drive,
    s_alpha_marginal_alpha,
)

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import band, box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene
from ._tokamak_geometry import _A, _R0, _camera_cut, _p3, _torus, _torus_normal, _torus_wireframe, _visible

_S = 1.0


@lru_cache(maxsize=8)
def _boundaries(s: float):
    """``s_alpha_marginal_alpha`` at one shear, cached: the diagrams ask for it on every build."""
    return s_alpha_marginal_alpha(s)


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


# ---------------------------------------------------------------------------
# Clebsch label
# ---------------------------------------------------------------------------


def clebsch_field_line_label(q: float = 2.5, *, labels: bool = True) -> Diagram:
    r"""A field line is where $\psi$ = const meets $\alpha$ = const.

    On one flux surface (safety factor ``q``), field lines of several
    labels $\alpha = \phi - q\theta$ (``field_line_label``), each followed
    over one toroidal transit. At a point of one line the two gradients --
    $\nabla\psi$ normal to the surface, $\nabla\alpha$ in it across the lines
    -- and $\mathbf B \propto \nabla\psi\times\nabla\alpha$ along the line. That
    order holds for ``helical_phase``'s angles ($\theta$ counter-clockwise from
    the outboard midplane, $\phi$ counter-clockwise from above, so
    $(\psi, \theta, \phi)$ is left-handed) with $\psi$ rising outward and
    $\mathbf B$ along $+\phi$, $+\theta$; in right-handed coordinates (e.g.
    $\theta$ clockwise) it is $\nabla\alpha\times\nabla\psi$, the
    Connor--Hastie--Taylor form.
    """
    try:
        q = float(q)
    except (TypeError, ValueError):
        raise ValueError(f"q must be a number, not {q!r}") from None
    if not (math.isfinite(q) and 1.0 <= q <= 5.0):
        raise ValueError(f"q must lie in [1, 5], not {q!r}")
    labels = _check_labels(labels)
    items: List = _torus_wireframe()
    phi_cut = _camera_cut()
    alphas = [phi_cut + 0.5 * math.pi + k * 0.5 for k in range(4)]
    lines = {}
    theta = np.linspace(-0.5 * math.pi / q, 1.5 * math.pi / q, 600)  # one toroidal transit, centred near the front
    for k, a in enumerate(alphas):
        phi = a + q * theta
        pts = _torus(_A, theta, phi)
        lines[k] = {"alpha": a, "points": pts, "label": field_line_label(phi, theta, q)}
        items += _split_line(pts, theta, phi, role=f"field_line:{k}")
    # at a visible point of the second line: grad psi, grad alpha and their cross product
    k0 = 1
    th0 = 0.35
    ph0 = alphas[k0] + q * th0
    P = _torus(_A, th0, ph0)
    n_hat = _torus_normal(np.array(th0), np.array(ph0))
    R = _R0 + _A * math.cos(th0)
    e_theta = np.array([-math.sin(th0) * math.cos(ph0), -math.sin(th0) * math.sin(ph0), math.cos(th0)])
    e_phi = np.array([-math.sin(ph0), math.cos(ph0), 0.0])
    grad_alpha = e_phi / R - q * e_theta / _A  # grad(phi - q theta), with theta the geometric angle here
    b_dir = np.cross(n_hat, grad_alpha)  # B ~ grad(psi) x grad(alpha): along +phi, +theta with psi rising outward
    tangent = (_torus(_A, th0 + 1e-4, ph0 + q * 1e-4) - P)
    L = 0.9
    items += [
        Arrow(tuple(_p3(P)), tuple(_p3(P + L * n_hat)), "drift", role="grad_psi"),
        Arrow(tuple(_p3(P)), tuple(_p3(P + L * grad_alpha / np.linalg.norm(grad_alpha))), "drift ion",
              role="grad_alpha"),
        Arrow(tuple(_p3(P)), tuple(_p3(P + L * b_dir / np.linalg.norm(b_dir))), "field vector", role="B"),
        Marker(tuple(_p3(P)), "o", "opoint", role="B"),
    ]
    if labels:
        items += [
            Label(tuple(_p3(P + 1.05 * L * n_hat)), "$\\nabla\\psi$", "small label", anchor="south", role="grad_psi"),
            Label(tuple(_p3(P + 1.1 * L * grad_alpha / np.linalg.norm(grad_alpha))), "$\\nabla\\alpha$",
                  "small label", anchor="west", role="grad_alpha"),
            Label(tuple(_p3(P + 1.1 * L * b_dir / np.linalg.norm(b_dir))), "$\\mathbf{B}$", "label", anchor="west",
                  role="B"),
        ]
        y_box = _below(items) - 0.2
        items += [
            Label((float(_p3(np.zeros(3))[0]), y_box),
                  f"$\\displaystyle {formula_equation(field_line_label)}$, $\\quad\\mathbf{{B}} \\propto \\nabla\\psi\\times\\nabla\\alpha$",
                  "formula box", anchor="north", role="equations"),
            _note(f"Lines of four $\\alpha$ on one surface, $q = {q:g}$; each is $\\psi$ = const "
                  "$\\cap$ $\\alpha$ = const", float(_p3(np.zeros(3))[0]), y_box - 1.1),
        ]
    model = {"q": q, "lines": lines, "point": P, "grad_alpha": grad_alpha, "normal": n_hat, "B_direction": b_dir,
             "tangent": tangent}
    return Diagram("clebsch_field_line_label", Scene(tuple(items)), model=model)


def _split_line(pts, theta, phi, role: str) -> List:
    from ._projection import split

    return split(_p3(pts), _visible(_torus_normal(theta, phi), pts), "field line", "field line hidden", role)


def _below(items) -> float:
    ys = [np.asarray(it.points)[:, 1].min() for it in items if hasattr(it, "points")]
    return float(min(ys)) - 0.3


# ---------------------------------------------------------------------------
# curvature along the extended angle
# ---------------------------------------------------------------------------


def _bands(x: np.ndarray, mask: np.ndarray):
    """Contiguous ``(x0, x1)`` intervals where ``mask`` is true."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((float(x[i]), float(x[j])))
            i = j + 1
        else:
            i += 1
    return out


def ballooning_curvature_drive(*, labels: bool = True) -> Diagram:
    r"""Normal curvature and the total ballooning drive along the extended angle.

    Shaded: the normal curvature $\cos\theta > 0$ -- bad, on the outboard
    side, once per $2\pi$ period of the covering space. Solid: the drive
    $K = \cos\theta + \Lambda\sin\theta$ of ``s_alpha_curvature_drive`` at
    $s = 1$, $\alpha = 0.4$; beyond the first period its positive lobes come
    from the geodesic term $\Lambda\sin\theta$, which grows with $|\theta|$
    through the local shear, not from the outboard position. Dashed: the same
    drive with the ballooning angle $\theta_0 = \pi/3$.
    """
    labels = _check_labels(labels)
    s, a, th0 = _S, 0.4, math.pi / 3
    theta = np.linspace(-3.0 * math.pi, 3.0 * math.pi, 2401)
    K = s_alpha_curvature_drive(theta, s, a)
    K0 = s_alpha_curvature_drive(theta, s, a, theta0=th0)
    top = float(max(np.abs(K).max(), np.abs(K0).max()))
    chart = Chart(x_range=(float(theta[0]), float(theta[-1])), y_range=(-1.1 * top, 1.1 * top))
    chart.curves.update({"drive": np.stack([theta, K], -1), "drive_theta0": np.stack([theta, K0], -1),
                         "zero": np.array([[theta[0], 0.0], [theta[-1], 0.0]])})
    bad = _bands(theta, np.cos(theta) > 0)
    chart.parameters.update({"s": s, "alpha": a, "theta0": th0, "bad_bands": bad,
                             "drive_bands": _bands(theta, K > 0)})
    ticks = tuple(k * math.pi for k in range(-2, 3, 2))
    scene = render_chart(chart, x_label="extended angle $\\theta$", y_label="$K(\\theta)$",
                         curve_styles={"zero": "approx", "drive_theta0": "approx", "drive": "boundary"},
                         region_text={}, x_ticks=ticks, x_tick_text=("$-2\\pi$", "$0$", "$2\\pi$"))
    items: List = []
    for b0, b1 in bad:
        x0, x1 = chart.to_cm(np.array([[b0, 0.0], [b1, 0.0]]))[:, 0]
        items.append(Polyline.of([(x0, 0.0), (x1, 0.0), (x1, CHART_HEIGHT), (x0, CHART_HEIGHT)], "layer",
                                 role="bad_curvature", closed=True))
    items += list(scene.items)
    if labels:
        items += [
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.1), "shaded: $\\cos\\theta > 0$,\\\\ bad normal curvature\\\\ "
                  "(outboard, each period)", "small label,align=left", anchor="north west", role="bad_curvature"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 1.8), "solid: total drive $K$;\\\\ its outer lobes are the\\\\ "
                  "geodesic term $\\Lambda\\sin\\theta$", "small label,align=left", anchor="north west", role="drive"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 3.5), "dashed: $K$ with\\\\ $\\theta_0 = \\pi/3$",
                  "small label,align=left", anchor="north west", role="theta0"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(s_alpha_curvature_drive)}$",
                  "formula box", anchor="north", role="equations"),
            _note(f"$s = {s:g}$, $\\alpha = {a:g}$; one field line followed over three poloidal turns",
                  CHART_WIDTH / 2, -2.75),
        ]
    return Diagram("ballooning_curvature_drive", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# Newcomb test
# ---------------------------------------------------------------------------


def ballooning_newcomb_test(*, labels: bool = True) -> Diagram:
    r"""Newcomb's test on the extended angle: stable, unstable and second stable.

    The even marginal ($\omega^2 = 0$) solution $F(\theta)$, $F(0) = 1$, of the
    $s$-$\alpha$ equation (``s_alpha_ballooning_solution``) at $s = 1$ for three
    pressure gradients: below the first boundary it stays positive, between
    the boundaries it crosses zero (unstable), beyond the second
    (``s_alpha_marginal_alpha``) it is positive again. These are marginal
    solutions, not localised eigenfunctions: a stable $F$ grows without
    decaying, and only the sign change carries the verdict.
    """
    labels = _check_labels(labels)
    alpha1, alpha2 = _boundaries(_S)
    cases = {"stable": 0.5 * alpha1, "unstable": 0.5 * (alpha1 + alpha2), "second_stable": 1.25 * alpha2}
    curves = {}
    for name, a in cases.items():
        theta, F = s_alpha_ballooning_solution(_S, a, theta_max=6.0 * math.pi)
        # arsinh keeps the sign -- the zero crossing is the verdict -- and compresses the growth
        curves[name] = np.stack([theta, np.arcsinh(F)], -1)
    lo = min(float(c[:, 1].min()) for c in curves.values())
    hi = max(float(c[:, 1].max()) for c in curves.values())
    chart = Chart(x_range=(0.0, 6.0 * math.pi), y_range=(min(lo, 0.0) - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)))
    chart.curves.update(curves)
    chart.curves["zero"] = np.array([[0.0, 0.0], [6.0 * math.pi, 0.0]])
    chart.parameters.update({"s": _S, "alpha": cases, "alpha1": alpha1, "alpha2": alpha2})
    scene = render_chart(chart, x_label="extended angle $\\theta$", y_label="$\\mathrm{arsinh}\\,F(\\theta)$",
                         curve_styles={"zero": "approx", "stable": "orbit ion", "second_stable": "boundary",
                                       "unstable": "orbit electron"},
                         region_text={}, x_ticks=(0.0, 2 * math.pi, 4 * math.pi, 6 * math.pi),
                         x_tick_text=("$0$", "$2\\pi$", "$4\\pi$", "$6\\pi$"), y_ticks=(0.0,))
    items: List = []
    if labels:
        items += [
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.1), f"light: $\\alpha = {cases['stable']:.2f}$, stable",
                  "small label", anchor="north west", role="stable"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.6), f"dark: $\\alpha = {cases['unstable']:.2f}$, unstable",
                  "small label", anchor="north west", role="unstable"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 1.1),
                  f"thick: $\\alpha = {cases['second_stable']:.2f}$, second stable", "small label",
                  anchor="north west", role="second_stable"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(s_alpha_ballooning_solution)}$",
                  "formula box", anchor="north", role="equations"),
            _note(f"$s = {_S:g}$: boundaries at $\\alpha_1 = {alpha1:.2f}$, $\\alpha_2 = {alpha2:.2f}$; marginal "
                  "solutions -- a zero crossing means unstable", CHART_WIDTH / 2, -2.9),
        ]
    return Diagram("ballooning_newcomb_test", scene + Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# harmonics and envelope
# ---------------------------------------------------------------------------

#: width of the model ballooning envelope along the extended angle [rad]
_ENVELOPE_WIDTH = 0.8


def ballooning_harmonic_envelope(n: int = 20, *, labels: bool = True) -> Diagram:
    r"""A ballooning mode is many coupled harmonics $m \approx nq$ under one envelope.

    The ballooning transform writes the poloidal harmonics on a surface as
    $a_m = \hat F(m - nq)$, the Fourier transform of the envelope $F$ on the
    extended angle. With a model envelope localised at the outboard midplane
    (a Gaussian of width 0.8 rad, stated, not solved), the bars are $|a_m|$
    for toroidal mode number $n$ and $q = 2.3$. Right: their sum
    $\mathrm{Re}\sum_m a_m e^{im\theta}$ -- fast oscillation at $m \approx nq$
    whose amplitude follows the envelope, the ballooning structure.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 5 <= n <= 60:
        raise ValueError(f"n must be an integer from 5 to 60, not {n!r}")
    labels = _check_labels(labels)
    q = 2.3
    eta = np.linspace(-4.0 * math.pi, 4.0 * math.pi, 8001)
    env = np.exp(-0.5 * (eta / _ENVELOPE_WIDTH) ** 2)
    m0 = n * q
    m = np.arange(int(m0) - 8, int(m0) + 10)
    amps = np.array([np.trapezoid(env * np.exp(-1j * (mm - m0) * eta), eta).real for mm in m]) / (2.0 * math.pi)
    grid = np.linspace(-math.pi, math.pi, 2401)
    field = np.real(sum(a * np.exp(1j * mm * grid) for a, mm in zip(amps, m)))
    field = field / np.max(np.abs(field))
    # left: the spectrum
    chart = Chart(x_range=(float(m[0]) - 0.5, float(m[-1]) + 0.5), y_range=(0.0, 1.15 * float(np.abs(amps).max())))
    for mm, a in zip(m, amps):
        chart.curves[f"bar_{mm}"] = np.array([[mm, 0.0], [mm, abs(a)]])
    chart.parameters.update({"n": n, "q": q, "m": m, "amplitudes": amps, "envelope_width": _ENVELOPE_WIDTH})
    scene = render_chart(chart, x_label="poloidal harmonic $m$", y_label="$|a_m|$",
                         curve_styles={f"bar_{mm}": "component imag" for mm in m}, region_text={},
                         x_ticks=(float(round(m0)),), x_tick_text=(f"$nq = {m0:g}$",))
    items: List = [Polyline(it.points, it.style, "harmonic", it.closed) if isinstance(it, Polyline)
                   and it.role.startswith("bar_") else it for it in scene.items]
    right = Chart(x_range=(-math.pi, math.pi), y_range=(-1.1, 1.1))
    right.curves["field"] = np.stack([grid, field], -1)
    right.curves["envelope"] = np.stack([grid, np.exp(-0.5 * (grid / _ENVELOPE_WIDTH) ** 2)], -1)
    sub = render_chart(right, x_label="$\\theta$", y_label="", curve_styles={"envelope": "approx", "field": "orbit ion"},
                       region_text={}, x_ticks=(-math.pi, 0.0, math.pi), x_tick_text=("$-\\pi$", "$0$", "$\\pi$"))
    items += list(sub.transformed(scale=0.6, offset=(CHART_WIDTH + 2.0, 0.2 * CHART_HEIGHT)).items)
    if labels:
        items += [
            Label((CHART_WIDTH + 2.0 + 0.3 * CHART_WIDTH, 0.2 * CHART_HEIGHT + 0.62 * CHART_HEIGHT),
                  "$\\mathrm{Re}\\sum_m a_m e^{im\\theta}$ and its envelope (dashed)", "small label", anchor="south",
                  role="field"),
            _note(f"$n = {n}$, $q = {q:g}$; $a_m = \\hat F(m - nq)$ of a model Gaussian envelope $F$ (not solved)",
                  0.5 * (CHART_WIDTH + 2.0 + 0.6 * CHART_WIDTH), -1.5),
        ]
    chart.parameters["field"] = np.stack([grid, field], -1)
    return Diagram("ballooning_harmonic_envelope", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# workflow and hierarchy
# ---------------------------------------------------------------------------


def ballooning_workflow(*, labels: bool = True) -> Diagram:
    r"""From global harmonics to the infinite-$n$ ballooning equation, and the coordinate genealogy.

    Top row: the coordinate hierarchy -- straight-field-line coordinates,
    the field-line label, field-aligned coordinates, ballooning / flux-tube
    representations. Middle: the BALOO-style path -- high-$n$ ordering, the
    ballooning transform onto the extended angle, one 1-D equation per
    surface, a Newcomb or eigenvalue test, an $(s, \alpha)$ verdict. Bottom:
    what a finite-$n$ global calculation keeps that it drops.
    """
    labels = _check_labels(labels)
    W, H = 4.0, 1.35
    items: List = []
    items += band(-1.0, 25.4, 2.0, 4.4, "coordinate genealogy", role="band:coordinates")
    items += band(-1.0, 25.4, -1.2, 1.45, "infinite-$n$ ballooning (BALOO-style)", role="band:ballooning")
    top = [("sfl", "straight-field-line\\\\ $(\\psi, \\theta, \\phi)$"), ("label", "field-line label\\\\ $\\alpha = \\phi - q\\theta$"),
           ("aligned", "field-aligned\\\\ $(\\psi, \\alpha, \\theta)$"), ("tube", "ballooning / flux tube\\\\ local, extended $\\theta$")]
    mid = [("global", "global harmonics\\\\ $m, n$ coupled"), ("transform", "$n \\to \\infty$: transform\\\\ onto extended $\\theta$, $\\theta_0$"),
           ("ode", "1-D ODE per surface\\\\ $F(\\theta)$"), ("test", "Newcomb / eigenvalue\\\\ stable or not, critical $\\alpha$")]
    nodes = {}
    for row, y in ((top, 3.1), (mid, 0.0)):
        for i, (key, text) in enumerate(row):
            b = box(1.4 + 6.4 * i, y, W + 0.8, H, text, role=f"node:{key}", latex=True)
            nodes[key] = b
            items += list(b.items)
        for (k1, _), (k2, _) in zip(row, row[1:]):
            items.append(connector(nodes[k1], nodes[k2], role=f"edge:{k1}->{k2}"))
    items.append(connector(nodes["tube"], nodes["ode"], role="edge:tube->ode"))
    glob = box(12.2, -2.9, 12.0, 1.4, "finite-$n$ global MHD (DCON, GPEC): keeps the global radial structure and the wall; "
               "ballooning is local to one surface, leading order in $1/n$", role="node:global_mhd", latex=True)
    items += list(glob.items)
    if labels:
        items.append(_note("See s\\_alpha\\_ballooning for the resulting $(s, \\alpha)$ diagram and "
                           "sfl\\_coordinate\\_taxonomy for the coordinates", 12.2, -3.9))
    model = {"nodes": tuple(nodes)}
    return Diagram("ballooning_workflow", Scene(tuple(items)), model=model)
