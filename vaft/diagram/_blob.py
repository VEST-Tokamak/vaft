"""SOL blobs and filaments: why a blob moves, how the current closes, how fast, and in which regime (#1211).

``blob_polarization``
    the mechanism in the blob's cross-section: a density monopole, the
    $\\nabla B$/curvature drifts separating ions and electrons into a dipole,
    the dipole's $E$ field and the $E\\times B$ motion -- outward for a blob,
    inward for a hole;
``blob_current_closure``
    the filament along the field between the two sheaths, and where its
    polarization current closes: through the sheaths, or across the field by
    ion inertia;
``blob_velocity_scaling``
    $\\hat v$ against $\\hat\\delta$: the inertial and sheath-connected limits,
    their crossing, and the interpolation, from ``vaft.formula.sol``;
``blob_regimes``
    the two-region regime plane, collisionality $\\Lambda$ against
    $\\Theta = \\hat\\delta^{5/2}$, whose boundaries are where the regime scalings
    of ``blob_regime_velocities`` meet.

Conventions of D'Ippolito, Myra and Zweben (2011): $x$ radial (outward,
larger $R$), $z$ along $\\mathbf B$, $\\hat y = \\hat z\\times\\hat x$; a Gaussian
blob of radius $\\delta$. With $\\mathbf B$ drawn into the page and $x$ to the
right, their $+y$ points down the page, so the page-up axis is labelled $-y$.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.sol import blob_crossover_size, blob_regime_velocities, interpolated_blob_velocity

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the X-point fanning parameter drawn by default: ~0.1 at 1 cm into the SOL (Myra et al. 2006, p. 092509-2)
EPSILON_X = 0.1


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def blob_polarization(*, perturbation: str = "blob", labels: bool = True) -> Diagram:
    r"""Why a blob moves: curvature polarizes it into a dipole, and the dipole drifts it.

    In the $(x, y)$ plane with $\mathbf B$ into the page and $x$ outward
    (larger $R$), $\nabla B$ points inward. The $\nabla B$ and curvature
    drifts, $\propto \mathbf B\times\nabla B/q$, carry ions up the page and
    electrons down. On a blob (``perturbation="blob"``, a density excess)
    the excess ions pile up above and the excess electrons below: a dipole
    whose $\mathbf E$ points down, and $\mathbf E\times\mathbf B$ points
    outward, down $\nabla B$. A hole (``"hole"``, a deficit) has the reverse
    dipole and moves inward.
    """
    labels = _check_labels(labels)
    if perturbation not in ("blob", "hole"):
        raise ValueError(f"perturbation must be 'blob' or 'hole', not {perturbation!r}")
    sign = 1.0 if perturbation == "blob" else -1.0
    items: List = []
    c = np.array([3.0, 2.6])
    t = np.linspace(0.0, 2.0 * np.pi, 97)
    for k, r in enumerate((0.45, 0.8, 1.15), start=1):
        items.append(Polyline.of(c + r * np.stack([np.cos(t), np.sin(t)], -1),
                                 "surface" if sign > 0 else "approx", role=f"density_{k}", closed=True))
    top, bottom = (c[0], c[1] + 1.45), (c[0], c[1] - 1.45)
    plus, minus = (top, bottom) if sign > 0 else (bottom, top)
    items += [Label(plus, "$+$", "label", role="charge_positive"),
              Label(minus, "$-$", "label", role="charge_negative"),
              # E from + to -
              Arrow((c[0], c[1] + 0.6 * sign), (c[0], c[1] - 0.6 * sign), "vector", role="electric_field"),
              Arrow((c[0] + sign * 1.6, c[1]), (c[0] + sign * 3.0, c[1]), "exb", role="exb_velocity"),
              # the drift arrows sit on the side away from the motion
              Arrow((c[0] - sign * 2.1, c[1] + 0.4), (c[0] - sign * 2.1, c[1] + 1.2), "drift ion", role="ion_drift"),
              Arrow((c[0] - sign * 2.1, c[1] - 0.4), (c[0] - sign * 2.1, c[1] - 1.2), "drift electron",
                    role="electron_drift"),
              Marker((0.2, 5.0), "x", "xpoint", role="magnetic_field"),
              Arrow((-0.2, -0.3), (0.9, -0.3), "chart axis", role="axes"),
              Arrow((-0.2, -0.3), (-0.2, 0.8), "chart axis", role="axes"),
              Arrow((2.6, -0.3), (1.6, -0.3), "vector", role="grad_b")]
    if labels:
        v_anchor = "west" if sign > 0 else "east"
        items += [Label((0.45, 5.0), "$\\mathbf B$ into the page", "small label", anchor="west",
                        role="magnetic_field"),
                  Label((0.95, -0.3), "$x$", "small label", anchor="west", role="axes"),
                  Label((-0.2, 0.85), "$-y$", "small label", anchor="south", role="axes"),
                  Label((2.65, -0.3), "$\\nabla B$ (to smaller $R$)", "small label", anchor="west", role="grad_b"),
                  Label((c[0] + 0.12, c[1] + 0.3), "$\\mathbf E$", "small label", anchor="west", role="electric_field"),
                  Label((c[0] + sign * 3.1, c[1]), "$\\mathbf v_E = \\mathbf E\\times\\mathbf B/B^2$", "small label",
                        anchor=v_anchor, role="exb_velocity"),
                  Label((c[0] - sign * 2.25, c[1] + 0.8), "ions", "small label", anchor="east" if sign > 0 else "west",
                        role="ion_drift"),
                  Label((c[0] - sign * 2.25, c[1] - 0.8), "electrons", "small label",
                        anchor="east" if sign > 0 else "west", role="electron_drift"),
                  Label((c[0] + 1.3, -0.75),
                        f"a {perturbation}: density {'excess' if sign > 0 else 'deficit'}, charge dipole, "
                        f"{'outward' if sign > 0 else 'inward'} $E\\times B$ motion", "note", anchor="north",
                        role="note")]
    return Diagram("blob_polarization", Scene(tuple(items)),
                   model={"perturbation": perturbation, "magnetic_field": "into the page (+z along B)",
                          "x": "outward, larger R", "y": "z x x: down the page", "grad_b": "-x",
                          "ion_drift": "-y (page up)", "positive_charge": "-y" if sign > 0 else "+y",
                          "exb_velocity": "+x" if sign > 0 else "-x"})


def blob_current_closure(*, regime: str = "sheath", labels: bool = True) -> Diagram:
    r"""Where the polarization current of a filament closes: the sheaths or the cross-field inertia.

    The filament drawn along $\mathbf B$ between two targets, its dipole as
    a charge excess along the top and a deficit along the bottom.
    ``regime="sheath"``: the current runs along the field to both sheaths
    and returns through them -- the least resistive path, the lowest
    potential and speed, $v \propto \delta^{-2}$. ``"inertial"``: the
    filament is cut off from the sheaths (by resistivity or X-point
    fanning) and the current closes across the field by the ion
    polarization current near the midplane, $v \propto \delta^{1/2}$.
    """
    labels = _check_labels(labels)
    if regime not in ("sheath", "inertial"):
        raise ValueError(f"regime must be 'sheath' or 'inertial', not {regime!r}")
    L, h = 8.0, 1.2
    items: List = [Polyline.of([(0.0, -0.3), (0.0, h + 0.3)], "machine", role="sheath_left"),
                   Polyline.of([(L, -0.3), (L, h + 0.3)], "machine", role="sheath_right"),
                   Polyline.of([(0.0, h), (L, h)], "surface", role="filament"),
                   Polyline.of([(0.0, 0.0), (L, 0.0)], "surface", role="filament"),
                   Arrow((L / 2 - 0.6, h + 0.9), (L / 2 + 0.6, h + 0.9), "vector", role="magnetic_field")]
    for x in np.linspace(1.0, L - 1.0, 4):
        items += [Label((float(x), h + 0.25), "$+$", "small label", role="charge_positive"),
                  Label((float(x), -0.25), "$-$", "small label", role="charge_negative")]
    if regime == "sheath":
        # along the top to each sheath, down through it, back along the bottom
        for side in (-1.0, 1.0):
            x0, x1 = L / 2 + side * 0.4, (L if side > 0 else 0.0) - side * 0.25
            items += [Arrow((x0, h - 0.2), (x1, h - 0.2), "drift", role="parallel_current"),
                      Arrow((x1, h - 0.3), (x1, 0.3), "drift", role="sheath_current"),
                      Arrow((x1, 0.2), (x0, 0.2), "drift", role="parallel_current")]
    else:
        # cut off from the sheaths; closure across the field at the midplane by ion polarization
        for x in (1.6, L - 1.6):
            items.append(Polyline.of([(x - 0.15, -0.1), (x + 0.15, h + 0.1)], "leader line", role="disconnection"))
        items.append(Arrow((L / 2, h - 0.2), (L / 2, 0.2), "drift", role="polarization_current"))
    if labels:
        title = {"sheath": "sheath-connected: closure through the sheaths, $v \\propto \\delta^{-2}$",
                 "inertial": "inertial: closure across the field by ion polarization, $v \\propto \\delta^{1/2}$"}
        items += [Label((L / 2 + 0.7, h + 0.9), "$\\mathbf B$", "small label", anchor="west", role="magnetic_field"),
                  Label((0.0, -0.4), "sheath", "small label", anchor="north", role="sheath_left"),
                  Label((L, -0.4), "sheath", "small label", anchor="north", role="sheath_right"),
                  Label((L / 2, -0.75), title[regime], "small label", anchor="north", role="title")]
        if regime == "inertial":
            items.append(Label((L / 2 + 0.15, h / 2), "$J_{\\perp,\\mathrm{pol}}$", "small label", anchor="west",
                               role="polarization_current"))
    return Diagram("blob_current_closure", Scene(tuple(items)), model={"regime": regime})


def blob_velocity_scaling(*, relative_amplitude: float = 1.0, labels: bool = True) -> Diagram:
    r"""Normalized blob velocity against normalized size: two closures, their crossing, and the bridge.

    Log--log $\hat v = v/v_*$ against $\hat\delta = \delta/\delta_*$ with the
    inertial limit $(\delta n/n)^{1/2}\hat\delta^{1/2}$ (current closed across
    the field), the sheath-connected limit $(\delta n/n)/\hat\delta^2$ (closed
    at the sheaths), their crossing at ``blob_crossover_size``, and
    ``interpolated_blob_velocity``, which peaks at $0.574$ of the crossing.
    """
    labels = _check_labels(labels)
    ld = np.linspace(-1.0, 1.0, 201)
    d = 10.0**ld
    f = float(relative_amplitude)
    v = np.asarray(interpolated_blob_velocity(d, relative_amplitude=f))
    inertial, sheath = np.sqrt(f) * np.sqrt(d), f / d**2
    dc = float(blob_crossover_size(relative_amplitude=f))
    chart = Chart(x_range=(-1.0, 1.0), y_range=(-2.0, 1.0))
    chart.curves.update({"interpolated": np.stack([ld, np.log10(v)], -1),
                         "inertial": np.stack([ld, np.log10(inertial)], -1),
                         "sheath": np.stack([ld, np.log10(sheath)], -1)})
    chart.points["crossover"] = (math.log10(dc), math.log10(math.sqrt(f) * math.sqrt(dc)))
    chart.labels.update({"inertial": (-0.55, 0.25), "sheath": (0.55, -0.35)})
    chart.parameters.update({"relative_amplitude": f, "crossover": dc})
    scene = render_chart(
        chart, x_label="$\\hat\\delta = \\delta/\\delta_*$", y_label="$\\hat v = v/v_*$",
        curve_styles={"inertial": "approx", "sheath": "slope plus", "interpolated": "boundary"},
        region_text={"inertial": "\\small inertial $\\propto\\hat\\delta^{1/2}$",
                     "sheath": "\\small sheath $\\propto\\hat\\delta^{-2}$"} if labels else {},
        x_ticks=[-1.0, 0.0, 1.0], x_tick_text=["$0.1$", "$1$", "$10$"], y_ticks=[-2.0, -1.0, 0.0, 1.0],
        y_tick_text=["$0.01$", "$0.1$", "$1$", "$10$"],
        note=(f"D'Ippolito, Myra and Zweben (2011) Eq. (9), $\\delta n/n = {f:g}$; dot: the limits cross" if labels
              else ""),
    )
    items = list(scene.items) + [Marker(tuple(chart.to_cm(np.array(chart.points["crossover"]))), "o", "opoint",
                                        role="crossover")]
    return Diagram("blob_velocity_scaling", Scene(tuple(items)), model=chart)


def blob_regimes(*, epsilon_x: float = EPSILON_X, labels: bool = True) -> Diagram:
    r"""The two-region regime plane of blob motion: collisionality against size.

    $\log\Lambda$ against $\log\Theta$, $\Theta = \hat\delta^{5/2}$. The
    boundaries are where neighbouring scalings of ``blob_regime_velocities``
    are equal -- $\Lambda = \Theta$, $\Lambda = \varepsilon_x\Theta$,
    $\Lambda = 1$ and $\Theta = 1/\varepsilon_x$ -- around resistive ballooning
    (RB, upper left), resistive X-point (RX, the middle band), connected
    ideal interchange ($C_i$, lower left) and sheath-connected ($C_s$, lower
    right). ``epsilon_x`` is the X-point fanning, ~0.1 at 1 cm into the SOL.
    """
    labels = _check_labels(labels)
    if not 0.0 < epsilon_x < 1.0:
        raise ValueError(f"epsilon_x must lie in (0, 1), not {epsilon_x!r}")
    le = math.log10(1.0 / epsilon_x)  # log of the Theta where C_i meets C_s
    x0, x1 = -2.0, max(3.0, le + 1.5)
    lt = np.linspace(x0, x1, 251)
    theta = 10.0**lt
    d = theta**0.4
    v = blob_regime_velocities(d, 1.0, epsilon_x)
    # each boundary: the Lambda at which two scalings are equal (RX is linear in Lambda)
    rb_rx = np.asarray(v.resistive_ballooning) / np.asarray(v.resistive_x_point)
    rx_ci = np.asarray(v.connected_ideal_interchange) / np.asarray(v.resistive_x_point)
    y0, y1 = -3.0, 2.0
    chart = Chart(x_range=(x0, x1), y_range=(y0, y1))
    chart.curves.update({
        "rb_rx": np.stack([lt, np.log10(rb_rx)], -1),
        "rx_ci": np.stack([lt[lt <= le], np.log10(rx_ci[lt <= le])], -1),
        "rx_cs": np.array([[le, 0.0], [x1, 0.0]]),
        "ci_cs": np.array([[le, y0], [le, 0.0]]),
    })
    # regions placed relative to the boundaries, so they move with epsilon_x
    chart.labels.update({"RB": (x0 + 0.3 * (le - x0), 1.35), "RX": (le + 0.9, 0.9),
                         "Ci": (x0 + 0.6 * (le - x0), -2.45), "Cs": (le + 0.5 * (x1 - le), -1.5)})
    chart.parameters.update({"epsilon_x": epsilon_x, "theta_ci_cs": 1.0 / epsilon_x})
    ticks = sorted({-2.0, 0.0, 2.0} | {round(le, 6)})
    tick_text = [("$1/\\varepsilon_x$" if abs(t - le) < 1e-9 else f"$10^{{{int(t)}}}$" if t else "$1$") for t in ticks]
    scene = render_chart(
        chart, x_label="$\\Theta = \\hat\\delta^{5/2}$", y_label="$\\Lambda$",
        curve_styles={"rb_rx": "boundary", "rx_ci": "boundary", "rx_cs": "boundary", "ci_cs": "boundary"},
        region_text={"RB": "\\begin{tabular}{c}\\small resistive ballooning\\\\\\small $\\hat v\\sim\\hat\\delta^{1/2}$"
                           "\\end{tabular}",
                     "RX": "\\begin{tabular}{c}\\small resistive X-point\\\\\\small $\\hat v\\sim\\Lambda/\\hat\\delta^2$"
                           "\\end{tabular}",
                     "Ci": "\\begin{tabular}{c}\\small $C_i$\\\\\\small $\\hat v\\sim\\varepsilon_x\\hat\\delta^{1/2}$"
                           "\\end{tabular}",
                     "Cs": "\\begin{tabular}{c}\\small sheath-connected $C_s$\\\\\\small $\\hat v\\sim 1/\\hat\\delta^2$"
                           "\\end{tabular}"} if labels else {},
        x_ticks=ticks, x_tick_text=tick_text,
        y_ticks=[-2.0, 0.0, 2.0], y_tick_text=["$10^{-2}$", "$1$", "$10^{2}$"],
        note=(f"After D'Ippolito, Myra and Zweben (2011) Fig. 23; $\\varepsilon_x = {epsilon_x:g}$; boundaries "
              "$\\Lambda = \\Theta$, $\\varepsilon_x\\Theta$, $1$ and $\\Theta = 1/\\varepsilon_x$" if labels else ""),
    )
    return Diagram("blob_regimes", scene, model=chart)
