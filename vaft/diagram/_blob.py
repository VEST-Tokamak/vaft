"""SOL blobs and filaments: why a blob moves, how fast, and in which regime (#1211).

``blob_polarization``
    the mechanism in the blob's cross-section: a density monopole, the
    $\\nabla B$/curvature drifts separating ions and electrons into a dipole,
    the dipole's $E$ field and the outward $E\\times B$ motion, and the two
    closures of the polarization current;
``blob_velocity_scaling``
    $\\hat v$ against $\\hat\\delta$: the inertial and sheath-connected limits and
    the interpolation between them, from ``vaft.formula.blob``;
``blob_regimes``
    the two-region regime plane, collisionality $\\Lambda$ against
    $\\Theta = \\hat\\delta^{5/2}$, whose boundaries are where the regime scalings
    of ``blob_regime_velocities`` meet.

Conventions of D'Ippolito, Myra and Zweben (2011) throughout: $x$ radial
(outward, larger $R$), $y$ binormal, $\\mathbf B$ along $z$; a Gaussian blob
of radius $\\delta$. A turbulence solver is not the point: these explain the
reduced physics that a camera or probe measurement is compared with.
"""

from __future__ import annotations

from typing import List

import numpy as np

from vaft.formula.blob import blob_regime_velocities, interpolated_blob_velocity

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the X-point fanning parameter drawn by default: ~0.1 at 1 cm into the SOL (Myra et al. 2006, p. 092509-2)
EPSILON_X = 0.1


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def blob_polarization(*, labels: bool = True) -> Diagram:
    r"""Why a blob moves: curvature polarizes it into a dipole, and the dipole drifts it outward.

    In the $(x, y)$ plane with $\mathbf B$ into the page and $x$ outward
    (larger $R$), $\nabla B$ points inward. The $\nabla B$ and curvature
    drifts, $\propto \mathbf B\times\nabla B/q$, carry ions up and electrons
    down: the density monopole becomes a charge dipole, + above and $-$
    below. Its $\mathbf E$ points down, and $\mathbf E\times\mathbf B$ points
    outward, down $\nabla B$ -- the blob moves radially out on the
    low-field side (a hole moves in). The polarization current closes
    along the field to the sheaths or across it by ion inertia; which one
    sets the speed (``blob_velocity_scaling``).
    """
    labels = _check_labels(labels)
    items: List = []
    c = np.array([3.0, 2.6])
    t = np.linspace(0.0, 2.0 * np.pi, 97)
    for k, r in enumerate((0.55, 1.0, 1.45), start=1):
        items.append(Polyline.of(c + r * np.stack([np.cos(t), np.sin(t)], -1), "surface", role=f"density_{k}",
                                 closed=True))
    items += [Label((c[0], c[1] + 1.05), "$+$", "label", role="charge_positive"),
              Label((c[0], c[1] - 1.05), "$-$", "label", role="charge_negative"),
              Arrow((c[0], c[1] + 0.7), (c[0], c[1] - 0.7), "vector", role="electric_field"),
              Arrow((c[0] + 1.7, c[1]), (c[0] + 3.1, c[1]), "exb", role="exb_velocity"),
              Arrow((c[0] - 2.1, c[1] + 0.4), (c[0] - 2.1, c[1] + 1.2), "drift ion", role="ion_drift"),
              Arrow((c[0] - 2.1, c[1] - 0.4), (c[0] - 2.1, c[1] - 1.2), "drift electron", role="electron_drift"),
              Arrow((c[0] + 3.1, c[1] - 1.2), (c[0] + 1.9, c[1] - 1.2), "vector", role="grad_b"),
              Marker((0.2, 5.0), "x", "xpoint", role="magnetic_field"),
              Arrow((-0.2, -0.3), (0.9, -0.3), "chart axis", role="axes"),
              Arrow((-0.2, -0.3), (-0.2, 0.8), "chart axis", role="axes")]
    if labels:
        items += [Label((0.45, 5.0), "$\\mathbf B$ into the page", "small label", anchor="west", role="magnetic_field"),
                  Label((0.95, -0.3), "$x$ (larger $R$)", "small label", anchor="west", role="axes"),
                  Label((-0.2, 0.85), "$y$", "small label", anchor="south", role="axes"),
                  Label((c[0] + 0.12, c[1] + 0.35), "$\\mathbf E$", "small label", anchor="west",
                        role="electric_field"),
                  Label((c[0] + 3.2, c[1]), "$\\mathbf v_E = \\mathbf E\\times\\mathbf B/B^2$", "small label",
                        anchor="west", role="exb_velocity"),
                  Label((c[0] - 2.25, c[1] + 0.8), "ions", "small label", anchor="east", role="ion_drift"),
                  Label((c[0] - 2.25, c[1] - 0.8), "electrons", "small label", anchor="east", role="electron_drift"),
                  Label((c[0] + 3.2, c[1] - 1.2), "$\\nabla B$", "small label", anchor="west", role="grad_b"),
                  Label((c[0], c[1] - 1.6), "density monopole, charge dipole", "small label", anchor="north",
                        role="density_1"),
                  Label((c[0] + 1.3, -0.75),
                        "polarization current closes along $\\mathbf B$ to the sheaths, or across it by ion inertia",
                        "note", anchor="north", role="note")]
    return Diagram("blob_polarization", Scene(tuple(items)),
                   model={"magnetic_field": "into the page", "x": "outward, larger R", "grad_b": "-x",
                          "ion_drift": "+y", "electron_drift": "-y", "dipole": ("+ at +y", "- at -y"),
                          "electric_field": "-y", "exb_velocity": "+x"})


def blob_velocity_scaling(*, relative_amplitude: float = 1.0, labels: bool = True) -> Diagram:
    r"""Normalized blob velocity against normalized size: two closures and the bridge between them.

    Log--log $\hat v = v/v_*$ against $\hat\delta = \delta/\delta_*$ with the
    inertial limit $(\delta n/n)^{1/2}\hat\delta^{1/2}$ (current closed across
    the field), the sheath-connected limit $(\delta n/n)/\hat\delta^2$ (closed
    at the sheaths), and ``interpolated_blob_velocity`` between them. The
    limits cross at $\hat\delta = (\delta n/n)^{1/5}$; blobs near there are the
    fastest of their family and the most coherent.
    """
    labels = _check_labels(labels)
    ld = np.linspace(-1.0, 1.0, 201)
    d = 10.0**ld
    f = float(relative_amplitude)
    v = np.asarray(interpolated_blob_velocity(d, relative_amplitude=f))
    inertial, sheath = np.sqrt(f) * np.sqrt(d), f / d**2
    chart = Chart(x_range=(-1.0, 1.0), y_range=(-2.0, 1.0))
    chart.curves.update({"interpolated": np.stack([ld, np.log10(v)], -1),
                         "inertial": np.stack([ld, np.log10(inertial)], -1),
                         "sheath": np.stack([ld, np.log10(sheath)], -1)})
    chart.labels.update({"inertial": (-0.55, -0.1), "sheath": (0.62, -0.6)})
    chart.parameters.update({"relative_amplitude": f})
    ticks = [-1.0, 0.0, 1.0]
    scene = render_chart(
        chart, x_label="$\\hat\\delta = \\delta/\\delta_*$", y_label="$\\hat v = v/v_*$",
        curve_styles={"inertial": "approx", "sheath": "approx", "interpolated": "boundary"},
        region_text={"inertial": "\\small inertial $\\propto\\hat\\delta^{1/2}$",
                     "sheath": "\\small sheath $\\propto\\hat\\delta^{-2}$"} if labels else {},
        x_ticks=ticks, x_tick_text=["$0.1$", "$1$", "$10$"], y_ticks=[-2.0, -1.0, 0.0, 1.0],
        y_tick_text=["$0.01$", "$0.1$", "$1$", "$10$"],
        note=(f"D'Ippolito, Myra and Zweben (2011) Eq. (9), $\\delta n/n = {f:g}$" if labels else ""),
    )
    return Diagram("blob_velocity_scaling", scene, model=chart)


def blob_regimes(*, epsilon_x: float = EPSILON_X, labels: bool = True) -> Diagram:
    r"""The two-region regime plane of blob motion: collisionality against size.

    $\log\Lambda$ against $\log\Theta$, $\Theta = \hat\delta^{5/2}$. The
    boundaries are where neighbouring scalings of ``blob_regime_velocities``
    are equal: resistive ballooning (RB, upper left), resistive X-point
    (RX, the middle band), connected ideal interchange ($C_i$, lower left)
    and sheath-connected ($C_s$, lower right). ``epsilon_x`` is the X-point
    fanning, ~0.1 at 1 cm into the SOL.
    """
    labels = _check_labels(labels)
    if not 0.0 < epsilon_x < 1.0:
        raise ValueError(f"epsilon_x must lie in (0, 1), not {epsilon_x!r}")
    lt = np.linspace(-2.0, 3.0, 251)
    theta = 10.0**lt
    d = theta**0.4
    v = blob_regime_velocities(d, 1.0, epsilon_x)
    # each boundary: the Lambda at which two scalings are equal (RX is linear in Lambda)
    rb_rx = np.asarray(v.resistive_ballooning) / np.asarray(v.resistive_x_point)
    rx_ci = np.asarray(v.connected_ideal_interchange) / np.asarray(v.resistive_x_point)
    t_cs = 1.0 / epsilon_x  # where C_i meets C_s
    chart = Chart(x_range=(-2.0, 3.0), y_range=(-3.0, 2.0))
    chart.curves.update({
        "rb_rx": np.stack([lt, np.log10(rb_rx)], -1),
        "rx_ci": np.stack([lt[theta <= t_cs], np.log10(rx_ci[theta <= t_cs])], -1),
        "rx_cs": np.array([[np.log10(t_cs), 0.0], [3.0, 0.0]]),
        "ci_cs": np.array([[np.log10(t_cs), -3.0], [np.log10(t_cs), 0.0]]),
    })
    chart.labels.update({"RB": (-1.2, 1.2), "RX": (1.0, 0.3), "Ci": (-0.6, -2.2), "Cs": (2.2, -1.6)})
    chart.parameters.update({"epsilon_x": epsilon_x, "theta_ci_cs": t_cs})
    scene = render_chart(
        chart, x_label="$\\Theta = \\hat\\delta^{5/2}$", y_label="$\\Lambda$",
        curve_styles={"rb_rx": "boundary", "rx_ci": "boundary", "rx_cs": "boundary", "ci_cs": "boundary"},
        region_text={"RB": "\\small resistive ballooning", "RX": "\\small resistive X-point",
                     "Ci": "\\small $C_i$", "Cs": "\\small sheath-connected $C_s$"} if labels else {},
        x_ticks=[-2.0, 0.0, 2.0], x_tick_text=["$10^{-2}$", "$1$", "$10^{2}$"],
        y_ticks=[-2.0, 0.0, 2.0], y_tick_text=["$10^{-2}$", "$1$", "$10^{2}$"],
        note=(f"After D'Ippolito, Myra and Zweben (2011) Fig. 23; $\\varepsilon_x = {epsilon_x:g}$; "
              "velocity rises toward the upper left" if labels else ""),
    )
    return Diagram("blob_regimes", scene, model=chart)
