"""Neutral beam injection: the particle lifecycle and the reduced neutral attenuation (#1136).

``nbi_particle_lifecycle``
    neutral injection -> ionisation -> fast-ion birth -> confined slowing down
    -> heating, momentum and current -> thermalisation, with the three losses
    kept apart: shine-through (still neutral), prompt orbit loss (after
    ionisation) and delayed loss (after confined-orbit evolution);
``nbi_neutral_attenuation``
    survival $S(s) = e^{-\\tau(s)}$ and birth density $b(s) = \\alpha S$ along a
    beam path, with the births and the shine-through, all evaluated by
    ``vaft.formula.nbi``.

Orbits, trapped and passing fast ions, and $P_\\phi$ are the particle-motion
diagrams' (``trapped_and_passing_orbits`` and siblings); nothing here redraws
them. Conceptual: no NUBEAM output is shown.
"""

from __future__ import annotations

from typing import List

import numpy as np

from vaft.formula.nbi import (
    beam_birth_probability_density,
    neutral_beam_optical_depth,
    neutral_survival_fraction_from_optical_depth,
    shine_through_fraction,
)

from ._chart import Chart, render_chart
from ._concept import box, connector
from ._render import Diagram
from ._scene import Label, Marker, Scene

#: the three loss channels, by the stage at which the particle is lost
LOSS_CHANNELS = {
    "shine_through": "neutral",
    "prompt_loss": "first orbit after ionisation",
    "delayed_loss": "after confined-orbit evolution",
}


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def nbi_particle_lifecycle(*, labels: bool = True) -> Diagram:
    r"""What happens to a beam particle, and the three different ways it is lost.

    The main path -- injection, ionisation, fast-ion birth, confinement,
    slowing down, heating electrons and ions and driving momentum and
    current, thermalisation -- runs down the centre; the losses branch to
    the right at the stage where they happen: shine-through while still
    neutral, prompt orbit loss on the first orbit after ionisation, delayed
    loss after the fast ion has been confined for a while. A conceptual
    lifecycle, not a NUBEAM power balance.
    """
    labels = _check_labels(labels)
    t = (lambda s: s) if labels else (lambda s: "")
    w, h, lw = 4.6, 0.95, 5.4
    nodes = {
        "injection": box(0.0, 10.0, w, h, t("neutral beam injection"), role="injection"),
        "ionisation": box(0.0, 8.4, w, h, t("ionisation and charge exchange (beam deposition)"), role="ionisation"),
        "birth": box(0.0, 6.8, w, h, t("fast-ion birth"), role="birth"),
        "confined": box(0.0, 5.2, w, h, t("confined fast ion"), role="confined"),
        "slowing": box(0.0, 3.6, w, h, t("slowing down and pitch-angle scattering"), role="slowing_down"),
        "electrons": box(-4.9, 1.8, 3.6, h, t("electron heating"), role="electron_heating"),
        "ions": box(0.0, 1.8, 3.6, h, t("ion heating"), role="ion_heating"),
        "momentum": box(4.9, 1.8, 3.6, h, t("momentum and current drive"), role="momentum_current"),
        "thermal": box(0.0, 0.2, w, h, t("thermalisation: joins the bulk plasma"), role="thermalisation"),
        "shine_through": box(6.1, 10.0, lw, h, t("shine-through: lost while still neutral, to the wall"),
                             style="concept leaf", role="shine_through"),
        "prompt_loss": box(6.1, 6.8, lw, h, t("prompt orbit loss: lost on the first orbit after ionisation"),
                           style="concept leaf", role="prompt_loss"),
        "delayed_loss": box(6.1, 5.2, lw, h, t("delayed loss: after confined-orbit evolution"),
                            style="concept leaf", role="delayed_loss"),
    }
    edges = [("injection", "ionisation"), ("ionisation", "birth"), ("birth", "confined"),
             ("confined", "slowing"), ("slowing", "electrons"), ("slowing", "ions"), ("slowing", "momentum"),
             ("ions", "thermal"), ("injection", "shine_through"), ("birth", "prompt_loss"),
             ("confined", "delayed_loss")]
    items: List = []
    for node in nodes.values():
        items += node.items
    for a, b in edges:
        items.append(connector(nodes[a], nodes[b], role=f"{a}->{b}"))
    if labels:
        items.append(Label((0.3, -0.9), "shine-through fraction: $e^{-\\tau_\\mathrm{exit}}$ "
                           "(\\texttt{shine\\_through\\_fraction}); orbits: the particle-motion diagrams",
                           "note", anchor="north", role="note"))
    return Diagram("nbi_particle_lifecycle", Scene(tuple(items)),
                   model={"nodes": tuple(nodes), "edges": tuple(edges), "loss_channels": LOSS_CHANNELS})


#: example path of ``nbi_neutral_attenuation``: length [m] and peak attenuation [1/m] (illustrative)
PATH_LENGTH = 1.0
ALPHA_PEAK = 2.5
N_BIRTH_MARKERS = 24


def example_attenuation(s: np.ndarray) -> np.ndarray:
    """A parabolic attenuation coefficient along a chord through the plasma: zero at entry and exit."""
    x = 2.0 * np.asarray(s, dtype=float) / PATH_LENGTH - 1.0
    return ALPHA_PEAK * np.clip(1.0 - x**2, 0.0, None)


def nbi_neutral_attenuation(*, labels: bool = True) -> Diagram:
    r"""Neutral survival and fast-ion birth along the beam path, and the shine-through at the exit.

    For the illustrative $\alpha(s)$ of ``example_attenuation`` (a chord
    through a parabolic plasma), the curves are ``neutral_beam_optical_depth``,
    ``neutral_survival_fraction_from_optical_depth`` and
    ``beam_birth_probability_density``; the dots are births at equal
    probability steps, so their spacing is the birth density; the exit value
    of $S$ is ``shine_through_fraction``.
    """
    labels = _check_labels(labels)
    s = np.linspace(0.0, PATH_LENGTH, 401)
    alpha = example_attenuation(s)
    tau = neutral_beam_optical_depth(s, alpha)
    S = np.asarray(neutral_survival_fraction_from_optical_depth(tau))
    b = beam_birth_probability_density(s, alpha)
    shine = float(shine_through_fraction(s, alpha))
    # births at equal steps of the absorbed fraction 1 - S(s)
    absorbed = 1.0 - S
    targets = (np.arange(N_BIRTH_MARKERS) + 0.5) / N_BIRTH_MARKERS * absorbed[-1]
    s_birth = np.interp(targets, absorbed, s)
    chart = Chart(x_range=(0.0, PATH_LENGTH), y_range=(0.0, 1.6))
    chart.curves.update({"survival": np.stack([s, S], axis=-1), "birth_density": np.stack([s, b], axis=-1),
                         "alpha": np.stack([s, alpha / ALPHA_PEAK], axis=-1)})
    chart.points["shine_through"] = (PATH_LENGTH, shine)
    chart.labels.update({"survival": (0.3, 0.48), "birth": (0.7, 1.35), "alpha": (0.86, 0.72)})
    chart.parameters.update({"shine_through_fraction": shine, "tau_exit": float(tau[-1]),
                             "alpha_peak": ALPHA_PEAK, "path_length": PATH_LENGTH})
    scene = render_chart(
        chart, x_label="path coordinate $s$ [m]", y_label="fraction",
        curve_styles={"alpha": "approx", "birth_density": "inner solution", "survival": "boundary"},
        region_text={"survival": "$S = e^{-\\tau}$", "birth": "$b = \\alpha S$ [m$^{-1}$]",
                     "alpha": "$\\alpha/\\alpha_\\mathrm{max}$"} if labels else {},
        x_ticks=[0.0, 0.25, 0.5, 0.75, 1.0], y_ticks=[0.0, 0.5, 1.0, 1.5],
        note=("$\\tau(s) = \\int_0^s \\alpha\\,dl$; dots: fast-ion births at equal probability; "
              "illustrative $\\alpha$" if labels else ""),
    )
    items = list(scene.items)
    cm = chart.to_cm
    items += [Marker(tuple(cm(np.array([x, 0.05]))), ".", "xpoint", role="birth") for x in s_birth]
    exit_point = cm(np.array(chart.points["shine_through"]))
    items.append(Marker(tuple(exit_point), "o", "opoint", role="shine_through"))
    if labels:
        items.append(Label((float(exit_point[0]) + 0.15, float(exit_point[1]) + 0.25),
                           f"$f_\\mathrm{{shine}} = {shine:.2f}$", "small label", anchor="south west",
                           role="shine_through"))
    return Diagram("nbi_neutral_attenuation", Scene(tuple(items)), model=chart)
