"""Equilibrium-derived analytic and reduced MHD stability diagnostics (#1635).

``stability_diagnostic_taxonomy``
    the criteria VAFT holds, by physical problem (rows) and logical status
    (columns): exact definitions, reduced models, empirical and
    semi-empirical boundaries, heuristics, and solver-derived quantities,
    with what is still absent and how reduced theory relates to the solvers;
``interchange_criteria``
    Suydam's cylindrical and Mercier's circular-tokamak criteria across one
    schematic profile: the toroidal average curvature $p'(1 - q^2)$
    stabilises interchanges outside $q = 1$, where Suydam's still fails.

The physics is :mod:`vaft.formula.stability` and :mod:`vaft.formula.geometry`;
profiles are schematic. Evaluating the criteria on a reconstructed equilibrium
belongs to the process and plotting layers, not to these figures.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from vaft.formula.geometry import peaked_current_safety_factor
from vaft.formula.stability import mercier_criterion_circular, suydam_criterion
from vaft.formula.utils import gradient

from ._chart import CHART_WIDTH, Chart, render_chart
from ._concept import band, box
from ._equations import formula_equation
from ._geometry import _check_labels
from ._render import Diagram
from ._scene import Arrow, Label, Scene

# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------

#: logical status, left to right
STATUSES: Tuple[Tuple[str, str], ...] = (
    ("definition", "exact definition"),
    ("reduced", "reduced model"),
    ("empirical", "empirical / semi-empirical"),
    ("heuristic", "heuristic or approximate"),
    ("solver", "solver-derived (read, not computed)"),
)

#: physical problem, top to bottom
PROBLEMS: Tuple[Tuple[str, str], ...] = (
    ("current", "current driven"),
    ("pressure", "pressure / curvature"),
    ("resistive", "resistive layer"),
    ("axisymmetric", "axisymmetric"),
    ("trigger", "trigger / nonlinear"),
    ("operational", "operational space"),
)

#: (problem, status) -> what VAFT holds there
CELLS: Dict[Tuple[str, str], str] = {
    ("current", "reduced"): "Kruskal--Shafranov / $q^*$;\\\\ Bussac $\\beta_{p1}$, $\\delta\\hat W_T$;\\\\ CFB $l_i$--$q_a$",
    ("current", "empirical"): "Wesson JET $l_i$--$q_\\psi$;\\\\ ITER $q_{95}$",
    ("current", "heuristic"): "$\\beta_{N,crit} = 2.8\\,q_{95}$\\\\ (deprecated)",
    ("current", "solver"): "DCON $\\delta W$\\\\ for each $n$",
    ("pressure", "definition"): "ballooning $\\alpha$, shear $\\hat s$;\\\\ magnetic well $W$",
    ("pressure", "reduced"): "Suydam $D_S$, Mercier $D_M$;\\\\ $s$--$\\alpha$ first and second\\\\ stability boundaries",
    ("pressure", "empirical"): "Troyon $\\beta_N$\\\\ (numerical fit)",
    ("pressure", "heuristic"): "$\\alpha_{crit} \\approx 0.6\\,\\hat s$\\\\ (line fit)",
    ("pressure", "solver"): "DCON Mercier $D_I$,\\\\ ballooning $C_A$",
    ("resistive", "definition"): "$\\Delta'$ from outer slopes;\\\\ GGJ $D_I$, $D_R$ of $E, F, H$",
    ("resistive", "reduced"): "sheared-slab flux:\\\\ tearing or twisting parity",
    ("resistive", "solver"): "RDCON / STRIDE\\\\ $\\Delta'$, $D_R$, $H$",
    ("axisymmetric", "definition"): "decay index $n$;\\\\ VDE growth rate $d\\ln|\\Delta Z|/dt$",
    ("axisymmetric", "reduced"): "thin-wall and\\\\ wall-mode times",
    ("trigger", "reduced"): "Kadomtsev $r_\\mathrm{mix}$;\\\\ island width",
    ("trigger", "heuristic"): "$\\beta_{p,crit} = 0.3(1 - q_0)$\\\\ (deprecated)",
    ("operational", "empirical"): "Greenwald, Murakami,\\\\ Hugill",
    ("operational", "heuristic"): "$\\beta$ margin, power-limit\\\\ helpers",
}

#: what the inventory found absent, in the order the issue lists it
ABSENT: Tuple[str, ...] = (
    "equilibrium-native Mercier $D_I$", "curvature $\\kappa_n$, $\\kappa_g$ and local shear", "critical decay index",
    "Porcelli $\\delta W$ terms", "Modified Rutherford (\\#1031)",
)

_CELL_W, _CELL_H = 4.3, 1.9
_X0, _Y0 = 0.0, 0.0


def _cell_center(problem: int, status: int) -> Tuple[float, float]:
    return (_X0 + status * (_CELL_W + 0.3), _Y0 - problem * (_CELL_H + 0.3))


def stability_diagnostic_taxonomy(*, labels: bool = True) -> Diagram:
    r"""Reduced MHD stability diagnostics by physical problem and logical status.

    Rows are the physical problem (current driven, pressure / curvature,
    resistive layer, axisymmetric, trigger / nonlinear, operational space),
    columns the logical status: exact definition, reduced model, empirical
    or semi-empirical boundary, heuristic or approximate, and solver-derived
    (DCON / RDCON / STRIDE quantities VAFT reads but does not compute). Each
    cell names what ``vaft.formula`` holds there; the gaps are listed. The
    reduced columns are compared with the solver column -- interpretation
    and limiting-case validation, not a screening gate.
    """
    labels = _check_labels(labels)
    items: List = []
    width = len(STATUSES) * (_CELL_W + 0.3)
    left = _X0 - 0.5 * _CELL_W - 3.0
    right = _X0 - 0.5 * _CELL_W + width
    for j, (key, text) in enumerate(STATUSES):
        x, _ = _cell_center(0, j)
        items.append(Label((x, _Y0 + 0.5 * _CELL_H + 0.2), text, "concept group title,text width=4.1cm,align=center",
                           anchor="south", role=f"status:{key}"))
    for i, (key, text) in enumerate(PROBLEMS):
        _, y = _cell_center(i, 0)
        items += band(left, right, y - 0.5 * _CELL_H - 0.12, y + 0.5 * _CELL_H + 0.12, role=f"problem:{key}")
        items.append(Label((left + 0.15, y), text, "concept band label,text width=2.6cm,align=left,execute at begin node={\\hyphenpenalty=10000}", anchor="west",
                           role=f"problem:{key}"))
    cells = {}
    for i, (problem, _) in enumerate(PROBLEMS):
        for j, (status, _) in enumerate(STATUSES):
            text = CELLS.get((problem, status))
            if text is None:
                continue
            x, y = _cell_center(i, j)
            style = "concept leaf" if status in ("heuristic", "solver") else "concept box"
            cells[(problem, status)] = box(x, y, _CELL_W, _CELL_H, text, style=style,
                                           role=f"cell:{problem}:{status}", latex=True)
            items += list(cells[(problem, status)].items)
    bottom = _cell_center(len(PROBLEMS) - 1, 0)[1] - 0.5 * _CELL_H - 0.4
    # the reduced and definition columns are compared with the solver column, not gated by it
    y_cmp = bottom - 0.45
    x_from = _cell_center(0, 1)[0]
    x_to = _cell_center(0, 4)[0]
    items.append(Arrow((x_from, y_cmp), (x_to, y_cmp), "connector both", role="comparison"))
    items.append(Label((0.5 * (x_from + x_to), y_cmp - 0.12),
                       "comparison and limiting-case validation where assumptions overlap -- not a screening gate",
                       "small label", anchor="north", role="comparison"))
    if labels:
        items.append(Label((0.5 * (left + right), y_cmp - 1.0),
                           "Not yet in VAFT: " + "; ".join(ABSENT), "note", anchor="north", role="note"))
        items.append(Label((0.5 * (left + right), y_cmp - 1.65),
                           "theory $\\to$ formula $\\to$ equilibrium-facing process $\\to$ plot: these cells are the "
                           "formula layer; evaluating them on an equilibrium is vaft.process, running solvers vaft.code",
                           "note", anchor="north", role="note"))
    model = {"statuses": tuple(k for k, _ in STATUSES), "problems": tuple(k for k, _ in PROBLEMS),
             "cells": dict(CELLS), "absent": ABSENT,
             "centers": {k: (b.x, b.y) for k, b in cells.items()}}
    return Diagram("stability_diagnostic_taxonomy", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# Suydam and Mercier on one profile
# ---------------------------------------------------------------------------

#: the schematic profile: Wesson's peaked current (q_a, nu), p = p0 (1 - x^2)^2, minor radius, field
_Q_A, _NU, _P0, _A, _B = 3.0, 3.0, 4.0e4, 0.5, 0.5


def interchange_criteria(*, labels: bool = True) -> Diagram:
    r"""Suydam's and Mercier's local interchange criteria across one schematic profile.

    $q(r)$ is Wesson's peaked-current profile (``peaked_current_safety_factor``,
    $q_a = 3$, $\nu = 3$, so $q_0 = 0.75$) and $p \propto (1 - r^2/a^2)^2$. Both
    criteria (``suydam_criterion``, ``mercier_criterion_circular``) are
    normalised by $p_0/a$; positive means the criterion holds. Mercier is
    Suydam plus $-p'q^2$: outward pressure fall stabilises outside $q = 1$,
    so its violated region stays inside $r_1$ while Suydam's extends past it.
    Both are necessary conditions for local interchanges, not global
    stability.
    """
    labels = _check_labels(labels)
    x = np.linspace(0.01, 1.0, 397)
    r = _A * x
    q = peaked_current_safety_factor(x, _Q_A, _NU)
    p = _P0 * (1.0 - x ** 2) ** 2
    dq, dp = gradient(r, q), gradient(r, p)
    scale = _A / _P0
    suydam = suydam_criterion(r, _B, q, dq, dp) * scale
    mercier = mercier_criterion_circular(r, _B, q, dq, dp) * scale
    r1 = float(np.interp(1.0, q, x))
    lo, hi = -1.2, 2.4
    chart = Chart(x_range=(0.0, 1.08), y_range=(lo, hi))
    chart.curves.update({
        "suydam": np.stack([x, suydam], -1), "mercier": np.stack([x, mercier], -1),
        "zero": np.array([[0.0, 0.0], [1.05, 0.0]]), "q_one": np.array([[r1, lo], [r1, hi]]),
    })

    def exit_x(values):
        return float(x[np.argmax(values >= hi)])

    def last_violation(values):
        return float(x[values < 0.0].max()) if np.any(values < 0.0) else 0.0

    chart.parameters.update({"q_a": _Q_A, "nu": _NU, "q0": float(q[0]), "r1": r1,
                             "suydam_violated_to": last_violation(suydam),
                             "mercier_violated_to": last_violation(mercier)})
    scene = render_chart(chart, x_label="$r/a$", y_label="criterion $\\times\\, a/p_0$", y_ticks=(0.0,),
                         curve_styles={"zero": "approx", "q_one": "rational", "suydam": "slope plus",
                                       "mercier": "outer solution"},
                         region_text={}, x_ticks=(r1, 1.0), x_tick_text=("$r_1$", "$1$"))
    items: List = []
    if labels:
        at = chart.to_cm
        items += [
            # each curve is named where it leaves the top of the frame
            Label(tuple(at(np.array([exit_x(mercier), hi]))), "Mercier (torus)", "small label",
                  anchor="south", role="mercier"),
            Label(tuple(at(np.array([exit_x(suydam), hi]))), "Suydam (cylinder)", "small label",
                  anchor="south", role="suydam"),
            Label(tuple(at(np.array([r1 - 0.02, hi - 0.1]))), "$q = 1$", "small label", anchor="north east",
                  role="q_one"),
            Label(tuple(at(np.array([0.02, 0.12]))), "holds", "small label", anchor="south west", role="sign"),
            Label(tuple(at(np.array([0.02, -0.3]))), "violated", "small label", anchor="north west", role="sign"),
            Label((CHART_WIDTH / 2, -1.4), f"$\\displaystyle {formula_equation(suydam_criterion)}$", "formula box",
                  anchor="north", role="equations"),
            Label((CHART_WIDTH / 2, -2.75), f"$\\displaystyle {formula_equation(mercier_criterion_circular)}$",
                  "formula box", anchor="north", role="equations"),
            Label((CHART_WIDTH / 2, -4.1), "Schematic $q$ and $p$; both are necessary conditions for local "
                  "interchanges, not global stability", "note", anchor="north", role="note"),
        ]
    return Diagram("interchange_criteria", scene + Scene(tuple(items)), model=chart)
