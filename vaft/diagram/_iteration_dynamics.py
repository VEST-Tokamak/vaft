"""Iteration behaviour, branch bifurcation and branch selection: why a solver does not converge (#1093).

``iteration_behavior``
    fixed-point convergence, divergence, a 2-cycle and a period-4 limit cycle
    in one common format ($x_k$ against $k$), computed from exact maps;
``branch_bifurcation``
    the fold (saddle-node) normal form $\\dot x = \\lambda + x - x^3$: stable
    and unstable branches, the two folds, the jumps and the hysteresis loop;
``basin_of_attraction``
    the same normal form at $\\lambda = 0$: two stable solutions, and the
    initial condition decides which one a relaxation reaches;
``grid_induced_two_cycle``
    a numerical artifact: a continuous optimum between two grid nodes makes
    the discrete state hop between them while the fit hardly changes.

The message: non-convergence can come from true branch structure or from
numerical cycling, and the two have to be told apart before a
non-converged solution is called unphysical. These are generic concept
diagrams -- no EFIT shot, residual or grid comparison is shown or implied.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: iterations drawn in every panel of ``iteration_behavior``
N_ITERATIONS = 24
#: the maps behind ``iteration_behavior``: name -> (title, map description, parameter)
ITERATION_MAPS = {
    "fixed_point": ("fixed-point convergence", "logistic", 2.8),
    "divergence": ("divergence", "linear", 1.3),
    "two_cycle": ("2-cycle", "logistic", 3.2),
    "limit_cycle": ("limit cycle (period 4)", "logistic", 3.5),
}
_X0 = {"logistic": 0.2, "linear": 0.52}
#: panel size and spacing of ``iteration_behavior`` [cm]
_PW, _PH, _GAP = 4.6, 2.8, 1.1


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def iterate(name: str, n: int = N_ITERATIONS) -> np.ndarray:
    """$x_0 \\ldots x_n$ of one ``ITERATION_MAPS`` entry, rounded each step so every platform draws the same orbit."""
    _, kind, p = ITERATION_MAPS[name]
    x = [_X0[kind]]
    for _ in range(n):
        if kind == "logistic":
            nxt = p * x[-1] * (1.0 - x[-1])  # x_{k+1} = r x_k (1 - x_k)
        else:
            nxt = 0.5 + p * (x[-1] - 0.5)  # x_{k+1} = x* + a (x_k - x*), |a| > 1
        x.append(round(nxt, 12))
    return np.asarray(x)


def _panel(name: str, labels: bool) -> List:
    title, kind, p = ITERATION_MAPS[name]
    x = iterate(name)
    k = np.arange(x.size)
    sx, sy = _PW / N_ITERATIONS, _PH / 1.0
    shown = x <= 1.0
    pts = np.stack([k[shown] * sx, x[shown] * sy], axis=-1)
    items: List = [Arrow((0.0, 0.0), (_PW + 0.3, 0.0), "chart axis", role="axes"),
                   Arrow((0.0, 0.0), (0.0, _PH + 0.3), "chart axis", role="axes"),
                   Polyline.of(pts, "connector line", role=f"{name}:orbit")]
    items += [Marker(tuple(pt), "o", "xpoint", role=f"{name}:iterate") for pt in pts]
    if not shown.all():
        # the orbit leaves the box: an arrow from the last point inside
        last = pts[-1]
        items.append(Arrow(tuple(last), (last[0] + 0.35, _PH + 0.25), "vector", role=f"{name}:escape"))
    fixed = 1.0 - 1.0 / p if kind == "logistic" else 0.5
    items.append(Polyline.of([(0.0, fixed * sy), (_PW, fixed * sy)], "approx", role=f"{name}:fixed_point"))
    if labels:
        items += [Label((_PW / 2, _PH + 0.35), title, "small label", anchor="south", role=f"{name}:title"),
                  Label((_PW + 0.3, -0.1), "$k$", "small label", anchor="north", role="axes"),
                  Label((-0.1, _PH + 0.3), "$x_k$", "small label", anchor="east", role="axes")]
    return items


def iteration_behavior(*, labels: bool = True) -> Diagram:
    r"""Four iteration behaviours in one format: state $x_k$ against iteration $k$.

    Computed, not sketched: the logistic map $x_{k+1} = rx_k(1 - x_k)$ at
    $r = 2.8$ (convergence to $x^* = 1 - 1/r$), $3.2$ (a 2-cycle) and $3.5$
    (a period-4 cycle), and $x_{k+1} = x^* + a(x_k - x^*)$ with $a = 1.3$
    (divergence). The dashed line is the fixed point $x^*$ in each panel --
    it exists in all four; only in the first does the iteration reach it.
    """
    labels = _check_labels(labels)
    items: List = []
    offsets: Dict[str, tuple] = {}
    for i, name in enumerate(ITERATION_MAPS):
        offsets[name] = ((i % 2) * (_PW + _GAP + 0.6), (1 - i // 2) * (_PH + _GAP + 0.3))
        items += Scene(tuple(_panel(name, labels))).transformed(offset=offsets[name]).items
    if labels:
        items.append(Label((_PW + _GAP / 2 + 0.3, -0.85), "the dashed line is the fixed point $x^*$ of each map",
                           "note", anchor="north", role="note"))
    return Diagram("iteration_behavior", Scene(tuple(items)),
                   model={"maps": ITERATION_MAPS, "x0": _X0, "orbits": {n: iterate(n) for n in ITERATION_MAPS},
                          "panel_offsets": offsets})


def fold_normal_form_equilibria(x):
    """$\\lambda(x) = x^3 - x$: the equilibria of $\\dot x = \\lambda + x - x^3$, parametrised by $x$."""
    x = np.asarray(x, dtype=float)
    return x**3 - x


#: the two folds of the normal form: x = -+1/sqrt(3) at lambda = +-2/(3 sqrt(3))
FOLD_X = 1.0 / np.sqrt(3.0)
FOLD_LAMBDA = 2.0 / (3.0 * np.sqrt(3.0))


def branch_bifurcation(*, labels: bool = True) -> Diagram:
    r"""Stable and unstable branches, two folds, the jumps and the hysteresis loop.

    The saddle-node normal form $\dot x = \lambda + x - x^3$: equilibria on
    $\lambda = x^3 - x$, stable where $1 - 3x^2 < 0$ (solid) and unstable
    between the folds $x = \pm 1/\sqrt3$ (dashed). Raising $\lambda$ along
    the lower branch, the solution disappears at the fold
    $\lambda = 2/(3\sqrt3)$ and jumps to the upper branch; lowering it, the
    upper branch is followed down to $\lambda = -2/(3\sqrt3)$ -- the path
    depends on the direction, the hysteresis. Between the folds three
    equilibria coexist.
    """
    labels = _check_labels(labels)
    x = np.linspace(-1.45, 1.45, 581)
    lam = fold_normal_form_equilibria(x)
    stable = np.abs(x) > FOLD_X
    chart = Chart(x_range=(-1.0, 1.0), y_range=(-1.6, 1.6))
    chart.curves.update({"stable_lower": np.stack([lam[x < -FOLD_X], x[x < -FOLD_X]], axis=-1),
                         "stable_upper": np.stack([lam[x > FOLD_X], x[x > FOLD_X]], axis=-1),
                         "unstable": np.stack([lam[~stable], x[~stable]], axis=-1)})
    # the jump lands on the other stable branch: the third root of x^3 - x - lambda_fold
    chart.points.update({"fold_upper": (-FOLD_LAMBDA, FOLD_X), "fold_lower": (FOLD_LAMBDA, -FOLD_X),
                         "jump_up_to": (FOLD_LAMBDA, 2.0 * FOLD_X), "jump_down_to": (-FOLD_LAMBDA, -2.0 * FOLD_X)})
    chart.labels.update({"stable": (0.75, 0.88), "unstable": (0.1, 0.5)})
    chart.parameters.update({"fold_lambda": FOLD_LAMBDA, "fold_x": FOLD_X})
    scene = render_chart(
        chart, x_label="control parameter $\\lambda$", y_label="state $x$",
        curve_styles={"stable_lower": "boundary", "stable_upper": "boundary", "unstable": "approx"},
        region_text={"stable": "stable", "unstable": "unstable"} if labels else {},
        note="Saddle-node normal form $\\dot x = \\lambda + x - x^3$" if labels else "",
    )
    cm = chart.to_cm
    items = list(scene.items)
    items += [Marker(tuple(cm(np.array(chart.points[p]))), "x", "xpoint", role="fold") for p in ("fold_upper",
                                                                                                "fold_lower")]
    items += [Arrow(tuple(cm(np.array(chart.points["fold_lower"]))), tuple(cm(np.array(chart.points["jump_up_to"]))),
                    "drift", role="jump_up"),
              Arrow(tuple(cm(np.array(chart.points["fold_upper"]))),
                    tuple(cm(np.array(chart.points["jump_down_to"]))), "drift", role="jump_down")]
    # direction of travel along the branches
    items += [Arrow(tuple(cm(np.array([-0.6, _branch_x(-0.6, -1)]))),
                    tuple(cm(np.array([-0.35, _branch_x(-0.35, -1)]))), "vector", role="increasing"),
              Arrow(tuple(cm(np.array([0.6, _branch_x(0.6, 1)]))), tuple(cm(np.array([0.35, _branch_x(0.35, 1)]))),
                    "vector", role="decreasing")]
    bistable = [cm(np.array([-FOLD_LAMBDA, -1.4])), cm(np.array([FOLD_LAMBDA, -1.4]))]
    items.append(Arrow(tuple(bistable[0]), tuple(bistable[1]), "connector", role="bistable_range", both=True))
    if labels:
        items += [Label(tuple(cm(np.array(chart.points["fold_lower"])) + [0.15, -0.15]), "fold", "small label",
                        anchor="north west", role="fold"),
                  Label(tuple(cm(np.array(chart.points["fold_upper"])) + [-0.15, 0.15]), "fold", "small label",
                        anchor="south east", role="fold"),
                  Label(tuple(cm(np.array([-0.62, _branch_x(-0.62, -1)])) + [0.0, -0.25]), "$\\lambda$ increasing",
                        "small label", anchor="north", role="increasing"),
                  Label(tuple(cm(np.array([0.62, _branch_x(0.62, 1)])) + [0.0, 0.25]), "$\\lambda$ decreasing",
                        "small label", anchor="south", role="decreasing"),
                  Label(tuple((bistable[0] + bistable[1]) / 2 + [0.0, 0.08]), "three equilibria", "small label",
                        anchor="south", role="bistable_range"),
                  Label(tuple(cm(np.array([FOLD_LAMBDA, 0.2])) + [0.12, 0.0]), "jump", "small label",
                        anchor="west", role="jump_up"),
                  Label(tuple(cm(np.array([-FOLD_LAMBDA, -0.2])) + [-0.12, 0.0]), "jump", "small label",
                        anchor="east", role="jump_down")]
    return Diagram("branch_bifurcation", Scene(tuple(items)), model=chart)


def _branch_x(lam: float, sign: int) -> float:
    """The stable equilibrium of the normal form at ``lam`` on the upper (+1) or lower (-1) branch."""
    roots = np.roots([1.0, 0.0, -1.0, -lam])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    return float(real[-1] if sign > 0 else real[0])


def relaxation(x0, t):
    """Exact solution of $\\dot x = x - x^3$ (the normal form at $\\lambda = 0$) from $x(0) = x_0$."""
    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t, dtype=float)
    growth = np.exp(t)
    return x0 * growth / np.sqrt(1.0 + x0**2 * (growth**2 - 1.0))


#: initial conditions of ``basin_of_attraction``: symmetric, some close to the basin boundary
INITIAL_CONDITIONS = (-1.45, -0.7, -0.25, -0.04, 0.04, 0.25, 0.7, 1.45)


def basin_of_attraction(*, labels: bool = True) -> Diagram:
    r"""Branch selection by the initial condition: two stable solutions at one control parameter.

    The normal form of ``branch_bifurcation`` at $\lambda = 0$ relaxed in
    time, $\dot x = x - x^3$, solved exactly: every start above the unstable
    equilibrium $x = 0$ -- the basin boundary -- reaches branch B ($x = 1$),
    every start below reaches branch A ($x = -1$). Both are physical
    solutions; which one an iteration returns is decided by where it starts.
    """
    labels = _check_labels(labels)
    t = np.linspace(0.0, 6.0, 241)
    chart = Chart(x_range=(0.0, 6.0), y_range=(-1.6, 1.6))
    for x0 in INITIAL_CONDITIONS:
        chart.curves[f"trajectory {x0:+.2f}"] = np.stack([t, relaxation(x0, t)], axis=-1)
        chart.points[f"start {x0:+.2f}"] = (0.0, x0)
    chart.curves.update({"branch_A": np.array([[0.0, -1.0], [6.0, -1.0]]),
                         "branch_B": np.array([[0.0, 1.0], [6.0, 1.0]]),
                         "basin_boundary": np.array([[0.0, 0.0], [6.0, 0.0]])})
    chart.labels.update({"basin_B": (4.6, 0.5), "basin_A": (4.6, -0.5)})
    chart.parameters.update({"lambda": 0.0})
    styles = {name: "connector line" for name in chart.curves if name.startswith("trajectory")}
    styles.update({"branch_A": "boundary", "branch_B": "boundary", "basin_boundary": "approx"})
    scene = render_chart(
        chart, x_label="relaxation time", y_label="state $x$", curve_styles=styles,
        region_text={"basin_B": "basin of B", "basin_A": "basin of A"} if labels else {},
        note="$\\dot x = x - x^3$: the same control parameter, two stable solutions" if labels else "",
    )
    items = list(scene.items)
    items += [Marker(tuple(chart.to_cm(np.array(chart.points[p]))), "o", "xpoint", role="initial_condition")
              for p in chart.points]
    if labels:
        right = chart.to_cm(np.array([6.0, 0.0]))[0] + 0.15
        items += [Label((right, float(chart.to_cm(np.array([0.0, 1.0]))[1])), "branch B", "small label",
                        anchor="west", role="branch_B"),
                  Label((right, float(chart.to_cm(np.array([0.0, -1.0]))[1])), "branch A", "small label",
                        anchor="west", role="branch_A"),
                  Label((right, float(chart.to_cm(np.array([0.0, 0.0]))[1])),
                        "\\begin{tabular}{l}basin boundary\\\\(unstable)\\end{tabular}", "small label", anchor="west",
                        role="basin_boundary")]
    return Diagram("basin_of_attraction", Scene(tuple(items)), model=chart)


#: grid of ``grid_induced_two_cycle``: nodes per side and spacing [cm]
_NX, _NY, _DX = 6, 5, 1.0


def grid_induced_two_cycle(*, labels: bool = True) -> Diagram:
    r"""A numerical 2-cycle: the discrete state hops between two adjacent grid nodes.

    The continuous optimum (cross) lies between nodes A and B. A solver
    that snaps the state to the grid -- a magnetic axis located on the
    nearest node, say -- lands on A, re-solves, lands on B, and back. The
    fit quality and the equilibrium are nearly the same on both; only the
    discrete index alternates, so a convergence test on the state is never
    met. On the right, the index alternates while the fit residual is flat:
    the signature that separates this from a physical branch switch.
    """
    labels = _check_labels(labels)
    xs, ys = np.arange(_NX) * _DX, np.arange(_NY) * _DX
    items: List = []
    for x in xs:
        items.append(Polyline.of([(x, ys[0]), (x, ys[-1])], "mesh", role="grid"))
    for y in ys:
        items.append(Polyline.of([(xs[0], y), (xs[-1], y)], "mesh", role="grid"))
    a, b = (2.0, 2.0), (3.0, 2.0)
    optimum = (2.5, 2.1)
    items += [Marker(a, "o", "opoint", role="cell_A"), Marker(b, "o", "opoint", role="cell_B"),
              Marker(optimum, "x", "xpoint", role="continuous_optimum"),
              Arrow((a[0] + 0.1, a[1] + 0.35), (b[0] - 0.1, b[1] + 0.35), "drift", role="hop"),
              Arrow((b[0] - 0.1, b[1] - 0.35), (a[0] + 0.1, a[1] - 0.35), "drift", role="hop")]
    # right: discrete index and fit residual against iteration
    ox, w, h = xs[-1] + 1.6, 4.0, ys[-1]
    k = np.arange(12)
    index = np.where(k % 2 == 0, 0.62, 0.38) * h
    residual = np.full(k.shape, 0.15 * h)
    kx = ox + k * w / 11
    items += [Arrow((ox, 0.0), (ox + w + 0.3, 0.0), "chart axis", role="axes"),
              Arrow((ox, 0.0), (ox, h + 0.3), "chart axis", role="axes"),
              Polyline.of(np.stack([kx, index], axis=-1), "connector line", role="index"),
              Polyline.of(np.stack([kx, residual], axis=-1), "boundary", role="residual")]
    items += [Marker((float(x), float(y)), "o", "xpoint", role="index") for x, y in zip(kx, index)]
    if labels:
        items += [Label((a[0], a[1] - 0.2), "A", "small label", anchor="north east", role="cell_A"),
                  Label((b[0], b[1] - 0.2), "B", "small label", anchor="north west", role="cell_B"),
                  Polyline.of([(optimum[0], optimum[1] + 0.12), (optimum[0], ys[-1] + 0.25)], "leader line",
                              role="continuous_optimum"),
                  Label((optimum[0], ys[-1] + 0.3), "continuous optimum", "small label", anchor="south",
                        role="continuous_optimum"),
                  Label((xs[-1] / 2, -0.35), "solver grid", "small label", anchor="north", role="grid"),
                  Label((ox + w + 0.3, -0.1), "iteration $k$", "small label", anchor="north", role="axes"),
                  Label((ox + w, index[-1] + 0.95), "node index: A, B, A, ...", "small label", anchor="south east",
                        role="index"),
                  Label((ox + w, residual[-1] + 0.15), "fit residual: flat", "small label", anchor="south east",
                        role="residual"),
                  Label((ox / 2 + w / 2, -1.1), "Numerical 2-cycle: same fit, alternating discrete state -- "
                        "not a second physical branch", "note", anchor="north", role="note")]
    return Diagram("grid_induced_two_cycle", Scene(tuple(items)),
                   model={"cell_A": a, "cell_B": b, "continuous_optimum": optimum, "grid_spacing": _DX})
