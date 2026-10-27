r"""Safety-factor asymptotics at a separatrix and the numerical treatment they call for (#2152).

``separatrix_q_asymptotics``
    a limited and a lower single-null Solov'ev equilibrium side by side: the
    boundary topology, $q(\psi_N)$, $q$ against $\ln[1/(1-\psi_N)]$ with the
    X-point's logarithmic asymptote, and where the rational surfaces fall;
``xpoint_refinement_hierarchy``
    why a non-degenerate X-point makes the edge numerically delicate, and the
    separate treatments CHEASE, DCON/GPEC and the coordinate construction
    apply, all ending in validation.

The safety factor here is not a gridded contour integral: it is traced on
the analytic Cerfon-Freidberg flux of ``vaft.process.equilibrium.
solovev_example`` -- $q = (F/2\pi)\oint d\ell/(R|\nabla\psi|)$ along the exact
field line -- so it stays accurate to $1 - \psi_N = 10^{-8}$, far closer to the
separatrix than any stored grid resolves. The asymptotic slope is
``vaft.formula.geometry.xpoint_log_q_coefficient`` evaluated from the analytic
X-point Hessian, never fitted; the rational surfaces are found on the traced
profile by ``vaft.process.equilibrium.rational_surfaces``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from vaft.formula.equilibrium import separatrix_rational_offset
from vaft.formula.geometry import xpoint_log_q_coefficient, xpoint_resolution_length, xpoint_saddle_expansion

from ._chart import Chart, render_chart
from ._concept import box, connector
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the two equilibria compared: identical shape, current and field, only the boundary topology differs
TOPOLOGIES = ("limited", "single_null")
#: closest approach to the separatrix that is traced; far above the flux round-off of the analytic X-point
DELTA_MIN = 1.0e-8
#: psihigh of the packaged equil.in (vaft/data/gpec): where DCON/GPEC's equilibrium coordinates end
DCON_PSIHIGH = 0.994
#: toroidal mode numbers whose resonances are marked
MODE_NUMBERS = (1, 2, 3)


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


@lru_cache(maxsize=None)
def _equilibrium(topology: str) -> Dict:
    """The analytic flux of one Solov'ev example, its axis, X-points and F, in per-radian units."""
    from scipy.optimize import root

    from vaft.process._equilibrium_parametric import _cf_dx, _cf_dy, _cf_eval, _cf_terms
    from vaft.process.equilibrium import solovev_example

    eq = solovev_example(topology, convention=11)
    md, r0 = eq.metadata, float(eq.r0)
    terms = _cf_terms(md["coefficients"], md["a_parameter"])
    tx, ty = _cf_dx(terms), _cf_dy(terms)
    txx, tyy, txy = _cf_dx(tx), _cf_dy(ty), _cf_dy(tx)
    psi0 = float(md["psi0"])
    ffprime = -psi0 * md["a_parameter"] / r0**2
    f0 = float(eq.bt0) * r0
    ax, ay = eq.magnetic_axis[0] / r0, eq.magnetic_axis[1] / r0
    psi_axis = float(_cf_eval(terms, ax, ay))  # normalized; zero on the boundary
    model = {"eq": eq, "r0": r0, "psi0": psi0, "terms": terms, "tx": tx, "ty": ty, "axis": (ax, ay),
             "psi_axis": psi_axis, "f": lambda p: float(np.sqrt(f0**2 + 2.0 * ffprime * psi0 * p)),
             "x_points": []}
    for requested in md["x_points_requested"]:
        x, y = root(lambda p: [float(_cf_eval(tx, *p)), float(_cf_eval(ty, *p))],
                    [requested[0] / r0, requested[1] / r0], tol=1e-14).x
        # per-radian flux Psi = psi0 psi(R/R0, Z/R0): its (R, Z) Hessian is psi0/R0**2 times the normalized one
        h = [psi0 / r0**2 * float(_cf_eval(k, x, y)) for k in (txx, tyy, txy)]
        saddle = xpoint_saddle_expansion(*h)
        model["x_points"].append({"R": x * r0, "Z": y * r0, "F": model["f"](float(_cf_eval(terms, x, y))),
                                  "saddle": saddle})
    if model["x_points"]:
        model["A"] = xpoint_log_q_coefficient([p["F"] for p in model["x_points"]],
                                              [p["R"] for p in model["x_points"]],
                                              [p["saddle"].det_hessian for p in model["x_points"]])
    return model


def _eval(model: Dict, which: str, x, y):
    from vaft.process._equilibrium_parametric import _cf_eval

    return _cf_eval(model[which], x, y)


@lru_cache(maxsize=None)
def traced_q(topology: str, psi_n: float) -> float:
    """$|q|$ on one flux surface of the analytic example, traced once around the exact field line [-]."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    m = _equilibrium(topology)
    ax, ay = m["axis"]
    level = m["psi_axis"] * (1.0 - psi_n)
    x_start = brentq(lambda x: float(_eval(m, "terms", x, ay)) - level, ax + 1e-9, ax + 2.0, xtol=1e-15)

    # along the contour, d(x, y)/dtau = (-dpsi/dy, dpsi/dx) makes dtau = dl/|grad psi|; integrate 1/x and the angle
    def rhs(_, s):
        gx, gy = float(_eval(m, "tx", s[0], s[1])), float(_eval(m, "ty", s[0], s[1]))
        dx, dy = s[0] - ax, s[1] - ay
        return [-gy, gx, 1.0 / s[0], (dx * gx + dy * gy) / (dx * dx + dy * dy)]

    def turn(_, s):
        return abs(s[3]) - 2.0 * np.pi

    turn.terminal, turn.direction = True, 1
    sol = solve_ivp(rhs, (0.0, 1e12), [x_start, ay, 0.0, 0.0], events=turn, rtol=1e-10, atol=1e-14)
    if not sol.t_events[0].size:
        raise RuntimeError(f"the field line at psi_N = {psi_n} did not close")
    integral = float(sol.y_events[0][0][2])
    # q = F/(2 pi) oint dl/(R |grad Psi|) = F/(2 pi) (R0/|psi0|) oint dl_n/(x |grad psi_n|)
    return m["f"](level) / (2.0 * np.pi) * m["r0"] / abs(m["psi0"]) * integral


def _profile(topology: str, psi_n) -> np.ndarray:
    return np.array([traced_q(topology, round(float(p), 12)) for p in psi_n])


@lru_cache(maxsize=None)
def separatrix_model() -> Dict:
    """Everything the figure shows, computed once: profiles, asymptote, rational surfaces, resolution limit."""
    from vaft.process.equilibrium import rational_surfaces

    diverted = _equilibrium("single_null")
    A = diverted["A"]
    psi_n = np.concatenate([np.linspace(0.05, 0.9, 12), 1.0 - np.geomspace(0.1, 1e-4, 10)])
    delta = np.geomspace(0.3, DELTA_MIN, 22)
    out = {"A": A, "psi_n": psi_n, "delta": delta, "q_psi": {}, "q_delta": {}}
    for topology in TOPOLOGIES:
        out["q_psi"][topology] = _profile(topology, psi_n)
        out["q_delta"][topology] = _profile(topology, 1.0 - delta)
    # the regular offset B: what the traced q adds to the X-point slope at the closest traced surface
    q_close = out["q_delta"]["single_null"]
    B = float(q_close[-1] - A * np.log(1.0 / delta[-1]))
    out["B"] = B
    out["fitted_slope"] = float(np.polyfit(np.log(1.0 / delta[-6:]), q_close[-6:], 1)[0])
    # resonances on the traced diverted profile (the canonical finder), against the asymptotic prediction
    x_log = np.log(1.0 / delta)
    found = {}
    # rational_surfaces interpolates linearly in psi_N, so its profile is sampled densely in ln(1/delta)
    fine = np.geomspace(0.1, DELTA_MIN, 64)
    q_fine = _profile("single_null", 1.0 - fine)
    for n in MODE_NUMBERS:
        q_lo, q_hi = float(q_fine[0]), float(q_fine[-1])
        ms = [m for m in range(int(np.ceil(q_lo * n)), int(np.floor(q_hi * n)) + 1) if m / n > q_lo]
        records = rational_surfaces(1.0 - fine, q_fine, resonances=[(m, n) for m in ms])
        by_q = {round(r.q_target, 9): r for r in records}
        numeric = [float(np.log(1.0 / (1.0 - by_q[round(m / n, 9)].roots[-1].psi_norm))) for m in ms]
        predicted = [float(np.log(1.0 / separatrix_rational_offset(m, n, A, B))) for m in ms]
        found[n] = {"m": ms, "numeric": numeric, "predicted": predicted}
    out["rational"] = found
    # where a 129-point grid stops resolving surfaces near the X-point: l_X(delta) = grid spacing
    eq = diverted["eq"]
    h = float(np.diff(eq.r)[0])
    saddle = diverted["x_points"][0]["saddle"]
    kappa = min(saddle.kappa_u, saddle.kappa_v)
    span = abs(diverted["psi_axis"] * diverted["psi0"])  # |psi_X - psi_axis|, per radian
    out["grid_spacing"] = h
    out["delta_grid"] = float(kappa * h * h / (2.0 * span))
    assert abs(float(xpoint_resolution_length(out["delta_grid"] * span, kappa)) - h) < 1e-12 * h
    out["x_log"] = x_log
    return out


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

_PANEL = 0.72  # chart scale inside the 2x2 figure
_COLOUR = {"limited": "driftred", "single_null": "chartblue"}
_NAME = {"limited": "limited", "single_null": "lower single null"}


def _flux_map(topology: str, offset, width: float) -> List:
    """Closed surfaces, the boundary and, when diverted, the separatrix legs and the X-point."""
    from contourpy import LineType, contour_generator

    m = _equilibrium(topology)
    eq, r0 = m["eq"], m["r0"]
    frame = _equilibrium("single_null")["eq"]  # one frame for both, so the two plasmas are drawn to one scale
    r, z = frame.r, frame.z
    scale = width / float(r[-1] - r[0])
    rr, zz = np.meshgrid(r / r0, z / r0, indexing="ij")
    psi = np.asarray(_eval(m, "terms", rr, zz))
    gen = contour_generator(x=r, y=z, z=psi.T, line_type=LineType.Separate)

    def cm(pts):
        pts = np.asarray(pts, dtype=float)
        return np.stack([offset[0] + scale * (pts[:, 0] - r[0]), offset[1] + scale * pts[:, 1]], -1)

    items: List = [Polyline.of(cm(np.array([(r[0], z[0]), (r[-1], z[0]), (r[-1], z[-1]), (r[0], z[-1])])),
                               "machine", role=f"frame:{topology}", closed=True)]
    for f in (0.2, 0.4, 0.6, 0.8):
        for line in gen.lines(m["psi_axis"] * (1.0 - f)):
            items.append(Polyline.of(cm(np.round(line, 9)), "surface", role=f"surface:{topology}"))
    if topology == "limited":
        lcfs = eq.lcfs
        items.append(Polyline.of(cm(np.column_stack([lcfs.r, lcfs.z])), f"boundary,{_COLOUR[topology]}",
                                 role=f"boundary:{topology}", closed=True))
    else:
        # a hair inside the saddle value: core surface plus private-flux legs, never the tie at the null itself
        for line in gen.lines(m["psi_axis"] * 1e-6):
            items.append(Polyline.of(cm(np.round(line, 9)), "separatrix", role=f"boundary:{topology}"))
        for p in m["x_points"]:
            items.append(Marker(tuple(cm(np.array([[p["R"], p["Z"]]]))[0]), "x", "xpoint", role="x_point"))
    items.append(Marker(tuple(cm(np.array([[m["axis"][0] * r0, m["axis"][1] * r0]]))[0]), "o", "opoint",
                        role="axis"))
    return items


def _panel(chart: Chart, offset, **kwargs) -> List:
    return list(render_chart(chart, **kwargs).transformed(scale=_PANEL, offset=offset).items)


def _at(chart: Chart, xy, offset) -> Tuple[float, float]:
    p = chart.to_cm(np.asarray(xy, dtype=float)) * _PANEL + np.asarray(offset)
    return float(p[0]), float(p[1])


def separatrix_q_asymptotics(*, labels: bool = True) -> Diagram:
    r"""Finite edge $q$ for a limited plasma, logarithmic divergence for a diverted one (#2152).

    Two Solov'ev equilibria of ``solovev_example`` with the same shape,
    current and field, one limited and one lower single null:

    A. the boundary topology -- the limited LCFS, where $B_p$ stays finite,
       and the separatrix through an X-point, where $B_p \to 0$;
    B. $|q(\psi_N)|$, traced on the exact field lines: the limited profile
       ends at a finite $q_a$, the diverted one turns up at $\psi_N \to 1$;
    C. the same against $\ln[1/(1-\psi_N)]$: limited flattens, diverted
       becomes a straight line whose slope is
       ``xpoint_log_q_coefficient`` of the X-point Hessian (not fitted;
       the fitted slope agrees), with ``separatrix_q_asymptotic`` dashed. The
       ``psihigh`` of the packaged DCON/GPEC ``equil.in`` and the point where a 129-point grid stops
       resolving surfaces near the X-point (``xpoint_resolution_length``
       equal to the grid spacing) are marked;
    D. the resonances $|q| = m/n$ for $n = 1, 2, 3$: found on the traced
       profile by ``rational_surfaces`` (dots) and predicted by
       ``separatrix_rational_offset`` (ticks), evenly spaced by $1/(nA)$ in
       $\ln(1/\delta)$, i.e. exponentially crowded in $\psi_N$.

    Numerical $q$, the asymptotic form, the rational crossings and the
    resolution limit are drawn distinctly. This is an analytic reference,
    not a machine equilibrium; $A$ is specific to it.
    """
    labels = _check_labels(labels)
    data = separatrix_model()
    A, B = data["A"], data["B"]
    items: List = []
    col, row = 10.6, -8.6

    # A: topology
    origin = (0.0, 0.0)
    for i, topology in enumerate(TOPOLOGIES):
        items += _flux_map(topology, (origin[0] + 0.6 + i * 4.7, origin[1] + 2.4), 3.6)
    if labels:
        items.append(Label((0.0, 6.0), "\\textbf{A}\\quad boundary topology", "label", anchor="south west",
                           role="title:A"))
        items += [Label((2.4, -0.85), "limited: $B_p \\neq 0$ on the\\\\boundary, $q_a$ finite",
                        "small label,align=center", anchor="north", role="caption:limited"),
                  Label((7.1, -0.85), "diverted: $B_p \\to 0$ at the\\\\X-point, $q \\to \\infty$",
                        "small label,align=center", anchor="north", role="caption:single_null")]

    # B: q(psi_N)
    q_top = 5.0
    chart_b = Chart(x_range=(0.0, 1.05), y_range=(0.0, q_top))
    styles = {}
    for topology in TOPOLOGIES:
        chart_b.curves[topology] = np.stack([data["psi_n"], data["q_psi"][topology]], -1)
        styles[topology] = f"boundary,{_COLOUR[topology]}"
    ob = (col + 1.2, 1.0)
    items += _panel(chart_b, ob, x_label="$\\psi_N$", y_label="$|q|$", curve_styles=styles, region_text={},
                    x_ticks=(0.0, 0.5, 1.0), y_ticks=(1, 2, 3, 4))
    if labels:
        items.append(Label((col, 6.0), "\\textbf{B}\\quad safety-factor profile", "label", anchor="south west",
                           role="title:B"))
        q_lim = float(data["q_psi"]["limited"][-1])
        q_div = float(data["q_psi"]["single_null"][-1])
        items.append(Label(_at(chart_b, (1.0, q_lim - 0.15), ob), "limited:\\\\$q_a$ finite",
                           "small label,align=left", anchor="north west", role="curve:limited"))
        items.append(Label(_at(chart_b, (0.97, q_div + 0.1), ob), "diverted: rising\\\\at $\\psi_N \\to 1$",
                           "small label,align=right", anchor="south east", role="curve:single_null"))

    # C: q against ln(1/delta)
    x_log = data["x_log"]
    x_top = float(np.ceil(x_log[-1])) + 1.0
    chart_c = Chart(x_range=(0.0, x_top), y_range=(2.0, q_top))
    for topology in TOPOLOGIES:
        chart_c.curves[topology] = np.stack([x_log, data["q_delta"][topology]], -1)
    asym_x = np.linspace(3.0, x_log[-1], 50)
    chart_c.curves["asymptote"] = np.stack([asym_x, A * asym_x + B], -1)
    x_dcon = float(np.log(1.0 / (1.0 - DCON_PSIHIGH)))
    x_grid = float(np.log(1.0 / data["delta_grid"]))
    chart_c.curves["dcon"] = np.array([[x_dcon, 2.0], [x_dcon, q_top]])
    chart_c.curves["grid"] = np.array([[x_grid, 2.0], [x_grid, q_top]])
    oc = (1.2, row + 1.0)
    items += _panel(chart_c, oc, x_label="$\\ln[1/(1-\\psi_N)]$", y_label="$|q|$",
                    curve_styles={"limited": styles["limited"], "single_null": styles["single_null"],
                                  "asymptote": "approx", "dcon": "tick,dashed", "grid": "tick,densely dotted"},
                    region_text={}, x_ticks=tuple(range(0, int(x_top), 5)), y_ticks=(2, 3, 4, 5))
    if labels:
        items.append(Label((0.0, row + 6.0), "\\textbf{C}\\quad logarithmic representation", "label",
                           anchor="south west", role="title:C"))
        x_mid = 0.5 * (x_grid + x_log[-1])
        items.append(Label(_at(chart_c, (x_mid + 0.6, A * x_mid + B - 0.15), oc),
                           f"diverted: slope $A = {A:.3f}$\\\\from the X-point Hessian", "small label,align=left",
                           anchor="north west", role="slope"))
        q_lim = float(data["q_delta"]["limited"][-1])
        items.append(Label(_at(chart_c, (x_log[-1], q_lim - 0.12), oc), "limited", "small label",
                           anchor="north east", role="curve:limited"))
        items.append(Label(_at(chart_c, (x_dcon - 0.2, q_top - 0.1), oc), "DCON/GPEC\\\\$\\psi_\\mathrm{high}$",
                           "small label,align=right", anchor="north east", role="dcon"))
        items.append(Label(_at(chart_c, (x_grid + 0.2, q_top - 0.1), oc), "129-point grid\\\\stops resolving",
                           "small label,align=left", anchor="north west", role="grid"))

    # D: rational surfaces
    chart_d = Chart(x_range=(0.0, x_top), y_range=(0.4, len(MODE_NUMBERS) + 0.6))
    od = (col + 1.2, row + 1.0)
    items += _panel(chart_d, od, x_label="$\\ln[1/(1-\\psi_N)]$", y_label="$n$", curve_styles={}, region_text={},
                    x_ticks=tuple(range(0, int(x_top), 5)), y_ticks=MODE_NUMBERS)
    for n in MODE_NUMBERS:
        entry = data["rational"][n]
        for m, xn, xp in zip(entry["m"], entry["numeric"], entry["predicted"]):
            items.append(Marker(_at(chart_d, (xn, n), od), "o", "opoint", role=f"rational:{m}/{n}"))
            c = _at(chart_d, (xp, n), od)
            items.append(Polyline.of([(c[0], c[1] - 0.28), (c[0], c[1] + 0.28)], "tick,black",
                                     role=f"predicted:{m}/{n}"))
    if labels:
        items.append(Label((col, row + 6.0), "\\textbf{D}\\quad rational surfaces accumulate", "label",
                           anchor="south west", role="title:D"))
        items.append(Label(_at(chart_d, (0.5, len(MODE_NUMBERS) + 0.35), od),
                           "dots: found on the traced $q$; ticks: $\\delta_{m/n} = C e^{-|m|/(|n|A)}$",
                           "small label", anchor="south west", role="legend"))
    if labels:
        early = [(n, xp) for n in MODE_NUMBERS for xn, xp in zip(data["rational"][n]["numeric"],
                                                                 data["rational"][n]["predicted"]) if abs(xn - xp) > 0.1]
        if early:
            n0, x0 = min(early, key=lambda e: e[1])
            items.append(Label(_at(chart_d, (x0 - 0.3, n0 - 0.32), od), "nearest the core: not yet\\\\in the "
                               "logarithmic regime", "small label,align=left", anchor="north west", role="early"))
        items.append(Label((col - 0.5, row - 1.2),
                           f"Analytic Solov'ev reference (same shape, current and field), $q$ traced on exact field "
                           f"lines to $1-\\psi_N = 10^{{-8}}$. Fitted slope {data['fitted_slope']:.3f}, X-point "
                           f"$A$ {A:.3f}; $A$ is specific to this equilibrium",
                           "note", anchor="north", role="note"))
    return Diagram("separatrix_q_asymptotics", Scene(tuple(items)),
                   model={"A": A, "B": B, "fitted_slope": data["fitted_slope"],
                          "delta_grid": data["delta_grid"], "grid_spacing": data["grid_spacing"],
                          "dcon_psihigh": DCON_PSIHIGH, "rational": data["rational"],
                          "q_edge_limited": float(data["q_delta"]["limited"][-1]),
                          "topologies": TOPOLOGIES})


def xpoint_refinement_hierarchy(*, labels: bool = True) -> Diagram:
    r"""Why the separatrix needs its own numerical treatment, and which code applies which (#2152).

    A non-degenerate X-point has four consequences: $\nabla\psi \to 0$ so
    $B_p \to 0$; $q \sim A\ln(1/\delta)$ with $dq/d\psi_N \sim A/\delta$;
    rational surfaces accumulating, which sets the radial and harmonic
    resolution; and an ill-conditioned flux-coordinate map, which spoils
    metrics and quadrature. Three separate numerical responses follow:
    CHEASE refines the boundary and mesh (``target_psin``, ``ns``/``nt``) and
    so changes the equilibrium, not only its coordinates; DCON/GPEC limit the
    domain they integrate (``psihigh`` for the coordinates; ``qhigh``,
    ``sas_flag``/``dmlim`` and the ``psiedge`` scan for the stability
    integration, which are not a fixed cutoff); and a coordinate treatment
    can integrate the asymptotic form or adapt the grid (Benjamin, Logan and
    Hansen 2025). Each ends in validation from the actual equilibrium,
    energies and mode structures: these are conditioning descriptors, not one
    stiffness parameter, and no setting is prescribed.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []

    def node(key, x, y, w, h, text, style="concept box"):
        b = box(x, y, w, h, text, style=style, role=f"node:{key}", latex=True)
        items.extend(b.items)
        return b

    def link(a, b, src, dst, style="connector"):
        items.append(connector(a, b, style=style, role=f"edge:{src}->{dst}"))
        edges.append((src, dst))

    root = node("xpoint", 0.0, 9.6, 6.4, 1.0, "\\textbf{Non-degenerate X-point}\\\\{\\small $\\nabla\\psi = 0$, "
                "$\\det H_\\psi < 0$}", style="concept strong")
    causes = [
        ("field", "$B_p \\to 0$", "linearly at the null"),
        ("q", "$q \\sim A\\ln(1/\\delta)$", "$dq/d\\psi_N \\sim A/\\delta$"),
        ("rational", "rational surfaces accumulate", "radial and harmonic resolution"),
        ("mapping", "flux-coordinate map ill-conditioned", "metrics and quadrature"),
    ]
    xs = (-9.3, -3.1, 3.1, 9.3)
    cause_boxes = {}
    for (key, title, sub), x in zip(causes, xs):
        cause_boxes[key] = node(key, x, 7.2, 5.6, 1.35, "\\textbf{" + title + "}\\\\{\\small " + sub + "}")
        link(root, cause_boxes[key], "xpoint", key)
    strategies = [
        ("chease", "CHEASE refinement", "boundary and mesh: \\texttt{target\\_psin}, \\texttt{ns}/\\texttt{nt};"
         " changes the equilibrium"),
        ("dcon", "DCON / GPEC domain", "\\texttt{psihigh} (coordinates); \\texttt{qhigh}, \\texttt{sas\\_flag}/"
         "\\texttt{dmlim}, \\texttt{psiedge} scan (stability)"),
        ("coordinates", "Coordinate treatment", "asymptotic integration or adapted grids near the separatrix"),
    ]
    sx = (-7.6, 0.0, 7.6)
    hub = node("strategies", 0.0, 4.6, 6.4, 0.9, "\\textbf{Numerical strategies}", style="concept leaf")
    for key in cause_boxes:
        link(cause_boxes[key], hub, key, "strategies")
    strat_boxes = {}
    for (key, title, sub), x in zip(strategies, sx):
        strat_boxes[key] = node(key, x, 2.1, 7.0, 1.9, "\\textbf{" + title + "}\\\\{\\small " + sub + "}")
        link(hub, strat_boxes[key], "strategies", key)
    check = node("validation", 0.0, -0.6, 11.0, 1.2, "\\textbf{Validation required}\\\\{\\small actual equilibrium, "
                 "energies, mode structures, effective limits}", style="concept strong")
    for key in strat_boxes:
        link(strat_boxes[key], check, key, "validation")
    if labels:
        items.append(Label((0.0, -1.7), "Conditioning descriptors, not one stiffness parameter: a coordinate "
                           "singularity, contour conditioning, harmonic resolution, rational-surface singularities "
                           "and ODE stiffness are different problems, and no setting is prescribed",
                           "note", anchor="north", role="note"))
    return Diagram("xpoint_refinement_hierarchy", Scene(tuple(items)),
                   model={"causes": tuple(k for k, *_ in causes), "strategies": tuple(k for k, *_ in strategies),
                          "edges": tuple(edges)})
