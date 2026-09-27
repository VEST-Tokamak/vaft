"""Straight-field-line coordinates: one equilibrium, many angles (#1074).

``sfl_coordinate_grids``
    PEST, Boozer, Hamada and equal-arc constant-angle lines on the same
    shaped, low-aspect-ratio equilibrium: the surfaces are identical, only
    the poloidal angle differs;
``sfl_coordinate_taxonomy``
    the family tree -- flux coordinates, the straight-field-line condition
    and the freedom it leaves, named and generalised choices, derived
    computational coordinates -- with COCOS as a separate convention layer;
``sfl_fourier_convergence``
    one physical perturbation has different poloidal spectra in each angle.

Every angle is ``vaft.formula.generalized_straight_field_line_angle`` on
``miller_surface`` surfaces with $\\psi \\propto r^2$ and $B_\\phi = B_0R_0/R$;
the equilibrium is schematic (no Grad--Shafranov solve), which changes
neither the construction nor the comparison.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List

import numpy as np

from vaft.formula.equilibrium import generalized_straight_field_line_angle, miller_surface, vacuum_toroidal_field

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import band, box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the shaped, low-aspect-ratio equilibrium the grids share
_R0, _A, _KAPPA, _DELTA, _B0 = 1.7, 1.0, 2.0, 0.45, 1.0
#: poloidal field scale: psi = _PSI_SCALE * r^2
_PSI_SCALE = 0.25
#: (p_Bp, p_B, p_R) of the named members of the generalised family
COORDINATES = {"PEST": (0.0, 0.0, 2.0), "Boozer": (0.0, 2.0, 0.0), "Hamada": (0.0, 0.0, 0.0),
               "equal-arc": (1.0, 0.0, 0.0)}
_N_THETA = 721
_CM = 1.25


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _section(r, theta):
    return miller_surface(r, theta, _R0, _KAPPA, _DELTA * r / _A)


@lru_cache(maxsize=64)
def _surface(r: float) -> Dict[str, np.ndarray]:
    """Geometry, fields and every named straight-field-line angle on one surface."""
    theta = np.linspace(0.0, 2.0 * math.pi, _N_THETA)
    h = 1e-6
    R, Z = _section(r, theta)
    dR_dr = (_section(r + h, theta)[0] - _section(r - h, theta)[0]) / (2 * h)
    dZ_dr = (_section(r + h, theta)[1] - _section(r - h, theta)[1]) / (2 * h)
    dR_dt = (_section(r, theta + h)[0] - _section(r, theta - h)[0]) / (2 * h)
    dZ_dt = (_section(r, theta + h)[1] - _section(r, theta - h)[1]) / (2 * h)
    jac2d = dR_dr * dZ_dt - dR_dt * dZ_dr
    jacobian = R * jac2d
    grad_r = np.hypot(dR_dt, dZ_dt) / np.abs(jac2d)
    B_p = 2.0 * _PSI_SCALE * r * grad_r / R
    B_phi = vacuum_toroidal_field(_B0, _R0, R)
    B = np.hypot(B_p, B_phi)
    angles = {name: generalized_straight_field_line_angle(theta, jacobian, R, B_p, B, *powers)
              for name, powers in COORDINATES.items()}
    return {"theta": theta, "R": R, "Z": Z, "B_p": B_p, "B": B, "jacobian": jacobian, "angles": angles}


def _angle_lines(name: str, n_lines: int = 16, radii=None) -> List[np.ndarray]:
    """Constant-angle lines of one coordinate, each through all the surfaces."""
    radii = np.linspace(0.04, 1.0, 40) if radii is None else radii
    lines = []
    for k in range(n_lines):
        target = 2.0 * math.pi * k / n_lines
        pts = []
        for r in radii:
            s = _surface(float(r))
            th = np.interp(target, s["angles"][name], s["theta"])
            pts.append(np.array(_section(float(r), th)))
        lines.append(np.array(pts))
    return lines


def _xy(points, x0: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.stack([(points[..., 0] - _R0) * _CM + x0, points[..., 1] * _CM], axis=-1)


# ---------------------------------------------------------------------------
# the grids
# ---------------------------------------------------------------------------


def sfl_coordinate_grids(*, labels: bool = True) -> Diagram:
    r"""PEST, Boozer, Hamada and equal-arc on the same equilibrium: only the angle changes.

    Each panel draws the same nested surfaces ($R_0/a = 1.7$, $\kappa = 2$,
    $\delta = 0.45$) and sixteen lines of constant poloidal angle in one
    member of the generalised straight-field-line family
    (``generalized_straight_field_line_angle``). Field lines are straight in
    every one of them; what differs is how the angle is distributed.
    """
    labels = _check_labels(labels)
    radii = (0.25, 0.5, 0.75, 1.0)
    gap = 3.1 * _A * _CM
    items: List = []
    for i, name in enumerate(COORDINATES):
        x0 = i * gap
        theta = np.linspace(0.0, 2.0 * math.pi, 241)
        for r in radii:
            R, Z = _section(r, theta)
            items.append(Polyline.of(_xy(np.stack([R, Z], -1), x0), "lcfs" if r == 1.0 else "surface",
                                     role=f"surface:{name}", closed=True))
        for line in _angle_lines(name):
            items.append(Polyline.of(_xy(line, x0), "field line", role=f"grid:{name}"))
        items.append(Marker((x0, 0.0), "o", "opoint", role=f"surface:{name}"))
        if labels:
            p = COORDINATES[name]
            items.append(Label((x0, -_KAPPA * _A * _CM - 0.3),
                               f"{name}\\\\ $(p_{{Bp}}, p_B, p_R) = ({p[0]:g}, {p[1]:g}, {p[2]:g})$",
                               "small label,align=center", anchor="north", role=f"title:{name}"))
    if labels:
        items += [
            Label((1.5 * gap, -_KAPPA * _A * _CM - 1.5),
                  f"$\\displaystyle {formula_equation(generalized_straight_field_line_angle)}$", "formula box",
                  anchor="north", role="equations"),
            Label((1.5 * gap, -_KAPPA * _A * _CM - 3.0),
                  "Same surfaces in every panel; $R_0/a = 1.7$, $\\kappa = 2$, $\\delta = 0.45$; schematic $\\psi \\propto r^2$",
                  "note", anchor="north", role="note"),
        ]
    model = {"coordinates": dict(COORDINATES), "radii": radii}
    return Diagram("sfl_coordinate_grids", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# the taxonomy
# ---------------------------------------------------------------------------


def sfl_coordinate_taxonomy(*, labels: bool = True) -> Diagram:
    r"""Flux coordinates, the straight-field-line condition, and what each named choice fixes.

    The straight-field-line condition $d\zeta/d\theta = q(\psi)$ leaves a
    freedom that PEST, Boozer, Hamada and equal-arc fix in different ways
    (the generalised family); canonical and generalised-Boozer coordinates
    branch further. Clebsch field-line labels and the field-aligned, ballooning
    and flux-tube coordinates are derived from them. COCOS is a separate
    layer -- orientation and sign conventions -- across all of it, not a
    coordinate choice. Nested-surface coordinates fail at separatrices,
    islands and stochastic regions.
    """
    labels = _check_labels(labels)
    W, H = 4.3, 1.15
    items: List = []
    nodes = {
        "flux": box(0.0, 0.0, 5.2, H, "magnetic flux coordinates\\\\ $(\\psi, \\theta, \\zeta)$", role="node:flux",
                    latex=True),
        "sfl": box(0.0, -2.1, 6.4, 1.35, "straight-field-line condition\\\\ $d\\zeta/d\\theta = q(\\psi)$: a family, "
                   "not one system", role="node:sfl", latex=True),
        "pest": box(-7.2, -4.6, W, H, "PEST\\\\ keeps geometric $\\phi$", role="node:pest", latex=True),
        "boozer": box(-2.4, -4.6, W, H, "Boozer\\\\ $\\mathcal{J} \\propto B^{-2}$", role="node:boozer", latex=True),
        "hamada": box(2.4, -4.6, W, H, "Hamada\\\\ $\\mathbf{B}$ and $\\mathbf{J}$ lines straight",
                      role="node:hamada", latex=True),
        "equal_arc": box(7.2, -4.6, W, H, "equal-arc\\\\ even arc sampling", role="node:equal_arc",
                         latex=True),
        "gen_boozer": box(-2.4, -7.7, W, H, "generalised Boozer", role="node:generalized_boozer"),
        "canonical": box(-7.2, -7.7, W, H, "canonical SFL\\\\ Hamiltonian structure", role="node:canonical",
                         latex=True),
        "clebsch": box(4.8, -7.7, W + 0.4, H, "Clebsch labels\\\\ $\\mathbf{B} = \\nabla\\alpha\\times\\nabla\\psi$",
                       role="node:clebsch", latex=True),
        "aligned": box(4.8, -9.8, W + 0.4, 1.35, "field-aligned, ballooning,\\\\ flux-tube, X-point-adapted",
                       role="node:derived", latex=True),
    }
    items += band(-9.8, 9.8, -5.9, -3.8, role="family")
    items.append(Label((-9.65, -5.85), "generalised family $\\mathcal{J} \\propto R^{p_R}/(B_p^{p_{Bp}}B^{p_B})$",
                       "concept band label", anchor="south west", role="family"))
    for b in nodes.values():
        items += list(b.items)
    for a, b in (("flux", "sfl"), ("sfl", "pest"), ("sfl", "boozer"), ("sfl", "hamada"), ("sfl", "equal_arc"),
                 ("boozer", "gen_boozer"), ("clebsch", "aligned")):
        items.append(connector(nodes[a], nodes[b], role=f"edge:{a}->{b}"))
    # canonical SFL and Clebsch labels come from the family as a whole, not from one member:
    # their arrows leave the band between the member boxes
    for name, gap_x in (("canonical", -4.8), ("clebsch", 4.8)):
        n = nodes[name]
        items.append(Arrow((gap_x, -5.9), (n.x + (0.3 if n.x < gap_x else -0.3 if n.x > gap_x else 0.0),
                                           n.y + 0.5 * n.height + 0.08), "connector", role=f"edge:family->{name}"))
    # COCOS: a convention layer across the whole tree, drawn beside it
    cocos = box(12.6, -3.3, 4.2, 5.6, "COCOS\\\\[3pt] signs and orientation of $\\psi$, $q$, $\\phi$, $\\theta$, "
                "$I_p$, $B_\\phi$\\\\[3pt] across all nodes: a convention, not a coordinate", role="node:cocos",
                latex=True)
    items += list(cocos.items)
    if labels:
        items += [
            Label((0.0, -10.8), "Nested-surface coordinates are singular at separatrices and X-points and fail in islands "
                  "and stochastic regions", "note", anchor="north", role="note"),
        ]
    model = {"nodes": tuple(nodes), "family": tuple(COORDINATES)}
    return Diagram("sfl_coordinate_taxonomy", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# Fourier representation
# ---------------------------------------------------------------------------

#: the surface and the perturbation of the spectrum comparison
_R_SPECTRUM, _WIDTH = 0.8, 0.35
_M_MAX = 40


def perturbation_spectra(centre: float = 0.0) -> Dict[str, Dict]:
    """$|f_m|$ of one physical perturbation in each angle, and the harmonics that hold 99 % of its power.

    The perturbation is fixed in space and axisymmetric ($n = 0$): a Gaussian
    in the geometric polar angle $\\vartheta$ from the magnetic axis,
    $f = \\exp[-((\\vartheta - \\vartheta_0)/w)^2]$, centred outboard ($\\vartheta_0 = 0$)
    or anywhere else. It is sampled on an even grid of each coordinate's angle
    and transformed. For $n \\ne 0$ the toroidal shift $\\nu$ of every member
    but PEST would add a factor $e^{-in\\nu}$ and couple harmonics further; that
    is not included.
    """
    s = _surface(_R_SPECTRUM)
    out = {}
    for name in COORDINATES:
        grid = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
        th = np.interp(grid, s["angles"][name], s["theta"])
        R, Z = _section(_R_SPECTRUM, th)
        polar = np.arctan2(Z, R - _R0)
        offset = np.angle(np.exp(1j * (polar - centre)))
        f = np.exp(-(offset / _WIDTH) ** 2)
        amp = np.abs(np.fft.rfft(f)) / len(grid)
        power = amp ** 2
        power[1:] *= 2.0
        cum = np.cumsum(power) / power.sum()
        out[name] = {"amplitude": amp[: _M_MAX + 1], "m99": int(np.searchsorted(cum, 0.99))}
    return out


def sfl_fourier_convergence(*, labels: bool = True) -> Diagram:
    r"""The same physical perturbation has a different poloidal spectrum in each straight-field-line angle.

    One outboard-localised perturbation on the $r = 0.8$ surface of the
    grids' equilibrium, expanded in $e^{im\theta}$ for PEST, Boozer, Hamada
    and equal-arc; the legend gives the number of harmonics that carry 99 %
    of its power in each. Where an angle stretches the region the
    perturbation lives in, its spectrum narrows; where it compresses it, the
    spectrum broadens -- the same structure, a different numerical cost.
    """
    labels = _check_labels(labels)
    spectra = perturbation_spectra()
    inboard = perturbation_spectra(math.pi)
    m = np.arange(_M_MAX + 1, dtype=float)
    top = max(float(v["amplitude"].max()) for v in spectra.values())
    styles = {"PEST": "orbit ion", "Boozer": "orbit electron", "Hamada": "boundary", "equal-arc": "approx"}
    keys = {"PEST": "light", "Boozer": "dark, on PEST", "Hamada": "thick", "equal-arc": "dashed"}
    # log scale on y, drawn as log10
    chart = Chart(x_range=(0.0, float(_M_MAX) + 0.5), y_range=(-5.0, math.log10(1.5 * top)))
    for name, v in spectra.items():
        chart.curves[name] = np.stack([m, np.log10(np.maximum(v["amplitude"], 1e-6))], -1)
    chart.parameters.update({name: v["m99"] for name, v in spectra.items()})
    chart.parameters["inboard"] = {name: v["m99"] for name, v in inboard.items()}
    scene = render_chart(chart, x_label="poloidal harmonic $m$", y_label="$\\log_{10}|f_m|$",
                         curve_styles=styles, region_text={}, x_ticks=(0.0, 10.0, 20.0, 30.0, 40.0),
                         y_ticks=(-4.0, -2.0))
    items: List = []
    if labels:
        # beside the chart: the steep spectra cross every corner of it
        items.append(Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.1), "$m_{99}$: outboard / inboard bump", "small label",
                           anchor="north west", role="legend"))
        for i, (name, v) in enumerate(spectra.items()):
            items.append(Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.6 - 0.45 * i),
                               f"{name} ({keys[name]}): {v['m99']} / {inboard[name]['m99']}", "small label",
                               anchor="north west", role=f"legend:{name}"))
        items.append(Label((CHART_WIDTH / 2, -1.45), "Outboard bump drawn, $n = 0$; which angle is compact depends "
                           "on where the structure sits", "note", anchor="north", role="note"))
    return Diagram("sfl_fourier_convergence", scene + Scene(tuple(items)), model=chart)
