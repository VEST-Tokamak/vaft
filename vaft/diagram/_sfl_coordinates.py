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

from vaft.formula.equilibrium import (
    generalized_straight_field_line_angle,
    miller_surface,
    sfl_toroidal_angle_shift,
    vacuum_toroidal_field,
)

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
    # q = (1/2 pi) \oint (B_phi / R) |J| / psi' d theta, psi = _PSI_SCALE r^2
    integrand = B_phi / R * np.abs(jacobian) / (2.0 * _PSI_SCALE * r)
    q = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(theta)) / (2.0 * math.pi))
    return {"theta": theta, "R": R, "Z": Z, "B_p": B_p, "B": B, "jacobian": jacobian, "angles": angles, "q": q}


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


def perturbation_spectra(centre: float = 0.0, n: int = 0) -> Dict[str, Dict]:
    """$|f_m|$ of one physical perturbation in each angle, and the harmonics that hold 99 % of its power.

    The perturbation is fixed in space and axisymmetric ($n = 0$): a Gaussian
    in the geometric polar angle $\\vartheta$ from the magnetic axis,
    $f = \\exp[-((\\vartheta - \\vartheta_0)/w)^2]$, centred outboard ($\\vartheta_0 = 0$)
    or anywhere else. It is sampled on an even grid of each coordinate's angle
    and transformed. For a toroidal mode number $n \\ne 0$ the perturbation is
    $f(\\vartheta)e^{-in\\phi}$; at fixed $\\zeta$ each member but PEST sees the
    extra factor $e^{in\\nu(\\theta)}$ of ``sfl_toroidal_angle_shift``, which
    couples poloidal harmonics, and the spectrum then runs over negative and
    positive $m$.
    """
    s = _surface(_R_SPECTRUM)
    out = {}
    if n:
        return _spectra_with_shift(s, centre, int(n))
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


def _spectra_with_shift(s, centre: float, n: int) -> Dict[str, Dict]:
    """The n != 0 spectra: the physical f(vartheta) e^{-i n phi} read at fixed zeta in each coordinate."""
    out = {}
    grid = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    for name in COORDINATES:
        th = np.interp(grid, s["angles"][name], s["theta"])
        R, Z = _section(_R_SPECTRUM, th)
        polar = np.arctan2(Z, R - _R0)
        f = np.exp(-(np.angle(np.exp(1j * (polar - centre))) / _WIDTH) ** 2)
        theta_pest = np.interp(th, s["theta"], s["angles"]["PEST"])
        nu = np.asarray(sfl_toroidal_angle_shift(s["q"], grid, theta_pest))
        coeff = np.fft.fft(f * np.exp(1j * n * nu)) / len(grid)
        m = np.fft.fftfreq(len(grid), 1.0 / len(grid)).astype(int)
        order = np.argsort(m)
        m, amp = m[order], np.abs(coeff[order])
        keep = np.abs(m) <= _M_MAX
        power = amp**2
        # as for n = 0: the smallest M with 99 % of the power in |m| <= M
        by_m = np.array([power[np.abs(m) <= M].sum() for M in range(len(grid) // 2)])
        m99 = int(np.searchsorted(by_m / power.sum(), 0.99))
        out[name] = {"m": m[keep], "amplitude": amp[keep], "m99": m99, "nu": nu}
    return out


def sfl_fourier_convergence(*, n: int = 0, labels: bool = True) -> Diagram:
    r"""The same physical perturbation has a different poloidal spectrum in each straight-field-line angle.

    One outboard-localised perturbation on the $r = 0.8$ surface of the
    grids' equilibrium, expanded in $e^{im\theta}$ for PEST, Boozer, Hamada
    and equal-arc; the legend gives the number of harmonics that carry 99 %
    of its power in each. Where an angle stretches the region the
    perturbation lives in, its spectrum narrows; where it compresses it, the
    spectrum broadens -- the same structure, a different numerical cost.
    With a toroidal mode number ``n`` $\ne 0$ the members other than PEST
    also carry the toroidal shift $\nu$ (``sfl_toroidal_angle_shift``), whose
    $e^{in\nu}$ couples harmonics: their spectra broaden with $n$ while PEST's
    does not.
    """
    labels = _check_labels(labels)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 0 <= n <= 8:
        raise ValueError(f"n must be an integer from 0 to 8, not {n!r}")
    if n:
        return _fourier_convergence_shifted(int(n), labels)
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


def _fourier_convergence_shifted(n: int, labels: bool) -> Diagram:
    spectra = perturbation_spectra(n=n)
    base = perturbation_spectra()
    top = max(float(v["amplitude"].max()) for v in spectra.values())
    styles = {"PEST": "orbit ion", "Boozer": "orbit electron", "Hamada": "boundary", "equal-arc": "approx"}
    keys = {"PEST": "light", "Boozer": "dark", "Hamada": "thick", "equal-arc": "dashed"}
    chart = Chart(x_range=(-float(_M_MAX) - 0.5, float(_M_MAX) + 0.5), y_range=(-5.0, math.log10(1.5 * top)))
    for name, v in spectra.items():
        chart.curves[name] = np.stack([v["m"].astype(float), np.log10(np.maximum(v["amplitude"], 1e-6))], -1)
    chart.parameters.update({name: v["m99"] for name, v in spectra.items()})
    chart.parameters["n"] = n
    chart.parameters["n0"] = {name: v["m99"] for name, v in base.items()}
    scene = render_chart(chart, x_label="poloidal harmonic $m$", y_label="$\\log_{10}|f_m|$",
                         curve_styles=styles, region_text={}, x_ticks=(-40.0, -20.0, 0.0, 20.0, 40.0),
                         y_ticks=(-4.0, -2.0))
    items: List = []
    if labels:
        items.append(Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.1), f"$M_{{99}}$: $n = 0$ / $n = {n}$",
                           "small label", anchor="north west", role="legend"))
        for i, (name, v) in enumerate(spectra.items()):
            items.append(Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.6 - 0.45 * i),
                               f"{name} ({keys[name]}): {base[name]['m99']} / {v['m99']}", "small label",
                               anchor="north west", role=f"legend:{name}"))
        items.append(Label((CHART_WIDTH / 2, -1.45), f"The outboard bump times $e^{{-in\\phi}}$, $n = {n}$: read at "
                           "fixed $\\zeta$, each shifted angle adds $e^{in\\nu}$; PEST has $\\nu = 0$", "note",
                           anchor="north", role="note"))
    return Diagram("sfl_fourier_convergence", scene + Scene(tuple(items)), model=chart)



# ---------------------------------------------------------------------------
# action-angle, validity near a separatrix, coordinates vs COCOS (#1074 part 2)
# ---------------------------------------------------------------------------


def field_line_action_angle(*, labels: bool = True) -> Diagram:
    r"""Straight-field-line coordinates are the action-angle variables of the field-line flow.

    Left, the chain: nested flux surfaces make the field-line flow
    integrable; its action is the flux $\psi$ and its angles make the flow
    uniform, $\theta = \theta_0 + \iota(\psi)\zeta$ -- the straight-field-line
    coordinates. Right, one field line on the $r = 0.8$ surface of the
    grids' equilibrium, over one poloidal turn: against the geometric angles
    $(\phi, \vartheta)$ it bends, against $(\zeta, \theta_\mathrm{PEST})$ it is the
    straight line of slope $\iota = 1/q$, with $q$ the surface's own.
    """
    labels = _check_labels(labels)
    s = _surface(_R_SPECTRUM)
    q = s["q"]
    theta_p = s["angles"]["PEST"]
    # along the field line d phi = q d theta_PEST, starting at the outboard midplane
    phi = q * theta_p
    R, Z = s["R"], s["Z"]
    polar = np.unwrap(np.arctan2(Z, R - _R0))
    chain = [("surfaces", "nested flux surfaces"), ("integrable", "integrable field-line flow"),
             ("action_angle", "action $\\psi$, angles: uniform flow"), ("sfl", "straight-field-line coordinates")]
    items: List = []
    boxes = {}
    for k, (role, text) in enumerate(chain):
        boxes[role] = box(0.0, 5.6 - 1.7 * k, 3.6, 0.95, text if labels else "", role=role, latex=True)
        items += boxes[role].items
    for a, b in zip(chain, chain[1:]):
        items.append(connector(boxes[a[0]], boxes[b[0]], role=f"{a[0]}->{b[0]}"))
    chart = Chart(x_range=(0.0, 2.0 * math.pi * q), y_range=(0.0, 2.0 * math.pi))
    chart.curves["geometric"] = np.stack([phi, polar - polar[0]], -1)
    chart.curves["straight"] = np.stack([phi, theta_p], -1)
    chart.labels.update({"geometric": (0.62 * 2 * math.pi * q, 0.95 * 2 * math.pi),
                         "straight": (0.62 * 2 * math.pi * q, 0.38 * 2 * math.pi)})
    chart.parameters.update({"q": q, "iota": 1.0 / q, "r": _R_SPECTRUM})
    ticks = [0.0, math.pi, 2.0 * math.pi]
    scene = render_chart(
        chart, x_label="toroidal angle along the line, $\\phi$ or $\\zeta$", y_label="poloidal angle",
        curve_styles={"geometric": "approx", "straight": "boundary"},
        region_text={"geometric": "\\small $\\vartheta(\\phi)$: geometric, bends",
                     "straight": f"\\small $\\theta_\\mathrm{{PEST}} = \\zeta/q$, $q = {q:.2f}$"} if labels else {},
        x_ticks=[0.0, math.pi * q, 2.0 * math.pi * q], x_tick_text=["$0$", "$\\pi q$", "$2\\pi q$"],
        y_ticks=ticks, y_tick_text=["$0$", "$\\pi$", "$2\\pi$"],
        note=("One field line over a poloidal turn: the same curve, bent in geometric angles, straight in "
              "action-angle (SFL) ones" if labels else ""),
    )
    items += list(scene.transformed(offset=(3.6, -0.2)).items)
    return Diagram("field_line_action_angle", Scene(tuple(items)), model=chart)


#: 1 - psi_N at which q is evaluated toward the separatrix
_EDGE_DISTANCES = np.array([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 3e-4])


@lru_cache(maxsize=2)
def separatrix_q_profiles() -> Dict[str, np.ndarray]:
    """$|q|$ approaching the last closed surface of a limited and of a single-null Solov'ev equilibrium."""
    from vaft.process.equilibrium import calculate_q_profile_from_psi, solovev_example

    from ._equilibrium_geometry import _cocos

    out = {"distance": _EDGE_DISTANCES}
    for topology in ("limited", "single_null"):
        eq = solovev_example(topology, a_parameter=0.0, resolution=257)
        q = calculate_q_profile_from_psi(eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary,
                                         1.0 - _EDGE_DISTANCES, axis_rz=eq.magnetic_axis,
                                         boundary=(eq.lcfs.r, eq.lcfs.z), cocos=_cocos(eq))
        out[topology] = np.abs(np.asarray(q, dtype=float))
    return out


def sfl_coordinate_validity(*, labels: bool = True) -> Diagram:
    r"""Where nested-surface straight-field-line coordinates hold, and where they fail.

    Left, computed: $|q|$ toward the last closed surface of a limited and of
    a single-null Solov'ev equilibrium (``calculate_q_profile_from_psi``).
    Limited, $q$ settles; diverted, $B_p \to 0$ at the X-point and $q$ grows
    as $\ln(1 - \psi_N)$ without bound -- the angle $\theta^*$, whose rate is
    $\propto 1/q$ along the line, degenerates there. Right, the three cases:
    nested closed surfaces (valid), a separatrix or X-point (singular, special
    treatment), islands or stochastic fields (no global flux surfaces; see
    ``magnetic_island`` and ``stochastic_layer``).
    """
    labels = _check_labels(labels)
    prof = separatrix_q_profiles()
    x = np.log10(prof["distance"])
    chart = Chart(x_range=(float(x.min()) - 0.1, float(x.max()) + 0.1), y_range=(2.0, 4.2))
    chart.curves["limited"] = np.stack([x, prof["limited"]], -1)
    chart.curves["single_null"] = np.stack([x, prof["single_null"]], -1)
    chart.labels.update({"limited": (-2.0, float(prof["limited"][-4]) - 0.22),
                         "single_null": (-2.6, float(prof["single_null"][-2]) + 0.25)})
    chart.parameters.update({"q_limited_edge": float(prof["limited"][-1]),
                             "q_single_null_edge": float(prof["single_null"][-1])})
    scene = render_chart(
        chart, x_label="$1 - \\psi_N$", y_label="$|q|$",
        curve_styles={"limited": "approx", "single_null": "boundary"},
        region_text={"limited": "\\small limited: settles", "single_null": "\\small single null: $\\to\\infty$"}
        if labels else {},
        x_ticks=[-3.0, -2.0, -1.0], x_tick_text=["$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
        y_ticks=[2.0, 3.0, 4.0],
        note="Solov'ev equilibria, $q$ from contour integration; the separatrix is to the left" if labels else "",
    )
    items: List = list(scene.items)
    cases = [("valid", "nested closed surfaces: standard SFL coordinates hold"),
             ("singular", "separatrix, X-point: $B_p \\to 0$, $q \\to \\infty$, special treatment"),
             ("fails", "islands, stochastic field: no global flux surfaces")]
    boxes = {}
    for k, (role, text) in enumerate(cases):
        boxes[role] = box(CHART_WIDTH + 4.4, CHART_HEIGHT - 0.9 - 2.1 * k, 4.8, 1.2, text if labels else "",
                          style="concept box" if role == "valid" else "concept leaf", role=role, latex=True)
        items += boxes[role].items
    for a, b in (("valid", "singular"), ("singular", "fails")):
        items.append(connector(boxes[a], boxes[b], role=f"{a}->{b}"))
    return Diagram("sfl_coordinate_validity", Scene(tuple(items)), model=chart)


#: coordinate choices and example COCOS indices for the orthogonal-axes diagram
_COORDINATE_AXIS = ("PEST", "Boozer", "Hamada", "equal-arc")
_COCOS_AXIS = (1, 2, 11, 13, 17)


def coordinates_vs_cocos(*, labels: bool = True) -> Diagram:
    r"""Coordinate choice and COCOS convention are independent axes, not branches of one tree.

    Horizontal, the coordinate choice -- how the geometry is parameterised
    (PEST, Boozer, Hamada, equal-arc, all straight-field-line). Vertical,
    the COCOS convention -- orientation of the angles, signs of $\psi$, $q$,
    $B_\phi$, $I_p$ and whether $\psi$ is per radian. Every point of the grid
    is a valid, different description of the same equilibrium: changing one
    axis does not change the other (Sauter and Medvedev 2013).
    """
    labels = _check_labels(labels)
    dx, dy = 2.2, 1.0
    items: List = [Arrow((0.0, 0.0), (dx * len(_COORDINATE_AXIS) + 0.4, 0.0), "chart axis", role="axes"),
                   Arrow((0.0, 0.0), (0.0, dy * len(_COCOS_AXIS) + 0.4), "chart axis", role="axes")]
    for i, name in enumerate(_COORDINATE_AXIS):
        for j, cocos in enumerate(_COCOS_AXIS):
            items.append(Marker((dx * (i + 0.6), dy * (j + 0.6)), "o", "opoint", role=f"combination:{name}:{cocos}"))
        if labels:
            items.append(Label((dx * (i + 0.6), -0.15), name, "small label", anchor="north", role="coordinate_axis"))
    if labels:
        for j, cocos in enumerate(_COCOS_AXIS):
            items.append(Label((-0.15, dy * (j + 0.6)), f"COCOS {cocos}", "small label", anchor="east",
                               role="cocos_axis"))
        items += [Label((dx * len(_COORDINATE_AXIS) / 2, -0.75),
                        "coordinate choice: how the geometry is parameterised", "small label", anchor="north",
                        role="coordinate_axis"),
                  Label((-1.8, dy * len(_COCOS_AXIS) + 0.55),
                        "COCOS convention: orientation, signs, $2\\pi$", "small label", anchor="south west",
                        role="cocos_axis"),
                  Label((dx * len(_COORDINATE_AXIS) / 2, -1.35), "Every combination is valid: coordinate system "
                        "$\\neq$ COCOS convention", "note", anchor="north", role="note")]
    return Diagram("coordinates_vs_cocos", Scene(tuple(items)),
                   model={"coordinates": _COORDINATE_AXIS, "cocos": _COCOS_AXIS})
