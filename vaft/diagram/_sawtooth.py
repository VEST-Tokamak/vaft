"""A Kadomtsev-like sawtooth on an arbitrary equilibrium, as three distinct states (#1209).

``sawtooth(stage=...)``
    ``precursor``: the $m/n = 1/1$ internal-kink displacement of the core
    inside $q = 1$ (``kink_mode``'s displacement); ``reconnection``: a
    shrinking hot core pushed against the outer separatrix, where the X-point
    sits, and the growing $1/1$ island between them; ``post_crash``: nested
    surfaces again, the region inside the Kadomtsev mixing radius flattened
    (profile relaxation shown separately from topology).

Model class: reduced model. The reconnection geometry is drawn in the
straight-field-line plane $(\\rho\\cos\\theta^*, \\rho\\sin\\theta^*)$ of the
equilibrium and mapped to its real $(R, Z)$: a core of radius
$\\rho_1(1 - f)$ shifted to touch an outer separatrix of radius
$\\rho_1 + f(\\rho_\\mathrm{mix} - \\rho_1)$, $f$ the reconnected fraction,
$\\rho_\\mathrm{mix}$ from ``kadomtsev_mixing_radius``. Complete (Kadomtsev)
reconnection is the $f \\to 1$ limit, not a claim about every crash.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.stability import kadomtsev_mixing_radius

from ._chart import Chart, render_chart
from ._equations import formula_equation
from ._equilibrium_geometry import equilibrium_geometry
from ._mhd_mode import SURFACES, _S, displacement
from ._render import Diagram
from ._scene import Label, Marker, Polyline, Scene

STAGES = ("precursor", "reconnection", "post_crash")


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _mixing(geom) -> float:
    rho = np.linspace(0.0, 0.97, 400)
    return kadomtsev_mixing_radius(rho, geom.q(rho))


def _circle(center_x: float, radius: float, n: int = 241) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n)
    return np.stack([center_x + radius * np.cos(t), radius * np.sin(t)], -1)


def reconnection_geometry(rho_1: float, rho_mix: float, fraction: float) -> dict:
    """Core, outer separatrix and island of the reduced model, in the straight-field-line plane.

    Core radius $\\rho_c = \\rho_1(1 - f)$, outer separatrix radius
    $\\rho_o = \\rho_1 + f(\\rho_\\mathrm{mix} - \\rho_1)$, core centre shifted by
    $d = \\rho_o - \\rho_c$ along $\\theta^* = 0$ so that the two touch there:
    the X-point. The island function $g = (|\\mathbf p - \\mathbf d|^2 -
    \\rho_c^2)(\\rho_o^2 - |\\mathbf p|^2)$ vanishes on both circles and is
    positive in the crescent between them; its level sets are the island
    surfaces and its maximum, on $\\theta^* = \\pi$, the O-point.
    """
    rho_c = rho_1 * (1.0 - fraction)
    rho_o = rho_1 + fraction * (rho_mix - rho_1)
    d = rho_o - rho_c
    x_o_lo, x_o_hi = -rho_o, d - rho_c  # the crescent's extent on the theta* = pi side
    xs = np.linspace(x_o_lo, x_o_hi, 2001)
    g_line = ((xs - d) ** 2 - rho_c**2) * (rho_o**2 - xs**2)
    x_O = float(xs[np.argmax(g_line)])
    return {"rho_c": rho_c, "rho_o": rho_o, "shift": d, "x_point": (rho_o, 0.0), "o_point": (x_O, 0.0),
            "g_max": float(g_line.max())}


def _island_lines(geo: dict, n_levels: int = 4) -> List[np.ndarray]:
    from contourpy import LineType, contour_generator

    L = geo["rho_o"] * 1.02
    # 301 samples: nodes off the 4-decimal rounding ties (see #1063)
    x = np.linspace(-L, L, 301)
    y = np.linspace(-L, L, 301)
    X, Y = np.meshgrid(x, y)
    g = ((X - geo["shift"]) ** 2 + Y**2 - geo["rho_c"] ** 2) * (geo["rho_o"] ** 2 - X**2 - Y**2)
    inside = (X**2 + Y**2 < geo["rho_o"] ** 2) & ((X - geo["shift"]) ** 2 + Y**2 > geo["rho_c"] ** 2)
    g = np.where(inside, g, -1.0)
    gen = contour_generator(x=x, y=y, z=g, line_type=LineType.Separate)
    out = []
    for level in np.linspace(0.2, 0.85, n_levels) * geo["g_max"]:
        out += [np.asarray(line) for line in gen.lines(level) if len(line) > 3]
    return out


def _to_rz(geom, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    ts = np.arctan2(pts[:, 1], pts[:, 0])
    R, Z = geom.to_rz(rho, ts)
    return _S * np.stack([R, Z], -1)


def sawtooth(equilibrium=None, stage: str = "precursor", *, amplitude: float = 0.06,
             reconnection_fraction: float = 0.5, labels: bool = True) -> Diagram:
    r"""A reduced, Kadomtsev-like sawtooth crash in one of three states, on an equilibrium's surfaces.

    ``precursor``: the core inside $q = 1$ displaced by the $1/1$
    internal-kink envelope of ``kink_mode`` (``amplitude`` in units of the
    minor radius); nested topology is kept. ``reconnection``: a hot core of
    radius $\rho_1(1 - f)$ shifted against the outer separatrix, radius
    $\rho_1 + f(\rho_\mathrm{mix} - \rho_1)$, with the X-point where they
    touch and the $1/1$ island in the crescent between
    (``reconnection_geometry``), $f$ = ``reconnection_fraction``.
    ``post_crash``: nested surfaces with $\rho \le \rho_\mathrm{mix}$
    (``kadomtsev_mixing_radius`` on the equilibrium's $q(\rho)$) shaded and,
    inset, the temperature before and after flattening at conserved
    $\int T\rho\,d\rho$. Requires a $q = 1$ surface. Geometry in
    $(\rho = \sqrt{\psi_N}, \theta^*)$, drawn on the real $(R, Z)$.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
    if not (isinstance(amplitude, (int, float)) and 0.0 <= amplitude <= 0.15):
        raise ValueError(f"amplitude must lie in [0, 0.15], not {amplitude!r}")
    if not (isinstance(reconnection_fraction, (int, float)) and 0.0 < reconnection_fraction < 1.0):
        raise ValueError(f"reconnection_fraction must lie in (0, 1), not {reconnection_fraction!r}")
    labels = _check_labels(labels)
    geom = equilibrium_geometry(equilibrium)
    rho_1 = geom.rho_at_q(1.0)
    if rho_1 is None:
        raise ValueError(f"a sawtooth needs a q = 1 surface; this equilibrium has q from {geom.q0:.2f} "
                         f"to {geom.q_profile[-1]:.2f}")
    rho_mix = _mixing(geom)
    items: List = []
    model = {"classification": "reduced_model", "stage": stage, "rho_1": rho_1, "rho_mix": rho_mix, "geometry": geom}
    s1 = geom.surface(rho_1)
    q1_line = _S * np.stack([s1.R, s1.Z], -1)
    if stage == "precursor":
        surfaces = []
        for rho in SURFACES:
            d = displacement(geom, rho, n=1, amplitude=float(amplitude), harmonics={1: 1.0}, kind="internal",
                             rho_s=rho_1, phase=0.0)
            items.append(Polyline.of(_S * np.stack([d["R"], d["Z"]], -1),
                                     "boundary" if rho == SURFACES[-1] else "orbit electron", role="surface",
                                     closed=True))
            surfaces.append({"rho": float(rho), "xi": d["xi"], "theta_star": d["surface"].theta_star})
        items.append(Polyline.of(q1_line, "rational", role="q1", closed=True))
        model.update({"amplitude": float(amplitude), "surfaces": surfaces})
    elif stage == "reconnection":
        geo = reconnection_geometry(rho_1, rho_mix, float(reconnection_fraction))
        for rho in SURFACES:
            if rho > geo["rho_o"] + 0.02:
                s = geom.surface(rho)
                items.append(Polyline.of(_S * np.stack([s.R, s.Z], -1),
                                         "boundary" if rho == SURFACES[-1] else "orbit electron", role="surface",
                                         closed=True))
        core = [_circle(geo["shift"], k * geo["rho_c"]) for k in (0.35, 0.7, 1.0)]
        for i, c in enumerate(core):
            items.append(Polyline.of(_to_rz(geom, c), "trough" if i == 2 else "orbit electron", role="core",
                                     closed=True))
        items.append(Polyline.of(_to_rz(geom, _circle(0.0, geo["rho_o"])), "separatrix", role="outer_separatrix",
                                 closed=True))
        island = _island_lines(geo)
        for line in island:
            items.append(Polyline.of(_to_rz(geom, line), "orbit ion", role="island", closed=True))
        x_rz = _to_rz(geom, np.array([geo["x_point"]]))[0]
        o_rz = _to_rz(geom, np.array([geo["o_point"]]))[0]
        c_rz = _to_rz(geom, np.array([[geo["shift"], 0.0]]))[0]
        items += [Marker(tuple(x_rz), "x", "xpoint", role="x_point"), Marker(tuple(o_rz), "o", "opoint",
                                                                              role="o_point")]
        items.append(Polyline.of(q1_line, "rational", role="q1", closed=True))
        model.update({"fraction": float(reconnection_fraction), **geo, "island": island,
                      "x_point_rz": tuple(x_rz / _S), "o_point_rz": tuple(o_rz / _S), "core_centre_rz": tuple(c_rz / _S)})
    else:
        mix = geom.surface(min(rho_mix, 0.97))
        items.append(Polyline.of(_S * np.stack([mix.R, mix.Z], -1), "layer", role="mixing_region", closed=True))
        for rho in SURFACES:
            s = geom.surface(rho)
            items.append(Polyline.of(_S * np.stack([s.R, s.Z], -1),
                                     "boundary" if rho == SURFACES[-1] else "orbit electron", role="surface",
                                     closed=True))
        items.append(Polyline.of(q1_line, "rational", role="q1_before", closed=True))
        rho = np.linspace(0.0, 1.0, 201)
        before = (1.0 - rho**2) ** 2
        inside = rho <= rho_mix
        # conserved int T rho drho over the same samples the profile is drawn on
        flat = float(np.trapezoid(before[inside] * rho[inside], rho[inside])
                     / np.trapezoid(rho[inside], rho[inside]))
        after = np.where(inside, flat, before)
        chart = Chart(x_range=(0.0, 1.0), y_range=(0.0, 1.1))
        chart.curves.update({"before": np.stack([rho, before], -1), "after": np.stack([rho, after], -1)})
        inset = render_chart(chart, x_label="", y_label="",
                             curve_styles={"before": "approx", "after": "component real"}, region_text={},
                             x_ticks=(rho_1, rho_mix), x_tick_text=("$\\rho_1$", "$\\rho_\\mathrm{mix}$"))
        right = _S * float(np.max(geom.equilibrium.lcfs.r)) + 1.8
        origin = (right, -_S * 0.24)
        items += list(inset.transformed(0.42, origin).items)
        items += [Label((origin[0] + 0.42 * 9.0 + 0.35, origin[1]), "$\\rho$", "small label", anchor="west",
                        role="inset"),
                  Label((origin[0], origin[1] + 0.42 * 6.5 + 0.35), "$T_e$: dashed before, red after", "small label",
                        anchor="south west", role="inset")]
        model.update({"T_before": before, "T_after": after, "rho": rho, "T_flat": flat})
    if stage == "post_crash":
        items.append(Marker(tuple(_S * np.array(geom.axis)), "o", "opoint", role="axis"))
    elif stage == "precursor":  # the axis moves with the core: the innermost surface's centroid shift
        inner = geom.surface(SURFACES[0])
        d0 = displacement(geom, SURFACES[0], n=1, amplitude=float(amplitude), harmonics={1: 1.0}, kind="internal",
                          rho_s=rho_1, phase=0.0)
        shift = np.array([d0["R"].mean() - inner.R.mean(), d0["Z"].mean() - inner.Z.mean()])
        items.append(Marker(tuple(_S * (np.array(geom.axis) + shift)), "o", "opoint", role="axis"))
        model["axis_shift"] = tuple(shift)
    if labels:
        top = _S * float(np.max(geom.equilibrium.lcfs.z)) + 0.5
        x_mid = _S * geom.axis[0]
        right = _S * float(np.max(geom.equilibrium.lcfs.r)) + 0.6
        title = {"precursor": "precursor: $1/1$ internal kink inside $q = 1$",
                 "reconnection": f"reconnection: hot core against the X-point, $f = {reconnection_fraction:g}$",
                 "post_crash": "after the crash: core mixed inside $\\rho_\\mathrm{mix}$"}[stage]
        items.append(Label((x_mid, top), title, "label", anchor="south", role="title"))
        legend = {"precursor": "blue dashed: $q = 1$\\\\ nested surfaces kept",
                  "reconnection": "red: hot core (shifted)\\\\ blue: $1/1$ island, O and X\\\\ "
                                  "solid blue: outer separatrix\\\\ dashed: $q = 1$ before",
                  "post_crash": f"shaded: $\\rho \\le \\rho_\\mathrm{{mix}} = {rho_mix:.2f}$\\\\ "
                                f"dashed: $q = 1$ at $\\rho_1 = {rho_1:.2f}$\\\\ "
                                "flat at conserved $\\int T\\rho\\,d\\rho$; the step\\\\ at "
                                "$\\rho_\\mathrm{mix}$ is idealised"}[stage]
        items.append(Label((right, _S * 0.3), legend, "small label,align=left", anchor="north west", role="legend"))
        items.append(Label((x_mid, -top - 0.2), f"$\\displaystyle {formula_equation(kadomtsev_mixing_radius)}$",
                           "formula box", anchor="north", role="equations"))
        items.append(_note("Reduced, Kadomtsev-like: complete reconnection is the $f \\to 1$ limit, not a claim for "
                           "every crash; geometry in $(\\sqrt{\\psi_N}, \\theta^*)$ drawn on the real surfaces",
                           x_mid + 2.0, -top - 1.6))
    return Diagram(f"sawtooth_{stage}", Scene(tuple(items)), model=model)
