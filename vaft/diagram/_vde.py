"""Vertical displacement events: hot and cold VDE, halo currents and timescales (#1042).

``hot_vde_sequence``
    a still-hot plasma drifting into the wall: the limiting surface shrinks
    as the edge is scraped, and at constant $I_p$ the edge $q$ falls -- the
    route by which the displacement can trigger the thermal quench;
``cold_vde_bifurcation``
    the cold-VDE concept after the quench: as the current decays through a
    critical value the centred vertical equilibrium is lost and the plasma
    follows an off-centre branch into the wall (a labelled schematic normal
    form, not a selected model);
``plasma_wall_halo_current``
    wall contact: halo current closing through the scrape-off layer and the
    wall, eddy currents in the wall, and the $J\\times B$ force on it;
``vde_timescales``
    the Alfvenic, wall and current-quench times of one parameter set on a
    common log axis.

Hot-VDE geometry is the default Solov'ev equilibrium of
``_equilibrium_geometry`` and its own limiter; the edge $q$ is the cylindrical
estimate of ``vaft.formula.geometry``. Nothing here is a VDE model: the named
reduced models (edge-current loss, filament-plus-wall, analytic halo) are left
to be selected, as #1042 asks.
"""

from __future__ import annotations

import copy
import math
from functools import lru_cache
from typing import List

import numpy as np

from vaft.formula.constants import MI_P
from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
from vaft.formula.geometry import cylindrical_poloidal_field, cylindrical_safety_factor_from_r_B
from vaft.formula.stability import v_alfven_from_B_n_mi
from vaft.formula.startup import (
    lr_time_from_L_R,
    plasma_inductance_circular_from_R0_a_li,
    plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa,
)
from vaft.formula.vde import halo_current_fraction, thin_wall_time, wall_mode_decay_time

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._equilibrium_geometry import default_equilibrium
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: drawing scale of the cross-sections [cm per m]
_S = 9.0
#: downward shifts of the hot-VDE frames [m]
SHIFTS = (0.0, 0.08, 0.16)


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


@lru_cache(maxsize=1)
def hot_vde_frames() -> dict:
    r"""For each downward shift: the largest surface of the shifted plasma inside the limiter, its a and edge q.

    The equilibrium is moved rigidly down by the shift; the limiting surface
    is the largest $\rho = \sqrt{\psi_N}$ whose shifted contour lies inside
    the limiter polygon (what is outside has been scraped off). Its half
    midplane width is the minor radius $a$, and the new edge's $q$ is the
    frozen equilibrium's $q$ on that surface -- the edge-current-loss picture:
    the current outside it is gone, so this is an upper bound on $q_\\mathrm{edge}$
    at fixed $I_p$, which is lower still. ``q_cyl`` is the cylindrical
    estimate at fixed $I_p$ (``cylindrical_safety_factor_from_r_B`` of
    ``cylindrical_poloidal_field``), without elongation or toroidal factors:
    meaningful only as a ratio, falling as $a^2$.
    """
    geom = default_equilibrium()
    eq = geom.equilibrium
    wall = np.stack([np.asarray(eq.limiter.r, float), np.asarray(eq.limiter.z, float)], -1)
    rhos = np.round(np.linspace(0.3, 0.97, 68), 4)
    frames = []
    for shift in SHIFTS:
        fit = None
        for rho in rhos[::-1]:
            s = geom.surface(float(rho))
            pts = np.stack([s.R, s.Z - shift], -1)
            # inside, and at least 0.5 mm from the wall: explicit, not a plotting library's radius convention
            if _inside(wall, pts).all() and _distance_to_polygon(wall, pts).min() > 5e-4:
                fit = float(rho)
                break
        s = geom.surface(fit)
        a = 0.5 * float(s.R.max() - s.R.min())
        q = float(cylindrical_safety_factor_from_r_B(a, cylindrical_poloidal_field(a, float(eq.ip)), float(eq.bt0),
                                                     float(eq.r0)))
        frames.append({"shift": shift, "rho": fit, "a": a, "q_edge": float(geom.q(fit)), "q_cyl": q})
    return {"frames": frames, "geometry": geom}


def hot_vde_sequence(*, labels: bool = True) -> Diagram:
    r"""A hot VDE: the plasma drifts into the wall, is scraped, and its edge $q$ falls at constant $I_p$.

    Three frames of the default Solov'ev plasma moved down by 0, 8 and 16 cm
    inside its limiter (``hot_vde_frames``): the limiting surface is the
    largest flux surface that still fits, the scraped part shaded. The edge
    moves inward to a surface of lower $q$ (the frozen profile's; at fixed
    $I_p$, the hot-VDE assumption before the quench, lower still),
    towards the MHD limit. The displacement
    can trigger the thermal quench, not the other way round. Below, the
    causal chain.
    """
    labels = _check_labels(labels)
    data = hot_vde_frames()
    geom = data["geometry"]
    eq = geom.equilibrium
    wall = np.stack([np.asarray(eq.limiter.r, float), np.asarray(eq.limiter.z, float)], -1)
    lcfs = geom.surface(0.97)
    items: List = []
    width = _S * (float(wall[:, 0].max() - wall[:, 0].min())) + 1.2
    for i, f in enumerate(data["frames"]):
        ox = i * width - _S * float(wall[:, 0].min())

        def cm(R, Z, ox=ox):
            return np.stack([ox + _S * np.asarray(R), _S * np.asarray(Z)], -1)

        items.append(Polyline.of(cm(lcfs.R, lcfs.Z - f["shift"]), "layer", role="scraped", closed=True))
        s = geom.surface(f["rho"])
        items.append(Polyline.of(cm(s.R, s.Z - f["shift"]), "concept band", role="kept", closed=True))
        items.append(Polyline.of(cm(lcfs.R, lcfs.Z - f["shift"]), "approx", role="original_boundary", closed=True))
        for rho in np.linspace(0.3, f["rho"], 4)[:-1]:
            si = geom.surface(float(round(rho, 4)))
            items.append(Polyline.of(cm(si.R, si.Z - f["shift"]), "orbit electron", role="surface", closed=True))
        items.append(Polyline.of(cm(s.R, s.Z - f["shift"]), "boundary", role="limiting_surface", closed=True))
        items.append(Polyline.of(cm(wall[:, 0], wall[:, 1]), "machine", role="wall", closed=True))
        if i:
            c = cm(geom.axis[0], geom.axis[1])
            items.append(Arrow((float(c[0]), float(c[1]) + 0.3), (float(c[0]), float(c[1]) - 0.5), "exb",
                               role="motion"))
        if labels:
            top = _S * float(wall[:, 1].max()) + 0.25
            x_mid = ox + _S * geom.axis[0]
            items.append(Label((x_mid, top), f"$\\Delta Z = {-100 * f['shift']:.0f}$ cm" if f["shift"] else
                               "$\\Delta Z = 0$", "small label", anchor="south", role="title"))
            items.append(Label((x_mid, _S * min(float(wall[:, 1].min()), float(lcfs.Z.min()) - f["shift"]) - 0.25),
                               f"$a = {100 * f['a']:.1f}$ cm, $q_\\mathrm{{edge}} = {f['q_edge']:.2f}$", "small label",
                               anchor="north", role="q_edge"))
    if labels:
        y0 = _S * (float(lcfs.Z.min()) - SHIFTS[-1]) - 1.6
        chain = ["vertical control lost", "drift into the wall", "edge scraped, $a$ falls",
                 "$q_\\mathrm{edge}$ falls", "MHD, thermal quench", "current quench"]
        boxes = [box(1.4 + 3.4 * k, y0, 2.7, 1.2, text, role=f"chain:{k}", latex=True) for k, text in enumerate(chain)]
        for b in boxes:
            items += list(b.items)
        for a_, b_ in zip(boxes[:-1], boxes[1:]):
            items.append(connector(a_, b_, role="chain_edge"))
        items.append(_note("Pink: scraped off (outside the wall); dashed: the unscraped boundary. $q_\\mathrm{edge}$: the "
                           "frozen profile's $q$ on the new edge (edge current lost); at fixed $I_p$ it falls further",
                           1.5 * width, y0 - 0.9))
    # a copy: hot_vde_frames is lru_cached, and a caller's edit must not reach the next build
    return Diagram("hot_vde_sequence", Scene(tuple(items)), model={"frames": copy.deepcopy(data["frames"])})


def cold_vde_branches(I, I_crit: float = 1.0, Z_s: float = 1.0):
    """Schematic pitchfork: centred equilibrium for $I > I_c$; off-centre $\\pm Z_s\\sqrt{(I_c - I)/I_c}$ below."""
    I = np.asarray(I, dtype=float)
    off = Z_s * np.sqrt(np.clip((I_crit - I) / I_crit, 0.0, None))
    return off


def cold_vde_bifurcation(*, labels: bool = True) -> Diagram:
    r"""Cold VDE as the loss of a centred vertical equilibrium while the current decays (schematic).

    After the thermal quench the current decays; below a critical current
    $I_c$ the centred ($Z = 0$) equilibrium stops being stable and two
    off-centre branches $Z = \pm Z_s\sqrt{(I_c - I)/I_c}$ appear
    (``cold_vde_branches``). The plasma, following the decaying current from
    right to left, leaves along one of them into the wall. This is the
    normal form of a symmetric pitchfork drawn to explain the concept; the
    branch structure, $I_c$ and the stability of each branch are properties
    of a selected reduced model (e.g. a current filament with a conducting
    wall), which is not chosen here.
    """
    labels = _check_labels(labels)
    I = np.linspace(0.0, 1.6, 321)
    off = cold_vde_branches(I, 1.0, 1.3)
    chart = Chart(x_range=(0.0, 1.6), y_range=(-1.25, 1.25))
    below = I <= 1.0
    chart.curves.update({
        "centred_stable": np.stack([I[~below], np.zeros((~below).sum())], -1),
        "centred_unstable": np.stack([I[below], np.zeros(below.sum())], -1),
        # the branches end at the walls, |Z| = 1.1
        "upper": np.stack([I[below & (off <= 1.1)], off[below & (off <= 1.1)]], -1),
        "lower": np.stack([I[below & (off <= 1.1)], -off[below & (off <= 1.1)]], -1),
        "wall": np.array([[0.0, -1.1], [1.6, -1.1]]),
        "wall_upper": np.array([[0.0, 1.1], [1.6, 1.1]]),
    })
    scene = render_chart(chart, x_label="$I_p/I_c$", y_label="$Z_\\mathrm{eq}$",
                         curve_styles={"centred_stable": "boundary", "centred_unstable": "approx",
                                       "upper": "boundary", "lower": "boundary", "wall": "machine",
                                       "wall_upper": "machine"},
                         region_text={}, x_ticks=(1.0,), x_tick_text=("$1$",), y_ticks=(0.0,), y_tick_text=("$0$",))
    items: List = list(scene.items)
    # the path the plasma follows as the current decays
    path_I = np.concatenate([np.linspace(1.5, 1.0, 20), np.linspace(1.0, 0.0, 60)])
    path_Z = np.concatenate([np.zeros(20), -cold_vde_branches(np.linspace(1.0, 0.0, 60), 1.0, 1.3)])
    keep = path_Z >= -1.1  # the plasma stops at the wall
    path = chart.to_cm(np.stack([path_I[keep], path_Z[keep]], -1))
    items.append(Polyline.of(path, "trough", role="trajectory"))
    for k in (8, min(40, len(path) - 4)):
        items.append(Arrow(tuple(path[k]), tuple(path[k + 3]), "drift", role="trajectory_direction"))
    if labels:
        wall_y = float(chart.to_cm(np.array([0.0, -1.1]))[1])
        items += [
            Label((CHART_WIDTH - 0.1, float(chart.to_cm(np.array([0.0, 0.0]))[1]) + 0.15), "centred (assumed stable)",
                  "small label", anchor="south east", role="branch"),
            Label((0.3, float(chart.to_cm(np.array([0.0, 0.0]))[1]) + 0.15), "centred (assumed unstable)",
                  "small label", anchor="south west", role="branch"),
            Label((0.2, wall_y + 0.08), "wall", "small label", anchor="south west", role="wall"),
            Label((CHART_WIDTH + 0.4, CHART_HEIGHT), "heavy: branches assumed stable\\\\ dashed: assumed unstable\\\\ red: the plasma "
                  "as $I_p$ decays\\\\ (right to left), to the wall", "small label,align=left", anchor="north west", role="legend"),
            _note("Schematic normal form of a current-dependent loss of vertical equilibrium; the real branches, "
                  "$I_c$ and stability come from a selected reduced model", CHART_WIDTH / 2 + 1.0, -1.4),
        ]
    return Diagram("cold_vde_bifurcation", Scene(tuple(items)), model={"I": I, "off": off, "I_crit": 1.0})


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Point in polygon by ray casting (even-odd), plain numpy: independent of a plotting library's version."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 != y0, y1 - y0, 1.0) + x0)
    return (np.count_nonzero(crosses, axis=1) % 2) == 1


def _distance_to_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to the polygon's edges."""
    a = poly[None, :, :]
    b = np.roll(poly, -1, axis=0)[None, :, :]
    p = pts[:, None, :]
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-300), 0.0, 1.0)
    return np.min(np.linalg.norm(p - (a + t[..., None] * ab), axis=-1), axis=1)


#: toroidal field direction drawn: +phi, into the page for (R right, Z up) with (R, phi, Z) right-handed
B_PHI_SIGN = +1


def halo_circuit() -> dict:
    """One halo-current circuit at wall contact, oriented so that $J\\times B_\\phi$ pushes the wall outward.

    The 16 cm hot-VDE frame's limiting surface touches the limiter; the
    circuit leaves the plasma edge at one wall point, runs through the wall
    along it to a second point about 18 cm away, and returns along the plasma
    edge (drawn 2 cm inside it, in the halo region) to close.
    Its sense is chosen so that the force $\\mathbf J\\times\\mathbf B_\\phi$ on the
    wall segment, with $B_\\phi$ along $+\\hat\\phi$ (into the page), points out
    of the plasma -- the halo load pushes the wall away, as measured.
    """
    data = hot_vde_frames()
    geom = data["geometry"]
    eq = geom.equilibrium
    frame = data["frames"][-1]
    wall = np.stack([np.asarray(eq.limiter.r, float), np.asarray(eq.limiter.z, float)], -1)
    s = geom.surface(frame["rho"])
    edge = np.stack([s.R, s.Z - frame["shift"]], -1)
    d = _distance_to_polygon(wall, edge)
    touch = edge[int(np.argmin(d))]
    iw = int(np.argmin(np.linalg.norm(wall - touch, axis=1)))
    # wall points about 9 cm either side of the contact, along the wall
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wall, axis=0), axis=1))])

    def along(step):
        target = arc[iw] + step
        return int(np.argmin(np.abs(arc - target)))

    ia, ib = along(-0.09), along(+0.09)
    lo, hi = sorted((ia, ib))
    path = wall[lo:hi + 1]
    tangent = path[-1] - path[0]
    normal_out = wall[iw] - np.array(geom.axis)  # from the plasma towards the wall at the contact
    normal_out = normal_out / np.linalg.norm(normal_out)
    # force on the wall current: J x B with J along the tangent, B = B_phi phi_hat; R x phi = Z, Z x phi = -R
    force = B_PHI_SIGN * np.array([-tangent[1], tangent[0]])
    if np.dot(force, normal_out) < 0.0:
        path = path[::-1]
        tangent = -tangent
        force = -force
    ends = [path[0], path[-1]]
    i_feet = [int(np.argmin(np.linalg.norm(edge - e, axis=1))) for e in ends]
    # the return leg: back along the plasma edge, 2 cm inside it, over the shorter arc between the feet
    n = len(edge)
    fwd = (i_feet[0] - i_feet[1]) % n
    idx = [(i_feet[1] + k) % n for k in range(fwd + 1)] if fwd <= n - fwd else \
        [(i_feet[1] - k) % n for k in range(n - fwd + 1)]
    axis = np.array(geom.axis) - np.array([0.0, frame["shift"]])
    inner = edge[idx] + 0.02 * (axis - edge[idx]) / np.linalg.norm(axis - edge[idx], axis=1)[:, None]
    circuit = np.concatenate([inner[-1:], path, inner])
    return {"circuit": circuit, "wall_path": path, "wall_tangent": tangent / np.linalg.norm(tangent),
            "force": force / np.linalg.norm(force), "normal_out": normal_out, "contact": wall[iw],
            "edge": edge, "frame": frame, "wall": wall}


def plasma_wall_halo_current(*, labels: bool = True) -> Diagram:
    r"""Wall contact in a VDE: one halo-current circuit, the eddy currents, and the force on the wall.

    The 16 cm hot-VDE frame, its limiting surface touching the lower
    outboard limiter. One poloidal halo circuit (``halo_circuit``) leaves the
    plasma edge into the scrape-off layer, runs through the wall and returns;
    its sense is set so that $\mathbf J_\mathrm{halo}\times\mathbf B_\phi$ on
    the wall segment -- with $B_\phi$ into the page, $\otimes$ -- points out of
    the plasma, the measured sense of the halo load. $I_p$ is drawn out of the
    page ($\odot$); in the current-quench phase shown, the wall's eddy
    currents run parallel to it ($\odot$). Not a halo-current model:
    ``halo_current_fraction`` and ``toroidal_peaking_factor`` characterise one.
    """
    labels = _check_labels(labels)
    h = halo_circuit()
    geom = hot_vde_frames()["geometry"]
    frame = h["frame"]
    items: List = []

    def cm(pts):
        return _S * np.asarray(pts, dtype=float)

    halo = geom.surface(min(0.97, frame["rho"] + 0.12))
    items.append(Polyline.of(cm(np.stack([halo.R, halo.Z - frame["shift"]], -1)), "layer", role="halo_region",
                             closed=True))
    items.append(Polyline.of(cm(h["edge"]), "concept band", role="plasma", closed=True))
    items.append(Polyline.of(cm(h["edge"]), "boundary", role="plasma_boundary", closed=True))
    items.append(Polyline.of(cm(h["wall"]), "machine", role="wall", closed=True))
    items.append(Polyline.of(cm(h["circuit"]), "trough", role="halo_current"))
    c = h["circuit"]
    n_wall = 1 + len(h["wall_path"])
    for a, b in ((c[n_wall // 2 - 1], c[n_wall // 2]), (c[n_wall + (len(c) - n_wall) // 2 - 1],
                                                         c[n_wall + (len(c) - n_wall) // 2])):
        mid = 0.5 * (a + b)
        step = 0.35 * (b - a) / max(np.linalg.norm(b - a), 1e-12) / _S
        items.append(Arrow(tuple(cm(mid - step)), tuple(cm(mid + step)), "drift", role="halo_current_direction"))
    mid = c[n_wall // 2]
    items.append(Arrow(tuple(cm(mid) + 0.2 * h["force"]), tuple(cm(mid) + 1.1 * h["force"]), "exb", role="force"))
    axis = np.array(geom.axis) - np.array([0.0, frame["shift"]])
    items.append(Label(tuple(cm(axis)), "$\\odot\\,I_p$", "small label", role="plasma_current"))
    items.append(Label(tuple(cm(axis + np.array([0.0, 0.08]))), "$\\otimes\\,B_\\phi$", "small label",
                       role="toroidal_field"))
    for k in (-3, 3):
        p = h["wall"][(int(np.argmin(np.linalg.norm(h["wall"] - h["contact"], axis=1))) + 12 * k) % len(h["wall"])]
        items.append(Label(tuple(cm(p)), "$\\odot$", "charge small", role="eddy_current"))
    if labels:
        right = _S * float(h["wall"][:, 0].max()) + 0.6
        items += [
            Label((right, _S * 0.3), "grey: plasma (16 cm down, touching)\\\\ pink: scrape-off layer\\\\ "
                  "red: one poloidal halo circuit\\\\ $\\odot$ on the wall: eddy currents,\\\\ \\quad parallel to "
                  "$I_p$ (current-quench phase)\\\\ black arrow: $\\mathbf{J}_\\mathrm{halo}\\times\\mathbf{B}_\\phi$ "
                  "on the wall", "small label,align=left", anchor="north west", role="legend"),
            Label((right, -_S * 0.1), f"$\\displaystyle {formula_equation(halo_current_fraction)}$", "formula box",
                  anchor="north west", role="equations"),
            _note("Directions for $I_p$ out of and $B_\\phi$ into the page; the circuit's sense is chosen so the "
                  "force on the wall points away from the plasma. Characterisation, not a halo model",
                  right + 1.0, -_S * 0.5),
        ]
    return Diagram("plasma_wall_halo_current", Scene(tuple(items)),
                   model={"shift": frame["shift"], "rho": frame["rho"], "circuit": h["circuit"],
                          "force": h["force"], "normal_out": h["normal_out"], "wall_tangent": h["wall_tangent"]})


#: one parameter set for the timescales: a generic medium tokamak with a stainless-steel vessel
TIMESCALE_PARAMETERS = {"R0": 1.7, "a": 0.5, "kappa": 1.6, "l_i": 1.0, "B0": 2.0, "n_i": 5e19, "m_i": 2 * MI_P,
                        "sigma_wall": 1.4e6, "d_wall": 0.02, "b_wall": 0.8, "Z_eff": 1.5, "ln_Lambda": 12.0}


def vde_timescales_table() -> dict:
    """$\\tau_A = a/v_A$, the $m = 1$ wall time, and the L/R current-quench time at 5 and 20 eV [s]."""
    p = TIMESCALE_PARAMETERS
    tau_A = p["a"] / float(v_alfven_from_B_n_mi(p["B0"], p["n_i"], p["m_i"]))
    tau_w = float(thin_wall_time(p["sigma_wall"], p["d_wall"], p["b_wall"]))
    tau_1 = float(wall_mode_decay_time(tau_w, 1))
    L_p = float(plasma_inductance_circular_from_R0_a_li(p["R0"], p["a"], p["l_i"], p["kappa"]))
    out = {"tau_A": tau_A, "tau_w": tau_w, "tau_wall_m1": tau_1}
    for T in (5.0, 20.0):
        eta = float(spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(T, p["Z_eff"], p["ln_Lambda"]))
        R_p = float(plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa(eta, p["R0"], p["a"], p["kappa"]))
        out[f"tau_CQ_{T:g}eV"] = float(lr_time_from_L_R(L_p, R_p))
    return out


def vde_timescales(*, labels: bool = True) -> Diagram:
    r"""The VDE's timescales on one log axis, for one parameter set.

    The Alfvenic time $a/v_A$ (``v_alfven_from_B_n_mi``) sets ideal vertical
    growth without a wall; the $m = 1$ eddy time of a thin stainless vessel
    (``thin_wall_time``, ``wall_mode_decay_time``) slows it to a wall-mediated
    VDE; the L/R current-quench time at 5 and 20 eV (Spitzer resistivity,
    ``startup`` circuit) paces a cold VDE and the halo phase. One generic
    medium tokamak ($R_0 = 1.7$ m, 2 T, $5\times10^{19}$ m$^{-3}$, 2 cm steel
    at 0.8 m): the ordering is this set's, not universal.
    """
    labels = _check_labels(labels)
    t = vde_timescales_table()
    rows = [("tau_A", "Alfv\\'en $a/v_A$", "sets ideal growth without a wall"),
            ("tau_CQ_5eV", "L/R CQ at 5 eV", "paces a cold VDE and the halo phase"),
            ("tau_wall_m1", "$m = 1$ wall eddy time $\\tau_w/2$", "VDE growth $\\sim$ this / stability margin"),
            ("tau_CQ_20eV", "L/R CQ at 20 eV", "a warmer, slower quench")]
    lo, hi = -8.0, 0.0
    chart = Chart(x_range=(lo, hi), y_range=(0.0, len(rows) + 0.5))
    scene = render_chart(chart, x_label="$\\log_{10}(\\tau/\\mathrm{s})$", y_label="", curve_styles={},
                         region_text={}, x_ticks=tuple(float(v) for v in range(-8, 1, 2)),
                         x_tick_text=tuple(f"${v}$" for v in range(-8, 1, 2)))
    items: List = list(scene.items)
    for k, (key, _name, _regime) in enumerate(rows):
        at = chart.to_cm(np.array([math.log10(t[key]), len(rows) - k]))
        items.append(Marker((float(at[0]), float(at[1])), "o", "opoint", role=f"tau:{key}"))
        items.append(Polyline.of([(float(at[0]), 0.0), (float(at[0]), float(at[1]))], "approx", role=f"tau:{key}"))
    if labels:
        for k, (key, name, regime) in enumerate(rows):
            end = chart.to_cm(np.array([math.log10(t[key]), len(rows) - k]))
            mant, expo = f"{t[key]:.1e}".split("e")
            items.append(Label((float(end[0]) + 0.15, float(end[1])),
                               f"{name}: ${mant}\\times10^{{{int(expo)}}}$ s -- {regime}", "small label",
                               anchor="west", role=f"row:{key}"))
        items.append(_note("One parameter set (medium tokamak, stainless vessel), ordered by $\\tau$; not universal. "
                           "The VDE growth and halo-contact times need a selected model (deferred)", CHART_WIDTH / 2 + 1.0,
                           -1.4))
    return Diagram("vde_timescales", Scene(tuple(items)), model=t)
