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
    """For each downward shift: the largest surface of the shifted plasma inside the limiter, its a and edge q.

    The equilibrium is moved rigidly down by the shift; the limiting surface
    is the largest $\\rho = \\sqrt{\\psi_N}$ whose shifted contour lies inside
    the limiter polygon (what is outside has been scraped off). Its half
    midplane width is the minor radius $a$, and the new edge's $q$ is the
    equilibrium's own $q$ on that surface -- scraping moves the edge inward to
    a surface of lower $q$. The cylindrical estimate at fixed $I_p$
    (``cylindrical_safety_factor_from_r_B`` of ``cylindrical_poloidal_field``)
    falls the same way, as $a^2$, and is recorded as ``q_cyl``.
    """
    from matplotlib.path import Path

    geom = default_equilibrium()
    eq = geom.equilibrium
    wall = Path(np.stack([np.asarray(eq.limiter.r, float), np.asarray(eq.limiter.z, float)], -1))
    rhos = np.round(np.linspace(0.3, 0.97, 68), 4)
    frames = []
    for shift in SHIFTS:
        fit = None
        for rho in rhos[::-1]:
            s = geom.surface(float(rho))
            pts = np.stack([s.R, s.Z - shift], -1)
            if wall.contains_points(pts, radius=-1e-4).all():
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
    moves inward to a surface of lower $q$ (the equilibrium's own profile),
    towards the MHD limit -- with $I_p$ held, the hot-VDE assumption before
    the quench, the cylindrical estimate falls as $a^2$ too. The displacement
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

        items.append(Polyline.of(cm(lcfs.R, lcfs.Z - f["shift"]), "concept band", role="scraped", closed=True))
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
            items.append(Label((x_mid, -_S * float(wall[:, 1].max()) - 0.25 - _S * 0.18),
                               f"$a = {100 * f['a']:.1f}$ cm, $q_\\mathrm{{edge}} = {f['q_edge']:.2f}$", "small label",
                               anchor="north", role="q_edge"))
    if labels:
        y0 = -_S * float(wall[:, 1].max()) - 3.0
        chain = ["vertical control lost", "drift into the wall", "edge scraped, $a$ falls",
                 "$q_\\mathrm{edge}$ falls at fixed $I_p$", "MHD, thermal quench", "current quench"]
        boxes = [box(1.4 + 3.4 * k, y0, 2.7, 1.2, text, role=f"chain:{k}", latex=True) for k, text in enumerate(chain)]
        for b in boxes:
            items += list(b.items)
        for a_, b_ in zip(boxes[:-1], boxes[1:]):
            items.append(connector(a_, b_, role="chain_edge"))
        items.append(_note("Shaded grey: scraped off; dashed: the unscraped boundary. $q_\\mathrm{edge}$: the "
                           "equilibrium's $q$ on the new edge surface -- not a named edge-current-loss model",
                           1.5 * width, y0 - 0.9))
    return Diagram("hot_vde_sequence", Scene(tuple(items)), model={"frames": data["frames"]})


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
    off = cold_vde_branches(I)
    chart = Chart(x_range=(0.0, 1.6), y_range=(-1.25, 1.25))
    below = I <= 1.0
    chart.curves.update({
        "centred_stable": np.stack([I[~below], np.zeros((~below).sum())], -1),
        "centred_unstable": np.stack([I[below], np.zeros(below.sum())], -1),
        "upper": np.stack([I[below], off[below]], -1),
        "lower": np.stack([I[below], -off[below]], -1),
        "wall": np.array([[0.0, -1.1], [1.6, -1.1]]),
    })
    scene = render_chart(chart, x_label="$I_p/I_c$", y_label="$Z_\\mathrm{eq}$",
                         curve_styles={"centred_stable": "boundary", "centred_unstable": "approx",
                                       "upper": "boundary", "lower": "boundary", "wall": "machine"},
                         region_text={}, x_ticks=(1.0,), x_tick_text=("$1$",), y_ticks=(0.0,), y_tick_text=("$0$",))
    items: List = list(scene.items)
    # the path the plasma follows as the current decays
    path_I = np.concatenate([np.linspace(1.5, 1.0, 20), np.linspace(1.0, 0.0, 60)])
    path_Z = np.concatenate([np.zeros(20), -cold_vde_branches(np.linspace(1.0, 0.0, 60))])
    path = chart.to_cm(np.stack([path_I, np.maximum(path_Z, -1.1)], -1))
    items.append(Polyline.of(path, "trough", role="trajectory"))
    for k in (8, 45):
        items.append(Arrow(tuple(path[k]), tuple(path[k + 3]), "drift", role="trajectory_direction"))
    if labels:
        wall_y = float(chart.to_cm(np.array([0.0, -1.1]))[1])
        items += [
            Label((CHART_WIDTH - 0.1, float(chart.to_cm(np.array([0.0, 0.0]))[1]) + 0.15), "centred, stable",
                  "small label", anchor="south east", role="branch"),
            Label((0.3, float(chart.to_cm(np.array([0.0, 0.0]))[1]) + 0.15), "centred, unstable",
                  "small label", anchor="south west", role="branch"),
            Label((0.2, wall_y - 0.1), "wall", "small label", anchor="north west", role="wall"),
            Label((CHART_WIDTH + 0.4, CHART_HEIGHT), "heavy: stable branches\\\\ dashed: unstable\\\\ red: the plasma "
                  "as $I_p$ decays\\\\ (right to left)", "small label,align=left", anchor="north west", role="legend"),
            _note("Schematic normal form of a current-dependent loss of vertical equilibrium; the real branches, "
                  "$I_c$ and stability come from a selected reduced model", CHART_WIDTH / 2 + 1.0, -1.4),
        ]
    return Diagram("cold_vde_bifurcation", Scene(tuple(items)), model={"I": I, "off": off, "I_crit": 1.0})


def plasma_wall_halo_current(*, labels: bool = True) -> Diagram:
    r"""Wall contact in a VDE: halo current through the scrape-off layer and the wall, eddy currents, force.

    The default Solov'ev plasma moved down onto the lower part of its
    limiter. Poloidal halo current flows along open field lines in the
    scrape-off layer, enters the wall, runs through it and returns (red
    loops); eddy currents flow toroidally in the wall ($\odot$), induced by
    the moving current; the poloidal halo current crossing the toroidal field
    $B_\phi$ ($\otimes$) pushes on the wall, $\mathbf J_\mathrm{halo}\times
    \mathbf B_\phi$ (arrows), downwards where it crosses the floor. Qualitative
    directions for one sign of $I_p$ and $B_\phi$; not a halo-current model
    (``halo_current_fraction`` and ``toroidal_peaking_factor`` characterise
    one).
    """
    labels = _check_labels(labels)
    geom = default_equilibrium()
    eq = geom.equilibrium
    wall = np.stack([np.asarray(eq.limiter.r, float), np.asarray(eq.limiter.z, float)], -1)
    shift = 0.12
    items: List = []

    def cm(R, Z):
        return np.stack([_S * np.asarray(R), _S * np.asarray(Z)], -1)

    frame = hot_vde_frames()["frames"]
    kept = next(f for f in frame if f["shift"] >= shift - 1e-9)
    shift = kept["shift"]  # the frame's own shift, so the kept surface is the one touching the wall
    s = geom.surface(kept["rho"])
    halo = geom.surface(min(0.97, kept["rho"] + 0.12))
    items.append(Polyline.of(cm(halo.R, halo.Z - shift), "layer", role="halo_region", closed=True))
    items.append(Polyline.of(cm(s.R, s.Z - shift), "concept band", role="plasma", closed=True))
    items.append(Polyline.of(cm(s.R, s.Z - shift), "boundary", role="plasma_boundary", closed=True))
    items.append(Polyline.of(cm(wall[:, 0], wall[:, 1]), "machine", role="wall", closed=True))
    lower = wall[wall[:, 1] < geom.axis[1]]
    order = np.argsort(lower[:, 0])

    def z_wall(R):
        """Height of the lower wall at major radius R."""
        return float(np.interp(R, lower[order, 0], lower[order, 1]))

    r_c = geom.axis[0]
    # halo current loops: out of the plasma edge, down through the SOL into the floor, along it, back up
    for dx in (-0.09, 0.09):
        x0 = r_c + dx
        z_edge = float(np.interp(x0, np.sort(s.R[s.Z < 0]), (s.Z - shift)[s.Z < 0][np.argsort(s.R[s.Z < 0])]))
        x1 = x0 + 0.06 * np.sign(dx)
        z_edge1 = float(np.interp(x1, np.sort(s.R[s.Z < 0]), (s.Z - shift)[s.Z < 0][np.argsort(s.R[s.Z < 0])]))
        loop = np.array([[x0, z_edge], [x0, z_wall(x0)], [x1, z_wall(x1)], [x1, z_edge1]])
        items.append(Polyline.of(cm(loop[:, 0], loop[:, 1]), "trough", role="halo_current"))
        a, b = cm(loop[0:2, 0], loop[0:2, 1])
        items.append(Arrow(tuple(a), tuple(0.5 * (a + b)), "drift", role="halo_current_direction"))
        # the J x B_phi force where the halo current runs along the floor: normal to the floor
        f0 = cm(x0 + 0.03 * np.sign(dx), z_wall(x0 + 0.03 * np.sign(dx)))
        items.append(Arrow((float(f0[0]), float(f0[1]) - 0.15), (float(f0[0]), float(f0[1]) - 0.95), "exb",
                           role="force"))
    for R in np.linspace(r_c - 0.2, r_c + 0.15, 5):
        items.append(Label(tuple(cm(R, z_wall(R)) + [0.0, -0.18]), "$\\odot$", "charge small", role="eddy_current"))
    items.append(Label(tuple(cm(r_c, geom.axis[1] - shift + 0.12)), "$\\otimes\\,B_\\phi$", "small label",
                       role="toroidal_field"))
    if labels:
        right = _S * float(wall[:, 0].max()) + 0.6
        items += [
            Label((right, _S * 0.3), "grey: plasma (shifted down)\\\\ pink: scrape-off layer, halo region\\\\ "
                  "red: poloidal halo current paths\\\\ $\\odot$: eddy currents in the wall\\\\ "
                  "black arrows: $\\mathbf{J}_\\mathrm{halo}\\times\\mathbf{B}_\\phi$ on the floor",
                  "small label,align=left", anchor="north west", role="legend"),
            Label((right, -_S * 0.12), f"$\\displaystyle {formula_equation(halo_current_fraction)}$", "formula box",
                  anchor="north west", role="equations"),
            _note("Qualitative directions for one sign of $I_p$ and $B_\\phi$; the halo fraction and toroidal "
                  "peaking characterise a halo current, they do not model it", right + 1.5, -_S * 0.62),
        ]
    return Diagram("plasma_wall_halo_current", Scene(tuple(items)), model={"shift": shift, "rho": kept["rho"]})


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
    rows = [("tau_A", "Alfv\\'en $a/v_A$", "ideal growth, no wall"),
            ("tau_wall_m1", "wall, $m = 1$: $\\tau_w/2$", "wall-mediated VDE"),
            ("tau_CQ_5eV", "L/R CQ at 5 eV", "cold VDE, halo phase"),
            ("tau_CQ_20eV", "L/R CQ at 20 eV", "")]
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
            items.append(Label((float(end[0]) + 0.15, float(end[1])), f"{name}: {t[key]:.1e} s"
                               + (f" -- {regime}" if regime else ""), "small label", anchor="west", role=f"row:{key}"))
        items.append(_note("One parameter set (medium tokamak, stainless vessel); the ordering is not universal -- "
                           "thin or thick walls, hotter or colder quenches reorder it", CHART_WIDTH / 2 + 1.0, -1.4))
    return Diagram("vde_timescales", Scene(tuple(items)), model=t)
