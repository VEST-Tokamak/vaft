"""Asymptotic orderings: the time and length scales a reduced model assumes (#1627, #1739).

``timescale_hierarchy``
    the characteristic times of one illustrative low-field spherical-tokamak
    state on a single logarithmic axis -- inverse ion and electron gyrofrequencies,
    collision time, Alfven time, current-ramp evolution, wall time, pulse
    and resistive diffusion times -- each computed by a ``vaft.formula``
    kernel, with the ratios that are ordering parameters. It is a worked
    example of evaluating a hierarchy, not an assumed one.
``ordering_contract_map``
    which model assumes which ordering, read from the contracts of
    :mod:`vaft.validation.orderings`: ordering quantities grouped by the
    scale they are taken on, models grouped by family.

Every number is :mod:`vaft.formula` (``ordering``, ``particle``,
``stability``, ``equilibrium``, ``vde``) applied to the inputs in
:data:`ILLUSTRATIVE_STATE`; none is a measurement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from vaft.formula.constants import ME, MI_P, QE
from vaft.formula.equilibrium import coulomb_logarithm_from_n_T, spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
from vaft.formula.ordering import (
    alfven_time,
    braginskii_electron_collision_time,
    evolution_time,
    inertial_length,
    resistive_diffusion_time,
    thermal_speed,
)
from vaft.formula.neoclassical import deeply_trapped_bounce_frequency, transit_frequency
from vaft.formula.particle import gyrofrequency
from vaft.formula.sol import ion_sound_speed
from vaft.formula.stability import v_alfven_from_B_n_mi
from vaft.formula.vde import thin_wall_time

from ._geometry import _check_labels
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: The illustrative state: a low-field spherical tokamak, not a measured discharge. The ions are
#: hydrogen for the mass density (v_A); Z_eff enters only the resistivity and the electron collision time.
ILLUSTRATIVE_STATE: Dict[str, float] = {
    "B": 0.1,              # field [T]
    "n": 1.0e19,           # density [m^-3]
    "T_e": 50.0,           # electron temperature [eV]
    "Z_eff": 2.0,          # [-]
    "a": 0.25,             # minor radius, the ordering length [m]
    "I_p": 1.0e5,          # plasma current [A]
    "dI_dt": 2.0e7,        # current ramp rate [A/s]
    "pulse": 0.015,        # discharge duration [s]
    "wall_sigma": 1.39e6,  # wall conductivity, stainless steel [S/m]
    "wall_d": 0.006,       # wall thickness [m]
    "wall_b": 0.6,         # wall radius [m]
    # added for the plasma-model guide (#1739): illustrative, not measured
    "T_i": 30.0,           # ion temperature [eV]
    "R0": 0.4,             # major radius [m]
    "q": 2.5,              # safety factor at mid-radius [-]
}


def _free_level(placed, x: float, up: bool, threshold: float) -> int:
    """The lowest stacking level no earlier label on the same side within ``threshold`` cm uses."""
    used = {level for (px, pup, level) in placed if pup == up and abs(px - x) < threshold}
    return min(set(range(len(used) + 1)) - used)


def timescales(state: Dict[str, float] = ILLUSTRATIVE_STATE) -> Dict[str, Tuple[str, float, bool]]:
    """Each timescale of ``state``: its symbol, its value [s] and whether it is an input rather than computed."""
    s = state
    ln_lambda = coulomb_logarithm_from_n_T(s["n"], s["T_e"])
    eta = spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(s["T_e"], s["Z_eff"], ln_lambda)
    v_A = v_alfven_from_B_n_mi(s["B"], s["n"], MI_P)
    v_ti = thermal_speed(s["T_i"], MI_P)
    epsilon = 0.5 * s["a"] / s["R0"]            # mid-radius inverse aspect ratio
    return {
        "sound": ("$a/c_s$", s["a"] / ion_sound_speed(s["T_e"], s["T_i"], MI_P), False),
        "transit": ("$\\omega_{t,i}^{-1}$", 1.0 / transit_frequency(v_ti, s["q"], s["R0"]), False),
        "bounce": ("$\\omega_{b,i}^{-1}$", 1.0 / deeply_trapped_bounce_frequency(v_ti, s["q"], s["R0"], epsilon),
                   False),
        "electron_gyration": ("$\\Omega_{ce}^{-1}$", 1.0 / abs(gyrofrequency(-QE, ME, s["B"])), False),
        "ion_gyration": ("$\\Omega_{ci}^{-1}$", 1.0 / abs(gyrofrequency(QE, MI_P, s["B"])), False),
        "alfven": ("$\\tau_A$", alfven_time(s["a"], v_A), False),
        "collision": ("$\\tau_e$", braginskii_electron_collision_time(s["n"], s["T_e"], ln_lambda, s["Z_eff"]),
                      False),
        "evolution": ("$\\tau_{evol}$", evolution_time(s["I_p"], s["dI_dt"]), False),
        "wall": ("$\\tau_w$", thin_wall_time(s["wall_sigma"], s["wall_d"], s["wall_b"]), False),
        "pulse": ("$\\tau_{pulse}$", s["pulse"], True),
        "resistive": ("$\\tau_R$", resistive_diffusion_time(s["a"], eta), False),
    }


_X0, _DECADE = -11.0, 1.75   # log10 of the axis origin [s], centimetres per decade
_DECADES = 10


def _x(t: float) -> float:
    return (math.log10(t) - _X0) * _DECADE


def _fmt(t: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("$\\mu$s", 1e-6), ("ns", 1e-9), ("ps", 1e-12)):
        if t >= scale:
            return f"{t / scale:.3g} {unit}"
    return f"{t:.2g} s"


def _sci(x: float) -> str:
    exponent = int(math.floor(math.log10(x)))
    mantissa = x / 10 ** exponent
    return f"{mantissa:.1f}\\times10^{{{exponent}}}" if exponent not in (0, 1, -1) else f"{x:.2g}"


def timescale_hierarchy(*, labels: bool = True) -> Diagram:
    r"""The timescales of one illustrative state on a logarithmic axis, with the ratios that order them.

    Inverse gyrofrequencies $1/\Omega$ (``particle.gyrofrequency``), the electron collision time
    (``ordering.braginskii_electron_collision_time``), the Alfven time over
    the minor radius, the sound crossing time ``a/c_s`` (``sol.ion_sound_speed``,
    $c_s = \\sqrt{(T_e + T_i)/m_i}$), the inverse thermal-ion transit and
    deeply-trapped bounce frequencies at $v_{ti} = \\sqrt{T_i/m_i}$
    (``neoclassical.transit_frequency``, ``deeply_trapped_bounce_frequency``;
    a period is $2\\pi$ times longer),
    the $I_p$-ramp evolution time, the thin-wall time
    (``vde.thin_wall_time``), the pulse (an input) and the resistive
    diffusion time from Spitzer resistivity. The drift precession has no
    formula kernel and $\tau_E$ needs an energy inventory the state does not
    carry; both are left out and the caption says so. The ratios below the axis --
    $S = \tau_R/\tau_A$, $\tau_{evol}/\tau_A$, $\tau_{pulse}/\tau_R$, $d_i/a$
    -- are the ordering parameters; the hierarchy is evaluated, not
    assumed.
    """
    labels = _check_labels(labels)
    times = timescales()
    items: List = []
    length = _DECADES * _DECADE
    items.append(Arrow((0.0, 0.0), (length + 0.5, 0.0), "chart axis", role="axis"))
    for k in range(_DECADES + 1):
        x = k * _DECADE
        items.append(Polyline.of([(x, -0.12), (x, 0.12)], "connector line", role="ticks"))
        items.append(Label((x, -0.25), f"$10^{{{int(_X0) + k}}}$", "ticklabel", anchor="north", role="ticks"))
    items.append(Label((length + 0.6, -0.25), "$t$ [s]", "xlabel", anchor="north west", role="axis"))
    order = sorted(times.items(), key=lambda kv: kv[1][1])
    placed, below = [], []
    for i, (key, (symbol, value, given)) in enumerate(order):
        x = _x(value)
        items.append(Marker((x, 0.0), "o", "opoint" if not given else "xpoint", role=f"time:{key}"))
        up = i % 2 == 0
        # stack labels that would collide with an earlier one on the same side: the first free level
        level = _free_level(placed, x, up, 3.4)
        placed.append((x, up, level))
        y = (0.9 + 1.25 * level) * (1 if up else -1) - (0.0 if up else 0.55)
        if not up:
            below.append(y)
        # below the axis the leader starts under the tick labels, so it never strikes through one
        items.append(Polyline.of([(x, 0.0 if up else -0.75), (x, y)], "approx", role=f"time:{key}"))
        if labels:
            text = f"{symbol}\\\\ {_fmt(value)}" + ("\\\\ (input)" if given else "")
            items.append(Label((x, y), text, "small label,align=center", anchor="south" if up else "north",
                               role=f"time:{key}"))
    s = ILLUSTRATIVE_STATE
    ratios = {
        "lundquist": times["resistive"][1] / times["alfven"][1],  # S = tau_R / tau_A at the same length
        "quasi_static": times["evolution"][1] / times["alfven"][1],
        "relaxation": times["pulse"][1] / times["resistive"][1],
        "hall": inertial_length(s["n"], MI_P) / s["a"],
    }
    lowest = min(below + [-0.75]) - 1.6   # under the deepest label (up to three lines tall)
    if labels:
        items.append(Label((0.5 * length, lowest - 0.3),
                           f"$S = \\tau_R/\\tau_A \\approx {_sci(ratios['lundquist'])}$, "
                           f"$\\tau_{{evol}}/\\tau_A \\approx {_sci(ratios['quasi_static'])}$: well separated. "
                           f"$\\tau_{{pulse}}/\\tau_R \\approx {ratios['relaxation']:.2g}$, "
                           f"$d_i/a \\approx {ratios['hall']:.2g}$: of order one, not ordered",
                           "label", anchor="north", role="ratios"))
        items.append(Label((0.5 * length, lowest - 1.1),
                           f"Illustrative low-field spherical tokamak, not a measurement: $B = {s['B']:g}$ T, "
                           f"$n = 10^{{19}}$ m$^{{-3}}$, $T_e = {s['T_e']:g}$ eV, $T_i = {s['T_i']:g}$ eV, "
                           f"$a = {s['a']:g}$ m, $R_0 = {s['R0']:g}$ m, $q = {s['q']:g}$, "
                           f"$I_p = 100$ kA ramped at $20$ MA/s ($\\tau_{{evol}} = I_p/\\dot I_p$).\\\\ "
                           "The order is evaluated for one illustrative state; it depends on species, machine, "
                           "radius, $T$, $B$, collisionality and mode frequency.\\\\ Transit and bounce are inverse "
                           "frequencies at $v_{ti} = \\sqrt{T_i/m_i}$; $c_s = \\sqrt{(T_e + T_i)/m_i}$. Drift "
                           "precession (no kernel) and $\\tau_E$ (no energy inventory) are left out.",
                           "note,align=center", anchor="north", role="note"))
    model = {"times": {k: v[1] for k, v in times.items()}, "inputs": {k for k, v in times.items() if v[2]},
             "ratios": ratios, "state": dict(ILLUSTRATIVE_STATE)}
    return Diagram("timescale_hierarchy", Scene(tuple(items)), model=model)


def lengths(state: Dict[str, float] = ILLUSTRATIVE_STATE) -> Dict[str, Tuple[str, float, bool]]:
    """Each length of ``state``: its symbol, its value [m] and whether it is an assumed scale rather than computed."""
    from vaft.formula.neoclassical import banana_width
    from vaft.formula.ordering import sound_gyroradius
    from vaft.formula.particle import larmor_radius

    s = state
    rho_e = larmor_radius(-QE, ME, thermal_speed(s["T_e"], ME), s["B"])
    rho_i = larmor_radius(QE, MI_P, thermal_speed(s["T_i"], MI_P), s["B"])
    epsilon = 0.5 * s["a"] / s["R0"]
    return {
        "rho_e": ("$\\rho_e$", rho_e, False),
        "rho_i": ("$\\rho_i$", rho_i, False),
        "rho_s": ("$\\rho_s$", sound_gyroradius(s["T_e"], MI_P, s["B"]), False),
        "d_i": ("$d_i$", inertial_length(s["n"], MI_P), False),
        "banana": ("$\\Delta_b$", banana_width(rho_i, s["q"], epsilon), False),
        "gradient": ("$L = a/2$", 0.5 * s["a"], True),
        "minor": ("$a$", s["a"], True),
        "major": ("$R_0$", s["R0"], True),
        "connection": ("$qR_0$", s["q"] * s["R0"], True),
    }


_LX0, _LDECADES, _LDECADE = -5.0, 6, 2.6   # length axis: log10 origin [m], decades, cm per decade


def spatial_scale_orderings(*, labels: bool = True) -> Diagram:
    r"""The lengths of one illustrative state on a logarithmic axis, with the ratios that order them.

    Electron and ion Larmor radii (``particle.larmor_radius`` at the thermal
    speed), the sound gyroradius (``ordering.sound_gyroradius``), the ion
    skin depth (``ordering.inertial_length``), the ion banana width
    (``neoclassical.banana_width``), an illustrative gradient length $a/2$,
    the minor radius, the connection length $qR_0$ and $R_0$, all on the
    state of :func:`timescale_hierarchy`. Below the axis, the ordering ratios
    a length fixes ($\rho_i/L$, $d_i/a$, $\Delta_b/L$) and, as brackets, the
    ones that need a mode ($k_\perp\rho$, $k_\parallel/k_\perp$,
    $\omega/\Omega_c$). The order is state-dependent.
    """
    labels = _check_labels(labels)
    values = lengths()
    items: List = []
    length = _LDECADES * _LDECADE
    items.append(Arrow((0.0, 0.0), (length + 0.5, 0.0), "chart axis", role="axis"))
    for k in range(_LDECADES + 1):
        x = k * _LDECADE
        items.append(Polyline.of([(x, -0.12), (x, 0.12)], "connector line", role="ticks"))
        items.append(Label((x, -0.25), f"$10^{{{int(_LX0) + k}}}$", "ticklabel", anchor="north", role="ticks"))
    items.append(Label((length + 0.6, -0.25), "length [m]", "xlabel", anchor="north west", role="axis"))
    order = sorted(values.items(), key=lambda kv: kv[1][1])
    placed, below = [], []
    for i, (key, (symbol, value, assumed)) in enumerate(order):
        x = (math.log10(value) - _LX0) * _LDECADE
        items.append(Marker((x, 0.0), "x" if assumed else "o", "xpoint" if assumed else "opoint",
                            role=f"length:{key}"))
        up = i % 2 == 0
        level = _free_level(placed, x, up, 2.4)
        placed.append((x, up, level))
        y = (0.9 + 1.25 * level) * (1 if up else -1) - (0.0 if up else 0.55)
        if not up:
            below.append(y)
        items.append(Polyline.of([(x, 0.0 if up else -0.75), (x, y)], "approx", role=f"length:{key}"))
        if labels:
            text = f"{symbol}\\\\ {_fmt_length(value)}"
            items.append(Label((x, y), text, "small label,align=center", anchor="south" if up else "north",
                               role=f"length:{key}"))
    lowest = min(below + [-0.75]) - 1.0   # under the deepest label (about two lines tall)
    v = {k: val[1] for k, val in values.items()}
    ratios = {"rho_i_over_L": v["rho_i"] / v["gradient"], "d_i_over_a": v["d_i"] / v["minor"],
              "banana_over_L": v["banana"] / v["gradient"]}
    if labels:
        items.append(Label((0.5 * length, lowest - 0.5),
                           f"$\\rho_i/L \\approx {ratios['rho_i_over_L']:.2g}$, "
                           f"$\\Delta_b/L \\approx {ratios['banana_over_L']:.2g}$, "
                           f"$d_i/a \\approx {ratios['d_i_over_a']:.2g}$.\\\\ "
                           "$k_\\perp\\rho$, $k_\\parallel/k_\\perp$ and $\\omega/\\Omega_c$: "
                           "[ need a mode $(k, \\omega)$ ], not a property of the state",
                           "label,align=center", anchor="north", role="ratios"))
        s = ILLUSTRATIVE_STATE
        items.append(Label((0.5 * length, lowest - 1.8),
                           f"Illustrative state, not a measurement ($B = {s['B']:g}$ T, $n = 10^{{19}}$ m$^{{-3}}$, "
                           f"$T_e = {s['T_e']:g}$ eV, $T_i = {s['T_i']:g}$ eV, $a = {s['a']:g}$ m, "
                           f"$R_0 = {s['R0']:g}$ m, $q = {s['q']:g}$ assumed). The order is state-dependent; "
                           "crosses mark assumed scales. $\\rho_s$ is cold-ion ($c_s = \\sqrt{T_e/m_i}$).",
                           "note", anchor="north", role="note"))
    model = {"lengths": v, "assumed": {k for k, val in values.items() if val[2]}, "ratios": ratios,
             "state": dict(ILLUSTRATIVE_STATE)}
    return Diagram("spatial_scale_orderings", Scene(tuple(items)), model=model)


def _fmt_length(x: float) -> str:
    for unit, scale in (("m", 1.0), ("cm", 1e-2), ("mm", 1e-3), ("$\\mu$m", 1e-6)):
        if x >= scale:
            return f"{x / scale:.3g} {unit}"
    return f"{x:.2g} m"


# ---------------------------------------------------------------------------
# which model assumes which ordering
# ---------------------------------------------------------------------------


_GROUP_TITLES = {
    "foundational": "foundational",
    "global": "global ($a$, $R_0$)",
    "profile": "equilibrium profile ($L_p$, $L_T$, $qR$)",
    "perturbation": "perturbation ($k$)",
    "layer": "inner layer ($\\delta$)",
    "time_history": "time history",
}

#: column heads and the model families they are grouped in
_MODELS = {
    "ideal_single_fluid_mhd": "ideal MHD",
    "resistive_mhd": "resistive MHD",
    "hall_mhd": "Hall MHD",
    "braginskii_two_fluid": "Braginskii two-fluid",
    "flr_small_fluid": "FLR-corrected fluid",
    "low_beta_reduced_mhd": "low-$\\beta$ reduced MHD",
    "high_beta_reduced_mhd": "high-$\\beta$ reduced MHD",
    "drift_kinetic": "drift kinetics",
    "gyrokinetic_delta_f": "$\\delta f$ gyrokinetics",
    "local_neoclassical": "local neoclassical",
    "banana_regime_neoclassical": "banana regime",
    "pfirsch_schlueter_neoclassical": 'Pfirsch--Schl\\"uter regime',
    "quasi_static_equilibrium": "equilibrium sequence",
    "resistively_relaxed_current": "relaxed current",
    "gyrokinetic_transport_separation": "GK--transport separation",
}
_FAMILIES = (
    ("fluid", ("ideal_single_fluid_mhd", "resistive_mhd", "hall_mhd", "braginskii_two_fluid", "flr_small_fluid")),
    ("reduced MHD", ("low_beta_reduced_mhd", "high_beta_reduced_mhd")),
    ("kinetic", ("drift_kinetic", "gyrokinetic_delta_f")),
    ("neoclassical", ("local_neoclassical", "banana_regime_neoclassical", "pfirsch_schlueter_neoclassical")),
    ("evolution", ("quasi_static_equilibrium", "resistively_relaxed_current", "gyrokinetic_transport_separation")),
)

_COL, _ROW, _LEFT = 1.3, 0.62, 3.6


def ordering_contract_map(*, labels: bool = True) -> Diagram:
    r"""Which reduced model assumes which ordering, read from the registered contracts.

    Rows are the ordering quantities of ``vaft.validation.orderings.ORDERING_QUANTITIES``,
    grouped by the scale they are taken on -- foundational, global,
    equilibrium profile, perturbation ($k$), inner layer ($\delta$), time
    history -- so a global $d_i/a$, a mode's $k\,d_i$ and a layer's
    $d_i/\delta$ are separate rows. Columns are the contracts of
    ``CONTRACTS``, grouped by model family; a cell says whether the model
    needs the quantity $\gg 1$ or $\ll 1$, and an empty cell means the model
    does not order it -- $k_\perp\rho_i$ in gyrokinetics, $k\,d_i$ in Hall
    MHD. Every threshold is order unity, and evaluating a state gives a
    continuous margin per cell ($\mp\log_{10}x$), not one valid/invalid flag.
    """
    from vaft.validation.orderings import CONTRACTS, GROUPS, ORDERING_QUANTITIES, SYMBOLS as _SYMBOLS

    labels = _check_labels(labels)
    columns = [name for _, names in _FAMILIES for name in names]
    if sorted(columns) != sorted(CONTRACTS) or set(_SYMBOLS) != set(ORDERING_QUANTITIES):
        raise RuntimeError("ordering_contract_map is out of step with vaft.validation.orderings")
    rows = list(ORDERING_QUANTITIES)
    items: List = []
    right = _LEFT + len(columns) * _COL

    def cx(name):
        return _LEFT + (columns.index(name) + 0.5) * _COL

    for name in columns:
        items.append(Label((cx(name) - 0.15, 0.15), _MODELS[name], "small label,rotate=55", anchor="west",
                           role=f"column:{name}"))
    y = 0.0
    row_y = {}
    for group in GROUPS:
        members = [k for k in rows if ORDERING_QUANTITIES[k].group == group]
        y -= _ROW
        items.append(Polyline.of([(0.0, y - 0.5 * _ROW), (right, y - 0.5 * _ROW), (right, y + 0.5 * _ROW),
                                  (0.0, y + 0.5 * _ROW)], "concept band", role=f"group:{group}", closed=True))
        items.append(Label((0.1, y), _GROUP_TITLES[group], "concept band label", anchor="west",
                           role=f"group:{group}"))
        for key in members:
            y -= _ROW
            row_y[key] = y
            items.append(Label((0.25, y), _SYMBOLS[key], "small label", anchor="west", role=f"row:{key}"))
    bottom = y - 0.5 * _ROW
    for (_, names) in _FAMILIES[1:]:
        x = _LEFT + columns.index(names[0]) * _COL
        items.append(Polyline.of([(x, 0.0), (x, bottom)], "im grid", role="families"))
    for family, names in _FAMILIES:
        x0 = _LEFT + columns.index(names[0]) * _COL
        items.append(Label((x0 + 0.5 * len(names) * _COL, bottom - 0.1), family, "concept band label",
                           anchor="north", role="families"))
    for name in columns:
        for a in CONTRACTS[name].assumptions:
            x, yy = cx(name), row_y[a.quantity]
            text, style = ("$\\gg 1$", "concept source") if a.ordering == "large" else ("$\\ll 1$", "concept leaf")
            items.append(Polyline.of([(x - 0.5, yy - 0.24), (x + 0.5, yy - 0.24), (x + 0.5, yy + 0.24),
                                      (x - 0.5, yy + 0.24)], style, role=f"cell:{name}:{a.quantity}", closed=True))
            items.append(Label((x, yy), text, "small label", role=f"cell:{name}:{a.quantity}"))
    if labels:
        items.append(Label((0.5 * right, bottom - 0.9),
                           "Each cell is one ordering, evaluated separately against an order-unity threshold: "
                           "a margin $m = \\mp\\log_{10}x$ per cell, not one flag.\\\\ "
                           "An empty cell is not ordered by that model: "
                           "$k_\\perp\\rho_i$ may be $O(1)$ in gyrokinetics, $k\\,d_i$ in Hall MHD, "
                           "$\\tau_{evol}/\\tau_A$ in ideal MHD (Alfvenic dynamics)",
                           "note", anchor="north", role="note"))
    model = {"rows": tuple(rows), "columns": tuple(columns),
             "cells": {(n, a.quantity): a.ordering for n in columns for a in CONTRACTS[n].assumptions}}
    return Diagram("ordering_contract_map", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# which code adapter consumes which contract (#1823)
# ---------------------------------------------------------------------------

#: display names of the adapter modes of vaft.validation.model_contracts.ADAPTER_MODES
_MODE_NAMES = {
    "efit:reconstruction": "EFIT", "chease:fixed_boundary": "CHEASE", "tokamaker:free_boundary": "TokaMaker",
    "tes:free_boundary": "TES", "nice:reconstruction": "NICE",
    "gpec:dcon_ideal": "DCON ideal", "gpec:dcon_kinetic": "DCON kinetic", "gpec:gpec_ideal": "GPEC ideal",
    "gpec:gpec_kinetic": "GPEC kinetic", "gpec:rdcon_rmatch": "RDCON + RMATCH",
    "gpec:stride": "STRIDE", "gpec:thresholds": "GPEC thresholds", "pentrc:ntv_closure": "PENTRC (NTV)",
    "gacode:neo": "NEO", "gacode:neo_analytic": "NEO analytic", "gacode:cgyro_linear": "CGYRO linear",
    "gacode:cgyro_nonlinear": "CGYRO nonlinear", "gacode:tglf_linear": "TGLF linear", "gacode:tglf": "TGLF",
    "gacode:tglf_nn": "TGLF-NN", "nubeam:fast_ions": "NUBEAM", "genray:ray_tracing": "GENRAY",
    "transp:integrated": "TRANSP", "flare:field_lines": "FLARE", "mitim:drivers": "MITIM",
}
_MODE_FAMILIES = {"equilibrium": "equilibrium", "mhd_stability": "MHD stability (GPEC suite)",
                  "neoclassical": "neoclassical", "gyrokinetic": "gyrokinetic"}
#: why a mode consumes no ordering contract, in the note
_UNCOVERED = {"nubeam:fast_ions": "test particles", "genray:ray_tracing": "ray tracing",
              "transp:integrated": "integrated", "gpec:thresholds": "unresolved layer models",
              "flare:field_lines": "field lines only", "mitim:drivers": "driver of NEO/TGLF"}
#: the words written in a cell for each role of vaft.validation.model_contracts.ROLES
_ROLE_WORDS = {"governing": "governing", "outer_solution": "outer solution", "layer_solution": "layer solution",
               "fluid_part": "fluid part", "kinetic_response": "kinetic part", "closure": "closure",
               "inherited": "inherited", "transport_coupling": "coupling"}
#: cell style per role: a governing assumption is filled, an inherited one dashed
_ROLE_STYLES = {"governing": "concept source", "inherited": "concept group"}

_ACOL, _AROW, _ALEFT = 2.6, 0.62, 3.9


def adapter_contract_map(*, labels: bool = True) -> Diagram:
    r"""Which VAFT code adapter consumes which ordering contract, and in what role.

    Rows are the modes of ``vaft.validation.model_contracts.ADAPTER_MODES``,
    grouped by family (equilibrium, MHD stability, neoclassical,
    gyrokinetic); columns are the contracts any of them consumes, in the
    model-family order of :func:`ordering_contract_map`. A cell names how the
    assumption enters the calculation -- not how much of the plasma it covers:
    it *governs* the model solved; it holds for RDCON's ideal *outer solution*
    or its resistive *layer solution*; it underlies DCON-kinetic's *fluid part* or
    drift-kinetic *kinetic part*, or PENTRC's *closure*; or a reduced model
    (TGLF) has *inherited* it from a theory it does not solve. An empty cell
    means no registered contract, not that the calculation is independent of
    that physics. Modes no contract covers are named in the note; the
    validity conditions themselves are documented in the guide, not drawn.
    """
    from vaft.validation.model_contracts import ADAPTER_MODES, ROLES

    labels = _check_labels(labels)
    covered = [m for m in ADAPTER_MODES if m.contracts()[0]]
    uncovered = [m for m in ADAPTER_MODES if not m.contracts()[0]]
    if (set(_ROLE_WORDS) != set(ROLES) or {m.key for m in ADAPTER_MODES} != set(_MODE_NAMES)
            or any(m.family not in _MODE_FAMILIES for m in covered) or {m.key for m in uncovered} != set(_UNCOVERED)):
        raise RuntimeError("adapter_contract_map is out of step with vaft.validation.model_contracts")
    consumed = {u.contract for m in covered for u in m.contracts()[0]}
    columns = [name for _, names in _FAMILIES for name in names if name in consumed]
    right = _ALEFT + len(columns) * _ACOL
    items: List = []

    def cx(name):
        return _ALEFT + (columns.index(name) + 0.5) * _ACOL

    for name in columns:
        items.append(Label((cx(name) - 0.15, 0.15), _MODELS[name], "small label,rotate=55", anchor="west",
                           role=f"column:{name}"))
    y = 0.0
    used = {}
    for family, title in _MODE_FAMILIES.items():
        members = [m for m in covered if m.family == family]
        if not members:
            continue
        y -= _AROW
        items.append(Polyline.of([(0.0, y - 0.5 * _AROW), (right, y - 0.5 * _AROW), (right, y + 0.5 * _AROW),
                                  (0.0, y + 0.5 * _AROW)], "concept band", role=f"family:{family}", closed=True))
        items.append(Label((0.1, y), title, "concept band label", anchor="west", role=f"family:{family}"))
        for mode in members:
            y -= _AROW
            items.append(Label((0.25, y), _MODE_NAMES[mode.key], "small label", anchor="west", role=f"mode:{mode.key}"))
            for use in mode.contracts()[0]:
                x = cx(use.contract)
                used[(mode.key, use.contract)] = use.role
                style = _ROLE_STYLES.get(use.role, "concept leaf")
                items.append(Polyline.of([(x - 1.22, y - 0.24), (x + 1.22, y - 0.24), (x + 1.22, y + 0.24),
                                          (x - 1.22, y + 0.24)], style, role=f"use:{mode.key}:{use.contract}",
                                         closed=True))
                items.append(Label((x, y), _ROLE_WORDS[use.role], "small label",
                                   role=f"use:{mode.key}:{use.contract}"))
    bottom = y - 0.5 * _AROW
    for i in range(1, len(columns)):
        x = _ALEFT + i * _ACOL
        items.append(Polyline.of([(x, 0.0), (x, bottom)], "im grid", role="columns"))
    if labels:
        names = [f"{_MODE_NAMES[m.key]} ({_UNCOVERED[m.key]})" for m in uncovered]
        half = (len(names) + 1) // 2
        none = ", ".join(names[:half]) + ",\\\\ " + ", ".join(names[half:])
        items.append(Label((0.5 * right, bottom - 0.2),
                           "A cell says how the assumption enters the calculation, not how much of the plasma "
                           "it covers.\\\\ Filled: governing. Dashed: inherited by a reduced model that does "
                           "not solve its parent theory.\\\\ An empty cell means no registered contract, "
                           "not physical independence.\\\\ "
                           f"No ordering contract yet: {none}",
                           "note", anchor="north", role="note"))
    model = {"rows": tuple(m.key for m in covered), "columns": tuple(columns), "uses": used,
             "uncovered": tuple(m.key for m in uncovered)}
    return Diagram("adapter_contract_map", Scene(tuple(items)), model=model)
