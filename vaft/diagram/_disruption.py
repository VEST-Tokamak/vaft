"""Disruption physics: the quench sequence, its causal chain, runaway generation and energy paths (#1041).

``disruption_timeline``
    a 0-D reference model integrated on a common time axis: prescribed
    thermal quench -> Spitzer resistivity -> L/R current quench -> induced
    $E_\\parallel$ -> Dreicer seed and avalanche -> runaway plateau;
``disruption_causal_chain``
    the same chain as cause and effect, each link named by the formula that
    carries it;
``runaway_generation``
    the Dreicer and avalanche rates against $E/E_c$, with the regimes
    $E < E_c$, avalanche-only and Dreicer marked;
``disruption_energy_pathways``
    where the thermal and the magnetic energy go.

The timeline's model is diagram-private and deliberately minimal: every rate
in it is a ``vaft.formula.disruption`` or ``startup`` relation, but the
temperature is prescribed and the runaways are a single density with no
transport, radiation or vessel coupling. It explains the sequence; it does
not predict a discharge.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List

import numpy as np

from vaft.formula.disruption import (
    avalanche_growth_rate,
    connor_hastie_critical_field,
    dreicer_field,
    dreicer_generation_rate,
    inductive_parallel_electric_field,
    relativistic_collision_time,
    runaway_current_from_density,
    thermal_quench_temperature,
)
from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
from vaft.formula.constants import C_LIGHT, QE
from vaft.formula.startup import (
    plasma_inductance_circular_from_R0_a_li,
    plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa,
)

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Label, Polyline, Scene

#: the reference plasma of the timeline: a generic medium tokamak, not a VEST shot
REFERENCE = {"R0": 1.7, "a": 0.5, "kappa": 1.6, "l_i": 1.0, "I_0": 1.0e6, "n_e": 5.0e19, "T_0": 2000.0,
             "T_final": 5.0, "tau_TQ": 0.5e-3, "Z_eff": 1.5, "ln_Lambda_rel": 15.0, "ln_Lambda_th": 12.0}


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


@lru_cache(maxsize=4)
def reference_model(t_end: float = 20e-3, steps: int = 8000) -> dict:
    r"""Integrate the 0-D reference model; arrays over time.

    Circuit: $L_p\,d(I_\Omega + I_\mathrm{RE})/dt = -R_pI_\Omega$ with $R_p$ from
    the Spitzer resistivity of the prescribed $T_e(t)$; field
    $E = R_pI_\Omega/(2\pi R_0)$, which is ``inductive_parallel_electric_field``
    of the total current's decay; runaways
    $dn_\mathrm{RE}/dt = S_\mathrm{Dreicer} + \gamma_\mathrm{av}n_\mathrm{RE}$,
    carrying $I_\mathrm{RE} = ecn_\mathrm{RE}A$. Explicit Euler; the step
    resolves the quench, the L/R time and the avalanche. Once the ohmic
    current is gone $E$ is zero and the plateau is frozen: the model has no
    $E \approx E_c$ self-consistency, no runaway loss.
    """
    p = REFERENCE
    L_p = float(plasma_inductance_circular_from_R0_a_li(p["R0"], p["a"], p["l_i"], p["kappa"]))
    area = math.pi * p["a"] ** 2 * p["kappa"]
    E_c = float(connor_hastie_critical_field(p["n_e"], p["ln_Lambda_rel"]))
    tau_c = float(relativistic_collision_time(p["n_e"], p["ln_Lambda_rel"]))
    t = np.linspace(-0.1 * t_end, t_end, steps + 1)
    dt = float(t[1] - t[0])
    out = {k: np.empty_like(t) for k in ("T_e", "eta", "E", "I_p", "I_RE", "n_RE", "dreicer", "gamma")}
    I_ohm, n_RE = p["I_0"], 0.0
    for i, ti in enumerate(t):
        T = float(thermal_quench_temperature(ti, p["T_0"], p["T_final"], p["tau_TQ"]))
        eta = float(spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(T, p["Z_eff"], p["ln_Lambda_th"]))
        R_p = float(plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa(eta, p["R0"], p["a"], p["kappa"]))
        dI_total_dt = -R_p * I_ohm / L_p
        E = float(inductive_parallel_electric_field(L_p, dI_total_dt, p["R0"]))
        S = float(dreicer_generation_rate(p["n_e"], T, E, p["Z_eff"], p["ln_Lambda_th"], prefactor=1.0))
        g = float(avalanche_growth_rate(E, E_c, p["Z_eff"], p["ln_Lambda_rel"]))
        I_RE = float(runaway_current_from_density(n_RE, area))
        for key, v in (("T_e", T), ("eta", eta), ("E", E), ("I_p", I_ohm + I_RE), ("I_RE", I_RE), ("n_RE", n_RE),
                       ("dreicer", S), ("gamma", g)):
            out[key][i] = v
        n_new = n_RE + (S + g * n_RE) * dt
        dI_RE = float(runaway_current_from_density(n_new, area)) - I_RE
        I_ohm_new = I_ohm + dI_total_dt * dt - dI_RE
        if I_ohm_new < -1e-9 * p["I_0"]:
            raise RuntimeError("the runaways outgrew the ohmic current in one step: reduce the step")
        I_ohm = max(I_ohm_new, 0.0)
        n_RE = n_new
    area_c = float(QE * C_LIGHT * area)
    seed = float(np.trapezoid(out["dreicer"], t)) * area_c  # current of the primary (Dreicer) runaways alone
    out.update({"t": t, "L_p": L_p, "area": area, "E_c": E_c, "tau_c": tau_c, "params": dict(p),
                "seed_current": seed, "avalanche_gain": float(out["I_RE"][-1] / seed) if seed > 0 else math.inf})
    for v in out.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)  # cached: callers must not be able to corrupt it
    return out


def _phase_edges(m: dict) -> dict:
    """Where the stages begin: TQ at t = 0, CQ when I_p has lost 5 %, plateau when I_RE carries half."""
    t, I_p, I_RE = m["t"], m["I_p"], m["I_RE"]
    cq = float(t[np.argmax(I_p < 0.95 * m["params"]["I_0"])])
    half = I_RE > 0.5 * I_p
    plateau = float(t[np.argmax(half)]) if half.any() else float(t[-1])
    return {"thermal_quench": 0.0, "current_quench": cq, "re_plateau": plateau}


def disruption_timeline(*, labels: bool = True) -> Diagram:
    r"""A disruption on one time axis, from a 0-D reference model built of the formulas.

    Four stacked panels over $-2$ to $20$ ms: the prescribed thermal quench
    $T_e/T_0$ (``thermal_quench_temperature``, 2 keV to 5 eV in 0.5 ms);
    the plasma and runaway currents; $\log_{10}(E_\parallel/E_c)$ with $E_c$
    from ``connor_hastie_critical_field``; and the resistivity
    (``spitzer_resistivity_from_T_e_Z_eff_ln_Lambda``). The current decays by
    L/R with $R_p$ rising as the plasma cools, the induced field reaches about
    $10^3E_c$ (and $E/E_D \approx 2\,\%$), a Dreicer seed of a few kA forms
    and the avalanche multiplies it a few e-folds -- at 1 MA only a few; at
    reactor currents tens -- until a runaway plateau carries part of the
    current (``reference_model``; $R_0 = 1.7$ m, $a = 0.5$ m, 1 MA,
    $5\times10^{19}$ m$^{-3}$). Illustrative magnitudes, not to scale for any
    machine: no universal waveform or timescale is implied.
    """
    labels = _check_labels(labels)
    m = reference_model()
    p = m["params"]
    t_ms = m["t"] * 1e3
    panels = [
        ("T_e", "$T_e/T_0$", m["T_e"] / p["T_0"], (0.0, 1.1), "component real"),
        ("I", "$I/I_0$", None, (0.0, 1.1), None),
        ("E", "$\\log_{10}(E_\\parallel/E_c)$", np.log10(np.maximum(m["E"] / m["E_c"], 1e-3)), (-2.0, 4.0),
         "orbit electron"),
        ("eta", "$\\log_{10}(\\eta/\\eta_0)$", np.log10(m["eta"] / m["eta"][0]), (0.0, 5.5), "orbit ion"),
    ]
    height = 2.2
    items: List = []
    edges = _phase_edges(m)
    x_range = (float(t_ms[0]), float(t_ms[-1]))
    for k, (key, ylabel, y, y_range, style) in enumerate(panels):
        chart = Chart(x_range=x_range, y_range=y_range)
        if key == "I":
            chart.curves["I_p"] = np.stack([t_ms, m["I_p"] / p["I_0"]], -1)
            chart.curves["I_RE"] = np.stack([t_ms, m["I_RE"] / p["I_0"]], -1)
            styles = {"I_p": "boundary", "I_RE": "crest"}
        else:
            chart.curves[key] = np.stack([t_ms, y], -1)
            styles = {key: style}
        if key == "E":
            chart.curves["E_c"] = np.array([[x_range[0], 0.0], [x_range[1], 0.0]])
            styles["E_c"] = "rational"
        y_ticks = {"E": (0.0, 3.0), "eta": (0.0, 5.0), "T_e": (0.0, 1.0), "I": (0.0, 1.0)}[key]
        scene = render_chart(chart, x_label="$t$ [ms]" if k == len(panels) - 1 else "", y_label="",
                             curve_styles=styles, region_text={},
                             x_ticks=(0.0, 10.0, 20.0) if k == len(panels) - 1 else (), y_ticks=y_ticks,
                             y_tick_text=tuple(f"${v:g}$" for v in y_ticks))
        offset = (0.0, (len(panels) - 1 - k) * (height + 0.5))
        items += _squash(scene, height / CHART_HEIGHT, offset)
        if labels:
            items.append(Label((-0.75, offset[1] + 0.5 * height), ylabel, "small label", anchor="east",
                               role=f"panel:{key}"))
    total_h = len(panels) * (height + 0.5) - 0.5
    # stage boundaries across all panels
    chart0 = Chart(x_range=x_range, y_range=(0.0, 1.0))
    for name, t0 in edges.items():
        x = float(chart0.to_cm(np.array([t0 * 1e3, 0.0]))[0])
        items.append(Polyline.of([(x, 0.0), (x, total_h)], "approx", role=f"stage:{name}"))
        if labels:
            text = {"thermal_quench": "TQ", "current_quench": "CQ", "re_plateau": "RE plateau"}[name]
            items.append(Label((x + 0.08, total_h + 0.1), text, "small label", anchor="south west",
                               role=f"stage:{name}"))
    if labels:
        items += [
            Label((0.1, total_h + 0.1), "pre", "small label", anchor="south west", role="stage:pre"),
            Label((CHART_WIDTH + 0.4, total_h - 0.2), "red: $T_e$\\\\ heavy: $I_p$\\\\ blue: $I_\\mathrm{RE}$\\\\ "
                  "blue dashed: $E = E_c$\\\\ grey dashed: stage starts\\\\ "
                  f"Dreicer seed {m['seed_current'] / 1e3:.1f} kA, avalanche $\\times{m['avalanche_gain']:.0f}$",
                  "small label,align=left", anchor="north west", role="legend"),
            Label((CHART_WIDTH / 2, -1.25), f"$\\displaystyle {formula_equation(inductive_parallel_electric_field)}$",
                  "formula box", anchor="north", role="equations"),
            _note("0-D reference model: prescribed $T_e$, Spitzer $\\eta$, L/R, Dreicer + avalanche; "
                  "no transport, radiation or vessel. Illustrative magnitudes", CHART_WIDTH / 2, -2.4),
        ]
    model = {"time": m["t"], "edges": edges, "peak_E_over_Ec": float(np.max(m["E"] / m["E_c"])),
             "I_RE_final": float(m["I_RE"][-1]), "seed_current": m["seed_current"],
             "avalanche_gain": m["avalanche_gain"], "reference": m}
    return Diagram("disruption_timeline", Scene(tuple(items)), model=model)


def _squash(scene: Scene, y_scale: float, offset) -> List:
    """The chart scene with its height scaled by ``y_scale`` and shifted, text left unscaled."""
    from dataclasses import replace

    from ._scene import Arrow, Marker

    def f(pt):
        return (pt[0] + offset[0], pt[1] * y_scale + offset[1])

    out = []
    for it in scene.items:
        if isinstance(it, Polyline):
            out.append(replace(it, points=tuple(f(q) for q in it.points)))
        elif isinstance(it, Arrow):
            out.append(replace(it, start=f(it.start), end=f(it.end)))
        elif isinstance(it, (Marker, Label)):
            out.append(replace(it, at=f(it.at)))
    return out


def disruption_causal_chain(*, labels: bool = True) -> Diagram:
    r"""The thermal-quench -> current-quench -> runaway chain as cause and effect.

    Each box a physical state, each arrow the relation that links it to the
    next, named by its ``vaft.formula`` function: loss of confinement ->
    $T_e$ drops (``thermal_quench_temperature``) -> $\eta \propto T_e^{-3/2}$
    rises (``spitzer_resistivity_from_T_e_Z_eff_ln_Lambda``) -> $R_p$ rises
    and $I_p$ decays by L/R (``lr_time_from_L_R``) -> $E_\parallel$ induced
    (``inductive_parallel_electric_field``) -> $E > E_c$
    (``connor_hastie_critical_field``) -> seed (``dreicer_generation_rate``,
    hot tail) -> avalanche (``avalanche_growth_rate``) -> runaway current.
    """
    labels = _check_labels(labels)
    steps = [
        ("MHD / loss of confinement", "", "start"),
        ("thermal quench: $T_e$ drops", "thermal\\_quench\\_temperature", "tq"),
        ("resistivity rises, $\\eta \\propto T_e^{-3/2}$", "spitzer\\_resistivity\\_from\\_T\\_e\\_Z\\_eff\\_ln\\_Lambda",
         "eta"),
        ("current quench, $\\tau = L_p/R_p$", "lr\\_time\\_from\\_L\\_R", "cq"),
        ("induced $E_\\parallel = -L_p\\dot I_p/(2\\pi R_0)$", "inductive\\_parallel\\_electric\\_field", "E"),
        ("$E_\\parallel > E_c$: runaways possible", "connor\\_hastie\\_critical\\_field", "Ec"),
        ("seed: Dreicer, hot tail", "dreicer\\_generation\\_rate", "seed"),
        ("avalanche: $\\dot n_\\mathrm{RE} = \\gamma_\\mathrm{av}n_\\mathrm{RE}$", "avalanche\\_growth\\_rate", "aval"),
        ("runaway plateau / termination", "runaway\\_current\\_from\\_density", "plateau"),
    ]
    items: List = []
    boxes = []
    for i, (text, fn, role) in enumerate(steps):
        col, row = i % 3, i // 3
        x = 2.6 + 5.4 * col if row % 2 == 0 else 2.6 + 5.4 * (2 - col)
        y = -2.4 * row
        b = box(x, y, 5.0, 1.3, text + (f"\\\\ {{\\scriptsize\\texttt{{{fn}}}}}" if (fn and labels) else ""),
                role=f"step:{role}", latex=True)
        boxes.append(b)
        items += list(b.items)
    for a, b in zip(boxes[:-1], boxes[1:]):
        items.append(connector(a, b, role="edge"))
    if labels:
        items.append(_note("Each arrow is a relation in vaft.formula; the thermal and magnetic energies take "
                           "different paths (disruption\\_energy\\_pathways)", 8.0, -6.0))
    return Diagram("disruption_causal_chain", Scene(tuple(items)), model={"steps": [s[2] for s in steps]})


def runaway_generation(*, labels: bool = True) -> Diagram:
    r"""Dreicer and avalanche generation against the field, and the regimes they define.

    For $n_e = 5\times10^{19}$ m$^{-3}$, $T_e = 200$ eV, $Z_\mathrm{eff} = 1.5$:
    $\log_{10}$ of the avalanche growth rate $\gamma_\mathrm{av}\tau_c$
    (``avalanche_growth_rate``, zero below $E_c$) and of the Dreicer rate per
    electron $S_D\tau_c/n_e$ (``dreicer_generation_rate``) against
    $\log_{10}(E/E_c)$. Below $E_c$ nothing runs away; between $E_c$ and a
    few per cent of $E_D$ only an existing seed multiplies; towards $E_D$
    the thermal tail itself runs away. The hot-tail seed of a fast quench is
    not drawn.
    """
    labels = _check_labels(labels)
    n_e, T_e, Z, lnr, lnt = 5e19, 200.0, 1.5, 15.0, 12.0
    E_c = float(connor_hastie_critical_field(n_e, lnr))
    tau_c = float(relativistic_collision_time(n_e, lnr))
    E_D = float(dreicer_field(n_e, T_e, lnt))
    x = np.linspace(-0.5, math.log10(E_D / E_c) + 0.1, 400)
    E = E_c * 10.0**x
    aval = avalanche_growth_rate(E, E_c, Z, lnr) * tau_c
    dre = dreicer_generation_rate(n_e, T_e, E, Z, lnt, prefactor=1.0) * tau_c / n_e
    valid = E <= 0.1 * E_D  # the asymptotic Dreicer formula's own range
    floor = -12.0
    chart = Chart(x_range=(float(x[0]), float(x[-1])), y_range=(floor, 1.0))
    keep_a = aval > 10.0**floor
    keep_d = (dre > 10.0**floor) & valid
    chart.curves.update({"avalanche": np.stack([x[keep_a], np.log10(aval[keep_a])], -1),
                         "dreicer": np.stack([x[keep_d], np.log10(dre[keep_d])], -1)})
    xd = math.log10(E_D / E_c)
    scene = render_chart(chart, x_label="$\\log_{10}(E/E_c)$", y_label="$\\log_{10}$ rate $\\times\\tau_c$",
                         curve_styles={"avalanche": "component imag", "dreicer": "component real"}, region_text={},
                         x_ticks=(0.0, xd), x_tick_text=("$E_c$", "$E_D$"), y_ticks=(floor, 0.0),
                         y_tick_text=(f"${floor:g}$", "$0$"))
    items: List = []
    x_c = float(chart.to_cm(np.array([0.0, 0.0]))[0])
    x_v = float(chart.to_cm(np.array([math.log10(0.1 * E_D / E_c), 0.0]))[0])
    items.append(Polyline.of([(0.0, 0.0), (x_c, 0.0), (x_c, CHART_HEIGHT), (0.0, CHART_HEIGHT)], "concept band",
                             role="regime:none", closed=True))
    items.append(Polyline.of([(x_v, 0.0), (CHART_WIDTH, 0.0), (CHART_WIDTH, CHART_HEIGHT), (x_v, CHART_HEIGHT)],
                             "concept band", role="regime:dreicer_invalid", closed=True))
    items += list(scene.items)
    if labels:
        items += [
            Label((0.5 * x_c, CHART_HEIGHT - 0.2), "no\\\\ runaway", "small label,align=center", anchor="north",
                  role="regime:none"),
            Label((0.42 * CHART_WIDTH, 0.55 * CHART_HEIGHT), "avalanche\\\\ multiplies a seed", "small label,align=center",
                  anchor="north", role="regime:avalanche"),
            Label((0.5 * (x_v + CHART_WIDTH), 0.35 * CHART_HEIGHT), "$E > 0.1E_D$:\\\\ asymptotic\\\\ Dreicer\\\\ "
                  "invalid", "small label,align=center", anchor="center", role="regime:dreicer_invalid"),
            Label((CHART_WIDTH + 0.4, CHART_HEIGHT), "blue: $\\gamma_\\mathrm{av}\\tau_c$ (avalanche, per runaway)"
                  "\\\\ red: $S_D\\tau_c/n_e$ (Dreicer, per electron)\\\\ "
                  f"$T_e = {T_e:g}$ eV, $E_D/E_c = {E_D / E_c:.0f}$", "small label,align=left",
                  anchor="north west", role="legend"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(avalanche_growth_rate)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Different normalisations -- per existing runaway vs per thermal electron -- so the curves' "
                  "magnitudes are not compared. Hot-tail seeding not drawn", CHART_WIDTH / 2 + 1.5, -2.7),
        ]
    return Diagram("runaway_generation", Scene(tuple(items)),
                   model={"E_c": E_c, "E_D": E_D, "tau_c": tau_c, "x": x, "avalanche": aval, "dreicer": dre,
                          "dreicer_drawn": x[keep_d]})


def disruption_energy_pathways(*, labels: bool = True) -> Diagram:
    r"""Where a disruption's thermal and magnetic energies go.

    The thermal energy $W_\mathrm{th} = \tfrac32\int(n_eT_e + n_iT_i)\,dV$
    leaves in the thermal quench, by conduction and convection to the wall
    and by radiation. The magnetic energy $\tfrac12L_pI_p^2$ of the poloidal
    field leaves in the current quench: ohmically heating the cold plasma
    (and so radiated), inductively coupled into the vessel and coils, and --
    if runaways form -- into their kinetic energy, deposited locally on the
    wall. Two pools, two timescales, different damage mechanisms.
    """
    labels = _check_labels(labels)
    items: List = []
    w_th = box(2.5, 0.0, 4.6, 1.5, "thermal energy $W_\\mathrm{th}$\\\\ (released in the thermal quench)", role="pool:thermal",
               latex=True)
    w_mag = box(2.5, -5.4, 4.6, 1.5, "magnetic energy $\\tfrac12L_pI_p^2$\\\\ (released in the current quench)",
                role="pool:magnetic", latex=True)
    sinks = {
        "conduction": box(9.9, 1.0, 6.0, 1.0, "conduction, convection to the wall", role="sink:conduction",
                          latex=True),
        "radiation": box(9.9, -1.0, 6.0, 1.0, "radiation (impurities)", role="sink:radiation", latex=True),
        "ohmic": box(9.9, -3.0, 6.0, 1.0, "ohmic heating of the cold plasma", role="sink:ohmic", latex=True),
        "vessel": box(9.9, -4.6, 6.0, 1.0, "induced currents in vessel and coils", role="sink:vessel", latex=True),
        "runaway": box(9.9, -6.2, 6.0, 1.0, "runaway kinetic energy $\\to$ local wall damage", role="sink:runaway",
                       latex=True),
        "halo": box(9.9, -7.8, 6.0, 1.0, "halo currents in the wall (VDE) $\\to$ forces", role="sink:halo",
                    latex=True),
    }
    for b in (w_th, w_mag, *sinks.values()):
        items += list(b.items)
    edges = [(w_th, sinks["conduction"]), (w_th, sinks["radiation"]), (w_mag, sinks["ohmic"]),
             (w_mag, sinks["vessel"]), (w_mag, sinks["runaway"]), (w_mag, sinks["halo"]),
             (sinks["ohmic"], sinks["radiation"])]
    for a, b in edges:
        items.append(connector(a, b, role="edge"))
    if labels:
        items.append(_note("Two pools on two timescales; magnetic energy can exceed the thermal one in a "
                           "disruption, and runaways concentrate part of it on the wall; halo currents are issue 1042",
                           6.0, -8.8))
    return Diagram("disruption_energy_pathways", Scene(tuple(items)),
                   model={"pools": ("thermal", "magnetic"), "sinks": tuple(sinks)})
