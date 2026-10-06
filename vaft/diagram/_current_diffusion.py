"""Current diffusion and current drive: how $j_\\phi(\\rho, t)$ and $q(\\rho, t)$ evolve (#1605).

``current_diffusion``
    a fast ohmic ramp ($t_\\mathrm{ramp} \\ll \\tau_R$) leaves the current in an
    off-axis shell -- a hollow $j_\\phi$ -- and a reversed-shear $q$; the current then penetrates
    and relaxes towards $j \\propto 1/\\eta$ -- peaked because the core is
    hotter, flat if $\\eta$ were uniform;
``current_drive_profiles``
    ohmic, ECCD and NBCD side by side: the drive's own source profile, the
    total current density it leads to in time, and the $q$ response.

Both evolve one reduced model: the enclosed current $I(\\rho, t)$ of a
straight cylinder under ``cylindrical_current_diffusion_rate`` (Faraday,
Ampère and Ohm's law with a non-inductive source), integrated implicitly,
with a fixed Spitzer resistivity
(``spitzer_resistivity_from_T_e_Z_eff_ln_Lambda``) on a fixed $T_e(\\rho)$
and the boundary current set by the circuit. Time is in units of the core
``resistive_diffusion_time``. The profiles are representative, not
universal: no transport, bootstrap current or toroidal geometry, and the
relaxed shape follows $\\eta(\\rho)$ and the sources rather than being an
endpoint every plasma reaches. The static shapes and $q$ landmarks are
``current_profile_shapes``, ``q_profile_topologies`` and
``q_profile_landmarks``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from vaft.formula.constants import MU0
from vaft.formula.equilibrium import (
    resistive_diffusion_time,
    shear_from_r_q,
    spitzer_resistivity_from_T_e_Z_eff_ln_Lambda,
)
from vaft.formula.geometry import cylindrical_current_diffusion_rate

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart
from ._concept import box
from ._equations import formula_equation
from ._equilibrium_profiles import (
    _PANEL,
    _SHEAR_FILL,
    _SHEAR_TEXT,
    _check_labels,
    _check_name,
    _note,
    _panel,
    _shear_runs,
    _to_panel,
)
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: radial grid points of the diffusion model
_POINTS = 201
#: the fixed electron temperature [eV]: a hot core over a cold edge
_T_CORE, _T_EDGE = 100.0, 10.0
_Z_EFF, _LN_LAMBDA = 1.0, 15.0
#: edge safety factor once the ramp is over
_Q_A = 3.5
#: ramp duration and the snapshots of ``current_diffusion`` [core tau_R]
_T_RAMP = 0.01
_STAGES = (("early", 0.03), ("penetration", 0.05), ("relaxed", 3.0))
_STAGE_TEXT = {"early": "after the ramp", "penetration": "penetration", "relaxed": "relaxed"}
#: snapshots of ``current_drive_profiles`` after the source is switched on [core tau_R]
_CD_EARLY, _CD_RELAXED = 0.002, 3.0
#: (centre, width, fraction of I_p) of each driven source, by deposition
_SOURCES = {
    "off_axis": {"ECCD": (0.5, 0.06, 0.15), "NBCD": (0.45, 0.2, 0.35)},
    "on_axis": {"ECCD": (0.0, 0.15, 0.06), "NBCD": (0.0, 0.35, 0.3)},
}
_STEP = 2e-4


def _temperature(x: np.ndarray) -> np.ndarray:
    return _T_EDGE + (_T_CORE - _T_EDGE) * (1.0 - x * x) ** 1.5


class _Cylinder:
    """The diffusing cylinder in units $a = 1$, $I_p = 1$ and $\\tau_R(0) = 1$.

    Spitzer $\\eta(\\rho)$ is rescaled by $\\mu_0/\\eta(0)$, which makes the core
    ``resistive_diffusion_time`` exactly one; the equation is linear in $I$,
    so its matrix is read off ``cylindrical_current_diffusion_rate`` itself.
    """

    def __init__(self, uniform: bool = False, points: int = _POINTS):
        self.x = np.linspace(0.0, 1.0, points)
        spitzer = spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(_temperature(self.x), _Z_EFF, _LN_LAMBDA)
        self.eta_physical = spitzer
        self.eta = MU0 * (np.full_like(self.x, spitzer[0]) if uniform else spitzer) / spitzer[0]
        self.tau_R = resistive_diffusion_time(1.0, self.eta[0])
        n = points
        zero = np.zeros(n)
        self.matrix = np.zeros((n, n))
        for k in range(1, n - 1):
            unit = zero.copy()
            unit[k] = 1.0
            self.matrix[1:-1, k] = cylindrical_current_diffusion_rate(self.x, unit, self.eta)[1:-1]
        edge = zero.copy()
        edge[-1] = 1.0
        self.matrix[1:-1, -1] = cylindrical_current_diffusion_rate(self.x, edge, self.eta)[1:-1]
        self._lu = {}

    def source(self, centre: float, width: float, fraction: float) -> np.ndarray:
        """A Gaussian driven current density carrying ``fraction`` of $I_p$ [$I_p/a^2$]."""
        g = np.exp(-(((self.x - centre) / width) ** 2))
        return g * fraction / (2.0 * np.pi * np.trapezoid(g * self.x, self.x))

    def _factor(self, dt: float):
        if dt not in self._lu:
            n = self.x.size
            a = np.eye(n)
            a[1:-1] -= dt * self.matrix[1:-1]
            self._lu[dt] = lu_factor(a)
        return self._lu[dt]

    def evolve(self, I: np.ndarray, t0: float, times: Sequence[float], *, ramp: float = 0.0,
               j_ni: np.ndarray = None) -> Dict[float, np.ndarray]:
        """Backward-Euler $I(\\rho)$ at each of ``times``, the edge current ramping linearly over ``ramp``."""
        forcing = np.zeros_like(self.x)
        if j_ni is not None:
            forcing[1:-1] = cylindrical_current_diffusion_rate(self.x, np.zeros_like(self.x), self.eta, j_ni)[1:-1]
        I, t, out = I.copy(), t0, {}
        for target in sorted(times):
            while t < target - 1e-12:
                dt = min(_STEP if t < 0.2 else 20 * _STEP, target - t)
                t += dt
                rhs = I + dt * forcing
                rhs[0], rhs[-1] = 0.0, min(t / ramp, 1.0) if ramp > 0 else 1.0
                I = lu_solve(self._factor(dt), rhs)
            out[target] = I.copy()
        return out


def _profiles(x: np.ndarray, I: np.ndarray) -> Dict[str, np.ndarray]:
    """$j/\\bar j$, $q$ and $s$ of one enclosed current, $q_a$ at the full $I_p$."""
    j = np.empty_like(x)
    j[1:-1] = (I[2:] - I[:-2]) / (x[2:] - x[:-2]) / (2.0 * x[1:-1])
    j[0], j[-1] = j[1] + (j[1] - j[2]) * x[1] / (x[2] - x[1]), 2.0 * j[-2] - j[-3]
    q = np.empty_like(x)
    q[1:] = _Q_A * x[1:] ** 2 / I[1:]
    q[0] = _Q_A / j[0]
    return {"j": j, "q": q, "s": shear_from_r_q(x, q)}


def _ramp(uniform: bool = False) -> Tuple[_Cylinder, Dict[str, Dict[str, np.ndarray]]]:
    cyl = _Cylinder(uniform=uniform)
    states = cyl.evolve(np.zeros_like(cyl.x), 0.0, [t for _, t in _STAGES], ramp=_T_RAMP)
    return cyl, {name: {"t": t, "I": states[t], **_profiles(cyl.x, states[t])} for name, t in _STAGES}


# ---------------------------------------------------------------------------
# current diffusion
# ---------------------------------------------------------------------------


def current_diffusion(*, labels: bool = True) -> Diagram:
    r"""Ohmic current diffusion after a fast ramp: hollow current, reversed shear, relaxation.

    The edge current rises to $I_p$ in $t_\mathrm{ramp} = 0.01\,\tau_R$, far
    faster than the core resistive time $\tau_R = \mu_0 a^2/\eta(0)$
    (``resistive_diffusion_time``) -- the ordering of a fast ramp such as
    VEST's, though the model's ratio is illustrative, not a VEST estimate.
    Columns after the ramp: the current still in an off-axis shell (hollow
    $j_\phi$, $q_{\min}$ off axis, $s < 0$ inside it), already depleted at
    the cold, resistive edge; penetrating inward (weaker reversal); relaxed. The
    relaxed current is $\propto 1/\eta$ with one $E_\phi$ across the radius:
    peaked here because the core is hotter ($\eta \propto T_e^{-3/2}$); with a
    uniform $\eta$ (dashed) the same diffusion relaxes only to a flat current.
    Reduced cylinder at fixed $T_e(\rho)$, evolved by
    ``cylindrical_current_diffusion_rate``; representative, not universal.
    """
    labels = _check_labels(labels)
    cyl, states = _ramp()
    _, flat = _ramp(uniform=True)
    x = cyl.x
    j_top = 1.12 * max(float(s["j"].max()) for s in states.values())
    q_top = 8.0
    step = 7.4
    x_first = 6.4
    j_base = _PANEL * CHART_HEIGHT + 2.3
    items: List = []
    shear_runs = {}
    # left column: the ramp in time and the resistivity in radius
    tc = Chart(x_range=(0.0, 0.065), y_range=(0.0, 1.25))
    t = np.linspace(0.0, 0.065, 131)
    tc.curves["I_p"] = np.stack([t, np.minimum(t / _T_RAMP, 1.0)], -1)
    items += _panel(tc, (0.0, j_base), scale=0.45, x_label="$t/\\tau_R$", y_label="$I_p$",
                    curve_styles={"I_p": "boundary"}, region_text={}, x_ticks=(_T_RAMP, 0.03, 0.05),
                    x_tick_text=("$t_\\mathrm{ramp}$", "", ""))
    for name, ts in _STAGES[:2]:
        items.append(Marker(_to_panel(tc, (ts, 1.0), (0.0, j_base), 0.45), "o", "opoint", role=f"stage:{name}"))
    ec = Chart(x_range=(0.0, 1.05), y_range=(0.0, 1.15 * float(cyl.eta.max() / cyl.eta[0])))
    ec.curves["eta"] = np.stack([x, cyl.eta / cyl.eta[0]], -1)
    items += _panel(ec, (0.0, 0.0), scale=0.45, x_label="$\\rho$", y_label="$\\eta/\\eta(0)$",
                    curve_styles={"eta": "inner solution"}, region_text={}, x_ticks=(0.0, 1.0),
                    y_ticks=(1.0,), y_tick_text=("$1$",))
    for k, (name, _) in enumerate(_STAGES):
        st = states[name]
        x0 = x_first + k * step
        jc = Chart(x_range=(0.0, 1.05), y_range=(0.0, j_top))
        jc.curves["j"] = np.stack([x, st["j"]], -1)
        styles = {"j": "boundary"}
        if name == "relaxed":
            jc.curves["j_uniform_eta"] = np.stack([x, flat["relaxed"]["j"]], -1)
            styles["j_uniform_eta"] = "approx"
            relaxed_at = (jc, x0)
        items += _panel(jc, (x0, j_base), x_label="$\\rho$", y_label="$j_\\phi/\\bar j$", curve_styles=styles,
                        region_text={}, x_ticks=(0.0, 1.0), y_ticks=(1.0,), y_tick_text=("$1$",))
        qc = Chart(x_range=(0.0, 1.05), y_range=(0.0, q_top))
        q_clipped = np.minimum(st["q"], 0.98 * q_top)
        qc.curves["q"] = np.stack([x, q_clipped], -1)
        runs = _shear_runs(x, st["s"])
        shear_runs[name] = runs
        for sign, a, b in runs:
            (xa, _), (xb, _) = _to_panel(qc, (a, 0.0), (x0, 0.0)), _to_panel(qc, (b, 0.0), (x0, 0.0))
            items.append(Polyline.of([(xa, 0.0), (xb, 0.0), (xb, _PANEL * CHART_HEIGHT), (xa, _PANEL * CHART_HEIGHT)],
                                     _SHEAR_FILL[sign], role=f"shear:{sign}", closed=True))
            if labels and xb - xa > 0.7:
                items.append(Label((0.5 * (xa + xb), _PANEL * CHART_HEIGHT - 0.05), _SHEAR_TEXT[sign],
                                   "small label", anchor="north", role=f"shear:{sign}"))
        q_styles = {"q": "boundary"}
        if name == "relaxed":
            qc.curves["q_uniform_eta"] = np.stack([x, np.minimum(flat["relaxed"]["q"], 0.98 * q_top)], -1)
            q_styles["q_uniform_eta"] = "approx"
        items += _panel(qc, (x0, 0.0), x_label="$\\rho$", y_label="$q$", curve_styles=q_styles, region_text={},
                        x_ticks=(0.0, 1.0), y_ticks=tuple(range(2, int(q_top) + 1, 2)))
        i_min = int(np.argmin(st["q"]))
        at = _to_panel(qc, (x[i_min], st["q"][i_min]), (x0, 0.0))
        items.append(Marker(at, "o", "opoint", role="q_min"))
        if labels:
            text = "$q_{\\min} = q_0$" if x[i_min] < 0.02 else "$q_{\\min}$"
            items.append(Label((at[0] + 0.1, at[1] - 0.12), text, "small label", anchor="north west", role="q_min"))
            if st["q"][0] > q_top:
                top_left = _to_panel(qc, (0.0, q_top), (x0, 0.0))
                items.append(Label((top_left[0] + 0.15, top_left[1] + 0.2), f"$q_0 \\approx {st['q'][0]:.0f}$ (clipped)",
                                   "small label", anchor="south west", role="q0"))
            stage = _STAGE_TEXT[name] + ("" if name == "relaxed" else f", $t = {st['t']:g}\\,\\tau_R$")
            items.append(Label((x0 + _PANEL * 0.5 * CHART_WIDTH, j_base + _PANEL * CHART_HEIGHT + 0.45), stage,
                               "subtitle", anchor="south", role="title"))
        if k:
            y = j_base + 0.5 * _PANEL * CHART_HEIGHT
            items.append(Arrow((x0 - step + _PANEL * CHART_WIDTH + 0.35, y), (x0 - 1.55, y), "connector",
                               role="chain"))
    if labels:
        mid = x_first + 0.5 * (2 * step + _PANEL * CHART_WIDTH)
        items += [
            Label((0.45 * 0.5 * CHART_WIDTH, j_base + 0.45 * CHART_HEIGHT + 0.45), "fast ramp", "subtitle",
                  anchor="south", role="title"),
            Label((0.45 * 0.5 * CHART_WIDTH, 0.45 * CHART_HEIGHT + 0.45), "hot core, cold edge", "subtitle",
                  anchor="south", role="title"),
            Label(_to_panel(relaxed_at[0], (1.05, 0.92 * relaxed_at[0].y_range[1]), (relaxed_at[1], j_base)),
                  "dashed: uniform $\\eta$", "small label", anchor="north east", role="uniform_eta"),
            Label((mid - 4.0, -1.45), f"$\\displaystyle {formula_equation(cylindrical_current_diffusion_rate)}$",
                  "formula box", anchor="north", role="equations"),
            Label((mid + 6.0, -1.45), f"$\\displaystyle {formula_equation(resistive_diffusion_time)}$",
                  "formula box", anchor="north", role="equations"),
            _note(f"$t_\\mathrm{{ramp}} = {_T_RAMP:g}\\,\\tau_R \\ll \\tau_R$: the current is driven at the edge "
                  "faster than it can diffuse in: it is left in an off-axis shell, and $q$ is reversed.", mid - 3.2, -3.2),
            _note("Relaxed $j_\\phi \\propto 1/\\eta$: peaked because the core is hotter; with a uniform $\\eta$ "
                  "the same diffusion gives a flat current. Reduced cylinder at fixed $T_e(\\rho)$.", mid - 3.2,
                  -3.8),
        ]
    model = {"x": x, "stages": states, "uniform_eta": flat, "eta": cyl.eta / cyl.eta[0], "tau_R": cyl.tau_R,
             "t_ramp": _T_RAMP, "shear_runs": shear_runs}
    return Diagram("current_diffusion", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# current-drive profiles
# ---------------------------------------------------------------------------

_ROWS = ("Ohmic", "ECCD", "NBCD")
_ACTUATOR = {
    "Ohmic": "transformer, loop voltage $E_\\phi$",
    "ECCD": "EC wave, localized",
    "NBCD": "neutral beam, broad",
}


def _drive_states(deposition: str) -> Dict[str, Dict]:
    """Source, and total $j$ and $q$ before, early and relaxed, for each actuator."""
    cyl, ramp = _ramp()
    x = cyl.x
    sigma = cyl.eta[0] / cyl.eta
    rows = {"Ohmic": {
        "source": sigma / (2.0 * np.pi * np.trapezoid(sigma * x, x)),
        "before": None, "early": ramp["early"], "relaxed": ramp["relaxed"], "times": (_STAGES[0][1], None)}}
    base = ramp["relaxed"]["I"]
    for name in ("ECCD", "NBCD"):
        source = cyl.source(*_SOURCES[deposition][name])
        out = cyl.evolve(base, 0.0, (_CD_EARLY, _CD_RELAXED), j_ni=source)
        rows[name] = {"source": source, "before": ramp["relaxed"],
                      "early": {"I": out[_CD_EARLY], **_profiles(x, out[_CD_EARLY])},
                      "relaxed": {"I": out[_CD_RELAXED], **_profiles(x, out[_CD_RELAXED])},
                      "times": (_CD_EARLY, _CD_RELAXED), "parameters": _SOURCES[deposition][name]}
    return {"x": x, "rows": rows}


def current_drive_profiles(deposition: str = "off_axis", *, labels: bool = True) -> Diagram:
    r"""Ohmic, ECCD and NBCD: actuator $\to$ source $j_\mathrm{drive}(\rho)$ $\to$ total $j_\phi(\rho, t)$ $\to$ $q(\rho, t)$.

    One row per drive, at fixed $I_p$. The source column is what the
    actuator itself drives: the conductivity-weighted ohmic shape
    $\propto 1/\eta$, a narrow ECCD Gaussian, a broad NBCD one
    (``deposition="off_axis"`` or ``"on_axis"``). The total current and $q$
    are drawn before the source is switched on (grey), shortly after
    (dashed) and relaxed (solid), from ``cylindrical_current_diffusion_rate``
    with the source as $j_\mathrm{ni}$: the driven current persists where it
    is deposited, while the ohmic current around it readjusts resistively --
    the source does not itself diffuse. The ohmic row's dashed curve is the
    hollow current after a fast ramp. On-axis drive takes $q_0$ below 1; no
    sawtooth is modelled. Representative cylinder profiles; NBI's pressure,
    rotation, fast-ion and bootstrap effects and counter-drive are not drawn.
    """
    deposition = _check_name(deposition, tuple(_SOURCES), "deposition")
    labels = _check_labels(labels)
    data = _drive_states(deposition)
    x, rows = data["x"], data["rows"]
    j_top = 1.12 * max(float(max(r["early"]["j"].max(), r["relaxed"]["j"].max())) for r in rows.values())
    q_top = 8.0
    row_step = _PANEL * CHART_HEIGHT + 2.2
    col = {"actuator": 0.0, "source": 5.2, "total": 12.6, "q": 20.0}
    items: List = []
    for i, name in enumerate(_ROWS):
        r = rows[name]
        y0 = (len(_ROWS) - 1 - i) * row_step
        b = box(col["actuator"] + 1.6, y0 + 0.5 * _PANEL * CHART_HEIGHT, 3.2, 1.7,
                f"\\textbf{{{name}}}\\\\{_ACTUATOR[name]}" if labels else f"\\textbf{{{name}}}", latex=True,
                role=f"actuator:{name}")
        items += list(b.items)
        sc = Chart(x_range=(0.0, 1.05), y_range=(0.0, 1.12 * float(r["source"].max()) * np.pi))
        sc.curves["source"] = np.stack([x, r["source"] * np.pi], -1)  # in units of j_bar = I_p / pi a^2
        items += _panel(sc, (col["source"], y0), x_label="$\\rho$" if i == 2 else "",
                        y_label="$j_\\mathrm{drive}/\\bar j$", curve_styles={"source": "inner solution"},
                        region_text={}, x_ticks=(0.0, 1.0) if i == 2 else ())
        tc = Chart(x_range=(0.0, 1.05), y_range=(0.0, j_top))
        qc = Chart(x_range=(0.0, 1.05), y_range=(0.0, q_top))
        styles_j, styles_q = {}, {}
        for when, style in (("before", "approx"), ("early", "slope minus"), ("relaxed", "boundary")):
            state = r[when]
            if state is None:
                continue
            tc.curves[when] = np.stack([x, state["j"]], -1)
            qc.curves[when] = np.stack([x, np.minimum(state["q"], 0.98 * q_top)], -1)
            styles_j[when] = styles_q[when] = style
        items += _panel(tc, (col["total"], y0), x_label="$\\rho$" if i == 2 else "", y_label="$j_\\phi/\\bar j$",
                        curve_styles=styles_j, region_text={}, x_ticks=(0.0, 1.0) if i == 2 else ())
        items += _panel(qc, (col["q"], y0), x_label="$\\rho$" if i == 2 else "", y_label="$q$",
                        curve_styles=styles_q, region_text={}, x_ticks=(0.0, 1.0) if i == 2 else (),
                        y_ticks=(2.0, 4.0, 6.0))
        relaxed = r["relaxed"]
        i_min = int(np.argmin(relaxed["q"]))
        items.append(Marker(_to_panel(qc, (x[i_min], relaxed["q"][i_min]), (col["q"], y0)), "o", "opoint",
                            role=f"q_min:{name}"))
        yc = y0 + 0.5 * _PANEL * CHART_HEIGHT
        for a, bx in (("actuator", "source"), ("source", "total"), ("total", "q")):
            start = col[a] + (3.3 if a == "actuator" else _PANEL * CHART_WIDTH + 0.35)
            items.append(Arrow((start, yc), (col[bx] - 1.55, yc), "connector", role="chain"))
        if labels:
            if x[i_min] > 0.02:
                items.append(Label(_to_panel(qc, (x[i_min], relaxed["q"][i_min]), (col["q"], y0 - 0.12)),
                                   "$q_{\\min}$, $s < 0$ inside", "small label", anchor="north", role=f"q_min:{name}"))
            elif np.any(relaxed["s"][1:-1] < -0.1):
                k = int(np.argmin(relaxed["s"][1:-1])) + 1
                items.append(Label(_to_panel(qc, (x[k], relaxed["q"][k]), (col["q"], y0 + 0.25)), "local $s < 0$",
                                   "small label", anchor="south", role=f"shear:{name}"))
    if labels:
        top = (len(_ROWS) - 1) * row_step + _PANEL * CHART_HEIGHT + 0.45
        for key, text in (("source", "source $j_\\mathrm{drive}(\\rho)$"), ("total", "total $j_\\phi(\\rho, t)$"),
                          ("q", "$q(\\rho, t)$")):
            items.append(Label((col[key] + _PANEL * 0.5 * CHART_WIDTH, top), text, "subtitle", anchor="south",
                               role="title"))
        items.append(Label((col["actuator"] + 1.6, top), "actuator", "subtitle", anchor="south", role="title"))
        mid = 0.5 * (col["q"] + _PANEL * CHART_WIDTH)
        items += [
            _note(f"Grey: before switch-on; dashed: ${_CD_EARLY:g}\\,\\tau_R$ after (ohmic row: the hollow current "
                  f"${_STAGES[0][1]:g}\\,\\tau_R$ into a fast ramp); solid: relaxed. Fixed $I_p$, "
                  f"{'off-axis' if deposition == 'off_axis' else 'on-axis'} deposition.", mid, -1.45),
            _note("The source persists where it is deposited; the ohmic current around it readjusts resistively. "
                  "Reduced cylinder; NBI's pressure,", mid, -2.05),
            _note("rotation, fast-ion and bootstrap effects and counter-drive are not drawn. Representative, "
                  "not universal profiles.", mid, -2.6),
        ]
    return Diagram("current_drive_profiles", Scene(tuple(items)), model={"deposition": deposition, **data})
