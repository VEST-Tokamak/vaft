"""Cold-plasma waves from their equations: O and X modes, the CMA diagram and a profile's cutoffs (#1113).

``o_mode_cutoff``
    $n_O^2 = P$ against $\\omega/\\omega_{pe}$: the cutoff $P = 0$ at $\\omega_{pe}$;
``x_mode_dispersion``
    $n_X^2 = RL/S$ against $\\omega/|\\Omega_e|$: the $L$ and $R$ cutoffs, the
    upper-hybrid resonance $S = 0$ and the evanescent gaps between;
``cma_diagram``
    the $(X, Y)$ plane of a cold electron plasma with the boundaries $P = 0$,
    $R = 0$, $L = 0$, $S = 0$ and the cyclotron resonance $Y = 1$;
``profile_propagation``
    $n_O^2$ and $n_X^2$ along the midplane of an example tokamak at one
    frequency, with the cutoff and resonance layers located on the profile.

Every curve and boundary is evaluated from ``vaft.formula.waves``: the CMA
boundaries and the profile's layers are zeros (or poles) of the Stix
parameters found by bracketing, not closed forms typed into the drawing.
Cold electrons only, ions immobile -- the electron-cyclotron range.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from vaft.formula.constants import EPS0, ME, QE
from vaft.formula.waves import (
    cma_coordinates,
    cold_plasma_refractive_index_squared,
    perpendicular_refractive_index_squared,
    propagation_regime,
    stix_parameters,
)

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Label, Polyline, Scene

_Q, _M = np.array([-QE]), np.array([ME])
#: an electron density at which omega_pe = 2 pi x 1 GHz, the frequency unit of the normalised diagrams [m^-3]
_N_REF = (2.0 * np.pi * 1e9) ** 2 * EPS0 * ME / QE**2
_W_REF = 2.0 * np.pi * 1e9


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _electron_stix(omega, n_e, B):
    return stix_parameters(omega, np.asarray(n_e, dtype=float)[None, ...], _Q, _M, B)


def _field_for_Y(Y, omega):
    """The field magnitude at which |Omega_e| / omega = Y."""
    return np.asarray(Y, dtype=float) * omega * ME / QE


def _density_for_X(X, omega):
    return np.asarray(X, dtype=float) * _N_REF * (omega / _W_REF) ** 2


def _bisect(f: Callable, lo: float, hi: float, n: int = 60) -> float:
    """A zero of ``f`` in [lo, hi], where f changes sign."""
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _clip_runs(x, y, limit):
    """Runs of (x, y) with |y| <= limit, split at the poles."""
    ok = np.isfinite(y) & (np.abs(y) <= limit)
    runs, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        if (not good or i == len(ok) - 1) and start is not None:
            end = i + 1 if good else i
            if end - start >= 2:
                runs.append(np.stack([x[start:end], y[start:end]], axis=-1))
            start = None
    return runs


def o_mode_cutoff(*, labels: bool = True) -> Diagram:
    r"""O-mode refractive index against frequency: propagation above $\omega_{pe}$, evanescence below.

    $n_O^2 = P = 1 - \omega_{pe}^2/\omega^2$ from ``stix_parameters``; the
    cutoff $P = 0$ is located by bracketing the formula and lands on
    $\omega = \omega_{pe}$. Independent of $B$.
    """
    labels = _check_labels(labels)
    ratio = np.linspace(0.35, 3.0, 400)
    omega = ratio * _W_REF
    P = _electron_stix(omega, _N_REF, 0.0).P
    cutoff = _bisect(lambda r: _electron_stix(r * _W_REF, _N_REF, 0.0).P, 0.5, 2.0)
    chart = Chart(x_range=(0.0, 3.0), y_range=(-3.0, 1.5))
    chart.curves["n2_O"] = np.stack([ratio, P], axis=-1)[P >= -3.0]
    chart.curves["zero"] = np.array([[0.0, 0.0], [3.0, 0.0]])
    chart.curves["cutoff"] = np.array([[cutoff, -3.0], [cutoff, 1.5]])
    chart.points["cutoff"] = (cutoff, 0.0)
    chart.labels.update({"evanescent": (0.55, -2.3), "propagating": (2.2, 0.55), "cutoff": (cutoff + 0.45, -1.0)})
    chart.parameters.update({"cutoff_omega_over_omega_pe": cutoff})
    scene = render_chart(
        chart, x_label="$\\omega/\\omega_{pe}$", y_label="$n_O^2 = P$",
        curve_styles={"zero": "approx", "cutoff": "approx", "n2_O": "boundary"},
        region_text={"evanescent": "evanescent", "propagating": "propagating",
                     "cutoff": "\\small cutoff $P = 0$"} if labels else {},
        x_ticks=[0.0, 1.0, 2.0, 3.0], y_ticks=[-3.0, -2.0, -1.0, 0.0, 1.0],
        note="Cold electrons: $n_O^2 = 1 - \\omega_{pe}^2/\\omega^2$" if labels else "",
    )
    return Diagram("o_mode_cutoff", scene, model=chart)


def x_mode_dispersion(*, omega_pe_over_omega_ce: float = 1.2, labels: bool = True) -> Diagram:
    r"""X-mode refractive index against frequency: two cutoffs, the upper-hybrid resonance and two gaps.

    $n_X^2 = RL/S$ from ``perpendicular_refractive_index_squared``, for a
    cold electron plasma with $\omega_{pe}/|\Omega_e|$ =
    ``omega_pe_over_omega_ce``. Located from the formula by bracketing: the
    $L$ cutoff $\omega_L$, the upper-hybrid resonance $S = 0$ at
    $\omega_{UH}$, and the $R$ cutoff $\omega_R$. The X mode is evanescent
    below $\omega_L$, propagates
    between $\omega_L$ and $\omega_{UH}$, is evanescent between
    $\omega_{UH}$ and $\omega_R$, and propagates above $\omega_R$. Poles are
    masked, not drawn through.
    """
    labels = _check_labels(labels)
    if not omega_pe_over_omega_ce > 0.0:
        raise ValueError("omega_pe_over_omega_ce must be positive")
    n_e = _N_REF * omega_pe_over_omega_ce**2  # omega_pe in units of the reference, |Omega_e| = reference
    B = _field_for_Y(1.0, _W_REF)
    stix = lambda r: _electron_stix(r * _W_REF, n_e, B)
    w_uh_guess = np.sqrt(1.0 + omega_pe_over_omega_ce**2)
    omega_L = _bisect(lambda r: stix(r).L, 1e-3, w_uh_guess - 1e-6)
    omega_UH = _bisect(lambda r: stix(r).S, 1.0 + 1e-9, 3.5 * w_uh_guess)
    omega_R = _bisect(lambda r: stix(r).R, 1.0 + 1e-9, 3.5 * w_uh_guess)
    x_max = float(np.ceil(1.4 * omega_R * 2.0) / 2.0)
    ratio = np.linspace(0.05, x_max, 1400)
    s = _electron_stix(ratio * _W_REF, n_e, B)
    _, n2_X = perpendicular_refractive_index_squared(s.R, s.L, s.P)
    limit = 6.0
    chart = Chart(x_range=(0.0, x_max), y_range=(-limit, limit))
    for i, run in enumerate(_clip_runs(ratio, np.asarray(n2_X), limit)):
        chart.curves[f"n2_X {i}"] = run
    chart.curves["zero"] = np.array([[0.0, 0.0], [x_max, 0.0]])
    for name, value in (("omega_L", omega_L), ("omega_UH", omega_UH), ("omega_R", omega_R)):
        chart.curves[name] = np.array([[value, -limit], [value, limit]])
        chart.points[name] = (value, 0.0)
    chart.labels.update({"gap1": (omega_L / 2, -3.2), "prop": (0.5 * (omega_R + x_max), 2.2),
                         "prop_low": (0.5 * (omega_L + omega_UH) - 0.06, 2.7)})
    chart.parameters.update({"omega_pe_over_omega_ce": omega_pe_over_omega_ce, "omega_L": omega_L,
                             "omega_UH": omega_UH, "omega_R": omega_R})
    styles: Dict[str, str] = {name: "boundary" for name in chart.curves if name.startswith("n2_X")}
    styles.update({"zero": "approx", "omega_L": "approx", "omega_UH": "approx", "omega_R": "approx"})
    scene = render_chart(
        chart, x_label="$\\omega/|\\Omega_e|$", y_label="$n_X^2 = RL/S$", curve_styles=styles,
        region_text={"gap1": "\\small evanescent", "prop": "\\small propagating",
                     "prop_low": "\\small propagating"} if labels else {},
        x_ticks=[float(v) for v in range(int(x_max) + 1)], y_ticks=[-6.0, -3.0, 0.0, 3.0, 6.0],
        note=(f"Cold electrons, $\\omega_{{pe}}/|\\Omega_e| = {omega_pe_over_omega_ce:g}$; "
              "the pole at $S = 0$ is the upper-hybrid resonance" if labels else ""),
    )
    items = list(scene.items)
    if labels:
        cm = chart.to_cm
        top = float(cm(np.array([0.0, limit]))[1]) + 0.15
        items += [Label((float(cm(np.array([omega_L, 0.0]))[0]), top), "$L = 0$", "small label", anchor="south",
                        role="omega_L"),
                  Label((float(cm(np.array([omega_UH, 0.0]))[0]) - 0.05, top), "$S = 0$", "small label",
                        anchor="south east", role="omega_UH"),
                  Label((float(cm(np.array([omega_R, 0.0]))[0]) + 0.05, top), "$R = 0$", "small label",
                        anchor="south west", role="omega_R")]
        gap = cm(np.array([0.5 * (omega_UH + omega_R), -2.6]))
        items += [Polyline.of([tuple(gap), (float(gap[0]) + 1.1, float(gap[1]))], "leader line", role="gap2"),
                  Label((float(gap[0]) + 1.15, float(gap[1])), "evanescent", "small label", anchor="west",
                        role="gap2")]
    return Diagram("x_mode_dispersion", Scene(tuple(items)), model=chart)


def cma_boundaries(Y: np.ndarray) -> Dict[str, np.ndarray]:
    """The CMA boundaries X(Y), each a zero of one Stix parameter found by bracketing at every Y."""
    omega = _W_REF
    out: Dict[str, List] = {"P": [], "R": [], "L": [], "S": []}
    for y in Y:
        B = _field_for_Y(y, omega)
        comp = lambda X, name: getattr(_electron_stix(omega, _density_for_X(X, omega), B), name)
        out["P"].append(_bisect(lambda X: comp(X, "P"), 0.0, 10.0))
        out["L"].append(_bisect(lambda X: comp(X, "L"), 0.0, 10.0))
        # R and S have zeros in X > 0 only below the cyclotron resonance
        out["R"].append(_bisect(lambda X: comp(X, "R"), 0.0, 10.0) if y < 1.0 else np.nan)
        out["S"].append(_bisect(lambda X: comp(X, "S"), 0.0, 10.0) if y < 1.0 else np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


def cma_diagram(*, labels: bool = True) -> Diagram:
    r"""The Clemmow--Mullaly--Allis diagram of a cold electron plasma.

    Horizontal $X = \omega_{pe}^2/\omega^2$, vertical $Y = |\Omega_e|/\omega$
    (``cma_coordinates``). The cutoffs $P = 0$ (O mode), $R = 0$ and $L = 0$
    (X mode), the upper-hybrid resonance $S = 0$ and the cyclotron resonance
    $Y = 1$ (a pole of $R$), each found as the zero of the Stix parameter
    from ``stix_parameters`` at every $Y$. Ions immobile, so the ion
    resonances and the lower hybrid lie off this chart.
    """
    labels = _check_labels(labels)
    Y = np.linspace(0.0, 2.0, 201)
    bounds = cma_boundaries(Y)
    chart = Chart(x_range=(0.0, 2.5), y_range=(0.0, 2.0))
    for name in ("P", "R", "L", "S"):
        ok = np.isfinite(bounds[name])
        chart.curves[f"{name}=0"] = np.stack([bounds[name][ok], Y[ok]], axis=-1)
    chart.curves["Y=1"] = np.array([[0.0, 1.0], [2.5, 1.0]])
    chart.labels.update({"P": (1.15, 1.85), "R": (0.3, 0.55), "L": (2.2, 1.4), "S": (0.46, 0.85),
                         "Y": (2.25, 1.08)})
    chart.parameters.update({"electrons_only": 1.0})
    styles = {"P=0": "boundary", "R=0": "boundary", "L=0": "boundary", "S=0": "approx", "Y=1": "approx"}
    scene = render_chart(
        chart, x_label="$X = \\omega_{pe}^2/\\omega^2$", y_label="$Y = |\\Omega_e|/\\omega$", curve_styles=styles,
        region_text={"P": "\\small $P = 0$", "R": "\\small $R = 0$", "L": "\\small $L = 0$",
                     "S": "\\small $S = 0$", "Y": "\\small $Y = 1$"} if labels else {},
        x_ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], y_ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        note=("Cutoffs solid, resonances dashed; each boundary is a zero or pole of a Stix parameter"
              if labels else ""),
    )
    return Diagram("cma_diagram", scene, model=chart)


#: example tokamak of ``profile_propagation`` (illustrative, not a device): R0, a [m], B0 [T], n0 [m^-3], f [Hz]
EXAMPLE_PROFILE = {"R0": 1.0, "a": 0.3, "B0": 1.0, "n0": 3.0e19, "frequency": 28.0e9}


def profile_quantities(R: np.ndarray, p: Dict[str, float] = EXAMPLE_PROFILE):
    """n_e(R), B(R) and the perpendicular n^2 of the O and X modes along the midplane."""
    B = p["B0"] * p["R0"] / R
    n_e = p["n0"] * np.clip(1.0 - ((R - p["R0"]) / p["a"]) ** 2, 0.0, None)
    s = _electron_stix(2.0 * np.pi * p["frequency"], n_e, B)
    n2_O, n2_X = perpendicular_refractive_index_squared(s.R, s.L, s.P)
    return n_e, B, s, np.asarray(n2_O), np.asarray(n2_X)


def profile_layers(p: Dict[str, float] = EXAMPLE_PROFILE) -> Dict[str, List[float]]:
    """Cutoff and resonance layers on the midplane: sign changes of P, R, L, S and of 1 - Y, refined by bisection."""
    R = np.linspace(p["R0"] - p["a"], p["R0"] + p["a"], 2001)
    _, B, s, _, _ = profile_quantities(R, p)
    omega = 2.0 * np.pi * p["frequency"]
    one_minus_Y = 1.0 - QE * B / (ME * omega)
    series = {"P": s.P, "R": s.R, "L": s.L, "S": s.S, "ECR": one_minus_Y}
    layers: Dict[str, List[float]] = {}
    for name, values in series.items():
        values = np.asarray(values)
        finite = np.isfinite(values)
        idx = np.where(finite[:-1] & finite[1:] & (np.sign(values[:-1]) != np.sign(values[1:])))[0]
        found = []
        for i in idx:
            if name == "ECR":
                f = lambda r: 1.0 - QE * p["B0"] * p["R0"] / r / (ME * omega)
            else:
                f = lambda r, name=name: getattr(profile_quantities(np.array([r]), p)[2], name)[0]
            root = _bisect(f, R[i], R[i + 1])
            # a sign change across a pole (R and S at the cyclotron layer) is a resonance, not a zero
            if name in ("R", "S") and not abs(f(root)) < 1e-6:
                continue
            found.append(root)
        layers[name] = found
    return layers


def profile_propagation(*, theta=None, labels: bool = True) -> Diagram:
    r"""O- and X-mode $n^2$ along the midplane of an example tokamak, with its cutoff and resonance layers.

    Example parameters (``EXAMPLE_PROFILE``, illustrative, not a device):
    $B = B_0R_0/R$, a parabolic $n_e$, one frequency at the on-axis
    electron cyclotron frequency. The layers -- O cutoff $P = 0$, X cutoffs
    $R = 0$ and $L = 0$, upper-hybrid $S = 0$, cyclotron $Y = 1$ -- are
    sign changes of the Stix parameters along the profile, refined by
    bisection; ``propagation_regime`` classifies the samples between them.

    ``theta`` (default: perpendicular, the O and X modes) draws instead both
    roots of ``cold_plasma_refractive_index_squared`` at that angle to
    $\mathbf B$, unnamed and in one colour, and adds the oblique resonance
    layers $A = S\sin^2\theta + P\cos^2\theta = 0$.
    """
    labels = _check_labels(labels)
    oblique = theta is not None and not np.isclose(_check_theta(theta), 0.5 * np.pi)
    p = EXAMPLE_PROFILE
    R = np.linspace(p["R0"] - p["a"], p["R0"] + p["a"], 1201)
    _, _, s, n2_O, n2_X = profile_quantities(R, p)
    layers = profile_layers(p)
    if oblique:
        theta = float(theta)
        roots = _modes(s, theta)
        n2_O, n2_X = roots["+"], roots["-"]
        layers["S"] = []  # the upper hybrid is the perpendicular resonance; oblique it moves to A = 0
        layers["A"] = _resonance_cone(R, lambda r: profile_quantities(np.atleast_1d(r), p)[2], theta)
    limit = 3.0
    chart = Chart(x_range=(R[0], R[-1]), y_range=(-limit, 2.0))
    if oblique:  # an oblique root has gaps (the cyclotron pole): split it into runs, never chord across them
        for i, run in enumerate(_clip_runs(R, n2_O, limit)):
            chart.curves[f"n2_O {i}"] = run
    else:
        chart.curves["n2_O"] = np.stack([R, n2_O], axis=-1)[n2_O >= -limit]
    for i, run in enumerate(_clip_runs(R, n2_X, limit)):
        chart.curves[f"n2_X {i}"] = run
    chart.curves["zero"] = np.array([[R[0], 0.0], [R[-1], 0.0]])
    names = {"P": "$P{=}0$", "R": "$R{=}0$", "L": "$L{=}0$", "S": "UH", "ECR": "ECR", "A": "$A{=}0$"}
    for name, roots in layers.items():
        for j, r in enumerate(roots):
            key = f"{name} {j}"
            chart.curves[key] = np.array([[r, -limit], [r, 2.0]])
            chart.points[key] = (r, 0.0)
    chart.parameters.update({k: float(v) for k, v in p.items()})
    if oblique:
        chart.parameters["theta"] = theta
    # where each mode propagates, as strips along the bottom: propagation_regime on the sampled profile
    # off perpendicular the algebraic roots swap at the cyclotron layer: one strip, where either root propagates
    strips = ((("any", np.fmax(n2_O, n2_X), -2.6),) if oblique
              else (("O", n2_O, -2.45), ("X", n2_X, -2.75)))
    for mode, n2, y in strips:
        regime = propagation_regime(np.where(np.isfinite(n2), n2, np.inf))
        for i, run in enumerate(_clip_runs(R, np.where(regime == "propagating", y, np.nan), 10.0)):
            chart.curves[f"{mode} propagates {i}"] = run
        chart.parameters[f"{mode}_propagating_fraction"] = float(np.mean(regime == "propagating"))
    styles: Dict[str, str] = {"zero": "approx"}
    styles.update({name: "boundary" for name in chart.curves if name.split(" ")[0] == "n2_O"})
    styles.update({name: "boundary" if oblique else "inner solution" for name in chart.curves
                   if name.startswith(("n2_X", "X propagates"))})
    styles.update({name: "boundary" for name in chart.curves if name.startswith(("O propagates", "any propagates"))})
    styles.update({name: "approx" for name in chart.curves if name.split(" ")[0] in names})
    scene = render_chart(
        chart, x_label="$R$ [m]", y_label="$n^2$" if oblique else "$n^2$ (perpendicular)", curve_styles=styles,
        region_text={},
        x_ticks=[0.7, 0.85, 1.0, 1.15, 1.3], y_ticks=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
        note=(f"Example: $R_0 = {p['R0']:g}$ m, $a = {p['a']:g}$ m, $B_0 = {p['B0']:g}$ T, "
              f"$n_0 = 3\\times10^{{19}}$ m$^{{-3}}$, $f = {p['frequency'] / 1e9:g}$ GHz. "
              + (f"${_theta_text(theta)}$, both roots, unnamed; strip: where either propagates" if oblique
                 else "O blue, X red; strips: where each propagates")
              if labels else ""),
    )
    items = list(scene.items)
    if labels:
        # layer names above the box, stacked into rows so that neighbours closer than a label width do not collide
        top = float(chart.to_cm(np.array([R[0], 2.0]))[1]) + 0.12
        placed = sorted((float(chart.to_cm(np.array([r, 0.0]))[0]), name, j)
                        for name, roots in layers.items() for j, r in enumerate(roots))
        merged: List = []
        for x, name, j in placed:
            if merged and x - merged[-1][0] < 0.35:  # closer than a label: one callout for both layers
                x0, text, roles = merged[-1]
                merged[-1] = (0.5 * (x0 + x), f"{text} / {names[name]}", roles + [f"{name} {j}"])
            else:
                merged.append((x, names[name], [f"{name} {j}"]))
        row_end: List[float] = []
        for x, text, roles in merged:
            row = next((k for k, end in enumerate(row_end) if x - end > 0.9), len(row_end))
            if row == len(row_end):
                row_end.append(x)
            row_end[row] = x
            if row:
                items.append(Polyline.of([(x, top - 0.1), (x, top + 0.4 * row)], "leader line", role=roles[0]))
            items.append(Label((x, top + 0.4 * row), text, "small label", anchor="south", role=" + ".join(roles)))
    return Diagram("profile_propagation", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# #1113 section A: the omega-k, n^2-X and n^2-Y views, perpendicular or oblique
# ---------------------------------------------------------------------------


#: smallest angle the oblique views accept [rad]: below it the resonance cone sits within a grid cell of the
#: cyclotron pole and parallel propagation (R and L, no cone) is the better picture
THETA_MIN = np.radians(5.0)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, not {value!r}")
    return float(value)


def _check_theta(theta) -> float:
    theta = _number(theta, "theta")
    if not THETA_MIN <= theta <= 0.5 * np.pi:
        raise ValueError(f"theta must be an angle from 5 degrees to pi/2, not {theta!r}")
    return theta


def _modes(s, theta: float):
    """The two n^2 branches and their names: O/X at theta = pi/2 (tracked by formula), +/- otherwise."""
    if np.isclose(theta, 0.5 * np.pi):
        n2_O, n2_X = perpendicular_refractive_index_squared(s.R, s.L, s.P)
        return {"O": np.asarray(n2_O, dtype=float), "X": np.asarray(n2_X, dtype=float)}
    with np.errstate(invalid="ignore"):  # NaN at the cyclotron pole, masked by the plot
        plus, minus = cold_plasma_refractive_index_squared(s.R, s.L, s.P, theta)
    return {"+": np.asarray(plus, dtype=float), "-": np.asarray(minus, dtype=float)}


#: styles of the branches: O blue and X red, as in ``profile_propagation``; off $\\theta = \\pi/2$ the two
#: algebraic roots swap at the cyclotron layer, so they share one colour rather than imply a mode identity
_BRANCH_STYLE = {"O": "boundary", "X": "inner solution", "+": "boundary", "-": "boundary"}


def _branch_note(theta: float) -> str:
    if np.isclose(theta, 0.5 * np.pi):
        return "O blue, X red"
    return "both roots of $An^4 - Bn^2 + C = 0$, unnamed off $\\theta = \\pi/2$"


def _theta_text(theta: float) -> str:
    if np.isclose(theta, 0.5 * np.pi):
        return "\\theta = \\pi/2"
    return f"\\theta = {np.degrees(theta):.0f}^\\circ"


def wave_dispersion_omega_k(*, omega_pe_over_omega_ce: float = 1.2, theta: float = 0.5 * np.pi,
                            labels: bool = True) -> Diagram:
    r"""The cold electron plasma in the $\omega$-$k$ plane: where each branch exists and where it ends.

    For each frequency the refractive index from the Stix parameters gives
    $k = n\omega/c$ wherever $n^2 > 0$; the branches start at $k = 0$ on their
    cutoffs ($P$, $R$ or $L = 0$) -- except the oblique whistler, which leaves
    the origin -- and run to $k \to \infty$ at a resonance
    ($S = 0$, the upper hybrid, at $\theta = \pi/2$), approaching the light
    line $\omega = ck$ from above at high frequency. ``theta = pi/2`` names
    the branches O and X (``perpendicular_refractive_index_squared``);
    another angle draws both roots of ``cold_plasma_refractive_index_squared``
    in one colour, since its algebraic $\pm$ branches swap at the cyclotron
    layer, and marks the resonances $A = S\sin^2\theta + P\cos^2\theta = 0$.
    Axes in units of $|\Omega_e|$.
    """
    labels = _check_labels(labels)
    theta = _check_theta(theta)
    omega_pe_over_omega_ce = _number(omega_pe_over_omega_ce, "omega_pe_over_omega_ce")
    if not 0.0 < omega_pe_over_omega_ce <= 3.0:
        raise ValueError(f"omega_pe_over_omega_ce must be in (0, 3] to keep the cutoffs on the chart, "
                         f"not {omega_pe_over_omega_ce!r}")
    n_e = _N_REF * omega_pe_over_omega_ce**2
    B = _field_for_Y(1.0, _W_REF)
    w = np.linspace(0.02, 4.0, 3000)
    s = _electron_stix(w * _W_REF, n_e, B)
    k_max = 4.0
    chart = Chart(x_range=(0.0, k_max), y_range=(0.0, 4.0))
    for name, n2 in _modes(s, theta).items():
        with np.errstate(invalid="ignore"):
            k = np.where(np.isfinite(n2) & (n2 > 0), np.sqrt(np.clip(n2, 0, None)) * w, np.nan)
        for i, run in enumerate(_clip_runs(w, k, k_max)):
            chart.curves[f"{name} {i}"] = run[:, ::-1]  # (k, omega)
    chart.curves["light"] = np.array([[0.0, 0.0], [4.0, 4.0]])
    stix = lambda r: _electron_stix(r * _W_REF, n_e, B)
    w_uh = np.sqrt(1.0 + omega_pe_over_omega_ce**2)
    layers = {"P": _bisect(lambda r: stix(r).P, 1e-3, 10.0),
              "L": _bisect(lambda r: stix(r).L, 1e-3, w_uh - 1e-6),
              "R": _bisect(lambda r: stix(r).R, 1.0 + 1e-9, 3.5 * w_uh)}
    if np.isclose(theta, 0.5 * np.pi):
        layers["S"] = _bisect(lambda r: stix(r).S, 1.0 + 1e-9, 3.5 * w_uh)
    else:
        layers.update({("A" if i == 0 else f"A{i + 1}"): r for i, r in enumerate(_resonance_cone(np.linspace(0.02, 4.0, 801), stix, theta))})
    for name, value in layers.items():
        if chart.y_range[0] < value < chart.y_range[1]:
            chart.curves[f"layer {name}"] = np.array([[0.0, value], [k_max, value]])
    chart.parameters.update({"omega_pe_over_omega_ce": omega_pe_over_omega_ce, "theta": theta,
                             **{f"omega_{k}": v for k, v in layers.items()}})
    styles = {n: _BRANCH_STYLE[n.split(" ")[0]] for n in chart.curves if n.split(" ")[0] in _BRANCH_STYLE}
    styles.update({"light": "approx"})
    styles.update({n: "mesh" for n in chart.curves if n.startswith("layer")})
    scene = render_chart(
        chart, x_label="$ck/|\\Omega_e|$", y_label="$\\omega/|\\Omega_e|$", curve_styles=styles, region_text={},
        x_ticks=[0.0, 1.0, 2.0, 3.0, 4.0], y_ticks=[0.0, 1.0, 2.0, 3.0, 4.0],
        note=(f"Cold electrons, $\\omega_{{pe}}/|\\Omega_e| = {omega_pe_over_omega_ce:g}$, ${_theta_text(theta)}$; "
              f"{_branch_note(theta)}; dashed: $\\omega = ck$" if labels else ""),
    )
    items = list(scene.items)
    if labels:
        text = {"P": "$P = 0$", "L": "$L = 0$", "R": "$R = 0$", "S": "$S = 0$ (UH)", "A": "$A = 0$ (resonance)",
                "A2": "$A = 0$ (resonance)"}
        x = float(chart.to_cm(np.array([k_max, 0.0]))[0]) + 0.15
        for name, value in sorted(layers.items(), key=lambda kv: kv[1]):
            if not chart.y_range[0] < value < chart.y_range[1]:
                continue
            y = float(chart.to_cm(np.array([0.0, value]))[1])
            items.append(Label((x, y), text[name], "small label", anchor="west", role=f"layer {name}"))
    return Diagram("wave_dispersion_omega_k", Scene(tuple(items)), model=chart)


def _index_chart(axis: str, values, s, theta, layers, fixed_text, limit=3.0):
    chart = Chart(x_range=(float(values[0]), float(values[-1])), y_range=(-limit, limit))
    for name, n2 in _modes(s, theta).items():
        for i, run in enumerate(_clip_runs(values, n2, limit)):
            chart.curves[f"{name} {i}"] = run
    chart.curves["zero"] = np.array([[values[0], 0.0], [values[-1], 0.0]])
    for name, v in layers.items():
        if values[0] < v < values[-1]:
            chart.curves[f"layer {name}"] = np.array([[v, -limit], [v, limit]])
            chart.points[f"layer {name}"] = (v, 0.0)
    styles = {n: _BRANCH_STYLE[n.split(" ")[0]] for n in chart.curves if n.split(" ")[0] in _BRANCH_STYLE}
    styles.update({"zero": "approx"})
    styles.update({n: "mesh" for n in chart.curves if n.startswith("layer")})
    return chart, styles, f"{fixed_text}, ${_theta_text(theta)}$; {_branch_note(theta)}"


def _layer_labels(chart, layers, text, limit=3.0) -> List:
    items: List = []
    top = float(chart.to_cm(np.array([chart.x_range[0], limit]))[1]) + 0.15
    for k, (name, v) in enumerate(sorted(layers.items(), key=lambda kv: kv[1])):
        if chart.x_range[0] < v < chart.x_range[1]:
            items.append(Label((float(chart.to_cm(np.array([v, 0.0]))[0]), top + 0.42 * (k % 2)), text[name],
                               "small label", anchor="south", role=f"layer {name}"))
    return items


_LAYER_TEXT = {"P": "$P{=}0$", "R": "$R{=}0$", "L": "$L{=}0$", "S": "$S{=}0$", "ECR": "$Y{=}1$",
               "A": "$A{=}0$", "A2": "$A{=}0$"}


def _resonance_cone(grid, stix_at, theta: float) -> List[float]:
    """Zeros of $A = S\\sin^2\\theta + P\\cos^2\\theta$ along ``grid``: where an oblique branch has $n^2 \\to \\infty$.

    Sign changes across a pole of $S$ (the cyclotron layer) are not zeros and are dropped.
    """
    A = lambda v: float(np.squeeze(stix_at(v).S * np.sin(theta) ** 2 + stix_at(v).P * np.cos(theta) ** 2))
    values = np.array([A(v) for v in grid])
    roots = []
    for i in np.where(np.isfinite(values[:-1]) & np.isfinite(values[1:])
                      & (np.sign(values[:-1]) != np.sign(values[1:])))[0]:
        root = _bisect(A, grid[i], grid[i + 1])
        if abs(A(root)) < 1e-6:
            roots.append(root)
    return roots


def refractive_index_vs_X(*, Y: float = 0.5, theta: float = 0.5 * np.pi, labels: bool = True) -> Diagram:
    r"""$n^2$ against $X = \omega_{pe}^2/\omega^2$ at fixed $Y$: rising density at one frequency and field.

    The cutoffs are the zeros of the Stix parameters along $X$ -- $P = 0$ at
    $X = 1$, $R = 0$ at $X = 1 - Y$, $L = 0$ at $X = 1 + Y$ -- and at
    $\theta = \pi/2$ the upper-hybrid resonance $S = 0$ at $X = 1 - Y^2$,
    all located by bracketing ``stix_parameters`` rather than from these
    closed forms. Branches as in ``wave_dispersion_omega_k``.
    """
    labels = _check_labels(labels)
    theta = _check_theta(theta)
    Y = _number(Y, "Y")
    if not 0.0 < Y < 1.0:
        raise ValueError(f"Y must be between 0 and 1 (below the cyclotron resonance), not {Y!r}")
    X = np.linspace(0.0, 2.5, 2000)
    B = _field_for_Y(Y, _W_REF)
    s = _electron_stix(_W_REF, _density_for_X(X, _W_REF), B)
    at = lambda name: (lambda x: getattr(_electron_stix(_W_REF, _density_for_X(x, _W_REF), B), name))
    layers = {"P": _bisect(at("P"), 0.5, 2.0), "R": _bisect(at("R"), 1e-6, 1.0),
              "L": _bisect(at("L"), 1.0, 2.5)}
    if np.isclose(theta, 0.5 * np.pi):
        layers["S"] = _bisect(at("S"), 1e-6, 1.0 - 1e-9)
    else:
        layers.update({("A" if i == 0 else f"A{i + 1}"): r for i, r in enumerate(_resonance_cone(
            np.linspace(1e-6, 2.5, 501), lambda x: _electron_stix(_W_REF, _density_for_X(x, _W_REF), B), theta))})
    chart, styles, note = _index_chart("X", X, s, theta, layers, f"Cold electrons, $Y = {Y:g}$")
    chart.parameters.update({"Y": Y, "theta": theta, **{f"X_{k}": v for k, v in layers.items()}})
    scene = render_chart(chart, x_label="$X = \\omega_{pe}^2/\\omega^2$", y_label="$n^2$", curve_styles=styles,
                         region_text={}, x_ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
                         y_ticks=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], note=note if labels else "")
    items = list(scene.items) + (_layer_labels(chart, layers, _LAYER_TEXT) if labels else [])
    return Diagram("refractive_index_vs_X", Scene(tuple(items)), model=chart)


def refractive_index_vs_Y(*, X: float = 0.5, theta: float = 0.5 * np.pi, labels: bool = True) -> Diagram:
    r"""$n^2$ against $Y = |\Omega_e|/\omega$ at fixed $X$: rising field at one frequency and density.

    At $\theta = \pi/2$ the O branch ($n^2 = P = 1 - X$) does not depend on $Y$; the other
    branch has the $R = 0$ cutoff at $Y = 1 - X$ and, at $\theta = \pi/2$,
    the upper-hybrid resonance $S = 0$ at $Y = \sqrt{1 - X}$; $Y = 1$ is the
    electron cyclotron resonance, where $R$ has its pole. Located by
    bracketing ``stix_parameters``. Branches as in ``wave_dispersion_omega_k``.
    """
    labels = _check_labels(labels)
    theta = _check_theta(theta)
    X = _number(X, "X")
    if not 0.0 < X < 1.0:
        raise ValueError(f"X must be between 0 and 1 (above the O cutoff), not {X!r}")
    Yv = np.linspace(0.0, 2.0, 2001)[1:]
    n_e = _density_for_X(X, _W_REF)
    s = _electron_stix(_W_REF, n_e, _field_for_Y(Yv, _W_REF))
    at = lambda name: (lambda y: getattr(_electron_stix(_W_REF, n_e, _field_for_Y(y, _W_REF)), name))
    layers = {"R": _bisect(at("R"), 1e-6, 1.0 - 1e-9), "ECR": 1.0}
    if np.isclose(theta, 0.5 * np.pi):
        layers["S"] = _bisect(at("S"), 1e-6, 1.0 - 1e-9)
    else:
        layers.update({("A" if i == 0 else f"A{i + 1}"): r for i, r in enumerate(_resonance_cone(
            np.linspace(1e-3, 2.0, 801), lambda y: _electron_stix(_W_REF, n_e, _field_for_Y(y, _W_REF)), theta))})
    chart, styles, note = _index_chart("Y", Yv, s, theta, layers, f"Cold electrons, $X = {X:g}$")
    chart.parameters.update({"X": X, "theta": theta, **{f"Y_{k}": v for k, v in layers.items()}})
    scene = render_chart(chart, x_label="$Y = |\\Omega_e|/\\omega$", y_label="$n^2$", curve_styles=styles,
                         region_text={}, x_ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
                         y_ticks=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], note=note if labels else "")
    items = list(scene.items) + (_layer_labels(chart, layers, _LAYER_TEXT) if labels else [])
    return Diagram("refractive_index_vs_Y", Scene(tuple(items)), model=chart)
