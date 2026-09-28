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

from vaft.formula.constants import ME, QE
from vaft.formula.waves import (
    cma_coordinates,
    perpendicular_refractive_index_squared,
    propagation_regime,
    stix_parameters,
)

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Label, Polyline, Scene

_Q, _M = np.array([-QE]), np.array([ME])
#: an electron density at which omega_pe = 2 pi x 1 GHz, the frequency unit of the normalised diagrams [m^-3]
_N_REF = (2.0 * np.pi * 1e9) ** 2 * 8.8541878128e-12 * ME / QE**2
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
    ratio = np.linspace(0.05, 3.5, 1400)
    s = _electron_stix(ratio * _W_REF, n_e, B)
    _, n2_X = perpendicular_refractive_index_squared(s.R, s.L, s.P)
    stix = lambda r: _electron_stix(r * _W_REF, n_e, B)
    w_uh_guess = np.sqrt(1.0 + omega_pe_over_omega_ce**2)
    omega_L = _bisect(lambda r: stix(r).L, 1e-3, w_uh_guess - 1e-6)
    omega_UH = _bisect(lambda r: stix(r).S, 1.0 + 1e-9, 3.5 * w_uh_guess)
    omega_R = _bisect(lambda r: stix(r).R, 1.0 + 1e-9, 3.5 * w_uh_guess)
    limit = 6.0
    chart = Chart(x_range=(0.0, 3.5), y_range=(-limit, limit))
    for i, run in enumerate(_clip_runs(ratio, np.asarray(n2_X), limit)):
        chart.curves[f"n2_X {i}"] = run
    chart.curves["zero"] = np.array([[0.0, 0.0], [3.5, 0.0]])
    for name, value in (("omega_L", omega_L), ("omega_UH", omega_UH), ("omega_R", omega_R)):
        chart.curves[name] = np.array([[value, -limit], [value, limit]])
        chart.points[name] = (value, 0.0)
    chart.labels.update({"gap1": (omega_L / 2, -3.2), "prop": (3.05, 2.2),
                         "prop_low": (0.5 * (omega_L + omega_UH) - 0.06, 2.7)})
    chart.parameters.update({"omega_pe_over_omega_ce": omega_pe_over_omega_ce, "omega_L": omega_L,
                             "omega_UH": omega_UH, "omega_R": omega_R})
    styles: Dict[str, str] = {name: "boundary" for name in chart.curves if name.startswith("n2_X")}
    styles.update({"zero": "approx", "omega_L": "approx", "omega_UH": "approx", "omega_R": "approx"})
    scene = render_chart(
        chart, x_label="$\\omega/|\\Omega_e|$", y_label="$n_X^2 = RL/S$", curve_styles=styles,
        region_text={"gap1": "\\small evanescent", "prop": "\\small propagating",
                     "prop_low": "\\small propagating"} if labels else {},
        x_ticks=[0.0, 1.0, 2.0, 3.0], y_ticks=[-6.0, -3.0, 0.0, 3.0, 6.0],
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
    chart.labels.update({"P": (1.15, 1.85), "R": (0.3, 0.55), "L": (2.2, 1.4), "S": (0.86, 0.5),
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
            if name in ("R", "S") and abs(f(root)) > 1.0:
                continue
            found.append(root)
        layers[name] = found
    return layers


def profile_propagation(*, labels: bool = True) -> Diagram:
    r"""O- and X-mode $n^2$ along the midplane of an example tokamak, with its cutoff and resonance layers.

    Example parameters (``EXAMPLE_PROFILE``, illustrative, not a device):
    $B = B_0R_0/R$, a parabolic $n_e$, one frequency at the on-axis
    electron cyclotron frequency. The layers -- O cutoff $P = 0$, X cutoffs
    $R = 0$ and $L = 0$, upper-hybrid $S = 0$, cyclotron $Y = 1$ -- are
    sign changes of the Stix parameters along the profile, refined by
    bisection; ``propagation_regime`` classifies the samples between them.
    """
    labels = _check_labels(labels)
    p = EXAMPLE_PROFILE
    R = np.linspace(p["R0"] - p["a"], p["R0"] + p["a"], 1201)
    _, _, _, n2_O, n2_X = profile_quantities(R, p)
    layers = profile_layers(p)
    limit = 3.0
    chart = Chart(x_range=(R[0], R[-1]), y_range=(-limit, 2.0))
    chart.curves["n2_O"] = np.stack([R, n2_O], axis=-1)[n2_O >= -limit]
    for i, run in enumerate(_clip_runs(R, n2_X, limit)):
        chart.curves[f"n2_X {i}"] = run
    chart.curves["zero"] = np.array([[R[0], 0.0], [R[-1], 0.0]])
    names = {"P": "$P{=}0$", "R": "$R{=}0$", "L": "$L{=}0$", "S": "UH", "ECR": "ECR"}
    for name, roots in layers.items():
        for j, r in enumerate(roots):
            key = f"{name} {j}"
            chart.curves[key] = np.array([[r, -limit], [r, 2.0]])
            chart.points[key] = (r, 0.0)
    regime_O = propagation_regime(np.where(np.isfinite(n2_O), n2_O, np.inf))
    chart.parameters.update({k: float(v) for k, v in p.items()})
    chart.parameters["O_propagating_fraction"] = float(np.mean(regime_O == "propagating"))
    styles: Dict[str, str] = {"zero": "approx", "n2_O": "boundary"}
    styles.update({name: "inner solution" for name in chart.curves if name.startswith("n2_X")})
    styles.update({name: "approx" for name in chart.curves if name.split(" ")[0] in names})
    scene = render_chart(
        chart, x_label="$R$ [m]", y_label="$n^2$ (perpendicular)", curve_styles=styles, region_text={},
        x_ticks=[0.7, 0.85, 1.0, 1.15, 1.3], y_ticks=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
        note=(f"Example: $R_0 = {p['R0']:g}$ m, $a = {p['a']:g}$ m, $B_0 = {p['B0']:g}$ T, "
              f"$n_0 = 3\\times10^{{19}}$ m$^{{-3}}$, $f = {p['frequency'] / 1e9:g}$ GHz. O mode blue, X mode red"
              if labels else ""),
    )
    items = list(scene.items)
    if labels:
        # layer names above the box, stacked into rows so that neighbours closer than a label width do not collide
        top = float(chart.to_cm(np.array([R[0], 2.0]))[1]) + 0.12
        placed = sorted((float(chart.to_cm(np.array([r, 0.0]))[0]), name, j)
                        for name, roots in layers.items() for j, r in enumerate(roots))
        row_end: List[float] = []
        for x, name, j in placed:
            row = next((k for k, end in enumerate(row_end) if x - end > 0.9), len(row_end))
            if row == len(row_end):
                row_end.append(x)
            row_end[row] = x
            if row:
                items.append(Polyline.of([(x, top - 0.1), (x, top + 0.4 * row)], "leader line", role=f"{name} {j}"))
            items.append(Label((x, top + 0.4 * row), names[name], "small label", anchor="south",
                               role=f"{name} {j}"))
    return Diagram("profile_propagation", Scene(tuple(items)), model=chart)
