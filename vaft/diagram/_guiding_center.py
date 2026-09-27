"""Guiding-centre invariants and toroidal symmetry (#1092).

The global view of the orbit physics whose local view is ``curvature_drift``
and ``toroidal_drift``:

``guiding_center_invariants``
    gyration, bounce and toroidal drift, each with its invariant --
    $\\mu$, $J_\\parallel$, $P_\\phi$ -- and the ordering
    $\\Omega_c \\gg \\omega_b \\gg \\omega_d$;
``canonical_toroidal_momentum``
    one banana orbit built from $P_\\phi$ conservation, with the flux and
    mechanical parts of ``guiding_center_toroidal_momentum`` at points along
    it summing to the same $P_\\phi$;
``toroidal_symmetry_breaking``
    $P_\\phi(t)$ constant in axisymmetry, oscillating under a non-resonant 3-D
    field and drifting secularly where ``bounce_harmonic_detuning`` vanishes.

Units are normalised ($q = m = B_0 = 1$); widths are schematic. No orbit
integrator is added: the orbit follows from the two invariants directly.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List

import numpy as np

from vaft.formula.equilibrium import vacuum_toroidal_field
from vaft.formula.particle import (
    bounce_harmonic_detuning,
    guiding_center_toroidal_momentum,
    parallel_speed_from_mu,
)

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the normalised torus: major radius, minor radius, safety factor of psi(r) = B0 r^2 / (2 q)
_R0, _A, _QS, _B0 = 3.0, 1.0, 2.0, 1.0
#: start of the orbit (outboard midplane), its speed and pitch there
_R_START, _SPEED, _PITCH = 0.55, 0.012, 0.18
_CM = 2.4


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _psi(r):
    """Poloidal flux per radian of the large-aspect-ratio circular equilibrium, $B_0r^2/(2q)$."""
    return _B0 * np.asarray(r, dtype=float) ** 2 / (2.0 * _QS)


def _r_of_psi(psi):
    return np.sqrt(np.maximum(2.0 * _QS * np.asarray(psi, dtype=float) / _B0, 0.0))


def _leg_radius(P: float, theta: float, sign: float, B_start: float) -> float:
    """Radius on one leg at ``theta``: the root of $P - q\\psi(r) - mv_\\parallel(r, \\theta)R(r, \\theta)$."""
    from scipy.optimize import brentq

    def f(r):
        R = _R0 + r * math.cos(theta)
        speed = parallel_speed_from_mu(_SPEED, _PITCH, B_start, float(vacuum_toroidal_field(_B0, _R0, R)))
        return P - float(_psi(r)) - sign * speed * R if np.isfinite(speed) else np.nan

    def edge(r_in, r_out):
        """The boundary of the reachable region between a reachable and an unreachable radius."""
        for _ in range(80):
            mid = 0.5 * (r_in + r_out)
            if np.isfinite(f(mid)):
                r_in = mid
            else:
                r_out = mid
        return r_in

    grid = np.linspace(0.3, 0.9, 241)
    values = np.array([f(r) for r in grid])
    for i in range(len(grid) - 1):
        lo, hi, a, b = grid[i], grid[i + 1], values[i], values[i + 1]
        if np.isfinite(a) != np.isfinite(b):
            # near a bounce tip the root sits at the edge of the reachable region, where v_par = 0
            e = edge(lo, hi) if np.isfinite(a) else edge(hi, lo)
            lo, hi = (lo, e) if np.isfinite(a) else (e, hi)
            a, b = f(lo), f(hi)
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0.0:
            return float(brentq(f, lo, hi, xtol=1e-13)) if a != b else float(lo)
    return float("nan")


def banana_orbit(n: int = 400) -> Dict[str, np.ndarray]:
    """One bounce of a trapped guiding centre; see ``_banana_orbit``. Cached: it takes no physics inputs."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in _banana_orbit(int(n)).items()}


@lru_cache(maxsize=4)
def _banana_orbit(n: int) -> Dict[str, np.ndarray]:
    """One bounce of a trapped guiding centre, from $\\mu$ and $P_\\phi$ conservation, in time order.

    Along the orbit $v_\\parallel$ follows from $\\mu$ and energy
    (``parallel_speed_from_mu``) in $B = B_0R_0/R$, and $\\psi$ from
    $P_\\phi = q\\psi + mv_\\parallel R$ held fixed
    (``guiding_center_toroidal_momentum`` with $b_\\phi = 1$): at each poloidal
    angle the radius is the bracketed root of the two together. The tips are
    exact: $v_\\parallel = 0$ there, so $\\psi = P_\\phi$ and the mirror sets the angle.
    The path starts outboard with $v_\\parallel > 0$, runs to the upper tip,
    back along the other leg, to the lower tip and home.
    """
    R_start = _R0 + _R_START
    B_start = float(vacuum_toroidal_field(_B0, _R0, R_start))
    P = guiding_center_toroidal_momentum(1.0, 1.0, _SPEED * _PITCH, R_start, 1.0, float(_psi(_R_START)))
    r_tip = float(_r_of_psi(P))
    R_tip = _R0 * _B0 * (1.0 - _PITCH**2) / B_start  # where the mirror stops the particle
    theta_tip = math.acos((R_tip - _R0) / r_tip)
    theta = theta_tip * (1.0 - (1.0 - np.linspace(0.0, 1.0, n)) ** 2)[:-1]  # denser towards the tip
    theta = np.concatenate([theta[:1], theta[1:]])
    legs = {}
    for sign in (1.0, -1.0):
        r = np.array([_leg_radius(P, t, sign, B_start) for t in theta])
        legs[sign] = np.append(r, r_tip)
    th = np.append(theta, theta_tip)

    def speed(r, t):
        R = _R0 + r * np.cos(t)
        return parallel_speed_from_mu(_SPEED, _PITCH, B_start, vacuum_toroidal_field(_B0, _R0, R))

    up_r, down_r = legs[1.0], legs[-1.0]
    up_v, down_v = np.nan_to_num(speed(up_r, th)), -np.nan_to_num(speed(down_r, th))
    # time order: up leg to the upper tip, back on the down leg, on to the lower tip, home on the up leg
    theta_path = np.concatenate([th, th[::-1][1:], -th[1:], -th[::-1][1:]])
    r_path = np.concatenate([up_r, down_r[::-1][1:], down_r[1:], up_r[::-1][1:]])
    v_path = np.concatenate([up_v, down_v[::-1][1:], down_v[1:], up_v[::-1][1:]])
    R = _R0 + r_path * np.cos(theta_path)
    Z = r_path * np.sin(theta_path)
    return {"theta": theta_path, "r": r_path, "R": R, "Z": Z, "v_par": v_path, "psi": _psi(r_path), "P_phi": P,
            "bounce_theta": theta_tip, "r_tip": r_tip}


def _xy(R, Z):
    return np.stack([(np.asarray(R) - _R0) * _CM, np.asarray(Z) * _CM], axis=-1)


def _surfaces(items: List, radii=(0.35, 0.55, 0.75, 1.0)) -> None:
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    for r in radii:
        items.append(Polyline.of(np.stack([r * np.cos(t), r * np.sin(t)], -1) * _CM,
                                 "lcfs" if r == _A else "surface", role="flux_surface", closed=True))


# ---------------------------------------------------------------------------
# the three invariants
# ---------------------------------------------------------------------------


def guiding_center_invariants(*, labels: bool = True) -> Diagram:
    r"""Gyration, bounce and toroidal drift, and the invariant each carries.

    Left, a trapped guiding centre (the banana of ``banana_orbit``) and, along
    part of it, the particle gyrating about it -- Larmor radius exaggerated.
    Right, the three motions from fastest to slowest with their invariants:
    $\mu = mv_\perp^2/(2B)$ for gyration, $J_\parallel = \oint p_\parallel\,dl$
    for the bounce, $P_\phi$ for the toroidal drift. Each is conserved only
    while its motion is fast against changes of the field it sees
    ($\Omega_c \gg \omega_b \gg \omega_d$); $P_\phi$ needs axisymmetry.
    """
    labels = _check_labels(labels)
    orbit = banana_orbit()
    items: List = []
    _surfaces(items)
    gc = _xy(orbit["R"], orbit["Z"])
    items.append(Polyline.of(gc, "field line", role="guiding_center", closed=True))
    # the particle about part of the upper leg: guiding centre plus a gyration, schematic radius
    k = np.arange(len(gc) // 16, len(gc) // 4)
    phase = np.linspace(0.0, 9.0 * 2.0 * math.pi, len(k))
    rho = 0.1
    particle = gc[k] + rho * np.stack([np.cos(phase), np.sin(phase)], -1)
    items.append(Polyline.of(particle, "orbit ion", role="particle_orbit"))
    x0 = 1.6 * _A * _CM
    rows = [
        ("gyro", "mu", "gyration, $\\Omega_c$\\\\ $\\mu = mv_\\perp^2/(2B)$"),
        ("bounce", "j_parallel", "bounce / transit, $\\omega_b$\\\\ $J_\\parallel = \\oint p_\\parallel\\,dl$"),
        ("drift", "p_phi", "toroidal drift, $\\omega_d$\\\\ $P_\\phi = q\\psi + mv_\\parallel Rb_\\phi$"),
    ]
    boxes = []
    for i, (motion, invariant, text) in enumerate(rows):
        b = box(x0 + 3.0, 1.9 - 1.9 * i, 5.4, 1.35, text, role=f"invariant:{invariant}", latex=True)
        boxes.append(b)
        items += list(b.items)
        items.append(Label((x0 + 5.9, 1.9 - 1.9 * i), {"gyro": "fastest", "bounce": "", "drift": "slowest"}[motion]
                           if labels else "", "small label", anchor="west", role=f"motion:{motion}"))
    items += [connector(boxes[0], boxes[1], role="ordering"), connector(boxes[1], boxes[2], role="ordering")]
    if labels:
        top = _A * _CM
        items += [
            Label((0.0, top + 0.15), "guiding centre (banana) and particle", "small label", anchor="south",
                  role="guiding_center"),
            Label((x0 + 3.0, 3.1), "$\\Omega_c \\gg \\omega_b \\gg \\omega_d$", "label", anchor="south",
                  role="ordering"),
            Label((0.5 * (x0 + 3.0), -top - 0.5),
                  "Each invariant holds while its motion is fast against the field it sees; $P_\\phi$ needs axisymmetry",
                  "note", anchor="north", role="note"),
        ]
    model = {"orbit": orbit, "rho": rho, "invariants": ("mu", "j_parallel", "p_phi")}
    return Diagram("guiding_center_invariants", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# P_phi decomposition
# ---------------------------------------------------------------------------


def canonical_toroidal_momentum(phase: float = 0.45, *, labels: bool = True) -> Diagram:
    r"""$P_\phi = q\psi + mv_\parallel Rb_\phi$ along one banana: the parts trade, the sum stays.

    Three points of the orbit of ``banana_orbit`` -- the outboard start, the
    upper bounce tip and the point a fraction ``phase`` of the way round --
    with bars of their flux part $q\psi$ and mechanical part $mv_\parallel Rb_\phi$
    from ``guiding_center_toroidal_momentum``. Where $v_\parallel$ falls the flux
    part rises: the guiding centre crosses surfaces, which is the orbit width.
    ``phase`` is the state coordinate a later animation would step.
    """
    try:
        phase = float(phase)
    except (TypeError, ValueError):
        raise ValueError(f"phase must be a number in [0, 1), not {phase!r}") from None
    if not 0.0 <= phase < 1.0:
        raise ValueError(f"phase must lie in [0, 1), not {phase!r}")
    labels = _check_labels(labels)
    orbit = banana_orbit()
    n = len(orbit["theta"])
    tip = int(np.argmax(orbit["theta"]))  # the upper bounce tip, v_par = 0 there
    picks = {"A": 0, "B": tip, "C": int(round(phase * (n - 1)))}
    items: List = []
    _surfaces(items)
    items.append(Polyline.of(_xy(orbit["R"], orbit["Z"]), "field line", role="guiding_center", closed=True))
    parts = {}
    for name, i in picks.items():
        flux = float(orbit["psi"][i])
        mech = float(orbit["v_par"][i] * orbit["R"][i])
        total = guiding_center_toroidal_momentum(1.0, 1.0, orbit["v_par"][i], orbit["R"][i], 1.0, orbit["psi"][i])
        parts[name] = {"flux": flux, "mechanical": mech, "P_phi": total}
        at = tuple(_xy(orbit["R"][i], orbit["Z"][i]))
        items.append(Marker(at, "o", "opoint", role=f"point:{name}"))
        if labels:
            items.append(Label((at[0] + 0.12, at[1] + 0.12), name, "small label", anchor="south west",
                               role=f"point:{name}"))
    # the bars: each point's change of the two parts from the bounce tip B, where v_par = 0.
    # They are equal and opposite -- the sum, P_phi, does not move.
    x0 = 1.5 * _A * _CM + 3.2
    ref = parts["B"]
    changes = {k: (v["flux"] - ref["flux"], v["mechanical"] - ref["mechanical"]) for k, v in parts.items()}
    scale = 2.8 / max(max(abs(a), abs(b)) for a, b in changes.values())
    for j, (name, (d_flux, d_mech)) in enumerate(changes.items()):
        y = 1.6 - 1.6 * j
        if abs(d_flux) * scale > 0.05:  # at B both changes are zero: nothing to draw
            items += [
                Arrow((x0, y + 0.14), (x0 + d_flux * scale, y + 0.14), "drift ion", role=f"bar:flux:{name}"),
                Arrow((x0 + d_flux * scale, y - 0.14), (x0 + (d_flux + d_mech) * scale, y - 0.14), "drift",
                      role=f"bar:mechanical:{name}"),
            ]
        else:
            items.append(Marker((x0, y), "o", "opoint", role=f"bar:flux:{name}"))
        if labels:
            items.append(Label((x0 - 3.35, y), name, "small label", anchor="east", role=f"bar:{name}"))
    items.append(Polyline.of([(x0, 2.2), (x0, -2.2 - 0.4)], "rational", role="p_phi_level"))
    if labels:
        top = _A * _CM
        items += [
            Label((x0, 2.3), "$\\Delta P_\\phi = 0$", "label", anchor="south", role="p_phi_level"),
            Label((x0 - 3.3, -2.8), "change from B: blue $\\Delta(q\\psi)$, red $\\Delta(mv_\\parallel Rb_\\phi)$, returning to $\\Delta P_\\phi = 0$",
                  "small label", anchor="north west", role="bar:key"),
            Label((0.5 * (x0 + 1.0), -top - 1.3), f"$\\displaystyle {formula_equation(guiding_center_toroidal_momentum)}$",
                  "formula box", anchor="north", role="equations"),
            _note_label("Axisymmetric: $\\partial\\mathcal{L}/\\partial\\phi = 0$ keeps $P_\\phi$ fixed, so $\\psi$ moves "
                        "with $v_\\parallel$; normalised units", 0.5 * (x0 + 1.0), -top - 2.5),
        ]
    model = {"orbit": orbit, "points": picks, "parts": parts, "changes": changes, "phase": phase}
    return Diagram("canonical_toroidal_momentum", Scene(tuple(items)), model=model)


def _note_label(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


# ---------------------------------------------------------------------------
# symmetry breaking
# ---------------------------------------------------------------------------


def toroidal_symmetry_breaking(*, labels: bool = True) -> Diagram:
    r"""Axisymmetry keeps $P_\phi$; a 3-D field changes it, secularly only at resonance.

    With a toroidal-$n$ perturbation, $dP_\phi/dt \propto \cos(\Delta\omega_\mathrm{BH}t)$
    along the orbit, $\Delta\omega_\mathrm{BH}$ from ``bounce_harmonic_detuning``:
    away from resonance $P_\phi$ oscillates with no net change, at
    $\Delta\omega_\mathrm{BH} = 0$ it drifts linearly. The drift of one orbit
    is not NTV; torque and transport need the kinetic response of the whole
    distribution (#1111).
    """
    labels = _check_labels(labels)
    t = np.linspace(0.0, 60.0, 1201)
    kick = 0.02
    detuned = bounce_harmonic_detuning(1.0, 0.25, 0.1, l=1, n=2)  # 1 - 0.7 = 0.3: three periods shown
    resonant = bounce_harmonic_detuning(1.0, 0.375, 0.125, l=1, n=2)  # exactly 0
    curves = {"axisymmetric": np.ones_like(t),
              "non_resonant": 1.0 + kick * np.sin(detuned * t) / detuned,
              "resonant": 1.0 + kick * t}
    chart = Chart(x_range=(0.0, 60.0), y_range=(0.6, 2.5))
    for name, c in curves.items():
        chart.curves[name] = np.stack([t, c], -1)
    chart.parameters.update({"detuning_non_resonant": detuned, "detuning_resonant": resonant, "kick": kick})
    scene = render_chart(chart, x_label="$\\omega_b t$", y_label="$P_\\phi/P_{\\phi,0}$",
                         curve_styles={"axisymmetric": "approx", "non_resonant": "orbit ion", "resonant": "orbit electron"},
                         region_text={}, x_ticks=(0.0, 20.0, 40.0, 60.0), y_ticks=(1.0, 2.0))
    items: List = []
    if labels:
        items += [
            Label((0.25, CHART_HEIGHT - 0.1), "dark: 3-D, $\\Delta\\omega_\\mathrm{BH} = 0$: secular change",
                  "small label", anchor="north west", role="resonant"),
            Label((0.25, CHART_HEIGHT - 0.6), "light: 3-D, $\\Delta\\omega_\\mathrm{BH} \\ne 0$: oscillates, no net change",
                  "small label", anchor="north west", role="non_resonant"),
            Label((0.25, CHART_HEIGHT - 1.1), "dashed: axisymmetric, $\\partial\\mathcal{L}/\\partial\\phi = 0$",
                  "small label", anchor="north west", role="axisymmetric"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(bounce_harmonic_detuning)}$",
                  "formula box", anchor="north", role="equations"),
            _note_label("One orbit's response; NTV needs the kinetic response of the distribution", CHART_WIDTH / 2,
                        -2.6),
        ]
    return Diagram("toroidal_symmetry_breaking", scene + Scene(tuple(items)), model=chart)
