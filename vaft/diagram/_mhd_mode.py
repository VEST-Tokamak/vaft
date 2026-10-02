"""A prescribed ideal displacement on an arbitrary equilibrium (#1209).

``kink_mode``
    the flux surfaces of an equilibrium moved along their normals by
    $\\xi_n = A\\,a\\,F(\\rho)\\,\\mathrm{Re}\\sum_m c_m e^{i(m\\theta^* - n\\phi + \\varphi_0)}$,
    with a named schematic envelope $F$; the phase lives in the
    straight-field-line angle $\\theta^*$, the drawing on the real surfaces.

Model class: synthetic parameterization. The envelope is prescribed, not an
eigenfunction: nothing here solves $-\\rho\\omega^2\\boldsymbol\\xi =
\\mathbf F[\\boldsymbol\\xi]$. The cylindrical reference view of internal and
external kinks is ``internal_external_kink`` (#1072); stability-code
eigenfunctions belong to result plotting.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from vaft.formula.equilibrium import flux_perturbation_from_normal_displacement

from ._equations import formula_equation
from ._equilibrium_geometry import EquilibriumGeometry, equilibrium_geometry
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

ENVELOPES = ("internal", "global", "edge")
#: surfaces drawn, as rho = sqrt(psi_N); the last stands in for the boundary
SURFACES = tuple(np.round(np.linspace(0.12, 0.84, 7), 4)) + (0.95,)
#: drawing scale [cm per m]
_S = 12.0
#: half-width of the internal envelope's edge, in rho
_EDGE = 0.035


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def edge_width(amplitude: float) -> float:
    """Half-width of the internal envelope's edge, in $\\rho$: at least 0.035, and wider than the amplitude.

    The displacement changes by $A\\,a$ across the edge, so with $|dF/d\\rho|
    \\le 1/(2w)$ the surfaces keep their order while $A\\,a/(2w)$ stays below
    their spacing $\\sim a\\,d\\rho$: $w = \\max(0.035, 0.9A)$ keeps a margin.
    """
    return max(_EDGE, 0.9 * float(amplitude))


def envelope(kind: str, rho, *, m: int, rho_s: Optional[float], amplitude: float = 0.0) -> np.ndarray:
    """The schematic radial envelope $F(\\rho)$, 1 at its maximum.

    ``internal``: a top hat inside the resonant surface $\\rho_s$, smoothed
    over $\\pm$ ``edge_width(amplitude)`` and times $(\\rho/\\rho_s)^{m-1}$ so
    that it is regular at the axis; ``global``: $\\rho^{m-1}$, reaching
    the boundary (for $m = 1$ a rigid shift of every surface); ``edge``:
    $\\rho^{4m}$, confined to the edge.
    """
    rho = np.asarray(rho, dtype=float)
    if kind == "internal":
        # regular at the axis: an m >= 2 displacement vanishes there like rho^(m - 1)
        regular = np.minimum(rho / rho_s, 1.0) ** (m - 1)
        return regular * 0.5 * (1.0 - np.tanh((rho - rho_s) / edge_width(amplitude)))
    if kind == "global":
        return rho ** (m - 1)
    return rho ** (4 * m)


def displacement(geom: EquilibriumGeometry, rho: float, *, n: int, amplitude: float, harmonics: Dict[int, complex],
                 kind: str, rho_s: Optional[float], phase: float, phi: float = 0.0) -> dict:
    """One displaced surface: $\\xi_n$ on it, and the moved points.

    For a single $m = 1$ harmonic the surface is shifted rigidly, by the vector
    whose normal component is $\\xi_n$ -- the $m = 1$ kink is a shift, and
    moving each point along its own normal would turn a small inner surface
    into a limacon. Otherwise the points move by $\\xi_n\\hat{\\mathbf n}$.
    """
    s = geom.surface(float(rho))
    m_lead = min(harmonics)
    scale = amplitude * geom.minor_radius * envelope(kind, rho, m=m_lead, rho_s=rho_s, amplitude=amplitude)
    if set(harmonics) == {1}:
        c = harmonics[1]
        # Re[c e^{i(theta* - n phi + phase)}] peaks at theta* = -(phase - n phi + arg c): shift towards that point
        peak = -(phase - n * phi + np.angle(c))
        axis = np.array(geom.axis)
        toward = np.array(geom.point(0.3, peak)) - axis
        e = toward / np.linalg.norm(toward)
        shift = scale * abs(c) * e
        xi = shift[0] * s.normal_R + shift[1] * s.normal_Z
        return {"surface": s, "xi": xi, "R": s.R + shift[0], "Z": s.Z + shift[1], "shift": shift}
    helical = sum(c * np.exp(1j * (m * s.theta_star - n * phi + phase)) for m, c in harmonics.items())
    xi = scale * np.real(helical)
    return {"surface": s, "xi": xi, "R": s.R + xi * s.normal_R, "Z": s.Z + xi * s.normal_Z}


def _validate(m, n, amplitude, radial_profile, harmonics):
    for name, v in (("m", m), ("n", n)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 1 <= v <= 10:
            raise ValueError(f"{name} must be an integer from 1 to 10, not {v!r}")
    if not (isinstance(amplitude, (int, float)) and 0.0 <= amplitude <= 0.15):
        raise ValueError(f"amplitude must lie in [0, 0.15] (units of the minor radius), not {amplitude!r}")
    if radial_profile not in ENVELOPES:
        raise ValueError(f"radial_profile must be one of {ENVELOPES}, not {radial_profile!r}")
    if harmonics is None:
        return {int(m): 1.0 + 0.0j}
    if not isinstance(harmonics, dict) or not harmonics:
        raise ValueError("harmonics must be a non-empty dict {m: complex coefficient}")
    return {int(k): complex(v) for k, v in harmonics.items()}


def kink_mode(equilibrium=None, m: int = 1, n: int = 1, *, amplitude: float = 0.06, radial_profile: str = "internal",
              phase: float = 0.0, harmonics: Optional[Dict[int, complex]] = None, labels: bool = True) -> Diagram:
    r"""A prescribed $m/n$ ideal displacement drawn on an equilibrium's own flux surfaces.

    Each surface $\rho = \sqrt{\psi_N}$ of ``equilibrium`` (an
    ``EquilibriumData``; default the Solov'ev equilibrium of
    ``_equilibrium_geometry``) moves along its unit normal by
    $\xi_n = A\,a\,F(\rho)\,\mathrm{Re}\sum_m c_m e^{i(m\theta^* - n\phi +
    \varphi_0)}$ at $\phi = 0$, with $a$ the minor radius, $\theta^*$ the
    equilibrium's PEST angle and $F$ the named envelope (``internal`` needs a
    $q = m/n$ surface, which is drawn dashed). ``harmonics`` adds sidebands,
    e.g. ``{1: 1, 2: 0.3}``. Under flux freezing this is the perturbed flux
    of ``flux_perturbation_from_normal_displacement``. Dashed grey: the
    unperturbed surfaces. A synthetic parameterization, not an eigenfunction.
    """
    harmonics = _validate(m, n, amplitude, radial_profile, harmonics)
    labels = _check_labels(labels)
    geom = equilibrium_geometry(equilibrium)
    rho_s = geom.rho_at_q(m / n)
    if radial_profile == "internal" and rho_s is None:
        raise ValueError(f"the internal envelope needs a q = {m}/{n} surface inside the plasma; this equilibrium has "
                         f"q from {geom.q0:.2f} to {geom.q_profile[-1]:.2f}")
    items: List = []
    surfaces = []
    for rho in SURFACES:
        d = displacement(geom, rho, n=n, amplitude=float(amplitude), harmonics=harmonics, kind=radial_profile,
                         rho_s=rho_s, phase=float(phase))
        s = d["surface"]
        edge = rho == SURFACES[-1]
        items.append(Polyline.of(_S * np.stack([s.R, s.Z], -1), "approx", role="unperturbed", closed=True))
        items.append(Polyline.of(_S * np.stack([d["R"], d["Z"]], -1), "boundary" if edge else "orbit electron",
                                 role="displaced", closed=True))
        surfaces.append({"rho": float(rho), **{k: d[k] for k in ("xi", "R", "Z")}, "R0": s.R, "Z0": s.Z,
                         "theta_star": s.theta_star, "normal": (s.normal_R, s.normal_Z)})
    if rho_s is not None:
        s = geom.surface(rho_s)
        items.append(Polyline.of(_S * np.stack([s.R, s.Z], -1), "rational", role="resonant_surface", closed=True))
    # displacement arrows on the surface where the envelope is largest, x3 for visibility
    # on the outermost drawn surface whose displacement is (nearly) the largest
    peak = [np.abs(sfc["R"] - sfc["R0"]).max() + np.abs(sfc["Z"] - sfc["Z0"]).max() for sfc in surfaces]
    show = surfaces[max(i for i, v in enumerate(peak) if v >= 0.9 * max(peak))] if max(peak) > 0 else surfaces[-1]
    arrows = []
    for i in range(0, len(show["xi"]), len(show["xi"]) // 10)[:10]:
        moved = np.array([show["R"][i] - show["R0"][i], show["Z"][i] - show["Z0"][i]])
        if np.hypot(*moved) < 0.1 * geom.minor_radius * max(amplitude, 1e-9):
            continue
        p0 = np.array([show["R0"][i], show["Z0"][i]])
        p1 = p0 + 3.0 * moved
        items.append(Arrow(tuple(_S * p0), tuple(_S * p1), "drift", role="xi"))
        arrows.append((p0, p1))
    # the axis moves with the innermost surface (a rigid shift for m = 1, none for m >= 2)
    inner = surfaces[0]
    axis_shift = (float(inner["R"].mean() - inner["R0"].mean()), float(inner["Z"].mean() - inner["Z0"].mean()))
    items.append(Marker(tuple(_S * (np.array(geom.axis) + axis_shift)), "o", "opoint", role="axis"))
    if labels:
        top = _S * float(np.max(geom.equilibrium.lcfs.z)) + 0.5
        x_mid = _S * geom.axis[0]
        right = _S * float(np.max(geom.equilibrium.lcfs.r)) + 0.6
        def term(k, c):
            # the coefficient the drawing uses: a real one with its sign, a complex one as |c| and its phase
            basis = "e^{i\\theta^*}" if k == 1 else f"e^{{i{k}\\theta^*}}"
            if c.imag == 0.0:
                magnitude = abs(c.real)
                return ("-" if c.real < 0.0 else "+"), ("" if magnitude == 1.0 else f"{magnitude:g}\\,") + basis
            angle = ("" if k == 1 else f"{k}") + f"\\theta^* {np.angle(c):+.3g}"
            return "+", f"{abs(c):g}\\,e^{{i({angle})}}"

        signed = [term(k, c) for k, c in sorted(harmonics.items())]
        terms = ("-" if signed[0][0] == "-" else "") + signed[0][1] + "".join(f" {s} {t}" for s, t in signed[1:])
        items += [
            Label((x_mid, top), f"prescribed $m/n = {m}/{n}$ displacement, {radial_profile} envelope", "label",
                  anchor="south", role="title"),
            Label((right + 0.5, _S * 0.25), "solid: displaced\\\\ dashed grey: unperturbed\\\\ red arrows: "
                  "displacement, $\\times 3$" + (f"\\\\ blue dashed: $q = {m if n == 1 else f'{m}/{n}'}$ (unperturbed)"
                                                if rho_s else ""),
                  "small label,align=left", anchor="north west", role="legend"),
            Label((right + 0.5, -_S * 0.05), f"$\\xi_n = A\\,a\\,F(\\rho)\\,\\mathrm{{Re}}[({terms})\\,e^{{-in\\phi + i\\varphi_0}}]$\\\\ "
                  f"$A = {amplitude:g}$, $\\phi = 0$", "small label,align=left", anchor="north west", role="model"),
            Label((x_mid, -top - 0.2),
                  f"$\\displaystyle {formula_equation(flux_perturbation_from_normal_displacement)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Synthetic parameterization, not an eigenfunction: the phase uses the equilibrium's $\\theta^*$ "
                  "(PEST), the drawing its real $(R, Z)$ surfaces; $m = 1$ drawn as the rigid shift; the helicity "
                  "sign is not drawn ($\\phi = 0$ only)", x_mid + 2.0, -top - 1.5),
        ]
    model = {"classification": "synthetic_parameterization", "m": m, "n": n, "amplitude": float(amplitude),
             "radial_profile": radial_profile, "harmonics": harmonics, "phase": float(phase), "rho_s": rho_s,
             "minor_radius": geom.minor_radius, "surfaces": surfaces, "arrows": arrows, "axis_shift": axis_shift,
             "geometry": geom}
    return Diagram(f"kink_mode_{m}_{n}_{radial_profile}", Scene(tuple(items)), model=model)
