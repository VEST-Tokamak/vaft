"""Plasma-wall interaction concepts: processes, reflection, sputtering, recycling, energy partition (#1047).

``plasma_wall_interaction_processes``
    one impact and its outcomes: reflection, implantation/retention,
    re-emission, sputtering, heat;
``plasma_wall_interaction_reflection``
    incident and reflected particle, with their energies and angles kept as
    separate quantities, and particle versus energy reflection;
``plasma_wall_interaction_sputtering``
    a projectile's collision cascade ejecting a target atom, and why a light
    projectile has a high threshold;
``plasma_wall_interaction_recycling``
    reflection, re-emission and retention as flux bookkeeping;
``plasma_wall_interaction_energy_partition``
    particle balance and energy balance side by side: not the same thing.

Level 0 of the issue's enrichment: semantics only. No reflection or
sputtering coefficient, threshold or yield is drawn unless computed from a
``vaft.formula.pwi`` relation with inputs the caller supplies (a threshold
needs the surface binding energy). Projectile and target species are checked
against :mod:`vaft.spectroscopy`'s element vocabulary and kept visually apart
(projectile blue, target dark).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from vaft.formula.pwi import (
    binary_collision_energy_transfer_factor,
    mean_reflected_energy_fraction,
    recycling_coefficient,
    sputtering_threshold_bohdansky,
)
from vaft.spectroscopy import ATOMIC_NUMBERS, parse_species

from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: atomic masses [u] of the species the diagrams know by name (for the kinematic factor only)
MASS_U = {"H": 1.008, "D": 2.014, "T": 3.016, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
          "N": 14.007, "O": 15.999, "Ne": 20.18, "Ar": 39.95, "Fe": 55.85, "Mo": 95.95, "W": 183.84}


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _species(name) -> str:
    """The element symbol (D and T kept as isotopes of hydrogen, named as such) of a species name."""
    sp = parse_species(name)
    if sp is None or sp.element not in ATOMIC_NUMBERS:
        raise ValueError(f"{name!r} is not a species vaft.spectroscopy recognises")
    if sp.element == "H" and sp.mass_number in (2, 3):
        return {2: "D", 3: "T"}[sp.mass_number]
    return sp.element


def _wall(x0: float, x1: float, depth: float = 2.2) -> List:
    return [Polyline.of([(x0, 0.0), (x1, 0.0), (x1, -depth), (x0, -depth)], "section fill", role="wall", closed=True),
            Polyline.of([(x0, 0.0), (x1, 0.0)], "machine", role="surface")]


def _atom(x: float, y: float, projectile: bool, role: str) -> Marker:
    return Marker((x, y), "o", "opoint" if projectile else "xpoint", role=role)


def plasma_wall_interaction_processes(projectile="D", target="W", *, labels: bool = True) -> Diagram:
    r"""One ion impact and its outcomes, the projectile and target species kept apart.

    An incident ion (projectile, blue) strikes the wall (target material,
    grey): it may be reflected at once as a fast atom; or implanted, where
    it stays (retention) or later diffuses back out, recombines and leaves as
    a thermal molecule (re-emission); it may knock a target atom out
    (sputtering, dark); and the energy it does not carry away is deposited as
    heat. Reflection plus re-emission is recycling. Semantic: no
    coefficients.
    """
    labels = _check_labels(labels)
    p, t = _species(projectile), _species(target)
    items: List = _wall(0.0, 12.0)
    hit = (5.0, 0.0)
    items.append(Arrow((1.5, 3.2), hit, "drift ion", role="incident"))
    outcomes = {
        "reflection": ((8.8, 3.6), True),
        "sputtering": ((11.0, 2.2), False),
        "re-emission": ((1.2, 1.9), True),
    }
    items.append(Arrow((hit[0] + 0.1, 0.1), outcomes["reflection"][0], "drift ion", role="outcome:reflection"))
    items.append(Arrow((hit[0] + 0.6, 0.05), outcomes["sputtering"][0], "vector", role="outcome:sputtering"))
    items.append(_atom(outcomes["sputtering"][0][0] + 0.15, outcomes["sputtering"][0][1] + 0.15, False,
                       "sputtered_atom"))
    # implantation: the ion comes to rest inside; some diffuses back and leaves as a molecule
    items.append(Polyline.of([hit, (5.4, -0.6), (5.1, -1.1)], "orbit ion", role="implantation"))
    items.append(_atom(5.1, -1.1, True, "implanted"))
    items.append(Polyline.of([(4.2, -0.9), (3.6, -0.4), (3.3, -0.05)], "orbit ion", role="diffusion_back"))
    items.append(Arrow((3.3, 0.05), outcomes["re-emission"][0], "connector", role="outcome:re-emission"))
    for dx in (0.0, 0.22):
        items.append(_atom(outcomes["re-emission"][0][0] - 0.1 + dx, outcomes["re-emission"][0][1] + 0.15, True,
                           "molecule"))
    items.append(Arrow((6.4, -0.3), (6.4, -1.8), "drift", role="heat"))
    if labels:
        items += [
            Label((1.5, 3.3), f"incident {p}$^+$ (projectile)", "small label", anchor="south", role="incident"),
            Label((8.9, 3.7), f"reflection: fast {p}$^0$", "small label", anchor="south", role="reflection"),
            Label((11.1, 2.4), f"sputtering:\\\\ {t} atom (target)", "small label,align=center", anchor="south",
                  role="sputtering"),
            Label((1.1, 2.15), f"re-emission:\\\\ thermal {p}$_2$", "small label,align=center", anchor="south",
                  role="re-emission"),
            Label((4.95, -1.2), "implantation /\\\\ retention", "small label,align=right", anchor="north east",
                  role="retention"),
            Label((6.55, -1.7), "heat", "small label", anchor="west", role="heat"),
            Label((0.2, -0.3), f"wall: {t}", "small label", anchor="north west", role="wall"),
            _note(f"Blue: projectile {p}, dark: target {t}. Recycling = reflection + re-emission; retention is what "
                  "neither returns. No coefficients drawn", 6.0, -2.5),
        ]
    return Diagram("plasma_wall_interaction_processes", Scene(tuple(items)),
                   model={"projectile": p, "target": t, "outcomes": ("reflection", "implantation", "re-emission",
                                                                     "sputtering", "heat")})


def plasma_wall_interaction_reflection(projectile="D", target="W", *, labels: bool = True) -> Diagram:
    r"""Particle reflection: incident and reflected energy and angle are separate quantities.

    An incident particle $(E_\mathrm{in}, \theta_\mathrm{in})$ and a
    reflected one $(E_\mathrm{refl}, \theta_\mathrm{refl})$, angles from the
    surface normal; the reflected particle leaves with less energy and at a
    different angle, both distributed. $R_N$ counts reflected particles, $R_E$
    reflected energy; their ratio is the mean reflected-energy fraction
    (``mean_reflected_energy_fraction``). Both coefficients are data, not
    drawn.
    """
    labels = _check_labels(labels)
    p, t = _species(projectile), _species(target)
    items: List = _wall(0.0, 10.0, 1.2)
    hit = (5.0, 0.0)
    th_in, th_out = math.radians(50.0), math.radians(30.0)
    L_in, L_out = 3.6, 2.4
    start = (hit[0] - L_in * math.sin(th_in), L_in * math.cos(th_in))
    end = (hit[0] + L_out * math.sin(th_out), L_out * math.cos(th_out))
    items += [Polyline.of([hit, (hit[0], 3.2)], "approx", role="normal"),
              Arrow(start, hit, "drift ion", role="incident"),
              Arrow(hit, end, "drift ion", role="reflected")]
    for th, sign, role in ((th_in, -1, "angle:in"), (th_out, 1, "angle:out")):
        a = np.linspace(0.0, th, 30)
        items.append(Polyline.of(np.stack([hit[0] + sign * 1.0 * np.sin(a), 1.0 * np.cos(a)], -1), "angle arc",
                                 role=role))
    if labels:
        items += [
            Label((start[0] - 0.1, start[1]), f"incident {p}: $E_\\mathrm{{in}}$", "small label", anchor="east",
                  role="incident"),
            Label((end[0] + 0.1, end[1]), "reflected: $E_\\mathrm{refl} < E_\\mathrm{in}$", "small label",
                  anchor="west", role="reflected"),
            Label((hit[0] - 0.55, 1.2), "$\\theta_\\mathrm{in}$", "small label", anchor="south", role="angle:in"),
            Label((hit[0] + 0.45, 1.15), "$\\theta_\\mathrm{refl}$", "small label", anchor="south", role="angle:out"),
            Label((hit[0], 3.25), "normal", "small label", anchor="south", role="normal"),
            Label((0.2, -0.3), f"target {t}", "small label", anchor="north west", role="wall"),
            Label((5.0, -1.6), f"$\\displaystyle {formula_equation(mean_reflected_energy_fraction)}$", "formula box",
                  anchor="north", role="equations"),
            _note("$R_N$ (particles) and $R_E$ (energy) are different data for each projectile, target, "
                  "energy and angle; none is drawn here", 5.0, -2.9),
        ]
    return Diagram("plasma_wall_interaction_reflection", Scene(tuple(items)),
                   model={"projectile": p, "target": t, "theta_in": th_in, "theta_refl": th_out})


def plasma_wall_interaction_sputtering(projectile="D", target="W", surface_binding_energy: Optional[float] = None, *,
                                       labels: bool = True) -> Diagram:
    r"""Physical sputtering: a collision cascade ejects a target atom; a light projectile has a high threshold.

    The projectile enters a lattice of target atoms, recoils cascade, and a
    surface atom receives outward momentum above its binding energy and
    leaves. One elastic collision gives at most $\gamma E$
    (``binary_collision_energy_transfer_factor``, computed for the pair); the
    threshold is drawn only if ``surface_binding_energy`` is given, from
    Bohdansky's fit (``sputtering_threshold_bohdansky``). No yield is drawn.
    """
    labels = _check_labels(labels)
    p, t = _species(projectile), _species(target)
    if p not in MASS_U or t not in MASS_U:
        raise ValueError(f"no mass tabulated here for {p if p not in MASS_U else t}")
    gamma = float(binary_collision_energy_transfer_factor(MASS_U[p], MASS_U[t]))
    E_th = None
    if surface_binding_energy is not None:
        E_th = float(sputtering_threshold_bohdansky(surface_binding_energy, MASS_U[p], MASS_U[t]))
    items: List = _wall(0.0, 10.0, 2.4)
    lattice = [(0.7 + 0.8 * i + (0.4 if j % 2 else 0.0), -0.35 - 0.6 * j) for j in range(4) for i in range(11)]
    for k, (x, y) in enumerate(lattice):
        items.append(_atom(x, y, False, f"lattice:{k}"))
    path = [(3.2, 2.6), (4.4, -0.35), (4.9, -1.1), (5.6, -1.6)]
    items.append(Polyline.of(path, "orbit ion", role="projectile_path"))
    items.append(Arrow(path[0], (3.7, 1.35), "drift ion", role="incident"))
    cascade = [((4.4, -0.35), (5.2, -0.95)), ((5.2, -0.95), (6.0, -0.35)), ((6.0, -0.35), (6.3, 0.0))]
    for a, b in cascade:
        items.append(Arrow(a, b, "vector", role="recoil"))
    items.append(Arrow((6.3, 0.05), (7.6, 2.0), "vector", role="sputtered"))
    items.append(_atom(7.7, 2.15, False, "sputtered_atom"))
    if labels:
        items += [
            Label((3.1, 2.7), f"{p}$^+$", "small label", anchor="south", role="incident"),
            Label((7.8, 2.2), f"sputtered {t}", "small label", anchor="west", role="sputtered"),
            Label((6.4, 0.15), "recoil cascade", "small label", anchor="south west", role="recoil"),
            Label((10.3, 1.4), f"{p} on {t}: $\\gamma = {gamma:.3f}$\\\\ at most $\\gamma E$ per collision"
                  + (f"\\\\ $E_\\mathrm{{th}} = {E_th:.0f}$ eV for $E_s = {surface_binding_energy:g}$ eV" if E_th else
                     "\\\\ $E_\\mathrm{th}$: needs $E_s$ (not assumed)"), "small label,align=left",
                  anchor="north west", role="kinematics"),
            Label((5.0, -2.8), f"$\\displaystyle {formula_equation(binary_collision_energy_transfer_factor)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Physical sputtering only (no chemical erosion); yields and thresholds are material data -- "
                  "a threshold is shown only for a supplied $E_s$ (Bohdansky fit)", 5.0, -4.0),
        ]
    return Diagram("plasma_wall_interaction_sputtering", Scene(tuple(items)),
                   model={"projectile": p, "target": t, "gamma": gamma, "threshold_eV": E_th})


def plasma_wall_interaction_recycling(*, labels: bool = True) -> Diagram:
    r"""Reflection, re-emission and retention as flux bookkeeping.

    The incident flux splits into a promptly reflected part and an implanted
    part; the implanted part is either re-emitted later (diffusion,
    recombination, desorption) or retained as wall inventory. Recycling is
    reflection plus re-emission (``recycling_coefficient``); retention is the
    rest; a saturated or outgassing wall can return more than it receives.
    """
    labels = _check_labels(labels)
    items: List = []
    incident = box(1.8, 0.0, 3.0, 1.1, "incident flux $\\Gamma_\\mathrm{in}$", role="flux:incident", latex=True)
    reflected = box(7.0, 1.6, 4.2, 1.3, "reflected (prompt, fast)\\\\ $\\Gamma_\\mathrm{refl}$", role="flux:reflected",
                    latex=True)
    implanted = box(7.0, -1.2, 4.2, 1.1, "implanted", role="flux:implanted", latex=True)
    reemitted = box(12.6, -0.2, 4.6, 1.3, "re-emitted: later, thermal\\\\ $\\Gamma_\\mathrm{re\\text{-}em}$",
                    role="flux:reemitted", latex=True)
    retained = box(12.6, -2.4, 4.6, 1.1, "retained: wall inventory", role="flux:retained", latex=True)
    for b in (incident, reflected, implanted, reemitted, retained):
        items += list(b.items)
    for a, b in ((incident, reflected), (incident, implanted), (implanted, reemitted), (implanted, retained)):
        items.append(connector(a, b, role="edge"))
    if labels:
        items += [
            Label((12.6, 1.6), "back to the plasma: recycling", "small label", anchor="south", role="recycling"),
            Polyline.of([(9.2, 1.6), (10.0, 1.6), (10.0, 0.45)], "leader line", role="recycling"),
            Label((7.0, -3.2), f"$\\displaystyle {formula_equation(recycling_coefficient)}$", "formula box",
                  anchor="north", role="equations"),
            _note("Reflection is prompt, re-emission delayed; recycling counts both, retention neither. Atoms "
                  "counted: a D$_2$ molecule is two", 7.0, -4.6),
        ]
    return Diagram("plasma_wall_interaction_recycling", Scene(tuple(items)),
                   model={"returned": ("reflected", "re-emitted"), "kept": ("retained",)})


def plasma_wall_interaction_energy_partition(*, labels: bool = True) -> Diagram:
    r"""Particle balance and energy balance, side by side: not the same bookkeeping.

    Left, where the particles go: reflected ($R_N$), re-emitted, retained.
    Right, where the energy goes: carried off by reflected particles ($R_E$),
    deposited as heat in the wall, the potential (ionization and molecular
    binding) energy released at the surface on recombination, and a small
    part carried by sputtered atoms. A surface can return most particles while
    keeping most of their energy, $R_E/R_N < 1$
    (``mean_reflected_energy_fraction``).
    """
    labels = _check_labels(labels)
    items: List = []
    pb = box(2.5, 0.0, 4.2, 1.1, "particle balance\\\\ per incident particle", role="balance:particles", latex=True)
    eb = box(11.0, 0.0, 4.2, 1.1, "energy balance\\\\ per incident energy", role="balance:energy", latex=True)
    parts_p = ["reflected, $R_N$", "re-emitted (delayed)", "retained"]
    parts_e = ["carried off by reflected particles, $R_E$", "deposited as heat",
               "potential energy released (recombination)", "carried by sputtered atoms"]
    items += list(pb.items) + list(eb.items)
    for k, text in enumerate(parts_p):
        b = box(2.5, -1.6 - 1.25 * k, 4.2, 0.95, text, role=f"particles:{k}", latex=True, style="concept leaf")
        items += list(b.items)
        items.append(connector(pb, b, role="edge")) if k == 0 else None
    for k, text in enumerate(parts_e):
        b = box(11.0, -1.6 - 1.25 * k, 6.4, 0.95, text, role=f"energy:{k}", latex=True, style="concept leaf")
        items += list(b.items)
        items.append(connector(eb, b, role="edge")) if k == 0 else None
    if labels:
        items += [
            Label((6.5, 0.9), "$\\neq$", "legend symbol", role="not_equal"),
            Label((6.5, -6.7), f"$\\displaystyle {formula_equation(mean_reflected_energy_fraction)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Fractions are data for a projectile, target, energy and angle; none is drawn. Reflection "
                  "returns particles more readily than energy", 6.5, -8.0),
        ]
    return Diagram("plasma_wall_interaction_energy_partition", Scene(tuple(items)),
                   model={"particles": tuple(parts_p), "energy": tuple(parts_e)})
