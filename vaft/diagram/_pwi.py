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
against :mod:`vaft.data.atomic`'s element vocabulary and kept visually apart
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
from vaft.data.atomic import ATOMIC_NUMBERS, parse_species

from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: standard atomic weights [u] (IUPAC 2021, abridged; D and T their nuclide masses) of the species the
#: diagrams know by name -- constants for the kinematic factor, not surface-response data
MASS_U = {"H": 1.008, "D": 2.014, "T": 3.016, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
          "N": 14.007, "O": 15.999, "Ne": 20.18, "Ar": 39.95, "Fe": 55.85, "Mo": 95.95, "W": 183.84}


#: where each drawn quantity lives in IMAS (``wall.global_quantities.neutral[:]``). The IMAS "recycling energy
#: coefficient" is energy returned per incident energy by all recycling channels -- not the prompt-reflection
#: R_E of these diagrams, which is one part of it.
IMAS_WALL_PATHS = {
    "recycling": "wall.global_quantities.neutral[:].recycling_particles_coefficient",
    "recycling_energy": "wall.global_quantities.neutral[:].recycling_energy_coefficient",
    "incident_flux": "wall.global_quantities.neutral[:].particle_flux_from_plasma",
    "returned_flux": "wall.global_quantities.neutral[:].particle_flux_from_wall",
    "retention": "wall.global_quantities.neutral[:].wall_inventory",
    "sputtering_physical": "wall.global_quantities.neutral[:].incident_species[:].sputtering_physical_coefficient",
    "sputtering_chemical": "wall.global_quantities.neutral[:].incident_species[:].sputtering_chemical_coefficient",
}


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
        raise ValueError(f"{name!r} is not a species vaft.data.atomic recognises")
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
        "re-emission": ((0.9, 1.3), True),
    }
    items.append(Arrow((hit[0] + 0.1, 0.1), outcomes["reflection"][0], "drift ion", role="outcome:reflection"))
    items.append(Arrow((hit[0] + 0.6, 0.05), outcomes["sputtering"][0], "vector", role="outcome:sputtering"))
    items.append(_atom(outcomes["sputtering"][0][0] + 0.15, outcomes["sputtering"][0][1] + 0.15, False,
                       "sputtered_atom"))
    # implantation: the ion comes to rest inside; some diffuses back and leaves as a molecule
    items.append(Polyline.of([hit, (5.4, -0.6), (5.1, -1.1)], "orbit ion", role="implantation"))
    items.append(_atom(5.1, -1.1, True, "implanted"))
    items.append(Polyline.of([(5.1, -1.1), (4.2, -0.9), (3.6, -0.4), (3.3, -0.05)], "orbit ion",
                             role="diffusion_back"))
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
            Label((0.6, 1.5), f"re-emission:\\\\ thermal {p}$_2$", "small label,align=right", anchor="south east",
                  role="re-emission"),
            Label((4.95, -1.2), "implantation /\\\\ retention", "small label,align=right", anchor="north east",
                  role="retention"),
            Label((6.55, -1.7), "heat", "small label", anchor="west", role="heat"),
            Label((0.2, -0.3), f"wall: {t}", "small label", anchor="north west", role="wall"),
            _note(f"Blue: projectile {p}, dark: target {t}. Recycling = reflection + re-emission; retention is "
                  "the rest", 6.0, -2.5),
        ]
    return Diagram("plasma_wall_interaction_processes", Scene(tuple(items)),
                   model={"projectile": p, "target": t, "outcomes": ("reflection", "implantation", "re-emission",
                                                                     "sputtering", "heat"), "imas": IMAS_WALL_PATHS})


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
            Label((hit[0] + 0.75, 1.0), "$\\theta_\\mathrm{refl}$", "small label", anchor="south west", role="angle:out"),
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
    r"""Physical sputtering by a light projectile: backscatter from below ejects a surface atom.

    For a light projectile (D on W) the route near threshold is: the
    projectile penetrates, is backscattered by a deeper target atom keeping
    $(1 - \gamma)E$, and on its way out strikes a surface atom from below,
    giving it up to $\gamma(1 - \gamma)E$; if that exceeds the surface
    binding energy the atom leaves. Hence Bohdansky's threshold
    $E_s/\gamma(1 - \gamma)$, with $\gamma$ from
    ``binary_collision_energy_transfer_factor`` for the pair; the threshold is
    drawn only if ``surface_binding_energy`` is given
    (``sputtering_threshold_bohdansky``). No yield is drawn.
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
    deep = (5.1, -1.55)        # the target atom that backscatters the projectile
    surface_atom = (5.9, -0.35)  # the surface atom it strikes from below on the way out
    for k, (x, y) in enumerate(lattice):
        items.append(_atom(x, y, False, f"lattice:{k}"))
    path_in = [(3.6, 2.4), (4.3, 0.0), (5.0, -1.4)]
    path_out = [(5.0, -1.4), (5.55, -0.75), (5.85, -0.45)]
    items += [Arrow(path_in[0], (3.95, 1.2), "drift ion", role="incident"),
              Polyline.of(path_in, "orbit ion", role="projectile_in"),
              Polyline.of(path_out, "orbit ion", role="projectile_backscattered"),
              _atom(*deep, False, "backscatterer"),
              Arrow((6.0, -0.25), (7.4, 1.9), "vector", role="sputtered"),
              _atom(7.5, 2.05, False, "sputtered_atom")]
    if labels:
        items += [
            Label((3.5, 2.5), f"{p}$^+$, $E$", "small label", anchor="south", role="incident"),
            Label((7.6, 2.1), f"sputtered {t}", "small label", anchor="west", role="sputtered"),
            Polyline.of([deep, (2.0, -2.9)], "leader line", role="backscatter"),
            Label((2.0, -2.95), f"backscattered by a deep {t}: keeps $(1-\\gamma)E$", "small label", anchor="north",
                  role="backscatter"),
            Polyline.of([surface_atom, (10.3, -0.7)], "leader line", role="transfer"),
            Label((10.35, -0.7), "hits a surface atom from below:\\\\ gives up to $\\gamma(1-\\gamma)E$",
                  "small label,align=left", anchor="west", role="transfer"),
            Label((10.3, 1.4), f"{p} on {t}: $\\gamma = {gamma:.3f}$"
                  + (f"\\\\ $E_\\mathrm{{th}} = {E_th:.0f}$ eV for $E_s = {surface_binding_energy:g}$ eV" if E_th else
                     "\\\\ $E_\\mathrm{th}$: needs $E_s$ (not assumed)"), "small label,align=left",
                  anchor="north west", role="kinematics"),
            Label((6.5, -3.6), f"$\\displaystyle {formula_equation(sputtering_threshold_bohdansky)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Physical sputtering by a light projectile near threshold (no chemical erosion); yields and "
                  "thresholds are material data -- a threshold only for a supplied $E_s$", 6.5, -5.8),
        ]
    return Diagram("plasma_wall_interaction_sputtering", Scene(tuple(items)),
                   model={"projectile": p, "target": t, "gamma": gamma, "threshold_eV": E_th, "imas": IMAS_WALL_PATHS})


def plasma_wall_interaction_recycling(*, labels: bool = True) -> Diagram:
    r"""Reflection, re-emission and retention as flux bookkeeping.

    The incident flux splits into a promptly reflected part and an implanted
    part; the implanted part is either re-emitted later (diffusion,
    recombination, desorption) or retained as wall inventory. Recycling is
    reflection plus re-emission (``recycling_coefficient``); retention is the
    rest; a wall releasing an earlier inventory returns more than it
    receives ($R > 1$), a saturated one about as much ($R \\to 1$).
    """
    labels = _check_labels(labels)
    items: List = []
    incident = box(1.8, 0.0, 3.0, 1.1, "incident flux $\\Gamma_\\mathrm{in}$", role="flux:incident", latex=True)
    reflected = box(7.0, 1.6, 4.2, 1.3, "reflected (prompt, fast)\\\\ $\\Gamma_\\mathrm{refl}$", role="flux:reflected",
                    latex=True, style="concept leaf")
    implanted = box(7.0, -1.2, 4.2, 1.1, "implanted", role="flux:implanted", latex=True)
    reemitted = box(12.6, -0.2, 4.6, 1.3, "re-emitted: later, thermal\\\\ $\\Gamma_\\mathrm{re\\text{-}em}$",
                    role="flux:reemitted", latex=True, style="concept leaf")
    retained = box(12.6, -2.4, 4.6, 1.1, "retained: wall inventory", role="flux:retained", latex=True)
    for b in (incident, reflected, implanted, reemitted, retained):
        items += list(b.items)
    for a, b in ((incident, reflected), (incident, implanted), (implanted, reemitted), (implanted, retained)):
        items.append(connector(a, b, role="edge"))
    if labels:
        items += [
            Label((12.6, 1.6), "white boxes: back to the plasma\\\\ = recycling", "small label,align=center",
                  anchor="center", role="recycling"),
            Label((7.0, -3.2), f"$\\displaystyle {formula_equation(recycling_coefficient)}$", "formula box",
                  anchor="north", role="equations"),
            _note("Reflection is prompt, re-emission delayed; recycling counts both, retention neither. Atoms "
                  "counted: a D$_2$ molecule is two", 7.0, -4.6),
        ]
    return Diagram("plasma_wall_interaction_recycling", Scene(tuple(items)),
                   model={"returned": ("reflected", "re-emitted"), "kept": ("retained",), "imas": IMAS_WALL_PATHS})


def plasma_wall_interaction_energy_partition(*, labels: bool = True) -> Diagram:
    r"""Particle balance and energy balance, side by side: not the same bookkeeping.

    Left, where the projectile particles go: reflected ($R_N$), re-emitted,
    retained. Right, the power to the surface: its inputs -- ion kinetic
    energy including what the sheath adds, the potential energy released when
    the ion recombines (and atoms form molecules), and the electrons' heat
    across the sheath -- and its outputs: carried off by reflected particles
    ($R_E$ of the ion kinetic energy), by sputtered atoms and by thermal
    re-emission, with the remainder deposited as heat. A surface can return
    most particles while keeping most of their energy, $R_E/R_N < 1$
    (``mean_reflected_energy_fraction``).
    """
    labels = _check_labels(labels)
    items: List = []
    pb = box(2.5, 0.0, 4.2, 1.1, "projectile particles\\\\ per incident ion", role="balance:particles", latex=True)
    eb = box(11.5, 0.0, 5.0, 1.1, "power to the surface\\\\ per incident ion", role="balance:energy", latex=True)
    items += list(pb.items) + list(eb.items)
    parts_p = ["reflected, $R_N$", "re-emitted (later)", "retained"]
    inputs = ["ion kinetic energy, incl. sheath gain", "potential energy: recombination", "electron heat across the sheath"]
    outputs = ["carried off by reflected particles, $R_E$", "carried off by sputtered atoms",
               "thermal re-emission (small)", "the remainder: heat in the wall"]

    def column(x, y0, texts, role, width):
        boxes = []
        for k, text in enumerate(texts):
            b = box(x, y0 - 1.1 * k, width, 0.85, text, role=f"{role}:{k}", latex=True, style="concept leaf")
            boxes.append(b)
        return boxes

    left = column(2.5, -1.6, parts_p, "particles", 4.2)
    ins = column(9.0, -1.9, inputs, "energy_in", 5.2)
    outs = column(14.9, -1.9, outputs, "energy_out", 6.2)
    for b in left + ins + outs:
        items += list(b.items)
    # a spine from each header to its leaves, so every leaf is connected
    for head, leaves, x_spine in ((pb, left, 0.2), (eb, ins, 6.2), (eb, outs, 18.2)):
        y_top = head.y - 0.55 if head is pb else -0.9
        items.append(Polyline.of([(x_spine, y_top), (x_spine, leaves[-1].y)], "connector line", role="spine"))
        for leaf in leaves:
            edge = leaf.x - leaf.width / 2 if x_spine < leaf.x else leaf.x + leaf.width / 2
            items.append(Arrow((x_spine, leaf.y), (edge, leaf.y), "connector", role="edge"))
    items.append(Polyline.of([(6.2, -0.9), (18.2, -0.9)], "connector line", role="spine"))
    items.append(Polyline.of([(eb.x, -0.55), (eb.x, -0.9)], "connector line", role="spine"))
    items.append(Polyline.of([(pb.x - 2.1, -0.55), (0.2, -0.55)], "connector line", role="spine"))
    if labels:
        items += [
            Label((9.0, -1.35), "inputs", "small label", anchor="south", role="group"),
            Label((14.9, -1.35), "outputs", "small label", anchor="south", role="group"),
            Label((6.7, 0.0), "$\\neq$", "legend symbol", role="not_equal"),
            Label((9.0, -6.3), f"$\\displaystyle {formula_equation(mean_reflected_energy_fraction)}$",
                  "formula box", anchor="north", role="equations"),
            _note("No fractions drawn: they are data per projectile, target, energy and angle. The potential energy "
                  "is an input on top of the kinetic energy; most of it ends as heat", 9.0, -7.6),
        ]
    return Diagram("plasma_wall_interaction_energy_partition", Scene(tuple(items)),
                   model={"particles": tuple(parts_p), "energy_in": tuple(inputs), "energy_out": tuple(outputs),
                          "imas": IMAS_WALL_PATHS})
