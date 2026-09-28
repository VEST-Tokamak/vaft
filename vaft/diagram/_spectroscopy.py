"""Spectroscopy and ionization concept diagrams, with progressive metadata (#1046).

``spectroscopy_ionization_stages``
    every ionization stage of an element, the selected one highlighted, with
    ionization and recombination between neighbours;
``spectroscopy_transitions``
    the transition behind one emission term: exact levels and wavelength for
    a hydrogenic series member, and for any other line only what its label
    declares;
``spectroscopy_energy_levels``
    the hydrogenic level ladder with the Lyman, Balmer and Paschen series;
``spectroscopy_spectrum``
    a set of declared lines on a wavelength axis.

The vocabulary is :mod:`vaft.spectroscopy` (``parse_emission_term``,
``parse_line_label``, ``Species``, ``LineIdentity``) -- the same parser
``emission=`` uses in :mod:`vaft.plot`, so a term that selects a trace
selects the same diagram. Metadata enrichment is progressive: level 0 is the
semantic identity (stage, charge, element); level 1 adds what the data
declares (a wavelength in the IMAS label); hydrogenic lines add exact Bohr
levels (``hydrogenic_energy_level``, ``hydrogenic_transition_wavelength``).
Nothing is fabricated: a many-electron line without a declared wavelength is
drawn without one, and its levels wait for OPEN-ADAS ADF04 / ADF15, which
are extension points, not loaded here.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from vaft.formula.atomic import hydrogenic_energy_level, hydrogenic_transition_wavelength
from vaft.spectroscopy import (
    ATOMIC_NUMBERS,
    SERIES_NAMES,
    LineIdentity,
    charge_state_of,
    format_species,
    parse_emission_term,
    parse_line_label,
)

from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

#: the labels the VEST UV/visible spectrometer declares (``vaft.machine_mapping.spectrometer_uv``);
#: IMAS processed_line syntax, wavelength in Angstrom. A test keeps the two in step.
DECLARED_LABELS = ("H-alpha_6563", "OI_7770", "H-beta_4861", "H-gamma_4340", "CII_4267", "CIII_1909",
                   "OII_3726", "OV_629")
#: Balmer upper level of each series letter (the fusion-diagnostic convention: alpha is 3 -> 2)
BALMER_UPPER = {name: 3 + k for k, name in enumerate(SERIES_NAMES)}
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def identity(term) -> LineIdentity:
    """The line identity of a term or IMAS label, through the same parsers ``emission=`` uses."""
    if isinstance(term, LineIdentity):
        return term
    # the emission-term parser first (it keeps the isotope: "H-alpha" is protium), the IMAS label parser second
    found = parse_emission_term(term) or parse_line_label(term)
    if found is None:
        raise ValueError(f"{term!r} names no species or line vaft.spectroscopy recognises")
    return found


def _roman(stage: int) -> str:
    return _ROMAN[stage - 1] if stage <= len(_ROMAN) else str(stage)


def _stage_text(element: str, stage: int) -> str:
    q = charge_state_of(stage)
    return f"{element} {_roman(stage)}\\\\ $\\mathrm{{{element}}}^{{{q}+}}$" if q else f"{element} I\\\\ neutral"


def spectroscopy_ionization_stages(term="C III", *, labels: bool = True) -> Diagram:
    r"""All ionization stages of an element, the one a term names highlighted.

    ``term`` is anything :mod:`vaft.spectroscopy` parses -- ``"C III"``,
    ``"C2+"``, ``"carbon"``, ``"CIII_1909"``. Stage $s$ is charge $s - 1$
    (C III is C$^{2+}$); hydrogen isotopes are hydrogen with a mass number,
    not elements of their own. Arrows: ionization to the right, recombination
    to the left. Elements with more than twelve stages show the first ten and
    the fully stripped ion. Semantic only: no atomic data are needed.
    """
    labels = _check_labels(labels)
    ident = identity(term)
    element = "H" if ident.species.element in ("D", "T") else ident.species.element
    Z = ATOMIC_NUMBERS[element]
    stages = list(range(1, Z + 2))
    if len(stages) > 12:
        stages = stages[:10] + [None, Z + 1]
    selected = ident.species.ionization_stage
    items: List = []
    boxes = []
    for k, stage in enumerate(stages):
        x = 1.3 + 2.9 * k
        if stage is None:
            items.append(Label((x, 0.0), "$\\cdots$", "label", role="ellipsis"))
            boxes.append(None)
            continue
        b = box(x, 0.0, 2.1, 1.3, _stage_text(element, stage), role=f"stage:{stage}", latex=True)
        items += list(b.items)
        if stage == selected:  # the stage the term names: outlined in red
            items.append(Polyline.of([(x - 1.15, -0.75), (x + 1.15, -0.75), (x + 1.15, 0.75), (x - 1.15, 0.75)],
                                     "trough", role="selected", closed=True))
        boxes.append(b)
    for a, b in zip(boxes[:-1], boxes[1:]):
        if a is None or b is None:
            continue
        items.append(Arrow((a.x + 1.1, 0.3), (b.x - 1.1, 0.3), "connector", role="ionization"))
        items.append(Arrow((b.x - 1.1, -0.3), (a.x + 1.1, -0.3), "connector", role="recombination"))
    if labels:
        mass = ident.species.mass_number
        name = {1: "hydrogen", 2: "deuterium", 3: "tritium"}.get(mass, element) if element == "H" else element
        items += [
            Label((1.3 + 1.45 * (len(stages) - 1), 1.1), f"{name}: ionization stages (upper arrows: ionization, "
                  "lower: recombination)", "label", anchor="south", role="title"),
            _note(f"Stage $s$ = charge $s - 1$; highlighted: {term!s} as parsed by vaft.spectroscopy. Semantic "
                  "only: no atomic data used", 1.3 + 1.45 * (len(stages) - 1), -1.1),
        ]
    return Diagram("spectroscopy_ionization_stages", Scene(tuple(items)),
                   model={"identity": ident, "element": element, "stages": stages, "selected": selected})


def _balmer(ident: LineIdentity):
    """(n_upper, n_lower) of a hydrogen series term, or None."""
    if ident.species.element in ("H", "D", "T") and ident.series in BALMER_UPPER:
        return BALMER_UPPER[ident.series], 2
    return None


def spectroscopy_transitions(term="H-alpha", *, labels: bool = True) -> Diagram:
    r"""The transition behind one emission term, with no more metadata than exists.

    A hydrogen series member (``"H-alpha"``, ``"D-beta"``, ``"H-alpha_6563"``)
    is a Balmer line: its levels $n = 3, 4, \ldots \to 2$ and vacuum wavelength
    are exact (``hydrogenic_energy_level``,
    ``hydrogenic_transition_wavelength``, with the isotope's reduced mass;
    an unspecified isotope is taken as protium). Any
    other line (``"OI_7770"``, ``"C III"``) is drawn as an unnamed upper and
    lower level of its ionization stage, with the wavelength only if its label
    declares one; level identities would need ADF04, photon emissivities
    ADF15.
    """
    labels = _check_labels(labels)
    ident = identity(term)
    pair = _balmer(ident)
    items: List = []
    model = {"identity": ident, "hydrogenic": pair is not None}
    if pair is not None:
        n_u, n_l = pair
        mass = ident.species.mass_number or 1  # an unspecified hydrogen isotope is taken as protium
        E_u, E_l = hydrogenic_energy_level(n_u, 1, mass), hydrogenic_energy_level(n_l, 1, mass)
        lam = hydrogenic_transition_wavelength(n_u, n_l, 1, mass)
        scale = 6.0 / abs(E_l)
        y_u, y_l = 6.0 + scale * E_u + 2.0, 6.0 + scale * E_l + 2.0
        items += [Polyline.of([(0.0, y_u), (5.0, y_u)], "boundary", role="level:upper"),
                  Polyline.of([(0.0, y_l), (5.0, y_l)], "boundary", role="level:lower"),
                  Polyline.of([(0.0, 8.0), (5.0, 8.0)], "approx", role="ionization_limit"),
                  Arrow((2.5, y_u), (2.5, y_l), "drift", role="transition")]
        model.update({"n_upper": n_u, "n_lower": n_l, "E_upper_eV": E_u, "E_lower_eV": E_l, "wavelength_m": lam})
        if labels:
            items += [Label((5.2, y_u), f"$n = {n_u}$, ${E_u:.3f}$ eV", "small label", anchor="west", role="level"),
                      Label((5.2, y_l), f"$n = {n_l}$, ${E_l:.3f}$ eV", "small label", anchor="west", role="level"),
                      Label((5.2, 8.0), "ionization limit, 0 eV", "small label", anchor="west", role="level"),
                      Label((2.7, 0.5 * (y_u + y_l)), f"$\\lambda = {lam * 1e9:.2f}$ nm (vacuum)", "small label",
                            anchor="west", role="wavelength")]
            declared = ident.wavelength_angstrom
            if declared:
                items.append(Label((2.7, 0.5 * (y_u + y_l) - 0.5), f"label declares {declared / 10:g} nm (air)",
                                   "small label", anchor="west", role="wavelength_declared"))
            title = f"{format_species(ident.species)}-{ident.series}: Balmer $n = {n_u} \\to {n_l}$"
            items += [Label((2.5, 8.6), title, "label", anchor="south", role="title"),
                      Label((2.5, -0.3), f"$\\displaystyle {formula_equation(hydrogenic_transition_wavelength)}$",
                            "formula box", anchor="north", role="equations")]
    else:
        items += [Polyline.of([(0.0, 5.0), (5.0, 5.0)], "boundary", role="level:upper"),
                  Polyline.of([(0.0, 1.5), (5.0, 1.5)], "boundary", role="level:lower"),
                  Arrow((2.5, 5.0), (2.5, 1.5), "drift", role="transition")]
        lam = ident.wavelength_angstrom
        model["wavelength_m"] = lam * 1e-10 if lam else None
        if labels:
            stage = ident.species.ionization_stage
            who = format_species(ident.species)
            items += [Label((5.2, 5.0), "upper level (not identified)", "small label", anchor="west", role="level"),
                      Label((5.2, 1.5), "lower level (not identified)", "small label", anchor="west", role="level"),
                      Label((2.7, 3.25), f"$\\lambda = {lam / 10:g}$ nm, as declared" if lam else
                            "$\\lambda$ not declared", "small label", anchor="west", role="wavelength"),
                      Label((2.5, 5.6), f"{who}" + (f" (charge {charge_state_of(stage)}+)" if stage else "")
                            + ": a line of this stage", "label", anchor="south", role="title"),
                      _note("Levels need ADF04, emissivity ADF15 (extension points, not loaded); nothing here is "
                            "invented", 2.5, 0.8)]
    if labels and pair is not None:
        items.append(_note("Hydrogenic levels are exact in the Bohr model (no fine structure); vacuum wavelength -- "
                           "tabulated visible lines are in air", 2.5, -1.9))
    return Diagram("spectroscopy_transitions", Scene(tuple(items)), model=model)


def spectroscopy_energy_levels(term="H-alpha", *, n_max: int = 7, labels: bool = True) -> Diagram:
    r"""The hydrogenic level ladder, with the Lyman, Balmer and Paschen series.

    Levels $n = 1\ldots$ ``n_max`` of ``hydrogenic_energy_level`` for the
    term's isotope, drawn to scale in energy; downward arrows for the first
    members of the Lyman ($\to 1$, UV), Balmer ($\to 2$, visible) and Paschen
    ($\to 3$, IR) series, the term's own transition highlighted. Hydrogenic
    only: many-electron level structure needs ADF04 and raises here.
    """
    labels = _check_labels(labels)
    ident = identity(term)
    if ident.species.element not in ("H", "D", "T"):
        raise ValueError(f"{term!r} is not hydrogenic: its levels need OPEN-ADAS ADF04 data, not loaded here")
    if isinstance(n_max, bool) or not isinstance(n_max, int) or not 4 <= n_max <= 10:
        raise ValueError(f"n_max must be an integer from 4 to 10, not {n_max!r}")
    mass = ident.species.mass_number or 1  # an unspecified hydrogen isotope is taken as protium
    E = {n: hydrogenic_energy_level(n, 1, mass) for n in range(1, n_max + 1)}
    scale = 7.0 / abs(E[1])

    def y(e):
        return 7.5 + scale * e

    items: List = [Polyline.of([(0.0, y(0.0)), (11.0, y(0.0))], "approx", role="ionization_limit")]
    for n, e in E.items():
        items.append(Polyline.of([(0.0, y(e)), (11.0, y(e))], "surface" if n > 3 else "boundary", role=f"level:{n}"))
    selected = _balmer(ident)
    series = {"Lyman": (1, 0.8), "Balmer": (2, 4.3), "Paschen": (3, 8.0)}
    lines = []
    for name, (n_l, x0) in series.items():
        for k, n_u in enumerate(range(n_l + 1, min(n_l + 4, n_max + 1))):
            x = x0 + 0.6 * k
            style = "drift" if (n_u, n_l) == selected else "connector"
            items.append(Arrow((x, y(E[n_u])), (x, y(E[n_l])), style, role=f"line:{name}:{n_u}"))
            lines.append((name, n_u, n_l, hydrogenic_transition_wavelength(n_u, n_l, 1, mass)))
    if labels:
        for n, e in E.items():
            if n <= 4 or n == n_max:
                items.append(Label((11.2, y(e)), f"$n = {n}$: ${e:.2f}$ eV", "small label", anchor="west",
                                   role="level"))
        for name, (n_l, x0) in series.items():
            band = {"Lyman": "UV", "Balmer": "visible", "Paschen": "IR"}[name]
            items.append(Label((x0 + 0.6, y(E[n_l]) - 0.25), f"{name} ({band})", "small label", anchor="north",
                               role="series"))
        items += [Label((5.5, y(0.0) + 0.3), f"{format_species(ident.species)}: hydrogenic levels"
                        + (f", red: {ident.label}" if selected else ""), "label", anchor="south", role="title"),
                  _note("Bohr levels with the reduced mass (hydrogenic\\_energy\\_level); no fine structure", 5.5,
                        y(E[1]) - 0.9)]
    return Diagram("spectroscopy_energy_levels", Scene(tuple(items)),
                   model={"identity": ident, "levels": E, "lines": lines, "selected": selected})


def spectroscopy_spectrum(line_labels: Optional[Sequence[str]] = None, *, labels: bool = True) -> Diagram:
    r"""Declared spectral lines on a wavelength axis, each at the wavelength its label states.

    ``line_labels`` are IMAS ``processed_line.label`` strings (default: the
    lines the VEST spectrometer declares, ``DECLARED_LABELS``). A line is
    placed at the wavelength in its label; a hydrogen series member without
    one at its computed vacuum wavelength (marked); any other line without a
    wavelength is listed, not placed. Log wavelength axis, 50 to 1000 nm.
    """
    labels = _check_labels(labels)
    line_labels = tuple(DECLARED_LABELS if line_labels is None else line_labels)
    W = 16.0
    lo, hi = math.log10(50.0), math.log10(1000.0)

    def x(nm):
        return W * (math.log10(nm) - lo) / (hi - lo)

    items: List = [Arrow((0.0, 0.0), (W + 0.4, 0.0), "chart axis", role="axis")]
    for nm in (50, 100, 200, 500, 1000):
        items.append(Polyline.of([(x(nm), 0.0), (x(nm), -0.15)], "tick", role="tick"))
        if labels:
            items.append(Label((x(nm), -0.25), f"{nm}", "small label", anchor="north", role="tick"))
    placed, unplaced = [], []
    for k, lab in enumerate(line_labels):
        ident = identity(lab)
        if ident.wavelength_angstrom:
            nm, source = ident.wavelength_angstrom / 10.0, "declared"
        elif _balmer(ident):
            n_u, n_l = _balmer(ident)
            nm, source = hydrogenic_transition_wavelength(n_u, n_l, 1, ident.species.mass_number or 1) * 1e9, "computed"
        else:
            unplaced.append(lab)
            continue
        placed.append((lab, nm, source))
    placed.sort(key=lambda r: r[1])
    # label positions: at the line, pushed right to keep 0.6 cm between neighbours, joined by a leader
    label_x: List[float] = []
    for _lab, nm, _src in placed:
        label_x.append(max(x(nm), label_x[-1] + 0.6) if label_x else x(nm))
    h = 1.6
    for (lab, nm, source), lx in zip(placed, label_x):
        items.append(Polyline.of([(x(nm), 0.0), (x(nm), h)], "component real" if source == "declared" else "approx",
                                 role=f"line:{lab}"))
        if labels:
            items.append(Polyline.of([(x(nm), h), (lx, h + 0.45)], "leader line", role=f"leader:{lab}"))
            items.append(Label((lx - 0.05, h + 0.5), lab.replace("_", "\\_"), "small label,rotate=50",
                               anchor="south west", role=f"line:{lab}"))
    if labels:
        items += [Label((W / 2, -0.8), "wavelength [nm], log scale", "small label", anchor="north", role="axis"),
                  Label((W / 2, 4.3), "declared spectral lines", "label", anchor="south", role="title"),
                  _note("Placed at the wavelength each label declares (IMAS processed\\_line syntax, \\AA); dashed: a "
                        "hydrogen series member placed at its computed vacuum wavelength"
                        + (f"; not placed (no wavelength): {', '.join(unplaced)}" if unplaced else ""), W / 2, -1.4)]
    return Diagram("spectroscopy_spectrum", Scene(tuple(items)),
                   model={"placed": placed, "unplaced": unplaced, "labels": line_labels})
