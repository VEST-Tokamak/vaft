"""Stability and transport as consumers and producers of a shared plasma state, drawn as inputs and outputs.

Two builders:

:func:`stability_transport_io`
    Shared inputs -- the ``equilibrium`` IDS, the measured kinetic profiles in
    ``core_profiles`` and external assumptions -- are consumed by two
    calculations, MHD stability and transport, and each writes its results
    back as IMAS IDS.
:func:`equilibrium_stability_transport`
    The manuscript figure: on the left the research modes around the IMAS
    equilibrium IDS (:func:`~vaft.diagram.integrated_scientific_framework`
    without frame, analysis or references), on the right the input-output view
    above, on one canvas at one scale, so both halves share fonts and box
    styles; one arrow hands the equilibrium from left to right.

The input-output view is the simplified form of
:func:`~vaft.diagram.cross_domain_scientific_architecture`: it keeps only what
goes in, what is computed and what comes out. Nothing is restated by hand: the
codes, the IDS each calculation writes with their representative quantities,
and the assumptions are read from that figure's definitions
(:mod:`vaft.diagram._cross_domain_concept`), so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ._concept import Box, escape_latex
from ._cross_domain_concept import IMAS, INPUTS, OPERATIONS
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene
from ._vaft_concepts import _MODE_OFFSET, _check_labels, _record, box, integrated_scientific_framework

#: the two calculations: (key, the cross-domain section they summarize, title, the physics they cover)
CALCULATIONS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("mhd_stability", "mhd", "MHD Stability", ("ideal and resistive stability,", "plasma response")),
    ("transport", "transport", "Transport", ("classical, neoclassical, turbulent",)),
)


def _codes(section: str) -> Tuple[str, ...]:
    """The codes of a section, as the cross-domain figure names them: the first plain line, else the card title."""
    names = []
    for op in OPERATIONS[section]:
        plain = [line for line in op.lines if not line.startswith("\\itshape")]
        names.append(plain[0] if plain else op.title)
    return tuple(names)


#: the codes behind each calculation, printed in the style of the equilibrium panel's routes
CODES: Dict[str, Tuple[str, ...]] = {key: _codes(section) for key, section, _, _ in CALCULATIONS}
#: the IDS each calculation writes, with representative quantities: (ids, quantities)
OUTPUTS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    key: tuple((ids, quantities) for ids, quantities, _ in IMAS[section]) for key, section, _, _ in CALCULATIONS}
#: a chip's lines carry no title in the cross-domain figure (its row says what it is); here one is named
_ASSUMPTION_PREFIX = {"tr_closures": "closures "}
#: the external assumptions, gathered from the inputs each downstream section takes besides measured profiles
ASSUMPTIONS: Tuple[str, ...] = tuple(_ASSUMPTION_PREFIX.get(chip.key, "") + line
                                     for section in ("transport", "mhd") for chip in INPUTS[section]
                                     if chip.kind != "measured" for line in chip.lines)
#: the shared inputs: (key, title, IDS name or "", detail)
SHARED_INPUTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("equilibrium", "Equilibrium", "equilibrium", r"$\psi(R,Z)$, $q$, geometry"),
    ("core_profiles", "Kinetic Profiles", "core_profiles", r"$n_i$, $n_e$, $T_e$, $T_i$, $v_\phi$"),
    ("assumptions", "External Assumptions", "", "; ".join(ASSUMPTIONS)),
)
#: which shared inputs each calculation reads (the measured profiles enter transport only)
READS: Dict[str, Tuple[str, ...]] = {
    "mhd_stability": ("equilibrium", "assumptions"),
    "transport": ("equilibrium", "core_profiles", "assumptions"),
}

# layout [cm], centred on x = 0 and spanning the height of the equilibrium panel (modes at +-3.5, 1.45 high)
_HALF = 5.75
_TOP = _MODE_OFFSET + 0.725  # level with the top of the upper research modes
_INPUT_H = 1.9
_INPUT_Y = _TOP - 0.3 - 0.5 * _INPUT_H
#: input card widths, left to right: the assumptions card is the widest, its clauses being the longest
_INPUT_W = (3.4, 3.4, 4.3)
_INPUT_GAP = 0.5 * (2 * _HALF - sum(_INPUT_W))
_GROUP_BOTTOM = _INPUT_Y - 0.5 * _INPUT_H - 0.25
_CALC_H, _CALC_W = 2.0, 5.3
_CALC_Y = _GROUP_BOTTOM - 1.05 - 0.5 * _CALC_H
_OUT_TOP, _OUT_W = _CALC_Y - 0.5 * _CALC_H - 0.85, 5.3
_COL = {"mhd_stability": -0.5 * (_HALF * 2 - _CALC_W), "transport": 0.5 * (_HALF * 2 - _CALC_W)}
_GAP = 0.08
#: codes per line under a calculation
_CODES_PER_LINE = 3
#: the horizontal distance between the centres of the two panels
_PANEL_DX = 2 * _HALF + 2.2


def _calc_text(key: str, title: str, physics: Tuple[str, ...]) -> str:
    text = "\\textbf{" + escape_latex(title) + "}"
    # the physics in italics, the codes as the equilibrium panel prints its routes: small, upright, comma-separated
    text += "".join("\\\\{\\small\\itshape " + escape_latex(p) + "}" for p in physics)
    codes = [escape_latex(c) for c in CODES[key]]
    lines = [", ".join(codes[i:i + _CODES_PER_LINE]) for i in range(0, len(codes), _CODES_PER_LINE)]
    return text + "\\\\[2pt]" + "\\\\".join("{\\small " + line + "}" for line in lines)


def _output_text(key: str) -> str:
    rows = ["{\\small\\texttt{" + escape_latex(ids) + "}: " + quantities + "}" for ids, quantities in OUTPUTS[key]]
    return "\\textbf{IMAS Output IDS}\\\\" + "\\\\".join(rows)


def stability_transport_io(*, labels: bool = True) -> Diagram:
    r"""Shared inputs consumed by MHD stability and transport, and the IDS each writes.

    Top: the shared inputs -- the ``equilibrium`` IDS, the measured kinetic
    profiles in ``core_profiles`` and external assumptions (closures, mode
    numbers and wall model, 3-D coil excitation). Middle: MHD stability (ideal
    and resistive stability, plasma response) reads the equilibrium and the
    assumptions; transport (classical, neoclassical, turbulent) reads all
    three. The codes sit beneath each, in the style the equilibrium panel
    gives its routes. Bottom: the IDS each calculation writes -- ``mhd_linear`` and
    ``ntms``; ``core_transport``, an updated ``core_profiles`` and
    ``gyrokinetics_local`` -- with representative quantities.

    Arrows mark implemented interfaces, not a fixed execution order; the
    figure carries no annotations of its own (what each calculation reads is
    in ``model["reads"]`` and the caption).
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    nodes: Dict[str, Box] = {}
    # the shared inputs, inside one quiet group whose title sits on its outline
    xs, left = [], -_HALF
    for w in _INPUT_W:
        xs.append(left + 0.5 * w)
        left += w + _INPUT_GAP
    gx0, gx1 = -_HALF - 0.25, _HALF + 0.25
    items.append(Polyline.of([(gx0, _GROUP_BOTTOM), (gx1, _GROUP_BOTTOM), (gx1, _TOP), (gx0, _TOP)],
                             "concept frame", role="group:inputs", closed=True))
    if labels:
        items.append(Label((gx0 + 0.3, _TOP), "\\textbf{Shared inputs}",
                           "concept group title,fill=white,inner sep=2pt", anchor="west", role="group:inputs"))
    for (key, title, ids, detail), x, w in zip(SHARED_INPUTS, xs, _INPUT_W):
        text = "\\textbf{" + escape_latex(title) + "}"
        if ids:
            text += "\\\\{\\small\\ttfamily " + escape_latex(ids) + "}\\\\{\\footnotesize " + detail + "}"
        else:
            text += "".join("\\\\{\\footnotesize " + part + "}" for part in ASSUMPTIONS)
        b = box(x, _INPUT_Y, w, _INPUT_H, text, style="concept hub" if ids else "concept box", role=f"input:{key}",
                latex=True)
        nodes[key] = b
        items += list(b.items)
    # the calculations, each fed from the input group and annotated with what it reads
    for key, _, title, physics in CALCULATIONS:
        x = _COL[key]
        b = box(x, _CALC_Y, _CALC_W, _CALC_H, _calc_text(key, title, physics), style="concept strong",
                role=f"node:{key}", latex=True)
        nodes[key] = b
        items += list(b.items)
        start, end = (x, _GROUP_BOTTOM - _GAP), (x, _CALC_Y + 0.5 * _CALC_H + _GAP)
        items.append(Arrow(start, end, "connector", role=f"edge:inputs->{key}"))
        for src in READS[key]:
            _record(edges, src, key, "forward")
        # what it writes
        h = 0.75 + 0.48 * len(OUTPUTS[key])
        out = box(x, _OUT_TOP - 0.5 * h, _OUT_W, h, _output_text(key), style="concept hub",
                  role=f"output:{key}", latex=True)
        nodes[f"output:{key}"] = out
        items += list(out.items)
        items.append(Arrow((x, _CALC_Y - 0.5 * _CALC_H - _GAP), (x, _OUT_TOP + _GAP), "connector",
                           role=f"edge:{key}->output:{key}"))
        for ids, _ in OUTPUTS[key]:
            _record(edges, key, ids, "forward")
    return Diagram("stability_transport_io", Scene(tuple(items)),
                   model={"inputs": tuple(k for k, *_ in SHARED_INPUTS), "reads": dict(READS),
                          "calculations": tuple(k for k, *_ in CALCULATIONS), "codes": dict(CODES),
                          "outputs": {k: tuple(i for i, _ in v) for k, v in OUTPUTS.items()},
                          "edges": tuple(edges),
                          "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in nodes.items()}})


def equilibrium_stability_transport(*, labels: bool = True) -> Diagram:
    r"""The equilibrium as a common representation, and what stability and transport do with it.

    Left, the four research modes -- experiment (EFIT), modelling (CHEASE,
    TokaMaker), theory (Solov'ev; Guazzotto & Freidberg) and data-driven
    models -- exchanging with the IMAS equilibrium IDS, its key
    ``equilibrium`` named, as in ``integrated_scientific_framework(
    "equilibrium", frame=False, analysis=False, references=False,
    ids_key=True)``. Right, the :func:`stability_transport_io` view. Both are
    drawn on one canvas at one scale, so their fonts and box styles match; an
    arrow from the equilibrium IDS to the shared inputs marks that the same
    equilibrium is the input downstream. No panel letters: the two halves read
    as one flow.
    """
    labels = _check_labels(labels)
    a = integrated_scientific_framework("equilibrium", labels=labels, frame=False, analysis=False, references=False,
                                        ids_key=True)
    b = stability_transport_io(labels=labels)
    items: List = list(a.scene.items) + list(b.scene.transformed(offset=(_PANEL_DX, 0.0)).items)
    # the hand-off: from the right of the hub, out between the panels, up into the shared inputs at the level of
    # the equilibrium card
    hub_x = max(p[0] for item in a.scene.role("node:hub") if isinstance(item, Polyline) for p in item.points)
    eq_x, eq_y, eq_w, _ = b.model["boxes"]["equilibrium"]
    eq_left = _PANEL_DX + eq_x - 0.5 * eq_w
    between = 0.5 * (_HALF + _PANEL_DX - _HALF - 0.25)
    items.append(Polyline.of([(hub_x + 0.08, 0.0), (between, 0.0), (between, eq_y), (eq_left - 0.08, eq_y)],
                             "connector", role="edge:hub->equilibrium"))
    edges = tuple(a.model["edges"]) + (("hub", "equilibrium", "forward"),) + tuple(b.model["edges"])
    return Diagram("equilibrium_stability_transport", Scene(tuple(items)),
                   model={"equilibrium": a.model, "io": {k: v for k, v in b.model.items() if k != "boxes"},
                          "edges": edges})
