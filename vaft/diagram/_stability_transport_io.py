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
from ._workflow import kind_style
from ._vaft_concepts import _MODE_OFFSET, _check_labels, _record, box, integrated_scientific_framework

#: the two calculations: (key, the cross-domain section they summarize, title, the physics they cover)
CALCULATIONS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("mhd_stability", "mhd", "MHD Stability Model", ("ideal & resistive energy principle,", "perturbed equilibria")),
    ("transport", "transport", "Transport Model", ("classical, neoclassical, turbulent",)),
)
#: the title of each calculation's output card
_OUTPUT_TITLE = {"mhd_stability": "IMAS Stability IDS", "transport": "IMAS Transport IDS"}


def _codes(section: str) -> Tuple[str, ...]:
    """The codes of a section, as the cross-domain figure names them: the first plain line, else the card title."""
    names = []
    for op in OPERATIONS[section]:
        plain = [line for line in op.lines if not line.startswith("\\itshape")]
        # a code family named "A / B" in the cross-domain figure is listed with the others' commas
        names.append((plain[0] if plain else op.title).replace(" / ", ", "))
    return tuple(names)


#: the codes behind each calculation, printed in the style of the equilibrium panel's routes
#: the order the codes are listed in where it differs from the cross-domain figure's: ideal, then resistive
#: stability, then the perturbed equilibrium built on the ideal one
_CODE_ORDER = {"mhd_stability": ("DCON", "RDCON", "GPEC")}


def _ordered(key: str, codes: Tuple[str, ...]) -> Tuple[str, ...]:
    order = _CODE_ORDER.get(key)
    if order is None:
        return codes
    if sorted(order) != sorted(codes):
        raise ValueError(f"the code order of {key} {order} does not match the codes {codes}")
    return order


CODES: Dict[str, Tuple[str, ...]] = {key: _ordered(key, _codes(section)) for key, section, _, _ in CALCULATIONS}
#: the IDS each calculation writes, with representative quantities: (ids, quantities)
OUTPUTS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    key: tuple((ids, quantities) for ids, quantities, _ in IMAS[section]) for key, section, _, _ in CALCULATIONS}
#: chips named by their role here rather than by their quantities: the transport closures' Z_eff is a kinetic
#: profile in this figure, and what remains of the chip is the closure and how the profiles are fitted; the 3-D
#: coil excitation is machine data (``coils_non_axisymmetric``), drawn with the experiment, not assumed
_ASSUMPTION_TEXT = {"tr_closures": ("physics closure, fitting method",), "mhd_drive": ()}
#: the external assumptions, gathered from the inputs each downstream section takes besides measured profiles
ASSUMPTIONS: Tuple[str, ...] = tuple(line for section in ("transport", "mhd") for chip in INPUTS[section]
                                     if chip.kind != "measured"
                                     for line in _ASSUMPTION_TEXT.get(chip.key, chip.lines))
#: the shared inputs: (key, title, IDS name or "", detail)
SHARED_INPUTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("equilibrium", "Equilibrium", "equilibrium", r"$\psi$, $p$, $J_\phi$, $q$"),
    ("core_profiles", "Kinetic Profiles", "core_profiles", r"$n_i$, $n_e$, $T_e$, $T_i$, $Z_{\rm eff}$, $v_\phi$"),
    ("assumptions", "Assumptions", "", "; ".join(ASSUMPTIONS)),
)
#: the kind of each box, as the cross-domain figure classifies it, so a colour means the same in both figures: the
#: IDS are the shared state, the assumptions an assumption, the two physics models numerical solvers
KIND: Dict[str, str] = {"equilibrium": "standardized", "core_profiles": "standardized", "assumptions": "prior",
                        "mhd_stability": "solver", "transport": "solver", "output": "standardized"}
#: which shared inputs each calculation reads (the measured profiles enter transport only)
READS: Dict[str, Tuple[str, ...]] = {
    "mhd_stability": ("equilibrium", "assumptions"),
    "transport": ("equilibrium", "core_profiles", "assumptions"),
}

#: what the reconstruction and the kinetic profiles are made from, as IMAS IDS, a line each: wall and axisymmetric
#: coils; 3-D coils and heating; measurements. The 3-D coil excitation is ``coils_non_axisymmetric``, so it is drawn
#: here as machine data, not among the assumptions
EXPERIMENT_TITLE = "IMAS Machine Configuration & Experimental Data IDS"
EXPERIMENT_IDS: Tuple[Tuple[str, ...], ...] = (
    ("wall", "pf_active", "pf_passive", "em_coupling"),
    ("coils_non_axisymmetric", "nbi", "ec_launchers"),
    ("magnetics", "thomson_scattering", "charge_exchange"),
)
#: the representative quantities of each, as VAFT reads them: limiter and vessel outline
#: (``description_2d.*.outline``), coil and passive-loop currents (``coil.current``, ``loop.current``), the mutual
#: inductance matrices (``mutual_*``), 3-D coil currents, EC launched power and frequency (``beam.power_launched``,
#: ``beam.frequency``), plasma current, poloidal field probes, flux loops and diamagnetic flux, Thomson and
#: charge-exchange profiles. For NBI VAFT maps only the units, so the quantities are the data dictionary's
#: ``unit.power_launched`` and ``unit.energy``.
EXPERIMENT_QUANTITIES: Dict[str, str] = {
    "wall": r"$(R,Z)_{\rm wall}$",
    "pf_active": r"$I_{\rm coil}$",
    "pf_passive": r"$I_{\rm eddy}$",
    "em_coupling": r"$M_{ij}$",
    "coils_non_axisymmetric": r"$I_{\rm 3D}$",
    "nbi": r"$P_{\rm NBI}$, $E_b$",
    "ec_launchers": r"$P_{\rm EC}$, $f$",
    "magnetics": r"$I_p$, $B_p$, $\psi_p$, $\Phi_{\rm dia}$",
    "thomson_scattering": r"$T_e$, $n_e$",
    "charge_exchange": r"$T_i$, $v_\phi$",
}
assert set(EXPERIMENT_QUANTITIES) == {i for line in EXPERIMENT_IDS for i in line}
#: what the kinetic profiles are made from inside the shared input, when they are drawn as derived
PROFILE_SOURCES: Tuple[str, ...] = ("equilibrium", "assumptions")

# layout [cm], centred on x = 0 and spanning the height of the equilibrium panel (modes at +-3.5, 1.45 high)
_HALF = 6.0
_TOP = _MODE_OFFSET + 0.725  # level with the top of the upper research modes
_INPUT_H = 1.9
_INPUT_Y = _TOP - 0.3 - 0.5 * _INPUT_H
#: input card widths, left to right: the assumptions card is the widest, its clauses being the longest
_INPUT_W = (3.4, 3.6, 4.6)
#: the extra half width the input group takes when arrows run between its cards
_PROFILE_WIDEN = 0.6
_GROUP_BOTTOM = _INPUT_Y - 0.5 * _INPUT_H - 0.25
_CALC_H, _CALC_W = 2.0, 5.3
_CALC_Y = _GROUP_BOTTOM - 1.05 - 0.5 * _CALC_H
_OUT_TOP, _OUT_W = _CALC_Y - 0.5 * _CALC_H - 0.85, 5.3
_GAP = 0.08
#: codes per line under a calculation
_CODES_PER_LINE = 3
#: the half width of the equilibrium panel (research modes at +-3.5, 4.5 wide) and the space between the panels
_LEFT_HALF = _MODE_OFFSET + 2.25
_BETWEEN = 2.2
#: the horizontal distance between the centres of the two panels
_PANEL_DX = _LEFT_HALF + _BETWEEN + _HALF


def _calc_text(key: str, title: str, physics: Tuple[str, ...]) -> str:
    text = "\\textbf{" + escape_latex(title) + "}"
    # the physics in italics, the codes as the equilibrium panel prints its routes: small, upright, comma-separated
    text += "".join("\\\\{\\small\\itshape " + escape_latex(p) + "}" for p in physics)
    codes = [escape_latex(c) for c in CODES[key]]
    lines = [", ".join(codes[i:i + _CODES_PER_LINE]) for i in range(0, len(codes), _CODES_PER_LINE)]
    return text + "\\\\[2pt]" + "\\\\".join("{\\small " + line + "}" for line in lines)


def _output_text(key: str) -> str:
    rows = ["{\\small\\texttt{" + escape_latex(ids) + "}: " + quantities + "}" for ids, quantities in OUTPUTS[key]]
    return "\\textbf{" + _OUTPUT_TITLE[key] + "}\\\\" + "\\\\".join(rows)


def stability_transport_io(*, labels: bool = True, derived_profiles: bool = False) -> Diagram:
    r"""Shared inputs consumed by MHD stability and transport, and the IDS each writes.

    Top: the shared plasma state -- the ``equilibrium`` IDS, the measured kinetic
    profiles in ``core_profiles`` and external assumptions (physics closure and fitting method, mode
    numbers and wall model). Middle: MHD stability (ideal
    & resistive energy principle, perturbed equilibria) reads the equilibrium and the
    assumptions; transport (classical, neoclassical, turbulent) reads all
    three. The codes sit beneath each, in the style the equilibrium panel
    gives its routes. Bottom: the IDS each calculation writes -- ``mhd_linear`` and
    ``ntms``; ``core_transport``, an updated ``core_profiles`` and
    ``gyrokinetics_local`` -- with representative quantities.

    Arrows mark implemented interfaces, not a fixed execution order; the
    figure carries no annotations of its own (what each calculation reads is
    in ``model["reads"]`` and the caption).

    ``derived_profiles=True`` draws the kinetic profiles as made from the
    other two inputs (equilibrium -> kinetic profiles <- external
    assumptions): the group widens so arrows fit between its cards, and its
    title moves to the right end of the outline, leaving the top of the
    kinetic profiles card free for an arrow from above.
    """
    labels = _check_labels(labels)
    if not isinstance(derived_profiles, bool):
        raise ValueError(f"derived_profiles must be True or False, not {derived_profiles!r}")
    half = _HALF + (_PROFILE_WIDEN if derived_profiles else 0.0)
    input_gap = 0.5 * (2 * half - sum(_INPUT_W))
    col = {"mhd_stability": -0.5 * (half * 2 - _CALC_W), "transport": 0.5 * (half * 2 - _CALC_W)}
    items: List = []
    edges: List = []
    nodes: Dict[str, Box] = {}
    # the shared inputs, inside one quiet group whose title sits on its outline
    xs, left = [], -half
    for w in _INPUT_W:
        xs.append(left + 0.5 * w)
        left += w + input_gap
    gx0, gx1 = -half - 0.25, half + 0.25
    items.append(Polyline.of([(gx0, _GROUP_BOTTOM), (gx1, _GROUP_BOTTOM), (gx1, _TOP), (gx0, _TOP)],
                             "concept frame", role="group:inputs", closed=True))
    if labels:
        items.append(Label((gx1 - 0.3, _TOP) if derived_profiles else (gx0 + 0.3, _TOP),
                           "\\textbf{Shared Plasma State}", "concept group title,fill=white,inner sep=2pt",
                           anchor="east" if derived_profiles else "west", role="group:inputs"))
    for (key, title, ids, detail), x, w in zip(SHARED_INPUTS, xs, _INPUT_W):
        text = "\\textbf{" + escape_latex(title) + "}"
        if ids:
            text += "\\\\{\\small\\ttfamily " + escape_latex(ids) + "}\\\\{\\footnotesize " + detail + "}"
        else:
            text += "".join("\\\\{\\footnotesize " + part + "}" for part in ASSUMPTIONS)
        b = box(x, _INPUT_Y, w, _INPUT_H, text, style=kind_style(KIND[key]), role=f"input:{key}", latex=True)
        nodes[key] = b
        items += list(b.items)
    if derived_profiles:
        target = nodes["core_profiles"]
        for src in PROFILE_SOURCES:
            s_box = nodes[src]
            sign = 1.0 if s_box.x < target.x else -1.0
            items.append(Arrow((s_box.x + sign * (0.5 * s_box.width + _GAP), _INPUT_Y),
                               (target.x - sign * (0.5 * target.width + _GAP), _INPUT_Y), "connector",
                               role=f"edge:{src}->core_profiles"))
            _record(edges, src, "core_profiles", "forward")
    # the calculations, each fed from the input group and annotated with what it reads
    for key, _, title, physics in CALCULATIONS:
        x = col[key]
        b = box(x, _CALC_Y, _CALC_W, _CALC_H, _calc_text(key, title, physics), style=kind_style(KIND[key]),
                role=f"node:{key}", latex=True)
        nodes[key] = b
        items += list(b.items)
        start, end = (x, _GROUP_BOTTOM - _GAP), (x, _CALC_Y + 0.5 * _CALC_H + _GAP)
        items.append(Arrow(start, end, "connector", role=f"edge:inputs->{key}"))
        for src in READS[key]:
            _record(edges, src, key, "forward")
        # what it writes
        # both output boxes take the taller one's height, so their bottoms align
        h = 0.75 + 0.48 * max(len(v) for v in OUTPUTS.values())
        out = box(x, _OUT_TOP - 0.5 * h, _OUT_W, h, _output_text(key), style=kind_style(KIND["output"]),
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
                          "derived_profiles": PROFILE_SOURCES if derived_profiles else (), "half": half,
                          "kinds": dict(KIND),
                          "edges": tuple(edges),
                          "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in nodes.items()}})


#: the experiment box mixes machine configuration and measurement, so it takes no kind of its own: a plain outline
_EXPERIMENT_STYLE = "draw=black!60,line width=1.0pt,rounded corners=3pt,fill=black!4"


def _experiment_text() -> str:
    # each IDS as the output boxes print theirs: the name in typewriter, a colon, its quantities
    rows = ["{\\small " + " $\\cdot$ ".join("\\texttt{" + escape_latex(i) + "}: " + EXPERIMENT_QUANTITIES[i] for i in line)
            + "}" for line in EXPERIMENT_IDS]
    return "\\textbf{" + escape_latex(EXPERIMENT_TITLE) + "}\\\\" + "\\\\".join(rows)


def equilibrium_stability_transport(*, labels: bool = True, measurements: bool = False) -> Diagram:
    r"""The equilibrium as a common representation, and what stability and transport do with it.

    Left, four equilibrium routes -- reconstruction (EFIT),
    theory (Solov'ev; Guazzotto & Freidberg), fixed-boundary (CHEASE) and
    free-boundary models (TokaMaker, TES) -- exchanging with the IMAS
    equilibrium IDS, its key ``equilibrium`` named, as in
    ``integrated_scientific_framework("equilibrium", frame=False,
    analysis=False, references=False, ids_key=True, boundary_models=True)``. Right, the :func:`stability_transport_io` view. Both are
    drawn on one canvas at one scale, so their fonts and box styles match; an
    arrow from the equilibrium IDS to the shared inputs marks that the same
    equilibrium is the input downstream. No panel letters: the two halves read
    as one flow.

    Every box takes the kind the cross-domain figure gives it, so its colour
    means what it means there: the IDS are the shared state; reconstruction,
    the boundary models and the two physics models numerical solvers; theory
    an analytical model; the assumptions an assumption. The four equilibrium
    routes are drawn alike, and no legend is drawn: the caption says what the
    styles mean.

    ``measurements=True`` adds what the reconstruction is made from: one
    IMAS Machine Configuration & Experimental Data IDS box above the panels
    (``wall``, ``pf_active``, ``pf_passive``, ``em_coupling``,
    ``coils_non_axisymmetric``; ``nbi``, ``ec_launchers``, ``magnetics``,
    ``thomson_scattering``, ``charge_exchange``), consumed by the
    reconstruction and by the kinetic profiles. It mixes configuration and
    measurement, so it takes a plain outline rather than a kind. The reconstruction then only writes the equilibrium IDS, and the
    kinetic profiles are drawn as made from the equilibrium and the external
    assumptions as well (``stability_transport_io(derived_profiles=True)``).
    """
    labels = _check_labels(labels)
    if not isinstance(measurements, bool):
        raise ValueError(f"measurements must be True or False, not {measurements!r}")
    a = integrated_scientific_framework("equilibrium", labels=labels, frame=False, analysis=False, references=False,
                                        ids_key=True, boundary_models=True,
                                        one_way=("experiment",) if measurements else ())
    b = stability_transport_io(labels=labels, derived_profiles=measurements)
    dx = _LEFT_HALF + _BETWEEN + b.model["half"]
    items: List = list(a.scene.items) + list(b.scene.transformed(offset=(dx, 0.0)).items)
    # the hand-off: from the right of the hub, out between the panels, up into the shared inputs at the level of
    # the equilibrium card
    hub_x = max(p[0] for item in a.scene.role("node:hub") if isinstance(item, Polyline) for p in item.points)
    eq_x, eq_y, eq_w, _ = b.model["boxes"]["equilibrium"]
    eq_left = dx + eq_x - 0.5 * eq_w
    between = 0.5 * (_LEFT_HALF + dx - b.model["half"] - 0.25)
    items.append(Polyline.of([(hub_x + 0.08, 0.0), (between, 0.0), (between, eq_y), (eq_left - 0.08, eq_y)],
                             "connector", role="edge:hub->equilibrium"))
    edges = tuple(a.model["edges"]) + (("hub", "equilibrium", "forward"),) + tuple(b.model["edges"])
    if measurements:
        # the experiment box above the panels, between the reconstruction and the kinetic profiles, feeding both
        efit = next(i for i in a.scene.role("node:experiment") if isinstance(i, Polyline))
        efit_x = 0.5 * (min(p[0] for p in efit.points) + max(p[0] for p in efit.points))
        efit_top = max(p[1] for p in efit.points)
        kin_x, kin_y, _, kin_h = b.model["boxes"]["core_profiles"]
        kin_x += dx
        h, w = 1.95, 14.2
        cx, cy = 0.5 * (efit_x + kin_x), _TOP + 0.8 + 0.5 * h
        data = box(cx, cy, w, h, _experiment_text(), style=_EXPERIMENT_STYLE, role="node:experiment_data", latex=True)
        items += list(data.items)
        items.append(Polyline.of([(cx - 0.5 * w - _GAP, cy), (efit_x, cy), (efit_x, efit_top + _GAP)], "connector",
                                 role="edge:experiment_data->experiment"))
        items.append(Polyline.of([(cx + 0.5 * w + _GAP, cy), (kin_x, cy), (kin_x, kin_y + 0.5 * kin_h + _GAP)],
                                 "connector", role="edge:experiment_data->core_profiles"))
        edges += (("experiment_data", "experiment", "forward"), ("experiment_data", "core_profiles", "forward"))
    return Diagram("equilibrium_stability_transport", Scene(tuple(items)),
                   model={"equilibrium": a.model, "io": {k: v for k, v in b.model.items() if k != "boxes"},
                          "measurements": tuple(i for line in EXPERIMENT_IDS for i in line) if measurements
                          else (), "edges": edges})
