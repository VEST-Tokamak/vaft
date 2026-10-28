"""The scientific architecture behind the equilibrium, MHD and transport figure, drawn for a manuscript.

The conceptual companion of :func:`~vaft.diagram.cross_domain_scientific_workflows`:
shared data and metadata on the left, the equilibrium operations, then MHD
stability / plasma response and transport, each section ending with the IMAS IDS
its operations write, representative quantities in them, and the quantities kept
only in the solver's own form (keyed *native*, never given an IDS). Where the technical
figure draws each solver with the inputs it reads and each result under its
producer, this one groups them -- inputs into data and metadata cards, solvers
into the operations they perform, results into one quantity list per section -- and
keeps only the relations themselves: which methods are alternatives, and that an
equilibrium feeds MHD and transport.

The figure carries no explanation of itself. The qualifications that matter --
one IDS format holding distinct plasma states, each downstream model checking its
own inputs, arrows marking implemented interfaces rather than validated
workflows, the transport resolver taking a reconstructed equilibrium only --
belong to the caption (see the handoff) and to the technical figure.

Nothing here is restated by hand. Every card, operation and IDS entry names the
technical elements it summarizes (``covers``), and the tests hold the two figures
to each other: every technical input and method is covered exactly once, every
IDS named is one the technical figure's writers produce, and no output without a
writer is given an IDS: the native solver outputs are listed as native, and the
Sauter / Redl return value is left to the technical figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ._concept import Box, box, connector, escape_latex
from ._cross_domain import CONNECTION_STYLE, ELEMENT, REGIONS, connection_legend
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene
from ._workflow import KIND_SYMBOL, KINDS, kind_legend, kind_style
from .pictograms import pictogram

_TITLE = {key: title for key, title, _ in REGIONS}


@dataclass(frozen=True)
class Item:
    """One card of the figure and the technical elements it stands for.

    ``kind`` is a workflow kind; ``title`` the bold first line ("\\n" breaks it);
    ``lines`` the LaTeX lines under it (one starting ``\\itshape`` names a physical
    model); ``idss`` the IDS names printed in typewriter at its foot. ``covers``
    names :data:`vaft.diagram._cross_domain.ELEMENT` keys.
    """

    key: str
    kind: str
    title: str
    covers: Tuple[str, ...]
    lines: Tuple[str, ...] = ()
    idss: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"item {self.key!r}: kind must be one of {KINDS}, not {self.kind!r}")
        unknown = [k for k in self.covers if k not in ELEMENT]
        if not self.covers or unknown:
            raise ValueError(f"item {self.key!r}: covers must name technical elements, not {unknown or 'nothing'}")


I = Item
#: shared data and metadata: what was measured, then the context needed to interpret it and run an operation
CONTEXT: Tuple[Item, ...] = (
    I("measurements", "measured", "Magnetic diagnostics", ("tm_targets", "eq_coils", "ef_magnetics"),
      (r"$I_p$, magnetics, coil currents",)),
    # Thomson T_e, n_e; charge-exchange T_i and toroidal velocity (machine_mapping.charge_exchange)
    I("profiles", "measured", "Kinetic diagnostics\n\\& profiles", ("ef_kinetic", "tr_profiles"),
      (r"$T_e$, $n_e$, $T_i$, $v_\phi$",)),
    I("machine", "machine", "Machine \\& diagnostics", ("eq_machine", "gpec_coils"),
      (r"wall, vessel, coils, sensor positions",)),
    I("models", "prior", "Assumptions \\& parameters",
      ("an_shape", "tm_profiles", "tr_closures", "sr_closures", "gpec_drive"),
      (r"closures $Z_{\mathrm{eff}}$, $\gamma_E$;", r"profile shapes; 3-D excitation")),
    I("conventions", "convention", "Conventions \\& provenance", ("ef_basis", "ch_cocos", "dcon_in"),
      (r"COCOS, profile bases,", r"mode numbers, wall model;", "provenance")),
)
#: the two groups of the left panel: (title, item keys)
CONTEXT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Data", ("measurements", "profiles")),
    ("Metadata", ("machine", "models", "conventions")),
)
#: scientific operations, by domain; lines are the code, then (italic) the physical model it rests on.
#: A card holding alternative methods lists each code with its model in italics under it, in turn
OPERATIONS: Dict[str, Tuple[Item, ...]] = {
    "equilibrium": (
        # an analytical model, though the technical figure files it with the solvers (its spec nodes are solvers)
        I("analytic", "derived", "Analytic\nGrad-Shafranov", ("analytic",),
          ("Solov'ev family", r"\itshape assumed shape, profiles")),
        I("free_boundary", "solver", "Free-boundary\nmodel", ("tokamaker",), ("TokaMaker", r"\itshape assumed profiles")),
        I("reconstruction", "solver", "Reconstruction", ("efit",), ("EFIT", r"\itshape fit with a profile basis")),
        I("refinement", "solver", "Fixed-boundary\nequilibrium", ("chease",), ("CHEASE", r"\itshape prescribed boundary")),
    ),
    "mhd": (
        I("ideal", "solver", "Ideal\nstability", ("dcon",), ("DCON", r"\itshape energy principle")),
        I("response", "solver", "Plasma\nresponse", ("gpec",), ("GPEC", r"\itshape uses DCON")),
        I("tearing", "solver", "Resistive\nstability", ("rdcon",), ("RDCON", r"\itshape resistive layers")),
    ),
    # alternatives that produce the same scientific output share a dash-dot enclosure titled by that physics; each
    # code is its own card inside it, titled by the code
    "transport": (
        I("classical", "derived", "Classical", ("classical",), ("Braginskii", r"\itshape magnetized")),
        I("sauter_redl", "derived", "Sauter / Redl", ("sauter_redl",), (r"\itshape analytic fit",)),
        I("neo", "solver", "NEO", ("neo",), (r"\itshape drift-kinetic",)),
        I("tglf", "solver", "TGLF", ("tglf",), (r"\itshape QL gyrofluid",)),  # reduced, quasilinear: not gyrokinetic
        I("cgyro", "solver", "CGYRO", ("cgyro",), (r"\itshape gyrokinetic",)),
    ),
}
OPERATION: Dict[str, Item] = {op.key: op for ops in OPERATIONS.values() for op in ops}
#: what each downstream domain takes from the shared data & metadata besides an equilibrium: one row of chips in the
#: style (and with the pictogram) of the left-panel card they come from
INPUTS: Dict[str, Tuple[Item, ...]] = {
    "mhd": (I("mhd_modes", "convention", "", ("dcon_in",), (r"mode numbers, wall model",)),
            I("mhd_drive", "prior", "", ("gpec_drive",), (r"3-D coil excitation",))),
    # no v_phi: TGLF and CGYRO run with gamma_E = 0 and no rotation reaches them (tglf/inputs.py, #553)
    "transport": (I("tr_kinetic", "measured", "", ("tr_profiles",), (r"$T_e$, $n_e$, $T_i$",)),
                  # T_i sits with the kinetic profiles; gamma_E is the general input, its value in each run's config
                  I("tr_closures", "prior", "", ("tr_closures", "sr_closures"), (r"$Z_{\mathrm{eff}}$, $\gamma_E$",))),
}
#: alternative methods for the same scientific output, each set in one dash-dot enclosure: the four equilibrium
#: formulations (all writing an equilibrium state; peers, not a chain), the neoclassical and the turbulent models
ALTERNATIVES: Tuple[Tuple[str, ...], ...] = (
    ("analytic", "free_boundary", "reconstruction", "refinement"),
    ("sauter_redl", "neo"),
    ("tglf", "cgyro"),
)
#: the physics an enclosure stands for, printed at its top (the equilibrium one sits in its own section)
GROUP_TITLE: Dict[Tuple[str, ...], str] = {("sauter_redl", "neo"): "Neoclassical", ("tglf", "cgyro"): "Turbulent"}
#: what each section writes to IMAS: (ids, representative quantities, technical elements). Only IDS with a writer
#: are named; a code is named where the section holds several and only one writes the IDS
IMAS: Dict[str, Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = {
    "equilibrium": (("equilibrium", r"$\psi$, $q$, $p$", ("eq_analytic", "eq_model", "eq_reconstructed", "eq_refined")),),
    "mhd": (("mhd_linear", r"$\delta W_n$, $\delta\mathbf B\cdot\nabla\psi$", ("ml_dcon", "ml_gpec")),
            ("ntms", r"$\Delta'$", ("ntms_rdcon",))),
    "transport": (("core_transport", r"$\Gamma_s$, $Q_s$", ("ct_classical", "ct_neo", "ct_tglf")),
                  ("core_profiles", r"$j_{\mathrm{bs}}$, $\sigma_\parallel$", ("ct_neo",)),  # written by NEO
                  ("gyrokinetics_local", r"$\gamma$, $\omega$", ("gk_cgyro",))),  # TGLF has a writer too
}
#: what each section keeps in the solver's own form, with no IMAS mapping yet: (quantities, technical elements).
#: Listed under the IDS with an italic "native" key in place of an IDS name, so it never reads as mapped
NATIVE: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "mhd": ((r"$\delta W_p$, $\delta W_v$; $\Delta'$ matrix", ("dcon_native", "rdcon_native")),),
}
#: native results left out of the figure on purpose, to keep the lists representative: GPEC's resonant response and
#: the Sauter / Redl return value (the technical figure draws both)
OMITTED: Tuple[str, ...] = ("gpec_native", "sr_native")
#: the shared equilibrium state between the equilibrium operations and their consumers: what the IDS holds
STATE = I("state", "standardized", "Equilibrium state", IMAS["equilibrium"][0][2],
          ("\\texttt{" + IMAS["equilibrium"][0][0] + "}: " + IMAS["equilibrium"][0][1],))
#: (source, target): what provides input to what; the equilibrium state branches to both consumers
EDGES: Tuple[Tuple[str, str], ...] = (
    ("context", "equilibrium"),
    ("context", "inputs"),
    ("equilibrium", "state"),
    ("state", "mhd"),
    ("state", "transport"),
)
#: edges that mean "writes" rather than "provides input to": the equilibrium operations write the shared state
WRITES: Tuple[Tuple[str, str], ...] = (("equilibrium", "state"),)
#: an open-headed arrow for a write, distinct from a data input without colour
WRITE_STYLE = "-{Stealth[open,length=2.8mm,width=2.4mm,line width=0.8pt]},absequil!90!black,line width=1.2pt"
#: the legend: each card kind with its pictogram (the cards carry the same pictogram), then the two line styles
CONNECTION_WORDS = {"flow": "provides input to", "write": "writes", "alternative": "alternative methods"}
#: the two method kinds name the primary computational approach: an analytical model or formula, or a numerical
#: method or solver (whether or not the formula is evaluated numerically)
KIND_WORDS = {"measured": "measured", "machine": "configuration", "prior": "assumption",
              "convention": "convention", "derived": "analytical model / formula",
              "solver": "numerical method / solver", "standardized": "shared state"}
#: the analytical models: a closed-form solution, the Braginskii closures and the Sauter / Redl fitted formulas
ANALYTICAL: Tuple[str, ...] = ("analytic", "classical", "sauter_redl")

# ---------------------------------------------------------------------------
# layout [cm]: x from 0 to _WIDTH, y downward from 0
# ---------------------------------------------------------------------------

_WIDTH = 23.5
_MAIN = 6.25  # where the computational sections start, right of the left panel and its arrows
_BOTTOM = -11.6
#: (x0, x1, y_top, y_bottom) of each section
_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "context": (0.0, 5.4, 0.0, _BOTTOM),
    "equilibrium": (_MAIN, _WIDTH, 0.0, -3.45),
    "mhd": (_MAIN, 14.65, -5.45, _BOTTOM),
    "transport": (14.9, _WIDTH, -5.45, _BOTTOM),
}
_X = {"equilibrium": (8.55, 12.77, 16.99, 21.21), "mhd": (7.7, 10.45, 13.2), "transport": (16.45, 19.2, 21.95)}
_W = {"equilibrium": 3.7, "mhd": 2.5, "transport": 2.5}
_ROW = {"equilibrium": -1.8, "mhd": -8.35, "transport": -8.35}
_CARD_H = {"equilibrium": 2.15, "mhd": 3.0, "transport": 3.0}  # a top strip holds the pictogram
#: the top of each section's quantity list
_LIST_TOP = {"mhd": -10.15, "transport": -10.15}
#: transport cards: (x, y, width, height). Classical is a full card; each enclosed code is a half-height card
_TR_PLACE = {"classical": (16.05, -8.35, 2.0, 3.0),
             "sauter_redl": (18.68, -7.93, 2.7, 1.2), "neo": (18.68, -9.22, 2.7, 1.2),
             "tglf": (21.79, -7.93, 2.7, 1.2), "cgyro": (21.79, -9.22, 2.7, 1.2)}
#: room above an enclosure's cards for its title [cm]
_GROUP_HEAD = 0.45
#: the input chips: their row, height, and (x, width) per chip
_INPUT_Y, _INPUT_H = -6.45, 0.55
_INPUT_X = {"mhd": ((8.6, 4.3), (12.65, 3.4)), "transport": ((16.4, 2.6), (19.95, 4.1))}
#: the shared equilibrium state: (x, y, width, height)
_STATE = (14.775, -4.3, 5.6, 0.8)
_CONTEXT_Y = {"measurements": -1.7, "profiles": -3.05, "machine": -5.1, "models": -7.4, "conventions": -9.85}
_CONTEXT_H = {"measurements": 1.0, "profiles": 1.3, "machine": 1.3, "models": 1.75, "conventions": 2.25}
_GROUP_Y = {"Data": -0.85, "Metadata": -4.1}
#: edges drawn along given points: the left panel feeds the first section of each tier; an equilibrium leaves
#: its section for the centre of each consumer
_ROUTES = {
    # from the panel, level with a gap between cards, so no single card reads as the source; the second meets the
    # downstream input row
    ("context", "equilibrium"): ((5.48, -2.3), (6.17, -2.3)),
    ("context", "inputs"): ((5.48, -6.2), (6.17, -6.2)),
    # from the enclosure, not from any one route, and over the top so it stays clear of the arrow to the state
    ("equilibrium", "state"): ((14.775, -3.53), (14.775, -3.82)),
    ("state", "mhd"): ((11.895, -4.3), (10.45, -4.3), (10.45, -5.37)),
    ("state", "transport"): ((17.655, -4.3), (19.2, -4.3), (19.2, -5.37)),
}
_ENCLOSE = 0.13
#: sections: one quiet solid outline, so dashes are left to the alternatives
_SECTION = "draw=black!30,line width=0.7pt,rounded corners=4pt,fill=white"
#: one text height and depth, so every legend label shares a baseline whatever its letters
_LEGEND = "concept plain,text=black!80,text height=1.6ex,text depth=0.45ex"
#: the kind pictogram on every card: its size and its inset from the card's top-left corner [cm]
_ICON, _ICON_INSET = 0.36, 0.3


def _card(item: Item, x: float, y: float, w: float, h: float, role: str, icon: str) -> Box:
    """A card in its kind's style with that kind's pictogram: ``icon="left"`` beside the text (a wide card),
    ``"top"`` in a strip above it (a narrow one), so the pictogram never meets a centred title."""
    outline = box(x, y, w, h, "", style=kind_style(item.kind), role=role).items[0]
    if icon == "left":
        at, tx, ty, tw = (x - 0.5 * w + _ICON_INSET, y), x + 0.22, y, w - 0.3 - 0.44
    else:
        at, tx, ty, tw = (x - 0.5 * w + _ICON_INSET, y + 0.5 * h - _ICON_INSET), x, y - 0.22, w - 0.3
    label = Label((tx, ty), _card_text(item), f"concept plain,text width={tw:.2f}cm", role=role)
    symbol = pictogram(KIND_SYMBOL[item.kind], at, _ICON, role=f"symbol:{role}").items
    return Box(x, y, w, h, (outline, label) + tuple(symbol))


def _card_text(item: Item) -> str:
    """A bold title, then the lines (wrapped to the width), then the IDS names (wrapped at the separators)."""
    parts = ["{\\small\\bfseries " + "\\\\".join(item.title.split("\n")) + "}"] if item.title else []
    parts += ["{\\footnotesize " + (line if "$" in line or "\\" in line else escape_latex(line)) + "}"
              for line in item.lines]
    text = "\\\\[1pt]".join(parts)  # a title breaks where the data says; a chip has no title
    if item.idss:
        text += ("\\\\[2pt]{\\footnotesize\\ttfamily "
                 + "~$\\cdot$ ".join(escape_latex(i) for i in item.idss) + "}")  # breaks after a dot
    return text


def _quantity_items(domain: str, labels: bool) -> List:
    """The section's quantities: one IDS per line with what it holds, then what stays native, keyed in italics."""
    if not labels:
        return []
    x0, *_ = _BOUNDS[domain]
    rows = ["{\\small\\texttt{" + escape_latex(ids) + "}:\\ " + quantities + "}" for ids, quantities, _ in IMAS[domain]]
    rows += ["{\\small\\textit{native}:\\ " + quantities + "}" for quantities, _ in NATIVE.get(domain, ())]
    return [Label((x0 + 0.25, _LIST_TOP[domain]), "\\\\".join(rows), "concept plain,align=left", anchor="north west",
                  role=f"quantities:{domain}")]


def _section(key: str, labels: bool) -> List:
    x0, x1, y0, y1 = _BOUNDS[key]
    items: List = [Polyline.of([(x0, y1), (x1, y1), (x1, y0), (x0, y0)], _SECTION, role=f"region:{key}", closed=True)]
    if labels:
        items.append(Label((x0 + 0.2, y0 - 0.3), "\\textbf{" + escape_latex(_TITLE[key]) + "}", "concept group title",
                           anchor="west", role=f"region title:{key}"))
    return items


def _bounds_box(key: str) -> Box:
    x0, x1, y0, y1 = _BOUNDS[key]
    return Box(0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y0 - y1, ())


def cross_domain_scientific_architecture(*, labels: bool = True) -> Diagram:
    """The scientific architecture of equilibrium, MHD and transport in VAFT, for a manuscript.

    Shared data (diagnostic signals, experimental profiles) and metadata
    (machine and diagnostic configuration, assumptions and parameters,
    conventions and provenance) feed the operations. The alternative
    equilibrium routes share a dash-dot enclosure and CHEASE refines a
    reconstruction; an equilibrium feeds MHD stability / plasma response and
    transport. Each section ends with the IMAS IDS its operations write and
    the quantities kept only in the solver's own form, keyed *native*.
    :func:`~vaft.diagram.cross_domain_scientific_workflows` draws the same
    architecture solver by solver.
    """
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    items: List = []
    boxes: Dict[str, Box] = {}
    # the left panel: data, then metadata
    x0, x1, y0, y1 = _BOUNDS["context"]
    items.append(Polyline.of([(x0, y1), (x1, y1), (x1, y0), (x0, y0)],
                             "draw=black!30,line width=0.7pt,rounded corners=4pt,fill=black!3",
                             role="region:context", closed=True))
    boxes["context"] = _bounds_box("context")
    if labels:
        items.append(Label((x0 + 0.2, y0 - 0.38), "\\textbf{Shared data \\& metadata}", "concept group title",
                           anchor="west", role="region title:context"))
        for title, _ in CONTEXT_GROUPS:
            items.append(Label((x0 + 0.2, _GROUP_Y[title]), "{\\small\\bfseries\\color{black!75}" + title + "}",
                               "concept plain", anchor="west", role=f"context group:{title}"))
    for item in CONTEXT:
        b = _card(item, 0.5 * (x0 + x1), _CONTEXT_Y[item.key], x1 - x0 - 0.4, _CONTEXT_H[item.key],
                  f"context:{item.key}", icon="left")
        boxes[item.key] = b
        items += list(b.items)
    # the computational sections, their operations and the IDS they write
    for domain in ("equilibrium", "mhd", "transport"):
        items += _section(domain, labels)
        boxes[domain] = _bounds_box(domain)
        for i, op in enumerate(OPERATIONS[domain]):
            place = _TR_PLACE[op.key] if domain == "transport" else (_X[domain][i], _ROW[domain], _W[domain],
                                                                     _CARD_H[domain])
            b = _card(op, *place, f"operation:{op.key}", icon="top")
            boxes[op.key] = b
            items += list(b.items)
        for chip, (cx, cw) in zip(INPUTS.get(domain, ()), _INPUT_X.get(domain, ())):
            b = _card(chip, cx, _INPUT_Y, cw, _INPUT_H, f"input:{chip.key}", icon="left")
            boxes[chip.key] = b
            items += list(b.items)
        if domain != "equilibrium":
            items += _quantity_items(domain, labels)
    b = _card(STATE, *_STATE, "state", icon="left")
    boxes["state"] = b
    items += list(b.items)
    for keys in ALTERNATIVES:  # around the member cards, with room for the group's title above them
        bx0 = min(boxes[k].x - 0.5 * boxes[k].width for k in keys) - _ENCLOSE
        bx1 = max(boxes[k].x + 0.5 * boxes[k].width for k in keys) + _ENCLOSE
        by1 = min(boxes[k].y - 0.5 * boxes[k].height for k in keys) - _ENCLOSE
        by0 = max(boxes[k].y + 0.5 * boxes[k].height for k in keys) + _ENCLOSE
        title = GROUP_TITLE.get(keys)
        if title:
            by0 += _GROUP_HEAD
            if labels:
                items.append(Label((0.5 * (bx0 + bx1), by0 - 0.26), "{\\small\\bfseries " + title + "}", "concept plain",
                                   role="group title:" + ",".join(keys)))
        items.append(Polyline.of([(bx0, by1), (bx1, by1), (bx1, by0), (bx0, by0)], CONNECTION_STYLE["alternative"],
                                 role="alternative:" + ",".join(keys), closed=True))
        boxes["alternative:" + ",".join(keys)] = Box(0.5 * (bx0 + bx1), 0.5 * (by0 + by1), bx1 - bx0, by0 - by1, ())
    for src, dst in EDGES:
        role = f"edge:flow:{src}->{dst}"
        path = _ROUTES.get((src, dst))
        if path is None:
            a = connector(boxes[src], boxes[dst], role=role)
            path = (a.start, a.end)
        if (src, dst) in WRITES:
            role, style = f"edge:write:{src}->{dst}", WRITE_STYLE
        else:
            style = CONNECTION_STYLE["flow"]
        if len(path) > 2:  # a routed arrow: the style carries the head
            items.append(Polyline.of(path, style, role=role))
        else:
            items.append(Arrow(path[0], path[1], style, role=role))
    if labels:  # the kinds with their pictograms, then the two line styles
        cards = CONTEXT + tuple(OPERATION.values()) + (STATE,) + tuple(c for row in INPUTS.values() for c in row)
        used = [k for k in KINDS if any(i.kind == k for i in cards)]
        legend, y = kind_legend(used, 0.0, _WIDTH, _BOTTOM - 0.65, label_style=_LEGEND, labels=KIND_WORDS)
        items += legend
        items += connection_legend(0.0, _WIDTH, y - 0.05, ("flow", "write", "alternative"), CONNECTION_WORDS, _LEGEND,
                                   styles={"write": WRITE_STYLE}, arrows=("flow", "write"))
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"context": CONTEXT, "operations": OPERATIONS, "alternatives": ALTERNATIVES, "imas": IMAS, "native": NATIVE,
             "inputs": INPUTS, "state": STATE,
             "edges": EDGES, "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in boxes.items()}}
    return Diagram("cross_domain_scientific_architecture", Scene(tuple(items)), model=model)
