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
      (r"closures $Z_{\mathrm{eff}}$, $T_i$, $\gamma_E$;", r"profile shapes; 3-D excitation")),
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
        I("analytic", "solver", "Analytic\nGrad-Shafranov", ("analytic",),
          ("Solov'ev family", r"\itshape assumed shape, profiles")),
        I("free_boundary", "solver", "Free-boundary\nmodel", ("tokamaker",), ("TokaMaker", r"\itshape assumed profiles")),
        I("reconstruction", "solver", "Reconstruction", ("efit",), ("EFIT", r"\itshape fit with a profile basis")),
        I("refinement", "solver", "Fixed-boundary\nrefinement", ("chease",), ("CHEASE", r"\itshape fixed boundary")),
    ),
    "mhd": (
        I("ideal", "solver", "Ideal\nstability", ("dcon",), ("DCON", r"\itshape energy principle")),
        I("response", "solver", "Plasma\nresponse", ("gpec",), ("GPEC", r"\itshape uses DCON")),
        I("tearing", "solver", "Resistive\nstability", ("rdcon",), ("RDCON", r"\itshape resistive layers")),
    ),
    "transport": (
        I("classical", "derived", "Classical", ("classical",), ("Braginskii", r"\itshape magnetized")),
        I("neoclassical", "solver", "Neoclassical", ("sauter_redl", "neo"),
          ("Sauter / Redl", r"\itshape closed-form fit", "NEO", r"\itshape drift-kinetic")),
        I("turbulent", "solver", "Turbulent", ("tglf", "cgyro"),
          ("TGLF", r"\itshape gyrofluid", "CGYRO", r"\itshape gyrokinetic")),
    ),
}
OPERATION: Dict[str, Item] = {op.key: op for ops in OPERATIONS.values() for op in ops}
#: alternative operations drawn in one dash-dot enclosure
ALTERNATIVES: Tuple[Tuple[str, ...], ...] = (("analytic", "free_boundary", "reconstruction"),)
#: what each section writes to IMAS: (ids, representative quantities, technical elements). Only IDS with a writer
#: are named; a code is named where the section holds several and only one writes the IDS
IMAS: Dict[str, Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = {
    "equilibrium": (("equilibrium", r"$\psi$, $q$, $p$", ("eq_analytic", "eq_model", "eq_reconstructed", "eq_refined")),),
    "mhd": (("mhd_linear", r"$\delta W_n$ (normalized), $\delta\mathbf B\cdot\nabla\psi$", ("ml_dcon", "ml_gpec")),
            ("ntms", r"$\Delta'$", ("ntms_rdcon",))),
    "transport": (("core_transport", r"$\Gamma_s$, $Q_s$", ("ct_classical", "ct_neo", "ct_tglf")),
                  ("core_profiles", r"$j_{\mathrm{bs}}$, $\sigma_\parallel$", ("ct_neo",)),  # written by NEO
                  ("gyrokinetics_local", r"$\gamma$, $\omega$", ("gk_cgyro",))),  # TGLF has a writer too
}
#: what each section keeps in the solver's own form, with no IMAS mapping yet: (quantities, technical elements).
#: Listed under the IDS with an italic "native" key in place of an IDS name, so it never reads as mapped
NATIVE: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "mhd": ((r"$\delta W_p$, $\delta W_v$; $\Phi_{\mathrm{res}}$, $w_{\mathrm{isl}}$; $\Delta'$ matrix",
             ("dcon_native", "gpec_native", "rdcon_native")),),
}
#: native results left out of the figure on purpose: the Sauter / Redl return value (its having no writer is a
#: detail for the technical figure)
OMITTED: Tuple[str, ...] = ("sr_native",)
#: (source, target): data flows between drawn cards and sections
EDGES: Tuple[Tuple[str, str], ...] = (
    ("context", "equilibrium"),
    ("context", "mhd"),
    ("reconstruction", "refinement"),
    ("equilibrium", "mhd"),
    ("equilibrium", "transport"),
)
#: the legend: each card kind with its pictogram (the cards carry the same pictogram), then the two line styles
CONNECTION_WORDS = {"flow": "data flow (implemented interface)", "alternative": "alternative methods"}
KIND_WORDS = {"measured": "measured", "machine": "configuration", "prior": "assumption / parameter",
              "convention": "convention", "derived": "derived", "solver": "solver / model"}

# ---------------------------------------------------------------------------
# layout [cm]: x from 0 to _WIDTH, y downward from 0
# ---------------------------------------------------------------------------

_WIDTH = 23.5
_MAIN = 6.25  # where the computational sections start, right of the left panel and its arrows
_BOTTOM = -10.0
#: (x0, x1, y_top, y_bottom) of each section
_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "context": (0.0, 5.4, 0.0, _BOTTOM),
    "equilibrium": (_MAIN, _WIDTH, 0.0, -3.5),
    "mhd": (_MAIN, 14.65, -4.3, _BOTTOM),
    "transport": (14.9, _WIDTH, -4.3, _BOTTOM),
}
_X = {"equilibrium": (8.525, 12.525, 16.525, 21.225), "mhd": (7.7, 10.45, 13.2), "transport": (16.45, 19.2, 21.95)}
_W = {"equilibrium": 3.7, "mhd": 2.5, "transport": 2.5}
_ROW = {"equilibrium": -1.8, "mhd": -6.55, "transport": -6.55}
_CARD_H = {"equilibrium": 2.15, "mhd": 3.0, "transport": 3.0}  # a top strip holds the pictogram
#: the top of each section's quantity list
_LIST_TOP = {"equilibrium": -3.08, "mhd": -8.25, "transport": -8.25}
_CONTEXT_Y = {"measurements": -1.65, "profiles": -2.95, "machine": -4.8, "models": -6.45, "conventions": -8.7}
_CONTEXT_H = {"measurements": 1.0, "profiles": 1.3, "machine": 1.3, "models": 1.75, "conventions": 2.25}
_GROUP_Y = {"Data": -0.85, "Metadata": -3.9}
#: edges drawn along given points: the left panel feeds the first section of each tier; an equilibrium leaves
#: its section for the centre of each consumer
_ROUTES = {
    # from the panel, level with a gap between cards, so no single card reads as the source
    ("context", "equilibrium"): ((5.48, -2.225), (6.17, -2.225)),
    ("context", "mhd"): ((5.48, -7.45), (6.17, -7.45)),
    ("equilibrium", "mhd"): ((10.45, -3.58), (10.45, -4.22)),
    ("equilibrium", "transport"): ((19.2, -3.58), (19.2, -4.22)),
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
    text = "{\\small\\bfseries " + "\\\\".join(item.title.split("\n")) + "}"  # breaks where the data says
    for line in item.lines:
        text += "\\\\[1pt]{\\footnotesize " + (line if "$" in line or "\\" in line else escape_latex(line)) + "}"
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
        for op, x in zip(OPERATIONS[domain], _X[domain]):
            b = _card(op, x, _ROW[domain], _W[domain], _CARD_H[domain], f"operation:{op.key}", icon="top")
            boxes[op.key] = b
            items += list(b.items)
        items += _quantity_items(domain, labels)
    for keys in ALTERNATIVES:
        xs = [boxes[k].x for k in keys]
        half = 0.5 * _CARD_H["equilibrium"] + _ENCLOSE
        bx0, bx1 = min(xs) - 0.5 * _W["equilibrium"] - _ENCLOSE, max(xs) + 0.5 * _W["equilibrium"] + _ENCLOSE
        items.append(Polyline.of([(bx0, _ROW["equilibrium"] - half), (bx1, _ROW["equilibrium"] - half),
                                  (bx1, _ROW["equilibrium"] + half), (bx0, _ROW["equilibrium"] + half)],
                                 CONNECTION_STYLE["alternative"], role="alternative:" + ",".join(keys), closed=True))
    for src, dst in EDGES:
        role = f"edge:flow:{src}->{dst}"
        path = _ROUTES.get((src, dst))
        if path is None:
            a = connector(boxes[src], boxes[dst], role=role)
            path = (a.start, a.end)
        items.append(Arrow(path[0], path[1], CONNECTION_STYLE["flow"], role=role))
    if labels:  # the kinds with their pictograms, then the two line styles
        used = [k for k in KINDS if any(i.kind == k for i in CONTEXT + tuple(OPERATION.values()))]
        legend, y = kind_legend(used, 0.0, _WIDTH, _BOTTOM - 0.65, label_style=_LEGEND, labels=KIND_WORDS)
        items += legend
        items += connection_legend(0.0, _WIDTH, y - 0.05, ("flow", "alternative"), CONNECTION_WORDS, _LEGEND)
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"context": CONTEXT, "operations": OPERATIONS, "alternatives": ALTERNATIVES, "imas": IMAS, "native": NATIVE,
             "edges": EDGES, "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in boxes.items()}}
    return Diagram("cross_domain_scientific_architecture", Scene(tuple(items)), model=model)
