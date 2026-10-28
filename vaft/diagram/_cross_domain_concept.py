"""The scientific architecture behind the equilibrium, MHD and transport figure, drawn for a manuscript.

The conceptual companion of :func:`~vaft.diagram.cross_domain_scientific_workflows`,
read along one hierarchy: shared data and metadata, then equilibrium, then MHD
stability / plasma response and transport, then IMAS-compatible representations
with complementary native results. Where the technical figure draws each solver
with the inputs it reads, this one groups them -- inputs into data and metadata
cards, solvers into the operations they perform, results into one output section
per domain -- and keeps only the relations that carry the argument: which methods
are alternatives, how an equilibrium is reused downstream under model-specific
eligibility, and what is mapped to IMAS and what stays native.

Data and metadata are kept apart: data are what was measured (signals and
experimental profiles); metadata are the explicit context needed to interpret
them and to run an operation -- machine and diagnostic configuration, model
assumptions and parameters, conventions and provenance.

Nothing here is restated by hand. Every card, operation and result names the
technical elements it summarizes (``covers``), and the tests hold the two
figures to each other: every technical input, method and result is covered
exactly once, alternatives here are alternatives there, and the IDS named here
are the ones the technical figure's writers produce. The arrows claim
interfaces that exist, not workflows that were validated end to end.

Simplified on purpose (the technical figure keeps each of these):

- solver-specific inputs are folded into the data and metadata cards;
- the resolved local transport state, its ``state_identity`` and the
  reconstruction it requires are left to the technical figure;
- results are listed per domain, not per solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ._concept import Box, box, connector, escape_latex
from ._cross_domain import CONNECTION_STYLE, ELEMENT, REGIONS, connection_legend
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene
from ._workflow import KINDS, kind_legend, kind_style

_TITLE = {key: title for key, title, _ in REGIONS}
OUTPUT_TITLE = "IMAS-compatible representations with complementary native results"


@dataclass(frozen=True)
class Item:
    """One card of the figure and the technical elements it stands for.

    ``kind`` is a workflow kind; ``title`` the bold first line; ``lines`` the
    LaTeX lines under it (one starting ``\\itshape`` names a physical model);
    ``idss`` the IDS names printed in typewriter at its foot. ``covers`` names
    :data:`vaft.diagram._cross_domain.ELEMENT` keys.
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
    I("measurements", "measured", "Diagnostic signals", ("tm_targets", "eq_coils", "ef_magnetics"),
      (r"$I_p$, magnetics, coil currents",), ("magnetics", "pf_active")),
    I("profiles", "measured", "Experimental profiles", ("ef_kinetic", "tr_profiles"),
      (r"$T_e(\rho)$, $n_e(\rho)$",), ("thomson_scattering", "core_profiles")),
    I("machine", "machine", "Machine and diagnostic\nconfiguration", ("eq_machine", "gpec_coils"),
      (r"wall, vessel, coils, sensor positions",),
      ("wall", "pf_active", "pf_passive", "coils_non_axisymmetric")),
    I("models", "prior", "Model assumptions\nand parameters",
      ("an_shape", "tm_profiles", "tr_closures", "sr_closures", "gpec_drive"),
      (r"closures ($Z_{\mathrm{eff}}$, $T_i$, $\gamma_E$)", r"profile shapes ($p'$, $FF'$)", "3-D coil excitation")),
    I("conventions", "convention", "Conventions and provenance", ("ef_basis", "ch_cocos", "dcon_in"),
      (r"profile bases, COCOS,", r"mode numbers, wall model;", "recorded provenance")),
)
#: the two groups of the left panel: (title, subtitle, item keys)
CONTEXT_GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("Data", "", ("measurements", "profiles")),
    ("Metadata", "context for data and operations", ("machine", "models", "conventions")),
)
#: scientific operations, by domain; lines are the method, then (italic) the physical model it rests on.
#: A card holding alternative methods lists each code with its model in italics under it, in turn
OPERATIONS: Dict[str, Tuple[Item, ...]] = {
    "equilibrium": (
        I("analytic", "solver", "Analytic\nGrad-Shafranov", ("analytic",),
          ("Solov'ev family", r"\itshape assumed shape, profiles")),
        I("free_boundary", "solver", "Free-boundary model", ("tokamaker",), ("TokaMaker", r"\itshape assumed profiles")),
        I("reconstruction", "solver", "Reconstruction", ("efit",), ("EFIT", r"\itshape fit with a profile basis")),
        I("refinement", "solver", "Fixed-boundary\nrefinement", ("chease",), ("CHEASE", r"\itshape fixed boundary")),
    ),
    "mhd": (
        I("ideal", "solver", "Ideal\nstability", ("dcon",), ("DCON", r"\itshape energy principle")),
        # GPEC reads DCON's ideal eigenfunctions: said on the card, the gap is too narrow for an arrow
        I("response", "solver", "Plasma\nresponse", ("gpec",), ("GPEC", r"\itshape from DCON modes")),
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
#: alternative operations drawn in one dash-dot enclosure (inside a card, alternatives share it line by line)
ALTERNATIVES: Tuple[Tuple[str, ...], ...] = (("analytic", "free_boundary", "reconstruction"),)
#: the equilibrium representation: one entry under each equilibrium operation, inside the equilibrium section
EQUILIBRIUM_ENTRIES: Tuple[Item, ...] = (
    I("eq_analytic", "standardized", "analytic", ("eq_analytic",)),
    I("eq_model", "standardized", "free-boundary model", ("eq_model",)),
    I("eq_reconstructed", "standardized", "reconstructed", ("eq_reconstructed",)),
    I("eq_refined", "standardized", "refined", ("eq_refined",)),
)
#: the output section, one card per domain and row: (ids or "", quantities) per line
RESULTS: Tuple[Tuple[Item, Tuple[Tuple[str, str], ...]], ...] = (
    (I("mhd_imas", "standardized", "", ("ml_dcon", "ml_gpec", "ntms_rdcon")),
     (("mhd_linear", r"$\delta W_n$ (normalized), $\delta\mathbf B\cdot\nabla\psi$"), ("ntms", r"$\Delta'$ (diagonal)"))),
    (I("transport_imas", "standardized", "", ("ct_neo", "ct_classical", "ct_tglf", "gk_cgyro")),
     (("core_transport", r"$\Gamma_s$, $Q_s$"), ("core_profiles", r"$j_{\mathrm{bs}}$, $\sigma_\parallel$ (NEO)"),
      ("gyrokinetics_local", r"$\gamma$, $\omega$ (linear; not yet stored)"))),
    (I("mhd_native", "native_result", "", ("dcon_native", "gpec_native", "rdcon_native")),
     (("", r"$\delta W_p$, $\delta W_v$;\ \ $\Phi_{\mathrm{res}}$, $w_{\mathrm{isl}}$;\ \ $\Delta'$ matrix"),)),
    (I("transport_native", "native_result", "", ("sr_native",)),
     (("", r"closed-form $j_{\mathrm{bs}}$, $\sigma_\parallel$ (Sauter / Redl)"),)),
)
RESULT: Dict[str, Item] = {item.key: item for item, _ in RESULTS}
#: the two rows of the output section: (key, heading, kind)
OUTPUT_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("imas", "Implemented IMAS mappings", "standardized"),
    ("native", "Native outputs, no IMAS mapping implemented", "native_result"),
)
#: (source, target, label): data flows between drawn cards and sections; a label line break is "\n"
EDGES: Tuple[Tuple[str, str, str], ...] = (
    ("context", "equilibrium", ""),
    ("context", "mhd", ""),
    ("reconstruction", "refinement", ""),
    ("equilibrium", "mhd", "eligible equilibrium\n+ model inputs"),
    ("equilibrium", "transport", "eligible equilibrium\n+ kinetic profiles"),
)
#: the legend's words for the two relations drawn as lines
CONNECTION_WORDS = {"flow": "implemented interface, not a validated workflow", "alternative": "alternative methods"}
#: the legend's words for the kinds: short, since every card also says what it is
KIND_WORDS = {"measured": "measured", "derived": "derived", "prior": "assumption / parameter",
              "convention": "convention", "machine": "configuration", "solver": "solver / model"}

# ---------------------------------------------------------------------------
# layout [cm]: x from 0 to _WIDTH, y downward from 0
# ---------------------------------------------------------------------------

_WIDTH = 23.5
_MAIN = 6.25  # where the computational sections start, right of the left panel and its arrows
#: (x0, x1, y_top, y_bottom) of each section
_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "context": (0.0, 5.4, 0.0, -12.95),
    "equilibrium": (_MAIN, _WIDTH, 0.0, -4.15),
    "mhd": (_MAIN, 14.65, -5.05, -8.65),
    "transport": (14.9, _WIDTH, -5.05, -8.65),
    "outputs": (_MAIN, _WIDTH, -9.05, -12.95),
}
_SUBTITLE = {
    "equilibrium": "alternative routes and a refinement; their entries share one IDS format, not one plasma state",
    "mhd": "complementary analyses",
    "transport": "independent models; a card may hold alternatives",
}
_X = {"equilibrium": (8.525, 12.525, 16.525, 21.225), "mhd": (7.7, 10.45, 13.2), "transport": (16.45, 19.2, 21.95)}
_W = {"equilibrium": 3.7, "mhd": 2.5, "transport": 2.5}  # one width below the equilibrium, 0.25 cm apart
_ROW = {"equilibrium": -2.0, "mhd": -7.3, "transport": -7.3}
_CARD_H = {"equilibrium": 1.85, "mhd": 2.3, "transport": 2.3}
_ENTRY_Y, _ENTRY_W, _ENTRY_H = -3.55, 3.3, 0.5
_CONTEXT_Y = {"measurements": -1.95, "profiles": -3.75, "machine": -6.75, "models": -9.25, "conventions": -11.6}
_CONTEXT_H = {"measurements": 1.4, "profiles": 1.85, "machine": 2.45, "models": 2.2, "conventions": 1.95}
_GROUP_Y = {"Data": -0.92, "Metadata": -5.15}
#: output rows: heading y, card y, card height
_OUTPUT_ROW = {"imas": (-9.9, -10.85, 1.45), "native": (-11.85, -12.4, 0.6)}
#: edges drawn along given points: the left panel feeds the first section of each tier, and an equilibrium
#: leaves its section from below the entries, clear of any one entry
_ROUTES = {
    ("context", "equilibrium"): ((5.48, _ROW["equilibrium"]), (6.17, _ROW["equilibrium"])),
    ("context", "mhd"): ((5.48, _ROW["mhd"]), (6.17, _ROW["mhd"])),
    ("equilibrium", "mhd"): ((10.5, -4.23), (10.5, -4.97)),  # each from a gap between entries, not from one
    ("equilibrium", "transport"): ((18.9, -4.23), (18.9, -4.97)),
}
_LABEL_AT = {("equilibrium", "mhd"): (10.65, -4.6), ("equilibrium", "transport"): (19.05, -4.6)}
_ENCLOSE = 0.13
#: gray annotations, darker than the template's default
_NOTE = "concept annotation,text=black!80"
_SUB = "{\\small\\itshape\\color{black!75}"


def _fit(width: float, text: str) -> str:
    """``text`` shrunk to the box's text width when wider; never enlarged."""
    return f"\\vaftfit{{{width - 0.32:.2f}cm}}{{{text}}}"


def _card_text(item: Item) -> str:
    """A bold title, then the lines (wrapped to the width), then the IDS names (wrapped at the separators)."""
    # one size: a long title breaks where the data says ("\n"), never shrinks
    text = "{\\small\\bfseries " + "\\\\".join(escape_latex(t) for t in item.title.split("\n")) + "}"
    for line in item.lines:
        text += "\\\\[1pt]{\\footnotesize " + (line if "$" in line or "\\" in line else escape_latex(line)) + "}"
    if item.idss:
        text += ("\\\\[2pt]{\\footnotesize\\ttfamily "
                 + "~$\\cdot$ ".join(escape_latex(i) for i in item.idss) + "}")  # breaks after a dot
    return text


def _result_text(lines: Tuple[Tuple[str, str], ...], width: float) -> str:
    rows = []
    for ids, quantities in lines:
        row = ("\\texttt{" + escape_latex(ids) + "}\\ \\ " if ids else "") + quantities
        rows.append(_fit(width, "{\\small " + row + "}"))
    return "\\\\[2pt]".join(rows)


def _section(key: str, title: str, subtitle: str, labels: bool) -> List:
    """A section outline, its bold title and, on the line under it, its subtitle."""
    x0, x1, y0, y1 = _BOUNDS[key]
    items: List = [Polyline.of([(x0, y1), (x1, y1), (x1, y0), (x0, y0)], "concept group", role=f"region:{key}",
                               closed=True)]
    if labels:
        items.append(Label((x0 + 0.2, y0 - 0.3), "\\textbf{" + escape_latex(title) + "}", "concept group title",
                           anchor="west", role=f"region title:{key}"))
        if subtitle:
            items.append(Label((x0 + 0.2, y0 - 0.68), _SUB + escape_latex(subtitle) + "}", "concept plain",
                               anchor="west", role=f"region subtitle:{key}"))
    return items


def _bounds_box(key: str) -> Box:
    x0, x1, y0, y1 = _BOUNDS[key]
    return Box(0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y0 - y1, ())


def cross_domain_scientific_architecture(*, labels: bool = True) -> Diagram:
    """The scientific architecture of equilibrium, MHD and transport in VAFT, for a manuscript.

    Shared data (measurements and experimental profiles) and metadata
    (machine and diagnostic configuration, model assumptions and parameters,
    conventions and provenance) feed scientific operations, each named by its
    method and the physical model it rests on. Alternative equilibrium routes
    share a dash-dot enclosure, and each writes its own entry in one
    equilibrium representation -- a shared format, not equivalent states.
    MHD stability and plasma response and transport reuse an eligible
    equilibrium under their own input requirements. Their results are
    IMAS-compatible representations, where a mapping is implemented, with
    complementary native results.
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
                             "draw=islanddark!70,line width=1.0pt,rounded corners=4pt,fill=black!3",
                             role="region:context", closed=True))
    boxes["context"] = _bounds_box("context")
    if labels:
        items.append(Label((x0 + 0.2, y0 - 0.38), "\\textbf{Shared data \\& metadata}", "concept group title",
                           anchor="west", role="region title:context"))
        for title, subtitle, _ in CONTEXT_GROUPS:
            text = "{\\small\\bfseries\\color{black!75}" + escape_latex(title) + "}"
            if subtitle:  # under the heading, one size down: the column is narrow
                text += "\\\\{\\footnotesize\\itshape\\color{black!75}" + escape_latex(subtitle) + "}"
            items.append(Label((x0 + 0.2, _GROUP_Y[title]), text,
                               f"concept plain,align=left,text width={x1 - x0 - 0.4:.2f}cm", anchor="west",
                               role=f"context group:{title}"))
    for item in CONTEXT:
        b = box(0.5 * (x0 + x1), _CONTEXT_Y[item.key], x1 - x0 - 0.4, _CONTEXT_H[item.key], _card_text(item),
                style=kind_style(item.kind), text_style="concept plain", role=f"context:{item.key}", latex=True)
        boxes[item.key] = b
        items += list(b.items)
    # computational sections and their operations
    for domain in ("equilibrium", "mhd", "transport"):
        items += _section(domain, _TITLE[domain], _SUBTITLE[domain], labels)
        boxes[domain] = _bounds_box(domain)
        for op, x in zip(OPERATIONS[domain], _X[domain]):
            b = box(x, _ROW[domain], _W[domain], _CARD_H[domain], _card_text(op), style=kind_style(op.kind),
                    text_style="concept plain", role=f"operation:{op.key}", latex=True)
            boxes[op.key] = b
            items += list(b.items)
    for keys in ALTERNATIVES:
        xs = [boxes[k].x for k in keys]
        half = 0.5 * _CARD_H["equilibrium"] + _ENCLOSE
        bx0, bx1 = min(xs) - 0.5 * _W["equilibrium"] - _ENCLOSE, max(xs) + 0.5 * _W["equilibrium"] + _ENCLOSE
        items.append(Polyline.of([(bx0, _ROW["equilibrium"] - half), (bx1, _ROW["equilibrium"] - half),
                                  (bx1, _ROW["equilibrium"] + half), (bx0, _ROW["equilibrium"] + half)],
                                 CONNECTION_STYLE["alternative"], role="alternative:" + ",".join(keys), closed=True))
    for entry, x in zip(EQUILIBRIUM_ENTRIES, _X["equilibrium"]):  # each entry under the operation that writes it
        b = box(x, _ENTRY_Y, _ENTRY_W, _ENTRY_H, _fit(_ENTRY_W, "{\\small " + escape_latex(entry.title) + "}"),
                style=kind_style(entry.kind), text_style="concept plain", role=f"representation:{entry.key}", latex=True)
        boxes[entry.key] = b
        items += list(b.items)
    # one output section: implemented mappings, then native outputs
    items += _section("outputs", OUTPUT_TITLE, "", labels)
    boxes["outputs"] = _bounds_box("outputs")
    if labels:
        for row, heading, _ in OUTPUT_ROWS:
            items.append(Label((_MAIN + 0.2, _OUTPUT_ROW[row][0]), "{\\small\\bfseries\\color{black!75}"
                               + escape_latex(heading) + "}", "concept plain", anchor="west", role=f"output row:{row}"))
    for item, lines in RESULTS:
        row = "imas" if item.kind == "standardized" else "native"
        domain = "mhd" if item.key.startswith("mhd") else "transport"
        bx0, bx1, *_ = _BOUNDS[domain]
        _, y, h = _OUTPUT_ROW[row]
        w = bx1 - bx0 - 0.4
        b = box(0.5 * (bx0 + bx1), y, w, h, _result_text(lines, w), style=kind_style(item.kind),
                text_style="concept plain", role=f"representation:{item.key}", latex=True)
        boxes[item.key] = b
        items += list(b.items)
    for src, dst, text in EDGES:
        role = f"edge:flow:{src}->{dst}"
        path = _ROUTES.get((src, dst))
        if path is None:
            a = connector(boxes[src], boxes[dst], role=role)
            path = (a.start, a.end)
        items.append(Arrow(path[0], path[1], CONNECTION_STYLE["flow"], role=role))
        if text and labels:
            items.append(Label(_LABEL_AT[(src, dst)], "\\\\".join(escape_latex(t) for t in text.split("\n")),
                               _NOTE + ",align=left", anchor="west", role=f"edge label:{src}->{dst}"))
    if labels:  # the output rows name the standardized and native kinds, so the legend need not
        used = [k for k in KINDS if k not in ("standardized", "native_result")
                and any(i.kind == k for i in CONTEXT + tuple(OPERATION.values()))]
        legend, y = kind_legend(used, 0.0, _WIDTH, y1 - 0.65, label_style=_NOTE, labels=KIND_WORDS)
        items += legend
        items += connection_legend(0.0, _WIDTH, y - 0.05, ("flow", "alternative"), CONNECTION_WORDS, _NOTE)
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"context": CONTEXT, "operations": OPERATIONS, "alternatives": ALTERNATIVES,
             "entries": EQUILIBRIUM_ENTRIES, "results": RESULTS, "edges": EDGES,
             "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in boxes.items()}}
    return Diagram("cross_domain_scientific_architecture", Scene(tuple(items)), model=model)
