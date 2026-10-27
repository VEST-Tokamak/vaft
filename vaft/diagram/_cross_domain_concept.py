"""The scientific architecture behind the equilibrium, MHD and transport figure, drawn for a manuscript.

The conceptual companion of :func:`~vaft.diagram.cross_domain_scientific_workflows`:
shared scientific state and context on the left, then scientific operations and
the reusable representations they produce. Where the technical figure draws
each solver with the inputs it reads, this one groups them -- inputs into four
context cards, solvers into the operations they perform, IDS results into one
representation per domain -- and keeps only the relations that carry the
argument: which methods are alternatives, how the equilibrium representation is
reused downstream, and what is mapped to IMAS and what stays native.

Nothing here is restated by hand. Every card, operation and representation
names the technical elements it summarizes (``covers``), and the tests hold the
two figures to each other: every technical input, method and result is covered
exactly once, alternatives here are alternatives there, and the IDS named here
are the ones the technical figure's writers produce. The figure claims
interfaces that exist, not workflows that were validated end to end; each
downstream model still checks its own inputs, which the edge labels and the
footnote say.

Simplified on purpose (the technical figure keeps each of these):

- solver-specific inputs (mode numbers, excitation, resistive layers, COCOS
  of one code) are folded into the context cards;
- the resolved local transport state and its ``state_identity`` are not
  drawn; the edge into transport names the reconstructed equilibrium;
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


@dataclass(frozen=True)
class Item:
    """One card of the figure and the technical elements it stands for.

    ``kind`` is a workflow kind; ``title`` the bold first line; ``lines`` the
    LaTeX lines under it; ``idss`` the IDS names printed in typewriter at its
    foot. ``covers`` names :data:`vaft.diagram._cross_domain.ELEMENT` keys.
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
#: shared scientific state and context: two kinds of data, then the knowledge an operation is defined by
CONTEXT: Tuple[Item, ...] = (
    I("measurements", "measured", "Measurements",
      ("tm_targets", "eq_coils", "ef_magnetics", "ef_kinetic", "tr_profiles"),
      (r"$I_p$, magnetics, coil currents", r"$T_e(\rho)$, $n_e(\rho)$"),
      ("magnetics", "pf_active", "thomson_scattering", "core_profiles")),
    I("machine", "machine", "Machine configuration", ("eq_machine", "gpec_coils"),
      (r"wall, PF coils, vessel, 3-D coils",), ("wall", "pf_active", "pf_passive", "coils_non_axisymmetric")),
    I("models", "prior", "Physical assumptions", ("an_shape", "tm_profiles", "tr_closures", "sr_closures", "gpec_drive"),
      (r"closures ($Z_{\mathrm{eff}}$, $T_i$, $\gamma_E$)", r"profile shapes ($p'$, $FF'$)", "3-D coil excitation")),
    I("conventions", "convention", "Conventions", ("ef_basis", "ch_cocos", "dcon_in"),
      (r"profile bases, COCOS,", r"mode numbers, wall model")),
)
#: the two groups of the context column: (title, subtitle, item keys)
CONTEXT_GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("Data", "", ("measurements", "machine")),
    ("Scientific knowledge", "explicit inputs, distinct from data", ("models", "conventions")),
)
#: scientific operations, by domain; lines are the method, then the physical model it rests on
OPERATIONS: Dict[str, Tuple[Item, ...]] = {
    "equilibrium": (
        I("analytic", "solver", "Analytic Grad-Shafranov", ("analytic",), ("Solov'ev family", r"\itshape assumed shape, profiles")),
        I("free_boundary", "solver", "Free-boundary model", ("tokamaker",), ("TokaMaker", r"\itshape assumed profiles")),
        I("reconstruction", "solver", "Reconstruction", ("efit",), ("EFIT", r"\itshape fit with a profile basis")),
        I("refinement", "solver", "Fixed-boundary refinement", ("chease",), ("CHEASE", r"\itshape fixed boundary")),
    ),
    "mhd": (
        I("ideal", "solver", "Ideal stability", ("dcon",), ("DCON", r"\itshape energy principle")),
        I("response", "solver", "Plasma response", ("gpec",), ("GPEC", r"\itshape ideal, 3-D coils")),
        I("tearing", "solver", "Resistive stability", ("rdcon",), ("RDCON", r"\itshape resistive layers")),
    ),
    "transport": (
        I("classical", "derived", "Classical", ("classical",), ("Braginskii", r"\itshape magnetized")),
        I("neoclassical", "solver", "Neoclassical", ("sauter_redl", "neo"),
          ("Sauter-Redl or NEO", r"\itshape closed form or kinetic")),
        I("turbulent", "solver", "Turbulent", ("tglf", "cgyro"),
          ("TGLF or CGYRO", r"\itshape quasilinear or linear GK")),
    ),
}
OPERATION: Dict[str, Item] = {op.key: op for ops in OPERATIONS.values() for op in ops}
#: alternative operations drawn in one dash-dot enclosure (inside a card, "or" joins alternatives)
ALTERNATIVES: Tuple[Tuple[str, ...], ...] = (("analytic", "free_boundary", "reconstruction"),)
#: the equilibrium representation: one entry under each equilibrium operation
EQUILIBRIUM_ENTRIES: Tuple[Item, ...] = (
    I("eq_analytic", "standardized", "analytic", ("eq_analytic",)),
    I("eq_model", "standardized", "free-boundary model", ("eq_model",)),
    I("eq_reconstructed", "standardized", "reconstructed", ("eq_reconstructed",)),
    I("eq_refined", "standardized", "refined", ("eq_refined",)),
)
#: the downstream representations, one card per domain and band: (ids or "", quantities) per line
RESULTS: Tuple[Tuple[Item, Tuple[Tuple[str, str], ...]], ...] = (
    (I("mhd_imas", "standardized", "", ("ml_dcon", "ml_gpec", "ntms_rdcon")),
     (("mhd_linear", r"$\delta W_n$ (normalized), $\delta\mathbf B\cdot\nabla\psi$"), ("ntms", r"$\Delta'$ (diagonal)"))),
    (I("transport_imas", "standardized", "", ("ct_neo", "ct_classical", "ct_tglf", "gk_cgyro")),
     (("core_transport", r"$\Gamma_s$, $Q_s$"), ("core_profiles", r"$j_{\mathrm{bs}}$, $\sigma_\parallel$ (NEO)"),
      ("gyrokinetics_local", r"$\gamma$, $\omega$ (linear; not yet stored)"))),
    (I("mhd_native", "native_result", "", ("dcon_native", "gpec_native", "rdcon_native")),
     (("", r"$\delta W_p$, $\delta W_v$;\ \ $\Phi_{\mathrm{res}}$, $w_{\mathrm{isl}}$;\ \ $\Delta'$ matrix"),)),
    (I("transport_native", "native_result", "", ("sr_native",)),
     (("", r"closed-form $j_{\mathrm{bs}}$, $\sigma_\parallel$ (Sauter-Redl)"),)),
)
RESULT: Dict[str, Item] = {item.key: item for item, _ in RESULTS}
#: (source, target, label): data flows between drawn cards and panels
EDGES: Tuple[Tuple[str, str, str], ...] = (
    ("reconstruction", "refinement", ""),
    ("ideal", "response", ""),
    ("equilibrium_ids", "mhd", "eligible equilibrium + model inputs"),
    ("eq_reconstructed", "transport", "reconstructed, for the resolved state"),
)
FOOTNOTE = ("Arrows mark implemented interfaces, not validated end-to-end workflows; "
            "each model checks its own required inputs.")

# ---------------------------------------------------------------------------
# layout [cm]: x from 0 to _WIDTH, y downward from 0
# ---------------------------------------------------------------------------

_WIDTH = 22.5
_MAIN = 5.75  # where the operations start, right of the context column and its arrow
#: (x0, x1, y_top, y_bottom) of each panel and band
_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "context": (0.0, 4.9, 0.0, -13.3),
    "equilibrium": (_MAIN, _WIDTH, 0.0, -3.2),
    "equilibrium_ids": (_MAIN, _WIDTH, -3.5, -5.1),
    "mhd": (_MAIN, 14.05, -5.8, -9.7),
    "transport": (14.3, _WIDTH, -5.8, -9.7),
    "imas": (_MAIN, _WIDTH, -10.05, -11.85),
    "native": (_MAIN, _WIDTH, -12.1, -13.3),
}
_BAND_TITLE = {
    "equilibrium_ids": ("Equilibrium representation", "one IDS type; each entry keeps its own plasma state"),
    "imas": ("IMAS-compatible representations", "per domain: the IDS its operations can write"),
    "native": ("Native results", "retained in the solver's own form"),
}
_SUBTITLE = {"equilibrium": "alternative routes and a refinement", "mhd": "complementary analyses",
             "transport": "independent models"}
_X = {"equilibrium": (8.075, 11.875, 15.675, 20.175), "mhd": (7.1, 10.0, 12.7), "transport": (15.65, 18.4, 21.15)}
_W = {"equilibrium": 3.5, "mhd": 2.5, "transport": 2.55}
_ROW = {"equilibrium": -2.05, "mhd": -8.3, "transport": -8.3}
_CARD_H = {"equilibrium": 1.85, "mhd": 2.25, "transport": 2.25}
_CONTEXT_Y = {"measurements": -2.9, "machine": -5.65, "models": -9.15, "conventions": -11.5}
_CONTEXT_H = {"measurements": 2.7, "machine": 2.35, "models": 1.9, "conventions": 1.6}
_GROUP_Y = {"Data": -1.18, "Scientific knowledge": -7.35}
_ENTRY_Y, _ENTRY_W = -4.55, 3.2
_RESULT_PLACE = {"mhd_imas": ("mhd", -11.07, 0.95), "transport_imas": ("transport", -11.13, 1.3),
                 "mhd_native": ("mhd", -12.87, 0.55), "transport_native": ("transport", -12.87, 0.55)}
_ROUTES = {
    ("equilibrium_ids", "mhd"): ((6.1, -5.18), (6.1, -5.72)),  # from the band, clear of any one entry
    ("eq_reconstructed", "transport"): ((_X["equilibrium"][2], _ENTRY_Y - 0.355), (_X["equilibrium"][2], -5.72)),
}
_LABEL_AT = {("equilibrium_ids", "mhd"): (6.25, -5.45),
             ("eq_reconstructed", "transport"): (_X["equilibrium"][2] + 0.15, -5.45)}
_ENCLOSE = 0.13


def _fit(width: float, text: str) -> str:
    """``text`` shrunk to the box's text width when wider; never enlarged."""
    return f"\\vaftfit{{{width - 0.32:.2f}cm}}{{{text}}}"


def _card_text(item: Item, width: float) -> str:
    """A bold title, then the lines (wrapped to the width), then the IDS names (wrapped at the separators)."""
    text = "{\\small\\bfseries " + escape_latex(item.title) + "}"  # wraps rather than shrinks: one size
    for line in item.lines:
        size = "\\footnotesize"
        text += "\\\\[1pt]{" + size + " " + (line if "$" in line or "\\" in line else escape_latex(line)) + "}"
    if item.idss:
        text += ("\\\\[2pt]{\\footnotesize\\ttfamily "
                 + "~$\\cdot$ ".join(escape_latex(i) for i in item.idss) + "}")  # breaks after a dot
    return text


def _result_text(lines: Tuple[Tuple[str, str], ...], width: float) -> str:
    rows = []
    for ids, quantities in lines:
        row = ("\\texttt{" + escape_latex(ids) + "}\\ \\ " if ids else "") + quantities
        rows.append(_fit(width, "{\\small " + row + "}"))
    return "\\\\[1pt]".join(rows)


def _panel(key: str, title: str, subtitle: str, style: str, labels: bool, two_lines: bool = False) -> List:
    """A panel or band outline with its title; ``two_lines`` sets the subtitle under the title."""
    x0, x1, y0, y1 = _BOUNDS[key]
    items: List = [Polyline.of([(x0, y1), (x1, y1), (x1, y0), (x0, y0)], style, role=f"region:{key}", closed=True)]
    if labels:
        sub = "{\\small\\itshape\\color{black!60}" + escape_latex(subtitle) + "}"
        head = "\\textbf{" + escape_latex(title) + "}"
        if two_lines:
            items.append(Label((x0 + 0.2, y0 - 0.3), head, "concept group title", anchor="west",
                               role=f"region title:{key}"))
            items.append(Label((x0 + 0.2, y0 - 0.68), sub, "concept plain", anchor="west",
                               role=f"region subtitle:{key}"))
        else:
            items.append(Label((x0 + 0.2, y0 - 0.3), head + "\\quad" + sub, "concept group title", anchor="west",
                               role=f"region title:{key}"))
    return items


def cross_domain_scientific_architecture(*, labels: bool = True) -> Diagram:
    """The scientific architecture of equilibrium, MHD and transport in VAFT, for a manuscript.

    Shared scientific state and context -- measurements and machine
    configuration as data, physical models, assumptions and conventions as
    explicit knowledge -- feed scientific operations in three domains, each
    named by its method and the physical model it rests on. Alternative
    equilibrium routes share a dash-dot enclosure; their results become one
    equilibrium representation whose entries keep their own states, and that
    representation is reused by MHD and transport operations whose own input
    requirements still apply. Results are IMAS-compatible representations or
    native results. :func:`~vaft.diagram.cross_domain_scientific_workflows`
    draws the same architecture solver by solver.
    """
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    items: List = []
    boxes: Dict[str, Box] = {}
    # context column, and one arrow from it into every operation
    x0, x1, y0, y1 = _BOUNDS["context"]
    items.append(Polyline.of([(x0, y1), (x1, y1), (x1, y0), (x0, y0)],
                             "draw=islanddark!70,line width=1.0pt,rounded corners=4pt,fill=black!3",
                             role="region:context", closed=True))
    if labels:
        items.append(Label((x0 + 0.2, y0 - 0.42), "\\textbf{Shared scientific state}\\\\\\textbf{and context}",
                           "concept group title,align=left", anchor="west", role="region title:context"))
        for title, subtitle, _ in CONTEXT_GROUPS:
            text = "{\\small\\bfseries\\color{black!70}" + escape_latex(title) + "}"
            if subtitle:  # on its own line: the column is narrow
                text += "\\\\{\\footnotesize\\itshape\\color{black!60}" + escape_latex(subtitle) + "}"
            items.append(Label((x0 + 0.2, _GROUP_Y[title]), text, "concept plain,align=left", anchor="west",
                               role=f"context group:{title}"))
    for item in CONTEXT:
        b = box(0.5 * (x0 + x1), _CONTEXT_Y[item.key], x1 - x0 - 0.4, _CONTEXT_H[item.key],
                _card_text(item, x1 - x0 - 0.4), style=kind_style(item.kind), text_style="concept plain",
                role=f"context:{item.key}", latex=True)
        boxes[item.key] = b
        items += list(b.items)
    # it ends level with the operations: context feeds operations, not their results
    chevron = [(5.0, -1.0), (5.3, -1.0), (5.62, -5.35), (5.3, -9.7), (5.0, -9.7), (5.32, -5.35)]
    items.append(Polyline.of(chevron, "draw=islanddark!60,fill=islanddark!22,line width=0.6pt", role="edge:context",
                             closed=True))
    # operations and the equilibrium representation
    for domain in ("equilibrium", "mhd", "transport"):
        items += _panel(domain, _TITLE[domain], _SUBTITLE[domain], "concept group", labels, two_lines=True)
        for op, x in zip(OPERATIONS[domain], _X[domain]):
            b = box(x, _ROW[domain], _W[domain], _CARD_H[domain], _card_text(op, _W[domain]), style=kind_style(op.kind),
                    text_style="concept plain", role=f"operation:{op.key}", latex=True)
            boxes[op.key] = b
            items += list(b.items)
    for keys in ALTERNATIVES:
        xs = [boxes[k].x for k in keys]
        bx0, bx1 = min(xs) - 0.5 * _W["equilibrium"] - _ENCLOSE, max(xs) + 0.5 * _W["equilibrium"] + _ENCLOSE
        half = 0.5 * _CARD_H["equilibrium"] + _ENCLOSE
        by0, by1 = _ROW["equilibrium"] + half, _ROW["equilibrium"] - half
        items.append(Polyline.of([(bx0, by1), (bx1, by1), (bx1, by0), (bx0, by0)], CONNECTION_STYLE["alternative"],
                                 role="alternative:" + ",".join(keys), closed=True))
    for key, connection in (("equilibrium_ids", "mapping"), ("imas", "mapping"), ("native", "native")):
        title, subtitle = _BAND_TITLE[key]
        items += _panel(key, title, subtitle, CONNECTION_STYLE[connection], labels)
        bx0, bx1, by0, by1 = _BOUNDS[key]
        boxes[key] = Box(0.5 * (bx0 + bx1), 0.5 * (by0 + by1), bx1 - bx0, by0 - by1, ())
    for key in ("mhd", "transport"):
        bx0, bx1, by0, by1 = _BOUNDS[key]
        boxes[key] = Box(0.5 * (bx0 + bx1), 0.5 * (by0 + by1), bx1 - bx0, by0 - by1, ())
    for entry, x in zip(EQUILIBRIUM_ENTRIES, _X["equilibrium"]):
        b = box(x, _ENTRY_Y, _ENTRY_W, 0.55, _fit(_ENTRY_W, "{\\small " + escape_latex(entry.title) + "}"),
                style=kind_style(entry.kind), text_style="concept plain", role=f"representation:{entry.key}", latex=True)
        boxes[entry.key] = b
        items += list(b.items)
    for item, lines in RESULTS:
        domain, y, h = _RESULT_PLACE[item.key]
        bx0, bx1, *_ = _BOUNDS[domain]
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
            items.append(Label(_LABEL_AT[(src, dst)], escape_latex(text), "concept annotation", anchor="west",
                               role=f"edge label:{src}->{dst}"))
    if labels:
        used = [k for k in KINDS if any(i.kind == k for i in CONTEXT + tuple(OPERATION.values())
                                        + EQUILIBRIUM_ENTRIES + tuple(RESULT.values()))]
        legend, y = kind_legend(used, 0.0, _WIDTH, _BOUNDS["native"][3] - 0.7)
        items += legend
        items += connection_legend(0.0, _WIDTH, y - 0.1, ("flow", "alternative"))
        items.append(Label((0.5 * _WIDTH, y - 0.75), escape_latex(FOOTNOTE), "concept annotation",
                           role="footnote"))
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"context": CONTEXT, "operations": OPERATIONS, "alternatives": ALTERNATIVES,
             "entries": EQUILIBRIUM_ENTRIES, "results": RESULTS, "edges": EDGES,
             "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in boxes.items()}}
    return Diagram("cross_domain_scientific_architecture", Scene(tuple(items)), model=model)
