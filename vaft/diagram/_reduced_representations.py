"""Reduced physical representations: how plasma information is compressed (#1626).

``reduced_representation_hierarchy``
    fields, profiles and scalars down one axis, dimensional and
    dimensionless across the other -- dimensionless is not 0-D -- then
    similarity coordinates and closures; each cell counts the catalogued
    formulas whose ``Reduction`` section lands there;
``reduction_graph``
    one family (``current_q``, ``pressure_energy``, ``kinetic_profiles``,
    ``dimensionless_similarity``) as a layered graph of quantities. Edges are
    :data:`vaft.formula._taxonomy.REDUCTION_FAMILIES`; an edge a formula
    performs takes its kind from that formula's catalog entry, so the figure
    cannot disagree with the docstrings, and a step VAFT does elsewhere is
    dashed.

The physics stays in :mod:`vaft.formula`; this module only lays the
metadata out.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from vaft.formula._taxonomy import QUANTITIES, REDUCTION_FAMILIES, Relation

from ._concept import band, box, connector
from ._geometry import _check_labels
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

FAMILIES = tuple(REDUCTION_FAMILIES)

#: representation -> box style and the words shown under the symbol
_STYLE = {
    "field_2d": ("concept source", "field"),
    "profile_1d": ("concept box", "profile"),
    "scalar_0d": ("concept base", "scalar"),
}


def relation_kind(relation: Relation) -> str:
    """The relation's reduction kind: the formula's catalog entry when it names one."""
    if relation.formula is None:
        if relation.kind is None:
            raise ValueError(f"relation to {relation.target!r} names neither a formula nor a kind")
        return relation.kind
    from vaft.formula.catalog import describe

    spec = describe(relation.formula)
    if spec.reduction is None:
        raise ValueError(f"{relation.formula} has no Reduction section, so it cannot label a reduction edge")
    return spec.reduction.kind


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

#: height of one line of small-label text, and its gap above an arrow [cm]
_LINE, _LABEL_GAP = 0.45, 0.12

_ROWS = (("field_2d", "2-D field"), ("profile_1d", "1-D profile"), ("scalar_0d", "0-D scalar"))


def _counts() -> Dict[Tuple[str, bool], int]:
    from vaft.formula.catalog import list_formulas

    counts: Dict[Tuple[str, bool], int] = {}
    for spec in list_formulas():
        r = spec.reduction
        if r is not None:
            if "field_2d" in r.input:
                counts[("field_input", False)] = counts.get(("field_input", False), 0) + 1
            # the column is the catalogued unit, not the kind: a dimensionless shear is still a differential
            dimensionless = bool(spec.returns) and spec.returns[0].unit == "-"
            key = (r.output, dimensionless)
            counts[key] = counts.get(key, 0) + 1
            if r.role == "similarity_coordinate":
                counts[("similarity", True)] = counts.get(("similarity", True), 0) + 1
            if r.kind in ("closure", "empirical_scaling"):
                counts[("closure", False)] = counts.get(("closure", False), 0) + 1
    return counts


def reduced_representation_hierarchy(*, labels: bool = True) -> Diagram:
    r"""Spatial reduction and dimensionless normalisation as two independent axes.

    Rows are the output representation (2-D field, 1-D profile, 0-D scalar),
    columns dimensional and dimensionless: a local $\nu^*(\rho)$ is a
    dimensionless *profile*, a global $\beta$ a dimensionless *scalar*.
    Below them, similarity coordinates and closures / empirical scalings. Each
    cell gives examples and the number of catalogued formulas whose
    ``Reduction`` section outputs there, the column decided by the formula's
    documented return unit (``-`` is dimensionless).
    """
    labels = _check_labels(labels)
    counts = _counts()
    examples = {
        ("field_2d", False): "$B_p(R, Z)$, $n(R, Z)$, $T(R, Z)$",
        ("profile_1d", False): "$j_\\phi(\\rho)$, $p(\\rho)$, $B_\\theta(r)$",
        ("profile_1d", True): "$q(\\rho)$, $\\hat s(\\rho)$, $a/L_T$, $\\nu^*(\\rho)$",
        ("scalar_0d", False): "$I_p$, $W$, $\\langle p\\rangle$, $\\beta_N$",
        ("scalar_0d", True): "$l_i$, $\\beta_t$, $\\beta_p$, $f_G$, $\\rho_*$",
    }
    xs = {False: 0.0, True: 9.6}
    ys = {"field_2d": 0.0, "profile_1d": -2.6, "scalar_0d": -5.2}
    items: List = []
    for column, text in ((False, "dimensional"), (True, "dimensionless")):
        items.append(Label((xs[column], 1.25), text, "concept group title", anchor="south",
                           role=f"column:{'dimensionless' if column else 'dimensional'}"))
    for rep, text in _ROWS:
        items += band(-5.3, 13.2, ys[rep] - 0.95, ys[rep] + 0.95, role=f"row:{rep}")
        items.append(Label((-5.15, ys[rep]), text, "concept band label,text width=1.6cm,align=left", anchor="west",
                           role=f"row:{rep}"))
    cells = {}
    for (rep, dimless), example in examples.items():
        if rep == "field_2d":
            n = counts.get(("field_input", False), 0)
            tally = f"input to {n} catalogued formula{'s' if n != 1 else ''}"
        else:
            n = counts.get((rep, dimless), 0)
            tally = f"output of {n} catalogued formula{'s' if n != 1 else ''}"
        cells[(rep, dimless)] = box(xs[dimless], ys[rep], 5.6, 1.5, f"{example}\\\\ {{\\small {tally}}}",
                                    style=_STYLE[rep][0],
                                    role=f"cell:{rep}:{'dimensionless' if dimless else 'dimensional'}", latex=True)
        items += list(cells[(rep, dimless)].items)
    lower = {
        "similarity": box(9.6, -7.9, 5.6, 1.5, "similarity coordinates:\\\\ $\\rho_*$, $\\nu_*$, $\\Omega_i\\tau_E$\\\\ "
                          f"{{\\small output of {counts.get(('similarity', True), 0)} catalogued formulas}}",
                          style="concept strong", role="cell:similarity", latex=True),
        "closure": box(0.0, -7.9, 5.6, 1.5, "closures, empirical scalings:\\\\ $\\tau_E$, $q_{95}$ estimate\\\\ "
                       f"{{\\small output of {counts.get(('closure', False), 0)} catalogued formulas}}",
                       style="concept leaf", role="cell:closure", latex=True),
    }
    for b in lower.values():
        items += list(b.items)
    arrows = [
        (cells[("field_2d", False)], cells[("profile_1d", False)], "flux-surface average, projection"),
        (cells[("profile_1d", False)], cells[("scalar_0d", False)], "integral, moment, feature"),
        (cells[("profile_1d", False)], cells[("profile_1d", True)], "local\\\\ normalisation"),
        (cells[("scalar_0d", False)], cells[("scalar_0d", True)], "global\\\\ normalisation"),
        (cells[("profile_1d", True)], cells[("scalar_0d", True)], "feature, average"),
        (cells[("scalar_0d", True)], lower["similarity"], "similarity transform"),
        (lower["similarity"], lower["closure"], "scaling law"),
    ]
    for a, b, text in arrows:
        if abs(b.x - a.x) > abs(b.y - a.y):
            # a horizontal arrow and its label are centred together, as one block, on the boxes' mid-line
            label_height = _LINE * (text.count("\\\\") + 1) if labels else 0.0
            y = a.y - 0.5 * (label_height + _LABEL_GAP) if labels else a.y
            side = 1.0 if b.x > a.x else -1.0
            start = (a.x + side * (0.5 * a.width + 0.08), y)
            end = (b.x - side * (0.5 * b.width + 0.08), y)
            items.append(Arrow(start, end, "connector", role="reduction"))
            if labels:
                items.append(Label((0.5 * (start[0] + end[0]), y + _LABEL_GAP), text, "small label,align=center",
                                   anchor="south", role="reduction"))
        else:
            arrow = connector(a, b, role="reduction")
            items.append(arrow)
            if labels:
                my = 0.5 * (arrow.start[1] + arrow.end[1])
                items.append(Label((arrow.start[0] + 0.15, my), text, "small label,align=center", anchor="west",
                                   role="reduction"))
    if labels:
        items.append(Label((4.8, -9.1), "Down: spatial reduction. Across: dimensionless normalisation -- an "
                           "independent axis, so a dimensionless quantity can still be a profile",
                           "note", anchor="north", role="note"))
    model = {"counts": dict(counts), "cells": {k: (b.x, b.y) for k, b in cells.items()}}
    return Diagram("reduced_representation_hierarchy", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# family graphs
# ---------------------------------------------------------------------------

_DX, _DY, _W, _H = 7.0, 1.85, 5.4, 1.5


def _layers(relations) -> Dict[str, int]:
    depth: Dict[str, int] = {}
    for _ in range(len(relations) + 1):  # longest path; the graphs are small DAGs
        for rel in relations:
            for s in rel.sources:
                depth.setdefault(s, 0)
            depth[rel.target] = max(depth.get(rel.target, 0), 1 + max(depth[s] for s in rel.sources))
    return depth


def _heights(order: Dict[int, List[str]], incoming: Dict[str, Relation]) -> Dict[str, float]:
    """Vertical positions: each source's targets a contiguous block, each source level with its block.

    The first target layer is stacked about zero; deeper layers centre each
    group on its source; the inputs (layer 0) then sit at the mean height of
    their targets. Spacing never falls below ``_DY``.
    """
    ys: Dict[str, float] = {}
    layers = sorted(order)
    for layer in layers[1:]:
        keys = order[layer]
        if layer == layers[1]:
            top = 0.5 * (len(keys) - 1) * _DY
            ys.update({key: top - i * _DY for i, key in enumerate(keys)})
            continue
        groups: List[List[str]] = []
        for key in keys:
            parent = incoming[key].sources[0]
            if groups and incoming[groups[-1][0]].sources[0] == parent:
                groups[-1].append(key)
            else:
                groups.append([key])
        floor = None
        for group in groups:
            centre = ys[incoming[group[0]].sources[0]]
            top = centre + 0.5 * (len(group) - 1) * _DY
            if floor is not None:
                top = min(top, floor - _DY)
            for i, key in enumerate(group):
                ys[key] = top - i * _DY
            floor = ys[group[-1]]
    floor = None
    for key in order[layers[0]]:
        children = [t for t, rel in incoming.items() if key in rel.sources and t in ys]
        y = sum(ys[c] for c in children) / len(children) if children else 0.0
        if floor is not None:
            y = min(y, floor - _DY)
        ys[key] = floor = y
    return ys


def reduction_graph(family: str = "current_q", *, labels: bool = True) -> Diagram:
    r"""One reduction family as a layered graph of quantities.

    Nodes are quantities, laid out left to right by how many reductions
    separate them from the family's inputs and styled by representation
    (field, profile, scalar). A solid edge is a ``vaft.formula`` function and
    is labelled with the kind its ``Reduction`` section declares; a dashed
    edge is a step VAFT performs elsewhere (a flux-surface average, a
    profile feature) and states its own kind.
    """
    if family not in REDUCTION_FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, not {family!r}")
    labels = _check_labels(labels)
    relations = REDUCTION_FAMILIES[family]
    depth = _layers(relations)
    order: Dict[int, List[str]] = {}
    for rel in relations:
        for key in (*rel.sources, rel.target):
            column = order.setdefault(depth[key], [])
            if key not in column:
                column.append(key)
    # order each layer by where its sources sit (barycentre), which removes most crossings
    rank = {key: i for i, key in enumerate(order[0])}
    for layer in sorted(order)[1:]:
        def barycentre(key):
            sources = [s for rel in relations if rel.target == key for s in rel.sources]
            return sum(rank[s] for s in sources) / len(sources)
        appearance = {key: i for i, key in enumerate(order[layer])}
        order[layer] = sorted(order[layer], key=lambda k: (barycentre(k), appearance[k]))
        rank.update({key: i for i, key in enumerate(order[layer])})
    incoming = {rel.target: rel for rel in relations}
    ys = _heights(order, incoming)
    tallest = max(len(v) for v in order.values())
    nodes = {}
    items: List = []
    for layer, keys in sorted(order.items()):
        for key in keys:
            q = QUANTITIES[key]
            style, word = _STYLE[q.representation]
            sub = word + (", dimensionless" if q.dimensionless else "")
            text = f"{q.symbol}\\\\ {{\\small {sub}}}"
            if key in incoming and labels:
                text += f"\\\\ {{\\small\\itshape\\hyphenpenalty=10000 via {relation_kind(incoming[key]).replace('_', ' ')}}}"
            nodes[key] = box(layer * _DX, ys[key], _W, _H, text, style=style, role=f"node:{key}", latex=True)
            items += list(nodes[key].items)
    # Orthogonal edges: every edge leaves its source at the midpoint of the right side -- one shared start
    # per source -- runs to a vertical trunk in the middle of the gap, and enters the target at the midpoint
    # of its left side. _heights keeps each source's targets a contiguous block around it, so the trunks
    # of one gap never overlap and a single target is a straight line.
    trunks = {(depth[k], k): depth[k] * _DX + 0.5 * _DX for k in depth}
    # The stub from the source and its trunk are drawn once, plain, per source; each edge is the branch
    # from the trunk into its target, solid or dashed. A source with one level target is one straight arrow.
    targets: Dict[str, List[str]] = {}
    for rel in relations:
        for source in rel.sources:
            targets.setdefault(source, []).append(rel.target)
    straight = {s for s, t in targets.items() if len(t) == 1 and abs(nodes[t[0]].y - nodes[s].y) < 1e-9}
    for source, ts in targets.items():
        if source in straight:
            continue
        a, xm = nodes[source], trunks[(depth[source], source)]
        span = [nodes[t].y for t in ts] + [a.y]
        items.append(Polyline.of([(a.x + 0.5 * a.width, a.y), (xm, a.y)], "connector line", role=f"bus:{source}"))
        if max(span) - min(span) > 1e-9:
            items.append(Polyline.of([(xm, max(span)), (xm, min(span))], "connector line", role=f"bus:{source}"))
    edges = []
    for rel in relations:
        kind = relation_kind(rel)
        for source in rel.sources:
            style = "connector" if rel.formula else "connector feedback"
            a, b = nodes[source], nodes[rel.target]
            end = (b.x - 0.5 * b.width - 0.08, b.y)
            start = (a.x + 0.5 * a.width, a.y) if source in straight else (trunks[(depth[source], source)], b.y)
            items.append(Polyline.of([start, end], style, role=f"edge:{source}->{rel.target}"))
            edges.append((source, rel.target, kind, rel.formula))
    if labels:
        width = max(order) * _DX
        bottom = min(ys.values()) - 0.5 * _H - 0.5
        items.append(Label((0.5 * width, bottom), "Solid: a vaft.formula function; \\emph{via} names the reduction kind "
                           "its docstring declares. Dashed: a step VAFT performs elsewhere", "note", anchor="north",
                           role="note"))
    model = {"family": family, "layers": dict(depth), "edges": tuple(edges),
             "nodes": {k: (b.x, b.y) for k, b in nodes.items()}}
    return Diagram(f"reduction_graph_{family}", Scene(tuple(items)), model=model)
