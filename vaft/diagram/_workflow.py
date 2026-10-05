"""Physics workflows as data: what a step takes in, what it produces, and on what grounds (#1585).

The level-2 view below the platform overview (``vest_data_platform``): one
:class:`WorkflowSpec` per derivation, inference or solver coupling, drawn by
one renderer (:func:`_render_workflow`) and tabulated by one function
(:func:`workflow_table`), so the figure, the documentation table and the tests
read the same record.

Every node has a :data:`KINDS` value saying how the quantity was obtained --
measured, reconstructed, derived, inferred, assumed (a policy or closure the
user or machine supplies), code input, solver, native result, standardized --
and the kind sets its style. Assumptions are drawn entering from the left, so
they never read as observations. Equations come only from ``vaft.formula``
through :func:`vaft.diagram._equations.formula_equation`; a node names the
function, never restates the physics. Each node may name the public API that
implements it (``api``) and the IMAS IDS it reads or writes (``ids``); the
tests require every ``api`` and ``equation`` to resolve on this tree.

Implementation maturity is explicit (:data:`STATUSES`), per workflow and,
where it differs, per node -- never inferred from the drawing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ._concept import Box, box as _concept_box, connector, escape_latex
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Label, Polyline, Scene

#: how a quantity was obtained, in reading order; each has one style
KINDS: Tuple[str, ...] = ("measured", "reconstructed", "derived", "inferred", "prior", "convention", "machine",
                          "code_input", "solver", "native_result", "standardized")
KIND_LABEL: Dict[str, str] = {
    "measured": "measured", "reconstructed": "reconstructed", "derived": "derived", "inferred": "inferred",
    "prior": "model assumption / prior", "convention": "model choice / convention",
    "machine": "machine geometry / static data", "code_input": "code input", "solver": "solver / model",
    "native_result": "native result", "standardized": "standardized IMAS",
}
#: kinds that enter a step from the side: configuration of the step, not its physical input chain
SIDE_KINDS = ("prior", "convention", "machine")
#: implementation maturity, stated, never inferred from the drawing
STATUSES: Tuple[str, ...] = ("implemented", "partial", "experimental", "planned", "design_shell")


def kind_style(kind: str) -> str:
    return "wf " + kind.replace("_", " ")


@dataclass(frozen=True)
class Node:
    """One quantity or step: its label, how it was obtained, and where VAFT implements or stores it.

    ``symbols`` is inline LaTeX math naming the representative variables an
    input or output carries (``"T_e(\\rho),\\ n_e(\\rho)"``); it is drawn under the
    label. ``ids`` is the IMAS IDS or path the quantity is stored at; where
    VAFT has no IMAS mapping yet, ``mapping_todo`` says so and is drawn as a
    TODO tag instead of silently omitting the path. ``relation`` is LaTeX for
    a relation the step implements in process or code rather than in
    ``vaft.formula``; it requires ``api``, the function that implements it,
    so the source is named. ``status`` is
    implementation metadata for the documentation table; physics figures never
    draw it.
    """

    key: str
    label: str
    kind: str
    api: Optional[str] = None
    ids: Optional[str] = None
    equation: Optional[str] = None
    status: Optional[str] = None
    symbols: Optional[str] = None
    mapping_todo: Optional[str] = None
    relation: Optional[str] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"node {self.key!r}: kind must be one of {KINDS}, not {self.kind!r}")
        if self.status is not None and self.status not in STATUSES:
            raise ValueError(f"node {self.key!r}: status must be one of {STATUSES}, not {self.status!r}")
        if self.relation is not None and self.api is None:
            raise ValueError(f"node {self.key!r}: a relation needs the api that implements it")
        for name in ("api", "equation"):
            value = getattr(self, name)
            if value is not None and not value.startswith("vaft."):
                raise ValueError(f"node {self.key!r}: {name} must be a dotted vaft path, not {value!r}")


@dataclass(frozen=True)
class WorkflowSpec:
    """A workflow: nodes laid out in ``rows`` (top to bottom), ``edges`` between them, and side inputs.

    A ``None`` in a row is an empty slot, keeping a column aligned across rows.

    ``side`` pairs an input node with the step it enters from the left
    (assumptions, policies, closures); side nodes are not in ``rows``.
    """

    key: str
    title: str
    family: str
    status: str
    nodes: Tuple[Node, ...]
    rows: Tuple[Tuple[Optional[str], ...], ...]
    edges: Tuple[Tuple[str, str, str], ...]
    side: Tuple[Tuple[str, str], ...] = ()
    summary: str = ""
    references: Tuple[str, ...] = field(default_factory=tuple)
    #: implementation or IMAS-mapping gaps the figure exposes: listed under its table, never drawn as capability
    todos: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        keys = [n.key for n in self.nodes]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.key}: duplicate node keys")
        if self.status not in STATUSES:
            raise ValueError(f"{self.key}: status must be one of {STATUSES}, not {self.status!r}")
        placed = [k for row in self.rows for k in row if k is not None]
        sides = [s for s, _ in self.side]
        if len(set(placed)) != len(placed) or set(placed) & set(sides):
            raise ValueError(f"{self.key}: a node is placed twice")
        if set(placed) | set(sides) != set(keys):
            raise ValueError(f"{self.key}: nodes {sorted(set(keys) - set(placed) - set(sides))} are not placed")
        for src, dst, _ in self.edges:
            if src not in keys or dst not in keys:
                raise ValueError(f"{self.key}: edge {src!r} -> {dst!r} names an unknown node")
        for src, dst in self.side:
            if dst not in placed:
                raise ValueError(f"{self.key}: side input {src!r} enters {dst!r}, which is not in a row")

    def node(self, key: str) -> Node:
        return next(n for n in self.nodes if n.key == key)


def resolve(dotted: str):
    """The object a dotted ``vaft`` path names (module attribute, possibly a sub-attribute)."""
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot resolve {dotted!r}")


def node_equation(node: Node) -> Optional[str]:
    """The node's defining equation, from the formula catalog."""
    return formula_equation(resolve(node.equation)) if node.equation else None


def workflow_table(spec: WorkflowSpec) -> List[dict]:
    """One documentation row per node: what it is, how it was obtained, and where VAFT has it."""
    side_targets = dict(spec.side)
    return [{"node": n.label, "kind": KIND_LABEL[n.kind], "symbols": n.symbols or "", "api": n.api or "",
             "ids": n.ids or "",
             "equation": node_equation(n) or "", "status": n.status or spec.status,
             "mapping_todo": n.mapping_todo or "", "enters": side_targets.get(n.key, "")} for n in spec.nodes]


def workflow_markdown(spec: WorkflowSpec) -> str:
    """The documentation table of a workflow, as Diagrams.md carries it (a test keeps the two equal)."""
    def cell(text):
        return text.replace("|", "\\|")

    lines = ["| Node | Kind | Variables | API | IDS |", "| --- | --- | --- | --- | --- |"]
    for row in workflow_table(spec):
        api = f"`{row['api']}`" if row["api"] else ""
        ids = f"`{row['ids']}`" if row["ids"] else ""
        if row["mapping_todo"]:
            ids = (ids + "; " if ids else "") + f"**IMAS mapping TODO:** {cell(row['mapping_todo'])}"
        kind = row["kind"] + (f" (enters {spec.node(row['enters']).label})" if row["enters"] else "")
        symbols = f"${cell(row['symbols']).replace(chr(92) + 'qquad', ',')}$" if row["symbols"] else ""
        lines.append(f"| {cell(row['node'])} | {cell(kind)} | {symbols} | {api} | {ids} |")
    if spec.todos:
        lines += ["", "Follow-up TODOs (implementation or IMAS mapping):", ""]
        lines += [f"- {todo}" for todo in spec.todos]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_W, _GAP, _VGAP = 5.6, 0.9, 0.75
#: node height: the label line, then one line each for variables, equation and tags [cm]
_H_LABEL, _H_SYMBOLS, _H_EQUATION, _H_TAGS = 0.75, 0.45, 0.7, 0.4
#: shrink an equation to the box's text width only when it is wider; never enlarge it
_TEXT_W = f"{_W - 0.4:.2f}cm"
_FIT = f"\\vaftfit{{{_TEXT_W}}}"


def _equation_parts(node: Node) -> List[str]:
    """The catalog definition, split where it joins separate relations (``,\\qquad``): one line each."""
    if not node.equation:
        return []
    return [part.strip().rstrip(",") for part in node_equation(node).split("\\qquad") if part.strip().rstrip(",")]


def _symbol_parts(node: Node) -> List[str]:
    """The representative variables, one line per ``\\qquad``-separated group."""
    if not node.symbols:
        return []
    return [part.strip().rstrip(",") for part in node.symbols.split("\\qquad") if part.strip().rstrip(",")]


def _relation_parts(node: Node) -> List[str]:
    if not node.relation:
        return []
    return [part.strip().rstrip(",") for part in node.relation.split("\\qquad") if part.strip().rstrip(",")]


#: a bold label wraps after about this many characters at the box width
_LABEL_CHARS = 29


def _label_lines(node: Node) -> int:
    return max(1, -(-len(node.label) // _LABEL_CHARS))


def _node_height(node: Node, spec_status: str) -> float:
    h = _H_LABEL + 0.2 + 0.45 * (_label_lines(node) - 1)
    h += _H_SYMBOLS * len(_symbol_parts(node))
    h += _H_EQUATION * (len(_equation_parts(node)) + len(_relation_parts(node)))
    h += _H_TAGS * (bool(node.ids) + bool(node.mapping_todo))
    return h


def _node_text(node: Node, spec_status: str) -> str:
    lines = ["\\textbf{" + escape_latex(node.label) + "}"]
    for part in _symbol_parts(node):
        lines.append(_FIT + f"{{${part}$}}")
    for part in _equation_parts(node) + _relation_parts(node):
        lines.append(_FIT + "{$\\displaystyle " + part + "$}")
    if node.ids:
        lines.append(_FIT + "{\\footnotesize\\texttt{" + escape_latex(node.ids) + "}}")
    if node.mapping_todo:
        lines.append(_FIT + "{\\footnotesize\\color{driftred}IMAS mapping TODO: " + escape_latex(node.mapping_todo) + "}")
    return "\\\\[2pt]".join(lines)


def _render_workflow(spec: WorkflowSpec, *, labels: bool = True) -> Diagram:
    """The workflow as a figure: rows top to bottom, assumptions from the left, equations in their node."""
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    items: List = []
    boxes: Dict[str, Box] = {}
    heights = {n.key: _node_height(n, spec.status) for n in spec.nodes}
    widest = max(len(row) for row in spec.rows)
    half_span = 0.5 * (widest * _W + (widest - 1) * _GAP)
    # which side each side input enters from: the rightmost box of a multi-box row takes its inputs from the
    # right, a lone box alternates left and right so a stack never outgrows its row, any other box the left
    sides: Dict[str, Dict[str, List[str]]] = {}
    for row in spec.rows:
        keys = [k for k in row if k is not None]
        for k in keys:
            sources = [src for src, dst in spec.side if dst == k]
            if len(keys) > 1:
                where = "right" if k == keys[-1] else "left"
                sides[k] = {"left": [] if where == "right" else sources, "right": sources if where == "right" else []}
            else:
                sides[k] = {"left": sources[0::2], "right": sources[1::2]}

    def stack_height(sources: List[str]) -> float:
        return sum(heights[s] for s in sources) + 0.2 * max(len(sources) - 1, 0)

    y = 0.0
    for r, row in enumerate(spec.rows):
        row_h = max(max(heights[k], stack_height(sides[k]["left"]), stack_height(sides[k]["right"]))
                    for k in row if k is not None)
        if r:
            y -= 0.5 * prev_h + _VGAP + 0.5 * row_h
        prev_h = row_h
        span = len(row) * _W + (len(row) - 1) * _GAP
        for c, key in enumerate(row):
            if key is None:
                continue
            node = spec.node(key)
            x = -0.5 * span + 0.5 * _W + c * (_W + _GAP)
            boxes[key] = _concept_box(x, y, _W, heights[key], _node_text(node, spec.status),
                                      style=kind_style(node.kind), text_style="concept plain", role=f"node:{key}",
                                      latex=True)
    bottom = y - 0.5 * prev_h
    # side inputs: stacked beside the step they enter, just outside the row
    side_x = -half_span - 1.6 - 0.5 * _W  # the farthest a side input can sit; most sit closer
    for row in spec.rows:
        keys = [k for k in row if k is not None]
        left = min(boxes[k].x - 0.5 * boxes[k].width for k in keys)
        right = max(boxes[k].x + 0.5 * boxes[k].width for k in keys)
        for dst in keys:
            for where, sx in (("left", left - 0.8 - 0.5 * _W), ("right", right + 0.8 + 0.5 * _W)):
                sources = sides[dst][where]
                top = boxes[dst].y + 0.5 * stack_height(sources)
                for src in sources:
                    node = spec.node(src)
                    cy = top - 0.5 * heights[src]
                    top -= heights[src] + 0.2
                    boxes[src] = _concept_box(sx, cy, _W, heights[src], _node_text(node, spec.status),
                                              style=kind_style(node.kind), text_style="concept plain",
                                              role=f"node:{src}", latex=True)
    for b in boxes.values():
        items += list(b.items)
    edges = list(spec.edges) + [(src, dst, "") for src, dst in spec.side]
    for src, dst, text in edges:
        arrow = connector(boxes[src], boxes[dst], style="connector strong", role=f"edge:{src}->{dst}")
        items.append(arrow)
        if text and labels:
            mx, my = 0.5 * (arrow.start[0] + arrow.end[0]), 0.5 * (arrow.start[1] + arrow.end[1])
            items.append(Label((mx - 0.15, my), escape_latex(text), "concept annotation", anchor="east",
                               role=f"edge label:{src}->{dst}"))
    equations = {n.key: node_equation(n) for n in spec.nodes if n.equation}
    if labels:
        top = max(b.y + 0.5 * b.height for b in boxes.values())
        mid = 0.5 * (min(b.x - 0.5 * b.width for b in boxes.values()) + max(b.x + 0.5 * b.width for b in boxes.values()))
        items.append(Label((mid, top + 0.35), "\\textbf{" + escape_latex(spec.title) + "}", "concept group title",
                           anchor="south", role="title"))
        # legend: the kinds this figure uses, in reading order
        used = [k for k in KINDS if any(n.kind == k for n in spec.nodes)]
        # wrapped to the figure's width so a many-kind legend never widens the figure
        left = min(b.x - 0.5 * b.width for b in boxes.values())
        right = max(b.x + 0.5 * b.width for b in boxes.values())
        widths = {k: 0.9 + 0.2 * len(KIND_LABEL[k]) for k in used}  # swatch + label, by label length
        lines: List[List[str]] = [[]]
        for kind in used:
            if lines[-1] and sum(widths[k] for k in lines[-1]) + widths[kind] > max(right - left, 12.0):
                lines.append([])
            lines[-1].append(kind)
        ly = bottom - 0.9
        for line in lines:
            x = 0.5 * (left + right) - 0.5 * sum(widths[k] for k in line)
            for kind in line:
                items += [Polyline.of([(x, ly - 0.17), (x + 0.45, ly - 0.17), (x + 0.45, ly + 0.17),
                                       (x, ly + 0.17)],
                                      f"{kind_style(kind)},rounded corners=1pt", role=f"legend:{kind}", closed=True),
                          Label((x + 0.55, ly), escape_latex(KIND_LABEL[kind]), "concept annotation", anchor="west",
                                role=f"legend:{kind}")]
                x += widths[kind]
            ly -= 0.55
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"spec": spec, "centers": {k: b.center for k, b in boxes.items()},
             "equations": equations, "side_x": side_x, "half_span": half_span}
    return Diagram(spec.key, Scene(tuple(items)), model=model)
