"""Scientific computation contracts and computation records (#2150).

``scientific_computation_contract``
    one computation's scientific inputs and outputs, kept apart from its
    scientific configuration, its execution settings and its execution
    record;
``computation_record_layers``
    what one run leaves behind, in three layers that are never merged:
    scientific identity, observed execution and scientific assessment.

Both share one visual grammar with ``vaft.diagram._workflow_execution``:
a solid arrow carries scientific data, a dashed arrow configuration or
metadata, a dotted arrow a provenance or evidence reference. Every node
names the source it was drawn from in ``Diagram.model["evidence"]``, and
every node carries an implementation status, so a figure never shows
planned capability as done. ``scientific_credibility(organization=
"integrated")`` already draws qualification as independent axes ending in
a profile rather than one score, so the issue's optional single-run
qualification figure is not built; the assessment layer here carries the
single-run view instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ._concept import Box, connector, escape_latex
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene
from ._vaft_concepts import _check_labels, box

#: what an edge means, and how it is drawn: the style alone never decides the meaning, the model records it
RELATIONS: Dict[str, str] = {
    "data": "connector",
    "configuration": "connector,dashed,black!60",
    "provenance": "connector,densely dotted,black!60",
    "part": "connector dependency",
}
#: how far a capability is implemented today
STATUSES = ("implemented", "partial", "experimental", "planned")
#: the legend's wording for each drawn relation
RELATION_LEGEND: Tuple[Tuple[str, str], ...] = (
    ("data", "scientific data"), ("configuration", "configuration or metadata"),
    ("provenance", "provenance or evidence reference"),
)


class Graph:
    """Nodes, typed edges, statuses and evidence of one figure, drawn as they are recorded."""

    def __init__(self):
        self.items: List = []
        self.nodes: Dict[str, Box] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.status: Dict[str, str] = {}
        self.evidence: Dict[str, Tuple[Tuple[str, str], ...]] = {}

    def node(self, key: str, x: float, y: float, width: float, height: float, title: str,
             lines: Sequence[str] = (), *, status: str = "implemented", note: str = "", footer: str = "",
             evidence: Sequence[Tuple[str, str]] = (), style: Optional[str] = None,
             title_size: str = "") -> Box:
        """A box: a bold title, plain lines, an optional italic note; dashed when it is planned."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, not {status!r}")
        text = "\\textbf{" + title_size + escape_latex(title) + "}"
        if lines:
            text += "\\\\{\\small " + "\\\\".join(escape_latex(t) for t in lines) + "}"
        if note:
            text += "\\\\{\\footnotesize\\itshape " + escape_latex(note) + "}"
        if footer:  # raw LaTeX closing the box, set off from its contents
            text += "\\\\[4pt]{\\footnotesize\\color{black!60}" + footer + "}"
        if style is None:
            style = "concept group" if status == "planned" else "concept box"
        b = box(x, y, width, height, text, style=style, role=f"node:{key}", latex=True)
        self.items += list(b.items)
        self.nodes[key] = b
        self.status[key] = status
        self.evidence[key] = tuple(evidence)
        return b

    def link(self, src: str, dst: str, relation: str, *, via: Sequence[Tuple[float, float]] = (),
             start: Optional[Tuple[float, float]] = None, end: Optional[Tuple[float, float]] = None) -> None:
        """One recorded edge, drawn once: centre to centre, between given points, or as one bent polyline."""
        if relation not in RELATIONS:
            raise ValueError(f"relation must be one of {tuple(RELATIONS)}, not {relation!r}")
        style, role = RELATIONS[relation], f"edge:{src}->{dst}"
        if start is None and end is None and not via:
            self.items.append(connector(self.nodes[src], self.nodes[dst], style=style, role=role))
        elif via:
            self.items.append(Polyline.of([start, *via, end], style, role=role))
        else:
            self.items.append(Arrow(start, end, style, role=role))
        self.edges.append((src, dst, relation))

    def label(self, at, text: str, *, anchor: str = "center", style: str = "concept annotation",
              role: str = "annotation") -> None:
        self.items.append(Label(at, text, style, anchor=anchor, role=role))

    def legend(self, x: float, y: float, *, relations: Sequence[str] = ("data", "configuration", "provenance"),
               spacing: float = 6.4) -> None:
        """Short sample arrows with their meaning, left to right from ``(x, y)``."""
        words = dict(RELATION_LEGEND)
        for i, relation in enumerate(relations):
            x0 = x + i * spacing
            self.items.append(Arrow((x0, y), (x0 + 1.0, y), RELATIONS[relation], role="legend"))
            self.items.append(Label((x0 + 1.2, y), escape_latex(words[relation]), "concept annotation",
                                    anchor="west", role="legend"))

    def diagram(self, name: str, **model) -> Diagram:
        return Diagram(name, Scene(tuple(self.items)),
                       model={"nodes": tuple(self.nodes), "edges": tuple(self.edges), "status": dict(self.status),
                              "evidence": dict(self.evidence), **model})


def scientific_computation_contract(*, labels: bool = True) -> Diagram:
    r"""One computation's scientific I/O, apart from its configuration, execution settings and record.

    Scientific inputs -- IDS or ODS data, or typed data such as arrays and
    GEQDSK or k-files -- enter a computation, a Process function or a Code
    adapter. Two kinds of setting reach it: the scientific configuration
    (physical assumptions, model and numerical choices, units and
    conventions -- everything that changes the result; EFIT hashes it into
    its scientific identity) and the execution settings (executable,
    backend and resources, timeout, work directory -- which do not). The
    computation returns a standardized output (IDS or ODS, the subset
    comparable across codes) beside its native and typed results (solver
    files, parsed result objects), which keep the solver-specific science.
    The configuration may travel with the standardized output as the IDS
    ``code.parameters`` string; it is not necessarily an external file.
    The execution record (return code, runtime status, elapsed time, logs)
    is separate metadata.

    Not every Process or Code reads or writes IDS, and none needs an Actor.
    """
    labels = _check_labels(labels)
    g = Graph()
    g.node("scientific_configuration", -3.7, 4.7, 6.8, 1.9, "Scientific configuration",
           ("physical assumptions · model and", "numerical choices · units and conventions"),
           note="changes the result", evidence=(("vaft/code/efit/config.py", "class EFITScientificConfig"),))
    g.node("execution_settings", 3.7, 4.7, 6.8, 1.9, "Execution settings",
           ("executable · backend and resources", "timeout · work directory"),
           note="does not change the result", evidence=(("vaft/code/base.py", "class CodeConfig"),
                                                        ("vaft/code/execution.py", "class ExecutionRequest")))
    g.node("computation", 0.0, 1.2, 6.0, 1.6, "Scientific computation", ("a Process function or a Code adapter",),
           style="concept strong", evidence=(("vaft/code/base.py", "class CodeRunner"),))
    g.node("inputs", -8.7, 1.2, 5.6, 2.0, "Scientific inputs",
           ("IDS / ODS, or typed data:", "arrays, GEQDSK, k-files"),
           evidence=(("vaft/code/base.py", "class CodeInputs"), ("vaft/code/chease.py", "def prepare_chease_inputs")))
    g.node("standardized_output", 8.7, 2.5, 5.6, 1.9, "Standardized output",
           ("IDS / ODS: the comparable subset;", "configuration may ride along", "as code.parameters"),
           evidence=(("vaft/imas/code_parameters.py", "def as_entry_payload"),))
    g.node("native_results", 8.7, -0.1, 5.6, 1.6, "Native and typed results",
           ("solver files · parsed result objects",), evidence=(("vaft/code/base.py", "class CodeResult"),))
    g.node("execution_record", 0.0, -1.95, 8.4, 1.5, "Execution record (separate metadata)",
           ("return code · runtime status · elapsed time · logs",), status="partial",
           note="partial: kept per adapter, not yet uniformly persisted",
           evidence=(("vaft/code/execution.py", "class ExecutionResult"),))
    g.link("inputs", "computation", "data")
    g.link("computation", "standardized_output", "data")
    g.link("computation", "native_results", "data")
    g.link("scientific_configuration", "computation", "configuration")
    g.link("execution_settings", "computation", "configuration")
    g.link("computation", "execution_record", "provenance")
    g.legend(-9.2, -3.3)
    if labels:
        g.label((0.0, -3.9), "Not every Process or Code reads or writes IDS; the standardized output keeps a "
                "comparable subset beside the solver-specific science", anchor="north", style="note", role="note")
    return g.diagram("scientific_computation_contract", relations=RELATIONS)


#: the three record layers: (key, title, contents, what keeps it apart, status, status note, evidence)
RECORD_LAYERS: Tuple[Tuple[str, str, Tuple[str, ...], str, str, str, Tuple[Tuple[str, str], ...]], ...] = (
    ("identity", "Scientific identity",
     ("input digests", "code and model version", "scientific configuration", "conventions", "content digests"),
     "each digest states what enters it", "implemented",
     "scientific SHA-256 for EFIT; input and output digests per stage",
     (("vaft/code/efit/config.py", "def sha256"), ("vaft/database/replication.py", "product_sha256"))),
    ("execution", "Observed execution",
     ("backend", "job ID", "runtime and resources", "exit status", "stdout / stderr logs"),
     "nondeterministic: kept out of the identity", "partial",
     "recorded per adapter; uniform persistence planned (#1019)",
     (("vaft/code/execution.py", "class ExecutionResult"), ("vaft/database/worker/state.py", "runs"))),
    ("assessment", "Scientific assessment",
     ("execution success", "numerical convergence", "uncertainty", "physical checks", "applicability (advisory)",
      "validation"),
     "separate axes, not one score", "partial",
     "axes implemented; not every axis has a producer",
     (("vaft/validation/credibility.py", "AXES"), ("vaft/code/efit/status.py", "class EFITSliceStatus"),
      ("vaft/validation/model_contracts.py", "def safe_ordering_provenance"))),
)


def computation_record_layers(*, labels: bool = True) -> Diagram:
    r"""What one computation run leaves behind, in three layers kept apart (#2150).

    ``"identity"``
        the scientific identity: input digests, code and model version,
        scientific configuration, conventions and content digests. Each
        digest declares its scope -- EFIT's scientific SHA-256 covers the
        scientific configuration, not the execution block, the input data or
        the advisory ordering check -- rather than one universal hash of all
        metadata;
    ``"execution"``
        the observed execution: backend, job ID, runtime and resources, exit
        status, stdout and stderr. These values are nondeterministic and stay
        out of the identity while remaining provenance for reproducibility;
        today they are recorded per adapter, and uniform persistence is
        planned (#1019);
    ``"assessment"``
        the scientific assessment: execution success, numerical convergence,
        uncertainty, physical checks, applicability (advisory) and
        validation, as separate axes rather than one score.

    Execution success, provenance completeness, numerical convergence,
    validation and model applicability are not conflated: each is its own
    entry. All three layers point to the traceable result.
    """
    labels = _check_labels(labels)
    g = Graph()
    g.node("run", 0.0, 5.4, 5.0, 0.95, "Computation run", style="concept strong")
    width, height, y, xs = 6.0, 5.0, 1.45, (-6.6, 0.0, 6.6)
    for (key, title, lines, apart, status, status_note, evidence), x in zip(RECORD_LAYERS, xs):
        # the implementation status closes the box, apart from what it records
        g.node(key, x, y, width, height, title, lines, status=status, note=apart, evidence=evidence,
               footer="\\textit{" + escape_latex(status) + "}: " + escape_latex(status_note))
        g.link("run", key, "part")
    result_y = -2.4
    g.node("result", 0.0, result_y, 8.4, 1.0, "Traceable result and its references", style="concept strong")
    bottom = y - 0.5 * height - 0.08
    for key, x in zip(("identity", "execution", "assessment"), xs):
        g.link(key, "result", "provenance", start=(x, bottom), end=(0.38 * x, result_y + 0.5 + 0.08))
    g.legend(-5.4, -3.55, relations=("provenance",))
    g.items.append(Arrow((1.0, -3.55), (2.0, -3.55), RELATIONS["part"], role="legend"))
    g.label((2.2, -3.55), "part of the run's record", anchor="west", role="legend")
    if labels:
        g.label((0.0, -4.1), "Execution success $\\neq$ numerical convergence $\\neq$ physical validity; "
                "provenance completeness $\\neq$ scientific correctness", anchor="north", style="note", role="note")
    return g.diagram("computation_record_layers", layers=tuple(k for k, *_ in RECORD_LAYERS), relations=RELATIONS)
