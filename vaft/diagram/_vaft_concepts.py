"""VAFT framework concept diagrams (#1090): what the framework is and how scientific data moves through it.

``vaft_four_pillars``
    four capabilities at one level -- standardized data interface, traceable
    and reproducible pipeline, FAIR scientific data repository, machine
    knowledge archive -- under the purpose and on the design principles;
``fusion_science_knowledge_lifecycle``
    the research-learning cycle from experiment back to experiment, named at
    its centre;
``scientific_workflow``
    a managed processing pipeline from heterogeneous sources to qualified,
    analysis-ready data, with cross-cutting management and a V&V feedback loop;
``interoperability_layers``
    from the machine to scientific workflows, native and standardized
    representations kept side by side;
``scientific_provenance_chain``
    traceable provenance from raw signal to analysis, with versioned inputs
    and configurations kept apart from quality metadata;
``scientific_infrastructure_principles``
    FAIR, W3C PROV and TRUST (common principles) beside three fusion-community
    requirements, each converging on VAFT, with references beneath each side;
``machine_agnostic_architecture``
    machine-specific access and mapping absorbs device differences beneath one
    Common Data Model (IMAS) and one shared framework, serving existing
    experiments and future devices;
``experiment_modeling_theory_data_network``
    why a common data model: point to point, through a common model, and the
    IMAS equilibrium as a tokamak example;
``human_ai_interface``
    human researchers and AI agents collaborating through shared interfaces
    on one backend;
``machine_research_archive``
    machine history, the research carried out on VEST and research knowledge
    since 2012, feeding a living archive that new research builds on (#497).

These are conceptual, capability-level diagrams: no storage backend,
endpoint, path or module name appears in them (implementation diagrams are
separate). The topology is in each diagram's model so tests and
documentation can read it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ._concept import Box, band, escape_latex
from ._concept import box as _concept_box
from ._concept import connector
from ._concept import database as _concept_database
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

#: patterns that name an implementation, never a concept: none may appear in these diagrams
IMPLEMENTATION_PATTERNS: Tuple[str, ...] = (r"\bHSDS\b", r"\bFileDB\b", r"\bh5pyd\b", r"\bREST\b", r"\bendpoints?\b",
                                            r"\bHDF5\b", r"\bSnakemake\b", r"\bvaft\.", r"\.py\b", r"\b[a-z_]+/[a-z_]+")
#: what a recorded edge means: one direction, both, a complementary path beside the main stack, or a feedback loop
EDGE_KINDS = ("forward", "both", "complement", "feedback")
#: the database, as every diagram of this module names it
DATABASE_TEXT = "\\textbf{IMAS Database}\\\\{\\footnotesize experimental shots \\& simulation runs}"


def box(x: float, y: float, width: float, height: float, text: str, **kwargs) -> Box:
    """A concept box whose text never hyphenates: these are names, read at README scale."""
    kwargs.setdefault("text_style", "concept plain")
    return _concept_box(x, y, width, height, text, **kwargs)


def database(x: float, y: float, width: float, height: float, text: str, **kwargs) -> Box:
    """A database drum whose text never hyphenates."""
    kwargs.setdefault("text_style", "concept plain")
    return _concept_database(x, y, width, height, text, **kwargs)


def _record(edges: List, src: str, dst: str, kind: str) -> None:
    if kind not in EDGE_KINDS:
        raise ValueError(f"edge kind must be one of {EDGE_KINDS}, not {kind!r}")
    edges.append((src, dst, kind))


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _edge(items: List, edges: List, a: Box, b: Box, src: str, dst: str, *, style: str = "connector",
          both: bool = False) -> None:
    arrow = connector(a, b, style=style, role=f"edge:{src}->{dst}")
    if both:
        arrow = Arrow(arrow.start, arrow.end, style, role=arrow.role, both=True)
    items.append(arrow)
    _record(edges, src, dst, "both" if both else "forward")


def _down_or_up(items: List, edges: List, a: Box, b: Box, src: str, dst: str, *, at_x=None,
                both: bool = False) -> None:
    """A vertical arrow between a and a box directly above or below it, at at_x (default a's x)."""
    x = a.x if at_x is None else at_x
    sign = 1.0 if b.y > a.y else -1.0
    start = (x, a.y + sign * (0.5 * a.height + 0.08))
    end = (x, b.y - sign * (0.5 * b.height + 0.08))
    items.append(Arrow(start, end, "connector both" if both else "connector", role=f"edge:{src}->{dst}", both=both))
    _record(edges, src, dst, "both" if both else "forward")


def _titled(title: str, sub: str = "", *, sub_size: str = "small") -> str:
    """LaTeX for a bold title with an optional smaller line below; both are escaped plain text."""
    text = "\\textbf{" + escape_latex(title) + "}"
    if sub:
        text += "\\\\{\\" + sub_size + " " + escape_latex(sub) + "}"
    return text


#: the four pillars, one abstraction level, named as the README's sections: (key, title, short phrases)
PILLARS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("interface", "Standardized Data Interface",
     ("Common Data Model (IMAS)", "machine-specific data access & mapping", "common data contract")),
    ("pipeline", "Traceable & Reproducible Pipeline",
     ("provenance of every product", "calibration & configuration", "reproducible reruns")),
    ("repository", "FAIR Scientific Data Repository",
     ("findable, accessible data", "native + standardized products", "validated, reusable")),
    ("archive", "Machine Knowledge Archive",
     ("machine configuration & history", "decisions & documentation", "tutorials, notebooks, know-how")),
)
#: the framework-level design principles, stated once for all four pillars
DESIGN_PRINCIPLES: Tuple[str, ...] = ("standardized", "traceable", "reproducible", "interoperable",
                                      "machine-agnostic")


def vaft_four_pillars(*, labels: bool = True) -> Diagram:
    r"""The four pillars of VAFT, under its purpose and on its design principles.

    An architecture figure: four capabilities at one level of abstraction --
    a standardized data interface, a traceable and reproducible pipeline, a
    FAIR scientific data repository and a machine knowledge archive -- hold
    up the purpose: integrate fusion experiment, modelling, data and
    knowledge so that plasma states are findable, comparable, reproducible
    and testable. They stand on the shared design principles. The research
    process they serve is ``fusion_science_knowledge_lifecycle``. VEST is
    labelled as the reference implementation, not drawn as the foundation:
    the architecture is machine-agnostic.
    """
    labels = _check_labels(labels)
    width, gap, height = 4.1, 0.45, 3.9
    span = 4 * width + 3 * gap
    x0 = -0.5 * span
    items: List = []
    roof = box(0.0, 0.5 * height + 0.95, span + 0.8, 1.3, "{\\large\\textbf{Integrate fusion experiment, modelling, "
               "data, and knowledge}}\\\\[2pt]to make plasma states findable, comparable, reproducible, and testable",
               style="concept roof", role="roof", latex=True)
    items += list(roof.items)
    for i, (key, title, points) in enumerate(PILLARS):
        cx = x0 + 0.5 * width + i * (width + gap)
        body = ("\\textbf{" + escape_latex(title) + "}\\\\[5pt]{\\small "
                + "\\\\[1pt]".join(map(escape_latex, points)) + "}")
        b = box(cx, 0.0, width, height, body, style="concept pillar", role=f"pillar:{key}", latex=True)
        items += list(b.items)
    principles = box(0.0, -0.5 * height - 0.6, span + 0.8, 0.8,
                     "\\textbf{Design principles}: " + " $\\cdot$ ".join(DESIGN_PRINCIPLES), style="concept base",
                     role="principles", latex=True)
    items += list(principles.items)
    items.append(Label((0.5 * span + 0.4, -0.5 * height - 1.1), "Reference implementation: VEST",
                       "concept annotation", anchor="north east", role="reference"))
    if labels:
        items.append(Label((0.0, -0.5 * height - 1.7), "Four capabilities at one level; the architecture is "
                           "machine-agnostic", "note", anchor="north", role="note"))
    return Diagram("vaft_four_pillars", Scene(tuple(items)),
                   model={"pillars": tuple(k for k, *_ in PILLARS), "titles": tuple(t for _, t, _ in PILLARS),
                          "principles": DESIGN_PRINCIPLES, "reference": "VEST"})


#: the research-learning cycle: (key, stage, what it holds)
LIFECYCLE: Tuple[Tuple[str, str, str], ...] = (
    ("experiment", "Experiment", "design and execution"),
    ("raw", "Machine Description & Raw Data", "geometry, coils, walls, diagnostics, conventions + raw measurements"),
    ("processing", "Data Processing & Qualification", "qualified scientific inputs"),
    ("modelling", "Modelling & Analysis", "reconstruction and modelling"),
    ("interpretation", "Physical Interpretation", "what the plasma did, and why"),
    ("comparison", "Comparison & Synthesis", "cross-shot, cross-campaign, cross-machine, cross-model"),
    ("discovery", "Discovery & New Questions", "new questions drive the next experiment"),
)


def fusion_science_knowledge_lifecycle(*, labels: bool = True) -> Diagram:
    r"""The research-learning cycle the infrastructure enables: from experiment back to experiment.

    Experiment; machine description and raw data -- the computationally
    usable machine (geometry, coils, walls, diagnostics, configurations,
    conventions) together with the raw measurements, kept apart from later
    plasma modelling; data processing and qualification into qualified
    scientific inputs (verification and validation cut across the whole
    pipeline, ``scientific_workflow``); modelling and analysis; physical
    interpretation; comparison and synthesis across shots, campaigns,
    machines and models; and discovery and new
    questions, which feed back into the design and execution of the next
    experiment -- the only stage they return to.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    boxes: Dict[str, Box] = {}
    rx, ry, w, h = 7.7, 5.3, 4.9, 1.7
    n = len(LIFECYCLE)
    for i, (key, title, sub) in enumerate(LIFECYCLE):
        a = math.radians(90.0 - 360.0 * i / n)
        style = "concept strong" if key == "experiment" else "concept box"
        b = box(rx * math.cos(a), ry * math.sin(a), w, h, _titled(title, sub, sub_size="footnotesize"), style=style,
                role=f"node:{key}", latex=True)
        boxes[key] = b
        items += list(b.items)
    keys = [k for k, _, _ in LIFECYCLE]
    for k0, k1 in zip(keys, keys[1:]):
        _edge(items, edges, boxes[k0], boxes[k1], k0, k1)
    # discoveries drive the next experiment: the one feedback edge, drawn stronger
    _edge(items, edges, boxes["discovery"], boxes["experiment"], "discovery", "experiment", style="connector strong")
    items.append(Label((0.0, 0.0), "{\\Large\\textbf{Research-learning cycle}}", "concept plain", role="title"))
    if labels:
        items.append(Label((0.0, -ry - 1.2), "Discoveries return to the experiment, the only stage new questions feed",
                           "note", anchor="north", role="note"))
    return Diagram("fusion_science_knowledge_lifecycle", Scene(tuple(items)),
                   model={"nodes": tuple(keys), "edges": tuple(edges), "feedback": ("discovery", "experiment")})


#: heterogeneous sources, the three interdependent phases and the cross-cutting management
SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("daq", "DAQ signals", ""), ("local", "Local PCs & files", "maintained by device owners"),
    ("cad", "Machine CAD & geometry", ""), ("logs", "Maintenance & operation logs", ""),
)
PHASES: Tuple[Tuple[str, str, str], ...] = (
    ("diagnostic", "Diagnostic Processing", "calibration, synchronization, derived signals"),
    ("reconstruction", "Reconstruction", "MHD equilibrium, kinetic profile fitting"),
    ("simulation", "Interpretive Simulation", "transport, stability, plasma response, synthetic diagnostics"),
)
MANAGEMENT: Tuple[Tuple[str, str, str], ...] = (
    ("configuration", "Configuration & Description", "settings, calibration, algorithms, machine descriptions, "
                                                      "conventions"),
    ("provenance", "Provenance & Versioning", "every product, its inputs and versions"),
    ("vv", "Verification, Validation & Quality Assessment", "quality metrics"),
)


def scientific_workflow(*, labels: bool = True) -> Diagram:
    r"""A managed scientific processing pipeline: heterogeneous sources to qualified, analysis-ready data.

    Heterogeneous machine and experimental sources (DAQ signals, local
    files kept by device owners, machine CAD and geometry, maintenance and
    operation logs) enter data ingestion and orchestration (source registry,
    legacy-to-IMAS mapping, access control, synchronization, provenance
    tracking). Diagnostic processing, equilibrium reconstruction and profile
    fitting, and interpretive simulation are an interdependent sequence that
    all read and write the standardized scientific state -- data in the
    Common Data Model (IMAS), a shared contract rather than a serial step. Configuration and description, provenance
    and versioning, and verification, validation and quality assessment cut
    across the pipeline, and V&V feeds back to the configurations: settings,
    calibration, algorithms, machine descriptions and conventions are
    refined continually. The product is qualified, analysis-ready data --
    quality flags, uncertainty, validation status, applicability -- selected
    by explicit quality criteria for integrated modelling, data analysis and
    physical interpretation.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    nodes: Dict[str, Box] = {}
    main_w = 15.2
    # the sources: one blue group box, the individual sources inside in a different colour
    items.append(Polyline.of([(-0.5 * main_w, 10.3), (0.5 * main_w, 10.3), (0.5 * main_w, 12.9), (-0.5 * main_w, 12.9)],
                             "concept box", role="group:sources", closed=True))
    items.append(Label((0.0, 12.75), escape_latex("Heterogeneous Machine & Experimental Sources"), "concept group title",
                       anchor="north", role="group:sources"))
    sw = (main_w - 0.8 - 3 * 0.35) / 4
    for i, (key, title, sub) in enumerate(SOURCES):
        x = -0.5 * main_w + 0.4 + 0.5 * sw + i * (sw + 0.35)
        b = box(x, 11.2, sw, 1.3, _titled(title, sub, sub_size="footnotesize"), style="concept source",
                role=f"source:{key}", latex=True)
        nodes[f"source:{key}"] = b
        items += list(b.items)
    nodes["ingestion"] = box(0.0, 8.8, main_w, 1.2, _titled(
        "Data Ingestion & Orchestration", "") + "\\\\{\\small source registry $\\cdot$ legacy $\\leftrightarrow$ IMAS "
        "mapping $\\cdot$ access control $\\cdot$ synchronization $\\cdot$ provenance tracking}",
        role="node:ingestion", latex=True)
    items.append(Arrow((0.0, 10.3 - 0.08), (0.0, 9.4 + 0.08), "connector", role="edge:sources->ingestion"))
    _record(edges, "sources", "ingestion", "forward")
    pw = 4.5
    for i, (key, title, sub) in enumerate(PHASES):
        nodes[key] = box((i - 1) * 5.35, 6.3, pw, 2.0, _titled(title, sub, sub_size="footnotesize"),
                         role=f"node:{key}", latex=True)
    nodes["state"] = box(0.0, 3.8, main_w, 1.05, "\\textbf{Standardized scientific state}: data in the Common Data "
                         "Model (IMAS), read and written by every phase", style="concept state", role="node:state",
                         latex=True)
    nodes["qualified"] = box(0.0, 1.9, 10.0, 1.2, _titled("Qualified Analysis-Ready Data",
                                                          "quality flags, uncertainty, validation status, "
                                                          "applicability"), style="concept strong",
                             role="node:qualified", latex=True)
    nodes["analysis"] = box(0.0, 0.0, 6.4, 0.95, "Integrated Modelling & Data Analysis", role="node:analysis")
    nodes["interpretation"] = box(0.0, -1.8, 6.4, 0.95, "Physical Interpretation", role="node:interpretation")
    # cross-cutting management, a column beside the pipeline
    mx, mw = 0.5 * main_w + 3.3, 5.0
    items += band(mx - 0.5 * mw - 0.3, mx + 0.5 * mw + 0.3, 2.75, 10.5, "cross-cutting", role="group:management")
    for (key, title, sub), y, hh in zip(MANAGEMENT, (8.8, 6.3, 3.85), (2.0, 1.6, 1.6)):
        style = "concept vv" if key == "vv" else "concept leaf"
        nodes[key] = box(mx, y, mw, hh, _titled(title, sub, sub_size="footnotesize"), style=style,
                         role=f"node:{key}", latex=True)
    for key, b in nodes.items():
        if not key.startswith("source:"):  # the sources were drawn inside their group
            items += list(b.items)
    first = nodes["diagnostic"]
    _down_or_up(items, edges, nodes["ingestion"], first, "ingestion", "diagnostic", at_x=first.x)
    for (k0, _, _), (k1, _, _) in zip(PHASES, PHASES[1:]):
        _edge(items, edges, nodes[k0], nodes[k1], k0, k1)
    for key, _, _ in PHASES:  # each phase reads and writes the shared state
        _down_or_up(items, edges, nodes[key], nodes["state"], key, "state", both=True)
    _down_or_up(items, edges, nodes["state"], nodes["qualified"], "state", "qualified")
    _edge(items, edges, nodes["qualified"], nodes["analysis"], "qualified", "analysis")
    _edge(items, edges, nodes["analysis"], nodes["interpretation"], "analysis", "interpretation")
    # management: configuration drives the orchestration, provenance records the phases, V&V evaluates the state
    _edge(items, edges, nodes["configuration"], nodes["ingestion"], "configuration", "ingestion",
          style="connector dependency")
    _edge(items, edges, nodes["simulation"], nodes["provenance"], "simulation", "provenance",
          style="connector dependency")
    _edge(items, edges, nodes["state"], nodes["vv"], "state", "vv", style="connector dependency")
    _edge(items, edges, nodes["vv"], nodes["qualified"], "vv", "qualified")
    # the improvement loop: V&V refines the configurations
    vv, cfg = nodes["vv"], nodes["configuration"]
    xr = mx + 0.5 * mw + 0.75
    items.append(Polyline.of([(vv.x + 0.5 * mw + 0.08, vv.y), (xr, vv.y), (xr, cfg.y), (cfg.x + 0.5 * mw + 0.08, cfg.y)],
                             "connector feedback", role="edge:vv->configuration"))
    _record(edges, "vv", "configuration", "feedback")
    items.append(Label((xr + 0.15, 0.5 * (vv.y + cfg.y)), "continuous improvement", "concept annotation,rotate=-90",
                       anchor="south", role="improvement"))
    if labels:
        items.append(Label((0.0, -2.7), "Not storage alone: reproducible orchestration, evaluation, qualification and "
                           "continual improvement of scientific data products", "note", anchor="north", role="note"))
    return Diagram("scientific_workflow", Scene(tuple(items)),
                   model={"nodes": tuple(k for k in nodes if not k.startswith("source:")),
                          "sources": tuple(k for k, _, _ in SOURCES), "phases": tuple(k for k, _, _ in PHASES),
                          "management": tuple(k for k, _, _ in MANAGEMENT), "edges": tuple(edges)})


#: the interoperability stack, machine to workflows
LAYERS: Tuple[Tuple[str, str], ...] = (
    ("machine", "Machine / Experiment"),
    ("native", "Native Scientific Representation"),
    ("standardization", "Validation / Standardization"),
    ("imas", "Common Data Model (IMAS)"),
    ("database", "IMAS Database: experimental shots & simulation runs"),
    ("workflows", "Scientific Workflows"),
)


def interoperability_layers(*, labels: bool = True) -> Diagram:
    r"""Interoperability as layers: from the machine to scientific workflows, in both directions.

    Machine, native scientific representation, validation and
    standardization, the Common Data Model (IMAS), the IMAS database and scientific
    workflows, each exchanging with its neighbours. Native artifacts are
    kept beside the standardized representation -- the dashed path -- so
    standardization complements the native form rather than replacing it.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    w, h, step = 9.6, 0.95, 1.95
    boxes: Dict[str, Box] = {}
    for i, (key, text) in enumerate(LAYERS):
        style = "concept state" if key == "imas" else "concept box"
        b = box(0.0, -i * step, w, h, text, style=style, role=f"layer:{key}")
        boxes[key] = b
        items += list(b.items)
    keys = [k for k, _ in LAYERS]
    for a, b in zip(keys, keys[1:]):
        _edge(items, edges, boxes[a], boxes[b], a, b, both=True)
    # native artifacts stay available next to the standard: a path around the standardization
    native, db = boxes["native"], boxes["database"]
    x = 0.5 * w + 1.0
    items.append(Polyline.of([(0.5 * w + 0.08, native.y), (x, native.y), (x, db.y), (0.5 * w + 0.08, db.y)],
                             "connector feedback,<->", role="edge:native->database"))
    _record(edges, "native", "database", "complement")
    if labels:
        items.append(Label((x + 0.15, 0.5 * (native.y + db.y)), "native artifacts stored alongside",
                           "concept annotation,rotate=-90", anchor="south", role="complement"))
        items.append(Label((0.0, -(len(LAYERS) - 1) * step - 0.9), "Standardized access complements native "
                           "scientific artifacts; it does not replace them", "note", anchor="north", role="note"))
    return Diagram("interoperability_layers", Scene(tuple(items)),
                   model={"layers": tuple(keys), "edges": tuple(edges)})


#: provenance steps and what each one records
PROVENANCE: Tuple[Tuple[str, str, str], ...] = (
    ("raw", "Raw Signal", "shot, channel, acquisition"),
    ("processed", "Processed Data", "calibration, mapping, processing configuration"),
    ("reconstruction", "Equilibrium Reconstruction & Profile Fitting", "code, model, configuration, conventions"),
    ("derived", "Derived Physics Quantities", "definition, units, uncertainty"),
    ("analysis", "Analysis & Visualization", "analysis method, selection criteria, environment"),
)
#: what is versioned, and what quality metadata travels with every product
VERSIONED: Tuple[str, ...] = ("machine description", "geometry", "calibration", "mappings", "conventions",
                              "processing and model configuration", "schema")
QUALITY: Tuple[str, ...] = ("validation status", "uncertainty", "applicability")


def _dotted(parts: Sequence[str]) -> str:
    return " $\\cdot$ ".join(map(escape_latex, parts))


def scientific_provenance_chain(*, labels: bool = True) -> Diagram:
    r"""Traceable provenance: every scientific product points to its inputs, configuration and version.

    Raw signal, processed data, equilibrium reconstruction and profile
    fitting, derived physics quantities, analysis and visualization. Each
    step records what produced it; calibration, mapping and processing
    configuration are provenance of the raw-to-processed step, not a stage
    of their own. Two kinds of metadata are kept apart: versioned
    provenance (machine description, geometry, calibration, mappings,
    conventions, configurations, schema) and quality metadata (validation
    status, uncertainty, applicability), which cuts across every product
    rather than being a final stage. Reproducibility needs both the lineage
    and the exact configurations, conventions, versions and quality
    assessments behind each quantity.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    w, h, gap = 3.4, 1.55, 0.95
    boxes: List[Box] = []
    for i, (key, text, records) in enumerate(PROVENANCE):
        b = box(i * (w + gap), 0.0, w, h, text, role=f"step:{key}")
        boxes.append(b)
        items += list(b.items)
        items.append(Label((b.x, -0.5 * h - 0.15), _dotted(records.split(", ")),
                           "concept annotation,text width=3.2cm", anchor="north", role=f"record:{key}"))
    for (k0, _, _), (k1, _, _), a, b in zip(PROVENANCE, PROVENANCE[1:], boxes, boxes[1:]):
        _edge(items, edges, a, b, k0, k1)
    first, last = boxes[0], boxes[-1]
    left, right = first.x - 0.5 * w, last.x + 0.5 * w
    y = 0.5 * h + 0.7
    items.append(Polyline.of([(last.x, 0.5 * h + 0.08), (last.x, y), (first.x, y), (first.x, 0.5 * h + 0.08)],
                             "connector feedback", role="trace"))
    items.append(Label((left, y + 0.75), "Example: a tokamak analysis provenance chain", "concept band label",
                       anchor="south west", role="example"))
    items.append(Label((0.5 * (left + right), y + 0.1), "\\textbf{Traceable provenance}: every scientific product "
                       "points to its inputs, configuration, and version", "concept plain", anchor="south",
                       role="trace"))
    # quality metadata cuts across every product
    qy = -3.35
    quality = box(0.5 * (left + right), qy, right - left, 0.75, "\\textbf{Quality metadata} (on every product, not "
                  "a stage): " + _dotted(QUALITY), style="concept state", role="quality", latex=True)
    items += list(quality.items)
    for b in boxes:
        items.append(Polyline.of([(b.x, -0.5 * h - 1.45), (b.x, qy + 0.375)], "concept tick", role="quality"))
    versioned = box(0.5 * (left + right), qy - 1.25, right - left, 1.05, "\\textbf{Versioned inputs \\& configurations}: "
                    + _dotted(VERSIONED), style="concept leaf", role="versioned", latex=True)
    items += list(versioned.items)
    if labels:
        items.append(Label((0.5 * (left + right), qy - 2.0), "Reproducibility needs the lineage and the exact "
                           "configurations, conventions, versions and quality assessments behind every product",
                           "note", anchor="north", role="note"))
    return Diagram("scientific_provenance_chain", Scene(tuple(items)),
                   model={"steps": tuple(k for k, _, _ in PROVENANCE), "edges": tuple(edges),
                          "records": {k: r for k, _, r in PROVENANCE}, "trace": ("analysis", "raw"),
                          "versioned": VERSIONED, "quality": QUALITY})


#: common principles for modern scientific infrastructure: (key, title, subtitle, content)
COMMON_PRINCIPLES: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("fair", "FAIR Principles", "data & research software", ("Findable", "Accessible", "Interoperable", "Reusable")),
    ("prov", "W3C PROV", "provenance & traceability", ("products, inputs", "activities, configurations",
                                                        "versions, agents")),
    ("trust", "TRUST", "trustworthy repositories", ("Transparency", "Responsibility", "User focus",
                                                    "Sustainability", "Technology")),
)
#: the fusion-specific complement: methodological requirements and practice of the fusion community
FUSION_REQUIREMENTS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("vvuq", "Verification & Validation",
     ("model-to-model benchmark", "validation against experiment", "uncertainty quantification",
      "domain of validity")),
    ("integrated", "Integrated Modelling & Data Analysis",
     ("common data model", "experiment-model integration", "heterogeneous diagnostics", "coupled physics models")),
    ("multimachine", "Multi-machine Comparison & Extrapolation",
     ("qualified experimental data", "scaling & model validation", "extrapolation to future fusion devices")),
)
#: representative references of each side
PRINCIPLE_REFERENCES: Tuple[str, ...] = (
    "FAIR / FAIR4RS: Wilkinson et al. (2016); Barker et al. (2022)",
    "Provenance: W3C PROV (2013)",
    "Trustworthy repositories: Lin et al. (2020)",
)
#: the fusion references, grouped by the block they support
FUSION_REFERENCES: Tuple[str, ...] = (
    "V&V: Terry et al. (2008); Greenwald (2010)",
    "Integrated analysis & modelling: Fischer & Dinklage (2004);", "Imbeaux et al. (2015); Meneghini et al. (2015)",
    "Multi-machine comparison & extrapolation:", "ITER Physics Basis (1999, 2007, 2025); ITPA / ITPEA activities",
)


def scientific_infrastructure_principles(*, labels: bool = True) -> Diagram:
    r"""Two complementary foundations of VAFT, converging on one infrastructure.

    On the left, common principles for modern scientific infrastructure,
    adopted from established cross-disciplinary practice: FAIR for data and
    research software, W3C PROV for provenance and traceability, TRUST for
    trustworthy repository operation. On the right, the methodological
    requirements and practice established in the fusion community, in three
    blocks that balance the left without a one-to-one correspondence:
    verification and validation (model-to-model benchmarks, validation
    against experiment, uncertainty quantification, domain of validity);
    integrated modelling and data analysis, starting from a common data
    model; multi-machine comparison and extrapolation. The two headings are
    parallel, and each block converges independently on VAFT, an
    architecture designed around both (not a claim of complete compliance).
    Beneath each side, its references grouped by block; FAIR4RS is named
    there, next to FAIR.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    w, gap, h, split = 3.8, 0.3, 5.0, 1.2
    side = 3 * w + 2 * gap
    total = 2 * side + split
    x0 = -0.5 * total
    left_c, right_c = x0 + 0.5 * side, 0.5 * total - 0.5 * side
    vaft = box(0.0, -0.5 * h - 1.3, total, 0.95, "\\textbf{VAFT}: integrated scientific infrastructure designed around both",
               style="concept strong", role="node:vaft", latex=True)
    items += list(vaft.items)
    blocks = [(key, _titled(title, sub) + "\\\\[5pt]{\\small " + "\\\\".join(map(escape_latex, content)) + "}",
               "concept pillar", "principle") for key, title, sub, content in COMMON_PRINCIPLES]
    for key, title, content in FUSION_REQUIREMENTS:
        lines = list(map(escape_latex, content))
        blocks.append((key, _titled(title) + "\\\\[6pt]{\\small " + "\\\\[3pt]".join(lines) + "}",
                       "concept requirement", "requirement"))
    for i, (key, body, style, kind) in enumerate(blocks):
        x = x0 + 0.5 * w + i * (w + gap) + (split - gap if i >= 3 else 0.0)
        b = box(x, 0.0, w, h, body, style=style, role=f"{kind}:{key}", latex=True)
        items += list(b.items)
        _down_or_up(items, edges, b, vaft, f"{kind}:{key}", "vaft")
    # parallel headings over the two sides
    top = 0.5 * h + 0.15
    for c, text, role in ((left_c, "Common Principles for\\\\Modern Scientific Infrastructure", "heading:common"),
                          (right_c, "Fusion-Community Requirements\\\\for Scientific Validity \\& Reliability",
                           "heading:fusion")):
        lo, hi = c - 0.5 * side, c + 0.5 * side
        items += [Polyline.of([(lo, top), (lo, top + 0.2), (hi, top + 0.2), (hi, top)], "concept tick", role=role),
                  Label((c, top + 0.3), text, "concept group title,align=center", anchor="south", role=role)]
    # compact foundations and references beneath each side
    fy = vaft.y - 0.8
    items += [Label((left_c, fy), "\\textbf{Common-principle references}", "concept plain", anchor="north",
                    role="footer:common"),
              Label((left_c, fy - 0.55), "\\\\".join(map(escape_latex, PRINCIPLE_REFERENCES)), "concept reference",
                    anchor="north", role="references"),
              Label((right_c, fy), "\\textbf{Fusion-community references}", "concept plain", anchor="north",
                    role="footer:fusion"),
              Label((right_c, fy - 0.55), "\\\\".join(map(escape_latex, FUSION_REFERENCES)), "concept reference",
                    anchor="north", role="fusion_references")]
    if labels:
        items.append(Label((0.0, fy - 2.7), "General scientific-infrastructure principles and fusion-specific validity "
                           "requirements, integrated in one architecture", "note", anchor="north", role="note"))
    return Diagram("scientific_infrastructure_principles", Scene(tuple(items)),
                   model={"principles": tuple(k for k, *_ in COMMON_PRINCIPLES),
                          "requirements": tuple(k for k, *_ in FUSION_REQUIREMENTS),
                          "references": PRINCIPLE_REFERENCES, "fusion_references": FUSION_REFERENCES,
                          "edges": tuple(edges)})


#: the research modes the framework serves
RESEARCH: Tuple[Tuple[str, str], ...] = (("theory", "Theory"), ("experiment", "Experiment"),
                                         ("simulation", "Modelling & Simulation"),
                                         ("data_driven", "Data-driven Methods (AI/ML)"))
#: the two application domains, by the science they serve rather than by device or implementation status
DOMAINS: Tuple[Tuple[str, str, str], ...] = (
    ("existing", "Existing Fusion Experiments", "measured and reconstructed plasma states"),
    ("future", "Future Fusion Devices & Reactor Concepts", "design studies, predicted plasma states, simulation"),
)


def machine_agnostic_architecture(*, labels: bool = True) -> Diagram:
    r"""A machine-agnostic scientific framework for integrated fusion research.

    Theory, experiment, modelling and simulation, and data-driven methods
    share one scientific framework of reusable building blocks -- formulas,
    data processing, code interfaces, workflows, plots, diagrams. Below it,
    one Common Data Model (IMAS) describes design, experimental and
    simulation data alike, and is stored and reused in the IMAS database of
    experimental shots and simulation runs. Machine-specific data access and
    mapping (native access, metadata and conventions, mapping into IMAS)
    absorbs the differences between devices, so that everything above stays
    machine-agnostic. The same architecture serves two domains: existing
    fusion experiments (measured and reconstructed plasma states) and future
    fusion devices and reactor concepts (design studies, predicted states,
    simulation). No device is named.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    span = 17.0
    gap = 0.35
    rw = (span - 3 * gap) / 4
    research: Dict[str, Box] = {}
    for i, (key, text) in enumerate(RESEARCH):
        x = -0.5 * span + 0.5 * rw + i * (rw + gap)
        research[key] = box(x, 8.65, rw, 1.0, text, style="concept state", role=f"research:{key}")
        items += list(research[key].items)
    framework = box(0.0, 6.3, span, 1.4, "\\textbf{Shared Scientific Framework}\\\\{\\small reusable building "
                    "blocks: formulas $\\cdot$ data processing $\\cdot$ code interfaces $\\cdot$ workflows $\\cdot$ "
                    "plots $\\cdot$ diagrams}", style="concept strong", role="layer:framework", latex=True)
    standard = box(0.0, 4.0, span, 1.2, "\\textbf{Common Data Model (IMAS)}\\\\{\\small design "
                   "$\\cdot$ experimental $\\cdot$ simulation data}", style="concept state", role="layer:standard",
                   latex=True)
    mapping = box(0.0, 1.7, span, 1.3, "\\textbf{Machine-specific Data Access \\& Mapping}\\\\{\\small native "
                  "data access $\\cdot$ metadata \\& conventions $\\cdot$ mapping into IMAS}", style="concept leaf",
                  role="layer:mapping", latex=True)
    items += list(framework.items) + list(standard.items) + list(mapping.items)
    for key, r in research.items():
        items.append(Arrow((r.x, framework.y + 0.7 + 0.08), (r.x, r.y - 0.5 - 0.08), "connector both",
                           role=f"edge:framework->research:{key}", both=True))
        _record(edges, "framework", f"research:{key}", "both")
    _edge(items, edges, standard, framework, "standard", "framework", both=True)
    _edge(items, edges, mapping, standard, "mapping", "standard")
    db = database(0.5 * span + 3.2, 4.0, 4.4, 2.5, DATABASE_TEXT, role="node:database", latex=True)
    items += list(db.items)
    _edge(items, edges, standard, db, "standard", "database", both=True)
    dw = 0.5 * (span - 0.6)
    for (key, title, sub), x in zip(DOMAINS, (-0.25 * span - 0.15, 0.25 * span + 0.15)):
        d = box(x, -0.6, dw, 1.25, _titled(title, sub), style="concept base", role=f"domain:{key}", latex=True)
        items += list(d.items)
        items.append(Arrow((x, d.y + 0.625 + 0.08), (x, mapping.y - 0.65 - 0.08), "connector",
                           role=f"edge:domain:{key}->mapping"))
        _record(edges, f"domain:{key}", "mapping", "forward")
    items.append(Label((-0.5 * span - 0.3, mapping.y), "device differences\\\\absorbed here", "concept annotation",
                       anchor="east", role="absorb"))
    items.append(Label((-0.5 * span - 0.3, 0.5 * (framework.y + standard.y)), "machine-agnostic\\\\above",
                       "concept annotation", anchor="east", role="absorb"))
    if labels:
        items.append(Label((0.0, -1.55), "One framework, representation and database for present experimental data "
                           "and for predictive, design-oriented research", "note", anchor="north", role="note"))
    return Diagram("machine_agnostic_architecture", Scene(tuple(items)),
                   model={"research": tuple(k for k, _ in RESEARCH), "domains": tuple(k for k, _, _ in DOMAINS),
                          "edges": tuple(edges)})


#: the four research modes, with their colours
ACTIVITIES: Tuple[Tuple[str, str, str], ...] = (
    ("experiment", "Experiment", "absfluid"), ("modelling", "Modelling", "driftred"),
    ("theory", "Theory", "abskinetic"), ("data_driven", "Data-driven (AI/ML)", "islandblue"),
)
#: the tokamak example: representative equilibrium routes of each mode
EQUILIBRIUM_ROUTES: Dict[str, str] = {
    "experiment": "EFIT", "modelling": "CHEASE, TokaMaker", "theory": "Solov'ev, Guazzotto & Freidberg",
    "data_driven": "neural / surrogate equilibrium models",
}
#: compact references of the tokamak example
EQUILIBRIUM_REFERENCES: Tuple[str, ...] = (
    "EFIT: Lao et al., Nucl. Fusion 25, 1611 (1985)",
    "CHEASE: L\\\"utjens, Bondeson \\& Sauter, Comput. Phys. Commun. 97, 219 (1996)",
    "TokaMaker: Hansen et al., Comput. Phys. Commun. 298, 109111 (2024)",
    "Solov'ev, Sov. Phys. JETP 26, 400 (1968)",
    "Guazzotto \\& Freidberg, J. Plasma Phys. 87, 905870303 (2021)",
    "Neural equilibrium: Joung et al., Nucl. Fusion 60, 016034 (2020)",
)
#: the references, one or two a line so the footer is no wider than the figure: indices into the tuple
_REFERENCE_LINES = ((0, 3), (1,), (2,), (4,), (5,))
_COMMUNICATION = ("point_to_point", "common_model", "equilibrium")


#: half the distance between the four research modes: they sit at the corners of a square around the shared state
_MODE_OFFSET = 3.5


def _research_modes(items: List, example: bool) -> Dict[str, Box]:
    """The four research modes at the corners, as every common-model figure places them; boxes appended to items."""
    d = _MODE_OFFSET
    nodes: Dict[str, Box] = {}
    for (key, text, colour), (x, y) in zip(ACTIVITIES, [(-d, d), (d, d), (-d, -d), (d, -d)]):
        body = _titled(text, EQUILIBRIUM_ROUTES[key]) if example else f"\\textbf{{{escape_latex(text)}}}"
        b = box(x, y, 4.5, 1.45 if example else 1.0, body,
                style=f"concept actor,fill={colour}!18,draw={colour}!80", role=f"node:{key}", latex=True)
        nodes[key] = b
        items += list(b.items)
    return nodes


def _common_hub(items: List, edges: List, nodes: Dict[str, Box], example: bool) -> Box:
    """The shared state at the centre, exchanging with every research mode."""
    hub_text = ("\\textbf{IMAS Equilibrium IDS}\\\\[3pt]$\\Delta^{*}\\psi = -\\mu_0 R^2 p'(\\psi) - FF'(\\psi)$"
                if example else
                "\\textbf{Common Data Model (IMAS)}\\\\{\\small shared standardized representation}")
    hub = box(0.0, 0.0, 5.6, 1.4, hub_text, style="concept hub", role="node:hub", latex=True)
    items += list(hub.items)
    for k, _, _ in ACTIVITIES:
        _edge(items, edges, nodes[k], hub, k, "hub", both=True)
    return hub


def experiment_modeling_theory_data_network(communication: str = "common_model", *,
                                            labels: bool = True) -> Diagram:
    r"""Why a common fusion data model: four research modes, pairwise against a shared representation.

    ``communication`` selects one diagram of a three-step sequence:

    ``"point_to_point"``
        experiment, modelling, theory and data-driven (AI/ML) research
        connected pairwise -- without a common model, $N(N-1)/2$ pairwise
        adapters, and a new mode needs $N-1$ more;
    ``"common_model"``
        the same four modes each reading from and writing to one Common Data
        Model (IMAS) -- $N$ common-model adapters, and a new mode needs one;
    ``"equilibrium"``
        the concrete tokamak case: the IMAS equilibrium IDS at the centre and
        representative routes of each mode (EFIT; CHEASE, TokaMaker;
        Solov'ev, Guazzotto & Freidberg; neural / surrogate equilibrium
        models). The routes produce, consume and compare a common
        standardized equilibrium representation; they are not equivalent
        methods. Compact references sit beneath.
    """
    labels = _check_labels(labels)
    if communication not in _COMMUNICATION:
        raise ValueError(f"communication must be one of {_COMMUNICATION}, not {communication!r}")
    items: List = []
    edges: List = []
    d = _MODE_OFFSET
    example = communication == "equilibrium"
    nodes = _research_modes(items, example)
    keys = [k for k, _, _ in ACTIVITIES]
    n = len(keys)
    if communication == "point_to_point":
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                _edge(items, edges, nodes[a], nodes[b], a, b, both=True)
        adapters = n * (n - 1) // 2
        caption = f"\\textbf{{Without a common model}}: $N(N-1)/2 = {adapters}$ pairwise adapters"
        note = "A new research mode needs $N-1$ new adapters"
    else:
        _common_hub(items, edges, nodes, example)
        adapters = n
        caption = (f"\\textbf{{With a common model}}: $N = {adapters}$ common-model adapters" if not example else
                   "Different scientific routes produce, consume, and compare\\\\a common standardized equilibrium "
                   "representation")
        note = ("A new research mode needs 1 new adapter" if not example else
                "Representative routes, not equivalent methods")
    items.append(Label((0.0, -d - 1.1), caption, "concept plain", anchor="north", role="caption"))
    if example:
        lines = ("; ".join(EQUILIBRIUM_REFERENCES[i] for i in group) for group in _REFERENCE_LINES)
        items.append(Label((0.0, -d - 2.3), "\\\\".join(lines), "concept reference", anchor="north",
                           role="references"))
    if labels:
        items.append(Label((0.0, -d - (4.7 if example else 1.75)), note, "note", anchor="north", role="note"))
    return Diagram("experiment_modeling_theory_data_network", Scene(tuple(items)),
                   model={"communication": communication, "nodes": tuple(keys), "edges": tuple(edges),
                          "adapters": adapters,
                          "routes": dict(EQUILIBRIUM_ROUTES) if example else {}})


#: the scientific domains the framework figure can be specialised to
_FRAMEWORK_DOMAINS = (None, "equilibrium")
#: what an equilibrium is analysed into: equilibrium-derived descriptors, not downstream stability models
EQUILIBRIUM_ANALYSIS: Tuple[Tuple[str, str], ...] = (
    ("mhd_parameters", "MHD Parameters"), ("plasma_shape", "Plasma Shape"), ("operational_space", "Operational Space"),
)


def integrated_scientific_framework(domain=None, *, labels: bool = True) -> Diagram:
    r"""The research modes and their common state inside one integrated framework, serving analysis (#1698).

    The four research modes and the Common Data Model (IMAS) sit exactly as in
    ``experiment_modeling_theory_data_network("common_model")``, inside an
    Integrated Framework boundary, and the shared state feeds one downstream
    Analysis node. Three concepts are kept apart: the Common Data Model is the
    shared representation; the framework connects, runs, compares and
    reproduces research through it; analysis is the scientific use of the
    integrated state.

    ``domain="equilibrium"`` specialises the figure: the representative
    equilibrium routes of each mode and the IMAS equilibrium IDS, with an
    analysis of MHD parameters, plasma shape and operational space -- the
    equilibrium-derived descriptors. Stability codes are downstream models of
    the equilibrium and stay out of it. Compact references sit beneath.
    """
    labels = _check_labels(labels)
    if domain not in _FRAMEWORK_DOMAINS:
        raise ValueError(f"domain must be one of {_FRAMEWORK_DOMAINS}, not {domain!r}")
    example = domain == "equilibrium"
    d = _MODE_OFFSET
    items: List = []
    edges: List = []
    # the framework boundary first, so everything else is drawn over it
    frame_x, frame_top = d + 2.25 + 0.55, d + (0.725 if example else 0.5) + 1.25
    analysis_y = -d - 2.45
    analysis_h = 1.15 if example else 0.95
    frame_bottom = analysis_y - 0.5 * analysis_h - 0.5
    items.append(Polyline.of([(-frame_x, frame_bottom), (frame_x, frame_bottom), (frame_x, frame_top),
                              (-frame_x, frame_top)], "concept frame", role="framework", closed=True))
    items.append(Label((-frame_x + 0.25, frame_top - 0.15), "\\textbf{Integrated Framework}", "concept group title",
                       anchor="north west", role="framework"))
    items.append(Label((frame_x - 0.25, frame_top - 0.2), "connects, runs, compares and reproduces research",
                       "concept annotation", anchor="north east", role="framework"))
    nodes = _research_modes(items, example)
    hub = _common_hub(items, edges, nodes, example)
    body = "\\textbf{Analysis}"
    if example:
        body += "\\\\{\\small " + " $\\cdot$ ".join(escape_latex(t) for _, t in EQUILIBRIUM_ANALYSIS) + "}"
    analysis = box(0.0, analysis_y, 9.4 if example else 4.0, analysis_h, body, style="concept strong",
                   role="node:analysis", latex=True)
    items += list(analysis.items)
    _down_or_up(items, edges, hub, analysis, "hub", "analysis")
    if example:
        lines = ("; ".join(EQUILIBRIUM_REFERENCES[i] for i in group) for group in _REFERENCE_LINES)
        items.append(Label((0.0, frame_bottom - 0.25), "\\\\".join(lines), "concept reference", anchor="north",
                           role="references"))
    if labels:
        note = "Common Data Model: shared representation; framework: connects and reproduces; analysis: the use"
        items.append(Label((0.0, frame_bottom - (2.75 if example else 0.3)), note, "note", anchor="north", role="note"))
    return Diagram("integrated_scientific_framework", Scene(tuple(items)),
                   model={"framework": "integrated", "domain": domain,
                          "activities": tuple(k for k, _, _ in ACTIVITIES),
                          "shared_state": "equilibrium" if example else "common_data_model",
                          "analysis": "analysis",
                          "analysis_categories": tuple(k for k, _ in EQUILIBRIUM_ANALYSIS) if example else (),
                          "routes": dict(EQUILIBRIUM_ROUTES) if example else {}, "edges": tuple(edges)})


#: interfaces and the interaction each one is for: (key, name, use, planned)
INTERFACES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("python", "Python API", "scripts, notebooks", False),
    ("cli", "CLI", "commands, batch runs", False),
    ("gui", "GUI", "visual, interactive", False),
    ("repository", "Repository & Docs", "code, issues, releases, documentation", False),
    ("mcp", "MCP", "agent tools", True),
)


def human_ai_interface(*, labels: bool = True) -> Diagram:
    r"""Human-AI collaborative access: who, how, and what.

    Human researchers and AI agents collaborate (a two-way relationship,
    not two isolated user classes) and reach one shared interface layer:
    the Python API (scripts, notebooks), the CLI (commands, batch runs), a
    GUI (visual, interactive), the repository and docs (code, issues,
    releases and documentation) and MCP (agent tools; planned).
    Every interface reaches the same backend -- the VAFT framework and its
    scientific data repository -- and none is reserved for one kind of user.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    w, gap = 3.9, 0.3
    span = len(INTERFACES) * w + (len(INTERFACES) - 1) * gap
    human = box(-4.6, 5.0, 4.0, 1.0, "\\textbf{Human researchers}", style="concept actor", role="actor:human",
                latex=True)
    agent = box(4.6, 5.0, 4.0, 1.0, "\\textbf{AI agents}", style="concept actor", role="actor:agent", latex=True)
    items += list(human.items) + list(agent.items)
    _edge(items, edges, human, agent, "actor:human", "actor:agent", both=True)
    items.append(Label((0.0, 5.1), "collaborate", "concept annotation", anchor="south", role="collaborate"))
    bar_y = 3.75
    xs = [-0.5 * span + 0.5 * w + i * (w + gap) for i in range(len(INTERFACES))]
    items.append(Polyline.of([(xs[0], bar_y), (xs[-1], bar_y)], "connector line", role="access"))
    for key, actor in (("human", human), ("agent", agent)):
        items.append(Arrow((actor.x, actor.y - 0.5 - 0.08), (actor.x, bar_y), "connector",
                           role=f"edge:actor:{key}->access"))
        _record(edges, f"actor:{key}", "access", "forward")
    # the shared interface layer, labelled, collected into one connection to the backend
    items += band(-0.5 * span - 0.25, 0.5 * span + 0.25, 0.2, 3.05, role="layer:interfaces")
    items.append(Label((-0.5 * span - 0.4, 1.6), "Shared Access Interfaces", "concept band label,rotate=90", anchor="south",
                       role="layer:interfaces"))
    collector_y, backend_top, backend_bottom = -0.3, -1.05, -3.45
    items.append(Polyline.of([(xs[0], collector_y), (xs[-1], collector_y)], "connector line", role="collector"))
    items.append(Arrow((0.0, collector_y), (0.0, backend_top + 0.08), "connector strong",
                       role="edge:interfaces->backend"))
    _record(edges, "interfaces", "backend", "forward")
    items += band(-0.5 * span + 2.0, 0.5 * span - 2.0, backend_bottom, backend_top, "shared backend", role="backend")
    framework = box(-3.2, -2.25, 5.2, 1.0, "\\textbf{VAFT Framework}", role="node:framework", latex=True)
    repository = database(3.6, -2.25, 5.2, 2.45, DATABASE_TEXT, role="node:repository", latex=True)
    items += list(framework.items) + list(repository.items)
    _edge(items, edges, framework, repository, "framework", "repository", both=True)
    for (key, name, use, planned), x in zip(INTERFACES, xs):
        body = _titled(name, use)
        if planned:
            body += "\\\\{\\small\\itshape planned}"
        b = box(x, 1.45, w, 2.1, body, style="concept group" if planned else "concept box",
                role=f"interface:{key}", latex=True)
        items += list(b.items)
        items.append(Arrow((x, bar_y), (x, b.y + 1.05 + 0.08), "connector", role=f"edge:access->interface:{key}"))
        _record(edges, "access", f"interface:{key}", "forward")
        items.append(Polyline.of([(x, b.y - 1.05), (x, collector_y)], "connector line", role=f"edge:interface:{key}->interfaces"))
        _record(edges, f"interface:{key}", "interfaces", "forward")
    if labels:
        items.append(Label((0.0, backend_bottom - 0.3), "Any combination of shared interfaces reaches the same "
                           "framework, project knowledge and scientific data", "note", anchor="north", role="note"))
    return Diagram("human_ai_interface", Scene(tuple(items)),
                   model={"actors": ("human", "agent"), "interfaces": tuple(k for k, *_ in INTERFACES),
                          "planned": tuple(k for k, *_, p in INTERFACES if p), "backend": ("framework", "repository"),
                          "edges": tuple(edges)})


#: the three tracks of the archive, in no particular chronology: what each accumulates
ARCHIVE_TRACKS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("machine", "Machine history", ("geometry & hardware revisions", "diagnostic additions",
                                    "calibration changes", "operation & maintenance logs")),
    ("studies", "Research on VEST", ("spherical-torus operation", "diagnostic development",
                                     "start-up, heating & current drive", "disruptions & transient MHD",
                                     "equilibrium reconstruction", "confinement & operational limits")),
    ("research", "Research knowledge", ("experimental procedures", "reconstruction & modelling workflows",
                                        "documentation & tutorials", "reference datasets & notebooks",
                                        "publications & reproducible analyses")),
)


def machine_research_archive(*, labels: bool = True) -> Diagram:
    r"""The machine and research archive: VEST's institutional and scientific memory since 2012.

    Three tracks run from the start of VEST operation in 2012 to today:
    machine history (geometry and hardware revisions, diagnostic additions,
    calibration changes, operation and maintenance logs); the research
    actually carried out on VEST, after the README's "Research historically
    performed on VEST" (spherical-torus operation, diagnostic development,
    start-up, heating and current drive, disruptions and transient MHD,
    equilibrium reconstruction, confinement and operational limits); and
    research knowledge (procedures, reconstruction and modelling workflows,
    documentation and tutorials, reference datasets and notebooks,
    publications and reproducible analyses). All three feed one living
    research archive, which is not the end of the pipeline: new analyses,
    workflows and research build on it and extend every track. No dates are
    drawn beyond the start of operation; the figure shows what accumulates,
    not when.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    x0, x1 = 0.0, 17.5
    track_y = {"machine": 6.55, "studies": 3.6, "research": 0.65}
    for key, title, entries in ARCHIVE_TRACKS:
        y = track_y[key]
        items += band(x0, x1, y - 1.3, y + 1.3, role=f"track:{key}")
        items.append(Label((x0 + 0.2, y + 1.0), "\\textbf{" + escape_latex(title) + "}", "concept plain",
                           anchor="north west", role=f"track:{key}"))
        n = len(entries)
        w = (x1 - x0 - 0.6 - (n - 1) * 0.25) / n
        for i, text in enumerate(entries):
            b = box(x0 + 0.3 + 0.5 * w + i * (w + 0.25), y - 0.3, w, 1.45, text, style="concept leaf",
                    role=f"entry:{key}:{i}")
            items += list(b.items)
    archive = box(x1 + 2.9, track_y["studies"], 3.6, 2.4,
                  "\\textbf{Living research archive}\\\\[3pt]{\\small machine and research memory, kept usable}",
                  style="concept strong", role="node:archive", latex=True)
    items += list(archive.items)
    for key, y in track_y.items():  # every track ends in the one archive
        bx, by = archive.boundary_point((x1, y))
        sx, sy = x1 + 0.08, y
        length = math.hypot(bx - sx, by - sy)
        end = (float(bx - 0.08 * (bx - sx) / length), float(by - 0.08 * (by - sy) / length))
        items.append(Arrow((sx, sy), end, "connector", role=f"edge:track:{key}->archive"))
        _record(edges, f"track:{key}", "archive", "forward")
    research = box(archive.x, -2.0, 3.6, 1.0, "New analyses, workflows & research", role="node:next")
    items += list(research.items)
    _down_or_up(items, edges, archive, research, "archive", "next")
    # the archive is an input: new work extends every track
    y_back = -2.0
    xl = x0 - 0.8
    items.append(Polyline.of([(research.x - 0.5 * research.width - 0.08, y_back), (xl, y_back), (xl, track_y["machine"]),
                              (x0 - 0.08, track_y["machine"])], "connector feedback", role="edge:next->track:machine"))
    _record(edges, "next", "track:machine", "feedback")
    for key in ("studies", "research"):
        items.append(Arrow((xl, track_y[key]), (x0 - 0.08, track_y[key]), "connector feedback",
                           role=f"edge:next->track:{key}"))
        _record(edges, "next", f"track:{key}", "feedback")
    items.append(Label((0.5 * (xl + research.x - 0.5 * research.width), y_back - 0.1),
                       "new shots and studies extend every track", "concept annotation", anchor="north",
                       role="feedback_label"))
    items += [Arrow((x0, -0.85), (x1, -0.85), "connector line,->", role="time"),
              Label((x0, -0.95), "2012: VEST operation begins", "concept annotation", anchor="north west",
                    role="time"),
              Label((x1, -0.95), "today", "concept annotation", anchor="north east", role="time")]
    if labels:
        items.append(Label((0.5 * (xl + archive.x + 1.8), -3.0), "A living research archive, not a file store: "
                           "what VEST has learned stays usable for verification, comparison and study",
                           "note", anchor="north", role="note"))
    return Diagram("machine_research_archive", Scene(tuple(items)),
                   model={"tracks": tuple(k for k, _, _ in ARCHIVE_TRACKS),
                          "entries": {k: e for k, _, e in ARCHIVE_TRACKS}, "edges": tuple(edges), "start": 2012})
