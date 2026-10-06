"""Research-infrastructure concept diagrams: why VAFT is built the way it is.

Four fragmented-versus-integrated pairs, one per level of scientific
integration, share one visual grammar; two further diagrams place the
research community and the ownership of scientific logic.

``scientific_representation`` (#1641, level 1)
    one scientific meaning, many encodings, structures, names and words;
``experimental_research_infrastructure`` (#1636, level 2)
    scattered sources and research paths against a Common Data Model, a FAIR
    scientific data repository and a research framework;
``scientific_credibility`` (#1638, level 3)
    locally reasonable steps that lose evidence, against a common credibility
    and traceability structure ending in a qualified scientific state;
``research_modality_architecture`` (#1640, level 4)
    science locked into one interface, language or environment, against one
    modular scientific core with many ways to research;
``fusion_research_ecosystem`` (#1643)
    research roles, activities, shared scientific states and research
    contexts, in a detailed and a presentation rendering of one model;
``scientific_ownership_architecture`` (#1645)
    where reusable scientific logic belongs, how workflow logic matures into
    it, and where validation, use policy, Study and Research sit.

The pair grammar. ``organization="fragmented"`` draws the problem: for
levels 1, 2 and 4 the research paths as dashed silos joined only by red,
dashed ad-hoc links; for level 3, one chain of steps with the evidence each
step loses beneath it. ``organization="integrated"`` draws the same entities
around the shared layer that answers them. Both variants of a pair carry the
level tag and a title, the take-home note, and numbered notes as columns of
text: what goes wrong (plain words, the technical term and, where there is
one, a reference) in the fragmented figure, what answers it under the same
number in the integrated one. The numbers are cited like footnotes -- a grey
superscript after the text it belongs to -- to mark where a symptom arises
and which part of the architecture answers it. They run in the order the
fragmented figure cites them, so the integrated figure cites the same numbers
in its own order.

As in ``_vaft_concepts`` these are capability-level figures: no backend,
endpoint, path or module path appears in them. Their content is the data
at the top of each section, and the topology is in each diagram's model.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Dict, List, Sequence, Tuple

from ._concept import Box, band, escape_latex
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene
from ._vaft_concepts import _check_labels, _down_or_up, _edge, _record, _titled, box, database

#: the two variants of every pair
ORGANIZATIONS = ("fragmented", "integrated")
#: the levels of scientific integration, in reading order: (key, tag)
LEVELS: Tuple[Tuple[str, str], ...] = (
    ("representation", "Level 1 · Scientific representation"),
    ("infrastructure", "Level 2 · Research infrastructure"),
    ("credibility", "Level 3 · Scientific credibility"),
    ("modality", "Level 4 · Research modality & portability"),
)
#: how a symptom, a response and an ad-hoc link are drawn (inline options: the template is shared by every diagram)
SYMPTOM_STYLE = "concept leaf,draw=driftred!70,fill=driftred!6"
ADHOC_STYLE = "connector,driftred!75,dashed"
SILO_STYLE = "concept group"
#: correspondence numbers are citations: a quiet grey superscript, the same for symptoms and responses
CITE_COLOUR = "black!50"
_NOTES_TITLE = {"symptom": "What goes wrong", "response": "What answers it (same numbers)"}


#: the few symbols these figures use, set as math (escape_latex leaves them untouched)
_SYMBOLS = (("·", "$\\cdot$"), ("≈", "$\\approx$"), ("≠", "$\\neq$"), ("→", "$\\rightarrow$"), ("×", "$\\times$"),
            ("“", "``"), ("”", "''"))


def _tex(text: str) -> str:
    """Plain text made literal for LaTeX, with :data:`_SYMBOLS` set as math."""
    text = escape_latex(text)
    for symbol, math in _SYMBOLS:
        text = text.replace(symbol, math)
    return text


def _check_organization(organization: str) -> str:
    if organization not in ORGANIZATIONS:
        raise ValueError(f"organization must be one of {ORGANIZATIONS}, not {organization!r}")
    return organization


def _cite(numbers: Sequence[int]) -> str:
    """LaTeX for one or more correspondence numbers, cited like a footnote: a grey superscript ``4,8``."""
    return "\\textsuperscript{\\textcolor{" + CITE_COLOUR + "}{" + ",".join(map(str, numbers)) + "}}"


def _chip_text(number: int, text: str) -> str:
    """A chip's LaTeX: its citation number and plain label (the term and reference are in the notes)."""
    return "\\small " + _cite((number,)) + "\\," + _tex(text)


def _heading(items: List, x0: float, y: float, level: str, title: str) -> None:
    """The level tag at the top-left and the variant's title centred below it."""
    tag = dict(LEVELS)[level]
    items.append(Label((x0, y + 0.95), _tex(tag), "concept band label", anchor="south west", role="level"))
    items.append(Label((0.0, y), "{\\Large\\textbf{" + _tex(title) + "}}", "concept plain", anchor="south",
                       role="title"))


def _after_title(text: str, cite: str) -> str:
    """``text`` with ``cite`` right after its title: the first ``\\textbf{...}`` or leading ``{...}`` group."""
    start = text.find("\\textbf{")
    if start < 0 and text.startswith("{"):
        start = 0
    if start < 0:
        return text + cite
    depth = 0
    for i in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[:i + 1] + cite + text[i + 1:]
    raise ValueError(f"unbalanced braces in {text!r}")


def _tag(items: List, tags: Dict, b: Box, key: str, numbers: Sequence[int]) -> None:
    """Cite ``numbers`` after the title of box ``b``, whose items are already in ``items``."""
    label = b.items[-1]
    i = next(j for j, it in enumerate(items) if it is label)
    items[i] = replace(label, text=_after_title(label.text, _cite(numbers)))
    tags[key] = tuple(numbers)


def _numbers(keys: Sequence[str], correspondence) -> Tuple[int, ...]:
    """The citation numbers of symptom ``keys``: their 1-based places in ``correspondence``, ascending."""
    order = [k for k, *_ in correspondence]
    return tuple(sorted(order.index(k) + 1 for k in keys))


def _footnotes(items: List, labels: bool, correspondence, kind: str, x0: float, x1: float, y_top: float, *,
               columns: int, note: str) -> None:
    """The take-home note, then the numbered notes as columns of text.

    A symptom reads ``n plain words (technical term) [reference]``; a response
    ``n plain words``. Numbers run down the first column, then the next.
    """
    _note(items, labels, y_top, note)
    title_y = y_top - 0.7
    items.append(Label((x0, title_y), _tex(_NOTES_TITLE[kind]), "concept band label", anchor="north west",
                       role=f"notes:{kind}"))
    lines = []
    for i, (key, symptom, term, reference, response) in enumerate(correspondence, start=1):
        line = "\\hangindent=1.1em\\hangafter=1\\noindent" + _cite((i,)) + "\\," + _tex(
            symptom if kind == "symptom" else response)
        if kind == "symptom" and term:
            line += " \\textit{(" + _tex(term) + ")}"
        if kind == "symptom" and reference:
            line += " \\mbox{\\textcolor{" + CITE_COLOUR + "}{[" + _tex(reference) + "]}}"  # never split
        lines.append(line)
    per = -(-len(lines) // columns)
    width = (x1 - x0) / columns
    for c in range(columns):
        chunk = lines[c * per:(c + 1) * per]
        if chunk:
            items.append(Label((x0 + c * width, title_y - 0.45), "\\par ".join(chunk),
                               f"concept plain,align=left,font=\\small,text width={width - 0.35:.2f}cm",
                               anchor="north west", role=f"notes:{kind}:{c}"))


def _silo(items: List, x0: float, x1: float, y0: float, y1: float, role: str, title: str = "") -> None:
    """A dashed panel around one isolated research path, its title at the top-left."""
    items.append(Polyline.of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], SILO_STYLE, role=role, closed=True))
    if title:
        items.append(Label((x0 + 0.15, y1 - 0.1), title, "concept band label", anchor="north west", role=role))


def _horizontal(items: List, edges: List, a: Box, b: Box, src: str, dst: str, *, style: str = "connector",
                both: bool = False) -> None:
    """A horizontal arrow between two boxes on one row."""
    sign = 1.0 if b.x > a.x else -1.0
    start = (a.x + sign * (0.5 * a.width + 0.08), a.y)
    end = (b.x - sign * (0.5 * b.width + 0.08), a.y)
    items.append(Arrow(start, end, style, role=f"edge:{src}->{dst}", both=both))
    _record(edges, src, dst, "both" if both else "forward")


def _correspondence(entries) -> Tuple[Tuple[str, str, str, str, str], ...]:
    """(key, symptom, technical term, reference, response) for the model of either variant."""
    return tuple(tuple(entry) for entry in entries)


def _note(items: List, labels: bool, y: float, text: str) -> None:
    if labels:
        items.append(Label((0.0, y), _tex(text), "note", anchor="north", role="note"))


# --------------------------------------------------------------------------------------------------------------
# Level 2: experimental research infrastructure (#1636)
# --------------------------------------------------------------------------------------------------------------

#: heterogeneous experimental sources, as both variants name them
INFRA_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("daq", "DAQ signals"), ("machine_db", "Machine database"), ("diagnostic_files", "Diagnostic files"),
    ("geometry", "Machine geometry"), ("records", "Operation & maintenance records"),
    ("calibration", "Calibration & configuration"),
)
#: scientific activities, as both variants name them
INFRA_ACTIVITIES: Tuple[Tuple[str, str], ...] = (
    ("diagnostic", "Diagnostic Processing"), ("reconstruction", "Reconstruction"), ("profile", "Profile Analysis"),
    ("simulation", "Simulation / Modelling"), ("physics", "Physics Analysis"),
)
#: the fragmented research paths: (lane, source, ad-hoc step, activity, the private copy the lane keeps)
INFRA_LANES: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("acquisition", "daq", "local preprocessing", "diagnostic", "own calibration copy"),
    ("equilibrium", "machine_db", "conversion script", "reconstruction", "own settings & conventions"),
    ("profiles", "diagnostic_files", "fitting notebook", "profile", "own calibration copy"),
    ("modelling", "geometry", "custom converter", "simulation", "own geometry conventions"),
    ("analysis", "records", "manual lookup", "physics", "own selection criteria"),
)
#: the lanes that receive a hand-copied equilibrium from the reconstruction lane
INFRA_EQUILIBRIUM_COPIES: Tuple[str, ...] = ("profile", "simulation", "physics")
#: the three complementary capabilities and their distinct roles: (key, title, role, provides)
INFRA_CAPABILITIES: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("repository", "FAIR Scientific Data Repository", "organization · preservation · reuse",
     ("persistent, findable products", "provenance & metadata", "reuse across shots, campaigns, machines")),
    ("cdm", "Common Data Model (IMAS)", "representation · interoperability",
     ("standardized scientific representation", "shared semantic contract", "interoperable inputs & outputs")),
    ("framework", "Research Framework", "workflow · computation · reproducibility",
     ("ingestion, processing, reconstruction", "analysis & simulation interfaces", "validation, reproducible runs")),
)
#: symptom -> response, one-to-one, numbered in the order the fragmented figure cites them:
#: (key, plain symptom, technical term, reference, response)
INFRA_CORRESPONDENCE: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("conversion", "Every path writes its own converter", "ad-hoc conversion", "", "One mapping into a common model"),
    ("personal", "Only its author can rerun it", "researcher-specific workflow", "", "Workflows anyone can rerun"),
    ("configuration", "Settings copied and kept apart", "scattered configuration", "", "Versioned configuration"),
    ("duplication", "Same processing redone in each path", "duplicated processing", "", "Shared, tested processing"),
    ("provenance", "Unclear where a result came from", "provenance loss", "W3C PROV 2013",
     "Provenance on every product"),
)
#: where each symptom arises (fragmented) and which capability answers it (integrated)
INFRA_SYMPTOM_TAGS: Dict[str, Tuple[str, ...]] = {
    "step": ("conversion", "personal"), "copy": ("configuration",), "equilibrium_copies": ("duplication", "provenance")}
INFRA_RESPONSE_TAGS: Dict[str, Tuple[str, ...]] = {
    "repository": ("provenance",), "cdm": ("conversion",), "framework": ("personal", "configuration", "duplication")}

_INFRA_HALF_WIDTH = 9.9


def experimental_research_infrastructure(organization: str = "integrated", *, labels: bool = True) -> Diagram:
    r"""Experimental research, fragmented and integrated: why a Common Data Model alone is not enough (#1636).

    ``organization`` selects one figure of the pair:

    ``"fragmented"``
        five research paths, each a silo from a heterogeneous source (DAQ
        signals, machine database, diagnostic files, machine geometry,
        operation and maintenance records) through its own ad-hoc step
        (local preprocessing, a conversion script, a fitting notebook, a
        custom converter, a manual lookup) to one activity (diagnostic
        processing, reconstruction, profile analysis, simulation, physics
        analysis). Each path keeps its own calibration, settings or
        conventions; the reconstruction is copied by hand into three other
        paths. Every link is a plausible dependency, not decoration. The
        symptoms -- ad-hoc conversion, duplicated processing, scattered
        configuration, weak provenance, researcher-specific workflows -- are
        listed beneath, as consequences rather than stages;
    ``"integrated"``
        the same sources -- with calibration and configuration, kept as
        private copies in the fragmented figure, now a source of its own --
        and the same activities meet in one research infrastructure
        of three complementary capabilities with distinct roles: the Common
        Data Model (IMAS) for representation and interoperability, the FAIR
        scientific data repository for organization, preservation and reuse,
        and the research framework for workflow, computation and
        reproducibility. Activities connect only through it.

    The data-level argument (point to point against a common model) is
    ``experiment_modeling_theory_data_network``; the managed pipeline is
    ``scientific_workflow``; the VEST implementation is ``vest_data_platform``.
    """
    organization = _check_organization(organization)
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    tags: Dict[str, Tuple[int, ...]] = {}
    cited = partial(_numbers, correspondence=INFRA_CORRESPONDENCE)
    names = dict(INFRA_SOURCES)
    activities = dict(INFRA_ACTIVITIES)
    x0, x1 = -_INFRA_HALF_WIDTH, _INFRA_HALF_WIDTH
    top = 11.0
    notes_top = -0.55
    if organization == "fragmented":
        _heading(items, x0, top, "infrastructure", "Fragmented experimental research")
        lane_h, step_y = 1.85, 2.0
        xs, xc, xa, lane_right = -6.6, -0.9, 4.8, 7.6
        boxes: Dict[str, Box] = {}
        for i, (lane, source, step, activity, copy) in enumerate(INFRA_LANES):
            y = 9.3 - i * step_y
            _silo(items, x0, lane_right, y - 0.5 * lane_h, y + 0.5 * lane_h, f"lane:{lane}")
            s = box(xs, y + 0.1, 4.3, 0.95, names[source], style="concept source", role=f"source:{source}")
            c = box(xc, y + 0.1, 3.6, 0.95, step, style=SYMPTOM_STYLE, role=f"step:{lane}")
            a = box(xa, y + 0.1, 4.3, 0.95, activities[activity], role=f"activity:{activity}")
            boxes[source], boxes[f"step:{lane}"], boxes[activity] = s, c, a
            items += list(s.items) + list(c.items) + list(a.items)
            _horizontal(items, edges, s, c, f"source:{source}", f"step:{lane}", style=ADHOC_STYLE)
            _horizontal(items, edges, c, a, f"step:{lane}", f"activity:{activity}", style=ADHOC_STYLE)
            items.append(Label((xc, y - 0.42), _tex(copy) + _cite(cited(INFRA_SYMPTOM_TAGS["copy"])),
                               "concept annotation", anchor="north", role=f"copy:{lane}"))
        # what each column is; the ad-hoc steps carry their symptoms
        for x, text, role in ((xs, "sources", "column:sources"), (xa, "activities", "column:activities")):
            items.append(Label((x, 9.3 + 0.5 * lane_h + 0.08), text, "concept band label", anchor="south", role=role))
        items.append(Label((xc, 9.3 + 0.5 * lane_h + 0.08),
                           "private, ad-hoc steps" + _cite(cited(INFRA_SYMPTOM_TAGS["step"])), "concept band label",
                           anchor="south", role="tag:step"))
        tags["step"] = cited(INFRA_SYMPTOM_TAGS["step"])
        tags["copy"] = cited(INFRA_SYMPTOM_TAGS["copy"])
        # processed signals handed from the acquisition path to the reconstruction path
        _down_or_up(items, edges, boxes["diagnostic"], boxes["reconstruction"], "activity:diagnostic",
                    "activity:reconstruction")
        items[-1] = Arrow(items[-1].start, items[-1].end, ADHOC_STYLE, role=items[-1].role)
        # the reconstruction, copied by hand into three other paths
        rec = boxes["reconstruction"]
        bus = lane_right + 0.75
        right = rec.x + 0.5 * rec.width
        last = boxes[INFRA_EQUILIBRIUM_COPIES[-1]]
        items.append(Polyline.of([(right + 0.08, rec.y), (bus, rec.y), (bus, last.y)], "connector line,driftred!75,dashed",
                                 role="bus:equilibrium"))
        for key in INFRA_EQUILIBRIUM_COPIES:
            b = boxes[key]
            items.append(Arrow((bus, b.y), (b.x + 0.5 * b.width + 0.08, b.y), ADHOC_STYLE,
                               role=f"edge:activity:reconstruction->activity:{key}"))
            _record(edges, "activity:reconstruction", f"activity:{key}", "forward")
        tags["equilibrium_copies"] = cited(INFRA_SYMPTOM_TAGS["equilibrium_copies"])
        items.append(Label((bus + 0.15, 0.5 * (rec.y + last.y)), "equilibrium files copied by hand"
                           + _cite(tags["equilibrium_copies"]), "concept annotation,rotate=-90", anchor="south",
                           role="bus:equilibrium"))
        _footnotes(items, labels, INFRA_CORRESPONDENCE, "symptom", x0, x1, notes_top, columns=2,
                   note="Each path works locally; the research process as a whole is neither shared nor reproducible")
        model = {"lanes": tuple(k for k, *_ in INFRA_LANES), "sources": tuple(s for _, s, *_ in INFRA_LANES),
                 "activities": tuple(a for _, _, _, a, _ in INFRA_LANES), "hub": None}
    else:
        _heading(items, x0, top, "infrastructure", "Integrated research infrastructure")
        n = len(INFRA_SOURCES)
        sw = (2 * _INFRA_HALF_WIDTH - (n - 1) * 0.25) / n
        sources_y, collector = 9.45, 8.55
        xs = [x0 + 0.5 * sw + i * (sw + 0.25) for i in range(n)]
        for (key, text), x in zip(INFRA_SOURCES, xs):
            b = box(x, sources_y, sw, 1.35, text, style="concept source", role=f"source:{key}")
            items += list(b.items)
            items.append(Polyline.of([(x, sources_y - 0.675), (x, collector)], "connector line",
                                     role=f"edge:source:{key}->sources"))
            _record(edges, f"source:{key}", "sources", "forward")
        items.append(Polyline.of([(xs[0], collector), (xs[-1], collector)], "connector line", role="sources"))
        frame_top, frame_bottom = 7.75, 2.55
        items += band(x0, x1, frame_bottom, frame_top, role="infrastructure")
        items.append(Label((0.0, frame_top - 0.12), "\\textbf{Integrated research infrastructure} $=$ Common Data "
                           "Model $+$ FAIR Scientific Data Repository $+$ Research Framework", "concept plain",
                           anchor="north", role="infrastructure"))
        caps: Dict[str, Box] = {}
        cw, cgap = 5.9, 1.0
        for i, (key, title, role, provides) in enumerate(INFRA_CAPABILITIES):
            x = (i - 1) * (cw + cgap)
            body = (_titled(title) + "\\\\{\\small\\itshape " + _tex(role) + "}\\\\[4pt]{\\small "
                    + "\\\\".join(map(_tex, provides)) + "}")
            style = "concept hub" if key == "cdm" else "concept strong" if key == "framework" else "concept pillar"
            caps[key] = box(x, 4.65, cw, 3.1, body, style=style, role=f"capability:{key}", latex=True)
            items += list(caps[key].items)
            _tag(items, tags, caps[key], key, cited(INFRA_RESPONSE_TAGS[key]))
        _horizontal(items, edges, caps["repository"], caps["cdm"], "capability:repository", "capability:cdm",
                    style="connector both", both=True)
        _horizontal(items, edges, caps["framework"], caps["cdm"], "capability:framework", "capability:cdm",
                    style="connector both", both=True)
        fw = caps["framework"]
        # ingestion enters the infrastructure as a whole, above its title
        items.append(Arrow((fw.x, collector), (fw.x, frame_top + 0.08), "connector", role="edge:sources->infrastructure"))
        _record(edges, "sources", "infrastructure", "forward")
        items.append(Label((fw.x - 0.15, 0.5 * (collector + frame_top)), "ingestion", "concept annotation",
                           anchor="east", role="ingestion"))
        # activities reach the infrastructure through one shared connection, never pairwise
        n = len(INFRA_ACTIVITIES)
        aw = (2 * _INFRA_HALF_WIDTH - (n - 1) * 0.3) / n
        act_y, bar = 0.65, 1.5
        axs = [x0 + 0.5 * aw + i * (aw + 0.3) for i in range(n)]
        for (key, text), x in zip(INFRA_ACTIVITIES, axs):
            b = box(x, act_y, aw, 0.95, text, role=f"activity:{key}")
            items += list(b.items)
            items.append(Polyline.of([(x, act_y + 0.475), (x, bar)], "connector line",
                                     role=f"edge:activity:{key}->activities"))
            _record(edges, f"activity:{key}", "activities", "forward")
        items.append(Polyline.of([(axs[0], bar), (axs[-1], bar)], "connector line", role="activities"))
        items.append(Arrow((0.0, bar), (0.0, frame_bottom - 0.08), "connector both", role="edge:activities->infrastructure",
                           both=True))
        _record(edges, "activities", "infrastructure", "both")
        _footnotes(items, labels, INFRA_CORRESPONDENCE, "response", x0, x1, notes_top, columns=3,
                   note="A Common Data Model gives interoperability, not persistence, provenance or reproducible "
                   "execution: all three capabilities are needed")
        model = {"sources": tuple(k for k, _ in INFRA_SOURCES), "activities": tuple(k for k, _ in INFRA_ACTIVITIES),
                 "capabilities": tuple(k for k, *_ in INFRA_CAPABILITIES), "hub": "infrastructure"}
    model.update({"organization": organization, "level": "infrastructure", "edges": tuple(edges), "tags": tags,
                  "correspondence": _correspondence(INFRA_CORRESPONDENCE)})
    return Diagram("experimental_research_infrastructure", Scene(tuple(items)), model=model)


# --------------------------------------------------------------------------------------------------------------
# Level 3: scientific credibility and traceability (#1638)
# --------------------------------------------------------------------------------------------------------------

#: the inference stages of the fragmented figure: (key, title)
CREDIBILITY_STAGES: Tuple[Tuple[str, str], ...] = (
    ("measurement", "Measurement"), ("processing", "Processing"), ("inference", "Reconstruction / Inference"),
    ("modelling", "Modelling / Simulation"), ("comparison", "Comparison & Interpretation"),
)
#: what the end of the chain looks like when nothing connects the evidence
APPARENT_SUCCESS: Tuple[str, ...] = ("small error bars", "good fit", "converged solver", "model ≈ experiment")
#: symptom -> response, one-to-one, numbered step by step as the fragmented figure cites them:
#: (key, plain symptom, technical term, reference, response)
CREDIBILITY_CORRESPONDENCE: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("inconsistent", "Measurements disagree", "data inconsistency", "Fischer & Dinklage 2004",
     "Consistency checks, joint inference"),
    ("unknown_source", "Unknown inputs or settings", "provenance loss", "W3C PROV 2013",
     "Provenance & versioned configuration"),
    ("lost_uncertainty", "Error bars dropped on the way", "complex error propagation", "Fischer & Dinklage 2004",
     "Uncertainty propagated with the data"),
    ("hidden_dependencies", "Unseen links between inputs", "diagnostic interdependencies", "Fischer & Dinklage 2004",
     "Explicit dependency graph"),
    ("coupled", "Parameters trade off against each other", "parametric entanglement", "Fischer & Dinklage 2004",
     "Complementary diagnostics, identifiability"),
    ("unstable", "Small changes flip the answer", "ill-posed inversion", "", "Regularization & identifiability evidence"),
    ("numerics", "Solver accuracy never checked", "code verification", "Greenwald 2010",
     "Verification & convergence studies"),
    ("model_range", "Model used beyond its validity", "domain of applicability", "Terry et al. 2008",
     "Applicability audit"),
    ("training_range", "Surrogate used beyond its data", "training domain", "", "Training-domain audit"),
    ("sensitivity", "Result hinges on unseen choices", "sensitivity", "Terry et al. 2008", "Sensitivity analysis"),
    ("agreement", "Right answer for the wrong reason", "fortuitous agreement", "Terry et al. 2008",
     "Discriminating validation evidence"),
    ("derived", "Inferred value treated as measured", "primacy hierarchy", "Terry et al. 2008",
     "Explicit inference chain"),
)
#: the stage each symptom arises at (fragmented)
CREDIBILITY_STAGE_OF: Dict[str, str] = {
    "inconsistent": "measurement", "unknown_source": "measurement", "lost_uncertainty": "processing",
    "hidden_dependencies": "processing", "coupled": "inference", "unstable": "inference", "numerics": "modelling",
    "model_range": "modelling", "training_range": "modelling", "sensitivity": "modelling", "agreement": "comparison",
    "derived": "comparison",
}
#: the evidence dimensions of the integrated figure: (key, title, what it holds)
CREDIBILITY_EVIDENCE: Tuple[Tuple[str, str, str], ...] = (
    ("provenance", "Provenance & lineage", "how was it produced? (not: is it valid?)"),
    ("uncertainty", "Uncertainty & covariance", "propagated through every operation"),
    ("assumptions", "Assumptions & applicability", "physics model and training domain, checked apart"),
)
CREDIBILITY_CHECKS: Tuple[Tuple[str, str, str], ...] = (
    ("verification", "Verification & numerics", "is the computation adequate? (not: does the model apply?)"),
    ("validation", "Validation & comparison", "independent, discriminating tests"),
    ("sensitivity", "Sensitivity & identifiability", "what the result depends on"),
)
#: an example assessment: several dimensions, never one score
ASSESSMENT_PROFILE: Tuple[Tuple[str, str], ...] = (
    ("data validity", "supported"), ("numerics", "supported"), ("physical applicability", "marginal"),
    ("uncertainty", "large"), ("identifiability", "weak"), ("provenance", "complete"),
)
#: which integrated element answers which symptom
CREDIBILITY_RESPONSE_TAGS: Dict[str, Tuple[str, ...]] = {
    "provenance": ("unknown_source", "hidden_dependencies"), "uncertainty": ("lost_uncertainty",),
    "assumptions": ("model_range", "training_range"), "verification": ("numerics",),
    "validation": ("inconsistent", "agreement", "derived"), "sensitivity": ("coupled", "unstable", "sensitivity"),
}

_CRED_HALF_WIDTH = 10.5


def scientific_credibility(organization: str = "integrated", *, labels: bool = True) -> Diagram:
    r"""Scientific credibility, fragmented and qualified: why reproducibility is not enough (#1638).

    ``organization`` selects one figure of the pair:

    ``"fragmented"``
        a chain of locally reasonable steps -- measurement, processing,
        reconstruction or inference, modelling or simulation, comparison and
        interpretation -- ending in an apparently precise result (small error
        bars, a good fit, a converged solver, model and experiment agreeing).
        Beneath each step, the evidence it loses, with a plain label and the
        literature term where one exists: inconsistent measurements (data
        inconsistency), unknown source or configuration (provenance loss),
        lost uncertainty (complex error propagation), hidden dependencies,
        coupled parameters (parametric entanglement), unstable inference
        (ill-posed inversion), unverified numerics, outside the model's range
        (domain of applicability), outside the training data,
        hidden sensitivity, accidental agreement (fortuitous agreement) and
        derived quantities taken as direct (primacy hierarchy);
    ``"integrated"``
        a scientific state carries three kinds of evidence -- provenance and
        lineage, uncertainty and covariance, assumptions and applicability --
        into a common credibility and traceability structure, which is
        examined by verification, validation and sensitivity or
        identifiability analysis. The assessment is a profile over several
        dimensions, not one score, and yields a qualified scientific state.
        Provenance is kept apart from validity, numerical verification from
        physical applicability, and physics-model applicability from the
        training domain of a surrogate.

    The fragmented figure's notes cite the terminology sources compactly
    (full references are in the documentation). The principles behind it
    are ``scientific_infrastructure_principles``; the provenance of one chain
    is ``scientific_provenance_chain``.
    """
    organization = _check_organization(organization)
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    tags: Dict[str, Tuple[int, ...]] = {}
    cited = partial(_numbers, correspondence=CREDIBILITY_CORRESPONDENCE)
    x0, x1 = -_CRED_HALF_WIDTH, _CRED_HALF_WIDTH
    top = 10.4
    numbers = {key: i + 1 for i, (key, *_) in enumerate(CREDIBILITY_CORRESPONDENCE)}
    if organization == "fragmented":
        _heading(items, x0, top, "credibility", "Unqualified scientific inference")
        n = len(CREDIBILITY_STAGES) + 1
        w, gap = 2.85, 0.78
        xs = [x0 + 0.5 * w + i * (w + gap) for i in range(n)]
        stage_y = 9.0
        stages: List[Box] = []
        for (key, title), x in zip(CREDIBILITY_STAGES, xs):
            b = box(x, stage_y, w, 1.05, "\\textbf{" + _tex(title) + "}", role=f"stage:{key}", latex=True)
            stages.append(b)
            items += list(b.items)
        result = box(xs[-1], stage_y - 1.55, w, 4.15, "\\textbf{Apparently precise result}\\\\[5pt]{\\small "
                     + "\\\\".join(map(_tex, APPARENT_SUCCESS)) + "}", style="concept strong", role="result",
                     latex=True)
        items += list(result.items)
        keys = [k for k, _ in CREDIBILITY_STAGES]
        for k0, k1, a, b in zip(keys, keys[1:], stages, stages[1:]):
            _horizontal(items, edges, a, b, f"stage:{k0}", f"stage:{k1}")
        _horizontal(items, edges, stages[-1], Box(result.x, stage_y, w, 1.05, ()), f"stage:{keys[-1]}", "result")
        # what each step loses, beneath it
        chip_h, chip_gap = 1.35, 0.25
        lowest = stage_y
        for key, b in zip(keys, stages):
            below = [(c, s) for c, s, *_ in CREDIBILITY_CORRESPONDENCE if CREDIBILITY_STAGE_OF[c] == key]
            for j, (c, symptom) in enumerate(below):
                y = stage_y - 1.05 - 0.5 * chip_h - j * (chip_h + chip_gap)
                tags[c] = (numbers[c],)  # each symptom is cited where it arises, on its own chip
                chip = box(b.x, y, w, chip_h, _chip_text(numbers[c], symptom), style=SYMPTOM_STYLE,
                           role=f"symptom:{c}", latex=True)
                items += list(chip.items)
                lowest = min(lowest, y - 0.5 * chip_h)
                items.append(Polyline.of([(b.x, (y + 0.5 * chip_h + chip_gap) if j else b.y - 0.525),
                                          (b.x, y + 0.5 * chip_h)], "concept tick", role=f"loses:{c}"))
        items.append(Label((x0, stage_y + 0.6), "each step is locally reasonable; beneath it, the evidence it loses",
                           "concept band label", anchor="south west", role="losses"))
        footer_top = min(lowest, result.y - 0.5 * result.height) - 0.45
        _footnotes(items, labels, CREDIBILITY_CORRESPONDENCE, "symptom", x0, x1, footer_top, columns=2,
                   note="Reasonable local steps can still give a weak global inference when assumptions, "
                   "uncertainty, dependencies and evidence are not connected")
        model = {"stages": tuple(keys), "stage_of": dict(CREDIBILITY_STAGE_OF), "result": APPARENT_SUCCESS}
    else:
        _heading(items, x0, top, "credibility", "Qualified scientific state")
        span3 = 3 * 6.2 + 2 * 0.8
        state = box(0.0, 9.0, span3, 0.85, "\\textbf{Scientific state or result}", style="concept state",
                    role="node:state", latex=True)
        items += list(state.items)
        w3, gap3 = 6.2, 0.8
        xs = [(i - 1) * (w3 + gap3) for i in range(3)]
        nodes: Dict[str, Box] = {"state": state}
        for (key, title, sub), x in zip(CREDIBILITY_EVIDENCE, xs):
            nodes[key] = box(x, 7.1, w3, 1.35, _titled(title, sub), role=f"evidence:{key}", latex=True)
        nodes["model"] = box(0.0, 5.1, 2 * w3 + 2 * gap3 + w3, 1.0, "\\textbf{Common scientific credibility \\& "
                             "traceability structure}: assumptions $\\rightarrow$ evidence $\\rightarrow$ uncertainty "
                             "$\\rightarrow$ V\\&V $\\rightarrow$ applicability", style="concept hub",
                             role="node:model", latex=True)
        for (key, title, sub), x in zip(CREDIBILITY_CHECKS, xs):
            nodes[key] = box(x, 3.1, w3, 1.35, _titled(title, sub), style="concept vv", role=f"check:{key}",
                             latex=True)
        profile = " $\\cdot$ ".join(_tex(d) + ": \\textit{" + _tex(s) + "}"
                                    for d, s in ASSESSMENT_PROFILE)
        nodes["assessment"] = box(0.0, 1.0, 2 * w3 + 2 * gap3 + w3, 1.3, "\\textbf{Evidence-based assessment}: a "
                                  "profile, not one score\\\\{\\small for example: " + profile + "}", style="concept leaf",
                                  role="node:assessment", latex=True)
        nodes["qualified"] = box(0.0, -0.9, 7.5, 0.85, "\\textbf{Qualified scientific state}", style="concept strong",
                                 role="node:qualified", latex=True)
        for b in list(nodes.values())[1:]:
            items += list(b.items)
        for key, *_ in CREDIBILITY_EVIDENCE:
            b = nodes[key]
            _down_or_up(items, edges, state, b, "state", key, at_x=b.x)
            _down_or_up(items, edges, b, nodes["model"], key, "model", at_x=b.x)
        for key, *_ in CREDIBILITY_CHECKS:
            _down_or_up(items, edges, nodes["model"], nodes[key], "model", key, at_x=nodes[key].x)
            _down_or_up(items, edges, nodes[key], nodes["assessment"], key, "assessment", at_x=nodes[key].x)
        _down_or_up(items, edges, nodes["assessment"], nodes["qualified"], "assessment", "qualified")
        for key, keys in CREDIBILITY_RESPONSE_TAGS.items():
            _tag(items, tags, nodes[key], key, cited(keys))
        _footnotes(items, labels, CREDIBILITY_CORRESPONDENCE, "response", x0, x1, -1.7, columns=3,
                   note="Trust comes from a traceable chain of assumptions, evidence, uncertainty, V&V and "
                   "applicability, not from storage, reruns or execution alone")
        model = {"nodes": tuple(nodes), "evidence": tuple(k for k, *_ in CREDIBILITY_EVIDENCE),
                 "checks": tuple(k for k, *_ in CREDIBILITY_CHECKS), "assessment": ASSESSMENT_PROFILE}
    model.update({"organization": organization, "level": "credibility", "edges": tuple(edges), "tags": tags,
                  "correspondence": _correspondence(CREDIBILITY_CORRESPONDENCE)})
    return Diagram("scientific_credibility", Scene(tuple(items)), model=model)


# --------------------------------------------------------------------------------------------------------------
# Level 4: research modality, portability and software architecture (#1640)
# --------------------------------------------------------------------------------------------------------------

#: the locked-in research paths, the row of each entry aligned across silos: (key, actor, medium, entries)
MODALITY_SILOS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("gui", "GUI application", "own data handling", ("GUI-specific data access", "custom processing",
                                                    "reimplemented formulas")),
    ("notebook", "Researcher A", "Jupyter notebook", ("data loading", "processing", "physics formulas", "plotting")),
    ("matlab", "Researcher B", "MATLAB workflow", ("separate data loading", "separate calibration",
                                                    "diverged formulas", "one machine's conventions")),
    ("hpc", "Researcher C", "HPC shell scripts", ("cluster-specific paths", "native solver run",
                                                   "custom post-processing")),
    ("agent", "AI / automation", "no capability interface", ("one opaque all-in-one script",)),
)
#: the formulas copied out of the notebook: (from silo, to silo), on the formula row
MODALITY_COPIES: Tuple[Tuple[str, str], ...] = (("notebook", "gui"), ("notebook", "matlab"))
#: symptom -> response, one-to-one, numbered left to right as the fragmented figure cites them:
#: (key, plain symptom, technical term, reference, response)
MODALITY_CORRESPONDENCE: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("duplicated", "Same physics coded twice", "code duplication", "", "One scientific source of truth"),
    ("interface", "Science reachable from one tool only", "interface lock-in", "", "Sibling interfaces on shared APIs"),
    ("monolithic", "One script does everything", "monolithic workflow", "", "Modular scientific capabilities"),
    ("knowledge", "Know-how hidden in personal scripts", "tacit knowledge", "", "Public modules, docs & catalogs"),
    ("unversioned", "Changes untracked and untested", "no version control", "", "Issues, reviews, tests, CI, releases"),
    ("language", "Tied to one programming language", "language lock-in", "", "Stable contracts, language adapters"),
    ("machine", "Written for one device", "machine-specific code", "", "Machine mapping, common state"),
    ("environment", "Runs on one machine or cluster only", "environment lock-in", "", "Portable execution, local to HPC"),
    ("human_only", "AI tools cannot use it", "human-only interface", "", "Human and agent interfaces"),
)
MODALITY_SYMPTOM_TAGS: Dict[str, Tuple[str, ...]] = {
    "gui": ("duplicated", "interface"), "notebook": ("monolithic", "knowledge", "unversioned"),
    "matlab": ("duplicated", "language", "machine"), "hpc": ("language", "environment"), "agent": ("human_only",)}
#: research interfaces, siblings over the same public APIs: (key, name, what it is for, agent-oriented)
MODALITY_INTERFACES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("python", "Python", "composition", False), ("jupyter", "Jupyter", "exploration", False),
    ("cli", "CLI", "automation", False), ("gui", "GUI", "interactive work", False),
    ("docs", "Documentation", "learning", False), ("mcp", "MCP / AI agent", "structured tools", True),
)
#: the modular scientific core: separate responsibilities with stable contracts between them (key, name, question)
MODALITY_CORE: Tuple[Tuple[str, str, str], ...] = (
    ("formula", "formula", "quantities & relations"), ("process", "process", "data transformations"),
    ("database", "database", "state access & persistence"), ("validation", "validation", "quality & validity"),
    ("code", "code", "external solvers"), ("plot", "plot", "visualization"),
    ("diagram", "diagram", "concept explanation"), ("machine_mapping", "machine mapping", "device representations"),
)
#: implementation runtimes: current mechanisms, each solving a different problem, and future bindings
MODALITY_RUNTIMES: Tuple[Tuple[str, str], ...] = (
    ("python", "Python modules: the integration layer"),
    ("adapters", "native Fortran / C solvers kept, through adapters"),
    ("jit", "selective JIT kernels where profiling shows value"),
    ("jax", "optional JAX kernels: differentiable, batched"),
)
MODALITY_FUTURE: Tuple[str, ...] = ("MATLAB", "Julia")
MODALITY_PLATFORMS: Tuple[str, ...] = ("Windows", "macOS", "Linux")
MODALITY_EXECUTION: Tuple[str, ...] = ("local", "workstation / server", "HPC cluster", "scheduler (Slurm)")
MODALITY_LIFECYCLE: Tuple[str, ...] = ("Git", "issues", "pull requests", "tests", "CI", "documentation", "notebooks",
                                       "releases")
MODALITY_RESPONSE_TAGS: Dict[str, Tuple[str, ...]] = {
    "interfaces": ("interface", "human_only"), "core": ("duplicated", "monolithic", "knowledge"),
    "machine_mapping": ("machine",), "runtimes": ("language",), "execution": ("environment",),
    "lifecycle": ("unversioned",)}

_MODALITY_HALF_WIDTH = 10.0


def research_modality_architecture(organization: str = "integrated", *, labels: bool = True) -> Diagram:
    r"""Research software, locked in and modular: one scientific core, many ways to research (#1640).

    ``organization`` selects one figure of the pair:

    ``"fragmented"``
        five plausible research paths, each a silo that holds its own copy of
        the science: a GUI application with its own data handling and
        reimplemented formulas; a researcher's Jupyter notebook (data loading,
        processing, physics formulas, plotting); another researcher's MATLAB
        workflow with diverged formulas and one machine's conventions; HPC
        shell scripts with cluster-specific paths; and AI or automation with
        no capability interface, only an opaque script. The formulas are
        copied out of the notebook. No language, interface or environment is
        the problem -- the coupling of scientific semantics to one of them is;
    ``"integrated"``
        sibling research interfaces -- Python, Jupyter, CLI, GUI,
        documentation (human-oriented) and MCP for AI agents -- over the same
        public scientific APIs, above one modular scientific core of separate
        responsibilities (formula, process, database, validation, code, plot,
        diagram, machine mapping). Below the core, two different axes:
        language and runtime interoperability (Python as the integration
        layer, native solvers kept through adapters, selective JIT kernels,
        optional JAX kernels; MATLAB and Julia bindings marked as future) and
        portable execution (Windows, macOS, Linux; local to HPC and a
        scheduler -- external solvers keep their own platform limits). The
        versioned research lifecycle (Git, issues, pull requests, tests, CI,
        documentation, notebooks, releases) runs alongside as its own
        dimension.

    Standardize the science, not the scientist. The VEST implementation of
    the interfaces and platforms is ``vest_data_platform``.
    """
    organization = _check_organization(organization)
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    tags: Dict[str, Tuple[int, ...]] = {}
    cited = partial(_numbers, correspondence=MODALITY_CORRESPONDENCE)
    x0, x1 = -_MODALITY_HALF_WIDTH, _MODALITY_HALF_WIDTH
    top = 11.3
    notes_top = -1.0
    if organization == "fragmented":
        _heading(items, x0, top, "modality", "Locked-in research software")
        n = len(MODALITY_SILOS)
        gap = 0.55
        w = (x1 - x0 - (n - 1) * gap) / n
        silo_top, silo_bottom = 10.2, 2.75
        rows = [6.9 - 1.15 * i for i in range(4)]
        cells: Dict[Tuple[str, int], Box] = {}
        for i, (key, actor, medium, entries) in enumerate(MODALITY_SILOS):
            x = x0 + 0.5 * w + i * (w + gap)
            _silo(items, x - 0.5 * w, x + 0.5 * w, silo_bottom, silo_top, f"silo:{key}")
            head = box(x, 8.6, w - 0.4, 1.8, _titled(actor, medium), style="concept actor", role=f"actor:{key}",
                       latex=True)
            items += list(head.items)
            _tag(items, tags, head, key, cited(MODALITY_SYMPTOM_TAGS[key]))
            for j, text in enumerate(entries):
                b = box(x, rows[j], w - 0.75, 0.85, text, style="concept leaf", role=f"entry:{key}:{j}")
                cells[(key, j)] = b
                items += list(b.items)
        formula_row = 2
        for src, dst in MODALITY_COPIES:
            _horizontal(items, edges, cells[(src, formula_row)], cells[(dst, formula_row)], f"entry:{src}",
                        f"entry:{dst}", style=ADHOC_STYLE)
        items.append(Label((0.0, silo_bottom - 0.1), "the physics formulas, copied out of one notebook, reimplemented "
                           "in the GUI and diverged in MATLAB", "concept annotation", anchor="north",
                           role="copies"))
        _footnotes(items, labels, MODALITY_CORRESPONDENCE, "symptom", x0, x1, 1.9, columns=2,
                   note="No language, interface or environment is the problem: tying the science to one of them is")
        model = {"silos": tuple(k for k, *_ in MODALITY_SILOS), "copies": MODALITY_COPIES, "core": None}
    else:
        _heading(items, x0, top, "modality", "One modular scientific core, many ways to research")
        lifecycle_w = 2.4
        x1c = x1 - lifecycle_w - 0.3  # the main column ends here; the lifecycle runs beside it
        span = x1c - x0
        cx = 0.5 * (x0 + x1c)
        n = len(MODALITY_INTERFACES)
        iw = (span - (n - 1) * 0.15) / n
        iy, api_y = 9.75, 8.25
        ixs = [x0 + 0.5 * iw + i * (iw + 0.15) for i in range(n)]
        interface_boxes: List[Box] = []
        for (key, name, use, agent), x in zip(MODALITY_INTERFACES, ixs):
            b = box(x, iy, iw, 1.25, "{\\small\\bfseries " + _tex(name) + "}\\\\{\\footnotesize " + _tex(use) + "}",
                    style="concept actor" if agent else "concept box", role=f"interface:{key}", latex=True)
            interface_boxes.append(b)
            items += list(b.items)
        humans = ixs[:-1]
        for xa, xb, text, role in ((humans[0] - 0.5 * iw, humans[-1] + 0.5 * iw, "human-oriented", "audience:human"),
                                   (ixs[-1] - 0.5 * iw, ixs[-1] + 0.5 * iw, "agent-oriented", "audience:agent")):
            items += [Polyline.of([(xa, iy + 0.72), (xa, iy + 0.85), (xb, iy + 0.85), (xb, iy + 0.72)], "concept tick",
                                  role=role),
                      Label((0.5 * (xa + xb), iy + 0.9), text, "concept annotation", anchor="south", role=role)]
        api = box(cx, api_y, span, 0.75, "\\textbf{Public scientific APIs} $\\cdot$ one contract for every interface",
                  style="concept state", role="node:api", latex=True)
        items += list(api.items)
        _tag(items, tags, api, "interfaces", cited(MODALITY_RESPONSE_TAGS["interfaces"]))
        for (key, *_), b in zip(MODALITY_INTERFACES, interface_boxes):
            items.append(Polyline.of([(b.x, b.y - 0.625), (b.x, api_y + 0.375)], "connector line",
                                     role=f"edge:interface:{key}->api"))
            _record(edges, f"interface:{key}", "api", "forward")
        core_top, core_bottom = 6.75, 3.85
        core = Box(cx, 0.5 * (core_top + core_bottom), span, core_top - core_bottom, ())
        items.append(Polyline.of([(x0, core_bottom), (x1c, core_bottom), (x1c, core_top), (x0, core_top)],
                                 "concept strong", role="node:core", closed=True))
        tags["core"] = cited(MODALITY_RESPONSE_TAGS["core"])
        items.append(Label((cx, core_top - 0.12), "\\textbf{One modular scientific core}" + _cite(tags["core"])
                           + ": separate responsibilities, stable contracts between them", "concept plain",
                           anchor="north", role="node:core"))
        mw = (span - 0.5 - 3 * 0.22) / 4
        for i, (key, name, question) in enumerate(MODALITY_CORE):
            row, col = divmod(i, 4)
            x = x0 + 0.25 + 0.5 * mw + col * (mw + 0.22)
            y = 5.6 - row * 1.0
            title = "\\textbf{" + _tex(name) + "}"
            if key in MODALITY_RESPONSE_TAGS:
                title += _cite(cited(MODALITY_RESPONSE_TAGS[key]))
                tags[key] = cited(MODALITY_RESPONSE_TAGS[key])
            b = box(x, y, mw, 0.92, title + "\\\\{\\footnotesize " + _tex(question) + "}", style="concept leaf",
                    role=f"module:{key}", latex=True)
            items += list(b.items)
        _down_or_up(items, edges, api, core, "api", "core", both=True)
        # two different axes beneath the core
        hw = 0.5 * (span - 0.5)
        rx, ex = x0 + 0.5 * hw, x1c - 0.5 * hw
        runtime_text = (_titled("Language & runtime interoperability") + "\\\\[2pt]{\\small "
                        + "\\\\".join(_tex(t) for _, t in MODALITY_RUNTIMES) + "}")
        runtimes = box(rx, 1.7, hw, 2.35, runtime_text, role="node:runtimes", latex=True)
        execution_text = (_titled("Portable execution") + "\\\\[2pt]{\\small " + " $\\cdot$ ".join(MODALITY_PLATFORMS)
                          + "\\\\" + " $\\cdot$ ".join(map(_tex, MODALITY_EXECUTION)) + "\\\\[2pt]\\itshape external "
                          "solvers keep their own platform limits}")
        execution = box(ex, 1.7, hw, 2.35, execution_text, role="node:execution", latex=True)
        items += list(runtimes.items) + list(execution.items)
        _tag(items, tags, runtimes, "runtimes", cited(MODALITY_RESPONSE_TAGS["runtimes"]))
        _tag(items, tags, execution, "execution", cited(MODALITY_RESPONSE_TAGS["execution"]))
        for b, key in ((runtimes, "runtimes"), (execution, "execution")):
            _down_or_up(items, edges, Box(b.x, core.y, b.width, core.height, ()), b, "core", key, both=True)
        future = box(rx, -0.1, hw, 0.7, "{\\small\\itshape future: " + ", ".join(MODALITY_FUTURE) + " bindings on the "
                     "same contracts}", style="concept group", role="node:future", latex=True)
        items += list(future.items)
        # the versioned lifecycle: a dimension of its own, beside every layer
        lx = x1 - 0.5 * lifecycle_w
        lifecycle = box(lx, 0.5 * (iy + 0.575), lifecycle_w, iy + 0.775,
                        "\\textbf{Versioned research lifecycle}\\\\[6pt]{\\small " + "\\\\[2pt]".join(
                            map(_tex, MODALITY_LIFECYCLE)) + "}", style="concept pillar", role="node:lifecycle",
                        latex=True)
        items += list(lifecycle.items)
        _tag(items, tags, lifecycle, "lifecycle", cited(MODALITY_RESPONSE_TAGS["lifecycle"]))
        _footnotes(items, labels, MODALITY_CORRESPONDENCE, "response", x0, x1, notes_top, columns=3,
                   note="Standardize the science, not the scientist: interfaces and runtimes change, the scientific "
                   "contract stays")
        model = {"interfaces": tuple(k for k, *_ in MODALITY_INTERFACES),
                 "agent_interfaces": tuple(k for k, *_, agent in MODALITY_INTERFACES if agent),
                 "core": tuple(k for k, *_ in MODALITY_CORE), "runtimes": tuple(k for k, _ in MODALITY_RUNTIMES),
                 "future": MODALITY_FUTURE, "platforms": MODALITY_PLATFORMS, "execution": MODALITY_EXECUTION,
                 "lifecycle": MODALITY_LIFECYCLE}
    model.update({"organization": organization, "level": "modality", "edges": tuple(edges), "tags": tags,
                  "correspondence": _correspondence(MODALITY_CORRESPONDENCE)})
    return Diagram("research_modality_architecture", Scene(tuple(items)), model=model)


# --------------------------------------------------------------------------------------------------------------
# Level 1: multi-layer scientific representation and semantic access (#1641)
# --------------------------------------------------------------------------------------------------------------

#: the four representation layers, top to bottom: (key, row title, what the row holds)
REPRESENTATION_LAYERS: Tuple[Tuple[str, str, str], ...] = (
    ("vocabulary", "spoken term", "what people call it"), ("naming", "stored name", "the identifier in the data"),
    ("structure", "structure", "how it is organized"), ("encoding", "encoding", "how it is saved"),
)
#: one quantity -- the plasma current -- in five native representations:
#: (key, holder, spoken term, (stored identifier, what kind of identifier), structure, encoding)
REPRESENTATION_SILOS: Tuple[Tuple[str, str, str, Tuple[str, str], str, str], ...] = (
    ("machine_a", "Machine A", "“Ip”", ("IPR01", "DAQ channel"), "time × channel array", "binary files"),
    ("machine_b", "Machine B", "“plasma current”", ("ipla", "NetCDF variable"), "hierarchical tree", "NetCDF"),
    ("solver_c", "Solver C", "“I_p”", ("CUR", "output key"), "code-specific records", "solver-native output"),
    ("researcher_d", "Researcher D", "“current”", ("d.ip", "struct field"), "personal structures", "MATLAB files"),
    ("database_e", "Database E", "“measured Ip”", ("ip_meas", "dataset"), "storage groups", "HDF5"),
)
#: symptom -> response, one-to-one, numbered top to bottom as the fragmented figure cites them:
#: (key, plain symptom, technical term, reference, response)
REPRESENTATION_CORRESPONDENCE: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("terms", "People use different words for it", "terminology fragmentation", "", "Taxonomy & strict aliases"),
    ("language", "Plain questions cannot find the data", "no semantic access", "", "Intent → taxonomy → concept"),
    ("names", "Each source names it differently", "semantic fragmentation", "", "Canonical scientific semantics"),
    ("paths", "You must know where it is stored", "path-based access", "", "Concept-based discovery"),
    ("structures", "Same data, different layouts", "structural fragmentation", "", "Common structural representation"),
    ("formats", "Saved in many file formats", "format fragmentation", "", "Adapters at import and export"),
    ("full_files", "Whole files loaded for one value", "eager loading", "", "Lazy, partial, cached access"),
    ("storage_leak", "Analysis code tied to the file layout", "storage coupling", "", "Storage behind an access layer"),
)
#: where the symptoms arise (fragmented): a layer's row label, or the database's stored file
REPRESENTATION_SYMPTOM_TAGS: Dict[str, Tuple[str, ...]] = {
    "vocabulary": ("terms", "language"), "naming": ("names", "paths"), "structure": ("structures",),
    "encoding": ("formats",), "storage": ("full_files", "storage_leak")}
#: the integrated stack, top to bottom: (key, title, content, question)
REPRESENTATION_STACK: Tuple[Tuple[str, str, str, str], ...] = (
    ("vocabulary", "Researchers, interfaces & agents", "“Ip” · “plasma current” · “I_p” · “show me the q profile”",
     "what does the researcher call it?"),
    ("semantic", "Semantic access: taxonomy · strict aliases · discovery",
     "aliases name one quantity; a family groups several", "which concept is meant?"),
    ("cdm", "Common Data Model (IMAS)", "canonical scientific semantics and structure", "what does it mean?"),
    ("mapping", "Format & structure mappings", "adapters: binary · NetCDF · HDF5 · solver-native · arrays · trees",
     "how is it encoded and organized?"),
    ("native", "Native representations", "machines · solvers · researchers · databases", ""),
)
REPRESENTATION_STORAGE: Tuple[str, ...] = ("local files · remote store · cache", "eager · lazy · partial · cached")
REPRESENTATION_RESPONSE_TAGS: Dict[str, Tuple[str, ...]] = {
    "vocabulary": ("language",), "semantic": ("terms", "paths"), "cdm": ("names",), "mapping": ("formats", "structures"),
    "storage": ("full_files", "storage_leak")}
#: implementation names this figure may draw although IMPLEMENTATION_PATTERNS forbids them elsewhere: HDF5,
#: shown as one serialization format (beside NetCDF and native files) to set it apart from the scientific model
REPRESENTATION_FORMAT_EXAMPLES: Tuple[str, ...] = ("HDF5",)

_REPRESENTATION_HALF_WIDTH = 10.0


def scientific_representation(organization: str = "integrated", *, labels: bool = True) -> Diagram:
    r"""Scientific representation, fragmented and layered: one meaning, many representations and names (#1641).

    ``organization`` selects one figure of the pair:

    ``"fragmented"``
        one quantity, the plasma current, held five ways -- by two machines,
        a solver, a researcher and a database -- and in each at four distinct
        layers: what the researcher says ("Ip", "plasma current", "I_p"),
        what the data calls it, how it is organized (a time-by-channel
        array, a hierarchical tree, code records) and how it is encoded
        (binary, NetCDF, solver-native output, HDF5). The layers fragment
        separately, and neighbours are mapped pairwise by hand;
    ``"integrated"``
        the layers are kept apart and connected: format and structure
        mappings bring native representations into the Common Data Model
        (IMAS), which owns canonical scientific semantics and structure; a
        semantic-access layer of taxonomy, strict aliases and discovery
        connects researchers, interfaces and agents to canonical concepts
        (an alias names one quantity, a family groups several); storage and
        access -- local or remote, eager, lazy, partial, cached -- sit beside
        the model, orthogonal to what the data means.

    HDF5 is drawn only as a serialization format, to show it is not the
    scientific model; no storage service is named. The data-level argument
    for a common model is ``experiment_modeling_theory_data_network``.
    """
    organization = _check_organization(organization)
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    tags: Dict[str, Tuple[int, ...]] = {}
    cited = partial(_numbers, correspondence=REPRESENTATION_CORRESPONDENCE)
    x0, x1 = -_REPRESENTATION_HALF_WIDTH, _REPRESENTATION_HALF_WIDTH
    top = 11.1
    notes_top = -1.1
    if organization == "fragmented":
        _heading(items, x0, top, "representation", "Fragmented scientific representation")
        label_w = 2.9
        n = len(REPRESENTATION_SILOS)
        gap = 1.0
        w = (x1 - x0 - label_w - n * gap) / n
        rows = {key: 8.2 - 1.9 * i for i, (key, *_) in enumerate(REPRESENTATION_LAYERS)}
        for key, title, question in REPRESENTATION_LAYERS:
            y = rows[key]
            items += band(x0, x1, y - 0.75, y + 0.75, role=f"layer:{key}")
            items.append(Label((x0 + 0.15, y), "\\textbf{" + _tex(title) + "}"
                               + _cite(cited(REPRESENTATION_SYMPTOM_TAGS[key])) + "\\\\{\\footnotesize\\itshape "
                               + _tex(question) + "}", "concept plain,align=left", anchor="west", role=f"tag:{key}"))
            tags[key] = cited(REPRESENTATION_SYMPTOM_TAGS[key])
        cells: Dict[Tuple[str, str], Box] = {}
        for i, (key, holder, spoken, (stored, kind), structure, encoding) in enumerate(REPRESENTATION_SILOS):
            x = x0 + label_w + gap + 0.5 * w + i * (w + gap)
            pad = 0.32  # wide enough that a link's arrowhead lands inside the silo, not on its outline
            _silo(items, x - 0.5 * w + 0.05 - pad, x + 0.5 * w - 0.05 + pad, rows["encoding"] - 0.85,
                  rows["vocabulary"] + 2.0,
                  f"silo:{key}")
            head = box(x, rows["vocabulary"] + 1.15, w + 0.3, 0.85, "{\\small\\bfseries " + _tex(holder) + "}",
                       style="concept actor", role=f"holder:{key}", latex=True)
            items += list(head.items)
            texts = {"vocabulary": "{\\small " + _tex(spoken) + "}",
                     "naming": "{\\small\\ttfamily " + _tex(stored) + "}\\\\{\\footnotesize " + _tex(kind) + "}",
                     "structure": "{\\small " + _tex(structure) + "}", "encoding": "{\\small " + _tex(encoding) + "}"}
            for layer, *_ in REPRESENTATION_LAYERS:
                b = box(x, rows[layer], w - 0.1, 1.2, texts[layer], style="concept leaf", role=f"cell:{key}:{layer}",
                        latex=True)
                cells[(key, layer)] = b
                items += list(b.items)
            if encoding in REPRESENTATION_FORMAT_EXAMPLES:  # the stored file is where storage leaks into the science
                _tag(items, tags, cells[(key, "encoding")], "storage",
                     cited(REPRESENTATION_SYMPTOM_TAGS["storage"]))
        keys = [k for k, *_ in REPRESENTATION_SILOS]
        for a, b in zip(keys, keys[1:]):
            _horizontal(items, edges, cells[(a, "naming")], cells[(b, "naming")], f"cell:{a}", f"cell:{b}",
                        style=ADHOC_STYLE, both=True)
        items.append(Label((0.5 * (x0 + label_w + x1), rows["encoding"] - 0.95), "neighbours mapped pairwise by hand "
                           "(red); every layer fragments separately", "concept annotation", anchor="north",
                           role="pairwise"))
        _footnotes(items, labels, REPRESENTATION_CORRESPONDENCE, "symptom", x0, x1, rows["encoding"] - 1.45, columns=2,
                   note="Not one file-format problem: words, names, structure and encoding are different layers")
        model = {"layers": tuple(k for k, *_ in REPRESENTATION_LAYERS), "silos": tuple(keys), "hub": None}
    else:
        _heading(items, x0, top, "representation", "One scientific meaning, many representations and names")
        sw = 12.0
        sx = x0 + 0.5 * sw
        stack: Dict[str, Box] = {}
        n = len(REPRESENTATION_STACK)
        for i, (key, title, content, question) in enumerate(REPRESENTATION_STACK):
            y = 9.6 - 2.4 * i
            body = _titled(title) + "\\\\{\\small " + _tex(content) + "}"
            if question:
                body += "\\\\{\\footnotesize\\itshape " + _tex(question) + "}"
            style = {"vocabulary": "concept actor", "cdm": "concept hub", "native": "concept source"}.get(key,
                                                                                                       "concept box")
            stack[key] = box(sx, y, sw, 1.4 if question else 1.0, body, style=style, role=f"layer:{key}", latex=True)
            items += list(stack[key].items)
            if key in REPRESENTATION_RESPONSE_TAGS:
                _tag(items, tags, stack[key], key, cited(REPRESENTATION_RESPONSE_TAGS[key]))
        keys = [k for k, *_ in REPRESENTATION_STACK]
        for a, b in zip(keys, keys[1:]):
            _down_or_up(items, edges, stack[a], stack[b], a, b, both=True)
        cdm = stack["cdm"]
        dw = x1 - (x0 + sw) - 2.1
        storage = database(x1 - 0.5 * dw, cdm.y, dw, 3.6, _titled("Storage & access") + "\\\\[3pt]{\\small "
                           + "\\\\".join(map(_tex, REPRESENTATION_STORAGE)) + "}\\\\{\\footnotesize\\itshape where is "
                           "it stored, how is it read?}", role="node:storage", latex=True, ry=0.45)
        items += list(storage.items)
        _tag(items, tags, storage, "storage", cited(REPRESENTATION_RESPONSE_TAGS["storage"]))
        _horizontal(items, edges, cdm, storage, "cdm", "storage", style="connector both", both=True)
        items.append(Label((0.5 * (cdm.x + 0.5 * cdm.width + storage.x - 0.5 * storage.width), cdm.y - 0.15),
                           "orthogonal", "concept annotation", anchor="north", role="orthogonal"))
        _footnotes(items, labels, REPRESENTATION_CORRESPONDENCE, "response", x0, x1, notes_top + 0.3, columns=3,
                   note="The data model fixes meaning; mappings, aliases and storage are separate layers around it")
        model = {"layers": tuple(keys), "storage": REPRESENTATION_STORAGE, "hub": "cdm"}
    model.update({"organization": organization, "level": "representation", "edges": tuple(edges), "tags": tags,
                  "correspondence": _correspondence(REPRESENTATION_CORRESPONDENCE),
                  "format_examples": REPRESENTATION_FORMAT_EXAMPLES})
    return Diagram("scientific_representation", Scene(tuple(items)), model=model)


# --------------------------------------------------------------------------------------------------------------
# The fusion-research community, its activities and shared scientific states (#1643)
# --------------------------------------------------------------------------------------------------------------

#: research roles: functions a person may combine, not professions (key, name)
ECOSYSTEM_ROLES: Tuple[Tuple[str, str], ...] = (
    ("experiment", "Experiment & Diagnostics"), ("theory", "Theory & Modelling"), ("data_ai", "Data Science & AI"),
    ("software", "Scientific Software & Data"), ("planning", "Research Planning & Strategy"),
    ("learners", "Graduate Researchers & Learners"),
)
#: the cross-cutting role, drawn beside every activity
ECOSYSTEM_CROSS_CUTTING = "learners"
#: activity groups and what each holds: (key, name, activities)
ECOSYSTEM_ACTIVITIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("ask", "Ask & Plan", ("research question", "hypothesis", "scenario & campaign planning", "shot & run design")),
    ("produce", "Produce & Observe", ("experiment", "machine operation", "diagnostics", "simulation")),
    ("infer", "Process & Infer", ("calibration", "processing", "reconstruction", "profile & statistical inference")),
    ("model", "Model & Predict", ("theory", "physics modelling", "surrogate models", "prediction & extrapolation")),
    ("test", "Test & Synthesize", ("verification", "validation", "uncertainty & sensitivity", "comparison",
                                   "synthesis")),
    ("preserve", "Preserve & Transfer", ("documentation", "tutorials", "research archive", "reusable notebooks",
                                         "machine knowledge", "education")),
)
#: the roles a scientific state can play -- not file types, and not every state passes through every role
ECOSYSTEM_STATES: Tuple[Tuple[str, str], ...] = (
    ("planned", "Planned"), ("measured", "Measured"), ("processed", "Processed"), ("reconstructed", "Reconstructed"),
    ("simulated", "Simulated"), ("predicted", "Predicted"), ("qualified", "Qualified"),
)
#: research contexts, generic: only the reference implementation is named (key, name, the states it works with)
ECOSYSTEM_CONTEXTS: Tuple[Tuple[str, str, str], ...] = (
    ("reference", "Reference implementation: VEST", "the framework in use"),
    ("existing", "Existing fusion experiments", "measured & reconstructed states"),
    ("campaigns", "Planned experimental campaigns", "planned shots & scenarios"),
    ("design", "Predictive & design studies", "planned runs & simulated states"),
    ("future", "Future fusion devices & reactor concepts", "design & predicted states"),
    ("population", "Cross-machine & population studies", "comparable qualified states"),
)
#: the presentation rendering, a projection of the model above: (key, name, the full items it stands for)
PRESENTATION_ROLES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("experiment", "Experiment & Diagnostics", ("experiment",)), ("theory", "Theory & Modelling", ("theory",)),
    ("data_software", "Data, AI & Software", ("data_ai", "software")),
)
PRESENTATION_VERBS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Plan", ("ask",)), ("Observe", ("produce",)), ("Infer", ("infer",)), ("Model", ("model",)), ("Test", ("test",)),
    ("Compare", ("test",)), ("Learn", ("preserve",)),
)
PRESENTATION_STATES: Tuple[str, ...] = ("planned", "measured", "reconstructed", "simulated", "predicted", "qualified")
PRESENTATION_CONTEXTS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("experiments", "Fusion experiments", ("existing", "campaigns")),
    ("predictive", "Predictive & design studies", ("design",)),
    ("future", "Future research", ("future", "population")),
)
#: compact references of the detailed figure
ECOSYSTEM_REFERENCES: Tuple[str, ...] = (
    "Integrated modelling & data mapping: Imbeaux et al. (2015); integrated data analysis: Fischer & Dinklage (2004)",
    "Fusion V&V: Terry et al. (2008); staged research planning: ITER Research Plan (2018); "
    "multi-machine extrapolation: ITER Physics Basis (1999, 2007)",
)
STATE_STYLE = "concept state,draw=islandblue,line width=1.3pt,fill=qblue!10"
_DETAILS = ("full", "presentation")


def fusion_research_ecosystem(detail: str = "full", *, labels: bool = True) -> Diagram:
    r"""Who does fusion research, what they do, and the shared scientific states that connect it (#1643).

    ``detail`` selects one of two renderings of one model:

    ``"full"``
        the documentation figure. Research roles -- experiment and
        diagnostics, theory and modelling, data science and AI, scientific
        software and data, research planning and strategy, with graduate
        researchers and learners across every activity -- participate in
        research activities rather than owning them. A research question
        and plan lead to a planned shot and a planned simulation run;
        experiment and simulation run in parallel as epistemically distinct
        producers; measured, processed and reconstructed states on one side
        and simulated and predicted states on the other meet in comparison,
        validation and synthesis, which yields qualified states and feeds new
        questions back to planning. Knowledge is preserved and transferred,
        and the states serve generic research contexts, of which only the
        reference implementation, VEST, is named;
    ``"presentation"``
        a one-slide projection of the same model: three role groups, the
        activity verbs (plan, observe, infer, model, test, compare, learn),
        the shared scientific states at the centre and three research
        contexts.

    The research-learning cycle itself is ``fusion_science_knowledge_lifecycle``.
    """
    if detail not in _DETAILS:
        raise ValueError(f"detail must be one of {_DETAILS}, not {detail!r}")
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    roles = dict(ECOSYSTEM_ROLES)
    states = dict(ECOSYSTEM_STATES)
    if detail == "presentation":
        items.append(Label((0.0, 8.6), "{\\Large\\textbf{Connecting the activities of fusion research}}",
                           "concept plain", anchor="south", role="title"))
        items += band(-8.6, 8.6, 6.1, 8.25, "research community", role="community")
        role_boxes: List[Box] = []
        for i, (key, name, _) in enumerate(PRESENTATION_ROLES):
            b = box((i - 1) * 5.6, 7.05, 5.0, 1.0, "\\textbf{" + _tex(name) + "}", style="concept actor",
                    role=f"role:{key}", latex=True)
            role_boxes.append(b)
            items += list(b.items)
        activities = box(0.0, 4.6, 13.0, 1.25, "\\textbf{Research activities}\\\\" + " $\\cdot$ ".join(
            v for v, _ in PRESENTATION_VERBS), role="node:activities", latex=True)
        items += list(activities.items)
        for (key, *_), b in zip(PRESENTATION_ROLES, role_boxes):
            _edge(items, edges, b, activities, f"role:{key}", "activities")
        hub = box(0.0, 2.1, 15.0, 1.75, "{\\Large\\textbf{Shared scientific states}}\\\\[3pt]" + " $\\cdot$ ".join(
            states[k].lower() for k in PRESENTATION_STATES), style="concept hub", role="node:states", latex=True)
        items += list(hub.items)
        _down_or_up(items, edges, activities, hub, "activities", "states", both=True)
        context_boxes: List[Box] = []
        for i, (key, name, _) in enumerate(PRESENTATION_CONTEXTS):
            b = box((i - 1) * 5.6, -0.65, 5.0, 0.95, name, style="concept base", role=f"context:{key}")
            context_boxes.append(b)
            items += list(b.items)
            _down_or_up(items, edges, Box(b.x, hub.y, b.width, hub.height, ()), b, "states", f"context:{key}")
        items.append(Label((8.6, -1.25), "Reference implementation: VEST", "concept annotation", anchor="north east",
                           role="reference"))
        _note(items, labels, -1.85, "Different research roles plan, produce, infer, test, and reuse shared "
              "scientific states")
        model = {"detail": detail, "roles": tuple(k for k, *_ in PRESENTATION_ROLES),
                 "role_projection": {k: v for k, _, v in PRESENTATION_ROLES},
                 "verbs": tuple(v for v, _ in PRESENTATION_VERBS),
                 "verb_projection": {v: g for v, g in PRESENTATION_VERBS}, "states": PRESENTATION_STATES,
                 "contexts": tuple(k for k, *_ in PRESENTATION_CONTEXTS),
                 "context_projection": {k: v for k, _, v in PRESENTATION_CONTEXTS}, "reference": "VEST",
                 "edges": tuple(edges)}
        return Diagram("fusion_research_ecosystem", Scene(tuple(items)), model=model)
    # the detailed figure
    x0, x1 = -10.4, 10.4
    groups = {k: n for k, n, _ in ECOSYSTEM_ACTIVITIES}
    items += band(x0, x1, 13.35, 15.35, "research community: roles a person may combine", role="community")
    main_roles = [(k, n) for k, n in ECOSYSTEM_ROLES if k != ECOSYSTEM_CROSS_CUTTING]
    rw = (x1 - x0 - 0.5 - (len(main_roles) - 1) * 0.25) / len(main_roles)
    for i, (key, name) in enumerate(main_roles):
        b = box(x0 + 0.25 + 0.5 * rw + i * (rw + 0.25), 14.15, rw, 1.0, "\\textbf{" + _tex(name) + "}",
                style="concept actor", role=f"role:{key}", latex=True)
        items += list(b.items)
    items.append(Arrow((0.0, 13.35 - 0.05), (0.0, 12.6 + 0.08), "connector strong", role="edge:community->activities"))
    _record(edges, "community", "activities", "forward")
    items.append(Label((0.15, 12.95), "participate in", "concept annotation", anchor="west", role="participate"))
    xl, xr = -2.3, 5.6
    # activity bands, top to bottom, labelled at the left
    band_spans = (("ask", 9.6, 12.6), ("produce", 6.4, 7.95), ("infer", 3.05, 4.85), ("test", -0.05, 1.2),
                  ("preserve", -3.2, -1.95))
    fx0, fx1 = -8.2, xr + 0.5 * 4.9 + 0.4  # the bands stop short of the feedback loop
    for key, lo, hi in band_spans:
        items += band(fx0, fx1, lo, hi, _tex(groups[key]), role=f"activity:{key}")
    # the experimental and the computational branch share one band: name the second at its right end
    items.append(Label((3.05, 4.85 - 0.15), _tex(groups["model"]), "concept band label", anchor="north east",
                       role="activity:model"))
    # the cross-cutting role, beside every activity band
    learners = box(x0 + 0.8, 4.7, 1.4, 15.8, "", style="concept actor", role=f"role:{ECOSYSTEM_CROSS_CUTTING}")
    items += [learners.items[0], Label((learners.x, learners.y), "\\textbf{" + _tex(roles[ECOSYSTEM_CROSS_CUTTING])
                                       + "} \\textit{across all activities}", "concept plain,rotate=90",
                                       role=f"role:{ECOSYSTEM_CROSS_CUTTING}")]
    nodes: Dict[str, Box] = {}

    def node(key, x, y, w, h, text, style="concept box"):
        nodes[key] = box(x, y, w, h, text, style=style, role=f"node:{key}", latex=True)
        items.extend(nodes[key].items)

    cx = 0.5 * (xl + xr)
    node("question", cx, 11.85, 7.2, 0.85, "\\textbf{Research question \\& hypothesis}")
    node("planning", cx, 10.2, 7.2, 0.85, "\\textbf{Design \\& planning} {\\small scenario $\\cdot$ campaign}")
    node("planned_shot", xl, 8.8, 4.4, 0.7, "{\\small planned} \\textbf{shot}", STATE_STYLE)
    node("planned_run", xr, 8.8, 4.4, 0.7, "{\\small planned} \\textbf{simulation run}", STATE_STYLE)
    node("experiment", xl, 7.1, 4.9, 1.05, _titled("Experiment & diagnostics", "observed evidence",
                                                    sub_size="footnotesize"))
    node("simulation", xr, 7.1, 4.9, 1.05, _titled("Simulation", "model-generated evidence",
                                                    sub_size="footnotesize"))
    node("measured", xl, 5.4, 4.4, 0.7, "\\textbf{measured}", STATE_STYLE)
    node("simulated", xr, 5.4, 4.4, 0.7, "\\textbf{simulated}", STATE_STYLE)
    node("inference", xl, 3.75, 4.9, 1.05, _titled("Processing & inference", "calibration, reconstruction, fitting",
                                                   sub_size="footnotesize"))
    node("modelling", xr, 3.75, 4.9, 1.05, _titled("Modelling & prediction", "theory, physics models, surrogates",
                                                   sub_size="footnotesize"))
    node("processed", xl - 1.4, 2.05, 2.6, 0.7, "\\textbf{processed}", STATE_STYLE)
    node("reconstructed", xl + 1.4, 2.05, 2.6, 0.7, "\\textbf{reconstructed}", STATE_STYLE)
    node("predicted", xr, 2.05, 4.4, 0.7, "\\textbf{predicted}", STATE_STYLE)
    node("comparison", cx, 0.575, 7.2, 0.85, "\\textbf{Comparison $\\cdot$ validation $\\cdot$ synthesis}")
    node("qualified", cx, -1.0, 4.4, 0.7, "\\textbf{qualified}", STATE_STYLE.replace("line width=1.3pt",
                                                                                    "line width=2pt"))
    node("preserve", 0.5 * (fx0 + 3.7 + fx1 - 0.3), -2.575, fx1 - 0.3 - fx0 - 3.7, 0.85, "{\\small " + " $\\cdot$ ".join(
        map(_tex, dict((k, a) for k, _, a in ECOSYSTEM_ACTIVITIES)["preserve"])) + "}", "concept leaf")
    flow = (("question", "planning"), ("experiment", "measured"), ("simulation", "simulated"),
            ("measured", "inference"), ("simulated", "modelling"), ("modelling", "predicted"),
            ("comparison", "qualified"), ("qualified", "preserve"))
    for a, b in flow:
        _down_or_up(items, edges, nodes[a], nodes[b], a, b)
    for a, b in (("planning", "planned_shot"), ("planning", "planned_run"), ("planned_shot", "experiment"),
                 ("planned_run", "simulation"), ("inference", "processed"), ("inference", "reconstructed"),
                 ("reconstructed", "comparison"), ("predicted", "comparison"), ("reconstructed", "modelling")):
        _edge(items, edges, nodes[a], nodes[b], a, b, style="connector dependency" if (a, b) == (
            "reconstructed", "modelling") else "connector")
    # synthesis feeds new questions back to planning
    comp, ques = nodes["comparison"], nodes["question"]
    xf = xr + 0.5 * 4.9 + 0.75
    items.append(Polyline.of([(comp.x + 0.5 * comp.width + 0.08, comp.y), (xf, comp.y), (xf, ques.y),
                              (ques.x + 0.5 * ques.width + 0.08, ques.y)], "connector feedback",
                             role="edge:comparison->question"))
    _record(edges, "comparison", "question", "feedback")
    items.append(Label((xf + 0.15, 0.5 * (comp.y + ques.y)), "new questions, next plan", "concept annotation,rotate=-90",
                       anchor="south", role="feedback_label"))  # outside the loop, clear of the bands' labels
    items.append(Label((x1, -1.0), "rounded blue: shared scientific states", "concept annotation",
                       anchor="east", role="legend"))
    # research contexts
    n = len(ECOSYSTEM_CONTEXTS)
    cw = (x1 - x0 - (n - 1) * 0.2) / n
    ctx_y = -5.1
    items.append(Label((x0, ctx_y + 1.2), "research contexts", "concept band label", anchor="south west",
                       role="contexts"))
    for i, (key, name, sub) in enumerate(ECOSYSTEM_CONTEXTS):
        b = box(x0 + 0.5 * cw + i * (cw + 0.2), ctx_y, cw, 2.25, "{\\small\\bfseries " + _tex(name)
                + "}\\\\[2pt]{\\footnotesize " + _tex(sub) + "}", style="concept base", role=f"context:{key}",
                latex=True)
        items += list(b.items)
    pres = nodes["preserve"]
    items.append(Arrow((cx, pres.y - 0.5 * pres.height - 0.08), (cx, ctx_y + 1.125 + 0.08), "connector",
                       role="edge:preserve->contexts"))
    _record(edges, "preserve", "contexts", "forward")
    items.append(Label((0.0, ctx_y - 1.3), "\\\\".join(map(_tex, ECOSYSTEM_REFERENCES)), "concept reference",
                       anchor="north", role="references"))
    _note(items, labels, ctx_y - 2.4, "Different research roles plan, produce, infer, test, and reuse shared "
          "scientific states")
    state_nodes = {"planned_shot": "planned", "planned_run": "planned"}
    model = {"detail": detail, "roles": tuple(k for k, _ in ECOSYSTEM_ROLES), "cross_cutting": ECOSYSTEM_CROSS_CUTTING,
             "activities": tuple(k for k, *_ in ECOSYSTEM_ACTIVITIES), "states": tuple(k for k, _ in ECOSYSTEM_STATES),
             "state_nodes": {k: state_nodes.get(k, k) for k in nodes if k in states or k in state_nodes},
             "contexts": tuple(k for k, *_ in ECOSYSTEM_CONTEXTS), "reference": "VEST", "nodes": tuple(nodes),
             "edges": tuple(edges), "references": ECOSYSTEM_REFERENCES}
    return Diagram("fusion_research_ecosystem", Scene(tuple(items)), model=model)


# --------------------------------------------------------------------------------------------------------------
# Scientific ownership, maturation and research context (#1645)
# --------------------------------------------------------------------------------------------------------------

#: the bands, top to bottom: (key, name)
OWNERSHIP_BANDS: Tuple[Tuple[str, str], ...] = (
    ("context", "Scientific context"), ("composition", "Research composition"),
    ("computation", "Reusable computation"), ("assessment", "Scientific assessment"),
    ("policy", "Operational policy"),
)
#: the computational implementation classes and what each owns: (key, name, owns)
OWNERSHIP_IMPLEMENTATIONS: Tuple[Tuple[str, str, str], ...] = (
    ("formula", "Formula", "relations, definitions"),
    ("process", "Process", "transformations, inference"),
    ("code", "Code", "external solvers"),
    ("learned", "Learned model", "surrogates, closures"),
)
#: the implementations an optional Actor contract may span
OWNERSHIP_ACTOR_SPAN: Tuple[str, ...] = ("process", "code")
#: graduation: promote matured logic by what it means (meaning, owner)
OWNERSHIP_GRADUATION: Tuple[Tuple[str, str], ...] = (
    ("pure relation", "formula"), ("native transformation", "process"), ("solver operation", "code"),
    ("scientific assessment", "validation"), ("retrieval & persistence", "database"),
    ("learned task", "owning domain + learned model"), ("study-specific orchestration", "stays in the workflow"),
)
#: when to review logic for promotion
OWNERSHIP_TRIGGERS: Tuple[str, ...] = ("reused or copied elsewhere", "needs its own tests", "a stable operation",
                                       "in routine production")
OWNERSHIP_STUDIES: Tuple[str, ...] = ("A", "B", "C")
OWNERSHIP_POLICY: Tuple[str, ...] = ("accept", "review", "warn", "reject", "rerun", "fallback")
OWNERSHIP_VALIDATION: Tuple[str, ...] = ("quality", "credibility", "identifiability", "applicability",
                                         "model discrepancy")


def scientific_ownership_architecture(*, labels: bool = True) -> Diagram:
    r"""Where scientific logic lives as research software matures (#1645).

    Five bands, top to bottom. Scientific context: a Research object -- a
    question, hypothesis and interpretation -- groups Studies by membership,
    not by execution order, and each Study records the selections, method
    and result references and provenance of one reproducible analysis;
    neither executes anything. Research composition: a workflow or notebook
    composes computation for one study (case selection, scans,
    orchestration, figures) and matures reusable logic out of itself.
    Reusable computation: data and the database own scientific state,
    retrieval and persistence; Formula, Process, Code and learned models own
    the computation (an optional Actor contract spans Process and Code
    implementations of one operation, off the normal path); computation and
    data both produce results and evidence. Scientific assessment:
    validation interprets that evidence. Operational policy, optional and
    downstream, decides what a workflow does about an assessment. Beside
    them, the graduation rule: matured logic is promoted by meaning --
    relation to formula, native transformation to process, solver operation
    to code, assessment to validation, retrieval to database, learned task to
    its owning domain -- while study-specific orchestration stays in the
    workflow. The zoomed computational-layer view is in the computational
    layers guide.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List = []
    x0, x1 = -10.6, 10.6
    mx0, mx1 = -6.9, 3.7  # the main column; band labels to its left, the graduation panel to its right
    cx = 0.5 * (mx0 + mx1)
    spans = {"context": (10.6, 13.7), "composition": (8.3, 10.2), "computation": (2.55, 7.9),
             "assessment": (0.65, 2.25), "policy": (-1.05, 0.35)}
    for key, name in OWNERSHIP_BANDS:
        lo, hi = spans[key]
        items += band(x0, mx1 + 0.3, lo, hi, _tex(name), role=f"band:{key}")
    nodes: Dict[str, Box] = {}

    def node(key, x, y, w, h, text, style="concept box"):
        nodes[key] = box(x, y, w, h, text, style=style, role=f"node:{key}", latex=True)
        items.extend(nodes[key].items)

    # context: Research groups Studies; membership is not a dependency between them
    node("research", cx, 12.85, 6.4, 0.95, _titled("Research", "question · hypothesis · interpretation",
                                                   sub_size="footnotesize"), "concept pillar")
    sw = 2.2
    study_xs = [cx + (i - 1) * (sw + 0.5) for i in range(len(OWNERSHIP_STUDIES))]
    for name, x in zip(OWNERSHIP_STUDIES, study_xs):
        node(f"study:{name}", x, 11.15, sw, 0.75, "\\textbf{Study " + name + "}", "concept pillar")
        items.append(Polyline.of([(x, 11.525 + 0.05), (x, 12.375 - 0.05)], "concept tick", role=f"member:{name}"))
    items.append(Label((study_xs[0] - 0.5 * sw - 0.2, 11.15), "membership,\\\\not execution order",
                       "concept annotation", anchor="east", role="membership"))
    gx0 = mx1 + 0.9
    items.append(Label((gx0, 12.2), "\\textbf{Study}: selections, method \\& result\\\\references, provenance\\\\"
                       "\\textbf{Research}: why Studies belong together\\\\neither executes computation",
                       "concept annotation,align=left", anchor="west", role="context_rule"))
    # composition: the workflow composes and incubates
    node("workflow", cx, 9.2, mx1 - mx0 - 0.6, 1.05, _titled("Workflow / notebook", "case selection · scans · "
                                                             "orchestration · figures · one-off comparison"),
         "concept group")
    _down_or_up(items, edges, nodes["study:B"], nodes["workflow"], "study:B", "workflow")
    items[-1] = Arrow(items[-1].start, items[-1].end, "connector feedback", role=items[-1].role)
    items.append(Label((cx + 0.15, 10.4), "references", "concept annotation", anchor="west", role="references"))
    # computation: data beside four implementation classes, both producing results and evidence
    node("database", x0 + 0.25 + 1.4, 4.85, 2.8, 3.2, _titled("Data & database", "state · retrieval · selection · "
                                                              "persistence", sub_size="footnotesize"), "concept base")
    n = len(OWNERSHIP_IMPLEMENTATIONS)
    iw = (mx1 - mx0 - 0.6 - (n - 1) * 0.3) / n
    ixs = [mx0 + 0.3 + 0.5 * iw + i * (iw + 0.3) for i in range(n)]
    impl_y, impl_h = 5.8, 1.7
    for (key, name, owns), x in zip(OWNERSHIP_IMPLEMENTATIONS, ixs):
        node(key, x, impl_y, iw, impl_h, _titled(name, owns, sub_size="footnotesize"))
    calls_y = 7.35
    items.append(Polyline.of([(ixs[0], calls_y), (ixs[-1], calls_y)], "connector line", role="calls"))
    for key, x in zip((k for k, *_ in OWNERSHIP_IMPLEMENTATIONS), ixs):
        items.append(Arrow((x, calls_y), (x, impl_y + 0.5 * impl_h + 0.08), "connector", role=f"edge:calls->{key}"))
        _record(edges, "calls", key, "forward")
    wf = nodes["workflow"]
    items.append(Arrow((cx, wf.y - 0.525 - 0.08), (cx, calls_y), "connector", role="edge:workflow->calls"))
    _record(edges, "workflow", "calls", "forward")
    items.append(Label((cx + 0.15, calls_y + 0.3), "composes", "concept annotation", anchor="west",
                       role="composes"))
    # the optional Actor: an overlay around the implementations it may span, off the normal path
    span = [x for (k, *_), x in zip(OWNERSHIP_IMPLEMENTATIONS, ixs) if k in OWNERSHIP_ACTOR_SPAN]
    ax0, ax1 = min(span) - 0.5 * iw - 0.12, max(span) + 0.5 * iw + 0.12
    ay0, ay1 = impl_y - 0.5 * impl_h - 0.8, impl_y + 0.5 * impl_h + 0.12
    # the label sits between the two arrows that leave the boxes through the frame
    items += [Polyline.of([(ax0, ay0), (ax1, ay0), (ax1, ay1), (ax0, ay1)], "concept frame", role="node:actor",
                          closed=True),
              Label((0.5 * (ax0 + ax1), ay0 + 0.4), "optional Actor\\\\contract", "concept annotation",
                    role="node:actor")]
    ev_y, ev_h = 3.15, 1.0
    node("evidence", cx, ev_y, mx1 - mx0 - 0.6, ev_h, _titled("Results & evidence", "state · prediction · residuals "
                                                              "· convergence · comparisons", sub_size="footnotesize"),
         "concept state")
    for key, x in zip((k for k, *_ in OWNERSHIP_IMPLEMENTATIONS), ixs):
        items.append(Arrow((x, impl_y - 0.5 * impl_h - 0.08), (x, ev_y + 0.5 * ev_h + 0.08), "connector", role=f"edge:{key}->evidence"))
        _record(edges, key, "evidence", "forward")
    db = nodes["database"]
    # each arrow at a height both boxes span, so it leaves the database and enters its target
    for target, y in ((nodes["formula"], impl_y - 0.2), (nodes["evidence"], ev_y + 0.25)):
        key = "formula" if target is nodes["formula"] else "evidence"
        items.append(Arrow((db.x + 0.5 * db.width + 0.08, y), (target.x - 0.5 * target.width - 0.08, y), "connector",
                           role=f"edge:database->{key}"))
        _record(edges, "database", key, "forward")
    # assessment and policy
    node("validation", cx, 1.4, mx1 - mx0 - 0.6, 0.95, "\\textbf{Validation} interprets evidence: {\\small "
         + ", ".join(OWNERSHIP_VALIDATION) + "}", "concept vv")
    _down_or_up(items, edges, nodes["evidence"], nodes["validation"], "evidence", "validation")
    node("policy", cx, -0.35, mx1 - mx0 - 0.6, 0.95, "\\textbf{Optional use policy}: {\\small " + " $\\cdot$ ".join(
        OWNERSHIP_POLICY) + "}", "concept group")
    _down_or_up(items, edges, nodes["validation"], nodes["policy"], "validation", "policy")
    items[-1] = Arrow(items[-1].start, items[-1].end, "connector feedback", role=items[-1].role)
    # graduation: promote by meaning
    gw = x1 - gx0
    gx = gx0 + 0.5 * gw
    g_top, g_bottom = 8.05, 2.15
    rows = "\\\\[1pt]".join(_tex(m) + " $\\rightarrow$ \\textbf{" + _tex(o) + "}" for m, o in OWNERSHIP_GRADUATION)
    graduation = box(gx, 0.5 * (g_top + g_bottom), gw, g_top - g_bottom, "\\textbf{Graduation by meaning}\\\\[2pt]"
                     "{\\footnotesize\\itshape review when " + ", ".join(OWNERSHIP_TRIGGERS) + "}\\\\[4pt]{\\small "
                     + rows + "}", style="concept leaf", role="node:graduation", latex=True)
    items += list(graduation.items)
    # one bent arrow: a single path, so the corner is joined and only the end carries a head
    items.append(Polyline.of([(wf.x + 0.5 * wf.width + 0.08, wf.y), (gx, wf.y), (gx, g_top + 0.08)],
                             "connector strong", role="edge:workflow->graduation"))
    _record(edges, "workflow", "graduation", "forward")
    items.append(Label((0.5 * (wf.x + 0.5 * wf.width + gx), wf.y + 0.1), "maturation", "concept annotation",
                       anchor="south", role="maturation"))
    for y, text in ((1.4, "computation produces evidence;\\\\validation interprets it"),
                    (-0.35, "policy decides what to do\\\\with an assessment")):
        items.append(Label((gx0, y), text, "concept annotation,align=left", anchor="west", role="rule"))
    _note(items, labels, -1.4, "Reusable scientific logic belongs to its semantic owner; workflows compose it, "
          "validation assesses it, Study and Research preserve why it was used")
    model = {"bands": tuple(k for k, _ in OWNERSHIP_BANDS),
             "implementations": tuple(k for k, *_ in OWNERSHIP_IMPLEMENTATIONS), "actor_span": OWNERSHIP_ACTOR_SPAN,
             "studies": OWNERSHIP_STUDIES, "graduation": OWNERSHIP_GRADUATION, "triggers": OWNERSHIP_TRIGGERS,
             "policy": OWNERSHIP_POLICY, "nodes": tuple(nodes), "edges": tuple(edges)}
    return Diagram("scientific_ownership_architecture", Scene(tuple(items)), model=model)
