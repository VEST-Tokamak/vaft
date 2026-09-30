"""Integrated-modeling diagrams: three independent model axes, their space, and model coupling (#1085).

``knowledge_basis``
    where a model's knowledge comes from: first-principles to data-driven,
    with physics-informed as a bridge over the axis rather than one point;
``computational_realization``
    how the model is evaluated: analytical, semi-analytical, reduced
    numerical, fully numerical;
``physical_abstraction``
    at what description level the system is represented, from particle
    orbits to static equilibrium, with the reduction between levels;
``integrated_modeling_space``
    the three combined: knowledge basis across, realization up, physical
    abstraction as the colour of each model; conceptual and heuristic models
    are an explanatory layer beside the space, not a fourth axis;
``integrated_modeling_process``
    how models, data and states are connected, with typed couplings.

The vocabulary and the example placements are data in ``_modeling_schema``.
The diagrams communicate location and character, not ranking: no end of an
axis is better.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import _modeling_schema as schema
from ._concept import Box, box, connector
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

_ABSTRACTION_KEYS = tuple(k for k, _, _ in schema.PHYSICAL_ABSTRACTION)
_COUPLING_LABEL = dict(schema.COUPLING_TYPES)
_EXAMPLES = ("generic", "fusion", "tearing")


def _coupling_style(key: str) -> str:
    """The template style of a coupling type: ``coupling <key with spaces>``."""
    return "coupling " + key.replace("_", " ")


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def capsule_half_width(name: str) -> float:
    """Estimated half-width of a model capsule [cm]: \\small text, about 0.145 cm a character (measured on the render), plus padding."""
    return 0.5 * (0.145 * len(name) + 0.45)


def _capsule_style(abstraction) -> str:
    return f"im {abstraction}" if abstraction is not None else "im global"


def knowledge_basis(*, labels: bool = True) -> Diagram:
    r"""Where a model's knowledge comes from: first-principles to data-driven.

    Four stations on one axis -- mechanistic/first-principles,
    semi-empirical/closure, empirical, data-driven -- each with a generic
    example. Physics-informed models combine physical constraints with
    fitted or learned parts, so they are drawn as a bridge spanning the axis,
    not as a point on it. Semi-empirical is a knowledge basis;
    semi-analytical is a computational realization (``computational_realization``).
    """
    labels = _check_labels(labels)
    width = 13.0
    items: List = [Arrow((0.0, 0.0), (width, 0.0), "im axis", role="axis", both=True)]
    lo, hi = schema.PHYSICS_INFORMED_SPAN
    items.append(Polyline.of([(lo * width, 0.75), (hi * width, 0.75), (hi * width, 1.65), (lo * width, 1.65)],
                             "im bridge", role="knowledge:physics_informed", closed=True))
    items.append(Label((0.5 * (lo + hi) * width, 1.2),
                       "\\textbf{physics-informed}: physical constraints\\\\ with fitted or learned parts",
                       "im station", role="knowledge:physics_informed"))
    for key, text, pos, example in schema.KNOWLEDGE_BASIS:
        x = pos * width
        items += [Polyline.of([(x, -0.15), (x, 0.15)], "im tick", role=f"knowledge:{key}"),
                  Label((x, -0.35), f"\\textbf{{{text}}}", "im station,text width=3.6cm", anchor="north",
                        role=f"knowledge:{key}")]
        if labels:
            items.append(Label((x, -1.25), example, "im example,text width=3.6cm", anchor="north",
                               role=f"knowledge:{key}:example"))
    items += [Label((-0.25, 0.0), "FIRST-PRINCIPLES", "im axis title", anchor="east", role="axis"),
              Label((width + 0.25, 0.0), "DATA-DRIVEN", "im axis title", anchor="west", role="axis")]
    if labels:
        items.append(Label((0.5 * width, -2.4), "Knowledge basis answers where the model knowledge comes from. "
                           "Semi-empirical belongs here; semi-analytical is a computational realization.",
                           "note", anchor="north", role="note"))
    model = {"axis": ("first-principles", "data-driven"),
             "stations": tuple(k for k, *_ in schema.KNOWLEDGE_BASIS),
             "bridge": ("physics_informed", schema.PHYSICS_INFORMED_SPAN)}
    return Diagram("knowledge_basis", Scene(tuple(items)), model=model)


def computational_realization(*, labels: bool = True) -> Diagram:
    r"""How a model is evaluated: analytical, semi-analytical, reduced numerical, fully numerical.

    A ladder from the closed form at the bottom to the full iterative PDE
    solver at the top, each rung with generic examples. Learned surrogates
    and neural operators are not a rung: they are placed in the modeling
    space by their knowledge basis and their role.
    """
    labels = _check_labels(labels)
    height, box_w, box_h = 8.0, 3.8, 1.0
    items: List = [Arrow((0.0, -0.3), (0.0, height + 0.3), "im axis", role="axis", both=True)]
    for key, text, pos, example in schema.COMPUTATIONAL_REALIZATION:
        y = pos * height
        b = box(2.6, y, box_w, box_h, f"\\textbf{{{text}}}", role=f"realization:{key}", latex=True)
        items += [Polyline.of([(-0.15, y), (0.15, y)], "im tick", role=f"realization:{key}"),
                  Polyline.of([(0.15, y), (2.6 - 0.5 * box_w, y)], "im grid", role=f"realization:{key}")]
        items += list(b.items)
        if labels:
            items.append(Label((2.6 + 0.5 * box_w + 0.3, y), example, "im example,text width=5.6cm,align=left",
                               anchor="west", role=f"realization:{key}:example"))
    items += [Label((-0.3, -0.3), "ANALYTICAL", "im axis title", anchor="east", role="axis"),
              Label((-0.3, height + 0.3), "NUMERICAL", "im axis title", anchor="east", role="axis")]
    if labels:
        items.append(Label((3.6, -1.0), "Learned surrogates are not a rung: they are placed by knowledge basis "
                           "and role", "note", anchor="north", role="note"))
    model = {"axis": ("analytical", "numerical"),
             "rungs": tuple(k for k, *_ in schema.COMPUTATIONAL_REALIZATION)}
    return Diagram("computational_realization", Scene(tuple(items)), model=model)


def physical_abstraction(*, labels: bool = True) -> Diagram:
    r"""At what description level the system is represented, and how each level reduces to the next.

    Particle/orbit, kinetic, moment/fluid, MHD, equilibrium/static, each
    with what it resolves; the arrows are the reductions -- ensemble average,
    velocity moments with a closure, single fluid at low frequency,
    stationary force balance. A description level, not a fidelity ranking:
    each level is adequate for the questions it keeps.
    """
    labels = _check_labels(labels)
    step, box_w, box_h = 1.9, 4.2, 1.0
    items: List = []
    levels: List[Box] = []
    for i, (key, text, resolves) in enumerate(schema.PHYSICAL_ABSTRACTION):
        y = -i * step
        b = box(0.0, y, box_w, box_h, f"\\textbf{{{text}}}", style=f"im level {key}", role=f"abstraction:{key}",
                latex=True)
        levels.append(b)
        items += list(b.items)
        if labels:
            items.append(Label((0.5 * box_w + 0.4, y), resolves, "im example,text width=4.6cm,align=left",
                               anchor="west", role=f"abstraction:{key}:resolves"))
    for (a, b), reduction in zip(zip(levels, levels[1:]), schema.ABSTRACTION_REDUCTIONS):
        items.append(connector(a, b, style="im reduction", role="abstraction:reduction"))
        items.append(Label((-0.25, 0.5 * (a.y + b.y)), reduction, "im station", anchor="east",
                           role="abstraction:reduction"))
    if labels:
        items.append(Label((0.0, -(len(levels) - 1) * step - 1.1),
                           "A description level, not a fidelity ranking", "note", anchor="north", role="note"))
    model = {"levels": _ABSTRACTION_KEYS, "reductions": schema.ABSTRACTION_REDUCTIONS}
    return Diagram("physical_abstraction", Scene(tuple(items)), model=model)


_SPACE_W, _SPACE_H = 15.0, 9.0


def _space_xy(knowledge: float, realization: float) -> Tuple[float, float]:
    return (knowledge * _SPACE_W, realization * _SPACE_H)


def integrated_modeling_space(examples: str = "generic", *, labels: bool = True) -> Diagram:
    r"""The three axes combined: where models sit, and what character they have.

    Knowledge basis across (``knowledge_basis``), computational realization
    up (``computational_realization``), physical abstraction as the colour of
    each model (``physical_abstraction``; dashed white for a global 0-D
    quantity). The shaded bridge under the axis is the physics-informed span.
    Conceptual and heuristic models (physical picture, cartoon, scaling
    argument, toy model) form an explanatory layer beside the space, not a
    fourth axis.

    ``examples`` is ``"generic"`` (domain-independent models), ``"fusion"``
    (a small overlay of fusion codes and models) or ``"tearing"`` (one
    phenomenon from a physical picture to a nonlinear simulation). The
    placements are illustrative; they show location and character, not
    ranking.
    """
    labels = _check_labels(labels)
    if examples not in _EXAMPLES:
        raise ValueError(f"examples must be one of {_EXAMPLES}, not {examples!r}")
    W, H = _SPACE_W, _SPACE_H
    items: List = [Polyline.of([(0, 0), (W, 0), (W, H), (0, H)], "im grid", role="plane", closed=True)]
    # realization rungs and knowledge stations as a faint grid
    for key, text, pos, _ in schema.COMPUTATIONAL_REALIZATION:
        y = pos * H
        items += [Polyline.of([(0, y), (W, y)], "im grid", role=f"realization:{key}"),
                  Label((-0.2, y), text, "im station", anchor="east", role=f"realization:{key}")]
    for key, text, pos, _ in schema.KNOWLEDGE_BASIS:
        x = pos * W
        items += [Polyline.of([(x, 0), (x, H)], "im grid", role=f"knowledge:{key}"),
                  Label((x, -0.3), text, "im station,text width=2.9cm", anchor="north", role=f"knowledge:{key}")]
    lo, hi = schema.PHYSICS_INFORMED_SPAN
    items += [Polyline.of([(lo * W, -1.75), (hi * W, -1.75), (hi * W, -1.3), (lo * W, -1.3)], "im bridge,rounded corners=4pt",
                          role="knowledge:physics_informed", closed=True),
              Label((0.5 * (lo + hi) * W, -1.525), "physics-informed", "im station",
                    role="knowledge:physics_informed")]
    items += [Arrow((0.0, -2.2), (W, -2.2), "im axis", role="axis", both=True),
              Label((0.0, -2.35), "FIRST-PRINCIPLES", "im axis title", anchor="north west", role="axis"),
              Label((W, -2.35), "DATA-DRIVEN", "im axis title", anchor="north east", role="axis"),
              Arrow((-3.2, 0.0), (-3.2, H), "im axis", role="axis", both=True),
              Label((-3.35, 0.0), "ANALYTICAL", "im axis title,rotate=90", anchor="south west", role="axis"),
              Label((-3.35, H), "NUMERICAL", "im axis title,rotate=90", anchor="south east", role="axis")]
    # the explanatory layer, beside the space
    cx = W + 3.3
    bottom = 0.0
    layer = [(cx - 2.2, bottom), (cx + 2.2, bottom), (cx + 2.2, 3.4), (cx - 2.2, 3.4)]
    items += [Polyline.of(layer, "im conceptual", role="conceptual_layer", closed=True),
              Label((cx, 3.2), "\\textbf{conceptual / heuristic}", "im station", anchor="north",
                    role="conceptual_layer"),
              Label((cx, 2.6), ", ".join(schema.CONCEPTUAL_LAYER), "im example,text width=4.2cm", anchor="north",
                    role="conceptual_layer"),
              Label((cx, 1.45), "explains; not a fourth axis", "im station", role="conceptual_layer")]
    # physical abstraction legend
    items.append(Label((cx - 2.2, H), "\\textbf{physical abstraction}", "im station", anchor="north west",
                       role="legend"))
    for i, (key, text, _) in enumerate(schema.PHYSICAL_ABSTRACTION):
        items.append(Label((cx - 2.2, H - 0.75 - 0.62 * i), text, f"im {key}", anchor="west",
                           role=f"legend:{key}"))
    items.append(Label((cx - 2.2, H - 0.75 - 0.62 * len(schema.PHYSICAL_ABSTRACTION)), "global (0-D)",
                       "im global", anchor="west", role="legend:global"))
    # the models
    models: Sequence[schema.ModelDescriptor] = {
        "generic": schema.GENERIC_MODELS, "fusion": schema.FUSION_MODELS, "tearing": schema.TEARING_PATH,
    }[examples]
    at = {m.name: _space_xy(m.knowledge, m.realization) for m in models}
    for m in models:
        items.append(Label(at[m.name], m.name, _capsule_style(m.abstraction), role=f"model:{m.name}"))
        if m.emulates is not None:  # a surrogate sits at the y of the model it replaces, linked to it
            (x0, y0), (x1, _) = at[m.name], at[m.emulates]
            sign = 1.0 if x1 > x0 else -1.0
            items.append(Arrow((x0 + sign * (capsule_half_width(m.name) + 0.1), y0),
                               (x1 - sign * (capsule_half_width(m.emulates) + 0.1), y0), "coupling surrogate",
                               role=f"emulates:{m.name}"))
    if examples == "tearing":
        # the physical picture lives in the explanatory layer; the path starts there
        picture = (cx, 0.6)
        items.append(Label(picture, "island picture", "im global", role="conceptual_layer:picture"))
        first = _space_xy(models[0].knowledge, models[0].realization)
        items.append(Arrow((cx - 1.1, picture[1]), (first[0] + 2.0, first[1]), "im reduction", role="tearing_path"))
        path = [_space_xy(m.knowledge, m.realization) for m in models]
        for a, b in zip(path, path[1:]):
            items.append(_gap_arrow(a, b, 0.42, "im reduction", "tearing_path"))
    if labels:
        note = {"generic": "Domain-independent examples; placements show location and character, not ranking",
                "fusion": "Fusion overlay: illustrative placements of familiar codes, not a classification",
                "tearing": "One phenomenon: a physical picture, then first-principles models of rising cost"}[examples]
        items.append(Label((0.5 * W, -3.1), note, "note", anchor="north", role="note"))
    model = {"x_axis": "knowledge_basis", "y_axis": "computational_realization", "colour": "physical_abstraction",
             "examples": examples, "models": tuple(models), "explanatory_layer": schema.CONCEPTUAL_LAYER}
    return Diagram("integrated_modeling_space", Scene(tuple(items)), model=model)


def _gap_arrow(a, b, gap: float, style: str, role: str) -> Arrow:
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = (b - a) / np.hypot(*(b - a))
    return Arrow(tuple(a + gap * u), tuple(b - gap * u), style, role=role)


#: node centres of the process graph [cm]
_PROCESS_AT: Dict[str, Tuple[float, float]] = {
    "experiment": (0.0, 10.8), "measurement": (0.0, 9.2), "processing": (0.0, 7.6), "inverse": (0.0, 6.0),
    "state": (0.0, 4.4), "forward": (-3.4, 2.4), "data_driven": (3.4, 2.4), "prediction": (0.0, 0.4),
    "validation": (0.0, -1.2),
}
_NODE_W, _NODE_H = 3.6, 0.9


def integrated_modeling_process(*, labels: bool = True) -> Diagram:
    r"""How models interact: a graph of models, data and states with typed couplings.

    The experiment is measured; processed data feed an inverse model --
    parameter inference -- that yields a physical state. A forward model,
    coupled iteratively to that state, gives a prediction. A data-driven
    model is used in one of several alternative ways: as a closure inside
    the forward model, or as its surrogate replacement, or as a residual
    correction of its prediction (the hybrid mode). Validation compares the
    prediction with the processed measurement, calibrates the data-driven
    model and closes the loop as feedback or control on the experiment. The
    legend lists the coupling types; a coupling is a label on an edge.
    """
    labels = _check_labels(labels)
    items: List = []
    nodes: Dict[str, Box] = {}
    for role, text in schema.PROCESS_NODES:
        x, y = _PROCESS_AT[role]
        b = box(x, y, _NODE_W, _NODE_H, text, role=f"node:{role}")
        nodes[role] = b
        items += list(b.items)
    hw, gap = 0.5 * _NODE_W, 0.08
    for c in schema.PROCESS_COUPLINGS:
        style, role = _coupling_style(c.coupling), f"coupling:{c.coupling}"
        s, t = nodes[c.source], nodes[c.target]
        if c.coupling == "feedback":  # around the far left, up to the experiment
            x, y0 = -6.6, s.y - 0.2
            items.append(Polyline.of([(s.x - hw, y0), (x, y0), (x, t.y), (t.x - hw - gap, t.y)], style, role=role))
            if labels:
                items.append(Label((x - 0.15, 0.5 * (y0 + t.y)), "feedback / control", "im station,rotate=90",
                                   anchor="south", role=role))
        elif c.source == "processing" and c.target == "validation":  # the measured data validation compares with
            x, y1 = -5.8, t.y + 0.2
            items.append(Polyline.of([(s.x - hw, s.y), (x, s.y), (x, y1), (t.x - hw - gap, y1)], style, role=role))
            if labels:
                items.append(Label((x - 0.15, 0.5 * (s.y + y1)), "measured data", "im station,rotate=90",
                                   anchor="south", role=role))
        elif c.source == "validation" and c.target == "data_driven":  # around the right, up into the model
            x = 6.6
            items.append(Polyline.of([(s.x + hw, s.y), (x, s.y), (x, t.y), (t.x + hw + gap, t.y)], style,
                                     role=role))
            if labels:
                items.append(Label((x + 0.15, 0.5 * (s.y + t.y)), "calibration", "im station,rotate=90",
                                   anchor="north", role=role))
        elif c.source == "data_driven" and c.target == "forward":  # alternative uses of one data-driven model
            dy = 0.2 if c.coupling == "closure" else -0.2
            items.append(Arrow((s.x - hw - gap, s.y + dy), (t.x + hw + gap, t.y + dy), style, role=role))
            if labels:
                text = "closure" if c.coupling == "closure" else "or replacement"
                items.append(Label((0.0, s.y + dy + (0.1 if dy > 0 else -0.1)), text, "im station",
                                   anchor="south" if dy > 0 else "north", role=role))
        else:
            arrow = connector(s, t, style=style, role=role)
            if c.coupling == "iterative":
                arrow = Arrow(arrow.start, arrow.end, style, role=role, both=True)
            items.append(arrow)
            if labels and c.coupling in ("iterative", "residual", "validation", "calibration"):
                mid = 0.5 * (np.asarray(arrow.start) + np.asarray(arrow.end))
                text = {"calibration": "parameter inference"}.get(c.coupling, _COUPLING_LABEL[c.coupling])
                if abs(arrow.end[0] - arrow.start[0]) < 1e-9:  # vertical: beside it
                    items.append(Label((mid[0] + 0.15, mid[1]), text, "im station", anchor="west", role=role))
                else:  # diagonal: just outside its outer side
                    side = "south east" if mid[0] < 0.0 else "north west"
                    items.append(Label((mid[0] + (-0.05 if mid[0] < 0 else 0.05), mid[1]), text, "im station",
                                       anchor=side, role=role))
    # legend of every coupling type, bottom right
    lx, ly = 8.2, 3.6
    items.append(Label((lx, ly + 0.2), "\\textbf{coupling types}", "im station", anchor="south west", role="legend"))
    for i, (key, text) in enumerate(schema.COUPLING_TYPES):
        y = ly - 0.62 * i
        items += [Arrow((lx, y), (lx + 1.2, y), _coupling_style(key), role=f"legend:{key}", both=key == "iterative"),
                  Label((lx + 1.4, y), text, "im station", anchor="west", role=f"legend:{key}")]
    if labels:
        items.append(Label((0.0, -2.2), "A generic process: models, data and states joined by typed couplings; "
                           "the data-driven uses are alternatives", "note", anchor="north", role="note"))
    model = {"nodes": tuple(r for r, _ in schema.PROCESS_NODES), "couplings": schema.PROCESS_COUPLINGS,
             "coupling_types": tuple(k for k, _ in schema.COUPLING_TYPES)}
    return Diagram("integrated_modeling_process", Scene(tuple(items)), model=model)
