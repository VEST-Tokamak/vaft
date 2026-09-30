"""Integrated-modeling diagrams (#1085): three independent axes, their space, and typed coupling."""

import dataclasses
import itertools
import re

import pytest

import vaft.diagram
from vaft.diagram import _modeling_schema as schema
from vaft.diagram import _render
from vaft.diagram._integrated_modeling import _SPACE_H, _SPACE_W, capsule_half_width
from vaft.diagram._scene import Arrow, Label, Polyline

BUILDERS = ("knowledge_basis", "computational_realization", "physical_abstraction",
            "integrated_modeling_space", "integrated_modeling_process")


def _texts(diagram):
    return " ".join(i.text for i in diagram.scene.items if isinstance(i, Label))


def test_the_three_axes_are_independent_and_semi_words_sit_on_the_right_axis():
    kb = {k for k, *_ in schema.KNOWLEDGE_BASIS}
    cr = {k for k, *_ in schema.COMPUTATIONAL_REALIZATION}
    pa = {k for k, *_ in schema.PHYSICAL_ABSTRACTION}
    assert not (kb & cr or kb & pa or cr & pa)
    assert "semi_empirical" in kb and "semi_empirical" not in cr
    assert "semi_analytical" in cr and "semi_analytical" not in kb
    # the drawn stations are the schema's, in order along each axis
    assert vaft.diagram.knowledge_basis().model["stations"] == tuple(k for k, *_ in schema.KNOWLEDGE_BASIS)
    assert vaft.diagram.computational_realization().model["rungs"] == tuple(
        k for k, *_ in schema.COMPUTATIONAL_REALIZATION)
    assert vaft.diagram.physical_abstraction().model["levels"] == tuple(k for k, *_ in schema.PHYSICAL_ABSTRACTION)
    for axis in (schema.KNOWLEDGE_BASIS, schema.COMPUTATIONAL_REALIZATION):
        positions = [pos for _, _, pos, _ in axis]
        assert positions == sorted(positions) and 0 < positions[0] and positions[-1] < 1


def test_physics_informed_is_a_bridge_not_a_station():
    d = vaft.diagram.knowledge_basis()
    assert "physics_informed" not in d.model["stations"]
    lo, hi = d.model["bridge"][1]
    stations = [pos for _, _, pos, _ in schema.KNOWLEDGE_BASIS]
    assert lo < stations[1] and hi > stations[2]  # it spans more than one station
    bridge = [i for i in d.scene.role("knowledge:physics_informed") if isinstance(i, Polyline)]
    assert len(bridge) == 1 and bridge[0].closed


def test_learned_surrogates_are_not_a_rung_and_intuitive_is_not_a_label():
    assert not any("surrogate" in text or "neural" in text for _, text, _, _ in schema.COMPUTATIONAL_REALIZATION)
    for name in BUILDERS:
        assert "intuitive" not in _texts(getattr(vaft.diagram, name)()).lower()
    # a surrogate takes the y of what it emulates, and is linked to it
    for models in (schema.GENERIC_MODELS, schema.FUSION_MODELS):
        by_name = {m.name: m for m in models}
        surrogates = [m for m in models if m.role == "surrogate"]
        assert surrogates
        for m in surrogates:
            assert m.emulates in by_name and m.realization == by_name[m.emulates].realization


def test_positions_are_semantic_first_principles_codes_share_the_left_column():
    first_principles = next(pos for k, _, pos, _ in schema.KNOWLEDGE_BASIS if k == "first_principles")
    semi_empirical = next(pos for k, _, pos, _ in schema.KNOWLEDGE_BASIS if k == "semi_empirical")
    fusion = {m.name: m for m in schema.FUSION_MODELS}
    column = [fusion[n].knowledge for n in ("Solov'ev", "CHEASE", "GPEC", "DCON / RDCON", "ASCOT5", "BEAMS3D")]
    assert len(set(column)) == 1 and abs(column[0] - first_principles) < 0.05
    # GPEC builds on DCON, ASCOT5 and BEAMS3D are both orbit codes: no one is "less first-principles"
    assert fusion["EFIT"].role == "inverse" and first_principles < fusion["EFIT"].knowledge < semi_empirical
    # the tearing path: the Rutherford equation (fitted coefficients) sits right of the first-principles steps
    path = {m.name: m for m in schema.TEARING_PATH}
    assert path["Rutherford equation"].knowledge > max(m.knowledge for n, m in path.items()
                                                       if n != "Rutherford equation")


def test_the_abstraction_ladder_names_each_reduction():
    d = vaft.diagram.physical_abstraction()
    arrows = [i for i in d.scene.role("abstraction:reduction") if isinstance(i, Arrow)]
    assert len(arrows) == len(schema.PHYSICAL_ABSTRACTION) - 1 == len(schema.ABSTRACTION_REDUCTIONS)
    assert all(a.start[1] > a.end[1] for a in arrows)  # downward: from more to less resolved
    assert "velocity moments + closure" in _texts(d)


@pytest.mark.parametrize("examples", ["generic", "fusion", "tearing"])
def test_each_model_is_drawn_where_its_descriptor_puts_it_in_its_abstraction_colour(examples):
    d = vaft.diagram.integrated_modeling_space(examples)
    assert (d.model["x_axis"], d.model["y_axis"], d.model["colour"]) == (
        "knowledge_basis", "computational_realization", "physical_abstraction")
    boxes = []
    for m in d.model["models"]:
        (capsule,) = d.scene.role(f"model:{m.name}")
        assert capsule.at == pytest.approx((m.knowledge * _SPACE_W, m.realization * _SPACE_H))
        assert capsule.style == (f"im {m.abstraction}" if m.abstraction else "im global")
        half_w, half_h = capsule_half_width(m.name), 0.27
        x, y = capsule.at
        assert 0.0 < x - half_w and x + half_w < _SPACE_W and 0.0 < y - half_h and y + half_h < _SPACE_H
        boxes.append((m.name, x - half_w, x + half_w, y - half_h, y + half_h))
    for (a, ax0, ax1, ay0, ay1), (b, bx0, bx1, by0, by1) in itertools.combinations(boxes, 2):
        assert ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0, f"{a} overlaps {b}"
    # conceptual models are an explanatory layer beside the space, not an axis
    layer = [i for i in d.scene.role("conceptual_layer") if isinstance(i, Polyline)]
    assert layer and min(p[0] for p in layer[0].points) > 15.0
    assert "explains; not a fourth axis" in _texts(d)


def test_the_fusion_overlay_names_the_issue_examples_and_the_tearing_path_rises():
    names = {m.name for m in schema.FUSION_MODELS}
    for code in ("Solov'ev", "EFIT", "CHEASE", "DCON / RDCON", "GPEC", "TGLF", "TRANSP", "ASCOT5", "BEAMS3D"):
        assert code in names
    assert {m.role for m in schema.FUSION_MODELS} >= {"forward", "inverse", "surrogate", "classifier"}
    rise = [m.realization for m in schema.TEARING_PATH]
    assert rise == sorted(rise) and len({m.abstraction for m in schema.TEARING_PATH}) == 1
    tearing = vaft.diagram.integrated_modeling_space("tearing")
    arrows = tearing.scene.role("tearing_path")
    assert len(arrows) == 1 + (len(schema.TEARING_PATH) - 1)  # picture -> first model, then model to model
    assert arrows[0].start[0] > _SPACE_W  # it starts in the explanatory layer, outside the space
    with pytest.raises(ValueError):
        vaft.diagram.integrated_modeling_space("stellarator")


def test_descriptors_reject_unknown_vocabulary():
    with pytest.raises(ValueError):
        schema.ModelDescriptor("x", 0.5, 0.5, "gyrokinetic", "forward")
    with pytest.raises(ValueError):
        schema.ModelDescriptor("x", 0.5, 1.5, "mhd", "forward")
    with pytest.raises(ValueError):
        schema.ModelCoupling("a", "b", "magic")
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.GENERIC_MODELS[0].knowledge = 0.0


def test_the_process_graph_draws_every_coupling_with_its_type():
    d = vaft.diagram.integrated_modeling_process()
    nodes = {r for r, _ in schema.PROCESS_NODES}
    assert set(d.model["nodes"]) == nodes
    for c in schema.PROCESS_COUPLINGS:
        assert c.source in nodes and c.target in nodes
    used = {c.coupling for c in schema.PROCESS_COUPLINGS}
    assert used == {k for k, _ in schema.COUPLING_TYPES}  # every type appears on some edge
    for key, _ in schema.COUPLING_TYPES:
        drawn = [i for i in d.scene.role(f"coupling:{key}") if isinstance(i, (Arrow, Polyline))]
        assert len(drawn) == sum(c.coupling == key for c in schema.PROCESS_COUPLINGS)
        assert len({i.style for i in drawn}) == 1
        assert d.scene.role(f"legend:{key}")
    (iterative,) = [i for i in d.scene.role("coupling:iterative") if isinstance(i, Arrow)]
    assert iterative.both


def test_the_process_is_connected_and_validation_compares_with_measured_data():
    couplings = schema.PROCESS_COUPLINGS
    touched = {c.source for c in couplings} | {c.target for c in couplings}
    assert touched == {r for r, _ in schema.PROCESS_NODES}  # no orphan node
    into_validation = {c.source for c in couplings if c.target == "validation"}
    assert {"prediction", "processing"} <= into_validation  # a prediction against the measurement
    # control acts on the experiment, not on the measurement
    assert ("validation", "experiment", "feedback") in {(c.source, c.target, c.coupling) for c in couplings}
    assert ("processing", "inverse", "calibration") in {(c.source, c.target, c.coupling) for c in couplings}


def test_every_modeling_style_is_defined_in_the_template():
    defined = set(re.findall(r"^\s*([\w ]+)/\.style=", _render.template(), re.M))
    for name in BUILDERS:
        for variant in ((), ("fusion",), ("tearing",)) if name == "integrated_modeling_space" else ((),):
            for item in getattr(vaft.diagram, name)(*variant).scene.items:
                head = item.style.split(",")[0].strip()
                if head.startswith(("im ", "coupling ", "concept ", "connector")):
                    assert head in defined, f"{name}: undefined style {head!r}"


def test_labels_off_drops_annotations_but_keeps_structure_and_legends():
    for name in BUILDERS:
        build = getattr(vaft.diagram, name)
        assert build().scene.role("note") and not build(labels=False).scene.role("note")
    assert not any(i.role.endswith(":example") for i in vaft.diagram.knowledge_basis(labels=False).scene.items)
    bare = vaft.diagram.integrated_modeling_process(labels=False)
    assert all(bare.scene.role(f"legend:{k}") for k, _ in schema.COUPLING_TYPES)
    assert all(bare.scene.role(f"node:{r}") for r, _ in schema.PROCESS_NODES)
    assert vaft.diagram.knowledge_basis(labels=False).scene.role("knowledge:semi_empirical")
    with pytest.raises(ValueError):
        vaft.diagram.knowledge_basis(labels="yes")


def test_registered_and_canonical():
    from vaft.diagram import build
    for name in BUILDERS:
        assert name in vaft.diagram.__all__
        assert f"{name}.svg" in build.CANONICAL
    assert build.CANONICAL["integrated_modeling_space_fusion.svg"] == ("integrated_modeling_space",
                                                                       {"examples": "fusion"})
