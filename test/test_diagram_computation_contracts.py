"""Computation contracts, workflow execution and provenance diagrams (#2150): semantics and evidence, not pixels."""

from pathlib import Path

import pytest

import vaft
import vaft.diagram
from vaft.diagram import _computation_contract as cc
from vaft.diagram import _workflow_execution as we
from vaft.diagram._scene import Arrow, Label, Polyline

ROOT = Path(vaft.__file__).resolve().parents[1]
BUILDERS = ("scientific_computation_contract", "computation_record_layers", "workflow_specification_and_run",
            "imas_scientific_workflow_example")


def _diagrams():
    for name in BUILDERS:
        yield name, getattr(vaft.diagram, name)()


def _text(diagram) -> str:
    return " ".join(i.text for i in diagram.scene.items if isinstance(i, Label))


def _edges(diagram):
    return {(a, b): relation for a, b, relation in diagram.model["edges"]}


def test_registered_and_canonical():
    from vaft.diagram import build
    for name in BUILDERS:
        assert name in vaft.diagram.__all__
        assert build.CANONICAL[f"{name}.svg"] == (name, {})


@pytest.mark.parametrize("name, diagram", list(_diagrams()))
def test_every_recorded_edge_is_drawn_once_with_its_relation_style(name, diagram):
    drawn = [i for i in diagram.scene.items if i.role.startswith("edge:")]
    assert sorted(i.role[len("edge:"):] for i in drawn) == sorted(f"{a}->{b}" for a, b, _ in diagram.model["edges"])
    for item in drawn:
        src, dst = item.role[len("edge:"):].split("->")
        assert item.style == cc.RELATIONS[_edges(diagram)[(src, dst)]], item.role
        assert isinstance(item, (Arrow, Polyline))


@pytest.mark.parametrize("name, diagram", list(_diagrams()))
def test_statuses_are_known_and_planned_boxes_look_planned(name, diagram):
    status = diagram.model["status"]
    assert set(diagram.model["nodes"]) <= set(status)
    assert set(status.values()) <= set(cc.STATUSES)
    for key in diagram.model["nodes"]:
        outline = diagram.scene.role(f"node:{key}")[0]
        assert (outline.style == "concept group") == (status[key] == "planned"), key


@pytest.mark.parametrize("name, diagram", list(_diagrams()))
def test_every_cited_source_exists_and_defines_the_cited_symbol(name, diagram):
    cited = [pair for pairs in diagram.model["evidence"].values() for pair in pairs]
    assert cited, name
    for path, symbol in cited:
        source = ROOT / path
        assert source.is_file(), path
        assert symbol in source.read_text(encoding="utf-8"), f"{path}: {symbol!r}"


@pytest.mark.parametrize("name, diagram", list(_diagrams()))
def test_arrows_are_long_enough_to_read(name, diagram):
    for item in diagram.scene.items:
        if isinstance(item, Arrow):
            length = ((item.end[0] - item.start[0]) ** 2 + (item.end[1] - item.start[1]) ** 2) ** 0.5
            assert length >= 0.6, f"{name}: {item.role} is {length:.2f} cm"


@pytest.mark.parametrize("name", BUILDERS)
def test_labels_false_drops_only_the_note(name):
    build = getattr(vaft.diagram, name)
    full, bare = build(), build(labels=False)
    assert full.scene.role("note") and not bare.scene.role("note")
    assert full.model == bare.model
    with pytest.raises(ValueError, match="labels"):
        build(labels="yes")


def test_the_contract_keeps_configuration_settings_and_records_apart():
    d = vaft.diagram.scientific_computation_contract()
    edges = _edges(d)
    assert edges[("inputs", "computation")] == "data"
    assert edges[("computation", "standardized_output")] == edges[("computation", "native_results")] == "data"
    # what changes the result and what does not are separate configuration, neither of them data
    assert edges[("scientific_configuration", "computation")] == "configuration"
    assert edges[("execution_settings", "computation")] == "configuration"
    assert edges[("computation", "execution_record")] == "provenance"
    assert d.model["status"]["execution_record"] == "partial"
    text = _text(d)
    assert "code.parameters" in text and "typed data" in text and "Not every Process or Code" in text
    assert "Actor" not in text  # no Actor is implied


def test_a_specification_gives_runs_with_three_artifact_classes():
    d = vaft.diagram.workflow_specification_and_run()
    edges = _edges(d)
    assert d.model["runs"] == 2
    assert edges[("specification", "orchestration")] == "configuration"
    for i in (1, 2):
        run = f"run:{i}"
        assert edges[("orchestration", run)] == "data"
        # outputs, logs and provenance are different artifact classes, each its own node
        assert {b for a, b in edges if a == run} == {f"{run}:outputs", f"{run}:logs", f"{run}:provenance",
                                                     "specification"}
        # the run-to-definition link is not recorded today: drawn as a planned reference
        assert edges[(run, "specification")] == "provenance"
    assert d.model["status"]["reference"] == "planned"
    assert len(d.scene.role("reference")) == 2 and all("planned" in i.text for i in d.scene.role("reference"))
    # the specification exists: its outline is solid, not the dashed planned look
    assert "dashed" not in d.scene.role("specification")[0].style


def test_the_record_layers_stay_separate_and_state_their_status():
    d = vaft.diagram.computation_record_layers()
    edges = _edges(d)
    assert d.model["layers"] == ("identity", "execution", "assessment")
    assert {d.model["status"][k] for k in d.model["layers"]} <= {"implemented", "partial"}
    assert d.model["status"]["identity"] == "implemented" and d.model["status"]["execution"] == "partial"
    for layer in d.model["layers"]:
        assert edges[("run", layer)] == "part" and edges[(layer, "result")] == "provenance"
        footer = d.scene.role(f"node:{layer}")[1].text
        assert "\\textit{" + d.model["status"][layer] + "}" in footer
    assert "\\#1019" in d.scene.role("node:execution")[1].text
    assert "not one score" in d.scene.role("node:assessment")[1].text
    assert "nondeterministic" in d.scene.role("node:execution")[1].text


def test_the_example_chain_reads_native_files_and_records_the_digest_links():
    d = vaft.diagram.imas_scientific_workflow_example()
    edges = _edges(d)
    assert d.model["stages"] == ("efit", "chease", "stability")
    # CHEASE reads the native EFIT g-files and the stability codes the refined ones: never a published IDS
    assert edges[("gfiles", "chease")] == edges[("refined_gfiles", "stability")] == "data"
    assert not [a for a, b in edges if a in d.model["published"]]
    for stage, native, published in zip(d.model["stages"], d.model["native"], d.model["published"]):
        assert edges[(stage, native)] == edges[(stage, published)] == "data"
        assert edges[(f"configuration:{stage}", stage)] == "configuration"
    # each native product carries its recorded digest into the cross-stage check, which is not in the pipeline yet
    assert {a for a, b in edges if b == "digest_check"} == set(d.model["native"])
    assert all(edges[(n, "digest_check")] == "provenance" for n in d.model["native"])
    assert d.model["status"]["digest_check"] == "partial"
    inputs = d.scene.role("node:inputs")[1].text
    assert "magnetics" in inputs and "pf\\_passive" in inputs and "wall" not in inputs
    text = _text(d)
    assert "input\\_equilibrium\\_sha256" in text and "verify\\_chain" in text
    assert (ROOT / d.model["pipeline"] / "Snakefile").is_file()


def test_the_single_run_qualification_figure_is_deliberately_absent():
    # scientific_credibility(organization="integrated") already draws independent axes ending in a profile
    assert not hasattr(vaft.diagram, "scientific_result_qualification")
    assert "not built" in cc.__doc__
