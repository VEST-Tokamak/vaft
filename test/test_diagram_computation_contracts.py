"""Computation contracts, workflow execution and provenance diagrams (#2150): semantics and evidence, not pixels."""

import ast
import re
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


def _defines(source: Path, symbol: str) -> bool:
    """Whether ``source`` defines ``symbol``: a qualified Python name, else a line that starts with it."""
    text = source.read_text(encoding="utf-8")
    if source.suffix != ".py" or not re.fullmatch(r"[A-Za-z_][\w.]*", symbol):
        return any(line.strip().startswith(symbol) for line in text.splitlines())
    scope = ast.parse(text).body
    for part in symbol.split("."):
        found = None
        for node in scope:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part:
                found = node
            elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) == part for t in node.targets):
                found = node
            elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == part:
                found = node
        if found is None:
            return False
        scope = getattr(found, "body", [])
    return True


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
    assert set(diagram.model["nodes"]) == set(status)
    assert set(status.values()) <= set(cc.STATUSES)
    for key in diagram.model["nodes"]:
        drawn = diagram.scene.role(f"node:{key}")
        if drawn:  # a frame such as the workflow specification is recorded without a node box
            assert (drawn[0].style == "concept group") == (status[key] == "planned"), key
    # every edge joins recorded nodes
    assert {n for a, b, _ in diagram.model["edges"] for n in (a, b)} <= set(diagram.model["nodes"])


@pytest.mark.parametrize("name, diagram", list(_diagrams()))
def test_every_cited_source_exists_and_defines_the_cited_symbol(name, diagram):
    evidence, generic = diagram.model["evidence"], set(diagram.model["generic"])
    for key in diagram.model["nodes"]:
        # a node that stands for code cites it; only generic placeholders cite nothing
        assert bool(evidence[key]) != (key in generic), key
        for path, symbol in evidence[key]:
            source = ROOT / path
            assert source.is_file(), path
            assert _defines(source, symbol), f"{path}: {symbol!r}"


def test_the_evidence_check_resolves_names_rather_than_substrings():
    config = ROOT / "vaft/code/efit/config.py"
    assert _defines(config, "EFITScientificConfig.sha256")
    assert not _defines(config, "EFITScientificConfig.no_such_hash")
    assert not _defines(ROOT / "vaft/validation/credibility.py", "XES")


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
        assert edges[("orchestration", run)] == "execution"
        # outputs, logs and provenance are different artifact classes, each its own node
        assert {b for a, b in edges if a == run} == {f"{run}:outputs", f"{run}:logs", f"{run}:provenance",
                                                     "specification"}
        # only the outputs are scientific data; the log and the provenance record the run
        assert edges[(run, f"{run}:outputs")] == "data"
        assert edges[(run, f"{run}:logs")] == edges[(run, f"{run}:provenance")] == "part"
        # the run-to-definition link is not recorded today: drawn as a planned reference
        assert edges[(run, "specification")] == "provenance"
    assert d.model["planned_edges"] == (("run:1", "specification"), ("run:2", "specification"))
    assert len(d.scene.role("reference")) == 2 and all("planned" in i.text for i in d.scene.role("reference"))
    # the specification exists: its outline is solid, not the dashed planned look
    assert "dashed" not in d.scene.role("specification")[0].style


def test_the_record_layers_stay_separate_and_state_their_status():
    d = vaft.diagram.computation_record_layers()
    edges = _edges(d)
    assert d.model["layers"] == ("identity", "execution", "assessment")
    # only EFIT hashes its configuration, and execution records are per adapter: both partial today
    assert all(d.model["status"][k] == "partial" for k in d.model["layers"])
    for layer in d.model["layers"]:
        assert edges[("run", layer)] == "part" and edges[(layer, "result")] == "provenance"
        footer = d.scene.role(f"node:{layer}")[1].text
        assert "\\textit{" + d.model["status"][layer] + "}" in footer
    assert "\\#1019" in d.scene.role("node:execution")[1].text
    assert "not one score" in d.scene.role("node:assessment")[1].text
    assert "nondeterministic" in d.scene.role("node:execution")[1].text
    # execution success is a precondition recorded with the execution, not an assessment axis
    assert "execution success" in d.scene.role("node:execution")[1].text
    assert "execution success" not in d.scene.role("node:assessment")[1].text
    # the bent run arrows pass above the middle layer, never through it
    top = 1.45 + 2.5
    for key in ("identity", "assessment"):
        (edge,) = d.scene.role(f"edge:run->{key}")
        assert all(abs(x) >= 3.0 or y > top for x, y in edge.points[1:-1])


def test_the_example_chain_reads_native_files_and_records_the_digest_links():
    d = vaft.diagram.imas_scientific_workflow_example()
    edges = _edges(d)
    assert d.model["stages"] == ("efit", "chease", "stability")
    # EFIT reads k-files, CHEASE the native EFIT g-files, the stability codes the refined ones: never an IDS
    assert edges[("inputs", "kfiles")] == edges[("kfiles", "efit")] == "data"
    assert ("inputs", "efit") not in edges
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
    assert "magnetics" in inputs and "pf\\_passive from the eddy stage" in inputs and "wall" not in inputs
    # the Green table is recorded apart from the scientific SHA-256, and the hash covers the initialization
    efit = d.scene.role("node:configuration:efit")[1].text
    assert "initialization" in efit and "Green table recorded apart" in efit
    # each published IDS cites the rule that builds it, not the solver rule
    rules = {k: d.model["evidence"][k][0][1] for k in d.model["published"]}
    assert rules == {"efit_ids": "rule generate_efit_ods:", "chease_ids": "rule generate_chease_ods:",
                     "stability_ids": "rule build_mhd_linear:"}
    text = _text(d)
    assert "input\\_equilibrium\\_sha256" in text and "verify\\_chain" in text
    assert (ROOT / d.model["pipeline"] / "Snakefile").is_file()


def test_the_single_run_qualification_figure_is_deliberately_absent():
    # scientific_credibility(organization="integrated") already draws independent axes ending in a profile
    assert not hasattr(vaft.diagram, "scientific_result_qualification")
    assert "not built" in cc.__doc__
