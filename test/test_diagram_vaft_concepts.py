"""VAFT framework concept diagrams (#1090): topology and implementation neutrality, not pixels."""

import re
from pathlib import Path

import pytest

import vaft.diagram
from vaft.diagram import _render
from vaft.diagram import _vaft_concepts as vc
from vaft.diagram._scene import Arrow, Label, Polyline

#: every builder the module registers, read from the registry so a new one cannot be skipped here
BUILDERS = tuple(name for name, where in vaft.diagram._LOCATIONS.items() if where == "._vaft_concepts")


def test_the_registry_lists_the_expected_builders():
    assert set(BUILDERS) == {
        "fusion_science_knowledge_lifecycle", "vaft_four_pillars", "scientific_workflow", "interoperability_layers",
        "scientific_provenance_chain", "scientific_infrastructure_principles", "machine_agnostic_architecture",
        "experiment_modeling_theory_data_network", "human_ai_interface"}


def _all_variants():
    for name in BUILDERS:
        build = getattr(vaft.diagram, name)
        yield name, build()
        yield name, build(labels=False)
    for communication in ("point_to_point", "equilibrium"):
        yield "network", vaft.diagram.experiment_modeling_theory_data_network(communication)


def _edges(diagram):
    return {(a, b) for a, b, _ in diagram.model["edges"]}


@pytest.mark.parametrize("name, diagram", list(_all_variants()))
def test_no_implementation_detail_appears_in_a_concept_diagram(name, diagram):
    text = " ".join(i.text for i in diagram.scene.items if isinstance(i, Label)) + " " + repr(diagram.model)
    for pattern in vc.IMPLEMENTATION_PATTERNS:
        assert not re.search(pattern, text), f"{name}: implementation detail {pattern!r}"


@pytest.mark.parametrize("name, diagram", list(_all_variants()))
def test_the_drawn_edges_are_the_model_edges(name, diagram):
    drawn = sorted(i.role[len("edge:"):] for i in diagram.scene.items if i.role.startswith("edge:"))
    recorded = sorted(f"{a}->{b}" for a, b, _ in diagram.model.get("edges", ()))
    assert drawn == recorded  # each edge drawn exactly once
    assert {kind for *_, kind in diagram.model.get("edges", ())} <= set(vc.EDGE_KINDS)


@pytest.mark.parametrize("name, diagram", list(_all_variants()))
def test_arrows_are_long_enough_to_read(name, diagram):
    for item in diagram.scene.items:
        if isinstance(item, Arrow):
            length = ((item.end[0] - item.start[0]) ** 2 + (item.end[1] - item.start[1]) ** 2) ** 0.5
            assert length >= (0.8 if item.both else 0.6), f"{name}: {item.role} is {length:.2f} cm"


def test_the_workflow_is_managed_phases_on_one_state_with_a_vv_feedback_loop():
    d = vaft.diagram.scientific_workflow()
    edges = _edges(d)
    kinds = {(a, b): k for a, b, k in d.model["edges"]}
    phases = d.model["phases"]
    assert phases == ("diagnostic", "reconstruction", "simulation")
    assert set(zip(phases, phases[1:])) <= edges  # an interdependent sequence
    assert "Reconstruction" in d.scene.role("node:reconstruction")[1].text
    assert d.scene.role("source:daq")[0].style == "concept source"  # sources differ from the group box
    # the standardized state is a shared contract every phase reads and writes, not a serial step
    assert all(kinds[(p, "state")] == "both" for p in phases)
    assert ("sources", "ingestion") in edges and ("ingestion", "diagnostic") in edges
    # V&V evaluates the state, qualifies the product and feeds back to the configurations
    assert ("state", "vv") in edges and ("vv", "qualified") in edges
    assert kinds[("vv", "configuration")] == "feedback"
    assert ("configuration", "ingestion") in edges
    # the product is qualified data for analysis and interpretation, not storage
    assert ("state", "qualified") in edges and ("qualified", "analysis") in edges
    assert ("analysis", "interpretation") in edges
    text = " ".join(i.text for i in d.scene.items if isinstance(i, Label))
    for phrase in ("quality flags", "uncertainty", "validation status", "applicability", "Legacy",
                   "Common Data Model (IMAS)", "Standardized scientific state"):
        assert phrase.lower() in text.lower()
    assert "database" not in text.lower()  # no state-versus-database split


def test_interoperability_keeps_native_artifacts_beside_the_standard():
    d = vaft.diagram.interoperability_layers()
    assert d.model["layers"] == tuple(k for k, _ in vc.LAYERS)
    assert d.model["layers"].index("native") < d.model["layers"].index("imas") < d.model["layers"].index("database")
    kinds = {(a, b): kind for a, b, kind in d.model["edges"]}
    assert all(kinds[(a, b)] == "both" for a, b in zip(d.model["layers"], d.model["layers"][1:]))
    assert kinds[("native", "database")] == "complement"  # complementary, not a replacement layer


def test_the_research_cycle_returns_only_to_the_experiment():
    d = vaft.diagram.fusion_science_knowledge_lifecycle()
    nodes = d.model["nodes"]
    assert nodes[0] == "experiment" and nodes[1] == "raw" and nodes[-1] == "discovery"
    assert _edges(d) == {(a, b) for a, b in zip(nodes, nodes[1:] + nodes[:1])}
    assert [b for a, b in _edges(d) if a == "discovery"] == ["experiment"]
    (feedback,) = [i for i in d.scene.items if isinstance(i, Arrow) and i.role == "edge:discovery->experiment"]
    assert feedback.style == "connector strong"
    assert "Research-learning cycle" in " ".join(i.text for i in d.scene.role("title"))
    assert "next experiment" in d.scene.role("node:discovery")[1].text  # inside the box, not beside it


def test_two_parallel_foundations_converge_on_vaft():
    d = vaft.diagram.scientific_infrastructure_principles()
    assert d.model["principles"] == ("fair", "prov", "trust")  # FAIR covers data and software at this level
    assert d.model["requirements"] == ("vvuq", "integrated", "multimachine")  # three blocks balance three
    nodes = [f"principle:{k}" for k in d.model["principles"]] + [f"requirement:{k}" for k in d.model["requirements"]]
    assert _edges(d) == {(n, "vaft") for n in nodes}  # each block converges independently
    styles = {n: d.scene.role(n)[0].style for n in nodes}
    assert len({styles[n] for n in nodes[:3]}) == len({styles[n] for n in nodes[3:]}) == 1
    assert styles[nodes[0]] != styles[nodes[3]]
    headings = {r: d.scene.role(r)[1] for r in ("heading:common", "heading:fusion")}
    assert headings["heading:common"].style == headings["heading:fusion"].style  # parallel hierarchy
    assert "FAIR4RS" in " ".join(i.text for i in d.scene.role("references"))  # named in the references only
    assert "FAIR4RS" not in " ".join(i.text for n in nodes for i in d.scene.role(n) if isinstance(i, Label))
    refs = " ".join(i.text for i in d.scene.role("references"))
    for author in ("Wilkinson", "Barker", "W3C PROV", "Lin"):
        assert author in refs
    fusion_refs = " ".join(i.text for i in d.scene.role("fusion_references"))
    for source in ("Terry", "Greenwald", "Fischer \\& Dinklage", "Imbeaux", "Meneghini", "ITER Physics Basis", "ITPA"):
        assert source in fusion_refs


def test_the_provenance_chain_records_every_step_and_traces_back():
    d = vaft.diagram.scientific_provenance_chain()
    steps = d.model["steps"]
    assert steps == ("raw", "processed", "reconstruction", "derived", "analysis")
    assert _edges(d) == set(zip(steps, steps[1:]))
    assert all(d.model["records"][s] for s in steps)
    assert "calibration" in d.model["records"]["processed"]  # metadata of a transition, not a stage
    (trace,) = [i for i in d.scene.role("trace") if isinstance(i, Polyline)]
    assert trace.points[0][0] > trace.points[-1][0]  # from the result back to the raw signal
    # quality metadata and versioned provenance are separate categories
    assert set(d.model["quality"]) == {"validation status", "uncertainty", "applicability"}
    assert not set(d.model["quality"]) & set(d.model["versioned"])
    ticks = [i for i in d.scene.role("quality") if isinstance(i, Polyline) and i.style == "concept tick"]
    assert len(ticks) == len(steps)  # it cuts across every product


def test_the_four_pillars_are_the_readme_sections_and_vest_is_not_the_foundation():
    d = vaft.diagram.vaft_four_pillars()
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for title in d.model["titles"]:
        assert f"### {title}" in readme
    assert "edges" not in d.model  # an architecture figure: the research process is the cycle's job
    (principles_box, principles_text) = d.scene.role("principles")
    for p in vc.DESIGN_PRINCIPLES:
        assert p in principles_text.text
    (tag,) = d.scene.role("reference")
    assert tag.text == "Reference implementation: VEST"
    assert "VEST" not in principles_text.text


def test_the_architecture_is_machine_agnostic_and_names_no_device():
    d = vaft.diagram.machine_agnostic_architecture()
    edges = _edges(d)
    assert all(("framework", f"research:{r}") in edges for r in d.model["research"])
    assert ("standard", "framework") in edges and ("mapping", "standard") in edges
    assert ("standard", "database") in edges
    assert d.model["domains"] == ("existing", "future")
    assert all((f"domain:{k}", "mapping") in edges for k in d.model["domains"])  # device differences end here
    text = " ".join(i.text for i in d.scene.items if isinstance(i, Label))
    for device in ("VEST", "MAST", "KSTAR", "ITER", "DIII-D", "JET"):
        assert device not in text
    assert "planned" not in text and "reference" not in text.lower()


@pytest.mark.parametrize("communication, expected", [("point_to_point", 6), ("common_model", 4), ("equilibrium", 4)])
def test_the_network_counts_adapters_as_drawn(communication, expected):
    d = vaft.diagram.experiment_modeling_theory_data_network(communication)
    n = len(d.model["nodes"])
    assert d.model["adapters"] == expected == (n * (n - 1) // 2 if communication == "point_to_point" else n)
    arrows = [i for i in d.scene.items if isinstance(i, Arrow) and i.role.startswith("edge:")]
    assert all(a.both for a in arrows)
    assert len(arrows) == expected  # the caption counts the lines a reader sees
    caption = " ".join(i.text for i in d.scene.role("caption"))
    if communication != "equilibrium":
        assert f"= {expected}$" in caption and "adapters" in caption


def test_the_network_rejects_an_unknown_communication():
    with pytest.raises(ValueError):
        vaft.diagram.experiment_modeling_theory_data_network("mesh")


def test_the_equilibrium_example_names_routes_and_references_only_there():
    example = vaft.diagram.experiment_modeling_theory_data_network("equilibrium")
    assert example.model["routes"]["experiment"] == "EFIT"
    assert "TokaMaker" in example.model["routes"]["modelling"]
    refs = " ".join(i.text for i in example.scene.role("references"))
    for name in ("Lao", "Sauter", "Hansen", "Solov'ev", "Guazzotto", "Joung"):
        assert name in refs
    hub = example.scene.role("node:hub")[1].text
    assert "IMAS Equilibrium IDS" in hub and "\\Delta^{*}\\psi" in hub  # the Grad-Shafranov equation
    for line in " ".join(i.text for i in example.scene.role("references")).split("\\\\"):
        assert line.count(";") <= 1  # one or two references a line
    for concept in ("point_to_point", "common_model"):  # the first two stay uncluttered
        assert not vaft.diagram.experiment_modeling_theory_data_network(concept).scene.role("references")


def test_humans_and_agents_collaborate_through_shared_interfaces_on_one_backend():
    d = vaft.diagram.human_ai_interface()
    edges = _edges(d)
    kinds = {(a, b): k for a, b, k in d.model["edges"]}
    assert kinds[("actor:human", "actor:agent")] == "both"  # collaboration, not two isolated classes
    assert set(d.model["interfaces"]) == {"python", "cli", "gui", "repository", "mcp"}
    assert all((f"actor:{a}", "access") in edges for a in d.model["actors"])  # both actors share one access
    for i in d.model["interfaces"]:
        assert ("access", f"interface:{i}") in edges and (f"interface:{i}", "interfaces") in edges
    assert ("interfaces", "backend") in edges  # one common connection, not one per interface
    assert [a for a, b in edges if b == "backend"] == ["interfaces"]
    assert d.scene.role("layer:interfaces")  # the three layers are labelled
    assert kinds[("framework", "repository")] == "both"
    assert set(d.model["planned"]) == {"mcp"}
    for i in d.model["interfaces"]:
        outline = d.scene.role(f"interface:{i}")[0]
        assert (outline.style == "concept group") == (i in d.model["planned"])


def test_styles_are_defined_and_labels_off_drops_only_notes():
    defined = set(re.findall(r"^\s*([\w ]+)/\.style=", _render.template(), re.M))
    for name, diagram in _all_variants():
        for item in diagram.scene.items:
            head = item.style.split(",")[0].strip()
            if head.startswith(("concept", "connector")):
                assert head in defined, f"{name}: undefined style {head!r}"
    for name in BUILDERS:
        build = getattr(vaft.diagram, name)
        full, bare = build(), build(labels=False)
        assert full.scene.role("note") and not bare.scene.role("note")
        annotation = ("note", "complement")
        assert [i for i in full.scene.items if i.role not in annotation] == list(bare.scene.items)


def test_registered_and_canonical():
    from vaft.diagram import build
    for name in BUILDERS:
        assert name in vaft.diagram.__all__
        assert f"{name}.svg" in build.CANONICAL
    # every canonical asset of this module builds with its recorded arguments, and the variant is the named one
    entries = {f: (b, kw) for f, (b, kw) in build.CANONICAL.items() if b in BUILDERS}
    assert len(entries) == len(BUILDERS) + 2  # the network's three variants
    for filename, (builder, kwargs) in entries.items():
        model = getattr(vaft.diagram, builder)(**kwargs).model
        if builder == "experiment_modeling_theory_data_network":
            variant = kwargs.get("communication", "common_model")
            assert model["communication"] == variant
            assert filename.removesuffix(".svg").endswith(variant if kwargs else "network")
