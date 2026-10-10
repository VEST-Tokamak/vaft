"""Research-infrastructure concept diagrams (#1636, #1638, #1640, #1641, #1643, #1645): topology, not pixels."""

import re

import pytest

import vaft.diagram
from vaft.diagram import _render
from vaft.diagram import _research_concepts as rc
from vaft.diagram import _vaft_concepts as vc
from vaft.diagram._scene import Arrow, Label, Polyline

#: every builder the module registers, read from the registry so a new one cannot be skipped here
BUILDERS = tuple(name for name, where in vaft.diagram._LOCATIONS.items() if where == "._research_concepts")
#: the four fragmented/integrated pairs, in level order
PAIRS = ("scientific_representation", "experimental_research_infrastructure", "scientific_credibility",
         "research_modality_architecture")
#: present-day devices a machine-agnostic figure must not name (VEST is the reference implementation)
DEVICES = ("KSTAR", "DIII-D", "JET", "EAST", "ASDEX", "JT-60", "MAST", "NSTX", "TCV", "WEST", "W7-X", "SPARC")


def test_the_registry_lists_the_expected_builders():
    assert set(BUILDERS) == set(PAIRS) | {"fusion_research_ecosystem", "scientific_ownership_architecture"}


def _all_variants():
    for name in PAIRS:
        for organization in rc.ORGANIZATIONS:
            yield f"{name}:{organization}", getattr(vaft.diagram, name)(organization)
    for detail in ("full", "presentation"):
        yield f"ecosystem:{detail}", vaft.diagram.fusion_research_ecosystem(detail)
    yield "ownership", vaft.diagram.scientific_ownership_architecture()


VARIANTS = list(_all_variants())


def _text(diagram, *, skip=("references",)):
    return " ".join(i.text for i in diagram.scene.items if isinstance(i, Label) and i.role not in skip)


def _edges(diagram):
    return {(a, b) for a, b, _ in diagram.model["edges"]}


def _roles(diagram, prefix):
    return {i.role for i in diagram.scene.items if i.role.startswith(prefix)}


@pytest.mark.parametrize("name, diagram", VARIANTS)
def test_no_implementation_detail_appears(name, diagram):
    text = _text(diagram, skip=()) + " " + repr(diagram.model)
    allowed = set(diagram.model.get("format_examples", ()))
    for pattern in vc.IMPLEMENTATION_PATTERNS:
        if any(re.fullmatch(pattern, a) for a in allowed):
            continue
        assert not re.search(pattern, text), f"{name}: implementation detail {pattern!r}"


@pytest.mark.parametrize("name, diagram", VARIANTS)
def test_no_device_but_the_reference_implementation_is_named(name, diagram):
    text = _text(diagram)  # the reference footers cite the ITER Physics Basis and Research Plan by title
    for device in DEVICES + ("ITER",):
        assert not re.search(rf"\b{re.escape(device)}\b", text), f"{name}: names {device}"


@pytest.mark.parametrize("name, diagram", VARIANTS)
def test_the_drawn_edges_are_the_model_edges(name, diagram):
    drawn = sorted(i.role[len("edge:"):] for i in diagram.scene.items if i.role.startswith("edge:"))
    recorded = sorted(f"{a}->{b}" for a, b, _ in diagram.model["edges"])
    assert drawn == recorded  # each edge drawn exactly once
    assert {kind for *_, kind in diagram.model["edges"]} <= set(vc.EDGE_KINDS)


@pytest.mark.parametrize("name, diagram", VARIANTS)
def test_arrows_are_long_enough_to_read(name, diagram):
    for item in diagram.scene.items:
        if isinstance(item, Arrow):
            length = ((item.end[0] - item.start[0]) ** 2 + (item.end[1] - item.start[1]) ** 2) ** 0.5
            assert length >= (0.8 if item.both else 0.6), f"{name}: {item.role} is {length:.2f} cm"


def test_styles_are_defined_and_labels_off_drops_only_the_note():
    defined = set(re.findall(r"^\s*([\w ]+)/\.style=", _render.template(), re.M))
    for name, diagram in VARIANTS:
        for item in diagram.scene.items:
            head = item.style.split(",")[0].strip()
            if head.startswith(("concept", "connector")):
                assert head in defined, f"{name}: undefined style {head!r}"
    for name in PAIRS:
        for organization in rc.ORGANIZATIONS:
            full = getattr(vaft.diagram, name)(organization)
            bare = getattr(vaft.diagram, name)(organization, labels=False)
            assert full.scene.role("note") and not bare.scene.role("note")
            assert [i for i in full.scene.items if i.role != "note"] == list(bare.scene.items)
    for build, kwargs in ((vaft.diagram.fusion_research_ecosystem, {}),
                          (vaft.diagram.fusion_research_ecosystem, {"detail": "presentation"}),
                          (vaft.diagram.scientific_ownership_architecture, {})):
        full, bare = build(**kwargs), build(**kwargs, labels=False)
        assert [i for i in full.scene.items if i.role != "note"] == list(bare.scene.items)


@pytest.mark.parametrize("name", PAIRS)
def test_a_pair_rejects_an_unknown_organization(name):
    with pytest.raises(ValueError, match="organization"):
        getattr(vaft.diagram, name)("modular")
    with pytest.raises(ValueError, match="labels"):
        getattr(vaft.diagram, name)(labels="yes")


def test_the_ecosystem_rejects_an_unknown_detail():
    with pytest.raises(ValueError, match="detail"):
        vaft.diagram.fusion_research_ecosystem("compact")


# -- the pair grammar ------------------------------------------------------------------------------------------

def _notes(diagram, kind):
    return " ".join(i.text for i in diagram.scene.items if i.role.startswith(f"notes:{kind}:"))


@pytest.mark.parametrize("name", PAIRS)
def test_a_pair_shares_its_level_and_numbers_symptoms_and_responses_alike(name):
    fragmented, integrated = (getattr(vaft.diagram, name)(o) for o in rc.ORGANIZATIONS)
    assert fragmented.model["level"] == integrated.model["level"]
    assert fragmented.model["correspondence"] == integrated.model["correspondence"]
    keys = [k for k, *_ in fragmented.model["correspondence"]]
    assert len(set(keys)) == len(keys)
    # the problem figure lists every symptom as a numbered note, the solution figure every response
    symptoms, responses = _notes(fragmented, "symptom"), _notes(integrated, "response")
    assert not _notes(fragmented, "response") and not _notes(integrated, "symptom")
    for i, (key, symptom, term, reference, response) in enumerate(fragmented.model["correspondence"], start=1):
        assert rc._cite((i,)) + "\\," + rc._tex(symptom) in symptoms
        assert rc._cite((i,)) + "\\," + rc._tex(response) in responses
        if term:
            assert "\\textit{(" + rc._tex(term) + ")}" in symptoms
        if reference:
            assert rc._tex(reference) in symptoms
    for diagram in (fragmented, integrated):
        # every number is cited somewhere in the figure, and only valid numbers are
        placed = {n for numbers in diagram.model["tags"].values() for n in numbers}
        assert placed == set(range(1, len(keys) + 1)), f"{name} places {sorted(placed)}"
        text = _text(diagram)
        assert all(rc._cite(n) in text for n in diagram.model["tags"].values())  # each citation is drawn
    # both figures carry the same level tag and a title of their own
    level = dict(rc.LEVELS)[fragmented.model["level"]]
    for diagram in (fragmented, integrated):
        assert rc._tex(level) in diagram.scene.role("level")[0].text
    assert fragmented.scene.role("title")[0].text != integrated.scene.role("title")[0].text


@pytest.mark.parametrize("name", PAIRS)
def test_the_problem_figure_cites_its_notes_in_order(name):
    d = getattr(vaft.diagram, name)("fragmented")
    cited = [i for i in d.scene.items if isinstance(i, Label) and "textsuperscript" in i.text
             and not i.role.startswith("notes:")]
    # reading order: top to bottom, left to right; the credibility chain is read step by step, column by column
    key = (lambda i: (round(i.at[0], 1), -i.at[1])) if name == "scientific_credibility" else (
        lambda i: (-round(i.at[1], 1), i.at[0]))
    seen: list = []
    for label in sorted(cited, key=key):
        for group in re.findall(r"\\textsuperscript\{\\textcolor\{[^}]*\}\{([\d,]+)\}\}", label.text):
            seen += [int(n) for n in group.split(",") if int(n) not in seen]
    assert seen == list(range(1, len(d.model["correspondence"]) + 1))


def test_the_levels_are_in_reading_order():
    assert [k for k, _ in rc.LEVELS] == [getattr(vaft.diagram, n)().model["level"] for n in PAIRS]
    assert [tag.split(" ")[1] for _, tag in rc.LEVELS] == ["1", "2", "3", "4"]


def test_ad_hoc_links_exist_only_in_the_fragmented_figures():
    for name in PAIRS:
        fragmented, integrated = (getattr(vaft.diagram, name)(o) for o in rc.ORGANIZATIONS)
        if name != "scientific_credibility":  # one inference chain: what it lacks is evidence, not links
            assert any(rc.ADHOC_STYLE in i.style for i in fragmented.scene.items if isinstance(i, Arrow)), name
        assert not any("driftred" in i.style for i in integrated.scene.items
                       if isinstance(i, (Arrow, Polyline))), name


# -- #1636 experimental research infrastructure -------------------------------------------------------------------

def test_fragmented_research_is_silos_of_ad_hoc_paths_without_a_hub():
    d = vaft.diagram.experimental_research_infrastructure("fragmented")
    edges = _edges(d)
    assert d.model["hub"] is None
    for lane, source, _, activity, _ in rc.INFRA_LANES:
        assert d.scene.role(f"lane:{lane}")
        assert (f"source:{source}", f"step:{lane}") in edges and (f"step:{lane}", f"activity:{activity}") in edges
    # the reconstruction is copied by hand into other paths: duplicated intermediate representations
    assert {("activity:reconstruction", f"activity:{k}") for k in rc.INFRA_EQUILIBRIUM_COPIES} <= edges
    classes = {"reconstruction", "simulation", "diagnostic"} | {"profile", "physics"}
    assert classes <= set(d.model["activities"])
    text = _text(d)
    for word in ("Common Data Model", "Repository", "Research Framework"):
        assert word not in text


def test_integrated_research_meets_in_three_distinct_capabilities():
    d = vaft.diagram.experimental_research_infrastructure()
    edges = _edges(d)
    assert d.model["capabilities"] == ("repository", "cdm", "framework")
    text = _text(d)
    for title in ("Common Data Model", "FAIR Scientific Data Repository", "Research Framework"):
        assert title in text
    roles = {k: r for k, _, r, _ in rc.INFRA_CAPABILITIES}
    assert len(set(roles.values())) == 3  # three roles, not three interchangeable layers
    # every source enters through ingestion; activities reach the infrastructure through one shared connection
    assert all((f"source:{k}", "sources") in edges for k in d.model["sources"])
    assert ("sources", "infrastructure") in edges
    assert all((f"activity:{k}", "activities") in edges for k in d.model["activities"])
    assert ("activities", "infrastructure") in edges
    assert not {(a, b) for a, b in edges if a.startswith("activity:") and b.startswith("activity:")}
    assert set(d.model["sources"]) >= set(vaft.diagram.experimental_research_infrastructure(
        "fragmented").model["sources"])


# -- #1638 scientific credibility --------------------------------------------------------------------------------

def test_unqualified_inference_loses_the_named_evidence_at_plausible_steps():
    d = vaft.diagram.scientific_credibility("fragmented")
    terms = {k: t for k, _, t, *_ in d.model["correspondence"]}
    notes = _notes(d, "symptom")
    for key, term in (("coupled", "parametric entanglement"), ("agreement", "fortuitous agreement"),
                      ("unknown_source", "provenance loss"), ("derived", "primacy hierarchy"),
                      ("model_range", "domain of applicability")):
        assert terms[key] == term and term in notes
    for i, (key, symptom, *_) in enumerate(d.model["correspondence"], start=1):
        assert d.scene.role(f"symptom:{key}")[1].text == "\\small " + rc._cite((i,)) + "\\," + rc._tex(symptom)
    # physics-model applicability and a surrogate's training domain are separate failures, at the modelling step
    assert d.model["stage_of"]["model_range"] == d.model["stage_of"]["training_range"] == "modelling"
    assert d.model["stage_of"]["numerics"] == "modelling"
    keys = d.model["stages"]
    assert set(zip(keys, keys[1:])) <= {(a[len("stage:"):], b[len("stage:"):]) for a, b in _edges(d)}
    assert ("stage:comparison", "result") in _edges(d)
    assert set(d.model["stage_of"]) == {k for k, *_ in d.model["correspondence"]}


def test_qualification_keeps_its_evidence_dimensions_apart():
    d = vaft.diagram.scientific_credibility()
    edges = _edges(d)
    nodes = set(d.model["nodes"])
    assert {"provenance", "uncertainty", "assumptions", "verification", "validation", "sensitivity",
            "assessment", "qualified"} <= nodes
    # provenance is not validity, and numerical verification is not physical applicability: separate nodes
    assert all(("state", k) in edges and (k, "model") in edges for k in d.model["evidence"])
    assert all(("model", k) in edges and (k, "assessment") in edges for k in d.model["checks"])
    assert ("assessment", "qualified") in edges
    assert "not: is it valid?" in d.scene.role("evidence:provenance")[1].text
    assert "not: does the model apply?" in d.scene.role("check:verification")[1].text
    # credibility is a profile over several dimensions, never one score
    assert len(d.model["assessment"]) >= 4 and len({s for _, s in d.model["assessment"]}) >= 3
    assert "not one score" in d.scene.role("node:assessment")[1].text
    assert not re.search(r"score\s*=|valid\s*=\s*True", _text(d))


def test_the_credibility_references_sit_in_the_notes():
    notes = _notes(vaft.diagram.scientific_credibility("fragmented"), "symptom")
    for source in ("Fischer \\& Dinklage 2004", "Terry et al. 2008", "W3C PROV 2013", "Greenwald 2010"):
        assert "[" + source + "]" in notes


# -- #1640 research modality and portability ---------------------------------------------------------------------

def test_locked_in_software_copies_the_science_into_every_modality():
    d = vaft.diagram.research_modality_architecture("fragmented")
    assert set(d.model["silos"]) == {"gui", "notebook", "matlab", "hpc", "agent"}
    assert d.model["core"] is None
    assert ("entry:notebook", "entry:gui") in _edges(d) and ("entry:notebook", "entry:matlab") in _edges(d)
    symptoms = {k for k, *_ in d.model["correspondence"]}
    assert {"duplicated", "interface", "language", "environment", "machine", "unversioned"} <= symptoms


def test_interfaces_are_siblings_over_one_core_and_future_bindings_are_not_interfaces():
    d = vaft.diagram.research_modality_architecture()
    edges = _edges(d)
    interfaces = d.model["interfaces"]
    assert {"python", "jupyter", "cli", "gui", "mcp", "docs"} == set(interfaces)
    # GUI, CLI and MCP are interfaces, never part of the core; each reaches it only through the public APIs
    assert not {"gui", "cli", "mcp"} & set(d.model["core"])
    assert {(f"interface:{k}", "api") for k in interfaces} <= edges and ("api", "core") in edges
    assert d.model["agent_interfaces"] == ("mcp",)
    assert "concept actor" in d.scene.role("interface:mcp")[0].style
    # MATLAB and Julia are future bindings on the same contracts, drawn dashed and apart from the interfaces
    names = " ".join(n for _, n, *_ in rc.MODALITY_INTERFACES)
    assert all(lang not in names for lang in d.model["future"])
    future = d.scene.role("node:future")
    assert future[0].style == "concept group" and "future" in future[1].text
    # language interop and portable execution are two different axes under the core
    assert ("core", "runtimes") in edges and ("core", "execution") in edges
    assert set(d.model["platforms"]) == {"Windows", "macOS", "Linux"}
    assert any("Slurm" in e for e in d.model["execution"])
    assert "own platform limits" in d.scene.role("node:execution")[1].text  # no universal OS claim for solvers
    assert {"Git", "CI", "tests", "releases"} <= set(d.model["lifecycle"])
    assert {"formula", "process", "database", "validation", "code", "plot", "diagram",
            "machine_mapping"} == set(d.model["core"])


# -- #1641 scientific representation ------------------------------------------------------------------------------

def test_fragmented_representation_separates_four_layers_in_every_holder():
    d = vaft.diagram.scientific_representation("fragmented")
    assert d.model["layers"] == ("vocabulary", "naming", "structure", "encoding")
    for silo in d.model["silos"]:
        for layer in d.model["layers"]:
            assert d.scene.role(f"cell:{silo}:{layer}")
        # what people say is a quoted word; what the data stores is an identifier, set in monospace
        assert d.scene.role(f"cell:{silo}:vocabulary")[1].text.startswith("{\\small ``")
        assert "\\ttfamily" in d.scene.role(f"cell:{silo}:naming")[1].text
    # HDF5 is a serialization format: it appears in the encoding row and nowhere else
    hdf5 = [i.role for i in d.scene.items if isinstance(i, Label) and "HDF5" in i.text]
    assert hdf5 and all(r.endswith(":encoding") for r in hdf5)


def test_the_layered_model_keeps_meaning_names_formats_and_storage_apart():
    d = vaft.diagram.scientific_representation()
    edges = {(a, b): k for a, b, k in d.model["edges"]}
    assert d.model["layers"] == ("vocabulary", "semantic", "cdm", "mapping", "native")
    for a, b in zip(d.model["layers"], d.model["layers"][1:]):
        assert edges[(a, b)] == "both"
    # storage is beside the data model, reached from it alone: scientific identity is not a storage path
    assert edges[("cdm", "storage")] == "both"
    assert {k for k in edges if "storage" in k} == {("cdm", "storage")}
    assert "Common Data Model (IMAS)" in d.scene.role("layer:cdm")[1].text
    assert "HDF5" not in d.scene.role("layer:cdm")[1].text and "HDF5" in d.scene.role("layer:mapping")[1].text
    # an alias names one quantity; a family groups several
    assert "aliases name one quantity" in d.scene.role("layer:semantic")[1].text
    for access in ("lazy", "partial", "cached", "remote"):
        assert access in " ".join(d.model["storage"])
    assert d.model["format_examples"] == ("HDF5",)


# -- #1643 the research ecosystem --------------------------------------------------------------------------------

def test_the_detailed_ecosystem_connects_roles_activities_states_and_contexts():
    d = vaft.diagram.fusion_research_ecosystem()
    edges = {(a, b): k for a, b, k in d.model["edges"]}
    assert {"experiment", "theory", "data_ai", "software", "planning", "learners"} == set(d.model["roles"])
    assert d.model["cross_cutting"] == "learners"
    assert ("community", "activities") in edges  # roles participate in activities, no actor-to-task map
    # planning precedes data; experiment and simulation are parallel producers
    for a, b in (("question", "planning"), ("planning", "planned_shot"), ("planning", "planned_run"),
                 ("planned_shot", "experiment"), ("planned_run", "simulation"), ("experiment", "measured"),
                 ("simulation", "simulated"), ("reconstructed", "comparison"), ("predicted", "comparison"),
                 ("comparison", "qualified")):
        assert (a, b) in edges, (a, b)
    assert edges[("comparison", "question")] == "feedback"  # synthesis returns to new questions
    assert set(d.model["state_nodes"].values()) == set(d.model["states"])  # every state role is drawn
    assert {"reference", "existing", "future", "population"} <= set(d.model["contexts"])
    assert "VEST" in d.scene.role("context:reference")[1].text
    text = _text(d)
    for word in ("documentation", "tutorials", "education", "validation", "synthesis"):
        assert word in text


def test_the_presentation_ecosystem_is_a_projection_of_the_detailed_one():
    full = vaft.diagram.fusion_research_ecosystem()
    slide = vaft.diagram.fusion_research_ecosystem("presentation")
    roles, activities = set(full.model["roles"]), set(full.model["activities"])
    assert all(set(v) <= roles for v in slide.model["role_projection"].values())
    assert all(set(v) <= activities for v in slide.model["verb_projection"].values())
    assert set(slide.model["states"]) <= set(full.model["states"])
    assert all(set(v) <= activities for v in slide.model["use_projection"].values())
    # It keeps community, activities and shared states, then connects three uses without making them a sequence.
    assert {"experiment", "theory"} <= set(slide.model["roles"])
    assert slide.scene.role("node:activities") and "Shared scientific states" in slide.scene.role("node:states")[1].text
    assert slide.model["uses"] == ("experiments", "interpretation", "modelling")
    exchange = {(a, b) for a, b, _ in slide.model["edges"] if a.startswith("use:")}
    assert exchange == {("use:experiments", "use:interpretation"),
                        ("use:interpretation", "use:experiments"),
                        ("use:interpretation", "use:modelling"),
                        ("use:modelling", "use:interpretation")}
    assert slide.model["reference"] == "VEST" and slide.scene.role("reference")


# -- #1645 scientific ownership ----------------------------------------------------------------------------------

def test_ownership_bands_and_responsibility_boundaries():
    d = vaft.diagram.scientific_ownership_architecture()
    edges = {(a, b): k for a, b, k in d.model["edges"]}
    assert d.model["bands"] == ("context", "composition", "computation", "assessment", "policy")
    assert d.model["implementations"] == ("formula", "process", "code", "learned")
    # computation and data produce evidence; validation interprets it; use policy follows, optional and dashed
    assert all((k, "evidence") in edges for k in d.model["implementations"])
    assert ("database", "evidence") in edges and ("evidence", "validation") in edges
    assert ("validation", "policy") in edges
    assert "dashed" in _render.template().split("connector feedback/.style=")[1].split("\n")[0]
    assert d.scene.role("edge:validation->policy")[0].style == "connector feedback"
    assert d.scene.role("node:policy")[0].style == "concept group"
    # Study and Research never execute: they reach the workflow by a dashed reference, nothing else
    assert d.scene.role("edge:study:B->workflow")[0].style == "connector feedback"
    assert not {k for k in edges if k[0] in ("research",) or (k[0].startswith("study:") and k[1] != "workflow")}
    assert not {k for k in edges if k[0].startswith("study:") and k[1].startswith("study:")}  # membership only
    # the workflow composes computation and matures logic into the graduation rule
    assert ("workflow", "calls") in edges and ("workflow", "graduation") in edges
    owners = " ".join(o for _, o in d.model["graduation"])
    for owner in ("formula", "process", "code", "validation", "database", "learned model", "stays in the workflow"):
        assert owner in owners
    # the Actor is an optional overlay, on no edge
    assert d.model["actor_span"] == ("process", "code")
    assert not [k for k in edges if "actor" in k[0] or "actor" in k[1]]
    assert d.scene.role("node:actor")[0].style == "concept frame"
    # the data arrows leave the database and land on their targets: at a height both boxes span
    for target in ("formula", "evidence"):
        arrow = d.scene.role(f"edge:database->{target}")[0]
        for role in ("node:database", f"node:{target}"):
            outline = d.scene.role(role)[0]
            ys = [y for _, y in outline.points]
            assert min(ys) < arrow.start[1] < max(ys), (target, role)


def test_registered_and_canonical():
    from vaft.diagram import build
    for name in BUILDERS:
        assert name in vaft.diagram.__all__
        assert f"{name}.svg" in build.CANONICAL
    entries = {f: (b, kw) for f, (b, kw) in build.CANONICAL.items() if b in BUILDERS}
    assert len(entries) == 2 * len(PAIRS) + 3
    for filename, (builder, kwargs) in entries.items():
        model = getattr(vaft.diagram, builder)(**kwargs).model
        if builder in PAIRS:
            assert model["organization"] == kwargs.get("organization", "integrated")
            assert filename.removesuffix(".svg").endswith("_fragmented") == (model["organization"] == "fragmented")
        elif builder == "fusion_research_ecosystem":
            assert model["detail"] == kwargs.get("detail", "full")
