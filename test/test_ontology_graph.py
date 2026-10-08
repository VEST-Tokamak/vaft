"""The generated scientific ontology behind /reference/ontology/ (#1702).

These pin the ontology's contract rather than its current coverage: identity
is strict (aliases resolve, families never merge, unknown terms are audited,
not invented), ids are namespaced, relations come from the fixed vocabulary,
every fact comes from the registry that owns it, and generation is offline.
"""

from __future__ import annotations

import hashlib
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vaft import _ontology_graph as ontology

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROVENANCE = {"commit": "0" * 40, "ref": "test"}


@pytest.fixture(scope="module")
def snapshot():
    return ontology.ontology_snapshot(PROVENANCE)


def _nodes(snapshot):
    return {node["id"]: node for node in snapshot["nodes"]}


def _edges(snapshot, kind):
    return {(e["source"], e["target"]) for e in snapshot["edges"] if e["kind"] == kind}


def test_the_snapshot_is_deterministic(snapshot):
    assert ontology._dump(ontology.ontology_snapshot(PROVENANCE)) == ontology._dump(snapshot)


def test_ids_are_unique_and_namespaced_by_kind(snapshot):
    ids = [node["id"] for node in snapshot["nodes"]]
    assert len(ids) == len(set(ids))
    for node in snapshot["nodes"]:
        assert node["kind"] in ontology.KINDS
        assert node["id"].split(":", 1)[0] == ontology.NAMESPACES.get(node["kind"], node["kind"])
        assert node["origins"], node["id"]


def test_edges_are_unique_typed_and_closed(snapshot):
    keys = [(e["source"], e["target"], e["kind"]) for e in snapshot["edges"]]
    assert len(keys) == len(set(keys))
    known = {node["id"] for node in snapshot["nodes"]}
    for edge in snapshot["edges"]:
        assert edge["kind"] in ontology.RELATIONS
        assert edge["source"] in known and edge["target"] in known
        assert edge["origins"]


def test_every_strict_alias_resolves_to_exactly_one_canonical_concept(snapshot):
    from vaft.plot import taxonomy

    nodes = _nodes(snapshot)
    owners = {}
    for node in snapshot["nodes"]:
        for alias in node["facets"].get("aliases", []):
            assert alias not in owners, f"{alias} is an alias of both {owners[alias]} and {node['id']}"
            owners[alias] = node["id"]
    for subject in taxonomy.SUBJECTS.values():
        for alias in subject.aliases:
            assert ontology.resolve_term(alias) == ontology.node_id_for_subject(subject.name)
    assert ontology.resolve_term("ip") == ontology.resolve_term("I_p") == "concept:plasma_current"
    assert "ip" in nodes["concept:plasma_current"]["facets"]["aliases"]


def test_family_membership_is_not_synonymy(snapshot):
    assert len({ontology.resolve_term(term) for term in ("beta_n", "beta_p", "beta_t")}) == 3
    members = {source for source, target in _edges(snapshot, "member_of") if target == "concept_family:beta"}
    assert members == {"concept:beta_n", "concept:beta_p", "concept:beta_t"}
    assert ontology.resolve_term("beta") is None  # a family name is not a concept


def test_unknown_terms_are_audited_never_canonicalized(snapshot):
    for term in ("electron temperature", "Electron_Temperature", "temperature_e", "plasma current"):
        assert ontology.resolve_term(term) is None
    audited = {(row["term"], row["origin"]) for row in snapshot["unresolved"]}
    # registry terms the vocabulary does not define are listed, with where they came from ...
    assert ("line_den", "vaft.machine_mapping.registry") in audited
    assert ("rogowski_current", "vaft.machine_mapping.registry") in audited
    # ... a composite subject is not a measured quantity, so "current" is audited too ...
    assert ("current", "vaft.machine_mapping.registry") in audited
    # ... and a term the vocabulary does define never is
    assert not {term for term, _ in audited} & {"electron_density", "electron_temperature", "plasma_current", "ip"}
    assert all(row["contexts"] for row in snapshot["unresolved"])


def test_diagnostic_identity_is_the_registrys_declared_subject(snapshot):
    from vaft.machine_mapping.registry import load_diagnostic_registry

    nodes = _nodes(snapshot)
    part_of = _edges(snapshot, "part_of")
    # one record per subject is that diagnostic; the B-probe set is one node, not two
    assert nodes["diagnostic:b_field_probe"]["facets"]["registry_id"] == "magnetics.b_field_pol_probe"
    assert "diagnostic:magnetics.b_field_pol_probe" not in nodes
    assert ("diagnostic:b_field_probe", "validation:diagnostic_fit.bpol_probe") in _edges(snapshot, "assessed_by")
    # several records naming one subject are its parts
    assert ("diagnostic:charge_exchange.ces", "diagnostic:charge_exchange") in part_of
    assert ("diagnostic:magnetics.ip", "diagnostic:magnetics") in part_of
    # and a record without a subject keeps its own node rather than being matched by spelling
    plain = [rid for rid, record in load_diagnostic_registry().items() if not record.get("subject")]
    assert plain and all(f"diagnostic:{rid}" in nodes for rid in plain)


def test_an_alias_that_shares_a_bare_name_with_another_namespace_is_kept(snapshot):
    # cold review 0.8.0 delta-absorb-19-docs F2: ids are namespaced, so `tf` being both an alias
    # of machine:tf_coil and the name of ids:tf is no ambiguity -- the alias resolves to one subject.
    # (This replaces the test that pinned dropping these aliases as "also naming another node".)
    nodes = _nodes(snapshot)
    expected = {
        "coils_non_axisymmetric": "machine:coil_3d", "gyrokinetics_local": "concept:gyrokinetics",
        "nubeam": "machine:nbi", "pf_active": "machine:pf_coil", "pf_passive": "machine:passive_structure",
        "tf": "machine:tf_coil", "vacuum_field": "concept:vacuum",
    }
    unresolved = {row["term"] for row in snapshot["unresolved"]}
    for alias, node_id in expected.items():
        assert ontology.resolve_term(alias) == node_id
        assert alias in nodes[node_id]["facets"]["aliases"], alias
        assert alias not in unresolved, alias
    assert not [row for row in snapshot["unresolved"] if row["origin"] == "vaft.plot.taxonomy"]


def test_an_alias_that_identifies_two_subjects_is_dropped_and_audited():
    graph = ontology._Graph()
    origin = "test"
    graph.node("machine:a", "machine", "a", origin, aliases=["shared", "ip", "only_a"])
    graph.node("machine:b", "machine", "b", origin, aliases=["shared"])
    graph.node("ids:only_a", "ids", "only_a", origin)
    ontology._ambiguous_aliases(graph)
    # claimed by two nodes: dropped from both ...
    assert graph.nodes["machine:a"]["facets"]["aliases"] == ["only_a"]
    assert graph.nodes["machine:b"]["facets"]["aliases"] == []
    # ... resolving to a different concept than the node listing it: dropped ...
    audited = {(row["term"], context) for row in graph.unresolved.values() for context in row["contexts"]}
    assert ("shared", "alias of machine:a also identifies machine:b") in audited
    assert ("shared", "alias of machine:b also identifies machine:a") in audited
    assert ("ip", "alias of machine:a also identifies concept:plasma_current") in audited
    # ... and a bare-name collision with another namespace is not an ambiguity
    assert "only_a" not in {term for term, _ in audited}


def test_the_generator_rejects_a_registry_subject_outside_the_vocabulary(monkeypatch):
    # cold review 0.8.0 delta-absorb-19-docs F3: the vocabulary check moved out of the registry
    # loader (machine_mapping never imports vaft.plot); the generator must still enforce it.
    from vaft.machine_mapping import registry as registry_module

    records = registry_module.load_diagnostic_registry()
    records["pf_active"] = {**records["pf_active"], "subject": "not_a_subject"}
    monkeypatch.setattr(registry_module, "load_diagnostic_registry", lambda path=None: records)
    with pytest.raises(registry_module.DiagnosticRegistryError, match="not a vaft.plot.taxonomy subject"):
        ontology._diagnostics(ontology._Graph())


def test_plot_quantities_are_quantities_not_overview_subjects(snapshot):
    visualized = _edges(snapshot, "visualized_by")
    assert ("concept:current", "plot:pf_coil_time_current") not in visualized
    assert ("machine:pf_coil", "plot:pf_coil_time_current") in visualized


def test_a_measured_quantity_must_be_a_quantity():
    # "current" is a composite overview subject, not the quantity a coil measures
    assert ontology.resolve_term("current", quantities_only=True) is None
    assert ontology.resolve_term("electron_density", quantities_only=True) == "concept:electron_density"


def test_every_plot_visualizes_its_registered_subject(snapshot):
    import vaft.plot  # noqa: F401
    from vaft.plot import registry

    visualized = _edges(snapshot, "visualized_by")
    for spec in registry.specs(status=None):
        assert (ontology.node_id_for_subject(spec.subject), f"plot:{spec.name}") in visualized


def test_diagnostics_measure_what_the_registry_says(snapshot):
    measures = _edges(snapshot, "measures")
    assert ("diagnostic:thomson_scattering", "concept:electron_density") in measures
    assert ("diagnostic:thomson_scattering", "concept:electron_temperature") in measures
    assert ("diagnostic:magnetics.ip", "concept:plasma_current") in _edges(snapshot, "derives")
    assert ("diagnostic:thomson_scattering", "ids:thomson_scattering") in _edges(snapshot, "represented_by")


def test_dd_paths_are_canonical_and_carry_the_dd_resolvers_metadata(snapshot):
    from vaft.plot.backend import dd

    for node in snapshot["nodes"]:
        if node["kind"] != "dd_path":
            continue
        path = node["label"]
        assert path == ontology.canonical_dd(path)  # every index written (:)
        assert ontology.canonical_dd(dd.parse(path).canonical) == path
        info = dd.resolve(path, cross_check=False)
        assert node["facets"].get("units", "") == info.units
        assert node["facets"].get("data_type", "") == info.data_type
    represented = _edges(snapshot, "represented_by")
    assert ("concept:plasma_current", "dd:magnetics/ip(:)/data") in represented


def test_validation_and_conventions_come_from_their_registries(snapshot):
    from vaft.data import cocos
    from vaft.validation.registry import CHECKS

    nodes = _nodes(snapshot)
    provided = _edges(snapshot, "provided_by")
    for key, spec in CHECKS.items():
        assert (f"validation:{key}", f"api:{spec.provider}") in provided
    assert ("concept:plasma_current", "validation:diagnostic_fit.ip") in _edges(snapshot, "assessed_by")
    conventions = _edges(snapshot, "uses_convention")
    for name in cocos.known_codes():
        number = cocos.convention_for(name).cocos
        source = f"data_format:{name}" if f"data_format:{name}" in nodes else f"code:{name}"
        assert ((source, f"convention:cocos_{number}") in conventions) == (number is not None), name


def test_external_codes_come_from_the_ecosystem_catalog(snapshot):
    from vaft import _ecosystem

    implemented = _edges(snapshot, "implemented_by")
    for code in _ecosystem.EXTERNAL_CODES:
        assert (f"code:{code.id}", f"api:{code.adapter}") in implemented
    assert ("code:chease", "ids:equilibrium") in _edges(snapshot, "produces")


def test_reductions_connect_a_formula_only_to_quantities_the_vocabulary_resolves(snapshot):
    from vaft.formula import _taxonomy
    from vaft.formula.catalog import describe

    nodes = _nodes(snapshot)
    consumes, produces = _edges(snapshot, "consumes"), _edges(snapshot, "produces")
    audited = {(row["term"], row["origin"]) for row in snapshot["unresolved"]}
    origin = "vaft.formula._taxonomy"
    def concept(key):
        declared = _taxonomy.QUANTITIES[key].concept
        return ontology.resolve_term(declared, quantities_only=True) if declared else None

    for relations in _taxonomy.REDUCTION_FAMILIES.values():
        for relation in relations:
            for key in (*relation.sources, relation.target):
                if concept(key) is None:
                    assert (key, origin) in audited, key
            if not relation.formula:
                continue
            spec = describe(relation.formula)
            api = f"api:{spec.module}.{spec.qualname.rsplit('.', 1)[-1]}"
            target = concept(relation.target)
            sources = [concept(key) for key in relation.sources]
            assert ((api, target) in produces) == bool(target and api in nodes), relation
            for source in filter(None, sources):
                assert (api, source) in consumes, relation
    assert ("api:vaft.formula.equilibrium.shear_from_r_q", "concept:q") in consumes
    # s_hat is magnetic_shear because its Quantity says so, not because of its spelling
    assert ("api:vaft.formula.equilibrium.shear_from_r_q", "concept:magnetic_shear") in produces
    assert ("q_features", origin) in audited  # a composite declares no concept: audited, not invented


def test_a_reduction_step_outside_a_formula_derives_the_target_from_the_source(monkeypatch):
    from vaft.formula import _taxonomy

    families = {"synthetic": (
        _taxonomy.Relation(("q",), "I_p", kind="integral"),  # both ends declare a concept
        _taxonomy.Relation(("alpha",), "q_features", "stability.kadomtsev_mixing_radius"),  # neither does
    )}
    monkeypatch.setattr(_taxonomy, "REDUCTION_FAMILIES", families)
    snapshot = ontology.ontology_snapshot()
    assert ("concept:plasma_current", "concept:q") in _edges(snapshot, "derived_from")  # target <- source
    assert ("concept:q", "concept:plasma_current") not in _edges(snapshot, "derived_from")
    # a formula whose quantities the vocabulary does not know is audited, not drawn as an island
    assert "api:vaft.formula.stability.kadomtsev_mixing_radius" not in _nodes(snapshot)
    audited = {row["term"] for row in snapshot["unresolved"] if row["origin"] == "vaft.formula._taxonomy"}
    assert {"alpha", "q_features"} <= audited


def test_a_formula_node_carries_its_reduction_section(snapshot):
    from vaft.formula.catalog import describe

    for node in snapshot["nodes"]:
        if node["kind"] != "api" or node["facets"].get("role") != "formula":
            continue
        module, _, name = node["label"].rpartition(".")
        spec = describe(f"{module.rsplit('.', 1)[-1]}.{name}")
        reduction = spec.reduction
        if reduction is None:
            assert "reduction_kind" not in node["facets"]
            continue
        facets = node["facets"]
        assert "vaft.formula.catalog" in node["origins"]  # the catalog owns the Reduction section
        assert (facets["reduction_input"], facets["reduction_output"], facets["reduction_kind"],
                facets["locality"], facets["physical_role"]) == (
            list(reduction.input), reduction.output, reduction.kind, reduction.locality, reduction.role)


def test_model_contracts_and_ordering_quantities_come_from_the_orderings_registry(snapshot):
    from vaft.formula.catalog import describe
    from vaft.validation import orderings

    nodes = _nodes(snapshot)
    assumes, provided = _edges(snapshot, "assumes"), _edges(snapshot, "provided_by")
    for name, contract in orderings.CONTRACTS.items():
        node = nodes[f"model:{name}"]
        assert node["facets"]["physical_model"] == contract.physical_model
        assert {target for source, target in assumes if source == f"model:{name}"} == {
            f"ordering_quantity:{a.quantity}" for a in contract.assumptions}
    for name, quantity in orderings.ORDERING_QUANTITIES.items():
        kernels = {f"api:{describe(k).module}.{describe(k).qualname.rsplit('.', 1)[-1]}" for k in quantity.kernels}
        assert {target for source, target in provided if source == f"ordering_quantity:{name}"} == kernels
    # a contract is a model of its own registry, never folded into a taxonomy subject by spelling
    assert not {n.split(":", 1)[1] for n in nodes if n.startswith("model:")} & {
        n.split(":", 1)[1] for n in nodes if n.startswith("concept:")}


def test_generation_is_offline(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("ontology generation opened a network socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert ontology.ontology_snapshot()["nodes"]


def test_import_vaft_does_not_build_the_ontology():
    result = subprocess.run([sys.executable, "-c", "import sys, vaft; print('vaft._ontology_graph' in sys.modules)"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_the_source_receipt_build_py_checks(snapshot):
    spec = importlib.util.spec_from_file_location("vaft_docs_build_for_ontology", DOCS / "build.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)
    pairs = build._SOURCE_PAIR.findall(ontology._dump(snapshot))
    assert {path for path, _ in pairs} >= {"vaft/_ontology_graph.py", "vaft/plot/taxonomy.py",
                                           "vaft/machine_mapping/vest.yaml", "vaft/formula/_taxonomy.py",
                                           "vaft/validation/orderings.py"}
    for relative, digest in pairs:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert snapshot["provenance"] == PROVENANCE


def test_the_docs_build_declares_the_generator_page_and_endpoint():
    generators = yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))["generators"]
    assert {"module": "vaft._ontology_graph", "output": "_data/ontology_graph.yml"} in generators
    page = (DOCS / "_guide" / "Ontology.md").read_text(encoding="utf-8")
    assert "permalink: /reference/ontology/" in page and 'adapter="ontology"' in page
    endpoint = (DOCS / "assets" / "graph" / "ontology-graph.json").read_text(encoding="utf-8")
    for field in ("kinds", "relations", "views", "nodes", "edges", "provenance"):
        assert f"o.{field} | jsonify" in endpoint
    for linking in ("API_reference.md", "Formula_reference.md", "Process_reference.md", "Plot_reference.md",
                    "Dependency_graph.md", "Pipeline_graph.md"):
        assert "/reference/ontology/" in (DOCS / "_guide" / linking).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# phase 3: the Semantics docstring section and the shared quantity vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [
    ("consumes: plasma_current, b_t\nproduces: q95", (("plasma_current", "b_t"), ("q95",))),
    ("produces: tau_e", ((), ("tau_e",))),
    (None, None),
])
def test_a_semantics_section_parses_its_two_keys(text, expected):
    from vaft._semantics import parse_semantics

    semantics, errors = parse_semantics(text)
    assert errors == ()
    assert (None if semantics is None else (semantics.consumes, semantics.produces)) == expected


@pytest.mark.parametrize("text", [
    "inputs: q", "consumes q", "consumes:", "consumes: q\nconsumes: q95", "consumes: q, q",
    "produces: q-95", "\n\n",
])
def test_a_malformed_semantics_section_is_rejected_whole(text):
    from vaft._semantics import parse_semantics

    semantics, errors = parse_semantics(text)
    assert semantics is None
    assert errors or not text.strip()


def test_a_malformed_semantics_section_makes_the_formula_nonconforming():
    from vaft.formula import catalog

    def toy(x):
        """Toy.

        Parameters
        ----------
        x : float [m]
            A length.

        Returns
        -------
        float [m]
            The length.

        Semantics
        ---------
        consumes minor_radius
        """
        return x

    spec = catalog._spec(toy, "toy", "utils", "vaft.formula.utils", ())
    assert spec.semantics is None and any("Semantics" in error for error in spec.errors)


def test_every_semantics_term_is_a_quantity_of_the_vocabulary_and_becomes_an_edge(snapshot):
    from vaft.formula.catalog import list_formulas
    from vaft.process.catalog import list_processes

    consumes, produces = _edges(snapshot, "consumes"), _edges(snapshot, "produces")
    annotated = [spec for spec in [*list_formulas(), *list_processes()] if spec.semantics is not None]
    assert annotated  # the pilot set
    for spec in annotated:
        api = f"api:{spec.module}.{spec.name}"
        for term in spec.semantics.consumes:
            assert (api, ontology.resolve_term(term, quantities_only=True)) in consumes, (spec.name, term)
        for term in spec.semantics.produces:
            assert (api, ontology.resolve_term(term, quantities_only=True)) in produces, (spec.name, term)
    assert ("api:vaft.formula.equilibrium.estimated_q95", "concept:q95") in produces
    assert ("api:vaft.process.langmuir.electron_density", "concept:electron_temperature") in consumes


def test_an_unknown_semantics_term_fails_generation(monkeypatch):
    from vaft._semantics import Semantics
    from vaft.formula import catalog

    spec = catalog.describe("equilibrium.estimated_q95")
    broken = type(spec)(**{**spec.__dict__, "semantics": Semantics(produces=("estimated_q95_value",))})
    monkeypatch.setattr(catalog, "list_formulas", lambda *a, **k: [broken])
    with pytest.raises(ontology.OntologyError, match="no quantity"):
        ontology.ontology_snapshot()


def test_every_declared_reduction_concept_is_a_quantity_of_the_vocabulary():
    from vaft.formula import _taxonomy

    for key, quantity in _taxonomy.QUANTITIES.items():
        if quantity.concept is not None:
            assert ontology.resolve_term(quantity.concept, quantities_only=True) == f"concept:{quantity.concept}", key
    # representations of one quantity share one concept; a composite declares none
    assert _taxonomy.QUANTITIES["j_phi"].concept == _taxonomy.QUANTITIES["j_phi_field"].concept == "j_tor"
    assert _taxonomy.QUANTITIES["q_features"].concept is None
