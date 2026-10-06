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
    for row in snapshot["unresolved"]:
        # an audited term never names a quantity concept (a composite like "current" may share its spelling)
        assert ontology.resolve_term(row["term"], quantities_only=True) is None, row
        assert row["origin"] and row["contexts"]
    assert any(row["term"] == "line_den" for row in snapshot["unresolved"])


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
                                           "vaft/machine_mapping/vest.yaml"}
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
