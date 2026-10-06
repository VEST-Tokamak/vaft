"""The observed import graph behind /reference/dependency-graph/ (#1646).

The generator is Grimp plus the API catalog; these tests pin what the snapshot
claims (every module, its direct imports and where they are written, its
layer, its public API and links) and that Grimp stays an optional extra.  The
API catalog is replaced by a small stand-in so the module runs in well under a
second; its own tests cover it.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

grimp = pytest.importorskip("grimp")

from vaft import _dependency_graph as dependency_graph  # noqa: E402

API_STAND_IN = {
    "modules": [
        {"name": "vaft.formula", "summary": "Explicit relations.", "page": "formula"},
    ],
    "entries": [
        {"id": "vaft._dependency_graph.layer_of", "name": "layer_of", "module": "vaft._dependency_graph",
         "page": "", "kind": "function", "summary": "Private stand-in.", "deprecated": "",
         "reference": "", "source": {"path": "vaft/_dependency_graph.py", "line": 80, "end_line": 88}},
        {"id": "vaft.formula.stand_in", "name": "stand_in", "module": "vaft.formula", "page": "formula",
         "kind": "function", "summary": "Documented stand-in.", "deprecated": "",
         "reference": "/reference/formula/geometry/#stand_in",
         "source": {"path": "vaft/formula/__init__.py", "line": 1, "end_line": 2}},
    ],
}
PROVENANCE = {"commit": "0" * 40, "ref": "test"}


@pytest.fixture(scope="module")
def snapshot():
    return dependency_graph.dependency_snapshot(API_STAND_IN, PROVENANCE)


def _nodes(snapshot):
    return {node["id"]: node for node in snapshot["nodes"]}


def _source_modules() -> set[str]:
    found = set()
    for path in (ROOT / "vaft").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(ROOT).with_suffix("").parts
        found.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return found


def test_the_snapshot_is_deterministic(snapshot):
    again = dependency_graph.dependency_snapshot(API_STAND_IN, PROVENANCE)
    assert dependency_graph._dump(again) == dependency_graph._dump(snapshot)


def test_every_source_module_is_a_node_and_nothing_else_is(snapshot):
    internal = {node["id"] for node in snapshot["nodes"] if node["kind"] != "external"}
    assert internal == _source_modules()


def test_nodes_and_edges_are_unique(snapshot):
    ids = [node["id"] for node in snapshot["nodes"]]
    assert len(ids) == len(set(ids))
    pairs = [(edge["source"], edge["target"]) for edge in snapshot["edges"]]
    assert len(pairs) == len(set(pairs))
    known = set(ids)
    assert all(source in known and target in known for source, target in pairs)


def test_edges_are_the_import_statements_grimp_reads(snapshot):
    nodes = _nodes(snapshot)
    graph = grimp.build_graph("vaft", include_external_packages=True, cache_dir=None)
    for edge in snapshot["edges"]:
        assert edge["kind"] in {"internal", "external"}
        assert graph.direct_import_exists(importer=edge["source"], imported=edge["target"])
        text = (ROOT / nodes[edge["source"]]["source"]["path"]).read_text(encoding="utf-8").splitlines()
        for line in edge["lines"]:
            assert "import" in text[line - 1], (edge, line)


def test_a_known_function_level_import_is_an_edge_at_its_line(snapshot):
    # _api_catalog.source_of imports vaft._docstring inside the function body.
    edge = next(e for e in snapshot["edges"]
                if e["source"] == "vaft._api_catalog" and e["target"] == "vaft._docstring")
    source = (ROOT / "vaft" / "_api_catalog.py").read_text(encoding="utf-8").splitlines()
    assert any("from vaft._docstring import" in source[line - 1] for line in edge["lines"])


def test_layers_are_the_package_hierarchy(snapshot):
    nodes = _nodes(snapshot)
    assert nodes["vaft.formula"]["layer"] == "formula"
    assert all(node["layer"] == "process" for node in snapshot["nodes"] if node["id"].startswith("vaft.process."))
    assert nodes["vaft"]["layer"] == "other"
    assert nodes["vaft.compat"]["layer"] == "other"  # a module, not a package
    assert nodes["vaft._dependency_graph"]["layer"] == "other"  # private
    assert all(node["layer"] in snapshot["layers"] for node in snapshot["nodes"] if node["kind"] != "external")
    public_packages = {
        path.parent.name for path in (ROOT / "vaft").glob("*/__init__.py") if not path.parent.name.startswith("_")
    }
    assert set(snapshot["layers"]) - {"other"} <= public_packages


def test_externals_are_third_party_imports_only(snapshot):
    external = {node["id"] for node in snapshot["nodes"] if node["kind"] == "external"}
    assert "numpy" in external and "omas" in external
    assert not external & (set(sys.stdlib_module_names) | dependency_graph._LATER_STDLIB)
    assert not external & {"tomllib", "annotationlib"}  # stdlib on some supported Python
    # scientific codes are executables, never inferred from imports
    assert not external & {"efit", "chease", "gpec", "nubeam", "EFIT", "CHEASE"}
    assert all(not edge["target"].startswith("vaft") for edge in snapshot["edges"] if edge["kind"] == "external")


def test_api_nodes_and_links_come_from_the_api_catalog(snapshot):
    api = {entry["id"]: entry for entry in snapshot["api"]}
    assert set(api) == {entry["id"] for entry in API_STAND_IN["entries"]}
    documented = api["vaft.formula.stand_in"]
    assert documented["api_url"] == "/reference/api/formula/#vaft.formula.stand_in"
    assert documented["reference_url"] == "/reference/formula/geometry/#stand_in"
    assert documented["source"] == {"path": "vaft/formula/__init__.py", "line": 1, "end_line": 2}
    assert api["vaft._dependency_graph.layer_of"]["api_url"] == ""  # no page, no link
    nodes = _nodes(snapshot)
    assert nodes["vaft.formula"]["api_url"] == "/reference/api/formula/#module-vaft.formula"
    assert nodes["vaft.formula"]["summary"] == "Explicit relations."
    assert nodes["vaft.formula"]["exports"] == 1
    assert nodes["vaft.process"]["api_url"] == ""


def test_module_sources_span_the_file(snapshot):
    for node in snapshot["nodes"]:
        if node["kind"] == "external":
            continue
        path = ROOT / node["source"]["path"]
        assert path.is_file()
        assert node["source"]["end_line"] == len(path.read_text(encoding="utf-8").splitlines())


def test_metrics_and_cycles_are_consistent(snapshot):
    nodes = _nodes(snapshot)
    for node_id, node in nodes.items():
        assert node["imports"] == sum(1 for e in snapshot["edges"] if e["source"] == node_id)
        assert node["imported_by"] == sum(1 for e in snapshot["edges"] if e["target"] == node_id)
    for number, members in enumerate(snapshot["cycles"], 1):
        assert len(members) > 1
        assert all(nodes[member]["cycle"] == number for member in members)


def test_provenance_and_the_source_receipt_build_py_checks(snapshot):
    assert snapshot["provenance"] == PROVENANCE
    dumped = dependency_graph._dump(snapshot)
    spec = importlib.util.spec_from_file_location("vaft_docs_build_for_graph", DOCS / "build.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)
    pairs = build._SOURCE_PAIR.findall(dumped)
    assert pairs and len(pairs) == len(snapshot["source"])
    for relative, digest in pairs:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_check_mode_accepts_a_fresh_snapshot_and_rejects_a_stale_one(tmp_path, snapshot, monkeypatch):
    target = tmp_path / "dependency_graph.yml"
    target.write_text(dependency_graph._dump(snapshot), encoding="utf-8")
    real = dependency_graph.dependency_snapshot
    monkeypatch.setattr(dependency_graph, "dependency_snapshot",
                        lambda api=None, provenance=None: real(API_STAND_IN, provenance))
    dependency_graph.main(["--output", str(target), "--check"])
    stale = yaml.safe_load(target.read_text(encoding="utf-8"))
    stale["edges"] = stale["edges"][1:]
    target.write_text(dependency_graph._dump(stale), encoding="utf-8")
    with pytest.raises(SystemExit, match="stale"):
        dependency_graph.main(["--output", str(target), "--check"])


def test_without_grimp_vaft_imports_and_the_generator_says_how_to_install(tmp_path):
    script = (
        "import sys\n"
        "sys.modules['grimp'] = None\n"
        "import vaft, vaft.formula\n"
        "from vaft import _dependency_graph as g\n"
        "try:\n"
        "    g.dependency_snapshot({'entries': [], 'modules': []})\n"
        "except g.ArchitectureDependencyError as error:\n"
        "    print(error)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "optional 'architecture' dependency" in result.stdout
    assert 'pip install -e ".[architecture]"' in result.stdout


def test_import_vaft_does_not_load_grimp():
    result = subprocess.run([sys.executable, "-c", "import sys, vaft; print('grimp' in sys.modules)"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_grimp_is_an_optional_extra_not_a_runtime_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any(re.match(r"grimp\b", requirement) for requirement in project["dependencies"])
    assert project["optional-dependencies"]["architecture"] == ["grimp>=3.17,<4"]
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert workflow.count('pip install -e ".[dev,architecture]"') == 2


def test_the_docs_build_declares_the_generator_and_the_page():
    generators = yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))["generators"]
    assert {"module": "vaft._dependency_graph", "output": "_data/dependency_graph.yml"} in generators
    page = (DOCS / "_guide" / "Dependency_graph.md").read_text(encoding="utf-8")
    assert "permalink: /reference/dependency-graph/" in page
    assert 'adapter="dependency"' in page
    navigation = yaml.safe_load((DOCS / "_data" / "navigation.yml").read_text(encoding="utf-8"))
    urls = [item["url"] for section in navigation["sections"] for item in section["items"]]
    assert "/reference/dependency-graph/" in urls


def test_the_vendored_renderer_matches_its_pin():
    manifest = yaml.safe_load((DOCS / "assets" / "lib" / "vendor.yml").read_text(encoding="utf-8"))
    package = json.loads((DOCS / "package.json").read_text(encoding="utf-8"))
    pins = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    include = (DOCS / "_includes" / "graph" / "viewer.html").read_text(encoding="utf-8")
    for library in manifest["libraries"]:
        assert pins[library["name"]] == library["version"]
        assert library["version"] in library["file"]
        path = DOCS / "assets" / "lib" / library["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == library["sha256"]
        assert f"/assets/lib/{library['file']}" in include
    assert "unpkg.com" not in include and "jsdelivr" not in include


def test_the_json_endpoint_serves_every_field_the_adapter_reads():
    endpoint = (DOCS / "assets" / "graph" / "dependency-graph.json").read_text(encoding="utf-8")
    adapter = (DOCS / "assets" / "graph" / "dependency-graph.js").read_text(encoding="utf-8")
    for field in ("layers", "nodes", "edges", "cycles", "api", "provenance"):
        assert f"g.{field} | jsonify" in endpoint
    for field in ("layers", "nodes", "edges", "cycles", "api"):
        assert re.search(rf"\bdata\.{field}\b", adapter), f"the adapter no longer reads {field}"
    assert "site.data.dependency_graph" in endpoint
    assert "g.source" not in endpoint  # the build receipt is not served


def test_the_generator_module_has_no_import_time_dependency_on_grimp():
    tree = ast.parse((ROOT / "vaft" / "_dependency_graph.py").read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in top_level for alias in node.names}
    names |= {node.module for node in top_level if isinstance(node, ast.ImportFrom) and node.module}
    assert "grimp" not in names


def test_vendored_files_are_checked_out_byte_exact():
    """A CRLF checkout changes the pinned hash (Windows CI): every vendored file is ``-text``."""
    manifest = yaml.safe_load((DOCS / "assets" / "lib" / "vendor.yml").read_text(encoding="utf-8"))
    paths = [f"docs/assets/lib/{library['file']}" for library in manifest["libraries"]]
    result = subprocess.run(["git", "check-attr", "text", "--", *paths], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    for line in result.stdout.splitlines():
        assert line.endswith(": text: unset"), line
