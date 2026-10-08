"""The production pipelines' lineage graphs behind /reference/pipeline-graph/ (#1647).

The generator dry-runs both canonical Snakefiles; these tests pin that what it
reports is Snakemake's topology, PipelinePaths' artifact identity and
STAGE_REPLICATION's publication semantics -- and that generating it touches
nothing but a temporary directory.  One module-scoped snapshot is shared, since
each one is six Snakemake dry runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
P2_SNAKEFILE = ROOT / "workflow" / "automatic_pipeline_2_corrective_data_update" / "Snakefile"


def _snakemake_importable() -> bool:
    try:
        import snakemake  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not (ROOT / "workflow").exists(), reason="workflow scripts are not part of the distribution"),
    pytest.mark.skipif(not _snakemake_importable(), reason="snakemake is not installed"),
]

from vaft import _pipeline_graph as pipeline_graph  # noqa: E402

PROVENANCE = {"commit": "0" * 40, "ref": "test"}


@pytest.fixture(scope="module")
def snapshot():
    return pipeline_graph.pipeline_snapshot(PROVENANCE)


@pytest.fixture(scope="module")
def paths_module():
    return pipeline_graph._load_paths_module()


def _nodes(snapshot):
    return {node["id"]: node for node in snapshot["nodes"]}


def _edges(snapshot, kind=None, view=None):
    return {
        (edge["source"], edge["target"])
        for edge in snapshot["edges"]
        if (kind is None or edge["kind"] == kind) and (view is None or view in edge["views"])
    }


def test_both_canonical_pipelines_are_documented(snapshot):
    assert [p["id"] for p in snapshot["pipelines"]] == ["routine", "corrective"]
    for pipeline in snapshot["pipelines"]:
        assert pipeline["rules"] > 5 and pipeline["jobs"] >= pipeline["rules"] - 1
        assert (ROOT / pipeline["snakefile"]).is_file()
    assert snapshot["engine"]["name"] == "snakemake" and snapshot["engine"]["version"]


def test_the_snapshot_is_deterministic(snapshot):
    assert pipeline_graph._dump(pipeline_graph.pipeline_snapshot(PROVENANCE)) == pipeline_graph._dump(snapshot)


def test_artifact_ids_never_carry_the_temporary_workspace(snapshot):
    """An id or path with the scratch directory in it is unreproducible (the Windows gate)."""
    for node in snapshot["nodes"]:
        if node["kind"] == "artifact":
            assert not re.match(r"^[A-Za-z]:/|^/", node["path"]), node["id"]
            assert "vaft-pipeline-graph-" not in node["id"], node["id"]


def test_workspace_prefixes_are_spelled_like_pipeline_paths():
    """PipelinePaths renders POSIX separators on every platform; the stripped prefixes must too."""
    from pathlib import PurePosixPath, PureWindowsPath

    assert pipeline_graph._workspace_prefixes(PureWindowsPath(r"C:\Temp\ws")) == (
        "C:/Temp/ws/filedb", "C:/Temp/ws/unused")
    assert pipeline_graph._workspace_prefixes(PurePosixPath("/tmp/ws")) == ("/tmp/ws/filedb", "/tmp/ws/unused")


def test_nodes_and_edges_are_unique_and_closed(snapshot):
    ids = [node["id"] for node in snapshot["nodes"]]
    assert len(ids) == len(set(ids))
    keys = [(e["source"], e["target"], e["kind"]) for e in snapshot["edges"]]
    assert len(keys) == len(set(keys))
    known = set(ids)
    assert all(e["source"] in known and e["target"] in known for e in snapshot["edges"])


def test_execution_edges_are_snakemake_rule_dependencies(snapshot):
    execution = _edges(snapshot, "execution", "rules")
    for edge in [
        ("routine:generate_constraints_ods", "routine:generate_kfile"),
        ("routine:run_efit_reconstruction", "routine:generate_efit_ods"),
        ("routine:generate_efit_ods", "routine:replicate_efit_to_hsds"),
        ("corrective:generate_thomson_ods", "corrective:generate_core_profiles_ods"),
        ("corrective:generate_core_profiles_ods", "corrective:generate_kinetic_efit_ods"),
    ]:
        assert edge in execution, edge
    # Snakemake never schedules across pipelines: those relations are references only.
    assert all(a.split(":")[0] == b.split(":")[0] for a, b in execution)
    assert all(a.split(":")[0] == b.split(":")[0] for a, b in _edges(snapshot, "execution", "dag"))


def test_the_rule_graph_is_parsed_from_snakemake_output():
    dot = (
        'digraph snakemake_dag {\n'
        '\t0[label = "all", color = "0.1 0.6 0.85", style="rounded"];\n'
        '\t1[label = "make\\nshot: 7", color = "0.2 0.6 0.85", style="rounded"];\n'
        '\t1 -> 0\n}\n'
    )
    assert pipeline_graph.parse_labels(dot) == {0: ["all"], 1: ["make", "shot: 7"]}
    assert pipeline_graph.parse_edges(dot) == [(1, 0)]


def test_constrained_wildcards_are_the_same_wildcard():
    assert pipeline_graph.strip_constraints("a/{product,dcon\\-peeling|rdcon}/{shot}/x") == "a/{product}/{shot}/x"
    assert pipeline_graph.strip_constraints("a/{shot,\\d{5}}/x") == "a/{shot}/x"
    with pytest.raises(pipeline_graph.PipelineGraphError):
        pipeline_graph.strip_constraints("a/{shot/x")


def test_production_parses_never_define_the_documentation_target():
    text = (ROOT / "workflow" / "automatic_pipeline_1_routine_data_processing" / "Snakefile").read_text(encoding="utf-8")
    definition = text.index("rule configured_products:")
    assert 'if config.get("documentation_graph", False):' in text[definition - 200:definition]
    assert text.index("rule all:") < definition  # never the default target
    assert pipeline_graph.PIPELINES[0].config["documentation_graph"] is True


def test_declared_scientific_references_are_exactly_the_drawn_ones(snapshot, paths_module):
    declared = {(f"corrective:{r.rule}", r.param, r.product) for r in paths_module.SCIENTIFIC_REFERENCES}
    assert {(r["rule"], r["param"], r["product"]) for r in snapshot["references"]} == declared
    for view in ("rules", "artifacts", "dag"):
        drawn = [e for e in snapshot["edges"] if e["kind"] == "scientific_reference" and view in e["views"]]
        assert drawn, view
    # The known pipeline-2 -> pipeline-1 lineage, drawn as a non-execution edge.
    assert ("routine:generate_efit_ods", "corrective:generate_core_profiles_ods") in _edges(
        snapshot, "scientific_reference", "rules")
    assert ("routine:generate_efit_ods", "corrective:generate_core_profiles_ods") not in _edges(snapshot, "execution")


def test_pipeline_2_builds_every_upstream_param_from_the_declaration(paths_module):
    """A param naming a pipeline-1 product must come from SCIENTIFIC_REFERENCES, not be inlined."""
    text = P2_SNAKEFILE.read_text(encoding="utf-8")
    product_params = set(re.findall(r"--(\w+)-product \{params\.(\w+)\}", text))
    declared_params = {r.param for r in paths_module.SCIENTIFIC_REFERENCES}
    assert {param for _, param in product_params} <= declared_params
    # and no params block names a PipelinePaths product directly, under any flag
    for block in re.findall(r"^[ \t]*params:\n((?:[ \t]+.*\n)+?)[ \t]*(?:conda|shell|resources|log|output|input):", text, re.MULTILINE):
        for line in block.splitlines():
            assert "PATHS." not in line or "scientific_reference(PATHS" in line, line
    assert "efit_for" not in text and "constraints_for" not in text
    for reference in paths_module.SCIENTIFIC_REFERENCES:
        assert hasattr(paths_module.PipelinePaths, reference.product)
    with pytest.raises(KeyError, match="not a declared scientific reference"):
        paths_module.scientific_reference(None, "generate_core_profiles_ods", "undeclared")


def test_artifacts_are_named_by_pipeline_paths(snapshot, paths_module, tmp_path):
    nodes = _nodes(snapshot)
    paths = paths_module.PipelinePaths(str(tmp_path), "filedb")
    expected = paths.shot_pattern("efit_ods")[len(str(tmp_path)) + 1:]
    artifact = nodes[f"routine:file:{expected}"]
    assert artifact["product"] == "efit_ods" and artifact["stage"] == "efit" and artifact["role"] == "product"
    record = nodes[f"routine:file:{paths.shot_pattern('replication_record', 'efit')[len(str(tmp_path)) + 1:]}"]
    assert record["role"] == "replication_record" and record["stage"] == "efit"
    identified = [n for n in snapshot["nodes"] if n["kind"] == "artifact" and n["product"]]
    assert len(identified) > 0.9 * sum(1 for n in snapshot["nodes"] if n["kind"] == "artifact")


def test_publication_follows_stage_replication(snapshot):
    from vaft.database.sources import STAGE_REPLICATION

    nodes = _nodes(snapshot)
    owns = _edges(snapshot, "owns")
    publishes = _edges(snapshot, "publishes", "publication")
    for stage, entry in STAGE_REPLICATION.items():
        node = nodes[f"stage:{stage}"]
        assert node["optional"] == entry.optional and node["produced_by"] == entry.produced_by
        assert {t for s, t in owns if s == f"stage:{stage}"} == {f"ids:{stage}:{ids}" for ids in entry.ids}
        for ids in entry.ids:
            assert ((f"ids:{stage}:{ids}", f"source:{entry.source}") in publishes) == bool(entry.source)
    # eddy solves against diagnostics IDS but owns, and so publishes, only pf_passive
    assert {t for s, t in owns if s == "stage:eddy"} == {"ids:eddy:pf_passive"}


def test_replication_is_stage_wise_and_records_are_evidence(snapshot):
    nodes = _nodes(snapshot)
    records = [n for n in snapshot["nodes"] if n["kind"] == "artifact" and n["role"] == "replication_record"]
    assert {n["stage"] for n in records} >= {"diagnostics", "eddy", "efit", "chease", "thomson"}
    produced = _edges(snapshot, "produces")
    for record in records:
        producers = [s for s, t in produced if t == record["id"]]
        assert producers and all(nodes[p]["label"].startswith("replicate_") for p in producers)
    # every replicate rule handles one stage, not the whole pipeline
    replicated_by = _edges(snapshot, "replicated_by")
    rules = [t for _, t in replicated_by]
    assert len(rules) == len(set(rules))


def test_validation_products_are_never_inputs_to_science(snapshot):
    nodes = _nodes(snapshot)
    for source, target in _edges(snapshot, "consumes"):
        if nodes[source]["role"] == "validation":
            assert nodes[target].get("aggregate"), (source, target)


def test_rules_link_to_their_definition_and_script(snapshot):
    for node in snapshot["nodes"]:
        if node["kind"] != "rule":
            continue
        line = (ROOT / node["source"]["path"]).read_text(encoding="utf-8").splitlines()[node["source"]["line"] - 1]
        assert re.search(r"\b(rule|checkpoint)\b|name:", line), (node["label"], line)
        if not node["aggregate"]:
            assert node["script"] and (ROOT / node["script"]).is_file(), node["label"]
    nodes = _nodes(snapshot)
    assert nodes["routine:preflight_raw_dumps"]["checkpoint"]
    assert nodes["routine:configured_products"]["aggregate"]


def test_generation_is_a_dry_run_without_credentials(monkeypatch, tmp_path):
    """No SQL/HSDS credential reaches Snakemake, executables are dummies, and every call is -n."""
    monkeypatch.setenv("HSDS_PASSWORD", "secret")
    monkeypatch.setenv("HS_ENDPOINT", "http://hsds.invalid")
    monkeypatch.setenv("VAFT_DB_PASSWORD", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    calls = []
    real = subprocess.run

    def recording(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return real(command, **kwargs)

    monkeypatch.setattr(pipeline_graph.subprocess, "run", recording)
    pipeline_graph._dry_run(pipeline_graph.PIPELINES[1], tmp_path, "rulegraph")
    command, environment = calls[-1]
    assert "-n" in command
    assert not any(key.startswith(("HS_", "HSDS_", "VAFT_DB_", "GITHUB_")) for key in environment)
    assert environment["HOME"].startswith(str(tmp_path))  # no ~/.hscfg or database config is reachable
    for name in ("EFIT", "CHEASE", "GPECHOME", "VAFT_FILEDB_DIR", "VAFT_DATA_DIR"):
        assert environment[name].startswith(str(tmp_path)) and not Path(environment[name]).exists()


def test_the_source_receipt_build_py_checks(snapshot):
    spec = importlib.util.spec_from_file_location("vaft_docs_build_for_pipeline_graph", DOCS / "build.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)
    pairs = build._SOURCE_PAIR.findall(pipeline_graph._dump(snapshot))
    assert {path for path, _ in pairs} >= {
        "vaft/_pipeline_graph.py", "vaft/database/sources.py",
        "workflow/automatic_pipeline_1_routine_data_processing/Snakefile",
        "workflow/automatic_pipeline_1_routine_data_processing/paths.py",
        "workflow/automatic_pipeline_2_corrective_data_update/Snakefile",
    }
    for relative, digest in pairs:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert snapshot["provenance"] == PROVENANCE


def test_check_mode_accepts_a_fresh_snapshot_and_rejects_a_stale_one(snapshot, tmp_path, monkeypatch):
    import yaml

    target = tmp_path / "pipeline_graph.yml"
    target.write_text(pipeline_graph._dump(snapshot), encoding="utf-8")
    monkeypatch.setattr(pipeline_graph, "pipeline_snapshot", lambda provenance=None: dict(snapshot, provenance=provenance))
    pipeline_graph.main(["--output", str(target), "--check"])
    stale = yaml.safe_load(target.read_text(encoding="utf-8"))
    stale["edges"] = stale["edges"][1:]
    target.write_text(pipeline_graph._dump(stale), encoding="utf-8")
    with pytest.raises(SystemExit, match="stale"):
        pipeline_graph.main(["--output", str(target), "--check"])


def test_the_docs_build_declares_the_generator_page_and_layout():
    import json

    import yaml

    generators = yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))["generators"]
    assert {"module": "vaft._pipeline_graph", "output": "_data/pipeline_graph.yml"} in generators
    page = (DOCS / "_guide" / "Pipeline_graph.md").read_text(encoding="utf-8")
    assert "permalink: /reference/pipeline-graph/" in page and 'adapter="pipeline"' in page and 'layout="dagre"' in page
    for linking in ("Pipelines.md", "Database.md", "Dependency_graph.md"):
        assert "/reference/pipeline-graph/" in (DOCS / "_guide" / linking).read_text(encoding="utf-8")
    endpoint = (DOCS / "assets" / "graph" / "pipeline-graph.json").read_text(encoding="utf-8")
    adapter = (DOCS / "assets" / "graph" / "pipeline-graph.js").read_text(encoding="utf-8")
    for field in ("pipelines", "references", "nodes", "edges", "provenance"):
        assert f"g.{field} | jsonify" in endpoint
    for field in ("pipelines", "nodes", "edges"):
        assert re.search(rf"\bdata\.{field}\b", adapter)
    manifest = yaml.safe_load((DOCS / "assets" / "lib" / "vendor.yml").read_text(encoding="utf-8"))
    pins = json.loads((DOCS / "package.json").read_text(encoding="utf-8"))["devDependencies"]
    assert pins["cytoscape-dagre"] == next(lib["version"] for lib in manifest["libraries"] if lib["name"] == "cytoscape-dagre")
