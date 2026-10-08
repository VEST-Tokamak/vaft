"""The implemented production pipelines as lineage graphs, for the documentation site.

``python -m vaft._pipeline_graph --output docs/_data/pipeline_graph.yml`` writes
what ``/reference/pipeline-graph/`` renders (issue #1647).  It describes what the
canonical Snakemake pipelines *do* -- which rule runs after which, which file
each produces and consumes, what each stage publishes where -- and answers it
from the owners of each fact rather than from a drawing:

* **Snakemake owns execution topology.**  Every rule and job edge is read from
  Snakemake's own ``--rulegraph``, ``--filegraph`` and ``--dag`` output for a
  small documentation configuration (one representative shot per pipeline),
  in a dry run inside a temporary directory.  Nothing here reimplements its
  scheduling.  Pipeline 1's ``all`` reaches its per-shot products only through
  checkpoints, which a dry run cannot see past, so it is asked for the
  ``configured_products`` target instead: the same products with the
  checkpoints taken as passed.
* **PipelinePaths and FileDB own artifact identity.**  A file pattern is named
  by asking :class:`PipelinePaths` (``workflow/.../paths.py``) for the pattern
  of each of its products and matching exactly; nothing parses the FileDB
  grammar out of a path.
* **``STAGE_REPLICATION`` owns publication.**  Stage -> owned IDS -> HSDS
  source edges, optionality, deferral and the source tree are copied from
  :mod:`vaft.database.sources`, so ownership is never inferred from what a
  product happens to contain.
* **Scientific references are declared, not inferred.**  Pipeline 2 consults
  pipeline 1's EFIT products through ``params`` that Snakemake deliberately
  does not schedule; those, and only those, come from
  ``paths.SCIENTIFIC_REFERENCES``, which the Snakefile builds its params from.

Generation is offline: no SQL, HSDS, solver executable or credential is
touched (the executables the configuration interpolates are dummy paths that a
dry run never runs), and nothing is written outside a temporary directory.

The snapshot records a checksum of every file it describes so
``docs/build.py`` can prove which tree it came from, and the provenance commit
its source links are pinned to.  It depends on the Snakemake version, which it
records; the documentation build uses whichever release its environment
installs (CI's, for the published site).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft._pipeline_graph --output docs/_data/pipeline_graph.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parent
WORKFLOW = _ROOT / "workflow"
PATHS_MODULE = WORKFLOW / "automatic_pipeline_1_routine_data_processing" / "paths.py"

#: Configuration values a Snakefile interpolates; a dry run never executes them.
_DUMMY_ENVIRONMENT = ("VAFT_FILEDB_DIR", "VAFT_DATA_DIR", "EFIT", "CHEASE", "GPECHOME")
#: The only variables forwarded to the dry run (an allowlist: credentials, tokens
#: and config-file locations never reach it, and HOME is the temporary workspace).
_FORWARDED_ENVIRONMENT = (
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV",
)


class PipelineGraphError(RuntimeError):
    """Snakemake could not build a documentation graph."""


@dataclass(frozen=True)
class DocumentationPipeline:
    """One canonical production pipeline and the configuration it is documented with.

    ``config`` is merged over the pipeline's own ``config.yaml``; ``base_dir``
    and ``shots`` are filled in at run time.  It switches on every optional
    branch the pipeline has (replication, plots, IMPA, every stability module)
    so the graph shows each rule the production configuration can request.
    """

    id: str
    title: str
    directory: str
    target: str
    shot: int
    config: Mapping[str, Any] = field(default_factory=dict)
    branches: tuple[str, ...] = ()


PIPELINES = (
    DocumentationPipeline(
        id="routine",
        title="Pipeline 1: routine data processing",
        directory="automatic_pipeline_1_routine_data_processing",
        target="configured_products",
        shot=39915,
        config={
            "layout": "filedb",
            "raw": {"mode": "sql"},
            "hsds": {"replicate": True},
            "impa": {"enable": True},
            "gpec": {"modules": ["dcon", "rdcon", "stride", "gpec"], "modes": [1]},
            "conda": None,
            # defines the configured_products target; production never sets it
            "documentation_graph": True,
        },
        branches=("HSDS replication", "validation plots", "IMPA", "MHD / GPEC"),
    ),
    DocumentationPipeline(
        id="corrective",
        title="Pipeline 2: corrective / kinetic update",
        directory="automatic_pipeline_2_corrective_data_update",
        target="all",
        shot=48226,
        config={"layout": "filedb", "hsds": {"replicate": True}, "conda": None},
        branches=("HSDS replication", "kinetic profiles"),
    ),
)


# --------------------------------------------------------------------------
# running Snakemake
# --------------------------------------------------------------------------


def _snakemake_version() -> str:
    try:
        import snakemake
    except ImportError as error:  # a core dependency; only a broken install lands here
        raise PipelineGraphError("the pipeline graph needs Snakemake, a core VAFT dependency") from error
    return str(getattr(snakemake, "__version__", ""))


def _environment(workspace: Path) -> dict[str, str]:
    environment = {key: os.environ[key] for key in _FORWARDED_ENVIRONMENT if key in os.environ}
    (workspace / "home").mkdir(parents=True, exist_ok=True)
    environment["HOME"] = environment["USERPROFILE"] = str(workspace / "home")
    for name in _DUMMY_ENVIRONMENT:
        environment[name] = str(workspace / "unused" / name.lower())
    # Snakemake keeps a source cache under the user cache directory; keep it here.
    environment["XDG_CACHE_HOME"] = str(workspace / "cache")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _dry_run(pipeline: DocumentationPipeline, workspace: Path, flag: str) -> str:
    """Snakemake's ``--<flag>`` output for the documentation configuration."""
    config = {**pipeline.config, "base_dir": str(workspace / "filedb"), "shots": [pipeline.shot]}
    config_path = workspace / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    directory = WORKFLOW / pipeline.directory
    result = subprocess.run(
        [
            sys.executable, "-m", "snakemake",
            "--snakefile", str(directory / "Snakefile"),
            "--configfile", str(config_path),
            "--directory", str(workspace / "run"),
            "--cores", "1", "-n",
            pipeline.target, f"--{flag}",
        ],
        cwd=str(directory),
        env=_environment(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "digraph" not in result.stdout:
        raise PipelineGraphError(
            f"snakemake --{flag} failed for {pipeline.id} (exit {result.returncode}):\n"
            + (result.stderr or result.stdout)[-3000:]
        )
    return result.stdout[result.stdout.index("digraph"):]


# --------------------------------------------------------------------------
# parsing Snakemake's DOT
# --------------------------------------------------------------------------

_EDGE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)", re.MULTILINE)
_LABEL = re.compile(r'^\s*(\d+)\s*\[\s*label\s*=\s*"((?:[^"\\]|\\.)*)"', re.MULTILINE)
_HTML_NODE = re.compile(r"^(\d+)\s*\[\s*shape=none.*?label=<(.*?)>\s*\]", re.MULTILINE | re.DOTALL)
_HTML_TITLE = re.compile(r'<font point-size="18">(.*?)</font>', re.DOTALL)
_HTML_FILE = re.compile(r'<font face="monospace">(.*?)</font>', re.DOTALL)


def parse_labels(dot: str) -> dict[int, list[str]]:
    """``node -> label lines`` of a ``--rulegraph`` or ``--dag`` digraph."""
    return {int(number): label.replace("\\n", "\n").split("\n") for number, label in _LABEL.findall(dot)}


def parse_edges(dot: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _EDGE.findall(dot)]


def parse_filegraph(dot: str) -> dict[int, dict[str, Any]]:
    """``node -> {rule, inputs, outputs}`` of a ``--filegraph`` digraph."""
    nodes = {}
    for number, label in _HTML_NODE.findall(dot):
        title = _HTML_TITLE.search(label)
        head, _, tail = label.partition("output &rarr;")
        nodes[int(number)] = {
            "rule": html.unescape(title.group(1)).strip() if title else "",
            "inputs": [html.unescape(path).strip() for path in _HTML_FILE.findall(head)],
            "outputs": [html.unescape(path).strip() for path in _HTML_FILE.findall(tail)],
        }
    return nodes


# --------------------------------------------------------------------------
# artifact identity, from PipelinePaths
# --------------------------------------------------------------------------


def _load_paths_module():
    """The workflow's ``paths.py``, which is not part of the installed package."""
    import importlib.util

    name = "vaft_pipeline_graph_paths"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PATHS_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass resolves its module while being defined
    spec.loader.exec_module(module)
    return module


def _stages() -> list[str]:
    from vaft.database.sources import STAGE_REPLICATION

    paths = _load_paths_module()
    return sorted(set(STAGE_REPLICATION) | set(paths.SHOT_STAGES))


def product_patterns(paths_module, base_dir: str) -> dict[str, dict[str, Any]]:
    """``file pattern -> {product, stage}`` for every product :class:`PipelinePaths` names.

    Each public method is asked for its own wildcard pattern through the
    helpers the Snakefiles use (``shot_pattern``, ``version_pattern``,
    ``product_pattern``, ``gpec_module_pattern``), with the stage argument
    enumerated where a method takes one.  A product a layout refuses is
    simply absent.
    """
    pipeline_paths = paths_module.PipelinePaths(base_dir, "filedb")
    stages = _stages()
    found: dict[str, dict[str, Any]] = {}

    def add(pattern: str, product: str, stage: str = "") -> None:
        found.setdefault(pattern, {"product": product, "stage": stage})

    for name, method in sorted(inspect.getmembers(type(pipeline_paths), inspect.isfunction)):
        if name.startswith("_") or name.endswith("_pattern") or name in {"from_config", "log", "static_log"}:
            continue
        parameters = list(inspect.signature(method).parameters)[1:]
        attempts: list[tuple[str, tuple, str]] = []
        if not parameters:
            attempts.append(("call", (), ""))
        elif parameters == ["shot"]:
            attempts.append(("shot_pattern", (), ""))
        elif parameters == ["machine_version"]:
            attempts.append(("version_pattern", (), ""))
        elif parameters == ["shot", "product"]:
            attempts += [("shot_pattern", (), ""), ("product_pattern", (), "")]
        elif parameters == ["shot", "code", "mode"]:
            attempts.append(("gpec_module_pattern", (), ""))
        elif parameters[:2] == ["shot", "stage"]:
            for stage in stages:
                attempts += [("shot_pattern", (stage,), stage), ("product_pattern", (stage,), stage)]
        for helper, args, stage in attempts:
            try:
                if helper == "call":
                    pattern = getattr(pipeline_paths, name)()
                elif helper == "gpec_module_pattern":
                    pattern = pipeline_paths.gpec_module_pattern(name)
                else:
                    pattern = getattr(pipeline_paths, helper)(name, *args)
            except Exception:  # noqa: BLE001 - a product this layout or stage does not have
                continue
            add(pattern, name, stage)
    return found


def strip_constraints(path: str) -> str:
    """``{product,dcon\\-peeling|rdcon}`` -> ``{product}``: a constrained wildcard is the same wildcard.

    Brace-balanced, because a constraint may itself contain braces (``\\d{5}``).
    """
    out, index = [], 0
    while index < len(path):
        if path[index] != "{":
            out.append(path[index])
            index += 1
            continue
        depth, end = 0, index
        while end < len(path):
            depth += {"{": 1, "}": -1}.get(path[end], 0)
            if depth == 0:
                break
            end += 1
        if depth:
            raise PipelineGraphError(f"unbalanced wildcard braces in {path!r}")
        out.append("{" + path[index + 1:end].split(",", 1)[0] + "}")
        index = end + 1
    return "".join(out)


def _stage_of(product: str, stage: str, stages: list[str]) -> str:
    if stage:
        return stage
    matches = [s for s in stages if product == s or product.startswith(s + "_")]
    return max(matches, key=len) if matches else ""


def _role(product: str) -> str:
    if product == "replication_record":
        return "replication_record"
    if "plot" in product:
        return "validation"
    if product.endswith("_manifest") or product.endswith("_status") or product in {
        "preflight_eligible", "preflight_excluded", "impa_selection"
    }:
        return "record"
    return "product" if product else "file"


# --------------------------------------------------------------------------
# Snakefile source locations
# --------------------------------------------------------------------------

_RULE_LINE = re.compile(r"^[ \t]*(?:rule|checkpoint)\s+(\w+)\s*:", re.MULTILINE)
_CHECKPOINT_LINE = re.compile(r"^[ \t]*checkpoint\s+(\w+)\s*:", re.MULTILINE)
_DYNAMIC_NAME = re.compile(r'^[ \t]*name:\s*(.+)$', re.MULTILINE)
_SCRIPT = re.compile(r"\{(\w+)\}/(\w+\.py)|\{(\w+)\}\s")


def rule_locations(snakefile: Path) -> tuple[dict[str, int], set[str], list[tuple[re.Pattern, int]]]:
    """Static rule lines, checkpoint names, and templates of loop-generated rule names."""
    text = snakefile.read_text(encoding="utf-8")
    line_of = lambda offset: text.count("\n", 0, offset) + 1  # noqa: E731
    static = {match.group(1): line_of(match.start()) for match in _RULE_LINE.finditer(text)}
    checkpoints = {match.group(1) for match in _CHECKPOINT_LINE.finditer(text)}
    templates = []
    for match in _DYNAMIC_NAME.finditer(text):
        literals = re.findall(r'"([^"]*)"', match.group(1))
        if literals:
            pattern = re.compile("^" + ".+".join(re.escape(piece) for piece in literals) + "$")
            templates.append((pattern, line_of(match.start())))
    return static, checkpoints, templates


def _rule_line(rule: str, static: Mapping[str, int], templates) -> int:
    if rule in static:
        return static[rule]
    for pattern, line in templates:
        if pattern.match(rule):
            return line
    return 0


def _rule_script(snakefile: Path, line: int) -> str:
    """The workflow script a rule's shell runs, relative to the repository, if any."""
    if not line:
        return ""
    lines = snakefile.read_text(encoding="utf-8").splitlines()
    block = [lines[line - 1]]
    for text in lines[line:]:
        # the next rule, or the next top-level statement, ends this one
        if re.match(r"^\s*(?:rule|checkpoint)\b", text) or re.match(r"^(?:def|if|for|[A-Za-z_]\w*\s*=)", text):
            break
        block.append(text)
    body = "\n".join(block)
    script = re.search(r"\{SCRIPT_DIR\}/(\w+\.py)", body)
    if script:
        candidate = snakefile.parent / script.group(1)
        return candidate.relative_to(_ROOT).as_posix() if candidate.is_file() else ""
    variable = re.search(r"python \{(\w+)\}", body)
    if variable:
        assignment = re.search(
            rf"^{variable.group(1)}\s*=.*?\"(\w+\.py)\"", snakefile.read_text(encoding="utf-8"), re.MULTILINE
        )
        if assignment:
            for candidate in sorted(WORKFLOW.rglob(assignment.group(1))):
                return candidate.relative_to(_ROOT).as_posix()
    return ""


def _definition_line(path: Path, needle: str) -> int:
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in text:
            return number
    return 0


# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}

    def node(self, node_id: str, **attributes) -> dict:
        node = self.nodes.setdefault(node_id, {"id": node_id, "views": []})
        views = attributes.pop("views", [])
        for key, value in attributes.items():
            node.setdefault(key, value)
        node["views"] = sorted(set(node["views"]) | set(views))
        return node

    def edge(self, source: str, target: str, kind: str, views, **attributes) -> None:
        edge = self.edges.setdefault(
            (source, target, kind), {"source": source, "target": target, "kind": kind, "views": []}
        )
        edge.update({k: v for k, v in attributes.items() if k not in edge})
        edge["views"] = sorted(set(edge["views"]) | set(views))


def _workspace_prefixes(workspace: PurePath) -> tuple[str, str]:
    """The FileDB root and the dummy-variable root, spelled as the dry run spells them.

    PipelinePaths renders every path with ``PurePosixPath`` (``C:/Users/...`` on
    Windows), so the prefixes ``tidy`` strips must be POSIX too: with ``str()``
    the Windows leg stripped nothing and every artifact kept its temporary
    directory in its id (release gate 0.8.0).
    """
    return (workspace / "filedb").as_posix(), (workspace / "unused").as_posix()


def _pipeline_graph(pipeline: DocumentationPipeline, graph: _Graph, paths_module, patterns_by_pipeline) -> dict:
    snakefile = WORKFLOW / pipeline.directory / "Snakefile"
    static, checkpoints, templates = rule_locations(snakefile)
    with tempfile.TemporaryDirectory(prefix=f"vaft-pipeline-graph-{pipeline.id}-") as scratch:
        workspace = Path(scratch).resolve()
        rule_dot = _dry_run(pipeline, workspace, "rulegraph")
        file_dot = _dry_run(pipeline, workspace, "filegraph")
        job_dot = _dry_run(pipeline, workspace, "dag")
        base, unused = _workspace_prefixes(workspace)
        patterns = product_patterns(paths_module, base)

    def tidy(path: str) -> str:
        """Relative to the FileDB root, with the documented shot as the ``{shot}`` wildcard."""
        if path.startswith(base + "/"):
            path = path[len(base) + 1:]
        elif path.startswith(unused + "/"):
            variable, _, rest = path[len(unused) + 1:].partition("/")
            path = "$" + variable.upper() + "/" + rest
        return strip_constraints(path).replace(str(pipeline.shot), "{shot}")

    identities = {tidy(pattern): identity for pattern, identity in patterns.items()}
    wildcard_identities = [
        (re.compile("^" + re.sub(r"\\\{(\w+)\\\}", "[^/]+", re.escape(pattern)) + "$"), identity)
        for pattern, identity in sorted(identities.items()) if re.search(r"\{(?!shot\})\w+\}", pattern)
    ]

    def identify(path: str) -> dict:
        if path in identities:
            return identities[path]
        for expression, identity in wildcard_identities:
            if expression.match(path):
                return identity
        return {}

    stages = _stages()
    prefix = f"{pipeline.id}:"
    rule_labels = parse_labels(rule_dot)
    rule_number = {number: lines[0] for number, lines in rule_labels.items()}
    for number, rule in sorted(rule_number.items(), key=lambda item: item[1]):
        line = _rule_line(rule, static, templates)
        graph.node(
            prefix + rule, kind="rule", pipeline=pipeline.id, label=rule,
            checkpoint=rule in checkpoints, aggregate=rule in {"all", pipeline.target},
            source={"path": _relative(snakefile), "line": line, "end_line": line},
            script=_rule_script(snakefile, line), views=["rules", "artifacts"],
        )
    for a, b in parse_edges(rule_dot):
        graph.edge(prefix + rule_number[a], prefix + rule_number[b], "execution", ["rules"])

    files = parse_filegraph(file_dot)
    outputs = sorted({tidy(path) for node in files.values() for path in node["outputs"]})
    output_expressions = [
        (re.compile("^" + re.sub(r"\\\{(\w+)\\\}", "[^/]+", re.escape(pattern)) + "$"), pattern)
        for pattern in outputs if re.search(r"\{(?!shot\})\w+\}", pattern)
    ]

    def canonical(path: str) -> str:
        """An input is the artifact of the output pattern that produces it."""
        if path in outputs:
            return path
        for expression, pattern in output_expressions:
            if expression.match(path):
                return pattern
        return path

    artifacts_of: dict[str, dict] = {}
    for node in files.values():
        rule_id = prefix + node["rule"]
        artifacts_of[node["rule"]] = node
        for direction in ("outputs", "inputs"):
            for raw in node[direction]:
                if raw.startswith("<"):  # Snakemake's placeholder for an input function
                    continue
                path = canonical(tidy(raw))
                identity = identify(path)
                product = identity.get("product", "")
                artifact = graph.node(
                    f"{prefix}file:{path}", kind="artifact", pipeline=pipeline.id,
                    label=path.rsplit("/", 1)[-1], path=path, product=product,
                    stage=_stage_of(product, identity.get("stage", ""), stages), role=_role(product),
                    views=["artifacts"],
                )
                if direction == "outputs":
                    graph.edge(rule_id, artifact["id"], "validates" if artifact["role"] == "validation" else "produces",
                               ["artifacts"])
                else:
                    graph.edge(artifact["id"], rule_id, "consumes", ["artifacts"])
    patterns_by_pipeline[pipeline.id] = {"identities": identities, "artifacts_of": {
        rule: {"outputs": [tidy(p) for p in node["outputs"]]} for rule, node in artifacts_of.items()}}

    job_labels = parse_labels(job_dot)
    counts: dict[str, int] = {}
    job_id: dict[int, str] = {}
    for number in sorted(job_labels):
        lines = job_labels[number]
        rule = lines[0]
        counts[rule] = counts.get(rule, 0) + 1
        job_id[number] = f"{prefix}job:{rule}:{counts[rule]}"
        graph.node(
            job_id[number], kind="job", pipeline=pipeline.id, rule=prefix + rule, label=rule,
            wildcards=[text.strip() for text in lines[1:] if text.strip()], views=["dag"],
        )
    for a, b in parse_edges(job_dot):
        graph.edge(job_id[a], job_id[b], "execution", ["dag"])

    return {
        "id": pipeline.id,
        "title": pipeline.title,
        "snakefile": _relative(snakefile),
        "target": pipeline.target,
        "shot": pipeline.shot,
        "branches": list(pipeline.branches),
        "config": json.loads(json.dumps(pipeline.config, sort_keys=True)),
        "rules": len(rule_number),
        "jobs": len(job_labels),
    }


def _references(graph: _Graph, paths_module, resolved: Mapping[str, Mapping]) -> list[dict]:
    """Declared non-scheduling references, drawn from the producing rule to the consulting one."""
    rows = []
    for reference in paths_module.SCIENTIFIC_REFERENCES:
        consumer = next((p for p in PIPELINES if p.id == reference.consumer), None)
        if consumer is None or f"{consumer.id}:{reference.rule}" not in graph.nodes:
            raise PipelineGraphError(
                f"{reference.consumer}:{reference.rule} is declared in SCIENTIFIC_REFERENCES but has no such rule")
        producer_pipeline = resolved[reference.pipeline]
        pattern = next(
            (path for path, identity in producer_pipeline["identities"].items()
             if identity["product"] == reference.product and not identity["stage"] and "{shot}" in path),
            None,
        )
        producer_rule = next(
            (rule for rule, node in producer_pipeline["artifacts_of"].items() if pattern in node["outputs"]),
            None,
        )
        if producer_rule is None:
            raise PipelineGraphError(f"no {reference.pipeline} rule produces {reference.product}")
        source = f"{reference.pipeline}:{producer_rule}"
        target = f"{consumer.id}:{reference.rule}"
        artifact = f"{reference.pipeline}:file:{pattern}"
        graph.edge(source, target, "scientific_reference", ["rules"], param=reference.param,
                   product=reference.product, note=reference.note)
        graph.edge(artifact, target, "scientific_reference", ["artifacts"], param=reference.param,
                   product=reference.product, note=reference.note)
        for job in [n for n in graph.nodes.values() if n["kind"] == "job" and n["rule"] == target]:
            graph.edge(artifact, job["id"], "scientific_reference", ["dag"], param=reference.param,
                       product=reference.product, note=reference.note)
            graph.nodes[artifact]["views"] = sorted(set(graph.nodes[artifact]["views"]) | {"dag"})
        rows.append({"rule": target, "param": reference.param, "product": reference.product,
                     "producer": source, "artifact": artifact, "note": reference.note})
    return rows


def _publication(graph: _Graph) -> None:
    """Stage -> owned IDS -> HSDS source, from STAGE_REPLICATION and the source catalog."""
    from vaft.database import sources

    registry = Path(sources.__file__).resolve()
    catalog = {source.name: source for source in sources.CATALOG.values()} if isinstance(
        sources.CATALOG, Mapping) else {source.name: source for source in sources.CATALOG}

    def source_node(name: str) -> str:
        node_id = f"source:{name}"
        if node_id in graph.nodes:
            return node_id
        entry = catalog.get(name)
        graph.node(
            node_id, kind="source", label=name, purpose=getattr(entry, "purpose", ""),
            sparse=bool(getattr(entry, "sparse", False)), writable=bool(getattr(entry, "writable", True)),
            parent=getattr(entry, "parent", None) or "",
            source={"path": _relative(registry), "line": _definition_line(registry, f'"{name}"'), "end_line": 0},
            views=["publication"],
        )
        parent = getattr(entry, "parent", None)
        if parent:
            graph.edge(node_id, source_node(parent), "parent", ["publication"])
        return node_id

    for stage, entry in sorted(sources.STAGE_REPLICATION.items()):
        stage_id = f"stage:{stage}"
        line = _definition_line(registry, f'"{stage}": StageReplication(')
        graph.node(
            stage_id, kind="stage", label=stage, produced_by=entry.produced_by, optional=bool(entry.optional),
            deferred_to=entry.deferred_to or "", note=entry.note, destination=entry.source or "",
            replicable=bool(entry.replicable), occurrence=int(entry.occurrence),
            source={"path": _relative(registry), "line": line, "end_line": line}, views=["publication"],
        )
        for ids in entry.ids:
            ids_id = f"ids:{stage}:{ids}"
            graph.node(ids_id, kind="ids", label=ids, stage=stage, views=["publication"])
            graph.edge(stage_id, ids_id, "owns", ["publication"])
            if entry.source:
                graph.edge(ids_id, source_node(entry.source), "publishes", ["publication"],
                           deferred=bool(entry.deferred_to))
        # the rules that replicate this stage, and the evidence each records
        for node in list(graph.nodes.values()):
            if node["kind"] != "artifact" or node["role"] != "replication_record" or node["stage"] != stage:
                continue
            for edge in [e for e in graph.edges.values() if e["target"] == node["id"] and e["kind"] == "produces"]:
                graph.node(edge["source"], views=["publication"])
                graph.node(node["id"], views=["publication"])
                graph.edge(stage_id, edge["source"], "replicated_by", ["publication"])
                graph.edge(edge["source"], node["id"], "records", ["publication"])
                if entry.source:
                    graph.edge(edge["source"], source_node(entry.source), "publishes", ["publication"])


def pipeline_snapshot(provenance: Mapping[str, str] | None = None) -> dict:
    """Every documented pipeline as one deterministic, documentation-ready mapping."""
    version = _snakemake_version()
    paths_module = _load_paths_module()
    graph = _Graph()
    resolved: dict[str, dict] = {}
    pipelines = [_pipeline_graph(pipeline, graph, paths_module, resolved) for pipeline in PIPELINES]
    references = _references(graph, paths_module, resolved)
    _publication(graph)

    from vaft.database import sources as sources_module

    files = [
        Path(__file__).resolve(), PATHS_MODULE, Path(sources_module.__file__).resolve(),
        _ROOT / "vaft" / "database" / "filedb.py",
        *[WORKFLOW / pipeline.directory / "Snakefile" for pipeline in PIPELINES],
    ]
    nodes = sorted(graph.nodes.values(), key=lambda node: node["id"])
    edges = sorted(graph.edges.values(), key=lambda edge: (edge["source"], edge["target"], edge["kind"]))
    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "engine": {"name": "snakemake", "version": version},
        "source": [
            {"path": _relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(set(files), key=_relative)
        ],
        "pipelines": pipelines,
        "references": references,
        "nodes": nodes,
        "edges": edges,
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def _dump(snapshot: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(snapshot), allow_unicode=True, sort_keys=False, default_flow_style=False, width=100)


def export_pipeline_snapshot(output: str | Path, provenance: Mapping[str, str] | None = None) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_dump(pipeline_snapshot(provenance)), encoding="utf-8")
    return destination


def check_pipeline_snapshot(existing: str | Path) -> bool:
    """Whether ``existing`` is exactly what this tree derives (its own provenance is kept)."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    recorded = yaml.load(Path(existing).read_text(encoding="utf-8"), Loader=loader)  # noqa: S506 - a safe loader
    return pipeline_snapshot(recorded.get("provenance")) == recorded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the production pipelines' lineage graphs for the documentation site.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot")
    parser.add_argument("--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot")
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 unless --output already holds what this tree derives")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            if not check_pipeline_snapshot(arguments.output):
                raise SystemExit(f"{arguments.output} is stale: regenerate it with {_GENERATOR}")
            return
        provenance = {
            key: value
            for key, value in (("commit", arguments.provenance_commit), ("ref", arguments.provenance_ref))
            if value
        }
        export_pipeline_snapshot(arguments.output, provenance or None)
    except PipelineGraphError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
