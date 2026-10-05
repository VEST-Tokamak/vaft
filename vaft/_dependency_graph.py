"""The observed import graph of VAFT, read from the source, for the documentation site.

``python -m vaft._dependency_graph --output docs/_data/dependency_graph.yml``
writes what ``/reference/dependency-graph/`` renders (issue #1646).  The page
shows the architecture that *exists* -- which module imports which -- next to
the curated conceptual one; it says nothing about which dependencies *should*
exist, and no edge here is a verdict.

Two owners supply everything in the snapshot, and this module is neither:

* `Grimp <https://grimp.readthedocs.io/>`_ builds the import graph.  It is the
  optional ``architecture`` extra, never a runtime dependency of ``vaft``, and
  there is no fallback analyser: without it the generator stops with an
  installation hint rather than produce a second, subtly different graph.
* :mod:`vaft._api_catalog` decides what is public and where it is documented.
  Every API node -- its kind, summary, API page, scientific reference page and
  source span -- is copied from that catalog's snapshot, so the graph cannot
  disagree with the API reference about what VAFT publishes.

What an edge means is deliberately narrow: *module A contains an import
statement naming module B*.  That includes imports inside functions and
``TYPE_CHECKING`` blocks, because Grimp reads statements, not execution.  It is
a source dependency, not a scientific one, and not a call graph.  External
packages are the third-party top-level names Grimp sees imported (the standard
library is dropped); external *scientific codes* (EFIT, CHEASE, ...) are never
inferred from imports.

Layers are the package hierarchy and nothing more: ``vaft.<name>.*`` belongs to
layer ``<name>`` when ``vaft.<name>`` is a public subpackage; private packages
and the modules directly under ``vaft`` are ``other``.

The snapshot is a function of the source tree (and of the provenance, when it
is passed).  It records a checksum of every file it describes so
``docs/build.py`` can prove which tree it came from, and ``--check`` compares
a fresh derivation with an existing snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft._dependency_graph --output docs/_data/dependency_graph.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parent
OTHER_LAYER = "other"

#: Standard-library top-level names that ``sys.stdlib_module_names`` lacks on some
#: supported interpreter (``requires-python >=3.10``): added in a later release.
#: Without them the external set would depend on which Python ran the generator.
_LATER_STDLIB = frozenset({"tomllib", "annotationlib", "compression"})

MISSING_GRIMP = (
    "VAFT architecture graph generation requires the optional 'architecture' dependency.\n"
    "\n"
    "Install with:\n"
    "\n"
    '    pip install -e ".[architecture]"'
)


class ArchitectureDependencyError(ImportError):
    """Grimp, the optional ``architecture`` extra, is not installed."""


def _grimp():
    try:
        import grimp
    except ImportError as error:
        raise ArchitectureDependencyError(MISSING_GRIMP) from error
    return grimp


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


def _is_private(name: str) -> bool:
    return any(part.startswith("_") for part in name.split("."))


def layer_of(module: str, packages: Iterable[str]) -> str:
    """``vaft.<x>.*`` -> ``<x>`` when ``vaft.<x>`` is a public package, else ``other``."""
    parts = module.split(".")
    if len(parts) < 2:
        return OTHER_LAYER
    top = f"vaft.{parts[1]}"
    if top in packages and not _is_private(top):
        return parts[1]
    return OTHER_LAYER


def module_path(module: str) -> Path:
    """The source file of a ``vaft`` module, from the tree this module sits in."""
    relative = Path(*module.split("."))
    package = _ROOT / relative / "__init__.py"
    return package if package.is_file() else _ROOT / relative.with_suffix(".py")


def _strongly_connected(nodes: list[str], successors: Mapping[str, list[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative so a long chain cannot hit the recursion limit."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(successors.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(successors.get(child, ()))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return components


def _api_page_url(page: str, anchor: str) -> str:
    return f"/reference/api/{page}/#{anchor}" if page else ""


def _api_nodes(api_snapshot: Mapping[str, Any]) -> tuple[list[dict], dict[str, str]]:
    """API nodes copied from the API catalog, and ``module -> its API page anchor``."""
    nodes = []
    documented_modules: dict[str, str] = {}
    for entry in api_snapshot.get("entries") or []:
        source = entry.get("source") or {}
        nodes.append({
            "id": entry["id"],
            "name": entry["name"],
            "module": entry["module"],
            "kind": entry["kind"],
            "summary": entry.get("summary", ""),
            "deprecated": bool(entry.get("deprecated")),
            "api_url": _api_page_url(entry.get("page", ""), entry["id"]),
            "reference_url": entry.get("reference", ""),
            "source": {
                "path": source.get("path", ""),
                "line": int(source.get("line") or 0),
                "end_line": int(source.get("end_line") or 0),
            },
        })
        if entry.get("page"):
            documented_modules.setdefault(entry["module"], entry["page"])
    # Only from homed entries: api-package.html heads a module (#module-<name>) only when
    # at least one object is documented there, so a module row alone has no anchor.
    nodes.sort(key=lambda node: node["id"])
    return nodes, documented_modules


def dependency_snapshot(
    api_snapshot: Mapping[str, Any] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> dict:
    """The import graph of ``vaft`` as a deterministic, documentation-ready mapping.

    ``api_snapshot`` is :func:`vaft._api_catalog.documentation_snapshot`'s
    result; it is computed here when not passed (which imports every public
    module, so tests pass a small one instead).
    """
    grimp = _grimp()
    if api_snapshot is None:
        from vaft import _api_catalog

        api_snapshot = _api_catalog.documentation_snapshot()

    graph = grimp.build_graph("vaft", include_external_packages=True, cache_dir=None)
    internal = sorted(name for name in graph.modules if name == "vaft" or name.startswith("vaft."))
    internal_set = set(internal)
    stdlib = set(sys.stdlib_module_names) | _LATER_STDLIB
    external = sorted(name for name in graph.modules if name not in internal_set and name not in stdlib)
    external_set = set(external)
    packages = {name for name in internal if module_path(name).name == "__init__.py"}

    api_nodes, documented_modules = _api_nodes(api_snapshot)
    exports: dict[str, int] = {}
    for node in api_nodes:
        exports[node["module"]] = exports.get(node["module"], 0) + 1
    module_summaries = {row["name"]: row.get("summary", "") for row in api_snapshot.get("modules") or []}

    edges: list[dict] = []
    successors: dict[str, list[str]] = {}
    for importer in internal:
        for imported in sorted(graph.find_modules_directly_imported_by(importer)):
            if imported in internal_set:
                kind = "internal"
            elif imported in external_set:
                kind = "external"
            else:
                continue
            details = graph.get_import_details(importer=importer, imported=imported)
            lines = sorted({int(detail["line_number"]) for detail in details})
            edges.append({"source": importer, "target": imported, "kind": kind, "lines": lines})
            if kind == "internal" and imported != importer:
                successors.setdefault(importer, []).append(imported)

    imports = {name: 0 for name in internal}
    imported_by = {name: 0 for name in internal + external}
    for edge in edges:
        imports[edge["source"]] += 1
        imported_by[edge["target"]] += 1

    cycles = [component for component in _strongly_connected(internal, successors) if len(component) > 1]
    cycles.sort(key=lambda component: (-len(component), component[0]))
    cycle_of = {member: number for number, component in enumerate(cycles, 1) for member in component}

    nodes: list[dict] = []
    for name in internal:
        path = module_path(name)
        line_count = len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0
        page = documented_modules.get(name, "")
        nodes.append({
            "id": name,
            "kind": "package" if name in packages else "module",
            "layer": layer_of(name, packages),
            "parent": name.rpartition(".")[0],
            "public": not _is_private(name),
            "summary": module_summaries.get(name, ""),
            "api_url": _api_page_url(page, f"module-{name}"),
            "exports": exports.get(name, 0),
            "source": {"path": _relative(path), "line": 1, "end_line": line_count},
            "imports": imports[name],
            "imported_by": imported_by[name],
            "cycle": cycle_of.get(name, 0),
        })
    for name in external:
        nodes.append({
            "id": name,
            "kind": "external",
            "layer": "external",
            "parent": "",
            "public": True,
            "summary": "",
            "api_url": "",
            "exports": 0,
            "source": {"path": "", "line": 0, "end_line": 0},
            "imports": 0,
            "imported_by": imported_by[name],
            "cycle": 0,
        })

    layers = sorted({node["layer"] for node in nodes if node["kind"] != "external"} - {OTHER_LAYER})
    sources = sorted(path for path in _PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "engine": {"name": "grimp", "version": getattr(grimp, "__version__", "")},
        "source": [
            {"path": _relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(sources, key=_relative)
        ],
        "layers": [*layers, OTHER_LAYER],
        "nodes": nodes,
        "edges": edges,
        "cycles": cycles,
        "api": api_nodes,
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def _dump(snapshot: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(snapshot), allow_unicode=True, sort_keys=False, default_flow_style=False, width=100)


def export_dependency_snapshot(
    output: str | Path,
    provenance: Mapping[str, str] | None = None,
    api_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_dump(dependency_snapshot(api_snapshot, provenance)), encoding="utf-8")
    return destination


def check_dependency_snapshot(
    existing: str | Path,
    api_snapshot: Mapping[str, Any] | None = None,
) -> bool:
    """Whether ``existing`` is exactly what this tree derives (its own provenance is kept)."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    recorded = yaml.load(Path(existing).read_text(encoding="utf-8"), Loader=loader)  # noqa: S506 - a safe loader
    fresh = dependency_snapshot(api_snapshot, recorded.get("provenance"))
    # the snapshot holds only mappings, lists, strings and ints, so it round-trips through YAML exactly
    return fresh == recorded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the observed VAFT import graph for the documentation site.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot")
    parser.add_argument("--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot")
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 unless --output already holds what this tree derives")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            if not check_dependency_snapshot(arguments.output):
                raise SystemExit(f"{arguments.output} is stale: regenerate it with {_GENERATOR}")
            return
        provenance = {
            key: value
            for key, value in (("commit", arguments.provenance_commit), ("ref", arguments.provenance_ref))
            if value
        }
        export_dependency_snapshot(arguments.output, provenance or None)
    except ArchitectureDependencyError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
