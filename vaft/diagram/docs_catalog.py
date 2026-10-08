"""The canonical diagrams as a deterministic snapshot for the documentation site.

``python -m vaft.diagram.docs_catalog --output docs/_data/diagram_catalog.yml``
writes what ``/reference/diagram/`` renders: every public builder of
:mod:`vaft.diagram`, every committed asset :data:`vaft.diagram.build.CANONICAL`
declares (builder, arguments, SVG path, the hashes ``manifest.json`` records),
and the ``vaft.formula`` functions each builder draws on.

It only *reads* :mod:`vaft.diagram.build` and the committed manifest; it
builds no diagram, needs no TeX, and does not judge freshness --
``python -m vaft.diagram.build --check`` does that.

The formula list is read from the source, not from a declaration: the names a
builder module imports from ``vaft.formula``, restricted to those reachable
from the builder's own body through the module-level functions and classes it
references.  A builder that resolves a formula some other way (by string, or
through another module) is not seen, so the list is labelled "referenced"
rather than "used".
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import re
from collections.abc import Mapping
from pathlib import Path

from vaft._docstring import source_span

from ._render import SAVE_FORMATS

__all__ = [
    "ASSET_DIR",
    "SCHEMA_VERSION",
    "builder_names",
    "documentation_snapshot",
    "export_documentation_snapshot",
    "formula_references",
    "main",
]

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft.diagram.docs_catalog --output docs/_data/diagram_catalog.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parents[1]
ASSET_DIR = Path("docs") / "assets" / "diagrams"
#: What ``__all__`` exports that is not a builder.
_NOT_BUILDERS = frozenset({"Diagram", "DiagramToolchainError"})
_ISSUE_SUFFIX = re.compile(r"\s*\(#\d+\)\.?$")


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def builder_names() -> list[str]:
    """Public builders: the functions of ``vaft.diagram.__all__``, in its order."""
    import vaft.diagram as diagram

    return [
        name for name in diagram.__all__
        if name not in _NOT_BUILDERS and inspect.isfunction(getattr(diagram, name))
    ]


def _family_title(module) -> str:
    first = (module.__doc__ or module.__name__).strip().splitlines()[0]
    return _ISSUE_SUFFIX.sub("", first).rstrip(".")


# --------------------------------------------------------------------------
# formula references
# --------------------------------------------------------------------------


def _formula_imports(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """``local name -> vaft.formula.<module>.<name>``, and ``local alias -> formula module``."""
    names: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "vaft.formula" or node.module.startswith("vaft.formula.")
        ):
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "vaft.formula" or alias.name.startswith("vaft.formula."):
                    # ``import vaft.formula.x`` binds ``vaft``; the dotted use spells it out.
                    local = alias.asname or alias.name
                    modules[local] = alias.name if alias.asname else local
    return names, modules


def _referenced_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def formula_references(module_path: Path, builder: str) -> list[str]:
    """``vaft.formula`` functions reachable from ``builder`` in its own module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported, module_aliases = _formula_imports(tree)
    local = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if builder not in local:
        return []
    seen: set[str] = set()
    pending = [builder]
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        body = local[name]
        for referenced in _referenced_names(body):
            if referenced in imported:
                found.add(imported[referenced])
            elif referenced in local:
                pending.append(referenced)
        for child in ast.walk(body):
            dotted = _dotted(child) if isinstance(child, ast.Attribute) else None
            for alias, target in module_aliases.items():
                if dotted and dotted.startswith(alias + "."):
                    found.add(target + dotted[len(alias):])
    return sorted(qualname for qualname in found if _is_formula_function(qualname))


def _is_formula_function(qualname: str) -> bool:
    """Constants and classes imported from ``vaft.formula`` are not formulas."""
    module_name, _, name = qualname.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    return inspect.isfunction(getattr(module, name, None))


# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------


def _argument(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _plain_signature(function) -> str:
    """The call signature with annotations stripped, as the formula and process catalogs print it."""
    signature = inspect.signature(function)
    return str(signature.replace(
        parameters=[p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()],
        return_annotation=inspect.Signature.empty,
    ))


def documentation_snapshot(provenance: Mapping[str, str] | None = None) -> dict:
    """Builders in ``__all__`` order, grouped by the module that defines them, and their assets."""
    import vaft.diagram as diagram

    from . import build

    asset_dir = _ROOT / ASSET_DIR
    manifest_path = asset_dir / build.MANIFEST
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("diagrams", {})

    assets_by_builder: dict[str, list[str]] = {}
    assets: list[dict] = []
    for asset, (builder, kwargs) in build.CANONICAL.items():
        entry = recorded.get(asset, {})
        assets_by_builder.setdefault(builder, []).append(asset)
        assets.append(
            {
                "id": Path(asset).stem,
                "asset": asset,
                "builder": builder,
                "arguments": {key: _argument(value) for key, value in kwargs.items()},
                "call": entry.get("call") or build._call_text(builder, kwargs),
                "svg": (ASSET_DIR.relative_to("docs") / asset).as_posix(),
                "svg_sha256": entry.get("svg_sha256", ""),
                "source_sha256": entry.get("source_sha256", ""),
            }
        )

    families: dict[str, dict] = {}
    builders: list[dict] = []
    # The whole package (builders, scene, the TikZ template), the manifest, and
    # every vaft.formula module a formula reference was resolved in.
    sources: set[Path] = {
        path for path in _PACKAGE.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    }
    sources.add(manifest_path)
    sources.add(_ROOT / "vaft" / "_docstring.py")  # decides every source span
    for name in builder_names():
        function = inspect.unwrap(getattr(diagram, name))
        module = inspect.getmodule(function)
        path = Path(inspect.getsourcefile(function)).resolve()
        sources.add(path)
        family = module.__name__.rsplit(".", 1)[-1].lstrip("_")
        if family not in families:
            families[family] = {"name": family, "title": _family_title(module), "module": module.__name__,
                                "builders": []}
        families[family]["builders"].append(name)
        referenced = formula_references(path, function.__name__)
        for qualname in referenced:
            sources.add(Path(inspect.getsourcefile(importlib.import_module(qualname.rpartition(".")[0]))).resolve())
        formula_rows = [
            {"id": qualname, "category": qualname.split(".")[2] if qualname.count(".") > 2 else "",
             "name": qualname.rsplit(".", 1)[-1]}
            for qualname in referenced
        ]
        summary = (inspect.getdoc(function) or "").strip().split("\n\n")[0].replace("\n", " ")
        builders.append(
            {
                "id": name,
                "name": name,
                "family": family,
                "module": module.__name__,
                "summary": summary,
                "signature": _plain_signature(function),
                "formula": formula_rows,
                "assets": assets_by_builder.get(name, []),
                "source": source_span(function, _ROOT),
            }
        )

    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {"path": _relative(path), "sha256": _sha256(path)}
            for path in sorted(sources, key=_relative)
        ],
        "families": list(families.values()),
        "builders": builders,
        "assets": assets,
        # what Diagram.save can write, so every entry can say how to get each format
        "formats": [
            {"suffix": suffix, "description": description, "requires": list(tools)}
            for suffix, (description, tools) in SAVE_FORMATS.items()
        ],
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def export_documentation_snapshot(output: str | Path, provenance: Mapping[str, str] | None = None) -> Path:
    """Write the YAML snapshot and return its path."""
    import yaml

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            documentation_snapshot(provenance),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the vaft.diagram catalog for the documentation site.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot")
    parser.add_argument("--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot")
    arguments = parser.parse_args(argv)
    provenance = {
        key: value
        for key, value in (("commit", arguments.provenance_commit), ("ref", arguments.provenance_ref))
        if value
    }
    export_documentation_snapshot(arguments.output, provenance or None)


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
