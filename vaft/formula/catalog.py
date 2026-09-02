"""Searchable catalog of the formula layer, read out of the docstrings.

This module is the optional discovery layer requested by issue #248.  It is
*not* imported when ``vaft.formula`` or one of its submodules loads; it is
pulled in the first time anything touches ``vaft.formula.describe``,
``vaft.formula.search``, ``vaft.formula.list_formulas``,
``vaft.formula.categories`` or ``vaft.formula.catalog`` itself.  The physics
submodules stay lightweight: nothing here registers anything at their import
time, and the docstrings they already carry are the only source of truth.

>>> import vaft.formula as F
>>> print(F.describe("greenwald_density"))          # one function, rendered
>>> F.search("Sauter")                              # specs whose text mentions it
>>> F.list_formulas(category="stability")           # imports only stability

``python -m vaft.formula.catalog --output _data/formula_catalog.yml`` writes
the deterministic YAML snapshot the documentation site renders, in the same
spirit as ``python -m vaft.machine_mapping.registry``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import inspect
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import _IMPORT_ORDER, _submodule
from ._docstring import (
    CUSTOM_SECTIONS,
    ModuleDoc,
    ParamDoc,
    ParsedDocstring,
    Reference,
    ReturnDoc,
    parse_docstring,
    parse_module_docstring,
    strip_roles,
)

__all__ = [
    "CATEGORIES",
    "CategoryDoc",
    "FormulaSpec",
    "categories",
    "describe",
    "documentation_snapshot",
    "export_documentation_snapshot",
    "list_formulas",
    "search",
]

#: Submodules whose functions the catalog covers, in package import order.
#: ``constants`` defines no functions and appears only through :func:`categories`.
CATEGORIES: tuple[str, ...] = tuple(key for key in _IMPORT_ORDER if key != "constants")

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft.formula.catalog --output _data/formula_catalog.yml"


@dataclass(frozen=True)
class FormulaSpec:
    """Everything the catalog knows about one public formula function.

    All text fields come from the function's docstring as parsed by
    :mod:`vaft.formula._docstring`; ``signature`` is the call signature with
    annotations stripped; ``aliases`` are other module-level names bound to
    the same function object; ``shadowed_by`` names the later category whose
    same-named function ``vaft.formula.<name>`` resolves to instead.
    """

    name: str
    category: str
    module: str
    signature: str
    summary: str
    description: str
    parameters: tuple[ParamDoc, ...]
    returns: tuple[ReturnDoc, ...]
    sections: tuple[tuple[str, str], ...]
    references: tuple[Reference, ...]
    empirical: bool
    convention_sensitive: bool
    deprecated: bool
    aliases: tuple[str, ...] = ()
    shadowed_by: str | None = None
    errors: tuple[str, ...] = ()

    @property
    def qualname(self) -> str:
        """``category.name``, the unambiguous catalog key."""
        return f"{self.category}.{self.name}"

    def section(self, title: str) -> str | None:
        """Text of one section, or ``None`` when the docstring lacks it."""
        for name, text in self.sections:
            if name == title:
                return text
        return None

    def render(self) -> str:
        """Terminal-friendly description: signature, units, sections, references."""
        flags = []
        if self.empirical:
            flags.append("empirical fit")
        if self.convention_sensitive:
            flags.append("convention-sensitive")
        if self.deprecated:
            flags.append("deprecated")
        lines = [f"{self.qualname}{self.signature}"]
        if self.aliases:
            lines.append(f"  aliases: {', '.join(self.aliases)}")
        if self.shadowed_by:
            lines.append(
                f"  note: vaft.formula.{self.name} resolves to {self.shadowed_by}.{self.name}"
            )
        lines.append(f"  {self.summary}" + (f"   [{'; '.join(flags)}]" if flags else ""))
        if self.description:
            lines.append("")
            lines.extend(f"  {line}" for line in self.description.splitlines())
        if self.parameters:
            lines.extend(["", "Parameters"])
            for item in self.parameters:
                unit = f" [{item.unit}]" if item.unit else ""
                type_ = f" : {item.type}" if item.type else ""
                lines.append(f"  {item.name}{type_}{unit}")
                if item.description:
                    lines.append(f"      {item.description}")
        if self.returns:
            lines.extend(["", "Returns"])
            for ret in self.returns:
                unit = f" [{ret.unit}]" if ret.unit else ""
                head = f"{ret.name} : {ret.type}" if ret.name else ret.type
                lines.append(f"  {head}{unit}")
                if ret.description:
                    lines.append(f"      {ret.description}")
        for title, text in self.sections:
            if title in ("Parameters", "Returns", "Yields", "References"):
                continue
            lines.extend(["", title])
            lines.extend(f"  {line}" for line in text.splitlines())
        if self.references:
            lines.extend(["", "References"])
            lines.extend(f"  [{ref.label}] {ref.text}" for ref in self.references)
        return "\n".join(lines)

    __str__ = render

    def as_dict(self) -> dict:
        """The snapshot row: plain scalars and lists, Sphinx roles stripped."""
        return {
            "id": self.qualname,
            "name": self.name,
            "category": self.category,
            "module": self.module,
            "signature": self.signature,
            "summary": strip_roles(self.summary),
            "description": strip_roles(self.description),
            "parameters": [
                {
                    "name": item.name,
                    "type": item.type,
                    "unit": item.unit,
                    "description": strip_roles(item.description),
                }
                for item in self.parameters
            ],
            "returns": [
                {
                    "name": ret.name,
                    "type": ret.type,
                    "unit": ret.unit,
                    "description": strip_roles(ret.description),
                }
                for ret in self.returns
            ],
            "sections": [
                {"title": title, "text": strip_roles(text)}
                for title, text in self.sections
                if title not in ("Parameters", "Returns", "Yields", "References")
            ],
            "references": [
                {"label": ref.label, "text": strip_roles(ref.text)} for ref in self.references
            ],
            "empirical": self.empirical,
            "convention_sensitive": self.convention_sensitive,
            "deprecated": self.deprecated,
            "aliases": list(self.aliases),
            "shadowed_by": self.shadowed_by,
        }


@dataclass(frozen=True)
class CategoryDoc:
    """One submodule as a documentation category."""

    name: str
    module: str
    title: str
    overview: str
    notation: tuple
    conventions: str
    count: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "title": strip_roles(self.title),
            "overview": strip_roles(self.overview),
            "notation": [
                {"symbol": row.symbol, "description": row.description, "unit": row.unit}
                for row in self.notation
            ],
            "conventions": strip_roles(self.conventions),
            "count": self.count,
        }


def _require_docstrings() -> None:
    if sys.flags.optimize >= 2:
        raise RuntimeError(
            "docstrings are stripped under `python -OO`; the formula catalog reads them "
            "and cannot run in this mode"
        )


def _source_path(category: str) -> Path:
    return Path(__file__).with_name(f"{category}.py")


@lru_cache(maxsize=None)
def _defined_names(category: str) -> frozenset[str]:
    """Top-level ``def`` and assignment targets of a submodule, without importing it.

    Used to decide which category ``vaft.formula.<name>`` resolves to, so that
    ``describe("greenwald_density")`` imports ``stability`` alone rather than
    walking every later submodule.
    """
    tree = ast.parse(_source_path(category).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return frozenset(name for name in names if not name.startswith("_"))


def _shadowing_category(category: str, name: str) -> str | None:
    """The later category whose own definition of ``name`` wins the package lookup."""
    later = CATEGORIES[CATEGORIES.index(category) + 1 :]
    winner = None
    for other in later:
        if name in _defined_names(other):
            winner = other
    return winner


def _signature(fn) -> str:
    signature = inspect.signature(fn)
    stripped = signature.replace(
        parameters=[p.replace(annotation=p.empty) for p in signature.parameters.values()],
        return_annotation=inspect.Signature.empty,
    )
    return str(stripped)


def _spec(fn, name: str, category: str, module_name: str, aliases: tuple[str, ...]) -> FormulaSpec:
    parsed: ParsedDocstring = parse_docstring(fn.__doc__)
    return FormulaSpec(
        name=name,
        category=category,
        module=module_name,
        signature=_signature(fn),
        summary=parsed.summary,
        description=parsed.description,
        parameters=parsed.parameters,
        returns=parsed.returns,
        sections=parsed.sections,
        references=parsed.references,
        empirical=parsed.empirical,
        convention_sensitive=parsed.convention_sensitive,
        deprecated=parsed.deprecated,
        aliases=aliases,
        shadowed_by=_shadowing_category(category, name),
        errors=parsed.errors,
    )


@lru_cache(maxsize=None)
def _specs_for(category: str) -> dict[str, FormulaSpec]:
    """Specs of every public function *defined* in one submodule, by name.

    Imports only that submodule.  Selection is by ``__module__``, not by
    ``__all__``: modules without an ``__all__`` leak their own imports
    (``np``, ``warnings``, ``curve_fit``) through the package's star export,
    and those are not formulas.
    """
    _require_docstrings()
    if category not in CATEGORIES:
        raise KeyError(
            f"no formula category {category!r}; choose one of {', '.join(CATEGORIES)}"
        )
    module = _submodule(category)
    bound = {
        name: obj
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == module.__name__
    }
    canonical: dict[int, str] = {}
    for name, obj in bound.items():
        if obj.__name__ == name:
            canonical[id(obj)] = name
    for name, obj in bound.items():
        canonical.setdefault(id(obj), name)
    aliases: dict[str, list[str]] = {}
    for name, obj in bound.items():
        primary = canonical[id(obj)]
        if name != primary:
            aliases.setdefault(primary, []).append(name)
    specs = {
        name: _spec(bound[name], name, category, module.__name__, tuple(sorted(aliases.get(name, ()))))
        for name in sorted(set(canonical.values()))
    }
    return specs


def _alias_index(category: str) -> dict[str, str]:
    return {
        alias: spec.name
        for spec in _specs_for(category).values()
        for alias in spec.aliases
    }


def _resolve_category(name: str) -> str | None:
    """Which category ``vaft.formula.<name>`` binds, by the package's own rule."""
    winner = None
    for category in CATEGORIES:
        if name in _defined_names(category):
            winner = category
    return winner


def describe(name: str) -> FormulaSpec:
    """The spec of one formula.

    ``name`` is either ``"category.function"`` -- which imports only that
    category -- or a bare function name, resolved exactly the way
    ``vaft.formula.<name>`` is: the last category in import order that
    defines it wins.  Aliases resolve to their canonical function.
    """
    category, _, bare = name.rpartition(".")
    if category:
        if category not in CATEGORIES:
            raise KeyError(
                f"no formula category {category!r} in {name!r}; "
                f"use vaft.formula.list_formulas() to list them"
            )
        candidates = [category]
    else:
        bare = name
        resolved = _resolve_category(bare)
        candidates = [resolved] if resolved else []
    for candidate in candidates:
        specs = _specs_for(candidate)
        if bare in specs:
            return specs[bare]
        alias = _alias_index(candidate).get(bare)
        if alias:
            return specs[alias]
    raise KeyError(f"no formula named {name!r}; use vaft.formula.list_formulas() to list them")


def list_formulas(category: str | None = None) -> list[FormulaSpec]:
    """Specs of every formula, or of one category, in category-then-name order.

    Passing a category imports only that submodule.
    """
    selected = CATEGORIES if category is None else (category,)
    result: list[FormulaSpec] = []
    for key in selected:
        result.extend(_specs_for(key)[name] for name in sorted(_specs_for(key)))
    return result


_SEARCH_FIELDS = ("name", "summary", "description", "sections", "references", "parameters")


def search(text: str, *, category: str | None = None) -> list[FormulaSpec]:
    """Formulas whose name, prose, sections, parameters or references mention ``text``.

    Case-insensitive substring match; empty ``text`` returns everything.
    """
    needle = text.lower()

    def haystack(spec: FormulaSpec) -> str:
        parts = [spec.name, *spec.aliases, spec.summary, spec.description]
        parts.extend(body for _, body in spec.sections)
        parts.extend(ref.text for ref in spec.references)
        parts.extend(f"{p.name} {p.type} {p.unit or ''} {p.description}" for p in spec.parameters)
        parts.extend(f"{r.name or ''} {r.type} {r.unit or ''} {r.description}" for r in spec.returns)
        return "\n".join(parts).lower()

    return [spec for spec in list_formulas(category) if needle in haystack(spec)]


def categories() -> list[CategoryDoc]:
    """Every submodule as a documentation category, ``constants`` included."""
    _require_docstrings()
    result: list[CategoryDoc] = []
    for key in _IMPORT_ORDER:
        module = _submodule(key)
        doc: ModuleDoc = parse_module_docstring(module.__doc__)
        count = len(_specs_for(key)) if key in CATEGORIES else 0
        result.append(
            CategoryDoc(key, module.__name__, doc.title, doc.overview, doc.notation, doc.conventions, count)
        )
    return result


def documentation_snapshot(category: str | None = None) -> dict:
    """Deterministic, site-ready representation of the whole catalog."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {
                "path": f"vaft/formula/{key}.py",
                "sha256": hashlib.sha256(_source_path(key).read_bytes()).hexdigest(),
            }
            for key in _IMPORT_ORDER
        ],
        "categories": [doc.as_dict() for doc in categories()],
        "formulas": [spec.as_dict() for spec in list_formulas(category)],
    }


def export_documentation_snapshot(output: str | Path, category: str | None = None) -> Path:
    """Write the YAML snapshot and return its path."""
    import yaml  # deferred: the catalog itself must stay stdlib-only

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            documentation_snapshot(category),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the vaft.formula catalog for gh-pages.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--category", choices=CATEGORIES, help="Restrict the formulas to one category")
    arguments = parser.parse_args(argv)
    export_documentation_snapshot(arguments.output, arguments.category)


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
