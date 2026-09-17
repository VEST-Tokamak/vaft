"""Searchable catalog of the processing layer, read out of the docstrings.

This is the discovery layer issue #252 asks for, built the way
:mod:`vaft.formula.catalog` is.  It is *not* imported when ``vaft.process``
or one of its submodules loads; it is pulled in the first time anything
touches ``vaft.process.describe``, ``vaft.process.search``,
``vaft.process.list_processes``, ``vaft.process.categories`` or
``vaft.process.catalog`` itself.  The processing submodules stay lightweight:
nothing here registers anything at their import time, and the docstrings they
already carry are the only source of truth.

>>> import vaft.process as P
>>> print(P.describe("repair_clipped_interval"))       # one routine, rendered
>>> P.search("vest.yaml")                              # specs whose text mentions it
>>> P.list_processes(category="signal_processing")     # imports only that submodule

``python -m vaft.process.catalog --output docs/_data/process_catalog.yml``
writes the deterministic YAML snapshot the documentation site renders.

One thing this catalog records that the formula one does not: whether each
function *conforms* to the contract.  The processing layer is being brought
under the contract one submodule at a time (#417-#421), and the site has to
say which categories are documented and which are still pending, so every
spec carries its parser errors and every category says whether all of its
functions are clean.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from collections.abc import Mapping
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
    machine_scope,
    parse_docstring,
    parse_module_docstring,
    strip_roles,
)

__all__ = [
    "CATEGORIES",
    "CategoryDoc",
    "ProcessSpec",
    "categories",
    "describe",
    "documentation_snapshot",
    "export_documentation_snapshot",
    "list_processes",
    "search",
]

#: Submodules whose functions the catalog covers, in package import order,
#: plus ``cocos``, which the package never star-imported but which callers
#: reach by name.  ``_equilibrium_parametric`` is not a category: its
#: functions are public through ``equilibrium``, and that is where they appear.
CATEGORIES: tuple[str, ...] = (*_IMPORT_ORDER, "cocos")

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft.process.catalog --output docs/_data/process_catalog.yml"
_PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProcessSpec:
    """Everything the catalog knows about one public processing function.

    All text fields come from the function's docstring as parsed by
    :mod:`vaft.process._docstring`; ``signature`` is the call signature with
    annotations stripped; ``aliases`` are other names in the same submodule's
    ``__all__`` bound to the same function object; ``machine_scope`` is what
    ``Applicability`` declares; ``conforming`` is ``not errors``.
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
    machine_scope: str | None
    convention_sensitive: bool
    deprecated: bool
    aliases: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def qualname(self) -> str:
        """``category.name``, the unambiguous catalog key."""
        return f"{self.category}.{self.name}"

    @property
    def conforming(self) -> bool:
        return not self.errors

    def section(self, title: str) -> str | None:
        """Text of one section, or ``None`` when the docstring lacks it."""
        for name, text in self.sections:
            if name == title:
                return text
        return None

    def render(self) -> str:
        """Terminal-friendly description: signature, units, sections, provenance."""
        flags = []
        if self.machine_scope == "independent":
            flags.append("machine-independent")
        elif self.machine_scope == "vest":
            flags.append("VEST-specific")
        if self.convention_sensitive:
            flags.append("convention-sensitive")
        if self.deprecated:
            flags.append("deprecated")
        lines = [f"{self.qualname}{self.signature}"]
        if self.aliases:
            lines.append(f"  aliases: {', '.join(self.aliases)}")
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
            if title in ("Parameters", "Returns", "Yields", "Provenance"):
                continue
            lines.extend(["", title])
            lines.extend(f"  {line}" for line in text.splitlines())
        if self.references:
            lines.extend(["", "Provenance"])
            lines.extend(f"  [{ref.label}] {ref.text}" for ref in self.references)
        if self.errors:
            lines.extend(["", "Contract violations"])
            lines.extend(f"  - {error}" for error in self.errors)
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
                if title not in ("Parameters", "Returns", "Yields", "Provenance")
            ],
            "provenance": [
                {"label": ref.label, "text": strip_roles(ref.text)} for ref in self.references
            ],
            "machine_scope": self.machine_scope,
            "convention_sensitive": self.convention_sensitive,
            "deprecated": self.deprecated,
            "conforming": self.conforming,
            "aliases": list(self.aliases),
            "errors": list(self.errors),
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
    documented: int

    @property
    def conforming(self) -> bool:
        """Every function in the category is clean under the contract."""
        return self.count > 0 and self.documented == self.count

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
            "documented": self.documented,
            "conforming": self.conforming,
        }


def _require_docstrings() -> None:
    if sys.flags.optimize >= 2:
        raise RuntimeError(
            "docstrings are stripped under `python -OO`; the process catalog reads them "
            "and cannot run in this mode"
        )


def _source_files() -> list[Path]:
    """Every module file the snapshot describes, ``__init__`` excepted.

    One entry per file rather than per category, because ``equilibrium``
    re-exports ``_equilibrium_parametric``: a checksum keyed by category
    would pass while an edit to the private module went unnoticed.
    """
    return sorted(
        path for path in _PACKAGE.glob("*.py") if path.name != "__init__.py"
    )


def _signature(fn) -> str:
    signature = inspect.signature(fn)
    stripped = signature.replace(
        parameters=[p.replace(annotation=p.empty) for p in signature.parameters.values()],
        return_annotation=inspect.Signature.empty,
    )
    return str(stripped)


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _structural_violations(parsed: ParsedDocstring, fn) -> list[str]:
    """What the contract requires of every function, beyond parsing cleanly.

    The parser only reports sections that are present and malformed; a
    one-line docstring parses without error.  ``conforming`` has to mean
    more than that, so the unconditional requirements are checked here: the
    parameters match the signature and carry units, there is a ``Returns``
    with units, and ``Applicability`` declares a machine scope.  Deprecated
    shims are exempt from everything but a summary.  Requirements that are
    physics judgement -- which functions need ``Provenance``, ``Processing
    steps`` or ``Convention`` -- stay in test_process_docstrings.py.
    """
    violations: list[str] = []
    if not parsed.summary:
        return ["missing summary line"]
    if parsed.deprecated:
        return violations
    expected = [
        p.name for p in inspect.signature(fn).parameters.values() if p.kind not in _VARIADIC
    ]
    documented = [name.strip() for item in parsed.parameters for name in item.name.split(",")]
    if documented != expected:
        violations.append(
            f"Parameters documents {documented} but the signature has {expected}"
        )
    for item in parsed.parameters:
        if not item.unit:
            violations.append(f"parameter {item.name} lacks a [unit] tag")
    if not parsed.returns:
        violations.append("missing Returns section")
    for item in parsed.returns:
        if not item.unit:
            violations.append(f"return {item.name or item.type} lacks a [unit] tag")
    if machine_scope(parsed) is None:
        violations.append(
            "Applicability must open with 'Machine-independent.' or 'VEST-specific.'"
        )
    return violations


def _spec(fn, name: str, category: str, module_name: str, aliases: tuple[str, ...]) -> ProcessSpec:
    parsed: ParsedDocstring = parse_docstring(fn.__doc__)
    errors = list(dict.fromkeys([*parsed.errors, *_structural_violations(parsed, fn)]))
    return ProcessSpec(
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
        machine_scope=machine_scope(parsed),
        convention_sensitive=parsed.convention_sensitive,
        deprecated=parsed.deprecated,
        aliases=aliases,
        errors=tuple(errors),
    )


@lru_cache(maxsize=None)
def _specs_for(category: str) -> dict[str, ProcessSpec]:
    """Specs of every public function one submodule *exports*, by name.

    Imports only that submodule.  Selection is by ``__all__``, which every
    process submodule declares (#249), not by ``__module__``: that is what
    puts the ``_equilibrium_parametric`` functions under ``equilibrium``,
    the submodule that is their documented public location.
    """
    _require_docstrings()
    if category not in CATEGORIES:
        raise KeyError(
            f"no process category {category!r}; choose one of {', '.join(CATEGORIES)}"
        )
    module = _submodule(category)
    exported = getattr(module, "__all__", None)
    if exported is None:
        raise RuntimeError(f"{module.__name__} declares no __all__; #249 requires one")
    bound = {
        name: getattr(module, name)
        for name in exported
        if inspect.isfunction(getattr(module, name, None))
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
    return {
        name: _spec(bound[name], name, category, module.__name__, tuple(sorted(aliases.get(name, ()))))
        for name in sorted(set(canonical.values()))
    }


def _alias_index(category: str) -> dict[str, str]:
    return {
        alias: spec.name
        for spec in _specs_for(category).values()
        for alias in spec.aliases
    }


def describe(name: str) -> ProcessSpec:
    """The spec of one processing function.

    ``name`` is either ``"category.function"`` -- which imports only that
    category -- or a bare function name.  A bare name is looked up across the
    categories in import order; since #249 established that no two process
    submodules export different objects under one name, the first hit is the
    only hit.  Aliases resolve to their canonical function.
    """
    category, _, bare = name.rpartition(".")
    if category:
        if category not in CATEGORIES:
            raise KeyError(
                f"no process category {category!r} in {name!r}; "
                f"use vaft.process.list_processes() to list them"
            )
        candidates: tuple[str, ...] = (category,)
    else:
        bare = name
        candidates = CATEGORIES
    for candidate in candidates:
        specs = _specs_for(candidate)
        if bare in specs:
            return specs[bare]
        alias = _alias_index(candidate).get(bare)
        if alias:
            return specs[alias]
    raise KeyError(
        f"no processing function named {name!r}; use vaft.process.list_processes() to list them"
    )


def list_processes(category: str | None = None) -> list[ProcessSpec]:
    """Specs of every processing function, or of one category, in category-then-name order.

    Passing a category imports only that submodule.
    """
    selected = CATEGORIES if category is None else (category,)
    result: list[ProcessSpec] = []
    for key in selected:
        result.extend(_specs_for(key)[name] for name in sorted(_specs_for(key)))
    return result


def search(text: str, *, category: str | None = None) -> list[ProcessSpec]:
    """Functions whose name, prose, sections, parameters or provenance mention ``text``.

    Case-insensitive substring match; empty ``text`` returns everything.
    """
    needle = text.lower()

    def haystack(spec: ProcessSpec) -> str:
        parts = [spec.name, *spec.aliases, spec.summary, spec.description]
        parts.extend(body for _, body in spec.sections)
        parts.extend(ref.text for ref in spec.references)
        parts.extend(f"{p.name} {p.type} {p.unit or ''} {p.description}" for p in spec.parameters)
        parts.extend(f"{r.name or ''} {r.type} {r.unit or ''} {r.description}" for r in spec.returns)
        return "\n".join(parts).lower()

    return [spec for spec in list_processes(category) if needle in haystack(spec)]


def categories() -> list[CategoryDoc]:
    """Every submodule as a documentation category."""
    _require_docstrings()
    result: list[CategoryDoc] = []
    for key in CATEGORIES:
        module = _submodule(key)
        doc: ModuleDoc = parse_module_docstring(module.__doc__)
        specs = _specs_for(key)
        result.append(
            CategoryDoc(
                key,
                module.__name__,
                doc.title,
                doc.overview,
                doc.notation,
                doc.conventions,
                len(specs),
                sum(1 for spec in specs.values() if spec.conforming),
            )
        )
    return result


def documentation_snapshot(
    category: str | None = None,
    provenance: Mapping[str, str] | None = None,
) -> dict:
    """Deterministic, site-ready representation of the whole catalog.

    ``provenance`` records which source tree the snapshot describes -- the
    commit and ref the documentation build extracted -- and is omitted
    entirely when it is not supplied, so the default output is a pure
    function of the source files.
    """
    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {
                "path": f"vaft/process/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in _source_files()
        ],
        "categories": [doc.as_dict() for doc in categories()],
        "functions": [spec.as_dict() for spec in list_processes(category)],
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def export_documentation_snapshot(
    output: str | Path,
    category: str | None = None,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Write the YAML snapshot and return its path."""
    import yaml  # deferred: the catalog itself must stay stdlib-only

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            documentation_snapshot(category, provenance),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the vaft.process catalog for the documentation site."
    )
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--category", choices=CATEGORIES, help="Restrict the functions to one category")
    parser.add_argument(
        "--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot"
    )
    parser.add_argument(
        "--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot"
    )
    arguments = parser.parse_args(argv)
    provenance = {
        key: value
        for key, value in (
            ("commit", arguments.provenance_commit),
            ("ref", arguments.provenance_ref),
        )
        if value
    }
    export_documentation_snapshot(arguments.output, arguments.category, provenance or None)


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
