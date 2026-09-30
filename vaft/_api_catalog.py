"""The published Python API of VAFT, read from the source, for the documentation site.

``python -m vaft._api_catalog --output docs/_data/api_catalog.yml`` writes what
``/reference/api/<page>/`` renders (issue #162).  What is published is decided
in two places, and neither is this module:

* each public module's ``__all__`` -- the module's own statement of what it
  publishes (a module is public when no part of its dotted name starts with
  ``_``);
* ``docs/api_inventory.yml`` -- which page a module belongs to, and which
  public-looking modules declare no ``__all__`` and so publish nothing yet.

For every published object the snapshot records what the source says: its
signature with the annotations as written, the first paragraph of its
docstring, whether it is deprecated, where it is defined, and every other
public name it is exported under.  An object is documented once, at its
*home*: the module that defines it if that module exports it, otherwise the
most specific module that does and is not itself deprecated.  A constant's
defining module is read from the source (it carries no ``__module__``), so two
constants that merely share a name and a value stay two entries.

Functions that already have a scientific detail page -- the formula and
process catalogs, registered plots, diagram builders -- are recorded with a
``reference`` URL and rendered as a link, not a second copy of their
documentation.

An object is deprecated when the source says so in one of three ways: the PEP 702
``__deprecated__`` attribute, a docstring whose summary starts with
"Deprecated" (the rule ``vaft._docstring`` already applies), or a
``DeprecationWarning`` raised when the name is read (a relocated name that
``__all__`` still lists).  A function whose body raises a
``DeprecationWarning`` on some path -- usually for one argument -- is marked
``warns_deprecation`` instead: it is not itself deprecated.

The snapshot is a function of the source tree and of the interpreter that reads
it (signatures and default reprs depend on the Python and NumPy versions, which
the documentation build pins), plus the provenance when it is passed.  It
records a checksum of every file it describes so ``docs/build.py`` can prove
which tree it came from.
"""

from __future__ import annotations

import argparse
import ast
import enum
import functools
import hashlib
import importlib
import inspect
import pkgutil
import re
import textwrap
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft._api_catalog --output docs/_data/api_catalog.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parent
INVENTORY = Path("docs") / "api_inventory.yml"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_MAX_VALUE = 120


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


def is_public(module_name: str) -> bool:
    return not any(part.startswith("_") for part in module_name.split("."))


def load_inventory(root: Path = _ROOT) -> dict:
    import yaml

    return yaml.safe_load((root / INVENTORY).read_text(encoding="utf-8"))


def covers(prefix: str, module_name: str) -> bool:
    """Whether an inventory ``prefix`` covers ``module_name``.

    A prefix covers itself and its submodules; the root ``vaft`` covers only
    itself, so a module is never on a page by default.
    """
    if module_name == prefix:
        return True
    return prefix != "vaft" and module_name.startswith(prefix + ".")


def page_for(module_name: str, inventory: Mapping) -> str | None:
    """The page slug whose longest covering module prefix matches ``module_name``."""
    best, length = None, -1
    for page in inventory.get("pages") or []:
        for prefix in page.get("modules") or []:
            if covers(prefix, module_name) and len(prefix) > length:
                best, length = page["slug"], len(prefix)
    return best


def public_modules() -> list[tuple[str, Any]]:
    """Every public ``vaft`` module, imported, in dotted-name order."""
    import vaft

    found = [("vaft", vaft)]
    for info in pkgutil.walk_packages(vaft.__path__, "vaft."):
        if is_public(info.name):
            found.append((info.name, importlib.import_module(info.name)))
    return sorted(found, key=lambda item: item[0])


# --------------------------------------------------------------------------
# reading one object
# --------------------------------------------------------------------------


def _clean(text: str) -> str:
    return _ADDRESS.sub("", text)


def _annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return _clean(inspect.formatannotation(annotation))


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_VALUE else text[: _MAX_VALUE - 3] + "..."


def _default(value: Any) -> str:
    if isinstance(value, (set, frozenset)):
        return _truncate(f"{type(value).__name__}({{{', '.join(sorted(map(repr, value)))}}})")
    return _truncate(_clean(repr(value)))


class _Shown:
    """Prints as the given text: lets ``inspect.Signature`` do the punctuation."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _signature(obj: Any) -> inspect.Signature:
    """``inspect.signature`` with the annotations as source text where Python can say so.

    Python 3.14 evaluates annotations lazily; asking for ``Format.STRING`` gives
    the text as written and never evaluates a ``TYPE_CHECKING``-only name.
    """
    try:
        import annotationlib

        return inspect.signature(obj, annotation_format=annotationlib.Format.STRING)
    except (ImportError, TypeError):  # before 3.14: string annotations come from the __future__ import
        return inspect.signature(obj)


def signature_of(obj: Any) -> str:
    """The call signature with the annotations as written in the source.

    ``inspect.Signature`` keeps the ``/``, ``*`` and ``**`` rules; only the
    annotations (strings under ``from __future__ import annotations``) and the
    defaults are replaced by their display text, the latter without memory
    addresses so the snapshot is deterministic.
    """
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return ""  # an enum is called with one of its values; its members are listed instead
    try:
        signature = _signature(obj)
    except Exception:  # an annotation naming a TYPE_CHECKING-only import, a C callable, ...
        return ""
    empty = inspect.Parameter.empty
    parameters = [
        parameter.replace(
            annotation=_Shown(_annotation(parameter.annotation)) if parameter.annotation is not empty else empty,
            default=_Shown(_default(parameter.default)) if parameter.default is not empty else empty,
        )
        for parameter in signature.parameters.values()
    ]
    returns = signature.return_annotation
    shown = signature.replace(
        parameters=parameters,
        return_annotation=_Shown(_annotation(returns)) if returns is not inspect.Signature.empty else inspect.Signature.empty,
    )
    return str(shown)


def own_doc(obj: Any) -> str:
    """The docstring the object itself carries.

    Not ``inspect.getdoc``, which falls back to a base class's docstring and so
    gives a ``str`` enum the summary of ``str``.  A dataclass without a
    docstring gets one generated from its signature; that is not documentation
    either.
    """
    doc = vars(obj).get("__doc__") if inspect.isclass(obj) else getattr(obj, "__doc__", None)
    if not isinstance(doc, str):
        return ""
    doc = inspect.cleandoc(doc)
    if inspect.isclass(obj) and doc.startswith(f"{obj.__name__}(") and doc.endswith(")") and "\n" not in doc:
        return ""
    return doc


def summary_of(obj: Any) -> str:
    """The first paragraph of the own docstring, one line, Sphinx roles made literal."""
    from ._docstring import strip_roles

    return strip_roles(" ".join(own_doc(obj).strip().split("\n\n")[0].split()))


def _warns_deprecation(function: Any) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False
    return any(_is_deprecation_warn(node) for node in ast.walk(tree))


_DEPRECATION_CATEGORIES = ("DeprecationWarning", "FutureWarning", "PendingDeprecationWarning")


def _is_deprecation_warn(node: ast.AST) -> bool:
    """A call to ``warn``/``warnings.warn`` whose arguments name a deprecation category.

    Suppressing the warning (``except DeprecationWarning``,
    ``filterwarnings(..., DeprecationWarning)``) is not raising it.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if name != "warn":
        return False
    for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
        for child in ast.walk(argument):
            if isinstance(child, ast.Name) and child.id in _DEPRECATION_CATEGORIES:
                return True
            if isinstance(child, ast.Attribute) and child.attr in _DEPRECATION_CATEGORIES:
                return True
    return False


def module_deprecation(module: Any) -> str:
    """Why a whole module is deprecated -- it warns at import -- or ``""``.

    Read from the module's top-level statements, not by catching the warning,
    which only fires on the first import in a process.
    """
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return ""
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if _is_deprecation_warn(node):
                message = next((argument.value for argument in node.args
                                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)), "")
                return _truncate(" ".join(message.split())) or "the module warns on import that it is deprecated"
    return ""


def deprecation_of(obj: Any, access_warning: str = "") -> str:
    """Why ``obj`` itself is deprecated, or ``""``.

    ``access_warning`` is the ``DeprecationWarning`` reading the name raised,
    if any: a relocated name in ``__all__`` warns on access.
    """
    if access_warning:
        return access_warning
    marker = vars(obj).get("__deprecated__") if inspect.isclass(obj) else getattr(obj, "__deprecated__", None)
    if isinstance(marker, str) and marker:
        return marker
    summary = summary_of(obj)
    if summary.lower().startswith("deprecated"):
        return summary
    return ""


def warns_deprecation(obj: Any) -> bool:
    """Whether the function's body raises a ``DeprecationWarning`` on some path.

    Kept apart from :func:`deprecation_of`: a function that deprecates one of
    its arguments is not itself deprecated.
    """
    return inspect.isfunction(obj) and _warns_deprecation(inspect.unwrap(obj))


def source_of(obj: Any) -> dict:
    try:
        target = inspect.unwrap(obj)
        path = inspect.getsourcefile(target)
        _, line = inspect.getsourcelines(target)
        return {"path": _relative(path), "line": line}
    except (OSError, TypeError, ValueError):
        return {"path": "", "line": 0}


def members_of(cls: type) -> list[dict]:
    """Public methods and properties the class itself defines, in definition order."""
    rows = []
    for name, attribute in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(attribute, property):
            kind, target = "property", attribute.fget
        elif isinstance(attribute, (classmethod, staticmethod)):
            kind, target = type(attribute).__name__, attribute.__func__
        elif inspect.isfunction(attribute):
            kind, target = "method", attribute
        else:
            continue
        rows.append({"name": name, "kind": kind, "summary": summary_of(target) if target else ""})
    return rows


def fields_of(cls: type) -> list[dict]:
    import dataclasses

    if issubclass(cls, enum.Enum):
        return [{"name": name, "type": _truncate(_clean(repr(member.value)))} for name, member in cls.__members__.items()]
    if not dataclasses.is_dataclass(cls):
        return []
    return [
        {"name": field.name, "type": _annotation(field.type)}
        for field in dataclasses.fields(cls)
        if not field.name.startswith("_")
    ]


def _stable_value(value: Any) -> str:
    if isinstance(value, (set, frozenset)):
        # sorted, so the text does not depend on the hash seed
        return _truncate(f"{type(value).__name__}({{{', '.join(sorted(map(repr, value)))}}})")
    if isinstance(value, (bool, int, float, complex, str, bytes, type(None))):
        return _default(value)
    if isinstance(value, (tuple, list)) and all(isinstance(item, (bool, int, float, str, type(None))) for item in value):
        return _default(value)
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


_CODE_SPAN = re.compile(r"(``[^`]+``|`[^`]+`)")


def summary_markdown(text: str) -> str:
    """A summary made safe to render as markdown on the API pages.

    Code spans are kept as they are (kramdown escapes their content); outside
    them ``&``, ``<`` and ``>`` become entities, so a docstring cannot inject
    HTML.  The pages mark the entries ``tex2jax_ignore``, so a ``$VAR`` is not
    typeset as mathematics.
    """
    parts = _CODE_SPAN.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = parts[index].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "".join(parts)


def kind_of(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "data"


# --------------------------------------------------------------------------
# links to the scientific reference pages
# --------------------------------------------------------------------------


def _reference_links() -> dict[int, str]:
    """``id(function) -> URL`` for functions that already have a detail page."""
    links: dict[int, str] = {}
    from vaft.formula import catalog as formula_catalog
    from vaft.process import catalog as process_catalog

    for spec in formula_catalog.list_formulas():
        obj = getattr(importlib.import_module(spec.module), spec.name, None)
        if obj is not None:
            links.setdefault(id(obj), f"/reference/formula/{spec.category}/#{spec.name}")
    for spec in process_catalog.list_processes():
        obj = getattr(importlib.import_module(spec.module), spec.name, None)
        if obj is not None:
            links.setdefault(id(obj), f"/reference/process/{spec.category}/#{spec.name}")
    try:
        import vaft.plot as plot
        from vaft.plot import docs_catalog as plot_catalog
        from vaft.plot import registry

        for spec in registry.specs(status=None):
            links.setdefault(id(spec.renderer), f"/reference/plot/#{spec.name}")
        for name, _status in plot_catalog.entry_point_names():
            links.setdefault(id(getattr(plot, name)), f"/reference/plot/#{name}")
    except ImportError:  # a tree without the plot catalog
        pass
    try:
        import vaft.diagram as diagram
        from vaft.diagram import docs_catalog as diagram_catalog

        for name in diagram_catalog.builder_names():
            links.setdefault(id(getattr(diagram, name)), f"/reference/diagram/#{name}")
    except ImportError:
        pass
    return links


# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------


def _access_warning(module: Any, name: str) -> str:
    """The ``DeprecationWarning`` reading ``module.name`` raises, or ``""``.

    A relocated name is served by the module's ``__getattr__``, which may cache
    it after the first read; calling ``__getattr__`` itself asks again, so the
    answer does not depend on what was imported before.
    """
    hook = vars(module).get("__getattr__")
    if hook is None:
        return ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            hook(name)
        except Exception:  # a lazy namespace hook that does not serve this name
            return ""
    return next((str(item.message) for item in caught
                 if issubclass(item.category, (DeprecationWarning, FutureWarning))), "")


@functools.lru_cache(maxsize=None)
def _top_level_bindings(module_name: str) -> tuple[dict[str, tuple[str, str]], tuple[str, ...]]:
    """``name -> (source module, source name)`` for top-level ``from ... import``, the names the
    module assigns itself (mapped to itself), and the modules it star-imports."""
    try:
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))
    except Exception:
        return {}, ()
    package = module_name if hasattr(module, "__path__") else module_name.rpartition(".")[0]
    bindings: dict[str, tuple[str, str]] = {}
    stars: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            base = statement.module or ""
            if statement.level:
                parts = package.split(".")
                parent = ".".join(parts[: len(parts) - statement.level + 1])
                base = f"{parent}.{base}" if base else parent
            for alias in statement.names:
                if alias.name == "*":
                    stars.append(base)
                else:
                    bindings[alias.asname or alias.name] = (base, alias.name)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        bindings[node.id] = (module_name, node.id)
    return bindings, tuple(stars)


def defining_module(module_name: str, name: str, _depth: int = 0) -> tuple[str, str]:
    """Where a module-level constant is assigned, following ``from ... import`` chains.

    Constants carry no ``__module__``, so this reads the source; a chain that
    cannot be followed ends at the module that exports the name.
    """
    if _depth > 8:
        return module_name, name
    bindings, stars = _top_level_bindings(module_name)
    source = bindings.get(name)
    if source == (module_name, name):
        return source
    if source and source[0].startswith("vaft"):
        return defining_module(*source, _depth=_depth + 1)
    for star in stars:
        if star.startswith("vaft"):
            found = defining_module(star, name, _depth + 1)
            if found != (star, name) or _top_level_bindings(star)[0].get(name) == (star, name):
                return found
    return module_name, name


def _key(module_name: str, name: str, obj: Any):
    """Which exports are the same object: identity for callables, the defining assignment for data.

    Two constants that merely share a name and a value (``SCHEMA_VERSION = 1``
    in five schemas) are different API; one constant re-exported under several
    modules is not.
    """
    if inspect.isclass(obj) or callable(obj):
        return ("object", id(obj))
    return ("data", *defining_module(module_name, name))


def documentation_snapshot(provenance: Mapping[str, str] | None = None) -> dict:
    inventory = load_inventory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modules = public_modules()
        links = _reference_links()

    exports: dict[Any, list[tuple[str, str, Any]]] = {}
    access_warnings: dict[str, str] = {}
    deprecated_modules: dict[str, str] = {}
    module_rows: list[dict] = []
    for module_name, module in modules:
        declared = getattr(module, "__all__", None)
        whole = module_deprecation(module)
        if whole:
            deprecated_modules[module_name] = whole
        module_rows.append({
            "name": module_name,
            "page": page_for(module_name, inventory) or "",
            "summary": summary_markdown(summary_of(module)) if module.__doc__ else "",
            "declared": declared is not None,
            "exports": len(declared or ()),
            "deprecated": whole,
        })
        for name in declared or ():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    obj = getattr(module, name)
                except AttributeError:  # catalog_coverage.py reports the dangling name
                    continue
            if inspect.ismodule(obj):
                continue
            relocated = _access_warning(module, name)
            if relocated:
                access_warnings[f"{module_name}.{name}"] = relocated
            exports.setdefault(_key(module_name, name, obj), []).append((module_name, name, obj))

    entries: list[dict] = []
    for key, sites in exports.items():
        obj = sites[0][2]
        if kind_of(obj) == "data":
            defining, defined_name = key[1], key[2]
        else:
            defining, defined_name = getattr(inspect.unwrap(obj), "__module__", None), getattr(obj, "__name__", None)
        own = [site for site in sites if site[0] == defining and site[1] == defined_name]
        if not own:
            own = [site for site in sites if site[0] == defining]
        # Otherwise the most specific exporting module that is not itself deprecated.
        home = own[0] if own else sorted(
            sites, key=lambda site: (site[0] in deprecated_modules, -site[0].count("."), site[0], site[1]))[0]
        stale = {f"{site[0]}.{site[1]}" for site in sites
                 if f"{site[0]}.{site[1]}" in access_warnings or site[0] in deprecated_modules}
        module_name, name, _ = home
        kind = kind_of(obj)
        entry = {
            "id": f"{module_name}.{name}",
            "name": name,
            "module": module_name,
            "page": page_for(module_name, inventory) or "",
            "kind": kind,
            "signature": signature_of(obj) if kind != "data" else "",
            "summary": summary_markdown(summary_of(obj)) if kind != "data" else "",
            "value": _stable_value(obj) if kind == "data" else "",
            "deprecated": deprecation_of(obj, access_warnings.get(f"{module_name}.{name}", "")
                                         or deprecated_modules.get(module_name, "")),
            "warns_deprecation": warns_deprecation(obj),
            "source": source_of(obj) if kind != "data" else {"path": "", "line": 0},
            "reference": links.get(id(obj), "") if kind == "function" else "",
            "exported_as": sorted(f"{site[0]}.{site[1]}" for site in sites if site != home),
            # other names that still work but warn: relocated re-exports, deprecated modules
            "deprecated_aliases": sorted(name_ for name_ in stale if name_ != f"{module_name}.{name}"),
        }
        if kind == "class":
            entry["fields"] = fields_of(obj)
            entry["members"] = members_of(obj)
        entries.append(entry)

    page_order = {page["slug"]: index for index, page in enumerate(inventory.get("pages") or [])}
    entries.sort(key=lambda entry: (page_order.get(entry["page"], len(page_order)), entry["module"], entry["name"].lower(), entry["name"]))
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["page"]] = counts.get(entry["page"], 0) + 1

    sources = sorted(path for path in _PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
    sources.append(_ROOT / INVENTORY)
    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {"path": _relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(sources, key=_relative)
        ],
        "pages": [
            {**page, "count": counts.get(page["slug"], 0)} for page in inventory.get("pages") or []
        ],
        "modules": module_rows,
        "entries": entries,
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def export_documentation_snapshot(output: str | Path, provenance: Mapping[str, str] | None = None) -> Path:
    import yaml

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(documentation_snapshot(provenance), allow_unicode=True, sort_keys=False,
                       default_flow_style=False, width=100),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the published VAFT Python API for the documentation site.")
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
