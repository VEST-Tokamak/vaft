#!/usr/bin/env python3
"""Fail when a generated reference catalog and the library's public surface disagree.

    cd docs && PYTHONPATH=.. python scripts/catalog_coverage.py

``docs/build.py`` runs this against every track that ships it, right after the
generators, with ``PYTHONPATH`` pointing at that track's source tree.  For
each catalog ``generators.yml`` declares it compares the snapshot, in both
directions, with the public surface:

missing
    a public object no catalog entry describes;
vanished
    a catalog entry whose object is no longer public where it says it lives.

The generators run seconds earlier on the same tree, so comparing a snapshot
with the enumeration its own generator performs could only catch a generator
bug.  Each layer is therefore also enumerated from something the generator
does not read: the source files on disk, or the namespace users actually type.

formula / process
    every public submodule found **on disk** (``pkgutil``, not the package's
    ``_IMPORT_ORDER``), minus the catalog module itself: each function in its
    ``__all__`` -- or, without an ``__all__``, each public function defined in
    it -- must be described by an entry.  So must every function in the
    package namespace ``vaft.formula.__all__`` / ``vaft.process.__all__``.
    Entries are matched by the object their ``module`` and ``name`` resolve
    to, so an entry that describes a different function of the same name does
    not count.  Classes, constants and submodules are API, not catalog
    entries; they belong to the API reference (#162).
plot
    every function in ``vaft.plot.__all__`` is either a registered renderer
    (and then a ``plots`` entry) or, when its name starts with ``plot_`` (it
    draws a figure), an ``entry_points`` entry; every other name ``dir(vaft.plot)``
    offers beyond ``__all__`` (the legacy cross-shot statistics, served
    without a warning) is an ``entry_points`` entry too.  What remains of
    ``__all__`` -- view models, ``render_*`` bodies, ``*_model`` builders,
    helpers -- is support API for #162.  Every registered spec, whatever its
    status, is a ``plots`` entry.
plot thumbnails
    ``vaft.plot.docs_thumbnails.check`` without the samples: every registered
    plot has a manifest entry, every rendered one its PNG, no PNG or entry is
    orphaned, and no PNG differs from its recorded hash.  A stale thumbnail is
    printed as a warning and does not fail.
diagram
    every public function **defined in a vaft.diagram module on disk** that is
    annotated to return a ``Diagram`` is a builder entry and is in
    ``vaft.diagram.__all__``; every asset :data:`vaft.diagram.build.CANONICAL`
    declares and every ``*.svg`` committed under ``docs/assets/diagrams`` is an
    asset entry; every builder has at least one asset.

python API (#162)
    every public module found **on disk** either declares ``__all__`` and
    belongs to a page of ``docs/api_inventory.yml``, or is listed there as
    ``undeclared``; a stale ``undeclared`` entry fails too.  Every name in an
    ``__all__`` must exist and be public, and every exported object must be
    described by exactly one entry of ``api_catalog.yml`` (matched by object
    identity; constants by exported name).

source spans (#1069)
    every ``source`` span in every catalog names a file of the tree being
    documented and a whole ``def`` or ``class`` of it (first decorator to last
    line, read here with ``ast``), and its inline ``code`` is exactly those
    lines.  Every entry but an API constant or alias has a span.  The tree is
    the provenance commit's, so the pinned links and the inline source show
    the revision the pages describe.

Finally every ``vaft`` module imported while checking must come from the tree
being documented, so an editable install that serves a missing subpackage from
another checkout cannot pass for this one.

The rendered pages are checked against the catalogs by ``validate_docs.rb``.

Exit status is 1 when anything disagrees, and every disagreement is printed.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]


def _load(path: Path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _entry_names(entry: dict) -> list[str]:
    return [entry["name"], *(entry.get("aliases") or [])]


# --------------------------------------------------------------------------
# formula and process
# --------------------------------------------------------------------------


def _package_functions(module_names: tuple[str, ...]) -> dict[str, object]:
    public: dict[str, object] = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if inspect.isfunction(obj):
                public.setdefault(f"{module_name}.{name}", obj)
    return public


def _submodule_functions(package_name: str, skip: frozenset[str]) -> dict[str, object]:
    """Public functions of every public submodule on disk."""
    package = importlib.import_module(package_name)
    public: dict[str, object] = {}
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda info: info.name):
        if info.name.startswith("_") or info.name in skip:
            continue
        module = importlib.import_module(f"{package_name}.{info.name}")
        exported = getattr(module, "__all__", None)
        if exported is None:
            exported = [name for name, obj in vars(module).items()
                        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == module.__name__]
        for name in exported:
            obj = getattr(module, name, None)
            if inspect.isfunction(obj):
                public.setdefault(f"{module.__name__}.{name}", obj)
    return public


def check_functions(kind: str, snapshot: dict, rows_key: str, package: str,
                    extra_namespaces: tuple[str, ...] = ()) -> list[str]:
    problems: list[str] = []
    described: dict[int, set[str]] = {}
    for entry in snapshot.get(rows_key) or []:
        label = entry.get("id") or entry.get("name")
        try:
            module = importlib.import_module(entry["module"])
        except Exception as error:  # an entry naming a module that is gone
            problems.append(f"{kind}: catalog entry {label} names module {entry.get('module')!r}, "
                            f"which does not import ({type(error).__name__})")
            continue
        exported = set(getattr(module, "__all__", ()))
        obj = getattr(module, entry["name"], None)
        if not inspect.isfunction(obj) or entry["name"] not in exported:
            problems.append(f"{kind}: catalog entry {label} vanished: "
                            f"{entry['module']}.{entry['name']} is no longer a public function")
            continue
        for alias in entry.get("aliases") or []:
            if getattr(module, alias, None) is not obj or alias not in exported:
                problems.append(f"{kind}: catalog entry {label} lists alias {alias}, "
                                f"which {entry['module']} no longer exports as the same function")
        described.setdefault(id(inspect.unwrap(obj)), set()).update(_entry_names(entry))

    public = _package_functions((package, *extra_namespaces))
    for qualified, obj in _submodule_functions(package, frozenset({"catalog"})).items():
        public.setdefault(qualified, obj)
    for qualified, obj in sorted(public.items()):
        name = qualified.rsplit(".", 1)[-1]
        names = described.get(id(inspect.unwrap(obj)))
        if names is None:
            problems.append(f"{kind}: {obj.__module__}.{obj.__qualname__} is public as "
                            f"{qualified} but no catalog entry describes it")
        elif name not in names:
            problems.append(f"{kind}: {qualified} is public but its catalog entry "
                            f"lists it only as {', '.join(sorted(names))}")
    return problems


def check_formula(snapshot: dict) -> list[str]:
    return check_functions("formula", snapshot, "formulas", "vaft.formula")


def check_process(snapshot: dict) -> list[str]:
    return check_functions("process", snapshot, "functions", "vaft.process")


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def check_plot(snapshot: dict) -> list[str]:
    import warnings

    import vaft.plot as plot
    from vaft.plot import registry

    problems: list[str] = []
    registered = {spec.name: spec for spec in registry.specs(status=None)}
    renderers = {id(spec.renderer): spec.name for spec in registered.values()}
    catalog = {entry["name"]: entry for entry in snapshot.get("plots") or []}
    entry_points = {entry["name"]: entry for entry in snapshot.get("entry_points") or []}

    # The namespace users type.
    expected_entry_points: dict[str, str] = {}
    for name in sorted(plot.__all__):
        obj = getattr(plot, name, None)
        if not inspect.isfunction(obj):
            continue
        if id(obj) in renderers:
            if renderers[id(obj)] not in catalog:
                problems.append(f"plot: vaft.plot.{name} renders registered plot {renderers[id(obj)]}, "
                                f"which is not in plot_catalog.yml")
        elif name.startswith("plot_"):
            expected_entry_points[name] = "support"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)   # a warning name is not offered as API
        for name in sorted(set(dir(plot)) - set(plot.__all__)):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(plot, name)
            except (AttributeError, DeprecationWarning):
                continue
            if inspect.isfunction(obj) and id(obj) not in renderers:
                expected_entry_points[name] = "legacy"
    for name in sorted(set(expected_entry_points) - set(entry_points)):
        problems.append(f"plot: vaft.plot.{name} is a public plotting function but is not in plot_catalog.yml")
    for name in sorted(set(entry_points) - set(expected_entry_points)):
        problems.append(f"plot: catalog entry point {name} vanished: vaft.plot no longer offers it")
    for name in sorted(set(entry_points) & set(expected_entry_points)):
        if entry_points[name].get("status") != expected_entry_points[name]:
            problems.append(f"plot: catalog entry point {name} is recorded as {entry_points[name].get('status')!r}, "
                            f"vaft.plot offers it as {expected_entry_points[name]!r}")

    # The registry, whatever the status.
    for name in sorted(set(registered) - set(catalog)):
        problems.append(f"plot: registered plot {name} is not in plot_catalog.yml")
    for name in sorted(set(catalog) - set(registered)):
        problems.append(f"plot: catalog entry {name} vanished: the registry no longer holds it")
    for name in sorted(set(catalog) & set(registered)):
        spec, entry = registered[name], catalog[name]
        for field in ("subject", "view", "quantity", "status"):
            if entry.get(field) != getattr(spec, field):
                problems.append(f"plot: catalog entry {name} records {field}={entry.get(field)!r}, "
                                f"the registry says {getattr(spec, field)!r}")
    for name, spec in sorted(registered.items()):
        if spec.status == "canonical" and (name not in plot.__all__ or getattr(plot, name, None) is not spec.renderer):
            problems.append(f"plot: canonical plot {name} is not bound as vaft.plot.{name}")
    return problems


def check_plot_thumbnails(snapshot: dict, root: Path) -> list[str]:
    """Missing, orphaned or hand-edited thumbnails fail; stale ones are only reported.

    Structural only (``full=False``): the samples are not loaded here.
    """
    assets = root / "docs" / "assets" / "plots"
    if not assets.is_dir():
        return []
    try:
        from vaft.plot import docs_thumbnails
    except ImportError:
        return ["plot: docs/assets/plots is committed but vaft.plot.docs_thumbnails is missing"]
    problems, notes = docs_thumbnails.check(assets, full=False)
    for note in notes:
        print(f"warning: plot thumbnail {note}")
    problems = [f"plot thumbnail {problem}" for problem in problems]
    for entry in snapshot.get("plots") or []:
        if "thumbnail" not in entry:
            problems.append(f"plot: catalog entry {entry['name']} carries no thumbnail although docs/assets/plots exists")
            break
    return problems


# --------------------------------------------------------------------------
# diagram
# --------------------------------------------------------------------------


def _returns_diagram(function) -> bool:
    annotation = inspect.signature(function).return_annotation
    return annotation == "Diagram" or getattr(annotation, "__name__", None) == "Diagram"


def _diagram_functions_on_disk() -> dict[str, object]:
    import vaft.diagram as diagram

    found: dict[str, object] = {}
    for info in sorted(pkgutil.iter_modules(diagram.__path__), key=lambda info: info.name):
        module = importlib.import_module(f"vaft.diagram.{info.name}")
        for name, obj in vars(module).items():
            if (not name.startswith("_") and inspect.isfunction(obj)
                    and obj.__module__ == module.__name__ and _returns_diagram(obj)):
                found[name] = obj
    return found


def check_diagram(snapshot: dict, root: Path) -> list[str]:
    import vaft.diagram as diagram
    from vaft.diagram import build

    problems: list[str] = []
    public = {name for name in diagram.__all__ if inspect.isfunction(getattr(diagram, name, None))}
    builders = {entry["name"]: entry for entry in snapshot.get("builders") or []}
    for name in sorted(public - set(builders)):
        problems.append(f"diagram: public builder vaft.diagram.{name} is not in diagram_catalog.yml")
    for name in sorted(set(builders) - public):
        problems.append(f"diagram: catalog builder {name} vanished: vaft.diagram no longer exports it")
    for name, obj in sorted(_diagram_functions_on_disk().items()):
        if name not in builders:
            problems.append(f"diagram: {obj.__module__}.{name} builds a Diagram but is not in diagram_catalog.yml")
        if getattr(diagram, name, None) is not obj or name not in diagram.__all__:
            problems.append(f"diagram: {obj.__module__}.{name} builds a Diagram but is not exported as vaft.diagram.{name}")

    declared = set(build.CANONICAL)
    committed = {path.name for path in (root / "docs" / "assets" / "diagrams").glob("*.svg")}
    catalogued = {entry["asset"] for entry in snapshot.get("assets") or []}
    for name in sorted(declared - catalogued):
        problems.append(f"diagram: canonical asset {name} is not in diagram_catalog.yml")
    for name in sorted(catalogued - declared):
        problems.append(f"diagram: catalog asset {name} vanished: vaft.diagram.build.CANONICAL no longer declares it")
    for name in sorted(catalogued - committed):
        problems.append(f"diagram: catalog asset {name} has no committed SVG in docs/assets/diagrams")
    for name in sorted(committed - catalogued):
        problems.append(f"diagram: docs/assets/diagrams/{name} is committed but in no catalog entry (orphaned)")

    drawn = {builder for builder, _ in build.CANONICAL.values()}
    for name in sorted(public - drawn):
        problems.append(f"diagram: public builder vaft.diagram.{name} has no canonical asset to show")
    for entry in snapshot.get("assets") or []:
        builder = entry.get("builder")
        if builder in builders and entry["asset"] not in (builders[builder].get("assets") or []):
            problems.append(f"diagram: asset {entry['asset']} is not listed under its builder {builder}")
    return problems


# --------------------------------------------------------------------------
# the Python API reference (#162)
# --------------------------------------------------------------------------


def check_api(snapshot: dict, root: Path) -> list[str]:
    """Every public module is accounted for, and every published object has one entry.

    The module list comes from the files on disk, not from the generator: a
    public module (no ``_`` in its dotted name) either declares ``__all__`` and
    belongs to an inventory page, or is listed under ``undeclared`` in
    ``docs/api_inventory.yml``.  The objects come from each module's
    ``__all__``, read here again, and are matched to entries by identity.
    """
    import pkgutil
    import warnings

    import vaft
    import yaml

    problems: list[str] = []
    inventory = yaml.safe_load((root / "docs" / "api_inventory.yml").read_text(encoding="utf-8")) or {}
    prefixes = {prefix for page in inventory.get("pages") or [] for prefix in page.get("modules") or []}
    undeclared = set(inventory.get("undeclared") or [])

    def on_a_page(name: str) -> bool:
        # the root "vaft" covers only itself, as in vaft._api_catalog.covers
        return any(name == prefix or (prefix != "vaft" and name.startswith(prefix + ".")) for prefix in prefixes)

    names = ["vaft"] + [info.name for info in pkgutil.walk_packages(vaft.__path__, "vaft.")]
    public = sorted(name for name in names if not any(part.startswith("_") for part in name.split(".")))
    #: every published name -> the object it is bound to (kept alive, so no id is ever reused)
    published: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in public:
            module = importlib.import_module(name)
            declared = getattr(module, "__all__", None)
            if declared is None:
                if name not in undeclared:
                    problems.append(f"api: public module {name} declares no __all__ and is not listed under "
                                    f"undeclared in docs/api_inventory.yml; declare what it publishes")
                continue
            if name in undeclared:
                problems.append(f"api: {name} now declares __all__; remove it from undeclared in docs/api_inventory.yml")
            if not on_a_page(name):
                problems.append(f"api: public module {name} belongs to no page of docs/api_inventory.yml")
            for export in declared:
                if export.startswith("_") and not (export.startswith("__") and export.endswith("__")):
                    problems.append(f"api: {name}.__all__ lists private name {export}")
                if not hasattr(module, export):
                    problems.append(f"api: {name}.__all__ lists {export}, which the module does not define")
                    continue
                obj = getattr(module, export)
                if not inspect.ismodule(obj):
                    published[f"{name}.{export}"] = obj
    for name in sorted(undeclared - set(public)):
        problems.append(f"api: {name} is listed under undeclared in docs/api_inventory.yml but is not a public module")

    # Every published name is claimed by exactly one entry -- as its id or one of
    # its exported_as names -- and that entry describes the object the name is
    # bound to: the same object for a function or class, a constant for a constant.
    claimed: dict[str, str] = {}
    for entry in snapshot.get("entries") or []:
        names_of_entry = [entry["id"], *(entry.get("exported_as") or [])]
        if f"{entry['module']}.{entry['name']}" != entry["id"]:
            problems.append(f"api: entry {entry['id']} is not {entry['module']}.{entry['name']}")
        home = published.get(entry["id"])
        if entry["id"] not in published:
            problems.append(f"api: entry {entry['id']} vanished: {entry['module']} no longer exports {entry['name']}")
        for qualified in names_of_entry:
            if qualified in claimed:
                problems.append(f"api: {qualified} is claimed by more than one entry: {claimed[qualified]}, {entry['id']}")
                continue
            claimed[qualified] = entry["id"]
            if qualified not in published:
                if qualified != entry["id"]:
                    problems.append(f"api: entry {entry['id']} lists {qualified}, which is not published")
                continue
            obj = published[qualified]
            is_data = not (inspect.isclass(obj) or callable(obj))
            if (entry.get("kind") == "data") != is_data:
                problems.append(f"api: entry {entry['id']} is a {entry.get('kind')} but {qualified} is "
                                f"{'data' if is_data else 'callable'}")
            elif not is_data and home is not None and obj is not home:
                problems.append(f"api: entry {entry['id']} lists {qualified}, which is a different object")
    for qualified in sorted(set(published) - set(claimed)):
        problems.append(f"api: {qualified} is published but no entry of api_catalog.yml describes it")
    return problems


# --------------------------------------------------------------------------
# source spans (#1069)
# --------------------------------------------------------------------------


def _spans(node, where: str = ""):
    """``(entry, span)`` for every source span in a snapshot: a ``source`` mapping with a ``line``."""
    if isinstance(node, dict):
        where = str(node.get("id") or node.get("name") or where)
        for key, value in node.items():
            if key == "source" and isinstance(value, dict) and "line" in value:
                yield where, value
            else:
                yield from _spans(value, where)
    elif isinstance(node, list):
        for item in node:
            yield from _spans(item, where)


def _definitions(path: Path, cache: dict = {}) -> frozenset[tuple[int, int]]:  # noqa: B006 -- a per-run cache
    """``(first line, last line)`` of every def and class in a file, decorators included."""
    import ast

    if path not in cache:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cache[path] = frozenset(
            (min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)]), node.end_lineno)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    return cache[path]


def check_sources(label: str, snapshot: dict, root: Path, *, located: bool = True) -> list[str]:
    """Every source span names a whole definition in ``root``, and its inline code is those lines.

    The pages link each span as ``blob/<provenance commit>/<path>#L<line>-L<end_line>``
    and show ``code`` beside it (``validate_docs.rb`` checks that); this checks
    the span itself against the tree the catalog was generated from, which is
    that commit's tree.  So a stale or shifted range, a range cut short, or
    inline code that is not the linked lines fails the build.  With ``located``
    every span must have a line: only the API catalog has objects without a
    ``def`` (constants, typing aliases).
    """
    import textwrap

    problems: list[str] = []
    for where, span in _spans(snapshot):
        path, line, end = str(span.get("path") or ""), int(span.get("line") or 0), int(span.get("end_line") or 0)
        if line <= 0:
            if located:
                problems.append(f"{label} {where}: has no source location")
            continue
        file = (root / path).resolve()
        if not path.startswith("vaft/") or not file.is_relative_to(root.resolve() / "vaft") or not file.is_file():
            problems.append(f"{label} {where}: source {path!r} is not a file of the tree being documented")
            continue
        if (line, end) not in _definitions(file):
            problems.append(f"{label} {where}: {path}#L{line}-L{end} is not a whole definition in that file")
            continue
        code = span.get("code") or ""
        lines = file.read_text(encoding="utf-8").split("\n")
        if code and code != textwrap.dedent("\n".join(lines[line - 1:end])).rstrip():
            problems.append(f"{label} {where}: inline source is not {path} lines {line}-{end}")
    return problems


def check_api_sources(snapshot: dict) -> list[str]:
    """Every API function and class defined in ``vaft`` has a source span."""
    import warnings

    problems: list[str] = []
    for entry in snapshot.get("entries") or []:
        if entry.get("kind") == "data" or int((entry.get("source") or {}).get("line") or 0) > 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                target = inspect.unwrap(getattr(importlib.import_module(entry["module"]), entry["name"]))
            except Exception:  # noqa: BLE001 -- check_api reports what does not resolve
                continue
        if (inspect.isfunction(target) or inspect.isclass(target)) and str(target.__module__).startswith("vaft"):
            problems.append(f"api {entry['id']}: has no source location")
    return problems


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


#: generator module -> how to check its output.
CHECKS = {
    "vaft.formula.catalog": lambda snapshot, root: check_formula(snapshot),
    "vaft.process.catalog": lambda snapshot, root: check_process(snapshot),
    "vaft.plot.docs_catalog": lambda snapshot, root: check_plot(snapshot) + check_plot_thumbnails(snapshot, root),
    "vaft.diagram.docs_catalog": check_diagram,
    "vaft._api_catalog": lambda snapshot, root: check_api(snapshot, root) + check_api_sources(snapshot),
}


def foreign_modules(root: Path) -> list[str]:
    """Imported ``vaft`` modules whose file is not under ``root``."""
    root = root.resolve()
    foreign = []
    for name, module in sorted(sys.modules.items()):
        if name != "vaft" and not name.startswith("vaft."):
            continue
        location = getattr(module, "__file__", None)
        if location and not Path(location).resolve().is_relative_to(root):
            foreign.append(f"{name} ({location})")
    return foreign


def check(docs: Path = DOCS, root: Path | None = None) -> list[str]:
    """Every disagreement between this docs tree's catalogs and the importable library.

    ``root`` is the source tree the catalogs describe, ``docs/..`` unless given;
    every imported ``vaft`` module has to be that tree's copy or nothing is judged.
    """
    import vaft

    root = (root or docs.resolve().parent).resolve()
    problems: list[str] = []
    imported_from = Path(vaft.__file__).resolve().parents[1]
    if imported_from != root:
        problems.append(f"vaft was imported from {imported_from}, not from the tree being documented ({root}); "
                        f"run with PYTHONPATH={root}")
        return problems
    manifest = docs / "generators.yml"
    generators = ((_load(manifest) or {}).get("generators") or []) if manifest.is_file() else []
    for generator in generators:
        checker = CHECKS.get(generator["module"])
        if checker is None:
            continue
        output = docs / generator["output"]
        if not output.is_file():
            problems.append(f"{generator['output']} is declared but was not generated")
            continue
        snapshot = _load(output) or {}
        problems.extend(checker(snapshot, root))
        problems.extend(check_sources(generator["module"], snapshot, root,
                                      located=generator["module"] != "vaft._api_catalog"))
    for module in foreign_modules(root):
        problems.append(f"{module} was imported from outside the tree being documented ({root})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--docs", type=Path, default=DOCS, help="docs/ directory to check (default: %(default)s)")
    arguments = parser.parse_args(argv)
    problems = check(arguments.docs)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"catalog coverage: {len(problems)} problems", file=sys.stderr)
        return 1
    print("catalog coverage: every public formula, process, plot, diagram and API object is catalogued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
