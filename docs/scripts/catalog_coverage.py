#!/usr/bin/env python3
"""Fail when a generated reference catalog and the library's public surface disagree.

    cd docs && PYTHONPATH=.. python scripts/catalog_coverage.py

``docs/build.py`` runs this against every track that ships it, right after the
generators, with ``PYTHONPATH`` pointing at that track's source tree.  For
each catalog ``generators.yml`` declares it compares the snapshot with an
enumeration of the public surface that does **not** go through the catalog's
own code path, in both directions:

missing
    a public object no catalog entry describes;
vanished
    a catalog entry whose object is no longer public where it says it lives.

What "public" means is each layer's existing rule, not a new one:

formula / process
    the functions of the package namespace, ``vaft.formula.__all__`` and
    ``vaft.process.__all__`` (plus ``vaft.process.cocos.__all__``, which the
    package does not star-import but callers reach by name).  Classes,
    constants and submodules in the same ``__all__`` are API, not catalog
    entries; they belong to the API reference (#162).  An entry is matched by
    the object its ``module`` and ``name`` resolve to, so an entry that
    describes a different function of the same name does not count.
plot
    every :class:`~vaft.plot.registry.PlotSpec` the registry holds, whatever
    its status, each of which ``vaft.plot`` must also bind to its renderer.
    The support API in ``vaft.plot.__all__`` (view models, ``render_*``
    bodies, helpers) is not a plot and belongs to #162 as well.
diagram
    the functions of ``vaft.diagram.__all__``; the assets
    :data:`vaft.diagram.build.CANONICAL` declares; and the ``*.svg`` files
    actually committed under ``docs/assets/diagrams`` -- the only independent
    source here, and the one that catches an orphaned or missing picture.
    Every public builder must have at least one asset.

The rendered pages are checked against the catalogs by ``validate_docs.rb``;
between the two, a public name cannot be absent from the site without the
build failing.

Exit status is 1 when anything disagrees, and every disagreement is printed.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]


def _load(path: Path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _entry_names(entry: dict) -> list[str]:
    return [entry["name"], *(entry.get("aliases") or [])]


# --------------------------------------------------------------------------
# formula and process: package-level functions against catalog entries
# --------------------------------------------------------------------------


def _public_functions(module_names: tuple[str, ...]) -> dict[str, object]:
    public: dict[str, object] = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if inspect.isfunction(obj):
                public.setdefault(name, obj)
    return public


def check_functions(kind: str, snapshot: dict, rows_key: str, packages: tuple[str, ...]) -> list[str]:
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

    for name, obj in sorted(_public_functions(packages).items()):
        names = described.get(id(inspect.unwrap(obj)))
        if names is None:
            problems.append(f"{kind}: {obj.__module__}.{obj.__qualname__} is public as "
                            f"{packages[0]}.{name} but no catalog entry describes it")
        elif name not in names:
            problems.append(f"{kind}: {packages[0]}.{name} is public but its catalog entry "
                            f"lists it only as {', '.join(sorted(names))}")
    return problems


def check_formula(snapshot: dict) -> list[str]:
    return check_functions("formula", snapshot, "formulas", ("vaft.formula",))


def check_process(snapshot: dict) -> list[str]:
    return check_functions("process", snapshot, "functions", ("vaft.process", "vaft.process.cocos"))


# --------------------------------------------------------------------------
# plot: the registry against the catalog
# --------------------------------------------------------------------------


def check_plot(snapshot: dict) -> list[str]:
    import vaft.plot as plot
    from vaft.plot import registry

    problems: list[str] = []
    registered = {spec.name: spec for spec in registry.specs(status=None)}
    catalog = {entry["name"]: entry for entry in snapshot.get("plots") or []}

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


# --------------------------------------------------------------------------
# diagram: builders, CANONICAL and the committed SVGs
# --------------------------------------------------------------------------


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
# entry point
# --------------------------------------------------------------------------


#: generator module -> how to check its output.
CHECKS = {
    "vaft.formula.catalog": lambda snapshot, root: check_formula(snapshot),
    "vaft.process.catalog": lambda snapshot, root: check_process(snapshot),
    "vaft.plot.docs_catalog": lambda snapshot, root: check_plot(snapshot),
    "vaft.diagram.docs_catalog": check_diagram,
}


def check(docs: Path = DOCS, root: Path | None = None) -> list[str]:
    """Every disagreement between this docs tree's catalogs and the importable library.

    ``root`` is the source tree the catalogs describe, ``docs/..`` unless given;
    the importable ``vaft`` has to be that tree's copy or nothing is judged.
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
        problems.extend(checker(_load(output) or {}, root))
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
    print("catalog coverage: every public formula, process, plot and diagram is catalogued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
