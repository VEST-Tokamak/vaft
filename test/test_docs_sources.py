"""Revision-pinned source navigation on the generated reference pages (#1069).

Every generated entry carries a source span, ``{path, line, end_line, code}``,
read by :func:`vaft._docstring.source_span`.  The pages link it as
``blob/<provenance commit>/<path>#L<line>-L<end_line>`` and show ``code``
collapsed beside it.  Here: the spans are what ``inspect`` reads, decorators
included, on representative formula, process, plot, diagram and API objects;
and ``catalog_coverage.check_sources`` rejects a span that is not a whole
definition of the tree, or inline code that is not its lines.  The rendered
links are checked by ``validate_docs.rb`` (``test_docs_site.py``).
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

from vaft import _docstring

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@pytest.fixture(scope="module")
def coverage():
    spec = importlib.util.spec_from_file_location("vaft_docs_sources_coverage", DOCS / "scripts" / "catalog_coverage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def snapshots():
    from vaft import _api_catalog
    from vaft.diagram import docs_catalog as diagram_catalog
    from vaft.formula import catalog as formula_catalog
    from vaft.plot import docs_catalog as plot_catalog
    from vaft.process import catalog as process_catalog

    return {
        "vaft.formula.catalog": formula_catalog.documentation_snapshot(),
        "vaft.process.catalog": process_catalog.documentation_snapshot(),
        "vaft.plot.docs_catalog": plot_catalog.documentation_snapshot(),
        "vaft.diagram.docs_catalog": diagram_catalog.documentation_snapshot(),
        "vaft._api_catalog": _api_catalog.documentation_snapshot(),
    }


def _expected(obj) -> dict:
    target = inspect.unwrap(obj)
    lines, start = inspect.getsourcelines(target)
    return {
        "path": Path(inspect.getsourcefile(target)).resolve().relative_to(ROOT).as_posix(),
        "line": start,
        "end_line": start + len(lines) - 1,
        "code": textwrap.dedent("".join(lines)).rstrip() if len(lines) <= _docstring.INLINE_SOURCE_LINES else "",
    }


# --------------------------------------------------------------------------
# the span itself
# --------------------------------------------------------------------------


def test_a_decorated_function_spans_from_its_first_decorator(tmp_path, monkeypatch):
    package = tmp_path / "pkg_span"
    package.mkdir()
    (package / "__init__.py").write_text(textwrap.dedent('''\
        import functools


        def wrap(f):
            @functools.wraps(f)
            def inner(*a):
                return f(*a)
            return inner


        @wrap
        @wrap
        def decorated(x):
            """Doc."""
            return x


        class Holder:
            def method(self):
                return 1
        '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("pkg_span")
    span = _docstring.source_span(module.decorated, tmp_path)
    assert (span["path"], span["line"], span["end_line"]) == ("pkg_span/__init__.py", 11, 15)
    assert span["code"].splitlines()[0] == "@wrap" and span["code"].splitlines()[-1] == "    return x"
    method = _docstring.source_span(module.Holder.method, tmp_path)
    assert (method["line"], method["end_line"]) == (19, 20)
    assert method["code"] == "def method(self):\n    return 1"  # dedented
    assert _docstring.source_span(module.Holder, tmp_path, inline=False)["code"] == ""


def test_a_long_definition_keeps_the_link_but_not_the_inline_code(tmp_path, monkeypatch):
    body = "\n".join(f"    x{i} = {i}" for i in range(_docstring.INLINE_SOURCE_LINES))
    (tmp_path / "pkg_long.py").write_text(f"def long():\n{body}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    span = _docstring.source_span(importlib.import_module("pkg_long").long, tmp_path)
    assert (span["line"], span["end_line"], span["code"]) == (1, _docstring.INLINE_SOURCE_LINES + 1, "")


def test_an_unlocatable_object_has_an_empty_span(tmp_path):
    namespace: dict = {}
    exec("def made_up():\n    return 1\n", namespace)
    empty = {"path": "", "line": 0, "end_line": 0, "code": ""}
    assert _docstring.source_span(namespace["made_up"], tmp_path) == empty
    assert _docstring.source_span(_docstring.source_span, tmp_path) == empty  # outside the root
    assert _docstring.source_span(None, tmp_path) == empty


# --------------------------------------------------------------------------
# representative entries of every catalog
# --------------------------------------------------------------------------


def test_formula_and_process_entries_carry_their_span(snapshots):
    import vaft.formula.stability as stability
    import vaft.process as process

    formula = next(row for row in snapshots["vaft.formula.catalog"]["formulas"] if row["name"] == "greenwald_density")
    assert formula["source"] == _expected(stability.greenwald_density)
    row = snapshots["vaft.process.catalog"]["functions"][0]
    assert row["source"] == _expected(getattr(importlib.import_module(row["module"]), row["name"]))
    assert all(r["source"]["line"] > 0 and r["source"]["end_line"] >= r["source"]["line"]
               for r in snapshots["vaft.process.catalog"]["functions"])
    assert process  # the namespace imports


def test_a_decorated_plot_renderer_links_from_its_decorator(snapshots):
    """``@presented`` renderers link to the decorator line, and the range covers the whole body."""
    from vaft.plot import registry

    rows = {row["name"]: row for row in snapshots["vaft.plot.docs_catalog"]["plots"]}
    decorated = 0
    for spec in registry.specs(status=None):
        row = rows[spec.name]
        assert row["source"] == _expected(spec.renderer)
        lines = (ROOT / row["source"]["path"]).read_text(encoding="utf-8").split("\n")
        if lines[row["source"]["line"] - 1].lstrip().startswith("@"):
            decorated += 1
            body = lines[row["source"]["line"] - 1:row["source"]["end_line"]]
            assert any(line.lstrip().startswith(f"def {inspect.unwrap(spec.renderer).__name__}(") for line in body)
    assert decorated > 0


def test_diagram_builders_carry_their_span(snapshots):
    import vaft.diagram as diagram

    for row in snapshots["vaft.diagram.docs_catalog"]["builders"]:
        assert row["source"] == _expected(getattr(diagram, row["name"]))


def test_api_functions_methods_and_classes_carry_their_span(snapshots):
    from vaft.code.chease import CHEASEConfig
    from vaft.database import export

    entries = {entry["id"]: entry for entry in snapshots["vaft._api_catalog"]["entries"]}
    assert entries["vaft.database.export"]["source"] == _expected(export)
    config = entries["vaft.code.chease.CHEASEConfig"]
    # a class links its definition; its body is shown member by member
    assert config["source"] == {**_expected(CHEASEConfig), "code": ""}
    members = {member["name"]: member for member in config["members"]}
    assert members["resolved_mesh"]["source"] == _expected(CHEASEConfig.resolved_mesh.fget)
    # a function with a scientific page is shown there, so the API page links only
    linked = next(entry for entry in entries.values() if entry["reference"])
    assert linked["source"]["line"] > 0 and linked["source"]["code"] == ""


def test_every_catalog_span_is_a_whole_definition_of_the_tree(coverage, snapshots):
    for module, snapshot in snapshots.items():
        assert coverage.check_sources(module, snapshot, ROOT, located=module != "vaft._api_catalog") == [], module
    assert coverage.check_api_sources(snapshots["vaft._api_catalog"]) == []


# --------------------------------------------------------------------------
# what the check rejects
# --------------------------------------------------------------------------


def _mutated(snapshot, mutate):
    changed = copy.deepcopy(snapshot)
    mutate(next(row for row in changed["formulas"] if row["name"] == "greenwald_density")["source"])
    return changed


@pytest.mark.parametrize("mutate, message", [
    (lambda src: src.update(line=src["line"] + 1), "is not a whole definition in that file"),
    (lambda src: src.update(end_line=src["end_line"] - 1), "is not a whole definition in that file"),
    (lambda src: src.update(code=src["code"].replace("def ", "def x", 1)), "inline source is not vaft/formula/stability.py lines"),
    (lambda src: src.update(path="vaft/formula/nowhere.py"), "is not a file of the tree being documented"),
    (lambda src: src.update(path="../outside.py"), "is not a file of the tree being documented"),
    (lambda src: src.update(line=0, end_line=0), "greenwald_density: has no source location"),
])
def test_a_stale_or_forged_span_is_caught(coverage, snapshots, mutate, message):
    problems = coverage.check_sources("vaft.formula.catalog", _mutated(snapshots["vaft.formula.catalog"], mutate), ROOT)
    assert any(message in problem for problem in problems), problems


def test_an_api_function_without_a_span_is_caught(coverage, snapshots):
    changed = copy.deepcopy(snapshots["vaft._api_catalog"])
    entry = next(entry for entry in changed["entries"] if entry["id"] == "vaft.database.export")
    entry["source"] = {"path": "", "line": 0, "end_line": 0, "code": ""}
    assert coverage.check_api_sources(changed) == ["api vaft.database.export: has no source location"]
    # the span check itself allows it in the API catalog, where constants have no def
    assert coverage.check_sources("vaft._api_catalog", changed, ROOT, located=False) == []
