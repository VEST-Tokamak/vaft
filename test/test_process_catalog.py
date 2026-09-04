"""The process catalog: coverage, resolution, conformance and laziness (issue #417).

Built the way ``vaft.formula.catalog`` is, with two deliberate differences
these tests pin.  Selection is by ``__all__`` rather than ``__module__``,
because every process submodule declares one (#249) and because that is what
puts the ``_equilibrium_parametric`` functions under ``equilibrium``, their
documented public location.  And every spec carries ``errors`` and a
``conforming`` flag, because the layer is being brought under the contract
one submodule at a time (#418-#421) and the site has to say which categories
are done.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
import yaml

import vaft.process
from vaft.process import catalog
from vaft.process._docstring import SECTION_VOCABULARY


def _run(code: str, *flags: str) -> str:
    """stdout of ``code`` run in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, *flags, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout


def _loaded_after(statement: str) -> set[str]:
    """The ``vaft.process.*`` and ``vaft._docstring`` modules present after ``statement``."""
    code = (
        "import sys\n"
        f"{statement}\n"
        "print(' '.join(sorted(m for m in sys.modules "
        "if m.startswith('vaft.process.') or m == 'vaft._docstring')))\n"
    )
    return set(_run(code).split())


# --- coverage and resolution -------------------------------------------------


def test_categories_are_the_import_order_plus_cocos():
    assert catalog.CATEGORIES == (*vaft.process._IMPORT_ORDER, "cocos")
    assert "_equilibrium_parametric" not in catalog.CATEGORIES


@pytest.mark.parametrize("category", catalog.CATEGORIES)
def test_catalog_covers_every_function_the_submodule_exports(category):
    module = vaft.process._submodule(category)
    exported = {
        name for name in module.__all__ if inspect.isfunction(getattr(module, name))
    }
    specs = catalog._specs_for(category)
    covered = set(specs) | {alias for spec in specs.values() for alias in spec.aliases}
    assert covered == exported


def test_the_parametric_api_is_catalogued_under_equilibrium():
    """`equilibrium` re-exports `_equilibrium_parametric`; the catalog follows."""
    names = set(catalog._specs_for("equilibrium"))
    assert {"validate_equilibrium", "evaluate_miller", "solve_solovev_constraints"} <= names
    spec = catalog.describe("equilibrium.validate_equilibrium")
    assert spec.module == "vaft.process.equilibrium"


def test_every_category_has_at_least_one_function():
    for doc in catalog.categories():
        assert doc.count > 0, doc.name


def test_bare_name_lookup_finds_the_same_object_the_package_does():
    spec = catalog.describe("repair_clipped_interval")
    assert spec.category == "signal_processing"
    assert vaft.process.repair_clipped_interval is getattr(
        vaft.process.signal_processing, spec.name
    )


def test_qualified_lookup_imports_only_that_category():
    loaded = _loaded_after("import vaft.process; vaft.process.describe('numerical.time_derivative')")
    assert loaded == {"vaft.process.catalog", "vaft.process._docstring", "vaft._docstring",
                      "vaft.process.numerical"}


def test_describe_of_an_unknown_name_points_at_the_discovery_helper():
    with pytest.raises(KeyError, match="list_processes"):
        catalog.describe("no_such_function")
    with pytest.raises(KeyError, match="no process category"):
        catalog.describe("no_such_category.smooth")


def test_list_processes_is_sorted_by_category_order_then_name():
    specs = catalog.list_processes()
    order = {name: index for index, name in enumerate(catalog.CATEGORIES)}
    keys = [(order[spec.category], spec.name) for spec in specs]
    assert keys == sorted(keys)


def test_search_matches_provenance_and_prose_case_insensitively():
    hits = {spec.qualname for spec in catalog.search("VEST.YAML")}
    assert "signal_processing.repair_clipped_interval" in hits
    assert catalog.search("") == catalog.list_processes()


def test_render_shows_signature_units_and_violations():
    text = catalog.describe("numerical.time_derivative").render()
    assert text.startswith("numerical.time_derivative(time, data)")
    assert "Parameters" in text and "Returns" in text
    spec = catalog.describe("numerical.time_derivative")
    if spec.errors:
        assert "Contract violations" in text


# --- conformance ---------------------------------------------------------------


def test_conforming_means_structurally_complete_not_merely_parseable():
    """A one-line docstring parses without error; it must not count as conforming."""
    parsed = catalog.parse_docstring("Just a summary.")
    assert parsed.errors == ()

    def fn(x):
        """Just a summary."""

    violations = catalog._structural_violations(parsed, fn)
    joined = "\n".join(violations)
    assert "signature has ['x']" in joined
    assert "missing Returns section" in joined
    assert "Applicability must open with" in joined


def test_deprecated_shims_need_only_a_summary():
    parsed = catalog.parse_docstring("Deprecated compatibility wrapper for :func:`x`.")

    def fn(a, b):
        pass

    assert catalog._structural_violations(parsed, fn) == []


def test_category_conformance_is_all_or_nothing():
    for doc in catalog.categories():
        assert doc.conforming == (doc.documented == doc.count)
        assert 0 <= doc.documented <= doc.count


# --- namespace hygiene --------------------------------------------------------


def test_catalog_names_are_reachable_on_the_package_but_never_exported():
    for name in sorted(vaft.process._CATALOG_NAMES):
        assert name not in vaft.process.__all__
        assert name in dir(vaft.process)
    assert vaft.process.describe is catalog.describe
    assert vaft.process.catalog is catalog


# --- laziness (each check in its own interpreter) -----------------------------


def test_importing_the_package_loads_neither_catalog_nor_parser():
    assert _loaded_after("import vaft.process") == set()


def test_importing_a_processing_submodule_does_not_load_the_catalog():
    loaded = _loaded_after("import vaft.process.signal_processing")
    assert loaded == {"vaft.process.signal_processing"}


def test_the_star_import_neither_loads_nor_binds_the_catalog():
    output = _run(
        "import sys\n"
        "from vaft.process import *\n"
        "print('describe' in dir(), 'vaft.process.catalog' in sys.modules, "
        "'vaft._docstring' in sys.modules)\n"
    )
    assert output.split() == ["False", "False", "False"]


def test_touching_describe_loads_the_catalog_and_nothing_processing():
    loaded = _loaded_after("import vaft.process; vaft.process.describe")
    assert loaded == {"vaft.process.catalog", "vaft.process._docstring", "vaft._docstring"}


def test_listing_one_category_imports_only_that_submodule():
    loaded = _loaded_after(
        "from vaft.process.catalog import list_processes; list_processes(category='numerical')"
    )
    assert "vaft.process.profile" not in loaded
    assert "vaft.process.magnetics" not in loaded
    assert "vaft.process.numerical" in loaded


def test_the_catalog_refuses_to_run_without_docstrings():
    result = subprocess.run(
        [sys.executable, "-OO", "-c", "import vaft.process as P; P.describe('time_derivative')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr and "-OO" in result.stderr


# --- snapshot ------------------------------------------------------------------

_ROW_KEYS = {
    "id", "name", "category", "module", "signature", "summary", "description",
    "parameters", "returns", "sections", "provenance", "machine_scope",
    "convention_sensitive", "deprecated", "conforming", "aliases", "errors",
}
_CATEGORY_KEYS = {
    "name", "module", "title", "overview", "notation", "conventions",
    "count", "documented", "conforming",
}


def test_snapshot_schema():
    snapshot = catalog.documentation_snapshot()
    assert snapshot["schema_version"] == catalog.SCHEMA_VERSION
    assert set(snapshot) == {"schema_version", "generator", "source", "categories", "functions"}
    assert "vaft.process.catalog" in snapshot["generator"]
    paths = [entry["path"] for entry in snapshot["source"]]
    assert "vaft/process/_equilibrium_parametric.py" in paths, "the private module must be checksummed too"
    assert "vaft/process/__init__.py" not in paths
    for entry in snapshot["source"]:
        assert len(entry["sha256"]) == 64
    assert [doc["name"] for doc in snapshot["categories"]] == list(catalog.CATEGORIES)
    for doc in snapshot["categories"]:
        assert set(doc) == _CATEGORY_KEYS
    ids = [row["id"] for row in snapshot["functions"]]
    assert len(ids) == len(set(ids))
    for row in snapshot["functions"]:
        assert set(row) == _ROW_KEYS, row["id"]
        assert row["conforming"] == (row["errors"] == [])
        assert row["machine_scope"] in (None, "independent", "vest")
        for section in row["sections"]:
            assert section["title"] in SECTION_VOCABULARY, row["id"]
        assert ":func:" not in yaml.safe_dump(row), row["id"]


def test_snapshot_omits_provenance_unless_asked():
    assert "provenance" not in catalog.documentation_snapshot()
    stamped = catalog.documentation_snapshot(provenance={"ref": "develop", "commit": "a" * 40})
    assert stamped["provenance"] == {"commit": "a" * 40, "ref": "develop"}


def test_cli_round_trip(tmp_path):
    output = tmp_path / "process_catalog.yml"
    subprocess.run(
        [sys.executable, "-m", "vaft.process.catalog", "--output", str(output)], check=True
    )
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == catalog.documentation_snapshot()


def test_cli_can_restrict_to_one_category(tmp_path):
    output = tmp_path / "numerical.yml"
    catalog.main(["--output", str(output), "--category", "numerical"])
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert {row["category"] for row in data["functions"]} == {"numerical"}
