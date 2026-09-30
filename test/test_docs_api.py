"""The generated Python API reference (#162) and the check that nothing escapes it.

Python only: ``vaft._api_catalog`` and ``docs/scripts/catalog_coverage.py`` run
before Jekyll, so everything here works without Ruby.  The rendered pages are
checked against the catalog by ``validate_docs.rb`` (``test_docs_site.py``).
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from vaft import _api_catalog as api

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@pytest.fixture(scope="module")
def snapshot():
    return api.documentation_snapshot()


@pytest.fixture(scope="module")
def coverage():
    spec = importlib.util.spec_from_file_location("vaft_docs_api_coverage", DOCS / "scripts" / "catalog_coverage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(snapshot, entry_id):
    return next(entry for entry in snapshot["entries"] if entry["id"] == entry_id)


# --------------------------------------------------------------------------
# what the catalog holds
# --------------------------------------------------------------------------


def test_the_snapshot_is_deterministic_and_verifiable(snapshot):
    assert api.documentation_snapshot() == snapshot
    assert "provenance" not in snapshot
    for source in snapshot["source"]:
        import hashlib

        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert {source["path"] for source in snapshot["source"]} >= {"docs/api_inventory.yml", "vaft/__init__.py"}


def test_every_primary_package_is_represented(snapshot):
    """The packages #162 names, each through a representative object."""
    representatives = {
        "database": "vaft.database",
        "data": "vaft.data",
        "machine-mapping": "vaft.machine_mapping",
        "process": "vaft.process",
        "omas": "vaft.omas",
        "plot": "vaft.plot",
        "code": "vaft.code.efit",
    }
    by_page = {}
    for entry in snapshot["entries"]:
        by_page.setdefault(entry["page"], []).append(entry)
    for page, package in representatives.items():
        assert any(entry["module"].startswith(package) for entry in by_page.get(page, [])), page
    run_efit = _entry(snapshot, "vaft.code.efit.run_efit")
    assert "vaft.code.run_efit" in run_efit["exported_as"]


def test_private_helpers_are_not_published(snapshot):
    for entry in snapshot["entries"]:
        assert not any(part.startswith("_") for part in entry["module"].split(".")), entry["id"]
        assert not entry["name"].startswith("_") or entry["name"].startswith("__"), entry["id"]


def test_every_entry_resolves_to_the_object_its_module_exports(snapshot):
    for entry in snapshot["entries"]:
        module = importlib.import_module(entry["module"])
        assert entry["name"] in module.__all__, entry["id"]


def test_signatures_and_source_lines_come_from_the_source(snapshot):
    from vaft.code.efit import run_efit

    entry = _entry(snapshot, "vaft.code.efit.run_efit")
    parameters = inspect.signature(run_efit).parameters
    assert entry["signature"].startswith("(")
    for name in parameters:
        assert name in entry["signature"]
    target = inspect.unwrap(run_efit)
    _, line = inspect.getsourcelines(target)
    assert entry["source"] == {"path": Path(inspect.getsourcefile(target)).resolve().relative_to(ROOT).as_posix(),
                               "line": line}
    assert entry["summary"] == " ".join((inspect.getdoc(run_efit) or "").split("\n\n")[0].split())


def test_no_memory_address_leaks_into_a_signature(snapshot):
    for entry in snapshot["entries"]:
        assert " at 0x" not in entry["signature"] + entry.get("value", ""), entry["id"]


def test_functions_with_a_scientific_page_are_links(snapshot):
    greenwald = next(entry for entry in snapshot["entries"] if entry["name"] == "greenwald_density")
    assert greenwald["reference"] == "/reference/formula/stability/#greenwald_density"
    plot = _entry(snapshot, "vaft.plot.renderers.lines.plasma_current_time")
    assert plot["reference"] == "/reference/plot/#plasma_current_time"


def test_deprecation_is_read_from_the_source(snapshot):
    deprecated = [entry for entry in snapshot["entries"] if entry["deprecated"]]
    assert deprecated
    for entry in deprecated:
        obj = getattr(importlib.import_module(entry["module"]), entry["name"])
        summary = api.summary_of(obj)
        assert (summary.lower().startswith("deprecated") or getattr(obj, "__deprecated__", None)
                or "moved" in entry["deprecated"] or entry["deprecated"] == summary), entry["id"]
    relocated = [entry for entry in snapshot["entries"] if entry["deprecated_aliases"]]
    assert relocated, "vaft.validation re-exports names that moved to vaft.database.production_qa"


# --------------------------------------------------------------------------
# the coverage check passes on the real tree ...
# --------------------------------------------------------------------------


def test_the_api_is_fully_catalogued(coverage, snapshot):
    assert coverage.check_api(snapshot, ROOT) == []


def _tree_with_inventory(tmp_path, mutate):
    root = tmp_path / "tree"
    (root / "docs").mkdir(parents=True)
    inventory = yaml.safe_load((DOCS / "api_inventory.yml").read_text(encoding="utf-8"))
    mutate(inventory)
    (root / "docs" / "api_inventory.yml").write_text(yaml.safe_dump(inventory), encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# ... and fails when a public module or object escapes it
# --------------------------------------------------------------------------


def test_a_new_module_without_all_is_caught(coverage, snapshot, tmp_path, monkeypatch):
    import vaft

    extra = tmp_path / "pkg"
    extra.mkdir()
    (extra / "brand_new_module.py").write_text('"""New."""\n\ndef helper():\n    """New."""\n', encoding="utf-8")
    monkeypatch.setattr(vaft, "__path__", [*vaft.__path__, str(extra)])
    try:
        problems = coverage.check_api(snapshot, ROOT)
    finally:
        sys.modules.pop("vaft.brand_new_module", None)
    assert problems == ["api: public module vaft.brand_new_module declares no __all__ and is not listed under "
                        "undeclared in docs/api_inventory.yml; declare what it publishes"]


def test_a_stale_undeclared_entry_is_caught(coverage, snapshot, tmp_path):
    root = _tree_with_inventory(tmp_path, lambda inv: inv["undeclared"].extend(["vaft.no_such_module", "vaft.code.efit"]))
    problems = coverage.check_api(snapshot, root)
    assert "api: vaft.no_such_module is listed under undeclared in docs/api_inventory.yml but is not a public module" in problems
    assert "api: vaft.code.efit now declares __all__; remove it from undeclared in docs/api_inventory.yml" in problems


def test_a_module_on_no_page_is_caught(coverage, snapshot, tmp_path):
    def drop_cli(inv):
        inv["pages"] = [page for page in inv["pages"] if page["slug"] != "cli"]
    problems = coverage.check_api(snapshot, _tree_with_inventory(tmp_path, drop_cli))
    assert problems and all(problem.startswith("api: public module vaft.cli") and problem.endswith("belongs to no page "
                                                                                             "of docs/api_inventory.yml")
                            for problem in problems), problems


def test_an_all_naming_something_undefined_is_caught(coverage, snapshot, monkeypatch):
    import vaft.imas

    monkeypatch.setattr(vaft.imas, "__all__", [*vaft.imas.__all__, "no_such_name"])
    assert "api: vaft.imas.__all__ lists no_such_name, which the module does not define" in coverage.check_api(snapshot, ROOT)


def test_a_published_object_missing_from_the_catalog_is_caught(coverage, snapshot):
    mutated = copy.deepcopy(snapshot)
    mutated["entries"] = [entry for entry in mutated["entries"] if entry["id"] != "vaft.code.efit.run_efit"]
    problems = coverage.check_api(mutated, ROOT)
    assert any("run_efit is published but no entry of api_catalog.yml describes it" in problem for problem in problems)


def test_a_catalog_entry_that_vanished_is_caught(coverage, snapshot):
    mutated = copy.deepcopy(snapshot)
    ghost = copy.deepcopy(_entry(mutated, "vaft.code.efit.run_efit"))
    ghost.update(id="vaft.code.efit.run_ghost", name="run_ghost", exported_as=[])
    mutated["entries"].append(ghost)
    assert "api: entry vaft.code.efit.run_ghost vanished: vaft.code.efit no longer exports run_ghost" in \
        coverage.check_api(mutated, ROOT)


def test_a_duplicated_entry_is_caught(coverage, snapshot):
    mutated = copy.deepcopy(snapshot)
    twin = copy.deepcopy(_entry(mutated, "vaft.code.efit.run_efit"))
    twin.update(id="vaft.code.run_efit", module="vaft.code", exported_as=[])
    mutated["entries"].append(twin)
    assert any("is described by more than one entry" in problem for problem in coverage.check_api(mutated, ROOT))
