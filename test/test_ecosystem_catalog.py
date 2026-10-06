"""The dependency and external-code catalog (#1648) cannot drift from the repository.

Every fact :mod:`vaft._ecosystem` states is held against its owner here:
package names and extras against ``pyproject.toml`` (and no version is
restated), adapters, installers and checkers against the tree, ``{CODE}HOME``
variables against the adapters' own constants, standardized results against
``STAGE_REPLICATION`` or a callable mapper, workflow rules against the
Snakefiles, and the generated snapshot and figures against the registry.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

from vaft import _ecosystem as ecosystem
from vaft import _ecosystem_catalog as catalog

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INSTALL = ROOT / "install"


@pytest.fixture(scope="module")
def snapshot():
    return catalog.ecosystem_snapshot({"commit": "0" * 40, "ref": "test"})


@pytest.fixture(scope="module")
def project():
    return catalog.load_project()


def test_dependencies_and_extras_are_exactly_pyprojects(project):
    runtime = {catalog.requirement_name(r) for r in project["dependencies"]}
    assert set(ecosystem.RUNTIME_ROLES) == runtime
    assert set(ecosystem.EXTRA_ROLES) == set(project["optional-dependencies"])
    capabilities = {c.id for c in ecosystem.CAPABILITIES}
    for capability, _ in [*ecosystem.RUNTIME_ROLES.values(), *ecosystem.EXTRA_ROLES.values()]:
        assert capability in capabilities
    runtime_caps = {c for c, _ in ecosystem.RUNTIME_ROLES.values()}
    assert all(next(c for c in ecosystem.CAPABILITIES if c.id == cap).scope == "runtime" for cap in runtime_caps)


def test_the_registry_restates_no_version_constraint():
    text = (ROOT / "vaft" / "_ecosystem.py").read_text(encoding="utf-8")
    text = re.sub(r"10\.\d{4,9}/\S+", "", text)  # DOIs are identifiers, not versions
    assert not re.search(r"(?:>=|<=|===|==|~=|!=|[<>])\s*\d|\b\d+\.\d+\.\d+\b", text)


def test_the_snapshot_carries_pyprojects_requirements_verbatim(snapshot, project):
    assert [d["requirement"] for d in snapshot["dependencies"]] == list(project["dependencies"])
    for row in snapshot["extras"]:
        assert row["requirements"] == list(project["optional-dependencies"][row["name"]])


def test_every_adapter_exists_and_every_home_is_the_adapters_own():
    from vaft._help._providers import CODES

    provider_homes = {variable for _, variable, _ in CODES}
    for code in ecosystem.EXTERNAL_CODES:
        assert importlib.util.find_spec(code.adapter) is not None, code.adapter
        if not code.home:
            continue
        variable = catalog.home_variable(code.home)
        # a {CODE}HOME install root, or the interpreter of a code run in its own environment
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*(?:HOME|_PYTHON)", variable), (code.id, variable)
        if ":" not in code.home:  # a literal is only allowed where the help table states the same name
            assert variable in provider_homes, code.id
    # every code the help table probes is in the catalog, under the same variable
    catalogued = {catalog.home_variable(c.home) for c in ecosystem.EXTERNAL_CODES if c.home}
    assert provider_homes <= catalogued


def test_installers_and_checkers_exist_and_every_installer_is_catalogued():
    named = set()
    for code in ecosystem.EXTERNAL_CODES:
        for path in (*code.installers, *([code.checker] if code.checker else [])):
            assert (ROOT / path).is_file(), (code.id, path)
            named.add(path)
        if code.installation == "vaft_managed_source_build":
            assert code.installers, code.id
        if code.installation in {"site_managed", "reader_only"}:
            assert not code.installers, code.id
        if code.installation == "python_package":
            assert code.extra in ecosystem.EXTRA_ROLES, code.id
    on_disk = {p.relative_to(ROOT).as_posix() for p in INSTALL.glob("install_*.sh")}
    # every code installer, on any platform (VAFT's own bootstrap scripts are not code installers)
    on_disk |= {p.relative_to(ROOT).as_posix() for p in INSTALL.glob("install_*.ps1")}
    on_disk |= {p.relative_to(ROOT).as_posix() for p in INSTALL.glob("*/[lmw]*.sh")
                if p.stem in {"linux", "macos", "windows"}}
    on_disk |= {p.relative_to(ROOT).as_posix() for p in INSTALL.glob("*/windows.ps1")}
    on_disk |= {p.relative_to(ROOT).as_posix() for p in INSTALL.glob("check_*.py")
                if not p.stem.startswith("check_vaft")}
    assert on_disk - named == set(), "an installer or checker has no catalog entry"


def test_build_records_are_exactly_the_installers_that_write_one():
    marker = re.compile(r"VAFT_EXTERNAL_MANIFEST_NAME|vaft-external-install\.json|Write-InstallManifest")
    for code in ecosystem.EXTERNAL_CODES:
        writers = {p for p in code.installers if marker.search((ROOT / p).read_text(encoding="utf-8"))}
        assert set(code.provenance) == writers, code.id


def test_modes_match_how_the_adapter_runs_the_code():
    for code in ecosystem.EXTERNAL_CODES:
        spec = importlib.util.find_spec(code.adapter)
        files = ([Path(spec.origin)] if not spec.submodule_search_locations
                 else [p for d in spec.submodule_search_locations for p in Path(d).rglob("*.py")])
        source = "\n".join(p.read_text(encoding="utf-8") for p in files)
        launches = "resolve_backend(" in source
        assert launches == (code.mode == "subprocess_executable"), (code.id, code.mode)


def test_standardized_results_resolve_to_a_stage_or_a_mapper():
    from vaft.database.sources import STAGE_REPLICATION

    for code in ecosystem.EXTERNAL_CODES:
        for entry in code.standardized:
            kind, _, target = entry.via.partition(":")
            if kind == "stage":
                assert tuple(STAGE_REPLICATION[target].ids) == entry.ids, (code.id, target)
            else:
                assert kind == "mapper"
                module, _, function = target.rpartition(".")
                assert callable(getattr(importlib.import_module(module), function)), target


def test_workflow_rules_exist_in_the_production_snakefiles():
    from vaft import _pipeline_graph

    known = set()
    for pipeline in _pipeline_graph.PIPELINES:
        static, _, templates = _pipeline_graph.rule_locations(ROOT / "workflow" / pipeline.directory / "Snakefile")
        known |= {f"{pipeline.id}:{rule}" for rule in static}
        known |= {(pipeline.id, pattern) for pattern, _ in templates}
    for code in ecosystem.EXTERNAL_CODES:
        for node in code.workflow:
            pipeline, _, rule = node.partition(":")
            assert node in known or any(
                p == pipeline and pattern.match(rule) for p, pattern in (k for k in known if isinstance(k, tuple))
            ), node


def _github_anchor(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def test_install_sections_are_real_install_readme_headings():
    readme = (INSTALL / "README.md").read_text(encoding="utf-8")
    anchors = {_github_anchor(line.lstrip("#")) for line in readme.splitlines() if line.startswith("#")}
    for code in ecosystem.EXTERNAL_CODES:
        if code.install_section:
            assert code.install_section in anchors, (code.id, code.install_section)


def test_links_are_typed_and_references_carry_dois():
    roles = {"repository", "homepage", "documentation", "primary_reference", "method_reference"}
    for code in ecosystem.EXTERNAL_CODES:
        for link in code.links:
            assert link.role in roles, (code.id, link.role)
            assert link.url.startswith("https://"), (code.id, link.url)
            if link.role.endswith("_reference"):
                assert re.fullmatch(r"10\.\d{4,9}/\S+", link.doi), (code.id, link.doi)
                assert link.url == f"https://doi.org/{link.doi}"
        assert code.mode in {"subprocess_executable", "in_process_python", "native_reader"}
        assert code.maturity in {"supported", "experimental", "read_only"}
        assert code.access in {"public", "registration", "not_open_source", "not_stated"}


def test_platforms_come_from_installers_never_from_vaft(snapshot):
    rows = {row["id"]: row for row in snapshot["codes"]}
    assert "Windows" not in rows["genray"]["platforms"]  # no Windows installer
    assert "Windows" not in rows["gacode"]["platforms"]
    assert rows["tes"]["platforms"] == []  # site-managed: nothing is claimed
    assert rows["efit"]["home_variable"] == "EFITHOME"
    assert rows["tokamaker"]["execution"] == ["in process"]
    assert rows["transp"]["execution"] == ["reads finished results"]


def test_the_snapshot_receipt_and_determinism(snapshot):
    spec = importlib.util.spec_from_file_location("vaft_docs_build_for_ecosystem", DOCS / "build.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)
    dumped = catalog._dump(snapshot)
    pairs = build._SOURCE_PAIR.findall(dumped)
    assert {path for path, _ in pairs} >= {"pyproject.toml", "vaft/_ecosystem.py", "vaft/_ecosystem_catalog.py"}
    for relative, digest in pairs:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert catalog._dump(catalog.ecosystem_snapshot({"commit": "0" * 40, "ref": "test"})) == dumped


def test_the_figures_are_drawn_from_the_registry():
    import vaft.diagram as diagram

    software = diagram.software_dependency_ecosystem()
    assert {cap for scope in software.model.values() for cap in scope} == {c.id for c in ecosystem.CAPABILITIES}
    integration = diagram.external_code_integration()
    drawn = {name for names in integration.model["installation"].values() for name in names}
    assert drawn == {code.name for code in ecosystem.EXTERNAL_CODES}
    assert integration.model["lifecycle"] == tuple(key for key, _ in ecosystem.INTEGRATION_LIFECYCLE)


def test_the_docs_build_declares_the_generator_and_both_pages():
    generators = yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))["generators"]
    assert {"module": "vaft._ecosystem_catalog", "output": "_data/ecosystem.yml"} in generators
    for page, route in (("External_codes.md", "/reference/external-codes/"),
                        ("Software_dependencies.md", "/reference/software-dependencies/")):
        assert f"permalink: {route}" in (DOCS / "_guide" / page).read_text(encoding="utf-8")
    readme = (INSTALL / "README.md").read_text(encoding="utf-8")
    assert "/reference/external-codes/" in readme and "external_code_integration.svg" in readme
