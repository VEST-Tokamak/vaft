"""Documentation-source checks that need neither Ruby nor a built site.

``scripts/validate_docs.rb`` is the full gate, but it needs Jekyll, a rendered
``_site`` and a Ruby toolchain, so it cannot run in the pytest job that gates
merges.  Everything it checks about the *source* -- navigation, redirects,
resource references, notebook inventory, artifact checksums, provenance markers
-- is checked here instead, in plain Python, so documentation drift fails a
normal ``pytest -q`` on the branch that caused it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="this branch has no docs/ directory")


def _data(name: str):
    return yaml.safe_load((DOCS / "_data" / name).read_text(encoding="utf-8"))


def _front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---", text, re.S)
    return yaml.safe_load(match.group(1)) if match else {}


def _pages() -> list[Path]:
    return sorted(DOCS.glob("_guide/*.md")) + sorted(DOCS.glob("_pages/*.md"))


def _canonical_urls() -> set[str]:
    return {
        item["url"]
        for section in _data("navigation.yml")["sections"]
        for item in section["items"]
    }


# --- navigation and redirects ------------------------------------------------


def test_navigation_ids_and_urls_are_unique():
    items = [item for s in _data("navigation.yml")["sections"] for item in s["items"]]
    for field in ("id", "url"):
        values = [item[field] for item in items]
        duplicates = sorted({v for v in values if values.count(v) > 1})
        assert not duplicates, f"duplicate navigation {field}: {duplicates}"


def test_every_navigation_target_has_a_page():
    """Each canonical URL is produced by a permalink somewhere in the source."""
    permalinks = {
        front.get("permalink")
        for front in (_front_matter(p) for p in _pages())
        if front.get("permalink")
    }
    missing = sorted(_canonical_urls() - permalinks)
    assert not missing, f"navigation points at URLs no page claims: {missing}"


def test_page_migrations_are_unique_and_canonical():
    migrations = _data("page_migrations.yml")
    legacy = [m["legacy_url"] for m in migrations]
    assert len(legacy) == len(set(legacy)), "duplicate legacy_url in page_migrations.yml"
    canonical = _canonical_urls()
    stray = sorted(m["canonical_url"] for m in migrations if m["canonical_url"] not in canonical)
    assert not stray, f"migrations point at non-canonical URLs: {stray}"


def test_every_redirect_page_is_declared_as_a_migration():
    declared = {m["legacy_url"] for m in _data("page_migrations.yml")}
    sources = sorted(DOCS.glob("_redirects/*.md")) + sorted(DOCS.glob("_guide/*.md"))
    sources += [DOCS / "guide/Examples.md"]
    unaccounted = sorted(
        front["permalink"]
        for front in (_front_matter(p) for p in sources if p.is_file())
        # A redirect without a permalink redirects nothing; the Ruby validator
        # drops those too.
        if front.get("layout") == "redirect"
        and front.get("permalink")
        and front["permalink"] not in declared
    )
    assert not unaccounted, f"redirect pages missing from page_migrations.yml: {unaccounted}"


# --- resource references -----------------------------------------------------


def test_resource_references_resolve():
    resources = _data("resources.yml")
    kinds = {
        "notebooks": resources["notebooks"],
        "api": resources["api"],
        "data_sources": resources["data_sources"],
        "outputs": _data("notebook_outputs.yml")["outputs"],
    }
    problems = []
    for page in _pages():
        for kind, ids in (_front_matter(page).get("related") or {}).items():
            if kind not in kinds:
                continue
            for identifier in ids or []:
                if identifier not in kinds[kind]:
                    problems.append(f"{page.name}: unknown {kind} resource {identifier!r}")
    assert not problems, problems


def test_notebook_resources_point_at_real_notebooks():
    missing = [
        f"{identifier}: {notebook['path']}"
        for identifier, notebook in _data("resources.yml")["notebooks"].items()
        if not (ROOT / notebook["path"]).is_file()
    ]
    assert not missing, f"resources.yml names notebooks that do not exist: {missing}"


def test_notebook_inventory_matches_this_branch():
    """The Examples page lists exactly the notebooks the branch actually has.

    This drifted by seven notebooks before the site moved into the repository,
    because the check could only run against a checkout that was not there.
    """
    listed = set(re.findall(r"[A-Za-z0-9_]+\.ipynb", (DOCS / "_guide/Examples.md").read_text(encoding="utf-8")))
    actual = {p.name for p in (ROOT / "notebooks").glob("*.ipynb")}
    assert sorted(actual - listed) == [], "notebooks missing from the inventory"
    assert sorted(listed - actual) == [], "inventory names notebooks that do not exist"


# --- published outputs and their provenance ----------------------------------

LEGACY = "legacy-unreproducible"


def _outputs():
    return _data("notebook_outputs.yml")


def test_published_artifacts_match_their_checksums():
    """The images are the part that is still verifiable, so they stay verified."""
    problems = []
    for identifier, output in _outputs()["outputs"].items():
        for artifact in output.get("artifacts") or []:
            path = DOCS / artifact["path"]
            if not path.is_file():
                problems.append(f"{identifier}: missing artifact {artifact['path']}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact["sha256"]:
                problems.append(f"{identifier}: {artifact['path']} does not match its checksum")
    assert not problems, problems


def test_every_output_names_the_notebook_it_came_from():
    missing = [
        f"{identifier}: {output['notebook_path']}"
        for identifier, output in _outputs()["outputs"].items()
        if not (ROOT / output["notebook_path"]).is_file()
    ]
    assert not missing, missing


def test_legacy_outputs_say_why_and_where_it_is_tracked():
    provenance = _outputs()
    legacy = {i: o for i, o in provenance["outputs"].items() if o.get("verification") == LEGACY}
    if not legacy:
        pytest.skip("no legacy outputs on this branch")
    assert provenance.get("provenance_status") == LEGACY, (
        "outputs are marked legacy but the file is not"
    )
    assert str(provenance.get("provenance_reason", "")).strip(), "legacy provenance needs a reason"
    assert provenance.get("provenance_issue"), "legacy provenance needs an issue number"
    for identifier, output in legacy.items():
        assert str(output.get("verification_reason", "")).strip(), f"{identifier}: no reason"
        assert output.get("verification_issue"), f"{identifier}: no issue number"


def test_outputs_that_are_not_legacy_still_match_their_notebooks():
    """A newly published output has to be genuinely reproducible."""
    problems = []
    for identifier, output in _outputs()["outputs"].items():
        if output.get("verification") == LEGACY:
            continue
        notebook = ROOT / output["notebook_path"]
        if not notebook.is_file():
            problems.append(f"{identifier}: notebook missing")
            continue
        if hashlib.sha256(notebook.read_bytes()).hexdigest() != output["notebook_sha256"]:
            problems.append(
                f"{identifier}: notebook checksum does not match; either regenerate the "
                f"output or mark it {LEGACY} with a reason and an issue"
            )
    assert not problems, problems


# --- configuration -----------------------------------------------------------


def _config(name: str) -> dict:
    return yaml.safe_load((DOCS / name).read_text(encoding="utf-8"))


def test_the_develop_overlay_changes_only_what_it_must():
    overlay = _config("_config.develop.yml")
    assert overlay["baseurl"] == "/vaft/develop"
    assert overlay["track"] == "development"
    assert overlay["noindex"] is True
    assert overlay["stable_url"] == "/vaft/"
    allowed = set(_config("_config.yml")) | {"track", "track_label", "noindex", "stable_url"}
    assert set(overlay) <= allowed, f"overlay introduces unknown keys: {set(overlay) - allowed}"


def test_the_site_config_excludes_tooling():
    """Anything that is not content must not be copied into the published tree."""
    excluded = set(_config("_config.yml")["exclude"])
    required = {
        "scripts", "build.py", "generators.yml", "Gemfile", "Gemfile.lock",
        "package.json", "package-lock.json", "playwright.config.js", "README.md",
        "vendor", "tests", "node_modules",
    }
    assert required <= excluded, f"not excluded from the build: {sorted(required - excluded)}"


def test_the_inert_remote_theme_is_gone():
    """It named a theme that is in neither the plugin list nor the Gemfile.

    Leaving it invites someone to restore it over the vendored layouts, which
    are locally modified and which the whole site depends on.
    """
    assert "remote_theme" not in _config("_config.yml")


def test_generators_declare_modules_this_branch_has():
    import importlib.util

    generators = yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))["generators"]
    assert generators, "generators.yml declares nothing"
    for generator in generators:
        assert importlib.util.find_spec(generator["module"]), (
            f"generators.yml declares {generator['module']}, which this branch does not ship"
        )
        assert generator["output"].startswith("_data/")


def test_generated_data_is_not_committed():
    """The snapshots embed checksums of the library, so committing them means
    every change to the library silently makes the site stale."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "docs/_data"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    generated = {
        "docs/" + g["output"]
        for g in yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))["generators"]
    } | {"docs/_data/provenance.yml"}
    committed = sorted(generated.intersection(tracked))
    assert not committed, f"generated data is committed: {committed}"


def test_visual_baselines_still_point_at_canonical_pages():
    """The Playwright suite is manual-only now, so nothing else notices it rot."""
    spec = DOCS / "tests/visual/docs.spec.js"
    if not spec.is_file():
        pytest.skip("visual suite is not part of this branch")
    canonical = _canonical_urls()
    referenced = set(re.findall(r"'(/vaft/[^']+)'", spec.read_text(encoding="utf-8")))
    stale = sorted(url[len("/vaft"):] for url in referenced
                   if url[len("/vaft"):] not in canonical)
    assert not stale, f"visual specs assert on URLs that are no longer canonical: {stale}"
