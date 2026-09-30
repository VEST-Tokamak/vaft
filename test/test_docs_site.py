"""End-to-end documentation checks that need Ruby, Bundler and Jekyll.

These skip themselves when the toolchain is absent, so a bare ``pytest -q`` in
a Python-only environment stays green.  Bundler being on ``PATH`` is not enough:
``docs/Gemfile`` also has to be installed, which is probed rather than assumed.

The site is built from a copy in ``tmp_path`` so the checkout is never written
to, matching how ``docs/build.py`` works in earnest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: Resolved once, and used instead of the bare name everywhere below. See the
#: note in _jekyll_available().
BUNDLE = shutil.which("bundle")


def _jekyll_available() -> bool:
    if not DOCS.is_dir() or shutil.which("ruby") is None or BUNDLE is None:
        return False
    try:
        # The resolved path, not the bare name. `shutil.which` searches the
        # whole of PATHEXT and finds `bundle.bat`, but CreateProcess only ever
        # appends `.exe` -- which is why a bare "git" works on Windows and a
        # bare "bundle" raises WinError 2 instead of reporting a non-zero exit.
        probe = subprocess.run(
            [BUNDLE, "exec", "jekyll", "--version"],
            cwd=str(DOCS), capture_output=True, text=True,
        )
    except OSError:
        # This runs at import time to decide a skipif, so whatever is wrong
        # with the toolchain it has to answer False rather than fail
        # collection for the whole module.
        return False
    return probe.returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DOCS.is_dir(), reason="this branch has no docs/ directory"),
    pytest.mark.skipif(
        not _jekyll_available(),
        reason="ruby, bundler and an installed docs/Gemfile are required",
    ),
]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Generate this branch's data and build both tracks from a copy."""
    workspace = tmp_path_factory.mktemp("docs-site")
    source = workspace / "docs"
    shutil.copytree(
        DOCS, source,
        ignore=shutil.ignore_patterns("_site*", ".jekyll-cache", "vendor", "node_modules", ".bundle"),
    )

    import yaml

    generators = yaml.safe_load((source / "generators.yml").read_text(encoding="utf-8"))["generators"]
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    # Source links are pinned to the generating commit (#1069), as docs/build.py records it.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
                            capture_output=True, text=True).stdout.strip()
    for generator in generators:
        subprocess.run(
            [sys.executable, "-m", generator["module"], "--output", str(source / generator["output"]),
             "--provenance-commit", commit, "--provenance-ref", "test"],
            cwd=str(ROOT), env=environment, check=True, capture_output=True,
        )

    builds = {}
    for track, configs, baseurl in (
        ("stable", "_config.yml", "/vaft"),
        ("development", "_config.yml,_config.develop.yml", "/vaft/develop"),
    ):
        destination = workspace / f"site-{track}"
        subprocess.run(
            [
                BUNDLE, "exec", "jekyll", "build",
                "--source", str(source),
                "--config", ",".join(str(source / c) for c in configs.split(",")),
                "--baseurl", baseurl,
                "--destination", str(destination),
            ],
            cwd=str(DOCS), check=True, capture_output=True, text=True,
        )
        builds[track] = (destination, baseurl)
    return source, builds


def _validate(source: Path, site_dir: Path, baseurl: str) -> subprocess.CompletedProcess:
    environment = dict(
        os.environ,
        VAFT_DOCS_BASEURL=baseurl,
        VAFT_DOCS_SITE=str(site_dir),
        VAFT_NOTEBOOK_SOURCE=str(ROOT),
        VAFT_REGISTRY_SOURCE=str(ROOT),
    )
    return subprocess.run(
        ["ruby", "scripts/validate_docs.rb"],
        cwd=str(source), env=environment, capture_output=True, text=True,
    )


def test_the_stable_track_builds_and_validates(site):
    source, builds = site
    destination, baseurl = builds["stable"]
    assert (destination / "index.html").is_file()
    result = _validate(source, destination, baseurl)
    assert result.returncode == 0, result.stderr or result.stdout


def test_the_development_track_builds_and_validates(site):
    source, builds = site
    destination, baseurl = builds["development"]
    result = _validate(source, destination, baseurl)
    assert result.returncode == 0, result.stderr or result.stdout


def test_the_validator_rejects_a_baseurl_the_site_was_not_built_with(site):
    """The regression test for the failure that prompted this whole change.

    The baseurl used to be the literal string "/vaft" in two places. Building
    the development track and validating it as though it were the stable one
    used to strip the wrong prefix and report every link as broken -- silently
    wrong, and indistinguishable from a content error. It must now fail loudly.
    """
    source, builds = site
    destination, _ = builds["development"]
    result = _validate(source, destination, "/vaft")
    assert result.returncode != 0
    assert "canonical" in (result.stderr + result.stdout) or "broken internal link" in (
        result.stderr + result.stdout
    )


def test_the_development_track_marks_itself(site):
    source, builds = site
    destination, _ = builds["development"]
    home = (destination / "index.html").read_text(encoding="utf-8")
    assert 'name="robots"' in home and "noindex" in home
    assert "Development documentation" in home
    assert "/vaft/" in home, "the banner has to link back to the stable site"


def test_the_stable_track_does_not(site):
    source, builds = site
    destination, _ = builds["stable"]
    home = (destination / "index.html").read_text(encoding="utf-8")
    assert "noindex" not in home
    assert "Development documentation" not in home


def test_redirect_pages_carry_the_track_they_belong_to(site):
    """redirect.html builds its own <head>, so it is the easy one to forget."""
    source, builds = site
    for track, expected in (("stable", False), ("development", True)):
        destination, _ = builds[track]
        redirects = [
            path for path in destination.rglob("index.html")
            if "http-equiv=\"refresh\"" in path.read_text(encoding="utf-8")
        ]
        assert redirects, f"{track}: no redirect pages were built"
        for path in redirects:
            text = path.read_text(encoding="utf-8")
            assert ("noindex" in text) is expected, f"{track}: {path.name}"


def test_no_tooling_is_published(site):
    source, builds = site
    for track, (destination, _) in builds.items():
        for unwanted in ("Gemfile", "package.json", "playwright.config.js", "build.py",
                         "generators.yml", "scripts", "README.md"):
            assert not (destination / unwanted).exists(), f"{track} published {unwanted}"


@pytest.mark.parametrize("page, entry, message", [
    ("reference/plot/index.html", "plasma_current_time",
     "plot plasma_current_time is in the catalog but not rendered on /vaft/develop/reference/plot/"),
    ("reference/diagram/index.html", "hugill",
     "diagram hugill is in the catalog but not rendered on /vaft/develop/reference/diagram/"),
    ("reference/formula/stability/index.html", "greenwald_density",
     "formula greenwald_density is in the catalog but not rendered on /vaft/develop/reference/formula/stability/"),
    ("reference/api/code/index.html", "vaft.code.efit.run_efit",
     "api vaft.code.efit.run_efit is in the catalog but not rendered on /vaft/develop/reference/api/code/"),
])
def test_a_catalog_entry_missing_from_its_rendered_page_is_caught(site, tmp_path, page, entry, message):
    """validate_docs.rb reads the built HTML, so a page that drops an entry fails the build."""
    source, builds = site
    destination, baseurl = builds["development"]
    if not (source / "_data" / "plot_catalog.yml").is_file():
        pytest.skip("this branch does not generate the plot and diagram catalogs")
    mutated = tmp_path / "site"
    shutil.copytree(destination, mutated)
    html = (mutated / page).read_text(encoding="utf-8")
    marker = f'id="{entry}" data-catalog='
    assert marker in html
    (mutated / page).write_text(html.replace(marker, f'id="{entry}-removed" data-catalog='), encoding="utf-8")
    result = _validate(source, mutated, baseurl)
    assert result.returncode != 0
    output = result.stderr + result.stdout
    assert message in output, output
    assert f"{entry}-removed is rendered on" in output


def test_a_generated_page_without_its_data_is_caught(site, tmp_path):
    """Dropping a generator from generators.yml must not leave an empty page that validates."""
    source, builds = site
    destination, baseurl = builds["development"]
    if not (source / "_data" / "plot_catalog.yml").is_file():
        pytest.skip("this branch does not generate the plot and diagram catalogs")
    copy = tmp_path / "docs"
    shutil.copytree(source, copy)
    (copy / "_data" / "plot_catalog.yml").unlink()
    result = _validate(copy, destination, baseurl)
    assert result.returncode != 0
    assert "_guide/Plot_reference.md is published but _data/plot_catalog.yml was not generated" in (
        result.stderr + result.stdout)


def test_a_thumbnail_missing_from_the_page_is_caught(site, tmp_path):
    source, builds = site
    destination, baseurl = builds["development"]
    if not (source / "assets" / "plots").is_dir():
        pytest.skip("this branch commits no plot thumbnails")
    mutated = tmp_path / "site"
    shutil.copytree(destination, mutated)
    page = mutated / "reference" / "plot" / "index.html"
    html = page.read_text(encoding="utf-8")
    marker = 'data-thumbnail="plasma_current_time"'
    assert marker in html
    page.write_text(html.replace(marker, 'data-thumbnail-removed="plasma_current_time"'), encoding="utf-8")
    result = _validate(source, mutated, baseurl)
    assert result.returncode != 0
    assert "plot thumbnail plasma_current_time is in the catalog but not rendered on /vaft/develop/reference/plot/" in (
        result.stderr + result.stdout)


# --------------------------------------------------------------------------
# source navigation (#1069)
# --------------------------------------------------------------------------


def _source_pages(source):
    if not (source / "_data" / "api_catalog.yml").is_file():
        pytest.skip("this branch does not generate the API catalog")


def test_every_source_link_is_pinned_to_the_generating_commit(site):
    """No generated page links to a branch; every link names a line range at the commit."""
    import re

    source, builds = site
    _source_pages(source)
    destination, _ = builds["development"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
                            capture_output=True, text=True).stdout.strip()
    pinned = re.compile(rf'href="https://github.com/VEST-Tokamak/vaft/blob/{commit}/vaft/[^"#]+\.py#L(\d+)-L(\d+)"')
    counted = 0
    for page in (destination / "reference").rglob("index.html"):
        for anchor in re.findall(r'<a class="ref-source"[^>]*>', page.read_text(encoding="utf-8")):
            match = pinned.search(anchor)
            assert match, (page, anchor)
            assert int(match.group(1)) <= int(match.group(2))
            counted += 1
    assert counted > 1000


@pytest.mark.parametrize("mutation, message", [
    # a moving branch instead of the commit
    (lambda html, commit: html.replace(f"/blob/{commit}/", "/blob/develop/", 1), "source link is https://github.com/VEST-Tokamak/vaft/blob/develop/"),
    # a range cut short
    (lambda html, commit: __import__("re").sub(r'(data-source="vaft\.database\.export" href="[^"]*#L\d+-L)(\d+)',
                                               lambda m: m.group(1) + str(int(m.group(2)) - 1), html, count=1),
     "api vaft.database.export: source link is"),
    # inline code that is not the linked lines
    (lambda html, commit: html.replace('<span class="nf">export</span>', '<span class="nf">exported</span>', 1),
     "api vaft.database.export: inline source on /vaft/develop/reference/api/database/ differs from the catalog's"),
])
def test_a_wrong_source_link_or_inline_source_is_caught(site, tmp_path, mutation, message):
    source, builds = site
    _source_pages(source)
    destination, baseurl = builds["development"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
                            capture_output=True, text=True).stdout.strip()
    mutated = tmp_path / "site"
    shutil.copytree(destination, mutated)
    page = mutated / "reference" / "api" / "database" / "index.html"
    html = page.read_text(encoding="utf-8")
    changed = mutation(html, commit)
    assert changed != html
    page.write_text(changed, encoding="utf-8")
    result = _validate(source, mutated, baseurl)
    assert result.returncode != 0
    assert message in result.stderr + result.stdout, result.stderr + result.stdout


def test_a_catalog_without_a_provenance_commit_is_caught(site, tmp_path):
    """Without the commit there is nothing to pin to, so the build fails rather than link a branch."""
    import yaml

    source, builds = site
    _source_pages(source)
    destination, baseurl = builds["development"]
    copy = tmp_path / "docs"
    shutil.copytree(source, copy)
    target = copy / "_data" / "formula_catalog.yml"
    snapshot = yaml.safe_load(target.read_text(encoding="utf-8"))
    snapshot.pop("provenance")
    target.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    result = _validate(copy, destination, baseurl)
    assert result.returncode != 0
    assert "formula snapshot records no provenance commit, so its source links cannot be pinned" in (
        result.stderr + result.stdout)
