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
    for generator in generators:
        subprocess.run(
            [sys.executable, "-m", generator["module"], "--output", str(source / generator["output"])],
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
