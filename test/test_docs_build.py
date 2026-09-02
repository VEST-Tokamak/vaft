"""Unit tests for ``docs/build.py``.

The parts that need Ruby and Jekyll live in ``test_docs_site.py`` and skip
without them.  What is covered here is everything that decides whether the
published site is correct and whether a failure can damage anything:
composition, the composed-tree invariants, provenance, and the publish step.

Publishing is exercised against a real git repository created in ``tmp_path``.
That is offline, takes well under a second, and is the only way to prove the
properties that matter -- that a publish is exactly one commit on the previous
tip, that a concurrent publish is refused rather than forced, and that nothing
outside the temporary directory is touched.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docs" / "build.py"

pytestmark = pytest.mark.skipif(not BUILD.is_file(), reason="this branch has no docs/build.py")


@pytest.fixture(scope="module")
def build():
    """Load docs/build.py by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("vaft_docs_build", BUILD)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: build.py uses `from __future__ import
    # annotations`, and dataclasses resolves those string annotations through
    # sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# synthetic tracks
# --------------------------------------------------------------------------

STABLE_PAGE = """<!DOCTYPE html><html><head><title>Stable</title></head>
<body><a href="/vaft/guide/">guide</a><img src="/vaft/assets/x.png"></body></html>
"""

DEVELOP_PAGE = """<!DOCTYPE html><html><head><meta name="robots" content="noindex, nofollow">
<title>Development</title></head>
<body><a href="/vaft/develop/guide/">guide</a></body></html>
"""


def _track(build, name, tmp_path, commit):
    spec = build.TRACKS[name]
    site = tmp_path / f"site-{name}"
    (site / "guide").mkdir(parents=True)
    page = STABLE_PAGE if name == "stable" else DEVELOP_PAGE
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / "guide" / "index.html").write_text(page, encoding="utf-8")
    if name == "stable":
        (site / "assets").mkdir()
        (site / "assets" / "x.png").write_bytes(b"\x89PNG")
    return build.Track(
        spec=spec,
        ref=spec.default_ref,
        commit=commit,
        commit_date="2026-01-01T00:00:00+00:00",
        root=tmp_path / f"src-{name}",
        site=site,
        vaft_version="0.6.0",
        pages=2,
        outputs=["_data/vest_diagnostics.yml"],
    )


@pytest.fixture
def tracks(build, tmp_path):
    return [
        _track(build, "stable", tmp_path, "a" * 40),
        _track(build, "development", tmp_path, "b" * 40),
    ]


@pytest.fixture
def composed(build, tracks, tmp_path):
    return build.compose(tracks, tmp_path / "composed")


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def test_compose_puts_the_stable_track_at_the_root_and_develop_beneath_it(composed):
    assert (composed / "index.html").is_file()
    assert (composed / "develop" / "index.html").is_file()
    assert "Stable" in (composed / "index.html").read_text()
    assert "Development" in (composed / "develop" / "index.html").read_text()


def test_compose_writes_the_files_that_make_the_branch_publishable(composed):
    assert (composed / ".nojekyll").is_file(), "without it Pages would try to build the output"
    assert (composed / "provenance.yml").is_file()
    readme = (composed / "README.md").read_text()
    assert "generated output" in readme and "docs/" in readme


def test_provenance_names_both_tracks_and_their_commits(composed, tracks):
    recorded = yaml.safe_load((composed / "provenance.yml").read_text())["tracks"]
    assert recorded["stable"]["commit"] == "a" * 40
    assert recorded["development"]["commit"] == "b" * 40
    assert recorded["stable"]["prefix"] == "/"
    assert recorded["development"]["prefix"] == "/develop/"
    assert recorded["development"]["baseurl"] == "/vaft/develop"


def test_compose_is_repeatable(build, tracks, tmp_path):
    first = build.compose(tracks, tmp_path / "one")
    names = sorted(p.relative_to(first).as_posix() for p in first.rglob("*"))
    second = build.compose(tracks, tmp_path / "one")  # same destination, rebuilt
    assert sorted(p.relative_to(second).as_posix() for p in second.rglob("*")) == names


# --------------------------------------------------------------------------
# the composed-tree invariants, one broken thing at a time
# --------------------------------------------------------------------------


def test_a_good_tree_validates(build, composed, tracks):
    build.validate_composed(composed, tracks)


def _expect_failure(build, composed, tracks, fragment):
    with pytest.raises(build.BuildError) as failure:
        build.validate_composed(composed, tracks)
    assert fragment in str(failure.value)


def test_a_missing_track_index_is_caught(build, composed, tracks):
    (composed / "develop" / "index.html").unlink()
    _expect_failure(build, composed, tracks, "no index.html")


def test_a_missing_nojekyll_is_caught(build, composed, tracks):
    (composed / ".nojekyll").unlink()
    _expect_failure(build, composed, tracks, "missing .nojekyll")


def test_provenance_that_disagrees_with_the_build_is_caught(build, composed, tracks):
    data = yaml.safe_load((composed / "provenance.yml").read_text())
    data["tracks"]["development"]["commit"] = "f" * 40
    (composed / "provenance.yml").write_text(yaml.safe_dump(data))
    _expect_failure(build, composed, tracks, "provenance.yml records")


def test_a_dangling_absolute_link_is_caught(build, composed, tracks):
    page = composed / "index.html"
    page.write_text(page.read_text().replace("/vaft/guide/", "/vaft/gone/"))
    _expect_failure(build, composed, tracks, "unresolved link")


def test_a_development_page_without_noindex_is_caught(build, composed, tracks):
    page = composed / "develop" / "index.html"
    page.write_text(page.read_text().replace('<meta name="robots" content="noindex, nofollow">', ""))
    _expect_failure(build, composed, tracks, "development page without noindex")


def test_a_stable_page_carrying_noindex_is_caught(build, composed, tracks):
    page = composed / "index.html"
    page.write_text(page.read_text().replace("<head>", '<head><meta name="robots" content="noindex">'))
    _expect_failure(build, composed, tracks, "stable page carries noindex")


def test_a_stable_page_linking_into_the_development_track_is_caught(build, composed, tracks):
    page = composed / "index.html"
    page.write_text(page.read_text().replace('href="/vaft/guide/"', 'href="/vaft/develop/guide/"'))
    _expect_failure(build, composed, tracks, "links into the development track")


@pytest.mark.parametrize("leaked", ["Gemfile", "package.json", "build.py", "generators.yml"])
def test_leaked_tooling_is_caught(build, composed, tracks, leaked):
    (composed / leaked).write_text("x")
    _expect_failure(build, composed, tracks, "tooling leaked")


def test_an_oversized_tree_is_caught(build, composed, tracks, monkeypatch):
    monkeypatch.setattr(build, "MAX_SITE_MIB", 0)
    _expect_failure(build, composed, tracks, "over the")


# --------------------------------------------------------------------------
# the command line refuses what it cannot do safely
# --------------------------------------------------------------------------


def test_publishing_a_single_track_is_refused(build, capsys):
    assert build.main(["--track", "development", "--publish"]) == build.EXIT_REFUSED
    message = capsys.readouterr().err
    assert "refusing to publish a single track" in message
    assert "would delete the other" in message


def test_the_script_never_runs_a_destructive_git_command():
    """The repository-wide promise: tooling does not discard anyone's work.

    ``git worktree add`` would have been the obvious way to get an isolated
    source tree, and it is not used, partly for this reason and partly because
    the repository already has many worktrees registered.
    """
    destructive = (r"git\s+reset", r"git\s+clean", r"git\s+checkout", r"git\s+stash", r"git\s+restore")
    source = re.sub(r"(?m)#.*$", "", BUILD.read_text(encoding="utf-8"))
    for pattern in destructive:
        assert not re.search(pattern, source), f"docs/build.py contains `{pattern}`"


def test_publishing_never_forces():
    """The push is a plain fast-forward, which is what makes a race detectable.

    ``git add --force`` is used and is fine: it stages the composed tree past
    docs/.gitignore. What must never appear is a forced *push*.
    """
    source = BUILD.read_text(encoding="utf-8")
    assert "--force-with-lease" not in source
    pushes = [line for line in source.splitlines() if '"push"' in line]
    assert pushes, "no push found in docs/build.py"
    for line in pushes:
        assert "--force" not in line, f"forced push: {line.strip()}"


# --------------------------------------------------------------------------
# publishing, against a real repository
# --------------------------------------------------------------------------

pytest_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A work repository and a bare origin, isolated from the developer's git."""
    for name, value in (
        ("GIT_CONFIG_GLOBAL", os.devnull),
        ("GIT_CONFIG_SYSTEM", os.devnull),
        ("GIT_AUTHOR_NAME", "Test"), ("GIT_AUTHOR_EMAIL", "test@example.invalid"),
        ("GIT_COMMITTER_NAME", "Test"), ("GIT_COMMITTER_EMAIL", "test@example.invalid"),
    ):
        monkeypatch.setenv(name, value)

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-b", "main", str(work)], check=True, capture_output=True)

    def git(*args, cwd=work, check=True):
        return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                              capture_output=True, text=True)

    (work / "seed.txt").write_text("seed\n")
    git("add", "seed.txt")
    git("commit", "-m", "seed")
    git("remote", "add", "origin", str(origin))
    git("push", "-q", "origin", "main")
    # a starting point for gh-pages, standing in for the site as published today
    git("push", "-q", "origin", "main:refs/heads/gh-pages")
    return work, origin, git


@pytest_git
def test_publish_creates_exactly_one_commit_on_the_previous_tip(build, sandbox, composed, tmp_path):
    work, origin, git = sandbox
    before = build.remote_tip(work, "origin", "gh-pages")

    commit = build.publish(
        composed, work, before, "docs: publish site (main aaaaaaa, develop bbbbbbb)\n",
        scratch=tmp_path / "scratch-publish",
    )

    log = git("log", "--format=%H %P", "gh-pages", cwd=origin).stdout.split("\n")
    assert log[0].split()[0] == commit
    assert log[0].split()[1] == before, "the publish is a plain child of the tip it read"
    assert len(git("log", "--oneline", "gh-pages", cwd=origin).stdout.strip().splitlines()) == 2

    listing = git("ls-tree", "-r", "--name-only", "gh-pages", cwd=origin).stdout.split()
    assert "index.html" in listing and "develop/index.html" in listing
    assert ".nojekyll" in listing and "provenance.yml" in listing
    assert "seed.txt" not in listing, "the published tree replaces the branch wholesale"


@pytest_git
def test_a_concurrent_publish_is_reported_and_nothing_is_overwritten(build, sandbox, composed, tmp_path):
    work, origin, git = sandbox
    stale = build.remote_tip(work, "origin", "gh-pages")

    # someone else publishes while our build is running
    (work / "other.txt").write_text("other\n")
    git("add", "other.txt")
    git("commit", "-m", "another publish")
    git("push", "-q", "origin", "HEAD:refs/heads/gh-pages")
    theirs = build.remote_tip(work, "origin", "gh-pages")
    assert theirs != stale

    with pytest.raises(build.PublishRace) as race:
        build.publish(composed, work, stale, "docs: publish\n", scratch=tmp_path / "scratch-race")
    message = str(race.value)
    assert "moved while this build was running" in message
    assert stale in message and theirs in message
    assert build.remote_tip(work, "origin", "gh-pages") == theirs, "their publish survived"


@pytest_git
def test_publishing_leaves_the_checkout_and_its_index_alone(build, sandbox, composed, tmp_path):
    work, origin, git = sandbox
    before_status = git("status", "--porcelain", "--untracked-files=all").stdout
    before_worktrees = git("worktree", "list").stdout

    # Read the index immediately either side of the publish and run nothing
    # else in between: `git status` refreshes the index's stat cache itself, so
    # asking git anything after publishing would rewrite the very file this is
    # checking publish did not touch.
    index = work / ".git" / "index"
    before_index = index.read_bytes() if index.exists() else None
    build.publish(composed, work, build.remote_tip(work, "origin", "gh-pages"),
                  "docs: publish\n", scratch=tmp_path / "scratch-clean")
    after_index = index.read_bytes() if index.exists() else None
    assert after_index == before_index, "the repository index was written to"

    assert git("status", "--porcelain", "--untracked-files=all").stdout == before_status
    assert git("worktree", "list").stdout == before_worktrees


@pytest_git
def test_the_commit_message_names_both_source_commits(build, tracks):
    message = build._commit_message(tracks)
    assert message.startswith("docs: publish site (main aaaaaaa, develop bbbbbbb)")
    assert "a" * 40 in message and "b" * 40 in message
    assert "output only" in message
