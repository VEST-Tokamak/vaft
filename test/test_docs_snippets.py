"""Documentation snippets must name APIs that exist, and current ones.

``vaft.omas.sample_odc()`` was deliberately removed -- ``test_load_sample.py``
asserts it is gone -- and went on appearing in five rendered pages.
``docs/_guide/Installation.md``, the page ``docs/index.markdown`` sends a
first-time reader to, opened with ``vaft.plot.magnetics_time_ip(ods)``, which
warns *and* raises ``TypeError``: the canonical renderers take a view model, so
the fix was never the rename alone.  ``from vaft.imas import save_omas_imas``
and ``vaft.database.load_ids`` never worked at all.  None of it was caught,
because nothing in the suite had ever read a fenced code block.

Both checks here are static; no snippet is executed.  The first resolves every
``vaft.*`` name a ```` ```python ```` block mentions.  The second rejects names
the migration tables in ``vaft/plot/_migration.py`` have already renamed,
deprecated or relocated -- which resolution alone cannot catch, because a
deprecated alias resolves perfectly well through ``vaft.plot.__getattr__``, and
the plain aliases in ``vaft/plot/time.py`` do not even warn.

What this deliberately misses, so nobody assumes more coverage than exists:
names not rooted at the literal token ``vaft`` (``import vaft.plot as vp`` and
then ``vp.foo()``), chains continued across lines, dynamic ``getattr``, and
names in prose backticks -- prose legitimately names a removed API while
explaining that it was removed.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import types
import warnings
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="this branch has no docs/ directory")

FENCE = re.compile(r"^\s*```([A-Za-z0-9_+-]*)\s*$")
PYTHON_TAGS = {"python", "py"}
DOTTED = re.compile(r"(?<![\w.])vaft(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
FROM_IMPORT = re.compile(
    r"^\s*(?:>>>\s*|\.\.\.\s*)?from\s+(vaft(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+)$"
)

#: Floors, not exact counts: adding a page must not edit this file, but a
#: broken fence regex or a moved docs/ directory must not pass silently either.
#: Every other test here asserts an empty list, so vacuity is the real risk.
MINIMUM_PAGES = 45
MINIMUM_NAMES = 200

#: Snippets that name something a guard would otherwise reject, and why.
#: Each entry is (page relative to docs/, dotted name, reason). An entry is a
#: promise that the snippet is *about* the name rather than using it;
#: ``test_the_allowlist_has_no_stale_entries`` deletes the promise once the
#: page stops needing it.
ALLOWED: tuple[tuple[str, str, str], ...] = ()


def _front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---", text, re.S)
    return yaml.safe_load(match.group(1)) if match else {}


def _all_pages() -> list[Path]:
    pages = sorted(DOCS.glob("_guide/*.md")) + sorted(DOCS.glob("_pages/*.md"))
    return [p for p in pages + [DOCS / "index.markdown"] if p.is_file()]


def _rendered_pages() -> list[Path]:
    """Pages the site actually publishes.

    A ``layout: redirect`` stub never renders its body, so asserting about it
    is asserting about dead text -- and ``_guide/Plotting.md`` alone would need
    a fifty-entry allowlist on day one, which would make the allowlist bigger
    than the guard.  The skip is asserted rather than silent, so a page cannot
    duck these checks by gaining a redirect layout.
    """
    return [p for p in _all_pages() if _front_matter(p).get("layout") != "redirect"]


def _snippet_names(path: Path) -> list[tuple[int, str]]:
    """Every ``vaft.*`` name a python fence in this page mentions."""
    found: list[tuple[int, str]] = []
    tag: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fence = FENCE.match(line)
        if fence:
            tag = None if tag is not None else (fence.group(1) or "")
            continue
        if tag not in PYTHON_TAGS:
            continue
        for hit in DOTTED.finditer(line):
            found.append((lineno, hit.group(0)))
        # A chain-only regex sees `vaft.imas` here -- which resolves -- and
        # misses the imported name, which is the half that was wrong.
        imported = FROM_IMPORT.match(line.split("#", 1)[0])
        if imported:
            module, names = imported.group(1), imported.group(2)
            for raw in names.replace("(", "").replace(")", "").split(","):
                name = raw.strip().split(" as ")[0].strip()
                if name and name != "*":
                    found.append((lineno, f"{module}.{name}"))
    return found


@lru_cache(maxsize=None)
def _resolve(dotted: str) -> str | None:
    """Return why ``dotted`` does not resolve, or None if it is fine.

    The walk stops at the first attribute that is not a module or a class:
    beyond that the chain is a runtime object and nothing static can be said
    about it.  ``find_spec`` before ``import_module`` is what separates "vaft
    does not ship this module" from "this environment lacks an optional
    dependency" -- resolving the corpus imports ``imas``, ``numba``, ``h5pyd``
    and more, and without the distinction a machine missing one extra would
    report fabricated documentation defects.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        obj: object = importlib.import_module("vaft")
        walked = "vaft"
        for part in dotted.split(".")[1:]:
            if not isinstance(obj, (types.ModuleType, type)):
                return None
            try:
                obj = getattr(obj, part)
            except AttributeError:
                if not isinstance(obj, types.ModuleType):
                    return f"{walked} has no attribute {part!r}"
                candidate = f"{walked}.{part}"
                try:
                    spec = importlib.util.find_spec(candidate)
                except (ImportError, ValueError):
                    spec = None
                if spec is None:
                    return f"{walked} has no attribute or submodule {part!r}"
                try:
                    obj = importlib.import_module(candidate)
                except ImportError:
                    return None
            walked = f"{walked}.{part}"
    return None


def _legacy(dotted: str) -> tuple[str, str] | None:
    """Return (replacement, removal release) if this name has been superseded.

    Matching is namespace-qualified on purpose.  ``RELOCATED`` keys are also
    valid modern names elsewhere -- ``is_signal_active`` is wrong at
    ``vaft.plot.`` and right at ``vaft.process.`` -- so a last-segment match
    would flag correct code.  ``LEGACY`` is not matched at all: those names are
    still supported and have no replacement to suggest.
    """
    from vaft.plot._migration import (
        DEPRECATED,
        RELOCATED,
        REMOVAL_RELEASE,
        RENAMED,
        RENAMED_REMOVAL_RELEASE,
    )

    head, _, leaf = dotted.rpartition(".")
    if head == "vaft.plot" or head.startswith("vaft.plot."):
        if leaf in RENAMED:
            return f"vaft.omas.plot_{RENAMED[leaf]}", RENAMED_REMOVAL_RELEASE
        if leaf in DEPRECATED:
            return f"vaft.omas.plot_{DEPRECATED[leaf]}", REMOVAL_RELEASE
        if leaf in RELOCATED:
            return RELOCATED[leaf], REMOVAL_RELEASE
    elif head in ("vaft.omas", "vaft.omas.plotting") and leaf.startswith("plot_"):
        # Only RENAMED grew `plot_<old>` adapters; a `plot_<deprecated>`
        # spelling does not exist and is caught by the resolution guard.
        stem = leaf[len("plot_"):]
        if stem in RENAMED:
            return f"vaft.omas.plot_{RENAMED[stem]}", RENAMED_REMOVAL_RELEASE
    return None


@pytest.fixture(scope="module")
def names() -> list[tuple[Path, int, str]]:
    """(page, line, dotted name) for every rendered page, scanned once."""
    return [
        (page, lineno, dotted)
        for page in _rendered_pages()
        for lineno, dotted in _snippet_names(page)
    ]


def _allowed(page: Path, dotted: str) -> bool:
    relative = page.relative_to(DOCS).as_posix()
    return any(entry[0] == relative and entry[1] == dotted for entry in ALLOWED)


def _report(problems: list[str]) -> str:
    return "\n" + "\n".join(problems) + (
        "\n\nIf a snippet names an old API on purpose, add it to ALLOWED in "
        "test/test_docs_snippets.py with a one-line reason."
    )


# --- the corpus itself --------------------------------------------------------


def test_only_redirect_stubs_are_left_unscanned():
    """Keep the scope honest: a page cannot duck the guards by redirecting."""
    scanned = set(_rendered_pages())
    skipped = set(_all_pages()) - scanned
    assert all(_front_matter(p).get("layout") == "redirect" for p in skipped), (
        "a page is being skipped that is not a redirect stub: "
        f"{sorted(p.name for p in skipped if _front_matter(p).get('layout') != 'redirect')}"
    )
    assert len(scanned) >= MINIMUM_PAGES, (
        f"only {len(scanned)} pages scanned; docs/ may have moved"
    )


def test_the_extractor_still_sees_snippet_code(tmp_path, names):
    """The other tests assert empty lists, so they pass vacuously if this breaks."""
    sample = tmp_path / "sample.md"
    sample.write_text(
        "prose naming `vaft.omas.sample_odc` in backticks\n"
        "```python\n"
        "ods = vaft.omas.sample_ods()\n"
        "from vaft.formula import greenwald_density   # flat namespace\n"
        ">>> vaft.plot.plasma_current_time\n"
        "```\n"
        "```bash\n"
        "vaft.plot.not_python\n"
        "```\n",
        encoding="utf-8",
    )
    found = {name for _, name in _snippet_names(sample)}
    assert "vaft.omas.sample_ods" in found
    assert "vaft.formula.greenwald_density" in found, "from-imports must be extracted"
    assert "vaft.plot.plasma_current_time" in found, ">>> sessions must be extracted"
    assert "vaft.plot.not_python" not in found, "only python fences are read"
    assert "vaft.omas.sample_odc" not in found, "prose is not read"
    assert len(names) >= MINIMUM_NAMES, (
        f"only {len(names)} names extracted from the real corpus; the fence or "
        "name regex is probably broken"
    )


# --- the two guards -----------------------------------------------------------


def test_every_documented_vaft_attribute_resolves(names):
    """A snippet that names something absent cannot run for anyone (#775)."""
    problems = [
        f"{page.relative_to(ROOT)}:{lineno}: {dotted} does not resolve\n    ({why})"
        for page, lineno, dotted in names
        if (why := _resolve(dotted)) and not _allowed(page, dotted)
    ]
    assert not problems, _report(problems)


def test_no_snippet_teaches_a_superseded_plot_name(names):
    """Resolution cannot see this: a deprecated alias resolves fine (#774)."""
    problems = []
    for page, lineno, dotted in names:
        superseded = _legacy(dotted)
        if not superseded or _allowed(page, dotted):
            continue
        replacement, removal = superseded
        problems.append(
            f"{page.relative_to(ROOT)}:{lineno}: {dotted} was superseded by "
            f"{replacement} and is removed in {removal}.\n"
            "    Note the namespace: vaft.plot.* renderers take a view model, so from "
            "an ODS\n    the entry point is the vaft.omas.plot_* adapter, not the "
            "renamed renderer."
        )
    assert not problems, _report(problems)


def test_the_allowlist_has_no_stale_entries(names):
    """An allowlist that only ever grows stops meaning anything."""
    live = {
        (page.relative_to(DOCS).as_posix(), dotted)
        for page, _, dotted in names
        if _resolve(dotted) or _legacy(dotted)
    }
    stale = [entry for entry in ALLOWED if (entry[0], entry[1]) not in live]
    assert not stale, (
        "these ALLOWED entries no longer match anything a guard rejects; delete them:\n"
        + "\n".join(f"  {page}: {name}  ({why})" for page, name, why in stale)
    )
