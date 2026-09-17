"""Documentation snippets must name APIs that exist, and current ones.

``vaft.omas.sample_odc()`` was deliberately removed -- ``test_load_sample.py``
asserts it is gone -- and went on appearing in five rendered pages.
``docs/_guide/Installation.md``, the page ``docs/index.markdown`` sends a
first-time reader to, opened with ``vaft.plot.magnetics_time_ip(ods)``, which
warns *and* raises ``TypeError``: the canonical renderers take a view model, so
the fix was never the rename alone.  ``from vaft.imas import save_omas_imas``
and ``vaft.database.load_ids`` never worked at all.  None of it was caught,
because nothing in the suite had ever read a fenced code block.

Every check here is static; no snippet is executed (``test_docs_snippets_run.py``
does that).  The first resolves every
``vaft.*`` name a ```` ```python ```` block mentions.  The second rejects names
the migration tables in ``vaft/plot/_migration.py`` have already renamed,
deprecated or relocated -- which resolution alone cannot catch, because a
deprecated alias resolves perfectly well through ``vaft.plot.__getattr__``, and
the plain aliases in ``vaft/plot/time.py`` do not even warn.

A third binds the arguments of every documented ``vaft.*`` call against the
callee's current signature.  0.7.0 added a required ``epsilon`` to the beta
conversions and a required ``q`` to the ballooning alpha, and four guide lines
went on showing the old arity -- inside fragments with placeholder operands,
which is exactly the kind of fence that can never be executed (cold review docs
F1).

What this deliberately misses, so nobody assumes more coverage than exists:
names not rooted at the literal token ``vaft`` (``import vaft.plot as vp`` and
then ``vp.foo()``), chains continued across lines, dynamic ``getattr``, and
names in prose backticks -- prose legitimately names a removed API while
explaining that it was removed.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import textwrap
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


_ATTRIBUTE_WALK = """
import importlib.util, json, sys, warnings
warnings.simplefilter("ignore")
import vaft
missing = []
for dotted in json.loads(sys.stdin.read()):
    obj, walked = vaft, "vaft"
    for part in dotted.split(".")[1:]:
        if not hasattr(obj, "__path__"):
            break  # past the packages: a module's or object's own attributes
        try:
            obj = getattr(obj, part)
        except AttributeError:
            try:
                spec = importlib.util.find_spec(f"{walked}.{part}")
            except (ImportError, ValueError):
                spec = None
            if spec is not None:
                missing.append([dotted, f"{walked}.{part}"])
            break
        except Exception:
            break  # an optional dependency is absent; not a documentation defect
        walked = f"{walked}.{part}"
print(json.dumps(missing))
"""


def _attribute_chains(path: Path) -> list[tuple[int, str]]:
    """Dotted ``vaft.*`` chains used as expressions, not named by an import statement."""
    found = []
    tag: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fence = FENCE.match(line)
        if fence:
            tag = None if tag is not None else (fence.group(1) or "")
            continue
        code = line.split("#", 1)[0]
        if tag not in PYTHON_TAGS or re.match(r"^\s*(?:>>>\s*)?(?:from|import)\s", code):
            continue
        found += [(lineno, hit.group(0)) for hit in DOTTED.finditer(code)]
    return found


def test_a_submodule_used_as_an_attribute_is_one_after_a_plain_import():
    """``vaft.omas.plasma_timing.plasma_timing(ods)`` resolved and did not run.

    ``_resolve`` falls back to importing a submodule it cannot reach by
    attribute, so a chain through a submodule that ``import vaft`` never loads
    passes it -- and raises ``AttributeError`` for the reader (cold review docs
    D4).  This walks the same chains by attribute access alone, in a fresh
    interpreter, because in this one some earlier test has usually imported the
    submodule already.
    """
    import json
    import subprocess
    import sys

    chains: dict[str, list[str]] = {}
    for page in _rendered_pages():
        for lineno, dotted in _attribute_chains(page):
            chains.setdefault(dotted, []).append(f"{page.relative_to(ROOT)}:{lineno}")
    done = subprocess.run(
        [sys.executable, "-c", _ATTRIBUTE_WALK],
        input=json.dumps(sorted(chains)), capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    problems = [
        f"{where}: {dotted}: {submodule} is a submodule that `import vaft` does not load; "
        f"write `from {submodule} import ...`"
        for dotted, submodule in json.loads(done.stdout.strip().splitlines()[-1])
        for where in chains[dotted]
    ]
    assert not problems, "\n" + "\n".join(problems)


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


def _python_fences(path: Path) -> list[tuple[int, str]]:
    """(line of the first body line, dedented source) for each python fence."""
    fences: list[tuple[int, str]] = []
    tag: str | None = None
    body: list[str] = []
    start = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fence = FENCE.match(line)
        if fence:
            if tag is None:
                tag, body, start = fence.group(1) or "", [], lineno + 1
            else:
                if tag in PYTHON_TAGS:
                    fences.append((start, textwrap.dedent("\n".join(body)) + "\n"))
                tag = None
            continue
        if tag is not None:
            body.append(line)
    return fences


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _callable_for(dotted: str):
    obj: object = importlib.import_module("vaft")
    walked = "vaft"
    for part in dotted.split(".")[1:]:
        walked = f"{walked}.{part}"
        try:
            obj = getattr(obj, part)
        except AttributeError:
            try:
                obj = importlib.import_module(walked)
            except ImportError:
                return None
    return obj if callable(obj) and not inspect.isclass(obj) else None


def _unbindable_calls(path: Path) -> list[tuple[int, str, str, str]]:
    """(line, dotted name, TypeError text, signature) for each call that cannot bind.

    From-imports carry over to the later fences of a page, as they do for a
    reader.  Fences that are not valid Python (signature listings) are skipped,
    and so are calls with ``*args``/``**kwargs``, whose arity is unknowable.
    """
    problems: list[tuple[int, str, str, str]] = []
    alias: dict[str, str] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start, source in _python_fences(path):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "vaft":
                    for item in node.names:
                        alias[item.asname or item.name] = f"{node.module}.{item.name}"
                elif isinstance(node, ast.Import):
                    for item in node.names:
                        if item.asname and item.name.split(".")[0] == "vaft":
                            alias[item.asname] = item.name
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _dotted(node.func)
                if not name:
                    continue
                head = name.split(".")[0]
                if head in alias:
                    name = alias[head] + name[len(head):]
                if not name.startswith("vaft."):
                    continue
                if any(isinstance(a, ast.Starred) for a in node.args) or any(
                    k.arg is None for k in node.keywords
                ):
                    continue
                target = _callable_for(name)
                if target is None:
                    continue
                try:
                    signature = inspect.signature(target)
                except (TypeError, ValueError):
                    continue
                try:
                    signature.bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})
                except TypeError as error:
                    problems.append((start + node.lineno - 1, name, str(error), str(signature)))
    return problems


def test_the_binder_sees_a_missing_argument(tmp_path):
    """The real corpus asserts an empty list, so prove the check can fail."""
    page = tmp_path / "page.md"
    page.write_text(
        "```python\n"
        "from vaft.formula import ballooning_alpha_from_p_B_R\n"
        "```\n"
        "prose\n"
        "```python\n"
        "alpha = ballooning_alpha_from_p_B_R(p, B, R)\n"
        "beta_p = vaft.formula.beta_pol_from_beta_tor(beta_tor, q_95)\n"
        "ok = vaft.formula.beta_pol_from_beta_tor(beta_tor, q_95, epsilon=eps)\n"
        "```\n",
        encoding="utf-8",
    )
    found = _unbindable_calls(page)
    assert [(line, name.rsplit(".", 1)[1]) for line, name, _, _ in found] == [
        (6, "ballooning_alpha_from_p_B_R"),
        (7, "beta_pol_from_beta_tor"),
    ], found
    assert "'q'" in found[0][2] and "'epsilon'" in found[1][2]


def test_every_documented_call_binds_to_the_current_signature():
    """A renamed or newly required argument breaks the sample for every reader."""
    pages = _rendered_pages() + [p for p in (ROOT / "README.md", ROOT / "README.ko.md") if p.is_file()]
    problems = [
        f"{page.relative_to(ROOT)}:{line}: {name}{signature}\n    {why}"
        for page in pages
        for line, name, why, signature in _unbindable_calls(page)
    ]
    assert not problems, "\n" + "\n".join(problems)


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
