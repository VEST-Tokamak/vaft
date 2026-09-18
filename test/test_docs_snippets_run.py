"""Documentation snippets are executed, not only name-checked.

``test_docs_snippets.py`` resolves the ``vaft.*`` names a fence mentions and
binds call arguments; it runs nothing.  0.7.0 shipped with samples that pass
both and still raise for every reader (cold review docs F3, D1-D4): a sample
that reads ``beta_normal`` from a packaged shot that has none fails on the
*data*, which no static check can see.

This module runs every ```` ```python ```` fence of the rendered guide pages and
the two READMEs, offline:

* one child interpreter for the whole corpus (``vaft`` is imported once), with
  the Matplotlib ``Agg`` backend, every socket connect refused, and a scratch
  working directory, so a snippet can neither reach HSDS/MySQL nor write into
  the checkout;
* the fences of one page share a namespace, top to bottom, the way a reader
  follows the page -- a later fence may use what an earlier one defined;
* that namespace starts with the names the guide assumes everywhere:
  ``vaft``, ``np``, ``plt``, ``ods = vaft.omas.sample_ods()`` and
  ``shot = 39915`` (``PRELUDE`` below).  Nothing else is provided.

A fence that cannot run like that says so, with a reason, in an HTML comment on
the line directly above it (invisible on the site, on GitHub and on PyPI)::

    <!-- docs-snippet: skip needs-database (loads shot 39915 from HSDS) -->
    ```python
    ods = vaft.database.load(39915, source="public")
    ```

The classes are ``SKIP_CLASSES``.  The marker is the reviewable record: a new
fence without one is executed, so a sample that needs a database has to say so
and a sample that claims to run offline has to.  The reason text is free-form
but mandatory.

Cost: two to three minutes for ~95 executed fences (measured on the
development laptop, 115 s outside pytest and 180 s under the shared resource
queue; the wall-reduction eigen-decomposition alone is 20 s).  The
module is nevertheless *not* marked ``slow``: ``slow`` moves a test to the
``main`` gate only, and a sample broken by a library change should surface on
the push to ``develop`` that broke it, not at release qualification.  It is
deliberately not in the ``core`` develop gate (``test/core_selection.py``),
whose documentation group is "file reads and getattr only".
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="this branch has no docs/ directory")

OPEN_FENCE = re.compile(r"^(\s*)(`{3,})\s*(python|py)\s*$")
MARKER = re.compile(r"^\s*<!--\s*docs-snippet:\s*(.*?)\s*-->\s*$")
SKIP_MARKER = re.compile(r"^skip\s+([a-z-]+)\s*\((.+)\)$")

#: Why a fence is not executed.  Each is a property of the fence a reviewer can
#: check by reading it, not a statement that it failed once.
SKIP_CLASSES = {
    "needs-database": "reads or writes a VEST database source (HSDS, a lab folder, MySQL)",
    "needs-raw-source": "reads raw DAQ signals or a machine_mapping mapper that does",
    "needs-external-code": "runs an external physics code or a pipeline stage that does",
    "needs-file": "opens a user-supplied file or directory the repository does not ship",
    "needs-data": "needs an IDS the packaged sample shot does not carry",
    "fragment": "uses placeholder names the page never defines; shows a call shape",
    "signature": "a signature listing or pseudo-code, not a Python program",
}

#: Floors, as in test_docs_snippets.py: a broken fence regex must not pass as
#: "nothing to run".
MINIMUM_FENCES = 200
MINIMUM_EXECUTED = 60

PRELUDE = """\
import numpy as np
import matplotlib.pyplot as plt
import vaft
ods = vaft.omas.sample_ods()
shot = 39915
"""


def _front_matter(path: Path) -> dict:
    match = re.match(r"\A---\s*\n(.*?)\n---", path.read_text(encoding="utf-8"), re.S)
    return yaml.safe_load(match.group(1)) if match else {}


def _pages() -> list[Path]:
    """Rendered guide/site pages plus the READMEs (the PyPI long description)."""
    pages = sorted(DOCS.glob("_guide/*.md")) + sorted(DOCS.glob("_pages/*.md"))
    pages = [p for p in pages + [DOCS / "index.markdown"] if p.is_file()]
    pages = [p for p in pages if _front_matter(p).get("layout") != "redirect"]
    return pages + [p for p in (ROOT / "README.md", ROOT / "README.ko.md") if p.is_file()]


def extract_fences(path: Path, root: Path = ROOT) -> tuple[list[dict], list[str]]:
    """Return (fences, problems) for one page.

    A fence is ``{"page", "line", "source", "skip", "reason"}``; ``line`` is the
    1-based line of the opening fence.  ``problems`` lists malformed or orphaned
    markers, which would otherwise silently fail to skip -- or to run.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    relative = path.relative_to(root).as_posix()
    orphan = "{}:{}: marker is not directly above a python fence"
    fences: list[dict] = []
    problems: list[str] = []
    pending: tuple[int, str] | None = None
    i = 0
    while i < len(lines):
        marker = MARKER.match(lines[i])
        opened = None if marker else OPEN_FENCE.match(lines[i])
        if marker or not opened:
            if pending:
                problems.append(orphan.format(relative, pending[0]))
            pending = (i + 1, marker.group(1)) if marker else None
            i += 1
            continue
        indent, ticks = opened.group(1), opened.group(2)
        close = re.compile(rf"^\s*{ticks}`*\s*$")
        start = i
        i += 1
        body: list[str] = []
        while i < len(lines) and not close.match(lines[i]):
            body.append(lines[i])
            i += 1
        i += 1
        source = textwrap.dedent("\n".join(
            line[len(indent):] if line.startswith(indent) else line for line in body
        )) + "\n"
        skip = reason = None
        if pending:
            parsed = SKIP_MARKER.match(pending[1])
            if not parsed or parsed.group(1) not in SKIP_CLASSES:
                problems.append(
                    f"{relative}:{pending[0]}: unreadable marker {pending[1]!r}; expected "
                    f"'skip <class> (<reason>)' with a class from {sorted(SKIP_CLASSES)}"
                )
            else:
                skip, reason = parsed.group(1), parsed.group(2)
            pending = None
        fences.append(
            {"page": relative, "line": start + 1, "source": source, "skip": skip, "reason": reason}
        )
    if pending:
        problems.append(orphan.format(relative, pending[0]))
    return fences, problems


@lru_cache(maxsize=None)
def _corpus() -> tuple[tuple[dict, ...], tuple[str, ...]]:
    fences: list[dict] = []
    problems: list[str] = []
    for page in _pages():
        page_fences, page_problems = extract_fences(page)
        fences += page_fences
        problems += page_problems
    return tuple(fences), tuple(problems)


def _executed() -> list[dict]:
    return [f for f in _corpus()[0] if f["skip"] is None]


# --- the child interpreter ----------------------------------------------------

RUNNER = r'''
import json, os, socket, sys, traceback, warnings

def _refuse(*args, **kwargs):
    raise OSError("docs-snippet runner: network access is disabled")

socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse
socket.create_connection = _refuse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(sys.argv[1], encoding="utf-8") as handle:
    job = json.load(handle)
import vaft
expected = os.path.realpath(job["root"])
assert os.path.realpath(vaft.__file__).startswith(expected + os.sep), (
    f"runner imported {vaft.__file__}, not the checkout under test {expected}"
)
warnings.simplefilter("ignore")

results = []
namespaces = {}
for fence in job["fences"]:
    label = f"{fence['page']}:{fence['line']}"
    namespace = namespaces.get(fence["page"])
    if namespace is None:
        namespace = namespaces[fence["page"]] = {"__name__": "__docs_snippet__"}
        exec(job["prelude"], namespace)
    try:
        exec(compile(fence["source"], label, "exec"), namespace)
        outcome = None
    except BaseException as error:  # SystemExit in a snippet is a failure too
        frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename == label]
        where = fence["line"] + frames[-1].lineno if frames else fence["line"]
        outcome = {"line": where, "error": f"{type(error).__name__}: {error}"[:600]}
    plt.close("all")
    results.append({"page": fence["page"], "line": fence["line"], "failure": outcome})
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(results, handle)
'''


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory) -> dict:
    """Run every unmarked fence once; map (page, fence line) -> failure or None."""
    work = tmp_path_factory.mktemp("docs_snippets")
    cwd = work / "cwd"
    cwd.mkdir()
    job, out, runner = work / "job.json", work / "out.json", work / "runner.py"
    job.write_text(
        json.dumps({"root": str(ROOT), "prelude": PRELUDE, "fences": _executed()}),
        encoding="utf-8",
    )
    runner.write_text(RUNNER, encoding="utf-8")
    # PYTHONPATH pins the checkout under test even when an editable install of
    # another checkout is present; the runner asserts that it won.
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT), MPLCONFIGDIR=str(work / "mpl"))
    done = subprocess.run(
        [sys.executable, str(runner), str(job), str(out)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=900,
    )
    results = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else []
    table = {(r["page"], r["line"]): r["failure"] for r in results}
    if done.returncode != 0:
        # The interpreter died (or never started): every fence it did not reach
        # fails with the reason, rather than the module erroring opaquely.
        tail = (done.stderr or done.stdout).strip()[-1500:]
        for fence in _executed():
            table.setdefault(
                (fence["page"], fence["line"]),
                {"line": fence["line"], "error": f"runner exited {done.returncode}: {tail}"},
            )
    return table


# --- the corpus and its markers -----------------------------------------------


def test_the_extractor_reads_fences_and_markers(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "prose\n"
        "```python\nx = 1\n```\n"
        "<!-- docs-snippet: skip needs-database (loads a shot) -->\n"
        "```python\nvaft.database.load(1)\n```\n"
        "1. a list item\n\n"
        "   ```python\n   y = 2\n   if y:\n       z = 3\n   ```\n"
        "```bash\nnot python\n```\n"
        "<!-- docs-snippet: skip because (unknown class) -->\n"
        "```python\npass\n```\n"
        "<!-- docs-snippet: skip fragment (orphan) -->\n"
        "prose again\n",
        encoding="utf-8",
    )
    fences, problems = extract_fences(page, root=tmp_path)
    assert [f["source"] for f in fences[:3]] == [
        "x = 1\n", "vaft.database.load(1)\n", "y = 2\nif y:\n    z = 3\n",
    ]
    assert [f["skip"] for f in fences] == [None, "needs-database", None, None]
    assert fences[1]["reason"] == "loads a shot"
    assert len(problems) == 2, problems
    assert "unreadable marker" in problems[0] and "not directly above" in problems[1]


def test_every_marker_is_well_formed_and_the_corpus_is_not_empty():
    fences, problems = _corpus()
    assert not problems, "\n" + "\n".join(problems)
    assert len(fences) >= MINIMUM_FENCES, f"only {len(fences)} python fences found"
    assert len(_executed()) >= MINIMUM_EXECUTED, (
        f"only {len(_executed())} fences are executed; the rest are marked skip. "
        "A marker is for a fence that cannot run offline, not for one that is inconvenient."
    )


def test_signature_markers_mean_what_they_say():
    """``signature`` is for text that is not Python; a fence that parses is a fragment or a sample."""
    wrong = []
    for fence in _corpus()[0]:
        if fence["skip"] != "signature":
            continue
        try:
            ast.parse(fence["source"])
        except SyntaxError:
            continue
        wrong.append(f"{fence['page']}:{fence['line']}: marked 'signature' but it is valid Python")
    assert not wrong, "\n" + "\n".join(wrong)


# --- execution ----------------------------------------------------------------


@pytest.mark.parametrize("fence", _executed(), ids=lambda f: f"{f['page']}:{f['line']}")
def test_the_snippet_runs_offline_on_the_packaged_sample(fence, outcomes):
    key = (fence["page"], fence["line"])
    assert key in outcomes, f"{key} was never reached by the runner"
    failure = outcomes[key]
    assert failure is None, (
        f"\n{fence['page']}:{failure['line']}: {failure['error']}\n\n"
        "The fence is executed offline after the page's earlier fences and the prelude "
        "(vaft, np, plt, ods = vaft.omas.sample_ods(), shot).\nFix the sample, or -- if it "
        "cannot run like that -- put a marker on the line above it:\n"
        "    <!-- docs-snippet: skip <class> (<reason>) -->\n"
        f"classes: {', '.join(sorted(SKIP_CLASSES))}"
    )
