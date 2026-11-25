"""The tutorial clean filter stores notebooks the way verify_tutorial requires.

``tutorial/_strip_outputs.py`` is registered as a git clean filter for
``tutorial/**/*.ipynb`` (#2319). It must give back the committed bytes for an
executed notebook, so a run leaves no diff, and it must never reach the
notebooks under ``notebooks/``, which keep their outputs on purpose.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tutorial" / "_strip_outputs.py"
TUTORIALS = sorted((ROOT / "tutorial").glob("[0-9][0-9]_*.ipynb"))

_spec = importlib.util.spec_from_file_location("_strip_outputs", SCRIPT)
strip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strip)


def _executed(text: str) -> str:
    """What Jupyter saves after running every code cell of a clean notebook."""
    notebook = json.loads(text)
    for count, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            cell["execution_count"] = count
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": ["ψ_N = 0.5\n"]}]
            cell["metadata"]["execution"] = {"iopub.status.idle": "2026-11-19T00:00:00Z"}
            cell["metadata"]["scrolled"] = True
    notebook["metadata"]["language_info"] = {
        "codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
        "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3", "version": "3.14.0",
    }
    # nbformat writes sorted keys, whatever order the file had.
    return json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.name[:2])
def test_a_committed_tutorial_is_a_fixed_point(path):
    text = path.read_text(encoding="utf-8")
    assert strip.strip_text(text) == text


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.name[:2])
def test_an_executed_tutorial_strips_back_to_the_committed_bytes(path):
    text = path.read_text(encoding="utf-8")
    executed = _executed(text)
    if any(c["cell_type"] == "code" for c in json.loads(text)["cells"]):
        assert executed != text
    assert strip.strip_text(executed) == text


def test_stripping_keeps_sources_and_notebook_metadata():
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "a", "metadata": {"tags": ["keep"]}, "source": ["# t"]},
            {"cell_type": "code", "id": "b", "metadata": {"tags": ["x"], "execution": {}}, "execution_count": 3,
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1"]}], "source": ["print(1)"]},
        ],
        "metadata": {"vaft_tutorial": {"session": 1}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = json.loads(strip.strip_text(json.dumps(notebook)))
    assert out["cells"][1] == {"cell_type": "code", "id": "b", "metadata": {"tags": ["x"]},
                               "execution_count": None, "outputs": [], "source": ["print(1)"]}
    assert out["cells"][0] == notebook["cells"][0]
    assert out["metadata"] == notebook["metadata"]


@pytest.mark.parametrize("text", ["", "not json", "<<<<<<< HEAD\n{}\n", '{"no": "cells"}', '{"cells": 3}'])
def test_anything_that_is_not_a_notebook_passes_through(text):
    assert strip.strip_text(text) == text


def test_the_filter_covers_tutorial_notebooks_only():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    notebook = next((ROOT / "notebooks").glob("*.ipynb"))
    result = subprocess.run(
        ["git", "check-attr", "filter", "--", TUTORIALS[0].relative_to(ROOT).as_posix(),
         notebook.relative_to(ROOT).as_posix()],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    tutorial_line, notebook_line = result.stdout.splitlines()
    assert tutorial_line.endswith(f"filter: {strip.FILTER_NAME}")
    assert notebook_line.endswith("filter: unspecified")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_installed_filter_keeps_outputs_out_of_the_index(tmp_path):
    def git(*args):
        # The notebooks are UTF-8; Windows would decode git's output with the locale code page.
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, encoding="utf-8",
                              check=True).stdout

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (tmp_path / "tutorial").mkdir()
    shutil.copy(SCRIPT, tmp_path / "tutorial" / SCRIPT.name)
    (tmp_path / ".gitattributes").write_text(f"tutorial/**/*.ipynb filter={strip.FILTER_NAME}\n")
    subprocess.run([sys.executable, "tutorial/_strip_outputs.py", "--install"], cwd=tmp_path, check=True,
                   capture_output=True)

    clean = TUTORIALS[0].read_text(encoding="utf-8")
    target = tmp_path / "tutorial" / TUTORIALS[0].name
    target.write_text(clean, encoding="utf-8", newline="\n")
    git("add", ".")
    git("commit", "-q", "-m", "clean")

    target.write_text(_executed(clean), encoding="utf-8", newline="\n")
    # git status flags the size change without re-running the filter; the content diff is empty.
    assert git("diff", "--stat") == ""
    git("add", f"tutorial/{target.name}")
    assert git("status", "--porcelain") == ""
    assert git("show", f":tutorial/{target.name}") == clean
    assert "ψ_N = 0.5" in target.read_text(encoding="utf-8")  # the working copy keeps its outputs

    subprocess.run([sys.executable, "tutorial/_strip_outputs.py", "--uninstall"], cwd=tmp_path, check=True,
                   capture_output=True)
    assert strip.FILTER_NAME not in git("config", "--local", "--list")


def test_the_filter_hands_back_non_utf8_input_untouched():
    data = b"\xff\xfe{not utf-8"
    result = subprocess.run([sys.executable, str(SCRIPT)], input=data, capture_output=True, check=True)
    assert result.stdout == data
