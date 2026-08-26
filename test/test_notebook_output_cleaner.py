"""Tests for the version-control notebook output policy."""

from __future__ import annotations

import copy
from pathlib import Path

import nbformat

from notebooks._clean_outputs import clean_notebook, clean_path


FIXTURE_PATH = Path(__file__).parent / "data" / "notebook_output_policy.ipynb"


def _read_notebook(path: Path):
    with path.open(encoding="utf-8") as source:
        return nbformat.read(source, as_version=nbformat.NO_CONVERT)


def test_clean_notebook_preserves_only_static_outputs_and_sources():
    notebook = _read_notebook(FIXTURE_PATH)
    original_sources = [copy.deepcopy(cell.source) for cell in notebook.cells]

    assert clean_notebook(notebook)

    assert [cell.source for cell in notebook.cells] == original_sources
    assert notebook.metadata == {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    code_cell = notebook.cells[1]
    assert code_cell.metadata == {"tags": ["keep-for-rendering"]}
    assert code_cell.execution_count is None
    assert [output.output_type for output in code_cell.outputs] == [
        "display_data",
        "execute_result",
        "stream",
        "stream",
    ]

    static_data = code_cell.outputs[0].data
    assert static_data == {
        "text/plain": "static text",
        "text/markdown": "**formatted text**",
        "image/png": "cG5n",
        "image/jpeg": "anBlZw==",
        "image/svg+xml": "<svg></svg>",
    }
    assert code_cell.outputs[0].metadata == {}
    assert code_cell.outputs[1].data == {"text/plain": "mixed result"}
    assert code_cell.outputs[1].execution_count is None
    assert code_cell.outputs[1].metadata == {}
    assert code_cell.outputs[2].name == "stdout"
    assert code_cell.outputs[2].text == "ordinary stream\n"
    assert code_cell.outputs[3].name == "stderr"
    assert code_cell.outputs[3].text == "ordinary error stream\n"


def test_cleaner_rewrites_once_and_is_idempotent(tmp_path):
    path = tmp_path / "fixture.ipynb"
    path.write_bytes(FIXTURE_PATH.read_bytes())

    assert clean_path(path)
    once = path.read_bytes()
    assert not clean_path(path)
    assert path.read_bytes() == once
