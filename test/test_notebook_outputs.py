"""What a committed notebook must carry, read statically.

`test_notebook_reliability.py` executes notebooks, so it is `slow` and runs on
the main gate. This module never starts a kernel: it reads the `.ipynb` JSON.
That is the whole point -- the failure it catches happens on a `develop` pull
request, and by the time the main gate runs, the page has already been merged
without its results.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _offline_notebooks() -> list[str]:
    """The set `notebooks/_verify_rendering.py` declares runnable without services.

    Read from that module rather than restated here: two lists that must agree
    are one list and a bug.
    """
    spec = importlib.util.spec_from_file_location(
        "_verify_rendering", NOTEBOOKS / "_verify_rendering.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted(module.OFFLINE_NOTEBOOKS)


OFFLINE_NOTEBOOKS = _offline_notebooks()


@pytest.mark.parametrize("name", OFFLINE_NOTEBOOKS)
def test_a_notebook_declared_offline_ships_its_figures(name):
    """Every notebook declared offline carries at least one committed figure.

    Executing a page and committing it without its output are indistinguishable
    in a diff, and a reader on GitHub sees the difference immediately: no plot.
    `plotting_sample_using_vaft_plot_module` -- 42 000 characters about how
    plots are made -- has lost all of its figures three times: at `f08a72c`,
    when `x=` was documented into it (#481), and when `method=` was (#484).
    Each time an edit was committed without a re-run.

    `notebooks/_clean_outputs.py` drops `error` outputs, so a cell that raised
    looks exactly like a cell that never ran. Stored figures are the only cheap
    evidence a page was executed at all.
    """
    notebook = nbformat.read(NOTEBOOKS / name, as_version=4)
    figures = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]

    assert figures, (
        f"{name} is declared offline and renders figures, but none are stored: "
        "execute it and commit the outputs"
    )
