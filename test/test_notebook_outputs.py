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

#: Pages whose "saved outputs" cell lists what the run wrote: the directory it
#: names must be the repository's, as the prose above the cell promises.
SAVED_OUTPUTS_LISTINGS = {
    "linear_resistive_stability_analysis_with_rdcon.ipynb": "notebooks/outputs/docs:",
}


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


@pytest.mark.parametrize("name, heading", sorted(SAVED_OUTPUTS_LISTINGS.items()))
def test_a_saved_outputs_listing_names_the_repository_directory(name, heading):
    """The cell that lists the page's durable products must say where they are.

    A run with ``$VAFT_DOCS_OUTPUT_DIR`` under a temporary directory committed
    ``<tmp>`` (the output cleaner's placeholder) as the heading while the
    markdown above promised ``notebooks/outputs/docs`` (cold review 0.8.0
    delta-absorb-19 physics F3).  The placeholder is right for a work
    directory that is released at the end of the page; it is wrong for the
    listing of what the page keeps.
    """
    notebook = nbformat.read(NOTEBOOKS / name, as_version=4)
    listings = [
        "".join(output.get("text", ""))
        for cell in notebook.cells
        if cell.cell_type == "code" and "_brief(OUTPUT_DIR" in "".join(cell.source)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream" and output.get("name") == "stdout"
    ]
    assert listings, f"{name}: the saved-outputs cell carries no stdout"
    for text in listings:
        first = text.splitlines()[0]
        assert first == heading, f"{name}: saved-outputs listing is headed {first!r}, not {heading!r}"
        assert "<tmp>" not in text
