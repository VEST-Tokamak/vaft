"""Contract for the Session 01 tutorial notebook.

Session 01 is a beginner's walkthrough of the everyday VAFT workflow:

    load an ODS -> inspect its IDS roots -> look at the geometry
                -> plot diagnostics -> look at the equilibrium

It is deliberately a plain notebook. It has no exercise framework, no hidden
validation, and no tutorial-specific helpers, so the tests below mostly guard
that it stays that way -- simple, offline, and expressed in public API.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorial"
NOTEBOOK = TUTORIAL / "01_getting_started_with_vaft.ipynb"

# One ODS repr is roughly half a megabyte of array dumps. Any output near that
# size means a cell is echoing a data object instead of a summary of one.
MAX_OUTPUT_BYTES = 200_000

# Rendered figures are measured separately, and only loosely. A machine
# cross-section legitimately carries every one of the sample's 950 passive
# loops -- dense pixels, not an echoed data object, so the text budget above
# does not describe it. Encoded size then varies with the platform's backend
# DPI, fonts and Matplotlib version: the same figure is ~345 kB locally on
# macOS and ~764 kB on Linux CI. A tight bound here would be a portability trap
# rather than a guard, so this is a sanity ceiling that catches a figure which
# has genuinely run away (megabytes), not a budget calibrated to any one
# machine.
MAX_IMAGE_BYTES = 2_000_000

#: Payloads that are rendered pixels rather than text a reader would scroll.
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")


def _source(cell) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


@pytest.fixture(scope="module")
def book():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    """Run the notebook once, offline, and share the result across tests."""
    import os

    previous = {key: os.environ.get(key) for key in ("MPLBACKEND", "VAFT_TUTORIAL_MODE")}
    os.environ["MPLBACKEND"] = "inline"
    os.environ.pop("VAFT_TUTORIAL_MODE", None)
    try:
        executed_book = nbformat.from_dict(nbformat.read(NOTEBOOK, as_version=4))
        NotebookClient(
            executed_book,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        return executed_book
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


# ---------------------------------------------------------------------------
# It runs, offline, from the packaged sample
# ---------------------------------------------------------------------------


def test_the_notebook_runs_top_to_bottom(executed):
    """A first tutorial must simply work when a student presses Run All."""
    assert _code_cells(executed)


def test_every_plotting_cell_produces_a_figure(executed):
    figures = [
        cell.id
        for cell in _code_cells(executed)
        if any(output.output_type == "display_data" for output in cell.get("outputs", []))
    ]
    assert len(figures) >= 10, f"expected a figure per plotting cell, got {figures}"


def test_no_cell_dumps_a_data_object(executed):
    """Echoing an ODS writes ~0.5 MB of array text into the notebook.

    The budget applies to *text*, which is what echoing a data object produces.
    Measuring the whole output would instead police figure resolution: the
    poloidal geometry plot draws all 950 passive loops and encodes to ~345 kB
    of PNG, which is a detailed picture rather than a dumped object.
    """
    for cell in _code_cells(executed):
        for output in cell.get("outputs", []):
            data = output.get("data") or {}
            text = str(output.get("text", "")) + "".join(
                str(value)
                for mime, value in data.items()
                if mime not in IMAGE_MIME_TYPES
            )
            assert len(text) < MAX_OUTPUT_BYTES, (
                f"{cell.id}: {output.output_type} text output is {len(text)} "
                "bytes; summarise the object instead of echoing it"
            )
            for mime in IMAGE_MIME_TYPES:
                if mime in data:
                    assert len(str(data[mime])) < MAX_IMAGE_BYTES, (
                        f"{cell.id}: {mime} payload is {len(str(data[mime]))} "
                        "bytes; shrink the figure rather than the data it shows"
                    )


def test_the_notebook_needs_no_credentials(book):
    """The whole lesson runs from the packaged sample."""
    executable = "\n".join(_source(cell) for cell in _code_cells(book))
    assert "vaft.omas.sample_ods()" in executable
    # vaft.database.load is taught, but only ever shown -- never executed.
    assert "vaft.database.load" not in re.sub(r"(?m)^\s*#.*$", "", executable)


def test_the_committed_notebook_stores_no_outputs(book):
    for cell in _code_cells(book):
        assert cell.execution_count is None
        assert cell.outputs == []


# ---------------------------------------------------------------------------
# It stays at the public API level
# ---------------------------------------------------------------------------


def test_imports_are_minimal_and_explained(book):
    """A beginner should not meet unexplained boilerplate in cell one."""
    first = _code_cells(book)[0]
    assert first.id == "session01-imports"
    imports = [line.strip() for line in _source(first).splitlines() if line.strip()]
    assert imports == ["import vaft", "import matplotlib.pyplot as plt"]

    index = book.cells.index(first)
    intro = _source(book.cells[index - 1])
    assert "## Imports" in intro
    assert "`vaft`" in intro and "matplotlib" in intro


def test_no_tutorial_specific_machinery(book):
    """No exercise framework, no environment plumbing, no output management."""
    executable = "\n".join(_source(cell) for cell in _code_cells(book))
    for banned in (
        "BLANK",
        "require(",
        "check_values",
        "exercise_support",
        "VAFT_TUTORIAL_MODE",
        "VAFT_TUTORIAL_OUTPUT_DIR",
        "find_repository_root",
        "savefig",
        "getattr(vaft",
        "load_diagnostic_registry",
        "os.environ",
    ):
        assert banned not in executable, f"Tutorial 01 must not use {banned!r}"


def test_the_helper_module_is_gone():
    assert not (TUTORIAL / "exercise_support.py").exists()


def test_no_notebook_local_helper_functions(book):
    """Tutorial code should demonstrate VAFT, not reimplement it."""
    for cell in _code_cells(book):
        source = _source(cell)
        assert not re.search(r"(?m)^\s*def\s", source), f"{cell.id} defines a function"
        assert not re.search(r"(?m)^\s*class\s", source), f"{cell.id} defines a class"


#: The cell that teaches the display policy deliberately calls the same plot
#: three ways to show unit, override and refusal; it is not a diagnostic cell.
_CONCEPT_CELLS = {"session01-axes-units"}

#: The Data Dictionary version the tutorial documents, and the one VAFT reads
#: through OMAS. Pinned so the compatibility note cannot quietly go stale.
DOCUMENTED_DD_VERSION = "3.41.0"


def test_plotting_cells_use_the_one_documented_pattern(book):
    """vaft.omas.plot_<something>(ods) then plt.show(), and nothing else."""
    pattern = re.compile(
        r"^vaft\.omas\.plot_[a-z0-9_]+\(ods(?:, channels=\[[0-9, ]+\])?\)\nplt\.show\(\)$"
    )
    plotting = [
        cell
        for cell in _code_cells(book)
        if "vaft.omas.plot_" in _source(cell)
        and cell.id not in _CONCEPT_CELLS
        and not _source(cell).lstrip().startswith("#")
    ]
    assert len(plotting) >= 10
    for cell in plotting:
        assert pattern.match(_source(cell).strip()), (
            f"{cell.id} deviates from the documented pattern:\n{_source(cell)}"
        )


def test_the_notebook_teaches_the_current_naming_grammar(book):
    """The names are subject-centred since #251; the prose must say so."""
    markdown = "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )
    assert "{subject}_{view}" in markdown
    # The claim this replaced -- that a plot name names its IDS -- is now wrong.
    assert "name says which IDS" not in markdown
    assert "plasma_current_time` reads the `magnetics` IDS" in markdown


def test_the_notebook_explains_the_display_policy(book):
    """A reader meets kA on the first plot; #256 is why."""
    markdown = "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )
    assert "Unit and scaling always move together" in markdown
    assert "yunit=" in markdown

    units = next(cell for cell in _code_cells(book) if cell.id == "session01-axes-units")
    source = _source(units)
    assert 'yunit="A"' in source
    assert "except ValueError" in source, "the refusal is the point, not a footnote"


def test_the_unit_override_really_rescales(executed):
    """Pins the promise the prose makes, against the rendered figure."""
    cell = next(c for c in _code_cells(executed) if c.id == "session01-axes-units")
    text = "\n".join(
        output.get("text", "")
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert "default: Plasma Current [kA]" in text
    assert "yunit='A': Plasma Current [A]" in text
    assert "refused: unsupported display unit" in text


def test_no_raw_machine_mapping_in_the_first_tutorial(book):
    """Data ingestion belongs in a later, dedicated tutorial."""
    executable = "\n".join(_source(cell) for cell in _code_cells(book))
    assert "machine_mapping" not in executable


def test_one_dataset_throughout(book):
    """Session 01 stays on the packaged sample rather than hopping between shots."""
    executable = "\n".join(_source(cell) for cell in _code_cells(book))
    uncommented = re.sub(r"(?m)^\s*#.*$", "", executable)
    assert "sample_ods" in uncommented
    assert "data_path" not in uncommented


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_each_diagnostic_is_one_markdown_cell_then_one_code_cell(book):
    """The diagnostics section should be repetitive in a good way."""
    cells = list(book.cells)
    start = next(
        index
        for index, cell in enumerate(cells)
        if cell.cell_type == "markdown" and "## Diagnostics" in _source(cell)
    )
    end = next(
        index
        for index, cell in enumerate(cells)
        if cell.cell_type == "markdown" and "## Equilibrium" in _source(cell)
    )
    kinds = [cell.cell_type for cell in cells[start:end]]
    assert kinds == ["markdown", "code"] * (len(kinds) // 2), (
        f"diagnostics must alternate markdown/code, got {kinds}"
    )


def test_the_exercise_is_an_ordinary_one(book):
    """No hidden validation: the exercise is a comment block a student edits."""
    exercise = next(cell for cell in book.cells if cell.get("id") == "session01-exercise-cell")
    body = [line for line in _source(exercise).splitlines() if line.strip()]
    assert body, "the exercise cell should offer a starting point"
    assert all(line.lstrip().startswith("#") for line in body), (
        "the exercise cell must not execute anything on its own"
    )


def test_the_summary_recaps_the_workflow(book):
    summary = next(cell for cell in book.cells if cell.get("id") == "session01-summary")
    text = _source(summary)
    for step in ("sample_ods", "keys()", "plot_machine_geometry_poloidal", "plot_"):
        assert step in text


# ---------------------------------------------------------------------------
# Additional Resources appendix
# ---------------------------------------------------------------------------


def test_external_links_are_kept_out_of_the_teaching_flow(book):
    """Reference material belongs in the appendix, not in the introduction."""
    cells = list(book.cells)
    appendix = next(
        index
        for index, cell in enumerate(cells)
        if cell.cell_type == "markdown" and "## Additional Resources" in _source(cell)
    )
    before = "\n".join(_source(cell) for cell in cells[:appendix])
    for host in ("imas-python.readthedocs.io", "gafusion.github.io",
                 "imas-data-dictionary.readthedocs.io", "imas-matlab.readthedocs.io",
                 "projecttorreypines.github.io", "iterorganization/IMAS-tutorial"):
        assert host not in before, (
            f"{host} appears before the appendix; it would weigh down the lesson"
        )


def test_the_appendix_covers_each_documented_group(book):
    resources = next(cell for cell in book.cells if cell.get("id") == "session01-resources")
    text = _source(resources)
    for heading in ("### Tutorials and hands-on examples",
                    "### API documentation by language",
                    "### IMAS Data Dictionary and schema references"):
        assert heading in text
    for language in ("**Python**", "**MATLAB**", "**Julia**"):
        assert language in text


def test_the_compatibility_note_names_the_version_vaft_actually_uses(book):
    """The note is a factual claim about this package, so pin it to the package."""
    import vaft

    resources = next(cell for cell in book.cells if cell.get("id") == "session01-resources")
    text = _source(resources)
    assert DOCUMENTED_DD_VERSION in text
    assert "4.1.1" in text, "the latest DD is worth pointing at, clearly labelled"
    assert str(vaft.omas.sample_ods().imas_version) == DOCUMENTED_DD_VERSION, (
        "VAFT no longer reads the Data Dictionary version the tutorial documents"
    )


def test_the_appendix_shows_the_version_rather_than_only_asserting_it(executed):
    cell = next(c for c in _code_cells(executed) if c.id == "session01-dd-version")
    rendered = "\n".join(
        str(output.get("data", {}).get("text/plain", ""))
        for output in cell.get("outputs", [])
    )
    assert DOCUMENTED_DD_VERSION in rendered
