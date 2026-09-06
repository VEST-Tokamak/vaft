"""Session 03 executes offline and teaches at the public API level.

Modelled on ``test_tutorial_session_01.py``. Session 03 is an analysis session,
so it keeps the eight-heading structure sessions 02-06 share rather than
session 01's bespoke one, and its code cells are allowed to do more than one
call each -- but they must still be VAFT calls, not reimplementations.
"""

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorial"
NOTEBOOK = TUTORIAL / "03_equilibrium_and_kinetic_profiles.ipynb"

MAX_OUTPUT_BYTES = 200_000
MAX_IMAGE_BYTES = 2_000_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")


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
            timeout=900,
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


def _code_cells(book):
    return [cell for cell in book.cells if cell.cell_type == "code"]


def _source(cell):
    return cell.source if isinstance(cell.source, str) else "".join(cell.source)


def _executable(book):
    """Every code line that is not commented out."""
    joined = "\n".join(_source(cell) for cell in _code_cells(book))
    return re.sub(r"(?m)^\s*#.*$", "", joined)


# ---------------------------------------------------------------------------
# Runs offline
# ---------------------------------------------------------------------------

def test_the_notebook_runs_top_to_bottom(executed):
    assert _code_cells(executed)


def test_every_analysis_step_produces_a_figure(executed):
    drawn = [
        cell
        for cell in _code_cells(executed)
        if any(output.get("output_type") == "display_data" for output in cell.get("outputs", []))
    ]
    assert len(drawn) >= 10, [cell.id for cell in drawn]


def test_no_cell_dumps_a_data_object(executed):
    for cell in _code_cells(executed):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime, payload in data.items():
                if mime in IMAGE_MIME_TYPES:
                    assert len(payload) < MAX_IMAGE_BYTES, (cell.id, mime)
                else:
                    text = payload if isinstance(payload, str) else "".join(payload)
                    assert len(text) < MAX_OUTPUT_BYTES, (cell.id, mime)
            text = output.get("text", "")
            assert len("".join(text)) < MAX_OUTPUT_BYTES, cell.id


def test_the_notebook_needs_no_credentials_and_no_external_code(book):
    """Offline mode must not reach the database, and must not run CHEASE.

    The CHEASE workflow is taught, but only in the commented exercise, so a
    reader without it installed still executes every cell.
    """
    executable = _executable(book)
    assert "vaft.omas.sample_ods()" in executable
    assert "vaft.database" not in executable
    assert "scan_chease" not in executable
    assert "run_chease" not in executable
    assert "refine_equilibrium" not in executable


def test_the_committed_notebook_stores_no_outputs(book):
    for cell in _code_cells(book):
        assert cell.execution_count is None, cell.id
        assert cell.get("outputs", []) == [], cell.id


# ---------------------------------------------------------------------------
# Stays at the public API level
# ---------------------------------------------------------------------------

def test_no_notebook_local_helper_functions(book):
    for cell in _code_cells(book):
        source = _source(cell)
        assert not re.search(r"(?m)^\s*def\s", source), cell.id
        assert not re.search(r"(?m)^\s*class\s", source), cell.id


def test_no_tutorial_specific_machinery(book):
    banned = ("BLANK", "require(", "check_values", "exercise_support",
              "find_repository_root", "savefig", "getattr(vaft")
    executable = _executable(book)
    for token in banned:
        assert token not in executable, token


def test_the_derivation_step_is_taught_rather_than_assumed(book, executed):
    """The session's most transferable habit: a refused plot is often underived.

    Both halves have to be present -- the before, where the plots are refused,
    and the call that makes them available.
    """
    executable = _executable(book)
    assert "update_equilibrium_derived_profiles" in executable

    printed = "\n".join(
        "".join(output.get("text", ""))
        for cell in _code_cells(executed)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "beta_n available?  False" in printed
    assert "beta_n available?  True" in printed


def test_the_residual_is_measured_on_both_equilibria(executed):
    """The integrated analysis rests on these two numbers being real."""
    printed = "\n".join(
        "".join(output.get("text", ""))
        for cell in _code_cells(executed)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "EFIT   median relative residual" in printed
    assert "CHEASE median relative residual" in printed


def test_each_supported_coordinate_is_actually_demonstrated(book):
    """The session claims a vocabulary; it has to show it."""
    executable = _executable(book)
    for coordinate in ("psi_norm", "rho_tor_norm", "r_major"):
        assert f'coordinate="{coordinate}"' in executable, coordinate


def test_the_exercise_executes_nothing_on_its_own(book):
    exercise = next(cell for cell in _code_cells(book) if cell.id == "s03-exercise-cell")
    for line in _source(exercise).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 3
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]


def test_the_unsupported_coordinate_is_named_rather_than_quietly_skipped(book):
    """Straight-field-line coordinates are what a reader will ask for next.

    Session 03 does not teach them because VAFT does not compute them, and
    saying so is better than leaving the reader to discover the gap.
    """
    markdown = "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )
    assert "traight-field-line" in markdown
    assert "#472" in markdown
