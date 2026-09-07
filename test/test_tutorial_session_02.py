"""Session 02 executes offline and teaches at the public API level.

Modelled on ``test_tutorial_session_03.py``. The session's spine is that a
*modelling* step -- solving the vessel currents -- is what makes the
vacuum-field analysis possible, so the tests pin that sequence rather than only
the individual calls.
"""

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorial" / "02_operation_scenario_and_vacuum_fields.ipynb"

MAX_OUTPUT_BYTES = 200_000
MAX_IMAGE_BYTES = 2_000_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")


@pytest.fixture(scope="module")
def book():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def executed():
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
    joined = "\n".join(_source(cell) for cell in _code_cells(book))
    return re.sub(r"(?m)^\s*#.*$", "", joined)


def _printed(executed):
    return "\n".join(
        "".join(output.get("text", ""))
        for cell in _code_cells(executed)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


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
            for mime, payload in output.get("data", {}).items():
                if mime in IMAGE_MIME_TYPES:
                    assert len(payload) < MAX_IMAGE_BYTES, (cell.id, mime)
                else:
                    text = payload if isinstance(payload, str) else "".join(payload)
                    assert len(text) < MAX_OUTPUT_BYTES, (cell.id, mime)
            assert len("".join(output.get("text", ""))) < MAX_OUTPUT_BYTES, cell.id


def test_the_notebook_needs_no_credentials(book):
    executable = _executable(book)
    assert "vaft.omas.sample_ods()" in executable
    assert "vaft.database" not in executable


def test_the_committed_notebook_stores_no_outputs(book):
    for cell in _code_cells(book):
        assert cell.execution_count is None, cell.id
        assert cell.get("outputs", []) == [], cell.id


# ---------------------------------------------------------------------------
# The session's spine
# ---------------------------------------------------------------------------

def test_the_vessel_currents_are_solved_before_anything_is_derived(book):
    """Every vacuum-field number rests on that solve, so it has to come first."""
    sources = [_source(cell) for cell in _code_cells(book)]
    solve = next(i for i, s in enumerate(sources) if "compute_eddy_currents" in s)
    for token in (
        "compute_startup_loop_voltage_ods",
        "compute_decay_index_ods",
        "plot_magnetics_overview_vacuum",
        "plot_equilibrium_field_psi_vacuum",
    ):
        uses = [i for i, s in enumerate(sources) if token in s and not s.lstrip().startswith("#")]
        assert uses, token
        assert min(uses) > solve, token


def test_the_solve_is_shown_to_unlock_the_analysis(executed):
    """The session's point: a modelling step changes what can be plotted."""
    printed = _printed(executed)
    assert "passive current available? False" in printed
    assert "magnetics_overview_vacuum" in printed


def test_the_model_is_checked_against_the_magnetics_before_it_is_used(book):
    """The validation comes before the physics, not after."""
    sources = [_source(cell) for cell in _code_cells(book)]
    check = next(i for i, s in enumerate(sources) if "plot_magnetics_overview_vacuum" in s)
    null = next(i for i, s in enumerate(sources) if "plot_equilibrium_field_psi_vacuum" in s)
    assert check < null


def test_the_derived_quantities_are_actually_reported(executed):
    printed = _printed(executed)
    assert "peak |V_loop|" in printed
    assert "breakdown onset" in printed
    assert "entirely inside 0 < n < 1.5: True" in printed


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


def test_the_spectroscopy_is_asked_for_by_species(book):
    """Session 01 teaches emission=; this session uses it for burn-through."""
    executable = _executable(book)
    assert 'emission="H_alpha"' in executable
    assert 'emission="CIII"' in executable


def test_the_exercise_executes_nothing_on_its_own(book):
    exercise = next(cell for cell in _code_cells(book) if cell.id == "s02-exercise-cell")
    for line in _source(exercise).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line


# ---------------------------------------------------------------------------
# Structure and honesty
# ---------------------------------------------------------------------------

def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 2
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]


def test_the_two_unsupported_exercises_are_named_rather_than_faked(book):
    """#230 asks for connection length and the EC resonance layer.

    Neither exists in VAFT. An unlabelled approximation in teaching material is
    worse than an honest gap, so the session says which two and why.
    """
    markdown = "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )
    assert "Connection length" in markdown
    assert "2.45 GHz" in markdown
    assert "#230" in markdown


def test_the_unmapped_actuators_are_named(book):
    """Gas and EC trigger are recorded on VEST but not mapped into IMAS."""
    markdown = "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )
    assert "gas valve command" in markdown
    assert "mapping gap" in markdown
