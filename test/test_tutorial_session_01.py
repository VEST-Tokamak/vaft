"""Execution and scaffolding contract for the Session 01 tutorial worksheet.

Session 01 is a guided worksheet, not a demonstration (issue #224). Two
properties matter and pull in opposite directions:

* a fresh student copy must **not** complete itself under *Run All*; and
* everything that is not an exercise must still run offline, so a student who
  is stuck on exercise 4 is stuck on the physics, not on a broken notebook.

The tests below pin both, plus the scaffolding conventions that later sessions
are expected to copy. The fully answered notebook lives in the separate private
solution repository and is not a runtime dependency of this suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorial"
NOTEBOOK = TUTORIAL / "01_getting_started_with_vaft.ipynb"
SUPPORT = TUTORIAL / "exercise_support.py"

REQUIRE_CALL = re.compile(r"\brequire\(\s*(\d+)")
CHECK_CALL = re.compile(r"\b(?:check_values|confirm)\(\s*(\d+)")


def _source_text(cell) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


@pytest.fixture(scope="module")
def book():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def exercise_cells(book):
    return [
        cell
        for cell in book.cells
        if cell.cell_type == "code" and REQUIRE_CALL.search(_source_text(cell))
    ]


# ---------------------------------------------------------------------------
# Scaffolding structure
# ---------------------------------------------------------------------------


def test_exercise_support_module_ships_with_the_tutorial():
    assert SUPPORT.is_file(), "tutorial/exercise_support.py is the scaffolding contract"


def test_the_worksheet_has_exercises(exercise_cells):
    assert len(exercise_cells) >= 5, "a worksheet needs more than a token exercise"


def test_every_exercise_cell_offers_a_blank_to_fill(exercise_cells):
    """70-80% scaffolded: instructions plus a skeleton, never an empty cell."""
    for cell in exercise_cells:
        source = _source_text(cell)
        assert "BLANK" in source, f"{cell.id}: an exercise must leave a BLANK to fill"
        assert "TODO" in source, f"{cell.id}: mark the blanks with a TODO comment"
        assert len(source.splitlines()) > 4, f"{cell.id}: too little scaffolding"


def test_exercises_are_numbered_contiguously_from_one(exercise_cells):
    numbers = [int(REQUIRE_CALL.search(_source_text(cell)).group(1)) for cell in exercise_cells]
    assert numbers == sorted(numbers), "exercise numbers must increase down the notebook"
    assert numbers == list(range(1, len(numbers) + 1)), f"non-contiguous numbering: {numbers}"


def test_each_exercise_validates_the_answer_it_asked_for(exercise_cells):
    """A wrong answer should be caught here, not ten cells later."""
    for cell in exercise_cells:
        source = _source_text(cell)
        required = int(REQUIRE_CALL.search(source).group(1))
        checked = {int(match) for match in CHECK_CALL.findall(source)}
        assert checked == {required}, (
            f"{cell.id}: exercise {required} must confirm its own answer"
        )


def test_every_exercise_is_introduced_by_instructions(book, exercise_cells):
    """Markdown states the task; the code cell below it is the skeleton."""
    cells = list(book.cells)
    exercise_ids = {cell.id for cell in exercise_cells}
    for index, cell in enumerate(cells):
        if cell.id not in exercise_ids:
            continue
        preceding = [item for item in cells[:index] if item.cell_type == "markdown"]
        assert preceding, f"{cell.id}: no Markdown instructions precede this exercise"
        heading = _source_text(preceding[-1])
        assert "Exercise" in heading, (
            f"{cell.id}: the Markdown cell above must introduce the exercise"
        )


def test_the_notebook_points_at_the_bootstrap_instead_of_repeating_them(book):
    """Issue #225 owns installation; the tutorial must not fork a second copy."""
    markdown = "\n".join(
        _source_text(cell) for cell in book.cells if cell.cell_type == "markdown"
    )
    assert "install/README.md" in markdown
    assert "check_vaft_environment.py" in markdown
    assert "conda env create" not in markdown, "installation belongs in install/README.md"


# ---------------------------------------------------------------------------
# Execution behaviour
# ---------------------------------------------------------------------------


def _client(book, path: Path) -> NotebookClient:
    return NotebookClient(
        book, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )


def test_a_fresh_worksheet_cannot_be_solved_by_run_all(monkeypatch, tmp_path):
    """The defining property of a worksheet: Run All stops at the first blank."""
    monkeypatch.setenv("VAFT_TUTORIAL_MODE", "offline")
    monkeypatch.setenv("VAFT_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    executed = nbformat.from_dict(nbformat.read(NOTEBOOK, as_version=4))
    with pytest.raises(CellExecutionError) as failure:
        _client(executed, tmp_path).execute()

    assert failure.value.ename == "ExerciseIncomplete"
    assert "Exercise 1 is not complete" in str(failure.value)
    assert "fill in the TODO fields" in str(failure.value)


def test_everything_before_the_first_exercise_runs_offline(monkeypatch, tmp_path):
    """Setup, sample loading, and the inventory must work without any answers."""
    monkeypatch.setenv("VAFT_TUTORIAL_MODE", "offline")
    monkeypatch.setenv("VAFT_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    source_book = nbformat.read(NOTEBOOK, as_version=4)
    first_exercise = next(
        index
        for index, cell in enumerate(source_book.cells)
        if cell.cell_type == "code" and REQUIRE_CALL.search(_source_text(cell))
    )
    prefix = nbformat.from_dict(source_book)
    prefix.cells = prefix.cells[:first_exercise]

    _client(prefix, tmp_path).execute()

    output_text = "\n".join(
        output.get("text", "")
        for cell in prefix.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert "Mode: offline; source: VAFT packaged sample; shot: 39915" in output_text
    assert "VEST diagnostic registry" in output_text
    assert "VAFT can draw" in output_text


def test_the_committed_notebook_stores_no_outputs():
    """Committed notebooks are source, not result archives."""
    committed = nbformat.read(NOTEBOOK, as_version=4)
    for cell in committed.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


def test_session_01_default_output_is_stable_from_tutorial_directory(monkeypatch):
    """Launching Jupyter from tutorial/ still uses the ignored output tree."""
    book = nbformat.read(NOTEBOOK, as_version=4)
    setup_cell = next(cell for cell in book.cells if cell.get("id") == "session01-setup")

    monkeypatch.chdir(TUTORIAL)
    monkeypatch.delenv("VAFT_TUTORIAL_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("VAFT_TUTORIAL_MODE", "offline")

    namespace = {}
    exec(compile(setup_cell.source, f"{NOTEBOOK.name}:session01-setup", "exec"), namespace)

    assert namespace["OUTPUT_DIR"] == TUTORIAL / "outputs" / "01"


def test_lab_branch_is_skipped_offline(capsys):
    """Offline is the default, and it must never reach for the network."""
    book = nbformat.read(NOTEBOOK, as_version=4)
    lab_cell = next(cell for cell in book.cells if cell.get("id") == "session01-lab-read")

    def forbidden(*args, **kwargs):
        raise AssertionError("the offline lesson contacted HSDS")

    namespace = {
        "MODE": "offline",
        "SHOT": 39915,
        "np": np,
        "vaft": SimpleNamespace(database=SimpleNamespace(open=forbidden)),
    }
    exec(compile(lab_cell.source, f"{NOTEBOOK.name}:session01-lab-read", "exec"), namespace)

    assert namespace["lab_time_s"] is None
    assert namespace["lab_ip_a"] is None
    assert "Lab extension skipped in offline mode" in capsys.readouterr().out


def test_session_01_lab_branch_propagates_unexpected_api_errors():
    """The lab gate handles access failures without hiding programming defects."""
    book = nbformat.read(NOTEBOOK, as_version=4)
    lab_cell = next(cell for cell in book.cells if cell.get("id") == "session01-lab-read")
    plot_cell = next(
        cell for cell in book.cells if cell.get("id") == "session01-current-plot"
    )

    def broken_open(*args, **kwargs):
        raise RuntimeError("simulated API regression")

    namespace = {
        "MODE": "lab",
        "SHOT": 39915,
        "np": np,
        "vaft": SimpleNamespace(database=SimpleNamespace(open=broken_open)),
    }
    with pytest.raises(RuntimeError, match="simulated API regression"):
        exec(compile(lab_cell.source, f"{NOTEBOOK.name}:session01-lab-read", "exec"), namespace)

    assert "lab_time_s" in plot_cell.source
    assert "lab_ip_a" in plot_cell.source


# ---------------------------------------------------------------------------
# exercise_support
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def support():
    sys.path.insert(0, str(TUTORIAL))
    try:
        import exercise_support
    finally:
        sys.path.remove(str(TUTORIAL))
    return exercise_support


def test_blank_repr_stays_readable(support):
    """Error messages and debugger sessions still need to show the placeholder."""
    assert repr(support.BLANK) == "BLANK"


@pytest.mark.parametrize(
    "operation",
    [
        lambda blank: {"a": 1}[blank],
        lambda blank: list(blank),
        lambda blank: str(blank),
        lambda blank: blank + 1,
        lambda blank: bool(blank),
        lambda blank: blank.anything,
        lambda blank: np.asarray(blank),
        lambda blank: blank == 1,
    ],
)
def test_using_a_blank_raises_immediately(support, operation):
    """This is what stops Run All at the exercise instead of far downstream."""
    with pytest.raises(support.ExerciseIncomplete):
        operation(support.BLANK)


def test_require_names_the_exercise_and_the_missing_answers(support):
    with pytest.raises(support.ExerciseIncomplete) as failure:
        support.require(4, channels=support.BLANK, probes=[0, 1])
    message = str(failure.value)
    assert "Exercise 4 is not complete" in message
    assert "channels" in message
    assert "probes" not in message


def test_require_passes_once_every_answer_is_filled(support):
    support.require(4, channels=[0, 1, 2], probes=[0, 1])


def test_require_looks_inside_containers(support):
    """`channels = [0, BLANK]` is still an unanswered exercise."""
    with pytest.raises(support.ExerciseIncomplete):
        support.require(4, channels=[0, support.BLANK])
    with pytest.raises(support.ExerciseIncomplete):
        support.require(4, options={"channel": support.BLANK})


def test_is_blank_uses_identity_not_equality(support):
    assert support.is_blank(support.BLANK)
    assert not support.is_blank(0)
    assert not support.is_blank("")


def test_check_values_reports_the_first_wrong_answer(support):
    with pytest.raises(support.ExerciseIncomplete) as failure:
        support.check_values(2, unit=("V", "A"))
    message = str(failure.value)
    assert "Exercise 2" in message
    assert "unit is 'V', expected 'A'" in message


def test_check_values_confirms_a_correct_answer(support, capsys):
    support.check_values(2, unit=("A", "A"))
    assert "Exercise 2 complete." in capsys.readouterr().out
