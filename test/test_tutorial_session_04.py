"""Session 04 executes offline and keeps its own argument honest.

Modelled on ``test_tutorial_session_03.py``. The session's spine is that what a
fluctuation measurement *licenses* is set by the array that recorded it, not by
the analysis applied to it -- so the tests pin the two measurements that make
that case, on a shot whose array cannot resolve a toroidal mode and on one that
can, rather than only the calls that produce them.
"""

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorial"
NOTEBOOK = TUTORIAL / "04_fluctuations_and_transient_events.ipynb"

MAX_OUTPUT_BYTES = 200_000
MAX_IMAGE_BYTES = 2_000_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")


@pytest.fixture(scope="module")
def book():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def executed():
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


def _markdown(book):
    return "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )


def _printed(book):
    return "\n".join(
        "".join(output.get("text", ""))
        for cell in _code_cells(book)
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
            data = output.get("data", {})
            for mime, payload in data.items():
                if mime in IMAGE_MIME_TYPES:
                    assert len(payload) < MAX_IMAGE_BYTES, (cell.id, mime)
                else:
                    text = payload if isinstance(payload, str) else "".join(payload)
                    assert len(text) < MAX_OUTPUT_BYTES, (cell.id, mime)
            text = output.get("text", "")
            assert len("".join(text)) < MAX_OUTPUT_BYTES, cell.id


def test_the_notebook_needs_no_credentials(book):
    executable = _executable(book)
    assert "vaft.omas.sample_ods()" in executable
    assert "vaft.database" not in executable
    # method="cwt" needs the optional fcwt package, so it is taught only in the
    # commented exercise; a reader without it still executes every cell.
    assert '"cwt"' not in executable


def test_the_committed_notebook_stores_no_outputs(book):
    for cell in _code_cells(book):
        assert cell.get("execution_count") is None, cell.id
        assert cell.get("outputs", []) == [], cell.id


# ---------------------------------------------------------------------------
# Stays at the public API level
# ---------------------------------------------------------------------------

def test_no_notebook_local_helper_functions(book):
    source = "\n".join(_source(cell) for cell in _code_cells(book))
    assert not re.search(r"(?m)^\s*def\s", source)
    assert not re.search(r"(?m)^\s*class\s", source)


def test_no_tutorial_specific_machinery(book):
    source = "\n".join(_source(cell) for cell in _code_cells(book))
    for token in ("BLANK", "require(", "check_values", "exercise_support",
                  "find_repository_root", "savefig", "getattr(vaft"):
        assert token not in source, token


def test_the_exercise_executes_nothing_on_its_own(book):
    cell = next(cell for cell in book.cells if cell.get("id") == "s04-exercise-cell")
    for line in _source(cell).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line


def test_channels_are_chosen_by_name_not_by_position(book):
    """Indices depend on which archive fields a shot happened to carry, so a
    literal index would silently select a different probe on another record."""
    executable = _executable(book)
    assert "probe_names.index(" in executable
    assert "fluctuation_names.index(" in executable


# ---------------------------------------------------------------------------
# The spine: what the array allows, not what the algorithm returns
# ---------------------------------------------------------------------------

def test_the_array_sets_the_resolution_and_the_session_says_so(executed):
    """Both shots report the angles that recorded and the modulus they imply,
    before either reports a mode number."""
    printed = _printed(executed)
    assert "toroidal angles that recorded: [0.0, 240.0]" in printed
    assert "n is resolved modulo 3" in printed
    # The identifiers carry the VEST clock angles 45/135/225; their IMAS
    # toroidal angles are the reflection of those (issue #718).
    assert "toroidal angles that recorded: [135.0, 225.0, 315.0]" in printed
    assert "n is resolved modulo 4" in printed


def test_the_packaged_shot_s_second_position_is_the_first_one_again(executed):
    """The hinge. The two toroidal entries on 39915 are one physical channel, so
    the fit across them compares a signal with itself -- which is why the session
    needs a second shot at all."""
    printed = _printed(executed)
    assert "the two toroidal entries carry identical samples: True" in printed
    assert ":phase_reference" in printed


def test_the_two_shots_resolve_their_window_from_different_sources(executed):
    """Same helper, different evidence, and it names which -- the packaged shot
    from the light, the fluctuation shot from the current because only magnetics
    was mapped."""
    printed = _printed(executed)
    assert "window source : h_alpha_primary" in printed
    assert "window source : ip_principal" in printed


def test_the_transient_is_measured_against_a_floor(executed):
    printed = _printed(executed)
    assert "band power before the coils fired" in printed
    assert "fluctuation power rises above the pre-discharge floor: True" in printed


def test_the_absent_rational_surfaces_are_reported_as_absent(executed):
    """q on axis is above two here, so the 1/1 and 2/1 readings are unavailable
    as a fact about this equilibrium. The session prints the absence."""
    printed = _printed(executed)
    assert "q = 1 surface in this equilibrium: False" in printed
    assert "q = 2 surface in this equilibrium: False" in printed
    assert "q = 3 surface in this equilibrium: True" in printed


def test_the_spectral_index_is_reported_as_a_derivative_index(executed):
    """A Mirnov voltage is dB/dt, so an index fitted to it is the field's plus
    two. The session prints both columns rather than quoting one number."""
    printed = _printed(executed)
    assert "alpha(dB/dt)" in printed and "alpha(B)" in printed
    assert "R^2" in printed


def test_the_session_chose_the_voltage_for_a_stated_reason(executed):
    """Every channel's integrated field is flagged invalid on this shot, so the
    choice of voltage is evidence-led rather than a preference."""
    assert "field.validity values present: [-2]" in _printed(executed)


# ---------------------------------------------------------------------------
# Structure and honesty
# ---------------------------------------------------------------------------

def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 4
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]


def test_the_named_gaps_are_stated_with_their_issues(book):
    """An unlabelled approximation in teaching material is worse than an honest
    gap, so the session names which chains are broken and where they are tracked."""
    markdown = _markdown(book)
    assert "#506" in markdown          # no (m, n) -> surface resolver
    assert "#460" in markdown          # no rotation, so no mode tracks
    assert "#724" in markdown          # the duplicate-probe mapping defect
    assert "classifier" in markdown    # no IRE / sawtooth / ELM / disruption


def test_the_deeper_notebook_is_handed_off_rather_than_duplicated(book):
    """The single-channel spectral theory already exists in depth elsewhere; the
    session teaches the reading and points at it by name."""
    assert "notebooks/fluctuation_diagnostics_analysis.ipynb" in _markdown(book)


def test_the_checkout_only_record_is_declared_as_such(book):
    """45531 is not in the wheel. The session says so where it loads it."""
    markdown = _markdown(book)
    assert "source checkout" in markdown
