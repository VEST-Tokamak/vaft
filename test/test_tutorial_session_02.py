"""Contract for Tutorial 02: VEST startup and vacuum-field analysis.

The tutorial runs offline from the packaged VEST sample, while teaching the
same public workflow that can be applied to database-loaded shots.  The tests
pin the narrative spine as well as the important physics/data-honesty rules:
one early breakdown time, explicit reduced-model assumptions, and no invented
EC power or raw diagnostics.
"""

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorial" / "02_startup_scenario_and_vacuum_fields.ipynb"

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
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _code(book):
    return "\n".join(_source(cell) for cell in _code_cells(book))


def _executable(book):
    return re.sub(r"(?m)^\s*#.*$", "", _code(book))


def _markdown(book):
    return "\n".join(
        _source(cell) for cell in book.cells if cell.cell_type == "markdown"
    )


def _printed(executed):
    return "\n".join(
        "".join(output.get("text", ""))
        for cell in _code_cells(executed)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


# ---------------------------------------------------------------------------
# Runs offline, but teaches the database path
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


def test_the_default_path_needs_no_credentials_but_database_loading_is_taught(book):
    executable = _executable(book)
    code = _code(book)
    assert "vaft.omas.sample_ods()" in executable
    assert "vaft.database.load(" in code
    assert "vaft.database.load(" not in executable


def test_the_committed_notebook_stores_no_outputs(book):
    for cell in _code_cells(book):
        assert cell.execution_count is None, cell.id
        assert cell.get("outputs", []) == [], cell.id


# ---------------------------------------------------------------------------
# Observe response first; establish one common breakdown time
# ---------------------------------------------------------------------------


def test_the_response_section_includes_the_requested_signal_families(book):
    executable = _executable(book)
    for api in (
        "plot_plasma_current_time",
        "plot_spectrometer_uv_time_intensity",
        "plot_diamagnetic_flux_time",
        "plot_flux_loop_time_voltage",
        "plot_b_field_probe_time_field",
    ):
        assert api in executable, api


def test_the_spectroscopy_is_not_reduced_to_one_impurity_line(book):
    executable = _executable(book)
    assert 'emission="H_alpha"' in executable
    # The tutorial should request the mapped impurity families/lines together,
    # rather than presenting CIII as if it were the only impurity observable.
    assert "CIII" not in executable or "emission=[" in executable
    assert "O" in executable and "C" in executable


def test_breakdown_time_is_detected_once_and_reused(book):
    code = _code(book)
    executable = _executable(book)
    assert "t_breakdown =" in executable
    assert executable.count("find_breakdown_onset") == 1
    assert "0.3307" not in code


def test_breakdown_is_established_before_the_eddy_solve(book):
    sources = [_source(cell) for cell in _code_cells(book)]
    onset = next(i for i, s in enumerate(sources) if "t_breakdown =" in s)
    solve = next(i for i, s in enumerate(sources) if "compute_eddy_currents" in s)
    assert onset < solve


# ---------------------------------------------------------------------------
# Actuators and the explicit 2.45 GHz teaching assumption
# ---------------------------------------------------------------------------


def test_startup_actuators_include_tf_pf_pressure_and_ampere_turns(book):
    executable = _executable(book)
    for api in (
        "plot_tf_coil_time_current",
        "plot_tf_coil_time_b_t",
        "plot_pf_coil_time_current",
        "plot_pf_coil_time_current_turns",
        "plot_barometry_time_pressure",
    ):
        assert api in executable, api


def test_ec_frequency_is_one_explicit_tutorial_assumption(book):
    code = _code(book)
    markdown = _markdown(book)
    assert "EC_FREQUENCY_HZ = 2.45e9" in code
    assert code.count("2.45e9") == 1
    assert "2.45 GHz" in markdown
    assert "resonance" in markdown.lower()


def test_ec_assumption_does_not_invent_ec_power(book):
    executable = _executable(book)
    assert "ec_launchers" not in executable
    assert "power_launched" not in executable
    markdown = _markdown(book)
    assert "#165" in markdown


# ---------------------------------------------------------------------------
# Geometry -> Green response -> eddy solve -> magnetics validation
# ---------------------------------------------------------------------------


def test_geometry_and_green_function_are_taught_before_the_eddy_solve(book):
    sources = [_source(cell) for cell in book.cells]
    solve = next(i for i, s in enumerate(sources) if "compute_eddy_currents" in s)
    geometry = min(
        i
        for i, s in enumerate(sources)
        if "plot_pf_coil_geometry_poloidal" in s
        or "plot_machine_geometry_poloidal" in s
    )
    green = min(i for i, s in enumerate(sources) if "Green" in s and "mutual" in s.lower())
    assert geometry < solve
    assert green < solve


def test_the_vessel_currents_are_solved_before_vacuum_quantities(book):
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


def test_the_model_is_checked_against_magnetics_before_null_physics(book):
    sources = [_source(cell) for cell in _code_cells(book)]
    check = next(i for i, s in enumerate(sources) if "plot_magnetics_overview_vacuum" in s)
    residual = next(i for i, s in enumerate(sources) if "plot_magnetics_overview_plasma_residual" in s)
    null = next(i for i, s in enumerate(sources) if "plot_equilibrium_field_psi_vacuum" in s)
    assert check < residual < null


def test_rogowski_explanation_distinguishes_sensor_current_from_raw_daq(book):
    markdown = _markdown(book).lower()
    assert "rogowski_coil" in markdown
    assert "raw daq" in markdown
    assert "calibrated" in markdown
    assert "magnetics.ip" in markdown
    assert "magnetics.diamagnetic_flux" in markdown


# ---------------------------------------------------------------------------
# Reduced startup proxies and interactive field interpretation
# ---------------------------------------------------------------------------


def test_reduced_proxy_caveats_are_explicit(book):
    markdown = _markdown(book)
    lower = markdown.lower()
    assert "radial force" in lower
    assert "proxy" in lower
    assert "decay index" in lower
    assert "Ohmic" in markdown or "ohmic" in lower
    assert "dissipated" in lower or "not identical" in lower


def test_ecr_radius_is_used_in_the_field_and_camera_story(book):
    executable = _executable(book)
    markdown = _markdown(book)
    assert "B_ECR" in executable
    assert "R_ECR" in executable
    assert "R_ECR" in code_or_markdown(book)
    assert "camera" in markdown.lower()


def code_or_markdown(book):
    return _code(book) + "\n" + _markdown(book)


def test_the_interactive_cells_run_headless(book):
    for cell in _code_cells(book):
        source = _source(cell)
        if "interactive=True" in source:
            assert 'interaction_backend="none"' in source, cell.id


# ---------------------------------------------------------------------------
# Comparative workflow and exercise
# ---------------------------------------------------------------------------


def test_multiple_shot_loading_is_taught_but_not_executed_offline(book):
    code = _code(book)
    executable = _executable(book)
    assert "vaft.database.load([" in code or "vaft.database.load(shots" in code
    assert "vaft.database.load([" not in executable
    assert "vaft.database.load(shots" not in executable


def test_the_exercise_executes_nothing_on_its_own(book):
    exercise = next(cell for cell in _code_cells(book) if cell.id == "s02-exercise-cell")
    for line in _source(exercise).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line


def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 2
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]
