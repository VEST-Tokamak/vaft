"""Contract for Tutorial 02: VEST startup and vacuum-field analysis (issue #888).

The tutorial runs offline from the packaged VEST samples, while teaching the
same public workflow on any database shot.  The tests pin the narrative spine
(response -> camera -> one breakdown time -> actuators -> Green function and
vessel solve -> model check -> proxies -> midplane -> maps -> camera field
lines -> multi-shot comparison) and the data-honesty rules: one data-derived
breakdown time and no literal times, reduced-model caveats stated, sensor
currents not called raw, and every lab-mode line a real, parseable call.
"""

import ast
import os
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

#: The two lab-mode lines, commented in the committed notebook, and the offline
#: lines each one replaces when uncommented.
DATABASE_SWAPS = {
    "s02-load-sample": (
        "ods = vaft.omas.sample_ods(SHOT)",
        "# ods = vaft.database.load(SHOT)",
    ),
    "s02-multi-load": (
        "ods_list = [vaft.omas.sample_ods(shot) for shot in shots]",
        "# ods_list = vaft.database.load(shots)",
    ),
}


def _execute(notebook):
    previous = {key: os.environ.get(key) for key in ("MPLBACKEND", "VAFT_TUTORIAL_MODE")}
    os.environ["MPLBACKEND"] = "inline"
    os.environ.pop("VAFT_TUTORIAL_MODE", None)
    try:
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        return notebook
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def book():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def executed():
    return _execute(nbformat.from_dict(nbformat.read(NOTEBOOK, as_version=4)))


def _code_cells(book):
    return [cell for cell in book.cells if cell.cell_type == "code"]


def _source(cell):
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _strip_comments(source):
    return re.sub(r"(?m)^\s*#.*$", "", source)


def _code(book):
    return "\n".join(_source(cell) for cell in _code_cells(book))


def _executable(book):
    return _strip_comments(_code(book))


def _markdown(book):
    """All markdown, whitespace collapsed so a pinned phrase may wrap across lines."""
    return " ".join(
        " ".join(_source(cell).split()) for cell in book.cells if cell.cell_type == "markdown"
    )


def _printed(executed):
    return "\n".join(
        "".join(output.get("text", ""))
        for cell in _code_cells(executed)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def _cell(book, cell_id):
    return next(cell for cell in book.cells if cell.id == cell_id)


def _first_code_index(book, token):
    """Index (in cell order) of the first code cell *executing* ``token``."""
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code" and token in _strip_comments(_source(cell)):
            return index
    raise AssertionError(f"no executable code cell uses {token!r}")


def _uncomment_database_lines(source):
    """Uncomment exactly the lab-mode ``vaft.database.load`` assignments."""
    return re.sub(
        r"(?m)^(\s*)#\s*((?:ods|ods_list) = vaft\.database\.load\()",
        r"\1\2",
        source,
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
    assert len(drawn) >= 32, [cell.id for cell in drawn]


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


def test_the_default_path_needs_no_credentials(book):
    executable = _executable(book)
    assert "ods = vaft.omas.sample_ods(SHOT)" in executable
    assert "vaft.omas.sample_ods(shot) for shot in shots" in executable
    assert "vaft.database" not in executable


def test_database_loading_is_taught_where_the_data_is_loaded(book):
    for cell_id, (offline, lab) in DATABASE_SWAPS.items():
        source = _source(_cell(book, cell_id))
        assert offline in source, cell_id
        assert lab in source, cell_id


def test_uncommented_database_lines_are_valid_python(book):
    for cell_id, (offline, lab) in DATABASE_SWAPS.items():
        source = _source(_cell(book, cell_id))
        uncommented = _uncomment_database_lines(source)
        assert lab.lstrip("# ") in _strip_comments(uncommented), cell_id
        ast.parse(uncommented)
    # Nothing else in the notebook is uncommented by the swap.
    touched = [
        cell.id
        for cell in _code_cells(book)
        if _uncomment_database_lines(_source(cell)) != _source(cell)
    ]
    assert sorted(touched) == sorted(DATABASE_SWAPS), touched


@pytest.mark.integration
def test_the_notebook_runs_on_database_shots(book):
    """Opt-in: the same notebook with the lab-mode lines swapped in.

    Needs HSDS credentials (``~/.hscfg``) and ``VAFT_DATABASE_TESTS=1``.
    """
    if not (Path.home() / ".hscfg").is_file():
        pytest.skip("no ~/.hscfg: HSDS credentials are not configured")
    if not os.environ.get("VAFT_DATABASE_TESTS"):
        pytest.skip("set VAFT_DATABASE_TESTS=1 to execute the tutorial against the database")
    lab = nbformat.from_dict(nbformat.read(NOTEBOOK, as_version=4))
    for cell_id, (offline, lab_line) in DATABASE_SWAPS.items():
        cell = _cell(lab, cell_id)
        source = _source(cell)
        assert offline in source, cell_id
        # The offline line becomes the lab-mode call; the commented original stays a comment.
        cell.source = source.replace(offline, lab_line.lstrip("# "), 1)
    assert "vaft.database.load(SHOT)" in _executable(lab)
    assert "vaft.database.load(shots)" in _executable(lab)
    _execute(lab)


def test_the_committed_notebook_stores_no_outputs(book):
    for cell in _code_cells(book):
        assert cell.execution_count is None, cell.id
        assert cell.get("outputs", []) == [], cell.id


# ---------------------------------------------------------------------------
# Stays at the public API level, and works for any shot
# ---------------------------------------------------------------------------


def test_no_notebook_local_helper_functions(book):
    for cell in _code_cells(book):
        tree = ast.parse(_source(cell))
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)), cell.id


def test_no_tutorial_specific_machinery(book):
    banned = ("BLANK", "require(", "check_values", "exercise_support",
              "find_repository_root", "savefig", "getattr(vaft")
    executable = _executable(book)
    for token in banned:
        assert token not in executable, token


def test_no_literal_discharge_times(book):
    """Times come from the data: no float literal where VEST discharge times live."""
    for cell in _code_cells(book):
        for node in ast.walk(ast.parse(_source(cell))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                assert not 0.2 <= node.value <= 0.4, (cell.id, node.value)
    assert "0.3307" not in _code(book)


def test_the_shot_number_is_written_once_in_part_one(book):
    executable = _executable(book)
    assert executable.count("SHOT = 39915") == 1
    part_one = [cell for cell in _code_cells(book) if not cell.id.startswith("s02-multi")]
    for cell in part_one:
        if cell.id == "s02-load-sample":
            continue
        assert "39915" not in _strip_comments(_source(cell)), cell.id


def test_breakdown_time_is_detected_once_and_reused(book):
    executable = _executable(book)
    assert executable.count("find_breakdown_onset") == 1
    assert "t_breakdown = vaft.omas.find_breakdown_onset(ods)" in executable
    onset = _first_code_index(book, "find_breakdown_onset")
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code" and "t_breakdown" in _strip_comments(_source(cell)):
            assert index >= onset, cell.id


# ---------------------------------------------------------------------------
# The session's order
# ---------------------------------------------------------------------------


ORDER = (
    "plot_plasma_current_time",
    "plot_b_field_probe_time_field",
    "plot_camera_visible_image(",
    "find_breakdown_onset",
    "plt.subplots(6, 1",
    "plot_tf_coil_time_current",
    "plot_pf_coil_time_current",
    "plot_barometry_time_pressure",
    "plot_ec_launchers_time_power",
    "plot_machine_geometry_poloidal",
    "green_psi_exact",
    "compute_eddy_currents",
    "plot_magnetics_overview_vacuum",
    "plot_magnetics_overview_plasma_residual",
    "plot_rogowski_coil_time_current",
    "plot_startup_proxies_time",
    "compute_decay_index_ods",
    "plot_vacuum_field_midplane",
    "plot_equilibrium_field_psi_vacuum",
    "plot_vacuum_field(",
    "compute_connection_length_map_ods",
    "plot_camera_visible_image_vacuum_field_line",
    "vaft.omas.sample_ods(shot) for shot in shots",
)


def test_the_analysis_runs_in_the_session_order(book):
    positions = [_first_code_index(book, token) for token in ORDER]
    assert positions == sorted(positions), dict(zip(ORDER, positions))


def test_geometry_and_green_function_are_taught_before_the_eddy_solve(book):
    sources = [_source(cell) for cell in book.cells]
    solve = _first_code_index(book, "compute_eddy_currents")
    green = min(i for i, s in enumerate(sources) if "Green" in s and "mutual" in s.lower())
    assert green < solve
    markdown = _markdown(book)
    for token in ("green_psi_exact", "greens_function_exact", "green_br_bz_exact",
                  "mutual_inductance", "self_inductance"):
        assert token in markdown, token
    assert r"\mathbf{M}_{va}" in markdown and r"\mathbf{M}_{vp}" in markdown
    assert r"\mathbf{L}_{vv}" in markdown


def test_the_vessel_currents_are_solved_before_vacuum_quantities(book):
    solve = _first_code_index(book, "compute_eddy_currents")
    for token in (
        "compute_startup_loop_voltage_ods",
        "compute_decay_index_ods",
        "plot_magnetics_overview_vacuum",
        "plot_equilibrium_field_psi_vacuum",
        "plot_vacuum_field_midplane",
    ):
        assert _first_code_index(book, token) > solve, token


# ---------------------------------------------------------------------------
# Every new API is used
# ---------------------------------------------------------------------------


def test_the_response_section_uses_every_signal_family(book):
    executable = _executable(book)
    assert 'emission="H_alpha"' in executable
    assert 'emission="all"' in executable
    assert 'plot_flux_loop_time_voltage(ods, selection="inboard_mid")' in executable
    assert 'plot_b_field_probe_time_field(ods, selection="inboard_mid")' in executable
    assert "plot_diamagnetic_flux_time" in executable


def test_the_startup_apis_are_all_used(book):
    executable = _executable(book)
    for api in (
        "plot_ec_launchers_time_power",
        "plot_rogowski_coil_time_current",
        "plot_startup_proxies_time",
        "plot_vacuum_field_midplane",
        "ec_frequency_Hz=",
        "plot_camera_visible_image_vacuum_field_line",
        "startup_summary",
        "compute_prefill_pressure_ods(ods, before=t_breakdown)",
        "electron_cyclotron_resonance_radius",
        "vertical_field_from_I_p_R0_a_beta_p_li",
        "mutual_inductance",
        "self_inductance",
        "change_time_convention",
        "plot_pf_coil_geometry_poloidal",
        "plot_current_overview",
    ):
        assert api in executable, api


def test_the_commented_alternatives_are_taught(book):
    code = _code(book)
    assert "# vaft.omas.plot_tf_coil_time_b_t(ods)" in code
    assert "# vaft.omas.plot_pf_coil_time_current_turns(ods)" in code
    assert 'yunit="Pa"' in code
    assert "PA_PER_TORR" in _executable(book)


def test_multi_shot_comparison_overlays_lists(book):
    executable = _executable(book)
    assert "shots = [39915, 41524, 41672]" in executable
    assert "plot_startup_proxies_time(ods_list" in executable
    assert "plot_plasma_current_time(ods_list" in executable
    assert "copy.deepcopy" in executable


def test_the_interactive_cells_run_headless(book):
    """`interaction_backend="auto"` resolves to ipywidgets under nbclient, which
    leaves a widget that is dead on GitHub -- so every interactive cell pins it."""
    interactive = 0
    for cell in _code_cells(book):
        source = _source(cell)
        if "interactive=True" in source:
            interactive += 1
            assert 'interaction_backend="none"' in source, cell.id
    assert interactive >= 3


# ---------------------------------------------------------------------------
# What the notebook prints
# ---------------------------------------------------------------------------


def test_the_solve_is_shown_to_unlock_the_analysis(executed):
    printed = _printed(executed)
    assert "passive current available? False" in printed
    assert "magnetics_overview_vacuum" in printed


def test_the_onset_criterion_is_shown_and_not_only_its_answer(executed):
    printed = _printed(executed)
    assert "breakdown onset" in printed
    assert "decided by" in printed
    assert "light and current" in printed


def test_the_derived_quantities_are_actually_reported(executed):
    printed = _printed(executed)
    assert "peak |V_loop|" in printed
    assert "V_loop at breakdown" in printed
    assert "mutual_inductance, filaments" in printed
    assert "(ratio 1.000000)" in printed
    # The rigid-ring band is left at breakdown and regained by the current peak.
    assert re.search(r"breakdown onset .*entirely inside 0 < n < 1.5: False", printed)
    assert re.search(r"I_p peak .*entirely inside 0 < n < 1.5: True", printed)
    assert "R_ECR" in printed
    assert "prefill (median of the 20 ms before breakdown)" in printed


def test_the_connection_length_is_traced_rather_than_assumed(executed):
    printed = _printed(executed)
    assert "scaling / trace" in printed
    assert "never met the wall" in printed


def test_the_multi_shot_table_is_printed(executed):
    printed = _printed(executed)
    for shot in ("39915", "41524", "41672"):
        assert shot in printed
    assert "prefill [mPa]" in printed


# ---------------------------------------------------------------------------
# Physics and data honesty in the prose
# ---------------------------------------------------------------------------


def test_the_actuator_roles_are_stated_correctly(book):
    markdown = _markdown(book)
    assert "PF1" in markdown and "central solenoid" in markdown
    assert "PF5" in markdown and "null" in markdown
    assert "PF9/10" in markdown
    assert "capacitor-bank" in markdown
    assert "feedback" in markdown and "feedforward" in markdown
    assert "tf.r0" in markdown and "1/R" in markdown


def test_the_unmapped_actuators_are_named(book):
    markdown = _markdown(book)
    assert "gas valve command" in markdown
    assert "mapping gap" in markdown
    assert "#165" in markdown


def test_ec_power_is_described_as_an_estimate(book):
    markdown = _markdown(book).lower()
    assert "2.45 ghz" in markdown
    assert "net launched-power estimate" in markdown
    assert "smooth=" in markdown
    assert "resonance" in markdown


def test_reduced_proxy_caveats_are_explicit(book):
    markdown = _markdown(book)
    lower = markdown.lower()
    assert "startup reference point" in lower
    assert "radial force-balance proxy" in lower
    assert "vertical-stability proxy" in lower
    assert "not* the ohmic power" in lower
    assert "1000 V/m" in markdown and "100 V/m" in markdown
    assert "Townsend" in markdown and "Lloyd" in markdown


def test_rogowski_explanation_distinguishes_sensor_current_from_raw_daq(book):
    markdown = _markdown(book)
    lower = markdown.lower()
    assert "magnetics.rogowski_coil" in markdown
    assert "rogowski_coil:plasma_current" in markdown
    assert "rogowski_coil:diamagnetic_tf_current" in markdown
    assert "raw daq" in lower
    assert "calibrated sensor currents" in lower
    assert "magnetics.ip" in markdown
    assert "magnetics.diamagnetic_flux" in markdown


def test_the_camera_overlay_is_vacuum_not_efit(book):
    markdown = _markdown(book)
    assert "vacuum field lines" in markdown.lower()
    assert "never an EFIT" in markdown


def test_what_230_asked_for_is_computed_or_named_rather_than_faked(book):
    markdown = _markdown(book)
    assert "Connection length" in markdown
    assert "2.45 GHz" in markdown
    assert "#230" in markdown


def test_the_exercise_is_a_one_page_report_from_the_log(book):
    markdown = _markdown(book).lower()
    assert "one-page" in markdown
    assert "experiment log" in markdown
    exercise = _cell(book, "s02-exercise-cell")
    for line in _source(exercise).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line


def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 2
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]
