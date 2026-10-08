"""Contract for Tutorial 05: MHD stability and perturbed equilibria (issue #1023).

The session runs offline from packaged data -- an EFIT g-file, the VEST 3-D coil
geometry and its vacuum field -- and runs DCON, RDCON and GPEC only in lab mode
with ``$GPECHOME``. The tests pin the narrative spine (rational surfaces ->
ideal stability -> tearing -> 3-D coils -> sector harmonics -> vacuum field ->
relative phase by linearity -> GPEC -> multi-time and multi-n) and the honesty
rules: no solver in the default path, a time read from the file rather than
typed, sector angles taken from each coil set, a vacuum-approximation result
labelled as one, and model excitations never presented as hardware scenarios.
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
NOTEBOOK = ROOT / "tutorial" / "05_mhd_stability_and_perturbed_equilibria.ipynb"

MAX_OUTPUT_BYTES = 200_000
MAX_IMAGE_BYTES = 2_000_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")

#: The offline run draws 18 figures and diagrams; the floor leaves a margin.
FIGURE_FLOOR = 16

#: Every solver the session can launch, and the flag that must gate it.
SOLVER_CALLS = ("run_gpec_suite_case",)


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


def _executable(book):
    return _strip_comments("\n".join(_source(cell) for cell in _code_cells(book)))


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
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code" and token in _strip_comments(_source(cell)):
            return index
    raise AssertionError(f"no executable code cell uses {token!r}")


# ---------------------------------------------------------------------------
# Runs offline, launches nothing
# ---------------------------------------------------------------------------


def test_the_notebook_runs_top_to_bottom(executed):
    assert _code_cells(executed)


def test_every_analysis_step_produces_a_figure(executed):
    drawn = [
        cell
        for cell in _code_cells(executed)
        if any(output.get("output_type") in ("display_data", "execute_result")
               and any(mime in output.get("data", {}) for mime in IMAGE_MIME_TYPES)
               for output in cell.get("outputs", []))
    ]
    assert len(drawn) >= FIGURE_FLOOR, [cell.id for cell in drawn]


def test_no_cell_dumps_a_data_object(executed):
    for cell in _code_cells(executed):
        for output in cell.get("outputs", []):
            for mime, payload in output.get("data", {}).items():
                text = payload if isinstance(payload, str) else "".join(payload)
                limit = MAX_IMAGE_BYTES if mime in IMAGE_MIME_TYPES else MAX_OUTPUT_BYTES
                assert len(text) < limit, (cell.id, mime)
            assert len("".join(output.get("text", ""))) < MAX_OUTPUT_BYTES, cell.id


def test_the_default_path_runs_no_solver(executed):
    """Offline is the default even on a machine that has $GPECHOME set."""
    printed = _printed(executed)
    for solver in ("dcon", "rdcon", "gpec"):
        assert re.search(rf"{solver}\s+skipped: offline mode", printed), solver
    for cell_id in ("s05-dcon-run", "s05-rdcon-run", "s05-gpec-run"):
        assert "skipped" in "".join(
            output.get("text", "") for output in _cell(executed, cell_id).outputs
        ), cell_id


def test_every_solver_launch_is_gated_by_the_mode_and_the_preflight(book):
    assert 'MODE = os.environ.get("VAFT_TUTORIAL_MODE", "offline")' in _executable(book)
    assert 'MODE == "lab" and path is not None' in _executable(book)
    launches = 0
    for cell in _code_cells(book):
        tree = ast.parse(_source(cell))
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "RUN[" in ast.unparse(node.test):
                guarded.update(id(child) for statement in node.body for child in ast.walk(statement))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in SOLVER_CALLS):
                launches += 1
                assert id(node) in guarded, f"{cell.id}: {ast.unparse(node)[:80]} runs outside `if RUN[...]`"
    assert launches >= 4


def test_solver_workdirs_are_short_temporary_directories(book):
    """The GPEC suite reads paths into fixed-length Fortran strings (#1239)."""
    for cell in _code_cells(book):
        source = _strip_comments(_source(cell))
        if "GPECCaseInputs(" in source:
            assert "tempfile.mkdtemp(prefix=" in source, cell.id


def test_the_notebook_uses_only_public_vaft_modules(book):
    tree = ast.parse(_executable(book))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vaft"):
            assert not any(part.startswith("_") for part in node.module.split(".")), node.module
        if isinstance(node, ast.FunctionDef):
            raise AssertionError(f"notebook-local function {node.name!r}: call VAFT instead")


# ---------------------------------------------------------------------------
# The narrative spine
# ---------------------------------------------------------------------------


def test_the_session_climbs_from_surfaces_to_driven_response(book):
    order = [
        _first_code_index(book, "find_rational_surfaces("),
        _first_code_index(book, 'modules=("dcon",)'),
        _first_code_index(book, "vaft.diagram.tearing_layer_matching()"),
        _first_code_index(book, 'modules=("rdcon",)'),
        _first_code_index(book, "plot_coil_3d_geometry_topview"),
        _first_code_index(book, "CoilExcitation.from_mode"),
        _first_code_index(book, "biot_savart_filaments([f.points_xyz"),
        _first_code_index(book, "np.exp(1j * delta)"),
        _first_code_index(book, 'modules=("dcon", "gpec")'),
    ]
    assert order == sorted(order), order


def test_the_diagrams_are_the_canonical_ones_so_they_render_without_tex(book):
    """With no TeX, a diagram is served from its committed asset only when it
    was built with the canonical arguments; any other call would refuse."""
    calls = re.findall(r"vaft\.diagram\.(\w+)\(([^)]*)\)", _executable(book))
    assert calls
    assert all(arguments.strip() == "" for _name, arguments in calls), calls


# ---------------------------------------------------------------------------
# Data honesty
# ---------------------------------------------------------------------------


def test_the_time_is_read_from_the_file_not_typed(book):
    executable = _executable(book)
    assert 'TIME_MS = int(GEQDSK.name.split(".")[-1])' in executable
    without_paths = re.sub(r'"[^"]*"', '""', executable)
    assert not re.search(r"\b31[6-9]\b|\b32\d\b|\b331\b", without_paths)


def test_every_sector_pattern_uses_its_own_coil_sets_angles(book):
    """UP and LOW sit at 15, 75, ... degrees; the default angles are MID's."""
    calls = [
        node for node in ast.walk(ast.parse(_executable(book)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_mode"
    ]
    assert len(calls) >= 3
    for call in calls:
        assert "sector_angles_deg" in {keyword.arg for keyword in call.keywords}, ast.unparse(call)


def test_the_relative_phase_scan_is_checked_against_a_direct_calculation(executed):
    printed = _printed(executed)
    match = re.search(r"direct Biot-Savart = ([0-9.e+-]+) of the peak", printed)
    assert match, printed[-2000:]
    assert float(match.group(1)) < 1e-3


def test_the_linear_picture_is_checked_before_it_is_used(executed):
    printed = _printed(executed)
    fractions = [float(value) for value in re.findall(r"delta B / B0 = ([0-9.]+)%", printed)]
    assert len(fractions) == 3
    assert max(fractions) < 10.0


def test_the_vacuum_approximation_is_labelled_and_its_gap_named(book):
    text = _markdown(book)
    assert "vacuum-approximation" in text or "vacuum approximation" in text
    assert "#1243" in text
    assert "Vacuum approximation: no screening, no amplification" in _executable(book)


def test_model_excitations_are_not_presented_as_hardware_scenarios(book):
    text = _markdown(book)
    assert "Model space and hardware space are not the same" in text
    assert "Hardware feasibility" in text


def test_the_two_reconstructions_are_compared_and_the_solver_gap_named(book, executed):
    text = _markdown(book)
    assert "#1246" in text
    printed = "".join(output.get("text", "") for output in _cell(executed, "s05-two-reconstructions").outputs)
    assert "in the g-file" in printed and "in the sample ODS" in printed


def test_the_actuator_alias_step_is_taught_beside_the_sensor_one(book):
    text = _markdown(book)
    assert "finite toroidal actuators" in text and "finite toroidal sensors" in text
