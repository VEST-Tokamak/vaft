"""Contract for Tutorial 05: MHD stability and perturbed equilibria (#1023, #2278).

The session follows EFIT -> CHEASE -> DCON/RDCON -> GPEC -> IMAS -> VAFT plots.
Offline (the default) every solver stage loads a precomputed, validated IMAS
sample (``vaft/data/samples/39915/mhd_stability``); lab mode runs a stage when
its executables are ready and maps the result to the same IDS. The tests pin
the narrative spine (rational surfaces -> CHEASE refinement -> ideal stability
-> tearing -> 3-D coils -> sector harmonics -> vacuum field -> relative phase by
linearity -> GPEC -> multi-time and multi-n) and the honesty rules: no solver in
the default path, every stage's execution / data source / validation reported
apart, a time read from the file rather than typed, sector angles taken from
each coil set, vacuum and total response kept distinct, and model excitations
never presented as hardware scenarios.
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

#: The offline run draws 30 figures and diagrams; the floor leaves a margin.
FIGURE_FLOOR = 27

#: Every solver the session can launch; each must sit under ``if READY[...]``.
SOLVER_CALLS = ("run_gpec_suite_case", "scan_chease")
STAGES = ("CHEASE", "DCON", "RDCON", "GPEC")


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


def test_the_default_path_runs_no_solver_and_says_so(executed):
    """Offline is the default even on a machine that has the solvers installed."""
    summary = "".join(output.get("text", "") for output in _cell(executed, "s05-stage-summary").outputs)
    for stage in STAGES:
        line = next(line for line in summary.splitlines() if line.startswith(stage))
        assert "not attempted" in line and "precomputed sample" in line and "passed" in line, line
        assert "offline mode" in line, line
    assert "Would " not in _printed(executed)


def test_every_solver_launch_is_gated_by_the_mode_and_the_preflight(book):
    executable = _executable(book)
    assert 'MODE = os.environ.get("VAFT_TUTORIAL_MODE", "offline")' in executable
    assert 'READY[stage] = MODE == "lab" and not missing' in executable
    assert "executable_ready(" in executable
    launches = 0
    for cell in _code_cells(book):
        tree = ast.parse(_source(cell))
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "READY[" in ast.unparse(node.test):
                guarded.update(id(child) for statement in node.body for child in ast.walk(statement))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in SOLVER_CALLS:
                launches += 1
                assert id(node) in guarded, f"{cell.id}: {ast.unparse(node)[:80]} runs outside `if READY[...]`"
    assert launches >= 5


def test_solver_workdirs_are_short_temporary_directories(book):
    """The GPEC suite reads paths into fixed-length Fortran strings (#1239)."""
    executable = _executable(book)
    assert 'LAB_ROOT = Path(tempfile.mkdtemp(prefix="vaft-t05-"))' in executable
    for cell in _code_cells(book):
        source = _strip_comments(_source(cell))
        if "GPECCaseInputs(" in source:
            assert "LAB_ROOT" in source or "tempfile.mkdtemp(prefix=" in source, cell.id


def test_live_and_sample_paths_meet_in_the_same_ids(book):
    """Each stage's live branch maps to IMAS and its fallback loads the IMAS sample."""
    for cell_id, builder in (("s05-dcon-run", "build_mhd_linear_ods("), ("s05-rdcon-run", "build_mhd_linear_ods("),
                             ("s05-gpec-run", "build_gpec_ideal_ods("), ("s05-chease-run", ".to_omas()")):
        source = _strip_comments(_source(_cell(book, cell_id)))
        assert builder in source and "mhd_stability_sample(" in source, cell_id
    for name in ("gpec.read_dcon_output(", "read_gpec_netcdf(", "read_pest3_matching_output("):
        assert name not in _executable(book), f"a plot path reads native output: {name}"


def test_the_offline_sample_holds_no_solver_intermediates():
    sample = ROOT / "vaft" / "data" / "samples" / "39915" / "mhd_stability"
    names = [path.name for path in sample.iterdir()]
    assert "manifest.yaml" in names
    assert all(name.endswith(".json.gz") or name == "manifest.yaml" for name in names), names


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
        _first_code_index(book, "scan_chease("),
        _first_code_index(book, 'modules=("dcon",)'),
        _first_code_index(book, '"kinked inward"'),
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
    was built with the canonical arguments; any other call would refuse.

    Every ``vaft.diagram.<builder>(...)`` call must match a
    :data:`vaft.diagram.build.CANONICAL` entry exactly; a keyword bound to the
    loop variable of a comprehension over a literal tuple is expanded.
    ``side_by_side`` only lays rendered diagrams out and is not a builder.
    """
    from vaft.diagram import build

    canonical = {(builder, repr(sorted(kwargs.items()))) for builder, kwargs in build.CANONICAL.values()}
    tree = ast.parse(_executable(book))
    loops = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            for generator in node.generators:
                if isinstance(generator.target, ast.Name):
                    try:
                        loops[generator.target.id] = ast.literal_eval(generator.iter)
                    except ValueError:   # a comprehension over runtime data, not a diagram variant
                        continue
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "diagram"):
            continue
        name = node.func.attr
        if name == "side_by_side":
            continue
        assert not node.args, (name, "positional arguments")
        variants = [{}]
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name) and keyword.value.id in loops:
                variants = [dict(v, **{keyword.arg: value}) for v in variants for value in loops[keyword.value.id]]
            else:
                variants = [dict(v, **{keyword.arg: ast.literal_eval(keyword.value)}) for v in variants]
        for kwargs in variants:
            assert (name, repr(sorted(kwargs.items()))) in canonical, (name, kwargs)
            checked += 1
    assert checked > 15


# ---------------------------------------------------------------------------
# Data honesty
# ---------------------------------------------------------------------------


def test_the_time_is_read_from_the_sample_manifest_not_typed(book):
    executable = _executable(book)
    assert 'SHOT, TIME_MS = int(MANIFEST["shot"]), int(MANIFEST["time_ms"])' in executable
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


def test_vacuum_and_total_response_are_kept_distinct(book, executed):
    text = _markdown(book)
    assert "#1243" in text and "con_flag" in text
    assert "vacuum against total" in _executable(book)
    printed = "".join(output.get("text", "") for output in _cell(executed, "s05-gpec-run").outputs)
    ratios = [float(value) for value in re.findall(r"^\s+\d+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)$",
                                                   printed, re.M)]
    assert len(ratios) == 12, printed          # 6 n=1 surfaces, UP and LOW
    assert max(ratios) < 1.0                   # screened everywhere on this equilibrium


def test_the_dcon_comparison_is_verified_at_two_resolutions(executed):
    printed = "".join(output.get("text", "") for output in _cell(executed, "s05-dcon-run").outputs)
    assert "-> stable: ideally STABLE at both resolutions" in printed
    assert "-> high_ip: ideally UNSTABLE, dominant m = 3" in printed


def test_rdcon_reports_resolution_sensitivity_rather_than_a_verdict(book, executed):
    printed = "".join(output.get("text", "") for output in _cell(executed, "s05-rdcon-table").outputs)
    assert "sign changes with resolution: unresolved" in printed
    assert "classically stable" in printed
    assert "not a tearing verdict" in _markdown(book)


def test_the_chease_step_compares_residuals_without_ranking_them(book, executed):
    assert "compute_grad_shafranov_residual(" in _executable(book)
    assert "**not** a ranking" in _markdown(book)
    printed = "".join(output.get("text", "") for output in _cell(executed, "s05-gs-residual").outputs)
    assert printed.count("median |residual| / RMS source") == 2


def test_model_excitations_are_not_presented_as_hardware_scenarios(book):
    text = _markdown(book)
    assert "Model space and hardware space are not the same" in text
    assert "Hardware feasibility" in text


def test_one_reconstruction_feeds_the_session_and_the_solver_gap_is_named(book):
    """The sample ODS's EFIT is the only equilibrium source; the legacy g-files are not read."""
    assert "#1246" in _markdown(book)
    executable = _executable(book)
    assert "vaft.omas.sample_ods(SHOT)" in executable
    assert 'data_path("efit' not in executable and 'data_path(f"efit' not in executable


def test_concept_diagrams_sit_with_the_physics_before_any_data(book):
    """Concept diagrams belong to Physical Context; workflow diagrams stay with their step."""
    load = next(i for i, cell in enumerate(book.cells)
                if cell.cell_type == "markdown" and cell.source.startswith("## Load / Prepare Data"))
    for name in ("sfl_coordinate_validity", "rational_surface", "kink_mode", "s_alpha_ballooning",
                 "tearing_layer_matching", "delta_prime"):
        assert _first_code_index(book, f"vaft.diagram.{name}()") < load, name
    for name in ("chease_coupling", "dcon_rdcon_stability", "gpec_plasma_response"):
        assert _first_code_index(book, f"vaft.diagram.{name}()") > load, name


def test_the_actuator_alias_step_is_taught_beside_the_sensor_one(book):
    text = _markdown(book)
    assert "finite toroidal actuators" in text and "finite toroidal sensors" in text
