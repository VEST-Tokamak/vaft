"""Contract for Tutorial 06: operational space and data-driven analysis (issue #1091).

The session builds its population offline from the six packaged VEST samples
through the canonical summary presets, and in lab mode from
``vaft.database.summary``. The tests pin the spine (units of analysis ->
coverage and validity -> representative states -> distributions ->
operational space -> beta validity -> density -> dimensionless variables ->
confinement -> confounding -> similarity -> events -> planned data-driven
representations) and the honesty rules: tables come only from the presets,
every validity rule prints what it removed, a reference limit carries its
source, the unsourced kink heuristic is never drawn, events are joined by time,
and machine learning is presented as planned, not as a working model.
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
NOTEBOOK = ROOT / "tutorial" / "06_operational_space_and_data_driven_analysis.ipynb"

MAX_OUTPUT_BYTES = 200_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")
FIGURE_FLOOR = 5


def _execute(notebook):
    previous = {key: os.environ.get(key) for key in ("MPLBACKEND", "VAFT_TUTORIAL_MODE")}
    os.environ["MPLBACKEND"] = "inline"
    os.environ.pop("VAFT_TUTORIAL_MODE", None)
    try:
        NotebookClient(notebook, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
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
    return " ".join(" ".join(_source(cell).split()) for cell in book.cells if cell.cell_type == "markdown")


def _cell(book, cell_id):
    return next(cell for cell in book.cells if cell.id == cell_id)


def _printed_by(executed, cell_id):
    return "".join(output.get("text", "") for output in _cell(executed, cell_id).outputs)


def _first_code_index(book, token):
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code" and token in _strip_comments(_source(cell)):
            return index
    raise AssertionError(f"no executable code cell uses {token!r}")


# ---------------------------------------------------------------------------
# Runs offline, from the presets
# ---------------------------------------------------------------------------


def test_the_notebook_runs_top_to_bottom(executed):
    assert _code_cells(executed)


def test_every_analysis_view_produces_a_figure(executed):
    drawn = [
        cell for cell in _code_cells(executed)
        if any(any(mime in output.get("data", {}) for mime in IMAGE_MIME_TYPES) for output in cell.get("outputs", []))
    ]
    assert len(drawn) >= FIGURE_FLOOR, [cell.id for cell in drawn]


def test_no_cell_dumps_a_data_object(executed):
    for cell in _code_cells(executed):
        for output in cell.get("outputs", []):
            for mime, payload in output.get("data", {}).items():
                if mime not in IMAGE_MIME_TYPES:
                    text = payload if isinstance(payload, str) else "".join(payload)
                    assert len(text) < MAX_OUTPUT_BYTES, (cell.id, mime)
            assert len("".join(output.get("text", ""))) < MAX_OUTPUT_BYTES, cell.id


def test_the_tables_come_from_the_canonical_presets(book):
    executable = _executable(book)
    assert "vaft.database.get_summary_preset(preset).extractor(ods, shot)" in executable
    assert "# tables = {preset: vaft.database.summary(" in _source(_cell(book, "s06-load-population"))


def test_the_database_path_is_gated_by_the_mode(book, executed):
    source = _strip_comments(_source(_cell(book, "s06-lab-population")))
    assert 'LAB = os.environ.get("VAFT_TUTORIAL_MODE", "offline") == "lab"' in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "LAB":
            assert "vaft.database.summary" in "\n".join(ast.unparse(statement) for statement in node.body)
            break
    else:
        raise AssertionError("the database summary is not inside `if LAB:`")
    assert "Offline: the database population needs lab mode" in _printed_by(executed, "s06-lab-population")
    assert "vaft.database.summary(" not in _strip_comments(
        "\n".join(_source(cell) for cell in _code_cells(book) if cell.id != "s06-lab-population"))


def test_the_notebook_uses_only_public_vaft_modules_and_no_local_functions(book):
    tree = ast.parse(_executable(book))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vaft"):
            assert not any(part.startswith("_") for part in node.module.split(".")), node.module
        assert not isinstance(node, ast.FunctionDef), f"notebook-local function {getattr(node, 'name', '')}"


# ---------------------------------------------------------------------------
# The spine
# ---------------------------------------------------------------------------


def test_the_session_climbs_from_records_to_representations(book):
    order = [
        _first_code_index(book, "per_shot = slices.groupby"),
        _first_code_index(book, "coverage = pd.DataFrame("),
        _first_code_index(book, "at_max_ip = "),
        _first_code_index(book, "ECDF"),
        _first_code_index(book, "stability.empirical_li_qa()"),
        _first_code_index(book, "beta_N_from_beta_a_B0_Ip"),
        _first_code_index(book, "greenwald_density"),
        _first_code_index(book, "rho_star_from_M_T_B_R_epsilon"),
        _first_code_index(book, "kadomtsev_constraint_from_engineering_exponents"),
        _first_code_index(book, "perform_ols_regression("),
        _first_code_index(book, "standardized = "),
        _first_code_index(book, "current_quench("),
    ]
    assert order == sorted(order), order
    assert book.cells.index(_cell(book, "s06-ml")) > book.cells.index(_cell(book, "s06-events"))


# ---------------------------------------------------------------------------
# Honesty
# ---------------------------------------------------------------------------


def test_every_validity_rule_reports_what_it_removed(executed):
    printed = _printed_by(executed, "s06-coverage")
    assert re.search(r"rule 1 -- .*: \d+ removed", printed)
    assert re.search(r"rule 2 -- .*: \d+ removed", printed)
    assert re.search(r"slices kept: \d+", printed)


def test_a_failed_preset_is_recorded_not_swallowed(executed):
    printed = _printed_by(executed, "s06-load-population")
    assert "48224 shot_overview: PlasmaTimingError" in printed


def test_the_representative_rule_is_stated_and_compared(book):
    source = _source(_cell(book, "s06-representative"))
    assert 'idxmax()' in source and "max Ip" in source and "max W_mhd" in source
    assert "A representative state is a modelling decision" in _markdown(book)


def test_reference_limits_carry_their_source_and_the_heuristic_is_not_drawn(book):
    text = _markdown(book)
    assert "reference drawn for comparison, not a VEST limit" in text
    assert "#350" in text
    assert "kink_stability_criterion(" not in _executable(book)
    assert "reference only" in _source(_cell(book, "s06-opspace"))


def test_an_observed_envelope_is_never_called_a_limit(book):
    text = _markdown(book)
    assert "An empty region is not by itself an operating limit" in text
    assert "an observed envelope, not a limit" in _source(_cell(book, "s06-opspace"))


def test_beta_validity_is_checked_before_beta_is_used(book, executed):
    assert "#386" in _markdown(book)
    assert "constrained by kinetic profiles: [48224]" in _printed_by(executed, "s06-beta")
    assert "vaft.diagram.troyon()" in _executable(book)


def test_greenwald_is_formed_in_one_unit(book, executed):
    source = _source(_cell(book, "s06-greenwald"))
    assert 'greenwald_fraction(state["ne_line_1e19_m3"], n_g)' in source
    f_g = float(re.search(r"f_G = ([0-9.]+)", _printed_by(executed, "s06-greenwald")).group(1))
    assert 0.01 < f_g < 1.0


def test_confounding_is_shown_within_and_across_shots(executed):
    printed = _printed_by(executed, "s06-regression")
    assert "pooled over" in printed and "alone (" in printed


def test_events_are_joined_to_states_by_time_not_position(book):
    source = _source(_cell(book, "s06-events"))
    assert 'slices["time_s"] < quench.time_80' in source
    assert "idxmax()" in source


def test_machine_learning_is_presented_as_planned_not_as_a_model(book):
    text = _markdown(book)
    assert "Planned — not implemented as a validated VAFT workflow" in text
    assert "train_model(" not in _executable(book)
    assert "split by shot" in text.lower()


def test_the_diagrams_are_canonical_so_they_render_without_tex(book):
    calls = re.findall(r"vaft\.diagram\.(\w+)\(([^)]*)\)", _executable(book))
    assert calls and all(arguments.strip() == "" for _name, arguments in calls)
