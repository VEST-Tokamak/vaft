"""Contract for Tutorial 03: equilibrium and kinetic profiles (issue #952).

The tutorial runs offline from the packaged VEST samples, while teaching the
same public workflow on any database shot.  The tests pin the narrative spine
(representation -> analytic geometry and equilibria -> solver taxonomy ->
experimental constraints -> kinetic mapping and fitting -> single-discharge
EFIT analysis -> flux coordinates -> kinetic state -> Grad-Shafranov residual
-> multi-time and multi-shot comparison) and the data-honesty rules: slices
found from the data rather than typed, no literal discharge times, the EFIT
weighting stated as it is, and every lab-mode line a real, parseable call.
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
NOTEBOOK = ROOT / "tutorial" / "03_equilibrium_and_kinetic_profiles.ipynb"

MAX_OUTPUT_BYTES = 200_000
MAX_IMAGE_BYTES = 2_000_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")

#: The two lab-mode lines, commented in the committed notebook, and the offline
#: lines each one replaces when uncommented.
DATABASE_SWAPS = {
    "s03-load-sample": (
        "ods = vaft.omas.sample_ods(SHOT)",
        "# ods = vaft.database.load(SHOT)",
    ),
    "s03-multi-load": (
        "ods_list = [vaft.omas.sample_ods(shot) for shot in shots]",
        "# ods_list = vaft.database.load(shots)",
    ),
}

#: Keywords that select an equilibrium slice or a camera frame: never a literal.
SLICE_KEYWORDS = ("time_slice", "frame_index", "time_index")


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
    assert len(drawn) >= 27, [cell.id for cell in drawn]


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


def test_the_default_path_needs_no_credentials_and_no_external_code(book):
    """Offline mode reaches neither the database nor CHEASE.

    CHEASE is taught, but only in the commented exercise, so a reader without
    it installed still executes every cell.
    """
    executable = _executable(book)
    assert "ods = vaft.omas.sample_ods(SHOT)" in executable
    assert "vaft.omas.sample_ods(shot) for shot in shots" in executable
    assert "vaft.database" not in executable
    for token in ("scan_chease", "run_chease", "refine_equilibrium", "run_kinetic_efit"):
        assert token not in executable, token


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
              "find_repository_root", "savefig", "getattr(vaft", "data_path(")
    executable = _executable(book)
    for token in banned:
        assert token not in executable, token


def test_no_literal_discharge_times(book):
    """Times come from the data: no float literal where VEST discharge times live."""
    for cell in _code_cells(book):
        for node in ast.walk(ast.parse(_source(cell))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                assert not 0.2 <= node.value <= 0.4, (cell.id, node.value)


def test_no_literal_equilibrium_slice_or_frame(book):
    """Slices and frames are found from the data, never typed.

    No ``time_slice=3`` / ``frame_index=10``, no ``[8]`` on an equilibrium
    slice array, and no ``equilibrium.time_slice.3`` path.
    """
    for cell in _code_cells(book):
        source = _source(cell)
        assert not re.search(r"time_slice\.\d", _strip_comments(source)), cell.id
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.keyword) and node.arg in SLICE_KEYWORDS:
                assert not isinstance(node.value, ast.Constant), (cell.id, node.arg)
            # an index straight into the slice array: ods["equilibrium.time_slice"][8]
            if isinstance(node, ast.Subscript) and re.search(r"time_slice['\"]\]$", ast.unparse(node.value)):
                index = node.slice
                if isinstance(index, ast.UnaryOp):
                    index = index.operand
                assert not (isinstance(index, ast.Constant) and isinstance(index.value, int)), cell.id


def test_the_shot_number_is_written_once_in_part_one(book):
    executable = _executable(book)
    assert executable.count("SHOT = 39915") == 1
    part_one = [cell for cell in _code_cells(book) if not cell.id.startswith("s03-multi")]
    for cell in part_one:
        if cell.id == "s03-load-sample":
            continue
        assert "39915" not in _strip_comments(_source(cell)), cell.id


def test_the_slices_are_found_once_and_reused(book):
    executable = _executable(book)
    assert executable.count("i_rep = ") == 1
    assert "usable = np.isfinite(ip) & (ip != 0)" in executable
    found = _first_code_index(book, "i_rep = ")
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code" and "i_rep" in _strip_comments(_source(cell)):
            assert index >= found, cell.id


# ---------------------------------------------------------------------------
# The session's order
# ---------------------------------------------------------------------------


ORDER = (
    "vaft.omas.sample_ods(SHOT)",
    "usable_slices = ",
    "vaft.omas.sample_ods(KINETIC_SHOT)",
    "plot_equilibrium_field_2d",
    "plot_equilibrium_overview_profiles",
    "fit_miller_surface",
    "plot_miller_surfaces",
    "solovev_example",
    "plot_equilibrium_overview_constraints",
    "plot_equilibrium_overview_residuals",
    "fit_quality_metrics(ods",
    "sigma_unit_factor(table)",
    "plot_equilibrium_overview_constraint_weights",
    "compare_flux_mapping",
    "plot_thomson_scattering_profile_fit",
    "profile_fit_report_thomson_scattering",
    "plot_charge_exchange_profile_fit",
    "electron_pressure",
    "z_eff_from_n_s_Z_s",
    "update_equilibrium_derived_profiles(ods)",
    "plot_equilibrium_overview_histories",
    "plot_equilibrium_time_shape(ods)",
    "plot_camera_visible_image_efit_overlay",
    'plot_equilibrium_profile_q(ods, time_slice=i_rep, coordinate="psi_norm")',
    'plot_equilibrium_profile_q(ods, time_slice=i_rep, coordinate="r_major")',
    "normalized_gradient_scale_length",
    "compute_grad_shafranov_residual",
    "vaft.omas.sample_ods(shot) for shot in shots",
    "plot_equilibrium_time_shape(ods_list)",
)


def test_the_analysis_runs_in_the_session_order(book):
    positions = [_first_code_index(book, token) for token in ORDER]
    assert positions == sorted(positions), dict(zip(ORDER, positions))


def test_the_constraints_come_before_the_reconstruction_is_analysed(book):
    """#952: what the experiment constrains is taught before what EFIT returned."""
    derive = _first_code_index(book, "update_equilibrium_derived_profiles(ods)")
    for token in ("plot_equilibrium_overview_constraints", "compare_flux_mapping",
                  "profile_fit_report_charge_exchange"):
        assert _first_code_index(book, token) < derive, token


def test_the_fixed_headings_wrap_every_subsection(book):
    """The eight ``##`` headings are the only second-level ones; everything else nests."""
    for cell in book.cells:
        if cell.cell_type != "markdown":
            continue
        for line in _source(cell).splitlines():
            if line.startswith("#") and not line.startswith("# 03."):
                assert re.match(r"#{2,4} ", line), (cell.id, line)


# ---------------------------------------------------------------------------
# Every new API is used
# ---------------------------------------------------------------------------


def test_the_analytic_apis_are_used(book):
    executable = _executable(book)
    for api in ("fit_miller_surface", "miller_surfaces(", "plot_miller_surfaces",
                "solovev_example", "plot_solovev_equilibrium", "derive_boundary_representation",
                "as_equilibrium", '"x_points"'):
        assert api in executable, api
    for topology in ('"limited"', '"single_null"', '"double_null"'):
        assert topology in executable, topology


def test_the_constraint_apis_are_used(book):
    executable = _executable(book)
    for api in ("plot_equilibrium_overview_constraints", "plot_equilibrium_overview_constraint_coverage",
                "plot_equilibrium_overview_residuals", "plot_equilibrium_overview_fit_quality",
                "plot_equilibrium_overview_constraint_weights", "fit_quality_metrics",
                "constraint_table", "sigma_unit_factor"):
        assert api in executable, api


def test_the_kinetic_apis_are_used(book):
    executable = _executable(book)
    for api in ("sample_equilibria(KINETIC_SHOT)", "compare_flux_mapping",
                "equilibrium_mapping_thomson_scattering", "equilibrium_mapping_charge_exchange",
                "profile_fit_report_thomson_scattering", "profile_fit_report_charge_exchange",
                'fitting_function="gp"', 'fitting_function="free_polynomial"',
                'field="te"', 'field="ne"', 'field="ti"', 'field="vphi"',
                ".summary()", "plot_electron_temperature_profile", "plot_electron_density_profile"):
        assert api in executable, api
    for key in ('"efit_magnetic"', '"efit_kinetic"', '"chease"'):
        assert key in executable, key


def test_the_derived_state_formulas_are_used(book):
    executable = _executable(book)
    for api in ("vaft.formula.electron_pressure", "vaft.formula.ion_pressure",
                "vaft.formula.z_eff_from_n_s_Z_s", "vaft.formula.impurity_fraction_from_effective_charge",
                "vaft.formula.normalized_gradient_scale_length",
                "vaft.formula.electron_collisionality_sauter",
                "vaft.formula.rho_star_from_M_T_B_R_epsilon"):
        assert api in executable, api


def test_each_supported_coordinate_is_actually_demonstrated(book):
    """The session claims a vocabulary; it has to show it."""
    executable = _executable(book)
    for coordinate in ("psi_norm", "rho_tor_norm", "r_major"):
        assert f'coordinate="{coordinate}"' in executable, coordinate


def test_the_interactive_cells_run_headless(book):
    """`interaction_backend="auto"` resolves to ipywidgets under nbclient."""
    for cell in _code_cells(book):
        source = _source(cell)
        if "interactive=True" in source:
            assert 'interaction_backend="none"' in source, cell.id


def test_the_camera_overlay_is_guarded_and_uses_the_shot(book):
    executable = _executable(book)
    assert "shot=SHOT" in executable
    assert "except FileNotFoundError" in _source(_cell(book, "s03-camera-load"))
    assert "else:" in _source(_cell(book, "s03-camera-overlay"))


def test_multi_shot_comparison_uses_lists(book):
    executable = _executable(book)
    assert "shots = [39915, 41524, 41672]" in executable
    assert "plot_equilibrium_time_shape(ods_list)" in executable


# ---------------------------------------------------------------------------
# What the notebook prints
# ---------------------------------------------------------------------------


def test_the_usable_window_is_found_from_the_data(executed):
    printed = _printed(executed)
    assert "usable slices: 8 of 9" in printed
    assert "representative slice:" in printed
    assert "every stored slice is usable" not in printed


def test_the_derivation_step_is_taught_rather_than_assumed(book, executed):
    """The session's most transferable habit: a refused plot is often underived."""
    assert "update_equilibrium_derived_profiles" in _executable(book)
    printed = _printed(executed)
    assert "beta_n available?  False" in printed
    assert "beta_n available?  True" in printed


def test_the_rho_proxy_is_checked_before_and_after_derivation(executed):
    printed = _printed(executed)
    assert "stored rho_tor_norm is sqrt(psi_N): True" in printed
    assert "derived rho_tor_norm vs sqrt(psi_N)" in printed


def test_the_efit_weighting_is_shown_as_it_is(executed):
    printed = _printed(executed)
    assert "total chi-square" in printed
    assert "Ip chi-square as EFIT reported it" in printed
    assert "vessel accounting term" in printed
    assert "Ip chi-square, plasma-only residual" in printed
    assert "sigma EFIT used" in printed
    assert "diamagnetic flux: measured +" in printed  # paramagnetic (#1196)


def test_the_topologies_are_classified(executed):
    printed = _printed(executed)
    assert re.search(r"limited\s+-> topology limited", printed)
    assert re.search(r"single_null\s+-> topology \w*single_null", printed)
    assert re.search(r"double_null\s+-> topology double_null", printed)


def test_the_flux_mapping_and_fits_are_quantified(executed):
    printed = _printed(executed)
    assert "largest rho_N shift between the two equilibria" in printed
    for quantity in ("t_e [polynomial", "t_e [gp", "t_i [polynomial", "velocity_tor [polynomial",
                     "velocity_tor [free_polynomial"):
        assert quantity in printed, quantity
    assert "chi2/nu=" in printed


def test_the_derived_state_is_reported(executed):
    printed = _printed(executed)
    assert "p_th =" in printed
    assert "Z_eff =" in printed
    assert "nu_e* =" in printed


def test_the_residual_is_measured_on_both_equilibria(executed):
    """Step 8 rests on these two numbers being real."""
    printed = _printed(executed)
    assert "EFIT   median relative residual" in printed
    assert "CHEASE median relative residual" in printed


def test_the_multi_shot_table_is_printed(executed):
    printed = _printed(executed)
    for shot in ("39915", "41524", "41672"):
        assert shot in printed
    assert "beta_N" in printed and "q95" in printed


# ---------------------------------------------------------------------------
# Physics and data honesty in the prose
# ---------------------------------------------------------------------------


def test_the_grad_shafranov_assumptions_are_stated(book):
    markdown = _markdown(book).lower()
    for token in ("axisymmetry", "isotropic scalar pressure", "static equilibrium", "session 05"):
        assert token in markdown, token


def test_shape_is_not_equilibrium(book):
    markdown = _markdown(book)
    assert "shape parameterisation" in markdown
    assert "force-balanced solution" in markdown
    assert "Solov'ev" in markdown and "Miller" in markdown


def test_the_solver_taxonomy_names_what_vaft_runs(book):
    markdown = _markdown(book)
    for token in ("vaft.code.chease", "vaft.code.tes", "vaft.code.tokamaker", "vaft.code.efit",
                  "fixed boundary", "free boundary"):
        assert token in markdown, token


def test_the_efit_weighting_facts_are_stated(book):
    markdown = _markdown(book)
    assert "#918" in markdown
    assert "1000 T" in markdown
    assert "SERROR" in markdown
    assert "the weights, not the diagnostics, set this fit" in markdown
    assert "did not constrain the fit" in markdown


def test_zeff_is_not_presented_as_a_reconstructed_profile(book):
    markdown = _markdown(book)
    assert "quasi-neutrality" in markdown
    assert "no visible-bremsstrahlung" in markdown


def test_the_layers_are_named(book):
    markdown = _markdown(book)
    for token in ("thomson_scattering", "charge_exchange", "core_profiles", "Measured", "Derived"):
        assert token in markdown, token
    assert "not a plotting option" in markdown


def test_the_exercise_executes_nothing_on_its_own(book):
    exercise = _cell(book, "s03-exercise-cell")
    for line in _source(exercise).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line
    markdown = _markdown(book).lower()
    for token in ("measured", "reconstructed", "assumed", "derived"):
        assert token in markdown, token


def test_the_unsupported_coordinate_is_named_rather_than_quietly_skipped(book):
    """Straight-field-line coordinates are what a reader will ask for next."""
    markdown = _markdown(book)
    assert "traight-field-line" in markdown
    assert "#472" in markdown


def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 3
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]
