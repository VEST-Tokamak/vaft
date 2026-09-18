"""Contract for Tutorial 04: fluctuation diagnostics for plasma perturbations
and transient events (issue #1005).

The tutorial runs offline from the packaged VEST samples, while teaching the
same public workflow on any database shot. The tests pin the narrative spine
(perturbation -> lab and plasma frame -> what each diagnostic measures and at
what bandwidth -> Mirnov -> soft X-ray -> fast camera -> edge -> coherence and
phase -> mode evidence -> rotation and rational surfaces -> transients as
sequences -> multi-time and multi-shot comparison) and the data-honesty rules:
windows found from the data, no literal discharge times, a mode number always
printed with its alias step, coherence always with its significance line, a
refused fit reported with its reason, and no event named by the notebook.
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
NOTEBOOK = (
    ROOT / "tutorial"
    / "04_fluctuation_diagnostics_for_plasma_perturbations_and_transient_events.ipynb"
)

MAX_OUTPUT_BYTES = 200_000
MAX_IMAGE_BYTES = 2_000_000
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")

#: The two lab-mode lines, commented in the committed notebook, and the offline
#: lines each one replaces when uncommented.
DATABASE_SWAPS = {
    "s04-load-sample": (
        "ods = vaft.omas.sample_ods(SHOT)",
        "# ods = vaft.database.load(SHOT)",
    ),
    "s04-multi-load": (
        "ods_list = [vaft.omas.sample_ods(shot) for shot in shots]",
        "# ods_list = vaft.database.load(shots)",
    ),
}

#: Keywords that select an equilibrium slice or a camera frame: never a literal.
SLICE_KEYWORDS = ("time_slice", "frame_index", "time_index")

#: The offline run draws 30 figures; the floor leaves a small margin so that a
#: cell which legitimately degrades (and prints why) does not fail it.
FIGURE_FLOOR = 28


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
    assert len(drawn) >= FIGURE_FLOOR, [cell.id for cell in drawn]


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


def test_the_default_path_needs_no_credentials_and_no_optional_package(book):
    """Offline mode reaches neither the database nor fcwt.

    ``method="cwt"`` needs the optional fcwt package, so it is taught only in
    the commented exercise; a reader without it still executes every cell.
    """
    executable = _executable(book)
    assert "ods = vaft.omas.sample_ods(SHOT)" in executable
    assert "vaft.omas.sample_ods(shot) for shot in shots" in executable
    assert "vaft.database" not in executable
    assert '"cwt"' not in executable


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

    Needs HSDS credentials (``~/.hscfg``) and ``VAFT_DATABASE_TESTS=1``. A
    database record without soft X-ray or the fluctuation array degrades: the
    cells that need them print why and draw nothing.
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
            assert not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ), cell.id


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

    No ``time_index=3`` / ``frame_index=10``, no ``[8]`` on an equilibrium
    slice array, and no ``equilibrium.time_slice.3`` path.
    """
    for cell in _code_cells(book):
        source = _source(cell)
        assert not re.search(r"time_slice\.\d", _strip_comments(source)), cell.id
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.keyword) and node.arg in SLICE_KEYWORDS:
                assert not isinstance(node.value, ast.Constant), (cell.id, node.arg)
            if isinstance(node, ast.Subscript) and re.search(r"time_slice['\"]\]$", ast.unparse(node.value)):
                index = node.slice
                if isinstance(index, ast.UnaryOp):
                    index = index.operand
                assert not (isinstance(index, ast.Constant) and isinstance(index.value, int)), cell.id


def test_channels_are_chosen_by_name_not_by_position(book):
    """Indices depend on which archive fields a shot carried; names do not."""
    executable = _executable(book)
    assert "probe_names.index(" in executable
    assert "sxr_names.index(" in executable
    assert "camera_probe_names.index(" in executable


def test_the_shot_number_is_written_once_in_part_one(book):
    executable = _executable(book)
    assert executable.count("SHOT = 45531") == 1
    for cell in _code_cells(book):
        if cell.id == "s04-load-sample" or cell.id.startswith("s04-multi"):
            continue
        assert "45531" not in _strip_comments(_source(cell)), cell.id


def test_the_windows_come_from_the_timing_helpers(book):
    executable = _executable(book)
    assert executable.count("timing = plasma_timing(ods)") == 1
    assert "window = (timing.onset, timing.offset)" in executable
    assert "floor_window = " in executable
    assert "plasma_timing(camera_ods)" in executable
    assert "plasma_timing(reference)" in executable


def test_the_interactive_cells_run_headless(book):
    """`interaction_backend="auto"` resolves to ipywidgets under nbclient."""
    for cell in _code_cells(book):
        source = _source(cell)
        if "interactive=True" in source:
            assert 'interaction_backend="none"' in source, cell.id


# ---------------------------------------------------------------------------
# The session's order
# ---------------------------------------------------------------------------


ORDER = (
    "vaft.omas.sample_ods(SHOT)",
    "vaft.omas.sample_ods(CAMERA_SHOT)",
    "plasma_timing(ods)",
    "fluctuation_bandwidths(ods)",
    "plot_fluctuation_frequency_coverage",
    "plot_magnetics_geometry_poloidal",
    "plot_mirnov_time_voltage",
    "track_dominant_frequency(mode_evolution",
    'method="hann_fft"',
    "plot_mirnov_spectrum(",
    "compute_band_power(mode_evolution",
    "plot_mirnov_spatial_phase(ods, time=t_mode",
    "toroidal_phase_fit_at_time(trio_time",
    "plot_mirnov_spatial_phase(reference",
    "plot_soft_x_rays_geometry_lines_of_sight",
    "sxr_band_signals",
    "plot_soft_x_rays_spectrogram",
    "sxr_electron_temperature",
    "rank_toroidal_mode_numbers",
    "track_reference_frequency",
    "plot_camera_visible_image_fluctuation",
    "plot_camera_visible_spectrogram",
    "pixelwise_spectrogram",
    "plot_camera_visible_image_mhd_power",
    "stats.pearsonr",
    'ods["langmuir_probes.embedded"]',
    "significance_95",
    "plot_diagnostics_spectrum_coherence",
    "plot_cross_spectrum",
    "evidence = [",
    "velocity_tor",
    "find_rational_surfaces",
    "derive_global_descriptors",
    "current_quench(ip_time, ip)",
    "vertical_position_history(ods)",
    "smoothing_s=smoothing",
    "vertical_position_history(reference)",
    "n_time, n_value",
    "vaft.omas.sample_ods(shot) for shot in shots",
    "columns = (",
)


def test_the_analysis_runs_in_the_session_order(book):
    positions = [_first_code_index(book, token) for token in ORDER]
    assert positions == sorted(positions), dict(zip(ORDER, positions))


def test_the_diagnostics_come_in_the_issue_order(book):
    """#1005: Mirnov, then SXR, then the camera, then density and edge, then their combination."""
    steps = [
        _first_code_index(book, token)
        for token in ("plot_mirnov_time_voltage", "sxr_band_signals", "pixelwise_spectrogram",
                      'ods["langmuir_probes.embedded"]', "plot_diagnostics_spectrum_coherence")
    ]
    assert steps == sorted(steps)


def test_the_fixed_headings_wrap_every_subsection(book):
    """The eight ``##`` headings are the only second-level ones; everything else nests."""
    for cell in book.cells:
        if cell.cell_type != "markdown":
            continue
        for line in _source(cell).splitlines():
            if line.startswith("#") and not line.startswith("# 04."):
                assert re.match(r"#{2,4} ", line), (cell.id, line)


# ---------------------------------------------------------------------------
# Every new API is used
# ---------------------------------------------------------------------------


def test_the_spectral_apis_are_used(book):
    executable = _executable(book)
    for api in ("compute_psd(", "compute_spectrogram(", "compute_band_power(", "fit_power_law_spectrum(",
                "cross_spectrum(", "track_dominant_frequency(", "track=(", "max_jump=",
                "vaft.omas.fluctuation_bandwidths", "vaft.plot.plot_fluctuation_frequency_coverage",
                "vaft.omas.plot_diagnostics_spectrum_coherence", "vaft.plot.plot_cross_spectrum",
                "x_signal=", "y_signal=", "phenomena="):
        assert api in executable, api


def test_the_mirnov_apis_are_used(book):
    executable = _executable(book)
    for api in ("plot_mirnov_time_voltage", "plot_mirnov_spectrogram", "plot_mirnov_spectrum(",
                "plot_mirnov_spatial_phase", "toroidal_phase_fit_at_time(", "toroidal_array_for_shot(",
                "show_fit=False", 'method="stft"', 'method="hann_fft"', "available_plots("):
        assert api in executable, api


def test_the_soft_x_ray_apis_are_used(book):
    executable = _executable(book)
    for api in ("soft_x_rays.sxr_band_signals", "soft_x_rays.sxr_te_pairs_from_ods",
                "soft_x_rays.sxr_electron_temperature", "soft_x_rays.load_te_ratio_calibration",
                "soft_x_rays.hilbert_instantaneous_phase", "soft_x_rays.rank_toroidal_mode_numbers",
                "plot_soft_x_rays_spectrogram", "plot_soft_x_rays_geometry_lines_of_sight",
                "plot_soft_x_rays_time_power", "baseline_start="):
        assert api in executable, api


def test_the_camera_apis_are_used(book):
    executable = _executable(book)
    for api in ("camera_fluctuation.subtract_temporal_background", "camera_fluctuation.summed_region_signal",
                "camera_fluctuation.pixelwise_spectrogram", "camera_fluctuation.track_reference_frequency",
                "camera_fluctuation.mhd_band_power", "camera_fluctuation.normalize_by_local_emission",
                "plot_camera_visible_spectrogram", "plot_camera_visible_image_fluctuation",
                "plot_camera_visible_image_mhd_power", "centre_frequency="):
        assert api in executable, api


def test_the_transient_and_equilibrium_apis_are_used(book):
    executable = _executable(book)
    for api in ("current_quench(", "current_spike(", "vaft.omas.vertical_position_history",
                "smoothing_s=", "lookback_s=", "find_rational_surfaces", "as_equilibrium",
                "derive_global_descriptors"):
        assert api in executable, api


def test_multi_shot_comparison_uses_lists(book):
    executable = _executable(book)
    assert "shots = [45531, 39915, 41524, 41672]" in executable
    assert "for shot, case in zip(shots, ods_list)" in executable


# ---------------------------------------------------------------------------
# What the notebook prints
# ---------------------------------------------------------------------------


def test_the_bandwidths_are_read_from_the_records(executed):
    printed = _printed(executed)
    assert re.search(r"45531\s+Mirnov / magnetic probes \(2000 kHz\)\s+2000\.0 kHz\s+1000\.0 kHz", printed)
    assert re.search(r"45531\s+Soft X-ray\s+976\.6 kHz\s+488\.3 kHz", printed)
    assert re.search(r"40600\s+FAST camera\s+50\.0 kHz\s+25\.0 kHz", printed)
    assert "no Interferometer record in this input" in printed


def test_the_transient_is_measured_against_a_floor(executed):
    printed = _printed(executed)
    assert "band power before the coils fired" in printed
    assert "fluctuation power rises above the pre-discharge floor: True" in printed


def test_the_spectral_index_is_reported_as_a_derivative_index(executed):
    """A Mirnov voltage is dB/dt, so an index fitted to it is the field's plus two."""
    printed = _printed(executed)
    assert "alpha(dB/dt)" in printed and "alpha(B)" in printed
    assert "R^2" in printed


def test_the_array_sets_the_resolution_and_the_session_says_so(executed):
    """45531's outboard array: three IMAS angles, 90 degrees apart, n modulo 4.

    The identifiers carry the VEST clock angles 45/135/225; their IMAS toroidal
    angles are the reflection of those (issue #718).
    """
    printed = _printed(executed)
    assert "toroidal angles that recorded: [135.0, 225.0, 315.0]" in printed
    assert "n is resolved modulo 4" in printed
    assert "alias step 4" in printed


def test_the_mode_number_is_reported_with_its_limits(executed):
    """The midplane trio carries n = 1 modulo 4; the other trios do not reproduce it."""
    printed = _printed(executed)
    assert re.search(r"L1-03\s+0\.00\s+1\s+\d+\.\d deg", printed)
    assert "n = 1 modulo 4 on the midplane trio" in printed
    assert "off-midplane trios disagree" in printed
    assert "n=1 mod 4" in printed


def test_the_shot_without_an_array_refuses_the_fit(executed):
    """#724/#825: 39915 publishes field 171 once, and the guard counts acquisitions.

    Replaces the xfail pins on the pre-fix mapper, which printed a
    ``:phase_reference`` twin whose samples equalled C2-05's.
    """
    printed = _printed(executed)
    assert "shot 39915: mirnov_spatial_phase available? False" in printed
    assert "each carrying its own waveform" in printed
    assert "the call refuses:" in printed
    assert "toroidal array: None" in printed
    assert ":phase_reference" not in printed


def test_the_soft_x_ray_steps_report_what_they_did(executed):
    printed = _printed(executed)
    assert "vacuum-reference subtraction: not applied" in printed
    assert "16 Be/Al chord pairs on the bottom array" in printed
    assert "n candidates, best first" in printed


def test_coherence_is_reported_with_its_significance(executed):
    printed = _printed(executed)
    match = re.search(r"chords above the line: (\d+) of 52", printed)
    assert match and int(match.group(1)) >= 1
    assert "95 % line" in printed
    # The short window averages two segments, and its line rises accordingly.
    assert re.search(r"first 4 ms\s+:\s+2 segments, 95 % line 0\.95", printed)


def test_the_camera_follows_the_probe(executed):
    printed = _printed(executed)
    match = re.search(r"track correlation, detrended: r = ([+-]\d\.\d+)", printed)
    assert match and float(match.group(1)) > 0.3
    assert "coherent bins 3-15 kHz:" in printed


def test_the_edge_diagnostics_are_bounded_by_their_bandwidth(executed):
    printed = _printed(executed)
    assert "no interferometer record" in printed
    assert "H-alpha Nyquist 12.5 kHz" in printed


def test_the_rotation_estimate_is_labelled_as_another_shot(executed):
    printed = _printed(executed)
    assert "shot 48224:" in printed
    assert "n = 1: n f_phi" in printed


def test_n_alone_does_not_fix_a_surface(executed):
    printed = _printed(executed)
    assert re.search(r"n = 1: \d+ surfaces, m = 3-", printed)
    assert "q = 1 surface at the representative slice: False" in printed


def test_the_transients_are_measured_not_named(executed):
    printed = _printed(executed)
    assert "vertical position: no equilibrium time slices" in printed
    assert "80-20 time 7.23 ms" in printed
    assert "spike inside the quench: +6.9 kA at 327.52 ms" in printed
    # The steepest slope depends on the smoothing; the 80-20 time does not.
    slopes = [float(value) for value in re.findall(r"steepest dIp/dt\s+(-?\d+\.\d) MA/s", printed)]
    assert len(slopes) >= 4 and max(slopes[-3:]) - min(slopes[-3:]) > 20.0
    for label in ("IRE", "disruption"):
        assert label not in printed, label


def test_the_multi_shot_table_is_printed(executed):
    printed = _printed(executed)
    for shot in ("45531", "39915", "41524", "41672"):
        assert shot in printed
    assert "not allowed" in printed
    assert "n on the midplane trio over" in printed


# ---------------------------------------------------------------------------
# Physics and data honesty in the prose
# ---------------------------------------------------------------------------


def test_the_frames_and_the_doppler_shift_are_taught(book):
    markdown = _markdown(book)
    for token in (r"f_{\mathrm{lab}} \approx f_{\mathrm{plasma}} + n\, f_\phi", "locked mode",
                  "Alfvén eigenmode", "tearing mode", "Zero observed frequency is not zero perturbation"):
        assert token in markdown, token


def test_the_scale_table_is_order_of_magnitude(book):
    markdown = _markdown(book)
    assert "order of magnitude" in markdown
    assert "not a mode-identification rule" in markdown


def test_the_named_gaps_are_stated_with_their_issues(book):
    """An unlabelled approximation in teaching material is worse than an honest gap."""
    markdown = _markdown(book)
    assert "#1005" in markdown
    assert "#506" in markdown                           # no (m, n) -> surface resolver
    assert "#460" in markdown                           # no f = n f_phi overlay
    assert "#724" in markdown and "#825" in markdown    # field 171 published twice
    assert "#472" in markdown                           # no straight-field-line coordinates
    assert "classifier" in markdown                     # no IRE / disruption classifier
    assert "Poloidal $m$ is not supported" in markdown


def test_the_method_notebook_is_the_depth_reference(book):
    assert "notebooks/fluctuation_diagnostics_analysis.ipynb" in _markdown(book)


def test_coincidence_is_not_coherence(book):
    markdown = _markdown(book)
    assert "The same event timing is not the same coherent mode" in markdown
    assert "segment count" in markdown
    assert "Sampling rate alone is never detectability" in markdown


def test_the_repository_only_records_are_declared(book):
    assert "source checkout" in _markdown(book)


def test_the_exercise_executes_nothing_on_its_own(book):
    exercise = _cell(book, "s04-exercise-cell")
    for line in _source(exercise).splitlines():
        assert not line.strip() or line.lstrip().startswith("#"), line
    assert "Classify the observation" in _markdown(book)


def test_the_session_declares_itself_complete_in_both_modes(book):
    metadata = book.metadata.get("vaft_tutorial", {})
    assert metadata.get("session") == 4
    assert metadata.get("status") == "complete"
    assert list(metadata.get("modes", [])) == ["offline", "lab"]
