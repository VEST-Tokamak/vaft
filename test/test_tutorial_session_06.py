"""Contract for Tutorial 06: operational history, summary contracts and stability space.

The session runs offline from the repository-only sample collections
``vaft/data/samples/vest_operational_history`` and ``vaft/data/samples/vest_tier_a``
(read through ``vaft.data.data_path``) and from the packaged sample shots, and in
lab mode rebuilds the population with ``vaft.database.summary``. The tests pin
the spine (A: database -> summary row and its physics chain; B: the MHD
parameters recomputed from their integrals; C: the registered stability limits
with their sources and schematics; D: the summary tables; E: operating space,
stability join, classification, distributions) and the honesty rules: only the three in-scope presets, skipped
shots printed, file hashes checked against the manifest, joins validated at
their grain, no boundary drawn on a foreign quantity, ideal/resistive/local
layers and their QA kept separate, and database access only in lab mode.
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

IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")
FIGURE_FLOOR = 26  # the offline run draws 29 figures and schematics
IN_SCOPE_PRESETS = {"shot_overview", "equilibrium_global", "core_profiles"}
OUT_OF_SCOPE_PRESETS = ("efit_magnetic_reliability", "efit_kinetic_reliability", "efit_reliability",
                        "volume_averaged", "neoclassical", "turbulent_transport", "classical_transport")


def _execute(notebook, output_dir):
    keys = ("MPLBACKEND", "VAFT_TUTORIAL_MODE", "VAFT_TUTORIAL_OUTPUT_DIR")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["MPLBACKEND"] = "inline"
    os.environ.pop("VAFT_TUTORIAL_MODE", None)
    os.environ["VAFT_TUTORIAL_OUTPUT_DIR"] = str(output_dir)
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
def executed(tmp_path_factory):
    return _execute(nbformat.from_dict(nbformat.read(NOTEBOOK, as_version=4)), tmp_path_factory.mktemp("s06"))


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
    return next(cell for cell in book.cells if cell.get("id") == cell_id)


def _text(cell):
    parts = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            parts.append(output.get("text", ""))
        parts.append(str(output.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def _first_cell_with(book, token):
    for index, cell in enumerate(_code_cells(book)):
        if token in _strip_comments(_source(cell)):
            return index
    raise AssertionError(f"no code cell contains {token!r}")


def test_notebook_runs_offline_and_draws_its_figures(executed):
    errors = [o for c in _code_cells(executed) for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors
    figures = sum(any(m in o.get("data", {}) for m in IMAGE_MIME_TYPES)
                  for c in _code_cells(executed) for o in c.get("outputs", []))
    assert figures >= FIGURE_FLOOR


def test_the_spine_runs_in_order(book):
    tokens = ["vaft.database.open(", "get_summary_preset(name)", "summary(odss=samples",
              "li_3_from_Bp2_volume_integral(", "fb.get_boundary(", "vaft.diagram.troyon()",
              "overview_preset = ", "ip_measured_a", "h_factor(", '(axes[0], "qstar_in"', "li_qa_pair(",
              'operational_space_population(opspace, "beta_n_li"', 'operational_space_population(opspace, "troyon"',
              '"hugill_st", color=', "stability_state = ", "stability_atlas_populations("]
    positions = [_first_cell_with(book, token) for token in tokens]
    assert positions == sorted(positions), dict(zip(tokens, positions))


def test_only_the_three_in_scope_presets(book):
    code = _executable(book)
    assert not [name for name in OUT_OF_SCOPE_PRESETS if name in code]
    named = set(re.findall(r'preset="(\w+)"', code)) | set(re.findall(r'PRESETS = \(([^)]*)\)', code)[0].replace('"', "").replace(" ", "").split(","))
    assert named - {""} == IN_SCOPE_PRESETS


def test_samples_come_from_the_collections_with_checked_hashes(book, executed):
    code = _executable(book)
    assert 'data_path("samples/vest_tier_a")' in code
    assert 'data_path("samples/vest_operational_history")' in code
    assert "notebooks/data" not in code
    manifest = _text(_cell(executed, "s06-manifest"))
    assert "True" in manifest and "False" not in manifest


def test_skipped_shots_are_printed(executed):
    assert "skipped 48224 shot_overview: PlasmaTimingError" in _text(_cell(executed, "s06-samples"))


def test_database_access_only_in_lab_mode(book, executed):
    lab = _source(_cell(book, "s06-lab-population"))
    assert 'LAB = os.environ.get("VAFT_TUTORIAL_MODE", "offline") == "lab"' in lab
    assert "summary(shot_range=" in lab.split("if LAB:", 1)[1].split("else:", 1)[0]
    others = [_strip_comments(_source(c)) for c in _code_cells(book) if c.id != "s06-lab-population"]
    assert not any("summary(shot_range=" in source for source in others)
    access = _source(_cell(book, "s06-access")).split("if LAB:", 1)[1].split("else:", 1)[0]
    assert "vaft.database.open(" in access and "vaft.database.load(" in access
    assert "Offline: the database population needs lab mode" in _text(_cell(executed, "s06-lab-population"))
    assert not re.search(r"\b(save_ods|save|upload|export_ods)\(", _executable(book).replace("export_summary(", ""))


def test_joins_are_validated_at_their_grain(book, executed):
    code = _executable(book)
    assert code.count('validate="one_to_one"') >= 3
    grain = _text(_cell(executed, "s06-load-tables"))
    assert "duplicate keys" in grain
    report = _text(_cell(executed, "s06-join"))
    assert re.search(r"matched / state only / atlas only: 150 / 0 / 0", report)
    assert re.search(r"duplicate keys after the join\s*: 0", report)
    assert "electron_kinetic" in report


def test_boundaries_stay_on_their_own_quantities(book, executed):
    code = _executable(book)
    assert '"edge_safety_factor"' not in code  # no q95 or qlim relabelled as q_psi
    assert "kink_stability_criterion(" not in code
    assert "Strait 4 l_i(1) on this l_i(3) axis: False" in _text(_cell(executed, "s06-beta-li"))
    # the state table's li is l_i(1): Strait is drawn there, and only there
    assert "'strait_1994_beta_n_li'" in _text(_cell(executed, "s06-beta-li1"))
    assert '"internal_inductance_li1": mag["li"]' in _source(_cell(book, "s06-opspace-prepare"))
    assert "applicability:" in _text(_cell(executed, "s06-qstar"))
    prepare = _source(_cell(book, "s06-opspace-prepare"))
    assert 'mag["ip_efit_A"]' in prepare and 'mag["i_p_A"]' not in prepare  # one current: the equilibrium's
    assert "factor R_geo/R0 = 0.731–1.139" in _text(_cell(executed, "s06-opspace-prepare"))
    import vaft.diagram
    for name in re.findall(r"vaft\.diagram\.(\w+)\(", code):
        assert callable(getattr(vaft.diagram, name)), name


def test_stability_layers_stay_separate(book, executed):
    join = _source(_cell(book, "s06-join"))
    for token in ("dcon_full_sign_class", "dcon_trunc_sign_class", "rdcon", "stride",
                  "_classical_tearing_drive_qa", '"not assessed"', '"unresolved"'):
        assert token in join
    assert ".astype(bool)" not in join  # an empty (failed-run) flag must not become a drive
    text = _text(_cell(executed, "s06-classify"))
    for label in ("ideal, full edge", "ideal, trunc. edge", "RDCON", "STRIDE", "Mercier", "ballooning"):
        assert label in text
    # the frozen Tier A atlas pins these per-state counts; 39915 @ 0.322 s has failed n=2 runs and
    # must stay "raw drive only", not "QA-supported drive" (which bool(NaN) once made it)
    assert "'stable': 129, 'unstable': 13, 'unresolved': 8" in text
    assert "'stable': 146, 'unstable': 3, 'unresolved': 1" in text
    assert re.search(r"RDCON  n=1,2\s*:.*'QA-supported drive': 9\b", text)
    assert re.search(r"STRIDE n=1,2\s*:.*'QA-supported drive': 38\b", text)
    markdown = _markdown(book)
    for phrase in ("not a tearing verdict", "never stable", "An empty region is not by itself an operating limit",
                   "not a VEST limit"):
        assert phrase in markdown


def test_code_stays_plain(book):
    for cell in _code_cells(book):
        tree = ast.parse(_source(cell))
        assert not any(isinstance(n, (ast.FunctionDef, ast.Lambda, ast.ClassDef)) for n in ast.walk(tree)), cell.id
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(part.startswith("_") for part in node.module.split(".")), node.module


def test_the_physics_chain_is_reproduced_by_hand(executed):
    """Part B recomputes one summary row from its integrals, written out without vaft.formula.

    The "by hand" column must not call the functions the summary uses, or the
    agreement would be automatic; the displayed table wraps, so the relative
    difference is on the last line that names the column.
    """
    source = _source(_cell(executed, "s06-physics-recompute"))
    hand = source.split("by_hand = {", 1)[1].split("}", 1)[0]
    assert "eq_formula" not in hand, hand
    text = _text(_cell(executed, "s06-physics-recompute"))
    for column in ("energy_mhd_J", "beta_tor", "beta_pol", "beta_normal", "li_3", "normalized_plasma_current"):
        line = [line for line in text.splitlines() if line.startswith(column)][-1]
        relative = float(line.split()[-1])
        assert abs(relative) < (0.02 if column == "li_3" else 1e-9), line


def test_every_limit_carries_its_source_and_schematic(book, executed):
    source = _source(_cell(book, "s06-limits-catalogue"))
    assert "fb.get_boundary(key)" in source and "boundary.sources[0]" in source
    keys = re.findall(r'\("(\w+)", "', source)
    import vaft.diagram
    from vaft.formula import boundaries as fb
    for key, schematic in re.findall(r'\("(\w+)", "[^"]+", "(\w+)"\)', source):
        assert fb.get_boundary(key).sources, key
        assert callable(getattr(vaft.diagram, schematic)), schematic
    assert {"menard_2004_qstar_min", "troyon", "sykes_2000_st_hugill", "wesson_1989_jet_li_qpsi_lower"} <= set(keys)
    limit_cells = [c for c in _code_cells(book) if c.id.startswith("s06-limits-") and c.id.endswith("-diagram")]
    assert len(limit_cells) >= 5
