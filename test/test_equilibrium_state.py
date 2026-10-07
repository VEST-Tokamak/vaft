"""The equilibrium-state table of the multi-machine operational-space notebook (#1620)."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import vaft
from vaft.diagram._op_space import get_projection, list_projections
from vaft.omas import EQUILIBRIUM_STATE_UNITS, equilibrium_state_rows, equilibrium_state_table

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NEW_NOTEBOOK = NOTEBOOKS / "multi_machine_operation_space_database.ipynb"

#: build_efit_base.py's column names for the quantities the adapter names by registry identity.
_ATLAS_NAMES = {"plasma_current": "plasma_current_ma", "minor_radius": "minor_radius_m",
                "major_radius": "major_radius_geo_m", "b0": "b0_t", "reference_major_radius": "r_reference_m"}


@pytest.fixture(scope="module")
def vest_ods():
    return vaft.omas.load(vaft.data.sample(39915, representation="imas"))


@pytest.fixture(scope="module")
def vest_rows(vest_ods):
    return equilibrium_state_rows(vest_ods, {"machine": "VEST", "machine_class": "spherical_tokamak"})


def _atlas_base():
    path = ROOT / "workflow" / "operational_space_atlas" / "build_efit_base.py"
    spec = importlib.util.spec_from_file_location("build_efit_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


def test_every_slice_has_a_row_and_a_missing_input_costs_only_its_columns(vest_rows):
    """The packaged sample's last slice has no boundary: its shape columns are NaN with the reason, not a lost row."""
    assert [r["time_index"] for r in vest_rows] == list(range(9))
    assert [r["state_status"] for r in vest_rows] == ["valid"] * 9
    assert all(r["state_notes"] == "" for r in vest_rows[:8])
    last = vest_rows[-1]
    assert "boundary.outline" in last["state_notes"] and last["update_routine_status"] == "skipped"
    assert math.isnan(last["edge_safety_factor_95"]) and math.isnan(last["internal_inductance_li3"])
    assert all(r["machine"] == "VEST" and r["machine_class"] == "spherical_tokamak" for r in vest_rows)
    assert all(r["cocos_target"] == 11 and r["cocos_source"] == 11 for r in vest_rows[:8])
    assert all("ambiguous" in r["cocos_status"] for r in vest_rows[:8])   # an ambiguity is said, not hidden


def test_vest_slices_give_the_atlas_base_table_numbers(vest_ods, vest_rows):
    """Same derivation as workflow/operational_space_atlas/build_efit_base.py, to rounding."""
    atlas = _atlas_base()
    compared = 0
    for row in vest_rows[:8]:
        reference = atlas._row(vest_ods, row["time_s"])
        assert reference["base_status"] == "valid"
        for column, value in reference.items():
            name = {v: k for k, v in _ATLAS_NAMES.items()}.get(column, column)
            if name in EQUILIBRIUM_STATE_UNITS and isinstance(value, float):
                np.testing.assert_allclose(row[name], value, rtol=1e-12, err_msg=f"{column} at {row['time_s']}")
                compared += 1
        assert row["li_beta_source"] == reference["li_beta_source"]
        assert row["li_beta_crosscheck"] == reference["li_beta_crosscheck"]
    assert compared >= 8 * 15


def test_table_units_are_the_projection_axis_units():
    """Every projection axis the table supplies is declared in the axis unit, so the renderer draws its boundary."""
    supplied = 0
    for key in list_projections():
        p = get_projection(key)
        for q in (p.x, p.y):
            if q.name in EQUILIBRIUM_STATE_UNITS:
                assert EQUILIBRIUM_STATE_UNITS[q.name] == q.unit, (key, q.name)
                supplied += 1
    assert supplied >= 18


def test_table_is_one_frame_with_units_and_provenance(vest_ods):
    table = equilibrium_state_table([(vest_ods, {"machine": "VEST", "time_indices": [0, 3]})])
    assert list(table["time_index"]) == [0, 3]
    assert "time_indices" not in table.columns
    assert table.attrs["units"] == EQUILIBRIUM_STATE_UNITS
    assert list(table.columns[:4]) == ["machine", "machine_class", "dataset_source", "dataset_type"]
    assert table["edge_safety_factor_95"].dtype.kind == "f"
    assert (table["edge_safety_factor_95"] > 0).all()   # a magnitude after COCOS 11


def test_asserted_cocos_is_recorded_and_held_to_amperes_law(vest_ods):
    (row,) = equilibrium_state_rows(vest_ods, {"machine": "VEST"}, time_indices=[0], cocos=11)
    assert row["cocos_status"] == "asserted" and row["cocos_source"] == 11 and row["state_status"] == "valid"
    (wrong,) = equilibrium_state_rows(vest_ods, {"machine": "VEST"}, time_indices=[0], cocos=1)
    assert wrong["state_status"] == "flux_conflict" and "COCOS 1 is asserted" in wrong["state_reason"]


def test_a_psi_map_that_fails_amperes_law_is_not_a_state(vest_ods, vest_rows):
    """A slice whose psi is 2 pi off its family (the TCV g-file case) is reported, with its values kept."""
    ods = copy.deepcopy(vest_ods)
    ods["equilibrium.time_slice.2.profiles_2d.0.psi"] = ods["equilibrium.time_slice.2.profiles_2d.0.psi"] * 2 * math.pi
    rows = equilibrium_state_rows(ods, {"machine": "VEST"}, time_indices=[1, 2])
    assert [r["state_status"] for r in rows] == ["valid", "flux_conflict"]
    assert "Ampere" in rows[1]["state_reason"] or "disagree" in rows[1]["state_reason"]
    assert rows[1]["internal_inductance_li3"] > 5 * vest_rows[2]["internal_inductance_li3"]


def test_a_source_without_an_equilibrium_keeps_one_failed_row():
    from omas import ODS

    (row,) = equilibrium_state_rows(ODS(), {"machine": "none"})
    assert row["state_status"] == "failed" and row["machine"] == "none"


def test_confinement_notebook_was_renamed_and_every_reference_follows():
    assert (NOTEBOOKS / "multi_machine_confinement_database.ipynb").is_file()
    assert not (NOTEBOOKS / "multi_machine_database_comparison.ipynb").exists()
    for path in (NOTEBOOKS / "README.md", ROOT / "docs" / "_guide" / "Examples.md"):
        text = path.read_text(encoding="utf-8")
        assert "multi_machine_database_comparison" not in text, path
        assert "multi_machine_confinement_database.ipynb" in text, path
        assert "multi_machine_operation_space_database.ipynb" in text, path


def test_operation_space_notebook_uses_the_comparison_notebooks_sources():
    """One source set for both public-equilibrium notebooks, without a dataset registry (#1620 B)."""
    def defaults(path):
        """The ``*_url`` strings a notebook assigns, evaluated (``FUSE + "..."`` and implicit concatenation)."""
        source = "\n".join(line for line in _code(path).splitlines() if not line.lstrip().startswith("%"))
        names, found = {}, {}

        def value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return names.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = value(node.left), value(node.right)
                return left + right if left is not None and right is not None else None
            return None

        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                names[node.targets[0].id] = value(node.value)
                if node.targets[0].id.endswith("_url"):
                    found[node.targets[0].id] = names[node.targets[0].id]
        return found

    comparison, operation = defaults(NOTEBOOKS / "multiple_tokamak_comparison.ipynb"), defaults(NEW_NOTEBOOK)
    assert all(comparison.get(n, "").startswith("https://") for n in ("d3d_url", "tcv_url", "sparc_url"))
    for name in ("d3d_url", "mastu_url", "jet_url", "tcv_url", "sparc_url", "iter_url"):
        assert operation.get(name) == comparison.get(name), name
    assert "vaft.data.sample(39915" in _code(NEW_NOTEBOOK)


def test_operation_space_notebook_discovers_projections_and_restates_no_boundary():
    code = _code(NEW_NOTEBOOK)
    assert "list_projections()" in code and "operational_space_population(" in code
    assert "equilibrium_state_table(" in code
    assert "plot_stability_limit" not in code   # the legacy workflow is history, not a source
    # no registered coefficient is restated: compare every numeric literal of the code with every coefficient
    from vaft.formula import boundaries as B

    literals = {node.value for node in ast.walk(ast.parse("\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("%"))))
        if isinstance(node, ast.Constant) and isinstance(node.value, float)}
    coefficients = {key: entry.coefficient for key in B.list_boundaries()
                    for entry in [B.get_boundary(key)] if getattr(entry, "coefficient", None) not in (None, 0.0, 1.0, 2.0)}
    restated = {key: c for key, c in coefficients.items() for v in literals if math.isclose(v, c, rel_tol=0.02)}
    assert coefficients and not restated, restated


def _stretched(x: np.ndarray, s: float) -> np.ndarray:
    """The same extent and point count as ``x``, tanh-clustered towards both ends."""
    u = np.linspace(-1.0, 1.0, x.size)
    w = np.tanh(s * u) / np.tanh(s)
    return x[0] + (w + 1.0) / 2.0 * (x[-1] - x[0])


@pytest.mark.parametrize("stretch", [0.5, 1.0, 1.5])
def test_bp2_volume_integral_does_not_assume_a_uniform_grid(vest_ods, stretch):
    # The cell volume used the first spacing everywhere, so a DD-legal non-equispaced
    # profiles_2d grid gave x0.675 / x0.235 / x0.052 of the integral and
    # internal_inductance_li3 = 0.13 instead of 0.55 when the DD routine is withheld
    # (cold review 0.8.0 delta-absorb-18 stability-opspace F4).
    from scipy.interpolate import RectBivariateSpline

    from vaft.omas.equilibrium_state import _bp2_volume_integral

    k = 0
    grid = f"equilibrium.time_slice.{k}.profiles_2d.0"
    r = np.asarray(vest_ods[f"{grid}.grid.dim1"], dtype=float)
    z = np.asarray(vest_ods[f"{grid}.grid.dim2"], dtype=float)
    psi = np.asarray(vest_ods[f"{grid}.psi"], dtype=float)
    assert np.ptp(np.diff(r)) < 1e-9 and np.ptp(np.diff(z)) < 1e-9   # the packaged sample is uniform
    reference = _bp2_volume_integral(vest_ods, k)
    atlas = _atlas_base()
    assert atlas._bp2_volume_integral(vest_ods, k) == pytest.approx(reference, rel=1e-6)

    work = copy.deepcopy(vest_ods)
    rn, zn = _stretched(r, stretch), _stretched(z, stretch)
    work[f"{grid}.grid.dim1"] = rn
    work[f"{grid}.grid.dim2"] = zn
    work[f"{grid}.psi"] = RectBivariateSpline(r, z, psi, kx=3, ky=3)(rn, zn)
    assert (rn[1] - rn[0]) < 0.9 * (r[1] - r[0])   # the first cell really is narrower
    for integral in (_bp2_volume_integral, atlas._bp2_volume_integral):
        assert integral(work, k) == pytest.approx(reference, rel=0.02), integral.__module__


def test_uniform_grid_integral_is_unchanged_by_the_per_cell_widths(vest_ods):
    # On the uniform packaged grid np.gradient of the coordinates is the constant spacing,
    # so the table numbers do not move (same finding, the "unchanged" half).
    from vaft.omas.equilibrium_state import _bp2_volume_integral

    k = 0
    grid = f"equilibrium.time_slice.{k}.profiles_2d.0"
    r = np.asarray(vest_ods[f"{grid}.grid.dim1"], dtype=float)
    z = np.asarray(vest_ods[f"{grid}.grid.dim2"], dtype=float)
    psi = np.asarray(vest_ods[f"{grid}.psi"], dtype=float)
    from matplotlib.path import Path as _Path

    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor

    ts = f"equilibrium.time_slice.{k}"
    dpsi_dr, dpsi_dz = np.gradient(psi * ods_psi_to_wb_per_radian_factor(vest_ods), r, z, edge_order=2)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    outline = np.c_[np.asarray(vest_ods[f"{ts}.boundary.outline.r"]), np.asarray(vest_ods[f"{ts}.boundary.outline.z"])]
    inside = _Path(outline).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(rr.shape)
    uniform = float(np.sum((dpsi_dr**2 + dpsi_dz**2) / rr**2 * 2.0 * np.pi * rr * (r[1] - r[0]) * (z[1] - z[0]) * inside))
    assert _bp2_volume_integral(vest_ods, k) == pytest.approx(uniform, rel=1e-6)
