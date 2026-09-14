from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import xarray as xr

from vaft.code.efit import (
    EFITConfig,
    EFITInputs,
    EFITLinearizationError,
    analyze_efit_identifiability,
    read_efit_linearization,
    resolved_efit_configuration,
    run_efit,
)
from vaft.code.efit.magnetic import _execution_kfiles, _restart_control_sha256


def _characters(values: list[str], width: int) -> np.ndarray:
    result = np.full((len(values), width), b" ", dtype="S1")
    for row, value in enumerate(values):
        encoded = value.encode("ascii")[:width]
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return result


def _parameter_digest(
    names: list[str], roles: list[str], indices: list[int], units: list[str]
) -> str:
    payload = {
        "parameters": [
            {"index": index, "name": name, "role": role, "units": unit}
            for name, role, index, unit in zip(names, roles, indices, units)
        ]
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _write_sidecar(
    path: Path,
    *,
    complete: int = 1,
    exported_first_residual: float = 2.0,
    executable_sha256: str = "a" * 64,
    main_outer_iteration: int = 5,
) -> Path:
    a = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    residual = np.asarray([2.0, -2.0, 0.0, 0.0, 0.0])
    exported_residual = residual.copy()
    exported_residual[0] = exported_first_residual
    singular = np.zeros(3)
    singular[:3] = np.linalg.svd(a, compute_uv=False)
    _, _, vh = np.linalg.svd(a, full_matrices=True)
    dataset = xr.Dataset(
        data_vars={
            "shot": np.int32(41672),
            "time_ms": np.float64(347.0),
            "jtime": np.int32(1),
            "outer_iteration": np.int32(5),
            "total_iteration": np.int32(12),
            "main_outer_iteration": np.int32(main_outer_iteration),
            "main_total_iteration": np.int32(12),
            "ivesel": np.int32(1),
            "main_nrow": np.int32(5),
            "main_ncol": np.int32(3),
            "main_nexact": np.int32(0),
            "main_status": np.int32(0),
            "main_condin": np.float64(1.0e-6),
            "main_condno": np.float64(singular[0] / singular[-1]),
            "ip_measured": np.float64(50_000.0),
            "ip_reconstructed": np.float64(50_001.0),
            "vessel_current_sum": np.float64(3.0),
            "chipasma": np.float64(9.0),
            "external_current_nrow": np.int32(0),
            "external_current_ncol": np.int32(0),
            "main_weighted_a": (("main_row", "main_col"), a),
            "main_weighted_rhs": (("main_row",), -residual),
            "main_solver_a": (("main_row", "main_col"), a),
            "main_solver_rhs": (("main_row",), -residual),
            "main_solver_solution": (("main_col",), np.zeros(3)),
            "main_physical_solution": (("main_col",), np.zeros(3)),
            "main_column_scale": (("main_col",), np.ones(3)),
            "main_singular_values": (("main_col",), singular),
            "main_right_singular_vectors": (
                ("main_col", "main_col_v"),
                vh.T,
            ),
            "main_retained_mask": (("main_col",), np.ones(3, dtype=np.int32)),
            "main_row_family_id": (
                ("main_row",),
                np.asarray([8, 8, 12, 25, 11], dtype=np.int32),
            ),
            "main_row_channel": (("main_row",), np.asarray([0, 1, 0, 0, 0])),
            "main_row_statistical": (
                ("main_row",),
                np.asarray([1, 1, 1, 0, 1], dtype=np.int32),
            ),
            "main_row_measurement": (
                ("main_row",),
                np.asarray([50_000.0, 50_000.0, 1.0, 0.0, 2.0]),
            ),
            "main_row_reconstruction": (
                ("main_row",),
                np.asarray([50_001.0, 49_999.0, 1.0, 0.0, 2.0]),
            ),
            "main_row_physical_residual": (
                ("main_row",),
                np.asarray([1.0, -1.0, 0.0, 0.0, 0.0]),
            ),
            "main_row_uncertainty": (
                ("main_row",),
                np.asarray([1.0, 1.0, 2.0, np.nan, 10.0]),
            ),
            "main_row_submitted_fwt": (
                ("main_row",),
                np.asarray([2.0, 2.0, 4.0, 1.0, 1.0]),
            ),
            "main_row_processed_weight": (
                ("main_row",),
                np.asarray([2.0, 2.0, 2.0, 1.0, 0.1]),
            ),
            "main_row_solver_weighted_residual": (("main_row",), exported_residual),
            "main_parameter_name": (
                ("main_col", "name_strlen"),
                _characters(["coil_1", "pprime_1", "ffprime_1"], 16),
            ),
            "main_parameter_role_id": (
                ("main_col",),
                np.asarray([1, 2, 3], dtype=np.int32),
            ),
            "main_parameter_index": (
                ("main_col",),
                np.asarray([1, 1, 1], dtype=np.int32),
            ),
            "main_parameter_units": (
                ("main_col", "unit_strlen"),
                _characters(["A", "Pa/Wb", "T2/Wb"], 12),
            ),
            "final_brsp": (("main_col",), np.asarray([0.1, 0.2, 0.3])),
        },
        attrs={
            "schema_id": "efit_response_diagnostics_v1",
            "schema_version": 1,
            "writer_complete": complete,
            "matrix_state": "last_accepted_linear_solve",
            "main_solver_method": "truncated_svd",
            "efit_commit": "4d10ed592f8c9d295d393d0cf331f2d8f6be3034",
            "executable_sha256": executable_sha256,
            "family_map": json.dumps(
                {8: "ip", 11: "diamagnetic_flux", 12: "pf_current", 25: "pf_relation"}
            ),
            "parameter_role_map": json.dumps(
                {1: "pf_current", 2: "pprime", 3: "ffprime"}
            ),
            "parameter_order_sha256": _parameter_digest(
                ["coil_1", "pprime_1", "ffprime_1"],
                ["pf_current", "pprime", "ffprime"],
                [1, 1, 1],
                ["A", "Pa/Wb", "T2/Wb"],
            ),
        },
    )
    dataset.to_netcdf(path, engine="scipy")
    return path


def _write_constrained_sidecar(path: Path) -> Path:
    """Write a synthetic DGGLSE-shaped fixture with non-unit scaling."""
    _write_sidecar(path)
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    column_scale = np.asarray([2.0, 0.5, 4.0])
    solver_solution = np.asarray([1.0, 2.0, 3.0])
    physical_solution = column_scale * solver_solution
    solver_a = dataset["main_weighted_a"].values * column_scale[np.newaxis, :]
    rhs = solver_a @ solver_solution
    solver_c = np.asarray([[1.0, 0.0, -1.0 / 3.0]])
    _, singular, vh = np.linalg.svd(solver_a, full_matrices=True)
    dataset["main_nexact"] = np.int32(1)
    dataset["main_exact_c"] = (
        ("main_exact", "main_col"),
        solver_c / column_scale[np.newaxis, :],
    )
    dataset["main_exact_d"] = (("main_exact",), np.asarray([0.0]))
    dataset["main_solver_exact_c"] = (("main_exact", "main_col"), solver_c)
    dataset["main_solver_exact_d"] = (("main_exact",), np.asarray([0.0]))
    dataset["main_exact_residual"] = (("main_exact",), np.zeros(1))
    dataset["main_solver_exact_residual"] = (("main_exact",), np.zeros(1))
    dataset["main_solver_a"] = (("main_row", "main_col"), solver_a)
    dataset["main_weighted_rhs"] = (("main_row",), rhs)
    dataset["main_solver_rhs"] = (("main_row",), rhs)
    dataset["main_solver_solution"] = (("main_col",), solver_solution)
    dataset["main_physical_solution"] = (("main_col",), physical_solution)
    dataset["main_column_scale"] = (("main_col",), column_scale)
    dataset["main_singular_values"] = (("main_col",), singular)
    dataset["main_condno"] = np.float64(singular[0] / singular[-1])
    dataset["main_right_singular_vectors"] = (
        ("main_col", "main_col_v"),
        vh.T,
    )
    dataset["main_row_solver_weighted_residual"] = (
        ("main_row",),
        np.zeros(solver_a.shape[0]),
    )
    dataset["main_row_reconstruction"] = dataset["main_row_measurement"]
    dataset["main_row_physical_residual"] = (
        ("main_row",),
        np.zeros(solver_a.shape[0]),
    )
    dataset.attrs["main_solver_method"] = "dgglse"
    dataset["final_brsp"] = (("main_col",), physical_solution)
    dataset.to_netcdf(path, engine="scipy", mode="w")
    return path


def _write_reduced_direction_sidecar(path: Path) -> Path:
    """Write a small retained-plus-target reduced DGGLSE fixture."""
    _write_sidecar(path)
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    weighted_a = dataset["main_weighted_a"].values * np.asarray(
        [10.0, 1.0e-8, 1.0e-9]
    )[np.newaxis, :]
    _, singular, vh = np.linalg.svd(weighted_a, full_matrices=True)
    right_vectors = vh.T
    retained = np.asarray([True, False, False])
    solver_c = right_vectors[:, 2][np.newaxis, :]
    discarded = right_vectors[:, ~retained]
    discarded_coordinates = discarded.T @ solver_c[0]
    target_basis = discarded @ discarded_coordinates / np.linalg.norm(
        discarded_coordinates
    )
    basis = np.column_stack((right_vectors[:, 0], target_basis))
    offset = np.zeros(3)
    reduced_solution = np.asarray([0.5, 0.25])
    reduced_a = weighted_a @ basis
    reduced_rhs = reduced_a @ reduced_solution
    reduced_c = solver_c @ basis
    reduced_d = reduced_c @ reduced_solution
    solver_solution = basis @ reduced_solution

    dataset["main_nexact"] = np.int32(1)
    dataset["main_exact_c"] = (("main_exact", "main_col"), solver_c)
    dataset["main_exact_d"] = (("main_exact",), reduced_d)
    dataset["main_solver_exact_c"] = (("main_exact", "main_col"), solver_c)
    dataset["main_solver_exact_d"] = (("main_exact",), reduced_d)
    dataset["main_exact_residual"] = (("main_exact",), np.zeros(1))
    dataset["main_solver_exact_residual"] = (("main_exact",), np.zeros(1))
    dataset["main_weighted_a"] = (("main_row", "main_col"), weighted_a)
    dataset["main_solver_a"] = (("main_row", "main_col"), weighted_a)
    dataset["main_weighted_rhs"] = (("main_row",), reduced_rhs)
    dataset["main_solver_rhs"] = (("main_row",), reduced_rhs)
    dataset["main_solver_solution"] = (("main_col",), solver_solution)
    dataset["main_physical_solution"] = (("main_col",), solver_solution)
    dataset["main_column_scale"] = (("main_col",), np.ones(3))
    dataset["main_singular_values"] = (("main_col",), singular)
    dataset["main_condno"] = np.float64(singular[0] / singular[-1])
    dataset["main_right_singular_vectors"] = (
        ("main_col", "main_col_v"),
        right_vectors,
    )
    dataset["main_retained_mask"] = (("main_col",), retained.astype(np.int32))
    dataset["main_row_solver_weighted_residual"] = (
        ("main_row",),
        np.zeros(weighted_a.shape[0]),
    )
    dataset["main_row_reconstruction"] = dataset["main_row_measurement"]
    dataset["main_row_physical_residual"] = (
        ("main_row",),
        np.zeros(weighted_a.shape[0]),
    )
    dataset["final_brsp"] = (("main_col",), solver_solution)
    dataset["main_reduced_ncol"] = np.int32(2)
    dataset["main_reduced_status"] = np.int32(0)
    dataset["main_reduced_condno"] = np.float64(0.0)
    dataset["main_direction_discarded_projection_norm"] = np.float64(
        np.linalg.norm(discarded_coordinates)
    )
    dataset["main_reduced_solver_a"] = (
        ("main_row", "main_reduced_col"),
        reduced_a,
    )
    dataset["main_reduced_solver_rhs"] = (("main_row",), reduced_rhs)
    dataset["main_reduced_exact_c"] = (
        ("main_exact", "main_reduced_col"),
        reduced_c,
    )
    dataset["main_reduced_exact_d"] = (("main_exact",), reduced_d)
    dataset["main_reduced_solver_basis"] = (
        ("main_col", "main_reduced_col"),
        basis,
    )
    dataset["main_reduced_physical_from_solver"] = (
        ("main_col", "main_reduced_col"),
        basis,
    )
    dataset["main_reduced_solver_offset"] = (("main_col",), offset)
    dataset["main_reduced_physical_offset"] = (("main_col",), offset)
    dataset["main_reduced_solver_solution"] = (
        ("main_reduced_col",),
        reduced_solution,
    )
    dataset["main_reduced_basis_kind"] = (
        ("main_reduced_col",),
        np.asarray([1, 2], dtype=np.int32),
    )
    dataset.attrs["main_solver_method"] = (
        "truncated_svd_reduced_dgglse_exact_direction"
    )
    dataset.to_netcdf(path, engine="scipy", mode="w")
    return path


def test_native_sidecar_parser_preserves_solver_and_diagnostic_semantics(tmp_path):
    problem = read_efit_linearization(_write_sidecar(tmp_path / "response.nc"))

    assert problem.shot == 41672
    assert problem.time_seconds == pytest.approx(0.347)
    assert problem.external_current is None
    assert problem.main.row_family == (
        "plasma_current",
        "plasma_current",
        "pf_current",
        "pf_relation",
        "diamagnetic_flux",
    )
    assert problem.main.row_kind[3] == "soft_structural_relation"
    assert problem.main.parameter_name == ("coil_1", "pprime_1", "ffprime_1")
    np.testing.assert_allclose(problem.main.solver_residual, [2.0, -2.0, 0.0, 0.0, 0.0])
    assert problem.final_brsp.tolist() == pytest.approx([0.1, 0.2, 0.3])

    report = analyze_efit_identifiability(problem)
    np.testing.assert_allclose(
        report.gauss_newton_curvature,
        problem.main.weighted_a.T @ problem.main.weighted_a,
    )
    ip_row = report.row_audit[0]
    assert ip_row.diagnostic_chi2 == pytest.approx(1.0)
    assert ip_row.solver_objective == pytest.approx(4.0)
    assert report.ip_accounting is not None
    assert report.ip_accounting.predicted_vcurrt_chi2 == pytest.approx(9.0)
    assert report.ip_accounting.reported_chipasma == pytest.approx(9.0)
    assert report.families["diamagnetic_flux"].rank_gain == (1, 1, 1)
    assert "independent_information" in report.families[
        "diamagnetic_flux"
    ].classifications
    assert report.maximum_family_share_sum_error < 1.0e-12
    np.testing.assert_allclose(
        np.linalg.norm(problem.main.weighted_a * report.parameter_scale, axis=0),
        np.ones(3),
    )
    json.dumps(report.to_dict(), allow_nan=False)


def test_true_equalities_are_projected_before_svd(tmp_path):
    problem = read_efit_linearization(_write_sidecar(tmp_path / "response.nc"))
    constrained = replace(
        problem.main,
        solver_exact_c=np.asarray([[0.0, 0.0, 1.0]]),
        solver_exact_d=np.asarray([0.0]),
        exact_c=np.asarray([[0.0, 0.0, 1.0]]),
        exact_d=np.asarray([0.0]),
    )

    report = analyze_efit_identifiability(constrained)

    assert report.exact_constraint_rank == 1
    assert report.null_space_basis.shape == (3, 2)
    assert report.singular_values.shape == (2,)
    assert report.nullity == 0
    assert report.families["diamagnetic_flux"].rank_gain == (0, 0, 0)


def test_native_dgglse_constraint_is_converted_to_physical_coordinates(tmp_path):
    problem = read_efit_linearization(
        _write_constrained_sidecar(tmp_path / "constrained.nc")
    )
    block = problem.main

    np.testing.assert_allclose(
        block.solver_exact_c,
        [[1.0, 0.0, -1.0 / 3.0]],
    )
    np.testing.assert_allclose(
        block.exact_c,
        block.solver_exact_c / block.column_scale,
    )
    np.testing.assert_allclose(
        block.solver_exact_c @ block.solver_solution, block.solver_exact_d
    )
    np.testing.assert_allclose(block.exact_c @ block.physical_solution, block.exact_d)
    assert block.validation.exact_constraint_residual_norm == pytest.approx(0.0)
    assert block.validation.singular_value_relative_error < 1.0e-10
    assert block.validation.solver_reproduction_relative_error < 1.0e-8
    assert analyze_efit_identifiability(problem).exact_constraint_rank == 1


def test_exact_but_nonoptimal_dgglse_solution_is_rejected(tmp_path):
    path = _write_constrained_sidecar(tmp_path / "nonoptimal.nc")
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    solver_solution = dataset["main_solver_solution"].values.copy()
    # [1, 0, 3] is in the null space of the fixture's exact row, so this
    # perturbation stays exactly feasible while ceasing to minimize ||Ax-b||.
    solver_solution += np.asarray([1.0, 0.0, 3.0])
    column_scale = dataset["main_column_scale"].values
    physical_solution = column_scale * solver_solution
    solver_a = dataset["main_solver_a"].values
    solver_rhs = dataset["main_solver_rhs"].values
    dataset["main_solver_solution"] = (("main_col",), solver_solution)
    dataset["main_physical_solution"] = (("main_col",), physical_solution)
    dataset["final_brsp"] = (("main_col",), physical_solution)
    dataset["main_row_solver_weighted_residual"] = (
        ("main_row",),
        solver_a @ solver_solution - solver_rhs,
    )
    dataset.to_netcdf(path, engine="scipy", mode="w")

    with pytest.raises(EFITLinearizationError, match="DGGLSE solution cannot be reproduced"):
        read_efit_linearization(path)


def test_reduced_direction_dgglse_is_independently_reproduced(tmp_path):
    problem = read_efit_linearization(
        _write_reduced_direction_sidecar(tmp_path / "reduced.nc")
    )
    block = problem.main
    reduced = block.reduced_solve

    assert block.solver_method == "truncated_svd_reduced_dgglse_exact_direction"
    assert block.nexact == 1
    assert reduced is not None
    assert reduced.solver_a.shape == (5, 2)
    assert reduced.basis_kind.tolist() == [1, 2]
    assert reduced.transform_relative_error < 1.0e-10
    assert reduced.reproduction_relative_error < 1.0e-8
    assert block.validation.solver_reproduction_relative_error < 1.0e-8


def test_incomplete_or_inconsistent_sidecars_fail_loudly(tmp_path):
    with pytest.raises(EFITLinearizationError, match="incomplete"):
        read_efit_linearization(
            _write_sidecar(tmp_path / "incomplete.nc", complete=0)
        )

    path = _write_sidecar(
        tmp_path / "bad_residual.nc", exported_first_residual=3.0
    )
    with pytest.raises(EFITLinearizationError, match="residuals disagree"):
        read_efit_linearization(path)

    with pytest.raises(EFITLinearizationError, match="executable SHA-256"):
        read_efit_linearization(
            _write_sidecar(tmp_path / "missing_identity.nc", executable_sha256="")
        )

    short_revision = _write_sidecar(tmp_path / "short_revision.nc")
    with xr.open_dataset(short_revision, decode_cf=False) as opened:
        dataset = opened.load()
    dataset.attrs["efit_commit"] = "4d10ed5"
    dataset.to_netcdf(short_revision, engine="scipy", mode="w")
    with pytest.raises(EFITLinearizationError, match="full EFIT source revision"):
        read_efit_linearization(short_revision)

    with pytest.raises(EFITLinearizationError, match="iteration metadata disagree"):
        read_efit_linearization(
            _write_sidecar(tmp_path / "wrong_iteration.nc", main_outer_iteration=4)
        )


def test_native_retained_mask_must_match_condin_cutoff(tmp_path):
    path = _write_sidecar(tmp_path / "wrong_mask.nc")
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    dataset["main_retained_mask"] = (
        ("main_col",),
        np.asarray([1, 1, 0], dtype=np.int32),
    )
    dataset.to_netcdf(path, engine="scipy", mode="w")

    with pytest.raises(EFITLinearizationError, match=r"condin\*smax"):
        read_efit_linearization(path)


@pytest.mark.parametrize("method", (None, "", "black_box_solver", "dgglse"))
def test_unconstrained_block_requires_recognized_truncated_svd_method(
    tmp_path, method
):
    path = _write_sidecar(tmp_path / "wrong_solver_method.nc")
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    if method is None:
        dataset.attrs.pop("main_solver_method")
    else:
        dataset.attrs["main_solver_method"] = method
    dataset.to_netcdf(path, engine="scipy", mode="w")

    with pytest.raises(EFITLinearizationError, match="solver_method|truncated-SVD"):
        read_efit_linearization(path)


@pytest.mark.parametrize(
    "variable",
    (
        "main_row_measurement",
        "main_row_reconstruction",
        "main_row_physical_residual",
        "main_row_uncertainty",
        "main_row_submitted_fwt",
        "main_row_processed_weight",
        "main_row_solver_weighted_residual",
        "main_parameter_name",
        "main_parameter_units",
        "main_parameter_index",
        "main_nexact",
        "writer_complete",
        "final_brsp",
        "jtime",
        "ivesel",
        "ip_measured",
        "ip_reconstructed",
        "vessel_current_sum",
        "chipasma",
    ),
)
def test_v1_required_diagnostics_cannot_be_omitted(tmp_path, variable):
    path = _write_sidecar(tmp_path / f"missing_{variable}.nc")
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    if variable == "writer_complete":
        dataset.attrs.pop(variable)
    else:
        dataset = dataset.drop_vars(variable)
    dataset.to_netcdf(path, engine="scipy", mode="w")

    with pytest.raises(EFITLinearizationError, match="missing|required|incomplete"):
        read_efit_linearization(path)


def test_v1_row_finiteness_rules_distinguish_soft_rows(tmp_path):
    statistical_path = _write_sidecar(tmp_path / "bad_statistical_uncertainty.nc")
    with xr.open_dataset(statistical_path, decode_cf=False) as opened:
        dataset = opened.load()
    uncertainty = dataset["main_row_uncertainty"].values.copy()
    uncertainty[0] = np.nan
    dataset["main_row_uncertainty"] = (("main_row",), uncertainty)
    dataset.to_netcdf(statistical_path, engine="scipy", mode="w")
    with pytest.raises(EFITLinearizationError, match="statistical rows"):
        read_efit_linearization(statistical_path)

    structural_path = _write_sidecar(tmp_path / "invented_structural_uncertainty.nc")
    with xr.open_dataset(structural_path, decode_cf=False) as opened:
        dataset = opened.load()
    uncertainty = dataset["main_row_uncertainty"].values.copy()
    uncertainty[3] = 1.0
    dataset["main_row_uncertainty"] = (("main_row",), uncertainty)
    dataset.to_netcdf(structural_path, engine="scipy", mode="w")
    with pytest.raises(EFITLinearizationError, match="must not invent"):
        read_efit_linearization(structural_path)


@pytest.mark.parametrize(
    "variable",
    (
        "main_exact_c",
        "main_exact_d",
        "main_solver_exact_c",
        "main_solver_exact_d",
        "main_exact_residual",
        "main_solver_exact_residual",
    ),
)
def test_true_equality_arrays_are_explicitly_required(tmp_path, variable):
    path = _write_constrained_sidecar(tmp_path / f"missing_{variable}.nc")
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load().drop_vars(variable)
    dataset.to_netcdf(path, engine="scipy", mode="w")

    with pytest.raises(EFITLinearizationError, match="missing required variable"):
        read_efit_linearization(path)


def test_parameter_order_digest_and_role_indices_are_validated(tmp_path):
    digest_path = _write_sidecar(tmp_path / "wrong_parameter_digest.nc")
    with xr.open_dataset(digest_path, decode_cf=False) as opened:
        dataset = opened.load()
    dataset.attrs["parameter_order_sha256"] = "d" * 64
    dataset.to_netcdf(digest_path, engine="scipy", mode="w")
    with pytest.raises(EFITLinearizationError, match="parameter-order SHA-256 disagrees"):
        read_efit_linearization(digest_path)

    index_path = _write_sidecar(tmp_path / "noncontiguous_role_index.nc")
    with xr.open_dataset(index_path, decode_cf=False) as opened:
        dataset = opened.load()
    dataset["main_parameter_index"] = (
        ("main_col",),
        np.asarray([2, 1, 1], dtype=np.int32),
    )
    dataset.attrs["parameter_order_sha256"] = _parameter_digest(
        ["coil_1", "pprime_1", "ffprime_1"],
        ["pf_current", "pprime", "ffprime"],
        [2, 1, 1],
        ["A", "Pa/Wb", "T2/Wb"],
    )
    dataset.to_netcdf(index_path, engine="scipy", mode="w")
    with pytest.raises(EFITLinearizationError, match="contiguous from 1"):
        read_efit_linearization(index_path)


def test_row_family_channel_pairs_are_unique(tmp_path):
    path = _write_sidecar(tmp_path / "duplicate_row_channel.nc")
    with xr.open_dataset(path, decode_cf=False) as opened:
        dataset = opened.load()
    dataset["main_row_channel"] = (
        ("main_row",),
        np.asarray([0, 0, 0, 0, 0], dtype=np.int32),
    )
    dataset.to_netcdf(path, engine="scipy", mode="w")

    with pytest.raises(EFITLinearizationError, match="family/channel pairs"):
        read_efit_linearization(path)


def test_execution_controls_are_not_part_of_the_scientific_hash(tmp_path):
    restart = tmp_path / "source-esave.dat"
    direction = tmp_path / "direction.nc"
    restart.write_bytes(b"restart")
    direction.write_bytes(b"direction")
    baseline = EFITConfig()
    controlled = EFITConfig(
        export_linearization=True,
        direction_constraint=direction,
        restart_from=restart,
        write_restart=True,
    )

    assert baseline.scientific_config().sha256 == controlled.scientific_config().sha256
    execution = resolved_efit_configuration(controlled)["execution"]
    assert execution["export_linearization"] is True
    assert len(execution["direction_constraint_sha256"]) == 64
    assert len(execution["restart_from_sha256"]) == 64


def test_run_stages_restart_and_patches_only_execution_kfile(tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    source_dir = tmp_path / "source"
    workdir.mkdir()
    source_dir.mkdir()
    kfile = source_dir / "k041672.00347"
    original = " &IN1\n IOUT=4\n ICINIT = 2\n /\n MAG\n"
    kfile.write_text(original, encoding="utf-8")
    restart = source_dir / "esave.dat"
    restart.write_bytes(b"restart-input")
    direction = source_dir / "direction.nc"
    xr.Dataset(
        attrs={
            "schema_id": "efit_direction_constraint_v1",
            "schema_version": 1,
            "writer_complete": 1,
            "source_sidecar_sha256": "b" * 64,
            "parameter_order_sha256": "c" * 64,
        }
    ).to_netcdf(direction, engine="scipy")
    executable = source_dir / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        (Path(kwargs["cwd"]) / "esave.dat").write_bytes(b"restart-output")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = EFITConfig(
        executable=str(executable),
        workdir=workdir,
        shot=41672,
        stack_size_kb=None,
        export_linearization=True,
        direction_constraint=direction,
        restart_from=restart,
        write_restart=True,
    )

    result = run_efit(EFITInputs(workdir=workdir, kfiles=(kfile,)), config)

    staged_name = str(captured["input"]).splitlines()[2]
    staged = workdir / staged_name
    staged_text = staged.read_text(encoding="utf-8")
    assert "IOUT=20" in staged_text
    assert "ICINIT = -2" in staged_text
    assert kfile.read_text(encoding="utf-8") == original
    assert (workdir / "esave.dat").read_bytes() == b"restart-output"
    environment = captured["env"]
    assert environment["EFIT_DIRECTION_CONSTRAINT_FILE"] == str(direction.resolve())
    assert environment["EFIT_DIRECTION_SOURCE_SIDECAR_SHA256"] == "b" * 64
    assert environment["EFIT_EXECUTABLE_SHA256"] == __import__("hashlib").sha256(
        executable.read_bytes()
    ).hexdigest()
    restart_control = _restart_control_sha256((kfile,), direction)
    assert environment["EFIT_RESTART_CONTROL_SHA256"] == restart_control
    assert (
        result.configuration["execution"]["restart_control_sha256"]
        == restart_control
    )
    diagnostics = Path(environment["EFIT_RESPONSE_DIAGNOSTICS_DIR"])
    assert diagnostics.is_dir() and diagnostics.parent == workdir / "response_diagnostics"
    assert result.diagnostic_errors == (
        "requested EFIT response diagnostics were not produced",
    )
    assert result.restart_file == workdir / "esave.dat"


def test_run_refuses_an_unwritten_stale_restart_output(tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    workdir.mkdir()
    kfile = tmp_path / "k041672.00331"
    kfile.write_text(" &IN1\n IOUT=4\n ICINIT=2\n /\n", encoding="utf-8")
    restart = tmp_path / "input-esave.dat"
    restart.write_bytes(b"restart-input")
    executable = tmp_path / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    result = run_efit(
        EFITInputs(workdir=workdir, kfiles=(kfile,)),
        EFITConfig(
            executable=str(executable),
            workdir=workdir,
            stack_size_kb=None,
            restart_from=restart,
            write_restart=True,
        ),
    )

    assert result.restart_file is None
    assert any("stale esave.dat" in message for message in result.diagnostic_errors)


def test_run_ignores_unmodified_stale_equilibrium_outputs(tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    workdir.mkdir()
    kfile = tmp_path / "k041672.00331"
    kfile.write_text(" &IN1\n IOUT=4\n ICINIT=2\n /\n", encoding="utf-8")
    for prefix in ("g", "a", "m"):
        (workdir / f"{prefix}041672.00331").write_bytes(b"stale")
    executable = tmp_path / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    result = run_efit(
        EFITInputs(workdir=workdir, kfiles=(kfile,)),
        EFITConfig(
            executable=str(executable),
            workdir=workdir,
            shot=41672,
            times=(0.331,),
            stack_size_kb=None,
        ),
    )

    assert result.gfiles == result.afiles == result.mfiles == ()
    assert sum("stale EFIT" in message for message in result.diagnostic_errors) == 3


def test_restart_control_identity_uses_unstaged_kfile_and_direction_bytes(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    kfile = tmp_path / "k041672.00331"
    kfile.write_text(" &IN1\n IOUT=4\n ICINIT=2\n FWTDLC=1\n /\n", encoding="utf-8")
    direction = tmp_path / "direction.nc"
    direction.write_bytes(b"direction-a")
    original = _restart_control_sha256((kfile,), direction)

    staged = _execution_kfiles(
        EFITInputs(workdir=workdir, kfiles=(kfile,)),
        EFITConfig(
            workdir=workdir,
            restart_from=tmp_path / "unused-esave.dat",
            write_restart=True,
        ),
    )
    assert "IOUT=20" in staged[0].read_text(encoding="utf-8")
    assert "ICINIT=-2" in staged[0].read_text(encoding="utf-8")
    # Staging does not mutate the source or the already-computed identity.
    assert _restart_control_sha256((kfile,), direction) == original

    kfile.write_text(" &IN1\n IOUT=4\n ICINIT=2\n FWTDLC=10\n /\n", encoding="utf-8")
    assert _restart_control_sha256((kfile,), direction) != original
    kfile.write_text(" &IN1\n IOUT=4\n ICINIT=2\n FWTDLC=1\n /\n", encoding="utf-8")
    direction.write_bytes(b"direction-b")
    assert _restart_control_sha256((kfile,), direction) != original
    assert _restart_control_sha256((kfile,), None) != original


def test_export_only_stages_long_external_kfile_path(tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    source_dir = tmp_path / ("long-source-directory-" * 8)
    workdir.mkdir()
    source_dir.mkdir()
    kfile = source_dir / "k041672.00331"
    original = " &IN1\n IOUT=4\n ICINIT=2\n /\n MAG\n"
    kfile.write_text(original, encoding="utf-8")
    executable = tmp_path / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_efit(
        EFITInputs(workdir=workdir, kfiles=(kfile,)),
        EFITConfig(
            executable=str(executable),
            workdir=workdir,
            stack_size_kb=None,
            export_linearization=True,
        ),
    )

    staged_name = str(captured["input"]).splitlines()[2]
    staged = workdir / staged_name
    assert staged.parent.parent == workdir / ".vaft_efit_execution"
    assert staged.read_text(encoding="utf-8") == original
    assert kfile.read_text(encoding="utf-8") == original


def test_direction_and_restart_controls_require_one_kfile(tmp_path):
    executable = tmp_path / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    first = tmp_path / "k01.00001"
    second = tmp_path / "k01.00002"
    first.write_text(" &IN1\n /\n", encoding="utf-8")
    second.write_text(" &IN1\n /\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        run_efit(
            EFITInputs(tmp_path, kfiles=(first, second)),
            EFITConfig(
                executable=str(executable),
                workdir=tmp_path,
                stack_size_kb=None,
                write_restart=True,
            ),
        )


def test_missing_direction_is_never_silently_ignored(tmp_path):
    executable = tmp_path / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    kfile = tmp_path / "k01.00001"
    kfile.write_text(" &IN1\n /\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="direction constraint"):
        run_efit(
            EFITInputs(tmp_path, kfiles=(kfile,)),
            EFITConfig(
                executable=str(executable),
                workdir=tmp_path,
                stack_size_kb=None,
                direction_constraint=tmp_path / "missing.nc",
            ),
        )


def test_invalid_direction_schema_is_never_passed_to_efit(tmp_path):
    executable = tmp_path / "efit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    kfile = tmp_path / "k01.00001"
    kfile.write_text(" &IN1\n /\n", encoding="utf-8")
    direction = tmp_path / "direction.nc"
    xr.Dataset(attrs={"schema_id": "wrong", "schema_version": 1}).to_netcdf(
        direction, engine="scipy"
    )

    with pytest.raises(ValueError, match="unsupported EFIT direction"):
        run_efit(
            EFITInputs(tmp_path, kfiles=(kfile,)),
            EFITConfig(
                executable=str(executable),
                workdir=tmp_path,
                stack_size_kb=None,
                direction_constraint=direction,
            ),
        )
