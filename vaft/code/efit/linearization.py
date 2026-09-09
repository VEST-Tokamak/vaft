"""Read and analyse EFIT's opt-in response-matrix diagnostics.

The sidecar is intentionally a solver diagnostic, not another equilibrium
format.  In particular, its matrix and solution describe the last *accepted*
linear solve.  ``final_brsp`` is recorded separately because EFIT may
renormalise profile coefficients or replace external currents afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_ID = "efit_response_diagnostics_v1"
SCHEMA_VERSION = 1
SOLVER_METHODS = frozenset(
    {
        "truncated_svd",
        "dgglse",
        "truncated_svd_reduced_dgglse_exact_direction",
    }
)

FAMILY_IDS: Mapping[int, str] = MappingProxyType(
    {
        1: "flux_loop",
        2: "bpol_probe",
        3: "mse",
        4: "mse_line_split_b",
        5: "mse_line_split_e",
        6: "ece",
        7: "ece_bz",
        8: "plasma_current",
        9: "q_axis",
        10: "beta_p",
        11: "diamagnetic_flux",
        12: "pf_current",
        13: "pressure",
        14: "pressure_axis",
        15: "rotational_pressure",
        16: "current_density",
        17: "pprime_relation",
        18: "ffprime_relation",
        19: "rotation_relation",
        20: "boundary",
        21: "e_current",
        22: "reference_flux",
        23: "pf_current_sum",
        24: "vertical_stabilization",
        25: "pf_relation",
        26: "sol",
        99: "other",
    }
)

PARAMETER_ROLE_IDS: Mapping[int, str] = MappingProxyType(
    {
        1: "pf_current",
        2: "pprime",
        3: "ffprime",
        4: "rotation",
        5: "vessel_current",
        6: "pressure_boundary",
        7: "vertical_shift",
        8: "e_current",
        9: "reference_flux",
        10: "electric_field",
        11: "pprime_edge",
        12: "ffprime_edge",
        99: "other",
    }
)


class EFITLinearizationError(ValueError):
    """Raised when a response sidecar is incomplete or internally invalid."""


@dataclass(frozen=True)
class LinearSolveValidation:
    """Numerical reproduction checks performed while reading a sidecar."""

    column_scaling_relative_error: float
    solution_scaling_relative_error: float
    singular_value_relative_error: float | None
    solver_reproduction_relative_error: float | None
    exact_constraint_residual_norm: float | None
    exact_constraint_tolerance: float | None


@dataclass(frozen=True)
class EFITReducedLinearSolve:
    """Actual reduced-coordinate DGGLSE problem used for a direction row."""

    solver_a: np.ndarray
    solver_rhs: np.ndarray
    exact_c: np.ndarray
    exact_d: np.ndarray
    solver_basis: np.ndarray
    physical_from_solver: np.ndarray
    solver_offset: np.ndarray
    physical_offset: np.ndarray
    solver_solution: np.ndarray
    basis_kind: np.ndarray
    status: int
    condno: float
    discarded_projection_norm: float
    transform_relative_error: float
    reproduction_relative_error: float


@dataclass(frozen=True)
class EFITLinearizationBlock:
    """One response solve exported by EFIT in both physical and solver units."""

    name: str
    weighted_a: np.ndarray
    weighted_rhs: np.ndarray
    solver_a: np.ndarray
    solver_rhs: np.ndarray
    solver_exact_c: np.ndarray
    solver_exact_d: np.ndarray
    exact_c: np.ndarray
    exact_d: np.ndarray
    solver_solution: np.ndarray
    physical_solution: np.ndarray
    column_scale: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    retained_mask: np.ndarray
    row_family_id: np.ndarray
    row_family: tuple[str, ...]
    row_kind: tuple[str, ...]
    row_channel: np.ndarray
    row_statistical: np.ndarray
    row_measurement: np.ndarray
    row_reconstruction: np.ndarray
    row_physical_residual: np.ndarray
    row_uncertainty: np.ndarray
    row_submitted_fwt: np.ndarray
    row_processed_weight: np.ndarray
    row_solver_weighted_residual: np.ndarray
    parameter_name: tuple[str, ...]
    parameter_role_id: np.ndarray
    parameter_role: tuple[str, ...]
    parameter_index: np.ndarray
    parameter_units: tuple[str, ...]
    validation: LinearSolveValidation
    solver_method: str = ""
    reduced_solve: EFITReducedLinearSolve | None = None
    outer_iteration: int | None = None
    total_iteration: int | None = None
    status: int | None = None
    condin: float | None = None
    condno: float | None = None

    @property
    def nrow(self) -> int:
        return int(self.weighted_a.shape[0])

    @property
    def ncol(self) -> int:
        return int(self.weighted_a.shape[1])

    @property
    def nexact(self) -> int:
        return int(self.exact_c.shape[0])

    @property
    def solver_residual(self) -> np.ndarray:
        """Residual paired with the exported linear solve, never final BRSP."""
        return self.solver_a @ self.solver_solution - self.solver_rhs


@dataclass(frozen=True)
class EFITLinearization:
    """Validated contents of one ``efit_response_diagnostics_v1`` sidecar."""

    path: Path
    sha256: str
    schema_id: str
    schema_version: int
    source_revision: str
    executable_sha256: str
    matrix_state: str
    shot: int
    time_ms: float
    jtime: int | None
    outer_iteration: int | None
    total_iteration: int | None
    ivesel: int | None
    ip_measured: float | None
    ip_reconstructed: float | None
    vessel_current_sum: float | None
    chipasma: float | None
    main: EFITLinearizationBlock
    external_current: EFITLinearizationBlock | None
    final_brsp: np.ndarray
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def time_seconds(self) -> float:
        return float(self.time_ms) / 1000.0

    def block(self, name: str = "main") -> EFITLinearizationBlock:
        if name == "main":
            return self.main
        if name == "external_current" and self.external_current is not None:
            return self.external_current
        raise KeyError(f"linearization block is unavailable: {name}")


@dataclass(frozen=True)
class IdentifiabilityConfig:
    """Numerical policy for conditional information classification."""

    block: str = "main"
    condin: float | None = None
    cutoff_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0)
    resolved_multiple: float = 10.0
    unresolved_multiple: float = 0.1
    independent_norm_fraction: float = 0.1
    reinforcing_weak_lift: float = 10.0
    reinforcing_trace_reduction: float = 0.2
    overwhelmed_objective_share: float = 0.01
    core_families: tuple[str, ...] = (
        "plasma_current",
        "pf_current",
        "pf_relation",
    )
    nonlinear_material_response: frozenset[str] = frozenset()
    structural_effects: frozenset[str] = frozenset()
    accounting_confounded: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.block not in {"main", "external_current"}:
            raise ValueError("block must be 'main' or 'external_current'")
        if self.condin is not None and (
            not math.isfinite(self.condin) or self.condin <= 0
        ):
            raise ValueError("condin must be finite and greater than zero")
        if not self.cutoff_multipliers or any(
            not math.isfinite(value) or value <= 0
            for value in self.cutoff_multipliers
        ):
            raise ValueError("cutoff_multipliers must be finite and positive")
        for name in (
            "resolved_multiple",
            "unresolved_multiple",
            "independent_norm_fraction",
            "reinforcing_weak_lift",
            "reinforcing_trace_reduction",
            "overwhelmed_objective_share",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "nonlinear_material_response",
            "structural_effects",
            "accounting_confounded",
        ):
            object.__setattr__(self, name, frozenset(getattr(self, name)))


@dataclass(frozen=True)
class RowInformation:
    index: int
    family: str
    kind: str
    channel: int
    statistical: bool
    physical_residual: float | None
    uncertainty: float | None
    submitted_fwt: float | None
    processed_weight: float | None
    diagnostic_chi2: float | None
    solver_objective: float


@dataclass(frozen=True)
class ModeInformation:
    index: int
    singular_value: float
    state: str
    free_direction: np.ndarray
    scaled_parameter_direction: np.ndarray
    physical_parameter_direction: np.ndarray
    family_shares: Mapping[str, float]
    parameter_role_shares: Mapping[str, float]


@dataclass(frozen=True)
class SubsetInformation:
    name: str
    families: tuple[str, ...]
    row_count: int
    singular_values: np.ndarray
    ranks: tuple[int, ...]
    nullity: int
    curvature_inverse_trace: float
    maximum_principal_angle_degrees: float | None


@dataclass(frozen=True)
class FamilyInformation:
    family: str
    row_count: int
    diagnostic_chi2: float | None
    solver_objective: float
    solver_objective_share: float | None
    rank_gain: tuple[int, ...]
    weak_subspace_fraction: float
    maximum_weak_singular_lift: float
    curvature_inverse_trace_reduction: float
    classifications: tuple[str, ...]


@dataclass(frozen=True)
class IpAccounting:
    measured: float | None
    reconstructed: float | None
    solve_residual: float | None
    vessel_current_sum: float | None
    predicted_vcurrt_chi2: float | None
    reported_chipasma: float | None


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Local response information with sensitivity to EFIT's rank cutoff."""

    block: str
    condin: float
    cutoff_multipliers: tuple[float, ...]
    cutoffs: tuple[float, ...]
    nominal_cutoff: float
    parameter_scale: np.ndarray
    null_space_basis: np.ndarray
    gauss_newton_curvature: np.ndarray
    scaled_projected_curvature: np.ndarray
    exact_constraint_rank: int
    singular_values: np.ndarray
    ranks: tuple[int, ...]
    nullity: int
    modes: tuple[ModeInformation, ...]
    maximum_family_share_sum_error: float
    families: Mapping[str, FamilyInformation]
    subsets: Mapping[str, SubsetInformation]
    row_audit: tuple[RowInformation, ...]
    ip_accounting: IpAccounting | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for workflow artifacts."""
        return _json_compatible(self)


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_compatible(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar(dataset: Any, name: str, default: Any = None) -> Any:
    if name in dataset.variables:
        values = np.asarray(dataset[name].values)
        if values.size != 1:
            raise EFITLinearizationError(f"{name} must be scalar")
        return _attribute(values.reshape(-1)[0])
    return _attribute(dataset.attrs.get(name, default))


def _array(
    dataset: Any,
    name: str,
    *,
    dtype: Any = float,
    default: Any = None,
) -> np.ndarray:
    if name not in dataset.variables:
        if default is None:
            raise EFITLinearizationError(f"missing required variable {name}")
        return np.asarray(default, dtype=dtype)
    return np.asarray(dataset[name].values, dtype=dtype)


def _strings(dataset: Any, name: str, count: int) -> tuple[str, ...]:
    if name not in dataset.variables:
        raise EFITLinearizationError(f"missing required variable {name}")
    values = np.asarray(dataset[name].values)
    if values.ndim == 0:
        decoded = [str(_attribute(values.item()))]
    elif values.dtype.kind in {"S", "U"} and values.ndim >= 2:
        decoded = []
        for row in values.reshape(values.shape[0], -1):
            pieces = [
                item.decode("utf-8", errors="replace")
                if isinstance(item, (bytes, np.bytes_))
                else str(item)
                for item in row
            ]
            decoded.append("".join(pieces).split("\x00", 1)[0].rstrip())
    else:
        decoded = [str(_attribute(item)).rstrip("\x00 ") for item in values.reshape(-1)]
    if len(decoded) != count:
        raise EFITLinearizationError(
            f"{name} has {len(decoded)} strings, expected {count}"
        )
    if any(not value for value in decoded):
        raise EFITLinearizationError(f"{name} contains an empty string")
    return tuple(decoded)


def _id_map(raw: Any, fallback: Mapping[int, str]) -> Mapping[int, str]:
    raw = _attribute(raw)
    if raw in (None, ""):
        return fallback
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            return {int(key): str(value) for key, value in payload.items()}
        except (ValueError, TypeError, AttributeError):
            result: dict[int, str] = {}
            for item in raw.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    result[int(key.strip())] = value.strip()
            if result:
                return result
    raise EFITLinearizationError("invalid integer-to-name map attribute")


def _mapped_names(ids: np.ndarray, names: Mapping[int, str], label: str) -> tuple[str, ...]:
    missing = sorted({int(value) for value in ids if int(value) not in names})
    if missing:
        raise EFITLinearizationError(f"{label} map does not define IDs {missing}")
    aliases = {"ip": "plasma_current", "b_pol_probe": "bpol_probe"}
    return tuple(aliases.get(names[int(value)], names[int(value)]) for value in ids)


def _parameter_order_sha256(
    names: Sequence[str],
    roles: Sequence[str],
    indices: np.ndarray,
    units: Sequence[str],
) -> str:
    """Reproduce the native compact sorted-key parameter-order digest."""
    payload = {
        "parameters": [
            {
                "index": int(index),
                "name": str(name),
                "role": str(role),
                "units": str(unit),
            }
            for name, role, index, unit in zip(names, roles, indices, units)
        ]
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _required_row_array(dataset: Any, name: str, nrow: int) -> np.ndarray:
    values = _array(dataset, name)
    if values.shape != (nrow,):
        raise EFITLinearizationError(f"{name} has shape {values.shape}, expected {(nrow,)}")
    return values


def _validate_row_metadata(
    *,
    prefix: str,
    statistical: np.ndarray,
    measurement: np.ndarray,
    reconstruction: np.ndarray,
    physical_residual: np.ndarray,
    uncertainty: np.ndarray,
    submitted_fwt: np.ndarray,
    processed_weight: np.ndarray,
    solver_weighted_residual: np.ndarray,
) -> None:
    """Enforce the v1 row contract without inventing statistical metadata."""
    arrays = {
        "row_measurement": measurement,
        "row_reconstruction": reconstruction,
        "row_physical_residual": physical_residual,
        "row_uncertainty": uncertainty,
        "row_submitted_fwt": submitted_fwt,
        "row_processed_weight": processed_weight,
        "row_solver_weighted_residual": solver_weighted_residual,
    }
    expected = statistical.shape
    for name, values in arrays.items():
        if values.shape != expected:
            raise EFITLinearizationError(
                f"{prefix}{name} has shape {values.shape}, expected {expected}"
            )

    finite_physical = (
        np.isfinite(measurement)
        & np.isfinite(reconstruction)
        & np.isfinite(physical_residual)
        & np.isfinite(submitted_fwt)
        & np.isfinite(processed_weight)
        & np.isfinite(solver_weighted_residual)
    )
    if not np.all(finite_physical):
        bad = np.flatnonzero(~finite_physical).tolist()
        raise EFITLinearizationError(
            f"{prefix}row physical/weight metadata is non-finite at rows {bad}"
        )
    if np.any(processed_weight == 0.0):
        raise EFITLinearizationError(
            f"{prefix}exported active rows must have non-zero processed weight"
        )

    statistical_uncertainty = uncertainty[statistical]
    if (
        not np.all(np.isfinite(statistical_uncertainty))
        or np.any(statistical_uncertainty <= 0.0)
    ):
        raise EFITLinearizationError(
            f"{prefix}statistical rows require finite positive uncertainty"
        )
    if np.any(np.isfinite(uncertainty[~statistical])):
        raise EFITLinearizationError(
            f"{prefix}non-statistical rows must not invent an uncertainty"
        )
    residual_identity_error = np.abs(
        reconstruction - measurement - physical_residual
    )
    residual_identity_scale = (
        np.abs(measurement) + np.abs(reconstruction) + np.abs(physical_residual)
    )
    residual_identity_tolerance = (
        100.0 * np.finfo(float).eps * residual_identity_scale
        + 1.0e-10 * np.abs(physical_residual)
        + 1.0e-14
    )
    if np.any(residual_identity_error > residual_identity_tolerance):
        raise EFITLinearizationError(
            f"{prefix}measurement, reconstruction, and physical residual disagree"
        )
    if not np.allclose(
        processed_weight[statistical],
        submitted_fwt[statistical] / statistical_uncertainty,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise EFITLinearizationError(
            f"{prefix}statistical processed weight is not submitted FWT/uncertainty"
        )
    if not np.allclose(
        np.abs(solver_weighted_residual),
        np.abs(processed_weight * physical_residual),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise EFITLinearizationError(
            f"{prefix}solver-weighted and physical residuals disagree"
        )


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(actual)), float(np.linalg.norm(expected)), 1.0)
    return float(np.linalg.norm(actual - expected) / denominator)


def _constrained_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
    exact_c: np.ndarray,
    exact_d: np.ndarray,
) -> np.ndarray:
    """Independently solve ``min ||Ax-b||`` subject to ``Cx=d``.

    EFIT uses DGGLSE for these systems.  Reproduction deliberately uses a
    separate null-space construction and direct SVD/lstsq calls so a feasible
    but non-optimal exported vector cannot validate itself.
    """
    ncol = int(matrix.shape[1])
    if exact_c.shape[0] == 0:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    u_c, singular_c, vh_c = np.linalg.svd(exact_c, full_matrices=True)
    cutoff = (
        np.finfo(float).eps
        * max(exact_c.shape)
        * (float(singular_c[0]) if singular_c.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_c > cutoff))
    if rank != exact_c.shape[0]:
        raise EFITLinearizationError("exported exact constraints are rank deficient")
    particular = np.linalg.lstsq(exact_c, exact_d, rcond=None)[0]
    if _relative_error(exact_c @ particular, exact_d) > 1.0e-12:
        raise EFITLinearizationError("exported exact constraints are inconsistent")
    null_basis = vh_c[rank:, :].T
    if null_basis.shape != (ncol, ncol - rank):
        raise EFITLinearizationError("cannot construct exact-constraint null space")
    if null_basis.shape[1] == 0:
        return particular
    correction = np.linalg.lstsq(
        matrix @ null_basis,
        rhs - matrix @ particular,
        rcond=None,
    )[0]
    return particular + null_basis @ correction


def _read_reduced_solve(
    dataset: Any,
    *,
    prefix: str,
    nrow: int,
    ncol: int,
    nexact: int,
    weighted_a: np.ndarray,
    solver_a: np.ndarray,
    solver_rhs: np.ndarray,
    solver_exact_c: np.ndarray,
    solver_exact_d: np.ndarray,
    solver_solution: np.ndarray,
    physical_solution: np.ndarray,
    column_scale: np.ndarray,
    right_vectors: np.ndarray,
    retained_mask: np.ndarray,
) -> EFITReducedLinearSolve | None:
    reduced_ncol_value = _scalar(dataset, f"{prefix}reduced_ncol")
    if reduced_ncol_value is None:
        return None
    if prefix != "main_":
        raise EFITLinearizationError("only the main solve may use reduced coordinates")
    if nexact != 1:
        raise EFITLinearizationError(
            "the directional reduced solve requires exactly one exact row"
        )
    reduced_ncol = int(reduced_ncol_value)
    if reduced_ncol <= 0 or reduced_ncol > ncol:
        raise EFITLinearizationError(
            f"{prefix}reduced_ncol={reduced_ncol} is outside 1..{ncol}"
        )
    names_and_shapes = {
        "reduced_solver_a": (nrow, reduced_ncol),
        "reduced_solver_rhs": (nrow,),
        "reduced_exact_c": (nexact, reduced_ncol),
        "reduced_exact_d": (nexact,),
        "reduced_solver_basis": (ncol, reduced_ncol),
        "reduced_physical_from_solver": (ncol, reduced_ncol),
        "reduced_solver_offset": (ncol,),
        "reduced_physical_offset": (ncol,),
        "reduced_solver_solution": (reduced_ncol,),
    }
    values: dict[str, np.ndarray] = {}
    for suffix, shape in names_and_shapes.items():
        value = _array(dataset, f"{prefix}{suffix}")
        if value.shape != shape:
            raise EFITLinearizationError(
                f"{prefix}{suffix} has shape {value.shape}, expected {shape}"
            )
        if not np.all(np.isfinite(value)):
            raise EFITLinearizationError(f"{prefix}{suffix} contains non-finite values")
        values[suffix] = value
    basis_kind = _array(dataset, f"{prefix}reduced_basis_kind", dtype=int)
    if basis_kind.shape != (reduced_ncol,) or not set(np.unique(basis_kind)).issubset(
        {1, 2}
    ):
        raise EFITLinearizationError(
            f"{prefix}reduced_basis_kind must contain {reduced_ncol} retained/target IDs"
        )
    reduced_status = int(_scalar(dataset, f"{prefix}reduced_status", -1))
    reduced_condno = float(_scalar(dataset, f"{prefix}reduced_condno", np.nan))
    discarded_norm = float(
        _scalar(dataset, f"{prefix}direction_discarded_projection_norm", np.nan)
    )
    if (
        reduced_status != 0
        or not math.isfinite(reduced_condno)
        or reduced_condno < 0.0
        or not math.isfinite(discarded_norm)
        or discarded_norm < 0.0
    ):
        raise EFITLinearizationError(f"{prefix}reduced solve metadata is invalid")

    basis = values["reduced_solver_basis"]
    solver_offset = values["reduced_solver_offset"]
    physical_map = values["reduced_physical_from_solver"]
    physical_offset = values["reduced_physical_offset"]
    reduced_a = values["reduced_solver_a"]
    reduced_rhs = values["reduced_solver_rhs"]
    reduced_c = values["reduced_exact_c"]
    reduced_d = values["reduced_exact_d"]
    reduced_solution = values["reduced_solver_solution"]
    retained_vectors = right_vectors[:, retained_mask]
    discarded_vectors = right_vectors[:, ~retained_mask]
    discarded_coordinates = discarded_vectors.T @ solver_exact_c[0]
    expected_discarded_norm = float(np.linalg.norm(discarded_coordinates))
    has_discarded_target = expected_discarded_norm > (
        100.0 * np.finfo(float).eps * float(np.linalg.norm(solver_exact_c[0]))
    )
    expected_ncol = retained_vectors.shape[1] + int(has_discarded_target)
    if reduced_ncol != expected_ncol:
        raise EFITLinearizationError(
            f"{prefix}reduced basis does not contain the retained modes plus one target mode"
        )
    if not np.array_equal(
        basis_kind,
        np.asarray(
            [1] * retained_vectors.shape[1]
            + ([2] if has_discarded_target else []),
            dtype=int,
        ),
    ):
        raise EFITLinearizationError(f"{prefix}reduced basis-kind ordering is invalid")
    basis_error = _relative_error(
        basis[:, : retained_vectors.shape[1]], retained_vectors
    )
    if has_discarded_target:
        expected_target = (
            discarded_vectors @ discarded_coordinates / expected_discarded_norm
        )
        basis_error = max(
            basis_error,
            _relative_error(basis[:, -1], expected_target),
        )
    basis_error = max(
        basis_error,
        _relative_error(basis.T @ basis, np.eye(reduced_ncol)),
        abs(discarded_norm - expected_discarded_norm)
        / max(discarded_norm, expected_discarded_norm, 1.0),
    )
    if basis_error > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix}reduced basis disagrees with the native retained/discarded SVD"
        )
    transform_error = max(
        basis_error,
        _relative_error(reduced_a, solver_a @ basis),
        _relative_error(reduced_rhs, solver_rhs - solver_a @ solver_offset),
        _relative_error(reduced_c, solver_exact_c @ basis),
        _relative_error(
            reduced_d,
            solver_exact_d - solver_exact_c @ solver_offset,
        ),
        _relative_error(
            physical_map,
            column_scale[:, np.newaxis] * basis,
        ),
        _relative_error(physical_offset, column_scale * solver_offset),
        _relative_error(
            solver_solution,
            solver_offset + basis @ reduced_solution,
        ),
        _relative_error(
            physical_solution,
            physical_offset + physical_map @ reduced_solution,
        ),
        _relative_error(reduced_a, weighted_a @ physical_map),
    )
    if transform_error > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix}reduced-coordinate transform cannot be reproduced"
        )
    reproduced = _constrained_least_squares(
        reduced_a,
        reduced_rhs,
        reduced_c,
        reduced_d,
    )
    reproduction_error = _relative_error(reduced_solution, reproduced)
    if reproduction_error > 1.0e-8:
        raise EFITLinearizationError(
            f"{prefix}reduced DGGLSE solution cannot be reproduced"
        )
    return EFITReducedLinearSolve(
        solver_a=reduced_a,
        solver_rhs=reduced_rhs,
        exact_c=reduced_c,
        exact_d=reduced_d,
        solver_basis=basis,
        physical_from_solver=physical_map,
        solver_offset=solver_offset,
        physical_offset=physical_offset,
        solver_solution=reduced_solution,
        basis_kind=basis_kind,
        status=reduced_status,
        condno=reduced_condno,
        discarded_projection_norm=discarded_norm,
        transform_relative_error=transform_error,
        reproduction_relative_error=reproduction_error,
    )


def _validate_linear_solve(
    *,
    prefix: str,
    weighted_a: np.ndarray,
    weighted_rhs: np.ndarray,
    solver_a: np.ndarray,
    solver_rhs: np.ndarray,
    solver_exact_c: np.ndarray,
    solver_exact_d: np.ndarray,
    exact_c: np.ndarray,
    exact_d: np.ndarray,
    solver_solution: np.ndarray,
    physical_solution: np.ndarray,
    column_scale: np.ndarray,
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    retained_mask: np.ndarray,
    condin: float,
    solver_method: str,
    status: int | None,
    reduced_solve: EFITReducedLinearSolve | None,
) -> LinearSolveValidation:
    scaled_a = weighted_a * column_scale[np.newaxis, :]
    scaling_error = max(
        _relative_error(solver_a, scaled_a),
        _relative_error(solver_rhs, weighted_rhs),
    )
    if scaling_error > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix} physical-to-solver column transform disagrees with exported arrays"
        )
    solution_scaling_error = _relative_error(
        physical_solution, column_scale * solver_solution
    )
    if solution_scaling_error > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix} physical and solver solutions disagree with column_scale"
        )

    # Recompute the native full-matrix spectrum directly from A, including
    # exact-row cases.  The response SVD describes the truncation basis; it is
    # intentionally distinct from any reduced DGGLSE factorization.
    u, recomputed, vh = np.linalg.svd(solver_a, full_matrices=True)
    padded = np.zeros(solver_a.shape[1])
    padded[: recomputed.size] = recomputed
    singular_error: float | None = _relative_error(singular_values, padded)
    if singular_error > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix} native and recomputed singular values disagree"
        )
    if np.any(retained_mask & (padded <= 0)):
        raise EFITLinearizationError(
            f"{prefix} retained mask includes a zero singular value"
        )
    retained_indices = np.flatnonzero(retained_mask)
    if np.any(retained_indices >= recomputed.size):
        raise EFITLinearizationError(
            f"{prefix} retained mask extends beyond the numerical SVD rank"
        )
    if solver_method in {
        "truncated_svd",
        "truncated_svd_reduced_dgglse_exact_direction",
    } or (not solver_method and exact_c.shape[0] == 0):
        expected_retained = singular_values > condin * (
            singular_values[0] if singular_values.size else 0.0
        )
        if not np.array_equal(retained_mask, expected_retained):
            raise EFITLinearizationError(
                f"{prefix}retained_mask disagrees with s > condin*smax"
            )

    reproduction_error: float | None = None
    exact_residual: float | None = None
    exact_tolerance: float | None = None
    if exact_c.shape[0]:
        solver_residual = solver_exact_c @ solver_solution - solver_exact_d
        physical_residual = exact_c @ physical_solution - exact_d
        solver_residual_norm = float(np.linalg.norm(solver_residual))
        physical_residual_norm = float(np.linalg.norm(physical_residual))
        solver_tolerance = float(
            100.0
            * np.finfo(float).eps
            * (
                np.linalg.norm(solver_exact_c) * np.linalg.norm(solver_solution)
                + np.linalg.norm(solver_exact_d)
            )
        )
        physical_tolerance = float(
            100.0
            * np.finfo(float).eps
            * (
                np.linalg.norm(exact_c) * np.linalg.norm(physical_solution)
                + np.linalg.norm(exact_d)
            )
        )
        exact_residual = max(solver_residual_norm, physical_residual_norm)
        exact_tolerance = max(solver_tolerance, physical_tolerance)
        if (
            solver_residual_norm > solver_tolerance
            or physical_residual_norm > physical_tolerance
        ):
            raise EFITLinearizationError(
                f"{prefix} solution violates exported exact constraints"
            )
        if reduced_solve is not None:
            reproduction_error = reduced_solve.reproduction_relative_error
        else:
            reproduced = _constrained_least_squares(
                solver_a,
                solver_rhs,
                solver_exact_c,
                solver_exact_d,
            )
            reproduction_error = _relative_error(solver_solution, reproduced)
            if reproduction_error > 1.0e-8:
                raise EFITLinearizationError(
                    f"{prefix} DGGLSE solution cannot be reproduced"
                )
    else:
        # ``sdecm`` uses ``ier=33`` to flag an ill-conditioned matrix even
        # though it has produced a usable SVD, and older builds may leave
        # ``ier`` undefined on the ordinary-success path.  The presence of a
        # complete unconstrained block is therefore the validation signal;
        # treating the raw status integer as a conventional zero-success code
        # would skip the most important reproduction check.
        if reduced_solve is not None:
            raise EFITLinearizationError(
                f"{prefix} reduced directional solve is missing its exact row"
            )
        reproduced = np.zeros(solver_a.shape[1])
        if retained_indices.size:
            coefficients = (
                u[:, retained_indices].T @ solver_rhs
            ) / recomputed[retained_indices]
            reproduced = vh[retained_indices].T @ coefficients
        reproduction_error = _relative_error(solver_solution, reproduced)
        if reproduction_error > 1.0e-8:
            raise EFITLinearizationError(
                f"{prefix} truncated-SVD solution cannot be reproduced"
            )

    return LinearSolveValidation(
        column_scaling_relative_error=scaling_error,
        solution_scaling_relative_error=solution_scaling_error,
        singular_value_relative_error=singular_error,
        solver_reproduction_relative_error=reproduction_error,
        exact_constraint_residual_norm=exact_residual,
        exact_constraint_tolerance=exact_tolerance,
    )


def _read_block(
    dataset: Any,
    prefix: str,
    *,
    family_names: Mapping[int, str],
    role_names: Mapping[int, str],
    required: bool,
) -> EFITLinearizationBlock | None:
    scalar_prefix = prefix
    nrow_value = _scalar(dataset, f"{scalar_prefix}nrow")
    ncol_value = _scalar(dataset, f"{scalar_prefix}ncol")
    if nrow_value is None or ncol_value is None:
        if required:
            raise EFITLinearizationError(f"missing {prefix.rstrip('_')} dimensions")
        return None
    nrow, ncol = int(nrow_value), int(ncol_value)
    if not required and nrow == 0 and ncol == 0:
        return None
    if nrow < 0 or ncol <= 0:
        raise EFITLinearizationError(f"invalid {prefix.rstrip('_')} shape {nrow}x{ncol}")
    nexact_value = _scalar(dataset, f"{prefix}nexact")
    if nexact_value is None:
        raise EFITLinearizationError(f"missing required scalar {prefix}nexact")
    nexact_float = float(nexact_value)
    if not math.isfinite(nexact_float) or not nexact_float.is_integer():
        raise EFITLinearizationError(f"{prefix}nexact must be a finite integer")
    nexact = int(nexact_float)
    if nexact < 0 or nexact > ncol:
        raise EFITLinearizationError(
            f"{prefix}nexact={nexact} is outside the valid range 0..{ncol}"
        )

    weighted_a = _array(dataset, f"{prefix}weighted_a")
    solver_a = _array(dataset, f"{prefix}solver_a")
    expected = (nrow, ncol)
    for name, values in (("weighted_a", weighted_a), ("solver_a", solver_a)):
        if values.shape != expected:
            raise EFITLinearizationError(
                f"{prefix}{name} has shape {values.shape}, expected {expected}"
            )
        if not np.all(np.isfinite(values)):
            raise EFITLinearizationError(f"{prefix}{name} contains non-finite values")
    weighted_rhs = _array(dataset, f"{prefix}weighted_rhs")
    solver_rhs = _array(dataset, f"{prefix}solver_rhs")
    for name, values in (("weighted_rhs", weighted_rhs), ("solver_rhs", solver_rhs)):
        if values.shape != (nrow,):
            raise EFITLinearizationError(
                f"{prefix}{name} has shape {values.shape}, expected {(nrow,)}"
            )
        if not np.all(np.isfinite(values)):
            raise EFITLinearizationError(f"{prefix}{name} contains non-finite values")

    # New native sidecars expose both forms: ``exact_c`` acts on the physical
    # parameter vector, while ``solver_exact_c`` is exactly what DGGLSE saw.
    # The solver form is optional only for the zero-row legacy baseline.
    exact_c = _array(
        dataset,
        f"{prefix}exact_c",
        default=np.empty((0, ncol)) if nexact == 0 else None,
    )
    exact_d = _array(
        dataset,
        f"{prefix}exact_d",
        default=np.empty(0) if nexact == 0 else None,
    )
    if exact_c.shape != (nexact, ncol) or exact_d.shape != (nexact,):
        raise EFITLinearizationError(
            f"{prefix} exact-constraint shapes do not match nexact={nexact}"
        )

    solver_solution = _array(dataset, f"{prefix}solver_solution")
    physical_solution = _array(dataset, f"{prefix}physical_solution")
    column_scale = _array(dataset, f"{prefix}column_scale")
    singular_values = _array(dataset, f"{prefix}singular_values")
    right_vectors = _array(dataset, f"{prefix}right_singular_vectors")
    raw_retained_mask = _array(dataset, f"{prefix}retained_mask", dtype=int)
    if not set(np.unique(raw_retained_mask)).issubset({0, 1}):
        raise EFITLinearizationError(f"{prefix}retained_mask must contain only 0 or 1")
    retained_mask = raw_retained_mask.astype(bool)
    for name, values in (
        ("solver_solution", solver_solution),
        ("physical_solution", physical_solution),
        ("column_scale", column_scale),
        ("singular_values", singular_values),
        ("retained_mask", retained_mask),
    ):
        if values.shape != (ncol,):
            raise EFITLinearizationError(
                f"{prefix}{name} has shape {values.shape}, expected {(ncol,)}"
            )
    for name, values in (
        ("solver_solution", solver_solution),
        ("physical_solution", physical_solution),
        ("column_scale", column_scale),
        ("singular_values", singular_values),
        ("right_singular_vectors", right_vectors),
        ("exact_c", exact_c),
        ("exact_d", exact_d),
    ):
        if not np.all(np.isfinite(values)):
            raise EFITLinearizationError(f"{prefix}{name} contains non-finite values")
    if np.any(column_scale <= 0):
        raise EFITLinearizationError(f"{prefix}column_scale must be positive")
    solver_exact_c = _array(
        dataset,
        f"{prefix}solver_exact_c",
        default=np.empty((0, ncol)) if nexact == 0 else None,
    )
    solver_exact_d = _array(
        dataset,
        f"{prefix}solver_exact_d",
        default=np.empty(0) if nexact == 0 else None,
    )
    if solver_exact_c.shape != (nexact, ncol) or solver_exact_d.shape != (nexact,):
        raise EFITLinearizationError(
            f"{prefix} solver exact-constraint shapes do not match nexact={nexact}"
        )
    derived_physical_c = solver_exact_c / column_scale[np.newaxis, :]
    if _relative_error(exact_c, derived_physical_c) > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix} physical and solver exact constraints disagree with column_scale"
        )
    if _relative_error(exact_d, solver_exact_d) > 1.0e-12:
        raise EFITLinearizationError(
            f"{prefix} physical and solver exact right-hand sides disagree"
        )
    for name, values in (
        ("solver_exact_c", solver_exact_c),
        ("solver_exact_d", solver_exact_d),
    ):
        if not np.all(np.isfinite(values)):
            raise EFITLinearizationError(f"{prefix}{name} contains non-finite values")
    if nexact:
        calculated_physical_exact_residual = exact_c @ physical_solution - exact_d
        calculated_solver_exact_residual = (
            solver_exact_c @ solver_solution - solver_exact_d
        )
        exported_physical_exact_residual = _array(
            dataset,
            f"{prefix}exact_residual",
        )
        exported_solver_exact_residual = _array(
            dataset,
            f"{prefix}solver_exact_residual",
        )
        if (
            exported_physical_exact_residual.shape != (nexact,)
            or exported_solver_exact_residual.shape != (nexact,)
            or not np.all(np.isfinite(exported_physical_exact_residual))
            or not np.all(np.isfinite(exported_solver_exact_residual))
            or _relative_error(
                exported_physical_exact_residual,
                calculated_physical_exact_residual,
            )
            > 1.0e-10
            or _relative_error(
                exported_solver_exact_residual,
                calculated_solver_exact_residual,
            )
            > 1.0e-10
        ):
            raise EFITLinearizationError(
                f"{prefix} exported exact-constraint residuals disagree"
            )
    if np.any(singular_values < 0):
        raise EFITLinearizationError(f"{prefix}singular_values must be non-negative")
    if right_vectors.shape != (ncol, ncol):
        raise EFITLinearizationError(
            f"{prefix}right_singular_vectors has shape {right_vectors.shape}, "
            f"expected {(ncol, ncol)}"
        )
    right_vector_error = max(
        _relative_error(right_vectors.T @ right_vectors, np.eye(ncol)),
        _relative_error(
            solver_a.T @ solver_a,
            right_vectors
            @ np.diag(np.square(singular_values))
            @ right_vectors.T,
        ),
    )
    if right_vector_error > 1.0e-10:
        raise EFITLinearizationError(
            f"{prefix}right singular vectors do not reproduce A.T@A"
        )

    family_ids = _array(dataset, f"{prefix}row_family_id", dtype=int)
    channels = _array(dataset, f"{prefix}row_channel", dtype=int)
    raw_statistical = _array(dataset, f"{prefix}row_statistical", dtype=int)
    if not set(np.unique(raw_statistical)).issubset({0, 1}):
        raise EFITLinearizationError(f"{prefix}row_statistical must contain only 0 or 1")
    statistical = raw_statistical.astype(bool)
    for name, values in (
        ("row_family_id", family_ids),
        ("row_channel", channels),
        ("row_statistical", statistical),
    ):
        if values.shape != (nrow,):
            raise EFITLinearizationError(
                f"{prefix}{name} has shape {values.shape}, expected {(nrow,)}"
            )
    families = _mapped_names(family_ids, family_names, "family")
    row_keys = [
        (int(family), int(channel))
        for family, channel in zip(family_ids, channels)
    ]
    if len(set(row_keys)) != len(row_keys):
        raise EFITLinearizationError(
            f"{prefix}row family/channel pairs must be unique"
        )
    if any(
        family == "pf_relation" and is_statistical
        for family, is_statistical in zip(families, statistical)
    ):
        raise EFITLinearizationError(
            f"{prefix}PF relation rows must be non-statistical soft rows"
        )
    kinds = tuple(
        "soft_structural_relation"
        if family == "pf_relation"
        else ("statistical" if is_statistical else "structural")
        for family, is_statistical in zip(families, statistical)
    )

    role_ids = _array(dataset, f"{prefix}parameter_role_id", dtype=int)
    parameter_indices = _array(dataset, f"{prefix}parameter_index", dtype=int)
    if role_ids.shape != (ncol,) or parameter_indices.shape != (ncol,):
        raise EFITLinearizationError(f"{prefix} parameter metadata length mismatch")
    roles = _mapped_names(role_ids, role_names, "parameter role")
    parameter_names = _strings(dataset, f"{prefix}parameter_name", ncol)
    parameter_units = _strings(dataset, f"{prefix}parameter_units", ncol)
    if np.any(parameter_indices <= 0):
        raise EFITLinearizationError(
            f"{prefix}parameter_index values must be positive"
        )
    parameter_keys = [
        (int(role), int(index))
        for role, index in zip(role_ids, parameter_indices)
    ]
    if len(set(parameter_keys)) != len(parameter_keys):
        raise EFITLinearizationError(
            f"{prefix}parameter role/index pairs must be unique"
        )
    for role_id in sorted(set(int(value) for value in role_ids)):
        role_indices = sorted(
            int(index)
            for index, value in zip(parameter_indices, role_ids)
            if int(value) == role_id
        )
        if role_indices != list(range(1, len(role_indices) + 1)):
            raise EFITLinearizationError(
                f"{prefix}parameter indices for role {role_id} "
                "must be contiguous from 1"
            )

    status_value = _scalar(dataset, f"{prefix}status")
    condin_value = _scalar(dataset, f"{prefix}condin")
    condno_value = _scalar(dataset, f"{prefix}condno")
    if status_value is None or condin_value is None or condno_value is None:
        raise EFITLinearizationError(
            f"{prefix}status, condin, and condno are required"
        )
    status = int(status_value)
    condin = float(condin_value)
    condno = float(condno_value)
    if not math.isfinite(condin) or condin <= 0.0:
        raise EFITLinearizationError(f"{prefix}condin must be finite and positive")
    if not math.isfinite(condno) or condno < 0.0:
        raise EFITLinearizationError(f"{prefix}condno must be finite and non-negative")
    solver_method = str(
        _attribute(dataset.attrs.get(f"{prefix.rstrip('_')}_solver_method", ""))
    ).strip()
    if solver_method not in SOLVER_METHODS:
        raise EFITLinearizationError(
            f"{prefix}solver_method is missing or unrecognized: {solver_method!r}"
        )
    if nexact == 0 and solver_method != "truncated_svd":
        raise EFITLinearizationError(
            f"{prefix}unconstrained block must identify the truncated-SVD solver"
        )
    if nexact > 0 and solver_method == "truncated_svd":
        raise EFITLinearizationError(
            f"{prefix}exact rows require an exact-constrained solver method"
        )
    if (
        singular_values.size
        and singular_values[-1] > 0.0
        and (nexact == 0 or solver_method == "truncated_svd_reduced_dgglse_exact_direction")
    ):
        direct_condno = float(singular_values[0] / singular_values[-1])
        condno_relative_error = abs(condno - direct_condno) / max(
            abs(direct_condno), np.finfo(float).tiny
        )
        if condno_relative_error > 5.0e-6:
            raise EFITLinearizationError(
                f"{prefix}condno disagrees with the exported response spectrum"
            )
    reduced_solve = _read_reduced_solve(
        dataset,
        prefix=prefix,
        nrow=nrow,
        ncol=ncol,
        nexact=nexact,
        weighted_a=weighted_a,
        solver_a=solver_a,
        solver_rhs=solver_rhs,
        solver_exact_c=solver_exact_c,
        solver_exact_d=solver_exact_d,
        solver_solution=solver_solution,
        physical_solution=physical_solution,
        column_scale=column_scale,
        right_vectors=right_vectors,
        retained_mask=retained_mask,
    )
    if reduced_solve is not None and solver_method != (
        "truncated_svd_reduced_dgglse_exact_direction"
    ):
        raise EFITLinearizationError(
            f"{prefix}reduced arrays require the directional reduced-DGGLSE method"
        )
    if reduced_solve is None and solver_method == (
        "truncated_svd_reduced_dgglse_exact_direction"
    ):
        raise EFITLinearizationError(
            f"{prefix}directional reduced-DGGLSE method requires reduced arrays"
        )
    validation = _validate_linear_solve(
        prefix=prefix,
        weighted_a=weighted_a,
        weighted_rhs=weighted_rhs,
        solver_a=solver_a,
        solver_rhs=solver_rhs,
        solver_exact_c=solver_exact_c,
        solver_exact_d=solver_exact_d,
        exact_c=exact_c,
        exact_d=exact_d,
        solver_solution=solver_solution,
        physical_solution=physical_solution,
        column_scale=column_scale,
        singular_values=singular_values,
        right_vectors=right_vectors,
        retained_mask=retained_mask,
        condin=condin,
        solver_method=solver_method,
        status=status,
        reduced_solve=reduced_solve,
    )

    row_measurement = _required_row_array(
        dataset, f"{prefix}row_measurement", nrow
    )
    row_reconstruction = _required_row_array(
        dataset, f"{prefix}row_reconstruction", nrow
    )
    row_physical_residual = _required_row_array(
        dataset, f"{prefix}row_physical_residual", nrow
    )
    row_uncertainty = _required_row_array(
        dataset, f"{prefix}row_uncertainty", nrow
    )
    row_submitted_fwt = _required_row_array(
        dataset, f"{prefix}row_submitted_fwt", nrow
    )
    row_processed_weight = _required_row_array(
        dataset, f"{prefix}row_processed_weight", nrow
    )
    row_solver_weighted_residual = _required_row_array(
        dataset, f"{prefix}row_solver_weighted_residual", nrow
    )
    _validate_row_metadata(
        prefix=prefix,
        statistical=statistical,
        measurement=row_measurement,
        reconstruction=row_reconstruction,
        physical_residual=row_physical_residual,
        uncertainty=row_uncertainty,
        submitted_fwt=row_submitted_fwt,
        processed_weight=row_processed_weight,
        solver_weighted_residual=row_solver_weighted_residual,
    )
    result = EFITLinearizationBlock(
        name=prefix.rstrip("_"),
        weighted_a=weighted_a,
        weighted_rhs=weighted_rhs,
        solver_a=solver_a,
        solver_rhs=solver_rhs,
        solver_exact_c=solver_exact_c,
        solver_exact_d=solver_exact_d,
        exact_c=exact_c,
        exact_d=exact_d,
        solver_solution=solver_solution,
        physical_solution=physical_solution,
        column_scale=column_scale,
        singular_values=singular_values,
        right_singular_vectors=right_vectors,
        retained_mask=retained_mask,
        row_family_id=family_ids,
        row_family=families,
        row_kind=kinds,
        row_channel=channels,
        row_statistical=statistical,
        row_measurement=row_measurement,
        row_reconstruction=row_reconstruction,
        row_physical_residual=row_physical_residual,
        row_uncertainty=row_uncertainty,
        row_submitted_fwt=row_submitted_fwt,
        row_processed_weight=row_processed_weight,
        row_solver_weighted_residual=row_solver_weighted_residual,
        parameter_name=parameter_names,
        parameter_role_id=role_ids,
        parameter_role=roles,
        parameter_index=parameter_indices,
        parameter_units=parameter_units,
        validation=validation,
        solver_method=solver_method,
        reduced_solve=reduced_solve,
        outer_iteration=_optional_int(
            _scalar(dataset, f"{prefix}outer_iteration")
        ),
        total_iteration=_optional_int(
            _scalar(dataset, f"{prefix}total_iteration")
        ),
        status=status,
        condin=condin,
        condno=condno,
    )

    exported_residual = result.row_solver_weighted_residual
    if np.any(np.isfinite(exported_residual)):
        calculated = result.solver_residual
        active = np.isfinite(exported_residual)
        scale = max(float(np.linalg.norm(exported_residual[active])), 1.0)
        if np.linalg.norm(calculated[active] - exported_residual[active]) > 1.0e-10 * scale:
            raise EFITLinearizationError(
                f"{prefix} exported and reconstructed solver residuals disagree"
            )
    return result


def read_efit_linearization(path: str | Path) -> EFITLinearization:
    """Load and structurally validate one native EFIT response sidecar."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    import xarray as xr

    try:
        with xr.open_dataset(source, decode_cf=False) as opened:
            dataset = opened.load()
    except Exception as exc:
        raise EFITLinearizationError(f"cannot read NetCDF sidecar: {exc}") from exc
    attrs = {str(key): _attribute(value) for key, value in dataset.attrs.items()}
    schema_id = str(attrs.get("schema_id", ""))
    try:
        schema_version = int(attrs.get("schema_version", -1))
        writer_complete = int(attrs.get("writer_complete", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EFITLinearizationError(
            "response sidecar has invalid schema/completion attributes"
        ) from exc
    if schema_id != SCHEMA_ID or schema_version != SCHEMA_VERSION:
        raise EFITLinearizationError(
            f"unsupported response schema {schema_id!r} version {schema_version}"
        )
    if writer_complete != 1:
        raise EFITLinearizationError("response sidecar is incomplete (writer_complete != 1)")
    writer_complete_value = _scalar(dataset, "writer_complete")
    try:
        writer_complete_number = float(writer_complete_value)
    except (TypeError, ValueError):
        writer_complete_number = math.nan
    if (
        not math.isfinite(writer_complete_number)
        or not writer_complete_number.is_integer()
        or int(writer_complete_number) != 1
    ):
        raise EFITLinearizationError(
            "response sidecar is incomplete (writer_complete variable != 1)"
        )
    source_revision = str(
        attrs.get("source_revision", attrs.get("efit_commit", ""))
    ).strip()
    executable_sha256 = str(attrs.get("executable_sha256", "")).strip()
    if (
        len(source_revision) not in {40, 64}
        or source_revision.lower() != source_revision
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise EFITLinearizationError(
            "response sidecar has a missing or invalid full EFIT source revision"
        )
    if (
        len(executable_sha256) != 64
        or executable_sha256.lower() != executable_sha256
        or any(character not in "0123456789abcdef" for character in executable_sha256)
    ):
        raise EFITLinearizationError(
            "response sidecar has a missing or invalid executable SHA-256"
        )
    parameter_order_digest = str(attrs.get("parameter_order_sha256", "")).strip()
    if (
        len(parameter_order_digest) != 64
        or parameter_order_digest.lower() != parameter_order_digest
        or any(
            character not in "0123456789abcdef"
            for character in parameter_order_digest
        )
    ):
        raise EFITLinearizationError(
            "response sidecar has a missing or invalid parameter-order SHA-256"
        )
    matrix_state = str(attrs.get("matrix_state", ""))
    if matrix_state != "last_accepted_linear_solve":
        raise EFITLinearizationError(f"unsupported matrix state {matrix_state!r}")

    family_names = _id_map(attrs.get("family_map"), FAMILY_IDS)
    role_names = _id_map(attrs.get("parameter_role_map"), PARAMETER_ROLE_IDS)
    main = _read_block(
        dataset,
        "main_",
        family_names=family_names,
        role_names=role_names,
        required=True,
    )
    assert main is not None
    calculated_parameter_order_digest = _parameter_order_sha256(
        main.parameter_name,
        main.parameter_role,
        main.parameter_index,
        main.parameter_units,
    )
    if parameter_order_digest != calculated_parameter_order_digest:
        raise EFITLinearizationError(
            "response sidecar parameter-order SHA-256 disagrees with metadata"
        )
    external = _read_block(
        dataset,
        "external_current_",
        family_names=family_names,
        role_names=role_names,
        required=False,
    )
    final_brsp = _array(dataset, "final_brsp")
    if final_brsp.shape != (main.ncol,):
        raise EFITLinearizationError("final_brsp length does not match main_ncol")
    if not np.all(np.isfinite(final_brsp)):
        raise EFITLinearizationError("final_brsp contains non-finite values")

    required_scalar_names = (
        "shot",
        "time_ms",
        "jtime",
        "ivesel",
        "ip_measured",
        "ip_reconstructed",
        "vessel_current_sum",
        "chipasma",
    )
    required_scalars = {
        name: _scalar(dataset, name) for name in required_scalar_names
    }
    if any(value is None for value in required_scalars.values()):
        missing = [name for name, value in required_scalars.items() if value is None]
        raise EFITLinearizationError(
            "response sidecar is missing required scalars: " + ", ".join(missing)
        )
    if not all(
        math.isfinite(float(required_scalars[name]))
        for name in (
            "time_ms",
            "ip_measured",
            "ip_reconstructed",
            "vessel_current_sum",
            "chipasma",
        )
    ):
        raise EFITLinearizationError("response sidecar contains non-finite physics scalars")
    for name in ("shot", "jtime", "ivesel"):
        value = float(required_scalars[name])
        if not math.isfinite(value) or not value.is_integer():
            raise EFITLinearizationError(f"{name} must be a finite integer")
    if int(required_scalars["shot"]) <= 0 or int(required_scalars["jtime"]) <= 0:
        raise EFITLinearizationError("shot and jtime must be positive")
    root_outer_iteration = _optional_int(_scalar(dataset, "outer_iteration"))
    root_total_iteration = _optional_int(_scalar(dataset, "total_iteration"))
    if main.outer_iteration is None or main.total_iteration is None:
        raise EFITLinearizationError("main block iteration metadata is required")
    if (
        root_outer_iteration != main.outer_iteration
        or root_total_iteration != main.total_iteration
    ):
        raise EFITLinearizationError(
            "root and main block iteration metadata disagree"
        )
    if external is not None and (
        external.outer_iteration is None or external.total_iteration is None
    ):
        raise EFITLinearizationError(
            "external-current block iteration metadata is required"
        )
    result = EFITLinearization(
        path=source,
        sha256=_sha256(source),
        schema_id=schema_id,
        schema_version=schema_version,
        source_revision=source_revision,
        executable_sha256=executable_sha256,
        matrix_state=matrix_state,
        shot=int(required_scalars["shot"]),
        time_ms=float(required_scalars["time_ms"]),
        jtime=int(required_scalars["jtime"]),
        outer_iteration=root_outer_iteration,
        total_iteration=root_total_iteration,
        ivesel=int(required_scalars["ivesel"]),
        ip_measured=float(required_scalars["ip_measured"]),
        ip_reconstructed=float(required_scalars["ip_reconstructed"]),
        vessel_current_sum=float(required_scalars["vessel_current_sum"]),
        chipasma=float(required_scalars["chipasma"]),
        main=main,
        external_current=external,
        final_brsp=final_brsp,
        attributes=MappingProxyType(attrs),
    )
    if not math.isfinite(result.time_ms):
        raise EFITLinearizationError("time_ms must be finite")
    return result


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _null_space(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    ncol = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(ncol), 0
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    tolerance = (
        max(matrix.shape) * np.finfo(float).eps * singular[0]
        if singular.size and singular[0] > 0
        else 0.0
    )
    rank = int(np.count_nonzero(singular > tolerance))
    return vh[rank:].T.copy(), rank


def _full_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all right vectors and one singular value per column."""
    ncol = matrix.shape[1]
    if ncol == 0:
        return np.empty(0), np.empty((0, 0))
    if matrix.shape[0] == 0:
        return np.zeros(ncol), np.eye(ncol)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    values = np.zeros(ncol)
    values[: singular.size] = singular
    return values, vh.T


def _curvature_trace(singular: np.ndarray, cutoff: float) -> float:
    retained = singular > cutoff
    return float(np.sum(1.0 / singular[retained] ** 2)) if np.any(retained) else 0.0


def _principal_angle_degrees(
    full_vectors: np.ndarray,
    full_singular: np.ndarray,
    subset_vectors: np.ndarray,
    subset_singular: np.ndarray,
    cutoff: float,
) -> float | None:
    left = full_vectors[:, full_singular > cutoff]
    right = subset_vectors[:, subset_singular > cutoff]
    if not left.shape[1] and not right.shape[1]:
        return 0.0
    if not left.shape[1] or not right.shape[1]:
        return 90.0
    cosines = np.linalg.svd(left.T @ right, compute_uv=False)
    smallest = float(np.clip(np.min(cosines), 0.0, 1.0))
    angle = math.degrees(math.acos(smallest))
    return max(angle, 90.0 if left.shape[1] != right.shape[1] else 0.0)


def _finite_sum(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    return float(np.sum(array[finite])) if np.any(finite) else None


def _subset(
    name: str,
    families: Sequence[str],
    row_families: np.ndarray,
    scaled_projected_a: np.ndarray,
    cutoffs: tuple[float, ...],
    nominal_cutoff: float,
    full_vectors: np.ndarray,
    full_singular: np.ndarray,
) -> tuple[SubsetInformation, np.ndarray, np.ndarray]:
    family_tuple = tuple(sorted(set(families)))
    mask = np.isin(row_families, family_tuple)
    singular, vectors = _full_svd(scaled_projected_a[mask])
    ranks = tuple(int(np.count_nonzero(singular > cutoff)) for cutoff in cutoffs)
    result = SubsetInformation(
        name=name,
        families=family_tuple,
        row_count=int(np.count_nonzero(mask)),
        singular_values=singular,
        ranks=ranks,
        nullity=int(singular.size - ranks[len(ranks) // 2]),
        curvature_inverse_trace=_curvature_trace(singular, nominal_cutoff),
        maximum_principal_angle_degrees=_principal_angle_degrees(
            full_vectors,
            full_singular,
            vectors,
            singular,
            nominal_cutoff,
        ),
    )
    return result, singular, vectors


def analyze_efit_identifiability(
    problem: EFITLinearization | EFITLinearizationBlock | str | Path,
    config: IdentifiabilityConfig | None = None,
) -> IdentifiabilityReport:
    """Analyse row-family information in a native EFIT response matrix.

    The analysis always performs SVD on the row matrix itself.  It never
    eigendecomposes ``A.T @ A``, which would square the condition number.
    """
    policy = config or IdentifiabilityConfig()
    source: EFITLinearization | None
    if isinstance(problem, (str, Path)):
        source = read_efit_linearization(problem)
        block = source.block(policy.block)
    elif isinstance(problem, EFITLinearization):
        source = problem
        block = problem.block(policy.block)
    elif isinstance(problem, EFITLinearizationBlock):
        source = None
        block = problem
    else:
        raise TypeError("problem must be a path, EFITLinearization, or block")

    a = np.asarray(block.weighted_a, dtype=float)
    column_norm = np.linalg.norm(a, axis=0)
    parameter_scale = np.ones(block.ncol)
    nonzero = column_norm > 0
    parameter_scale[nonzero] = 1.0 / column_norm[nonzero]
    scaled_a = a * parameter_scale[np.newaxis, :]
    scaled_c = block.exact_c * parameter_scale[np.newaxis, :]
    null_basis, exact_rank = _null_space(scaled_c)
    scaled_projected_a = scaled_a @ null_basis
    singular, vectors = _full_svd(scaled_projected_a)

    condin = policy.condin if policy.condin is not None else block.condin
    if condin is None or not math.isfinite(condin) or condin <= 0:
        raise ValueError("a finite positive condin is required for identifiability analysis")
    largest = float(singular[0]) if singular.size else 0.0
    nominal_cutoff = float(condin) * largest
    cutoffs = tuple(nominal_cutoff * value for value in policy.cutoff_multipliers)
    ranks = tuple(int(np.count_nonzero(singular > cutoff)) for cutoff in cutoffs)
    nominal_index = min(
        range(len(policy.cutoff_multipliers)),
        key=lambda index: abs(math.log(policy.cutoff_multipliers[index])),
    )
    nominal_rank = ranks[nominal_index]

    row_families = np.asarray(block.row_family, dtype=object)
    family_names = tuple(sorted(set(block.row_family)))
    modes: list[ModeInformation] = []
    for index, (value, free_direction) in enumerate(zip(singular, vectors.T)):
        if largest == 0:
            state = "unresolved"
        elif value > policy.resolved_multiple * nominal_cutoff:
            state = "resolved"
        elif value < policy.unresolved_multiple * nominal_cutoff:
            state = "unresolved"
        else:
            state = "borderline"
        scaled_direction = null_basis @ free_direction
        physical_direction = parameter_scale * scaled_direction
        # In exact arithmetic the sum of the family response powers is
        # ``s_k**2``.  For the nearly-null modes of interest here, forming
        # ``A @ v`` involves enough cancellation that dividing each term by
        # the independently rounded singular value can leave the shares a few
        # parts in 1e10 away from unity.  Sum the same family powers for the
        # denominator: this is algebraically identical to the specified
        # q_gk definition and numerically stable at the rank boundary.
        family_power: dict[str, float] = {}
        for family in family_names:
            response = scaled_projected_a[row_families == family] @ free_direction
            family_power[family] = float(response @ response)
        denominator = float(sum(family_power.values()))
        family_shares = {
            family: (power / denominator if denominator > 0 else 0.0)
            for family, power in family_power.items()
        }
        component_power = scaled_direction**2
        total_power = float(np.sum(component_power))
        role_shares = {
            role: (
                float(np.sum(component_power[np.asarray(block.parameter_role) == role]))
                / total_power
                if total_power > 0
                else 0.0
            )
            for role in sorted(set(block.parameter_role))
        }
        modes.append(
            ModeInformation(
                index=index,
                singular_value=float(value),
                state=state,
                free_direction=free_direction.copy(),
                scaled_parameter_direction=scaled_direction,
                physical_parameter_direction=physical_direction,
                family_shares=MappingProxyType(family_shares),
                parameter_role_shares=MappingProxyType(role_shares),
            )
        )

    calculated_solver_residual = block.solver_residual
    exported = block.row_solver_weighted_residual
    solver_residual = np.where(np.isfinite(exported), exported, calculated_solver_residual)
    solver_terms = solver_residual**2
    diagnostic_terms = np.full(block.nrow, np.nan)
    valid_diagnostic = (
        block.row_statistical
        & np.isfinite(block.row_physical_residual)
        & np.isfinite(block.row_uncertainty)
        & (block.row_uncertainty > 0)
    )
    diagnostic_terms[valid_diagnostic] = (
        block.row_physical_residual[valid_diagnostic]
        / block.row_uncertainty[valid_diagnostic]
    ) ** 2
    row_audit = tuple(
        RowInformation(
            index=index,
            family=block.row_family[index],
            kind=block.row_kind[index],
            channel=int(block.row_channel[index]),
            statistical=bool(block.row_statistical[index]),
            physical_residual=_optional_float(block.row_physical_residual[index]),
            uncertainty=_optional_float(block.row_uncertainty[index]),
            submitted_fwt=_optional_float(block.row_submitted_fwt[index]),
            processed_weight=_optional_float(block.row_processed_weight[index]),
            diagnostic_chi2=_optional_float(diagnostic_terms[index]),
            solver_objective=float(solver_terms[index]),
        )
        for index in range(block.nrow)
    )

    subsets: dict[str, SubsetInformation] = {}
    subset_singular: dict[str, np.ndarray] = {}
    subset_vectors: dict[str, np.ndarray] = {}

    def add_subset(name: str, selected: Sequence[str]) -> None:
        report, values, directions = _subset(
            name,
            selected,
            row_families,
            scaled_projected_a,
            cutoffs,
            nominal_cutoff,
            vectors,
            singular,
        )
        subsets[name] = report
        subset_singular[name] = values
        subset_vectors[name] = directions

    add_subset("full", family_names)
    for family in family_names:
        add_subset(f"without_{family}", tuple(item for item in family_names if item != family))
    core = tuple(family for family in policy.core_families if family in family_names)
    add_subset("core", core)
    add_subset("core_plus_flux_loop", (*core, "flux_loop"))
    add_subset("core_plus_bpol_probe", (*core, "bpol_probe"))
    add_subset("core_plus_magnetics", (*core, "flux_loop", "bpol_probe"))
    add_subset("core_plus_diamagnetic_flux", (*core, "diamagnetic_flux"))
    add_subset(
        "core_plus_magnetics_plus_diamagnetic",
        (*core, "flux_loop", "bpol_probe", "diamagnetic_flux"),
    )

    total_solver_objective = float(np.sum(solver_terms))
    families: dict[str, FamilyInformation] = {}
    for family in family_names:
        mask = row_families == family
        base_name = f"without_{family}"
        base_singular = subset_singular[base_name]
        base_vectors = subset_vectors[base_name]
        rank_gain = tuple(
            full_rank - base_rank
            for full_rank, base_rank in zip(ranks, subsets[base_name].ranks)
        )
        # Weak and borderline directions are both relevant here.  Using only
        # the nominally truncated modes would miss a family that stabilises a
        # direction immediately above EFIT's cutoff.
        weak_mask = base_singular <= policy.resolved_multiple * nominal_cutoff
        weak_basis = base_vectors[:, weak_mask]
        family_matrix = scaled_projected_a[mask]
        # The numerator asks how much of the family acts in the feasible weak
        # subspace.  The denominator deliberately retains the family's full
        # scaled norm, including directions removed by exact constraints.
        denominator = float(np.linalg.norm(scaled_a[mask]))
        weak_fraction = (
            float(np.linalg.norm(family_matrix @ weak_basis)) / denominator
            if denominator > 0 and weak_basis.shape[1]
            else 0.0
        )
        candidate = base_singular <= policy.resolved_multiple * nominal_cutoff
        ratios: list[float] = []
        for before, after in zip(base_singular[candidate], singular[candidate]):
            if before == 0:
                ratios.append(math.inf if after > 0 else 1.0)
            else:
                ratios.append(float(after / before))
        weak_lift = max(ratios, default=1.0)
        base_trace = subsets[base_name].curvature_inverse_trace
        full_trace = subsets["full"].curvature_inverse_trace
        trace_reduction = (
            max(0.0, (base_trace - full_trace) / base_trace)
            if base_trace > 0 and all(value == 0 for value in rank_gain)
            else 0.0
        )
        independent = all(value >= 1 for value in rank_gain) and (
            weak_fraction >= policy.independent_norm_fraction
        )
        if independent:
            labels = ["independent_information"]
        elif (
            weak_lift >= policy.reinforcing_weak_lift
            or trace_reduction >= policy.reinforcing_trace_reduction
        ):
            labels = ["reinforcing"]
        else:
            labels = ["inactive_redundant"]
        objective = float(np.sum(solver_terms[mask]))
        share = objective / total_solver_objective if total_solver_objective > 0 else None
        if (
            family in policy.nonlinear_material_response
            and share is not None
            and share < policy.overwhelmed_objective_share
        ):
            labels.append("overwhelmed")
        if family in policy.structural_effects:
            labels.append("structural_anchor")
        if family in policy.accounting_confounded:
            labels.append("accounting_confounded")
        families[family] = FamilyInformation(
            family=family,
            row_count=int(np.count_nonzero(mask)),
            diagnostic_chi2=_finite_sum(diagnostic_terms[mask]),
            solver_objective=objective,
            solver_objective_share=share,
            rank_gain=rank_gain,
            weak_subspace_fraction=weak_fraction,
            maximum_weak_singular_lift=weak_lift,
            curvature_inverse_trace_reduction=trace_reduction,
            classifications=tuple(labels),
        )

    ip_accounting = None
    if source is not None and "plasma_current" in family_names:
        ip_rows = np.flatnonzero(row_families == "plasma_current")
        ip_index = int(ip_rows[0]) if ip_rows.size else -1
        residual = (
            _optional_float(block.row_physical_residual[ip_index])
            if ip_index >= 0
            else None
        )
        if residual is None and source.ip_measured is not None and source.ip_reconstructed is not None:
            residual = source.ip_reconstructed - source.ip_measured
        uncertainty = (
            _optional_float(block.row_uncertainty[ip_index])
            if ip_index >= 0
            else None
        )
        predicted = None
        if uncertainty is not None and uncertainty > 0 and source.vessel_current_sum is not None:
            predicted = (source.vessel_current_sum / uncertainty) ** 2
        ip_accounting = IpAccounting(
            measured=source.ip_measured,
            reconstructed=source.ip_reconstructed,
            solve_residual=residual,
            vessel_current_sum=source.vessel_current_sum,
            predicted_vcurrt_chi2=predicted,
            reported_chipasma=source.chipasma,
        )

    share_errors = [
        abs(sum(mode.family_shares.values()) - 1.0)
        for mode in modes
        if mode.singular_value > max(np.finfo(float).eps * largest, 0.0)
    ]
    maximum_share_error = max(share_errors, default=0.0)
    return IdentifiabilityReport(
        block=block.name,
        condin=float(condin),
        cutoff_multipliers=policy.cutoff_multipliers,
        cutoffs=cutoffs,
        nominal_cutoff=nominal_cutoff,
        parameter_scale=parameter_scale,
        null_space_basis=null_basis,
        # These are Gauss--Newton curvature/information matrices. They are
        # not labelled covariance without an independent-Gaussian model.
        gauss_newton_curvature=a.T @ a,
        scaled_projected_curvature=scaled_projected_a.T @ scaled_projected_a,
        exact_constraint_rank=exact_rank,
        singular_values=singular,
        ranks=ranks,
        nullity=int(singular.size - nominal_rank),
        modes=tuple(modes),
        maximum_family_share_sum_error=maximum_share_error,
        families=MappingProxyType(families),
        subsets=MappingProxyType(subsets),
        row_audit=row_audit,
        ip_accounting=ip_accounting,
    )


__all__ = [
    "EFITLinearization",
    "EFITLinearizationBlock",
    "EFITLinearizationError",
    "FAMILY_IDS",
    "FamilyInformation",
    "IdentifiabilityConfig",
    "IdentifiabilityReport",
    "IpAccounting",
    "LinearSolveValidation",
    "ModeInformation",
    "PARAMETER_ROLE_IDS",
    "RowInformation",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SubsetInformation",
    "analyze_efit_identifiability",
    "read_efit_linearization",
]
