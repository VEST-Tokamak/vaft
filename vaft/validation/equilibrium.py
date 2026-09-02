"""The unified equilibrium validation report (issue #72; #253 §13-§16).

One call, :func:`validate_equilibrium`, answers four separately queryable
questions about a reconstructed equilibrium:

``verification``
    Was the calculation performed as intended?  Structure, time-slice
    continuity, the COCOS convention, and the solver's own convergence.
``diagnostic_fit``
    Does the reconstruction reproduce the measurements it was fitted to?
    Per family, per scalar constraint, and globally, from #69's residuals.
``physical_validity``
    Does the result satisfy force balance and look like a plasma?  Virial and
    Shafranov quantities, the consistency of two beta_p definitions, the q and
    pressure profiles, and the reconstructed diamagnetic flux.
``independent_validation``
    Does it agree with measurements it never used?  Kinetic pressure from
    ``core_profiles`` or Thomson scattering, and the diamagnetic stored energy.

This module **composes** and never calculates (#253 §16).  Every number comes
from a canonical provider -- :mod:`vaft.omas.efit_quality`,
:mod:`vaft.omas.process_wrapper`, :mod:`vaft.process.equilibrium`,
:mod:`vaft.process.cocos`, :mod:`vaft.formula.equilibrium`,
:mod:`vaft.formula.statistics` -- and the module's own work is to select, map,
compare and grade.  What is stable about each check -- unit, provider,
tolerance, the rule that yields its status -- lives once in
:mod:`vaft.validation.registry`, so results stay compact (#253 §4).

Report shape (#253 §15)::

    {
        "schema_version": 1,
        "status": "warn",                       # aggregate, never hides not_available
        "summary": {"verification": "pass", ...},
        "provenance": {...},                    # once per report, not per check
        "time_slices": [0, 1, ...], "time": [...],
        "verification": {"structure": {"status": ..., "counts": {...}, "slices": [...]}, ...},
        ...
    }

Per-check results carry a ``status`` and a ``slices`` list of compact
per-slice mappings; whole-IDS checks carry their values directly.  Everything
is JSON-serializable and deterministic; non-finite floats serialize as
``null``.  Nothing here imports a plotting backend, and the input ODS is never
mutated: the one provider that writes into its argument runs on a copy.

Statuses (#253 §4): ``not_available`` means the evidence was never produced,
``indeterminate`` that it does not decide (a near-singular denominator, a
missing boundary), and neither is a ``pass``.  Aggregation keeps that: a
category with one failing slice fails, and one that could only be assessed
on some slices is ``indeterminate``, not ``pass``.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vaft.ods_access import path_count, path_value
from vaft.validation.model import ValidationStatus
from vaft.validation.registry import CHECKS, checks_in, describe

__all__ = [
    "EQUILIBRIUM_CATEGORIES",
    "aggregate_status",
    "status_summary",
    "validate_equilibrium",
    "validate_independent",
    "validate_magnetic_fit",
    "validate_physical",
    "verify_continuity",
    "verify_convention",
    "verify_convergence",
    "verify_structure",
]

#: The categories this report answers, in report order.  ``source_validity``
#: is deliberately absent: whether the *inputs* were usable is answered on the
#: diagnostics themselves (:mod:`vaft.validation.magnetics`), not by the
#: equilibrium that consumed them.
EQUILIBRIUM_CATEGORIES = (
    "verification",
    "diagnostic_fit",
    "physical_validity",
    "independent_validation",
)

_ELEMENTARY_CHARGE = 1.602176634e-19

#: Names a per-slice virial row must carry finite for the closure to be decided.
_VIRIAL_REQUIRED = ("s_1", "s_2", "s_3", "alpha", "B_pa", "beta_p", "li")
_VIRIAL_BOUNDS = {"beta_p": (0.0, 10.0), "li": (0.0, 3.0)}

#: Errors a provider raises when the data cannot feed it.  They are recorded
#: on the slice as ``not_available`` with the message; anything else is a bug
#: and propagates.
_UNFEEDABLE = (KeyError, IndexError, ValueError)


# ---------------------------------------------------------------------------
# vocabulary helpers
# ---------------------------------------------------------------------------

_SEVERITY = {
    ValidationStatus.FAIL: 4,
    ValidationStatus.WARN: 3,
    ValidationStatus.INDETERMINATE: 2,
    ValidationStatus.PASS: 1,
    ValidationStatus.NOT_AVAILABLE: 0,
}


def aggregate_status(statuses: Iterable[ValidationStatus | str]) -> ValidationStatus:
    """Summarize several statuses without collapsing the undecided ones.

    Worst wins among ``fail`` > ``warn`` > ``indeterminate``.  Otherwise the
    result is ``pass`` only when *every* status is a pass: a mix of ``pass``
    and ``not_available`` is ``indeterminate``, because part of the evidence
    was never produced, and all-``not_available`` stays ``not_available``.
    An empty collection is ``not_available``.
    """
    values = [ValidationStatus(status) for status in statuses]
    if not values:
        return ValidationStatus.NOT_AVAILABLE
    worst = max(values, key=_SEVERITY.__getitem__)
    if worst in (ValidationStatus.FAIL, ValidationStatus.WARN, ValidationStatus.INDETERMINATE):
        return worst
    if all(value is ValidationStatus.NOT_AVAILABLE for value in values):
        return ValidationStatus.NOT_AVAILABLE
    if all(value is ValidationStatus.PASS for value in values):
        return ValidationStatus.PASS
    return ValidationStatus.INDETERMINATE


def _graded(key: str, residual: float) -> ValidationStatus:
    """The registry's tolerance applied to a normalized residual."""
    spec = describe(key)
    if spec.tolerance is None:
        raise ValueError(f"{key} is graded by rule, not by tolerance")
    value = _float(residual)
    if not math.isfinite(value):
        return ValidationStatus.INDETERMINATE
    warn, fail = spec.tolerance
    if abs(value) <= warn:
        return ValidationStatus.PASS
    if abs(value) <= fail:
        return ValidationStatus.WARN
    return ValidationStatus.FAIL


def _float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _log_ratio(value: float, reference: float) -> float:
    """``ln(value / reference)`` for two positive quantities, else NaN."""
    value, reference = _float(value), _float(reference)
    if not (math.isfinite(value) and math.isfinite(reference)) or value <= 0 or reference <= 0:
        return math.nan
    return math.log(value / reference)


def _result(status: ValidationStatus, **fields: Any) -> dict[str, Any]:
    return {"status": str(status), **fields}


def _unavailable(reason: str, **fields: Any) -> dict[str, Any]:
    return _result(ValidationStatus.NOT_AVAILABLE, reason=reason, **fields)


def _json_safe(value: Any) -> Any:
    """Plain Python for ``json.dumps``; non-finite floats become ``None``."""
    if isinstance(value, ValidationStatus):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


# ---------------------------------------------------------------------------
# reading the equilibrium
# ---------------------------------------------------------------------------

def _slice_indices(ods: Any, time_slice: int | Sequence[int] | None) -> list[int]:
    count = path_count(ods, "equilibrium.time_slice")
    if time_slice is None:
        return list(range(count))
    requested = [int(time_slice)] if isinstance(time_slice, (int, np.integer)) else [int(i) for i in time_slice]
    for index in requested:
        if not 0 <= index < count:
            raise IndexError(f"time_slice {index} is out of range for {count} equilibrium time slices")
    return requested


def _slice_time(ods: Any, index: int) -> float:
    value = _float(path_value(ods, f"equilibrium.time_slice.{index}.time"))
    if not math.isfinite(value):
        times = path_value(ods, "equilibrium.time")
        if times is not None and index < len(times):
            value = _float(times[index])
    return value


def _array(ods: Any, path: str) -> np.ndarray | None:
    value = path_value(ods, path)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.size else None


def _psi_norm_1d(ods: Any, index: int) -> np.ndarray | None:
    """``profiles_1d.psi`` as psi_n in [0, 1], from the slice's own flux bounds."""
    root = f"equilibrium.time_slice.{index}"
    psi = _array(ods, f"{root}.profiles_1d.psi")
    axis = _float(path_value(ods, f"{root}.global_quantities.psi_axis"))
    boundary = _float(path_value(ods, f"{root}.global_quantities.psi_boundary"))
    if psi is None:
        return None
    if not (math.isfinite(axis) and math.isfinite(boundary)) or axis == boundary:
        axis, boundary = float(psi[0]), float(psi[-1])
        if axis == boundary:
            return None
    return (psi - axis) / (boundary - axis)


def _pressure_on(ods: Any, index: int, psi_norm: np.ndarray) -> np.ndarray | None:
    """The reconstructed pressure interpolated onto ``psi_norm`` points."""
    pressure = _array(ods, f"equilibrium.time_slice.{index}.profiles_1d.pressure")
    grid = _psi_norm_1d(ods, index)
    if pressure is None or grid is None or pressure.size != grid.size:
        return None
    order = np.argsort(grid)
    return np.interp(psi_norm, grid[order], pressure[order])


def _equilibrium_data(ods: Any, index: int):
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(ods, time_index=index)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_structure(equilibrium: Any, *, time_slice: int) -> dict[str, Any]:
    """Numerical and structural checks for one slice (#72 §1).

    Composes :func:`vaft.process.equilibrium.check_equilibrium_requirements`
    -- the algorithms' own precondition, which knows the grid/psi shape and
    the flux bounds -- with finiteness of the 2-D psi and the 1-D profiles,
    monotonicity of ``profiles_1d.psi``, the magnetic axis lying inside the
    LCFS box, and positive area and volume from
    :func:`vaft.process.equilibrium.derive_global_descriptors`.  Convention
    findings are left to :func:`verify_convention`.
    """
    from vaft.process.equilibrium import check_equilibrium_requirements, derive_global_descriptors

    index = int(time_slice)
    root = f"equilibrium.time_slice.{index}"
    try:
        data = _equilibrium_data(equilibrium, index)
    except _UNFEEDABLE as exc:
        return _unavailable(f"the slice cannot be read as an equilibrium: {exc}")

    report = check_equilibrium_requirements(data, required_for="general")
    issues = [item for item in report.issues if "cocos" not in item.code]
    codes = [item.code for item in issues]
    errors = any(item.severity == "error" for item in issues)

    psi = None if data.psi is None else np.asarray(data.psi, dtype=float)
    psi_finite = bool(psi is not None and np.isfinite(psi).all())
    if psi is not None and not psi_finite:
        codes.append("psi_non_finite")
        errors = True

    profiles_finite: dict[str, bool] = {}
    for name in ("psi", "pressure", "f", "q", "dpressure_dpsi", "f_df_dpsi"):
        profile = _array(equilibrium, f"{root}.profiles_1d.{name}")
        if profile is not None:
            profiles_finite[name] = bool(np.isfinite(profile).all())
    if profiles_finite and not all(profiles_finite.values()):
        codes.append("profile_non_finite")
        errors = True

    psi_1d = _array(equilibrium, f"{root}.profiles_1d.psi")
    monotonic = None
    if psi_1d is not None and psi_1d.size > 1:
        steps = np.diff(psi_1d)
        monotonic = bool((steps > 0).all() or (steps < 0).all())
        if not monotonic:
            codes.append("profiles_1d_psi_not_monotonic")

    axis_inside = None
    if data.magnetic_axis is not None and data.lcfs is not None and data.lcfs.r.size >= 3:
        r_axis, z_axis = data.magnetic_axis
        axis_inside = bool(
            data.lcfs.r.min() <= r_axis <= data.lcfs.r.max()
            and data.lcfs.z.min() <= z_axis <= data.lcfs.z.max()
        )
        if not axis_inside:
            codes.append("magnetic_axis_outside_lcfs")
            errors = True

    volume = area = math.nan
    if data.lcfs is not None and data.lcfs.r.size >= 3:
        descriptors = derive_global_descriptors(data).values
        volume = _float(descriptors["volume"].value)
        area = _float(descriptors["cross_section_area"].value)
        for name, value in (("volume", volume), ("cross_section_area", area)):
            if math.isfinite(value) and value <= 0:
                codes.append(f"{name}_not_positive")
                errors = True

    if errors:
        status = ValidationStatus.FAIL
    elif codes:
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS
    return _result(
        status,
        issues=codes,
        psi_finite=psi_finite,
        psi_range=[_float(data.psi_axis), _float(data.psi_boundary)],
        profiles_1d_psi_monotonic=monotonic,
        profiles_finite=profiles_finite,
        magnetic_axis_inside_lcfs=axis_inside,
        volume=volume,
        cross_section_area=area,
    )


def verify_continuity(equilibrium: Any, *, time_slice: int | Sequence[int] | None = None) -> dict[str, Any]:
    """Time-slice continuity across the selected slices (#72 §1).

    A whole-IDS check: the slice times must be finite and strictly increasing
    (``fail`` otherwise), and the plasma current should neither change sign
    nor step by more than half of its median between consecutive slices
    (``warn``).  Below two slices there is no continuity to assess.
    """
    indices = _slice_indices(equilibrium, time_slice)
    if len(indices) < 2:
        return _unavailable("continuity needs at least two time slices", slice_count=len(indices))
    times = np.array([_slice_time(equilibrium, index) for index in indices], dtype=float)
    ip = np.array(
        [_float(path_value(equilibrium, f"equilibrium.time_slice.{index}.global_quantities.ip")) for index in indices],
        dtype=float,
    )
    time_finite = bool(np.isfinite(times).all())
    time_monotonic = bool(time_finite and (np.diff(times) > 0).all())
    finite_ip = ip[np.isfinite(ip)]
    nonzero = finite_ip[finite_ip != 0]
    sign_consistent = bool(nonzero.size == 0 or (np.sign(nonzero) == np.sign(nonzero[0])).all())
    scale = float(np.median(np.abs(nonzero))) if nonzero.size else math.nan
    max_step = math.nan
    if finite_ip.size >= 2 and math.isfinite(scale) and scale > 0:
        max_step = float(np.max(np.abs(np.diff(finite_ip))) / scale)

    if not time_finite or not time_monotonic:
        status = ValidationStatus.FAIL
    elif not sign_consistent or (math.isfinite(max_step) and max_step > 0.5):
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS
    return _result(
        status,
        slice_count=len(indices),
        time_finite=time_finite,
        time_strictly_increasing=time_monotonic,
        ip_sign_consistent=sign_consistent,
        ip_max_relative_step=max_step,
        ip_non_finite_slices=int(np.count_nonzero(~np.isfinite(ip))),
    )


def verify_convention(equilibrium: Any, *, time_slice: int) -> dict[str, Any]:
    """COCOS sign relations for one slice, via :func:`vaft.process.cocos.validate_cocos`.

    A violated relation is a ``fail``; a violated q sign only is a ``warn``
    (codes commonly emit ``|q|``); an undeclared index, or one whose inputs are
    unavailable, is ``indeterminate`` -- the relations were not checked, which
    is not the same as their holding.
    """
    from vaft.process.cocos import validate_cocos

    index = int(time_slice)
    try:
        data = _equilibrium_data(equilibrium, index)
    except _UNFEEDABLE as exc:
        return _unavailable(f"the slice cannot be read as an equilibrium: {exc}")
    report = validate_cocos(data)
    codes = [item.code for item in report.issues]
    convention = data.convention
    if "cocos_undeclared" in codes:
        status = ValidationStatus.INDETERMINATE
    elif any(item.severity == "error" for item in report.issues):
        status = ValidationStatus.FAIL
    elif any(code not in ("cocos_unverifiable",) for code in codes):
        status = ValidationStatus.WARN
    elif codes:
        status = ValidationStatus.INDETERMINATE
    else:
        status = ValidationStatus.PASS
    return _result(
        status,
        issues=codes,
        cocos=convention.cocos,
        candidates=list(convention.candidates or ()),
        identified=list(convention.identified or ()),
        psi_per_radian=convention.psi_per_radian,
        source=convention.source,
    )


def verify_convergence(equilibrium: Any, *, time_slice: int) -> dict[str, Any]:
    """The solver's own convergence evidence for one slice (#70).

    Composes :func:`vaft.omas.efit_quality.convergence_metrics` with the
    native ``code.output_flag`` and ``convergence.result`` readers of
    :mod:`vaft.validation.imas`, keeping the three distinct.  ``fail`` when the
    run flag is negative, EFIT's verdict rejects the slice, or the final
    iteration error exceeds the acceptance tolerance; ``warn`` when it hit the
    iteration cap or stopped short of its exit tolerance; ``not_available``
    when no solver left any evidence at all.
    """
    from vaft.omas.efit_quality import convergence_metrics
    from vaft.validation.imas import read_convergence_result, read_output_flag

    index = int(time_slice)
    metrics = convergence_metrics(equilibrium, time_slice=index)
    flag = read_output_flag(equilibrium, time_slice=index)
    declared = read_convergence_result(equilibrium, time_slice=index)
    verdict, error, iterations = metrics["verdict"], metrics["error"], metrics["iterations"]
    accepted = verdict.get("accepted")
    final_error = _float(error.get("final_error"))
    fields = dict(
        output_flag=flag,
        declared_result=declared,
        accepted_by_solver=accepted,
        final_error=final_error,
        exit_tolerance=_float(error.get("exit_tolerance")),
        reached_exit_tolerance=bool(error.get("reached_exit_tolerance")),
        acceptance_tolerance=_float(error.get("acceptance_tolerance")),
        within_acceptance_tolerance=bool(error.get("within_acceptance_tolerance")),
        iterations=_float(iterations.get("iterations")),
        iteration_cap=_float(iterations.get("iteration_cap")),
        hit_iteration_cap=bool(iterations.get("hit_cap")),
        chi_squared_total=_float(error.get("chi_squared_total")),
    )
    if accepted is None and not math.isfinite(final_error) and flag is None and declared is None:
        return _unavailable("no solver left convergence evidence on this slice", **fields)
    reasons: list[str] = []
    if flag is not None and int(flag) < 0:
        reasons.append(f"code.output_flag {int(flag)} says the result must not be used")
    if accepted is False:
        reasons.append("the solver's own verdict rejected the slice")
    if math.isfinite(final_error) and not fields["within_acceptance_tolerance"]:
        reasons.append("the final iteration error exceeds the acceptance tolerance")
    if reasons:
        return _result(ValidationStatus.FAIL, reason="; ".join(reasons), **fields)
    if fields["hit_iteration_cap"]:
        reasons.append("the iteration cap was reached")
    if math.isfinite(final_error) and not fields["reached_exit_tolerance"]:
        reasons.append("the exit tolerance was not reached")
    if reasons:
        return _result(ValidationStatus.WARN, reason="; ".join(reasons), **fields)
    return _result(ValidationStatus.PASS, **fields)


# ---------------------------------------------------------------------------
# diagnostic fit
# ---------------------------------------------------------------------------

def validate_magnetic_fit(equilibrium: Any, *, time_slice: int) -> dict[str, dict[str, Any]]:
    """Constraint-fit assessment for one slice, per check (#72 §2, #69).

    Everything is read from :func:`vaft.omas.efit_quality.fit_quality_metrics`:
    the per-family normalized residual RMS, the scalar constraints' normalized
    residuals, and the reduced chi-square.  A family the solver only
    prescribed (no reconstructed values) is ``not_available``, not a pass.
    Returns ``{check: result}`` for the ``diagnostic_fit`` checks.
    """
    from vaft.omas.efit_quality import FAMILIES, fit_quality_metrics

    index = int(time_slice)
    metrics = fit_quality_metrics(equilibrium, time_slice=index)
    results: dict[str, dict[str, Any]] = {}
    for family, _title, _unit, _scale, _is_array in FAMILIES:
        entry = metrics["families"].get(family)
        channels = entry.get("channels") if entry else None
        count = len(channels) if isinstance(channels, (list, tuple)) else channels
        if entry is None or entry.get("fit_role") != "fitted":
            role = "absent" if entry is None else entry.get("fit_role")
            results[family] = _unavailable(f"{family} was {role}, not fitted", fit_role=role, channels=count)
            continue
        z_rms = _float(entry.get("z_rms"))
        results[family] = _result(
            _graded(f"diagnostic_fit.{family}", z_rms),
            fit_role="fitted",
            channels=count,
            z_rms=z_rms,
            z_bias=_float(entry.get("z_bias")),
            z_bias_significant=bool(entry.get("z_bias_significant")),
            z_abs_max=_float(entry.get("z_abs_max")),
            z_abs_max_channel=entry.get("z_abs_max_channel"),
            chi_squared_sum=_float(entry.get("chi_squared_sum")),
            outlier_fraction=_json_safe(entry.get("outlier_fraction")),
        )
    for name in ("ip", "diamagnetic_flux"):
        scalar = metrics["scalars"].get(name)
        if not scalar:
            results[name] = _unavailable(f"no reconstructed {name} constraint on this slice")
            continue
        measured, reconstructed = _float(scalar.get("measured")), _float(scalar.get("reconstructed"))
        z = _float(scalar.get("z"))
        results[name] = _result(
            _graded(f"diagnostic_fit.{name}", z),
            measured=measured,
            reconstructed=reconstructed,
            residual=measured - reconstructed,
            z=z,
            chi_squared=_float(scalar.get("chi_squared")),
            sigma_from_weight=_float(scalar.get("sigma_from_weight")),
        )
    fitted = int(metrics.get("fitted_channel_count") or 0)
    reduced = _float(metrics.get("chi_squared_reduced"))
    fields = dict(
        chi_squared_total=_float(metrics.get("chi_squared_total")),
        degrees_of_freedom=_float(metrics.get("degrees_of_freedom")),
        chi_squared_reduced=reduced,
        fitted_channel_count=fitted,
        chi_squared_share=_json_safe(metrics.get("chi_squared_share", {})),
    )
    if fitted == 0:
        results["global"] = _unavailable("no channel was fitted on this slice", **fields)
    else:
        results["global"] = _result(_graded("diagnostic_fit.global", reduced), **fields)
    return results


# ---------------------------------------------------------------------------
# physical validity
# ---------------------------------------------------------------------------

def _working_copy(equilibrium: Any, diagnostics: Any | None) -> Any:
    """A copy the mutating providers may write into, carrying the measured
    diamagnetic flux from ``diagnostics`` when the equilibrium ODS lacks it."""
    work = copy.deepcopy(equilibrium)
    if diagnostics is not None and diagnostics is not equilibrium:
        if not path_count(work, "magnetics.diamagnetic_flux") and path_count(diagnostics, "magnetics.diamagnetic_flux"):
            time = path_value(diagnostics, "magnetics.time")
            data = path_value(diagnostics, "magnetics.diamagnetic_flux.0.data")
            if time is not None and data is not None:
                work["magnetics.time"] = np.asarray(time, dtype=float)
                work["magnetics.diamagnetic_flux.0.data"] = np.asarray(data, dtype=float)
    return work


def _virial_rows(work: Any, indices: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Virial quantities per slice, from the ODS wrapper, on the working copy."""
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods

    rows: dict[int, dict[str, Any]] = {}
    for index in indices:
        try:
            rows[index] = compute_virial_equilibrium_quantities_ods(work, time_slice=index)[index]
        except _UNFEEDABLE as exc:
            rows[index] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return rows


def _diamagnetic_rows(work: Any, indices: Sequence[int]) -> dict[int, dict[str, Any] | None]:
    from vaft.omas.process_wrapper import compute_diamagnetic_flux_measured_vs_computed

    rows: dict[int, dict[str, Any] | None] = {}
    for index in indices:
        try:
            rows[index] = compute_diamagnetic_flux_measured_vs_computed(work, time_slice=index)[index]
        except _UNFEEDABLE:
            rows[index] = None
    return rows


def _virial_result(row: Mapping[str, Any]) -> dict[str, Any]:
    if "unavailable" in row:
        return _unavailable(row["unavailable"])
    values = {name: _float(row.get(name)) for name in (
        "s_1", "s_2", "s_3", "alpha", "B_pa", "V_p", "rt", "mui", "phi_dia_comp", "W_kin", "W_mag",
    )}
    lao, bongard = row.get("virial_lao", {}), row.get("virial_bongard", {})
    values.update(
        beta_p=_float(row.get("beta_p")),
        li=_float(row.get("li")),
        beta_p_lao=_float(lao.get("beta_p")),
        li_lao=_float(lao.get("li")),
        beta_p_bongard=_float(bongard.get("beta_p")),
        li_bongard=_float(bongard.get("li")),
        beta_p_diamagnetic=_float(row.get("beta_pd_vir")),
    )
    missing = [name for name in _VIRIAL_REQUIRED if not math.isfinite(values[name])]
    if missing:
        return _result(
            ValidationStatus.INDETERMINATE,
            reason=f"non-finite {', '.join(missing)}: near-singular denominator or missing boundary",
            **values,
        )
    outside = [
        name for name, (low, high) in _VIRIAL_BOUNDS.items()
        if not low < values[name] <= high
    ]
    if outside:
        return _result(
            ValidationStatus.FAIL,
            reason="outside plausibility bounds: " + ", ".join(
                f"{name}={values[name]:.3g} not in ({_VIRIAL_BOUNDS[name][0]}, {_VIRIAL_BOUNDS[name][1]}]"
                for name in outside
            ),
            **values,
        )
    return _result(ValidationStatus.PASS, **values)


def _pressure_consistency(descriptors: Mapping[str, Any] | None, virial: Mapping[str, Any]) -> dict[str, Any]:
    beta_p_virial = _float(virial.get("beta_p"))
    if descriptors is None or "beta_p_boundary_average" not in descriptors:
        return _unavailable("no LCFS-bounded pressure integral is available", beta_p_virial=beta_p_virial)
    entry = descriptors["beta_p_boundary_average"]
    beta_p_pressure = _float(entry.value)
    if not entry.available:
        return _unavailable(entry.reason or "the pressure-integral beta_p is unavailable", beta_p_virial=beta_p_virial)
    ratio = _log_ratio(beta_p_pressure, beta_p_virial)
    status = _graded("physical_validity.pressure_consistency", ratio)
    fields = dict(
        beta_p_pressure_integral=beta_p_pressure,
        beta_p_virial=beta_p_virial,
        ratio=(beta_p_pressure / beta_p_virial) if beta_p_virial else math.nan,
        log_ratio=ratio,
        pressure_integral=_float(descriptors["pressure_integral"].value),
        thermal_energy=_float(descriptors["thermal_energy"].value),
        W_kin_virial=_float(virial.get("W_kin")),
    )
    if status is ValidationStatus.INDETERMINATE:
        fields["reason"] = "one of the two beta_p values is non-finite or not positive"
    return _result(status, **fields)


def _q_profile(ods: Any, index: int) -> dict[str, Any]:
    root = f"equilibrium.time_slice.{index}"
    q = _array(ods, f"{root}.profiles_1d.q")
    if q is None:
        return _unavailable("no profiles_1d.q on this slice")
    finite = bool(np.isfinite(q).all())
    q0 = _float(path_value(ods, f"{root}.global_quantities.q_axis"))
    q95 = _float(path_value(ods, f"{root}.global_quantities.q_95"))
    psi_norm = _psi_norm_1d(ods, index)
    if not math.isfinite(q0):
        q0 = float(q[0])
    if not math.isfinite(q95) and psi_norm is not None and psi_norm.size == q.size:
        order = np.argsort(psi_norm)
        q95 = float(np.interp(0.95, psi_norm[order], q[order]))
    ordered = q
    if psi_norm is not None and psi_norm.size == q.size:
        ordered = q[np.argsort(psi_norm)]
    nonzero = q[np.isfinite(q)]
    sign_consistent = bool(nonzero.size and (nonzero != 0).all() and (np.sign(nonzero) == np.sign(nonzero[0])).all())
    # The share of axis-to-edge steps in which |q| does not fall: 1.0 is monotonic.
    fraction = float(np.mean(np.diff(np.abs(ordered)) >= 0)) if finite and q.size > 1 else math.nan
    if not finite:
        status, reason = ValidationStatus.FAIL, "profiles_1d.q has non-finite values"
    elif not sign_consistent:
        status, reason = ValidationStatus.FAIL, "profiles_1d.q changes sign or vanishes"
    elif math.isfinite(fraction) and fraction < 1.0:
        status, reason = ValidationStatus.WARN, "profiles_1d.q is not monotonic"
    else:
        status, reason = ValidationStatus.PASS, None
    fields = dict(q0=q0, q95=q95, q_edge=float(ordered[-1]), finite=finite, sign_consistent=sign_consistent,
                  non_decreasing_fraction=fraction, minimum_abs_q=float(np.min(np.abs(nonzero))) if nonzero.size else math.nan)
    return _result(status, **({"reason": reason} if reason else {}), **fields)


def _pressure_profile(ods: Any, index: int) -> dict[str, Any]:
    root = f"equilibrium.time_slice.{index}"
    pressure = _array(ods, f"{root}.profiles_1d.pressure")
    if pressure is None:
        return _unavailable("no profiles_1d.pressure on this slice")
    finite = bool(np.isfinite(pressure).all())
    psi_norm = _psi_norm_1d(ods, index)
    ordered = pressure
    if psi_norm is not None and psi_norm.size == pressure.size:
        ordered = pressure[np.argsort(psi_norm)]
    negative = float(np.mean(ordered < 0)) if finite else math.nan
    # The share of axis-to-edge steps in which p does not rise: 1.0 is non-increasing.
    fraction = float(np.mean(np.diff(ordered) <= 0)) if finite and ordered.size > 1 else math.nan
    reasons = []
    if not finite:
        status = ValidationStatus.FAIL
        reasons.append("profiles_1d.pressure has non-finite values")
    else:
        status = ValidationStatus.PASS
        if negative > 0:
            status = ValidationStatus.WARN
            reasons.append("profiles_1d.pressure is negative somewhere")
        if math.isfinite(fraction) and fraction < 1.0:
            status = ValidationStatus.WARN
            reasons.append("profiles_1d.pressure is not non-increasing from axis to edge")
    fields = dict(p_axis=float(ordered[0]), p_edge=float(ordered[-1]), finite=finite,
                  negative_fraction=negative, non_increasing_fraction=fraction)
    return _result(status, **({"reason": "; ".join(reasons)} if reasons else {}), **fields)


def _diamagnetic_flux(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return _unavailable("no magnetics.diamagnetic_flux measurement to compare against")
    measured, computed = _float(row.get("measured")), _float(row.get("computed"))
    if not math.isfinite(measured) or measured == 0.0:
        return _result(ValidationStatus.INDETERMINATE, reason="the measured diamagnetic flux is zero or non-finite",
                       measured=measured, computed=computed)
    relative = _float(row.get("relative_error"))
    return _result(
        _graded("physical_validity.diamagnetic_flux", relative),
        time=_float(row.get("time")),
        measured=measured,
        computed=computed,
        difference=_float(row.get("difference")),
        relative_error=relative,
        sign_agreement=bool(np.sign(measured) == np.sign(computed)) if math.isfinite(computed) else None,
    )


def _descriptors(ods: Any, index: int) -> Mapping[str, Any] | None:
    from vaft.process.equilibrium import derive_global_descriptors

    try:
        return derive_global_descriptors(_equilibrium_data(ods, index)).values
    except _UNFEEDABLE:
        return None


def _physical_slice(ods: Any, index: int, virial_row: Mapping[str, Any],
                    dia_row: Mapping[str, Any] | None, descriptors: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        "virial": _virial_result(virial_row),
        "pressure_consistency": _pressure_consistency(descriptors, virial_row),
        "q_profile": _q_profile(ods, index),
        "pressure_profile": _pressure_profile(ods, index),
        "diamagnetic_flux": _diamagnetic_flux(dia_row),
    }


def validate_physical(equilibrium: Any, *, time_slice: int, diagnostics: Any | None = None) -> dict[str, dict[str, Any]]:
    """Force-balance and plausibility checks for one slice (#72 §3).

    The virial quantities come from
    :func:`vaft.omas.process_wrapper.compute_virial_equilibrium_quantities_ods`,
    run on a deep copy because it refreshes boundary geometry in its argument;
    ``beta_p``/``li`` are the Lao closure, with Bongard's alongside under their
    own names rather than mixed in.  The pressure-integral beta_p it is
    compared against comes from
    :func:`vaft.process.equilibrium.derive_global_descriptors`.  Returns
    ``{check: result}`` for the ``physical_validity`` checks.
    """
    index = int(time_slice)
    work = _working_copy(equilibrium, diagnostics)
    return _physical_slice(
        equilibrium, index, _virial_rows(work, [index])[index], _diamagnetic_rows(work, [index])[index],
        _descriptors(equilibrium, index),
    )


# ---------------------------------------------------------------------------
# independent validation
# ---------------------------------------------------------------------------

def _nearest(times: np.ndarray, target: float) -> tuple[int, float]:
    offsets = np.abs(np.asarray(times, dtype=float) - target)
    position = int(np.nanargmin(offsets))
    return position, float(offsets[position])


def _time_tolerance(ods: Any) -> float:
    times = _array(ods, "equilibrium.time")
    if times is None or times.size < 2:
        return 1.0e-3
    return max(0.5 * float(np.median(np.diff(np.sort(times)))), 1.0e-3)


def _compare_pressure(key: str, kinetic: np.ndarray, reconstructed: np.ndarray, coverage: str) -> tuple[ValidationStatus, dict[str, Any]]:
    """Grade a kinetic pressure sample against the reconstructed one.

    The headline measure is the log-ratio of the two sums over the sampled
    points.  Electron-only coverage is one-sided: electrons alone exceeding
    the reconstructed total is a failure of the reconstruction, but falling
    short of it is at most a warning, since the ions are unmeasured.
    """
    from vaft.formula.statistics import pearson_correlation, rms

    ratio = _log_ratio(float(np.sum(kinetic)), float(np.sum(reconstructed)))
    status = _graded(key, ratio)
    if coverage == "electrons" and math.isfinite(ratio) and ratio < 0 and status is ValidationStatus.FAIL:
        status = ValidationStatus.WARN
    scale = float(np.max(np.abs(kinetic))) if kinetic.size else math.nan
    fields = dict(
        coverage=coverage,
        points=int(kinetic.size),
        log_ratio=ratio,
        sum_ratio=math.exp(ratio) if math.isfinite(ratio) else math.nan,
        normalized_rms=(_float(rms(reconstructed - kinetic)) / scale) if scale else math.nan,
        correlation=_float(pearson_correlation(kinetic, reconstructed)) if kinetic.size > 2 else math.nan,
    )
    if status is ValidationStatus.INDETERMINATE:
        fields["reason"] = "the sampled pressures are not both positive"
    return status, fields


def _kinetic_pressure(ods: Any, index: int, profiles: Any) -> dict[str, Any]:
    key = "independent_validation.kinetic_pressure"
    count = path_count(profiles, "core_profiles.profiles_1d")
    if count == 0:
        return _unavailable("no core_profiles.profiles_1d to compare against")
    target = _slice_time(ods, index)
    times = _array(profiles, "core_profiles.time")
    if times is None or times.size != count:
        times = np.array([_float(path_value(profiles, f"core_profiles.profiles_1d.{j}.time")) for j in range(count)])
    if not math.isfinite(target) or not np.isfinite(times).any():
        return _unavailable("no finite times to match a core_profiles slice on")
    position, offset = _nearest(times, target)
    tolerance = _time_tolerance(ods)
    if offset > tolerance:
        return _unavailable(f"the nearest core_profiles slice is {offset:.4g} s away, beyond {tolerance:.4g} s",
                            time_offset=offset)
    root = f"core_profiles.profiles_1d.{position}"
    kinetic, coverage = None, None
    for path, label in (("pressure_thermal", "thermal_total"),
                        ("electrons.pressure_thermal", "electrons"), ("electrons.pressure", "electrons")):
        kinetic = _array(profiles, f"{root}.{path}")
        if kinetic is not None:
            coverage = label
            break
    if kinetic is None:
        return _unavailable("the core_profiles slice carries no thermal pressure", time_offset=offset)
    psi_norm, coordinate = None, None
    rho = _array(profiles, f"{root}.grid.rho_tor_norm")
    if rho is not None and rho.size == kinetic.size:
        eq_rho = _array(ods, f"equilibrium.time_slice.{index}.profiles_1d.rho_tor_norm")
        eq_psi = _psi_norm_1d(ods, index)
        if eq_rho is not None and eq_psi is not None and eq_rho.size == eq_psi.size:
            order = np.argsort(eq_rho)
            psi_norm, coordinate = np.interp(rho, eq_rho[order], eq_psi[order]), "rho_tor_norm"
    if psi_norm is None:
        rho_pol = _array(profiles, f"{root}.grid.rho_pol_norm")
        if rho_pol is not None and rho_pol.size == kinetic.size:
            psi_norm, coordinate = rho_pol ** 2, "rho_pol_norm"
    if psi_norm is None:
        psi = _array(profiles, f"{root}.grid.psi")
        if psi is not None and psi.size == kinetic.size:
            axis = _float(path_value(profiles, f"{root}.grid.psi_magnetic_axis"))
            boundary = _float(path_value(profiles, f"{root}.grid.psi_boundary"))
            if not (math.isfinite(axis) and math.isfinite(boundary)):
                axis = _float(path_value(ods, f"equilibrium.time_slice.{index}.global_quantities.psi_axis"))
                boundary = _float(path_value(ods, f"equilibrium.time_slice.{index}.global_quantities.psi_boundary"))
            if math.isfinite(axis) and math.isfinite(boundary) and axis != boundary:
                psi_norm, coordinate = (psi - axis) / (boundary - axis), "psi"
    if psi_norm is None:
        return _unavailable("the core_profiles grid carries no coordinate the equilibrium can map", time_offset=offset)
    inside = np.isfinite(psi_norm) & np.isfinite(kinetic) & (psi_norm >= 0) & (psi_norm <= 1)
    reconstructed = _pressure_on(ods, index, psi_norm[inside])
    if reconstructed is None or not inside.any():
        return _unavailable("the reconstructed pressure cannot be sampled on the kinetic grid", time_offset=offset)
    status, fields = _compare_pressure(key, kinetic[inside], reconstructed, coverage)
    return _result(status, time_offset=offset, coordinate=coordinate, **fields)


def _thomson_pressure(ods: Any, index: int, diagnostics: Any) -> dict[str, Any]:
    from scipy.interpolate import RectBivariateSpline

    key = "independent_validation.thomson_pressure"
    channels = path_count(diagnostics, "thomson_scattering.channel")
    if channels == 0:
        return _unavailable("no thomson_scattering channels to compare against")
    target = _slice_time(ods, index)
    times = _array(diagnostics, "thomson_scattering.time")
    if times is None:
        times = _array(diagnostics, "thomson_scattering.channel.0.n_e.time")
    if times is None or not math.isfinite(target):
        return _unavailable("no thomson_scattering time base to match on")
    position, offset = _nearest(times, target)
    tolerance = _time_tolerance(ods)
    if offset > tolerance:
        return _unavailable(f"the nearest Thomson time is {offset:.4g} s away, beyond {tolerance:.4g} s",
                            time_offset=offset)
    root = f"equilibrium.time_slice.{index}"
    r_grid, z_grid = _array(ods, f"{root}.profiles_2d.0.grid.dim1"), _array(ods, f"{root}.profiles_2d.0.grid.dim2")
    psi_2d = _array(ods, f"{root}.profiles_2d.0.psi")
    axis = _float(path_value(ods, f"{root}.global_quantities.psi_axis"))
    boundary = _float(path_value(ods, f"{root}.global_quantities.psi_boundary"))
    if r_grid is None or z_grid is None or psi_2d is None or not (math.isfinite(axis) and math.isfinite(boundary)) or axis == boundary:
        return _unavailable("the slice has no 2-D psi grid to map channel positions through", time_offset=offset)
    if psi_2d.shape == (z_grid.size, r_grid.size) and r_grid.size != z_grid.size:
        psi_2d = psi_2d.T
    spline = RectBivariateSpline(r_grid, z_grid, psi_2d)
    points, pressures = [], []
    for channel in range(channels):
        base = f"thomson_scattering.channel.{channel}"
        r = _float(path_value(diagnostics, f"{base}.position.r"))
        z = _float(path_value(diagnostics, f"{base}.position.z"))
        n_e, t_e = _array(diagnostics, f"{base}.n_e.data"), _array(diagnostics, f"{base}.t_e.data")
        if not (math.isfinite(r) and math.isfinite(z)) or n_e is None or t_e is None:
            continue
        if position >= n_e.size or position >= t_e.size:
            continue
        density, temperature = float(n_e[position]), float(t_e[position])
        if not (math.isfinite(density) and math.isfinite(temperature)) or density <= 0 or temperature <= 0:
            continue
        psi_norm = (float(spline(r, z)[0, 0]) - axis) / (boundary - axis)
        if 0.0 <= psi_norm <= 1.0:
            points.append(psi_norm)
            pressures.append(density * temperature * _ELEMENTARY_CHARGE)
    if not points:
        return _unavailable("no Thomson channel with finite n_e and T_e lies inside the LCFS at this time",
                            time_offset=offset)
    reconstructed = _pressure_on(ods, index, np.asarray(points))
    if reconstructed is None:
        return _unavailable("the reconstructed pressure cannot be sampled at the channel positions", time_offset=offset)
    status, fields = _compare_pressure(key, np.asarray(pressures), reconstructed, "electrons")
    return _result(status, time_offset=offset, channels_inside=len(points), **fields)


def _diamagnetic_energy(ods: Any, index: int, virial: Mapping[str, Any], dia_row: Mapping[str, Any] | None,
                        descriptors: Mapping[str, Any] | None) -> dict[str, Any]:
    from vaft.formula.equilibrium import kinetic_energy_from_beta_p_B_pa_V_p, virial_beta_pd_from_S_mu_rt
    from vaft.process.equilibrium import computed_diamagnetism_from_phi

    if dia_row is None or not math.isfinite(_float(dia_row.get("measured"))):
        return _unavailable("no measured diamagnetic flux at this slice")
    measured = _float(dia_row["measured"])
    needed = {name: _float(virial.get(name)) for name in ("s_1", "s_2", "B_pa", "V_p", "rt", "W_kin")}
    b0 = _array(ods, "equilibrium.vacuum_toroidal_field.b0")
    b_t0 = float(b0[index] if index < b0.size else b0[0]) if b0 is not None else math.nan
    if not math.isfinite(b_t0):
        b_t0 = _float(path_value(ods, f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis.b_field_tor"))
    r_0 = _float(descriptors["major_radius"].value) if descriptors and "major_radius" in descriptors else math.nan
    missing = [name for name, value in {**needed, "B_t0": b_t0, "R_0": r_0}.items() if not math.isfinite(value)]
    if missing or needed["V_p"] <= 0 or needed["B_pa"] <= 0 or r_0 == 0:
        return _result(ValidationStatus.INDETERMINATE,
                       reason=f"the virial inputs are not decided: {', '.join(missing) or 'non-positive volume, field or radius'}",
                       measured_flux=measured)
    mui = _float(computed_diamagnetism_from_phi(measured, b_t0, r_0, needed["V_p"], needed["B_pa"]))
    beta_pd = _float(virial_beta_pd_from_S_mu_rt(needed["s_1"], needed["s_2"], mui, needed["rt"] / r_0))
    fields = dict(measured_flux=measured, B_t0=b_t0, R_0=r_0, mui_measured=mui, beta_p_diamagnetic=beta_pd,
                  W_kin_virial=needed["W_kin"],
                  thermal_energy=_float(descriptors["thermal_energy"].value) if descriptors and "thermal_energy" in descriptors else math.nan)
    if not math.isfinite(beta_pd) or beta_pd <= 0:
        return _result(ValidationStatus.INDETERMINATE, reason="the diamagnetic beta_p is not positive", **fields)
    w_dia = _float(kinetic_energy_from_beta_p_B_pa_V_p(beta_pd, needed["B_pa"], needed["V_p"]))
    ratio = _log_ratio(w_dia, needed["W_kin"])
    return _result(_graded("independent_validation.diamagnetic_energy", ratio),
                   W_diamagnetic=w_dia, log_ratio=ratio, **fields)


def _independent_slice(ods: Any, index: int, *, kinetic_profiles: Any | None, diagnostics: Any | None,
                       virial_row: Mapping[str, Any], dia_row: Mapping[str, Any] | None,
                       descriptors: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    profiles = kinetic_profiles if kinetic_profiles is not None else ods
    thomson = diagnostics if diagnostics is not None else ods
    return {
        "kinetic_pressure": _kinetic_pressure(ods, index, profiles),
        "thomson_pressure": _thomson_pressure(ods, index, thomson),
        "diamagnetic_energy": _diamagnetic_energy(ods, index, virial_row, dia_row, descriptors),
    }


def validate_independent(equilibrium: Any, *, time_slice: int, kinetic_profiles: Any | None = None,
                         diagnostics: Any | None = None) -> dict[str, dict[str, Any]]:
    """Agreement with measurements the reconstruction never used, one slice (#72 §4).

    ``kinetic_profiles`` is an ODS carrying ``core_profiles`` (the equilibrium
    ODS itself when omitted); ``diagnostics`` one carrying
    ``thomson_scattering`` and ``magnetics.diamagnetic_flux``.  Returns
    ``{check: result}`` for the ``independent_validation`` checks.
    """
    index = int(time_slice)
    work = _working_copy(equilibrium, diagnostics)
    return _independent_slice(
        equilibrium, index, kinetic_profiles=kinetic_profiles, diagnostics=diagnostics,
        virial_row=_virial_rows(work, [index])[index], dia_row=_diamagnetic_rows(work, [index])[index],
        descriptors=_descriptors(equilibrium, index),
    )


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _collect(indices: Sequence[int], times: Sequence[float], per_slice: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Fold per-slice results of one check into its report entry."""
    slices = []
    for index, time in zip(indices, times):
        entry = {"time_slice": int(index), "time": float(time)}
        entry.update(per_slice[index])
        slices.append(entry)
    statuses = [entry["status"] for entry in slices]
    return {
        "status": str(aggregate_status(statuses)),
        "counts": dict(sorted(Counter(statuses).items())),
        "slices": slices,
    }


def _provenance(ods: Any, indices: Sequence[int], *, diagnostics: Any | None, kinetic_profiles: Any | None) -> dict[str, Any]:
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
    from vaft.omas.general import ods_cocos

    identified: list[int] = []
    psi_per_radian = None
    if indices:
        try:
            convention = _equilibrium_data(ods, indices[0]).convention
            identified = list(convention.identified or ())
            psi_per_radian = convention.psi_per_radian
        except _UNFEEDABLE:
            pass
    try:
        factor = float(ods_psi_to_wb_per_radian_factor(ods, indices[0] if indices else 0))
    except Exception:  # the factor is a best-effort declaration, never a gate
        factor = math.nan
    return {
        "equilibrium": {
            "code": path_value(ods, "equilibrium.code.name"),
            "code_version": path_value(ods, "equilibrium.code.version"),
            "time_slice_count": path_count(ods, "equilibrium.time_slice"),
            "homogeneous_time": path_value(ods, "equilibrium.ids_properties.homogeneous_time"),
        },
        "conventions": {
            "cocos_declared": ods_cocos(ods),
            "cocos_identified": identified,
            "psi_per_radian": psi_per_radian,
            "psi_to_wb_per_radian_factor": factor,
        },
        "methods": {
            "boundary_normalization": {
                "function": "vaft.process.equilibrium.prepare_boundary_for_shafranov",
                "n_points": 256,
                "enforce_ccw": True,
            },
            "cell_weighting": {
                "function": "vaft.process.equilibrium.fractional_cell_weights_from_boundary",
                "samples_per_axis": 5,
            },
            "poloidal_field": "profiles_2d.0.b_field_r/z when present, else grad(psi)/R on the 2-D grid",
            "virial_closure": "lao",
            "settings_of": "vaft.omas.process_wrapper.compute_virial_equilibrium_quantities_ods",
        },
        "inputs": {
            "diagnostics": diagnostics is not None,
            "kinetic_profiles": kinetic_profiles is not None,
        },
        "providers": sorted({spec.provider for spec in CHECKS.values()}),
    }


def validate_equilibrium(
    equilibrium: Any,
    *,
    diagnostics: Any | None = None,
    kinetic_profiles: Any | None = None,
    checks: Iterable[str] = EQUILIBRIUM_CATEGORIES,
    time_slice: int | Sequence[int] | None = None,
) -> dict[str, Any]:
    """The unified equilibrium validation report (#72).

    Parameters
    ----------
    equilibrium
        An ODS carrying ``equilibrium``; never mutated.
    diagnostics
        Optional ODS carrying ``magnetics.diamagnetic_flux`` and
        ``thomson_scattering`` when they are not on ``equilibrium`` itself.
    kinetic_profiles
        Optional ODS carrying ``core_profiles``.
    checks
        Categories to assess, any of :data:`EQUILIBRIUM_CATEGORIES`.  A
        category not requested is absent from the report, which is distinct
        from being ``not_available``.
    time_slice
        One index, several, or ``None`` for every slice.

    Returns
    -------
    dict
        JSON-serializable and deterministic; see the module docstring for the
        shape.  Each category is separately queryable under its own key, and
        ``summary`` carries one status per category with
        :func:`aggregate_status` semantics.  ``registry.describe`` answers what
        each check's numbers mean.
    """
    selected = tuple(dict.fromkeys((checks,) if isinstance(checks, str) else checks))
    unknown = sorted(set(selected) - set(EQUILIBRIUM_CATEGORIES))
    if unknown:
        raise ValueError(f"unknown validation categories {unknown}; choose from {EQUILIBRIUM_CATEGORIES}")
    indices = _slice_indices(equilibrium, time_slice)
    times = [_slice_time(equilibrium, index) for index in indices]
    report: dict[str, Any] = {"schema_version": 1}
    provenance = _provenance(equilibrium, indices, diagnostics=diagnostics, kinetic_profiles=kinetic_profiles)

    if not indices:
        report["status"] = str(ValidationStatus.NOT_AVAILABLE)
        report["summary"] = {category: str(ValidationStatus.NOT_AVAILABLE) for category in selected}
        report["reason"] = "the ODS carries no equilibrium time slice"
        report.update(provenance=provenance, time_slices=[], time=[])
        for category in selected:
            report[category] = {}
        return _json_safe(report)

    needs_physics = any(category in selected for category in ("physical_validity", "independent_validation"))
    virial_rows: dict[int, dict[str, Any]] = {}
    dia_rows: dict[int, dict[str, Any] | None] = {}
    descriptors: dict[int, Mapping[str, Any] | None] = {}
    if needs_physics:
        work = _working_copy(equilibrium, diagnostics)
        virial_rows = _virial_rows(work, indices)
        dia_rows = _diamagnetic_rows(work, indices)
        descriptors = {index: _descriptors(equilibrium, index) for index in indices}

    categories: dict[str, dict[str, Any]] = {}
    if "verification" in selected:
        categories["verification"] = {
            "structure": _collect(indices, times, {i: verify_structure(equilibrium, time_slice=i) for i in indices}),
            "continuity": verify_continuity(equilibrium, time_slice=indices),
            "convention": _collect(indices, times, {i: verify_convention(equilibrium, time_slice=i) for i in indices}),
            "convergence": _collect(indices, times, {i: verify_convergence(equilibrium, time_slice=i) for i in indices}),
        }
    if "diagnostic_fit" in selected:
        per_slice = {i: validate_magnetic_fit(equilibrium, time_slice=i) for i in indices}
        categories["diagnostic_fit"] = {
            spec.key.split(".", 1)[1]: _collect(indices, times, {i: per_slice[i][spec.key.split(".", 1)[1]] for i in indices})
            for spec in checks_in("diagnostic_fit")
        }
    if "physical_validity" in selected:
        per_slice = {i: _physical_slice(equilibrium, i, virial_rows[i], dia_rows[i], descriptors[i]) for i in indices}
        categories["physical_validity"] = {
            spec.key.split(".", 1)[1]: _collect(indices, times, {i: per_slice[i][spec.key.split(".", 1)[1]] for i in indices})
            for spec in checks_in("physical_validity")
        }
    if "independent_validation" in selected:
        per_slice = {
            i: _independent_slice(equilibrium, i, kinetic_profiles=kinetic_profiles, diagnostics=diagnostics,
                                  virial_row=virial_rows[i], dia_row=dia_rows[i], descriptors=descriptors[i])
            for i in indices
        }
        categories["independent_validation"] = {
            spec.key.split(".", 1)[1]: _collect(indices, times, {i: per_slice[i][spec.key.split(".", 1)[1]] for i in indices})
            for spec in checks_in("independent_validation")
        }

    summary = {
        category: str(aggregate_status(entry["status"] for entry in results.values()))
        for category, results in categories.items()
    }
    report["status"] = str(aggregate_status(summary.values()))
    report["summary"] = summary
    report["provenance"] = provenance
    report["time_slices"] = [int(index) for index in indices]
    report["time"] = [float(time) for time in times]
    report.update(categories)
    return _json_safe(report)


def status_summary(report: Mapping[str, Any]) -> dict[str, ValidationStatus]:
    """One :class:`ValidationStatus` per category present in ``report``."""
    return {category: ValidationStatus(status) for category, status in report.get("summary", {}).items()}
