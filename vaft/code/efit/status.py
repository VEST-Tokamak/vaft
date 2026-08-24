"""Structured, JSON-compatible EFIT slice status and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EFIT_FAILURE_CODES = (
    "runtime_error",
    "timeout",
    "nonzero_returncode",
    "missing_kfile",
    "missing_gfile",
    "missing_afile",
    "missing_mfile",
    "parse_error",
    "nonconverged",
    "invalid_boundary",
    "invalid_flux_ordering",
    "negative_pressure",
    "negative_stored_energy",
    "invalid_li",
    "invalid_beta",
    "invalid_q",
    "invalid_volume",
    "nonfinite_profile",
    "excessive_diagnostic_residual",
    "temporal_discontinuity",
)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


@dataclass(frozen=True)
class EFITValidationConfig:
    """Thresholds used to decide whether an EFIT slice is scientifically usable."""

    require_afile: bool = False
    require_mfile: bool = False
    pressure_negative_tolerance: float = 0.0
    stored_energy_negative_tolerance: float = 0.0
    li_range: tuple[float, float] = (0.0, 10.0)
    beta_range: tuple[float, float] = (0.0, 10.0)
    q_absolute_range: tuple[float, float] = (0.0, 100.0)
    volume_range: tuple[float, float] = (0.0, 100.0)
    minimum_boundary_area: float = 1e-8
    maximum_axis_step: float | None = 0.2
    maximum_diagnostic_residual: float | None = None


@dataclass(frozen=True)
class EFITSliceStatus:
    """Independent runtime, output, numerical, and physical status for one slice."""

    shot: int
    time: float
    runtime_ok: bool
    output_ok: bool
    numerical_ok: bool
    physical_ok: bool
    overall_status: str
    failure_codes: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return (
            self.runtime_ok
            and self.output_ok
            and self.numerical_ok
            and self.physical_ok
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return _json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EFITSliceStatus":
        """Restore a status serialized by :meth:`to_dict`."""
        values = dict(payload)
        values["failure_codes"] = tuple(values.get("failure_codes", ()))
        return cls(**values)


def _path_exists(path: str | Path | None) -> bool:
    return path is not None and Path(path).is_file()


def _finite_scalar(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _polygon_area_and_centroid_r(r: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    r_next = np.roll(r, -1)
    z_next = np.roll(z, -1)
    cross = r * z_next - r_next * z
    signed_area = 0.5 * float(np.sum(cross))
    area = abs(signed_area)
    if area == 0.0:
        return 0.0, float("nan")
    centroid_r = float(np.sum((r + r_next) * cross) / (6.0 * signed_area))
    return area, centroid_r


def _overall_status(
    runtime_ok: bool,
    output_ok: bool,
    numerical_ok: bool,
    physical_ok: bool,
) -> str:
    if not runtime_ok:
        return "runtime_failed"
    if not output_ok:
        return "output_failed"
    if not numerical_ok:
        return "numerical_failed"
    if not physical_ok:
        return "physical_failed"
    return "usable"


def validate_efit_slice(
    *,
    shot: int,
    time: float,
    runtime_status: str,
    returncode: int | None,
    kfile: str | Path | None,
    gfile: str | Path | None,
    afile: str | Path | None = None,
    mfile: str | Path | None = None,
    geqdsk: Mapping[str, Any] | None = None,
    parse_error: str | None = None,
    converged: bool | None = None,
    metrics: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
    config: EFITValidationConfig | None = None,
) -> EFITSliceStatus:
    """Validate one EFIT attempt without conflating process and science status."""
    cfg = config or EFITValidationConfig()
    failures: list[str] = []
    measured = {
        key: float(value)
        for key, value in (metrics or {}).items()
        if _finite_scalar(value) is not None
    }

    runtime_ok = runtime_status in {"completed", "collected"} and returncode in {
        None,
        0,
    }
    if runtime_status == "timeout":
        failures.extend(("runtime_error", "timeout"))
    elif not runtime_ok:
        failures.append("runtime_error")
    if returncode not in {None, 0}:
        failures.append("nonzero_returncode")

    if not _path_exists(kfile):
        failures.append("missing_kfile")
    if not _path_exists(gfile):
        failures.append("missing_gfile")
    if cfg.require_afile and not _path_exists(afile):
        failures.append("missing_afile")
    if cfg.require_mfile and not _path_exists(mfile):
        failures.append("missing_mfile")
    if parse_error:
        failures.append("parse_error")
    output_ok = not any(
        code in failures
        for code in (
            "missing_kfile",
            "missing_gfile",
            "missing_afile",
            "missing_mfile",
            "parse_error",
        )
    )

    numerical_ok = output_ok
    physical_ok = output_ok
    if converged is False:
        failures.append("nonconverged")
        numerical_ok = False

    if geqdsk is not None:
        psi_axis = _finite_scalar(geqdsk.get("SIMAG"))
        psi_boundary = _finite_scalar(geqdsk.get("SIBRY"))
        axis_r = _finite_scalar(geqdsk.get("RMAXIS"))
        axis_z = _finite_scalar(geqdsk.get("ZMAXIS"))
        plasma_current = _finite_scalar(geqdsk.get("CURRENT"))
        for name, value in (
            ("psi_axis", psi_axis),
            ("psi_boundary", psi_boundary),
            ("magnetic_axis_r", axis_r),
            ("magnetic_axis_z", axis_z),
            ("plasma_current", plasma_current),
        ):
            if value is not None:
                measured[name] = value
        if (
            psi_axis is None
            or psi_boundary is None
            or np.isclose(psi_axis, psi_boundary)
        ):
            failures.append("invalid_flux_ordering")
            numerical_ok = False

        boundary_r = np.asarray(geqdsk.get("RBBBS", []), dtype=float)
        boundary_z = np.asarray(geqdsk.get("ZBBBS", []), dtype=float)
        boundary_valid = (
            boundary_r.size >= 3
            and boundary_r.shape == boundary_z.shape
            and np.all(np.isfinite(boundary_r))
            and np.all(np.isfinite(boundary_z))
        )
        if boundary_valid:
            area, centroid_r = _polygon_area_and_centroid_r(boundary_r, boundary_z)
            volume = 2.0 * np.pi * centroid_r * area
            measured["boundary_area"] = area
            measured["volume"] = volume
            boundary_valid = (
                area > cfg.minimum_boundary_area
                and np.isfinite(centroid_r)
                and centroid_r > 0.0
            )
        if not boundary_valid or axis_r is None or axis_z is None:
            failures.append("invalid_boundary")
            numerical_ok = False

        profile_arrays = {
            name: np.asarray(geqdsk.get(name, []), dtype=float)
            for name in ("PRES", "PPRIME", "FFPRIM", "QPSI")
        }
        if any(values.size and not np.all(np.isfinite(values)) for values in profile_arrays.values()):
            failures.append("nonfinite_profile")
            numerical_ok = False
        pressure = profile_arrays["PRES"]
        if pressure.size:
            finite_pressure = pressure[np.isfinite(pressure)]
            if finite_pressure.size:
                measured["pressure_min"] = float(np.min(finite_pressure))
                measured["pressure_max"] = float(np.max(finite_pressure))
                if measured["pressure_min"] < -cfg.pressure_negative_tolerance:
                    failures.append("negative_pressure")
                    physical_ok = False
        q_values = np.abs(profile_arrays["QPSI"])
        finite_q = q_values[np.isfinite(q_values)]
        if finite_q.size:
            measured["q_absolute_min"] = float(np.min(finite_q))
            measured["q_absolute_max"] = float(np.max(finite_q))
            if (
                measured["q_absolute_min"] <= cfg.q_absolute_range[0]
                or measured["q_absolute_max"] > cfg.q_absolute_range[1]
            ):
                failures.append("invalid_q")
                physical_ok = False

    stored_energy = measured.get("stored_energy")
    if (
        stored_energy is not None
        and stored_energy < -cfg.stored_energy_negative_tolerance
    ):
        failures.append("negative_stored_energy")
        physical_ok = False
    for name, failure_code, limits in (
        ("li", "invalid_li", cfg.li_range),
        ("beta", "invalid_beta", cfg.beta_range),
        ("volume", "invalid_volume", cfg.volume_range),
    ):
        value = measured.get(name)
        if value is not None and not limits[0] < value <= limits[1]:
            failures.append(failure_code)
            physical_ok = False
    residual = measured.get("diagnostic_residual")
    if (
        residual is not None
        and cfg.maximum_diagnostic_residual is not None
        and residual > cfg.maximum_diagnostic_residual
    ):
        failures.append("excessive_diagnostic_residual")
        physical_ok = False

    if not output_ok:
        numerical_ok = False
        physical_ok = False
    failure_codes = tuple(dict.fromkeys(failures))
    details = dict(provenance or {})
    details.update(
        {
            "runtime_status": runtime_status,
            "returncode": returncode,
            "runtime": {
                "executable_resolved": bool((provenance or {}).get("executable")),
                "process_started": runtime_status in {"completed", "timeout"},
                "timed_out": runtime_status == "timeout",
                "returncode": returncode,
            },
            "converged": converged,
            "outputs": {
                "kfile": str(kfile) if kfile is not None else None,
                "gfile": str(gfile) if gfile is not None else None,
                "afile": str(afile) if afile is not None else None,
                "mfile": str(mfile) if mfile is not None else None,
            },
            "optional_outputs": {"afile": not cfg.require_afile, "mfile": not cfg.require_mfile},
        }
    )
    return EFITSliceStatus(
        shot=int(shot),
        time=float(time),
        runtime_ok=runtime_ok,
        output_ok=output_ok,
        numerical_ok=numerical_ok,
        physical_ok=physical_ok,
        overall_status=_overall_status(
            runtime_ok, output_ok, numerical_ok, physical_ok
        ),
        failure_codes=failure_codes,
        metrics=measured,
        provenance=_json_compatible(details),
    )


def apply_temporal_continuity(
    statuses: Sequence[EFITSliceStatus],
    config: EFITValidationConfig | None = None,
) -> tuple[EFITSliceStatus, ...]:
    """Mark axis jumps without discarding otherwise successful neighboring slices."""
    cfg = config or EFITValidationConfig()
    ordered = sorted(statuses, key=lambda item: item.time)
    if cfg.maximum_axis_step is None:
        return tuple(ordered)
    output: list[EFITSliceStatus] = []
    previous_with_axis: EFITSliceStatus | None = None
    for status in ordered:
        current = status
        if previous_with_axis is not None:
            previous_axis = (
                previous_with_axis.metrics.get("magnetic_axis_r"),
                previous_with_axis.metrics.get("magnetic_axis_z"),
            )
            current_axis = (
                status.metrics.get("magnetic_axis_r"),
                status.metrics.get("magnetic_axis_z"),
            )
            if None not in previous_axis + current_axis:
                axis_step = float(
                    np.hypot(
                        current_axis[0] - previous_axis[0],
                        current_axis[1] - previous_axis[1],
                    )
                )
                if axis_step > cfg.maximum_axis_step:
                    metrics = dict(status.metrics)
                    metrics["magnetic_axis_step"] = axis_step
                    failures = tuple(
                        dict.fromkeys((*status.failure_codes, "temporal_discontinuity"))
                    )
                    current = replace(
                        status,
                        numerical_ok=False,
                        overall_status=_overall_status(
                            status.runtime_ok,
                            status.output_ok,
                            False,
                            status.physical_ok,
                        ),
                        failure_codes=failures,
                        metrics=metrics,
                    )
        output.append(current)
        if (
            status.metrics.get("magnetic_axis_r") is not None
            and status.metrics.get("magnetic_axis_z") is not None
        ):
            previous_with_axis = status
    return tuple(output)


__all__ = [
    "EFIT_FAILURE_CODES",
    "EFITSliceStatus",
    "EFITValidationConfig",
    "apply_temporal_continuity",
    "validate_efit_slice",
]
