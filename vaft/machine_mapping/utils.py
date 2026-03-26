"""Shared orchestration helpers for `vaft.machine_mapping`."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from vaft.database import raw as raw_db
from vaft.process.signal_processing import process_signal as process_signal_impl


DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR = (
    1.0e-4,
    1.0e-4,
    5.0e-2,
    3.0e-2,
    1.0e-2,
    1.0e-1,
    1.0e-2,
    1.0e-1,
    1.0e-2,
)

DEFAULT_CONSTRAINT_UNCERTAINTIES = {
    "pf_active_current": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[0],
    "tf_b_field_tor_vacuum_r": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[1],
    "magnetics_ip": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[2],
    "magnetics_diamagnetic_flux": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[3],
    "magnetics_bpol_inboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[4],
    "magnetics_bpol_side": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[5],
    "magnetics_bpol_outboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[6],
    "magnetics_flux_loop_inboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[7],
    "magnetics_flux_loop_outboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[8],
}


def package_data_path(filename: str) -> str:
    """Return an absolute path to a file shipped with `vaft.machine_mapping`."""
    return os.path.join(os.path.dirname(__file__), filename)


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the default on-disk data root for donor-style assets."""
    if data_root is not None:
        return Path(data_root)
    return Path(__file__).resolve().parents[1] / "data"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {} if data is None else data


def _set_nested_mapping_value(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = mapping

    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        next_is_list = not is_last and parts[index + 1].isdigit()

        if isinstance(current, dict):
            if is_last:
                current[part] = value
                return
            next_value = current.get(part)
            if next_is_list:
                if not isinstance(next_value, list):
                    next_value = []
                    current[part] = next_value
            else:
                if not isinstance(next_value, dict):
                    next_value = {}
                    current[part] = next_value
            current = next_value
            continue

        if isinstance(current, list):
            slot = int(part)
            while len(current) <= slot:
                current.append(None)
            if is_last:
                current[slot] = value
                return
            next_value = current[slot]
            if next_is_list:
                if not isinstance(next_value, list):
                    next_value = []
                    current[slot] = next_value
            else:
                if not isinstance(next_value, dict):
                    next_value = {}
                    current[slot] = next_value
            current = next_value
            continue

        raise TypeError(f"Cannot set nested value on non-container at {part!r} in {path!r}")


def _get_nested_mapping_value(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
            continue
        if isinstance(current, list):
            current = current[int(part)]
            continue
        raise KeyError(path)
    return current


def set_path(ods: Any, path: str, value: Any) -> None:
    """Write a dotted path into either a plain dict or an OMAS ODS object."""
    if isinstance(ods, dict):
        _set_nested_mapping_value(ods, path, value)
        return
    ods[path] = value


def get_path(ods: Any, path: str) -> Any:
    """Read a dotted path from either a plain dict or an OMAS ODS object."""
    if isinstance(ods, dict):
        return _get_nested_mapping_value(ods, path)
    return ods[path]


def path_exists(ods: Any, path: str) -> bool:
    """Return whether a dotted path can be resolved from a dict or ODS object."""
    try:
        get_path(ods, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def _normalize_shot_key(source: Any) -> str:
    try:
        return str(int(source))
    except Exception:
        return str(source)


def load_raw_data(
    source: str, field: str | int, options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw data by delegating DB access to `vaft.database.raw`."""
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")

    if source_type == "shot":
        try:
            numeric_field = int(field)
            loaded = raw_db.load(int(source), numeric_field)
        except (TypeError, ValueError):
            loaded = raw_db.vest_load_by_name(int(source), str(field))
        if loaded is None:
            return np.array([0.0]), np.array([0.0])
        time, data = loaded
        return np.asarray(time), np.asarray(data)

    file_format = options.get("file_format", "mat")
    if file_format != "mat":
        raise ValueError(f"Unsupported file format: {file_format}")

    from scipy.io import loadmat

    mat_data = loadmat(source)
    time = np.asarray(mat_data.get("time", np.array([]))).reshape(-1)
    data = np.asarray(mat_data.get(str(field), np.array([]))).reshape(-1)
    return time, data


def process_signal(
    time: np.ndarray, data: np.ndarray, options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Thin wrapper delegating signal conditioning to `vaft.process.signal_processing`."""
    return process_signal_impl(time, data, options)


def get_diagnostic_info(
    source: str, diagnostic_type: str, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")
    info_file = options.get("info_file") or package_data_path("vest.yaml")
    info = load_yaml(info_file)

    if source_type == "shot":
        shot_key = _normalize_shot_key(source)
        shot_block = info.get(shot_key)
        default_block = info.get("0") or info.get(0) or info.get("static") or info
        block = shot_block or default_block
    else:
        block = info

    if diagnostic_type not in block:
        raise ValueError(
            f"No information found for diagnostic '{diagnostic_type}' in '{info_file}'"
        )

    return block[diagnostic_type]


def get_static_info(
    source: str, diagnostic_type: str, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")
    info_file = options.get("info_file") or package_data_path("vest.yaml")
    info = load_yaml(info_file)

    def pick_block() -> Dict[str, Any]:
        if source_type != "shot":
            return info
        shot_key = _normalize_shot_key(source)
        return info.get(shot_key) or info.get("0") or info.get(0) or info

    block = pick_block()

    if isinstance(block, dict) and "static" in block:
        static = block.get("static") or {}
        if diagnostic_type not in static:
            raise ValueError(
                f"No static information for '{diagnostic_type}' in '{info_file}'"
            )
        return static[diagnostic_type]

    if diagnostic_type in block:
        return block[diagnostic_type]

    for _, group in (block or {}).items():
        if isinstance(group, dict) and diagnostic_type in group:
            return group[diagnostic_type]

    raise ValueError(f"No static information found for '{diagnostic_type}' in '{info_file}'")


def process_static_geometry(ods: Any, diagnostic_type: str, static_info: Dict[str, Any]) -> None:
    if "geometry" not in static_info:
        return

    geometry = static_info["geometry"]

    if diagnostic_type == "flux_loop":
        for i, loop in enumerate(geometry.get("loops", [])):
            set_path(ods, f"flux_loop.loop.{i}.position.r", loop.get("r", 0.0))
            set_path(ods, f"flux_loop.loop.{i}.position.z", loop.get("z", 0.0))
            set_path(ods, f"flux_loop.loop.{i}.position.phi", loop.get("phi", 0.0))
            set_path(ods, f"flux_loop.loop.{i}.area", loop.get("area", 0.0))

    elif diagnostic_type == "b_field_pol_probe":
        for i, probe in enumerate(geometry.get("probes", [])):
            set_path(ods, f"b_field_pol_probe.probe.{i}.position.r", probe.get("r", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.position.z", probe.get("z", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.position.phi", probe.get("phi", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.orientation.r", probe.get("orientation_r", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.orientation.z", probe.get("orientation_z", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.orientation.phi", probe.get("orientation_phi", 0.0))

    elif diagnostic_type == "rogowski_coil":
        for i, coil in enumerate(geometry.get("coils", [])):
            set_path(ods, f"rogowski_coil.coil.{i}.position.r", coil.get("r", 0.0))
            set_path(ods, f"rogowski_coil.coil.{i}.position.z", coil.get("z", 0.0))
            set_path(ods, f"rogowski_coil.coil.{i}.position.phi", coil.get("phi", 0.0))
            set_path(ods, f"rogowski_coil.coil.{i}.turns", coil.get("turns", 1))
            set_path(ods, f"rogowski_coil.coil.{i}.area", coil.get("area", 0.0))


def process_static_channels(ods: Any, diagnostic_type: str, static_info: Dict[str, Any]) -> None:
    if "channels" not in static_info:
        return

    channels = static_info["channels"]

    if diagnostic_type == "flux_loop":
        for i, channel in enumerate(channels):
            set_path(ods, f"flux_loop.loop.{i}.name", channel.get("name", f"FL{i}"))
            set_path(ods, f"flux_loop.loop.{i}.gain", channel.get("gain", 1.0))
            set_path(ods, f"flux_loop.loop.{i}.offset", channel.get("offset", 0.0))
            set_path(
                ods,
                f"flux_loop.loop.{i}.calibration_factor",
                channel.get("calibration_factor", 1.0),
            )

    elif diagnostic_type == "b_field_pol_probe":
        for i, channel in enumerate(channels):
            set_path(ods, f"b_field_pol_probe.probe.{i}.name", channel.get("name", f"BP{i}"))
            set_path(ods, f"b_field_pol_probe.probe.{i}.gain", channel.get("gain", 1.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.offset", channel.get("offset", 0.0))
            set_path(
                ods,
                f"b_field_pol_probe.probe.{i}.calibration_factor",
                channel.get("calibration_factor", 1.0),
            )

    elif diagnostic_type == "rogowski_coil":
        for i, channel in enumerate(channels):
            set_path(ods, f"rogowski_coil.coil.{i}.name", channel.get("name", f"RC{i}"))
            set_path(ods, f"rogowski_coil.coil.{i}.gain", channel.get("gain", 1.0))
            set_path(ods, f"rogowski_coil.coil.{i}.offset", channel.get("offset", 0.0))
            set_path(
                ods,
                f"rogowski_coil.coil.{i}.calibration_factor",
                channel.get("calibration_factor", 1.0),
            )


def _scaled_uncertainty(values: Any, relative_error: float):
    array = np.abs(float(relative_error) * np.asarray(values, dtype=float))
    if array.ndim == 0:
        return float(array)
    return array


def _annotate_series(
    ods: Any,
    base_path: str,
    relative_error: float,
    time_source_path: str | None = None,
) -> None:
    data_path = f"{base_path}.data"
    if not path_exists(ods, data_path):
        return
    if time_source_path is not None and path_exists(ods, time_source_path):
        set_path(ods, f"{base_path}.time", get_path(ods, time_source_path))
    set_path(ods, f"{base_path}.data_error_upper", _scaled_uncertainty(get_path(ods, data_path), relative_error))


@lru_cache(maxsize=1)
def _load_magnetics_channel_groups() -> dict[str, list[int]]:
    geometry_path = resolve_data_root() / "geometry" / "VEST_MagneticsGeometry_Full_ver_2302.yaml"
    with open(geometry_path, "r", encoding="utf-8") as handle:
        channels = yaml.safe_load(handle)["channels"]

    groups = {
        "magnetics_bpol_inboard": [],
        "magnetics_bpol_side": [],
        "magnetics_bpol_outboard": [],
        "magnetics_flux_loop_inboard": [],
        "magnetics_flux_loop_outboard": [],
    }

    flux_index = 0
    probe_index = 0
    for channel in channels:
        radial = float(channel["r"])
        vertical = float(channel["z"])
        if channel["kind"] == "flux_loop":
            if radial < 0.15:
                groups["magnetics_flux_loop_inboard"].append(flux_index)
            elif radial > 0.5:
                groups["magnetics_flux_loop_outboard"].append(flux_index)
            flux_index += 1
            continue

        if radial < 0.09:
            groups["magnetics_bpol_inboard"].append(probe_index)
        elif abs(vertical) > 0.8:
            groups["magnetics_bpol_side"].append(probe_index)
        elif radial > 0.795:
            groups["magnetics_bpol_outboard"].append(probe_index)
        probe_index += 1

    return groups


def normalize_constraint_uncertainties(
    uncertainty: Sequence[float] | Mapping[str, float] | None = None,
) -> dict[str, float]:
    if uncertainty is None:
        return dict(DEFAULT_CONSTRAINT_UNCERTAINTIES)

    if isinstance(uncertainty, Mapping):
        normalized = dict(DEFAULT_CONSTRAINT_UNCERTAINTIES)
        unknown = set(uncertainty) - set(normalized)
        if unknown:
            unknown_text = ", ".join(sorted(str(item) for item in unknown))
            raise KeyError(f"Unknown uncertainty keys: {unknown_text}")
        normalized.update({key: float(value) for key, value in uncertainty.items()})
        return normalized

    values = tuple(float(value) for value in uncertainty)
    if len(values) != len(DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR):
        raise ValueError(
            "Constraint uncertainty vector must contain "
            f"{len(DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR)} entries"
        )

    return dict(zip(DEFAULT_CONSTRAINT_UNCERTAINTIES.keys(), values))


def apply_pf_active_current_uncertainties(ods: Any, relative_error: float | None = None) -> None:
    if not path_exists(ods, "pf_active.coil"):
        return
    if relative_error is None:
        relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["pf_active_current"]
    for coil_index, _ in enumerate(get_path(ods, "pf_active.coil")):
        _annotate_series(
            ods,
            f"pf_active.coil.{coil_index}.current",
            relative_error,
            time_source_path="pf_active.time",
        )


def apply_tf_uncertainties(ods: Any, relative_error: float | None = None) -> None:
    if relative_error is None:
        relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["tf_b_field_tor_vacuum_r"]
    _annotate_series(
        ods,
        "tf.b_field_tor_vacuum_r",
        relative_error,
        time_source_path="tf.time",
    )


def apply_magnetics_uncertainties(
    ods: Any,
    *,
    ip_relative_error: float | None = None,
    diamagnetic_flux_relative_error: float | None = None,
    bpol_inboard_relative_error: float | None = None,
    bpol_side_relative_error: float | None = None,
    bpol_outboard_relative_error: float | None = None,
    flux_loop_inboard_relative_error: float | None = None,
    flux_loop_outboard_relative_error: float | None = None,
    fl_correct_coeff: Sequence[float] | None = None,
) -> None:
    groups = _load_magnetics_channel_groups()

    if ip_relative_error is None:
        ip_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_ip"]
    if diamagnetic_flux_relative_error is None:
        diamagnetic_flux_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_diamagnetic_flux"]
    if bpol_inboard_relative_error is None:
        bpol_inboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_inboard"]
    if bpol_side_relative_error is None:
        bpol_side_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_side"]
    if bpol_outboard_relative_error is None:
        bpol_outboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_outboard"]
    if flux_loop_inboard_relative_error is None:
        flux_loop_inboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_flux_loop_inboard"]
    if flux_loop_outboard_relative_error is None:
        flux_loop_outboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_flux_loop_outboard"]

    _annotate_series(ods, "magnetics.ip.0", ip_relative_error)
    _annotate_series(
        ods,
        "magnetics.diamagnetic_flux.0",
        diamagnetic_flux_relative_error,
        time_source_path="magnetics.time",
    )

    if fl_correct_coeff is not None and path_exists(ods, "magnetics.flux_loop"):
        for flux_index, coeff in enumerate(fl_correct_coeff):
            flux_data_path = f"magnetics.flux_loop.{flux_index}.flux.data"
            if not path_exists(ods, flux_data_path):
                continue
            set_path(ods, flux_data_path, np.asarray(get_path(ods, flux_data_path), dtype=float) / float(coeff))

    for probe_index in groups["magnetics_bpol_inboard"]:
        _annotate_series(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.field",
            bpol_inboard_relative_error,
            time_source_path="magnetics.time",
        )
    for probe_index in groups["magnetics_bpol_side"]:
        _annotate_series(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.field",
            bpol_side_relative_error,
            time_source_path="magnetics.time",
        )
    for probe_index in groups["magnetics_bpol_outboard"]:
        _annotate_series(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.field",
            bpol_outboard_relative_error,
            time_source_path="magnetics.time",
        )

    for flux_index in groups["magnetics_flux_loop_inboard"]:
        _annotate_series(
            ods,
            f"magnetics.flux_loop.{flux_index}.flux",
            flux_loop_inboard_relative_error,
            time_source_path="magnetics.time",
        )
    for flux_index in groups["magnetics_flux_loop_outboard"]:
        _annotate_series(
            ods,
            f"magnetics.flux_loop.{flux_index}.flux",
            flux_loop_outboard_relative_error,
            time_source_path="magnetics.time",
        )


def apply_default_constraint_uncertainties(
    ods: Any,
    uncertainty: Sequence[float] | Mapping[str, float] | None = None,
    *,
    fl_correct_coeff: Sequence[float] | None = None,
) -> None:
    normalized = normalize_constraint_uncertainties(uncertainty)
    apply_pf_active_current_uncertainties(ods, normalized["pf_active_current"])
    apply_tf_uncertainties(ods, normalized["tf_b_field_tor_vacuum_r"])
    apply_magnetics_uncertainties(
        ods,
        ip_relative_error=normalized["magnetics_ip"],
        diamagnetic_flux_relative_error=normalized["magnetics_diamagnetic_flux"],
        bpol_inboard_relative_error=normalized["magnetics_bpol_inboard"],
        bpol_side_relative_error=normalized["magnetics_bpol_side"],
        bpol_outboard_relative_error=normalized["magnetics_bpol_outboard"],
        flux_loop_inboard_relative_error=normalized["magnetics_flux_loop_inboard"],
        flux_loop_outboard_relative_error=normalized["magnetics_flux_loop_outboard"],
        fl_correct_coeff=fl_correct_coeff,
    )


__all__ = [
    "DEFAULT_CONSTRAINT_UNCERTAINTIES",
    "DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR",
    "apply_default_constraint_uncertainties",
    "apply_magnetics_uncertainties",
    "apply_pf_active_current_uncertainties",
    "apply_tf_uncertainties",
    "get_diagnostic_info",
    "get_path",
    "get_static_info",
    "load_raw_data",
    "load_yaml",
    "normalize_constraint_uncertainties",
    "package_data_path",
    "path_exists",
    "process_signal",
    "process_static_channels",
    "process_static_geometry",
    "resolve_data_root",
    "set_path",
]
