"""Canonical pf_active builders integrated under machine_mapping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.io

from vaft.database import raw as raw_db
from vaft.process.signal_processing import smooth, vest_coil_current_noise_reduction

from .utils import set_path

PF_COIL_COUNT = 10
COPPER_RESISTIVITY = 1.68e-8
DEFAULT_SAMPLE_COUNT = 25_000
DEFAULT_TIME_AXIS = np.linspace(0.0, 0.99996, DEFAULT_SAMPLE_COUNT)
PF_REFERENCE_FIELD_CODE = 59
PF2_REFERENCE_SHOT = 32527
PF2_REFERENCE_FIELD_CODE = 4
PF_WIDTH_BY_COIL = [0.0172, 0.04, 0.028, 0.028, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042]
PF_RADIUS_BY_COIL = [0.053, 0.104, 0.29, 0.57, 0.71, 0.71, 0.71, 0.71, 0.93, 0.93]
PF_HEIGHT_BY_COIL_1906 = [2.4, 0.76, 0.029, 0.029, 0.029, 0.029, 0.0648, 0.0648, 0.0648, 0.0648]
PF_HEIGHT_BY_COIL_2507 = [2.4, 0.76, 0.029, 0.029, 0.029, 0.0616, 0.0324, 0.0648, 0.0648, 0.0648]


def _candidate_geometry_roots() -> list[Path]:
    return [
        Path(__file__).resolve().parents[1] / "data" / "geometry",
        Path(__file__).resolve().parents[3] / "vest_database" / "OMAS" / "Geometry",
    ]


def resolve_geometry_asset(filename: str, geometry_root: str | Path | None = None) -> Path:
    candidates = [Path(geometry_root)] if geometry_root is not None else _candidate_geometry_roots()
    for root in candidates:
        candidate = root / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root / filename) for root in candidates)
    raise FileNotFoundError(f"Cannot resolve geometry asset {filename!r}; searched {searched}")


def _safe_vest_load(shot: int, field: int):
    if not raw_db.sql_loading_available():
        return None
    return raw_db.vest_load(shot, field)


def _build_time_axis(source_time: np.ndarray, tstart: float, tend: float, dt: float) -> np.ndarray:
    if dt > 0:
        start = max(tstart, float(source_time[0])) if source_time.size > 0 else tstart
        end = min(tend, float(source_time[-1])) if source_time.size > 0 else tend
        if end <= start:
            return np.array([start], dtype=float)
        return np.arange(start, end, dt)
    if source_time.size > 0:
        return source_time
    return np.array([tstart], dtype=float)


def _geometry_profile_for_shot(shot: int | None) -> tuple[str, list[float]]:
    if shot is not None and shot > 45957:
        return "VEST_DiscretizedCoilGeometry_Full_ver_2507.mat", PF_HEIGHT_BY_COIL_2507
    return "VEST_DiscretizedCoilGeometry_Full_ver_1906.mat", PF_HEIGHT_BY_COIL_1906


def _baseline_mean(values: np.ndarray, count: int) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values[: min(count, values.size)]))


def _coerce_signal_to_reference(
    reference_time: np.ndarray,
    signal_time: np.ndarray,
    signal_values: np.ndarray,
) -> np.ndarray:
    if signal_time.size <= 1 or signal_values.size <= 1:
        return np.zeros(reference_time.size, dtype=float)
    if signal_time.size == reference_time.size and np.allclose(signal_time, reference_time):
        return signal_values
    return np.interp(reference_time, signal_time, signal_values)


def _coil_gain_by_index(shot: int) -> dict[int, float]:
    gains = {}
    for coil_index in range(PF_COIL_COUNT):
        if coil_index == 0:
            if shot < 20259:
                gain = 1e4
            elif 20258 < shot < 38361:
                gain = -5e4
            elif 38360 < shot < 38401:
                gain = 5e4
            else:
                gain = -5e4
        elif coil_index == 1:
            gain = 1e3
        elif coil_index == 4:
            gain = 1e4 if shot < 38110 else -1e4
        elif coil_index == 5:
            gain = 1e3 if shot < 38110 else -1e3
        elif coil_index in (8, 9):
            gain = -1e3 if shot < 19287 else -0.5e3
        else:
            continue
        gains[coil_index] = gain
    return gains


def vfit_pf(shot: int, geometry_root: str | Path | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    coil_info = scipy.io.loadmat(resolve_geometry_asset("Coil_info.mat", geometry_root=geometry_root))
    coil_numbers = np.asarray(coil_info["CoilNumber"][0], dtype=int) - 1
    coil_codes = np.asarray(coil_info["CoilCode"][0], dtype=int)
    coil_gains = _coil_gain_by_index(shot)
    active_coils = set(int(index) for index in coil_numbers.tolist())

    reference_time = DEFAULT_TIME_AXIS.copy()
    first_loaded = _safe_vest_load(shot, int(coil_codes[0])) if coil_codes.size > 0 else None
    if first_loaded is not None and len(first_loaded[0]) > 1:
        reference_time = np.asarray(first_loaded[0], dtype=float)

    pf2_reference = _safe_vest_load(PF2_REFERENCE_SHOT, PF2_REFERENCE_FIELD_CODE)
    if pf2_reference is None:
        pf2_noise = np.zeros(reference_time.size, dtype=float)
    else:
        _, temp_pf2_noise = pf2_reference
        pf2_noise = vest_coil_current_noise_reduction(np.asarray(temp_pf2_noise, dtype=float)) * coil_gains.get(1, 0.0)
        pf2_noise = _coerce_signal_to_reference(reference_time, DEFAULT_TIME_AXIS, pf2_noise)

    pf_data: list[np.ndarray] = []
    code_index = 0
    for coil_index in range(PF_COIL_COUNT):
        if coil_index in active_coils:
            field_code = int(coil_codes[code_index])
            loaded = _safe_vest_load(shot, field_code)
            if loaded is None:
                current = np.zeros(reference_time.size, dtype=float)
            else:
                waveform_time, raw_values = loaded
                waveform_time = np.asarray(waveform_time, dtype=float)
                raw_values = np.asarray(raw_values, dtype=float)
                current = raw_values - _baseline_mean(raw_values, 5000)
                current = smooth(current, 50)
                current = vest_coil_current_noise_reduction(current) * coil_gains.get(coil_index, 0.0)
                current = _coerce_signal_to_reference(reference_time, waveform_time, current)
            code_index += 1
        else:
            current = np.zeros(reference_time.size, dtype=float)
        pf_data.append(current)

    if len(pf_data) > 1:
        pf_data[1] = np.zeros(reference_time.size, dtype=float)
    return reference_time, pf_data


def vfit_pf_active_static(
    ods: object,
    shot: int | None = None,
    geometry_root: str | Path | None = None,
) -> None:
    geometry_file, height_by_coil = _geometry_profile_for_shot(shot)
    line_data = scipy.io.loadmat(resolve_geometry_asset(geometry_file, geometry_root=geometry_root))[
        "DiscretizedCoilGeometry"
    ]

    set_path(ods, "pf_active.ids_properties.comment", "PF config from vest_pf_active")
    set_path(ods, "pf_active.ids_properties.homogeneous_time", 1)

    for coil_index in range(PF_COIL_COUNT):
        set_path(ods, f"pf_active.coil.{coil_index}.name", f"PF{coil_index + 1}")
        set_path(ods, f"pf_active.coil.{coil_index}.identifier", f"PF{coil_index + 1}")
        area = PF_WIDTH_BY_COIL[coil_index] * height_by_coil[coil_index]
        resistance = 2.0 * math.pi * COPPER_RESISTIVITY * PF_RADIUS_BY_COIL[coil_index] / area
        set_path(ods, f"pf_active.coil.{coil_index}.resistance", resistance)

    element_counts = np.zeros(PF_COIL_COUNT, dtype=int)
    for line in line_data:
        coil_index = int(line[7]) - 1
        element_index = int(element_counts[coil_index])
        width = float(line[2])
        height = float(line[3])
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.turns_with_sign", float(line[5]))
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.geometry.geometry_type", 2)
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.geometry.rectangle.r", float(line[0]))
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.geometry.rectangle.z", float(line[1]))
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.geometry.rectangle.width", width)
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.geometry.rectangle.height", height)
        set_path(ods, f"pf_active.coil.{coil_index}.element.{element_index}.area", width * height)
        element_counts[coil_index] += 1


def vfit_pf_active_dynamic(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    reference = _safe_vest_load(shot, PF_REFERENCE_FIELD_CODE)
    waveform_time, pf_data = vfit_pf(shot)
    if reference is not None and len(reference[0]) > 1:
        source_time = np.asarray(reference[0], dtype=float)
    else:
        source_time = waveform_time

    time_axis = _build_time_axis(source_time, tstart, tend, dt)
    set_path(ods, "pf_active.time", time_axis)

    for coil_index in range(PF_COIL_COUNT):
        current = np.interp(time_axis, waveform_time, pf_data[coil_index])
        set_path(ods, f"pf_active.coil.{coil_index}.current.time", time_axis)
        set_path(ods, f"pf_active.coil.{coil_index}.current.data", current)


def vfit_pf_active_for_shot(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    geometry_root: str | Path | None = None,
) -> None:
    vfit_pf_active_static(ods, shot=shot, geometry_root=geometry_root)
    vfit_pf_active_dynamic(ods, shot=shot, tstart=tstart, tend=tend, dt=dt)


def pf_active(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    geometry_root: str | Path | None = None,
) -> None:
    vfit_pf_active_for_shot(ods, shot, tstart, tend, dt, geometry_root=geometry_root)


__all__ = [
    "pf_active",
    "resolve_geometry_asset",
    "vfit_pf",
    "vfit_pf_active_dynamic",
    "vfit_pf_active_for_shot",
    "vfit_pf_active_static",
]
