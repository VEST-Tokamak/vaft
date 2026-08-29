"""Canonical pf_active builders integrated under machine_mapping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.io
from scipy import ndimage, signal

from vaft.database import raw as raw_db
from vaft.process.signal_processing import repair_clipped_interval

from .utils import resolve_vest_diagnostic, set_path

PF_COIL_COUNT = 10
COPPER_RESISTIVITY = 1.68e-8
PF_WIDTH_BY_COIL = [0.0172, 0.04, 0.028, 0.028, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042]
PF_RADIUS_BY_COIL = [0.053, 0.104, 0.29, 0.57, 0.71, 0.71, 0.71, 0.71, 0.93, 0.93]
PF_HEIGHT_BY_COIL_1906 = [2.4, 0.76, 0.029, 0.029, 0.029, 0.029, 0.0648, 0.0648, 0.0648, 0.0648]
PF_HEIGHT_BY_COIL_2507 = [2.4, 0.76, 0.029, 0.029, 0.029, 0.0616, 0.0324, 0.0648, 0.0648, 0.0648]
PF_GEOMETRY_2507_FIRST_SHOT = 45958


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


def _safe_vest_load(
    shot: int,
    field: int,
    raw_source: raw_db.RawSource | None = None,
):
    return raw_db.vest_load(
        shot,
        field,
        sample_opt=False if raw_source is None else raw_source,
    )


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


def pf_geometry_version_for_shot(shot: int | None) -> str:
    """Return the PF geometry version used for *shot*.

    Keeping this boundary in one public helper ensures that static PF geometry
    and geometry-dependent coupling matrices always select the same version.
    ``None`` retains the historical geometry for backward compatibility.
    """
    if shot is not None and shot >= PF_GEOMETRY_2507_FIRST_SHOT:
        return "2507"
    return "1906"


def _geometry_profile_for_shot(shot: int | None) -> tuple[str, list[float]]:
    if pf_geometry_version_for_shot(shot) == "2507":
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
    if signal_time.size == reference_time.size and np.allclose(signal_time, reference_time):
        return signal_values
    return np.interp(reference_time, signal_time, signal_values)


def _coil_gain_by_index(shot: int) -> dict[int, float]:
    gains = resolve_vest_diagnostic(shot, "pf_active")["processing"]["coil_gains"]
    return {int(coil_index): float(gain) for coil_index, gain in gains.items()}


def _legacy_pf_filter(values: np.ndarray, processing: dict) -> np.ndarray:
    filter_config = processing["filter"]
    taps = signal.firwin(
        int(filter_config["taps"]),
        float(filter_config["cutoff_frequency"]),
        pass_zero="lowpass",
        fs=float(filter_config["sample_rate"]),
    )
    # scipy.signal.filtfilt requires a long acquisition. Tiny synthetic test
    # dumps retain a deterministic forward-filter fallback.
    if values.size > 3 * (taps.size - 1):
        filtered = signal.filtfilt(taps, 1, values)
    else:
        filtered = signal.lfilter(taps, 1, values)
    return ndimage.uniform_filter1d(filtered, size=min(int(filter_config["smoothing_window"]), values.size))


def vfit_pf(
    shot: int,
    geometry_root: str | Path | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    processing = resolve_vest_diagnostic(shot, "pf_active")["processing"]
    coil_info = scipy.io.loadmat(resolve_geometry_asset("Coil_info.mat", geometry_root=geometry_root))
    coil_numbers = np.asarray(coil_info["CoilNumber"][0], dtype=int) - 1
    coil_codes = np.asarray(coil_info["CoilCode"][0], dtype=int)
    coil_gains = _coil_gain_by_index(shot)
    active_coils = set(int(index) for index in coil_numbers.tolist())

    if coil_codes.size == 0:
        raise ValueError("Coil_info.mat defines no active PF coil signals")

    required_signals = {
        int(field_code): raw_db.require_signal(
            _safe_vest_load(shot, int(field_code), raw_source),
            shot=shot,
            field=int(field_code),
            signal_name="PF active coil current",
        )
        for field_code in np.unique(coil_codes)
    }
    reference_time = required_signals[int(coil_codes[0])][0]

    saturation_repair = {
        int(coil_index): repair
        for coil_index, repair in (processing.get("saturation_repair") or {}).items()
    }

    pf_data: list[np.ndarray] = []
    code_index = 0
    for coil_index in range(PF_COIL_COUNT):
        if coil_index in active_coils:
            field_code = int(coil_codes[code_index])
            waveform_time, raw_values = required_signals[field_code]
            current = raw_values - _baseline_mean(raw_values, int(processing["baseline_samples"]))
            current = _legacy_pf_filter(current, processing) * coil_gains.get(coil_index, 0.0)
            current = _coerce_signal_to_reference(reference_time, waveform_time, current)
            repair = saturation_repair.get(coil_index)
            if repair is not None:
                # Acquisition clipping (VEST PF6 near -5000 A). A
                # SignalRepairError propagates deliberately: fabricating a
                # waveform would be worse than reporting it is unrecoverable.
                current = repair_clipped_interval(
                    reference_time,
                    current,
                    clip_value=float(repair["value"]),
                    tolerance=float(repair["tolerance"]),
                )
            code_index += 1
        else:
            # Coil_info.mat intentionally marks this hardware channel disabled;
            # unlike a missing acquired signal, an explicit zero is meaningful.
            current = np.zeros(reference_time.size, dtype=float)
        pf_data.append(current)

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


def vfit_pf_active_dynamic(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    waveform_time, pf_data = vfit_pf(shot, raw_source=raw_source)
    time_axis = (
        np.asarray(target_time, dtype=float)
        if target_time is not None
        else _build_time_axis(waveform_time, tstart, tend, dt)
    )
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
    *,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    vfit_pf_active_static(ods, shot=shot, geometry_root=geometry_root)
    vfit_pf_active_dynamic(ods, shot=shot, tstart=tstart, tend=tend, dt=dt, raw_source=raw_source)


def pf_active(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    geometry_root: str | Path | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    vfit_pf_active_for_shot(
        ods,
        shot,
        tstart,
        tend,
        dt,
        geometry_root=geometry_root,
        raw_source=raw_source,
    )


def pf_active_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    raw_source = options.get("raw_source") if options else None
    pf_active(ods, shot, tstart, tend, dt, raw_source=raw_source)


__all__ = [
    "PF_GEOMETRY_2507_FIRST_SHOT",
    "pf_active",
    "pf_active_from_raw_database",
    "pf_geometry_version_for_shot",
    "resolve_geometry_asset",
    "vfit_pf",
    "vfit_pf_active_dynamic",
    "vfit_pf_active_for_shot",
    "vfit_pf_active_static",
]
