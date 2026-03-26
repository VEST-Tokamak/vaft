"""Canonical barometry builders integrated under machine_mapping."""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from vaft.database import raw as raw_db

from .utils import set_path

BAROMETRY_FIELD_CODE = 13
TORR_TO_PA = 133.3223684211
DEFAULT_DT = 4e-5
MEDIAN_KERNEL = 101


def _safe_vest_load(shot: int, field: int):
    if not raw_db.sql_loading_available():
        return None
    return raw_db.vest_load(shot, field)


def _build_target_time(
    source_time: np.ndarray,
    tstart: float,
    tend: float,
    dt: float,
) -> np.ndarray:
    if dt > 0 and source_time.size > 0:
        start = max(tstart, float(source_time[0]))
        end = min(tend, float(source_time[-1]))
        if end > start:
            return np.arange(start, end, dt)
    step = dt if dt > 0 else DEFAULT_DT
    return np.arange(tstart, tend, step)


def vfit_barometry_static(ods: object) -> None:
    set_path(ods, "barometry.ids_properties.comment", "VEST Pressure Gauge data")
    set_path(ods, "barometry.ids_properties.homogeneous_time", 1)
    set_path(ods, "barometry.gauge.0.name", "PKR-251 Main Gauge")
    set_path(ods, "barometry.gauge.0.type.index", 0)
    set_path(ods, "barometry.gauge.0.type.name", "Penning")
    set_path(ods, "barometry.gauge.0.type.description", "PKR-251 Main Gauge")


def vfit_barometry_dynamic(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    loaded = _safe_vest_load(shot, BAROMETRY_FIELD_CODE)
    if loaded is None:
        time = _build_target_time(np.array([]), tstart, tend, dt)
        set_path(ods, "barometry.gauge.0.pressure.time", time)
        set_path(ods, "barometry.gauge.0.pressure.data", np.zeros_like(time))
        return

    source_time, source_data = loaded
    source_time = np.asarray(source_time, dtype=float)
    source_data = np.asarray(source_data, dtype=float)
    time = _build_target_time(source_time, tstart, tend, dt)

    if source_data.size <= 1 or source_time.size <= 1:
        set_path(ods, "barometry.gauge.0.pressure.time", time)
        set_path(ods, "barometry.gauge.0.pressure.data", np.zeros_like(time))
        return

    pressure_torr = medfilt(source_data, kernel_size=MEDIAN_KERNEL)
    pressure_pa = pressure_torr * TORR_TO_PA
    data = np.interp(time, source_time, pressure_pa)

    set_path(ods, "barometry.gauge.0.pressure.time", time)
    set_path(ods, "barometry.gauge.0.pressure.data", data)


def barometry(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    vfit_barometry_static(ods)
    vfit_barometry_dynamic(ods, shot, tstart, tend, dt)


__all__ = ["barometry", "vfit_barometry_dynamic", "vfit_barometry_static"]
