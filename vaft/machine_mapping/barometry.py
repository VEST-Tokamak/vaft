"""Canonical barometry builders integrated under machine_mapping."""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from vaft.database import raw as raw_db
from vaft.process.signal_processing import resample_to_time

from .utils import (
    build_window_time_axis,
    calibrate_vest_signal,
    resolve_vest_diagnostic,
    set_path,
)


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


def vfit_barometry_static(ods: object) -> None:
    set_path(ods, "barometry.ids_properties.comment", "VEST Pressure Gauge data")
    # barometry never gets a top-level `.time`; each gauge stores its own
    # `pressure.time`. Per the DD, homogeneous_time=1 requires the shared
    # time node just below the IDS root, so storage at a lower level is 0.
    set_path(ods, "barometry.ids_properties.homogeneous_time", 0)
    set_path(ods, "barometry.gauge.0.name", "PKR-251 Main Gauge")
    set_path(ods, "barometry.gauge.0.type.index", 0)
    set_path(ods, "barometry.gauge.0.type.name", "Penning")
    set_path(ods, "barometry.gauge.0.type.description", "PKR-251 Main Gauge")


def vfit_barometry_dynamic(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    config = resolve_vest_diagnostic(shot, "barometry_main")
    field = int(config["source"]["field"])
    processing = config["processing"]
    source_time, source_data = raw_db.require_signal(
        _safe_vest_load(shot, field, raw_source),
        shot=shot,
        field=field,
        signal_name="PKR-251 main gauge",
    )
    time = (
        np.asarray(target_time, dtype=float)
        if target_time is not None
        else build_window_time_axis(source_time, tstart, tend, dt)
    )

    pressure_torr = calibrate_vest_signal(source_data, config["calibration"])
    # medfilt is a de-spiker for the gauge's digitiser glitches, not an
    # anti-alias filter -- it has no defined stopband.  The rate change below is
    # handled separately.
    pressure_torr = medfilt(pressure_torr, kernel_size=int(processing["median_kernel"]))
    pressure_pa = pressure_torr * float(processing["unit_conversion"]["factor"])
    data = resample_to_time(source_time, pressure_pa, time)

    set_path(ods, "barometry.gauge.0.pressure.time", time)
    set_path(ods, "barometry.gauge.0.pressure.data", data)


def barometry(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    vfit_barometry_static(ods)
    vfit_barometry_dynamic(
        ods, shot, tstart, tend, dt, raw_source=raw_source, target_time=target_time
    )


def barometry_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    raw_source = options.get("raw_source") if options else None
    barometry(ods, shot, tstart, tend, dt, raw_source=raw_source)


__all__ = [
    "barometry",
    "barometry_from_raw_database",
    "vfit_barometry_dynamic",
    "vfit_barometry_static",
]
