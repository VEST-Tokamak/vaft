"""Canonical tf builders integrated under machine_mapping."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from vaft.database import raw as raw_db
from vaft.process.signal_processing import smooth

from .utils import calibrate_vest_signal, resolve_vest_diagnostic, set_path

Signal = tuple[np.ndarray, np.ndarray]

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


def _build_target_time_axis(
    source_time: np.ndarray,
    tstart: float,
    tend: float,
    dt: float,
) -> np.ndarray:
    if source_time.size == 0:
        return np.array([0.0])
    if dt <= 0:
        return source_time
    start = max(tstart, float(source_time[0]))
    end = min(tend, float(source_time[-1]))
    if end <= start:
        return source_time
    return np.arange(start, end, dt)


def vfit_tf_current(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> Signal:
    config = resolve_vest_diagnostic(shot, "tf")
    field = int(config["source"]["field"])
    processing = config["processing"]
    time_tf, raw_tf = raw_db.require_signal(
        _safe_vest_load(shot, field, raw_source),
        shot=shot,
        field=field,
        signal_name="TF coil current",
    )

    taps = signal.firwin(
        int(processing["taps"]), float(processing["cutoff_frequency"]),
        pass_zero="lowpass", fs=float(processing["sample_rate"]),
    )
    data_raw_tf = calibrate_vest_signal(raw_tf, config["calibration"])

    baseline_samples = min(int(processing["baseline_samples"]), data_raw_tf.size)
    data_raw_tf = data_raw_tf - float(np.mean(data_raw_tf[:baseline_samples]))

    tf_current_waveform = signal.lfilter(taps, 1, data_raw_tf)
    tf_current_waveform = smooth(tf_current_waveform, int(processing["smoothing_window"]))
    return np.asarray(time_tf, dtype=float), np.asarray(tf_current_waveform, dtype=float)


def vfit_tf_bt_r(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> Signal:
    time, tf_current = vfit_tf_current(shot, raw_source=raw_source)
    turns = float(resolve_vest_diagnostic(shot, "tf")["output"]["turns"])
    bt_r = 4 * math.pi * 1e-7 * turns * tf_current / (2.0 * math.pi)
    return time, bt_r


def vfit_tf_dynamic(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    source_time, tf_current = vfit_tf_current(shot, raw_source=raw_source)
    output = resolve_vest_diagnostic(shot, "tf")["output"]
    turns = float(output["turns"])
    reference_radius = float(output["reference_radius"])
    target_time = (
        np.asarray(target_time, dtype=float)
        if target_time is not None
        else _build_target_time_axis(source_time, tstart, tend, dt)
    )

    bt_r = 4 * math.pi * 1e-7 * turns * tf_current / (2.0 * math.pi)
    btor = bt_r / reference_radius

    set_path(ods, "tf.b_field_tor_vacuum_r.time", target_time)
    set_path(ods, "tf.b_field_tor_vacuum_r.data", np.interp(target_time, source_time, btor) * reference_radius)
    set_path(ods, "tf.coil.0.current.time", target_time)
    set_path(ods, "tf.coil.0.current.data", np.interp(target_time, source_time, tf_current))
    set_path(ods, "tf.time", target_time)


def vfit_tf_static(ods: object) -> None:
    reference_radius = float(resolve_vest_diagnostic(0, "tf")["output"]["reference_radius"])
    set_path(ods, "tf.ids_properties.comment", "tf from vfit_tf")
    set_path(ods, "tf.ids_properties.homogeneous_time", 1)
    set_path(ods, "tf.r0", reference_radius)


def tf(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    vfit_tf_static(ods)
    vfit_tf_dynamic(ods, shot, tstart, tend, dt, raw_source=raw_source)


def tf_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    raw_source = options.get("raw_source") if options else None
    tf(ods, shot, tstart, tend, dt, raw_source=raw_source)


__all__ = [
    "tf",
    "tf_from_raw_database",
    "vfit_tf_bt_r",
    "vfit_tf_current",
    "vfit_tf_dynamic",
    "vfit_tf_static",
]


vfit_tf_btR = vfit_tf_bt_r
