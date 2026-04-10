"""Canonical tf builders integrated under machine_mapping."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from vaft.database import raw as raw_db
from vaft.process.signal_processing import smooth

from .utils import set_path

Signal = tuple[np.ndarray, np.ndarray]

TF_TURN_NUMBER = 24
TF_REFERENCE_RADIUS = 0.4
TF_FIELD_CODE = 1
TF_SAMPLE_RATE = 25e3
TF_CUTOFF_FREQUENCY = 2.5e3
TF_HALL_GAIN = -3e4


def _safe_vest_load(shot: int, field: int):
    if not raw_db.sql_loading_available():
        return None
    return raw_db.vest_load(shot, field)


def _default_signal() -> Signal:
    return np.array([0.0]), np.array([0.0])


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


def vfit_tf_current(shot: int) -> Signal:
    loaded = _safe_vest_load(shot, TF_FIELD_CODE)
    if loaded is None:
        return _default_signal()

    time_tf, raw_tf = loaded
    if len(raw_tf) == 0:
        return _default_signal()

    taps = signal.firwin(26, TF_CUTOFF_FREQUENCY, pass_zero="lowpass", fs=TF_SAMPLE_RATE)
    data_raw_tf = np.asarray(raw_tf, dtype=float) * TF_HALL_GAIN

    baseline_samples = min(1000, data_raw_tf.size)
    data_raw_tf = data_raw_tf - float(np.mean(data_raw_tf[:baseline_samples]))

    tf_current_waveform = signal.lfilter(taps, 1, data_raw_tf)
    tf_current_waveform = smooth(tf_current_waveform, 50)
    return np.asarray(time_tf, dtype=float), np.asarray(tf_current_waveform, dtype=float)


def vfit_tf_bt_r(shot: int) -> Signal:
    time, tf_current = vfit_tf_current(shot)
    bt_r = 4 * math.pi * 1e-7 * TF_TURN_NUMBER * tf_current / (2.0 * math.pi)
    return time, bt_r


def vfit_tf_dynamic(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    source_time, tf_current = vfit_tf_current(shot)
    target_time = _build_target_time_axis(source_time, tstart, tend, dt)

    bt_r = 4 * math.pi * 1e-7 * TF_TURN_NUMBER * tf_current / (2.0 * math.pi)
    btor = bt_r / TF_REFERENCE_RADIUS

    set_path(ods, "tf.b_field_tor_vacuum_r.time", target_time)
    set_path(ods, "tf.b_field_tor_vacuum_r.data", np.interp(target_time, source_time, btor) * TF_REFERENCE_RADIUS)
    set_path(ods, "tf.coil.0.current.time", target_time)
    set_path(ods, "tf.coil.0.current.data", np.interp(target_time, source_time, tf_current))
    set_path(ods, "tf.time", target_time)


def vfit_tf_static(ods: object) -> None:
    set_path(ods, "tf.ids_properties.comment", "tf from vfit_tf")
    set_path(ods, "tf.ids_properties.homogeneous_time", 1)
    set_path(ods, "tf.r0", TF_REFERENCE_RADIUS)


def tf(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    vfit_tf_static(ods)
    vfit_tf_dynamic(ods, shot, tstart, tend, dt)


def tf_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    del options
    tf(ods, shot, tstart, tend, dt)


__all__ = [
    "tf",
    "tf_from_raw_database",
    "vfit_tf_bt_r",
    "vfit_tf_current",
    "vfit_tf_dynamic",
    "vfit_tf_static",
]


vfit_tf_btR = vfit_tf_bt_r
