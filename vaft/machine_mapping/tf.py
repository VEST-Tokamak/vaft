"""Canonical tf builders integrated under machine_mapping."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from vaft.database import raw as raw_db
from vaft.process.signal_processing import smooth

from .utils import (
    build_window_time_axis,
    calibrate_vest_signal,
    resolve_vest_diagnostic,
    set_path,
)

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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _bridge(t: np.ndarray, values: np.ndarray, bad: np.ndarray, span: float) -> np.ndarray:
    """``values`` with each bad run replaced by a straight-line fit to the good
    samples within ``span`` seconds either side.  A least-squares line over
    tens of milliseconds averages the raw noise away, which neither an end
    sample nor a running median contaminated by the excursion itself can do.
    """
    out = values.copy()
    for first, stop in _runs(bad):
        near = (t >= t[first] - span) & (t <= t[stop - 1] + span) & ~bad
        if near.sum() < 2:
            continue
        slope, intercept = np.polyfit(t[near], values[near], 1)
        out[first:stop] = slope * t[first:stop] + intercept
    return out


def repair_tf_excursions(
    time: np.ndarray,
    current: np.ndarray,
    repair: dict | None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Replace acquisition excursions of the TF current by its slow trend (#1543).

    The TF circuit's L/R time is seconds: within a few milliseconds its current
    can change by well under a percent.  Some shots nevertheless record the TF
    current falling to -15 kA or rising to +25 kA from plasma termination
    (~0.303 s) to ~0.326 s and then returning to the 12 kA plateau
    (48238-48269, 46250, 46500).  That is the measurement, not the coil, and
    the excursion passes straight into ``b_field_tor_vacuum_r`` -- EFIT's BTOR.

    Inside ``repair["window"]`` a sample is an excursion when its
    ``detect_window_s`` median departs from the slow trend by more than
    ``core_fraction`` of the plateau; the region grows while the departure
    stays above ``extend_fraction``, plus ``margin_s`` either side, and runs
    closer than ``merge_gap_s`` are joined.  The trend
    is a ``trend_window_s`` running median, re-derived once with the first
    pass's excursions bridged so that they cannot bias it.  Excursion samples
    are then replaced by a straight line fitted to the good samples within
    ``trend_window_s`` either side.  A shot whose TF is off (plateau below
    ``min_plateau``) is left alone.

    Returns the repaired current and the repaired ``(start, end)`` intervals.
    """
    values = np.asarray(current, dtype=float).copy()
    t = np.asarray(time, dtype=float)
    if not repair or not repair.get("enabled", False) or values.size < 3:
        return values, []
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return values, []
    span = float(repair["trend_window_s"])
    trend_size = int(span / dt) | 1
    local = median_filter(values, size=int(float(repair["detect_window_s"]) / dt) | 1, mode="nearest")
    start, end = (float(bound) for bound in repair["window"])
    window = (t >= start) & (t <= end)
    if not window.any():
        return values, []
    margin = int(round(float(repair["margin_s"]) / dt))
    merge_gap = int(round(float(repair.get("merge_gap_s", 0.0)) / dt))

    def detect(trend: np.ndarray) -> tuple[np.ndarray, float]:
        plateau = float(np.median(trend[window]))
        if abs(plateau) < float(repair["min_plateau"]):
            return np.zeros(values.size, dtype=bool), plateau
        departure = np.abs(local - trend) / abs(plateau)
        core = window & (departure > float(repair["core_fraction"]))
        candidate = window & (departure > float(repair["extend_fraction"]))
        bad = np.zeros(values.size, dtype=bool)
        for first, stop in _runs(candidate):
            if core[first:stop].any():
                bad[max(first - margin, 0):min(stop + margin, values.size)] = True
        # Inside a burst the 1 ms median can pass back through the trend for a
        # moment; a gap shorter than ``merge_gap_s`` between two repaired runs
        # is part of the same excursion (48625 kept a +130 kA spike otherwise).
        runs = _runs(bad)
        for (_, stop), (first, _) in zip(runs, runs[1:]):
            if first - stop < merge_gap:
                bad[stop:first] = True
        return bad, plateau

    bad, _ = detect(median_filter(values, size=trend_size, mode="nearest"))
    if not bad.any():
        return values, []
    bridged = _bridge(t, values, bad, span)
    bad, _ = detect(median_filter(bridged, size=trend_size, mode="nearest"))
    if not bad.any():
        return values, []
    repaired = _bridge(t, values, bad, span)
    intervals = [(float(t[first]), float(t[stop - 1])) for first, stop in _runs(bad)]
    return repaired, intervals


def vfit_tf_current_detailed(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """TF current [A] and the intervals whose acquisition excursion was repaired."""
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
    # Before the low-pass, so the filter never smears an excursion into its
    # neighbours.
    data_raw_tf, repaired = repair_tf_excursions(
        np.asarray(time_tf, dtype=float), data_raw_tf, processing.get("excursion_repair")
    )

    tf_current_waveform = signal.lfilter(taps, 1, data_raw_tf)
    tf_current_waveform = smooth(tf_current_waveform, int(processing["smoothing_window"]))
    return np.asarray(time_tf, dtype=float), np.asarray(tf_current_waveform, dtype=float), repaired


def vfit_tf_current(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> Signal:
    time, current, _repaired = vfit_tf_current_detailed(shot, raw_source=raw_source)
    return time, current


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
    source_time, tf_current, repaired = vfit_tf_current_detailed(shot, raw_source=raw_source)
    output = resolve_vest_diagnostic(shot, "tf")["output"]
    turns = float(output["turns"])
    reference_radius = float(output["reference_radius"])
    target_time = (
        np.asarray(target_time, dtype=float)
        if target_time is not None
        else build_window_time_axis(source_time, tstart, tend, dt)
    )

    bt_r = 4 * math.pi * 1e-7 * turns * tf_current / (2.0 * math.pi)
    btor = bt_r / reference_radius

    set_path(ods, "tf.b_field_tor_vacuum_r.time", target_time)
    # anti-alias: vfit_tf_current low-passes the TF waveform with a firwin
    # filter on the source grid before either projection below.
    set_path(ods, "tf.b_field_tor_vacuum_r.data", np.interp(target_time, source_time, btor) * reference_radius)
    set_path(ods, "tf.coil.0.current.time", target_time)
    set_path(ods, "tf.coil.0.current.data", np.interp(target_time, source_time, tf_current))
    set_path(ods, "tf.time", target_time)
    if repaired:
        # The Data Dictionary gives tf.coil.current no validity node, so the
        # repair is stated where a reader of the IDS will find it.
        spans = ", ".join(f"{start:.4f}-{end:.4f} s" for start, end in repaired)
        set_path(
            ods,
            "tf.ids_properties.comment",
            "tf from vfit_tf; TF-current acquisition excursion replaced by its "
            f"60 ms running-median trend over {spans} (issue #1543)",
        )


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
    "repair_tf_excursions",
    "tf",
    "tf_from_raw_database",
    "vfit_tf_bt_r",
    "vfit_tf_current",
    "vfit_tf_current_detailed",
    "vfit_tf_dynamic",
    "vfit_tf_static",
]


vfit_tf_btR = vfit_tf_bt_r
