"""Canonical magnetics mapping integrated under machine_mapping."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import integrate, signal

from vaft.database import raw as raw_db
from vaft.process.signal_processing import smooth, vfit_signal_start_end

from .utils import resolve_data_root, set_path

DEFAULT_MAGNETICS_TIME = np.linspace(0.0, 0.99996, 25_000)
PROBE_LENGTH = 0.01
POLOIDAL_ANGLE = 3 * math.pi / 2


def _safe_vest_load(shot: int, field: int):
    if not raw_db.sql_loading_available():
        return None
    return raw_db.vest_load(shot, field)


def _geometry_root() -> Path:
    return resolve_data_root() / "geometry"


@lru_cache(maxsize=1)
def _load_md_channels() -> list[dict[str, Any]]:
    with open(_geometry_root() / "MD.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["channels"]


@lru_cache(maxsize=1)
def _load_static_channels() -> list[dict[str, Any]]:
    with open(_geometry_root() / "VEST_MagneticsGeometry_Full_ver_2302.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["channels"]


@lru_cache(maxsize=1)
def _load_names_by_code() -> dict[int, str]:
    with open(_geometry_root() / "table.yaml", "r", encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)["entries"]
    return {int(entry["field_code"]): str(entry["name"]) for entry in entries}


def _default_signal() -> tuple[np.ndarray, np.ndarray]:
    return DEFAULT_MAGNETICS_TIME.copy(), np.zeros(DEFAULT_MAGNETICS_TIME.size, dtype=float)


def _fallback_window(tstart: float, tend: float, dt: float) -> np.ndarray:
    if dt > 0:
        if tend <= tstart:
            return np.array([tstart], dtype=float)
        return np.arange(tstart, tend, dt)
    return DEFAULT_MAGNETICS_TIME[6000:8501]


def _interp_or_zero(target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    if source_time.size <= 1 or values.size <= 1:
        return np.zeros(target_time.size, dtype=float)
    return np.interp(target_time, source_time, values)


def _polyfit_baseline(time_axis: np.ndarray, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    valid = indices[(indices >= 0) & (indices < values.size)]
    if valid.size < 2:
        return np.zeros(values.size, dtype=float)
    return np.polyval(np.polyfit(time_axis[valid], values[valid], 1), time_axis)


def vfit_plasma_current(shot: int, ref: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return processed plasma-current waveform with offline zero fallback."""
    if ref == -1:
        x_ip = 102
        x_flux_loop = 25
        ind_mutual = 2.8e-4 if shot < 17455 else 5.0e-4

        loaded_ip = _safe_vest_load(shot, x_ip)
        loaded_flux = _safe_vest_load(shot, x_flux_loop)
        if loaded_ip is None or loaded_flux is None:
            return _default_signal()

        time, raw_ip = loaded_ip
        _, raw_flux = loaded_flux
        time = np.asarray(time, dtype=float)
        raw_ip = np.asarray(raw_ip, dtype=float)
        raw_flux = np.asarray(raw_flux, dtype=float)
        if time.size <= 1 or raw_ip.size <= 1 or raw_flux.size <= 1:
            return _default_signal()

        if (41446 <= shot <= 41451) or shot >= 41660:
            x_time = np.arange(7250, 8750)
        else:
            x_time = np.arange(6000, 8500)
        x_window = 500
        x_base = np.arange(x_time[0] - x_window, x_time[0] + 1, dtype=int)
        x_base = x_base[(x_base >= 0) & (x_base < time.size)]
        if x_base.size < 2:
            x_base = np.arange(min(500, time.size), dtype=int)

        ip_shot = raw_ip - np.polyval(np.polyfit(time[x_base], raw_ip[x_base], 1), time)
        ip_ref = raw_flux * 11 / ind_mutual
        ip_ref = ip_ref - np.polyval(np.polyfit(time[x_base], ip_ref[x_base], 1), time)
        ip = ip_shot - ip_ref

        if shot >= 20259:
            ip = -ip
        return time, ip

    loaded_ref = _safe_vest_load(ref, 102)
    loaded_shot = _safe_vest_load(shot, 102)
    if loaded_ref is None or loaded_shot is None:
        return _default_signal()

    _, reference_values = loaded_ref
    time, shot_values = loaded_shot
    time = np.asarray(time, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    shot_values = np.asarray(shot_values, dtype=float)
    if time.size <= 1 or reference_values.size <= 1 or shot_values.size <= 1:
        return _default_signal()

    sample_rate = 25e3
    cutoff_frequency = 2.5e3
    taps = signal.firwin(26, cutoff_frequency, pass_zero="lowpass", fs=sample_rate)
    plasma_current = -(shot_values - reference_values)
    baseline_index = min(7499, plasma_current.size - 1)
    plasma_current = plasma_current - plasma_current[baseline_index]
    return time, signal.lfilter(taps, 1, plasma_current)


def vfit_plasma_mgods_startend(ods: object) -> tuple[float, float]:
    """Estimate discharge start/end directly from `magnetics.ip.0.*`."""
    try:
        magnetics = ods["magnetics"]
        if isinstance(magnetics, dict) and "ip" in magnetics:
            time = np.asarray(magnetics["ip"][0]["time"], dtype=float)
            ip = np.asarray(magnetics["ip"][0]["data"], dtype=float)
        else:
            time = np.asarray(magnetics["ip.0.time"], dtype=float)
            ip = np.asarray(magnetics["ip.0.data"], dtype=float)
    except Exception:
        return -1.0, -1.0

    if time.size < 2 or ip.size < 2:
        return -1.0, -1.0

    filtered_ip = smooth(ip, 10)
    span = max(1, min(20, filtered_ip.size // 20 if filtered_ip.size >= 20 else filtered_ip.size))

    if time[0] < 0.3:
        start_ref_index = int(np.argmin(np.abs(time - 0.3)))
        baseline_slice = np.abs(filtered_ip[: max(start_ref_index, 1)])
    else:
        baseline_slice = np.abs(filtered_ip[: max(filtered_ip.size // 10, 1)])
    baseline_mean = float(np.mean(baseline_slice)) if baseline_slice.size > 0 else 0.0

    start_index = None
    for idx in range(0, filtered_ip.size - span + 1):
        if np.mean(np.abs(filtered_ip[idx : idx + span])) > max(10.0 * baseline_mean, 1e-9):
            start_index = idx
            break
    if start_index is None:
        start_index = 0

    while start_index > 0 and abs(filtered_ip[start_index]) > baseline_mean:
        start_index -= 1

    if time[-1] > 0.33:
        end_ref_index = int(np.argmin(np.abs(time - 0.33)))
        tail_slice = np.abs(filtered_ip[end_ref_index:])
    else:
        tail_slice = np.abs(filtered_ip[-max(filtered_ip.size // 10, 1) :])
    tail_mean = float(np.mean(tail_slice)) if tail_slice.size > 0 else 0.0

    end_index = None
    for idx in range(filtered_ip.size, start_index + span, -1):
        if np.mean(np.abs(filtered_ip[idx - span : idx])) > max(15.0 * tail_mean, 1e-9):
            end_index = idx - 1
            break
    if end_index is None:
        end_index = filtered_ip.size - 1

    while end_index < filtered_ip.size - 1 and abs(filtered_ip[end_index]) > tail_mean:
        end_index += 1

    return float(time[start_index]), float(time[end_index])


def vest_diamagnetic_flux(shot: int, plasma_start: float, plasma_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute corrected diamagnetic flux waveform with offline zero fallback."""
    field_code = 246 if shot < 37505 else 4 if shot < 38452 else 257
    loaded = _safe_vest_load(shot, field_code)
    if loaded is None:
        return _default_signal()

    temp_time, raw_values = loaded
    temp_time = np.asarray(temp_time, dtype=float)
    raw_values = np.asarray(raw_values, dtype=float)
    if temp_time.size <= 1 or raw_values.size <= 1:
        return _default_signal()

    turn_tf = 24
    ind_tf = 9.3e-4
    res_tf = 0.0279
    cap_tf = 120.0
    rogo_gain = -1 / 8.12e-3

    integrated = integrate.cumulative_trapezoid(raw_values, temp_time, initial=0.0) * rogo_gain
    start_index = int(np.argmin(np.abs(temp_time - plasma_start)))
    end_index = int(np.argmin(np.abs(temp_time - plasma_end)))
    if end_index <= start_index:
        return temp_time, np.zeros(temp_time.size, dtype=float)

    ref_signal = np.interp(
        temp_time,
        np.concatenate((temp_time[: start_index + 1], temp_time[end_index:])),
        np.concatenate((integrated[: start_index + 1], integrated[end_index:])),
    )
    delta_i_tf = integrated - ref_signal

    cum1 = integrate.cumulative_trapezoid(delta_i_tf, temp_time, initial=0.0)
    cum2 = integrate.cumulative_trapezoid(cum1, temp_time, initial=0.0)
    dia_flux = ind_tf / turn_tf * delta_i_tf + res_tf / turn_tf * cum1 + 1 / cap_tf / turn_tf * cum2

    coeff = np.polyfit(
        np.array([temp_time[start_index], temp_time[end_index]]),
        np.array([dia_flux[start_index], dia_flux[end_index]]),
        1,
    )
    baseline = np.polyval(coeff, temp_time)
    baseline[: start_index + 1] = 0.0

    dia_flux_final = dia_flux - baseline
    dia_flux_final[end_index:] = 0.0
    return temp_time, dia_flux_final


def vfit_md(
    shot: int,
    indices: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Process magnetic probe and flux-loop data using YAML channel metadata."""
    f_sample_fast = 250000
    f_cut_low = 2500
    d_low_fast = signal.firwin(251, f_cut_low, pass_zero="lowpass", fs=f_sample_fast)

    index1 = 6000
    index2 = 8500
    index3 = 8500
    if (41446 <= shot <= 41451) or shot >= 41660:
        index1 = 6500
        index2 = 9000
        index3 = 5000

    magnetics_time = DEFAULT_MAGNETICS_TIME[index1 : index2 + 1]
    channels = _load_md_channels()
    if indices is not None:
        channels = [channels[int(index)] for index in indices]

    data_flux_loops: list[np.ndarray] = []
    data_probes: list[np.ndarray] = []
    flux_loop_counter = 0

    for channel in channels:
        field_code = int(channel["field_code"])
        calibration = float(channel["calibration"])
        loaded = _safe_vest_load(shot, field_code)
        if loaded is None:
            if channel["kind"] == "b_field_pol_probe":
                data_probes.append(np.zeros(magnetics_time.size, dtype=float))
            else:
                data_flux_loops.append(np.zeros(magnetics_time.size, dtype=float))
            continue

        source_time, source_data = loaded
        source_time = np.asarray(source_time, dtype=float)
        source_data = np.asarray(source_data, dtype=float)
        if source_time.size <= 1 or source_data.size <= 1:
            if channel["kind"] == "b_field_pol_probe":
                data_probes.append(np.zeros(magnetics_time.size, dtype=float))
            else:
                data_flux_loops.append(np.zeros(magnetics_time.size, dtype=float))
            continue

        if channel["kind"] == "b_field_pol_probe":
            filtered = signal.lfilter(d_low_fast, 1, source_data)
            integrated = -integrate.cumulative_trapezoid(filtered / calibration, source_time, initial=0.0)
            baseline_end = min(index3, integrated.size)
            baseline = _polyfit_baseline(source_time, integrated, np.arange(baseline_end))
            corrected = integrated - baseline
            data_probes.append(_interp_or_zero(magnetics_time, source_time, corrected))
        else:
            flux_loop_counter += 1
            integrated = -integrate.cumulative_trapezoid(source_data / calibration, source_time, initial=0.0) / (
                2 * math.pi
            )
            if flux_loop_counter in (9, 10, 11):
                baseline_indices = np.arange(5999, min(7000, integrated.size))
            else:
                first = np.arange(3499, min(5000, integrated.size))
                second = np.arange(11999, min(15000, integrated.size))
                baseline_indices = np.concatenate((first, second))
            baseline = _polyfit_baseline(source_time, integrated, baseline_indices)
            corrected = integrated - baseline
            data_flux_loops.append(_interp_or_zero(magnetics_time, source_time, corrected))

    return magnetics_time, data_flux_loops, data_probes


def vfit_magnetics_static(ods: object) -> None:
    """Populate static magnetics metadata from YAML geometry assets."""
    names = _load_names_by_code()
    geometry = _load_static_channels()

    set_path(ods, "magnetics.ids_properties.comment", "magnetics config from vest_magnetics")
    set_path(ods, "magnetics.ids_properties.homogeneous_time", 1)

    flux_loop_index = 0
    probe_index = 0
    for channel in geometry:
        field_code = int(channel["field_code"])
        name = names[field_code]
        r_pos = float(channel["r"])
        z_pos = float(channel["z"])

        if channel["kind"] == "flux_loop":
            set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.name", name)
            set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.identifier", name)
            set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.position.0.r", r_pos)
            set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.position.0.z", z_pos)
            flux_loop_index += 1
        else:
            set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.name", name)
            set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.identifier", name)
            set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.r", r_pos)
            set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.z", z_pos)
            set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.length", PROBE_LENGTH)
            set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.poloidal_angle", POLOIDAL_ANGLE)
            probe_index += 1


def vfit_magnetics_dynamic(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    """Populate dynamic magnetics nodes with offline zero-waveform fallback."""
    time, data_flux_loops, data_probes = vfit_md(shot)

    if dt > 0 and time.size > 0:
        start = max(tstart, float(time[0]))
        end = min(tend, float(time[-1]))
        if end > start:
            target_time = np.arange(start, end, dt)
        else:
            target_time = _fallback_window(tstart, tend, dt)
    elif time.size > 0:
        target_time = time
    else:
        target_time = _fallback_window(tstart, tend, dt)

    set_path(ods, "magnetics.time", target_time)

    for index, values in enumerate(data_flux_loops):
        set_path(ods, f"magnetics.flux_loop.{index}.flux.data", _interp_or_zero(target_time, time, values) * 2 * math.pi)
    for index, values in enumerate(data_probes):
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.field.data", _interp_or_zero(target_time, time, values))

    ip_time, ip = vfit_plasma_current(shot)
    if dt > 0:
        set_path(ods, "magnetics.ip.0.data", _interp_or_zero(target_time, ip_time, ip))
        set_path(ods, "magnetics.ip.0.time", target_time)
    else:
        set_path(ods, "magnetics.ip.0.data", ip)
        set_path(ods, "magnetics.ip.0.time", ip_time)

    halpha = _safe_vest_load(shot, 101)
    if halpha is not None and len(halpha[1]) > 1:
        h_time = np.asarray(halpha[0], dtype=float)
        h_data = smooth(np.asarray(halpha[1], dtype=float), 10)
        index_a = int(np.argmin(np.abs(h_time - 0.3)))
        index_b = int(np.argmin(np.abs(h_time - 0.36)))
        window = h_data[index_a:index_b] if index_b > index_a else h_data
        minimum = float(np.min(window)) if window.size > 0 else -1.0
        if minimum != 0.0:
            normalized = h_data / minimum
            tstart2, tend2 = vfit_signal_start_end(h_time[index_a:index_b], normalized[index_a:index_b])
        else:
            tstart2, tend2 = vfit_plasma_mgods_startend(ods)
    else:
        tstart2, tend2 = vfit_plasma_mgods_startend(ods)

    time_dia, dia_flux = vest_diamagnetic_flux(shot, tstart2, tend2)
    set_path(ods, "magnetics.diamagnetic_flux.0.data", _interp_or_zero(target_time, time_dia, dia_flux))
    set_path(ods, "magnetics.diamagnetic_flux.0.time", target_time)


def vfit_magnetics_for_shot(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    """Populate canonical static and dynamic magnetics nodes for one shot."""
    vfit_magnetics_dynamic(ods, shot, tstart, tend, dt)
    vfit_magnetics_static(ods)


def magnetics(ods: object, shot: int, tstart: float, tend: float, dt: float) -> None:
    """Canonical machine_mapping entry point for the magnetics IDS."""
    vfit_magnetics_for_shot(ods, shot, tstart, tend, dt)


def flux_loop_from_raw_database(ods: object, shot: int) -> None:
    del ods, shot
    raise NotImplementedError(
        "flux_loop_from_raw_database is not implemented in canonical machine_mapping. "
        "Use IDS-level magnetics mapping entry points instead."
    )


def b_field_pol_probe_from_raw_database(ods: object, shot: int) -> None:
    del ods, shot
    raise NotImplementedError(
        "b_field_pol_probe_from_raw_database is not implemented in canonical machine_mapping. "
        "Use IDS-level magnetics mapping entry points instead."
    )


def rogowski_coil_and_ip_from_raw_database(ods: object, shot: int) -> None:
    del ods, shot
    raise NotImplementedError(
        "rogowski_coil_and_ip_from_raw_database is not implemented in canonical machine_mapping. "
        "Use IDS-level magnetics mapping entry points instead."
    )


def magnetics_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    del options
    magnetics(ods, shot, tstart, tend, dt)


__all__ = [
    "b_field_pol_probe_from_raw_database",
    "flux_loop_from_raw_database",
    "magnetics_from_raw_database",
    "vest_diamagnetic_flux",
    "magnetics",
    "rogowski_coil_and_ip_from_raw_database",
    "vfit_plasma_current",
    "vfit_md",
    "vfit_magnetics_dynamic",
    "vfit_magnetics_for_shot",
    "vfit_magnetics_static",
    "vfit_plasma_mgods_startend",
]


VEST_DiamagneticFlux = vest_diamagnetic_flux
vfit_PlasmaCurrent = vfit_plasma_current
vfit_plasmaMGods_startend = vfit_plasma_mgods_startend
