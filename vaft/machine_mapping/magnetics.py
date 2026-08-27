"""Canonical magnetics mapping integrated under machine_mapping."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import integrate, signal

from vaft.database import raw as raw_db
from vaft.process.magnetics import VestMagneticsProcessingConfig, vest_equilibrium_magnetics_signals
from vaft.process.signal_processing import detect_active_window, smooth

from .utils import (
    calibrate_vest_signal,
    get_path,
    path_exists,
    resolve_data_root,
    resolve_shot_revisions,
    resolve_vest_diagnostic,
    set_path,
)

DEFAULT_TSTART = 0.26
DEFAULT_TEND = 0.36
DEFAULT_DT = 4e-5
PROBE_LENGTH = 0.01
POLOIDAL_ANGLE = 3 * math.pi / 2
MIRNOV_TYPE_INDEX = 2
OUTBOARD_MIRNOV_MAJOR_RADIUS = 0.796
# Fluctuation Mirnov probes became physically operational at this shot; the
# 30-channel array (issue #155) is only mapped for shot >= this boundary.
FLUCTUATION_MIRNOV_FIRST_SHOT = 44156
TOROIDAL_MIRNOV_REFERENCE_CHANNELS = (
    {
        "field_code": 207,
        "name": "OutMirnov_130_Bz",
        "r": 0.796,
        "z": 0.02,
        "phi": 0.0,
        "toroidal_angle": 0.0,
        "gain": 9.0e-4,
    },
    {
        "field_code": 241,
        "name": "OutMirnov_530_Bz",
        "r": 0.796,
        "z": 0.02,
        "phi": 2 * math.pi / 3,
        "toroidal_angle": 2 * math.pi / 3,
        "gain": -9.0e-4,
    },
    {
        "field_code": 209,
        "name": "OutMirnov_730_Bz",
        "r": 0.796,
        "z": 0.02,
        "phi": math.pi,
        "toroidal_angle": math.pi,
        "gain": 9.0e-4,
    },
    {
        "field_code": 171,
        "name": "MagneticFieldProbe_C2-05_Bz",
        "r": 0.796,
        "z": 0.02,
        "phi": 4 * math.pi / 3,
        "toroidal_angle": 4 * math.pi / 3,
        "gain": 0.004529,
    },
)

# Database identifiers, rather than older UI labels, define the physical
# limiter segment. The 0.1 ohm-equivalent resistance stores the Pearson Model
# 411 transfer sensitivity (0.1 V/A), not a limiter-ground resistor. Fig. 5
# of Lee et al. (2018) provides only approximate R-Z locations, while IMAS
# 3.41 shunt positions are electrical terminal endpoints, so no position is
# written until authoritative endpoint geometry is available.
LIMITER_SHUNT_CHANNELS = (
    {
        "field_code": 216,
        "identifier": "LimiterCurrentMonitor_LC",
        "name": "Lower-corner limiter current monitor",
    },
    {
        "field_code": 217,
        "identifier": "LimiterCurrentMonitor_UC",
        "name": "Upper-corner limiter current monitor",
    },
    {
        "field_code": 218,
        "identifier": "LimiterCurrentMonitor_MM",
        "name": "Midplane limiter current monitor",
    },
)
LIMITER_SHUNT_RESISTANCE = 0.1
LIMITER_SHUNT_BASELINE_WINDOW = (0.0, 0.2)


@lru_cache(maxsize=1024)
def _safe_vest_load_cached(shot: int, field: int, raw_source: str | None):
    return raw_db.vest_load(
        shot,
        field,
        sample_opt=False if raw_source is None else raw_source,
    )


def _safe_vest_load(
    shot: int,
    field: int,
    raw_source: raw_db.RawSource | None = None,
):
    source_key = None if raw_source is None else os.fspath(raw_source)
    return _safe_vest_load_cached(int(shot), int(field), source_key)


def _geometry_root() -> Path:
    return resolve_data_root() / "geometry"


@lru_cache(maxsize=1)
def _load_equilibrium_magnetics_channels() -> list[dict[str, Any]]:
    with open(_geometry_root() / "MD.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["channels"]


def vest_equilibrium_magnetics_channel_definitions() -> tuple[dict[str, Any], ...]:
    """Return ordered VEST equilibrium-magnetics channel metadata for provenance/preflight."""
    return tuple(dict(channel) for channel in _load_equilibrium_magnetics_channels())


@lru_cache(maxsize=1)
def _load_fluctuation_mirnov_channels() -> list[dict[str, Any]]:
    with open(_geometry_root() / "FluctuationMirnov.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["channels"]


def fluctuation_mirnov_channel_definitions() -> tuple[dict[str, Any], ...]:
    """Return ordered VEST outboard fluctuation-Mirnov channel metadata for provenance."""
    return tuple(dict(channel) for channel in _load_fluctuation_mirnov_channels())


@lru_cache(maxsize=1)
def _load_static_channels() -> list[dict[str, Any]]:
    with open(_geometry_root() / "VEST_MagneticsGeometry_Full_ver_2302.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["channels"]


@lru_cache(maxsize=1)
def _load_names_by_code() -> dict[int, str]:
    with open(_geometry_root() / "table.yaml", "r", encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)["entries"]
    return {int(entry["field_code"]): str(entry["name"]) for entry in entries}


def _fallback_window(tstart: float, tend: float, dt: float) -> np.ndarray:
    if dt > 0:
        if tend <= tstart:
            return np.array([tstart], dtype=float)
        return np.arange(tstart, tend, dt)
    return np.linspace(0.0, 0.99996, 25_000)[6000:8501]


def _build_target_time(source_time: np.ndarray, tstart: float, tend: float, dt: float) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=float)
    if dt > 0 and source_time.size > 0:
        start = max(tstart, float(source_time[0]))
        end = min(tend, float(source_time[-1]))
        if end > start:
            return np.arange(start, end, dt)
    if dt <= 0 and source_time.size > 0:
        return source_time
    return _fallback_window(tstart, tend, dt)


@dataclass(frozen=True)
class _MagneticsContext:
    source_time: np.ndarray
    target_time: np.ndarray
    flux_loops: list[np.ndarray]
    probes: list[np.ndarray]


def _prepare_magnetics_context(
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    processing_config: VestMagneticsProcessingConfig | None,
    raw_source: raw_db.RawSource | None = None,
    allow_missing_channels: bool = False,
) -> _MagneticsContext:
    source_time, flux_loops, probes = vfit_equilibrium_magnetics(
        shot,
        processing_config=processing_config,
        raw_source=raw_source,
        allow_missing_channels=allow_missing_channels,
    )
    source_time = np.asarray(source_time, dtype=float)
    return _MagneticsContext(
        source_time=source_time,
        target_time=_build_target_time(source_time, tstart, tend, dt),
        flux_loops=flux_loops,
        probes=probes,
    )


def _interpolate_signal(target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    if source_time.size <= 1 or values.size <= 1:
        raise ValueError("Cannot interpolate a signal with fewer than two samples")
    return np.interp(target_time, source_time, values)


def _raw_time_data_with_validity(
    shot: int,
    field_code: int,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        time, data = raw_db.require_signal(
            _safe_vest_load(shot, field_code, raw_source),
            shot=shot,
            field=field_code,
            signal_name="raw Mirnov voltage",
        )
    except raw_db.RawSignalUnavailableError:
        return np.array([], dtype=float), np.array([], dtype=float), -2
    return time, data, 0


def _crop_native_window(
    time: np.ndarray,
    data: np.ndarray,
    *,
    tstart: float | None,
    tend: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a native-rate waveform to the half-open analysis window."""
    if tstart is None or tend is None:
        return np.asarray(time, dtype=float), np.asarray(data, dtype=float)
    time_array = np.asarray(time, dtype=float)
    data_array = np.asarray(data, dtype=float)
    if time_array.shape != data_array.shape:
        raise ValueError("Raw voltage time and data arrays must have identical shapes")
    keep = (time_array >= tstart) & (time_array < tend)
    return time_array[keep], data_array[keep]


def _set_voltage_signal(
    ods: object,
    base_path: str,
    time: np.ndarray,
    data: np.ndarray,
    validity: int,
) -> None:
    set_path(ods, f"{base_path}.voltage.time", np.asarray(time, dtype=float))
    set_path(ods, f"{base_path}.voltage.data", np.asarray(data, dtype=float))
    set_path(ods, f"{base_path}.voltage.validity", int(validity))


def _polyfit_baseline(time_axis: np.ndarray, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    valid = indices[(indices >= 0) & (indices < values.size)]
    if valid.size < 2:
        return np.zeros(values.size, dtype=float)
    return np.polyval(np.polyfit(time_axis[valid], values[valid], 1), time_axis)


def _plasma_processing_for_shot(shot: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve nested plasma-current processing eras declared in vest.yaml."""
    config = resolve_vest_diagnostic(shot, "plasma_current")
    processing = config["processing"]

    def nested(name: str) -> dict[str, Any]:
        item = processing[name]
        return resolve_shot_revisions(
            {key: value for key, value in item.items() if key != "revisions"},
            item.get("revisions"),
            shot,
            context=f"VEST plasma_current {name}",
        )

    return config, nested("reference"), nested("baseline"), nested("sign")


def vfit_plasma_current(
    shot: int,
    ref: int = -1,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the processed plasma-current waveform."""
    config, reference_config, baseline_config, sign_config = _plasma_processing_for_shot(shot)
    plasma_field = int(config["source"]["field"])
    if ref == -1:
        x_flux_loop = int(reference_config["field"])
        ind_mutual = float(reference_config["mutual_inductance"])

        time, raw_ip = raw_db.require_signal(
            _safe_vest_load(shot, plasma_field, raw_source),
            shot=shot,
            field=plasma_field,
            signal_name="plasma-current Rogowski coil",
        )
        flux_time, raw_flux = raw_db.require_signal(
            _safe_vest_load(shot, x_flux_loop, raw_source),
            shot=shot,
            field=x_flux_loop,
            signal_name="plasma-current flux compensation",
        )
        if flux_time.size != time.size or not np.allclose(flux_time, time):
            raw_flux = np.interp(time, flux_time, raw_flux)

        x_time = np.arange(
            int(baseline_config["analysis_start"]), int(baseline_config["analysis_end"])
        )
        x_window = int(baseline_config["lookback"])
        x_base = np.arange(x_time[0] - x_window, x_time[0] + 1, dtype=int)
        x_base = x_base[(x_base >= 0) & (x_base < time.size)]
        if x_base.size < 2:
            x_base = np.arange(min(500, time.size), dtype=int)

        calibrated_ip = calibrate_vest_signal(raw_ip, config["calibration"])
        ip_shot = calibrated_ip - np.polyval(np.polyfit(time[x_base], calibrated_ip[x_base], 1), time)
        ip_ref = raw_flux * float(reference_config["flux_gain"]) / ind_mutual
        ip_ref = ip_ref - np.polyval(np.polyfit(time[x_base], ip_ref[x_base], 1), time)
        ip = (ip_shot - ip_ref) * float(sign_config["multiply"])
        return time, ip

    reference_source = resolve_vest_diagnostic(ref, "plasma_current")
    reference_time, reference_values = raw_db.require_signal(
        _safe_vest_load(ref, int(reference_source["source"]["field"]), raw_source),
        shot=ref,
        field=int(reference_source["source"]["field"]),
        signal_name="reference plasma-current Rogowski coil",
    )
    time, shot_values = raw_db.require_signal(
        _safe_vest_load(shot, plasma_field, raw_source),
        shot=shot,
        field=plasma_field,
        signal_name="plasma-current Rogowski coil",
    )
    if reference_time.size != time.size or not np.allclose(reference_time, time):
        reference_values = np.interp(time, reference_time, reference_values)

    comparison = config["processing"]["reference_comparison"]
    taps = signal.firwin(
        int(comparison["taps"]), float(comparison["cutoff_frequency"]),
        pass_zero="lowpass", fs=float(comparison["sample_rate"]),
    )
    plasma_current = -(
        calibrate_vest_signal(shot_values, config["calibration"])
        - calibrate_vest_signal(reference_values, reference_source["calibration"])
    )
    baseline_index = min(int(comparison["baseline_index"]), plasma_current.size - 1)
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


def vest_diamagnetic_flux(
    shot: int,
    plasma_start: float,
    plasma_end: float,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the corrected diamagnetic flux waveform."""
    field_code = 246 if shot < 37505 else 4 if shot < 38452 else 257
    temp_time, raw_values = raw_db.require_signal(
        _safe_vest_load(shot, field_code, raw_source),
        shot=shot,
        field=field_code,
        signal_name="diamagnetic flux",
    )

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

    dia_flux_final = -1000.0 * (dia_flux - baseline)
    dia_flux_final[end_index:] = 0.0
    return temp_time, dia_flux_final


def vfit_equilibrium_magnetics(
    shot: int,
    indices: list[int] | np.ndarray | None = None,
    processing_config: VestMagneticsProcessingConfig | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
    allow_missing_channels: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Process magnetic probe and flux-loop data using VAFT process helpers."""
    return vest_equilibrium_magnetics_signals(
        int(shot),
        _load_equilibrium_magnetics_channels(),
        lambda source_shot, field: _safe_vest_load(source_shot, field, raw_source),
        indices=indices,
        config=processing_config,
        allow_missing=allow_missing_channels,
    )


def _set_magnetics_properties(ods: object) -> None:
    set_path(ods, "magnetics.ids_properties.comment", "magnetics config from vest_magnetics")
    set_path(ods, "magnetics.ids_properties.homogeneous_time", 1)


def _set_magnetics_time(ods: object, target_time: np.ndarray) -> None:
    target = np.asarray(target_time, dtype=float)
    if path_exists(ods, "magnetics.time"):
        existing = np.asarray(get_path(ods, "magnetics.time"), dtype=float)
        if existing.shape != target.shape or not np.array_equal(existing, target):
            raise ValueError(
                "magnetics.time already exists with a different timebase; "
                "map signals together or use matching tstart/tend/dt settings"
            )
        return
    set_path(ods, "magnetics.time", target)


def _populate_flux_loop_static(ods: object) -> None:
    names = _load_names_by_code()
    geometry = _load_static_channels()
    flux_loop_index = 0
    for channel in geometry:
        if channel["kind"] != "flux_loop":
            continue
        field_code = int(channel["field_code"])
        name = names[field_code]
        r_pos = float(channel["r"])
        z_pos = float(channel["z"])
        set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.name", name)
        set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.identifier", name)
        set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.position.0.r", r_pos)
        set_path(ods, f"magnetics.flux_loop.{flux_loop_index}.position.0.z", z_pos)
        flux_loop_index += 1


def _populate_probe_static(ods: object) -> None:
    names = _load_names_by_code()
    probe_index = 0
    for channel in _load_static_channels():
        if channel["kind"] != "b_field_pol_probe":
            continue
        field_code = int(channel["field_code"])
        name = names[field_code]
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.name", name)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.identifier", name)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.r", float(channel["r"]))
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.z", float(channel["z"]))
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.phi", 0.0)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.length", PROBE_LENGTH)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.poloidal_angle", POLOIDAL_ANGLE)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.toroidal_angle", 0.0)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.type.index", MIRNOV_TYPE_INDEX)
        probe_index += 1

    for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS:
        name = str(channel["name"])
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.name", name)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.identifier", f"{name}:phase_reference")
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.r", float(channel["r"]))
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.z", float(channel["z"]))
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.phi", float(channel["phi"]))
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.length", PROBE_LENGTH)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.poloidal_angle", POLOIDAL_ANGLE)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.toroidal_angle", float(channel["toroidal_angle"]))
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.type.index", MIRNOV_TYPE_INDEX)
        probe_index += 1


def _populate_fluctuation_mirnov_static(ods: object) -> None:
    """Append the 45/135/225 deg outboard fluctuation-Mirnov array (issue #155).

    Continues the existing ``b_field_pol_probe`` index sequence so equilibrium
    probe ordering/indices are never shifted. Only called for shots at or
    after ``FLUCTUATION_MIRNOV_FIRST_SHOT``, since these probes are not
    physically wired before that shot.
    """
    probe_index = (
        len(get_path(ods, "magnetics.b_field_pol_probe"))
        if path_exists(ods, "magnetics.b_field_pol_probe")
        else 0
    )
    for channel in _load_fluctuation_mirnov_channels():
        identifier = str(channel["identifier"])
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.name", identifier)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.identifier", identifier)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.r", OUTBOARD_MIRNOV_MAJOR_RADIUS)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.position.z", float(channel["z"]))
        set_path(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.position.phi",
            math.radians(float(channel["toroidal_angle_deg"])),
        )
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.length", PROBE_LENGTH)
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.poloidal_angle", POLOIDAL_ANGLE)
        set_path(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.toroidal_angle",
            math.radians(float(channel["toroidal_angle_deg"])),
        )
        set_path(ods, f"magnetics.b_field_pol_probe.{probe_index}.type.index", MIRNOV_TYPE_INDEX)
        probe_index += 1


def _map_fluctuation_mirnov_voltage(
    ods: object,
    shot: int,
    *,
    start_index: int,
    raw_source: raw_db.RawSource | None = None,
    tstart: float | None = None,
    tend: float | None = None,
) -> None:
    """Populate native-rate raw voltage for the fluctuation-Mirnov array.

    ``start_index`` must match the first index written by
    :func:`_populate_fluctuation_mirnov_static`. Mirrors
    :func:`vfit_mirnov_raw_dynamic`'s crop-without-resample policy. Only
    called for shot >= ``FLUCTUATION_MIRNOV_FIRST_SHOT``.
    """
    probe_index = start_index
    for channel in _load_fluctuation_mirnov_channels():
        time, data, validity = _raw_time_data_with_validity(shot, int(channel["field"]), raw_source)
        if validity == 0:
            time, data = _crop_native_window(time, data, tstart=tstart, tend=tend)
        _set_voltage_signal(ods, f"magnetics.b_field_pol_probe.{probe_index}", time, data, validity)
        probe_index += 1


def _populate_limiter_shunt_static(ods: object) -> None:
    """Populate electrical limiter monitors without inventing endpoint geometry."""
    for index, channel in enumerate(LIMITER_SHUNT_CHANNELS):
        base_path = f"magnetics.shunt.{index}"
        set_path(ods, f"{base_path}.name", str(channel["name"]))
        set_path(ods, f"{base_path}.identifier", str(channel["identifier"]))
        set_path(ods, f"{base_path}.resistance", LIMITER_SHUNT_RESISTANCE)


def vfit_magnetics_static(ods: object) -> None:
    """Populate static magnetics metadata from YAML geometry assets."""
    _set_magnetics_properties(ods)
    _populate_flux_loop_static(ods)
    _populate_probe_static(ods)
    _populate_limiter_shunt_static(ods)


def vfit_mirnov_raw_dynamic(
    ods: object,
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
    tstart: float | None = None,
    tend: float | None = None,
) -> None:
    """Populate raw Mirnov voltage traces at their native acquisition timebase.

    When an analysis window is supplied, samples are cropped to
    ``tstart <= time < tend`` without interpolation or downsampling.
    """
    probe_index = 0
    for channel in _load_static_channels():
        if channel["kind"] != "b_field_pol_probe":
            continue
        time, data, validity = _raw_time_data_with_validity(shot, int(channel["field_code"]), raw_source)
        if validity == 0:
            time, data = _crop_native_window(time, data, tstart=tstart, tend=tend)
        _set_voltage_signal(ods, f"magnetics.b_field_pol_probe.{probe_index}", time, data, validity)
        probe_index += 1

    for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS:
        time, data, validity = _raw_time_data_with_validity(shot, int(channel["field_code"]), raw_source)
        if validity == 0:
            time, data = _crop_native_window(time, data, tstart=tstart, tend=tend)
        _set_voltage_signal(ods, f"magnetics.b_field_pol_probe.{probe_index}", time, data, validity)
        probe_index += 1


def _baseline_correct_limiter_voltage(
    time: np.ndarray, data: np.ndarray
) -> np.ndarray | None:
    """Remove the robust 0.0--0.2 s pre-plasma baseline from a shunt voltage."""
    start, end = LIMITER_SHUNT_BASELINE_WINDOW
    time_array = np.asarray(time, dtype=float)
    data_array = np.asarray(data, dtype=float)
    samples = data_array[(time_array >= start) & (time_array < end)]
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None
    return data_array - np.median(samples)


def vfit_limiter_shunts_dynamic(
    ods: object,
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Map baseline-corrected limiter-monitor voltages at their native timebase.

    Stored voltage divided by ``magnetics.shunt[].resistance`` reconstructs
    Pearson Model 411 monitor current under VAFT's effective-V/I convention.
    """
    for index, channel in enumerate(LIMITER_SHUNT_CHANNELS):
        time, data, validity = _raw_time_data_with_validity(
            shot, int(channel["field_code"]), raw_source
        )
        if validity == 0:
            corrected = _baseline_correct_limiter_voltage(time, data)
            if corrected is None:
                time = np.array([], dtype=float)
                data = np.array([], dtype=float)
                validity = -2
            else:
                data = corrected
        _set_voltage_signal(ods, f"magnetics.shunt.{index}", time, data, validity)


def _map_flux_loops(ods: object, context: _MagneticsContext) -> None:
    _set_magnetics_time(ods, context.target_time)
    for index, values in enumerate(context.flux_loops):
        if np.asarray(values).size < 2:
            continue
        data = _interpolate_signal(context.target_time, context.source_time, values) * 2 * math.pi
        set_path(ods, f"magnetics.flux_loop.{index}.flux.time", context.target_time)
        set_path(ods, f"magnetics.flux_loop.{index}.flux.data", data)


def _map_probes(
    ods: object,
    shot: int,
    context: _MagneticsContext,
    raw_source: raw_db.RawSource | None = None,
    tstart: float | None = None,
    tend: float | None = None,
) -> None:
    _set_magnetics_time(ods, context.target_time)
    # Static geometry assets may carry scalar NaN placeholders for probes that
    # have no mapped processed field. In heterogeneous mode they are treated
    # as malformed dynamic signals, so omit them rather than assigning a time
    # coordinate to a non-waveform.
    probe_count = (
        len(get_path(ods, "magnetics.b_field_pol_probe"))
        if path_exists(ods, "magnetics.b_field_pol_probe")
        else 0
    )
    for index in range(probe_count):
        data_path = f"magnetics.b_field_pol_probe.{index}.field.data"
        if not path_exists(ods, data_path):
            continue
        data = np.asarray(get_path(ods, data_path))
        if data.ndim == 0:
            set_path(ods, data_path, np.array([], dtype=float))
            set_path(
                ods,
                f"magnetics.b_field_pol_probe.{index}.field.time",
                np.array([], dtype=float),
            )
    mapped_probe_count = len(context.probes)
    for index, values in enumerate(context.probes):
        if np.asarray(values).size < 2:
            set_path(
                ods,
                f"magnetics.b_field_pol_probe.{index}.field.time",
                np.array([], dtype=float),
            )
            set_path(
                ods,
                f"magnetics.b_field_pol_probe.{index}.field.data",
                np.array([], dtype=float),
            )
            continue
        data = _interpolate_signal(context.target_time, context.source_time, values)
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.field.time", context.target_time)
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.field.data", data)
    vfit_mirnov_raw_dynamic(
        ods, shot, raw_source=raw_source, tstart=tstart, tend=tend
    )
    # Captured before the fluctuation-Mirnov array (if any) is appended below,
    # so the toroidal-reference "explicitly empty field" loop never reaches
    # into the fluctuation probes -- those carry no `field` node at all.
    toroidal_reference_end = len(get_path(ods, "magnetics.b_field_pol_probe"))
    if int(shot) >= FLUCTUATION_MIRNOV_FIRST_SHOT:
        fluctuation_start_index = toroidal_reference_end
        _populate_fluctuation_mirnov_static(ods)
        _map_fluctuation_mirnov_voltage(
            ods,
            shot,
            start_index=fluctuation_start_index,
            raw_source=raw_source,
            tstart=tstart,
            tend=tend,
        )
    # Toroidal reference probes are raw-voltage-only channels; their processed
    # field signal is explicitly empty, not an IMAS scalar NaN placeholder.
    for index in range(mapped_probe_count, toroidal_reference_end):
        set_path(
            ods,
            f"magnetics.b_field_pol_probe.{index}.field.time",
            np.array([], dtype=float),
        )
        set_path(
            ods,
            f"magnetics.b_field_pol_probe.{index}.field.data",
            np.array([], dtype=float),
        )


def _map_ip(ods: object, target_time: np.ndarray, ip_time: np.ndarray, ip: np.ndarray) -> None:
    _set_magnetics_time(ods, target_time)
    set_path(ods, "magnetics.ip.0.data", _interpolate_signal(target_time, ip_time, ip))
    set_path(ods, "magnetics.ip.0.time", target_time)


def _plasma_window(
    ods: object,
    shot: int,
    target_time: np.ndarray,
    ip_time: np.ndarray,
    ip: np.ndarray,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[float, float]:
    halpha = _safe_vest_load(shot, 101, raw_source)
    if halpha is not None and len(halpha[1]) > 1:
        h_time = np.asarray(halpha[0], dtype=float)
        h_data = smooth(np.asarray(halpha[1], dtype=float), 10)
        index_a = int(np.argmin(np.abs(h_time - 0.3)))
        index_b = int(np.argmin(np.abs(h_time - 0.36)))
        window = h_data[index_a:index_b] if index_b > index_a else h_data
        minimum = float(np.min(window)) if window.size > 0 else -1.0
        if minimum != 0.0:
            normalized = h_data / minimum
            tstart2, tend2 = detect_active_window(
                h_time[index_a:index_b], normalized[index_a:index_b]
            )
        else:
            tstart2, tend2 = vfit_plasma_mgods_startend(ods)
    else:
        tstart2, tend2 = vfit_plasma_mgods_startend(ods)

    if tstart2 < 0 or tend2 <= tstart2:
        temporary: dict[str, Any] = {}
        _map_ip(temporary, target_time, ip_time, ip)
        tstart2, tend2 = vfit_plasma_mgods_startend(temporary)
    return tstart2, tend2


def _map_diamagnetic_flux(
    ods: object,
    shot: int,
    target_time: np.ndarray,
    ip_time: np.ndarray,
    ip: np.ndarray,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    tstart2, tend2 = _plasma_window(ods, shot, target_time, ip_time, ip, raw_source)

    time_dia, dia_flux = vest_diamagnetic_flux(shot, tstart2, tend2, raw_source=raw_source)
    set_path(ods, "magnetics.diamagnetic_flux.0.data", _interpolate_signal(target_time, time_dia, dia_flux))
    set_path(ods, "magnetics.diamagnetic_flux.0.time", target_time)


def vfit_magnetics_dynamic(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    processing_config: VestMagneticsProcessingConfig | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    """Populate dynamic magnetics nodes from required raw waveforms."""
    context = _prepare_magnetics_context(
        shot,
        tstart,
        tend,
        dt,
        processing_config,
        raw_source,
        allow_missing_channels=True,
    )
    if target_time is not None:
        context = _MagneticsContext(
            source_time=context.source_time,
            target_time=np.asarray(target_time, dtype=float),
            flux_loops=context.flux_loops,
            probes=context.probes,
        )
    _map_flux_loops(ods, context)
    _map_probes(ods, shot, context, raw_source, tstart=tstart, tend=tend)
    vfit_limiter_shunts_dynamic(ods, shot, raw_source=raw_source)
    ip_time, ip = vfit_plasma_current(shot, raw_source=raw_source)
    _map_ip(ods, context.target_time, ip_time, ip)
    _map_diamagnetic_flux(ods, shot, context.target_time, ip_time, ip, raw_source)


def vfit_magnetics_for_shot(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    processing_config: VestMagneticsProcessingConfig | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Populate canonical static and dynamic magnetics nodes for one shot."""
    # Create the full ordered channel structures first so a missing early
    # channel can remain empty without shifting or invalidating later channels.
    vfit_magnetics_static(ods)
    vfit_magnetics_dynamic(
        ods,
        shot,
        tstart,
        tend,
        dt,
        processing_config=processing_config,
        raw_source=raw_source,
    )


def magnetics(
    ods: object,
    shot: int,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = DEFAULT_DT,
    processing_config: VestMagneticsProcessingConfig | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Canonical machine_mapping entry point for the magnetics IDS."""
    vfit_magnetics_for_shot(
        ods,
        shot,
        tstart,
        tend,
        dt,
        processing_config=processing_config,
        raw_source=raw_source,
    )


def flux_loop_from_raw_database(
    ods: object,
    shot: int,
    *,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = DEFAULT_DT,
    processing_config: VestMagneticsProcessingConfig | None = None,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Map calibrated flux-loop signals and metadata from the VEST archive."""
    context = _prepare_magnetics_context(shot, tstart, tend, dt, processing_config, raw_source)
    _set_magnetics_properties(ods)
    _populate_flux_loop_static(ods)
    _map_flux_loops(ods, context)


def b_field_pol_probe_from_raw_database(
    ods: object,
    shot: int,
    *,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = DEFAULT_DT,
    processing_config: VestMagneticsProcessingConfig | None = None,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Map calibrated and raw poloidal-field probe signals and metadata."""
    context = _prepare_magnetics_context(shot, tstart, tend, dt, processing_config, raw_source)
    _set_magnetics_properties(ods)
    _populate_probe_static(ods)
    _map_probes(ods, shot, context, raw_source)


def ip_rogowski_coil_from_raw_database(
    ods: object,
    shot: int,
    *,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = DEFAULT_DT,
    processing_config: VestMagneticsProcessingConfig | None = None,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Map the processed plasma-current Rogowski signal."""
    del processing_config
    ip_time, ip = vfit_plasma_current(shot, raw_source=raw_source)
    target_time = _build_target_time(ip_time, tstart, tend, dt)
    _set_magnetics_properties(ods)
    _map_ip(ods, target_time, ip_time, ip)


def diamagnetic_flux_rogowski_coil_from_raw_database(
    ods: object,
    shot: int,
    *,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = DEFAULT_DT,
    processing_config: VestMagneticsProcessingConfig | None = None,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    """Map the diamagnetic-flux Rogowski signal without adding plasma current."""
    del processing_config
    ip_time, ip = vfit_plasma_current(shot, raw_source=raw_source)
    target_time = _build_target_time(ip_time, tstart, tend, dt)
    _set_magnetics_properties(ods)
    _set_magnetics_time(ods, target_time)
    _map_diamagnetic_flux(ods, shot, target_time, ip_time, ip, raw_source)


def magnetics_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    processing_config = None
    raw_source = None
    if options and "processing_config" in options:
        processing_config = options["processing_config"]
    if options and "raw_source" in options:
        raw_source = options["raw_source"]
    magnetics(
        ods,
        shot,
        tstart,
        tend,
        dt,
        processing_config=processing_config,
        raw_source=raw_source,
    )


__all__ = [
    "LIMITER_SHUNT_CHANNELS",
    "LIMITER_SHUNT_BASELINE_WINDOW",
    "LIMITER_SHUNT_RESISTANCE",
    "b_field_pol_probe_from_raw_database",
    "diamagnetic_flux_rogowski_coil_from_raw_database",
    "flux_loop_from_raw_database",
    "ip_rogowski_coil_from_raw_database",
    "magnetics_from_raw_database",
    "vest_diamagnetic_flux",
    "vest_equilibrium_magnetics_channel_definitions",
    "fluctuation_mirnov_channel_definitions",
    "magnetics",
    "vfit_plasma_current",
    "vfit_equilibrium_magnetics",
    "vfit_magnetics_dynamic",
    "vfit_magnetics_for_shot",
    "vfit_magnetics_static",
    "vfit_limiter_shunts_dynamic",
    "vfit_mirnov_raw_dynamic",
    "vfit_plasma_mgods_startend",
]


VEST_DiamagneticFlux = vest_diamagnetic_flux
vfit_PlasmaCurrent = vfit_plasma_current
vfit_plasmaMGods_startend = vfit_plasma_mgods_startend
# Pre-rename names: kept as plain aliases so a direct
# `from vaft.machine_mapping.magnetics import ...` still works. The
# deprecation warning for the package-level path lives in
# vaft/machine_mapping/__init__.py's _LEGACY_REPLACEMENTS.
vfit_md = vfit_equilibrium_magnetics
vest_md_channel_definitions = vest_equilibrium_magnetics_channel_definitions
