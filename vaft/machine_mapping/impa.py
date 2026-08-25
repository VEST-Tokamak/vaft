"""VEST IMPA mapping: raw fields 114-121 to the magnetics IDS.

The array is processed from raw data for a single shot.  No reference or
vacuum shot is required at any point: the legacy pairing existed only so that
VFIT could build a plasma-only residual for equilibrium fitting, which is not
part of diagnostic calibration.

Machine constants live in ``vaft/machine_mapping/vest.yaml`` under
``magnetics.impa``; the algorithms live in :mod:`vaft.process.impa`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vaft.database import raw as raw_db
from vaft.process.impa import (
    ImpaProcessingConfig,
    ImpaResult,
    TfWindowCriteria,
    process_impa,
)

from .magnetics import PROBE_LENGTH, vfit_plasma_current
from .tf import vfit_tf_current
from .utils import _deep_merge, _normalize_shot_key, _resolve_info_file_path, load_yaml, path_exists, set_path

__all__ = [
    "HALL_PROBE_TYPE_INDEX",
    "IMPA_IDENTIFIER_PREFIX",
    "IMPA_PROBE_NODES",
    "impa",
    "impa_from_raw_database",
    "impa_probe_indices",
    "impa_probe_node",
    "load_impa_inputs",
    "process_impa_shot",
    "resolve_impa_config",
]

#: ``magnetics/magnetics_probe_type_identifier``: 2 = mirnov, 3 = hall.
HALL_PROBE_TYPE_INDEX = 3
#: Identifier prefix so consumers select IMPA semantically, never by index.
IMPA_IDENTIFIER_PREFIX = "impa:"
#: The array lands in the node matching how it is mounted.  A toroidally
#: aligned probe measures the toroidal field and belongs in
#: ``b_field_tor_probe``; only a poloidal mounting yields a Bz for
#: ``b_field_pol_probe``.
IMPA_PROBE_NODES = ("magnetics.b_field_pol_probe", "magnetics.b_field_tor_probe")
#: A midplane probe measuring the vertical field looks "up" in the poloidal plane.
IMPA_POLOIDAL_ANGLE = 0.0


def _safe_vest_load(shot: int, field: int, raw_source: raw_db.RawSource | None = None):
    return raw_db.vest_load(
        shot,
        field,
        sample_opt=False if raw_source is None else raw_source,
    )


def _vest_config(info_file: str | None) -> Mapping[str, Any]:
    # Deliberately uncached: the mapping file is small, this is not a hot path,
    # and a cache here would serve stale settings to anything that rewrites it.
    return load_yaml(_resolve_info_file_path(info_file))


def resolve_impa_config(shot: int, info_file: str | None = None) -> dict[str, Any]:
    """Return the IMPA block for ``shot``, merging shot overrides over defaults."""
    content = _vest_config(info_file)
    default_block = content.get("0") or content.get(0) or {}
    shot_block = content.get(_normalize_shot_key(shot), {}) or {}
    merged = _deep_merge(default_block, shot_block)
    impa_config = (merged.get("magnetics") or {}).get("impa")
    if not isinstance(impa_config, Mapping):
        raise ValueError(
            f"No IMPA configuration for shot {shot}; expected magnetics.impa in the VEST machine mapping"
        )
    return dict(impa_config)


def _channel_specs(config: Mapping[str, Any], shot: int) -> list[dict[str, Any]]:
    """Return the channels wired for ``shot``, in array order.

    A campaign may not wire every position -- the 2022-04-23 block runs seven
    channels, without field 121 -- so an era may narrow the default set.
    """
    channels = config.get("channels") or {}
    specs = []
    for key in sorted(channels, key=lambda value: int(value)):
        entry = dict(channels[key])
        entry["index"] = int(key)
        specs.append(entry)
    if not specs:
        raise ValueError("The IMPA configuration defines no channels")

    for era in (config.get("calibration") or {}).get("shot_era_overrides") or ():
        if not int(era.get("min_shot", 0)) <= int(shot) <= int(era.get("max_shot", 0)):
            continue
        wired = era.get("channels")
        if wired:
            allowed = {int(field) for field in wired}
            specs = [spec for spec in specs if int(spec["field"]) in allowed]
        break
    return specs


def _resolve_gain(calibration: Mapping[str, Any], shot: int) -> float:
    """Return the Hall gain for ``shot``, honouring verified shot-era ranges."""
    gain = float(calibration.get("gain", 2.0 / 15.0))
    for era in calibration.get("shot_era_overrides") or ():
        if int(era.get("min_shot", 0)) <= int(shot) <= int(era.get("max_shot", 0)):
            return float(era.get("gain", gain))
    return gain


def _processing_config(config: Mapping[str, Any], shot: int) -> ImpaProcessingConfig:
    processing = config.get("processing") or {}
    calibration = config.get("calibration") or {}
    tf_compensation = config.get("tf_compensation") or {}
    bounds = tf_compensation.get("tilt_bounds_deg", (-10.0, 10.0))
    return ImpaProcessingConfig(
        sample_rate=float(processing.get("sample_rate", 25_000.0)),
        signal_lowpass_hz=float(processing.get("signal_lowpass_hz", 250.0)),
        position_lowpass_hz=float(processing.get("position_lowpass_hz", 2_500.0)),
        gain=_resolve_gain(calibration, shot),
        baseline=str(processing.get("baseline", "first_sample")),
        baseline_samples=int(processing.get("baseline_samples", 2_500)),
        tf_turns=int(tf_compensation.get("tf_turns", 24)),
        tilt_bounds_deg=(float(min(bounds)), float(max(bounds))),
        orientation=str(tf_compensation.get("orientation", "toroidal")),
        alpha_tolerance=float(tf_compensation.get("alpha_tolerance", 0.15)),
    )


def _window_criteria(config: Mapping[str, Any]) -> TfWindowCriteria:
    window = config.get("calibration_window") or {}
    defaults = TfWindowCriteria()
    return TfWindowCriteria(
        tf_current_min=float(window.get("tf_current_min", defaults.tf_current_min)),
        tf_current_min_fraction=float(
            window.get("tf_current_min_fraction", defaults.tf_current_min_fraction)
        ),
        ip_max=float(window.get("ip_max", defaults.ip_max)),
        pf_current_max=float(window.get("pf_current_max", defaults.pf_current_max)),
        min_duration=float(window.get("min_duration", defaults.min_duration)),
        tf_dynamic_range_min=float(window.get("tf_dynamic_range_min", defaults.tf_dynamic_range_min)),
        max_relative_noise=float(window.get("max_relative_noise", defaults.max_relative_noise)),
        smoothing_samples=int(window.get("smoothing_samples", defaults.smoothing_samples)),
    )


def _resample(reference_time: np.ndarray, time: np.ndarray, values: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.size == reference_time.size and np.allclose(time, reference_time):
        return values
    return np.interp(reference_time, time, values)


def load_impa_inputs(
    shot: int,
    config: Mapping[str, Any],
    *,
    raw_source: raw_db.RawSource | None = None,
) -> dict[str, Any]:
    """Load the raw IMPA channels plus the same-shot TF, Ip and PF context.

    A missing IMPA channel is recorded as invalid rather than aborting the
    shot; missing Ip or PF context only disables the corresponding
    clean-window criterion, which the returned notes make explicit.
    """
    specs = _channel_specs(config, int(shot))
    notes: list[str] = []

    reference_time: np.ndarray | None = None
    raw_signals: list[np.ndarray | None] = []
    for spec in specs:
        field = int(spec["field"])
        try:
            time, values = raw_db.require_signal(
                _safe_vest_load(shot, field, raw_source),
                shot=shot,
                field=field,
                signal_name=f"raw IMPA voltage ({spec.get('label', field)})",
            )
        except raw_db.RawSignalUnavailableError as error:
            notes.append(str(error))
            raw_signals.append(None)
            continue
        if reference_time is None:
            reference_time = np.asarray(time, dtype=float)
        raw_signals.append(_resample(reference_time, time, values))

    if reference_time is None:
        fields = [int(spec["field"]) for spec in specs]
        raise raw_db.RawSignalUnavailableError(
            shot,
            fields[0],
            f"none of the IMPA channels {fields} returned a waveform",
            signal_name="raw IMPA voltage",
        )

    n_samples = reference_time.size
    raw = np.full((len(specs), n_samples), np.nan)
    channel_valid = np.zeros(len(specs), dtype=bool)
    for index, values in enumerate(raw_signals):
        if values is None:
            continue
        raw[index] = values
        channel_valid[index] = True

    tf_time, i_tf = vfit_tf_current(shot, raw_source=raw_source)
    i_tf = _resample(reference_time, tf_time, i_tf)

    ip: np.ndarray | None = None
    try:
        ip_time, ip_values = vfit_plasma_current(shot, raw_source=raw_source)
    except (raw_db.RawSignalUnavailableError, FileNotFoundError, ValueError) as error:
        notes.append(f"plasma current unavailable; Ip criterion skipped ({error})")
    else:
        ip = _resample(reference_time, ip_time, ip_values)

    pf_currents: np.ndarray | None = None
    try:
        from .pf_active import vfit_pf

        pf_time, pf_values = vfit_pf(shot, raw_source=raw_source)
    except Exception as error:  # noqa: BLE001 - PF context is optional here
        notes.append(f"PF currents unavailable; PF criterion skipped ({error})")
    else:
        pf_currents = np.vstack([_resample(reference_time, pf_time, values) for values in pf_values])

    return {
        "time": reference_time,
        "raw": raw,
        "channel_valid": channel_valid,
        "i_tf": i_tf,
        "ip": ip,
        "pf_currents": pf_currents,
        "specs": specs,
        "notes": notes,
    }


def process_impa_shot(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
    config: Mapping[str, Any] | None = None,
    reference_shot: int | None = None,
) -> tuple[ImpaResult, dict[str, Any]]:
    """Resolve configuration, load one shot and run the IMPA pipeline.

    ``reference_shot`` optionally names a TF reference taken with the array in
    the same position; its geometry and coupling are then applied to ``shot``
    instead of being re-fitted.  The single-shot path never requires one.
    """
    config = dict(config) if config is not None else resolve_impa_config(shot)
    reference = None
    if reference_shot is not None:
        reference, _ = process_impa_shot(
            int(reference_shot), raw_source=raw_source, config=config
        )
        if not reference.quality.is_usable:
            raise ValueError(
                f"IMPA reference shot {reference_shot} is not usable as a calibration "
                f"({reference.quality.status}): {'; '.join(reference.quality.reasons)}"
            )
    inputs = load_impa_inputs(shot, config, raw_source=raw_source)
    geometry = config.get("geometry") or {}
    quality = config.get("quality") or {}
    r_bounds = geometry.get("r_bounds", (0.1, 0.8))
    configured_r = geometry.get("r")

    result = process_impa(
        inputs["time"],
        inputs["raw"],
        inputs["i_tf"],
        config=_processing_config(config, int(shot)),
        criteria=_window_criteria(config),
        ip=inputs["ip"],
        pf_currents=inputs["pf_currents"],
        channel_valid=inputs["channel_valid"],
        r=np.asarray(configured_r, dtype=float) if configured_r else None,
        z=float(geometry.get("z", 0.0)),
        pitch=float(geometry.get("radial_pitch", 0.05)),
        r_bounds=(float(min(r_bounds)), float(max(r_bounds))),
        r0_initial=float(geometry.get("r0_initial", 0.4)),
        max_normalized_rmse=float(quality.get("max_normalized_rmse", 0.1)),
        reference=reference,
    )
    object.__setattr__(result, "provenance", {**result.provenance, "shot": int(shot)})
    return result, inputs


def _probe_node(orientation: str) -> str:
    return "magnetics.b_field_tor_probe" if orientation == "toroidal" else "magnetics.b_field_pol_probe"


def _existing_probe_count(ods: Any, node: str) -> int:
    """Return the current probe-array length without creating ODS branches."""
    leaf = node.split(".")[-1]
    try:
        probes = ods[node] if not isinstance(ods, dict) else ods["magnetics"][leaf]
        return len(probes)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def impa_probe_node(ods: Any) -> str | None:
    """Return which magnetics node holds this ODS's IMPA channels, if any."""
    for node in IMPA_PROBE_NODES:
        if _impa_indices_in(ods, node):
            return node
    return None


def _impa_indices_in(ods: Any, node: str) -> list[int]:
    leaf = node.split(".")[-1]
    indices = []
    for index in range(_existing_probe_count(ods, node)):
        path = f"{node}.{index}.identifier"
        if not path_exists(ods, path):
            continue
        value = ods[path] if not isinstance(ods, dict) else ods["magnetics"][leaf][index]["identifier"]
        if str(value).startswith(IMPA_IDENTIFIER_PREFIX):
            indices.append(index)
    return indices


def impa_probe_indices(ods: Any, node: str | None = None) -> list[int]:
    """Return the probe indices holding IMPA channels, located by identifier.

    Searches both magnetics probe nodes unless ``node`` names one, since the
    array lands in whichever matches its mounting.
    """
    if node is not None:
        return _impa_indices_in(ods, node)
    for candidate in IMPA_PROBE_NODES:
        found = _impa_indices_in(ods, candidate)
        if found:
            return found
    return []


def _target_time(source_time: np.ndarray, tstart: float, tend: float, dt: float) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=float)
    if dt > 0 and source_time.size:
        start = max(float(tstart), float(source_time[0]))
        end = min(float(tend), float(source_time[-1]))
        if end > start:
            return np.arange(start, end, dt)
    return source_time


def _channel_status(result: ImpaResult, index: int) -> tuple[int, str]:
    """Return the IDS validity flag and a short reason for one channel."""
    if not bool(result.channel_valid[index]):
        return -2, "raw signal unavailable"
    if result.coupling is None or not np.isfinite(result.coupling.alpha[index]):
        return -1, "TF coupling could not be fitted"
    if not np.any(np.isfinite(result.b_z[index])):
        return -1, "compensated Bz is not finite"
    if result.quality.status == "invalid":
        return -1, "shot-level IMPA calibration was rejected"
    if result.quality.status == "warning":
        return 1, "calibration accepted with warnings"
    return 0, "valid"


def impa(
    ods: Any,
    shot: int,
    tstart: float = 0.26,
    tend: float = 0.36,
    dt: float = 4.0e-5,
    *,
    raw_source: raw_db.RawSource | None = None,
    reference_shot: int | None = None,
) -> dict[str, Any]:
    """Map one shot's calibrated IMPA measurement into the magnetics IDS.

    The eight channels are appended after the probes already present and are
    identified by ``identifier``; existing probe indices never move.  Returns a
    status dictionary carrying the calibration provenance and quality verdict.
    """
    result, inputs = process_impa_shot(
        int(shot), raw_source=raw_source, reference_shot=reference_shot
    )
    specs = inputs["specs"]
    orientation = str(result.provenance.get("orientation", "toroidal"))
    node = _probe_node(orientation)
    base = _existing_probe_count(ods, node)
    target_time = _target_time(result.time, tstart, tend, dt)

    channel_status: dict[str, Any] = {}
    for offset, spec in enumerate(specs):
        index = base + offset
        prefix = f"{node}.{index}"
        name = str(spec.get("label", f"IMPA {offset + 1:02d}"))
        validity, reason = _channel_status(result, offset)

        set_path(ods, f"{prefix}.name", name)
        set_path(ods, f"{prefix}.identifier", f"{IMPA_IDENTIFIER_PREFIX}{name}")
        set_path(ods, f"{prefix}.position.r", float(result.geometry.r[offset]))
        set_path(ods, f"{prefix}.position.z", float(result.geometry.z[offset]))
        set_path(ods, f"{prefix}.position.phi", 0.0)
        set_path(ods, f"{prefix}.length", PROBE_LENGTH)
        set_path(ods, f"{prefix}.poloidal_angle", IMPA_POLOIDAL_ANGLE)
        set_path(ods, f"{prefix}.toroidal_angle", 0.0)
        set_path(ods, f"{prefix}.type.index", HALL_PROBE_TYPE_INDEX)
        set_path(ods, f"{prefix}.type.name", "hall")
        set_path(ods, f"{prefix}.type.description", "VEST internal magnetic probe array (Hall probe)")

        # Raw provenance is always meaningful, even for a rejected calibration.
        if bool(result.channel_valid[offset]):
            set_path(ods, f"{prefix}.voltage.time", np.asarray(result.time, dtype=float))
            set_path(ods, f"{prefix}.voltage.data", np.asarray(inputs["raw"][offset], dtype=float))
        set_path(ods, f"{prefix}.voltage.validity", int(validity))
        set_path(ods, f"{prefix}.field.validity", int(validity))

        # A failed channel is left without a field waveform: a zero-filled
        # trace would be indistinguishable from a real measurement.
        if validity >= 0:
            # ``b_field_tor_probe.field`` is the toroidal field the probe
            # measures; only a poloidal mounting yields a compensated Bz.
            measurement = result.b_measured if orientation == "toroidal" else result.b_z
            values = np.asarray(measurement[offset], dtype=float)
            finite = np.isfinite(values)
            if finite.any():
                set_path(ods, f"{prefix}.field.time", target_time)
                set_path(
                    ods,
                    f"{prefix}.field.data",
                    np.interp(target_time, result.time[finite], values[finite]),
                )

        channel_status[name] = {
            "field": int(spec["field"]),
            "probe_index": index,
            "validity": int(validity),
            "reason": reason,
            "r": float(result.geometry.r[offset]),
            "z": float(result.geometry.z[offset]),
        }
        if result.coupling is not None:
            channel_status[name].update(
                {
                    "alpha": float(result.coupling.alpha[offset]),
                    "tilt_deg": float(result.coupling.tilt_deg[offset]),
                    "coupling_ratio": float(result.coupling.coupling_ratio[offset]),
                    # Only alpha/R is observable from a TF-only window, so the
                    # radius implied by assuming a toroidally aligned probe is
                    # what a future geometry survey can actually be checked
                    # against.
                    "implied_radius_alpha_unity": (
                        float(result.geometry.r[offset] / result.coupling.alpha[offset])
                        if np.isfinite(result.coupling.alpha[offset]) and result.coupling.alpha[offset] != 0
                        else float("nan")
                    ),
                    "nrmse": float(result.coupling.nrmse[offset]),
                    "r_squared": float(result.coupling.r_squared[offset]),
                    "bound_hit": bool(result.coupling.bound_hit[offset]),
                }
            )

    status: dict[str, Any] = {
        "shot": int(shot),
        "status": result.quality.status,
        "orientation": orientation,
        "ids_node": node,
        "checks": dict(result.quality.checks),
        "reasons": list(result.quality.reasons) + list(inputs["notes"]),
        "provenance": dict(result.provenance),
        "geometry_method": result.geometry.method,
        "channels": channel_status,
    }
    if result.geometry.r0 is not None:
        status["r0"] = float(result.geometry.r0)
    if result.geometry.nrmse is not None:
        status["geometry_nrmse"] = float(result.geometry.nrmse)
    return status


def impa_from_raw_database(
    ods: Any,
    shot: int,
    tstart: float = 0.26,
    tend: float = 0.36,
    dt: float = 4.0e-5,
    options: dict | None = None,
) -> dict[str, Any]:
    """Raw-database entry point matching the other VEST diagnostic mappers."""
    raw_source = options.get("raw_source") if options else None
    return impa(ods, shot, tstart, tend, dt, raw_source=raw_source)
