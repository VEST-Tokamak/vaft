"""VEST 6 kW 2.45 GHz ECH power mapped into the IMAS ``ec_launchers`` IDS.

The source is the pair of slow-DAQ log-detector voltages on the ECH line at
port 5ML10: field 27 (forward) and field 28 (reflected), 25 kHz over 0-1 s,
recorded from shot ~29500 onward (issue #165).  Both go through the legacy
VFIT transfer function declared in ``vest.yaml``.

The data dictionary has no forward/reflected leaves, so the IDS stores the net
estimate ``forward - reflected`` in ``beam.power_launched``; the separate
traces are available from :func:`vest_ec_power`.

The launch condition -- position, steering angles -- comes from the EC
officer's CAD through ``vest.yaml`` ``0: ec_launchers`` (issue #266). It is
provisional until the vertical datum and the installation era are confirmed,
and the IDS comment says so. Mode, O-mode fraction, spot and phase stay unset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vaft.database import raw as raw_db
from vaft.process.signal_processing import resample_to_time

from .registry import PortMapError, port_phi
from .utils import (
    VestConfigurationError,
    _resolve_info_file_path,
    build_window_time_axis,
    calibrate_vest_signal,
    load_yaml,
    resolve_shot_revisions_with_provenance,
    resolve_vest_diagnostic,
    set_path,
)


#: Canonical diagnostic keys in ``vest.yaml`` ``0: diagnostics:``.
FORWARD_DIAGNOSTIC = "ech_6kw_forward"
REFLECTED_DIAGNOSTIC = "ech_6kw_reflected"

BEAM_NAME = "ECH 6 kW 2.45 GHz"
BEAM_IDENTIFIER = "5ML10"
#: Source frequency of the 6 kW magnetron line [Hz].
BEAM_FREQUENCY_HZ = 2.45e9
#: Key of the 6 kW line in ``vest.yaml`` ``0: ec_launchers: beams``.
BEAM_KEY = "ech_6kw"


def steering_angles_from_direction(direction: Any) -> tuple[float, float]:
    """IMAS ``(steering_angle_pol, steering_angle_tor)`` of a propagation direction.

    Parameters
    ----------
    direction : array_like, shape (3,)
        Wave vector ``(k_R, k_phi, k_Z)`` at the launch point, any length [-].

    Returns
    -------
    tuple of float
        ``angle_pol = atan2(-k_Z, -k_R)`` and ``angle_tor = arcsin(k_phi / |k|)``,
        the definitions of DD 3.41 ``ec_launchers.beam.steering_angle_*`` [rad].
        A beam aimed at the axis has both zero; aiming downward makes
        ``angle_pol`` positive, aiming toward ``+phi`` makes ``angle_tor``
        positive.
    """
    k = np.asarray(direction, dtype=float).reshape(-1)
    if k.shape != (3,) or not np.all(np.isfinite(k)):
        raise ValueError("direction must be three finite components (k_R, k_phi, k_Z)")
    norm = float(np.linalg.norm(k))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    k_r, k_phi, k_z = k / norm
    # "+ 0.0" folds atan2(-0.0, 1.0) = -0.0 into a plain zero.
    return float(np.arctan2(-k_z, -k_r)) + 0.0, float(np.arcsin(np.clip(k_phi, -1.0, 1.0))) + 0.0


def _vector3(value: Any, context: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise VestConfigurationError(f"{context} must be three numbers") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise VestConfigurationError(f"{context} must be three finite numbers")
    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise VestConfigurationError(f"{context} must be a unit vector (|v| = {norm:g})")
    return vector


def resolve_ec_launcher_geometry(
    shot: int,
    beam: str = BEAM_KEY,
    *,
    info_file: str | None = None,
) -> dict[str, Any] | None:
    """The launch condition of one EC beam for ``shot``, or ``None`` outside its era.

    Parameters
    ----------
    shot : int
        VEST shot number [-].
    beam : str, optional
        Key under ``vest.yaml`` ``0: ec_launchers: beams`` [-].
    info_file : str or None, optional
        Alternative ``vest.yaml`` [-].

    Returns
    -------
    dict or None
        ``r``, ``z`` [m], ``phi`` [rad, from the port map], ``direction`` and
        ``polarization`` (unit ``(R, phi, Z)`` vectors) [-],
        ``steering_angle_pol``/``steering_angle_tor`` [rad], ``status``,
        ``source``, ``launch_plane``, ``port`` and the applied revision.
        ``None`` when no geometry revision covers ``shot``.
    """
    content = load_yaml(_resolve_info_file_path(info_file))
    defaults = content.get("0") or content.get(0) or {}
    beams = ((defaults.get("ec_launchers") or {}).get("beams") or {}) if isinstance(defaults, Mapping) else {}
    context = f"vest.yaml ec_launchers.beams.{beam}"
    if beam not in beams or not isinstance(beams[beam], Mapping):
        raise VestConfigurationError(f"{context} is not configured")
    config = beams[beam]
    base = {key: value for key, value in config.items() if key != "revisions"}
    resolved, provenance = resolve_shot_revisions_with_provenance(
        base, config.get("revisions"), int(shot), context=context
    )
    if provenance["revision_index"] is None:
        return None

    position = resolved.get("launching_position") or {}
    try:
        r, z = float(position["r"]), float(position["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VestConfigurationError(f"{context}: launching_position needs numeric r and z") from exc
    if not (np.isfinite(r) and np.isfinite(z) and r > 0.0):
        raise VestConfigurationError(f"{context}: launching_position must be finite with r > 0")
    direction = _vector3(resolved.get("direction"), f"{context}: direction")
    polarization = _vector3(resolved.get("polarization"), f"{context}: polarization")
    if abs(float(direction @ polarization)) > 1e-6:
        raise VestConfigurationError(f"{context}: polarization must be transverse to direction")
    angle_pol, angle_tor = steering_angles_from_direction(direction)
    # phi always comes from the packaged port map, whatever `info_file` says.
    port = resolved.get("port")
    if not isinstance(port, str) or not port:
        raise VestConfigurationError(f"{context}: port must name a port in the port map")
    try:
        phi = port_phi(port)
    except PortMapError as exc:
        raise VestConfigurationError(f"{context}: {exc}") from exc
    return {
        "r": r,
        "z": z,
        "phi": phi,
        "port": port,
        "direction": direction,
        "polarization": polarization,
        "steering_angle_pol": angle_pol,
        "steering_angle_tor": angle_tor,
        "status": str(resolved.get("geometry_status", "unspecified")),
        "source": str(resolved.get("source", "")),
        "launch_plane": str(resolved.get("launch_plane", "")),
        "revision": provenance,
    }


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


def _valid_input_range(config: Mapping[str, Any], diagnostic: str) -> tuple[float, float]:
    processing = config.get("processing") or {}
    bounds = processing.get("valid_input_range")
    context = f"VEST diagnostic {diagnostic!r}: processing.valid_input_range"
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise VestConfigurationError(f"{context} must be a [low, high] pair")
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError) as exc:
        raise VestConfigurationError(f"{context} must be numeric") from exc
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise VestConfigurationError(f"{context} must be finite with low < high")
    return low, high


def _mask_detector_voltage(voltage: Any, config: Mapping[str, Any], diagnostic: str) -> np.ndarray:
    """Detector voltage with samples outside the valid input range set to NaN."""
    voltage = np.array(voltage, dtype=float)
    low, high = _valid_input_range(config, diagnostic)
    voltage[~(np.isfinite(voltage) & (voltage >= low) & (voltage <= high))] = np.nan
    return voltage


def _detector_power(voltage: Any, config: Mapping[str, Any], diagnostic: str) -> np.ndarray:
    """Calibrate detector voltage to power [W], NaN outside the valid input range."""
    voltage = np.asarray(voltage, dtype=float)
    low, high = _valid_input_range(config, diagnostic)
    valid = np.isfinite(voltage) & (voltage >= low) & (voltage <= high)
    # Calibrate only the valid samples: a masked spike never reaches the
    # exponent, so nothing overflows on the way to NaN.
    power = np.full(voltage.shape, np.nan, dtype=float)
    power[valid] = calibrate_vest_signal(voltage[valid], config["calibration"])
    return power


def _load_detector(
    shot: int,
    diagnostic: str,
    signal_name: str,
    raw_source: raw_db.RawSource | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = resolve_vest_diagnostic(shot, diagnostic)
    field = int(config["source"]["field"])
    time, voltage = raw_db.require_signal(
        _safe_vest_load(shot, field, raw_source),
        shot=shot,
        field=field,
        signal_name=signal_name,
    )
    return (
        np.asarray(time, dtype=float).reshape(-1),
        np.asarray(voltage, dtype=float).reshape(-1),
        config,
    )


def _ec_power_on(
    shot: int,
    raw_source: raw_db.RawSource | None,
    axis: Any,
) -> dict[str, Any]:
    """Shared implementation: ``axis`` is ``None`` (native) or a callable/array."""
    forward_time, forward_voltage, forward_config = _load_detector(
        shot, FORWARD_DIAGNOSTIC, "6 kW ECH forward power detector", raw_source
    )
    reflected_time, reflected_voltage, reflected_config = _load_detector(
        shot, REFLECTED_DIAGNOSTIC, "6 kW ECH reflected power detector", raw_source
    )
    if callable(axis):
        time = np.asarray(axis(forward_time), dtype=float).reshape(-1)
    elif axis is not None:
        time = np.asarray(axis, dtype=float).reshape(-1)
    else:
        time = forward_time

    # The mask acts on the native samples, before any resampling: a -5 V spike
    # interpolated or anti-alias filtered onto a coarser grid would smear into
    # neighbouring in-range voltages and calibrate to kilowatts that never
    # existed.  Masked samples become NaN, which resample_to_time treats as
    # gaps; the voltages (the band-limited physical signal) are what is
    # resampled, and the calibration below masks once more on the output grid.
    forward_voltage = _mask_detector_voltage(forward_voltage, forward_config, FORWARD_DIAGNOSTIC)
    reflected_voltage = _mask_detector_voltage(
        reflected_voltage, reflected_config, REFLECTED_DIAGNOSTIC
    )
    if time is not forward_time:
        forward_voltage = resample_to_time(forward_time, forward_voltage, time)
    if not (
        reflected_time.shape == time.shape
        and np.allclose(reflected_time, time, rtol=0.0, atol=1e-12)
    ):
        reflected_voltage = resample_to_time(reflected_time, reflected_voltage, time)

    forward = _detector_power(forward_voltage, forward_config, FORWARD_DIAGNOSTIC)
    reflected = _detector_power(reflected_voltage, reflected_config, REFLECTED_DIAGNOSTIC)
    return {
        "time": time,
        "forward": forward,
        "reflected": reflected,
        # NaN on either side leaves the net unknown, which is what it is.
        "net": forward - reflected,
        "fields": {
            "forward": int(forward_config["source"]["field"]),
            "reflected": int(reflected_config["source"]["field"]),
        },
        "valid_input_range": {
            "forward": _valid_input_range(forward_config, FORWARD_DIAGNOSTIC),
            "reflected": _valid_input_range(reflected_config, REFLECTED_DIAGNOSTIC),
        },
        "calibration": dict(forward_config["calibration"]),
    }


def vest_ec_power(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
    time: Any | None = None,
) -> dict[str, Any]:
    """Calibrated, masked 6 kW ECH forward and reflected power for one shot.

    Parameters
    ----------
    shot : int
        VEST shot number [-].
    raw_source : path or None, optional
        Archived raw dump; ``None`` reads the live VEST database [-].
    time : array_like or None, optional
        Output instants; ``None`` keeps the native slow-DAQ timebase [s].

    Returns
    -------
    dict
        ``time`` [s], ``forward``, ``reflected`` and ``net = forward -
        reflected`` [W] (NaN wherever a detector voltage lies outside its
        configured ``valid_input_range``), plus ``fields``,
        ``valid_input_range`` [V] and ``calibration`` provenance.

    Raises
    ------
    vaft.database.raw.RawSignalUnavailableError
        Field 27 or 28 is absent for the shot (before ~29500).
    """
    return _ec_power_on(int(shot), raw_source, time)


def _comment(power: Mapping[str, Any], geometry: Mapping[str, Any] | None) -> str:
    fields = power["fields"]
    low, high = power["valid_input_range"]["forward"]
    calibration = power["calibration"]
    return (
        f"VEST {BEAM_NAME} (port {BEAM_IDENTIFIER}) from slow-DAQ log-detector "
        f"voltages: field {fields['forward']} forward, field {fields['reflected']} "
        "reflected. Calibration (legacy VFIT): P[W] = "
        f"{calibration['scale']:g} * {calibration['base']:g}**((V - "
        f"{calibration['input_offset']:g}) / {calibration['slope']:g}). Detector "
        f"voltages outside [{low:g}, {high:g}] V are masked to NaN. "
        "beam.power_launched is a net launched-power estimate, forward minus "
        "reflected, measured upstream of the vessel and NaN where either is "
        "masked; the calibration is uncertified and values above the 6 kW "
        "rating occur. "
        + _geometry_comment(geometry)
    )


def _geometry_comment(geometry: Mapping[str, Any] | None) -> str:
    if geometry is None:
        return "Launching position and steering are unset: no geometry era covers this shot."
    return (
        f"Launching position ({geometry['launch_plane']}) and steering from "
        f"{geometry['source']}, status {geometry['status']}: z assumes the CAD "
        "shield centre is the midplane and its +Y is up, phi comes from the "
        f"port map ({geometry['port']}). Mode, O-mode fraction, spot and phase "
        "are not mapped."
    )


def ec_launchers_geometry(ods: object, geometry: Mapping[str, Any], time: Any) -> None:
    """Write one beam's fixed launch condition on ``time`` into ``ec_launchers.beam.0``.

    The launcher does not move, so every sample repeats the configured value;
    ``beam.time`` is the DD time base of these quantities.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    constant = lambda value: np.full(time.shape, float(value))  # noqa: E731
    set_path(ods, "ec_launchers.beam.0.time", time)
    set_path(ods, "ec_launchers.beam.0.launching_position.r", constant(geometry["r"]))
    set_path(ods, "ec_launchers.beam.0.launching_position.z", constant(geometry["z"]))
    set_path(ods, "ec_launchers.beam.0.launching_position.phi", constant(geometry["phi"]))
    set_path(ods, "ec_launchers.beam.0.steering_angle_pol", constant(geometry["steering_angle_pol"]))
    set_path(ods, "ec_launchers.beam.0.steering_angle_tor", constant(geometry["steering_angle_tor"]))


def ec_launchers_static(ods: object) -> None:
    # Per-beam signal time nodes and no root `ec_launchers.time`: the DD then
    # requires homogeneous_time = 0, as for barometry.
    set_path(ods, "ec_launchers.ids_properties.homogeneous_time", 0)
    set_path(ods, "ec_launchers.beam.0.name", BEAM_NAME)
    set_path(ods, "ec_launchers.beam.0.identifier", BEAM_IDENTIFIER)


def ec_launchers_dynamic(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    axis = (
        np.asarray(target_time, dtype=float)
        if target_time is not None
        else (lambda source_time: build_window_time_axis(source_time, tstart, tend, dt))
    )
    power = _ec_power_on(int(shot), raw_source, axis)
    time = power["time"]
    geometry = resolve_ec_launcher_geometry(int(shot))

    set_path(ods, "ec_launchers.ids_properties.comment", _comment(power, geometry))
    if geometry is not None:
        ec_launchers_geometry(ods, geometry, time)
    set_path(ods, "ec_launchers.beam.0.power_launched.time", time)
    set_path(ods, "ec_launchers.beam.0.power_launched.data", power["net"])
    set_path(ods, "ec_launchers.beam.0.frequency.time", time)
    set_path(ods, "ec_launchers.beam.0.frequency.data", np.full(time.shape, BEAM_FREQUENCY_HZ))


def ec_launchers(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    """Map the VEST 6 kW 2.45 GHz ECH power and launch condition into ``ec_launchers``.

    Raises :class:`vaft.database.raw.RawSignalUnavailableError` when field 27
    or 28 is absent; nothing is written as zeros.
    """
    ec_launchers_static(ods)
    ec_launchers_dynamic(
        ods, shot, tstart, tend, dt, raw_source=raw_source, target_time=target_time
    )


def ec_launchers_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    raw_source = options.get("raw_source") if options else None
    ec_launchers(ods, shot, tstart, tend, dt, raw_source=raw_source)


__all__ = [
    "ec_launchers",
    "ec_launchers_dynamic",
    "ec_launchers_from_raw_database",
    "ec_launchers_geometry",
    "ec_launchers_static",
    "resolve_ec_launcher_geometry",
    "steering_angles_from_direction",
    "vest_ec_power",
]
