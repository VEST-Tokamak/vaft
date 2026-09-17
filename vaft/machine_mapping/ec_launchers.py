"""VEST 6 kW 2.45 GHz ECH power mapped into the IMAS ``ec_launchers`` IDS.

The source is the pair of slow-DAQ log-detector voltages on the ECH line at
port 5ML10: field 27 (forward) and field 28 (reflected), 25 kHz over 0-1 s,
recorded from shot ~29500 onward (issue #165).  Both go through the legacy
VFIT transfer function declared in ``vest.yaml``.

The data dictionary has no forward/reflected leaves, so the IDS stores the net
estimate ``forward - reflected`` in ``beam.power_launched``; the separate
traces are available from :func:`vest_ec_power`.

Launching position, mode and steering are deliberately left unset (issue #266).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vaft.database import raw as raw_db
from vaft.process.signal_processing import resample_to_time

from .utils import (
    VestConfigurationError,
    build_window_time_axis,
    calibrate_vest_signal,
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

    # Voltages, not powers, are brought onto the output grid: the detector
    # voltage is the band-limited physical signal, and the mask has to act on
    # it before the log calibration.  On the native 25 kHz grid this is a
    # bit-for-bit interpolation at the sample instants.
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


def _comment(power: Mapping[str, Any]) -> str:
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
        "rating occur."
    )


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

    set_path(ods, "ec_launchers.ids_properties.comment", _comment(power))
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
    """Map the VEST 6 kW 2.45 GHz ECH power into ``ec_launchers``.

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
    "ec_launchers_static",
    "vest_ec_power",
]
