"""Canonical magnetics mapping integrated under machine_mapping."""

from __future__ import annotations

import math
import os
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import integrate, signal

from vaft.database import raw as raw_db
from vaft.process.magnetics import (
    VestEquilibriumMagneticsResult,
    VestMagneticsProcessingConfig,
    vest_equilibrium_magnetics_detailed,
)
from vaft.process.signal_processing import (
    SignalRepairError,
    detect_active_window,
    detect_clipped_samples,
    repair_clipped_interval,
    smooth,
)

from .utils import (
    VestConfigurationError,
    _resolve_info_file_path,
    calibrate_vest_signal,
    get_path,
    load_yaml,
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
#: Orientation of VEST's poloidal probes, in the IMAS DD convention.
#:
#: The DD defines ``poloidal_angle`` as a **clockwise** theta-like angle from the
#: horizontal plane, zero when the sensor normal points toward increasing major
#: radius.  Clockwise from ``+R`` turns toward ``-Z``, so the sensitive axis is
#: ``(cos(poloidal_angle), -sin(poloidal_angle))`` in ``(R, Z)`` and a probe
#: measuring ``+Bz`` must be stored as ``3*pi/2``, not ``pi/2``.
#:
#: That the probes measure ``+Bz`` is established empirically **relative to the
#: PF coil current polarity**, not assumed.  The forward model takes the coil
#: currents and ``turns_with_sign`` as given, and both are VEST-native DAQ
#: signs, so a global inversion of the PF chain would flip every correlation
#: with it.  What the check establishes is the probe orientation *within* that
#: chain, which is what the stored angle describes; the absolute orientation is
#: the open question tracked in :mod:`vaft.machine_mapping.conventions`.
#: Forward-modelling the PF coil response with
#: :func:`vaft.formula.green.green_br_bz_exact` over the packaged
#: ``pf_active`` geometry and correlating it against the mapped probe signals
#: during the pre-plasma vacuum phase gives, on shot 39915, a median correlation
#: of +0.85 with 62 of 63 probes positive; shot 41672 gives +0.84 with 59 of 63.
#: See ``test_probe_orientation_is_established_from_the_coil_forward_model``.
#:
#: This value was ``pi/2`` until the audit for issue #288.  That was not a
#: standalone error: the consumer in :mod:`vaft.omas.vacuum_magnetics` projected
#: with ``(cos, +sin)``, the counter-clockwise reading, so the two cancelled and
#: VAFT's own analysis was self-consistent.  What was wrong was the value written
#: into the IDS, which told any DD-conformant reader that the probes measure
#: ``-Bz``.  The stored angle and the projection must move together.
POLOIDAL_ANGLE = 3 * math.pi / 2
MIRNOV_TYPE_INDEX = 2
# Poloidal-probe and flux-loop families, by position. These are the boundaries
# the EFIT k-file writer submits constraints by (vaft.code.efit.kfile), kept
# here with the rest of the probe geometry so a validation forward model and the
# reconstruction cannot disagree about what "inboard" means.
INBOARD_PROBE_MAX_R = 0.09
OUTBOARD_PROBE_MIN_R = 0.795
SIDE_PROBE_MIN_ABS_Z = 0.8
INBOARD_FLUX_LOOP_MAX_R = 0.15
OUTBOARD_FLUX_LOOP_MIN_R = 0.5
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


#: Every outboard fluctuation-Mirnov identifier, parsed strictly rather than by
#: substring. ``sub_array`` and ``position`` are *derived* at load time from the
#: identifier -- issue #155 forbids storing an ``L1``/``L2`` layer key in the
#: configuration, since the identifier already carries it.
_FLUCTUATION_IDENTIFIER = re.compile(
    r"^OutMirnov_(?P<angle>45|135|225)_(?P<sub_array>L[12])-(?P<position>0[1-5])$"
)
_FLUCTUATION_ARRAY_DEFAULTS = ("role", "preserve_native_voltage")


@lru_cache(maxsize=8)
def _fluctuation_mirnov_config(shot: int = 0) -> dict[str, Any]:
    """Return the resolved ``fluctuation_mirnov`` block from ``vest.yaml``.

    ``shot=0`` means the base, un-revised configuration, and this function
    enforces that rather than assuming it. :func:`resolve_vest_diagnostic`
    layers any matching ``revisions`` on top of the base entry, and
    :func:`resolve_shot_revisions` treats a missing ``from_shot`` as
    unbounded below -- so a ``to_shot``-only revision *would* match shot 0 and
    quietly change what "the inventory" means. Several things depend on it not
    doing that: the module-level constants derived at import, and
    :func:`fluctuation_mirnov_gain_by_identifier`, which takes no shot at all.
    So a revision without an explicit ``from_shot`` is rejected here.

    The channel table -- identifiers, geometry and gains alike -- is
    shot-independent today, which is why ``shot=0`` is the right default for
    any caller that only wants the inventory. The one genuinely shot-dependent
    value is ``first_operational_shot``, and that is a gate the caller applies,
    not a per-channel field. The ``shot`` parameter exists so a future
    calibration era can be expressed in ``revisions`` without another API
    change.
    """
    base = load_yaml(_resolve_info_file_path(None)).get(0, {})
    entry = base.get("diagnostics", {}).get("fluctuation_mirnov", {})
    for index, revision in enumerate(entry.get("revisions") or ()):
        if not isinstance(revision, Mapping) or revision.get("from_shot") is None:
            raise VestConfigurationError(
                f"VEST diagnostic 'fluctuation_mirnov' revision {index}: from_shot is "
                "required. Without it the revision also matches shot 0, which is "
                "reserved for the base inventory that the shot-independent channel "
                "and gain lookups read."
            )
    return resolve_vest_diagnostic(int(shot), "fluctuation_mirnov")


@lru_cache(maxsize=8)
def _load_fluctuation_mirnov_channels(shot: int = 0) -> tuple[dict[str, Any], ...]:
    """Validate and materialise the fluctuation-Mirnov channel table.

    Array-level defaults are merged down into freshly built per-channel dicts
    here -- the mapping :func:`resolve_vest_diagnostic` returned is never
    mutated -- so every consumer sees one uniform shape. See
    :func:`_fluctuation_mirnov_config` for what ``shot=0`` means.
    """
    config = _fluctuation_mirnov_config(int(shot))
    defaults = {key: config[key] for key in _FLUCTUATION_ARRAY_DEFAULTS if key in config}
    raw_unit = config.get("source", {}).get("raw_unit")

    channels: list[dict[str, Any]] = []
    seen_identifiers: dict[str, int] = {}
    seen_fields: dict[int, str] = {}
    for entry in config.get("channels", ()):
        identifier = str(entry["identifier"])
        context = f"VEST diagnostic 'fluctuation_mirnov' channel {identifier!r}"
        match = _FLUCTUATION_IDENTIFIER.fullmatch(identifier)
        if match is None:
            raise VestConfigurationError(
                f"{context}: identifier does not match the outboard Mirnov naming schema "
                f"{_FLUCTUATION_IDENTIFIER.pattern}"
            )
        angle = float(entry["toroidal_angle_deg"])
        if angle != float(match.group("angle")):
            raise VestConfigurationError(
                f"{context}: toroidal_angle_deg {angle} disagrees with the "
                f"{match.group('angle')} deg encoded in the identifier"
            )
        if identifier in seen_identifiers:
            raise VestConfigurationError(f"{context}: duplicate identifier")
        field = int(entry["field"])
        if field in seen_fields:
            raise VestConfigurationError(
                f"{context}: raw field {field} is already mapped by {seen_fields[field]!r}"
            )
        seen_identifiers[identifier] = field
        seen_fields[field] = identifier

        channel = dict(defaults)
        channel.update(entry)
        channel["identifier"] = identifier
        channel["field"] = field
        channel["toroidal_angle_deg"] = angle
        channel["z"] = float(entry["z"])
        channel["gain"] = float(entry["gain"])
        channel["sub_array"] = match.group("sub_array")
        channel["position"] = int(match.group("position"))
        if raw_unit is not None:
            channel.setdefault("unit", raw_unit)
        channels.append(channel)
    return tuple(channels)


def fluctuation_mirnov_channel_definitions(shot: int = 0) -> tuple[dict[str, Any], ...]:
    """Return ordered VEST outboard fluctuation-Mirnov channel metadata for provenance.

    Copies are returned so a caller can never corrupt the cached table.
    """
    return tuple(dict(channel) for channel in _load_fluctuation_mirnov_channels(int(shot)))


def select_fluctuation_mirnov_channels(
    *,
    role: str | None = "fluctuation",
    toroidal_angle_deg: float | Sequence[float] | None = None,
    sub_array: str | None = None,
    z_range: tuple[float, float] | None = None,
    shot: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Select fluctuation-Mirnov channels by role and geometry, in canonical order.

    This is the supported way for fluctuation and mode-analysis code to find
    probes: by what they are, not by where they happen to land in
    ``b_field_pol_probe``. Pair it with
    :func:`fluctuation_mirnov_probe_indices` to go from the selected
    identifiers to ODS indices.

    Unknown ``sub_array`` or ``toroidal_angle_deg`` values raise rather than
    quietly returning nothing -- a silent empty result reads like "no probes
    are installed there", which is a different and much more misleading answer
    than "you asked for something that does not exist".
    """
    channels = _load_fluctuation_mirnov_channels(int(shot))

    if sub_array is not None:
        known_sub_arrays = {channel["sub_array"] for channel in channels}
        if sub_array not in known_sub_arrays:
            raise ValueError(
                f"Unknown fluctuation-Mirnov sub-array {sub_array!r}; "
                f"expected one of {sorted(known_sub_arrays)}."
            )

    if z_range is not None and z_range[0] > z_range[1]:
        raise ValueError(
            f"z_range {z_range} is reversed; expected (low, high). The canonical "
            "channel order is z descending, so writing the bounds that way is an "
            "easy slip -- and it would otherwise select nothing at all."
        )

    angles: set[float] | None = None
    if toroidal_angle_deg is not None:
        # np.ndim, not isinstance: NumPy scalars (np.int64, np.float32, an
        # element of an angle array) are not int/float, and mode-analysis code
        # hands us exactly those.
        requested = (
            [float(toroidal_angle_deg)]
            if np.ndim(toroidal_angle_deg) == 0
            else [float(value) for value in np.atleast_1d(toroidal_angle_deg)]
        )
        known_angles = {channel["toroidal_angle_deg"] for channel in channels}
        unknown = sorted(set(requested) - known_angles)
        if unknown:
            raise ValueError(
                f"No fluctuation-Mirnov channels at toroidal angle(s) {unknown}; "
                f"expected one of {sorted(known_angles)}."
            )
        angles = set(requested)

    selected = []
    for channel in channels:
        if role is not None and channel.get("role") != role:
            continue
        if angles is not None and channel["toroidal_angle_deg"] not in angles:
            continue
        if sub_array is not None and channel["sub_array"] != sub_array:
            continue
        if z_range is not None and not (z_range[0] <= channel["z"] <= z_range[1]):
            continue
        selected.append(dict(channel))
    return tuple(selected)


def fluctuation_mirnov_probe_indices(ods: object, *, shot: int = 0) -> dict[str, int]:
    """Map fluctuation-Mirnov identifiers to their ``b_field_pol_probe`` indices.

    Resolves probes semantically, so nothing downstream has to hard-code a
    position such as ``b_field_pol_probe.68``. Probes absent from the ODS are
    simply absent from the result -- shots before
    ``FLUCTUATION_MIRNOV_FIRST_SHOT`` map none of them, which is legitimate --
    so callers that need a full array should check what they got back.
    """
    registered = {channel["identifier"] for channel in _load_fluctuation_mirnov_channels(int(shot))}
    if not path_exists(ods, "magnetics.b_field_pol_probe"):
        return {}

    indices: dict[str, int] = {}
    for index in range(len(get_path(ods, "magnetics.b_field_pol_probe"))):
        path = f"magnetics.b_field_pol_probe.{index}.identifier"
        if not path_exists(ods, path):
            continue
        identifier = str(get_path(ods, path))
        if identifier not in registered:
            continue
        if identifier in indices:
            raise VestConfigurationError(
                f"Fluctuation-Mirnov identifier {identifier!r} appears at both "
                f"b_field_pol_probe index {indices[identifier]} and {index}; "
                "identifier-based probe discovery requires unique identifiers."
            )
        indices[identifier] = index
    return indices


@lru_cache(maxsize=1)
def _fluctuation_mirnov_gains() -> dict[str, float]:
    gains: dict[str, float] = {}
    for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS:
        gains[f"{channel['name']}:phase_reference"] = float(channel["gain"])
    for channel in _load_fluctuation_mirnov_channels():
        gains[str(channel["identifier"])] = float(channel["gain"])
    return gains


def fluctuation_mirnov_gain_by_identifier() -> dict[str, float]:
    """Per-channel voltage gains for raw-Mirnov probes, keyed by ODS identifier.

    IMAS ``b_field_pol_probe`` has no calibration-factor node, so this gain
    metadata can only live in the channel registries, not in the ODS itself.
    Covers the toroidal phase-reference probes as well as the 30-channel
    outboard fluctuation array. Takes no ``shot``: these gains are not
    shot-dependent.

    A fresh dict is returned each call, for the same reason
    :func:`fluctuation_mirnov_channel_definitions` returns copies -- a caller
    that mutates the result must not be able to corrupt the table every later
    lookup reads, plot-layer gain resolution included.
    """
    return dict(_fluctuation_mirnov_gains())


#: Derived from the ``fluctuation_mirnov`` block in ``vest.yaml`` rather than
#: duplicated here, so the configuration stays the single source of truth.
FLUCTUATION_MIRNOV_FIRST_SHOT = int(_fluctuation_mirnov_config()["first_operational_shot"])
OUTBOARD_MIRNOV_MAJOR_RADIUS = float(_fluctuation_mirnov_config()["geometry"]["major_radius"])


class UnsupportedMagneticsGeometryError(NotImplementedError):
    """Raised when a shot needs a magnetics geometry this repository lacks."""


# Issue #195 section 5: the archived VFIT source loads
# `VEST_MagneticsGeometry_Full_ver_2310` for shot 39204 specifically and
# `ver_2409` for every other shot. This repository ships neither -- it
# carries ver_2302, which it applies to all shots. For most shots that is a
# documented, deliberate difference, but for 39204 the legacy source goes
# out of its way to override the geometry, so quietly handing it ver_2302
# would assign knowingly-wrong geometry. Fail clearly instead, until the
# 2310 geometry is imported as a repository-native asset. Deliberately not
# resolved by loading an external MATLAB file at runtime.
UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS: dict[int, str] = {39204: "2310"}


def require_supported_magnetics_geometry(shot: int | None) -> None:
    """Raise if *shot* requires a magnetics geometry version VAFT lacks."""
    if shot is None:
        return
    required = UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS.get(int(shot))
    if required is None:
        return
    raise UnsupportedMagneticsGeometryError(
        f"Shot {int(shot)} requires VEST magnetics geometry version {required}, "
        "which is not available in this repository (it ships ver_2302). The "
        "legacy VFIT source overrides the geometry for this shot specifically, "
        "so processing it with the shipped geometry would assign "
        "knowingly-incorrect sensor positions. Import the "
        f"ver_{required} geometry as a repository-native asset before "
        "processing this shot (issue #195)."
    )


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
    # Native-rate calibrated flux-loop terminal voltages, index-aligned with
    # `flux_loops` (empty arrays for unavailable channels).
    flux_loop_voltage_time: list[np.ndarray] = dataclass_field(default_factory=list)
    flux_loop_voltage: list[np.ndarray] = dataclass_field(default_factory=list)


def _prepare_magnetics_context(
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    processing_config: VestMagneticsProcessingConfig | None,
    raw_source: raw_db.RawSource | None = None,
    allow_missing_channels: bool = False,
) -> _MagneticsContext:
    result = vfit_equilibrium_magnetics_detailed(
        shot,
        processing_config=processing_config,
        raw_source=raw_source,
        allow_missing_channels=allow_missing_channels,
    )
    source_time = np.asarray(result.time, dtype=float)
    return _MagneticsContext(
        source_time=source_time,
        target_time=_build_target_time(source_time, tstart, tend, dt),
        flux_loops=result.flux_loops,
        probes=result.probes,
        flux_loop_voltage_time=result.flux_loop_voltage_time,
        flux_loop_voltage=result.flux_loop_voltage,
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


def _set_current_signal(
    ods: object,
    base_path: str,
    time: np.ndarray,
    data: np.ndarray,
    validity: int,
) -> None:
    set_path(ods, f"{base_path}.current.time", np.asarray(time, dtype=float))
    set_path(ods, f"{base_path}.current.data", np.asarray(data, dtype=float))
    set_path(ods, f"{base_path}.current.validity", int(validity))


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


def _plasma_current_baseline_indices(
    time: np.ndarray, baseline_config: dict[str, Any]
) -> np.ndarray:
    x_time = np.arange(
        int(baseline_config["analysis_start"]), int(baseline_config["analysis_end"])
    )
    x_window = int(baseline_config["lookback"])
    x_base = np.arange(x_time[0] - x_window, x_time[0] + 1, dtype=int)
    x_base = x_base[(x_base >= 0) & (x_base < time.size)]
    if x_base.size < 2:
        x_base = np.arange(min(500, time.size), dtype=int)
    return x_base


def _linear_baseline_subtract(time: np.ndarray, values: np.ndarray, x_base: np.ndarray) -> np.ndarray:
    return values - np.polyval(np.polyfit(time[x_base], values[x_base], 1), time)


def _apply_fl10_windowed_compensation(
    shot: int,
    time: np.ndarray,
    ip_shot: np.ndarray,
    reference_config: dict[str, Any],
    raw_source: raw_db.RawSource | None,
) -> np.ndarray:
    """Apply the later-era (46403-47116) FL10 compensation used in place of
    the legacy full-trace subtraction: decimate/gain/offset/smooth the FL10
    reference, interpolate onto the RC03 time grid, and subtract it from
    `ip_shot` only inside the documented compensation window.
    """
    fl10_config = reference_config["fl10"]
    fl10_field = int(fl10_config["field"])
    effective_resistance = float(reference_config["effective_resistance_ohm"])

    fl10_time, raw_fl10 = raw_db.require_signal(
        _safe_vest_load(shot, fl10_field, raw_source),
        shot=shot,
        field=fl10_field,
        signal_name="plasma-current FL10 reference (windowed compensation)",
    )

    shifted_time = fl10_time + float(fl10_config["time_offset_s"])
    decimate_factor = int(fl10_config["decimate_factor"])
    # MATLAB `decimate(temp2, 10)` defaults to an order-8 Chebyshev type I
    # filter, which is exactly what scipy's default `ftype="iir"` builds.
    decimated_flux = (
        signal.decimate(raw_fl10, decimate_factor) if decimate_factor > 1 else raw_fl10
    )
    decimated_time = shifted_time[::decimate_factor][: decimated_flux.size]

    # FL10 raw signal is a loop VOLTAGE here (never integrated), so the
    # divisor must be in ohms for the result to be a current -- see
    # `vest_flux_loop_voltage` for the integrated flux path (issue #214).
    ip_ref = decimated_flux * float(fl10_config["gain_numerator"]) / effective_resistance

    # Donor: `ipRef = ipRef - polyval(polyfit(time2(1), ipRef(175), 1), time2)`
    # (`vest_ip.m`). That is a degree-1 fit through a single (x, y) point, so
    # it is rank deficient. MATLAB solves it as V\y on the 1x2 Vandermonde
    # [x 1] via QR with column pivoting, which selects the larger-magnitude
    # column -- the constant column, since x = time2(1) ~ 0.26 < 1. The fit
    # therefore has zero slope and evaluates to the constant ipRef(175), so
    # this reduces to subtracting that single sample. Both the mechanism and
    # the evident intent agree, but this was reasoned from the source rather
    # than executed in MATLAB; the pinning test guards the convention.
    # `reference_offset_index` indexes the *decimated* array, as in the donor.
    offset_index = int(fl10_config["reference_offset_index"]) - 1  # 1-based -> 0-based
    offset_index = min(max(offset_index, 0), ip_ref.size - 1)
    ip_ref = ip_ref - ip_ref[offset_index]

    # Donor: `ipRef = smoothdata(ipRef, 10)`. Read strictly, smoothdata's
    # two-argument form takes a *dimension*, not a window length, so passing
    # 10 for a vector smooths along a singleton dimension and returns the
    # input unchanged -- i.e. the donor line is a no-op, and VAFT's legacy
    # `mode: subtract` path likewise applies no smoothing. The issue text
    # reads it as an intended 10-sample moving average, so the span stays
    # configurable: set `smooth_span: 1` to reproduce the donor literally.
    smooth_span = int(fl10_config["smooth_span"])
    ip_ref = smooth(ip_ref, smooth_span)

    # Donor uses `interp1(..., 'linear', 0)`: zero outside the FL10 record,
    # not edge-clamped as numpy would default to.
    ip_ref_interp = np.interp(time, decimated_time, ip_ref, left=0.0, right=0.0)

    window_start, window_end = (float(bound) for bound in fl10_config["subtract_window"])
    mask = (time >= window_start) & (time <= window_end)

    compensated = ip_shot.copy()
    compensated[mask] = ip_shot[mask] - ip_ref_interp[mask]
    return compensated


def vest_plasma_rogowski_current(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the calibration-only plasma-current Rogowski sensor current [A].

    This is the sensor-level physical quantity: the channel calibration and
    shot-era gain have been applied, but not the baseline removal, the FL10
    compensation, or the shot-era sign convention that together turn it into
    the plasma-current estimate.  It is the single definition of the Rogowski
    calibration used by both :func:`vfit_plasma_current` and the
    ``magnetics.rogowski_coil[0].current`` mapping (issue #215).
    """
    config, _, _, _ = _plasma_processing_for_shot(shot)
    plasma_field = int(config["source"]["field"])
    time, raw_ip = raw_db.require_signal(
        _safe_vest_load(shot, plasma_field, raw_source),
        shot=shot,
        field=plasma_field,
        signal_name="plasma-current Rogowski coil",
    )
    return time, calibrate_vest_signal(raw_ip, config["calibration"])


def vfit_plasma_current(
    shot: int,
    ref: int = -1,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the processed plasma-current waveform.

    Composition of :func:`vest_plasma_rogowski_current` with the shot-era
    baseline, FL10 compensation, and sign convention -- numerically unchanged
    by the issue-#215 split.
    """
    config, reference_config, baseline_config, sign_config = _plasma_processing_for_shot(shot)
    plasma_field = int(config["source"]["field"])
    if ref == -1:
        time, calibrated_ip = vest_plasma_rogowski_current(shot, raw_source=raw_source)
        x_base = _plasma_current_baseline_indices(time, baseline_config)
        ip_shot = _linear_baseline_subtract(time, calibrated_ip, x_base)

        mode = reference_config.get("mode", "subtract")
        if mode == "disabled":
            # Shot >= 47117 (#195): the FL10 waveform must not affect the
            # resulting plasma current at all.
            ip = ip_shot * float(sign_config["multiply"])
            return time, ip

        if mode == "subtract_fl10_windowed":
            # Shots 46403-47116 (#195): later-era FL10 acquisition path,
            # subtracted only inside the documented compensation window.
            ip_shot = _apply_fl10_windowed_compensation(
                shot, time, ip_shot, reference_config, raw_source
            )
            ip = ip_shot * float(sign_config["multiply"])
            return time, ip

        # Legacy full-trace subtraction (default, unchanged since #135).
        x_flux_loop = int(reference_config["field"])
        effective_resistance = float(reference_config["effective_resistance_ohm"])
        flux_time, raw_flux = raw_db.require_signal(
            _safe_vest_load(shot, x_flux_loop, raw_source),
            shot=shot,
            field=x_flux_loop,
            signal_name="plasma-current flux compensation",
        )
        if flux_time.size != time.size or not np.allclose(flux_time, time):
            raw_flux = np.interp(time, flux_time, raw_flux)

        # `raw_flux` is FL10's loop VOLTAGE, deliberately not integrated:
        # V / R gives the current-equivalent reference subtracted from the
        # Rogowski measurement.  Dividing by an inductance instead would give
        # A/s, which is not subtractable from a current (issue #214).
        #
        # The subtracted quantity is a proxy for the induced current in the
        # tungsten limiter surrounding the CS wall -- not a general vessel
        # eddy current: the plasma-current channel is the INNER Rogowski coil.
        # See vest.yaml; the method for computing that effective current is
        # under revision as of 2026-09-01.
        ip_ref = raw_flux * float(reference_config["flux_gain"]) / effective_resistance
        ip_ref = _linear_baseline_subtract(time, ip_ref, x_base)
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


def _diamagnetic_config(shot: int) -> dict[str, Any]:
    """Resolve the shot-era diamagnetic-Rogowski configuration from `vest.yaml`."""
    return resolve_vest_diagnostic(shot, "diamagnetic_flux")


def _saturation_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, stop)`` index pairs for each saturated run."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    stops = np.concatenate((indices[breaks], [indices[-1]]))
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _diamagnetic_saturation_report(
    shot: int,
    field_code: int,
    time: np.ndarray,
    mask: np.ndarray,
    limits: Mapping[str, Any],
    *,
    plasma_start: float | None = None,
    plasma_end: float | None = None,
    repaired: bool = False,
) -> dict[str, Any]:
    """Summarise which samples sat at the acquisition rail, and where."""
    runs = _saturation_runs(mask)
    n_saturated = int(mask.sum())
    in_window = 0
    if plasma_start is not None and plasma_end is not None and n_saturated:
        window = (time >= float(plasma_start)) & (time <= float(plasma_end))
        in_window = int(np.count_nonzero(mask & window))

    if n_saturated == 0:
        reason = "no sample reached the acquisition limit"
    elif not repaired:
        reason = f"{n_saturated} samples at the acquisition limit, not repaired"
    elif in_window:
        reason = (
            f"{n_saturated} samples reconstructed at the acquisition limit, "
            f"{in_window} of them inside the plasma window"
        )
    else:
        reason = (
            f"{n_saturated} samples reconstructed at the acquisition limit, "
            "none inside the plasma window"
        )

    return {
        "shot": int(shot),
        "field": int(field_code),
        "clip_values": [float(v) for v in limits.get("values", ())],
        "tolerance": float(limits["tolerance"]) if "tolerance" in limits else None,
        "n_samples": int(time.size),
        "n_saturated": n_saturated,
        "n_intervals": len(runs),
        "longest_run": max((b - a + 1 for a, b in runs), default=0),
        "first_time": float(time[runs[0][0]]) if runs else None,
        "last_time": float(time[runs[-1][1]]) if runs else None,
        "n_saturated_in_window": in_window,
        "plasma_start": None if plasma_start is None else float(plasma_start),
        "plasma_end": None if plasma_end is None else float(plasma_end),
        "repaired": bool(repaired and n_saturated),
        "reason": reason,
    }


def diamagnetic_saturation_report(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
    plasma_start: float | None = None,
    plasma_end: float | None = None,
) -> dict[str, Any]:
    """Report acquisition-limit saturation on the diamagnetic Rogowski channel.

    Detection only -- no integration, no repair. This is the seam a validity
    mapper should call (`magnetics.rogowski_coil[:].current.validity`, issue
    #215) so that saturation is detected in exactly one place.
    """
    config = _diamagnetic_config(shot)
    field_code = int(config["source"]["field"])
    limits = config["processing"]["saturation_repair"]
    temp_time, raw_values = raw_db.require_signal(
        _safe_vest_load(shot, field_code, raw_source),
        shot=shot,
        field=field_code,
        signal_name="diamagnetic flux",
    )
    mask = detect_clipped_samples(
        raw_values, clip_values=limits["values"], tolerance=float(limits["tolerance"])
    )
    return _diamagnetic_saturation_report(
        shot,
        field_code,
        temp_time,
        mask,
        limits,
        plasma_start=plasma_start,
        plasma_end=plasma_end,
        repaired=False,
    )


def vest_diamagnetic_rogowski_current(
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the calibrated hi-sensitivity TF-current Rogowski signal [A].

    This is the sensor-level quantity feeding the diamagnetic-flux
    calculation, before the reference-waveform subtraction that produces
    ``delta_i_tf``. ``delta_i_tf`` depends on the chosen plasma interval, so it
    is this signal that is stored as ``magnetics.rogowski_coil[1].current``
    (issue #215).

    It reads the same `diamagnetic_flux` configuration and applies the same
    `repair_clipped_interval` primitive as
    :func:`vest_diamagnetic_flux_detailed`, so the stored sensor current is the
    waveform the flux is actually derived from -- saturation repair of issue
    #285 included. It stops at the first integration rather than running the
    full triple-integration chain, and `test_rogowski_coil_mapping` pins the
    two against each other so the shared expression cannot drift.
    """
    config = _diamagnetic_config(shot)
    processing = config["processing"]
    limits = processing["saturation_repair"]
    field_code = int(config["source"]["field"])

    time, raw_values = raw_db.require_signal(
        _safe_vest_load(shot, field_code, raw_source),
        shot=shot,
        field=field_code,
        signal_name="diamagnetic hi-sensitivity TF Rogowski coil",
    )
    repaired, _ = repair_clipped_interval(
        time,
        raw_values,
        clip_value=limits["values"],
        tolerance=float(limits["tolerance"]),
        return_mask=True,
    )
    rogo_gain = -1 / float(processing["rogowski_shunt"])
    integrated = integrate.cumulative_trapezoid(repaired, time, initial=0.0) * rogo_gain
    return np.asarray(time, dtype=float), np.asarray(integrated, dtype=float)


def vest_diamagnetic_flux_detailed(
    shot: int,
    plasma_start: float,
    plasma_end: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    with_stages: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute the diamagnetic flux waveform and report input saturation.

    The raw Rogowski voltage is integrated three times on the way to
    `magnetics.diamagnetic_flux`, so a sample pinned at the acquisition rail is
    a known underestimate whose error does not average out. Saturated samples
    are reconstructed with the shared `repair_clipped_interval` primitive
    before any integration, and the outcome is returned rather than discarded.

    With ``with_stages=True`` the report also carries a ``"stages"`` mapping of
    the measured and intermediate waveforms, so the effect of the repair can be
    inspected at each integration without reimplementing the chain elsewhere.
    """
    config = _diamagnetic_config(shot)
    field_code = int(config["source"]["field"])
    processing = config["processing"]
    limits = processing["saturation_repair"]
    temp_time, raw_values = raw_db.require_signal(
        _safe_vest_load(shot, field_code, raw_source),
        shot=shot,
        field=field_code,
        signal_name="diamagnetic flux",
    )

    measured_values = np.asarray(raw_values, dtype=float).copy()
    # A SignalRepairError propagates deliberately: fabricating a waveform
    # would be worse than reporting that it is unrecoverable (cf. PF6 in
    # `vaft/machine_mapping/pf_active.py`).
    raw_values, saturated = repair_clipped_interval(
        temp_time,
        raw_values,
        clip_value=limits["values"],
        tolerance=float(limits["tolerance"]),
        return_mask=True,
    )
    report = _diamagnetic_saturation_report(
        shot,
        field_code,
        temp_time,
        saturated,
        limits,
        plasma_start=plasma_start,
        plasma_end=plasma_end,
        repaired=True,
    )
    if report["n_saturated_in_window"]:
        warnings.warn(
            f"Shot {shot}: the diamagnetic Rogowski channel (field {field_code}) "
            f"saturated at its acquisition limit on {report['n_saturated_in_window']} "
            "samples inside the plasma window; the reconstructed waveform there is "
            "an interpolation, not a measurement.",
            RuntimeWarning,
            stacklevel=2,
        )

    tf_circuit = processing["tf_circuit"]
    turn_tf = float(tf_circuit["turns"])
    ind_tf = float(tf_circuit["inductance"])
    res_tf = float(tf_circuit["resistance"])
    cap_tf = float(tf_circuit["capacitance"])
    rogo_gain = -1 / float(processing["rogowski_shunt"])

    integrated = integrate.cumulative_trapezoid(raw_values, temp_time, initial=0.0) * rogo_gain
    start_index = int(np.argmin(np.abs(temp_time - plasma_start)))
    end_index = int(np.argmin(np.abs(temp_time - plasma_end)))
    if end_index <= start_index:
        empty = np.zeros(temp_time.size, dtype=float)
        if with_stages:
            report["stages"] = {
                "time": temp_time,
                "raw": measured_values,
                "raw_repaired": raw_values,
                "saturated": saturated,
                "integrated": integrated,
                "delta_i_tf": empty,
                "flux": empty,
            }
        return temp_time, empty, report

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
    if with_stages:
        report["stages"] = {
            "time": temp_time,
            "raw": measured_values,
            "raw_repaired": raw_values,
            "saturated": saturated,
            "integrated": integrated,
            "delta_i_tf": delta_i_tf,
            "flux": dia_flux_final,
        }
    return temp_time, dia_flux_final, report


def vest_diamagnetic_flux(
    shot: int,
    plasma_start: float,
    plasma_end: float,
    *,
    raw_source: raw_db.RawSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the corrected diamagnetic flux waveform.

    Thin wrapper over `vest_diamagnetic_flux_detailed` for callers that do not
    need the saturation report.
    """
    time, flux, _ = vest_diamagnetic_flux_detailed(
        shot, plasma_start, plasma_end, raw_source=raw_source
    )
    return time, flux


def _equilibrium_magnetics_window_for_shot(shot: int) -> dict[str, Any]:
    """Resolve the shot-era equilibrium-magnetics acquisition policy from vest.yaml."""
    config = resolve_vest_diagnostic(shot, "equilibrium_magnetics")
    window = config["processing"]["window"]
    return resolve_shot_revisions(
        {key: value for key, value in window.items() if key != "revisions"},
        window.get("revisions"),
        shot,
        context="VEST equilibrium_magnetics window",
    )


def equilibrium_magnetics_processing_config(shot: int) -> VestMagneticsProcessingConfig:
    """Build the processing config for *shot* from its resolved vest.yaml era."""
    window = _equilibrium_magnetics_window_for_shot(shot)
    flux_window = window.get("flux_baseline_window")
    flux_samples = window.get("flux_baseline_samples")
    return VestMagneticsProcessingConfig(
        window_override=(
            int(window["index_start"]),
            int(window["index_end"]),
            int(window["probe_baseline_end"]),
        ),
        flux_baseline_window=(
            None if flux_window is None else (float(flux_window[0]), float(flux_window[1]))
        ),
        flux_baseline_samples=None if flux_samples is None else int(flux_samples),
        daq_mode=str(window["daq_mode"]),
    )


def vfit_equilibrium_magnetics_detailed(
    shot: int,
    indices: list[int] | np.ndarray | None = None,
    processing_config: VestMagneticsProcessingConfig | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
    allow_missing_channels: bool = False,
) -> VestEquilibriumMagneticsResult:
    """Process magnetics channels, keeping native flux-loop terminal voltages.

    Shot-era resolution is identical to `vfit_equilibrium_magnetics`: an
    explicit ``processing_config`` wins, otherwise the era policy comes from
    ``vest.yaml`` (issue #195).
    """
    require_supported_magnetics_geometry(int(shot))
    config = (
        processing_config
        if processing_config is not None
        else equilibrium_magnetics_processing_config(int(shot))
    )
    return vest_equilibrium_magnetics_detailed(
        int(shot),
        _load_equilibrium_magnetics_channels(),
        lambda source_shot, field: _safe_vest_load(source_shot, field, raw_source),
        indices=indices,
        config=config,
        allow_missing=allow_missing_channels,
    )


def vfit_equilibrium_magnetics(
    shot: int,
    indices: list[int] | np.ndarray | None = None,
    processing_config: VestMagneticsProcessingConfig | None = None,
    *,
    raw_source: raw_db.RawSource | None = None,
    allow_missing_channels: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Process magnetic probe and flux-loop data using VAFT process helpers.

    An explicitly supplied ``processing_config`` still wins, preserving the
    existing override path used for parameter scans; otherwise the shot-era
    policy is resolved from ``vest.yaml`` (issue #195).
    """
    result = vfit_equilibrium_magnetics_detailed(
        shot,
        indices,
        processing_config,
        raw_source=raw_source,
        allow_missing_channels=allow_missing_channels,
    )
    return result.time, result.flux_loops, result.probes


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


def _populate_fluctuation_mirnov_static(ods: object, shot: int = 0) -> None:
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
    for channel in _load_fluctuation_mirnov_channels(int(shot)):
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
    for channel in _load_fluctuation_mirnov_channels(int(shot)):
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


def _map_flux_loops(
    ods: object,
    context: _MagneticsContext,
    *,
    tstart: float | None = None,
    tend: float | None = None,
) -> None:
    """Map flux-loop terminal voltage and integrated flux (issue #209).

    The processing chain is kept explicit and quantity-specific::

        raw DAQ waveform
            -> channel calibration / sign convention
            -> magnetics.flux_loop[i].voltage   (native acquisition timebase,
                                                 cropped to [tstart, tend))
            -> time integration + linear baseline removal
            -> magnetics.flux_loop[i].flux      (canonical magnetics.time grid)

    So ``flux.data = -integral(voltage dt) - 2*pi*baseline``: integrating the
    stored voltage reproduces the stored flux up to the removed linear
    baseline term. Voltage is never reconstructed by differentiating flux, and
    voltage validity is independent of the presence of processed flux.
    """
    _set_magnetics_time(ods, context.target_time)
    for index, values in enumerate(context.flux_loops):
        _map_flux_loop_voltage(ods, context, index, tstart=tstart, tend=tend)
        if np.asarray(values).size < 2:
            continue
        data = _interpolate_signal(context.target_time, context.source_time, values) * 2 * math.pi
        set_path(ods, f"magnetics.flux_loop.{index}.flux.time", context.target_time)
        set_path(ods, f"magnetics.flux_loop.{index}.flux.data", data)


def _map_flux_loop_voltage(
    ods: object,
    context: _MagneticsContext,
    index: int,
    *,
    tstart: float | None = None,
    tend: float | None = None,
) -> None:
    """Store one calibrated pre-integration flux-loop voltage at native rate."""
    if index < len(context.flux_loop_voltage):
        native_time = np.asarray(context.flux_loop_voltage_time[index], dtype=float)
        native_data = np.asarray(context.flux_loop_voltage[index], dtype=float)
    else:
        native_time = np.array([], dtype=float)
        native_data = np.array([], dtype=float)

    if native_time.size < 2 or native_time.shape != native_data.shape:
        _set_voltage_signal(
            ods,
            f"magnetics.flux_loop.{index}",
            np.array([], dtype=float),
            np.array([], dtype=float),
            -2,
        )
        return

    time, data = _crop_native_window(native_time, native_data, tstart=tstart, tend=tend)
    validity = 0 if data.size else -2
    _set_voltage_signal(ods, f"magnetics.flux_loop.{index}", time, data, validity)


# Sensor slots in `magnetics.rogowski_coil`.  Fixed so a shot missing one
# sensor never shifts the other's index.
_ROGOWSKI_PLASMA_CURRENT = 0
_ROGOWSKI_DIAMAGNETIC_TF = 1

# The DD enumerates only plasma(1), plasma_eddy(2), eddy(3), halo(4) and
# compound(5) for `measured_quantity`, and requires private identifiers to use
# a negative index.  The hi-sensitivity TF-current sensor is none of the five,
# so it takes a private index rather than being misfiled as one of them.
_ROGOWSKI_TF_CURRENT_INDEX = -1


def _map_rogowski_coils(
    ods: object,
    shot: int,
    *,
    raw_source: raw_db.RawSource | None = None,
    tstart: float | None = None,
    tend: float | None = None,
) -> None:
    """Map the two physical VEST Rogowski sensors (issue #215).

    VEST processes two Rogowski coils but previously stored only what they were
    processed *into*.  Both are now represented as measurements in their own
    right, at the sensor level, with the derived quantities left untouched::

        plasma-current Rogowski
            -> magnetics.rogowski_coil[0].current   (calibration only)
            -> baseline removal, FL10 compensation, shot-era sign
            -> magnetics.ip[0]

        hi-sensitivity TF-current Rogowski
            -> magnetics.rogowski_coil[1].current   (integration + calibration)
            -> reference-waveform subtraction (delta_i_tf)
            -> magnetics.diamagnetic_flux[0]

    The stored current is the sensor-level quantity in both cases, never the
    processing intermediate and never reconstructed from the derived product.
    Currents keep their native acquisition timebase, cropped to the analysis
    window, exactly as the flux-loop and Mirnov voltages do (issue #209).
    """
    _map_one_rogowski_coil(
        ods,
        _ROGOWSKI_PLASMA_CURRENT,
        name="Plasma-current Rogowski coil",
        identifier="rogowski_coil:plasma_current",
        measured_name="plasma_eddy",
        measured_index=2,
        measured_description=(
            "Currents linked by the inner plasma-current Rogowski contour, "
            "before the FL10 compensation that subtracts a proxy for the "
            "induced current in the tungsten limiter around the CS wall"
        ),
        loader=lambda: vest_plasma_rogowski_current(shot, raw_source=raw_source),
        # Calibration only, no reconstruction: acquisition validity is all
        # that can be asserted here.
        validity=lambda _n: 0,
        tstart=tstart,
        tend=tend,
    )
    _map_one_rogowski_coil(
        ods,
        _ROGOWSKI_DIAMAGNETIC_TF,
        name="Diamagnetic hi-sensitivity TF-current Rogowski coil",
        identifier="rogowski_coil:diamagnetic_tf_current",
        measured_name="tf_coil_current",
        measured_index=_ROGOWSKI_TF_CURRENT_INDEX,
        measured_description=(
            "Toroidal-field coil current measured by the high-sensitivity "
            "Rogowski coil used as the input to the diamagnetic-flux "
            "diagnostic; a private identifier because the DD enumeration "
            "covers only plasma, plasma_eddy, eddy, halo and compound sensors"
        ),
        loader=lambda: vest_diamagnetic_rogowski_current(shot, raw_source=raw_source),
        validity=lambda _n: _diamagnetic_current_validity(shot, raw_source),
        tstart=tstart,
        tend=tend,
    )


def _diamagnetic_current_validity(
    shot: int, raw_source: raw_db.RawSource | None
) -> int:
    """Validity for the diamagnetic sensor current, honest about repair.

    ``0`` only when no sample sat at the acquisition rail. Otherwise ``1``
    ("valid from manual/automated processing but with a caveat" in DD terms is
    not available, so the non-zero code marks the waveform as carrying
    reconstructed samples) -- the count and extent live in
    :func:`diamagnetic_saturation_report`.
    """
    try:
        report = diamagnetic_saturation_report(shot, raw_source=raw_source)
    except (raw_db.RawSignalUnavailableError, SignalRepairError, KeyError):
        return -2
    return 0 if not report["n_saturated"] else 1


def _map_one_rogowski_coil(
    ods: object,
    index: int,
    *,
    name: str,
    identifier: str,
    measured_name: str,
    measured_index: int,
    measured_description: str,
    loader: Callable[[], tuple[np.ndarray, np.ndarray]],
    validity: Callable[[int], int],
    tstart: float | None,
    tend: float | None,
) -> None:
    """Write one Rogowski sensor, keeping its slot even when unavailable.

    Geometry (`position`, `turns_per_metre`, `area`) is deliberately left
    unset: VEST has no authoritative winding contour or turn density recorded,
    and inventing one to satisfy the schema would be worse than omitting it.
    """
    base = f"magnetics.rogowski_coil.{index}"
    set_path(ods, f"{base}.name", name)
    set_path(ods, f"{base}.identifier", identifier)
    set_path(ods, f"{base}.measured_quantity.name", measured_name)
    set_path(ods, f"{base}.measured_quantity.index", int(measured_index))
    set_path(ods, f"{base}.measured_quantity.description", measured_description)

    try:
        native_time, native_data = loader()
    except (raw_db.RawSignalUnavailableError, SignalRepairError):
        # Keep the slot so the other sensor's index never moves.
        #
        # SignalRepairError is caught here, unlike in the flux path where it
        # deliberately propagates: a caller asking for plasma current has not
        # asked for the diamagnetic channel, so an unrecoverable waveform on
        # one sensor must degrade that sensor rather than fail the whole
        # magnetics component (issue #285 interaction).
        _set_current_signal(ods, base, np.array([], dtype=float), np.array([], dtype=float), -2)
        return

    time, data = _crop_native_window(native_time, native_data, tstart=tstart, tend=tend)
    if not data.size:
        _set_current_signal(ods, base, time, data, -2)
        return
    # A repaired sample is an interpolation, not a measurement, so a waveform
    # containing any must not be published as `validity = 0` ("valid from
    # automated processing"). This is the structured home for the issue-#285
    # saturation outcome that `_map_diamagnetic_flux` points at.
    _set_current_signal(ods, base, time, data, validity(len(data)))


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
        _populate_fluctuation_mirnov_static(ods, shot)
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


_IP_METHOD_NAME = (
    "VEST inner plasma-current Rogowski: shot-era channel calibration, "
    "linear baseline removal, FL10 compensation for the induced current in "
    "the tungsten limiter around the CS wall (disabled from shot 47117, when "
    "the inboard was changed to carbon), shot-era sign convention. The "
    "pre-compensation sensor current is retained in "
    "magnetics.rogowski_coil[0].current."
)

_DIAMAGNETIC_METHOD_NAME = (
    "VEST TF-current-change diamagnetic flux: hi-sensitivity TF Rogowski "
    "current, reference-waveform subtraction over the plasma interval, then "
    "the TF-circuit L/R/C response. The sensor current is retained in "
    "magnetics.rogowski_coil[1].current."
)


def _map_ip(ods: object, target_time: np.ndarray, ip_time: np.ndarray, ip: np.ndarray) -> None:
    _set_magnetics_time(ods, target_time)
    set_path(ods, "magnetics.ip.0.data", _interpolate_signal(target_time, ip_time, ip))
    set_path(ods, "magnetics.ip.0.time", target_time)
    set_path(ods, "magnetics.ip.0.method_name", _IP_METHOD_NAME)


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


def _diamagnetic_method_name(report: Mapping[str, Any]) -> str:
    """One-line provenance string for `magnetics.diamagnetic_flux[0].method_name`."""
    base = (
        "Rogowski triple-integration of VEST raw field "
        f"{report['field']}"
    )
    if not report["n_saturated"]:
        return f"{base}; no acquisition-limit saturation detected"
    return (
        f"{base}; {report['n_saturated']}/{report['n_samples']} samples "
        f"reconstructed at the acquisition limit in {report['n_intervals']} intervals, "
        f"{report['n_saturated_in_window']} inside the plasma window"
    )


def _map_diamagnetic_flux(
    ods: object,
    shot: int,
    target_time: np.ndarray,
    ip_time: np.ndarray,
    ip: np.ndarray,
    raw_source: raw_db.RawSource | None = None,
) -> None:
    tstart2, tend2 = _plasma_window(ods, shot, target_time, ip_time, ip, raw_source)

    time_dia, dia_flux, report = vest_diamagnetic_flux_detailed(
        shot, tstart2, tend2, raw_source=raw_source
    )
    set_path(ods, "magnetics.diamagnetic_flux.0.data", _interpolate_signal(target_time, time_dia, dia_flux))
    set_path(ods, "magnetics.diamagnetic_flux.0.time", target_time)
    # The IMAS DD has no `validity` under `magnetics.diamagnetic_flux`, so the
    # saturation outcome is recorded in the one string field it does offer.
    # `magnetics.rogowski_coil[:].current.validity` is the structured home and
    # belongs to issue #215; it can call `diamagnetic_saturation_report`.
    set_path(ods, "magnetics.diamagnetic_flux.0.method_name", _diamagnetic_method_name(report))


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
            flux_loop_voltage_time=context.flux_loop_voltage_time,
            flux_loop_voltage=context.flux_loop_voltage,
        )
    _map_flux_loops(ods, context, tstart=tstart, tend=tend)
    _map_probes(ods, shot, context, raw_source, tstart=tstart, tend=tend)
    vfit_limiter_shunts_dynamic(ods, shot, raw_source=raw_source)
    _map_rogowski_coils(ods, shot, raw_source=raw_source, tstart=tstart, tend=tend)
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
    _map_flux_loops(ods, context, tstart=tstart, tend=tend)


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
    """Map the plasma-current Rogowski sensor and its derived plasma current."""
    del processing_config
    ip_time, ip = vfit_plasma_current(shot, raw_source=raw_source)
    target_time = _build_target_time(ip_time, tstart, tend, dt)
    _set_magnetics_properties(ods)
    # The sensor itself, not only what it was processed into (issue #215):
    # which entry point a caller reaches for must not decide whether the
    # physical coil appears.
    _map_rogowski_coils(ods, shot, raw_source=raw_source, tstart=tstart, tend=tend)
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
    """Map the diamagnetic Rogowski sensor and flux, without plasma current."""
    del processing_config
    ip_time, ip = vfit_plasma_current(shot, raw_source=raw_source)
    target_time = _build_target_time(ip_time, tstart, tend, dt)
    _set_magnetics_properties(ods)
    _set_magnetics_time(ods, target_time)
    # See ip_rogowski_coil_from_raw_database: the sensor travels with its
    # derived quantity regardless of entry point (issue #215).
    _map_rogowski_coils(ods, shot, raw_source=raw_source, tstart=tstart, tend=tend)
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
    "UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS",
    "UnsupportedMagneticsGeometryError",
    "equilibrium_magnetics_processing_config",
    "require_supported_magnetics_geometry",
    "b_field_pol_probe_from_raw_database",
    "diamagnetic_flux_rogowski_coil_from_raw_database",
    "flux_loop_from_raw_database",
    "ip_rogowski_coil_from_raw_database",
    "magnetics_from_raw_database",
    "diamagnetic_saturation_report",
    "vest_diamagnetic_flux",
    "vest_diamagnetic_flux_detailed",
    "vest_equilibrium_magnetics_channel_definitions",
    "fluctuation_mirnov_channel_definitions",
    "fluctuation_mirnov_gain_by_identifier",
    "fluctuation_mirnov_probe_indices",
    "select_fluctuation_mirnov_channels",
    "magnetics",
    "vfit_plasma_current",
    "vfit_equilibrium_magnetics",
    "vfit_equilibrium_magnetics_detailed",
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
