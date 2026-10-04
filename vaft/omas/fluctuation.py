"""ODS-level helpers for fluctuation and transient analysis (issue #1005).

Two questions a fluctuation study asks of a discharge before any spectrum is
computed, answered from what the ODS actually stores:

* :func:`fluctuation_bandwidths` -- what sample rate, and so what Nyquist
  frequency, each fluctuation-capable diagnostic was recorded at.  Read off the
  stored time bases, never off a table of nominal rates.
* :func:`vertical_position_history` -- where the magnetic axis sits vertically,
  and how fast it moves, over the stored equilibrium slices.

Every read goes through :mod:`vaft.ods_access`, so nothing is created on the
ODS and a native IMAS entry is read the same way.  Both return numbers only:
a Nyquist frequency is an upper bound on what a record can show, not the band
it can resolve, and a vertical drift is a position history, not a named event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from vaft.ods_access import path_count, path_value

__all__ = [
    "DiagnosticSelection",
    "SelectedDiagnostics",
    "select_fluctuation_records",
    "FLUCTUATION_DIAGNOSTICS",
    "VerticalPositionHistory",
    "fluctuation_bandwidths",
    "vertical_position_history",
]


@dataclass(frozen=True)
class DiagnosticSelection:
    """One explicit representative scalar signal for one diagnostic family."""

    diagnostic: str
    channel: int | str | None = None
    emission: str | None = None
    region: tuple[int, int, int, int] | None = None
    detector: int = 0
    component_time: Any = None
    component_data: Any = None
    component_label: str | None = None
    usable_bandwidth_hz: float | None = None
    units: str | None = None
    background_frames: int | None = None


@dataclass(frozen=True)
class SelectedDiagnostics:
    """Selected records and matching source descriptions in diagnostic order."""

    records: tuple[Any, ...]
    sources: tuple[str, ...]


_SCALAR_PATHS = {
    "mirnov": ("magnetics.b_field_pol_probe", ("field",), "T"),
    "soft_x_rays": ("soft_x_rays.channel", ("brightness", "power"), "a.u."),
    "interferometer": ("interferometer.channel", ("n_e_line",), "m^-2"),
}


def _selected_channel(ods: Any, container: str, channel: int | str | None) -> int:
    """Require an explicit channel index or a unique stored channel name."""
    if isinstance(channel, (int, np.integer)) and not isinstance(channel, (bool, np.bool_)):
        index = int(channel)
        if 0 <= index < path_count(ods, container):
            return index
    elif isinstance(channel, str) and channel:
        matches = [index for index in range(path_count(ods, container))
                   if path_value(ods, f"{container}.{index}.name") == channel]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"{container}: channel name {channel!r} is ambiguous")
    raise ValueError(f"{container}: select an existing channel by index or unique name")


def _stored_scalar(ods: Any, selection: DiagnosticSelection):
    diagnostic = selection.diagnostic
    container, leaves, default_units = _SCALAR_PATHS[diagnostic]
    index = _selected_channel(ods, container, selection.channel)
    for leaf in leaves:
        base = f"{container}.{index}.{leaf}"
        data = path_value(ods, f"{base}.data")
        if data is None:
            continue
        values = np.asarray(data, dtype=float).reshape(-1)
        time = path_value(ods, f"{base}.time")
        if time is None:
            time = path_value(ods, f"{container.rsplit('.', 1)[0]}.time")
        if time is None or np.size(time) != values.size:
            raise ValueError(f"{base}: matching stored time axis is required")
        return np.asarray(time, dtype=float).reshape(-1), values, base, default_units
    if diagnostic == "mirnov":
        raise ValueError(
            f"{container}.{index}: integrated/calibrated field is required; "
            "process pickup voltage with vaft.process.magnetics.b_field_pol_probe_field first"
        )
    raise ValueError(f"{container}.{index}: no selected scalar signal")


def _camera_scalar(ods: Any, selection: DiagnosticSelection):
    if selection.component_time is not None or selection.component_data is not None:
        if selection.component_time is None or selection.component_data is None:
            raise ValueError("camera temporal component requires both time and data")
        if not isinstance(selection.component_label, str) or not selection.component_label:
            raise ValueError("camera temporal component requires a provenance label")
        return (np.asarray(selection.component_time, dtype=float).reshape(-1),
                np.asarray(selection.component_data, dtype=float).reshape(-1),
                f"camera_visible:temporal component:{selection.component_label}",
                selection.units or "a.u.")
    if selection.region is None:
        raise ValueError("camera requires an explicit ROI or temporal component")
    channel = _selected_channel(ods, "camera_visible.channel", selection.channel)
    detector = int(selection.detector)
    prefix = f"camera_visible.channel.{channel}.detector.{detector}.frame"
    count = path_count(ods, prefix)
    if count < 2:
        raise ValueError(f"{prefix}: at least two frames are required")
    region = selection.region
    if len(region) != 4 or any(not isinstance(v, (int, np.integer)) for v in region):
        raise ValueError("camera ROI must be four integer pixel bounds")
    first_frame = path_value(ods, f"{prefix}.0.image_raw")
    if first_frame is None or np.asarray(first_frame).ndim != 2:
        raise ValueError(f"{prefix}.0: a two-dimensional image_raw is required")
    height, width = np.asarray(first_frame).shape
    r0, r1, c0, c1 = region
    if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
        raise ValueError(f"camera ROI {region!r} exceeds frame shape {(height, width)}")
    from vaft.process.camera_fluctuation import (
        BACKGROUND_FRAMES_50KFPS, subtract_temporal_background,
        summed_region_signal,
    )

    time = np.empty(count)
    summed = np.empty(count)
    for index in range(count):
        time[index] = float(path_value(ods, f"{prefix}.{index}.time"))
        frame = path_value(ods, f"{prefix}.{index}.image_raw")
        if frame is None:
            raise ValueError(f"{prefix}.{index}: image_raw is missing")
        if np.shape(frame) != (height, width):
            raise ValueError(f"{prefix}.{index}: frame shape changed")
        summed[index] = summed_region_signal(
            np.asarray(frame)[None, ...], region=region
        )[0]
    background = (BACKGROUND_FRAMES_50KFPS if selection.background_frames is None
                  else int(selection.background_frames))
    signal = subtract_temporal_background(
        summed[:, None], window_frames=background
    )[:, 0]
    return time, signal, f"{prefix}:ROI{selection.region}:background{background}", "count"


def _uv_scalar(ods: Any, selection: DiagnosticSelection):
    if not isinstance(selection.emission, str) or not selection.emission:
        raise ValueError("spectrometer_uv requires an explicit emission identity")
    from vaft.spectroscopy import matches, parse_emission_term, parse_line_label

    container = "spectrometer_uv.channel"
    if selection.channel is None:
        channels = range(path_count(ods, container))
    else:
        channels = (_selected_channel(ods, container, selection.channel),)
    wanted = parse_emission_term(selection.emission)
    candidates = []
    for channel in channels:
        for line in range(path_count(ods, f"{container}.{channel}.processed_line")):
            base = f"{container}.{channel}.processed_line.{line}"
            label = path_value(ods, f"{base}.label")
            identity = parse_line_label(label)
            if label == selection.emission or (
                wanted is not None and identity is not None and matches(wanted, identity)
            ):
                candidates.append((base, label))
    if len(candidates) != 1:
        raise ValueError(
            f"emission {selection.emission!r} matched {len(candidates)} lines; "
            "specify one channel and an unambiguous emission label"
        )
    base, label = candidates[0]
    data = path_value(ods, f"{base}.intensity.data")
    time = path_value(ods, f"{base}.intensity.time")
    if time is None:
        time = path_value(ods, "spectrometer_uv.time")
    if data is None or time is None or np.size(data) != np.size(time):
        raise ValueError(f"{base}: intensity and matching time are required")
    return (np.asarray(time, dtype=float).reshape(-1),
            np.asarray(data, dtype=float).reshape(-1), f"{base}:{label}", "a.u.")


def select_fluctuation_records(
    ods: Any, selections: Sequence[DiagnosticSelection],
) -> SelectedDiagnostics:
    """Read one explicitly chosen scalar representative per diagnostic.

    Stored Mirnov field is used only after calibration/integration. SXR uses a
    chosen chord, interferometry its line-integrated density, camera a stated
    ROI or externally prepared temporal component, and UV one emission label.
    Multiple channels never enter one diagnostic's matrix slot implicitly.
    The returned ``sources`` preserve channel, ROI, or line selection; the
    records retain units and declared usable bandwidth for spectral analysis.
    Missing data raise rather than silently filling or selecting a neighbour.
    """
    from vaft.process.fluctuation import FluctuationRecord

    chosen = tuple(selections)
    if not chosen or any(not isinstance(item, DiagnosticSelection) for item in chosen):
        raise ValueError("selections must contain DiagnosticSelection objects")
    names = tuple(item.diagnostic for item in chosen)
    if len(set(names)) != len(names):
        raise ValueError("select exactly one representative per diagnostic")
    records = []
    sources = []
    for item in chosen:
        if item.diagnostic in _SCALAR_PATHS:
            time, data, source, units = _stored_scalar(ods, item)
        elif item.diagnostic == "camera_visible":
            time, data, source, units = _camera_scalar(ods, item)
        elif item.diagnostic == "spectrometer_uv":
            time, data, source, units = _uv_scalar(ods, item)
        else:
            raise ValueError(f"unsupported diagnostic {item.diagnostic!r}")
        if time.size != data.size or time.size < 2:
            raise ValueError(f"{source}: at least two paired samples are required")
        records.append(FluctuationRecord(
            item.diagnostic, time, data, item.units or units,
            item.usable_bandwidth_hz, source,
        ))
        sources.append(source)
    return SelectedDiagnostics(tuple(records), tuple(sources))

#: The fluctuation-capable diagnostics and where their time bases live:
#: ``label -> (container, data leaves, time leaves)``.  ``{i}`` indexes the
#: container; the first data leaf present marks a channel as recorded, and
#: the first time leaf present with a matching length is its time base.
FLUCTUATION_DIAGNOSTICS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "Mirnov / magnetic probes": (
        "magnetics.b_field_pol_probe",
        ("magnetics.b_field_pol_probe.{i}.voltage.data", "magnetics.b_field_pol_probe.{i}.field.data"),
        ("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.b_field_pol_probe.{i}.field.time",
         "magnetics.time"),
    ),
    "Soft X-ray": (
        "soft_x_rays.channel",
        ("soft_x_rays.channel.{i}.brightness.data", "soft_x_rays.channel.{i}.power.data"),
        ("soft_x_rays.channel.{i}.brightness.time", "soft_x_rays.channel.{i}.power.time", "soft_x_rays.time"),
    ),
    "Interferometer": (
        "interferometer.channel",
        ("interferometer.channel.{i}.n_e_line.data",),
        ("interferometer.channel.{i}.n_e_line.time", "interferometer.time"),
    ),
    "Filterscope / UV line": (
        "spectrometer_uv.channel",
        ("spectrometer_uv.channel.{i}.processed_line.0.intensity.data",),
        ("spectrometer_uv.channel.{i}.processed_line.0.intensity.time", "spectrometer_uv.time"),
    ),
    "Langmuir probe": (
        "langmuir_probes.embedded",
        ("langmuir_probes.embedded.{i}.n_e.data", "langmuir_probes.embedded.{i}.t_e.data"),
        ("langmuir_probes.embedded.{i}.time", "langmuir_probes.time"),
    ),
}

_CAMERA_FRAMES = "camera_visible.channel.{i}.detector.{j}.frame"

#: Relative difference below which two sample rates are one group.
_RATE_TOLERANCE = 1e-3


def _rate(time: Any) -> float | None:
    """1 / median sample spacing of a stored time base, or ``None`` if it has none."""
    if time is None:
        return None
    values = np.asarray(time, dtype=float).reshape(-1)
    steps = np.diff(values[np.isfinite(values)])
    steps = steps[steps > 0]
    if steps.size == 0:
        return None
    return float(1.0 / np.median(steps))


def _channel_rate(source: Any, data_leaves: tuple[str, ...], time_leaves: tuple[str, ...], index: int) -> float | None:
    data = None
    for template in data_leaves:
        data = path_value(source, template.format(i=index))
        if data is not None and np.size(data) > 1:
            break
        data = None
    if data is None:
        return None
    for template in time_leaves:
        time = path_value(source, template.format(i=index))
        if time is not None and np.size(time) == np.size(data):
            return _rate(time)
    return None


def _camera_rates(source: Any) -> list[float]:
    rates = []
    for i in range(path_count(source, "camera_visible.channel")):
        for j in range(path_count(source, f"camera_visible.channel.{i}.detector")):
            frames = _CAMERA_FRAMES.format(i=i, j=j)
            times = [path_value(source, f"{frames}.{k}.time") for k in range(path_count(source, frames))]
            rate = _rate([t for t in times if t is not None])
            if rate is not None:
                rates.append(rate)
    return rates


def _grouped(label: str, rates: list[float]) -> dict[str, tuple[float, float]]:
    groups: list[list[float]] = []
    for rate in sorted(rates):
        if groups and abs(rate - groups[-1][0]) <= _RATE_TOLERANCE * groups[-1][0]:
            groups[-1].append(rate)
        else:
            groups.append([rate])
    out: dict[str, tuple[float, float]] = {}
    for group in groups:
        fs = float(np.median(group))
        key = label if len(groups) == 1 else f"{label} ({fs / 1e3:.4g} kHz)"
        out[key] = (fs, fs / 2.0)
    return out


def fluctuation_bandwidths(ods: Any) -> dict[str, tuple[float, float]]:
    """``{diagnostic: (sample_rate [Hz], nyquist [Hz])}`` read off the stored time bases.

    Only channels that carry data are counted, and each channel's rate is
    ``1 / median(diff(time))`` of its own time base, so a record decimated
    when it was written reports the rate it was written at, not the
    digitizer's.  A diagnostic whose channels were stored at more than one
    rate gets one entry per rate, labelled with the rate.  Diagnostics the
    input does not carry are absent from the mapping.

    A Nyquist frequency bounds what a record can represent; the usable band
    is lower, set by the sensor response, the analogue and anti-alias
    filters, the exposure and the signal-to-noise ratio.
    """
    out: dict[str, tuple[float, float]] = {}
    for label, (container, data_leaves, time_leaves) in FLUCTUATION_DIAGNOSTICS.items():
        rates = [
            rate
            for index in range(path_count(ods, container))
            if (rate := _channel_rate(ods, data_leaves, time_leaves, index)) is not None
        ]
        if rates:
            out.update(_grouped(label, rates))
    camera = _camera_rates(ods)
    if camera:
        out.update(_grouped("FAST camera", camera))
    return out


@dataclass(frozen=True)
class VerticalPositionHistory:
    """Magnetic-axis height and its rate over the equilibrium slices, time-ordered."""

    time: np.ndarray | None
    z_axis: np.ndarray | None
    dz_dt: np.ndarray | None
    r_axis: np.ndarray | None
    valid: np.ndarray | None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.reason is None


def vertical_position_history(ods: Any) -> VerticalPositionHistory:
    """Magnetic-axis ``Z(t)`` [m] and ``dZ/dt`` [m/s] from the stored equilibrium slices.

    Each slice is placed at its own ``time_slice[i].time`` (``equilibrium.time[i]``
    only when a slice carries none), and the slices are sorted by that time, so a
    slice is never paired with a neighbour's time by position.  A slice that
    reconstructs no plasma -- zero or missing ``global_quantities.ip``, or a
    non-finite axis -- is kept on the time axis with ``NaN`` position and rate and
    ``valid`` false; it is not a position.  ``dz_dt`` is the centred difference
    over the valid slices at their own (non-uniform) times.

    No equilibrium, or no valid slice, returns ``None`` arrays and a ``reason``;
    a single valid slice returns its position with ``dz_dt`` ``NaN`` and a
    ``reason``, because one slice has no rate.
    """
    count = path_count(ods, "equilibrium.time_slice")
    if count == 0:
        return VerticalPositionHistory(None, None, None, None, None, reason="no equilibrium time slices")

    stored_times = path_value(ods, "equilibrium.time")
    stored_times = None if stored_times is None else np.asarray(stored_times, dtype=float).reshape(-1)
    times, zs, rs, valid = [], [], [], []
    for index in range(count):
        base = f"equilibrium.time_slice.{index}"
        t = path_value(ods, f"{base}.time")
        if t is None and stored_times is not None and index < stored_times.size:
            t = stored_times[index]
        if t is None:
            return VerticalPositionHistory(
                None, None, None, None, None,
                reason=f"equilibrium slice {index} has no time",
            )
        z = path_value(ods, f"{base}.global_quantities.magnetic_axis.z")
        r = path_value(ods, f"{base}.global_quantities.magnetic_axis.r")
        ip = path_value(ods, f"{base}.global_quantities.ip")
        z = np.nan if z is None else float(z)
        r = np.nan if r is None else float(r)
        ok = ip is not None and np.isfinite(float(ip)) and float(ip) != 0.0 and np.isfinite(z) and np.isfinite(r)
        times.append(float(t))
        zs.append(z if ok else np.nan)
        rs.append(r if ok else np.nan)
        valid.append(ok)

    order = np.argsort(times, kind="stable")
    time = np.asarray(times)[order]
    z_axis = np.asarray(zs)[order]
    r_axis = np.asarray(rs)[order]
    mask = np.asarray(valid, dtype=bool)[order]
    dz_dt = np.full(time.size, np.nan)
    kept = np.nonzero(mask)[0]
    if kept.size == 0:
        return VerticalPositionHistory(
            None, None, None, None, None,
            reason="no equilibrium slice carries a plasma (every ip is zero or missing)",
        )
    if kept.size == 1:
        return VerticalPositionHistory(
            time, z_axis, dz_dt, r_axis, mask,
            reason="only one valid equilibrium slice; a rate needs two",
        )
    dz_dt[kept] = np.gradient(z_axis[kept], time[kept])
    return VerticalPositionHistory(time, z_axis, dz_dt, r_axis, mask)
