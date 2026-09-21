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
from typing import Any

import numpy as np

from vaft.ods_access import path_count, path_value

__all__ = [
    "FLUCTUATION_DIAGNOSTICS",
    "VerticalPositionHistory",
    "fluctuation_bandwidths",
    "vertical_position_history",
]

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
