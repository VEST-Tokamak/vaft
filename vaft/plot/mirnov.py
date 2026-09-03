"""Mirnov fluctuation plotting helpers for OMAS ``magnetics`` IDS data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from vaft.ods_access import get_path, path_count, path_value
from vaft.machine_mapping.magnetics import (
    fluctuation_mirnov_channel_definitions,
    fluctuation_mirnov_gain_by_identifier,
)
from vaft.process.signal_processing import resample_to_time
from vaft.process.magnetics import (
    mirnov_preprocess_signal,
    mirnov_spectrogram as compute_mirnov_spectrogram,
    toroidal_phase_fit_at_time,
    toroidal_mode_analysis,
)

_DEFAULT_TOROIDAL_REFERENCE_PAIR = (
    "OutMirnov_530_Bz:phase_reference",
    "MagneticFieldProbe_C2-05_Bz:phase_reference",
)


def _known_gain_by_identifier() -> dict[str, float]:
    """Delegate to the machine_mapping gain registry.

    The lookup itself lives in ``vaft.machine_mapping.magnetics`` so
    ``vaft.process`` can reach it without importing the plot layer.
    """
    return fluctuation_mirnov_gain_by_identifier()


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _channel_count(ods: Any, probe_group: str) -> int:
    """How many channels the group holds, 0 when the ODS carries none.

    The 0 is what this has always returned for an absent group -- the previous
    ``except`` was unreachable, because reading a missing path yielded an empty
    placeholder whose ``len`` is 0 rather than an error. What is new is that
    asking no longer creates the group (issue #118).
    """
    return path_count(ods, f"magnetics.{probe_group}")


def _channel_label(ods: Any, probe_group: str, index: int) -> str:
    """The channel's own name, falling back to a positional label.

    The fallback used to be unreachable: reading a missing ``name`` returned an
    empty placeholder rather than raising, and ``str()`` of one is the IDS path
    -- so an unnamed channel was labelled ``magnetics.b_field_pol_probe.0.name``
    in the legend it appeared in (issue #118).
    """
    for field in ("name", "identifier"):
        value = path_value(ods, f"magnetics.{probe_group}.{index}.{field}")
        if value is not None:
            return str(value)
    return f"{probe_group} {index}"


def _normalise_channels(ods: Any, probe_group: str, channels: Sequence[int | str] | int | str | None) -> list[int]:
    n_channels = _channel_count(ods, probe_group)
    if channels is None:
        return list(range(n_channels))
    if isinstance(channels, (int, str)):
        candidates: Sequence[int | str] = [channels]
    else:
        candidates = channels

    selected: list[int] = []
    labels = {
        _channel_label(ods, probe_group, index).lower(): index
        for index in range(n_channels)
    }
    for channel in candidates:
        if isinstance(channel, str):
            key = channel.lower()
            if key not in labels:
                raise KeyError(f"No {probe_group} channel named {channel!r}.")
            selected.append(labels[key])
            continue
        index = int(channel)
        if index < 0 or index >= n_channels:
            raise IndexError(f"{probe_group} channel index {index} is outside 0..{n_channels - 1}.")
        selected.append(index)
    return selected


def _voltage_time_signal(ods: Any, probe_group: str, channel: int) -> tuple[np.ndarray, np.ndarray]:
    base = f"magnetics.{probe_group}.{channel}.voltage"
    time = _as_array(get_path(ods, f"{base}.time"))
    data = _as_array(get_path(ods, f"{base}.data"))
    if data.ndim > 1:
        data = np.squeeze(data)
    if time.size != data.size:
        raise ValueError(f"{probe_group} channel {channel} has inconsistent voltage time/data lengths.")
    return time, data


def _time_mask(time: np.ndarray, time_range: tuple[float, float] | None) -> np.ndarray:
    if time_range is None:
        return np.ones(time.size, dtype=bool)
    start, stop = time_range
    return (time >= float(start)) & (time <= float(stop))


def _sample_rate_from_time(time: np.ndarray) -> float:
    if time.size < 2:
        return 250_000.0
    dt = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        return 250_000.0
    return 1.0 / dt


def _gain_for_channel(ods: Any, gains: Any, channel: int, probe_group: str) -> float:
    if gains is None:
        stored = path_value(ods, f"magnetics.{probe_group}.{channel}.calibration_factor")
        if stored is not None:
            try:
                return float(stored)
            except (TypeError, ValueError):
                pass
        if probe_group == "b_field_pol_probe":
            label = _channel_label(ods, probe_group, channel)
            if label in _known_gain_by_identifier():
                return _known_gain_by_identifier()[label]
        return 1.0
    if isinstance(gains, dict):
        return float(gains.get(channel, 1.0))
    return float(gains[channel])


def _maybe_preprocess(
    time: np.ndarray,
    data: np.ndarray,
    *,
    preprocess: bool,
    gain: float,
    sample_rate: float | None,
) -> np.ndarray:
    if not preprocess:
        return data
    return mirnov_preprocess_signal(
        data,
        sample_rate=float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time),
        amplifier_gain=gain,
    )


def mirnov_signal(
    ods: Any,
    channels: Sequence[int | str] | int | str | None = None,
    *,
    probe_group: str = "b_field_pol_probe",
    time_range: tuple[float, float] | None = None,
    preprocess: bool = False,
    gains: Any = None,
    ax: Any | None = None,
    show: bool = True,
):
    """Plot raw or preprocessed Mirnov voltage traces from ``magnetics``."""
    selected = _normalise_channels(ods, probe_group, channels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
    else:
        fig = ax.figure

    for channel in selected:
        time, data = _voltage_time_signal(ods, probe_group, channel)
        mask = _time_mask(time, time_range)
        time = time[mask]
        data = data[mask]
        data = _maybe_preprocess(
            time,
            data,
            preprocess=preprocess,
            gain=_gain_for_channel(ods, gains, channel, probe_group),
            sample_rate=None,
        )
        ax.plot(time * 1e3, data, lw=1.0, label=_channel_label(ods, probe_group, channel))

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Mirnov signal")
    if selected:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def mirnov_spectrogram(
    ods: Any,
    channel: int | str = 0,
    *,
    probe_group: str = "b_field_pol_probe",
    time_range: tuple[float, float] | None = None,
    preprocess: bool = True,
    gain: float | None = None,
    sample_rate: float | None = None,
    window_size: int = 500,
    time_resolution: int = 1,
    max_frequency: float | None = None,
    cmap: str = "hot_r",
    ax: Any | None = None,
    show: bool = True,
    return_result: bool = False,
):
    """Plot a MATLAB-compatible Mirnov spectrogram from ODS raw voltage data."""
    index = _normalise_channels(ods, probe_group, channel)[0]
    time, data = _voltage_time_signal(ods, probe_group, index)
    mask = _time_mask(time, time_range)
    time = time[mask]
    data = data[mask]
    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time)
    data = _maybe_preprocess(
        time,
        data,
        preprocess=preprocess,
        gain=float(gain) if gain is not None else _gain_for_channel(ods, None, index, probe_group),
        sample_rate=fs,
    )
    result = compute_mirnov_spectrogram(
        time,
        data,
        sample_rate=fs,
        window_size=window_size,
        time_resolution=time_resolution,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(result.time * 1e3, result.frequency / 1e3, result.magnitude, shading="auto", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label="Magnitude")
    if max_frequency is not None:
        ax.set_ylim(0.0, float(max_frequency) / 1e3)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_title(_channel_label(ods, probe_group, index))
    fig.tight_layout()
    if show:
        plt.show()
    if return_result:
        return fig, ax, result
    return fig, ax


def _common_timebase(
    time_a: np.ndarray,
    data_a: np.ndarray,
    time_b: np.ndarray,
    data_b: np.ndarray,
    time_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if time_a.size == 0 or time_b.size == 0:
        raise ValueError("Both Mirnov channels must contain voltage data.")
    start = max(float(time_a[0]), float(time_b[0]))
    stop = min(float(time_a[-1]), float(time_b[-1]))
    if time_range is not None:
        start = max(start, float(time_range[0]))
        stop = min(stop, float(time_range[1]))
    mask = (time_a >= start) & (time_a <= stop)
    common_time = time_a[mask]
    if common_time.size == 0:
        raise ValueError("Mirnov channels do not overlap in the requested time range.")
    return common_time, data_a[mask], resample_to_time(time_b, data_b, common_time)


def _common_timebase_many(
    times: Sequence[np.ndarray],
    data: Sequence[np.ndarray],
    time_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not times:
        raise ValueError("At least one signal is required.")
    for time in times:
        if time.size == 0:
            raise ValueError("All Mirnov channels must contain voltage data.")
    start = max(float(time[0]) for time in times)
    stop = min(float(time[-1]) for time in times)
    if time_range is not None:
        start = max(start, float(time_range[0]))
        stop = min(stop, float(time_range[1]))
    mask = (times[0] >= start) & (times[0] <= stop)
    common_time = times[0][mask]
    if common_time.size == 0:
        raise ValueError("Mirnov channels do not overlap in the requested time range.")
    stacked = [data[0][mask]]
    for time, values in zip(times[1:], data[1:]):
        stacked.append(resample_to_time(time, values, common_time))
    return common_time, np.vstack(stacked)


def _channel_toroidal_angle(ods: Any, probe_group: str, channel: int) -> float:
    for suffix in ("toroidal_angle", "position.phi"):
        stored = path_value(ods, f"magnetics.{probe_group}.{channel}.{suffix}")
        if stored is None:
            continue
        try:
            return float(stored)
        except (TypeError, ValueError):
            continue
    raise KeyError(f"{probe_group} channel {channel} does not define toroidal_angle or position.phi.")


def _with_phase_jumps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_out = [float(x[0])]
    y_out = [float(y[0])]
    for index in range(1, x.size):
        if abs(float(y[index] - y[index - 1])) > 180.0:
            x_out.append(np.nan)
            y_out.append(np.nan)
        x_out.append(float(x[index]))
        y_out.append(float(y[index]))
    return np.asarray(x_out), np.asarray(y_out)


def toroidal_mode_spectrum(
    ods: Any,
    channel_pair: tuple[int | str, int | str] = _DEFAULT_TOROIDAL_REFERENCE_PAIR,
    *,
    probe_group: str = "b_field_pol_probe",
    time_range: tuple[float, float] | None = None,
    preprocess: bool = True,
    gains: Any = None,
    phase_geometry: float = np.pi / 6,
    peak_threshold: float = 0.1,
    sample_rate: float | None = None,
    axes: Sequence[Any] | None = None,
    show: bool = True,
    return_result: bool = False,
):
    """Plot cross power, toroidal mode number, and coherence for a Mirnov pair."""
    first, second = _normalise_channels(ods, probe_group, channel_pair)
    time_a, data_a = _voltage_time_signal(ods, probe_group, first)
    time_b, data_b = _voltage_time_signal(ods, probe_group, second)
    time, data_a, data_b = _common_timebase(time_a, data_a, time_b, data_b, time_range)
    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(time)
    data_a = _maybe_preprocess(
        time,
        data_a,
        preprocess=preprocess,
        gain=_gain_for_channel(ods, gains, first, probe_group),
        sample_rate=fs,
    )
    data_b = _maybe_preprocess(
        time,
        data_b,
        preprocess=preprocess,
        gain=_gain_for_channel(ods, gains, second, probe_group),
        sample_rate=fs,
    )
    result = toroidal_mode_analysis(
        data_a,
        data_b,
        sample_rate=fs,
        phase_geometry=phase_geometry,
        peak_threshold=peak_threshold,
    )

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    else:
        axes = list(axes)
        fig = axes[0].figure

    frequency_khz = result.spectrum_frequency / 1e3
    power = np.abs(result.cross_power)
    normalised_power = power / np.max(power) if power.size and np.max(power) > 0 else power
    axes[0].plot(frequency_khz, normalised_power, color="k", lw=1.0)
    if result.peak_indices.size:
        axes[0].plot(frequency_khz[result.peak_indices], normalised_power[result.peak_indices], "or")
    axes[0].set_ylabel("|Pxy|")

    axes[1].plot(frequency_khz, result.n_rounded, color="k", lw=1.0)
    axes[1].plot(frequency_khz, result.n_raw, color="0.55", lw=0.8)
    if result.peak_indices.size:
        axes[1].plot(frequency_khz[result.peak_indices], result.n_rounded[result.peak_indices], "or")
    axes[1].set_ylabel("n")

    axes[2].plot(frequency_khz, result.coherence, color="k", lw=1.0)
    axes[2].set_ylabel("Coherence")
    axes[2].set_xlabel("Frequency [kHz]")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    if return_result:
        return fig, axes, result
    return fig, axes


def toroidal_phase_mode_fit(
    ods: Any,
    center_time: float,
    *,
    channels: Sequence[int | str] = (
        "OutMirnov_130_Bz:phase_reference",
        "OutMirnov_530_Bz:phase_reference",
        "OutMirnov_730_Bz:phase_reference",
        "MagneticFieldProbe_C2-05_Bz:phase_reference",
    ),
    probe_group: str = "b_field_pol_probe",
    time_range: tuple[float, float] | None = None,
    frequencies: Sequence[float] | None = None,
    num_modes: int = 2,
    candidate_n: Sequence[int] = tuple(range(0, 7)),
    window_size: int = 500,
    preprocess: bool = True,
    gains: Any = None,
    sample_rate: float | None = None,
    peak_threshold: float = 0.1,
    ax: Any | None = None,
    show: bool = True,
    save_path: str | None = None,
    return_result: bool = False,
):
    """Plot toroidal phase variation and best-fit wrapped ``n`` mode lines."""
    selected = _normalise_channels(ods, probe_group, channels)
    times: list[np.ndarray] = []
    data: list[np.ndarray] = []
    angles = np.asarray([_channel_toroidal_angle(ods, probe_group, channel) for channel in selected], dtype=float)
    for channel in selected:
        time, values = _voltage_time_signal(ods, probe_group, channel)
        times.append(time)
        data.append(values)
    common_time, stacked = _common_timebase_many(times, data, time_range)
    fs = float(sample_rate) if sample_rate is not None else _sample_rate_from_time(common_time)
    if preprocess:
        stacked = np.vstack(
            [
                _maybe_preprocess(
                    common_time,
                    stacked[index],
                    preprocess=True,
                    gain=_gain_for_channel(ods, gains, channel, probe_group),
                    sample_rate=fs,
                )
                for index, channel in enumerate(selected)
            ]
        )

    result = toroidal_phase_fit_at_time(
        common_time,
        stacked,
        angles,
        center_time=float(center_time),
        sample_rate=fs,
        window_size=window_size,
        frequencies=frequencies,
        num_modes=num_modes,
        candidate_n=candidate_n,
        peak_threshold=peak_threshold,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
    else:
        fig = ax.figure

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    angle_deg = np.rad2deg(result.toroidal_angle) % 360.0
    order = np.argsort(angle_deg)
    line_angle_deg = np.linspace(0.0, 360.0, 721)
    line_angle_rad = np.deg2rad(line_angle_deg)

    for mode_index, mode in enumerate(result.modes):
        color = colors[mode_index % len(colors)]
        phase_deg = np.rad2deg(mode.phase)
        label = f"{mode.frequency / 1e3:.1f} kHz, n={mode.n}"
        ax.scatter(angle_deg, phase_deg, s=48, color=color, label=label, zorder=3)

        fitted_line = np.rad2deg((mode.intercept - mode.n * line_angle_rad + np.pi) % (2 * np.pi) - np.pi)
        x_line, y_line = _with_phase_jumps(line_angle_deg, fitted_line)
        ax.plot(x_line, y_line, "--", color=color, lw=1.5)

        text_x = angle_deg[order[min(mode_index, order.size - 1)]]
        text_y = phase_deg[order[min(mode_index, order.size - 1)]]
        ax.text(text_x + 8.0, text_y + 8.0, f"n = {mode.n}", color=color, weight="bold")

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.arange(0.0, 361.0, 60.0))
    ax.set_yticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_xlabel("Toroidal Angle [degree]")
    ax.set_ylabel("Phase Variation [degree]")
    ax.set_title(f"Toroidal phase fit @ {result.time * 1e3:.1f} ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    if return_result:
        return fig, ax, result
    return fig, ax
