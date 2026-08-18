"""Soft X-ray plotting helpers for OMAS ``soft_x_rays`` IDS data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

_ARRAY_COLORS = {
    "horizontal": "tab:blue",
    "vertical": "tab:orange",
    "lowermid": "tab:green",
    "bottom": "tab:red",
    "digitizer": "tab:gray",
    None: "tab:gray",
}


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _soft_x_ray_channel_count(ods: Any) -> int:
    try:
        return len(ods["soft_x_rays.channel"])
    except Exception as exc:
        raise KeyError("ODS does not contain soft_x_rays.channel data.") from exc


def _channel_name(ods: Any, index: int) -> str:
    try:
        return str(ods[f"soft_x_rays.channel.{index}.name"])
    except Exception:
        return f"SXR Ch {index + 1}"


def _channel_identifier(ods: Any, index: int) -> str:
    try:
        return str(ods[f"soft_x_rays.channel.{index}.identifier"])
    except Exception:
        return _channel_name(ods, index)


def _channel_array_and_number(ods: Any, index: int) -> tuple[str | None, int | None]:
    identifier = _channel_identifier(ods, index)
    parts = identifier.split(":")
    if len(parts) >= 4 and parts[-2].lower() in {"be", "al", "none"}:
        try:
            return parts[-3], int(parts[-1])
        except ValueError:
            return parts[-3], None
    if len(parts) >= 3:
        try:
            return parts[-2], int(parts[-1])
        except ValueError:
            return parts[-2], None

    name = _channel_name(ods, index).lower()
    if "lower-mid" in name or "lowermid" in name:
        return "lowermid", None
    if "bottom" in name:
        return "bottom", None
    if "horizontal" in name:
        return "horizontal", None
    if "vertical" in name:
        return "vertical", None
    return None, None


def _normalise_channels(
    ods: Any,
    channels: Sequence[int] | int | None = None,
    arrays: Sequence[str] | str | None = None,
) -> list[int]:
    n_channels = _soft_x_ray_channel_count(ods)
    if channels is None:
        selected = list(range(n_channels))
    elif isinstance(channels, int):
        selected = [channels]
    else:
        selected = [int(channel) for channel in channels]

    for channel in selected:
        if channel < 0 or channel >= n_channels:
            raise IndexError(f"SXR channel index {channel} is outside 0..{n_channels - 1}.")

    if arrays is None:
        return selected
    if isinstance(arrays, str):
        wanted = {arrays.lower()}
    else:
        wanted = {str(array).lower() for array in arrays}
    return [idx for idx in selected if (_channel_array_and_number(ods, idx)[0] or "").lower() in wanted]


def _channel_time_and_signal(ods: Any, channel: int) -> tuple[np.ndarray, np.ndarray]:
    base = f"soft_x_rays.channel.{channel}.brightness"
    try:
        time = _as_array(ods[f"{base}.time"])
    except Exception:
        time = _as_array(ods["soft_x_rays.time"])
    data = _as_array(ods[f"{base}.data"])
    if data.ndim > 1:
        # IMAS defines brightness.data as (energy_band, time).  Preserve
        # compatibility with older VEST ODS files that used (time, 1).
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[1] == 1:
            data = data[:, 0]
        else:
            raise ValueError(
                f"Channel {channel} has multiple energy bands; select a band before plotting."
            )
    data = np.squeeze(data)
    if time.size != data.size:
        raise ValueError(f"Channel {channel} has inconsistent brightness time/data lengths.")
    return time, data


def _time_mask(time: np.ndarray, time_range: tuple[float, float] | None) -> np.ndarray:
    mask = np.ones(time.size, dtype=bool)
    if time_range is None:
        return mask
    start, stop = time_range
    return (time >= float(start)) & (time <= float(stop))


def _baseline_correct(
    time: np.ndarray,
    signal: np.ndarray,
    baseline_range: tuple[float, float] | None,
) -> np.ndarray:
    corrected = np.asarray(signal, dtype=float).copy()
    if baseline_range is None:
        return corrected
    mask = _time_mask(time, baseline_range)
    if np.any(mask):
        corrected -= float(np.nanmean(corrected[mask]))
    return corrected


def _sample_rate_from_time(time: np.ndarray) -> float:
    dt = np.nanmedian(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cannot infer sample rate from non-monotonic or degenerate time array.")
    return float(1.0 / dt)


def _line_of_sight_points(ods: Any, channel: int) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    base = f"soft_x_rays.channel.{channel}.line_of_sight"
    try:
        first = (
            float(ods[f"{base}.first_point.r"]),
            float(ods[f"{base}.first_point.z"]),
        )
        second = (
            float(ods[f"{base}.second_point.r"]),
            float(ods[f"{base}.second_point.z"]),
        )
        phi = float(ods[f"{base}.first_point.phi"])
    except Exception:
        return None
    return first, second, phi


def _plot_wall_from_ods(ax: Any, ods: Any) -> bool:
    try:
        descriptions = ods["wall.description_2d"]
    except Exception:
        return False

    plotted = False
    try:
        iterator = range(len(descriptions))
    except Exception:
        return False

    for desc_idx in iterator:
        try:
            units = descriptions[desc_idx]["limiter.unit"]
            unit_iter = range(len(units))
        except Exception:
            continue
        for unit_idx in unit_iter:
            try:
                r = _as_array(units[unit_idx]["outline.r"])
                z = _as_array(units[unit_idx]["outline.z"])
            except Exception:
                continue
            ax.plot(r, z, color="0.55", lw=1.0, alpha=0.8)
            plotted = True
    return plotted


def plot_soft_x_ray_los(
    ods: Any,
    channels: Sequence[int] | int | None = None,
    arrays: Sequence[str] | str | None = None,
    ax: Any | None = None,
    show_wall: bool = True,
    show_channel_labels: bool = False,
    title: str | None = None,
    show: bool = True,
):
    """Plot SXR line-of-sight endpoints from ``soft_x_rays.channel`` metadata."""
    selected = _normalise_channels(ods, channels=channels, arrays=arrays)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 6.0))
    else:
        fig = ax.figure

    if show_wall:
        _plot_wall_from_ods(ax, ods)

    labels_used: set[str] = set()
    phis: list[float] = []
    plotted = 0
    for channel in selected:
        los = _line_of_sight_points(ods, channel)
        if los is None:
            continue
        first, second, phi = los
        array_name, array_channel = _channel_array_and_number(ods, channel)
        color = _ARRAY_COLORS.get(array_name, "tab:gray")
        label = array_name or "SXR"
        ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=color,
            lw=1.0,
            alpha=0.65,
            label=label if label not in labels_used else None,
        )
        labels_used.add(label)
        phis.append(phi)
        plotted += 1
        if show_channel_labels:
            text = str(array_channel if array_channel is not None else channel + 1)
            ax.text(second[0], second[1], text, fontsize=7, color=color)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    if plotted:
        unique_phi = np.unique(np.round(np.asarray(phis) * 180.0 / np.pi, 3))
        phi_text = f" @ phi={unique_phi[0]:g} deg" if unique_phi.size == 1 else ""
        ax.set_title(title or f"SXR Line of Sight{phi_text}")
    else:
        ax.set_title(title or "SXR Line of Sight")
    if labels_used:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_soft_x_ray_signal(
    ods: Any,
    channels: Sequence[int] | int | None = None,
    arrays: Sequence[str] | str | None = None,
    time_range: tuple[float, float] | None = None,
    baseline_range: tuple[float, float] | None = None,
    scale: float = 1.0,
    ylabel: str = "Brightness proxy [a.u.]",
    ax: Any | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Plot one or more SXR brightness time traces."""
    selected = _normalise_channels(ods, channels=channels, arrays=arrays)
    if channels is None and arrays is None:
        selected = selected[:1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
    else:
        fig = ax.figure

    for channel in selected:
        time, signal = _channel_time_and_signal(ods, channel)
        signal = _baseline_correct(time, signal, baseline_range)
        mask = _time_mask(time, time_range)
        label = _channel_name(ods, channel)
        ax.plot(time[mask], signal[mask] * scale, lw=1.4, label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "SXR Representative Signal")
    if len(selected) > 1:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_soft_x_ray_spectrogram(
    ods: Any,
    channel: int = 0,
    time_range: tuple[float, float] | None = None,
    baseline_range: tuple[float, float] | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    max_frequency: float | None = 90_000.0,
    log_power: bool = True,
    ax: Any | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Plot a single-channel SXR spectrogram."""
    time, signal = _channel_time_and_signal(ods, int(channel))
    signal = _baseline_correct(time, signal, baseline_range)
    mask = _time_mask(time, time_range)
    time_window = time[mask]
    signal_window = signal[mask]
    if signal_window.size < 4:
        raise ValueError("Selected time window is too short for spectrogram.")

    fs = _sample_rate_from_time(time_window)
    nperseg = int(min(nperseg, signal_window.size))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(noverlap, max(nperseg - 1, 0)))
    frequencies, segment_time, power = scipy_spectrogram(
        signal_window,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    segment_time = segment_time + float(time_window[0])
    if max_frequency is not None:
        freq_mask = frequencies <= float(max_frequency)
        frequencies = frequencies[freq_mask]
        power = power[freq_mask]

    image_data = 10.0 * np.log10(power + np.finfo(float).eps) if log_power else power
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(segment_time, frequencies / 1e3, image_data, shading="auto", cmap="turbo")
    label = "PSD [dB]" if log_power else "PSD"
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_title(title or f"SXR Spectrogram - {_channel_name(ods, int(channel))}")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_soft_x_ray_pattern(
    ods: Any,
    channels: Sequence[int] | int | None = None,
    arrays: Sequence[str] | str | None = None,
    time_range: tuple[float, float] | None = None,
    baseline_range: tuple[float, float] | None = None,
    scale: float = 1.0,
    orientation: str = "time_vertical",
    cmap: str = "turbo",
    ax: Any | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Plot SXR brightness as a chord-time heatmap."""
    selected = _normalise_channels(ods, channels=channels, arrays=arrays)
    if not selected:
        raise ValueError("No SXR channels selected for pattern plot.")

    traces = []
    chord_numbers = []
    reference_time = None
    reference_mask = None
    for channel in selected:
        time, signal = _channel_time_and_signal(ods, channel)
        signal = _baseline_correct(time, signal, baseline_range)
        mask = _time_mask(time, time_range)
        if reference_time is None:
            reference_time = time[mask]
            reference_mask = mask
        elif time.size != reference_mask.size or not np.allclose(time[mask], reference_time):
            raise ValueError("Selected channels do not share the same time base/window.")
        traces.append(signal[mask] * scale)
        _, array_channel = _channel_array_and_number(ods, channel)
        chord_numbers.append(array_channel if array_channel is not None else channel + 1)

    data = np.column_stack(traces)
    chord_numbers = np.asarray(chord_numbers, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
    else:
        fig = ax.figure

    if orientation == "time_vertical":
        mesh = ax.pcolormesh(chord_numbers, reference_time, data, shading="auto", cmap=cmap)
        ax.set_xlabel("Chord #")
        ax.set_ylabel("Time [s]")
    elif orientation == "time_horizontal":
        mesh = ax.pcolormesh(reference_time, chord_numbers, data.T, shading="auto", cmap=cmap)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Chord #")
    else:
        raise ValueError("orientation must be 'time_vertical' or 'time_horizontal'.")

    fig.colorbar(mesh, ax=ax, label="Brightness proxy [a.u.]")
    ax.set_title(title or "SXR Chord-Time Pattern")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_soft_x_ray_overview(
    ods: Any,
    los_arrays: Sequence[str] | str | None = None,
    signal_channels: Sequence[int] | int | None = None,
    spectrogram_channel: int = 0,
    pattern_channels: Sequence[int] | int | None = None,
    pattern_arrays: Sequence[str] | str | None = None,
    time_range: tuple[float, float] | None = None,
    baseline_range: tuple[float, float] | None = None,
    show: bool = True,
):
    """Create a compact 2x2 overview of SXR LOS, signal, spectrogram, and pattern."""
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.5))
    plot_soft_x_ray_los(ods, arrays=los_arrays, ax=axs[0, 0], show=False)
    plot_soft_x_ray_signal(
        ods,
        channels=signal_channels if signal_channels is not None else spectrogram_channel,
        time_range=time_range,
        baseline_range=baseline_range,
        ax=axs[0, 1],
        show=False,
    )
    plot_soft_x_ray_spectrogram(
        ods,
        channel=spectrogram_channel,
        time_range=time_range,
        baseline_range=baseline_range,
        ax=axs[1, 0],
        show=False,
    )
    plot_soft_x_ray_pattern(
        ods,
        channels=pattern_channels,
        arrays=pattern_arrays,
        time_range=time_range,
        baseline_range=baseline_range,
        orientation="time_horizontal",
        ax=axs[1, 1],
        show=False,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


soft_x_rays_los = plot_soft_x_ray_los
soft_x_rays_time = plot_soft_x_ray_signal
soft_x_rays_signal = plot_soft_x_ray_signal
soft_x_rays_spectrogram = plot_soft_x_ray_spectrogram
soft_x_rays_pattern = plot_soft_x_ray_pattern
soft_x_rays_overview = plot_soft_x_ray_overview

__all__ = [
    "plot_soft_x_ray_los",
    "plot_soft_x_ray_signal",
    "plot_soft_x_ray_spectrogram",
    "plot_soft_x_ray_pattern",
    "plot_soft_x_ray_overview",
    "soft_x_rays_los",
    "soft_x_rays_time",
    "soft_x_rays_signal",
    "soft_x_rays_spectrogram",
    "soft_x_rays_pattern",
    "soft_x_rays_overview",
]
