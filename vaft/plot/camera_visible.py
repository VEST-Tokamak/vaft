"""FAST-camera frame plotting for OMAS ``camera_visible`` IDS data.

OMAS provides no built-in plot method for ``camera_visible`` (unlike
``equilibrium``/``wall``/etc., which are wrapped via upstream
``ods.plot_*`` methods in :mod:`vaft.plot.topview`) and no generic 2D-image
plot helper (``ods.plot_quantity`` only handles 1D quantities via
``ods.xarray``). Following the same convention as
:mod:`vaft.plot.soft_x_rays` for a diagnostic IDS without upstream OMAS plot
support, this module reads ``camera_visible`` paths directly off the ODS
with private helpers and renders with matplotlib -- no separate data-access
layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CMAP = "gray"


def _camera_visible_frame_count(ods: Any, channel: int = 0, detector: int = 0) -> int:
    try:
        return len(ods[f"camera_visible.channel.{channel}.detector.{detector}.frame"])
    except Exception as exc:
        raise KeyError(
            f"ODS does not contain camera_visible.channel.{channel}.detector.{detector}.frame data."
        ) from exc


def _channel_name(ods: Any, channel: int = 0) -> str:
    try:
        return str(ods[f"camera_visible.channel.{channel}.name"])
    except Exception:
        return f"Camera Ch {channel}"


def _frame_times(ods: Any, channel: int = 0, detector: int = 0) -> np.ndarray:
    n_frames = _camera_visible_frame_count(ods, channel, detector)
    prefix = f"camera_visible.channel.{channel}.detector.{detector}.frame"
    return np.asarray([float(ods[f"{prefix}.{i}.time"]) for i in range(n_frames)], dtype=float)


def _nearest_frame_index(ods: Any, time: float, channel: int = 0, detector: int = 0) -> int:
    times = _frame_times(ods, channel, detector)
    if times.size == 0:
        raise KeyError(f"camera_visible.channel.{channel}.detector.{detector} has no frames.")
    return int(np.argmin(np.abs(times - float(time))))


def _resolve_frame_index(
    ods: Any,
    *,
    channel: int,
    detector: int,
    frame_index: int | None,
    time: float | None,
) -> int:
    if frame_index is not None and time is not None:
        raise ValueError("Specify at most one of frame_index or time.")
    if frame_index is not None:
        return int(frame_index)
    if time is not None:
        return _nearest_frame_index(ods, time, channel=channel, detector=detector)
    return 0


def _frame_image(ods: Any, frame_index: int, channel: int = 0, detector: int = 0) -> np.ndarray:
    n_frames = _camera_visible_frame_count(ods, channel, detector)
    if not (0 <= frame_index < n_frames):
        raise IndexError(f"frame_index {frame_index} out of range [0, {n_frames}).")
    path = f"camera_visible.channel.{channel}.detector.{detector}.frame.{frame_index}.image_raw"
    return np.asarray(ods[path])


def _frame_time(ods: Any, frame_index: int, channel: int = 0, detector: int = 0) -> float:
    return float(ods[f"camera_visible.channel.{channel}.detector.{detector}.frame.{frame_index}.time"])


def _frame_title(channel_name: str, frame_index: int, time_value: float) -> str:
    return f"{channel_name} frame {frame_index} @ t={time_value:.4f}s"


def plot_camera_visible_frame(
    ods: Any,
    *,
    channel: int = 0,
    detector: int = 0,
    frame_index: int | None = None,
    time: float | None = None,
    ax: Any | None = None,
    cmap: str = DEFAULT_CMAP,
    title: str | None = None,
    colorbar: bool = True,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot one FAST-camera frame from ``camera_visible.channel[:].detector[:].frame[:]``.

    Select a frame either by explicit ``frame_index`` or by nearest ``time``
    (seconds); if neither is given, the first frame is shown. The image is
    plotted with grayscale intensity (raw digital levels, uncalibrated) using
    ``origin="upper"`` so row 0 is the top of the sensor, matching the
    row-major ``(lines_n, columns_n)`` orientation the frame array is stored
    in.
    """
    idx = _resolve_frame_index(ods, channel=channel, detector=detector, frame_index=frame_index, time=time)
    image = _frame_image(ods, idx, channel=channel, detector=detector)
    time_value = _frame_time(ods, idx, channel=channel, detector=detector)
    channel_name = _channel_name(ods, channel)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 6.0))
    else:
        fig = ax.figure

    im = ax.imshow(image, cmap=cmap, origin="upper", aspect="equal")
    if colorbar:
        fig.colorbar(im, ax=ax, label="Digital levels")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title or _frame_title(channel_name, idx, time_value))

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def animate_camera_visible(
    ods: Any,
    *,
    channel: int = 0,
    detector: int = 0,
    frame_indices: Sequence[int] | None = None,
    interval_ms: float = 100.0,
    cmap: str = DEFAULT_CMAP,
    title: str | None = None,
    ax: Any | None = None,
    colorbar: bool = True,
    save_path: str | Path | None = None,
    fps: float = 10.0,
    show: bool = True,
):
    """Animate a sequence of FAST-camera frames, optionally saving to disk.

    ``frame_indices`` defaults to every frame in the detector, in order. If
    ``save_path`` is given, the animation is written there instead of shown
    live -- ``.gif`` uses Pillow, any other extension (e.g. ``.mp4``) uses
    ffmpeg. All frames use a shared, fixed color scale (computed once from the
    selected frames) so intensity is directly comparable across the sequence.

    Returns ``(fig, ax, anim)`` -- callers must keep a reference to ``anim``
    alive (e.g. by not discarding the return value) or matplotlib may garbage
    collect the animation before it plays/renders.
    """
    from matplotlib import animation

    n_frames = _camera_visible_frame_count(ods, channel=channel, detector=detector)
    indices = list(frame_indices) if frame_indices is not None else list(range(n_frames))
    if not indices:
        raise ValueError("No frames to animate.")

    images = [_frame_image(ods, idx, channel=channel, detector=detector) for idx in indices]
    times = {idx: _frame_time(ods, idx, channel=channel, detector=detector) for idx in indices}
    channel_name = _channel_name(ods, channel)

    vmin = min(float(np.min(image)) for image in images)
    vmax = max(float(np.max(image)) for image in images)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 6.0))
    else:
        fig = ax.figure

    im = ax.imshow(images[0], cmap=cmap, origin="upper", aspect="equal", vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(im, ax=ax, label="Digital levels")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    title_text = ax.set_title(title or _frame_title(channel_name, indices[0], times[indices[0]]))

    def _update(step: int):
        idx = indices[step]
        im.set_data(images[step])
        if title is None:
            title_text.set_text(_frame_title(channel_name, idx, times[idx]))
        return im, title_text

    anim = animation.FuncAnimation(fig, _update, frames=len(indices), interval=interval_ms, blit=False)

    if save_path is not None:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=fps) if save_path.suffix.lower() == ".gif" else animation.FFMpegWriter(fps=fps)
        anim.save(str(save_path), writer=writer)
    elif show:
        plt.show()

    return fig, ax, anim


def _draw_efit_overlay(
    ax: Any,
    overlay: dict[str, Any],
    *,
    show_wall: bool,
    show_lcfs: bool,
    show_magnetic_axis: bool,
) -> None:
    """Scatter the wall/LCFS/magnetic-axis/flux-surface projections computed by
    ``compute_camera_visible_efit_overlay`` onto ``ax``. Shared by
    :func:`plot_camera_visible_efit_overlay` and
    :func:`plot_camera_visible_field_line` so the drawing logic lives once.
    """
    if show_wall and overlay["wall_uv"].size:
        ax.scatter(overlay["wall_uv"][:, 0], overlay["wall_uv"][:, 1], s=2, color="yellow", label="Wall")
    if show_lcfs and overlay["lcfs_uv"].size:
        ax.scatter(overlay["lcfs_uv"][:, 0], overlay["lcfs_uv"][:, 1], s=3, color="magenta", label="LCFS")
    if show_magnetic_axis and overlay["magnetic_axis_uv"].size:
        ax.scatter(
            overlay["magnetic_axis_uv"][:, 0],
            overlay["magnetic_axis_uv"][:, 1],
            s=30,
            color="cyan",
            marker="+",
            linewidths=1.5,
            label="Magnetic axis",
        )

    flux_surfaces_uv = overlay.get("flux_surfaces_uv") or {}
    if flux_surfaces_uv:
        levels_sorted = sorted(flux_surfaces_uv)
        surface_cmap = plt.get_cmap("cool", max(len(levels_sorted), 1))
        for i, level in enumerate(levels_sorted):
            points = flux_surfaces_uv[level]
            if points.size == 0:
                continue
            label = "psi surfaces (inner to outer)" if i == 0 else None
            ax.scatter(points[:, 0], points[:, 1], s=1, color=surface_cmap(i / max(len(levels_sorted) - 1, 1)), label=label)


def plot_camera_visible_efit_overlay(
    ods: Any,
    shot: int,
    *,
    channel: int = 0,
    detector: int = 0,
    frame_index: int | None = None,
    time: float | None = None,
    ax: Any | None = None,
    cmap: str = DEFAULT_CMAP,
    show_wall: bool = True,
    show_lcfs: bool = True,
    show_magnetic_axis: bool = True,
    flux_surface_levels: Sequence[float] = (0.25, 0.5, 0.75, 0.95),
    title: str | None = None,
    colorbar: bool = True,
    show: bool = True,
) -> tuple[Any, Any]:
    """Overlay projected EFIT/equilibrium geometry on a FAST-camera frame.

    Draws the raw camera frame (see :func:`plot_camera_visible_frame`) and
    scatters the calibrated pinhole-camera projection of the wall (yellow),
    LCFS (magenta), magnetic axis (cyan '+'), and selected flux surfaces
    (colormap, inner to outer) computed by
    :func:`vaft.omas.process_wrapper.compute_camera_visible_efit_overlay`.
    Only the three shots with packaged calibration are supported (34764,
    39915, 47518); other shots raise ``FileNotFoundError``.

    Frame selection (``frame_index``/``time``) matches
    :func:`plot_camera_visible_frame`. The camera pose is used only to
    project geometry -- projected coordinates are never written back into
    ``ods``.
    """
    from vaft.omas.process_wrapper import compute_camera_visible_efit_overlay

    idx = _resolve_frame_index(ods, channel=channel, detector=detector, frame_index=frame_index, time=time)
    image = _frame_image(ods, idx, channel=channel, detector=detector)
    time_value = _frame_time(ods, idx, channel=channel, detector=detector)
    channel_name = _channel_name(ods, channel)

    overlay = compute_camera_visible_efit_overlay(
        ods,
        shot,
        channel=channel,
        detector=detector,
        frame_index=idx,
        flux_surface_levels=tuple(flux_surface_levels),
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 6.0))
    else:
        fig = ax.figure

    im = ax.imshow(image, cmap=cmap, origin="upper", aspect="equal")
    if colorbar:
        fig.colorbar(im, ax=ax, label="Digital levels")

    _draw_efit_overlay(ax, overlay, show_wall=show_wall, show_lcfs=show_lcfs, show_magnetic_axis=show_magnetic_axis)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title or f"{channel_name} frame {idx} @ t={time_value:.4f}s -- shot {shot} EFIT overlay")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_camera_visible_field_line(
    ods: Any,
    shot: int,
    r0: float,
    z0: float,
    *,
    phi0: float = 0.0,
    channel: int = 0,
    detector: int = 0,
    frame_index: int | None = None,
    time: float | None = None,
    dphi_deg: float = 1.0,
    max_length_m: float = 50.0,
    direction: str = "forward",
    use_wall_boundary: bool = True,
    ax: Any | None = None,
    cmap: str = DEFAULT_CMAP,
    field_line_color: str = "red",
    show_wall: bool = False,
    show_lcfs: bool = False,
    show_magnetic_axis: bool = False,
    flux_surface_levels: Sequence[float] = (),
    title: str | None = None,
    colorbar: bool = True,
    show: bool = True,
) -> tuple[Any, Any]:
    """Overlay a traced magnetic field line on a FAST-camera frame.

    Traces a field line from ``(r0, z0, phi0)`` (see
    :func:`vaft.omas.process_wrapper.compute_field_line_trace` for the
    integration/termination contract: fixed-step RK4 in toroidal angle,
    ``dphi_deg`` step, terminating at the wall or ``max_length_m``) using the
    equilibrium time slice nearest the selected camera frame's time, and
    projects it through the calibrated pinhole camera model for ``shot``
    (34764, 39915, or 47518 -- other shots raise ``FileNotFoundError``).
    Draws it as a connected line with markers at the traced start/end points.

    Optionally also draws the wall/LCFS/magnetic-axis/flux-surface overlay
    (``show_wall``/``show_lcfs``/``show_magnetic_axis``/``flux_surface_levels``,
    all off by default here to keep the field line legible) using the same
    projection as :func:`plot_camera_visible_efit_overlay`.
    """
    from vaft.omas.process_wrapper import compute_camera_visible_efit_overlay, compute_camera_visible_field_line_overlay

    idx = _resolve_frame_index(ods, channel=channel, detector=detector, frame_index=frame_index, time=time)
    image = _frame_image(ods, idx, channel=channel, detector=detector)
    time_value = _frame_time(ods, idx, channel=channel, detector=detector)
    channel_name = _channel_name(ods, channel)

    result = compute_camera_visible_field_line_overlay(
        ods, shot, r0=r0, z0=z0, phi0=phi0, channel=channel, detector=detector,
        frame_index=idx, dphi_deg=dphi_deg, max_length_m=max_length_m,
        direction=direction, use_wall_boundary=use_wall_boundary,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 6.0))
    else:
        fig = ax.figure

    im = ax.imshow(image, cmap=cmap, origin="upper", aspect="equal")
    if colorbar:
        fig.colorbar(im, ax=ax, label="Digital levels")

    if show_wall or show_lcfs or show_magnetic_axis or flux_surface_levels:
        efit_overlay = compute_camera_visible_efit_overlay(
            ods, shot, channel=channel, detector=detector, frame_index=idx,
            flux_surface_levels=tuple(flux_surface_levels),
        )
        _draw_efit_overlay(ax, efit_overlay, show_wall=show_wall, show_lcfs=show_lcfs, show_magnetic_axis=show_magnetic_axis)

    field_line_uv = result["field_line_uv"]
    if field_line_uv.shape[0] >= 2:
        ax.plot(field_line_uv[:, 0], field_line_uv[:, 1], color=field_line_color, linewidth=1.5, label="Field line")
        ax.scatter(field_line_uv[0, 0], field_line_uv[0, 1], color="lime", s=40, zorder=5, label="Start")
        ax.scatter(field_line_uv[-1, 0], field_line_uv[-1, 1], color="blue", s=40, zorder=5, label="End")
    elif field_line_uv.shape[0] == 1:
        ax.scatter(field_line_uv[0, 0], field_line_uv[0, 1], color="lime", s=40, zorder=5, label="Start")

    reason = result["trace"]["termination_reason"]
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(
        title
        or f"{channel_name} frame {idx} @ t={time_value:.4f}s -- shot {shot} field line\n"
        f"R0={r0:.3f}m, Z0={z0:.3f}m, stop: {reason}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


__all__ = [
    "DEFAULT_CMAP",
    "animate_camera_visible",
    "plot_camera_visible_efit_overlay",
    "plot_camera_visible_field_line",
    "plot_camera_visible_frame",
]
