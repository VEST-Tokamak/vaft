"""Canonical ``<domain>_image_<quantity>`` and ``<domain>_animation_<quantity>``
renderers for raster pixel data (camera frames), as opposed to
:mod:`vaft.plot.renderers.fields`'s physical ``(r, z)`` scalar fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Image2D, ImageSequence
from ..registry import renderer
from ..style import finalize, resolve_axes
from .geometry import draw_geometry_layer

__all__ = [
    "camera_visible_animation_frames",
    "camera_visible_image_efit_overlay",
    "camera_visible_image_field_line",
    "camera_visible_image_frame",
    "render_image_2d",
    "render_image_sequence",
]

_DEFAULT_FIGSIZE = (5.0, 6.0)


def render_image_2d(
    model: Image2D,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw an :class:`Image2D` with ``imshow`` plus its pixel-space overlays."""
    if not isinstance(model, Image2D):
        raise TypeError(
            f"expected a vaft.plot.models.Image2D; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    image = axes.imshow(
        model.values, cmap=model.cmap, origin=model.origin,
        vmin=model.vmin, vmax=model.vmax, **style,
    )
    if colorbar:
        figure.colorbar(image, ax=axes, label=model.value_label)

    labelled = False
    for layer in model.overlays:
        draw_geometry_layer(axes, layer)
        labelled = labelled or bool(layer.label)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    if model.title:
        axes.set_title(model.title)
    if model.aspect_equal:
        axes.set_aspect("equal", adjustable="box")
    else:
        # imshow defaults to equal aspect; an aspect_equal=False model (e.g. a
        # chord-versus-time map with wildly unequal extents) must actually fill
        # the axes rather than collapse to the pixel aspect ratio.
        axes.set_aspect("auto")
    if labelled:
        axes.legend(loc="best", fontsize="small")
    return finalize(figure, axes, show=show)


def render_image_sequence(
    model: ImageSequence,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    interval_ms: float = 100.0,
    save_path: str | Path | None = None,
    fps: float = 10.0,
    **style: Any,
):
    """Animate an :class:`ImageSequence`, optionally saving it to disk.

    Returns ``(Figure, Axes, FuncAnimation)`` -- the one extension to the
    ``(Figure, Axes)`` renderer contract, since none of the other five view
    kinds models a time animation. Callers must keep a reference to the
    ``FuncAnimation`` alive (e.g. by not discarding the return value) or
    Matplotlib may garbage-collect it before it plays/renders. If
    ``save_path`` is given, the animation is written there instead of shown
    live -- ``.gif`` uses Pillow, any other extension (e.g. ``.mp4``) uses
    ffmpeg.
    """
    from matplotlib import animation

    if not isinstance(model, ImageSequence):
        raise TypeError(
            f"expected a vaft.plot.models.ImageSequence; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    image = axes.imshow(
        model.frames[0], cmap=model.cmap, origin=model.origin,
        vmin=model.vmin, vmax=model.vmax, **style,
    )
    if colorbar:
        figure.colorbar(image, ax=axes, label=model.value_label)
    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    title_text = axes.set_title(model.title)
    if model.aspect_equal:
        axes.set_aspect("equal", adjustable="box")
    else:
        # imshow defaults to equal aspect; an aspect_equal=False model (e.g. a
        # chord-versus-time map with wildly unequal extents) must actually fill
        # the axes rather than collapse to the pixel aspect ratio.
        axes.set_aspect("auto")

    def _update(step: int):
        image.set_data(model.frames[step])
        if not model.title:
            title_text.set_text(f"t={model.time[step]:.4f}s")
        return image, title_text

    anim = animation.FuncAnimation(
        figure, _update, frames=len(model.frames), interval=interval_ms, blit=False
    )

    if save_path is not None:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = (
            animation.PillowWriter(fps=fps)
            if save_path.suffix.lower() == ".gif"
            else animation.FFMpegWriter(fps=fps)
        )
        anim.save(str(save_path), writer=writer)
    elif show:
        import matplotlib.pyplot as plt

        plt.show()

    return figure, axes, anim


def _image_renderer(*, domain: str, subject: str, quantity: str, description: str,
                    ids: tuple[str, ...], required_paths: tuple[str, ...],
                    optional_paths: tuple[str, ...] = ()):
    return renderer(
        domain=domain, subject=subject, view="image", quantity=quantity,
        model=Image2D, description=description, ids=ids,
        required_paths=required_paths, optional_paths=optional_paths,
    )


@_image_renderer(
    domain="camera_visible", quantity="frame",
    subject="camera_visible",
    description="One FAST-camera frame (raw digital levels, uncalibrated).",
    ids=("camera_visible",),
    required_paths=(
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw",
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.time",
    ),
)
def camera_visible_image_frame(
    model: Image2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """One FAST-camera frame (raw digital levels, uncalibrated)."""
    return render_image_2d(model, ax=ax, show=show, **style)


@_image_renderer(
    domain="camera_visible", quantity="efit_overlay",
    subject="camera_visible",
    description=(
        "FAST-camera frame with the calibrated pinhole projection of the "
        "wall, LCFS, magnetic axis, and flux surfaces overlaid."
    ),
    ids=("camera_visible", "equilibrium", "wall"),
    required_paths=(
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw",
        "equilibrium.time_slice.{i}.boundary.outline.r",
        "wall.description_2d.{i}.limiter.unit.{j}.outline.r",
    ),
)
def camera_visible_image_efit_overlay(
    model: Image2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """FAST-camera frame with the projected EFIT/wall overlay."""
    return render_image_2d(model, ax=ax, show=show, **style)


@_image_renderer(
    domain="camera_visible", quantity="field_line",
    subject="camera_visible",
    description=(
        "FAST-camera frame with a traced magnetic field line projected onto it."
    ),
    ids=("camera_visible", "equilibrium"),
    required_paths=(
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
)
def camera_visible_image_field_line(
    model: Image2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """FAST-camera frame with a projected traced field line."""
    return render_image_2d(model, ax=ax, show=show, **style)


@renderer(
    domain="camera_visible", view="animation", quantity="frames", model=ImageSequence,
    subject="camera_visible",
    description="Animate a sequence of FAST-camera frames on a shared color scale.",
    ids=("camera_visible",),
    required_paths=(
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw",
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.time",
    ),
)
def camera_visible_animation_frames(
    model: ImageSequence, *, ax: Axes | None = None, show: bool = False, **style: Any
):
    """Animate a sequence of FAST-camera frames on a shared color scale."""
    return render_image_sequence(model, ax=ax, show=show, **style)
