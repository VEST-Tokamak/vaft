"""Canonical ``<domain>_spectrogram[_<quantity>]`` renderers."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Spectrogram
from ..registry import renderer
from ..style import finalize, resolve_axes

__all__ = [
    "interferometer_spectrogram",
    "magnetics_spectrogram_mirnov",
    "render_spectrogram",
    "soft_x_rays_spectrogram",
]

_DEFAULT_FIGSIZE = (8.0, 4.0)


def render_spectrogram(
    model: Spectrogram,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Spectrogram` as a time-frequency mesh."""
    if not isinstance(model, Spectrogram):
        raise TypeError(
            f"expected a vaft.plot.models.Spectrogram; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    style.setdefault("shading", "auto")
    style.setdefault("cmap", model.cmap)
    mesh = axes.pcolormesh(model.time, model.frequency, model.magnitude, **style)
    if colorbar:
        figure.colorbar(mesh, ax=axes, label=model.value_label)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    if model.title:
        axes.set_title(model.title)
    if model.max_frequency is not None:
        axes.set_ylim(0.0, model.max_frequency)
    return finalize(figure, axes, show=show)


@renderer(
    domain="magnetics",
    view="spectrogram",
    quantity="mirnov",
    model=Spectrogram,
    description="Time-frequency map of one Mirnov coil signal.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.voltage.data",),
    optional_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
)
def magnetics_spectrogram_mirnov(
    model: Spectrogram, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Time-frequency map of one Mirnov coil signal."""
    return render_spectrogram(model, ax=ax, show=show, **style)


@renderer(
    domain="soft_x_rays",
    view="spectrogram",
    model=Spectrogram,
    description="Time-frequency map of one soft X-ray channel.",
    ids=("soft_x_rays",),
    required_paths=("soft_x_rays.channel.{i}.power.data",),
    optional_paths=("soft_x_rays.channel.{i}.power.time", "soft_x_rays.time"),
)
def soft_x_rays_spectrogram(
    model: Spectrogram, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Time-frequency map of one soft X-ray channel."""
    return render_spectrogram(model, ax=ax, show=show, **style)


@renderer(
    domain="interferometer",
    view="spectrogram",
    model=Spectrogram,
    description="Time-frequency map of one interferometer channel's line density.",
    ids=("interferometer",),
    required_paths=("interferometer.channel.{i}.n_e_line.data",),
    optional_paths=("interferometer.channel.{i}.n_e_line.time", "interferometer.time"),
)
def interferometer_spectrogram(
    model: Spectrogram, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Time-frequency map of one interferometer channel's line density."""
    return render_spectrogram(model, ax=ax, show=show, **style)
