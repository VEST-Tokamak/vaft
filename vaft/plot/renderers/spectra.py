"""Canonical ``<domain>_spectrum[_<quantity>]`` renderers.

These draw a power spectral density on log-log axes.  The drawing body is
diagnostic-independent; the registered names are domain-scoped only because the
canonical ``<domain>_<view>_<quantity>`` grammar and the adapter layer require a
domain and its IDS paths.

Nothing here interprets a spectrum.  Reference-slope guides are drawn exactly as
the caller specified them, with the caller's own labels; VAFT supplies no slope
values and no physical meaning for any value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import PowerSpectrum, ReferenceSlope
from ..registry import renderer
from ..style import finalize, resolve_axes

__all__ = [
    "interferometer_spectrum",
    "magnetics_spectrum_mirnov",
    "render_power_spectrum",
    "soft_x_rays_spectrum",
]

_DEFAULT_FIGSIZE = (7.0, 4.5)


def _positive_range(model: PowerSpectrum) -> tuple[float, float] | None:
    """The drawn frequency span, restricted to what log axes can show."""
    if model.x_limits is not None:
        low, high = model.x_limits
    else:
        positive = model.frequency[model.frequency > 0]
        if positive.size == 0:
            return None
        low, high = (float(positive[0]), float(positive[-1]))
    return (low, high) if low > 0 and high > low else None


def _guide_points(
    model: PowerSpectrum, reference: ReferenceSlope
) -> tuple[np.ndarray, np.ndarray] | None:
    """Two endpoints of one power-law guide across the drawn frequency range."""
    span = _positive_range(model)
    if span is None:
        return None
    low, high = span
    edges = np.array([low, high], dtype=float)

    if reference.anchor is not None:
        anchor_frequency, anchor_psd = reference.anchor
    else:
        # Anchor on the measured PSD at the geometric-mean frequency: the
        # midpoint of the drawn range in log space, so the guide sits on the
        # data without favouring either end.
        anchor_frequency = float(np.sqrt(low * high))
        in_range = (model.frequency > 0) & (model.psd > 0)
        if not np.any(in_range):
            return None
        frequencies = model.frequency[in_range]
        index = int(np.argmin(np.abs(frequencies - anchor_frequency)))
        anchor_frequency = float(frequencies[index])
        anchor_psd = float(model.psd[in_range][index])

    return edges, anchor_psd * (edges / anchor_frequency) ** reference.slope


def render_power_spectrum(
    model: PowerSpectrum,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`PowerSpectrum` with its fits, guides and frequency markers."""
    if not isinstance(model, PowerSpectrum):
        raise TypeError(
            f"expected a vaft.plot.models.PowerSpectrum; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    axes.plot(model.frequency, model.psd, label=model.label or None, **style)

    for fit in model.fits:
        axes.plot(fit.x, fit.y, label=fit.label or None, **dict(fit.style))

    for reference in model.reference_slopes:
        points = _guide_points(model, reference)
        if points is None:
            continue
        guide_style = {"linestyle": "--", "linewidth": 1.0, "color": "0.4"}
        guide_style.update(dict(reference.style))
        # No label is synthesized beyond the bare exponent: naming what a slope
        # means is the caller's job, not this renderer's.
        axes.plot(
            *points,
            label=reference.label or f"f^{reference.slope:g}",
            **guide_style,
        )

    for frequency, label in model.marker_frequencies:
        axes.axvline(frequency, color="0.6", linestyle=":", linewidth=1.0,
                     label=label or None)

    if model.log_x:
        axes.set_xscale("log")
    if model.log_y:
        axes.set_yscale("log")
    if model.x_limits is not None:
        axes.set_xlim(*model.x_limits)
    if model.y_limits is not None:
        axes.set_ylim(*model.y_limits)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    if model.title:
        axes.set_title(model.title)
    if legend and axes.get_legend_handles_labels()[0]:
        axes.legend()
    return finalize(figure, axes, show=show)


@renderer(
    domain="magnetics",
    subject="mirnov",
    view="spectrum",
    quantity="mirnov",
    model=PowerSpectrum,
    description="Power spectral density of one Mirnov coil signal.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.voltage.data",),
    optional_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
)
def magnetics_spectrum_mirnov(
    model: PowerSpectrum, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Power spectral density of one Mirnov coil signal."""
    return render_power_spectrum(model, ax=ax, show=show, **style)


@renderer(
    domain="soft_x_rays",
    subject="soft_x_rays",
    view="spectrum",
    model=PowerSpectrum,
    description="Power spectral density of one soft X-ray channel.",
    ids=("soft_x_rays",),
    required_paths=("soft_x_rays.channel.{i}.brightness.data",),
    optional_paths=(
        "soft_x_rays.channel.{i}.brightness.time",
        "soft_x_rays.channel.{i}.power.data",
        "soft_x_rays.channel.{i}.power.time",
        "soft_x_rays.time",
    ),
)
def soft_x_rays_spectrum(
    model: PowerSpectrum, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Power spectral density of one soft X-ray channel."""
    return render_power_spectrum(model, ax=ax, show=show, **style)


@renderer(
    domain="interferometer",
    subject="interferometer",
    view="spectrum",
    model=PowerSpectrum,
    description="Power spectral density of one interferometer channel's line density.",
    ids=("interferometer",),
    required_paths=("interferometer.channel.{i}.n_e_line.data",),
    optional_paths=("interferometer.channel.{i}.n_e_line.time", "interferometer.time"),
)
def interferometer_spectrum(
    model: PowerSpectrum, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Power spectral density of one interferometer channel's line density."""
    return render_power_spectrum(model, ax=ax, show=show, **style)
