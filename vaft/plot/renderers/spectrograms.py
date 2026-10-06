"""Canonical ``<domain>_spectrogram[_<quantity>]`` renderers."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Spectrogram
from ..registry import renderer
from ..presentation import presented, resolve_color
from ..style import finalize, resolve_axes

__all__ = [
    "camera_visible_spectrogram",
    "interferometer_spectrogram",
    "mirnov_spectrogram",
    "render_spectrogram",
    "soft_x_rays_spectrogram",
]

_DEFAULT_FIGSIZE = (8.0, 4.0)


@presented(default_figsize=_DEFAULT_FIGSIZE)
def render_spectrogram(
    model: Spectrogram,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    format: str | None = None,
    theme: str | None = None,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Spectrogram` as a time-frequency mesh."""
    if not isinstance(model, Spectrogram):
        raise TypeError(
            f"expected a vaft.plot.models.Spectrogram; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)
    # A grid that reserved a colorbar cell beside this panel hands it over, so
    # the mesh keeps the width of the panels it is aligned with (#1467).
    colorbar_axes = style.pop("colorbar_ax", None)

    style.setdefault("shading", "auto")
    style.setdefault("cmap", model.cmap)
    mesh = axes.pcolormesh(model.time, model.frequency, model.magnitude, **style)
    if colorbar:
        if colorbar_axes is not None:
            figure.colorbar(mesh, cax=colorbar_axes, label=model.value_label)
        else:
            figure.colorbar(mesh, ax=axes, label=model.value_label)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    if model.title:
        axes.set_title(model.title)
    if model.max_frequency is not None:
        axes.set_ylim(0.0, model.max_frequency)
    if model.ridge_time is not None:
        # The tracked ridge (issue #1005); NaN windows leave gaps, not a line to zero.
        axes.plot(
            model.ridge_time, model.ridge_frequency, color=resolve_color("palette:2"),
            linewidth=1.4, label=model.ridge_label or "tracked ridge",
        )
        axes.legend(loc="upper right", fontsize="small")
    return finalize(figure, axes, show=show, tight_layout=ax is None)


@renderer(
    domain="camera_visible",
    subject="camera_visible",
    view="spectrogram",
    quantity="",
    model=Spectrogram,
    description=(
        "Time-frequency map of the FAST-camera intensity summed over one image "
        "region, the camera side of the published camera/magnetics comparison "
        "(issue #161)."
    ),
    ids=("camera_visible",),
    required_paths=(
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw",
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.time",
    ),
)
def camera_visible_spectrogram(
    model: Spectrogram, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Time-frequency map of the summed FAST-camera intensity."""
    return render_spectrogram(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="mirnov",
    view="spectrogram",
    quantity="",
    model=Spectrogram,
    description="Time-frequency map of one Mirnov coil signal.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.voltage.data",),
    optional_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
)
def mirnov_spectrogram(
    model: Spectrogram, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Time-frequency map of one Mirnov coil signal.

    Interpretation
    --------------
    The spectral magnitude of one Mirnov coil's voltage in sliding windows.  A
    coherent MHD mode appears as a narrow band, and its frequency history shows
    onset, frequency chirping as rotation changes, and locking when the band
    drops to zero frequency and vanishes; broadband activity and crashes appear
    as vertical stripes.

    Options
    -------
    ``method=`` chooses the transform: a short-time Fourier transform, the
    Hann-window FFT the VEST Mirnov analyses were written against, or a
    continuous wavelet, whose resolution varies with frequency and which needs
    a named ``frequency_range=``.  Window length trades time against frequency
    resolution.  ``frequency_range=`` names the analysis band.  ``track=``
    overlays the ridge of the dominant frequency in a band, with ``max_jump=``
    limiting how far it may move between windows.

    Limitations
    -----------
    One coil gives no mode numbers: identifying m and n needs the phase across
    an array.  The coil voltage is a time derivative, so higher frequencies are
    emphasized.  The observed frequency is in the laboratory frame and includes
    the plasma rotation's Doppler shift, so it is not the mode frequency in the
    plasma frame.  A narrow band can also be pickup or aliasing.

    See Also
    --------
    mirnov_time_voltage : the coil signals themselves.
    mirnov_spatial_phase : toroidal phase per band, for the mode number n.
    """
    return render_spectrogram(model, ax=ax, show=show, **style)


@renderer(
    domain="soft_x_rays",
    subject="soft_x_rays",
    view="spectrogram",
    model=Spectrogram,
    description="Time-frequency map of one soft X-ray channel.",
    ids=("soft_x_rays",),
    required_paths=("soft_x_rays.channel.{i}.brightness.data",),
    optional_paths=(
        "soft_x_rays.channel.{i}.brightness.time",
        "soft_x_rays.channel.{i}.power.data",
        "soft_x_rays.channel.{i}.power.time",
        "soft_x_rays.time",
    ),
)
def soft_x_rays_spectrogram(
    model: Spectrogram, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Time-frequency map of one soft X-ray channel."""
    return render_spectrogram(model, ax=ax, show=show, **style)


@renderer(
    domain="interferometer",
    subject="interferometer",
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
