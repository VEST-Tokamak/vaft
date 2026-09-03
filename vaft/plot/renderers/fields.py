"""Canonical ``<domain>_field_<quantity>`` renderers for 2D scalar fields."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Field2D
from ..registry import renderer
from ..style import finalize, resolve_axes
from .geometry import draw_geometry_layer

__all__ = [
    "electron_density_field",
    "electron_temperature_field",
    "equilibrium_field_psi",
    "equilibrium_field_psi_vacuum",
    "render_field_2d",
]

_DEFAULT_FIGSIZE = (6.0, 7.0)


def render_field_2d(
    model: Field2D,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    cmap: str = "viridis",
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Field2D` as filled or line contours with its overlays."""
    # A caller that owns a colorbar axes (a figure that redraws this panel,
    # issue #261) keeps its layout fixed by passing it; otherwise Matplotlib
    # takes the space from the panel as usual.  Taken out before the style
    # reaches the contour call.
    colorbar_axes = style.pop("colorbar_ax", None)
    if not isinstance(model, Field2D):
        raise TypeError(
            f"expected a vaft.plot.models.Field2D; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    levels = model.contour_levels
    contour_kwargs = {"cmap": cmap, **style}
    if levels is not None:
        contour_kwargs["levels"] = levels
    draw = axes.contourf if model.filled else axes.contour
    mappable = draw(model.r, model.z, model.values, **contour_kwargs)
    if colorbar:
        if colorbar_axes is not None:
            figure.colorbar(mappable, cax=colorbar_axes, label=model.value_label)
        else:
            figure.colorbar(mappable, ax=axes, label=model.value_label)

    for layer in model.overlays:
        draw_geometry_layer(axes, layer)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    if model.title:
        axes.set_title(model.title)
    if model.aspect_equal:
        axes.set_aspect("equal", adjustable="box")
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def _field_renderer(*, domain: str, subject: str, quantity: str, description: str,
                    ids: tuple[str, ...], required_paths: tuple[str, ...],
                    optional_paths: tuple[str, ...] = ()):
    return renderer(
        domain=domain, subject=subject, view="field", quantity=quantity,
        model=Field2D, description=description, ids=ids,
        required_paths=required_paths, optional_paths=optional_paths,
    )


@_field_renderer(
    domain="equilibrium", quantity="psi",
    subject="equilibrium",
    description="Reconstructed poloidal flux map on the equilibrium (R, Z) grid.",
    ids=("equilibrium", "wall"),
    required_paths=(
        "equilibrium.time_slice.{i}.profiles_2d.{j}.grid.dim1",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.grid.dim2",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
    optional_paths=(
        "equilibrium.time_slice.{i}.boundary.outline.r",
        "equilibrium.time_slice.{i}.boundary.outline.z",
        "wall.description_2d.{i}.limiter.unit.{j}.outline.r",
    ),
)
def equilibrium_field_psi(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Reconstructed poloidal flux map on the equilibrium (R, Z) grid."""
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="equilibrium", quantity="psi_vacuum",
    subject="equilibrium",
    description="Vacuum poloidal flux from the PF coils alone, without plasma.",
    ids=("pf_active", "pf_passive", "wall", "spectrometer_uv", "equilibrium"),
    required_paths=("pf_active.time", "pf_active.coil.{i}.current.data"),
    optional_paths=("wall.description_2d.{i}.limiter.unit.{j}.outline.r",),
)
def equilibrium_field_psi_vacuum(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Vacuum poloidal flux from the PF coils alone, without plasma."""
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="core_profiles", quantity="",
    subject="electron_temperature",
    description="Electron temperature mapped onto the poloidal plane.",
    ids=("core_profiles", "equilibrium", "wall"),
    required_paths=(
        "core_profiles.profiles_1d.{i}.electrons.temperature",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.rho_tor_norm",),
)
def electron_temperature_field(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Electron temperature mapped onto the poloidal plane."""
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="core_profiles", quantity="",
    subject="electron_density",
    description="Electron density mapped onto the poloidal plane.",
    ids=("core_profiles", "equilibrium", "wall"),
    required_paths=(
        "core_profiles.profiles_1d.{i}.electrons.density",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.rho_tor_norm",),
)
def electron_density_field(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Electron density mapped onto the poloidal plane."""
    return render_field_2d(model, ax=ax, show=show, **style)
