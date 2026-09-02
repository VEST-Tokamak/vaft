"""Canonical ``<domain>_geometry_<quantity>`` renderers.

VAFT draws machine and diagnostic geometry from
:class:`~vaft.plot.models.GeometryLayers` rather than delegating to OMAS's
``ODS.plot_*_overlay`` methods, so the same renderer serves ODS, IMAS, database
and file-backed inputs.  The ``machine`` domain covers composed views that span
several IDS.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Geometry3DLayers, GeometryLayer, GeometryLayers
from ..registry import renderer
from ..style import finalize, resolve_axes

__all__ = [
    "charge_exchange_geometry_poloidal",
    "coil_3d_geometry3d",
    "coil_3d_geometry_topview",
    "draw_geometry_layer",
    "equilibrium_geometry_boundary",
    "equilibrium_geometry_topview",
    "machine_geometry_poloidal",
    "machine_geometry_topview",
    "magnetics_geometry_poloidal",
    "passive_structure_geometry_poloidal",
    "pf_coil_geometry_poloidal",
    "render_geometry_3d_layers",
    "render_geometry_layers",
    "soft_x_rays_geometry_lines_of_sight",
    "thomson_scattering_geometry_poloidal",
    "wall_geometry_poloidal",
]

_DEFAULT_FIGSIZE = (6.0, 7.0)


def draw_geometry_layer(
    axes: Axes, layer: GeometryLayer, **defaults: Any
) -> None:
    """Draw one :class:`GeometryLayer` into ``axes``.

    ``defaults`` are applied to every layer; the layer's own ``style`` wins.
    """
    options = {**defaults, **layer.style}
    if layer.label:
        options.setdefault("label", layer.label)
    r, z = layer.r, layer.z
    if layer.kind == "points":
        options.setdefault("linestyle", "none")
        options.setdefault("marker", "o")
        axes.plot(r, z, **options)
        return
    if layer.kind == "polygon" and r.size and (r[0] != r[-1] or z[0] != z[-1]):
        r = np.concatenate([r, r[:1]])
        z = np.concatenate([z, z[:1]])
    axes.plot(r, z, **options)


def render_geometry_layers(
    model: GeometryLayers,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    grid: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`GeometryLayers` stack into one equal-aspect axes.

    Keyword styling is applied as a default for every layer; a layer's own
    ``style`` mapping takes precedence.
    """
    if not isinstance(model, GeometryLayers):
        raise TypeError(
            f"expected a vaft.plot.models.GeometryLayers; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    labelled = False
    for layer in model.layers:
        draw_geometry_layer(axes, layer, **style)
        labelled = labelled or bool(layer.label)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    if model.title:
        axes.set_title(model.title)
    if model.aspect_equal:
        axes.set_aspect("equal", adjustable="box")
    if grid:
        axes.grid(True, alpha=0.25)
    if legend and labelled:
        axes.legend(loc="best", fontsize="small")
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def _geometry_renderer(*, domain: str, subject: str, quantity: str, description: str,
                       ids: tuple[str, ...], required_paths: tuple[str, ...],
                       optional_paths: tuple[str, ...] = ()):
    return renderer(
        domain=domain, subject=subject, view="geometry", quantity=quantity,
        model=GeometryLayers, description=description, ids=ids,
        required_paths=required_paths, optional_paths=optional_paths,
    )


@_geometry_renderer(
    domain="pf_active", quantity="poloidal",
    subject="pf_coil",
    description="PF coil outlines in the poloidal plane.",
    ids=("pf_active",),
    required_paths=("pf_active.coil.{i}.element.{j}.geometry.outline.r",
                    "pf_active.coil.{i}.element.{j}.geometry.outline.z"),
    optional_paths=("pf_active.coil.{i}.name",),
)
def pf_coil_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """PF coil outlines in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="pf_passive", quantity="poloidal",
    subject="passive_structure",
    description="Passive conducting-structure loop outlines in the poloidal plane.",
    ids=("pf_passive",),
    required_paths=("pf_passive.loop.{i}.element.{j}.geometry.outline.r",
                    "pf_passive.loop.{i}.element.{j}.geometry.outline.z"),
    optional_paths=("pf_passive.loop.{i}.name",),
)
def passive_structure_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Passive conducting-structure loop outlines in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="magnetics", quantity="poloidal",
    subject="magnetics",
    description="Flux-loop and B-field-probe positions in the poloidal plane.",
    ids=("magnetics",),
    required_paths=(),
    optional_paths=("magnetics.flux_loop.{i}.position.0.r",
                    "magnetics.flux_loop.{i}.position.0.z",
                    "magnetics.b_field_pol_probe.{i}.position.r",
                    "magnetics.b_field_pol_probe.{i}.position.z"),
)
def magnetics_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Flux-loop and B-field-probe positions in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="wall", quantity="poloidal",
    subject="wall",
    description="First-wall and limiter outline in the poloidal plane.",
    ids=("wall",),
    required_paths=("wall.description_2d.{i}.limiter.unit.{j}.outline.r",
                    "wall.description_2d.{i}.limiter.unit.{j}.outline.z"),
)
def wall_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """First-wall and limiter outline in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="equilibrium", quantity="boundary",
    subject="equilibrium",
    description="Last-closed-flux-surface outline in the poloidal plane.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.boundary.outline.r",
                    "equilibrium.time_slice.{i}.boundary.outline.z"),
    optional_paths=("equilibrium.time_slice.{i}.global_quantities.magnetic_axis.r",
                    "equilibrium.time_slice.{i}.global_quantities.magnetic_axis.z"),
)
def equilibrium_geometry_boundary(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Last-closed-flux-surface outline in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="soft_x_rays", quantity="lines_of_sight",
    subject="soft_x_rays",
    description="Soft X-ray detector lines of sight over the poloidal cross-section.",
    ids=("soft_x_rays", "wall"),
    required_paths=("soft_x_rays.channel.{i}.line_of_sight.first_point.r",
                    "soft_x_rays.channel.{i}.line_of_sight.first_point.z",
                    "soft_x_rays.channel.{i}.line_of_sight.second_point.r",
                    "soft_x_rays.channel.{i}.line_of_sight.second_point.z"),
    optional_paths=("wall.description_2d.{i}.limiter.unit.{j}.outline.r",),
)
def soft_x_rays_geometry_lines_of_sight(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Soft X-ray detector lines of sight over the poloidal cross-section."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="thomson_scattering", quantity="poloidal",
    subject="thomson_scattering",
    description="Thomson-scattering measurement positions in the poloidal plane.",
    ids=("thomson_scattering",),
    required_paths=("thomson_scattering.channel.{i}.position.r",
                    "thomson_scattering.channel.{i}.position.z"),
)
def thomson_scattering_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Thomson-scattering measurement positions in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="charge_exchange", quantity="poloidal",
    subject="charge_exchange",
    description="Charge-exchange measurement positions in the poloidal plane.",
    ids=("charge_exchange",),
    required_paths=("charge_exchange.channel.{i}.position.r.data",
                    "charge_exchange.channel.{i}.position.z.data"),
)
def charge_exchange_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Charge-exchange measurement positions in the poloidal plane."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="machine", quantity="poloidal",
    subject="machine",
    description="Composed poloidal machine view: wall, coils, passive structure "
                "and diagnostic positions and sight lines in one axes.",
    ids=("wall", "pf_active", "pf_passive", "magnetics", "thomson_scattering",
         "charge_exchange", "soft_x_rays"),
    required_paths=(),
    optional_paths=("wall.description_2d.{i}.limiter.unit.{j}.outline.r",
                    "pf_active.coil.{i}.element.{j}.geometry.outline.r",
                    "pf_passive.loop.{i}.element.{j}.geometry.outline.r"),
)
def machine_geometry_poloidal(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Composed poloidal machine view."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="equilibrium", quantity="topview",
    subject="equilibrium",
    description="Equilibrium boundary projected into the machine top view.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.boundary.outline.r",),
)
def equilibrium_geometry_topview(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium boundary projected into the machine top view."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="machine", quantity="topview",
    subject="machine",
    description="Composed machine top view: machine-boundary and plasma extent "
                "plus launcher, antenna and pellet-injector geometry.",
    ids=("wall", "equilibrium", "lh_antennas", "ec_launchers", "pellets"),
    required_paths=(),
    optional_paths=("wall.description_2d.{i}.limiter.unit.{j}.outline.r",
                    "equilibrium.time_slice.{i}.boundary.outline.r",
                    "lh_antennas.antenna.{i}.position.r",
                    "ec_launchers.beam.{i}.launching_position.r",
                    "pellets.time_slice.{i}.pellet.{j}.path_geometry.first_point.r"),
)
def machine_geometry_topview(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Composed machine top view."""
    return render_geometry_layers(model, ax=ax, show=show, **style)


def _resolve_3d_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is None:
        figure = plt.figure(figsize=_DEFAULT_FIGSIZE)
        return figure, figure.add_subplot(projection="3d")
    if getattr(ax, "name", "") == "3d":
        return ax.figure, ax
    # A 2D axes was supplied (e.g. from plt.subplots()): replace it with a 3D
    # axes at the same position in the same figure, honoring the contract of
    # never creating a new figure when `ax` is given.
    figure = ax.figure
    subplotspec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
    ax.remove()
    if subplotspec is not None:
        return figure, figure.add_subplot(subplotspec, projection="3d")
    return figure, figure.add_subplot(projection="3d")


def render_geometry_3d_layers(
    model: Geometry3DLayers,
    *,
    ax: Axes | None = None,
    show: bool = False,
    legend: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Geometry3DLayers` stack into one 3D machine view.

    Keyword styling is applied as a default for every layer; a layer's own
    ``style`` mapping takes precedence.  Axis limits are set to a common scale
    so toroidal placement and coil size read true.
    """
    if not isinstance(model, Geometry3DLayers):
        raise TypeError(
            f"expected a vaft.plot.models.Geometry3DLayers; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = _resolve_3d_axes(ax)

    labelled = False
    for layer in model.layers:
        options = {**style, **layer.style}
        if layer.label:
            options.setdefault("label", layer.label)
            labelled = True
        if layer.kind == "points":
            options.setdefault("linestyle", "none")
            options.setdefault("marker", "o")
        axes.plot(layer.x, layer.y, layer.z, **options)

    if model.layers:
        stacked = np.concatenate(
            [np.column_stack([layer.x, layer.y, layer.z]) for layer in model.layers]
        )
        centre = (stacked.max(axis=0) + stacked.min(axis=0)) / 2.0
        half_span = float((stacked.max(axis=0) - stacked.min(axis=0)).max()) / 2.0 or 1.0
        axes.set_xlim(centre[0] - half_span, centre[0] + half_span)
        axes.set_ylim(centre[1] - half_span, centre[1] + half_span)
        axes.set_zlim(centre[2] - half_span, centre[2] + half_span)

    axes.set_xlabel(model.x_label)
    axes.set_ylabel(model.y_label)
    axes.set_zlabel(model.z_label)
    if model.title:
        axes.set_title(model.title)
    if legend and labelled:
        axes.legend(loc="best", fontsize="small")
    return finalize(figure, axes, show=show, tight_layout=ax is None)


@renderer(
    domain="coils_non_axisymmetric", view="geometry3d", model=Geometry3DLayers,
    subject="coil_3d",
    description="Non-axisymmetric 3D coil filaments in machine Cartesian coordinates.",
    ids=("coils_non_axisymmetric",),
    required_paths=(
        "coils_non_axisymmetric.coil.{i}.conductor.{j}.elements.start_points.r",
        "coils_non_axisymmetric.coil.{i}.conductor.{j}.elements.start_points.phi",
        "coils_non_axisymmetric.coil.{i}.conductor.{j}.elements.start_points.z",
    ),
)
def coil_3d_geometry3d(
    model: Geometry3DLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """3D validation view of the non-axisymmetric coil filaments."""
    return render_geometry_3d_layers(model, ax=ax, show=show, **style)


@_geometry_renderer(
    domain="coils_non_axisymmetric", quantity="topview",
    subject="coil_3d",
    description="Non-axisymmetric 3D coil filaments projected into the machine top view.",
    ids=("coils_non_axisymmetric",),
    required_paths=(
        "coils_non_axisymmetric.coil.{i}.conductor.{j}.elements.start_points.r",
        "coils_non_axisymmetric.coil.{i}.conductor.{j}.elements.start_points.phi",
    ),
)
def coil_3d_geometry_topview(
    model: GeometryLayers, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Top view (x-y) of the non-axisymmetric coil filaments."""
    return render_geometry_layers(model, ax=ax, show=show, **style)
