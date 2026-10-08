"""Canonical ``<domain>_field_<quantity>`` renderers for 2D scalar fields."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Field2D
from ..registry import renderer
from ..presentation import presented, resolve_color
from ..style import finalize, resolve_axes
from .geometry import draw_geometry_layer

__all__ = [
    "field_line_topology_field_connection_length",
    "mhd_linear_field_spectrum",
    "passive_structure_field_wall_reduction",
    "electron_density_field",
    "electron_temperature_field",
    "equilibrium_field_2d",
    "equilibrium_field_psi",
    "equilibrium_field_psi_vacuum",
    "vacuum_field",
    "render_field_2d",
]

_DEFAULT_FIGSIZE = (6.0, 7.0)


#: How many contours ``label_contours`` writes a value on.  Enough to read a
#: gradient off the map, few enough that the labels do not collide.
_MAX_CONTOUR_LABELS = 8


@presented(default_figsize=_DEFAULT_FIGSIZE)
def render_field_2d(
    model: Field2D,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    cmap: str = "viridis",
    format: str | None = None,
    theme: str | None = None,
    label_contours: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Field2D` as filled or line contours with its overlays.

    ``label_contours`` writes each contour's value onto the line itself.  It is
    off by default and worth turning on for a line-contour map, where reading a
    value off the curve beats matching a colour against a colorbar.
    """
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
    if "norm" not in contour_kwargs and colorbar and model.colorbar:
        # figure_options=FigureOptions(norm=...) (issue #1421): a contour set
        # places its levels now, so the norm has to be known now too.  Only a
        # map whose colours a colorbar explains takes it.
        from ..figure_options import draw_norm

        chosen = draw_norm(model.values)
        if chosen is not None:
            contour_kwargs["norm"] = chosen
            if levels is None and isinstance(chosen, LogNorm):
                # Linear levels would put every band in the top decade.
                levels = _levels_under(chosen, model.values)
            if levels is not None and model.extend == "neither":
                # A clim narrower than the data must saturate, not leave holes.
                contour_kwargs["extend"] = "both"
    if model.value_scale == "log" and "norm" not in contour_kwargs:
        # Both halves are needed: the norm spaces the colours and the levels
        # space the bands. A LogNorm with linear levels still puts every
        # band in the top decade, which is the thing a log scale is for.
        finite = model.values[np.isfinite(model.values)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            contour_kwargs["norm"] = LogNorm(vmin=low, vmax=high)
            if levels is None and high > low:
                count = int(style.pop("log_levels", 0)) or 24
                levels = np.logspace(np.log10(low), np.log10(high), count)
    if levels is not None:
        contour_kwargs["levels"] = levels
        if model.extend != "neither":
            contour_kwargs["extend"] = model.extend
    if model.secondary_levels:
        axes.contour(
            model.r, model.z, model.values, levels=list(model.secondary_levels),
            colors=resolve_color("emphasis:lower"), linewidths=0.5, linestyles="--",
        )
    draw = axes.contourf if model.filled else axes.contour
    values = model.values
    if model.region is not None:
        values = np.where(model.region, values, np.nan)
    mappable = draw(model.r, model.z, values, **contour_kwargs)
    if label_contours:
        # Every level labelled is a smear: a psi map draws 40 or more, and
        # their labels overlap into illegibility. Label an evenly spaced
        # handful instead, which is what a reader takes a value off.
        from matplotlib import patheffects

        drawn = np.asarray(getattr(mappable, "levels", ()), dtype=float)
        chosen = drawn[:: max(1, int(np.ceil(drawn.size / _MAX_CONTOUR_LABELS)))]
        labels = axes.clabel(mappable, levels=chosen, inline=True, fontsize=7)
        # A label sitting on a filled map takes the contour's own colour, which
        # is by construction the colour of what it is written on. The halo is
        # what makes it readable there.
        for label in labels:
            label.set_color("0.1")
            label.set_path_effects(
                [patheffects.withStroke(linewidth=2.0, foreground="white")]
            )
    if colorbar and model.colorbar:
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


def _levels_under(norm: Any, values: Any, count: int = 24) -> Any:
    """Contour levels spaced in decades over a log norm's range."""
    low, high = norm.vmin, norm.vmax
    if low is None or high is None or not high > low:
        return None
    return np.logspace(np.log10(low), np.log10(high), count)


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
    # The machine geometry the default overlays draw over the map (issue #483);
    # none of it is required, so availability is still the flux map's own.
    ids=("equilibrium", "wall", "pf_active", "pf_passive"),
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
    """Reconstructed poloidal flux map on the equilibrium (R, Z) grid.

    Interpretation
    --------------
    Shows the poloidal flux psi(R, Z) of one reconstructed slice.  Its contours
    are the cross-sections of the magnetic flux surfaces, so the map is read
    for the plasma's position and shape, the last closed flux surface, the
    magnetic axis, X-points, and how the coil field closes around the plasma.
    At equal level spacing, closely packed contours mark a strong poloidal
    field: the flux gradient is proportional to R B_p.

    Options
    -------
    ``style=`` chooses between flux surfaces at fixed steps of normalized flux
    (continued outside the plasma in grey), the flux normalized to 0 on the
    axis and 1 at the boundary, and a filled map of the flux itself.
    ``units=`` changes only the display scale between full weber and weber per
    radian. ``overlay=`` adds the machine and the equilibrium's own boundary,
    axis and X-points as geometric context.  ``rational_q=`` and
    ``resonances=`` draw the contours of chosen rational surfaces.

    Convention
    ----------
    VAFT stores psi in full weber (IMAS DD); a per-radian display divides by 2
    pi.  The sign and the offset of psi depend on the COCOS and the current
    direction, so between equilibria only flux differences and normalized flux
    are comparable.

    Limitations
    -----------
    The map is the solver's solution for one slice, not an interpolation in
    time.  Inside the plasma the contour shapes depend on the assumed profile
    parametrisation and are constrained only indirectly by external magnetics;
    outside, the flux contains whatever coil and vessel currents the
    reconstruction modelled.

    See Also
    --------
    equilibrium_field_2d : other reconstructed 2-D quantities on the same grid.
    equilibrium_overview : the same map with the slice's profiles and globals.
    """
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="equilibrium", quantity="2d",
    subject="equilibrium",
    description="Any reconstructed 2-D equilibrium quantity on the (R, Z) grid: flux, current density, pressure or field, with the machine drawn over it.",
    ids=("equilibrium", "wall", "pf_active", "pf_passive"),
    required_paths=(
        "equilibrium.time_slice.{i}.profiles_2d.{j}.grid.dim1",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.grid.dim2",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
    optional_paths=(
        "equilibrium.time_slice.{i}.profiles_1d.pressure",
        "equilibrium.time_slice.{i}.profiles_1d.f",
        "equilibrium.time_slice.{i}.profiles_1d.dpressure_dpsi",
        "equilibrium.time_slice.{i}.profiles_1d.f_df_dpsi",
        "equilibrium.time_slice.{i}.boundary.outline.r",
        "equilibrium.time_slice.{i}.boundary.outline.z",
        "pf_active.coil.{i}.element.{j}.geometry.outline.r",
        "wall.description_2d.{i}.limiter.unit.{j}.outline.r",
    ),
)
def equilibrium_field_2d(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """One reconstructed 2-D equilibrium quantity, chosen with ``field=``.

    Interpretation
    --------------
    Shows one quantity of a reconstructed slice on the poloidal (R, Z) grid:
    the poloidal flux, the toroidal current density, the pressure, or a
    component of the magnetic field.  The current-density map shows where the
    reconstruction places the plasma current, the pressure map where the stored
    energy sits, and the field components the field a probe or particle would
    see at a point.

    Options
    -------
    ``field=`` chooses the quantity.  A quantity the slice does not store is
    derived from it on a private copy -- the current density and the field
    components from the flux and the 1-D profiles, the pressure by mapping the
    1-D pressure profile through the slice's own flux -- so it carries no
    information beyond the reconstruction.  ``units=``, ``style=`` and
    ``overlay=`` act as in :func:`equilibrium_field_psi`.

    Limitations
    -----------
    Every quantity is the solver's, not a local measurement.  Derived fields
    inherit the grid resolution, and differentiation amplifies noise near the
    boundary.  Current density and pressure inside the plasma follow the
    assumed profile parametrisation; outside the boundary they say nothing
    about the plasma.

    See Also
    --------
    equilibrium_field_psi : the flux map with its contour styles.
    """
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="equilibrium", quantity="psi_vacuum",
    subject="equilibrium",
    description="Vacuum poloidal flux from the PF coils alone, without plasma.",
    ids=("pf_active", "pf_passive", "wall", "spectrometer_uv", "magnetics", "equilibrium"),
    required_paths=("pf_active.time", "pf_active.coil.{i}.current.data"),
    optional_paths=("wall.description_2d.{i}.limiter.unit.{j}.outline.r",),
)
def equilibrium_field_psi_vacuum(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Vacuum poloidal flux from the PF coils alone, without plasma."""
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="pf_active", quantity="",
    subject="vacuum",
    description="The vacuum field of the coils and vessel at one instant: the "
                "flux, the poloidal field strength, the decay index, |E_phi|, "
                "the breakdown figure of merit, or the Lloyd margin, chosen "
                "with field=.",
    # spectrometer_uv earns its place: with no time= the map is drawn at the
    # breakdown onset, and that timing reads H-alpha alongside the plasma
    # current.  An adapter that loads only the declared IDSs would otherwise
    # resolve a different instant than one that hands over the whole entry.
    # barometry for the same reason, by a different route: field="lloyd_margin"
    # reads the fill pressure, and discovery offers it only when a gauge is
    # present, so an adapter loading only these IDSs would never offer it.
    ids=("pf_active", "pf_passive", "wall", "tf", "equilibrium", "magnetics",
         "spectrometer_uv", "barometry"),
    required_paths=("pf_active.time", "pf_active.coil.{i}.current.data"),
    optional_paths=(
        "pf_passive.loop.{i}.element.{j}.geometry.outline.r",
        "tf.b_field_tor_vacuum_r.data",
        "wall.description_2d.{i}.limiter.unit.{j}.outline.r",
        "barometry.gauge.{i}.pressure.data",
    ),
)
def vacuum_field(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """One quantity of the coils' and vessel's vacuum field, at one instant."""
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="machine", quantity="wall_reduction",
    subject="passive_structure",
    description="The passive wall's poloidal flux on the equilibrium region -- "
                "full, reduced, or their difference -- at one instant of the "
                "shot's PF programme (vaft #494, vfit #10).",
    ids=("pf_active", "pf_passive", "em_coupling", "wall"),
    required_paths=("pf_passive.loop.{i}.resistance",
                    "pf_active.coil.{i}.element.{j}.geometry.geometry_type",
                    "pf_active.coil.{i}.current.data",
                    "wall.description_2d.{i}.limiter.unit.{j}.outline.r"),
    optional_paths=("em_coupling.mutual_passive_passive",),
)
def passive_structure_field_wall_reduction(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Full, reduced or difference wall flux map on the equilibrium region."""
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
    """Electron temperature mapped onto the poloidal plane.

    Interpretation
    --------------
    The electron temperature profile drawn on the poloidal (R, Z) plane through
    the equilibrium flux surfaces: each grid cell takes the profile's value at
    its own flux surface.  It shows where the hot core sits relative to the
    vessel and the diagnostics' lines of sight.

    Options
    -------
    ``time_slice=`` names the equilibrium slice; the profile mapped onto it is
    the core_profiles entry stored at that slice's time.

    Limitations
    -----------
    The map holds no information beyond the 1-D profile and the equilibrium: it
    assumes the quantity is constant on each flux surface, so poloidal
    asymmetries are absent by construction.  Cells are filled wherever the
    normalized flux is below 1 and left blank elsewhere, so a private-flux
    region below an X-point, or flux that recurs near a coil, is filled
    although it is not inside the plasma.

    See Also
    --------
    electron_temperature_profile : the 1-D profile that is mapped.
    """
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
    """Electron density mapped onto the poloidal plane.

    Interpretation
    --------------
    The electron density profile drawn on the poloidal (R, Z) plane through the
    equilibrium flux surfaces: each grid cell takes the profile's value at its
    own flux surface.  It shows where the density sits relative to the vessel
    and to diagnostic and heating beam paths.

    Options
    -------
    ``time_slice=`` names the equilibrium slice; the profile mapped onto it is
    the core_profiles entry stored at that slice's time.

    Limitations
    -----------
    The map holds no information beyond the 1-D profile and the equilibrium: it
    assumes the quantity is constant on each flux surface, so poloidal
    asymmetries are absent by construction.  Cells are filled wherever the
    normalized flux is below 1 and left blank elsewhere, so a private-flux
    region below an X-point, or flux that recurs near a coil, is filled
    although it is not inside the plasma.

    See Also
    --------
    electron_density_profile : the 1-D profile that is mapped.
    """
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="mhd_linear", quantity="spectrum",
    subject="mhd_linear",
    description="Perturbed normal flux amplitude over the mapped (psi_N, m) grid: "
                "the radial label on the abscissa and the poloidal harmonic on the "
                "ordinate, which is why this map is not drawn to an equal aspect.",
    ids=("mhd_linear",),
    required_paths=(
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.n_tor",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim1",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim2",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.real",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.imaginary",
    ),
    optional_paths=(
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.energy_perturbed",
    ),
)
def mhd_linear_field_spectrum(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Perturbed normal flux amplitude over the (psi_N, m) grid."""
    return render_field_2d(model, ax=ax, show=show, **style)


@_field_renderer(
    domain="plasma_initiation", quantity="connection_length",
    subject="field_line_topology",
    description="Total connection length of every traced field line in one "
                "poloidal plane, on the rectangular grid they were launched "
                "from. The title names the toroidal angle the plane was "
                "traced at, because the IDS entry carries no toroidal "
                "coordinate and the map means nothing without it.",
    ids=("plasma_initiation",),
    required_paths=(
        "plasma_initiation.b_field_lines.{i}.grid.dim1",
        "plasma_initiation.b_field_lines.{i}.grid.dim2",
        "plasma_initiation.b_field_lines.{i}.starting_positions.r",
        "plasma_initiation.b_field_lines.{i}.starting_positions.z",
        "plasma_initiation.b_field_lines.{i}.lengths",
    ),
    optional_paths=(
        "plasma_initiation.b_field_lines.{i}.open_fraction",
        "plasma_initiation.b_field_lines.{i}.time",
        "plasma_initiation.code.parameters",
    ),
)
def field_line_topology_field_connection_length(
    model: Field2D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Connection length over one traced poloidal plane."""
    return render_field_2d(model, ax=ax, show=show, **style)
