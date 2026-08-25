"""Canonical multi-panel renderers.

These consume a :class:`~vaft.plot.models.Panels` model -- a grid of the
single-axes view models -- and return ``(Figure, ndarray[Axes])``, so composite
figures stay inside the one renderer contract instead of managing pyplot state
themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from ..models import (
    Field2D,
    GeometryLayers,
    LineSeries,
    Panels,
    Profile1D,
    Spectrogram,
)
from ..registry import renderer
from ..style import finalize, resolve_axes

__all__ = [
    "core_profiles_time_volume_averaged",
    "electromagnetics_time_current",
    "equilibrium_overview",
    "equilibrium_time_virial",
    "magnetics_overview",
    "magnetics_overview_impa",
    "render_panels",
    "soft_x_rays_overview",
    "spectrometer_uv_time_impurity",
    "summary_time_beta",
    "summary_time_energy",
    "summary_time_power_balance",
    "summary_time_voltage_consumption",
]

_DEFAULT_PANEL_HEIGHT = 2.2
_DEFAULT_PANEL_WIDTH = 6.5


def _panel_drawer(model: Any):
    # Imported lazily to keep the renderer modules free of import cycles.
    from .fields import render_field_2d
    from .geometry import render_geometry_layers
    from .lines import render_line_series
    from .profiles import render_profile_1d
    from .spectrograms import render_spectrogram

    for model_type, draw in (
        (LineSeries, render_line_series),
        (Profile1D, render_profile_1d),
        (Field2D, render_field_2d),
        (GeometryLayers, render_geometry_layers),
        (Spectrogram, render_spectrogram),
    ):
        if isinstance(model, model_type):
            return draw
    raise TypeError(f"no renderer registered for panel model {type(model).__name__}")


def render_panels(
    model: Panels,
    *,
    ax: Any = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    **style: Any,
) -> tuple[Figure, np.ndarray]:
    """Draw each model in a :class:`Panels` grid into its own axes."""
    if not isinstance(model, Panels):
        raise TypeError(
            f"expected a vaft.plot.models.Panels; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    if figsize is None:
        figsize = (
            _DEFAULT_PANEL_WIDTH * model.ncols,
            _DEFAULT_PANEL_HEIGHT * model.nrows,
        )
    figure, axes = resolve_axes(
        ax,
        nrows=model.nrows,
        ncols=model.ncols,
        figsize=figsize,
        sharex=model.share_x,
        sharey=model.share_y,
        squeeze=False,
    )
    grid = np.asarray(axes, dtype=object).reshape(model.nrows, model.ncols)

    flat = grid.ravel()
    for index, panel_model in enumerate(model.models):
        draw = _panel_drawer(panel_model)
        draw(panel_model, ax=flat[index], show=False, **style)
    for unused in flat[len(model.models):]:
        unused.set_visible(False)

    if model.suptitle:
        figure.suptitle(model.suptitle)
    return finalize(figure, grid, show=show)


def _panel_renderer(*, domain: str, view: str, quantity: str, description: str,
                    ids: tuple[str, ...], required_paths: tuple[str, ...] = (),
                    optional_paths: tuple[str, ...] = ()):
    return renderer(
        domain=domain, view=view, quantity=quantity, model=Panels,
        description=description, ids=ids, required_paths=required_paths,
        optional_paths=optional_paths,
    )


@_panel_renderer(
    domain="summary", view="time", quantity="energy",
    description="Stored-energy comparison panels across available estimates.",
    ids=("equilibrium", "core_profiles", "magnetics"),
    required_paths=("equilibrium.time",),
)
def summary_time_energy(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Stored-energy comparison panels across available estimates."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="summary", view="time", quantity="beta",
    description="Poloidal, toroidal and normalized beta panels.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time",),
)
def summary_time_beta(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Poloidal, toroidal and normalized beta panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="summary", view="time", quantity="power_balance",
    description="Ohmic input, radiated and conducted power balance panels.",
    ids=("equilibrium", "core_profiles", "summary"),
    required_paths=("equilibrium.time",),
)
def summary_time_power_balance(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Ohmic input, radiated and conducted power balance panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="summary", view="time", quantity="voltage_consumption",
    description="Loop-voltage and flux-consumption panels.",
    ids=("magnetics", "pf_active"),
    required_paths=("magnetics.time",),
)
def summary_time_voltage_consumption(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Loop-voltage and flux-consumption panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium", view="time", quantity="virial",
    description="Virial-estimate equilibrium quantities against the reconstruction.",
    ids=("equilibrium", "magnetics"),
    required_paths=("equilibrium.time",),
)
def equilibrium_time_virial(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Virial-estimate equilibrium quantities against the reconstruction."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="electromagnetics", view="time", quantity="current",
    description="Plasma, PF coil and eddy current panels on a shared time axis.",
    ids=("magnetics", "pf_active", "pf_passive"),
    required_paths=("magnetics.ip.0.data",),
)
def electromagnetics_time_current(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Plasma, PF coil and eddy current panels on a shared time axis."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="core_profiles", view="time", quantity="volume_averaged",
    description="Volume-averaged core quantity panels on a shared time axis.",
    ids=("core_profiles",),
    required_paths=("core_profiles.time",),
)
def core_profiles_time_volume_averaged(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Volume-averaged core quantity panels on a shared time axis."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="spectrometer_uv", view="time", quantity="impurity",
    description="Impurity line-intensity panels against plasma current.",
    ids=("spectrometer_uv", "magnetics"),
    required_paths=("spectrometer_uv.time",),
)
def spectrometer_uv_time_impurity(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Impurity line-intensity panels against plasma current."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="magnetics", view="overview", quantity="diagnostics",
    description="Shot diagnostic overview: current, field, flux and geometry panels.",
    ids=("magnetics", "pf_active", "tf", "equilibrium"),
    required_paths=("magnetics.ip.0.data",),
)
def magnetics_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Shot diagnostic overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="magnetics", view="overview", quantity="impa",
    description="IMPA validation overview: raw voltages, compensated Bz and the 1/R position check.",
    ids=("magnetics", "tf"),
    required_paths=("magnetics.b_field_pol_probe.{i}.voltage.data",),
)
def magnetics_overview_impa(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """IMPA validation overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium", view="overview", quantity="analysis",
    description="Equilibrium analysis overview: global quantities plus poloidal geometry.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time",),
)
def equilibrium_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Equilibrium analysis overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="soft_x_rays", view="overview", quantity="channels",
    description="Soft X-ray overview: lines of sight, signals and channel pattern.",
    ids=("soft_x_rays",),
    required_paths=("soft_x_rays.channel.{i}.power.data",),
)
def soft_x_rays_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Soft X-ray overview panels."""
    return render_panels(model, ax=ax, show=show, **style)
