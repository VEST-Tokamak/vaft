"""Canonical multi-panel renderers.

These consume a :class:`~vaft.plot.models.Panels` model -- a grid of the
single-axes view models -- and return ``(Figure, ndarray[Axes])``, so composite
figures stay inside the one renderer contract instead of managing pyplot state
themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..models import (
    Field2D,
    GeometryLayers,
    LineSeries,
    Panels,
    Profile1D,
    Spectrogram,
    TextPanel,
)
from ..registry import renderer
from ..style import finalize, resolve_axes

__all__ = [
    "passive_structure_overview_wall_time",
    "chease_overview_profile_validity",
    "chease_overview_refinement_summary",
    "core_profiles_time_volume_averaged",
    "current_overview",
    "diagnostics_overview",
    "equilibrium_overview",
    "equilibrium_overview_constraint_coverage",
    "equilibrium_overview_constraints",
    "equilibrium_overview_convergence",
    "equilibrium_overview_fit_quality",
    "equilibrium_overview_histories",
    "equilibrium_overview_residuals",
    "equilibrium_overview_verification",
    "equilibrium_time_beta",
    "equilibrium_time_virial",
    "impa_overview",
    "interferometer_overview",
    "limiter_current_time",
    "magnetics_overview",
    "magnetics_overview_plasma_residual",
    "magnetics_overview_vacuum",
    "render_panels",
    "soft_x_rays_overview",
    "spectrometer_uv_time_impurity",
    "summary_time_energy",
    "summary_time_power_balance",
    "summary_time_voltage_consumption",
]

_DEFAULT_PANEL_HEIGHT = 2.2
_DEFAULT_PANEL_WIDTH = 6.5


def _mark_if_invalid(axis: Any, panel_model: Any, style: dict) -> None:
    """Mark a panel whose every trace the source flagged invalid.

    With per-panel legends off in a subplots layout, the grey dashed trace
    alone can be missed; the panel background and a corner note keep the
    display policy's "never silently hidden" promise (issue #256, #260).
    """
    if not isinstance(panel_model, LineSeries) or style.get("validity", "show") != "show":
        return
    if panel_model.series and all(s.is_invalid_channel for s in panel_model.series):
        axis.set_facecolor("0.94")
        axis.text(0.99, 0.95, "invalid", transform=axis.transAxes, ha="right",
                  va="top", color="0.5", fontsize="small")


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
        (TextPanel, _draw_text_panel),
    ):
        if isinstance(model, model_type):
            return draw
    raise TypeError(f"no renderer registered for panel model {type(model).__name__}")


def _draw_text_panel(
    model: TextPanel, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Place a :class:`TextPanel`'s lines in an axes with no frame."""
    axis = ax if ax is not None else plt.subplots()[1]
    axis.set_axis_off()
    axis.text(
        0.02, 0.98, "\n".join(model.lines), transform=axis.transAxes,
        ha="left", va="top", family="monospace", fontsize="small", linespacing=1.4,
    )
    if model.title:
        axis.set_title(model.title)
    return axis.figure, axis


def render_panels(
    model: Panels,
    *,
    ax: Any = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    **style: Any,
) -> tuple[Figure, np.ndarray]:
    """Draw each model in a :class:`Panels` grid into its own axes.

    A caller-supplied ``ax=`` is the grid: exactly one axes per panel, filled
    in order.  Those axes are the caller's to configure, so ``model.share_x``
    and the tight layout apply only to a figure this renderer creates itself.
    """
    if not isinstance(model, Panels):
        raise TypeError(
            f"expected a vaft.plot.models.Panels; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    occupied = len(model.models) + len(model.placeholders)
    if ax is not None:
        # Caller-supplied axes are authoritative and *are* the grid: exactly one
        # axes per panel, filled in order, whatever grid this model would have
        # chosen for itself (issue #260 section 8).  Nothing is truncated,
        # recycled or created.
        supplied = np.asarray(ax, dtype=object)
        if supplied.ndim == 0:
            supplied = supplied.reshape(1)
        if supplied.size != occupied:
            raise ValueError(
                f"this layout draws {occupied} panels but received {supplied.size} axes"
            )
        flat = supplied.ravel()
        for item in flat:
            if not isinstance(item, Axes):
                raise TypeError(
                    f"ax entries must be matplotlib Axes; got {type(item).__name__}"
                )
        figure = flat[0].figure
        grid = supplied if supplied.ndim == 2 else supplied.reshape(-1, 1)
    elif model.spans is not None:
        # Unequal panels: one axes per span on a gridspec.  The result is the
        # slot-ordered axes, one dimension, since no rectangular grid exists.
        if figsize is None:
            figsize = (
                _DEFAULT_PANEL_WIDTH * model.ncols,
                _DEFAULT_PANEL_HEIGHT * model.nrows,
            )
        figure = resolve_axes(None, figsize=figsize)[0]
        for axis in list(figure.axes):
            axis.remove()
        gridspec = figure.add_gridspec(model.nrows, model.ncols)
        flat = np.array(
            [figure.add_subplot(gridspec[r:r + rs, c:c + cs]) for r, c, rs, cs in model.spans],
            dtype=object,
        )
        grid = flat
    else:
        if figsize is None:
            figsize = (
                _DEFAULT_PANEL_WIDTH * model.ncols,
                _DEFAULT_PANEL_HEIGHT * model.nrows,
            )
        figure, axes = resolve_axes(
            None,
            nrows=model.nrows,
            ncols=model.ncols,
            figsize=figsize,
            sharex=model.share_x,
            sharey=model.share_y,
            squeeze=False,
        )
        grid = np.asarray(axes, dtype=object).reshape(model.nrows, model.ncols)
        flat = grid.ravel()
    placeholders = dict(model.placeholders)
    slots = [slot for slot in range(flat.size) if slot not in placeholders]
    if model.spans is not None and ax is None:
        _mark_spanned_slots_for_layout(flat)
    for index, panel_model in enumerate(model.models):
        axis = flat[slots[index]]
        member_style = dict(model.member_styles[index]) if model.member_styles else {}
        draw = _panel_drawer(panel_model)
        draw(panel_model, ax=axis, show=False, **{**member_style, **style})
        _mark_if_invalid(axis, panel_model, {**member_style, **style})
    for slot, text in placeholders.items():
        axis = flat[slot]
        axis.set_axis_off()
        axis.text(0.5, 0.5, text, transform=axis.transAxes, ha="center", va="center", color="0.4")
    for slot in slots[len(model.models) :]:
        flat[slot].set_visible(False)

    if ax is None and model.spans is None and model.share_x and model.nrows > 1:
        # Panels sharing a time base share its label: only the lowest drawn
        # panel of each column keeps the x label and tick labels.
        for column in range(model.ncols):
            column_axes = [grid[row, column] for row in range(model.nrows) if grid[row, column].get_visible()]
            for axis in column_axes[:-1]:
                axis.tick_params(labelbottom=False)
                axis.set_xlabel("")
            if column_axes:
                # A shared-x grid labels ticks on its structural last row only;
                # a column that ends earlier needs them switched on by hand.
                column_axes[-1].tick_params(labelbottom=True)
    if model.suptitle:
        figure.suptitle(model.suptitle)
    # A figure the caller owns keeps the caller's layout: tight_layout is
    # applied only to a figure this renderer created (issue #260 section 8;
    # a figure that redraws its panels must not be re-laid-out each time).
    return finalize(figure, grid, show=show, tight_layout=ax is None)


def _mark_spanned_slots_for_layout(flat: np.ndarray) -> None:
    """Nothing to mark today; tight_layout handles gridspec spans itself."""


def slice_grid_axes(figure: Any, grid: Any, model: Panels, *, top: int = 0, colorbar_slot: int | None = None):
    """Axes for ``model``'s slots on ``grid`` (a gridspec), rows offset by ``top``.

    Shared by the panel renderer and the slice-navigation figure, so an
    overview drawn on its own and inside the navigator have the same shape.
    Returns ``(axes, colorbar_axes)``; ``colorbar_slot`` reserves a narrow
    cell beside that slot's panel for a colorbar the caller redraws into.
    """
    spans = model.spans or tuple(
        (slot // model.ncols, slot % model.ncols, 1, 1)
        for slot in range(len(model.models) + len(model.placeholders))
    )
    axes = []
    colorbar_axes = None
    for slot, (r, c, rs, cs) in enumerate(spans):
        cell = grid[top + r:top + r + rs, c:c + cs]
        if slot == colorbar_slot:
            sub = cell.subgridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.06)
            axes.append(figure.add_subplot(sub[0, 0]))
            colorbar_axes = figure.add_subplot(sub[0, 1])
        else:
            axes.append(figure.add_subplot(cell))
    return np.array(axes, dtype=object), colorbar_axes


def _panel_renderer(
    *,
    domain: str,
    subject: str,
    view: str,
    quantity: str,
    description: str,
    ids: tuple[str, ...],
    required_paths: tuple[str, ...] = (),
    optional_paths: tuple[str, ...] = (),
):
    return renderer(
        domain=domain,
        subject=subject,
        view=view,
        quantity=quantity,
        model=Panels,
        description=description,
        ids=ids,
        required_paths=required_paths,
        optional_paths=optional_paths,
    )


@_panel_renderer(
    domain="machine",
    subject="passive_structure",
    view="overview",
    quantity="wall_time",
    description=(
        "Decay-time spectrum of the passive wall's segment-wise eigenmodes: "
        "one series per conductor segment, slowest mode first, with the "
        "whole wall's global spectrum for reference (vaft #473)."
    ),
    ids=("pf_active", "pf_passive", "em_coupling"),
    required_paths=("pf_passive.loop.{i}.resistance",
                    "pf_active.coil.{i}.element.{j}.geometry.geometry_type"),
    optional_paths=("em_coupling.mutual_passive_passive",),
)
def passive_structure_overview_wall_time(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Wall-mode decay times per segment."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="chease",
    subject="chease",
    view="overview",
    quantity="refinement_summary",
    description=(
        "How far CHEASE moved each profile and the boundary from the EFIT "
        "equilibrium it refined, slice by slice: profile and boundary RMS "
        "change, flux-normalization shift, and plasma-current self-consistency."
    ),
    ids=("equilibrium",),
    # `equilibrium.code.name` is not exclusive to CHEASE -- vaft.data.vfit
    # also sets it (to "VFIT") -- and the registry's availability check only
    # tests path *presence*, not value, so a shared path would offer this
    # plot for any equilibrium reconstruction that happens to set code.name.
    # `code.library.0.name` is written only by `generate_chease_ods.py`.
    required_paths=("equilibrium.code.library.0.name",),
)
def chease_overview_refinement_summary(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """CHEASE refinement-vs-input comparison, slice by slice."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="chease",
    subject="chease",
    view="overview",
    quantity="profile_validity",
    description=(
        "q0/q95, q-monotonicity and pressure positivity of the refined "
        "equilibrium: a converged CHEASE solution that is not physically "
        "sound flagged at a glance."
    ),
    ids=("equilibrium",),
    # Same reasoning as the refinement summary.
    required_paths=("equilibrium.code.library.0.name",),
)
def chease_overview_profile_validity(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Physical validity of the refined equilibrium's profiles."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="summary",
    subject="summary",
    view="time",
    quantity="energy",
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
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="beta",
    description="Poloidal, toroidal and normalized beta panels.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.global_quantities.beta_pol",),
)
def equilibrium_time_beta(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Poloidal, toroidal and normalized beta panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="summary",
    subject="summary",
    view="time",
    quantity="power_balance",
    description="Ohmic input, radiated and conducted power balance panels.",
    ids=("equilibrium", "core_profiles", "summary"),
    required_paths=(
        "equilibrium.time_slice.{i}.global_quantities.ip",
        "equilibrium.time_slice.{i}.global_quantities.volume",
    ),
)
def summary_time_power_balance(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Ohmic input, radiated and conducted power balance panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="summary",
    subject="summary",
    view="time",
    quantity="voltage_consumption",
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
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="virial",
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
    domain="electromagnetics",
    subject="current",
    view="overview",
    quantity="",
    description="Plasma, PF coil and eddy current panels on a shared time axis.",
    ids=("magnetics", "pf_active", "pf_passive"),
    required_paths=("magnetics.ip.0.data",),
)
def current_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Plasma, PF coil and eddy current panels on a shared time axis."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="core_profiles",
    subject="core_profiles",
    view="time",
    quantity="volume_averaged",
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
    domain="spectrometer_uv",
    subject="spectrometer_uv",
    view="time",
    quantity="impurity",
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
    domain="magnetics",
    subject="diagnostics",
    view="overview",
    quantity="",
    description=(
        "Time histories of every diagnostic subject, one panel each, in a fixed "
        "grid: a diagnostic absent from the input is a labelled empty panel, so "
        "the figure has the same shape on every shot. Channels the source "
        "flagged invalid are excluded by default."
    ),
    ids=("magnetics", "interferometer", "thomson_scattering", "charge_exchange",
         "spectrometer_uv", "barometry", "soft_x_rays"),
    required_paths=(),
)
def diagnostics_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Fixed-shape time overview across the diagnostic subjects."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="magnetics",
    subject="magnetics",
    view="overview",
    quantity="diagnostics",
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
    domain="magnetics", view="overview", quantity="",
    subject="impa",
    description="IMPA validation overview: raw voltages, compensated Bz and the 1/R position check.",
    ids=("magnetics", "tf"),
    required_paths=("magnetics.b_field_tor_probe.{i}.voltage.data",),
)
def impa_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """IMPA validation overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="magnetics",
    subject="limiter_current",
    view="time",
    quantity="",
    description="Lower-corner, upper-corner and midplane limiter currents in three panels.",
    ids=("magnetics",),
    required_paths=(
        "magnetics.shunt.{i}.voltage.data",
        "magnetics.shunt.{i}.resistance",
    ),
    optional_paths=(
        "magnetics.shunt.{i}.voltage.time",
        "magnetics.shunt.{i}.name",
        "magnetics.shunt.{i}.identifier",
    ),
)
def limiter_current_time(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Lower-corner, upper-corner and midplane limiter-current histories."""
    return render_panels(model, ax=ax, show=show, **style)


_VACUUM_IDS = ("magnetics", "pf_active", "pf_passive")
#: The reconstructed vacuum current system is what these two validate, so the
#: coil and passive-loop currents are the hard requirement; which magnetic
#: observables are available varies by shot and is resolved by the adapter.
_VACUUM_REQUIRED = (
    "pf_active.coil.{i}.current.data",
    "pf_passive.loop.{i}.current",
)
_VACUUM_OPTIONAL = (
    "magnetics.b_field_pol_probe.{i}.field.data",
    "magnetics.flux_loop.{i}.flux.data",
)


@_panel_renderer(
    domain="magnetics",
    subject="magnetics",
    view="overview",
    quantity="vacuum",
    description=(
        "Measured, coil-only and coil+eddy synthetic magnetics per channel: "
        "the forward-modeled validation of an eddy-current reconstruction."
    ),
    ids=_VACUUM_IDS,
    required_paths=_VACUUM_REQUIRED,
    optional_paths=_VACUUM_OPTIONAL,
)
def magnetics_overview_vacuum(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Measured against coil-only and coil+eddy synthetic vacuum magnetics."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="magnetics",
    subject="magnetics",
    view="overview",
    quantity="plasma_residual",
    description=(
        "Residual of measured minus coil+eddy synthetic magnetics against the "
        "pre-plasma noise band, with the plasma-current and residual onsets."
    ),
    ids=_VACUUM_IDS,
    required_paths=_VACUUM_REQUIRED + ("magnetics.ip.{i}.data",),
    optional_paths=_VACUUM_OPTIONAL,
)
def magnetics_overview_plasma_residual(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Plasma-signal residual left by the coil+eddy synthetic vacuum response."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium", view="overview", quantity="analysis",
    subject="equilibrium",
    description=(
        "One equilibrium slice from one figure: poloidal flux with the LCFS "
        "and axis, pressure and q profiles, and the slice's global quantities."
    ),
    ids=("equilibrium", "wall"),
    required_paths=("equilibrium.time",),
)
def equilibrium_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Static summary of one representative equilibrium slice (issue #261)."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium", view="overview", quantity="histories",
    subject="equilibrium",
    description="Equilibrium global quantities against time: Ip, beta_p, li, q95.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time",),
)
def equilibrium_overview_histories(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """The four equilibrium time histories the slice summary replaced."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium", subject="equilibrium", view="overview", quantity="profiles",
    description="Principal 1-D equilibrium profiles: pressure, toroidal current "
                "density and safety factor.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.psi",),
)
def equilibrium_overview_profiles(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Pressure, current density and safety factor in one figure."""
    return render_panels(model, ax=ax, show=show, **style)


_CONSTRAINT_IDS = ("equilibrium",)
_CONSTRAINT_SUBMITTED = (
    "equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.measured",
)
_CONSTRAINT_OPTIONAL = (
    "equilibrium.time_slice.{i}.constraints.flux_loop.{j}.measured",
    "equilibrium.time_slice.{i}.constraints.pf_current.{j}.measured",
    "equilibrium.time_slice.{i}.constraints.ip.measured",
    "equilibrium.time_slice.{i}.constraints.diamagnetic_flux.measured",
    "equilibrium.time_slice.{i}.convergence.grad_shafranov_deviation_value",
)


@_panel_renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="overview",
    quantity="constraints",
    description=(
        "Magnetic constraints as submitted to EFIT, per family, with enabled, "
        "disabled and missing channels distinguished."
    ),
    ids=_CONSTRAINT_IDS,
    required_paths=_CONSTRAINT_SUBMITTED,
    optional_paths=_CONSTRAINT_OPTIONAL,
)
def equilibrium_overview_constraints(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Magnetic constraints submitted to EFIT, by family and channel state."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="overview",
    quantity="constraint_coverage",
    description=(
        "Enabled, disabled and missing constraint channels per family across "
        "the reconstructed time slices."
    ),
    ids=_CONSTRAINT_IDS,
    required_paths=_CONSTRAINT_SUBMITTED,
    optional_paths=_CONSTRAINT_OPTIONAL,
)
def equilibrium_overview_constraint_coverage(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Constraint channel coverage across the reconstructed time slices."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="overview",
    quantity="residuals",
    description=(
        "Measured-minus-reconstructed residuals by diagnostic family, with the "
        "solver's convergence context beside them rather than in place of them."
    ),
    ids=_CONSTRAINT_IDS,
    required_paths=(
        "equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.reconstructed",
    ),
    optional_paths=_CONSTRAINT_SUBMITTED + _CONSTRAINT_OPTIONAL,
)
def equilibrium_overview_residuals(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """EFIT reconstruction residuals by diagnostic family."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="overview",
    quantity="fit_quality",
    description=(
        "EFIT goodness of fit: reduced chi-square against the degrees of freedom "
        "EFIT itself reports, which diagnostic family carries the chi-square, and "
        "residuals normalized by the uncertainty EFIT was given."
    ),
    ids=_CONSTRAINT_IDS,
    required_paths=(
        "equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.chi_squared",
    ),
    optional_paths=_CONSTRAINT_SUBMITTED + _CONSTRAINT_OPTIONAL,
)
def equilibrium_overview_fit_quality(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """EFIT goodness of fit against the uncertainties it was given."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="overview",
    quantity="convergence",
    description=(
        "EFIT numerical convergence: final Grad-Shafranov error against the "
        "requested tolerance, iteration count against its cap, the error history "
        "where one was written, and EFIT's outputs checked against each other."
    ),
    ids=_CONSTRAINT_IDS,
    # Only a reconstructed slice is required. The convergence node is written by
    # the m-file mapper and the verdict by the a-file, so requiring either would
    # fail the whole EFIT stage over one absent optional artifact when the figure
    # can still draw iterations, self-consistency and what it does have. The
    # builder raises when *nothing* is available, which is the real failure.
    required_paths=("equilibrium.time_slice.{i}.time",),
    optional_paths=(
        "equilibrium.time_slice.{i}.convergence.grad_shafranov_deviation_value",
        "equilibrium.time_slice.{i}.convergence.iterations_n",
    )
    + _CONSTRAINT_SUBMITTED
    + _CONSTRAINT_OPTIONAL,
)
def equilibrium_overview_convergence(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """EFIT numerical-convergence and self-consistency overview."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="overview",
    quantity="verification",
    description=(
        "EFIT verification overview: measured and reconstructed constraints "
        "beside the reconstructed poloidal-flux map."
    ),
    ids=("equilibrium", "wall"),
    required_paths=(
        "equilibrium.time_slice.{i}.profiles_2d.0.psi",
        "equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.measured",
    ),
    optional_paths=(
        "equilibrium.time_slice.{i}.constraints.flux_loop.{j}.measured",
        "equilibrium.time_slice.{i}.constraints.pf_current.{j}.measured",
        "equilibrium.time_slice.{i}.constraints.ip.measured",
        "equilibrium.time_slice.{i}.constraints.diamagnetic_flux.measured",
    ),
)
def equilibrium_overview_verification(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Measured-versus-reconstructed EFIT constraints and poloidal flux."""
    return render_panels(
        model,
        ax=ax,
        show=show,
        figsize=style.pop("figsize", (13.0, 10.0)),
        **style,
    )


@_panel_renderer(
    domain="soft_x_rays",
    subject="soft_x_rays",
    view="overview",
    quantity="channels",
    description="Soft X-ray overview: lines of sight, signals and channel pattern.",
    ids=("soft_x_rays", "wall"),
    required_paths=("soft_x_rays.channel.{i}.brightness.data",),
    optional_paths=("soft_x_rays.channel.{i}.power.data",),
)
def soft_x_rays_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Soft X-ray overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="interferometer", view="overview", quantity="channels",
    subject="interferometer",
    description="Interferometer overview: line density history and spectrogram.",
    ids=("interferometer",),
    required_paths=("interferometer.channel.{i}.n_e_line.data",),
)
def interferometer_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Interferometer overview panels."""
    return render_panels(model, ax=ax, show=show, **style)
