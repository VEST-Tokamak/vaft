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
    "chease_overview_profile_validity",
    "chease_overview_refinement_summary",
    "core_profiles_time_volume_averaged",
    "electromagnetics_time_current",
    "equilibrium_overview",
    "equilibrium_overview_constraint_coverage",
    "equilibrium_overview_constraints",
    "equilibrium_overview_convergence",
    "equilibrium_overview_fit_quality",
    "equilibrium_overview_residuals",
    "equilibrium_overview_verification",
    "equilibrium_time_virial",
    "interferometer_overview",
    "magnetics_overview",
    "magnetics_overview_impa",
    "magnetics_overview_plasma_residual",
    "magnetics_overview_vacuum",
    "magnetics_time_limiter_current",
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
    for unused in flat[len(model.models) :]:
        unused.set_visible(False)

    if model.suptitle:
        figure.suptitle(model.suptitle)
    return finalize(figure, grid, show=show)


def _panel_renderer(
    *,
    domain: str,
    view: str,
    quantity: str,
    description: str,
    ids: tuple[str, ...],
    required_paths: tuple[str, ...] = (),
    optional_paths: tuple[str, ...] = (),
):
    return renderer(
        domain=domain,
        view=view,
        quantity=quantity,
        model=Panels,
        description=description,
        ids=ids,
        required_paths=required_paths,
        optional_paths=optional_paths,
    )


@_panel_renderer(
    domain="chease",
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
    domain="summary",
    view="time",
    quantity="beta",
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
    domain="summary",
    view="time",
    quantity="power_balance",
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
    domain="summary",
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
    view="time",
    quantity="current",
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
    domain="core_profiles",
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
    domain="magnetics", view="overview", quantity="impa",
    description="IMPA validation overview: raw voltages, compensated Bz and the 1/R position check.",
    ids=("magnetics", "tf"),
    required_paths=("magnetics.b_field_tor_probe.{i}.voltage.data",),
)
def magnetics_overview_impa(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """IMPA validation overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="magnetics",
    view="time",
    quantity="limiter_current",
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
def magnetics_time_limiter_current(
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
    description="Equilibrium analysis overview: global quantities plus poloidal geometry.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time",),
)
def equilibrium_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Equilibrium analysis overview panels."""
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
    view="overview",
    quantity="verification",
    description=(
        "EFIT verification overview: measured and reconstructed constraints "
        "beside the reconstructed poloidal-flux map."
    ),
    ids=("equilibrium",),
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
    view="overview",
    quantity="channels",
    description="Soft X-ray overview: lines of sight, signals and channel pattern.",
    ids=("soft_x_rays",),
    required_paths=("soft_x_rays.channel.{i}.power.data",),
)
def soft_x_rays_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Soft X-ray overview panels."""
    return render_panels(model, ax=ax, show=show, **style)


@_panel_renderer(
    domain="interferometer", view="overview", quantity="channels",
    description="Interferometer overview: line density history and spectrogram.",
    ids=("interferometer",),
    required_paths=("interferometer.channel.{i}.n_e_line.data",),
)
def interferometer_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, np.ndarray]:
    """Interferometer overview panels."""
    return render_panels(model, ax=ax, show=show, **style)
