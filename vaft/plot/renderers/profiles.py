"""Canonical ``<domain>_profile_<quantity>`` renderers.

These consume a :class:`~vaft.plot.models.Profile1D`.  The radial coordinate is
carried by the model (``Series.x`` plus ``coordinate_label``), so a single
canonical name covers every coordinate choice.  Selecting ``rho_tor_norm``,
``psi_norm``, ``r_major`` or ``r_minor`` is an adapter argument, not part of the
renderer name -- this replaces the 24 generated ``equilibrium_<coord>_<quantity>``
globals that the old ``vaft.plot.onedim`` created at import time.
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Profile1D
from ..registry import renderer
from ..presentation import presented, resolve_style
from ..style import apply_legend, axis_label, draw_series, finalize, resolve_axes, trace_labels

__all__ = [
    "impa_profile_field",
    "charge_exchange_profile_fit",
    "charge_exchange_profile_ion_temperature",
    "charge_exchange_profile_velocity_tor",
    "electron_density_profile",
    "electron_temperature_profile",
    "ion_temperature_profile",
    "thermal_pressure_profile",
    "equilibrium_profile_f",
    "equilibrium_profile_ffprime",
    "equilibrium_profile_j_tor",
    "equilibrium_profile_pprime",
    "equilibrium_profile_pressure",
    "equilibrium_profile_q",
    "coil_3d_profile_current",
    "coil_3d_spectrum_current",
    "mhd_linear_profile_b_field_perturbed",
    "mhd_linear_profile_chirikov",
    "mhd_linear_profile_displacement",
    "mhd_linear_spectrum_b_field_perturbed",
    "render_profile_1d",
    "thomson_scattering_profile_electron_density",
    "thomson_scattering_profile_fit",
    "thomson_scattering_profile_electron_temperature",
]

_DEFAULT_FIGSIZE = (6.0, 4.0)


@presented(default_figsize=_DEFAULT_FIGSIZE)
def render_profile_1d(
    model: Profile1D,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    legend: bool | None = None,
    grid: bool = True,
    uncertainty: str = "auto",
    validity: str = "show",
    format: str | None = None,
    theme: str | None = None,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Profile1D` into one axes."""
    if not isinstance(model, Profile1D):
        raise TypeError(
            f"expected a vaft.plot.models.Profile1D; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    labels, legend_title = trace_labels(model.series, panel_title=model.title)
    for series, label in zip(model.series, labels):
        options = {**style, **series.style}
        if label:
            options.setdefault("label", label)
        draw_series(axes, series, uncertainty=uncertainty, validity=validity, **options)

    for line in model.reference_lines:
        axes.axvline(
            line.x,
            **resolve_style({"color": "emphasis:medium", "linestyle": ":", "linewidth": 1.0, **line.style}),
            label=line.label or None,
        )
    axes.set_xlabel(model.coordinate_label)
    axes.set_ylabel(axis_label(model.y_label, model.y_unit))
    if model.title:
        axes.set_title(model.title)
    if model.display is not None and model.display.notation == "scientific":
        axes.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if model.x_limits is not None:
        axes.set_xlim(model.x_limits)
    if grid:
        axes.grid(True, alpha=0.3)
    apply_legend(axes, legend=legend, title=legend_title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def _profile_renderer(
    *, domain: str, subject: str, quantity: str, description: str,
    ids: tuple[str, ...], required_paths: tuple[str, ...],
    optional_paths: tuple[str, ...] = (),
):
    return renderer(
        domain=domain,
        subject=subject,
        view="profile",
        quantity=quantity,
        model=Profile1D,
        description=description,
        ids=ids,
        required_paths=required_paths,
        optional_paths=optional_paths,
    )


_SENSOR_POSITIONS = {
    "flux_loop": ("magnetics.flux_loop.{i}.position.0.r", "magnetics.flux_loop.{i}.position.0.z"),
    "b_field_probe": (
        "magnetics.b_field_pol_probe.{i}.position.r",
        "magnetics.b_field_pol_probe.{i}.position.z",
        "magnetics.b_field_pol_probe.{i}.poloidal_angle",
    ),
}


@renderer(
    domain="magnetics",
    subject="mirnov",
    view="spatial",
    quantity="phase",
    model=Profile1D,
    description="Toroidal phase of each fluctuation band around the torus at one time, with the fitted n mode lines.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.voltage.data",),
    optional_paths=(
        "magnetics.b_field_pol_probe.{i}.position.phi",
        "magnetics.b_field_pol_probe.{i}.voltage.time",
        "magnetics.time",
    ),
)
def mirnov_spatial_phase(model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any) -> tuple[Figure, Axes]:
    """Toroidal phase per fluctuation band at one time (issue #485)."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="flux_loop",
    view="spatial",
    quantity="flux",
    model=Profile1D,
    description="Flux-loop flux against sensor position at one time: height (inboard and outboard panels) or poloidal angle about the layout centre.",
    ids=("magnetics",),
    required_paths=("magnetics.flux_loop.{i}.flux.data",),
    optional_paths=_SENSOR_POSITIONS["flux_loop"] + ("magnetics.flux_loop.{i}.flux.time", "magnetics.time"),
)
def flux_loop_spatial_flux(model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any) -> tuple[Figure, Axes]:
    """Flux-loop flux against sensor position at one time (issue #486)."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="b_field_probe",
    view="spatial",
    quantity="field",
    model=Profile1D,
    description="B-probe field against sensor position at one time: height (inboard and outboard panels) or poloidal angle about the layout centre.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.field.data",),
    optional_paths=_SENSOR_POSITIONS["b_field_probe"] + ("magnetics.b_field_pol_probe.{i}.field.time", "magnetics.time"),
)
def b_field_probe_spatial_field(model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any) -> tuple[Figure, Axes]:
    """B-probe poloidal field against sensor position at one time (issue #486)."""
    return render_profile_1d(model, ax=ax, show=show, **style)


_MHD_LINEAR_EIGENFUNCTION_PATHS = (
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.n_tor",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim1",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim2",
)


#: The coordinate leaves a profile may read; r_minor is computed from the
#: two radii and never stored (vaft.plot.display.PROFILE_COORDINATES).
_EQ_COORDS = (
    "equilibrium.time_slice.{i}.profiles_1d.rho_tor_norm",
    "equilibrium.time_slice.{i}.profiles_1d.psi_norm",
    "equilibrium.time_slice.{i}.profiles_1d.phi",
    "equilibrium.time_slice.{i}.profiles_1d.r_inboard",
    "equilibrium.time_slice.{i}.profiles_1d.r_outboard",
)


@_profile_renderer(
    domain="core_profiles", quantity="bootstrap_current",
    subject="neoclassical",
    description=(
        "Bootstrap current density from each neoclassical model on one radial axis: "
        "the Sauter and Redl formulas against whatever solver result the ODS carries."
    ),
    ids=("core_profiles", "equilibrium"),
    required_paths=(
        "equilibrium.time_slice.{i}.profiles_1d.rho_tor_norm",
        "equilibrium.time_slice.{i}.profiles_1d.psi",
        "equilibrium.time_slice.{i}.profiles_1d.q",
        "equilibrium.time_slice.{i}.profiles_1d.f",
        "core_profiles.profiles_1d.{i}.electrons.temperature",
        # The electron density is required too, in either of its two spellings,
        # which the recipe's own `available` predicate checks.
    ),
    optional_paths=(
        "equilibrium.time_slice.{i}.profiles_1d.trapped_fraction",
        "core_profiles.profiles_1d.{i}.zeff",
        "core_profiles.profiles_1d.{i}.j_bootstrap",
    ),
)
def neoclassical_profile_bootstrap_current(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Bootstrap current density, one series per neoclassical model."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="equilibrium", quantity="pressure",
    subject="equilibrium",
    description="Equilibrium 1D pressure profile.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.pressure",),
    optional_paths=_EQ_COORDS,
)
def equilibrium_profile_pressure(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium 1D pressure profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="equilibrium", quantity="q",
    subject="equilibrium",
    description="Equilibrium safety-factor profile.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.q",),
    optional_paths=_EQ_COORDS,
)
def equilibrium_profile_q(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium safety-factor profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="equilibrium", quantity="j_tor",
    subject="equilibrium",
    description="Equilibrium toroidal current-density profile.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.j_tor",),
    optional_paths=_EQ_COORDS,
)
def equilibrium_profile_j_tor(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium toroidal current-density profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="equilibrium", quantity="pprime",
    subject="equilibrium",
    description="Equilibrium dp/dpsi profile.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.dpressure_dpsi",),
    optional_paths=_EQ_COORDS + ("equilibrium.time_slice.{i}.profiles_1d.pprime",),
)
def equilibrium_profile_pprime(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium dp/dpsi profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="equilibrium", quantity="f",
    subject="equilibrium",
    description="Equilibrium poloidal current function F = R*B_t.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.f",),
    optional_paths=_EQ_COORDS,
)
def equilibrium_profile_f(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium poloidal current function F = R*B_t."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="equilibrium", quantity="ffprime",
    subject="equilibrium",
    description="Equilibrium F dF/dpsi profile.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.profiles_1d.f_df_dpsi",),
    optional_paths=_EQ_COORDS + ("equilibrium.time_slice.{i}.profiles_1d.ffprime",),
)
def equilibrium_profile_ffprime(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Equilibrium F dF/dpsi profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="core_profiles", quantity="",
    subject="electron_temperature",
    description="Core electron temperature profile.",
    ids=("core_profiles",),
    required_paths=("core_profiles.profiles_1d.{i}.electrons.temperature",),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.rho_tor_norm",),
)
def electron_temperature_profile(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Core electron temperature profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="core_profiles", quantity="",
    subject="electron_density",
    description="Core electron density profile.",
    ids=("core_profiles",),
    required_paths=("core_profiles.profiles_1d.{i}.electrons.density",),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.rho_tor_norm",),
)
def electron_density_profile(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Core electron density profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="core_profiles", quantity="",
    subject="ion_temperature",
    description="Core ion temperature profile.",
    ids=("core_profiles",),
    required_paths=("core_profiles.profiles_1d.{i}.ion.{j}.temperature",),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.rho_tor_norm",),
)
def ion_temperature_profile(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Core ion temperature profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="core_profiles", quantity="",
    subject="thermal_pressure",
    description="Core total pressure profile.",
    ids=("core_profiles",),
    required_paths=("core_profiles.profiles_1d.{i}.pressure_thermal",),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.rho_tor_norm",),
)
def thermal_pressure_profile(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Core total pressure profile."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="thomson_scattering", quantity="electron_temperature",
    subject="thomson_scattering",
    description="Thomson-scattering electron temperature versus position.",
    ids=("thomson_scattering",),
    required_paths=(
        "thomson_scattering.channel.{i}.t_e.data",
        "thomson_scattering.channel.{i}.position.r",
    ),
    optional_paths=("thomson_scattering.time",),
)
def thomson_scattering_profile_electron_temperature(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Thomson-scattering electron temperature versus position."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="thomson_scattering", quantity="electron_density",
    subject="thomson_scattering",
    description="Thomson-scattering electron density versus position.",
    ids=("thomson_scattering",),
    required_paths=(
        "thomson_scattering.channel.{i}.n_e.data",
        "thomson_scattering.channel.{i}.position.r",
    ),
    optional_paths=("thomson_scattering.time",),
)
def thomson_scattering_profile_electron_density(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Thomson-scattering electron density versus position."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="thomson_scattering", quantity="fit",
    subject="thomson_scattering",
    description=(
        "Thomson T_e or n_e (field=te|ne) at one time: channels with error bars, refused "
        "channels hollow, and the fit through them on psi_N, rho_N or R, mapped through "
        "the ODS's own or a given equilibrium (issue #952)."
    ),
    ids=("thomson_scattering", "equilibrium"),
    required_paths=(
        "thomson_scattering.time",
        "thomson_scattering.channel.{i}.t_e.data",
        "thomson_scattering.channel.{i}.n_e.data",
        "thomson_scattering.channel.{i}.position.r",
    ),
    optional_paths=(
        "thomson_scattering.channel.{i}.t_e.data_error_upper",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
)
def thomson_scattering_profile_fit(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Thomson points and their fit on a flux coordinate."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="charge_exchange", quantity="fit",
    subject="charge_exchange",
    description=(
        "Charge-exchange T_i or V_phi (field=ti|vphi) at one time: channels with error "
        "bars, refused channels hollow, and the fit through them on psi_N, rho_N or R "
        "(issue #952)."
    ),
    ids=("charge_exchange", "equilibrium"),
    required_paths=(
        "charge_exchange.time",
        "charge_exchange.channel.{i}.ion.{j}.t_i.data",
        "charge_exchange.channel.{i}.position.r.data",
    ),
    optional_paths=(
        "charge_exchange.channel.{i}.ion.{j}.velocity_tor.data",
        "equilibrium.time_slice.{i}.profiles_2d.{j}.psi",
    ),
)
def charge_exchange_profile_fit(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Charge-exchange points and their fit on a flux coordinate."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="charge_exchange", quantity="ion_temperature",
    subject="charge_exchange",
    description="Charge-exchange ion temperature versus position.",
    ids=("charge_exchange",),
    required_paths=(
        "charge_exchange.channel.{i}.ion.{j}.t_i.data",
        "charge_exchange.channel.{i}.position.r.data",
    ),
    optional_paths=("charge_exchange.time",),
)
def charge_exchange_profile_ion_temperature(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Charge-exchange ion temperature versus position."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="charge_exchange", quantity="velocity_tor",
    subject="charge_exchange",
    description="Charge-exchange toroidal rotation versus position.",
    ids=("charge_exchange",),
    required_paths=(
        "charge_exchange.channel.{i}.ion.{j}.velocity_tor.data",
        "charge_exchange.channel.{i}.position.r.data",
    ),
    optional_paths=("charge_exchange.time",),
)
def charge_exchange_profile_velocity_tor(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Charge-exchange toroidal rotation versus position."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="impa",
    view="profile",
    quantity="field",
    model=Profile1D,
    description="IMPA measured field against probe radius with the 1/R toroidal-field model.",
    ids=("magnetics", "tf"),
    required_paths=("magnetics.b_field_tor_probe.{i}.voltage.data",),
    optional_paths=(
        "magnetics.b_field_tor_probe.{i}.identifier",
        "magnetics.b_field_tor_probe.{i}.position.r",
        "magnetics.b_field_pol_probe.{i}.voltage.data",
        "tf.coil.{i}.current.data",
    ),
)
def impa_profile_field(
    model: Profile1D,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """IMPA radial profile against the 1/R toroidal-field model."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="mhd_linear", quantity="displacement",
    subject="mhd_linear",
    description="DCON displacement eigenfunction against normalized flux, one trace "
                "per poloidal harmonic; amplitudes are normalized to the peak because "
                "DCON's eigenvector normalization is arbitrary.",
    ids=("mhd_linear",),
    required_paths=_MHD_LINEAR_EIGENFUNCTION_PATHS + (
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.displacement_perpendicular.real",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.displacement_perpendicular.imaginary",
    ),
    optional_paths=(
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.energy_perturbed",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.m_pol_dominant",
    ),
)
def mhd_linear_profile_displacement(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """DCON displacement eigenfunction per poloidal harmonic."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="mhd_linear", quantity="b_field_perturbed",
    subject="mhd_linear",
    description="Normal perturbed field per poloidal harmonic against normalized flux, "
                "derived from the DCON eigenfunction as i(m - nq) xi.grad(psi).",
    ids=("mhd_linear",),
    required_paths=_MHD_LINEAR_EIGENFUNCTION_PATHS + (
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.real",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.imaginary",
    ),
    optional_paths=(
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.energy_perturbed",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.m_pol_dominant",
    ),
)
def mhd_linear_profile_b_field_perturbed(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Normal perturbed field per poloidal harmonic."""
    return render_profile_1d(model, ax=ax, show=show, **style)

_GPEC_RESONANT_PATHS = (
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.n_tor",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim1",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim2",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.real",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.imaginary",
    # The per-surface geometry and chi1 the derivation needs. They have no
    # IMAS slot -- `mhd_linear` has no per-surface numeric field and `ntms`'s
    # deltaw is m^-1 where GPEC's Delta is unitless -- so the mapper records
    # them here, and the adapter reads them back.
    "mhd_linear.code.parameters",
)


@_profile_renderer(
    domain="mhd_linear", quantity="resonant_flux",
    subject="mhd_linear",
    description="Pitch-resonant flux per rational surface against normalized poloidal "
                "flux, derived from the mapped perturbed flux by the jump across each "
                "singular surface rather than read from the IDS.",
    ids=("mhd_linear",),
    required_paths=_GPEC_RESONANT_PATHS,
)
def mhd_linear_profile_resonant_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Pitch-resonant flux per rational surface."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="mhd_linear", quantity="island_width",
    subject="mhd_linear",
    description="Saturated island width per rational surface against normalized "
                "poloidal flux, in psi_N as GPEC reports it, derived from the "
                "resonant flux.",
    ids=("mhd_linear",),
    required_paths=_GPEC_RESONANT_PATHS,
)
def mhd_linear_profile_island_width(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Saturated island width per rational surface."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="mhd_linear", quantity="chirikov",
    subject="mhd_linear",
    description="Island-overlap parameter per rational surface against normalized "
                "poloidal flux, in GPEC's own surface definition, with the K = 1 "
                "criterion drawn; derived from the mapped perturbed flux.",
    ids=("mhd_linear",),
    required_paths=_GPEC_RESONANT_PATHS,
)
def mhd_linear_profile_chirikov(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Island overlap per rational surface."""
    return render_profile_1d(model, ax=ax, show=show, **style)


_MHD_LINEAR_SPECTRUM_PATHS = (
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.n_tor",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim1",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.grid.dim2",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.real",
    "mhd_linear.time_slice.{i}.toroidal_mode.{j}.plasma.b_field_perturbed.coordinate1.imaginary",
)


@renderer(
    domain="mhd_linear", subject="mhd_linear", view="spectrum",
    quantity="b_field_perturbed", model=Profile1D,
    description="Perturbed normal flux amplitude against poloidal harmonic at one "
                "flux surface; the outermost mapped surface unless psi_n names "
                "another, and the title reports the surface actually drawn.",
    ids=("mhd_linear",),
    required_paths=_MHD_LINEAR_SPECTRUM_PATHS,
    optional_paths=(
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.energy_perturbed",
    ),
)
def mhd_linear_spectrum_b_field_perturbed(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Perturbed normal flux spectrum at one flux surface."""
    return render_profile_1d(model, ax=ax, show=show, **style)


_NBI_PROFILE_PATHS = (
    "core_sources.source.{i}.identifier.index",
    "core_sources.source.{i}.profiles_1d.{j}.grid.rho_tor_norm",
)


@_profile_renderer(
    domain="core_sources", quantity="electron_heating",
    subject="nbi",
    description="Beam power density to electrons against normalized toroidal flux, "
                "from a NUBEAM result mapped into core_sources.",
    ids=("core_sources",),
    required_paths=_NBI_PROFILE_PATHS + (
        "core_sources.source.{i}.profiles_1d.{j}.electrons.energy",
    ),
    optional_paths=("core_sources.source.{i}.profiles_1d.{j}.electrons.power_inside",),
)
def nbi_profile_electron_heating(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Beam power density to electrons."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="core_sources", quantity="ion_heating",
    subject="nbi",
    description="Beam power density to ions against normalized toroidal flux, from "
                "a NUBEAM result mapped into core_sources.",
    ids=("core_sources",),
    required_paths=_NBI_PROFILE_PATHS + (
        "core_sources.source.{i}.profiles_1d.{j}.total_ion_energy",
    ),
    optional_paths=("core_sources.source.{i}.profiles_1d.{j}.total_ion_power_inside",),
)
def nbi_profile_ion_heating(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Beam power density to ions."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_profile_renderer(
    domain="core_sources", quantity="current_drive",
    subject="nbi",
    description="Beam-driven parallel current density against normalized toroidal "
                "flux, <J.B>/B0. Derived from the solver's toroidal driven current "
                "and the equilibrium geometry, assuming the driven current is "
                "field-aligned on each flux surface; it is not a direct solver "
                "output.",
    ids=("core_sources",),
    required_paths=_NBI_PROFILE_PATHS + (
        "core_sources.source.{i}.profiles_1d.{j}.j_parallel",
    ),
    optional_paths=("core_sources.source.{i}.profiles_1d.{j}.current_parallel_inside",),
)
def nbi_profile_current_drive(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Beam-driven parallel current density."""
    return render_profile_1d(model, ax=ax, show=show, **style)


_COIL_3D_EXCITATION_PATHS = (
    "coils_non_axisymmetric.coil.{i}.name",
    "coils_non_axisymmetric.coil.{i}.current.data",
    "coils_non_axisymmetric.coil.{i}.conductor.0.elements.start_points.phi",
)


@_profile_renderer(
    domain="coils_non_axisymmetric", quantity="current",
    subject="coil_3d",
    description="Sector currents of each non-axisymmetric coil set against toroidal "
                "angle: one marker per sector, because that is the whole waveform a "
                "discrete coil set carries.",
    ids=("coils_non_axisymmetric",),
    required_paths=_COIL_3D_EXCITATION_PATHS,
    optional_paths=("coils_non_axisymmetric.coil.{i}.current.time",),
)
def coil_3d_profile_current(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Non-axisymmetric coil currents against toroidal angle."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain="coils_non_axisymmetric", subject="coil_3d", view="spectrum",
    quantity="current", model=Profile1D,
    description="Toroidal mode content |C_n| of each non-axisymmetric coil set's "
                "excitation, to the last harmonic its sectors resolve.",
    ids=("coils_non_axisymmetric",),
    required_paths=_COIL_3D_EXCITATION_PATHS,
    optional_paths=("coils_non_axisymmetric.coil.{i}.current.time",),
)
def coil_3d_spectrum_current(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Toroidal mode content of a non-axisymmetric coil excitation."""
    return render_profile_1d(model, ax=ax, show=show, **style)
