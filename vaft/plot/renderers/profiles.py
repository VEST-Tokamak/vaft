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
from ..style import axis_label, finalize, resolve_axes

__all__ = [
    "impa_profile_field",
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
    "render_profile_1d",
    "thomson_scattering_profile_electron_density",
    "thomson_scattering_profile_electron_temperature",
]

_DEFAULT_FIGSIZE = (6.0, 4.0)


def render_profile_1d(
    model: Profile1D,
    *,
    ax: Axes | None = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    grid: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`Profile1D` into one axes."""
    if not isinstance(model, Profile1D):
        raise TypeError(
            f"expected a vaft.plot.models.Profile1D; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    labelled = False
    for series in model.series:
        options = {**style, **series.style}
        if series.label:
            options.setdefault("label", series.label)
            labelled = True
        if series.yerr is not None:
            axes.errorbar(series.x, series.y, yerr=series.yerr, **options)
        else:
            axes.plot(series.x, series.y, **options)

    axes.set_xlabel(model.coordinate_label)
    axes.set_ylabel(axis_label(model.y_label, model.y_unit))
    if model.title:
        axes.set_title(model.title)
    if model.x_limits is not None:
        axes.set_xlim(model.x_limits)
    if grid:
        axes.grid(True, alpha=0.3)
    if legend and labelled:
        axes.legend(loc="best")
    return finalize(figure, axes, show=show)


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


_EQ_COORDS = (
    "equilibrium.time_slice.{i}.profiles_1d.rho_tor_norm",
    "equilibrium.time_slice.{i}.profiles_1d.psi_norm",
    "equilibrium.time_slice.{i}.profiles_1d.r_inboard",
    "equilibrium.time_slice.{i}.profiles_1d.r_outboard",
)


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
