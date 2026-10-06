"""Registered gyrokinetic renderers: the standardized layer of #1591.

These draw the IMAS-normalised view models built from ``gyrokinetics_local`` (one
local flux-tube calculation) and from ``core_transport`` (radial turbulent fluxes).
The native, solver-unit validation plots stay in :mod:`vaft.plot.gyrokinetics`; the
two layers are never mixed (GKDB ``binormal_wavevector_norm`` is not ``k_y rho_s``).
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import Panels, Profile1D
from ..registry import renderer
from .panels import render_panels
from .profiles import render_profile_1d

__all__ = [
    "gyrokinetics_overview",
    "gyrokinetics_profile_eigenfunction",
    "gyrokinetics_spectrum_energy_flux",
    "gyrokinetics_spectrum_frequency",
    "gyrokinetics_spectrum_growth_rate",
    "gyrokinetics_spectrum_particle_flux",
    "turbulent_transport_overview",
    "turbulent_transport_profile_energy_flux",
    "turbulent_transport_profile_particle_flux",
]

_GK = "gyrokinetics_local"


def _spectrum(quantity: str, description: str, required: tuple[str, ...], optional: tuple[str, ...] = ()):
    return renderer(
        domain=_GK, subject="gyrokinetics", view="spectrum", quantity=quantity,
        model=Profile1D, description=description, ids=(_GK,),
        required_paths=required, optional_paths=optional,
    )


@_spectrum(
    "growth_rate",
    "Linear growth rate against k_y (GKDB normalisation), every eigenmode, entries overlaid.",
    (f"{_GK}.linear.wavevector.{{i}}.binormal_wavevector_norm",
     f"{_GK}.linear.wavevector.{{i}}.eigenmode.{{j}}.growth_rate_norm"),
)
def gyrokinetics_spectrum_growth_rate(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Linear growth rate spectrum from gyrokinetics_local."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_spectrum(
    "frequency",
    "Linear real frequency against k_y with each IDS's recorded sign convention.",
    (f"{_GK}.linear.wavevector.{{i}}.binormal_wavevector_norm",
     f"{_GK}.linear.wavevector.{{i}}.eigenmode.{{j}}.frequency_norm"),
    (f"{_GK}.code.parameters",),
)
def gyrokinetics_spectrum_frequency(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Linear frequency spectrum from gyrokinetics_local."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_spectrum(
    "energy_flux",
    "ky-resolved energy flux per species (quasilinear or nonlinear, as the IDS records).",
    (f"{_GK}.non_linear.binormal_wavevector_norm",),
    (f"{_GK}.non_linear.fluxes_2d_k_x_sum.energy_phi_potential",
     f"{_GK}.non_linear.fluxes_2d_k_x_sum.energy_a_field_parallel",
     f"{_GK}.non_linear.fluxes_2d_k_x_sum.energy_b_field_parallel",
     f"{_GK}.non_linear.quasi_linear"),
)
def gyrokinetics_spectrum_energy_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """ky-resolved energy flux from gyrokinetics_local."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_spectrum(
    "particle_flux",
    "ky-resolved particle flux per species.",
    (f"{_GK}.non_linear.binormal_wavevector_norm",),
    (f"{_GK}.non_linear.fluxes_2d_k_x_sum.particles_phi_potential",
     f"{_GK}.non_linear.fluxes_2d_k_x_sum.particles_a_field_parallel",
     f"{_GK}.non_linear.fluxes_2d_k_x_sum.particles_b_field_parallel"),
)
def gyrokinetics_spectrum_particle_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """ky-resolved particle flux from gyrokinetics_local."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain=_GK, subject="gyrokinetics", view="profile", quantity="eigenfunction",
    model=Profile1D, ids=(_GK,),
    description="phi eigenfunction of the most unstable leading mode against the poloidal angle.",
    required_paths=(f"{_GK}.linear.wavevector.{{i}}.eigenmode.{{j}}.angle_pol",
                    f"{_GK}.linear.wavevector.{{i}}.eigenmode.{{j}}.fields.phi_potential_perturbed_norm"),
)
def gyrokinetics_profile_eigenfunction(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Eigenfunction magnitude, real and imaginary parts."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain=_GK, subject="gyrokinetics", view="overview", model=Panels, ids=(_GK,),
    description="One local gyrokinetic run: spectra, flux, eigenfunction and local state.",
    required_paths=(f"{_GK}.flux_surface.r_minor_norm",),
)
def gyrokinetics_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Composite of the gyrokinetic panels the run supports."""
    return render_panels(model, ax=ax, show=show, **style)


_DD_FLUX = {"energy_flux": "energy", "particle_flux": "particles"}


def _transport(quantity: str, description: str):
    return renderer(
        domain="core_transport", subject="turbulent_transport", view="profile",
        quantity=quantity, model=Profile1D, description=description, ids=("core_transport",),
        required_paths=("core_transport.model.{i}.identifier.index",
                        "core_transport.model.{i}.profiles_1d.{j}.grid_flux.rho_tor_norm",
                        f"core_transport.model.{{i}}.profiles_1d.{{j}}.electrons.{_DD_FLUX[quantity]}.flux"),
    )


@_transport("energy_flux", "Turbulent electron and ion energy flux against rho_tor_norm.")
def turbulent_transport_profile_energy_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Turbulent energy flux profiles, one pair per anomalous model."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@_transport("particle_flux", "Turbulent electron and ion particle flux against rho_tor_norm.")
def turbulent_transport_profile_particle_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Turbulent particle flux profiles, one pair per anomalous model."""
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain="core_transport", subject="turbulent_transport", view="overview", model=Panels,
    ids=("core_transport",), description="Turbulent energy and particle flux profiles side by side.",
    required_paths=("core_transport.model.{i}.identifier.index",),
)
def turbulent_transport_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Composite of the turbulent flux profiles."""
    return render_panels(model, ax=ax, show=show, **style)
