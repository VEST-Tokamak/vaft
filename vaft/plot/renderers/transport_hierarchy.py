"""Registered transport-hierarchy renderers: classical, neoclassical and turbulent on one axis.

These draw the overlay view models built from ``core_transport``: one series per
model entry -- classical (VAFT's private identifier -1), neoclassical (5: NEO and
the Chang-Hinton closed form) and anomalous (6: TGLF, CGYRO) -- for one plasma
state, against ``rho_tor_norm``.  Closed forms are continuous lines; solver
results are markers at the surfaces the solver actually solved, never joined.
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
    "transport_hierarchy_overview",
    "transport_hierarchy_profile_electron_energy_flux",
    "transport_hierarchy_profile_ion_energy_flux",
]


def _render(model: Profile1D, ax: Axes | None, show: bool, style: Any) -> tuple[Figure, Axes]:
    # The symmetric-log axis the builder records in ``metadata["y_scale"]`` is applied
    # by render_profile_1d itself, so a panel member gets it too.
    return render_profile_1d(model, ax=ax, show=show, **style)


def _hierarchy(quantity: str, channel: str, description: str):
    return renderer(
        domain="core_transport", subject="transport_hierarchy", view="profile",
        quantity=quantity, model=Profile1D, description=description, ids=("core_transport",),
        required_paths=("core_transport.model.{i}.identifier.index",
                        "core_transport.model.{i}.profiles_1d.{j}.grid_flux.rho_tor_norm",
                        f"core_transport.model.{{i}}.profiles_1d.{{j}}.{channel}.energy.flux"),
    )


@_hierarchy("electron_energy_flux", "electrons",
            "Classical, neoclassical and turbulent electron energy flux on one rho_tor_norm axis.")
def transport_hierarchy_profile_electron_energy_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Electron energy flux of every transport model of one state, overlaid.

    Interpretation
    --------------
    Shows the electron energy flux, in W/m2, that each ``core_transport`` model
    gives for one plasma state, against the normalised toroidal flux coordinate:
    classical (Braginskii), neoclassical (NEO) and turbulent (TGLF, CGYRO) entries
    in one colour per physics family.  Closed-form models are continuous lines;
    solver results are markers at the surfaces solved, not joined, so an unsolved
    radius stays visibly empty.  The axis is symmetric-logarithmic: the families
    differ by several decades and the sign follows the local gradient.  It is
    read for which family dominates at each radius and by how many decades.

    Options
    -------
    ``time=`` names the state's slice; every model is matched to it by its own
    slice time within ``time_tolerance=`` (default 0.5 ms), never by position.
    Without ``time=`` the first slice of the first model found sets it.

    Limitations
    -----------
    Every curve is a model output on the state it was given, not a measurement,
    and no power balance is set against it.  The Chang-Hinton closed form has no
    electron channel and does not appear here.  Classical and Chang-Hinton fluxes
    are conductive only; NEO's and TGLF's include convection.  A model with no
    slice within the tolerance is left out and named in the model metadata.

    See Also
    --------
    transport_hierarchy_profile_ion_energy_flux : the main-ion channel of the same models.
    turbulent_transport_profile_energy_flux : the anomalous models alone, all species.
    """
    return _render(model, ax, show, style)


@_hierarchy("ion_energy_flux", "ion.{k}",
            "Classical, neoclassical and turbulent main-ion energy flux on one rho_tor_norm axis.")
def transport_hierarchy_profile_ion_energy_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Main-ion energy flux of every transport model of one state, overlaid.

    Interpretation
    --------------
    Shows the energy flux of the main ion (the lowest-charge ion species, H+ on
    VEST), in W/m2, from each ``core_transport`` model of one plasma state, against
    the normalised toroidal flux coordinate: classical (Braginskii), neoclassical
    closed form (Chang-Hinton) and drift-kinetic (NEO), and turbulent (TGLF,
    CGYRO), one colour per family, closed forms as lines and solver results as
    unjoined markers, on a symmetric-logarithmic axis.  It is read for the radius
    where turbulence overtakes collisional transport, and for how far the closed
    form departs from the drift-kinetic solution at low aspect ratio.

    Options
    -------
    ``time=`` and ``time_tolerance=`` as for the electron channel.

    Limitations
    -----------
    The main ion only, so every family is compared on one species: the impurity
    heat flux NEO and TGLF also carry (classical and Chang-Hinton do not evaluate
    it) is not drawn, and the ion total is therefore not shown.  Classical and
    Chang-Hinton fluxes are conductive only; NEO's and TGLF's include convection.

    See Also
    --------
    transport_hierarchy_profile_electron_energy_flux : the electron channel.
    turbulent_transport_profile_energy_flux : the anomalous models alone, ions summed.
    """
    return _render(model, ax, show, style)


@renderer(
    domain="core_transport", subject="transport_hierarchy", view="overview", model=Panels,
    ids=("core_transport",),
    description="Electron and main-ion energy flux of classical, neoclassical and turbulent models.",
    required_paths=("core_transport.model.{i}.identifier.index",),
)
def transport_hierarchy_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Composite of the electron and main-ion transport-hierarchy profiles.

    Interpretation
    --------------
    The electron and main-ion energy-flux overlays side by side on one radial
    axis, so the classical, neoclassical and turbulent predictions for one plasma
    state can be read in both channels at once: which family carries each channel
    at each radius, and whether the radius where turbulence takes over is the same
    for electrons and ions.  Styles are those of the member profiles.

    Options
    -------
    ``time=`` and ``time_tolerance=`` are passed to both members.

    Limitations
    -----------
    A channel no model carries drops its panel.  The members carry their own
    limitations: model outputs, not measurements; main ion only; conductive-only
    closed forms against solver totals.

    See Also
    --------
    transport_hierarchy_profile_electron_energy_flux : the electron panel.
    transport_hierarchy_profile_ion_energy_flux : the main-ion panel.
    """
    return render_panels(model, ax=ax, show=show, **style)
