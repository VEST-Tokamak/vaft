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
    """Linear growth rate spectrum from gyrokinetics_local.

    Interpretation
    --------------
    Shows the linear growth rate of each eigenmode of one local flux-tube
    calculation against the binormal wavenumber, in the normalisation the IDS
    records (GKDB: lengths by R0, velocities by v_th,ref = sqrt(2 Te / mD), so
    the axes read k_y rho_ref and gamma R0 / v_th,ref).  It is read for which
    k_y range is unstable, where the growth rate peaks, and how the spectrum
    of one run compares with another's at the same surface: a scan merged
    from single-k_y runs or a quasilinear model's spectrum drawn over a
    gyrokinetic one.  The leading eigenmode of each entry is a solid line;
    further eigenmodes, where the solver writes them, are dotted and
    numbered.  A k_y at which a mode is absent is a gap, never a zero.  When
    several entries are overlaid, the title says "unmatched" and names what
    differs between them -- surface, q, shear, species charges, field model,
    frequency sign convention or normalisation -- read from each IDS's own
    metadata, so an overlay that is not one comparison is marked, not
    silently drawn.

    Options
    -------
    ``include_unconverged=`` (default False) also draws initial-value
    eigenmodes that reached no growth-rate tolerance; by default they are left
    out, because a run stopped at its time limit holds a last-step value, not
    an eigenvalue.

    Limitations
    -----------
    The values are in the IDS's GKDB normalisation, not the solver's native
    units (CGYRO's c_s / a, say); a native-unit spectrum is a different
    quantity and lives in the validation layer.  Unconverged modes are not
    drawn, so a gap may be an unfinished run rather than a stable k_y.  The
    mismatch mark compares metadata, not physics: two entries that agree on
    every listed key can still differ in resolution or collision model.  The
    plot infers no instability branch (ITG, TEM, ETG, KBM, MTM) from the
    curve.

    See Also
    --------
    gyrokinetics_spectrum_frequency : the real frequency of the same modes.
    gyrokinetics_overview : the spectra with the run's flux, eigenfunction and local state.
    """
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
    """Linear frequency spectrum from gyrokinetics_local.

    Interpretation
    --------------
    Shows the real frequency of each eigenmode against the binormal
    wavenumber, in the IDS's GKDB normalisation (k_y rho_ref against omega R0
    / v_th,ref), with the same leading-mode solid, further-mode dotted layout
    as the growth-rate spectrum.  The sign of the frequency tells the
    propagation direction, so the vertical label repeats the convention the
    IDS records in its code.parameters: "ion dia. < 0" when every entry
    stores the ion-diamagnetic-negative convention, "sign convention not
    recorded" when none does, and "sign conventions differ" when the entries
    disagree.  A change of sign along k_y marks a change of the dominant
    branch; the growth-rate spectrum says which of them is unstable.  Overlaid
    entries carry the same "unmatched" title note as the growth-rate spectrum.

    Options
    -------
    ``include_unconverged=`` (default False) also draws initial-value
    eigenmodes that reached no growth-rate tolerance.

    Limitations
    -----------
    The sign is drawn as stored and never flipped: entries recorded under
    different conventions are labelled as differing, not reconciled, so their
    curves must not be compared by sign.  The frequency of an unconverged
    initial-value run is its last-step value and is left out by default.  A
    propagation direction does not by itself identify the instability
    branch.

    See Also
    --------
    gyrokinetics_spectrum_growth_rate : the growth rate of the same modes.
    """
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
    """ky-resolved energy flux from gyrokinetics_local.

    Interpretation
    --------------
    Shows, for each species, the energy flux carried by each binormal
    wavenumber -- the k_x-summed ``fluxes_2d_k_x_sum`` of the IDS, summed over
    the field components it holds (phi, A_parallel, B_parallel) -- against k_y
    rho_ref, in the IDS's GKDB reference flux.  The title says whether the IDS
    records a quasilinear estimate (a TGLF saturation rule), a nonlinear
    simulation, or a mix of both across the entries.  It is read for which
    scales carry the transport (the k_y of the peak), how the flux is shared
    between electrons and ion species, and how two models' spectra differ at
    the same surface; the "unmatched" title note names what keeps two entries
    from being one comparison.  The horizontal range stops at 1.1 times the
    last k_y at which any species still carries one per cent of its peak
    flux, so a grid that runs into the electron scales does not flatten the
    ion-scale structure; the series themselves keep every point.

    Limitations
    -----------
    A quasilinear flux is the saturation model's estimate, not a prediction
    of the gyrokinetic equations, and its amplitude depends on the rule;
    the overlay does not say which rule is closer to a nonlinear result.  The
    flux is in the IDS's reference units, not W/m2, and the field components
    are summed, so an electromagnetic contribution cannot be separated here.
    The k_x structure is summed away.  The cut of the horizontal range is a
    display choice: a species whose flux never rises above one per cent of
    another's peak is not what sets it.

    See Also
    --------
    gyrokinetics_spectrum_particle_flux : the particle flux per k_y.
    turbulent_transport_profile_energy_flux : the radial energy flux the models give core_transport.
    """
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
    """ky-resolved particle flux from gyrokinetics_local.

    Interpretation
    --------------
    Shows, for each species, the particle flux carried by each binormal
    wavenumber -- the k_x-summed ``fluxes_2d_k_x_sum`` of the IDS, summed over
    the field components it holds -- against k_y rho_ref, in the IDS's GKDB
    reference flux, with the quasilinear or nonlinear origin in the title.  It
    is read for the sign of the particle flux per scale (an inward, pinch-like
    contribution at some k_y is as physical as an outward one), for the
    scales that carry it, and for the species balance that ambipolarity
    imposes on the sum.  Overlaid entries carry the "unmatched" title note of
    the other gyrokinetic spectra.  The horizontal range stops where every
    species' flux has fallen below one per cent of its peak magnitude; the
    series keep every point.

    Limitations
    -----------
    A quasilinear flux is the saturation model's estimate; its amplitude is
    the rule's.  The values are reference-normalised, not m-2 s-1, and the
    field components are summed.  The k_x structure is summed away.  The
    display cut of the horizontal range follows the peak magnitude, so a
    small flux of one species is not what sets it.

    See Also
    --------
    gyrokinetics_spectrum_energy_flux : the energy flux per k_y.
    turbulent_transport_profile_particle_flux : the radial particle flux the models give core_transport.
    """
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
    """Eigenfunction magnitude, real and imaginary parts.

    Interpretation
    --------------
    Shows the perturbed electrostatic potential of the leading eigenmode at
    the most unstable converged k_y of the run against the DD poloidal angle,
    in units of pi: its magnitude (emphasised), real and imaginary parts.  The
    angle runs clockwise from the low-field-side midplane and extends beyond
    one poloidal turn along the field line (the ballooning angle), so the
    width of the envelope says how far the mode extends along the line: a
    narrow envelope centred on zero is a ballooning mode localised on the
    outboard side; a wide or lobed one reaches the inboard side.  The
    amplitude is normalised as the eigenmode's code.parameters states (for
    the CGYRO mapping, phi divided by its value at angle zero, final time),
    and the title names the k_y the eigenfunction belongs to.

    Limitations
    -----------
    Only the first eigenmode of the k_y with the largest converged growth
    rate is drawn: further eigenmodes, other k_y, and the parallel vector
    potential and parallel magnetic field eigenfunctions are not.  The phase
    convention is the mapping's, so real and imaginary parts are comparable
    between runs only when both state the same normalisation; the magnitude
    is convention-free.  The extended-angle range is what the solver stored,
    so an asymmetric range is the run's, not a feature of the mode.

    See Also
    --------
    gyrokinetics_spectrum_growth_rate : the growth rate that picked this k_y.
    """
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain=_GK, subject="gyrokinetics", view="overview", model=Panels, ids=(_GK,),
    description="One local gyrokinetic run: spectra, flux, eigenfunction and local state.",
    required_paths=(f"{_GK}.flux_surface.r_minor_norm",),
)
def gyrokinetics_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Composite of the gyrokinetic panels the run supports.

    Interpretation
    --------------
    One figure for one local run: the growth-rate and frequency spectra,
    the k_y-resolved energy flux and the leading eigenfunction, each present
    only when the IDS carries it, beside a text column with the local state
    the run was given -- r/R0, q, shear, elongation, the reference beta,
    each species' normalised density and temperature gradients, the code and
    its commit, and the locality verdict when one was recorded.  It is read
    as the first look at a run: whether it is unstable, at which scale, in
    which direction the modes propagate, how much flux the model attributes
    to them, and what plasma state produced all that.  The suptitle names the
    code, its saturation rule and field model, and the surface.

    Options
    -------
    ``include_unconverged=`` (default False) also draws, in the two linear
    spectra, the initial-value eigenmodes that reached no growth-rate
    tolerance.  ``title=`` replaces the suptitle.

    Limitations
    -----------
    A missing quantity drops its panel without a note, so a figure with fewer
    panels says that the IDS lacks them, not that they were computed and
    empty.  The members carry the readings and limitations of their own plots
    (GKDB normalisation, converged modes only, leading eigenmode only, the
    flux panel's display range).  One run per figure: comparisons are made by
    overlaying entries on the individual spectra.

    See Also
    --------
    gyrokinetics_spectrum_growth_rate : the growth-rate panel, with entries overlaid.
    gyrokinetics_profile_eigenfunction : the eigenfunction panel.
    """
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
    """Turbulent energy flux profiles, one pair per anomalous model.

    Interpretation
    --------------
    Shows the radial profile of the turbulent energy flux that each anomalous
    core_transport model (identifier index 6: a TGLF saturation rule, a
    gyrokinetic surrogate, a nonlinear run) gives at its flux-surface grid,
    against the normalised toroidal flux coordinate, in W/m2.  Each model has
    one colour; its electron flux is solid and the sum of its ion species'
    fluxes dashed, and overlaid entries name the model after their label.  It
    is read for where the transport is predicted to sit radially, for the
    electron-to-ion share, and for how far the models of one family (several
    saturation rules, say) disagree on the same profile.

    Options
    -------
    ``time=`` selects, per model, the stored profiles_1d slice nearest the
    requested time; without it the first slice is drawn.

    Limitations
    -----------
    The flux is each model's output on the state it was given, not a
    measurement, and the plot does not set it against a power balance or an
    experimental flux.  Only anomalous models are drawn: neoclassical or
    total-transport models in the same IDS are not.  An ion species whose flux
    is not stored on the grid of the electron flux is left out of the ion sum
    without a note.  A radially sparse grid (a handful of surfaces) is drawn
    as stored, with no interpolation between them.

    See Also
    --------
    turbulent_transport_profile_particle_flux : the particle flux of the same models.
    gyrokinetics_spectrum_energy_flux : the k_y-resolved flux behind one surface's value.
    """
    return render_profile_1d(model, ax=ax, show=show, **style)


@_transport("particle_flux", "Turbulent electron and ion particle flux against rho_tor_norm.")
def turbulent_transport_profile_particle_flux(
    model: Profile1D, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Turbulent particle flux profiles, one pair per anomalous model.

    Interpretation
    --------------
    Shows the radial profile of the turbulent particle flux that each
    anomalous core_transport model (identifier index 6) gives at its
    flux-surface grid, against the normalised toroidal flux coordinate, in
    particles per m2 per s, with the same colour-per-model, electrons solid
    and ions dashed layout as the energy-flux profile.  It is read for the
    sign of the predicted particle flux along the radius (an inward,
    pinch-like region as much as an outward one), for the species balance,
    and for the spread between the models of one family.

    Options
    -------
    ``time=`` selects, per model, the stored profiles_1d slice nearest the
    requested time; without it the first slice is drawn.

    Limitations
    -----------
    The flux is each model's output, not a measurement, and no particle
    source or density evolution is set against it here.  Only anomalous
    models are drawn.  An ion species stored off the electron grid is left out
    of the ion sum without a note, and a sparse grid is drawn as stored.

    See Also
    --------
    turbulent_transport_profile_energy_flux : the energy flux of the same models.
    gyrokinetics_spectrum_particle_flux : the k_y-resolved flux behind one surface's value.
    """
    return render_profile_1d(model, ax=ax, show=show, **style)


@renderer(
    domain="core_transport", subject="turbulent_transport", view="overview", model=Panels,
    ids=("core_transport",), description="Turbulent energy and particle flux profiles side by side.",
    required_paths=("core_transport.model.{i}.identifier.index",),
)
def turbulent_transport_overview(
    model: Panels, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Composite of the turbulent flux profiles.

    Interpretation
    --------------
    The turbulent energy-flux and particle-flux profiles of the anomalous
    core_transport models side by side on one radial axis, so the scales that
    carry energy and the scales that carry particles can be read against each
    other for every model at once: where both peak, whether the particle flux
    changes sign where the energy flux does not, and whether the models of one
    family spread more in one channel than in the other.  Colours and line
    styles are those of the member profiles (one colour per model, electrons
    solid, ions dashed).

    Options
    -------
    ``time=`` selects, per model, the stored profiles_1d slice nearest the
    requested time in both panels.

    Limitations
    -----------
    The members' limitations apply: model outputs rather than measurements,
    anomalous models only, ion sums over the species stored on the electron
    grid, no interpolation between sparse surfaces.  The two panels share the
    radial axis, not a flux scale.

    See Also
    --------
    turbulent_transport_profile_energy_flux : the left panel on its own.
    turbulent_transport_profile_particle_flux : the right panel on its own.
    """
    return render_panels(model, ax=ax, show=show, **style)
