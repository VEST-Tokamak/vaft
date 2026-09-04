"""Plot adapters for native IMAS objects: ``IDSToplevel``, ``DBEntry``, handles.

Each ``plot_<canonical-stem>`` here interprets its input natively -- no
conversion to an ODS -- builds the typed view model the matching
:mod:`vaft.plot` renderer expects, and delegates rendering.  No Matplotlib
code lives in this namespace, and no imas-python class is ever patched
(issues #62 and #63).

Every adapter shares one signature::

    plot_<stem>(ids_or_dbentry_or_handle_or_list, *, ax=None, show=False, label="shot", **options)

and returns the renderer's ``(Figure, Axes)`` or ``(Figure, ndarray[Axes])``.
``label`` selects how entries are labeled -- ``"shot"``/``"pulse"`` (the data
entry's pulse, which a bare toplevel does not carry, so it is labelled by
position), ``"run"``, ``"key"``, or an explicit sequence -- and list ordering
is preserved.

The code-backed plots (those built by code rather than by path reads) convert
the IDS they declare to an OMAS ODS on the way, through
:meth:`vaft.imas.access.IDSEntry.as_ods_for`; ``available_plots(obj,
detail=True)`` marks them.  Use :func:`available_plots` to see which plots a
particular object can produce.
"""

from __future__ import annotations

from typing import Any, Sequence

from vaft.plot.backend.render import render_entries

from .entries import normalize_entries

__all__ = ["available_plots", "normalize_entries", "render"]


def render(
    name: str,
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Build the view model for ``name`` from a native IMAS ``source`` and render it."""
    return render_entries(
        name, normalize_entries(source, label=label), ax=ax, show=show,
        namespace="vaft.imas", subject="ids", **options,
    )


def available_plots(
    source: Any = None,
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
):
    """What can be plotted from native IMAS input, as the semantic catalog.

    Same contract as :func:`vaft.omas.available_plots`, for an ``IDSToplevel``,
    a ``DBEntry``, a handle, or a list of them.
    """
    from .discovery import describe

    return describe(
        source, query=query, detail=detail, available_only=available_only, **filters
    )


def plot_b_field_probe_time_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal field measured by each selected B-field probe.

    Renders with :func:`vaft.plot.b_field_probe_time_field` from native IMAS input.
    """
    return render("b_field_probe_time_field", source, ax=ax, show=show, label=label, **options)


def plot_barometry_time_pressure(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Neutral pressure history from the barometry gauges.

    Renders with :func:`vaft.plot.barometry_time_pressure` from native IMAS input.
    """
    return render("barometry_time_pressure", source, ax=ax, show=show, label=label, **options)


def plot_camera_visible_animation_frames(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Animate a sequence of FAST-camera frames on a shared color scale.

    Renders with :func:`vaft.plot.camera_visible_animation_frames` from native IMAS input.
    """
    return render("camera_visible_animation_frames", source, ax=ax, show=show, label=label, **options)


def plot_camera_visible_image(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One camera frame with optional overlays -- wall, equilibrium, field line -- through one projection (issue #261).

    Renders with :func:`vaft.plot.camera_visible_image` from native IMAS input.
    """
    return render("camera_visible_image", source, ax=ax, show=show, label=label, **options)


def plot_camera_visible_image_efit_overlay(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """FAST-camera frame with the calibrated pinhole projection of the wall, LCFS, magnetic axis, and flux surfaces overlaid.

    Renders with :func:`vaft.plot.camera_visible_image_efit_overlay` from native IMAS input.
    """
    return render("camera_visible_image_efit_overlay", source, ax=ax, show=show, label=label, **options)


def plot_camera_visible_image_field_line(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """FAST-camera frame with a traced magnetic field line projected onto it.

    Renders with :func:`vaft.plot.camera_visible_image_field_line` from native IMAS input.
    """
    return render("camera_visible_image_field_line", source, ax=ax, show=show, label=label, **options)


def plot_camera_visible_image_frame(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One FAST-camera frame (raw digital levels, uncalibrated).

    Renders with :func:`vaft.plot.camera_visible_image_frame` from native IMAS input.
    """
    return render("camera_visible_image_frame", source, ax=ax, show=show, label=label, **options)


def plot_charge_exchange_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Charge-exchange measurement positions in the poloidal plane.

    Renders with :func:`vaft.plot.charge_exchange_geometry_poloidal` from native IMAS input.
    """
    return render("charge_exchange_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


def plot_charge_exchange_profile_ion_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Charge-exchange ion temperature versus position.

    Renders with :func:`vaft.plot.charge_exchange_profile_ion_temperature` from native IMAS input.
    """
    return render("charge_exchange_profile_ion_temperature", source, ax=ax, show=show, label=label, **options)


def plot_charge_exchange_profile_velocity_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Charge-exchange toroidal rotation versus position.

    Renders with :func:`vaft.plot.charge_exchange_profile_velocity_tor` from native IMAS input.
    """
    return render("charge_exchange_profile_velocity_tor", source, ax=ax, show=show, label=label, **options)


def plot_charge_exchange_time_ion_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel ion temperature history from charge-exchange spectroscopy.

    Renders with :func:`vaft.plot.charge_exchange_time_ion_temperature` from native IMAS input.
    """
    return render("charge_exchange_time_ion_temperature", source, ax=ax, show=show, label=label, **options)


def plot_charge_exchange_time_velocity_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel toroidal rotation history from charge-exchange spectroscopy.

    Renders with :func:`vaft.plot.charge_exchange_time_velocity_tor` from native IMAS input.
    """
    return render("charge_exchange_time_velocity_tor", source, ax=ax, show=show, label=label, **options)


def plot_chease_overview_profile_validity(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """q0/q95, q-monotonicity and pressure positivity of the refined equilibrium: a converged CHEASE solution that is not physically sound flagged at a glance.

    Renders with :func:`vaft.plot.chease_overview_profile_validity` from native IMAS input.
    """
    return render("chease_overview_profile_validity", source, ax=ax, show=show, label=label, **options)


def plot_chease_overview_refinement_summary(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """How far CHEASE moved each profile and the boundary from the EFIT equilibrium it refined, slice by slice: profile and boundary RMS change, flux-normalization shift, and plasma-current self-consistency.

    Renders with :func:`vaft.plot.chease_overview_refinement_summary` from native IMAS input.
    """
    return render("chease_overview_refinement_summary", source, ax=ax, show=show, label=label, **options)


def plot_coil_3d_geometry3d(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Non-axisymmetric 3D coil filaments in machine Cartesian coordinates.

    Renders with :func:`vaft.plot.coil_3d_geometry3d` from native IMAS input.
    """
    return render("coil_3d_geometry3d", source, ax=ax, show=show, label=label, **options)


def plot_coil_3d_geometry_topview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Non-axisymmetric 3D coil filaments projected into the machine top view.

    Renders with :func:`vaft.plot.coil_3d_geometry_topview` from native IMAS input.
    """
    return render("coil_3d_geometry_topview", source, ax=ax, show=show, label=label, **options)


def plot_core_profiles_time_volume_averaged(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Volume-averaged core quantity panels on a shared time axis.

    Renders with :func:`vaft.plot.core_profiles_time_volume_averaged` from native IMAS input.
    """
    return render("core_profiles_time_volume_averaged", source, ax=ax, show=show, label=label, **options)


def plot_current_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Plasma, PF coil and eddy current panels on a shared time axis.

    Renders with :func:`vaft.plot.current_overview` from native IMAS input.
    """
    return render("current_overview", source, ax=ax, show=show, label=label, **options)


def plot_diagnostics_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time histories of every diagnostic subject, one panel each, in a fixed grid: a diagnostic absent from the input is a labelled empty panel, so the figure has the same shape on every shot. Channels the source flagged invalid are excluded by default.

    Renders with :func:`vaft.plot.diagnostics_overview` from native IMAS input.
    """
    return render("diagnostics_overview", source, ax=ax, show=show, label=label, **options)


def plot_diamagnetic_flux_time(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured diamagnetic flux history.

    Renders with :func:`vaft.plot.diamagnetic_flux_time` from native IMAS input.
    """
    return render("diamagnetic_flux_time", source, ax=ax, show=show, label=label, **options)


def plot_electron_density_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Electron density mapped onto the poloidal plane.

    Renders with :func:`vaft.plot.electron_density_field` from native IMAS input.
    """
    return render("electron_density_field", source, ax=ax, show=show, label=label, **options)


def plot_electron_density_profile(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core electron density profile.

    Renders with :func:`vaft.plot.electron_density_profile` from native IMAS input.
    """
    return render("electron_density_profile", source, ax=ax, show=show, label=label, **options)


def plot_electron_density_time(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Volume-averaged electron density history.

    Renders with :func:`vaft.plot.electron_density_time` from native IMAS input.
    """
    return render("electron_density_time", source, ax=ax, show=show, label=label, **options)


def plot_electron_temperature_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Electron temperature mapped onto the poloidal plane.

    Renders with :func:`vaft.plot.electron_temperature_field` from native IMAS input.
    """
    return render("electron_temperature_field", source, ax=ax, show=show, label=label, **options)


def plot_electron_temperature_profile(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core electron temperature profile.

    Renders with :func:`vaft.plot.electron_temperature_profile` from native IMAS input.
    """
    return render("electron_temperature_profile", source, ax=ax, show=show, label=label, **options)


def plot_electron_temperature_time(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Volume-averaged electron temperature history.

    Renders with :func:`vaft.plot.electron_temperature_time` from native IMAS input.
    """
    return render("electron_temperature_time", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_field_psi(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Reconstructed poloidal flux map on the equilibrium (R, Z) grid.

    Renders with :func:`vaft.plot.equilibrium_field_psi` from native IMAS input.
    """
    return render("equilibrium_field_psi", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_field_psi_vacuum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Vacuum poloidal flux from the PF coils alone, without plasma.

    Renders with :func:`vaft.plot.equilibrium_field_psi_vacuum` from native IMAS input.
    """
    return render("equilibrium_field_psi_vacuum", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_geometry_boundary(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Last-closed-flux-surface outline in the poloidal plane.

    Renders with :func:`vaft.plot.equilibrium_geometry_boundary` from native IMAS input.
    """
    return render("equilibrium_geometry_boundary", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_geometry_topview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium boundary projected into the machine top view.

    Renders with :func:`vaft.plot.equilibrium_geometry_topview` from native IMAS input.
    """
    return render("equilibrium_geometry_topview", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One equilibrium slice from one figure: poloidal flux with the LCFS and axis, pressure and q profiles, and the slice's global quantities.

    Renders with :func:`vaft.plot.equilibrium_overview` from native IMAS input.
    """
    return render("equilibrium_overview", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_constraint_coverage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Enabled, disabled and missing constraint channels per family across the reconstructed time slices.

    Renders with :func:`vaft.plot.equilibrium_overview_constraint_coverage` from native IMAS input.
    """
    return render("equilibrium_overview_constraint_coverage", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_constraints(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Magnetic constraints as submitted to EFIT, per family, with enabled, disabled and missing channels distinguished.

    Renders with :func:`vaft.plot.equilibrium_overview_constraints` from native IMAS input.
    """
    return render("equilibrium_overview_constraints", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_convergence(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT numerical convergence: final Grad-Shafranov error against the requested tolerance, iteration count against its cap, the error history where one was written, and EFIT's outputs checked against each other.

    Renders with :func:`vaft.plot.equilibrium_overview_convergence` from native IMAS input.
    """
    return render("equilibrium_overview_convergence", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_fit_quality(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT goodness of fit: reduced chi-square against the degrees of freedom EFIT itself reports, which diagnostic family carries the chi-square, and residuals normalized by the uncertainty EFIT was given.

    Renders with :func:`vaft.plot.equilibrium_overview_fit_quality` from native IMAS input.
    """
    return render("equilibrium_overview_fit_quality", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_histories(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium global quantities against time: Ip, beta_p, li, q95.

    Renders with :func:`vaft.plot.equilibrium_overview_histories` from native IMAS input.
    """
    return render("equilibrium_overview_histories", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_profiles(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Principal 1-D equilibrium profiles: pressure, toroidal current density and safety factor.

    Renders with :func:`vaft.plot.equilibrium_overview_profiles` from native IMAS input.
    """
    return render("equilibrium_overview_profiles", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_residuals(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured-minus-reconstructed residuals by diagnostic family, with the solver's convergence context beside them rather than in place of them.

    Renders with :func:`vaft.plot.equilibrium_overview_residuals` from native IMAS input.
    """
    return render("equilibrium_overview_residuals", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_verification(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT verification overview: measured and reconstructed constraints beside the reconstructed poloidal-flux map.

    Renders with :func:`vaft.plot.equilibrium_overview_verification` from native IMAS input.
    """
    return render("equilibrium_overview_verification", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_profile_f(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium poloidal current function F = R*B_t.

    Renders with :func:`vaft.plot.equilibrium_profile_f` from native IMAS input.
    """
    return render("equilibrium_profile_f", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_profile_ffprime(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium F dF/dpsi profile.

    Renders with :func:`vaft.plot.equilibrium_profile_ffprime` from native IMAS input.
    """
    return render("equilibrium_profile_ffprime", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_profile_j_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium toroidal current-density profile.

    Renders with :func:`vaft.plot.equilibrium_profile_j_tor` from native IMAS input.
    """
    return render("equilibrium_profile_j_tor", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_profile_pprime(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium dp/dpsi profile.

    Renders with :func:`vaft.plot.equilibrium_profile_pprime` from native IMAS input.
    """
    return render("equilibrium_profile_pprime", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_profile_pressure(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium 1D pressure profile.

    Renders with :func:`vaft.plot.equilibrium_profile_pressure` from native IMAS input.
    """
    return render("equilibrium_profile_pressure", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_profile_q(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium safety-factor profile.

    Renders with :func:`vaft.plot.equilibrium_profile_q` from native IMAS input.
    """
    return render("equilibrium_profile_q", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_beta(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal, toroidal and normalized beta panels.

    Renders with :func:`vaft.plot.equilibrium_time_beta` from native IMAS input.
    """
    return render("equilibrium_time_beta", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_beta_n(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Normalized beta history.

    Renders with :func:`vaft.plot.equilibrium_time_beta_n` from native IMAS input.
    """
    return render("equilibrium_time_beta_n", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_beta_p(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal beta history.

    Renders with :func:`vaft.plot.equilibrium_time_beta_p` from native IMAS input.
    """
    return render("equilibrium_time_beta_p", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_beta_t(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Toroidal beta history.

    Renders with :func:`vaft.plot.equilibrium_time_beta_t` from native IMAS input.
    """
    return render("equilibrium_time_beta_t", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_diamagnetic_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured versus reconstructed diamagnetic-flux constraint.

    Renders with :func:`vaft.plot.equilibrium_time_diamagnetic_flux` from native IMAS input.
    """
    return render("equilibrium_time_diamagnetic_flux", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_li(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Internal inductance li_3 history.

    Renders with :func:`vaft.plot.equilibrium_time_li` from native IMAS input.
    """
    return render("equilibrium_time_li", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_major_radius(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Geometric-axis major radius history.

    Renders with :func:`vaft.plot.equilibrium_time_major_radius` from native IMAS input.
    """
    return render("equilibrium_time_major_radius", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_plasma_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Reconstructed plasma current history.

    Renders with :func:`vaft.plot.equilibrium_time_plasma_current` from native IMAS input.
    """
    return render("equilibrium_time_plasma_current", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_q0(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Safety factor on axis.

    Renders with :func:`vaft.plot.equilibrium_time_q0` from native IMAS input.
    """
    return render("equilibrium_time_q0", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_q95(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Safety factor at the 95% flux surface.

    Renders with :func:`vaft.plot.equilibrium_time_q95` from native IMAS input.
    """
    return render("equilibrium_time_q95", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_qa(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Safety factor at the plasma edge.

    Renders with :func:`vaft.plot.equilibrium_time_qa` from native IMAS input.
    """
    return render("equilibrium_time_qa", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_virial(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Virial-estimate equilibrium quantities against the reconstruction.

    Renders with :func:`vaft.plot.equilibrium_time_virial` from native IMAS input.
    """
    return render("equilibrium_time_virial", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_w_mag(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Magnetic stored energy history.

    Renders with :func:`vaft.plot.equilibrium_time_w_mag` from native IMAS input.
    """
    return render("equilibrium_time_w_mag", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_w_mhd(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """MHD stored energy history.

    Renders with :func:`vaft.plot.equilibrium_time_w_mhd` from native IMAS input.
    """
    return render("equilibrium_time_w_mhd", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_time_w_tot(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Total stored energy history.

    Renders with :func:`vaft.plot.equilibrium_time_w_tot` from native IMAS input.
    """
    return render("equilibrium_time_w_tot", source, ax=ax, show=show, label=label, **options)


def plot_flux_loop_time_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal flux measured by each selected flux loop.

    Renders with :func:`vaft.plot.flux_loop_time_flux` from native IMAS input.
    """
    return render("flux_loop_time_flux", source, ax=ax, show=show, label=label, **options)


def plot_flux_loop_time_voltage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Loop voltage measured by each selected flux loop.

    Renders with :func:`vaft.plot.flux_loop_time_voltage` from native IMAS input.
    """
    return render("flux_loop_time_voltage", source, ax=ax, show=show, label=label, **options)


def plot_impa_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """IMPA validation overview: raw voltages, compensated Bz and the 1/R position check.

    Renders with :func:`vaft.plot.impa_overview` from native IMAS input.
    """
    return render("impa_overview", source, ax=ax, show=show, label=label, **options)


def plot_impa_profile_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """IMPA measured field against probe radius with the 1/R toroidal-field model.

    Renders with :func:`vaft.plot.impa_profile_field` from native IMAS input.
    """
    return render("impa_profile_field", source, ax=ax, show=show, label=label, **options)


def plot_impa_time_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Calibrated field from the IMPA Hall-probe array.

    Renders with :func:`vaft.plot.impa_time_field` from native IMAS input.
    """
    return render("impa_time_field", source, ax=ax, show=show, label=label, **options)


def plot_impa_time_voltage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Raw IMPA Hall-probe voltages, one trace per channel.

    Renders with :func:`vaft.plot.impa_time_voltage` from native IMAS input.
    """
    return render("impa_time_voltage", source, ax=ax, show=show, label=label, **options)


def plot_interferometer_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Interferometer overview: line density history and spectrogram.

    Renders with :func:`vaft.plot.interferometer_overview` from native IMAS input.
    """
    return render("interferometer_overview", source, ax=ax, show=show, label=label, **options)


def plot_interferometer_spectrogram(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of one interferometer channel's line density.

    Renders with :func:`vaft.plot.interferometer_spectrogram` from native IMAS input.
    """
    return render("interferometer_spectrogram", source, ax=ax, show=show, label=label, **options)


def plot_interferometer_spectrum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one interferometer channel's line density.

    Renders with :func:`vaft.plot.interferometer_spectrum` from native IMAS input.
    """
    return render("interferometer_spectrum", source, ax=ax, show=show, label=label, **options)


def plot_interferometer_time_n_e_line(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Interferometer line-integrated electron density history.

    Renders with :func:`vaft.plot.interferometer_time_n_e_line` from native IMAS input.
    """
    return render("interferometer_time_n_e_line", source, ax=ax, show=show, label=label, **options)


def plot_ion_temperature_profile(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core ion temperature profile.

    Renders with :func:`vaft.plot.ion_temperature_profile` from native IMAS input.
    """
    return render("ion_temperature_profile", source, ax=ax, show=show, label=label, **options)


def plot_limiter_current_time(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Lower-corner, upper-corner and midplane limiter currents in three panels.

    Renders with :func:`vaft.plot.limiter_current_time` from native IMAS input.
    """
    return render("limiter_current_time", source, ax=ax, show=show, label=label, **options)


def plot_machine_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Composed poloidal machine view: wall, coils, passive structure and diagnostic positions and sight lines in one axes.

    Renders with :func:`vaft.plot.machine_geometry_poloidal` from native IMAS input.
    """
    return render("machine_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


def plot_machine_geometry_topview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Composed machine top view: machine-boundary and plasma extent plus launcher, antenna and pellet-injector geometry.

    Renders with :func:`vaft.plot.machine_geometry_topview` from native IMAS input.
    """
    return render("machine_geometry_topview", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Flux-loop and B-field-probe positions in the poloidal plane.

    Renders with :func:`vaft.plot.magnetics_geometry_poloidal` from native IMAS input.
    """
    return render("magnetics_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Shot diagnostic overview: current, field, flux and geometry panels.

    Renders with :func:`vaft.plot.magnetics_overview` from native IMAS input.
    """
    return render("magnetics_overview", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_overview_plasma_residual(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Residual of measured minus coil+eddy synthetic magnetics against the pre-plasma noise band, with the plasma-current and residual onsets.

    Renders with :func:`vaft.plot.magnetics_overview_plasma_residual` from native IMAS input.
    """
    return render("magnetics_overview_plasma_residual", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_overview_vacuum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured, coil-only and coil+eddy synthetic magnetics per channel: the forward-modeled validation of an eddy-current reconstruction.

    Renders with :func:`vaft.plot.magnetics_overview_vacuum` from native IMAS input.
    """
    return render("magnetics_overview_vacuum", source, ax=ax, show=show, label=label, **options)


def plot_mhd_linear_time_energy_perturbed(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """DCON perturbed potential energy against time, one trace per toroidal mode number; a negative value is an ideal-MHD unstable mode.

    Renders with :func:`vaft.plot.mhd_linear_time_energy_perturbed` from native IMAS input.
    """
    return render("mhd_linear_time_energy_perturbed", source, ax=ax, show=show, label=label, **options)


def plot_mhd_linear_profile_displacement(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """DCON displacement eigenfunction against normalized flux, one trace per poloidal harmonic; amplitudes are normalized to the peak because DCON's eigenvector normalization is arbitrary.

    Renders with :func:`vaft.plot.mhd_linear_profile_displacement` from native IMAS input.
    """
    return render("mhd_linear_profile_displacement", source, ax=ax, show=show, label=label, **options)


def plot_mhd_linear_profile_b_field_perturbed(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Normal perturbed field per poloidal harmonic against normalized flux, derived from the DCON eigenfunction as i(m - nq) xi.grad(psi).

    Renders with :func:`vaft.plot.mhd_linear_profile_b_field_perturbed` from native IMAS input.
    """
    return render("mhd_linear_profile_b_field_perturbed", source, ax=ax, show=show, label=label, **options)


def plot_mhd_linear_overview_eigenfunction(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """The DCON eigenfunction of the least-stable mapped mode: displacement and normal perturbed field per poloidal harmonic.

    Renders with :func:`vaft.plot.mhd_linear_overview_eigenfunction` from native IMAS input.
    """
    return render("mhd_linear_overview_eigenfunction", source, ax=ax, show=show, label=label, **options)


def plot_mirnov_spectrogram(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of one Mirnov coil signal.

    Renders with :func:`vaft.plot.mirnov_spectrogram` from native IMAS input.
    """
    return render("mirnov_spectrogram", source, ax=ax, show=show, label=label, **options)


def plot_mirnov_spectrum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one Mirnov coil signal.

    Renders with :func:`vaft.plot.mirnov_spectrum` from native IMAS input.
    """
    return render("mirnov_spectrum", source, ax=ax, show=show, label=label, **options)


def plot_mirnov_time_voltage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Raw or preprocessed Mirnov coil voltage traces.

    Renders with :func:`vaft.plot.mirnov_time_voltage` from native IMAS input.
    """
    return render("mirnov_time_voltage", source, ax=ax, show=show, label=label, **options)


def plot_passive_structure_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Passive conducting-structure loop outlines in the poloidal plane.

    Renders with :func:`vaft.plot.passive_structure_geometry_poloidal` from native IMAS input.
    """
    return render("passive_structure_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


def plot_passive_structure_geometry_wall_mode(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One segment-local wall eigenmode coloured onto the passive structure.

    Options: ``segment`` (id, default the first), ``mode`` (index within the
    segment, default 0), ``basis`` (a precomputed ``WallModeBasis``),
    ``remap_em_coupling``.  Renders with
    :func:`vaft.plot.passive_structure_geometry_wall_mode` from native IMAS input.
    """
    return render(
        "passive_structure_geometry_wall_mode", source, ax=ax, show=show, label=label, **options
    )


def plot_passive_structure_overview_wall_reduction(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Reduced-wall response error against retained order (vaft #494).

    Options: ``rows`` (precomputed convergence rows), ``drive``, ``rules``,
    ``orders``, ``metrics``, ``remap_em_coupling``.  Renders with
    :func:`vaft.plot.passive_structure_overview_wall_reduction` from native IMAS input.
    """
    return render(
        "passive_structure_overview_wall_reduction", source, ax=ax, show=show, label=label, **options
    )


def plot_passive_structure_field_wall_reduction(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Full, reduced or difference wall flux map on the equilibrium region.

    Options: ``which`` (``full``/``reduced``/``difference``), ``selection`` or
    ``rule``+``M``, ``time``, ``grid_shape``, ``remap_em_coupling``.  Renders
    with :func:`vaft.plot.passive_structure_field_wall_reduction` from native IMAS input.
    """
    return render(
        "passive_structure_field_wall_reduction", source, ax=ax, show=show, label=label, **options
    )


def plot_passive_structure_overview_wall_time(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Decay-time spectrum of the passive wall's segment-wise eigenmodes.

    Options: ``max_modes`` per segment, ``whole_wall`` (draw the whole-wall
    spectrum, default True), ``basis``, ``remap_em_coupling``.  Renders with
    :func:`vaft.plot.passive_structure_overview_wall_time` from native IMAS input.
    """
    return render(
        "passive_structure_overview_wall_time", source, ax=ax, show=show, label=label, **options
    )


def plot_pf_coil_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """PF coil outlines in the poloidal plane.

    Renders with :func:`vaft.plot.pf_coil_geometry_poloidal` from native IMAS input.
    """
    return render("pf_coil_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


def plot_pf_coil_time_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-coil PF current history.

    Renders with :func:`vaft.plot.pf_coil_time_current` from native IMAS input.
    """
    return render("pf_coil_time_current", source, ax=ax, show=show, label=label, **options)


def plot_pf_coil_time_current_turns(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-coil PF current multiplied by the signed turn count (ampere-turns).

    Renders with :func:`vaft.plot.pf_coil_time_current_turns` from native IMAS input.
    """
    return render("pf_coil_time_current_turns", source, ax=ax, show=show, label=label, **options)


def plot_plasma_current_time(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured plasma current history from the Rogowski coil.

    Renders with :func:`vaft.plot.plasma_current_time` from native IMAS input.
    """
    return render("plasma_current_time", source, ax=ax, show=show, label=label, **options)


def plot_soft_x_rays_geometry_lines_of_sight(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Soft X-ray detector lines of sight over the poloidal cross-section.

    Renders with :func:`vaft.plot.soft_x_rays_geometry_lines_of_sight` from native IMAS input.
    """
    return render("soft_x_rays_geometry_lines_of_sight", source, ax=ax, show=show, label=label, **options)


def plot_soft_x_rays_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Soft X-ray overview: lines of sight, signals and channel pattern.

    Renders with :func:`vaft.plot.soft_x_rays_overview` from native IMAS input.
    """
    return render("soft_x_rays_overview", source, ax=ax, show=show, label=label, **options)


def plot_soft_x_rays_spectrogram(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of one soft X-ray channel.

    Renders with :func:`vaft.plot.soft_x_rays_spectrogram` from native IMAS input.
    """
    return render("soft_x_rays_spectrogram", source, ax=ax, show=show, label=label, **options)


def plot_soft_x_rays_spectrum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one soft X-ray channel.

    Renders with :func:`vaft.plot.soft_x_rays_spectrum` from native IMAS input.
    """
    return render("soft_x_rays_spectrum", source, ax=ax, show=show, label=label, **options)


def plot_soft_x_rays_time_power(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Soft X-ray channel signal history.

    Renders with :func:`vaft.plot.soft_x_rays_time_power` from native IMAS input.
    """
    return render("soft_x_rays_time_power", source, ax=ax, show=show, label=label, **options)


def plot_spectrometer_uv_time_impurity(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Impurity line-intensity panels against plasma current.

    Renders with :func:`vaft.plot.spectrometer_uv_time_impurity` from native IMAS input.
    """
    return render("spectrometer_uv_time_impurity", source, ax=ax, show=show, label=label, **options)


def plot_spectrometer_uv_time_intensity(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Processed spectral line intensity history.

    Renders with :func:`vaft.plot.spectrometer_uv_time_intensity` from native IMAS input.
    """
    return render("spectrometer_uv_time_intensity", source, ax=ax, show=show, label=label, **options)


def plot_summary_time_energy(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Stored-energy comparison panels across available estimates.

    Renders with :func:`vaft.plot.summary_time_energy` from native IMAS input.
    """
    return render("summary_time_energy", source, ax=ax, show=show, label=label, **options)


def plot_summary_time_power_balance(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Ohmic input, radiated and conducted power balance panels.

    Renders with :func:`vaft.plot.summary_time_power_balance` from native IMAS input.
    """
    return render("summary_time_power_balance", source, ax=ax, show=show, label=label, **options)


def plot_summary_time_voltage_consumption(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Loop-voltage and flux-consumption panels.

    Renders with :func:`vaft.plot.summary_time_voltage_consumption` from native IMAS input.
    """
    return render("summary_time_voltage_consumption", source, ax=ax, show=show, label=label, **options)


def plot_tf_coil_time_b_t(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Toroidal field history at the reference radius.

    Renders with :func:`vaft.plot.tf_coil_time_b_t` from native IMAS input.
    """
    return render("tf_coil_time_b_t", source, ax=ax, show=show, label=label, **options)


def plot_tf_coil_time_b_t_vacuum_r(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Vacuum toroidal field times major radius (B_t * R).

    Renders with :func:`vaft.plot.tf_coil_time_b_t_vacuum_r` from native IMAS input.
    """
    return render("tf_coil_time_b_t_vacuum_r", source, ax=ax, show=show, label=label, **options)


def plot_tf_coil_time_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """TF coil current history.

    Renders with :func:`vaft.plot.tf_coil_time_current` from native IMAS input.
    """
    return render("tf_coil_time_current", source, ax=ax, show=show, label=label, **options)


def plot_thermal_pressure_profile(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core total pressure profile.

    Renders with :func:`vaft.plot.thermal_pressure_profile` from native IMAS input.
    """
    return render("thermal_pressure_profile", source, ax=ax, show=show, label=label, **options)


def plot_thomson_scattering_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Thomson-scattering measurement positions in the poloidal plane.

    Renders with :func:`vaft.plot.thomson_scattering_geometry_poloidal` from native IMAS input.
    """
    return render("thomson_scattering_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


def plot_thomson_scattering_profile_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Thomson-scattering electron density versus position.

    Renders with :func:`vaft.plot.thomson_scattering_profile_electron_density` from native IMAS input.
    """
    return render("thomson_scattering_profile_electron_density", source, ax=ax, show=show, label=label, **options)


def plot_thomson_scattering_profile_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Thomson-scattering electron temperature versus position.

    Renders with :func:`vaft.plot.thomson_scattering_profile_electron_temperature` from native IMAS input.
    """
    return render("thomson_scattering_profile_electron_temperature", source, ax=ax, show=show, label=label, **options)


def plot_thomson_scattering_time_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel Thomson electron density history.

    Renders with :func:`vaft.plot.thomson_scattering_time_electron_density` from native IMAS input.
    """
    return render("thomson_scattering_time_electron_density", source, ax=ax, show=show, label=label, **options)


def plot_thomson_scattering_time_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel Thomson electron temperature history.

    Renders with :func:`vaft.plot.thomson_scattering_time_electron_temperature` from native IMAS input.
    """
    return render("thomson_scattering_time_electron_temperature", source, ax=ax, show=show, label=label, **options)


def plot_wall_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """First-wall and limiter outline in the poloidal plane.

    Renders with :func:`vaft.plot.wall_geometry_poloidal` from native IMAS input.
    """
    return render("wall_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


__all__ += [
    "plot_b_field_probe_time_field",
    "plot_barometry_time_pressure",
    "plot_camera_visible_animation_frames",
    "plot_camera_visible_image",
    "plot_camera_visible_image_efit_overlay",
    "plot_camera_visible_image_field_line",
    "plot_camera_visible_image_frame",
    "plot_charge_exchange_geometry_poloidal",
    "plot_charge_exchange_profile_ion_temperature",
    "plot_charge_exchange_profile_velocity_tor",
    "plot_charge_exchange_time_ion_temperature",
    "plot_charge_exchange_time_velocity_tor",
    "plot_chease_overview_profile_validity",
    "plot_chease_overview_refinement_summary",
    "plot_coil_3d_geometry3d",
    "plot_coil_3d_geometry_topview",
    "plot_core_profiles_time_volume_averaged",
    "plot_current_overview",
    "plot_diagnostics_overview",
    "plot_diamagnetic_flux_time",
    "plot_electron_density_field",
    "plot_electron_density_profile",
    "plot_electron_density_time",
    "plot_electron_temperature_field",
    "plot_electron_temperature_profile",
    "plot_electron_temperature_time",
    "plot_equilibrium_field_psi",
    "plot_equilibrium_field_psi_vacuum",
    "plot_equilibrium_geometry_boundary",
    "plot_equilibrium_geometry_topview",
    "plot_equilibrium_overview",
    "plot_equilibrium_overview_constraint_coverage",
    "plot_equilibrium_overview_constraints",
    "plot_equilibrium_overview_convergence",
    "plot_equilibrium_overview_fit_quality",
    "plot_equilibrium_overview_histories",
    "plot_equilibrium_overview_profiles",
    "plot_equilibrium_overview_residuals",
    "plot_equilibrium_overview_verification",
    "plot_equilibrium_profile_f",
    "plot_equilibrium_profile_ffprime",
    "plot_equilibrium_profile_j_tor",
    "plot_equilibrium_profile_pprime",
    "plot_equilibrium_profile_pressure",
    "plot_equilibrium_profile_q",
    "plot_equilibrium_time_beta",
    "plot_equilibrium_time_beta_n",
    "plot_equilibrium_time_beta_p",
    "plot_equilibrium_time_beta_t",
    "plot_equilibrium_time_diamagnetic_flux",
    "plot_equilibrium_time_li",
    "plot_equilibrium_time_major_radius",
    "plot_equilibrium_time_plasma_current",
    "plot_equilibrium_time_q0",
    "plot_equilibrium_time_q95",
    "plot_equilibrium_time_qa",
    "plot_equilibrium_time_virial",
    "plot_equilibrium_time_w_mag",
    "plot_equilibrium_time_w_mhd",
    "plot_equilibrium_time_w_tot",
    "plot_flux_loop_time_flux",
    "plot_flux_loop_time_voltage",
    "plot_impa_overview",
    "plot_impa_profile_field",
    "plot_impa_time_field",
    "plot_impa_time_voltage",
    "plot_interferometer_overview",
    "plot_interferometer_spectrogram",
    "plot_interferometer_spectrum",
    "plot_interferometer_time_n_e_line",
    "plot_ion_temperature_profile",
    "plot_limiter_current_time",
    "plot_machine_geometry_poloidal",
    "plot_machine_geometry_topview",
    "plot_magnetics_geometry_poloidal",
    "plot_magnetics_overview",
    "plot_magnetics_overview_plasma_residual",
    "plot_magnetics_overview_vacuum",
    "plot_mhd_linear_overview_eigenfunction",
    "plot_mhd_linear_profile_b_field_perturbed",
    "plot_mhd_linear_profile_displacement",
    "plot_mhd_linear_time_energy_perturbed",
    "plot_mirnov_spectrogram",
    "plot_mirnov_spectrum",
    "plot_mirnov_time_voltage",
    "plot_passive_structure_geometry_poloidal",
    "plot_passive_structure_field_wall_reduction",
    "plot_passive_structure_geometry_wall_mode",
    "plot_passive_structure_overview_wall_reduction",
    "plot_passive_structure_overview_wall_time",
    "plot_pf_coil_geometry_poloidal",
    "plot_pf_coil_time_current",
    "plot_pf_coil_time_current_turns",
    "plot_plasma_current_time",
    "plot_soft_x_rays_geometry_lines_of_sight",
    "plot_soft_x_rays_overview",
    "plot_soft_x_rays_spectrogram",
    "plot_soft_x_rays_spectrum",
    "plot_soft_x_rays_time_power",
    "plot_spectrometer_uv_time_impurity",
    "plot_spectrometer_uv_time_intensity",
    "plot_summary_time_energy",
    "plot_summary_time_power_balance",
    "plot_summary_time_voltage_consumption",
    "plot_tf_coil_time_b_t",
    "plot_tf_coil_time_b_t_vacuum_r",
    "plot_tf_coil_time_current",
    "plot_thermal_pressure_profile",
    "plot_thomson_scattering_geometry_poloidal",
    "plot_thomson_scattering_profile_electron_density",
    "plot_thomson_scattering_profile_electron_temperature",
    "plot_thomson_scattering_time_electron_density",
    "plot_thomson_scattering_time_electron_temperature",
    "plot_wall_geometry_poloidal",
]
