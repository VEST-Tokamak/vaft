"""Plot adapters for OMAS ``ODS``, ``ODC`` and lists of them.

Each ``plot_<canonical-stem>`` here interprets its input, builds the typed view
model the matching :mod:`vaft.plot` renderer expects, and delegates rendering.
No Matplotlib code lives in this namespace: the adapter owns data interpretation
and the renderer owns drawing (issues #62 and #63).

Every adapter shares one signature::

    plot_<stem>(ods_or_odc_or_list, *, ax=None, show=False, label="shot", **options)

and returns the renderer's ``(Figure, Axes)`` or ``(Figure, ndarray[Axes])``.
``label`` selects how entries are labeled -- ``"shot"``/``"pulse"``, ``"run"``,
``"key"``, or an explicit sequence -- and list/ODC ordering is preserved, so
repeated calls produce the same legend order.

Use :func:`available_plots` to see which plots a particular object can produce,
and :func:`enable_plot_methods` to opt in to ``ODS.plot_*`` methods.
:func:`enable_overlay_methods` does the same for OMAS' own
``ODS.plot_*_overlay`` methods, giving them the ``ax``/``show`` contract.
"""

from __future__ import annotations

from typing import Any, Sequence

from vaft.plot.registry import available_plots as _registry_available_plots
from vaft.plot.registry import get_spec, specs

from ._plot_recipes import (
    build_model,
    entry_supports,
    extract_labels_from_odc,
    normalize_entries,
)


#: Options consumed while building the view model.  Everything else a caller
#: passes is forwarded to the renderer as styling, so an unsupported Matplotlib
#: keyword fails loudly instead of being silently dropped.
_EXTRACTION_OPTIONS = frozenset(
    {
        "channel",
        "channels",
        "contour_levels",
        "coordinate",
        "detector",
        "detrend",
        "dphi_deg",
        "direction",
        "fit_ranges",
        "flux_surface_levels",
        "frame_index",
        "frame_indices",
        "log_y",
        "marker_frequencies",
        "max_frequency",
        "max_length_m",
        "noverlap",
        "nperseg",
        "per_family",
        "phi0",
        "quantity",
        "r0",
        "reference_slopes",
        "sample_rate",
        "series_label",
        "sigma",
        "shot",
        "show_lcfs",
        "show_magnetic_axis",
        "show_wall",
        "time",
        "time_range",
        "time_resolution",
        "time_slice",
        "title",
        "use_wall_boundary",
        "window",
        "window_size",
        "x_limits",
        "xunit",
        "z0",
        "yunit",
    }
)


def render(
    name: str,
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Build the view model for ``name`` from ``source`` and render it.

    This is the shared body behind every ``plot_*`` adapter below, and the entry
    point for rendering a canonical plot chosen at runtime.
    """
    spec = get_spec(name)
    entries = normalize_entries(source, label=label)
    model = build_model(name, entries, **options)
    style = {
        key: value for key, value in options.items() if key not in _EXTRACTION_OPTIONS
    }
    return spec.renderer(model, ax=ax, show=show, **style)


def available_plots(source: Any = None, **filters: Any) -> tuple[dict[str, Any], ...]:
    """Describe the plots available here, optionally filtered by an object.

    Without ``source`` this mirrors :func:`vaft.plot.available_plots`.  With an
    ``ODS``, ``ODC`` or list, only the rows whose required data is actually
    present are returned.
    """
    rows = _registry_available_plots(**filters)
    if source is None:
        return rows
    entries = normalize_entries(source, label="key")
    return tuple(
        row
        for row in rows
        if any(entry_supports(ods, row["name"]) for _, ods in entries)
    )


def enable_plot_methods(*, overwrite: bool = False) -> tuple[str, ...]:
    """Bind ``ODS.plot_<canonical-stem>`` methods for every canonical plot.

    Registration is explicit and idempotent: importing ``vaft`` never mutates
    OMAS.  OMAS ships its own ``plot_*`` methods, so any name that would replace
    one raises :class:`RuntimeError` listing the collisions unless ``overwrite``
    is passed.  Returns the names that are bound after the call.
    """
    from omas import ODS

    bound = getattr(ODS, "_vaft_plot_methods", frozenset())
    targets = {f"plot_{spec.name}": spec.name for spec in specs()}

    collisions = sorted(
        name for name in targets if hasattr(ODS, name) and name not in bound
    )
    if collisions and not overwrite:
        raise RuntimeError(
            "refusing to replace existing ODS methods: "
            + ", ".join(collisions)
            + ". Call vaft.omas.enable_plot_methods(overwrite=True) to take them "
            "over, or use the vaft.omas.plot_* functions instead."
        )

    for method_name, plot_name in targets.items():
        if method_name in bound:
            continue
        setattr(ODS, method_name, _make_method(plot_name))
    ODS._vaft_plot_methods = frozenset(targets)
    return tuple(sorted(targets))


def disable_plot_methods() -> None:
    """Remove the methods bound by :func:`enable_plot_methods`."""
    from omas import ODS

    for method_name in getattr(ODS, "_vaft_plot_methods", frozenset()):
        try:
            delattr(ODS, method_name)
        except AttributeError:
            pass
    ODS._vaft_plot_methods = frozenset()


def enable_overlay_methods(*, overwrite: bool = False) -> tuple[str, ...]:
    """Wrap OMAS' native ``ODS.plot_*_overlay`` so ``ax=None`` means a new figure.

    OMAS draws its overlays onto whatever axes Pyplot happens to have current, so
    two successive calls silently composite into a single figure.  The wrapper
    routes ``ax`` through the same :func:`vaft.plot.style.resolve_axes` contract
    every canonical renderer uses: ``ax=None`` creates a figure, and a
    caller-supplied ``ax`` stays authoritative so the compositional form keeps
    working.

    Like :func:`enable_plot_methods` this is explicit and idempotent -- importing
    ``vaft`` never mutates OMAS -- and ``show`` defaults to ``False``, because
    displaying a figure is the caller's decision.  ``overwrite`` re-wraps methods
    that some other layer has already replaced.  Returns the wrapped names.
    """
    from omas import ODS

    wrapped = getattr(ODS, "_vaft_overlay_methods", frozenset())
    targets = sorted(_discover_overlay_methods(ODS))
    if not targets:
        raise RuntimeError(
            "this OMAS release exposes no ODS.plot_*_overlay methods to wrap"
        )

    foreign = sorted(
        name
        for name in targets
        if name not in wrapped
        and getattr(getattr(ODS, name), "_vaft_overlay_wrapper", False)
    )
    if foreign and not overwrite:
        raise RuntimeError(
            "refusing to re-wrap already wrapped ODS methods: "
            + ", ".join(foreign)
            + ". Call vaft.omas.disable_overlay_methods() first, or pass "
            "overwrite=True."
        )

    for name in targets:
        if name in wrapped:
            continue
        setattr(ODS, name, _make_overlay_wrapper(getattr(ODS, name)))
    ODS._vaft_overlay_methods = frozenset(targets)
    return tuple(targets)


def disable_overlay_methods() -> None:
    """Restore the OMAS methods wrapped by :func:`enable_overlay_methods`."""
    from omas import ODS

    for name in getattr(ODS, "_vaft_overlay_methods", frozenset()):
        wrapper = getattr(ODS, name, None)
        original = getattr(wrapper, "__wrapped__", None)
        if original is not None:
            setattr(ODS, name, original)
    ODS._vaft_overlay_methods = frozenset()


def _discover_overlay_methods(ods_class: type) -> tuple[str, ...]:
    """Return every ``plot_*_overlay`` attribute OMAS exposes on ``ODS``."""
    return tuple(
        name
        for name in dir(ods_class)
        if name.startswith("plot_")
        and name.endswith("_overlay")
        and callable(getattr(ods_class, name, None))
    )


def _make_overlay_wrapper(original):
    import functools

    @functools.wraps(original)
    def wrapper(self, *args, ax=None, show=False, **options):
        from vaft.plot.style import finalize, resolve_axes

        figure, axes = resolve_axes(ax)
        result = original(self, *args, ax=axes, **options)
        finalize(figure, axes, show=show)
        return result

    wrapper._vaft_overlay_wrapper = True
    return wrapper


def _make_method(plot_name: str):
    spec = get_spec(plot_name)

    def method(self, *, ax=None, show=False, **options):
        return render(plot_name, self, ax=ax, show=show, **options)

    method.__name__ = f"plot_{plot_name}"
    method.__qualname__ = f"ODS.plot_{plot_name}"
    method.__doc__ = spec.description
    return method


def plot_barometry_time_pressure(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Neutral pressure history from the barometry gauges.

    Renders with :func:`vaft.plot.barometry_time_pressure`.
    """
    return render(
        "barometry_time_pressure", source, ax=ax, show=show, label=label, **options
    )


def plot_camera_visible_image_frame(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One FAST-camera frame, selected by ``frame_index=`` or nearest ``time=``.

    Renders with :func:`vaft.plot.camera_visible_image_frame`.
    """
    return render(
        "camera_visible_image_frame", source, ax=ax, show=show, label=label, **options
    )


def plot_camera_visible_image_efit_overlay(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """FAST-camera frame with the projected EFIT/wall overlay.

    Requires ``shot=`` (one of the calibrated shots: 34764, 39915, 47518).
    Renders with :func:`vaft.plot.camera_visible_image_efit_overlay`.
    """
    return render(
        "camera_visible_image_efit_overlay",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_camera_visible_image_field_line(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """FAST-camera frame with a projected traced magnetic field line.

    Requires ``shot=``, ``r0=``, ``z0=`` (the field-line start point, in
    meters). Renders with :func:`vaft.plot.camera_visible_image_field_line`.
    """
    return render(
        "camera_visible_image_field_line",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_camera_visible_animation_frames(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
):
    """Animate a sequence of FAST-camera frames on a shared color scale.

    Returns ``(Figure, Axes, FuncAnimation)``. Renders with
    :func:`vaft.plot.camera_visible_animation_frames`.
    """
    return render(
        "camera_visible_animation_frames",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_charge_exchange_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Charge-exchange measurement positions in the poloidal plane.

    Renders with :func:`vaft.plot.charge_exchange_geometry_poloidal`.
    """
    return render(
        "charge_exchange_geometry_poloidal",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_coils_non_axisymmetric_geometry3d(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Non-axisymmetric 3D coil filaments in machine Cartesian coordinates.

    Renders with :func:`vaft.plot.coils_non_axisymmetric_geometry3d`.
    """
    return render(
        "coils_non_axisymmetric_geometry3d",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_coils_non_axisymmetric_geometry_topview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Non-axisymmetric 3D coil filaments projected into the machine top view.

    Renders with :func:`vaft.plot.coils_non_axisymmetric_geometry_topview`.
    """
    return render(
        "coils_non_axisymmetric_geometry_topview",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_charge_exchange_profile_ion_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Charge-exchange ion temperature versus position.

    Renders with :func:`vaft.plot.charge_exchange_profile_ion_temperature`.
    """
    return render(
        "charge_exchange_profile_ion_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_charge_exchange_profile_velocity_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Charge-exchange toroidal rotation versus position.

    Renders with :func:`vaft.plot.charge_exchange_profile_velocity_tor`.
    """
    return render(
        "charge_exchange_profile_velocity_tor",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_charge_exchange_time_ion_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel ion temperature history from charge-exchange spectroscopy.

    Renders with :func:`vaft.plot.charge_exchange_time_ion_temperature`.
    """
    return render(
        "charge_exchange_time_ion_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_charge_exchange_time_velocity_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel toroidal rotation history from charge-exchange spectroscopy.

    Renders with :func:`vaft.plot.charge_exchange_time_velocity_tor`.
    """
    return render(
        "charge_exchange_time_velocity_tor",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_field_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Electron density mapped onto the poloidal plane.

    Renders with :func:`vaft.plot.core_profiles_field_electron_density`.
    """
    return render(
        "core_profiles_field_electron_density",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_field_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Electron temperature mapped onto the poloidal plane.

    Renders with :func:`vaft.plot.core_profiles_field_electron_temperature`.
    """
    return render(
        "core_profiles_field_electron_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_profile_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core electron density profile.

    Renders with :func:`vaft.plot.core_profiles_profile_electron_density`.
    """
    return render(
        "core_profiles_profile_electron_density",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_profile_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core electron temperature profile.

    Renders with :func:`vaft.plot.core_profiles_profile_electron_temperature`.
    """
    return render(
        "core_profiles_profile_electron_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_profile_ion_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core ion temperature profile.

    Renders with :func:`vaft.plot.core_profiles_profile_ion_temperature`.
    """
    return render(
        "core_profiles_profile_ion_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_profile_pressure(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Core total pressure profile.

    Renders with :func:`vaft.plot.core_profiles_profile_pressure`.
    """
    return render(
        "core_profiles_profile_pressure",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_time_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Volume-averaged electron density history.

    Renders with :func:`vaft.plot.core_profiles_time_electron_density`.
    """
    return render(
        "core_profiles_time_electron_density",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_time_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Volume-averaged electron temperature history.

    Renders with :func:`vaft.plot.core_profiles_time_electron_temperature`.
    """
    return render(
        "core_profiles_time_electron_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_core_profiles_time_volume_averaged(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Volume-averaged core quantity panels on a shared time axis.

    Renders with :func:`vaft.plot.core_profiles_time_volume_averaged`.
    """
    return render(
        "core_profiles_time_volume_averaged",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_electromagnetics_time_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Plasma, PF coil and eddy current panels on a shared time axis.

    Renders with :func:`vaft.plot.electromagnetics_time_current`.
    """
    return render(
        "electromagnetics_time_current",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_equilibrium_field_psi(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Reconstructed poloidal flux map on the equilibrium (R, Z) grid.

    Renders with :func:`vaft.plot.equilibrium_field_psi`.
    """
    return render(
        "equilibrium_field_psi", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_field_psi_vacuum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Vacuum poloidal flux from the PF coils alone, without plasma.

    Renders with :func:`vaft.plot.equilibrium_field_psi_vacuum`.
    """
    return render(
        "equilibrium_field_psi_vacuum", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_geometry_boundary(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Last-closed-flux-surface outline in the poloidal plane.

    Renders with :func:`vaft.plot.equilibrium_geometry_boundary`.
    """
    return render(
        "equilibrium_geometry_boundary",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_equilibrium_geometry_topview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium boundary projected into the machine top view.

    Renders with :func:`vaft.plot.equilibrium_geometry_topview`.
    """
    return render(
        "equilibrium_geometry_topview", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium analysis overview: global quantities plus poloidal geometry.

    Renders with :func:`vaft.plot.equilibrium_overview`.
    """
    return render(
        "equilibrium_overview", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_constraints(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT validation: the magnetic constraints actually submitted to the solver.

    Shows every channel of every family, with the enabled ones separated from the
    disabled and the missing, so a dead channel or a wrong weighting is visible
    before the reconstruction is interpreted. ``time_slice`` selects the slice.

    Renders with :func:`vaft.plot.equilibrium_overview_constraints`.
    """
    return render(
        "equilibrium_overview_constraints", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_constraint_coverage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT validation: how the fitted channel set changes across time slices.

    Flat lines mean a consistent constraint set; a step is channel/time
    misalignment.

    Renders with :func:`vaft.plot.equilibrium_overview_constraint_coverage`.
    """
    return render(
        "equilibrium_overview_constraint_coverage",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_chease_overview_refinement_summary(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """CHEASE validation: how far did refinement move the EFIT equilibrium?

    Profile and boundary RMS change, flux-normalization shift and plasma-current
    self-consistency, slice by slice -- read from `comparison_metrics`, embedded
    on `equilibrium.code.parameters` by the chease FileDB stage.

    Renders with :func:`vaft.plot.chease_overview_refinement_summary`.
    """
    return render(
        "chease_overview_refinement_summary", source, ax=ax, show=show, label=label, **options
    )


def plot_chease_overview_profile_validity(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """CHEASE validation: is the refined equilibrium itself physically sound?

    q0/q95, q-monotonicity and pressure positivity read straight off the
    refined time slices -- a converged solution that is not physical, flagged
    without needing the pre-refinement equilibrium at all.

    Renders with :func:`vaft.plot.chease_overview_profile_validity`.
    """
    return render(
        "chease_overview_profile_validity", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_fit_quality(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT validation: is the fit acceptable against the uncertainties EFIT was given?

    Reduced chi-square against EFIT's own degrees of freedom, which diagnostic
    family carries the chi-square, and per-channel residuals normalized by the
    uncertainty implied by EFIT's stored chi-square.

    Renders with :func:`vaft.plot.equilibrium_overview_fit_quality`.
    """
    return render(
        "equilibrium_overview_fit_quality", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_convergence(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT validation: is the solution converged and numerically self-consistent?

    Terminating successfully is not the same as converging: this shows the final
    Grad-Shafranov error against the tolerance that was requested, the iteration
    count against its cap, the error history where EFIT wrote one, and EFIT's own
    outputs checked against each other.

    Renders with :func:`vaft.plot.equilibrium_overview_convergence`.
    """
    return render(
        "equilibrium_overview_convergence", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_profiles(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """The principal 1-D equilibrium profiles side by side.

    Pressure, toroidal current density and safety factor are the three profiles
    that describe what the plasma is doing, in the order they are usually read.
    Drawing them together avoids the impression that any one of them
    characterises the equilibrium on its own.

    Renders with :func:`vaft.plot.equilibrium_overview_profiles`.
    """
    return render(
        "equilibrium_overview_profiles", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_residuals(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """EFIT validation: measured-minus-reconstructed residuals by diagnostic family.

    Convergence status is drawn beside the residuals rather than standing in for
    them: a converged solution with large residuals is still a bad one.

    Renders with :func:`vaft.plot.equilibrium_overview_residuals`.
    """
    return render(
        "equilibrium_overview_residuals", source, ax=ax, show=show, label=label, **options
    )


def plot_mhd_linear_time_energy_perturbed(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Linear MHD stability: DCON perturbed energy per toroidal mode against time.

    Traces are grouped by ``n_tor``, which is the only place the physical mode
    number lives -- ``toroidal_mode`` array position does not carry it.

    Renders with :func:`vaft.plot.mhd_linear_time_energy_perturbed`.
    """
    return render(
        "mhd_linear_time_energy_perturbed", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_overview_verification(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Render ``equilibrium_overview_verification`` from finalized EFIT data.

    Select the finalized equilibrium slice with ``time_slice=``.
    """
    return render(
        "equilibrium_overview_verification",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_equilibrium_profile_f(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium poloidal current function F = R*B_t.

    Renders with :func:`vaft.plot.equilibrium_profile_f`.
    """
    return render(
        "equilibrium_profile_f", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_profile_ffprime(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium F dF/dpsi profile.

    Renders with :func:`vaft.plot.equilibrium_profile_ffprime`.
    """
    return render(
        "equilibrium_profile_ffprime", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_profile_j_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium toroidal current-density profile.

    Renders with :func:`vaft.plot.equilibrium_profile_j_tor`.
    """
    return render(
        "equilibrium_profile_j_tor", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_profile_pprime(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium dp/dpsi profile.

    Renders with :func:`vaft.plot.equilibrium_profile_pprime`.
    """
    return render(
        "equilibrium_profile_pprime", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_profile_pressure(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium 1D pressure profile.

    Renders with :func:`vaft.plot.equilibrium_profile_pressure`.
    """
    return render(
        "equilibrium_profile_pressure", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_profile_q(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium safety-factor profile.

    Renders with :func:`vaft.plot.equilibrium_profile_q`.
    """
    return render(
        "equilibrium_profile_q", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_beta_n(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Normalized beta history.

    Renders with :func:`vaft.plot.equilibrium_time_beta_n`.
    """
    return render(
        "equilibrium_time_beta_n", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_beta_pol(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal beta history.

    Renders with :func:`vaft.plot.equilibrium_time_beta_pol`.
    """
    return render(
        "equilibrium_time_beta_pol", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_beta_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Toroidal beta history.

    Renders with :func:`vaft.plot.equilibrium_time_beta_tor`.
    """
    return render(
        "equilibrium_time_beta_tor", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_diamagnetic_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured versus reconstructed diamagnetic-flux constraint.

    Renders with :func:`vaft.plot.equilibrium_time_diamagnetic_flux`.
    """
    return render(
        "equilibrium_time_diamagnetic_flux",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_equilibrium_time_li(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Internal inductance li_3 history.

    Renders with :func:`vaft.plot.equilibrium_time_li`.
    """
    return render(
        "equilibrium_time_li", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_major_radius(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Geometric-axis major radius history.

    Renders with :func:`vaft.plot.equilibrium_time_major_radius`.
    """
    return render(
        "equilibrium_time_major_radius",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_equilibrium_time_plasma_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Reconstructed plasma current history.

    Renders with :func:`vaft.plot.equilibrium_time_plasma_current`.
    """
    return render(
        "equilibrium_time_plasma_current",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_equilibrium_time_q0(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Safety factor on axis.

    Renders with :func:`vaft.plot.equilibrium_time_q0`.
    """
    return render(
        "equilibrium_time_q0", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_q95(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Safety factor at the 95% flux surface.

    Renders with :func:`vaft.plot.equilibrium_time_q95`.
    """
    return render(
        "equilibrium_time_q95", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_qa(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Safety factor at the plasma edge.

    Renders with :func:`vaft.plot.equilibrium_time_qa`.
    """
    return render(
        "equilibrium_time_qa", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_virial(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Virial-estimate equilibrium quantities against the reconstruction.

    Renders with :func:`vaft.plot.equilibrium_time_virial`.
    """
    return render(
        "equilibrium_time_virial", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_w_mag(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Magnetic stored energy history.

    Renders with :func:`vaft.plot.equilibrium_time_w_mag`.
    """
    return render(
        "equilibrium_time_w_mag", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_w_mhd(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """MHD stored energy history.

    Renders with :func:`vaft.plot.equilibrium_time_w_mhd`.
    """
    return render(
        "equilibrium_time_w_mhd", source, ax=ax, show=show, label=label, **options
    )


def plot_equilibrium_time_w_tot(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Total stored energy history.

    Renders with :func:`vaft.plot.equilibrium_time_w_tot`.
    """
    return render(
        "equilibrium_time_w_tot", source, ax=ax, show=show, label=label, **options
    )


def plot_interferometer_time_n_e_line(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Interferometer line-integrated electron density history.

    Renders with :func:`vaft.plot.interferometer_time_n_e_line`.
    """
    return render("interferometer_time_n_e_line", source, ax=ax, show=show, label=label, **options)


def plot_interferometer_spectrum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one interferometer channel's line density.

    Renders with :func:`vaft.plot.interferometer_spectrum`.

    Reference slopes are entirely yours: pass ``reference_slopes=[-1.5, -2.0]``
    or :class:`~vaft.plot.models.ReferenceSlope` instances with your own labels.
    VAFT supplies none and reads no meaning into any value.
    """
    return render(
        "interferometer_spectrum", source, ax=ax, show=show, label=label, **options
    )


def plot_interferometer_spectrogram(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of one interferometer channel's line density.

    Renders with :func:`vaft.plot.interferometer_spectrogram`.
    """
    return render("interferometer_spectrogram", source, ax=ax, show=show, label=label, **options)


def plot_interferometer_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Interferometer overview: line density history and spectrogram.

    Renders with :func:`vaft.plot.interferometer_overview`.
    """
    return render("interferometer_overview", source, ax=ax, show=show, label=label, **options)


def plot_machine_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Composed poloidal machine view: wall, coils, passive structure and diagnostic positions in one axes.

    Renders with :func:`vaft.plot.machine_geometry_poloidal`.
    """
    return render(
        "machine_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


def plot_machine_geometry_topview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Composed machine top view: plasma extent plus launcher, antenna and pellet-injector geometry.

    Renders with :func:`vaft.plot.machine_geometry_topview`.
    """
    return render(
        "machine_geometry_topview", source, ax=ax, show=show, label=label, **options
    )


def plot_magnetics_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Flux-loop and B-field-probe positions in the poloidal plane.

    Renders with :func:`vaft.plot.magnetics_geometry_poloidal`.
    """
    return render(
        "magnetics_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


def plot_magnetics_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Shot diagnostic overview: current, field, flux and geometry panels.

    Renders with :func:`vaft.plot.magnetics_overview`.
    """
    return render(
        "magnetics_overview", source, ax=ax, show=show, label=label, **options
    )


def plot_magnetics_overview_impa(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """IMPA validation overview: raw voltages, compensated Bz and the 1/R position check.

    Renders with :func:`vaft.plot.magnetics_overview_impa`.
    """
    return render("magnetics_overview_impa", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_overview_vacuum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Eddy validation: measured against coil-only and coil+eddy synthetic magnetics.

    Forward-models the reconstructed vacuum current system at a representative
    set of B probes and flux loops. ``per_family`` sets how many channels of each
    family are drawn; ``channels`` selects ``(kind, index)`` pairs explicitly.

    Renders with :func:`vaft.plot.magnetics_overview_vacuum`.
    """
    return render(
        "magnetics_overview_vacuum", source, ax=ax, show=show, label=label, **options
    )


def plot_magnetics_overview_plasma_residual(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Eddy validation: the plasma signal left over after the vacuum response.

    A residual within the pre-plasma noise band before breakdown and emerging
    coherently at plasma-current onset is what a good eddy reconstruction looks
    like; a post-breakdown residual is the plasma and is expected.

    Renders with :func:`vaft.plot.magnetics_overview_plasma_residual`.
    """
    return render(
        "magnetics_overview_plasma_residual",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_magnetics_profile_impa_tf(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """IMPA measured field against probe radius with the 1/R toroidal-field model.

    Renders with :func:`vaft.plot.magnetics_profile_impa_tf`.
    """
    return render("magnetics_profile_impa_tf", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_time_impa_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Compensated internal Bz from the IMPA Hall-probe array.

    Renders with :func:`vaft.plot.magnetics_time_impa_field`.
    """
    return render("magnetics_time_impa_field", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_time_impa_voltage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Raw IMPA Hall-probe voltages, one trace per channel.

    Renders with :func:`vaft.plot.magnetics_time_impa_voltage`.
    """
    return render("magnetics_time_impa_voltage", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_spectrum_mirnov(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one Mirnov coil signal.

    Renders with :func:`vaft.plot.magnetics_spectrum_mirnov`.

    The spectrum is of the signal as stored.  A Mirnov coil measures ``dB/dt``,
    so this is the PSD of the derivative and its spectral index is that of ``B``
    plus two; integrate with
    :func:`vaft.process.magnetics.b_field_pol_probe_field` first if you want a
    magnetic-field spectrum.

    Reference slopes are entirely yours: pass ``reference_slopes=[-1.5, -2.0]``
    or :class:`~vaft.plot.models.ReferenceSlope` instances with your own labels.
    VAFT supplies none and reads no meaning into any value.
    """
    return render(
        "magnetics_spectrum_mirnov", source, ax=ax, show=show, label=label, **options
    )


def plot_magnetics_spectrogram_mirnov(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of one Mirnov coil signal.

    Renders with :func:`vaft.plot.magnetics_spectrogram_mirnov`.
    """
    return render(
        "magnetics_spectrogram_mirnov", source, ax=ax, show=show, label=label, **options
    )


def plot_magnetics_time_b_field_pol_probe_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal field measured by each selected B-field probe.

    Renders with :func:`vaft.plot.magnetics_time_b_field_pol_probe_field`.
    """
    return render(
        "magnetics_time_b_field_pol_probe_field",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_magnetics_time_diamagnetic_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured diamagnetic flux history.

    Renders with :func:`vaft.plot.magnetics_time_diamagnetic_flux`.
    """
    return render(
        "magnetics_time_diamagnetic_flux",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_magnetics_time_flux_loop_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal flux measured by each selected flux loop.

    Renders with :func:`vaft.plot.magnetics_time_flux_loop_flux`.
    """
    return render(
        "magnetics_time_flux_loop_flux",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_magnetics_time_flux_loop_voltage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Loop voltage measured by each selected flux loop.

    Renders with :func:`vaft.plot.magnetics_time_flux_loop_voltage`.
    """
    return render(
        "magnetics_time_flux_loop_voltage",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_magnetics_time_ip(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Measured plasma current history from the Rogowski coil.

    Renders with :func:`vaft.plot.magnetics_time_ip`.
    """
    return render("magnetics_time_ip", source, ax=ax, show=show, label=label, **options)


def plot_magnetics_time_limiter_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Three stacked VEST limiter-current histories (LC, UC and midplane).

    Current is derived from each IMAS-standard shunt voltage using its stored
    effective Pearson Model 411 V/I coefficient. Renders with
    :func:`vaft.plot.magnetics_time_limiter_current`.
    """
    return render(
        "magnetics_time_limiter_current",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_magnetics_time_mirnov_voltage(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Raw or preprocessed Mirnov coil voltage traces.

    Renders with :func:`vaft.plot.magnetics_time_mirnov_voltage`.
    """
    return render(
        "magnetics_time_mirnov_voltage",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_pf_active_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """PF coil outlines in the poloidal plane.

    Renders with :func:`vaft.plot.pf_active_geometry_poloidal`.
    """
    return render(
        "pf_active_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


def plot_pf_active_time_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-coil PF current history.

    Renders with :func:`vaft.plot.pf_active_time_current`.
    """
    return render(
        "pf_active_time_current", source, ax=ax, show=show, label=label, **options
    )


def plot_pf_active_time_current_turns(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-coil PF current multiplied by the signed turn count (ampere-turns).

    Renders with :func:`vaft.plot.pf_active_time_current_turns`.
    """
    return render(
        "pf_active_time_current_turns", source, ax=ax, show=show, label=label, **options
    )


def plot_pf_passive_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Passive conducting-structure loop outlines in the poloidal plane.

    Renders with :func:`vaft.plot.pf_passive_geometry_poloidal`.
    """
    return render(
        "pf_passive_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


def plot_soft_x_rays_geometry_lines_of_sight(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Soft X-ray detector lines of sight over the poloidal cross-section.

    Renders with :func:`vaft.plot.soft_x_rays_geometry_lines_of_sight`.
    """
    return render(
        "soft_x_rays_geometry_lines_of_sight",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_soft_x_rays_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Soft X-ray overview: lines of sight, signals and channel pattern.

    Renders with :func:`vaft.plot.soft_x_rays_overview`.
    """
    return render(
        "soft_x_rays_overview", source, ax=ax, show=show, label=label, **options
    )


def plot_soft_x_rays_spectrum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one soft X-ray channel.

    Renders with :func:`vaft.plot.soft_x_rays_spectrum`.

    Reference slopes are entirely yours: pass ``reference_slopes=[-1.5, -2.0]``
    or :class:`~vaft.plot.models.ReferenceSlope` instances with your own labels.
    VAFT supplies none and reads no meaning into any value.
    """
    return render(
        "soft_x_rays_spectrum", source, ax=ax, show=show, label=label, **options
    )


def plot_soft_x_rays_spectrogram(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of one soft X-ray channel.

    Renders with :func:`vaft.plot.soft_x_rays_spectrogram`.
    """
    return render(
        "soft_x_rays_spectrogram", source, ax=ax, show=show, label=label, **options
    )


def plot_soft_x_rays_time_power(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Soft X-ray channel signal history.

    Renders with :func:`vaft.plot.soft_x_rays_time_power`.
    """
    return render(
        "soft_x_rays_time_power", source, ax=ax, show=show, label=label, **options
    )


def plot_spectrometer_uv_time_impurity(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Impurity line-intensity panels against plasma current.

    Renders with :func:`vaft.plot.spectrometer_uv_time_impurity`.
    """
    return render(
        "spectrometer_uv_time_impurity",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_spectrometer_uv_time_intensity(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Processed spectral line intensity history.

    Renders with :func:`vaft.plot.spectrometer_uv_time_intensity`.
    """
    return render(
        "spectrometer_uv_time_intensity",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_summary_time_beta(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Poloidal, toroidal and normalized beta panels.

    Renders with :func:`vaft.plot.summary_time_beta`.
    """
    return render("summary_time_beta", source, ax=ax, show=show, label=label, **options)


def plot_summary_time_energy(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Stored-energy comparison panels across available estimates.

    Renders with :func:`vaft.plot.summary_time_energy`.
    """
    return render(
        "summary_time_energy", source, ax=ax, show=show, label=label, **options
    )


def plot_summary_time_power_balance(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Ohmic input, radiated and conducted power balance panels.

    Renders with :func:`vaft.plot.summary_time_power_balance`.
    """
    return render(
        "summary_time_power_balance", source, ax=ax, show=show, label=label, **options
    )


def plot_summary_time_voltage_consumption(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Loop-voltage and flux-consumption panels.

    Renders with :func:`vaft.plot.summary_time_voltage_consumption`.
    """
    return render(
        "summary_time_voltage_consumption",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_tf_time_b_field_tor(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Toroidal field history at the reference radius.

    Renders with :func:`vaft.plot.tf_time_b_field_tor`.
    """
    return render(
        "tf_time_b_field_tor", source, ax=ax, show=show, label=label, **options
    )


def plot_tf_time_b_field_tor_vacuum_r(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Vacuum toroidal field times major radius (B_t * R).

    Renders with :func:`vaft.plot.tf_time_b_field_tor_vacuum_r`.
    """
    return render(
        "tf_time_b_field_tor_vacuum_r", source, ax=ax, show=show, label=label, **options
    )


def plot_tf_time_coil_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """TF coil current history.

    Renders with :func:`vaft.plot.tf_time_coil_current`.
    """
    return render(
        "tf_time_coil_current", source, ax=ax, show=show, label=label, **options
    )


def plot_thomson_scattering_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Thomson-scattering measurement positions in the poloidal plane.

    Renders with :func:`vaft.plot.thomson_scattering_geometry_poloidal`.
    """
    return render(
        "thomson_scattering_geometry_poloidal",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_thomson_scattering_profile_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Thomson-scattering electron density versus position.

    Renders with :func:`vaft.plot.thomson_scattering_profile_electron_density`.
    """
    return render(
        "thomson_scattering_profile_electron_density",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_thomson_scattering_profile_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Thomson-scattering electron temperature versus position.

    Renders with :func:`vaft.plot.thomson_scattering_profile_electron_temperature`.
    """
    return render(
        "thomson_scattering_profile_electron_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_thomson_scattering_time_electron_density(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel Thomson electron density history.

    Renders with :func:`vaft.plot.thomson_scattering_time_electron_density`.
    """
    return render(
        "thomson_scattering_time_electron_density",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_thomson_scattering_time_electron_temperature(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-channel Thomson electron temperature history.

    Renders with :func:`vaft.plot.thomson_scattering_time_electron_temperature`.
    """
    return render(
        "thomson_scattering_time_electron_temperature",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
    )


def plot_wall_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """First-wall and limiter outline in the poloidal plane.

    Renders with :func:`vaft.plot.wall_geometry_poloidal`.
    """
    return render(
        "wall_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


__all__ = [
    "available_plots",
    "disable_overlay_methods",
    "disable_plot_methods",
    "enable_overlay_methods",
    "enable_plot_methods",
    "extract_labels_from_odc",
    "normalize_entries",
    "plot_barometry_time_pressure",
    "plot_camera_visible_animation_frames",
    "plot_camera_visible_image_efit_overlay",
    "plot_camera_visible_image_field_line",
    "plot_camera_visible_image_frame",
    "plot_charge_exchange_geometry_poloidal",
    "plot_charge_exchange_profile_ion_temperature",
    "plot_coils_non_axisymmetric_geometry3d",
    "plot_coils_non_axisymmetric_geometry_topview",
    "plot_charge_exchange_profile_velocity_tor",
    "plot_charge_exchange_time_ion_temperature",
    "plot_charge_exchange_time_velocity_tor",
    "plot_chease_overview_profile_validity",
    "plot_chease_overview_refinement_summary",
    "plot_core_profiles_field_electron_density",
    "plot_core_profiles_field_electron_temperature",
    "plot_core_profiles_profile_electron_density",
    "plot_core_profiles_profile_electron_temperature",
    "plot_core_profiles_profile_ion_temperature",
    "plot_core_profiles_profile_pressure",
    "plot_core_profiles_time_electron_density",
    "plot_core_profiles_time_electron_temperature",
    "plot_core_profiles_time_volume_averaged",
    "plot_electromagnetics_time_current",
    "plot_equilibrium_field_psi",
    "plot_equilibrium_field_psi_vacuum",
    "plot_equilibrium_geometry_boundary",
    "plot_equilibrium_geometry_topview",
    "plot_equilibrium_overview",
    "plot_equilibrium_overview_constraint_coverage",
    "plot_equilibrium_overview_constraints",
    "plot_equilibrium_overview_convergence",
    "plot_equilibrium_overview_fit_quality",
    "plot_equilibrium_overview_profiles",
    "plot_equilibrium_overview_residuals",
    "plot_equilibrium_overview_verification",
    "plot_equilibrium_profile_f",
    "plot_equilibrium_profile_ffprime",
    "plot_equilibrium_profile_j_tor",
    "plot_equilibrium_profile_pprime",
    "plot_equilibrium_profile_pressure",
    "plot_equilibrium_profile_q",
    "plot_equilibrium_time_beta_n",
    "plot_equilibrium_time_beta_pol",
    "plot_equilibrium_time_beta_tor",
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
    "plot_interferometer_overview",
    "plot_interferometer_spectrogram",
    "plot_interferometer_spectrum",
    "plot_interferometer_time_n_e_line",
    "plot_machine_geometry_poloidal",
    "plot_machine_geometry_topview",
    "plot_magnetics_geometry_poloidal",
    "plot_magnetics_overview",
    "plot_magnetics_overview_impa",
    "plot_magnetics_overview_plasma_residual",
    "plot_magnetics_overview_vacuum",
    "plot_mhd_linear_time_energy_perturbed",
    "plot_magnetics_profile_impa_tf",
    "plot_magnetics_time_impa_field",
    "plot_magnetics_time_impa_voltage",
    "plot_magnetics_spectrogram_mirnov",
    "plot_magnetics_spectrum_mirnov",
    "plot_magnetics_time_b_field_pol_probe_field",
    "plot_magnetics_time_diamagnetic_flux",
    "plot_magnetics_time_flux_loop_flux",
    "plot_magnetics_time_flux_loop_voltage",
    "plot_magnetics_time_ip",
    "plot_magnetics_time_limiter_current",
    "plot_magnetics_time_mirnov_voltage",
    "plot_pf_active_geometry_poloidal",
    "plot_pf_active_time_current",
    "plot_pf_active_time_current_turns",
    "plot_pf_passive_geometry_poloidal",
    "plot_soft_x_rays_geometry_lines_of_sight",
    "plot_soft_x_rays_overview",
    "plot_soft_x_rays_spectrogram",
    "plot_soft_x_rays_spectrum",
    "plot_soft_x_rays_time_power",
    "plot_spectrometer_uv_time_impurity",
    "plot_spectrometer_uv_time_intensity",
    "plot_summary_time_beta",
    "plot_summary_time_energy",
    "plot_summary_time_power_balance",
    "plot_summary_time_voltage_consumption",
    "plot_tf_time_b_field_tor",
    "plot_tf_time_b_field_tor_vacuum_r",
    "plot_tf_time_coil_current",
    "plot_thomson_scattering_geometry_poloidal",
    "plot_thomson_scattering_profile_electron_density",
    "plot_thomson_scattering_profile_electron_temperature",
    "plot_thomson_scattering_time_electron_density",
    "plot_thomson_scattering_time_electron_temperature",
    "plot_wall_geometry_poloidal",
    "render",
]
