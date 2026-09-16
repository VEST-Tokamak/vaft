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

Every ``plot_<stem>`` has two twins (umbrella #434): ``dd_<stem>()`` lists the
IMAS Data Dictionary paths it reads without touching data, and
``extract_<stem>(source, *, label="shot", **extraction_options)`` returns the
view model the plot draws, undrawn -- ``.to_xarray()`` on it gives an
:class:`xarray.Dataset`.  A rendering keyword (``ax=``, ``cmap=``) is refused
by ``extract_*``.

Use :func:`available_plots` to see which plots a particular object can produce,
and :func:`enable_plot_methods` to opt in to ``ODS.plot_*`` methods.
:func:`enable_overlay_methods` does the same for OMAS' own
``ODS.plot_*_overlay`` methods, giving them the ``ax``/``show`` contract.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

from vaft.plot.backend.render import render_entries
from vaft.plot.registry import get_spec, specs
from vaft.plot._migration import (
    RENAMED_REMOVAL_RELEASE as _RENAMED_REMOVAL_RELEASE,
)

from .interactive import (  # public entry points, issues #261 and #482
    plot_diagnostics_time_interactive,
    plot_equilibrium_interactive,
)
from .entries import extract_labels_from_odc, normalize_entries




def render(
    name: str,
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> Any:
    """Build the view model for ``name`` from ``source`` and render it.

    This is the shared body behind every ``plot_*`` adapter below, and the entry
    point for rendering a canonical plot chosen at runtime.  Input handling is
    this namespace's (:func:`normalize_entries`); everything after that is the
    backend-neutral :func:`vaft.plot.backend.render.render_entries`.

    ``backend="plotly"`` among the options draws with Plotly and returns a
    :class:`plotly.graph_objects.Figure` (no ``ax=``); the default is
    Matplotlib's ``(Figure, Axes)``.  See :mod:`vaft.plot.backends`.
    """
    return render_entries(
        name, normalize_entries(source, label=label), ax=ax, show=show,
        namespace="vaft.omas", subject="ods", **options,
    )


def available_plots(
    source: Any = None,
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
):
    """What can be plotted, as a semantic catalog (issue #262).

    Without ``source`` this is :func:`vaft.plot.available_plots` plus what the
    recipes declare: display units, layouts, analysis methods, overview
    members.  With an ``ODS``, ``ODC`` or list, the catalog holds the plots
    whose required data is actually present -- decided by the same test
    :func:`render` applies -- and, for multi-channel plots, the channel counts,
    regions and representatives the selection policy finds there.  Pass
    ``available_only=False`` to keep the unavailable plots with their reasons.

    The catalog prints as a tree; iterate it for records, which still answer
    ``row["name"]`` like the flat rows they replace.
    """
    from .discovery import describe

    return describe(
        source, query=query, detail=detail, available_only=available_only, **filters
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
        current = getattr(ODS, name)
        # Never wrap a wrapper: on the overwrite path `current` is already one
        # of ours, and nesting would leave disable_overlay_methods() restoring
        # the inner wrapper instead of OMAS' own method.
        if getattr(current, "_vaft_overlay_wrapper", False):
            current = getattr(current, "__wrapped__", current)
        setattr(ODS, name, _make_overlay_wrapper(current))
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
    """Return every ``plot_*_overlay`` attribute OMAS exposes on ``ODS``.

    VAFT's own canonical adapters are excluded even though one of them
    (``plot_camera_visible_image_efit_overlay``) matches the name pattern: they
    already implement the ax/show contract, so wrapping them would resolve the
    axes twice, discard the renderer's ``figsize`` and run ``finalize`` twice.
    The exclusion is by canonical name rather than by whether
    :func:`enable_plot_methods` happens to have run, so the two opt-ins are
    order-independent.

    OMAS' aggregate ``plot_overlay`` dispatcher matches the pattern too but is
    excluded: it forwards to the individual overlays (which are wrapped), and
    its ``return_overlay_list=True`` query path draws nothing, so wrapping it
    would leak a blank figure per query.
    """
    canonical = {f"plot_{spec.name}" for spec in specs()}
    return tuple(
        name
        for name in dir(ods_class)
        if name.startswith("plot_")
        and name.endswith("_overlay")
        and name != "plot_overlay"
        and name not in canonical
        and callable(getattr(ods_class, name, None))
    )


def _make_overlay_wrapper(original):
    import functools
    import inspect

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):  # pragma: no cover - builtins have none
        signature = None

    @functools.wraps(original)
    def wrapper(self, *args, **options):
        from vaft.plot.style import finalize, resolve_axes

        show = options.pop("show", False)
        # OMAS declares `ax` as an ordinary positional-or-keyword parameter, so
        # `ods.plot_wall_overlay(my_ax)` is a legal call. Bind through the
        # wrapped signature instead of assuming `ax` arrives as a keyword.
        bound = None
        if signature is not None:
            try:
                bound = signature.bind(self, *args, **options)
            except TypeError:
                bound = None
        ax = bound.arguments.get("ax") if bound is not None else options.get("ax")

        figure, axes = resolve_axes(ax)
        if bound is not None:
            bound.arguments["ax"] = axes
            result = original(*bound.args, **bound.kwargs)
        else:
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


def plot_camera_visible_image_fluctuation(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One FAST-camera frame with its local temporal background removed.

    The published step that brings fast filamentary structure out of the slowly
    varying line emission (issue #161).  ``background_frames=`` sets the window.

    Renders with :func:`vaft.plot.camera_visible_image_fluctuation`.
    """
    return render(
        "camera_visible_image_fluctuation", source, ax=ax, show=show, label=label, **options
    )


def plot_camera_visible_image_mhd_power(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Per-pixel MHD-band power normalised by the local average emission.

    The published spectrally filtered image: a band magnitude, not an
    inverse-transform reconstruction (issue #161).  ``centre_frequency=`` names the
    band the magnetics report; without it the camera's own dominant component stands in.

    Renders with :func:`vaft.plot.camera_visible_image_mhd_power`.
    """
    return render(
        "camera_visible_image_mhd_power", source, ax=ax, show=show, label=label, **options
    )


def plot_camera_visible_spectrogram(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time-frequency map of the camera intensity summed over one image region.

    The camera side of the published camera/magnetics comparison (issue #161).
    ``region=(row_start, row_stop, column_start, column_stop)`` chooses what is summed.

    Renders with :func:`vaft.plot.camera_visible_spectrogram`.
    """
    return render(
        "camera_visible_spectrogram", source, ax=ax, show=show, label=label, **options
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


def plot_equilibrium_field_2d(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> Any:
    """One reconstructed 2-D equilibrium quantity on the (R, Z) grid (issue #483).

    ``field=`` chooses it -- ``psi`` (default), ``j_tor``, ``pressure``,
    ``b_field_r``, ``b_field_z``, ``b_field_tor`` -- deriving what the slice
    does not store on a private copy; ``overlay=`` chooses what is drawn over
    it from ``coils``, ``passive``, ``wall``, ``boundary``, ``axis``.
    Renders with :func:`vaft.plot.equilibrium_field_2d` from OMAS input.
    """
    return render("equilibrium_field_2d", source, ax=ax, show=show, label=label, **options)


def plot_vacuum_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One quantity of the coils' and vessel's vacuum field, at one instant.

    ``field=`` chooses among ``psi``, ``b_poloidal``, ``decay_index`` and
    ``breakdown``; ``time_index=`` steps along the PF time base.

    Renders with :func:`vaft.plot.vacuum_field`.
    """
    return render(
        "vacuum_field", source, ax=ax, show=show, label=label, **options
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


def plot_ntms_time_delta_prime(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Classical tearing index per rational surface against time.

    RDCON's and STRIDE's physical result. A trace is one rational surface --
    an ``(m_pol, n_tor)`` pair the solver located in the equilibrium -- not one
    of the toroidal modes the caller requested, so several traces can share an
    ``n_tor``. A positive index is a tearing-unstable surface, which is the
    opposite convention to DCON's perturbed energy.

    Renders with :func:`vaft.plot.ntms_time_delta_prime`.
    """
    return render(
        "ntms_time_delta_prime", source, ax=ax, show=show, label=label, **options
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


def plot_mhd_linear_profile_displacement(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Linear MHD stability: DCON displacement eigenfunction per poloidal harmonic.

    One trace per poloidal mode number against normalized flux, for the
    least-stable mapped ``(time_slice, n_tor)`` cell unless ``time_slice`` or
    ``n_tor`` names one.  Amplitudes are normalized to the peak: DCON's
    eigenvector normalization is arbitrary, so only the shape and the relative
    harmonic content are meaningful.

    Renders with :func:`vaft.plot.mhd_linear_profile_displacement`.
    """
    return render(
        "mhd_linear_profile_displacement", source, ax=ax, show=show, label=label, **options
    )


def plot_mhd_linear_profile_b_field_perturbed(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Linear MHD stability: normal perturbed field per poloidal harmonic.

    The field DCON derives from its own eigenfunction as ``i(m - nq) xi``, so it
    vanishes on each resonant surface and carries the same arbitrary
    normalization as the displacement.

    Renders with :func:`vaft.plot.mhd_linear_profile_b_field_perturbed`.
    """
    return render(
        "mhd_linear_profile_b_field_perturbed", source, ax=ax, show=show, label=label, **options
    )


def plot_mhd_linear_profile_resonant_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Linear MHD stability: pitch-resonant flux per rational surface.

    Not read from the IDS -- there is no IMAS slot for it -- but derived from
    the mapped perturbed flux by the jump across each singular surface, with
    the surface geometry the ideal-GPEC mapper recorded in ``code.parameters``.
    The derivation runs once per figure, not once per trace.

    Renders with :func:`vaft.plot.mhd_linear_profile_resonant_flux`.
    """
    return render(
        "mhd_linear_profile_resonant_flux", source, ax=ax, show=show, label=label, **options
    )


def plot_mhd_linear_profile_island_width(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Linear MHD stability: saturated island width per rational surface.

    In normalized poloidal flux, as GPEC reports it -- converting to metres
    needs the equilibrium's ``dr/dpsi_N`` and is not done here. Derived from
    the resonant flux above.

    Renders with :func:`vaft.plot.mhd_linear_profile_island_width`.
    """
    return render(
        "mhd_linear_profile_island_width", source, ax=ax, show=show, label=label, **options
    )


def plot_mhd_linear_overview_eigenfunction(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Linear MHD stability: the least-stable mode's eigenfunction in one figure.

    Displacement beside normal perturbed field, sharing a flux axis.

    Renders with :func:`vaft.plot.mhd_linear_overview_eigenfunction`.
    """
    return render(
        "mhd_linear_overview_eigenfunction", source, ax=ax, show=show, label=label, **options
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


def plot_camera_visible_image(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """One camera frame, optionally with overlays through one projection.

    ``overlay=`` names what to draw over the frame -- ``"wall"``,
    ``"equilibrium"`` (LCFS, magnetic axis, flux surfaces), ``"field_line"``
    (needs ``field_line_start=(r0, z0[, phi0])``) or a tuple of them;
    ``projection=`` is ``"calibrated"`` (the model packaged for the shot) or a
    :class:`vaft.process.camera_geometry.CameraProjection`.  The
    ``plot_camera_visible_image_*`` functions are presets of this one.
    Renders with :func:`vaft.plot.camera_visible_image` (issue #261).
    """
    return render("camera_visible_image", source, ax=ax, show=show, label=label, **options)


def plot_equilibrium_overview_histories(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Equilibrium global quantities against time: Ip, beta_p, li, q95.

    This is the composite ``plot_equilibrium_overview`` drew before issue #261
    made that name a one-slice summary.  Renders with
    :func:`vaft.plot.equilibrium_overview_histories`.
    """
    return render(
        "equilibrium_overview_histories", source, ax=ax, show=show, label=label, **options
    )


def plot_diagnostics_overview(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Time overview across the diagnostic subjects, one panel per available member.

    A diagnostic the input lacks is left out and the grid shrinks (issue
    #476); ``members=`` picks the panels by name, and ``interactive=True``
    -- or :func:`plot_diagnostics_time_interactive` -- offers that and the
    presets every panel honours as controls (issue #482).

    Renders with :func:`vaft.plot.diagnostics_overview`.
    """
    return render(
        "diagnostics_overview", source, ax=ax, show=show, label=label, **options
    )


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


def plot_impa_time_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Compensated internal Bz from the IMPA Hall-probe array.

    Renders with :func:`vaft.plot.impa_time_field`.
    """
    return render("impa_time_field", source, ax=ax, show=show, label=label, **options)


def plot_mirnov_spectrum(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Power spectral density of one Mirnov coil signal.

    Renders with :func:`vaft.plot.mirnov_spectrum`.

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
        "mirnov_spectrum", source, ax=ax, show=show, label=label, **options
    )


def plot_flux_loop_spatial_flux(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> Any:
    """Flux-loop flux against sensor position at one time (issue #486).

    ``time=`` snaps to the nearest stored sample (``time_slice=`` maps
    through a stored equilibrium slice); ``coordinate="z"`` (default) draws
    the inboard and outboard loops as two panels, ``"theta"`` one panel
    against the poloidal angle about the layout centre (``centre=``).
    Renders with :func:`vaft.plot.flux_loop_spatial_flux` from OMAS input.
    """
    return render("flux_loop_spatial_flux", source, ax=ax, show=show, label=label, **options)


def plot_mirnov_spatial_phase(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> Any:
    """Toroidal phase of each fluctuation band at one time, with the fitted n lines.

    ``time=`` snaps to a stored sample; ``frequencies=`` names the bands (the
    strongest ``num_modes=`` are chosen otherwise); ``show_fit=False`` draws
    the measured points alone.  Needs two probes at distinct toroidal angles
    that both recorded a waveform, which ``available_plots`` states.
    Renders with :func:`vaft.plot.mirnov_spatial_phase` from OMAS input.
    """
    return render("mirnov_spatial_phase", source, ax=ax, show=show, label=label, **options)


def plot_b_field_probe_spatial_field(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> Any:
    """B-probe poloidal field against sensor position at one time (issue #486).

    Same options as :func:`plot_flux_loop_spatial_flux`; ``angle="stored"``
    reads each probe's IMAS ``poloidal_angle`` instead of the geometric one.
    Renders with :func:`vaft.plot.b_field_probe_spatial_field` from OMAS input.
    """
    return render("b_field_probe_spatial_field", source, ax=ax, show=show, label=label, **options)


def plot_limiter_current_time(
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
    :func:`vaft.plot.limiter_current_time`.
    """
    return render(
        "limiter_current_time",
        source,
        ax=ax,
        show=show,
        label=label,
        **options,
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

    Renders with :func:`vaft.plot.pf_coil_geometry_poloidal`.
    """
    return render(
        "pf_coil_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


def plot_passive_structure_time_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Eddy current induced in the passive structure.

    Summed over loops by default, because VEST's vessel is discretised into 950
    of them and the sum is what balances against the coil currents.  Pass
    ``channels=`` to inspect individual loops.

    Requires the eddy currents to have been solved --
    :func:`vaft.omas.compute_eddy_currents` writes ``pf_passive.time``.

    Renders with :func:`vaft.plot.passive_structure_time_current`.
    """
    return render(
        "passive_structure_time_current", source, ax=ax, show=show, label=label, **options
    )


def plot_passive_structure_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Passive conducting-structure loop outlines in the poloidal plane.

    Renders with :func:`vaft.plot.passive_structure_geometry_poloidal`.
    """
    return render(
        "passive_structure_geometry_poloidal", source, ax=ax, show=show, label=label, **options
    )


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
    :func:`vaft.plot.passive_structure_geometry_wall_mode`.
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
    :func:`vaft.plot.passive_structure_overview_wall_reduction`.
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
    with :func:`vaft.plot.passive_structure_field_wall_reduction`.
    """
    return render(
        "passive_structure_field_wall_reduction", source, ax=ax, show=show, label=label, **options
    )


def plot_pf_plasma_geometry_poloidal(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """The plasma-current elements of ``pf_plasma`` coloured by current.

    Options: ``time`` (instant; default the largest total current).
    Renders with :func:`vaft.plot.pf_plasma_geometry_poloidal`.
    """
    return render("pf_plasma_geometry_poloidal", source, ax=ax, show=show, label=label, **options)


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
    :func:`vaft.plot.passive_structure_overview_wall_time`.
    """
    return render(
        "passive_structure_overview_wall_time", source, ax=ax, show=show, label=label, **options
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


def _renamed_adapter(old_stem: str, new_stem: str):
    def adapter(
        source: Any,
        *,
        ax: Any = None,
        show: bool = False,
        label: str | Sequence[str] = "shot",
        **options: Any,
    ) -> tuple[Any, Any]:
        warnings.warn(
            f"vaft.omas.plot_{old_stem} was renamed to "
            f"vaft.omas.plot_{new_stem} in the subject-taxonomy redesign "
            f"(issue #251); the old name is removed in "
            f"{_RENAMED_REMOVAL_RELEASE}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return render(new_stem, source, ax=ax, show=show, label=label, **options)

    adapter.__name__ = f"plot_{old_stem}"
    adapter.__qualname__ = f"plot_{old_stem}"
    adapter.__doc__ = (
        f"Deprecated alias of :func:`plot_{new_stem}` (renamed by issue #251)."
    )
    return adapter


plot_coils_non_axisymmetric_geometry3d = _renamed_adapter("coils_non_axisymmetric_geometry3d", "coil_3d_geometry3d")
plot_coils_non_axisymmetric_geometry_topview = _renamed_adapter("coils_non_axisymmetric_geometry_topview", "coil_3d_geometry_topview")
plot_core_profiles_field_electron_density = _renamed_adapter("core_profiles_field_electron_density", "electron_density_field")
plot_core_profiles_field_electron_temperature = _renamed_adapter("core_profiles_field_electron_temperature", "electron_temperature_field")
plot_core_profiles_profile_electron_density = _renamed_adapter("core_profiles_profile_electron_density", "electron_density_profile")
plot_core_profiles_profile_electron_temperature = _renamed_adapter("core_profiles_profile_electron_temperature", "electron_temperature_profile")
plot_core_profiles_profile_ion_temperature = _renamed_adapter("core_profiles_profile_ion_temperature", "ion_temperature_profile")
plot_core_profiles_profile_pressure = _renamed_adapter("core_profiles_profile_pressure", "thermal_pressure_profile")
plot_core_profiles_time_electron_density = _renamed_adapter("core_profiles_time_electron_density", "electron_density_time")
plot_core_profiles_time_electron_temperature = _renamed_adapter("core_profiles_time_electron_temperature", "electron_temperature_time")
plot_electromagnetics_time_current = _renamed_adapter("electromagnetics_time_current", "current_overview")
plot_equilibrium_time_beta_pol = _renamed_adapter("equilibrium_time_beta_pol", "equilibrium_time_beta_p")
plot_equilibrium_time_beta_tor = _renamed_adapter("equilibrium_time_beta_tor", "equilibrium_time_beta_t")
plot_magnetics_overview_impa = _renamed_adapter("magnetics_overview_impa", "impa_overview")
plot_magnetics_profile_impa_tf = _renamed_adapter("magnetics_profile_impa_tf", "impa_profile_field")
plot_magnetics_spectrogram_mirnov = _renamed_adapter("magnetics_spectrogram_mirnov", "mirnov_spectrogram")
plot_magnetics_spectrum_mirnov = _renamed_adapter("magnetics_spectrum_mirnov", "mirnov_spectrum")
plot_magnetics_time_b_field_pol_probe_field = _renamed_adapter("magnetics_time_b_field_pol_probe_field", "b_field_probe_time_field")
plot_magnetics_time_diamagnetic_flux = _renamed_adapter("magnetics_time_diamagnetic_flux", "diamagnetic_flux_time")
plot_magnetics_time_flux_loop_flux = _renamed_adapter("magnetics_time_flux_loop_flux", "flux_loop_time_flux")
plot_magnetics_time_flux_loop_voltage = _renamed_adapter("magnetics_time_flux_loop_voltage", "flux_loop_time_voltage")
plot_magnetics_time_impa_field = _renamed_adapter("magnetics_time_impa_field", "impa_time_field")
plot_magnetics_time_impa_voltage = _renamed_adapter("magnetics_time_impa_voltage", "impa_time_voltage")
plot_magnetics_time_ip = _renamed_adapter("magnetics_time_ip", "plasma_current_time")
plot_magnetics_time_limiter_current = _renamed_adapter("magnetics_time_limiter_current", "limiter_current_time")
plot_magnetics_time_mirnov_voltage = _renamed_adapter("magnetics_time_mirnov_voltage", "mirnov_time_voltage")
plot_pf_active_geometry_poloidal = _renamed_adapter("pf_active_geometry_poloidal", "pf_coil_geometry_poloidal")
plot_pf_active_time_current = _renamed_adapter("pf_active_time_current", "pf_coil_time_current")
plot_pf_active_time_current_turns = _renamed_adapter("pf_active_time_current_turns", "pf_coil_time_current_turns")
plot_pf_passive_geometry_poloidal = _renamed_adapter("pf_passive_geometry_poloidal", "passive_structure_geometry_poloidal")
plot_summary_time_beta = _renamed_adapter("summary_time_beta", "equilibrium_time_beta")
plot_tf_time_b_field_tor = _renamed_adapter("tf_time_b_field_tor", "tf_coil_time_b_t")
plot_tf_time_b_field_tor_vacuum_r = _renamed_adapter("tf_time_b_field_tor_vacuum_r", "tf_coil_time_b_t_vacuum_r")
plot_tf_time_coil_current = _renamed_adapter("tf_time_coil_current", "tf_coil_time_current")


def plot_nbi_profile_electron_heating(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Beam power density to electrons against normalized toroidal flux.

    Reads a NUBEAM result mapped into ``core_sources`` by
    :func:`vaft.machine_mapping.core_sources.core_sources_from_nubeam`; the NBI
    entry is found by its identifier, or named with ``source=``.

    Renders with :func:`vaft.plot.nbi_profile_electron_heating`.
    """
    return render("nbi_profile_electron_heating", source, ax=ax, show=show, label=label, **options)


def plot_nbi_profile_ion_heating(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Beam power density to ions against normalized toroidal flux.

    Reads a NUBEAM result mapped into ``core_sources`` by
    :func:`vaft.machine_mapping.core_sources.core_sources_from_nubeam`; the NBI
    entry is found by its identifier, or named with ``source=``.

    Renders with :func:`vaft.plot.nbi_profile_ion_heating`.
    """
    return render("nbi_profile_ion_heating", source, ax=ax, show=show, label=label, **options)


def plot_nbi_profile_current_drive(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Beam-driven parallel current density against normalized toroidal flux.

    Reads a NUBEAM result mapped into ``core_sources`` by
    :func:`vaft.machine_mapping.core_sources.core_sources_from_nubeam`; the NBI
    entry is found by its identifier, or named with ``source=``.

    Renders with :func:`vaft.plot.nbi_profile_current_drive`.
    """
    return render("nbi_profile_current_drive", source, ax=ax, show=show, label=label, **options)


def plot_neoclassical_profile_bootstrap_current(
    source: Any,
    *,
    ax: Any = None,
    show: bool = False,
    label: str | Sequence[str] = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Bootstrap current density from each neoclassical model on one radial axis.

    The Sauter and Redl formulas against whatever solver result the ODS
    carries (:func:`vaft.validation.neoclassical.bootstrap_models`), one
    series per model, so the models are compared on one radial axis.

    Renders with :func:`vaft.plot.neoclassical_profile_bootstrap_current`.
    """
    return render("neoclassical_profile_bootstrap_current", source, ax=ax, show=show, label=label, **options)

__all__ = [
    "available_plots",
    "disable_overlay_methods",
    "disable_plot_methods",
    "enable_overlay_methods",
    "enable_plot_methods",
    "extract_labels_from_odc",
    "normalize_entries",
    "plot_camera_visible_animation_frames",
    "plot_camera_visible_image",
    "plot_camera_visible_image_fluctuation",
    "plot_neoclassical_profile_bootstrap_current",
    "plot_camera_visible_image_mhd_power",
    "plot_camera_visible_spectrogram",
    "plot_camera_visible_image_efit_overlay",
    "plot_camera_visible_image_field_line",
    "plot_camera_visible_image_frame",
    "plot_chease_overview_profile_validity",
    "plot_chease_overview_refinement_summary",
    "plot_equilibrium_field_2d",
    "plot_vacuum_field",
    "plot_equilibrium_overview",
    "plot_equilibrium_overview_constraint_coverage",
    "plot_equilibrium_overview_constraints",
    "plot_equilibrium_overview_convergence",
    "plot_equilibrium_overview_fit_quality",
    "plot_equilibrium_interactive",
    "plot_diagnostics_time_interactive",
    "plot_equilibrium_overview_histories",
    "plot_equilibrium_overview_profiles",
    "plot_equilibrium_overview_residuals",
    "plot_equilibrium_overview_verification",
    "plot_interferometer_spectrum",
    "plot_machine_geometry_poloidal",
    "plot_machine_geometry_topview",
    "plot_magnetics_overview_plasma_residual",
    "plot_magnetics_overview_vacuum",
    "plot_mhd_linear_overview_eigenfunction",
    "plot_mhd_linear_profile_b_field_perturbed",
    "plot_mhd_linear_profile_island_width",
    "plot_mhd_linear_profile_resonant_flux",
    "plot_mhd_linear_profile_displacement",
    "plot_nbi_profile_current_drive",
    "plot_nbi_profile_electron_heating",
    "plot_nbi_profile_ion_heating",
    "plot_mhd_linear_time_energy_perturbed",
    "plot_ntms_time_delta_prime",
    "plot_impa_time_field",
    "plot_mirnov_spectrum",
    "plot_diagnostics_overview",
    "plot_flux_loop_spatial_flux",
    "plot_mirnov_spatial_phase",
    "plot_b_field_probe_spatial_field",
    "plot_limiter_current_time",
    "plot_passive_structure_time_current",
    "plot_pf_coil_geometry_poloidal",
    "plot_passive_structure_geometry_poloidal",
    "plot_passive_structure_field_wall_reduction",
    "plot_passive_structure_geometry_wall_mode",
    "plot_passive_structure_overview_wall_reduction",
    "plot_passive_structure_overview_wall_time",
    "plot_pf_plasma_geometry_poloidal",
    "plot_soft_x_rays_spectrum",
    "render",
    # deprecated renamed adapters (issue #251):
    "plot_coils_non_axisymmetric_geometry3d",
    "plot_coils_non_axisymmetric_geometry_topview",
    "plot_core_profiles_field_electron_density",
    "plot_core_profiles_field_electron_temperature",
    "plot_core_profiles_profile_electron_density",
    "plot_core_profiles_profile_electron_temperature",
    "plot_core_profiles_profile_ion_temperature",
    "plot_core_profiles_profile_pressure",
    "plot_core_profiles_time_electron_density",
    "plot_core_profiles_time_electron_temperature",
    "plot_electromagnetics_time_current",
    "plot_equilibrium_time_beta_pol",
    "plot_equilibrium_time_beta_tor",
    "plot_magnetics_overview_impa",
    "plot_magnetics_profile_impa_tf",
    "plot_magnetics_spectrogram_mirnov",
    "plot_magnetics_spectrum_mirnov",
    "plot_magnetics_time_b_field_pol_probe_field",
    "plot_magnetics_time_diamagnetic_flux",
    "plot_magnetics_time_flux_loop_flux",
    "plot_magnetics_time_flux_loop_voltage",
    "plot_magnetics_time_impa_field",
    "plot_magnetics_time_impa_voltage",
    "plot_magnetics_time_ip",
    "plot_magnetics_time_limiter_current",
    "plot_magnetics_time_mirnov_voltage",
    "plot_pf_active_geometry_poloidal",
    "plot_pf_active_time_current",
    "plot_pf_active_time_current_turns",
    "plot_pf_passive_geometry_poloidal",
    "plot_summary_time_beta",
    "plot_tf_time_b_field_tor",
    "plot_tf_time_b_field_tor_vacuum_r",
    "plot_tf_time_coil_current",
]

# The other two verbs of every plot (umbrella #434): ``dd_<stem>()`` lists the
# Data Dictionary paths, ``extract_<stem>(source, ...)`` returns the view
# model undrawn.  Generated from the registry the ``plot_*`` above are written
# against, so the three surfaces cover one set of plots.
from vaft.plot.backend.facade import install_facades as _install_facades  # noqa: E402

__all__ += list(_install_facades(
    globals(), normalize=normalize_entries, namespace="vaft.omas", subject="ods",
))
