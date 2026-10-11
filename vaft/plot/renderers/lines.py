"""Canonical ``<domain>_time_<quantity>`` renderers.

Every renderer here consumes a :class:`~vaft.plot.models.LineSeries` and draws it
with the same body.  What distinguishes one canonical name from another is
registry metadata -- labels, units, the IDS roots and paths an adapter must
supply -- not duplicated Matplotlib code.

Each name is a real module-level ``def`` so documentation tools and static
analysis can see it.
"""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..models import LineSeries
from ..registry import renderer
from ..presentation import presented
from ..style import (
    apply_legend, axis_label, draw_series, finalize, resolve_axes, resolve_legend_placement, trace_labels,
)

_DEFAULT_FIGSIZE = (6.0, 2.5)


@presented(default_figsize=_DEFAULT_FIGSIZE)
def render_line_series(
    model: LineSeries,
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
    legend_placement: Mapping[str, Any] | None = None,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`LineSeries` into one axes.

    This is the shared body behind every ``<domain>_time_<quantity>`` renderer and
    is also usable directly for ad-hoc traces that have no canonical name.

    ``legend_placement`` sets where and how a drawn legend sits (see
    :func:`~vaft.plot.style.apply_legend`); its ``fontsize_scale`` key is a
    fraction of the format's base type size, never below
    :data:`vaft.plot.style.LEGEND_MIN_PT`, resolved here, inside the format, so it follows
    whatever size the figure is drawn at.
    """
    if not isinstance(model, LineSeries):
        raise TypeError(
            f"expected a vaft.plot.models.LineSeries; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    labels, legend_title = trace_labels(model.series, panel_title=model.title)
    secondary_axes = None
    if any(series.secondary for series in model.series):
        secondary_axes = axes.twinx()
        # The primary traces stay in front: a twin is drawn above its host
        # unless the host is raised, and its background would hide them.
        axes.set_zorder(secondary_axes.get_zorder() + 1)
        axes.patch.set_visible(False)
    for series, label in zip(model.series, labels):
        options = {**style, **series.style}
        if label:
            options.setdefault("label", label)
        target = secondary_axes if series.secondary else axes
        draw_series(target, series, uncertainty=uncertainty, validity=validity, **options)
        if series.secondary and label and target.lines:
            # One legend for the panel: a labelled secondary trace is keyed on
            # the primary axes by an empty proxy drawn like it -- opaque enough
            # to read, since a trace drawn translucent to show density would
            # all but vanish as a legend swatch.
            drawn = target.lines[-1]
            alpha = drawn.get_alpha()
            axes.add_line(Line2D(
                [], [], label=drawn.get_label(), color=drawn.get_color(),
                linestyle=drawn.get_linestyle(), linewidth=max(drawn.get_linewidth(), 1.0),
                marker=drawn.get_marker(), alpha=None if alpha is None else max(alpha, 0.6),
            ))

    axes.set_xlabel(axis_label(model.x_label, model.x_unit))
    axes.set_ylabel(axis_label(model.y_label, model.y_unit))
    if secondary_axes is not None:
        # Smaller than the primary label: the right axis is the subordinate
        # scale, and a label of full size crowds the panel beneath it.
        secondary_axes.set_ylabel(
            axis_label(model.secondary_y_label, model.secondary_y_unit), fontsize="small"
        )
    if model.title:
        axes.set_title(model.title)
    if model.display is not None and model.display.notation == "scientific":
        axes.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if model.log_y:
        axes.set_yscale("log")
    if model.x_limits is not None:
        axes.set_xlim(model.x_limits)
    if model.y_limits is not None:
        axes.set_ylim(model.y_limits)
    if grid:
        axes.grid(True, alpha=0.3)
    if secondary_axes is not None and not model.log_y:
        _align_zero(axes, secondary_axes)
    apply_legend(axes, legend=legend, title=legend_title, placement=resolve_legend_placement(legend_placement))
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def _align_zero(primary: Axes, secondary: Axes) -> None:
    """Put the secondary axis's zero at the height of the primary's.

    Two scales on one panel invite reading one trace against the other's
    ticks; with the zeros level, "above or below zero" at least reads the
    same on both.  The secondary range is widened, never cut, so every trace
    stays in view; nothing changes unless both ranges straddle zero.
    """
    low, high = primary.get_ylim()
    s_low, s_high = secondary.get_ylim()
    if not (low < 0.0 < high and s_low < 0.0 < s_high):
        return
    below = -low / (high - low)
    scale = max(-s_low / below, s_high / (1.0 - below))
    secondary.set_ylim(-below * scale, (1.0 - below) * scale)



@renderer(
    domain="magnetics",
    subject="plasma_current",
    view="time",
    quantity="",
    model=LineSeries,
    description="Measured plasma current history from the Rogowski coil.",
    ids=("magnetics",),
    required_paths=(
        "magnetics.ip.0.time",
        "magnetics.ip.0.data",
    ),
    optional_paths=(),
)
def plasma_current_time(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Measured plasma current history from the Rogowski coil.

    Interpretation
    --------------
    The toroidal plasma current against time.  It is the first trace read for
    any discharge: breakdown and current rise, flat-top, termination and
    current quench, and the time windows other analyses use.  It normalizes
    many derived quantities (normalized beta, the Greenwald density, q
    estimates) and compares discharges at a glance.

    Options
    -------
    ``synthetic=`` overlays, as markers at each equilibrium slice, the value
    the reconstruction predicts for this signal, so measurement and fit can be
    compared where the fit exists.  ``orientation=`` draws the current positive
    whatever the stored sign; ``x=`` replaces time by the sample index, for
    checking acquisition rather than physics.

    Convention
    ----------
    The stored sign follows the machine's current direction in the IMAS
    convention.  The default display draws the dominant polarity positive and
    says "sign flipped" in the title when it did; ``orientation="canonical"``
    keeps the stored sign, which is what tells the current direction.

    Limitations
    -----------
    A Rogowski coil measures all current threading it.  Depending on where it
    sits and on the processing that produced the stored waveform, currents
    induced in the vessel and passive structure may be included; they matter
    most at breakdown and during the current quench.

    See Also
    --------
    equilibrium_time_plasma_current : the current the reconstruction fitted.
    current_overview : plasma, coil and eddy currents together.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="diamagnetic_flux",
    view="time",
    quantity="",
    model=LineSeries,
    description="Measured diamagnetic flux history.",
    ids=("magnetics",),
    required_paths=(
        "magnetics.time",
        "magnetics.diamagnetic_flux.0.data",
    ),
    optional_paths=(),
)
def diamagnetic_flux_time(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Measured diamagnetic flux history.

    Interpretation
    --------------
    The change in toroidal flux through the plasma cross-section caused by the
    plasma.  Through radial pressure balance it measures the perpendicular
    pressure relative to the poloidal field: in the large-aspect-ratio limit a
    plasma with poloidal beta above 1 expels toroidal flux (diamagnetic), one
    below it draws flux in (paramagnetic).  It is the main magnetic measurement of stored energy and
    poloidal beta, and the one that separates beta_p from l_i in a
    reconstruction.

    Options
    -------
    ``synthetic=`` overlays, as markers at each equilibrium slice, the value
    the reconstruction predicts for this signal, so measurement and fit can be
    compared where the fit exists.  By default the dominant polarity is
    drawn positive, so a diamagnetic shot, whose stored flux is negative, is
    drawn positive too and the title says "sign flipped";
    ``orientation="canonical"`` keeps the stored sign and is needed to read
    para- or diamagnetism from the plot.

    Convention
    ----------
    In VEST products a positive stored flux is a paramagnetic plasma (#1196),
    a negative one a diamagnetic plasma.  The sign carries that meaning only
    under ``orientation="canonical"``.

    Limitations
    -----------
    The plasma's contribution is small beside the vacuum toroidal flux, so the
    trace depends on how the toroidal-field pickup and vessel currents were
    compensated, and integrated signals can drift.  Turning it into beta or
    stored energy needs the plasma shape, i.e. a reconstruction.

    See Also
    --------
    equilibrium_time_beta_p : the reconstruction's poloidal beta.
    equilibrium_time_diamagnetic_flux : measured against reconstructed constraint.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="flux_loop",
    view="time",
    quantity="flux",
    model=LineSeries,
    description="Poloidal flux measured by each selected flux loop.",
    ids=("magnetics",),
    required_paths=("magnetics.flux_loop.{i}.flux.data",),
    optional_paths=(
        "magnetics.flux_loop.time",
        "magnetics.time",
        "magnetics.flux_loop.{i}.position.0.r",
        "magnetics.flux_loop.{i}.position.0.z",
    ),
)
def flux_loop_time_flux(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Poloidal flux measured by each selected flux loop.

    Interpretation
    --------------
    The poloidal flux linked by each flux loop against time.  Flux loops sample
    psi at fixed points outside the plasma: their common rise follows the
    ohmic-coil and plasma flux, and the differences between loops carry the
    plasma's position and shape.  They are primary constraints of a magnetic
    equilibrium reconstruction, and the loop voltage is their time derivative.

    Options
    -------
    ``selection=`` / ``channels=`` choose the sensors and ``layout=`` whether
    they share one axes or are split into panels or groups.  ``validity=``
    decides what happens to channels the data flag as invalid: drawn but
    demoted (the default, so a reader sees them), removed, or treated as valid.
    ``synthetic=`` overlays, as markers at each equilibrium slice, the value
    the reconstruction predicts for this signal, so measurement and fit can be
    compared where the fit exists.

    Convention
    ----------
    Flux is in full weber, not weber per radian.

    Limitations
    -----------
    Each trace is the total flux at the loop -- coils, vessel currents and
    plasma together -- so the plasma's own contribution needs the vacuum
    response subtracted.  Integrated signals can drift, and a flagged channel
    is still drawn unless ``validity=`` says otherwise.

    See Also
    --------
    flux_loop_spatial_flux : the same loops against position at one time.
    magnetics_overview_plasma_residual : the signal left after the vacuum response.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="flux_loop",
    view="time",
    quantity="voltage",
    model=LineSeries,
    description="Loop voltage measured by each selected flux loop.",
    ids=("magnetics",),
    required_paths=("magnetics.flux_loop.{i}.voltage.data",),
    optional_paths=(
        "magnetics.flux_loop.time",
        "magnetics.time",
    ),
)
def flux_loop_time_voltage(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Loop voltage measured by each selected flux loop.

    Interpretation
    --------------
    The voltage induced in each flux loop, the rate of change of the poloidal
    flux it links.  It shows the inductive drive applied to the plasma -- the
    breakdown voltage, the ramp-up and flat-top loop voltage -- and its spatial
    variation between loops.

    Options
    -------
    ``selection=`` / ``channels=`` choose the sensors and ``layout=`` whether
    they share one axes or are split into panels or groups.  ``validity=``
    decides what happens to channels the data flag as invalid: drawn but
    demoted (the default, so a reader sees them), removed, or treated as valid.

    Limitations
    -----------
    The voltage at a loop is not the voltage at the plasma surface or on axis,
    and it contains the inductive change of the plasma's own flux; the
    resistive part needs the inductive correction.

    See Also
    --------
    flux_loop_time_flux : the integrated flux of the same loops.
    summary_time_voltage_consumption : resistive and inductive flux consumption.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="b_field_probe",
    view="time",
    quantity="field",
    model=LineSeries,
    description="Poloidal field measured by each selected B-field probe.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.field.data",),
    optional_paths=(
        "magnetics.b_field_pol_probe.time",
        "magnetics.time",
        "magnetics.b_field_pol_probe.{i}.position.r",
        "magnetics.b_field_pol_probe.{i}.position.z",
    ),
)
def b_field_probe_time_field(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Poloidal field measured by each selected B-field probe.

    Interpretation
    --------------
    The poloidal field each magnetic probe measures along its own orientation,
    against time.  Probes around the vessel sample the field at the boundary of
    the plasma region, which depends on the plasma current, its position and
    the current distribution; they are, with the flux loops, the main
    constraints of a magnetic reconstruction.

    Options
    -------
    ``selection=`` / ``channels=`` choose the sensors and ``layout=`` whether
    they share one axes or are split into panels or groups.  ``validity=``
    decides what happens to channels the data flag as invalid: drawn but
    demoted (the default, so a reader sees them), removed, or treated as valid.
    ``synthetic=`` overlays, as markers at each equilibrium slice, the value
    the reconstruction predicts for this signal, so measurement and fit can be
    compared where the fit exists.

    Limitations
    -----------
    Each probe measures one component, along its axis, of the total field --
    coils, vessel currents and plasma together.  Probe signals are integrated
    from induced voltages and can drift.

    See Also
    --------
    b_field_probe_spatial_field : the same probes against position at one time.
    mirnov_spectrogram : the fluctuating part of the probe signals.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="impa",
    view="time",
    quantity="field",
    model=LineSeries,
    description="Calibrated field from the IMPA Hall-probe array.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_tor_probe.{i}.field.data",),
    optional_paths=(
        "magnetics.b_field_tor_probe.{i}.identifier",
        "magnetics.b_field_tor_probe.{i}.position.r",
        "magnetics.b_field_pol_probe.{i}.field.data",
    ),
)
def impa_time_field(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Calibrated field from the IMPA Hall-probe array."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="impa",
    view="time",
    quantity="voltage",
    model=LineSeries,
    description="Raw IMPA Hall-probe voltages, one trace per channel.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_tor_probe.{i}.voltage.data",),
    optional_paths=(
        "magnetics.b_field_tor_probe.{i}.identifier",
        "magnetics.b_field_pol_probe.{i}.voltage.data",
    ),
)
def impa_time_voltage(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Raw IMPA Hall-probe voltages."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="mirnov",
    view="time",
    quantity="voltage",
    model=LineSeries,
    description="Raw or preprocessed Mirnov coil voltage traces.",
    ids=("magnetics",),
    required_paths=("magnetics.b_field_pol_probe.{i}.voltage.data",),
    optional_paths=(
        "magnetics.b_field_pol_probe.{i}.voltage.time",
        "magnetics.time",
    ),
)
def mirnov_time_voltage(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Raw or preprocessed Mirnov coil voltage traces.

    Interpretation
    --------------
    The voltage induced in each Mirnov coil, proportional to the rate of change
    of the field through it.  The traces show when MHD activity, sawtooth-like
    crashes or other fast magnetic events happen and how strongly each coil
    responds, before any spectral analysis.

    Options
    -------
    ``selection=`` / ``channels=`` choose the coils and ``layout=`` how they
    are arranged.

    Limitations
    -----------
    A coil voltage weights each frequency by that frequency, so a fast mode
    looks stronger than a slow one of equal field amplitude.  Slow equilibrium
    changes and pickup also appear, and comparing amplitudes between coils
    requires their effective areas.  The traces are as raw or as preprocessed
    as the stored signal.

    See Also
    --------
    mirnov_spectrogram : the frequency content of one coil against time.
    mirnov_spatial_phase : toroidal phase across the coils, for mode numbers.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="pf_passive",
    subject="passive_structure",
    view="time",
    quantity="current",
    model=LineSeries,
    description="Eddy current induced in the passive structure, summed over loops.",
    ids=("pf_passive",),
    required_paths=(
        "pf_passive.time",
        "pf_passive.loop.{i}.current",
    ),
    optional_paths=("pf_passive.loop.{i}.name",),
)
def passive_structure_time_current(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Eddy current in the passive structure."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="pf_active",
    subject="pf_coil",
    view="time",
    quantity="current",
    model=LineSeries,
    description="Per-coil PF current history.",
    ids=("pf_active",),
    required_paths=(
        "pf_active.time",
        "pf_active.coil.{i}.current.data",
    ),
    optional_paths=("pf_active.coil.{i}.name",),
)
def pf_coil_time_current(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Per-coil PF current history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="pf_active",
    subject="pf_coil",
    view="time",
    quantity="current_turns",
    model=LineSeries,
    description="Per-coil PF current multiplied by the signed turn count (ampere-turns).",
    ids=("pf_active",),
    required_paths=(
        "pf_active.time",
        "pf_active.coil.{i}.current.data",
        "pf_active.coil.{i}.element.{j}.turns_with_sign",
    ),
    optional_paths=("pf_active.coil.{i}.name",),
)
def pf_coil_time_current_turns(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Per-coil PF current multiplied by the signed turn count (ampere-turns)."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="plasma_current",
    model=LineSeries,
    description="Reconstructed plasma current history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.ip",
    ),
    optional_paths=(),
)
def equilibrium_time_plasma_current(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Reconstructed plasma current history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="li",
    model=LineSeries,
    description="Internal inductance li_3 history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.li_3",
    ),
    optional_paths=(),
)
def equilibrium_time_li(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Internal inductance li_3 history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="beta_p",
    model=LineSeries,
    description="Poloidal beta history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.beta_pol",
    ),
    optional_paths=(),
)
def equilibrium_time_beta_p(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Poloidal beta history.

    Interpretation
    --------------
    Poloidal beta at each reconstructed slice: the volume-averaged plasma
    pressure relative to the magnetic pressure of the poloidal field produced
    by the plasma current.  It measures how much pressure the plasma current
    confines, and enters the Shafranov shift and the vertical field needed for
    radial equilibrium.  beta_p near 1 separates a diamagnetic from a
    paramagnetic plasma.

    Limitations
    -----------
    Beta is a reconstruction output.  With magnetic constraints alone, poloidal
    beta and the internal inductance are not separately well determined --
    mainly their sum beta_p + l_i / 2 is -- particularly for nearly circular
    plasmas; a diamagnetic or kinetic pressure constraint improves it.
    Proximity to an empirical beta limit is context, not a stability verdict:
    the actual limit depends on the profiles, the shape and the wall.

    See Also
    --------
    diamagnetic_flux_time : the measurement most directly related to beta_p.
    equilibrium_time_li : the internal inductance it is entangled with.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="beta_t",
    model=LineSeries,
    description="Toroidal beta history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.beta_tor",
    ),
    optional_paths=(),
)
def equilibrium_time_beta_t(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Toroidal beta history.

    Interpretation
    --------------
    Toroidal beta at each reconstructed slice: the volume-averaged plasma
    pressure relative to the magnetic pressure of the vacuum toroidal field at
    the reference radius.  It measures how efficiently the toroidal field
    confines pressure; a reactor's fusion power density scales as beta_t^2 B^4
    at fixed temperature.

    Limitations
    -----------
    Beta is a reconstruction output.  With magnetic constraints alone, poloidal
    beta and the internal inductance are not separately well determined --
    mainly their sum beta_p + l_i / 2 is -- particularly for nearly circular
    plasmas; a diamagnetic or kinetic pressure constraint improves it.
    Proximity to an empirical beta limit is context, not a stability verdict:
    the actual limit depends on the profiles, the shape and the wall.

    See Also
    --------
    equilibrium_time_beta_n : toroidal beta on the Troyon scale.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="beta_n",
    model=LineSeries,
    description="Normalized beta history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.beta_normal",
    ),
    optional_paths=(),
)
def equilibrium_time_beta_n(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Normalized beta history.

    Interpretation
    --------------
    Normalized beta beta_N = beta_t[%] a B0 / I_p[MA] at each reconstructed
    slice: the toroidal beta scaled by the Troyon normalization, which removes
    most of the dependence on current and field and so compares discharges of
    different current and field on one scale.  It is the usual ordinate for
    judging how close a plasma comes to ideal pressure-driven limits.

    Limitations
    -----------
    Beta is a reconstruction output.  With magnetic constraints alone, poloidal
    beta and the internal inductance are not separately well determined --
    mainly their sum beta_p + l_i / 2 is -- particularly for nearly circular
    plasmas; a diamagnetic or kinetic pressure constraint improves it.
    Proximity to an empirical beta limit is context, not a stability verdict:
    the actual limit depends on the profiles, the shape and the wall.

    See Also
    --------
    equilibrium_time_beta : beta_p, beta_t and beta_N together.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="w_mhd",
    model=LineSeries,
    description="MHD stored energy history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.energy_mhd",
    ),
    optional_paths=(),
)
def equilibrium_time_w_mhd(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """MHD stored energy history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="w_mag",
    model=LineSeries,
    description="Magnetic stored energy history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.energy_mag",
    ),
    optional_paths=(),
)
def equilibrium_time_w_mag(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Magnetic stored energy history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="w_tot",
    model=LineSeries,
    description="Total stored energy history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.energy_total",
    ),
    optional_paths=(),
)
def equilibrium_time_w_tot(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Total stored energy history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="q0",
    model=LineSeries,
    description="Safety factor on axis.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.q_axis",
    ),
    optional_paths=(),
)
def equilibrium_time_q0(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Safety factor on axis."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="q95",
    model=LineSeries,
    description="Safety factor at the 95% flux surface.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.q_95",
    ),
    optional_paths=(),
)
def equilibrium_time_q95(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Safety factor at the 95% flux surface.

    Interpretation
    --------------
    q95, the safety factor on the flux surface enclosing 95 % of the poloidal
    flux, at each reconstructed slice.  It summarizes the edge field-line pitch
    -- the plasma current relative to the toroidal field and the shape -- in
    one number that stays finite for a diverted plasma, and is the usual
    coordinate for the low-q operating boundary: discharges approaching q95 of
    about 2 meet the external-kink limit and disrupt more often.

    Limitations
    -----------
    q95 exists only where a reconstruction exists; the line between slice
    markers joins them visually and is not a reconstruction.  Its accuracy is
    the equilibrium's: it depends on the boundary and the current-profile
    parametrisation.

    See Also
    --------
    summary_time_estimated_q95 : a q95 estimated from scalings without a reconstruction.
    equilibrium_profile_q : the whole q profile at one slice.
    """
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="qa",
    model=LineSeries,
    description="Safety factor at the plasma edge.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.global_quantities.qa",
    ),
    optional_paths=(),
)
def equilibrium_time_qa(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Safety factor at the plasma edge."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="minor_radius",
    model=LineSeries,
    description="Boundary minor radius history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.boundary.minor_radius",
    ),
    optional_paths=(),
)
def equilibrium_time_minor_radius(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Boundary minor radius history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="elongation",
    model=LineSeries,
    description="Boundary elongation history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.boundary.elongation",
    ),
    optional_paths=(),
)
def equilibrium_time_elongation(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Boundary elongation history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="triangularity",
    model=LineSeries,
    description="Boundary triangularity history, the mean of upper and lower.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.boundary.triangularity",
    ),
    optional_paths=(),
)
def equilibrium_time_triangularity(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Boundary triangularity history, the mean of upper and lower."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="triangularity_upper",
    model=LineSeries,
    description="Upper boundary triangularity history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.boundary.triangularity_upper",
    ),
    optional_paths=(),
)
def equilibrium_time_triangularity_upper(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Upper boundary triangularity history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="triangularity_lower",
    model=LineSeries,
    description="Lower boundary triangularity history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.boundary.triangularity_lower",
    ),
    optional_paths=(),
)
def equilibrium_time_triangularity_lower(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Lower boundary triangularity history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="major_radius",
    model=LineSeries,
    description="Geometric-axis major radius history.",
    ids=("equilibrium",),
    required_paths=(
        "equilibrium.time",
        "equilibrium.time_slice.{i}.boundary.geometric_axis.r",
    ),
    optional_paths=(),
)
def equilibrium_time_major_radius(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Geometric-axis major radius history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="time",
    quantity="diamagnetic_flux",
    model=LineSeries,
    description="Measured versus reconstructed diamagnetic-flux constraint.",
    ids=("equilibrium",),
    required_paths=("equilibrium.time",),
    optional_paths=(
        "equilibrium.time_slice.{i}.constraints.diamagnetic_flux.measured",
        "equilibrium.time_slice.{i}.constraints.diamagnetic_flux.reconstructed",
    ),
)
def equilibrium_time_diamagnetic_flux(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Measured versus reconstructed diamagnetic-flux constraint."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="tf",
    subject="tf_coil",
    view="time",
    quantity="b_t",
    model=LineSeries,
    description="Toroidal field history at the reference radius.",
    ids=("tf",),
    required_paths=(
        "tf.time",
        "tf.b_field_tor_vacuum_r.data",
    ),
    optional_paths=("tf.r0",),
)
def tf_coil_time_b_t(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Toroidal field history at the reference radius."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="tf",
    subject="tf_coil",
    view="time",
    quantity="b_t_vacuum_r",
    model=LineSeries,
    description="Vacuum toroidal field times major radius (B_t * R).",
    ids=("tf",),
    required_paths=(
        "tf.time",
        "tf.b_field_tor_vacuum_r.data",
    ),
    optional_paths=(),
)
def tf_coil_time_b_t_vacuum_r(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Vacuum toroidal field times major radius (B_t * R)."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="tf",
    subject="tf_coil",
    view="time",
    quantity="current",
    model=LineSeries,
    description="TF coil current history.",
    ids=("tf",),
    required_paths=(
        "tf.time",
        "tf.coil.{i}.current.data",
    ),
    optional_paths=("tf.coil.{i}.name",),
)
def tf_coil_time_current(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """TF coil current history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="spectrometer_uv",
    subject="spectrometer_uv",
    view="time",
    quantity="intensity",
    model=LineSeries,
    description="Processed spectral line intensity history.",
    ids=("spectrometer_uv",),
    required_paths=(
        "spectrometer_uv.time",
        "spectrometer_uv.channel.{i}.processed_line.{j}.intensity.data",
    ),
    optional_paths=("spectrometer_uv.channel.{i}.processed_line.{j}.label",),
)
def spectrometer_uv_time_intensity(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Processed spectral line intensity history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="ec_launchers",
    subject="ec_launchers",
    view="time",
    quantity="power",
    model=LineSeries,
    description="Net launched electron-cyclotron power (forward minus reflected) per beam; "
                "noisy, so smooth= is a rolling median in seconds.",
    ids=("ec_launchers",),
    required_paths=("ec_launchers.beam.{i}.power_launched.data",),
    optional_paths=(
        "ec_launchers.beam.{i}.power_launched.time",
        "ec_launchers.beam.{i}.name",
    ),
)
def ec_launchers_time_power(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Net launched electron-cyclotron power per beam."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="magnetics",
    subject="rogowski_coil",
    view="time",
    quantity="current",
    model=LineSeries,
    description="Current measured by each Rogowski coil.",
    ids=("magnetics",),
    required_paths=("magnetics.rogowski_coil.{i}.current.data",),
    optional_paths=(
        "magnetics.rogowski_coil.{i}.current.time",
        "magnetics.rogowski_coil.{i}.name",
    ),
)
def rogowski_coil_time_current(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Current measured by each Rogowski coil."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="pf_active",
    subject="vacuum",
    view="field",
    quantity="midplane",
    model=LineSeries,
    description="One vacuum startup quantity along the Z = 0 row of the vacuum map -- loop "
                "voltage, B_Z, the breakdown figure with its Ohmic and ECH-assisted thresholds, "
                "the Lloyd margin or the connection length -- with the ECR radius, chosen with field=.",
    ids=("pf_active", "pf_passive", "wall", "tf", "equilibrium", "magnetics",
         "spectrometer_uv", "barometry"),
    required_paths=("pf_active.time", "pf_active.coil.{i}.current.data"),
    optional_paths=(
        "tf.b_field_tor_vacuum_r.data",
        "wall.description_2d.{i}.limiter.unit.{j}.outline.r",
        "barometry.gauge.{i}.pressure.data",
    ),
)
def vacuum_field_midplane(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """One vacuum startup quantity along the midplane, at one instant."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="barometry",
    subject="barometry",
    view="time",
    quantity="pressure",
    model=LineSeries,
    description="Neutral pressure history from the barometry gauges.",
    ids=("barometry",),
    required_paths=(
        "barometry.gauge.{i}.pressure.time",
        "barometry.gauge.{i}.pressure.data",
    ),
    optional_paths=("barometry.gauge.{i}.name",),
)
def barometry_time_pressure(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Neutral pressure history from the barometry gauges."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="soft_x_rays",
    subject="soft_x_rays",
    view="time",
    quantity="power",
    model=LineSeries,
    description="Soft X-ray channel signal history.",
    ids=("soft_x_rays",),
    required_paths=("soft_x_rays.channel.{i}.brightness.data",),
    optional_paths=(
        "soft_x_rays.channel.{i}.brightness.time",
        "soft_x_rays.channel.{i}.power.data",
        "soft_x_rays.channel.{i}.power.time",
        "soft_x_rays.time",
        "soft_x_rays.channel.{i}.name",
    ),
)
def soft_x_rays_time_power(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Soft X-ray channel signal history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="interferometer",
    subject="interferometer",
    view="time",
    quantity="n_e_line",
    model=LineSeries,
    description="Interferometer line-integrated electron density history.",
    ids=("interferometer",),
    required_paths=("interferometer.channel.{i}.n_e_line.data",),
    optional_paths=(
        "interferometer.channel.{i}.n_e_line.time",
        "interferometer.time",
        "interferometer.channel.{i}.name",
    ),
)
def interferometer_time_n_e_line(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Interferometer line-integrated electron density history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="thomson_scattering",
    subject="thomson_scattering",
    view="time",
    quantity="electron_temperature",
    model=LineSeries,
    description="Per-channel Thomson electron temperature history.",
    ids=("thomson_scattering",),
    required_paths=(
        "thomson_scattering.time",
        "thomson_scattering.channel.{i}.t_e.data",
    ),
    optional_paths=("thomson_scattering.channel.{i}.name",),
)
def thomson_scattering_time_electron_temperature(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Per-channel Thomson electron temperature history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="thomson_scattering",
    subject="thomson_scattering",
    view="time",
    quantity="electron_density",
    model=LineSeries,
    description="Per-channel Thomson electron density history.",
    ids=("thomson_scattering",),
    required_paths=(
        "thomson_scattering.time",
        "thomson_scattering.channel.{i}.n_e.data",
    ),
    optional_paths=("thomson_scattering.channel.{i}.name",),
)
def thomson_scattering_time_electron_density(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Per-channel Thomson electron density history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="charge_exchange",
    subject="charge_exchange",
    view="time",
    quantity="ion_temperature",
    model=LineSeries,
    description="Per-channel ion temperature history from charge-exchange spectroscopy.",
    ids=("charge_exchange",),
    required_paths=("charge_exchange.channel.{i}.ion.{j}.t_i.data",),
    optional_paths=(
        "charge_exchange.channel.{i}.ion.{j}.t_i.time",
        "charge_exchange.time",
        "charge_exchange.channel.{i}.name",
    ),
)
def charge_exchange_time_ion_temperature(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Per-channel ion temperature history from charge-exchange spectroscopy."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="charge_exchange",
    subject="charge_exchange",
    view="time",
    quantity="velocity_tor",
    model=LineSeries,
    description="Per-channel toroidal rotation history from charge-exchange spectroscopy.",
    ids=("charge_exchange",),
    required_paths=("charge_exchange.channel.{i}.ion.{j}.velocity_tor.data",),
    optional_paths=(
        "charge_exchange.channel.{i}.ion.{j}.velocity_tor.time",
        "charge_exchange.time",
        "charge_exchange.channel.{i}.name",
    ),
)
def charge_exchange_time_velocity_tor(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Per-channel toroidal rotation history from charge-exchange spectroscopy."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="core_profiles",
    subject="electron_temperature",
    view="time",
    quantity="",
    model=LineSeries,
    description="Volume-averaged electron temperature history.",
    ids=("core_profiles",),
    required_paths=(
        "core_profiles.time",
        "core_profiles.profiles_1d.{i}.electrons.temperature",
    ),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.volume",),
)
def electron_temperature_time(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Volume-averaged electron temperature history."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="core_profiles",
    subject="electron_density",
    view="time",
    quantity="",
    model=LineSeries,
    description="Volume-averaged electron density history.",
    ids=("core_profiles",),
    required_paths=(
        "core_profiles.time",
        "core_profiles.profiles_1d.{i}.electrons.density",
    ),
    optional_paths=("core_profiles.profiles_1d.{i}.grid.volume",),
)
def electron_density_time(
    model: LineSeries,
    *,
    ax: Axes | None = None,
    show: bool = False,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Volume-averaged electron density history."""
    return render_line_series(model, ax=ax, show=show, **style)


__all__ = [
    "render_line_series",
    "barometry_time_pressure",
    "charge_exchange_time_ion_temperature",
    "charge_exchange_time_velocity_tor",
    "electron_density_time",
    "electron_temperature_time",
    "equilibrium_time_beta_n",
    "equilibrium_time_beta_p",
    "equilibrium_time_beta_t",
    "equilibrium_time_diamagnetic_flux",
    "equilibrium_time_li",
    "equilibrium_time_major_radius",
    "equilibrium_time_plasma_current",
    "equilibrium_time_q0",
    "equilibrium_time_q95",
    "equilibrium_time_minor_radius",
    "equilibrium_time_elongation",
    "equilibrium_time_triangularity",
    "equilibrium_time_triangularity_upper",
    "equilibrium_time_triangularity_lower",
    "equilibrium_time_qa",
    "equilibrium_time_w_mag",
    "equilibrium_time_w_mhd",
    "equilibrium_time_w_tot",
    "interferometer_time_n_e_line",
    "b_field_probe_time_field",
    "diamagnetic_flux_time",
    "ec_launchers_time_power",
    "flux_loop_time_flux",
    "flux_loop_time_voltage",
    "impa_time_field",
    "impa_time_voltage",
    "plasma_current_time",
    "rogowski_coil_time_current",
    "vacuum_field_midplane",
    "mhd_linear_time_energy_perturbed",
    "mirnov_time_voltage",
    "ntms_time_delta_prime",
    "pf_coil_time_current",
    "pf_coil_time_current_turns",
    "soft_x_rays_time_power",
    "spectrometer_uv_time_intensity",
    "tf_coil_time_b_t",
    "tf_coil_time_b_t_vacuum_r",
    "tf_coil_time_current",
    "thomson_scattering_time_electron_density",
    "thomson_scattering_time_electron_temperature",
]


@renderer(
    domain="mhd_linear",
    subject="ntms",
    view="time",
    quantity="delta_prime",
    model=LineSeries,
    description=(
        "Classical tearing index Delta-prime against time, one trace per "
        "rational surface; a positive value is a tearing-unstable surface. "
        "This is RDCON's and STRIDE's physical result, which has no slot under "
        "`toroidal_mode` and lives in `ntms`."
    ),
    ids=("ntms",),
    required_paths=(
        "ntms.time_slice.{i}.mode.{j}.n_tor",
        "ntms.time_slice.{i}.mode.{j}.m_pol",
        "ntms.time_slice.{i}.mode.{j}.deltaw.{k}.value",
    ),
)
def ntms_time_delta_prime(
    model: LineSeries, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Classical tearing index per rational surface, against time."""
    return render_line_series(model, ax=ax, show=show, **style)


@renderer(
    domain="mhd_linear",
    subject="mhd_linear",
    view="time",
    quantity="energy_perturbed",
    model=LineSeries,
    description=(
        "DCON perturbed potential energy against time, one trace per toroidal "
        "mode number; a negative value is an ideal-MHD unstable mode."
    ),
    ids=("mhd_linear",),
    required_paths=(
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.n_tor",
        "mhd_linear.time_slice.{i}.toroidal_mode.{j}.energy_perturbed",
    ),
)
def mhd_linear_time_energy_perturbed(
    model: LineSeries, *, ax: Any = None, show: bool = False, **style: Any
) -> tuple[Figure, Any]:
    """Perturbed potential energy history per toroidal mode."""
    return render_line_series(model, ax=ax, show=show, **style)
