"""Canonical ``<domain>_time_<quantity>`` renderers.

Every renderer here consumes a :class:`~vaft.plot.models.LineSeries` and draws it
with the same body.  What distinguishes one canonical name from another is
registry metadata -- labels, units, the IDS roots and paths an adapter must
supply -- not duplicated Matplotlib code.

Each name is a real module-level ``def`` so documentation tools and static
analysis can see it.
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import LineSeries
from ..registry import renderer
from ..style import apply_legend, axis_label, draw_series, finalize, resolve_axes, trace_labels

_DEFAULT_FIGSIZE = (6.0, 2.5)


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
    **style: Any,
) -> tuple[Figure, Axes]:
    """Draw a :class:`LineSeries` into one axes.

    This is the shared body behind every ``<domain>_time_<quantity>`` renderer and
    is also usable directly for ad-hoc traces that have no canonical name.
    """
    if not isinstance(model, LineSeries):
        raise TypeError(
            f"expected a vaft.plot.models.LineSeries; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    figure, axes = resolve_axes(ax, figsize=figsize or _DEFAULT_FIGSIZE)

    labels, legend_title = trace_labels(model.series, panel_title=model.title)
    for series, label in zip(model.series, labels):
        options = {**style, **series.style}
        if label:
            options.setdefault("label", label)
        draw_series(axes, series, uncertainty=uncertainty, validity=validity, **options)

    axes.set_xlabel(axis_label(model.x_label, model.x_unit))
    axes.set_ylabel(axis_label(model.y_label, model.y_unit))
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
    apply_legend(axes, legend=legend, title=legend_title)
    return finalize(figure, axes, show=show)



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
    """Measured plasma current history from the Rogowski coil."""
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
    """Measured diamagnetic flux history."""
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
    """Poloidal flux measured by each selected flux loop."""
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
    """Loop voltage measured by each selected flux loop."""
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
    """Poloidal field measured by each selected B-field probe."""
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
    """Raw or preprocessed Mirnov coil voltage traces."""
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
    """Poloidal beta history."""
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
    """Toroidal beta history."""
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
    """Normalized beta history."""
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
    """Safety factor at the 95% flux surface."""
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
    "equilibrium_time_qa",
    "equilibrium_time_w_mag",
    "equilibrium_time_w_mhd",
    "equilibrium_time_w_tot",
    "interferometer_time_n_e_line",
    "b_field_probe_time_field",
    "diamagnetic_flux_time",
    "flux_loop_time_flux",
    "flux_loop_time_voltage",
    "impa_time_field",
    "impa_time_voltage",
    "plasma_current_time",
    "mhd_linear_time_energy_perturbed",
    "mirnov_time_voltage",
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
