"""Turn OMAS ``ODS``/``ODC`` content into ``vaft.plot`` view models.

This module is the only place in VAFT that knows both OMAS data paths and the
shape of the plotting view models.  Renderers stay data-object free (issue #62)
and the ``vaft.omas.plot_*`` adapters (issue #63) stay thin: they normalize their
input, call :func:`build_model` here, and hand the result to the registered
renderer.

A *recipe* declares how to read one canonical plot out of an ODS.  The recipes
mirror the ``required_paths`` that :mod:`vaft.plot.registry` publishes, so the
declared data requirements and the actual reads cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from vaft.plot.models import (
    Field2D,
    GeometryLayer,
    GeometryLayers,
    LineSeries,
    Panels,
    Profile1D,
    Series,
    Spectrogram,
)
from vaft.plot.registry import get_spec

__all__ = [
    "CallableRecipe",
    "FieldRecipe",
    "GeometryRecipe",
    "LineRecipe",
    "PanelRecipe",
    "ProfileRecipe",
    "RECIPES",
    "SpectrogramRecipe",
    "build_model",
    "entry_supports",
    "extract_labels_from_odc",
    "normalize_entries",
]

# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

_LABEL_OPTIONS = ("shot", "pulse", "run", "key")


def extract_labels_from_odc(odc: Any, opt: str = "shot") -> list[str]:
    """Return one label per ODC entry.

    ``opt`` selects ``shot``/``pulse`` (the data-entry pulse number), ``run``, or
    ``key`` (the ODC key).  Entries missing the requested metadata fall back to
    their key, so the returned order always matches ``odc.keys()``.
    """
    if opt not in _LABEL_OPTIONS:
        opt = "key"
    labels: list[str] = []
    for key in odc.keys():
        if opt == "key":
            labels.append(str(key))
            continue
        field_name = "run" if opt == "run" else "pulse"
        try:
            data_entry = odc[key].get("dataset_description.data_entry", {})
            value = data_entry.get(field_name)
        except Exception:
            value = None
        labels.append(str(key) if value is None else str(value))
    return labels


def normalize_entries(
    source: Any, *, label: str | Sequence[str] = "shot"
) -> tuple[tuple[str, Any], ...]:
    """Return deterministic ``(label, ods)`` pairs for any supported input.

    Accepts a single ``ODS``, an ``ODC``, or a list/tuple of either.  Ordering is
    the caller's ordering: ODC key order, or list order.  ``label`` may be one of
    ``shot``/``pulse``/``run``/``key`` or an explicit sequence of labels.
    """
    from omas import ODC, ODS

    # ODC subclasses ODS in OMAS, so the collection check must come first.
    if isinstance(source, ODC):
        entries = [(str(key), source[key]) for key in source.keys()]
    elif isinstance(source, ODS):
        entries = [("0", source)]
    elif isinstance(source, (list, tuple)):
        entries = []
        for position, item in enumerate(source):
            for key, ods in normalize_entries(item, label="key"):
                suffix = f"{position}" if key == "0" else f"{position}.{key}"
                entries.append((suffix, ods))
    else:
        raise TypeError(
            "expected an omas ODS, an ODC, or a list of them; got "
            f"{type(source).__name__}"
        )

    if isinstance(label, (list, tuple)):
        supplied = [str(item) for item in label]
        if len(supplied) != len(entries):
            raise ValueError(
                f"received {len(supplied)} labels for {len(entries)} entries"
            )
        return tuple(zip(supplied, (ods for _, ods in entries)))

    if label == "key" or len(entries) == 0:
        return tuple(entries)

    labels = []
    field_name = "run" if label == "run" else "pulse"
    for key, ods in entries:
        try:
            data_entry = ods.get("dataset_description.data_entry", {})
            value = data_entry.get(field_name)
        except Exception:
            value = None
        labels.append(key if value is None else str(value))
    return tuple(zip(labels, (ods for _, ods in entries)))


# ---------------------------------------------------------------------------
# ODS path helpers
# ---------------------------------------------------------------------------


def _get(ods: Any, path: str, default: Any = None) -> Any:
    """Read ``path`` from ``ods``, returning ``default`` when it is absent."""
    try:
        if path not in ods:
            return default
        return ods[path]
    except Exception:
        return default


def _array(ods: Any, path: str) -> np.ndarray | None:
    value = _get(ods, path)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _count(ods: Any, container_path: str) -> int:
    container = _get(ods, container_path)
    try:
        return len(container)
    except TypeError:
        return 0


def _container_of(template: str, marker: str = "{i}") -> str:
    """``"pf_active.coil.{i}.current.data"`` -> ``"pf_active.coil"``."""
    head, _, _ = template.partition("." + marker)
    return head


def _resolve_indices(
    ods: Any, template: str, requested: Iterable[int] | int | None, marker: str = "{i}"
) -> list[int]:
    if isinstance(requested, int):
        return [requested]
    if requested is not None:
        return [int(item) for item in requested]
    return list(range(_count(ods, _container_of(template, marker))))


def _first_time(ods: Any, candidates: Sequence[str], **substitutions: Any) -> np.ndarray | None:
    for candidate in candidates:
        array = _array(ods, candidate.format(**substitutions) if substitutions else candidate)
        if array is not None:
            return array
    return None


def _channel_label(ods: Any, template: str, index: int, fallback: str) -> str:
    if not template:
        return fallback
    name = _get(ods, template.format(i=index, j=0))
    return fallback if name in (None, "") else str(name)


def entry_supports(ods: Any, name: str) -> bool:
    """Whether ``ods`` holds the data the plot ``name`` needs."""
    spec = get_spec(name)
    recipe = RECIPES.get(name)
    if isinstance(recipe, PanelRecipe):
        # A composite is only available when at least one of its panels is.
        return any(entry_supports(ods, member) for member in recipe.members)
    if not spec.required_paths:
        return any(root in ods for root in spec.ids) if spec.ids else False
    for template in spec.required_paths:
        if "{" not in template:
            if _get(ods, template) is None:
                return False
            continue
        container = _container_of(template)
        total = _count(ods, container)
        if total == 0:
            return False
        # A present container is not enough: the leaf itself must exist for at
        # least one index, otherwise the adapter would build an empty model.
        if not any(
            _get(ods, template.format(i=index, j=0)) is not None
            for index in range(total)
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineRecipe:
    """How to read a ``LineSeries`` from an ODS.

    ``index`` selects what the ``{i}`` in ``y_path`` iterates:

    ``none``
        the path is complete and yields one trace;
    ``channel``
        each index is a separate diagnostic channel and becomes its own trace;
    ``time_slice``
        each index is a time slice holding one scalar, gathered into one trace.
    """

    y_path: str
    x_paths: tuple[str, ...] = ()
    index: str = "none"
    y_label: str = ""
    y_unit: str = ""
    x_label: str = "Time"
    x_unit: str = "s"
    scale: float = 1.0
    label_path: str = ""
    weight_path: str = ""
    title: str = ""


@dataclass(frozen=True)
class ProfileRecipe:
    """How to read a ``Profile1D`` from an ODS."""

    y_path: str
    coordinate_paths: dict[str, str] = field(default_factory=dict)
    default_coordinate: str = "rho_tor_norm"
    slice_container: str = ""
    index: str = "time_slice"
    y_label: str = ""
    y_unit: str = ""
    label_path: str = ""
    fallback_y_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeometryRecipe:
    """How to read a ``GeometryLayers`` stack from an ODS."""

    #: ``(kind, r_template, z_template, container, label_template, style)``
    layers: tuple[tuple[str, str, str, str, str, dict], ...]
    x_label: str = "R [m]"
    y_label: str = "Z [m]"
    title: str = ""


@dataclass(frozen=True)
class FieldRecipe:
    """How to read a ``Field2D`` from an ODS."""

    r_path: str
    z_path: str
    value_path: str
    value_label: str = ""
    boundary_paths: tuple[str, str] = ()
    title: str = ""


@dataclass(frozen=True)
class SpectrogramRecipe:
    """How to read a ``Spectrogram`` from an ODS."""

    signal_path: str
    time_paths: tuple[str, ...] = ()
    container: str = ""
    label_path: str = ""
    value_label: str = "Magnitude"


@dataclass(frozen=True)
class CallableRecipe:
    """A plot whose extraction needs real computation, not just path reads.

    ``builder(ods, **options)`` returns the view model.  Used for composed views
    and for models derived through :mod:`vaft.omas` processing helpers.
    """

    builder: Any
    description: str = ""


@dataclass(frozen=True)
class PanelRecipe:
    """A composite built from other canonical plots, one per panel."""

    members: tuple[str, ...]
    ncols: int = 1
    share_x: bool = True
    suptitle: str = ""


# ---------------------------------------------------------------------------
# The recipe table: one entry per canonical vaft.plot renderer
# ---------------------------------------------------------------------------

_MAGNETICS_TIME = ("magnetics.time",)
_EQ_TIME = ("equilibrium.time",)

RECIPES: dict[str, Any] = {
    # --- magnetics -----------------------------------------------------------
    "magnetics_time_ip": LineRecipe(
        y_path="magnetics.ip.0.data", x_paths=("magnetics.ip.0.time",) + _MAGNETICS_TIME,
        y_label="Plasma Current", y_unit="A", title="Plasma Current",
    ),
    "magnetics_time_diamagnetic_flux": LineRecipe(
        y_path="magnetics.diamagnetic_flux.0.data",
        x_paths=("magnetics.diamagnetic_flux.0.time",) + _MAGNETICS_TIME,
        y_label="Diamagnetic Flux", y_unit="Wb", title="Diamagnetic Flux",
    ),
    "magnetics_time_flux_loop_flux": LineRecipe(
        y_path="magnetics.flux_loop.{i}.flux.data", index="channel",
        x_paths=("magnetics.flux_loop.{i}.flux.time", "magnetics.flux_loop.time",
                 "magnetics.time"),
        y_label="Poloidal Flux", y_unit="Wb",
        label_path="magnetics.flux_loop.{i}.name", title="Flux Loop Flux",
    ),
    "magnetics_time_flux_loop_voltage": LineRecipe(
        y_path="magnetics.flux_loop.{i}.voltage.data", index="channel",
        x_paths=("magnetics.flux_loop.{i}.voltage.time", "magnetics.flux_loop.time",
                 "magnetics.time"),
        y_label="Loop Voltage", y_unit="V",
        label_path="magnetics.flux_loop.{i}.name", title="Flux Loop Voltage",
    ),
    "magnetics_time_b_field_pol_probe_field": LineRecipe(
        y_path="magnetics.b_field_pol_probe.{i}.field.data", index="channel",
        x_paths=("magnetics.b_field_pol_probe.{i}.field.time",
                 "magnetics.b_field_pol_probe.time", "magnetics.time"),
        y_label="Poloidal Field", y_unit="T",
        label_path="magnetics.b_field_pol_probe.{i}.name", title="B-field Probes",
    ),
    "magnetics_time_mirnov_voltage": LineRecipe(
        y_path="magnetics.b_field_pol_probe.{i}.voltage.data", index="channel",
        x_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
        y_label="Mirnov Signal", y_unit="V",
        label_path="magnetics.b_field_pol_probe.{i}.name", title="Mirnov Coils",
    ),
    # --- pf_active -----------------------------------------------------------
    "pf_active_time_current": LineRecipe(
        y_path="pf_active.coil.{i}.current.data", index="channel",
        x_paths=("pf_active.coil.{i}.current.time", "pf_active.time"),
        y_label="Coil Current", y_unit="A",
        label_path="pf_active.coil.{i}.name", title="PF Coil Currents",
    ),
    "pf_active_time_current_turns": LineRecipe(
        y_path="pf_active.coil.{i}.current.data", index="channel",
        x_paths=("pf_active.coil.{i}.current.time", "pf_active.time"),
        y_label="Coil Ampere-turns", y_unit="A-turns",
        label_path="pf_active.coil.{i}.name",
        weight_path="pf_active.coil.{i}.element.:.turns_with_sign",
        title="PF Coil Ampere-turns",
    ),
    # --- equilibrium global quantities ---------------------------------------
    "equilibrium_time_plasma_current": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.ip", index="time_slice",
        x_paths=_EQ_TIME, y_label="Plasma Current", y_unit="A",
        title="Equilibrium Plasma Current",
    ),
    "equilibrium_time_li": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.li_3", index="time_slice",
        x_paths=_EQ_TIME, y_label="Internal Inductance li_3",
        title="Internal Inductance",
    ),
    "equilibrium_time_beta_pol": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.beta_pol", index="time_slice",
        x_paths=_EQ_TIME, y_label="Poloidal Beta", title="Poloidal Beta",
    ),
    "equilibrium_time_beta_tor": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.beta_tor", index="time_slice",
        x_paths=_EQ_TIME, y_label="Toroidal Beta", title="Toroidal Beta",
    ),
    "equilibrium_time_beta_n": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.beta_normal",
        index="time_slice", x_paths=_EQ_TIME, y_label="Normalized Beta",
        title="Normalized Beta",
    ),
    "equilibrium_time_w_mhd": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.energy_mhd",
        index="time_slice", x_paths=_EQ_TIME, y_label="MHD Stored Energy", y_unit="J",
        title="MHD Stored Energy",
    ),
    "equilibrium_time_w_mag": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.energy_mag",
        index="time_slice", x_paths=_EQ_TIME, y_label="Magnetic Stored Energy",
        y_unit="J", title="Magnetic Stored Energy",
    ),
    "equilibrium_time_w_tot": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.energy_total",
        index="time_slice", x_paths=_EQ_TIME, y_label="Total Stored Energy",
        y_unit="J", title="Total Stored Energy",
    ),
    "equilibrium_time_q0": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.q_axis", index="time_slice",
        x_paths=_EQ_TIME, y_label="q on axis", title="q0",
    ),
    "equilibrium_time_q95": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.q_95", index="time_slice",
        x_paths=_EQ_TIME, y_label="q95", title="q95",
    ),
    "equilibrium_time_qa": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.qa", index="time_slice",
        x_paths=_EQ_TIME, y_label="qa", title="qa",
    ),
    "equilibrium_time_major_radius": LineRecipe(
        y_path="equilibrium.time_slice.{i}.boundary.geometric_axis.r",
        index="time_slice", x_paths=_EQ_TIME, y_label="Major Radius", y_unit="m",
        title="Equilibrium Major Radius",
    ),
    "equilibrium_time_diamagnetic_flux": LineRecipe(
        y_path="equilibrium.time_slice.{i}.constraints.diamagnetic_flux.measured",
        index="time_slice", x_paths=_EQ_TIME, y_label="Diamagnetic Flux", y_unit="Wb",
        title="Diamagnetic Flux Constraint",
    ),
    # --- tf ------------------------------------------------------------------
    "tf_time_b_field_tor": LineRecipe(
        y_path="tf.b_field_tor_vacuum_r.data", x_paths=("tf.b_field_tor_vacuum_r.time",
                                                        "tf.time"),
        y_label="Toroidal Field", y_unit="T", title="Toroidal Field",
    ),
    "tf_time_b_field_tor_vacuum_r": LineRecipe(
        y_path="tf.b_field_tor_vacuum_r.data",
        x_paths=("tf.b_field_tor_vacuum_r.time", "tf.time"),
        y_label="B_t * R", y_unit="T m", title="Vacuum B_t * R",
    ),
    "tf_time_coil_current": LineRecipe(
        y_path="tf.coil.{i}.current.data", index="channel",
        x_paths=("tf.coil.{i}.current.time", "tf.time"),
        y_label="TF Coil Current", y_unit="A", label_path="tf.coil.{i}.name",
        title="TF Coil Current",
    ),
    # --- other diagnostics ---------------------------------------------------
    "spectrometer_uv_time_intensity": LineRecipe(
        y_path="spectrometer_uv.channel.{i}.processed_line.0.intensity.data",
        index="channel", x_paths=("spectrometer_uv.time",),
        y_label="Line Intensity", y_unit="a.u.",
        label_path="spectrometer_uv.channel.{i}.name", title="UV Line Intensity",
    ),
    "barometry_time_pressure": LineRecipe(
        y_path="barometry.gauge.{i}.pressure.data", index="channel",
        x_paths=("barometry.gauge.{i}.pressure.time",),
        y_label="Neutral Pressure", y_unit="Pa",
        label_path="barometry.gauge.{i}.name", title="Neutral Pressure",
    ),
    "soft_x_rays_time_power": LineRecipe(
        y_path="soft_x_rays.channel.{i}.power.data", index="channel",
        x_paths=("soft_x_rays.channel.{i}.power.time", "soft_x_rays.time"),
        y_label="Soft X-ray Signal", y_unit="a.u.",
        label_path="soft_x_rays.channel.{i}.name", title="Soft X-ray Signals",
    ),
    "thomson_scattering_time_electron_temperature": LineRecipe(
        y_path="thomson_scattering.channel.{i}.t_e.data", index="channel",
        x_paths=("thomson_scattering.channel.{i}.t_e.time", "thomson_scattering.time"),
        y_label="Electron Temperature", y_unit="eV",
        label_path="thomson_scattering.channel.{i}.name", title="Thomson T_e",
    ),
    "thomson_scattering_time_electron_density": LineRecipe(
        y_path="thomson_scattering.channel.{i}.n_e.data", index="channel",
        x_paths=("thomson_scattering.channel.{i}.n_e.time", "thomson_scattering.time"),
        y_label="Electron Density", y_unit="m^-3",
        label_path="thomson_scattering.channel.{i}.name", title="Thomson n_e",
    ),
    "charge_exchange_time_ion_temperature": LineRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.t_i.data", index="channel",
        x_paths=("charge_exchange.channel.{i}.ion.0.t_i.time", "charge_exchange.time"),
        y_label="Ion Temperature", y_unit="eV",
        label_path="charge_exchange.channel.{i}.name", title="CES T_i",
    ),
    "charge_exchange_time_velocity_tor": LineRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.velocity_tor.data", index="channel",
        x_paths=("charge_exchange.channel.{i}.ion.0.velocity_tor.time",
                 "charge_exchange.time"),
        y_label="Toroidal Rotation", y_unit="m/s",
        label_path="charge_exchange.channel.{i}.name", title="CES v_tor",
    ),
    "core_profiles_time_electron_temperature": LineRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.temperature",
        index="time_slice_mean", x_paths=("core_profiles.time",),
        y_label="<T_e>", y_unit="eV", title="Volume-averaged T_e",
    ),
    "core_profiles_time_electron_density": LineRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.density",
        index="time_slice_mean", x_paths=("core_profiles.time",),
        y_label="<n_e>", y_unit="m^-3", title="Volume-averaged n_e",
    ),
    # --- 1D profiles ---------------------------------------------------------
    "equilibrium_profile_pressure": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.pressure",
        y_label="Pressure", y_unit="Pa",
    ),
    "equilibrium_profile_q": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.q", y_label="Safety Factor q",
    ),
    "equilibrium_profile_j_tor": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.j_tor",
        y_label="Toroidal Current Density", y_unit="A/m^2",
    ),
    "equilibrium_profile_pprime": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.dpressure_dpsi",
        fallback_y_paths=("equilibrium.time_slice.{i}.profiles_1d.pprime",),
        y_label="dp/dpsi", y_unit="Pa/Wb",
    ),
    "equilibrium_profile_f": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.f", y_label="F = R B_t",
        y_unit="T m",
    ),
    "equilibrium_profile_ffprime": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.f_df_dpsi",
        fallback_y_paths=("equilibrium.time_slice.{i}.profiles_1d.ffprime",),
        y_label="F dF/dpsi", y_unit="T^2 m^2/Wb",
    ),
    "core_profiles_profile_electron_temperature": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.temperature",
        coordinate_paths={"rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm",
                          "psi_norm": "core_profiles.profiles_1d.{i}.grid.rho_pol_norm"},
        slice_container="core_profiles.profiles_1d",
        y_label="Electron Temperature", y_unit="eV",
    ),
    "core_profiles_profile_electron_density": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.density",
        coordinate_paths={"rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm",
                          "psi_norm": "core_profiles.profiles_1d.{i}.grid.rho_pol_norm"},
        slice_container="core_profiles.profiles_1d",
        y_label="Electron Density", y_unit="m^-3",
    ),
    "core_profiles_profile_ion_temperature": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.ion.0.temperature",
        coordinate_paths={"rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm"},
        slice_container="core_profiles.profiles_1d",
        y_label="Ion Temperature", y_unit="eV",
    ),
    "core_profiles_profile_pressure": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.pressure_thermal",
        coordinate_paths={"rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm"},
        slice_container="core_profiles.profiles_1d",
        y_label="Thermal Pressure", y_unit="Pa",
    ),
    "thomson_scattering_profile_electron_temperature": ProfileRecipe(
        y_path="thomson_scattering.channel.{i}.t_e.data", index="channel",
        coordinate_paths={"r_major": "thomson_scattering.channel.{i}.position.r"},
        default_coordinate="r_major", slice_container="thomson_scattering.channel",
        y_label="Electron Temperature", y_unit="eV",
    ),
    "thomson_scattering_profile_electron_density": ProfileRecipe(
        y_path="thomson_scattering.channel.{i}.n_e.data", index="channel",
        coordinate_paths={"r_major": "thomson_scattering.channel.{i}.position.r"},
        default_coordinate="r_major", slice_container="thomson_scattering.channel",
        y_label="Electron Density", y_unit="m^-3",
    ),
    "charge_exchange_profile_ion_temperature": ProfileRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.t_i.data", index="channel",
        coordinate_paths={"r_major": "charge_exchange.channel.{i}.position.r.data"},
        default_coordinate="r_major", slice_container="charge_exchange.channel",
        y_label="Ion Temperature", y_unit="eV",
    ),
    "charge_exchange_profile_velocity_tor": ProfileRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.velocity_tor.data", index="channel",
        coordinate_paths={"r_major": "charge_exchange.channel.{i}.position.r.data"},
        default_coordinate="r_major", slice_container="charge_exchange.channel",
        y_label="Toroidal Rotation", y_unit="m/s",
    ),
    # --- 2D fields -----------------------------------------------------------
    "equilibrium_field_psi": FieldRecipe(
        r_path="equilibrium.time_slice.{i}.profiles_2d.0.grid.dim1",
        z_path="equilibrium.time_slice.{i}.profiles_2d.0.grid.dim2",
        value_path="equilibrium.time_slice.{i}.profiles_2d.0.psi",
        value_label="Poloidal Flux [Wb]",
        boundary_paths=("equilibrium.time_slice.{i}.boundary.outline.r",
                        "equilibrium.time_slice.{i}.boundary.outline.z"),
        title="Poloidal Flux",
    ),
    # --- geometry ------------------------------------------------------------
    "pf_active_geometry_poloidal": GeometryRecipe(
        layers=(("polygon", "pf_active.coil.{i}.element.0.geometry.outline.r",
                 "pf_active.coil.{i}.element.0.geometry.outline.z",
                 "pf_active.coil", "pf_active.coil.{i}.name", {}),),
        title="PF Coils",
    ),
    "pf_passive_geometry_poloidal": GeometryRecipe(
        layers=(("polygon", "pf_passive.loop.{i}.element.0.geometry.outline.r",
                 "pf_passive.loop.{i}.element.0.geometry.outline.z",
                 "pf_passive.loop", "pf_passive.loop.{i}.name", {}),),
        title="Passive Structure",
    ),
    "wall_geometry_poloidal": GeometryRecipe(
        layers=(("polygon", "wall.description_2d.0.limiter.unit.{i}.outline.r",
                 "wall.description_2d.0.limiter.unit.{i}.outline.z",
                 "wall.description_2d.0.limiter.unit", "", {"color": "0.4"}),),
        title="First Wall",
    ),
    "magnetics_geometry_poloidal": GeometryRecipe(
        layers=(
            ("points", "magnetics.flux_loop.{i}.position.0.r",
             "magnetics.flux_loop.{i}.position.0.z", "magnetics.flux_loop", "",
             {"marker": "s", "markersize": 3, "color": "#377eb8"}),
            ("points", "magnetics.b_field_pol_probe.{i}.position.r",
             "magnetics.b_field_pol_probe.{i}.position.z",
             "magnetics.b_field_pol_probe", "",
             {"marker": "x", "markersize": 4, "color": "#ff7f00"}),
        ),
        title="Magnetic Diagnostics",
    ),
    "equilibrium_geometry_boundary": GeometryRecipe(
        layers=(("polygon", "equilibrium.time_slice.{i}.boundary.outline.r",
                 "equilibrium.time_slice.{i}.boundary.outline.z",
                 "equilibrium.time_slice", "", {"color": "#e41a1c"}),),
        title="Plasma Boundary",
    ),
    "thomson_scattering_geometry_poloidal": GeometryRecipe(
        layers=(("points", "thomson_scattering.channel.{i}.position.r",
                 "thomson_scattering.channel.{i}.position.z",
                 "thomson_scattering.channel", "", {"marker": "o", "markersize": 3}),),
        title="Thomson Scattering Positions",
    ),
    "charge_exchange_geometry_poloidal": GeometryRecipe(
        layers=(("points", "charge_exchange.channel.{i}.position.r.data",
                 "charge_exchange.channel.{i}.position.z.data",
                 "charge_exchange.channel", "", {"marker": "d", "markersize": 3}),),
        title="Charge-exchange Positions",
    ),
    # --- spectrograms --------------------------------------------------------
    "magnetics_spectrogram_mirnov": SpectrogramRecipe(
        signal_path="magnetics.b_field_pol_probe.{i}.voltage.data",
        time_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
        container="magnetics.b_field_pol_probe",
        label_path="magnetics.b_field_pol_probe.{i}.name",
    ),
    "soft_x_rays_spectrogram": SpectrogramRecipe(
        signal_path="soft_x_rays.channel.{i}.power.data",
        time_paths=("soft_x_rays.channel.{i}.power.time", "soft_x_rays.time"),
        container="soft_x_rays.channel",
        label_path="soft_x_rays.channel.{i}.name",
    ),
    # --- composites ----------------------------------------------------------
    "summary_time_energy": PanelRecipe(
        members=("equilibrium_time_w_mhd", "equilibrium_time_w_mag",
                 "equilibrium_time_w_tot"),
        suptitle="Stored Energy",
    ),
    "summary_time_beta": PanelRecipe(
        members=("equilibrium_time_beta_pol", "equilibrium_time_beta_tor",
                 "equilibrium_time_beta_n"),
        suptitle="Beta",
    ),
    "summary_time_power_balance": PanelRecipe(
        members=("equilibrium_time_plasma_current", "equilibrium_time_w_mhd",
                 "core_profiles_time_electron_temperature"),
        suptitle="Power Balance Inputs",
    ),
    "summary_time_voltage_consumption": PanelRecipe(
        members=("magnetics_time_ip", "magnetics_time_flux_loop_voltage"),
        suptitle="Voltage Consumption",
    ),
    "equilibrium_time_virial": PanelRecipe(
        members=("equilibrium_time_beta_pol", "equilibrium_time_li",
                 "equilibrium_time_w_mhd"),
        suptitle="Virial Equilibrium Quantities",
    ),
    "electromagnetics_time_current": PanelRecipe(
        members=("magnetics_time_ip", "pf_active_time_current"),
        suptitle="Electromagnetic Currents",
    ),
    "core_profiles_time_volume_averaged": PanelRecipe(
        members=("core_profiles_time_electron_temperature",
                 "core_profiles_time_electron_density"),
        suptitle="Volume-averaged Core Profiles",
    ),
    "spectrometer_uv_time_impurity": PanelRecipe(
        members=("magnetics_time_ip", "spectrometer_uv_time_intensity"),
        suptitle="Impurity Line Intensity",
    ),
    "magnetics_overview": PanelRecipe(
        members=("magnetics_time_ip", "pf_active_time_current",
                 "magnetics_time_flux_loop_flux",
                 "magnetics_time_b_field_pol_probe_field"),
        ncols=2, share_x=False, suptitle="Shot Diagnostics Overview",
    ),
    "equilibrium_overview": PanelRecipe(
        members=("equilibrium_time_plasma_current", "equilibrium_time_beta_pol",
                 "equilibrium_time_li", "equilibrium_time_q95"),
        ncols=2, share_x=False, suptitle="Equilibrium Analysis Overview",
    ),
    "soft_x_rays_overview": PanelRecipe(
        members=("soft_x_rays_time_power", "soft_x_rays_geometry_lines_of_sight"),
        share_x=False, suptitle="Soft X-ray Overview",
    ),
}


# ---------------------------------------------------------------------------
# Builders for the plots that need computation rather than a plain path read
# ---------------------------------------------------------------------------


def _build_lines_of_sight(ods: Any, *, channels: Any = None, **_: Any) -> GeometryLayers:
    """Soft X-ray lines of sight, drawn as one segment per channel."""
    template = "soft_x_rays.channel.{i}.line_of_sight.first_point.r"
    indices = _resolve_indices(ods, template, channels)
    layers: list[GeometryLayer] = []
    for index in indices:
        base = f"soft_x_rays.channel.{index}.line_of_sight"
        first_r = _get(ods, f"{base}.first_point.r")
        first_z = _get(ods, f"{base}.first_point.z")
        second_r = _get(ods, f"{base}.second_point.r")
        second_z = _get(ods, f"{base}.second_point.z")
        if None in (first_r, first_z, second_r, second_z):
            continue
        layers.append(
            GeometryLayer(
                r=[float(first_r), float(second_r)],
                z=[float(first_z), float(second_z)],
                kind="polyline",
                label=_channel_label(ods, "soft_x_rays.channel.{i}.name", index,
                                     f"ch{index}"),
                style={"lw": 0.8},
            )
        )
    layers.extend(_wall_layers(ods))
    return GeometryLayers(layers=tuple(layers), title="Soft X-ray Lines of Sight")


def _wall_layers(ods: Any) -> list[GeometryLayer]:
    template = "wall.description_2d.0.limiter.unit.{i}.outline.r"
    layers = []
    for index in _resolve_indices(ods, template, None):
        r = _array(ods, template.format(i=index))
        z = _array(ods, f"wall.description_2d.0.limiter.unit.{index}.outline.z")
        if r is None or z is None or r.size != z.size:
            continue
        layers.append(
            GeometryLayer(r=r, z=z, kind="polygon", style={"color": "0.4", "lw": 1.0})
        )
    return layers


def _build_machine_poloidal(ods: Any, **options: Any) -> GeometryLayers:
    """Compose wall, coils, passive structure and diagnostics into one view."""
    layers: list[GeometryLayer] = list(_wall_layers(ods))
    for member in (
        "pf_active_geometry_poloidal",
        "pf_passive_geometry_poloidal",
        "magnetics_geometry_poloidal",
        "thomson_scattering_geometry_poloidal",
        "charge_exchange_geometry_poloidal",
    ):
        if not entry_supports(ods, member):
            continue
        layers.extend(_build_geometry(ods, RECIPES[member], **options).layers)
    if not layers:
        raise ValueError(
            "none of the poloidal machine geometry IDS (wall, pf_active, "
            "pf_passive, magnetics, thomson_scattering, charge_exchange) are present"
        )
    return GeometryLayers(layers=tuple(layers), title="Machine Cross-section")


def _boundary_extent(ods: Any, time_slice: int) -> tuple[float, float] | None:
    r = _array(ods, f"equilibrium.time_slice.{time_slice}.boundary.outline.r")
    if r is None:
        return None
    return float(np.nanmin(r)), float(np.nanmax(r))


def _ring(radius: float, points: int = 181) -> tuple[np.ndarray, np.ndarray]:
    angle = np.linspace(0.0, 2.0 * np.pi, points)
    return radius * np.cos(angle), radius * np.sin(angle)


def _build_equilibrium_topview(
    ods: Any, *, time_slice: int = 0, **_: Any
) -> GeometryLayers:
    """Project the plasma boundary onto the machine top view as two rings."""
    extent = _boundary_extent(ods, time_slice)
    if extent is None:
        raise ValueError(
            f"equilibrium.time_slice.{time_slice}.boundary.outline.r is not available"
        )
    inner, outer = extent
    layers = []
    for radius, label in ((outer, "Plasma outboard"), (inner, "Plasma inboard")):
        x, y = _ring(radius)
        layers.append(
            GeometryLayer(r=x, z=y, kind="polyline", label=label,
                          style={"color": "#e41a1c"})
        )
    return GeometryLayers(
        layers=tuple(layers), x_label="x [m]", y_label="y [m]",
        title="Plasma Top View",
    )


def _build_machine_topview(ods: Any, *, time_slice: int = 0, **_: Any) -> GeometryLayers:
    """Compose the plasma extent with launcher, antenna and pellet geometry."""
    layers: list[GeometryLayer] = []
    if "equilibrium" in ods:
        try:
            layers.extend(_build_equilibrium_topview(ods, time_slice=time_slice).layers)
        except ValueError:
            pass
    for container, r_path, label, style in (
        ("lh_antennas.antenna", "lh_antennas.antenna.{i}.position.r",
         "LH antenna", {"marker": "s"}),
        ("ec_launchers.beam", "ec_launchers.beam.{i}.launching_position.r",
         "EC launcher", {"marker": "^"}),
    ):
        for index in range(_count(ods, container)):
            radius = _get(ods, r_path.format(i=index))
            phi = _get(ods, r_path.format(i=index).replace(".r", ".phi"), 0.0)
            if radius is None:
                continue
            radius, phi = float(radius), float(phi or 0.0)
            layers.append(
                GeometryLayer(
                    r=[radius * np.cos(phi)], z=[radius * np.sin(phi)], kind="points",
                    label=f"{label} {index}", style=style,
                )
            )
    if not layers:
        raise ValueError(
            "none of the top-view IDS (equilibrium, lh_antennas, ec_launchers, "
            "pellets) are present"
        )
    return GeometryLayers(
        layers=tuple(layers), x_label="x [m]", y_label="y [m]",
        title="Machine Top View",
    )


def _build_vacuum_psi(ods: Any, *, time: float | None = None, **_: Any) -> Field2D:
    """Vacuum poloidal flux from the PF coils, via the OMAS null-field helper."""
    from vaft.omas import compute_null_ods, find_breakdown_onset

    if time is None:
        time = find_breakdown_onset(ods)
    psi, r_grid, z_grid = compute_null_ods(ods, time)
    r_axis = np.asarray(r_grid)[0, :] if np.ndim(r_grid) == 2 else np.asarray(r_grid)
    z_axis = np.asarray(z_grid)[:, 0] if np.ndim(z_grid) == 2 else np.asarray(z_grid)
    values = np.asarray(psi, dtype=float)
    if values.shape != (z_axis.size, r_axis.size):
        values = values.T
    return Field2D(
        r=r_axis, z=z_axis, values=values,
        value_label="Vacuum Poloidal Flux [Wb]", filled=False, contour_levels=50,
        overlays=tuple(_wall_layers(ods)),
        title=f"Vacuum psi at t = {float(time) * 1e3:.1f} ms",
    )


def _build_core_profile_field(
    ods: Any, *, quantity: str, time_slice: int = 0, **_: Any
) -> Field2D:
    """Map a 1D core profile onto the poloidal plane using the equilibrium psi."""
    grid_r = _array(ods, f"equilibrium.time_slice.{time_slice}.profiles_2d.0.grid.dim1")
    grid_z = _array(ods, f"equilibrium.time_slice.{time_slice}.profiles_2d.0.grid.dim2")
    psi_2d = _array(ods, f"equilibrium.time_slice.{time_slice}.profiles_2d.0.psi")
    if grid_r is None or grid_z is None or psi_2d is None:
        raise ValueError(
            f"equilibrium.time_slice.{time_slice}.profiles_2d.0 is required to map a "
            "core profile onto the poloidal plane"
        )
    psi_axis = _get(ods, f"equilibrium.time_slice.{time_slice}.global_quantities.psi_axis")
    psi_boundary = _get(
        ods, f"equilibrium.time_slice.{time_slice}.global_quantities.psi_boundary"
    )
    profile = _array(ods, f"core_profiles.profiles_1d.{time_slice}.electrons.{quantity}")
    rho = _array(ods, f"core_profiles.profiles_1d.{time_slice}.grid.rho_tor_norm")
    if profile is None or rho is None:
        raise ValueError(
            f"core_profiles.profiles_1d.{time_slice}.electrons.{quantity} and its "
            "rho_tor_norm grid are required"
        )
    if psi_axis is None or psi_boundary is None:
        psi_axis, psi_boundary = float(np.nanmin(psi_2d)), float(np.nanmax(psi_2d))
    span = float(psi_boundary) - float(psi_axis)
    if span == 0.0:
        raise ValueError("equilibrium psi_axis and psi_boundary are equal")
    psi_norm = (psi_2d - float(psi_axis)) / span
    if psi_norm.shape != (grid_z.size, grid_r.size):
        psi_norm = psi_norm.T
    rho_2d = np.sqrt(np.clip(psi_norm, 0.0, 1.0))
    values = np.interp(rho_2d.ravel(), rho, profile).reshape(rho_2d.shape)
    values = np.where(psi_norm <= 1.0, values, np.nan)
    labels = {"temperature": "Electron Temperature [eV]",
              "density": "Electron Density [m^-3]"}
    return Field2D(
        r=grid_r, z=grid_z, values=values, value_label=labels[quantity],
        overlays=tuple(_wall_layers(ods)), title=labels[quantity],
    )


RECIPES["soft_x_rays_geometry_lines_of_sight"] = CallableRecipe(
    builder=_build_lines_of_sight,
    description="One polyline per detector line of sight, over the wall outline.",
)
RECIPES["machine_geometry_poloidal"] = CallableRecipe(
    builder=_build_machine_poloidal,
    description="Wall, coils, passive structure and diagnostic positions composed.",
)
RECIPES["equilibrium_geometry_topview"] = CallableRecipe(
    builder=_build_equilibrium_topview,
    description="Plasma inboard/outboard extent projected into the top view.",
)
RECIPES["machine_geometry_topview"] = CallableRecipe(
    builder=_build_machine_topview,
    description="Plasma extent plus launcher and antenna positions in the top view.",
)
RECIPES["equilibrium_field_psi_vacuum"] = CallableRecipe(
    builder=_build_vacuum_psi,
    description="Vacuum flux map from the PF currents via vaft.omas.compute_null_ods.",
)
RECIPES["core_profiles_field_electron_temperature"] = CallableRecipe(
    builder=lambda ods, **options: _build_core_profile_field(
        ods, **{**options, "quantity": "temperature"}
    ),
    description="Electron temperature mapped onto the poloidal plane.",
)
RECIPES["core_profiles_field_electron_density"] = CallableRecipe(
    builder=lambda ods, **options: _build_core_profile_field(
        ods, **{**options, "quantity": "density"}
    ),
    description="Electron density mapped onto the poloidal plane.",
)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

_TIME_SCALES = {"s": 1.0, "ms": 1e3, "us": 1e6}
_VALUE_SCALES = {"A": 1.0, "kA": 1e-3, "MA": 1e-6, "Wb": 1.0, "mWb": 1e3}


def _scaled(array: np.ndarray, unit: str, table: dict[str, float]) -> np.ndarray:
    return array * table.get(unit, 1.0)


def _entry_prefix(label: str, extra: str) -> str:
    if label and extra:
        return f"{label} {extra}"
    return label or extra


def _weight(ods: Any, template: str, index: int) -> float:
    if not template:
        return 1.0
    values = _get(ods, template.format(i=index))
    if values is None:
        return 1.0
    return float(np.sum(np.abs(np.asarray(values, dtype=float))))


def _build_line_traces(
    ods: Any,
    recipe: LineRecipe,
    *,
    entry_label: str,
    channels: Any = None,
    xunit: str = "s",
    yunit: str | None = None,
) -> list[Series]:
    time_scale = _TIME_SCALES.get(xunit, 1.0)
    value_scale = _VALUE_SCALES.get(yunit or recipe.y_unit, 1.0) * recipe.scale

    if recipe.index in ("time_slice", "time_slice_mean"):
        time = _first_time(ods, recipe.x_paths)
        indices = _resolve_indices(ods, recipe.y_path, None)
        values = []
        for index in indices:
            raw = _get(ods, recipe.y_path.format(i=index))
            if raw is None:
                values.append(np.nan)
            elif recipe.index == "time_slice_mean":
                values.append(float(np.nanmean(np.asarray(raw, dtype=float))))
            else:
                values.append(float(np.asarray(raw, dtype=float).ravel()[0]))
        if not values:
            return []
        y = np.asarray(values, dtype=float)
        if time is None or time.size != y.size:
            time = np.arange(y.size, dtype=float)
        return [Series(x=_scaled(time, xunit, _TIME_SCALES), y=y * value_scale,
                       label=entry_label)]

    if recipe.index == "channel":
        indices = _resolve_indices(ods, recipe.y_path, channels)
        traces = []
        for index in indices:
            y = _array(ods, recipe.y_path.format(i=index))
            if y is None:
                continue
            time = _first_time(ods, recipe.x_paths, i=index)
            if time is None or time.size != y.size:
                time = np.arange(y.size, dtype=float)
            traces.append(
                Series(
                    x=time * time_scale,
                    y=y * value_scale * _weight(ods, recipe.weight_path, index),
                    label=_entry_prefix(
                        entry_label,
                        _channel_label(ods, recipe.label_path, index, f"#{index}"),
                    ),
                )
            )
        return traces

    y = _array(ods, recipe.y_path)
    if y is None:
        return []
    time = _first_time(ods, recipe.x_paths)
    if time is None or time.size != y.size:
        time = np.arange(y.size, dtype=float)
    return [Series(x=time * time_scale, y=y * value_scale, label=entry_label)]


def _build_line_series(
    entries: Sequence[tuple[str, Any]], recipe: LineRecipe, **options: Any
) -> LineSeries:
    xunit = options.get("xunit", "s")
    yunit = options.get("yunit") or recipe.y_unit
    traces: list[Series] = []
    for entry_label, ods in entries:
        traces.extend(
            _build_line_traces(
                ods, recipe, entry_label=entry_label,
                channels=options.get("channels"), xunit=xunit, yunit=yunit,
            )
        )
    return LineSeries(
        series=tuple(traces),
        x_label=recipe.x_label, x_unit=xunit,
        y_label=recipe.y_label, y_unit=yunit,
        title=options.get("title", recipe.title),
        x_limits=options.get("x_limits"),
        log_y=bool(options.get("log_y", False)),
    )


_COORDINATE_LABELS = {
    "rho_tor_norm": "Normalized Toroidal Flux (rho_N)",
    "psi_norm": "Normalized Poloidal Flux (psi_N)",
    "r_major": "Major Radius R [m]",
    "r_minor": "Minor Radius r [m]",
}

_EQUILIBRIUM_COORDINATES = {
    "rho_tor_norm": "equilibrium.time_slice.{i}.profiles_1d.rho_tor_norm",
    "psi_norm": "equilibrium.time_slice.{i}.profiles_1d.psi_norm",
    "r_major": "equilibrium.time_slice.{i}.profiles_1d.r_outboard",
    "r_minor": "equilibrium.time_slice.{i}.profiles_1d.r_minor",
}


def _profile_coordinate(recipe: ProfileRecipe, name: str) -> str | None:
    if recipe.coordinate_paths:
        return recipe.coordinate_paths.get(name)
    return _EQUILIBRIUM_COORDINATES.get(name)


def _build_profile_1d(
    entries: Sequence[tuple[str, Any]], recipe: ProfileRecipe, **options: Any
) -> Profile1D:
    coordinate = options.get("coordinate") or recipe.default_coordinate
    time_slice = options.get("time_slice", 0)
    traces: list[Series] = []
    for entry_label, ods in entries:
        if recipe.index == "channel":
            indices = _resolve_indices(ods, recipe.y_path, options.get("channels"))
            x_values, y_values = [], []
            for index in indices:
                x = _get(ods, _profile_coordinate(recipe, coordinate).format(i=index))
                y = _get(ods, recipe.y_path.format(i=index))
                if x is None or y is None:
                    continue
                x_values.append(float(np.asarray(x, dtype=float).ravel()[0]))
                y_flat = np.asarray(y, dtype=float).ravel()
                position = min(time_slice, y_flat.size - 1) if y_flat.size else 0
                y_values.append(float(y_flat[position]) if y_flat.size else np.nan)
            if x_values:
                order = np.argsort(x_values)
                traces.append(
                    Series(
                        x=np.asarray(x_values)[order],
                        y=np.asarray(y_values)[order],
                        label=entry_label,
                        style={"marker": "o", "linestyle": "-"},
                    )
                )
            continue

        y = _array(ods, recipe.y_path.format(i=time_slice))
        for fallback in recipe.fallback_y_paths:
            if y is None:
                y = _array(ods, fallback.format(i=time_slice))
        if y is None:
            continue
        coordinate_path = _profile_coordinate(recipe, coordinate)
        x = _array(ods, coordinate_path.format(i=time_slice)) if coordinate_path else None
        if x is None or x.size != y.size:
            x = np.linspace(0.0, 1.0, y.size)
        traces.append(Series(x=x, y=y, label=entry_label))

    return Profile1D(
        series=tuple(traces),
        coordinate_label=_COORDINATE_LABELS.get(coordinate, coordinate),
        y_label=recipe.y_label,
        y_unit=options.get("yunit") or recipe.y_unit,
        title=options.get("title", recipe.y_label),
    )


def _build_geometry(ods: Any, recipe: GeometryRecipe, **options: Any) -> GeometryLayers:
    time_slice = options.get("time_slice", 0)
    layers: list[GeometryLayer] = []
    for kind, r_template, z_template, container, label_template, style in recipe.layers:
        if container == "equilibrium.time_slice":
            indices = [time_slice]
        else:
            indices = list(range(_count(ods, container)))
        if kind == "points":
            r_values, z_values = [], []
            for index in indices:
                r = _get(ods, r_template.format(i=index))
                z = _get(ods, z_template.format(i=index))
                if r is None or z is None:
                    continue
                r_values.append(float(np.asarray(r, dtype=float).ravel()[0]))
                z_values.append(float(np.asarray(z, dtype=float).ravel()[0]))
            if r_values:
                layers.append(
                    GeometryLayer(r=r_values, z=z_values, kind="points",
                                  label=recipe.title, style=style)
                )
            continue
        for index in indices:
            r = _array(ods, r_template.format(i=index))
            z = _array(ods, z_template.format(i=index))
            if r is None or z is None or r.size != z.size:
                continue
            layers.append(
                GeometryLayer(
                    r=r, z=z, kind=kind,
                    label=_channel_label(ods, label_template, index, "")
                    if label_template else "",
                    style=style,
                )
            )
    return GeometryLayers(
        layers=tuple(layers), x_label=recipe.x_label, y_label=recipe.y_label,
        title=options.get("title", recipe.title),
    )


def _build_field_2d(ods: Any, recipe: FieldRecipe, **options: Any) -> Field2D:
    time_slice = options.get("time_slice", 0)
    r = _array(ods, recipe.r_path.format(i=time_slice))
    z = _array(ods, recipe.z_path.format(i=time_slice))
    values = _array(ods, recipe.value_path.format(i=time_slice))
    if r is None or z is None or values is None:
        raise ValueError(
            f"{recipe.value_path.format(i=time_slice)} and its (R, Z) grid are required"
        )
    if values.shape != (z.size, r.size):
        values = values.T
    overlays = list(_wall_layers(ods))
    if recipe.boundary_paths:
        boundary_r = _array(ods, recipe.boundary_paths[0].format(i=time_slice))
        boundary_z = _array(ods, recipe.boundary_paths[1].format(i=time_slice))
        if boundary_r is not None and boundary_z is not None:
            overlays.append(
                GeometryLayer(r=boundary_r, z=boundary_z, kind="polygon",
                              label="Boundary", style={"color": "#e41a1c"})
            )
    return Field2D(
        r=r, z=z, values=values, value_label=recipe.value_label,
        contour_levels=options.get("contour_levels"),
        overlays=tuple(overlays), title=options.get("title", recipe.title),
    )


def _build_spectrogram(
    ods: Any, recipe: SpectrogramRecipe, **options: Any
) -> Spectrogram:
    from vaft.process import mirnov_spectrogram as compute_spectrogram

    index = int(options.get("channel", 0))
    signal = _array(ods, recipe.signal_path.format(i=index))
    if signal is None:
        raise ValueError(f"{recipe.signal_path.format(i=index)} is not available")
    time = _first_time(ods, recipe.time_paths, i=index)
    if time is None or time.size != signal.size:
        time = np.arange(signal.size, dtype=float)
    sample_rate = options.get("sample_rate")
    if sample_rate is None:
        steps = np.diff(time)
        positive = steps[steps > 0]
        sample_rate = 1.0 / float(np.median(positive)) if positive.size else 1.0
    result = compute_spectrogram(
        time, signal, sample_rate=float(sample_rate),
        window_size=int(options.get("window_size", 500)),
        time_resolution=int(options.get("time_resolution", 1)),
    )
    return Spectrogram.from_result(
        result,
        max_frequency=options.get("max_frequency"),
        cmap=options.get("cmap", "hot_r"),
        title=_channel_label(ods, recipe.label_path, index, f"channel {index}"),
        value_label=recipe.value_label,
    )


def _build_panels(
    entries: Sequence[tuple[str, Any]], recipe: PanelRecipe, **options: Any
) -> Panels:
    members = []
    for name in recipe.members:
        if not any(entry_supports(ods, name) for _, ods in entries):
            continue
        members.append(build_model(name, entries, **options))
    if not members:
        raise ValueError(
            "none of the panels " + ", ".join(recipe.members) + " have data in this input"
        )
    return Panels(
        models=tuple(members),
        ncols=recipe.ncols,
        share_x=recipe.share_x,
        suptitle=options.get("title", recipe.suptitle),
    )


def build_model(name: str, entries: Sequence[tuple[str, Any]], **options: Any) -> Any:
    """Build the view model for canonical plot ``name`` from ``entries``.

    ``entries`` is the ``(label, ods)`` sequence produced by
    :func:`normalize_entries`.  Single-ODS families (2D fields, geometry,
    spectrograms) use the first entry and ignore the rest.
    """
    try:
        recipe = RECIPES[name]
    except KeyError:
        raise KeyError(
            f"no OMAS extraction recipe for {name!r}; "
            "use vaft.omas.available_plots() to list the supported plots"
        ) from None
    if not entries:
        raise ValueError("no ODS entries were supplied")

    if isinstance(recipe, LineRecipe):
        return _build_line_series(entries, recipe, **options)
    if isinstance(recipe, ProfileRecipe):
        return _build_profile_1d(entries, recipe, **options)
    if isinstance(recipe, PanelRecipe):
        return _build_panels(entries, recipe, **options)
    if isinstance(recipe, GeometryRecipe):
        return _build_geometry(entries[0][1], recipe, **options)
    if isinstance(recipe, FieldRecipe):
        return _build_field_2d(entries[0][1], recipe, **options)
    if isinstance(recipe, SpectrogramRecipe):
        return _build_spectrogram(entries[0][1], recipe, **options)
    if isinstance(recipe, CallableRecipe):
        return recipe.builder(entries[0][1], **options)
    raise TypeError(f"unsupported recipe type {type(recipe).__name__} for {name!r}")
