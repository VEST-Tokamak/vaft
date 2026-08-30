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
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vaft.plot.models import (
    Field2D,
    GeometryLayer,
    GeometryLayers,
    Image2D,
    ImageSequence,
    LineSeries,
    Panels,
    PowerSpectrum,
    Profile1D,
    ReferenceSlope,
    Series,
    Spectrogram,
)
from vaft.plot.registry import get_spec

from vaft.formula.statistics import noise_band, rms

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


def _squeeze_energy_band(values: np.ndarray | None, *, where: str) -> np.ndarray | None:
    """Reduce a ``(energy_band, time)`` trace to the 1D series renderers expect.

    IMAS stores soft X-ray ``brightness.data`` as ``(energy_band, time)``, and
    older VEST ODS files used ``(time, 1)``.  Both carry a single band, so both
    reduce to one trace; several real bands need the caller to pick one.
    """
    if values is None or values.ndim <= 1:
        return values
    if values.shape[0] == 1:
        return values[0]
    if values.shape[-1] == 1:
        return values[..., 0]
    raise ValueError(
        f"{where} holds {values.shape[0]} energy bands; select one before plotting."
    )


def _first_array(
    ods: Any, candidates: Sequence[str], **substitutions: Any
) -> np.ndarray | None:
    """The first candidate path that holds a usable 1D trace.

    Diagnostics that store the same physical signal under different IDS leaves --
    soft X-rays under ``brightness`` here and ``power`` elsewhere -- give the
    recipe a candidate list rather than forcing one spelling.  A candidate whose
    trace holds several real energy bands is not usable as-is; the remaining
    candidates are still tried, and its error is raised only when nothing else
    provides the signal (so the caller learns *why* rather than "not available").
    """
    multi_band_error: ValueError | None = None
    for candidate in candidates:
        path = candidate.format(**substitutions) if substitutions else candidate
        array = _array(ods, path)
        if array is None:
            continue
        try:
            return _squeeze_energy_band(array, where=path)
        except ValueError as error:
            multi_band_error = error
    if multi_band_error is not None:
        raise multi_band_error
    return None


def _first_time(
    ods: Any, candidates: Sequence[str], **substitutions: Any
) -> np.ndarray | None:
    for candidate in candidates:
        path = candidate.format(**substitutions) if substitutions else candidate
        array = _array(ods, path)
        if array is not None:
            return _squeeze_energy_band(array, where=path)
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
    #: Alternative spellings of ``y_path``, tried in order when it holds no data.
    #: Diagnostics whose signal lives under different IDS leaves across sources
    #: (soft X-rays under ``brightness`` here, ``power`` elsewhere) list them here.
    fallback_y_paths: tuple[str, ...] = ()
    #: Optional scalar ODS path (e.g. ``"tf.r0"``) whose value divides ``y_path``.
    #: Missing or zero divides by 1.0 rather than raising or producing inf/nan.
    divide_by_path: str = ""
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
    values_order: str = "zr"


@dataclass(frozen=True)
class SpectrogramRecipe:
    """How to read a ``Spectrogram`` from an ODS."""

    signal_path: str
    fallback_signal_paths: tuple[str, ...] = ()
    time_paths: tuple[str, ...] = ()
    container: str = ""
    label_path: str = ""
    value_label: str = "Magnitude"


@dataclass(frozen=True)
class PowerSpectrumRecipe:
    """How to read a ``PowerSpectrum`` from an ODS."""

    signal_path: str
    fallback_signal_paths: tuple[str, ...] = ()
    time_paths: tuple[str, ...] = ()
    container: str = ""
    label_path: str = ""
    value_label: str = "PSD"


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
        y_path="magnetics.ip.0.data",
        x_paths=("magnetics.ip.0.time",) + _MAGNETICS_TIME,
        y_label="Plasma Current",
        y_unit="A",
        title="Plasma Current",
    ),
    "magnetics_time_diamagnetic_flux": LineRecipe(
        y_path="magnetics.diamagnetic_flux.0.data",
        x_paths=("magnetics.diamagnetic_flux.0.time",) + _MAGNETICS_TIME,
        y_label="Diamagnetic Flux",
        y_unit="Wb",
        title="Diamagnetic Flux",
    ),
    "magnetics_time_flux_loop_flux": LineRecipe(
        y_path="magnetics.flux_loop.{i}.flux.data",
        index="channel",
        x_paths=(
            "magnetics.flux_loop.{i}.flux.time",
            "magnetics.flux_loop.time",
            "magnetics.time",
        ),
        y_label="Poloidal Flux",
        y_unit="Wb",
        label_path="magnetics.flux_loop.{i}.name",
        title="Flux Loop Flux",
    ),
    "magnetics_time_flux_loop_voltage": LineRecipe(
        y_path="magnetics.flux_loop.{i}.voltage.data",
        index="channel",
        x_paths=(
            "magnetics.flux_loop.{i}.voltage.time",
            "magnetics.flux_loop.time",
            "magnetics.time",
        ),
        y_label="Loop Voltage",
        y_unit="V",
        label_path="magnetics.flux_loop.{i}.name",
        title="Flux Loop Voltage",
    ),
    "magnetics_time_b_field_pol_probe_field": LineRecipe(
        y_path="magnetics.b_field_pol_probe.{i}.field.data",
        index="channel",
        x_paths=(
            "magnetics.b_field_pol_probe.{i}.field.time",
            "magnetics.b_field_pol_probe.time",
            "magnetics.time",
        ),
        y_label="Poloidal Field",
        y_unit="T",
        label_path="magnetics.b_field_pol_probe.{i}.name",
        title="B-field Probes",
    ),
    "magnetics_time_mirnov_voltage": LineRecipe(
        y_path="magnetics.b_field_pol_probe.{i}.voltage.data",
        index="channel",
        x_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
        y_label="Mirnov Signal",
        y_unit="V",
        label_path="magnetics.b_field_pol_probe.{i}.name",
        title="Mirnov Coils",
    ),
    # --- pf_active -----------------------------------------------------------
    "pf_active_time_current": LineRecipe(
        y_path="pf_active.coil.{i}.current.data",
        index="channel",
        x_paths=("pf_active.coil.{i}.current.time", "pf_active.time"),
        y_label="Coil Current",
        y_unit="A",
        label_path="pf_active.coil.{i}.name",
        title="PF Coil Currents",
    ),
    "pf_active_time_current_turns": LineRecipe(
        y_path="pf_active.coil.{i}.current.data",
        index="channel",
        x_paths=("pf_active.coil.{i}.current.time", "pf_active.time"),
        y_label="Coil Ampere-turns",
        y_unit="A-turns",
        label_path="pf_active.coil.{i}.name",
        weight_path="pf_active.coil.{i}.element.:.turns_with_sign",
        title="PF Coil Ampere-turns",
    ),
    # --- equilibrium global quantities ---------------------------------------
    "equilibrium_time_plasma_current": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.ip",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Plasma Current",
        y_unit="A",
        title="Equilibrium Plasma Current",
    ),
    "equilibrium_time_li": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.li_3",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Internal Inductance li_3",
        title="Internal Inductance",
    ),
    "equilibrium_time_beta_pol": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.beta_pol",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Poloidal Beta",
        title="Poloidal Beta",
    ),
    "equilibrium_time_beta_tor": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.beta_tor",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Toroidal Beta",
        title="Toroidal Beta",
    ),
    "equilibrium_time_beta_n": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.beta_normal",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Normalized Beta",
        title="Normalized Beta",
    ),
    "equilibrium_time_w_mhd": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.energy_mhd",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="MHD Stored Energy",
        y_unit="J",
        title="MHD Stored Energy",
    ),
    "equilibrium_time_w_mag": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.energy_mag",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Magnetic Stored Energy",
        y_unit="J",
        title="Magnetic Stored Energy",
    ),
    "equilibrium_time_w_tot": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.energy_total",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Total Stored Energy",
        y_unit="J",
        title="Total Stored Energy",
    ),
    "equilibrium_time_q0": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.q_axis",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="q on axis",
        title="q0",
    ),
    "equilibrium_time_q95": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.q_95",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="q95",
        title="q95",
    ),
    "equilibrium_time_qa": LineRecipe(
        y_path="equilibrium.time_slice.{i}.global_quantities.qa",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="qa",
        title="qa",
    ),
    "equilibrium_time_major_radius": LineRecipe(
        y_path="equilibrium.time_slice.{i}.boundary.geometric_axis.r",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Major Radius",
        y_unit="m",
        title="Equilibrium Major Radius",
    ),
    "equilibrium_time_diamagnetic_flux": LineRecipe(
        y_path="equilibrium.time_slice.{i}.constraints.diamagnetic_flux.measured",
        index="time_slice",
        x_paths=_EQ_TIME,
        y_label="Diamagnetic Flux",
        y_unit="Wb",
        title="Diamagnetic Flux Constraint",
    ),
    # --- tf ------------------------------------------------------------------
    "tf_time_b_field_tor": LineRecipe(
        y_path="tf.b_field_tor_vacuum_r.data",
        x_paths=("tf.b_field_tor_vacuum_r.time", "tf.time"),
        y_label="Toroidal Field",
        y_unit="T",
        title="Toroidal Field",
        # tf.b_field_tor_vacuum_r.data is B_t * R [T*m]; divide by the reference
        # radius to recover the field itself, matching the legacy renderer.
        divide_by_path="tf.r0",
    ),
    "tf_time_b_field_tor_vacuum_r": LineRecipe(
        y_path="tf.b_field_tor_vacuum_r.data",
        x_paths=("tf.b_field_tor_vacuum_r.time", "tf.time"),
        y_label="B_t * R",
        y_unit="T m",
        title="Vacuum B_t * R",
    ),
    "tf_time_coil_current": LineRecipe(
        y_path="tf.coil.{i}.current.data",
        index="channel",
        x_paths=("tf.coil.{i}.current.time", "tf.time"),
        y_label="TF Coil Current",
        y_unit="A",
        label_path="tf.coil.{i}.name",
        title="TF Coil Current",
    ),
    # --- other diagnostics ---------------------------------------------------
    "spectrometer_uv_time_intensity": LineRecipe(
        y_path="spectrometer_uv.channel.{i}.processed_line.0.intensity.data",
        index="channel",
        x_paths=("spectrometer_uv.time",),
        y_label="Line Intensity",
        y_unit="a.u.",
        label_path="spectrometer_uv.channel.{i}.name",
        title="UV Line Intensity",
    ),
    "barometry_time_pressure": LineRecipe(
        y_path="barometry.gauge.{i}.pressure.data",
        index="channel",
        x_paths=("barometry.gauge.{i}.pressure.time",),
        y_label="Neutral Pressure",
        y_unit="Pa",
        label_path="barometry.gauge.{i}.name",
        title="Neutral Pressure",
    ),
    "soft_x_rays_time_power": LineRecipe(
        # VEST maps SXR digitizer traces to ``brightness`` (what IMAS defines for
        # a detector signal); ``power`` is kept for externally sourced ODS.
        y_path="soft_x_rays.channel.{i}.brightness.data",
        fallback_y_paths=("soft_x_rays.channel.{i}.power.data",),
        index="channel",
        x_paths=(
            "soft_x_rays.channel.{i}.brightness.time",
            "soft_x_rays.channel.{i}.power.time",
            "soft_x_rays.time",
        ),
        y_label="Soft X-ray Signal",
        y_unit="a.u.",
        label_path="soft_x_rays.channel.{i}.name",
        title="Soft X-ray Signals",
    ),
    "interferometer_time_n_e_line": LineRecipe(
        y_path="interferometer.channel.{i}.n_e_line.data", index="channel",
        x_paths=("interferometer.channel.{i}.n_e_line.time", "interferometer.time"),
        y_label="Line-integrated Electron Density", y_unit="10^18 m^-2", scale=1e-18,
        label_path="interferometer.channel.{i}.name", title="Interferometer Line Density",
    ),
    "thomson_scattering_time_electron_temperature": LineRecipe(
        y_path="thomson_scattering.channel.{i}.t_e.data",
        index="channel",
        x_paths=("thomson_scattering.channel.{i}.t_e.time", "thomson_scattering.time"),
        y_label="Electron Temperature",
        y_unit="eV",
        label_path="thomson_scattering.channel.{i}.name",
        title="Thomson T_e",
    ),
    "thomson_scattering_time_electron_density": LineRecipe(
        y_path="thomson_scattering.channel.{i}.n_e.data",
        index="channel",
        x_paths=("thomson_scattering.channel.{i}.n_e.time", "thomson_scattering.time"),
        y_label="Electron Density",
        y_unit="m^-3",
        label_path="thomson_scattering.channel.{i}.name",
        title="Thomson n_e",
    ),
    "charge_exchange_time_ion_temperature": LineRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.t_i.data",
        index="channel",
        x_paths=("charge_exchange.channel.{i}.ion.0.t_i.time", "charge_exchange.time"),
        y_label="Ion Temperature",
        y_unit="eV",
        label_path="charge_exchange.channel.{i}.name",
        title="CES T_i",
    ),
    "charge_exchange_time_velocity_tor": LineRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.velocity_tor.data",
        index="channel",
        x_paths=(
            "charge_exchange.channel.{i}.ion.0.velocity_tor.time",
            "charge_exchange.time",
        ),
        y_label="Toroidal Rotation",
        y_unit="m/s",
        label_path="charge_exchange.channel.{i}.name",
        title="CES v_tor",
    ),
    "core_profiles_time_electron_temperature": LineRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.temperature",
        index="time_slice_mean",
        x_paths=("core_profiles.time",),
        y_label="<T_e>",
        y_unit="eV",
        title="Volume-averaged T_e",
    ),
    "core_profiles_time_electron_density": LineRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.density",
        index="time_slice_mean",
        x_paths=("core_profiles.time",),
        y_label="<n_e>",
        y_unit="m^-3",
        title="Volume-averaged n_e",
    ),
    # --- 1D profiles ---------------------------------------------------------
    "equilibrium_profile_pressure": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.pressure",
        y_label="Pressure",
        y_unit="Pa",
    ),
    "equilibrium_profile_q": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.q",
        y_label="Safety Factor q",
    ),
    "equilibrium_profile_j_tor": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.j_tor",
        y_label="Toroidal Current Density",
        y_unit="A/m^2",
    ),
    "equilibrium_profile_pprime": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.dpressure_dpsi",
        fallback_y_paths=("equilibrium.time_slice.{i}.profiles_1d.pprime",),
        y_label="dp/dpsi",
        y_unit="Pa/Wb",
    ),
    "equilibrium_profile_f": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.f",
        y_label="F = R B_t",
        y_unit="T m",
    ),
    "equilibrium_profile_ffprime": ProfileRecipe(
        y_path="equilibrium.time_slice.{i}.profiles_1d.f_df_dpsi",
        fallback_y_paths=("equilibrium.time_slice.{i}.profiles_1d.ffprime",),
        y_label="F dF/dpsi",
        y_unit="T^2 m^2/Wb",
    ),
    "core_profiles_profile_electron_temperature": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.temperature",
        coordinate_paths={
            "rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm",
            "psi_norm": "core_profiles.profiles_1d.{i}.grid.rho_pol_norm",
        },
        slice_container="core_profiles.profiles_1d",
        y_label="Electron Temperature",
        y_unit="eV",
    ),
    "core_profiles_profile_electron_density": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.electrons.density",
        coordinate_paths={
            "rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm",
            "psi_norm": "core_profiles.profiles_1d.{i}.grid.rho_pol_norm",
        },
        slice_container="core_profiles.profiles_1d",
        y_label="Electron Density",
        y_unit="m^-3",
    ),
    "core_profiles_profile_ion_temperature": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.ion.0.temperature",
        coordinate_paths={
            "rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm"
        },
        slice_container="core_profiles.profiles_1d",
        y_label="Ion Temperature",
        y_unit="eV",
    ),
    "core_profiles_profile_pressure": ProfileRecipe(
        y_path="core_profiles.profiles_1d.{i}.pressure_thermal",
        coordinate_paths={
            "rho_tor_norm": "core_profiles.profiles_1d.{i}.grid.rho_tor_norm"
        },
        slice_container="core_profiles.profiles_1d",
        y_label="Thermal Pressure",
        y_unit="Pa",
    ),
    "thomson_scattering_profile_electron_temperature": ProfileRecipe(
        y_path="thomson_scattering.channel.{i}.t_e.data",
        index="channel",
        coordinate_paths={"r_major": "thomson_scattering.channel.{i}.position.r"},
        default_coordinate="r_major",
        slice_container="thomson_scattering.channel",
        y_label="Electron Temperature",
        y_unit="eV",
    ),
    "thomson_scattering_profile_electron_density": ProfileRecipe(
        y_path="thomson_scattering.channel.{i}.n_e.data",
        index="channel",
        coordinate_paths={"r_major": "thomson_scattering.channel.{i}.position.r"},
        default_coordinate="r_major",
        slice_container="thomson_scattering.channel",
        y_label="Electron Density",
        y_unit="m^-3",
    ),
    "charge_exchange_profile_ion_temperature": ProfileRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.t_i.data",
        index="channel",
        coordinate_paths={"r_major": "charge_exchange.channel.{i}.position.r.data"},
        default_coordinate="r_major",
        slice_container="charge_exchange.channel",
        y_label="Ion Temperature",
        y_unit="eV",
    ),
    "charge_exchange_profile_velocity_tor": ProfileRecipe(
        y_path="charge_exchange.channel.{i}.ion.0.velocity_tor.data",
        index="channel",
        coordinate_paths={"r_major": "charge_exchange.channel.{i}.position.r.data"},
        default_coordinate="r_major",
        slice_container="charge_exchange.channel",
        y_label="Toroidal Rotation",
        y_unit="m/s",
    ),
    # --- 2D fields -----------------------------------------------------------
    "equilibrium_field_psi": FieldRecipe(
        r_path="equilibrium.time_slice.{i}.profiles_2d.0.grid.dim1",
        z_path="equilibrium.time_slice.{i}.profiles_2d.0.grid.dim2",
        value_path="equilibrium.time_slice.{i}.profiles_2d.0.psi",
        value_label="Poloidal Flux [Wb]",
        boundary_paths=(
            "equilibrium.time_slice.{i}.boundary.outline.r",
            "equilibrium.time_slice.{i}.boundary.outline.z",
        ),
        title="Poloidal Flux",
        # OMAS equilibrium profiles_2d uses (dim1=R, dim2=Z).  Field2D and
        # Matplotlib consume (Z, R); square EFIT grids previously concealed
        # this transpose because their shapes are identical.
        values_order="rz",
    ),
    # --- geometry ------------------------------------------------------------
    "pf_active_geometry_poloidal": GeometryRecipe(
        layers=(
            (
                "polygon",
                "pf_active.coil.{i}.element.0.geometry.outline.r",
                "pf_active.coil.{i}.element.0.geometry.outline.z",
                "pf_active.coil",
                "pf_active.coil.{i}.name",
                {},
            ),
        ),
        title="PF Coils",
    ),
    "pf_passive_geometry_poloidal": GeometryRecipe(
        layers=(
            (
                "polygon",
                "pf_passive.loop.{i}.element.0.geometry.outline.r",
                "pf_passive.loop.{i}.element.0.geometry.outline.z",
                "pf_passive.loop",
                "pf_passive.loop.{i}.name",
                {},
            ),
        ),
        title="Passive Structure",
    ),
    "wall_geometry_poloidal": GeometryRecipe(
        layers=(
            (
                "polygon",
                "wall.description_2d.0.limiter.unit.{i}.outline.r",
                "wall.description_2d.0.limiter.unit.{i}.outline.z",
                "wall.description_2d.0.limiter.unit",
                "",
                {"color": "0.4"},
            ),
        ),
        title="First Wall",
    ),
    "magnetics_geometry_poloidal": GeometryRecipe(
        layers=(
            (
                "points",
                "magnetics.flux_loop.{i}.position.0.r",
                "magnetics.flux_loop.{i}.position.0.z",
                "magnetics.flux_loop",
                "",
                {"marker": "s", "markersize": 3, "color": "#377eb8"},
            ),
            (
                "points",
                "magnetics.b_field_pol_probe.{i}.position.r",
                "magnetics.b_field_pol_probe.{i}.position.z",
                "magnetics.b_field_pol_probe",
                "",
                {"marker": "x", "markersize": 4, "color": "#ff7f00"},
            ),
        ),
        title="Magnetic Diagnostics",
    ),
    "equilibrium_geometry_boundary": GeometryRecipe(
        layers=(
            (
                "polygon",
                "equilibrium.time_slice.{i}.boundary.outline.r",
                "equilibrium.time_slice.{i}.boundary.outline.z",
                "equilibrium.time_slice",
                "",
                {"color": "#e41a1c"},
            ),
        ),
        title="Plasma Boundary",
    ),
    "thomson_scattering_geometry_poloidal": GeometryRecipe(
        layers=(
            (
                "points",
                "thomson_scattering.channel.{i}.position.r",
                "thomson_scattering.channel.{i}.position.z",
                "thomson_scattering.channel",
                "",
                {"marker": "o", "markersize": 3},
            ),
        ),
        title="Thomson Scattering Positions",
    ),
    "charge_exchange_geometry_poloidal": GeometryRecipe(
        layers=(
            (
                "points",
                "charge_exchange.channel.{i}.position.r.data",
                "charge_exchange.channel.{i}.position.z.data",
                "charge_exchange.channel",
                "",
                {"marker": "d", "markersize": 3},
            ),
        ),
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
        signal_path="soft_x_rays.channel.{i}.brightness.data",
        fallback_signal_paths=("soft_x_rays.channel.{i}.power.data",),
        time_paths=(
            "soft_x_rays.channel.{i}.brightness.time",
            "soft_x_rays.channel.{i}.power.time",
            "soft_x_rays.time",
        ),
        container="soft_x_rays.channel",
        label_path="soft_x_rays.channel.{i}.name",
    ),
    "interferometer_spectrogram": CallableRecipe(
        builder=lambda ods, **options: _build_interferometer_spectrogram(ods, **options),
        description="Time-frequency map of one interferometer channel's density fluctuation.",
    ),
    # --- power spectra -------------------------------------------------------
    "magnetics_spectrum_mirnov": PowerSpectrumRecipe(
        signal_path="magnetics.b_field_pol_probe.{i}.voltage.data",
        time_paths=("magnetics.b_field_pol_probe.{i}.voltage.time", "magnetics.time"),
        container="magnetics.b_field_pol_probe",
        label_path="magnetics.b_field_pol_probe.{i}.name",
        value_label="PSD [V^2/Hz]",
    ),
    "soft_x_rays_spectrum": PowerSpectrumRecipe(
        signal_path="soft_x_rays.channel.{i}.brightness.data",
        fallback_signal_paths=("soft_x_rays.channel.{i}.power.data",),
        time_paths=(
            "soft_x_rays.channel.{i}.brightness.time",
            "soft_x_rays.channel.{i}.power.time",
            "soft_x_rays.time",
        ),
        container="soft_x_rays.channel",
        label_path="soft_x_rays.channel.{i}.name",
    ),
    "interferometer_spectrum": PowerSpectrumRecipe(
        signal_path="interferometer.channel.{i}.n_e_line.data",
        time_paths=("interferometer.channel.{i}.n_e_line.time", "interferometer.time"),
        container="interferometer.channel",
        label_path="interferometer.channel.{i}.name",
    ),
    # --- composites ----------------------------------------------------------
    "summary_time_energy": PanelRecipe(
        members=(
            "equilibrium_time_w_mhd",
            "equilibrium_time_w_mag",
            "equilibrium_time_w_tot",
        ),
        suptitle="Stored Energy",
    ),
    "summary_time_beta": PanelRecipe(
        members=(
            "equilibrium_time_beta_pol",
            "equilibrium_time_beta_tor",
            "equilibrium_time_beta_n",
        ),
        suptitle="Beta",
    ),
    "summary_time_voltage_consumption": PanelRecipe(
        members=("magnetics_time_ip", "magnetics_time_flux_loop_voltage"),
        suptitle="Voltage Consumption",
    ),
    "equilibrium_time_virial": PanelRecipe(
        members=(
            "equilibrium_time_beta_pol",
            "equilibrium_time_li",
            "equilibrium_time_w_mhd",
        ),
        suptitle="Virial Equilibrium Quantities",
    ),
    "electromagnetics_time_current": PanelRecipe(
        members=("magnetics_time_ip", "pf_active_time_current"),
        suptitle="Electromagnetic Currents",
    ),
    "core_profiles_time_volume_averaged": PanelRecipe(
        members=(
            "core_profiles_time_electron_temperature",
            "core_profiles_time_electron_density",
        ),
        suptitle="Volume-averaged Core Profiles",
    ),
    "spectrometer_uv_time_impurity": PanelRecipe(
        members=("magnetics_time_ip", "spectrometer_uv_time_intensity"),
        suptitle="Impurity Line Intensity",
    ),
    "magnetics_overview": PanelRecipe(
        members=(
            "magnetics_time_ip",
            "pf_active_time_current",
            "magnetics_time_flux_loop_flux",
            "magnetics_time_b_field_pol_probe_field",
        ),
        ncols=2,
        share_x=False,
        suptitle="Shot Diagnostics Overview",
    ),
    "equilibrium_overview": PanelRecipe(
        members=(
            "equilibrium_time_plasma_current",
            "equilibrium_time_beta_pol",
            "equilibrium_time_li",
            "equilibrium_time_q95",
        ),
        ncols=2,
        share_x=False,
        suptitle="Equilibrium Analysis Overview",
    ),
    "magnetics_time_impa_field": CallableRecipe(
        builder=lambda ods, **options: _build_impa_lines(ods, quantity="field", **options),
        description="Compensated internal Bz from the IMPA Hall-probe array.",
    ),
    "magnetics_time_impa_voltage": CallableRecipe(
        builder=lambda ods, **options: _build_impa_lines(ods, quantity="voltage", **options),
        description="Raw IMPA Hall-probe voltages.",
    ),
    "magnetics_profile_impa_tf": CallableRecipe(
        builder=lambda ods, **options: _build_impa_tf_profile(ods, **options),
        description="IMPA measured field against probe radius with the 1/R model.",
    ),
    "magnetics_overview_impa": PanelRecipe(
        members=("magnetics_time_impa_voltage", "magnetics_time_impa_field",
                 "magnetics_profile_impa_tf", "tf_time_coil_current"),
        ncols=2, share_x=False, suptitle="IMPA Validation Overview",
    ),
    "soft_x_rays_overview": PanelRecipe(
        members=("soft_x_rays_time_power", "soft_x_rays_geometry_lines_of_sight"),
        share_x=False,
        suptitle="Soft X-ray Overview",
    ),
    "interferometer_overview": PanelRecipe(
        members=("interferometer_time_n_e_line", "interferometer_spectrogram"),
        suptitle="Interferometer Overview",
    ),
}


# ---------------------------------------------------------------------------
# Builders for the plots that need computation rather than a plain path read
# ---------------------------------------------------------------------------


def _build_interferometer_spectrogram(ods: Any, *, channel: int = 0, **options: Any) -> Spectrogram:
    """Time-frequency map of one interferometer channel's density *fluctuation*.

    ``n_e_line`` is a line-integrated density, not an AC-coupled signal like a
    Mirnov voltage: its quasi-DC trend (up to ~1e19 m^-2) dwarfs any kHz-scale
    fluctuation, so a spectrogram of the raw trace is dominated by the ~0 Hz
    bin and shows nothing else. This high-pass filters the trend out first.
    The default window/step also adapt to the actual sample rate rather than
    assuming the ~250 kHz Mirnov default, since interferometer MAT files can
    be sampled far faster.
    """
    from scipy.signal import butter, filtfilt

    from vaft.process import mirnov_spectrogram as compute_spectrogram

    index = int(channel)
    signal_values = _array(ods, f"interferometer.channel.{index}.n_e_line.data")
    if signal_values is None:
        raise ValueError(f"interferometer.channel.{index}.n_e_line.data is not available")
    time = _first_time(
        ods,
        (f"interferometer.channel.{index}.n_e_line.time", "interferometer.time"),
    )
    if time is None or time.size != signal_values.size:
        time = np.arange(signal_values.size, dtype=float)

    sample_rate = options.get("sample_rate")
    if sample_rate is None:
        steps = np.diff(time)
        positive = steps[steps > 0]
        sample_rate = 1.0 / float(np.median(positive)) if positive.size else 1.0
    sample_rate = float(sample_rate)
    nyquist = sample_rate / 2.0

    fluctuation = signal_values
    highpass_cutoff = float(options.get("highpass_cutoff", 200.0))
    if signal_values.size > 32 and 0.0 < highpass_cutoff < nyquist:
        b, a = butter(4, highpass_cutoff / nyquist, btype="highpass")
        fluctuation = filtfilt(b, a, signal_values)

    window_size = options.get("window_size")
    if window_size is None:
        target_df = float(options.get("target_df", 300.0))
        window_size = max(64, int(round(sample_rate / target_df)))
        window_size -= window_size % 2
    else:
        window_size = int(window_size)
    time_resolution = options.get("time_resolution")
    time_resolution = int(time_resolution) if time_resolution else max(1, window_size // 40)

    result = compute_spectrogram(
        time, fluctuation, sample_rate=sample_rate,
        window_size=window_size, time_resolution=time_resolution,
    )
    return Spectrogram.from_result(
        result,
        max_frequency=options.get("max_frequency"),
        cmap=options.get("cmap", "turbo"),
        title=_channel_label(ods, "interferometer.channel.{i}.name", index, f"channel {index}"),
        value_label="Fluctuation Magnitude",
    )


#: Identifier prefix written by :mod:`vaft.machine_mapping.impa`.  IMPA probes
#: are located semantically so their position in the probe array can change.
_IMPA_IDENTIFIER_PREFIX = "impa:"


#: The array lands in whichever magnetics node matches its mounting.
_IMPA_PROBE_NODES = ("magnetics.b_field_tor_probe", "magnetics.b_field_pol_probe")


def _impa_probe_node(ods: Any) -> str:
    """Return the magnetics node holding IMPA channels for this ODS."""
    for node in _IMPA_PROBE_NODES:
        for index in range(_count(ods, node)):
            identifier = _get(ods, f"{node}.{index}.identifier")
            if identifier is not None and str(identifier).startswith(_IMPA_IDENTIFIER_PREFIX):
                return node
    return _IMPA_PROBE_NODES[-1]


def _impa_probe_indices(ods: Any, node: str | None = None) -> list[int]:
    """Return the probe indices holding IMPA channels, in array order."""
    node = node or _impa_probe_node(ods)
    indices = []
    for index in range(_count(ods, node)):
        identifier = _get(ods, f"{node}.{index}.identifier")
        if identifier is not None and str(identifier).startswith(_IMPA_IDENTIFIER_PREFIX):
            indices.append(index)
    return indices


def _build_impa_lines(ods: Any, *, quantity: str = "field", **_: Any) -> LineSeries:
    """Build per-channel IMPA traces for the compensated field or raw voltage."""
    node = _impa_probe_node(ods)
    toroidal = node.endswith("b_field_tor_probe")
    field_label = "Bt" if toroidal else "Bz"
    label, unit = (field_label, "T") if quantity == "field" else ("Probe voltage", "V")
    series = []
    for index in _impa_probe_indices(ods, node):
        prefix = f"{node}.{index}"
        values = _array(ods, f"{prefix}.{quantity}.data")
        if values is None:
            continue
        time = _first_time(ods, (f"{prefix}.{quantity}.time", "magnetics.time"))
        if time is None or time.size != values.size:
            time = np.arange(values.size, dtype=float)
        name = _get(ods, f"{prefix}.name") or f"IMPA {index}"
        series.append(Series(x=time, y=values, label=str(name)))
    return LineSeries(
        series=tuple(series),
        x_label="Time",
        x_unit="s",
        y_label=label,
        y_unit=unit,
        title=f"IMPA calibrated {field_label}" if quantity == "field" else "IMPA raw voltage",
    )


def _build_impa_tf_profile(ods: Any, *, time: float | None = None, **_: Any) -> Profile1D:
    """IMPA field against probe radius, next to the 1/R toroidal-field model.

    Channel order, polarity and a mis-fitted radial position all show up as a
    departure from the smooth 1/R curve.
    """
    # The compensated field is preferred, but a shot whose calibration was
    # rejected has none -- and that is exactly when the shape check matters
    # most, so fall back to the raw voltage.
    node = _impa_probe_node(ods)
    quantity = "field"
    if not any(
        _array(ods, f"{node}.{index}.field.data") is not None
        for index in _impa_probe_indices(ods, node)
    ):
        quantity = "voltage"

    radii, values = [], []
    axis = None
    for index in _impa_probe_indices(ods, node):
        prefix = f"{node}.{index}"
        radius = _get(ods, f"{prefix}.position.r")
        trace = _array(ods, f"{prefix}.{quantity}.data")
        if radius is None or trace is None:
            continue
        if axis is None:
            axis = _first_time(ods, (f"{prefix}.{quantity}.time", "magnetics.time"))
        sample = 0 if axis is None or time is None else int(np.argmin(np.abs(axis - float(time))))
        if quantity == "voltage":
            # Hall probes sit on a large zero-field offset; remove it with the
            # same first-sample convention the processing uses so the shape
            # comparison is fair.
            trace = trace - trace[0]
        radii.append(float(radius))
        values.append(float(trace[min(sample, trace.size - 1)]))

    series = []
    if radii:
        order = np.argsort(radii)
        r_array = np.asarray(radii)[order]
        b_array = np.asarray(values)[order]
        series.append(
            Series(
                x=r_array,
                y=b_array,
                label="IMPA measurement",
                style={"marker": "o", "linestyle": "none"},
            )
        )
        model_r = np.linspace(max(r_array.min() * 0.8, 1e-3), r_array.max() * 1.2, 201)
        tf_current = _array(ods, "tf.coil.0.current.data")
        if quantity == "field" and tf_current is not None:
            tf_time = _first_time(ods, ("tf.time",))
            sample = 0
            if tf_time is not None and time is not None:
                sample = int(np.argmin(np.abs(tf_time - float(time))))
            current = float(tf_current[min(sample, tf_current.size - 1)])
            series.append(
                Series(
                    x=model_r,
                    y=4.0e-7 * np.pi * 24 * current / (2.0 * np.pi * model_r),
                    label="mu0 N I / 2 pi R",
                )
            )
        elif np.all(np.isfinite(b_array)) and b_array[-1] != 0.0:
            # Raw volts cannot carry the absolute model, so compare the shape
            # only, anchored on the outermost channel.
            series.append(
                Series(
                    x=model_r,
                    y=b_array[-1] * r_array[-1] / model_r,
                    label="1/R shape (scaled)",
                    style={"linestyle": "--"},
                )
            )

    label, unit = (
        ("Bt" if node.endswith("b_field_tor_probe") else "Bz", "T")
        if quantity == "field"
        else ("Probe voltage", "V")
    )
    return Profile1D(
        series=tuple(series),
        coordinate_label="R [m]",
        y_label=label,
        y_unit=unit,
        title="IMPA radial profile against the toroidal-field model",
    )


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
                label=_channel_label(
                    ods, "soft_x_rays.channel.{i}.name", index, f"ch{index}"
                ),
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
            GeometryLayer(
                r=x, z=y, kind="polyline", label=label, style={"color": "#e41a1c"}
            )
        )
    return GeometryLayers(
        layers=tuple(layers),
        x_label="x [m]",
        y_label="y [m]",
        title="Plasma Top View",
    )


def _pellet_positions(ods: Any, time_slice: int) -> list[tuple[float, float]]:
    """Return ``(r, phi)`` for each pellet path's first point at ``time_slice``."""
    positions = []
    pellet_count = _count(ods, f"pellets.time_slice.{time_slice}.pellet")
    for index in range(pellet_count):
        base = (
            f"pellets.time_slice.{time_slice}.pellet.{index}.path_geometry.first_point"
        )
        radius = _get(ods, f"{base}.r")
        if radius is None:
            continue
        phi = _get(ods, f"{base}.phi", 0.0)
        positions.append((float(radius), float(phi or 0.0)))
    return positions


def _build_machine_topview(
    ods: Any, *, time_slice: int = 0, **_: Any
) -> GeometryLayers:
    """Compose the plasma extent with launcher, antenna and pellet geometry."""
    layers: list[GeometryLayer] = []
    if "equilibrium" in ods:
        try:
            layers.extend(_build_equilibrium_topview(ods, time_slice=time_slice).layers)
        except ValueError:
            pass
    for container, r_path, label, style in (
        (
            "lh_antennas.antenna",
            "lh_antennas.antenna.{i}.position.r",
            "LH antenna",
            {"marker": "s"},
        ),
        (
            "ec_launchers.beam",
            "ec_launchers.beam.{i}.launching_position.r",
            "EC launcher",
            {"marker": "^"},
        ),
    ):
        for index in range(_count(ods, container)):
            radius = _get(ods, r_path.format(i=index))
            phi = _get(ods, r_path.format(i=index).replace(".r", ".phi"), 0.0)
            if radius is None:
                continue
            radius, phi = float(radius), float(phi or 0.0)
            layers.append(
                GeometryLayer(
                    r=[radius * np.cos(phi)],
                    z=[radius * np.sin(phi)],
                    kind="points",
                    label=f"{label} {index}",
                    style=style,
                )
            )
    for index, (radius, phi) in enumerate(_pellet_positions(ods, time_slice)):
        layers.append(
            GeometryLayer(
                r=[radius * np.cos(phi)],
                z=[radius * np.sin(phi)],
                kind="points",
                label=f"Pellet {index}",
                style={"marker": "*", "color": "#984ea3"},
            )
        )
    if not layers:
        raise ValueError(
            "none of the top-view IDS (equilibrium, lh_antennas, ec_launchers, "
            "pellets) are present"
        )
    return GeometryLayers(
        layers=tuple(layers),
        x_label="x [m]",
        y_label="y [m]",
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
        r=r_axis,
        z=z_axis,
        values=values,
        value_label="Vacuum Poloidal Flux [Wb]",
        filled=False,
        contour_levels=50,
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
    psi_axis = _get(
        ods, f"equilibrium.time_slice.{time_slice}.global_quantities.psi_axis"
    )
    psi_boundary = _get(
        ods, f"equilibrium.time_slice.{time_slice}.global_quantities.psi_boundary"
    )
    profile = _array(
        ods, f"core_profiles.profiles_1d.{time_slice}.electrons.{quantity}"
    )
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
    labels = {
        "temperature": "Electron Temperature [eV]",
        "density": "Electron Density [m^-3]",
    }
    return Field2D(
        r=grid_r,
        z=grid_z,
        values=values,
        value_label=labels[quantity],
        overlays=tuple(_wall_layers(ods)),
        title=labels[quantity],
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


def _nearest_time_index(
    reference: np.ndarray, target: float, tolerance: float
) -> int | None:
    if reference.size == 0:
        return None
    index = int(np.argmin(np.abs(reference - target)))
    return index if np.abs(reference[index] - target) <= tolerance else None


def _build_power_balance(ods: Any, **_: Any) -> Panels:
    """The five-panel power-balance figure computed by ``compute_power_balance``.

    Mirrors ``vaft.plot.time.time_power_balance``: dW_th/dt, dW_mag,p/dt,
    input/ohmic power, loss decomposition, and radiation decomposition -- not
    just the inputs to that computation.
    """
    from vaft.omas.formula_wrapper import (
        compute_bremsstrahlung_power,
        compute_power_balance,
    )

    try:
        power_balance = compute_power_balance(ods)
    except Exception as exc:
        raise ValueError(f"failed to compute power balance: {exc}") from exc

    t = np.asarray(power_balance["time"], dtype=float)
    if t.size == 0:
        raise ValueError("compute_power_balance returned no time points")

    dW_thdt = np.asarray(power_balance.get("dWdt", np.zeros_like(t)), dtype=float)
    V_ind = np.asarray(power_balance.get("V_ind", np.full_like(t, np.nan)), dtype=float)

    eq_count = _count(ods, "equilibrium.time_slice")
    eq_times = np.asarray(
        [
            float(_get(ods, f"equilibrium.time_slice.{i}.time", i))
            for i in range(eq_count)
        ],
        dtype=float,
    )
    eq_ip = np.asarray(
        [
            float(_get(ods, f"equilibrium.time_slice.{i}.global_quantities.ip", np.nan))
            for i in range(eq_count)
        ],
        dtype=float,
    )

    # Robust time-matching tolerance for equilibrium/core_profiles links.
    tolerance = 1e-4
    if t.size > 1:
        diffs = np.diff(np.sort(t))
        finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if finite.size:
            tolerance = max(tolerance, 0.1 * float(np.min(finite)))

    ip_matched = np.full_like(t, np.nan, dtype=float)
    for i, tt in enumerate(t):
        index = _nearest_time_index(eq_times, tt, tolerance)
        if index is not None:
            ip_matched[i] = eq_ip[index]
    dW_magdt = V_ind * ip_matched

    P_in = np.asarray(power_balance.get("P_heat", np.zeros_like(t)), dtype=float)
    P_ohm = np.asarray(
        power_balance.get(
            "P_ohm_diss", power_balance.get("P_ohm_flux", np.zeros_like(t))
        ),
        dtype=float,
    )

    P_rad = np.asarray(power_balance.get("P_rad", np.zeros_like(t)), dtype=float)
    P_trans = np.asarray(
        power_balance.get("P_trans", power_balance.get("P_loss", np.zeros_like(t))),
        dtype=float,
    )
    P_loss = np.asarray(power_balance.get("P_loss_total", P_rad + P_trans), dtype=float)

    P_line = np.asarray(
        power_balance.get("P_rad_line", np.zeros_like(P_rad)), dtype=float
    )
    P_Br = np.asarray(
        power_balance.get("P_Br", np.full_like(P_rad, np.nan)), dtype=float
    )
    if np.all(~np.isfinite(P_Br)):
        P_Br = np.full_like(P_rad, np.nan, dtype=float)
        cp_count = _count(ods, "core_profiles.profiles_1d")
        if cp_count:
            cp_times = np.asarray(
                [
                    float(_get(ods, f"core_profiles.profiles_1d.{i}.time", i))
                    for i in range(cp_count)
                ],
                dtype=float,
            )
            for i, tt in enumerate(t):
                index = _nearest_time_index(cp_times, tt, tolerance)
                if index is None:
                    continue
                try:
                    _, p_br_electron = compute_bremsstrahlung_power(
                        ods, time_slice=int(index)
                    )
                    P_Br[i] = float(p_br_electron)
                except Exception:
                    # Keep the figure robust when a few slices fail.
                    P_Br[i] = np.nan

    P_sync = np.asarray(
        power_balance.get("P_sync", np.full_like(P_rad, np.nan)), dtype=float
    )
    if np.all(~np.isfinite(P_sync)):
        # No direct synchrotron model in this pipeline yet; use the residual.
        P_sync = P_rad - P_line - np.nan_to_num(P_Br, nan=0.0)

    def _panel(traces: list[Series], title: str = "") -> LineSeries:
        return LineSeries(
            series=tuple(traces),
            x_label="Time",
            x_unit="s",
            y_label="",
            y_unit="W",
            title=title,
        )

    panels = (
        _panel([Series(x=t, y=dW_thdt, label="dW_th/dt")]),
        _panel([Series(x=t, y=dW_magdt, label="dW_mag,p/dt")]),
        _panel(
            [Series(x=t, y=P_in, label="P_in"), Series(x=t, y=P_ohm, label="P_ohm")]
        ),
        _panel(
            [
                Series(x=t, y=P_loss, label="P_loss"),
                Series(x=t, y=P_trans, label="P_trans"),
                Series(x=t, y=P_rad, label="P_rad"),
            ]
        ),
        _panel(
            [
                Series(x=t, y=P_rad, label="P_rad"),
                Series(x=t, y=P_Br, label="P_Br"),
                Series(x=t, y=P_sync, label="P_sync"),
                Series(x=t, y=P_line, label="P_line"),
            ]
        ),
    )
    return Panels(models=panels, ncols=1, share_x=True, suptitle="Power Balance")


RECIPES["summary_time_power_balance"] = CallableRecipe(
    builder=_build_power_balance,
    description="Power-balance terms computed by vaft.omas.compute_power_balance.",
)


# ---------------------------------------------------------------------------
# camera_visible: raster frames and their pinhole-projected EFIT/field-line
# overlays. This needs real computation (frame resolution, projection), not a
# path read, so it uses CallableRecipe -- the extraction logic itself lives in
# vaft.omas.process_wrapper (already covered by its own tests); these builders
# only repackage its dict outputs into Image2D/GeometryLayer view models.
# ---------------------------------------------------------------------------


def _camera_visible_frame_prefix(channel: int, detector: int) -> str:
    return f"camera_visible.channel.{channel}.detector.{detector}.frame"


def _camera_visible_channel_name(ods: Any, channel: int) -> str:
    return str(
        _get(ods, f"camera_visible.channel.{channel}.name", f"Camera Ch {channel}")
    )


def _resolve_camera_visible_frame(
    ods: Any, *, channel: int, detector: int, options: Mapping[str, Any]
):
    from vaft.omas.process_wrapper import _resolve_camera_frame

    return _resolve_camera_frame(
        ods,
        channel=channel,
        detector=detector,
        frame_index=options.get("frame_index"),
        frame_time=options.get("time"),
    )


def _camera_visible_frame_image(
    ods: Any, *, channel: int, detector: int, frame_index: int
) -> np.ndarray:
    path = f"{_camera_visible_frame_prefix(channel, detector)}.{frame_index}.image_raw"
    image = _array(ods, path)
    if image is None:
        raise ValueError(f"{path} is not available")
    return image


def _efit_overlay_layers(
    overlay: Mapping[str, Any], *, options: Mapping[str, Any]
) -> list[GeometryLayer]:
    layers: list[GeometryLayer] = []
    if options.get("show_wall", True) and overlay["wall_uv"].size:
        layers.append(
            GeometryLayer(
                r=overlay["wall_uv"][:, 0],
                z=overlay["wall_uv"][:, 1],
                kind="points",
                label="Wall",
                style={"marker": "o", "markersize": 1, "color": "yellow"},
            )
        )
    if options.get("show_lcfs", True) and overlay["lcfs_uv"].size:
        layers.append(
            GeometryLayer(
                r=overlay["lcfs_uv"][:, 0],
                z=overlay["lcfs_uv"][:, 1],
                kind="points",
                label="LCFS",
                style={"marker": "o", "markersize": 1.5, "color": "magenta"},
            )
        )
    if options.get("show_magnetic_axis", True) and overlay["magnetic_axis_uv"].size:
        layers.append(
            GeometryLayer(
                r=overlay["magnetic_axis_uv"][:, 0],
                z=overlay["magnetic_axis_uv"][:, 1],
                kind="points",
                label="Magnetic axis",
                style={"marker": "+", "markersize": 8, "color": "cyan"},
            )
        )
    flux_surfaces_uv = overlay.get("flux_surfaces_uv") or {}
    for index, level in enumerate(sorted(flux_surfaces_uv)):
        points = flux_surfaces_uv[level]
        if points.size == 0:
            continue
        layers.append(
            GeometryLayer(
                r=points[:, 0],
                z=points[:, 1],
                kind="points",
                label="psi surfaces" if index == 0 else "",
                style={"marker": "o", "markersize": 1, "color": "tab:cyan"},
            )
        )
    return layers


def _build_camera_visible_image_frame(ods: Any, **options: Any) -> Image2D:
    channel = int(options.get("channel", 0))
    detector = int(options.get("detector", 0))
    idx, resolved_time, _shape = _resolve_camera_visible_frame(
        ods, channel=channel, detector=detector, options=options
    )
    image = _camera_visible_frame_image(
        ods, channel=channel, detector=detector, frame_index=idx
    )
    channel_name = _camera_visible_channel_name(ods, channel)
    title = options.get("title", f"{channel_name} frame {idx} @ t={resolved_time:.4f}s")
    return Image2D(values=image, value_label="Digital levels", title=title)


def _build_camera_visible_image_efit_overlay(ods: Any, **options: Any) -> Image2D:
    from vaft.omas.process_wrapper import compute_camera_visible_efit_overlay

    shot = options["shot"]
    channel = int(options.get("channel", 0))
    detector = int(options.get("detector", 0))
    idx, resolved_time, _shape = _resolve_camera_visible_frame(
        ods, channel=channel, detector=detector, options=options
    )
    image = _camera_visible_frame_image(
        ods, channel=channel, detector=detector, frame_index=idx
    )

    overlay = compute_camera_visible_efit_overlay(
        ods,
        shot,
        channel=channel,
        detector=detector,
        frame_index=idx,
        flux_surface_levels=tuple(
            options.get("flux_surface_levels", (0.25, 0.5, 0.75, 0.95))
        ),
    )
    layers = _efit_overlay_layers(overlay, options=options)

    channel_name = _camera_visible_channel_name(ods, channel)
    title = options.get(
        "title",
        f"{channel_name} frame {idx} @ t={resolved_time:.4f}s -- shot {shot} EFIT overlay",
    )
    return Image2D(
        values=image, value_label="Digital levels", title=title, overlays=tuple(layers)
    )


def _build_camera_visible_image_field_line(ods: Any, **options: Any) -> Image2D:
    from vaft.omas.process_wrapper import (
        compute_camera_visible_efit_overlay,
        compute_camera_visible_field_line_overlay,
    )

    shot = options["shot"]
    r0 = float(options["r0"])
    z0 = float(options["z0"])
    channel = int(options.get("channel", 0))
    detector = int(options.get("detector", 0))
    idx, resolved_time, _shape = _resolve_camera_visible_frame(
        ods, channel=channel, detector=detector, options=options
    )
    image = _camera_visible_frame_image(
        ods, channel=channel, detector=detector, frame_index=idx
    )

    result = compute_camera_visible_field_line_overlay(
        ods,
        shot,
        r0=r0,
        z0=z0,
        phi0=float(options.get("phi0", 0.0)),
        channel=channel,
        detector=detector,
        frame_index=idx,
        dphi_deg=float(options.get("dphi_deg", 1.0)),
        max_length_m=float(options.get("max_length_m", 50.0)),
        direction=options.get("direction", "forward"),
        use_wall_boundary=options.get("use_wall_boundary", True),
    )

    layers: list[GeometryLayer] = []
    field_line_uv = result["field_line_uv"]
    if field_line_uv.shape[0] >= 2:
        layers.append(
            GeometryLayer(
                r=field_line_uv[:, 0],
                z=field_line_uv[:, 1],
                kind="polyline",
                label="Field line",
                style={"color": "red", "linewidth": 1.5},
            )
        )
        layers.append(
            GeometryLayer(
                r=field_line_uv[:1, 0],
                z=field_line_uv[:1, 1],
                kind="points",
                label="Start",
                style={"marker": "o", "markersize": 8, "color": "lime"},
            )
        )
        layers.append(
            GeometryLayer(
                r=field_line_uv[-1:, 0],
                z=field_line_uv[-1:, 1],
                kind="points",
                label="End",
                style={"marker": "o", "markersize": 8, "color": "blue"},
            )
        )
    elif field_line_uv.shape[0] == 1:
        layers.append(
            GeometryLayer(
                r=field_line_uv[:1, 0],
                z=field_line_uv[:1, 1],
                kind="points",
                label="Start",
                style={"marker": "o", "markersize": 8, "color": "lime"},
            )
        )

    if (
        options.get("show_wall")
        or options.get("show_lcfs")
        or options.get("show_magnetic_axis")
        or options.get("flux_surface_levels")
    ):
        efit_overlay = compute_camera_visible_efit_overlay(
            ods,
            shot,
            channel=channel,
            detector=detector,
            frame_index=idx,
            flux_surface_levels=tuple(options.get("flux_surface_levels", ())),
        )
        layers.extend(_efit_overlay_layers(efit_overlay, options=options))

    reason = result["trace"]["termination_reason"]
    channel_name = _camera_visible_channel_name(ods, channel)
    title = options.get(
        "title",
        f"{channel_name} frame {idx} @ t={resolved_time:.4f}s -- shot {shot} field line\n"
        f"R0={r0:.3f}m, Z0={z0:.3f}m, stop: {reason}",
    )
    return Image2D(
        values=image, value_label="Digital levels", title=title, overlays=tuple(layers)
    )


def _build_camera_visible_animation_frames(ods: Any, **options: Any) -> ImageSequence:
    channel = int(options.get("channel", 0))
    detector = int(options.get("detector", 0))
    prefix = _camera_visible_frame_prefix(channel, detector)
    n_frames = _count(ods, prefix)
    indices = options.get("frame_indices")
    if indices is None:
        indices = range(n_frames)

    frames = []
    times = []
    for index in indices:
        image = _array(ods, f"{prefix}.{index}.image_raw")
        if image is None:
            continue
        frames.append(image)
        times.append(float(_get(ods, f"{prefix}.{index}.time", 0.0)))
    if not frames:
        raise ValueError("No camera_visible frames available to animate.")

    return ImageSequence(
        frames=tuple(frames),
        time=np.asarray(times, dtype=float),
        value_label="Digital levels",
    )


RECIPES["camera_visible_image_frame"] = CallableRecipe(
    builder=_build_camera_visible_image_frame,
    description="One FAST-camera frame, selected by frame_index or nearest time.",
)
RECIPES["camera_visible_image_efit_overlay"] = CallableRecipe(
    builder=_build_camera_visible_image_efit_overlay,
    description="FAST-camera frame with the projected EFIT/wall overlay (requires shot=).",
)
RECIPES["camera_visible_image_field_line"] = CallableRecipe(
    builder=_build_camera_visible_image_field_line,
    description="FAST-camera frame with a projected traced field line (requires shot=, r0=, z0=).",
)
RECIPES["camera_visible_animation_frames"] = CallableRecipe(
    builder=_build_camera_visible_animation_frames,
    description="Animate a sequence of FAST-camera frames on a shared color scale.",
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


def _divisor(ods: Any, path: str) -> float:
    """Read a scalar ODS value for dividing a trace; 1.0 when absent or zero."""
    if not path:
        return 1.0
    value = _get(ods, path)
    if value is None:
        return 1.0
    value = float(np.asarray(value, dtype=float).ravel()[0])
    return value if value != 0.0 else 1.0


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
        return [
            Series(
                x=_scaled(time, xunit, _TIME_SCALES),
                y=y * value_scale,
                label=entry_label,
            )
        ]

    if recipe.index == "channel":
        indices = _resolve_indices(ods, recipe.y_path, channels)
        traces = []
        for index in indices:
            try:
                y = _first_array(
                    ods, (recipe.y_path, *recipe.fallback_y_paths), i=index
                )
            except ValueError:
                # e.g. a channel with several real energy bands: skip it so one
                # odd channel does not abort a whole multi-channel figure.
                continue
            # IMAS arrays may include placeholder scalar values (commonly
            # ``nan``) for unpopulated channels.  They are data values, but
            # not time series, and ``Series`` correctly refuses to render
            # them.  Ignore those placeholders while retaining the valid
            # channels in the same IDS.
            if y is None or y.ndim != 1:
                continue
            time = _first_time(ods, recipe.x_paths, i=index)
            if time is None or time.ndim != 1 or time.size != y.size:
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
    y = y / _divisor(ods, recipe.divide_by_path)
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
                ods,
                recipe,
                entry_label=entry_label,
                channels=options.get("channels"),
                xunit=xunit,
                yunit=yunit,
            )
        )
    return LineSeries(
        series=tuple(traces),
        x_label=recipe.x_label,
        x_unit=xunit,
        y_label=recipe.y_label,
        y_unit=yunit,
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
        x = (
            _array(ods, coordinate_path.format(i=time_slice))
            if coordinate_path
            else None
        )
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
                    GeometryLayer(
                        r=r_values,
                        z=z_values,
                        kind="points",
                        label=recipe.title,
                        style=style,
                    )
                )
            continue
        for index in indices:
            r = _array(ods, r_template.format(i=index))
            z = _array(ods, z_template.format(i=index))
            if r is None or z is None or r.size != z.size:
                continue
            layers.append(
                GeometryLayer(
                    r=r,
                    z=z,
                    kind=kind,
                    label=_channel_label(ods, label_template, index, "")
                    if label_template
                    else "",
                    style=style,
                )
            )
    return GeometryLayers(
        layers=tuple(layers),
        x_label=recipe.x_label,
        y_label=recipe.y_label,
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
    if recipe.values_order == "rz":
        values = values.T
    elif values.shape != (z.size, r.size):
        values = values.T
    overlays = list(_wall_layers(ods))
    if recipe.boundary_paths:
        boundary_r = _array(ods, recipe.boundary_paths[0].format(i=time_slice))
        boundary_z = _array(ods, recipe.boundary_paths[1].format(i=time_slice))
        if boundary_r is not None and boundary_z is not None:
            overlays.append(
                GeometryLayer(
                    r=boundary_r,
                    z=boundary_z,
                    kind="polygon",
                    label="Boundary",
                    style={"color": "#e41a1c"},
                )
            )
    return Field2D(
        r=r,
        z=z,
        values=values,
        value_label=recipe.value_label,
        contour_levels=options.get("contour_levels"),
        overlays=tuple(overlays),
        title=options.get("title", recipe.title),
    )


def _first_channel_with_signal(
    ods: Any, recipe: "SpectrogramRecipe | PowerSpectrumRecipe"
) -> int:
    """The lowest channel index whose signal is actually present.

    Consults the recipe's fallback signal spellings too, so a source that
    stores e.g. ``power`` instead of ``brightness`` still auto-picks a channel.
    Falls back to ``0`` so the caller raises the usual "not available" error
    when no channel carries the signal at all.
    """
    total = _count(ods, recipe.container) if recipe.container else 0
    templates = (recipe.signal_path, *recipe.fallback_signal_paths)
    for index in range(total):
        if any(_array(ods, template.format(i=index)) is not None for template in templates):
            return index
    return 0


def _build_spectrogram(
    ods: Any, recipe: SpectrogramRecipe, **options: Any
) -> Spectrogram:
    from vaft.process import mirnov_spectrogram as compute_spectrogram

    requested = options.get("channel")
    if requested is None:
        # Availability accepts any channel that carries the signal, so the
        # default must be the first one that does.  Real arrays routinely
        # declare geometry for channels whose waveform was never acquired.
        index = _first_channel_with_signal(ods, recipe)
    else:
        index = int(requested)
    candidates = (recipe.signal_path, *recipe.fallback_signal_paths)
    signal = _first_array(ods, candidates, i=index)
    if signal is None:
        raise ValueError(f"{recipe.signal_path.format(i=index)} is not available")
    time = _first_time(ods, recipe.time_paths, i=index)
    if time is None or time.size != signal.size:
        time = np.arange(signal.size, dtype=float)

    time_range = options.get("time_range")
    if time_range is not None:
        window = (time >= float(time_range[0])) & (time <= float(time_range[1]))
        time, signal = time[window], signal[window]

    sample_rate = options.get("sample_rate")
    if sample_rate is None:
        steps = np.diff(time)
        positive = steps[steps > 0]
        sample_rate = 1.0 / float(np.median(positive)) if positive.size else 1.0
    result = compute_spectrogram(
        time,
        signal,
        sample_rate=float(sample_rate),
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


def _build_power_spectrum(
    ods: Any, recipe: PowerSpectrumRecipe, **options: Any
) -> PowerSpectrum:
    """Build a PSD view model from one channel of an ODS.

    Reference slopes and frequency markers are passed through from the caller
    untouched; this adapter never supplies one of its own.  The signal is
    analysed as stored: for a Mirnov voltage that means the PSD of ``dB/dt``, not
    of ``B`` -- integrate first if you want a field spectrum.
    """
    from vaft.process.fluctuation import compute_psd, fit_power_law_spectrum

    requested = options.get("channel")
    if requested is None:
        index = _first_channel_with_signal(ods, recipe)
    else:
        index = int(requested)
    candidates = (recipe.signal_path, *recipe.fallback_signal_paths)
    signal = _first_array(ods, candidates, i=index)
    if signal is None:
        raise ValueError(f"{recipe.signal_path.format(i=index)} is not available")
    time = _first_time(ods, recipe.time_paths, i=index)
    if time is None or time.size != signal.size:
        raise ValueError(
            f"a matching time axis for {recipe.signal_path.format(i=index)} is not "
            "available; a PSD needs a real timebase to set its frequency axis"
        )

    time_range = options.get("time_range")
    if time_range is not None:
        window = (time >= float(time_range[0])) & (time <= float(time_range[1]))
        time, signal = time[window], signal[window]

    spectrum = compute_psd(
        time,
        signal,
        sample_rate=options.get("sample_rate"),
        window=options.get("window", "hann"),
        nperseg=options.get("nperseg"),
        noverlap=options.get("noverlap"),
        detrend=options.get("detrend", "constant"),
    )

    fits = []
    for f_range in options.get("fit_ranges", ()):
        fit = fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=f_range)
        edges = np.array(fit.frequency_range, dtype=float)
        fits.append(
            Series(
                x=edges,
                y=10.0**fit.intercept * edges**fit.alpha,
                label=f"fit {fit.alpha:.2f} (R^2={fit.r_squared:.3f})",
                style={"linestyle": "-", "linewidth": 1.5},
            )
        )

    slopes = tuple(
        item if isinstance(item, ReferenceSlope) else ReferenceSlope(slope=float(item))
        for item in options.get("reference_slopes", ())
    )

    return PowerSpectrum(
        frequency=spectrum.frequency,
        psd=spectrum.psd,
        fits=tuple(fits),
        reference_slopes=slopes,
        marker_frequencies=tuple(options.get("marker_frequencies", ())),
        # ``label`` is taken by render() for entry naming, so the trace label
        # has its own option; it matters when several channels share one axes.
        label=str(options.get("series_label", "")),
        y_label=recipe.value_label,
        title=options.get(
            "title", _channel_label(ods, recipe.label_path, index, f"channel {index}")
        ),
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
            "none of the panels "
            + ", ".join(recipe.members)
            + " have data in this input"
        )
    return Panels(
        models=tuple(members),
        ncols=recipe.ncols,
        share_x=recipe.share_x,
        suptitle=options.get("title", recipe.suptitle),
    )


def _build_limiter_shunt_currents(ods: Any, **options: Any) -> Panels:
    """Build one current panel per VEST limiter monitor.

    Limiter shunts intentionally store only voltage in the ODS.  The current is
    therefore derived at plot time from the documented effective V/I
    ``resistance`` coefficient, leaving the IMAS tree free of a non-standard
    ``magnetics.shunt[].current`` path.
    """
    from vaft.machine_mapping.magnetics import LIMITER_SHUNT_CHANNELS

    xunit = str(options.get("xunit", "s"))
    panels: list[LineSeries] = []
    have_signal = False
    for index, channel in enumerate(LIMITER_SHUNT_CHANNELS):
        base = f"magnetics.shunt.{index}"
        name = _get(ods, f"{base}.name") or channel["name"]
        voltage = _array(ods, f"{base}.voltage.data")
        time = _first_time(ods, (f"{base}.voltage.time", "magnetics.time"))
        resistance = _get(ods, f"{base}.resistance")
        traces: tuple[Series, ...] = ()
        try:
            coefficient = float(np.asarray(resistance, dtype=float).ravel()[0])
        except (IndexError, TypeError, ValueError):
            coefficient = 0.0
        if (
            voltage is not None
            and time is not None
            and time.size == voltage.size
            and np.isfinite(coefficient)
            and coefficient != 0.0
        ):
            traces = (
                Series(x=_scaled(time, xunit, _TIME_SCALES), y=voltage / coefficient),
            )
            have_signal = True
        panels.append(
            LineSeries(
                series=traces,
                x_label="Time",
                x_unit=xunit,
                y_label="Limiter Current",
                y_unit="A",
                title=str(name),
                x_limits=options.get("x_limits"),
            )
        )
    if not have_signal:
        raise ValueError(
            "no limiter-shunt voltage data with a valid resistance is available"
        )
    return Panels(
        models=tuple(panels),
        share_x=True,
        suptitle=options.get("title", "Limiter Currents"),
    )


RECIPES["magnetics_time_limiter_current"] = CallableRecipe(
    builder=_build_limiter_shunt_currents,
    description="VEST limiter currents derived from shunt voltage / resistance.",
)


_VERIFICATION_FAMILIES = (
    ("bpol_probe", "Poloidal probes", "mT", 1e3, True),
    ("flux_loop", "Flux loops", "mWb", 1e3, True),
    ("pf_current", "PF currents", "kA", 1e-3, True),
)


# Constraint extraction lives with the metrics that consume it
# (:mod:`vaft.omas.efit_quality`), so the recipes and the goodness-of-fit
# numbers can never disagree about what a channel's state or residual is.
# Re-exported here under the names this module has always used.
from .efit_quality import (  # noqa: E402
    CONSTRAINT_STATES,
    ConstraintTable,
    constraint_state as _constraint_state,
    constraint_table as _constraint_table,
    fitted_mask as _fitted_mask,
    slice_times as _slice_times,
)


def _scalar(value: Any, scale: float = 1.0) -> float:
    try:
        return float(np.asarray(value)) * scale
    except (TypeError, ValueError):
        return float("nan")


def _verification_constraint_panel(
    ods: Any,
    *,
    time_slice: int,
    family: str,
    title: str,
    unit: str,
    scale: float,
    is_array: bool,
    show_uncertainty: bool = False,
) -> LineSeries:
    table = _constraint_table(
        ods, time_slice=time_slice, family=family, is_array=is_array, scale=scale
    )
    # This panel compares the two sides of the fit, so a channel without both a
    # finite measured and a finite reconstructed value has nothing to compare.
    keep = np.isfinite(table.measured) & np.isfinite(table.reconstructed)
    x = table.index[keep]
    measured_array = table.measured[keep]
    reconstructed_array = table.reconstructed[keep]
    uncertainty_array = table.uncertainty[keep]

    # Normalized by the RMS of the measurement, not by a residual baseline, so
    # this is a percentage of signal amplitude rather than a skill score.
    denominator = rms(measured_array)
    relative_error = (
        100.0 * rms(reconstructed_array - measured_array) / denominator
        if np.isfinite(denominator) and denominator != 0.0
        else np.nan
    )
    finite_weights = table.weight[keep]
    finite_weights = finite_weights[np.isfinite(finite_weights)]
    weight_text = f", W={np.mean(finite_weights):.2e}" if finite_weights.size else ""
    error_text = f"{relative_error:.2f}%" if np.isfinite(relative_error) else "n/a"
    measured_yerr = (
        uncertainty_array
        if show_uncertainty
        and uncertainty_array.size
        and np.all(np.isfinite(uncertainty_array))
        else None
    )
    return LineSeries(
        series=(
            Series(
                x=x,
                y=measured_array,
                yerr=measured_yerr,
                label="Measured",
                style={"color": "black", "marker": "o", "linestyle": "none"},
            ),
            Series(
                x=x,
                y=reconstructed_array,
                label="Reconstructed",
                style={"color": "red", "marker": "o", "linestyle": "none"},
            ),
        ),
        x_label="Constraint index",
        y_label=title,
        y_unit=unit,
        title=f"{title}: relative RMS error {error_text}{weight_text}",
    )


def _build_equilibrium_verification(ods: Any, **options: Any) -> Panels:
    time_slice = int(options.get("time_slice", 0))
    panels = [
        _verification_constraint_panel(
            ods,
            time_slice=time_slice,
            family=family,
            title=title,
            unit=unit,
            scale=scale,
            is_array=is_array,
            show_uncertainty=bool(options.get("show_uncertainty", False)),
        )
        for family, title, unit, scale, is_array in _VERIFICATION_FAMILIES
    ]
    field_recipe = RECIPES["equilibrium_field_psi"]
    raw_field = _build_field_2d(
        ods,
        field_recipe,
        time_slice=time_slice,
        contour_levels=options.get("contour_levels", np.linspace(0.0, 1.0, 16)),
    )
    root = f"equilibrium.time_slice.{time_slice}"
    psi_axis = float(_get(ods, f"{root}.global_quantities.psi_axis", np.nan))
    psi_boundary = float(
        _get(ods, f"{root}.global_quantities.psi_boundary", np.nan)
    )
    delta_psi = psi_boundary - psi_axis
    if not np.isfinite(delta_psi) or delta_psi == 0.0:
        raise ValueError("valid psi_axis and psi_boundary are required")
    psi_norm = (raw_field.values - psi_axis) / delta_psi
    # Filled psi_N is meaningful inside the last closed flux surface.  A value
    # threshold alone also selects disconnected coil-local contours, so mask
    # geometrically with EFIT's reconstructed LCFS.
    boundary_r = _array(ods, f"{root}.boundary.outline.r")
    boundary_z = _array(ods, f"{root}.boundary.outline.z")
    if boundary_r is not None and boundary_z is not None:
        from matplotlib.path import Path as MplPath

        rr, zz = np.meshgrid(raw_field.r, raw_field.z)
        inside = MplPath(np.column_stack((boundary_r, boundary_z))).contains_points(
            np.column_stack((rr.ravel(), zz.ravel()))
        ).reshape(rr.shape)
    else:
        inside = (psi_norm >= 0.0) & (psi_norm <= 1.0)
    psi_norm = np.where(
        inside & (psi_norm >= 0.0) & (psi_norm <= 1.0), psi_norm, np.nan
    )
    field = Field2D(
        r=raw_field.r,
        z=raw_field.z,
        values=psi_norm,
        value_label="Normalized Poloidal Flux",
        contour_levels=options.get("contour_levels", np.linspace(0.0, 1.0, 16)),
        overlays=raw_field.overlays,
        title="Normalized Poloidal Flux",
    )
    time_value = _get(ods, f"equilibrium.time_slice.{time_slice}.time")
    if time_value is None:
        times = _array(ods, "equilibrium.time")
        time_value = (
            times[time_slice]
            if times is not None and time_slice < times.size
            else np.nan
        )
    pulse = _get(ods, "dataset_description.data_entry.pulse", "")
    time_text = (
        f", t={float(time_value) * 1e3:.2f} ms" if np.isfinite(time_value) else ""
    )
    def scalar_summary(family: str, label: str, scale: float, unit: str) -> str:
        base = f"{root}.constraints.{family}"
        measured = float(_get(ods, f"{base}.measured", np.nan)) * scale
        reconstructed = float(_get(ods, f"{base}.reconstructed", np.nan)) * scale
        if not (np.isfinite(measured) and np.isfinite(reconstructed)):
            return f"{label}: unavailable"
        error = 100.0 * abs(reconstructed - measured) / abs(measured) if measured else np.nan
        return (
            f"{label}: measured {measured:.4g} {unit}, reconstructed "
            f"{reconstructed:.4g} {unit}, error {error:.2f}%"
        )

    scalar_text = "\n".join(
        (
            scalar_summary("ip", "Ip", 1e-3, "kA"),
            scalar_summary("diamagnetic_flux", "Diamagnetic flux", 1e3, "mWb"),
        )
    )
    title = options.get(
        "title", f"EFIT verification — shot {pulse}{time_text}\n{scalar_text}"
    )
    panels.append(field)
    return Panels(models=tuple(panels), ncols=2, share_x=False, suptitle=title)


RECIPES["equilibrium_overview_verification"] = CallableRecipe(
    builder=_build_equilibrium_verification,
    description="EFIT measured/reconstructed constraints and poloidal-flux map.",
)


# ---------------------------------------------------------------------------
# EFIT submitted constraints and reconstruction residuals (issue #139)
# ---------------------------------------------------------------------------

#: Marker style per channel state, shared by the submitted and residual views so
#: a dead channel looks the same in both.
_STATE_STYLE = {
    "enabled": {"color": "black", "marker": "o", "linestyle": "none"},
    "disabled": {"color": "tab:orange", "marker": "x", "linestyle": "none"},
    "missing": {"color": "tab:red", "marker": "s", "linestyle": "none",
                "markerfacecolor": "none"},
}


def _require_slices(ods: Any) -> int:
    count = _count(ods, "equilibrium.time_slice")
    if count == 0:
        raise ValueError(
            "equilibrium ODS carries no time slices; EFIT produced no accepted "
            "reconstruction for this shot"
        )
    return count


def _state_series(
    table: ConstraintTable,
    values: np.ndarray,
    errors: np.ndarray | None = None,
) -> list[Series]:
    """One trace per channel state, so the dead channels are visible, not absent.

    ``errors`` is masked through exactly the same channel selection as ``values``
    -- a channel dropped for a non-finite value must drop its error bar with it,
    or every later bar lands on the wrong channel.
    """
    series = []
    for state in CONSTRAINT_STATES:
        mask = table.mask(state)
        if not mask.any():
            continue
        y = values[mask]
        finite = np.isfinite(y)
        yerr = None
        if errors is not None and state == "enabled":
            selected = errors[mask][finite]
            if selected.size and np.all(np.isfinite(selected)):
                yerr = selected
        series.append(
            Series(
                x=table.index[mask][finite],
                y=y[finite],
                yerr=yerr,
                label=f"{state} ({int(mask.sum())})",
                style=dict(_STATE_STYLE[state]),
            )
        )
    return series


def _build_equilibrium_constraints(ods: Any, **options: Any) -> Panels:
    """What was submitted to EFIT, before anything is inferred from its answer."""
    _require_slices(ods)
    time_slice = int(options.get("time_slice", 0))
    times = _slice_times(ods)
    panels: list[Any] = []
    for family, title, unit, scale, is_array in _VERIFICATION_FAMILIES:
        table = _constraint_table(
            ods, time_slice=time_slice, family=family, is_array=is_array, scale=scale
        )
        series = _state_series(table, table.measured, errors=table.uncertainty)
        if not series:
            continue
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="Constraint index",
                y_label=title,
                y_unit=unit,
                title=f"{title} submitted ({table.count('enabled')}/{len(table.state)} fitted)",
            )
        )

    # The scalar constraints across every slice: also the "which time slices did
    # EFIT actually run" view, since the x positions are equilibrium.time.
    for family, title, unit, scale in (
        ("ip", "Plasma current", "kA", 1e-3),
        ("diamagnetic_flux", "Diamagnetic flux", "mWb", 1e3),
    ):
        values = np.array(
            [
                _scalar(
                    _get(ods, f"equilibrium.time_slice.{index}.constraints.{family}.measured"),
                    scale,
                )
                for index in range(times.size)
            ],
            dtype=float,
        )
        if not np.isfinite(values).any():
            continue
        panels.append(
            LineSeries(
                series=(
                    Series(x=times, y=values, label="submitted",
                           style={"marker": "o", "linestyle": "-"}),
                ),
                x_label="time",
                x_unit="s",
                y_label=title,
                y_unit=unit,
                title=f"{title} at the {times.size} selected slice(s)",
            )
        )
    return Panels(
        models=tuple(panels),
        ncols=2,
        share_x=False,
        suptitle=_efit_suptitle(ods, "EFIT submitted constraints", times, time_slice),
    )


def _efit_suptitle(ods: Any, headline: str, times: np.ndarray, time_slice: int | None) -> str:
    pulse = _get(ods, "dataset_description.data_entry.pulse", "")
    detail = ""
    if time_slice is not None and 0 <= time_slice < times.size:
        detail = f", slice {time_slice} at t={times[time_slice] * 1e3:.2f} ms"
    return f"{headline} — shot {pulse}{detail}"


def _build_equilibrium_constraint_coverage(ods: Any, **options: Any) -> Panels:
    """How the fitted channel set changes across the reconstructed slices."""
    count = _require_slices(ods)
    times = _slice_times(ods)
    panels: list[Any] = []
    for family, title, _unit, _scale, is_array in _VERIFICATION_FAMILIES:
        counts = {state: [] for state in CONSTRAINT_STATES}
        for index in range(count):
            table = _constraint_table(
                ods, time_slice=index, family=family, is_array=is_array
            )
            for state in CONSTRAINT_STATES:
                counts[state].append(table.count(state))
        series = tuple(
            Series(
                x=times,
                y=np.asarray(counts[state], dtype=float),
                label=state,
                style={"marker": ".", "color": _STATE_STYLE[state]["color"]},
            )
            for state in CONSTRAINT_STATES
            if any(counts[state])
        )
        if not series:
            continue
        panels.append(
            LineSeries(
                series=series,
                x_label="time",
                x_unit="s",
                y_label="channels",
                title=f"{title}: channel coverage",
            )
        )
    if not panels:
        raise ValueError("no constraint family carries channels to report coverage for")
    return Panels(
        models=tuple(panels),
        ncols=2,
        share_x=True,
        suptitle=_efit_suptitle(ods, "EFIT constraint coverage", times, None),
    )


def _build_equilibrium_residuals(ods: Any, **options: Any) -> Panels:
    """Reconstruction residuals by diagnostic family, beside convergence."""
    count = _require_slices(ods)
    time_slice = int(options.get("time_slice", 0))
    times = _slice_times(ods)
    panels: list[Any] = []
    rms_series: list[Series] = []
    exact_families: list[str] = []

    from .efit_quality import classify_fit_role

    for family, title, unit, scale, is_array in _VERIFICATION_FAMILIES:
        table = _constraint_table(
            ods, time_slice=time_slice, family=family, is_array=is_array, scale=scale
        )
        role = classify_fit_role(ods, table, time_slice=time_slice)
        series = _state_series(table, table.residual)
        if series:
            suffix = (
                " — prescribed, not fitted"
                if role == "prescribed"
                else ": measured − reconstructed"
            )
            panels.append(
                LineSeries(
                    series=tuple(series),
                    x_label="Constraint index",
                    y_label=f"{title} residual",
                    y_unit=unit,
                    title=f"{title}{suffix}",
                )
            )
        if role == "prescribed":
            exact_families.append(title)
            continue
        values = []
        for index in range(count):
            slice_table = _constraint_table(
                ods, time_slice=index, family=family, is_array=is_array, scale=scale
            )
            fitted = _fitted_mask(slice_table)
            values.append(rms(slice_table.residual[fitted]))
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size and np.all(finite == 0.0):
            # Defensive: a fitted family whose residual happens to vanish. A log
            # axis cannot show zero and a flat line says nothing, so name it.
            if title not in exact_families:
                exact_families.append(title)
        elif finite.size:
            rms_series.append(
                Series(x=times, y=array, label=f"{title} [{unit}]", style={"marker": "."})
            )

    if rms_series:
        exact_text = (
            f"; {', '.join(exact_families)} fitted exactly" if exact_families else ""
        )
        panels.append(
            LineSeries(
                series=tuple(rms_series),
                x_label="time",
                x_unit="s",
                y_label="fitted-channel residual RMS",
                log_y=True,
                title=f"Residual RMS by family, display units{exact_text}",
            )
        )

    # Convergence is context for the residuals, never a substitute for them.
    for path, title, unit in (
        ("convergence.grad_shafranov_deviation_value", "Grad-Shafranov deviation", ""),
        ("convergence.iterations_n", "Iterations", ""),
    ):
        values = np.array(
            [
                _scalar(_get(ods, f"equilibrium.time_slice.{index}.{path}"))
                for index in range(count)
            ],
            dtype=float,
        )
        if not np.isfinite(values).any():
            continue
        panels.append(
            LineSeries(
                series=(Series(x=times, y=values, label=title, style={"marker": "."}),),
                x_label="time",
                x_unit="s",
                y_label=title,
                y_unit=unit,
                title=title,
            )
        )
    if not panels:
        raise ValueError("equilibrium ODS carries no reconstructed constraints")
    return Panels(
        models=tuple(panels),
        ncols=2,
        share_x=False,
        suptitle=_efit_suptitle(ods, "EFIT reconstruction residuals", times, time_slice),
    )


def _build_equilibrium_fit_quality(ods: Any, **options: Any) -> Panels:
    """Is this fit statistically acceptable against EFIT's own uncertainties?"""
    from .efit_quality import FAMILIES, efit_quality_metrics

    count = _require_slices(ods)
    time_slice = int(options.get("time_slice", 0))
    times = _slice_times(ods)
    metrics = efit_quality_metrics(ods)
    fits = [entry["fit"] for entry in metrics["slices"]]
    panels: list[Any] = []

    # Reduced chi-square against the value a fit consistent with its assigned
    # uncertainties would produce.
    reduced = np.array([fit["chi_squared_reduced"] for fit in fits], dtype=float)
    if np.isfinite(reduced).any():
        panels.append(
            LineSeries(
                series=(
                    Series(x=times, y=reduced, label="reduced χ²", style={"marker": "."}),
                    Series(
                        x=times,
                        y=np.ones_like(times),
                        label="χ²/ν = 1",
                        style={"linestyle": "--", "color": "0.5", "lw": 1.0},
                    ),
                ),
                x_label="time",
                x_unit="s",
                y_label="χ² / ν",
                log_y=bool(np.nanmax(reduced) / max(np.nanmin(reduced[reduced > 0]), 1e-12) > 50)
                if np.any(reduced > 0)
                else False,
                title=(
                    f"Reduced χ² (ν={fits[time_slice]['degrees_of_freedom']:.0f}"
                    f" = {fits[time_slice]['degrees_of_freedom_inputs']['num_input_data']:.0f}"
                    f" − {fits[time_slice]['degrees_of_freedom_inputs']['num_fit_variables']:.0f}"
                    f" − {fits[time_slice]['degrees_of_freedom_inputs']['num_hard_constraints']:.0f})"
                ),
            )
        )

    # Which diagnostic actually determines the solution.
    share_names = sorted({name for fit in fits for name in fit["chi_squared_share"]})
    share_series = []
    for name in share_names:
        values = np.array(
            [fit["chi_squared_share"].get(name, np.nan) for fit in fits], dtype=float
        )
        if np.isfinite(values).any():
            share_series.append(Series(x=times, y=values, label=name, style={"marker": "."}))
    if share_series:
        dominant = max(
            fits[time_slice]["chi_squared_share"].items(),
            key=lambda item: item[1] if np.isfinite(item[1]) else -1,
            default=("", float("nan")),
        )
        panels.append(
            LineSeries(
                series=tuple(share_series),
                x_label="time",
                x_unit="s",
                y_label="share of total χ²",
                y_limits=(-0.05, 1.05),
                title=f"χ² share by family — {dominant[0]} dominates ({dominant[1]:.3f})",
            )
        )

    # Normalized residuals at the selected slice: the per-channel view, in units
    # of the uncertainty EFIT itself assigned.
    for family, title, _unit, _scale, is_array in FAMILIES:
        entry = fits[time_slice]["families"].get(family)
        if entry is None:
            continue
        table = _constraint_table(
            ods, time_slice=time_slice, family=family, is_array=is_array
        )
        if entry["fit_role"] == "prescribed":
            panels.append(
                LineSeries(
                    series=(
                        Series(
                            x=table.index,
                            y=np.zeros_like(table.index),
                            label="prescribed exactly",
                            style=dict(_STATE_STYLE["enabled"]),
                        ),
                    ),
                    x_label="Constraint index",
                    y_label="z",
                    title=f"{title}: prescribed, not fitted",
                )
            )
            continue
        from .efit_quality import normalized_residuals, sigma_unit_factor

        k, _spread = sigma_unit_factor(table)
        z = normalized_residuals(table, k)
        series = _state_series(table, z)
        # Two-point spans, so the bands read as lines rather than as a cloud of
        # markers competing with the residuals themselves.
        span = np.array([table.index.min(), table.index.max()], dtype=float)
        for level, style in ((2.0, ":"), (3.0, "--")):
            for sign in (1.0, -1.0):
                series.append(
                    Series(
                        x=span,
                        y=np.full(2, sign * level),
                        label=f"±{level:g}σ" if sign > 0 else "",
                        style={"linestyle": style, "color": "0.6", "lw": 0.9},
                    )
                )
        bias = entry.get("z_bias", float("nan"))
        se = entry.get("z_bias_standard_error", float("nan"))
        flag = " (significant)" if entry.get("z_bias_significant") else ""
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="Constraint index",
                y_label="z = (m − r)·w / k",
                title=(
                    f"{title}: z RMS {entry.get('z_rms', float('nan')):.3g}, "
                    f"bias {bias:+.3g} ± {se:.3g}{flag}, "
                    f"max |z| {entry.get('z_abs_max', float('nan')):.3g}"
                ),
            )
        )

    if not panels:
        raise ValueError("equilibrium ODS carries no fitted constraints to assess")
    return Panels(
        models=tuple(panels),
        ncols=2,
        share_x=False,
        suptitle=_efit_suptitle(ods, "EFIT fit quality", times, time_slice),
    )


def _build_equilibrium_convergence(ods: Any, **options: Any) -> Panels:
    """Did the solve reach what it was asked to reach, and is it self-consistent?"""
    from .efit_quality import efit_quality_metrics

    _require_slices(ods)
    times = _slice_times(ods)
    metrics = efit_quality_metrics(ods)
    blocks = [entry["convergence"] for entry in metrics["slices"]]
    panels: list[Any] = []

    final = np.array([block["error"]["final_error"] for block in blocks], dtype=float)
    exit_tolerance = np.array(
        [block["error"]["exit_tolerance"] for block in blocks], dtype=float
    )
    acceptance = np.array(
        [block["error"]["acceptance_tolerance"] for block in blocks], dtype=float
    )
    if np.isfinite(final).any():
        # Two different thresholds, and EFIT applies only the second one when it
        # decides whether to accept the slice.
        series = [Series(x=times, y=final, label="terror (final GS error)",
                         style={"marker": "."})]
        inert = blocks[0]["error"].get("exit_tolerance_effective") is False
        if np.isfinite(exit_tolerance).any():
            series.append(
                Series(
                    x=times,
                    y=exit_tolerance,
                    label=(
                        "error (inert: nxiter=1, iconvr=2)"
                        if inert
                        else "iteration exit tolerance (error)"
                    ),
                    style={
                        "linestyle": ":" if inert else "--",
                        "color": "0.75" if inert else "0.5",
                        "lw": 1.0,
                    },
                )
            )
        if np.isfinite(acceptance).any():
            name = blocks[0]["error"]["acceptance_tolerance_name"]
            source = blocks[0]["error"]["acceptance_tolerance_source"]
            series.append(
                Series(x=times, y=acceptance,
                       label=f"acceptance threshold ({name}, {source})",
                       style={"linestyle": "-.", "color": "tab:red", "lw": 1.0})
            )
        # For iconvr=2 the statistic with content is how the solve terminated,
        # not the ratio against a tolerance the solver never consults.
        stopped = sum(
            1 for block in blocks
            if block["iterations"]["stopped_on_criterion"]
        )
        capped = sum(1 for block in blocks if block["iterations"]["hit_cap"])
        if inert:
            headline = (
                f"{stopped}/{len(blocks)} stopped on the iconvr=2 criterion"
                + (f", {capped} exhausted iterations" if capped else "")
            )
        else:
            reached = sum(
                1 for block in blocks if block["error"]["reached_exit_tolerance"]
            )
            headline = f"{reached}/{len(blocks)} reached the exit tolerance"
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="time",
                x_unit="s",
                y_label="normalized GS error",
                log_y=True,
                title=f"{blocks[0]['error']['final_error_source']}: {headline}",
            )
        )

    iterations = np.array([block["iterations"]["iterations"] for block in blocks], dtype=float)
    caps = np.array([block["iterations"]["iteration_cap"] for block in blocks], dtype=float)
    if np.isfinite(iterations).any():
        series = [Series(x=times, y=iterations, label="iterations", style={"marker": "."})]
        if np.isfinite(caps).any():
            series.append(
                Series(x=times, y=caps, label="cap",
                       style={"linestyle": "--", "color": "0.5", "lw": 1.0})
            )
        hit = sum(1 for block in blocks if block["iterations"]["hit_cap"])
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="time", x_unit="s", y_label="iterations",
                title=f"Iterations against cap — {hit} slice(s) hit it",
            )
        )

    histories = [block["history"] for block in blocks if block["history"]["available"]]
    if histories:
        raw = [
            _array(ods, f"equilibrium.code.parameters.time_slice.{index}"
                        ".meqdsk.variables.cerror.data")
            for index in range(times.size)
        ]
        series = tuple(
            Series(
                x=np.arange(values.size, dtype=float),
                y=values,
                label=f"t={times[index] * 1e3:.1f} ms",
            )
            for index, values in enumerate(raw)
            if values is not None and values.size > 1
        )
        if series:
            panels.append(
                LineSeries(
                    series=series,
                    x_label="iteration",
                    y_label="Grad-Shafranov error",
                    log_y=True,
                    title="Convergence history per slice",
                )
            )
    elif panels:
        # Only worth a panel alongside real content -- appending it
        # unconditionally would mean this figure could never report that it has
        # nothing to show, which is the failure the stage contract relies on.
        panels.append(
            LineSeries(
                series=(
                    Series(x=times, y=np.full(times.size, np.nan), label="unavailable"),
                ),
                x_label="time", x_unit="s", y_label="Grad-Shafranov error",
                title="Convergence history: no m-file cerror was mapped",
            )
        )

    spread = np.array(
        [block["self_consistency"].get("ip_relative_spread", np.nan) for block in blocks],
        dtype=float,
    )
    axis_offset = np.array(
        [block["self_consistency"].get("psi_axis_grid_offset", np.nan)
         for block in blocks],
        dtype=float,
    )
    not_extremum = sum(
        1 for block in blocks
        if block["self_consistency"].get("magnetic_axis_is_local_extremum") is False
    )
    consistency = []
    if np.isfinite(spread).any():
        consistency.append(
            Series(x=times, y=spread, label="Ip relative spread", style={"marker": "."})
        )
    if np.isfinite(axis_offset).any():
        consistency.append(
            Series(x=times, y=axis_offset, label="ψ axis vs flux map",
                   style={"marker": "."})
        )
    if consistency:
        panels.append(
            LineSeries(
                series=tuple(consistency),
                x_label="time", x_unit="s", y_label="relative difference",
                log_y=True,
                title=(
                    "EFIT outputs against each other"
                    + (f" — {not_extremum} slice(s) axis not stationary" if not_extremum else "")
                ),
            )
        )

    verdicts = [block["verdict"] for block in blocks]
    known = [v for v in verdicts if v["accepted"] is not None]
    if known:
        margin = np.array(
            [block["error"]["chi_squared_margin"] for block in blocks], dtype=float
        )
        series = [
            Series(
                x=times,
                y=np.array(
                    [1.0 if v["accepted"] else 0.0 if v["accepted"] is not None
                     else np.nan for v in verdicts],
                    dtype=float,
                ),
                label="accepted (a-file jflag/lflag)",
                style={"marker": "o", "linestyle": "none"},
            )
        ]
        if np.isfinite(margin).any():
            series.append(
                Series(x=times, y=margin,
                       label=f"χ² / {blocks[0]['error']['chi_squared_limit_name']}",
                       style={"marker": "."})
            )
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="time", x_unit="s", y_label="accepted / χ² margin",
                y_limits=(-0.2, 1.2),
                title=(
                    f"EFIT acceptance — {sum(1 for v in known if v['accepted'])}/{len(known)}"
                    f" accepted; χ² ≤ {blocks[0]['error']['chi_squared_limit_name']}"
                    " is a stopping precondition, not a margin"
                ),
            )
        )

    if not panels:
        raise ValueError("equilibrium ODS carries no convergence information")
    return Panels(
        models=tuple(panels),
        ncols=2,
        share_x=False,
        suptitle=_efit_suptitle(ods, "EFIT numerical convergence", times, None),
    )


RECIPES["equilibrium_overview_fit_quality"] = CallableRecipe(
    builder=_build_equilibrium_fit_quality,
    description="Reduced chi-square, per-family chi-square share and normalized residuals.",
)
RECIPES["equilibrium_overview_convergence"] = CallableRecipe(
    builder=_build_equilibrium_convergence,
    description="Grad-Shafranov error against tolerance, iterations, and self-consistency.",
)


RECIPES["equilibrium_overview_constraints"] = CallableRecipe(
    builder=_build_equilibrium_constraints,
    description="Magnetic constraints submitted to EFIT, by family and channel state.",
)
RECIPES["equilibrium_overview_constraint_coverage"] = CallableRecipe(
    builder=_build_equilibrium_constraint_coverage,
    description="Enabled, disabled and missing constraint channels across slices.",
)
RECIPES["equilibrium_overview_residuals"] = CallableRecipe(
    builder=_build_equilibrium_residuals,
    description="Measured-minus-reconstructed residuals by family, beside convergence.",
)


# ---------------------------------------------------------------------------
# Linear MHD stability (issue #139)
# ---------------------------------------------------------------------------

def _build_mhd_linear_energy_perturbed(ods: Any, **options: Any) -> LineSeries:
    """Perturbed potential energy against time, one trace per toroidal mode.

    ``toroidal_mode`` is an IMAS array of structures whose position is not the
    mode number -- only ``n_tor`` is -- so the traces are pivoted by ``n_tor``
    rather than by array index.
    """
    count = _count(ods, "mhd_linear.time_slice")
    if count == 0:
        raise ValueError("mhd_linear ODS carries no time slices")
    times = _array(ods, "mhd_linear.time")
    if times is None or times.size < count:
        times = np.arange(count, dtype=float)
    times = np.asarray(times[:count], dtype=float)

    by_mode: dict[int, dict[int, float]] = {}
    for index in range(count):
        root = f"mhd_linear.time_slice.{index}.toroidal_mode"
        for position in range(_count(ods, root)):
            n_tor = _get(ods, f"{root}.{position}.n_tor")
            energy = _get(ods, f"{root}.{position}.energy_perturbed")
            if n_tor is None or energy is None:
                continue
            value = _scalar(energy)
            if not np.isfinite(value):
                continue
            by_mode.setdefault(int(n_tor), {})[index] = value
    if not by_mode:
        raise ValueError(
            "mhd_linear ODS carries no perturbed-energy values; only DCON writes "
            "energy_perturbed, and none was mapped for this shot"
        )

    series = []
    for n_tor in sorted(by_mode):
        samples = by_mode[n_tor]
        indexes = sorted(samples)
        series.append(
            Series(
                x=times[indexes],
                y=np.asarray([samples[index] for index in indexes], dtype=float),
                label=f"n={n_tor}",
                style={"marker": "."},
            )
        )
    pulse = _get(ods, "dataset_description.data_entry.pulse", "")
    return LineSeries(
        series=tuple(series),
        x_label="time",
        x_unit="s",
        y_label="perturbed energy $\\delta W$",
        title=f"Linear MHD stability — shot {pulse} (negative is unstable)",
    )


RECIPES["mhd_linear_time_energy_perturbed"] = CallableRecipe(
    builder=_build_mhd_linear_energy_perturbed,
    description="DCON perturbed potential energy against time, per toroidal mode.",
)


# ---------------------------------------------------------------------------
# Eddy-stage vacuum magnetics (issue #139)
# ---------------------------------------------------------------------------

def _vacuum_channels(ods: Any, options: Mapping[str, Any]):
    from vaft.omas.vacuum_magnetics import (
        plasma_onset_time,
        synthetic_vacuum_magnetics,
        vacuum_magnetics_metrics,
    )

    channels = synthetic_vacuum_magnetics(
        ods,
        per_family=int(options.get("per_family", 2)),
        channels=options.get("channels"),
    )
    onset = plasma_onset_time(ods)
    ip_time = _array(ods, "magnetics.ip.0.time")
    ip_data = _array(ods, "magnetics.ip.0.data")
    metrics = vacuum_magnetics_metrics(
        channels,
        plasma_onset=onset,
        plasma_current=(
            None if ip_time is None or ip_data is None else (ip_time, ip_data)
        ),
        # The same threshold the residual figure draws its band at, so the
        # markers and the Delta-t annotations describe the band the reader sees.
        sigma=float(options.get("sigma", 5.0)),
    )
    return channels, metrics


def _vacuum_suptitle(ods: Any, metrics: Mapping[str, Any], headline: str) -> str:
    pulse = _get(ods, "dataset_description.data_entry.pulse", "")
    summary = metrics["summary"]
    return (
        f"{headline} — shot {pulse}\n"
        f"{summary['channel_count']} channels, median eddy improvement "
        f"{summary['median_improvement']:.2f} (worst {summary['min_improvement']:.2f})"
    )


def _build_magnetics_vacuum(ods: Any, **options: Any) -> Panels:
    channels, metrics = _vacuum_channels(ods, options)
    rows = {(row["kind"], row["index"]): row for row in metrics["channels"]}
    panels = []
    for channel in channels:
        row = rows[(channel.kind, channel.index)]
        panels.append(
            LineSeries(
                series=(
                    Series(x=channel.time, y=channel.measured, label="measured",
                           style={"lw": 1.8}),
                    Series(x=channel.time, y=channel.coil, label="coil",
                           style={"lw": 1.2, "linestyle": "--"}),
                    Series(x=channel.time, y=channel.coil_eddy, label="coil+eddy",
                           style={"lw": 1.2}),
                ),
                x_label="time",
                x_unit="s",
                y_label=channel.unit,
                title=(
                    f"{channel.name} [{channel.family}]\n"
                    f"pre-plasma improvement {row['improvement']:.2f}"
                ),
            )
        )
    return Panels(
        models=tuple(panels),
        ncols=3,
        share_x=True,
        suptitle=_vacuum_suptitle(ods, metrics, "Synthetic vacuum magnetics"),
    )


def _onset_marker(time, values, onset: float, label: str, style: Mapping[str, Any]):
    """A vertical marker at ``onset``, spanning the panel's own value range."""
    if not np.isfinite(onset):
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = float(np.min(finite)), float(np.max(finite))
    if low == high:
        low, high = low - 1.0, high + 1.0
    return Series(
        x=np.array([onset, onset]), y=np.array([low, high]), label=label,
        style=dict(style),
    )


def _build_magnetics_plasma_residual(ods: Any, **options: Any) -> Panels:
    channels, metrics = _vacuum_channels(ods, options)
    rows = {(row["kind"], row["index"]): row for row in metrics["channels"]}
    sigma = float(options.get("sigma", 5.0))
    plasma_onset = float(metrics["plasma_onset"])
    current_onset = float(metrics["plasma_current_onset"])

    panels: list[Any] = []
    ip_time = _array(ods, "magnetics.ip.0.time")
    ip_data = _array(ods, "magnetics.ip.0.data")
    if ip_time is not None and ip_data is not None:
        ip_series = [Series(x=ip_time, y=ip_data * 1e-3, label="Ip", style={"lw": 1.8})]
        marker = _onset_marker(
            ip_time, ip_data * 1e-3, current_onset, "Ip onset",
            {"linestyle": "--", "color": "0.35", "lw": 1.0},
        )
        if marker is not None:
            ip_series.append(marker)
        panels.append(
            LineSeries(
                series=tuple(ip_series),
                x_label="time", x_unit="s", y_label="kA",
                title=f"Plasma current — onset {current_onset * 1e3:.1f} ms",
            )
        )

    for channel in channels:
        row = rows[(channel.kind, channel.index)]
        window = channel.time < plasma_onset
        residual = channel.residual
        reference = residual[window]
        baseline, noise = noise_band(reference)
        if not reference.size:
            baseline, noise = 0.0, 0.0
        band = sigma * noise
        series = [
            Series(x=channel.time, y=residual, label="residual", style={"lw": 1.4}),
            Series(
                x=channel.time,
                y=np.full(channel.time.size, baseline + band),
                label=f"±{sigma:g}σ pre-plasma",
                style={"lw": 0.9, "linestyle": ":", "color": "0.5"},
            ),
            Series(
                x=channel.time,
                y=np.full(channel.time.size, baseline - band),
                label="",
                style={"lw": 0.9, "linestyle": ":", "color": "0.5"},
            ),
        ]
        for onset, name, color in (
            (current_onset, "Ip onset", "0.35"),
            (row["residual_onset"], "residual onset", "tab:red"),
        ):
            marker = _onset_marker(
                channel.time, residual, onset, name,
                {"linestyle": "--", "color": color, "lw": 1.0},
            )
            if marker is not None:
                series.append(marker)
        delta = row["onset_delta"]
        timing = "no onset" if not np.isfinite(delta) else f"Δt {delta * 1e3:+.1f} ms"
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="time", x_unit="s", y_label=channel.unit,
                title=f"{channel.name} [{channel.family}]\n{timing}",
            )
        )

    return Panels(
        models=tuple(panels),
        ncols=3,
        share_x=True,
        suptitle=_vacuum_suptitle(ods, metrics, "Plasma residual after coil+eddy"),
    )


RECIPES["magnetics_overview_vacuum"] = CallableRecipe(
    builder=_build_magnetics_vacuum,
    description="Measured, coil-only and coil+eddy synthetic magnetics per channel.",
)
RECIPES["magnetics_overview_plasma_residual"] = CallableRecipe(
    builder=_build_magnetics_plasma_residual,
    description="Plasma residual left by the coil+eddy synthetic vacuum response.",
)


def _chease_comparison_metrics(ods: Any) -> dict[str, dict[str, float]]:
    """The per-time-slice `comparison_metrics` embedded by `generate_chease_ods.py`.

    Lives on `equilibrium.code.parameters` as a JSON blob, because CHEASE's
    refinement-vs-input comparison needs the *input* g-file too, which the
    refined ODS otherwise never carries.
    """
    raw = _get(ods, "equilibrium.code.parameters")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return payload.get("comparison_metrics", {}) or {}


def _build_chease_refinement_summary(ods: Any, **options: Any) -> Panels:
    """How far CHEASE moved each profile and the boundary, slice by slice."""
    count = _require_slices(ods)
    times = _slice_times(ods)
    comparison = _chease_comparison_metrics(ods)
    if not comparison:
        raise ValueError(
            "no CHEASE comparison_metrics embedded in equilibrium.code.parameters; "
            "the refined ODS carries no refinement-vs-input comparison to plot"
        )

    def series_for(key: str, label: str) -> Series | None:
        values = np.array(
            [comparison.get(str(index), {}).get(key, np.nan) for index in range(count)],
            dtype=float,
        )
        if not np.isfinite(values).any():
            return None
        return Series(x=times, y=values, label=label, style={"marker": "."})

    panels: list[Any] = []

    profile_series = [
        series
        for series in (
            series_for("q_rms_rel", "q"),
            series_for("pressure_rms_rel", "pressure"),
            series_for("pprime_rms_rel", "p′"),
            series_for("ffprim_rms_rel", "FF′"),
        )
        if series is not None
    ]
    if profile_series:
        panels.append(
            LineSeries(
                series=tuple(profile_series),
                x_label="time",
                x_unit="s",
                y_label="RMS-relative change",
                title="Profile change from refinement",
            )
        )

    boundary_series = [
        series
        for series in (
            series_for("boundary_r_rms", "R"),
            series_for("boundary_z_rms", "Z"),
            series_for("boundary_rz_rms", "R,Z"),
        )
        if series is not None
    ]
    if boundary_series:
        panels.append(
            LineSeries(
                series=tuple(boundary_series),
                x_label="time",
                x_unit="s",
                y_label="boundary RMS change [m]",
                title="Boundary displacement from refinement",
            )
        )

    psi_series = [
        series
        for series in (
            series_for("psi_axis_abs_diff", "ψ axis"),
            series_for("psi_boundary_abs_diff", "ψ boundary"),
        )
        if series is not None
    ]
    if psi_series:
        panels.append(
            LineSeries(
                series=tuple(psi_series),
                x_label="time",
                x_unit="s",
                y_label="|Δψ| [Wb]",
                title="Flux normalization shift",
            )
        )

    current_series = series_for("current_rel_diff", "Ip")
    if current_series is not None:
        panels.append(
            LineSeries(
                series=(current_series,),
                x_label="time",
                x_unit="s",
                y_label="relative Ip change",
                title="Plasma current self-consistency",
            )
        )

    if not panels:
        raise ValueError("CHEASE comparison_metrics carried no finite values to plot")

    return Panels(
        models=tuple(panels), ncols=2, share_x=True, suptitle="CHEASE refinement summary"
    )


def _build_chease_profile_validity(ods: Any, **options: Any) -> Panels:
    """q0/q95, q-monotonicity and pressure positivity of the refined equilibrium.

    Unlike the refinement summary, every value here is read straight off the
    refined time slices themselves -- no input-side comparison is needed to
    ask whether the *result* is physically sound.
    """
    count = _require_slices(ods)
    times = _slice_times(ods)

    q0 = np.array(
        [_scalar(_get(ods, f"equilibrium.time_slice.{i}.global_quantities.q_axis", np.nan)) for i in range(count)]
    )
    q95 = np.array(
        [_scalar(_get(ods, f"equilibrium.time_slice.{i}.global_quantities.q_95", np.nan)) for i in range(count)]
    )

    panels: list[Any] = []
    q_series = [
        Series(x=times, y=values, label=label, style={"marker": "."})
        for values, label in ((q0, "q0"), (q95, "q95"))
        if np.isfinite(values).any()
    ]
    if q_series:
        panels.append(
            LineSeries(
                series=tuple(q_series),
                x_label="time",
                x_unit="s",
                y_label="q",
                title="Core and edge safety factor",
            )
        )

    monotonic_flags = np.empty(count, dtype=float)
    pressure_flags = np.empty(count, dtype=float)
    for index in range(count):
        q = np.asarray(_get(ods, f"equilibrium.time_slice.{index}.profiles_1d.q", []), dtype=float)
        pressure = np.asarray(
            _get(ods, f"equilibrium.time_slice.{index}.profiles_1d.pressure", []), dtype=float
        )
        diffs = np.diff(q)
        monotonic_flags[index] = float(diffs.size == 0 or np.all(diffs >= 0) or np.all(diffs <= 0))
        pressure_flags[index] = float(pressure.size == 0 or np.all(pressure >= 0))

    panels.append(
        LineSeries(
            series=(
                Series(
                    x=times, y=monotonic_flags, label="q monotonic",
                    style={"marker": "o", "linestyle": "none"},
                ),
                Series(
                    x=times, y=pressure_flags, label="pressure ≥ 0",
                    style={"marker": "x", "linestyle": "none"},
                ),
            ),
            x_label="time",
            x_unit="s",
            y_label="1 = ok, 0 = flagged",
            y_limits=(-0.05, 1.05),
            title="Physical-validity flags",
        )
    )

    return Panels(
        models=tuple(panels), ncols=1, share_x=True, suptitle="CHEASE refined-profile validity"
    )


RECIPES["chease_overview_refinement_summary"] = CallableRecipe(
    builder=_build_chease_refinement_summary,
    description="Profile and boundary RMS change from refinement, slice by slice.",
)
RECIPES["chease_overview_profile_validity"] = CallableRecipe(
    builder=_build_chease_profile_validity,
    description="q0/q95, q-monotonicity and pressure positivity of the refined equilibrium.",
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
    if isinstance(recipe, PowerSpectrumRecipe):
        return _build_power_spectrum(entries[0][1], recipe, **options)
    if isinstance(recipe, CallableRecipe):
        return recipe.builder(entries[0][1], **options)
    raise TypeError(f"unsupported recipe type {type(recipe).__name__} for {name!r}")
