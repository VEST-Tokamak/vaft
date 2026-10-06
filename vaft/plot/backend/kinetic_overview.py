"""Cross-diagnostic local kinetic profiles, with measurement semantics intact."""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping

import numpy as np

from vaft.plot.backend.access import array, count, get
from vaft.plot.models import Panels, Profile1D, Series

__all__ = []  # Extraction details are private; the registered plot is public.


COORDINATES = ("auto", "R", "r_major", "psi_norm", "rho_pol_norm", "rho_tor_norm")
_LABELS = {
    "R": "Major radius R [m]",
    "r_major": "Major radius R [m]",
    "psi_norm": "Normalized poloidal flux ψ_N",
    "rho_pol_norm": "Normalized poloidal radius √ψ_N",
    "rho_tor_norm": "Normalized toroidal flux radius ρ_tor,N",
}
_FIELDS = (
    ("n_e", "Electron density", "m^-3", "n_e", "density", "n_e"),
    ("T_e", "Electron temperature", "eV", "t_e", "temperature", "t_e"),
    ("T_i", "Ion temperature", "eV", "t_i", "temperature", None),
    ("V_phi", "Toroidal velocity", "m/s", "velocity_tor", "velocity.toroidal", None),
)
_NOTICE = "Cross-shot composite — not a physical VEST discharge"


def _scalar(value) -> float | None:
    if value is None:
        return None
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return float(values[0]) if values.size else None


def _selected(array_value, index: int) -> float | None:
    values = np.asarray(array_value, dtype=float).reshape(-1) if array_value is not None else np.array([])
    return float(values[min(index, values.size - 1)]) if values.size else None


def _closest(times, target: float | None) -> int:
    if times is None or target is None:
        return 0
    axis = np.asarray(times, dtype=float).reshape(-1)
    return int(np.nanargmin(abs(axis - target))) if axis.size else 0


@functools.lru_cache(maxsize=1)
def _packaged_manifest() -> dict | None:
    """The repository fixture's validated manifest, or None when it is not checked out."""
    try:
        from vaft.data import unified_diagnostics_manifest

        return unified_diagnostics_manifest()
    except (FileNotFoundError, ValueError, OSError):
        return None


def composite_provenance(ods, manifest: Mapping | None = None) -> dict | None:
    """Reference and source shots of a cross-shot composite; None for a discharge.

    The composite is recognised by the notice in its ``dataset_description``.
    Its shot numbers are not literals here: they come from the fixture
    manifest -- the ``geometry_manifest`` option when the caller passes one,
    otherwise the packaged fixture's -- so a regenerated fixture relabels
    itself.  A composite with no manifest at hand keeps the geometry
    reference its notice names and has no source shots to show.
    """
    comment = str(get(ods, "dataset_description.ids_properties.comment", ""))
    if _NOTICE.split(" — ")[0] not in comment:
        return None
    if not (isinstance(manifest, Mapping) and manifest.get("kind") == "cross-shot-diagnostic-fixture"):
        manifest = _packaged_manifest()
    manifest = manifest or {}
    sources = {
        name: int(record["source_shot"])
        for name, record in manifest.get("sources", {}).items()
        if isinstance(record, Mapping) and isinstance(record.get("source_shot"), int)
    }
    reference = manifest.get("geometry_reference", {}).get("source_shot")
    if reference is None:
        found = re.search(r"geometry reference:? shot (\d+)", comment)
        reference = int(found.group(1)) if found else None
    return {"reference": reference, "sources": sources}


def _shot_label(shot: int | None) -> str:
    """``"shot N"`` when the manifest names the source; ``"source shot"`` when it cannot."""
    return f"shot {shot}" if shot is not None else "source shot"


def _equilibrium_time_axis(ods) -> np.ndarray | None:
    """The equilibrium slice times, or None when the ODS records none."""
    axis = array(ods, "equilibrium.time")
    if axis is None or not np.isfinite(axis).any():
        slices = [_scalar(get(ods, f"equilibrium.time_slice.{index}.time"))
                  for index in range(count(ods, "equilibrium.time_slice"))]
        if not any(value is not None for value in slices):
            return None
        axis = np.asarray([np.nan if value is None else value for value in slices], dtype=float)
    return np.asarray(axis, dtype=float).reshape(-1)


def _equilibrium_slice(ods, target: float | None) -> tuple[int, float | None]:
    """``(index, time)`` of the equilibrium slice nearest the kinetic target time.

    The kinetic sample is chosen by time, so the equilibrium it is mapped
    through must be too (slice 0 is only right when it is the nearest one).
    Without slice times, or without a target, slice 0 and ``time=None``.
    """
    axis = _equilibrium_time_axis(ods)
    if axis is None or target is None:
        return 0, None
    index = _closest(axis, target)
    when = float(axis[index])
    return index, (when if np.isfinite(when) else None)


def _mapped_coordinate(ods, r: np.ndarray, z: np.ndarray, coordinate: str,
                       target: float | None) -> np.ndarray | None:
    if coordinate in ("R", "r_major"):
        return r
    if count(ods, "equilibrium.time_slice") == 0:
        return None
    from vaft.process.profile import CoordinateUnavailableError, equilibrium_mapping_points

    _, when = _equilibrium_slice(ods, target)
    mapped = equilibrium_mapping_points(ods, r, z, time=when)
    try:
        result = mapped.select(coordinate)
    except CoordinateUnavailableError:
        # The recipe omits what it cannot place rather than failing the
        # whole overview; the fitters' refusal is theirs to raise.
        return None
    return None if result is None else np.asarray(result, dtype=float)


def _auto_coordinate(ods, target: float | None) -> str:
    """The flux coordinate ``auto`` resolves to, or ``R`` without an equilibrium or fit.

    ``rho_tor_norm`` is preferred, but only when the equilibrium can supply
    it: one without a usable ``q`` profile (a legacy fluxSurfaces mapping, a
    reconstruction with ``q`` missing) falls back to ``rho_pol_norm``, which
    every equilibrium gives, instead of mapping every measurement to nothing.
    """
    has_fit = bool(count(ods, "core_profiles.profiles_1d"))
    has_equilibrium = bool(count(ods, "equilibrium.time_slice"))
    has_local = bool(count(ods, "thomson_scattering.channel") or count(ods, "charge_exchange.channel"))
    if not (has_fit or (has_equilibrium and has_local)):
        return "R"
    if not (has_equilibrium and has_local):
        return "rho_tor_norm"
    index, when = _equilibrium_slice(ods, target)
    axis_r = _scalar(get(ods, f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis.r"))
    axis_z = _scalar(get(ods, f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis.z"))
    if axis_r is None or axis_z is None:
        return "rho_tor_norm"
    from vaft.process.profile import equilibrium_mapping_points

    probe = equilibrium_mapping_points(ods, np.array([axis_r]), np.array([axis_z]), time=when)
    return "rho_tor_norm" if probe.rho_tor_norm is not None else "rho_pol_norm"


def _core_coordinate(ods, index: int, coordinate: str,
                     target: float | None) -> np.ndarray | None:
    prefix = f"core_profiles.profiles_1d.{index}.grid"
    if coordinate in ("R", "r_major"):
        # A flux coordinate does not name one physical major radius. Refuse to
        # turn a fitted flux profile into an invented midplane R trace.
        return None
    if coordinate == "rho_tor_norm":
        return array(ods, f"{prefix}.rho_tor_norm")
    rho_pol = array(ods, f"{prefix}.rho_pol_norm")
    if rho_pol is not None:
        return rho_pol**2 if coordinate == "psi_norm" else rho_pol
    psi = array(ods, f"{prefix}.psi")
    slice_index, _ = _equilibrium_slice(ods, target)
    axis = _scalar(get(ods, f"equilibrium.time_slice.{slice_index}.global_quantities.psi_axis"))
    edge = _scalar(get(ods, f"equilibrium.time_slice.{slice_index}.global_quantities.psi_boundary"))
    if psi is None or axis is None or edge is None or edge == axis:
        return None
    normalized = (psi - axis) / (edge - axis)
    return normalized if coordinate == "psi_norm" else np.sqrt(np.clip(normalized, 0, None))


def _diagnostic_points(ods, family: str, signal: str, coordinate: str,
                       target: float | None, label: str, entry: str) -> Series | None:
    times = array(ods, f"{family}.time")
    time_index = _closest(times, target)
    radii, heights, values, errors = [], [], [], []
    for index in range(count(ods, f"{family}.channel")):
        prefix = f"{family}.channel.{index}"
        position = prefix + ".position"
        if family == "charge_exchange":
            r = _selected(array(ods, position + ".r.data"), time_index)
            z = _selected(array(ods, position + ".z.data"), time_index)
            path = prefix + ".ion.0." + signal
        else:
            r = _scalar(get(ods, position + ".r"))
            z = _scalar(get(ods, position + ".z"))
            path = prefix + "." + signal
        y = _selected(array(ods, path + ".data"), time_index)
        if r is None or z is None or y is None:
            continue
        radii.append(r)
        heights.append(z)
        values.append(y)
        errors.append(_selected(array(ods, path + ".data_error_upper"), time_index))
    if not values:
        return None
    x = _mapped_coordinate(ods, np.asarray(radii), np.asarray(heights), coordinate, target)
    if x is None:
        return None
    y = np.asarray(values)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    sigma = np.asarray([value if value is not None else np.nan for value in errors])
    return Series(
        x=x[finite], y=y[finite], yerr=sigma[finite] if np.isfinite(sigma[finite]).any() else None,
        label=label + (f" [{entry}]" if entry else ""), role="measurement", entry=entry,
        channel="Thomson" if family == "thomson_scattering" else "CX impurity ion",
        style={"marker": "o" if family == "thomson_scattering" else "s", "linestyle": "none"},
    )


def _core_profile(ods, path: str, coordinate: str, label: str,
                  target: float | None, entry: str) -> Series | None:
    times = array(ods, "core_profiles.time")
    index = _closest(times, target)
    values = array(ods, f"core_profiles.profiles_1d.{index}.{path}")
    x = _core_coordinate(ods, index, coordinate, target)
    if values is None or x is None or len(values) != len(x):
        return None
    finite = np.isfinite(x) & np.isfinite(values)
    if not finite.any():
        return None
    return Series(x=x[finite], y=values[finite], label=label + (f" [{entry}]" if entry else ""),
                  role="reconstruction", entry=entry, channel="Core profile fit",
                  style={"linestyle": "-", "linewidth": 1.8})


def _langmuir_points(ods, signal: str, label: str, entry: str) -> Series | None:
    radii, values = [], []
    for index in range(count(ods, "langmuir_probes.embedded")):
        prefix = f"langmuir_probes.embedded.{index}"
        r = _scalar(get(ods, prefix + ".position.r"))
        time = array(ods, prefix + ".time")
        y = array(ods, prefix + "." + signal + ".data")
        if r is None or time is None or y is None or len(time) != len(y):
            continue
        valid = array(ods, prefix + "." + signal + ".validity_timed")
        eligible = np.isfinite(y)
        if valid is not None and len(valid) == len(y):
            eligible &= valid >= 0
        if not eligible.any():
            continue
        # The composite's probes retain their own source-shot clock. Their
        # central valid sample is selected independently of the kinetic clock.
        selected = np.flatnonzero(eligible)[len(np.flatnonzero(eligible)) // 2]
        radii.append(r)
        values.append(y[selected])
    if not values:
        return None
    return Series(x=np.asarray(radii), y=np.asarray(values),
                  label=label + (f" [{entry}]" if entry else ""), role="derived",
                  entry=entry, channel="Triple probe",
                  style={"marker": "^", "linestyle": "none"})


def build_kinetic_overview(ods, *, coordinate: str = "auto", **options) -> Panels:
    """Build four local-profile panels; omit unavailable and nonlocal sources."""
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {', '.join(COORDINATES)}")
    if any(options.get(key) is not None for key in ("time", "time_index", "time_slice")):
        raise ValueError("kinetic overview has diagnostic-specific times; select a source plot to choose its time")
    provenance = composite_provenance(ods, options.get("geometry_manifest"))
    composite = provenance is not None

    def entry(source: str) -> str:
        """The series entry naming the composite's source shot for ``source``."""
        return _shot_label(provenance["sources"].get(source)) if composite else ""

    target = _selected(array(ods, "core_profiles.time"), 0)
    if coordinate == "auto":
        coordinate = _auto_coordinate(ods, target)
    panels = []
    for quantity, heading, unit, diagnostic_signal, core_signal, probe_signal in _FIELDS:
        series = []
        if quantity in ("n_e", "T_e"):
            measured = _diagnostic_points(ods, "thomson_scattering", diagnostic_signal, coordinate,
                                          target, "Thomson measured", entry("thomson_scattering"))
            if measured is not None:
                series.append(measured)
        else:
            measured = _diagnostic_points(ods, "charge_exchange", diagnostic_signal, coordinate,
                                          target, "CX measured (impurity ion)", entry("charge_exchange"))
            if measured is not None:
                series.append(measured)
        core_path = ("electrons." if quantity in ("n_e", "T_e") else "ion.0.") + core_signal
        fitted = _core_profile(ods, core_path, coordinate, "Core profile fit", target,
                               entry("core_profiles"))
        if fitted is not None:
            series.append(fitted)
        if coordinate in ("R", "r_major") and probe_signal is not None:
            probe = _langmuir_points(ods, probe_signal, "Triple probe derived", entry("langmuir_probes"))
            if probe is not None:
                series.append(probe)
        panels.append(Profile1D(series=tuple(series), coordinate_label=_LABELS[coordinate],
                                y_label=heading, y_unit=unit, title=quantity,
                                x_limits=None if coordinate in ("R", "r_major") else (0.0, 1.0)))
    if not any(panel.series for panel in panels):
        raise ValueError("no compatible local kinetic profiles are available")
    return Panels(models=tuple(panels), ncols=2, share_x=True, share_y=False,
                  suptitle="Kinetic profile overview" +
                  ("\n" + _NOTICE +
                   f"\nKinetic flux: {entry('equilibrium')} equilibrium; "
                   f"machine geometry reference: {_shot_label(provenance['reference'])}"
                   if composite else ""))
