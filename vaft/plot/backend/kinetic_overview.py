"""Cross-diagnostic local kinetic profiles, with measurement semantics intact."""

from __future__ import annotations

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


def _is_composite(ods) -> bool:
    return _NOTICE.split(" — ")[0] in str(get(ods, "dataset_description.ids_properties.comment", ""))


def _mapped_coordinate(ods, r: np.ndarray, z: np.ndarray, coordinate: str) -> np.ndarray | None:
    if coordinate in ("R", "r_major"):
        return r
    if count(ods, "equilibrium.time_slice") == 0:
        return None
    from vaft.process.profile import equilibrium_mapping_points

    mapped = equilibrium_mapping_points(ods, r, z)
    result = mapped.select(coordinate)
    return None if result is None else np.asarray(result, dtype=float)


def _core_coordinate(ods, index: int, coordinate: str) -> np.ndarray | None:
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
    axis = _scalar(get(ods, "equilibrium.time_slice.0.global_quantities.psi_axis"))
    edge = _scalar(get(ods, "equilibrium.time_slice.0.global_quantities.psi_boundary"))
    if psi is None or axis is None or edge is None or edge == axis:
        return None
    normalized = (psi - axis) / (edge - axis)
    return normalized if coordinate == "psi_norm" else np.sqrt(np.clip(normalized, 0, None))


def _diagnostic_points(ods, family: str, signal: str, coordinate: str,
                       target: float | None, label: str) -> Series | None:
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
    x = _mapped_coordinate(ods, np.asarray(radii), np.asarray(heights), coordinate)
    if x is None:
        return None
    y = np.asarray(values)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    sigma = np.asarray([value if value is not None else np.nan for value in errors])
    return Series(
        x=x[finite], y=y[finite], yerr=sigma[finite] if np.isfinite(sigma[finite]).any() else None,
        label=label, role="measurement",
        entry="shot 48224" if _is_composite(ods) else "",
        channel="Thomson" if family == "thomson_scattering" else "CX impurity ion",
        style={"marker": "o" if family == "thomson_scattering" else "s", "linestyle": "none"},
    )


def _core_profile(ods, path: str, coordinate: str, label: str,
                  target: float | None) -> Series | None:
    times = array(ods, "core_profiles.time")
    index = _closest(times, target)
    values = array(ods, f"core_profiles.profiles_1d.{index}.{path}")
    x = _core_coordinate(ods, index, coordinate)
    if values is None or x is None or len(values) != len(x):
        return None
    finite = np.isfinite(x) & np.isfinite(values)
    if not finite.any():
        return None
    return Series(x=x[finite], y=values[finite], label=label, role="reconstruction",
                  entry="shot 48224" if _is_composite(ods) else "", channel="Core profile fit",
                  style={"linestyle": "-", "linewidth": 1.8})


def _langmuir_points(ods, signal: str, label: str) -> Series | None:
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
        # The composite's probes retain their own 42699 clock. Their central
        # valid sample is selected independently of the 48224 kinetic clock.
        selected = np.flatnonzero(eligible)[len(np.flatnonzero(eligible)) // 2]
        radii.append(r)
        values.append(y[selected])
    if not values:
        return None
    return Series(x=np.asarray(radii), y=np.asarray(values), label=label, role="derived",
                  entry="shot 42699" if _is_composite(ods) else "", channel="Triple probe",
                  style={"marker": "^", "linestyle": "none"})


def build_kinetic_overview(ods, *, coordinate: str = "auto", **options) -> Panels:
    """Build four local-profile panels; omit unavailable and nonlocal sources."""
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {', '.join(COORDINATES)}")
    if coordinate == "auto":
        coordinate = (
            "rho_tor_norm" if count(ods, "core_profiles.profiles_1d")
            or (count(ods, "equilibrium.time_slice") and (
                count(ods, "thomson_scattering.channel")
                or count(ods, "charge_exchange.channel")
            )) else "R"
        )
    if any(options.get(key) is not None for key in ("time", "time_index", "time_slice")):
        raise ValueError("kinetic overview has diagnostic-specific times; select a source plot to choose its time")
    composite = _is_composite(ods)
    suffix = " [shot 48224]" if composite else ""
    target = _selected(array(ods, "core_profiles.time"), 0)
    panels = []
    for quantity, heading, unit, diagnostic_signal, core_signal, probe_signal in _FIELDS:
        series = []
        if quantity in ("n_e", "T_e"):
            measured = _diagnostic_points(ods, "thomson_scattering", diagnostic_signal,
                                          coordinate, target, "Thomson measured" + suffix)
            if measured is not None:
                series.append(measured)
        else:
            measured = _diagnostic_points(ods, "charge_exchange", diagnostic_signal,
                                          coordinate, target, "CX measured (impurity ion)" + suffix)
            if measured is not None:
                series.append(measured)
        core_path = ("electrons." if quantity in ("n_e", "T_e") else "ion.0.") + core_signal
        fitted = _core_profile(ods, core_path, coordinate, "Core profile fit" + suffix, target)
        if fitted is not None:
            series.append(fitted)
        if coordinate in ("R", "r_major") and probe_signal is not None:
            probe = _langmuir_points(ods, probe_signal,
                                     "Triple probe derived" + (" [shot 42699]" if composite else ""))
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
                   "\nKinetic flux: shot 48224 equilibrium; machine geometry reference: shot 39915"
                   if composite else ""))
