"""One matched kinetic state in four panels, from stored IMAS paths only (issue #1837).

``kinetic_overview_state`` draws the pressure, density, temperature and
``Z_eff`` profiles of **one** selected kinetic state: a ``core_profiles``
slice, the equilibrium slice of the same time and the Thomson channels mapped
through that equilibrium.  It reads what a pipeline stage stored and does no
physics of its own -- no EFIT, no Thomson fit, no OpenADAS charge states, no
composition and no ``T_i`` inference.  The only arithmetic is presentation:
``p_e + p_i`` for display, ``e n_e T_e`` when no electron pressure is stored
(labelled *derived*), the charged impurity density as the sum of the stored
charge-state densities, and the Thomson electron pressure ``e n_e T_e`` per
channel with its 1 sigma from the measured ``n_e`` and ``T_e`` errors, assumed
independent.

Evidence (the contract as amended by Lane KP on #1837)
------------------------------------------------------
* **Roles** come from the per-quantity ``*_fit.parameters`` records:
  ``origin=measured`` -> measurement, ``origin=assumed|derived`` -> assumed,
  ``origin=inferred`` -> inferred, a fit record (``coordinate=...;
  method=...``) -> fit (and its Thomson points -> fit input); an ion
  temperature record is read with
  :func:`vaft.machine_mapping.core_profiles.classify_ti_record`.  A quantity
  with no record is labelled ``stored``.  A record that is present but in no
  known grammar is refused, never read as "no record".
* **Validity** is the data's own: non-finite ``T_i``, ``p_i`` and ``Z_eff``
  points stay empty.  Thomson channels honour their ``validity`` and
  ``validity_timed`` flags (the renderer demotes flagged points).
* **Lineage and occurrence** are the ``equilibrium_lineage=...;
  equilibrium_occurrence=...`` fields of the main-ion
  ``temperature_fit.parameters`` record.  ``equilibrium_occurrence=`` is
  refused when it contradicts them; without a record the caller's value is
  shown as *unverified*.  The lineage decides the Thomson pressure role:
  ``magnetics`` -> independent validation, ``electron_kinetic`` -> fit input.
* **Z_eff reference**: ``core_profiles.global_quantities.z_eff_resistive`` at
  the slice's own entry of ``core_profiles.time``, else ``target_zeff=...`` of
  the ``zeff_fit.parameters`` record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from vaft.plot.backend.access import array, count, get
from vaft.plot.backend.kinetic_overview import _closest, _core_coordinate, _equilibrium_time_axis
from vaft.plot.display import COORDINATE_LABELS, resolve_display
from vaft.plot.intent import palette
from vaft.plot.models import Panels, Profile1D, ReferenceLine, Series

__all__ = []  # The registered plot is public; the extraction is not.

#: ``coordinate=`` of the view: the toroidal-flux radius the core grid is
#: stored on, or the normalized poloidal flux.
COORDINATES = ("rho_tor_norm", "psi_norm")
#: ``layout=``: a 2 x 2 grid or a 4 x 1 stack of the same four panels.
LAYOUTS = ("grid", "stack")
#: How far apart two samples may be [s] when the time base they are matched on
#: has a single sample, so no step to compare with -- the composition
#: resolver's own pairing tolerance.
SINGLE_SLICE_TOLERANCE = 5e-4
#: How closely ``core_profiles.time`` must hold the slice's own time for a
#: global quantity at that index to belong to it [s].
GLOBAL_TIME_TOLERANCE = 1e-6

#: Elementary charge [C].
_E = 1.602176634e-19

#: Recorded ``origin=`` -> role.
_ORIGIN_ROLES = {"measured": "measurement", "assumed": "assumed", "derived": "assumed", "inferred": "inferred"}
#: :func:`classify_ti_record` kind -> role (a fit record is told apart below).
_TI_ROLES = {"assumed": "assumed", "inferred": "inferred"}
#: Recorded equilibrium lineage -> the role of Thomson in its pressure.
_LINEAGE_TS_ROLES = {"magnetics": "independent validation", "electron_kinetic": "fit input"}
#: The T_i inference whose p_i closes p_e + p_i = p_eq by construction.
CLOSURE_METHODS = frozenset({"equilibrium_pressure_partition"})

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INTEGER = re.compile(r"[0-9]+")

_COLOURS = {
    "equilibrium": palette(0),
    "closure": palette(7),
    "electron": "role:reconstructed",
    "ion": palette(2),
    "impurity": palette(3),
    "measurement": "role:measured",
    "reference": "role:reference",
}


def _scalar(value: Any) -> float | None:
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return float(values[0]) if values.size and np.isfinite(values[0]) else None


def _step(axis: np.ndarray | None) -> float:
    """One sampling step of a time base [s]: its median spacing, or the lone-sample tolerance."""
    if axis is None:
        return SINGLE_SLICE_TOLERANCE
    finite = np.unique(np.asarray(axis, dtype=float)[np.isfinite(axis)])
    if finite.size < 2:
        return SINGLE_SLICE_TOLERANCE
    return float(np.median(np.diff(finite)))


# --- records -------------------------------------------------------------------------


def _record(ods: Any, path: str) -> str | None:
    """The record text at ``path``, ``None`` when absent or blank; refuse a non-text record."""
    raw = get(ods, path)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):
        raise ValueError(f"{path} is a {type(raw).__name__}, not record text: {raw!r}")
    return raw if raw.strip() else None


def _fields(path: str, text: str) -> dict[str, str]:
    from vaft.machine_mapping.core_profiles import ti_record_fields

    fields = ti_record_fields(text)
    if not fields:
        raise ValueError(f"{path} carries no key=value field; a record in no known grammar is refused: {text!r}")
    return fields


def _origin_role(path: str, text: str | None) -> str:
    """The role a composition/profile record states; ``stored`` without one."""
    if text is None:
        return "stored"
    fields = _fields(path, text)
    if "origin" in fields:
        role = _ORIGIN_ROLES.get(fields["origin"].lower())
        if role is None:
            raise ValueError(f"{path} states origin={fields['origin']!r}, which is none of {sorted(_ORIGIN_ROLES)}")
        return role
    if "coordinate" in fields:
        return "fit"
    raise ValueError(f"{path} states neither origin= nor a fit coordinate=: {text!r}")


def _ti_role(path: str, text: str | None) -> str:
    """The role an ion-temperature record states (:func:`classify_ti_record`)."""
    from vaft.machine_mapping.core_profiles import classify_ti_record

    if text is None:
        return "stored"
    fields = _fields(path, text)
    kind = classify_ti_record(text)
    if kind == "measured":
        return "measurement" if "origin" in fields else "fit"
    if kind in _TI_ROLES:
        return _TI_ROLES[kind]
    raise ValueError(f"{path} is an ion-temperature record in no known grammar: {text!r}")


def p_kin_role(ti_record_fields: Mapping[str, str] | None) -> str:
    """How ``p_e + p_i`` is labelled: the one rule, kept here so it can follow Lane KP.

    ``closure identity`` when the stored T_i (whose ``p_i = e n_i T_i``) was
    inferred by partitioning the equilibrium pressure -- then ``p_e + p_i =
    p_eq`` holds by construction and is no agreement test; otherwise ``display
    sum``.  ``pressure_ion_total`` has no ``*_fit`` record in the Data
    Dictionary, so the T_i record is the only place the method is stated.
    """
    if not ti_record_fields:
        return "display sum"
    origin = str(ti_record_fields.get("origin", "")).lower()
    method = str(ti_record_fields.get("method", ""))
    return "closure identity" if origin == "inferred" and method in CLOSURE_METHODS else "display sum"


def _lineage(path: str, text: str | None) -> tuple[str | None, int | None]:
    """``(equilibrium_lineage, equilibrium_occurrence)`` a T_i record states, type-checked."""
    if text is None:
        return None, None
    fields = _fields(path, text)
    lineage = fields.get("equilibrium_lineage")
    if lineage is not None and not _IDENTIFIER.fullmatch(lineage):
        raise ValueError(f"{path}: equilibrium_lineage={lineage!r} is not an identifier")
    occurrence = fields.get("equilibrium_occurrence")
    if occurrence is not None:
        if not _INTEGER.fullmatch(occurrence):
            raise ValueError(f"{path}: equilibrium_occurrence={occurrence!r} is not a non-negative integer")
        occurrence = int(occurrence)
    return lineage, occurrence


def _target_zeff(path: str, text: str | None) -> float | None:
    if text is None:
        return None
    raw = _fields(path, text).get("target_zeff")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{path}: target_zeff={raw!r} is not a number") from None
    if not np.isfinite(value) or value < 1.0:
        raise ValueError(f"{path}: target_zeff={raw!r} is not a finite Z_eff >= 1")
    return value


def _occurrence_option(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"equilibrium_occurrence must be a non-negative integer; got {value!r}")
    return int(value)


# --- selection -----------------------------------------------------------------------


def _available(ods: Any) -> str | None:
    """Why ``ods`` cannot draw the view with its defaults, or ``None``."""
    if not count(ods, "core_profiles.profiles_1d"):
        return "no core_profiles.profiles_1d slice"
    if not count(ods, "equilibrium.time_slice"):
        return "no equilibrium.time_slice to match the kinetic state to"
    index = 0  # the slice the view draws without time=
    base = f"core_profiles.profiles_1d.{index}"
    rho = array(ods, f"{base}.grid.rho_tor_norm")
    if rho is None:
        return f"{base}.grid.rho_tor_norm is not stored"
    if array(ods, f"{base}.electrons.temperature") is None or _electron_density(ods, base)[0] is None:
        return f"{base} stores no electron density and temperature"
    if _is_proxy(ods, index, rho, None):
        return f"{base}.grid.rho_tor_norm is the sqrt(psi_N) proxy, not the toroidal-flux radius"
    return None


def _core_times(ods: Any) -> np.ndarray | None:
    total = count(ods, "core_profiles.profiles_1d")
    base = array(ods, "core_profiles.time")
    times = []
    for index in range(total):
        value = _scalar(get(ods, f"core_profiles.profiles_1d.{index}.time"))
        if value is None and base is not None and index < base.size:
            value = float(base[index])
        times.append(np.nan if value is None else value)
    axis = np.asarray(times, dtype=float)
    return axis if np.isfinite(axis).any() else None


def _electron_density(ods: Any, base: str) -> tuple[np.ndarray | None, str]:
    for leaf in ("density", "density_thermal"):
        values = array(ods, f"{base}.electrons.{leaf}")
        if values is not None:
            return values, leaf
    return None, ""


def _is_proxy(ods: Any, index: int, rho: np.ndarray, target: float | None) -> bool:
    from vaft.data._derived import is_rho_pol_proxy

    psi_norm = _core_coordinate(ods, index, "psi_norm", target)
    if psi_norm is not None and psi_norm.shape != rho.shape:
        psi_norm = None
    return is_rho_pol_proxy(rho, psi_norm)


def _finite(values: np.ndarray | None) -> np.ndarray | None:
    """``values`` with every non-finite point left empty (NaN): the data's own validity."""
    if values is None:
        return None
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _sigma(ods: Any, path: str, size: int) -> np.ndarray | None:
    values = array(ods, path)
    if values is None or values.size != size or not np.isfinite(values).any():
        return None
    return values


# --- Thomson ----------------------------------------------------------------------------


def _thomson(ods: Any, target: float) -> dict | None:
    """Every Thomson channel's ``n_e``/``T_e`` at the sample nearest ``target``, matched by time.

    ``homogeneous_time = 0`` reads each signal's own ``<signal>.time``;
    otherwise ``thomson_scattering.time``.  A channel whose data length does
    not match its time base is skipped, never clamped; a signal whose nearest
    sample lies more than one sampling step of its time base from ``target``
    is left empty.  ``validity`` (channel) and ``validity_timed`` (sample)
    below zero mark a value invalid.  Returns ``None`` without channels.
    """
    total = count(ods, "thomson_scattering.channel")
    if not total:
        return None
    homogeneous = _scalar(get(ods, "thomson_scattering.ids_properties.homogeneous_time"))
    per_signal = homogeneous == 0
    shared = None if per_signal else array(ods, "thomson_scattering.time")
    if not per_signal and (shared is None or not shared.size):
        raise ValueError("thomson_scattering has channels but no time base; its samples cannot be matched to "
                         "the kinetic state by time")
    rows, skipped, times = [], [], []
    for c in range(total):
        prefix = f"thomson_scattering.channel.{c}"
        r, z = _scalar(get(ods, prefix + ".position.r")), _scalar(get(ods, prefix + ".position.z"))
        if r is None or z is None:
            skipped.append((c, "no position"))
            continue
        row = {"channel": c, "r": r, "z": z}
        for signal in ("n_e", "t_e"):
            path = f"{prefix}.{signal}"
            axis = array(ods, path + ".time") if per_signal else shared
            data = array(ods, path + ".data")
            value = error = np.nan
            valid = True
            if data is not None and axis is not None and data.size and axis.size:
                if data.size != axis.size:
                    skipped.append((c, f"{signal}.data has {data.size} samples for a time base of {axis.size}"))
                else:
                    k = _closest(axis, target)
                    if abs(float(axis[k]) - target) <= _step(axis) + 1e-12:
                        times.append(float(axis[k]))
                        value = float(data[k])
                        sigma = array(ods, path + ".data_error_upper")
                        error = float(sigma[k]) if sigma is not None and sigma.size == data.size else np.nan
                        flag = _scalar(get(ods, path + ".validity"))
                        timed = array(ods, path + ".validity_timed")
                        valid = (flag is None or flag >= 0) and (
                            timed is None or timed.size != data.size or timed[k] >= 0)
                    else:
                        skipped.append((c, f"{signal}: nearest sample {abs(float(axis[k]) - target) * 1e3:.3g} ms "
                                           "away, more than one sampling step"))
            row[signal], row[f"sigma_{signal}"], row[f"valid_{signal}"] = value, error, valid
        rows.append(row)
    if not rows:
        return {"rows": [], "skipped": skipped, "time": None}
    keys = ("channel", "r", "z", "n_e", "sigma_n_e", "valid_n_e", "t_e", "sigma_t_e", "valid_t_e")
    out = {key: np.asarray([row[key] for row in rows]) for key in keys}
    out["skipped"] = skipped
    unique = sorted(set(times))
    out["time"] = unique[0] if len(unique) == 1 else (None if not unique else (min(unique), max(unique)))
    out["rows"] = rows
    return out


# --- series ----------------------------------------------------------------------------


def _series(x, y, label, colour, *, yerr=None, linestyle="-", marker=None, valid=None) -> Series | None:
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        return None
    style: dict[str, Any] = {"color": colour, "linestyle": linestyle}
    if marker is not None:
        style.update(marker=marker, linestyle="none")
    mask = None if valid is None or np.all(valid) else np.asarray(valid, dtype=bool)
    return Series(x=x, y=y, label=label, yerr=yerr, style=style, valid_mask=mask)


def _note(label: str) -> Series:
    """A legend-only entry: a stated absence, drawn as nothing."""
    return Series(x=np.array([np.nan, np.nan]), y=np.array([np.nan, np.nan]), label=label,
                  style={"color": "emphasis:medium", "linestyle": "none"})


def _plain(text: str) -> str:
    """A stored label as legend text outside mathtext: no ``$`` or backslash survives."""
    return re.sub(r"[$\\]", "", str(text)).strip()


def build_kinetic_state(ods: Any, *, time: float | None = None, equilibrium_occurrence: Any = None,
                        layout: str = "grid", coordinate: str = "rho_tor_norm", **options: Any) -> Panels:
    """Four matched panels of one stored kinetic state; refuse what does not match."""
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {', '.join(COORDINATES)}; got {coordinate!r}")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}; got {layout!r}")
    if options.get("time_slice") is not None:
        raise ValueError("kinetic_overview_state selects its state by time=, matched by time on every IDS; "
                         "time_slice= would index one of them")
    asked_occurrence = _occurrence_option(equilibrium_occurrence)

    # --- the state: core_profiles slice and the equilibrium of the same time --
    if not count(ods, "core_profiles.profiles_1d"):
        raise ValueError("kinetic_overview_state needs a core_profiles.profiles_1d slice")
    core_times = _core_times(ods)
    j = _closest(core_times, time) if time is not None else 0
    t_core = None if core_times is None else float(core_times[j])
    if t_core is not None and not np.isfinite(t_core):
        t_core = None
    if t_core is None:
        raise ValueError(f"core_profiles.profiles_1d.{j} stores no time; the equilibrium cannot be matched to it")
    if time is not None and abs(t_core - float(time)) > _step(core_times) + 1e-12:
        raise ValueError(f"time={float(time):g} s is {abs(t_core - float(time)) * 1e3:.3g} ms from the nearest "
                         f"core_profiles slice (t = {t_core:.6g} s), more than one core_profiles step")
    eq_axis = _equilibrium_time_axis(ods)
    if not count(ods, "equilibrium.time_slice") or eq_axis is None:
        raise ValueError("kinetic_overview_state needs an equilibrium slice with a time to match the "
                         "core_profiles slice to")
    i = _closest(eq_axis, t_core)
    t_eq = float(eq_axis[i])
    step = _step(eq_axis)
    if not np.isfinite(t_eq) or abs(t_eq - t_core) > step + 1e-12:
        raise ValueError(
            f"the nearest equilibrium slice (t = {t_eq:.6g} s) is {abs(t_eq - t_core) * 1e3:.3g} ms from the "
            f"core_profiles slice (t = {t_core:.6g} s), more than one equilibrium step ({step * 1e3:.3g} ms); "
            "the kinetic state and the equilibrium would not be one state"
        )

    base = f"core_profiles.profiles_1d.{j}"
    eq_base = f"equilibrium.time_slice.{i}"

    # --- the records ---------------------------------------------------------------------
    main_path = _main_ion_path(ods, base)
    ti_path = f"{base}.t_i_average_fit.parameters"
    ti_text = _record(ods, ti_path)
    if ti_text is None and main_path is not None:
        ti_path = f"{main_path}.temperature_fit.parameters"
        ti_text = _record(ods, ti_path)
    lineage_path = f"{main_path or base + '.ion.0'}.temperature_fit.parameters"
    lineage, recorded_occurrence = _lineage(lineage_path, _record(ods, lineage_path))
    if asked_occurrence is not None and recorded_occurrence is not None and asked_occurrence != recorded_occurrence:
        raise ValueError(
            f"equilibrium_occurrence={asked_occurrence}, but the kinetic state was built on equilibrium "
            f"occurrence {recorded_occurrence} ({lineage or 'lineage not recorded'}; {lineage_path}); mapping its "
            "Thomson channels or comparing its pressure through another equilibrium would mix two states"
        )
    occurrence = recorded_occurrence if recorded_occurrence is not None else asked_occurrence
    occurrence_verified = recorded_occurrence is not None

    # --- coordinates ------------------------------------------------------------
    if coordinate == "rho_tor_norm":
        x = array(ods, f"{base}.grid.rho_tor_norm")
        if x is None:
            raise ValueError(f"{base}.grid.rho_tor_norm is not stored; pass coordinate='psi_norm'")
        if _is_proxy(ods, j, x, t_core):
            raise ValueError(
                f"{base}.grid.rho_tor_norm is the sqrt(psi_N) proxy, not the toroidal-flux radius (#276); "
                "it is refused, never relabelled -- re-derive the grid or pass coordinate='psi_norm'"
            )
        x_eq = array(ods, f"{eq_base}.profiles_1d.rho_tor_norm")
        psi_eq_norm = _equilibrium_psi_norm(ods, eq_base)
        if x_eq is not None:
            from vaft.data._derived import is_rho_pol_proxy

            if is_rho_pol_proxy(x_eq, psi_eq_norm if psi_eq_norm is not None and psi_eq_norm.shape == x_eq.shape
                                else None):
                raise ValueError(f"{eq_base}.profiles_1d.rho_tor_norm is the sqrt(psi_N) proxy; it is refused")
    else:
        x = _core_coordinate(ods, j, "psi_norm", t_core)
        if x is None:
            raise ValueError(f"{base} stores neither grid.psi nor grid.rho_pol_norm to place it in psi_norm")
        x_eq = _equilibrium_psi_norm(ods, eq_base)

    # --- core_profiles quantities ------------------------------------------------
    size = x.size
    ne, ne_leaf = _electron_density(ods, base)
    te = array(ods, f"{base}.electrons.temperature")
    if ne is None or te is None or ne.size != size or te.size != size:
        raise ValueError(f"{base} needs electrons.density and electrons.temperature on its grid")
    ne_record = f"{base}.electrons.{ne_leaf}_fit.parameters"
    te_record = f"{base}.electrons.temperature_fit.parameters"
    ne_text, te_text = _record(ods, ne_record), _record(ods, te_record)
    ne_role, te_role = _origin_role(ne_record, ne_text), _origin_role(te_record, te_text)
    pe_stored = array(ods, f"{base}.electrons.pressure")
    pe_derived = pe_stored is None or pe_stored.size != size
    pe = _E * ne * te if pe_derived else pe_stored
    pi = array(ods, f"{base}.pressure_ion_total")
    pi = _finite(pi) if pi is not None and pi.size == size else None
    ti = array(ods, f"{base}.t_i_average")
    ti = _finite(ti) if ti is not None and ti.size == size else None
    ti_role = _ti_role(ti_path, ti_text)
    ti_fields = _fields(ti_path, ti_text) if ti_text is not None else None
    zeff = array(ods, f"{base}.zeff")
    zeff = _finite(zeff) if zeff is not None and zeff.size == size else None
    zeff_record = f"{base}.zeff_fit.parameters"
    zeff_text = _record(ods, zeff_record)
    zeff_role = _origin_role(zeff_record, zeff_text)
    main, impurity = _ion_densities(ods, base, size)

    # --- Thomson, mapped through the selected equilibrium -------------------------
    ts = _thomson(ods, t_core)
    ts_x = None
    ts_note = None
    if ts is not None and len(ts["rows"]):
        from vaft.process.profile import CoordinateUnavailableError, equilibrium_mapping_points

        mapped = equilibrium_mapping_points(ods, ts["r"], ts["z"], time=t_eq)
        try:
            ts_x = mapped.select(coordinate)
        except CoordinateUnavailableError as exc:
            raise ValueError(f"the Thomson channels cannot be placed in {coordinate}: {exc}") from exc
        if not (np.isfinite(ts["n_e"]) | np.isfinite(ts["t_e"])).any():
            ts_note = "no Thomson sample within one sampling step"
    elif ts is not None:
        ts_note = "no usable Thomson channel"

    def ts_points(values, sigma, valid, label, colour, marker, scale=1.0):
        if ts_x is None:
            return None
        keep = np.isfinite(ts_x) & np.isfinite(values)
        if not keep.any():
            return None
        order = np.argsort(ts_x[keep])
        err = (np.asarray(sigma, dtype=float)[keep] * scale)[order]
        return _series(ts_x[keep][order], (np.asarray(values)[keep] * scale)[order], label, colour,
                       yerr=err if np.isfinite(err).any() else None, marker=marker,
                       valid=np.asarray(valid, dtype=bool)[keep][order])

    ts_fit_input = {"n_e": "fit input" if ne_role == "fit" else "measured",
                    "t_e": "fit input" if te_role == "fit" else "measured"}
    ts_pressure_role = _LINEAGE_TS_ROLES.get(lineage or "", "measured")

    # --- panels ------------------------------------------------------------------------
    traces: list[dict] = []
    pressure: list[Series] = []
    density: list[Series] = []
    temperature: list[Series] = []
    charge: list[Series] = []
    panel_name = {id(pressure): "pressure", id(density): "density", id(temperature): "temperature",
                  id(charge): "zeff"}

    def add(panel: list, series: Series | None, quantity: str, path: str, role: str, sigma: str) -> None:
        if series is None:
            return
        panel.append(series)
        traces.append({"panel": panel_name[id(panel)], "label": series.label, "quantity": quantity,
                       "path": path, "role": role, "sigma": sigma})

    p_eq = array(ods, f"{eq_base}.profiles_1d.pressure")
    p_eq_reason = None
    if p_eq is None:
        p_eq_reason = f"{eq_base}.profiles_1d.pressure not stored"
    elif x_eq is None:
        p_eq_reason = f"{eq_base} has no {coordinate} for it"
    elif x_eq.shape != p_eq.shape:
        p_eq_reason = f"{eq_base} pressure and {coordinate} differ in length"
    if p_eq_reason is None:
        add(pressure, _series(x_eq, p_eq, r"$p_{\mathrm{eq}}$ (stored)", _COLOURS["equilibrium"]),
            "p_eq", f"{eq_base}.profiles_1d.pressure", "stored", "none")
    else:
        add(pressure, _note(rf"$p_{{\mathrm{{eq}}}}$ unavailable: {p_eq_reason}"),
            "p_eq", f"{eq_base}.profiles_1d.pressure", "unavailable", "none")
    if pe_derived:
        add(pressure, _series(x, pe, r"$p_e = e\,n_e T_e$ (derived)", _COLOURS["electron"]),
            "p_e", f"{base}.electrons.{ne_leaf} * temperature", "derived", "none")
    else:
        add(pressure, _series(x, pe, r"$p_e$ (stored)", _COLOURS["electron"],
                              yerr=_sigma(ods, f"{base}.electrons.pressure_error_upper", size)),
            "p_e", f"{base}.electrons.pressure", "stored", "stored")
    if pi is not None:
        add(pressure, _series(x, pi, rf"$p_i$ ({ti_role})", _COLOURS["ion"]),
            "p_i", f"{base}.pressure_ion_total", ti_role, "none")
        p_kin = pe + pi
        if np.isfinite(p_kin).any():
            role = p_kin_role(ti_fields)
            add(pressure, _series(x, p_kin, rf"$p_e + p_i$ ({role})", _COLOURS["closure"], linestyle="--"),
                "p_kin", "p_e + p_i", role, "none")
    if ts_x is not None:
        p_ts = _E * ts["n_e"] * ts["t_e"]
        # Independent n_e and T_e errors: no covariance is stored to do better.
        sp_ts = _E * np.hypot(ts["t_e"] * ts["sigma_n_e"], ts["n_e"] * ts["sigma_t_e"])
        add(pressure, ts_points(p_ts, sp_ts, ts["valid_n_e"] & ts["valid_t_e"],
                                rf"$p_e^{{\mathrm{{TS}}}}$ ({ts_pressure_role})", _COLOURS["measurement"], "D"),
            "p_e_thomson", "e * thomson_scattering.channel.{c}.n_e.data * t_e.data", ts_pressure_role,
            "propagated, n_e and T_e errors independent")

    display = resolve_display("m^-3", unit="10^19 m^-3")
    scale = display.scale
    sigma = _sigma(ods, f"{base}.electrons.{ne_leaf}_error_upper", size)
    add(density, _series(x, ne * scale, rf"$n_e$ ({ne_role})", _COLOURS["electron"],
                         yerr=None if sigma is None else sigma * scale),
        "n_e", f"{base}.electrons.{ne_leaf}", ne_role, "stored" if sigma is not None else "none")
    if main is not None:
        values, labels, paths, role = main
        add(density, _series(x, _finite(values) * scale, rf"$n_i$ {', '.join(labels)} ({role})", _COLOURS["ion"]),
            "n_main_ion", " + ".join(paths), role, "none")
    if impurity is not None:
        values, labels, paths, role = impurity
        add(density, _series(x, _finite(values) * scale,
                             rf"$n_{{\mathrm{{imp}}}}^{{+}}$ {', '.join(labels)} ({role})", _COLOURS["impurity"]),
            "n_impurity_ion", " + ".join(paths), role, "none")
    if ts_x is not None:
        role = ts_fit_input["n_e"]
        add(density, ts_points(ts["n_e"], ts["sigma_n_e"], ts["valid_n_e"],
                               rf"$n_e^{{\mathrm{{TS}}}}$ ({role})", _COLOURS["measurement"], "o", scale),
            "n_e_thomson", "thomson_scattering.channel.{c}.n_e.data", role, "stored")

    sigma = _sigma(ods, f"{base}.electrons.temperature_error_upper", size)
    add(temperature, _series(x, te, rf"$T_e$ ({te_role})", _COLOURS["electron"], yerr=sigma),
        "T_e", f"{base}.electrons.temperature", te_role, "stored" if sigma is not None else "none")
    valid_range = None
    if ti is not None:
        finite = np.isfinite(ti)
        if finite.any():
            valid_range = [float(np.min(x[finite])), float(np.max(x[finite]))]
        note = f", {int(finite.sum())}/{size} finite" if not finite.all() else ""
        add(temperature, _series(x, ti, rf"$T_i$ ({ti_role}{note})", _COLOURS["ion"]),
            "T_i", f"{base}.t_i_average", ti_role, "none")
    if ts_x is not None:
        role = ts_fit_input["t_e"]
        add(temperature, ts_points(ts["t_e"], ts["sigma_t_e"], ts["valid_t_e"],
                                   rf"$T_e^{{\mathrm{{TS}}}}$ ({role})", _COLOURS["measurement"], "o"),
            "T_e_thomson", "thomson_scattering.channel.{c}.t_e.data", role, "stored")

    if zeff is not None:
        add(charge, _series(x, zeff, rf"$Z_{{\mathrm{{eff}}}}$ ({zeff_role})", _COLOURS["impurity"]),
            "Z_eff", f"{base}.zeff", zeff_role, "none")
    reference = _zeff_reference(ods, t_core, zeff_record, zeff_text, zeff_role)
    if reference is not None:
        value, label, path, role = reference
        add(charge, _series(np.array([0.0, 1.0]), np.full(2, value), label, _COLOURS["reference"],
                            linestyle="--"),
            "Z_eff_reference", path, role, "none")

    # --- the measured span (valid channels only), shaded -----------------------------
    span_line = None
    if ts_x is not None:
        measured = np.isfinite(ts_x) & ((np.isfinite(ts["n_e"]) & ts["valid_n_e"])
                                        | (np.isfinite(ts["t_e"]) & ts["valid_t_e"]))
        if measured.sum() >= 2:
            span_line = (float(np.min(ts_x[measured])), float(np.max(ts_x[measured])))

    def spans(first: bool) -> tuple[ReferenceLine, ...]:
        if span_line is None:
            return ()
        return (ReferenceLine(x=span_line[0], x_end=span_line[1], label="TS span" if first else ""),)

    selection = {
        "time": t_core, "core_profiles_slice": j, "equilibrium_time": t_eq, "equilibrium_slice": i,
        "equilibrium_lineage": lineage, "equilibrium_occurrence": occurrence,
        "equilibrium_occurrence_verified": occurrence_verified,
        "lineage_record": lineage_path,
        "thomson_time": None if ts is None else ts["time"],
        "thomson_channels": [] if ts is None or not len(ts["rows"]) else [int(c) for c in ts["channel"]],
        "thomson_skipped": [] if ts is None else [list(item) for item in ts["skipped"]],
        "thomson_note": ts_note,
        "thomson_span": None if span_line is None else list(span_line),
        "t_i_finite_range": valid_range, "coordinate": coordinate,
    }
    label = COORDINATE_LABELS[coordinate]

    def panel(series, y_label, unit, name, first=False, display_spec=None):
        return Profile1D(
            series=tuple(series), coordinate_label=label, y_label=y_label, y_unit=unit,
            x_limits=(0.0, 1.0), display=display_spec, reference_lines=spans(first),
            metadata={"panel": name, "traces": [t for t in traces if t["panel"] == name], **selection},
        )

    models = (
        panel(pressure, "$p$", "Pa", "pressure", first=True),
        panel(density, "$n$", display.unit, "density", display_spec=display),
        panel(temperature, "$T$", "eV", "temperature"),
        panel(charge, r"$Z_{\mathrm{eff}}$", "", "zeff"),
    )
    # One line: a stacked layout needs every inch of its height for the panels.
    if occurrence is None:
        occurrence_text = ""
    else:
        occurrence_text = f" #{occurrence}" + ("" if occurrence_verified else " (unverified)")
    eq_text = f"{lineage or 'lineage not recorded'}{occurrence_text} at {t_eq * 1e3:.1f} ms"
    if ts_note:
        ts_text = f", {ts_note}"
    elif ts is None or ts["time"] is None:
        ts_text = ""
    elif isinstance(ts["time"], tuple):
        ts_text = f", TS {ts['time'][0] * 1e3:.1f}-{ts['time'][1] * 1e3:.1f} ms"
    else:
        ts_text = f", TS {ts['time'] * 1e3:.1f} ms"
    grid = layout == "grid"
    # Legends beside the panels (a stack in two columns); the renderer moves
    # them inside for a format too narrow for that.  Underscored: internal
    # plumbing to the Profile1D renderer, not caller options.
    legends = ({"_legend_loc": "outside", "_legend_fontsize": "x-small"} if grid else
               {"_legend_loc": "outside", "_legend_ncols": 2, "_legend_fontsize": "x-small"},) * len(models)
    return Panels(models=models, nrows=2 if grid else 4, ncols=2 if grid else 1, share_x=True,
                  member_styles=legends,
                  suptitle=f"Kinetic state at {t_core * 1e3:.1f} ms ({eq_text}{ts_text})")


def _equilibrium_psi_norm(ods: Any, eq_base: str) -> np.ndarray | None:
    psi = array(ods, f"{eq_base}.profiles_1d.psi")
    axis = _scalar(get(ods, f"{eq_base}.global_quantities.psi_axis"))
    edge = _scalar(get(ods, f"{eq_base}.global_quantities.psi_boundary"))
    if psi is None or axis is None or edge is None or edge == axis:
        return None
    return (psi - axis) / (edge - axis)


def _z_n(ods: Any, prefix: str) -> float | None:
    return _scalar(get(ods, f"{prefix}.element.0.z_n"))


def _main_ion_path(ods: Any, base: str) -> str | None:
    """The one hydrogenic ``ion[]`` entry (``element[0].z_n == 1``), or ``None``."""
    for k in range(count(ods, f"{base}.ion")):
        z_n = _z_n(ods, f"{base}.ion.{k}")
        if z_n is not None and round(z_n) == 1:
            return f"{base}.ion.{k}"
    return None


def _ion_densities(ods: Any, base: str, size: int):
    """``(main, impurity)``: each ``(density, labels, paths, role)`` or ``None``.

    Species are classified by the nuclear charge of their first element,
    ``element[0].z_n``: 1 is the hydrogenic main ion (several are summed), more
    is an impurity; an entry without it is in neither, its kind not being
    stated.  A molecular or multi-element ion is classified by that first
    element alone.  An impurity's *charged* density is the sum of its
    ``state[].density`` when the states are stored (the bundled writer's
    ``density`` counts the neutral atoms too), else its ``density`` as stored.
    """
    groups: dict[str, list] = {"main": [], "impurity": []}
    for k in range(count(ods, f"{base}.ion")):
        prefix = f"{base}.ion.{k}"
        z_n = _z_n(ods, prefix)
        if z_n is None:
            continue
        hydrogenic = round(z_n) == 1
        density, path = array(ods, f"{prefix}.density"), f"{prefix}.density"
        if not hydrogenic:
            states = [array(ods, f"{prefix}.state.{q}.density") for q in range(count(ods, f"{prefix}.state"))]
            if states and all(s is not None and s.size == size for s in states):
                density, path = np.sum(states, axis=0), f"{prefix}.state[:].density"
        if density is None or density.size != size:
            continue
        record = f"{prefix}.density_fit.parameters"
        role = _origin_role(record, _record(ods, record))
        label = _plain(get(ods, f"{prefix}.label", "") or f"ion {k}") or f"ion {k}"
        groups["main" if hydrogenic else "impurity"].append((density, label, path, role))
    out = []
    for name in ("main", "impurity"):
        items = groups[name]
        if not items:
            out.append(None)
            continue
        roles = sorted({item[3] for item in items})
        out.append((np.sum([item[0] for item in items], axis=0), [item[1] for item in items],
                    [item[2] for item in items], "/".join(roles)))
    return tuple(out)


def _zeff_reference(ods: Any, t_core: float, record_path: str, record_text: str | None, zeff_role: str):
    """``(value, label, path, role)`` of the Z_eff reference, or ``None``.

    The stored resistive ``Z_eff`` at the slice's own entry of
    ``core_profiles.time`` -- only when that array is as long as the time base
    and the entry is the slice's time -- else the ``target_zeff=`` field of the
    ``zeff`` record, whose role is the record's.
    """
    stored = array(ods, "core_profiles.global_quantities.z_eff_resistive")
    times = array(ods, "core_profiles.time")
    if stored is not None and times is not None and stored.size == times.size and times.size:
        k = _closest(times, t_core)
        value = float(stored.reshape(-1)[k])
        if abs(float(times[k]) - t_core) <= GLOBAL_TIME_TOLERANCE and np.isfinite(value):
            return (value, rf"$Z_{{\mathrm{{eff}}}}^{{\mathrm{{res}}}} = {value:.3g}$ (stored)",
                    "core_profiles.global_quantities.z_eff_resistive", "stored")
    target = _target_zeff(record_path, record_text)
    if target is None:
        return None
    return (target, rf"target $\langle Z_{{\mathrm{{eff}}}}\rangle = {target:g}$ ({zeff_role})",
            f"{record_path} target_zeff", zeff_role)
