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
  the ``zeff_fit.parameters`` record.  A ``resistive_method=`` field of that
  record names the estimate in the label (Romero, #781).
* **Impurity pressure** ``p_imp`` is the sum of the stored ``ion[k].pressure``
  of every impurity (``element[0].z_n > 1``), drawn as the *part of* ``p_i``
  it is: ``pressure_ion_total`` is the DD's sum over all ion species, so
  ``p_e + p_i`` already holds it and it is never added a second time.  Without
  a stored impurity pressure it is omitted and the legend says so.
* **Assumptions** are read from the records, never hard-coded: an
  ``assumption=...`` field of any record the plot reads, else a record whose
  ``origin=`` or ``status=`` is ``assumed`` (named by its quantity and
  ``method=``, or ``T_i = r T_e`` for a ratio record), are listed in one
  figure note, ``Assumed: T_i = T_e; n_C = n_O``.  Nothing is listed when no
  record states an assumption.

Inference stages (#1957)
------------------------
The maintainer's chain -- equilibrium and Thomson in, ``T_i = T_e`` and
``n_C = n_O`` assumed, the Romero ``Z_eff`` first, then ``n_I,eff``, ``n_i``
and ``Z_eff(rho)`` -- fixes the stage of every trace by its *quantity* alone
(:data:`QUANTITY_STAGES`); the data never does.  ``stage=k`` cuts the figure
at stage ``k``: only quantities of stage ``<= k`` are drawn, every one as in
the full figure (no stage is faded), every panel on the limits of the full
figure so the data scale does not jump, each legend naming only what the
stage draws (the layout may shift between stages); the newest stage's traces are
marked ``emphasis="newest"`` in the trace records.  ``stage=None`` (the
default) is the full figure.  ``layout="stages"`` puts the cumulative figure
at several stages side by side, one column per stage.  A procedure is shown as
one image per stage by a loop over ``stage=k`` (each ``savefig``'d), and
``animation=True`` walks the same stages as one movie.
The electron pressure profile ``p_e`` is a closure input (stage 8), not a fit
product: the Thomson pressure points ``e n_e T_e`` are the stage-1 data.  The
stage names are neutral (``T_i``, not ``T_i = T_e``): what a stage assumed is
the records' to say, in the note and the legend roles, so a slice built in
another order is drawn as stored and never claimed to follow this one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from vaft.plot.backend.access import array, count, get
from vaft.plot.backend.kinetic_overview import _closest, _core_coordinate, _equilibrium_time_axis
from vaft.plot.display import COORDINATE_LABELS, resolve_display
from vaft.plot.intent import palette
from vaft.plot.models import Panels, Profile1D, Series, TextPanel

__all__ = []  # The registered plot is public; the extraction is not.

#: ``coordinate=`` of the view: the toroidal-flux radius the core grid is
#: stored on, or the normalized poloidal flux.
COORDINATES = ("rho_tor_norm", "psi_norm")
#: ``layout=``: a 2 x 2 grid or a 4 x 1 stack of the same four panels, or the
#: four panels as rows with one column per inference stage (``stages=``).
LAYOUTS = ("grid", "stack", "stages")

#: The inference stages (#1957), in order: ``(number, name)``.  The names are
#: plain text, for a control, a caption and a column title alike.
STAGES = (
    (1, "Thomson and equilibrium"),
    (2, "n_e, T_e fits"),
    (3, "T_i"),
    (4, "Z_eff reference"),
    (5, "impurity density"),
    (6, "main-ion density"),
    (7, "Z_eff profile"),
    (8, "pressure closure"),
)
STAGE_NAMES = dict(STAGES)
#: The stage each quantity belongs to; a trace's stage is its quantity's.
QUANTITY_STAGES = {
    "p_eq": 1, "p_e_thomson": 1, "n_e_thomson": 1, "T_e_thomson": 1,
    "n_e": 2, "T_e": 2,
    "T_i": 3,
    "Z_eff_reference": 4,
    "n_impurity_ion": 5,
    "n_main_ion": 6,
    "Z_eff": 7,
    "p_e": 8, "p_i": 8, "p_imp": 8, "p_kin": 8,
}
#: The stage names as the figure writes them (mathtext), in a caption ...
#: Every one carries a subscripted symbol, so the caption line has one height
#: whatever the stage and a sequence of stages keeps one layout.
_STAGE_CAPTIONS = {
    1: r"Thomson $n_e$, $T_e$ and $p_{\mathrm{eq}}$", 2: r"$n_e$, $T_e$ fits", 3: r"$T_i$",
    4: r"$Z_{\mathrm{eff}}$ reference", 5: r"impurity density $n_{\mathrm{imp}}$", 6: r"main-ion density $n_i$",
    7: r"$Z_{\mathrm{eff}}$ profile", 8: r"pressure closure $p_e + p_i$ vs $p_{\mathrm{eq}}$",
}
#: ... and as a column title of ``layout="stages"``, where a column is narrow.
_STAGE_TITLES = {
    1: "data", 2: "fits", 3: r"$T_i$", 4: r"$Z_{\mathrm{eff}}$ ref.", 5: r"$n_I$", 6: r"$n_i$",
    7: r"$Z_{\mathrm{eff}}(\rho)$", 8: "closure",
}
#: The symbol each quantity is named by in the compact legends of
#: ``layout="stages"``; the roles are the single figure's to state.
_SYMBOLS = {
    "p_eq": r"$p_{\mathrm{eq}}$", "p_e_thomson": r"$p_e^{\mathrm{TS}}$", "n_e_thomson": r"$n_e^{\mathrm{TS}}$",
    "T_e_thomson": r"$T_e^{\mathrm{TS}}$", "n_e": r"$n_e$", "T_e": r"$T_e$", "T_i": r"$T_i$",
    "Z_eff_reference": r"$Z_{\mathrm{eff}}^{\mathrm{ref}}$", "n_impurity_ion": r"$n_{\mathrm{imp}}^{+}$",
    "n_main_ion": r"$n_i$", "Z_eff": r"$Z_{\mathrm{eff}}$", "p_e": r"$p_e$", "p_i": r"$p_i$",
    "p_imp": r"$p_{\mathrm{imp}}$", "p_kin": r"$p_e + p_i$",
}
#: The columns ``layout="stages"`` draws without ``stages=``: data, fits, the
#: Romero reference, the impurity it fixes, the Z_eff profile and the closure --
#: six columns, what an 11-inch slide holds legibly.  ``stages=range(1, 9)``
#: draws all eight.
DEFAULT_STAGE_COLUMNS = (1, 2, 4, 5, 7, 8)
#: Thomson points: marker size as a fraction of the format's, so the error
#: bars stand out of the markers; bars in the points' colour, capped.
THOMSON_POINT_STYLE = {"markersize_scale": 0.4, "elinewidth": 1.3, "capsize": 2.5, "capthick": 1.3,
                       "zorder": 4}
#: How far apart two samples may be [s] when the time base they are matched on
#: has a single sample, so no step to compare with -- the composition
#: resolver's own pairing tolerance.
SINGLE_SLICE_TOLERANCE = 5e-4
#: How closely ``core_profiles.time`` must hold the slice's own time for a
#: global quantity at that index to belong to it [s].
GLOBAL_TIME_TOLERANCE = 1e-6

#: Legend type as a fraction of the format's base size.
LEGEND_FONTSIZE_SCALE = 0.58


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
    origin = fields.get("origin", "").lower()
    if origin in ("assumed", "derived"):
        # ``origin=assumed; method=ti_equals_te`` (#1957): an assumption by the
        # composition grammar, which classify_ti_record does not read.
        return "assumed"
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


def _stage_option(value: Any, name: str = "stage") -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) not in STAGE_NAMES:
        raise ValueError(f"{name} must be an inference stage, one of {', '.join(map(str, STAGE_NAMES))}; "
                         f"got {value!r}")
    return int(value)


def _stage_columns(value: Any, layout: str, stage: int | None) -> tuple[int, ...] | None:
    """The stages ``layout="stages"`` draws as columns, validated; ``None`` for another layout."""
    if layout != "stages":
        if value is not None:
            raise ValueError("stages= names the columns of layout='stages'; pass layout='stages' or stage= to "
                             "cut one figure")
        return None
    if stage is not None:
        raise ValueError("layout='stages' draws one column per stage; give stages=(...), not stage=")
    if value is None:
        return DEFAULT_STAGE_COLUMNS
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"stages= takes a sequence of stages, e.g. (1, 4, 8); got {value!r}")
    chosen = tuple(_stage_option(v, "each of stages=") for v in value)
    if not chosen or any(b <= a for a, b in zip(chosen, chosen[1:])):
        raise ValueError(f"stages= must hold distinct stages in increasing order; got {tuple(value)!r}")
    return chosen


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
    is left empty.  A sample whose value or stored σ is not finite is invalid,
    the convention the Thomson mapping uses (``equilibrium_mapping_points``
    drops such channels); the DD carries no validity flag on Thomson
    ``n_e``/``t_e``.  Returns ``None`` without channels.
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
                        valid = bool(np.isfinite(value)) and (sigma is None or bool(np.isfinite(error)))
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
                        layout: str = "grid", coordinate: str = "rho_tor_norm", stage: Any = None,
                        stages: Any = None, **options: Any) -> Panels:
    """Four matched panels of one stored kinetic state; refuse what does not match."""
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {', '.join(COORDINATES)}; got {coordinate!r}")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}; got {layout!r}")
    stage = _stage_option(stage)
    columns = _stage_columns(stages, layout, stage)
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
    # The DD keeps one fit record for the electron density, ``density_fit``,
    # whichever density leaf carries the values.
    ne_record = f"{base}.electrons.density_fit.parameters"
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
    p_imp, p_imp_paths, p_imp_reason = _impurity_pressure(ods, base, size)
    # Every record the plot reads, by the quantity it belongs to: the
    # assumptions note lists what they state (#1957).
    records = [("n_e", ne_record, ne_text), ("T_e", te_record, te_text), ("T_i", ti_path, ti_text),
               ("Z_eff", zeff_record, zeff_text)]
    for quantity, group in (("n_main_ion", main), ("n_impurity_ion", impurity)):
        if group is not None:
            records.extend((quantity, path, text) for path, text in group[4])

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
        points = _series(ts_x[keep][order], (np.asarray(values)[keep] * scale)[order], label, colour,
                         yerr=err if np.isfinite(err).any() else None, marker=marker,
                         valid=np.asarray(valid, dtype=bool)[keep][order])
        return replace(points, style={**points.style, **THOMSON_POINT_STYLE})

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
                       "path": path, "role": role, "sigma": sigma, "stage": QUANTITY_STAGES[quantity]})

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
    if p_imp is not None:
        # A part of pressure_ion_total (the DD's sum over every ion species),
        # drawn as such and never added to p_e + p_i a second time.
        add(pressure, _series(x, _finite(p_imp), r"$p_{\mathrm{imp}}$ (stored, part of $p_i$)", _COLOURS["impurity"],
                              linestyle="-."),
            "p_imp", " + ".join(p_imp_paths), "stored", "none")
    elif p_imp_reason is not None:
        # Short: the legend sits beside the panel and its width is the panel's loss.
        add(pressure, _note(r"$p_{\mathrm{imp}}$ not stored"),
            "p_imp", " + ".join(p_imp_paths), "unavailable", "none")
    if pi is not None:
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
        values, labels, paths, role, _ = main
        add(density, _series(x, _finite(values) * scale, rf"$n_i$ {', '.join(labels)} ({role})", _COLOURS["ion"]),
            "n_main_ion", " + ".join(paths), role, "none")
    if impurity is not None:
        values, labels, paths, role, _ = impurity
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
        add(temperature, _series(x, ti, rf"$T_i$ ({ti_role}{note})", _COLOURS["ion"], linestyle="--"),
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

    assumptions = _assumptions(records, None if reference is None or reference[3] != "stored" else reference[0])
    selection = {
        "time": t_core, "core_profiles_slice": j, "equilibrium_time": t_eq, "equilibrium_slice": i,
        "equilibrium_lineage": lineage, "equilibrium_occurrence": occurrence,
        "equilibrium_occurrence_verified": occurrence_verified,
        "lineage_record": lineage_path,
        "thomson_time": None if ts is None else ts["time"],
        "thomson_channels": [] if ts is None or not len(ts["rows"]) else [int(c) for c in ts["channel"]],
        "thomson_skipped": [] if ts is None else [list(item) for item in ts["skipped"]],
        "thomson_note": ts_note,
        "t_i_finite_range": valid_range, "coordinate": coordinate,
        "assumptions": [{"quantity": q, "record": path, "statement": text} for q, path, text in assumptions],
    }
    label = COORDINATE_LABELS[coordinate]
    panels = (
        ("pressure", pressure, "$p$", "Pa", None),
        ("density", density, "$n$", display.unit, display),
        ("temperature", temperature, "$T$", "eV", None),
        ("zeff", charge, r"$Z_{\mathrm{eff}}$", "", None),
    )
    cut = stage is not None or columns is not None
    # The full figure's limits, kept by every cut of it so stages do not jump.
    limits = {name: _y_limits(series) for name, series, *_ in panels} if cut else {}

    def panel(name, series, y_label, unit, display_spec, at, *, title="", extra=None, compact=False):
        records_ = [t for t in traces if t["panel"] == name]
        if at is not None:
            kept = [(s, dict(t)) for s, t in zip(series, records_) if t["stage"] <= at]
            series = [s for s, _ in kept]
            records_ = [t for _, t in kept]
            for t in records_:
                t["emphasis"] = "newest" if t["stage"] == at else "earlier"
        if compact:
            # A narrow column: symbols in the legend, the bare symbol on the
            # axis; a stated absence is the single figure's to name.
            kept = [(s, t) for s, t in zip(series, records_) if t["role"] != "unavailable"]
            series = [replace(s, label=_SYMBOLS[t["quantity"]]) for s, t in kept]
            records_ = [t for _, t in kept]
            for s, t in zip(series, records_):
                t["label"] = s.label
        return Profile1D(
            series=tuple(series), coordinate_label=_symbol(label) if compact else label, y_label=y_label,
            y_unit=unit, title=title,
            x_limits=(0.0, 1.0), y_limits=limits.get(name), display=display_spec,
            metadata={"panel": name, "traces": records_, **selection, "stage": at,
                      "stage_name": STAGE_NAMES.get(at) if at is not None else None, **(extra or {})},
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
    suptitle = f"Kinetic state at {t_core * 1e3:.1f} ms ({eq_text}{ts_text})"
    # Beside the panels, a stack's in two columns; the renderer moves them
    # inside for a format too narrow for that.
    beside = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0), "borderaxespad": 0.0,
              "fontsize_scale": LEGEND_FONTSIZE_SCALE, "labelspacing": 0.3, "handlelength": 1.5,
              "handletextpad": 0.5, "borderpad": 0.3}

    if columns is not None:
        # Rows are the four quantities, columns the cumulative figure at each
        # stage; the rightmost column's legend speaks for its row.
        shown = max(columns)
        caption = _caption(None, [a for a in assumptions if QUANTITY_STAGES[a[0]] <= shown], ())
        models, styles = [], []
        for row, (name, series, y_label, unit, display_spec) in enumerate(panels):
            for col, at in enumerate(columns):
                title = f"{at}: {_STAGE_TITLES[at]}" if row == 0 else ""
                models.append(panel(name, series, y_label, unit, display_spec, at, title=title,
                                    extra={"layout": "stages", "column": col}, compact=True))
                styles.append({"legend": False})
            # A cell of its own for the row's legend, which the renderer fills
            # from the row's last column: inside an axes, the layout makes room
            # for it, where a legend hung beside the column would squeeze the rows.
            models.append(TextPanel(lines=(), title=""))
            styles.append({})
        return Panels(models=tuple(models), nrows=4, ncols=len(columns) + 1, share_x=True,
                      member_styles=tuple(styles), suptitle=suptitle + caption)

    caption = _caption(
        stage, [a for a in assumptions if stage is None or QUANTITY_STAGES[a[0]] <= stage],
        [t for t in traces if stage is not None and t["stage"] == stage],
    )
    models = tuple(panel(name, series, y_label, unit, display_spec, stage)
                   for name, series, y_label, unit, display_spec in panels)
    grid = layout == "grid"
    legends = tuple(_legend_style(model, beside if grid else {**beside, "ncol": 2}) for model in models)
    return Panels(models=models, nrows=2 if grid else 4, ncols=2 if grid else 1, share_x=True,
                  member_styles=legends, suptitle=suptitle + caption)


def _legend_style(model: Profile1D, placement: Mapping[str, Any]) -> dict[str, Any]:
    """A panel's legend: drawn even for one trace -- its role is the point -- and absent when empty."""
    style: dict[str, Any] = {"legend_placement": dict(placement)}
    if model.series:
        style["legend"] = True
    return style


def _symbol(label: str) -> str:
    """The mathtext symbol of a coordinate label: ``Normalized Toroidal Flux $\\rho_N$`` -> ``$\\rho_N$``."""
    found = re.findall(r"\$[^$]+\$", label)
    return found[-1] if found else label


def _y_limits(series: list[Series]) -> tuple[float, float] | None:
    """The y range Matplotlib would autoscale every series (and error bar) of a panel to."""
    low, high = np.inf, -np.inf
    for item in series:
        y = np.asarray(item.y, dtype=float)
        parts = [y]
        if item.yerr is not None:
            err = np.asarray(item.yerr, dtype=float)
            lo, hi = (err[0], err[1]) if err.ndim == 2 else (err, err)
            parts += [y - lo, y + hi]
        values = np.concatenate([np.ravel(v) for v in parts])
        values = values[np.isfinite(values)]
        if values.size:
            low, high = min(low, float(values.min())), max(high, float(values.max()))
    if not np.isfinite(low):
        return None
    margin = 0.05 * (high - low) if high > low else max(abs(high) * 0.05, 1e-12)
    return (low - margin, high + margin)


def _statement(quantity: str, fields: Mapping[str, str]) -> str | None:
    """What one record says was assumed, from its own fields; ``None`` when nothing."""
    if "assumption" in fields:
        return re.sub(r"\s*=\s*", " = ", fields["assumption"])
    origin, status = fields.get("origin", "").lower(), fields.get("status", "").lower()
    if origin != "assumed" and status != "assumed":
        return None
    if "ti_te_ratio" in fields:
        return f"T_i = {fields['ti_te_ratio']} T_e"
    method = fields.get("method")
    return f"{quantity} ({method})" if method else quantity


def _zeff_constraint(records: list, resistive: float | None) -> tuple[str, str, str] | None:
    """The total ``Z_eff`` a composition was partitioned to, as the records state it, or ``None``.

    ``normalization=resistive_closure`` names a resistive value: the record's
    ``resistive_target=``, else the stored ``z_eff_resistive`` of the slice,
    named by ``resistive_method=`` (``<Z_eff> = 2 (Romero)``).  Any other
    normalization with a ``target_zeff=`` on a record names that target
    (``<Z_eff> = 2 (target)``).  Attributed to the composition record that
    states the normalization.
    """
    fields = [(quantity, path, _fields(path, text)) for quantity, path, text in records if text is not None]
    normalized = [(q, path, f) for q, path, f in fields if "normalization" in f]
    if not normalized:
        return None
    quantity, path, first = min(normalized, key=lambda item: QUANTITY_STAGES[item[0]])
    pooled = {key: value for _, _, f in fields for key, value in f.items()}
    if first["normalization"] == "resistive_closure":
        value = pooled.get("resistive_target")
        if value is None and resistive is not None:
            value = f"{resistive:g}"
        if value is None:
            return None
        method = pooled.get("resistive_method", "resistive")
        return quantity, path, f"<Z_eff> = {value} ({method[:1].upper() + method[1:]})"
    if "target_zeff" in pooled:
        return quantity, path, f"<Z_eff> = {pooled['target_zeff']} (target)"
    return None


def _assumptions(records: list[tuple[str, str, str | None]], resistive: float | None = None
                 ) -> list[tuple[str, str, str]]:
    """``(quantity, record path, statement)`` for every record stating an assumption, once per statement.

    A statement several records make is kept with the earliest stage's
    quantity, so a stage cut names it from the stage that first relies on it.
    The total ``Z_eff`` the composition was partitioned to
    (:func:`_zeff_constraint`) is listed with them.
    """
    out, seen = [], set()
    for quantity, path, text in sorted(records, key=lambda item: QUANTITY_STAGES[item[0]]):
        if text is None:
            continue
        statement = _statement(quantity, _fields(path, text))
        if statement is not None and statement not in seen:
            seen.add(statement)
            out.append((quantity, path, statement))
    constraint = _zeff_constraint(records, resistive)
    if constraint is not None:
        out.append(constraint)
    return out


#: A relation of symbols such as ``T_i = T_e`` or ``T_i = 0.5 T_e``, which the
#: caption may set in mathtext; anything else stays plain text.
_RELATION = re.compile(r"[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)?(\s*[=<>]\s*([0-9.]+\s*)?[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)?)+")


_ZEFF_CONSTRAINT = re.compile(r"<Z_eff> = ([^ ]+)( \(.*\))?")


def _math(statement: str) -> str:
    """``T_i = T_e`` -> ``$T_i = T_e$``; a statement in no such form is left as it is."""
    total = _ZEFF_CONSTRAINT.fullmatch(statement.strip())
    if total:
        method = f" {_plain(total.group(2))}" if total.group(2) else ""
        return rf"$\langle Z_{{\mathrm{{eff}}}}\rangle = {total.group(1)}$" + method
    return f"${statement}$" if _RELATION.fullmatch(statement.strip()) else _plain(statement)


def _caption(stage: int | None, assumptions: list, newest: list) -> str:
    """The figure note under the title: the stage shown and what the records assumed."""
    parts = []
    if stage is not None:
        text = f"Stage {stage}/{len(STAGES)}: {_STAGE_CAPTIONS[stage]}"
        if not any(t["role"] != "unavailable" for t in newest):
            text += " (nothing stored)"
        parts.append(text)
    if assumptions:
        parts.append("Assumed: " + "; ".join(_math(statement) for _, _, statement in assumptions))
    return "".join("\n" + part for part in parts)


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
    """``(main, impurity)``: each ``(density, labels, paths, role, records)`` or ``None``.

    Species are classified by the nuclear charge of their first element,
    ``element[0].z_n``: 1 is the hydrogenic main ion (several are summed), more
    is an impurity; an entry without it is in neither, its kind not being
    stated.  A molecular or multi-element ion is classified by that first
    element alone.  An impurity's *charged* density is the sum of its
    ``state[].density`` when the states are stored (the bundled writer's
    ``density`` counts the neutral atoms too), else its ``density`` as stored.
    ``records`` are the ``(path, text)`` of each species' density record.
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
        text = _record(ods, record)
        role = _origin_role(record, text)
        label = _plain(get(ods, f"{prefix}.label", "") or f"ion {k}") or f"ion {k}"
        groups["main" if hydrogenic else "impurity"].append((density, label, path, role, (record, text)))
    out = []
    for name in ("main", "impurity"):
        items = groups[name]
        if not items:
            out.append(None)
            continue
        roles = sorted({item[3] for item in items})
        out.append((np.sum([item[0] for item in items], axis=0), [item[1] for item in items],
                    [item[2] for item in items], "/".join(roles), [item[4] for item in items]))
    return tuple(out)


def _impurity_pressure(ods: Any, base: str, size: int):
    """``(p_imp, paths, reason)``: the summed stored impurity ``ion[k].pressure``.

    ``p_imp`` is ``None`` when any impurity (``element[0].z_n > 1``) stores no
    pressure on the grid -- then ``reason`` says so -- or when the slice holds
    no impurity at all (``reason`` ``None``: nothing to omit).
    """
    values, paths, missing = [], [], []
    for k in range(count(ods, f"{base}.ion")):
        prefix = f"{base}.ion.{k}"
        z_n = _z_n(ods, prefix)
        if z_n is None or round(z_n) <= 1:
            continue
        path = f"{prefix}.pressure"
        pressure = array(ods, path)
        paths.append(path)
        if pressure is None or pressure.size != size:
            missing.append(k)
        else:
            values.append(pressure)
    if not paths:
        return None, [], None
    if missing:
        return None, paths, "no impurity ion[k].pressure stored"
    return np.sum(values, axis=0), paths, None


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
            fields = _fields(record_path, record_text) if record_text is not None else {}
            method = _plain(fields.get("resistive_method", ""))
            how = f"stored, {method}" if method else "stored"
            return (value, rf"$Z_{{\mathrm{{eff}}}}^{{\mathrm{{res}}}} = {value:.3g}$ ({how})",
                    "core_profiles.global_quantities.z_eff_resistive", "stored")
    target = _target_zeff(record_path, record_text)
    if target is None:
        return None
    return (target, rf"target $\langle Z_{{\mathrm{{eff}}}}\rangle = {target:g}$ ({zeff_role})",
            f"{record_path} target_zeff", zeff_role)
