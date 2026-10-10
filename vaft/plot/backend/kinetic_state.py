"""One matched kinetic state in four panels, from stored IMAS paths only (issue #1837).

``kinetic_overview_state`` draws the pressure, density, temperature and
``Z_eff`` profiles of **one** selected kinetic state: a ``core_profiles``
slice, the equilibrium slice of the same time and the Thomson channels mapped
through that equilibrium.  It reads what a pipeline stage stored and does no
physics of its own -- no EFIT, no Thomson fit, no OpenADAS charge states, no
composition and no ``T_i`` inference.  The only arithmetic is presentation:
``p_kin = p_e + p_i`` for display, ``e n_e T_e`` when no electron pressure is
stored (labelled *derived*), and the Thomson electron pressure ``e n_e T_e``
per channel with its 1 sigma from the measured ``n_e`` and ``T_e`` errors,
assumed independent.

The evidence roles and the ``T_i`` validity mask come from the JSON record the
stage writes on ``core_profiles.code.parameters`` under ``kinetic_state``
(the path contract on #1837)::

    {"kinetic_state": {"schema": 1,
                       "equilibrium": {"lineage": ..., "occurrence": ...},
                       "roles": {quantity: role},
                       "t_i_valid": [bool on the core grid],
                       "composition": {"preset": ..., "target_zeff": ...}}}

A quantity whose role is not recorded is labelled ``stored`` (a Thomson
channel ``measured``); nothing is guessed.
"""

from __future__ import annotations

import json
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
#: The ``kinetic_state`` record schema this view reads.
SCHEMA = 1
#: How far apart a lone equilibrium slice and the core_profiles slice may be
#: [s] when the equilibrium has no step to compare with -- the composition
#: resolver's own pairing tolerance.
SINGLE_SLICE_TOLERANCE = 5e-4

#: Elementary charge [C].
_E = 1.602176634e-19

#: Recorded role -> legend wording.  An unlisted role is shown as recorded.
ROLE_TEXT = {
    "measurement": "measured", "measured": "measured",
    "fit": "fit", "fitted": "fit",
    "fit input": "fit input", "fit_input": "fit input",
    "assumed": "assumed", "inferred": "inferred", "derived": "derived",
    "reconstruction": "reconstruction",
    "closure identity": "closure identity", "closure_identity": "closure identity",
    "independent validation": "independent validation",
    "independent_validation": "independent validation",
    "invalid": "invalid", "stored": "stored",
}

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


def kinetic_state_record(ods: Any) -> dict | None:
    """The ``kinetic_state`` record on ``core_profiles.code.parameters``, or ``None``.

    ``code.parameters`` reaches a reader as the JSON text a stage wrote or, after
    some loaders, as a decoded mapping; both are read.  Text that is not JSON
    (provenance lines, XML) carries no record.  A record of another schema is
    refused rather than half-read.
    """
    raw = get(ods, "core_profiles.code.parameters")
    if raw is None:
        return None
    payload: Any = None
    if isinstance(raw, Mapping) or hasattr(raw, "keys"):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return None
    try:
        record = payload.get("kinetic_state") if hasattr(payload, "get") else None
    except Exception:  # noqa: BLE001 -- a foreign tree is simply not this record
        return None
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError("core_profiles.code.parameters kinetic_state is not a JSON object")
    if record.get("schema") != SCHEMA:
        raise ValueError(
            f"core_profiles.code.parameters kinetic_state has schema {record.get('schema')!r}; "
            f"this view reads schema {SCHEMA}"
        )
    return dict(record)


def _role(roles: Mapping[str, Any], key: str, default: str = "stored") -> str:
    value = roles.get(key)
    if value is None:
        return default
    return ROLE_TEXT.get(str(value).strip().lower(), str(value))


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


def _equilibrium_step(axis: np.ndarray) -> float:
    finite = np.unique(axis[np.isfinite(axis)])
    if finite.size < 2:
        return SINGLE_SLICE_TOLERANCE
    return float(np.median(np.diff(finite)))


def _masked(values: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    out = np.asarray(values, dtype=float).copy()
    if mask is not None:
        out[~mask] = np.nan
    return out


def _sigma(ods: Any, path: str, size: int) -> np.ndarray | None:
    values = array(ods, path)
    if values is None or values.size != size or not np.isfinite(values).any():
        return None
    return values


def _thomson(ods: Any, target: float) -> dict | None:
    """Every usable Thomson channel at the sample nearest ``target``."""
    total = count(ods, "thomson_scattering.channel")
    if not total:
        return None
    index = _closest(array(ods, "thomson_scattering.time"), target)
    times = array(ods, "thomson_scattering.time")
    when = float(times[min(index, times.size - 1)]) if times is not None and times.size else None

    def pick(path: str) -> float:
        values = array(ods, path)
        if values is None or not values.size:
            return np.nan
        return float(values.reshape(-1)[min(index, values.size - 1)])

    rows = []
    for c in range(total):
        prefix = f"thomson_scattering.channel.{c}"
        r, z = _scalar(get(ods, prefix + ".position.r")), _scalar(get(ods, prefix + ".position.z"))
        if r is None or z is None:
            continue
        rows.append((c, r, z, pick(prefix + ".n_e.data"), pick(prefix + ".n_e.data_error_upper"),
                     pick(prefix + ".t_e.data"), pick(prefix + ".t_e.data_error_upper")))
    if not rows:
        return None
    table = np.asarray(rows, dtype=float)
    return {"index": table[:, 0].astype(int), "r": table[:, 1], "z": table[:, 2],
            "n_e": table[:, 3], "sigma_n_e": table[:, 4], "t_e": table[:, 5], "sigma_t_e": table[:, 6],
            "time": when}


def _series(x, y, label, colour, *, yerr=None, linestyle="-", marker=None) -> Series | None:
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        return None
    style: dict[str, Any] = {"color": colour, "linestyle": linestyle}
    if marker is not None:
        style.update(marker=marker, linestyle="none")
    return Series(x=x, y=y, label=label, yerr=yerr, style=style)


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

    # --- the state: core_profiles slice and the equilibrium of the same time --
    if not count(ods, "core_profiles.profiles_1d"):
        raise ValueError("kinetic_overview_state needs a core_profiles.profiles_1d slice")
    core_times = _core_times(ods)
    j = _closest(core_times, time) if time is not None else 0
    t_core = None if core_times is None else float(core_times[j])
    if t_core is not None and not np.isfinite(t_core):
        t_core = None
    eq_axis = _equilibrium_time_axis(ods)
    if not count(ods, "equilibrium.time_slice") or eq_axis is None:
        raise ValueError("kinetic_overview_state needs an equilibrium slice with a time to match the "
                         "core_profiles slice to")
    if t_core is None:
        raise ValueError(f"core_profiles.profiles_1d.{j} stores no time; the equilibrium cannot be matched to it")
    i = _closest(eq_axis, t_core)
    t_eq = float(eq_axis[i])
    step = _equilibrium_step(eq_axis)
    if not np.isfinite(t_eq) or abs(t_eq - t_core) > step + 1e-12:
        raise ValueError(
            f"the nearest equilibrium slice (t = {t_eq:.6g} s) is {abs(t_eq - t_core) * 1e3:.3g} ms from the "
            f"core_profiles slice (t = {t_core:.6g} s), more than one equilibrium step ({step * 1e3:.3g} ms); "
            "the kinetic state and the equilibrium would not be one state"
        )

    record = kinetic_state_record(ods) or {}
    roles = record.get("roles") or {}
    if not isinstance(roles, Mapping):
        raise ValueError("kinetic_state roles must be a {quantity: role} object")
    recorded_eq = record.get("equilibrium") or {}
    lineage = recorded_eq.get("lineage")
    recorded_occurrence = recorded_eq.get("occurrence")
    if equilibrium_occurrence is not None and recorded_occurrence is not None \
            and int(equilibrium_occurrence) != int(recorded_occurrence):
        raise ValueError(
            f"equilibrium_occurrence={equilibrium_occurrence!r}, but the kinetic state was built on equilibrium "
            f"occurrence {recorded_occurrence!r} ({lineage or 'lineage not recorded'}); mapping its Thomson "
            "channels or comparing its pressure through another equilibrium would mix two states"
        )
    occurrence = equilibrium_occurrence if equilibrium_occurrence is not None else recorded_occurrence

    base = f"core_profiles.profiles_1d.{j}"
    eq_base = f"equilibrium.time_slice.{i}"

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
    t_i_valid = record.get("t_i_valid")
    valid = None
    if t_i_valid is not None:
        valid = np.asarray(t_i_valid, dtype=bool).reshape(-1)
        if valid.size != size:
            raise ValueError(f"kinetic_state t_i_valid has {valid.size} entries for a core grid of {size}")
    pe_stored = array(ods, f"{base}.electrons.pressure")
    pe_derived = pe_stored is None or pe_stored.size != size
    pe = _E * ne * te if pe_derived else pe_stored
    pi = array(ods, f"{base}.pressure_ion_total")
    pi = _masked(pi, valid) if pi is not None and pi.size == size else None
    ti = array(ods, f"{base}.t_i_average")
    ti = _masked(ti, valid) if ti is not None and ti.size == size else None
    zeff = array(ods, f"{base}.zeff")
    zeff = zeff if zeff is not None and zeff.size == size else None
    main, impurity = _ion_densities(ods, base, size)

    # --- Thomson, mapped through the selected equilibrium -------------------------
    ts = _thomson(ods, t_core)
    ts_x = None
    if ts is not None:
        from vaft.process.profile import CoordinateUnavailableError, equilibrium_mapping_points

        mapped = equilibrium_mapping_points(ods, ts["r"], ts["z"], time=t_eq)
        try:
            ts_x = mapped.select(coordinate)
        except CoordinateUnavailableError as exc:
            raise ValueError(f"the Thomson channels cannot be placed in {coordinate}: {exc}") from exc

    def ts_points(values, sigma, label, colour, marker, scale=1.0):
        if ts_x is None:
            return None
        keep = np.isfinite(ts_x) & np.isfinite(values)
        if not keep.any():
            return None
        err = np.asarray(sigma, dtype=float)[keep] * scale
        return _series(ts_x[keep], np.asarray(values)[keep] * scale, label, colour,
                       yerr=err if np.isfinite(err).any() else None, marker=marker)

    # --- pressure ---------------------------------------------------------------------
    traces: list[dict] = []

    def add(panel: list, series: Series | None, quantity: str, path: str, role: str, sigma: str) -> None:
        if series is None:
            return
        panel.append(series)
        traces.append({"panel": panel_name[id(panel)], "label": series.label, "quantity": quantity,
                       "path": path, "role": role, "sigma": sigma})

    pressure: list[Series] = []
    density: list[Series] = []
    temperature: list[Series] = []
    charge: list[Series] = []
    panel_name = {id(pressure): "pressure", id(density): "density", id(temperature): "temperature",
                  id(charge): "zeff"}

    p_eq = array(ods, f"{eq_base}.profiles_1d.pressure")
    role = _role(roles, "equilibrium.pressure")
    add(pressure, _series(x_eq, p_eq, rf"$p_{{\mathrm{{eq}}}}$ ({role})", _COLOURS["equilibrium"]),
        "p_eq", f"{eq_base}.profiles_1d.pressure", role, "none")
    if pe_derived:
        add(pressure, _series(x, pe, r"$p_e = e\,n_e T_e$ (derived)", _COLOURS["electron"]),
            "p_e", f"{base}.electrons.{ne_leaf} * temperature", "derived", "none")
    else:
        role = _role(roles, "electrons.pressure")
        add(pressure, _series(x, pe, rf"$p_e$ ({role})", _COLOURS["electron"],
                              yerr=_sigma(ods, f"{base}.electrons.pressure_error_upper", size)),
            "p_e", f"{base}.electrons.pressure", role, "stored")
    if pi is not None:
        role = _role(roles, "pressure_ion_total")
        add(pressure, _series(x, pi, rf"$p_i$ ({role})", _COLOURS["ion"]),
            "p_i", f"{base}.pressure_ion_total", role, "none")
        p_kin = pe + pi
        if np.isfinite(p_kin).any():
            role = _role(roles, "p_kin", "display sum")
            add(pressure, _series(x, p_kin, rf"$p_e + p_i$ ({role})", _COLOURS["closure"],
                                  linestyle="--"),
                "p_kin", "p_e + p_i", role, "none")
    if ts is not None:
        p_ts = _E * ts["n_e"] * ts["t_e"]
        # Independent n_e and T_e errors: no covariance is stored to do better.
        sp_ts = _E * np.hypot(ts["t_e"] * ts["sigma_n_e"], ts["n_e"] * ts["sigma_t_e"])
        role = _role(roles, "thomson_scattering.p_e", "measured")
        add(pressure, ts_points(p_ts, sp_ts, rf"$p_e^{{\mathrm{{TS}}}}$ ({role})",
                                _COLOURS["measurement"], "D"),
            "p_e_thomson", "e * thomson_scattering.channel.{c}.n_e.data * t_e.data", role,
            "propagated, n_e and T_e errors independent")

    # --- density ------------------------------------------------------------------------
    display = resolve_display("m^-3", unit="10^19 m^-3")
    scale = display.scale
    role = _role(roles, f"electrons.{ne_leaf}", _role(roles, "electrons.density"))
    sigma = _sigma(ods, f"{base}.electrons.{ne_leaf}_error_upper", size)
    add(density, _series(x, ne * scale, rf"$n_e$ ({role})", _COLOURS["electron"],
                         yerr=None if sigma is None else sigma * scale),
        "n_e", f"{base}.electrons.{ne_leaf}", role, "stored" if sigma is not None else "none")
    role = _role(roles, "ion.density")
    if main is not None:
        values, labels, paths = main
        add(density, _series(x, values * scale, rf"$n_{{\mathrm{{{_math(labels)}}}}}$ ({role})", _COLOURS["ion"]),
            "n_main_ion", " + ".join(paths), role, "none")
    if impurity is not None:
        values, labels, paths = impurity
        add(density, _series(x, values * scale, rf"$n_{{\mathrm{{{_math(labels)}}}}}$ ({role})",
                             _COLOURS["impurity"]),
            "n_impurity_ion", " + ".join(paths), role, "none")
    if ts is not None:
        role = _role(roles, "thomson_scattering.n_e", "measured")
        add(density, ts_points(ts["n_e"], ts["sigma_n_e"], rf"$n_e^{{\mathrm{{TS}}}}$ ({role})",
                               _COLOURS["measurement"], "o", scale),
            "n_e_thomson", "thomson_scattering.channel.{c}.n_e.data", role, "stored")

    # --- temperature --------------------------------------------------------------------
    role = _role(roles, "electrons.temperature")
    sigma = _sigma(ods, f"{base}.electrons.temperature_error_upper", size)
    add(temperature, _series(x, te, rf"$T_e$ ({role})", _COLOURS["electron"], yerr=sigma),
        "T_e", f"{base}.electrons.temperature", role, "stored" if sigma is not None else "none")
    valid_range = None
    if ti is not None:
        role = _role(roles, "t_i_average")
        finite = np.isfinite(ti)
        if finite.any():
            valid_range = [float(np.min(x[finite])), float(np.max(x[finite]))]
        count_note = f", {int(finite.sum())}/{size} valid" if valid is not None else ""
        add(temperature, _series(x, ti, rf"$T_i$ ({role}{count_note})", _COLOURS["ion"]),
            "T_i", f"{base}.t_i_average", role, "none")
    if ts is not None:
        role = _role(roles, "thomson_scattering.t_e", "measured")
        add(temperature, ts_points(ts["t_e"], ts["sigma_t_e"], rf"$T_e^{{\mathrm{{TS}}}}$ ({role})",
                                   _COLOURS["measurement"], "o"),
            "T_e_thomson", "thomson_scattering.channel.{c}.t_e.data", role, "stored")

    # --- Z_eff --------------------------------------------------------------------------------
    if zeff is not None:
        role = _role(roles, "zeff")
        add(charge, _series(x, zeff, rf"$Z_{{\mathrm{{eff}}}}$ ({role})", _COLOURS["impurity"]),
            "Z_eff", f"{base}.zeff", role, "none")
    reference = _zeff_reference(ods, j, record)
    if reference is not None:
        value, label, path, role = reference
        span = np.array([0.0, 1.0])
        add(charge, _series(span, np.full(2, value), label, _COLOURS["reference"], linestyle="--"),
            "Z_eff_reference", path, role, "none")

    # --- the measured span, shaded ----------------------------------------------------------
    span_line = None
    if ts_x is not None:
        measured = np.isfinite(ts_x) & (np.isfinite(ts["n_e"]) | np.isfinite(ts["t_e"]))
        if measured.sum() >= 2:
            span_line = (float(np.min(ts_x[measured])), float(np.max(ts_x[measured])))

    def spans(first: bool) -> tuple[ReferenceLine, ...]:
        if span_line is None:
            return ()
        return (ReferenceLine(x=span_line[0], x_end=span_line[1], label="TS span" if first else ""),)

    selection = {
        "time": t_core, "core_profiles_slice": j, "equilibrium_time": t_eq, "equilibrium_slice": i,
        "equilibrium_lineage": lineage, "equilibrium_occurrence": occurrence,
        "thomson_time": None if ts is None else ts["time"],
        "thomson_channels": [] if ts is None else [int(c) for c in ts["index"]],
        "thomson_span": None if span_line is None else list(span_line),
        "t_i_valid_range": valid_range, "kinetic_state_recorded": bool(record),
        "roles_recorded": bool(roles), "composition": record.get("composition"),
        "coordinate": coordinate,
    }
    label = COORDINATE_LABELS[coordinate]
    limits = (0.0, 1.0)

    def panel(series, y_label, unit, name, first=False, display_spec=None):
        return Profile1D(
            series=tuple(series), coordinate_label=label, y_label=y_label, y_unit=unit,
            x_limits=limits, display=display_spec, reference_lines=spans(first),
            metadata={"panel": name, "traces": [t for t in traces if t["panel"] == name], **selection},
        )

    models = (
        panel(pressure, "$p$", "Pa", "pressure", first=True),
        panel(density, "$n$", display.unit, "density", display_spec=display),
        panel(temperature, "$T$", "eV", "temperature"),
        panel(charge, r"$Z_{\mathrm{eff}}$", "", "zeff"),
    )
    if not any(model.series for model in models):
        raise ValueError("no kinetic-state quantity is stored at this time")
    # One line: a stacked layout needs every inch of its height for the panels.
    eq_text = (f"{lineage or 'equilibrium lineage not recorded'}"
               + (f" #{occurrence}" if occurrence is not None else "")
               + f" at {t_eq * 1e3:.1f} ms")
    ts_text = "" if ts is None or ts["time"] is None else f", TS {ts['time'] * 1e3:.1f} ms"
    grid = layout == "grid"
    # A short stacked panel has no room for its legend: it goes beside it.
    legends = ({"legend_loc": "outside", "legend_fontsize": "x-small"} if grid else {"legend_loc": "outside", "legend_ncols": 2, "legend_fontsize": "x-small"},) * len(models)
    return Panels(models=models, nrows=2 if grid else 4, ncols=2 if grid else 1, share_x=True,
                  member_styles=legends,
                  suptitle=f"Kinetic state at {t_core * 1e3:.1f} ms ({eq_text}{ts_text})")


def _math(labels: list[str]) -> str:
    """Species labels as one mathtext subscript: ``H+`` -> ``H^+``, joined by commas."""
    text = ",".join(label.replace("+", "^+").replace("$", "") for label in labels)
    return text


def _equilibrium_psi_norm(ods: Any, eq_base: str) -> np.ndarray | None:
    psi = array(ods, f"{eq_base}.profiles_1d.psi")
    axis = _scalar(get(ods, f"{eq_base}.global_quantities.psi_axis"))
    edge = _scalar(get(ods, f"{eq_base}.global_quantities.psi_boundary"))
    if psi is None or axis is None or edge is None or edge == axis:
        return None
    return (psi - axis) / (edge - axis)


def _ion_densities(ods: Any, base: str, size: int):
    """``(main, impurity)``: each ``(density, labels, paths)`` or ``None``.

    The main ion is every hydrogenic species (``element[0].z_n == 1``); the
    charged impurity-ion density is the sum over the species with ``z_n > 1``.
    A species without a nuclear charge is in neither: its kind is not stated.
    """
    groups: dict[str, list] = {"main": [], "impurity": []}
    for k in range(count(ods, f"{base}.ion")):
        prefix = f"{base}.ion.{k}"
        z_n = _scalar(get(ods, f"{prefix}.element.0.z_n"))
        density = array(ods, f"{prefix}.density")
        if z_n is None or density is None or density.size != size:
            continue
        label = str(get(ods, f"{prefix}.label", "") or f"ion {k}").strip()
        groups["main" if round(z_n) == 1 else "impurity"].append((density, label, f"{prefix}.density"))
    out = []
    for name in ("main", "impurity"):
        items = groups[name]
        if not items:
            out.append(None)
            continue
        total = np.sum([item[0] for item in items], axis=0)
        out.append((total, [item[1] for item in items], [item[2] for item in items]))
    return tuple(out)


def _zeff_reference(ods: Any, j: int, record: Mapping[str, Any]):
    """``(value, label, path, role)`` of the Z_eff reference, or ``None``.

    The stored resistive ``Z_eff`` of the slice's time when there is one, else
    the composition target the kinetic-state record states.
    """
    roles = record.get("roles") or {}
    stored = array(ods, "core_profiles.global_quantities.z_eff_resistive")
    if stored is not None and stored.size:
        value = float(stored.reshape(-1)[min(j, stored.size - 1)])
        if np.isfinite(value):
            role = _role(roles, "z_eff_resistive")
            return (value, rf"$Z_{{\mathrm{{eff}}}}^{{\mathrm{{res}}}} = {value:.3g}$ ({role})",
                    "core_profiles.global_quantities.z_eff_resistive", role)
    composition = record.get("composition") or {}
    target = _scalar(composition.get("target_zeff")) if isinstance(composition, Mapping) else None
    if target is None:
        return None
    role = _role(roles, "composition")
    return (target, rf"target $\langle Z_{{\mathrm{{eff}}}}\rangle = {target:g}$ ({role})",
            "core_profiles.code.parameters kinetic_state.composition.target_zeff", role)
