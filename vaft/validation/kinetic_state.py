"""Kinetic state of a reconstructed slice: Thomson against EFIT pressure (#1430).

The conference atlas lanes (#1454) share one row identity,
``(shot, time_efit_s, efit_lineage)``, and every row carries the EFIT quality
label of the magnetics slice at that time.  This module computes the
slice-level quantities those rows hold; the population, file layout and
schemas live in ``workflow/kinetic_state``.

It sits in the validation layer because what it produces is a comparison of a
reconstruction against a measurement it never used (#1430 treats the
magnetics-only ratio as held-out kinetic validation), and because the
transport-readiness rule that follows it has its precedent here
(:func:`vaft.validation.neoclassical.input_readiness`).  It composes providers
and computes ratios; it grades nothing.

Two rules run through everything here, because they are the ones VAFT keeps
breaking:

* **Slices are found by time, never by index.**  :func:`slice_at_time` takes a
  time and a tolerance and refuses beyond it; a one-slice sample cannot tell the
  two apart, so the tests use shuffled multi-slice equilibria.
* **Inputs are never mutated.**  Reads go through :mod:`vaft.ods_access`; an
  OMAS read of a missing path would otherwise create it.

The Thomson comparison itself is not reimplemented: it is
:func:`vaft.validation.equilibrium.thomson_pressure_samples`, the sampler behind
the ``thomson_pressure`` check that ``criteria.py`` grades slices with, so the
dataset and the grade cannot drift apart.

Ratios
------
``R_p``
    ``p_EFIT / p_e`` at each Thomson channel inside the LCFS.
``R_sum``
    ``sum p_EFIT / sum p_e`` over those channels -- ``exp(-log_ratio)`` of the
    validation check, which is the quantity the criteria band bounds.
``R_W``
    ``int p_EFIT dV / int p_e dV`` over the flux span the channels cover, with
    ``p_e`` from the ``core_profiles`` fit at the matched time.  No
    extrapolation.
``R_W_full``
    The same over the whole plasma, which extrapolates the ``core_profiles`` fit
    beyond the channels; reported, and flagged as extrapolated.

Inferred ion temperature
------------------------
:func:`infer_ti_pressure_partition` is #1426's closure,
``T_i = (p_eq - e n_e T_e) / (e sum n_i)``, with the ion density from
:func:`ion_composition`.  It is an *inferred* quantity and a separate lineage:
the machine's Ti/Te policy (#1414) is not replaced, and a pressure that was
itself fitted with an assumed Ti/Te is refused as circular.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping, Sequence

import numpy as np

from vaft.ods_access import path_count, path_value

__all__ = [
    "CONTRACT_VERSION",
    "INDEPENDENT_LINEAGES",
    "TI_FLAGS",
    "TI_NOTES",
    "default_tolerance",
    "LINEAGES",
    "QUALITIES",
    "core_profiles_electron_pressure",
    "infer_ti_pressure_partition",
    "integrated_pressure_ratio",
    "ion_composition",
    "match_thomson_pressure",
    "pressure_sigma",
    "rho_tor_norm_of",
    "slice_at_time",
    "state_time",
]

#: Version of the state key contract (#1454).  A change to the row identity, a
#: column's meaning or the matching rule bumps it.
CONTRACT_VERSION = "1"

#: ``electron_kinetic`` is the ``electron_efit`` stage: Thomson pressure plus the
#: machine's Ti/Te policy.  Measured-Ti ``kinetic_efit`` has no Tier A shot.
LINEAGES = ("magnetics", "electron_kinetic")

#: criteria.py ``slice_labels`` labels a row may carry; unreconstructible slices
#: never become rows.
QUALITIES = ("good", "admissible")

_ELEMENTARY_CHARGE = 1.602176634e-19
_TIME_DIGITS = 4


def state_time(time_s: float) -> float:
    """A slice time as it appears in the state key: seconds, to 1e-4 s."""
    return round(float(time_s), _TIME_DIGITS)


def _array(ods: Any, path: str) -> np.ndarray | None:
    value = path_value(ods, path)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.size else None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _slice_times(ods: Any, ids: str) -> np.ndarray | None:
    """Every slice time of ``ids``, from the per-slice ``time`` when present."""
    container = "time_slice" if ids == "equilibrium" else "profiles_1d"
    count = path_count(ods, f"{ids}.{container}")
    if count == 0:
        return None
    # The per-slice time wins where present, as in vaft.validation.equilibrium:
    # the slice that is compared and the time it is reported under must be one.
    times = np.array([_float(path_value(ods, f"{ids}.{container}.{i}.time")) for i in range(count)])
    shared = _array(ods, f"{ids}.time")
    if shared is not None and shared.size == count:
        times = np.where(np.isfinite(times), times, shared)
    return times


def default_tolerance(times: np.ndarray) -> float:
    """Half the median cadence, at least 1 ms -- the rule ``vaft.validation`` uses."""
    times = np.asarray(times, dtype=float)
    times = np.sort(times[np.isfinite(times)])
    if times.size < 2:
        return 1.0e-3
    return max(0.5 * float(np.median(np.diff(times))), 1.0e-3)


def slice_at_time(ods: Any, time_s: float, *, ids: str = "equilibrium",
                  tolerance_s: float | None = None) -> tuple[int, float]:
    """The ``ids`` slice nearest ``time_s``, as ``(index, signed offset)``.

    The offset is ``t_slice - time_s``.  Raises :class:`LookupError` when there
    is no slice within ``tolerance_s`` (default: :func:`default_tolerance` of
    the slice times).
    """
    times = _slice_times(ods, ids)
    if times is None or not np.isfinite(times).any():
        raise LookupError(f"no {ids} slice times to match {time_s} s on")
    offsets = times - float(time_s)
    index = int(np.nanargmin(np.abs(offsets)))
    tolerance = default_tolerance(times) if tolerance_s is None else float(tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        # NaN would compare False against every offset and accept any slice
        raise ValueError(f"tolerance_s must be a finite non-negative number, not {tolerance_s!r}")
    if abs(offsets[index]) > tolerance:
        raise LookupError(
            f"the nearest {ids} slice to {time_s} s is {offsets[index]:+.4g} s away, beyond {tolerance:.4g} s"
        )
    return index, float(offsets[index])


def _psi_norm_profile(ods: Any, index: int) -> np.ndarray | None:
    root = f"equilibrium.time_slice.{index}"
    psi = _array(ods, f"{root}.profiles_1d.psi")
    axis = _float(path_value(ods, f"{root}.global_quantities.psi_axis"))
    boundary = _float(path_value(ods, f"{root}.global_quantities.psi_boundary"))
    if psi is None:
        return None
    if not (math.isfinite(axis) and math.isfinite(boundary)) or axis == boundary:
        axis, boundary = float(psi[0]), float(psi[-1])
        if axis == boundary:
            return None
    return (psi - axis) / (boundary - axis)


def rho_tor_norm_of(ods: Any, index: int) -> dict[str, Any]:
    """The slice's ``(psi_norm, rho_tor_norm)`` profile pair, or why there is none.

    ``rho_tor_norm`` is taken from the slice's toroidal flux ``phi`` when it is
    there, else from a stored ``rho_tor_norm`` that is not the ``sqrt(psi_N)``
    proxy.  It is never substituted by ``sqrt(psi_N)``: the contract marks the
    coordinate ``unavailable`` instead.
    """
    from vaft.data._derived import is_rho_pol_proxy

    root = f"equilibrium.time_slice.{index}.profiles_1d"
    psi_norm = _psi_norm_profile(ods, index)
    if psi_norm is None:
        return {"coordinate": "unavailable", "reason": "the slice has no psi profile"}
    order = np.argsort(psi_norm)
    phi = _array(ods, f"{root}.phi")
    if phi is not None and phi.size == psi_norm.size and np.all(np.isfinite(phi)):
        # normalized from the axis end, whichever end of the array that is
        flux = np.abs(phi - phi[order[0]])
        if flux[order[-1]] > 0 and np.all(np.diff(flux[order]) >= 0):
            rho = np.sqrt(flux / flux[order[-1]])
            return {"coordinate": "rho_tor_norm", "source": "phi", "psi_norm": psi_norm, "rho_tor_norm": rho}
    stored = _array(ods, f"{root}.rho_tor_norm")
    if stored is not None and stored.size == psi_norm.size:
        if is_rho_pol_proxy(stored, psi_norm):
            return {"coordinate": "unavailable", "reason": "rho_tor_norm is the sqrt(psi_N) proxy", "psi_norm": psi_norm}
        along = stored[order]
        plausible = (np.all(np.isfinite(along)) and np.all(np.diff(along) >= 0)
                     and abs(along[0]) < 0.05 and abs(along[-1] - 1.0) < 0.05)
        if plausible:
            return {"coordinate": "rho_tor_norm", "source": "stored", "psi_norm": psi_norm, "rho_tor_norm": stored}
        return {"coordinate": "unavailable", "psi_norm": psi_norm,
                "reason": "the stored rho_tor_norm is not a monotonic 0-to-1 coordinate"}
    return {"coordinate": "unavailable", "reason": "no usable phi and no rho_tor_norm", "psi_norm": psi_norm}


def match_thomson_pressure(equilibrium: Any, thomson: Any, *, time_s: float,
                           slice_tolerance_s: float | None = None,
                           ts_tolerance_s: float | None = None) -> dict[str, Any]:
    """Thomson electron pressure against the EFIT pressure of the slice at ``time_s``.

    ``slice_tolerance_s`` bounds how far the equilibrium slice may sit from
    ``time_s``; ``ts_tolerance_s`` how far the Thomson sample may sit from that
    slice (default: half the equilibrium cadence, at least 1 ms).  The contract
    (#1454) fixes the latter to the *magnetics* EFIT cadence for both lineages,
    so a caller with a one-slice kinetic equilibrium passes it explicitly.

    Returns a dict with ``ts_status`` in ``matched | unmatched | invalid``:
    ``unmatched`` when no Thomson sample is within tolerance, ``invalid`` when
    one is but nothing can be compared.  Only ``matched`` carries ratios.
    """
    from vaft.validation.equilibrium import thomson_pressure_samples

    index, slice_offset = slice_at_time(equilibrium, time_s, tolerance_s=slice_tolerance_s)
    times = _slice_times(equilibrium, "equilibrium")
    time_efit = float(times[index])
    out: dict[str, Any] = {"time_slice": index, "time_efit_s": state_time(time_efit),
                           "slice_offset_s": slice_offset}
    sampled = thomson_pressure_samples(equilibrium, index, thomson, tolerance=ts_tolerance_s)
    if "ts_time" in sampled:
        out.update(time_ts_s=sampled["ts_time"], dt_ts_efit_s=sampled["ts_time"] - time_efit,
                   ts_tolerance_s=sampled["tolerance"])
    if not sampled["available"]:
        unmatched = sampled["code"] in ("beyond_tolerance", "no_thomson")
        out.update(ts_status="unmatched" if unmatched else "invalid", reason=sampled["reason"])
        return out
    coordinate = rho_tor_norm_of(equilibrium, index)
    channels = []
    for sample in sampled["channels"]:
        row = dict(sample)
        row["r_p"] = sample["p_recon"] / sample["p_e"]
        if coordinate["coordinate"] == "rho_tor_norm":
            order = np.argsort(coordinate["psi_norm"])
            row["rho_tor_norm"] = float(np.interp(sample["psi_norm"], coordinate["psi_norm"][order],
                                                  coordinate["rho_tor_norm"][order]))
        else:
            row["rho_tor_norm"] = math.nan
        channels.append(row)
    p_e = np.array([row["p_e"] for row in channels])
    p_recon = np.array([row["p_recon"] for row in channels])
    total_e, total_recon = float(p_e.sum()), float(p_recon.sum())
    r_sum = total_recon / total_e if total_e > 0 else math.nan
    out.update(
        ts_status="matched",
        rho_coordinate=coordinate["coordinate"],
        rho_reason=coordinate.get("reason"),
        channels=channels,
        points=len(channels),
        r_sum=r_sum,
        log_ratio=math.log(total_e / total_recon) if total_e > 0 and total_recon > 0 else math.nan,
        psi_norm_span=(float(min(r["psi_norm"] for r in channels)), float(max(r["psi_norm"] for r in channels))),
    )
    return out


def core_profiles_electron_pressure(profiles: Any, *, time_s: float,
                                    tolerance_s: float | None = None) -> dict[str, Any]:
    """``p_e = e n_e T_e`` of the ``core_profiles`` slice at ``time_s``, on its ``rho_tor_norm``.

    Returns ``{"available": False, "reason"}`` when there is no slice within
    tolerance or it lacks a positive density and temperature.
    """
    try:
        index, offset = slice_at_time(profiles, time_s, ids="core_profiles", tolerance_s=tolerance_s)
    except LookupError as missing:
        return {"available": False, "reason": str(missing)}
    root = f"core_profiles.profiles_1d.{index}"
    rho = _array(profiles, f"{root}.grid.rho_tor_norm")
    temperature = _array(profiles, f"{root}.electrons.temperature")
    if rho is None or temperature is None:
        return {"available": False, "reason": "the core_profiles slice lacks rho_tor_norm or T_e"}
    # density_thermal when it is there and on this grid, else density
    density, leaf = None, None
    for name in ("density_thermal", "density"):
        candidate = _array(profiles, f"{root}.electrons.{name}")
        if candidate is not None and candidate.size == rho.size:
            density, leaf = candidate, name
            break
    if density is None or temperature.size != rho.size:
        return {"available": False, "reason": "core_profiles rho_tor_norm, n_e and T_e are missing or differ in length"}
    times = _slice_times(profiles, "core_profiles")
    return {"available": True, "time_s": float(times[index]), "index": index, "offset_s": offset,
            "rho_tor_norm": rho, "n_e": density, "t_e": temperature, "density_leaf": leaf,
            "p_e": _ELEMENTARY_CHARGE * density * temperature}


def integrated_pressure_ratio(equilibrium: Any, index: int, rho_tor_norm: np.ndarray, p_e: np.ndarray,
                              *, psi_norm_span: tuple[float, float] | None = None) -> dict[str, Any]:
    """``int p_EFIT dV / int p_e dV`` over a flux span of one slice.

    Integrated on the slice's own ``(R, Z)`` grid with the axisymmetric volume
    element and the boundary-outline cell weights of
    :func:`vaft.process.equilibrium.plasma_cell_weights`, so no volume profile
    is needed (the magnetics EFIT product carries none).  ``p_e`` is given on
    ``rho_tor_norm`` and placed on psi_N through the slice's own coordinate
    pair; outside the profile's ``rho_tor_norm`` range it is not extrapolated
    (those cells are dropped from both integrals).

    ``psi_norm_span`` restricts both integrals to ``lo <= psi_N <= hi``; ``None``
    integrates the whole plasma.
    """
    from vaft.process.equilibrium import plasma_cell_weights

    coordinate = rho_tor_norm_of(equilibrium, index)
    if coordinate["coordinate"] != "rho_tor_norm":
        return {"available": False, "reason": coordinate["reason"]}
    root = f"equilibrium.time_slice.{index}"
    r_grid, z_grid = _array(equilibrium, f"{root}.profiles_2d.0.grid.dim1"), _array(equilibrium, f"{root}.profiles_2d.0.grid.dim2")
    psi_2d = _array(equilibrium, f"{root}.profiles_2d.0.psi")
    pressure = _array(equilibrium, f"{root}.profiles_1d.pressure")
    axis = _float(path_value(equilibrium, f"{root}.global_quantities.psi_axis"))
    boundary = _float(path_value(equilibrium, f"{root}.global_quantities.psi_boundary"))
    if r_grid is None or z_grid is None or psi_2d is None or pressure is None or not (
            math.isfinite(axis) and math.isfinite(boundary)) or axis == boundary:
        return {"available": False, "reason": "the slice has no 2-D psi, pressure or flux bounds"}
    if pressure.size != coordinate["psi_norm"].size:
        return {"available": False, "reason": "profiles_1d.pressure and psi differ in length"}
    if psi_2d.shape == (z_grid.size, r_grid.size) and r_grid.size != z_grid.size:
        psi_2d = psi_2d.T
    psi_n = (psi_2d - axis) / (boundary - axis)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        weights = plasma_cell_weights(r_grid, z_grid, psi_n,
                                      _array(equilibrium, f"{root}.boundary.outline.r"),
                                      _array(equilibrium, f"{root}.boundary.outline.z"))
    outline = not any("no usable boundary outline" in str(w.message) for w in caught)
    rm, _ = np.meshgrid(r_grid, z_grid, indexing="ij")
    area = np.gradient(r_grid)[:, None] * np.gradient(z_grid)[None, :]
    dv = 2.0 * np.pi * rm * area * weights
    grid_psi, grid_rho = coordinate["psi_norm"], coordinate["rho_tor_norm"]
    order = np.argsort(grid_psi)
    clipped = np.clip(psi_n, 0.0, 1.0)
    p_efit = np.interp(clipped, grid_psi[order], pressure[order])
    rho_cells = np.interp(clipped, grid_psi[order], grid_rho[order])
    rho_tor_norm, p_e = np.asarray(rho_tor_norm, float), np.asarray(p_e, float)
    by_rho = np.argsort(rho_tor_norm)
    covered = (rho_cells >= rho_tor_norm.min()) & (rho_cells <= rho_tor_norm.max())
    p_e_cells = np.interp(rho_cells, rho_tor_norm[by_rho], p_e[by_rho])
    mask = (weights > 0) & covered
    if psi_norm_span is not None:
        lo, hi = psi_norm_span
        mask &= (psi_n >= lo) & (psi_n <= hi)
    if not mask.any():
        return {"available": False, "reason": "no plasma cell lies in the requested span"}
    numerator = float(np.sum(p_efit[mask] * dv[mask]))
    denominator = float(np.sum(p_e_cells[mask] * dv[mask]))
    return {
        "available": True,
        "ratio": numerator / denominator if denominator > 0 else math.nan,
        "w_efit_j": 1.5 * numerator,
        "w_e_j": 1.5 * denominator,
        "volume_m3": float(np.sum(dv[mask])),
        "cell_weights": "outline" if outline else "psi_threshold",
        "psi_norm_span": None if psi_norm_span is None else (float(psi_norm_span[0]), float(psi_norm_span[1])),
    }


# ---------------------------------------------------------------------------
# ion temperature from the pressure partition (#1426)
# ---------------------------------------------------------------------------

#: Equilibrium lineages whose pressure carries information independent of the
#: ion temperature being inferred.  ``electron_kinetic`` is excluded: its
#: pressure was fitted to ``n_e T_e (1 + Ti/Te)`` with Ti/Te *assumed*, so
#: inverting it returns the assumption (#1426 section 6).
INDEPENDENT_LINEAGES = ("magnetics",)

#: The flags that void a pressure-partition point: a flagged point has no T_i.
TI_FLAGS = ("p_i_nonpositive", "p_i_not_significant", "non_finite_input")

#: Notes a point can carry that leave its T_i in place.  ``sigma_unavailable``:
#: an input uncertainty is missing, so neither sigma(T_i) nor the significance
#: test exists for that point.
TI_NOTES = ("sigma_unavailable",)


def ion_composition(z_eff: float = 2.0, impurity: str = "C") -> dict[str, Any]:
    """Ion species and densities per electron from a Z_eff closure.

    One hydrogenic main ion (H+) and one fully stripped impurity, fixed by
    quasi-neutrality and the definition of Z_eff through
    :func:`vaft.code.gacode.inputs.impurity_fractions`, the closure GACODE input
    preparation uses.  The total ion density per electron comes *out of* that
    closure (5/6 for carbon at Z_eff = 2); it is not a separate constant.
    """
    from vaft.code.gacode.inputs import HYDROGEN_ISOTOPE_MASSES, IMPURITIES, impurity_fractions

    charge, mass, label = IMPURITIES[impurity]
    main, imp = impurity_fractions(z_eff, charge)
    species = (
        {"label": "H+", "z_ion": 1.0, "a": HYDROGEN_ISOTOPE_MASSES["H"], "density_per_electron": main},
        {"label": label, "z_ion": float(charge), "a": float(mass), "density_per_electron": imp},
    )
    return {
        "species": species,
        "ion_density_per_electron": main + imp,
        "z_eff": float(z_eff),
        "composition_source": f"policy: H+ with {label}, Z_eff={z_eff:g} (vaft.code.gacode.inputs.impurity_fractions)",
    }


def pressure_sigma(pressure: np.ndarray, neighbours: Sequence[np.ndarray] = (), *,
                   floor: float = 0.17) -> tuple[np.ndarray, str]:
    """The EFIT pressure uncertainty: ensemble spread with a model-form floor.

    ``pressure`` is the slice's pressure at the evaluation points and
    ``neighbours`` the pressure of other good or admissible slices of the same
    shot at the *same* normalized flux (the caller picks them by time, within
    1 ms).  The spread is the sample standard deviation over all of them; the
    result is ``max(spread, floor * pressure)`` pointwise.  ``floor`` defaults
    to 0.17, the KPPCUR model-form uncertainty on the pressure-weighted area
    found in #874.  Returns ``(sigma, basis)`` with ``basis`` either
    ``"ensemble(n=k)"`` when the spread exceeds the floor anywhere, or
    ``"floor"``.  A neighbour value that is not finite is left out of the
    spread at that point, and a point with fewer than two finite values keeps
    the floor.
    """
    pressure = np.asarray(pressure, dtype=float)
    floor_sigma = floor * np.abs(pressure)
    if not neighbours:
        return floor_sigma, "floor"
    stack = np.vstack([pressure, *[np.broadcast_to(np.asarray(n, dtype=float), pressure.shape) for n in neighbours]])
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: handled below
        spread = np.nanstd(stack, axis=0, ddof=1)
    # fewer than two finite values at a point: no spread there, the floor stands
    spread = np.where(counts >= 2, spread, 0.0)
    sigma = np.maximum(spread, floor_sigma)
    used = int(counts.max()) if counts.size else 0
    basis = f"ensemble(n={used})" if np.any(spread > floor_sigma) else "floor"
    return sigma, basis


def infer_ti_pressure_partition(p_eq: Any, n_e: Any, t_e: Any, *, composition: Mapping[str, Any],
                                sigma_p_eq: Any, sigma_n_e: Any, sigma_t_e: Any,
                                equilibrium_lineage: str) -> dict[str, Any]:
    """T_i from ``p_i = p_eq - e n_e T_e`` and ``T_i = p_i / (e sum n_i)``.

    All arrays are on the same points: ``p_eq`` [Pa], ``n_e`` [m^-3], ``t_e`` [eV]
    and their one-sigma uncertainties.  The ion density is
    ``composition["ion_density_per_electron"] * n_e`` with a common ion
    temperature, so ``T_i = p_eq / (e f n_e) - T_e / f``.

    The uncertainty is propagated linearly through that form, which keeps the
    n_e correlation between ``p_e`` and the ion density::

        dT_i/dp_eq = 1/(e f n_e), dT_i/dn_e = -p_eq/(e f n_e^2), dT_i/dT_e = -1/f

    Points are **flagged, never filled**: ``p_i <= 0`` (``p_i_nonpositive``) and
    ``p_i < sigma(p_i)`` (``p_i_not_significant``) get ``T_i = NaN``.

    An input uncertainty that is missing (NaN) does not void the point: its
    T_i stands, ``sigma_t_i`` is NaN, the significance test is skipped, and the
    point carries the note ``sigma_unavailable`` (:data:`TI_NOTES`).

    The inference is refused (``eligible = False``) unless the equilibrium
    lineage is in :data:`INDEPENDENT_LINEAGES`: a pressure fitted with an
    assumed Ti/Te returns the assumption.  **The guard trusts the declared
    lineage**; the equilibrium products carry no machine-readable record of
    one, so a caller must derive ``equilibrium_lineage`` from *where the
    pressure came from* (the FileDB stage it loaded), never from intent.  The
    result is an *inferred* quantity, never a measurement.

    Scalars broadcast against arrays; every input is brought to one shape.
    """
    if equilibrium_lineage not in INDEPENDENT_LINEAGES:
        return {"eligible": False, "origin": "inferred", "method": "equilibrium_pressure_partition",
                "reason": (f"circular: the {equilibrium_lineage!r} equilibrium pressure was reconstructed with an "
                           "assumed Ti/Te, so partitioning it returns that assumption")}
    p_eq, n_e, t_e, s_p, s_n, s_t = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (p_eq, n_e, t_e, sigma_p_eq, sigma_n_e, sigma_t_e)))
    fraction = float(composition["ion_density_per_electron"])
    p_e = _ELEMENTARY_CHARGE * n_e * t_e
    p_i = p_eq - p_e
    sigma_p_e = _ELEMENTARY_CHARGE * np.hypot(t_e * s_n, n_e * s_t)
    sigma_p_i = np.hypot(s_p, sigma_p_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_i = p_i / (_ELEMENTARY_CHARGE * fraction * n_e)
        d_p = 1.0 / (_ELEMENTARY_CHARGE * fraction * n_e)
        d_n = -p_eq / (_ELEMENTARY_CHARGE * fraction * n_e ** 2)
        d_t = -1.0 / fraction
        sigma_t_i = np.sqrt((d_p * s_p) ** 2 + (d_n * s_n) ** 2 + (d_t * s_t) ** 2)
        ti_te = t_i / t_e
    flags: list[list[str]] = []
    for k in range(p_eq.size):
        point: list[str] = []
        values = (p_eq.flat[k], n_e.flat[k], t_e.flat[k])
        sigmas_known = all(math.isfinite(v) for v in (s_p.flat[k], s_n.flat[k], s_t.flat[k]))
        if not all(math.isfinite(v) for v in values) or n_e.flat[k] <= 0 or t_e.flat[k] <= 0:
            point.append("non_finite_input")
        elif p_i.flat[k] <= 0:
            point.append("p_i_nonpositive")
        elif sigmas_known and p_i.flat[k] < sigma_p_i.flat[k]:
            point.append("p_i_not_significant")
        if not point and not sigmas_known:
            point.append("sigma_unavailable")
        flags.append(point)
    bad = np.array([any(f in TI_FLAGS for f in point) for point in flags], dtype=bool).reshape(p_eq.shape)
    t_i = np.where(bad, np.nan, t_i)
    sigma_t_i = np.where(bad, np.nan, sigma_t_i)
    ti_te = np.where(bad, np.nan, ti_te)
    return {
        "eligible": True, "origin": "inferred", "method": "equilibrium_pressure_partition",
        "equilibrium_lineage": equilibrium_lineage,
        "assumptions": ("common ion temperature", composition["composition_source"]),
        "p_e": p_e, "p_i": p_i, "sigma_p_i": sigma_p_i,
        "t_i": t_i, "sigma_t_i": sigma_t_i, "ti_te": ti_te, "flags": flags,
    }
