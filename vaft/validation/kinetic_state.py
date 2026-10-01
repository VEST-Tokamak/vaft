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
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from vaft.ods_access import path_count, path_value

__all__ = [
    "CONTRACT_VERSION",
    "default_tolerance",
    "LINEAGES",
    "QUALITIES",
    "core_profiles_electron_pressure",
    "integrated_pressure_ratio",
    "match_thomson_pressure",
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
    times = _array(ods, f"{ids}.time")
    if times is not None and times.size == count:
        return times
    return np.array([_float(path_value(ods, f"{ids}.{container}.{i}.time")) for i in range(count)])


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
    phi = _array(ods, f"{root}.phi")
    if phi is not None and phi.size == psi_norm.size and np.all(np.isfinite(phi)) and phi[-1] != phi[0]:
        rho = np.sqrt(np.clip((phi - phi[0]) / (phi[-1] - phi[0]), 0.0, None))
        return {"coordinate": "rho_tor_norm", "source": "phi", "psi_norm": psi_norm, "rho_tor_norm": rho}
    stored = _array(ods, f"{root}.rho_tor_norm")
    if stored is not None and stored.size == psi_norm.size and not is_rho_pol_proxy(stored, psi_norm):
        return {"coordinate": "rho_tor_norm", "source": "stored", "psi_norm": psi_norm, "rho_tor_norm": stored}
    reason = "rho_tor_norm is the sqrt(psi_N) proxy" if stored is not None else "no phi and no rho_tor_norm"
    return {"coordinate": "unavailable", "reason": reason, "psi_norm": psi_norm}


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
        beyond = "ts_time" in sampled and sampled["time_offset"] > sampled["tolerance"]
        unmatched = beyond or "time base" in sampled["reason"] or "no thomson" in sampled["reason"]
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
    density = _array(profiles, f"{root}.electrons.density_thermal")
    if density is None:
        density = _array(profiles, f"{root}.electrons.density")
    temperature = _array(profiles, f"{root}.electrons.temperature")
    if rho is None or density is None or temperature is None:
        return {"available": False, "reason": "the core_profiles slice lacks rho_tor_norm, n_e or T_e"}
    if not (rho.size == density.size == temperature.size):
        return {"available": False, "reason": "core_profiles rho_tor_norm, n_e and T_e differ in length"}
    times = _slice_times(profiles, "core_profiles")
    return {"available": True, "time_s": float(times[index]), "offset_s": offset, "rho_tor_norm": rho,
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
