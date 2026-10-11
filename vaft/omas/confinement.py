"""ODS adapter of the experimental Ohmic confinement chain (issue #1905).

:func:`ohmic_confinement_ods` is the one ODS-facing entry to
:func:`vaft.process.confinement.ohmic_confinement_series`: it reads the
measured plasma current and loop voltage, matches them to the equilibrium
slices by physical time, takes $l_{i,3}$ and the reference radius from the
existing update routines, picks the stored energy named by ``w_source``, and
records every one of those choices next to the result. The physics -- the
inductive split, the power balance, the confinement time -- is the process
kernel's; nothing here re-derives it.

What stays out: which equilibrium slices are trustworthy (a study's labels,
passed as ``time_slices``), study thresholds (``workflow/confinement_scaling``),
and the legacy modelled power budget (:func:`vaft.omas.formula_wrapper.compute_power_balance`,
Spitzer heating and modelled radiation), which answers a different question
and keeps its own definition.

Nothing here writes to the ODS it reads: $l_{i,3}$ and the core-profile
energy are evaluated on scratch copies.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vaft.process.confinement import OhmicConfinement, ohmic_confinement_series

logger = logging.getLogger(__name__)

__all__ = [
    "W_SOURCES",
    "OhmicConfinementODS",
    "ohmic_confinement_ods",
]

#: Stored-energy sources and the definition recorded with each. Neither is a
#: global energy (#1713): thermal-as-global is a caller's explicit choice downstream.
W_SOURCES: Mapping[str, str] = {
    "efit_pressure": ("W_MHD = (3/2) int p_EFIT dV of the equilibrium slice: a reconstruction-derived pressure "
                      "energy, not an independent kinetic measurement"),
    "core_profiles_thermal": ("W_th = (3/2) <p> V with p = 2 n_e T_e e from core_profiles (T_i = T_e assumed) and "
                              "V of the equilibrium slice: the energy of the legacy compute_power_balance"),
}


@dataclass(frozen=True)
class OhmicConfinementODS:
    """:class:`~vaft.process.confinement.OhmicConfinement` of an ODS, with what was read to build it.

    ``eq_index`` are the equilibrium slices of ``chain.time``; ``loop_index``,
    ``loop_name``, ``loop_r_m`` and ``loop_z_m`` identify the flux loop whose
    voltage was used, ``voltage_source`` whether it was the stored voltage or
    derived from the stored flux, ``w_source`` the key of :data:`W_SOURCES`.
    """

    chain: OhmicConfinement
    eq_index: np.ndarray
    loop_index: int
    loop_name: str
    loop_r_m: float
    loop_z_m: float
    voltage_source: str
    w_source: str


def _window_mean(t: np.ndarray, y: np.ndarray, at: np.ndarray, width: float) -> np.ndarray:
    """Mean of the finite samples within ``width``/2 of each ``at`` (NaN where none)."""
    out = np.full(np.shape(at), np.nan)
    for k, t0 in enumerate(np.atleast_1d(at)):
        sel = (np.abs(t - t0) <= 0.5 * width) & np.isfinite(y)
        if np.any(sel):
            out[k] = float(np.mean(y[sel]))
    return out


def _slice_times(ods: Any) -> np.ndarray:
    n = len(ods["equilibrium.time_slice"])
    times = np.full(n, np.nan)
    for k in range(n):
        path = f"equilibrium.time_slice.{k}.time"
        if path in ods:
            times[k] = float(ods[path])
    if not np.isfinite(times).all() and "equilibrium.time" in ods:
        stored = np.asarray(ods["equilibrium.time"], dtype=float)
        if stored.size == n:
            times = np.where(np.isfinite(times), times, stored)
    return times


def _li3_all(ods: Any) -> np.ndarray:
    """$l_{i,3}$ of every slice from the post-#1477 update routine, on a scratch copy."""
    from vaft.omas.update import update_equilibrium_global_quantities_beta_li

    work = copy.deepcopy(ods)
    n = len(work["equilibrium.time_slice"])
    for k in range(n):
        for name in ("li_3", "beta_normal"):
            path = f"equilibrium.time_slice.{k}.global_quantities.{name}"
            if path in work:
                del work[path]
    update_equilibrium_global_quantities_beta_li(work, time_slice=list(range(n)))
    li = np.full(n, np.nan)
    for k in range(n):
        path = f"equilibrium.time_slice.{k}.global_quantities.li_3"
        if path in work:
            li[k] = float(work[path])
    return li


def _energy_efit_pressure(ods: Any, indices: Sequence[int]) -> np.ndarray:
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    out = np.full(len(indices), np.nan)
    for j, k in enumerate(indices):
        try:
            item = derive_global_descriptors(as_equilibrium(ods, time_index=int(k))).values.get("thermal_energy")
            out[j] = float(item.value) if item is not None and item.value is not None else np.nan
        except Exception as exc:  # noqa: BLE001 - one bad slice is NaN, not a lost shot
            logger.debug("equilibrium slice %s: no stored energy (%s)", k, exc)
    return out


def _energy_core_profiles(ods: Any, indices: Sequence[int]) -> np.ndarray:
    """The legacy energy, (3/2) <p>_V V, exactly as compute_power_balance forms it, on a scratch copy."""
    from vaft.formula.equilibrium import thermal_energy_from_p_V
    from vaft.omas.process_wrapper import compute_volume_averaged_pressure
    from vaft.omas.update import update_equilibrium_global_quantities_volume

    work = copy.deepcopy(ods)
    update_equilibrium_global_quantities_volume(work, time_slice=None)
    p_avg = np.asarray(compute_volume_averaged_pressure(work, time_slice=None, option="core_profiles"), dtype=float)
    out = np.full(len(indices), np.nan)
    for j, k in enumerate(indices):
        path = f"equilibrium.time_slice.{int(k)}.global_quantities.volume"
        if path in work and int(k) < p_avg.size:
            out[j] = float(thermal_energy_from_p_V(p_avg[int(k)], float(work[path])))
    return out


def ohmic_confinement_ods(
    ods: Any,
    *,
    diagnostics: Any = None,
    time_slices: Optional[Sequence[int]] = None,
    w_source: str = "efit_pressure",
    require_positive_energy: bool = False,
    loop_index: Optional[int] = None,
    rate_window_s: float = 3e-3,
    rate_polyorder: int = 1,
    dip_window_s: float = 1e-3,
    average_window_s: float = 5e-4,
    hold_li3_where_missing: bool = True,
) -> OhmicConfinementODS:
    """Ohmic power balance and confinement time of a discharge from its ODS (#1905 Sec. 3).

    Parameters
    ----------
    ods : ODS
        Carries ``equilibrium`` (and ``core_profiles`` for
        ``w_source="core_profiles_thermal"``); read, never written.
    diagnostics : ODS, optional
        Where ``magnetics.ip`` and ``magnetics.flux_loop`` are read; default ``ods``.
    time_slices : sequence of int, optional
        Equilibrium slices to use, e.g. a study's validated slices; rates are
        taken across these only, so an unreconstructed slice must not be among
        them. Default every slice.
    w_source : str, optional
        Key of :data:`W_SOURCES`; recorded with the result.
    require_positive_energy : bool, optional
        Drop slices whose stored energy is not finite and positive before any
        rate is taken (numerical validity, not study acceptance).
    loop_index : int, optional
        Flux loop to use; default :func:`vaft.omas.discharge_timing.inboard_midplane_loop`.
    rate_window_s, rate_polyorder : float, int, optional
        Local-fit window and degree of $dl_{i,3}/dt$ and $dW/dt$.
    dip_window_s : float, optional
        Local-quadratic window of $dI_p/dt$ on the current record.
    average_window_s : float, optional
        Averaging window of the measured current and loop voltage at each slice time.
    hold_li3_where_missing : bool, optional
        See :func:`vaft.process.confinement.ohmic_confinement_series`.

    Returns
    -------
    OhmicConfinementODS
        The chain on the selected slices and what was read to build it.

    Raises
    ------
    ValueError
        No equilibrium slice, an unknown ``w_source``, no measured plasma
        current, no usable flux loop, or no slice left to evaluate.

    Notes
    -----
    Defaults are the VEST Tier A reference settings of #548 (recorded in the
    result's ``settings``), numerical choices rather than physical constants.
    The measured current is paired with the EFIT $l_{i,3}$ (a hybrid), and the
    loop is not on the plasma boundary: both are recorded assumptions.
    """
    from vaft.omas.discharge_timing import inboard_midplane_loop, loop_voltage
    from vaft.omas.update import resolve_reference_major_radius
    from vaft.process.confinement import smoothed_time_derivative

    if w_source not in W_SOURCES:
        raise ValueError(f"w_source must be one of {', '.join(W_SOURCES)}, got {w_source!r}")
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        raise ValueError("the ODS has no equilibrium slice")
    diag = ods if diagnostics is None else diagnostics
    if "magnetics.ip.0.data" not in diag or "magnetics.ip.0.time" not in diag:
        raise ValueError("no measured plasma current (magnetics.ip.0) to pair with the equilibrium")

    times = _slice_times(ods)
    indices = np.arange(times.size) if time_slices is None else np.asarray(list(time_slices), dtype=int)
    indices = indices[np.isfinite(times[indices])]
    indices = indices[np.argsort(times[indices], kind="stable")]  # physical time order, whatever was passed
    w_all = (_energy_efit_pressure(ods, indices) if w_source == "efit_pressure"
             else _energy_core_profiles(ods, indices))
    if require_positive_energy:
        keep = np.isfinite(w_all) & (w_all > 0)
        indices, w_all = indices[keep], w_all[keep]
    if indices.size == 0:
        raise ValueError("no equilibrium slice left to evaluate")
    t = times[indices]

    ip_t = np.asarray(diag["magnetics.ip.0.time"], dtype=float)
    ip_y = np.asarray(diag["magnetics.ip.0.data"], dtype=float)
    ip = _window_mean(ip_t, ip_y, t, average_window_s)
    dip_dt = smoothed_time_derivative(ip_t, ip_y, window_s=dip_window_s, at=t)

    loop = inboard_midplane_loop(diag) if loop_index is None else int(loop_index)
    if loop is None:
        raise ValueError("no inboard flux loop carries a waveform")
    vt, vy, source = loop_voltage(diag, loop, prefer_measured=True)
    v_loop = _window_mean(np.asarray(vt, dtype=float), np.asarray(vy, dtype=float), t, average_window_s)
    base = f"magnetics.flux_loop.{loop}"
    name = str(diag[f"{base}.name"]) if f"{base}.name" in diag else str(loop)
    r_m = float(diag[f"{base}.position.0.r"]) if f"{base}.position.0.r" in diag else float("nan")
    z_m = float(diag[f"{base}.position.0.z"]) if f"{base}.position.0.z" in diag else float("nan")
    voltage_source = "measured voltage" if source == "voltage" else "derived -dPhi/dt from the stored flux"

    li3 = _li3_all(ods)[indices]
    r0 = float(resolve_reference_major_radius(ods))
    chain = ohmic_confinement_series(
        t, ip, dip_dt, v_loop, li3, r0, w_all, rate_window_s=rate_window_s, rate_polyorder=rate_polyorder,
        hold_li3_where_missing=hold_li3_where_missing,
        assumptions={
            "v_loop": (f"{voltage_source} of flux loop {name} (index {loop}, R = {r_m:.3f} m, Z = {z_m:.3f} m), "
                       f"mean over {average_window_s} s"),
            "i_p": (f"measured magnetics.ip mean over {average_window_s} s; dI_p/dt local quadratic over "
                    f"{dip_window_s} s; paired with the EFIT l_i3 (hybrid)"),
            "li_3": "vaft.omas.update.update_equilibrium_global_quantities_beta_li (IMAS DD l_i3)",
            "r0": f"{r0} m from vaft.omas.update.resolve_reference_major_radius (normalises l_i3; not R_geo)",
            "w": W_SOURCES[w_source],
            "time_slices": ("caller-selected equilibrium slices" if time_slices is not None else
                            "every equilibrium slice") + (", finite positive energy only"
                                                          if require_positive_energy else ""),
        },
    )
    return OhmicConfinementODS(chain=chain, eq_index=indices, loop_index=int(loop), loop_name=name,
                               loop_r_m=r_m, loop_z_m=z_m, voltage_source=voltage_source, w_source=w_source)
