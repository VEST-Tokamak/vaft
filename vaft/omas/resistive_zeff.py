"""ODS adapters for the resistive effective charge (issue #1214).

Two readers turn an ODS into the plain containers of
:mod:`vaft.process.resistive_zeff`:

* :func:`romero_boundary_flux_ods` -- the one sanctioned path from a stored
  equilibrium to Romero's full-weber boundary flux.  It refuses an ODS whose
  flux unit only the terminal fallback of
  :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor` decided, and one on
  which that detector and :func:`vaft.process.equilibrium.as_equilibrium`
  disagree -- a silent 2 pi is the defect #652 and #1214 Sec. 4 guard.
* :func:`flux_surface_state_ods` -- one equilibrium slice and the
  ``core_profiles`` slice at the same time, with ``<J.B>`` from the
  Grad-Shafranov profiles and ``<B^2>`` from the traced geometry.

Neither writes to the ODS it reads, and nothing here writes
``core_profiles.zeff``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional, Tuple

import numpy as np

from vaft.process.resistive_zeff import FluxSurfaceState, RomeroBoundaryFlux

logger = logging.getLogger(__name__)

__all__ = ["romero_boundary_flux_ods", "flux_surface_state_ods", "detected_psi_per_radian"]


def detected_psi_per_radian(ods: Any, time_index: int) -> Optional[bool]:
    """Whether the stored psi is per radian, *as detected* -- ``None`` if undecided.

    The decision of :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor`
    without its terminal fallback: a declared COCOS, else the probe tiers.
    """
    from vaft.data.eqdsk import _declared_flux_exponent, ods_flux_exponent

    exponent = _declared_flux_exponent(ods)
    if exponent is None:
        exponent = ods_flux_exponent(ods, time_index)
    if exponent is None:
        return None
    return exponent == 0


def romero_boundary_flux_ods(
    ods: Any,
    *,
    time_range: Optional[Tuple[float, float]],
    source: Optional[dict] = None,
) -> RomeroBoundaryFlux:
    """Romero boundary flux, current and ``li_3`` of the slices in ``time_range``.

    Args:
        ods: ODS with a multi-slice ``equilibrium``; not modified.
        time_range: ``(t_start, t_end)`` inclusive [s]; required in substance --
            ``None`` takes every slice and fails on any slice without current.
        source: Extra provenance (shot, lineage, product path) to carry along.

    Raises:
        ValueError: The flux unit is undetected (only the fallback would
            decide), or the two detectors disagree on any selected slice.
    """
    from vaft.omas.process_wrapper import _romero_boundary_histories
    from vaft.process.equilibrium import as_equilibrium

    hist = _romero_boundary_histories(ods, time_range)
    for idx in hist["time_index"]:
        detected = detected_psi_per_radian(ods, int(idx))
        if detected is None:
            raise ValueError(
                f"equilibrium slice {int(idx)}: the psi unit is not detected (no declared "
                "COCOS and every probe abstained); the 1/(2 pi) fallback is a default, "
                "not a detection, and is refused here (#652)"
            )
        convention = bool(as_equilibrium(ods, time_index=int(idx)).convention.psi_per_radian)
        if detected != convention:
            raise ValueError(
                f"equilibrium slice {int(idx)}: the probes say per-radian={detected} but "
                f"as_equilibrium says per-radian={convention}; refusing a possible 2 pi slip"
            )
    normalization = (
        "stored Wb/rad, multiplied by 2 pi" if hist["psi_per_radian"] else "stored full Wb"
    )
    return RomeroBoundaryFlux(
        time=hist["time"],
        I_p=hist["I_p"],
        psi_boundary=hist["psi_boundary"],
        li_3=hist["li_3"],
        R0=float(hist["R0"]),
        flux_normalization=normalization,
        flux_sign=float(hist["flux_sign"]),
        source={"time_index": ",".join(str(int(i)) for i in hist["time_index"]),
                **(source or {})},
    )


def _leaf(ts, name):
    path = f"profiles_1d.{name}"
    if path not in ts:
        return None
    values = np.asarray(ts[path], dtype=float).reshape(-1)
    return values if values.size else None


def flux_surface_state_ods(
    ods: Any,
    *,
    time_slice: int,
    bootstrap: str = "none",
    bootstrap_z_eff: Optional[float] = None,
    time_tolerance_s: float = 5e-5,
    gs_tolerance: float = 0.1,
    source: Optional[dict] = None,
) -> FluxSurfaceState:
    """One paired equilibrium / core_profiles slice as a :class:`FluxSurfaceState`.

    Args:
        ods: ODS with ``equilibrium`` and ``core_profiles``; not modified.
        time_slice: The ``core_profiles`` slice; the equilibrium slice is the
            one at the same time (matched by time, never by index).
        bootstrap: ``"none"``, ``"sauter"`` or ``"redl"`` -- the analytic
            ``<J_bs.B>`` of :func:`vaft.omas.neoclassical.compute_bootstrap_current`.
        bootstrap_z_eff: The Z_eff the bootstrap current is evaluated at;
            required when ``bootstrap`` is not ``"none"``. It is held fixed
            while Z_eff is fitted, which is recorded in the label.
        time_tolerance_s: Largest core_profiles/equilibrium time gap accepted [s];
            the same tolerance the resistive Z_eff atlas keys on.
        gs_tolerance: Largest relative mismatch between ``ip`` and the current
            Grad-Shafranov gives from the converted ``p'`` and ``FF'``.
        source: Extra provenance to carry along.

    Raises:
        ValueError: ... or the converted derivatives fail the Grad-Shafranov
            current check -- psi converted but its derivatives not (or the
            reverse) would enter R_p squared as (2 pi)^2.

    Notes:
        ``<J.B> = -(F p' + F F' <B^2> / (mu0 F))`` with ``p'`` and ``F F'``
        converted to per-radian psi; its overall sign is taken so that it has
        the sign of ``I_p``. Surfaces where T_e or n_e is missing or not
        positive are dropped (the profiles are not extrapolated), and the
        share of the Grad-Shafranov current they carry is recorded as
        ``excluded_current_fraction``.
    """
    from vaft.formula.constants import MU0
    from vaft.omas.general import find_matching_time_indices
    from vaft.omas.neoclassical import _kinetic_on, _radial_extent

    cp_index, eq_index, time = find_matching_time_indices(
        ods, time_slice=time_slice, atol=time_tolerance_s)
    detected = detected_psi_per_radian(ods, eq_index)
    if detected is None:
        raise ValueError(
            f"equilibrium slice {eq_index}: psi unit not detected; refusing the fallback (#652)"
        )
    to_radian = 1.0 if detected else 1.0 / (2.0 * np.pi)

    work = ods
    ts = ods["equilibrium.time_slice"][eq_index]
    if any(_leaf(ts, name) is None for name in ("gm1", "gm5", "volume")):
        from omas import ODS

        from vaft.omas.update import update_equilibrium_profiles_1d_geometry

        work = ODS(consistency_check=False)
        work["equilibrium"] = copy.deepcopy(ods["equilibrium"])
        update_equilibrium_profiles_1d_geometry(work, time_slice=eq_index)
        ts = work["equilibrium.time_slice"][eq_index]

    leaves = {name: _leaf(ts, name) for name in (
        "psi", "rho_tor_norm", "q", "f", "dpressure_dpsi", "f_df_dpsi", "gm1", "gm5",
        "volume", "trapped_fraction")}
    missing = [n for n in ("psi", "rho_tor_norm", "q", "f", "dpressure_dpsi", "f_df_dpsi",
                           "gm1", "gm5", "volume") if leaves[n] is None]
    if missing:
        raise ValueError(f"equilibrium slice {eq_index} lacks {', '.join(missing)}")
    size = leaves["psi"].size
    if any(leaves[n].size != size for n in ("rho_tor_norm", "q", "f", "gm5", "volume")):
        raise ValueError(f"equilibrium slice {eq_index}: profiles_1d leaves differ in length")

    psi = leaves["psi"]
    psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
    gq = ts["global_quantities"]
    ip = float(gq["ip"])

    f = leaves["f"]
    dp = leaves["dpressure_dpsi"] / to_radian
    ffp = leaves["f_df_dpsi"] / to_radian
    j_dot_b = -(f * dp + ffp * leaves["gm5"] / (MU0 * f))
    # Fix the overall sign by the current, not by an assumed COCOS.
    from scipy.integrate import trapezoid

    # Check the converted derivatives themselves, not only psi's unit: the
    # Grad-Shafranov current int <j_phi/R> dV / 2 pi, with
    # <j_phi/R> = -(p' + FF' <1/R^2> / mu0), must reproduce ip.
    gs_current = abs(trapezoid(-(dp + ffp * leaves["gm1"] / MU0), leaves["volume"])) / (2 * np.pi)
    ratio = gs_current / abs(ip)
    if not abs(ratio - 1.0) <= gs_tolerance:
        raise ValueError(
            f"equilibrium slice {eq_index}: the Grad-Shafranov current from p' and FF' is "
            f"{ratio:.3g} x ip; the psi derivatives are not in the unit psi was detected in"
        )

    j_dot_b = j_dot_b * np.sign(ip) * np.sign(trapezoid(j_dot_b, leaves["volume"]))

    extent = _radial_extent(work, eq_index, size)
    if extent is None:
        raise ValueError(f"equilibrium slice {eq_index}: r_inboard/r_outboard underivable")
    r_in, r_out = extent

    grid = leaves["rho_tor_norm"]
    t_e = _kinetic_on(ods, cp_index, "electrons.temperature", grid)
    n_e = _kinetic_on(ods, cp_index, "electrons.density_thermal", grid)
    if n_e is None:
        n_e = _kinetic_on(ods, cp_index, "electrons.density", grid)
    if t_e is None or n_e is None:
        raise ValueError(f"core_profiles slice {cp_index} lacks electron T_e or n_e on rho_tor_norm")

    keep = np.isfinite(t_e) & np.isfinite(n_e) & (t_e > 0.0) & (n_e > 0.0)
    if keep.sum() < 3:
        raise ValueError("fewer than three surfaces carry positive T_e and n_e")
    first, last = int(np.argmax(keep)), int(size - 1 - np.argmax(keep[::-1]))
    span = slice(first, last + 1)
    if not np.all(keep[span]):
        raise ValueError("the kinetic profiles have interior gaps; refusing to bridge them")

    # Share of the Grad-Shafranov current (already checked against ip above)
    # carried by surfaces outside the profiles' support; always computable,
    # unlike j_tor/area, which EFIT products do not store.
    current_density = -(dp + ffp * leaves["gm1"] / MU0)
    total = trapezoid(current_density, leaves["volume"])
    inside = trapezoid(current_density[span], leaves["volume"][span])
    excluded = float(1.0 - inside / total) if total else None

    jbs = None
    label = "none"
    if bootstrap != "none":
        if bootstrap_z_eff is None:
            raise ValueError("bootstrap_z_eff is required with a bootstrap model")
        from vaft.omas.neoclassical import compute_bootstrap_current

        result = compute_bootstrap_current(ods, model=bootstrap, time_slice=cp_index,
                                           z_eff=float(bootstrap_z_eff))
        jbs_grid = np.interp(grid, result.rho_tor_norm, np.abs(result.parallel_current))
        jbs = (np.sign(j_dot_b) * jbs_grid)[span]
        label = f"{bootstrap}@Zeff={float(bootstrap_z_eff):g}"

    provenance = {
        "eq_index": int(eq_index),
        "cp_index": int(cp_index),
        "psi_per_radian": bool(detected),
        "j_dot_b": "-(F p' + F F' <B^2>/(mu0 F)), sign of I_p",
        "psi_norm_range": f"{psi_norm[first]:.3f}-{psi_norm[last]:.3f}",
        "gs_current_ratio": float(ratio),
        "excluded_current_fraction": excluded,
        **(source or {}),
    }
    trapped = leaves["trapped_fraction"]
    return FluxSurfaceState(
        time=float(time),
        psi_norm=psi_norm[span],
        volume=leaves["volume"][span],
        T_e=t_e[span],
        n_e=n_e[span],
        q=leaves["q"][span],
        r_inboard=r_in[span],
        r_outboard=r_out[span],
        j_dot_b=j_dot_b[span],
        b2_average=leaves["gm5"][span],
        I_p=ip,
        trapped_fraction=None if trapped is None or trapped.size != size else trapped[span],
        j_bootstrap_dot_b=jbs,
        bootstrap_model=label,
        source=provenance,
    )
