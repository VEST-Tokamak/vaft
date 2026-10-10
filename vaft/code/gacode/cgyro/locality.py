"""Locality QA for local (flux-tube) CGYRO runs: is the turbulence small enough? (#1354)

A local gyrokinetic run assumes the turbulence lives on scales much smaller than the
scales on which the equilibrium varies (``rho* -> 0``). VEST has weak scale separation
(``a/rho_s ~ 30-50``), and a converged nonlinear radial box ``L_x ~ 30-45 rho_s`` can be a
large fraction of the minor radius. That alone does **not** invalidate a local run: the
box is a numerical domain, and what has to be small is the *turbulence*. So this module
keeps the two questions apart:

``box adequacy``
    Is the numerical box large enough for the turbulence it contains?
    ``l_corr / L_x`` -- the radial correlation length against the box. A box several
    correlation lengths wide lets eddies and zonal flows develop; a box comparable to
    ``l_corr`` truncates them (the runaway seen on 39916 r/a 0.8 with ``L_x = 15 rho_s``).
    Box-size *convergence* (the flux unchanged between two boxes) is separate and is
    shown by running two boxes; this ratio is a per-run screen.

``locality``
    Is the turbulence small compared with the equilibrium?
    ``epsilon_local = l_corr / min(L_Ti, L_Te, L_n, L_q)`` -- the measured radial
    correlation length against the shortest profile scale length at the surface.
    ``rho*``, ``L_x/a`` and the distance to the plasma edge are reported beside it, as
    context, not as the verdict.

``l_corr`` is measured from CGYRO's own output, not inferred: the radial spectrum
``S(k_x) = <|phi(k_x, k_y)|^2>`` over ``k_y > 0``, the outboard midplane and a time
window (``bin.cgyro.kxky_phi``) is transformed into the radial autocorrelation
``C(dx)``, and ``l_corr`` is where ``C`` first falls to ``1/e``. That is the standard
definition for gyrokinetic radial correlation lengths (e.g. the GYRO/CGYRO literature uses
the same spectrum-based autocorrelation); the integral scale is reported as well.

The thresholds are **heuristics**, named and stored with every verdict so a reader can
re-judge: ``epsilon_local < 0.1`` local approximation well satisfied, ``< 0.3`` marginal,
above that questionable; ``l_corr / L_x < 0.25`` box adequate, ``< 0.5`` marginal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "BOX_THRESHOLDS",
    "LOCALITY_THRESHOLDS",
    "locality_report",
    "radial_correlation_length",
    "scale_lengths",
]

#: epsilon_local below which the local approximation is called well satisfied / marginal.
LOCALITY_THRESHOLDS: Mapping[str, float] = {"ok": 0.1, "marginal": 0.3}
#: l_corr / L_x below which the box is called adequate / marginal.
BOX_THRESHOLDS: Mapping[str, float] = {"ok": 0.25, "marginal": 0.5}


def _verdict(value: Optional[float], thresholds: Mapping[str, float],
             labels: tuple[str, str, str]) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value < thresholds["ok"]:
        return labels[0]
    if value < thresholds["marginal"]:
        return labels[1]
    return labels[2]


def scale_lengths(local: Any) -> dict[str, Optional[float]]:
    """Equilibrium/profile scale lengths at the surface, in units of the minor radius.

    ``L_X/a = 1/|a dlnX/dr|`` from the same gradients CGYRO was given (``DLNTDR``,
    ``DLNNDR``); ``L_q/a = r/(a s)`` from the magnetic shear. A flat profile (zero
    gradient) has no finite scale and is reported ``None`` rather than infinity.
    Ions are taken as the main ion (the first species with ``Z > 0``).
    """
    species = local.species
    z = np.asarray(species["Z"], dtype=float)
    electron = int(np.flatnonzero(z < 0)[0])
    ion = int(np.flatnonzero(z > 0)[0])

    def length(gradient: float) -> Optional[float]:
        gradient = abs(float(gradient))
        return None if gradient == 0.0 else 1.0 / gradient

    shear = abs(float(local.geometry["S"]))
    return {
        "L_Ti": length(species["DLNTDR"][ion]),
        "L_Te": length(species["DLNTDR"][electron]),
        "L_n": length(species["DLNNDR"][electron]),
        "L_q": None if shear == 0.0 else float(local.geometry["RMIN"]) / shear,
    }


def _kxky_phi(directory: Path, grid: Mapping[str, Any], hiprec: bool) -> Optional[np.ndarray]:
    path = directory / "bin.cgyro.kxky_phi"
    if not path.is_file():
        return None
    data = np.fromfile(path, dtype=np.complex128 if hiprec else np.complex64)
    per_step = grid["n_radial"] * grid["theta_plot"] * grid["n_n"]
    steps = data.size // per_step
    if steps < 1:
        return None
    return data[: per_step * steps].reshape(
        (grid["n_radial"], grid["theta_plot"], grid["n_n"], steps), order="F")


def radial_correlation_length(
    outputs: Any, window: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """Radial correlation length of the potential, in ``rho_s``, from ``kxky_phi``.

    Parameters
    ----------
    outputs
        A nonlinear :class:`~vaft.code.gacode.cgyro.outputs.CgyroOutputs`; its directory
        must hold ``bin.cgyro.kxky_phi``.
    window
        ``(t0, t1)`` in ``a/c_s``; the last half of the run when omitted.

    Returns
    -------
    dict
        ``l_corr`` (1/e crossing of the autocorrelation), ``l_integral`` (integral scale
        to the first zero), ``window``, ``n_samples``, ``method``; or ``{"reason": ...}``
        when it cannot be measured.
    """
    grid = getattr(outputs, "grid", None)
    if not grid or int(grid.get("n_n", 1)) < 2:
        return {"reason": "not a multi-mode (nonlinear) run"}
    hiprec = bool((outputs.equilibrium or {}).get("hiprec_flag"))
    field = _kxky_phi(Path(outputs.directory), grid, hiprec)
    if field is None:
        return {"reason": "no bin.cgyro.kxky_phi in the run"}
    if hasattr(outputs, "align_records"):
        field = outputs.align_records(field)
        time = np.asarray(outputs.time, dtype=float)[: field.shape[-1]]
    else:
        time = np.asarray(outputs.time, dtype=float)[: field.shape[-1]]
        field = field[..., : time.size]
    if window is None:
        window = (0.5 * float(time[-1]), float(time[-1]))
    mask = (time >= window[0]) & (time <= window[1])
    if np.count_nonzero(mask) < 2:
        return {"reason": f"fewer than two kxky_phi samples in {tuple(window)}"}

    length = abs(float(grid["length"]))                     # L_x in rho_s
    p = np.asarray(grid["p"], dtype=float)
    # S(k_x): turbulent modes only (n >= 1; n = 0 is the zonal flow), averaged over the
    # stored poloidal angles and the time window.
    spectrum = np.mean(np.abs(field[:, :, 1:, mask]) ** 2, axis=(1, 3)).sum(axis=1)
    if not np.any(spectrum > 0):
        return {"reason": "zero turbulent amplitude in the window"}
    kx = 2.0 * np.pi * p / length
    dx = np.linspace(0.0, 0.5 * length, 513)
    correlation = (spectrum[None, :] * np.cos(np.outer(dx, kx))).sum(axis=1)
    correlation /= correlation[0]
    below = np.flatnonzero(correlation <= np.exp(-1.0))
    l_corr = None
    if below.size:
        j = int(below[0])
        x0, x1, c0, c1 = dx[j - 1], dx[j], correlation[j - 1], correlation[j]
        l_corr = float(x0 + (np.exp(-1.0) - c0) * (x1 - x0) / (c1 - c0))
    zero = np.flatnonzero(correlation <= 0.0)
    stop = int(zero[0]) if zero.size else dx.size
    integrate = getattr(np, "trapezoid", None) or np.trapz
    l_integral = float(integrate(correlation[:stop], dx[:stop]))
    return {
        "l_corr": l_corr,
        # The autocorrelation never fell to 1/e inside half the (periodic) box: the box
        # truncates the turbulence, and l_corr is only known to exceed L_x/2.
        "l_corr_lower_bound": None if l_corr is not None else 0.5 * length,
        "l_integral": l_integral,
        "window": [float(window[0]), float(window[1])],
        "n_samples": int(np.count_nonzero(mask)),
        "method": "1/e crossing of the radial autocorrelation of phi (n>=1, theta_plot, "
                  "time-averaged), from bin.cgyro.kxky_phi",
        "box_rho_s": length,
    }


def locality_report(
    local: Any, outputs: Any = None, *, window: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """Locality and box-adequacy QA for one CGYRO run.

    Works for linear runs too (``rho*``, ``L_x/a`` and the scale lengths; no correlation
    length, so ``locality`` is ``unknown`` there by construction).

    Returns
    -------
    dict
        ``rho_star``, ``L_x_rho_s``, ``L_x_over_a``, ``l_corr_rho_s``, ``l_corr_over_a``,
        ``l_corr_over_L_x``, ``scale_lengths_over_a``, ``limiting_scale``,
        ``epsilon_local``, ``distance_to_edge_over_a``, ``l_corr_over_edge_distance``,
        ``verdict`` (``locality``, ``box``), ``thresholds`` and ``correlation``.
    """
    norm = getattr(local, "normalisation", None)
    rho_star = None if norm is None else abs(float(norm.gyroradius)) / float(norm.minor_radius)
    grid = getattr(outputs, "grid", None) or {}
    box = abs(float(grid["length"])) if grid.get("length") else None
    scales = scale_lengths(local)
    finite = {k: v for k, v in scales.items() if v is not None}
    limiting = min(finite, key=finite.get) if finite else None

    correlation: dict[str, Any] = {"reason": "no outputs given"}
    if outputs is not None:
        correlation = radial_correlation_length(outputs, window)
    l_corr = correlation.get("l_corr")
    bound = correlation.get("l_corr_lower_bound")
    measured = l_corr if l_corr is not None else bound
    l_corr_a = None if (measured is None or rho_star is None) else measured * rho_star
    epsilon = None if (l_corr_a is None or limiting is None) else l_corr_a / finite[limiting]
    edge = 1.0 - float(local.r_over_a)
    box_ratio = None if (measured is None or not box) else measured / box
    box_verdict = _verdict(box_ratio, BOX_THRESHOLDS,
                           ("box_adequate", "box_marginal", "box_too_small"))
    locality_verdict = _verdict(epsilon, LOCALITY_THRESHOLDS,
                                ("local_ok", "local_marginal", "local_questionable"))
    if l_corr is None and bound is not None:
        # Only a lower bound: the box verdict is certain, the locality one only when the
        # bound already crosses a threshold.
        box_verdict = "box_too_small"
        if locality_verdict == "local_ok":
            locality_verdict = "unknown"
    return {
        "rho_star": rho_star,
        "a_over_rho_s": None if not rho_star else 1.0 / rho_star,
        "L_x_rho_s": box,
        "L_x_over_a": None if (box is None or rho_star is None) else box * rho_star,
        "l_corr_rho_s": l_corr,
        "l_corr_is_lower_bound": bool(l_corr is None and bound is not None),
        "l_corr_lower_bound_rho_s": bound,
        "l_corr_over_a": l_corr_a,
        "l_corr_over_L_x": box_ratio,
        "scale_lengths_over_a": scales,
        "limiting_scale": limiting,
        "epsilon_local": epsilon,
        "distance_to_edge_over_a": edge,
        "l_corr_over_edge_distance": None if l_corr_a is None else l_corr_a / edge,
        "verdict": {"locality": locality_verdict, "box": box_verdict},
        "thresholds": {"locality": dict(LOCALITY_THRESHOLDS), "box": dict(BOX_THRESHOLDS),
                       "kind": "heuristic"},
        "correlation": correlation,
    }
