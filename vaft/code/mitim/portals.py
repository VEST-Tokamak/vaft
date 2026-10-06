"""PORTALS/powertorch synthetic closed loop (#1588 stage C1).

A flux-matching solver is only trustworthy if it returns a profile it was built
to reach. This drives MITIM with a *manufactured* problem: the electron source is
constructed so that the given (true) profile is flux-matched by the chosen
transport model, the solver starts from perturbed gradients, and the recovered
``a/L_Te`` is compared with the true one. The source is built inside MITIM, in
MITIM's own geometry (``V'`` and its volume integral), so any residual at the
true profile is MITIM's discretisation, which the result reports.

Sign convention (#1588 section 11): ``R = Q_model - Q_required``. MITIM's
``powerstate`` uses ``S = P_target - P_transport = -R``. A start steeper than the
truth carries more flux than required, so ``R > 0`` there; the result reports
``R`` at the start so the convention is checked on every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .availability import MITIMAvailability
from .config import MITIMConfig, MITIMResult

__all__ = ["ClosedLoopReport", "closed_loop_report", "run_portals_closed_loop"]


@dataclass(frozen=True)
class ClosedLoopReport:
    """What a closed loop established, from the driver's ``result.json``.

    ``manufacture_error`` is max |required - model| / |model| at the true profile
    (the source construction); ``recovery_error`` is max |a/L_Te recovered - true| /
    |true|; ``final_flux_error`` the same for the fluxes at the end. ``sign_ok`` checks
    the residual convention: for a transport model whose flux rises with a/L_Te,
    R = Q_model - Q_required has the sign of a/L_Te(start) - a/L_Te(true) at every
    radius. (A hollow profile has a/L_Te < 0 inside, so "steeper" is not "larger".)
    """

    r_over_a: tuple[float, ...]
    manufacture_error: float
    recovery_error: float
    final_flux_error: float
    sign_ok: bool


def _max_rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(scale > 0, np.abs(a - b) / np.where(scale > 0, scale, 1.0), 0.0)))


def closed_loop_report(result: dict, perturbation: float) -> ClosedLoopReport:
    """Reduce a driver ``result.json`` to :class:`ClosedLoopReport`."""
    residual = np.asarray(result["model_minus_required_at_start_MWm2"], dtype=float)
    offset = (np.asarray(result["aLte_start"], dtype=float)
              - np.asarray(result["aLte_true"], dtype=float))
    sign_ok = bool(np.all(np.sign(residual) == np.sign(offset)) and np.all(offset != 0))
    return ClosedLoopReport(
        r_over_a=tuple(float(x) for x in result["r_over_a"]),
        manufacture_error=_max_rel(result["target_after_manufacture_MWm2"],
                                   result["transport_after_manufacture_MWm2"]),
        recovery_error=_max_rel(result["aLte_recovered"], result["aLte_true"]),
        final_flux_error=_max_rel(result["final_model_MWm2"], result["final_required_MWm2"]),
        sign_ok=sign_ok,
    )


def run_portals_closed_loop(
    profile: Any,
    r_over_a,
    workdir: str | Path,
    config: MITIMConfig | None = None,
    *,
    model: str = "analytic",
    chi_e: float = 1.0,
    chi_i: float = 1.0,
    perturbation: float = 0.3,
    tglf_code_settings: str = "SAT3",
    tglf_extra_options: Optional[dict] = None,
    neo_extra_options: Optional[dict] = None,
    initial_training: int = 5,
    maximum_iterations: int = 10,
    availability: Optional[MITIMAvailability] = None,
) -> tuple[MITIMResult, Optional[ClosedLoopReport]]:
    """Manufacture a flux-matched problem from ``profile`` and check MITIM recovers it.

    Parameters
    ----------
    profile
        The true state, a :class:`~vaft.code.gacode._profiles.GACODEProfile` [-].
    r_over_a
        Predicted radii, converted to MITIM's rho_tor_norm with VAFT's bridge [-].
    model
        ``"analytic"``: MITIM's conductive diffusion model with constant ``chi_e``,
        ``chi_i`` [m^2/s], evaluated in-memory by powertorch and flux-matched by its
        root solver. ``"tglf_neo"``: a full PORTALS run (TGLF + NEO, Bayesian
        optimisation) with ``predicted_roa`` = ``r_over_a``; TGLF and NEO settings as
        given (#1744/#1757 alignment: NKY 12, NMODES 2, USE_MHD_RULE, ROTATION_MODEL 1).
    perturbation
        ``analytic``: the solver starts from the true a/L_Te times (1 + perturbation).
        ``tglf_neo``: the same, written by PORTALS's own powerstate as the start file;
        only the node gradients move, so Te outside the last radius (the anchor) stays
        the truth's [-].

    Returns
    -------
    (MITIMResult, ClosedLoopReport or None)
        The report is None when the driver did not finish.
    """
    from ..gacode._input_gacode import write_input_gacode
    from .coordinates import rho_tor_norm_at
    from .runner import _sha256, run_mitim_driver

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    r_over_a = [float(r) for r in r_over_a]
    rho = [float(x) for x in rho_tor_norm_at(profile, r_over_a)]
    input_path = write_input_gacode(profile, workdir / "input.gacode")
    arguments = {"input_gacode": str(input_path), "rho_tor_norm": rho, "r_over_a": r_over_a,
                 "model": model, "chi_e": float(chi_e), "chi_i": float(chi_i),
                 "perturbation": float(perturbation),
                 "input_gacode_sha256": _sha256(input_path)}
    if model == "tglf_neo":
        arguments.update(folder=str(workdir / "portals"), tglf_code_settings=tglf_code_settings,
                         tglf_extra_options=dict(tglf_extra_options or {}),
                         neo_extra_options=dict(neo_extra_options or {}),
                         initial_training=int(initial_training),
                         maximum_iterations=int(maximum_iterations))
        driver = "portals_tglf_closed_loop"
    elif model == "analytic":
        driver = "portals_closed_loop"
    else:
        raise ValueError(f"model must be 'analytic' or 'tglf_neo', got {model!r}")
    result = run_mitim_driver(driver, arguments, workdir, config, availability=availability)
    report = closed_loop_report(result.result, perturbation) if result.ok else None
    return result, report
