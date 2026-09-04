"""Quasi-static eddy-current evolution of VEST equilibria with TokaMaker.

Reruns a shot as a chain of static solves with self-consistent wall eddy
currents: at each slice the measured coil currents and Ip target are applied
and the previous flux state feeds TokaMaker's backward-Euler wall term via
``set_psi_dt(psi0, dt)`` (paper eq. 14; CUTE pulse-design pattern). One
TokaMaker instance lives across the whole shot and is reset once at the end.

Two modes share the loop:

- **plasma** (default): ``solve()`` per slice with measured-current drive and
  the per-slice Ip target; converged slices export ``g<shot>.<ms>`` files that
  are merged into a multi-slice ``equilibrium`` IDS.
- **vacuum** (``evolve_vacuum=True``): ``vac_solve()`` per slice with no
  plasma and no targets — the wall responds to the coil waveforms alone
  (the issue-#190 vacuum-window benchmark drive).

Per slice the net toroidal current of every vessel region is recorded, and
optional ``evolve_field_probes`` get B/psi evaluations (used to compare
against measured magnetics). Slice failures follow ``evolve_on_failure``:
``"continue"`` records the failure and keeps stepping from the last converged
flux state (the elapsed ``dt`` then spans the gap), ``"stop"`` aborts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ._oft import get_oft_env, import_oft
from .config import (
    TokaMakerConfig,
    TokaMakerEvolutionInputs,
    TokaMakerEvolutionResult,
    TokaMakerStepRecord,
)
from .mesh import build_tokamaker_mesh
from .outputs import EVOLUTION_SIDECAR_NAME, _merge_equilibrium
from .runner import _apply_profiles, _apply_vsc, _configure_tokamaker, _json_safe

_log = logging.getLogger(__name__)

# Default nonlinear tolerance for eddy-coupled steps. The wall term gives the
# quasi-static fixed point a discretization-level residual floor (~2e-6 on the
# VEST mesh, measured on shot 39915) just above the solver's static default of
# 1e-6, while the equilibrium itself is frozen to 5+ digits long before —
# tightening the tolerance past the floor only burns iterations and reports
# false failures. An explicit ``TokaMakerConfig.nl_tol`` always wins.
EVOLVE_NL_TOL = 1.0e-5


def _vessel_currents(mygs, psi, cond_regions: dict[str, dict]) -> dict[str, float]:
    """Net toroidal current [A] per conductor region for the given flux state."""
    if not cond_regions:
        return {}
    _mask, current_density = mygs.get_conductor_currents(psi)
    return {
        name: float(mygs.area_integral(current_density, reg_mask=entry["reg_id"]))
        for name, entry in sorted(cond_regions.items())
    }


def _probe_fields(mygs, probes) -> dict[str, list[float]]:
    """Evaluate B_r, B_z [T] and psi [Wb/rad] at the probe points."""
    if not probes:
        return {}
    b_eval = mygs.get_field_eval("B")
    psi_eval = mygs.get_field_eval("psi")
    br, bz, psi = [], [], []
    for r, z in probes:
        b = np.asarray(b_eval.eval(np.array([float(r), float(z)])), dtype=float)
        br.append(float(b[0]))
        bz.append(float(b[2]))
        psi.append(float(np.asarray(psi_eval.eval(np.array([float(r), float(z)])), dtype=float)[0]))
    return {"br": br, "bz": bz, "psi": psi}


def run_tokamaker_evolution(
    inputs: TokaMakerEvolutionInputs, config: TokaMakerConfig
) -> TokaMakerEvolutionResult:
    """March the quasi-static evolution and collect per-slice outputs."""
    oft = import_oft()
    env = get_oft_env(config.nthreads)
    base = inputs.base
    if not base.mesh_file.is_file():
        build_tokamaker_mesh(base.geometry, base.mesh_file, config)

    shot = int(base.shot)
    probes = list(config.evolve_field_probes or ())
    records: list[TokaMakerStepRecord] = []
    aborted_error = ""

    mygs = oft.TokaMaker(env)
    try:
        cond_regions = _configure_tokamaker(oft, mygs, base, config)
        if config.nl_tol is None:
            # settings changed after setup() need an explicit push
            mygs.settings.nl_tol = EVOLVE_NL_TOL
            mygs.update_settings()
        _apply_vsc(mygs, config)
        if not inputs.vacuum:
            _apply_profiles(oft, mygs, config)

        psi0 = None          # unnormalized flux of the last converged slice
        t_ref = inputs.times[0]  # its time
        for index, time in enumerate(inputs.times):
            ms = int(round(time * 1000))
            currents = {
                name: float(wave[index]) for name, wave in inputs.coil_waveforms.items()
            }
            mygs.set_coil_currents(currents)

            converged = False
            error = ""
            stats: dict[str, Any] = {}
            gfile = None
            try:
                if inputs.vacuum:
                    if psi0 is not None and config.evolve_eddy:
                        mygs.set_psi_dt(psi0, time - t_ref)
                    psi_new = mygs.vac_solve()
                    mygs.set_psi(psi_new)   # vac_solve does not update internal state
                    psi0, t_ref = psi_new, time
                    converged = True
                else:
                    targets = dict(base.targets)
                    targets["Ip"] = float(inputs.ip_targets[index])
                    mygs.set_targets(**targets)
                    if psi0 is None:
                        mygs.init_psi(
                            config.init_r0, config.init_z0, config.init_a0,
                            config.init_kappa, config.init_delta,
                        )
                    elif config.evolve_eddy:
                        mygs.set_psi_dt(psi0, time - t_ref)
                    mygs.solve()
                    psi0, t_ref = mygs.get_psi(False), time
                    converged = True
                    stats = dict(mygs.get_stats())
                    gfile = base.workdir / f"g{shot:06d}.{ms:05d}"
                    mygs.save_eqdsk(
                        str(gfile),
                        nr=config.eqdsk_nr,
                        nz=config.eqdsk_nz,
                        lcfs_pad=config.eqdsk_lcfs_pad,
                        run_info=f"# {shot} {ms}ms",
                        cocos=config.eqdsk_cocos,
                    )
            except Exception as exc:
                error = str(exc)
                _log.warning(
                    "TokaMaker evolution slice %d (t=%.4f s) failed: %s", index, time, exc
                )
                if psi0 is not None:
                    # a failed solve() leaves its diverged iterate in the solver;
                    # restore the last converged flux so "continue" mode really
                    # steps from the last good state instead of cascading
                    try:
                        mygs.set_psi(psi0)
                    except Exception:  # pragma: no cover - defensive
                        _log.warning("Could not restore the last converged flux state",
                                     exc_info=True)

            vessel_currents: dict[str, float] = {}
            probe_fields: dict[str, list[float]] = {}
            if converged and psi0 is not None:
                vessel_currents = _vessel_currents(mygs, psi0, cond_regions)
                probe_fields = _probe_fields(mygs, probes)

            records.append(TokaMakerStepRecord(
                index=index,
                time=time,
                converged=converged,
                error=error,
                gfile=gfile,
                stats=stats,
                coil_currents_A=currents,
                vessel_currents_A=vessel_currents,
                probe_fields=probe_fields,
            ))

            if not converged and config.evolve_on_failure == "stop":
                aborted_error = error or f"slice {index} failed"
                break

        if psi0 is not None:
            # leave the solver free of eddy terms for any later static use
            mygs.set_psi_dt(np.zeros_like(np.asarray(psi0)), -1.0)
    finally:
        try:
            mygs.reset()
        except Exception:  # pragma: no cover - defensive
            _log.warning("TokaMaker reset failed after evolution", exc_info=True)

    sidecar_payload = {
        "shot": shot,
        "vacuum": inputs.vacuum,
        "times": list(inputs.times),
        "probes": [[float(r), float(z)] for r, z in probes],
        "steps": [
            {
                "index": rec.index,
                "time": rec.time,
                "converged": rec.converged,
                "error": rec.error,
                "gfile": rec.gfile.name if rec.gfile else None,
                "stats": dict(rec.stats),
                "coil_currents_A": dict(rec.coil_currents_A),
                "vessel_currents_A": dict(rec.vessel_currents_A),
                "probe_fields": dict(rec.probe_fields),
            }
            for rec in records
        ],
    }
    sidecar_file = base.workdir / EVOLUTION_SIDECAR_NAME
    sidecar_file.write_text(json.dumps(_json_safe(sidecar_payload), indent=2, sort_keys=True), encoding="utf-8")

    gfiles = tuple(rec.gfile for rec in records if rec.gfile is not None)
    gtimes = tuple(rec.time for rec in records if rec.gfile is not None)
    ods, merge_error = _merge_equilibrium(gfiles, gtimes)

    n_failed = sum(1 for rec in records if not rec.converged)
    all_ran = len(records) == len(inputs.times)
    scalars: dict[str, Any] = {
        "n_steps": len(records),
        "n_failed": n_failed,
        "vacuum": inputs.vacuum,
    }
    if merge_error:
        scalars["_merge_error"] = merge_error

    return TokaMakerEvolutionResult(
        returncode=0 if (n_failed == 0 and all_ran) else 1,
        workdir=base.workdir,
        times=tuple(inputs.times),
        steps=tuple(records),
        gfiles=gfiles,
        sidecar_file=sidecar_file,
        mesh_file=base.mesh_file if base.mesh_file.is_file() else None,
        error=aborted_error,
        ods=ods,
        scalars=scalars,
    )
