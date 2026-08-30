"""Run a TokaMaker forward free-boundary solve on prepared inputs.

Unlike the subprocess adapters there is no external binary: TokaMaker is
driven in-process through its Python API. The subprocess result conventions
are kept — ``returncode`` 0/1 with the solver message in ``error`` — and all
run artefacts (g-file, ``tokamaker_result.json`` sidecar) are written to the
working directory so ``collect_tokamaker_outputs`` can rebuild the result
from disk alone, exactly like the other adapters.

``OFT_env`` is a per-interpreter singleton and TokaMaker holds one mesh at a
time, so the runner always releases the solver with ``reset()`` in a
``finally`` block; sequential runs and same-process scans then work in one
Python kernel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ._oft import get_oft_env, import_oft
from .config import TokaMakerConfig, TokaMakerInputs, TokaMakerResult
from .mesh import build_tokamaker_mesh
from .outputs import _parse_gfile, collect_tokamaker_outputs

_log = logging.getLogger(__name__)

SIDECAR_NAME = "tokamaker_result.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_sidecar(workdir: Path, payload: dict[str, Any]) -> Path:
    path = workdir / SIDECAR_NAME
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return path


def run_tokamaker(inputs: TokaMakerInputs, config: TokaMakerConfig) -> TokaMakerResult:
    """Execute a forward solve and collect the produced outputs.

    Builds the mesh on a cache miss, then runs the canonical TokaMaker
    sequence (mesh → regions → setup → coil currents → targets → profiles →
    ``init_psi`` → ``solve``) and exports the equilibrium as an EFIT g-file
    named ``g<shot>.<time_ms>`` (COCOS per ``config.eqdsk_cocos``). A failed
    solve is reported through ``result.ok``/``result.error`` rather than
    raised, mirroring the subprocess adapters.
    """
    oft = import_oft()
    env = get_oft_env(config.nthreads)

    if not inputs.mesh_file.is_file():
        build_tokamaker_mesh(inputs.geometry, inputs.mesh_file, config)

    shot = int(inputs.shot)
    ctime = int(round(inputs.time * 1000))
    gpath = inputs.workdir / f"g{shot:06d}.{ctime:05d}"

    returncode = 1
    error = ""
    sidecar: dict[str, Any] = {
        "converged": False,
        "shot": shot,
        "time_s": inputs.time,
        "targets": dict(inputs.targets),
        "coil_currents_A": dict(inputs.coil_currents),
        "f0": inputs.f0,
        "cocos": config.eqdsk_cocos,
    }

    mygs = oft.TokaMaker(env)
    try:
        pts, lc, reg, coil_dict, cond_dict = oft.meshing.load_gs_mesh(str(inputs.mesh_file))
        mygs.setup_mesh(pts, lc, reg)
        mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
        if config.quiet:
            mygs.settings.pm = False
        if config.maxits is not None:
            mygs.settings.maxits = int(config.maxits)
        if config.urf is not None:
            mygs.settings.urf = float(config.urf)
        if config.nl_tol is not None:
            mygs.settings.nl_tol = float(config.nl_tol)
        mygs.setup(order=config.order, F0=inputs.f0)

        mygs.set_coil_currents(dict(inputs.coil_currents))
        mygs.set_targets(**inputs.targets)
        mygs.set_profiles(
            ffp_prof=oft.util.create_power_flux_fun(config.nprof, config.alpha_f_a, config.alpha_f_b),
            pp_prof=oft.util.create_power_flux_fun(config.nprof, config.alpha_p_a, config.alpha_p_b),
        )

        mygs.init_psi(
            config.init_r0, config.init_z0, config.init_a0,
            config.init_kappa, config.init_delta,
        )
        mygs.solve()

        sidecar["converged"] = True
        sidecar["stats"] = mygs.get_stats()
        sidecar["coil_currents_A"] = dict(mygs.get_coil_currents()[0])
        sidecar["o_point"] = mygs.o_point
        sidecar["diverted"] = bool(mygs.diverted)
        mygs.save_eqdsk(
            str(gpath),
            nr=config.eqdsk_nr,
            nz=config.eqdsk_nz,
            lcfs_pad=config.eqdsk_lcfs_pad,
            run_info=f"# {shot} {ctime}ms",
            cocos=config.eqdsk_cocos,
        )
        returncode = 0
    except Exception as exc:
        error = str(exc)
        sidecar["error"] = error
        _log.warning("TokaMaker solve failed for shot %s @ %s ms: %s", shot, ctime, exc)
    finally:
        try:
            mygs.reset()
        except Exception:  # pragma: no cover - defensive
            _log.warning("TokaMaker reset failed; the kernel may need a restart", exc_info=True)

    _write_sidecar(inputs.workdir, sidecar)

    result = collect_tokamaker_outputs(inputs.workdir, config)
    result.returncode = returncode
    result.error = error
    result.mesh_file = inputs.mesh_file if inputs.mesh_file.is_file() else None
    # The collector globs the workdir, which in a reused directory can surface
    # a g-file from an EARLIER run. This run's result must carry exactly the
    # equilibrium it produced — the file written above on success, none at all
    # on failure.
    if returncode == 0:
        if result.gfile != gpath:
            result.gfile = gpath
            result.geqdsk, result.ods, geqdsk_error = _parse_gfile(gpath)
            if geqdsk_error:
                result.scalars["_geqdsk_error"] = geqdsk_error
            else:
                result.scalars.pop("_geqdsk_error", None)
    elif result.gfile is not None:
        result.gfile = None
        result.geqdsk = ()
        result.ods = None
        result.scalars.pop("_geqdsk_error", None)
    return result
