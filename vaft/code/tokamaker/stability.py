"""Wall eigenmodes and vertical-stability growth rates via TokaMaker.

Two in-process entry points, both requiring a vessel-conductor mesh
(``TokaMakerConfig.include_vessel``):

- :func:`run_tokamaker_wall_eigenmodes` — resistive wall L/R eigenmodes
  (``eig_wall``). Needs only the mesh + finite-element setup, no equilibrium:
  eigenvalues are ``1/tau`` in 1/s, ascending, so ``tau_wall_s[0]`` is the
  longest wall time constant.
- :func:`run_tokamaker_vertical_stability` — a full forward static solve
  (optionally held by the VSC pair + ``v0_target``) followed by ``eig_td``:
  the returned growth rate is ``gamma = -eig_vals[0, 0]`` [1/s], positive
  when the equilibrium is vertically unstable.

Eigenvectors and the mesh arrays are saved to an ``.npz`` next to the JSON
sidecar so mode patterns can be re-plotted (``plt.tripcolor``) without a live
OpenFUSIONToolkit handle. On OFT builds without ARPACK both eigensolvers
silently return zeros; that is detected and reported as a failure with an
actionable message rather than passed through as physics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ._oft import get_oft_env, import_oft
from .config import TokaMakerConfig, TokaMakerInputs, TokaMakerStabilityResult
from .mesh import build_tokamaker_mesh
from .outputs import STABILITY_SIDECAR_NAME
from .runner import _apply_profiles, _apply_vsc, _configure_tokamaker, _json_safe

_log = logging.getLogger(__name__)

EIG_WALL_FILE = "tokamaker_eig_wall.npz"
EIG_TD_FILE = "tokamaker_eig_td.npz"

_NO_ARPACK_MESSAGE = (
    "eigenvalue solve returned all zeros — this OpenFUSIONToolkit build has no "
    "ARPACK support. Rebuild OFT with ARPACK (OFT_BUILD_ARPACK) to use "
    "eig_wall/eig_td."
)


def _require_vessel(config: TokaMakerConfig, what: str) -> None:
    if not config.include_vessel:
        raise ValueError(
            f"{what} requires vessel conductor regions: set "
            "TokaMakerConfig.include_vessel=True (and rebuild inputs/mesh)."
        )


def _converged_eigenvalues(eig_vals, context: str) -> np.ndarray:
    """Drop unconverged (all-zero) eigenvalue rows, preserving order.

    ARPACK leaves exact-zero rows for modes it failed to converge; all rows
    zero means the OFT build has no ARPACK at all. Row order (ascending real
    part) is preserved so the leading entry keeps its meaning.
    """
    vals = np.asarray(eig_vals, dtype=float)
    converged = np.any(vals != 0.0, axis=1)
    if not converged.any():
        raise RuntimeError(_NO_ARPACK_MESSAGE)
    if not converged.all():
        _log.warning(
            "%s: only %d of %d eigenvalues converged; ignoring the rest",
            context, int(converged.sum()), len(vals),
        )
    return vals[converged]


def _update_sidecar(workdir: Path, key: str, payload: dict[str, Any]) -> Path:
    """Merge one section into the shared stability sidecar."""
    path = workdir / STABILITY_SIDECAR_NAME
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    existing[key] = _json_safe(payload)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return path


def _save_modes(path: Path, eig_vals, eig_vecs, mygs) -> None:
    np.savez_compressed(
        path,
        eig_vals=np.asarray(eig_vals, dtype=float),
        eig_vecs=np.asarray(eig_vecs, dtype=float),
        mesh_r=np.asarray(mygs.r, dtype=float),
        mesh_lc=np.asarray(mygs.lc),
        mesh_reg=np.asarray(mygs.reg),
    )


def run_tokamaker_wall_eigenmodes(
    inputs: TokaMakerInputs, config: TokaMakerConfig
) -> TokaMakerStabilityResult:
    """Compute resistive wall L/R eigenmodes (no equilibrium solve needed)."""
    _require_vessel(config, "run_tokamaker_wall_eigenmodes")
    oft = import_oft()
    env = get_oft_env(config.nthreads)
    if not inputs.mesh_file.is_file():
        build_tokamaker_mesh(inputs.geometry, inputs.mesh_file, config)

    returncode = 1
    error = ""
    tau_wall: tuple[float, ...] = ()
    eig_file = None

    mygs = oft.TokaMaker(env)
    try:
        _configure_tokamaker(oft, mygs, inputs, config)
        eig_vals, eig_vecs = mygs.eig_wall(config.wall_neigs)
        good = _converged_eigenvalues(eig_vals, "eig_wall")
        good = good[good[:, 0] > 0.0]          # wall eigenvalues are 1/tau > 0
        if not len(good):
            raise RuntimeError(_NO_ARPACK_MESSAGE)
        tau_wall = tuple(float(t) for t in 1.0 / good[:, 0])
        eig_file = inputs.workdir / EIG_WALL_FILE
        _save_modes(eig_file, eig_vals, eig_vecs, mygs)
        returncode = 0
    except Exception as exc:
        error = str(exc)
        _log.warning("TokaMaker wall-eigenmode solve failed: %s", exc)
    finally:
        try:
            mygs.reset()
        except Exception:  # pragma: no cover - defensive
            _log.warning("TokaMaker reset failed after eig_wall", exc_info=True)

    payload: dict[str, Any] = {
        "neigs": config.wall_neigs,
        "tau_wall_s": list(tau_wall),
        "tau_wall_max_s": tau_wall[0] if tau_wall else None,
        "converged": returncode == 0,
    }
    if error:
        payload["error"] = error
    stats_file = _update_sidecar(inputs.workdir, "wall", payload)

    scalars: dict[str, Any] = {"tau_wall_s": list(tau_wall)}
    if tau_wall:
        scalars["tau_wall_max_s"] = tau_wall[0]
    return TokaMakerStabilityResult(
        returncode=returncode,
        workdir=inputs.workdir,
        tau_wall_s=tau_wall,
        eig_file=eig_file,
        stats_file=stats_file,
        error=error,
        scalars=scalars,
    )


def run_tokamaker_vertical_stability(
    inputs: TokaMakerInputs, config: TokaMakerConfig
) -> TokaMakerStabilityResult:
    """Forward static solve followed by the n=0 growth-rate eigensolve.

    The static sequence mirrors ``run_tokamaker`` (including the optional VSC
    pair + ``v0_target`` to hold vertically unstable cases); ``eig_td`` then
    yields ``gamma = -eig_vals[0, 0]`` [1/s].
    """
    _require_vessel(config, "run_tokamaker_vertical_stability")
    oft = import_oft()
    env = get_oft_env(config.nthreads)
    if not inputs.mesh_file.is_file():
        build_tokamaker_mesh(inputs.geometry, inputs.mesh_file, config)

    shot = int(inputs.shot)
    ctime = int(round(inputs.time * 1000))
    gpath = inputs.workdir / f"g{shot:06d}.{ctime:05d}"

    returncode = 1
    error = ""
    gamma = None
    eig_file = None
    stats: dict[str, Any] = {}
    gfile = None

    mygs = oft.TokaMaker(env)
    try:
        _configure_tokamaker(oft, mygs, inputs, config)
        _apply_vsc(mygs, config)
        mygs.set_coil_currents(dict(inputs.coil_currents))
        mygs.set_targets(**inputs.targets)
        _apply_profiles(oft, mygs, config)
        mygs.init_psi(
            config.init_r0, config.init_z0, config.init_a0,
            config.init_kappa, config.init_delta,
        )
        mygs.solve()
        stats = dict(mygs.get_stats())
        mygs.save_eqdsk(
            str(gpath),
            nr=config.eqdsk_nr,
            nz=config.eqdsk_nz,
            lcfs_pad=config.eqdsk_lcfs_pad,
            run_info=f"# {shot} {ctime}ms",
            cocos=config.eqdsk_cocos,
        )
        gfile = gpath

        eig_vals, eig_vecs = mygs.eig_td(
            omega=config.td_omega,
            neigs=config.td_neigs,
            include_bounds=config.td_include_bounds,
            damping_scale=config.td_damping_scale,
        )
        good = _converged_eigenvalues(eig_vals, "eig_td")
        gamma = float(-good[0, 0])
        eig_file = inputs.workdir / EIG_TD_FILE
        _save_modes(eig_file, eig_vals, eig_vecs, mygs)
        returncode = 0
    except Exception as exc:
        error = str(exc)
        _log.warning(
            "TokaMaker vertical-stability solve failed for shot %s @ %s ms: %s",
            shot, ctime, exc,
        )
    finally:
        try:
            mygs.reset()
        except Exception:  # pragma: no cover - defensive
            _log.warning("TokaMaker reset failed after eig_td", exc_info=True)

    payload: dict[str, Any] = {
        "gamma_s": gamma,
        "stable": None if gamma is None else gamma <= 0.0,
        "neigs": config.td_neigs,
        "omega": config.td_omega,
        "targets": dict(inputs.targets),
        "vsc_coil": config.vsc_coil,
        "stats": stats,
        "converged": returncode == 0,
    }
    if error:
        payload["error"] = error
    stats_file = _update_sidecar(inputs.workdir, "vertical", payload)

    scalars: dict[str, Any] = dict(stats)
    if gamma is not None:
        scalars["gamma_s"] = gamma
        scalars["stable"] = gamma <= 0.0
    return TokaMakerStabilityResult(
        returncode=returncode,
        workdir=inputs.workdir,
        gamma_s=gamma,
        eig_file=eig_file,
        stats_file=stats_file,
        gfile=gfile,
        error=error,
        scalars=scalars,
    )
