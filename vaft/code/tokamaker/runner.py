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
import os
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
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
#  Shared lifecycle helpers (also used by evolve.py / stability.py)
# --------------------------------------------------------------------------- #
def _configure_tokamaker(oft, mygs, inputs, config: TokaMakerConfig) -> dict[str, dict]:
    """Load the mesh into ``mygs`` and run the common setup sequence.

    Returns a snapshot of the conductor-region entries ``{name: {reg_id, eta,
    ...}}`` taken BEFORE ``setup_regions`` — which mutates the passed
    ``cond_dict`` (vacuum entries are moved out) — so callers can integrate
    per-region eddy currents later.
    """
    pts, lc, reg, coil_dict, cond_dict = oft.meshing.load_gs_mesh(str(inputs.mesh_file))
    cond_regions = {
        str(name): dict(entry)
        for name, entry in cond_dict.items()
        if isinstance(entry, dict) and "eta" in entry
    }
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
    if config.lim_zmax is not None:
        mygs.settings.lim_zmax = float(config.lim_zmax)
        points = neck_limiter_points(inputs.geometry["limiter"], config)
        if len(points):
            path = Path(inputs.workdir) / LIMITER_POINTS_NAME
            try:
                c_path = mygs._oft_env.path2c(str(path))
            except ValueError:
                # OFT caps paths at OFT_PATH_SLEN (200) characters
                import tempfile
                handle, short = tempfile.mkstemp(prefix="oft_lim_", suffix=".dat")
                os.close(handle)
                path = Path(short)
                c_path = mygs._oft_env.path2c(str(path))
            _write_limiter_points(path, points)
            mygs.settings.limiter_file = c_path
    mygs.setup(order=config.order, F0=inputs.f0)
    return cond_regions


LIMITER_POINTS_NAME = "limiter_points.dat"


def neck_limiter_points(limiter: Any, config: TokaMakerConfig) -> np.ndarray:
    """Limiter points on the wall between ``lim_zmax`` and ``neck_limiter_zmax``.

    ``lim_zmax`` removes every mesh limiter node in a cell reaching above it.
    The faces between the main chamber and that height are still wall, so they
    are sampled along the limiter polygon every ``dx_plasma`` and returned as
    an ``(n, 2)`` array of (R, Z) for OFT's explicit limiter points.
    """
    if config.lim_zmax is None or config.neck_limiter_zmax is None:
        return np.zeros((0, 2))
    lo, hi = float(config.lim_zmax), float(config.neck_limiter_zmax)
    if hi <= lo:
        return np.zeros((0, 2))
    poly = np.asarray(limiter, dtype=float)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    points = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / config.dx_plasma)))
        t = np.arange(n + 1)[:, None] / n
        segment = a + t * (b - a)
        keep = (np.abs(segment[:, 1]) > lo) & (np.abs(segment[:, 1]) <= hi)
        points.append(segment[keep])
    points = np.concatenate(points) if points else np.zeros((0, 2))
    return np.unique(np.round(points, 9), axis=0)


def _write_limiter_points(path: Path, points: np.ndarray) -> None:
    """OFT ``limiter_file`` format: the point count, then one ``R Z`` per line."""
    lines = [str(len(points))] + [f"{r:.9f} {z:.9f}" for r, z in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wall_check(mygs, limiter: Any, tolerance: float = 1.0e-3) -> dict[str, Any]:
    """Does the native LCFS stay inside the full limiter polygon?

    The limiting-point search ignores the chambers above ``lim_zmax`` and any
    wall between explicit points, so this is checked on the result instead:
    the solver's own trace of the psi_N = 0.999 surface against the whole
    first wall. Returns ``lcfs_inside_wall`` and the largest excursion outside
    it [m]; both are None when the trace fails.
    """
    from matplotlib.path import Path as PolygonPath

    # just inside the boundary flux: OFT's own boundary statistics trace
    # 1 - pad, because the exact separatrix of a diverted plasma does not close
    try:
        lcfs = mygs.trace_surf(1.0 - 1.0e-3)
    except Exception:  # pragma: no cover - solver-side trace failure
        lcfs = None
    if lcfs is None or len(lcfs) == 0:
        return {"lcfs_inside_wall": None, "lcfs_wall_excursion_m": None}
    lcfs = np.asarray(lcfs, dtype=float)[:, :2]
    poly = np.asarray(limiter, dtype=float)
    outside = ~PolygonPath(poly).contains_points(lcfs)
    excursion = 0.0
    if outside.any():
        from .topology import _min_distance_to_polyline

        excursion = max(
            _min_distance_to_polyline(np.array([r]), np.array([z]), poly[:, 0], poly[:, 1])[0]
            for r, z in lcfs[outside]
        )
    inside = bool(excursion <= tolerance)
    if not inside:
        _log.warning("TokaMaker LCFS leaves the first wall by %.1f mm", excursion * 1e3)
    return {"lcfs_inside_wall": inside, "lcfs_wall_excursion_m": float(excursion)}


def _apply_vsc(mygs, config: TokaMakerConfig) -> None:
    """Wire the Vertical Stability Coil pair when ``config.vsc_coil`` is set.

    The named coil's halves are separate coil sets (see the geometry builder);
    they get gains +1/-1 and the virtual ``'#VSC'`` amplitude is regularized
    toward zero. NOTE: the ``V0`` target this enables is silently ignored by
    TokaMaker whenever isoflux/flux constraints are active — the forward
    adapter never sets those.
    """
    if config.vsc_coil is None:
        return
    parent = str(config.vsc_coil).upper()
    mygs.set_coil_vsc({f"{parent}_U": 1.0, f"{parent}_L": -1.0})
    term = mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=config.vsc_weight)
    mygs.set_coil_reg(reg_terms=[term])


def _apply_profiles(oft, mygs, config: TokaMakerConfig) -> None:
    mygs.set_profiles(
        ffp_prof=oft.util.create_power_flux_fun(config.nprof, config.alpha_f_a, config.alpha_f_b),
        pp_prof=oft.util.create_power_flux_fun(config.nprof, config.alpha_p_a, config.alpha_p_b),
    )


def _save_eqdsk(mygs, path: Path, config: TokaMakerConfig, run_info: str) -> None:
    """Export the current equilibrium as a gEQDSK whose boundary is the LCFS.

    ``truncate_eq=False`` keeps ``lcfs_pad`` for tracing but extrapolates the
    contour back to the true boundary flux. OFT's default (``True``) writes the
    ``1 - lcfs_pad`` surface as RBBBS/ZBBBS and as the boundary flux, so a
    limited plasma never shows wall contact in the export (issue #882 item 1).
    """
    mygs.save_eqdsk(
        str(path),
        nr=config.eqdsk_nr,
        nz=config.eqdsk_nz,
        lcfs_pad=config.eqdsk_lcfs_pad,
        truncate_eq=False,
        run_info=run_info,
        cocos=config.eqdsk_cocos,
    )


def _boundary_state(mygs) -> dict[str, Any]:
    """Native boundary bookkeeping: what set the boundary flux, and where.

    OFT's ``lim_point`` is the limiting wall point of a limited plasma and the
    active X-point of a diverted one, so it is recorded under the matching
    name. OFT leaves it at ``(-1, 1e99)`` when no bound was found.
    """
    diverted = bool(mygs.diverted)
    state: dict[str, Any] = {"diverted": diverted}
    point = getattr(mygs, "lim_point", None)
    if point is not None:
        point = [float(v) for v in np.asarray(point).ravel()[:2]]
        if len(point) == 2 and point[0] >= 0.0 and abs(point[1]) < 1.0e98:
            state["active_x_point" if diverted else "lim_point"] = point
    return state


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
    if config.include_vessel:
        sidecar["vessel_regions"] = sorted((inputs.geometry.get("vessel") or {}).keys())

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

        sidecar["converged"] = True
        sidecar["stats"] = mygs.get_stats()
        sidecar["coil_currents_A"] = dict(mygs.get_coil_currents()[0])
        sidecar["o_point"] = mygs.o_point
        sidecar.update(_boundary_state(mygs))
        sidecar.update(_wall_check(mygs, inputs.geometry["limiter"]))
        _save_eqdsk(mygs, gpath, config, f"# {shot} {ctime}ms")
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
