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
    """Limiter points on the wall from one cell below ``lim_zmax`` up to
    ``neck_limiter_zmax``.

    OFT keeps a mesh limiter node only through a cell that lies entirely at
    or below ``lim_zmax`` (``grad_shaf.F90``: a cell with ANY node above it
    is skipped), so the cut is per cell, not per node: wall nodes within one
    cell height below ``lim_zmax`` can lose every cell they belong to and
    drop out of the candidate set. The band from ``lim_zmax`` minus the
    largest cell edge that can touch the wall (``dx_plasma`` on the plasma
    side, ``dx_conductor`` when the vessel is meshed) up to
    ``neck_limiter_zmax`` is therefore sampled along the limiter polygon
    every ``dx_plasma`` and returned as an ``(n, 2)`` array of (R, Z) for
    OFT's explicit limiter points. Points that duplicate a surviving mesh
    node are harmless.
    """
    if config.lim_zmax is None or config.neck_limiter_zmax is None:
        return np.zeros((0, 2))
    hi = float(config.neck_limiter_zmax)
    if hi <= float(config.lim_zmax):
        return np.zeros((0, 2))
    lo = float(config.lim_zmax) - max(float(config.dx_plasma), float(config.dx_conductor))
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


def _apply_vessel_currents(mygs, inputs) -> dict[str, float]:
    """Impose the eddy-stage wall currents on the vessel coil regions (#1534).

    Each vessel region is a one-turn coil driven at 1 A. OFT spreads a coil's
    current as ``I * dist / area`` over the region, so the distribution is set
    to ``J * area``. Each cell takes the current density of its nearest
    pf_passive loop (the loop current over the mesh area that loop owns), and
    each node the area-weighted density of the cells around it. The region
    then carries the sum of its loop currents, distributed along the wall loop
    by loop, and mixed-sign loops need no normalisation. Returns the total
    current imposed per region [A].
    """
    if not inputs.vessel_loop_currents:
        return {}
    oft = import_oft()
    coil_dict = oft.meshing.load_gs_mesh(str(inputs.mesh_file))[3]
    r = np.asarray(mygs.r, dtype=float)[:, :2]
    lc = np.asarray(mygs.lc, dtype=int)
    reg = np.asarray(mygs.reg, dtype=int)
    tri = r[lc]                                                    # (nc, 3, 2)
    area = 0.5 * np.abs(
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
    )
    centroid = tri.mean(axis=1)
    totals: dict[str, float] = {}
    for region, loop_currents in inputs.vessel_loop_currents.items():
        cells = np.flatnonzero(reg == int(coil_dict[region]["reg_id"]))
        loops = inputs.geometry["vessel"][region]["loops"]
        centres = np.array([[loop["r"], loop["z"]] for loop in loops], dtype=float)
        current = np.array([loop_currents[int(loop["index"])] for loop in loops], dtype=float)

        def nearest(points: np.ndarray) -> np.ndarray:
            d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            return np.argmin(d2, axis=1)

        owner = nearest(centroid[cells])
        owned_area = np.bincount(owner, weights=area[cells], minlength=len(loops))
        density = np.divide(current, owned_area, out=np.zeros_like(current), where=owned_area > 0)
        # a loop owning no cell (thinner than the mesh) hands its current to
        # the nearest loop that does
        has = np.flatnonzero(owned_area > 0)
        for k in np.flatnonzero(owned_area <= 0):
            j = has[np.argmin(((centres[has] - centres[k]) ** 2).sum(axis=1))]
            density[j] += current[k] / owned_area[j]
        region_area = float(area[cells].sum())
        # a node takes the area-weighted density of the cells around it, so a
        # node on the border of two loops does not hand one of them both sides
        weight = np.zeros(r.shape[0])
        value = np.zeros(r.shape[0])
        cell_density = density[owner] * area[cells]
        for corner in range(lc.shape[1]):
            np.add.at(weight, lc[cells, corner], area[cells])
            np.add.at(value, lc[cells, corner], cell_density)
        dist = np.divide(value, weight, out=np.zeros_like(value), where=weight > 0) * region_area
        mygs.set_coil_current_dist(region, dist)
        totals[region] = float(current.sum())
    return totals


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


def _apply_profiles(oft, mygs, config: TokaMakerConfig, profiles=None) -> None:
    if config.profile_mode != "power_law":
        from .profiles import profiles_for_config
        profiles = profiles if profiles is not None else profiles_for_config(config)
        mygs.set_profiles(**profiles.solver_tables())
        if not np.any(profiles.pprime):
            mygs.pnorm = 0.0
        return
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


def _native_active_x_points(mygs) -> dict[str, Any]:
    """Retain native FE saddles on the active boundary flux before g-file export."""
    points, diverted = mygs.get_xpoints()
    if not diverted or points is None:
        return {"native_active_x_points_m": []}
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    reference = np.asarray(mygs.lim_point, dtype=float).reshape(-1)[:2]
    field = mygs.get_field_eval("psi")
    reference_psi = float(np.asarray(field.eval(reference)).ravel()[0])
    span = abs(float(mygs.get_stats().get("dflux", 1.)))
    if not np.isfinite(span) or span <= 0:
        return {"native_active_x_points_m": [], "native_x_points_reason": "invalid flux span"}
    active = []
    all_points = []
    for point in points:
        residual = abs(float(np.asarray(field.eval(point)).ravel()[0]) - reference_psi) / span
        all_points.append({"rz_m": point.tolist(), "relative_boundary_flux_residual": residual})
        if np.isfinite(residual) and residual <= 1e-4:
            active.append(point.tolist())
    return {"native_active_x_points_m": active, "native_all_x_points": all_points,
            "native_x_flux_tolerance_fraction": 1e-4}


def _initial_current_density(equilibrium, points):
    """Physical Jphi [A/m²] seed on native nodes; no plasma Green integration."""
    from matplotlib.path import Path as PolygonPath
    from scipy.interpolate import RegularGridInterpolator
    from vaft.process.equilibrium import convert_cocos
    eq = convert_cocos(equilibrium, 11)
    inside = PolygonPath(eq.lcfs.points).contains_points(points)
    psi = RegularGridInterpolator((eq.r, eq.z), eq.psi, bounds_error=False,
                                  fill_value=np.nan)(points[inside])
    order = np.argsort(eq.psi_1d)
    pp = np.interp(psi, eq.psi_1d[order], eq.pprime[order])
    ffp = np.interp(psi, eq.psi_1d[order], eq.ffprime[order])
    r = points[inside, 0]
    current = np.zeros(len(points))
    current[inside] = -2*np.pi*(r*pp + ffp/(4e-7*np.pi*r))
    if not np.isfinite(current).all():
        raise ValueError("initial equilibrium grid must cover its LCFS")
    return current


def run_tokamaker(inputs: TokaMakerInputs, config: TokaMakerConfig, *, refinement=None,
                  verify_refinement: bool = False) -> TokaMakerResult:
    """Execute a forward solve and collect the produced outputs.

    Builds the mesh on a cache miss, then runs the canonical TokaMaker
    sequence (mesh → regions → setup → coil currents → targets → profiles →
    ``init_psi`` → ``solve``) and exports the equilibrium as an EFIT g-file
    named ``g<shot>.<time_ms>`` (COCOS per ``config.eqdsk_cocos``). A failed
    solve is reported through ``result.ok``/``result.error`` rather than
    raised, mirroring the subprocess adapters.
    """
    if verify_refinement and refinement is None:
        raise ValueError("frozen verification requires shape refinement")
    if refinement is not None:
        if set(refinement.reference_currents_A) != set(inputs.coil_currents) or any(
            abs(refinement.reference_currents_A[n] - inputs.coil_currents[n]) > 1e-6
            for n in inputs.coil_currents
        ):
            raise ValueError("refinement reference currents must match prescribed initial PF circuits")
        if config.vsc_coil or inputs.vessel_loop_currents:
            raise ValueError("shape refinement requires no VSC or imposed vessel currents")
    oft = import_oft()
    env = get_oft_env(config.nthreads)

    if not inputs.mesh_file.is_file():
        build_tokamaker_mesh(inputs.geometry, inputs.mesh_file, config)

    shot = int(inputs.shot)
    ctime = int(round(inputs.time * 1000))
    gpath = inputs.workdir / f"g{shot:06d}.{ctime:05d}"

    returncode = 1
    error = ""
    verified_error = None
    verified_dir = inputs.workdir / "verified_free"
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
        mygs.settings.free_boundary = True
        sidecar["free_boundary"] = True
        _apply_vsc(mygs, config)

        mygs.set_coil_currents(dict(inputs.coil_currents))
        vessel_totals = _apply_vessel_currents(mygs, inputs)
        if vessel_totals:
            sidecar["vessel_currents_A"] = vessel_totals
            sidecar["vessel_current_total_A"] = float(sum(vessel_totals.values()))
        from .profiles import profiles_for_config, profile_targets
        profiles = profiles_for_config(config)
        effective_targets = profile_targets(profiles, inputs.targets)
        mygs.set_targets(**effective_targets)
        _apply_profiles(oft, mygs, config, profiles)
        sidecar["targets"] = effective_targets
        sidecar["profile_mode"] = config.profile_mode

        # Pure-pressure normalization omits Ip during nonlinear iterations,
        # but its initial uniform plasma must still have a physical amplitude.
        seed_ip = profiles.ip_A if profiles is not None and "Ip" not in effective_targets else None
        if seed_ip is not None:
            mygs.set_targets(Ip=seed_ip)
        if config.init_equilibrium is not None:
            # curr_source in init_psi is an assembled FE load, not nodal J.
            # vac_solve(rhs_source=...) integrates physical density natively.
            seed = mygs.vac_solve(rhs_source=_initial_current_density(config.init_equilibrium, mygs.r[:, :2]))
            psi = seed.get_psi(normalized=False) if hasattr(seed, "get_psi") else seed
            mygs.set_psi(psi, update_bounds=True)
        else:
            mygs.init_psi(
                config.init_r0, config.init_z0, config.init_a0,
                config.init_kappa, config.init_delta,
            )
        if seed_ip is not None:
            mygs.set_targets(**effective_targets)
        if refinement is not None:
            from .refinement import apply_shape_refinement
            sidecar["shape_refinement"] = apply_shape_refinement(mygs, refinement)
        mygs.solve()

        sidecar["converged"] = True
        sidecar["stats"] = mygs.get_stats()
        if profiles is not None:
            from .profiles import source_profile_diagnostics
            sidecar["source_profiles"] = source_profile_diagnostics(mygs, profiles)
        # vessel regions run as 1 A coils; their real currents are vessel_currents_A
        sidecar["coil_currents_A"] = {
            name: value for name, value in dict(mygs.get_coil_currents()[0]).items()
            if name not in inputs.vessel_loop_currents
        }
        sidecar["o_point"] = mygs.o_point
        sidecar.update(_boundary_state(mygs))
        sidecar.update(_wall_check(mygs, inputs.geometry["limiter"]))
        _save_eqdsk(mygs, gpath, config, f"# {shot} {ctime}ms")
        returncode = 0
        if verify_refinement:
            # Keep the converged native FE state. Reconstructing it from a
            # coarse g-file can select another free-boundary solution branch.
            verified_dir.mkdir(parents=True, exist_ok=False)
            currents = dict(sidecar["coil_currents_A"])
            verified_sidecar = {key: sidecar[key] for key in ("shot", "time_s", "targets", "f0", "cocos")}
            verified_sidecar.update(converged=False, free_boundary=True,
                                    shape_constraints_cleared=True, coil_currents_A=currents)
            try:
                isoflux = getattr(mygs, "set_isoflux_constraints", None)
                if isoflux is None:
                    isoflux = mygs.set_isoflux
                saddles = getattr(mygs, "set_saddle_constraints", None)
                if saddles is None:
                    saddles = mygs.set_saddles
                isoflux(None)
                saddles(None)
                mygs.set_coil_currents(currents)
                mygs.solve()
                realized = dict(mygs.get_coil_currents()[0])
                if set(realized) != set(currents) or any(
                    abs(realized[name] - current) > 1e-6 for name, current in currents.items()
                ):
                    raise RuntimeError("PF currents changed during frozen verification")
                verified_sidecar.update(converged=True, coil_currents_A=realized,
                                        stats=mygs.get_stats(), o_point=mygs.o_point)
                verified_sidecar.update(_boundary_state(mygs))
                verified_sidecar.update(_native_active_x_points(mygs))
                verified_sidecar.update(_wall_check(mygs, inputs.geometry["limiter"]))
                _save_eqdsk(mygs, verified_dir / gpath.name, config, f"# {shot} {ctime}ms frozen")
            except Exception as exc:
                verified_error = str(exc)
                verified_sidecar["error"] = verified_error
            _write_sidecar(verified_dir, verified_sidecar)
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
    if verify_refinement and verified_dir.is_dir():
        verified = collect_tokamaker_outputs(verified_dir, config)
        verified.returncode = 1 if verified_error else 0
        verified.error = verified_error or ""
        verified.mesh_file = result.mesh_file
        if verified_error:
            verified.gfile = None
            verified.geqdsk = ()
            verified.ods = None
        result.verified_free = verified
    return result


__all__ = [
    "run_tokamaker",
]
