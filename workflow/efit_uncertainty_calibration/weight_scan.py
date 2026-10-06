"""#891 stage 2: the weight scan with the diamagnetic flux fitted, judged per slice.

Every constraint time of the reference shots is reconstructed under each
setting -- profile basis, probe/loop sigma and now the diamagnetic sigma
(#1196 corrected its sign) -- and every slice is judged by ``criteria.py``:
admissibility, per-family chi-square, virial beta_p and Grad-Shafranov
residual decide ``good``; the Thomson pressure band, where Thomson exists, is
reported beside it as physical consistency and never decides ``good`` or a
setting (criteria v2).

The contract is the calibration's: one EFIT call per slice in an emptied
workdir, every non-reference run from the same initialization fingerprint, no
continuation.  The ``routine`` setting is the production configuration, run for
comparison only.

Stage 1 (``--stage 1``): three bases x five diamagnetic sigma, probe and loop
sigma held near the calibration's best point.  Stage 2 grids probe x loop x
diamagnetic sigma (kept for reproducibility).  Stage 3, the study's rule:
basis x diamagnetic sigma x Ip sigma are gridded, and in each cell the probe
and loop sigma are solved for self-consistently -- ``m <- m sqrt(median
chi2r)`` per family over the admissible slices until the medians sit in the
calibration band -- instead of being gridded.

Slices run in parallel (``--workers``, default 24): one task per slice, its
constraints built once and its settings run in turn.  Workers cannot share
the running initialization baseline, so the cold-start contract is enforced
when their records are merged (``check_fingerprints``).  EFIT's own MPI would
split the slices of one call; with one slice per call the process pool is the
same parallelism without coupling the runs.

    python workflow/efit_uncertainty_calibration/weight_scan.py --stage 3 --workers 24 \
        --output /scratch/weights --table /scratch/weights/weight_scan.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
SCHEMA = 1


def _module(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = _module(HERE / "sigma_calibration.py", "sigma_calibration")
criteria = _module(HERE / "criteria.py", "weight_criteria")

#: The stored diamagnetic sigma is 3 % of |m|; these multiply it.  ``None`` is
#: the row held inactive, as the first calibration did.
DIAMAGNETIC_MULTIPLIERS = (1, 4, 16, 64, None)
BASES = ((1, 1), (2, 1), (1, 2))
#: The calibration's best region (#891, PR #1182): probes x16 with a 2 % floor
#: bring the probe chi2r near 1; loops already sit at 1.2-2.4 at x1.
STAGE1_PROBE, STAGE1_LOOP, STAGE1_FLOOR = 16, 2, 0.02
#: The psi convergence every stage runs at unless a setting names its own.
ERRMIN = 1.0e-4


def setting_name(basis: Sequence[int], probe: float, loop: float, dia: float | None, floor: float,
                 ip: float = 1.0) -> str:
    dia_tag = "off" if dia is None else f"x{dia:g}"
    ip_tag = "" if ip == 1.0 else f"_ip_x{ip:g}"
    return (f"p{basis[0]}f{basis[1]}_probe_x{probe:g}_loop_x{loop:g}_dia_{dia_tag}{ip_tag}"
            f"_floor{round(100 * floor)}pct")


def working_settings() -> list[dict[str, Any]]:
    """Stage 5: the working setting (2,1) chosen 2026-09-29, at ERRMIN 1e-4 and 1e-3.

    The stage-4 calibration of that basis (probe x3.62, loop x2.15, diamagnetic
    x16, Ip 20 %, psi-only exit).  The looser ERRMIN asks how much of the
    ramp-up and ramp-down divergence is the tolerance rather than the fit.
    """
    base = {"basis": [2, 1], "probe": 3.62, "loop": 2.15, "dia": 16, "ip": 4.0, "floor": STAGE1_FLOOR,
            "psi_exit": True}
    name = setting_name(base["basis"], base["probe"], base["loop"], base["dia"], base["floor"], base["ip"])
    return [{"name": "routine", "routine": True}] + [
        {**base, "name": f"{name}_psiexit_errmin{tag}", "error_minimum": value}
        for tag, value in (("1e-4", 1.0e-4), ("1e-3", 1.0e-3))
    ]


#: Stage 6 (#579, 2026-10-02): the model/weight sensitivity ensemble.  Not a
#: search for the best setting -- every configuration is judged on residual
#: credibility, Thomson consistency and the stability of W, beta_p, l_i and q
#: across the configurations, and Thomson never selects one.
SENSITIVITY_DIA = (None, 4, 16, 64)
SENSITIVITY_BASES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (3, 2))
#: (probe, loop) sigma multipliers: the working setting, then each family
#: halved and doubled about it with the other held.
SENSITIVITY_PROBE_LOOP = ((3.62, 2.15), (1.81, 2.15), (7.24, 2.15), (3.62, 1.075), (3.62, 4.3))
SENSITIVITY_IP = 4.0


def sensitivity_settings() -> list[dict[str, Any]]:
    """Stage 6: SENSITIVITY_DIA x SENSITIVITY_BASES x SENSITIVITY_PROBE_LOOP (120), psi-only exit."""
    settings = []
    for basis in SENSITIVITY_BASES:
        for dia in SENSITIVITY_DIA:
            for probe, loop in SENSITIVITY_PROBE_LOOP:
                name = setting_name(basis, probe, loop, dia, STAGE1_FLOOR, SENSITIVITY_IP) + "_psiexit"
                settings.append({"name": name, "basis": list(basis), "probe": probe, "loop": loop, "dia": dia,
                                 "ip": SENSITIVITY_IP, "floor": STAGE1_FLOOR, "psi_exit": True})
    return settings


def stage1_settings() -> list[dict[str, Any]]:
    settings = [{"name": "routine", "routine": True}]
    for basis in BASES:
        for dia in DIAMAGNETIC_MULTIPLIERS:
            settings.append({
                "name": setting_name(basis, STAGE1_PROBE, STAGE1_LOOP, dia, STAGE1_FLOOR),
                "basis": list(basis), "probe": STAGE1_PROBE, "loop": STAGE1_LOOP,
                "dia": dia, "floor": STAGE1_FLOOR,
            })
    return settings


#: Stage 2 (#891): stage 1 left the probes at chi2r ~0.1 and the loops ~0.3,
#: the diamagnetic row unsolvable at x1-x4, and the flat-top slices failing
#: the 3 % Ip bound against a 5 % Ip sigma -- so both Ip sigmas are compared.
STAGE2_PROBE = (4, 8)
STAGE2_LOOP = (1, 2)
STAGE2_DIA = (16, 64, None)
STAGE2_BASES = ((1, 1), (1, 2))
STAGE2_IP = (1.0, 0.4)


def stage2_settings() -> list[dict[str, Any]]:
    settings = [{"name": "routine", "routine": True}]
    for basis in STAGE2_BASES:
        for ip in STAGE2_IP:
            for probe in STAGE2_PROBE:
                for loop in STAGE2_LOOP:
                    for dia in STAGE2_DIA:
                        settings.append({
                            "name": setting_name(basis, probe, loop, dia, STAGE1_FLOOR, ip),
                            "basis": list(basis), "probe": probe, "loop": loop, "dia": dia,
                            "ip": ip, "floor": STAGE1_FLOOR,
                        })
    return settings


def uncertainty_scales(setting: Mapping[str, Any]) -> dict[str, float]:
    """``uncertainty_scales`` divides sigma: a multiplier m is a scale 1/m."""
    dia = setting["dia"]
    return {
        "bpol_probe": 1.0 / float(setting["probe"]),
        "flux_loop": 1.0 / float(setting["loop"]),
        "plasma_current": 1.0 / float(setting.get("ip", 1.0)),
        "diamagnetic_flux": calibration.DIAMAGNETIC_INACTIVE_SCALE if dia is None else 1.0 / float(dia),
    }


def fitted_families(setting: Mapping[str, Any]) -> tuple[str, ...]:
    families = tuple(calibration.FITTED_FAMILIES)
    return families if setting.get("dia") is None else families + ("diamagnetic_flux",)


def inactive_families(setting: Mapping[str, Any]) -> tuple[str, ...]:
    """The criteria's family names for rows the setting scaled out of the fit."""
    if setting.get("routine"):
        return ()
    return ("dia",) if setting.get("dia") is None else ()


def _pressure_min(mapping: Mapping[str, Any]) -> float:
    pressure = np.asarray(mapping.get("PRES", []), dtype=float)
    return float(np.nanmin(pressure)) if pressure.size else float("nan")


def _virial(gfile: Path) -> dict[str, float]:
    """beta_p from the pressure integral and from the pair_13 closure, on the g-file."""
    from omas import ODS

    from vaft.data import read_geqdsk
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    ods = ODS(consistency_check=False)
    read_geqdsk(gfile).to_omas(ods)
    try:
        v = compute_virial_equilibrium_quantities_ods(copy.deepcopy(ods), time_slice=0)[0]
        descriptors = derive_global_descriptors(as_equilibrium(ods, time_index=0)).values
        entry = descriptors.get("beta_p_boundary_average")
        integral = float(entry.value) if entry is not None and entry.available else float("nan")
        return {
            "beta_p_integral": integral,
            "beta_p_pair_13": float(v["pair_13"]["beta_p"]),
            "denominator_pair_13": float(v["conditioning"]["denominators"]["pair_13"]),
            "alpha": float(v["alpha"]),
            "li": float(v["li"]),
        }
    except Exception as error:  # a degenerate equilibrium is a result, not a crash
        return {"error": repr(error)}


def _thomson(gfile: Path, diagnostics, thomson_check) -> dict[str, Any] | None:
    """The Thomson comparison, or ``None`` where no sample is near the slice."""
    if diagnostics is None:
        return None
    result = thomson_check.check_gfile(gfile, diagnostics)
    if result.get("status") in (None, "not_available") or not result.get("points"):
        return None
    return {k: result.get(k) for k in ("status", "log_ratio", "sum_ratio", "points", "time_offset_s", "reason")}


def _afile_extras(workdir: Path) -> dict[str, float]:
    from vaft.data import read_aeqdsk

    afiles = sorted(workdir.glob("a0*"))
    if not afiles:
        return {}
    scalars = read_aeqdsk(afiles[0]).scalars
    return {k: float(scalars[k]) for k in ("cdflux", "betapd", "wdia", "betat", "li3") if k in scalars}


def _shape(mapping: Mapping[str, Any]) -> dict[str, float] | None:
    r = np.asarray(mapping.get("RBBBS", []), float)
    z = np.asarray(mapping.get("ZBBBS", []), float)
    if r.size < 5:
        return None
    r0, a = 0.5 * (r.max() + r.min()), 0.5 * (r.max() - r.min())
    top, bottom = z.argmax(), z.argmin()
    return {"R0": r0, "a": a, "kappa": (z[top] - z[bottom]) / (2 * a),
            "delta": 0.5 * ((r0 - r[top]) + (r0 - r[bottom])) / a, "zc": 0.5 * (z[top] + z[bottom])}


def _ip_sigma(record: Mapping[str, Any], built) -> float | None:
    sigma = (record.get("fit") or {}).get("ip_sigma_median")
    ip = abs(float(built["equilibrium.time_slice.0.constraints.ip.measured"]))
    return float(sigma) / ip if sigma and ip else None


def exit_chi_squared(setting: Mapping[str, Any], fitted: int) -> float:
    """The SAICON a setting runs with: the statistical target, or out of reach on a psi-only exit."""
    return PSI_EXIT_SAICON if setting.get("psi_exit") else calibration.chi_squared_target(fitted)


def run_slice(study, built, *, shot, chosen, setting, output, efit, diagnostics, thomson_check, baseline):
    """One setting on one slice; returns the record and the (possibly new) baseline fingerprint."""
    ods = copy.deepcopy(built)
    workdir = output / f"shot_{shot}" / f"t{round(float(chosen[0]) * 1e6):07d}" / setting["name"]
    if setting.get("routine"):
        case = {"grid": study.ROUTINE_GRID, "name": "routine", "routine": True}
        mode, scales = "legacy_weight", None
    else:
        if setting["floor"]:
            calibration.apply_sigma_floor(ods, setting["floor"])
        target = exit_chi_squared(setting, calibration.fitted_constraint_count(built, families=fitted_families(setting)))
        case = {
            "grid": study.ROUTINE_GRID, "table": study.PACKAGED, "inner_iterations": 1,
            "error_minimum": float(setting.get("error_minimum", ERRMIN)), "max_iterations": calibration.MAX_ITERATIONS,
            "kppcur": setting["basis"][0], "kffcur": setting["basis"][1],
            "chi_squared_target": target, "name": setting["name"],
        }
        mode, scales = "standard_deviation", uncertainty_scales(setting)
    run = study.run_case(ods, shot=shot, times=chosen, case=case, workdir=workdir, efit=efit,
                         uncertainty_mode=mode, uncertainty_scales=scales)
    record = run["records"][0]
    fingerprint = (record.get("initialization") or {}).get("sha256")
    if not setting.get("routine"):
        baseline = calibration.require_same_start(baseline, fingerprint, f"{shot}@{record['time_ms']} {setting['name']}")
    mapping = record.get("geqdsk")
    converged = study.converged(record)
    gfiles = sorted(workdir.glob("g0*"))
    out = {
        "setting": setting["name"], "shot": shot, "time_ms": record["time_ms"],
        "converged": converged, "exit_path": record.get("exit_path"), "outcome": record.get("outcome"),
        "iterations": record.get("iterations_n"), "solver_errors": record.get("solver_errors") or [],
        "scalars": record.get("scalars") or {},
        "fit": record.get("fit"), "gs": {k: v for k, v in (record.get("gs") or {}).items() if k != "profile"},
        "ip_measured": float(built["equilibrium.time_slice.0.constraints.ip.measured"]),
        "dia_measured": float(built["equilibrium.time_slice.0.constraints.diamagnetic_flux.measured"])
        if "equilibrium.time_slice.0.constraints.diamagnetic_flux.measured" in built else None,
        "inactive_families": list(inactive_families(setting)),
        # the Ip sigma the fit was given (the admissible Ip ratio's upper edge);
        # the routine's legacy weight is not a sigma
        "ip_sigma_relative": None if setting.get("routine") else _ip_sigma(record, built),
        "fingerprint": fingerprint,
        "afile": _afile_extras(workdir),
    }
    if mapping is not None and gfiles:
        out["pressure_min"] = _pressure_min(mapping)
        out["shape"] = _shape(mapping)
        out["virial"] = _virial(gfiles[0])
        out["thomson"] = _thomson(gfiles[0], diagnostics, thomson_check)
    out["evaluation"] = criteria.evaluate(out)
    return out, baseline


# --------------------------------------------------------------------------
# Stage 3: probe and loop sigma found self-consistently, not gridded
# --------------------------------------------------------------------------

#: The axes that need a judgement; probe and loop sigma are solved for.
#: Ip sigma 5 %, 20 %, 50 % (x1, x4, x10): in the ramp-up the closed-surface
#: current can be 30-80 % of the measured Ip.
STAGE3_BASES = ((1, 1), (2, 1), (1, 2))
STAGE3_DIA = (16, 64, None)
STAGE3_IP = (1.0, 4.0, 10.0)
STAGE3_START = {"probe": 4.0, "loop": 1.0}
STAGE3_ROUNDS = 4
MULTIPLIER_CLAMP = (0.5, 64.0)

# Stage 4: the fit stops on psi convergence alone (study rule, 2026-09-28).
# Stage 3 let SAICON = N + 3 sqrt(2N) gate EFIT's exit: at a calibrated sigma
# most slices fit the probes at chi2r ~ 7, so they ran all 514 iterations with
# psi converged to ~2e-8 and were reported unconverged -- and the median that
# set sigma came only from the few that fitted.  With SAICON out of reach the
# exit is ERRMIN plus a chi-square stall; chi-square is judged by the criteria.
# The diamagnetic and Ip sigma axes moved nothing in stage 3 and are fixed.
PSI_EXIT_SAICON = 1.0e10
# (1,3) and (2,2) added 2026-09-29: the (1,1) basis pins li near 0.93, so a
# slice whose magnetics want a broader current is fitted with beta_p < 0.
STAGE4_BASES = STAGE3_BASES + ((1, 3), (2, 2))
STAGE4_DIA = 16
STAGE4_IP = 4.0
STAGE4_START = {"probe": 8.0, "loop": 2.0}
STAGE4_ROUNDS = 6


def _round3(value: float) -> float:
    return float(f"{value:.3g}")


def next_multipliers(current: Mapping[str, float], medians: Mapping[str, float],
                     clamp: Sequence[float] = MULTIPLIER_CLAMP) -> dict[str, float]:
    """One self-consistency step: ``m <- m * sqrt(median chi2r)`` per family.

    A family's reduced chi-square scales as ``1/m**2`` at fixed residuals, so
    this sends it to 1 in one step when the residuals do not move and in a
    few when they do.  A family with no finite median keeps its multiplier.
    Rounded to three significant figures so names stay stable across rounds.
    """
    low, high = clamp
    out = {}
    for family, m in current.items():
        chi = medians.get(family, float("nan"))
        step = m * math.sqrt(chi) if chi is not None and math.isfinite(chi) and chi > 0 else m
        out[family] = _round3(min(max(step, low), high))
    return out


def backoff_multipliers(previous: Mapping[str, float], current: Mapping[str, float]) -> dict[str, float]:
    """The geometric midpoint of two rounds' multipliers, for a round in which nothing converged."""
    return {f: _round3(math.sqrt(previous[f] * current[f])) for f in current}


def stage3_cells() -> list[dict[str, Any]]:
    """The gridded axes; each cell carries its own probe/loop multipliers."""
    return [
        {"basis": list(basis), "dia": dia, "ip": ip, "floor": STAGE1_FLOOR}
        for basis in STAGE3_BASES for dia in STAGE3_DIA for ip in STAGE3_IP
    ]


def stage4_cells(bases: Sequence[Sequence[int]] = STAGE4_BASES) -> list[dict[str, Any]]:
    """The bases at diamagnetic x16 and Ip 20 %, stopping on psi convergence."""
    return [{"basis": list(basis), "dia": STAGE4_DIA, "ip": STAGE4_IP, "floor": STAGE1_FLOOR, "psi_exit": True}
            for basis in bases]


def cell_setting(cell: Mapping[str, Any], multipliers: Mapping[str, float]) -> dict[str, Any]:
    probe, loop = multipliers["probe"], multipliers["loop"]
    name = setting_name(cell["basis"], probe, loop, cell["dia"], cell["floor"], cell["ip"])
    setting = {"name": name, "basis": list(cell["basis"]), "probe": probe, "loop": loop, "dia": cell["dia"],
               "ip": cell["ip"], "floor": cell["floor"]}
    if cell.get("psi_exit"):
        setting.update(name=name + "_psiexit", psi_exit=True)
    return setting


def check_fingerprints(records: Sequence[Mapping[str, Any]]) -> str | None:
    """Every non-reference record started from one initial state, or raise.

    Workers cannot share a running baseline, so the cold-start contract is
    checked when their records are merged.
    """
    seen = {r.get("fingerprint") for r in records
            if r.get("setting") != "routine" and "error" not in r and r.get("fingerprint")}
    if len(seen) > 1:
        raise RuntimeError(f"runs started from {len(seen)} different initial states: {sorted(seen)}")
    return next(iter(seen), None)


# --------------------------------------------------------------------------
# Running: one task per slice, slices in parallel
# --------------------------------------------------------------------------

_CONTEXT: dict[str, Any] = {}
#: Set by ``main`` before the context is built (workers fork after it):
#: ``products_dir`` holds ``<shot>.json.gz`` diagnostics+eddy products beyond
#: the packaged reference set; ``thomson_root`` is the raw Thomson data root.
_INPUTS: dict[str, Any] = {"products_dir": None, "thomson_root": None}


def _context(efit_home: str | None) -> dict[str, Any]:
    """Modules and paths a worker needs, loaded once per process."""
    if _CONTEXT:
        return _CONTEXT
    if efit_home:
        os.environ["EFITHOME"] = str(Path(efit_home).expanduser())
    study = calibration._module(calibration.CONVERGENCE_STUDY, "convergence_study")
    from vaft.code.efit.toolchain import resolve_toolchain
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    reference = json.loads(calibration.REFERENCE_SET.read_text(encoding="utf-8"))
    _CONTEXT.update(
        study=study,
        seed_study=study._module(study.SEED_STUDY, "seed_basin"),
        wrapper=study._module(
            REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py",
            "generate_constraints_ods_wrapper",
        ),
        thomson_check=_module(REPOSITORY / "workflow" / "efit_diamagnetic_weight" / "thomson_pressure_check.py",
                              "thomson_pressure_check"),
        resolved=resolved,
        efit=resolved.get("efit"),
        tables=str(Path(data_path("efit")).resolve()) + "/",
        products={int(i["shot"]): Path(data_path(i["files"]["product"]["path"])) for i in reference["shots"]
                  if i["files"]["product"] and i["files"]["product"]["exists"]},
        diagnostics={},
        thomson_root=_INPUTS["thomson_root"],
    )
    if _INPUTS["products_dir"]:
        for path in sorted(Path(_INPUTS["products_dir"]).expanduser().glob("*.json.gz")):
            _CONTEXT["products"][int(path.name.split(".")[0])] = path
    return _CONTEXT


def process_slice(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every setting on one (shot, time) slice; resumable from its checkpoint."""
    ctx = _context(task["efit_home"])
    study, shot, time = ctx["study"], int(task["shot"]), float(task["time"])
    settings, output = task["settings"], Path(task["output"])
    tag = f"t{round(time * 1e6):07d}"
    checkpoint = output / f"shot_{shot}" / tag / "records.json"
    names = [s["name"] for s in settings]
    if checkpoint.is_file():
        saved = json.loads(checkpoint.read_text())
        if saved.get("settings") == names:
            return saved["records"]
    if shot not in ctx["diagnostics"]:
        ctx["diagnostics"][shot] = ctx["thomson_check"].thomson_diagnostics(shot, ctx["thomson_root"])
    try:
        built, chosen = study.prepare_constraints(
            shot, ctx["products"][shot], [time], workdir=output / f"shot_{shot}" / tag / "constraints",
            tables=ctx["tables"], tstep=0.001, average_window=0.0005, seed_study=ctx["seed_study"],
        )
    except Exception as error:
        print(f"{shot}@{tag}: constraints failed: {error!r}", flush=True)
        return []
    records, baseline = [], None
    for setting in settings:
        try:
            record, baseline = run_slice(study, built, shot=shot, chosen=chosen, setting=setting, output=output,
                                         efit=str(ctx["efit"]), diagnostics=ctx["diagnostics"][shot],
                                         thomson_check=ctx["thomson_check"], baseline=baseline)
        except RuntimeError:
            raise  # a refused initialization is a contract violation, not a slice result
        except Exception as error:
            record = {"setting": setting["name"], "shot": shot, "time_ms": round(time * 1e3), "error": repr(error)}
        for key in ("round",):
            if key in task:
                record[key] = task[key]
        records.append(record)
        good = (record.get("evaluation") or {}).get("good")
        print(f"{shot}@{record['time_ms']} {setting['name']}: {record.get('exit_path')} good={good}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"settings": names, "baseline": baseline, "records": records},
                                     default=study._json_default))
    return records


def slice_times(shots: Sequence[int], times_ms: set[int] | None, efit_home: str | None) -> list[tuple[int, float]]:
    ctx = _context(efit_home)
    out = []
    for shot in shots:
        ods = ctx["seed_study"].load_product(ctx["products"][shot])
        times, _ = ctx["wrapper"]._select_times(ods, "auto", 0.001, None, None)
        out.extend((shot, float(t)) for t in times if times_ms is None or round(float(t) * 1e3) in times_ms)
    return out


def run_all(slices, settings, output: Path, *, workers: int, efit_home: str | None, extra=None) -> list[dict[str, Any]]:
    """Every setting on every slice, ``workers`` slices at a time."""
    tasks = [{"shot": shot, "time": time, "settings": settings, "output": str(output), "efit_home": efit_home,
              **(extra or {})} for shot, time in slices]
    if workers <= 1:
        results = [process_slice(task) for task in tasks]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(process_slice, tasks))
    return [record for batch in results for record in batch]


def _write(table: Path, payload: Mapping[str, Any], study) -> None:
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(payload, indent=1, default=study._json_default) + "\n")
    print(f"wrote {table}", flush=True)


def run_stage3(slices, output: Path, *, workers: int, efit_home: str | None, stage: int = 3,
               bases: Sequence[Sequence[int]] | None = None) -> dict[str, Any]:
    """Probe/loop multipliers per cell by self-consistency, all cells a round at a time."""
    cells, start, rounds = ((stage4_cells(bases or STAGE4_BASES), STAGE4_START, STAGE4_ROUNDS) if stage == 4
                            else (stage3_cells(), STAGE3_START, STAGE3_ROUNDS))
    state = [{"cell": c, "multipliers": dict(start), "done": False, "history": []} for c in cells]
    records: list[dict[str, Any]] = []
    for round_index in range(rounds):
        active = [s for s in state if not s["done"]]
        if not active:
            break
        settings = ([{"name": "routine", "routine": True}] if round_index == 0 else []) + [
            cell_setting(s["cell"], s["multipliers"]) for s in active
        ]
        batch = run_all(slices, settings, output / f"round{round_index}", workers=workers,
                        efit_home=efit_home, extra={"round": round_index})
        check_fingerprints(records + batch)
        records.extend(batch)
        for s in active:
            name = cell_setting(s["cell"], s["multipliers"])["name"]
            calibration_now = criteria.setting_calibration([r for r in batch if r.get("setting") == name
                                                            and "error" not in r])
            advance_cell(s, round_index, name, calibration_now)
            print(f"round {round_index} {name}: {s['history'][-1]['medians']} -> {s['multipliers']} "
                  f"status={s['status']}", flush=True)
    for s in state:
        finish_cell(s)
    return {"cells": state, "records": records}


def _finite_medians(medians: Mapping[str, Any]) -> bool:
    return any(v is not None and math.isfinite(v) for v in medians.values())


def advance_cell(s: dict[str, Any], round_index: int, name: str, calibration_now: Mapping[str, Any]) -> None:
    """Record one round of a cell and choose its next multipliers, or finish it.

    ``status`` ends as ``calibrated``; ``no_converged_slice`` (nothing ever
    converged); or ``stalled`` (the next multipliers equal these, e.g. at the
    clamp, which would rerun the same setting under the same name).  A round in
    which nothing converged steps back halfway towards the last round that did.
    """
    medians = {f: v["median_reduced_chi2"] for f, v in calibration_now["families"].items()}
    s["history"].append({"round": round_index, "setting": name, "multipliers": dict(s["multipliers"]),
                         "medians": medians, "calibration": calibration_now})
    s.setdefault("status", "running")
    if calibration_now["calibrated"]:
        s["done"], s["status"] = True, "calibrated"
        return
    if _finite_medians(medians):
        proposed = next_multipliers(s["multipliers"], medians)
    else:
        # Nothing converged here (a loose fit loses the boundary in `bound`).
        anchor = next((h for h in reversed(s["history"][:-1]) if _finite_medians(h["medians"])), None)
        if anchor is None:
            s["done"], s["status"] = True, "no_converged_slice"
            return
        proposed = backoff_multipliers(anchor["multipliers"], s["multipliers"])
    if proposed == s["multipliers"]:
        s["done"], s["status"] = True, "stalled"
        return
    s["multipliers"] = proposed


def finish_cell(s: dict[str, Any]) -> None:
    """Leave ``multipliers`` at the last multipliers actually run.

    A cell still uncalibrated after the last round keeps its untested next
    step under ``proposed_multipliers``, so ``multipliers`` is never read as a
    solved sigma that was not run.
    """
    if not s["history"]:
        return
    if not s["done"]:
        s["done"], s["status"] = True, "rounds_exhausted"
        s["proposed_multipliers"] = s["multipliers"]
    s["multipliers"] = dict(s["history"][-1]["multipliers"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--settings", type=Path, help="JSON list of settings (stages 1, 2, 5, 6)")
    parser.add_argument("--products-dir", type=Path, default=None,
                        help="<shot>.json.gz diagnostics+eddy products (compose_products.py) beyond the reference set")
    parser.add_argument("--thomson-root", type=Path, default=None,
                        help="raw Thomson data root (default: the packaged legacy data)")
    # --stage 3: self-consistent sigma, SAICON-gated exit; --stage 4: the same on psi convergence alone
    parser.add_argument("--bases", default=None, help="stage 4 only: kppcur,kffcur pairs, e.g. '1,3;2,2'")
    parser.add_argument("--shots", default="39915,41524,41672")
    parser.add_argument("--times", default=None, help="comma-separated ms subset (smoke runs)")
    parser.add_argument("--workers", type=int, default=24, help="slices run at once (1 = serial)")
    parser.add_argument("--efit-home", default=None)
    args = parser.parse_args(argv)
    if args.bases and args.stage != 4:
        parser.error("--bases applies to --stage 4 only")
    if args.settings and args.stage in (3, 4):
        parser.error("--settings applies to stages 1, 2 and 5; stages 3 and 4 solve their own")

    _INPUTS.update(products_dir=args.products_dir, thomson_root=args.thomson_root)
    ctx = _context(args.efit_home)
    if ctx["efit"] is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    from vaft.code.efit.toolchain import toolchain_identities

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    times_ms = {int(t) for t in args.times.split(",")} if args.times else None
    slices = slice_times([int(s) for s in args.shots.split(",")], times_ms, args.efit_home)
    print(f"{len(slices)} slices, {args.workers} workers", flush=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA, "stage": args.stage,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(ctx["resolved"]), "tables": ctx["tables"],
        "products": {str(shot): str(path) for shot, path in sorted(ctx["products"].items())},
        "thomson_root": None if ctx["thomson_root"] is None else str(ctx["thomson_root"]),
        "criteria": criteria.CRITERIA, "criteria_version": criteria.CRITERIA_VERSION, "workers": args.workers,
    }
    if args.stage in (3, 4):
        bases = [tuple(int(k) for k in pair.split(",")) for pair in args.bases.split(";")] if args.bases else None
        result = run_stage3(slices, output, workers=args.workers, efit_home=args.efit_home, stage=args.stage,
                            bases=bases)
        records = result["records"]
        payload.update(cells=result["cells"])
    else:
        settings = (json.loads(args.settings.read_text()) if args.settings
                    else {1: stage1_settings, 2: stage2_settings, 5: working_settings,
                          6: sensitivity_settings}[args.stage]())
        records = run_all(slices, settings, output, workers=args.workers, efit_home=args.efit_home)
        payload.update(settings=settings)
    payload.update(
        initialization_fingerprint=check_fingerprints(records),
        summary=criteria.summarize([r for r in records if "error" not in r]),
        slices=criteria.slice_labels(records),
        records=records,
    )
    _write(args.table.expanduser(), payload, ctx["study"])
    for row in sorted(payload["summary"], key=lambda r: -criteria.good_fraction(r)):
        cal = row["calibration"]
        chis = "/".join(f"{v['median_reduced_chi2']:.2f}" for v in cal["families"].values())
        print(f"{row['setting']:60s} good {row['good']}/{row['slices']} chi2r {chis} calibrated={cal['calibrated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
