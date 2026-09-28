"""#891 stage 2: the weight scan with the diamagnetic flux fitted, judged per slice.

Every constraint time of the reference shots is reconstructed under each
setting -- profile basis, probe/loop sigma and now the diamagnetic sigma
(#1196 corrected its sign) -- and every slice is judged by ``criteria.py``:
admissibility, per-family chi-square, virial beta_p, Grad-Shafranov residual,
and the Thomson pressure band where Thomson exists (39915).

The contract is the calibration's: one EFIT call per slice in an emptied
workdir, every non-reference run from the same initialization fingerprint, no
continuation.  The ``routine`` setting is the production configuration, run for
comparison only.

Stage 1 (``--stage 1``): three bases x five diamagnetic sigma, probe and loop
sigma held near the calibration's best point.  Stage 2 grids probe x loop x
diamagnetic sigma over the region stage 1 passes (``--settings`` JSON).

    python workflow/efit_uncertainty_calibration/weight_scan.py --stage 1 \
        --output /scratch/weights --table /scratch/weights/weight_scan.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
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


def setting_name(basis: Sequence[int], probe: float, loop: float, dia: float | None, floor: float,
                 ip: float = 1.0) -> str:
    dia_tag = "off" if dia is None else f"x{dia:g}"
    ip_tag = "" if ip == 1.0 else f"_ip_x{ip:g}"
    return (f"p{basis[0]}f{basis[1]}_probe_x{probe:g}_loop_x{loop:g}_dia_{dia_tag}{ip_tag}"
            f"_floor{round(100 * floor)}pct")


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
        target = calibration.chi_squared_target(
            calibration.fitted_constraint_count(built, families=fitted_families(setting))
        )
        case = {
            "grid": study.ROUTINE_GRID, "table": study.PACKAGED, "inner_iterations": 1,
            "error_minimum": 1.0e-4, "max_iterations": calibration.MAX_ITERATIONS,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--settings", type=Path, help="JSON list of settings (stage 2)")
    parser.add_argument("--shots", default="39915,41524,41672")
    parser.add_argument("--times", default=None, help="comma-separated ms subset (smoke runs)")
    parser.add_argument("--efit-home", default=None)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())
    study = calibration._module(calibration.CONVERGENCE_STUDY, "convergence_study")
    seed_study = study._module(study.SEED_STUDY, "seed_basin")
    wrapper = study._module(
        REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py",
        "generate_constraints_ods_wrapper",
    )
    thomson_check = _module(REPOSITORY / "workflow" / "efit_diamagnetic_weight" / "thomson_pressure_check.py",
                            "thomson_pressure_check")
    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    efit = resolved.get("efit")
    if efit is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    if args.settings:
        settings = json.loads(args.settings.read_text())
    else:
        settings = {1: stage1_settings, 2: stage2_settings}[args.stage]()
    tables = str(Path(data_path("efit")).resolve()) + "/"
    reference = json.loads(calibration.REFERENCE_SET.read_text(encoding="utf-8"))
    products = {int(i["shot"]): i["files"]["product"]["path"] for i in reference["shots"]
                if i["files"]["product"] and i["files"]["product"]["exists"]}
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    baseline: str | None = None
    records: list[dict[str, Any]] = []
    for shot in [int(s) for s in args.shots.split(",")]:
        product = Path(data_path(products[shot]))
        ods = seed_study.load_product(product)
        times, _ = wrapper._select_times(ods, "auto", 0.001, None, None)
        if args.times:
            wanted = {int(t) for t in args.times.split(",")}
            times = [t for t in times if round(float(t) * 1e3) in wanted]
        diagnostics = thomson_check.thomson_diagnostics(shot)
        for time in times:
            tag = f"t{round(float(time) * 1e6):07d}"
            checkpoint = output / f"shot_{shot}" / tag / "records.json"
            if checkpoint.is_file():
                saved = json.loads(checkpoint.read_text())
                if saved.get("settings") == [s["name"] for s in settings]:
                    records.extend(saved["records"])
                    baseline = saved.get("baseline") or baseline
                    print(f"{shot}@{tag}: resumed", flush=True)
                    continue
            try:
                built, chosen = study.prepare_constraints(
                    shot, product, [time], workdir=output / f"shot_{shot}" / tag / "constraints", tables=tables,
                    tstep=0.001, average_window=0.0005, seed_study=seed_study,
                )
            except Exception as error:
                print(f"{shot}@{tag}: constraints failed: {error!r}", flush=True)
                continue
            slice_records = []
            for setting in settings:
                try:
                    record, baseline = run_slice(study, built, shot=shot, chosen=chosen, setting=setting,
                                                 output=output, efit=str(efit), diagnostics=diagnostics,
                                                 thomson_check=thomson_check, baseline=baseline)
                except RuntimeError:
                    raise  # a refused initialization is a contract violation, not a slice result
                except Exception as error:
                    record = {"setting": setting["name"], "shot": shot, "time_ms": round(float(time) * 1e3),
                              "error": repr(error)}
                slice_records.append(record)
                good = (record.get("evaluation") or {}).get("good")
                print(f"{shot}@{record['time_ms']} {setting['name']}: {record.get('exit_path')} good={good}", flush=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"settings": [s["name"] for s in settings], "baseline": baseline,
                                              "records": slice_records}, default=study._json_default))
            records.extend(slice_records)

    summary = criteria.summarize([r for r in records if "error" not in r])
    payload = {
        "schema_version": SCHEMA, "stage": args.stage,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved), "tables": tables,
        "criteria": criteria.CRITERIA, "settings": settings,
        "initialization_fingerprint": baseline, "summary": summary, "records": records,
    }
    table = args.table.expanduser()
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(payload, indent=1, default=study._json_default) + "\n")
    print(f"wrote {table}")
    for row in sorted(summary, key=lambda r: -criteria.good_fraction(r)):
        print(f"{row['setting']:48s} good {row['good']}/{row['slices']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
