"""Linear CGYRO sensitivity to the impurity composition (Lane L, #1569; Lane Y path #1484).

The same three compositions as ``run_tglf_composition.py`` (``c6_zeff2``,
``co_stripped``, ``co_openadas``), on one or two of Lane Y's states and surfaces,
with Lane Y's linear settings: the state is resolved by Lane T's
``resolve_transport_state``, each composition is put into its profile per surface
(:func:`vaft.process.impurity.surface_composition_profile`), and Lane Y's own job
functions run CGYRO (``workflow/gyrokinetic/run_linear.py`` is imported, not copied)
and the linear TGLF at the same ky.  The species each code saw are recorded per run
(``local.json``: names, charges, densities), so a reader can check that CGYRO and
TGLF were given identical species.

Run inside one allocation, as Lane Y's README asks, e.g.::

    sbatch -p lowpri-short -n 32 --mem=32G -t 06:00:00 --wrap \\
      "~/work/lane-l-cgyro.sh --out ~/runs/lane-l/cgyro --n-mpi 8 --workers 4"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[list[str]] = None) -> int:
    lane_y = _module("workflow/gyrokinetic/run_linear.py", "lane_y_run_linear")
    tglf_composition = _module("workflow/impurity_zeff/run_tglf_composition.py", "lane_l_tglf_composition")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--states-csv", type=Path, required=True, help="Lane K atlas/v1/state.csv (criteria v2)")
    parser.add_argument("--ages", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--states", nargs="*", default=["39916:0.321:magnetics"])
    parser.add_argument("--surfaces", type=float, nargs="*", default=[0.6, 0.8])
    parser.add_argument("--ky", type=float, nargs="*", default=list(lane_y.DEFAULT_KY))
    parser.add_argument("--field-models", nargs="*", default=["es"])
    parser.add_argument("--cases", nargs="*", default=list(tglf_composition.CASES),
                        choices=tglf_composition.CASES)
    for name, default in (("--n-energy", 8), ("--n-xi", 24), ("--n-theta", 32), ("--n-radial", 8),
                          ("--n-mpi", 8), ("--workers", 4), ("--delta-t-method", 1)):
        parser.add_argument(name, type=int, default=default)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument("--delta-t", type=float, default=0.01)
    parser.add_argument("--freq-tol", type=float, default=1e-3)
    parser.add_argument("--gacode-home")
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--slurm-mode", default="auto")
    parser.add_argument("--partition", default="lowpri-short")
    parser.add_argument("--account")
    parser.add_argument("--mem-mb", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=4 * 3600.0)
    parser.add_argument("--max-wait", type=float, default=24 * 3600.0)
    parser.add_argument("--tglf-sat-rule", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    lane_y._assert_checkout()
    import vaft
    from vaft.code.gacode.cgyro import prepare_cgyro_input
    from vaft.process.transport_state import resolve_transport_state

    helpers = lane_y._lane_t_helpers()
    wanted = [(int(s.split(":")[0]), float(s.split(":")[1]), s.split(":")[2]) for s in args.states]
    labels = tglf_composition._labels(args.states_csv, ("good", "admissible"))
    ages = tglf_composition._ages(args.ages)
    states, counts = helpers.enumerate_states(args.filedb, labels, sorted({w[0] for w in wanted}),
                                              sorted({w[2] for w in wanted}))
    chosen = [s for s in states if any(s[0].shot == w[0] and s[0].efit_lineage == w[2]
                                       and abs(s[0].time_efit_s - w[1]) < 6e-4 for w in wanted)]
    backend = lane_y.build_backend(args)
    args.out.mkdir(parents=True, exist_ok=True)
    cgyro_jobs, tglf_jobs, written = [], [], []
    for key, label, source, cp_path, eq_path in chosen:
        ods = helpers.compose(helpers._load(eq_path), helpers._load(cp_path))
        state = resolve_transport_state(ods, key, efit_quality=label, quality_source=source)
        if not state.resolved:
            print(f"{key}: not resolved {state.reasons}", file=sys.stderr)
            continue
        age = ages.get((key.shot, key.time_ms))
        for r_over_a in args.surfaces:
            try:
                profiles, notes = tglf_composition.variants(state, r_over_a, age)
            except Exception as exc:  # noqa: BLE001 -- one surface, not the allocation
                print(f"{key} r/a={r_over_a}: composition failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            for case in args.cases:
                try:
                    local = prepare_cgyro_input(profiles[case], r_over_a)
                except ValueError as exc:
                    print(f"{key} {case} r/a={r_over_a}: {exc}", file=sys.stderr)
                    continue
                absent = local.tglf.check_tglf_requirements()
                if absent:
                    print(f"{key} {case} r/a={r_over_a}: lacks {absent}", file=sys.stderr)
                    continue
                surface_dir = args.out / case / key.slug() / f"r{r_over_a:.2f}"
                surface_dir.mkdir(parents=True, exist_ok=True)
                (surface_dir / "local.json").write_text(json.dumps({
                    "state": state.summary(), "case": case, "composition": notes[case],
                    "r_over_a": r_over_a, "names": list(local.names),
                    "species": {k: np.asarray(v).tolist() for k, v in local.species.items()},
                    "z_eff": getattr(local, "z_eff", None), "provenance": local.provenance,
                }, indent=1, default=str), encoding="utf-8")
                written.append(str(surface_dir))
                state_key = {"shot": key.shot, "time_efit_s": key.time_efit_s, "efit_lineage": key.efit_lineage,
                             "state_identity": state.identity, "case": case}
                for field_model in args.field_models:
                    for ky in args.ky:
                        meta = {**state_key, "efit_quality": label, "r_over_a": r_over_a,
                                "field_model": field_model, "ky": float(ky)}
                        cgyro_jobs.append({"workdir": str(surface_dir / field_model / "cgyro" / f"ky{ky:.3f}"),
                                           "local": local, "state_key": state_key, "meta": meta,
                                           "config": lane_y.cgyro_config(args, field_model, ky, backend)})
                        tglf_jobs.append({"workdir": str(surface_dir / field_model / "tglf" / f"ky{ky:.3f}"),
                                          "local": local, "state_key": state_key, "meta": meta,
                                          "config": lane_y.tglf_linear_config(field_model, ky, args)})
    print(f"{len(chosen)} states, {len(written)} surface-cases, {len(cgyro_jobs)} CGYRO + {len(tglf_jobs)} TGLF",
          flush=True)
    records: list[dict] = []
    if not args.dry_run:
        for job in tglf_jobs:
            records.append(lane_y.run_tglf_job(job))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            futures = {pool.submit(lane_y.run_cgyro_job, job): job for job in cgyro_jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()
                records.append(record)
                print(f"[{done}/{len(cgyro_jobs)}] {futures[future]['workdir']} {record.get('status')} "
                      f"gamma={record.get('gamma')} {record.get('elapsed_s')}", flush=True)
        with open(args.out / "runs.jsonl", "w", encoding="utf-8") as index:
            for record in records:
                index.write(json.dumps({k: v for k, v in record.items() if k != "provenance"}, default=float) + "\n")
    manifest = {"producer": "workflow/impurity_zeff/run_cgyro_composition.py (Lane L, #1569)",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "argv": sys.argv,
                "vaft": vaft.__file__, "vaft_git": helpers._git_sha(ROOT), "states": [list(w) for w in wanted],
                "enumeration": counts, "cgyro_jobs": len(cgyro_jobs), "tglf_jobs": len(tglf_jobs),
                "resolution": lane_y.cgyro_config(args, "es", 0.3, None).resolution(), "n_mpi": args.n_mpi}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
