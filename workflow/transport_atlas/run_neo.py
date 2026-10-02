"""Routine NEO over the same resolved Tier A states as routine TGLF (issue #1431).

Every state goes through the same :func:`vaft.process.transport_state.resolve_transport_state`
call that ``run_tglf.py`` makes. The upstream identity is therefore the same, and a
NEO and a TGLF result that share ``state_identity`` describe one plasma. NEO is a
profile code, so each state is **one** run with ``N_RADIAL`` surfaces spaced linearly
from the first to the last requested r/a. The default scan (0.30 ... 0.80 in steps of
0.10) is exactly the TGLF surface set, and anything not evenly spaced is refused rather
than silently resampled.

Output tree (``--out``)::

    run_manifest.json
    states.jsonl
    <shot>/<lineage>/<time_ms>/state.json      resolution, readiness, mapped fluxes
    <shot>/<lineage>/<time_ms>/neo/            native NEO directory + outputs.json

Fluxes are taken through :func:`vaft.machine_mapping.neoclassical.core_transport_from_neo`,
so they are the SI quantities the IMAS projection carries. NEO's own r/a and
rho_tor_norm (``out.neo.exprhon``) are both kept per surface.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vaft.process.transport_state import (
    inferred_ti_supported,
    DEFAULT_RHO_MAX,
    DEFAULT_SURFACES,
    assess_neo_readiness,
    physics_parameters,
    resolve_transport_state,
    run_identity,
)


def _tglf_driver():
    """The sibling driver, for its enumeration and product loading (one definition)."""
    name = "transport_atlas_run_tglf"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("run_tglf.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def radial_policy(surfaces) -> tuple[int, float, float]:
    """``(N_RADIAL, RMIN_OVER_A, RMIN_OVER_A_2)`` reproducing ``surfaces`` exactly."""
    values = np.asarray(sorted(float(s) for s in surfaces))
    if values.size < 2:
        raise ValueError("NEO's routine scan needs at least two surfaces")
    if not np.allclose(np.diff(values), values[1] - values[0], atol=1e-9):
        raise ValueError(
            f"NEO spaces N_RADIAL surfaces linearly; {values.tolist()} is not evenly "
            "spaced, so it cannot share surfaces with the TGLF scan"
        )
    return int(values.size), float(values[0]), float(values[-1])


def build_config(args) -> Any:
    from vaft.code.gacode.neo import NEOConfig

    n_radial, first, last = radial_policy(args.surfaces)
    backend = None
    if args.backend == "slurm":
        from vaft.code.slurm import SlurmBackend

        backend = SlurmBackend(partition=args.partition, account=args.account,
                               mode="batch", max_wait=args.max_wait)
    return NEOConfig(n_radial=n_radial, rmin_over_a=first, rmin_over_a_2=last,
                     backend=backend, timeout=args.timeout, memory_mb=int(args.mem_mb))


def project(native: Any, time: float) -> dict[str, Any]:
    """SI fluxes per NEO surface through ``core_transport_from_neo``."""
    from omas import ODS

    from vaft.machine_mapping.neoclassical import core_transport_from_neo

    ods = ODS(consistency_check=False)
    report = core_transport_from_neo(ods, native, time=time)
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    if f"{base}.grid_flux.rho_tor_norm" not in ods:
        return {"written": report["written"], "skipped": report["skipped"], "surfaces": []}
    rho = np.asarray(ods[f"{base}.grid_flux.rho_tor_norm"], dtype=float)
    r_over_a = np.asarray(native.normalisation.r_over_a, dtype=float)

    def column(path: str) -> Optional[np.ndarray]:
        return np.asarray(ods[path], dtype=float) if path in ods else None

    qe = column(f"{base}.electrons.energy.flux")
    ge = column(f"{base}.electrons.particles.flux")
    ions = []
    index = 0
    while f"{base}.ion.{index}.z_ion" in ods:
        if any(float(ods[f"{base}.ion.{index}.z_ion"]) == z for z, _, _ in ions):
            raise ValueError("two NEO ion species share a charge; rows are keyed by charge")
        ions.append((float(ods[f"{base}.ion.{index}.z_ion"]),
                     column(f"{base}.ion.{index}.energy.flux"),
                     column(f"{base}.ion.{index}.particles.flux")))
        index += 1
    rows = []
    for i in range(rho.size):
        rows.append({
            "r_over_a": float(r_over_a[i]), "rho_tor_norm": float(rho[i]),
            "electron_energy_flux_W_m2": None if qe is None else float(qe[i]),
            "electron_particle_flux_m2_s": None if ge is None else float(ge[i]),
            # Keyed by charge: NEO writes no labels, and charge is what TGLF and NEO
            # demonstrably share for the H+/C6+ species list both were given.
            "ion_energy_flux_W_m2": {f"z={z:g}": None if q is None else float(q[i]) for z, q, _ in ions},
            "ion_particle_flux_m2_s": {f"z={z:g}": None if g is None else float(g[i]) for z, _, g in ions},
        })
    code = {k: ods[f"core_transport.code.{k}"] for k in ("name", "version")
            if f"core_transport.code.{k}" in ods}
    return {"written": report["written"], "skipped": report["skipped"], "code": code,
            "surfaces": rows}


def run_state(job: dict, config: Any) -> dict:
    """Run (or reuse) one state's NEO case; an infrastructure error is recorded, not raised."""
    try:
        return _run_state(job, config)
    except Exception as error:  # noqa: BLE001 - recorded per state, as in run_tglf.run_surface
        return {"run_identity": job["identity"], "status": "failed", "runtime_status": "error",
                "returncode": None, "elapsed_s": None,
                "errors": [f"{type(error).__name__}: {error}"], "version": None,
                "effective_charge": None, "cached": False}


def _run_state(job: dict, config: Any) -> dict:
    from vaft.code.gacode.neo import run_neo_case

    workdir = Path(job["workdir"])
    record_path = workdir / "record.json"
    # A solved record is only reusable with the outputs.json it was written from;
    # a pruned native directory that kept record.json is re-run, not projected.
    if record_path.is_file() and (workdir / "outputs.json").is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("run_identity") == job["identity"] and previous.get("status") == "solved":
            previous["cached"] = True
            return previous
    workdir.mkdir(parents=True, exist_ok=True)
    result = run_neo_case(job["profile"], workdir, config, check=False)
    native = result.outputs_native
    record = {
        "run_identity": job["identity"],
        "status": "solved" if result.ok else "failed",
        "runtime_status": result.runtime_status,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "errors": [] if native is None else list(getattr(native, "errors", ()) or ()),
        "version": None if native is None else native.version,
        "effective_charge": None if native is None else _scalar(native.effective_charge),
        "cached": False,
    }
    if native is not None:
        native.write_json(workdir / "outputs.json")
    record_path.write_text(json.dumps(record, indent=1, default=_json), encoding="utf-8")
    return record


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array.tolist()


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--core-profiles-dir", type=Path,
                        help="per-shot <shot>.json.gz core_profiles instead of the FileDB stage")
    parser.add_argument("--use-inferred-ti", action="store_true",
                        help="use a core_profiles product's own inferred T_i (lane K #1426). Off: "
                             "such a state is refused and the atlas keeps Ti = Te (#1414)")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--lineages", nargs="*", default=["magnetics", "electron_kinetic"])
    parser.add_argument("--times-ms", type=int, nargs="*")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--surfaces", type=float, nargs="*", default=list(DEFAULT_SURFACES))
    parser.add_argument("--ti-te-ratio", default="policy")
    parser.add_argument("--rho-max", type=float, default=DEFAULT_RHO_MAX)
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--partition", default="lowpri-short")
    parser.add_argument("--account")
    parser.add_argument("--mem-mb", type=int, default=2048,
                        help="memory reserved per run [MB], as in run_tglf.py")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-wait", type=float, default=6 * 3600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import vaft

    tglf = _tglf_driver()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    labels = tglf.load_labels(args.labels)
    shots = args.shots or sorted({shot for shot, _ in labels})
    states, counts = tglf.enumerate_states(args.filedb, labels, shots, args.lineages,
                                           args.core_profiles_dir)
    if args.times_ms:
        states = [s for s in states if s[0].time_ms in set(args.times_ms)]
    if args.max_states:
        states = states[: args.max_states]
    config = build_config(args)
    revision = tglf.gacode_revision(config)
    parameters = physics_parameters(config)
    ratio = args.ti_te_ratio if args.ti_te_ratio == "policy" else float(args.ti_te_ratio)

    loaded: dict[Path, Any] = {}

    def cached(path: Path) -> Any:
        if path not in loaded:
            loaded[path] = tglf._load(path)
        return loaded[path]

    prepared = []
    for key, label, source, cp_path, eq_path in states:
        ods = tglf.compose(cached(eq_path), cached(cp_path))
        state = resolve_transport_state(
            ods, key, efit_quality=label, quality_source=source, ti_te_ratio=ratio,
            use_stored_inferred_ti=args.use_inferred_ti, rho_max=args.rho_max,
            inputs={"core_profiles": {"path": str(cp_path), "sha256": tglf._sha256(cp_path)},
                    "equilibrium": {"path": str(eq_path), "sha256": tglf._sha256(eq_path)},
                    "profile_mapped_on": "magnetics"},
        )
        readiness = assess_neo_readiness(state, args.surfaces)
        state_dir = out / key.slug()
        job = None
        if readiness.runnable:
            job = {"workdir": str(state_dir / "neo"), "profile": state.profile,
                   "identity": run_identity(state, solver="neo", parameters=parameters,
                                            solver_revision=revision)}
        prepared.append((state, readiness, job, state_dir))

    results: dict[str, dict] = {}
    jobs = [job for _, _, job, _ in prepared if job is not None]
    if not args.dry_run and jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_state, job, config): job for job in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                job = futures[future]
                results[job["workdir"]] = future.result()
                print(f"[{done}/{len(jobs)}] {job['workdir']} {results[job['workdir']]['status']}"
                      f" {results[job['workdir']].get('runtime_status')}", flush=True)

    from vaft.code.gacode.neo.outputs import NeoOutputs

    with open(out / "states.jsonl", "w", encoding="utf-8") as index:
        for state, readiness, job, state_dir in prepared:
            record = None if job is None else results.get(job["workdir"])
            if args.dry_run:
                status = "dry_run"
            elif not state.resolved or not readiness.runnable:
                status = "not_ready"
            elif record is None:
                status = "not_run"
            else:
                status = record["status"]
            mapping = None
            reasons = list(state.reasons)
            if status == "solved":
                try:
                    native = NeoOutputs.read_json(Path(job["workdir"]) / "outputs.json")
                    mapping = project(native, state.key.time_efit_s)
                    # NEO solves every surface of the profile; mark the ones whose input
                    # depends on an inferred-Ti gap fill so the atlas never partitions them.
                    for row in mapping["surfaces"]:
                        try:
                            row["ti_supported"] = bool(
                                inferred_ti_supported(state, row["r_over_a"], solver="neo"))
                        except Exception:  # noqa: BLE001 - an unjudgeable surface is unsupported
                            row["ti_supported"] = False
                except Exception as error:  # noqa: BLE001 - one state's row, not the batch
                    # A missing or unreadable outputs.json fails this state; raising
                    # here would truncate states.jsonl and lose every finished state.
                    status = "failed"
                    reasons.append(f"projection_error: {type(error).__name__}: {error}")
            if mapping is not None:
                requested = sorted(float(s) for s in args.surfaces)
                got = sorted(row["r_over_a"] for row in mapping["surfaces"])
                if not got:
                    # NEO exited cleanly but the mapper wrote nothing (no exprhon /
                    # expnorm / species table): a state without fluxes is not solved.
                    status = "failed"
                    reasons += [f"no_surface_mapped: {reason}" for reason in mapping["skipped"]]
                elif len(got) != len(requested) or not np.allclose(got, requested, atol=1e-4):
                    status = "partial"
            payload = {**state.summary(), "solver": "neo", "status": status, "reasons": reasons,
                       "readiness": readiness.summary(), "run": record,
                       "run_identity": None if job is None else job["identity"],
                       "core_transport": mapping, "neo_parameters": parameters,
                       "gacode_revision": revision}
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "state.json").write_text(json.dumps(payload, indent=1, default=_json),
                                                  encoding="utf-8")
            index.write(json.dumps({k: payload[k] for k in (
                "shot", "time_efit_s", "efit_lineage", "efit_quality", "quality_source",
                "ti_lineage", "status", "reasons", "state_identity")}, default=_json) + "\n")

    status_counts: dict[str, int] = {}
    for line in (out / "states.jsonl").read_text(encoding="utf-8").splitlines():
        status = json.loads(line)["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv[:1] + list(argv if argv is not None else sys.argv[1:]),
        "vaft": vaft.__file__, "vaft_version": vaft.__version__,
        "vaft_git": tglf._git_sha(Path(vaft.__file__).parent.parent),
        "gacode_revision": revision, "neo_parameters": parameters,
        "enumeration": counts, "states_run": len(prepared), "jobs": len(jobs),
        "status_counts": status_counts,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str),
                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("states_run", "jobs", "status_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
