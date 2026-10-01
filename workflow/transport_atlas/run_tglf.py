"""Routine multi-surface TGLF over the #1331 Tier A states (issue #1428).

Reads a campaign FileDB (``omas/core_profiles``, ``omas/efit/magnetic``,
``omas/electron_efit``) and the #1331 slice-label table, enumerates every
good/admissible state of both EFIT lineages, resolves each through
:func:`vaft.process.transport_state.resolve_transport_state`, and runs native TGLF on
every ready surface through the existing runner and execution backend.

Output tree (``--out``)::

    run_manifest.json                       enumeration counts, settings, versions
    states.jsonl                            one line per state (summary + status)
    <shot>/<lineage>/<time_ms>/state.json   resolution, readiness, mapping, surfaces
    <shot>/<lineage>/<time_ms>/r0.30/       native TGLF directory + record.json

Status vocabulary per state: ``not_ready`` (the state did not resolve or no solver
input could be built), ``skipped`` (resolved but no surface ready), ``failed`` (every
run failed), ``partial`` and ``solved``.  A solver timeout is a failed run whose
``runtime_status`` says ``timeout``; it is recorded, not raised.

Unreconstructible slices are never enumerated: they are counted in the manifest and
produce no rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from vaft.process.transport_state import (
    DEFAULT_RHO_MAX,
    DEFAULT_SURFACES,
    DEFAULT_TIME_TOLERANCE_S,
    EFIT_LABELS,
    TransportStateKey,
    assess_tglf_readiness,
    physics_parameters,
    resolve_transport_state,
    run_identity,
)

STAGES = {
    "core_profiles": ("omas/core_profiles", "core_profiles.json.gz"),
    "magnetics-only": ("omas/efit/magnetic", "efit.json.gz"),
    "electron-kinetic": ("omas/electron_efit", "electron_efit.json.gz"),
}


# --------------------------------------------------------------------------- inputs


def load_labels(path: Path) -> dict[tuple[int, int], str]:
    """``{(shot, time_ms): label}`` from a #1331 analysis file (its ``labels`` list)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["labels"] if isinstance(payload, dict) else payload
    return {(int(r["shot"]), int(r["time_ms"])): str(r["label"]) for r in rows}


def product(filedb: Path, stage: str, shot: int) -> tuple[Optional[Path], dict]:
    subdir, name = STAGES[stage]
    root = Path(filedb) / subdir / str(shot)
    manifest = {}
    manifests = sorted((root / "metadata").glob("*.json"))
    if manifests:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    path = root / "output" / name
    return (path if path.is_file() else None), manifest


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    from vaft.omas import load

    return load(path)


def compose(equilibrium_ods: Any, profiles_ods: Any) -> Any:
    """One ODS holding the lineage's equilibrium and the shot's core_profiles."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium"] = equilibrium_ods["equilibrium"]
    ods["core_profiles"] = profiles_ods["core_profiles"]
    if "dataset_description" in profiles_ods:
        ods["dataset_description"] = profiles_ods["dataset_description"]
    return ods


def enumerate_states(filedb: Path, labels: dict, shots: Iterable[int], lineages: Iterable[str]):
    """Yield ``(key, efit_label, quality_source)`` for every good/admissible state.

    Returns the generator's bookkeeping through the ``counts`` dict it fills.
    """
    counts: dict[str, Any] = {"excluded_unreconstructible": 0, "excluded_unlabelled": 0,
                              "missing_products": [], "states": 0}
    states = []
    for shot in shots:
        cp_path, cp_manifest = product(filedb, "core_profiles", shot)
        if cp_path is None or cp_manifest.get("status") not in (None, "success"):
            counts["missing_products"].append({"shot": shot, "stage": "core_profiles",
                                               "status": cp_manifest.get("status")})
            continue
        for lineage in lineages:
            eq_path, eq_manifest = product(filedb, lineage, shot)
            if eq_path is None or eq_manifest.get("status") not in (None, "success"):
                counts["missing_products"].append({"shot": shot, "stage": lineage,
                                                   "status": eq_manifest.get("status")})
                continue
            eq = _load(eq_path)
            times = np.atleast_1d(np.asarray(eq["equilibrium.time"], dtype=float))
            if lineage == "magnetics-only":
                # A magnetics state exists where a core_profiles slice sits on the slice.
                cp_times = np.atleast_1d(np.asarray(_load(cp_path)["core_profiles.time"], dtype=float))
                candidates = [t for t in times
                              if np.min(np.abs(cp_times - t)) <= DEFAULT_TIME_TOLERANCE_S]
                source = "criteria"
            else:
                candidates = list(times)
                source = "magnetic_slice_at_same_time"
            for t in candidates:
                key = TransportStateKey(int(shot), float(t), lineage)
                label = labels.get((int(shot), key.time_ms))
                if label is None:
                    counts["excluded_unlabelled"] += 1
                    continue
                if label not in EFIT_LABELS:
                    counts["excluded_unreconstructible"] += 1
                    continue
                states.append((key, label, source, cp_path, eq_path))
    counts["states"] = len(states)
    return states, counts


# --------------------------------------------------------------------------- running


def gacode_revision() -> Optional[str]:
    root = os.environ.get("GACODE_ROOT")
    if not root:
        return None
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_config(args) -> Any:
    from vaft.code.gacode.tglf import TGLFConfig

    backend = None
    if args.backend == "slurm":
        from vaft.code.slurm import SlurmBackend

        backend = SlurmBackend(partition=args.partition, account=args.account,
                               mode="batch", max_wait=args.max_wait)
    return TGLFConfig(sat_rule=args.sat_rule, use_bper=args.use_bper,
                      backend=backend, timeout=args.timeout)


def _local_summary(local: Any) -> dict[str, Any]:
    """The TGLF local inputs the atlas uses as drive coordinates (dimensionless)."""
    rmin = float(local.rmin_loc)
    q = float(local.q_loc)
    shear = float(local.q_prime_loc) * (rmin / q) ** 2 if q else float("nan")
    norm = local.normalisation
    return {
        "r_over_a": float(local.rho),
        "rmin_loc": rmin, "rmaj_loc": float(local.rmaj_loc), "q": q, "shear": shear,
        "kappa": float(local.kappa_loc), "delta": float(local.delta_loc),
        "betae": float(local.betae), "xnue": float(local.xnue), "zeff": float(local.zeff),
        "species": list(local.names),
        "a_over_ln": [float(v) for v in local.rlns],
        "a_over_lt": [float(v) for v in local.rlts],
        "taus": [float(v) for v in local.taus],
        "as": [float(v) for v in local.as_],
        "vexb_shear": None if local.vexb_shear is None else float(local.vexb_shear),
        "normalisation": None if norm is None else {
            "a_m": float(norm.a), "b_unit_T": float(norm.b_unit),
            "q_gb_W_m2": float(norm.energy_flux), "gamma_gb_m2_s": float(norm.particle_flux),
        },
    }


def run_surface(job: dict, config: Any) -> dict:
    """Run (or reuse) one surface. Returns its record; never raises on a solver failure."""
    from vaft.code.gacode.tglf import run_tglf_case
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    workdir = Path(job["workdir"])
    record_path = workdir / "record.json"
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("run_identity") == job["identity"] and previous.get("status") == "solved":
            previous["cached"] = True
            return previous
    workdir.mkdir(parents=True, exist_ok=True)
    result = run_tglf_case(job["profile"], job["r_over_a"], workdir, config, check=False)
    native = result.outputs_native
    record = {
        "r_over_a": job["r_over_a"],
        "run_identity": job["identity"],
        "status": "solved" if result.ok else "failed",
        "runtime_status": result.runtime_status,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "errors": [] if native is None else list(native.errors),
        "version": None if native is None else native.version,
        "local": job["local"],
        "cached": False,
    }
    if native is not None:
        native.write_json(workdir / "outputs.json")
    record_path.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    return record


def project_state(state, surfaces: list[dict], jobs: dict) -> dict:
    """Project the solved surfaces of one state through ``core_transport_from_tglf``."""
    from omas import ODS

    from vaft.code.gacode.tglf.outputs import TglfOutputs
    from vaft.machine_mapping.turbulence import core_transport_from_tglf

    pairs = []
    for record in surfaces:
        if record.get("status") != "solved":
            continue
        job = jobs[record["r_over_a"]]
        native = TglfOutputs.read_json(Path(job["workdir"]) / "outputs.json")
        pairs.append((job["local_input"], native))
    if not pairs:
        return {"written": [], "skipped": ["no solved surface"], "surfaces": []}
    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, pairs, state.profile, time=state.key.time_efit_s)
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    rho = np.asarray(ods[f"{base}.grid_flux.rho_tor_norm"], dtype=float)
    radii = sorted(float(local.rho) for local, _ in pairs)

    def column(path: str) -> list:
        return [float(v) for v in np.asarray(ods[path], dtype=float)] if path in ods else [None] * len(radii)

    rows = []
    qe = column(f"{base}.electrons.energy.flux")
    ge = column(f"{base}.electrons.particles.flux")
    ions = []
    index = 0
    while f"{base}.ion.{index}.label" in ods:
        ions.append((str(ods[f"{base}.ion.{index}.label"]),
                     column(f"{base}.ion.{index}.energy.flux"),
                     column(f"{base}.ion.{index}.particles.flux")))
        index += 1
    for i, radius in enumerate(radii):
        rows.append({
            "r_over_a": radius, "rho_tor_norm": float(rho[i]),
            "electron_energy_flux_W_m2": qe[i], "electron_particle_flux_m2_s": ge[i],
            "ion_energy_flux_W_m2": {label: q[i] for label, q, _ in ions},
            "ion_particle_flux_m2_s": {label: g[i] for label, _, g in ions},
        })
    code = {k: ods[f"core_transport.model.{report['model']}.code.{k}"]
            for k in ("name", "version") if f"core_transport.model.{report['model']}.code.{k}" in ods}
    return {"written": report["written"], "skipped": report["skipped"], "code": code,
            "surfaces": rows}


def state_status(readiness, surfaces: list[dict], resolved: bool) -> str:
    if not resolved:
        return "not_ready"
    if not readiness.runnable:
        return "skipped" if "no_ready_surface" in readiness.reasons else "not_ready"
    ran = [s for s in surfaces if s.get("status") in ("solved", "failed")]
    solved = [s for s in ran if s["status"] == "solved"]
    if not ran:
        return "skipped"
    if not solved:
        return "failed"
    return "solved" if len(solved) == len(readiness.surfaces) else "partial"


# --------------------------------------------------------------------------- main


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="#1331 analysis JSON (its 'labels' list)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--lineages", nargs="*", default=["magnetics-only", "electron-kinetic"])
    parser.add_argument("--times-ms", type=int, nargs="*", help="restrict to these slice times")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--surfaces", type=float, nargs="*", default=list(DEFAULT_SURFACES))
    parser.add_argument("--ti-te-ratio", default="policy",
                        help="'policy' (vest.yaml, #1414) or a number")
    parser.add_argument("--rho-max", type=float, default=DEFAULT_RHO_MAX)
    parser.add_argument("--sat-rule", type=int, default=3)
    parser.add_argument("--use-bper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--partition", default="lowpri-short")
    parser.add_argument("--account")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-wait", type=float, default=6 * 3600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="resolve and assess only")
    args = parser.parse_args(argv)

    import vaft

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    shots = args.shots or sorted({shot for shot, _ in labels})
    states, counts = enumerate_states(args.filedb, labels, shots, args.lineages)
    if args.times_ms:
        states = [s for s in states if s[0].time_ms in set(args.times_ms)]
    if args.max_states:
        states = states[: args.max_states]
    config = build_config(args)
    revision = gacode_revision()
    parameters = physics_parameters(config)
    ratio = args.ti_te_ratio if args.ti_te_ratio == "policy" else float(args.ti_te_ratio)

    loaded: dict[Path, Any] = {}

    def cached(path: Path) -> Any:
        if path not in loaded:
            loaded[path] = _load(path)
        return loaded[path]

    prepared = []
    jobs = []
    for key, label, source, cp_path, eq_path in states:
        ods = compose(cached(eq_path), cached(cp_path))
        state = resolve_transport_state(
            ods, key, efit_label=label, quality_source=source, ti_te_ratio=ratio,
            rho_max=args.rho_max,
            inputs={"core_profiles": {"path": str(cp_path), "sha256": _sha256(cp_path)},
                    "equilibrium": {"path": str(eq_path), "sha256": _sha256(eq_path)},
                    "profile_mapped_on": "magnetics-only"},
        )
        readiness = assess_tglf_readiness(state, args.surfaces, config=config)
        state_dir = out / key.slug()
        state_jobs = {}
        for surface in readiness.surfaces:
            if not surface.ready:
                continue
            identity = run_identity(state, solver="tglf", parameters=parameters,
                                    surface=surface.r_over_a, solver_revision=revision)
            job = {"workdir": str(state_dir / f"r{surface.r_over_a:.2f}"),
                   "r_over_a": surface.r_over_a, "identity": identity,
                   "profile": state.profile, "local_input": surface.local_input,
                   "local": _local_summary(surface.local_input)}
            state_jobs[surface.r_over_a] = job
            jobs.append(job)
        prepared.append((state, readiness, state_jobs, state_dir))

    results: dict[str, dict] = {}
    if not args.dry_run and jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_surface, job, config): job for job in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                job = futures[future]
                record = future.result()
                results[job["workdir"]] = record
                print(f"[{done}/{len(jobs)}] {job['workdir']} {record['status']}"
                      f" {record.get('runtime_status')}", flush=True)

    with open(out / "states.jsonl", "w", encoding="utf-8") as index:
        for state, readiness, state_jobs, state_dir in prepared:
            surfaces = []
            for surface in readiness.surfaces:
                entry = {"r_over_a": surface.r_over_a, "readiness": surface.status,
                         "detail": surface.detail}
                job = state_jobs.get(surface.r_over_a)
                if job is not None:
                    entry["run_identity"] = job["identity"]
                    entry["local"] = job["local"]
                    record = results.get(job["workdir"])
                    if record is not None:
                        entry.update({k: record.get(k) for k in
                                      ("status", "runtime_status", "returncode", "errors",
                                       "version", "elapsed_s", "cached")})
                    else:
                        entry["status"] = "not_run"
                else:
                    entry["status"] = "not_ready"
                surfaces.append(entry)
            status = state_status(readiness, surfaces, state.resolved) if not args.dry_run else "dry_run"
            mapping = project_state(state, surfaces, state_jobs) if status in ("solved", "partial") else None
            payload = {**state.summary(), "solver": "tglf", "status": status,
                       "readiness": readiness.summary(), "surfaces": surfaces,
                       "core_transport": mapping, "tglf_parameters": parameters,
                       "gacode_revision": revision}
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "state.json").write_text(json.dumps(payload, indent=1, default=float),
                                                  encoding="utf-8")
            index.write(json.dumps({k: payload[k] for k in (
                "shot", "time_efit_s", "efit_lineage", "efit_label", "quality_source",
                "ti_lineage", "status", "reasons", "state_identity")}, default=float) + "\n")

    manifest = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv, "vaft": vaft.__file__,
        "vaft_git": _git_sha(Path(vaft.__file__).parent.parent),
        "gacode_revision": revision, "tglf_parameters": parameters,
        "enumeration": counts, "states_run": len(prepared), "surface_jobs": len(jobs),
        "status_counts": _count(out / "states.jsonl"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str),
                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("states_run", "surface_jobs", "status_counts")}))
    return 0


def _git_sha(root: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _count(index: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        status = json.loads(line)["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
