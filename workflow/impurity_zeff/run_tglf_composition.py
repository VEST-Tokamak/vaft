"""TGLF sensitivity to the impurity composition on the Tier A good states (Lane L, #1565 / #1569).

Each state is resolved by Lane T's own path -- ``resolve_transport_state`` with its
defaults, so T_i, geometry and the electron profiles are Lane T's (#1428) -- and
then run with the same TGLF settings under three compositions that differ in
nothing else:

* ``c6_zeff2``     -- the Lane T atlas composition: H+ and C6+ at Z_eff = 2;
* ``co_stripped``  -- the VEST preset: H+, C6+ and O8+ at 1/86 each, Z_eff = 2;
* ``co_openadas``  -- H+, C and O with OpenADAS charge states (transient, at the
  plasma age), elemental density normalised to an n_e-weighted Z_eff of 2,
  lumped per surface (:func:`vaft.process.impurity.surface_composition_profile`).

The first two isolate "add oxygen at the same Z_eff"; the last two "real charge
states and the dilution they imply".  Lane T's driver functions are imported
(``workflow/transport_atlas/run_tglf.py``) -- enumeration, configuration, the
runner and its cache -- and the spectral descriptors come from its atlas builder,
so every number is defined as in the transport atlas.

Inputs it cannot compute on the cluster are passed in:

* ``--states``: the criteria-v2 labels (Lane K ``atlas/v1/state.csv``; the cluster's
  analysis JSON predates criteria v2);
* ``--ages``:   the plasma age per state (Lane L ``atlas/impurity/states.csv``; the
  cluster FileDB has no diagnostics product).  A state without one runs coronal.

Output (``--out``): ``<case>/<shot>/<lineage>/<ms>/r0.xx/`` native runs,
``tglf_composition.csv`` (one row per state x surface x case) and
``tglf_composition_pairs.csv`` (one row per state x surface: the three cases
side by side and their ratios), plus ``MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ("c6_zeff2", "co_stripped", "co_openadas")
WEIGHTS = {"C": 1.0, "O": 1.0}


def _module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _labels(states_csv: Path, qualities: tuple[str, ...]) -> dict[tuple[int, int], str]:
    labels = {}
    for r in csv.DictReader(open(states_csv)):
        if r["efit_lineage"] == "magnetics" and r["efit_quality"] in qualities:
            labels[(int(r["shot"]), int(round(float(r["time_efit_s"]) * 1e3)))] = r["efit_quality"]
    return labels


def _ages(ages_csv: Optional[Path]) -> dict[tuple[int, int], float]:
    ages = {}
    if ages_csv is None:
        return ages
    for r in csv.DictReader(open(ages_csv)):
        if r.get("plasma_age_s") not in (None, "", "None"):
            ages[(int(r["shot"]), int(round(float(r["time_efit_s"]) * 1e3)))] = float(r["plasma_age_s"])
    return ages


def _profile_sha(profile: Any) -> str:
    digest = hashlib.sha256()
    for name in ("z", "mass", "ni", "ti", "ne", "te"):
        value = getattr(profile, name)
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes())
    digest.update(json.dumps(list(profile.name)).encode())
    return digest.hexdigest()


def variants(state: Any, r_over_a: float, age: Optional[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The three profiles of one surface, and what each one assumed."""
    from vaft.process.impurity import (
        resolve_impurity_composition,
        resolve_radial_composition,
        surface_composition_profile,
    )

    profile = state.profile
    out = {"c6_zeff2": profile}
    notes: dict[str, Any] = {"c6_zeff2": {"composition": dict(state.composition)}}
    preset = resolve_impurity_composition(machine_preset="vest", shot=state.key.shot)
    out["co_stripped"] = surface_composition_profile(profile, preset, r_over_a)
    notes["co_stripped"] = {"composition": "vest impurity_model, fully stripped, Z_eff = 2"}
    ionization = "transient" if age is not None and age > 0.0 else "coronal"
    radial = resolve_radial_composition(
        np.asarray(profile.te) * 1e3, np.asarray(profile.ne) * 1e19, np.asarray(profile.rho), WEIGHTS,
        normalization="ne_weighted_mean", target_zeff=2.0, ionization=ionization,
        plasma_age_s=age if ionization == "transient" else None)
    out["co_openadas"] = surface_composition_profile(profile, radial, r_over_a)
    notes["co_openadas"] = {"ionization": ionization, "plasma_age_s": age, "scale": radial.scale,
                            "lumped_charge": out["co_openadas"].provenance["ni"]["lumped_charge"]}
    return out, notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True, help="Lane K atlas/v1/state.csv (criteria v2)")
    parser.add_argument("--ages", type=Path, help="Lane L atlas/impurity/states.csv (plasma_age_s)")
    parser.add_argument("--qualities", nargs="*", default=["good"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--surfaces", type=float, nargs="*")
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=CASES)
    parser.add_argument("--sat-rule", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--field-model", required=True)
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--partition", default="lowpri-short")
    parser.add_argument("--account")
    parser.add_argument("--mem-mb", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-wait", type=float, default=6 * 3600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    lane_t = _module("workflow/transport_atlas/run_tglf.py", "lane_t_run_tglf")
    atlas = _module("workflow/transport_atlas/build_atlas.py", "lane_t_build_atlas")
    from vaft.code.gacode.tglf import prepare_tglf_input
    from vaft.process.transport_state import DEFAULT_SURFACES, assess_tglf_readiness, resolve_transport_state

    surfaces = list(args.surfaces or DEFAULT_SURFACES)
    labels = _labels(args.states, tuple(args.qualities))
    ages = _ages(args.ages)
    shots = args.shots or sorted({shot for shot, _ in labels})
    enumerated, counts = lane_t.enumerate_states(args.filedb, labels, shots, ["magnetics"], None)
    if args.max_states:
        enumerated = enumerated[: args.max_states]
    config = lane_t.build_config(args)
    revision = lane_t.gacode_revision(config)
    parameters = lane_t.physics_parameters(config)
    tglf_config = lane_t.config_label(args.sat_rule, args.field_model)
    out = args.out / tglf_config
    out.mkdir(parents=True, exist_ok=True)

    loaded: dict[Path, Any] = {}

    def cached(path: Path) -> Any:
        if path not in loaded:
            loaded[path] = lane_t._load(path)
        return loaded[path]

    jobs, rows_meta = [], []
    for key, label, source, cp_path, eq_path in enumerated:
        ods = lane_t.compose(cached(eq_path), cached(cp_path))
        state = resolve_transport_state(ods, key, efit_quality=label, quality_source=source)
        base = {"shot": key.shot, "time_efit_s": key.time_efit_s, "efit_lineage": key.efit_lineage,
                "efit_quality": label, "state_status": state.status, "state_reasons": ";".join(state.reasons)}
        if not state.resolved:
            rows_meta.append({**base, "case": "all", "status": "not_ready"})
            continue
        readiness = assess_tglf_readiness(state, surfaces, config=config)
        age = ages.get((key.shot, key.time_ms))
        for surface in readiness.surfaces:
            if not surface.ready:
                rows_meta.append({**base, "r_over_a": surface.r_over_a, "case": "all",
                                  "status": "not_ready", "reason": surface.status})
                continue
            try:
                profiles, notes = variants(state, surface.r_over_a, age)
            except Exception as exc:  # noqa: BLE001 -- one surface's composition, recorded
                rows_meta.append({**base, "r_over_a": surface.r_over_a, "case": "all", "status": "error",
                                  "reason": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            for case in args.cases:
                profile = profiles[case]
                try:
                    local = surface.local_input if case == "c6_zeff2" else prepare_tglf_input(profile, surface.r_over_a)
                except Exception as exc:  # noqa: BLE001
                    rows_meta.append({**base, "r_over_a": surface.r_over_a, "case": case, "status": "error",
                                      "reason": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                identity = hashlib.sha256(json.dumps(
                    {"state": state.identity_payload() if hasattr(state, "identity_payload") else key.slug(),
                     "case": case, "surface": surface.r_over_a, "profile": _profile_sha(profile),
                     "parameters": parameters, "revision": revision}, sort_keys=True, default=str).encode()).hexdigest()
                job = {"workdir": str(out / case / key.slug() / f"r{surface.r_over_a:.2f}"),
                       "r_over_a": surface.r_over_a, "identity": identity, "profile": profile,
                       "local_input": local, "local": lane_t._local_summary(local)}
                jobs.append(job)
                rows_meta.append({**base, "r_over_a": surface.r_over_a, "case": case, "job": job,
                                  "notes": notes[case]})
    print(json.dumps({"states": len(enumerated), "jobs": len(jobs), "enumeration": counts}), flush=True)

    results: dict[str, dict] = {}
    if not args.dry_run and jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(lane_t.run_surface, job, config): job for job in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                job = futures[future]
                results[job["workdir"]] = record = future.result()
                print(f"[{done}/{len(jobs)}] {job['workdir']} {record['status']} {record.get('runtime_status')}",
                      flush=True)

    from vaft.code.gacode.tglf.outputs import TglfOutputs

    rows = []
    for meta in rows_meta:
        job = meta.pop("job", None)
        notes = meta.pop("notes", None)
        row = dict(meta)
        if job is not None:
            local = job["local"]
            row.update(zeff_local=local["zeff"], species="/".join(local["species"]),
                       main_ion_as=local["as"][1] if len(local["as"]) > 1 else None,
                       a_over_lne=local["a_over_ln"][0], a_over_lte=local["a_over_lt"][0],
                       betae=local["betae"], xnue=local["xnue"], notes=json.dumps(notes, default=float))
            record = results.get(job["workdir"])
            row["status"] = "dry_run" if record is None else record["status"]
            outputs = Path(job["workdir"]) / "outputs.json"
            if record is not None and record["status"] == "solved" and outputs.is_file():
                row.update(atlas.spectral_descriptors(TglfOutputs.read_json(outputs)))
        rows.append(row)

    columns = sorted({k for r in rows for k in r}, key=lambda c: (
        c not in ("shot", "time_efit_s", "efit_lineage", "efit_quality", "r_over_a", "case", "status"), c))
    with open(out / "tglf_composition.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    pairs: dict[tuple, dict] = {}
    measures = ("q_tot_gb", "qe_gb", "qi_gb", "gamma_e_gb", "gamma_max", "ky_at_gamma_max",
                "gamma_max_ion_scale", "zeff_local", "main_ion_as")
    for r in rows:
        if r.get("case") not in CASES or r.get("status") != "solved":
            continue
        k = (r["shot"], r["time_efit_s"], r["r_over_a"])
        entry = pairs.setdefault(k, {"shot": k[0], "time_efit_s": k[1], "r_over_a": k[2],
                                     "efit_quality": r["efit_quality"]})
        for m in measures:
            entry[f"{m}_{r['case']}"] = r.get(m)
    for entry in pairs.values():
        for m in ("q_tot_gb", "qe_gb", "qi_gb", "gamma_max"):
            ref = entry.get(f"{m}_c6_zeff2")
            for case in ("co_stripped", "co_openadas"):
                value = entry.get(f"{m}_{case}")
                if ref not in (None, 0.0) and value is not None:
                    entry[f"{m}_ratio_{case}"] = value / ref
    pair_rows = sorted(pairs.values(), key=lambda e: (e["shot"], e["time_efit_s"], e["r_over_a"]))
    if pair_rows:
        cols = list(dict.fromkeys(k for e in pair_rows for k in e))
        with open(out / "tglf_composition_pairs.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            writer.writerows(pair_rows)
    import vaft

    manifest = {"producer": "workflow/impurity_zeff/run_tglf_composition.py (Lane L, log #1569)",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "argv": sys.argv, "vaft": vaft.__file__, "vaft_git": lane_t._git_sha(ROOT),
                "gacode_revision": revision, "tglf_config": tglf_config, "tglf_parameters": parameters,
                "cases": args.cases, "states": len(enumerated), "jobs": len(jobs),
                "status_counts": {s: sum(1 for r in rows if r.get("status") == s) for s in {r.get("status") for r in rows}}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("states", "jobs", "status_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
