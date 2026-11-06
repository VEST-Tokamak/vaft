#!/usr/bin/env python3
"""Native TGLF (SAT0-3 x ES/EM) against the public TGLF-NN families on one input set (#2274, phase B).

Both backends see the same ``TGLFInput``. Every state is resolved through
``audit_tglf_nn.resolve_state``, the single definition the training-domain audit
uses: same population (the Tier A confinement table), same T_i scenarios, same
surfaces. So a difference between a network and the solver it stands in for is the
network's, not a difference in what it was asked.

Stages (``--stage``, default ``all``):

* ``native`` -- native TGLF for every configuration ``--sat-rules`` x
  ``--field-models`` on every ready surface, through ``run_tglf.run_surface``. Each
  run is cached by its run identity (state, physics parameters, surface, GACODE
  revision), so an interrupted grid resumes where it stopped. Writes ``native.csv``.
* ``nn`` -- every single-stage family on every ready surface: the domain audit
  (z-score, and containment when ``xbounds.txt`` was exported) and the ensemble
  prediction, *whether or not* the input is in domain, because the in-domain vs
  extrapolated contrast is the point. Writes ``nn.csv``.
* ``compare`` -- pairs each family with the native configuration it was trained on
  (same SAT rule; ``em`` families with ``em-bper``, ``es`` with ``es``;
  ``em-bper-bpar`` is never grouped under EM) and writes ``compare.csv`` (one row
  per surface x family) and ``compare_summary.csv`` (per scenario x subset x
  family x domain class). ``model_form.csv`` is the spread of the native fluxes
  across SAT rules at one field model -- the model-form reference an NN error is
  read against.

Channels (upstream ``src/tglf_nn.jl``): ``OUT_G_elec`` electron particle flux,
``OUT_Q_elec`` electron energy flux, ``OUT_Q_ions`` ion energy flux summed over
ions, ``OUT_P_ions`` ion toroidal stress (momentum, not particle), all gyro-Bohm.

Domain classes of a surface for one family: ``applicable`` (z within threshold and
inside the bounds), ``clip_only`` (outside only at upstream's 3.3-sigma clip),
``past_data_edge`` (some input beyond where the training data ended),
``out_of_domain`` (z-score only, when no bounds were exported).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

CHANNELS = ("OUT_G_elec", "OUT_Q_elec", "OUT_Q_ions", "OUT_P_ions")
#: Channels whose sign is physical and positive in a turbulent-loss state; their
#: log ratio is reported. The particle and momentum channels can change sign, so
#: they are compared by sign agreement and rank only.
LOG_CHANNELS = ("OUT_Q_elec", "OUT_Q_ions")
DEFAULT_SAT_RULES = (0, 1, 2, 3)
DEFAULT_FIELD_MODELS = ("es", "em-bper", "em-bper-bpar")


def _sibling(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _audit():
    return _sibling("transport_atlas_audit_tglf_nn", "audit_tglf_nn.py")


def _driver():
    return _sibling("transport_atlas_run_tglf", "run_tglf.py")


# --------------------------------------------------------------------------- channels


def native_channels(native: Any) -> dict[str, float]:
    """The four NN channels from a native ``TglfOutputs`` (electrons first)."""
    flux = native.gbflux or {}

    def species(quantity: str, ions: bool) -> float:
        values = np.asarray(flux.get(quantity, []), dtype=float)
        if values.size == 0:
            return math.nan
        return float(np.sum(values[1:])) if ions else float(values[0])

    return {"OUT_G_elec": species("particle", False), "OUT_Q_elec": species("energy", False),
            "OUT_Q_ions": species("energy", True), "OUT_P_ions": species("momentum", True)}


def trained_config(model: str) -> Optional[str]:
    """The native configuration a family was trained on, ``tglf-sat<n>-<es|em-bper>``.

    Read from the upstream name (the only machine-readable statement of it). A variant
    tag (``quench``, ``geo``) is not a native TGLF switch VAFT sets, so such a family is
    paired with its base rule and the tag is carried in ``nn_tags``.
    """
    from vaft.code.gacode.tglf.surrogate import parse_model_name

    parsed = parse_model_name(model)
    if parsed["sat_rule"] is None or parsed["electromagnetic"] is None:
        return None
    return f"tglf-sat{int(parsed['sat_rule'])}-{'em-bper' if parsed['electromagnetic'] else 'es'}"


def domain_class(audit: Any) -> str:
    """``applicable``, ``clip_only``, ``past_data_edge`` or ``out_of_domain``."""
    if not getattr(audit, "bounds_available", False):
        return "applicable" if audit.in_domain else "out_of_domain"
    if getattr(audit, "bound_violations_at_data_edge", ()):
        return "past_data_edge"
    if audit.applicable:
        return "applicable"
    return "clip_only"


# --------------------------------------------------------------------------- resolution


def resolved_population(args) -> tuple[list[tuple[dict, Any]], list[dict], list[dict]]:
    """Every resolved (ident, state) of the requested population, scenarios and subset."""
    from vaft.data.public import load_vest_tier_a_confinement

    audit, driver = _audit(), _driver()
    table = load_vest_tier_a_confinement(selection=None, **({"path": args.confinement} if args.confinement else {}))
    records = [r for r in audit.population(table) if args.subset in r["subsets"]]
    if args.shots:
        records = [r for r in records if r["key"].shot in set(args.shots)]
    if args.max_states:
        records = records[: args.max_states]
    plan = audit.scenarios(args.profile_ti_dir, args.ratios)
    loaded: dict[Path, Any] = {}

    def load(path: Path) -> Any:
        if path not in loaded:
            loaded[path] = driver._load(path)
        return loaded[path]

    resolved, rows = [], []
    for scenario in plan:
        for record in records:
            ident, row, state = audit.resolve_state(driver, args.filedb, record, scenario,
                                                    rho_max=args.rho_max, load=load)
            rows.append(row)
            if state is not None and state.resolved:
                resolved.append((ident, state))
    return resolved, rows, plan


def ready_surfaces(state: Any, surfaces: Iterable[float], config: Any = None) -> list[Any]:
    from vaft.process.transport_state import assess_tglf_readiness

    return [s for s in assess_tglf_readiness(state, tuple(surfaces), config=config).surfaces if s.ready]


# --------------------------------------------------------------------------- native


def native_stage(args, resolved) -> list[dict[str, Any]]:
    from vaft.code.gacode.tglf.outputs import TglfOutputs
    from vaft.process.transport_state import physics_parameters, run_identity

    driver = _driver()
    jobs = []
    for sat_rule in args.sat_rules:
        for field_model in args.field_models:
            ns = argparse.Namespace(sat_rule=sat_rule, field_model=field_model, backend=args.backend,
                                    partition=args.partition, account=args.account, max_wait=args.max_wait,
                                    timeout=args.timeout, mem_mb=args.mem_mb)
            config = driver.build_config(ns)
            label = driver.config_label(sat_rule, field_model)
            revision = driver.gacode_revision(config)
            parameters = physics_parameters(config)
            for ident, state in resolved:
                for surface in ready_surfaces(state, args.surfaces, config):
                    identity = run_identity(state, solver="tglf", parameters=parameters,
                                            surface=surface.r_over_a, solver_revision=revision)
                    workdir = (args.out / "native" / label / ident["scenario"] / state.key.slug()
                               / f"r{surface.r_over_a:.2f}")
                    jobs.append(({**ident, "r_over_a": surface.r_over_a, "config": label,
                                  "sat_rule": sat_rule, "field_model": field_model,
                                  "gacode_revision": revision},
                                 {"workdir": str(workdir), "r_over_a": surface.r_over_a, "identity": identity,
                                  "profile": state.profile, "local_input": surface.local_input,
                                  "local": driver._local_summary(surface.local_input)},
                                 config))
    print(f"native: {len(jobs)} surface runs", flush=True)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(driver.run_surface, job, config): (meta, job) for meta, job, config in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            meta, job = futures[future]
            record = future.result()
            row = {**meta, "status": record["status"], "runtime_status": record.get("runtime_status"),
                   "cached": record.get("cached"), "elapsed_s": record.get("elapsed_s"),
                   "run_identity": job["identity"]}
            outputs = Path(job["workdir"]) / "outputs.json"
            if record["status"] == "solved" and outputs.is_file():
                row.update(native_channels(TglfOutputs.read_json(outputs)))
            rows.append(row)
            if done % 200 == 0 or done == len(jobs):
                print(f"native: {done}/{len(jobs)}", flush=True)
    return rows


# --------------------------------------------------------------------------- NN


def nn_stage(args, resolved) -> list[dict[str, Any]]:
    from vaft.code.gacode.tglf.surrogate import (
        ModelContractError,
        audit_features,
        build_features,
        ensemble_predict,
    )

    audit = _audit()
    models, skipped = audit.families(args.model_dir, args.models)
    if not models:
        raise SystemExit("no single-stage TGLF-NN family is resolvable")
    rows = []
    for ident, state in resolved:
        for surface in ready_surfaces(state, args.surfaces):
            for identity, metadata in models:
                try:
                    features, assumed = build_features(surface.local_input, metadata)
                except ModelContractError as error:
                    rows.append({**ident, "r_over_a": surface.r_over_a, "model": identity.name,
                                 "domain_class": "refused", "detail": str(error)[:240]})
                    continue
                verdict = audit_features(features, assumed, identity, metadata)
                mean, spread = ensemble_predict(features, identity, metadata)
                row = {**ident, "r_over_a": surface.r_over_a, "model": identity.name,
                       "nn_sat_rule": identity.sat_rule, "nn_electromagnetic": identity.electromagnetic,
                       "nn_tags": " ".join(identity.tags), "domain_class": domain_class(verdict),
                       "in_domain": verdict.in_domain, "contained": verdict.contained,
                       "max_abs_z": verdict.max_abs_z,
                       "bound_violations": " ".join(verdict.bound_violations),
                       "bound_violations_at_data_edge": " ".join(
                           getattr(verdict, "bound_violations_at_data_edge", ()))}
                for channel in CHANNELS:
                    row[channel] = mean.get(channel, math.nan)
                    row[f"{channel}_spread"] = spread.get(channel, math.nan)
                rows.append(row)
    print(f"nn: {len(rows)} predictions, {len(models)} families, skipped {[s['model'] for s in skipped]}",
          flush=True)
    return rows


# --------------------------------------------------------------------------- compare


KEY = ("scenario", "shot", "time_efit_s", "efit_lineage", "r_over_a")


def _key(row: Mapping[str, Any]) -> tuple:
    return tuple(round(float(row[k]), 6) if k in ("time_efit_s", "r_over_a") else
                 (int(row[k]) if k == "shot" else str(row[k])) for k in KEY)


def compare(native_rows: list[dict], nn_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Pair NN predictions with their trained-on native configuration; summarise."""
    from scipy.stats import spearmanr

    solved = {}
    for row in native_rows:
        if row.get("status") == "solved":
            solved[(_key(row), row["config"])] = row
    paired = []
    for row in nn_rows:
        if row.get("domain_class") == "refused":
            continue
        config = trained_config(row["model"])
        native = None if config is None else solved.get((_key(row), config))
        if native is None:
            continue
        out = {k: row[k] for k in (*KEY, "subsets", "model", "domain_class", "max_abs_z")}
        out["config"] = config
        for channel in CHANNELS:
            n, m, s = float(native[channel]), float(row[channel]), float(row[f"{channel}_spread"])
            out[f"{channel}_native"], out[f"{channel}_nn"], out[f"{channel}_spread"] = n, m, s
            both = np.isfinite(n) and np.isfinite(m)
            out[f"{channel}_sign_agree"] = bool(np.sign(n) == np.sign(m)) if both and n != 0 and m != 0 else None
            out[f"{channel}_within_2spread"] = bool(abs(m - n) <= 2.0 * s) if both and np.isfinite(s) else None
            if channel in LOG_CHANNELS:
                out[f"{channel}_log10_ratio"] = (math.log10(m / n) if both and n > 0 and m > 0 else None)
        paired.append(out)

    groups: dict[tuple, list[dict]] = {}
    for row in paired:
        for subset in str(row["subsets"]).split():
            for cls in (row["domain_class"], "all"):
                groups.setdefault((row["scenario"], subset, row["model"], row["config"], cls), []).append(row)
    summary = []
    for (scenario, subset, model, config, cls), members in sorted(groups.items()):
        entry = {"scenario": scenario, "subset": subset, "model": model, "config": config,
                 "domain_class": cls, "surfaces": len(members)}
        for channel in CHANNELS:
            native = np.array([r[f"{channel}_native"] for r in members], dtype=float)
            nn = np.array([r[f"{channel}_nn"] for r in members], dtype=float)
            finite = np.isfinite(native) & np.isfinite(nn)
            native, nn = native[finite], nn[finite]
            signs = [r[f"{channel}_sign_agree"] for r in members if r[f"{channel}_sign_agree"] is not None]
            within = [r[f"{channel}_within_2spread"] for r in members if r[f"{channel}_within_2spread"] is not None]
            entry[f"{channel}_sign_agree"] = float(np.mean(signs)) if signs else None
            entry[f"{channel}_within_2spread"] = float(np.mean(within)) if within else None
            entry[f"{channel}_spearman"] = (float(spearmanr(native, nn).statistic)
                                            if native.size >= 5 and np.ptp(native) > 0 and np.ptp(nn) > 0
                                            else None)
            if channel in LOG_CHANNELS:
                logs = np.array([r[f"{channel}_log10_ratio"] for r in members
                                 if r[f"{channel}_log10_ratio"] is not None], dtype=float)
                entry[f"{channel}_median_log10_ratio"] = float(np.median(logs)) if logs.size else None
                entry[f"{channel}_median_abs_log10_ratio"] = float(np.median(np.abs(logs))) if logs.size else None
        summary.append(entry)

    model_form = []
    by_surface: dict[tuple, dict[str, dict]] = {}
    for (key, config), row in solved.items():
        by_surface.setdefault((key, row["field_model"]), {})[config] = row
    for (key, field_model), configs in sorted(by_surface.items()):
        entry = dict(zip(KEY, key))
        entry.update(field_model=field_model, configs=len(configs))
        for channel in LOG_CHANNELS:
            values = np.array([c[channel] for c in configs.values()], dtype=float)
            positive = values[np.isfinite(values) & (values > 0)]
            entry[f"{channel}_log10_max_over_min"] = (
                float(np.log10(positive.max() / positive.min())) if positive.size >= 2 else None)
        model_form.append(entry)
    return paired, summary, model_form


# --------------------------------------------------------------------------- driver


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--confinement", type=Path)
    parser.add_argument("--profile-ti-dir", type=Path)
    parser.add_argument("--ratios", type=float, nargs="*", default=[1.0])
    parser.add_argument("--subset", default="primary", choices=("primary", "thomson_consistent", "tier_a"))
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--surfaces", type=float, nargs="*")
    parser.add_argument("--rho-max", type=float)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--sat-rules", type=int, nargs="*", default=list(DEFAULT_SAT_RULES))
    parser.add_argument("--field-models", nargs="*", default=list(DEFAULT_FIELD_MODELS),
                        choices=DEFAULT_FIELD_MODELS)
    parser.add_argument("--stage", default="all", choices=("native", "nn", "compare", "all"))
    parser.add_argument("--backend", default="local", choices=("local", "slurm"))
    parser.add_argument("--partition", default="lowpri-short")
    parser.add_argument("--account")
    parser.add_argument("--max-wait", type=float, default=6 * 3600.0)
    parser.add_argument("--mem-mb", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from vaft.process.transport_state import DEFAULT_RHO_MAX, DEFAULT_SURFACES

    args.surfaces = list(DEFAULT_SURFACES) if args.surfaces is None else args.surfaces
    args.rho_max = DEFAULT_RHO_MAX if args.rho_max is None else args.rho_max
    args.out.mkdir(parents=True, exist_ok=True)
    audit = _audit()

    if args.stage in ("native", "nn", "all"):
        resolved, state_rows, plan = resolved_population(args)
        audit._write_csv(args.out / "states.csv", state_rows)
        print(f"resolved {len(resolved)} state x scenario pairs", flush=True)
        if args.stage in ("native", "all"):
            audit._write_csv(args.out / "native.csv", native_stage(args, resolved))
        if args.stage in ("nn", "all"):
            audit._write_csv(args.out / "nn.csv", nn_stage(args, resolved))
    if args.stage in ("compare", "all"):
        paired, summary, model_form = compare(_read_csv(args.out / "native.csv"), _read_csv(args.out / "nn.csv"))
        audit._write_csv(args.out / "compare.csv", paired)
        audit._write_csv(args.out / "compare_summary.csv", summary)
        audit._write_csv(args.out / "model_form.csv", model_form)
        print(f"compare: {len(paired)} paired surfaces", flush=True)

    manifest_path = args.out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"runs": []}
    manifest["runs"].append({
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv if argv is None else ["compare_tglf_nn.py", *argv],
        "code_revision": _driver()._git_sha(Path(__file__).resolve().parent),
        "stage": args.stage,
    })
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
