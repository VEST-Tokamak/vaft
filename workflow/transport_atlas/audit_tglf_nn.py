#!/usr/bin/env python3
"""TGLF-NN training-domain audit over the validated confinement population.

#834 (#553 increment 2) asked whether any public TGLF-NN family applies to VEST, and
answered on one slice (48224 at 300 ms, r/a = 0.5, CES-measured T_i). This driver asks
the same question of the population the confinement scaling is built on: every state
of the Tier A confinement table (``vaft/data/confinement/vest_tier_a_confinement.csv``),
reported for three subsets:

* ``primary``: the confinement selection (``selected``: I_p >= 30 kA,
  dIp/tau_E <= 0.20, |dW/dt|/P_OH <= 1);
* ``thomson_consistent``: those of them whose Thomson pressure agrees with EFIT;
* ``tier_a``: every good/admissible state in the table.

The answer depends on T_i, because TAUS = T_i/T_e is a model input and VEST has no
routine T_i. So every state is audited under several T_i *scenarios*:

* ``pressure_partition_profile`` (the reference): T_i = (T_i/T_e)(rho) * T_e, the
  BIC-chosen polynomial fitted to the magnetics-EFIT pressure partition. Its products
  come from the ``ti_inferred`` stage run with ``--ti-form profile`` and are read from
  ``--profile-ti-dir/<shot>.json.gz``. Only their ion temperature is taken (as the
  resolver's ``inferred_ti``); n_e, T_e and the composition closure are the FileDB
  product's, as in every other scenario, so the scenarios differ in T_i alone;
* ``ratio_<r>`` for each ``--ratios`` value, e.g. 1.0 (the pipeline's T_i = T_e
  assumption) and 0.17 (the superseded pressure-matching value that #834's sample
  reflects).

A measured T_i, when a slice has one, is used under every scenario; the row's
``ti_lineage`` says which T_i was used.

The audit is a z-score against the model's training moments
(:func:`vaft.code.gacode.tglf.surrogate.audit_training_domain`). It needs no
onnxruntime and runs no network. Two-stage ``*_gknn*`` families are listed as skipped.

Output (``--out``)::

    audit.csv       one row per state x surface x scenario x family
    states.csv      one row per state x scenario: resolution status and T_i lineage
    summary.csv     in-domain fraction per subset x scenario x family
    MANIFEST.json   inputs, models (sha256), settings, code revision, counts
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from vaft.process.transport_state import (
    DEFAULT_RHO_MAX,
    DEFAULT_SURFACES,
    DEFAULT_TIME_TOLERANCE_S,
    TransportStateKey,
    assess_tglf_readiness,
    resolve_transport_state,
)

REFERENCE_SCENARIO = "pressure_partition_profile"
SUBSETS = ("primary", "thomson_consistent", "tier_a")
DEFAULT_RATIOS = (1.0, 0.17)


def _tglf_driver():
    """The sibling driver, for its product lookup and loading (one definition)."""
    name = "transport_atlas_run_tglf"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("run_tglf.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- population


def population(table: Any, lineages: Iterable[str] = ("magnetics",)) -> list[dict[str, Any]]:
    """The confinement table's states as ``{key, efit_quality, subsets}`` records.

    Only good/admissible states are kept: the table carries nothing else, and the
    transport-state resolver refuses anything else.
    """
    rows = []
    for _, row in table.iterrows():
        lineage = str(row["efit_lineage"])
        quality = str(row["efit_quality"])
        if lineage not in lineages or quality not in ("good", "admissible"):
            continue
        selected = bool(row["selected"]) if row["selected"] == row["selected"] else False
        consistent = row.get("thomson_consistent")
        consistent = bool(consistent) if consistent == consistent and consistent is not None else False
        subsets = ["tier_a"]
        if selected:
            subsets.append("primary")
            if consistent:
                subsets.append("thomson_consistent")
        rows.append({"key": TransportStateKey(int(row["shot"]), float(row["time_efit_s"]), lineage),
                     "efit_quality": quality, "subsets": tuple(subsets)})
    return rows


def scenarios(profile_ti_dir: Optional[Path], ratios: Iterable[float]) -> list[dict[str, Any]]:
    """The T_i scenarios to audit, the reference first when its products are given."""
    out = []
    if profile_ti_dir is not None:
        # "policy" only fills points where the stored profile is not positive (an edge
        # where T_e itself is missing); the resolver records the fill and no surface
        # whose input depends on it is ready (readiness ``ti_not_inferred_here``).
        out.append({"name": REFERENCE_SCENARIO, "profile_ti_dir": Path(profile_ti_dir),
                    "ti_te_ratio": "policy"})
    for ratio in ratios:
        ratio = float(ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(f"a T_i/T_e scenario must be positive; got {ratio!r}")
        out.append({"name": f"ratio_{ratio:g}", "profile_ti_dir": None, "ti_te_ratio": ratio})
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate T_i scenarios: {names}")
    return out


def families(model_dir: Optional[Path] = None, names: Optional[Iterable[str]] = None):
    """Resolvable single-stage families as ``(identity, metadata)``, and the skipped ones."""
    from vaft.code.gacode.tglf.surrogate import (
        ModelContractError,
        available_models,
        load_metadata,
        resolve_model,
    )

    # An explicit directory is the whole search: no environment, no Julia depot, so the
    # audited set is exactly what the caller pointed at.
    env = {} if model_dir is not None else None
    found = available_models(model_dir=model_dir, env=env)
    wanted = sorted(found) if names is None else list(names)
    usable, skipped = [], []
    for name in wanted:
        identity = resolve_model(found.get(name, name), model_dir=model_dir, env=env)
        try:
            metadata = load_metadata(identity)
        except ModelContractError as error:
            # the *_gknn* two-stage families take another model's fluxes as inputs
            skipped.append({"model": identity.name, "reason": "model_contract", "detail": str(error)[:240]})
            continue
        if metadata.requires_base_model_outputs:
            skipped.append({"model": identity.name, "reason": "two_stage_gknn",
                            "inputs": list(metadata.requires_base_model_outputs)})
            continue
        usable.append((identity, metadata))
    return usable, skipped


def inferred_ti_for(profiles: Any, time_s: float, tolerance: float) -> tuple[Optional[dict], str]:
    """The stored inferred ion temperature of the slice nearest ``time_s``, as ``inferred_ti``.

    Returns ``(mapping, "")`` or ``(None, reason)``. Only a temperature whose record
    classifies as inferred is taken; a measured or assumed one is not this scenario's.
    """
    from vaft.machine_mapping.core_profiles import classify_ti_record, ti_record_fields

    try:
        times = np.atleast_1d(np.asarray(profiles["core_profiles.time"], dtype=float))
    except Exception:  # noqa: BLE001 - a product with no time has no slice
        return None, "no_profile_ti_slices"
    if times.size == 0:
        return None, "no_profile_ti_slices"
    index = int(np.argmin(np.abs(times - time_s)))
    if abs(float(times[index]) - time_s) > tolerance:
        return None, "no_profile_ti_slice_within_tolerance"
    base = f"core_profiles.profiles_1d.{index}.ion.0"
    try:
        record = profiles[f"{base}.temperature_fit.parameters"]
        temperature = np.asarray(profiles[f"{base}.temperature"], dtype=float)
    except Exception:  # noqa: BLE001 - absent leaves read as "not inferred here"
        return None, "profile_ti_missing"
    kind = classify_ti_record(record)
    if kind != "inferred":
        return None, f"profile_ti_{kind}"
    fields = ti_record_fields(record)
    if "equilibrium_time_s" in fields:
        # the partition must come from this state's own EFIT slice
        try:
            eq_time = float(fields["equilibrium_time_s"])
        except ValueError:
            return None, "profile_ti_record_unreadable"
        if abs(eq_time - time_s) > tolerance:
            return None, "profile_ti_from_another_equilibrium"
    return {"temperature": temperature, "time": float(times[index]), "source": str(record),
            "method": fields.get("method", "equilibrium_pressure_partition")}, ""


# --------------------------------------------------------------------------- audit


def _rho_valid(source: Any) -> Optional[tuple[float, float]]:
    """The fitted span ``rho_valid=a:b`` of an inferred-T_i record, if it names one."""
    for part in str(source or "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == "rho_valid" and ":" in value:
            low, high = value.split(":", 1)
            try:
                return float(low), float(high)
            except ValueError:
                return None
    return None


def _surface_rho(profile: Any, r_over_a: float) -> float:
    """``rho_tor_norm`` of the surface at ``r/a`` on the state's profile (NaN if unknown)."""
    rmin = getattr(profile, "rmin", None)
    rho = getattr(profile, "rho", None)
    if rmin is None or rho is None or np.size(rmin) < 2:
        return float("nan")
    rmin, rho = np.asarray(rmin, float), np.asarray(rho, float)
    return float(np.interp(r_over_a * rmin[-1], rmin, rho))


def audit_state(state: Any, surfaces: Iterable[float], models, *, threshold: float) -> list[dict[str, Any]]:
    """One row per ready surface x family for a resolved state; a row per unready surface."""
    from vaft.code.gacode.tglf.surrogate import ModelContractError, audit_training_domain

    readiness = assess_tglf_readiness(state, tuple(surfaces))
    span = _rho_valid(state.ti.get("source"))
    rows = []
    for surface in readiness.surfaces:
        base = {"r_over_a": float(surface.r_over_a), "readiness": surface.status}
        if not surface.ready:
            rows.append({**base, "model": None, "audit_status": "not_ready", "detail": surface.detail})
            continue
        local = surface.local_input
        rho = _surface_rho(state.profile, surface.r_over_a)
        base.update({
            "rho_tor_norm": rho,
            "ti_extrapolated": (None if span is None or not np.isfinite(rho)
                                else bool(rho < span[0] or rho > span[1])),
            "taus_main": float(local.taus[1]) if np.size(local.taus) > 1 else float("nan"),
            "betae": float(local.betae), "xnue": float(local.xnue), "zeff": float(local.zeff),
            "q": float(local.q_loc),
        })
        for identity, metadata in models:
            try:
                audit = audit_training_domain(local, identity, metadata, threshold=threshold)
            except ModelContractError as error:
                rows.append({**base, "model": identity.name, "audit_status": "refused",
                             "detail": str(error)[:240]})
                continue
            worst = max(audit.z_scores, key=lambda name: abs(audit.z_scores[name]))
            taus = {name: z for name, z in audit.z_scores.items() if name.startswith("TAUS")}
            rows.append({
                **base, "model": identity.name, "audit_status": "audited",
                "in_domain": audit.in_domain, "max_abs_z": audit.max_abs_z,
                "worst_feature": worst, "worst_z": audit.z_scores[worst],
                "n_violations": len(audit.violations),
                "violations": " ".join(audit.violations),
                "taus_z": " ".join(f"{k}={v:+.2f}" for k, v in sorted(taus.items())),
                "taus_violates": any(name.startswith("TAUS") for name in audit.violations),
                # in domain once the T_i channel is set aside: what binds without TAUS
                "violations_without_taus": " ".join(v for v in audit.violations if not v.startswith("TAUS")),
                "assumed": " ".join(audit.assumed),
            })
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In-domain counts per subset x scenario x family over the audited surfaces."""
    out = []
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("audit_status") != "audited":
            continue
        for subset in row["subsets"].split():
            groups.setdefault((subset, row["scenario"], row["model"]), []).append(row)
    for (subset, scenario, model), members in sorted(groups.items()):
        states = {(r["shot"], r["time_efit_s"]) for r in members}
        in_states = {(r["shot"], r["time_efit_s"]) for r in members if r["in_domain"]}
        binding: dict[str, int] = {}
        for r in members:
            if not r["in_domain"]:
                binding[r["worst_feature"]] = binding.get(r["worst_feature"], 0) + 1
        clean = [r for r in members if not r["violations_without_taus"]]
        # the reference T_i outside the partition's fitted span is the polynomial's
        # extrapolation; count the surfaces that stand on fitted T_i separately
        fitted = [r for r in members if r.get("ti_extrapolated") is not True]
        out.append({
            "subset": subset, "scenario": scenario, "model": model,
            "states": len(states), "surfaces": len(members),
            "in_domain_surfaces": sum(bool(r["in_domain"]) for r in members),
            "states_with_any_in_domain_surface": len(in_states),
            "in_domain_if_taus_ignored": len(clean),
            "surfaces_ti_fitted": len(fitted),
            "in_domain_surfaces_ti_fitted": sum(bool(r["in_domain"]) for r in fitted),
            "taus_violating_surfaces": sum(bool(r["taus_violates"]) for r in members),
            "median_max_abs_z": float(np.median([r["max_abs_z"] for r in members])),
            "median_taus_main": float(np.nanmedian([r["taus_main"] for r in members])),
            "binding_features": " ".join(f"{k}:{v}" for k, v in
                                         sorted(binding.items(), key=lambda kv: -kv[1])[:4]),
        })
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


# --------------------------------------------------------------------------- driver


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filedb", type=Path, required=True,
                        help="campaign FileDB root (omas/core_profiles, omas/efit/magnetic)")
    parser.add_argument("--confinement", type=Path,
                        help="Tier A confinement table; default: the packaged vaft/data/confinement sample")
    parser.add_argument("--profile-ti-dir", type=Path,
                        help="per-shot <shot>.json.gz products of the ti_inferred stage run with "
                             "--ti-form profile (the reference scenario); omitted: ratio scenarios only")
    parser.add_argument("--ratios", type=float, nargs="*", default=list(DEFAULT_RATIOS),
                        help="fixed T_i/T_e sensitivity scenarios")
    parser.add_argument("--model-dir", type=Path, help="TurbulentTransport.jl models directory")
    parser.add_argument("--models", nargs="*", help="restrict to these families")
    parser.add_argument("--surfaces", type=float, nargs="*", default=list(DEFAULT_SURFACES))
    parser.add_argument("--threshold", type=float, default=None, help="domain distance [sigma]")
    parser.add_argument("--rho-max", type=float, default=DEFAULT_RHO_MAX)
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import vaft
    from vaft.code.gacode.tglf.surrogate import DEFAULT_DOMAIN_THRESHOLD
    from vaft.data.public import load_vest_tier_a_confinement

    threshold = DEFAULT_DOMAIN_THRESHOLD if args.threshold is None else float(args.threshold)
    driver = _tglf_driver()
    table = load_vest_tier_a_confinement(selection=None, **({"path": args.confinement} if args.confinement else {}))
    states = population(table)
    if args.shots:
        states = [s for s in states if s["key"].shot in set(args.shots)]
    models, skipped_models = families(args.model_dir, args.models)
    if not models:
        parser.error("no single-stage TGLF-NN family is resolvable; pass --model-dir or set $TURBULENTTRANSPORTHOME")
    plan = scenarios(args.profile_ti_dir, args.ratios)
    args.out.mkdir(parents=True, exist_ok=True)

    loaded: dict[Path, Any] = {}

    def cached(path: Path) -> Any:
        if path not in loaded:
            loaded[path] = driver._load(path)
        return loaded[path]

    audit_rows, state_rows = [], []
    for scenario in plan:
        for record in states:
            key = record["key"]
            ident = {"shot": key.shot, "time_efit_s": key.time_efit_s, "efit_lineage": key.efit_lineage,
                     "efit_quality": record["efit_quality"], "subsets": " ".join(record["subsets"]),
                     "scenario": scenario["name"]}
            cp_path, cp_manifest = driver.core_profiles_product(args.filedb, key.shot)
            eq_path, eq_manifest = driver.product(args.filedb, key.efit_lineage, key.shot)
            missing = [name for name, path, manifest in (("core_profiles", cp_path, cp_manifest),
                                                         ("equilibrium", eq_path, eq_manifest))
                       if path is None or manifest.get("status") not in (None, "success")]
            if missing:
                state_rows.append({**ident, "status": "missing_product", "reasons": " ".join(missing)})
                continue
            inferred_ti = None
            if scenario["profile_ti_dir"] is not None:
                ti_path = Path(scenario["profile_ti_dir"]) / f"{key.shot}.json.gz"
                if not ti_path.is_file():
                    state_rows.append({**ident, "status": "missing_product", "reasons": "profile_ti"})
                    continue
                inferred_ti, why = inferred_ti_for(cached(ti_path), key.time_efit_s, DEFAULT_TIME_TOLERANCE_S)
                if inferred_ti is None:
                    state_rows.append({**ident, "status": "no_inferred_ti", "reasons": why})
                    continue
            ods = driver.compose(cached(eq_path), cached(cp_path))
            state = resolve_transport_state(
                ods, key, efit_quality=record["efit_quality"], quality_source="confinement_table",
                ti_te_ratio=scenario["ti_te_ratio"], inferred_ti=inferred_ti, rho_max=args.rho_max,
                inputs={"core_profiles": {"path": str(cp_path)}, "equilibrium": {"path": str(eq_path)}},
            )
            state_rows.append({**ident, "status": state.status, "reasons": " ".join(state.reasons),
                               "ti_lineage": state.ti.get("lineage"), "ti_method": state.ti.get("method"),
                               "ti_te_ratio": state.ti.get("ratio"), "ti_source": state.ti.get("source"),
                               "ti_fill_points": (state.ti.get("fill") or {}).get("points", 0),
                               "z_eff": state.settings.get("z_eff"), "impurity": state.settings.get("impurity")})
            if not state.resolved:
                continue
            for row in audit_state(state, args.surfaces, models, threshold=threshold):
                audit_rows.append({**ident, "ti_lineage": state.ti.get("lineage"), **row})
        print(f"[{scenario['name']}] {sum(r['scenario'] == scenario['name'] and r['status'] == 'resolved' for r in state_rows)}"
              f"/{len(states)} states resolved", flush=True)

    summary = summarise(audit_rows)
    _write_csv(args.out / "audit.csv", audit_rows)
    _write_csv(args.out / "states.csv", state_rows)
    _write_csv(args.out / "summary.csv", summary)
    manifest = {
        "schema_version": 1,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "vaft_version": getattr(vaft, "__version__", None),
        "code_revision": driver._git_sha(Path(__file__).resolve().parent),
        "inputs": {"filedb": str(args.filedb),
                   "confinement": str(args.confinement) if args.confinement else "packaged sample",
                   "profile_ti_dir": None if args.profile_ti_dir is None else str(args.profile_ti_dir)},
        "settings": {"surfaces": list(args.surfaces), "threshold_sigma": threshold,
                     "rho_max": args.rho_max, "scenarios": [s["name"] for s in plan],
                     "reference_scenario": REFERENCE_SCENARIO if args.profile_ti_dir else None},
        "models": [{"name": i.name, "directory": i.directory, "upstream_version": i.upstream_version,
                    "sha256": dict(i.sha256)} for i, _ in models],
        "skipped_models": skipped_models,
        "counts": {"states": len(states),
                   "by_subset": {s: sum(s in r["subsets"] for r in states) for s in SUBSETS},
                   "state_status": _counts(state_rows, "status"),
                   "audit_rows": len(audit_rows)},
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


def _counts(rows: list[Mapping[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[f"{row['scenario']}:{row[column]}"] = out.get(f"{row['scenario']}:{row[column]}", 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
