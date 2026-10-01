"""Build the #1430 matched TS-EFIT state tables for the #1331 Tier A campaign.

Reads, never writes, the campaign FileDB and the slice grades
``analyze_tierA.py`` produced; writes ``state.csv``, ``profiles.csv``, their
JSON Schemas and a manifest into one versioned atlas directory (contract #1454).

Rows
----
* one ``magnetics`` row per (shot, time) graded ``good`` or ``admissible`` by
  ``criteria.slice_labels`` re-run on the analysis records with this checkout;
* one ``electron_kinetic`` row per successful ``electron_efit`` whose paired
  magnetics slice (same shot, same time) is ``good`` or ``admissible``.  Its
  ``efit_quality`` is that pair's label; ``kinetic_admissible`` is the kinetic
  fit's own admissibility veto.  The Thomson criterion is never applied to it.

Unreconstructible slices never become rows.  A magnetics row whose time has an
electron-EFIT attempt records that attempt's outcome in ``paired_kin_status``,
so a kinetic failure stays distinguishable from a physical inconsistency.

Usage (vestserver, with the lane's sitecustomize shim on PYTHONPATH)::

    python3 workflow/kinetic_state/build_state.py \\
        --filedb ~/runs/campaign/filedb \\
        --analysis ~/runs/campaign/tierA_analysis.json \\
        --out ~/runs/campaign/atlas/v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]

#: How far the EFIT slice may sit from a graded time.  Graded times are
#: ``round(t * 1e3)`` ms, so up to half a millisecond plus float noise.
SLICE_TOLERANCE_S = 0.6e-3
CRITERIA_PATH = REPOSITORY / "workflow" / "efit_uncertainty_calibration" / "criteria.py"

#: Column name -> (JSON type, description).  The schema files are generated from
#: this, so a column cannot exist in the CSV without being described.
STATE_COLUMNS: dict[str, tuple[str, str]] = {
    "contract_version": ("string", "state key contract version (#1454)"),
    "shot": ("integer", "VEST shot number"),
    "time_efit_s": ("number", "EFIT slice time, rounded to 1e-4 s"),
    "efit_lineage": ("string", "magnetics | electron_kinetic"),
    "efit_quality": ("string", "criteria.py label of the magnetics slice at this time: good | admissible"),
    "efit_setting": ("string", "EFIT preset name"),
    "efit_product": ("string", "FileDB product the row was computed from"),
    "efit_product_sha256": ("string", "sha256 of that product file"),
    "ti_lineage": ("string", "none | ti_eq_te_assumed | pressure_partition_inferred"),
    "ti_te_ratio_assumed": ("number", "Ti/Te the reconstruction assumed (electron_kinetic only)"),
    "efit_status": ("string", "valid | unavailable: whether this row's reconstruction could be read at this time"),
    "efit_status_reason": ("string", "why not valid"),
    "kinetic_admissible": ("string", "pass | fail: criteria admissibility veto on the kinetic fit itself"),
    "kinetic_admissible_reasons": ("string", "veto reasons, ';'-joined"),
    "kinetic_chi2": ("number", "electron-EFIT total chi-square"),
    "paired_kin_status": ("string", "magnetics rows: valid | failed | unavailable | not_attempted for the electron EFIT at this time"),
    "paired_kin_reason": ("string", "electron-EFIT failure reason"),
    "ts_status": ("string", "matched | unmatched | invalid"),
    "ts_reason": ("string", "why not matched"),
    "time_ts_s": ("number", "Thomson sample time used"),
    "dt_ts_efit_s": ("number", "t_TS - t_EFIT"),
    "ts_tolerance_s": ("number", "allowed |dt|: max(0.5 median magnetics-EFIT cadence, 1 ms)"),
    "ts_points": ("integer", "Thomson channels inside the LCFS"),
    "rho_coordinate": ("string", "rho_tor_norm | unavailable"),
    "r_sum": ("number", "sum p_EFIT / sum p_e over the channels"),
    "log_ratio": ("number", "ln(sum p_e / sum p_EFIT), the criteria quantity"),
    "criteria_log_ratio": ("number", "the same quantity as recorded in the analysis JSON (cross-check)"),
    "psi_norm_lo": ("number", "lowest channel psi_N"),
    "psi_norm_hi": ("number", "highest channel psi_N"),
    "cp_time_s": ("number", "core_profiles slice time used for p_e(rho)"),
    "r_w": ("number", "int p_EFIT dV / int p_e dV over [psi_norm_lo, psi_norm_hi], p_e from core_profiles"),
    "r_w_full": ("number", "same over the whole plasma; extrapolates the core_profiles fit"),
    "r_w_cell_weights": ("string", "outline | psi_threshold"),
    "r_w_reason": ("string", "why r_w is empty"),
    "r_w_on_pair_span": ("number", "electron_kinetic rows: r_w over the paired magnetics row's psi_N span"),
    "delta_r_w": ("number", "electron_kinetic rows: r_w_on_pair_span - r_w of the paired magnetics row"),
    "c_p": ("number", "electron_kinetic rows: ln(r_w_on_pair_span / r_w of the paired magnetics row)"),
    "ip_measured_a": ("number", "measured Ip [A]"),
    "ip_reconstructed_a": ("number", "reconstructed Ip [A]"),
    "betap": ("number", "poloidal beta"),
    "li": ("number", "internal inductance (magnetics rows)"),
    "q95": ("number", "q at 95 % flux"),
    "wmhd_j": ("number", "stored energy [J]"),
    "probe_reduced_chi2": ("number", "magnetics rows: probe reduced chi-square"),
    "loop_reduced_chi2": ("number", "magnetics rows: flux-loop reduced chi-square"),
    "virial_status": ("string", "magnetics rows: criteria virial verdict"),
    "gs_status": ("string", "magnetics rows: criteria Grad-Shafranov verdict"),
    "thomson_criterion_status": ("string", "magnetics rows: criteria Thomson verdict (good rows are gated on it)"),
}

PROFILE_COLUMNS: dict[str, tuple[str, str]] = {
    "contract_version": ("string", "state key contract version (#1454)"),
    "shot": ("integer", "VEST shot number"),
    "time_efit_s": ("number", "EFIT slice time, rounded to 1e-4 s"),
    "efit_lineage": ("string", "magnetics | electron_kinetic"),
    "efit_quality": ("string", "good | admissible"),
    "channel": ("integer", "Thomson channel index"),
    "r_m": ("number", "channel major radius [m]"),
    "z_m": ("number", "channel height [m]"),
    "psi_norm": ("number", "normalized poloidal flux at the channel"),
    "rho_tor_norm": ("number", "normalized toroidal-flux radius at the channel; empty when unavailable"),
    "n_e_m3": ("number", "Thomson n_e [m^-3]"),
    "n_e_error_m3": ("number", "Thomson n_e error (upper) [m^-3]"),
    "t_e_ev": ("number", "Thomson T_e [eV]"),
    "t_e_error_ev": ("number", "Thomson T_e error (upper) [eV]"),
    "p_e_pa": ("number", "e n_e T_e [Pa]"),
    "p_efit_pa": ("number", "EFIT pressure at the channel's psi_N [Pa]"),
    "r_p": ("number", "p_EFIT / p_e"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _criteria():
    spec = importlib.util.spec_from_file_location("lane_k_criteria", CRITERIA_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPOSITORY), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _f(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _status(record: Mapping[str, Any], name: str) -> str | None:
    return ((record.get("evaluation") or {}).get("verdicts") or {}).get(name, {}).get("status")


def _kinetic_veto(criteria, equilibrium: Mapping[str, Any], manifest: Mapping[str, Any],
                  magnetic_record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """criteria.admissible on the electron EFIT, from what its product carries."""
    import tempfile

    from omas import load_omas_json

    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    root = equilibrium["equilibrium"]["time_slice"][0]
    gq = root.get("global_quantities", {})
    # descriptors need an ODS; go through the documented JSON loader rather than
    # assigning into one (an ODS assignment is where paths get vivified)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"equilibrium": equilibrium["equilibrium"]}, handle)
    try:
        ods = load_omas_json(handle.name, consistency_check=False)
    finally:
        Path(handle.name).unlink()
    values = derive_global_descriptors(as_equilibrium(ods, time_index=0)).values

    def pick(key):
        return float(values[key].value) if key in values and values[key].available else math.nan

    scalars = {"betap": pick("beta_p_boundary_average"), "wmhd": pick("thermal_energy"),
               "q95": _f(gq.get("q_95")), "ipmhd": _f(gq.get("ip"))}
    pressure = np.asarray(root.get("profiles_1d", {}).get("pressure", []), dtype=float)
    record = {
        "setting": (manifest.get("configuration", {}).get("efit_preset") or {}).get("name"),
        "converged": bool((manifest.get("reconstruction") or {}).get("converged")),
        "pressure_min": float(pressure.min()) if pressure.size else math.nan,
        "scalars": scalars,
        "ip_measured": magnetic_record.get("ip_measured"),
        "fit": {"ip_sigma_median": (magnetic_record.get("fit") or {}).get("ip_sigma_median")},
    }
    return criteria.admissible(record), scalars


def build(filedb: Path, analysis_path: Path, out: Path, *, ti_te_ratio: float) -> dict[str, Any]:
    from vaft.validation import kinetic_state as ks

    criteria = _criteria()
    analysis = _load(analysis_path)
    # Labels are re-derived from the records with this checkout's criteria, so
    # gating and the verdict columns always come from one rule; disagreements
    # with the labels stored in the JSON are counted in the manifest.
    derived = criteria.slice_labels(analysis["records"])
    stored = {(int(l["shot"]), int(l["time_ms"])): l["label"] for l in analysis.get("labels", [])}
    label_rows = {(int(l["shot"]), int(l["time_ms"])): l for l in derived}
    relabelled = sorted(k for k, l in label_rows.items() if stored.get(k, l["label"]) != l["label"])
    labels = {k: l["label"] for k, l in label_rows.items() if l["label"] in ks.QUALITIES}
    # One slice can carry a record per study setting, plus the routine and error
    # records.  The row's context comes from the setting that earned the label
    # (the first good one, else the first admissible one, in name order).
    by_slice: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in analysis["records"]:
        if "error" not in record:
            by_slice.setdefault((int(record["shot"]), int(record["time_ms"])), []).append(record)
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for key, label in labels.items():
        earned = label_rows[key]["good"] or label_rows[key]["admissible"]
        chosen = next(r for name in earned for r in by_slice[key] if r.get("setting") == name)
        records[key] = {**chosen, "evaluation": criteria.evaluate(chosen)}
    labels_sha = _sha256(analysis_path)
    criteria_sha = _git("hash-object", str(CRITERIA_PATH))
    omas = filedb / "omas"

    state_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    def key_fields(shot, time_s, lineage, quality):
        return {"contract_version": ks.CONTRACT_VERSION, "shot": shot, "time_efit_s": ks.state_time(time_s),
                "efit_lineage": lineage, "efit_quality": quality}

    def add_match(row, eq, thomson, cp, time_s, tolerance):
        try:
            match = ks.match_thomson_pressure(eq, thomson, time_s=time_s, slice_tolerance_s=SLICE_TOLERANCE_S,
                                              ts_tolerance_s=tolerance)
        except LookupError as missing:
            row.update(efit_status="unavailable", efit_status_reason=str(missing))
            return None
        row["efit_status"] = "valid"
        row.update(ts_status=match["ts_status"], ts_reason=match.get("reason"),
                   time_ts_s=match.get("time_ts_s"), dt_ts_efit_s=match.get("dt_ts_efit_s"),
                   ts_tolerance_s=match.get("ts_tolerance_s"))
        if match["ts_status"] != "matched":
            return match
        lo, hi = match["psi_norm_span"]
        row.update(ts_points=match["points"], rho_coordinate=match["rho_coordinate"], r_sum=match["r_sum"],
                   log_ratio=match["log_ratio"], psi_norm_lo=lo, psi_norm_hi=hi)
        for channel in match["channels"]:
            profile_rows.append({**{k: row[k] for k in ("contract_version", "shot", "time_efit_s", "efit_lineage", "efit_quality")},
                                 "channel": channel["channel"], "r_m": channel["r"], "z_m": channel["z"],
                                 "psi_norm": channel["psi_norm"], "rho_tor_norm": channel["rho_tor_norm"],
                                 "n_e_m3": channel["n_e"], "n_e_error_m3": channel["n_e_error"],
                                 "t_e_ev": channel["t_e"], "t_e_error_ev": channel["t_e_error"],
                                 "p_e_pa": channel["p_e"], "p_efit_pa": channel["p_recon"], "r_p": channel["r_p"]})
        if cp is None:
            row["r_w_reason"] = "no core_profiles product"
            return match
        electron = ks.core_profiles_electron_pressure(cp, time_s=match["time_ts_s"], tolerance_s=tolerance)
        if not electron["available"]:
            row["r_w_reason"] = electron["reason"]
            return match
        row["cp_time_s"] = electron["time_s"]
        span = ks.integrated_pressure_ratio(eq, match["time_slice"], electron["rho_tor_norm"], electron["p_e"],
                                            psi_norm_span=(lo, hi))
        full = ks.integrated_pressure_ratio(eq, match["time_slice"], electron["rho_tor_norm"], electron["p_e"])
        if span["available"]:
            row.update(r_w=span["ratio"], r_w_cell_weights=span["cell_weights"])
        else:
            row["r_w_reason"] = span["reason"]
        if full["available"]:
            row["r_w_full"] = full["ratio"]
        match["integrate"] = lambda psi_span: ks.integrated_pressure_ratio(  # noqa: E731
            eq, match["time_slice"], electron["rho_tor_norm"], electron["p_e"], psi_norm_span=psi_span)
        return match

    shots = sorted({shot for shot, _ in labels})
    for shot in shots:
        mag_path = omas / "efit" / "magnetic" / str(shot) / "output" / "efit.json.gz"
        thomson_path = omas / "thomson" / str(shot) / "output" / "thomson.json.gz"
        cp_path = omas / "core_profiles" / str(shot) / "output" / "core_profiles.json.gz"
        kin_dir = omas / "electron_efit" / str(shot)
        eq = _load(mag_path)
        thomson = _load(thomson_path) if thomson_path.is_file() else {}
        cp = _load(cp_path) if cp_path.is_file() else None
        tolerance = ks.default_tolerance(np.asarray(eq["equilibrium"]["time"], dtype=float))
        manifest_path = kin_dir / "metadata" / "manifest.json"
        kin_manifest = _load(manifest_path) if manifest_path.is_file() else {}
        kin_time_ms = (kin_manifest.get("configuration") or {}).get("time_ms")
        kin_time_ms = int(round(kin_time_ms)) if kin_time_ms is not None else None
        mag_sha = _sha256(mag_path)
        mag_rows: dict[int, dict[str, Any]] = {}
        for (label_shot, time_ms), quality in sorted(labels.items()):
            if label_shot != shot:
                continue
            record = records[(shot, time_ms)]
            scalars = record.get("scalars") or {}
            fit = record.get("fit") or {}
            row = {**key_fields(shot, time_ms * 1e-3, "magnetics", quality),
                   "efit_setting": record.get("setting"), "efit_product": str(mag_path.relative_to(filedb)),
                   "efit_product_sha256": mag_sha, "ti_lineage": "none",
                   "criteria_log_ratio": (record.get("thomson") or {}).get("log_ratio"),
                   "ip_measured_a": record.get("ip_measured"), "ip_reconstructed_a": scalars.get("ipmhd"),
                   "betap": scalars.get("betap"), "li": scalars.get("li"), "q95": scalars.get("q95"),
                   "wmhd_j": scalars.get("wmhd"), "probe_reduced_chi2": fit.get("probe_reduced_chi2"),
                   "loop_reduced_chi2": fit.get("loop_reduced_chi2"), "virial_status": _status(record, "virial"),
                   "gs_status": _status(record, "grad_shafranov"),
                   "thomson_criterion_status": _status(record, "thomson")}
            if kin_time_ms == time_ms:
                status = kin_manifest.get("status")
                row["paired_kin_status"] = {"success": "valid", "no_output": "failed"}.get(status, "unavailable")
                row["paired_kin_reason"] = kin_manifest.get("error") or None
            else:
                row["paired_kin_status"] = "not_attempted"
            add_match(row, eq, thomson, cp, time_ms * 1e-3, tolerance)
            state_rows.append(row)
            mag_rows[time_ms] = row
        # electron-kinetic row, only on a good/admissible magnetics pair
        if kin_manifest.get("status") != "success" or kin_time_ms not in mag_rows:
            continue
        kin_path = kin_dir / "output" / "electron_efit.json.gz"
        kin = _load(kin_path)
        pair = mag_rows[kin_time_ms]
        veto, kin_scalars = _kinetic_veto(criteria, kin, kin_manifest, records[(shot, kin_time_ms)])
        row = {**key_fields(shot, kin_time_ms * 1e-3, "electron_kinetic", pair["efit_quality"]),
               "efit_setting": ((kin_manifest.get("configuration") or {}).get("efit_preset") or {}).get("name"),
               "efit_product": str(kin_path.relative_to(filedb)), "efit_product_sha256": _sha256(kin_path),
               "ti_lineage": "ti_eq_te_assumed", "ti_te_ratio_assumed": ti_te_ratio,
               "kinetic_admissible": veto["status"], "kinetic_admissible_reasons": ";".join(veto["reasons"]) or None,
               "kinetic_chi2": (kin_manifest.get("reconstruction") or {}).get("chi2"),
               "criteria_log_ratio": next((k.get("ts_log_ratio_kin") for k in analysis.get("kinetic", [])
                                           if int(k["shot"]) == shot and int(k["time_ms"]) == kin_time_ms), None),
               "ip_measured_a": pair["ip_measured_a"], "ip_reconstructed_a": kin_scalars["ipmhd"],
               "betap": kin_scalars["betap"], "q95": kin_scalars["q95"], "wmhd_j": kin_scalars["wmhd"]}
        match = add_match(row, kin, thomson, cp, kin_time_ms * 1e-3, tolerance)
        # Compare the two reconstructions over ONE domain: the magnetics row's
        # channel span, so c_p is not partly a difference of integration domains.
        if match is not None and "integrate" in match and math.isfinite(_f(pair.get("r_w"))):
            shared = match["integrate"]((pair["psi_norm_lo"], pair["psi_norm_hi"]))
            if shared["available"] and shared["ratio"] > 0 and pair["r_w"] > 0:
                row["r_w_on_pair_span"] = shared["ratio"]
                row["delta_r_w"] = shared["ratio"] - pair["r_w"]
                row["c_p"] = math.log(shared["ratio"] / pair["r_w"])
        state_rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    (out / "schema").mkdir(exist_ok=True)
    for name, columns, rows in (("state", STATE_COLUMNS, state_rows), ("profiles", PROFILE_COLUMNS, profile_rows)):
        with open(out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _cell(row.get(k)) for k in columns})
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"VAFT kinetic state {name} table, contract v{ks.CONTRACT_VERSION} (#1454)",
            "type": "object",
            "required": ["contract_version", "shot", "time_efit_s", "efit_lineage", "efit_quality"],
            "properties": {k: {"type": [t, "null"], "description": d} for k, (t, d) in columns.items()},
        }
        schema["properties"]["efit_lineage"]["enum"] = list(ks.LINEAGES)
        schema["properties"]["efit_quality"]["enum"] = list(ks.QUALITIES)
        (out / "schema" / f"{name}.schema.json").write_text(json.dumps(schema, indent=1) + "\n")
    manifest = {
        "contract_version": ks.CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "vaft_git": _git("rev-parse", "HEAD"),
        "vaft_dirty": bool(_git("status", "--porcelain", "--", "vaft", "workflow/kinetic_state")),
        "criteria_sha": criteria_sha,
        "inputs": {"filedb": str(filedb), "analysis": str(analysis_path), "analysis_sha256": labels_sha},
        "rows": {"state": len(state_rows), "profiles": len(profile_rows)},
        "labels": {"derived_with": "criteria.slice_labels", "differ_from_stored": len(relabelled),
                   "differing": [list(k) for k in relabelled]},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return {"state": state_rows, "profiles": profile_rows, "manifest": manifest}


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (np.floating,)):
        return _cell(float(value))
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ti-te-ratio", type=float, default=1.0,
                        help="Ti/Te the electron-EFIT products in --filedb were run with (#1414: 1.0)")
    args = parser.parse_args(argv)
    result = build(args.filedb.expanduser(), args.analysis.expanduser(), args.out.expanduser(),
                   ti_te_ratio=args.ti_te_ratio)
    print(json.dumps(result["manifest"]["rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
