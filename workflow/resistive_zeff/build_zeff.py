"""Build the resistive Z_eff atlas (#1214 Phase G) on the Lane K Tier A states.

Reads, never writes, the campaign FileDB and the Lane K state atlas
(contract #1454); writes ``zeff.csv`` (one row per window), ``slices.csv``
(one row per Lane K key in a window), ``sensitivity.csv``, their JSON Schemas
and ``MANIFEST.json`` into one versioned directory.

Windows
-------
A window is the longest run of consecutive EFIT slices of one shot whose
``magnetics`` rows are graded ``good`` or ``admissible``; Romero's balance
differentiates in time, so a window needs three slices.  It is classified
``flattop`` when the median ``|V_I|/|V_B|`` is below ``--flattop-max`` and
``ramp`` otherwise.  A window that cannot be inferred is still a row, with
``status = not_identifiable`` and the reason (#1214 Sec. 10):

* fewer than three graded slices in a row;
* no slice in the window has electron profiles at its own time;
* no matched slice has a positive observed resistance;
* the inductive correction dominates (``|V_I|/|V_B| >= --inductive-max``),
  so ``V_R`` would be a small difference of large numbers.

The electron profiles are Lane K's ``core_profiles/<shot>.json.gz`` fits,
matched to EFIT slices by time.  No bootstrap current is subtracted
(``bootstrap = none``, the #1214 Sec. 8 reference): it needs T_i, which every
Tier A row only has as an assumption.  Nothing is written to any
``core_profiles.zeff``.

Usage (vestserver, with the lane's sitecustomize shim on PYTHONPATH)::

    python3 workflow/resistive_zeff/build_zeff.py \\
        --filedb ~/runs/campaign/filedb \\
        --atlas ~/runs/campaign/atlas/v1 \\
        --out ~/runs/campaign/atlas/zeff
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "zeff-1"
KEY_TOLERANCE_S = 5e-5

#: Column name -> (JSON type, description); schemas are generated from these.
WINDOW_COLUMNS: dict[str, tuple[str, str]] = {
    "contract_version": ("string", "resistive Z_eff atlas contract (zeff-1)"),
    "state_contract_version": ("string", "Lane K state key contract the keys follow (#1454)"),
    "shot": ("integer", "VEST shot number"),
    "efit_lineage": ("string", "magnetics (the only lineage with time histories)"),
    "t_start_s": ("number", "first EFIT slice time of the window"),
    "t_end_s": ("number", "last EFIT slice time of the window"),
    "n_slices": ("integer", "EFIT slices in the window"),
    "n_states": ("integer", "slices with electron profiles at their own time (fitted)"),
    "window_class": ("string", "flattop | ramp | unknown"),
    "median_inductive_fraction": ("number", "median |V_I|/|V_B| over the window"),
    "status": ("string", "ok | bound_hit | non_monotonic | not_identifiable"),
    "reason": ("string", "why there is no estimate, or why it is poorly constrained"),
    "quantity": ("string", "Zeff_resistive: model-inferred, not a composition measurement"),
    "zeff": ("number", "resistive effective charge (nominal model)"),
    "zeff_uncertainty": ("number", "statistical 1-sigma from the residual scatter; empty with one state"),
    "z_min": ("number", "lower inference bound"),
    "z_max": ("number", "upper inference bound"),
    "conductivity_model": ("string", "spitzer_nrl | sauter | redl"),
    "ln_lambda": ("string", "Coulomb logarithm: 'sauter' (per surface) or a number"),
    "bootstrap_model": ("string", "none: no bootstrap current subtracted"),
    "current_source_assumption": ("string", "I_ni assumption stated to the observed path"),
    "smoothing": ("string", "time smoothing before differentiation"),
    "flux_normalization": ("string", "how the stored psi was brought to full Wb"),
    "residual_rms_v": ("number", "rms of V_R^obs - V_R^model at the fit [V]"),
    "normalized_residual": ("number", "residual rms over rms V_R^obs"),
    "zeff_spitzer_nrl": ("number", "Z_eff with the NRL parallel Spitzer model"),
    "zeff_sauter": ("number", "Z_eff with Sauter"),
    "zeff_redl": ("number", "Z_eff with Redl"),
    "max_abs_delta_t_e": ("number", "largest |dZ| for T_e x(1 +/- d)"),
    "max_abs_delta_n_e": ("number", "largest |dZ| for n_e x(1 +/- d)"),
    "max_abs_delta_li_3": ("number", "largest |dZ| for li_3 x(1 +/- d)"),
    "max_abs_delta_conductivity_model": ("number", "largest |dZ| across the other models"),
    "efit_product": ("string", "EFIT product path relative to the FileDB"),
    "efit_product_sha256": ("string", "sha256 of that product"),
    "profiles_source": ("string", "Lane K core_profiles file relative to the state atlas"),
}

SLICE_COLUMNS: dict[str, tuple[str, str]] = {
    "contract_version": ("string", "resistive Z_eff atlas contract (zeff-1)"),
    "state_contract_version": ("string", "Lane K state key contract (#1454)"),
    "shot": ("integer", "VEST shot number"),
    "time_efit_s": ("number", "EFIT slice time, rounded to 1e-4 s (Lane K key)"),
    "efit_lineage": ("string", "magnetics"),
    "efit_quality": ("string", "good | admissible (Lane K)"),
    "window_t_start_s": ("number", "the window this slice belongs to"),
    "ip_a": ("number", "plasma current [A]"),
    "li_3": ("number", "li_3 recomputed from int B_p^2 dV"),
    "v_b_v": ("number", "boundary loop voltage, Romero -dpsi_B/dt [V]"),
    "v_i_v": ("number", "inductive voltage L_i dI/dt + I/2 dL_i/dt [V]"),
    "v_r_v": ("number", "resistive voltage V_B - V_I [V]"),
    "r_p_obs_ohm": ("number", "observed plasma resistance V_R / (I_p - I_ni) [Ohm]"),
    "inductive_fraction": ("number", "|V_I| / |V_B|"),
    "flags": ("string", "observed-path flags, ';'-separated"),
    "has_state": ("boolean", "electron profiles at this time were used in the fit"),
    "r_p_model_ohm": ("number", "nominal-model resistance at the fitted Z_eff [Ohm]"),
    "excluded_current_fraction": ("number", "share of j_tor outside the profiles' support"),
}

SENSITIVITY_COLUMNS: dict[str, tuple[str, str]] = {
    "contract_version": ("string", "resistive Z_eff atlas contract (zeff-1)"),
    "shot": ("integer", "VEST shot number"),
    "t_start_s": ("number", "window start"),
    "input": ("string", "perturbed input"),
    "setting": ("string", "the perturbation"),
    "zeff": ("number", "re-inferred Z_eff"),
    "delta": ("number", "zeff - nominal"),
    "status": ("string", "inference status of the perturbed fit"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPOSITORY), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_ods(path: Path):
    from omas import ODS

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    ods = ODS(consistency_check=False)
    for ids, tree in data.items():
        ods[ids] = ODS(consistency_check=False).from_structure(tree)
    return ods


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _write(path: Path, columns: dict[str, tuple[str, str]], rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in columns})


def _schema(name: str, columns: dict[str, tuple[str, str]], required: list[str]) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"resistive Z_eff atlas: {name}",
        "type": "object",
        "required": required,
        "properties": {col: {"type": [kind, "null"] if col not in required else kind,
                             "description": text}
                       for col, (kind, text) in columns.items()},
    }


def _graded_runs(times: list[float], product_times: np.ndarray) -> list[list[float]]:
    """Maximal runs of graded slices that are consecutive in the EFIT product."""
    index = {}
    for t in times:
        k = int(np.argmin(np.abs(product_times - t)))
        if abs(product_times[k] - t) <= 6e-4:
            index[k] = t
    runs, current = [], []
    for k in sorted(index):
        if current and k != current[-1][0] + 1:
            runs.append(current)
            current = []
        current.append((k, index[k]))
    if current:
        runs.append(current)
    return [[float(product_times[k]) for k, _ in run] for run in runs]


def _states_for(ods, window: list[float]):
    from vaft.omas.resistive_zeff import flux_surface_state_ods

    states, notes = [], []
    if "core_profiles.profiles_1d" not in ods:
        return states, ["no core_profiles"]
    cp = ods["core_profiles.profiles_1d"]
    for i in range(len(cp)):
        t = float(cp[i]["time"]) if "time" in cp[i] else float(ods["core_profiles.time"][i])
        if not any(abs(t - w) <= KEY_TOLERANCE_S for w in window):
            continue
        try:
            states.append(flux_surface_state_ods(ods, time_slice=i))
        except ValueError as error:
            notes.append(f"t={t:.4f}: {error}")
    return states, notes


def infer_window(shot: int, window: list[float], ods, args) -> tuple[dict, list[dict], list[dict]]:
    from vaft.omas.resistive_zeff import romero_boundary_flux_ods
    from vaft.process.resistive_zeff import (
        Smoothing,
        model_resistance,
        observed_resistance,
        resistive_zeff_sensitivity,
    )

    row: dict[str, Any] = {
        "t_start_s": window[0], "t_end_s": window[-1], "n_slices": len(window),
        "z_min": args.bounds[0], "z_max": args.bounds[1], "conductivity_model": args.model,
        "ln_lambda": args.ln_lambda, "bootstrap_model": "none", "window_class": "unknown",
        "quantity": "Zeff_resistive (model-inferred; not a composition measurement)",
    }
    slices: list[dict] = []
    if len(window) < 3:
        row.update(status="not_identifiable",
                   reason=f"{len(window)} consecutive graded slices; the balance needs three")
        return row, slices, []

    smoothing = Smoothing("none")
    flux = romero_boundary_flux_ods(ods, time_range=(window[0] - KEY_TOLERANCE_S,
                                                      window[-1] + KEY_TOLERANCE_S),
                                    source={"shot": shot})
    observed = observed_resistance(flux, I_ni=0.0, smoothing=smoothing)
    median_fraction = float(np.nanmedian(observed.inductive_fraction))
    row.update(
        median_inductive_fraction=median_fraction,
        window_class="flattop" if median_fraction < args.flattop_max else "ramp",
        current_source_assumption=observed.provenance["current_source_assumption"],
        smoothing=smoothing.describe(), flux_normalization=flux.flux_normalization,
    )
    for k, t in enumerate(observed.time):
        slices.append({
            "time_efit_s": round(float(t), 4), "window_t_start_s": window[0],
            "ip_a": observed.I_p[k], "li_3": observed.li_3[k], "v_b_v": observed.V_B[k],
            "v_i_v": observed.V_I[k], "v_r_v": observed.V_R[k], "r_p_obs_ohm": observed.R_p[k],
            "inductive_fraction": observed.inductive_fraction[k],
            "flags": ";".join(observed.flags[k]), "has_state": False,
        })

    states, notes = _states_for(ods, window)
    row["n_states"] = len(states)
    if not states:
        row.update(status="not_identifiable",
                   reason="no slice has electron profiles at its own time"
                   + (f" ({'; '.join(notes)})" if notes else ""))
        return row, slices, []
    usable = [s for s in states
              if observed.inductive_fraction[int(np.argmin(np.abs(observed.time - s.time)))]
              < args.inductive_max]
    if not usable:
        row.update(status="not_identifiable",
                   reason=f"|V_I|/|V_B| >= {args.inductive_max} at every profiled slice: "
                          "V_R is a small difference of large numbers")
        return row, slices, []

    nominal, table = resistive_zeff_sensitivity(
        flux, usable, I_ni=0.0, smoothing=smoothing, model=args.model,
        ln_lambda=args.ln_lambda, bounds=tuple(args.bounds), weights="uniform",
        relative_perturbation=args.perturbation, smoothing_factors=(),
        models=("spitzer_nrl", "sauter", "redl"),
    )
    est = nominal.estimate
    row.update(status=est["status"], reason=nominal.reason or "", zeff=est["zeff"],
               zeff_uncertainty=est["uncertainty"],
               residual_rms_v=nominal.quality.get("residual_rms_V"),
               normalized_residual=nominal.quality.get("normalized_residual"))
    row[f"zeff_{args.model}"] = est["zeff"]
    for entry in table:
        if entry["input"] == "conductivity_model":
            row[f"zeff_{entry['setting']}"] = entry["zeff"]
    for key, value in nominal.sensitivity.items():
        row[key.lower()] = value  # max_abs_delta_T_e -> max_abs_delta_t_e
    if est["zeff"] is not None:
        for state in usable:
            k = int(np.argmin(np.abs(observed.time - state.time)))
            slices[k]["has_state"] = True
            slices[k]["r_p_model_ohm"] = model_resistance(
                state, est["zeff"], model=args.model, ln_lambda=args.ln_lambda).R_p
            slices[k]["excluded_current_fraction"] = state.source.get("excluded_current_fraction")
    sens = [{"t_start_s": window[0], **entry} for entry in table]
    return row, slices, sens


def build(filedb: Path, atlas: Path, out: Path, args) -> dict:
    states = list(csv.DictReader(open(atlas / "state.csv")))
    quality = {}
    by_shot: dict[int, list[float]] = defaultdict(list)
    for r in states:
        if r["efit_lineage"] != "magnetics" or r["efit_quality"] not in ("good", "admissible"):
            continue
        shot, t = int(r["shot"]), float(r["time_efit_s"])
        by_shot[shot].append(t)
        quality[(shot, round(t, 4))] = (r["efit_quality"], r["efit_product"],
                                        r["efit_product_sha256"], r["contract_version"])
    windows, slices, sensitivity = [], [], []
    for shot in sorted(by_shot):
        first = next(v for (s, _), v in quality.items() if s == shot)
        product = filedb / first[1]
        base = {"contract_version": CONTRACT_VERSION, "state_contract_version": first[3],
                "shot": shot, "efit_lineage": "magnetics", "efit_product": first[1],
                "efit_product_sha256": first[2],
                "profiles_source": f"core_profiles/{shot}.json.gz"}
        try:
            ods = _load_ods(product)
            profiles = atlas / "core_profiles" / f"{shot}.json.gz"
            if profiles.exists():
                cp = _load_ods(profiles)
                ods["core_profiles"] = cp["core_profiles"]
            product_times = np.asarray(ods["equilibrium.time"], dtype=float)
            runs = _graded_runs(sorted(by_shot[shot]), product_times)
        except Exception as error:  # noqa: BLE001 - recorded, the atlas goes on
            windows.append({**base, "status": "not_identifiable",
                            "reason": f"could not read inputs: {type(error).__name__}: {error}"})
            continue
        window = max(runs, key=len) if runs else []
        try:
            row, srows, sens = infer_window(shot, window, ods, args)
        except Exception as error:  # noqa: BLE001
            row, srows, sens = ({"t_start_s": window[0] if window else None,
                                 "n_slices": len(window), "status": "not_identifiable",
                                 "reason": f"{type(error).__name__}: {error}"}, [], [])
        windows.append({**base, **row})
        for s in srows:
            key = (shot, round(s["time_efit_s"], 4))
            s.update(contract_version=CONTRACT_VERSION, state_contract_version=first[3],
                     shot=shot, efit_lineage="magnetics",
                     efit_quality=quality.get(key, ("",))[0])
            slices.append(s)
        for s in sens:
            sensitivity.append({"contract_version": CONTRACT_VERSION, "shot": shot, **s})
        print(f"{shot}: {windows[-1].get('status')} zeff={windows[-1].get('zeff')} "
              f"{windows[-1].get('reason', '')}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "schema").mkdir(exist_ok=True)
    for name, columns, rows, required in (
        ("zeff", WINDOW_COLUMNS, windows, ["contract_version", "shot", "status"]),
        ("slices", SLICE_COLUMNS, slices,
         ["contract_version", "shot", "time_efit_s", "efit_lineage", "efit_quality"]),
        ("sensitivity", SENSITIVITY_COLUMNS, sensitivity, ["contract_version", "shot", "input"]),
    ):
        _write(out / f"{name}.csv", columns, rows)
        (out / "schema" / f"{name}.schema.json").write_text(
            json.dumps(_schema(name, columns, required), indent=1) + "\n")
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vaft_commit": _git("rev-parse", "HEAD"),
        "command": " ".join(sys.argv),
        "inputs": {"filedb": str(filedb), "atlas": str(atlas),
                   "state_sha256": _sha256(atlas / "state.csv")},
        "settings": {"model": args.model, "ln_lambda": args.ln_lambda, "bounds": args.bounds,
                     "flattop_max": args.flattop_max, "inductive_max": args.inductive_max,
                     "perturbation": args.perturbation, "bootstrap": "none",
                     "I_ni": 0.0, "smoothing": "none"},
        "counts": {status: sum(1 for w in windows if w.get("status") == status)
                   for status in ("ok", "bound_hit", "non_monotonic", "not_identifiable")},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="redl", choices=("spitzer_nrl", "sauter", "redl"))
    parser.add_argument("--ln-lambda", default="sauter")
    parser.add_argument("--bounds", type=float, nargs=2, default=(1.0, 8.0))
    parser.add_argument("--flattop-max", type=float, default=0.3,
                        help="median |V_I|/|V_B| below which a window is a flat top")
    parser.add_argument("--inductive-max", type=float, default=1.0,
                        help="slices with |V_I|/|V_B| at or above this are not fitted")
    parser.add_argument("--perturbation", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.ln_lambda != "sauter":
        args.ln_lambda = float(args.ln_lambda)
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore", RuntimeWarning)
    manifest = build(args.filedb.expanduser(), args.atlas.expanduser(), args.out.expanduser(), args)
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
