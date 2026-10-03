"""Lane D confinement regression and identifiability (#548 sections 3, 4, 6; layers B and C).

Reads the Tier A confinement table that ``build_table.py`` wrote and fits

    tau_E = C I_p^aI B_T^aB P_net^aP                (A: density-free, primary)
    tau_E = C I_p^aI B_T^aB P_net^aP n_e^an         (B: Thomson line density)

with errors clustered by shot, a shot bootstrap, leave-one-shot-out refits,
influence and a Huber fit. Every fit is repeated on the sensitivity selection.
Only predictors VEST actually scans are primary (I_p, B_T, P_net); R, epsilon and
kappa stay descriptive (#548 section 3). Fit B and fit A_ts use the same Thomson
subset, so the change in aB, aP and the condition number is due to n_e alone,
not to a different sample.

Not done here: the "W formulation" of section 6. Under least squares, regressing
ln W = ln tau + ln P on the same predictors returns the tau fit with aP shifted by
exactly 1, so it cannot reveal the tau = W/P error coupling; that needs an
errors-in-variables (ODR) fit, a follow-up on the Lane D log (#1490). Bootstrap and
leave-one-shot-out refits are least squares, so they are reported only for the
least-squares fits.

Selections are decided from the table's evidence columns with
vaft.process.confinement.confinement_slice_decision, so they can be changed
without rebuilding the table (decision of 2026-10-02 on #1490):

    primary      |dIp/dt|/Ip * tau_E <= 0.20, |dW/dt|/P_OH <= 1.0, |Ip| >= 30 kA
    sensitivity  0.05 and 0.5

Usage::

    python fit.py --table ~/runs/campaign/atlas/confinement/table.csv \\
        --out ~/runs/campaign/atlas/confinement/fits
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SELECTIONS = {
    "primary": {"ip_min": 30e3, "max_dwdt_fraction": 1.0, "max_ip_change_per_tau": 0.20},
    "sensitivity": {"ip_min": 30e3, "max_dwdt_fraction": 0.5, "max_ip_change_per_tau": 0.05},
}
BASE = {"i_p": "i_p_A", "b_t": "b_t_T", "p_net": "p_loss_W"}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return ""


def select(table: pd.DataFrame, thresholds: dict) -> np.ndarray:
    from vaft.process.confinement import ConfinementSliceEvidence, confinement_slice_decision

    ev = ConfinementSliceEvidence(
        ip_abs=table["i_p_A"].to_numpy(float), ip_rate=table["ip_rate_1_s"].to_numpy(float),
        ip_change_per_tau=table["ip_change_per_tau"].to_numpy(float),
        dwdt_fraction=table["dwdt_fraction"].to_numpy(float),
        finite=table["rule_finite"].astype("boolean").fillna(False).to_numpy(bool),
    )
    accepted = confinement_slice_decision(ev, **thresholds)["accepted"]
    return accepted & (table["quality_status"] == "evaluated").to_numpy()


def run_fit(name, frame, response_col, predictors, *, robust=False, n_boot=2000):
    from vaft.process.confinement import (
        bootstrap_confinement_scaling,
        fit_confinement_scaling,
        leave_one_group_out_scaling,
    )

    y = frame[response_col].to_numpy(float)
    x = {k: frame[c].to_numpy(float) for k, c in predictors.items()}
    groups = frame["shot"].to_numpy()
    fit = fit_confinement_scaling(y, x, groups, robust=robust)
    k = len(fit.names)
    if robust:  # the resampling refits are least squares: not reported for a Huber fit
        boot = {"percentile95": np.full((k, 2), np.nan), "samples": np.empty((0, k)), "rejected": 0}
        loso = {"coef": np.full((1, k), np.nan), "rmse_log": np.nan}
    else:
        boot = bootstrap_confinement_scaling(y, x, groups, n_boot=n_boot, seed=548)
        loso = leave_one_group_out_scaling(y, x, groups)
    rows = []
    for j, term in enumerate(fit.names):
        loso_j = loso["coef"][:, j]
        has_loso = bool(np.isfinite(loso_j).any())
        rows.append({
            "fit": name, "term": term, "coef": fit.coef[j],
            "se_cluster": fit.stderr[j], "se_iid": fit.stderr_iid[j],
            "ci95_lo": fit.ci95[j, 0], "ci95_hi": fit.ci95[j, 1],
            "boot95_lo": boot["percentile95"][j, 0], "boot95_hi": boot["percentile95"][j, 1],
            "loso_min": np.nanmin(loso_j) if has_loso else np.nan,
            "loso_max": np.nanmax(loso_j) if has_loso else np.nan,
        })
    summary = {
        "fit": name, "response": response_col, "robust": robust, "n": fit.n, "shots": fit.n_groups,
        "r2": fit.r2, "rmse_log": fit.rmse_log, "loso_rmse_log": loso["rmse_log"],
        "boot_kept": int(boot["samples"].shape[0]), "boot_rejected": int(boot["rejected"]),
        "max_cooks_distance": float(np.max(fit.cooks_distance)),
    }
    kept = frame.loc[fit.mask]
    influence = pd.DataFrame({
        "fit": name, "shot": kept["shot"].to_numpy(), "time_s": kept["time_s"].to_numpy(),
        "leverage": fit.leverage, "cooks_distance": fit.cooks_distance, "residual_log": fit.residuals,
    })
    return rows, summary, influence


def main(argv=None) -> int:
    from vaft.process.confinement import assess_predictor_identifiability

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    table_path = Path(args.table).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(table_path)

    coef_rows, summaries, influences, ident = [], [], [], {}
    for sel_name, thresholds in SELECTIONS.items():
        frame = table.loc[select(table, thresholds)].reset_index(drop=True)
        ts = frame.loc[np.isfinite(frame["n_e_line_avg_m3"]) & (frame["n_e_line_avg_m3"] > 0)]
        with_n = {**BASE, "n_e": "n_e_line_avg_m3"}
        plan = [
            (f"{sel_name}:A", frame, "tau_e_th_s", BASE, {}),
            (f"{sel_name}:A_robust", frame, "tau_e_th_s", BASE, {"robust": True}),
            (f"{sel_name}:A_ts_subset", ts, "tau_e_th_s", BASE, {}),
            (f"{sel_name}:B_with_n_e", ts, "tau_e_th_s", with_n, {}),
        ]
        # Identifiability on exactly the rows the fits can use (finite, positive tau_E).
        usable = lambda f: f.loc[np.isfinite(f["tau_e_th_s"]) & (f["tau_e_th_s"] > 0)]  # noqa: E731
        frame_u, ts_u = usable(frame), usable(ts)
        ident[sel_name] = {
            "rows": int(len(frame)), "shots": int(frame["shot"].nunique()),
            "ts_rows": int(len(ts)), "ts_shots": int(ts["shot"].nunique()),
            "log_spread": {k: float(np.nanstd(np.log(frame_u[c].where(frame_u[c] > 0))))
                           for k, c in with_n.items()},
            "A": assess_predictor_identifiability({k: frame_u[c].to_numpy(float) for k, c in BASE.items()}),
            "B": (assess_predictor_identifiability({k: ts_u[c].to_numpy(float) for k, c in with_n.items()})
                  if len(ts_u) > 4 else None),
        }
        for name, data, resp, preds, kw in plan:
            try:
                rows, summary, influence = run_fit(name, data, resp, preds, n_boot=args.n_boot, **kw)
            except ValueError as exc:  # too few rows or shots for this fit
                summaries.append({"fit": name, "response": resp, "error": str(exc)})
                continue
            coef_rows += rows
            summaries.append(summary)
            influences.append(influence)

    pd.DataFrame(coef_rows).to_csv(out / "coefficients.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    if influences:
        pd.concat(influences).to_csv(out / "influence.csv", index=False)
    (out / "identifiability.json").write_text(json.dumps(ident, indent=2), encoding="utf-8")
    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "vaft_git": _git("rev-parse", "HEAD"),
        "vaft_dirty": bool(_git("status", "--porcelain")),
        "table": {"path": str(table_path), "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()},
        "selections": SELECTIONS, "n_boot": args.n_boot, "bootstrap_seed": 548,
        "notes": ("coefficients are exponents (log_C is ln C in SI units); se_cluster/ci95 cluster by shot "
                  "(CR1, t with G-1 dof); boot95 is the shot bootstrap; loso_min/max the range over "
                  "leave-one-shot-out refits (least-squares fits only)"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
