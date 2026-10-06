"""Lane D: errors in variables, Kadomtsev completion and closures (#548 sections 6-9; layers D-F).

Reads the Tier A confinement table and, on each selection of ``fit.py``:

1. **Errors in variables (ODR).**
   - The ODR fits W = C I^aI B^aB P^(1+aP). Fitting W, not tau = W/P, keeps the
     response error independent of the power's.
   - It is scanned over the assumed log error of P_net. That error is the one
     that couples tau = W/P to its own predictor.
   - The empirical P scatter, from the two P_OH paths (flux loop and EFIT
     boundary flux), is reported beside the scan.
   - A shot bootstrap gives the uncertainty at the reference error.
2. **Kadomtsev completion and dimensionless form (D, E).**
   - The size exponent comes from the Kadomtsev constraint, with the full
     cluster covariance propagated.
   - The completed (mu_rho, mu_beta, mu_nu, mu_q) are given with their errors.
   - Every value is labelled ``assumed_not_measured``: VEST has no size scan.
3. **Closures (F).** Three alternative hypotheses, each fitted as a linear
   constraint on the exponents:
   - mu_rho = -2 (Bohm);
   - mu_rho = -2.5 (intermediate, one index, not a sum);
   - mu_rho = -3 (gyro-Bohm).

   Each is compared with the free fit on:
   - RMS log residual and Gaussian AIC/BIC;
   - a cluster Wald test of the constraint, F(1, G-1);
   - leave-one-shot-out error;
   - shot-bootstrap stability;
   - the density treatment (A without n_e, B with the Thomson n_e on its subset)
     and the selection.
4. **NSTX (section 9).**
   - Compared against, never used as a prior: Buxton 2019 (gyro-Bohm
     completed) and Kaye 2006 (NSTX2006L/H).
   - Their aI, aB, aP and completed mu_rho are set beside VEST's.

Usage::

    python closures.py --table ~/runs/campaign/atlas/confinement/table.csv \\
        --out ~/runs/campaign/atlas/confinement/closures
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import BASE, SELECTIONS, _git, select  # noqa: E402

CLOSURES = {"bohm": -2.0, "intermediate": -2.5, "gyro_bohm": -3.0}
#: Equation error of ln W (measurement plus intrinsic scatter) and measurement
#: error of ln P, scanned as a grid: only their ratio matters to the solution.
SIGMA_LOG_W_SCAN = (0.10, 0.20, 0.30)
SIGMA_LOG_IB = 0.02
SIGMA_LOG_P_SCAN = (0.05, 0.10, 0.20, 0.30)
#: Reference point for the shot bootstrap: P error near the P_OH path scatter,
#: W equation error near the free fit's RMS log residual.
SIGMA_REF = (0.20, 0.20)
#: The power predictor of the ODR. P_OH is independent of W; P_net = P_OH - dW/dt
#: carries W's noise through dW/dt and is reported for comparison only.
ODR_POWER = {"p_ohm": "p_ohm_W", "p_net": "p_loss_W"}
NSTX = {
    # Buxton et al., PPCF 61 (2019) 035006: gyro-Bohm-completed NSTX scaling, as
    # quoted in #548 Sec. 9 (exponents not re-checked against the paper here).
    "NSTX Buxton2019 (gyro-Bohm completed)": {"i_p": 0.54, "b_t": 0.91, "p_net": -0.38, "n_e": -0.05},
}


def _nstx_from_constants():
    from vaft.formula.constants import _SCALING_COEFS

    out = {}
    for key in ("NSTX2006L", "NSTX2006H"):
        e = _SCALING_COEFS[key]["exponents"]
        out[f"{key} (Kaye 2006)"] = {"i_p": e["Ip_MA"], "b_t": e["Bt"], "p_net": e["P_MW"], "n_e": e["n_19"]}
    return out


def _gaussian_ic(rss, n, k):
    """AIC and BIC of a Gaussian log-linear fit with k free coefficients."""
    ll_term = n * np.log(rss / n)
    return ll_term + 2 * k, ll_term + k * np.log(n)


def _loso_constrained(y, x, groups, coef, rhs):
    from vaft.process.confinement import fit_constrained_confinement_scaling

    labels = np.unique(groups)
    err = []
    for label in labels:
        held = groups == label
        if np.unique(groups[~held]).size < 2:
            continue
        try:
            fit = fit_constrained_confinement_scaling(
                y[~held], {k: v[~held] for k, v in x.items()}, groups[~held], coef, rhs)
        except ValueError:
            continue
        logx = np.column_stack([np.ones(held.sum())] + [np.log(x[k][held]) for k in x])
        ok = np.isfinite(logx).all(axis=1) & (y[held] > 0)
        err.extend(np.log(y[held][ok]) - logx[ok] @ fit.coef)
    err = np.asarray(err)
    return float(np.sqrt(np.mean(err**2))) if err.size else np.nan


def _boot_constrained(y, x, groups, coef, rhs, n_boot, seed):
    from vaft.process.confinement import fit_constrained_confinement_scaling

    rng = np.random.default_rng(seed)
    labels = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in labels]
    draws = []
    for _ in range(n_boot):
        draw = rng.integers(0, labels.size, labels.size)
        rows = np.concatenate([members[i] for i in draw])
        # A resample repeats shots; each copy is its own cluster.
        relabel = np.concatenate([np.full(members[i].size, j) for j, i in enumerate(draw)])
        try:
            fit = fit_constrained_confinement_scaling(
                y[rows], {k: v[rows] for k, v in x.items()}, relabel, coef, rhs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        draws.append(fit.coef[1:])
    return np.asarray(draws)


def closures_for(name, frame, predictors, n_boot):
    """Free fit and the three closures on one data set, with comparison metrics."""
    from scipy import stats

    from vaft.process.confinement import (
        closure_constraint,
        dimensionless_confinement_indices,
        infer_kadomtsev_size_exponent,
        fit_confinement_scaling,
        fit_constrained_confinement_scaling,
        leave_one_group_out_scaling,
    )

    y = frame["tau_e_th_s"].to_numpy(float)
    x = {k: frame[c].to_numpy(float) for k, c in predictors.items()}
    g = frame["shot"].to_numpy()
    free = fit_confinement_scaling(y, x, g)
    alpha = free.exponents()
    try:
        dim = dimensionless_confinement_indices(alpha, free.cov[1:, 1:])
    except ValueError:  # aP exactly -1: no dimensionless form, the fits still stand
        nan4 = np.full(4, np.nan)
        dim = {"alpha_R": np.nan, "alpha_R_stderr": np.nan, "value": nan4, "stderr": nan4,
               "names": ("mu_rho", "mu_beta", "mu_nu", "mu_q")}
    rows = [{
        "data": name, "model": "free", "mu_rho_imposed": np.nan, "n": free.n, "shots": free.n_groups,
        **{f"a_{k}": v for k, v in alpha.items()},
        **{f"se_{k}": s for k, s in zip(free.names[1:], free.stderr[1:])},
        "alpha_R_kadomtsev": dim["alpha_R"], "alpha_R_kadomtsev_se": dim["alpha_R_stderr"],
        **{f"{k}_completed": v for k, v in zip(dim["names"], dim["value"])},
        **{f"{k}_completed_se": v for k, v in zip(dim["names"], dim["stderr"])},
        "rmse_log": free.rmse_log, "aic": _gaussian_ic(np.sum(free.residuals**2), free.n, len(free.coef))[0],
        "bic": _gaussian_ic(np.sum(free.residuals**2), free.n, len(free.coef))[1],
        "loso_rmse_log": leave_one_group_out_scaling(y, x, g)["rmse_log"],
        "wald_F": np.nan, "wald_p": np.nan, "size_exponent_status": "assumed_not_measured",
        # Every dimensionless index divides by 1 + aP: within ~2 sigma of zero
        # the completed indices are undetermined, whatever their quoted error.
        "one_plus_aP_over_se": (1.0 + alpha["p_net"]) / free.stderr[free.names.index("p_net")],
    }]
    for label, mu in CLOSURES.items():
        coef, rhs = closure_constraint(mu, list(x))
        fit = fit_constrained_confinement_scaling(y, x, g, coef, rhs)
        c = np.array([0.0] + [coef[k] for k in x])
        gap = float(c @ free.coef - rhs)
        var = float(c @ free.cov @ c)
        wald = gap**2 / var if var > 0 else np.inf
        boot = _boot_constrained(y, x, g, coef, rhs, n_boot, seed=548)
        a = fit.exponents()
        rss = float(np.sum(fit.residuals**2))
        aic, bic = _gaussian_ic(rss, fit.n, len(fit.coef) - 1)
        rows.append({
            "data": name, "model": label, "mu_rho_imposed": mu, "n": fit.n, "shots": fit.n_groups,
            **{f"a_{k}": v for k, v in a.items()},
            **{f"se_{k}": s for k, s in zip(fit.names[1:], fit.stderr[1:])},
            **{f"boot_sd_{k}": s for k, s in zip(fit.names[1:], boot.std(axis=0) if len(boot) else
                                                     [np.nan] * len(a))},
            "alpha_R_kadomtsev": infer_kadomtsev_size_exponent(a, fit.cov[1:, 1:])[0],
            "rmse_log": fit.rmse_log, "aic": aic, "bic": bic,
            "loso_rmse_log": _loso_constrained(y, x, g, coef, rhs),
            "wald_F": wald, "wald_p": float(stats.f.sf(wald, 1, free.n_groups - 1)),
            "size_exponent_status": "assumed_not_measured",
        })
    return rows


def odr_scan(name, frame, n_boot):
    """ODR of W on (I_p, B_T, P) over a grid of error assumptions, for P = P_OH and P = P_net."""
    from vaft.process.confinement import fit_confinement_scaling_odr

    y = frame["w_th_J"].to_numpy(float)
    groups = frame["shot"].to_numpy()
    labels = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in labels]
    rows = []
    for power, column in ODR_POWER.items():
        x = {"i_p": frame["i_p_A"].to_numpy(float), "b_t": frame["b_t_T"].to_numpy(float),
             power: frame[column].to_numpy(float)}
        for sw in SIGMA_LOG_W_SCAN:
            for sp in SIGMA_LOG_P_SCAN:
                errors = {"i_p": SIGMA_LOG_IB, "b_t": SIGMA_LOG_IB, power: sp}
                out = fit_confinement_scaling_odr(y, x, sigma_log_response=sw, sigma_log_predictors=errors)
                row = {"data": name, "power": power, "sigma_log_w": sw, "sigma_log_p": sp,
                       "ratio_p_over_w": sp / sw, "n": out["n"]}
                for k, b, o in zip(out["names"], out["coef"], out["ols_coef"]):
                    shift = 1.0 if k == power else 0.0  # W ~ P^(1 + aP): report the tau exponent
                    key = "power" if k == power else k
                    row.update({f"a_{key}": b - shift, f"ols_a_{key}": o - shift})
                if (sp, sw) == SIGMA_REF:
                    rng = np.random.default_rng(548)
                    draws = []
                    for _ in range(n_boot):
                        r = np.concatenate([members[i] for i in rng.integers(0, labels.size, labels.size)])
                        try:
                            draws.append(fit_confinement_scaling_odr(
                                y[r], {k: v[r] for k, v in x.items()}, sigma_log_response=sw,
                                sigma_log_predictors=errors)["coef"])
                        except ValueError:
                            continue
                    if draws:
                        draws = np.asarray(draws)
                        for j, k in enumerate(out["names"]):
                            shift = 1.0 if k == power else 0.0
                            key = "power" if k == power else k
                            lo, hi = np.percentile(draws[:, j], [2.5, 97.5]) - shift
                            row.update({f"boot95_lo_{key}": lo, f"boot95_hi_{key}": hi})
                    row["boot_kept"] = len(draws)
                rows.append(row)
    return rows


#: Intrinsic scatter added in quadrature to the per-row EFIT model-form spread, as
#: a lower and a central assumption: the spread is one part of the equation error
#: of ln W, not all of it (#579, #548).
SIGMA_LOG_INTRINSIC = (0.0, 0.10)


def join_model_spread(frame: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    """Attach the #579 per-state EFIT model-form spread of W, matched by (shot, time) to 1e-4 s."""
    keep = ["shot", "w_mhd_J_admissible_sigma_log", "w_mhd_J_admissible_median", "admissible_bimodal",
            "w_mhd_J_viable_sigma_log", "w_mhd_J_viable_n"]
    right = spread.assign(time_key=spread["time_efit_s"].round(4))[keep + ["time_key"]]
    out = frame.assign(time_key=frame["time_efit_s"].round(4)).merge(
        right, on=["shot", "time_key"], how="left", validate="many_to_one")
    out["admissible_bimodal"] = out["admissible_bimodal"].map(
        {True: True, False: False, "True": True, "False": False}).astype("boolean")
    return out.drop(columns="time_key")


def odr_model_spread(name: str, frame: pd.DataFrame) -> list:
    """ODR of W on (I_p, B_T, P_OH) with the EFIT model-form spread as the per-row W error.

    On the rows the #579 ensemble covers: a constant sigma_W = 0.2 (the closure
    scan's reference) against per-row sigma_W = sqrt(sigma_EFIT^2 + sigma_int^2),
    with and without the bimodal rows (two solution branches, one sigma is wrong
    there), and with the tight 'viable' spread where it exists.
    """
    from vaft.process.confinement import fit_confinement_scaling_odr

    errors = {"i_p": SIGMA_LOG_IB, "b_t": SIGMA_LOG_IB, "p_ohm": SIGMA_REF[0]}
    covered = frame.loc[np.isfinite(frame["w_mhd_J_admissible_sigma_log"])]
    unimodal = covered.loc[~covered["admissible_bimodal"].fillna(False).astype(bool)]
    viable = frame.loc[np.isfinite(frame["w_mhd_J_viable_sigma_log"])]
    cases = [("constant 0.2", unimodal, None, 0.0)]
    for intrinsic in SIGMA_LOG_INTRINSIC:
        cases += [(f"admissible, unimodal, +{intrinsic:g} intrinsic", unimodal, "w_mhd_J_admissible_sigma_log", intrinsic),
                  (f"admissible, all, +{intrinsic:g} intrinsic", covered, "w_mhd_J_admissible_sigma_log", intrinsic),
                  (f"viable, +{intrinsic:g} intrinsic", viable, "w_mhd_J_viable_sigma_log", intrinsic)]
    rows = []
    for label, data, column, intrinsic in cases:
        row = {"data": name, "case": label, "rows": len(data), "shots": int(data["shot"].nunique())}
        if column is None:
            sigma = SIGMA_REF[1]
        else:
            sigma = np.sqrt(data[column].to_numpy(float) ** 2 + intrinsic**2)
            row["sigma_w_median"] = float(np.median(sigma))
        x = {"i_p": data["i_p_A"].to_numpy(float), "b_t": data["b_t_T"].to_numpy(float),
             "p_ohm": data["p_ohm_W"].to_numpy(float)}
        try:
            out = fit_confinement_scaling_odr(data["w_th_J"].to_numpy(float), x, sigma_log_response=sigma,
                                              sigma_log_predictors=errors)
        except ValueError as exc:
            rows.append({**row, "error": str(exc)[:120]})
            continue
        coef = dict(zip(out["names"], out["coef"]))
        ols = dict(zip(out["names"], out["ols_coef"]))
        rows.append({**row, "n": out["n"], "method": out["method"],
                     "a_i_p": coef["i_p"], "a_b_t": coef["b_t"], "a_p": coef["p_ohm"] - 1.0,
                     "ols_a_p": ols["p_ohm"] - 1.0})
    return rows


def main(argv=None) -> int:
    from vaft.process.confinement import dimensionless_confinement_indices

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--model-spread", default=None,
                   help="#579 model_spread.csv: per-state EFIT model-form spread of W for the ODR (optional)")
    args = p.parse_args(argv)
    table_path = Path(args.table).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(table_path)

    closure_rows, odr_rows, failures, p_scatter = [], [], [], {}
    for sel_name, thresholds in SELECTIONS.items():
        frame = table.loc[select(table, thresholds)].reset_index(drop=True)
        frame = frame.loc[np.isfinite(frame["tau_e_th_s"]) & (frame["tau_e_th_s"] > 0)].reset_index(drop=True)
        ts = frame.loc[np.isfinite(frame["n_e_line_avg_m3"]) & (frame["n_e_line_avg_m3"] > 0)]
        ratio = np.log(frame["p_ohm_W"] / frame["p_ohm_efit_flux_W"])
        ratio = ratio[np.isfinite(ratio)]
        # Two independent estimates of the same P_OH: their log difference has
        # variance 2 sigma^2 if both carry the same error.
        dwdt_share = (frame["dwdt_W"].abs() / frame["p_ohm_W"]).replace([np.inf, -np.inf], np.nan)
        p_scatter[sel_name] = {
            "n": int(ratio.size), "median_log_ratio": float(np.median(ratio)),
            # Lower bound: the two paths share I_p, whose error cancels in the ratio.
            "sigma_log_p_ohm_each_lower_bound": float(np.std(ratio) / np.sqrt(2.0)),
            # P_net = P_OH - dW/dt: how much of it is the W-derived term.
            "median_abs_dwdt_over_p_ohm": float(np.nanmedian(dwdt_share)),
        }
        for data_name, data, preds in (
            (f"{sel_name}:A", frame, BASE),
            (f"{sel_name}:A_ts_subset", ts, BASE),
            (f"{sel_name}:B_with_n_e", ts, {**BASE, "n_e": "n_e_line_avg_m3"}),
        ):
            try:
                closure_rows += closures_for(data_name, data, preds, args.n_boot)
            except ValueError as exc:
                failures.append({"data": data_name, "step": "closures", "error": str(exc)})
        try:
            odr_rows += odr_scan(f"{sel_name}:A", frame, args.n_boot)
        except ValueError as exc:
            failures.append({"data": f"{sel_name}:A", "step": "odr", "error": str(exc)})

    nstx = {**NSTX, **_nstx_from_constants()}
    comparison = []
    for name, a in nstx.items():
        dim = dimensionless_confinement_indices(a, np.zeros((4, 4)))
        comparison.append({"scaling": name, **{f"a_{k}": v for k, v in a.items()},
                           "alpha_R_kadomtsev": dim["alpha_R"],
                           **{f"{k}_completed": v for k, v in zip(dim["names"], dim["value"])}})
    vest = [r for r in closure_rows if r["model"] == "free"]
    for r in vest:
        comparison.append({"scaling": f"VEST {r['data']}",
                           **{k: r[k] for k in r if k.startswith("a_") or k.endswith("_completed")
                              or k.endswith("_completed_se") or k.startswith("se_")},
                           "alpha_R_kadomtsev": r["alpha_R_kadomtsev"]})

    pd.DataFrame(closure_rows).to_csv(out / "closures.csv", index=False)
    pd.DataFrame(odr_rows).to_csv(out / "odr_scan.csv", index=False)
    spread_rows, spread_coverage = [], {}
    if args.model_spread:
        spread = pd.read_csv(Path(args.model_spread).expanduser())
        joined = join_model_spread(table, spread)
        for sel_name, thresholds in SELECTIONS.items():
            frame = joined.loc[select(joined, thresholds)]
            frame = frame.loc[np.isfinite(frame["tau_e_th_s"]) & (frame["tau_e_th_s"] > 0)].reset_index(drop=True)
            covered = frame.loc[np.isfinite(frame["w_mhd_J_admissible_sigma_log"])]
            spread_coverage[sel_name] = {
                "rows": len(frame), "covered": len(covered),
                "bimodal": int(covered["admissible_bimodal"].fillna(False).astype(bool).sum()),
                "viable": int(np.isfinite(frame["w_mhd_J_viable_sigma_log"]).sum()),
                "median_sigma_log_admissible": float(covered["w_mhd_J_admissible_sigma_log"].median()),
                # The ensemble's median W against the production reconstruction's W.
                "median_ensemble_w_over_table_w": float((covered["w_mhd_J_admissible_median"]
                                                         / covered["w_th_J"]).median()),
            }
            spread_rows += odr_model_spread(f"{sel_name}:A", frame)
        pd.DataFrame(spread_rows).to_csv(out / "odr_model_spread.csv", index=False)
    pd.DataFrame(comparison).to_csv(out / "nstx_comparison.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "vaft_git": _git("rev-parse", "HEAD"),
        "vaft_dirty": bool(_git("status", "--porcelain")),
        "table": {"path": str(table_path), "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()},
        "selections": {k: dict(v) for k, v in SELECTIONS.items()}, "closures": CLOSURES, "n_boot": args.n_boot,
        "odr": {"sigma_log_w_scan": SIGMA_LOG_W_SCAN, "sigma_log_i_b": SIGMA_LOG_IB,
                "sigma_log_p_scan": SIGMA_LOG_P_SCAN, "bootstrap_at_sigma_p_w": SIGMA_REF,
                "power_predictors": ODR_POWER, "p_ohm_path_scatter": p_scatter,
                "model_spread": ({"path": str(Path(args.model_spread).expanduser()),
                                  "sha256": hashlib.sha256(Path(args.model_spread).expanduser().read_bytes()).hexdigest(),
                                  "coverage": spread_coverage, "sigma_log_intrinsic": SIGMA_LOG_INTRINSIC}
                                 if args.model_spread else None),
                "independence": ("P_OH is independent of W; P_net = P_OH - dW/dt is not (dW/dt is "
                                 "W-derived), so the P_net rows are a comparison only")},
        "notes": ("alpha_R_kadomtsev and *_completed are Kadomtsev-completed (assumed size exponent, not "
                  "measured; VEST has no size scan). Closures are alternative hypotheses. NSTX values are "
                  "a comparison, never a prior. ODR fits W with P's exponent reported as the tau exponent."),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    cols = ["data", "model", "n", "shots", "a_i_p", "a_b_t", "a_p_net", "rmse_log", "aic", "bic",
            "loso_rmse_log", "wald_p", "mu_rho_completed", "one_plus_aP_over_se", "alpha_R_kadomtsev"]
    print(pd.DataFrame(closure_rows).reindex(columns=cols).round(3).to_string(index=False))
    print(pd.DataFrame(odr_rows).round(3).to_string(index=False))
    print(json.dumps(p_scatter, indent=1))
    if spread_rows:
        print(pd.DataFrame(spread_rows).round(3).to_string(index=False))
        print(json.dumps(spread_coverage, indent=1))
    if failures:
        print(pd.DataFrame(failures).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
