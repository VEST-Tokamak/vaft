"""Lane D extensions: what the B_T exponent measures, and where the exponents come from (#548, #1490).

Three analyses on the primary selection of ``fit.py``:

1. **B_T decomposition.** ``b_t_T`` is the vacuum field at R_geo, |B0 R0| / R_geo.
   In Tier A, B0R0 (the TF current) varies by +-7 %, while R_geo moves from 0.32 to
   0.44 m. Fitting B0R0 and R_geo as separate predictors shows how much of
   alpha_B is a field dependence and how much is plasma position.
2. **Shot fixed effects (within estimator).** Logs are demeaned within each shot,
   so every between-shot difference is absorbed: wall state, Z_eff, fuelling, and
   the TF setting. Only the time evolution inside a shot fits the exponents.
   - Shots with one slice carry no within information and are dropped.
   - The TF field is nearly constant within a shot (sd of log B0R0 about 0.1 %).
     So only the R_geo part of B_T is informative there; a B0R0 term within
     shots is reported, but it is unidentified.
   - Errors are clustered by shot (CR1), with no degrees-of-freedom correction
     for the absorbed means: the standard choice for fixed effects nested in the
     clusters (Cameron and Miller 2015).
   - Compare with the between-shot fit on shot means to see which variation
     drives the pooled exponents.
3. **Direct dimensionless regression** on the Thomson subset. It fits
   Omega_i tau_E = C rho*^a beta^b nu*^c q^d directly, with no elimination of T
   through P = W/tau, so it avoids the 1 + alpha_P singularity of the completed
   indices.
   - <T> = W_mhd / (3 n V e), with W_mhd the ``w_th_J`` column, assuming T_i = T_e and n_i = n_e. n is the Thomson
     line average and V = 2 pi^2 a^2 R kappa_area.
   - rho*, beta, nu*: the ``vaft.formula.equilibrium`` kernels.
   - q: q_cyl.
   - Caution: W enters tau_E and T, and through T it enters rho* (^1/2),
     beta (^1) and nu* (^-2). Their errors are therefore correlated by
     construction, and the indices carry that coupling. They are a consistency
     check on the completed indices, not a replacement for them.

4. **Campaign offset.** The Tier A shots come in two blocks, 399xx-403xx and
   429xx-430xx, and the second block sits at a single TF setting. A pooled fit with
   a campaign indicator, plus one fit per block, shows whether a B0R0 exponent is a
   field dependence or a campaign label. A campaign offset can be physics (wall,
   fuelling) or diagnostics (EFIT and magnetics era); it does not say which.

5. **Where the campaign offset lives** (``campaign_diagnostics``). Per block:
   - W_mhd / W_e,TS;
   - the two P_OH paths;
   - the diamagnetic flux over its paramagnetic scale I_p^2 / B0R0;
   - with ``--state``, Lane K's p_EFIT/p_e and the EFIT probe reduced chi^2.

   It also fits the block offset of the stored energy itself, W_mhd and the
   Thomson-only W_e, against I_p and the raw ohmic power I_p V_loop. Neither W_e nor
   I_p V_loop uses the EFIT, so a block offset present in W_mhd and absent from W_e
   locates the offset in the magnetics reconstruction. Medians are per slice, so
   shots with many slices weigh more.
6. **Criteria v2** (``block_offset_by_thomson_consistency``, #1521). The same block
   offset split by the Thomson verdict (p within [1, 2] p_e), with and without the
   Thomson density. On the consistent slices, with density in, the offset vanishes;
   it survives on the inconsistent ones, which are 84 % of the 429xx block.
7. **Ohmic regime** (``neo_alcator_regime``). In the linear ohmic confinement
   (LOC) regime, tau_E rises with density (neo-Alcator tau ~ n); in the
   saturated regime (SOC) it does not, so H_NA = tau / tau_NA falls as n^-1.
   - It fits tau against n_e at fixed I_p, with the block offset as a nuisance
     term.
   - It fits H_NA against n_e.
   - Both are done with W_mhd and with the Thomson-only W_e.

Usage::

    python extensions.py --table ~/runs/campaign/atlas/confinement/table.csv \\
        --out ~/runs/campaign/atlas/confinement/extensions
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
import extra_scalings  # noqa: E402,F401
from fit import SELECTIONS, _git, select  # noqa: E402

QE = 1.602176634e-19
#: First shot of the second Tier A block (#1331): 399xx-403xx against 429xx-430xx.
CAMPAIGN_SPLIT_SHOT = 42000


def usable(table: pd.DataFrame, selection: str = "primary") -> pd.DataFrame:
    frame = table.loc[select(table, SELECTIONS[selection])].copy()
    frame = frame.loc[np.isfinite(frame["tau_e_th_s"]) & (frame["tau_e_th_s"] > 0)]
    frame["b0r0_Tm"] = frame["b_t_T"] * frame["r_geo_m"]
    return frame.reset_index(drop=True)


def _complete(frame: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Rows where every used column is finite and positive, so all shot means use the same slices."""
    values = frame[list(columns.values())]
    return frame.loc[(np.isfinite(values) & (values > 0)).all(axis=1)]


def within_shot(frame: pd.DataFrame, columns: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Shot-demeaned logs, returned as exponentials so the power-law fitter can take them.

    A demeaned log is exp()-ed back; the fitter takes its log again, so it sees the
    demeaned values. Its intercept is then ~0 and carries no meaning.
    """
    frame = _complete(frame, columns)
    counts = frame.groupby("shot")["shot"].transform("size")
    f = frame.loc[counts >= 2]
    logs = np.log(f[list(columns.values())])
    demeaned = logs - logs.groupby(f["shot"]).transform("mean")
    names = list(columns)
    y = np.exp(demeaned[columns[names[0]]].to_numpy(float))
    x = {k: np.exp(demeaned[columns[k]].to_numpy(float)) for k in names[1:]}
    return y, x, f["shot"].to_numpy()


def between_shot(frame: pd.DataFrame, columns: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Shot means of the logs, one row per shot; groups are the shots themselves."""
    frame = _complete(frame, columns)
    logs = np.log(frame[list(columns.values())])
    means = logs.groupby(frame["shot"]).mean()
    names = list(columns)
    y = np.exp(means[columns[names[0]]].to_numpy(float))
    x = {k: np.exp(means[columns[k]].to_numpy(float)) for k in names[1:]}
    return y, x, means.index.to_numpy()


def dimensionless_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """<T>, rho*, beta_t, nu*, q_cyl and Omega_i tau_E per row (Thomson n_e required)."""
    from vaft.formula.equilibrium import (
        beta_t_from_n_T_B,
        nu_star_from_n_T_B_R_epsilon_kappa_I,
        omega_i_tau_E_from_B_tau_E_M,
        q_cyl_from_B_R_epsilon_kappa_I,
        rho_star_from_M_T_B_R_epsilon,
    )

    needed = ["n_e_line_avg_m3", "w_th_J", "a_m", "r_geo_m", "kappa_area", "epsilon", "m_eff_amu",
              "b_t_T", "i_p_A", "tau_e_th_s"]
    f = _complete(frame, {c: c for c in needed}).copy()
    volume = 2.0 * np.pi**2 * f["a_m"] ** 2 * f["r_geo_m"] * f["kappa_area"]
    f["t_avg_eV"] = f["w_th_J"] / (3.0 * f["n_e_line_avg_m3"] * volume * QE)
    a = {k: f[k].to_numpy(float) for k in ("m_eff_amu", "t_avg_eV", "b_t_T", "r_geo_m", "epsilon",
                                          "kappa_area", "i_p_A", "n_e_line_avg_m3", "tau_e_th_s")}
    f["rho_star"] = rho_star_from_M_T_B_R_epsilon(a["m_eff_amu"], a["t_avg_eV"], a["b_t_T"], a["r_geo_m"], a["epsilon"])
    f["beta_t"] = beta_t_from_n_T_B(a["n_e_line_avg_m3"], a["t_avg_eV"], a["b_t_T"], output="fraction")
    f["nu_star"] = nu_star_from_n_T_B_R_epsilon_kappa_I(a["n_e_line_avg_m3"], a["t_avg_eV"], a["b_t_T"],
                                                        a["r_geo_m"], a["epsilon"], a["kappa_area"], a["i_p_A"])
    f["q_cyl"] = q_cyl_from_B_R_epsilon_kappa_I(a["b_t_T"], a["r_geo_m"], a["epsilon"], a["kappa_area"], a["i_p_A"])
    f["omega_tau"] = omega_i_tau_E_from_B_tau_E_M(a["b_t_T"], a["tau_e_th_s"], a["m_eff_amu"])
    return f


def block_indicator(frame: pd.DataFrame) -> np.ndarray:
    """exp(1) on the 429xx-430xx block, 1 elsewhere: its log is the block dummy."""
    return np.where((frame["shot"] >= CAMPAIGN_SPLIT_SHOT).to_numpy(), np.e, 1.0)


def campaign_diagnostics(frame: pd.DataFrame, state: pd.DataFrame | None = None,
                         filedb: Path | None = None) -> pd.DataFrame:
    """Per-block medians of the quantities that locate the campaign offset."""
    import gzip

    f = frame.copy()
    f["block"] = np.where(f["shot"] >= CAMPAIGN_SPLIT_SHOT, "429xx-430xx", "399xx-403xx")
    with np.errstate(divide="ignore", invalid="ignore"):
        f["w_mhd_over_w_e_ts"] = f["w_th_J"] / f["w_e_ts_J"].where(f["w_e_ts_J"] > 0)
        f["p_ohm_flux_loop_over_efit"] = f["p_ohm_W"] / f["p_ohm_efit_flux_W"]
    columns = ["w_mhd_over_w_e_ts", "p_ohm_flux_loop_over_efit", "w_th_J", "w_e_ts_J", "i_p_A",
               "p_ohm_W", "n_e_line_avg_m3", "b0r0_Tm"]
    if filedb is not None:
        phi = []
        for _, r in f.iterrows():
            path = filedb / "omas/diagnostics" / str(int(r["shot"])) / "output/diagnostics.json.gz"
            try:
                mag = json.load(gzip.open(path))["magnetics"]
                d = mag["diamagnetic_flux"][0]
                t = np.asarray(d.get("time", mag.get("time")), dtype=float)
                phi.append(float(np.interp(r["time_s"], t, np.asarray(d["data"], dtype=float))))
            except (OSError, KeyError, IndexError, ValueError):
                phi.append(np.nan)
        # Paramagnetic scale mu0^2 I_p^2 / (4 pi B_T) with B_T = B0R0 / R_geo: the ratio
        # is then (1 - beta_p,dia) times a shape factor, so equal block medians mean an
        # unchanged beta_p,dia (and no gross Rogowski gain change), not a direct gain check.
        f["phi_dia_over_paramagnetic_scale"] = np.asarray(phi) / (
            4e-7 * np.pi * 4e-7 * np.pi * f["i_p_A"] ** 2 * f["r_geo_m"] / (4 * np.pi * f["b0r0_Tm"]))
        columns.append("phi_dia_over_paramagnetic_scale")
    out = f.groupby("block")[columns].median().T
    out.insert(0, "quantity", out.index)
    if state is not None:
        # The same slices as above: state rows joined on (shot, time) to the frame.
        st = state.loc[state["efit_lineage"] == "magnetics"].copy()
        st["_key_t"] = st["time_efit_s"].round(4)
        keys = pd.DataFrame({"shot": f["shot"].to_numpy(), "_key_t": f["time_s"].round(4).to_numpy()})
        st = st.merge(keys.drop_duplicates(), on=["shot", "_key_t"], how="inner")
        st["block"] = np.where(st["shot"] >= CAMPAIGN_SPLIT_SHOT, "429xx-430xx", "399xx-403xx")
        extra = st.groupby("block")[["r_sum", "probe_reduced_chi2", "loop_reduced_chi2", "betap"]].median().T
        extra.insert(0, "quantity", ["lane_k_r_sum_p_efit_over_p_e", "efit_probe_reduced_chi2",
                                     "efit_loop_reduced_chi2", "efit_betap"])
        out = pd.concat([out, extra])
    return out.reset_index(drop=True)


def block_offset_by_thomson_consistency(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 429xx block offset of W_mhd and W_e split by the criteria-v2 Thomson verdict (#1521).

    Energies are regressed on I_p and the raw ohmic power I_p V_loop (no EFIT), with
    and without the Thomson line density, on the Thomson-matched rows: all, the
    Thomson-consistent ones (p within [1, 2] p_e) and the inconsistent ones. Returns
    the offset table and a per-block census (rows, good fits, Thomson verdicts).
    """
    from vaft.process.confinement import fit_confinement_scaling as conf_fit

    f = frame.copy()
    f["block"] = np.where(f["shot"] >= CAMPAIGN_SPLIT_SHOT, "429xx-430xx", "399xx-403xx")
    verdict = f["thomson_consistent"].map(
        {True: "consistent", False: "inconsistent", "True": "consistent", "False": "inconsistent",
         "true": "consistent", "false": "inconsistent"}) if "thomson_consistent" in f else pd.Series(np.nan, f.index)
    f["thomson_verdict"] = verdict.fillna("no Thomson")
    census = f.groupby("block").agg(
        rows=("shot", "size"), shots=("shot", "nunique"),
        good=("efit_quality", lambda s: int((s == "good").sum())),
        consistent=("thomson_verdict", lambda s: int((s == "consistent").sum())),
        inconsistent=("thomson_verdict", lambda s: int((s == "inconsistent").sum())),
        n_e_median=("n_e_line_avg_m3", "median")).reset_index()
    ts = f[np.isfinite(f["w_e_ts_J"]) & (f["w_e_ts_J"] > 0)].copy()
    ts["p_loop_raw_W"] = ts["i_p_A"] * ts["v_loop_V"]
    ts = ts[(ts["p_loop_raw_W"] > 0) & np.isfinite(ts["n_e_line_avg_m3"]) & (ts["n_e_line_avg_m3"] > 0)]
    rows = []
    for subset, sub in (("all Thomson rows", ts), ("Thomson-consistent", ts[ts["thomson_verdict"] == "consistent"]),
                        ("Thomson-inconsistent", ts[ts["thomson_verdict"] == "inconsistent"])):
        for energy, column in (("W_mhd", "w_th_J"), ("W_e,TS", "w_e_ts_J")):
            for with_density in (False, True):
                x = {"i_p": sub["i_p_A"].to_numpy(float), "p_loop": sub["p_loop_raw_W"].to_numpy(float)}
                if with_density:
                    x["n_e"] = sub["n_e_line_avg_m3"].to_numpy(float)
                x["campaign2"] = block_indicator(sub)
                row = {"subset": subset, "energy": energy, "with_density": with_density}
                try:
                    fit = conf_fit(sub[column].to_numpy(float), x, sub["shot"].to_numpy())
                    j = fit.names.index("campaign2")
                    row.update(rows_used=fit.n, shots=fit.n_groups, block_offset_log=fit.coef[j],
                               block_offset_se=fit.stderr[j],
                               n_e_exponent=fit.coef[fit.names.index("n_e")] if with_density else np.nan)
                except ValueError as error:
                    row["error"] = str(error)[:80]
                rows.append(row)
    return pd.DataFrame(rows), census


def neo_alcator_regime(frame: pd.DataFrame) -> list:
    """tau and H_NA against n_e, with W_mhd and with the Thomson-only W_e; the block offset is a nuisance."""
    import extra_scalings

    f = frame.loc[np.isfinite(frame["n_e_line_avg_m3"]) & (frame["n_e_line_avg_m3"] > 0)].copy()
    f["tau_e_ts_s"] = f["w_e_ts_J"] / f["p_loss_W"]
    tau_na = extra_scalings.neo_alcator(f)
    rows = []
    for energy, tau_col in (("W_mhd", "tau_e_th_s"), ("W_e,TS", "tau_e_ts_s")):
        tau = f[tau_col].to_numpy(float)
        g = f["shot"].to_numpy()
        base = {"n_e": f["n_e_line_avg_m3"].to_numpy(float), "i_p": f["i_p_A"].to_numpy(float),
                "campaign2": block_indicator(f)}
        rows += run(f"ohmic:{energy}: tau ~ n, I_p, block", tau, base, g,
                    "LOC: a_n ~ 1 (neo-Alcator); SOC: a_n ~ 0; tau uses P_net (dW_mhd/dt, li_3 from EFIT)")
        rows += run(f"ohmic:{energy}: H_NA ~ n, block", tau / tau_na,
                    {"n_e": base["n_e"], "campaign2": base["campaign2"]}, g,
                    "LOC: flat; SOC: a_n ~ -1")
    return rows


def run(name, y, x, groups, note=""):
    from vaft.process.confinement import assess_predictor_identifiability, fit_confinement_scaling

    try:
        fit = fit_confinement_scaling(y, x, groups)
    except ValueError as exc:
        return [{"fit": name, "term": "", "error": str(exc), "note": note}]
    ident = assess_predictor_identifiability({k: np.asarray(v, dtype=float)[fit.mask] for k, v in x.items()})
    rows = []
    for j, term in enumerate(fit.names):
        rows.append({"fit": name, "term": term, "coef": fit.coef[j], "se_cluster": fit.stderr[j],
                     "ci95_lo": fit.ci95[j, 0], "ci95_hi": fit.ci95[j, 1], "n": fit.n, "shots": fit.n_groups,
                     "r2": fit.r2, "rmse_log": fit.rmse_log, "condition_number": ident["condition_number"],
                     "max_vif": max(ident["vif"].values()), "note": note})
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state", help="Lane K state.csv, for its p_EFIT/p_e and EFIT chi^2 by block")
    p.add_argument("--filedb", help="campaign FileDB, for the diamagnetic flux by block")
    args = p.parse_args(argv)
    table_path = Path(args.table).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    frame = usable(pd.read_csv(table_path))

    rows = []
    pooled = {"tau": "tau_e_th_s", "i_p": "i_p_A", "b_t": "b_t_T", "p_net": "p_loss_W"}
    split = {"tau": "tau_e_th_s", "i_p": "i_p_A", "b0r0": "b0r0_Tm", "r_geo": "r_geo_m", "p_net": "p_loss_W"}
    g = frame["shot"].to_numpy()
    for label, cols, note in (
        ("pooled:I,B,P", pooled, "reference: fit A of fit.py"),
        ("pooled:I,B0R0,R,P", split, "B_T split into TF current and plasma position"),
        ("pooled:I,B0R0,P", {k: v for k, v in split.items() if k != "r_geo"},
         "TF current only (R_geo left out)"),
    ):
        rows += run(label, frame[cols["tau"]].to_numpy(float),
                    {k: frame[c].to_numpy(float) for k, c in cols.items() if k != "tau"}, g, note)

    for label, cols, note in (
        ("within:I,R,P", {k: v for k, v in split.items() if k != "b0r0"},
         "shot fixed effects; B0R0 is constant within a shot"),
        ("within:I,P", {k: v for k, v in pooled.items() if k != "b_t"}, "shot fixed effects"),
        ("within:I,B0R0,R,P", split, "B0R0 nearly constant within a shot: expect it unidentified"),
    ):
        y, x, gw = within_shot(frame, cols)
        rows += run(label, y, x, gw, note)

    for label, cols, note in (
        ("between:I,B,P", pooled, "one row per shot (shot means of the logs)"),
        ("between:I,B0R0,R,P", split, "one row per shot"),
    ):
        y, x, gb = between_shot(frame, cols)
        rows += run(label, y, x, gb, note)

    second = (frame["shot"] >= CAMPAIGN_SPLIT_SHOT).to_numpy()
    xc = {k: frame[c].to_numpy(float) for k, c in split.items() if k not in ("tau", "r_geo")}
    # exp(1) on the second block: its log coefficient is the block's log offset.
    rows += run("pooled:I,B0R0,P+campaign", frame["tau_e_th_s"].to_numpy(float),
                {**xc, "campaign2": np.where(second, np.e, 1.0)}, g,
                "campaign2 = log offset of the 429xx-430xx block relative to 399xx-403xx; "
                "physics or diagnostics era")
    for block, mask in (("399xx-403xx", ~second), ("429xx-430xx", second)):
        sub = frame.loc[mask]
        rows += run(f"block {block}:I,B0R0,P", sub["tau_e_th_s"].to_numpy(float),
                    {k: sub[c].to_numpy(float) for k, c in split.items() if k not in ("tau", "r_geo")},
                    sub["shot"].to_numpy(), "one campaign block")

    dim = dimensionless_columns(frame)
    gd = dim["shot"].to_numpy()
    for label, keys, note in (
        ("dimensionless:rho,beta,nu,q", ("rho_star", "beta_t", "nu_star", "q_cyl"),
         "direct; W shared by tau, rho*, beta, nu*"),
        ("dimensionless:rho,beta,nu", ("rho_star", "beta_t", "nu_star"), "direct, q left out"),
        ("dimensionless:rho,q", ("rho_star", "q_cyl"), "direct, rho* and q only"),
    ):
        rows += run(label, dim["omega_tau"].to_numpy(float),
                    {k: dim[k].to_numpy(float) for k in keys}, gd, note)

    campaign_table = campaign_diagnostics(
        frame, pd.read_csv(Path(args.state).expanduser()) if args.state else None,
        Path(args.filedb).expanduser() if args.filedb else None)
    campaign_table.to_csv(out / "campaign_diagnostics.csv", index=False)
    ts = frame.loc[np.isfinite(frame["w_e_ts_J"]) & (frame["w_e_ts_J"] > 0)].copy()
    # EFIT-free power: I_p V_loop from the measured current and flux-loop voltage, with no
    # li_3 correction and no dW_mhd/dt. Responses are energies, not W/P, so the power
    # appears on one side only.
    ts["p_loop_raw_W"] = ts["i_p_A"] * ts["v_loop_V"]
    for energy, col in (("W_mhd", "w_th_J"), ("W_e,TS", "w_e_ts_J")):
        rows += run(f"campaign offset, {energy} ~ I_p, I_p V_loop (TS rows)", ts[col].to_numpy(float),
                    {"i_p": ts["i_p_A"].to_numpy(float), "p_loop": ts["p_loop_raw_W"].to_numpy(float),
                     "campaign2": block_indicator(ts)}, ts["shot"].to_numpy(),
                    "campaign2 = log offset of 429xx-430xx vs 399xx-403xx; W_e,TS and I_p V_loop use no EFIT; "
                    "W_mhd is the magnetics EFIT")
    rows += neo_alcator_regime(frame)
    # W bias is a property of the reconstruction, not of stationarity: diagnose it on
    # every evaluated state key as well as on the primary selection, side by side.
    every = pd.read_csv(table_path)
    every = every[every["quality_status"] == "evaluated"]
    offsets, census = [], []
    for population, data in (("all state keys", every), ("primary selection", frame)):
        o, c = block_offset_by_thomson_consistency(data)
        offsets.append(o.assign(population=population))
        census.append(c.assign(population=population))
    pd.concat(offsets).to_csv(out / "block_offset_criteria_v2.csv", index=False)
    pd.concat(census).to_csv(out / "block_census_criteria_v2.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(out / "coefficients.csv", index=False)
    dim[["shot", "time_s", "t_avg_eV", "rho_star", "beta_t", "nu_star", "q_cyl", "omega_tau"]].to_csv(
        out / "dimensionless_rows.csv", index=False)
    spread = {
        "b0r0_Tm": [float(frame["b0r0_Tm"].min()), float(frame["b0r0_Tm"].max())],
        "r_geo_m": [float(frame["r_geo_m"].min()), float(frame["r_geo_m"].max())],
        "sd_log": {c: float(np.log(frame[c]).std()) for c in ("b_t_T", "b0r0_Tm", "r_geo_m")},
        "corr_log_bt_r": float(np.corrcoef(np.log(frame["b_t_T"]), np.log(frame["r_geo_m"]))[0, 1]),
        "within_shot_sd_log_b0r0": float((np.log(frame["b0r0_Tm"])
                                          - np.log(frame["b0r0_Tm"]).groupby(frame["shot"]).transform("mean")).std()),
        "b0r0_by_block": {b: [float(v.min()), float(v.max())] for b, v in
                          frame.groupby(np.where(second, "429xx-430xx", "399xx-403xx"))["b0r0_Tm"]},
        "dimensionless_ranges": {c: [float(dim[c].min()), float(dim[c].max())]
                                 for c in ("t_avg_eV", "rho_star", "beta_t", "nu_star", "q_cyl")},
    }
    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "command": " ".join(sys.argv),
        "vaft_git": _git("rev-parse", "HEAD"), "vaft_dirty": bool(_git("status", "--porcelain")),
        "table": {"path": str(table_path), "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()},
        "selection": dict(SELECTIONS["primary"]), "spread": spread,
        "notes": ("within: shot fixed effects, CR1 cluster errors (standard for FE nested in clusters); "
                  "between: shot means; dimensionless: <T> = W_mhd/(3 n V e), Ti=Te, n = Thomson line average, "
                  "errors of tau, rho*, beta, nu* share W"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    show = result.loc[result["term"] != "log_C"]
    cols = [c for c in ("fit", "term", "coef", "se_cluster", "n", "shots", "rmse_log", "condition_number", "error")
            if c in show]
    print(show[cols].round(3).to_string(index=False))
    print(json.dumps(spread, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
