"""The three closures together on one slice: resistive Z_eff + impurity amplitude + inferred T_i.

``kinetic_state_stages(ion_temperature="inferred")`` on FileDB products:  Lane Z's Romero scalar closes
the impurity amplitude (C : O = 1 : 1), ``T_i`` is partitioned out of the magnetic-EFIT pressure with
that composition's ``n_i``.  The ``T_i = T_e`` variant of the same slice is drawn for comparison.
Reads the FileDB, writes under ``--out`` only.

    python three_closure_case.py --filedb DB --out DIR 39916:0.322 39916:0.321
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from staged_inference_filedb import _product  # sibling module


def _evaluate(rho, inside, pointwise, modes) -> dict:
    """How each mode's T_i scores: against the pointwise partition, against p_eq, and how much is extrapolated.

    * ``chi2_ti``: mean ``((T_i - T_i,pw)/sigma_pw)^2`` over the significant pointwise points
      (1 = consistent with the partition within its own errors);
    * ``volume_ratio`` / ``median_abs_rel_residual``: ``p_thermal`` against ``p_eq``;
    * ``extrapolated_rho``: fraction of the radius the mode fills with no partition point.
    """
    t_pw, s_pw = pointwise["T_i"], pointwise.get("T_i_pointwise_sigma")
    valid = np.isfinite(t_pw) & (s_pw is not None)
    if s_pw is not None:
        valid &= np.isfinite(s_pw) & (s_pw > 0)
    out = {}
    for name, f in modes.items():
        if f is None:
            out[name] = None
            continue
        rel = np.abs(f["relative_residual"])
        row = {"volume_ratio": f["closure_summary"]["volume_ratio"],
               "median_abs_rel_residual": float(np.nanmedian(rel[inside])),
               "chi2_ti": float(np.mean(((f["T_i"] - t_pw)[valid] / s_pw[valid]) ** 2)) if valid.any() else None}
        prof = f.get("ti_ratio_profile")
        if prof is not None:
            lo, hi = prof["rho_span"]
            row["extrapolated_rho"] = float(lo + max(0.0, 1.0 - hi))
        elif name == "scalar r":
            span = rho[np.isfinite(t_pw)]
            row["extrapolated_rho"] = float(span.min() + max(0.0, 1.0 - span.max())) if span.size else None
        else:
            row["extrapolated_rho"] = 0.0
        out[name] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("slices", nargs="+", metavar="SHOT:TIME")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omas import ODS

    from vaft.database._local import load_ods
    from vaft.omas import vest_upstream as vu
    from vaft.omas.kinetic_state_inference import kinetic_state_stages

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = [(int(a), float(b)) for a, b in (item.split(":") for item in args.slices)]
    fig, axes = plt.subplots(len(wanted), 4, figsize=(17, 3.8 * len(wanted)), squeeze=False)
    summary = []
    for row, (shot, time_s) in enumerate(wanted):
        efit = _product(args.filedb, "efit/magnetic", shot, "efit")
        lineage = vu._magnetic_efit_lineage(efit)  # read from where it was loaded, never asserted
        ods = ODS(consistency_check=False)
        ods["equilibrium"] = load_ods(efit)[0]["equilibrium"]
        ods["core_profiles"] = load_ods(_product(args.filedb, "core_profiles", shot))[0]["core_profiles"]
        ods["thomson_scattering"] = load_ods(_product(args.filedb, "thomson", shot))[0]["thomson_scattering"]
        onset, _ = vu._plasma_onset(_product(args.filedb, "diagnostics", shot))
        times = np.atleast_1d(np.asarray(ods["core_profiles.time"], dtype=float))
        t = float(times[int(np.argmin(np.abs(times - time_s)))])
        age = None if onset is None else t - onset
        kwargs = dict(time=t, ionization="transient", plasma_age_s=age if age and age > 0 else None,
                          equilibrium_lineage=lineage)
        three = kinetic_state_stages(ods, ion_temperature="inferred", **kwargs)
        base = kinetic_state_stages(ods, ion_temperature="assumed", **kwargs)
        ratio_run = kinetic_state_stages(ods, ion_temperature="inferred_ratio", **kwargs)
        profile_run = kinetic_state_stages(ods, ion_temperature="inferred_profile", **kwargs)
        status = {s.name: (s.status if s.ok else f"unavailable: {s.reason[:160]}") for s in three.stages}
        entry = {"shot": shot, "time_s": t, "complete": three.complete, "stages": status}
        summary.append(entry)
        ax = axes[row]
        if not three.complete:
            ax[0].set_title(f"{shot} @ {t*1e3:.0f} ms: unavailable ({[k for k, v in status.items() if v.startswith('unavail')][:1]})")
            continue
        data = three.stage("data").fields
        pc, pb = three.stage("pressure_closure").fields, base.stage("pressure_closure").fields
        pr = ratio_run.stage("pressure_closure")
        pr_f = pr.fields if pr.ok else None
        pp = profile_run.stage("pressure_closure")
        pp_f = pp.fields if pp.ok else None
        entry["evaluation"] = _evaluate(data["rho"], data["inside"], pc, {"assumed (T_i=T_e)": pb, "scalar r": pr_f,
                                                                    "profile r(rho)": pp_f})
        if pp_f is not None:
            prof = pp_f["ti_ratio_profile"]
            entry["profile_fit"] = {"order": prof["order"], "coefficients": prof["coefficients"],
                                    "bic": prof["bic"], "rho_span": prof["rho_span"]}
        else:
            entry["profile_fit"] = pp.reason
        zeff = three.stage("zeff_profile").fields
        rom = three.stage("romero_zeff").fields
        rho, ts = data["rho"], data["thomson"]
        ion = three.stage("ion_density").fields
        ne = three.stage("thomson_fits").fields["n_e"]
        te = three.stage("thomson_fits").fields["T_e"]
        z_res = float(rom["z_eff_resistive"])
        sig = rom.get("z_eff_resistive_sigma")
        entry.update(z_eff_resistive=z_res, z_eff_resistive_sigma=None if sig is None else float(sig),
                     ti_over_te_median=float(np.nanmedian(pc["T_i_over_T_e"])),
                     valid_ti_points=int(np.isfinite(pc["T_i"]).sum()), points=int(rho.size),
                     volume_ratio_if_ti_eq_te=pb["closure_summary"]["volume_ratio"])
        if pr_f is not None:
            est = pr_f["ti_ratio_estimate"]
            entry.update(ratio_estimate={k: float(v) for k, v in est.items()},
                         volume_ratio_ratio_mode=pr_f["closure_summary"]["volume_ratio"],
                         median_rel_residual_ratio_mode=pr_f["closure_summary"]["median_relative_residual"])
        else:
            entry["ratio_estimate"] = pr.reason
        ax[0].plot(rho, pc["p_eq"], "k-", label="$p_{eq}$ (magnetic EFIT)")
        ax[0].plot(rho, pc["p_thermal"], "r-", lw=1, label="$p_e+p_i$, pointwise $T_i$")
        if pr_f is not None:
            ax[0].plot(rho, pr_f["p_thermal"], color="darkorange", lw=2, label="$p_e+p_i$, $T_i=rT_e$")
        if pp_f is not None:
            ax[0].plot(rho, pp_f["p_thermal"], color="purple", lw=2, label="$p_e+p_i$, $T_i=r(\\rho)T_e$")
        ax[0].plot(rho, pc["p_e"], "b--", label="$p_e$")
        ax[0].plot(rho, pb["p_thermal"], color="gray", ls=":", label="$p_e+p_i$, $T_i=T_e$")
        if ts:
            ax[0].plot(ts["rho"], ts["p_e"], "kD", ms=4, label="$p_e^{TS}$")
        ax[0].set(ylabel="p [Pa]", title=f"{shot} @ {t*1e3:.0f} ms")
        ax[1].plot(rho, te, "b-", label="$T_e$")
        ax[1].plot(rho, pc["T_i"], "r-", lw=1, label="$T_i$ pointwise (partition)")
        if pr_f is not None:
            ax[1].plot(rho, pr_f["T_i"], color="darkorange", lw=2,
                       label=f"$T_i=rT_e$, $r={est['ratio']:.2f}\\pm{est['sigma']:.2f}$")
            ax[1].fill_between(rho, (est["ratio"] - est["sigma"]) * te, (est["ratio"] + est["sigma"]) * te,
                               color="darkorange", alpha=0.15)
        if pp_f is not None:
            prof = pp_f["ti_ratio_profile"]
            extra = prof["extrapolated"]
            t_prof = pp_f["T_i"]
            ax[1].plot(rho, np.where(extra, np.nan, t_prof), color="purple", lw=2,
                       label=f"$T_i=r(\\rho)T_e$, order {prof['order']}")
            ax[1].plot(rho, np.where(extra, t_prof, np.nan), color="purple", lw=2, ls="--",
                       label="  extrapolated")
            ax[1].fill_between(rho, (prof["ratio"] - prof["sigma"]) * te, (prof["ratio"] + prof["sigma"]) * te,
                               color="purple", alpha=0.12)
        ax[1].plot(rho, te, color="gray", ls=":", label="$T_i=T_e$")
        if ts:
            ax[1].errorbar(ts["rho"], ts["t_e"], yerr=ts["t_e_error"], fmt="kD", ms=4, label="$T_e^{TS}\\pm1\\sigma$")
        ax[1].set(ylabel="T [eV]")
        ax[2].plot(rho, ne, "b-", label="$n_e$")
        ax[2].plot(rho, ion["n_i_total"], "r--", label="$\\sum n_i$ (Romero-closed)")
        if ts:
            ax[2].errorbar(ts["rho"], ts["n_e"], yerr=ts["n_e_error"], fmt="kD", ms=4, label="$n_e^{TS}\\pm1\\sigma$")
        ax[2].set(ylabel="n [m$^{-3}$]")
        ax[3].plot(rho, zeff["z_eff"], "g-", label="$Z_{eff}(\\rho)$")
        ax[3].axhline(z_res, color="k", ls="--", label=f"$\\langle Z_{{eff}}\\rangle_{{Romero}}={z_res:.2f}$")
        if sig:
            ax[3].axhspan(z_res - float(sig), z_res + float(sig), color="k", alpha=0.1)
        ax[3].set(ylabel="$Z_{eff}$", ylim=(1.0, 2.6))
        for a in ax:
            a.set_xlabel("$\\rho_{tor,N}$"); a.set_xlim(0, 1); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "three_closure_case.png", dpi=140)
    (args.out / "three_closure_case.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
