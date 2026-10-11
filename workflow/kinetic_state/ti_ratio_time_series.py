"""T_i/T_e of the three-closure ``inferred_profile`` mode against time, over every slice of a shot (#1842).

For each ``core_profiles`` slice: ``kinetic_state_stages(ion_temperature="inferred_profile")`` (Romero-closed
composition, T_i/T_e(rho) by BIC) and the ``T_i = T_e`` variant.  Draws the ratio (axis and volume-weighted
mean, with its band), the chosen order, the pressure closure of both variants against ``p_eq`` and the
Tier A annotation.  Reads the FileDB, writes under ``--out`` only.

    python ti_ratio_time_series.py --filedb DB --out DIR --tier-a TABLE 39916
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from staged_inference_filedb import _product, load_tier_a, tier_a_row  # sibling module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier-a", type=Path, default=None)
    parser.add_argument("shots", nargs="+", type=int)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omas import ODS

    from vaft.database._local import load_ods
    from vaft.omas import vest_upstream as vu
    from vaft.omas.kinetic_state_inference import kinetic_state_stages

    tier_a = load_tier_a(args.tier_a)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for shot in args.shots:
        ods = ODS(consistency_check=False)
        ods["equilibrium"] = load_ods(_product(args.filedb, "efit/magnetic", shot, "efit"))[0]["equilibrium"]
        ods["core_profiles"] = load_ods(_product(args.filedb, "core_profiles", shot))[0]["core_profiles"]
        ods["thomson_scattering"] = load_ods(_product(args.filedb, "thomson", shot))[0]["thomson_scattering"]
        onset, _ = vu._plasma_onset(_product(args.filedb, "diagnostics", shot))
        for t in np.atleast_1d(np.asarray(ods["core_profiles.time"], dtype=float)):
            t = float(t)
            age = None if onset is None else t - onset
            kwargs = dict(time=t, ionization="transient", plasma_age_s=age if age and age > 0 else None)
            prof = kinetic_state_stages(ods, ion_temperature="inferred_profile", **kwargs)
            base = kinetic_state_stages(ods, ion_temperature="assumed", **kwargs)
            tier = tier_a_row(tier_a, shot, t)
            row = {"shot": shot, "time_s": t,
                   "tier_a": None if tier is None else {k: tier[k] for k in ("efit_quality", "thomson_consistent", "accepted")}}
            failed = next((s for s in prof.stages if not s.ok), None)
            row["status"] = "ok" if failed is None else f"{failed.name}: {failed.reason[:150]}"
            romero = prof.stage("romero_zeff").fields if prof.stage("romero_zeff").ok else None
            if romero:
                row["z_eff_resistive"] = float(romero["z_eff_resistive"])
            if base.stage("pressure_closure").ok:
                row["volume_ratio_ti_eq_te"] = base.stage("pressure_closure").fields["closure_summary"]["volume_ratio"]
            if failed is None:
                f = prof.stage("pressure_closure").fields
                fit = f["ti_ratio_profile"]
                rho = prof.stage("data").fields["rho"]
                te = prof.stage("thomson_fits").fields["T_e"]
                ne = prof.stage("thomson_fits").fields["n_e"]
                w = ne * te * np.gradient(rho**2)  # electron-energy weight on the profile grid
                ok = np.isfinite(fit["ratio"]) & np.isfinite(w)
                row.update(
                    order=fit["order"], coefficients=fit["coefficients"], rho_span=fit["rho_span"],
                    ratio_axis=float(fit["ratio"][0]), sigma_axis=float(fit["sigma"][0]),
                    ratio_mean=float(np.sum((fit["ratio"] * w)[ok]) / np.sum(w[ok])),
                    sigma_mean=float(np.sum((fit["sigma"] * w)[ok]) / np.sum(w[ok])),
                    volume_ratio_profile=f["closure_summary"]["volume_ratio"],
                    te_axis=float(te[0]), ne_axis=float(ne[0]),
                    valid_points=int(np.isfinite(f["T_i_pointwise"]).sum()), points=int(rho.size),
                )
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("shot", "time_s", "status", "order", "ratio_mean", "sigma_mean",
                                                       "volume_ratio_profile", "volume_ratio_ti_eq_te", "tier_a")}, default=str))
    (args.out / "ti_ratio_time_series.json").write_text(json.dumps(rows, indent=1, default=str))

    fig, axes = plt.subplots(3, len(args.shots), figsize=(7.5 * len(args.shots), 9.5), sharex="col", squeeze=False)
    for col, shot in enumerate(args.shots):
        mine = [r for r in rows if r["shot"] == shot]
        good = [r for r in mine if r["status"] == "ok"]
        t_ms = np.array([r["time_s"] * 1e3 for r in good])
        ax = axes[:, col]
        if good:
            r_mean = np.array([r["ratio_mean"] for r in good]); s_mean = np.array([r["sigma_mean"] for r in good])
            consistent = np.array([bool(r["tier_a"]) and r["tier_a"]["thomson_consistent"] == "True" for r in good])
            ax[0].fill_between(t_ms, r_mean - s_mean, r_mean + s_mean, color="purple", alpha=0.15)
            ax[0].plot(t_ms, r_mean, "-", color="purple", lw=1)
            ax[0].plot(t_ms[consistent], r_mean[consistent], "o", color="purple", label="Tier A, Thomson-consistent")
            ax[0].plot(t_ms[~consistent], r_mean[~consistent], "o", mfc="white", color="purple", label="other slices")
            ax[0].plot(t_ms, [r["ratio_axis"] for r in good], "x", color="gray", label="on axis")
            for r, x in zip(good, t_ms):
                if r["order"]:
                    ax[0].annotate(f"order {r['order']}", (x, r["ratio_mean"]), fontsize=7, xytext=(3, 6),
                                   textcoords="offset points")
            ax[1].plot(t_ms, [r["volume_ratio_profile"] for r in good], "o-", color="purple", label="$T_i=r(\\rho)T_e$")
            ax[1].plot(t_ms, [r.get("volume_ratio_ti_eq_te") for r in good], "s--", color="gray", label="$T_i=T_e$")
            ax[2].plot(t_ms, [r["te_axis"] for r in good], "b-o", label="$T_e(0)$ [eV]")
            ax2 = ax[2].twinx()
            ax2.plot(t_ms, [r["ne_axis"] / 1e19 for r in good], "g-s", label="$n_e(0)$ [$10^{19}$ m$^{-3}$]")
            ax2.set_ylabel("$n_e(0)$ [$10^{19}$ m$^{-3}$]", color="g")
        missing = [r for r in mine if r["status"] != "ok"]
        for r in missing:
            ax[0].axvline(r["time_s"] * 1e3, color="red", alpha=0.2, lw=4)
        z = next((r.get("z_eff_resistive") for r in mine if r.get("z_eff_resistive")), None)
        ax[0].axhline(1.0, color="k", ls=":", lw=1, label="$T_i=T_e$")
        ax[0].set(ylabel="$\\langle T_i/T_e\\rangle_{n_eT_e}$",
                  title=f"{shot}: three closures (Romero $Z_{{eff}}$={z:.2f})" if z else f"{shot}")
        ax[1].axhline(1.0, color="k", ls=":", lw=1)
        ax[1].set(ylabel="$\\int p_{th}dV / \\int p_{eq}dV$")
        ax[2].set(ylabel="$T_e(0)$ [eV]", xlabel="t [ms]")
        for a in ax:
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); ax[2].legend(fontsize=7, loc="upper left")
    fig.text(0.01, 0.005, "red bands: slices where a stage is unavailable (see JSON)", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "ti_ratio_time_series.png", dpi=140)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
