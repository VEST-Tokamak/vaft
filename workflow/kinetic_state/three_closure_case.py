"""The three closures together on one slice (#1842): resistive Z_eff + impurity amplitude + inferred T_i.

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
        ods = ODS(consistency_check=False)
        ods["equilibrium"] = load_ods(efit)[0]["equilibrium"]
        ods["core_profiles"] = load_ods(_product(args.filedb, "core_profiles", shot))[0]["core_profiles"]
        ods["thomson_scattering"] = load_ods(_product(args.filedb, "thomson", shot))[0]["thomson_scattering"]
        onset, _ = vu._plasma_onset(_product(args.filedb, "diagnostics", shot))
        times = np.atleast_1d(np.asarray(ods["core_profiles.time"], dtype=float))
        t = float(times[int(np.argmin(np.abs(times - time_s)))])
        age = None if onset is None else t - onset
        kwargs = dict(time=t, ionization="transient", plasma_age_s=age if age and age > 0 else None)
        three = kinetic_state_stages(ods, ion_temperature="inferred", **kwargs)
        base = kinetic_state_stages(ods, ion_temperature="assumed", **kwargs)
        status = {s.name: (s.status if s.ok else f"unavailable: {s.reason[:160]}") for s in three.stages}
        entry = {"shot": shot, "time_s": t, "complete": three.complete, "stages": status}
        summary.append(entry)
        ax = axes[row]
        if not three.complete:
            ax[0].set_title(f"{shot} @ {t*1e3:.0f} ms: unavailable ({[k for k, v in status.items() if v.startswith('unavail')][:1]})")
            continue
        data = three.stage("data").fields
        pc, pb = three.stage("pressure_closure").fields, base.stage("pressure_closure").fields
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
        ax[0].plot(rho, pc["p_eq"], "k-", label="$p_{eq}$ (magnetic EFIT)")
        ax[0].plot(rho, pc["p_thermal"], "r-", label="$p_e+p_i$, $T_i$ inferred")
        ax[0].plot(rho, pc["p_e"], "b--", label="$p_e$")
        ax[0].plot(rho, pb["p_thermal"], color="gray", ls=":", label="$p_e+p_i$, $T_i=T_e$")
        if ts:
            ax[0].plot(ts["rho"], ts["p_e"], "kD", ms=4, label="$p_e^{TS}$")
        ax[0].set(ylabel="p [Pa]", title=f"{shot} @ {t*1e3:.0f} ms")
        ax[1].plot(rho, te, "b-", label="$T_e$")
        ax[1].plot(rho, pc["T_i"], "r-", label="$T_i$ inferred (partition)")
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
