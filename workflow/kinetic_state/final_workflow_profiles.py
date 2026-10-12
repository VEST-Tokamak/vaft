"""The default Pipeline 2 product (``<Z_eff> = 2``, ``T_i = T_e``) on FileDB data, one panel set per slice (#1842).

Runs ``build_composition_ods`` exactly as the pipeline's composition stage does and draws,
for each requested slice, the pressure (EFIT, thermal total, electron), the temperatures,
the densities and ``Z_eff(rho)``.  Reads the FileDB; writes only under ``--out``.

    python final_workflow_profiles.py --filedb DB --out DIR 39916:0.322 39916:0.318
"""

from __future__ import annotations

import argparse
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

    from vaft.database._local import load_ods
    from vaft.ods_access import path_value
    from vaft.omas import save
    from vaft.omas import vest_upstream as vu
    from vaft.validation import kinetic_state as ks

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = [(int(a), float(b)) for a, b in (item.split(":") for item in args.slices)]
    fig, axes = plt.subplots(len(wanted), 4, figsize=(17, 3.6 * len(wanted)), squeeze=False)
    for row, (shot, time_s) in enumerate(wanted):
        efit = _product(args.filedb, "efit/magnetic", shot, "efit")
        diagnostics = _product(args.filedb, "diagnostics", shot)
        profiles = _product(args.filedb, "core_profiles", shot)
        composed, manifest = vu.build_composition_ods(
            shot=shot, core_profiles_product=profiles, efit_product=efit,
            diagnostics_product=diagnostics, ionization="transient")
        save(composed, args.out / f"composition_{shot}.json.gz")
        times = np.asarray(composed["core_profiles.time"], dtype=float)
        index = int(np.argmin(np.abs(times - time_s)))
        base = f"core_profiles.profiles_1d.{index}"
        rho = np.asarray(composed[f"{base}.grid.rho_tor_norm"], dtype=float)
        get = lambda leaf: np.asarray(composed[f"{base}.{leaf}"], dtype=float)  # noqa: E731
        te, ti = get("electrons.temperature"), get("ion.0.temperature")
        ne = get("electrons.density_thermal") if path_value(composed, f"{base}.electrons.density_thermal", None) is not None else get("electrons.density")
        ni = sum(get(f"ion.{k}.density") * float(composed[f"{base}.ion.{k}.z_ion"]) ** 0 for k in range(len(composed[f"{base}.ion"])))
        ptot, pe = get("pressure_thermal"), get("electrons.pressure")

        solution, _ = load_ods(efit)
        eq_ms = np.asarray(solution["equilibrium.time"], dtype=float) * 1e3
        near = int(np.argmin(np.abs(eq_ms - times[index] * 1e3)))
        electron = ks.core_profiles_electron_pressure(composed, time_s=float(times[index]), tolerance_s=5e-4)
        placed = ks.core_profiles_grid_on_slice(ks.rho_tor_norm_of(solution, near), electron)
        p_eq = None
        if placed["available"]:
            p_eq = vu._equilibrium_pressure_on(solution, near, np.asarray(placed["psi_norm"], dtype=float))
            p_rho = np.asarray(electron["rho_tor_norm"], dtype=float)

        label = f"{shot} @ {times[index] * 1e3:.0f} ms"
        ax = axes[row]
        if p_eq is not None:
            ax[0].plot(p_rho, p_eq, "k-", label="$p_{eq}$ (magnetic EFIT)")
        ax[0].plot(rho, ptot, "r-", label="$p_{th}=p_e+p_i$, $T_i=T_e$")
        ax[0].plot(rho, pe, "b--", label="$p_e$")
        ax[0].set(ylabel="p [Pa]", title=label)
        ax[1].plot(rho, te, "b-", label="$T_e$ (Thomson fit)")
        ax[1].plot(rho, ti, "r--", label="$T_i = T_e$ (assumed)")
        ax[1].set(ylabel="T [eV]")
        ax[2].plot(rho, ne, "b-", label="$n_e$")
        ax[2].plot(rho, ni, "r--", label="$\\sum n_i$ (diluted)")
        ax[2].set(ylabel="n [m$^{-3}$]")
        ax[3].plot(rho, get("zeff"), "g-", label="$Z_{eff}(\\rho)$")
        ax[3].axhline(2.0, color="k", ls="--", label="$\\langle Z_{eff}\\rangle=2$ (assumed)")
        ax[3].set(ylabel="$Z_{eff}$", ylim=(1.0, 2.6))
        for a in ax:
            a.set_xlabel("$\\rho_{tor,N}$")
            a.set_xlim(0, 1)
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "final_workflow_profiles.png", dpi=140)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
