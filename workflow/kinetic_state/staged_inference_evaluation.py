"""Chain A against chain B on the #1839 archive cases (issue #1842): the numbers behind the choice.

Two ways of getting from the equilibrium and Thomson to a kinetic state:

* **chain A**, what the corrective pipeline builds today and what the archive
  holds: an *assumed* composition (``C : O = 1 : 1`` with ``<Z_eff>_{n_e dV} = 2``),
  then ``T_i`` *inferred* out of the equilibrium pressure, so the pressure closes
  by construction wherever ``T_i`` is valid;
* **chain B**, the staged chain of :mod:`vaft.omas.kinetic_state_inference`: the
  Romero ``Z_eff`` is the constraint, ``T_i = T_e`` is assumed, and the pressure is
  only a check.

The Romero stage needs the discharge window of a multi-slice equilibrium, which
the archive does not carry (it keeps one slice's 1-D tables) and which lives in the
production FileDB.  So for what chain B would do *given* a Romero value, this script
sweeps the constraint: for each assumed target ``<Z_eff>`` it builds the
``C : O = 1 : 1`` composition, sets ``T_i = T_e`` and reports the pressure closure.
That is a what-if on the **volume-mean** ``Z_eff``; chain B's real constraint is the
resistive-projection scalar of Lane Z, a different weighting of the same profile, so
the swept ``Z`` is a stand-in and not the Romero value.  Which Romero value the
archive holds (if any) is printed beside it.

Usage::

    python3 workflow/kinetic_state/staged_inference_evaluation.py --out /path/to/dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_to_ods as A  # noqa: E402

CASES = ((39915, "316"), (40326, "322"))
Z_TARGETS = (1.2, 1.5, 2.0, 2.5, 3.0)
WEIGHTS = {"C": 1.0, "O": 1.0}


def _volume_weights(rho: np.ndarray, rho_eq: np.ndarray, volume: np.ndarray) -> np.ndarray:
    edges = np.concatenate(([0.0], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]))
    return np.clip(np.diff(np.interp(edges, rho_eq, volume)), 0.0, None)


def load(shot: int, ms: str) -> dict:
    archive = json.loads((Path(__file__).resolve().parent / "archive_data" / f"vest_{shot}_{ms}ms.json").read_text())
    cp, eq = archive["core_profiles"], archive["magnetic_equilibrium"]
    rho = np.asarray(cp["rho_tor_norm"], dtype=float)
    rho_eq = np.asarray(eq["rho_tor_norm"], dtype=float)
    order = np.argsort(rho_eq)
    return {
        "shot": shot, "time_s": float(archive["time_s"]), "rho": rho,
        "ne": np.asarray(cp["electron_density_m3"], dtype=float),
        "te": np.asarray(cp["electron_temperature_eV"], dtype=float),
        "p_eq": np.interp(rho, rho_eq[order], np.asarray(eq["pressure_Pa"], dtype=float)[order]),
        "dv": _volume_weights(rho, rho_eq[order], np.asarray(eq["volume_m3"], dtype=float)[order]),
        "age": float(archive["impurity_model"]["plasma_age_s"]),
        "romero": archive.get("resistive_zeff_window"),
    }


def composition(case: dict, target: float):
    from vaft.process.impurity import resolve_radial_composition

    return resolve_radial_composition(
        case["te"], case["ne"], case["rho"], WEIGHTS, normalization="ne_weighted_mean", target_zeff=target,
        ionization="transient", plasma_age_s=case["age"], volume_weights=case["dv"], time=case["time_s"],
    )


def charged_densities(case: dict, radial) -> dict[str, np.ndarray]:
    out = {"H+": case["ne"] * np.asarray(radial.main_ion_fraction, dtype=float)}
    for k, element in enumerate(radial.elements):
        out[element] = case["ne"] * radial.elemental_fractions[:, k] * (1.0 - radial.charge_state_fractions[k][:, 0])
    return out


def pressures(case: dict, densities: dict[str, np.ndarray], ti: np.ndarray):
    from vaft.process.kinetic_closure import assemble_pressure

    return assemble_pressure(case["ne"], case["te"], densities, {label: ti for label in densities})


def closure_numbers(case: dict, pressure) -> dict:
    ok = np.isfinite(pressure.p_thermal) & np.isfinite(case["p_eq"]) & (case["dv"] > 0) & (case["rho"] <= 1.0)
    relative = (case["p_eq"] - pressure.p_thermal) / case["p_eq"]
    core = ok & (case["rho"] <= 0.8)
    return {
        "volume_ratio": float(np.sum((pressure.p_thermal * case["dv"])[ok]) / np.sum((case["p_eq"] * case["dv"])[ok])),
        "median_relative_residual_rho_le_0p8": float(np.nanmedian(relative[core])),
        "p_i_over_p_e_median_rho_le_0p8": float(np.nanmedian((pressure.p_i_thermal / pressure.p_e)[core])),
    }


def evaluate() -> dict:
    results = {}
    for shot, ms in CASES:
        case = load(shot, ms)
        ods = A.kinetic_state_ods(shot, "magnetics")
        base = "core_profiles.profiles_1d.0"
        ti_a = np.asarray(ods[f"{base}.ion.0.temperature"], dtype=float)
        a_radial = composition(case, 2.0)
        sweep = {}
        for z in Z_TARGETS:
            radial = composition(case, z)
            densities = charged_densities(case, radial)
            p = pressures(case, densities, case["te"])
            sweep[z] = {
                "radial": radial, "pressure": p, "n_i_over_n_e": sum(densities.values()) / case["ne"],
                "numbers": closure_numbers(case, p),
            }
        valid = np.isfinite(ti_a)
        results[shot] = {
            "case": case, "ti_a": ti_a, "ti_te_a": ti_a / case["te"], "valid_a": valid, "a_radial": a_radial,
            "sweep": sweep,
        }
    return results


def summary(results: dict) -> dict:
    out = {}
    for shot, r in results.items():
        ti_te = r["ti_te_a"][r["valid_a"] & (r["case"]["rho"] <= 0.8)]
        out[str(shot)] = {
            "time_s": r["case"]["time_s"],
            "romero_archive": r["case"]["romero"] if r["case"]["romero"] is not None else "not computed",
            "chain_A": {
                "assumption": "<Z_eff>_{ne dV} = 2 (C:O = 1:1, transient); T_i inferred from the pressure partition",
                "Ti_over_Te_median_rho_le_0p8": float(np.median(ti_te)) if ti_te.size else None,
                "Ti_over_Te_range_rho_le_0p8": [float(ti_te.min()), float(ti_te.max())] if ti_te.size else None,
                "valid_points": f"{int(r['valid_a'].sum())}/{r['valid_a'].size}",
                "pressure_closure": "closes by construction where T_i is valid (p_thermal = p_eq)",
            },
            "chain_B_whatif": {
                str(z): {**s["numbers"], "Z_eff_axis": float(s["radial"].zeff[0]),
                         "n_i_over_n_e_axis": float(s["n_i_over_n_e"][0])}
                for z, s in r["sweep"].items()
            },
        }
    return out


def figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(results), figsize=(6.2 * len(results), 11.5), sharex="col")
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(Z_TARGETS)))
    for col, (shot, r) in enumerate(results.items()):
        case, rho = r["case"], r["case"]["rho"]
        inside = rho <= 1.0
        ax = axes[0, col]
        ax.set_title(f"{shot} @ {case['time_s'] * 1e3:.0f} ms")
        ax.plot(rho[inside], case["te"][inside], label="T_e (Thomson fit)")
        ax.plot(rho[inside], np.where(r["valid_a"], r["ti_a"], np.nan)[inside], "--",
                label="T_i  chain A (inferred)")
        ax.plot([], [], ":", color="k", label="T_i  chain B (= T_e, assumed)")
        ax.set_ylabel("temperature [eV]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1, col]
        for z, colour in zip(Z_TARGETS, colours):
            ax.plot(rho[inside], r["sweep"][z]["radial"].zeff[inside], color=colour, label=f"B what-if <Z_eff> = {z:g}")
        ax.plot(rho[inside], r["a_radial"].zeff[inside], "k--", lw=1.2, label="A (<Z_eff> = 2)")
        ax.set_ylabel("Z_eff(rho)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        ax = axes[2, col]
        for z, colour in zip(Z_TARGETS, colours):
            ax.plot(rho[inside], r["sweep"][z]["n_i_over_n_e"][inside], color=colour, label=f"<Z_eff> = {z:g}")
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_ylabel("n_i,tot / n_e"); ax.grid(alpha=0.3)

        ax = axes[3, col]
        for z, colour in zip(Z_TARGETS, colours):
            p = r["sweep"][z]["pressure"].p_thermal
            ax.plot(rho[inside], (p / case["p_eq"])[inside], color=colour,
                    label=f"B  <Z_eff> = {z:g}  (V-ratio {r['sweep'][z]['numbers']['volume_ratio']:.2f})")
        ax.axhline(1.0, color="k", ls="--", lw=1.2, label="A: p_thermal = p_eq (by construction)")
        ax.set_ylabel("p_thermal / p_eq"); ax.set_xlabel("rho_tor_norm"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.6)
    fig.suptitle("Chain A (assumed <Z_eff>=2, T_i inferred) vs chain B (T_i = T_e), what-if sweep of the Z_eff constraint\n"
                 "swept Z is the volume mean, a stand-in for the Romero R_p-equivalent scalar", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = evaluate()
    report = summary(results)
    (args.out / "staged_inference_evaluation.json").write_text(json.dumps(report, indent=1) + "\n")
    figure(results, args.out / "staged_inference_evaluation.png")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
