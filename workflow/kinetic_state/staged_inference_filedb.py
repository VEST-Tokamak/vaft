"""Chain A (the corrective pipeline as built) against chain B (Romero first, T_i = T_e) on FileDB products (issue #1842).

For each shot this reads the pipeline-1 magnetic EFIT and the pipeline-2 Thomson,
CES and ``core_profiles`` products from a FileDB and, slice by slice,

* **chain A**: runs the corrective stages exactly as the pipeline does --
  ``build_composition_ods`` (assumed ``C : O = 1 : 1`` at ``<Z_eff> = 2``, transient),
  ``build_ti_inferred_ods`` (``T_i`` inferred from the pressure partition) and
  ``build_resistive_zeff_ods`` (Lane Z's Romero window estimate, as a cross-check);
* **chain B**: runs :func:`vaft.omas.kinetic_state_inference.kinetic_state_stages`
  (the Romero value constrains the impurity amplitude, ``T_i = T_e`` is assumed, the
  pressure is a check).

It writes a JSON record, a CSV with one row per slice and a figure.  Nothing is
written to the FileDB: intermediate products go under ``--work``.

Usage::

    python3 staged_inference_filedb.py --filedb ~/runs/campaign/filedb \\
        --shots 39915 39916 40326 --out ~/work/kp-1842-out --work ~/work/kp-1842-out/tmp
"""

from __future__ import annotations

import argparse
import csv
import json
import time as clock
from pathlib import Path

import numpy as np


def _product(filedb: Path, stage: str, shot: int, name: str | None = None) -> Path:
    stem = name or stage.split("/")[-1]
    directory = filedb / "omas" / stage / str(shot) / "output"
    for suffix in (".json.gz", ".json"):
        if (directory / f"{stem}{suffix}").is_file():
            return directory / f"{stem}{suffix}"
    raise FileNotFoundError(f"{stage} product of shot {shot}: {directory}/{stem}.json[.gz]")


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def evaluate_shot(filedb: Path, shot: int, work: Path, *, max_slices: int | None) -> dict:
    from omas import ODS

    from vaft.database._local import load_ods
    from vaft.omas import save
    from vaft.omas import vest_upstream as vu
    from vaft.omas.kinetic_state_inference import kinetic_state_stages

    efit = _product(filedb, "efit/magnetic", shot, "efit")
    diagnostics = _product(filedb, "diagnostics", shot)
    thomson = _product(filedb, "thomson", shot)
    profiles = _product(filedb, "core_profiles", shot)
    out_dir = work / str(shot)
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"shot": shot, "products": {k: str(v) for k, v in
                                               dict(efit=efit, thomson=thomson, core_profiles=profiles).items()}}

    onset, onset_source = vu._plasma_onset(diagnostics)
    record["plasma_onset_s"], record["onset_source"] = onset, onset_source

    # -- chain A: the pipeline's own builders, on the same products --------------------------
    started = clock.time()
    composed, comp_manifest = vu.build_composition_ods(
        shot=shot, core_profiles_product=profiles, efit_product=efit, diagnostics_product=diagnostics,
        ionization="transient")
    composition_path = out_dir / "composition.json.gz"
    save(composed, composition_path)
    ti_ods, ti_manifest = vu.build_ti_inferred_ods(
        shot=shot, core_profiles_product=composition_path, efit_product=efit)
    ti_path = out_dir / "ti_inferred.json.gz"
    save(ti_ods, ti_path)
    resistive_ods, resistive_manifest = vu.build_resistive_zeff_ods(
        shot=shot, core_profiles_product=ti_path, efit_product=efit)
    record["chain_A"] = {
        "composition": {k: comp_manifest.get(k) for k in ("status", "slices", "error", "skipped_slices")},
        "ti_inferred": {k: ti_manifest.get(k) for k in ("status", "slices", "error", "inference", "skipped_slices")},
        "resistive_zeff": {k: resistive_manifest.get(k) for k in
                           ("status", "error", "window", "states", "inference", "per_slice_diagnostic", "crosscheck")},
        "seconds": clock.time() - started,
    }

    # -- chain B: the staged inference, slice by slice ------------------------------------------
    ods = ODS(consistency_check=False)
    efit_ods, _ = load_ods(efit)
    cp_ods, _ = load_ods(profiles)
    ts_ods, _ = load_ods(thomson)
    ods["equilibrium"] = efit_ods["equilibrium"]
    ods["core_profiles"] = cp_ods["core_profiles"]
    ods["thomson_scattering"] = ts_ods["thomson_scattering"]
    times = np.atleast_1d(np.asarray(ods["core_profiles.time"], dtype=float))
    index = list(range(times.size))
    if max_slices and times.size > max_slices:
        index = sorted({int(round(i)) for i in np.linspace(0, times.size - 1, max_slices)})
    slices = []
    started = clock.time()
    composed_times = np.atleast_1d(np.asarray(composed["core_profiles.time"], dtype=float))
    for j in index:
        t = float(times[j])
        age = None if onset is None else t - onset
        b = kinetic_state_stages(ods, time=t, ionization="transient", plasma_age_s=age if age and age > 0 else None)
        row = {"time_s": t, "plasma_age_s": age, "stages": [s.record() | {"fields": None} for s in b.stages]}
        row["chain_B"] = {
            "complete": b.complete,
            "stage_status": {s.name: (s.status if s.ok else f"unavailable: {s.reason[:140]}") for s in b.stages},
        }
        if b.stage("romero_zeff").fields:
            r = b.stage("romero_zeff").fields
            row["chain_B"]["romero"] = {k: _finite(r.get(k)) if not isinstance(r.get(k), (str, tuple, dict)) else r.get(k)
                                        for k in ("z_eff_resistive", "z_eff_resistive_sigma", "estimate_status", "estimate_value")}
        if b.complete:
            z = b.stage("zeff_profile").fields
            i = b.stage("ion_density").fields
            c = b.stage("pressure_closure").fields
            ne = b.stage("thomson_fits").fields["n_e"]
            rho = b.rho
            core = (rho <= 0.8) & np.isfinite(c["relative_residual"])
            row["chain_B"].update(
                z_eff_axis=_finite(z["z_eff"][0]), closure_error=_finite(z["closure_error"]),
                n_i_over_n_e_axis=_finite(i["n_i_total"][0] / ne[0]),
                pressure=c["closure_summary"],
                median_abs_relative_residual_rho_le_0p8=_finite(np.median(np.abs(c["relative_residual"][core]))) if core.any() else None,
            )
            row["_arrays_B"] = {"rho": rho, "z_eff": z["z_eff"], "n_i_over_n_e": i["n_i_total"] / ne, "T_e": c["T_i"],
                                "p_thermal_over_p_eq": c["p_thermal"] / c["p_eq"]}
        # chain A on the same slice, from the pipeline's own products
        k = int(np.argmin(np.abs(composed_times - t)))
        if abs(composed_times[k] - t) < 5e-4:
            base = f"core_profiles.profiles_1d.{k}"
            from vaft.ods_access import path_value

            ti = path_value(ti_ods, f"{base}.ion.0.temperature", None)
            te = path_value(ti_ods, f"{base}.electrons.temperature", None)
            rho_a = np.asarray(ti_ods[f"{base}.grid.rho_tor_norm"], dtype=float)
            row["chain_A"] = {"has_T_i": ti is not None}
            if ti is not None:
                ratio = np.asarray(ti, dtype=float) / np.asarray(te, dtype=float)
                mask = (rho_a <= 0.8) & np.isfinite(ratio)
                row["chain_A"].update(
                    Ti_over_Te_median_rho_le_0p8=_finite(np.median(ratio[mask])) if mask.any() else None,
                    Ti_over_Te_range_rho_le_0p8=[_finite(ratio[mask].min()), _finite(ratio[mask].max())] if mask.any() else None,
                    valid_points=f"{int(np.isfinite(ti).sum())}/{rho_a.size}")
                row["_arrays_A"] = {"rho": rho_a, "T_i": np.asarray(ti, dtype=float), "T_e": np.asarray(te, dtype=float),
                                    "z_eff": np.asarray(ti_ods[f"{base}.zeff"], dtype=float)}
            z_a = path_value(ti_ods, f"{base}.zeff", None)
            if z_a is not None:
                row["chain_A"]["z_eff_axis"] = _finite(np.asarray(z_a, dtype=float)[0])
        slices.append(row)
    record["slices"] = slices
    record["chain_B_seconds"] = clock.time() - started
    return record


def _figure(records: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(records), figsize=(6.2 * len(records), 11.5), sharex="col", squeeze=False)
    for col, record in enumerate(records):
        rows = [r for r in record["slices"] if "_arrays_B" in r] or record["slices"]
        row = rows[len(rows) // 2]
        title = f"{record['shot']} @ {row['time_s'] * 1e3:.0f} ms"
        a, b = row.get("_arrays_A"), row.get("_arrays_B")
        axes[0, col].set_title(title + ("" if b else "  (chain B unavailable)"))
        if a:
            axes[0, col].plot(a["rho"], a["T_e"], label="T_e (Thomson fit)")
            axes[0, col].plot(a["rho"], a["T_i"], "--", label="T_i  chain A (inferred)")
            axes[1, col].plot(a["rho"], a["z_eff"], "--", label="chain A (<Z_eff> = 2 assumed)")
        if b:
            axes[0, col].plot(b["rho"], b["T_e"], ":", color="k", label="T_i  chain B (= T_e)")
            axes[1, col].plot(b["rho"], b["z_eff"], label="chain B (Romero-constrained)")
            axes[2, col].plot(b["rho"], b["n_i_over_n_e"], label="chain B")
            axes[3, col].plot(b["rho"], b["p_thermal_over_p_eq"], label="chain B  p_thermal / p_eq")
            axes[3, col].axhline(1.0, color="k", ls="--", lw=1.0, label="chain A: closes by construction")
        else:
            reason = next((v for v in row["chain_B"]["stage_status"].values() if v.startswith("unavailable")), "")
            axes[3, col].text(0.02, 0.5, "chain B: " + reason[:160], transform=axes[3, col].transAxes, fontsize=8, wrap=True)
        for r, lab in enumerate(("temperature [eV]", "Z_eff(rho)", "n_i,tot / n_e", "p_thermal / p_eq")):
            axes[r, col].set_ylabel(lab)
            axes[r, col].grid(alpha=0.3)
            if axes[r, col].lines:
                axes[r, col].legend(fontsize=7)
        axes[3, col].set_xlabel("rho_tor_norm")
        axes[3, col].set_ylim(0.0, 1.8)
    fig.suptitle("Chain A (pipeline: assumed <Z_eff>=2, T_i inferred) vs chain B (Romero-constrained, T_i = T_e) on FileDB products", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max-slices", type=int, default=None, help="evaluate at most this many core_profiles slices per shot")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for shot in args.shots:
        try:
            records.append(evaluate_shot(args.filedb.expanduser(), shot, args.work.expanduser(), max_slices=args.max_slices))
        except Exception as error:  # noqa: BLE001 - one shot's absence must not hide the others
            import traceback

            records.append({"shot": shot, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "slices": []})
    plain = []
    for record in records:
        plain.append({**record, "slices": [{k: v for k, v in row.items() if not k.startswith("_")} for row in record["slices"]]})
    (args.out / "staged_inference_filedb.json").write_text(json.dumps(plain, indent=1, default=str) + "\n")
    with open(args.out / "staged_inference_filedb.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot", "time_s", "A_Ti_over_Te_median", "A_valid_points", "A_zeff_axis", "romero_status", "romero_zeff",
                         "B_complete", "B_first_unavailable", "B_zeff_axis", "B_ni_over_ne_axis", "B_volume_ratio",
                         "B_median_abs_rel_residual"])
        for record in records:
            for row in record["slices"]:
                b, a = row.get("chain_B", {}), row.get("chain_A", {})
                first = next((f"{n}: {s}" for n, s in b.get("stage_status", {}).items() if s != "ok"), "")
                writer.writerow([record["shot"], row["time_s"], a.get("Ti_over_Te_median_rho_le_0p8"), a.get("valid_points"),
                                 a.get("z_eff_axis"), b.get("romero", {}).get("estimate_status"), b.get("romero", {}).get("z_eff_resistive"),
                                 b.get("complete"), first[:120], b.get("z_eff_axis"), b.get("n_i_over_n_e_axis"),
                                 (b.get("pressure") or {}).get("volume_ratio"), b.get("median_abs_relative_residual_rho_le_0p8")])
    ok = [r for r in records if r["slices"]]
    if ok:
        _figure(records, args.out / "staged_inference_filedb.png")
    print(json.dumps([{k: r.get(k) for k in ("shot", "error", "plasma_onset_s", "onset_source")} for r in plain], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
