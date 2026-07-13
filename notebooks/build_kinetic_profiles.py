"""Build VEST kinetic profiles (TS + IDS/CES) into ODS/IMAS.

Per (shot, time):
  - equilibrium   : magnetic-EFIT gfile -> eq.to_omas()          (pre-mapping)
  - thomson_scattering / charge_exchange IDSs from MAT files     (pre-mapping)
  - psi_N mapping + profile fits (ne, Te from TS; Ti, Vtor from IDS/CES)
  - core_profiles : fitted ne/Te/Ti/Vtor + pressure_thermal      (post-mapping)
    on the equilibrium rho_tor_norm/psi grid, p = e*ne*(Te+Ti), n_i ~= n_e

Outputs (per slice, under --outdir):
  ods_<shot>_<time>ms.json   OMAS JSON (whole ODS: raw diagnostics + core_profiles)
  ods_<shot>_<time>ms.nc     IMAS netCDF export (skipped with a warning if the
                             IMAS backend is unavailable)
  summary.csv                ne0/Te0/Ti0/Vtor0/p_th0 per slice

Scope ends at the kinetic profiles — no EFIT constraint export, no remote DB.
Defaults: shots 48224/48226/48233 x 299/300/301 ms (298 excluded as artifact).
Run with the conda `vaft` interpreter:
  /opt/anaconda3/envs/vaft/bin/python notebooks/build_kinetic_profiles.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vaft
from vaft.process import profile as pp

vaft.apply_omfit_compat_patches()
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from omas import save_omas_json

DATA_ROOT = REPO_ROOT / "vaft" / "data"
DEFAULT_SHOTS = [48224, 48226, 48233]
DEFAULT_TIMES = [299, 300, 301]  # 298 ms = artifact (excluded)


def _gfile(shot: int, time_ms: int) -> Path:
    cand = sorted((DATA_ROOT / "efit_scaled_f" / str(shot)).glob(f"g*.{time_ms:05d}"))
    if not cand:
        raise FileNotFoundError(f"No gfile for shot {shot} @ {time_ms} ms")
    return cand[0]


def _resolve_source(shot: int, source: str) -> tuple[str, str | None]:
    """Return (charge_exchange options, mat_file) for the ion diagnostic."""
    if source == "ids" or (source == "auto" and (DATA_ROOT / f"IDS_{shot}.mat").exists()):
        return "ids", None  # IDS_{shot}.mat resolved by the builder
    ces_mat = DATA_ROOT / f"CES_{shot}.mat"
    if ces_mat.exists():
        return "ces", str(ces_mat)
    raise FileNotFoundError(f"No IDS_{shot}.mat or CES_{shot}.mat for shot {shot}")


def build(shot: int, time_ms: int, outdir: Path, args) -> dict:
    gfile = _gfile(shot, time_ms)
    eq = OMFITgeqdsk(str(gfile))
    eq["fluxSurfaces"].load()
    ods = eq.to_omas()
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    vaft.machine_mapping.dataset_description(
        ods,
        source=shot,
        options={"source_type": "shot",
                 "description": f"VEST kinetic profiles {shot}@{time_ms}ms (TS+ion Doppler)"},
    )

    # --- electrons: Thomson scattering (NeTe_<shot>.mat, native 7-channel schema) ---
    vaft.machine_mapping.thomson_scattering(ods, shot)
    rho_ts = pp.equilibrium_mapping_thomson_scattering(ods, eq)
    ne_f, te_f, *_ = pp.profile_fitting_thomson_scattering(
        ods, time_ms=time_ms, mapped_rho_position=rho_ts,
        Te_order=2, Ne_order=2, rho_points=100,
        fitting_function_te=args.te_mode, fitting_function_ne=args.ne_mode,
    )

    # --- ions: IDS or CES into the charge_exchange IDS ---
    source, ce_mat = _resolve_source(shot, args.source)
    vaft.machine_mapping.charge_exchange(ods, shotnumber=shot, options=source, mat_file=ce_mat)
    rho_ce = pp.equilibrium_mapping_charge_exchange(ods, eq)
    vtor_f, ti_f, *_ = pp.profile_fitting_charge_exchange(
        ods, time_ms=time_ms, mapped_rho_position=rho_ce,
        Ti_order=2, Vtor_order=2, rho_points=100,
        fitting_function_ti=args.ti_mode, fitting_function_vtor=args.vtor_mode,
        ion_index=0,
    )

    # --- post-mapping kinetic profiles on the equilibrium grid ---
    pp.core_profiles(
        ods, time_ms, rho_ts, ne_f, te_f,
        T_i_function=ti_f, V_tor_function=vtor_f, ti_mapped_rho_position=rho_ce,
    )

    out_json = outdir / f"ods_{shot}_{time_ms}ms.json"
    save_omas_json(ods, str(out_json))

    out_nc = outdir / f"ods_{shot}_{time_ms}ms.nc"
    try:
        if out_nc.exists():
            out_nc.unlink()
        # a plain *.nc path URI selects the imas-python netCDF backend
        vaft.imas.save_omas_imas(ods, uri=str(out_nc), new=True, verbose=False)
    except Exception as exc:
        print(f"[WARN] IMAS netCDF export skipped for {shot}@{time_ms}ms: {exc}")

    cp = "core_profiles.profiles_1d.0"
    ne0 = float(ods[f"{cp}.electrons.density"][0])
    te0 = float(ods[f"{cp}.electrons.temperature"][0])
    ti0 = float(ods[f"{cp}.ion.0.temperature"][0])
    vt0 = float(ods[f"{cp}.ion.0.velocity.toroidal"][0])
    pth0 = float(ods[f"{cp}.pressure_thermal"][0])
    return {
        "shot": shot, "time_ms": time_ms, "source": source, "gfile": gfile.name,
        "json": out_json.name, "nc": out_nc.name if out_nc.exists() else "",
        "ne0_m^-3": ne0, "Te0_eV": te0, "Ti0_eV": ti0,
        "Vtor0_m/s": vt0, "p_th0_Pa": pth0,
        "n_grid": int(np.asarray(ods[f"{cp}.grid.rho_tor_norm"]).size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build VEST kinetic profiles (TS + IDS/CES) into ODS/IMAS")
    ap.add_argument("--shots", nargs="*", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--times", nargs="*", type=int, default=DEFAULT_TIMES)
    ap.add_argument("--source", choices=["auto", "ids", "ces"], default="auto",
                    help="ion diagnostic: IDS_<shot>.mat or CES_<shot>.mat (auto = prefer IDS)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "notebooks" / "kinetic_profiles"))
    ap.add_argument("--te-mode", default="polynomial")
    ap.add_argument("--ne-mode", default="free_exponential",
                    help="density fit; free_exponential gives a small finite edge value "
                         "(decay is NOT enforced — check the fit if the edge rises)")
    ap.add_argument("--ti-mode", default="polynomial")
    ap.add_argument("--vtor-mode", default="polynomial")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for shot in args.shots:
        for t in args.times:
            try:
                meta = build(shot, t, outdir, args)
                rows.append(meta)
                print(f"[OK] {shot} @ {t} ms ({meta['source']}) -> {meta['json']}  "
                      f"(ne0={meta['ne0_m^-3']:.2e}, Te0={meta['Te0_eV']:.0f} eV, "
                      f"Ti0={meta['Ti0_eV']:.0f} eV, p_th0={meta['p_th0_Pa']:.0f} Pa)")
            except Exception as exc:
                import traceback
                print(f"[ERR] {shot}@{t}: {exc}")
                traceback.print_exc()

    if rows:
        with open(outdir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[DONE] {len(rows)} kinetic-profile slices -> {outdir}")


if __name__ == "__main__":
    main()
