"""TGLF saturation-rule sensitivity against the measured ohmic power balance (#1482, #548).

Lane T's sensitivity product runs SAT0-3 x {ES, EM-BPER} on the same local inputs of a
few Tier A states. This script turns those gyro-Bohm fluxes into watts and sets them
beside Lane D's measured P_OH and P_net for the same state keys (shot, EFIT time,
lineage; matched by time, never by index).

- **Units.** TGLF reports Q/Q_GB with Q_GB = n_e T_e c_s (rho_s/a)^2 (``q_gb_W_m2`` of
  the transport atlas). The flux is GACODE's <Q . grad r>, so the energy flow through
  the surface is P(r) = (Q_e + Q_i) dV/dr, r the midplane minor radius.
- **Geometry.** dV/dr comes from the Miller (rmaj, kappa, delta) profiles of the same
  ``input.gacode`` the atlas ran on: V(r) = pi * closed-integral R^2 dZ. Higher shape
  harmonics are not in those files and are ignored. The script refuses a state whose
  file's rmin[-1] disagrees with the atlas ``a_m``.
- **NEO.** The atlas neoclassical flux (configuration independent) is carried as
  ``P_neo_W``.

This is a consistency test, not an identity (#548 section 11):
- P_net is the total heating minus dW/dt. The flow through r carries only what is
  deposited inside r, so a ratio below 1 at an inner surface is expected; a ratio
  above 1 at r/a = 0.8 overpredicts even the total.
- T_i = T_e is assumed by Lane T (``ti_lineage``); P_rad is not subtracted.

Run on the atlas host::

    python tglf_link.py --atlas ~/runs/campaign/atlas --out ~/runs/campaign/atlas/confinement/tglf_link
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATE = ["shot", "time_efit_s", "efit_lineage"]
SURFACE = STATE + ["r_over_a"]
#: Join keys are rounded: the lineages spell EFIT times differently (build_table.py
#: keys them to 1e-4 s), and a solved surface is r/a within 1e-4 of the requested one.
TIME_DIGITS, SURFACE_DIGITS = 4, 3


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["time_efit_s"] = out["time_efit_s"].astype(float).round(TIME_DIGITS)
    if "r_over_a" in out:
        out["r_over_a"] = out["r_over_a"].astype(float).round(SURFACE_DIGITS)
    return out


def miller_volume(rmin, rmaj, kappa, delta, n_theta: int = 721):
    """V(r) and dV/dr [m^3, m^2] of Miller surfaces (no Z shift, no higher harmonics)."""
    if any(v is None for v in (rmin, rmaj, kappa, delta)):
        raise ValueError("Miller geometry needs rmin, rmaj, kappa and delta profiles")
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    rmin, rmaj, kappa, delta = (np.asarray(v, dtype=float) for v in (rmin, rmaj, kappa, delta))
    x = np.arcsin(np.clip(delta, -0.99, 0.99))
    R = rmaj[:, None] + rmin[:, None] * np.cos(theta[None, :] + x[:, None] * np.sin(theta[None, :]))
    Z = kappa[:, None] * rmin[:, None] * np.sin(theta[None, :])
    dZ = np.diff(Z, axis=1)
    R2 = 0.5 * (R[:, 1:] ** 2 + R[:, :-1] ** 2)
    volume = np.abs(np.pi * np.sum(R2 * dZ, axis=1))
    return volume, np.gradient(volume, rmin)


def input_gacode_path(atlas: Path, shot: int, time_s: float, lineage: str) -> Path:
    return atlas / "transport/neo" / str(shot) / lineage / f"{int(round(time_s * 1e3)):05d}" / "neo/input.gacode"


def power_link(sensitivity: pd.DataFrame, transport: pd.DataFrame, confinement: pd.DataFrame,
               geometry) -> pd.DataFrame:
    """One row per (state, r/a, TGLF configuration): P_TGLF, P_neo and the measured powers.

    ``geometry(shot, time_s, lineage)`` returns ``(rmin, rmaj, kappa, delta)`` profiles.
    """
    sensitivity, transport, confinement = _keyed(sensitivity), _keyed(transport), _keyed(confinement)
    geo = transport[SURFACE + ["q_gb_W_m2", "qe_neo_W_m2", "qi_neo_W_m2", "a_m"]].drop_duplicates(SURFACE)
    df = sensitivity.merge(geo, on=SURFACE, how="left", validate="many_to_one")
    if df["q_gb_W_m2"].isna().any():
        missing = df.loc[df["q_gb_W_m2"].isna(), SURFACE].drop_duplicates()
        raise ValueError(f"surfaces without a transport-atlas row: {missing.to_dict('records')}")
    rows = []
    for (shot, time_s, lineage), g in df.groupby(STATE):
        rmin, rmaj, kappa, delta = geometry(shot, time_s, lineage)
        volume, dvdr = miller_volume(rmin, rmaj, kappa, delta)
        a_cut = float(rmin[-1])
        if not np.isclose(a_cut, g["a_m"].iloc[0], rtol=2e-3):
            raise ValueError(f"{shot} {time_s} {lineage}: input.gacode rmin[-1] = {a_cut} m, atlas a_m = {g['a_m'].iloc[0]} m")
        r = g["r_over_a"].to_numpy(float) * a_cut
        vp = np.interp(r, rmin, dvdr)
        rows.append(pd.DataFrame({
            **{k: g[k].to_numpy() for k in SURFACE + ["tglf_config", "sat_rule", "field_model"]},
            "dV_dr_m2": vp,
            "volume_inside_m3": np.interp(r, rmin, volume),
            "P_tglf_W": (g["qe_gb"] + g["qi_gb"]).to_numpy(float) * g["q_gb_W_m2"].to_numpy(float) * vp,
            "P_neo_W": (g["qe_neo_W_m2"] + g["qi_neo_W_m2"]).to_numpy(float) * vp,
        }))
    out = pd.concat(rows, ignore_index=True)
    measured = confinement[["shot", "time_efit_s", "p_ohm_W", "p_net_W", "efit_quality", "thomson_consistent"]]
    # Lane D's table is magnetics-lineage only: an electron_kinetic state takes the
    # power of its magnetics twin (P_OH does not use the EFIT; P_net uses its dW/dt).
    out = out.merge(measured.rename(columns={"efit_quality": "efit_quality_magnetics"}),
                    on=["shot", "time_efit_s"], how="left", validate="many_to_one")
    if out["p_net_W"].isna().any():
        missing = out.loc[out["p_net_W"].isna(), STATE].drop_duplicates()
        raise ValueError(f"states without a measured power in the confinement table: {missing.to_dict('records')}")
    out["ratio_tglf_to_p_net"] = out["P_tglf_W"] / out["p_net_W"]
    out["ratio_tglf_neo_to_p_net"] = (out["P_tglf_W"] + out["P_neo_W"]) / out["p_net_W"]
    return out


def summary(link: pd.DataFrame, r_over_a: float = 0.8) -> pd.DataFrame:
    """Per state at one surface: measured powers, NEO, and the TGLF range over configurations."""
    s = link[np.isclose(link["r_over_a"], r_over_a, rtol=0.0, atol=1e-3)]
    if s.empty:
        raise ValueError(f"no surface at r/a = {r_over_a}")
    return s.groupby(STATE).agg(
        p_ohm_W=("p_ohm_W", "first"), p_net_W=("p_net_W", "first"), P_neo_W=("P_neo_W", "first"),
        P_tglf_min_W=("P_tglf_W", "min"), P_tglf_median_W=("P_tglf_W", "median"), P_tglf_max_W=("P_tglf_W", "max"),
        ratio_min=("ratio_tglf_to_p_net", "min"), ratio_max=("ratio_tglf_to_p_net", "max"),
        thomson_consistent=("thomson_consistent", "first")).reset_index()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlas", type=Path, required=True, help="campaign atlas root (transport/, transport_sensitivity/, confinement/)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    from vaft.code.gacode._input_gacode import read_input_gacode  # the library reader; no public alias yet

    def geometry(shot, time_s, lineage):
        prof = read_input_gacode(input_gacode_path(args.atlas, shot, time_s, lineage))
        return prof.rmin, prof.rmaj, prof.kappa, prof.delta

    link = power_link(pd.read_csv(args.atlas / "transport_sensitivity/sensitivity.csv"),
                      pd.read_csv(args.atlas / "transport/atlas.csv"),
                      pd.read_csv(args.atlas / "confinement/table.csv"), geometry)
    args.out.mkdir(parents=True, exist_ok=True)
    link.to_csv(args.out / "power_link.csv", index=False)
    table = summary(link)
    table.to_csv(args.out / "power_link_summary_r0.8.csv", index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
