"""Lane D Tier A confinement table (#548 sections 1, 2, 5; conference 2026-10-12).

One row per magnetics state key of Lane K's State key contract v1 (#1454): the
#1331 Tier A good + admissible EFIT slices. The paired electron_kinetic state at
the same (shot, time), when it exists, contributes its stored energy as a second
W column, so no state is lost and record_id stays unique.

Power balance (vaft.process.confinement), all on the EFIT time base of the
shot's magnetics product:

    P_OH      = I_p,meas * V_res,  V_res = V_loop,meas - (1/I_p) dW_int/dt
                V_loop,meas: the inboard-midplane flux loop (vest.yaml
                discharge_timing.loop_voltage), averaged over --vloop-window-s;
                W_int = mu0 R0 li_3 I_p^2 / 4 with li_3 from the post-#1477
                update routine and dli_3/dt kept (Romero NF 2010 eq. 24)
    dW/dt     local quadratic fit over --dwdt-window-s of W_mhd(t)
    P_net     P_OH - dW/dt                        -> p_loss_W (as DB5 PLTH)
    P_trans   P_net - P_rad,meas                   NaN: VEST maps no bolometer
    tau_E     W_mhd / P_net                        -> tau_e_th_s

Comparison columns (never the default): P_OH from the EFIT boundary flux
(vaft.omas.formula_wrapper.compute_voltage_consumption) and, with --spitzer,
the Spitzer integral with Z_eff and ln Lambda stated explicitly (#1188).

Slice quality (#548 section 5) is evidence plus a decision. The evidence columns
are always written; the decision uses the working thresholds below, and
threshold_sweep.csv reports how the accepted count moves with each one. None of
the thresholds is adopted from another database.

Time matching is by time (5e-5 s, half the state key's 1e-4 s rounding), never by
slice index. The product sha256 is checked against the state's; a regenerated
product gives quality_status = product_changed and no values.

Usage::

    python build_table.py --state ~/runs/campaign/atlas/v1/state.csv \\
        --filedb ~/runs/campaign/filedb --out ~/runs/campaign/atlas/confinement
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

TIME_TOLERANCE_S = 5e-5
CONTRACT_VERSION = 1
WORKING_THRESHOLDS = {"ip_min": 30e3, "max_dwdt_fraction": 0.5, "max_ip_change_per_tau": 0.05}
SWEEP = {
    "ip_min": [0.0, 30e3, 50e3, 80e3],
    "max_dwdt_fraction": [0.1, 0.2, 0.3, 0.5, 1.0],
    "max_ip_change_per_tau": [0.01, 0.02, 0.05, 0.1, 0.2],
}
EXTENSION_UNITS = {
    "time_efit_s": "s", "efit_lineage": "", "efit_quality": "", "efit_product_sha256": "",
    "ts_status": "", "thomson_consistent": "", "paired_electron_kinetic": "", "ip_efit_A": "A", "dip_dt_A_s": "A/s",
    "v_loop_V": "V", "v_ind_V": "V", "v_res_V": "V", "r_p_ohm": "Ohm", "n_labelled_slices": "-", "w_kin_status": "", "li_3": "-", "beta_normal": "% m T/MA",
    "w_mhd_J": "J", "w_kin_J": "J", "w_e_ts_J": "J", "dwdt_W": "W", "p_ohm_W": "W", "p_net_W": "W",
    "p_rad_W": "W", "p_transport_W": "W", "tau_e_kin_s": "s",
    "p_ohm_efit_flux_W": "W", "v_res_efit_flux_V": "V", "p_ohm_spitzer_W": "W",
    "ip_rate_1_s": "1/s", "ip_change_per_tau": "-", "dwdt_fraction": "-",
    "rule_finite": "", "rule_ip_min": "", "rule_dwdt_fraction": "", "rule_ip_change_per_tau": "",
    "accepted": "", "quality_status": "", "quality_reason": "", "assumptions": "json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(product: Path):
    from omas import load_omas_json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(gzip.open(product).read())
    try:
        return load_omas_json(tmp.name, consistency_check=False)
    finally:
        Path(tmp.name).unlink()


def _git(*args) -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return ""


def _match(times: np.ndarray, t: float):
    if np.size(times) == 0:
        return None
    k = int(np.argmin(np.abs(times - t)))
    return k if abs(times[k] - t) <= TIME_TOLERANCE_S else None


def _window_mean(t, y, at, width):
    out = np.full(np.shape(at), np.nan)
    for k, t0 in enumerate(np.atleast_1d(at)):
        sel = np.abs(t - t0) <= 0.5 * width
        if np.any(sel & np.isfinite(y)):
            out[k] = float(np.nanmean(y[sel]))
    return out


def _li3_beta(efit):
    """li_3 and beta_normal per slice from the post-#1477 update routine, on a scratch copy."""
    import copy

    from vaft.omas.update import update_equilibrium_global_quantities_beta_li

    work = copy.deepcopy(efit)
    n = len(work["equilibrium.time_slice"])
    for k in range(n):
        for name in ("li_3", "beta_normal"):
            path = f"equilibrium.time_slice.{k}.global_quantities.{name}"
            if path in work:
                del work[path]
    update_equilibrium_global_quantities_beta_li(work, time_slice=list(range(n)))
    li, bn = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(n):
        gq = f"equilibrium.time_slice.{k}.global_quantities"
        if f"{gq}.li_3" in work:
            li[k] = float(work[f"{gq}.li_3"])
        if f"{gq}.beta_normal" in work:
            bn[k] = float(work[f"{gq}.beta_normal"])
    return li, bn


def _merged(efit, core_profiles):
    """EFIT ODS with the shot's core_profiles attached (for the TS line density)."""
    if core_profiles is not None and "core_profiles" in core_profiles:
        efit["core_profiles"] = core_profiles["core_profiles"]
    return efit


def _descriptor_rows(ods, effective_mass_amu: float) -> list[dict]:
    """Canonical engineering columns per equilibrium slice, one slice at a time.

    The same quantities as ``vaft.data.public.vest_ods_to_confinement_rows``, but a
    slice whose descriptors fail (or give a negative stored energy, as the
    unreconstructed slices of a product do) yields NaNs instead of rejecting the
    whole product. The z = 0 line density is Lane M's own implementation.
    """
    from vaft.data.public.vest_confinement import _line_average_density_z0
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    b0 = np.ravel(ods["equilibrium.vacuum_toroidal_field.b0"]) if "equilibrium.vacuum_toroidal_field.b0" in ods else np.array([])
    r0 = float(ods["equilibrium.vacuum_toroidal_field.r0"]) if "equilibrium.vacuum_toroidal_field.r0" in ods else np.nan
    n = len(ods["equilibrium.time_slice"])
    rows = []
    for k in range(n):
        row = {"ok": False, "reason": ""}
        try:
            eq = as_equilibrium(ods, time_index=k)
            d = derive_global_descriptors(eq).values

            def value(name):
                item = d.get(name)
                return float(item.value) if item is not None and item.value is not None else np.nan

            r_geo, a, vol = value("major_radius"), value("minor_radius"), value("volume")
            row.update({
                "ip_efit_A": abs(value("ip")), "w_th_J": value("thermal_energy"), "r_geo_m": r_geo, "a_m": a,
                "epsilon": value("inverse_aspect_ratio"), "kappa": value("elongation"),
                "kappa_area": vol / (2.0 * np.pi**2 * a**2 * r_geo),
                "delta": 0.5 * (value("triangularity_upper") + value("triangularity_lower")),
                "b_t_T": abs(float(b0[k]) * r0) / r_geo if b0.size == n else np.nan,
                "n_e_line_avg_m3": _line_average_density_z0(ods, k, eq),
                "w_e_ts_J": _thomson_electron_energy(ods, k, eq),
                "m_eff_amu": float(effective_mass_amu), "ok": True,
            })
        except Exception as exc:  # noqa: BLE001 - recorded per slice
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _thomson_electron_energy(ods, k: int, eq) -> float:
    """W_e = 1.5 int p_e dV of the Thomson fit at the EFIT time, inside the EFIT LCFS.

    The fit's electron pressure is mapped through its own rho_pol_norm
    (psi_N = rho_pol^2) onto the EFIT grid, so no toroidal-flux coordinate is
    shared between the two; NaN without a fit within TIME_TOLERANCE_S. Beyond
    the fit's last grid point p_e is taken as zero, which underestimates W_e
    only for a fit grid that stops short of rho_pol = 1.
    """
    from matplotlib.path import Path as _Path

    if "core_profiles.time" not in ods or eq.lcfs is None:
        return np.nan
    t_eq = float(ods["equilibrium.time"][k])
    j = _match(np.asarray(ods["core_profiles.time"], dtype=float), t_eq)
    if j is None:
        return np.nan
    prof = f"core_profiles.profiles_1d.{j}"
    if f"{prof}.grid.rho_pol_norm" not in ods:
        return np.nan
    if f"{prof}.electrons.pressure" in ods:
        p_e = np.asarray(ods[f"{prof}.electrons.pressure"], dtype=float)
    elif f"{prof}.electrons.density" in ods and f"{prof}.electrons.temperature" in ods:
        # The fit writer stores pressure_thermal only; the pipeline's pressure
        # step adds electrons.pressure. n T e is the same quantity.
        p_e = (np.asarray(ods[f"{prof}.electrons.density"], dtype=float)
               * np.asarray(ods[f"{prof}.electrons.temperature"], dtype=float) * 1.602176634e-19)
    else:
        return np.nan
    psi_n_fit = np.asarray(ods[f"{prof}.grid.rho_pol_norm"], dtype=float) ** 2
    r, z = np.asarray(eq.r, float), np.asarray(eq.z, float)
    psi_n = (np.asarray(eq.psi, float) - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    outline = np.c_[np.asarray(eq.lcfs.r, float), np.asarray(eq.lcfs.z, float)]
    inside = _Path(outline).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(rr.shape)
    order = np.argsort(psi_n_fit)
    pe_grid = np.interp(np.clip(psi_n, 0.0, 1.0), psi_n_fit[order], p_e[order], right=0.0)
    dv = 2.0 * np.pi * rr * (r[1] - r[0]) * (z[1] - z[0])
    return float(1.5 * np.sum(pe_grid * dv * inside))


DEFINITIONS = {
    "b_t_definition": "vacuum field |b0 r0| / r_geo_m, as DB5 BT",
    "n_e_definition": ("z = 0 chord inside the LCFS through the Thomson core_profiles n_e(rho_tor_norm) at "
                       "the EFIT time (vaft.data.public, Lane M); NaN without a TS profile at that time"),
    "w_th_definition": ("W_mhd = 1.5 int p dV of the magnetics EFIT (statistical_891); W_kin (electron_kinetic "
                        "EFIT) in w_kin_J; W_e = 1.5 int p_e dV of the Thomson fit alone in w_e_ts_J"),
    "m_eff_source": "assumed H+ (state key contract v1), not measured",
}


def shot_series(shot: int, filedb: Path, labelled_times: list, args) -> dict:
    """Power balance and evidence of one shot on its magnetics EFIT time base.

    Rates of the EFIT quantities (dW/dt, dli_3/dt) use only the shot's labelled
    (good/admissible) slices: the product's other slices are unreconstructed and
    their stored energy is meaningless (down to -75 kJ on 42985).
    """
    from vaft.omas.discharge_timing import inboard_midplane_loop, loop_voltage
    from vaft.omas.formula_wrapper import compute_voltage_consumption
    from vaft.omas.update import resolve_reference_major_radius
    from vaft.process.confinement import (
        ConfinementPowerBalance,
        confinement_power_balance,
        confinement_slice_evidence,
        resistive_loop_voltage,
        smoothed_time_derivative,
    )

    efit_path = filedb / "omas/efit/magnetic" / str(shot) / "output/efit.json.gz"
    diag = _load(filedb / "omas/diagnostics" / str(shot) / "output/diagnostics.json.gz")
    cp_path = filedb / "omas/core_profiles" / str(shot) / "output/core_profiles.json.gz"
    efit = _merged(_load(efit_path), _load(cp_path) if cp_path.exists() else None)

    t_all = np.asarray(efit["equilibrium.time"], dtype=float)
    keep = np.array([any(abs(t - lt) <= TIME_TOLERANCE_S for lt in labelled_times) for t in t_all])
    idx = np.flatnonzero(keep)
    t = t_all[idx]
    every_slice = _descriptor_rows(efit, args.effective_mass_amu)
    descriptors = [every_slice[i] for i in idx]
    w_mhd = np.array([r.get("w_th_J", np.nan) for r in descriptors], dtype=float)

    # Measured current and its rate at the EFIT times.
    ip_t = np.asarray(diag["magnetics.ip.0.time"], dtype=float)
    ip_y = np.asarray(diag["magnetics.ip.0.data"], dtype=float)
    ip_meas = _window_mean(ip_t, ip_y, t, args.vloop_window_s)
    dip_dt = smoothed_time_derivative(ip_t, ip_y, window_s=args.dip_window_s, at=t)

    # Measured loop voltage at the inboard midplane.
    loop = inboard_midplane_loop(diag)
    if loop is None:
        raise RuntimeError("no inboard flux loop carries a waveform")
    vt, vy, v_source = loop_voltage(diag, loop, prefer_measured=True)
    v_loop = _window_mean(np.asarray(vt, float), np.asarray(vy, float), t, args.vloop_window_s)
    name_path = f"magnetics.flux_loop.{loop}.name"
    loop_name = str(diag[name_path]) if name_path in diag else str(loop)

    li3_all, beta_all = _li3_beta(efit)
    li3, beta_n = li3_all[idx], beta_all[idx]
    r0 = float(resolve_reference_major_radius(efit))

    def rate(y):
        if t.size < 2:
            return np.full(t.size, np.nan)
        return smoothed_time_derivative(t, y, window_s=args.dwdt_window_s, polyorder=args.rate_polyorder)

    split = resistive_loop_voltage(v_loop, ip_meas, dip_dt, li3, r0, dli3_dt=rate(li3))
    # A loop voltage and current of opposite orientation make every P_OH negative;
    # flag it instead of letting the shot fail its rules silently.
    orientation = float(np.nanmedian(np.sign(ip_meas * v_loop))) if t.size else np.nan
    if t.size >= 2:
        pb = confinement_power_balance(t, ip_meas, split.v_res, w_mhd, dwdt_window_s=args.dwdt_window_s,
                                       dwdt_polyorder=args.rate_polyorder)
    else:  # one labelled slice: no rate, so P_OH only
        nan = np.full(t.size, np.nan)
        pb = ConfinementPowerBalance(time=t, p_ohmic=ip_meas * split.v_res, dwdt=nan, p_net=nan,
                                     p_transport=nan, tau_e_net=nan, tau_e_transport=nan)
    ev = confinement_slice_evidence(ip_meas, dip_dt, w_mhd, pb.p_ohmic, pb.dwdt, pb.tau_e_net)

    # Comparison path: the EFIT boundary flux (#652) over the labelled slices; never the default.
    v_res_efit = np.full(t.size, np.nan)
    if t.size >= 2:
        try:
            v_res_efit = np.asarray(compute_voltage_consumption(efit, time_slice=list(idx))[3], dtype=float)
        except Exception:  # noqa: BLE001 - comparison only, recorded as NaN
            pass

    p_spitzer = np.full(t.size, np.nan)
    if args.spitzer:
        from vaft.omas.process_wrapper import compute_ohmic_heating_power_from_core_profiles

        for j, k in enumerate(idx):
            try:
                p_spitzer[j] = float(compute_ohmic_heating_power_from_core_profiles(
                    efit, time_slice=int(k), Z_eff=args.z_eff, ln_Lambda=args.ln_lambda))
            except Exception:  # noqa: BLE001 - comparison only
                pass

    return dict(
        t=t, descriptors=descriptors, w_mhd=w_mhd, ip_meas=ip_meas, dip_dt=dip_dt, v_loop=v_loop,
        v_ind=split.v_ind, v_res=split.v_res, r_p=split.r_p, li3=li3, beta_n=beta_n, r0=r0, pb=pb, ev=ev,
        v_res_efit=v_res_efit, p_spitzer=p_spitzer, loop_name=loop_name, v_source=v_source,
        efit_sha=_sha256(efit_path), orientation=orientation,
    )


def kinetic_energy(shot: int, filedb: Path, kinetic_states: dict, args) -> dict:
    """W of the electron_kinetic product at its own state-key times, keyed by rounded time.

    Only slices that are electron_kinetic states of contract v1 are read: the
    product's other slices are unvetted. A product whose sha256 differs from the
    state's gives NaN with status ``product_changed``.
    """
    if not kinetic_states:
        return {}
    path = filedb / "omas/electron_efit" / str(shot) / "output/electron_efit.json.gz"
    if not path.exists():
        return {t: (np.nan, "product_missing") for t in kinetic_states}
    sha = _sha256(path)
    ods = _load(path)
    times = np.asarray(ods["equilibrium.time"], dtype=float)
    rows = _descriptor_rows(ods, args.effective_mass_amu)
    out = {}
    for t_state, state_sha in kinetic_states.items():
        k = _match(times, t_state)
        if sha != state_sha:
            out[t_state] = (np.nan, "product_changed")
        elif k is None:
            out[t_state] = (np.nan, "time_unmatched")
        else:
            out[t_state] = (rows[k].get("w_th_J", np.nan), "evaluated")
    return out


def build(args) -> int:
    from vaft.data.public.schema import CONFINEMENT_COLUMNS, make_record_id
    from vaft.process.confinement import confinement_exclusion_table, confinement_slice_decision

    filedb = Path(args.filedb).expanduser()
    out = Path(args.out).expanduser()
    (out / "schema").mkdir(parents=True, exist_ok=True)
    (out / "series").mkdir(exist_ok=True)

    with open(Path(args.state).expanduser(), newline="") as fh:
        states = list(csv.DictReader(fh))
    magnetics = [s for s in states if s["efit_lineage"] == "magnetics"]
    kinetic = {}
    for s in states:
        if s["efit_lineage"] == "electron_kinetic":
            kinetic.setdefault(int(s["shot"]), {})[round(float(s["time_efit_s"]), 4)] = s["efit_product_sha256"]
    kinetic_keys = {(shot, t) for shot, times in kinetic.items() for t in times}

    assumptions = {
        "p_ohm": "I_p,meas * (V_loop,meas - (1/I_p) dW_int/dt), W_int = mu0 R0 li_3 I_p^2/4 (Romero 2010 eq. 24)",
        "v_loop": f"inboard-midplane flux loop, mean over {args.vloop_window_s} s",
        "dip_dt": f"local quadratic fit over {args.dip_window_s} s of the measured Ip",
        "dwdt": (f"local degree-{args.rate_polyorder} fit over {args.dwdt_window_s} s of W_mhd on the "
                 "shot's labelled (good/admissible) EFIT slices only; NaN with one labelled slice"),
        "li_3": "vaft.omas.update.update_equilibrium_global_quantities_beta_li after #1477",
        "p_rad": "not measured (no bolometer mapping): NaN; P_transport undefined",
        "m_eff_amu": f"{args.effective_mass_amu} (H+ per state key contract v1; not measured)",
        "external_inductance": "flux between the inboard loop and the LCFS is not removed",
        "tau_e_kin": ("W_kin of the paired electron_kinetic state over the magnetics P_net: its dW/dt is "
                      "dW_mhd/dt, since a single kinetic slice per shot has no rate of its own"),
        "spitzer": (f"Z_eff={args.z_eff}, ln_Lambda={args.ln_lambda} stated explicitly (#1188)"
                    if args.spitzer else "not evaluated"),
    }

    rows, failures = [], []
    shots = sorted({int(s["shot"]) for s in magnetics})
    for shot in shots:
        try:
            labelled = [float(s["time_efit_s"]) for s in magnetics if int(s["shot"]) == shot]
            series = shot_series(shot, filedb, labelled, args)
            w_kin = kinetic_energy(shot, filedb, kinetic.get(shot, {}), args)
        except Exception as exc:  # noqa: BLE001 - recorded per shot
            failures.append({"shot": shot, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc(limit=3)})
            series, w_kin = None, {}
        if series is not None:
            pb, ev = series["pb"], series["ev"]
            pd.DataFrame({
                "time_s": series["t"], "ip_meas_A": series["ip_meas"], "dip_dt_A_s": series["dip_dt"],
                "v_loop_V": series["v_loop"], "v_ind_V": series["v_ind"], "v_res_V": series["v_res"],
                "v_res_efit_flux_V": series["v_res_efit"], "li_3": series["li3"],
                "w_mhd_J": series["w_mhd"], "dwdt_W": pb.dwdt, "p_ohm_W": pb.p_ohmic,
                "p_net_W": pb.p_net, "tau_e_net_s": pb.tau_e_net,
            }).to_csv(out / "series" / f"{shot}.csv", index=False)

        for state in (s for s in magnetics if int(s["shot"]) == shot):
            t_state = float(state["time_efit_s"])
            key = (shot, round(t_state, 4))
            row = {c: np.nan for c in CONFINEMENT_COLUMNS}
            row.update({
                "machine": "VEST", "record_id": make_record_id("VEST", shot, t_state), "shot": shot,
                "time_s": t_state, "regime": "ohmic", "selected": False,
                "source_database": "VAFT Lane D Tier A confinement table (#548)",
                "source_release": f"atlas/confinement v{CONTRACT_VERSION}",
                "source_reference": "VEST Tier A #1331, EFIT statistical_891, state key contract v1 #1454",
                "time_efit_s": t_state, "efit_lineage": "magnetics",
                "efit_quality": state["efit_quality"], "efit_product_sha256": state["efit_product_sha256"],
                "ts_status": state.get("ts_status", ""),
                # Criteria v2 (#1521): Thomson consistency is a verdict beside fit
                # quality (p within [1, 2] p_e), never part of efit_quality.
                "thomson_consistent": {"true": True, "false": False, "1": True, "0": False}.get(
                    str(state.get("thomson_consistent", "")).strip().lower(), np.nan),
                "paired_electron_kinetic": key in kinetic_keys,
                "assumptions": json.dumps(assumptions, sort_keys=True),
            })
            if series is None:
                row.update(quality_status="shot_failed", quality_reason=failures[-1]["error"])
            elif series["efit_sha"] != state["efit_product_sha256"]:
                row.update(quality_status="product_changed",
                           quality_reason="EFIT product sha256 differs from the state's")
            elif (k := _match(series["t"], t_state)) is None:
                row.update(quality_status="time_unmatched",
                           quality_reason=f"no EFIT slice within {TIME_TOLERANCE_S} s")
            else:
                c = series["descriptors"][k]
                pb, ev = series["pb"], series["ev"]
                for col in ("b_t_T", "n_e_line_avg_m3", "r_geo_m", "a_m", "epsilon", "kappa",
                            "kappa_area", "delta", "m_eff_amu"):
                    row[col] = c.get(col, np.nan)
                row.update(DEFINITIONS)
                w_k, w_k_status = w_kin.get(key[1], (np.nan, "no_kinetic_state"))
                p_net = pb.p_net[k]
                row.update({
                    "i_p_A": abs(series["ip_meas"][k]), "ip_efit_A": c.get("ip_efit_A", np.nan),
                    "r_p_ohm": series["r_p"][k], "n_labelled_slices": int(series["t"].size),
                    "w_kin_status": w_k_status,
                    "dip_dt_A_s": series["dip_dt"][k], "v_loop_V": series["v_loop"][k],
                    "v_ind_V": series["v_ind"][k], "v_res_V": series["v_res"][k],
                    "li_3": series["li3"][k], "beta_normal": series["beta_n"][k],
                    "w_mhd_J": series["w_mhd"][k], "w_th_J": series["w_mhd"][k], "w_kin_J": w_k,
                    "w_e_ts_J": c.get("w_e_ts_J", np.nan),
                    "dwdt_W": pb.dwdt[k], "p_ohm_W": pb.p_ohmic[k], "p_net_W": p_net,
                    "p_rad_W": np.nan, "p_transport_W": pb.p_transport[k],
                    "p_loss_W": p_net if p_net > 0 else np.nan,
                    "tau_e_th_s": pb.tau_e_net[k],
                    "tau_e_kin_s": w_k / p_net if np.isfinite(w_k) and p_net > 0 else np.nan,
                    "p_ohm_efit_flux_W": series["ip_meas"][k] * series["v_res_efit"][k],
                    "v_res_efit_flux_V": series["v_res_efit"][k],
                    "p_ohm_spitzer_W": series["p_spitzer"][k],
                    "ip_rate_1_s": ev.ip_rate[k], "ip_change_per_tau": ev.ip_change_per_tau[k],
                    "dwdt_fraction": ev.dwdt_fraction[k], "rule_finite": bool(ev.finite[k]),
                    "quality_status": "evaluated",
                    "quality_reason": ("Ip * V_loop is negative on most labelled slices of this shot: "
                                       "a reversed late-discharge loop voltage or an orientation mismatch"
                                       if series["orientation"] < 0 else ""),
                    "p_loss_definition": ("P_net = P_OH - dW/dt, P_OH = I_p V_res from the measured inboard "
                                          "loop voltage minus the internal inductive voltage; radiation NOT "
                                          "subtracted (as DB5 PLTH); no auxiliary heating"),
                    "tau_e_definition": ("W_mhd / P_net; W_mhd = 1.5 int p dV of the magnetics EFIT; "
                                         "tau_e_kin_s = W_kin / P_net with the same (W_mhd) dW/dt"),
                })
            rows.append(row)

    table = pd.DataFrame(rows)
    extension = list(EXTENSION_UNITS)
    for col in extension:
        if col not in table:
            table[col] = np.nan
    table = table[list(CONFINEMENT_COLUMNS) + [c for c in extension if c not in CONFINEMENT_COLUMNS]]

    # Slice-quality decision at the working thresholds, and the sweep.
    from vaft.process.confinement import ConfinementSliceEvidence

    def _evidence(frame):
        return ConfinementSliceEvidence(
            ip_abs=frame["i_p_A"].to_numpy(float), ip_rate=frame["ip_rate_1_s"].to_numpy(float),
            ip_change_per_tau=frame["ip_change_per_tau"].to_numpy(float),
            dwdt_fraction=frame["dwdt_fraction"].to_numpy(float),
            finite=frame["rule_finite"].astype("boolean").fillna(False).to_numpy(bool),
        )

    ev = _evidence(table)
    rules = confinement_slice_decision(ev, **WORKING_THRESHOLDS)
    for name in ("ip_min", "dwdt_fraction", "ip_change_per_tau"):
        table[f"rule_{name}"] = rules[name]
    table["rule_finite"] = rules["finite"]
    table["accepted"] = rules["accepted"]
    table["selected"] = table["accepted"]

    exclusions = pd.DataFrame(confinement_exclusion_table(rules))
    exclusions.insert(0, "thresholds", json.dumps(WORKING_THRESHOLDS, sort_keys=True))
    sweep = []
    for ip_min, dw, ic in itertools.product(*SWEEP.values()):
        r = confinement_slice_decision(ev, ip_min=ip_min, max_dwdt_fraction=dw, max_ip_change_per_tau=ic)
        acc = r["accepted"]
        sweep.append({"ip_min": ip_min, "max_dwdt_fraction": dw, "max_ip_change_per_tau": ic,
                      "accepted": int(acc.sum()),
                      "accepted_good": int((acc & (table["efit_quality"] == "good").to_numpy()).sum()),
                      "shots": int(table.loc[acc, "shot"].nunique())})

    table.to_csv(out / "table.csv", index=False)
    exclusions.to_csv(out / "exclusions.csv", index=False)
    pd.DataFrame(sweep).to_csv(out / "threshold_sweep.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "VAFT Lane D Tier A confinement table",
        "description": ("Lane M CONFINEMENT_COLUMNS (vaft.data.public.schema; SI, NaN for missing, "
                        "record_id VEST:<shot>:<ms>) followed by Lane D extension columns. One row per "
                        "magnetics state key of contract v1 (#1454)."),
        "type": "object",
        "properties": {c: {"description": f"[{EXTENSION_UNITS.get(c, 'see CONFINEMENT_COLUMNS')}]"}
                       for c in table.columns},
        "required": ["record_id", "shot", "time_efit_s", "efit_lineage", "efit_quality"],
    }
    (out / "schema" / "table.schema.json").write_text(json.dumps(schema, indent=2))

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "vaft_git": _git("rev-parse", "HEAD"),
        "vaft_dirty": bool(_git("status", "--porcelain")),
        "inputs": {"state": {"path": str(Path(args.state).expanduser()),
                             "sha256": _sha256(Path(args.state).expanduser())},
                   "filedb": str(filedb)},
        "parameters": {k: v for k, v in vars(args).items() if k not in ("state", "filedb", "out")},
        "working_thresholds": WORKING_THRESHOLDS,
        "working_thresholds_status": "provisional working values, not adopted; read threshold_sweep.csv",
        "assumptions": assumptions,
        "rows": int(len(table)),
        "quality_status": table["quality_status"].value_counts().to_dict(),
        "accepted": int(table["accepted"].sum()),
        "shots": int(table["shot"].nunique()),
        "shot_failures": len(failures),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("rows", "quality_status", "accepted", "shots",
                                                "shot_failures")}, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--state", required=True, help="Lane K state.csv (contract v1)")
    p.add_argument("--filedb", required=True, help="campaign FileDB root (read only)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--dwdt-window-s", type=float, default=3e-3,
                   help="local-fit window for dW/dt and dli_3/dt (default 3 ms: up to 3 EFIT slices)")
    p.add_argument("--rate-polyorder", type=int, default=1,
                   help="degree of the local fit for dW/dt and dli_3/dt (1: 3 ms holds ~3 slices)")
    p.add_argument("--dip-window-s", type=float, default=1e-3,
                   help="local-quadratic window for dIp/dt of the 40 us current record")
    p.add_argument("--vloop-window-s", type=float, default=5e-4,
                   help="averaging window of the measured loop voltage and current")
    p.add_argument("--effective-mass-amu", type=float, default=1.0)
    p.add_argument("--spitzer", action="store_true", help="also evaluate the Spitzer comparison")
    p.add_argument("--z-eff", type=float, default=2.0, help="Z_eff of the Spitzer comparison")
    p.add_argument("--ln-lambda", type=float, default=17.0, help="ln Lambda of the Spitzer comparison")
    return build(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
