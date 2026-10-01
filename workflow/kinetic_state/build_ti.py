"""Infer T_i from the pressure partition on the atlas states (#1426).

Reads ``state.csv`` / ``profiles.csv`` written by ``build_state.py`` and the
campaign FileDB; writes, into the same atlas directory:

* ``ti_inferred.csv``: one row per evaluation point, either a Thomson channel
  (``kind = channel``, measured n_e and T_e, the primary product) or a
  ``core_profiles`` grid point (``kind = grid``, the fitted n_e and T_e;
  ``outside_ts_span`` flags the points the channels do not bracket);
* ``ti_state.csv``: per state key, whether the inference was eligible and how
  much of it survived;
* ``core_profiles/<shot>.json.gz``: an IMAS ``core_profiles`` ODS per shot with
  the inferred-T_i lineage on the grid, one slice per matched magnetics state;
* ``schema/ti_*.schema.json`` and ``ti_summary.json``.

Only ``magnetics`` rows are inferred from.  ``electron_kinetic`` rows are
recorded as refused, because their pressure was fitted with Ti = Te assumed
(#1426 section 6).  The Ti = Te policy itself (#1414) is untouched: this is an
additional lineage, ``ti_lineage = pressure_partition_inferred``.

sigma(p_EFIT) is the spread of the pressure over the shot's other good or
admissible magnetics slices within ``--window`` (default 1 ms) at the same
psi_N, floored at ``--floor`` (default 0.17) of the pressure; each row names
its basis.

Usage::

    python3 workflow/kinetic_state/build_ti.py --filedb ~/runs/campaign/filedb \\
        --atlas ~/runs/campaign/atlas/v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_state import SLICE_TOLERANCE_S, _cell, _git, _load  # noqa: E402

KEY = ("contract_version", "shot", "time_efit_s", "efit_lineage", "efit_quality")

POINT_COLUMNS: dict[str, tuple[str, str]] = {
    **{k: ("string" if k in ("contract_version", "efit_lineage", "efit_quality") else
           "integer" if k == "shot" else "number", "state key (#1454)") for k in KEY},
    "kind": ("string", "channel (Thomson, measured) | grid (core_profiles fit)"),
    "channel": ("integer", "Thomson channel index (kind = channel)"),
    "rho_tor_norm": ("number", "normalized toroidal-flux radius"),
    "psi_norm": ("number", "normalized poloidal flux"),
    "p_eq_pa": ("number", "EFIT pressure [Pa]"),
    "sigma_p_eq_pa": ("number", "EFIT pressure uncertainty [Pa]"),
    "sigma_p_eq_basis": ("string", "ensemble(n=k) | floor"),
    "n_e_m3": ("number", "electron density [m^-3]"),
    "sigma_n_e_m3": ("number", "its uncertainty [m^-3]"),
    "t_e_ev": ("number", "electron temperature [eV]"),
    "sigma_t_e_ev": ("number", "its uncertainty [eV]"),
    "p_e_pa": ("number", "e n_e T_e [Pa]"),
    "p_i_pa": ("number", "p_eq - p_e [Pa]"),
    "sigma_p_i_pa": ("number", "its uncertainty [Pa]"),
    "t_i_ev": ("number", "inferred T_i [eV]; empty where flagged"),
    "sigma_t_i_ev": ("number", "its uncertainty [eV]"),
    "ti_te": ("number", "T_i / T_e"),
    "flags": ("string", "';'-joined: p_i_nonpositive, p_i_not_significant, non_finite_input, outside_ts_span"),
    "origin": ("string", "inferred"),
    "method": ("string", "equilibrium_pressure_partition"),
    "composition_source": ("string", "ion composition closure used"),
}

STATE_COLUMNS: dict[str, tuple[str, str]] = {
    **{k: POINT_COLUMNS[k] for k in KEY},
    "ti_lineage": ("string", "pressure_partition_inferred"),
    "eligible": ("string", "true | false"),
    "reason": ("string", "why not eligible / not evaluated"),
    "channels": ("integer", "Thomson channels evaluated"),
    "channels_with_ti": ("integer", "channels with an unflagged T_i"),
    "grid_points_in_ts_span": ("integer", "grid points inside the channels' rho span"),
    "grid_with_ti_in_ts_span": ("integer", "of those, with an unflagged T_i"),
    "ti_te_channel_median": ("number", "median T_i/T_e over unflagged channels"),
    "sigma_p_eq_basis": ("string", "basis at the channels"),
    "neighbour_times_s": ("string", "';'-joined slice times the ensemble used"),
    "core_profiles_time_s": ("number", "core_profiles slice used for the grid"),
}


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write(path: Path, columns: dict[str, tuple[str, str]], rows: list[dict[str, Any]], title: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object",
              "required": list(KEY),
              "properties": {k: {"type": [t, "null"], "description": d} for k, (t, d) in columns.items()}}
    (path.parent / "schema").mkdir(exist_ok=True)
    (path.parent / "schema" / f"{path.stem}.schema.json").write_text(json.dumps(schema, indent=1) + "\n")


def _flags(result: dict[str, Any], extra: list[list[str]] | None = None) -> list[str]:
    out = []
    for k, flags in enumerate(result["flags"]):
        merged = list(flags) + (extra[k] if extra else [])
        out.append(";".join(merged))
    return out


def build(filedb: Path, atlas: Path, *, floor: float, window_s: float) -> dict[str, Any]:
    from omas import ODS, save_omas_json

    from vaft.validation import kinetic_state as ks
    from vaft.validation.equilibrium import _pressure_on

    state = _read(atlas / "state.csv")
    profiles = _read(atlas / "profiles.csv")
    channels_by_key: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in profiles:
        channels_by_key[tuple(row[k] for k in KEY)].append(row)
    composition = ks.ion_composition()
    fraction = composition["ion_density_per_electron"]
    magnetics_times: dict[int, list[float]] = defaultdict(list)
    for row in state:
        if row["efit_lineage"] == "magnetics" and row["efit_status"] == "valid":
            magnetics_times[int(row["shot"])].append(float(row["time_efit_s"]))

    points: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    cache: dict[tuple[str, int], Any] = {}
    slices_by_shot: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def product(stage: str, shot: int):
        if (stage, shot) not in cache:
            path = filedb / "omas" / stage / str(shot) / "output" / f"{stage.split('/')[0]}.json.gz"
            if stage == "efit/magnetic":
                path = filedb / "omas" / "efit" / "magnetic" / str(shot) / "output" / "efit.json.gz"
            cache[(stage, shot)] = _load(path) if path.is_file() else None
        return cache[(stage, shot)]

    for row in state:
        key = {k: row[k] for k in KEY}
        out = {**key, "ti_lineage": "pressure_partition_inferred"}
        if row["efit_lineage"] != "magnetics":
            refused = ks.infer_ti_pressure_partition([], [], [], composition=composition, sigma_p_eq=0,
                                                     sigma_n_e=0, sigma_t_e=0,
                                                     equilibrium_lineage=row["efit_lineage"])
            states.append({**out, "eligible": "false", "reason": refused["reason"]})
            continue
        if row["ts_status"] != "matched":
            states.append({**out, "eligible": "true", "reason": f"no Thomson match ({row['ts_status']})"})
            continue
        shot, time_s = int(row["shot"]), float(row["time_efit_s"])
        eq = product("efit/magnetic", shot)
        index, _ = ks.slice_at_time(eq, time_s, tolerance_s=SLICE_TOLERANCE_S)
        neighbour_times = [t for t in magnetics_times[shot] if t != time_s and abs(t - time_s) <= window_s + 1e-9]
        neighbour_index = [ks.slice_at_time(eq, t, tolerance_s=SLICE_TOLERANCE_S)[0] for t in neighbour_times]

        def sigma_at(psi_norm, p_eq):
            others = [_pressure_on(eq, j, psi_norm) for j in neighbour_index]
            return ks.pressure_sigma(p_eq, [o for o in others if o is not None], floor=floor)

        # -- channels: measured n_e, T_e; p_eq sampled on this slice by build_state
        chans = sorted(channels_by_key[tuple(row[k] for k in KEY)], key=lambda c: int(c["channel"]))
        psi_c = np.array([_f(c["psi_norm"]) for c in chans])
        p_eq_c = np.array([_f(c["p_efit_pa"]) for c in chans])
        n_c, t_c = np.array([_f(c["n_e_m3"]) for c in chans]), np.array([_f(c["t_e_ev"]) for c in chans])
        sn_c, st_c = np.array([_f(c["n_e_error_m3"]) for c in chans]), np.array([_f(c["t_e_error_ev"]) for c in chans])
        sp_c, basis_c = sigma_at(psi_c, p_eq_c)
        res_c = ks.infer_ti_pressure_partition(p_eq_c, n_c, t_c, composition=composition, sigma_p_eq=sp_c,
                                               sigma_n_e=sn_c, sigma_t_e=st_c, equilibrium_lineage="magnetics")
        for k, (c, flags) in enumerate(zip(chans, _flags(res_c))):
            points.append({**key, "kind": "channel", "channel": int(c["channel"]), "rho_tor_norm": _f(c["rho_tor_norm"]),
                           "psi_norm": psi_c[k], "p_eq_pa": p_eq_c[k], "sigma_p_eq_pa": sp_c[k],
                           "sigma_p_eq_basis": basis_c, "n_e_m3": n_c[k], "sigma_n_e_m3": sn_c[k], "t_e_ev": t_c[k],
                           "sigma_t_e_ev": st_c[k], "p_e_pa": res_c["p_e"][k], "p_i_pa": res_c["p_i"][k],
                           "sigma_p_i_pa": res_c["sigma_p_i"][k], "t_i_ev": res_c["t_i"][k],
                           "sigma_t_i_ev": res_c["sigma_t_i"][k], "ti_te": res_c["ti_te"][k], "flags": flags,
                           "origin": "inferred", "method": "equilibrium_pressure_partition",
                           "composition_source": composition["composition_source"]})
        good_c = np.isfinite(res_c["t_i"])
        summary = {**out, "eligible": "true", "channels": len(chans), "channels_with_ti": int(good_c.sum()),
                   "ti_te_channel_median": float(np.median(res_c["ti_te"][good_c])) if good_c.any() else None,
                   "sigma_p_eq_basis": basis_c, "neighbour_times_s": ";".join(f"{t:g}" for t in neighbour_times)}

        # -- grid: the core_profiles fit at the Thomson time, inside the LCFS
        cp = product("core_profiles", shot)
        coordinate = ks.rho_tor_norm_of(eq, index)
        electron = (ks.core_profiles_electron_pressure(cp, time_s=_f(row["time_ts_s"]),
                                                       tolerance_s=_f(row["ts_tolerance_s"]))
                    if cp is not None else {"available": False, "reason": "no core_profiles product"})
        if not electron["available"] or coordinate["coordinate"] != "rho_tor_norm":
            summary["reason"] = electron.get("reason") or coordinate.get("reason")
            states.append(summary)
            continue
        j, _ = ks.slice_at_time(cp, electron["time_s"], ids="core_profiles", tolerance_s=1e-6)
        root = cp["core_profiles"]["profiles_1d"][j]
        rho = np.asarray(root["grid"]["rho_tor_norm"], dtype=float)
        n_g = np.asarray(root["electrons"].get("density_thermal", root["electrons"].get("density")), dtype=float)
        t_g = np.asarray(root["electrons"]["temperature"], dtype=float)
        order = np.argsort(coordinate["rho_tor_norm"])
        inside = rho <= 1.0
        rho, n_g, t_g = rho[inside], n_g[inside], t_g[inside]
        psi_g = np.interp(rho, coordinate["rho_tor_norm"][order], coordinate["psi_norm"][order])
        p_eq_g = _pressure_on(eq, index, psi_g)
        rel_n = float(np.nanmedian(sn_c / n_c)) if np.isfinite(sn_c / n_c).any() else math.nan
        rel_t = float(np.nanmedian(st_c / t_c)) if np.isfinite(st_c / t_c).any() else math.nan
        sp_g, basis_g = sigma_at(psi_g, p_eq_g)
        res_g = ks.infer_ti_pressure_partition(p_eq_g, n_g, t_g, composition=composition, sigma_p_eq=sp_g,
                                               sigma_n_e=rel_n * n_g, sigma_t_e=rel_t * t_g,
                                               equilibrium_lineage="magnetics")
        rho_c = np.array([_f(c["rho_tor_norm"]) for c in chans])
        lo, hi = (np.nanmin(rho_c), np.nanmax(rho_c)) if np.isfinite(rho_c).any() else (math.nan, math.nan)
        span = (rho >= lo) & (rho <= hi)
        extra = [[] if s else ["outside_ts_span"] for s in span]
        for k, flags in enumerate(_flags(res_g, extra)):
            points.append({**key, "kind": "grid", "rho_tor_norm": rho[k], "psi_norm": psi_g[k], "p_eq_pa": p_eq_g[k],
                           "sigma_p_eq_pa": sp_g[k], "sigma_p_eq_basis": basis_g, "n_e_m3": n_g[k],
                           "sigma_n_e_m3": rel_n * n_g[k], "t_e_ev": t_g[k], "sigma_t_e_ev": rel_t * t_g[k],
                           "p_e_pa": res_g["p_e"][k], "p_i_pa": res_g["p_i"][k], "sigma_p_i_pa": res_g["sigma_p_i"][k],
                           "t_i_ev": res_g["t_i"][k], "sigma_t_i_ev": res_g["sigma_t_i"][k], "ti_te": res_g["ti_te"][k],
                           "flags": flags, "origin": "inferred", "method": "equilibrium_pressure_partition",
                           "composition_source": composition["composition_source"]})
        good_g = np.isfinite(res_g["t_i"])
        summary.update(grid_points_in_ts_span=int(span.sum()), grid_with_ti_in_ts_span=int((good_g & span).sum()),
                       core_profiles_time_s=electron["time_s"])
        states.append(summary)
        slices_by_shot[shot].append({"time_s": time_s, "quality": row["efit_quality"], "rho": rho, "n_e": n_g,
                                     "t_e": t_g, "t_i": res_g["t_i"], "sigma_t_i": res_g["sigma_t_i"],
                                     "profile_time_s": electron["time_s"], "span": (lo, hi)})

    _write(atlas / "ti_inferred.csv", POINT_COLUMNS, points, "VAFT inferred T_i points (#1426, contract #1454)")
    _write(atlas / "ti_state.csv", STATE_COLUMNS, states, "VAFT inferred T_i per state (#1426, contract #1454)")

    # IMAS core_profiles: the inferred lineage, one slice per matched magnetics state
    (atlas / "core_profiles").mkdir(exist_ok=True)
    for shot, slices in sorted(slices_by_shot.items()):
        slices.sort(key=lambda s: s["time_s"])
        ods = ODS(consistency_check=False)
        ods["core_profiles.ids_properties.homogeneous_time"] = 0
        ods["core_profiles.ids_properties.comment"] = (
            "T_i INFERRED from the equilibrium pressure partition (#1426), not measured; magnetics-only EFIT "
            "(statistical_891); NaN where p_i <= 0 or not significant")
        ods["core_profiles.ids_properties.provenance.node.0.path"] = "core_profiles.profiles_1d"
        ods["core_profiles.ids_properties.provenance.node.0.sources"] = [
            f"equilibrium: {filedb}/omas/efit/magnetic/{shot}", f"electrons: {filedb}/omas/core_profiles/{shot}"]
        ods["core_profiles.code.name"] = "vaft.validation.kinetic_state"
        ods["core_profiles.code.commit"] = _git("rev-parse", "HEAD")
        ods["core_profiles.code.parameters"] = json.dumps({
            "origin": "inferred", "method": "equilibrium_pressure_partition", "ti_lineage": "pressure_partition_inferred",
            "assumptions": ["common ion temperature", composition["composition_source"]],
            "sigma_p_eq": {"floor": floor, "window_s": window_s}, "contract_version": ks.CONTRACT_VERSION,
            "slices": [{"time_efit_s": s["time_s"], "efit_quality": s["quality"], "profile_time_s": s["profile_time_s"],
                        "ts_rho_span": list(s["span"])} for s in slices]})
        ods["core_profiles.time"] = np.array([s["time_s"] for s in slices])
        for i, s in enumerate(slices):
            root = f"core_profiles.profiles_1d.{i}"
            ods[f"{root}.time"] = s["time_s"]
            ods[f"{root}.grid.rho_tor_norm"] = s["rho"]
            ods[f"{root}.electrons.density_thermal"] = s["n_e"]
            ods[f"{root}.electrons.temperature"] = s["t_e"]
            for k, species in enumerate(composition["species"]):
                ion = f"{root}.ion.{k}"
                ods[f"{ion}.label"] = species["label"]
                ods[f"{ion}.z_ion"] = species["z_ion"]
                ods[f"{ion}.element.0.z_n"] = species["z_ion"]
                ods[f"{ion}.element.0.a"] = species["a"]
                ods[f"{ion}.density_thermal"] = species["density_per_electron"] * s["n_e"]
                ods[f"{ion}.temperature"] = s["t_i"]
        raw = atlas / "core_profiles" / f"{shot}.json"
        save_omas_json(ods, str(raw))
        with open(raw) as source, gzip.open(f"{raw}.gz", "wt") as target:
            target.write(source.read())
        raw.unlink()

    eligible = [s for s in states if s["eligible"] == "true" and s.get("channels")]
    ratios = [float(p["ti_te"]) for p in points if p["kind"] == "channel" and math.isfinite(_f(p["ti_te"]))]
    summary = {
        "states": len(states), "eligible_with_channels": len(eligible),
        "refused_circular": sum(1 for s in states if s["eligible"] == "false"),
        "channels": sum(s["channels"] for s in eligible),
        "channels_with_ti": sum(s["channels_with_ti"] for s in eligible),
        "channel_flags": {f: sum(1 for p in points if p["kind"] == "channel" and f in p["flags"].split(";"))
                          for f in ks.TI_FLAGS},
        "ti_te_channel": {"n": len(ratios), "median": float(np.median(ratios)) if ratios else None,
                          "q1": float(np.percentile(ratios, 25)) if ratios else None,
                          "q3": float(np.percentile(ratios, 75)) if ratios else None},
        "sigma_p_eq_basis": {b: sum(1 for s in eligible if s["sigma_p_eq_basis"] == b)
                             for b in sorted({s["sigma_p_eq_basis"] for s in eligible})},
        "core_profiles_shots": sorted(slices_by_shot),
        "floor": floor, "window_s": window_s,
    }
    (atlas / "ti_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.17, help="relative sigma(p_EFIT) floor (#874)")
    parser.add_argument("--window", type=float, default=1e-3, help="ensemble window [s]")
    args = parser.parse_args(argv)
    summary = build(args.filedb.expanduser(), args.atlas.expanduser(), floor=args.floor, window_s=args.window)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
