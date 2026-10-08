"""Assemble the VEST ordering-atlas tables for #1629 from the #1331 Tier A campaign.

Reads, never writes, the Tier A atlas (``state.csv``, ``core_profiles/``,
``ti_inferred.csv``), the confinement table and the campaign FileDB; writes
``states.csv`` (one row per state: inputs, global and time-history ordering
quantities) and ``profiles.csv`` (one row per state and radial point: profile
ordering quantities) into ``--out``, with a ``manifest.json``.

Every ordering quantity is computed by :mod:`vaft.process.ordering_state`.
This script only *assembles inputs*, and every assembly choice it makes is a
column or a manifest entry, never silent:

* states are the atlas ``magnetics`` rows (criteria v2, #1521); geometry,
  field and current history come from the confinement table, joined on shot
  and EFIT time;
* ``T_e``/``n_e`` are the atlas core-profile fits at the Thomson time nearest
  the EFIT time, within ``--time-tolerance``; their global values are the
  area-weighted means ``int f rho drho / int rho drho`` (``*_profile_mean``);
* ``q`` comes from the EFIT product's own slice at the state time, on
  ``rho_tor_norm``; ``r = rho_tor_norm * a`` (circular approximation, VEST
  ``kappa ~ 1.6``) and ``B = B0 R0 / (R0 + r)`` (vacuum field, outboard
  midplane);
* ``T_i`` is the pressure-partition-inferred value of ``ti_inferred.csv``
  (its ``grid`` rows) **only on states whose Thomson verdict is consistent**
  under criteria v2 (#1521, ``1 <= p/p_e <= 2``). Elsewhere the partition
  attributes ``p - p_e`` to the ions and gives ``T_i/T_e`` of 6 on average and
  up to ~200, which no ohmic VEST plasma carries; those states get
  ``t_i_source = inferred_inconsistent`` and no ion quantities. ``T_i`` is
  never set to ``T_e``;
* ``Z_eff`` is an explicit assumption (``--z-eff``, default 2.0, the VEST
  impurity preset) and ``ln Lambda`` the NRL value at the global ``n_e``,
  ``T_e``;
* the plasma age is the state time minus the onset that
  :func:`vaft.omas.plasma_timing.plasma_timing` finds in the diagnostics ODS;
* ``beta`` (toroidal, a fraction) is the confinement table's ``beta_normal``
  through the Troyon definition :func:`vaft.formula.stability.beta_N_from_beta_a_B0_Ip`
  inverted, with that state's ``a``, ``B0`` and ``I_p``.

Why not ``vaft.database.summary()``: its presets carry neither the Thomson
profile fits nor the EFIT ``q`` profile per slice, and its EFIT is the
production product rather than the #1331 campaign's graded reconstructions;
the campaign atlas is the population #1629 asks for, so it is assembled here
and the choice is recorded in the manifest.

Usage (vestserver, a worktree on PYTHONPATH, never the production checkout)::

    python3 workflow/ordering_atlas/build_states.py \\
        --atlas ~/runs/campaign/atlas/v1 \\
        --confinement ~/runs/campaign/atlas/confinement/table.csv \\
        --filedb ~/runs/campaign/filedb \\
        --out ~/runs/campaign/atlas/ordering
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]


def _load_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _profile_at(core_profiles: dict, time_s: float, tolerance: float):
    """The fitted electron profile nearest ``time_s``, or ``None`` beyond the tolerance."""
    times = np.asarray(core_profiles.get("time") or [], dtype=float)
    if not times.size:
        return None
    index = int(np.argmin(np.abs(times - time_s)))
    if abs(times[index] - time_s) > tolerance:
        return None
    slice_ = core_profiles["profiles_1d"][index]
    rho = np.asarray(slice_["grid"]["rho_tor_norm"], dtype=float)
    electrons = slice_["electrons"]
    return {"time": float(times[index]), "rho": rho,
            "n_e": np.asarray(electrons["density_thermal"], dtype=float),
            "t_e": np.asarray(electrons["temperature"], dtype=float)}


def _area_mean(rho, values) -> float:
    ok = np.isfinite(values) & np.isfinite(rho)
    if ok.sum() < 2:
        return math.nan
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(values[ok] * rho[ok], rho[ok]) / trapz(rho[ok], rho[ok]))


_EFIT_CACHE: dict[Path, dict] = {}


def _q_on(efit_path: Path, time_s: float, rho: np.ndarray, tolerance: float):
    """``|q|`` of the EFIT slice at ``time_s`` interpolated onto ``rho``, or ``None``."""
    if efit_path not in _EFIT_CACHE:
        _EFIT_CACHE.clear()  # one product per shot: keep only the current one
        _EFIT_CACHE[efit_path] = _load_json(efit_path)
    data = _EFIT_CACHE[efit_path]
    slices = (data.get("equilibrium") or {}).get("time_slice") or []
    best, best_dt = None, math.inf
    for slice_ in slices:
        dt = abs(_float(slice_.get("time")) - time_s)
        if dt < best_dt:
            best, best_dt = slice_, dt
    if best is None or best_dt > tolerance:
        return None
    p1 = best.get("profiles_1d") or {}
    q = np.abs(np.asarray(p1.get("q") or [], dtype=float))
    rho_eq = np.asarray(p1.get("rho_tor_norm") or [], dtype=float)
    if q.size < 2 or q.size != rho_eq.size:
        return None
    order = np.argsort(rho_eq)
    out = np.interp(rho, rho_eq[order], q[order], left=np.nan, right=np.nan)
    return out


def _onset(filedb: Path, shot: int):
    from vaft.omas.plasma_timing import plasma_timing

    path = filedb / "omas" / "diagnostics" / str(shot) / "output"
    candidates = sorted(path.glob("diagnostics.json*")) if path.is_dir() else []
    if not candidates:
        return math.nan, "no diagnostics ODS"
    try:
        import tempfile

        from omas import load_omas_json

        # the documented loader, never an ODS assignment (#118: assignment vivifies paths)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(_load_json(candidates[0]), handle)
        try:
            ods = load_omas_json(handle.name, consistency_check=False)
        finally:
            Path(handle.name).unlink()
        timing = plasma_timing(ods)
    except Exception as error:  # a broken ODS is a missing onset, recorded
        return math.nan, f"timing failed: {type(error).__name__}"
    if not timing.found:
        return math.nan, "no plasma window found"
    return float(timing.onset), str(timing.source)


def build(atlas: Path, confinement: Path, filedb: Path, z_eff: float, time_tolerance: float):
    from vaft.formula.equilibrium import coulomb_logarithm_from_n_T
    from vaft.formula.stability import beta_N_from_beta_a_B0_Ip
    from vaft.process.ordering_state import (
        global_ordering_quantities,
        profile_ordering_quantities,
        time_history_ordering_quantities,
    )

    geometry: dict[int, list[dict]] = {}
    for row in _rows(confinement):
        geometry.setdefault(int(row["shot"]), []).append(row)

    def geometry_at(shot: int, time_s: float) -> dict:
        """The confinement row nearest ``time_s`` within the tolerance (match by time, never by index)."""
        rows = geometry.get(shot, [])
        if not rows:
            return {}
        best = min(rows, key=lambda r: abs(_float(r["time_s"]) - time_s))
        return best if abs(_float(best["time_s"]) - time_s) <= time_tolerance else {}
    ti_rows: dict[tuple, list[dict]] = {}
    ti_path = atlas / "ti_inferred.csv"
    if ti_path.exists():
        for row in _rows(ti_path):
            if row.get("efit_lineage") == "magnetics" and row.get("kind", "grid") == "grid":
                ti_rows.setdefault((int(row["shot"]), round(_float(row["time_efit_s"]), 4)), []).append(row)
    onsets: dict[int, tuple] = {}
    states, profiles = [], []
    for row in _rows(atlas / "state.csv"):
        if row["efit_lineage"] != "magnetics":
            continue
        shot, time_s = int(row["shot"]), round(_float(row["time_efit_s"]), 4)
        geo = geometry_at(shot, time_s)
        a, r0, b0 = _float(geo.get("a_m")), _float(geo.get("r_geo_m")), _float(geo.get("b_t_T"))
        record = {"shot": shot, "time_efit_s": time_s, "efit_quality": row["efit_quality"],
                  "thomson_consistent": row.get("thomson_consistent", ""), "a_m": a, "r_geo_m": r0,
                  "b_t_T": b0, "i_p_A": _float(geo.get("i_p_A")), "dip_dt_A_s": _float(geo.get("dip_dt_A_s")),
                  "n_e_line_avg_m3": _float(geo.get("n_e_line_avg_m3")), "geometry_source": "confinement"
                  if geo else "missing"}
        cp_path = atlas / "core_profiles" / f"{shot}.json.gz"
        prof = _profile_at(_load_json(cp_path)["core_profiles"], time_s, time_tolerance) if cp_path.exists() else None
        record["profile_time_s"] = prof["time"] if prof else math.nan
        n_e = _area_mean(prof["rho"], prof["n_e"]) if prof else math.nan
        t_e = _area_mean(prof["rho"], prof["t_e"]) if prof else math.nan
        record.update(n_e_profile_mean_m3=n_e, t_e_profile_mean_eV=t_e, z_eff_assumed=z_eff)
        ln_lambda = float(coulomb_logarithm_from_n_T(n_e, t_e)) if (n_e > 0 and t_e > 0) else math.nan
        record["ln_lambda"] = ln_lambda
        if shot not in onsets:
            onsets[shot] = _onset(filedb, shot)
        onset, onset_source = onsets[shot]
        record.update(onset_s=onset, onset_source=onset_source,
                      plasma_age_s=time_s - onset if math.isfinite(onset) else math.nan)
        common = dict(minor_radius=a, b0=b0, n_e=n_e, t_e=t_e, z_eff=z_eff, ln_lambda=ln_lambda)
        beta_n, ip = _float(geo.get("beta_normal")), record["i_p_A"]
        beta = math.nan
        if all(math.isfinite(x) and x > 0 for x in (beta_n, a, b0)) and math.isfinite(ip) and ip != 0:
            beta = beta_n * abs(ip) / 1e6 / (a * b0) / 100.0  # Troyon: beta[%] = beta_N I_p[MA] / (a B0)
            assert math.isclose(beta_N_from_beta_a_B0_Ip(100.0 * beta, a, b0, abs(ip) / 1e6), beta_n)
        record["beta_t"] = beta
        record.update(global_ordering_quantities(major_radius=r0, beta=beta, **common))
        record.update(time_history_ordering_quantities(plasma_current=record["i_p_A"],
                                                       current_rate=record["dip_dt_A_s"],
                                                       plasma_age=record["plasma_age_s"], **common))
        q = None
        efit_product = row.get("efit_product")
        if prof is not None and efit_product and (filedb / efit_product).exists():
            q = _q_on(filedb / efit_product, time_s, prof["rho"], time_tolerance)
        record["q_source"] = "efit_slice" if q is not None else "missing"
        if prof is not None and all(math.isfinite(x) and x > 0 for x in (a, r0, b0)) and ln_lambda > 0:
            rho = prof["rho"]
            keep = rho > 0
            r = rho[keep] * a
            ti = None
            channels = ti_rows.get((shot, time_s))
            consistent = str(row.get("thomson_consistent", "")).lower() == "true"
            if channels and not consistent:
                record["t_i_source"] = "inferred_inconsistent"
                channels = None
            if channels:
                ti_rho = np.array([_float(c["rho_tor_norm"]) for c in channels])
                ti_val = np.array([_float(c["t_i_ev"]) for c in channels])
                good = np.isfinite(ti_rho) & np.isfinite(ti_val) & (ti_val > 0)
                if good.sum() >= 2:
                    order = np.argsort(ti_rho[good])
                    ti = np.interp(rho[keep], ti_rho[good][order], ti_val[good][order], left=np.nan, right=np.nan)
                    if not np.any(np.isfinite(ti)):
                        ti = None
                    else:
                        record["ti_over_te_median"] = float(np.nanmedian(ti / prof["t_e"][keep]))
            record.setdefault("t_i_source", "inferred" if ti is not None else "missing")
            quantities = profile_ordering_quantities(
                minor_radius_coordinate=r, n_e=prof["n_e"][keep], t_e=prof["t_e"][keep],
                magnetic_field=b0 * r0 / (r0 + r),
                safety_factor=q[keep] if q is not None else np.full(r.shape, np.nan),
                major_radius=r0, z_eff=z_eff, ln_lambda=ln_lambda, t_i=ti)
            for k, rho_k in enumerate(rho[keep]):
                profiles.append({"shot": shot, "time_efit_s": time_s, "efit_quality": row["efit_quality"],
                                 "rho_tor_norm": float(rho_k), "r_m": float(r[k]),
                                 **{name: float(values[k]) for name, values in quantities.items()}})
        else:
            record["t_i_source"] = "missing"
        states.append(record)
    return states, profiles


def _write(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--confinement", type=Path, required=True)
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--z-eff", type=float, default=2.0)
    parser.add_argument("--time-tolerance", type=float, default=1.5e-3)
    args = parser.parse_args(argv)
    states, profiles = build(args.atlas, args.confinement, args.filedb, args.z_eff, args.time_tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "states.csv", states)
    _write(args.out / "profiles.csv", profiles)
    commit = subprocess.run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "vaft_commit": commit,
                "atlas": str(args.atlas), "confinement": str(args.confinement), "filedb": str(args.filedb),
                "z_eff_assumed": args.z_eff, "time_tolerance_s": args.time_tolerance,
                "states": len(states), "profile_rows": len(profiles),
                "ln_lambda": "NRL electron-ion, coulomb_logarithm_from_n_T at the global n_e, T_e",
                "onset": "vaft.omas.plasma_timing.plasma_timing on the diagnostics ODS (H-alpha first, then I_p)",
                "beta": "beta_normal of the confinement table through the Troyon definition",
                "assumptions": ["r = rho_tor_norm * a", "B = B0 R0 / (R0 + r)",
                                "global n_e, T_e = area-weighted profile means", "n_i = n_e",
                                "Z_eff constant (assumed)",
                                "T_i only where pressure-partition inferred AND Thomson-consistent (criteria v2)"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(states)} states, {len(profiles)} profile rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
