"""Project composition Z_eff(rho) onto Lane Z's resistive windows (#1566, Lane L).

For every ``ok`` window of the Lane Z atlas (``atlas/zeff/zeff.csv``), rebuild
Lane Z's own flux-surface states -- same EFIT product, same Lane K
core_profiles, same state construction (``workflow/resistive_zeff/build_zeff.py``
is imported, not copied) -- put a composition Z_eff(psi_N) on each, and run
:func:`vaft.process.zeff_projection.project_window_to_resistive_scalar` with
the window's conductivity model and Coulomb logarithm.  The scalar is then
directly comparable with Lane Z's observed ``Z_eff^res`` (same objective, same
states).  Profiles:

* ``preset_flat``: the VEST preset, flat Z_eff = 2 (must return 2);
* ``coronal_ne_mean`` / ``transient_ne_mean``: OpenADAS charge states,
  n_e-weighted mean Z_eff = 2 (the Lane L default);
* ``transient_fixed``: n_C/n_e = n_O/n_e = 1/86 with transient charge states.

``resistive_closure`` then asks the inverse question: what common impurity
amplitude makes the transient profile reproduce the observed scalar on every
state of the window (#1566, closure of composition and resistance)?  The
residual is reported, never forced.  Writes ``resistive_projection.csv`` and
``resistive_projection.MANIFEST.json`` into ``--out``.

Usage (vestserver, with the lane's shim)::

    python3 workflow/impurity_zeff/project_resistive.py --filedb ~/runs/campaign/filedb \\
        --atlas ~/runs/campaign/atlas/v1 --zeff-atlas ~/runs/campaign/atlas/zeff \\
        --out ~/runs/campaign/atlas/impurity
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = {"C": 1.0, "O": 1.0}


def _lane_z():
    spec = importlib.util.spec_from_file_location("lane_z_build_zeff", ROOT / "workflow/resistive_zeff/build_zeff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _filled(z: np.ndarray) -> tuple[np.ndarray, int]:
    """NaN surfaces (n_e = 0 at the fitted edge) take their nearest defined value; they carry no current."""
    z = np.asarray(z, dtype=float).copy()
    bad = ~np.isfinite(z)
    if bad.all():
        raise ValueError("Z_eff undefined on every surface")
    if bad.any():
        index = np.arange(z.size)
        z[bad] = np.interp(index[bad], index[~bad], z[~bad])
    return np.maximum(z, 1.0), int(bad.sum())


def _onset(filedb: Path, shot: int) -> Optional[float]:
    path = filedb / "omas" / "diagnostics" / str(shot) / "output" / "diagnostics.json.gz"
    if not path.is_file():
        return None
    from vaft.omas import load
    from vaft.omas.plasma_timing import plasma_timing

    try:
        timing = plasma_timing(load(path))
    except Exception:  # noqa: BLE001
        return None
    return timing.onset if timing.found else None


def build(filedb: Path, atlas: Path, zeff_atlas: Path, out: Path) -> dict:
    from vaft.process.zeff_projection import (
        project_window_to_resistive_scalar,
        project_zeff_profile_to_resistive_scalar,
        spitzer_resistive_equivalent_zeff,
        zeff_profile_for_state,
    )

    lane_z = _lane_z()
    windows = [r for r in csv.DictReader(open(zeff_atlas / "zeff.csv")) if r["status"] == "ok"]
    slices = list(csv.DictReader(open(zeff_atlas / "slices.csv")))
    rows = []
    for w in windows:
        shot = int(w["shot"])
        t0, t1 = float(w["t_start_s"]), float(w["t_end_s"])
        model = w["conductivity_model"]
        ln_lambda = w["ln_lambda"] if w["ln_lambda"] == "sauter" else float(w["ln_lambda"])
        times = sorted({float(s["time_efit_s"]) for s in slices
                        if int(s["shot"]) == shot and s["window_t_start_s"] and abs(float(s["window_t_start_s"]) - t0) < 5e-5
                        and s.get("has_state") in ("True", "true", "1") and t0 - 5e-5 <= float(s["time_efit_s"]) <= t1 + 5e-5})
        base = {"shot": shot, "t_start_s": t0, "t_end_s": t1, "window_kind": w["window_kind"],
                "conductivity_model": model, "ln_lambda": w["ln_lambda"],
                "zeff_res_obs": float(w["zeff"]), "zeff_res_obs_uncertainty": w["zeff_uncertainty"],
                "n_states": len(times)}
        try:
            ods = lane_z._load_ods(filedb / w["efit_product"])
            ods["core_profiles"] = lane_z._load_ods(atlas / w["profiles_source"])["core_profiles"]
            states, notes = lane_z._states_for(ods, times)
        except Exception as exc:  # noqa: BLE001
            rows.append({**base, "profile": "all", "status": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if not states:
            rows.append({**base, "profile": "all", "status": "error", "reason": "; ".join(notes) or "no state"})
            continue
        onset = _onset(filedb, shot)
        profiles: dict[str, list] = {"preset_flat": [np.full(np.shape(s.psi_norm), 2.0) for s in states]}
        filled = 0
        for name, ionization, normalization in (("coronal_ne_mean", "coronal", "ne_weighted_mean"),
                                                ("transient_ne_mean", "transient", "ne_weighted_mean"),
                                                ("transient_fixed", "transient", "fixed")):
            if ionization == "transient" and onset is None:
                continue
            built = []
            for s in states:
                age = float(s.time) - onset if onset is not None else None
                r = zeff_profile_for_state(s, WEIGHTS, normalization=normalization, ionization=ionization,
                                           plasma_age_s=age if age and age > 0 else None)
                z, n_bad = _filled(r.zeff)
                filled += n_bad
                built.append(z)
            profiles[name] = built
        for name, zs in profiles.items():
            row = {**base, "profile": name, "states_used": len(states)}
            try:
                p = project_window_to_resistive_scalar(states, zs, model=model, ln_lambda=ln_lambda,
                                                       bounds=(1.0, 8.0), profile_source=name)
                spitzer = [spitzer_resistive_equivalent_zeff(s, z, ln_lambda=ln_lambda) for s, z in zip(states, zs)]
                row.update(status=p.convergence["status"], zeff_res_equiv=p.zeff_equivalent,
                           residual=p.zeff_equivalent - base["zeff_res_obs"],
                           zeff_profile_volume_mean=p.profile_volume_mean,
                           zeff_axis_mean=float(np.mean([z[0] for z in zs])),
                           zeff_spitzer_analytic_mean=float(np.mean(spitzer)))
            except Exception as exc:  # noqa: BLE001
                row.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:200])
            rows.append(row)
        # the inverse: the amplitude that makes the transient profile reproduce the observed scalar
        if onset is not None:
            scales, n_c, zmean = [], [], []
            for s in states:
                age = float(s.time) - onset

                def projection(z, _s=s):
                    zf, _ = _filled(z)
                    return project_zeff_profile_to_resistive_scalar(_s, zf, model=model, ln_lambda=ln_lambda,
                                                                    bounds=(1.0, 20.0)).zeff_equivalent
                try:
                    r = zeff_profile_for_state(s, WEIGHTS, normalization="resistive_closure", ionization="transient",
                                               plasma_age_s=age if age > 0 else None, projection=projection,
                                               resistive_target=base["zeff_res_obs"])
                    scales.append(r.scale)
                    valid = np.isfinite(r.elemental_fractions[:, 0])
                    n_c.append(float(r.elemental_fractions[valid, 0][0]))
                    zmean.append(float(np.nanmean(r.zeff)))
                except Exception as exc:  # noqa: BLE001
                    rows.append({**base, "profile": "resistive_closure", "status": "error",
                                 "reason": f"t={float(s.time):.4f}: {type(exc).__name__}: {exc}"[:200]})
            if scales:
                rows.append({**base, "profile": "resistive_closure", "status": "ok", "states_used": len(scales),
                             "closure_n_C_over_ne": float(np.median(n_c)),
                             "closure_n_C_over_ne_min": float(np.min(n_c)), "closure_n_C_over_ne_max": float(np.max(n_c)),
                             "zeff_profile_volume_mean": float(np.median(zmean))})
        if filled:
            for row in rows:
                if row.get("shot") == shot and row.get("t_start_s") == t0:
                    row["filled_edge_surfaces"] = filled
    out.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("shot", "t_start_s", "t_end_s", "profile"), c))
    with open(out / "resistive_projection.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    manifest = {"producer": "workflow/impurity_zeff/project_resistive.py (Lane L, #1566, log #1569)",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "vaft_git": sha,
                "zeff_atlas": str(zeff_atlas), "atlas": str(atlas), "filedb": str(filedb),
                "windows": len(windows), "rows": len(rows), "weights": WEIGHTS}
    (out / "resistive_projection.MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True, help="Lane K atlas (core_profiles/)")
    parser.add_argument("--zeff-atlas", type=Path, required=True, help="Lane Z atlas (zeff.csv, slices.csv)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    print(json.dumps(build(args.filedb.expanduser(), args.atlas.expanduser(), args.zeff_atlas.expanduser(),
                           args.out.expanduser())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
