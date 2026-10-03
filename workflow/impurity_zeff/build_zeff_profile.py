"""Build the impurity Z_eff(rho) atlas (#1565 Sec. 8, Lane L) on the #1331 Tier A states.

Reads, never writes, the campaign FileDB and the Lane K state table (contract
#1454); writes into ``--out``:

* ``zeff_profile.csv`` -- one row per (state, model, rho): Z_eff, S1, S2,
  Z_I,eff, dilution, n_s/n_e, <Z>_s, <Z^2>_s, the coronal <Z>_s and
  relaxation time, and whether the plasma is old enough to be coronal there;
* ``states.csv`` -- one row per (state, model): the scalars a reader compares
  (Z_eff on axis / n_e-weighted / at rho = 0.9, Z_I,eff on axis, n_C/n_e, the
  fraction of points that are not coronal), or the reason a state has none;
* ``MANIFEST.json`` -- code version, inputs and their hashes, ADF11 files,
  arguments and the column dictionary.

The state key is ``(shot, time_efit_s, efit_lineage, efit_quality)`` (#1454 v1);
only ``magnetics`` rows graded ``good`` or ``admissible`` are used.  T_e and
n_e are the FileDB ``core_profiles`` slice nearest the EFIT time within
``--tolerance`` -- the slice the transport resolver (#1428) pairs -- matched by
time, never by index.  The n_e-weighted mean uses the EFIT volume of that
time; the plasma age is the EFIT time minus the plasma onset of
:func:`vaft.omas.plasma_timing.plasma_timing` on the diagnostics product.

Models: ``coronal`` and ``transient`` (ionisation age = plasma age) charge
states, each normalised ``ne_weighted_mean`` (Z_eff = 2, the VEST default,
#1569) and ``fixed`` (n_C/n_e = n_O/n_e = 1/86, the fully stripped preset).
The elemental composition is the VEST ``impurity_model`` (C:O = 1:1).

Usage (vestserver, with the lane's sitecustomize shim on PYTHONPATH)::

    python3 workflow/impurity_zeff/build_zeff_profile.py \\
        --filedb ~/runs/campaign/filedb --atlas ~/runs/campaign/atlas/v1 \\
        --out ~/runs/campaign/atlas/impurity
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

MODELS = (
    ("coronal", "ne_weighted_mean"),
    ("coronal", "fixed"),
    ("transient", "ne_weighted_mean"),
    ("transient", "fixed"),
)

STATE_COLUMNS = {
    "shot": "shot number", "time_efit_s": "EFIT slice time [s]", "efit_lineage": "magnetics",
    "efit_quality": "criteria.py label (good | admissible)", "ionization": "coronal | transient",
    "normalization": "ne_weighted_mean | fixed", "status": "ok | no_core_profiles_slice | no_equilibrium | error",
    "reason": "why a state has no result", "time_cp_s": "matched core_profiles slice time [s]",
    "plasma_age_s": "EFIT time minus plasma onset [s]", "onset_source": "plasma timing source",
    "scale": "n_s/n_e = scale * w_s", "n_C_over_ne": "carbon elemental fraction",
    "n_O_over_ne": "oxygen elemental fraction", "zeff_axis": "Z_eff at the innermost point",
    "zeff_ne_mean": "n_e- and volume-weighted mean Z_eff", "zeff_rho09": "Z_eff at rho_tor_norm = 0.9",
    "z_i_eff_axis": "reduced impurity charge S2/S1 on axis", "dilution_axis": "1 - n_H/n_e on axis",
    "mean_z_C_axis": "<Z> of carbon on axis", "mean_z_O_axis": "<Z> of oxygen on axis",
    "te_axis_eV": "T_e at the innermost point [eV]", "not_coronal_fraction": "share of points not coronal at the plasma age",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    from vaft.omas import load

    return load(path)


def _product(filedb: Path, stage: str, shot: int, name: str) -> Optional[Path]:
    path = filedb / "omas" / stage / str(shot) / "output" / name
    return path if path.is_file() else None


def _value(ods: Any, path: str) -> Any:
    from vaft.ods_access import path_value

    return path_value(ods, path, None)


def _slice(ods: Any, ids: str, t: float, tolerance: float) -> Optional[int]:
    times = _value(ods, f"{ids}.time")
    if times is None:
        return None
    times = np.atleast_1d(np.asarray(times, dtype=float))
    index = int(np.argmin(np.abs(times - t)))
    return index if abs(times[index] - t) <= tolerance else None


def _volume_weights(eq: Any, t: float, rho: np.ndarray) -> Optional[np.ndarray]:
    """dV per core_profiles rho point from the EFIT V(rho_tor_norm) of the same time."""
    index = _slice(eq, "equilibrium", t, 5e-4)
    if index is None:
        return None
    base = f"equilibrium.time_slice.{index}.profiles_1d"
    rho_eq, volume = _value(eq, f"{base}.rho_tor_norm"), _value(eq, f"{base}.volume")
    if rho_eq is None or volume is None:
        return None
    rho_eq, volume = np.asarray(rho_eq, dtype=float), np.asarray(volume, dtype=float)
    ok = np.isfinite(rho_eq) & np.isfinite(volume)
    order = np.argsort(rho_eq[ok])
    edges = np.concatenate(([rho[0]], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]))
    v = np.interp(edges, rho_eq[ok][order], volume[ok][order])
    return np.clip(np.diff(v), 0.0, None)


def _onset(filedb: Path, shot: int, cache: dict) -> tuple[Optional[float], str]:
    if shot not in cache:
        path = _product(filedb, "diagnostics", shot, "diagnostics.json.gz")
        if path is None:
            cache[shot] = (None, "no diagnostics product")
        else:
            from vaft.omas.plasma_timing import plasma_timing

            try:
                timing = plasma_timing(_load(path))
                cache[shot] = (timing.onset, str(timing.source)) if timing.found else (None, "no plasma found")
            except Exception as exc:  # noqa: BLE001 -- recorded, the state still runs coronal
                cache[shot] = (None, f"{type(exc).__name__}: {exc}"[:120])
    return cache[shot]


def build(filedb: Path, atlas: Path, out: Path, *, tolerance: float, cache_dir: Optional[str]) -> dict:
    from vaft.machine_mapping.core_profiles import vest_impurity_model
    from vaft.process.impurity import resolve_radial_composition

    rows = [r for r in csv.DictReader(open(atlas / "state.csv"))
            if r["efit_lineage"] == "magnetics" and r["efit_quality"] in ("good", "admissible")]
    model = vest_impurity_model(None)
    weights = {item["element"]: item["relative_density"] for item in model["species"]}
    target = model["target_zeff"]
    profiles, states, inputs, onsets, cps, eqs = [], [], {}, {}, {}, {}
    for r in rows:
        shot, t = int(r["shot"]), float(r["time_efit_s"])
        key = {"shot": shot, "time_efit_s": round(t, 4), "efit_lineage": "magnetics",
               "efit_quality": r["efit_quality"]}
        cp_path = _product(filedb, "core_profiles", shot, "core_profiles.json.gz")
        eq_path = _product(filedb, "efit/magnetic", shot, "efit.json.gz")
        if cp_path is None or eq_path is None:
            states += [{**key, "ionization": i, "normalization": n, "status": "no_equilibrium" if cp_path else "no_core_profiles_slice",
                        "reason": "no FileDB product"} for i, n in MODELS]
            continue
        for p in (cp_path, eq_path):
            inputs.setdefault(str(p.relative_to(filedb)), _sha(p))
        cp = cps.setdefault(shot, _load(cp_path))
        eq = eqs.setdefault(shot, _load(eq_path))
        index = _slice(cp, "core_profiles", t, tolerance)
        if index is None:
            states += [{**key, "ionization": i, "normalization": n, "status": "no_core_profiles_slice",
                        "reason": f"no core_profiles slice within {tolerance:g} s"} for i, n in MODELS]
            continue
        base = f"core_profiles.profiles_1d.{index}"
        rho = np.asarray(_value(cp, f"{base}.grid.rho_tor_norm"), dtype=float)
        te = np.asarray(_value(cp, f"{base}.electrons.temperature"), dtype=float)
        ne = _value(cp, f"{base}.electrons.density_thermal")
        ne = np.asarray(ne if ne is not None else _value(cp, f"{base}.electrons.density"), dtype=float)
        t_cp = float(np.atleast_1d(_value(cp, "core_profiles.time"))[index])
        dv = _volume_weights(eq, t, rho)
        onset, onset_source = _onset(filedb, shot, onsets)
        age = None if onset is None else t - onset
        for ionization, normalization in MODELS:
            row = {**key, "ionization": ionization, "normalization": normalization, "time_cp_s": t_cp,
                   "plasma_age_s": age, "onset_source": onset_source}
            if ionization == "transient" and (age is None or age <= 0.0):
                states.append({**row, "status": "error", "reason": f"no plasma age ({onset_source})"})
                continue
            try:
                result = resolve_radial_composition(
                    te, ne, rho, weights, normalization=normalization, target_zeff=target,
                    ionization=ionization, plasma_age_s=age, volume_weights=dv, cache_dir=cache_dir, time=t_cp)
            except Exception as exc:  # noqa: BLE001 -- recorded per state
                states.append({**row, "status": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            valid = np.isfinite(result.zeff)
            weight = ne * (dv if dv is not None else rho)
            m = valid & np.isfinite(weight)
            first = int(np.flatnonzero(valid)[0])
            row.update(
                status="ok", reason="", scale=result.scale,
                n_C_over_ne=float(result.elemental_fractions[first, result.elements.index("C")]),
                n_O_over_ne=float(result.elemental_fractions[first, result.elements.index("O")]),
                zeff_axis=float(result.zeff[first]),
                zeff_ne_mean=float(np.sum((weight * result.zeff)[m]) / np.sum(weight[m])),
                zeff_rho09=float(np.interp(0.9, rho[valid], result.zeff[valid])),
                z_i_eff_axis=float(result.effective_charge[first]),
                dilution_axis=float(result.dilution_fraction[first]),
                mean_z_C_axis=float(result.mean_charge[first, result.elements.index("C")]),
                mean_z_O_axis=float(result.mean_charge[first, result.elements.index("O")]),
                te_axis_eV=float(te[first]),
                not_coronal_fraction=None if result.coronal_valid is None
                else float(np.mean(~result.coronal_valid[valid])),
            )
            states.append(row)
            for prow in result.as_rows(**key, ionization=ionization, normalization=normalization):
                profiles.append(prow)
            inputs.setdefault("adf11", result.provenance["tables"])
    out.mkdir(parents=True, exist_ok=True)
    for name, table, columns in (("states.csv", states, list(STATE_COLUMNS)),
                                 ("zeff_profile.csv", profiles, None)):
        columns = columns or sorted({k for x in table for k in x}, key=lambda c: (c not in ("shot", "time_efit_s", "efit_lineage", "efit_quality", "ionization", "normalization", "rho"), c))
        with open(out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table)
    try:
        sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    manifest = {
        "producer": "workflow/impurity_zeff/build_zeff_profile.py (Lane L, #1565, log #1569)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vaft_git": sha, "state_key_contract": "#1454 v1",
        "filedb": str(filedb), "atlas": str(atlas), "tolerance_s": tolerance,
        "composition": {"weights": weights, "target_zeff": target, "source": "vest.yaml impurity_model"},
        "models": [list(m) for m in MODELS], "inputs_sha256": inputs,
        "states": len(states), "profile_rows": len(profiles), "state_columns": STATE_COLUMNS,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True, help="Lane K atlas directory holding state.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=5e-4, help="core_profiles/EFIT time match [s]")
    parser.add_argument("--adf11-cache", default=None)
    args = parser.parse_args(argv)
    manifest = build(args.filedb, args.atlas, args.out, tolerance=args.tolerance, cache_dir=args.adf11_cache)
    print(json.dumps({k: manifest[k] for k in ("states", "profile_rows", "vaft_git")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
