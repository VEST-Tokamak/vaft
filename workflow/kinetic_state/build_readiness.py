"""TGLF/NEO input readiness per atlas state and T_i lineage (#1428, handed to lane T).

Reads ``state.csv`` (``build_state.py``), ``ti_inferred.csv`` and
``core_profiles/<shot>.json.gz`` (``build_ti.py``) and the campaign FileDB; writes
``readiness.csv`` (state key + ``ti_lineage`` + target ``rho_tor_norm``) and
``readiness_state.csv`` (one verdict per state key and lineage), with schemas.

Each Thomson-matched state is assessed under every T_i lineage it can carry:

``ti_eq_te_assumed``
    T_i = (Ti/Te) T_e with the machine policy ratio (#1414, ``vest.yaml``),
    for ``magnetics`` and ``electron_kinetic`` equilibria alike.
``pressure_partition_inferred``
    the #1426 T_i, ``magnetics`` equilibria only (electron-kinetic pressure is
    circular).

The decision itself is :func:`vaft.validation.kinetic_state.assess_transport_readiness`,
which asks the GACODE converters rather than restating them.  No solver runs.

Usage::

    python3 workflow/kinetic_state/build_readiness.py --filedb ~/runs/campaign/filedb \\
        --atlas ~/runs/campaign/atlas/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_state import SLICE_TOLERANCE_S, _cell, _load  # noqa: E402
from build_ti import KEY, _f, _read, _write  # noqa: E402

TARGETS = (0.3, 0.5, 0.7)

TARGET_COLUMNS: dict[str, tuple[str, str]] = {
    **{k: ("string" if k in ("contract_version", "efit_lineage", "efit_quality") else
           "integer" if k == "shot" else "number", "state key (#1454)") for k in KEY},
    "ti_lineage": ("string", "ti_eq_te_assumed | pressure_partition_inferred"),
    "rho_tor_norm": ("number", "target radius"),
    "r_over_a": ("number", "the same radius as TGLF's r/a on the converted profile"),
    "status": ("string", "ready | conditional | insufficient"),
    "reason": ("string", "why not ready; a converter refusal is quoted"),
    "a_lne": ("number", "a/L_ne at the target"),
    "a_lte": ("number", "a/L_Te at the target"),
    "a_lti": ("number", "a/L_Ti (main ion) at the target"),
    "ti_te": ("number", "T_i/T_e at the target"),
}

STATE_COLUMNS: dict[str, tuple[str, str]] = {
    **{k: TARGET_COLUMNS[k] for k in KEY},
    "ti_lineage": TARGET_COLUMNS["ti_lineage"],
    "status": ("string", "best status over the targets"),
    "reason": ("string", "reason attached to that status"),
    "ti_te_ratio_policy": ("number", "machine Ti/Te ratio used (ti_eq_te_assumed)"),
    "inferred_ti_rho_max": ("number", "where the inferred T_i stops being usable"),
    "assumptions": ("string", "';'-joined declared assumptions"),
}


def build(filedb: Path, atlas: Path, *, targets=TARGETS) -> dict[str, Any]:
    from vaft.machine_mapping.core_profiles import vest_core_profiles_policy
    from vaft.validation import kinetic_state as ks

    state = _read(atlas / "state.csv")
    inferred_points = defaultdict(list)
    channel_rho = defaultdict(list)
    ti_path = atlas / "ti_inferred.csv"
    for point in (_read(ti_path) if ti_path.is_file() else []):
        key = tuple(point[k] for k in KEY)
        if point["kind"] == "grid":
            inferred_points[key].append(point)
        elif math.isfinite(_f(point["rho_tor_norm"])):
            channel_rho[key].append(_f(point["rho_tor_norm"]))
    products: dict[tuple[str, int], Any] = {}

    def product(kind: str, shot: int):
        if (kind, shot) not in products:
            path = {
                "magnetics": filedb / "omas/efit/magnetic" / str(shot) / "output/efit.json.gz",
                "electron_kinetic": filedb / "omas/electron_efit" / str(shot) / "output/electron_efit.json.gz",
                "core_profiles": filedb / "omas/core_profiles" / str(shot) / "output/core_profiles.json.gz",
            }[kind]
            products[(kind, shot)] = _load(path) if path.is_file() else None
        return products[(kind, shot)]

    target_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []

    def record(key, lineage, result, **extra):
        for target in result["targets"]:
            target_rows.append({**key, "ti_lineage": lineage, "rho_tor_norm": target["rho"],
                                "r_over_a": target.get("r_over_a"), "status": target["status"],
                                "reason": target.get("reason"), "a_lne": target.get("a_lne"),
                                "a_lte": target.get("a_lte"), "a_lti": target.get("a_lti"),
                                "ti_te": target.get("ti_te")})
        state_rows.append({**key, "ti_lineage": lineage, "status": result["status"], "reason": result.get("reason"),
                           "inferred_ti_rho_max": result.get("provenance", {}).get("inferred_ti_rho_max"),
                           "assumptions": ";".join(result.get("assumptions", ())), **extra})

    for row in state:
        if row["ts_status"] != "matched" or row["efit_status"] != "valid":
            continue
        key = {k: row[k] for k in KEY}
        shot, time_s, ts_time = int(row["shot"]), float(row["time_efit_s"]), float(row["time_ts_s"])
        tolerance = float(row["ts_tolerance_s"])
        equilibrium = product(row["efit_lineage"], shot)

        # -- assumed Ti/Te: the campaign core_profiles fit at the Thomson time
        cp = product("core_profiles", shot)
        ratio = vest_core_profiles_policy(shot).ti_te_ratio
        electron = (ks.core_profiles_electron_pressure(cp, time_s=ts_time, tolerance_s=tolerance)
                    if cp is not None else {"available": False, "reason": "no core_profiles product"})
        if electron["available"]:
            j, _ = ks.slice_at_time(cp, electron["time_s"], ids="core_profiles", tolerance_s=1e-6)
            p = cp["core_profiles"]["profiles_1d"][j]
            n_e = np.asarray(p["electrons"].get("density_thermal", p["electrons"].get("density")), dtype=float)
            t_e = np.asarray(p["electrons"]["temperature"], dtype=float)
            result = ks.assess_transport_readiness(
                equilibrium, time_s=time_s, profile_time_s=electron["time_s"],
                rho_tor_norm=electron["rho_tor_norm"], n_e=n_e, t_e=t_e, t_i=ratio * t_e,
                ti_lineage="ti_eq_te_assumed", rho_targets=targets, tolerance_s=SLICE_TOLERANCE_S)
        else:
            reason = f"no electron profile: {electron['reason']}"
            result = {"status": "insufficient", "reason": reason,
                      "targets": [{"rho": t, "status": "insufficient", "reason": reason} for t in targets]}
        record(key, "ti_eq_te_assumed", result, ti_te_ratio_policy=ratio)

        # -- inferred T_i: magnetics only
        if row["efit_lineage"] != "magnetics":
            continue
        grid = sorted(inferred_points.get(tuple(row[k] for k in KEY), []), key=lambda g: _f(g["rho_tor_norm"]))
        if not grid:
            reason = "no inferred T_i on a core_profiles grid for this state"
            result = {"status": "insufficient", "reason": reason,
                      "targets": [{"rho": t, "status": "insufficient", "reason": reason} for t in targets]}
        else:
            rho_c = channel_rho.get(tuple(row[k] for k in KEY), [])
            result = ks.assess_transport_readiness(
                equilibrium, time_s=time_s, profile_time_s=ts_time,
                rho_tor_norm=[_f(g["rho_tor_norm"]) for g in grid], n_e=[_f(g["n_e_m3"]) for g in grid],
                t_e=[_f(g["t_e_ev"]) for g in grid], t_i=[_f(g["t_i_ev"]) for g in grid],
                ti_lineage="pressure_partition_inferred",
                ti_flags=[g["flags"].replace("outside_ts_span", "").strip(";") for g in grid],
                ts_rho_span=(min(rho_c), max(rho_c)) if rho_c else None,
                rho_targets=targets, tolerance_s=SLICE_TOLERANCE_S)
        record(key, "pressure_partition_inferred", result)

    _write(atlas / "readiness.csv", TARGET_COLUMNS, target_rows, "VAFT transport readiness per radius (#1428, contract #1454)")
    _write(atlas / "readiness_state.csv", STATE_COLUMNS, state_rows, "VAFT transport readiness per state (#1428, contract #1454)")
    summary = {
        "states": Counter(f"{r['efit_lineage']}/{r['ti_lineage']}/{r['status']}" for r in state_rows),
        "targets": Counter(f"{r['ti_lineage']}/{r['status']}" for r in target_rows),
        "insufficient_reasons": Counter(str(r["reason"]).split(":")[0][:80] for r in target_rows
                                        if r["status"] == "insufficient"),
        "targets_rho_tor_norm": list(targets),
    }
    (atlas / "readiness_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.filedb.expanduser(), args.atlas.expanduser()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
