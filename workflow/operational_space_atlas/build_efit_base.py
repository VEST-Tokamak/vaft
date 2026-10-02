"""Lane V base table for the operational-space atlas (#1456, conference 2026-10-12).

One row per state key of Lane K's State key contract v1 (#1454): the #1331 Tier A
good + admissible EFIT slices, magnetics and electron_kinetic lineages. For each
row it adds the operational-space coordinates, computed by VAFT from the row's own
EFIT product, under the quantity names of ``vaft.formula.boundaries`` so the
population renderer can check identity and unit:

    internal_inductance_li3   DD li_3 from vaft.omas.update.update_equilibrium_global_quantities_beta_li,
                              which needs #1477 (the fix of #1462); see the cross-check below
    normalized_beta           DD beta_normal from the same routine, % m T / MA
    toroidal_beta             100 * beta_t of vaft.process.equilibrium.derive_global_descriptors, %
    normalized_current        |I_p| [MA] / (a [m] |B_0| [T]), B_0 = vacuum_toroidal_field.b0 at r0
    edge_safety_factor_95     q95 (derive_global_descriptors)
    edge_safety_factor        q at the boundary surface, q_psi (derive_global_descriptors q_edge)
    area_elongation           cross-section area / (pi a^2)
    elongation                boundary elongation kappa (derive_global_descriptors)
    plasma_surface_area       LCFS surface area S (derive_global_descriptors), for L-H thresholds
    inverse_cylindrical_q     |I_p| R_geo / (5 a^2 kappa_a B_T(R_geo)), B_T(R_geo) = |b0| r0 / R_geo

Before #1477 that routine returned li_3 up to 10x and beta_normal up to 4x on many
Tier A slices (#1462: open grid-edge branches posed as flux surfaces). Every row is
therefore cross-checked against two independent paths: li3_grid_integral (2 int B_p^2 dV
summed on the EFIT grid inside the boundary outline, through
vaft.formula.equilibrium.li_3_from_Bp2_volume_integral) and beta_normal_descriptors
(derive_global_descriptors). li_beta_crosscheck is "agree" when both are within
CROSSCHECK_RTOL; the MANIFEST counts the rows that disagree.

The slice is matched to ``equilibrium.time`` by time (1 us: the state times come
from the same product, so a neighbouring slice must never stand in), never by
index; a row whose time has no slice is kept with ``base_status = time_unmatched``.
The product's sha256 is checked against the state's ``efit_product_sha256``; a
regenerated product gives ``base_status = product_changed`` and no values. Lane K's state columns (labels, R_p) are carried
through unchanged. Nothing is written outside ``--out``.

Usage::

    python build_efit_base.py --state ~/runs/campaign/atlas/v1/state.csv \
        --filedb ~/runs/campaign/filedb --out ~/runs/campaign/atlas/lane_v
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CROSSCHECK_RTOL = 0.10  # grid-cell bias of the li_3 integral is a few percent
TIME_TOLERANCE_S = 1e-6  # state times come from the same product: a match is exact up to float formatting
UNITS = {
    "internal_inductance_li3": "-",
    "normalized_beta": "% m T/MA",
    "toroidal_beta": "%",
    "normalized_current": "MA m^-1 T^-1",
    "edge_safety_factor_95": "-",
    "edge_safety_factor": "-",
    "area_elongation": "-",
    "elongation": "-",
    "plasma_surface_area": "m^2",
    "inverse_cylindrical_q": "-",
    "plasma_current_ma": "MA",
    "minor_radius_m": "m",
    "major_radius_geo_m": "m",
    "b0_t": "T",
    "r0_m": "m",
    "r_reference_m": "m",
    "li3_grid_integral": "-",
    "beta_normal_descriptors": "% m T/MA",
    "update_routine_status": "",
    "li_beta_crosscheck": "",
}
CARRIED = ("shot", "time_efit_s", "efit_lineage", "efit_quality", "efit_setting", "efit_product",
           "efit_product_sha256", "kinetic_admissible", "ts_status", "r_sum", "r_w", "r_w_reason", "ti_lineage")


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


def _leaf(ods, path):
    """A leaf value, or None -- without creating the path (omas reads create paths)."""
    return ods[path] if path in ods else None


def _bp2_volume_integral(ods, k: int) -> float:
    """int B_p^2 dV [T^2 m^3] over the grid cells inside the boundary outline of slice k."""
    from matplotlib.path import Path as _Path
    from vaft.omas import ods_psi_to_wb_per_radian_factor

    ts = f"equilibrium.time_slice.{k}"
    grid = f"{ts}.profiles_2d.0"
    r = np.asarray(ods[f"{grid}.grid.dim1"], dtype=float)
    z = np.asarray(ods[f"{grid}.grid.dim2"], dtype=float)
    psi = np.asarray(ods[f"{grid}.psi"], dtype=float) * ods_psi_to_wb_per_radian_factor(ods)
    dpsi_dr, dpsi_dz = np.gradient(psi, r, z, edge_order=2)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    outline = np.c_[np.asarray(ods[f"{ts}.boundary.outline.r"]), np.asarray(ods[f"{ts}.boundary.outline.z"])]
    inside = _Path(outline).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(rr.shape)
    dv = 2.0 * np.pi * rr * (r[1] - r[0]) * (z[1] - z[0])
    return float(np.sum((dpsi_dr**2 + dpsi_dz**2) / rr**2 * dv * inside))


def _routine_diagnostic(ods, k: int):
    """li_3 and beta_normal from update_equilibrium_global_quantities_beta_li, on a copy, in isolation.

    Returns NaNs and a status when the routine raises or leaves the leaves untouched (it skips slices
    with unusable inputs), so a value already stored in the product is never reported as its output.
    """
    import copy

    from vaft.omas.update import update_equilibrium_global_quantities_beta_li

    work = copy.deepcopy(ods)
    gq = f"equilibrium.time_slice.{k}.global_quantities"
    for name in ("li_3", "beta_normal"):
        if f"{gq}.{name}" in work:
            del work[f"{gq}.{name}"]
    try:
        update_equilibrium_global_quantities_beta_li(work, time_slice=k)
    except Exception as exc:  # diagnostic only: never costs the row its real columns
        return math.nan, math.nan, f"raised {type(exc).__name__}"
    li3, bn = (_leaf(work, f"{gq}.{n}") for n in ("li_3", "beta_normal"))
    if li3 is None and bn is None:
        return math.nan, math.nan, "skipped"
    return (float(li3) if li3 is not None else math.nan, float(bn) if bn is not None else math.nan, "wrote")


def _row(ods, t_s: float) -> dict:
    from vaft.formula.equilibrium import li_3_from_Bp2_volume_integral
    from vaft.omas import resolve_reference_major_radius
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    times = np.asarray(ods["equilibrium.time"], dtype=float)
    k = int(np.argmin(np.abs(times - t_s)))
    dt = float(times[k] - t_s)
    if abs(dt) > TIME_TOLERANCE_S:
        return {"base_status": "time_unmatched", "base_reason": f"nearest slice {dt:+.4g} s away", "dt_slice_s": dt}
    d = derive_global_descriptors(as_equilibrium(ods, time_index=k)).values
    bp2 = _bp2_volume_integral(ods, k)
    r_ref = float(resolve_reference_major_radius(ods))
    li3 = li_3_from_Bp2_volume_integral(bp2, abs(float(ods[f"equilibrium.time_slice.{k}.global_quantities.ip"])), r_ref)
    li3_routine, bn_routine, routine_status = _routine_diagnostic(ods, k)
    pick = lambda n: float(d[n].value) if n in d and d[n].available else math.nan
    ip_ma = abs(pick("ip")) * 1e-6
    a, r_geo = pick("minor_radius"), pick("major_radius")
    area = pick("cross_section_area")
    b0 = abs(float(np.atleast_1d(ods["equilibrium.vacuum_toroidal_field.b0"])[k]))
    r0 = float(ods["equilibrium.vacuum_toroidal_field.r0"])
    kappa_a = area / (math.pi * a * a)
    b_geo = b0 * r_ref / r_geo  # vacuum field at R_geo, from the field at the resolved reference radius
    return {
        "base_status": "valid",
        "base_reason": "",
        "dt_slice_s": dt,
        "internal_inductance_li3": li3_routine,
        "normalized_beta": bn_routine,
        "li3_grid_integral": float(li3),
        "beta_normal_descriptors": pick("beta_n"),
        "li_beta_crosscheck": ("agree" if routine_status == "wrote"
                               and abs(li3_routine / float(li3) - 1) <= CROSSCHECK_RTOL
                               and abs(bn_routine / pick("beta_n") - 1) <= CROSSCHECK_RTOL else "disagree"),
        "toroidal_beta": 100.0 * pick("beta_t"),
        "update_routine_status": routine_status,
        "r_reference_m": r_ref,
        "normalized_current": ip_ma / (a * b0),
        "edge_safety_factor_95": pick("q95"),
        "edge_safety_factor": pick("q_edge"),
        "area_elongation": kappa_a,
        "elongation": pick("elongation"),
        "plasma_surface_area": pick("surface_area"),
        "inverse_cylindrical_q": ip_ma * r_geo / (5.0 * a * a * kappa_a * b_geo),
        "plasma_current_ma": ip_ma,
        "minor_radius_m": a,
        "major_radius_geo_m": r_geo,
        "b0_t": b0,
        "r0_m": r0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--filedb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import vaft

    with open(args.state) as fh:
        states = list(csv.DictReader(fh))
    cache, rows = {}, []
    for s in states:
        product = args.filedb / s["efit_product"]
        if product not in cache:
            cache[product] = (_sha256(product), _load(product))
        sha, ods = cache[product]
        base = {c: s.get(c, "") for c in CARRIED}
        if s.get("efit_product_sha256") and s["efit_product_sha256"] != sha:
            base.update({"base_status": "product_changed",
                         "base_reason": f"product sha256 {sha[:12]} != state {s['efit_product_sha256'][:12]}"})
            rows.append(base)
            continue
        try:
            base.update(_row(ods, float(s["time_efit_s"])))
        except Exception as exc:  # recorded per row, never silently dropped
            base.update({"base_status": "failed", "base_reason": f"{type(exc).__name__}: {exc}"})
        rows.append(base)
        print(s["shot"], s["time_efit_s"], s["efit_lineage"], base["base_status"], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    columns = list(CARRIED) + ["base_status", "base_reason", "dt_slice_s"] + list(UNITS)
    with open(args.out / "efit_base.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    repo = Path(vaft.__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    manifest = {
        "contract": "Lane K State key contract v1 (#1454)",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *(argv or sys.argv[1:])],
        "vaft_git": git("rev-parse", "HEAD"),
        "vaft_dirty": bool(git("status", "--porcelain")),
        "inputs": {"state": str(args.state), "state_sha256": _sha256(args.state), "filedb": str(args.filedb)},
        "time_tolerance_s": TIME_TOLERANCE_S,
        "units": UNITS,
        "rows": len(rows),
        "status_counts": {k: sum(r["base_status"] == k for r in rows) for k in {r["base_status"] for r in rows}},
        "li_beta_crosscheck": {k: sum(r.get("li_beta_crosscheck") == k for r in rows) for k in ("agree", "disagree")},
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["status_counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
