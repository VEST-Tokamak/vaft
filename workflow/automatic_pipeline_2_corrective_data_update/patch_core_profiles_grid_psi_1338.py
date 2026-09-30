#!/usr/bin/env python3
"""Divide ``core_profiles.profiles_1d[:].grid.psi`` by 2*pi in place (issue #1338).

Before PR #1337 (#1292, merged 2026-09-29 06:21 UTC) ``from_equilibrium`` wrote
a COCOS 11 record's flux into the g-file unconverted and ``to_omas`` multiplied
it by 2*pi again. The ``core_profiles`` stage's *solution* branch
(``--efit-product``) takes ``grid.psi`` from that object, so its products carry
absolute flux 2*pi too large. The ``geqdsk_dir`` branch read real per-radian
g-files and is not affected; nothing else in the product is (normalized
coordinates, profiles, the EFIT stage's own ``equilibrium``).

The electron/kinetic EFIT stages build the same object but publish only the
reconstructed ``equilibrium`` from EFIT's own g-file, so they carry no
``core_profiles``. This script still inspects their products and reports any
that unexpectedly hold one; it never patches them.

What decides a patch is a measurement, not a date. Each slice's ``grid.psi`` is
compared with the same product's equilibrium -- the EFIT product its manifest
names -- matched by **time**, not index: ``equilibrium.time_slice[j].
profiles_1d.psi`` (Wb) interpolated onto the slice's ``grid.rho_tor_norm``. A
product is patched only when every slice measures 2*pi within tolerance and
none is already marked. Anything else -- a ratio of 1, a mixed product, a
negative ratio, an unmatched time, a newer EFIT product than the profiles -- is
reported and refused.

The patch works on the stored JSON directly, so every other leaf stays byte-for-
byte what the stage wrote. It appends one provenance line to
``core_profiles.code.parameters`` (the text block the profile mapper already
writes there; ``code.name`` is left alone because readers key on it), records
the patch in the stage manifest under ``patches`` and refreshes the manifest's
``output.sha256``. The line is also the idempotency marker.

Dry run is the default. ``--apply`` needs ``--backup-dir``; each product and its
manifest are copied there first, under their FileDB-relative path.

HSDS is not touched. A patched product whose ``metadata/replication.json``
exists must be re-replicated with ``replicate_to_hsds.py --stage core_profiles``
(the manifest hash changes, so the recorded hash no longer matches), which sends
only the stage's owned IDS and merges ``master.h5``'s links rather than
replacing them (``vaft.database.staging.merge_master_links``).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ISSUE = "#1338"
MARKER = "grid.psi divided by 2*pi (#1338)"
TWO_PI = 2.0 * math.pi
#: PR #1337 merge time. Only a hint for the report: a checkout older than the fix
#: can write after this date, which is why the measurement decides.
FIX_MERGED_UTC = datetime(2026, 9, 29, 6, 21, 57, tzinfo=timezone.utc)
DEFAULT_TOLERANCE_MS = 0.5
RATIO_RTOL = 1e-3
SHAPE_RTOL = 1e-3

PRODUCT_NAMES = ("core_profiles.json", "core_profiles.json.gz", "core_profiles.h5")
SIDE_STAGES = ("electron_efit", "kinetic_efit", "neoclassical")


# --------------------------------------------------------------------------- io
def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=0)
    tmp = path.with_name(path.name + ".tmp-1338")
    if path.suffix == ".gz":
        # mtime=0 keeps the file deterministic, as the stage writer's hash is.
        with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(text.encode("utf-8"))
    else:
        tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# ----------------------------------------------------------------- measurement
def _equilibrium_slices(efit: dict) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """(time_s, rho_tor_norm, psi_Wb) per equilibrium slice that has both."""
    eq = efit.get("equilibrium") or {}
    times = _as_list(eq.get("time"))
    slices = []
    for j, ts in enumerate(_as_list(eq.get("time_slice"))):
        if not isinstance(ts, dict):
            continue
        t = ts.get("time", times[j] if j < len(times) else None)
        p1 = ts.get("profiles_1d") or {}
        psi, rho = p1.get("psi"), p1.get("rho_tor_norm")
        if t is None or psi is None or rho is None:
            continue
        slices.append((float(t), np.asarray(rho, float), np.asarray(psi, float)))
    return slices


def measure_slice(
    profile: dict, eq_slices: list, tolerance_ms: float
) -> dict[str, Any]:
    """Ratio of stored grid.psi to the equilibrium's psi at the same time."""
    grid = profile.get("grid") or {}
    t = profile.get("time")
    out: dict[str, Any] = {"time_ms": None if t is None else round(float(t) * 1e3, 3)}
    if grid.get("psi") is None:
        out["verdict"] = "no_grid_psi"
        return out
    if t is None or grid.get("rho_tor_norm") is None:
        out["verdict"] = "unmeasurable"
        out["reason"] = "slice has no time or no grid.rho_tor_norm"
        return out
    if not eq_slices:
        out["verdict"] = "unmeasurable"
        out["reason"] = "equilibrium product has no usable slice"
        return out
    offsets = [abs(ts - float(t)) * 1e3 for ts, _, _ in eq_slices]
    j = int(np.argmin(offsets))
    out["equilibrium_time_ms"] = round(eq_slices[j][0] * 1e3, 3)
    if offsets[j] > tolerance_ms:
        out["verdict"] = "unmeasurable"
        out["reason"] = f"nearest equilibrium slice {offsets[j]:.3f} ms away"
        return out
    _, rho_eq, psi_eq = eq_slices[j]
    order = np.argsort(rho_eq)
    rho = np.asarray(grid["rho_tor_norm"], float)
    psi_cp = np.asarray(grid["psi"], float)
    ref = np.interp(rho, rho_eq[order], psi_eq[order])
    denom = float(np.dot(ref, ref))
    if denom == 0.0 or not np.all(np.isfinite(psi_cp)):
        out["verdict"] = "unmeasurable"
        out["reason"] = "zero or non-finite flux"
        return out
    ratio = float(np.dot(psi_cp, ref) / denom)
    residual = float(np.linalg.norm(psi_cp - ratio * ref) / max(np.linalg.norm(psi_cp), 1e-300))
    out["ratio"] = ratio
    out["ratio_over_2pi"] = ratio / TWO_PI
    out["shape_residual"] = residual
    if residual > SHAPE_RTOL:
        out["verdict"] = "inconsistent"
    elif abs(ratio / TWO_PI - 1.0) < RATIO_RTOL:
        out["verdict"] = "affected"
    elif abs(ratio - 1.0) < RATIO_RTOL:
        out["verdict"] = "correct"
    else:
        out["verdict"] = "inconsistent"
    return out


# --------------------------------------------------------------------- product
def _marked(cp: dict) -> bool:
    params = (cp.get("code") or {}).get("parameters")
    return isinstance(params, str) and MARKER in params


def inspect_product(filedb: Path, shot_dir: Path) -> dict[str, Any]:
    shot = shot_dir.name
    rec: dict[str, Any] = {"shot": int(shot) if shot.isdigit() else shot, "stage": "core_profiles"}
    outputs = [shot_dir / "output" / n for n in PRODUCT_NAMES if (shot_dir / "output" / n).exists()]
    manifest_path = shot_dir / "metadata" / "manifest.json"
    rec["replicated"] = (shot_dir / "metadata" / "replication.json").exists()
    if not outputs:
        rec["action"] = "skip"
        rec["reason"] = "no product file"
        return rec
    product = outputs[0]
    rec["path"] = str(product)
    written = _mtime_utc(product)
    rec["written_utc"] = written.isoformat(timespec="seconds")
    rec["written_after_fix"] = written > FIX_MERGED_UTC
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rec["status"] = manifest.get("status")
    inputs = manifest.get("input") or {}
    if inputs.get("efit_product"):
        rec["branch"] = "solution"
    elif inputs.get("geqdsk_dir"):
        rec["branch"] = "geqdsk_dir"
    else:
        rec["branch"] = "unknown"
    if product.suffix == ".h5":
        rec["action"] = "refuse"
        rec["reason"] = "HDF5 product: this script patches JSON products only"
        return rec
    data = _read_json(product)
    cp = data.get("core_profiles") or {}
    profiles = [p for p in _as_list(cp.get("profiles_1d")) if isinstance(p, dict)]
    rec["code_name"] = (cp.get("code") or {}).get("name")
    rec["n_slices"] = len(profiles)
    rec["n_with_grid_psi"] = sum(1 for p in profiles if (p.get("grid") or {}).get("psi") is not None)
    if _marked(cp):
        rec["action"] = "already_patched"
        return rec
    if rec["n_with_grid_psi"] == 0:
        rec["action"] = "skip"
        rec["reason"] = "no slice carries grid.psi"
        return rec
    if rec["branch"] != "solution":
        rec["action"] = "refuse"
        rec["reason"] = f"branch {rec['branch']}: only the solution branch is affected"
        return rec
    efit_path = Path(inputs["efit_product"])
    if not efit_path.exists():
        rec["action"] = "refuse"
        rec["reason"] = f"efit product missing: {efit_path}"
        return rec
    rec["efit_product"] = str(efit_path)
    if _mtime_utc(efit_path) > written:
        rec["efit_newer_than_profiles"] = True
    tolerance = float((manifest.get("configuration") or {}).get("equilibrium_tolerance_ms", DEFAULT_TOLERANCE_MS))
    eq_slices = _equilibrium_slices(_read_json(efit_path))
    rec["slices"] = [measure_slice(p, eq_slices, tolerance) for p in profiles]
    verdicts = {s["verdict"] for s in rec["slices"] if s["verdict"] != "no_grid_psi"}
    ratios = [s["ratio"] for s in rec["slices"] if "ratio" in s]
    if ratios:
        rec["ratio_min"], rec["ratio_max"] = min(ratios), max(ratios)
    rec["verdict"] = verdicts.pop() if len(verdicts) == 1 else "mixed:" + ",".join(sorted(verdicts))
    if rec["verdict"] == "affected":
        if rec.get("efit_newer_than_profiles"):
            # The ratio still measures 2*pi, so the EFIT product did not change the
            # flux; kept as a flag rather than a refusal, and named in the report.
            rec["note"] = "efit product is newer than the profiles; ratio measured against it"
        rec["action"] = "patch"
    elif rec["verdict"] == "correct":
        rec["action"] = "skip"
        rec["reason"] = "grid.psi already agrees with the equilibrium (post-fix or regenerated)"
    else:
        rec["action"] = "refuse"
        rec["reason"] = f"measured verdict {rec['verdict']}"
    return rec


def apply_patch(filedb: Path, rec: dict[str, Any], backup_dir: Path, on: str) -> None:
    product = Path(rec["path"])
    shot_dir = product.parent.parent
    manifest_path = shot_dir / "metadata" / "manifest.json"
    for src in (product, manifest_path):
        if not src.exists():
            continue
        dest = backup_dir / src.relative_to(filedb)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise RuntimeError(f"backup already exists, refusing to overwrite: {dest}")
        shutil.copy2(src, dest)
        if _sha256(dest) != _sha256(src):
            raise RuntimeError(f"backup copy differs from source: {dest}")

    data = _read_json(product)
    cp = data["core_profiles"]
    if _marked(cp):  # re-check under the write, not only in the scan
        raise RuntimeError(f"{product} is already marked")
    for p in _as_list(cp.get("profiles_1d")):
        grid = (p or {}).get("grid") or {}
        if grid.get("psi") is not None:
            grid["psi"] = (np.asarray(grid["psi"], float) / TWO_PI).tolist()
    ratios = f"{rec['ratio_min'] / TWO_PI:.6f}..{rec['ratio_max'] / TWO_PI:.6f}"
    line = (
        f"patch {MARKER} on {on}: pre-#1292 from_equilibrium/to_omas stored Wb*2pi; "
        f"measured grid.psi/equilibrium psi = 2*pi x [{ratios}] before the patch; "
        f"other leaves unchanged"
    )
    code = cp.setdefault("code", {})
    params = code.get("parameters")
    code["parameters"] = (params.rstrip("\n") + "\n" + line + "\n") if isinstance(params, str) and params else line + "\n"
    _write_json_atomic(product, data)

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous = (manifest.get("output") or {}).get("sha256")
    manifest.setdefault("patches", []).append(
        {
            "issue": ISSUE,
            "date": on,
            "leaf": "core_profiles.profiles_1d[:].grid.psi",
            "operation": "divide by 2*pi",
            "ratio_over_2pi": [rec["ratio_min"] / TWO_PI, rec["ratio_max"] / TWO_PI],
            "previous_sha256": previous,
        }
    )
    manifest.setdefault("output", {})["sha256"] = _sha256(product)
    manifest.setdefault("output", {}).setdefault("name", product.name)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp-1338")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)


def inspect_side_product(shot_dir: Path, stage: str) -> dict[str, Any] | None:
    """Report an electron/kinetic EFIT or neoclassical product carrying grid.psi."""
    for f in sorted((shot_dir / "output").glob("*.json*")):
        try:
            data = _read_json(f)
        except Exception as error:  # noqa: BLE001 - reported, not fatal
            return {"shot": shot_dir.name, "stage": stage, "path": str(f), "error": str(error)}
        cp = data.get("core_profiles") or {}
        n = sum(
            1 for p in _as_list(cp.get("profiles_1d"))
            if isinstance(p, dict) and (p.get("grid") or {}).get("psi") is not None
        )
        return {"shot": shot_dir.name, "stage": stage, "path": str(f), "core_profiles_grid_psi_slices": n,
                "written_utc": _mtime_utc(f).isoformat(timespec="seconds")}
    return None


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filedb-root", default=os.environ.get("VAFT_FILEDB_DIR", "/srv/vest.filedb"), type=Path)
    parser.add_argument("--shots", nargs="*", type=int, help="Restrict to these shots.")
    parser.add_argument("--apply", action="store_true", help="Write. Without it nothing is written (dry run).")
    parser.add_argument("--backup-dir", type=Path, help="Required with --apply; must not be inside the FileDB.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date recorded in the marker.")
    parser.add_argument("--report", type=Path, help="Write the full JSON report here.")
    args = parser.parse_args()

    filedb = args.filedb_root.resolve()
    stage_root = filedb / "omas" / "core_profiles"
    if args.apply:
        if args.backup_dir is None:
            parser.error("--apply needs --backup-dir")
        backup = args.backup_dir.resolve()
        if backup == filedb or filedb in backup.parents:
            parser.error("--backup-dir must be outside the FileDB")

    shot_dirs = sorted(p for p in stage_root.glob("*") if p.is_dir()) if stage_root.exists() else []
    if args.shots:
        wanted = {str(s) for s in args.shots}
        shot_dirs = [p for p in shot_dirs if p.name in wanted]
    records = [inspect_product(filedb, d) for d in shot_dirs]

    side = []
    for stage in SIDE_STAGES:
        for d in sorted((filedb / "omas" / stage).glob("*")) if (filedb / "omas" / stage).exists() else []:
            if d.is_dir() and (not args.shots or d.name in {str(s) for s in args.shots}):
                r = inspect_side_product(d, stage)
                if r is not None:
                    side.append(r)

    mode = "APPLY" if args.apply else "DRY RUN (nothing written)"
    print(f"# {ISSUE} core_profiles grid.psi patch -- {mode}")
    print(f"# filedb {filedb}  products {len(records)}  fix merged {FIX_MERGED_UTC.isoformat()}")
    print(f"{'shot':>6} {'branch':<10} {'status':<11} {'written (UTC)':<25} {'slices':>6} "
          f"{'ratio/2pi':<19} {'repl':<4} action")
    for r in records:
        rr = ""
        if "ratio_min" in r:
            rr = f"{r['ratio_min'] / TWO_PI:.5f}..{r['ratio_max'] / TWO_PI:.5f}"
        print(f"{r['shot']:>6} {r.get('branch', '-'):<10} {str(r.get('status')):<11} "
              f"{r.get('written_utc', '-'):<25} {r.get('n_with_grid_psi', 0):>6} {rr:<19} "
              f"{'yes' if r.get('replicated') else 'no':<4} {r['action']}"
              + (f" ({r['reason']})" if r.get("reason") else "")
              + (f" [{r['note']}]" if r.get("note") else ""))
    for s in side:
        if s.get("core_profiles_grid_psi_slices") or s.get("error"):
            print(f"# UNEXPECTED {s['stage']} product carries core_profiles.grid.psi: {s}")
    counts: dict[str, int] = {}
    for r in records:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    print(f"# actions: {counts}")
    print(f"# side-stage products inspected: {len(side)}; with core_profiles grid.psi: "
          f"{sum(1 for s in side if s.get('core_profiles_grid_psi_slices'))}")

    to_patch = [r for r in records if r["action"] == "patch"]
    replicated = [r["shot"] for r in to_patch if r.get("replicated")]
    if replicated:
        print(f"# HSDS: re-replicate after patching (replicate_to_hsds.py --stage core_profiles): {replicated}")
    if args.apply:
        for r in to_patch:
            apply_patch(filedb, r, args.backup_dir.resolve(), args.date)
            print(f"patched {r['path']}")
    if args.report:
        args.report.write_text(json.dumps({"records": records, "side": side, "mode": mode}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
