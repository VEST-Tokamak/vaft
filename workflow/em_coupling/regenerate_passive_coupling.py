#!/usr/bin/env python3
"""Repair the passive-passive coupling asset to exact reciprocity, with provenance (issues #373, #347).

`vaft/data/geometry/VEST_em_coupling_pf_versions.npz` was committed as an
opaque legacy binary.  Its `mutual_passive_passive` (950 x 950) violated
reciprocity by 1.27e-3 (relative), and the mapper has folded it to
(M + M^T)/2 on load since #347.  The defect has one cause: the donor MATLAB
kernel (`getMutualInductanceCoil.m`) tested the *first* conductor's material
twice where it meant to test both, so the relative permeability factor used
for SUS316LN (1.04) was applied only when the SUS loop came first.  The
matrix shows it exactly -- the SUS-SUS and W-W blocks are symmetric to zero,
and every SUS-W entry is 1.04 times its transpose.

This is therefore a **documented repair of the material factor**, not a
regeneration from geometry: the SUS-W block is set to the SUS-side value
(mu_r = 1.04 whenever either conductor is SUS, the donor's evident intent and
the factor `vaft.omas.process_wrapper` applies to every non-W11 loop), the
within-material blocks and the active-side matrices are copied verbatim, and
the asset gains a `provenance` key saying all of this.  Re-deriving the
matrices from `vaft.formula.green` would move every entry by up to 4 % and is
a different, larger decision.

The script refuses anything it cannot explain: a matrix whose asymmetry is
not exactly one constant factor on one cross-material block is not this
defect, and averaging it into symmetry here would hide whatever it is.

Usage::

    PYTHONPATH=. python workflow/em_coupling/regenerate_passive_coupling.py --verify
    PYTHONPATH=. python workflow/em_coupling/regenerate_passive_coupling.py --repair --dry-run
    PYTHONPATH=. python workflow/em_coupling/regenerate_passive_coupling.py --repair

`--repair` on an already-repaired asset exits 0 without touching it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

import vaft
from vaft.data.resources import data_path

ASSET = data_path("geometry/VEST_em_coupling_pf_versions.npz")
STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")
PROVENANCE_KEY = "provenance"
GENERATOR = "workflow/em_coupling/regenerate_passive_coupling.py"
ISSUES = (
    "https://github.com/VEST-Tokamak/vaft/issues/373",
    "https://github.com/VEST-Tokamak/vaft/issues/347",
)
#: Legacy source assets the npz was extracted from (vaft/data/README.md).
LEGACY_SOURCES = {
    "1909": "0f0d34ea98a14c32791db7bf5804bce537782993ab1cd7a9ca809b62d925eddf",
    "2507": "71c10a410b4bb180d5366f1bb7191a1a14e9277142af2cd20246af181f5b6830",
}
MATRIX_SHAPES = {
    "mutual_active_active_1906": (10, 10),
    "mutual_passive_active_1906": (950, 10),
    "mutual_active_active_2507": (10, 10),
    "mutual_passive_active_2507": (950, 10),
    "mutual_passive_passive": (950, 950),
}
PASSIVE_KEY = "mutual_passive_passive"
#: The cross-material ratio must be one constant to this absolute tolerance.
RATIO_ATOL = 1.0e-12
#: Reciprocity to float64 round-off; the repaired asset reads exactly 0.
ROUNDOFF = 1.0e-13


class RepairRefused(ValueError):
    """The matrix is not the defect this script repairs."""


class AlreadySymmetric(RepairRefused):
    """Nothing to repair."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_asymmetry(matrix: np.ndarray) -> float:
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    return float(np.max(np.abs(matrix - matrix.T))) / scale if scale > 0.0 else 0.0


def material_groups(static_geometry: str | Path = STATIC_GEOMETRY) -> dict[float, np.ndarray]:
    """Loop indexes by resistivity, from the packaged static geometry.

    The coupling asset carries no material column; the static geometry does,
    as ``pf_passive.loop[i].resistivity``.  Exactly two materials are expected
    (SUS316LN and tungsten); anything else is not the machine this repair
    describes.
    """
    with gzip.open(static_geometry, "rt", encoding="utf-8") as handle:
        geometry = json.load(handle)
    loops = geometry["pf_passive"]["loop"]
    resistivity = np.array([float(loop["resistivity"]) for loop in loops])
    groups = {float(value): np.flatnonzero(resistivity == value) for value in np.unique(resistivity)}
    if len(groups) != 2:
        raise RepairRefused(
            f"expected two passive materials in {Path(static_geometry).name}, "
            f"found {len(groups)}: {sorted(groups)}"
        )
    return groups


def repair_passive_passive(
    matrix: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    ratio_atol: float = RATIO_ATOL,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Set the cross-material block to ``group_a``'s side and report what was found.

    ``group_a`` is the material whose one-sided factor is kept (SUS).  The
    input must be exactly the donor defect: finite, square, symmetric within
    each material block, and with ``M[a, b] / M[b, a].T`` equal to one constant
    everywhere.  A matrix that is already symmetric raises
    :class:`AlreadySymmetric` so a repeated run changes nothing.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RepairRefused(f"expected a square matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RepairRefused("matrix contains non-finite entries")
    a = np.asarray(group_a, dtype=int).reshape(-1)
    b = np.asarray(group_b, dtype=int).reshape(-1)
    if a.size == 0 or b.size == 0 or np.intersect1d(a, b).size or a.size + b.size != matrix.shape[0]:
        raise RepairRefused("the two groups must partition the loop index set")

    input_asymmetry = relative_asymmetry(matrix)
    if input_asymmetry == 0.0:
        raise AlreadySymmetric("matrix is already exactly symmetric")

    for name, group in (("a", a), ("b", b)):
        block = matrix[np.ix_(group, group)]
        within = float(np.max(np.abs(block - block.T)))
        if within != 0.0:
            raise RepairRefused(
                f"within-material block {name} is asymmetric by {within:.3g}; "
                "that is not the one-sided material factor this script repairs"
            )

    cross = matrix[np.ix_(a, b)]
    mirrored = matrix[np.ix_(b, a)].T
    if np.any(mirrored == 0.0) or np.any(cross == 0.0):
        raise RepairRefused("a zero cross-material entry leaves the factor undefined")
    ratio = cross / mirrored
    factor = float(np.median(ratio))
    if not np.allclose(ratio, factor, rtol=0.0, atol=ratio_atol):
        raise RepairRefused(
            "the cross-material ratio is not a single constant "
            f"(spread {float(ratio.min()):.12g} .. {float(ratio.max()):.12g}); "
            "the asymmetry is not the one-sided material factor this script repairs"
        )

    repaired = matrix.copy()
    repaired[np.ix_(b, a)] = cross.T
    output_asymmetry = relative_asymmetry(repaired)
    if output_asymmetry != 0.0:  # pragma: no cover - the assignment above is exact
        raise RepairRefused(f"repair left an asymmetry of {output_asymmetry:.3g}")
    report = {
        "input_asymmetry": input_asymmetry,
        "output_asymmetry": output_asymmetry,
        "material_factor": factor,
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "cross_entries_changed": int(2 * a.size * b.size) if factor != 1.0 else 0,
    }
    return repaired, report


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _range(indexes: np.ndarray) -> list[int]:
    return [int(indexes.min()), int(indexes.max())]


def build_provenance(
    *,
    source_path: str | Path,
    static_geometry: str | Path,
    report: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, Any]],
    active_active_asymmetry: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "generator": GENERATOR,
        "issues": list(ISSUES),
        "generated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "vaft_version": str(getattr(vaft, "__version__", "unknown")),
        "git_commit": _git_commit(),
        "source_sha256": sha256_file(source_path),
        "static_geometry": Path(static_geometry).name,
        "static_geometry_sha256": sha256_file(static_geometry),
        "repair": (
            "mutual_passive_passive cross-material block set to the SUS-side value; "
            "the donor kernel applied the SUS relative-permeability factor from the "
            "first conductor only (getMutualInductanceCoil.m tested Coil1_Material twice)"
        ),
        "convention": "sus_side",
        "passive_material_factor": float(report["material_factor"]),
        "groups": dict(groups),
        "input_asymmetry": float(report["input_asymmetry"]),
        "output_asymmetry": float(report["output_asymmetry"]),
        "cross_entries_changed": int(report["cross_entries_changed"]),
        "active_active_asymmetry": {k: float(v) for k, v in active_active_asymmetry.items()},
        "legacy_sources": dict(LEGACY_SOURCES),
        "keys_copied_verbatim": [key for key in MATRIX_SHAPES if key != PASSIVE_KEY],
    }


def read_asset(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    matrices: dict[str, np.ndarray] = {}
    provenance = None
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key == PROVENANCE_KEY:
                provenance = json.loads(str(data[key][()]))
            else:
                matrices[key] = np.asarray(data[key], dtype=float)
    return matrices, provenance


def read_provenance(path: str | Path) -> dict[str, Any] | None:
    return read_asset(path)[1]


def write_asset(path: str | Path, arrays: Mapping[str, np.ndarray], provenance: Mapping[str, Any]) -> None:
    """Write atomically: a sibling temp file, then ``os.replace``."""
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    payload = {key: np.asarray(value, dtype=float) for key, value in arrays.items()}
    payload[PROVENANCE_KEY] = np.array(json.dumps(dict(provenance), sort_keys=True))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, target)


def verify_asset(
    path: str | Path = ASSET,
    *,
    static_geometry: str | Path = STATIC_GEOMETRY,
    require_provenance: bool = True,
) -> dict[str, Any]:
    """Refuse an asset that is not what the loader promises; return its provenance."""
    matrices, provenance = read_asset(path)
    missing = sorted(set(MATRIX_SHAPES) - set(matrices))
    if missing:
        raise RepairRefused(f"{Path(path).name} lacks matrices {missing}")
    for key, shape in MATRIX_SHAPES.items():
        if matrices[key].shape != shape:
            raise RepairRefused(f"{key} has shape {matrices[key].shape}, expected {shape}")
        if not np.all(np.isfinite(matrices[key])):
            raise RepairRefused(f"{key} contains non-finite entries")
    for key in MATRIX_SHAPES:
        if key.startswith(("mutual_active_active", "mutual_passive_passive")):
            asymmetry = relative_asymmetry(matrices[key])
            if asymmetry > ROUNDOFF:
                raise RepairRefused(f"{key} violates reciprocity by {asymmetry:.3g} (relative)")
    groups = material_groups(static_geometry)
    if sum(g.size for g in groups.values()) != MATRIX_SHAPES[PASSIVE_KEY][0]:
        raise RepairRefused("static geometry loop count does not match the passive matrix")
    if require_provenance:
        if provenance is None:
            raise RepairRefused(f"{Path(path).name} carries no {PROVENANCE_KEY!r} record")
        for key in ("generator", "passive_material_factor", "source_sha256", "convention"):
            if key not in provenance:
                raise RepairRefused(f"provenance lacks {key!r}")
    return provenance or {}


def repair_asset(
    path: str | Path = ASSET,
    *,
    output: str | Path | None = None,
    static_geometry: str | Path = STATIC_GEOMETRY,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repair ``path`` (in place unless ``output``) and return the report."""
    matrices, existing = read_asset(path)
    if PASSIVE_KEY not in matrices:
        raise RepairRefused(f"{Path(path).name} carries no {PASSIVE_KEY}")
    groups = material_groups(static_geometry)
    # SUS316LN is the higher-resistivity material and the one the donor
    # applied its 1.04 factor to; its side is kept.
    sus_rho, w_rho = sorted(groups, reverse=True)
    sus, tungsten = groups[sus_rho], groups[w_rho]
    active_asymmetry = {
        key.rsplit("_", 1)[1]: relative_asymmetry(matrices[key])
        for key in matrices
        if key.startswith("mutual_active_active")
    }
    for version, asymmetry in active_asymmetry.items():
        if asymmetry > ROUNDOFF:
            raise RepairRefused(
                f"mutual_active_active_{version} is asymmetric by {asymmetry:.3g}; "
                "this script only repairs the passive-passive material factor"
            )
    repaired, report = repair_passive_passive(matrices[PASSIVE_KEY], sus, tungsten)
    provenance = build_provenance(
        source_path=path,
        static_geometry=static_geometry,
        report=report,
        groups={
            "sus": {"resistivity": sus_rho, "n": int(sus.size), "range": _range(sus)},
            "tungsten": {"resistivity": w_rho, "n": int(tungsten.size), "range": _range(tungsten)},
        },
        active_active_asymmetry=active_asymmetry,
    )
    if existing is not None:
        provenance["superseded_provenance"] = existing
    result = {
        "asset": str(path),
        "output": str(output or path),
        "sha256_before": sha256_file(path),
        "report": report,
        "provenance": provenance,
        "written": False,
    }
    if dry_run:
        return result
    arrays = dict(matrices)
    arrays[PASSIVE_KEY] = repaired
    write_asset(output or path, arrays, provenance)
    verify_asset(output or path, static_geometry=static_geometry)
    result["written"] = True
    result["sha256_after"] = sha256_file(output or path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", type=Path, default=ASSET, help="The coupling npz to verify or repair.")
    parser.add_argument("--static-geometry", type=Path, default=STATIC_GEOMETRY, help="Static geometry naming the loop materials.")
    parser.add_argument("--verify", action="store_true", help="Verify the asset (the default when --repair is absent).")
    parser.add_argument("--repair", action="store_true", help="Repair the passive-passive material factor.")
    parser.add_argument("--output", type=Path, default=None, help="Write the repaired asset here instead of in place.")
    parser.add_argument("--dry-run", action="store_true", help="With --repair: report, write nothing.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)
    if args.verify and args.repair:
        parser.error("--verify and --repair are exclusive")

    def emit(payload: Mapping[str, Any], text: str) -> None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str) if args.json else text)

    if not args.repair:
        try:
            provenance = verify_asset(args.asset, static_geometry=args.static_geometry)
        except RepairRefused as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        emit(
            {"asset": str(args.asset), "sha256": sha256_file(args.asset), "provenance": provenance},
            f"{args.asset.name}: every reciprocity-bound matrix symmetric to {ROUNDOFF:g}; "
            f"provenance by {provenance.get('generator')} on {provenance.get('generated')} "
            f"(material factor {provenance.get('passive_material_factor')})",
        )
        return 0

    try:
        result = repair_asset(
            args.asset, output=args.output, static_geometry=args.static_geometry, dry_run=args.dry_run
        )
    except AlreadySymmetric:
        emit({"asset": str(args.asset), "status": "already repaired"}, f"{args.asset.name}: already exactly reciprocal; nothing written")
        return 0
    except RepairRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    report = result["report"]
    emit(
        result,
        f"{args.asset.name}: input asymmetry {report['input_asymmetry']:.6e}, material factor "
        f"{report['material_factor']:.12g} on {report['n_a']} x {report['n_b']} cross entries, "
        f"output asymmetry {report['output_asymmetry']:g}\n"
        + (
            f"  written -> {result['output']}\n  sha256 {result['sha256_before']} -> {result['sha256_after']}"
            if result["written"]
            else "  dry run: nothing written"
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
