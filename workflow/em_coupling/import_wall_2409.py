#!/usr/bin/env python3
"""Import the VEST wall 2409 passive additions, with provenance (issue #956).

From shot 43017 the VEST passive structure has fifteen more conductors than
the 950 VAFT has always carried: SUS316LN elements 20 x 6 mm at Z = -1.164 m,
R = 0.24 .. 0.52 m (VFIT ``VEST_WallLimiterGeometry_ver_2409``, group 11). The
first 950 rows of that file are VAFT's existing loops, element for element.

This script extracts **only the fifteen additions** into
``vaft/data/geometry/VEST_passive_wall_2409.npz``; the 950-loop assets are not
touched, so everything already built on them keeps its hash. The asset holds:

``loops``
    the fifteen ``pf_passive.loop`` records (JSON), in VAFT's own conventions:
    name ``W12`` (VAFT names its wall groups ``W1``..``W11``; VFIT's group 11
    is new here and VFIT's group 12 is VAFT's ``W11``), a rectangular outline,
    ``area = w * h``, nominal SUS resistivity and the nominal hoop resistance
    ``rho * 2*pi*R_mean / A`` -- the value nine of VAFT's eleven regions carry
    (`vaft.machine_mapping.wall_resistance`); no fitted band factor exists for
    these elements.
``mutual_passive_passive_rows`` (15 x 965)
    the new rows against the 950 existing loops and each other.
``mutual_passive_active_1906`` / ``_2507`` (15 x 10)
    the new rows against the PF coils of each geometry.

Only the *geometry* comes from VFIT. Every coupling entry is computed the way
VAFT's own 950-loop asset evidently was, which this script re-establishes on
every run before trusting it: off-diagonal passive entries are the filament
Green function ``green_r`` times mu_r (1.04 when either conductor is SUS),
matching the shipped block to 3e-13; the self-term is
``mu_r * mu0 * R * (ln(8R / sqrt(A/pi)) - 7/4)``, matching the shipped
diagonal to round-off; passive-active entries are
``vaft.process.compute_mutual_passive_active`` on the coil geometry the PF
mapper builds, matching the shipped 1906 and 2507 blocks to 1e-13. VFIT's own
matrices (``MatrixWallandPF_Cl_sim_ver_2409`` for PF 1906, ``_2511`` for PF
2507, whose last fifteen elements equal 2409's) come from a different kernel
and self-term; they are compared, not copied, and a disagreement above
``DONOR_AGREEMENT`` is refused.

Every claim above is checked when the asset is built and again by
``--verify``, which needs the VFIT checkout for the source comparison and
otherwise checks the packaged asset's own invariants.

Usage::

    PYTHONPATH=. python workflow/em_coupling/import_wall_2409.py --verify
    PYTHONPATH=. python workflow/em_coupling/import_wall_2409.py --vfit-root ~/git/VFIT_VEST-Equilibrium-Code --write
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from vaft.data.resources import data_path

ASSET = data_path("geometry/VEST_passive_wall_2409.npz")
BASE_COUPLING = data_path("geometry/VEST_em_coupling_pf_versions.npz")
STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")
GENERATOR = "workflow/em_coupling/import_wall_2409.py"
ISSUE = "https://github.com/VEST-Tokamak/vaft/issues/956"
DEFAULT_VFIT_ROOT = Path("~/git/VFIT_VEST-Equilibrium-Code").expanduser()
SOURCES = {
    "wall_geometry": "Input_Geometry/VEST_WallLimiterGeometry_ver_2409.mat",
    "coupling_pf1906": "Input_Geometry/VEST_MatrixWallandPF_Cl_sim_ver_2409.mat",
    "coupling_pf2507": "Input_Geometry/VEST_MatrixWallandPF_Cl_sim_ver_2511.mat",
}
N_BASE = 950
N_NEW = 15
NAME = "W12"
MU0 = 4.0e-7 * np.pi
SUS_MU_R = 1.04
SUS_RESISTIVITY = 7.8e-7
#: VFIT WallGeometry columns: R, Z, width, height, angle, turns, material, group.
MATERIAL_SUS, NEW_GROUP = 2, 11
#: How far VFIT's rows for the new elements may sit from VAFT's before the
#: import is refused: the two kernels differ by up to 0.9 % on the existing
#: 950 loops (VFIT's self-term is ~13 % higher and is excluded).
DONOR_AGREEMENT = 0.01
#: A shot in each PF geometry era, to build that era's coil geometry.
PF_REFERENCE_SHOT = {"1906": 43017, "2507": 45968}
#: How closely the recomputation must reproduce the shipped 950-loop asset
#: before the same method is trusted for the new rows.
CONVENTION_TOLERANCE = 1e-11


def thin_ring_self_inductance(r: np.ndarray, area: np.ndarray, mu_r: np.ndarray) -> np.ndarray:
    """VAFT's passive self-term: a thin ring of equivalent wire radius sqrt(A/pi)."""
    return mu_r * MU0 * r * (np.log(8.0 * r / np.sqrt(area / np.pi)) - 1.75)


def nominal_resistance(r_mean: float, area: float, resistivity: float) -> float:
    # Operation order of vaft.machine_mapping.wall_resistance.nominal_resistance.
    return resistivity * (2.0 * np.pi * r_mean) / area


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_loops() -> list[dict[str, Any]]:
    with gzip.open(STATIC_GEOMETRY, "rt", encoding="utf-8") as handle:
        return json.load(handle)["pf_passive"]["loop"]


def _base_coupling() -> dict[str, np.ndarray]:
    with np.load(BASE_COUPLING) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "provenance"}


def _loop_record(row: np.ndarray) -> dict[str, Any]:
    r, z, width, height = (float(value) for value in row[:4])
    outline_r = [r - width / 2, r + width / 2, r + width / 2, r - width / 2]
    outline_z = [z - height / 2, z - height / 2, z + height / 2, z + height / 2]
    area = width * height
    return {
        "element": [
            {
                "area": area,
                "geometry": {"geometry_type": 1, "outline": {"r": outline_r, "z": outline_z}},
                "identifier": NAME,
                "turns_with_sign": 1.0,
            }
        ],
        "name": NAME,
        "resistance": nominal_resistance(float(np.mean(outline_r)), area, SUS_RESISTIVITY),
        "resistivity": SUS_RESISTIVITY,
    }


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"refused: {message}")


def _base_geometry(loops: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.array([np.mean(loop["element"][0]["geometry"]["outline"]["r"]) for loop in loops])
    z = np.array([np.mean(loop["element"][0]["geometry"]["outline"]["z"]) for loop in loops])
    area = np.array([loop["element"][0]["area"] for loop in loops])
    return r, z, area


def _mu_r(names: list[str]) -> np.ndarray:
    return np.array([1.0 if name == "W11" else SUS_MU_R for name in names])


def passive_rows(
    r: np.ndarray, z: np.ndarray, area: np.ndarray, names: list[str], rows: np.ndarray
) -> np.ndarray:
    """VAFT's passive-passive entries for the loops *rows* against every loop."""
    import warnings

    from vaft.formula.green import green_r

    mu_r = _mu_r(names)
    out = np.empty((rows.size, r.size))
    for k, i in enumerate(rows):
        with warnings.catch_warnings():
            # green_r returns nan (and says so) at its own source point, which
            # the self-term below replaces.
            warnings.simplefilter("ignore", RuntimeWarning)
            response = np.asarray(green_r(r, z, r[i], z[i]), dtype=float)
        factor = np.where((mu_r == SUS_MU_R) | (mu_r[i] == SUS_MU_R), SUS_MU_R, 1.0)
        out[k] = factor * response
        out[k, i] = thin_ring_self_inductance(r[i : i + 1], area[i : i + 1], mu_r[i : i + 1])[0]
    return out


def passive_active_rows(
    r: np.ndarray, z: np.ndarray, names: list[str], rows: np.ndarray, pf_version: str
) -> np.ndarray:
    """VAFT's passive-active entries for the loops *rows*, PF geometry *pf_version*."""
    from omas import ODS

    from vaft.machine_mapping.pf_active import pf_geometry_version_for_shot, vfit_pf_active_static
    from vaft.process import compute_mutual_passive_active

    shot = PF_REFERENCE_SHOT[pf_version]
    _check(pf_geometry_version_for_shot(shot) == pf_version, f"shot {shot} is not PF {pf_version}")
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=shot)
    coils = [
        [
            (
                ods[f"pf_active.coil.{c}.element.{e}.geometry.rectangle.r"],
                ods[f"pf_active.coil.{c}.element.{e}.geometry.rectangle.z"],
                ods[f"pf_active.coil.{c}.element.{e}.turns_with_sign"],
            )
            for e in range(len(ods[f"pf_active.coil.{c}.element"]))
        ]
        for c in range(len(ods["pf_active.coil"]))
    ]
    mu_r = _mu_r(names)
    loops = [(names[i], float(r[i]), float(z[i]), float(mu_r[i])) for i in rows]
    return compute_mutual_passive_active(loops, coils)


def check_convention() -> dict[str, float]:
    """Recompute a sample of the shipped 950-loop asset; refuse if it misses."""
    loops = _base_loops()
    names = [loop["name"] for loop in loops]
    r, z, area = _base_geometry(loops)
    base = _base_coupling()
    sample = np.unique(np.r_[np.arange(0, N_BASE, 37), N_BASE - 1])
    worst = {}
    recomputed = passive_rows(r, z, area, names, sample)
    worst["passive_passive"] = float(
        np.max(np.abs(recomputed / base["mutual_passive_passive"][sample] - 1.0))
    )
    worst["self_term"] = float(
        np.max(
            np.abs(
                thin_ring_self_inductance(r, area, _mu_r(names))
                / np.diag(base["mutual_passive_passive"])
                - 1.0
            )
        )
    )
    for version in PF_REFERENCE_SHOT:
        recomputed = passive_active_rows(r, z, names, sample, version)
        worst[f"passive_active_{version}"] = float(
            np.max(np.abs(recomputed / base[f"mutual_passive_active_{version}"][sample] - 1.0))
        )
    for key, value in worst.items():
        _check(value < CONVENTION_TOLERANCE, f"{key}: VAFT's method misses its own asset by {value:.3g}")
    return worst


def assemble(new_loops: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """The asset's coupling arrays for *new_loops*, in VAFT's convention."""
    loops = _base_loops() + list(new_loops)
    names = [loop["name"] for loop in loops]
    r, z, area = _base_geometry(loops)
    rows = np.arange(N_BASE, N_BASE + len(new_loops))
    passive = passive_rows(r, z, area, names, rows)
    # green_r is reciprocal to round-off only; the base block is exact, so the
    # new block is made exact too (em_coupling mirrors the cross block itself).
    block = passive[:, N_BASE:]
    passive[:, N_BASE:] = 0.5 * (block + block.T)
    arrays = {"mutual_passive_passive_rows": passive}
    for version in PF_REFERENCE_SHOT:
        arrays[f"mutual_passive_active_{version}"] = passive_active_rows(r, z, names, rows, version)
    return arrays


def build(vfit_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from scipy.io import loadmat

    paths = {key: vfit_root / relative for key, relative in SOURCES.items()}
    for key, path in paths.items():
        _check(path.exists(), f"{key} source {path} not found")
    geometry = np.asarray(loadmat(paths["wall_geometry"])["WallGeometry"], dtype=float)
    matrix_1906 = loadmat(paths["coupling_pf1906"])
    wall_pf2507 = loadmat(paths["coupling_pf2507"], squeeze_me=True, struct_as_record=False)["VESTGeometry"]
    l_2409 = np.asarray(matrix_1906["InductanceWall"], dtype=float)
    c_1906 = np.asarray(matrix_1906["InductanceWallCoil"], dtype=float)
    c_2507 = np.asarray(wall_pf2507.WallCoilM, dtype=float)
    geometry_2511 = np.asarray(wall_pf2507.Wall, dtype=float)

    n_total = N_BASE + N_NEW
    _check(geometry.shape == (n_total, 8), f"wall 2409 geometry has shape {geometry.shape}")
    _check(l_2409.shape == (n_total, n_total), f"2409 passive matrix has shape {l_2409.shape}")
    _check(c_1906.shape == (n_total, 10) and c_2507.shape == (n_total, 10), "coil blocks are not 965 x 10")

    loops = _base_loops()
    r, z, area = _base_geometry(loops)
    _check(
        np.allclose(geometry[:N_BASE, 0], r, rtol=0, atol=1e-12)
        and np.allclose(geometry[:N_BASE, 1], z, rtol=0, atol=1e-12)
        and np.allclose(geometry[:N_BASE, 2] * geometry[:N_BASE, 3], area, rtol=1e-12, atol=0),
        "the first 950 wall 2409 elements are not VAFT's loops in VAFT's order",
    )
    new = geometry[N_BASE:]
    _check(
        bool(np.all(new[:, 6] == MATERIAL_SUS) and np.all(new[:, 7] == NEW_GROUP)),
        "the 15 additions are not all SUS group 11",
    )
    _check(
        np.array_equal(geometry_2511[N_BASE:], new),
        "the 2511 source's last 15 elements differ from 2409's; its PF 2507 rows are not theirs",
    )

    new_loops = [_loop_record(row) for row in new]
    arrays = assemble(new_loops)
    rows = arrays["mutual_passive_passive_rows"]
    # VFIT's new-SUS x tungsten entries lack the SUS factor: its kernel tested
    # one conductor's material where it meant both, the defect #373 repaired
    # in VAFT's base block (every such entry is exactly 1/1.04 of VAFT's).
    # Put the factor back before comparing, so the check measures geometry.
    tungsten = np.array([loop["name"] == "W11" for loop in loops])
    donor_pp = l_2409[N_BASE:, :N_BASE] * np.where(tungsten, SUS_MU_R, 1.0)
    donor_offsets = {}
    for label, donor, ours in (
        ("passive_passive", donor_pp, rows[:, :N_BASE]),
        ("passive_active_1906", c_1906[N_BASE:], arrays["mutual_passive_active_1906"]),
        ("passive_active_2507", c_2507[N_BASE:], arrays["mutual_passive_active_2507"]),
    ):
        donor_offsets[label] = float(np.max(np.abs(donor / ours - 1.0)))
    new_block = rows[:, N_BASE:]
    off = ~np.eye(N_NEW, dtype=bool)
    donor_offsets["passive_passive_new_block"] = float(
        np.max(np.abs(l_2409[N_BASE:, N_BASE:][off] / new_block[off] - 1.0))
    )
    for label, value in donor_offsets.items():
        _check(value < DONOR_AGREEMENT, f"VFIT {label} differs from VAFT's by {value:.3g}")

    provenance = {
        "generator": GENERATOR,
        "issue": ISSUE,
        "first_shot": 43017,
        "name": NAME,
        "vfit_group": NEW_GROUP,
        "sources": {
            key: {"path": SOURCES[key], "sha256": _sha256(path)} for key, path in paths.items()
        },
        "vfit_commit": _git_head(vfit_root),
        "convention": {
            "geometry": "VFIT WallLimiterGeometry_ver_2409 rows 951-965 (group 11, SUS)",
            "passive_passive": "mu_r*green_r filament, mu_r=1.04 if either loop is SUS (as the shipped 950 block)",
            "self_term": "mu_r*mu0*R*(ln(8R/sqrt(A/pi))-7/4), mu_r=1.04 (as the shipped diagonal)",
            "passive_active": "vaft.process.compute_mutual_passive_active on the PF mapper's coil geometry",
            "resistance": "nominal hoop resistance rho*2*pi*R_mean/A, rho=7.8e-7 (no fitted band)",
        },
        "vfit_matrix_agreement": donor_offsets,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    arrays = {
        "loops": np.asarray(json.dumps(new_loops, sort_keys=True)),
        **arrays,
        "provenance": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    return arrays, provenance


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_asset(path: Path = ASSET) -> list[str]:
    """The packaged asset's own invariants; returns the problems found."""
    problems: list[str] = []
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    loops = json.loads(str(arrays["loops"]))
    rows = arrays["mutual_passive_passive_rows"]
    if len(loops) != N_NEW or rows.shape != (N_NEW, N_BASE + N_NEW):
        problems.append(f"{len(loops)} loops and rows {rows.shape}, expected {N_NEW} and (15, 965)")
        return problems
    for key in ("mutual_passive_active_1906", "mutual_passive_active_2507"):
        if arrays[key].shape != (N_NEW, 10) or not np.all(np.isfinite(arrays[key])):
            problems.append(f"{key} is not a finite 15 x 10 block")
    block = rows[:, N_BASE:]
    if float(np.max(np.abs(block - block.T))) != 0.0:
        problems.append("the new-new block is not exactly reciprocal")
    r = np.array([np.mean(loop["element"][0]["geometry"]["outline"]["r"]) for loop in loops])
    area = np.array([loop["element"][0]["area"] for loop in loops])
    expected = thin_ring_self_inductance(r, area, np.full(N_NEW, SUS_MU_R))
    if not np.allclose(np.diag(block), expected, rtol=1e-14, atol=0):
        problems.append("the new diagonal is not VAFT's self-term")
    for loop, r_mean, a in zip(loops, r, area):
        if loop["name"] != NAME or loop["resistance"] != nominal_resistance(r_mean, a, SUS_RESISTIVITY):
            problems.append(f"loop {loop['name']} does not carry the nominal SUS hoop resistance")
            break
    provenance = json.loads(str(arrays["provenance"]))
    for key in ("generator", "issue", "sources", "convention", "first_shot"):
        if key not in provenance:
            problems.append(f"provenance lacks {key!r}")
    # Everything but the geometry is VAFT's own arithmetic, so it can be
    # recomputed from the asset's loops without the VFIT checkout.
    for key, value in assemble(loops).items():
        if not np.allclose(arrays[key], value, rtol=1e-12, atol=0):
            problems.append(f"{key} does not recompute from the asset's own loops")
    return problems


def _serialize(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vfit-root", type=Path, default=DEFAULT_VFIT_ROOT)
    parser.add_argument("--write", action="store_true", help="(re)build the asset from the VFIT sources")
    parser.add_argument("--verify", action="store_true", help="check the packaged asset (default)")
    args = parser.parse_args(argv)

    worst = check_convention()
    print(
        "VAFT's coupling method reproduces the shipped 950-loop asset: "
        + ", ".join(f"{key} {value:.1e}" for key, value in worst.items())
    )
    if args.write:
        arrays, _provenance = build(args.vfit_root)
        ASSET.write_bytes(_serialize(arrays))
        print(f"wrote {ASSET}")
    problems = verify_asset(ASSET)
    if args.vfit_root.exists() and not args.write:
        arrays, _provenance = build(args.vfit_root)
        with np.load(ASSET) as data:
            for key in ("mutual_passive_passive_rows", "mutual_passive_active_1906", "mutual_passive_active_2507"):
                if not np.array_equal(data[key], arrays[key]):
                    problems.append(f"{key} does not match a rebuild from {args.vfit_root}")
            if str(data["loops"]) != str(arrays["loops"]):
                problems.append(f"loops do not match a rebuild from {args.vfit_root}")
        print(f"rebuilt from {args.vfit_root} and compared; VFIT matrix agreement: {_provenance['vfit_matrix_agreement']}")
    for problem in problems:
        print(f"problem: {problem}")
    if problems:
        return 2
    print(f"{ASSET.name}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
