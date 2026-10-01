"""Build the n-resolved Tier A stability atlas from the batch run directories (#1429).

Reads what ``run_batch.py`` left under ``OUT/<lineage>/<shot>.<time_ms>/`` and writes:

``atlas_n.csv``
    One row per (shot, time_efit_s, efit_lineage, n_tor): DCON ideal energies
    for both edge treatments, Mercier/ballooning screening, the per-solver Δ′
    summaries over *resolved* surfaces only, and a status column per solver.
``atlas_surfaces.csv``
    One row per (shot, time_efit_s, efit_lineage, n_tor, solver, m): the
    diagonal Δ′ at mpsi 256 and 512, its convergence, and (RDCON only) the GGJ
    D_I, D_R, H and ballooning C_A at the surface.
``schema.json``
    The column dictionary, units, and the rules a consumer must keep.

Conventions (decided in #141, #792 and #1429):

* DCON energies are DCON's normalized least-stable eigenvalues, not Joules.
  ``ideal_stable`` is ``W_t > 0`` at mpsi 256. It is *resolved* only when the
  mpsi 512 run gives the same sign.
* Full-edge (``psiedge=1``) and peak-dW-truncated (``psiedge=0.95``) results
  are separate columns and are never combined. Full-edge W_t moves by up to
  100x with ``psihigh``.
* A surface's Δ′ is ``resolved`` when mpsi 256 and 512 agree in sign and to
  ``DELTA_PRIME_RTOL`` relative. Per-n Δ′ summaries use only resolved
  *interior* surfaces. The first and last rational surface of each run are
  excluded even when resolved: #141 found them unresolved on every reference
  case, and two mpsi values can agree by coincidence (a "resolved" first
  surface reached Δ′ = 6.5e5 in the batch). Every surface stays in the
  surface table with its flags. RDCON and STRIDE are reported side by side and
  never combined.
* Surfaces are matched across mpsi by poloidal mode number m at fixed n,
  with |Δψ_N| <= ``SURFACE_PSI_TOL`` required. No positional matching.
* Nothing is combined across n.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_slices import read as read_slices  # noqa: E402

#: #141: 20 % separates the converged mid-radius surfaces from the
#: unresolved first/last surfaces on all three reference equilibria.
DELTA_PRIME_RTOL = 0.20
#: Same rational surface in two runs of one equilibrium: ψ_N agrees to ~1e-5
#: in practice; 1e-3 still separates neighbouring surfaces (spacing > 5e-3).
SURFACE_PSI_TOL = 1e-3
ATLAS_VERSION = 1

ZERO_CROSSING = re.compile(r"Zero crossing at psi =\s*([-+0-9.Ee]+), q =\s*([-+0-9.Ee]+)")

STATUS_VALUES = (
    "VALID_STABLE",
    "VALID_UNSTABLE",
    "MARGINAL",
    "NUMERICALLY_UNRESOLVED",
    "INVALID_EQUILIBRIUM",
    "SOLVER_FAILURE",
    "NOT_APPLICABLE",
)


def _result(job: Path) -> Optional[dict]:
    path = job / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def _ok(result: Optional[dict]) -> bool:
    return result is not None and result.get("status") in ("completed", "stable")


def _real(value: Any) -> Optional[float]:
    return None if value is None else float(np.real(value))


def zero_crossings(run_dir: Path) -> list[tuple[float, float]]:
    """DCON's confirmed Newcomb zero crossings, ``(psi_n, q)``, from ``dcon.out``."""
    out = run_dir / "dcon.out"
    if not out.exists():
        return []
    return [(float(a), float(b)) for a, b in ZERO_CROSSING.findall(out.read_text(errors="replace"))]


def _extremum(values: Optional[np.ndarray], psi: Optional[np.ndarray], mask: Optional[np.ndarray], *, largest: bool):
    if values is None or psi is None:
        return None, None
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v) if mask is None else (np.isfinite(v) & np.asarray(mask, dtype=bool))
    if not keep.any():
        return None, None
    index = np.flatnonzero(keep)[int(np.argmax(v[keep]) if largest else np.argmin(v[keep]))]
    return float(v[index]), float(np.asarray(psi)[index])


def dcon_columns(result: Optional[dict], prefix: str) -> dict[str, Any]:
    """Scalar DCON columns for one run; ``prefix`` names the treatment/resolution."""
    from vaft.code.gpec import read_dcon_output

    columns: dict[str, Any] = {f"{prefix}_status_raw": None if result is None else result.get("status")}
    if not _ok(result):
        return columns
    run_dir = Path(result["run_dir"])
    out = read_dcon_output(run_dir, mode=int(result["n"]))
    crossings = zero_crossings(run_dir)
    columns.update(
        {
            f"{prefix}_W_t": _real(out.total1),
            f"{prefix}_W_p": _real(out.plasma1),
            f"{prefix}_W_v": _real(out.vacuum1),
            f"{prefix}_W_v_imag_ratio": None
            if out.vacuum1 is None or out.vacuum1.real == 0
            else abs(out.vacuum1.imag / out.vacuum1.real),
            f"{prefix}_m_dominant": out.m_pol_dominant,
            f"{prefix}_psilim": out.psilim,
            f"{prefix}_qlim": out.qlim,
            f"{prefix}_edge_treatment": out.edge_treatment,
            f"{prefix}_n_zero_crossings": len(crossings),
            f"{prefix}_first_zero_crossing_psi_n": crossings[0][0] if crossings else None,
        }
    )
    if prefix == "dcon_full_256":
        # Local criteria depend on the equilibrium, not on the edge treatment
        # or n; they are read once, from the reference run.
        psi = None if out.psi_n is None else np.asarray(out.psi_n, dtype=float)
        max_di, psi_di = _extremum(out.di, psi, None, largest=True)
        max_dr, psi_dr = _extremum(out.dr, psi, None, largest=True)
        evaluated = None if out.ca1_evaluated is None else np.asarray(out.ca1_evaluated, dtype=bool)
        min_ca, psi_ca = _extremum(out.ca1, psi, evaluated, largest=False)
        columns.update(
            {
                "max_D_I": max_di,
                "psi_n_at_max_D_I": psi_di,
                "max_D_R": max_dr,
                "psi_n_at_max_D_R": psi_dr,
                "min_C_A": min_ca,
                "psi_n_at_min_C_A": psi_ca,
                "mercier_evaluated": bool(out.evaluation.mer_flag) if out.evaluation else None,
                "ballooning_evaluated": None if evaluated is None else bool(evaluated.any()),
                # Sign rules from #142: D_I > 0 is ideal-interchange unstable,
                # D_R > 0 resistive-interchange unstable, C_A < 0 ballooning unstable.
                "ideal_interchange_unstable": None if max_di is None else bool(max_di > 0),
                "resistive_interchange_unstable": None if max_dr is None else bool(max_dr > 0),
                "ballooning_unstable": None if min_ca is None else bool(min_ca < 0),
            }
        )
    return columns


def dcon_status(w_256: Optional[float], w_512: Optional[float], raw_256: Optional[str]) -> str:
    if w_256 is None:
        return "SOLVER_FAILURE" if raw_256 is not None else "NOT_APPLICABLE"
    if w_512 is None or np.sign(w_256) != np.sign(w_512):
        return "NUMERICALLY_UNRESOLVED"
    return "VALID_STABLE" if w_256 > 0 else "VALID_UNSTABLE"


def matching_surfaces(result: Optional[dict], solver: str) -> list[dict]:
    """Per-surface diagonal Δ′ (and GGJ for RDCON) from one matching run."""
    from vaft.code.gpec import read_pest3_matching_output

    if not _ok(result):
        return []
    run_dir = Path(result["run_dir"])
    n = int(result["n"])
    out = read_pest3_matching_output(run_dir, solver=solver, mode=n)
    rows = out.delta_prime_diagonal()
    if solver == "rdcon":
        ggj = rdcon_ggj(run_dir / f"rdcon_output_n{n}.nc")
        for row in rows:
            row.update(ggj_at(ggj, row["psi_n"]))
    return rows


def rdcon_ggj(path: Path) -> Optional[dict[str, np.ndarray]]:
    """RDCON's Glasser-Greene-Johnson profiles on its own ψ_N grid.

    RDCON writes ``di`` (ideal Mercier D_I), ``dr`` (resistive D_R), ``h``
    (Glasser's H) and ``ca1`` (ballooning C_A) as 1-D profiles of ``psi_n`` in
    ``rdcon_output_n<n>.nc``. They are evaluated at each rational surface by
    linear interpolation in ψ_N.
    """
    if not path.exists():
        return None
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if "psi_n" not in ds.variables:
            return None
        return {k: np.asarray(ds.variables[k][:], dtype=float) for k in ("psi_n", "di", "dr", "h", "ca1") if k in ds.variables}


def ggj_at(ggj: Optional[dict[str, np.ndarray]], psi_n: float) -> dict[str, Optional[float]]:
    if ggj is None:
        return {"D_I": None, "D_R": None, "H": None, "C_A": None}
    grid = ggj["psi_n"]
    names = {"di": "D_I", "dr": "D_R", "h": "H", "ca1": "C_A"}
    return {names[k]: float(np.interp(psi_n, grid, ggj[k])) if k in ggj else None for k in names}


def pair_surfaces(primary: list[dict], check: list[dict]) -> list[dict]:
    """Attach the mpsi-512 Δ′ to each mpsi-256 surface, matched by m and ψ_N."""
    by_m = {row["m"]: row for row in check}
    paired = []
    for index, row in enumerate(primary):
        partner = by_m.get(row["m"])
        if partner is not None and abs(partner["psi_n"] - row["psi_n"]) > SURFACE_PSI_TOL:
            partner = None
        a = row["delta_prime_real"]
        b = None if partner is None else partner["delta_prime_real"]
        relative = None if b is None else abs(a - b) / max(abs(a), abs(b), 1e-12)
        resolved = b is not None and np.sign(a) == np.sign(b) and relative <= DELTA_PRIME_RTOL
        paired.append(
            {
                **row,
                "delta_prime_check": b,
                "delta_prime_relative_change": relative,
                "resolved": bool(resolved),
                "position": "first" if index == 0 else "last" if index == len(primary) - 1 else "interior",
            }
        )
    return paired


def summary_eligible(surface: dict) -> bool:
    return surface["resolved"] and surface["position"] == "interior"


def matching_summary(surfaces: list[dict], status_raw: Optional[str], prefix: str) -> dict[str, Any]:
    resolved = [s for s in surfaces if summary_eligible(s)]
    best = max(resolved, key=lambda s: s["delta_prime_real"], default=None)
    if not surfaces:
        status = "SOLVER_FAILURE" if status_raw is not None else "NOT_APPLICABLE"
    elif not resolved:
        status = "NUMERICALLY_UNRESOLVED"
    else:
        # Δ′ alone is not a tearing verdict (#939): no stable/unstable label.
        status = "VALID_UNSTABLE" if best["delta_prime_real"] > 0 else "VALID_STABLE"
    return {
        f"{prefix}_status": status,
        f"{prefix}_n_rational_surfaces": len(surfaces),
        f"{prefix}_n_resolved_surfaces": sum(s["resolved"] for s in surfaces),
        f"{prefix}_n_summary_surfaces": len(resolved),
        f"{prefix}_n_positive_delta_prime_summary": sum(s["delta_prime_real"] > 0 for s in resolved),
        f"{prefix}_delta_prime_max": None if best is None else best["delta_prime_real"],
        f"{prefix}_m_at_delta_prime_max": None if best is None else best["m"],
        f"{prefix}_psi_n_at_delta_prime_max": None if best is None else best["psi_n"],
        f"{prefix}_q_at_delta_prime_max": None if best is None else best["q"],
    }


def slice_rows(base: Path, row: dict, modes: Iterable[int], provenance: dict) -> tuple[list[dict], list[dict]]:
    label = f"{row['shot']}.{row['time_ms']:05d}"
    slice_dir = base / row["efit_lineage"] / label
    key = {
        "shot": row["shot"],
        "time_efit_s": row["time_efit_s"],
        "efit_lineage": row["efit_lineage"],
        "efit_label": row["efit_label"],
        "efit_setting": row["efit_setting"],
        "kinetic_chi2": row["kinetic_chi2"],
    }
    atlas, surfaces = [], []
    for n in modes:
        job = lambda variant: _result(slice_dir / variant / f"nn{n}")  # noqa: E731
        record: dict[str, Any] = {**key, "n_tor": n}
        full_256, full_512, trunc = job("dcon_mpsi256"), job("dcon_mpsi512"), job("dcon_trunc_mpsi256")
        record.update(dcon_columns(full_256, "dcon_full_256"))
        record.update(dcon_columns(full_512, "dcon_full_512"))
        record.update(dcon_columns(trunc, "dcon_trunc_256"))
        w256, w512 = record.get("dcon_full_256_W_t"), record.get("dcon_full_512_W_t")
        record["dcon_full_status"] = dcon_status(w256, w512, record["dcon_full_256_status_raw"])
        record["ideal_stable_full_edge"] = None if w256 is None else bool(w256 > 0)
        wt = record.get("dcon_trunc_256_W_t")
        record["dcon_trunc_status"] = (
            ("VALID_STABLE" if wt > 0 else "VALID_UNSTABLE") if wt is not None
            else ("SOLVER_FAILURE" if record["dcon_trunc_256_status_raw"] is not None else "NOT_APPLICABLE")
        )
        record["ideal_stable_truncated_edge"] = None if wt is None else bool(wt > 0)
        for solver in ("rdcon", "stride"):
            primary, check = job(f"{solver}_mpsi256"), job(f"{solver}_mpsi512")
            paired = pair_surfaces(matching_surfaces(primary, solver), matching_surfaces(check, solver))
            record.update(matching_summary(paired, None if primary is None else primary.get("status"), solver))
            for s in paired:
                surfaces.append(
                    {
                        **key,
                        "n_tor": n,
                        "solver": solver,
                        "m": s["m"],
                        "psi_n_s": s["psi_n"],
                        "q_s": s["q"],
                        "delta_prime": s["delta_prime_real"],
                        "delta_prime_imag": s["delta_prime_imag"],
                        "delta_prime_mpsi512": s["delta_prime_check"],
                        "delta_prime_relative_change": s["delta_prime_relative_change"],
                        "resolved": s["resolved"],
                        "position": s["position"],
                        "in_summary": summary_eligible(s),
                        "D_I": s.get("D_I"),
                        "D_R": s.get("D_R"),
                        "H": s.get("H"),
                        "C_A": s.get("C_A"),
                        "run_dir": None if primary is None else primary["run_dir"],
                    }
                )
        record.update(provenance)
        record["slice_dir"] = str(slice_dir)
        atlas.append(record)
    return atlas, surfaces


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_csv(rows: list[dict], path: Path) -> Path:
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


SCHEMA_NOTES = {
    "key": ["shot", "time_efit_s", "efit_lineage", "n_tor"],
    "key_contract": "provisional (shot, time_efit_s, efit_lineage) per lane log #1448, until lane K publishes its State key contract",
    "efit_label": "#1331 criteria label of the magnetics-only EFIT at this time: good | admissible. Unreconstructible slices are never present.",
    "efit_lineage": "magnetics-only (statistical_891 magnetic EFIT) | electron-kinetic (electron EFIT, label inherited from the magnetics-only slice at exactly the same time)",
    "equilibrium_input": "EFIT g-file refined by CHEASE with pipeline 1's run_chease_refinement.py defaults (target_psin 0.993, nw 513)",
    "energies": "DCON least-stable normalized eigenvalues (not Joules). Positive = stable.",
    "edge_treatments": "dcon_full_* = psiedge 1 (full edge, psihigh 0.994); dcon_trunc_256 = psiedge 0.95 (truncated at the dW_edge peak). Never combine. Full-edge W_t is psihigh-sensitive (#141, #792).",
    "resolution": "mpsi 256 is the reported value; mpsi 512 is the convergence check (#141).",
    "delta_prime": f"Diagonal classical Δ′ (RDCON/STRIDE PEST3 matching matrix). A surface is resolved when mpsi 256 and 512 agree in sign and within {DELTA_PRIME_RTOL:.0%}. Per-n summaries use resolved interior surfaces only (first/last rational surface excluded; in_summary column). RDCON and STRIDE are never combined.",
    "status": list(STATUS_VALUES),
    "status_semantics": "dcon_*_status: sign of W_t, NUMERICALLY_UNRESOLVED when mpsi 256/512 disagree in sign. rdcon/stride_status: VALID_UNSTABLE if any resolved surface has Δ′ > 0 (classical Δ′ only, not a tearing verdict; RMATCH out of scope, #797).",
    "local_criteria": "max_D_I > 0 ideal interchange, max_D_R > 0 resistive interchange, min_C_A < 0 ballooning (evaluated surfaces only); from DCON at mpsi 256.",
    "zero_crossings": "Confirmed Newcomb zero crossings from dcon.out (termbycross_flag=f).",
    "rules": ["no cross-n combination", "no DCON/RDCON combined scalar", "do not drop the efit_label/efit_lineage columns"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True, help="run_batch.py output directory")
    parser.add_argument("--out", type=Path, required=True, help="atlas output directory")
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--gpec-home", type=Path, required=True)
    parser.add_argument("--gpec-patch", default="", help="Describe any local GPEC patch used by some runs.")
    args = parser.parse_args()

    import vaft

    provenance = {
        "atlas_version": ATLAS_VERSION,
        "vaft_commit": _git(Path(vaft.__file__).resolve().parents[1], "rev-parse", "--short", "HEAD"),
        "gpec_commit": _git(args.gpec_home, "rev-parse", "--short", "HEAD"),
        "gpec_patch": args.gpec_patch,
    }
    batch = args.batch.expanduser().resolve()
    rows = read_slices(batch / "slices.csv")
    atlas, surfaces = [], []
    for row in rows:
        a, s = slice_rows(batch, row, args.modes, provenance)
        atlas.extend(a)
        surfaces.extend(s)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(atlas, out / "atlas_n.csv")
    write_csv(surfaces, out / "atlas_surfaces.csv")
    schema = {
        **SCHEMA_NOTES,
        "provenance": provenance,
        "atlas_n_columns": sorted({k for r in atlas for k in r}),
        "atlas_surfaces_columns": sorted({k for r in surfaces for k in r}),
        "rows": {"atlas_n": len(atlas), "atlas_surfaces": len(surfaces)},
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=1))
    print(f"{len(atlas)} atlas rows, {len(surfaces)} surface rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
