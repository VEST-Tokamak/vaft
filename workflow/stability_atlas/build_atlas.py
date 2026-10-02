"""Build the n-resolved Tier A stability atlas from the batch run directories (#1429).

Reads what ``run_batch.py`` left under ``OUT/<lineage>/<shot>.<time_ms>/`` and writes:

``atlas_n.csv``
    One row per (shot, time_efit_s, efit_lineage, n_tor): DCON ideal energies
    for both edge treatments, Mercier/ballooning screening, the per-solver Δ′
    summaries (raw and QA-filtered, under separate names), and a status
    column per solver.
``atlas_surfaces.csv``
    One row per (shot, time_efit_s, efit_lineage, n_tor, solver, m): the
    diagonal Δ′ at mpsi 256 and 512, its convergence, and (RDCON only) the GGJ
    D_I, D_R, H and ballooning C_A at the surface.
``schema.json``
    The column dictionary, units, and the rules a consumer must keep.

Atlas v2 (#1429 comment of 2026-10-02, maintainer-approved): physical
quantities are judged by universal criteria, and numerical QA only annotates
how far a result can be trusted. QA never redefines a quantity or removes a
solver result. Four layers, never overwriting each other:

1. **Raw** solver output: W_t per n, edge treatment and resolution; Δ′ per
   surface at mpsi 256 and 512; ``*_delta_prime_max_raw`` over every surface.
2. **QA annotation**: ``dcon_*_sign_class``; per surface ``sign_agreement``,
   ``relative_change``, ``absolute_change``, ``two_resolution_consistent``,
   ``ordinal_position``, local shear, distance to q_min and to the edge; and
   QA-filtered summaries under their own names (``*_qa``, ``*_qa_interior``).
3. **Universal physical criteria**: ideal stability is the sign of W_t (no
   magnitude band; #142); classical tearing drive is Δ′ > 0 at *any* rational
   surface; D_I > 0, D_R > 0, C_A < 0.
4. **Interpretation** (#939): Δ′ is an outer-region quantity. A tearing
   verdict needs an inner-layer treatment (RMATCH, #797), so this layer is
   not written here.

Conventions (decided in #141, #792 and #1429):

* DCON energies are DCON's normalized least-stable eigenvalues, not Joules.
  The ideal criterion is the sign of W_t. It is *robust* when the mpsi 256
  and 512 runs agree in sign. A small negative W_t whose sign is robust is
  ideal-unstable; there is no "marginal" magnitude band.
* Full-edge (``psiedge=1``) and peak-dW-truncated (``psiedge=0.95``) results
  are separate columns and are never combined. Full-edge W_t moves by up to
  100x with ``psihigh``.
* A surface's Δ′ is ``two_resolution_consistent`` when mpsi 256 and 512
  agree in sign and to ``DELTA_PRIME_RTOL`` relative. This is a heuristic QA
  flag, not a convergence proof. Ordinal position (first/interior/last) is a
  QA annotation, never an exclusion rule. Large |Δ′| is diagnosed (local
  shear, distance to q_min), not discarded. RDCON and STRIDE are reported
  side by side and never combined.
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

#: Heuristic: 20 % is the *median* relative change between mpsi 256 and 512
#: of #141's 64 mid-radius RDCON surfaces (0.19; 90th percentile 0.72), so
#: about half of ordinary interior surfaces fail it. It labels; it does not
#: define the quantity.
DELTA_PRIME_RTOL = 0.20
#: Same rational surface in two runs of one equilibrium: ψ_N agrees to ~1e-5
#: in practice; 1e-3 still separates neighbouring surfaces (spacing > 5e-3).
SURFACE_PSI_TOL = 1e-3
ATLAS_VERSION = 2

ZERO_CROSSING = re.compile(r"Zero crossing at psi =\s*([-+0-9.Ee]+), q =\s*([-+0-9.Ee]+)")

#: #142's enum, as used here. MARGINAL is reserved and never produced: no
#: tolerance band around W_t = 0 has been derived yet (#141 follow-up).
STATUS_VALUES = (
    "VALID_STABLE",
    "VALID_UNSTABLE",
    "MARGINAL",
    "NUMERICALLY_UNRESOLVED",
    "INVALID_EQUILIBRIUM",
    "SOLVER_FAILURE",
    "NOT_APPLICABLE",
)
#: Atlas extensions. NOT_CHECKED: the mpsi 512 check run is missing or
#: failed, so the QA check is unavailable (not "unresolved"). RESOLVED: a Δ′
#: column has at least one two-resolution-consistent surface, at any
#: position. Δ′ never gets a stable/unstable label (#939).
EXTENSION_VALUES = ("NOT_CHECKED", "RESOLVED")

#: QA classes of the sign of W_t (layer 2).
SIGN_CLASSES = ("robust_stable", "robust_unstable", "numerically_unresolved", "not_checked", "unavailable")


def _result(job: Path) -> Optional[dict]:
    path = job / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ok(result: Optional[dict]) -> bool:
    return result is not None and result.get("status") in ("completed", "stable")


def _real(value: Any) -> Optional[float]:
    return None if value is None else float(np.real(value))


def zero_crossings(run_dir: Path) -> list[tuple[float, float]]:
    """DCON's confirmed Newcomb zero crossings, ``(psi_n, q)``, from ``dcon.out``."""
    out = run_dir / "dcon.out"
    if not out.exists():
        return []
    return [(float(a), float(b)) for a, b in ZERO_CROSSING.findall(out.read_text(encoding="utf-8", errors="replace"))]


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
                "mercier_evaluated": None if out.evaluation is None else out.evaluation.mercier,
                "ballooning_evaluated": None
                if evaluated is None or out.evaluation is None
                else bool(out.evaluation.ballooning and evaluated.any()),
                # Sign rules from #142: D_I > 0 is ideal-interchange unstable,
                # D_R > 0 resistive-interchange unstable, C_A < 0 ballooning unstable.
                "ideal_interchange_unstable": None
                if max_di is None or not (out.evaluation and out.evaluation.mercier)
                else bool(max_di > 0),
                "resistive_interchange_unstable": None
                if max_dr is None or not (out.evaluation and out.evaluation.mercier)
                else bool(max_dr > 0),
                "ballooning_unstable": None if min_ca is None else bool(min_ca < 0),
            }
        )
    return columns


def dcon_status(
    w_256: Optional[float], w_512: Optional[float], raw_256: Optional[str], raw_512: Optional[str]
) -> str:
    """#142 status of one DCON treatment from its mpsi 256 run and mpsi 512 check."""
    if w_256 is None:
        return "SOLVER_FAILURE" if raw_256 is not None else "NOT_APPLICABLE"
    if w_512 is None:
        return "NOT_CHECKED"
    if np.sign(w_256) != np.sign(w_512):
        return "NUMERICALLY_UNRESOLVED"
    return "VALID_STABLE" if w_256 > 0 else "VALID_UNSTABLE"


def dcon_sign_class(w_256: Optional[float], w_512: Optional[float]) -> str:
    """Layer-2 QA of the sign of W_t across mpsi 256 and 512; no magnitude band (#142)."""
    if w_256 is None:
        return "unavailable"
    if w_512 is None:
        return "not_checked"
    if np.sign(w_256) != np.sign(w_512):
        return "numerically_unresolved"
    return "robust_stable" if w_256 > 0 else "robust_unstable"


def surface_geometry(
    psi_grid: Optional[np.ndarray], q_profile: Optional[np.ndarray], psi_s: float
) -> dict[str, Optional[float]]:
    """Layer-2 diagnostics of a rational surface from the solver's own q(ψ_N) profile.

    ``shear_psi`` is d ln q / d ln ψ_N at the surface; it is weak near q_min,
    where large |Δ′| clusters. ``psi_n_from_qmin`` and ``q_minus_qmin`` place
    the surface relative to the profile minimum. ``psi_n_to_edge`` is the
    distance to the end of the solver's profile grid (psilim).
    """
    empty = {"shear_psi": None, "psi_n_from_qmin": None, "q_minus_qmin": None, "psi_n_to_edge": None}
    if psi_grid is None or q_profile is None:
        return empty
    psi = np.asarray(psi_grid, dtype=float)
    q = np.asarray(q_profile, dtype=float)
    if psi.shape != q.shape or psi.size < 3 or not (psi[0] <= psi_s <= psi[-1]):
        return empty
    dq = np.gradient(q, psi)
    q_s = float(np.interp(psi_s, psi, q))
    imin = int(np.nanargmin(q))
    return {
        "shear_psi": float(psi_s * np.interp(psi_s, psi, dq) / q_s) if q_s else None,
        "psi_n_from_qmin": float(psi_s - psi[imin]),
        "q_minus_qmin": float(q_s - q[imin]),
        "psi_n_to_edge": float(psi[-1] - psi_s),
    }


def matching_surfaces(result: Optional[dict], solver: str) -> list[dict]:
    """Per-surface diagonal Δ′ (and GGJ for RDCON) from one matching run.

    The local criteria come from the library's
    :meth:`~vaft.code.gpec.Pest3MatchingOutput.rational_surface_stability`,
    which already applies RDCON's own rules: ``C_A`` is ``None`` where
    ``bal.f`` never evaluated the ballooning integral (it leaves ``ca1 = 0``
    where ``D_I > 0``), and every criterion is ``None`` for a surface outside
    the solver's ψ_N profile grid instead of a clamped value. Interpolating the
    raw netCDF profiles here reported those placeholder zeros as a marginal
    ``C_A`` (cold review 0.8.0 delta-absorb-6 F1/F2).
    """
    from vaft.code.gpec import read_pest3_matching_output

    if not _ok(result):
        return []
    run_dir = Path(result["run_dir"])
    n = int(result["n"])
    out = read_pest3_matching_output(run_dir, solver=solver, mode=n)
    rows = []
    for row in out.rational_surface_stability():
        if row["psi_n"] is None:
            continue
        criteria = {GGJ_COLUMNS[k]: row.pop(k) for k in GGJ_COLUMNS}
        if solver == "rdcon":
            row.update(criteria)
        row.update(surface_geometry(out.psi_n, out.q, row["psi_n"]))
        rows.append(row)
    return rows


#: Library profile name -> surface-table column (RDCON only; STRIDE writes no ``h``).
GGJ_COLUMNS = {"di": "D_I", "dr": "D_R", "h": "H", "ca1": "C_A"}


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
        agree = None if b is None else bool(np.sign(a) == np.sign(b))
        consistent = bool(agree and relative <= DELTA_PRIME_RTOL)
        position = "first" if index == 0 else "last" if index == len(primary) - 1 else "interior"
        paired.append(
            {
                **row,
                "delta_prime_check": b,
                "delta_prime_relative_change": relative,
                "delta_prime_absolute_change": None if b is None else abs(a - b),
                "sign_agreement": agree,
                "two_resolution_consistent": consistent,
                "resolved": consistent,  # deprecated v1 name
                "position": position,
            }
        )
    return paired


def summary_eligible(surface: dict) -> bool:
    """v1 rule, kept only for the deprecated ``*_delta_prime_max`` alias: consistent and interior."""
    return surface["two_resolution_consistent"] and surface["position"] == "interior"


def _extreme(surfaces: list[dict], prefix: str, suffix: str) -> dict[str, Any]:
    """Max Δ′ over ``surfaces`` and where it sits, under ``{prefix}_delta_prime_max{suffix}``."""
    best = max(surfaces, key=lambda s: s["delta_prime_real"], default=None)
    name = f"{prefix}_delta_prime_max{suffix}"
    return {
        name: None if best is None else best["delta_prime_real"],
        f"{name}_m": None if best is None else best["m"],
        f"{name}_psi_n": None if best is None else best["psi_n"],
        f"{name}_q": None if best is None else best["q"],
        f"{name}_position": None if best is None else best["position"],
    }


def matching_summary(
    surfaces: list[dict], status_raw: Optional[str], prefix: str, *, checked: bool = True
) -> dict[str, Any]:
    """Per-n Δ′ summaries: raw (every surface) and QA-filtered under explicit names.

    * ``*_delta_prime_max_raw``: over every surface, with m, ψ_N, q and position.
    * ``*_delta_prime_max_qa``: over two-resolution-consistent surfaces, any position.
    * ``*_delta_prime_max_qa_interior``: consistent and interior.
    * ``*_delta_prime_max``: deprecated v1 alias of ``*_qa_interior``.
    * ``*_classical_tearing_drive`` (layer 3): Δ′ > 0 at any surface;
      ``*_classical_tearing_drive_qa`` restricts that to consistent surfaces.

    The status says whether the QA check was possible and found a consistent
    surface. It is not a tearing verdict (#939).
    """
    consistent = [s for s in surfaces if s["two_resolution_consistent"]]
    interior = [s for s in consistent if s["position"] == "interior"]
    if status_raw is None:
        status = "NOT_APPLICABLE"
    elif status_raw not in ("completed", "stable"):
        status = "SOLVER_FAILURE"
    elif not surfaces:
        status = "NOT_APPLICABLE"  # the run completed but found no rational surface in range
    elif not checked:
        status = "NOT_CHECKED"
    elif not consistent:
        status = "NUMERICALLY_UNRESOLVED"
    else:
        status = "RESOLVED"
    first = surfaces[0] if surfaces else None
    summary: dict[str, Any] = {
        f"{prefix}_status": status,
        f"{prefix}_n_rational_surfaces": len(surfaces),
        f"{prefix}_n_two_resolution_consistent": len(consistent),
        f"{prefix}_n_resolved_surfaces": len(consistent),  # deprecated v1 name
        f"{prefix}_n_summary_surfaces": len(interior),  # deprecated v1 name
        f"{prefix}_n_positive_delta_prime_raw": sum(s["delta_prime_real"] > 0 for s in surfaces),
        f"{prefix}_n_positive_delta_prime_qa": sum(s["delta_prime_real"] > 0 for s in consistent),
        f"{prefix}_n_positive_delta_prime_summary": sum(s["delta_prime_real"] > 0 for s in interior),  # deprecated
        f"{prefix}_delta_prime_first_surface": None if first is None else first["delta_prime_real"],
        f"{prefix}_first_surface_m": None if first is None else first["m"],
        f"{prefix}_classical_tearing_drive": None if not surfaces else any(s["delta_prime_real"] > 0 for s in surfaces),
        f"{prefix}_classical_tearing_drive_qa": None
        if not checked or not surfaces
        else any(s["delta_prime_real"] > 0 for s in consistent),
    }
    summary.update(_extreme(surfaces, prefix, "_raw"))
    summary.update(_extreme(consistent, prefix, "_qa"))
    summary.update(_extreme(interior, prefix, "_qa_interior"))
    # Deprecated v1 names: the consistent-interior maximum, for one version.
    summary[f"{prefix}_delta_prime_max"] = summary[f"{prefix}_delta_prime_max_qa_interior"]
    summary[f"{prefix}_m_at_delta_prime_max"] = summary[f"{prefix}_delta_prime_max_qa_interior_m"]
    summary[f"{prefix}_psi_n_at_delta_prime_max"] = summary[f"{prefix}_delta_prime_max_qa_interior_psi_n"]
    summary[f"{prefix}_q_at_delta_prime_max"] = summary[f"{prefix}_delta_prime_max_qa_interior_q"]
    return summary


def equilibrium_status(slice_dir: Path, row: dict) -> str:
    """Whether the slice's equilibrium reached the solvers: ok | missing_source | chease_failed."""
    name = f"g{row['shot']:06d}.{row['time_ms']:05d}"
    if not (slice_dir / "source" / name).exists():
        return "missing_source"
    if not (slice_dir / "chease" / name).exists():
        return "chease_failed"
    return "ok"


def _solver_build(results: Iterable[Optional[dict]], builds: dict[str, str]) -> Optional[str]:
    """The GPEC build(s) that produced a solver's runs for one row.

    Taken from each job's own ``gpec_home`` (written by ``run_job``), not
    from the build-time environment. Rows from before that field existed
    report ``unrecorded``.
    """
    homes = sorted({r.get("gpec_home") or "unrecorded" for r in results if r is not None})
    if not homes:
        return None
    return ";".join(builds.get(home, home) for home in homes)


def slice_rows(
    base: Path, row: dict, modes: Iterable[int], provenance: dict, *, builds: Optional[dict] = None, config_sha=None
) -> tuple[list[dict], list[dict]]:
    builds = builds or {}
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
    equilibrium = equilibrium_status(slice_dir, row)
    atlas, surfaces = [], []
    for n in modes:
        job = lambda variant: _result(slice_dir / variant / f"nn{n}")  # noqa: E731
        record: dict[str, Any] = {
            **key,
            "n_tor": n,
            "equilibrium_status": equilibrium,
            "efit_config_sha256": config_sha,
        }
        runs = {v: job(v) for v in ("dcon_mpsi256", "dcon_mpsi512", "dcon_trunc_mpsi256", "dcon_trunc_mpsi512")}
        record.update(dcon_columns(runs["dcon_mpsi256"], "dcon_full_256"))
        record.update(dcon_columns(runs["dcon_mpsi512"], "dcon_full_512"))
        record.update(dcon_columns(runs["dcon_trunc_mpsi256"], "dcon_trunc_256"))
        record.update(dcon_columns(runs["dcon_trunc_mpsi512"], "dcon_trunc_512"))
        for treatment in ("full", "trunc"):
            record[f"dcon_{treatment}_status"] = dcon_status(
                record.get(f"dcon_{treatment}_256_W_t"),
                record.get(f"dcon_{treatment}_512_W_t"),
                record[f"dcon_{treatment}_256_status_raw"],
                record[f"dcon_{treatment}_512_status_raw"],
            )
            record[f"dcon_{treatment}_sign_class"] = dcon_sign_class(
                record.get(f"dcon_{treatment}_256_W_t"), record.get(f"dcon_{treatment}_512_W_t")
            )
            # Layer 3: the ideal criterion is the sign of W_t at the reported resolution.
            w = record.get(f"dcon_{treatment}_256_W_t")
            record[f"ideal_unstable_{treatment}_edge"] = None if w is None else bool(w < 0)
        status = {"full": record["dcon_full_status"], "trunc": record["dcon_trunc_status"]}
        # v1 columns: a stability flag only where its status is a VALID_* verdict.
        record["ideal_stable_full_edge"] = {"VALID_STABLE": True, "VALID_UNSTABLE": False}.get(status["full"])
        record["ideal_stable_truncated_edge"] = {"VALID_STABLE": True, "VALID_UNSTABLE": False}.get(status["trunc"])
        record["dcon_gpec_build"] = _solver_build(runs.values(), builds)
        for solver in ("rdcon", "stride"):
            primary, check = job(f"{solver}_mpsi256"), job(f"{solver}_mpsi512")
            paired = pair_surfaces(matching_surfaces(primary, solver), matching_surfaces(check, solver))
            record.update(
                matching_summary(
                    paired, None if primary is None else primary.get("status"), solver, checked=_ok(check)
                )
            )
            record[f"{solver}_gpec_build"] = _solver_build((primary, check), builds)
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
                        "delta_prime_absolute_change": s["delta_prime_absolute_change"],
                        "sign_agreement": s["sign_agreement"],
                        "two_resolution_consistent": s["two_resolution_consistent"],
                        "resolved": s["resolved"],  # deprecated v1 name
                        "position": s["position"],
                        "ordinal_position": s["position"],
                        "in_summary": summary_eligible(s),  # deprecated v1 rule
                        "shear_psi": s.get("shear_psi"),
                        "psi_n_from_qmin": s.get("psi_n_from_qmin"),
                        "q_minus_qmin": s.get("q_minus_qmin"),
                        "psi_n_to_edge": s.get("psi_n_to_edge"),
                        "D_I": s.get("D_I"),
                        "D_R": s.get("D_R"),
                        "H": s.get("H"),
                        "C_A": s.get("C_A"),
                        "gpec_build": _solver_build((primary,), builds),
                        "run_dir": None if primary is None else primary["run_dir"],
                    }
                )
        if equilibrium != "ok":
            for name in ("dcon_full_status", "dcon_trunc_status", "rdcon_status", "stride_status"):
                record[name] = "INVALID_EQUILIBRIUM"
        record.update(provenance)
        record["slice_dir"] = str(slice_dir)
        atlas.append(record)
    # Layer 3, derived per slice and labelled so: which (n, edge) are ideal-unstable.
    unstable = [
        f"n{r['n_tor']}:{treatment}"
        for r in atlas
        for treatment in ("full", "trunc")
        if r.get(f"ideal_unstable_{treatment}_edge")
    ]
    for record in atlas:
        record["derived_ideal_unstable_any_n"] = ";".join(unstable)
    return atlas, surfaces


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


#: Column dictionary for schema.json: (type, unit, meaning). Columns with a
#: treatment/resolution prefix are described once by their pattern.
COLUMNS: dict[str, tuple[str, str, str]] = {
    "shot": ("int", "", "VEST shot"),
    "time_efit_s": ("float", "s", "EFIT slice time"),
    "efit_lineage": ("str", "", "magnetics-only | electron-kinetic"),
    "efit_label": ("str", "", "#1331 label: good | admissible"),
    "efit_setting": ("str", "", "EFIT preset the label belongs to"),
    "efit_config_sha256": ("str", "", "magnetic EFIT configuration hash recorded by the campaign for this slice"),
    "kinetic_chi2": ("float", "", "electron-EFIT chi2 (electron-kinetic rows only)"),
    "n_tor": ("int", "", "toroidal mode number"),
    "equilibrium_status": ("str", "", "ok | missing_source | chease_failed"),
    "dcon_{t}_{r}_W_t": ("float", "DCON-normalized", "least-stable total energy eigenvalue; >0 stable"),
    "dcon_{t}_{r}_W_p": ("float", "DCON-normalized", "plasma energy at the least-stable mode"),
    "dcon_{t}_{r}_W_v": ("float", "DCON-normalized", "vacuum energy at the least-stable mode"),
    "dcon_{t}_{r}_W_v_imag_ratio": ("float", "", "|Im W_v / Re W_v| (should be ~0)"),
    "dcon_{t}_{r}_m_dominant": ("int", "", "dominant poloidal harmonic of the least-stable eigenfunction"),
    "dcon_{t}_{r}_psilim": ("float", "", "effective edge psi_N (post-truncation for trunc)"),
    "dcon_{t}_{r}_qlim": ("float", "", "q at psilim"),
    "dcon_{t}_{r}_n_zero_crossings": ("int", "", "confirmed Newcomb zero crossings in dcon.out"),
    "dcon_{t}_status": ("str", "", "#142 status of treatment t (full | trunc) from mpsi 256 + 512"),
    "ideal_stable_full_edge": ("bool", "", "only when dcon_full_status is VALID_*"),
    "ideal_stable_truncated_edge": ("bool", "", "only when dcon_trunc_status is VALID_*"),
    "max_D_I": ("float", "", "max Mercier D_I (DCON, mpsi 256, full edge); >0 ideal-interchange unstable"),
    "max_D_R": ("float", "", "max resistive interchange D_R; >0 unstable"),
    "min_C_A": ("float", "", "min ballooning C_A over evaluated surfaces; <0 unstable"),
    "dcon_{t}_sign_class": ("str", "", "layer 2: robust_stable | robust_unstable | numerically_unresolved | not_checked | unavailable"),
    "ideal_unstable_{t}_edge": ("bool", "", "layer 3: W_t < 0 at mpsi 256 (sign only; robustness in dcon_{t}_sign_class)"),
    "derived_ideal_unstable_any_n": ("str", "", "derived per slice: the n:edge pairs with ideal_unstable_*_edge true; same on every n row of the slice"),
    "{s}_status": ("str", "", "RESOLVED (>=1 two-resolution-consistent surface) | NUMERICALLY_UNRESOLVED | NOT_CHECKED | SOLVER_FAILURE | NOT_APPLICABLE | INVALID_EQUILIBRIUM; not a tearing verdict"),
    "{s}_delta_prime_max_raw": ("float", "1 (PEST3 normalization)", "layer 1: max diagonal Δ′ over every surface; _m/_psi_n/_q/_position locate it"),
    "{s}_delta_prime_max_qa": ("float", "1 (PEST3 normalization)", "layer 2: max over two-resolution-consistent surfaces, any position"),
    "{s}_delta_prime_max_qa_interior": ("float", "1 (PEST3 normalization)", "layer 2: max over consistent interior surfaces"),
    "{s}_delta_prime_first_surface": ("float", "1 (PEST3 normalization)", "Δ′ of the innermost rational surface"),
    "{s}_classical_tearing_drive": ("bool", "", "layer 3: Δ′ > 0 at any rational surface (classical drive, not a tearing verdict; #939)"),
    "{s}_classical_tearing_drive_qa": ("bool", "", "layer 3 restricted to two-resolution-consistent surfaces"),
    "{s}_n_two_resolution_consistent": ("int", "", "surfaces passing the mpsi 256/512 heuristic"),
    "{s}_delta_prime_max": ("float", "1 (PEST3 normalization)", "DEPRECATED v1 alias of {s}_delta_prime_max_qa_interior; removed in v3"),
    "{s}_n_summary_surfaces": ("int", "", "resolved interior surfaces used in the summary"),
    "{s}_gpec_build": ("str", "", "GPEC build(s) that ran this solver for the row"),
    "delta_prime": ("float", "1 (PEST3 normalization)", "surface table: diagonal Δ′ at mpsi 256"),
    "two_resolution_consistent": ("bool", "", "surface table: same sign and <= 20 % relative change between mpsi 256 and 512 (heuristic); 'resolved' is its deprecated name"),
    "sign_agreement": ("bool", "", "surface table: mpsi 256 and 512 agree in sign"),
    "ordinal_position": ("str", "", "surface table: first | interior | last (QA annotation, never an exclusion)"),
    "shear_psi": ("float", "", "surface table: d ln q / d ln psi_N at the surface (weak near q_min)"),
    "psi_n_from_qmin": ("float", "", "surface table: psi_N(surface) - psi_N(q_min)"),
    "q_minus_qmin": ("float", "", "surface table: q(surface) - q_min"),
    "psi_n_to_edge": ("float", "", "surface table: distance in psi_N to the end of the solver profile grid"),
    "D_I": ("float", "", "surface table, RDCON only: D_I at the surface"),
    "D_R": ("float", "", "surface table, RDCON only: D_R = D_I + (H - 1/2)^2"),
    "H": ("float", "", "surface table, RDCON only: Glasser H"),
    "C_A": ("float", "", "surface table, RDCON only: ballooning C_A"),
}


def write_csv(rows: list[dict], path: Path) -> Path:
    """Write rows with the union of their columns. Booleans are True/False, missing values empty."""
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


SCHEMA_NOTES = {
    "key": ["shot", "time_efit_s", "efit_lineage", "n_tor"],
    "key_contract": "provisional (shot, time_efit_s, efit_lineage) per lane log #1448, until lane K publishes its State key contract",
    "efit_label": "#1331 criteria label of the magnetics-only EFIT at this time: good | admissible. Unreconstructible slices are never present.",
    "efit_lineage": "magnetics-only (statistical_891 magnetic EFIT from the campaign FileDB the labels were computed on) | electron-kinetic (electron EFIT; label inherited from the magnetics-only slice at exactly the same time)",
    "equilibrium_input": "EFIT g-file refined by CHEASE with pipeline 1's run_chease_refinement.py defaults (target_psin 0.993, nw 513)",
    "energies": "DCON least-stable normalized eigenvalues (not Joules). Positive = stable.",
    "edge_treatments": "dcon_full_* = psiedge 1 (full edge, psihigh 0.994); dcon_trunc_* = psiedge 0.95 (truncated at the dW_edge peak). Never combine. Full-edge W_t is psihigh-sensitive (#141, #792).",
    "resolution": "{r} = 256 is the reported value; 512 is the convergence check (#141).",
    "delta_prime": f"Diagonal classical Δ′ (RDCON/STRIDE PEST3 matching matrix). two_resolution_consistent = same sign and within {DELTA_PRIME_RTOL:.0%} between mpsi 256 and 512 (heuristic: the median change of #141 mid-radius surfaces). Raw extrema use every surface; *_qa use consistent surfaces at any position; *_qa_interior also require an interior surface; *_delta_prime_max is the deprecated v1 alias of *_qa_interior. Position, shear and q_min distance are annotations, never exclusions. RDCON and STRIDE are never combined. RDCON Δ′ also moves ~3% between compiler builds (#1448).",
    "status": list(STATUS_VALUES) + list(EXTENSION_VALUES),
    "status_semantics": "dcon_*_status: VALID_* when mpsi 256 and 512 agree in the sign of W_t, NUMERICALLY_UNRESOLVED when they disagree, NOT_CHECKED when the 512 run is missing/failed. No magnitude band: a small negative W_t with a robust sign is ideal-unstable (#142). {s}_status never carries a stability verdict (#939; RMATCH out of scope, #797). INVALID_EQUILIBRIUM: the source g-file or its CHEASE refinement is missing. MARGINAL is reserved, not produced.",
    "layers": "1 raw solver output; 2 numerical QA annotations (sign_class, two_resolution_consistent, position, shear, q_min distance, *_qa summaries); 3 universal physical criteria (ideal_unstable_*_edge = sign of W_t, *_classical_tearing_drive = Δ′ > 0 at any surface, D_I > 0, D_R > 0, C_A < 0); 4 interpretation (#939) is not produced here and never overwrites 1-3.",
    "coverage": "DCON (full and truncated edge, mpsi 256/512) n = 1..6; RDCON and STRIDE n = 1, 2 only (NOT_APPLICABLE for n >= 3).",
    "local_criteria": "max_D_I > 0 ideal interchange, max_D_R > 0 resistive interchange, min_C_A < 0 ballooning (evaluated surfaces only); from DCON at mpsi 256 with mer_flag/bal_flag recorded.",
    "zero_crossings": "Confirmed Newcomb zero crossings from dcon.out (termbycross_flag=f).",
    "csv_types": "booleans are written True/False, missing values as empty fields; use the column dictionary for types.",
    "rules": ["no cross-n combination", "no DCON/RDCON combined scalar", "do not drop the efit_label/efit_lineage columns"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True, help="run_batch.py output directory")
    parser.add_argument("--out", type=Path, required=True, help="atlas output directory")
    parser.add_argument("--filedb", type=Path, required=True, help="campaign FileDB (for EFIT configuration hashes)")
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--gpec-build",
        action="append",
        default=[],
        metavar="HOME=LABEL",
        help="Label for a GPECHOME seen in the runs, e.g. ~/work/lane-n-gpec='e68d7ac2+gal.f i3.3 patch'.",
    )
    args = parser.parse_args()

    import vaft

    from run_batch import magnetic_config_sha

    builds = {}
    for item in args.gpec_build:
        home, label = item.split("=", 1)
        builds[str(Path(home).expanduser())] = label
    root = Path(vaft.__file__).resolve().parents[1]
    provenance = {
        "atlas_version": ATLAS_VERSION,
        # The tree the builder imports; the solver runs record their own GPECHOME per job.
        "vaft_commit_build": _git(root, "rev-parse", "--short", "HEAD"),
        "vaft_dirty_build": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
    }
    batch = args.batch.expanduser().resolve()
    rows = read_slices(batch / "slices.csv")
    atlas, surfaces = [], []
    for row in rows:
        sha = magnetic_config_sha(args.filedb, row) if row["efit_lineage"] == "magnetics-only" else None
        a, s = slice_rows(batch, row, args.modes, provenance, builds=builds, config_sha=sha)
        atlas.extend(a)
        surfaces.extend(s)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(atlas, out / "atlas_n.csv")
    write_csv(surfaces, out / "atlas_surfaces.csv")
    schema = {
        **SCHEMA_NOTES,
        "provenance": {**provenance, "gpec_builds": builds},
        "column_dictionary": {name: {"type": t, "unit": u, "meaning": m} for name, (t, u, m) in COLUMNS.items()},
        "column_patterns": {"{t}": "full | trunc", "{r}": "256 | 512", "{s}": "rdcon | stride"},
        "atlas_n_columns": sorted({k for r in atlas for k in r}),
        "atlas_surfaces_columns": sorted({k for r in surfaces for k in r}),
        "rows": {"atlas_n": len(atlas), "atlas_surfaces": len(surfaces)},
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    print(f"{len(atlas)} atlas rows, {len(surfaces)} surface rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
