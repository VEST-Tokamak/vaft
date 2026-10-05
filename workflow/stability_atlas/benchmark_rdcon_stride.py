"""RDCON vs STRIDE Δ′ benchmark on the Tier A atlas equilibria (#143).

Both codes solve the same PEST3 rational-surface matching problem and write
the same ``Delta_prime`` matrix (``Pest3MatchingOutput``). #143 asks whether
they agree on the same equilibrium, n, domain and surfaces, after a
convention audit, and only where each code is converged on its own.

``run``
    STRIDE ships ``delta_mhigh = 8``, while RDCON uses 16. For a
    like-for-like truncation this reruns STRIDE with ``delta_mhigh = 16``
    (``delta_mlow`` is 16 in both) at mpsi 256 and 512 for every atlas slice
    and n = 1, 2, through the memory-guarded batch runner. Results land
    beside the atlas jobs as ``stride_dmhigh16_mpsi{256,512}``.
``compare``
    Per (slice, n) it reads RDCON (``rdcon_mpsi*``) and the like-for-like
    STRIDE runs, and writes one row per pair to ``benchmark_143.csv`` and one
    row per surface to ``benchmark_143_surfaces.csv``.

Per pair:
* Surfaces are matched by m and ψ_N (``SURFACE_PSI_TOL``), never by position.
* ``internally_converged``: a surface whose Δ′ is two-resolution
  consistent in its own code (the atlas heuristic, ``DELTA_PRIME_RTOL``).
* Matrix metric on the common surfaces, at mpsi 256:
  ``||M_r - M_s||_F / max(||M_s||_F, eps)``, also against ``M_s^T`` and
  ``M_s^H``. The convention audit uses the surfaces converged in both codes
  only, so unresolved entries cannot fake a mismatch. It tests transpose and
  conjugate transpose only; an overall sign flip would show as near-zero
  sign agreement, and a normalization factor is not tested.
* Per surface, the real parts of the diagonal Δ′ are compared (the quantity
  the atlas and ``ntms.deltaw`` carry); imaginary parts are only in the
  Frobenius metric.
* Status (#143): PASS (every common surface converged in both codes and all
  agree), PARTIALLY_COMPARABLE (the converged surfaces agree but not every
  surface is converged in both, or the surface sets differ),
  CONVENTION_MISMATCH, NUMERICALLY_UNRESOLVED (no surface converged in both),
  PHYSICS_DISAGREEMENT (a converged surface disagrees), SOLVER_FAILURE (any of
  the four runs is missing or failed; ``failed`` names them), NOT_APPLICABLE
  (no rational surface in either code).

``AGREEMENT_RTOL`` reuses the atlas heuristic (20 %, the median mpsi 256/512
change of #141's mid-radius surfaces). It is not an independently derived
tolerance (#142), and the CSV keeps every number so a consumer can apply
their own.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas import DELTA_PRIME_RTOL, SURFACE_PSI_TOL, _ok, _result, write_csv  # noqa: E402
from run_batch import gfile_name, job_backend, slice_label  # noqa: E402
from scan_controls import Equilibrium, Variant, collect, materialize_templates, preflight, run_job  # noqa: E402
from select_slices import read as read_slices  # noqa: E402

#: Heuristic, shared with the atlas (see module docstring): codes agree on a surface.
AGREEMENT_RTOL = DELTA_PRIME_RTOL
#: Heuristic: a code is internally converged on a surface when mpsi 256 and 512
#: agree in sign and within this relative change (the atlas rule).
SELF_CONSISTENCY_RTOL = DELTA_PRIME_RTOL
#: A transposed/conjugated form must beat the plain comparison by this factor
#: to be called a convention mismatch.
CONVENTION_GAIN = 2.0
MODES = (1, 2)
STRIDE_VARIANTS = tuple(
    Variant(f"stride_dmhigh16_mpsi{m}", "stride", {"equil.in": {"mpsi": m}, "stride.in": {"delta_mhigh": 16}})
    for m in (256, 512)
)


def _matrix(result: Optional[dict], solver: str):
    from vaft.code.gpec import read_pest3_matching_output

    if not _ok(result):
        return None
    return read_pest3_matching_output(Path(result["run_dir"]), solver=solver, mode=int(result["n"]))


def _surfaces(out) -> list[tuple[int, int, float, complex]]:
    """[(index, m, psi_n, diagonal Δ′)] of one run. The same m can occur twice with non-monotonic q."""
    if out is None or out.m is None or out.Delta_prime is None or out.psi_n_rational is None:
        return []
    return [
        (i, int(m), float(out.psi_n_rational[i]), complex(out.Delta_prime[i, i])) for i, m in enumerate(out.m)
    ]


def _find(surfaces: list, m: int, psi: float) -> Optional[tuple]:
    """The surface with this m at this ψ_N (within ``SURFACE_PSI_TOL``), or None."""
    hits = [s for s in surfaces if s[1] == m and abs(s[2] - psi) <= SURFACE_PSI_TOL]
    return min(hits, key=lambda s: abs(s[2] - psi)) if hits else None


def _consistent(a: Optional[complex], b: Optional[complex]) -> Optional[bool]:
    if a is None or b is None:
        return None
    a, b = a.real, b.real
    return bool(np.sign(a) == np.sign(b) and abs(a - b) / max(abs(a), abs(b), 1e-12) <= SELF_CONSISTENCY_RTOL)


def common_surfaces(r: list, s: list) -> list[tuple[tuple, tuple]]:
    """Pairs (RDCON surface, STRIDE surface) with the same m at the same ψ_N, ordered by ψ_N."""
    pairs = []
    for surface in r:
        partner = _find(s, surface[1], surface[2])
        if partner is not None:
            pairs.append((surface, partner))
    return sorted(pairs, key=lambda p: p[0][2])


def frobenius(mr: np.ndarray, ms: np.ndarray) -> dict[str, float]:
    """Relative Frobenius distance of M_r to M_s, M_s^T and M_s^H."""
    scale = max(float(np.linalg.norm(ms)), 1e-12)
    return {
        "plain": float(np.linalg.norm(mr - ms)) / scale,
        "transpose": float(np.linalg.norm(mr - ms.T)) / scale,
        "conjugate_transpose": float(np.linalg.norm(mr - ms.conj().T)) / scale,
    }


def compare_pair(rdcon: tuple, stride: tuple) -> tuple[dict, list[dict]]:
    """Compare one (slice, n): ``rdcon``/``stride`` are (out256, out512) Pest3MatchingOutput or None."""
    r256, r512 = rdcon
    s256, s512 = stride
    row: dict[str, Any] = {}
    missing = [
        name
        for name, out in (("rdcon_mpsi256", r256), ("rdcon_mpsi512", r512), ("stride_mpsi256", s256), ("stride_mpsi512", s512))
        if out is None
    ]
    if missing:
        row["status"] = "SOLVER_FAILURE"
        row["failed"] = ";".join(missing)
        return row, []
    sr, sr512, ss, ss512 = _surfaces(r256), _surfaces(r512), _surfaces(s256), _surfaces(s512)
    common = common_surfaces(sr, ss)
    row.update(
        {
            "rdcon_msing": len(sr),
            "stride_msing": len(ss),
            "n_common": len(common),
            "rdcon_mlow_mhigh": f"{r256.mlow}..{r256.mhigh}",
            "stride_mlow_mhigh": f"{s256.mlow}..{s256.mhigh}",
            "truncation_match": bool((r256.mlow, r256.mhigh) == (s256.mlow, s256.mhigh)),
        }
    )
    surfaces = []
    for r_surface, s_surface in common:
        _, m, psi, dr = r_surface
        ds = s_surface[3]
        r_check = _find(sr512, m, psi)
        s_check = _find(ss512, m, s_surface[2])
        r_ok = _consistent(dr, None if r_check is None else r_check[3])
        s_ok = _consistent(ds, None if s_check is None else s_check[3])
        rel = abs(dr.real - ds.real) / max(abs(dr.real), abs(ds.real), 1e-12)
        surfaces.append(
            {
                "m": m,
                "psi_n": psi,
                "delta_prime_rdcon": dr.real,
                "delta_prime_stride": ds.real,
                "relative_difference": rel,
                "sign_agreement": bool(np.sign(dr.real) == np.sign(ds.real)),
                "rdcon_internally_converged": r_ok,
                "stride_internally_converged": s_ok,
                "both_converged": bool(r_ok and s_ok),
                "codes_agree": bool(np.sign(dr.real) == np.sign(ds.real) and rel <= AGREEMENT_RTOL),
            }
        )
    converged = [s for s in surfaces if s["both_converged"]]
    row.update(
        {
            "n_both_converged": len(converged),
            "n_agree_all": sum(s["codes_agree"] for s in surfaces),
            "n_agree_converged": sum(s["codes_agree"] for s in converged),
            "n_sign_agree_converged": sum(s["sign_agreement"] for s in converged),
        }
    )

    def distances(subset: list[tuple[tuple, tuple]]) -> dict[str, float]:
        ir = [p[0][0] for p in subset]
        js = [p[1][0] for p in subset]
        return frobenius(np.asarray(r256.Delta_prime)[np.ix_(ir, ir)], np.asarray(s256.Delta_prime)[np.ix_(js, js)])

    if common:
        row.update({f"frobenius_{k}": v for k, v in distances(common).items()})
        both = [pair for pair, s_ in zip(common, surfaces) if s_["both_converged"]]
        if both:
            row.update({f"frobenius_converged_{k}": v for k, v in distances(both).items()})
            row["frobenius_converged"] = row["frobenius_converged_plain"]  # name used before the review
    row["status"] = pair_status(row)
    return row, surfaces


def pair_status(row: dict) -> str:
    if row.get("status") == "SOLVER_FAILURE":
        return "SOLVER_FAILURE"
    if not row.get("rdcon_msing") and not row.get("stride_msing"):
        return "NOT_APPLICABLE"
    if not row.get("n_common"):
        return "PARTIALLY_COMPARABLE"
    if not row["n_both_converged"]:
        return "NUMERICALLY_UNRESOLVED"
    # The convention audit uses only surfaces both codes resolve.
    plain = row["frobenius_converged_plain"]
    best_other = min(row["frobenius_converged_transpose"], row["frobenius_converged_conjugate_transpose"])
    if best_other * CONVENTION_GAIN < plain:
        return "CONVENTION_MISMATCH"
    if row["n_agree_converged"] < row["n_both_converged"]:
        return "PHYSICS_DISAGREEMENT"
    complete = row["n_both_converged"] == row["n_common"] == max(row["rdcon_msing"], row["stride_msing"])
    return "PASS" if complete else "PARTIALLY_COMPARABLE"


def run(args) -> int:
    preflight({"stride"})
    batch = args.batch.expanduser().resolve()
    rows = read_slices(batch / "slices.csv")
    templates = {v.name: materialize_templates(v, batch / "_templates") for v in STRIDE_VARIANTS}
    jobs = []
    for row in rows:
        gfile = batch / row["efit_lineage"] / slice_label(row) / "chease" / gfile_name(row)
        if not gfile.exists():
            continue
        eq = Equilibrium(row["shot"], row["time_ms"], gfile)
        for n in MODES:
            for v in STRIDE_VARIANTS:
                jobs.append((eq, v, n, batch / row["efit_lineage"]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job, eq, v, n, templates[v.name], base, args.timeout, prune=True,
                backend=job_backend("stride", n, admission_wait_s=args.admission_wait),
            ): (eq, v, n, base)
            for eq, v, n, base in jobs
        }  # fmt: skip
        failed = 0
        for done, future in enumerate(as_completed(futures), 1):
            eq, v, n, base = futures[future]
            result = collect(future, eq, v, n)
            failed += result["status"] in ("failed", "error")
            print(f"[{done}/{len(jobs)}] {base.name} {eq.label} {v.name} n={n}: {result['status']}", flush=True)
    print(f"{failed} of {len(jobs)} jobs failed; compare reports them as SOLVER_FAILURE", flush=True)
    return 0


def compare(args) -> int:
    batch = args.batch.expanduser().resolve()
    pairs, surfaces = [], []
    for row in read_slices(batch / "slices.csv"):
        slice_dir = batch / row["efit_lineage"] / slice_label(row)
        for n in MODES:
            job = lambda v: _result(slice_dir / v / f"nn{n}")  # noqa: E731
            r = (_matrix(job("rdcon_mpsi256"), "rdcon"), _matrix(job("rdcon_mpsi512"), "rdcon"))
            s = (
                _matrix(job("stride_dmhigh16_mpsi256"), "stride"),
                _matrix(job("stride_dmhigh16_mpsi512"), "stride"),
            )
            key = {"shot": row["shot"], "time_efit_s": row["time_efit_s"], "efit_lineage": row["efit_lineage"], "efit_label": row["efit_label"], "n_tor": n}  # fmt: skip
            pair, surf = compare_pair(r, s)
            pairs.append({**key, **pair})
            surfaces.extend({**key, **s_} for s_ in surf)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(pairs, out / "benchmark_143.csv")
    write_csv(surfaces, out / "benchmark_143_surfaces.csv")
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    summary = {
        "pairs": len(pairs),
        "surfaces": len(surfaces),
        "status": counts,
        "agreement_rtol": AGREEMENT_RTOL,
        "convention_gain": CONVENTION_GAIN,
        "stride_truncation": "delta_mlow=16, delta_mhigh=16 (RDCON 16/16)",
    }
    (out / "benchmark_143_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run like-for-like STRIDE (delta_mhigh=16) on the atlas slices")
    p_run.add_argument("--batch", type=Path, required=True)
    p_run.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    p_run.add_argument("--timeout", type=float, default=2400.0)
    p_run.add_argument("--admission-wait", type=float, default=6 * 3600.0)
    p_cmp = sub.add_parser("compare", help="compare RDCON and like-for-like STRIDE")
    p_cmp.add_argument("--batch", type=Path, required=True)
    p_cmp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
