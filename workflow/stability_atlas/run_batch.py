"""Run the stability atlas batch over the selected Tier A slices (#1429).

For every row of ``slices.csv`` (from ``select_slices.py``):

1. **Source g-file.**
   - magnetics-only: the campaign's magnetic EFIT g-file at that time.
   - electron-kinetic: written from the electron-EFIT ODS with
     :func:`vaft.data.eqdsk.from_omas`. The time slice is chosen by time, not
     index, and must lie within ``KINETIC_TIME_TOLERANCE_S`` of the label time.
2. **CHEASE refinement** with the pipeline's own ``run_chease_refinement.py``
   at its defaults, so the input is what pipeline 1 would hand GPEC.
3. **GPEC solvers** through ``scan_controls.run_job``, one suite call per
   (variant, n), with the #141 choices: ``mpsi`` 256 and its 512 check,
   ``bal_flag=t``, and both DCON edge treatments.

Every stage is resumable: a slice whose refined g-file exists, or a job whose
``result.json`` exists, is skipped. The batch's ``slices.csv`` is not cached:
it always mirrors ``--slices``, so a resumed batch with a re-labelled list
builds its atlas from the new list (``build_atlas.py`` reads only the batch
copy). A failure (missing input, CHEASE failure,
solver failure or timeout) is recorded, never raised, so one bad slice cannot
stop the batch. Layout::

    OUT/<lineage>/<shot>.<time_ms:05d>/source/   g-file handed to CHEASE
    OUT/<lineage>/<shot>.<time_ms:05d>/chease/   refined g-file + logs
    OUT/<lineage>/<shot>.<time_ms:05d>/<variant>/nn<n>/result.json
    OUT/stages.jsonl                             one line per source/CHEASE outcome
"""

from __future__ import annotations

import argparse
import filecmp
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_controls import DCON, Equilibrium, Variant, collect, materialize_templates, preflight, run_job  # noqa: E402
from select_slices import ELECTRON_KINETIC, MAGNETICS_ONLY, read  # noqa: E402

#: #141 (comment 2026-10-01): mpsi=256 is the working resolution; 512 is
#: the per-surface/per-slice convergence check. mpsi=128 (packaged) is not
#: converged, and 1024 exceeds what the nw=513 CHEASE g-files support.
MPSI = (256, 512)
#: #792: the truncated solution is independent of psiedge once psiedge is
#: below the dW_edge peak, which sits above 0.98 on every reference case.
TRUNCATED_PSIEDGE = 0.95

#: The packaged dcon.in sets termbycross_flag=t (upstream default: f). DCON then
#: stops at the first sign change of its crit function, including sign flips
#: across a rational surface that its own midpoint test does not confirm as a
#: zero crossing, and it writes no netCDF. All five such stops in the first
#: batch pass were unconfirmed. With f, DCON completes; confirmed Newcomb
#: crossings are still logged ("Zero crossing at psi = ...") and the atlas
#: reads them from dcon.out.
NO_TERMINATION = {"termbycross_flag": False}

ATLAS_VARIANTS: tuple[Variant, ...] = (
    *(Variant(f"dcon_mpsi{m}", "dcon", {"equil.in": {"mpsi": m}, "dcon.in": NO_TERMINATION}) for m in MPSI),
    *(
        Variant(
            f"dcon_trunc_mpsi{m}",
            "dcon",
            {"equil.in": {"mpsi": m}, "dcon.in": NO_TERMINATION},
            dcon=replace(DCON, psiedge=TRUNCATED_PSIEDGE),
        )
        for m in MPSI
    ),
    *(Variant(f"rdcon_mpsi{m}", "rdcon", {"equil.in": {"mpsi": m}}) for m in MPSI),
    *(Variant(f"stride_mpsi{m}", "stride", {"equil.in": {"mpsi": m}}) for m in MPSI),
)

#: Memory per solver job [MiB]: (reservation, resident limit), from measured
#: peaks on vestserver (#1460): DCON/STRIDE <= 1 GB; RDCON n=1 a few GB; RDCON
#: n=2 12 GB at mpsi 256 and 24.6 GB at mpsi 512. n>=3 has more rational
#: surfaces and is extrapolated. A job is admitted only when the host has room
#: for its reservation next to everything already admitted, and stopped as
#: ``memory_limit`` if its process tree passes the limit.
MEMORY_MB: dict[tuple[str, int], tuple[float, float]] = {
    ("rdcon", 1): (8_000, 20_000),
    ("rdcon", 2): (26_000, 40_000),
}
MEMORY_MB_RDCON_HIGH_N = (40_000, 60_000)
MEMORY_MB_DEFAULT = (2_000, 8_000)
#: DCON per n [MiB]: its companion ``match`` builds the eigenfunction and
#: dominates the peak, which grows with the number of harmonics (~n*q range).
#: Measured on 39915@319 ms (q_edge 11.7), mpsi 512, n=6 (m = -12..86): 9.4 GB
#: full edge, 9.2 GB truncated. Reservation 1.8 GB*n and limit 3.6 GB*n
#: cover that with margin; small n keeps the default floor.
DCON_MB_PER_N = (1_800, 3_600)


def memory_policy(module: str, n: int) -> tuple[float, float]:
    """(reservation, limit) in MiB for one solver job."""
    if module == "rdcon" and n >= 3:
        return MEMORY_MB_RDCON_HIGH_N
    if module == "dcon":
        return (
            max(MEMORY_MB_DEFAULT[0], DCON_MB_PER_N[0] * n),
            max(MEMORY_MB_DEFAULT[1], DCON_MB_PER_N[1] * n),
        )
    return MEMORY_MB.get((module, n), MEMORY_MB_DEFAULT)


def job_backend(module: str, n: int, *, admission_wait_s: float):
    from vaft.code.execution import LocalBackend

    reserve, limit = memory_policy(module, n)
    return LocalBackend(reserve_mb=reserve, memory_limit_mb=limit, admission_wait_s=admission_wait_s)


#: The electron-EFIT manifest records ``equilibrium_tolerance_ms = 0.5``; a
#: slice further than that from the label time is not the same state.
KINETIC_TIME_TOLERANCE_S = 0.5e-3

PIPELINE = Path(__file__).resolve().parents[1] / "automatic_pipeline_1_routine_data_processing"


def slice_label(row: dict) -> str:
    return f"{row['shot']}.{row['time_ms']:05d}"


def gfile_name(row: dict) -> str:
    return f"g{row['shot']:06d}.{row['time_ms']:05d}"


def magnetic_config_sha(filedb: Path, row: dict) -> str | None:
    """The EFIT configuration hash the campaign recorded for this magnetic slice.

    The magnetics-only g-files come from the same campaign FileDB the #1331
    labels were computed from. The hash makes that link checkable per row
    instead of assumed.
    """
    manifest = filedb / "omas" / "efit" / "magnetic" / str(row["shot"]) / "metadata" / "manifest.json"
    if not manifest.exists():
        return None
    for status in json.loads(manifest.read_text(encoding="utf-8")).get("slice_statuses", []):
        # slice_statuses[].time is in seconds; labels are integer ms.
        if status.get("time") is not None and round(float(status["time"]) * 1000) == row["time_ms"]:
            return status.get("provenance", {}).get("configuration", {}).get("scientific_sha256")
    return None


def stage_slices(source: Path, out: Path) -> Path:
    """Mirror the slice list into the batch directory, overwriting a stale copy.

    ``build_atlas.py`` reads ``OUT/slices.csv`` and has no override, so the
    copy must be the list this invocation ran, not the one a previous
    invocation was started with (cold review 0.8.0 delta-absorb-6 F7). The
    copy is rewritten whenever its content differs from ``source``; an
    identical copy (including ``source`` being the batch copy itself) is
    left alone.
    """
    target = out / "slices.csv"
    if target.exists() and filecmp.cmp(source, target, shallow=False):
        return target
    _write_atomic(target, lambda path: shutil.copyfile(source, path))
    return target


def _write_atomic(path: Path, write) -> None:
    """Write via a temporary file so a killed job never leaves a truncated input behind."""
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_source(row: dict, filedb: Path, slice_dir: Path) -> dict:
    """Stage the source g-file for one slice; return a stage record."""
    source = slice_dir / "source" / gfile_name(row)
    record = {"stage": "source", "slice": slice_label(row), "lineage": row["efit_lineage"]}
    if source.exists():
        return {**record, "status": "ok", "path": str(source), "cached": True}
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():  # dangling: its target disappeared
        source.unlink()
    if row["efit_lineage"] == MAGNETICS_ONLY:
        native = filedb / "efit" / "magnetic" / str(row["shot"]) / "gfile" / gfile_name(row)
        if not native.exists():
            return {**record, "status": "missing", "reason": f"no g-file {native}"}
        source.symlink_to(native)
        return {**record, "status": "ok", "path": str(source), "efit_config_sha256": magnetic_config_sha(filedb, row)}
    if row["efit_lineage"] == ELECTRON_KINETIC:
        import omas

        from vaft.data.eqdsk import from_omas, write_geqdsk

        product = filedb / "omas" / "electron_efit" / str(row["shot"]) / "output" / "electron_efit.json.gz"
        if not product.exists():
            return {**record, "status": "missing", "reason": f"no electron-EFIT product {product}"}
        with gzip.open(product, "rt", encoding="utf-8") as handle:
            ods = omas.load_omas_json(handle)
        # from_omas reads equilibrium.time_slice[index], so the time must come
        # from the same slice, not from the separate equilibrium.time vector.
        count = len(ods["equilibrium.time_slice"])
        times = np.array([float(ods["equilibrium.time_slice"][i]["time"]) for i in range(count)])
        index = int(np.argmin(np.abs(times - row["time_efit_s"])))
        dt = float(times[index] - row["time_efit_s"])
        if abs(dt) > KINETIC_TIME_TOLERANCE_S:
            return {**record, "status": "missing", "reason": f"nearest electron-EFIT slice is {dt * 1e3:+.3f} ms away"}
        _write_atomic(source, lambda path: write_geqdsk(from_omas(ods, index), path))
        return {**record, "status": "ok", "path": str(source), "dt_s": dt}
    raise ValueError(f"unknown lineage {row['efit_lineage']!r}")


def refine(row: dict, slice_dir: Path, chease: str, timeout: float) -> dict:
    """CHEASE-refine one slice with the pipeline's script; return a stage record."""
    record = {"stage": "chease", "slice": slice_label(row), "lineage": row["efit_lineage"]}
    workdir = slice_dir / "chease"
    refined = workdir / gfile_name(row)
    if refined.exists():
        return {**record, "status": "ok", "path": str(refined), "cached": True}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "gfiles.txt").write_text(str((slice_dir / "source" / gfile_name(row)).resolve()) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(PIPELINE / "run_chease_refinement.py"),
        "--shot", str(row["shot"]),
        "--gfile-manifest", "gfiles.txt",
        "--output", "refined.txt",
        "--status", "status.txt",
        "--executable", chease,
        "--timeout", str(timeout),
        "--create-plot", "false",
    ]  # fmt: skip
    with (workdir / "chease.log").open("w", encoding="utf-8") as log:
        returncode = subprocess.run(command, cwd=workdir, stdout=log, stderr=subprocess.STDOUT).returncode
    status = (workdir / "status.txt").read_text(encoding="utf-8").strip() if (workdir / "status.txt").exists() else ""
    if returncode == 0 and refined.exists():
        return {**record, "status": "ok", "path": str(refined), "chease_status": status}
    return {**record, "status": "failed", "returncode": returncode, "chease_status": status}


def prepare(row: dict, filedb: Path, out: Path, chease: str, timeout: float) -> list[dict]:
    slice_dir = out / row["efit_lineage"] / slice_label(row)
    source = write_source(row, filedb, slice_dir)
    if source["status"] != "ok":
        return [source]
    return [source, refine(row, slice_dir, chease, timeout)]


def prepared(future: Any, row: dict) -> list[dict]:
    """A slice's stage records; an error while preparing it becomes an ``error`` record.

    A corrupt ODS or an unexpected layout in one slice must not end the
    preparation of the other 149 (the module docstring's promise).
    """
    error = future.exception()
    if error is None:
        return future.result()
    return [{"stage": "source", "slice": slice_label(row), "lineage": row["efit_lineage"], "status": "error", "reason": repr(error)}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slices", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filedb", type=Path, default=Path(os.environ.get("FILEDB", "")))
    parser.add_argument("--chease", default=os.environ.get("CHEASE", ""))
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=float, default=2400.0, help="Per GPEC job, seconds.")
    parser.add_argument("--chease-timeout", type=float, default=600.0)
    parser.add_argument("--only", nargs="*", default=None, help="Variant names (default: all atlas variants).")
    parser.add_argument("--keep-scratch", action="store_true", help="Keep euler.bin and friends.")
    parser.add_argument(
        "--admission-wait",
        type=float,
        default=6 * 3600.0,
        help="Seconds a job may wait for memory admission before it is recorded as not started.",
    )
    parser.add_argument(
        "--no-memory-guard", action="store_true", help="Plain LocalBackend: no memory admission or limit (#1460)."
    )
    args = parser.parse_args()

    variants = [v for v in ATLAS_VARIANTS if not args.only or v.name in args.only]
    preflight({v.module for v in variants})
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = read(args.slices)
    stage_slices(args.slices, out)
    # Good slices first: they carry the headline results.
    rows.sort(key=lambda r: (r["efit_label"] != "good", r["shot"], r["time_ms"], r["efit_lineage"]))

    stages: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (out / "stages.jsonl").open("a", encoding="utf-8") as log:
        futures = {pool.submit(prepare, row, args.filedb, out, args.chease, args.chease_timeout): row for row in rows}
        for future in as_completed(futures):
            records = prepared(future, futures[future])
            stages.extend(records)
            for record in records:
                if not record.get("cached"):
                    log.write(json.dumps(record) + "\n")
                    log.flush()
    refined = {(r["slice"], r["lineage"]): Path(r["path"]) for r in stages if r["stage"] == "chease" and r["status"] == "ok"}
    print(f"refined {len(refined)}/{len(rows)} slices", flush=True)

    templates = {v.name: materialize_templates(v, out / "_templates") for v in variants}
    jobs: list[tuple[Equilibrium, Variant, int, Path]] = []
    for row in rows:
        gfile = refined.get((slice_label(row), row["efit_lineage"]))
        if gfile is None:
            continue
        eq = Equilibrium(row["shot"], row["time_ms"], gfile)
        for n in args.modes:
            for variant in variants:
                jobs.append((eq, variant, n, out / row["efit_lineage"]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job,
                eq,
                v,
                n,
                templates[v.name],
                base,
                args.timeout,
                prune=not args.keep_scratch,
                backend=None if args.no_memory_guard else job_backend(v.module, n, admission_wait_s=args.admission_wait),
            ): (eq, v, n, base)
            for eq, v, n, base in jobs
        }
        for done, future in enumerate(as_completed(futures), 1):
            eq, v, n, base = futures[future]
            row: dict[str, Any] = collect(future, eq, v, n)
            print(f"[{done}/{len(jobs)}] {base.name} {eq.label} {v.name} n={n}: {row['status']} {row.get('wall_s')} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
