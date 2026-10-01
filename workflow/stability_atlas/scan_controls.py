"""Narrow numerical-control scan for DCON/RDCON/STRIDE (issue #141, minimal).

One axis at a time from the packaged templates. Each variant gets its own
templates directory (the packaged one with namelist keys patched), and each
(equilibrium, variant, module, n) runs as its own suite call in its own
workdir -- DCON and RDCON are never run in one call (#141: the coupled n=2
run did not finish in 40 min).

Usage::

    python scan_controls.py --out DIR --eq SHOT:TIME_MS:GFILE [--eq ...] \
        [--modes 1 2] [--workers 12] [--only VARIANT ...]

A finished job leaves ``result.json`` in its directory and is skipped on the
next invocation, so the scan is resumable. ``summarize`` reads the results
into one CSV table.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from vaft.code.gpec import (
    DCONOptions,
    GPECCaseInputs,
    GPECSuiteConfig,
    read_dcon_output,
    read_pest3_matching_output,
    run_gpec_suite_case,
)
from vaft.code.gpec._runtime import package_vest_dir, write_template


# Every DCON run evaluates the ballooning criterion: the packaged dcon.in has
# bal_flag=f, which leaves ca1 identically zero (#527).
DCON = DCONOptions(bal_flag=True)


@dataclass(frozen=True)
class Variant:
    """One point on one scan axis: a module and the template keys it changes."""

    name: str
    module: str
    patches: dict[str, dict[str, Any]] = field(default_factory=dict)  # template file -> {key: value}
    psihigh: float | None = None  # equil.in psihigh goes through the config, not a patch
    dcon: DCONOptions = DCON  # psiedge < psilim selects the peak-dW truncated edge (#792)


VARIANTS: tuple[Variant, ...] = (
    Variant("dcon_base", "dcon"),
    Variant("dcon_mpsi256", "dcon", {"equil.in": {"mpsi": 256}}),
    Variant("dcon_mpsi512", "dcon", {"equil.in": {"mpsi": 512}}),
    Variant("dcon_mpsi1024", "dcon", {"equil.in": {"mpsi": 1024}}),
    Variant("dcon_mtheta512", "dcon", {"equil.in": {"mtheta": 512}}),
    Variant("dcon_dm16", "dcon", {"dcon.in": {"delta_mlow": 16, "delta_mhigh": 16}}),
    Variant("dcon_dm24", "dcon", {"dcon.in": {"delta_mlow": 24, "delta_mhigh": 24}}),
    Variant("dcon_psihigh0990", "dcon", psihigh=0.990),
    Variant("dcon_psihigh0997", "dcon", psihigh=0.997),
    Variant("dcon_psiedge0950", "dcon", dcon=replace(DCON, psiedge=0.95)),
    Variant("dcon_psiedge0900", "dcon", dcon=replace(DCON, psiedge=0.90)),
    Variant("rdcon_base", "rdcon"),
    Variant("rdcon_nx128", "rdcon", {"rdcon.in": {"nx": 128}}),
    Variant("rdcon_nx512", "rdcon", {"rdcon.in": {"nx": 512}}),
    Variant("rdcon_nq4", "rdcon", {"rdcon.in": {"nq": 4}}),
    Variant("rdcon_nq8", "rdcon", {"rdcon.in": {"nq": 8}}),
    Variant("rdcon_cutoff5", "rdcon", {"rdcon.in": {"cutoff": 5}}),
    Variant("rdcon_cutoff20", "rdcon", {"rdcon.in": {"cutoff": 20}}),
    Variant("rdcon_dm8", "rdcon", {"rdcon.in": {"delta_mlow": 8, "delta_mhigh": 8}}),
    Variant("rdcon_dm24", "rdcon", {"rdcon.in": {"delta_mlow": 24, "delta_mhigh": 24}}),
    Variant("rdcon_mpsi256", "rdcon", {"equil.in": {"mpsi": 256}}),
    Variant("rdcon_mpsi512", "rdcon", {"equil.in": {"mpsi": 512}}),
    Variant("rdcon_mpsi1024", "rdcon", {"equil.in": {"mpsi": 1024}}),
    Variant("rdcon_psihigh0990", "rdcon", psihigh=0.990),
    Variant("rdcon_psihigh0997", "rdcon", psihigh=0.997),
    Variant("stride_base", "stride"),
    Variant("stride_dm16", "stride", {"stride.in": {"delta_mlow": 16, "delta_mhigh": 16}}),
    Variant("stride_mpsi256", "stride", {"equil.in": {"mpsi": 256}}),
    Variant("stride_mpsi512", "stride", {"equil.in": {"mpsi": 512}}),
    Variant("stride_mpsi1024", "stride", {"equil.in": {"mpsi": 1024}}),
)


@dataclass(frozen=True)
class Equilibrium:
    shot: int
    time_ms: int
    gfile: Path

    @property
    def label(self) -> str:
        return f"{self.shot}.{self.time_ms:05d}"


def parse_equilibrium(text: str) -> Equilibrium:
    shot, time_ms, gfile = text.split(":", 2)
    return Equilibrium(int(shot), int(time_ms), Path(gfile).expanduser().resolve())


def materialize_templates(variant: Variant, root: Path) -> Path:
    """Copy the packaged templates and apply the variant's patches."""
    target = root / variant.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(package_vest_dir(), target)
    for filename, keys in variant.patches.items():
        path = target / filename
        write_template(path, path, keys)
    return target


def run_job(eq: Equilibrium, variant: Variant, mode: int, templates: Path, out: Path, timeout: float) -> dict:
    jobdir = out / eq.label / variant.name / f"nn{mode}"
    done = jobdir / "result.json"
    if done.exists():
        return json.loads(done.read_text())
    jobdir.mkdir(parents=True, exist_ok=True)
    config = GPECSuiteConfig(
        modules=(variant.module,),
        modes=(mode,),
        run_mode="strict",
        templates_dir=templates,
        psihigh=variant.psihigh if variant.psihigh is not None else GPECSuiteConfig.psihigh,
        verify_outputs=True,
        timeout=timeout,
        dcon=variant.dcon,
    )
    start = time.monotonic()
    result = run_gpec_suite_case(
        GPECCaseInputs(shot=eq.shot, time_ms=eq.time_ms, geqdsk=eq.gfile, workdir=jobdir / "work"),
        config,
    )
    elapsed = time.monotonic() - start
    record = next(r for r in result.records if r.module == variant.module and r.mode == mode)
    row: dict[str, Any] = {
        "equilibrium": eq.label,
        "shot": eq.shot,
        "time_ms": eq.time_ms,
        "variant": variant.name,
        "module": variant.module,
        "n": mode,
        "status": record.status,
        "reason": record.reason,
        "wall_s": round(elapsed, 1),
        "run_dir": str(record.workdir),
    }
    if record.ok:
        row.update(metrics(variant.module, Path(record.workdir), mode))
    done.write_text(json.dumps(row, indent=1, default=str))
    return row


def metrics(module: str, run_dir: Path, mode: int) -> dict[str, Any]:
    if module == "dcon":
        out = read_dcon_output(run_dir, mode=mode)
        total = out.total1
        di = None if out.di is None else np.asarray(out.di, dtype=float)
        ca1 = None if out.ca1 is None else np.asarray(out.ca1, dtype=float)
        evaluated = None if out.ca1_evaluated is None else np.asarray(out.ca1_evaluated, dtype=bool)
        return {
            "W_t_min": None if total is None else float(total.real),
            "W_p_min": None if out.plasma1 is None else float(out.plasma1.real),
            "W_v_min": None if out.vacuum1 is None else float(out.vacuum1.real),
            "m_dominant": out.m_pol_dominant,
            "mlow": out.mlow,
            "mhigh": out.mhigh,
            "psilim": out.psilim,
            "qlim": out.qlim,
            "edge_treatment": out.edge_treatment,
            "edge_peak_psi": None if out.edge_scan is None else float(out.edge_scan.psi_n[int(np.nanargmax(out.edge_scan.dW))]),
            "max_di": None if di is None or not di.size else float(np.nanmax(di)),
            "n_ca1_negative": None if ca1 is None or evaluated is None else int(np.sum((ca1 < 0) & evaluated)),
            "n_ca1_evaluated": None if evaluated is None else int(np.sum(evaluated)),
        }
    out = read_pest3_matching_output(run_dir, solver=module, mode=mode)
    diag = out.delta_prime_diagonal()
    dp = None if out.Delta_prime is None else np.asarray(out.Delta_prime)
    return {
        "msing": out.msing,
        "mlow": out.mlow,
        "mhigh": out.mhigh,
        "delta_prime_diag": json.dumps([[d["m"], round(d["psi_n"], 5), d["delta_prime_real"]] for d in diag]),
        "delta_prime_frobenius": None if dp is None else float(np.linalg.norm(dp)),
        "delta_prime_max": max((d["delta_prime_real"] for d in diag), default=None),
        "n_positive_delta_prime": sum(d["delta_prime_real"] > 0 for d in diag),
        "W_t_min": None if out.total1 is None else float(out.total1.real),
    }


def summarize(out: Path) -> Path:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/*/nn*/result.json"))]
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    table = out / "scan_controls.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eq", action="append", required=True, type=parse_equilibrium, help="SHOT:TIME_MS:GFILE")
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--only", nargs="*", default=None, help="Variant names to run (default: all).")
    args = parser.parse_args()

    variants = [v for v in VARIANTS if not args.only or v.name in args.only]
    out = args.out.expanduser().resolve()
    templates = {v.name: materialize_templates(v, out / "_templates") for v in variants}
    jobs = [(eq, v, n) for eq in args.eq for v in variants for n in args.modes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, eq, v, n, templates[v.name], out, args.timeout): (eq, v, n) for eq, v, n in jobs}
        for future in as_completed(futures):
            eq, v, n = futures[future]
            row = future.result()
            print(f"{eq.label} {v.name} n={n}: {row['status']} {row['wall_s']} s", flush=True)
    print(summarize(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
