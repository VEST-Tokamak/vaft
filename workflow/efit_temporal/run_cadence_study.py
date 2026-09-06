#!/usr/bin/env python3
"""Qualify the EFIT reconstruction cadence and constraint-averaging window (issue #468).

Two temporal controls, varied one at a time under one fixed, serialized EFIT
numerical configuration:

* the reconstruction cadence ``dt_EFIT`` -- the spacing of the slices;
* the constraint-averaging half-window ``w`` -- each constraint is the box
  average of the diagnostic samples inside ``[t_i - w, t_i + w]`` (#433).

For every case the script forms the constraints, writes the k-files, runs
EFIT once over all slices, and records per slice what the solver reported:
``jflag``/``lflag`` from the a-file, the iteration count and final
Grad-Shafranov residual from the m-file (``cerror``), the total chi-square,
and the wall-clock cost.  Convergence first, smoothness later: the first
question is where finer sampling stops being numerically useful, not what
the equilibrium looks like.

Everything else is held fixed and recorded: the scientific configuration
(``EFITScientificConfig``, with its digest), the initialization policy, the
diagnostic set and weights, the spatial tables.  #171 and #196 own those
settings; this study does not scan them.

Usage::

    PYTHONPATH=. python workflow/efit_temporal/run_cadence_study.py \\
        --eddy-ods /path/to/41524/eddy/omas.json.gz --shot 41524 \\
        --efit ~/git/efit/build-mac/efit/efit --output study/ \\
        --cadences 1.0,0.4,0.2,0.08 --windows 0.5 \\
        --window-scan 1.0,0.5,0.2,0.1

Cadences and windows are in milliseconds.  ``--window-scan`` runs the listed
half-windows at the first cadence.  The input's own ``equilibrium`` (if any)
is discarded: the constraints are rebuilt for every case.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time as _clock
from typing import Any

import numpy as np

import vaft
import vaft.omas
from vaft.code.efit import generate_constraints_ods
from vaft.code.efit.config import EFITScientificConfig
from vaft.code.efit.magnetic import EFITConfig, prepare_efit_inputs, resolved_efit_configuration, run_efit
from vaft.data import read_aeqdsk
from vaft.data.meqdsk import read_meqdsk

DEFAULT_UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
DEFAULT_WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]
DIAGNOSTIC_DT = 4e-5
WRAPPER = Path(__file__).resolve().parents[1] / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py"


def _wrapper():
    spec = importlib.util.spec_from_file_location("generate_constraints_ods_wrapper", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plasma_window(ods) -> tuple[float, float, dict]:
    """The routine constraint window: the shared analysis range ∩ the detected plasma."""
    window = _wrapper()._constraint_window(ods, policy=None)
    return float(window.start), float(window.end), dict(window.record)


def slice_times(start: float, end: float, cadence_s: float) -> np.ndarray:
    """Slices at ``cadence_s`` from ``start`` to ``end`` inclusive, snapped to the diagnostics grid."""
    n = int(np.floor((end - start) / cadence_s + 1e-9))
    times = start + cadence_s * np.arange(n + 1)
    return np.round(times / DIAGNOSTIC_DT) * DIAGNOSTIC_DT


_ITERATION = re.compile(r"\bt=\s*(\d+)\s+it=\s*(\d+)\s+chi2=\s*([0-9.E+-]+).*?err=\s*([0-9.E+-]+)")
_DONE = re.compile(r"Done processing")


def iterations_from_log(text: str) -> list[dict[str, Any]]:
    """Per-slice iteration count, last chi2 and last GS error from EFIT's terminal log.

    The local build is linked without NetCDF and writes no m-file, so the
    solver's own progress lines are the record: one ``it=`` line per outer
    iteration, in k-file order, with the counter restarting at 1 on every
    slice (EFIT prints the time in whole milliseconds only, so the counter,
    not the time, delimits the slices).  A boundary-finder error between two
    blocks is attributed to the block it follows.
    """
    blocks: list[dict[str, Any]] = []
    current: list[tuple[int, float, float]] = []

    def flush() -> None:
        if current:
            blocks.append({
                "iterations_n": max(it for it, _c, _e in current),
                "chi2_log": current[-1][1],
                "gs_error_log": current[-1][2],
                "bound_error": False,
            })

    for line in text.splitlines():
        found = _ITERATION.search(line)
        if found:
            if int(found.group(2)) == 1:
                flush()
                current = []
            current.append((int(found.group(2)), float(found.group(3)), float(found.group(4))))
            continue
        if "ERROR in bound" in line and current:
            flush()
            blocks[-1]["bound_error"] = True
            current = []
    flush()
    return blocks


def _key_us(name: str) -> int:
    """``k041524.00320_400`` -> 320400: sort k-files by time, not lexically."""
    key = name.split(".", 1)[1]
    ms, _, us = key.partition("_")
    return int(ms) * 1000 + (int(us) if us else 0)


def per_slice_metrics(workdir: Path, shot: int) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for afile in sorted(workdir.glob(f"a0{shot}.*")):
        a = read_aeqdsk(afile)
        key = afile.name.split(".", 1)[1]
        rows[key] = {
            "afile": afile.name,
            "time_ms": float(a.time_ms),
            "jflag": int(a.jflag),
            "lflag": int(a.lflag),
            "chisq": float(a.scalars.get("chisq", float("nan"))),
            "ipmhd": float(a.scalars.get("ipmhd", float("nan"))),
            "betap": float(a.scalars.get("betap", float("nan"))),
            "li": float(a.scalars.get("li", float("nan"))),
            "q95": float(a.scalars.get("q95", float("nan"))),
            "condno": float(a.scalars.get("condno", float("nan"))),
        }
    for mfile in sorted(workdir.glob(f"m0{shot}.*")):
        key = mfile.name.split(".", 1)[1]
        try:
            m = read_meqdsk(mfile)
        except Exception as error:  # a truncated m-file is a finding, not a crash
            rows.setdefault(key, {})["mfile_error"] = str(error)[:120]
            continue
        row = rows.setdefault(key, {})
        cerror = m._at("cerror", 0)
        if cerror is not None:
            errors = np.asarray(cerror, dtype=float).reshape(-1)
            errors = errors[np.isfinite(errors)]
            row["iterations_n"] = int(errors.size)
            row["gs_residual_final"] = float(errors[-1]) if errors.size else float("nan")
        for name in ("chitot", "chifin", "cchisq"):
            value = m._at(name, 0)
            if value is not None:
                array = np.asarray(value, dtype=float).reshape(-1)
                if array.size:
                    row["chi_total"] = float(array[-1])
                    break
    return [rows[key] for key in sorted(rows)]


def run_case(
    source, *, shot: int, times: np.ndarray, window_s: float, workdir: Path,
    efit: str, tables: str, scientific: EFITScientificConfig, uncertainty, weighting,
) -> dict[str, Any]:
    ods = copy.deepcopy(source)
    if "equilibrium" in ods:
        del ods["equilibrium"]
    ods["equilibrium.time"] = times
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    started = _clock.perf_counter()
    generate_constraints_ods(
        ods, shot, str(workdir), tables, times, list(uncertainty), list(weighting),
        broken=[], fit=0, average_window=window_s,
    )
    constraints_seconds = _clock.perf_counter() - started
    config = EFITConfig(
        executable=efit, workdir=workdir, shot=shot, times=times.tolist(), args=("129",),
        npprime=scientific.profile.kppcur, nffprime=scientific.profile.kffcur,
    )
    inputs = prepare_efit_inputs(ods, config)
    started = _clock.perf_counter()
    result = run_efit(inputs, config)
    efit_seconds = _clock.perf_counter() - started
    slices = per_slice_metrics(workdir, shot)
    log_path = workdir / "run_efit.out"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else result.stdout
    progress = iterations_from_log(log_text)
    bound_errors = log_text.count("ERROR in bound")
    # k-files are processed in stdin order == ascending time; a-files exist
    # only for slices EFIT wrote, so align the log blocks on the k-file order.
    kfile_keys = [path.name.split(".", 1)[1] for path in sorted(inputs.kfiles, key=lambda p: _key_us(p.name))]
    by_key = {key: block for key, block in zip(kfile_keys, progress)}
    # The constraint current per slice, so a slice EFIT never wrote can be
    # explained (below CUTIP) rather than merely counted as failed.
    constraints_ip = {}
    try:
        from omas import load_omas_json

        product = load_omas_json(str(workdir / f"{shot}_constraints.json"), consistency_check=False)
        for i, t_slice in enumerate(np.asarray(product["equilibrium.time"], dtype=float)):
            constraints_ip[int(round(t_slice * 1e6))] = float(product[f"equilibrium.time_slice.{i}.constraints.ip.measured"])
    except Exception:
        pass
    by_afile = {row["afile"].split(".", 1)[1]: row for row in slices if row.get("afile")}
    slices = []
    for key, t_slice in zip(kfile_keys, times):
        row = by_afile.get(key, {"afile": None, "time_ms": float(t_slice * 1e3), "jflag": 0, "lflag": None})
        row["time_s"] = float(t_slice)
        row["kfile"] = f"k0{shot}.{key}"
        row["constraint_ip"] = constraints_ip.get(int(round(t_slice * 1e6)))
        row.update(by_key.get(key, {}))
        slices.append(row)
    converged = [row for row in slices if row.get("jflag") == 1]
    iterations = [block["iterations_n"] for block in progress if block["iterations_n"]]
    chisq = [row["chisq"] for row in converged if np.isfinite(row.get("chisq", np.nan))]
    return {
        "n_slices": int(times.size),
        "cadence_ms": float(np.median(np.diff(times)) * 1e3) if times.size > 1 else None,
        "window_ms": window_s * 1e3,
        "samples_per_slice_window": int(round(2 * window_s / DIAGNOSTIC_DT)) + 1,
        "returncode": result.returncode,
        "status": result.status,
        "afiles": len(result.afiles),
        "mfiles": len(result.mfiles),
        "log_slices": len(progress),
        "bound_errors": bound_errors,
        "converged": len(converged),
        "failed": int(times.size) - len(converged),
        "below_current_cut": sum(1 for row in slices if row.get("constraint_ip") is not None and row["constraint_ip"] < 50000.0),
        "iterations": {
            "median": float(np.median(iterations)) if iterations else None,
            "max": int(max(iterations)) if iterations else None,
        },
        "chisq": {
            "median": float(np.median(chisq)) if chisq else None,
            "max": float(max(chisq)) if chisq else None,
        },
        "seconds": {"constraints": constraints_seconds, "efit": efit_seconds,
                    "efit_per_slice": efit_seconds / max(int(times.size), 1)},
        "configuration": resolved_efit_configuration(config),
        "slices": slices,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| case | cadence [ms] | window [ms] | slices | below 50 kA | converged | bound errors | median iter | max iter | median chi2 | EFIT s/slice |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, case in payload["cases"].items():
        chi = case["chisq"]["median"]
        lines.append(
            f"| {name} | {case['cadence_ms']:.2f} | {case['window_ms']:.2f} | {case['n_slices']} | "
            f"{case['below_current_cut']} | {case['converged']} | {case['bound_errors']} | "
            f"{case['iterations']['median']} | {case['iterations']['max']} | "
            f"{'-' if chi is None else f'{chi:.3g}'} | {case['seconds']['efit_per_slice']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eddy-ods", required=True, type=Path)
    parser.add_argument("--shot", required=True, type=int)
    parser.add_argument("--efit", required=True, help="EFIT executable.")
    parser.add_argument("--tables", default=str(Path(vaft.__file__).parent / "data" / "efit") + "/")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cadences", default="1.0,0.4,0.2,0.08", help="ms, comma-separated.")
    parser.add_argument("--windows", default="0.5", help="Half-windows [ms] used with every cadence.")
    parser.add_argument("--window-scan", default="", help="Half-windows [ms] run at the first cadence only.")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--max-slices", type=int, default=None, help="Truncate every case to this many slices (smoke runs).")
    args = parser.parse_args(argv)

    source = vaft.omas.load(str(args.eddy_ods))
    if args.tstart is None or args.tend is None:
        start, end, record = plasma_window(source)
    else:
        start, end, record = float(args.tstart), float(args.tend), {"source": "argument"}
    scientific = EFITScientificConfig()
    cadences = [float(v) for v in args.cadences.split(",") if v.strip()]
    windows = [float(v) for v in args.windows.split(",") if v.strip()]
    scan = [float(v) for v in args.window_scan.split(",") if v.strip()]
    plan = [(c, w) for c in cadences for w in windows] + [(cadences[0], w) for w in scan if w not in windows]

    args.output.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    for cadence_ms, window_ms in plan:
        times = slice_times(start, end, cadence_ms * 1e-3)
        if args.max_slices:
            times = times[: args.max_slices]
        name = f"dt{cadence_ms:g}ms_w{window_ms:g}ms"
        print(f"{name}: {times.size} slices {times[0]:.4f}-{times[-1]:.4f} s", flush=True)
        cases[name] = run_case(
            source, shot=args.shot, times=times, window_s=window_ms * 1e-3, workdir=args.output / name,
            efit=args.efit, tables=args.tables, scientific=scientific,
            uncertainty=DEFAULT_UNCERTAINTY, weighting=DEFAULT_WEIGHTING,
        )
        c = cases[name]
        print(f"  converged {c['converged']}/{c['n_slices']}, EFIT {c['seconds']['efit']:.1f} s", flush=True)
    payload = {
        "schema_version": 1,
        "shot": args.shot,
        "window": {"start": start, "end": end, "record": record},
        "held_fixed": {
            "scientific_sha256": scientific.sha256,
            "scientific": scientific.to_dict(),
            "initialization": "independent slices (k-file default)",
            "tables": args.tables,
            "diagnostic_dt_s": DIAGNOSTIC_DT,
            "uncertainty": DEFAULT_UNCERTAINTY,
            "weighting": DEFAULT_WEIGHTING,
            "channel_selection": "projected validity only (no manual list)",
        },
        "cases": cases,
    }
    (args.output / "study.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    (args.output / "study.md").write_text(markdown(payload))
    print(markdown(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
