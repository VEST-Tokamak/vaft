"""How a VEST EFIT fit terminates today, over whole discharges (issue #171).

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_numerics/baseline_termination.py \\
            --shots 39915 --output /scratch/baseline --markdown /scratch/baseline.md

#171 asks for VEST's EFIT solver configuration to be characterized and then
stated explicitly instead of inherited. This is the characterization half, and
it deliberately changes nothing: it runs the routine configuration
(``EFITScientificConfig()``) over each shot's full constraint window at the
routine cadence and records *why* every slice stopped where it did.

The unit is a discharge, not a slice. A slice that fails in the boundary
finder during the ramp and a slice that fits cleanly in the flat-top are
different facts about the same configuration, and a study that averaged them
would report neither.

What the log is asked for, per slice:

* the **exit path** of the outer loop -- ``iconvr=2 satisfied`` (the
  chi-square criterion), exhaustion of ``MXITER``, or a solver error;
* the **acceptance failures** ``chkerr`` reports afterwards, by number, so
  "did not converge" is replaced by which of EFIT's twenty-two criteria was
  violated and by how much;
* iterations, last chi-square and last Grad-Shafranov error, so the exit can
  be read against the ``ERROR`` the k-file asked for.

Nothing here varies a setting. Varying ``ERRMIN``, ``SAICON``, ``NXITER``,
``MXITER`` and ``RELAX`` is the next step, and it needs this baseline to be
measured against.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import time as _clock
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
CONSTRAINTS_WRAPPER = (
    REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py"
)
SCHEMA = 1

DEFAULT_UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
DEFAULT_WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]
DEFAULT_TSTEP = 0.001
DEFAULT_WINDOW = 0.0005

_ITERATION = re.compile(r"\bt=\s*(\d+)\s+it=\s*(\d+)\s+chi2=\s*([0-9.E+-]+).*?err=\s*([0-9.E+-]+)")
_ICONVR = re.compile(r"iconvr=(\d+) satisfied")
_FAILED = re.compile(r"Failed to reach fit/convergence criteria, shot\s+(\d+)\s+([0-9.]+)")
_FAILURE = re.compile(r"Failure #(\d+),\s*([^=]*?)(?:=\s*([0-9.E+-]+))?\s*$")
_SOLVER_ERROR = re.compile(r"ERROR in (\w+) at r=\s*\d+, t=\s*(\d+): (.*)")


def _module(path: Path, name: str):
    """Load a path-run script, registered so its dataclasses resolve."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_slices(text: str) -> list[dict[str, Any]]:
    """Per-slice termination evidence, in the order EFIT processed them.

    EFIT prints the time in whole milliseconds only, so slices are delimited
    by the iteration counter restarting at 1, not by the time. Everything
    printed after a slice's last iteration and before the next slice's first
    belongs to that slice: its exit path, its solver errors and the acceptance
    failures ``chkerr`` reports.

    With one exception, and it is not a small one. A slice that fails in
    ``bound`` before the first Picard iteration prints no iteration line at
    all, only ``ERROR in bound at r=..., t=...``. Delimiting on the iteration
    counter alone hands that error to the *previous* slice and loses the slice
    itself, so a run ending in a run of pre-iteration collapses reports both a
    short universe and a slice carrying failures that are not its own. Solver
    errors therefore also open a slice when they name a time the current slice
    does not have.
    """
    slices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def start(time_ms: int) -> dict[str, Any]:
        return {
            "time_ms": time_ms,
            "iterations": [],
            "exit_path": None,
            "iconvr": None,
            "accepted": None,
            "failures": [],
            "solver_errors": [],
        }

    for line in text.splitlines():
        found = _ITERATION.search(line)
        if found:
            time_ms, iteration = int(found.group(1)), int(found.group(2))
            if iteration == 1:
                if current is not None:
                    slices.append(current)
                current = start(time_ms)
            if current is None:
                current = start(time_ms)
            current["iterations"].append(
                {"n": iteration, "chi2": float(found.group(3)), "gs_error": float(found.group(4))}
            )
            continue
        found = _SOLVER_ERROR.search(line)
        if found:
            named = int(found.group(2))
            if current is None or current["time_ms"] != named:
                if current is not None:
                    slices.append(current)
                current = start(named)
            current["solver_errors"].append({"routine": found.group(1), "detail": found.group(3).strip()})
            continue
        if current is None:
            continue
        found = _ICONVR.search(line)
        if found:
            current["iconvr"] = int(found.group(1))
            current["exit_path"] = f"iconvr={found.group(1)}"
            continue
        if _FAILED.search(line):
            current["accepted"] = False
            continue
        found = _FAILURE.search(line.strip())
        if found:
            current["failures"].append(
                {
                    "code": int(found.group(1)),
                    "criterion": found.group(2).strip().rstrip(",").strip(),
                    "value": float(found.group(3)) if found.group(3) else None,
                }
            )
    if current is not None:
        slices.append(current)

    for record in slices:
        iterations = record["iterations"]
        record["iterations_n"] = max((item["n"] for item in iterations), default=0)
        # The log's chi2 is not the fit's chi-square after the first step: on a
        # slice that collapses to a null solution it falls to ~1e-7 while the
        # a-file reports 200. So both ends are kept and neither is called
        # "the" chi-square -- EFIT's own answer is the a-file's, joined below.
        record["chi2_initial"] = iterations[0]["chi2"] if iterations else None
        record["chi2_final"] = iterations[-1]["chi2"] if iterations else None
        record["gs_error"] = iterations[-1]["gs_error"] if iterations else None
        # A null solution: the residual vanishes while the Grad-Shafranov
        # error does not, and the axis never leaves zero. It is not a fit.
        record["collapsed"] = bool(
            iterations
            and record["chi2_final"] is not None
            and record["chi2_final"] < 1.0e-5
            and record["gs_error"] is not None
            and record["gs_error"] > 0.1
        )
        if record["accepted"] is None:
            record["accepted"] = not record["failures"] and not record["solver_errors"]
        if record["exit_path"] is None:
            record["exit_path"] = "solver_error" if record["solver_errors"] else "iterations_exhausted"
        del record["iterations"]
    return slices


def run_shot(
    shot: int,
    product: Path,
    *,
    workdir: Path,
    efit: str,
    tables: str,
    tstep: float,
    average_window: float,
) -> dict[str, Any]:
    """Reconstruct one whole discharge with the routine configuration."""
    from omas import load_omas_json

    from vaft.code.efit import generate_constraints_ods
    from vaft.code.efit.config import EFITScientificConfig
    from vaft.code.efit.magnetic import (
        EFITConfig,
        prepare_efit_inputs,
        resolved_efit_configuration,
        run_efit,
    )

    wrapper = _module(CONSTRAINTS_WRAPPER, "generate_constraints_ods_wrapper")

    import gzip
    import tempfile

    with gzip.open(product, "rt", encoding="utf-8") as handle:
        payload = handle.read()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        staged.write(payload)
        staged_path = staged.name
    try:
        ods = load_omas_json(staged_path, consistency_check=False)
    finally:
        Path(staged_path).unlink(missing_ok=True)

    times, window = wrapper._select_times(ods, "auto", tstep, None, None)
    times = np.asarray(times, dtype=float)

    source = copy.deepcopy(ods)
    if "equilibrium" in source:
        del source["equilibrium"]
    source["equilibrium.time"] = times

    # The packaged products carry a projected validity for none of the
    # channels EFIT sees, so a constraint build that reads one finds nothing
    # and fits every channel -- including the probes the assessment condemns.
    # Assess and gate into a copy, then decide on that (#189, #296). Without
    # this the baseline would characterize a fit that includes a probe forty
    # times out of family, which is a statement about the input, not the
    # solver.
    from vaft.omas.vacuum_magnetics import quality_gate
    from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count

    gated, gate = quality_gate(source, window=(float(times[0]), float(times[-1])))
    nbprobe = efit_probe_count(gated)
    decisions = decide_efit_channels(gated, times, nbprobe=nbprobe)
    source = gated

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    started = _clock.perf_counter()
    generate_constraints_ods(
        source,
        shot,
        str(workdir),
        tables,
        times,
        list(DEFAULT_UNCERTAINTY),
        list(DEFAULT_WEIGHTING),
        decisions=decisions,
        average_window=average_window,
    )
    constraints_seconds = _clock.perf_counter() - started

    scientific = EFITScientificConfig()
    config = EFITConfig(
        executable=efit,
        workdir=workdir,
        shot=shot,
        times=times.tolist(),
        args=("129",),
        npprime=scientific.profile.kppcur,
        nffprime=scientific.profile.kffcur,
    )
    inputs = prepare_efit_inputs(source, config)
    started = _clock.perf_counter()
    result = run_efit(inputs, config)
    efit_seconds = _clock.perf_counter() - started

    log = workdir / "run_efit.out"
    text = log.read_text(errors="replace") if log.is_file() else result.stdout
    slices = parse_slices(text)

    # The discharge includes a pre-plasma stretch. A slice there "converges"
    # with zero plasma current, which is a vacuum solution and not evidence
    # about the fit; summarizing the two together would report neither. The
    # cut is the writer's own CUTIP, so this classification is the one EFIT
    # already applies.
    cut = float(scientific.initialization.current_threshold)
    constraint_ip: dict[int, float] = {}
    try:
        constraints = load_omas_json(str(workdir / f"{shot}_constraints.json"), consistency_check=False)
        for index, value in enumerate(np.asarray(constraints["equilibrium.time"], dtype=float)):
            constraint_ip[int(round(float(value) * 1e3))] = float(
                constraints[f"equilibrium.time_slice.{index}.constraints.ip.measured"]
            )
    except Exception:
        constraint_ip = {}
    for record in slices:
        current = constraint_ip.get(int(record["time_ms"]))
        record["constraint_ip"] = current
        record["phase"] = (
            "unknown" if current is None else ("vacuum" if abs(current) < cut else "plasma")
        )

    from vaft.data import read_aeqdsk

    afiles = {path.name.split(".", 1)[1]: path for path in sorted(workdir.glob(f"a0{shot}.*"))}
    accepted = []
    for key, path in afiles.items():
        record = read_aeqdsk(path)
        accepted.append(
            {
                "afile": path.name,
                "time_ms": float(record.time_ms),
                "jflag": int(record.jflag),
                "lflag": int(record.lflag),
                "chisq": float(record.scalars.get("chisq", float("nan"))),
                "terror": float(record.scalars.get("terror", float("nan"))),
                "ipmhd": float(record.scalars.get("ipmhd", float("nan"))),
            }
        )

    by_time = {int(round(item["time_ms"])): item for item in accepted}
    for record in slices:
        flags = by_time.get(int(record["time_ms"]))
        record["afile"] = None if flags is None else {
            "jflag": flags["jflag"],
            "lflag": flags["lflag"],
            "chisq": flags["chisq"],
            "terror": flags["terror"],
            "ipmhd": flags["ipmhd"],
        }

    return {
        "shot": shot,
        "current_cut": cut,
        "product": str(product.relative_to(REPOSITORY)) if product.is_relative_to(REPOSITORY) else str(product),
        "window": {
            "start": float(window.start) if window else float(times[0]),
            "end": float(window.end) if window else float(times[-1]),
            "source": str(window.source) if window else "manual",
            "tstep": float(tstep),
            "average_window": float(average_window),
        },
        "slices_requested": int(times.size),
        "channels": {
            "efit_facing": int(nbprobe) + 11,
            "excluded_by_the_gate": [entry["channel"] for entry in gate.record().get("excluded", [])],
            "rejected_at_some_slice": sorted(
                f"{kind}[{index}]"
                for (kind, index), decision in decisions.entries.items()
                if bool((decision.state == 2).any())
            ),
        },
        "returncode": result.returncode,
        "status": result.status,
        "afiles": len(afiles),
        "seconds": {"constraints": constraints_seconds, "efit": efit_seconds},
        "configuration": resolved_efit_configuration(config),
        "slices": slices,
        "afile_flags": accepted,
    }


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    numbers = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not numbers:
        return {"min": None, "median": None, "max": None, "n": 0}
    return {
        "min": float(min(numbers)),
        "median": float(np.median(numbers)),
        "max": float(max(numbers)),
        "n": len(numbers),
    }


def _phase_summary(slices: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures = Counter(
        f"#{failure['code']} {failure['criterion']}" for item in slices for failure in item["failures"]
    )
    solver = Counter(error["routine"] for item in slices for error in item["solver_errors"])
    return {
        "slices": len(slices),
        "exit_paths": dict(Counter(item["exit_path"] for item in slices)),
        "clean_exits": sum(1 for item in slices if item["accepted"]),
        "failures": dict(failures.most_common()),
        "solver_errors": dict(solver.most_common()),
        "collapsed": sum(1 for item in slices if item["collapsed"]),
        "with_afile": sum(1 for item in slices if item["afile"]),
        "iterations": _spread([item["iterations_n"] for item in slices if item["iterations_n"]]),
        "chi2_initial": _spread([item["chi2_initial"] for item in slices]),
        "afile_chisq": _spread([item["afile"]["chisq"] for item in slices if item["afile"]]),
        "afile_terror": _spread([item["afile"]["terror"] for item in slices if item["afile"]]),
        "gs_error": _spread([item["gs_error"] for item in slices]),
    }


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """What the discharge says about the configuration, not about one slice.

    Split by phase, because the two say different things: a vacuum slice is
    accepted with zero plasma current and no fit to speak of, and averaging it
    with a flat-top slice reports neither.
    """
    slices = record["slices"]
    plasma = [item for item in slices if item["phase"] == "plasma"]
    vacuum = [item for item in slices if item["phase"] == "vacuum"]
    by_time = {int(round(item["time_ms"])): item for item in record["afile_flags"]}
    jflag_by_phase: dict[str, Counter] = {"plasma": Counter(), "vacuum": Counter(), "unknown": Counter()}
    for item in slices:
        flags = by_time.get(int(item["time_ms"]))
        if flags is not None:
            jflag_by_phase[item["phase"]][int(flags["jflag"])] += 1
    return {
        "slices_logged": len(slices),
        "slices_requested": record["slices_requested"],
        "afiles": record["afiles"],
        "phases": {name: len([item for item in slices if item["phase"] == name]) for name in ("plasma", "vacuum", "unknown")},
        "plasma": _phase_summary(plasma),
        "vacuum": _phase_summary(vacuum),
        "jflag": {
            phase: {str(key): value for key, value in sorted(counter.items())}
            for phase, counter in jflag_by_phase.items()
            if counter
        },
        "plasma_current": _spread([item["constraint_ip"] for item in slices if item["phase"] == "plasma"]),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# EFIT termination baseline (#171)", ""]
    lines.append(
        "The routine configuration, unchanged, over each discharge's full constraint window. "
        "Nothing here is varied; this is what the studies are measured against."
    )
    lines.append("")
    tolerance = payload["requested_error_tolerance"]
    lines.append(
        f"Every k-file asks for `ERROR = {tolerance}`. EFIT's own `ERRMIN` default is 1e-2 and its "
        "`SAICON` default is 80, neither of which VEST sets."
    )
    lines.append("")
    lines.append("| shot | window [s] | slices | plasma | vacuum | a-files | plasma exits | iterations (med) | a-file chi2 (med) | GS error (med) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for record in payload["shots"]:
        summary = record["summary"]
        plasma = summary["plasma"]
        paths = ", ".join(f"{name} {count}" for name, count in plasma["exit_paths"].items()) or "–"
        lines.append(
            f"| {record['shot']} | {record['window']['start']:.3f}–{record['window']['end']:.3f} "
            f"| {summary['slices_requested']} | {summary['phases']['plasma']} | {summary['phases']['vacuum']} "
            f"| {summary['afiles']} | {paths} "
            f"| {_fmt(plasma['iterations']['median'])} | {_fmt(plasma['afile_chisq']['median'])} "
            f"| {_fmt(plasma['gs_error']['median'])} |"
        )
    lines.append("")
    for record in payload["shots"]:
        summary = record["summary"]
        lines.append(f"## {record['shot']}")
        lines.append("")
        lines.append(
            f"- {summary['slices_requested']} slices at {record['window']['tstep'] * 1e3:.0f} ms over "
            f"{record['window']['start']:.3f}–{record['window']['end']:.3f} s "
            f"({record['window']['source']}); {summary['afiles']} a-files written"
        )
        lines.append(
            f"- {summary['phases']['plasma']} plasma slices and {summary['phases']['vacuum']} below the "
            f"{record['current_cut']:.0f} A cut; plasma current "
            f"{_fmt(summary['plasma_current']['min'])}–{_fmt(summary['plasma_current']['max'])} A"
        )
        for phase in ("plasma", "vacuum"):
            block = summary[phase]
            if not block["slices"]:
                continue
            lines.append(f"- **{phase}** ({block['slices']} slices)")
            lines.append(
                f"  - exits: " + ", ".join(f"{name} × {count}" for name, count in block["exit_paths"].items())
            )
            lines.append(
                f"  - iterations {_fmt(block['iterations']['min'])}–{_fmt(block['iterations']['max'])} "
                f"(median {_fmt(block['iterations']['median'])}); GS error "
                f"{_fmt(block['gs_error']['min'])}–{_fmt(block['gs_error']['max'])} "
                f"against the requested {payload['requested_error_tolerance']}"
            )
            lines.append(
                f"  - {block['with_afile']} of {block['slices']} wrote an a-file; EFIT's own chi-square "
                f"{_fmt(block['afile_chisq']['min'])}–{_fmt(block['afile_chisq']['max'])} "
                f"(median {_fmt(block['afile_chisq']['median'])}), terror "
                f"{_fmt(block['afile_terror']['min'])}–{_fmt(block['afile_terror']['max'])}"
            )
            if block["collapsed"]:
                lines.append(
                    f"  - {block['collapsed']} slices collapsed to a null solution: the residual falls "
                    "below 1e-5 while the Grad-Shafranov error stays above 0.1"
                )
            if summary["jflag"].get(phase):
                lines.append(
                    "  - a-file jflag: "
                    + ", ".join(f"{key} × {value}" for key, value in summary["jflag"][phase].items())
                )
            if block["failures"]:
                lines.append("  - acceptance failures:")
                for name, count in block["failures"].items():
                    lines.append(f"    - {name}: {count} slices")
            if block["solver_errors"]:
                lines.append(
                    "  - solver errors: "
                    + ", ".join(f"{name} × {count}" for name, count in block["solver_errors"].items())
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", default=None, help="comma-separated; default: the reference set's magnetics shots")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--tables", default=None, help="EFIT table directory (default: the packaged one)")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=DEFAULT_TSTEP)
    parser.add_argument("--average-window", type=float, default=DEFAULT_WINDOW)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.config import EFITScientificConfig
    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    tables = args.tables or str(Path(data_path("efit")).resolve()) + "/"

    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(value) for value in args.shots.split(",")] if args.shots else sorted(products)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for shot in shots:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        print(f"{shot}: reconstructing the whole discharge ...", flush=True)
        record = run_shot(
            shot,
            Path(data_path(products[shot])),
            workdir=output / f"shot_{shot}",
            efit=str(resolved["efit"]),
            tables=tables,
            tstep=args.tstep,
            average_window=args.average_window,
        )
        record["summary"] = summarize(record)
        records.append(record)
        summary = record["summary"]
        print(
            f"  {summary['slices_requested']} slices ({summary['phases']['plasma']} plasma, "
            f"{summary['phases']['vacuum']} vacuum), {summary['afiles']} a-files; "
            f"plasma exits {summary['plasma']['exit_paths']}",
            flush=True,
        )

    payload = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "tables": tables,
        "scientific": EFITScientificConfig().to_dict(),
        "scientific_sha256": EFITScientificConfig().sha256,
        "requested_error_tolerance": EFITScientificConfig().numerics.error_tolerance,
        "shots": records,
    }
    (output / "baseline.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    text = markdown(payload)
    (args.markdown or output / "baseline.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
