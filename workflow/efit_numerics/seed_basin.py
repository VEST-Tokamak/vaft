"""Where should a VEST EFIT solve start, and how far can that move? (#588)

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_numerics/seed_basin.py --output /scratch/seed \\
            --table test/data/efit_seed_basin.json

Every VEST slice is seeded from the same fixed ellipse -- ``RELIP 0.4``,
``ZELIP 0.0``, ``AELIP 0.3``, ``EELIP 1.6`` -- and 29 of the 77 plasma slices
in the reference set collapse to a null solution and fail in ``bound``,
producing nothing at all. A 0.3 m minor radius is most of the machine, seeded
into a discharge that is 7 kA and small when the collapse block begins.

This study varies **the seed and nothing else**. Issue #588 is explicit that
seed geometry, temporal continuation (#196) and termination (#171) must stay
separate studies, because scanning them together makes any change
unattributable. So there is no ``ICINIT`` here and no numerics setting.

Two populations are asked different questions:

* the **collapse block** -- does any seed make these slices produce an
  equilibrium at all?
* the slices that **already converge** -- how far can the seed move before the
  answer changes? That is the basin, and it is the half that says whether the
  present seed is well chosen or merely lucky.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time as _clock
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
BASELINE = REPOSITORY / "workflow" / "efit_numerics" / "baseline_termination.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_seed_basin.json"

#: The routine seed, and the point every sweep is centred on.
ROUTINE_SEED = {
    "ellipse_rzero": 0.4,
    "zzero": 0.0,
    "minor_radius": 0.3,
    "elongation": 1.6,
}

#: One axis at a time. Five crossed axes would be hundreds of runs for less
#: information than the marginals give first.
AXES: dict[str, tuple[float, ...]] = {
    "initialization.minor_radius": (0.10, 0.15, 0.20, 0.25, 0.30),
    # `ellipse_rzero`, not `rzero`: the latter also drives RZERO, RCENTR and
    # through it BTOR, so sweeping it would move the seed, the normalisation
    # and the vacuum toroidal field together and attribute nothing.
    "initialization.ellipse_rzero": (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    "initialization.elongation": (1.0, 1.3, 1.6, 2.0),
    "initialization.zzero": (-0.05, 0.0, 0.05),
}

#: The 2-D map, run on one slice from each population.
MAP_AXES = {
    "initialization.ellipse_rzero": (0.30, 0.35, 0.40, 0.45),
    "initialization.minor_radius": (0.10, 0.15, 0.20, 0.25, 0.30),
}


def _module(path: Path, name: str):
    """Load a path-run script, registered so its dataclasses resolve."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_product(path: Path):
    from omas import load_omas_json

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = handle.read()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        staged.write(payload)
        name = staged.name
    try:
        return load_omas_json(name, consistency_check=False)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare_shot(shot: int, product: Path, *, workdir: Path, tables: str, tstep: float, average_window: float):
    """Build the constraints once; every seed reuses them.

    The seed lives in the k-file namelist, not in the constraints, so building
    them per configuration would cost minutes and change nothing.
    """
    from vaft.code.efit import generate_constraints_ods
    from vaft.omas.vacuum_magnetics import quality_gate
    from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count

    baseline = _module(BASELINE, "baseline_termination")
    wrapper = _module(
        REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py",
        "generate_constraints_ods_wrapper",
    )

    ods = load_product(product)
    times, window = wrapper._select_times(ods, "auto", tstep, None, None)
    times = np.asarray(times, dtype=float)

    source = copy.deepcopy(ods)
    if "equilibrium" in source:
        del source["equilibrium"]
    source["equilibrium.time"] = times

    # These products carry no stored validity for any channel EFIT sees, so
    # assess and gate rather than trusting what we are handed (#189, #296).
    gated, _ = quality_gate(source, window=(float(times[0]), float(times[-1])))
    decisions = decide_efit_channels(gated, times, nbprobe=efit_probe_count(gated))

    workdir.mkdir(parents=True, exist_ok=True)
    generate_constraints_ods(
        gated,
        shot,
        str(workdir),
        tables,
        times,
        list(baseline.DEFAULT_UNCERTAINTY),
        list(baseline.DEFAULT_WEIGHTING),
        decisions=decisions,
        average_window=average_window,
    )
    return gated, times, window, baseline


def run_seed(
    constraints_ods,
    *,
    shot: int,
    times: np.ndarray,
    workdir: Path,
    efit: str,
    scientific,
    baseline,
) -> dict[str, Any]:
    """One seed, one whole discharge; returns the per-slice outcome."""
    from vaft.code.efit.magnetic import EFITConfig, prepare_efit_inputs, run_efit

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    # Every part of the scientific configuration has to be handed over, not
    # only the profile orders: passing npprime alone leaves the seed at its
    # default and the sweep silently measures nothing.
    config = EFITConfig(
        executable=efit,
        workdir=workdir,
        shot=shot,
        times=times.tolist(),
        args=("129",),
        profile=scientific.profile,
        initialization=scientific.initialization,
        numerics=scientific.numerics,
        constraints=scientific.constraints,
    )
    inputs = prepare_efit_inputs(copy.deepcopy(constraints_ods), config)
    started = _clock.perf_counter()
    result = run_efit(inputs, config)
    seconds = _clock.perf_counter() - started

    log = workdir / "run_efit.out"
    text = log.read_text(errors="replace") if log.is_file() else result.stdout
    slices = baseline.parse_slices(text)

    from vaft.data import read_aeqdsk

    by_time: dict[int, dict[str, Any]] = {}
    for path in sorted(workdir.glob(f"a0{shot}.*")):
        record = read_aeqdsk(path)
        by_time[int(round(float(record.time_ms)))] = {
            "jflag": int(record.jflag),
            "chisq": float(record.scalars.get("chisq", float("nan"))),
            "rmagx": float(record.scalars.get("rm", float("nan"))),
            "zmagx": float(record.scalars.get("zm", float("nan"))),
            "aminor": float(record.scalars.get("aminor", float("nan"))),
            "elong": float(record.scalars.get("elong", float("nan"))),
        }
    for record in slices:
        record["afile"] = by_time.get(int(record["time_ms"]))
    return {"seconds": seconds, "returncode": result.returncode, "slices": slices}


def outcome(record: dict[str, Any]) -> str:
    """What became of one slice, in one word."""
    if record["collapsed"]:
        return "collapsed"
    if record["afile"] is None:
        return "no_output"
    return "accepted" if record["afile"]["jflag"] == 1 else "flagged"


def summarize(slices: Sequence[dict[str, Any]], reference: dict[int, str] | None = None) -> dict[str, Any]:
    counts = Counter(outcome(item) for item in slices)
    produced = sum(1 for item in slices if item["afile"])
    record = {
        "slices": len(slices),
        "outcomes": dict(sorted(counts.items())),
        "produced_an_equilibrium": produced,
        "collapsed": counts.get("collapsed", 0),
        "accepted": counts.get("accepted", 0),
        "bound_failures": sum(
            1 for item in slices for error in item["solver_errors"] if error["routine"] == "bound"
        ),
        "findax_failures": sum(
            1 for item in slices for error in item["solver_errors"] if error["routine"] == "findax"
        ),
    }
    if reference is not None:
        # What the seed changed, slice by slice, against the routine seed.
        moved = {
            int(item["time_ms"]): [reference.get(int(item["time_ms"])), outcome(item)]
            for item in slices
            if reference.get(int(item["time_ms"])) != outcome(item)
        }
        record["changed_vs_routine"] = moved
        record["recovered"] = sorted(
            time for time, (before, after) in moved.items()
            if before in {"collapsed", "no_output"} and after in {"accepted", "flagged"}
        )
        record["lost"] = sorted(
            time for time, (before, after) in moved.items()
            if before in {"accepted", "flagged"} and after in {"collapsed", "no_output"}
        )
    return record


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# The first-slice seed and the convergence basin (#588)", ""]
    lines.append(
        "Every slice is seeded from the same fixed ellipse. This varies the seed and nothing else: "
        "no `ICINIT`, no termination setting, so any change is attributable to the seed."
    )
    lines.append("")
    lines.append(f"Routine seed: {payload['routine_seed']}.")
    lines.append("")
    for shot, block in payload["shots"].items():
        lines.append(f"## {shot}")
        lines.append("")
        routine = block["routine"]
        lines.append(
            f"Routine seed: {routine['produced_an_equilibrium']} of {routine['slices']} slices produce an "
            f"equilibrium, {routine['accepted']} accepted, {routine['collapsed']} collapsed, "
            f"{routine['bound_failures']} `bound` failures."
        )
        lines.append("")
        lines.append("| axis | value | produced | accepted | collapsed | recovered | lost |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in block["sweep"]:
            lines.append(
                f"| {row['axis'].split('.')[-1]} | {row['value']} | {row['summary']['produced_an_equilibrium']} "
                f"| {row['summary']['accepted']} | {row['summary']['collapsed']} "
                f"| {len(row['summary'].get('recovered', []))} | {len(row['summary'].get('lost', []))} |"
            )
        lines.append("")
    verdict = payload["verdict"]
    lines.append("## What this says")
    lines.append("")
    for line in verdict:
        lines.append(f"- {line}")
    return "\n".join(lines) + "\n"


def verdict_lines(payload: dict[str, Any]) -> list[str]:
    recovered_any = 0
    best: tuple[int, str] | None = None
    for shot, block in payload["shots"].items():
        for row in block["sweep"]:
            count = len(row["summary"].get("recovered", []))
            recovered_any += count
            if count and (best is None or count > best[0]):
                best = (count, f"{shot} {row['axis'].split('.')[-1]}={row['value']}")
    lines = []
    if recovered_any == 0:
        lines.append(
            "**No seed in the sweep recovers a single collapsed slice.** The collapse is not a "
            "seed problem, and the leading block belongs with the boundary tracer (#459) rather "
            "than with initialization."
        )
    else:
        lines.append(
            f"**The seed recovers slices**: {recovered_any} slice-outcomes improved across the sweep, "
            f"best at {best[1]}. The collapse is at least partly an initialization problem (#588, then #196)."
        )
    fragile = [
        f"{shot} {row['axis'].split('.')[-1]}={row['value']}"
        for shot, block in payload["shots"].items()
        for row in block["sweep"]
        if row["summary"].get("lost")
    ]
    if fragile:
        lines.append(
            "**The present seed is not comfortably inside the basin**: these seeds lose slices that "
            "the routine seed reconstructs — " + ", ".join(fragile[:6]) + "."
        )
    else:
        lines.append(
            "**The basin is wide in every direction tried**: no seed in the sweep loses a slice the "
            "routine seed reconstructs, so today's choice is not perched on an edge."
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--shots", default=None, help="comma-separated; default: the reference set's magnetics shots")
    parser.add_argument("--tables", default=None, help="EFIT table directory (default: the packaged one)")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.config import EFITScientificConfig, efit_parameter_grid
    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    efit = str(resolved["efit"])
    tables = args.tables or str(Path(data_path("efit")).resolve()) + "/"

    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(v) for v in args.shots.split(",")] if args.shots else sorted(products)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    base = EFITScientificConfig()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "tables": tables,
        "routine_seed": dict(ROUTINE_SEED),
        "axes": {name: list(values) for name, values in AXES.items()},
        "shots": {},
    }

    for shot in shots:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        print(f"{shot}: building constraints once ...", flush=True)
        constraints_dir = output / f"shot_{shot}" / "constraints"
        ods, times, window, baseline = prepare_shot(
            shot,
            Path(data_path(products[shot])),
            workdir=constraints_dir,
            tables=tables,
            tstep=args.tstep,
            average_window=args.average_window,
        )

        routine = run_seed(
            ods, shot=shot, times=times, workdir=output / f"shot_{shot}" / "routine",
            efit=efit, scientific=base, baseline=baseline,
        )
        reference_outcomes = {int(item["time_ms"]): outcome(item) for item in routine["slices"]}
        block: dict[str, Any] = {
            "window": {"start": float(window.start), "end": float(window.end), "slices": int(times.size)},
            "routine": summarize(routine["slices"]),
            "routine_slices": routine["slices"],
            "sweep": [],
        }
        print(
            f"  routine: {block['routine']['produced_an_equilibrium']}/{block['routine']['slices']} produced, "
            f"{block['routine']['accepted']} accepted, {block['routine']['collapsed']} collapsed",
            flush=True,
        )

        for axis, values in AXES.items():
            for value in values:
                if abs(value - ROUTINE_SEED[axis.split(".")[-1]]) < 1e-12:
                    continue  # the routine point is already measured
                scientific = efit_parameter_grid(base, {axis: [value]})[0]
                run = run_seed(
                    ods, shot=shot, times=times,
                    workdir=output / f"shot_{shot}" / f"{axis.split('.')[-1]}_{value}",
                    efit=efit, scientific=scientific, baseline=baseline,
                )
                summary = summarize(run["slices"], reference_outcomes)
                block["sweep"].append({"axis": axis, "value": value, "summary": summary})
                print(
                    f"  {axis.split('.')[-1]}={value}: produced {summary['produced_an_equilibrium']}, "
                    f"accepted {summary['accepted']}, recovered {len(summary['recovered'])}, "
                    f"lost {len(summary['lost'])}",
                    flush=True,
                )
        payload["shots"][str(shot)] = block

    payload["verdict"] = verdict_lines(payload)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(payload)
    (args.markdown or output / "seed_basin.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
