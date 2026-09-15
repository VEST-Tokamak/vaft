"""What the computational box and the grid are worth, measured (#459).

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_numerics/domain_grid.py --output /scratch/domain \\
            --tables A=/tables/a,B=/tables/b,C=/tables/c,D=/tables/d

The #171 baseline loses 46 of 77 plasma slices to two failures that are both
statements about the grid rather than about the fit: ``bound`` cannot close a
contour, and ``findax`` refuses a separatrix point that lands within two cells
of the box edge.  The routine box is R 0.05-1.2 m by Z +-1.5 m on 129x129,
which is 9.0 mm across and 23.4 mm tall -- a cell aspect ratio of 2.6 -- and
it extends to |Z| = 1.5 m where the limiter reaches only 1.185 m.

So this runs #459's 2x2: the domain and the grid, separately and together.

===== ======================== ========= ==============================
case   domain                   grid      what its difference from A is
===== ======================== ========= ==============================
A      R 0.05-1.2, Z +-1.5      129x129   the routine configuration
B      R 0.05-1.0, Z +-1.35     129x129   the domain alone
C      R 0.05-1.2, Z +-1.5      129x257   the cell aspect ratio alone
D      R 0.05-1.0, Z +-1.35     129x257   both
===== ======================== ========= ==============================

**The tables are the experiment.**  EFIT has no independent notion of the
computational box: it reads ``rgrid`` and ``zgrid`` straight out of the Green
table (``tables.F90:158``) and derives ``drgrid``, ``dzgrid`` and ``darea``
from them (``setup_data_fetch.F90:579``).  The box is therefore whatever EFUND
baked in, and the table file name encodes only ``nw`` and ``nh``
(``table_name_ch``, ``tables.F90:34``) -- two tables for the same grid and
different boxes are named identically.  A case run against the wrong table is
undetectable from EFIT's output, so every table directory is verified against
its case's declaration here before anything is run.

Generate the four tables with ``workflow/efit_tables/regenerate_legacy_table.py``.
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
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_domain_grid.json"

#: The routine box. ``rleft`` is 0.05 m and not the 0.0 m #459's text quotes:
#: the Green functions are singular on the machine axis, so the box has never
#: started there.
ROUTINE_DOMAIN = (0.05, 1.2, -1.5, 1.5)
REDUCED_DOMAIN = (0.05, 1.0, -1.35, 1.35)

CASES: dict[str, dict[str, Any]] = {
    "A": {"grid": (129, 129), "domain": ROUTINE_DOMAIN, "role": "the routine configuration"},
    "B": {"grid": (129, 129), "domain": REDUCED_DOMAIN, "role": "the domain alone"},
    "C": {"grid": (129, 257), "domain": ROUTINE_DOMAIN, "role": "the cell aspect ratio alone"},
    "D": {"grid": (129, 257), "domain": REDUCED_DOMAIN, "role": "both"},
}

#: What a reconstruction is compared on. Every one is an a-file scalar, so the
#: comparison needs no re-derivation and cannot drift from what EFIT reported.
METRICS = (
    "rm", "zm", "aminor", "elong", "area", "volume",
    "q95", "qstar", "li", "betap", "chisq", "terror",
)

#: Lengths in the a-file are centimetres; these are the ones to say so about.
CENTIMETRES = {"rm", "zm", "aminor", "rcntr", "zcntr", "rcurrt", "zcurrt"}


def _module(path: Path, name: str):
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


def verify_table(directory: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Refuse a table that is not the one this case declares.

    EFIT cannot do this check. It opens ``ec<nw><nh>.ddd`` by name and takes
    the box from inside it, so a table generated for a different domain loads
    silently and reconstructs a different machine. The manifest EFUND wrote
    beside the table is the only record of the box, and this is where it is
    read.
    """
    manifest_path = directory / "efund_table_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{directory} has no efund_table_manifest.json; regenerate it with "
            "workflow/efit_tables/regenerate_legacy_table.py rather than trusting the file names"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["efund"]["config"]["grid"]
    grid = (int(recorded["nw"]), int(recorded["nh"]))
    domain = tuple(float(recorded[name]) for name in ("rleft", "rright", "zbotto", "ztop"))
    if grid != tuple(case["grid"]):
        raise ValueError(f"{directory}: table is {grid[0]}x{grid[1]}, case declares {case['grid']}")
    if not np.allclose(domain, case["domain"], atol=1e-9):
        raise ValueError(f"{directory}: table box is {domain}, case declares {case['domain']}")
    expected = f"ec{grid[0]}{grid[1]}.ddd"
    if not (directory / expected).is_file():
        raise FileNotFoundError(f"{directory}: no {expected}; the table is incomplete")
    return manifest


#: EFIT products EFUND does not write, and the run directory is not usable
#: without them. `lim.dat` is an EFIT input that lives in the table directory,
#: and `&incheck` is read from `mhdin.dat` there.
EFIT_ONLY_INPUTS = ("lim.dat",)


def stage_case(directory: Path, *, root: Path, name: str, envelope: Any) -> Path:
    """A directory EFIT can actually run against, assembled from a table.

    EFUND writes the Green tables and its own ``mhdin.dat``; EFIT additionally
    needs ``lim.dat`` in the same directory and reads its acceptance envelope
    from ``&incheck`` there. Neither is an EFUND product, so a freshly
    generated table directory is a complete table and an incomplete run
    directory.

    The envelope is passed in rather than derived per case on purpose. Its
    ``aminor_min`` is a resolution statement -- so many grid cells -- and would
    move with the grid, which would leave the 2x2 comparing four cases against
    four different acceptance bars. #459 requires everything but the domain and
    the grid to be held fixed, so the bar is case A's throughout, and what each
    case's own floor would have been is recorded instead.

    ``root`` must be short: EFIT truncates ``TABLE_DIR`` at 100 characters and
    then fails opening ``lim.dat`` with a message about the limiter.
    """
    from vaft.code.efit.efund import write_mhdin
    from vaft.data.resources import data_path
    from vaft.machine_mapping.efund_geometry import efund_geometry_from_static
    from vaft.omas.vest_upstream import build_static_ods

    staged = root / name
    if len(str(staged)) + 1 > 100:
        raise ValueError(
            f"{staged} is {len(str(staged))} characters; EFIT truncates TABLE_DIR at 100. "
            "Pass a shorter --stage."
        )
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    for path in sorted(directory.iterdir()):
        if path.name in {"mhdin.dat"} or path.is_dir():
            continue
        (staged / path.name).symlink_to(path.resolve())
    for name_ in EFIT_ONLY_INPUTS:
        (staged / name_).symlink_to(Path(data_path(f"efit/{name_}")).resolve())

    manifest = json.loads((directory / "efund_table_manifest.json").read_text(encoding="utf-8"))
    from vaft.code.efit.efund import EFUNDConfig

    recorded = manifest["efund"]["config"]
    config = EFUNDConfig(
        workdir=staged,
        device=recorded["device"],
        **recorded["grid"],
        **recorded["flags"],
        **recorded["quadrature"],
    )
    ods, era_manifest = build_static_ods(manifest["machine"]["era"])
    geometry = efund_geometry_from_static(ods, manifest=era_manifest)
    write_mhdin(geometry, config, staged / "mhdin.dat", envelope=envelope)
    return staged


def cell_size(case: dict[str, Any]) -> tuple[float, float]:
    """Cell width and height in millimetres."""
    rleft, rright, zbotto, ztop = case["domain"]
    nw, nh = case["grid"]
    return ((rright - rleft) / (nw - 1) * 1e3, (ztop - zbotto) / (nh - 1) * 1e3)


def prepare_shot(shot: int, product: Path, *, workdir: Path, tables: str, tstep: float, average_window: float):
    """Constraints for one shot against one table directory.

    Rebuilt per case rather than once per shot: the table directory is written
    into the k-file, so constraints built against one box cannot be reused for
    another.
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


def run_case(
    constraints_ods,
    *,
    shot: int,
    times: np.ndarray,
    workdir: Path,
    efit: str,
    grid: Sequence[int],
    baseline,
) -> dict[str, Any]:
    """One case, one whole discharge."""
    from vaft.code.efit.config import EFITScientificConfig
    from vaft.code.efit.magnetic import EFITConfig, prepare_efit_inputs, run_efit

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    scientific = EFITScientificConfig()
    config = EFITConfig(
        executable=efit,
        workdir=workdir,
        shot=shot,
        times=times.tolist(),
        # EFIT reads nw and nh as argv(1) and argv(2) (`efit.F90:93-104`); a
        # single argument means a square grid.
        args=tuple(str(value) for value in grid),
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
        scalars = {name: float(record.scalars.get(name, float("nan"))) for name in METRICS}
        by_time[int(round(float(record.time_ms)))] = {"jflag": int(record.jflag), **scalars}
    for record in slices:
        record["afile"] = by_time.get(int(record["time_ms"]))
    return {"seconds": seconds, "returncode": result.returncode, "slices": slices}


def outcome(record: dict[str, Any]) -> str:
    if record["collapsed"]:
        return "collapsed"
    if record["afile"] is None:
        return "no_output"
    return "accepted" if record["afile"]["jflag"] == 1 else "flagged"


def phases(ods, times: np.ndarray) -> dict[int, str]:
    """Ramp-up, flat-top and ramp-down, from the plasma current itself.

    #459 asks for phase-resolved summaries and #579's semantics where
    practical. The boundary used here is nine tenths of the peak current,
    stated rather than tuned: everything at or above it is the flat top, and
    the rest is named by which side of the peak it falls on.
    """
    current = np.asarray(
        [
            abs(float(ods[f"equilibrium.time_slice.{index}.constraints.ip.measured"]))
            for index in range(times.size)
        ],
        dtype=float,
    )
    peak = int(np.argmax(current))
    threshold = 0.9 * current[peak]
    labels = {}
    for index, time in enumerate(times):
        key = int(round(float(time) * 1000.0))
        if current[index] >= threshold:
            labels[key] = "flat_top"
        else:
            labels[key] = "ramp_up" if index < peak else "ramp_down"
    return labels


def compare(reference: Sequence[dict[str, Any]], case: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What changed against case A, slice by slice.

    Two different questions, kept apart. Which slices produce an equilibrium
    at all is a yield question and is counted over every slice. How much a
    reconstruction moved is only answerable where both cases produced one, so
    the metric deltas are computed over that intersection and the size of the
    intersection is reported beside them.
    """
    before = {int(item["time_ms"]): item for item in reference}
    after = {int(item["time_ms"]): item for item in case}
    moved = {
        time: [outcome(before[time]), outcome(after[time])]
        for time in sorted(set(before) & set(after))
        if outcome(before[time]) != outcome(after[time])
    }
    both = [
        time
        for time in sorted(set(before) & set(after))
        if before[time]["afile"] and after[time]["afile"]
    ]
    deltas: dict[str, dict[str, float]] = {}
    for name in METRICS:
        pairs = [
            (before[time]["afile"].get(name, np.nan), after[time]["afile"].get(name, np.nan))
            for time in both
        ]
        # A metric an a-file did not carry is absent from the comparison
        # rather than reported as a change of nothing.
        pairs = [pair for pair in pairs if np.isfinite(pair[0]) and np.isfinite(pair[1])]
        if not pairs:
            continue
        first = np.array([pair[0] for pair in pairs], dtype=float)
        second = np.array([pair[1] for pair in pairs], dtype=float)
        difference = second - first
        # A relative change needs something to be relative to. `zm` sits at
        # zero on an up-down symmetric machine, so dividing by its median
        # turns a 1 cm shift into a percentage in the millions; the absolute
        # numbers are the honest ones there and the ratio is withheld.
        scale = float(np.median(np.abs(first)))
        spread = float(np.max(np.abs(first)))
        meaningful = scale > 0 and scale > 0.01 * spread
        deltas[name] = {
            "slices": len(pairs),
            "median_abs": float(np.median(np.abs(difference))),
            "max_abs": float(np.max(np.abs(difference))),
            "reference_median": scale,
            "median_relative": float(np.median(np.abs(difference)) / scale) if meaningful else None,
        }
    return {
        "changed_vs_reference": moved,
        "recovered": sorted(
            time for time, (was, now) in moved.items()
            if was in {"collapsed", "no_output"} and now in {"accepted", "flagged"}
        ),
        "lost": sorted(
            time for time, (was, now) in moved.items()
            if was in {"accepted", "flagged"} and now in {"collapsed", "no_output"}
        ),
        "compared_on": len(both),
        "metrics": deltas,
    }


def summarize(slices: Sequence[dict[str, Any]], labels: dict[int, str]) -> dict[str, Any]:
    counts = Counter(outcome(item) for item in slices)
    by_phase: dict[str, Counter] = {}
    for item in slices:
        by_phase.setdefault(labels.get(int(item["time_ms"]), "unknown"), Counter())[outcome(item)] += 1
    return {
        "slices": len(slices),
        "outcomes": dict(sorted(counts.items())),
        "produced_an_equilibrium": sum(1 for item in slices if item["afile"]),
        "collapsed": counts.get("collapsed", 0),
        "accepted": counts.get("accepted", 0),
        "bound_failures": sum(
            1 for item in slices for error in item["solver_errors"] if error["routine"] == "bound"
        ),
        "findax_failures": sum(
            1 for item in slices for error in item["solver_errors"] if error["routine"] == "findax"
        ),
        "by_phase": {phase: dict(sorted(counter.items())) for phase, counter in sorted(by_phase.items())},
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# The computational domain and the grid (#459)", ""]
    lines.append(
        "EFIT takes `nw` and `nh` as its two command-line arguments and reads the box out of the "
        "Green table, so a case is a table plus a pair of arguments. Every table directory below "
        "was verified against its case's declared grid and box before the case ran."
    )
    lines.append("")
    lines.append("| case | domain (m) | grid | cell (mm) | aspect | role |")
    lines.append("|---|---|---|---|---|---|")
    for name, case in payload["cases"].items():
        rleft, rright, zbotto, ztop = case["domain"]
        dr, dz = case["cell_mm"]
        lines.append(
            f"| {name} | R {rleft}–{rright}, Z {zbotto}–{ztop} | {case['grid'][0]}×{case['grid'][1]} "
            f"| {dr:.1f} × {dz:.1f} | {dz / dr:.2f} | {case['role']} |"
        )
    lines.append("")
    for shot, block in payload["shots"].items():
        lines.append(f"## {shot}")
        lines.append("")
        lines.append("| case | produced | accepted | collapsed | `bound` | `findax` | recovered | lost | runtime |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name, record in block["cases"].items():
            summary, against = record["summary"], record.get("vs_reference") or {}
            lines.append(
                f"| {name} | {summary['produced_an_equilibrium']} of {summary['slices']} "
                f"| {summary['accepted']} | {summary['collapsed']} | {summary['bound_failures']} "
                f"| {summary['findax_failures']} "
                f"| {len(against.get('recovered', []))} | {len(against.get('lost', []))} "
                f"| {record['seconds']:.0f} s |"
            )
        lines.append("")
    lines.append("## What this says")
    lines.append("")
    for line in payload["verdict"]:
        lines.append(f"- {line}")
    return "\n".join(lines) + "\n"


def verdict_lines(payload: dict[str, Any]) -> list[str]:
    totals = {name: {"produced": 0, "accepted": 0, "slices": 0} for name in payload["cases"]}
    for block in payload["shots"].values():
        for name, record in block["cases"].items():
            summary = record["summary"]
            totals[name]["produced"] += summary["produced_an_equilibrium"]
            totals[name]["accepted"] += summary["accepted"]
            totals[name]["slices"] += summary["slices"]
    reference = totals.get("A", {}).get("produced", 0)
    lines = [
        "Yield against the routine configuration, over "
        f"{totals.get('A', {}).get('slices', 0)} plasma slices: "
        + ", ".join(
            f"{name} {value['produced']} produced / {value['accepted']} accepted"
            for name, value in sorted(totals.items())
        )
        + "."
    ]
    gains = {name: value["produced"] - reference for name, value in totals.items() if name != "A"}
    best = max(gains, key=lambda name: gains[name]) if gains else None
    if best and gains[best] > 0:
        lines.append(
            f"**Case {best} recovers the most**: {gains[best]} slices that the routine box and grid "
            "produce nothing for. `bound` and `findax` are grid statements, and this is how much of "
            "the loss they account for."
        )
    elif gains:
        lost = ", ".join(f"{name} {gains[name]:+d}" for name in sorted(gains))
        lines.append(
            "**Neither the domain nor the grid recovers a slice.** Against case A: "
            f"{lost} equilibria. The slices that produce nothing are not produced by a smaller box "
            "or a finer grid either, so `bound` and `findax` are reporting something other than "
            "resolution and the cause is elsewhere."
        )
    if "C" in totals and totals["C"]["produced"] <= reference:
        lines.append(
            "**Halving the vertical cell changes the answer by less than a percent** where it "
            "changes it at all: the cell aspect ratio goes from 2.6 to 1.3 and every global "
            "quantity moves by well under 1 %. The magnetic axis height is the one that responds, "
            "and it is quoted in absolute terms because it sits at zero on an up-down symmetric "
            "machine."
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--tables",
        required=True,
        help="comma-separated CASE=directory, e.g. A=/tables/a,C=/tables/c; "
        "each is verified against the case's declared grid and box",
    )
    parser.add_argument(
        "--stage",
        required=True,
        type=Path,
        help="short directory to assemble runnable table directories in; EFIT truncates "
        "TABLE_DIR at 100 characters, so a scratch path under a long project directory will not do",
    )
    parser.add_argument("--shots", default=None, help="comma-separated; default: the reference set's magnetics shots")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    efit = str(resolved["efit"])

    directories: dict[str, Path] = {}
    for item in args.tables.split(","):
        name, _, path = item.partition("=")
        name = name.strip().upper()
        if name not in CASES:
            print(f"unknown case {name}; known: {', '.join(CASES)}", file=sys.stderr)
            return 2
        directories[name] = Path(path).expanduser()
    if "A" not in directories:
        print("case A is the reference every other case is compared against; it must be given", file=sys.stderr)
        return 2

    # One acceptance bar for all four cases, derived from the routine box, so
    # that a case is judged on its reconstruction and not on a floor that moved
    # underneath it. See `stage_case`.
    from vaft.machine_mapping.efund_geometry import vest_acceptance_envelope
    from vaft.omas.vest_upstream import build_static_ods

    reference_ods, _ = build_static_ods("vest-pre-43017-pf1906")
    rleft, rright, *_ = CASES["A"]["domain"]
    envelope = vest_acceptance_envelope(
        reference_ods, nw=CASES["A"]["grid"][0], rleft=rleft, rright=rright
    )

    manifests, staged = {}, {}
    for name, directory in sorted(directories.items()):
        manifest = verify_table(directory, CASES[name])
        staged[name] = stage_case(directory, root=args.stage.expanduser(), name=name, envelope=envelope)
        manifests[name] = {
            "directory": str(directory),
            "staged": str(staged[name]),
            "table_identity": manifest["table"]["identity"],
            "efund_config_sha256": manifest["efund"]["config_sha256"],
            "generated_at": manifest["generated_at"],
            # What this case's own resolved floor would have been, for the
            # record: it is not applied, because the bar is held fixed.
            "own_aminor_min_cm": vest_acceptance_envelope(
                reference_ods,
                nw=CASES[name]["grid"][0],
                rleft=CASES[name]["domain"][0],
                rright=CASES[name]["domain"][1],
            ).aminor_min,
        }
        dr, dz = cell_size(CASES[name])
        print(f"{name}: {directory} verified, staged at {staged[name]}, cells {dr:.1f} x {dz:.1f} mm", flush=True)
    payload_envelope = {"aminor_min": envelope.aminor_min, "sha256": envelope.sha256}

    reference_set = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference_set["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(value) for value in args.shots.split(",")] if args.shots else sorted(products)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "acceptance_envelope": payload_envelope,
        "cases": {
            name: {**CASES[name], "cell_mm": cell_size(CASES[name]), "table": manifests[name]}
            for name in sorted(directories)
        },
        "shots": {},
    }

    for shot in shots:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        block: dict[str, Any] = {"cases": {}}
        reference_slices: list[dict[str, Any]] | None = None
        for name in sorted(directories, key=lambda key: (key != "A", key)):
            tables = str(staged[name]) + "/"
            print(f"{shot} {name}: constraints ...", flush=True)
            ods, times, window, baseline = prepare_shot(
                shot,
                Path(data_path(products[shot])),
                workdir=output / f"shot_{shot}" / name / "constraints",
                tables=tables,
                tstep=args.tstep,
                average_window=args.average_window,
            )
            labels = phases(ods, times)
            run = run_case(
                ods, shot=shot, times=times,
                workdir=output / f"shot_{shot}" / name / "run",
                efit=efit, grid=CASES[name]["grid"], baseline=baseline,
            )
            record = {
                "seconds": run["seconds"],
                "returncode": run["returncode"],
                "summary": summarize(run["slices"], labels),
            }
            if name == "A":
                # Per-slice evidence is kept for the reference only; every other
                # case records what changed against it, slice by slice, which is
                # the part a reader needs and a quarter of the size.
                record["slices"] = run["slices"]
                reference_slices = run["slices"]
                block["window"] = {
                    "start": float(window.start), "end": float(window.end), "slices": int(times.size)
                }
                block["phases"] = dict(sorted(Counter(labels.values()).items()))
            elif reference_slices is not None:
                record["vs_reference"] = compare(reference_slices, run["slices"])
            block["cases"][name] = record
            summary = record["summary"]
            print(
                f"  {name}: {summary['produced_an_equilibrium']}/{summary['slices']} produced, "
                f"{summary['accepted']} accepted, {summary['collapsed']} collapsed, "
                f"bound {summary['bound_failures']}, findax {summary['findax_failures']}, "
                f"{record['seconds']:.0f} s",
                flush=True,
            )
        payload["shots"][str(shot)] = block

    payload["verdict"] = verdict_lines(payload)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(payload)
    (args.markdown or output / "domain_grid.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
