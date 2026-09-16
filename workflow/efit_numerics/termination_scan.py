"""What should a VEST EFIT fit stop on, and what does tightening it cost? (#171)

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_numerics/termination_scan.py --output /scratch/term \\
            --table test/data/efit_termination_scan.json

Six of the parameters that decide when a VEST fit stops are EFIT's own
defaults that VAFT never wrote, and one it does write is inert.  #171
established that from the source; this measures what those defaults cost.

What actually terminates the fit
--------------------------------

VEST runs ``iconvr = 2``, so the outer loop leaves through ``ichisq``
(``response_matrix.F90:2839``) only when all of

    nniter >= minite                                  (8, hard-coded)
    errorm <= errmin                                  (1e-2 by default)
    saisq  <= saicon                                  (80 by default)
    |saisq - saiold| <= 0.10 or saisq >= saiold       (chi-square has stalled)

hold at once, and it then **restores the previous iterate**.  ``ERROR`` never
enters: it reaches the solver only through the inner loop, which VEST runs
once (``nxiter = 1``).

The coupling this study exists to measure
-----------------------------------------

``chkerr.f90:28-34`` sets ``chisq_max = saicon`` and ``error_max = errmin``
at ``iconvr = 2``.  **The same two numbers are the stopping preconditions and
the acceptance thresholds.**  Lowering ``SAICON`` therefore makes stopping
harder and acceptance stricter at once, and the net effect on yield is not
predictable from either half -- which is exactly why it has to be measured
rather than argued.

There is a second-order effect that makes the sign genuinely ambiguous: the
criterion tests the *new* ``saisq`` and then writes back the *previous* one,
so a slice can stop on the criterion and still be rejected on ``#1`` with the
restored chi-square.  The re-baseline (#852) shows that happening -- every
slice on 39915 exits on ``iconvr=2`` and four of them still fail ``#1``.

One axis at a time.  Five crossed axes would be hundreds of EFIT runs for less
information than the marginals give first, and #588 established for the seed
study that a scan which varies two things at once attributes nothing.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time as _clock
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
SEED_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "seed_basin.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_termination_scan.json"

#: The routine configuration, named so a re-run centres on whatever the
#: defaults actually are rather than on what they were when this was written.
#: ``None`` means "write no key", which is how EFIT's own default applies.
ROUTINE = {
    "numerics.error_minimum": None,
    "numerics.chi_squared_target": None,
    "numerics.inner_iterations": None,
    "numerics.max_iterations": 100,
    "numerics.relaxation": 1.0,
}

#: One axis at a time.  The first value of each is the routine point and is
#: skipped, because it is already measured as the baseline.
AXES: dict[str, tuple[Any, ...]] = {
    # EFIT's default is 1e-2, two orders looser than the ERROR VEST writes and
    # never reaches.  The writer's own commented-out note says 1e-3 moves a
    # fit from under a minute to about three; this measures whether it buys
    # anything for that.
    "numerics.error_minimum": (None, 3.0e-3, 1.0e-3),
    # Both the stopping precondition and the acceptance threshold.
    "numerics.chi_squared_target": (None, 60.0, 40.0),
    # The only way to make ERROR mean anything: the inner equilibrium loop.
    "numerics.inner_iterations": (None, 3),
    # 100 is VAFT's; 25 is EFIT's default; 200 asks whether the cap binds.
    "numerics.max_iterations": (100, 25, 200),
    # Divides errorm, so it interacts with error_minimum rather than acting
    # alone -- reported as a secondary axis for that reason.
    "numerics.relaxation": (1.0, 0.7),
}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _apply(scientific, axis: str, value: Any):
    """One setting changed on a copy; everything else is the routine value."""
    group, field = axis.split(".", 1)
    block = getattr(scientific, group)
    return replace(scientific, **{group: replace(block, **{field: value})})


def _routine_config():
    from vaft.code.efit.config import EFITScientificConfig

    scientific = EFITScientificConfig()
    for axis, value in ROUTINE.items():
        scientific = _apply(scientific, axis, value)
    return scientific


def summarize(slices: Sequence[dict[str, Any]], seed_study) -> dict[str, Any]:
    """Outcomes, plus what the fit actually did to get there.

    The seed study's `outcome` is reused rather than restated: a second
    definition of "accepted" is how two studies of the same reference set stop
    being comparable.
    """
    counts = Counter(seed_study.outcome(item) for item in slices)
    produced = [item for item in slices if item["afile"]]
    stopped = sum(1 for item in slices if item.get("iconvr") == 2)
    exhausted = sum(1 for item in slices if item["exit_path"] == "iterations_exhausted")
    iterations = [item["iterations_n"] for item in slices if item.get("iterations_n")]
    chi = [item["afile"]["chisq"] for item in produced if np.isfinite(item["afile"]["chisq"])]
    gs = [item["gs_error"] for item in slices if item.get("gs_error") is not None]
    failures = Counter(
        failure["code"] for item in slices for failure in item.get("failures", ())
    )
    return {
        "slices": len(slices),
        "outcomes": dict(sorted(counts.items())),
        "produced_an_equilibrium": len(produced),
        "accepted": counts.get("accepted", 0),
        "collapsed": counts.get("collapsed", 0),
        "stopped_on_criterion": stopped,
        "hit_iteration_cap": exhausted,
        "iterations_median": float(np.median(iterations)) if iterations else None,
        "chi_squared_median": float(np.median(chi)) if chi else None,
        "gs_error_median": float(np.median(gs)) if gs else None,
        "acceptance_failures": dict(sorted(failures.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--shots", default=None, help="comma-separated; default: the reference set")
    parser.add_argument("--tables", default=None, help="EFIT table directory (default: the packaged one)")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument(
        "--axis", default=None, help="restrict to one axis, e.g. numerics.chi_squared_target"
    )
    parser.add_argument("--values", default=None, help="comma-separated values for --axis")
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    seed_study = _module(SEED_STUDY, "seed_basin")
    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    efit = resolved.get("efit")
    if efit is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2

    axes: dict[str, tuple[Any, ...]] = dict(AXES)
    if args.axis:
        if args.axis not in AXES and not args.values:
            print(f"unknown axis {args.axis}; give --values to sweep it", file=sys.stderr)
            return 2
        if args.axis not in ROUTINE:
            known = ", ".join(sorted(ROUTINE))
            print(f"--axis {args.axis} is not a termination setting; expected one of {known}",
                  file=sys.stderr)
            return 2
        chosen = (
            tuple(json.loads(f"[{args.values}]")) if args.values else AXES[args.axis]
        )
        axes = {args.axis: chosen}

    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(v) for v in args.shots.split(",")] if args.shots else sorted(products)
    tables = args.tables or str(Path(data_path("efit")).resolve()) + "/"

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    routine = _routine_config()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "tables": tables,
        "routine": {axis: ROUTINE[axis] for axis in sorted(ROUTINE)},
        "axes": {name: list(values) for name, values in axes.items()},
        "shots": {},
    }

    for shot in shots:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        workdir = output / f"shot_{shot}"
        constraints_ods, times, window, baseline = seed_study.prepare_shot(
            shot,
            Path(data_path(products[shot])),
            workdir=workdir / "constraints",
            tables=tables,
            tstep=args.tstep,
            average_window=args.average_window,
        )
        from omas import load_omas_json

        built = load_omas_json(
            str(workdir / "constraints" / f"{shot}_constraints.json"), consistency_check=False
        )

        cases: dict[str, Any] = {}
        base = seed_study.run_seed(
            built, shot=shot, times=times, workdir=workdir / "routine",
            efit=str(efit), scientific=routine, baseline=baseline,
        )
        cases["routine"] = {
            "settings": {axis: ROUTINE[axis] for axis in sorted(ROUTINE)},
            "seconds": base["seconds"],
            "summary": summarize(base["slices"], seed_study),
            "slices": base["slices"],
        }
        print(f"{shot} routine: {cases['routine']['summary']['outcomes']} "
              f"in {base['seconds']:.0f} s", flush=True)

        for axis, values in axes.items():
            for value in values:
                if value == ROUTINE.get(axis):
                    continue  # the routine point, already measured above
                scientific = _apply(routine, axis, value)
                name = f"{axis}={value}"
                run = seed_study.run_seed(
                    built, shot=shot, times=times,
                    workdir=workdir / name.replace(".", "_").replace("=", "-"),
                    efit=str(efit), scientific=scientific, baseline=baseline,
                )
                cases[name] = {
                    "settings": {**{a: ROUTINE[a] for a in sorted(ROUTINE)}, axis: value},
                    "seconds": run["seconds"],
                    "summary": summarize(run["slices"], seed_study),
                    "slices": run["slices"],
                }
                print(f"{shot} {name}: {cases[name]['summary']['outcomes']} "
                      f"in {run['seconds']:.0f} s", flush=True)

        payload["shots"][str(shot)] = {
            "window": [float(times[0]), float(times[-1])],
            "cases": cases,
        }

    destination = args.table.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
