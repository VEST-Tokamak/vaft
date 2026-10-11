#!/usr/bin/env python3
"""EFIT outcomes by operating regime over a FileDB (#1891). Read-only.

    python workflow/efit_cohorts/run_efit_cohorts.py --filedb-root /srv/vest.filedb \\
        --output ~/runs/cohorts-1891 [--shots shots.txt] [--source production]

For every shot with a magnetic EFIT run, every **attempted** slice is labelled
and joined with how it ended:

* regime -- from the run's constraints ODS (``omas/efit/magnetic/<shot>/work/
  constraints.json``), which holds every slice EFIT was asked for, so slices
  that wrote no g-file are in the cohort instead of only in the log;
* termination -- :func:`vaft.code.efit.parse_slices` of ``run_efit.out``;
* graded status -- the product manifest's ``slice_statuses``;
* era -- :func:`vaft.omas.vest_upstream.machine_era_for_shot`;
* eddy -- the eddy product's ``pf_passive.loop[*].current``, summed;
* topology -- ``limloc`` from the run's a-files (the product does not keep it).

Writes ``cohorts.csv`` (+ ``.meta.json`` with the thresholds), ``summary.json``
(cohort counts by several group-bys, representative cases, and the shots that
could not be read and why) beside it. Nothing is written outside ``--output``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omas import ODS

from vaft.code.efit import parse_slices
from vaft.data.aeqdsk import read_aeqdsk
from vaft.machine_mapping.equilibrium_regime import vest_equilibrium_regime_policy
from vaft.omas.vest_upstream import machine_era_for_shot
from vaft.validation.equilibrium_cohorts import (
    cohort_rows,
    join_regime_and_termination,
    representative_cases,
    summarize_by_cohort,
    write_cohort_table,
)

GROUP_BYS = (
    ("era",),
    ("phase",),
    ("current_bin",),
    ("rate_bin",),
    ("era", "phase"),
    ("era", "current_bin"),
    ("phase", "rate_bin"),
    ("topology",),
)


def _constraints_ods(path: Path) -> ODS:
    """Only what the classifier reads: the slice times and measured Ip."""
    document = json.loads(path.read_text(encoding="utf-8"))
    equilibrium = document["equilibrium"]
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.asarray(equilibrium["time"], dtype=float)
    for index, item in enumerate(equilibrium.get("time_slice") or ()):
        ip = ((item.get("constraints") or {}).get("ip") or {}).get("measured")
        if ip is not None:
            ods[f"equilibrium.time_slice.{index}.constraints.ip.measured"] = float(ip)
    return ods


def _passive_current(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        passive = json.load(handle).get("pf_passive") or {}
    time = np.asarray(passive.get("time") or [], dtype=float)
    loops = [np.asarray(loop.get("current") or [], dtype=float) for loop in passive.get("loop") or ()]
    loops = [values for values in loops if values.size == time.size]
    if time.size < 2 or not loops:
        return None
    return time, np.sum(loops, axis=0)


def _limloc(afiles: Path) -> dict[int, str]:
    out = {}
    for path in sorted(afiles.glob("a*")):
        try:
            record = read_aeqdsk(path)
        except Exception:  # noqa: BLE001 - one unreadable a-file must not drop the shot
            continue
        out[int(round(float(record.time_ms)))] = str(record.limloc)
    return out


def shot_outcomes(root: Path, shot: int, source: str):
    product = root / "omas" / "efit" / "magnetic" / str(shot)
    run = root / "efit" / "magnetic" / str(shot)
    constraints = product / "work" / "constraints.json"
    log = run / "run_efit.out"
    if not constraints.exists():
        raise FileNotFoundError("no constraints ODS")
    if not log.exists():
        raise FileNotFoundError("no run_efit.out")
    manifest_path = product / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return join_regime_and_termination(
        _constraints_ods(constraints),
        parse_slices(log.read_text(encoding="utf-8", errors="replace")),
        slice_statuses=manifest.get("slice_statuses") or (),
        shot=shot,
        source=source,
        era=machine_era_for_shot(shot).name,
        passive_current=_passive_current(root / "omas" / "eddy" / str(shot) / "output" / "eddy.json.gz"),
        limloc=_limloc(run / "afile"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filedb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=Path, help="one shot per line; default: every shot with an EFIT run")
    parser.add_argument("--source", default="production")
    args = parser.parse_args(argv)

    if args.shots:
        shots = sorted({int(line.split()[0]) for line in args.shots.read_text().splitlines() if line.strip()})
    else:
        shots = sorted(int(p.name) for p in (args.filedb_root / "efit" / "magnetic").iterdir() if p.name.isdigit())

    outcomes: list[Any] = []
    skipped: dict[str, list[int]] = {}
    unmatched = 0
    for shot in shots:
        try:
            join = shot_outcomes(args.filedb_root, shot, args.source)
        except Exception as error:  # noqa: BLE001 - recorded per reason, never silent
            skipped.setdefault(f"{type(error).__name__}: {error}"[:160], []).append(shot)
            continue
        outcomes.extend(join)
        unmatched += len(join.unmatched_log_times)

    policy = vest_equilibrium_regime_policy()
    args.output.mkdir(parents=True, exist_ok=True)
    write_cohort_table(cohort_rows(outcomes), args.output / "cohorts.csv", policy=policy)
    summary = {
        "shots_requested": len(shots),
        "shots_read": len({item.shot for item in outcomes}),
        "slices": len(outcomes),
        "unmatched_log_slices": unmatched,
        "skipped": {reason: {"count": len(items), "shots": items[:20]} for reason, items in sorted(skipped.items())},
        "thresholds": policy.thresholds(),
        "cohorts": {
            "+".join(group): summarize_by_cohort(outcomes, group_by=group)["cohorts"] for group in GROUP_BYS
        },
        "representative_cases": {
            "+".join(group): representative_cases(outcomes, group_by=group)
            for group in (("era", "phase"), ("current_bin",), ("rate_bin",))
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"{summary['shots_read']}/{len(shots)} shots, {len(outcomes)} slices -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
