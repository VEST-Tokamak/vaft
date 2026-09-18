"""Does the reconstruction agree with pressure it never used? (#386)

    PYTHONPATH=$PWD python \\
        workflow/efit_diamagnetic_weight/thomson_pressure_check.py \\
            --baseline /scratch/baseline --shot 39915

Runs no EFIT.  It reads reconstructions someone else produced and asks the one
question the magnetics cannot answer about themselves: **is the pressure really
missing, or is the fit simply not being told about it?**

#386 records a reconstructed pressure roughly ninety times below what force
balance requires, measured against the virial estimate -- but the virial
estimate is built from the same magnetics as the reconstruction, and #649
established that at VEST's aspect ratio its conditioning is bad enough that it
cannot decide an accept/reject question on its own.  Thomson scattering is
outside that loop entirely.

39915 is the only shot in the reference set that can be asked.  Its ten Thomson
samples run 0.308-0.317 s and its routine EFIT window is 0.306-0.331 s, so
every sample falls inside the reconstructed window
(``workflow/efit_reference_set/README.md``).  41524 and 41672 carry no packaged
kinetic data at all.

The asymmetry this test has, and does not hide
----------------------------------------------

Thomson measures electrons.  ``n_e k T_e`` is therefore a *lower bound* on the
total pressure, and the registry already encodes that one-sidedness
(``vaft/validation/registry.py``): electrons exceeding the reconstructed total
is a failure of the reconstruction, falling short of it is at most a warning,
because the ions are unmeasured.

That makes the test decisive in exactly one direction.  If the electron
pressure alone runs far above the reconstructed total, no ion population can
explain it and the fit is wrong.  If it runs below, this test says nothing --
the ions could be anywhere.  A ninety-fold shortfall is far outside the
asymmetry, so the answer is expected to be informative; that it *is* informative
still has to be checked rather than assumed.

The packaged ``g039915.00319`` cannot be used: at 0.319 s it is 2 ms from the
nearest Thomson sample, beyond the half-cadence tolerance
``vaft/validation/equilibrium.py`` applies.  The reconstructions this reads are
on a 1 ms grid that lands on the sample times exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_thomson_pressure.json"
KEY = "independent_validation.thomson_pressure"

#: Beyond this the electron pressure alone is so far above the reconstructed
#: total that no unmeasured ion population can close the gap.  ln(3) is a
#: threefold disagreement; the registry's own fail threshold is ln(2).
DECISIVE_LOG_RATIO = math.log(3.0)


def thomson_diagnostics(shot: int, data_root: Path | None = None):
    """The shot's Thomson channels as a diagnostics ODS, or ``None``."""
    from omas import ODS
    from vaft.data.resources import data_path
    from vaft.machine_mapping.thomson_scattering import thomson_scattering

    diagnostics = ODS(consistency_check=False)
    root = str(data_root) if data_root else str(data_path("legacy"))
    try:
        thomson_scattering(diagnostics, int(shot), data_root=root)
    except Exception as exc:  # no packaged file for this shot
        print(f"{shot}: no Thomson source ({exc})", file=sys.stderr)
        return None
    if not len(diagnostics.get("thomson_scattering.channel", [])):
        return None
    return diagnostics


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def check_gfile(path: Path, diagnostics: Any) -> dict[str, Any]:
    """One reconstruction against the Thomson samples nearest it in time."""
    from omas import ODS
    from vaft.data import read_geqdsk
    from vaft.validation.equilibrium import validate_independent

    geq = read_geqdsk(path)
    equilibrium = ODS(consistency_check=False)
    geq.to_omas(equilibrium)
    result = validate_independent(equilibrium, time_slice=0, diagnostics=diagnostics)[
        "thomson_pressure"
    ]
    pressure = np.asarray(geq.mapping["PRES"], dtype=float)
    return {
        "path": path.name,
        "time_s": _finite(np.asarray(equilibrium["equilibrium.time"]).reshape(-1)[0]),
        "status": str(result.get("status")),
        "reason": result.get("reason"),
        "points": result.get("points"),
        "log_ratio": _finite(result.get("log_ratio")),
        "sum_ratio": _finite(result.get("sum_ratio")),
        "correlation": _finite(result.get("correlation")),
        "time_offset_s": _finite(result.get("time_offset")),
        "p_axis": _finite(pressure[0]) if pressure.size else None,
    }


def _time_ms(path: Path) -> int | None:
    from vaft.code.efit.slice_name import file_name_microseconds

    microseconds = file_name_microseconds(path)
    return None if microseconds is None else microseconds // 1000


def run_directories(root: Path, shot: int) -> list[Path]:
    """Every EFIT run directory under ``root``, and nothing that only looks like one.

    A run directory holds the k-files EFIT was given as well as the g-files it
    produced.  Without that second test the packaged EFIT table directory
    qualifies -- it ships ``g039915.00317`` and ``g039915.00319`` as reference
    products, and the workflow copies it next to the runs -- so a study would
    silently report the stored 2023 reconstruction as one of its own results.
    """
    return sorted(
        {
            path.parent
            for path in Path(root).rglob(f"g0{shot}.*")
            if path.is_file() and (path.parent / "kfile").is_dir()
        }
    )


def check_directory(workdir: Path, shot: int, diagnostics: Any) -> list[dict[str, Any]]:
    """Every g-file in one run directory, in time order."""
    return [
        check_gfile(path, diagnostics)
        for path in sorted(workdir.glob(f"g0{shot}.*"), key=lambda p: _time_ms(p) or 0)
    ]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the comparable slices say, and how many there were.

    Slices whose nearest Thomson sample is out of tolerance are counted, never
    averaged into the answer: a missing comparison is not agreement.
    """
    compared = [row for row in rows if row.get("log_ratio") is not None]
    ratios = np.asarray([row["log_ratio"] for row in compared], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    if not ratios.size:
        return {
            "reconstructions": len(rows),
            "compared": 0,
            "statuses": statuses,
            "median_log_ratio": None,
            "median_sum_ratio": None,
            "decisive_slices": 0,
        }
    median = float(np.median(ratios))
    return {
        "reconstructions": len(rows),
        "compared": int(ratios.size),
        "statuses": statuses,
        "median_log_ratio": median,
        "median_sum_ratio": math.exp(median),
        "min_sum_ratio": math.exp(float(ratios.min())),
        "max_sum_ratio": math.exp(float(ratios.max())),
        # Only an excess is decisive: a shortfall could be unmeasured ions.
        "decisive_slices": int(np.count_nonzero(ratios >= DECISIVE_LOG_RATIO)),
    }


def verdict(summary: Mapping[str, Any]) -> str:
    if not summary["compared"]:
        return (
            "no reconstruction fell within tolerance of a Thomson sample, so "
            "this shot's kinetic arm says nothing about the pressure here"
        )
    ratio = summary["median_sum_ratio"]
    decisive = summary["decisive_slices"]
    if decisive:
        return (
            f"the electron pressure alone is {ratio:.3g}x the reconstructed "
            f"total on the median slice, and above the decisive threshold on "
            f"{decisive} of {summary['compared']}: no unmeasured ion "
            "population can close that gap, so the pressure is real and the "
            "fit is missing it -- this is not a data problem"
        )
    if ratio >= 1.0:
        return (
            f"the electron pressure is {ratio:.3g}x the reconstructed total, "
            "above it but within what measurement scatter could explain; "
            "suggestive, not decisive"
        )
    return (
        f"the electron pressure is {ratio:.3g}x the reconstructed total, i.e. "
        "below it. Electrons alone falling short of the total says nothing "
        "either way, because the ions are unmeasured"
    )


def compare_ladder(
    workdirs: Mapping[str, Path], *, shot: int, output: Path | None = None
) -> dict[str, Any]:
    """The same comparison at every rung of the weight ladder.

    The question is not only whether the reconstruction disagrees with Thomson,
    but whether making the diamagnetic constraint reachable moves it *toward*
    the measurement it never used.  A rung that raises the weight and leaves
    this ratio where it was has not improved the reconstruction, whatever it
    did to the residual.
    """
    diagnostics = thomson_diagnostics(shot)
    if diagnostics is None:
        return {"available": False, "reason": f"no packaged Thomson data for {shot}"}
    rungs: dict[str, Any] = {}
    for name, workdir in workdirs.items():
        rows = check_directory(Path(workdir), shot, diagnostics)
        rungs[name] = {"summary": summarize(rows), "slices": rows}
    ordered = [
        (name, item["summary"]["median_log_ratio"])
        for name, item in rungs.items()
        if item["summary"]["median_log_ratio"] is not None
    ]
    best = min(ordered, key=lambda pair: abs(pair[1]), default=None)
    payload: dict[str, Any] = {
        "available": True,
        "shot": shot,
        "decisive_log_ratio": DECISIVE_LOG_RATIO,
        "rungs": rungs,
        "closest_to_thomson": None if best is None else best[0],
    }
    silent = [
        name
        for name, item in rungs.items()
        if not item["summary"]["compared"]
    ]
    payload["rungs_with_no_comparable_slice"] = silent
    if ordered:
        first = ordered[0]
        payload["verdict"] = (
            verdict(rungs[first[0]]["summary"])
            + ". Across the ladder the closest agreement is at "
            f"{best[0]} (median ratio "
            f"{rungs[best[0]]['summary']['median_sum_ratio']:.3g}x)"
        )
        if silent:
            # The slices a high weight keeps need not be the ones Thomson
            # covers, so this arm can fall silent exactly where it is needed.
            payload["verdict"] += (
                f". {len(silent)} rung(s) kept no slice inside the Thomson "
                f"window and say nothing: {', '.join(sorted(silent))}"
            )
    else:
        payload["verdict"] = (
            "no rung kept a slice inside the Thomson window, so the kinetic "
            "arm says nothing about this ladder"
        )
    if output is not None:
        destination = Path(output) / "thomson_pressure.json"
        destination.write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return payload


def markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Reconstructed pressure against Thomson (#386)",
        "",
        f"Run at {payload['run_at']}.",
        "",
        "`sum_ratio` is the electron pressure over the reconstructed total at "
        "the channel positions. Electrons are a lower bound, so only an "
        "*excess* is decisive.",
        "",
        "| source | compared | median ratio | min | max | decisive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    cell = lambda value: "-" if value is None else f"{value:.3g}"
    ordered = sorted(payload["sources"].items())
    for name, item in ordered:
        summary = item["summary"]
        lines.append(
            f"| {name} | {summary['compared']} | "
            f"{cell(summary.get('median_sum_ratio'))} | "
            f"{cell(summary.get('min_sum_ratio'))} | "
            f"{cell(summary.get('max_sum_ratio'))} | "
            f"{summary['decisive_slices']} |"
        )
    lines.append("")
    for name, item in ordered:
        lines += [f"**{name}.** {item['verdict']}.", ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="a directory of EFIT output; shot_<n>/ subdirectories are searched",
    )
    parser.add_argument("--shot", type=int, default=39915)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--thomson-root",
        type=Path,
        default=None,
        help="where the NeTe_Shot<n>*.mat files live (default: the packaged ones)",
    )
    args = parser.parse_args(argv)

    diagnostics = thomson_diagnostics(args.shot, args.thomson_root)
    if diagnostics is None:
        print(f"no Thomson data for {args.shot}", file=sys.stderr)
        return 2

    root = args.baseline.expanduser()
    candidates = run_directories(root, args.shot)
    if not candidates:
        print(
            f"no EFIT run directory under {root}: looked for g0{args.shot}.* "
            "beside a kfile/ directory",
            file=sys.stderr,
        )
        return 2

    sources: dict[str, Any] = {}
    for workdir in candidates:
        rows = check_directory(workdir, args.shot, diagnostics)
        name = str(workdir.relative_to(root)) or workdir.name
        summary = summarize(rows)
        sources[name] = {
            "path": str(workdir),
            "summary": summary,
            "verdict": verdict(summary),
            "slices": rows,
        }
        print(f"{name}: {summary['compared']} compared; {sources[name]['verdict']}")

    payload = {
        "schema_version": SCHEMA,
        "issue": 386,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shot": args.shot,
        "baseline": str(root),
        "decisive_log_ratio": DECISIVE_LOG_RATIO,
        "sources": sources,
    }
    destination = args.table.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    report = markdown(payload)
    if args.markdown:
        args.markdown.expanduser().write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
