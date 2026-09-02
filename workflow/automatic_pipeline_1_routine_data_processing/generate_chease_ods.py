#!/usr/bin/env python3
"""Generate a CHEASE OMAS ODS from refined CHEASE g-files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from omas import ODS, save_omas_json

from vaft.data.eqdsk import read_geqdsk
from vaft.omas.vest_upstream import write_manifest


LOGGER = logging.getLogger("vaft.generate_chease_ods")


def _minimal_chease_ods(shot: int, run: int, status: str, records_summary: list[dict[str, Any]]) -> ODS:
    ods = ODS()
    ods["dataset_description.data_entry.machine"] = "VEST"
    ods["dataset_description.data_entry.pulse"] = int(shot)
    ods["dataset_description.data_entry.run"] = int(run)
    ods["equilibrium.ids_properties.comment"] = f"CHEASE output unavailable: {status}"
    ods["equilibrium.code.name"] = "chease"
    # `code.name` alone is not exclusive to this stage -- vaft.data.vfit also
    # sets it -- and the vaft.plot registry's availability check only tests
    # path presence, not value; `code.library.0.name` is written only here,
    # so it is what the chease_overview_* renderers gate their availability on.
    ods["equilibrium.code.library.0.name"] = "chease"
    ods["equilibrium.code.parameters"] = json.dumps(
        {"comparison_metrics": {}, "records_summary": records_summary}
    )
    return ods


def _read_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    return tuple(Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_runs_summary(path: Path) -> dict[str, Any]:
    """The ``chease_runs.json`` written by ``run_chease_refinement.py``.

    Optional and tolerant of absence: a shot with no gfile manifest at all
    (upstream skip) never produced one.
    """
    if not path or not str(path).strip() or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read CHEASE runs summary %s: %s", path, exc)
        return {}


def _records_summary(runs_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-input-gfile status, including slices that were never refined."""
    return [
        {"input": record.get("input", ""), "status": record.get("status", "unknown")}
        for record in runs_summary.get("records", [])
    ]


def _comparison_metrics_by_time_index(
    gfiles: tuple[Path, ...], runs_summary: dict[str, Any]
) -> dict[str, Any]:
    """Match each refined gfile to its ``chease_runs.json`` comparison block.

    ``run_chease_refinement.py`` stages the refined gfile under
    ``record["staged"]``; that path is what ``--refined-gfile-manifest`` lists,
    one line per time slice, in the same order used here for ``time_index``.
    """
    staged_records = {
        record.get("staged", ""): record
        for record in runs_summary.get("records", [])
        if record.get("staged")
    }
    metrics: dict[str, Any] = {}
    for time_index, gfile in enumerate(gfiles):
        record = staged_records.get(str(gfile))
        if record is not None and record.get("comparison"):
            metrics[str(time_index)] = record["comparison"]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--refined-gfile-manifest", required=True, type=Path, help="Input CHEASE refined gfile manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Input CHEASE status file.")
    parser.add_argument("--output", required=True, type=Path, help="Output CHEASE ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    parser.add_argument(
        "--runs-summary",
        default="",
        help=(
            "chease_runs.json written by run_chease_refinement.py, carrying "
            "per-gfile comparison_metrics; validated by the chease FileDB stage."
        ),
    )
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    status_text = args.status.read_text(encoding="utf-8").strip() if args.status.exists() else "unknown"
    gfiles = _read_manifest(args.refined_gfile_manifest)
    runs_summary = _read_runs_summary(Path(args.runs_summary)) if args.runs_summary else {}
    records_summary = _records_summary(runs_summary)

    ods = None
    parse_errors = []
    for time_index, gfile in enumerate(gfiles):
        try:
            ods = read_geqdsk(gfile).to_omas(ods=ods, time_index=time_index)
        except Exception as exc:
            parse_errors.append(f"{gfile}: {exc}")
            LOGGER.warning("Could not parse CHEASE gfile %s: %s", gfile, exc)

    if ods is None:
        ods = _minimal_chease_ods(args.shot, args.run, status_text, records_summary)
    else:
        ods["dataset_description.data_entry.machine"] = "VEST"
        ods["dataset_description.data_entry.pulse"] = int(args.shot)
        ods["dataset_description.data_entry.run"] = int(args.run)
        if parse_errors:
            ods["equilibrium.ids_properties.comment"] = "CHEASE parse warnings: " + "; ".join(parse_errors[:5])
        ods["equilibrium.code.name"] = "chease"
        ods["equilibrium.code.library.0.name"] = "chease"
        ods["equilibrium.code.parameters"] = json.dumps(
            {
                "comparison_metrics": _comparison_metrics_by_time_index(gfiles, runs_summary),
                "records_summary": records_summary,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))

    # A refinement that parsed no g-file leaves a placeholder ODS on disk for
    # inspection. The manifest is where that is said, so nothing downstream
    # mistakes the placeholder for a refined equilibrium.
    write_manifest(
        {
            "schema_version": 1,
            "stage": "chease",
            "shot": int(args.shot),
            "run": int(args.run),
            "status": "success" if gfiles and not parse_errors else
                      "partial" if gfiles else "no_output",
            "chease_status": status_text,
            "gfiles": len(gfiles),
            "parse_errors": parse_errors,
            "records_summary": records_summary,
        },
        args.metadata,
    )
    LOGGER.info("CHEASE ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
