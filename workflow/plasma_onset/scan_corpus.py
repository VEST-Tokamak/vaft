"""Scan the raw database for the plasma-onset corpus (issue #409).

The rules in the ``plasma_timing`` block of ``vest.yaml`` were tuned on a
study of the raw database; this script is that study, reproducible.  For
every shot it loads the processed plasma current (``vfit_plasma_current``)
and the slow H-alpha field, runs the raw-side plasma-window detector
(:func:`vaft.machine_mapping.magnetics.detect_plasma_window`, the same rules
and range the ODS composer applies) and records the verdict with both
detectors' windows.  The table it writes (``--table``, by default
``test/data/onset_corpus.json``) is what the tests pin the quoted corpus
numbers against; the per-shot ``.npz`` records (``--npz-dir``; keys
``t_ip, ip, t_ha, ha`` on the analysis span) feed ``review_onset.py`` and
stay outside the repository.

    python workflow/plasma_onset/scan_corpus.py --shots 39900-41700 --npz-dir /path/corpus
    python workflow/plasma_onset/scan_corpus.py --shots 39915,41524,41672 --table /tmp/check.json

Shots the database does not carry both signals for are recorded as
``absent``, not skipped: which shots could not be judged is part of the
corpus.
"""
from __future__ import annotations

import argparse
import json
import logging
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vaft.cli.maintenance import _shots as parse_shot_specs
from vaft.database import raw as raw_db
from vaft.machine_mapping.magnetics import (
    HALPHA_RAW_FIELD,
    PLASMA_WINDOW_ANALYSIS_RANGE,
    _safe_vest_load,
    detect_plasma_window,
    vfit_plasma_current,
)
from vaft.machine_mapping.utils import (
    AGREEMENT_CONSISTENT,
    onset_agreement,
    resolve_plasma_timing_policy,
)

LOGGER = logging.getLogger("scan_corpus")

CORPUS_SCHEMA = 1
DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "test" / "data" / "onset_corpus.json"
#: Rows are checkpointed to the table this often, so an aborted scan keeps what it did.
CHECKPOINT_EVERY = 100


def parse_shots(text: str) -> list[int]:
    """``39900-41700`` (inclusive) or ``39915,41524,41672``, or a mix -- the CLI's own rule."""
    return parse_shot_specs(part for part in text.split(",") if part.strip())


def _window_record(window: dict[str, Any] | None) -> dict[str, Any] | None:
    """The part of a ``PulseWindow.as_dict()`` the corpus keeps: ``found`` is ``start is not None``."""
    if window is None:
        return None
    found = window.get("onset", {}).get("time") is not None and window.get("offset", {}).get("time") is not None
    return {
        "start": float(window["start"]) if found else None,
        "end": float(window["end"]) if found else None,
        "flags": list(window.get("flags", ())),
    }


def _found(record: dict[str, Any] | None) -> bool:
    return bool(record and record.get("start") is not None)


def scan_shot(shot: int, *, policy, npz_dir: Path | None) -> dict[str, Any]:
    """One corpus row: the raw-side verdict and both detectors' windows for ``shot``.

    A shot the database does not carry both signals for is ``absent``; any
    other failure is ``error`` with the exception, never filed as missing
    data -- a detector fault must stay visible in the corpus.  Configuration
    errors propagate: they are not per-shot facts.
    """
    row: dict[str, Any] = {"shot": int(shot)}
    try:
        ip_time, ip = vfit_plasma_current(shot)
    except raw_db.RawSignalUnavailableError as exc:
        row.update({"status": "absent", "reason": f"plasma current: {exc}"})
        return row
    ip_time = np.asarray(ip_time, float).reshape(-1)
    ip = np.asarray(ip, float).reshape(-1)
    halpha = _safe_vest_load(shot, HALPHA_RAW_FIELD)   # cached: the detector reads the same record
    if halpha is None:
        row.update({"status": "absent", "reason": f"field {HALPHA_RAW_FIELD} (slow H-alpha) not recorded"})
        return row
    if ip_time.size == 0 or float(ip_time[-1]) <= float(policy.window.tstart):
        row.update({"status": "absent", "reason": "the plasma current record ends before the analysis range"})
        return row
    try:
        choice = detect_plasma_window(shot, ip_time, ip, policy=policy)
    except Exception as exc:  # noqa: BLE001 - recorded, and the test refuses a table with errors
        row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        return row
    evidence = dict(choice.evidence)
    light, current = _window_record(evidence.get("h_alpha")), _window_record(evidence.get("ip"))
    row.update({
        "status": "judged",
        "source": choice.source,
        "start": float(choice.start),
        "end": float(choice.end),
        "flags": list(choice.flags),
        "fallback": bool(choice.fallback),
        "h_alpha": light,
        "ip": current,
    })
    if evidence.get("h_alpha_unusable"):
        row["h_alpha_unusable"] = evidence["h_alpha_unusable"]
    if _found(light) and _found(current):
        row["agreement"] = onset_agreement(current["start"] - light["start"], policy.agreement)
    if npz_dir is not None:
        t_ha, ha = np.asarray(halpha[0], float).reshape(-1), np.asarray(halpha[1], float).reshape(-1)
        keep_ip = (ip_time >= policy.baseline_start) & (ip_time < policy.window.tend)
        keep_ha = (t_ha >= policy.baseline_start) & (t_ha < policy.window.tend)
        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_dir / f"{shot}.npz",
            t_ip=ip_time[keep_ip].astype(np.float32), ip=ip[keep_ip].astype(np.float32),
            t_ha=t_ha[keep_ha].astype(np.float32), ha=ha[keep_ha].astype(np.float32),
        )
    return row


def summarize(rows: Iterable[dict[str, Any]], policy) -> dict[str, Any]:
    """The corpus numbers, counted from the rows -- the table's own summary, re-derivable."""
    rows = list(rows)
    judged = [r for r in rows if r.get("status") == "judged"]
    windows = [r for r in judged if r["source"] != PLASMA_WINDOW_ANALYSIS_RANGE]
    by_source: dict[str, int] = {}
    for r in judged:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    light = [r for r in judged if _found(r["h_alpha"])]
    current = [r for r in judged if _found(r["ip"])]
    both = [r for r in light if _found(r["ip"])]
    agreement: dict[str, int] = {}
    for r in both:
        agreement[r["agreement"]] = agreement.get(r["agreement"], 0) + 1
    consistent = [r for r in both if r["agreement"] == AGREEMENT_CONSISTENT]
    deltas = [r["ip"]["start"] - r["h_alpha"]["start"] for r in consistent]
    end_deltas = [r["ip"]["end"] - r["h_alpha"]["end"] for r in consistent]
    starts = [r["start"] for r in windows]
    ends = [r["end"] for r in windows]

    def spread(values):
        return None if not values else {
            "min": float(min(values)), "max": float(max(values)), "median": float(np.median(values)),
        }

    return {
        "shots": len(rows),
        "judged": len(judged),
        "absent": sum(r.get("status") == "absent" for r in rows),
        "errors": [r["shot"] for r in rows if r.get("status") == "error"],
        "windows": len(windows),
        "by_source": by_source,
        "light_windows": len(light),
        "current_windows": len(current),
        "both": len(both),
        "agreement": agreement,
        "light_only": [r["shot"] for r in light if not _found(r["ip"])],
        "current_only": [r["shot"] for r in current if not _found(r["h_alpha"])],
        "h_alpha_unusable": [r["shot"] for r in judged if r.get("h_alpha_unusable")],
        "no_plasma": len(judged) - len(windows),
        "start_range_s": [min(starts), max(starts)] if starts else None,
        "end_range_s": [min(ends), max(ends)] if ends else None,
        "consistent_ip_minus_light_onset_s": spread(deltas),
        "consistent_ip_minus_light_offset_s": spread(end_deltas),
        "inside_range": all(
            policy.window.tstart <= r["start"] <= r["end"] <= policy.window.tend for r in windows
        ),
    }


def write_table(path: Path, *, shots_requested: str, policy, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write the table: one row per line, so a re-scan diffs per shot."""
    table = {
        "schema_version": CORPUS_SCHEMA,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shots_requested": shots_requested,
        "policy": policy.as_dict(),
        "summary": summarize(rows, policy),
    }
    head = json.dumps(table, indent=1)[:-2]   # drop the closing "\n}"
    lines = ",\n".join("  " + json.dumps(row, separators=(", ", ": ")) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{head},\n \"rows\": [\n{lines}\n ]\n}}\n", encoding="utf-8")
    table["rows"] = rows
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", required=True, help="range 'A-B' and/or comma-separated shots")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="corpus table to write (JSON)")
    parser.add_argument("--npz-dir", type=Path, default=None, help="write per-shot npz records here (external)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    policy = resolve_plasma_timing_policy()
    shots = parse_shots(args.shots)
    if not shots:
        parser.error(f"--shots {args.shots!r} names no shot (a range is 'first-last', first <= last)")
    rows: list[dict[str, Any]] = []
    started = _time.time()
    for index, shot in enumerate(shots, 1):
        row = scan_shot(shot, policy=policy, npz_dir=args.npz_dir)
        rows.append(row)
        LOGGER.info("%s (%d/%d): %s", shot, index, len(shots), row.get("source") or row.get("status"))
        if index % CHECKPOINT_EVERY == 0:
            write_table(args.table, shots_requested=args.shots, policy=policy, rows=rows)
    table = write_table(args.table, shots_requested=args.shots, policy=policy, rows=rows)
    summary = table["summary"]
    print(f"{summary['judged']}/{summary['shots']} shots judged, {summary['windows']} plasma windows "
          f"({summary['by_source']}), {summary['no_plasma']} without a plasma, "
          f"{summary['absent']} absent, {len(summary['errors'])} errors; "
          f"{_time.time() - started:.0f} s -> {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
