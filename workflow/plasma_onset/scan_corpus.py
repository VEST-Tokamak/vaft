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

from vaft.database import raw as raw_db
from vaft.machine_mapping.magnetics import (
    PLASMA_WINDOW_ANALYSIS_RANGE,
    detect_plasma_window,
    vfit_plasma_current,
)
from vaft.machine_mapping.spectrometer_uv import SIGNALS
from vaft.machine_mapping.utils import crop_to_span, resolve_plasma_timing_policy

LOGGER = logging.getLogger("scan_corpus")

CORPUS_SCHEMA = 1
DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "test" / "data" / "onset_corpus.json"
#: The slow H-alpha field: the first configured signal, channel 0 line 0.
H_ALPHA_FIELD = next(field for field, channel, line, _, _ in SIGNALS if channel == 0 and line == 0)


def parse_shots(text: str) -> list[int]:
    """``39900-41700`` (inclusive) or ``39915,41524,41672``, or a mix."""
    shots: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            shots.extend(range(lo, hi + 1))
        else:
            shots.append(int(part))
    return sorted(set(shots))


def _window_record(window: dict[str, Any] | None) -> dict[str, Any] | None:
    """The part of a ``PulseWindow.as_dict()`` the corpus keeps."""
    if window is None:
        return None
    found = window.get("onset", {}).get("time") is not None and window.get("offset", {}).get("time") is not None
    return {
        "found": bool(found),
        "start": float(window["start"]) if found else None,
        "end": float(window["end"]) if found else None,
        "flags": list(window.get("flags", ())),
        "onset_flags": list(window.get("onset", {}).get("flags", ())),
    }


def scan_shot(shot: int, *, policy, npz_dir: Path | None) -> dict[str, Any]:
    """One corpus row: the raw-side verdict and both detectors' windows for ``shot``."""
    row: dict[str, Any] = {"shot": int(shot)}
    try:
        ip_time, ip = vfit_plasma_current(shot)
    except (raw_db.RawSignalUnavailableError, LookupError, ValueError) as exc:
        row.update({"status": "absent", "reason": f"plasma current: {exc}"})
        return row
    ip_time = np.asarray(ip_time, float).reshape(-1)
    ip = np.asarray(ip, float).reshape(-1)
    halpha = raw_db.vest_load(shot, H_ALPHA_FIELD)
    if halpha is None:
        row.update({"status": "absent", "reason": f"field {H_ALPHA_FIELD} (slow H-alpha) not recorded"})
        return row
    try:
        choice = detect_plasma_window(shot, ip_time, ip, policy=policy)
    except ValueError as exc:
        row.update({"status": "absent", "reason": str(exc)})
        return row
    evidence = dict(choice.evidence)
    span = dict(baseline_start=policy.baseline_start, tstart=policy.window.tstart, tend=policy.window.tend)
    if "ip" not in evidence:
        # the detector stops at the light; the corpus wants the current's window too
        evidence["ip"] = crop_to_span(ip_time, ip, **span).detect(policy.ip).as_dict()
    row.update({
        "status": "judged",
        "source": choice.source,
        "start": float(choice.start),
        "end": float(choice.end),
        "flags": list(choice.flags),
        "fallback": bool(choice.fallback),
        "h_alpha": _window_record(evidence.get("h_alpha")),
        "h_alpha_unusable": evidence.get("h_alpha_unusable"),
        "ip": _window_record(evidence.get("ip")),
        "n_samples": {"ip": int(ip.size), "h_alpha": int(np.size(halpha[1]))},
    })
    if npz_dir is not None:
        t_ha, ha = np.asarray(halpha[0], float).reshape(-1), np.asarray(halpha[1], float).reshape(-1)
        keep_ip = (ip_time >= span["baseline_start"]) & (ip_time < span["tend"])
        keep_ha = (t_ha >= span["baseline_start"]) & (t_ha < span["tend"])
        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_dir / f"{shot}.npz",
            t_ip=ip_time[keep_ip].astype(np.float32), ip=ip[keep_ip].astype(np.float32),
            t_ha=t_ha[keep_ha].astype(np.float32), ha=ha[keep_ha].astype(np.float32),
        )
        row["npz"] = f"{shot}.npz"
    return row


def summarize(rows: Iterable[dict[str, Any]], policy) -> dict[str, Any]:
    """The corpus numbers the policy comments quote, counted from the rows."""
    rows = list(rows)
    judged = [r for r in rows if r.get("status") == "judged"]
    windows = [r for r in judged if r["source"] != PLASMA_WINDOW_ANALYSIS_RANGE]
    by_source: dict[str, int] = {}
    for r in judged:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    light = [r for r in judged if r["h_alpha"] and r["h_alpha"]["found"]]
    current = [r for r in judged if r["ip"] and r["ip"]["found"]]
    both = [r for r in light if r["ip"] and r["ip"]["found"]]
    light_only = [r for r in light if not (r["ip"] and r["ip"]["found"])]
    current_only = [r for r in current if not (r["h_alpha"] and r["h_alpha"]["found"])]
    deltas = [r["ip"]["start"] - r["h_alpha"]["start"] for r in both]
    end_deltas = [r["ip"]["end"] - r["h_alpha"]["end"] for r in both]
    starts = [r["start"] for r in windows]
    ends = [r["end"] for r in windows]
    return {
        "shots": len(rows),
        "judged": len(judged),
        "absent": len(rows) - len(judged),
        "windows": len(windows),
        "by_source": by_source,
        "light_windows": len(light),
        "current_windows": len(current),
        "both": len(both),
        "light_only": [r["shot"] for r in light_only],
        "current_only": [r["shot"] for r in current_only],
        "h_alpha_unusable": [r["shot"] for r in judged if r.get("h_alpha_unusable")],
        "no_plasma": len(judged) - len(windows),
        "start_range_s": [min(starts), max(starts)] if starts else None,
        "end_range_s": [min(ends), max(ends)] if ends else None,
        "ip_minus_light_onset_s": (
            {"min": float(min(deltas)), "max": float(max(deltas)), "median": float(np.median(deltas))}
            if deltas else None
        ),
        "ip_minus_light_offset_s": (
            {"min": float(min(end_deltas)), "max": float(max(end_deltas)), "median": float(np.median(end_deltas))}
            if end_deltas else None
        ),
        "inside_range": all(
            policy.window.tstart <= r["start"] <= r["end"] <= policy.window.tend for r in windows
        ),
    }


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
    rows = []
    started = _time.time()
    for index, shot in enumerate(shots, 1):
        row = scan_shot(shot, policy=policy, npz_dir=args.npz_dir)
        rows.append(row)
        LOGGER.info("%s (%d/%d): %s", shot, index, len(shots), row.get("source") or row.get("status"))
    table = {
        "schema_version": CORPUS_SCHEMA,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shots_requested": args.shots,
        "policy": policy.as_dict(),
        "summary": summarize(rows, policy),
        "rows": rows,
    }
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    summary = table["summary"]
    print(f"{summary['judged']}/{summary['shots']} shots judged, {summary['windows']} plasma windows "
          f"({summary['by_source']}), {summary['no_plasma']} without a plasma, "
          f"{summary['absent']} absent; {_time.time() - started:.0f} s -> {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
