"""Choose the Tier A slices the stability atlas may use (#1429, #1331).

Only slices whose EFIT the #1331 criteria label ``good`` or ``admissible`` are
selected. ``unreconstructible`` slices never produce an atlas row.

Two lineages come out of the campaign analysis (``tierA_analysis*.json``):

``magnetics-only``
    Every entry of ``labels`` with label good/admissible. The g-file is the
    campaign's magnetic EFIT output at that time.
``electron-kinetic``
    One electron-EFIT reconstruction per shot (``kinetic``). It inherits the
    label of the magnetics-only slice **at exactly the same time**. Times are
    integer milliseconds on both sides, so the allowed mismatch is 0 ms and a
    kinetic entry without an exact partner is dropped, never paired with a
    neighbour. Its g-file is written from the electron-EFIT ODS later, by
    ``run_batch.py``.

The state key follows the provisional contract in #1448:
``(shot, time_efit_s, efit_lineage)``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

MAGNETICS_ONLY = "magnetics-only"
ELECTRON_KINETIC = "electron-kinetic"
ADMITTED_LABELS = ("good", "admissible")
COLUMNS = (
    "shot",
    "time_ms",
    "time_efit_s",
    "efit_lineage",
    "efit_label",
    "efit_setting",
    "kinetic_chi2",
)


def select(analysis: dict[str, Any], *, shots: Iterable[int] | None = None) -> tuple[list[dict], list[dict]]:
    """Return ``(selected, dropped)`` rows for the atlas.

    ``dropped`` records each kinetic entry that was not admitted and why. The
    magnetics-only slices that fail are not listed: they are the large majority,
    and their labels are already in the analysis file.
    """
    allowed = None if shots is None else {int(s) for s in shots}
    labels: dict[tuple[int, int], dict] = {}
    for row in analysis["labels"]:
        key = (int(row["shot"]), int(row["time_ms"]))
        if key in labels:
            raise ValueError(f"duplicate label for shot {key[0]} at {key[1]} ms")
        labels[key] = row

    selected: list[dict] = []
    for (shot, time_ms), row in sorted(labels.items()):
        if allowed is not None and shot not in allowed:
            continue
        if row["label"] in ADMITTED_LABELS:
            selected.append(_row(shot, time_ms, MAGNETICS_ONLY, row["label"], _setting(row)))

    dropped: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for entry in analysis.get("kinetic", []):
        shot, time_ms = int(entry["shot"]), int(entry["time_ms"])
        if (shot, time_ms) in seen:
            raise ValueError(f"duplicate electron-kinetic entry for shot {shot} at {time_ms} ms")
        seen.add((shot, time_ms))
        if allowed is not None and shot not in allowed:
            continue
        partner = labels.get((shot, time_ms))
        if partner is None:
            dropped.append({"shot": shot, "time_ms": time_ms, "reason": "no magnetics-only label at this time"})
            continue
        if partner["label"] not in ADMITTED_LABELS:
            dropped.append({"shot": shot, "time_ms": time_ms, "reason": f"label {partner['label']}"})
            continue
        selected.append(
            _row(shot, time_ms, ELECTRON_KINETIC, partner["label"], _setting(partner), kinetic_chi2=entry.get("chi2"))
        )
    selected.sort(key=lambda r: (r["shot"], r["time_ms"], r["efit_lineage"]))
    return selected, dropped


def _setting(row: dict) -> str:
    settings = row.get("admissible") or []
    return settings[0] if len(settings) == 1 else ";".join(settings)


def _row(shot: int, time_ms: int, lineage: str, label: str, setting: str, *, kinetic_chi2: Any = None) -> dict:
    return {
        "shot": shot,
        "time_ms": time_ms,
        "time_efit_s": time_ms / 1000.0,
        "efit_lineage": lineage,
        "efit_label": label,
        "efit_setting": setting,
        "kinetic_chi2": kinetic_chi2,
    }


def write(rows: list[dict], path: Path) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["shot"] = int(row["shot"])
        row["time_ms"] = int(row["time_ms"])
        row["time_efit_s"] = float(row["time_efit_s"])
        row["kinetic_chi2"] = float(row["kinetic_chi2"]) if row["kinetic_chi2"] else None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True, help="tierA_analysis.json")
    parser.add_argument("--out", type=Path, required=True, help="slices.csv to write")
    parser.add_argument("--shots", type=int, nargs="*", default=None)
    args = parser.parse_args()
    selected, dropped = select(json.loads(args.analysis.read_text()), shots=args.shots)
    write(selected, args.out)
    counts: dict[tuple[str, str], int] = {}
    for row in selected:
        key = (row["efit_lineage"], row["efit_label"])
        counts[key] = counts.get(key, 0) + 1
    for key, count in sorted(counts.items()):
        print(f"{key[0]:18s} {key[1]:11s} {count}")
    for row in dropped:
        print(f"dropped kinetic {row['shot']}@{row['time_ms']} ms: {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
