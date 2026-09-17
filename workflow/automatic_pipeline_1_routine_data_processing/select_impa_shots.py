#!/usr/bin/env python3
"""Select the shots whose IMPA product is worth producing (issue #305).

The `impa` HSDS source is sparse and explicit: it holds only the shots for which
an IMPA product was intentionally produced, so a shot missing from it means "no
published IMPA product" and nothing more. That meaning survives only if the
pipeline attempts IMPA where the array was actually recording, rather than for
every eligible shot and then filling the source with empty or failed products.

A shot is selected when its raw dump archives at least one of the Hall channels
its own machine era wires -- the 2022-04-23 block wires seven, not eight, so the
per-shot list from `impa_expected_fields` is what is checked, never the
unfiltered default.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from vaft.machine_mapping.impa import impa_expected_fields


def _archived_fields(path: Path) -> set[int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(code) for code in payload.get("fields", {})}


def select_impa_shots(shots, raw_dumps: dict[int, Path]) -> list[dict[str, Any]]:
    """Return one report per shot, saying whether IMPA is worth attempting."""
    reports = []
    for shot in sorted(int(value) for value in shots):
        dump = raw_dumps.get(shot)
        if dump is None or not Path(dump).exists():
            reports.append({"shot": shot, "selected": False, "reason": "no raw dump"})
            continue
        try:
            expected = sorted(impa_expected_fields(shot))
        except ValueError as error:
            reports.append({"shot": shot, "selected": False, "reason": str(error)})
            continue
        archived = _archived_fields(Path(dump))
        present = [field for field in expected if field in archived]
        reports.append(
            {
                "shot": shot,
                "selected": bool(present),
                "expected_fields": expected,
                "archived_fields": present,
                "reason": "" if present else "no wired IMPA channel is archived",
            }
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dumps", nargs="*", type=Path)
    parser.add_argument("--eligible", required=True, type=Path, help="Preflight eligible-shots JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Selected-shots JSON.")
    args = parser.parse_args()

    eligible = json.loads(args.eligible.read_text(encoding="utf-8"))["eligible_shots"]
    dumps: dict[int, Path] = {}
    for path in args.raw_dumps:
        try:
            dumps[int(path.name.split("_")[1])] = path
        except (IndexError, ValueError):
            continue

    reports = select_impa_shots(eligible, dumps)
    selected = [report["shot"] for report in reports if report["selected"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"impa_shots": selected, "reports": reports}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"IMPA selection: {len(selected)} of {len(reports)} eligible shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
