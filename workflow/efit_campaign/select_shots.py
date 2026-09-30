#!/usr/bin/env python3
"""Choose the shots for the #1331 EFIT campaign.

Joins three sources that answer different questions, per shot:

* the database's operational overview (``automatic_pipeline_3_data_summary``
  ``gen_omas_history.py``: peak Ip, pulse duration, shot class);
* the raw Thomson and ion uploads under the external data root (the
  ``/srv/vest.diagnostic`` layout), which decide whether the kinetic lineage
  can run at all -- file presence, so it stays out of the database summary;
* one or more magnetics-quality tables from
  ``workflow/magnetics_quality/scan_magnetics_quality.py``.

and labels each shot by the tiers #1331 set out:

``A``  Thomson, and magnetics like 39915's (few condemned probes, most of the
       outboard array usable) -- the calibration and kinetic set.
``B``  Thomson and an important discharge (record-class Ip or a long pulse)
       whose magnetics fall short of A but are not unfit.
``C``  Thomson, but no magnetics in the database -- needs ingest first.
``D``  no Thomson, but important with A-quality magnetics -- magnetic-only.

A shot is in at most one tier, the first that applies.  Thresholds are
arguments, recorded in the output, not constants.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vaft.machine_mapping.thomson_scattering import thomson_file_shot

#: Where the Thomson resolver searches, relative to the data root.
SEARCH_SUBDIRS = ("thomson_scattering", "legacy", "")
ION_FILE = re.compile(r"^(?P<kind>CES|IDS)_(?P<shot>\d+)\.mat$", re.IGNORECASE)


@dataclass(frozen=True)
class Thresholds:
    max_condemned: int = 4
    min_outboard_usable: int = 19
    important_ip_kA: float = 250.0
    important_duration_s: float = 0.030


def raw_inventory(root: Path) -> tuple[dict[int, str], dict[int, str]]:
    """Thomson and ion files per shot under ``root``: ``({shot: file}, {shot: file})``."""
    thomson: dict[int, str] = {}
    ions: dict[int, str] = {}
    for subdir in SEARCH_SUBDIRS:
        directory = root / subdir if subdir else root
        try:
            names = sorted(entry.name for entry in directory.iterdir() if entry.is_file())
        except OSError:
            continue
        for name in names:
            shot = thomson_file_shot(name)
            if shot is not None:
                thomson.setdefault(shot, name)
                continue
            match = ION_FILE.match(name)
            if match:
                ions.setdefault(int(match.group("shot")), name)
    return thomson, ions


def quality_by_shot(paths: Iterable[Path]) -> dict[int, dict[str, Any]]:
    """One magnetics-quality row per shot from table files or directories of them.

    A later table wins for a shot scanned twice.
    """
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    out: dict[int, dict[str, Any]] = {}
    for file in files:
        for row in json.loads(file.read_text(encoding="utf-8")).get("rows", []):
            out[int(row["shot"])] = row
    return out


def _quality_summary(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"magnetics": "not_scanned", "condemned": None, "outboard_usable": None}
    if row.get("status") == "absent":
        return {"magnetics": "absent", "condemned": None, "outboard_usable": None}
    families = row.get("families") or {}
    return {
        "magnetics": row.get("verdict") or row.get("status") or "unknown",
        "condemned": len(row.get("condemned") or []),
        "outboard_usable": (families.get("outboard") or {}).get("usable"),
    }


def classify(overview: Mapping[str, Any] | None, quality: Mapping[str, Any], *, thomson: bool,
             thresholds: Thresholds) -> dict[str, Any]:
    """The flags and the tier for one shot."""
    ip = (overview or {}).get("max_ip_kA")
    duration = (overview or {}).get("pulse_duration_s")
    important = bool(
        (ip is not None and ip == ip and ip >= thresholds.important_ip_kA)
        or (duration is not None and duration == duration and duration >= thresholds.important_duration_s)
    )
    scanned = quality["magnetics"] not in ("not_scanned", "absent", "unknown")
    good = bool(
        scanned
        and quality["magnetics"] != "unfit"
        and quality["condemned"] is not None and quality["condemned"] <= thresholds.max_condemned
        and quality["outboard_usable"] is not None and quality["outboard_usable"] >= thresholds.min_outboard_usable
    )
    if thomson and good:
        tier = "A"
    elif thomson and important and scanned and quality["magnetics"] != "unfit":
        tier = "B"
    elif thomson and quality["magnetics"] == "absent":
        tier = "C"
    elif important and good:
        tier = "D"
    else:
        tier = None
    return {"important": important, "good_magnetics": good, "tier": tier}


def select(overview_rows: Iterable[Mapping[str, Any]], quality: Mapping[int, Mapping[str, Any]],
           thomson: Mapping[int, str], ions: Mapping[int, str], thresholds: Thresholds) -> list[dict[str, Any]]:
    """Every shot any source knows of, with its tier (``None`` when it is in none)."""
    overview = {int(row["shot"]): dict(row) for row in overview_rows}
    shots = sorted(set(overview) | set(quality) | set(thomson))
    rows = []
    for shot in shots:
        summary = _quality_summary(quality.get(shot))
        base = overview.get(shot)
        rows.append({
            "shot": shot,
            "max_ip_kA": (base or {}).get("max_ip_kA"),
            "pulse_duration_s": (base or {}).get("pulse_duration_s"),
            "shot_class": (base or {}).get("shot_class"),
            "in_overview": base is not None,
            "thomson_file": thomson.get(shot),
            "ion_file": ions.get(shot),
            **summary,
            **classify(base, summary, thomson=shot in thomson, thresholds=thresholds),
        })
    return rows


def _read_overview(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_excel(path) if path.suffix in (".xlsx", ".xls") else pd.read_csv(path)
    return frame.to_dict("records")


def markdown(rows: list[dict[str, Any]], thresholds: Thresholds) -> str:
    lines = ["# #1331 campaign shot selection", "", f"Thresholds: `{json.dumps(asdict(thresholds))}`", ""]
    for tier in ("A", "B", "C", "D"):
        members = [r for r in rows if r["tier"] == tier]
        lines.append(f"## Tier {tier} ({len(members)})")
        lines.append("")
        if members:
            lines.append("| shot | Ip max [kA] | pulse [ms] | magnetics | condemned | outboard | Thomson | ion |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in members:
                ip = f"{r['max_ip_kA']:.0f}" if isinstance(r["max_ip_kA"], (int, float)) and r["max_ip_kA"] == r["max_ip_kA"] else "—"
                dur = (f"{1e3 * r['pulse_duration_s']:.1f}" if isinstance(r["pulse_duration_s"], (int, float))
                       and r["pulse_duration_s"] == r["pulse_duration_s"] else "—")
                lines.append(f"| {r['shot']} | {ip} | {dur} | {r['magnetics']} | {r['condemned'] if r['condemned'] is not None else '—'} "
                             f"| {r['outboard_usable'] if r['outboard_usable'] is not None else '—'} "
                             f"| {r['thomson_file'] or ''} | {r['ion_file'] or ''} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overview", type=Path, required=True, help="shot_overview .xlsx/.csv from gen_omas_history.py")
    parser.add_argument("--quality", type=Path, nargs="+", default=[], help="magnetics-quality tables or directories")
    parser.add_argument("--data-root", type=Path, required=True, help="external raw-diagnostic root (Thomson, CES/IDS)")
    parser.add_argument("--output", type=Path, required=True, help="selection JSON")
    parser.add_argument("--markdown", type=Path, default=None)
    defaults = Thresholds()
    parser.add_argument("--max-condemned", type=int, default=defaults.max_condemned)
    parser.add_argument("--min-outboard-usable", type=int, default=defaults.min_outboard_usable)
    parser.add_argument("--important-ip-kA", type=float, default=defaults.important_ip_kA)
    parser.add_argument("--important-duration-s", type=float, default=defaults.important_duration_s)
    args = parser.parse_args(argv)

    thresholds = Thresholds(args.max_condemned, args.min_outboard_usable, args.important_ip_kA,
                            args.important_duration_s)
    thomson, ions = raw_inventory(args.data_root)
    rows = select(_read_overview(args.overview), quality_by_shot(args.quality), thomson, ions, thresholds)
    payload = {"thresholds": asdict(thresholds), "data_root": str(args.data_root),
               "overview": str(args.overview), "quality": [str(p) for p in args.quality], "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    text = markdown(rows, thresholds)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    counts = {tier: sum(r["tier"] == tier for r in rows) for tier in "ABCD"}
    print(f"{len(rows)} shots; tiers {counts}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
