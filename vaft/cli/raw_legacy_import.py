"""Import pre-SQL VEST shots (per-field CSV folders) into a FileDB raw tree.

The legacy VEST database predates the ``shotDataWaveform`` SQL tables: every
shot is a directory ``<shot>/`` holding one headerless ``time,value`` CSV per
field code (``<code>.csv``) and an optional ``<shot>_remark.txt``.  This
command rewrites each such directory into the canonical raw archive
``raw/<shot>/vest_<shot>_daq_raw.json.gz`` plus its manifest, i.e. the exact
layout :func:`vaft.database.raw.dump_all_raw_signals_for_shot` produces, so the
mapping pipeline reads legacy shots through ``raw.mode: archive`` unchanged.

Existing products are never overwritten unless ``--force`` is given, so the
import can be re-run as more of the legacy tree becomes available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np

from vaft.cli.raw_redump import _exclusive_lock
from vaft.database import raw as raw_db
from vaft.database.filedb import FileDB
from vaft.omas.vest_upstream import sha256_file, write_manifest

# Same fast/slow split as ``dump_all_raw_signals_for_shot``.
SLOW_DT_THRESHOLD = 5e-6
# A timebase is uniform when rebuilding it as ``t0 + i*dt`` never misplaces a
# sample by more than this fraction of one sampling interval.
UNIFORM_TOLERANCE = 0.5
TIMEBASE_POLICIES = ("asis", "trigger")
# Shot lists in priority order; the newest list wins for a shot it covers.
SHOT_LIST_FILES = ("shotList3.csv", "shotList2.csv", "shotList.csv")
SOURCE_KIND = "legacy-csv"


class LegacyFieldError(ValueError):
    """A legacy CSV that cannot be represented as a ``t0``/``dt`` field."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def read_legacy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time, value)`` from one headerless two-column legacy CSV."""
    text = path.read_bytes().decode("ascii", errors="strict")
    if not text.strip():
        return np.array([], dtype=float), np.array([], dtype=float)
    flat = np.array(
        text.replace("\r", "").replace("\n", ",").rstrip(",").split(","),
        dtype=float,
    )
    if flat.size % 2:
        raise ValueError(f"odd number of values ({flat.size})")
    pairs = flat.reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def legacy_field_entry(
    time: np.ndarray,
    data: np.ndarray,
    *,
    time_offset: float = 0.0,
) -> dict:
    """Build one canonical archive field entry from a legacy time/value pair.

    ``time_offset`` is added to fast-DAQ records only, mirroring the trigger
    correction the SQL path applies to fast records.
    """
    n = len(time)
    if n == 0:
        return {"type": "unknown", "data": []}
    if n < 2:
        raise LegacyFieldError("too_short", f"{n} sample")
    t0 = float(time[0])
    dt = float((time[-1] - time[0]) / (n - 1))
    if not np.isfinite(dt) or dt <= 0:
        raise LegacyFieldError("nonuniform", f"dt={dt}")
    deviation = np.max(np.abs(time - (t0 + np.arange(n) * dt)))
    if deviation > UNIFORM_TOLERANCE * dt:
        raise LegacyFieldError("nonuniform", f"max deviation {deviation:.3g}s, dt={dt:.3g}s")
    daq_type = "slow" if (time[1] - time[0]) >= SLOW_DT_THRESHOLD else "fast"
    if daq_type == "fast":
        t0 += time_offset
    return {"type": daq_type, "data": data.tolist(), "t0": t0, "dt": dt}


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_pulse_datetimes(legacy_root: Path) -> dict[int, str]:
    """Map shot number to ISO pulse time from the legacy shot lists."""
    result: dict[int, str] = {}
    for name in reversed(SHOT_LIST_FILES):  # lowest priority first, overwritten later
        path = legacy_root / name
        if not path.is_file():
            continue
        rows = csv.DictReader(_read_text(path).splitlines())
        for row in rows:
            try:
                shot = int(row["shotNumber"])
                stamp = datetime.strptime(row["recordDateTime"].strip(), "%Y-%m-%d %H:%M")
            except (KeyError, TypeError, ValueError):
                continue
            result[shot] = stamp.isoformat()
    return result


def legacy_shot_dirs(legacy_root: Path, first: int | None, last: int | None) -> list[int]:
    shots = sorted(
        int(entry.name)
        for entry in os.scandir(legacy_root)
        if entry.is_dir() and entry.name.isdigit()
    )
    return [
        shot
        for shot in shots
        if (first is None or shot >= first) and (last is None or shot <= last)
    ]


def convert_shot(
    shot: int,
    shot_dir: Path,
    *,
    pulse_datetime: str | None,
    timebase_policy: str,
) -> tuple[dict, dict] | None:
    """Return ``(payload, legacy_info)`` for one legacy shot, or ``None`` if empty."""
    time_offset = (
        raw_db._daq_trigger_time_correction(shot) if timebase_policy == "trigger" else 0.0
    )
    fields: dict[str, dict] = {}
    field_quality: dict[str, str] = {}
    skipped: dict[str, str] = {}
    ignored_files: list[str] = []
    remark: str | None = None

    for path in sorted(shot_dir.iterdir()):
        stem = path.name[: -len(".csv")] if path.name.endswith(".csv") else None
        if stem is not None and stem.isdigit():
            code = str(int(stem))
            try:
                time, data = read_legacy_csv(path)
                entry = legacy_field_entry(time, data, time_offset=time_offset)
            except LegacyFieldError as error:
                skipped[code] = error.reason
                continue
            except (UnicodeDecodeError, ValueError):
                skipped[code] = "malformed"
                continue
            fields[code] = entry
            flag = raw_db._flagged_field_quality(np.asarray(entry["data"], dtype=float))
            if flag is not None:
                field_quality[code] = flag
        elif path.name == f"{shot}_remark.txt":
            remark = _read_text(path).strip() or None
        else:
            ignored_files.append(path.name)

    if not fields:
        return None
    payload: dict = {"shot": shot, "fields": dict(sorted(fields.items(), key=lambda kv: int(kv[0])))}
    if pulse_datetime is not None:
        payload["pulse_datetime"] = pulse_datetime
    if field_quality:
        payload["field_quality"] = dict(sorted(field_quality.items(), key=lambda kv: int(kv[0])))
    legacy_info = {
        "source_dir": str(shot_dir),
        "timebase_policy": timebase_policy,
        "fast_time_offset": time_offset,
        "remark": remark,
        "skipped_fields": dict(sorted(skipped.items(), key=lambda kv: int(kv[0]))),
        "ignored_files": ignored_files,
    }
    return payload, legacy_info


def build_manifest(shot: int, output: Path, payload: dict, legacy_info: dict) -> dict:
    """Raw-stage manifest, same shape as ``generate_raw_db_dump.build_raw_manifest``."""
    field_codes = sorted(int(code) for code in payload["fields"])
    field_quality = payload.get("field_quality", {})

    def flagged(flag: str) -> list[str]:
        return sorted((c for c, f in field_quality.items() if f == flag), key=int)

    return {
        "schema_version": 1,
        "stage": "raw",
        "shot": shot,
        "status": "success",
        "source": {"kind": SOURCE_KIND, "name": legacy_info["source_dir"]},
        "inventory": {"field_count": len(field_codes), "field_codes": field_codes},
        "pulse_datetime": payload.get("pulse_datetime"),
        "quality_summary": {
            "flagged_field_count": len(field_quality),
            "all_zero": flagged("all_zero"),
            "all_nan": flagged("all_nan"),
            "empty": flagged("empty"),
        },
        "legacy": legacy_info,
        "output": {"name": output.name, "sha256": sha256_file(output)},
    }


def import_shot(
    shot: int,
    legacy_root: str,
    filedb_root: str,
    pulse_datetime: str | None,
    timebase_policy: str,
    force: bool,
) -> tuple[int, str]:
    """Convert and atomically write one shot; returns ``(shot, status)``."""
    filedb = FileDB(filedb_root)
    output_dir = filedb.raw(shot)
    output = output_dir / f"vest_{shot}_daq_raw.json.gz"
    manifest_path = output_dir / f"vest_{shot}_daq_manifest.json"
    if output.exists() and not force:
        return shot, "skipped-existing"
    converted = convert_shot(
        shot,
        Path(legacy_root) / str(shot),
        pulse_datetime=pulse_datetime,
        timebase_policy=timebase_policy,
    )
    if converted is None:
        return shot, "skipped-empty"
    payload, legacy_info = converted
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".legacy-{shot}-", dir=output_dir) as tmpdir:
        temporary_output = Path(tmpdir) / output.name
        with gzip.open(temporary_output, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        manifest = build_manifest(shot, temporary_output, payload, legacy_info)
        manifest["output"]["name"] = output.name
        temporary_manifest = Path(tmpdir) / manifest_path.name
        write_manifest(manifest, temporary_manifest)
        os.replace(temporary_output, output)
        os.replace(temporary_manifest, manifest_path)
    skipped = legacy_info["skipped_fields"]
    note = f" ({len(skipped)} field(s) skipped)" if skipped else ""
    return shot, f"imported {len(payload['fields'])} field(s){note}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy-root", required=True, type=Path,
                        help="Directory holding the legacy <shot>/ folders.")
    parser.add_argument("--filedb-root", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--shots", nargs="+", type=int)
    selection.add_argument("--shot-range", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument(
        "--timebase-policy", choices=TIMEBASE_POLICIES, default="asis",
        help="asis: keep CSV time; trigger: add the SQL-path trigger offset to fast records.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing raw products.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the shots that would be imported and exit.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    legacy_root = args.legacy_root
    if args.shots:
        shots = sorted(set(args.shots))
    else:
        first, last = args.shot_range if args.shot_range else (None, None)
        shots = legacy_shot_dirs(legacy_root, first, last)
    if not shots:
        print("No legacy shots selected.")
        return 0
    print(f"Selected {len(shots)} legacy shot(s): {shots[0]}–{shots[-1]}", flush=True)
    if args.dry_run:
        return 0

    pulse_times = load_pulse_datetimes(legacy_root)
    filedb = FileDB(args.filedb_root)
    failures: list[int] = []
    jobs = [
        (shot, str(legacy_root), str(args.filedb_root), pulse_times.get(shot),
         args.timebase_policy, args.force)
        for shot in shots
    ]
    with _exclusive_lock(filedb.root / "raw"):
        if args.workers == 1:
            results = (_safe_import(*job) for job in jobs)
            for shot, status in results:
                print(f"shot {shot}: {status}", flush=True)
                if status.startswith("failed"):
                    failures.append(shot)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_safe_import, *job) for job in jobs]
                for future in as_completed(futures):
                    shot, status = future.result()
                    print(f"shot {shot}: {status}", flush=True)
                    if status.startswith("failed"):
                        failures.append(shot)
    if failures:
        print(f"Failed shot(s): {sorted(failures)}", file=sys.stderr)
        return 1
    return 0


def _safe_import(*job) -> tuple[int, str]:
    try:
        return import_shot(*job)
    except Exception as error:  # report and continue with the batch
        return job[0], f"failed: {type(error).__name__}: {error}"


__all__ = [
    "convert_shot",
    "import_shot",
    "legacy_field_entry",
    "load_pulse_datetimes",
    "main",
    "read_legacy_csv",
]
