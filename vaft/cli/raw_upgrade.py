"""In-place timebase upgrade for legacy FileDB raw archives.

The pre-2026-08 archive schema labeled every field only ``fast``/``slow`` and
reconstruction assumed exactly 250 kHz / 25 kHz, silently stretching any other
cadence (a 2 MHz outboard-Mirnov record loads eightfold too slow in time).
Because the VEST tables encode time with a span fixed by DAQ class, the true
cadence is fully recoverable from the archive alone; this command rewrites each
dump with explicit per-field ``t0``/``dt`` entries, equivalent in meaning to a
fresh export from the live database, without touching a single data sample.

The upgrade is idempotent and restartable: files whose fields already carry
``t0``/``dt`` are skipped, so it can be re-run after a backfill adds shots.
Optional spot verification (``--verify-every N``) reloads one fast and one slow
field of every Nth upgraded shot from the live database and compares the
reconstructed timebase.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np

from vaft.database import raw as raw_db
from vaft.cli.raw_redump import _exclusive_lock, _manifest_for_existing_dump
from vaft.omas.vest_upstream import write_manifest


def _iter_dumps(root: Path) -> Iterable[tuple[int, Path]]:
    raw_root = root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"{raw_root} is not a directory")
    for shot_dir in sorted(raw_root.iterdir(), key=lambda p: p.name):
        if not shot_dir.is_dir() or not shot_dir.name.isdigit():
            continue
        shot = int(shot_dir.name)
        dump = shot_dir / f"vest_{shot}_daq_raw.json.gz"
        if dump.is_file():
            yield shot, dump


def _verify_against_db(shot: int, path: Path) -> None:
    """Compare one fast and one slow field's timebase against live SQL."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    picked: dict[str, str] = {}
    for code, entry in payload["fields"].items():
        label = entry.get("type")
        if label in ("fast", "slow") and label not in picked and len(entry.get("data", [])) > 2:
            picked[label] = code
    for label, code in picked.items():
        archived = raw_db.load_raw(shot, int(code), sample_opt=str(path))
        live = raw_db.load_raw(shot, int(code))
        if archived is None or live is None:
            raise RuntimeError(f"shot {shot} field {code}: verification load failed")
        archived_t, live_t = np.ravel(archived[0]), np.ravel(live[0])
        if archived_t.size != live_t.size:
            raise RuntimeError(
                f"shot {shot} field {code}: sample-count mismatch "
                f"{archived_t.size} vs {live_t.size}"
            )
        # The DB stores times as ~7-significant-digit strings; the linear
        # model must agree within that quantization.
        worst = float(np.max(np.abs(archived_t - live_t)))
        if worst > 1e-6:
            raise RuntimeError(
                f"shot {shot} field {code} ({label}): reconstructed timebase "
                f"deviates from the live DB by {worst:.3g} s"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaft raw-upgrade", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--filedb-root", type=Path, required=True)
    parser.add_argument("--first-shot", type=int, default=None,
                        help="Lowest shot to touch (default: every dump found).")
    parser.add_argument("--last-shot", type=int, default=None,
                        help="Highest shot to touch (default: every dump found).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything.")
    parser.add_argument("--verify-every", type=int, default=0, metavar="N",
                        help="Spot-verify every Nth upgraded shot against live SQL "
                             "(0 disables; needs DB access).")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    root = args.filedb_root.expanduser()

    touched = unchanged = failed = 0
    non_nominal_shots: list[int] = []
    verified = 0
    upgraded_seen = 0

    def process() -> None:
        nonlocal touched, unchanged, failed, verified, upgraded_seen
        for shot, dump in _iter_dumps(root):
            if args.first_shot is not None and shot < args.first_shot:
                continue
            if args.last_shot is not None and shot > args.last_shot:
                continue
            try:
                with gzip.open(dump, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                report = raw_db.upgrade_archive_timebase(payload)
                if report["upgraded"] == 0:
                    unchanged += 1
                    continue
                if report["non_nominal"]:
                    non_nominal_shots.append(shot)
                    print(f"shot {shot}: {len(report['non_nominal'])} non-nominal "
                          f"field(s) {report['non_nominal'][:8]}"
                          f"{'...' if len(report['non_nominal']) > 8 else ''}")
                if args.dry_run:
                    touched += 1
                    continue
                # Atomic rewrite next to the target, then refresh the sidecar
                # manifest so its sha256 matches the upgraded dump.
                with tempfile.NamedTemporaryFile(
                    "wb", dir=dump.parent, prefix=dump.name, suffix=".tmp", delete=False
                ) as handle:
                    with gzip.open(handle, "wt", encoding="utf-8") as gz:
                        json.dump(payload, gz)
                    temporary = Path(handle.name)
                os.replace(temporary, dump)
                manifest_path = dump.parent / f"vest_{shot}_daq_manifest.json"
                write_manifest(_manifest_for_existing_dump(shot, dump, payload), manifest_path)
                touched += 1
                upgraded_seen += 1
                if args.verify_every > 0 and upgraded_seen % args.verify_every == 0:
                    _verify_against_db(shot, dump)
                    verified += 1
            except Exception as error:
                failed += 1
                print(f"shot {shot}: FAILED ({type(error).__name__}: {error})")
                raise

    if args.dry_run:
        process()
    else:
        with _exclusive_lock(root):
            process()

    mode = "would upgrade" if args.dry_run else "upgraded"
    print(f"\n{mode} {touched} dump(s); {unchanged} already current; "
          f"{failed} failed; {verified} spot-verified against SQL")
    if non_nominal_shots:
        print(f"shots with previously corrupted (non-nominal) channels: "
              f"{len(non_nominal_shots)} "
              f"[{non_nominal_shots[0]}..{non_nominal_shots[-1]}]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
