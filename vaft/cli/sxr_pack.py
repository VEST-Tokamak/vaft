"""Pack a soft X-ray archive's digitizer CSVs into lossless HDF5 containers.

Walks ``{root}/{shot}/digitizer_{daq}_{shot}.csv`` -- the layout of
``legacy/soft_x_rays`` in a FileDB -- and writes ``digitizer_{daq}_{shot}.h5``
beside each CSV (see :mod:`vaft.database.digitizer_hdf5`). A container is only
renamed into place after it regenerates the CSV byte for byte, so the CSV can
be removed with ``--delete-csv``; without it both are kept and the mapper reads
the container. A CSV that does not follow the writer's exact text format, or
does not parse (ragged rows), is logged and left alone.

Each shot's ``provenance.json`` gains a ``containers`` entry recording the
source CSV's size and sha256, which is what still identifies the original once
the CSV is gone.

Restartable: a shot whose CSV already has a container is re-verified against
that CSV (not re-packed) and, with ``--delete-csv``, only then removed. Every
outcome is appended to ``--log`` (JSON lines).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator

from vaft.cli.raw_redump import _lock_file_nonblocking, _unlock_file
from vaft.database import digitizer_hdf5 as dh

PROVENANCE_NAME = "provenance.json"
LOCK_NAME = ".sxr_pack.lock"
_CSV = re.compile(r"^digitizer_(?P<daq>\d+)_(?P<shot>\d+)\.csv$")


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    with (root / LOCK_NAME).open("a+", encoding="utf-8") as handle:
        try:
            _lock_file_nonblocking(handle)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise SystemExit(f"Another sxr-pack run holds {root / LOCK_NAME}") from error
        try:
            yield
        finally:
            _unlock_file(handle)


def _iter_shot_dirs(root: Path, first: int | None, last: int | None) -> Iterable[Path]:
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        shot = int(entry.name)
        if (first is not None and shot < first) or (last is not None and shot > last):
            continue
        yield entry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_provenance(shot_dir: Path, entries: dict[str, dict[str, Any]]) -> None:
    path = shot_dir / PROVENANCE_NAME
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.setdefault("containers", {}).update(entries)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object; not rewriting it")
    temp = path.with_name(f".{PROVENANCE_NAME}.partial")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dh.fsync_path(temp)
    os.replace(temp, path)
    dh.fsync_path(shot_dir)


def _pack_shot(shot_dir: Path, delete_csv: bool, dry_run: bool) -> list[dict[str, Any]]:
    """Pack every digitizer CSV of one shot. Runs in a worker process."""
    outcomes: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    for csv in sorted(shot_dir.iterdir()):
        match = _CSV.match(csv.name)
        if match is None or int(match["shot"]) != int(shot_dir.name):
            continue
        target = dh.container_path(csv)
        record: dict[str, Any] = {"shot": int(shot_dir.name), "csv": csv.name}
        try:
            if dry_run:
                record["status"] = "would_verify" if target.exists() else "would_pack"
                outcomes.append(record)
                continue
            if target.exists():
                # An earlier run packed it (and may have died before deleting
                # the CSV): prove the container against this CSV, never repack.
                digest = dh.verify_container(target, expected_sha256=_sha256(csv))
                record["status"] = "verified_existing"
                size = csv.stat().st_size
            else:
                result = dh.pack_digitizer_csv(csv, target)
                digest, size = result.source_sha256, result.source_size
                record.update(status="packed", shape=list(result.shape),
                              ratio=round(result.ratio, 4))
            record.update(csv_size=size, h5_size=target.stat().st_size, sha256=digest)
            provenance[csv.name] = {
                "container": target.name,
                "source_size": size,
                "source_sha256": digest,
                "schema": dh.SCHEMA,
                "schema_version": dh.SCHEMA_VERSION,
                "packed_at": datetime.now(timezone.utc).isoformat(),
            }
            if delete_csv:
                # Provenance first: once the CSV is gone its sha256 must
                # already be on disk somewhere other than inside the container.
                _record_provenance(shot_dir, {csv.name: provenance.pop(csv.name)})
                csv.unlink()
                dh.fsync_path(shot_dir)
                record["csv_deleted"] = True
        except Exception as exc:  # one bad record must not stop the archive
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        outcomes.append(record)
    if provenance and not dry_run:
        try:
            _record_provenance(shot_dir, provenance)
        except Exception as exc:
            # The containers are in place and verified; only the sha256 record
            # is missing. No CSV was deleted on this path (without
            # --delete-csv), so a rerun re-verifies and records them.
            outcomes.append({"shot": int(shot_dir.name), "csv": PROVENANCE_NAME,
                             "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return outcomes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaft sxr-pack", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="The soft X-ray tree: {root}/{shot}/digitizer_*.csv "
                             "(e.g. $VAFT_FILEDB_DIR/legacy/soft_x_rays).")
    parser.add_argument("--first-shot", type=int, default=None)
    parser.add_argument("--last-shot", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes (one shot each).")
    parser.add_argument("--delete-csv", action="store_true",
                        help="Remove each CSV once its container regenerates it byte for byte.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    parser.add_argument("--log", type=Path, default=None,
                        help="JSON-lines outcome log (default: {root}/_sxr_pack_log.jsonl).")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    root = args.root.expanduser()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    log_path = args.log or root / "_sxr_pack_log.jsonl"
    shot_dirs = list(_iter_shot_dirs(root, args.first_shot, args.last_shot))
    counts: dict[str, int] = {}
    before = after = 0

    with ExitStack() as stack:
        # A dry run leaves the archive exactly as it found it: no lock, no log.
        log = None
        if not args.dry_run:
            stack.enter_context(_exclusive_lock(root))
            log = stack.enter_context(log_path.open("a", encoding="utf-8"))

        def consume(outcomes: list[dict[str, Any]]) -> None:
            nonlocal before, after
            for record in outcomes:
                counts[record["status"]] = counts.get(record["status"], 0) + 1
                before += record.get("csv_size", 0)
                after += record.get("h5_size", 0)
                if log is not None:
                    log.write(json.dumps(record) + "\n")
                if record["status"] == "failed":
                    print(f"FAILED {record['shot']} {record['csv']}: {record['error']}",
                          file=sys.stderr)
            if log is not None:
                log.flush()

        if args.jobs <= 1:
            for shot_dir in shot_dirs:
                consume(_pack_shot(shot_dir, args.delete_csv, args.dry_run))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(_pack_shot, d, args.delete_csv, args.dry_run)
                           for d in shot_dirs]
                for shot_dir, future in zip(shot_dirs, futures):
                    try:
                        outcomes = future.result()
                    except Exception as exc:  # e.g. a worker killed for memory
                        outcomes = [{"shot": int(shot_dir.name), "csv": "*", "status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}"}]
                    consume(outcomes)

    summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"{len(shot_dirs)} shot directories: {summary or 'nothing to do'}")
    if before:
        print(f"CSV {before / 1e9:.2f} GB -> HDF5 {after / 1e9:.2f} GB ({after / before:.1%})")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
