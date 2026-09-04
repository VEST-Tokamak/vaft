"""Serial, restartable VEST raw-DAQ exports for an on-disk FileDB."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import IO, Iterable, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from vaft.database import raw as raw_db
from vaft.database.filedb import FileDB
from vaft.omas.vest_upstream import archive_raw_source, sha256_file, write_manifest


class RedumpAlreadyRunningError(RuntimeError):
    """Raised when another serial raw-redump process holds the FileDB lock."""


DEFAULT_FIRST_RAW_SHOT = 29350
DEFAULT_LAST_RAW_SHOT = 48823


def _lock_file_nonblocking(lock_file: IO[str]) -> None:
    """Acquire one byte of ``lock_file`` exclusively without waiting."""
    lock_file.seek(0)
    if os.name == "nt":
        # ``msvcrt.locking`` locks bytes from the current file position and an
        # empty file cannot be locked.  Keep a byte present before acquiring;
        # the PID written by the caller replaces it once the lock is held.
        if not lock_file.read(1):
            lock_file.seek(0)
            lock_file.write("\0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(lock_file: IO[str]) -> None:
    """Release the platform-specific lock held by ``lock_file``."""
    if os.name == "nt":
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _existing_dump(path: Path, shot: int) -> dict | None:
    """Return a valid raw-dump payload for ``shot``, otherwise ``None``."""
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("shot")) == int(shot) and isinstance(payload.get("fields"), dict):
            return payload
        return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _manifest_for_existing_dump(shot: int, output: Path, payload: dict) -> dict:
    """Create the missing sidecar without re-contacting SQL."""
    field_codes = sorted(int(code) for code in payload["fields"])
    return {
        "schema_version": 1,
        "stage": "raw",
        "shot": int(shot),
        "status": "success",
        "source": {"kind": "existing-filedb", "name": output.name},
        "inventory": {"field_count": len(field_codes), "field_codes": field_codes},
        "output": {"name": output.name, "sha256": sha256_file(output)},
    }


def _selected_shots(
    shots: list[int] | None,
    shot_range: list[int] | None,
) -> list[int]:
    """Return the user-requested sparse list or inclusive raw-shot range."""
    if shots:
        return list(dict.fromkeys(shots))
    first, last = shot_range or (DEFAULT_FIRST_RAW_SHOT, DEFAULT_LAST_RAW_SHOT)
    if first <= 0 or last <= 0 or first > last:
        raise ValueError("--shot-range must contain two positive shots in ascending order")
    return list(range(first, last + 1))


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    """Hold a non-blocking process lock for one FileDB raw-export run."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".redump.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            _lock_file_nonblocking(lock_file)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise RedumpAlreadyRunningError(
                f"Another raw redump is running for {root} ({lock_path})."
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            _unlock_file(lock_file)


def _export_one(
    *,
    shot: int,
    output: Path,
    manifest_path: Path,
    source: str | None,
    sql_retries: int,
) -> None:
    """Export atomically, leaving the previous good dump untouched on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".redump-{shot}-", dir=output.parent) as tmpdir:
        temporary_dir = Path(tmpdir)
        temporary_output = temporary_dir / output.name
        temporary_manifest = temporary_dir / manifest_path.name
        manifest = archive_raw_source(
            shot=shot,
            source=source,
            output=temporary_output,
            max_retries=sql_retries,
        )
        # The archive is written under a temporary name, but its stable manifest
        # must identify the final FileDB product.
        manifest["output"]["name"] = output.name
        write_manifest(manifest, temporary_manifest)
        os.replace(temporary_output, output)
        os.replace(temporary_manifest, manifest_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filedb-root", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--shots",
        nargs="+",
        type=int,
        help="Sparse list of shots to export.",
    )
    selection.add_argument(
        "--shot-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help=(
            "Inclusive shot range to export. Defaults to "
            f"{DEFAULT_FIRST_RAW_SHOT} {DEFAULT_LAST_RAW_SHOT}."
        ),
    )
    parser.add_argument(
        "--source-template",
        help="Optional offline archive path, optionally containing {shot}.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Whole-shot attempts before continuing to the next shot (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="Seconds to wait after a failed shot attempt (default: 30).",
    )
    parser.add_argument(
        "--inter-shot-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between completed shot exports (default: 2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even when a valid dump already exists.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1")
    if args.retry_delay < 0 or args.inter_shot_delay < 0:
        raise ValueError("delays must be non-negative")

    filedb = FileDB(args.filedb_root)
    raw_root = filedb.root / "raw"

    # The exporter reads fields serially. Limiting its pool to one connection
    # prevents the connector from reserving several server connections at once.
    raw_db.POOL_SIZE = 1
    raw_db.DB_POOL = None

    try:
        with _exclusive_lock(raw_root):
            shots = _selected_shots(
                args.shots,
                args.shot_range,
            )
            print(f"Selected {len(shots)} shot(s): {shots[0]}–{shots[-1]}")
            failures: list[int] = []
            for index, shot in enumerate(shots):
                output_dir = filedb.raw(shot)
                output = output_dir / f"vest_{shot}_daq_raw.json.gz"
                manifest = output_dir / f"vest_{shot}_daq_manifest.json"
                existing_payload = _existing_dump(output, shot)
                if not args.force and existing_payload is not None:
                    if not manifest.is_file():
                        write_manifest(
                            _manifest_for_existing_dump(shot, output, existing_payload),
                            manifest,
                        )
                        print(f"shot {shot}: created missing manifest from existing dump")
                    else:
                        print(f"shot {shot}: skipped (valid dump exists)")
                    continue

                source = (
                    args.source_template.format(shot=shot)
                    if args.source_template
                    else None
                )
                for attempt in range(1, args.attempts + 1):
                    try:
                        print(f"shot {shot}: export attempt {attempt}/{args.attempts}")
                        _export_one(
                            shot=shot,
                            output=output,
                            manifest_path=manifest,
                            source=source,
                            sql_retries=1,
                        )
                    except Exception as error:  # report and continue with the batch
                        if attempt == args.attempts:
                            failures.append(shot)
                            print(f"shot {shot}: failed: {error}", file=sys.stderr)
                        else:
                            print(
                                f"shot {shot}: failed ({error}); retrying in "
                                f"{args.retry_delay:g}s",
                                file=sys.stderr,
                            )
                            time.sleep(args.retry_delay)
                    else:
                        print(f"shot {shot}: saved {output}")
                        break

                if index < len(shots) - 1 and args.inter_shot_delay:
                    time.sleep(args.inter_shot_delay)
    except RedumpAlreadyRunningError as error:
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Raw redump interrupted.", file=sys.stderr)
        return 130

    if failures:
        print(f"Failed shots: {', '.join(str(shot) for shot in failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
