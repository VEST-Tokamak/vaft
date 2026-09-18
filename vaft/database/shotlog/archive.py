"""Keep the ShotLog in FileDB: the workbooks themselves and what was read from them.

Layout (#995)::

    legacy/shotlog/input/{YYYY}/ShotLog_YYYY_MM #a-b.xlsx    the workbooks, byte-identical
    legacy/shotlog/input/superseded/{name}.sha-{8}.xlsx      earlier bytes of a workbook that changed
    legacy/shotlog/input/manifest.json                       sha256, origin and discovery verdict per file
    legacy/shotlog/output/experiment-days/{YYYY}/*.yaml      one session document per worksheet
    legacy/shotlog/output/batch-manifest.json                what the last extraction read and skipped
    legacy/shotlog/{shot}/metadata/shotlog.json              one record per shot (``records.py``)

The workbooks are copied, never moved: the operators' folder stays the working
copy. A workbook whose bytes changed since the last archive (the month still in
progress) replaces the archived one, and the earlier bytes are kept under
``superseded/`` rather than lost -- a record built from them names their sha256.

Every write goes to a temporary file in the destination directory and is
renamed into place, and a file whose content is unchanged is not rewritten, so
re-running an extraction over an unchanged archive touches nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
import unicodedata

import yaml

from vaft.database.filedb import FileDB

from .batch import SourceWorkbook, discover_sources
from .converter import sha256_file, session_path

TREE = "shotlog"
RECORD_NAME = "shotlog.json"
INPUT_MANIFEST_NAME = "manifest.json"
BATCH_MANIFEST_NAME = "batch-manifest.json"
SUPERSEDED_DIR = "superseded"


def input_dir(filedb: FileDB) -> Path:
    return filedb.legacy(TREE, None, artifact="input")


def output_dir(filedb: FileDB) -> Path:
    return filedb.legacy(TREE, None, artifact="output")


def record_path(filedb: FileDB, shot: int) -> Path:
    return filedb.legacy(TREE, shot, artifact="metadata") / RECORD_NAME


def _atomic_write(path: Path, data: bytes) -> bool:
    """Write ``data`` unless ``path`` already holds exactly it. True if written."""
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return True


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _verified_copy(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    copied = sha256_file(temporary)
    if copied != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise OSError(f"copy of {source} does not match its sha256 ({copied} != {expected_sha256})")
    os.replace(temporary, destination)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def archive_workbooks(source_dir: str | Path, filedb: FileDB, *, dry_run: bool = False) -> dict[str, Any]:
    """Copy the chosen monthly workbooks from ``source_dir`` into FileDB.

    Returns a summary; the per-file state lives in ``input/manifest.json``,
    which is merged, not replaced, so archiving from two machines accumulates.
    """
    source_dir = Path(source_dir).expanduser()
    root = input_dir(filedb)
    manifest_path = root / INPUT_MANIFEST_NAME
    manifest = _load_json(manifest_path, {"manifest_version": 1, "workbooks": {}, "history": []})
    discovery = discover_sources(source_dir)
    now = datetime.now(timezone.utc).isoformat()
    summary = {"copied": [], "unchanged": [], "replaced": [], "excluded": len(discovery.excluded)}

    for workbook in discovery.included:
        # NFC so the archive reads the same on the Linux server as on macOS.
        relative = f"{workbook.year:04d}/{unicodedata.normalize('NFC', workbook.path.name)}"
        destination = root / relative
        digest = sha256_file(workbook.path)
        previous = manifest["workbooks"].get(relative)
        if destination.exists() and previous and previous.get("sha256") == digest:
            summary["unchanged"].append(relative)
            continue
        if dry_run:
            summary["replaced" if destination.exists() else "copied"].append(relative)
            continue
        if destination.exists():
            old = sha256_file(destination)
            if old == digest:
                summary["unchanged"].append(relative)
            else:
                kept = root / SUPERSEDED_DIR / f"{destination.stem}.sha-{old[:8]}{destination.suffix}"
                kept.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, kept)
                manifest["history"].append({
                    "workbook": relative, "superseded_sha256": old,
                    "kept_as": kept.relative_to(root).as_posix(), "at": now,
                })
                summary["replaced"].append(relative)
        if not destination.exists():
            _verified_copy(workbook.path, destination, digest)
            if relative not in summary["replaced"]:
                summary["copied"].append(relative)
        manifest["workbooks"][relative] = {
            "sha256": digest,
            "size": workbook.path.stat().st_size,
            "original_name": workbook.path.name,
            "original_path": str(workbook.path),
            "year": workbook.year,
            "month": workbook.month,
            "first_shot": workbook.first_shot,
            "last_shot": workbook.last_shot,
            "archived_at": now,
        }

    manifest["excluded"] = discovery.excluded
    manifest["source_dir"] = str(source_dir)
    manifest["updated_at"] = now
    if not dry_run:
        _atomic_write(manifest_path, _json_bytes(manifest))
    return summary


def write_extraction(
    sessions: Iterable[tuple[SourceWorkbook, dict[str, Any]]],
    records: dict[int, dict[str, Any]],
    batch_manifest: dict[str, Any],
    filedb: FileDB,
) -> dict[str, int]:
    """Write session documents, per-shot records and the batch manifest."""
    out = output_dir(filedb)
    counts = {"sessions_written": 0, "records_written": 0, "records_unchanged": 0}
    for _, dataset in sessions:
        text = yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False, width=120)
        if _atomic_write(session_path(out, dataset), text.encode("utf-8")):
            counts["sessions_written"] += 1
    for shot, record in records.items():
        if _atomic_write(record_path(filedb, shot), _json_bytes(record)):
            counts["records_written"] += 1
        else:
            counts["records_unchanged"] += 1
    # A record whose shot no longer appears (a corrected shot number, a sheet
    # reclassified) is reported, never deleted: removing archive content is a
    # decision for a person, not for a re-run.
    stale = sorted(set(recorded_shots(filedb)) - set(records))
    counts["records_stale"] = len(stale)
    stamped = {
        **batch_manifest,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "stale_records": stale,
    }
    _atomic_write(out / BATCH_MANIFEST_NAME, _json_bytes(stamped))
    return counts


def load_record(shot: int, data_root: str | Path) -> dict[str, Any]:
    """Read ``{data_root}/{shot}/metadata/shotlog.json``.

    ``data_root`` is ``legacy/shotlog`` -- the same per-diagnostic root every
    external-diagnostic mapping takes. Raises ``FileNotFoundError`` for a shot
    the ShotLog does not mention.
    """
    path = Path(data_root) / str(int(shot)) / "metadata" / RECORD_NAME
    if not path.exists():
        raise FileNotFoundError(f"No ShotLog record for shot {shot}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_shots(filedb: FileDB) -> list[int]:
    """Shots that have a record under ``legacy/shotlog``, without reading them."""
    root = input_dir(filedb).parent
    if not root.is_dir():
        return []
    return sorted(
        int(child.name) for child in root.iterdir()
        if child.name.isdigit() and (child / "metadata" / RECORD_NAME).exists()
    )


def load_records(filedb: FileDB) -> dict[int, dict[str, Any]]:
    """Every per-shot record under ``legacy/shotlog``, keyed by shot."""
    root = input_dir(filedb).parent
    records: dict[int, dict[str, Any]] = {}
    if not root.is_dir():
        return records
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            path = child / "metadata" / RECORD_NAME
            if path.exists():
                records[int(child.name)] = json.loads(path.read_text(encoding="utf-8"))
    return dict(sorted(records.items()))


__all__ = [
    "archive_workbooks",
    "load_records",
    "input_dir",
    "load_record",
    "output_dir",
    "record_path",
    "write_extraction",
]
