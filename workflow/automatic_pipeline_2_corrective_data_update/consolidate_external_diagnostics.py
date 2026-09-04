#!/usr/bin/env python3

"""Move inventoried soft X-ray and camera data into the FileDB, reversibly.

Reads the inventory produced by ``inventory_external_diagnostics.py``, derives
a target path for every artifact, and executes the moves. Two destinations:

``{root}/legacy/{diagnostic}/{shot}/``
    The canonical FileDB legacy domain (``vaft.database.filedb.FileDB.legacy``).
    Everything here is laid out exactly as the matching machine mapping expects
    to find it, so ingest needs a data root and nothing else.

``{root}/unmapped/{diagnostic}/{shot}/``
    Real data that no mapping can read yet: arranged frames, vendor ``.mcf``
    containers, the 2013-era CCD export, hard X-ray CSVs. Deliberately outside
    the ``FileDBDomain`` grammar so it cannot be handed to a mapper as a data
    root by accident.

Every operation is appended to a manifest before the next one starts, and
``--revert`` replays that manifest backwards. Moves within one filesystem are
renames, which is why the whole consolidation is fast; anything crossing a
device boundary is copied and verified by hash before the source is released.

Run::

    ./consolidate_external_diagnostics.py --inventory inventory.json --root /path/to/FileDB
    ./consolidate_external_diagnostics.py --inventory inventory.json --root ... --execute
    ./consolidate_external_diagnostics.py --revert move_manifest.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_external_diagnostics import (  # noqa: E402
    JUNK_NAMES,
    MAPPED_DIAGNOSTICS,
    UNMAPPED_DIAGNOSTICS,
    normalised_camera_names,
)

#: Where each diagnostic key lands. ``legacy`` is the canonical FileDB domain
#: that machine mappings resolve; ``unmapped`` is the parking area for data
#: that has no reader. One table so the split is stated once.
DESTINATION_DOMAIN = {
    **{key: "legacy" for key in MAPPED_DIAGNOSTICS},
    "soft_x_rays_geometry": "legacy",
    **{key: "unmapped" for key in UNMAPPED_DIAGNOSTICS},
}

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")

MANIFEST_NAME = "move_manifest.jsonl"
PROVENANCE_NAME = "provenance.json"
FLUCTUATION_INDEX_NAME = "index.json"


class ConsolidationError(RuntimeError):
    """Raised when the plan is unsafe to execute."""


def slugify(value: str) -> str:
    """Reduce a free-text export note to a filesystem-safe suffix."""
    return _SLUG_RE.sub("-", value).strip("-")


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass
class Operation:
    """One planned filesystem change, and everything needed to undo it."""

    kind: str  # move_directory | move_file | rename_members | write_file
    source: str
    target: str
    diagnostic: str
    shot: int | None
    size_bytes: int = 0
    file_count: int = 1
    renames: dict[str, str] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def target_for(entry: dict[str, Any], root: Path) -> Path | None:
    """Return where one inventory entry belongs, or ``None`` to leave it be."""
    diagnostic = entry["diagnostic"]
    if diagnostic is None or entry.get("duplicate_of"):
        return None
    domain = DESTINATION_DOMAIN.get(diagnostic)
    if domain is None:
        return None

    shot = entry["shot"]
    detail = entry.get("detail") or {}

    if diagnostic == "soft_x_rays_geometry":
        return root / "legacy" / "soft_x_rays" / "_geometry"

    if entry["kind"] == "digitizer_csv":
        return root / domain / diagnostic / str(shot) / Path(entry["source"]).name

    if entry["kind"] == "mcf_file":
        return root / domain / diagnostic / f"{shot}.mcf"

    if entry["kind"] == "ccd_directory":
        note = slugify(detail.get("note") or "")
        name = f"{shot}_{note}" if note else str(shot)
        return root / domain / diagnostic / name

    if entry["kind"] == "ccd_file":
        # Loose per-shot artifacts (`shot #5256.avi` and its `_avi.txt`
        # sidecar) belong with that shot's frames. Renamed to drop the
        # free-text `shot #` prefix so the whole tree reads one way.
        suffix = Path(entry["source"]).name.split("#", 1)[-1].lstrip("0123456789 ")
        return root / domain / diagnostic / str(shot) / f"{shot}{suffix}"

    if entry["kind"] == "camera_directory":
        return root / domain / diagnostic / str(shot)

    return None


#: The 2013-era CCD export is a hand-curated archive: the same shot appears at
#: the top level and again inside a per-person working folder, with different
#: contents. Those are disambiguated by their source folder rather than
#: rejected. Everywhere else a collision means two sources disagree about what
#: a shot is, which a human has to settle.
SELF_DISAMBIGUATING = {"camera_ccd_2013"}


def disambiguate(target: Path, source: Path) -> Path:
    """Give a colliding target a suffix drawn from where it came from."""
    parent = slugify(source.parent.name) or "alt"
    return target.with_name(f"{target.name}__{parent}")


def plan_operations(inventory: dict[str, Any], root: Path) -> tuple[list[Operation], list[dict[str, Any]]]:
    """Turn the inventory into an ordered, collision-checked list of moves."""
    operations: list[Operation] = []
    skipped: list[dict[str, Any]] = []
    claimed: dict[str, str] = {}

    for entry in inventory["entries"]:
        target = target_for(entry, root)
        if target is None:
            skipped.append(
                {
                    "source": entry["source"],
                    "reason": (
                        f"duplicate of {entry['duplicate_of']}"
                        if entry.get("duplicate_of")
                        else entry.get("reason") or "unclassified"
                    ),
                }
            )
            continue

        key = str(target)
        if key in claimed and entry["kind"] != "geometry_directory":
            if entry["diagnostic"] in SELF_DISAMBIGUATING:
                target = disambiguate(target, Path(entry["source"]))
                key = str(target)
            if key in claimed:
                raise ConsolidationError(
                    f"Two sources claim {key}:\n  {claimed[key]}\n  {entry['source']}\n"
                    "Resolve by hand before consolidating; refusing to overwrite."
                )
        claimed[key] = entry["source"]

        detail = entry.get("detail") or {}
        if entry["kind"] in {"camera_directory", "ccd_directory", "geometry_directory"}:
            renames: dict[str, str] = {}
            if entry["kind"] == "camera_directory":
                stem, shot = detail.get("stem"), entry["shot"]
                if stem and shot is not None and stem != str(shot):
                    members = _member_names(Path(entry["source"]))
                    renames = {
                        old: new
                        for old, new in normalised_camera_names(stem, shot, members).items()
                        if old != new
                    }
            operations.append(
                Operation(
                    kind="move_directory",
                    source=entry["source"],
                    target=key,
                    diagnostic=entry["diagnostic"],
                    shot=entry["shot"],
                    size_bytes=entry["size_bytes"],
                    file_count=entry["file_count"],
                    renames=renames,
                    detail=detail,
                )
            )
        else:
            operations.append(
                Operation(
                    kind="move_file",
                    source=entry["source"],
                    target=key,
                    diagnostic=entry["diagnostic"],
                    shot=entry["shot"],
                    size_bytes=entry["size_bytes"],
                    detail={**detail, "sha256": entry.get("sha256")},
                )
            )

    # A loose file can be destined for a directory that is itself being moved,
    # so every directory move goes first.
    operations.sort(key=lambda op: 0 if op.kind == "move_directory" else 1)
    return operations, skipped


def inside_git_worktree(path: Path) -> bool:
    """Report whether a path lives inside a git checkout.

    Data that sits in someone's working tree is copied rather than moved, even
    when a rename would be free: moving a tracked file out shows up as a
    deletion in a repository this consolidation has no business editing, and
    an untracked one is usually there because a person put it there.
    """
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return True
    return False


def _member_names(directory: Path) -> list[str]:
    try:
        return sorted(entry.name for entry in os.scandir(directory))
    except OSError:
        return []


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _same_device(source: Path, target_parent: Path) -> bool:
    """Report whether a rename would stay on one filesystem.

    The nearest existing ancestor of the target stands in for the target
    itself, which does not exist yet.
    """
    probe = target_parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return os.stat(source).st_dev == os.stat(probe).st_dev
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Manifest:
    """Append-only record of what was done, flushed after every operation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")

    def record(self, **payload: Any) -> None:
        payload["at"] = datetime.now(timezone.utc).isoformat()
        self._handle.write(json.dumps(payload) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _move_path(
    source: Path,
    target: Path,
    manifest: Manifest,
    *,
    kind: str,
    keep_source: bool = False,
    **extra: Any,
) -> str:
    """Move one path, by rename when possible and by verified copy otherwise.

    ``keep_source`` copies and leaves the original in place; the manifest still
    records the operation so revert removes the copy.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if keep_source:
        if source.is_dir():
            shutil.copytree(source, target)
            checksum = None
        else:
            shutil.copy2(source, target)
            checksum = _sha256(target)
            if checksum != _sha256(source):
                raise ConsolidationError(f"Checksum mismatch after copying {source} -> {target}")
        manifest.record(kind=kind, mode="copy_keep", source=str(source), target=str(target),
                        sha256=checksum, **extra)
        return "copy_keep"
    if _same_device(source, target.parent):
        os.rename(source, target)
        mode = "rename"
        checksum = None
    else:
        if source.is_dir():
            shutil.copytree(source, target)
            checksum = None
        else:
            shutil.copy2(source, target)
            checksum = _sha256(target)
            if checksum != _sha256(source):
                raise ConsolidationError(f"Checksum mismatch after copying {source} -> {target}")
        # Only release the source once the copy is known good.
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
        mode = "copy"
    manifest.record(kind=kind, mode=mode, source=str(source), target=str(target),
                    sha256=checksum, **extra)
    return mode


def _drop_junk(directory: Path, manifest: Manifest) -> list[str]:
    """Remove export-tool residue that should never reach the archive."""
    removed = []
    for name in _member_names(directory):
        if name.startswith("._") or name in JUNK_NAMES:
            try:
                (directory / name).unlink()
            except OSError:
                continue
            removed.append(name)
    if removed:
        manifest.record(kind="drop_junk", source=str(directory), target=str(directory),
                        removed=removed)
    return removed


def _write_provenance(target: Path, operation: Operation, manifest: Manifest) -> None:
    """Record where this shot came from and what was changed on the way in."""
    payload = {
        "diagnostic": operation.diagnostic,
        "shot": operation.shot,
        "original_path": operation.source,
        "original_name": Path(operation.source).name,
        "stripped_suffix": (operation.detail or {}).get("suffix") or None,
        "renamed_members": len(operation.renames),
        "consolidated_at": datetime.now(timezone.utc).isoformat(),
        **{
            key: value
            for key, value in (operation.detail or {}).items()
            if key in {"family", "frame_count", "header", "note", "daq_label", "variant"}
        },
    }
    path = target / PROVENANCE_NAME if target.is_dir() else target.parent / PROVENANCE_NAME
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    if existing and existing.get("shot") == payload["shot"]:
        sources = existing.setdefault("additional_sources", [])
        sources.append(payload["original_path"])
        payload = existing
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest.record(kind="write_file", source="", target=str(path))


def execute(operations: Sequence[Operation], manifest: Manifest) -> dict[str, int]:
    """Run the plan, recording each step before moving on to the next."""
    counts: dict[str, int] = defaultdict(int)
    for operation in operations:
        source = Path(operation.source)
        target = Path(operation.target)
        if not source.exists():
            manifest.record(kind="skip", source=str(source), target=str(target),
                            reason="source disappeared before the move")
            counts["missing"] += 1
            continue
        if target.exists() and operation.kind == "move_file":
            manifest.record(kind="skip", source=str(source), target=str(target),
                            reason="target already present")
            counts["already_present"] += 1
            continue

        if operation.kind == "move_directory":
            _drop_junk(source, manifest)
            mode = _move_path(source, target, manifest, kind="move_directory",
                              keep_source=inside_git_worktree(source),
                              diagnostic=operation.diagnostic, shot=operation.shot)
            if operation.renames:
                applied = {}
                for old, new in operation.renames.items():
                    old_path, new_path = target / old, target / new
                    if old_path.exists() and not new_path.exists():
                        os.rename(old_path, new_path)
                        applied[old] = new
                manifest.record(kind="rename_members", source=str(target), target=str(target),
                                renames=applied)
            _write_provenance(target, operation, manifest)
        else:
            mode = _move_path(source, target, manifest, kind="move_file",
                              keep_source=inside_git_worktree(source),
                              diagnostic=operation.diagnostic, shot=operation.shot)
            _write_provenance(target, operation, manifest)

        counts[mode] += 1
        counts["bytes"] += operation.size_bytes
    return dict(counts)


def write_fluctuation_index(root: Path, operations: Sequence[Operation]) -> Path | None:
    """Write the index of camera shots reserved for the fluctuation routine.

    Issue #161 needs to know which shots are available at fluctuation-grade
    frame rates without re-deriving it from 200 GB of headers, and needs the
    guarantee that routine ingest never consumed them.
    """
    reserved = [op for op in operations if op.diagnostic == "camera_visible_fluctuation"]
    if not reserved:
        return None
    directory = root / "legacy" / "camera_visible_fluctuation"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Camera acquisitions at fluctuation-grade frame rates, reserved for the "
            "pending FAST-camera fluctuation routine (issue #161). Not consumed by "
            "routine camera_visible IDS generation."
        ),
        "shots": sorted(
            (
                {
                    "shot": op.shot,
                    "frame_count": (op.detail or {}).get("frame_count"),
                    "frame_rate_hz": ((op.detail or {}).get("header") or {}).get("frame_rate_hz"),
                    "frame_size": ((op.detail or {}).get("header") or {}).get("frame_size"),
                    "original_name": Path(op.source).name,
                }
                for op in reserved
            ),
            key=lambda item: item["shot"] or 0,
        ),
    }
    path = directory / FLUCTUATION_INDEX_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Revert
# --------------------------------------------------------------------------


def revert(manifest_path: Path, *, execute_changes: bool) -> int:
    """Undo a consolidation by replaying its manifest backwards."""
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    undone = 0
    for record in reversed(records):
        kind = record["kind"]
        if kind == "rename_members":
            directory = Path(record["target"])
            for old, new in record.get("renames", {}).items():
                if not execute_changes:
                    print(f"  would rename {directory / new} -> {old}")
                    continue
                if (directory / new).exists():
                    os.rename(directory / new, directory / old)
            undone += 1
        elif kind in {"move_directory", "move_file"}:
            source, target = Path(record["source"]), Path(record["target"])
            if record.get("mode") == "copy_keep":
                # The original was never disturbed; undoing means dropping the copy.
                if not execute_changes:
                    print(f"  would delete copy {target}")
                elif target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                undone += 1
                continue
            if not execute_changes:
                print(f"  would move {target} -> {source}")
                undone += 1
                continue
            if not target.exists():
                continue
            source.parent.mkdir(parents=True, exist_ok=True)
            os.rename(target, source) if _same_device(target, source.parent) else shutil.move(
                str(target), str(source)
            )
            undone += 1
        elif kind == "write_file":
            path = Path(record["target"])
            if not execute_changes:
                print(f"  would delete {path}")
            elif path.exists():
                path.unlink()
            undone += 1
        elif kind == "drop_junk":
            # Export residue (Thumbs.db, AppleDouble sidecars) is not restored:
            # it carries no information and is regenerated by the tools that
            # made it.
            continue
    return undone


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _gib(value: int) -> float:
    return value / float(1 << 30)


def summarise(operations: Sequence[Operation], skipped: Sequence[dict[str, Any]]) -> str:
    grouped: dict[str, list[Operation]] = defaultdict(list)
    for operation in operations:
        grouped[operation.diagnostic].append(operation)
    lines = ["Planned moves", ""]
    for diagnostic in sorted(grouped):
        group = grouped[diagnostic]
        shots = {op.shot for op in group if op.shot is not None}
        lines.append(
            f"  {DESTINATION_DOMAIN.get(diagnostic, '?'):<9}/{diagnostic:<28} "
            f"{len(group):>5} entries  {len(shots):>5} shots  "
            f"{_gib(sum(op.size_bytes for op in group)):>8.2f} GiB"
        )
    renaming = sum(1 for op in operations if op.renames)
    lines += [
        "",
        f"  {renaming} directories need member renames to drop an acquisition suffix",
        f"  {len(skipped)} entries skipped (duplicates and unclassified)",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", help="Inventory JSON from inventory_external_diagnostics.py")
    parser.add_argument("--root", help="FileDB root to consolidate into")
    parser.add_argument("--manifest", help=f"Manifest path (default: <root>/{MANIFEST_NAME})")
    parser.add_argument(
        "--only",
        action="append",
        help="Restrict to these diagnostic keys (repeatable)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move data. Without this the plan is only printed.",
    )
    parser.add_argument("--revert", metavar="MANIFEST", help="Undo a previous consolidation")
    args = parser.parse_args(argv)

    if args.revert:
        undone = revert(Path(args.revert).expanduser(), execute_changes=args.execute)
        verb = "Reverted" if args.execute else "Would revert"
        print(f"{verb} {undone} operations from {args.revert}")
        if not args.execute:
            print("Dry run; pass --execute to apply.")
        return 0

    if not args.inventory or not args.root:
        parser.error("--inventory and --root are required unless --revert is given")

    inventory = json.loads(Path(args.inventory).expanduser().read_text(encoding="utf-8"))
    root = Path(args.root).expanduser()
    operations, skipped = plan_operations(inventory, root)
    if args.only:
        wanted = set(args.only)
        operations = [op for op in operations if op.diagnostic in wanted]

    print(summarise(operations, skipped))

    if not args.execute:
        print("\nDry run; pass --execute to apply.")
        return 0

    manifest_path = Path(args.manifest).expanduser() if args.manifest else root / MANIFEST_NAME
    with Manifest(manifest_path) as manifest:
        manifest.record(kind="begin", source=args.inventory, target=str(root),
                        operations=len(operations))
        counts = execute(operations, manifest)
        index = write_fluctuation_index(root, operations)
        if index:
            manifest.record(kind="write_file", source="", target=str(index))
        manifest.record(kind="end", source=args.inventory, target=str(root), counts=counts)

    print(
        f"\nMoved {counts.get('rename', 0)} by rename, {counts.get('copy', 0)} by copy, "
        f"{counts.get('copy_keep', 0)} copied out of git checkouts (originals left in place), "
        f"{_gib(counts.get('bytes', 0)):.2f} GiB total."
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
