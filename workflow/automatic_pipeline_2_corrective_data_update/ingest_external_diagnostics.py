#!/usr/bin/env python3

"""Generate soft X-ray, camera and ShotLog IDSs from the consolidated FileDB archive.

Once ``consolidate_external_diagnostics.py`` has laid the raw data out under
``{root}/legacy/{diagnostic}/{shot}/``, both machine mappings can read a whole
archive from a data root alone. This script walks that archive shot by shot,
builds one ODS per shot, and writes it with the same product/manifest pair
every pipeline stage produces.

Two deliberate exclusions:

``legacy/camera_visible_fluctuation/``
    Reserved for the pending FAST-camera fluctuation routine (issue #161).
    Routine ingest never touches it, so those acquisitions stay pristine as
    fixture input. ``--include-fluctuation`` overrides that for a deliberate
    one-off.

``unmapped/``
    Arranged frames, vendor ``.mcf`` containers, the 2013-era CCD export and
    hard X-ray CSVs. No mapping can read any of it yet.

Products land in the canonical ``omas/`` domain, resolved through
``FileDB.omas_product`` / ``FileDB.omas_manifest``. Each tree is a real
``OMASStage`` with a ``STAGE_REPLICATION`` entry, so these diagnostics publish
on the same contract as every other stage (#599). They used to sit in an
``ods/`` tree outside the ``FileDBDomain`` grammar, because no legal canonical
path existed for them yet.

Run::

    ./ingest_external_diagnostics.py --root /path/to/FileDB --diagnostic soft_x_rays
    ./ingest_external_diagnostics.py --root /path/to/FileDB --limit 5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Iterable, Sequence

LOGGER = logging.getLogger("ingest_external_diagnostics")

#: Which mapping reads which consolidated tree. The fluctuation tree uses the
#: same mapping as routine camera data -- it is held back by policy, not by
#: any technical difference.
DIAGNOSTIC_TREES = {
    "soft_x_rays": "soft_x_rays",
    "camera_visible": "camera_visible",
    "camera_visible_fluctuation": "camera_visible",
    # Per-shot ShotLog records written by `python -m vaft.cli shotlog extract`
    # (#995); the stage is `shotlog`, the IDS it publishes is pulse_schedule.
    "shotlog": "pulse_schedule",
}

RESERVED_TREES = ("camera_visible_fluctuation",)

REGISTRY_NAME = "ingested_shots.json"
MANIFEST_NAME = "manifest.json"

#: The product container is no longer chosen here. Each of these stages
#: declares it in `vaft.database.filedb.OMAS_PRODUCT_SUFFIXES`, and
#: `FileDB.omas_product` is what both this script and `replicate_stage` use to
#: name a product -- so a container picked at the command line could only
#: produce a file replication would fail to find.


def _filedb(root: Path):
    from vaft.database.filedb import FileDB

    return FileDB(root)


def _product_path(root: Path, tree: str, shot: int) -> Path:
    """Where one shot's product for ``tree`` goes, name and container included.

    Resolved rather than composed: ``replicate_stage`` locates a product with
    the same call, so a filename invented here would be a product replication
    could not find.
    """
    return _filedb(root).omas_product(tree, shot=shot)


def _stage_directory(root: Path, tree: str) -> Path:
    """Stage-wide scratch that is not shot-scoped -- the ingest registry.

    ``FileDB.omas`` is shot-scoped for every non-static stage, so the registry
    sits beside the shots rather than inside one. Derived from the resolver's
    own output so it moves with the grammar instead of being rebuilt by hand.
    """
    return _filedb(root).omas(tree, shot=1).parent


def _mapper(mapping_name: str) -> Callable[..., None]:
    """Import the machine mapping lazily, so ``--help`` needs no vaft."""
    import importlib

    module = importlib.import_module(f"vaft.machine_mapping.{mapping_name}")
    return getattr(module, mapping_name)


def discover_shots(root: Path, tree: str) -> list[int]:
    """Return the shots present in one consolidated tree, in order."""
    directory = root / "legacy" / tree
    if not directory.is_dir():
        return []
    shots = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and child.name.isdigit():
            shots.append(int(child.name))
    return sorted(shots)


def _channel_count(ods: Any, tree: str) -> int | None:
    """Report how much of a shot actually made it into the IDS.

    Coverage varies for real reasons and is not an error: most soft X-ray
    shots were recorded on one digitizer rather than two, so they yield half
    the channels. Recording the number keeps that visible in the manifest
    instead of turning it into a silent surprise later.
    """
    try:
        if tree.startswith("camera"):
            return len(ods["camera_visible.channel.0.detector.0.frame"])
        if tree == "shotlog":
            return len(ods["pulse_schedule.event"])
        return len(ods["soft_x_rays.channel"])
    except (KeyError, IndexError, ValueError):
        return None


#: Storage encoding for camera products. A FAST-camera frame is 8-bit
#: grayscale, but OMAS stores every ``INT_2D`` node as platform ``int64``, so
#: an unnarrowed product spends eight bytes on a value that never exceeds 255.
#: Narrowing to ``int32`` at the storage boundary and letting HDF5 gzip the
#: result shrinks a product by roughly an order of magnitude without altering
#: a single pixel or time. Only camera trees get this: applying it blindly
#: would re-type integer nodes in soft X-ray and magnetics products too.
CAMERA_STORAGE_WIDTH = "int32"
CAMERA_COMPRESSION = "gzip"


def _storage(tree: str) -> dict[str, Any] | None:
    """Describe how a product is encoded, or None when nothing is claimed.

    Camera products are also narrowed to ``int32``, because OMAS stores every
    ``INT_2D`` node as platform ``int64`` and a frame is 8-bit. Soft X-ray
    products carry floats, which have no width to reclaim, but they compress
    the same way: measured 154 MiB to 69 MiB on a real product.
    """
    if not tree.startswith("camera") and tree != "soft_x_rays":
        return None
    width = CAMERA_STORAGE_WIDTH if tree.startswith("camera") else None
    return {"width": width, "compression": CAMERA_COMPRESSION}


def build_shot(root: Path, tree: str, shot: int) -> tuple[Any, dict[str, Any]]:
    """Build one shot's ODS from the consolidated archive.

    Camera ODSs come back already narrowed to ``int32``, so the caller must
    write them straight out: re-enabling the OMAS consistency check would
    upcast every frame back to ``int64``.
    """
    from omas import ODS

    mapping_name = DIAGNOSTIC_TREES[tree]
    data_root = root / "legacy" / tree
    ods = ODS(consistency_check=True)
    _mapper(mapping_name)(ods, shot, data_root=data_root)

    provenance = {}
    provenance_path = data_root / str(shot) / "provenance.json"
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
    record_path = data_root / str(shot) / "metadata" / "shotlog.json"
    if tree == "shotlog" and record_path.exists():
        # The record is this product's whole input; pin its bytes and origin.
        import hashlib

        payload = record_path.read_bytes()
        record = json.loads(payload)
        provenance = {
            "record": str(record_path.relative_to(root)),
            "record_sha256": hashlib.sha256(payload).hexdigest(),
            "workbook": record.get("source", {}),
        }

    manifest = {
        "schema_version": 1,
        "stage": tree,
        "mapping": f"vaft.machine_mapping.{mapping_name}.{mapping_name}",
        "shot": shot,
        "status": "success",
        "data_root": str(data_root),
        "measured": {"count": _channel_count(ods, tree)},
        "provenance": provenance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    storage = _storage(tree)
    if storage is not None:
        if storage["width"] is not None:
            from vaft.machine_mapping.camera_visible import narrow_image_storage

            # Last thing before the manifest is sealed, and nothing but the
            # write may follow: narrowing leaves the consistency check off on
            # purpose.
            narrow_image_storage(ods)
        manifest["storage"] = storage
    return ods, manifest


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Registry at %s is unreadable; starting a fresh one", path)
        return {}


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    """Merge this run's records into whatever is on disk, then replace it.

    Two runs over different trees are a normal thing to want -- soft X-ray
    ingest takes hours, and there is no reason to wait for it before starting
    the camera. Rewriting a registry held in memory since startup would make
    the second run erase the first one's records, so the file is re-read and
    merged immediately before every write. The window between that read and
    the rename is small but not zero; two runs over the *same* tree would
    still race, and there is no reason to do that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = load_registry(path)
    merged.update(registry)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _unavailable_errors() -> tuple[type[BaseException], ...]:
    """Errors that mean "this shot has nothing to map", not "the run broke".

    Only the ShotLog raises these routinely: it has a record for every shot
    since 2013, but a trigger card only from 2023.
    """
    from vaft.machine_mapping.pulse_schedule import PulseScheduleUnavailableError

    return (PulseScheduleUnavailableError,)


def ingest(
    root: Path,
    trees: Sequence[str],
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    shots: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build every shot in the given trees, isolating per-shot failures.

    One unreadable shot must not stop the run: the archive holds a few
    directories with corrupt headers and partial exports, and finding them is
    part of what this pass is for. Each failure is recorded against its shot
    and the walk continues.
    """
    from vaft.omas.vest_upstream import write_stage_product

    summary: dict[str, Any] = {
        "succeeded": 0, "failed": 0, "skipped": 0, "unavailable": 0, "failures": [],
    }
    unavailable_errors = _unavailable_errors()

    for tree in trees:
        # One registry per tree, so a camera run and a soft X-ray run touch
        # different files rather than the same one.
        registry_path = _stage_directory(root, tree) / REGISTRY_NAME
        registry = load_registry(registry_path)
        available = discover_shots(root, tree)
        if shots is not None:
            wanted = set(shots)
            available = [shot for shot in available if shot in wanted]
        if limit is not None:
            available = available[:limit]
        LOGGER.info("%s: %d shots to build", tree, len(available))

        for shot in available:
            key = f"{tree}/{shot}"
            if not force and key in registry and registry[key].get("status") in {"success", "unavailable"}:
                summary["skipped"] += 1
                continue
            if dry_run:
                LOGGER.info("would ingest %s", key)
                continue

            product = _product_path(root, tree, shot)
            output_dir = product.parent
            try:
                ods, manifest = build_shot(root, tree, shot)
                output_dir.mkdir(parents=True, exist_ok=True)
                write_stage_product(
                    ods,
                    manifest,
                    output=product,
                    metadata=_filedb(root).omas_manifest(tree, shot=shot),
                    compression=manifest.get("storage", {}).get("compression"),
                )
            except unavailable_errors as exc:
                registry[key] = {"status": "unavailable", "reason": str(exc),
                                 "at": datetime.now(timezone.utc).isoformat()}
                summary["unavailable"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad shot must not stop the run
                reason = f"{type(exc).__name__}: {exc}"
                LOGGER.error("%s failed: %s", key, reason)
                LOGGER.debug("%s", traceback.format_exc())
                registry[key] = {"status": "failed", "reason": reason,
                                 "at": datetime.now(timezone.utc).isoformat()}
                summary["failed"] += 1
                summary["failures"].append({"shot": shot, "tree": tree, "reason": reason})
            else:
                registry[key] = {
                    "status": "success",
                    "measured": manifest["measured"]["count"],
                    "at": manifest["generated_at"],
                }
                summary["succeeded"] += 1
                LOGGER.info("%s -> %s (%s measured)", key, output_dir,
                            manifest["measured"]["count"])
            save_registry(registry_path, registry)

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Consolidated FileDB root")
    parser.add_argument(
        "--diagnostic",
        action="append",
        choices=sorted(DIAGNOSTIC_TREES),
        help="Restrict to these trees (repeatable). Default: every unreserved tree.",
    )
    parser.add_argument(
        "--include-fluctuation",
        action="store_true",
        help=(
            "Also ingest the acquisitions reserved for the fluctuation routine "
            "(issue #161). Off by default so the reserved set stays untouched."
        ),
    )
    parser.add_argument("--limit", type=int, help="Only the first N shots per tree")
    parser.add_argument(
        "--shot",
        type=int,
        action="append",
        help="Build only these shots (repeatable)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild shots already recorded")
    parser.add_argument("--dry-run", action="store_true", help="List what would be built")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.diagnostic:
        trees = list(args.diagnostic)
    else:
        trees = [t for t in DIAGNOSTIC_TREES if t not in RESERVED_TREES]
        if args.include_fluctuation:
            trees += list(RESERVED_TREES)

    reserved = [t for t in trees if t in RESERVED_TREES]
    if reserved and not args.include_fluctuation:
        LOGGER.warning(
            "Ingesting %s, which is reserved for issue #161. "
            "Requested explicitly, so proceeding.",
            ", ".join(reserved),
        )

    summary = ingest(
        Path(args.root).expanduser(),
        trees,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
        shots=args.shot,
    )

    LOGGER.info(
        "Done: %d succeeded, %d failed, %d unavailable, %d already present",
        summary["succeeded"], summary["failed"], summary["unavailable"], summary["skipped"],
    )
    for failure in summary["failures"][:20]:
        LOGGER.info("  failed %s/%s: %s", failure["tree"], failure["shot"], failure["reason"])
    return 1 if summary["failed"] and not summary["succeeded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
