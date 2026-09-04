#!/usr/bin/env python3

"""Generate soft X-ray and camera IDSs from the consolidated FileDB archive.

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

Products land in ``{root}/ods/{diagnostic}/{shot}/``. That is not one of the
``FileDBDomain`` values on purpose: neither diagnostic has an ``OMASStage`` of
its own, and inventing a path under ``omas/`` that ``FileDB.resolve`` would
reject is worse than an obviously separate tree. Giving these two diagnostics
real stages, and a ``STAGE_REPLICATION`` entry so their IDSs reach HSDS, is
issue #130's job.

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
}

RESERVED_TREES = ("camera_visible_fluctuation",)

REGISTRY_NAME = "ingested_shots.json"
MANIFEST_NAME = "manifest.json"

#: Product container. HDF5 by default: these are bulk numeric IDSs -- a single
#: soft X-ray shot is 128 channels x 39k samples, and a camera shot is a stack
#: of megapixel frames -- and OMAS's JSON writer spends roughly six bytes of
#: text per float. HDF5 is also what HSDS stores natively, so a product written
#: here needs no re-encoding to be replicated later.
PRODUCT_FORMATS = (".h5", ".json", ".json.gz")
DEFAULT_PRODUCT_FORMAT = ".h5"


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
        return len(ods["soft_x_rays.channel"])
    except (KeyError, IndexError, ValueError):
        return None


def build_shot(root: Path, tree: str, shot: int) -> tuple[Any, dict[str, Any]]:
    """Build one shot's ODS from the consolidated archive."""
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


def ingest(
    root: Path,
    trees: Sequence[str],
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    product_format: str = DEFAULT_PRODUCT_FORMAT,
    shots: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build every shot in the given trees, isolating per-shot failures.

    One unreadable shot must not stop the run: the archive holds a few
    directories with corrupt headers and partial exports, and finding them is
    part of what this pass is for. Each failure is recorded against its shot
    and the walk continues.
    """
    from vaft.omas.vest_upstream import write_stage_product

    summary: dict[str, Any] = {"succeeded": 0, "failed": 0, "skipped": 0, "failures": []}

    for tree in trees:
        # One registry per tree, so a camera run and a soft X-ray run touch
        # different files rather than the same one.
        registry_path = root / "ods" / tree / REGISTRY_NAME
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
            if not force and key in registry and registry[key].get("status") == "success":
                summary["skipped"] += 1
                continue
            if dry_run:
                LOGGER.info("would ingest %s", key)
                continue

            output_dir = root / "ods" / tree / str(shot)
            try:
                ods, manifest = build_shot(root, tree, shot)
                output_dir.mkdir(parents=True, exist_ok=True)
                write_stage_product(
                    ods,
                    manifest,
                    output=output_dir / f"{shot}_{tree}{product_format}",
                    metadata=output_dir / MANIFEST_NAME,
                )
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
    parser.add_argument(
        "--format",
        default=DEFAULT_PRODUCT_FORMAT,
        choices=PRODUCT_FORMATS,
        help="Product container (default: %(default)s)",
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
        product_format=args.format,
        shots=args.shot,
    )

    LOGGER.info(
        "Done: %d succeeded, %d failed, %d already present",
        summary["succeeded"], summary["failed"], summary["skipped"],
    )
    for failure in summary["failures"][:20]:
        LOGGER.info("  failed %s/%s: %s", failure["tree"], failure["shot"], failure["reason"])
    return 1 if summary["failed"] and not summary["succeeded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
