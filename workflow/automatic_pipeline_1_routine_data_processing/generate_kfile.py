#!/usr/bin/env python3
"""Generate EFIT kfiles from constraints OMAS ODS."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from omas import load_omas_json

from vaft.code.efit import EFITScientificConfig, efit_preset, generate_kfile, preset_of
from vaft.code.efit.applicability import constraints_not_applicable_reason, kfile_manifest_text
from vaft.code.efit.presets import DEFAULT_PRESET, PRESET_RECORD


LOGGER = logging.getLogger("vaft.generate_kfile")


def supersede_kfiles(kfile_dir: Path, shot: int) -> Path | None:
    """Move the shot's k-files from an earlier run into ``kfile/superseded/<UTC stamp>/``.

    The stage used to list its output by globbing ``kfile/``, so k-files of an
    earlier run (another configuration, another Green table, other instants)
    were run again with the new ones (#1786). EFIT reads the table of the first
    k-file of a batch, so a stale first file failed every new one, and stale
    files after a new first file produced g-files that passed as fresh. They
    are moved, not deleted: they are the record of what the earlier run used.
    """
    previous = sorted(kfile_dir.glob(f"k0{shot}.*"))
    if not previous:
        return None
    target = kfile_dir / "superseded" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True)
    for path in previous:
        path.rename(target / path.name)
    LOGGER.info("Moved %d k-file(s) of an earlier run of shot %s to %s", len(previous), shot, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument(
        "--constraints-ods",
        required=True,
        type=Path,
        help="Input constraints ODS JSON.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Manifest file to write."
    )
    parser.add_argument(
        "--npprime",
        type=int,
        help="KPPCUR override on the default configuration (deprecated; --preset routine selects the legacy set).",
    )
    parser.add_argument(
        "--nffprime",
        type=int,
        help="KFFCUR override on the default configuration (deprecated; --preset routine selects the legacy set).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Resolved EFIT scientific configuration or preparation manifest JSON.",
    )
    parser.add_argument(
        "--preset",
        help="Named EFIT configuration (vaft.code.efit.PRESETS); without it, --config or a "
        f"legacy basis, the default ({DEFAULT_PRESET}). Exclusive with --config, --npprime and --nffprime.",
    )
    args = parser.parse_args()
    if args.preset and (args.config is not None or args.npprime is not None or args.nffprime is not None):
        parser.error("--preset carries its own configuration and profile basis; drop --config/--npprime/--nffprime")

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    ods = load_omas_json(str(args.constraints_ods), consistency_check=False)
    not_applicable = constraints_not_applicable_reason(ods)
    if not_applicable is not None:
        # No k-file to write (#205); the manifest carries the verdict on.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / PRESET_RECORD).unlink(missing_ok=True)
        supersede_kfiles(args.output.parent.parent / "kfile", args.shot)
        args.output.write_text(kfile_manifest_text(not_applicable), encoding="utf-8")
        LOGGER.info("EFIT not applicable to shot %s: %s", args.shot, not_applicable)
        return 0

    efit_dir = args.output.parent.parent
    efit_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating kfiles for shot %s in %s", args.shot, efit_dir)
    scientific_config = None
    if args.config is not None:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        if "resolved" in payload:
            payload = payload["resolved"]
        if "scientific" in payload:
            payload = payload["scientific"]
        scientific_config = EFITScientificConfig.from_dict(payload)
    # A preset's record sits beside the manifest; a stale one from an earlier
    # preset run must not describe a run that used none.
    record_path = args.output.parent / PRESET_RECORD
    record_path.unlink(missing_ok=True)
    legacy_basis = args.npprime is not None or args.nffprime is not None
    preset_name = args.preset or (None if (args.config is not None or legacy_basis) else DEFAULT_PRESET)
    if preset_name is None and scientific_config is not None and not legacy_basis:
        # A --config payload that resolves (by sha) to a named preset is that
        # preset: record it, so the product does not read `unrecorded` and a
        # replay carries the same floor and provenance as a --preset run.
        preset_name = preset_of(scientific_config)
    if preset_name:
        preset = efit_preset(preset_name)
        ods, floor_changes = preset.prepare_constraints(ods)
        scientific_config = preset.scientific
        LOGGER.info("EFIT preset %s (scientific sha256 %s); sigma floor raised %d channel(s)",
                    preset.name, preset.scientific.sha256[:12], sum(c["raised"] for c in floor_changes))
        record_path.write_text(
            json.dumps({**preset.record(), "sigma_floor_changes": floor_changes}, indent=1) + "\n",
            encoding="utf-8",
        )
    kfile_dir = efit_dir / "kfile"
    supersede_kfiles(kfile_dir, args.shot)
    before = set(kfile_dir.glob(f"k0{args.shot}.*"))
    if before:  # supersede_kfiles must leave nothing behind to be listed as this run's
        raise RuntimeError(f"k-files of an earlier run remain in {kfile_dir}: {sorted(p.name for p in before)}")
    generate_kfile(
        ods,
        args.shot,
        args.npprime,
        args.nffprime,
        save_dir=str(efit_dir),
        config=scientific_config,
    )

    # Only what this run wrote: the directory was emptied of the shot's k-files above.
    kfiles = sorted(set(kfile_dir.glob(f"k0{args.shot}.*")) - before)
    if not kfiles:
        raise FileNotFoundError(f"No kfiles generated under {kfile_dir}")
    args.output.write_text(
        "\n".join(str(path) for path in kfiles) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %d kfile paths to %s", len(kfiles), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
