#!/usr/bin/env python3
"""Generate EFIT kfiles from constraints OMAS ODS."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from omas import load_omas_json

from vaft.code.efit import EFITScientificConfig, efit_preset, generate_kfile, preset_of
from vaft.code.efit.applicability import constraints_not_applicable_reason, kfile_manifest_text
from vaft.code.efit.presets import DEFAULT_PRESET, PRESET_RECORD


LOGGER = logging.getLogger("vaft.generate_kfile")

# Superseded generations kept per shot by default: the one just moved aside.
# A full regeneration pass over ~4,000 EFIT shots moves 22-28 MB per shot
# (65-80 instants of g/k/m/a-files) on every re-run, ~100 GB per pass, so the
# retention has to be bounded (cold review 0.8.0 delta-absorb-18 F3).
DEFAULT_KEEP_SUPERSEDED = 1
_STAMP = re.compile(r"^\d{8}T\d{12}Z$")


def _superseded_dirs(efit_dir: Path) -> list[Path]:
    return [efit_dir / f"{kind}file" for kind in "kgam"] + [efit_dir]


def prune_superseded(efit_dir: Path, keep: int = DEFAULT_KEEP_SUPERSEDED) -> list[str]:
    """Delete all but the newest ``keep`` ``superseded/<stamp>/`` generations of the shot.

    The stamps are UTC and fixed-width, so their names sort in time order.
    Only directories named like a stamp are considered; anything else under
    ``superseded/`` is left alone. Returns the stamps pruned, newest last.
    """
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")
    pruned: set[str] = set()
    for directory in _superseded_dirs(efit_dir):
        stamps = sorted(p for p in (directory / "superseded").glob("*") if p.is_dir() and _STAMP.match(p.name))
        for path in stamps[: len(stamps) - keep]:
            shutil.rmtree(path)
            pruned.add(path.name)
    return sorted(pruned)


def supersede_earlier_run(
    efit_dir: Path, shot: int, keep: int = DEFAULT_KEEP_SUPERSEDED
) -> Path | None:
    """Move the shot's k-, g-, a- and m-files of an earlier run under ``superseded/<UTC stamp>/``.

    The stage used to list its output by globbing ``kfile/``, so k-files of an
    earlier run (another configuration, another Green table, other instants)
    were run again with the new ones, and the collection read every g-file in
    ``gfile/`` (#1786). EFIT reads the table of the first k-file of a batch, so
    a stale first file failed every new one; stale files after a new first one
    produced g-files that passed as fresh. Each kind moves to its own
    ``<kind>file/superseded/<stamp>/`` (files EFIT left in the run directory
    itself to ``superseded/<stamp>/``). They are moved, not deleted: they are
    the record of what the earlier run used. The record is bounded: the
    newest ``keep`` generations stay (the default keeps the one just moved),
    older ones are deleted and their stamps logged.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    moved = 0
    for directory in _superseded_dirs(efit_dir):
        previous = [p for kind in "kgam" for p in sorted(directory.glob(f"{kind}0{shot}.*")) if p.is_file()]
        if not previous:
            continue
        target = directory / "superseded" / stamp
        target.mkdir(parents=True)
        for path in previous:
            path.rename(target / path.name)
        moved += len(previous)
    if moved:
        LOGGER.info("Moved %d EFIT file(s) of an earlier run of shot %s under superseded/%s", moved, shot, stamp)
    pruned = prune_superseded(efit_dir, keep)
    if pruned:
        LOGGER.info(
            "Pruned %d older superseded generation(s) of shot %s (keeping %d): %s",
            len(pruned), shot, keep, ", ".join(pruned),
        )
    return efit_dir if moved else None


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


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
    parser.add_argument(
        "--keep-superseded",
        type=_non_negative,
        default=DEFAULT_KEEP_SUPERSEDED,
        help="Superseded generations (superseded/<stamp>/ trees of the shot's earlier k/g/a/m-files) "
        f"to keep; older ones are deleted and logged. Default {DEFAULT_KEEP_SUPERSEDED}: the one just moved aside.",
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
        supersede_earlier_run(args.output.parent.parent, args.shot, keep=args.keep_superseded)
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
    # One writer per shot is assumed (the pipeline runs a shot's stages in one
    # chain); a second concurrent run of the same shot would move this one's files.
    supersede_earlier_run(efit_dir, args.shot, keep=args.keep_superseded)
    generate_kfile(
        ods,
        args.shot,
        args.npprime,
        args.nffprime,
        save_dir=str(efit_dir),
        config=scientific_config,
    )

    # Only what this run wrote: the shot's earlier files were moved aside above.
    kfiles = sorted(kfile_dir.glob(f"k0{args.shot}.*"))
    if not kfiles:
        raise FileNotFoundError(f"No kfiles generated under {kfile_dir}")
    args.output.write_text(
        "\n".join(str(path) for path in kfiles) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %d kfile paths to %s", len(kfiles), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
