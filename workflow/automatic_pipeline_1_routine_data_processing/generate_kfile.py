#!/usr/bin/env python3
"""Generate EFIT kfiles from constraints OMAS ODS."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omas import load_omas_json

from vaft.code.efit import EFITScientificConfig, efit_preset, generate_kfile, preset_of
from vaft.code.efit.presets import DEFAULT_PRESET, PRESET_RECORD


LOGGER = logging.getLogger("vaft.generate_kfile")


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
    generate_kfile(
        ods,
        args.shot,
        args.npprime,
        args.nffprime,
        save_dir=str(efit_dir),
        config=scientific_config,
    )

    kfiles = sorted((efit_dir / "kfile").glob(f"k0{args.shot}.*"))
    if not kfiles:
        raise FileNotFoundError(f"No kfiles generated under {efit_dir / 'kfile'}")
    args.output.write_text(
        "\n".join(str(path) for path in kfiles) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %d kfile paths to %s", len(kfiles), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
