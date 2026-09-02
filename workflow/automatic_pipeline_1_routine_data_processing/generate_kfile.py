#!/usr/bin/env python3
"""Generate EFIT kfiles from constraints OMAS ODS."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omas import load_omas_json

from vaft.code.efit import EFITScientificConfig, generate_kfile


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
        help="EFIT KPPCUR override; defaults to 2 without --config.",
    )
    parser.add_argument(
        "--nffprime",
        type=int,
        help="EFIT KFFCUR override; defaults to 2 without --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Resolved EFIT scientific configuration or preparation manifest JSON.",
    )
    args = parser.parse_args()

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
