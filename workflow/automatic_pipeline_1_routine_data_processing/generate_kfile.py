#!/usr/bin/env python3
"""Generate EFIT kfiles from constraints OMAS ODS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omas import load_omas_json

from vaft.code.efit import generate_kfile


LOGGER = logging.getLogger("vaft.generate_kfile")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--constraints-ods", required=True, type=Path, help="Input constraints ODS JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Manifest file to write.")
    parser.add_argument("--npprime", default=2, type=int, help="EFIT KPPCUR value.")
    parser.add_argument("--nffprime", default=2, type=int, help="EFIT KFFCUR value.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ods = load_omas_json(str(args.constraints_ods), consistency_check=False)

    efit_dir = args.output.parent.parent
    efit_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating kfiles for shot %s in %s", args.shot, efit_dir)
    generate_kfile(ods, args.shot, args.npprime, args.nffprime, save_dir=str(efit_dir))

    kfiles = sorted((efit_dir / "kfile").glob(f"k0{args.shot}.*"))
    if not kfiles:
        raise FileNotFoundError(f"No kfiles generated under {efit_dir / 'kfile'}")
    args.output.write_text("\n".join(str(path) for path in kfiles) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %d kfile paths to %s", len(kfiles), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
