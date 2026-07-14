#!/usr/bin/env python3
"""Generate diagnostics OMAS ODS from an archived VEST raw DAQ dump."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from omas import ODS, save_omas_json

from vaft.machine_mapping.barometry import barometry
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.magnetics import magnetics
from vaft.machine_mapping.pf_active import pf_active
from vaft.machine_mapping.spectrometer_uv import spectrometer_uv
from vaft.machine_mapping.tf import tf
from vaft.process.magnetics import VestMagneticsProcessingConfig


LOGGER = logging.getLogger("vaft.generate_diagnostics_ods")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_diagnostics_ods(
    *,
    shot: int,
    raw_dump: Path,
    tstart: float,
    tend: float,
    dt: float,
    run: int,
    vest_magnetics_processing: dict | None = None,
) -> ODS:
    """Build the diagnostics ODS using only VAFT machine-mapping helpers."""
    if not raw_dump.exists():
        raise FileNotFoundError(f"Raw dump not found: {raw_dump}")

    os.environ["VAFT_RAW_SAMPLE_PATH"] = str(raw_dump)
    os.environ["VAFT_RAW_OFFLINE_ONLY"] = "1"

    ods = ODS()
    magnetics_processing_config = (
        VestMagneticsProcessingConfig(**vest_magnetics_processing)
        if vest_magnetics_processing
        else None
    )
    dataset_description(
        ods,
        shot,
        {"source_type": "shot", "run": run, "machine": "VEST"},
    )
    pf_active(ods, shot, tstart, tend, dt)
    spectrometer_uv(ods, shot, tstart, tend, dt)
    barometry(ods, shot, tstart, tend, dt)
    tf(ods, shot, tstart, tend, dt)
    magnetics(ods, shot, tstart, tend, dt, processing_config=magnetics_processing_config)
    return ods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--raw-dump", required=True, type=Path, help="Archived raw DAQ JSON gzip file.")
    parser.add_argument("--output", required=True, type=Path, help="Output diagnostics ODS JSON file.")
    parser.add_argument("--tstart", default=0.26, type=float, help="Start time for mapped diagnostics.")
    parser.add_argument("--tend", default=0.36, type=float, help="End time for mapped diagnostics.")
    parser.add_argument("--dt", default=4e-5, type=float, help="Mapped diagnostics time step.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    parser.add_argument(
        "--vest-magnetics-processing-json",
        default="",
        help="JSON object overriding VEST magnetics processing defaults.",
    )
    args = parser.parse_args()

    _configure_logging()
    LOGGER.info("Generating diagnostics ODS for shot %s from %s", args.shot, args.raw_dump)
    ods = build_diagnostics_ods(
        shot=args.shot,
        raw_dump=args.raw_dump,
        tstart=args.tstart,
        tend=args.tend,
        dt=args.dt,
        run=args.run,
        vest_magnetics_processing=json.loads(args.vest_magnetics_processing_json)
        if args.vest_magnetics_processing_json.strip()
        else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))
    LOGGER.info("Diagnostics ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
