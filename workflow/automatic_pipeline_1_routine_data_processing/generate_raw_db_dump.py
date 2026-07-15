#!/usr/bin/env python3
"""
Generate Raw Dump File from DAQ Raw Data

Inputs:
    --shot   <shot_number>        Shot number or identifier.
    --output  Path and Filename to output Dump json.gz file.

Outputs:
    - An ODS file containing processed diagnostic data, saved to the specified output path.

Logging:
    - Logs are written to /srv/vest.filedb/public/<shot_number>/logs/generate_diagnostics_ods.log
"""
from vaft.database import dump_all_raw_signals_for_shot, init_pool
import argparse
import gzip
import json
import os
import shutil


def _copy_archived_sample(sample_path: str, output_path: str, shot: int) -> None:
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Archived raw sample not found: {sample_path}")

    with gzip.open(sample_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    file_shot = payload.get("shot")
    if file_shot is not None and int(file_shot) != int(shot):
        raise ValueError(f"Archived raw sample shot={file_shot}, requested shot={shot}")
    if not isinstance(payload.get("fields"), dict) or not payload["fields"]:
        raise ValueError(f"Archived raw sample has no fields: {sample_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shutil.copyfile(sample_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate raw db dump from DAQ raw data.")
    parser.add_argument("--shot", required=True, help="Shot number/ID for which to process diagnostics.")
    parser.add_argument("--output", required=False, help="output filename with path to save the output diagnostics ODS file.")
    parser.add_argument("--sample", required=False, help="Archived raw JSON gzip file to copy instead of reading SQL.")
    args = parser.parse_args()
    shot = int(args.shot)
    if args.output:
        output_path = args.output
    else:
        output_path = f"vest_{shot}_daq_raw.json.gz"

    if args.sample:
        _copy_archived_sample(args.sample, output_path, shot)
        print(f"Archived raw db dump copied from {args.sample} to {output_path}")
        return

    init_pool()
    dump_all_raw_signals_for_shot(shot = shot,output_path = output_path)
    print(f"Raw db dump file saved to {output_path}")

if __name__ == "__main__":
    main()
