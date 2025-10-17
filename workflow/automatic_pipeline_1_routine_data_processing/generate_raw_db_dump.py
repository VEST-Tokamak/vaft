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
import os
import sys
import argparse
import yaml
import logging

def main():
    parser = argparse.ArgumentParser(description="Generate raw db dump from DAQ raw data.")
    parser.add_argument("--shot", required=True, help="Shot number/ID for which to process diagnostics.")
    parser.add_argument("--output", required=False, help="output filename with path to save the output diagnostics ODS file.")
    args = parser.parse_args()
    shot = int(args.shot)
    if args.output:
        output_path = args.output
    else:
        output_path = f"vest_{shot}_daq_raw.json.gz"
    init_pool()
    dump_all_raw_signals_for_shot(shot = shot,output_path = output_path)
    print(f"Raw db dump file saved to {output_path}")

if __name__ == "__main__":
    main()