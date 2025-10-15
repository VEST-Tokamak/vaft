#!/usr/bin/env python3
"""
Script: run_add_gpec.py

Description:
    Executes the DB_add_gpec function and handles its output and errors.

Inputs:
    --shot   <shot_number>    Shot number/ID for logging context.
    --input  <input path>     Path to the GPEC directory.
    --param  <config file>    Path to yaml config file to get the excel file name
    --output <gpec_output>    Path for status file.

Outputs:
    - add gpec to DB status file, saved to the specified path.

Logging:
    - Logs are written to /srv/vest.filedb/public/<shot_number>/logs/run_DB_add_gpec.log (captures EFIT run info and errors).
"""
import os
import sys
import argparse
import logging
import subprocess
from VEST_efit import DB_add_gpec
import yaml
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run EFIT equilibrium solver.")
    parser.add_argument("--shot", required=True, help="Shot number/ID for logging context.")
    parser.add_argument("--input", required=True, help="Path to the EFIT directory.")
    parser.add_argument("--config", required=True, help="yaml file containing the options.")
    parser.add_argument("--output", required=True, help="Path to save the EFIT output equilibrium file.")
    args = parser.parse_args()

    shot = args.shot
    output_file = args.output # useless for now
    config_file = args.config

    # Set up logging
    log_dir = f"/srv/vest.filedb/public/{shot}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_DB_add_gpec.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    try:
        if config_file:
            logging.info(f"Using configuration from {config_file} for eddy calculation")
            # Read configuration yaml file
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            excelfile=config['DB_setting']['file_name']
            exc=excelfile.split('/')[-1]
            path=os.path.dirname(excelfile)
        else:
            logging.info(f"Using default configuration for eddy calculation")
            exc='VEST_DB.xlsx'
            path='/srv/vest.filedb/public'

        logging.info(f"Starting adding GPEC to the DB  for shot {shot}")
        # Ensure the output directory exists

        print(path,exc,int(shot))
        DB_add_gpec(path,exc,int(shot))

        logging.info(f"adding GPEC completed for shot {shot}")

    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
