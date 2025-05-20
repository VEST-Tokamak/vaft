#!/usr/bin/env python3
"""
Script: run_chease.py

Description:
    Run chease for all gfile available.

Inputs:
    --shot   <shot_number>    Shot number/ID for logging context.
    --gfile  <gfile_path>     Path to the EFIT gfile (equilibrium reconstructed by EFIT).
    --config  <yaml file>     Configuration file for CHEASE path.
    --output <chease_output>    refined_gfile_generated.txt.

Outputs:
    - EFIT raw output file refined by CHEASE.

Logging:
    - Logs are written to /srv/vest.filedb/public/<shot_number>/logs/run_chease.log (captures CHEASE run info and errors).
"""
import os
import sys
import argparse
import logging

from VEST_efit import chease_run
import yaml
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run CHEASE.")
    parser.add_argument("--shot", required=True, help="Shot number/ID for logging context.")
    parser.add_argument("--gfile", required=True, help="Path to the EFIT input gfile.")
    parser.add_argument("--config", required=True, help="yaml file containing the options.")
    parser.add_argument("--output", required=True, help="Path to save the CHEASE output equilibrium file.")
    args = parser.parse_args()

    shot = args.shot
    gfile = args.gfile
    config_file = args.config
    output_file = args.output
    input_dir=os.path.dirname(gfile)
    output_dir=os.path.dirname(output_file)

    # Set up logging
    log_dir = f"/srv/vest.filedb/public/{shot}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_chease.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    try:
        if config_file:
            logging.info(f"Using configuration from {config_file} for eddy calculation")
            # Read configuration yaml file
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            eqdskpy_dir=config['chease_setting']['eqdskpy_dir']
        else:
            logging.info(f"Using default configuration for eddy calculation")
            eqdskpy_dir= '/home/user1/GPEC/1.5cp/docs/examples/workflow/eqdsk.py'

        logging.info(f"Starting CHEASE run for shot {shot} using gfile {gfile}")
        # Ensure the output directory exists
        chease_run(int(shot),input_dir,output_dir,eqdskpy_dir)

        f=open(output_file,'w')
        dir_list = os.listdir(output_dir)
        for file in dir_list:
            if file.startswith('g0'):
                f.write(f'{output_dir}/{file}\n')
        f.close()

        logging.info(f"CHEASE run completed for shot {shot}, output saved to {output_file}")
    except Exception as e:
        logging.error(f"Error in run_chease: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
