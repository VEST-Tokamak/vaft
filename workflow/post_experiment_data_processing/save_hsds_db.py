#!/usr/bin/env python3
"""
Script: save_hsds_db.py

Description:
    Saves an ODS (OMAS Data Structure) file to the VAFT HSDS database.

Inputs:
    --shot <shot_number>     Shot number to save the data under
    --input <input_file>     Path to input ODS file (.json)

Outputs:
    - Saves the ODS data to the VAFT HSDS database under the specified shot number
    - Creates a marker file to indicate successful save

Logging:
    - Logs are written to /srv/vest.filedb/public/<shot_number>/logs/save_hsds_db.log
"""
import os
import sys
import argparse
import logging
import vaft
from omas import load_omas_json

def save_to_hsds(input_file, shot_number):
    """
    Save an ODS file to the VAFT HSDS database.

    Parameters
    ----------
    input_file : str
        Path to the input ODS JSON file
    shot_number : int
        Shot number under which to save the data

    Returns
    -------
    bool
        True if save was successful
    """
    try:
        # Load the ODS from file
        logging.info(f"Loading ODS from {input_file}")
        ods = load_omas_json(input_file)
        
        # Save to VAFT database
        logging.info(f"Saving to VAFT database with shot number {shot_number}")
        vaft.database.save(ods, shot_number, env='public')
        return True
    except Exception as e:
        logging.error(f"Failed to save to HSDS database: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Save ODS file to VAFT HSDS database.")
    parser.add_argument("--shot", required=True, help="Shot number to save the data under")
    parser.add_argument("--input", required=True, help="Path to input ODS file")
    args = parser.parse_args()

    # Setup logging
    log_dir = f"/srv/vest.filedb/public/{args.shot}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "save_hsds_db.log")

    logging.basicConfig(filename=log_path, level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(message)s",
                       datefmt="%Y-%m-%d %H:%M:%S")

    try:
        shot_number = int(args.shot)
        
        # Validate input file
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        logging.info(f"Starting HSDS database save process for shot {shot_number}")
        logging.info(f"Input file: {args.input}")

        # Save to database
        save_to_hsds(args.input, shot_number)
        
        # Create marker file to indicate successful save
        marker_file = os.path.join(os.path.dirname(args.input), "data_saved.txt")
        with open(marker_file, 'w') as f:
            f.write(f"Data successfully saved to HSDS database at {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, None, None, None))}")
        
        logging.info(f"Successfully saved ODS to HSDS database for shot {shot_number}")
        logging.info(f"Created marker file: {marker_file}")

    except Exception as e:
        logging.error(f"Error in save_hsds_db: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
