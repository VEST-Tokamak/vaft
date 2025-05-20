#!/usr/bin/env python3
"""
Script: run_add_efit.py

Description:
    Executes the DB_add_efit function and handles its output and errors.

Inputs:
    --shot   <shot_number>    Shot number/ID for logging context.
    --input  <input path>     Path to the EFIT directory.
    --param  <config file>    Path to yaml config file to get the excel file name
    --output <efit_output>    Path for status file.

Outputs:
    - add gfile to DB status file, saved to the specified path.

Logging:
    - Logs are written to /srv/vest.filedb/public/<shot_number>/logs/run_DB_add_efit.log (captures EFIT run info and errors).
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from VEST_efit import DB_add_efit

# Constants
DEFAULT_EXCEL_FILE = 'VEST_DB.xlsx'
DEFAULT_PUBLIC_PATH = '/srv/vest.filedb/public'

def setup_logging(shot: str) -> str:
    """Set up logging configuration and return log path."""
    log_dir = Path(DEFAULT_PUBLIC_PATH) / shot / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / "run_DB_add_efit.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return str(log_path)

def load_config(config_file: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file or return default config."""
    if not config_file:
        logging.info("Using default configuration for eddy calculation")
        return {
            'DB_setting': {
                'file_name': str(Path(DEFAULT_PUBLIC_PATH) / DEFAULT_EXCEL_FILE)
            }
        }
    
    logging.info(f"Using configuration from {config_file} for eddy calculation")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def parse_excel_path(config: Dict[str, Any]) -> tuple[str, str]:
    """Parse excel file path from config and return path and filename."""
    excelfile = config['DB_setting']['file_name']
    exc = Path(excelfile).name
    path = str(Path(excelfile).parent)
    return path, exc

def process_efit(shot: str, config_file: Optional[str]) -> None:
    """Main process to add EFIT to database."""
    try:
        config = load_config(config_file)
        path, exc = parse_excel_path(config)
        
        logging.info(f"Starting adding EFIT to the DB for shot {shot}")
        DB_add_efit(path, exc, int(shot))
        logging.info(f"Adding EFIT completed for shot {shot}")
        
    except Exception as e:
        logging.error(f"Error processing EFIT for shot {shot}: {str(e)}", exc_info=True)
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run EFIT equilibrium solver.")
    parser.add_argument("--shot", required=True, help="Shot number/ID for logging context.")
    parser.add_argument("--input", required=True, help="Path to the EFIT directory.")
    parser.add_argument("--config", required=True, help="yaml file containing the options.")
    parser.add_argument("--output", required=True, help="Path to save the EFIT output equilibrium file.")
    return parser.parse_args()

def main() -> None:
    """Main entry point."""
    args = parse_args()
    log_path = setup_logging(args.shot)
    
    try:
        process_efit(args.shot, args.config)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
