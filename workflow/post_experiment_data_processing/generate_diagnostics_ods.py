#!/usr/bin/env python3
"""
Script: generate_diagnostics_ods.py

Description:
    Reads raw diagnostic data for a given shot and generates a diagnostics ODS (OMAS Data Structure) file with structured diagnostic signals.

Inputs:
    --shot   <shot_number>        Shot number or identifier.
    --input  <raw_diag_file>      Path to raw diagnostics input file (e.g. measurements or IMAS IDS).
    --output <diagnostics_ods>    Path to output diagnostics ODS file.

Outputs:
    - An ODS file containing processed diagnostic data, saved to the specified output path.

Logging:
    - Logs are written to /srv/vest.filedb/public/<shot_number>/logs/generate_diagnostics_ods.log
"""
import os
import sys
import argparse
import logging
import datetime
from omas import ODS, save_omas_json, load_omas_json
from vaft.machine_mapping import get_machine_mapping
from vaft.database import vest_connection_pool
from vaft.machine_mapping import (
    pf_active,
    filterscope,
    barometry,
    tf,
    magnetics,
    dataset_description,
    summary
)

def generate_diagnostics_ods(shotnumber: int, save_dir: str, em_coupling_file: str = None) -> None:
    """
    Generate Diagnostics ODS file such that save_dir/{shotnumber}_diagnostics.json is created.
    
    Args:
        shotnumber: Shot number
        save_dir: Directory to save the output file
        em_coupling_file: Optional path to EM coupling file
    """
    start_time = datetime.datetime.now()

    # Get machine mapping configuration from vaft
    machine_mapping = get_machine_mapping()

    # Create ODS structure
    ods = ODS()

    # Add dataset description
    dataset_description(ods, str(shotnumber), {
        'source_type': 'shot',
        'version': 1,
        'description': f'VEST diagnostics data for shot {shotnumber}',
        'quality_level': 'processed',
        'quality_comment': 'Processed data from VEST database'
    })

    # Add experiment summary
    summary(ods, str(shotnumber), {
        'source_type': 'shot',
        'status': 'completed',
        'comment': f'VEST experiment data for shot {shotnumber}'
    })

    # Generate connection pool for VEST SQL database
    vest_connection_pool()

    # Process diagnostic data with static and dynamic sources
    pf_active(ods, str(shotnumber), str(shotnumber), {
        'source_type': 'shot',
        'static_source_type': 'shot',
        'dynamic_source_type': 'shot',
        'time_range': (
            machine_mapping.get('diagnostics', {}).get('tstart', 0.26),
            machine_mapping.get('diagnostics', {}).get('tend', 0.36)
        ),
        'dt': machine_mapping.get('diagnostics', {}).get('dt', 4e-5)
    })

    filterscope(ods, str(shotnumber), str(shotnumber), {
        'source_type': 'shot',
        'static_source_type': 'shot',
        'dynamic_source_type': 'shot',
        'time_range': (
            machine_mapping.get('diagnostics', {}).get('tstart', 0.26),
            machine_mapping.get('diagnostics', {}).get('tend', 0.36)
        ),
        'dt': machine_mapping.get('diagnostics', {}).get('dt', 4e-5)
    })

    barometry(ods, str(shotnumber), str(shotnumber), {
        'source_type': 'shot',
        'static_source_type': 'shot',
        'dynamic_source_type': 'shot',
        'time_range': (
            machine_mapping.get('diagnostics', {}).get('tstart', 0.26),
            machine_mapping.get('diagnostics', {}).get('tend', 0.36)
        ),
        'dt': machine_mapping.get('diagnostics', {}).get('dt', 4e-5)
    })

    # Load and process magnetic data
    if em_coupling_file is None:        
        if not os.path.exists(os.path.join(save_dir, 'emcoupling.json')):
            print('emcoupling.json file not found')
            # EM coupling calculation will be handled by model.py
            em_coupling_file = os.path.join(save_dir, 'emcoupling.json')
        else:
            em_coupling_file = os.path.join(save_dir, 'emcoupling.json')

    # Load EM coupling data
    if os.path.exists(em_coupling_file):
        ods_EM = load_omas_json(em_coupling_file)
        ods['em_coupling'] = ods_EM['em_coupling']

    tf(ods, str(shotnumber), str(shotnumber), {
        'source_type': 'shot',
        'static_source_type': 'shot',
        'dynamic_source_type': 'shot',
        'time_range': (
            machine_mapping.get('diagnostics', {}).get('tstart', 0.26),
            machine_mapping.get('diagnostics', {}).get('tend', 0.36)
        ),
        'dt': machine_mapping.get('diagnostics', {}).get('dt', 4e-5)
    })

    magnetics(ods, str(shotnumber), str(shotnumber), {
        'source_type': 'shot',
        'static_source_type': 'shot',
        'dynamic_source_type': 'shot',
        'time_range': (
            machine_mapping.get('diagnostics', {}).get('tstart', 0.26),
            machine_mapping.get('diagnostics', {}).get('tend', 0.36)
        ),
        'dt': machine_mapping.get('diagnostics', {}).get('dt', 4e-5)
    })

    # Save ODS to file
    fullfilename = os.path.join(save_dir, f'{shotnumber}_diagnostics.json')
    save_omas_json(ods, fullfilename)

    end_time = datetime.datetime.now()
    print(f'Diagnostics ODS file is generated in {end_time - start_time}')

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostics ODS from raw data.")
    parser.add_argument("--shot", required=True, help="Shot number/ID for which to process diagnostics.")
    parser.add_argument("--output", required=True, help="Path to save the output diagnostics ODS file.")
    args = parser.parse_args()

    shot = args.shot
    if args.output:
        out_file = args.output
        save_dir = os.path.dirname(out_file)
    else:
        out_file = f"/srv/vest.filedb/public/{shot}/omas/{shot}_diagnostics.json"
        save_dir = f"/srv/vest.filedb/public/{shot}/omas"

    # Ensure logs directory exists
    log_dir = f"/srv/vest.filedb/public/{shot}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "generate_diagnostics_ods.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    try:
        emcoupling_file = "/srv/vest.filedb/public/static/emcoupling.json"
        
        logging.info(f"Starting diagnostics ODS generation for shot {shot}")
        logging.info(f"Reading raw diagnostics input from SQL database for shot {shot}")
        logging.info("Processing raw data into ODS format")
        generate_diagnostics_ods(int(shot), save_dir, emcoupling_file)

        logging.info(f"Saving diagnostics ODS to {out_file}")
        logging.info(f"Diagnostics ODS generation completed for shot {shot}")
    except Exception as e:
        logging.error(f"Error in generate_diagnostics_ods: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
