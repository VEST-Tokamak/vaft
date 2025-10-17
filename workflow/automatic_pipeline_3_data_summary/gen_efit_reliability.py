import os
import glob
import csv
import json
import pandas as pd
import numpy as np
from omas import load_omas_json
from omas import *
import multiprocessing as mp
from functools import partial
import logging
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see more information
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BASE_PATH = "/srv/vest.filedb/public"
BATCH_SIZE = 20  # Number of files to process in each batch
OUTPUT_FILENAME = "efit_reliability_history.xlsx"
CSV_OUTPUT_FILENAME = "efit_reliability_history.csv"

def safe_get_nested(data, keys, default=None):
    """
    Safely get nested dictionary values using dot notation
    """
    try:
        for key in keys.split('.'):
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data
    except (KeyError, TypeError, AttributeError):
        return default

def find_efit_json_files(base_path):
    """
    Find all _efit.json files in the directory structure:
    /srv/vest.filedb/public/{shot}/omas/{shot}_efit.json
    """
    efit_files = []
    
    # Look for shot directories
    shot_dirs = glob.glob(os.path.join(base_path, "*"))
    
    for shot_dir in shot_dirs:
        if os.path.isdir(shot_dir):
            shot_number = os.path.basename(shot_dir)
            omas_dir = os.path.join(shot_dir, "omas")
            
            if os.path.exists(omas_dir):
                efit_file = os.path.join(omas_dir, f"{shot_number}_efit.json")
                if os.path.exists(efit_file):
                    efit_files.append((shot_number, efit_file))
    
    logger.info(f"Found {len(efit_files)} _efit.json files")
    return efit_files

def process_efit_file(shot_number, efit_file_path):
    """
    Process a single _efit.json file and extract reliability data
    """
    try:
        # Load the OMAS JSON file
        ods = load_omas_json(efit_file_path)
        
        reliability_data = []
        
        # Debug: Print the top-level keys to understand the structure
        logger.debug(f"Top-level keys in {efit_file_path}: {list(ods.keys())}")
        
        # Check if the file has the expected structure
        if 'equilibrium.time_slice' not in ods:
            logger.warning(f"File {efit_file_path} does not contain equilibrium.time_slice data")
            logger.debug(f"Available keys: {list(ods.keys())}")
            return reliability_data
        
        # Debug: Check the structure of equilibrium data
        if 'equilibrium' in ods:
            logger.debug(f"Equilibrium keys: {list(ods['equilibrium'].keys())}")
            if 'time_slice' in ods['equilibrium']:
                logger.debug(f"Time slice type: {type(ods['equilibrium.time_slice'])}")
                if len(ods['equilibrium.time_slice']) > 0:
                    logger.debug(f"First time slice type: {type(ods['equilibrium.time_slice'][0])}")
                    logger.debug(f"First time slice keys: {list(ods['equilibrium.time_slice'][0].keys())}")
        
        # Get time array
        time_array = ods['equilibrium.time']
        
        for j, time in enumerate(time_array):
            try:
                time_slice = ods['equilibrium.time_slice'][j]
                
                # Check if time_slice is a dictionary or ODS object
                if not (isinstance(time_slice, dict) or hasattr(time_slice, '__getitem__')):
                    logger.warning(f"Time slice {j} in shot {shot_number} is not accessible: {type(time_slice)}")
                    continue
                
                # Extract IP (plasma current) in kA
                try:
                    if 'global_quantities.ip' in time_slice:
                        ip = time_slice['global_quantities.ip'] / 1000  # Convert to kA
                    else:
                        ip = 0  # Default value if IP not available
                except Exception as e:
                    logger.warning(f"Could not extract IP from time slice {j} in shot {shot_number}: {e}")
                    ip = 0
                
                # Process Bpol probes (magnetic diagnostics)
                if 'constraints.bpol_probe' in time_slice:
                    try:
                        bpol_probes = time_slice['constraints.bpol_probe']
                        # Check if bpol_probes is a list or has length
                        if not (isinstance(bpol_probes, list) or hasattr(bpol_probes, '__len__')):
                            logger.warning(f"Bpol probes in shot {shot_number}, time slice {j} is not a list: {type(bpol_probes)}")
                            continue
                        
                        for k, probe in enumerate(bpol_probes):
                            # Check if probe is accessible (dict or ODS object)
                            if not (isinstance(probe, dict) or hasattr(probe, '__getitem__')):
                                logger.warning(f"Probe {k} in shot {shot_number}, time slice {j} is not accessible: {type(probe)}")
                                continue
                            
                            try:
                                measured = probe['measured']
                                reconstructed = probe['reconstructed']
                                chi_squared = probe['chi_squared']
                            except (KeyError, TypeError) as e:
                                logger.warning(f"Missing or invalid data in probe {k}, shot {shot_number}, time slice {j}: {e}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error processing Bpol probes in shot {shot_number}, time slice {j}: {e}")
                        continue
                        
                        # Determine probe type based on position (this might need adjustment based on your data structure)
                        probe_type = f"Bpol_Probe_{k+1}"
                        
                        reliability_data.append({
                            'shotnumber': shot_number,
                            'time': time,
                            'ip': ip,
                            'md_index': k + 1,
                            'type': probe_type,
                            'measured': measured,
                            'reconstructed': reconstructed,
                            'chi_squared': chi_squared
                        })
                
                # Process flux loops
                if 'constraints.flux_loop' in time_slice:
                    try:
                        flux_loops = time_slice['constraints.flux_loop']
                        # Check if flux_loops is a list or has length
                        if not (isinstance(flux_loops, list) or hasattr(flux_loops, '__len__')):
                            logger.warning(f"Flux loops in shot {shot_number}, time slice {j} is not a list: {type(flux_loops)}")
                            continue
                        
                        for k, loop in enumerate(flux_loops):
                            # Check if loop is accessible (dict or ODS object)
                            if not (isinstance(loop, dict) or hasattr(loop, '__getitem__')):
                                logger.warning(f"Flux loop {k} in shot {shot_number}, time slice {j} is not accessible: {type(loop)}")
                                continue
                            
                            try:
                                measured = loop['measured']
                                reconstructed = loop['reconstructed']
                                chi_squared = loop['chi_squared']
                            except (KeyError, TypeError) as e:
                                logger.warning(f"Missing or invalid data in flux loop {k}, shot {shot_number}, time slice {j}: {e}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error processing flux loops in shot {shot_number}, time slice {j}: {e}")
                        continue
                        
                        reliability_data.append({
                            'shotnumber': shot_number,
                            'time': time,
                            'ip': ip,
                            'md_index': k + 1,
                            'type': f"Flux_Loop_{k+1}",
                            'measured': measured,
                            'reconstructed': reconstructed,
                            'chi_squared': chi_squared
                        })
                
                # Process diamagnetic flux
                if 'constraints.diamagnetic_flux' in time_slice:
                    try:
                        diamag = time_slice['constraints.diamagnetic_flux']
                        # Check if diamag is accessible (dict or ODS object)
                        if not (isinstance(diamag, dict) or hasattr(diamag, '__getitem__')):
                            logger.warning(f"Diamagnetic flux in shot {shot_number}, time slice {j} is not accessible: {type(diamag)}")
                            continue
                        
                        try:
                            measured = diamag['measured']
                            reconstructed = diamag['reconstructed']
                            chi_squared = diamag['chi_squared']
                        except (KeyError, TypeError) as e:
                            logger.warning(f"Missing or invalid data in diamagnetic flux, shot {shot_number}, time slice {j}: {e}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error processing diamagnetic flux in shot {shot_number}, time slice {j}: {e}")
                        continue
                    
                    reliability_data.append({
                        'shotnumber': shot_number,
                        'time': time,
                        'ip': ip,
                        'md_index': 0,
                        'type': 'Diamagnetic_Flux',
                        'measured': measured,
                        'reconstructed': reconstructed,
                        'chi_squared': chi_squared
                    })
                
            except Exception as e:
                logger.warning(f"Error processing time slice {j} in shot {shot_number}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error loading file {efit_file_path}: {str(e)}")
        return []
    
    return reliability_data

def process_files_batch(file_batch):
    """
    Process a batch of files
    """
    all_data = []
    for shot_number, file_path in file_batch:
        data = process_efit_file(shot_number, file_path)
        all_data.extend(data)
    return all_data

def main():
    """
    Main function to mine all _efit.json files and generate reliability data
    """
    logger.info("Starting EFIT reliability data mining...")
    
    # Find all _efit.json files
    efit_files = find_efit_json_files(DEFAULT_BASE_PATH)
    
    if not efit_files:
        logger.error("No _efit.json files found!")
        return
    
    # Process files in batches
    all_reliability_data = []
    
    # Use multiprocessing for better performance
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Split files into batches
        batches = [efit_files[i:i + BATCH_SIZE] for i in range(0, len(efit_files), BATCH_SIZE)]
        
        # Process batches with progress bar
        for batch_data in tqdm(
            pool.imap(process_files_batch, batches),
            total=len(batches),
            desc="Processing file batches"
        ):
            all_reliability_data.extend(batch_data)
    
    logger.info(f"Processed {len(all_reliability_data)} reliability data points")
    
    if not all_reliability_data:
        logger.warning("No reliability data extracted!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_reliability_data)
    
    # Save to CSV
    df.to_csv(CSV_OUTPUT_FILENAME, index=False)
    logger.info(f"Saved reliability data to {CSV_OUTPUT_FILENAME}")
    
    # Save to Excel
    try:
        with pd.ExcelWriter(OUTPUT_FILENAME, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Reliability_Data', index=False)
            
            # Create summary sheet
            summary_data = []
            for shot in df['shotnumber'].unique():
                shot_data = df[df['shotnumber'] == shot]
                summary_data.append({
                    'shotnumber': shot,
                    'total_time_points': len(shot_data['time'].unique()),
                    'total_measurements': len(shot_data),
                    'avg_chi_squared': shot_data['chi_squared'].mean(),
                    'min_chi_squared': shot_data['chi_squared'].min(),
                    'max_chi_squared': shot_data['chi_squared'].max()
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Saved reliability data to {OUTPUT_FILENAME}")
        
    except Exception as e:
        logger.error(f"Error saving to Excel: {str(e)}")
        logger.info("CSV file was saved successfully")
    
    # Print summary statistics
    logger.info(f"\nSummary:")
    logger.info(f"Total shots processed: {df['shotnumber'].nunique()}")
    logger.info(f"Total time points: {df['time'].nunique()}")
    logger.info(f"Total measurements: {len(df)}")
    logger.info(f"Average chi-squared: {df['chi_squared'].mean():.4f}")
    logger.info(f"Measurement types: {df['type'].unique()}")

if __name__ == "__main__":
    main()

            

