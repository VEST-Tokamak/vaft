# This Python function aims to extract the most representative summary parameters from OMAS data for each shot number.
# The following key summary parameters can be extracted:
# 1. Shot number
# 2. Shot Class (Test, Vacuum, Plasma, etc.)
# 3. Plasma Onset Time (from DaQ time convention)
# 4. Pulse duration
# 5. Maximum Plasma current (Ip)
# 6. Toroidal field (Bt) at the vessel on-axis R=0.4m


# file dir /srv/vest.filedb/public/{shotnumber}/omas/{shotnumber}_combined.ods

# excel file name: {shotnumber}_omas_history.xlsx in current directory
import os
import re
import glob
import omas
import pandas as pd
from pathlib import Path
import vaft 
from omas import *

def find_ods_files(base_path=None):
    """Find all ODS files in the specified directory structure.
    
    Args:
        base_path (str, optional): Base directory path to search for ODS files. 
                                 Defaults to "/srv/vest.filedb/public".
    """
    if base_path is None:
    base_path = "/srv/vest.filedb/public"
    
    ods_files = []
    
    try:
        # Find all directories that match the pattern
        shot_dirs = glob.glob(os.path.join(base_path, "[0-9]" * 5))
        print(f"Found {len(shot_dirs)} shot directories in {base_path}")
        
        for shot_dir in shot_dirs:
            shot_number = os.path.basename(shot_dir)
            ods_path = os.path.join(shot_dir, "omas", f"{shot_number}_combined.json")
            
            if os.path.exists(ods_path):
                print(f"Found ODS file for shot {shot_number}")
                ods_files.append((int(shot_number), ods_path))
            else:
                print(f"No ODS file found for shot {shot_number} at {ods_path}")
        
        print(f"Total ODS files found: {len(ods_files)}")
        return ods_files
    except Exception as e:
        print(f"Error finding ODS files: {str(e)}")
        return []

def load_or_create_excel():
    """Load existing Excel file or create a new one."""
    excel_file = "omas_history.xlsx"
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
            # Ensure all required columns exist
            required_columns = [
                'shot_number',
                'shot_class',
                'plasma_onset_time',
                'pulse_duration',
                'max_plasma_current',
                'toroidal_field'
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
        except Exception as e:
            print(f"Error loading existing Excel file: {str(e)}")
            df = create_new_dataframe()
    else:
        df = create_new_dataframe()
    return df

def create_new_dataframe():
    """Create a new DataFrame with required columns."""
    return pd.DataFrame(columns=[
        'shot_number',
        'shot_class',
        'plasma_onset_time',
        'pulse_duration',
        'max_plasma_current',
        'toroidal_field'
    ])

def extract_shot_data(shot, ods_path):
    """Extract key parameters from OMAS data for a single shot."""
    try:
        if not os.path.exists(ods_path):
            print(f"ODS file not found for shot {shot}: {ods_path}")
            return None
            
        print(f"Loading ODS file for shot {shot} from {ods_path}")
        ods = load_omas_json(ods_path)
        
        # Extract information
        pulse_duration = vaft.omas.find_pulse_duration(ods) * 1000  # convert to ms
        breakdown_onset = vaft.omas.find_breakdown_onset(ods) * 1000  # convert to ms
        max_ip = vaft.omas.find_max_ip(ods) / 1000  # convert to kA
        bt = vaft.omas.find_bt(ods)
        shotclass = vaft.omas.find_shotclass(ods)

        # round the values to 2 decimal places
        pulse_duration = round(pulse_duration, 2)
        breakdown_onset = round(breakdown_onset, 2)
        max_ip = round(max_ip, 2)
        bt = round(bt, 3)
        
        print(f"Successfully extracted data for shot {shot}:")
        print(f"  - Pulse duration: {pulse_duration} ms")
        print(f"  - Breakdown onset: {breakdown_onset} ms")
        print(f"  - Max IP: {max_ip} kA")
        print(f"  - Toroidal field: {bt} T")
        print(f"  - Shot class: {shotclass}")
        
        return {
            'shot_number': shot,
            'shot_class': shotclass,
            'plasma_onset_time': breakdown_onset,
            'pulse_duration': pulse_duration,
            'max_plasma_current': max_ip,
            'toroidal_field': bt
        }
    except Exception as e:
        print(f"Error processing shot {shot}: {str(e)}")
        return None

def generate_omas_history_excel(base_path=None):
    """Main function to generate OMAS history Excel file.
    
    Args:
        base_path (str, optional): Base directory path to search for ODS files.
                                 Defaults to "/srv/vest.filedb/public".
    """
    try:
        # Step 1: Load or create Excel file
        df = load_or_create_excel()
        processed_shots = set(df['shot_number'].tolist())
        
        # Step 2: Find all ODS files with the specified base path
        ods_files = find_ods_files(base_path)
        
        # Step 3: Filter unprocessed shots
        unprocessed_shots = [(shot, path) for shot, path in ods_files if shot not in processed_shots]
        
        if not unprocessed_shots:
            print("No new shots to process.")
            return
            
        # Step 4: Process each unprocessed shot
        for shot_number, ods_path in unprocessed_shots:
            print(f"Processing shot {shot_number}...")
            shot_data = extract_shot_data(shot_number, ods_path)
            if shot_data:
                df = pd.concat([df, pd.DataFrame([shot_data])], ignore_index=True)
                # Sort DataFrame by shot_number before saving
                df = df.sort_values('shot_number')
                # Save after each shot to prevent data loss
                df.to_excel("omas_history.xlsx", index=False)
                print(f"Successfully processed shot {shot_number}")
            else:
                print(f"Failed to process shot {shot_number}")
        
        print("Processing complete!")
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate OMAS history Excel file')
    parser.add_argument('--base-path', type=str, 
                       default="/srv/vest.filedb/public",
                       help='Base directory path to search for ODS files')
    
    args = parser.parse_args()
    generate_omas_history_excel(args.base_path)

