# This Python function aims to extract the most representative summary parameters from OMAS data for each shot number.
# The following key summary parameters can be extracted:
# 1. Shot number
# 2. Shot Class (Test, Vacuum, Plasma, etc.)
# 3. Plasma Onset Time (from DaQ time convention)
# 4. Pulse duration
# 5. Maximum Plasma current (Ip)
# 6. Toroidal field (Bt) at the vessel on-axis R=0.4m


# file dir /srv/vest.filedb/public/{shotnumber}/omas/{shotnumber}_combined.ods

# excel file name: omas_history.xlsx in current directory
import os
import re
import glob
import omas
import pandas as pd
import logging
from pathlib import Path
import vaft.omas.general
from omas import *
import multiprocessing
import numpy as np

# Constants
DEFAULT_BASE_PATH = "/srv/vest.filedb/public"
EXCEL_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omas_history_new.xlsx")
REQUIRED_COLUMNS = [
    'shot',
    # 'shot_class',
    'plasma_onset_time',
    'pulse_duration',
    'max_plasma_current',
    'toroidal_field',
    'efit_kfile_count',
    'efit_gfile_count',
    'has_diagnostics_ods',
    'has_eddy_ods',
    'has_efit_ods',
    'has_chease_ods',
    'has_combined_ods'
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_max_ip(ods):
    """Find the maximum plasma current."""
    current = ods['magnetics.ip.0.data']
    from scipy.signal import medfilt
    data_filtered = medfilt(current, kernel_size=15)
    
    max_org = np.max(current)
    max_filtered = np.max(data_filtered)

    print(f"Original max IP: {max_org}, Filtered max IP: {max_filtered}")

    # if ods['dataset_description']['data_entry']['pulse'] == 40919 or ods['dataset_description']['data_entry']['pulse'] == '40919':
    #     if max_filtered > 100:
    #         raise RuntimeError("조건을 만족하지 않아 종료합니다.")
    
    return np.max(data_filtered)
import scipy.signal as signal
def signal_onoffset(time,data,smooth_window=5, threshold=0.01):
    print("threshold for signal detection:", threshold)
    # Smooth the data
    data=signal.savgol_filter(data, smooth_window, 3)

    # Find the onset and offset of a signal (e.g. Halpha signal)
    nbt=len(time)

    # index of maximum value
    indxm=min(range(len(data)), key=lambda i: abs(data[i]-max(data)))
    indxs=-1
    indxe=-1
    # We are looking for windows that constain continue data above threshold.
    # The window we are looking for, must contain the maximum value
    for i in range(nbt):
        if data[i]>= threshold:
            if indxs==-1:
                indxs=i # start of the window
        else:
            indxe=i-1 # end of the window
            if indxs < indxm and indxm < indxe:
                break # if the windiw contains the maximum, we stop
            indxs=-1
    onset=time[indxs]
    offset=time[indxe]
    return onset,offset

def find_pulse_duration(ods):
    """Find the duration of the pulse using spectrometer_uv signal."""
    time = ods.time('spectrometer_uv')
    data = ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
    onset, offset = signal_onoffset(time, data, threshold=0.05)
    return offset - onset

def _find_signal_onset(ods, time_key, data_key):
    """Helper to find signal onset using vaft.process.signal_onoffset."""
    time = ods.time(time_key)
    data = ods[data_key]
    onset, _ = signal_onoffset(time, data, threshold = 0.05)
    return onset

def find_breakdown_onset(ods):
    """Find the onset of the breakdown using spectrometer_uv signal."""
    return _find_signal_onset(ods, 'spectrometer_uv', 'spectrometer_uv.channel.0.processed_line.0.intensity.data')

def find_bt(ods):
    """Find the mean toroidal field at R=0.4m during plasma."""
    time = ods.time('tf.time')
    bt = ods['tf.b_field_tor_vacuum_r.data'] / ods['tf.r0']
    plasma_onset = find_breakdown_onset(ods)
    pulse_duration = find_pulse_duration(ods)
    plasma_offset = plasma_onset + pulse_duration
    onset_idx = np.searchsorted(time, plasma_onset)
    offset_idx = np.searchsorted(time, plasma_offset)
    bt = bt[onset_idx:offset_idx]
    return np.mean(bt)


def find_ods_files(base_path=None):
    """Find all ODS files in the specified directory structure.
    
    Args:
        base_path (str, optional): Base directory path to search for ODS files. 
                                 Defaults to DEFAULT_BASE_PATH.
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
    
    ods_files = []
    
    try:
        # Find all directories that match the pattern
        shot_dirs = glob.glob(os.path.join(base_path, "[0-9]" * 5))
        logger.info(f"Found {len(shot_dirs)} shot directories in {base_path}")
        
        for shot_dir in shot_dirs:
            shot = os.path.basename(shot_dir)
            ods_path = os.path.join(shot_dir, "omas", f"{shot}_diagnostics.json")
            
            if os.path.exists(ods_path):
                logger.info(f"Found ODS file for shot {shot}")
                ods_files.append((int(shot), ods_path))
            else:
                logger.warning(f"No ODS file found for shot {shot} at {ods_path}")
        
        logger.info(f"Total ODS files found: {len(ods_files)}")
        return ods_files
    except Exception as e:
        logger.error(f"Error finding ODS files: {str(e)}")
        return []

def load_or_create_excel():
    """Load existing Excel file or create a new one."""
    excel_file = EXCEL_FILENAME
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
            # Ensure all required columns exist
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = None
        except Exception as e:
            logger.error(f"Error loading existing Excel file: {str(e)}")
            df = create_new_dataframe()
    else:
        df = create_new_dataframe()
    return df

def create_new_dataframe():
    """Create a new DataFrame with required columns."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)

def extract_shot_data(shot, ods_path):
    """Extract key parameters from OMAS data for a single shot, and check kfile/gfile/omas file existence."""
    try:
        if not os.path.exists(ods_path):
            logger.error(f"ODS file not found for shot {shot}: {ods_path}")
            return None
        
        logger.info(f"Loading ODS file for shot {shot} from {ods_path}")
        ods = load_omas_json(ods_path)
        
        # Extract information

        pulse_duration = find_pulse_duration(ods) * 1000  # convert to ms
        breakdown_onset = find_breakdown_onset(ods) * 1000  # convert to ms
        max_ip = find_max_ip(ods) / 1000  # convert to kA
        bt = find_bt(ods)
        # shotclass = vaft.omas.find_shotclass(ods)

        # Validate data
        for val, name in zip([pulse_duration, breakdown_onset, max_ip, bt], ['pulse_duration', 'breakdown_onset', 'max_ip', 'bt']):
            if val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, (int, float)) and val < 0):
                logger.warning(f"Invalid value for {name} in shot {shot}: {val}")
                return None

        # round the values to 2 decimal places
        pulse_duration = round(pulse_duration, 2)
        breakdown_onset = round(breakdown_onset, 2)
        max_ip = round(max_ip, 2)
        bt = round(bt, 3)

        # --- 추가: kfile/gfile 개수 및 omas 주요 파일 존재여부 체크 ---
        shot_dir = os.path.dirname(os.path.dirname(ods_path))  # .../public/{shotnumber}
        efit_kfile_dir = os.path.join(shot_dir, 'efit', 'kfile')
        efit_gfile_dir = os.path.join(shot_dir, 'efit', 'gfile')
        chease_gfile_dir = os.path.join(shot_dir, 'chease', 'gfile')
        omas_dir = os.path.join(shot_dir, 'omas')
        shot_str = str(shot).zfill(6)
        efit_kfile_count = len([f for f in os.listdir(efit_kfile_dir) if f.startswith(f'k{shot_str}')]) if os.path.isdir(efit_kfile_dir) else 0
        efit_gfile_count = len([f for f in os.listdir(efit_gfile_dir) if f.startswith(f'g{shot_str}')]) if os.path.isdir(efit_gfile_dir) else 0
        chease_gfile_count = len([f for f in os.listdir(chease_gfile_dir) if f.startswith(f'g{shot_str}')]) if os.path.isdir(chease_gfile_dir) else 0
        def omas_file_exists(suffix):
            return os.path.isfile(os.path.join(omas_dir, f"{shot}_{suffix}.json"))
        has_diagnostics_ods = omas_file_exists('diagnostics')
        has_eddy_ods = omas_file_exists('eddy')
        has_efit_ods = omas_file_exists('efit')
        has_chease_ods = omas_file_exists('chease')
        has_combined_ods = os.path.isfile(os.path.join(omas_dir, f"{shot}_combined.json"))
        # ------------------------------------------------------
        logger.info(f"Successfully extracted data for shot {shot}:")
        logger.info(f"  - Pulse duration: {pulse_duration} ms")
        logger.info(f"  - Breakdown onset: {breakdown_onset} ms")
        logger.info(f"  - Max IP: {max_ip} kA")
        logger.info(f"  - Toroidal field: {bt} T")
        logger.info(f"  - efit_kfile_count: {efit_kfile_count}")
        logger.info(f"  - efit_gfile_count: {efit_gfile_count}")
        logger.info(f"  - has_diagnostics_ods: {has_diagnostics_ods}")
        logger.info(f"  - has_eddy_ods: {has_eddy_ods}")
        logger.info(f"  - has_efit_ods: {has_efit_ods}")
        logger.info(f"  - has_chease_ods: {has_chease_ods}")
        logger.info(f"  - has_combined_ods: {has_combined_ods}")
        # logger.info(f"  - Shot class: {shotclass}")
        
        return {
            'shot': shot,
            # 'shot_class': shotclass,
            'plasma_onset_time': breakdown_onset,
            'pulse_duration': pulse_duration,
            'max_plasma_current': max_ip,
            'toroidal_field': bt,
            'efit_kfile_count': efit_kfile_count,
            'efit_gfile_count': efit_gfile_count,
            'chease_gfile_count': chease_gfile_count,
            'has_diagnostics_ods': has_diagnostics_ods,
            'has_eddy_ods': has_eddy_ods,
            'has_efit_ods': has_efit_ods,
            'has_chease_ods': has_chease_ods,
            'has_combined_ods': has_combined_ods
        }
    except Exception as e:
        logger.error(f"Error processing shot {shot}: {str(e)}")
        return None

def generate_omas_history_excel(base_path=None, max_rows=10):
    """Main function to generate OMAS history Excel file.
    
    Args:
        base_path (str, optional): Base directory path to search for ODS files.
                                 Defaults to DEFAULT_BASE_PATH.
        max_rows (int or None, optional): Maximum number of rows to process. If None, process all. Defaults to 10.
    """
    try:
        # Step 1: Load or create Excel file
        df = load_or_create_excel()
        processed_shots = set(df['shot'].tolist())
        
        # Step 2: Find all ODS files with the specified base path
        ods_files = find_ods_files(base_path)
        
        # Step 3: Filter unprocessed shots
        unprocessed_shots = [(shot, path) for shot, path in ods_files if shot not in processed_shots]
        
        if not unprocessed_shots:
            logger.info("No new shots to process.")
            return
        
        # Limit the number of shots to process if max_rows is not None
        if max_rows is not None:
            unprocessed_shots = unprocessed_shots[:max_rows]
        logger.info(f"Processing {len(unprocessed_shots)} shots" + (f" (limited to {max_rows} rows)" if max_rows is not None else " (processing all)"))
        
        # Step 4: Process each unprocessed shot in parallel
        with multiprocessing.Pool() as pool:
            results = pool.starmap(extract_shot_data, unprocessed_shots)
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            df = pd.concat([df, pd.DataFrame(valid_results)], ignore_index=True)
            df = df.sort_values('shot').reset_index(drop=True)
            df.to_excel(EXCEL_FILENAME, index=False)
            logger.info(f"Successfully processed {len(valid_results)} shots.")
        else:
            logger.info("No valid shots processed.")
        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    # change path to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import argparse
    def none_or_int(val):
        if val is None or val == 'None':
            return None
        return int(val)
    parser = argparse.ArgumentParser(description='Generate OMAS history Excel file')
    parser.add_argument('--base-path', type=str, 
                       default=DEFAULT_BASE_PATH,
                       help='Base directory path to search for ODS files')
    parser.add_argument('--max-rows', type=none_or_int,
                       default=None, # for all shots
                    #    default=10, # for test
                       help='Maximum number of rows to process (use None for all)')
    args = parser.parse_args()
    generate_omas_history_excel(args.base_path, args.max_rows)

