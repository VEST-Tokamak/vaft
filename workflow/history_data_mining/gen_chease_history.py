# This Python script processes chease files in parallel using multiprocessing to extract equilibrium data.
# It reads g, m, a files from the chease directory for each shot number and extracts various information.
# The script uses batch processing to improve performance and includes progress tracking.
#
# Key features:
# - Parallel processing with configurable number of processes
# - Batch processing with configurable batch size
# - Progress tracking with tqdm
# - Comprehensive logging
# - Configurable file limits and processing parameters
#
# The following information can be extracted:
# 1. File name
# 2. Shot number
# 3. Time
# 4. Normalized time
# 5. Plasma current
# 6. Major radius
# 7. Minor radius
# 8. Elongation
# 9. Triangularity
# 10. Safety factor (q)
# 11. Magnetic axis position (R, Z)
# 12. X-point positions
# 13. Separatrix shape
# 14. Plasma boundary
# 15. Magnetic flux values
# 16. Poloidal beta
# 17. Internal inductance
# 18. Plasma stored energy
# 19. Magnetic field at the magnetic axis
# 20. Plasma volume
# 21. Plasma surface area
# 22. Last closed flux surface (LCFS) parameters
# 23. Strike point positions
# 24. Divertor configuration
# 25. Plasma shape parameters (kappa, delta)

import os
import glob
import pandas as pd
import numpy as np
import vaft
from omfit_classes.omfit_eqdsk import OMFITeqdsk, OMFITgeqdsk # dependency issue with scipy.integrate.cumtrapz in omfit-classes.eqdsk 
# from vaft.code.omfit_eqdsk import OMFITeqdsk # forked version of omfit-classes.eqdsk with scipy.integrate.cumulative_trapezoid
# from vaft.code.omfit_eqdsk import OMFITgeqdsk # forked version of omfit-classes.geqdsk with scipy.integrate.cumulative_trapezoid
import multiprocessing as mp
from functools import partial
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BASE_PATH = "/srv/vest.filedb/public"
BATCH_SIZE = 20  # Number of files to process in each batch
OUTPUT_FILENAME = "chease_history.xlsx"

def find_chease_files(base_path=None):
    """Find all chease files in the specified directory structure.
    
    Args:
        base_path (str, optional): Base directory path to search for chease files. 
                                 Defaults to "/srv/vest.filedb/public".
    """
    if base_path is None:
        base_path = "/srv/vest.filedb/public"
    
    chease_files = []
    
    # Find all directories that match the pattern
    shot_dirs = glob.glob(os.path.join(base_path, "[0-9]" * 5))
    print(f"Found {len(shot_dirs)} shot directories in {base_path}")
    
    for shot_dir in shot_dirs:
        shot = os.path.basename(shot_dir)
        # Pad shot number with leading zeros to 6 digits
        padded_shot = f"{int(shot):06d}"
        chease_dir = os.path.join(shot_dir, "chease")
        plots_dir = os.path.join(chease_dir, "plots")
        
        if os.path.exists(chease_dir) and os.path.exists(plots_dir):
            print(f"\nChecking shot {shot} (padded: {padded_shot})...")
            print(f"chease directory: {chease_dir}")
            print(f"plots directory: {plots_dir}")
            
            # Find all .png files in the plots directory
            plot_files = glob.glob(os.path.join(plots_dir, "*.png"))
            print(f"Found {len(plot_files)} .png files in {plots_dir}")
            
            for plot_file in plot_files:
                # Extract time from png filename (e.g., 00328.png -> 00328)
                time_str = os.path.splitext(os.path.basename(plot_file))[0]
                
                # Construct the gfile path
                gfile_path = os.path.join(chease_dir, f"g{padded_shot}.{time_str}")
                
                print(f"\nChecking for gfile corresponding to plot {plot_file}:")
                print(f"Expected Gfile: {gfile_path}")
                
                if os.path.exists(gfile_path):
                    print(f"Found Gfile: {gfile_path}")
                    chease_files.append({
                        'shot': int(shot),
                        'time': int(time_str),
                        'gfile': gfile_path
                    })
                else:
                    print(f"Gfile not found: {gfile_path}")
                            
    print(f"\nTotal chease file sets found based on existing plots: {len(chease_files)}")
    return chease_files


def round_sig(x, sig=3):
    """
    round_sig x to 'sig' significant figures. Works with scalars and numpy arrays.
    """
    scalar_input = np.isscalar(x)
    x = np.asarray(x)
    def _round_elem(val):
        if val == 0:
            return 0
        else:
            decimals = sig - 1 - int(np.floor(np.log10(abs(val))))
            return np.round(val, decimals)
    result = np.vectorize(_round_elem)(x)
    if scalar_input:
        return result.item()
    return result
    

def extract_chease_data(chease_set):
    """Extract key parameters from chease files for a single time point.
    
    Args:
        chease_set (dict): Dictionary containing paths to g, m, and a files
        
    Returns:
        dict: Extracted parameters or None if file is invalid
    """
    try:
        # Load chease files and convert to OMAS data structure
        gfile = OMFITgeqdsk(chease_set['gfile'])
        ods = gfile.to_omas()
        vaft.omas.update.update_equilibrium_boundary(ods)
        equilibrium = ods['equilibrium']['time_slice'][0]
        r_major = equilibrium['boundary.geometric_axis.r']
        a = equilibrium['boundary.minor_radius']
        bt = ods['equilibrium']['vacuum_toroidal_field']['b0'] * ods['equilibrium']['vacuum_toroidal_field']['r0'] / r_major
        bt = bt[0]
        ip = np.abs(round_sig(equilibrium['global_quantities']['ip']/1000, 3))
        aspect = r_major / a
        volume = equilibrium['profiles_1d']['volume'][-1]
        surface_area = volume / r_major / 2 / np.pi
        try:
            elongation = equilibrium['boundary.elongation']
        except Exception:
            elongation = np.nan
        # Triangularity, q95, magnetic axis, beta_poloidal 등 추가 추출
        try:
            triangularity = equilibrium['boundary.triangularity']
        except Exception:
            triangularity = np.nan
        try:
            q_axis = equilibrium['global_quantities']['q_axis']
        except Exception:
            q_axis = np.nan
        try:
            q_min = equilibrium['global_quantities']['q_min']['value']
        except Exception:
            q_min = np.nan
        try:
            q95 = equilibrium['global_quantities.q_95']
        except Exception:
            q95 = np.nan
        try:
            magnetic_axis_r = equilibrium['global_quantities.magnetic_axis.r']
        except Exception:
            magnetic_axis_r = np.nan
        try:
            magnetic_axis_z = equilibrium['global_quantities.magnetic_axis.z']
        except Exception:
            magnetic_axis_z = np.nan
        try:
            beta_poloidal = equilibrium['global_quantities']['beta_pol']
        except Exception:
            beta_poloidal = np.nan
        try:
            beta_toroidal = equilibrium['global_quantities']['beta_tor']
        except Exception:
            beta_toroidal = np.nan
        try:
            beta_normal = equilibrium['global_quantities']['beta_normal']
        except Exception:
            beta_normal = np.nan
        try:
            li_3 = equilibrium['global_quantities']['li_3']
        except Exception:
            li_3 = np.nan
        try:
            energy_mhd = equilibrium['global_quantities']['energy_mhd']
        except Exception:
            energy_mhd = np.nan
        # Strike point positions
        # 데이터 딕셔너리 구성
        data = {
            'shot': chease_set['shot'],
            'time [ms]': chease_set['time'],
            'ip [kA]': ip,
            'major_radius [m]': r_major,
            'minor_radius [m]': a,
            'aspect_ratio': aspect,
            'elongation': elongation,
            'triangularity': triangularity,
            'q_axis': q_axis,
            'q_min': q_min,
            'q95': q95,
            'magnetic_axis_r [m]': magnetic_axis_r,
            'magnetic_axis_z [m]': magnetic_axis_z,
            'beta_poloidal': beta_poloidal,
            'b_field_tor_axis [T]': bt,
            'plasma_volume [m^3]': volume,
            'plasma_surface_area [m^2]': surface_area,
            'beta_toroidal': beta_toroidal,
            'beta_normal': beta_normal,
            'li_3': li_3,
            'energy_mhd': energy_mhd
            }
        # # Print all extracted information (주석 처리)
        # print(f"\nSuccessfully extracted chease data for shot {chease_set['shot']} at time {chease_set['time']}:")
        # print("="*50)
        # print("Basic Information:")
        # print(f"  Shot Number: {data['shot']}")
        # print(f"  Time: {data['time']} ms")
        # print("\nPlasma Parameters:")
        # print(f"  Plasma Current: {data['ip']} kA")
        # print(f"  Major Radius: {data['major_radius']} m")
        # print(f"  Minor Radius: {data['minor_radius']} m")
        # print(f"  Elongation: {data['elongation']}")
        # print(f"  Triangularity: {data['triangularity']}")
        # print(f"  q95: {data['q95']}")
        # print("\nMagnetic Axis Position:")
        # print(f"  R: {data['magnetic_axis_r']} m")
        # print(f"  Z: {data['magnetic_axis_z']} m")
        # print("\nGlobal Parameters:")
        # print(f"  Beta Poloidal: {data['beta_poloidal']}")
        # print(f"  Torodial B-field at Axis: {data['b_field_tor_axis']} T")
        # print("\nGeometric Parameters:")
        # print(f"  Plasma Volume: {data['plasma_volume']} m^3")
        # print(f"  Plasma Surface: {data['plasma_surface_area']} m^2")
        # print("\nStrike Points:")
        # print(f"  Inner Strike Point (R,Z): ({data['strike_r_inner']}, {data['strike_z_inner']}) m")
        # print(f"  Outer Strike Point (R,Z): ({data['strike_r_outer']}, {data['strike_z_outer']}) m")
        # print("="*50)
        return data
    except (ValueError, KeyError, IndexError, Exception) as e:
        print(f"\nError processing shot {chease_set['shot']} at time {chease_set['time']}:")
        print(f"Error message: {str(e)}")
        print(f"Skipping this file...")
        return None

def process_batch(batch, base_path=None):
    """Process a batch of chease files.
    
    Args:
        batch (list): List of chease file sets to process
        base_path (str, optional): Base directory path
        
    Returns:
        list: List of successfully processed data dictionaries
    """
    results = []
    for chease_set in batch:
        try:
            data = extract_chease_data(chease_set)
            if data is not None:
                results.append(data)
        except Exception as e:
            logger.error(f"Error processing shot {chease_set['shot']} at time {chease_set['time']}: {str(e)}")
    return results

def generate_chease_history_excel(base_path=None, max_files=None, num_processes=20):
    """Main function to generate chease history Excel file.
    
    Args:
        base_path (str, optional): Base directory path to search for chease files.
                                 Defaults to DEFAULT_BASE_PATH.
        max_files (int, optional): Maximum number of files to process. Defaults to None (process all).
        num_processes (int, optional): Number of processes to use. Defaults to 20.
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
    
    logger.info(f"Starting processing with {num_processes} processes")
    
    # Step 1: Find all chease files
    chease_files = find_chease_files(base_path)
    
    if not chease_files:
        logger.info("No chease files found to process.")
        return
    
    # Limit number of files if specified
    if max_files is not None:
        chease_files = chease_files[:max_files]
        logger.info(f"Limited processing to {max_files} files")
    
    # Step 2: Split files into batches
    batches = [chease_files[i:i + BATCH_SIZE] for i in range(0, len(chease_files), BATCH_SIZE)]
    logger.info(f"Split {len(chease_files)} files into {len(batches)} batches of size {BATCH_SIZE}")
    
    # Step 3: Process batches in parallel
    all_data = []
    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_batch, base_path=base_path)
        for batch_results in tqdm(pool.imap(process_func, batches), total=len(batches), desc="Processing batches"):
            all_data.extend(batch_results)
    
    # Step 4: Create DataFrame and save to Excel
    if all_data:
        df = pd.DataFrame(all_data)
        # Sort by shot number and time
        df = df.sort_values(['shot', 'time [ms]'])
        # Save to Excel
        df.to_excel(OUTPUT_FILENAME, index=False)
        logger.info(f"Processing complete! Successfully processed {len(all_data)} out of {len(chease_files)} files.")
        logger.info(f"Saved to {OUTPUT_FILENAME}")
    else:
        logger.warning("No data was successfully processed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate chease history Excel file')
    parser.add_argument('--base-path', type=str, 
                       default=DEFAULT_BASE_PATH,
                       help='Base directory path to search for chease files')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--num-processes', type=int,
                       default=20,
                       help='Number of processes to use for parallel processing (default: 20)')
    parser.add_argument('--batch-size', type=int,
                       default=BATCH_SIZE,
                       help=f'Number of files to process in each batch (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size  # Update batch size if specified
    generate_chease_history_excel(args.base_path, args.max_files, args.num_processes)