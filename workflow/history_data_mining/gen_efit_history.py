# This Python function aims to read g, m, a files from the EFIT directory for each shot number and extract various information from them.
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
OUTPUT_FILENAME = "efit_history.xlsx"

def find_efit_files(base_path=None):
    """Find all EFIT files in the specified directory structure.
    
    Args:
        base_path (str, optional): Base directory path to search for EFIT files. 
                                 Defaults to "/srv/vest.filedb/public".
    """
    if base_path is None:
        base_path = "/srv/vest.filedb/public"
    
    efit_files = []
    
    # Find all directories that match the pattern
    shot_dirs = glob.glob(os.path.join(base_path, "[0-9]" * 5))
    print(f"Found {len(shot_dirs)} shot directories in {base_path}")
    
    for shot_dir in shot_dirs:
        shot_number = os.path.basename(shot_dir)
        # Pad shot number with leading zeros to 6 digits
        padded_shot = f"{int(shot_number):06d}"
        efit_dir = os.path.join(shot_dir, "efit")
        
        if os.path.exists(efit_dir):
            print(f"\nChecking shot {shot_number} (padded: {padded_shot})...")
            print(f"EFIT directory: {efit_dir}")
            
            # Check gfile directory
            gfile_dir = os.path.join(efit_dir, "gfile")
            if os.path.exists(gfile_dir):
                print(f"Gfile directory exists: {gfile_dir}")
                # List all files in the directory for debugging
                all_files = os.listdir(gfile_dir)
                print(f"All files in gfile directory: {all_files}")
                
                # Use padded shot number in pattern
                gfiles = glob.glob(os.path.join(gfile_dir, f"g{padded_shot}.[0-9][0-9][0-9][0-9][0-9]"))
                print(f"Found {len(gfiles)} gfiles matching pattern")
                
                for gfile in gfiles:
                    # Extract time from filename (e.g., g041660.00328 -> 00328)
                    time_str = os.path.basename(gfile).split('.')[1]
                    # Construct exact filenames with padded shot number
                    mfile = os.path.join(efit_dir, "mfile", f"m{padded_shot}.{time_str}")
                    afile = os.path.join(efit_dir, "afile", f"a{padded_shot}.{time_str}")
                    
                    print(f"\nChecking time {time_str}:")
                    print(f"Gfile: {gfile}")
                    print(f"Mfile: {mfile}")
                    print(f"Afile: {afile}")
                    
                    if os.path.exists(mfile) and os.path.exists(afile):
                        print("All files exist!")
                        efit_files.append({
                            'shot': int(shot_number),
                            'time': int(time_str),
                            'gfile': gfile,
                            'mfile': mfile,
                            'afile': afile
                        })
                    else:
                        print("Missing mfile or afile")
            else:
                print(f"Gfile directory does not exist: {gfile_dir}")
    
    print(f"\nTotal EFIT file sets found: {len(efit_files)}")
    return efit_files


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
    

def extract_efit_data(efit_set):
    """Extract key parameters from EFIT files for a single time point.
    
    Args:
        efit_set (dict): Dictionary containing paths to g, m, and a files
        
    Returns:
        dict: Extracted parameters or None if file is invalid
    """
    try:
        # Load EFIT files and convert to OMAS data structure
        gfile = OMFITgeqdsk(efit_set['gfile'])
        ods = gfile.to_omas()
        vaft.omas.update.update_equilibrium_boundary(ods)
        equilibrium = ods['equilibrium']['time_slice'][0]
        r_major = equilibrium['boundary.geometric_axis.r']
        a = equilibrium['boundary.minor_radius']
        bt = ods['equilibrium']['vacuum_toroidal_field']['b0'] * ods['equilibrium']['vacuum_toroidal_field']['r0'] / r_major
        bt = bt[0]
        ip = round_sig(equilibrium['global_quantities']['ip']/1000, 3)
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
            'shot_number': efit_set['shot'],
            'time [ms]': efit_set['time'],
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
        # print(f"\nSuccessfully extracted EFIT data for shot {efit_set['shot']} at time {efit_set['time']}:")
        # print("="*50)
        # print("Basic Information:")
        # print(f"  Shot Number: {data['shot_number']}")
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
        print(f"\nError processing shot {efit_set['shot']} at time {efit_set['time']}:")
        print(f"Error message: {str(e)}")
        print(f"Skipping this file...")
        return None

def process_batch(batch, base_path=None):
    """Process a batch of EFIT files.
    
    Args:
        batch (list): List of EFIT file sets to process
        base_path (str, optional): Base directory path
        
    Returns:
        list: List of successfully processed data dictionaries
    """
    results = []
    for efit_set in batch:
        try:
            data = extract_efit_data(efit_set)
            if data is not None:
                results.append(data)
        except Exception as e:
            logger.error(f"Error processing shot {efit_set['shot']} at time {efit_set['time']}: {str(e)}")
    return results

def generate_efit_history_excel(base_path=None, max_files=None, num_processes=20):
    """Main function to generate EFIT history Excel file.
    
    Args:
        base_path (str, optional): Base directory path to search for EFIT files.
                                 Defaults to DEFAULT_BASE_PATH.
        max_files (int, optional): Maximum number of files to process. Defaults to None (process all).
        num_processes (int, optional): Number of processes to use. Defaults to 20.
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
    
    logger.info(f"Starting processing with {num_processes} processes")
    
    # Step 1: Find all EFIT files
    efit_files = find_efit_files(base_path)
    
    if not efit_files:
        logger.info("No EFIT files found to process.")
        return
    
    # Limit number of files if specified
    if max_files is not None:
        efit_files = efit_files[:max_files]
        logger.info(f"Limited processing to {max_files} files")
    
    # Step 2: Split files into batches
    batches = [efit_files[i:i + BATCH_SIZE] for i in range(0, len(efit_files), BATCH_SIZE)]
    logger.info(f"Split {len(efit_files)} files into {len(batches)} batches of size {BATCH_SIZE}")
    
    # Step 3: Process batches in parallel
    all_data = []
    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_batch, base_path=base_path)
        for batch_results in tqdm(pool.imap(process_func, batches), total=len(batches), desc="Processing batches"):
            all_data.extend(batch_results)
    
    # Step 4: Create DataFrame and save to Excel
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['shot_number', 'time [ms]'])
        df.to_excel(OUTPUT_FILENAME, index=False)
        logger.info(f"Processing complete! Successfully processed {len(all_data)} out of {len(efit_files)} files.")
        logger.info(f"Saved to {OUTPUT_FILENAME}")
    else:
        logger.warning("No data was successfully processed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate EFIT history Excel file')
    parser.add_argument('--base-path', type=str, 
                       default=DEFAULT_BASE_PATH,
                       help='Base directory path to search for EFIT files')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--num-processes', type=int,
                       default=20,
                       help='Number of processes to use for parallel processing (default: 20)')
    parser.add_argument('--batch-size', type=int,
                       default=BATCH_SIZE,
                       help=f'Number of files to process in each batch (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    generate_efit_history_excel(args.base_path, args.max_files, args.num_processes)
