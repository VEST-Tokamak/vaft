import os
import pandas as pd
import numpy as np
import vaft
from vaft.database import ods as db_ods
from vaft.omas import formula_wrapper
from vaft.omas.general import find_matching_time_indices
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_FILENAME = "core_profiles_history.xlsx"
DEFAULT_Z_EFF = 2.0
DEFAULT_LN_LAMBDA = 17.0


def get_core_profile_shots():
    """Get list of shot numbers with core_profile status.
    
    Returns:
        list: List of shot numbers (integers)
    """
    df = db_ods.exist_ts_file()
    if df is None:
        logger.warning("No processed shots found.")
        return []
    
    core_profile_shots = df[df['Status'] == 'core_profile']['Shot Number'].values
    logger.info(f"Found {len(core_profile_shots)} shots with core_profile status")
    return core_profile_shots.tolist()


def extract_confinement_parameters(ods, shot_number, Z_eff=DEFAULT_Z_EFF):
    """Extract confinement time scaling parameters from a single ODS.
    
    Args:
        ods: OMAS data structure
        shot_number: Shot number (for logging)
        Z_eff: Effective charge number (default: 2.0)
    
    Returns:
        list: List of dictionaries, one per time slice with extracted parameters
    """
    results = []

    # update boundary-related quantities
    vaft.omas.update_equilibrium_boundary(ods)
    
    try:
        # Check if equilibrium and core_profiles exist
        if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
            logger.warning(f"Shot {shot_number}: No equilibrium time slices found")
            return results
        
        if 'core_profiles.profiles_1d' not in ods or len(ods['core_profiles.profiles_1d']) == 0:
            logger.warning(f"Shot {shot_number}: No core profiles found")
            return results
        
        # Process each core profile time slice and find matching equilibrium time slice
        n_cp = len(ods['core_profiles.profiles_1d'])
        logger.info(f"Shot {shot_number}: Processing {n_cp} core profile time slices")
        
        for cp_idx in range(n_cp):
            try:
                # Use find_matching_time_indices to ensure both core_profiles and equilibrium exist
                # and have matching times
                cp_idx_matched, eq_idx, time_val = find_matching_time_indices(ods, time_slice=cp_idx)
                
                # Get equilibrium time slice
                eq_ts = ods['equilibrium.time_slice'][eq_idx]
                
                # Get core profile time slice
                cp_ts = ods['core_profiles.profiles_1d'][cp_idx_matched]
                
                # Extract engineering parameters using compute_tau_E_engineering_parameters
                try:
                    eng_params = formula_wrapper.compute_tau_E_engineering_parameters(
                        ods, eq_idx, Z_eff=Z_eff, M=1.0
                    )
                    
                    # Extract parameters from dictionary
                    I_p = eng_params['I_p']  # [A]
                    Ip_MA = I_p / 1e6  # [MA]
                    Bt_T = eng_params['B_t']  # [T]
                    R_m = eng_params['R']  # [m]
                    epsilon = eng_params['epsilon']  # [-]
                    kappa = eng_params['kappa']  # [-]
                    P_loss = eng_params['P_loss']  # [W]
                    Ploss_MW = P_loss / 1e6  # [MW]
                    n_e_vol_avg = eng_params['n_e']  # [m^-3]
                    ne_19m3 = n_e_vol_avg / 1e19  # [10^19 m^-3]
                    
                except Exception as e:
                    logger.warning(f"Shot {shot_number}, cp_idx {cp_idx}, eq_idx {eq_idx}: Failed to compute engineering parameters: {e}")
                    continue
                
                # Compute confinement time parameters using compute_confiment_time_paramters
                tauE_IPB89 = np.nan
                tauE_H98y2 = np.nan
                tauE_NSTX = np.nan
                tauE_s = np.nan
                H_factor = np.nan
                try:
                    tauE_IPB89, tauE_H98y2, tauE_NSTX, H_factor, tauE_s = formula_wrapper.compute_confiment_time_paramters(
                        ods, eq_idx, Z_eff=Z_eff, M=1.0
                    )
                except Exception as e:
                    logger.warning(f"Shot {shot_number}, cp_idx {cp_idx}, eq_idx {eq_idx}: Failed to compute confinement time parameters: {e}")
                
                # Store results
                result = {
                    'shot': shot_number,
                    'time': time_val,
                    'Ip_MA': Ip_MA,
                    'Bt_T': Bt_T,
                    'Ploss_MW': Ploss_MW,
                    'tauE_s': tauE_s,
                    'tauE_IPB89': tauE_IPB89,
                    'tauE_H98y2': tauE_H98y2,
                    'tauE_NSTX': tauE_NSTX,
                    'ne_19m3': ne_19m3,
                    'R_m': R_m,
                    'epsilon': epsilon,
                    'kappa': kappa
                }
                results.append(result)
                
            except (KeyError, ValueError) as e:
                # Skip time slices where core_profiles and equilibrium don't match
                logger.debug(f"Shot {shot_number}, cp_idx {cp_idx}: Skipping - {type(e).__name__}: {e}")
                continue
            except Exception as e:
                logger.error(f"Shot {shot_number}, cp_idx {cp_idx}: Error extracting parameters: {e}")
                continue
        
        logger.info(f"Shot {shot_number}: Successfully extracted {len(results)} time slices")
        
    except Exception as e:
        logger.error(f"Shot {shot_number}: Error processing ODS: {e}")
    
    return results


def generate_core_profiles_history_excel(max_shots=None, Z_eff=DEFAULT_Z_EFF):
    """Main function to generate core profiles history Excel file.
    
    Args:
        max_shots (int, optional): Maximum number of shots to process. Defaults to None (process all).
        Z_eff (float, optional): Effective charge number. Defaults to 2.0.
    """
    logger.info("Starting core profiles history generation")
    
    # Step 1: Get core profile shots
    shot_numbers = get_core_profile_shots()
    
    if not shot_numbers:
        logger.warning("No core profile shots found to process.")
        return
    
    # Limit number of shots if specified
    if max_shots is not None:
        shot_numbers = shot_numbers[:max_shots]
        logger.info(f"Limited processing to {max_shots} shots")
    
    logger.info(f"Processing {len(shot_numbers)} shots")
    
    # Step 2: Process each shot
    all_data = []
    for shot_number in tqdm(shot_numbers, desc="Processing shots"):
        try:
            # Load ODS from database
            logger.info(f"Loading ODS for shot {shot_number}...")
            ods = db_ods.load(shot_number, directory='public')
            
            # Extract parameters
            results = extract_confinement_parameters(ods, shot_number, Z_eff=Z_eff)
            all_data.extend(results)
            
        except Exception as e:
            logger.error(f"Error processing shot {shot_number}: {e}")
            continue
    
    # Step 3: Create DataFrame and save to Excel
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['shot', 'time'])
        df.to_excel(OUTPUT_FILENAME, index=False)
        logger.info(f"Processing complete! Successfully processed {len(all_data)} time slices from {len(shot_numbers)} shots.")
        logger.info(f"Saved to {OUTPUT_FILENAME}")
        return df
    else:
        logger.warning("No data was successfully processed.")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate core profiles history Excel file')
    parser.add_argument('--max-shots', type=int,
                       help='Maximum number of shots to process')
    parser.add_argument('--Z-eff', type=float,
                       default=DEFAULT_Z_EFF,
                       help=f'Effective charge number (default: {DEFAULT_Z_EFF})')
    
    args = parser.parse_args()
    generate_core_profiles_history_excel(max_shots=args.max_shots, Z_eff=args.Z_eff)
