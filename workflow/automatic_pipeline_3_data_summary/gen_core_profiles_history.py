import os
import pandas as pd
import numpy as np
import vaft
from vaft.database import ods as db_ods
from vaft.omas import formula_wrapper
from vaft.omas.process_wrapper import compute_P_loss
from vaft.formula.equilibrium import confinement_time_from_P_loss_W_th
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
    
    try:
        # Check if equilibrium and core_profiles exist
        if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
            logger.warning(f"Shot {shot_number}: No equilibrium time slices found")
            return results
        
        if 'core_profiles.profiles_1d' not in ods or len(ods['core_profiles.profiles_1d']) == 0:
            logger.warning(f"Shot {shot_number}: No core profiles found")
            return results
        
        # Process each equilibrium time slice
        n_eq = len(ods['equilibrium.time_slice'])
        logger.info(f"Shot {shot_number}: Processing {n_eq} equilibrium time slices")
        
        for eq_idx in range(n_eq):
            try:
                eq_ts = ods['equilibrium.time_slice'][eq_idx]
                
                # Get time
                if 'time' in eq_ts:
                    time_val = float(eq_ts['time'])
                elif 'equilibrium.time' in ods and eq_idx < len(ods['equilibrium.time']):
                    time_val = float(ods['equilibrium.time'][eq_idx])
                else:
                    time_val = float(eq_idx)
                
                # Extract basic equilibrium parameters
                I_p = float(eq_ts['global_quantities.ip'])  # [A]
                Ip_MA = I_p / 1e6  # [MA]
                
                # Major and minor radius
                R_m = float(eq_ts['boundary.geometric_axis.r'])  # [m]
                a_m = float(eq_ts['boundary.minor_radius'])  # [m]
                
                # Toroidal magnetic field at geometric axis
                try:
                    B0 = float(ods['equilibrium.vacuum_toroidal_field.b0'])
                    R0 = float(ods['equilibrium.vacuum_toroidal_field.r0'])
                    Bt_T = B0 * R0 / R_m  # [T]
                except (KeyError, TypeError):
                    # Try alternative location
                    if isinstance(ods['equilibrium.vacuum_toroidal_field.b0'], (list, np.ndarray)):
                        B0 = float(ods['equilibrium.vacuum_toroidal_field.b0'][0])
                    else:
                        B0 = float(ods['equilibrium.vacuum_toroidal_field.b0'])
                    if isinstance(ods['equilibrium.vacuum_toroidal_field.r0'], (list, np.ndarray)):
                        R0 = float(ods['equilibrium.vacuum_toroidal_field.r0'][0])
                    else:
                        R0 = float(ods['equilibrium.vacuum_toroidal_field.r0'])
                    Bt_T = B0 * R0 / R_m  # [T]
                
                # Elongation
                try:
                    kappa = float(eq_ts['boundary.elongation'])
                except (KeyError, TypeError):
                    try:
                        kappa = float(eq_ts['global_quantities.elongation'])
                    except (KeyError, TypeError):
                        kappa = np.nan
                
                # Compute P_loss
                try:
                    P_loss = compute_P_loss(ods, eq_idx, Z_eff)  # [W]
                    Ploss_MW = P_loss / 1e6  # [MW]
                except Exception as e:
                    logger.warning(f"Shot {shot_number}, time_slice {eq_idx}: Failed to compute P_loss: {e}")
                    Ploss_MW = np.nan
                
                # Find matching core profile time slice
                cp_idx = None
                if 'core_profiles.profiles_1d' in ods:
                    min_time_diff = float('inf')
                    for idx in range(len(ods['core_profiles.profiles_1d'])):
                        cp_ts = ods['core_profiles.profiles_1d'][idx]
                        cp_time = float(cp_ts.get('time', idx))
                        time_diff = abs(cp_time - time_val)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            cp_idx = idx
                
                # Extract electron density (line-averaged or volume-averaged)
                ne_19m3 = np.nan
                if cp_idx is not None:
                    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
                    
                    # Try to get volume-averaged density
                    if 'global_quantities.density' in cp_ts:
                        n_e_vol_avg = float(cp_ts['global_quantities.density'])  # [m^-3]
                        ne_19m3 = n_e_vol_avg / 1e19  # [10^19 m^-3]
                    elif 'electrons.density' in cp_ts:
                        # Compute line-averaged or volume-averaged from profile
                        n_e_profile = np.asarray(cp_ts['electrons.density'], float)  # [m^-3]
                        # Use mean as approximation for line-averaged
                        n_e_avg = np.mean(n_e_profile)
                        ne_19m3 = n_e_avg / 1e19  # [10^19 m^-3]
                
                # Compute experimental confinement time
                tauE_s = np.nan
                if not np.isnan(Ploss_MW) and Ploss_MW > 0:
                    try:
                        # Get thermal energy
                        if 'global_quantities.energy_mhd' in eq_ts:
                            W_th = float(eq_ts['global_quantities.energy_mhd'])  # [J]
                        elif 'core_profiles.profiles_1d' in ods and cp_idx is not None:
                            cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
                            if 'global_quantities.energy' in cp_ts:
                                W_th = float(cp_ts['global_quantities.energy'])  # [J]
                            else:
                                W_th = np.nan
                        else:
                            W_th = np.nan
                        
                        if not np.isnan(W_th) and W_th > 0:
                            P_loss_W = Ploss_MW * 1e6  # Convert back to [W]
                            tauE_s = confinement_time_from_P_loss_W_th(P_loss_W, W_th)  # [s]
                    except Exception as e:
                        logger.warning(f"Shot {shot_number}, time_slice {eq_idx}: Failed to compute tauE: {e}")
                
                # Store results
                result = {
                    'shot': shot_number,
                    'time': time_val,
                    'Ip_MA': Ip_MA,
                    'Bt_T': Bt_T,
                    'Ploss_MW': Ploss_MW,
                    'tauE_s': tauE_s,
                    'ne_19m3': ne_19m3,
                    'R_m': R_m,
                    'a_m': a_m,
                    'kappa': kappa
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Shot {shot_number}, time_slice {eq_idx}: Error extracting parameters: {e}")
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
