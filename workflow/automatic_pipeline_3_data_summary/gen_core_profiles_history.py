import os
import pandas as pd
import numpy as np
import vaft
from vaft.database import ods as db_ods
from vaft.omas import formula_wrapper
from vaft.omas.general import find_matching_time_indices
import logging
from tqdm import tqdm
from pathlib import Path

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
EXPECTED_COLUMNS = [
    "shot",
    "time",
    "Ip_MA",
    "Bt_T",
    "Ploss_MW",
    "tauE_s",
    "tauE_IPB89",
    "tauE_H98y2",
    "tauE_NSTX",
    "tauE_NSTX2006L",
    "tauE_Kurskiev2022",
    "ne_19m3",
    "ne_line_19m3",
    "ne_vol_19m3",
    "T_eV",
    "R_m",
    "epsilon",
    "kappa",
]
KEY_COLUMNS = ["shot", "time"]
SORT_COLUMNS = ["shot", "time"]
REQUIRED_COLUMNS_FOR_REPAIR = [
    "Ip_MA",
    "Bt_T",
    "Ploss_MW",
    "tauE_s",
    "ne_19m3",
    "ne_line_19m3",
    "ne_vol_19m3",
    "T_eV",
]


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
                    n_e_line_avg = float(eng_params.get('n_e_line_avg', eng_params.get('n_e', np.nan)))  # [m^-3]
                    n_e_vol_avg = float(eng_params.get('n_e_vol_avg', np.nan))  # [m^-3]
                    if not np.isfinite(n_e_line_avg) or n_e_line_avg <= 0:
                        raise ValueError(f"Invalid n_e_line_avg: {n_e_line_avg}")
                    if not np.isfinite(n_e_vol_avg) or n_e_vol_avg <= 0:
                        raise ValueError(f"Invalid n_e_vol_avg: {n_e_vol_avg}")
                    ne_19m3 = n_e_line_avg / 1e19  # [10^19 m^-3] (legacy compatibility)
                    ne_line_19m3 = n_e_line_avg / 1e19
                    ne_vol_19m3 = n_e_vol_avg / 1e19
                    if 'electrons.temperature' not in cp_ts:
                        raise ValueError(
                            f"electrons.temperature not found in core_profiles.profiles_1d[{cp_idx_matched}]"
                        )
                    # Keep the source convention aligned with ne extraction: use the matched core_profile slice.
                    T_eV = float(np.nanmean(np.asarray(cp_ts['electrons.temperature'], dtype=float)))
                    if not np.isfinite(T_eV) or T_eV <= 0:
                        raise ValueError(f"Invalid T_eV: {T_eV}")
                    
                except Exception as e:
                    logger.warning(f"Shot {shot_number}, cp_idx {cp_idx}, eq_idx {eq_idx}: Failed to compute engineering parameters: {e}")
                    continue
                
                # Compute confinement time parameters using compute_confiment_time_paramters
                tauE_IPB89 = np.nan
                tauE_H98y2 = np.nan
                tauE_NSTX = np.nan
                tauE_NSTX2006L = np.nan
                tauE_Kurskiev2022 = np.nan
                tauE_s = np.nan
                H_factor = np.nan
                try:
                    (
                        tauE_IPB89,
                        tauE_H98y2,
                        tauE_NSTX,
                        tauE_NSTX2006L,
                        tauE_Kurskiev2022,
                        H_factor,
                        tauE_s,
                    ) = formula_wrapper.compute_confiment_time_paramters(
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
                    'tauE_NSTX2006L': tauE_NSTX2006L,
                    'tauE_Kurskiev2022': tauE_Kurskiev2022,
                    'ne_19m3': ne_19m3,
                    'ne_line_19m3': ne_line_19m3,
                    'ne_vol_19m3': ne_vol_19m3,
                    'T_eV': T_eV,
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


def _load_existing_or_empty(output_path, expected_columns):
    path = Path(output_path)
    if not path.exists():
        return pd.DataFrame(columns=expected_columns)

    try:
        existing_df = pd.read_excel(path)
        for col in expected_columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        return existing_df
    except Exception as exc:
        logger.warning(f"Failed to read existing Excel {output_path}: {exc}. Starting from empty.")
        return pd.DataFrame(columns=expected_columns)


def _has_invalid_values(df, columns):
    if df.empty:
        return True
    for col in columns:
        if col not in df.columns:
            return True
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            return True
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            return True
    return False


def _get_target_shots(candidate_shots, existing_df, required_columns):
    if existing_df.empty or "shot" not in existing_df.columns:
        return sorted({int(s) for s in candidate_shots}), 0, len(candidate_shots), 0

    existing_shots = {
        int(s)
        for s in pd.to_numeric(existing_df["shot"], errors="coerce").dropna().astype(int).tolist()
    }
    missing_shots = [int(s) for s in candidate_shots if int(s) not in existing_shots]

    defective_shots = []
    for shot in sorted(existing_shots.intersection({int(s) for s in candidate_shots})):
        shot_df = existing_df[existing_df["shot"] == shot]
        if _has_invalid_values(shot_df, required_columns):
            defective_shots.append(int(shot))

    target_shots = sorted(set(missing_shots + defective_shots))
    completed_count = len(candidate_shots) - len(target_shots)
    return target_shots, completed_count, len(missing_shots), len(defective_shots)


def _merge_upsert(existing_df, new_rows_df, key_columns, sort_columns, expected_columns):
    if existing_df.empty:
        merged = new_rows_df.copy()
    elif new_rows_df.empty:
        merged = existing_df.copy()
    else:
        merged = pd.concat([existing_df, new_rows_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_columns, keep="last")

    for col in expected_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[expected_columns]
    if not merged.empty:
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
    return merged


def _save_excel(df, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def generate_core_profiles_history_excel(
    max_shots=None,
    Z_eff=DEFAULT_Z_EFF,
    output_path=None,
    directory="public",
    rebuild=False,
    save_every=10,
):
    """Main function to generate core profiles history Excel file.
    
    Args:
        max_shots (int, optional): Maximum number of shots to process. Defaults to None (process all).
        Z_eff (float, optional): Effective charge number. Defaults to 2.0.
        output_path (str, optional): Path for the output Excel file.
            If None, saves next to this script.
    """
    logger.info("Starting core profiles history generation")
    if output_path is None:
        output_path = str(Path(__file__).with_name(OUTPUT_FILENAME))
    
    # Step 1: Get core profile shots
    shot_numbers = get_core_profile_shots()
    
    if not shot_numbers:
        logger.warning("No core profile shots found to process.")
        return
    
    # Limit number of shots if specified
    if max_shots is not None:
        shot_numbers = shot_numbers[:max_shots]
        logger.info(f"Limited processing to {max_shots} shots")
    
    existing_df = (
        pd.DataFrame(columns=EXPECTED_COLUMNS)
        if rebuild
        else _load_existing_or_empty(output_path, EXPECTED_COLUMNS)
    )
    target_shots, completed_count, missing_count, defective_count = _get_target_shots(
        shot_numbers, existing_df, REQUIRED_COLUMNS_FOR_REPAIR
    )
    logger.info(
        "Core-profile sheet candidates=%d, already-complete=%d, missing=%d, defective=%d, to-process=%d",
        len(shot_numbers),
        completed_count,
        missing_count,
        defective_count,
        len(target_shots),
    )
    if not target_shots:
        logger.info("No missing or defective shots found. Re-saving existing sheet.")
        if not existing_df.empty:
            existing_df = existing_df.sort_values(SORT_COLUMNS).reset_index(drop=True)
        _save_excel(existing_df, output_path)
        return existing_df

    logger.info(f"Processing {len(target_shots)} shots")

    working_df = existing_df.copy()
    save_every = max(1, int(save_every))
    processed = 0
    for shot_number in tqdm(target_shots, desc="Processing shots"):
        try:
            # Load ODS from database
            logger.info(f"Loading ODS for shot {shot_number}...")
            ods = db_ods.load(int(shot_number), directory=directory)
            
            # Extract parameters
            results = extract_confinement_parameters(ods, shot_number, Z_eff=Z_eff)
            if not results:
                logger.warning(f"Shot {shot_number}: no rows extracted, keeping existing rows.")
                continue
            shot_df = pd.DataFrame(results)
            working_df = _merge_upsert(working_df, shot_df, KEY_COLUMNS, SORT_COLUMNS, EXPECTED_COLUMNS)
            processed += 1
            if processed % save_every == 0:
                _save_excel(working_df, output_path)
                logger.info(f"Checkpoint saved after {processed} processed shots -> {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing shot {shot_number}: {e}")
            continue
    
    # Step 3: Final save
    if working_df.empty:
        logger.warning("No data was successfully processed.")
        return None
    working_df = working_df.sort_values(SORT_COLUMNS).reset_index(drop=True)
    _save_excel(working_df, output_path)
    logger.info(
        "Processing complete! Current table has %d rows. Saved to %s",
        len(working_df),
        output_path,
    )
    return working_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate core profiles history Excel file')
    parser.add_argument('--max-shots', type=int,
                       help='Maximum number of shots to process')
    parser.add_argument('--Z-eff', type=float,
                       default=DEFAULT_Z_EFF,
                       help=f'Effective charge number (default: {DEFAULT_Z_EFF})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output Excel file path')
    parser.add_argument('--directory', type=str, default='public',
                       help='HDF5 directory (default: public)')
    parser.add_argument('--rebuild', action='store_true',
                       help='Ignore existing Excel and rebuild from scratch')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N processed shots')
    
    args = parser.parse_args()
    generate_core_profiles_history_excel(
        max_shots=args.max_shots,
        Z_eff=args.Z_eff,
        output_path=args.output,
        directory=args.directory,
        rebuild=args.rebuild,
        save_every=args.save_every,
    )
