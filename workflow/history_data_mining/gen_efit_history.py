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
    x = np.asarray(x)
    with np.errstate(divide='ignore'):
        return np.round_sig(x, sig - 1 - np.floor(np.log10(np.abs(x))).astype(int))
    

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
        
        equilibrium = ods['equilibrium']['time_slice'][0]
        a = (equilibrium['profiles_1d']['r_outboard'][-1] - equilibrium['profiles_1d']['r_inboard'][-1]) / 2
        r_major = (equilibrium['profiles_1d']['r_outboard'][-1] + equilibrium['profiles_1d']['r_inboard'][-1]) / 2
        bt = ods['equilibrium']['vacuum_toroidal_field']['b0'] * ods['equilibrium']['vacuum_toroidal_field']['r0'] / r_major
        ip = equilibrium['global_quantities']['ip']
        aspect = r_major / a
        volume = equilibrium['profiles_1d']['volume'][-1]
        area = volume / r_major / 2 / np.pi
        kappa = area / np.pi / a / a

        # Extract global parameters from OMAS data structure
        data = {        }
        
        # Get strike points
        x_points = ods['equilibrium.time_slice.0.boundary.x_point']
        if len(x_points) >= 2:
            data.update({
                'strike_r_inner': round_sig(x_points[0]['r'], 3),
                'strike_z_inner': round_sig(x_points[0]['z'], 3),
                'strike_r_outer': round_sig(x_points[1]['r'], 3),
                'strike_z_outer': round_sig(x_points[1]['z'], 3)
            })
        else:
            data.update({
                'strike_r_inner': np.nan,
                'strike_z_inner': np.nan,
                'strike_r_outer': np.nan,
                'strike_z_outer': np.nan
            })
        
        # Print all extracted information
        print(f"\nSuccessfully extracted EFIT data for shot {efit_set['shot']} at time {efit_set['time']}:")
        print("="*50)
        print("Basic Information:")
        print(f"  Shot Number: {data['shot_number']}")
        print(f"  Time: {data['time']} ms")
        print("\nPlasma Parameters:")
        print(f"  Plasma Current: {data['ip']} kA")
        print(f"  Major Radius: {data['major_radius']} m")
        print(f"  Minor Radius: {data['minor_radius']} m")
        print(f"  Elongation: {data['elongation']}")
        print(f"  Triangularity: {data['triangularity']}")
        print(f"  q95: {data['q95']}")
        print("\nMagnetic Axis Position:")
        print(f"  R: {data['magnetic_axis_r']} m")
        print(f"  Z: {data['magnetic_axis_z']} m")
        print("\nGlobal Parameters:")
        print(f"  Beta Poloidal: {data['beta_poloidal']}")
        print(f"  Torodial B-field at Axis: {data['b_field_tor_axis']} T")
        print("\nGeometric Parameters:")
        print(f"  Plasma Volume: {data['plasma_volume']} m^3")
        print(f"  Plasma Surface: {data['plasma_surface_area']} m^2")
        print("\nStrike Points:")
        print(f"  Inner Strike Point (R,Z): ({data['strike_r_inner']}, {data['strike_z_inner']}) m")
        print(f"  Outer Strike Point (R,Z): ({data['strike_r_outer']}, {data['strike_z_outer']}) m")
        print("="*50)
        
        return data
        
    except (ValueError, KeyError, IndexError) as e:
        print(f"\nError processing shot {efit_set['shot']} at time {efit_set['time']}:")
        print(f"Error message: {str(e)}")
        print(f"Skipping this file...")
        return None

def generate_efit_history_excel(base_path=None):
    """Main function to generate EFIT history Excel file.
    
    Args:
        base_path (str, optional): Base directory path to search for EFIT files.
                                 Defaults to "/srv/vest.filedb/public".
    """
    # Step 1: Find all EFIT files
    efit_files = find_efit_files(base_path)
    
    if not efit_files:
        print("No EFIT files found to process.")
        return
        
    # Step 2: Process each EFIT file set
    all_data = []
    for efit_set in efit_files:
        print(f"Processing shot {efit_set['shot']} at time {efit_set['time']}...")
        data = extract_efit_data(efit_set)
        if data is not None:  # Only append if data was successfully extracted
            all_data.append(data)
    
    # Step 3: Create DataFrame and save to Excel
    if all_data:
        df = pd.DataFrame(all_data)
        # Sort by shot number and time
        df = df.sort_values(['shot_number', 'time'])
        # Save to Excel
        df.to_excel("efit_history.xlsx", index=False)
        print(f"\nProcessing complete! Successfully processed {len(all_data)} out of {len(efit_files)} files.")
        print("Saved to efit_history.xlsx")
    else:
        print("\nNo data was successfully processed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate EFIT history Excel file')
    parser.add_argument('--base-path', type=str, 
                       default="/srv/vest.filedb/public",
                       help='Base directory path to search for EFIT files')
    
    args = parser.parse_args()
    generate_efit_history_excel(args.base_path)