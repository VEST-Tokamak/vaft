import os
import glob
import argparse
import sys # Added for potential error messaging, though not used in this edit

def find_plotted_gfiles(base_path):
    """
    Finds all g{{shot}}.{{time}} files that have a corresponding chease/plots/{{time}}.png.
    Prints the absolute paths of these g-files to standard output.
    """
    chease_files_to_output = []
    # Removed shot limiting logic:
    # selected_shot_numbers = set()
    # ordered_encountered_shots = []

    shot_dirs = sorted(glob.glob(os.path.join(base_path, "[0-9]" * 5))) # Sort to ensure consistent shot ordering
    
    for shot_dir_abs in shot_dirs:
        shot_number_str = os.path.basename(shot_dir_abs)
        # Removed shot limiting logic:
        # if shot_number_str not in selected_shot_numbers and shot_number_str not in ordered_encountered_shots:
        #     ordered_encountered_shots.append(shot_number_str)
        
        # shots_to_process_fully = ordered_encountered_shots[:2] # Removed
        # selected_shot_numbers.update(shots_to_process_fully) # Removed

        # Iterate again, or structure to collect files only for selected_shot_numbers
        # The original second loop is now the main processing loop for each shot_dir_abs
        # No need to check current_shot_number_str against selected_shot_numbers
        
        padded_shot = f"{int(shot_number_str):06d}" # Use shot_number_str directly
        chease_dir_abs = os.path.join(shot_dir_abs, "chease")
        plots_dir_abs = os.path.join(chease_dir_abs, "plots")

        if os.path.exists(chease_dir_abs) and os.path.exists(plots_dir_abs):
            plot_files = sorted(glob.glob(os.path.join(plots_dir_abs, "*.png"))) # Sort for consistent time ordering
            
            for plot_file_abs in plot_files:
                time_str = os.path.splitext(os.path.basename(plot_file_abs))[0]
                
                if not (time_str.isdigit() and len(time_str) == 5):
                    continue

                gfile_path_abs = os.path.join(chease_dir_abs, f"g{padded_shot}.{time_str}")
                
                if os.path.exists(gfile_path_abs):
                    chease_files_to_output.append(gfile_path_abs)
                    
    for gfile_path in chease_files_to_output:
        print(gfile_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find g-files with corresponding plots.')
    parser.add_argument('--base-path', type=str, required=True, 
                        help='Base directory path to search for chease files (e.g., /srv/vest.filedb/public)')
    args = parser.parse_args()
    find_plotted_gfiles(args.base_path) 