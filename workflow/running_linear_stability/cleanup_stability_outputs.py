import os
import glob
import shutil
import argparse
import sys

def cleanup_stability_outputs(base_path):
    """
    Deletes 'linear_stability' and 'logs/linear_stability' directories 
    for all shot folders found under the given base_path.
    """
    print(f"Starting cleanup under base path: {base_path}\n")
    
    shot_dirs = glob.glob(os.path.join(base_path, "[0-9]" * 5))
    
    if not shot_dirs:
        print(f"No shot directories found matching pattern '[0-9]*5' in {base_path}")
        return

    print(f"Found {len(shot_dirs)} potential shot directories.")

    for shot_dir_abs_path in shot_dirs:
        if not os.path.isdir(shot_dir_abs_path):
            # print(f"Skipping {shot_dir_abs_path}, not a directory.") # Should not happen with glob
            continue

        shot_number = os.path.basename(shot_dir_abs_path)
        print(f"\nProcessing shot: {shot_number}")

        # Path to the main linear_stability directory for the shot
        linear_stability_dir = os.path.join(shot_dir_abs_path, "linear_stability")
        
        # Path to the logs/linear_stability directory for the shot
        logs_dir = os.path.join(shot_dir_abs_path, "logs")
        log_linear_stability_dir = os.path.join(logs_dir, "linear_stability")

        # Delete shot_dir/linear_stability/
        if os.path.exists(linear_stability_dir):
            if os.path.isdir(linear_stability_dir):
                try:
                    shutil.rmtree(linear_stability_dir)
                    print(f"  DELETED: {linear_stability_dir}")
                except OSError as e:
                    print(f"  ERROR deleting {linear_stability_dir}: {e}", file=sys.stderr)
            else:
                print(f"  SKIPPED (not a dir): {linear_stability_dir}")
        else:
            print(f"  NOT FOUND: {linear_stability_dir}")

        # Delete shot_dir/logs/linear_stability/
        if os.path.exists(log_linear_stability_dir):
            if os.path.isdir(log_linear_stability_dir):
                try:
                    shutil.rmtree(log_linear_stability_dir)
                    print(f"  DELETED: {log_linear_stability_dir}")
                    # If logs/linear_stability was the only thing in logs/, try to remove logs/ too
                    try:
                        if not os.listdir(logs_dir): # Check if logs_dir is now empty
                            os.rmdir(logs_dir)
                            print(f"  DELETED (empty parent): {logs_dir}")
                    except OSError:
                        pass # Ignore if logs_dir can't be removed (not empty or other issue)

                except OSError as e:
                    print(f"  ERROR deleting {log_linear_stability_dir}: {e}", file=sys.stderr)
            else:
                print(f"  SKIPPED (not a dir): {log_linear_stability_dir}")
        else:
            print(f"  NOT FOUND: {log_linear_stability_dir}")
            
    print("\nCleanup process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Clean up linear_stability and logs/linear_stability directories for all shots.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--base-path', type=str, 
                        default='/srv/vest.filedb/public',
                        help='Base directory path containing shot folders (default: /srv/vest.filedb/public)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be deleted without actually deleting anything.')
    
    args = parser.parse_args()

    if args.dry_run:
        print("\n--- DRY RUN MODE --- No files will be deleted. ---")
        # Override shutil.rmtree and os.rmdir for dry run
        def dry_run_rmtree(path):
            print(f"  WOULD DELETE (directory): {path}")
        def dry_run_rmdir(path):
            print(f"  WOULD DELETE (empty directory): {path}")
        shutil.rmtree_orig = shutil.rmtree
        os.rmdir_orig = os.rmdir
        shutil.rmtree = dry_run_rmtree
        os.rmdir = dry_run_rmdir
    
    cleanup_stability_outputs(args.base_path)

    if args.dry_run:
        # Restore original functions
        shutil.rmtree = shutil.rmtree_orig
        os.rmdir = os.rmdir_orig
        print("--- DRY RUN MODE COMPLETE ---") 