import pandas as pd
import os
import numpy as np

def join_chease_stability(chease_file="chease_history.xlsx", stability_file="stability_history.xlsx", output_file="chease_stability_history.xlsx"):
    """
    Joins CHEASE and stability history data, excluding error cases.
    Ensures all columns from both datasets are present in the final output.
    Only keeps rows where all columns have valid data (no NaN or None values).

    Args:
        chease_file (str): Path to the CHEASE history Excel file.
        stability_file (str): Path to the stability history Excel file.
        output_file (str): Path to save the merged Excel file.
    """
    if not os.path.exists(chease_file):
        print(f"Error: CHEASE history file not found at {chease_file}")
        return
    
    if not os.path.exists(stability_file):
        print(f"Error: Stability history file not found at {stability_file}")
        return

    print("Reading CHEASE history...")
    chease_df = pd.read_excel(chease_file)
    print("CHEASE history loaded:")
    print(chease_df.info())
    chease_columns = set(chease_df.columns)

    print("\nReading stability history...")
    stability_df = pd.read_excel(stability_file)
    print("Stability history loaded:")
    print(stability_df.info())
    stability_columns = set(stability_df.columns)

    # Filter out error cases from stability data
    print("\nFiltering out error cases from stability data...")
    stability_df = stability_df[
        (stability_df['ideal_stability'] != 'Error') & 
        (stability_df['resistive_stability'] != 'Error')
    ]
    print(f"After filtering errors: {stability_df.shape[0]} rows remaining")

    print("\nMerging dataframes on 'shot' and 'time [ms]'...")
    # 'inner' join will only keep rows where shot and time exist in both files
    merged_df = pd.merge(chease_df, stability_df, on=['shot', 'time [ms]'], how='inner')
    print(f"After merge: {merged_df.shape[0]} rows")

    # Check for missing values in each column
    print("\nChecking for missing values in each column...")
    missing_values = merged_df.isnull().sum()
    print("\nMissing values per column:")
    for col in merged_df.columns:
        missing = missing_values[col]
        if missing > 0:
            print(f"{col}: {missing} missing values")

    # Filter out rows with missing values in critical columns
    print("\nFiltering out rows with missing values in critical columns...")
    initial_rows = len(merged_df)
    
    # Define critical columns that should not have missing values
    critical_columns = [
        'shot', 'time [ms]', 'ip [kA]', 'major_radius [m]', 'minor_radius [m]',
        'q95', 'beta_normal', 'ideal_stability', 'resistive_stability'
    ]
    
    # Add deltaW series (n=1 to 6)
    critical_columns.extend([f'delta_w_n{n}' for n in range(1, 7)])
    
    # Add tearing index series for both rdcon and stride (n=1,2)
    critical_columns.extend([f'tearing_index_rdcon_n{n}' for n in range(1, 3)])
    critical_columns.extend([f'tearing_index_stride_n{n}' for n in range(1, 3)])

    # Add new stability columns: qlim, qa, ballooning_stability, ballooning_unstable_index
    critical_columns.extend(['qlim', 'qa', 'ballooning_stability', 'ballooning_unstable_index'])
    
    # Filter rows where critical columns have missing values
    merged_df = merged_df.dropna(subset=critical_columns)
    rows_after_filtering = len(merged_df)
    print(f"Removed {initial_rows - rows_after_filtering} rows with missing values in critical columns")
    print(f"Remaining rows: {rows_after_filtering}")

    # Verify all columns are present
    merged_columns = set(merged_df.columns)
    missing_chease_columns = chease_columns - merged_columns
    missing_stability_columns = stability_columns - merged_columns

    if missing_chease_columns or missing_stability_columns:
        print("\nWARNING: Some columns are missing from the merged dataset:")
        if missing_chease_columns:
            print("\nMissing CHEASE columns:")
            for col in sorted(missing_chease_columns):
                print(f"  - {col}")
        if missing_stability_columns:
            print("\nMissing stability columns:")
            for col in sorted(missing_stability_columns):
                print(f"  - {col}")
        print("\nThese columns will be excluded from the final output.")
    else:
        print("\nAll columns from both datasets are present in the merged output.")

    print("\nMerge complete. Merged data info:")
    print(merged_df.info())

    print(f"\nSaving merged data to {output_file}...")
    merged_df.to_excel(output_file, index=False)
    print("Save complete.")
    print(f"Final merged data has {merged_df.shape[0]} rows.")


if __name__ == '__main__':
    # cd to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    join_chease_stability() 