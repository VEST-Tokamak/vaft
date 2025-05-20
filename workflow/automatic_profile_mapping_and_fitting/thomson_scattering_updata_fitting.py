#!/usr/bin/python

"""
This script automatically updates Thomson scattering data and fits profiles
for all times when a new .mat file is detected in the specified directory.
It uses the watchdog library to monitor the directory and processes the
new file by loading the data into an ODS (OMAS Data Structure) object.

The script performs the following steps:
1. Monitors a specified directory for new .mat files.
2. When a new file is detected, it extracts the shot number from the filename.
3. Loads the ODS for the shot number.
4. Updates the Thomson scattering data in the ODS.
5. Fits the Thomson scattering profiles for all times in the ODS.
6. Saves the updated ODS back to the database.

How to use:
- Ensure that the vaft library is installed and properly configured.
- Set the WATCH_PATH variable to the directory you want to monitor.
- Run the script. It will continuously monitor the directory for new .mat files.

' nohup python3 thomson_scattering_update_fitting.py > thomson_scattering_update_fitting.log 2>&1 & '
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from vaft import database, machine_mapping, process
import vaft
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

WATCH_BASE = "/srv/vest.filedb/public"
CHECK_INTERVAL = 10  
SEEN_FILES_PATH = "seen_files.txt"

def load_seen_files():
    if os.path.exists(SEEN_FILES_PATH):
        with open(SEEN_FILES_PATH, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_seen_file(filepath):
    with open(SEEN_FILES_PATH, 'a') as f:
        f.write(filepath + '\n')

def update_thomson_auto(filepath):
    filename = os.path.basename(filepath)
    try:
        shotnumber = int(filename.split("Shot")[1].split("_")[0])
        print(f"[INFO] Processing shot: {shotnumber}")
    except Exception as e:
        print(f"[ERROR] Could not parse shot number from {filename}: {e}")
        return None

    try:
        ods = database.load(shotnumber)
    except Exception as e:
        print(f"[ERROR] Failed to load ODS for shot {shotnumber}: {e}")
        return None

    try:
        machine_mapping.thomson_scattering_static(ods)
        machine_mapping.thomson_scattering_from_file(ods, shotnumber, filepath)
        print(f"[SUCCESS] Thomson data loaded for shot {shotnumber}")
    except Exception as e:
        print(f"[ERROR] Failed to update Thomson data for shot {shotnumber}: {e}")
        return None
    
    database.save(ods, shotnumber)
    print(f"[SAVED] Updated ODS for shot {shotnumber}")

    return shotnumber


def fit_thomson_profile_auto_all_times(shotnumber, Te_order=3, Ne_order=3):
    try:
        ods = database.load(shotnumber)
        time_array = ods['thomson_scattering.time'] * 1e3  # in ms

        for t_idx, time_ms in enumerate(time_array):
            print(f"[INFO] Fitting profile for shot {shotnumber} at {time_ms:.1f} ms")

            geq_filename = f'/srv/vest.filedb/public/{shotnumber}/efit/gfile/g0{shotnumber}.00{int(time_ms):03}'
            if not os.path.exists(geq_filename):
                print(f"[WARNING] Geqdsk file not found at {geq_filename}")
                continue

            try:
                geq = OMFITgeqdsk(filename=geq_filename)
                mapped_rho = process.equilibrium_mapping_thomson_scattering(ods, geq)
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during rho mapping: {e}")
                continue

            try:
                result = process.profile_fitting_thomson_scattering(
                    ods,
                    time_ms=time_ms,
                    mapped_rho_position=mapped_rho,
                    Te_order=Te_order,
                    Ne_order=Ne_order,
                )
                n_e_fn, T_e_fn, *_ = result
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during profile fitting: {e}")
                continue

            try:
                machine_mapping.core_profile(ods, t_idx, mapped_rho, n_e_fn, T_e_fn)
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during core_profile mapping: {e}")
                continue

        database.save(ods, shotnumber)
        print(f"[SAVED] Updated ODS with fitted profiles for shot {shotnumber}")

    except Exception as e:
        print(f"[ERROR] Failed to fit Thomson profiles for shot {shotnumber}: {e}")

def main():
    seen_files = load_seen_files()
    while True:
        print("[POLLING] Scanning for new .mat files...")
        try:
            for subdir in os.listdir(WATCH_BASE):
                subpath = os.path.join(WATCH_BASE, subdir)
                if not os.path.isdir(subpath):
                    continue
                for fname in os.listdir(subpath):
                    if fname.endswith(".mat"):
                        full_path = os.path.join(subpath, fname)
                        if full_path in seen_files:
                            continue
                        print(f"[NEW FILE] {full_path}")
                        shotnumber = update_thomson_auto(full_path)
                        if shotnumber is not None:
                            fit_thomson_profile_auto_all_times(shotnumber)
                        save_seen_file(full_path)
                        seen_files.add(full_path)
        except Exception as e:
            print(f"[ERROR] Polling failed: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
