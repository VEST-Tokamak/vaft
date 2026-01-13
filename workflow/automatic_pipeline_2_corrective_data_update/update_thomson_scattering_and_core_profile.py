#!/home/user1/miniconda3/envs/vaft/bin/python

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
import os, shutil, re
import time
import numpy as np
from vaft import database, machine_mapping, process
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import h5pyd
from datetime import datetime
import omas
import h5py

def extract_shotnumber_of_thomson_scattering(fname: str):
    """
    Extract shotnumber from filename.
    Supports both:
    - 'Shot40330_v10.mat'
    - '40330_NeTe.mat'
    """
    match1 = re.search(r"Shot(\d+)", fname, re.IGNORECASE)
    if match1:
        return int(match1.group(1))

    match2 = re.match(r"^(\d+)_", fname)
    if match2:
        return int(match2.group(1))
    
    return None

PROCESSED_H5_PATH = "hdf5://public/processed_shots.h5"

def save_processed_shot(shotnumber, mtime, status="core_profile"):
    """Save shot number, last modified time, and processing status."""
    try:
        with h5pyd.File(PROCESSED_H5_PATH, "a") as f:
            if "shots" not in f:
                f.create_group("shots")

            g = f["shots"]
            key = str(shotnumber)

            if key in g:
                current_status = g[key]["status"][()]
                if isinstance(current_status, bytes):
                    current_status = current_status.decode()

                # core_profile/thomson_only 상태는 invalid로 덮지 않음
                if current_status in ["core_profile", "thomson_only"] and status == "invalid":
                    print(f"[SKIP] Shot {shotnumber} already marked as '{current_status}', not overwriting with 'invalid'")
                    return

                # overwrite values properly using [:]
                g[key]["timestamp"][...] = np.string_(mtime)
                g[key]["status"][...] = np.string_(status)
                print(f"[INFO] Updated shot {shotnumber}: status='{status}'")

            else:
                grp = g.create_group(key)
                vlen_str = h5py.string_dtype(encoding='utf-8')
                grp.create_dataset("timestamp", data=mtime, dtype=vlen_str)
                grp.create_dataset("status", data=status, dtype=vlen_str)
                print(f"[INFO] Added new shot {shotnumber} with status '{status}'")

    except Exception as e:
        print(f"[ERROR] Could not save processed shot {shotnumber}: {e}")



def load_processed_shots():
    """Return dict {shotnumber: {'timestamp': ..., 'status': ...}}"""
    shots = {}
    try:
        with h5pyd.File(PROCESSED_H5_PATH, "r") as f:
            if "shots" not in f:
                return shots
            g = f["shots"]
            for key in g.keys():
                shots[int(key)] = {
                    "timestamp": g[key]["timestamp"][()].decode()
                    if isinstance(g[key]["timestamp"][()], bytes)
                    else g[key]["timestamp"][()],
                    "status": g[key]["status"][()].decode()
                    if isinstance(g[key]["status"][()], bytes)
                    else g[key]["status"][()],
                }
    except Exception:
        print("[INFO] No processed_shots.h5 found, starting empty.")
    return shots

    
WATCH_BASE = "/srv/vest.filedb"
CHECK_INTERVAL = 10  

def update_thomson_auto(filepath):
    filename = os.path.basename(filepath)
    try:

        if "Shot" in filename:
            # e.g. NeTe_Shot39915_v9_rev.mat → 39915
            shotnumber = int(filename.split("Shot")[1].split("_")[0])
        else:
            # e.g. 46051_NeTe.mat → 46051
            shotnumber = int(filename.split("_")[0])        
        
        print(f"[INFO] Processing shot: {shotnumber}")
    except Exception as e:
        print(f"[ERROR] Could not parse shot number from {filename}: {e}")
        return None

    try:
        ods = database.load(shotnumber,'public')
    except Exception as e:
        print(f"[ERROR] Failed to load ODS for shot {shotnumber}: {e}")
        return None

    try:
        machine_mapping.thomson_scattering(ods, shotnumber, filepath)
        print(f"[SUCCESS] Thomson data loaded for shot {shotnumber}")
    except Exception as e:
        print(f"[ERROR] Failed to update Thomson data for shot {shotnumber}: {e}")
        return None
    
    database.save(ods, shotnumber)
    print(f"[SAVED] Updated ODS for shot {shotnumber}")

    return ods, shotnumber


def fit_thomson_profile_auto_all_times(ods, shotnumber):
    success_count = 0  #

    try:
        time_array = ods['thomson_scattering.time'] * 1e3  # in ms

        for t_idx, time_ms in enumerate(time_array):
            print(f"[INFO] Fitting profile for shot {shotnumber} at {time_ms:.1f} ms")

            geq_filename = f'/srv/vest.filedb/public/{shotnumber}/chease/g0{shotnumber}.00{int(time_ms):03}'
            if not os.path.exists(geq_filename):
                print(f"[WARNING] Geqdsk file not found at {geq_filename}")
                continue

            try:
                geq = OMFITgeqdsk(filename=geq_filename)
                geq['fluxSurfaces'].load()
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during rho mapping: {e}")
                continue

            try:
                mapped_rho = process.equilibrium_mapping_thomson_scattering(ods, geq)
                result = process.profile_fitting_thomson_scattering(
                    ods, time_ms, mapped_rho,
                    Te_order=3, Ne_order=3,
                    fitting_function_te='exponential',
                    fitting_function_ne='exponential'
                )
                n_e_fn, T_e_fn, *_ = result
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during profile fitting: {e}")
                continue

            try:
                ods = process.core_profiles(ods, time_ms, mapped_rho, n_e_fn, T_e_fn)
                omas.omas_physics.core_profiles_pressures(ods, update=True)
                omas.save_omas_json(ods, f"/srv/vest.filedb/public/{shotnumber}/omas/{shotnumber}_core_profile.json")
                success_count += 1 
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during core_profile mapping: {e}")
                continue

        database.save(ods, shotnumber)

        if success_count > 0:
            print(f"[SAVED] Updated ODS with fitted profiles for shot {shotnumber}")
            return True  
        else:
            print(f"[INFO] No valid GEQDSK files found — only Thomson update performed.")
            return False  

    except Exception as e:
        print(f"[ERROR] Failed to fit Thomson profiles for shot {shotnumber}: {e}")
        return False


WATCH_DIAG = "/srv/vest.diagnostic"
PUBLIC_BASE = "/srv/vest.filedb/public"
CHECK_INTERVAL = 10  # seconds

def main():
    processed_shots = load_processed_shots()

    try:
        while True:
            print("[POLLING] Scanning for new diagnostic .mat files...")

            try:
                for fname in os.listdir(WATCH_DIAG):
                    if not fname.endswith(".mat"):
                        continue

                    full_path = os.path.join(WATCH_DIAG, fname)

                    try:
                        shotnumber = extract_shotnumber_of_thomson_scattering(fname)
                        if shotnumber is None:
                            continue
                    except Exception:
                        continue
                    mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
                    prev_info = processed_shots.get(shotnumber)

                    if isinstance(prev_info, str):
                        prev_info = {"timestamp": prev_info, "status": "unknown"}

                    if prev_info:
                        prev_time = prev_info.get("timestamp", "")
                        prev_status = prev_info.get("status", "unknown")

                        if prev_time == mtime and prev_status != "invalid":
                            print(f"[SKIP] Shot {shotnumber} already processed successfully ({prev_status})")
                            continue

                    print(f"[UPDATE DETECTED] {fname} (shot {shotnumber})")
                    diag_dir = os.path.join(PUBLIC_BASE, f"{shotnumber}/diagnostics")
                    os.makedirs(diag_dir, exist_ok=True)
                    dest_path = os.path.join(diag_dir, fname)
                    shutil.copy2(full_path, dest_path)

                    try:
                        ods, shotnumber = update_thomson_auto(dest_path)
                        print(f"[UPDATED] Thomson data for shot {shotnumber}")
                    except Exception as e:
                        print(f"[ERROR] Could not parse or load {fname}: {e}")
                        save_processed_shot(shotnumber, mtime, status="invalid")
                        continue


                    chease_dir = os.path.join(PUBLIC_BASE, f"{shotnumber}/chease")
                    if os.path.exists(chease_dir):
                        try:
                            fitted = fit_thomson_profile_auto_all_times(ods, shotnumber)
                            print(f"[FITTED] Thomson profile for {shotnumber}")

                            if fitted:
                                save_processed_shot(shotnumber, mtime, status="core_profile")
                            else:
                                save_processed_shot(shotnumber, mtime, status="thomson_only")
                        except Exception as e:
                            print(f"[WARNING] Fitting failed for {shotnumber}: {e}")
                            save_processed_shot(shotnumber, mtime, status="thomson_only")
                    else:
                        print(f"[WARNING] CHEASE directory missing for shot {shotnumber}")
                        save_processed_shot(shotnumber, mtime, status="thomson_only")

                    processed_shots[shotnumber] = mtime
                    print(f"[SAVED] Processed shot {shotnumber}")

            except Exception as e:
                print(f"[ERROR] Polling failed: {e}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOPPED] Thomson auto-updater stopped by user.")


if __name__ == "__main__":
    main()
