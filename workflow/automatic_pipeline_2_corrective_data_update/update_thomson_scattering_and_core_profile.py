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

' nohup python3 update_thomson_scattering_and_core_profile.py > thomson_scattering_update_fitting.log 2>&1 & '
"""
import os, shutil, re
import time
import numpy as np
import vaft
from vaft import database, process
from vaft.data import read_geqdsk
from vaft.machine_mapping.thomson_scattering import thomson_scattering
import h5pyd
from datetime import datetime
import omas
import h5py

from vaft.database.sources import DEFAULT_SOURCE, LEGACY_SOURCE
from vaft.database.sources import resolve as resolve_source
from vaft.database.utils import processed_registry_uri, read_legacy_processed_registry

# The named HSDS source this updater reads and writes. The legacy ``public``
# namespace is a read-only reference now, so corrective results land in the
# VAFT-native source alongside the baseline they correct. Resolved as writable:
# this module opens the shot registry with raw h5pyd in append mode, which
# would otherwise sail straight past the read-only guarantee.
SOURCE = resolve_source(
    os.environ.get("VAFT_HSDS_SOURCE") or DEFAULT_SOURCE, writable=True
)

_TOTAL_PRESSURE_LEAVES = (
    "pressure_thermal", "pressure", "pressure_ion_total",
    "pressure_ion_total_thermal", "pressure_parallel", "pressure_perpendicular",
)


def strip_electron_only_pressure(ods):
    """Delete slice-total pressure from any core_profiles slice with no ion temperature.

    A Thomson-only slice has no ion measurement, so it must not carry a total
    (kinetic) pressure computed with a phantom Ti=Te.
    """
    if "core_profiles" not in ods:
        return
    for i in range(len(ods["core_profiles.profiles_1d"])):
        b = f"core_profiles.profiles_1d.{i}"
        if f"{b}.ion.0.temperature" in ods:  # a real kinetic slice keeps its pressure
            continue
        for leaf in _TOTAL_PRESSURE_LEAVES:
            key = f"{b}.{leaf}"
            if key in ods:
                del ods[key]


def extract_shotnumber_of_thomson_scattering(fname: str):
    """
    Extract shotnumber from a Thomson-scattering filename.

    Supported layouts (all styles present in /srv/vest.diagnostic):
    - 'Shot40330_v10.mat', 'NeTe_Shot48223_v9_rev.mat'  -> ...Shot<shot>...
    - '40330_NeTe.mat'                                   -> <shot>_...
    - 'NeTe_48223.mat'                                   -> NeTe_<shot>

    The ``Shot`` pattern is tried FIRST so that 'NeTe_Shot48223_v9.mat' resolves
    via the shot tag rather than being mis-parsed by the trailing-number rule.

    Returns None when the name matches no known layout, so callers can log and
    skip the file instead of silently dropping it.
    """
    match1 = re.search(r"Shot(\d+)", fname, re.IGNORECASE)
    if match1:
        return int(match1.group(1))

    match2 = re.match(r"^(\d+)_", fname)
    if match2:
        return int(match2.group(1))

    # 'NeTe_<shot>.mat' -- the layout used from the 48xxx campaign onward. Without
    # this the whole batch was skipped silently (shots 48222-48234 never reached
    # the database), mirroring the '^IDS[_-](\d+)' rule the CES updater already has.
    match3 = re.match(r"^NeTe[_-](\d+)", fname, re.IGNORECASE)
    if match3:
        return int(match3.group(1))

    return None

# The registry lives beside the shots it describes, so it moves with SOURCE.
PROCESSED_H5_PATH = processed_registry_uri(SOURCE, writable=True)

# Registry groups inside processed_shots.h5. Each diagnostic owns its own group
# so Thomson and charge_exchange records (both keyed by bare shot number) never
# collide. Mirrors vaft.database.utils.{TS,CX}_REGISTRY_GROUP.
TS_REGISTRY_GROUP = "shots"

# Statuses that must not be downgraded to 'invalid' on a later failed run.
_SUCCESS_STATUSES = ["core_profile", "thomson_only", "charge_exchange",
                     "kinetic_core_profile"]

def save_processed_shot(shotnumber, mtime, status="core_profile", group=TS_REGISTRY_GROUP):
    """Save shot number, last modified time, and processing status.

    ``group`` selects the per-diagnostic registry group (``shots`` for Thomson,
    ``cx_shots`` for charge_exchange), so the two updaters never overwrite each
    other's records.
    """
    try:
        with h5pyd.File(PROCESSED_H5_PATH, "a") as f:
            if group not in f:
                f.create_group(group)

            g = f[group]
            key = str(shotnumber)

            if key in g:
                current_status = g[key]["status"][()]
                if isinstance(current_status, bytes):
                    current_status = current_status.decode()

                # 성공 상태(core_profile/thomson_only/charge_exchange)는 invalid로 덮지 않음
                if current_status in _SUCCESS_STATUSES and status == "invalid":
                    print(f"[SKIP] Shot {shotnumber} already marked as '{current_status}', not overwriting with 'invalid'")
                    return

                # overwrite values properly using [:]
                # np.string_ was removed in NumPy 2.0; np.bytes_ is the replacement.
                g[key]["timestamp"][...] = np.bytes_(mtime)
                g[key]["status"][...] = np.bytes_(status)
                print(f"[INFO] Updated shot {shotnumber}: status='{status}'")

            else:
                grp = g.create_group(key)
                vlen_str = h5py.string_dtype(encoding='utf-8')
                grp.create_dataset("timestamp", data=mtime, dtype=vlen_str)
                grp.create_dataset("status", data=status, dtype=vlen_str)
                print(f"[INFO] Added new shot {shotnumber} with status '{status}'")

    except Exception as e:
        print(f"[ERROR] Could not save processed shot {shotnumber}: {e}")



def load_processed_shots(group=TS_REGISTRY_GROUP):
    """Return dict {shotnumber: {'timestamp': ..., 'status': ...}} for one group.

    The registry moved with the shots it describes, so a source that has just
    been provisioned starts empty and every watched file would look unprocessed.
    The read-only legacy registry seeds that first run.

    The seed is a bootstrap, not a permanent merge: once this source's group
    holds any record it is the whole answer. Re-reading legacy on every call
    would make ``reset_processed_shots`` unable to clear a legacy-known shot,
    because the next load would resurrect it.
    """
    shots = {}
    try:
        with h5pyd.File(PROCESSED_H5_PATH, "r") as f:
            if group in f:
                g = f[group]
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
        print(f"[INFO] No processed_shots.h5 in '{SOURCE}' yet.")

    if shots or SOURCE == LEGACY_SOURCE:
        return shots

    for key, record in read_legacy_processed_registry(group).items():
        try:
            shots[int(key)] = dict(record)
        except (TypeError, ValueError):
            continue
    if shots:
        print(
            f"[INFO] Seeded {len(shots)} '{group}' record(s) from the legacy "
            f"registry; new records are written to '{SOURCE}' only."
        )
    return shots

CHECK_INTERVAL = 10  

def update_thomson_auto(filepath):
    filename = os.path.basename(filepath)
    # Use the shared extractor rather than a second, divergent parser: the old
    # inline `int(filename.split("_")[0])` raised ValueError on 'NeTe_<shot>.mat'
    # (int("NeTe")), and returning None here makes the caller's tuple unpacking
    # fail with a confusing TypeError.
    shotnumber = extract_shotnumber_of_thomson_scattering(filename)
    if shotnumber is None:
        print(f"[ERROR] Could not parse shot number from {filename}")
        return None
    print(f"[INFO] Processing shot: {shotnumber}")

    try:
        ods = database.load(shotnumber, source=SOURCE)
    except Exception as e:
        print(f"[ERROR] Failed to load ODS for shot {shotnumber}: {e}")
        return None

    try:
        thomson_scattering(ods, shotnumber, filepath)
        print(f"[SUCCESS] Thomson data loaded for shot {shotnumber}")
    except Exception as e:
        print(f"[ERROR] Failed to update Thomson data for shot {shotnumber}: {e}")
        return None
    
    database.save(ods, shotnumber, source=SOURCE)
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
                geq = read_geqdsk(geq_filename)
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during rho mapping: {e}")
                continue

            try:
                mapped_rho = process.equilibrium_mapping_thomson_scattering(ods, geq)
                result = process.profile_fitting_thomson_scattering(
                    ods, time_ms, mapped_rho,
                    Te_order=2, Ne_order=2,
                    fitting_function_te='polynomial',
                    fitting_function_ne='exponential'
                )
                n_e_fn, T_e_fn, *_ = result
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during profile fitting: {e}")
                continue

            try:
                ods = process.core_profiles(ods, time_ms, mapped_rho, n_e_fn, T_e_fn, geq=geq, ti_te_fallback=False)
                omas.omas_physics.core_profiles_pressures(ods, update=True)
                omas.save_omas_json(ods, f"/srv/vest.filedb/public/{shotnumber}/omas/{shotnumber}_core_profile.json")
                success_count += 1 
            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms during core_profile mapping: {e}")
                continue

        strip_electron_only_pressure(ods)  # Thomson-only slices carry no total pressure
        database.save(ods, shotnumber, source=SOURCE)

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

#: Filenames already reported as unparseable, so the polling loop warns once
#: per file instead of on every pass.
_unparsed_reported = set()


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
                            # Log it: an unrecognised layout previously dropped whole
                            # campaigns without a trace (see the NeTe_<shot> case).
                            if fname not in _unparsed_reported:
                                _unparsed_reported.add(fname)
                                print(f"[WARNING] no shot number in filename, skipped: {fname}")
                            continue
                    except Exception as exc:  # noqa: BLE001
                        print(f"[WARNING] could not parse filename {fname}: {exc}")
                        continue
                    mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
                    prev_info = processed_shots.get(shotnumber)

                    if isinstance(prev_info, str):
                        prev_info = {"timestamp": prev_info, "status": "unknown"}

                    if prev_info:
                        prev_time = prev_info.get("timestamp", "")
                        prev_status = prev_info.get("status", "unknown")

                        if prev_time == mtime :
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
                    fitted = False
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

                    processed_shots[shotnumber] = {
                        "timestamp": mtime,
                        "status": "core_profile" if fitted else "thomson_only"
                    }

                    print(f"[SAVED] Processed shot {shotnumber}")

            except Exception as e:
                print(f"[ERROR] Polling failed: {e}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOPPED] Thomson auto-updater stopped by user.")

def reset_processed_shots(clear_entire_file=False):
    """
    Reset processed shot registry.

    - clear_entire_file=False (default): delete only /shots group (recreate empty)
    - clear_entire_file=True: delete everything in the file (dangerous)
    """
    with h5pyd.File(PROCESSED_H5_PATH, "a") as f:
        if clear_entire_file:
            # Delete all top-level objects
            for key in list(f.keys()):
                del f[key]
            # Recreate shots group
            f.create_group("shots")
            print("[DONE] Cleared entire file and recreated /shots")
            return

        # Default: only reset /shots
        if "shots" in f:
            del f["shots"]
        for shot in list(f.keys()):
            ods = database.load(shot, source=SOURCE)
            if 'thomson_scattering' in ods:
                ods['thomson_scattering'].clear()
            if 'core_profiles' in ods:
                ods['core_profiles'].clear()
            database.save(ods, shot, source=SOURCE)
        f.create_group("shots")
        print("[DONE] Reset /shots (all processed shot records cleared)")

if __name__ == "__main__":
    main()
    # reset_processed_shots()
