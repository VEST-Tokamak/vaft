"""
How to run : 
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from vaft.processing import *
from vaft import database

WATCH_PATH = "/srv/vest.filedb"

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
        vfit_thomson_scattering_static(ods)
        vfit_thomson_scattering_dynamic(ods, shotnumber, base_path=os.path.dirname(filepath))
        print(f"[SUCCESS] Thomson data loaded for shot {shotnumber}")
    except Exception as e:
        print(f"[ERROR] Failed to update Thomson data for shot {shotnumber}: {e}")
        return None

    return shotnumber

def store_single_fit_to_ods(ods, t_idx, mapped_rho_position, n_e_function, T_e_function):
    num_channels = len(ods['thomson_scattering.channel'])

    ne_meas = []
    te_meas = []

    for i in range(num_channels):
        ne = ods[f'thomson_scattering.channel.{i}.n_e.data'][t_idx]
        te = ods[f'thomson_scattering.channel.{i}.t_e.data'][t_idx]
        ne_meas.append(ne)
        te_meas.append(te)

    rho_clipped = np.clip(mapped_rho_position, 0, 1)
    ne_recon = n_e_function(rho_clipped).tolist()
    te_recon = T_e_function(rho_clipped).tolist()

    base_den = f'core_profiles.profiles_1d.{t_idx}.electrons.density_fit'
    base_tem = f'core_profiles.profiles_1d.{t_idx}.electrons.temperature_fit'

    ods[f'{base_den}.measured'] = ne_meas
    ods[f'{base_den}.reconstructed'] = ne_recon
    ods[f'{base_tem}.measured'] = te_meas
    ods[f'{base_tem}.reconstructed'] = te_recon

def fit_thomson_profile_auto_all_times(shotnumber, Te_order=3, Ne_order=3):
    try:
        ods = database.load(shotnumber)
        geq = get_geq(shotnumber)
        mapped_rho = equilibrium_mapping_thomson_scattering(ods, geq)

        time_array = ods['thomson_scattering.time'] * 1e3  # in ms

        for t_idx, time_ms in enumerate(time_array):
            print(f"[INFO] Fitting profile for shot {shotnumber} at {time_ms:.1f} ms")

            try:
                result = profile_fitting_thomson_scattering(
                    ods,
                    time_ms=time_ms,
                    mapped_rho_position=mapped_rho,
                    Te_order=Te_order,
                    Ne_order=Ne_order,
                )
                n_e_fn, T_e_fn, *_ = result
                store_single_fit_to_ods(ods, t_idx, mapped_rho, n_e_fn, T_e_fn)

            except Exception as e:
                print(f"[WARNING] Skipped time {time_ms:.1f} ms due to error: {e}")

        database.save(ods, shotnumber)
        print(f"[SAVED] Updated ODS with fitted profiles for shot {shotnumber}")

    except Exception as e:
        print(f"[ERROR] Profile fitting failed for shot {shotnumber}: {e}")

def on_created(event):
    if event.src_path.endswith(".mat"):
        print(f"[DETECTED] New .mat file: {event.src_path}")
        shotnumber = update_thomson_auto(event.src_path)
        if shotnumber is not None:
            fit_thomson_profile_auto_all_times(shotnumber)

def main():
    event_handler = PatternMatchingEventHandler(patterns=["*.mat"], ignore_directories=True)
    event_handler.on_created = on_created

    observer = Observer()
    observer.schedule(event_handler, path=WATCH_PATH, recursive=True)
    observer.start()

    print(f"[WATCHING] Folder: {WATCH_PATH}")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
