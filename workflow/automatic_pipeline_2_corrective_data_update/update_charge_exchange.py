#!/home/user1/miniconda3/envs/vaft/bin/python

"""Automatic charge_exchange (ion Doppler / CES) data update.

Ion analog of the Thomson half of ``update_thomson_scattering_and_core_profile.py``.
IDS (Ion Doppler Spectroscopy) and CES (Charge Exchange Spectroscopy) both feed the
OMAS ``charge_exchange`` IDS (only the ``.mat`` loader differs); this pipeline ingests
that raw diagnostic into the shot ODS and fits Ti(psi_N) / Vtor(psi_N) per time as a
coverage/QA check. The kinetic ``core_profiles`` assembly that consumes these fits lives
in ``update_core_profile.py``.

Per detected ``IDS_<shot>.mat`` / ``CES_<shot>.mat``:
  1. Load the shot ODS from the database.
  2. Ingest the ion diagnostic into ``charge_exchange`` (IDS preferred over CES).
  3. For each charge_exchange time with a matching CHEASE geqdsk, map channels to psi_N
     and fit Ti/Vtor (``profile_fitting_charge_exchange``) to confirm the fit is usable.
  4. Save the ODS back to the database and record the shot.

Run:
  nohup ./update_charge_exchange.py > charge_exchange_update.log 2>&1 &
"""
import os
import re
import shutil
import time
from datetime import datetime

import numpy as np
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

from vaft import database, machine_mapping, process

# Reuse the shot-registry helpers from the Thomson updater (same directory).
from update_thomson_scattering_and_core_profile import (
    load_processed_shots,
    save_processed_shot,
)

WATCH_DIAG = "/srv/vest.diagnostic"
PUBLIC_BASE = "/srv/vest.filedb/public"
CHECK_INTERVAL = 10  # seconds

# Filename prefix -> charge_exchange option; IDS is preferred over CES.
ION_PATTERNS = (
    (re.compile(r"^IDS[_-](\d+)", re.IGNORECASE), "ids"),
    (re.compile(r"^CES[_-](\d+)", re.IGNORECASE), "ces"),
)


def parse_ion_file(fname):
    """Return (shotnumber, option) for an IDS/CES filename, or (None, None)."""
    for pattern, option in ION_PATTERNS:
        m = pattern.match(fname)
        if m and fname.endswith(".mat"):
            return int(m.group(1)), option
    return None, None


def geqdsk_for_time(shotnumber, time_ms):
    """Return a loaded CHEASE geqdsk for (shot, time_ms), or None."""
    path = f"{PUBLIC_BASE}/{shotnumber}/chease/g0{shotnumber}.00{int(time_ms):03d}"
    if not os.path.exists(path):
        return None
    try:
        geq = OMFITgeqdsk(filename=path)
        geq["fluxSurfaces"].load()
        return geq
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not load geqdsk {path}: {exc}")
        return None


def update_charge_exchange_auto(ion_matfile):
    """Load the ODS and ingest the ion diagnostic. Returns (ods, shot, option) or None."""
    fname = os.path.basename(ion_matfile)
    shotnumber, option = parse_ion_file(fname)
    if shotnumber is None:
        print(f"[ERROR] Not an IDS/CES file: {fname}")
        return None

    try:
        ods = database.load(shotnumber, "public")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load ODS for shot {shotnumber}: {exc}")
        return None

    try:
        machine_mapping.charge_exchange(
            ods, shotnumber=shotnumber, options=option, mat_file=ion_matfile
        )
        print(f"[SUCCESS] charge_exchange ({option}) loaded for shot {shotnumber}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load {option} for shot {shotnumber}: {exc}")
        return None

    database.save(ods, shotnumber)
    print(f"[SAVED] charge_exchange for shot {shotnumber}")
    return ods, shotnumber, option


def fit_charge_exchange_profile_auto_all_times(ods, shotnumber):
    """Fit Ti/Vtor at every charge_exchange time with a geqdsk (QA/coverage check).

    Returns the number of times with a usable ion fit.
    """
    try:
        time_array = np.asarray(ods["charge_exchange.time"], dtype=float) * 1e3  # ms
    except Exception:
        print(f"[INFO] shot {shotnumber}: no charge_exchange.time")
        return 0

    fitted = 0
    for time_ms in time_array:
        geq = geqdsk_for_time(shotnumber, time_ms)
        if geq is None:
            continue
        try:
            mapped_rho = process.equilibrium_mapping_charge_exchange(ods, geq)
            process.profile_fitting_charge_exchange(
                ods, float(time_ms), mapped_rho,
                Ti_order=3, Vtor_order=3,
                fitting_function_ti="polynomial", fitting_function_vtor="polynomial",
                ion_index=0,
            )
            fitted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[INFO] {time_ms:.1f} ms: ion fit unavailable ({exc})")
    print(f"[INFO] shot {shotnumber}: ion fit usable at {fitted}/{len(time_array)} times")
    return fitted


def process_ion_file(ion_matfile):
    """One-shot entry point (also usable standalone / from a test)."""
    loaded = update_charge_exchange_auto(ion_matfile)
    if loaded is None:
        return None
    ods, shotnumber, _option = loaded
    if os.path.exists(f"{PUBLIC_BASE}/{shotnumber}/chease"):
        fit_charge_exchange_profile_auto_all_times(ods, shotnumber)
    else:
        print(f"[WARNING] CHEASE dir missing for shot {shotnumber}; raw ingest only")
    return shotnumber, "charge_exchange"


def main():
    processed = load_processed_shots()
    try:
        while True:
            print("[POLLING] Scanning for new IDS/CES .mat files...")
            try:
                for fname in os.listdir(WATCH_DIAG):
                    shotnumber, option = parse_ion_file(fname)
                    if shotnumber is None:
                        continue

                    full_path = os.path.join(WATCH_DIAG, fname)
                    mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
                    key = f"cx:{shotnumber}"
                    prev = processed.get(key)
                    if isinstance(prev, dict) and prev.get("timestamp") == mtime:
                        print(f"[SKIP] {option} shot {shotnumber} already processed")
                        continue

                    print(f"[UPDATE DETECTED] {fname} (shot {shotnumber}, {option})")
                    diag_dir = os.path.join(PUBLIC_BASE, f"{shotnumber}/diagnostics")
                    os.makedirs(diag_dir, exist_ok=True)
                    dest = os.path.join(diag_dir, fname)
                    shutil.copy2(full_path, dest)

                    try:
                        result = process_ion_file(dest)
                        status = result[1] if result else "invalid"
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ERROR] processing {fname} failed: {exc}")
                        status = "invalid"

                    save_processed_shot(shotnumber, mtime, status=status)
                    processed[key] = {"timestamp": mtime, "status": status}
                    print(f"[DONE] charge_exchange shot {shotnumber}: {status}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Polling failed: {exc}")
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\n[STOPPED] charge_exchange updater stopped by user.")


if __name__ == "__main__":
    main()
