#!/home/user1/miniconda3/envs/vaft/bin/python

"""Automatic kinetic core_profiles update (Thomson + charge_exchange -> core_profiles).

Assembly step of the pipeline: it consumes the diagnostics already ingested by
``update_thomson_scattering_and_core_profile.py`` (Thomson) and
``update_charge_exchange.py`` (ion Doppler / CES) and writes **kinetic**
``core_profiles`` -- real Ti / Vtor and ``pressure_thermal = e*ne*(Te+Ti)``, n_i ~= n_e --
onto the equilibrium ``rho_tor_norm`` grid.

Per shot:
  1. Load the shot ODS from the database (already carries thomson_scattering and,
     when available, charge_exchange).
  2. For every Thomson time with a matching CHEASE geqdsk, build kinetic core_profiles
     via ``vaft.code.efit.build_kinetic_core_profiles`` (falls back to a
     Thomson-only slice, Ti=Te, when the ion diagnostic has no data at that time).
  3. ``core_profiles_pressures`` + save the OMAS json and push the ODS to the database.
  4. Record status ``kinetic_core_profile`` / ``core_profile`` / ``thomson_only``.

Run:
  nohup ./update_core_profile.py > core_profile_update.log 2>&1 &
"""
import os
import time
from datetime import datetime

import numpy as np
import omas

import vaft
from vaft.data import read_geqdsk

from vaft import database, process
from vaft.code.efit import build_kinetic_core_profiles

# Reuse the shot-registry + Thomson-file discovery helpers (same directory).
from update_thomson_scattering_and_core_profile import (
    extract_shotnumber_of_thomson_scattering,
    load_processed_shots,
    SOURCE,
    save_processed_shot,
    strip_electron_only_pressure,
)

WATCH_DIAG = "/srv/vest.diagnostic"
PUBLIC_BASE = "/srv/vest.filedb/public"
CHECK_INTERVAL = 10  # seconds

# kinetic core_profiles records live in their own registry group so they don't
# collide with the Thomson 'shots' group. Mirrors vaft.database.utils.CP_REGISTRY_GROUP.
CP_REGISTRY_GROUP = "cp_shots"


def geqdsk_for_time(shotnumber, time_ms):
    """Return a loaded CHEASE geqdsk for (shot, time_ms), or None."""
    path = f"{PUBLIC_BASE}/{shotnumber}/chease/g0{shotnumber}.00{int(time_ms):03d}"
    if not os.path.exists(path):
        print(f"[WARNING] Geqdsk not found: {path}")
        return None
    try:
        return read_geqdsk(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not load geqdsk {path}: {exc}")
        return None


def build_core_profiles_all_times(ods, shotnumber):
    """Build (kinetic) core_profiles for every Thomson time with a geqdsk.

    Returns 'kinetic_core_profile' if any slice got real Ti, 'core_profile' if only
    Thomson-fitted slices were written, or 'thomson_only' if none.
    """
    has_ion = "charge_exchange" in ods and len(ods["charge_exchange.channel"]) > 0
    try:
        time_array = np.asarray(ods["thomson_scattering.time"], dtype=float) * 1e3  # ms
    except Exception:
        print(f"[ERROR] shot {shotnumber}: no thomson_scattering.time")
        return "thomson_only"

    # Clean rebuild: drop any previously-stored core_profiles so a re-run cannot
    # leave stale (e.g. Ti=Te) slices behind.
    if "core_profiles" in ods:
        del ods["core_profiles"]

    n_kinetic = 0
    n_thomson = 0
    for time_ms in time_array:
        geq = geqdsk_for_time(shotnumber, time_ms)
        if geq is None:
            continue

        built_kinetic = False
        if has_ion:
            try:
                # require_ion: only build a KINETIC slice where the ion diagnostic
                # actually covers this time. ti_te_fallback=False so a time without
                # ion data never gets a phantom Ti=Te / ion density / ion pressure.
                build_kinetic_core_profiles(
                    ods, geq, float(time_ms), ion_index=0,
                    require_thomson=True, require_ion=True, ti_te_fallback=False,
                )
                built_kinetic = True
                n_kinetic += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[INFO] {time_ms:.1f} ms: no ion data here ({exc}); electron-only")

        if not built_kinetic:
            try:
                mapped_rho = process.equilibrium_mapping_thomson_scattering(ods, geq)
                n_e_fn, T_e_fn, *_ = process.profile_fitting_thomson_scattering(
                    ods, float(time_ms), mapped_rho,
                    Te_order=2, Ne_order=2,
                    fitting_function_te="polynomial", fitting_function_ne="exponential",
                )
                # electron-only slice: ne/Te only, no phantom ion block
                process.core_profiles(
                    ods, float(time_ms), mapped_rho, n_e_fn, T_e_fn,
                    geq=geq, ti_te_fallback=False,
                )
                n_thomson += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] {time_ms:.1f} ms: Thomson fit failed ({exc})")
                continue

        try:
            omas.omas_physics.core_profiles_pressures(ods, update=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {time_ms:.1f} ms: core_profiles_pressures failed ({exc})")

    if n_kinetic == 0 and n_thomson == 0:
        print(f"[INFO] shot {shotnumber}: no core_profiles built (missing geqdsk/diagnostics)")
        return "thomson_only"

    # Option B: electron-only slices carry no total (kinetic) pressure.
    strip_electron_only_pressure(ods)

    omas.save_omas_json(
        ods, f"{PUBLIC_BASE}/{shotnumber}/omas/{shotnumber}_core_profile.json"
    )
    database.save(ods, shotnumber, source=SOURCE)
    print(f"[SAVED] shot {shotnumber}: {n_kinetic} kinetic + {n_thomson} Thomson-only slices")
    return "kinetic_core_profile" if n_kinetic > 0 else "core_profile"


def process_shot(shotnumber):
    """Build core_profiles for a single shot already loaded in the database."""
    try:
        ods = database.load(shotnumber, source=SOURCE)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load ODS for shot {shotnumber}: {exc}")
        return None
    if not os.path.exists(f"{PUBLIC_BASE}/{shotnumber}/chease"):
        print(f"[WARNING] CHEASE dir missing for shot {shotnumber}")
        return shotnumber, "thomson_only"
    status = build_core_profiles_all_times(ods, shotnumber)
    return shotnumber, status


def main():
    processed = load_processed_shots(group=CP_REGISTRY_GROUP)
    try:
        while True:
            print("[POLLING] Scanning for shots to (re)build core_profiles...")
            try:
                for fname in os.listdir(WATCH_DIAG):
                    if not fname.endswith(".mat"):
                        continue
                    shotnumber = extract_shotnumber_of_thomson_scattering(fname)
                    if shotnumber is None:
                        continue
                    full_path = os.path.join(WATCH_DIAG, fname)
                    mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
                    prev = processed.get(shotnumber)
                    if isinstance(prev, dict) and prev.get("timestamp") == mtime:
                        print(f"[SKIP] core_profiles for shot {shotnumber} already built")
                        continue

                    print(f"[BUILD] core_profiles for shot {shotnumber}")
                    try:
                        result = process_shot(shotnumber)
                        status = result[1] if result else "invalid"
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ERROR] core_profiles build for shot {shotnumber} failed: {exc}")
                        status = "invalid"

                    save_processed_shot(shotnumber, mtime, status=status, group=CP_REGISTRY_GROUP)
                    processed[shotnumber] = {"timestamp": mtime, "status": status}
                    print(f"[DONE] shot {shotnumber}: {status}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Polling failed: {exc}")
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\n[STOPPED] core_profile updater stopped by user.")


if __name__ == "__main__":
    main()
