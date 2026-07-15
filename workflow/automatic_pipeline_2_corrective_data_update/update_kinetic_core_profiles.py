#!/home/user1/miniconda3/envs/vaft/bin/python

"""Automatic KINETIC core-profile update (Thomson + ion Doppler/CES).

Companion to ``update_thomson_scattering_and_core_profile.py``. That script fits
electron profiles (ne/Te) from Thomson scattering only; this one additionally
ingests the ion diagnostic (IDS = Ion Doppler Spectroscopy, or CES = Charge
Exchange Spectroscopy -> the OMAS ``charge_exchange`` IDS) and writes **kinetic**
``core_profiles`` with real Ti / Vtor and ``pressure_thermal = e*ne*(Te+Ti)``.

Per detected shot:
  1. Load the shot ODS from the database.
  2. Load Thomson (``NeTe_<shot>.mat``) and the ion diagnostic
     (``IDS_<shot>.mat`` or ``CES_<shot>.mat``) into the ODS.
  3. For every Thomson time with a matching CHEASE geqdsk, build kinetic
     core_profiles via ``vaft.code.kineticEfit.build_kinetic_core_profiles``
     (falls back to a Thomson-only slice when the ion diagnostic has no data at
     that time), then save the OMAS json + push the ODS back to the database.
  4. Record the shot in ``processed_shots.h5`` (status ``kinetic_core_profile`` /
     ``core_profile`` / ``thomson_only``).

EFIT / CHEASE reconstruction is out of scope (that is pipeline-1 / the
``vaft.code`` adapters); this pipeline only corrects the kinetic profiles.

Run:
  nohup ./update_kinetic_core_profiles.py > kinetic_core_profile_update.log 2>&1 &
"""
import os
import re
import shutil
import time
from datetime import datetime

import omas
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

from vaft import database, machine_mapping
from vaft.code.kineticEfit import build_kinetic_core_profiles

# Reuse the shot registry + discovery helpers from the Thomson updater (same dir).
from update_thomson_scattering_and_core_profile import (
    extract_shotnumber_of_thomson_scattering,
    load_processed_shots,
    save_processed_shot,
)

WATCH_DIAG = "/srv/vest.diagnostic"
PUBLIC_BASE = "/srv/vest.filedb/public"
CHECK_INTERVAL = 10  # seconds

# Ion-diagnostic filename -> charge_exchange option. IDS/CES both feed the
# `charge_exchange` IDS; only the loader differs.
ION_PATTERNS = (
    (re.compile(r"^IDS[_-](\d+)", re.IGNORECASE), "ids"),
    (re.compile(r"^CES[_-](\d+)", re.IGNORECASE), "ces"),
)


def resolve_ion_diagnostic(shotnumber, search_dirs):
    """Return (option, mat_path) for the ion diagnostic of ``shotnumber``, or (None, None).

    Prefers IDS over CES; looks in each of ``search_dirs`` in order.
    """
    # Prefer IDS over CES: scan for each pattern (IDS first) across all dirs.
    for pattern, option in ION_PATTERNS:
        for directory in search_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            for fname in sorted(os.listdir(directory)):
                m = pattern.match(fname)
                if m and int(m.group(1)) == int(shotnumber) and fname.endswith(".mat"):
                    return option, os.path.join(directory, fname)
    return None, None


def geqdsk_for_time(shotnumber, time_ms):
    """Return a loaded CHEASE geqdsk for (shot, time_ms), or None if unavailable."""
    path = f"{PUBLIC_BASE}/{shotnumber}/chease/g0{shotnumber}.00{int(time_ms):03d}"
    if not os.path.exists(path):
        print(f"[WARNING] Geqdsk not found: {path}")
        return None
    try:
        geq = OMFITgeqdsk(filename=path)
        geq["fluxSurfaces"].load()
        return geq
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not load geqdsk {path}: {exc}")
        return None


def update_diagnostics_auto(ts_filepath, ion_option, ion_matfile):
    """Load the ODS and ingest Thomson + (optional) ion diagnostic. Returns (ods, shot)."""
    filename = os.path.basename(ts_filepath)
    shotnumber = extract_shotnumber_of_thomson_scattering(filename)
    if shotnumber is None:
        print(f"[ERROR] Could not parse shot number from {filename}")
        return None

    try:
        ods = database.load(shotnumber, "public")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load ODS for shot {shotnumber}: {exc}")
        return None

    try:
        machine_mapping.thomson_scattering(ods, shotnumber, ts_filepath)
        print(f"[SUCCESS] Thomson loaded for shot {shotnumber}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load Thomson for shot {shotnumber}: {exc}")
        return None

    if ion_matfile is not None:
        try:
            machine_mapping.charge_exchange(
                ods, shotnumber=shotnumber, options=ion_option, mat_file=ion_matfile
            )
            print(f"[SUCCESS] Ion diagnostic ({ion_option}) loaded for shot {shotnumber}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Failed to load ion diagnostic for shot {shotnumber}: {exc}")
            ion_matfile = None

    database.save(ods, shotnumber)
    return ods, shotnumber, ion_matfile


def fit_kinetic_profiles_all_times(ods, shotnumber, has_ion):
    """Build kinetic core_profiles for every Thomson time with a geqdsk.

    Returns 'kinetic_core_profile' if any slice got real Ti, 'core_profile' if
    only Thomson-fitted slices were written, or 'thomson_only' if none.
    """
    import numpy as np

    time_array = np.asarray(ods["thomson_scattering.time"], dtype=float) * 1e3  # ms
    n_kinetic = 0
    n_thomson = 0

    for time_ms in time_array:
        geq = geqdsk_for_time(shotnumber, time_ms)
        if geq is None:
            continue

        # Try the full kinetic build (TS + ion -> Ti/Vtor + pressure_thermal).
        built_kinetic = False
        if has_ion:
            try:
                build_kinetic_core_profiles(ods, geq, float(time_ms), ion_index=0)
                built_kinetic = True
                n_kinetic += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[INFO] {time_ms:.1f} ms: no kinetic (ion) fit ({exc}); Thomson-only")

        if not built_kinetic:
            # Thomson-only fallback (electron profiles, Ti = Te).
            from vaft import process
            try:
                mapped_rho = process.equilibrium_mapping_thomson_scattering(ods, geq)
                n_e_fn, T_e_fn, *_ = process.profile_fitting_thomson_scattering(
                    ods, float(time_ms), mapped_rho,
                    Te_order=2, Ne_order=2,
                    fitting_function_te="polynomial", fitting_function_ne="exponential",
                )
                process.core_profiles(ods, float(time_ms), mapped_rho, n_e_fn, T_e_fn)
                n_thomson += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] {time_ms:.1f} ms: Thomson fit failed ({exc})")
                continue

        try:
            omas.omas_physics.core_profiles_pressures(ods, update=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {time_ms:.1f} ms: core_profiles_pressures failed ({exc})")

    if n_kinetic == 0 and n_thomson == 0:
        return "thomson_only"

    omas.save_omas_json(
        ods, f"{PUBLIC_BASE}/{shotnumber}/omas/{shotnumber}_core_profile.json"
    )
    database.save(ods, shotnumber)
    print(f"[SAVED] shot {shotnumber}: {n_kinetic} kinetic + {n_thomson} Thomson-only slices")
    return "kinetic_core_profile" if n_kinetic > 0 else "core_profile"


def process_shot_from_thomson_file(ts_filepath):
    """One-shot processing entry point (also usable standalone / from a test)."""
    filename = os.path.basename(ts_filepath)
    shotnumber = extract_shotnumber_of_thomson_scattering(filename)
    if shotnumber is None:
        return None
    ion_option, ion_matfile = resolve_ion_diagnostic(
        shotnumber, [os.path.dirname(ts_filepath), WATCH_DIAG,
                     f"{PUBLIC_BASE}/{shotnumber}/diagnostics"]
    )
    loaded = update_diagnostics_auto(ts_filepath, ion_option, ion_matfile)
    if loaded is None:
        return None
    ods, shotnumber, ion_matfile = loaded
    if not os.path.exists(f"{PUBLIC_BASE}/{shotnumber}/chease"):
        print(f"[WARNING] CHEASE dir missing for shot {shotnumber}; Thomson-only load")
        return shotnumber, "thomson_only"
    status = fit_kinetic_profiles_all_times(ods, shotnumber, has_ion=ion_matfile is not None)
    return shotnumber, status


def main():
    processed = load_processed_shots()
    try:
        while True:
            print("[POLLING] Scanning for new Thomson .mat files...")
            try:
                for fname in os.listdir(WATCH_DIAG):
                    if not fname.endswith(".mat"):
                        continue
                    # Drive off the Thomson file; the ion diagnostic is resolved per shot.
                    if not (re.search(r"NeTe", fname, re.IGNORECASE) or re.match(r"^\d+_", fname)):
                        continue
                    shotnumber = extract_shotnumber_of_thomson_scattering(fname)
                    if shotnumber is None:
                        continue

                    full_path = os.path.join(WATCH_DIAG, fname)
                    mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
                    prev = processed.get(shotnumber)
                    if isinstance(prev, str):
                        prev = {"timestamp": prev, "status": "unknown"}
                    if prev and prev.get("timestamp") == mtime:
                        print(f"[SKIP] shot {shotnumber} already processed ({prev.get('status')})")
                        continue

                    print(f"[UPDATE DETECTED] {fname} (shot {shotnumber})")
                    diag_dir = os.path.join(PUBLIC_BASE, f"{shotnumber}/diagnostics")
                    os.makedirs(diag_dir, exist_ok=True)
                    dest = os.path.join(diag_dir, fname)
                    shutil.copy2(full_path, dest)

                    try:
                        result = process_shot_from_thomson_file(dest)
                        status = result[1] if result else "invalid"
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ERROR] processing shot {shotnumber} failed: {exc}")
                        status = "invalid"

                    save_processed_shot(shotnumber, mtime, status=status)
                    processed[shotnumber] = {"timestamp": mtime, "status": status}
                    print(f"[DONE] shot {shotnumber}: {status}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Polling failed: {exc}")
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\n[STOPPED] Kinetic core-profile updater stopped by user.")


if __name__ == "__main__":
    main()
