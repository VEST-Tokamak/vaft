import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import vaft
from vaft.database import ods as db_ods


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

QE = 1.602176634e-19
OUTPUT_FILENAME = "volume_averaged_parameters.xlsx"
EXPECTED_COLUMNS = [
    "shot",
    "cp_index",
    "eq_index",
    "time_core_s",
    "time_equilibrium_s",
    "time_diff_s",
    "n_e_vol_avg_m-3",
    "T_e_vol_avg_eV",
    "P_e_vol_avg_Pa",
    "P_eq_vol_avg_Pa",
    "P_e_over_P_eq",
]
KEY_COLUMNS = ["shot", "cp_index", "time_core_s"]
SORT_COLUMNS = ["shot", "time_core_s", "cp_index"]
REQUIRED_COLUMNS_FOR_REPAIR = [
    "n_e_vol_avg_m-3",
    "T_e_vol_avg_eV",
    "P_e_vol_avg_Pa",
    "P_eq_vol_avg_Pa",
    "P_e_over_P_eq",
]


def get_core_profile_shots() -> List[int]:
    """Return shot numbers whose processed status is `core_profile`."""
    df = db_ods.exist_ts_file()
    if df is None or len(df) == 0:
        logger.warning("No processed shots found.")
        return []

    required_columns = {"Status", "Shot Number"}
    if not required_columns.issubset(df.columns):
        logger.warning("Processed-shots table is missing required columns: %s", required_columns)
        return []

    shots = df[df["Status"] == "core_profile"]["Shot Number"].astype(int).tolist()
    logger.info("Found %d shots with core_profile status", len(shots))
    return shots


def _extract_times(ods, slice_path: str, time_path: str, n_slices: int) -> np.ndarray:
    times = []
    for idx in range(n_slices):
        ts = ods[slice_path][idx]
        if "time" in ts:
            times.append(float(ts["time"]))
        elif time_path in ods and idx < len(ods[time_path]):
            times.append(float(ods[time_path][idx]))
        else:
            times.append(float(idx))
    return np.asarray(times, dtype=float)


def _as_len(arr: Iterable[float], n: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    arr_np = np.asarray(list(arr), dtype=float)
    m = min(n, len(arr_np))
    if m > 0:
        out[:m] = arr_np[:m]
    return out


def _load_existing_or_empty(output_path: str, expected_columns: List[str]) -> pd.DataFrame:
    path = Path(output_path)
    if not path.exists():
        return pd.DataFrame(columns=expected_columns)
    try:
        existing_df = pd.read_excel(path)
        for col in expected_columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        return existing_df
    except Exception as exc:
        logger.warning("Failed to read existing Excel %s: %s. Starting from empty.", output_path, exc)
        return pd.DataFrame(columns=expected_columns)


def _has_invalid_values(df: pd.DataFrame, columns: List[str]) -> bool:
    if df.empty:
        return True
    for col in columns:
        if col not in df.columns:
            return True
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            return True
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            return True
    return False


def _get_target_shots(
    candidate_shots: List[int], existing_df: pd.DataFrame, required_columns: List[str]
) -> tuple[List[int], int, int, int]:
    if existing_df.empty or "shot" not in existing_df.columns:
        return sorted({int(s) for s in candidate_shots}), 0, len(candidate_shots), 0

    existing_shots = {
        int(s)
        for s in pd.to_numeric(existing_df["shot"], errors="coerce").dropna().astype(int).tolist()
    }
    missing_shots = [int(s) for s in candidate_shots if int(s) not in existing_shots]

    defective_shots: List[int] = []
    for shot in sorted(existing_shots.intersection({int(s) for s in candidate_shots})):
        shot_df = existing_df[existing_df["shot"] == shot]
        if _has_invalid_values(shot_df, required_columns):
            defective_shots.append(int(shot))

    target_shots = sorted(set(missing_shots + defective_shots))
    completed_count = len(candidate_shots) - len(target_shots)
    return target_shots, completed_count, len(missing_shots), len(defective_shots)


def _merge_upsert(
    existing_df: pd.DataFrame,
    new_rows_df: pd.DataFrame,
    key_columns: List[str],
    sort_columns: List[str],
    expected_columns: List[str],
) -> pd.DataFrame:
    if existing_df.empty:
        merged = new_rows_df.copy()
    elif new_rows_df.empty:
        merged = existing_df.copy()
    else:
        merged = pd.concat([existing_df, new_rows_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_columns, keep="last")

    for col in expected_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[expected_columns]
    if not merged.empty:
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
    return merged


def _save_excel(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def extract_volume_averaged_parameters(ods, shot_number: int) -> List[dict]:
    """
    Extract per-core-profile timepoint data:
    - <n_e>, <T_e> from core_profiles.global_quantities (volume average)
    - 2 * <n_e> * <T_e> * e as electron-pressure proxy (Z_eff ignored)
    - <P> from equilibrium volume-averaged pressure
    """
    if "core_profiles.profiles_1d" not in ods or len(ods["core_profiles.profiles_1d"]) == 0:
        logger.warning("Shot %s: no core_profiles.profiles_1d", shot_number)
        return []
    if "equilibrium.time_slice" not in ods or len(ods["equilibrium.time_slice"]) == 0:
        logger.warning("Shot %s: no equilibrium.time_slice", shot_number)
        return []

    n_cp = len(ods["core_profiles.profiles_1d"])
    n_eq = len(ods["equilibrium.time_slice"])

    # Fill core_profiles.global_quantities with volume-averaged n_e and T_e
    vaft.omas.update_core_profiles_global_quantities_volume_average(ods)
    gq = ods["core_profiles"].get("global_quantities", {})
    ne_vol = _as_len(gq.get("n_e_volume_average", []), n_cp)
    te_vol = _as_len(gq.get("t_e_volume_average", []), n_cp)
    p_e_vol = 2.0 * ne_vol * te_vol * QE

    # Equilibrium volume-averaged pressure for all equilibrium slices
    try:
        p_eq_all = vaft.omas.compute_volume_averaged_pressure(ods, time_slice=None, option="equilibrium")
    except Exception as exc:
        logger.warning("Shot %s: failed to compute equilibrium volume-averaged pressure: %s", shot_number, exc)
        p_eq_all = np.full(n_eq, np.nan, dtype=float)
    p_eq_all = _as_len(p_eq_all, n_eq)

    # Match update.py behavior:
    # - core profiles are stored under core_profiles.profiles_1d (not time_slice)
    # - equilibrium uses equilibrium.time_slice
    core_times = _extract_times(ods, "core_profiles.profiles_1d", "core_profiles.time", n_cp)
    equil_times = _extract_times(ods, "equilibrium.time_slice", "equilibrium.time", n_eq)

    rows: List[dict] = []
    for cp_idx in range(n_cp):
        cp_time = core_times[cp_idx]
        eq_idx = int(np.argmin(np.abs(equil_times - cp_time)))
        eq_time = float(equil_times[eq_idx])
        dt = abs(eq_time - cp_time)
        p_eq = float(p_eq_all[eq_idx]) if eq_idx < len(p_eq_all) else np.nan
        p_core = float(p_e_vol[cp_idx])
        ratio = p_core / p_eq if np.isfinite(p_core) and np.isfinite(p_eq) and p_eq != 0.0 else np.nan

        rows.append(
            {
                "shot": int(shot_number),
                "cp_index": int(cp_idx),
                "eq_index": int(eq_idx),
                "time_core_s": float(cp_time),
                "time_equilibrium_s": eq_time,
                "time_diff_s": float(dt),
                "n_e_vol_avg_m-3": float(ne_vol[cp_idx]),
                "T_e_vol_avg_eV": float(te_vol[cp_idx]),
                "P_e_vol_avg_Pa": p_core,
                "P_eq_vol_avg_Pa": p_eq,
                "P_e_over_P_eq": ratio,
            }
        )
    return rows


def generate_volume_averaged_parameter_sheet(
    shot_numbers: Optional[List[int]] = None,
    max_shots: Optional[int] = None,
    directory: str = "public",
    output_path: Optional[str] = None,
    rebuild: bool = False,
    save_every: int = 10,
    show_shot_progress: bool = False,
) -> Optional[pd.DataFrame]:
    """Generate and incrementally update volume-averaged parameter sheet."""
    if output_path is None:
        output_path = str(Path(__file__).with_name(OUTPUT_FILENAME))

    if shot_numbers is None:
        shot_numbers = get_core_profile_shots()
    if max_shots is not None:
        shot_numbers = shot_numbers[:max_shots]
    shot_numbers = [int(s) for s in shot_numbers]

    if not shot_numbers:
        logger.warning("No shots to process.")
        return None

    existing_df = pd.DataFrame(columns=EXPECTED_COLUMNS) if rebuild else _load_existing_or_empty(output_path, EXPECTED_COLUMNS)
    target_shots, completed_count, missing_count, defective_count = _get_target_shots(
        shot_numbers, existing_df, REQUIRED_COLUMNS_FOR_REPAIR
    )
    logger.info(
        "Volume-average sheet candidates=%d, already-complete=%d, missing=%d, defective=%d, to-process=%d",
        len(shot_numbers),
        completed_count,
        missing_count,
        defective_count,
        len(target_shots),
    )
    if not target_shots:
        logger.info("No missing or defective shots found. Re-saving existing sheet.")
        if not existing_df.empty:
            existing_df = existing_df.sort_values(SORT_COLUMNS).reset_index(drop=True)
        _save_excel(existing_df, output_path)
        return existing_df

    working_df = existing_df.copy()
    save_every = max(1, int(save_every))
    processed = 0

    progress = tqdm(target_shots, desc="Processing shots")
    total_targets = len(target_shots)
    for idx, shot in enumerate(progress, start=1):
        if show_shot_progress:
            progress.set_postfix_str(f"shot={shot} ({idx}/{total_targets})")
        try:
            ods = db_ods.load(int(shot), directory=directory)
            shot_rows = extract_volume_averaged_parameters(ods, int(shot))
            if not shot_rows:
                logger.warning("Shot %s: no rows extracted, keeping existing rows.", shot)
                continue
            shot_df = pd.DataFrame(shot_rows)
            working_df = _merge_upsert(working_df, shot_df, KEY_COLUMNS, SORT_COLUMNS, EXPECTED_COLUMNS)
            processed += 1
            if processed % save_every == 0:
                _save_excel(working_df, output_path)
                logger.info("Checkpoint saved after %d processed shots -> %s", processed, output_path)
        except Exception as exc:
            logger.warning("Shot %s: processing failed: %s", shot, exc)

    if working_df.empty:
        logger.warning("No data rows available after processing.")
        return None

    working_df = working_df.sort_values(SORT_COLUMNS).reset_index(drop=True)
    _save_excel(working_df, output_path)
    logger.info("Saved %d rows to %s", len(working_df), output_path)
    return working_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate volume-averaged parameter sheet.")
    parser.add_argument("--max-shots", type=int, default=None, help="Max number of shots to process")
    parser.add_argument("--directory", type=str, default="public", help="HDF5 directory (default: public)")
    parser.add_argument("--output", type=str, default=None, help="Output xlsx path")
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing Excel and rebuild from scratch")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N processed shots")
    args = parser.parse_args()

    generate_volume_averaged_parameter_sheet(
        max_shots=args.max_shots,
        directory=args.directory,
        output_path=args.output,
        rebuild=args.rebuild,
        save_every=args.save_every,
    )
