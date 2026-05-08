import logging
from pathlib import Path
from typing import Dict, List, Optional

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

OUTPUT_FILENAME = "equilibrium_global_history.xlsx"
EXPECTED_COLUMNS = [
    "shot",
    "eq_index",
    "time_s",
    "ip_A",
    "psi_axis_Wb",
    "psi_boundary_Wb",
    "q_axis",
    "q_95",
    "q_min",
    "beta_pol",
    "beta_tor",
    "beta_normal",
    "li_3",
    "energy_mhd_J",
    "area_m2",
    "volume_m3",
    "major_radius_m",
    "minor_radius_m",
    "aspect_ratio",
    "elongation",
    "triangularity",
    "triangularity_upper",
    "triangularity_lower",
    "magnetic_axis_r_m",
    "magnetic_axis_z_m",
    "magnetic_axis_btor_T",
    "vacuum_b0_T",
    "vacuum_r0_m",
    "measured_dia_flux_Wb",
    "reconstructed_dia_flux_Wb",
    "dia_flux_cmp_measured_Wb",
    "dia_flux_cmp_computed_Wb",
    "dia_flux_cmp_difference_Wb",
    "dia_flux_cmp_relative_error",
    # Virial/Shafranov integral outputs (surface + volumetric terms).
    "virial_s1",
    "virial_s2",
    "virial_s3",
    "virial_alpha",
    "virial_b_pa_T",
    "virial_mui",
    "virial_rt_m",
    "virial_phi_dia_comp_Wb",
    "virial_volume_m3",
    "virial_beta_pd_vir",
    "virial_beta",
    "virial_li",
    "virial_beta_lao",
    "virial_li_lao",
    "virial_beta_bongard",
    "virial_li_bongard",
]
KEY_COLUMNS = ["shot", "eq_index", "time_s"]
SORT_COLUMNS = ["shot", "time_s", "eq_index"]
REQUIRED_COLUMNS_FOR_REPAIR = [
    "q_min",
    "beta_pol",
    "beta_tor",
    "beta_normal",
    "li_3",
    "volume_m3",
    "virial_beta_lao",
    "virial_li_lao",
]


def get_all_processed_shots() -> List[int]:
    """Return all shot numbers listed in processed_shots index."""
    df = db_ods.exist_ts_file()
    if df is None or len(df) == 0:
        logger.warning("No processed shots found from exist_ts_file().")
        return []

    if "Shot Number" not in df.columns:
        logger.warning("Column 'Shot Number' is missing in processed shot index.")
        return []

    shots = df["Shot Number"].astype(int).tolist()
    logger.info("Found %d processed shots", len(shots))
    return shots


def _safe_get(container, key: str, default=np.nan):
    try:
        return container[key]
    except Exception:
        return default


def _as_float(value):
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return np.nan
        return float(arr.reshape(-1)[0])
    except Exception:
        return np.nan


def _extract_time(eq_ts, eq_times, idx: int) -> float:
    if "time" in eq_ts:
        return _as_float(eq_ts["time"])
    if eq_times is not None and idx < len(eq_times):
        return _as_float(eq_times[idx])
    return float(idx)


def _extract_q_min(eq_ts) -> float:
    q_min_obj = _safe_get(eq_ts, "global_quantities.q_min", np.nan)
    q_min = _as_float(q_min_obj)
    if np.isfinite(q_min):
        return q_min
    # Compatibility with structures storing q_min.value
    return _as_float(_safe_get(eq_ts, "global_quantities.q_min.value", np.nan))


def _extract_virial_by_slice(virial_out: Dict[int, Dict], eq_idx: int):
    if eq_idx not in virial_out:
        return {
            "virial_s1": np.nan,
            "virial_s2": np.nan,
            "virial_s3": np.nan,
            "virial_alpha": np.nan,
            "virial_b_pa_T": np.nan,
            "virial_mui": np.nan,
            "virial_rt_m": np.nan,
            "virial_phi_dia_comp_Wb": np.nan,
            "virial_volume_m3": np.nan,
            "virial_beta_pd_vir": np.nan,
            "virial_beta_lao": np.nan,
            "virial_li_lao": np.nan,
            "virial_beta_bongard": np.nan,
            "virial_li_bongard": np.nan,
        }
    vir = virial_out[eq_idx]
    # Surface integrals (dimensionless Shafranov integrals).
    s1 = _as_float(vir.get("s_1", np.nan))
    s2 = _as_float(vir.get("s_2", np.nan))
    s3 = _as_float(vir.get("s_3", np.nan))
    # Volumetric terms / closure parameters.
    alpha = _as_float(vir.get("alpha", np.nan))
    b_pa = _as_float(vir.get("B_pa", np.nan))
    mui = _as_float(vir.get("mui", vir.get("mui_hat", np.nan)))
    rt = _as_float(vir.get("rt", np.nan))
    phi_dia_comp = _as_float(vir.get("phi_dia_comp", np.nan))
    v_p = _as_float(vir.get("V_p", np.nan))
    beta_pd_vir = _as_float(vir.get("beta_pd_vir", np.nan))

    # Derived beta/li outputs (keep existing naming in this sheet).
    beta_lao = _as_float(vir.get("beta_p_vir_lao", vir.get("beta_p_vir", np.nan)))
    li_lao = _as_float(vir.get("li_vir_lao", vir.get("li_vir", np.nan)))
    beta_bongard = _as_float(vir.get("beta_p_vir_bongard", np.nan))
    li_bongard = _as_float(vir.get("li_vir_bongard", np.nan))
    return {
        "virial_s1": s1,
        "virial_s2": s2,
        "virial_s3": s3,
        "virial_alpha": alpha,
        "virial_b_pa_T": b_pa,
        "virial_mui": mui,
        "virial_rt_m": rt,
        "virial_phi_dia_comp_Wb": phi_dia_comp,
        "virial_volume_m3": v_p,
        "virial_beta_pd_vir": beta_pd_vir,
        "virial_beta_lao": beta_lao,
        "virial_li_lao": li_lao,
        "virial_beta_bongard": beta_bongard,
        "virial_li_bongard": li_bongard,
    }


def _extract_dia_flux_cmp_by_slice(dia_flux_cmp_out: Dict[int, Dict], eq_idx: int):
    if eq_idx not in dia_flux_cmp_out:
        return np.nan, np.nan, np.nan, np.nan
    cmp_out = dia_flux_cmp_out[eq_idx]
    return (
        _as_float(cmp_out.get("measured", np.nan)),
        _as_float(cmp_out.get("computed", np.nan)),
        _as_float(cmp_out.get("difference", np.nan)),
        _as_float(cmp_out.get("relative_error", np.nan)),
    )


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


def extract_equilibrium_global_rows(ods, shot_number: int) -> List[dict]:
    """Extract equilibrium globals + diamagnetic/virial quantities for one shot."""
    if "equilibrium.time_slice" not in ods or len(ods["equilibrium.time_slice"]) == 0:
        logger.warning("Shot %s: equilibrium.time_slice not found", shot_number)
        return []

    # Fill derived/optional quantities before extraction.
    for updater in (
        vaft.omas.update_equilibrium_boundary,
        vaft.omas.update_equilibrium_global_quantities_q_min,
        vaft.omas.update_equilibrium_global_quantities_volume,
        vaft.omas.update_equilibrium_stored_energy,
    ):
        try:
            updater(ods, time_slice=None)
        except Exception as exc:
            logger.debug("Shot %s: updater %s failed: %s", shot_number, updater.__name__, exc)

    try:
        vaft.omas.update_equilibrium_constraints_diamagnetic_flux(ods, time_slice=None)
    except Exception as exc:
        logger.debug("Shot %s: diamagnetic constraints update failed: %s", shot_number, exc)

    try:
        virial_out = vaft.omas.compute_virial_equilibrium_quantities_ods(ods, time_slice=None)
    except Exception as exc:
        logger.debug("Shot %s: virial computation failed: %s", shot_number, exc)
        virial_out = {}

    try:
        dia_flux_cmp_out = vaft.omas.compute_diamagnetic_flux_measured_vs_computed(
            ods, time_slice=None
        )
    except Exception as exc:
        logger.debug(
            "Shot %s: compute_diamagnetic_flux_measured_vs_computed failed: %s",
            shot_number,
            exc,
        )
        dia_flux_cmp_out = {}

    eq_times = np.asarray(ods["equilibrium.time"], dtype=float) if "equilibrium.time" in ods else None
    b0 = _as_float(_safe_get(ods, "equilibrium.vacuum_toroidal_field.b0", np.nan))
    r0 = _as_float(_safe_get(ods, "equilibrium.vacuum_toroidal_field.r0", np.nan))

    rows: List[dict] = []
    n_eq = len(ods["equilibrium.time_slice"])
    for eq_idx in range(n_eq):
        eq_ts = ods["equilibrium.time_slice"][eq_idx]
        time_s = _extract_time(eq_ts, eq_times, eq_idx)

        major_r = _as_float(_safe_get(eq_ts, "boundary.geometric_axis.r", np.nan))
        minor_r = _as_float(_safe_get(eq_ts, "boundary.minor_radius", np.nan))
        aspect_ratio = major_r / minor_r if np.isfinite(major_r) and np.isfinite(minor_r) and minor_r != 0 else np.nan
        vir = _extract_virial_by_slice(virial_out, eq_idx)
        dia_cmp_measured, dia_cmp_computed, dia_cmp_difference, dia_cmp_rel_err = (
            _extract_dia_flux_cmp_by_slice(dia_flux_cmp_out, eq_idx)
        )

        row = {
            "shot": int(shot_number),
            "eq_index": int(eq_idx),
            "time_s": float(time_s),
            "ip_A": _as_float(_safe_get(eq_ts, "global_quantities.ip", np.nan)),
            "psi_axis_Wb": _as_float(_safe_get(eq_ts, "global_quantities.psi_axis", np.nan)),
            "psi_boundary_Wb": _as_float(_safe_get(eq_ts, "global_quantities.psi_boundary", np.nan)),
            "q_axis": _as_float(_safe_get(eq_ts, "global_quantities.q_axis", np.nan)),
            "q_95": _as_float(_safe_get(eq_ts, "global_quantities.q_95", np.nan)),
            "q_min": _extract_q_min(eq_ts),
            "beta_pol": _as_float(_safe_get(eq_ts, "global_quantities.beta_pol", np.nan)),
            "beta_tor": _as_float(_safe_get(eq_ts, "global_quantities.beta_tor", np.nan)),
            "beta_normal": _as_float(_safe_get(eq_ts, "global_quantities.beta_normal", np.nan)),
            "li_3": _as_float(_safe_get(eq_ts, "global_quantities.li_3", np.nan)),
            "energy_mhd_J": _as_float(_safe_get(eq_ts, "global_quantities.energy_mhd", np.nan)),
            "area_m2": _as_float(_safe_get(eq_ts, "global_quantities.area", np.nan)),
            "volume_m3": _as_float(_safe_get(eq_ts, "global_quantities.volume", np.nan)),
            "major_radius_m": major_r,
            "minor_radius_m": minor_r,
            "aspect_ratio": float(aspect_ratio),
            "elongation": _as_float(_safe_get(eq_ts, "boundary.elongation", np.nan)),
            "triangularity": _as_float(_safe_get(eq_ts, "boundary.triangularity", np.nan)),
            "triangularity_upper": _as_float(_safe_get(eq_ts, "boundary.triangularity_upper", np.nan)),
            "triangularity_lower": _as_float(_safe_get(eq_ts, "boundary.triangularity_lower", np.nan)),
            "magnetic_axis_r_m": _as_float(_safe_get(eq_ts, "global_quantities.magnetic_axis.r", np.nan)),
            "magnetic_axis_z_m": _as_float(_safe_get(eq_ts, "global_quantities.magnetic_axis.z", np.nan)),
            "magnetic_axis_btor_T": _as_float(
                _safe_get(eq_ts, "global_quantities.magnetic_axis.b_field_tor", np.nan)
            ),
            "vacuum_b0_T": float(b0),
            "vacuum_r0_m": float(r0),
            "measured_dia_flux_Wb": _as_float(
                _safe_get(eq_ts, "constraints.diamagnetic_flux.measured", np.nan)
            ),
            "reconstructed_dia_flux_Wb": _as_float(
                _safe_get(eq_ts, "constraints.diamagnetic_flux.reconstructed", np.nan)
            ),
            "dia_flux_cmp_measured_Wb": float(dia_cmp_measured),
            "dia_flux_cmp_computed_Wb": float(dia_cmp_computed),
            "dia_flux_cmp_difference_Wb": float(dia_cmp_difference),
            "dia_flux_cmp_relative_error": float(dia_cmp_rel_err),
            # Virial/Shafranov: surface + volumetric integrals/terms (store all scalar outputs).
            "virial_s1": float(vir["virial_s1"]),
            "virial_s2": float(vir["virial_s2"]),
            "virial_s3": float(vir["virial_s3"]),
            "virial_alpha": float(vir["virial_alpha"]),
            "virial_b_pa_T": float(vir["virial_b_pa_T"]),
            "virial_mui": float(vir["virial_mui"]),
            "virial_rt_m": float(vir["virial_rt_m"]),
            "virial_phi_dia_comp_Wb": float(vir["virial_phi_dia_comp_Wb"]),
            "virial_volume_m3": float(vir["virial_volume_m3"]),
            "virial_beta_pd_vir": float(vir["virial_beta_pd_vir"]),
            # Backward-compatible aliases use Lao definition.
            "virial_beta": float(vir["virial_beta_lao"]),
            "virial_li": float(vir["virial_li_lao"]),
            "virial_beta_lao": float(vir["virial_beta_lao"]),
            "virial_li_lao": float(vir["virial_li_lao"]),
            "virial_beta_bongard": float(vir["virial_beta_bongard"]),
            "virial_li_bongard": float(vir["virial_li_bongard"]),
        }
        rows.append(row)

    return rows


def generate_equilibrium_global_history_excel(
    shot_numbers: Optional[List[int]] = None,
    max_shots: Optional[int] = None,
    directory: str = "public",
    output_path: Optional[str] = None,
    rebuild: bool = False,
    save_every: int = 10,
) -> Optional[pd.DataFrame]:
    """Generate and incrementally update equilibrium global history Excel."""
    if output_path is None:
        output_path = str(Path(__file__).with_name(OUTPUT_FILENAME))

    if shot_numbers is None:
        shot_numbers = get_all_processed_shots()
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
        "Equilibrium sheet candidates=%d, already-complete=%d, missing=%d, defective=%d, to-process=%d",
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

    for shot in tqdm(target_shots, desc="Processing shots"):
        try:
            ods = db_ods.load(int(shot), directory=directory)
            shot_rows = extract_equilibrium_global_rows(ods, int(shot))
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

    parser = argparse.ArgumentParser(description="Generate equilibrium global history Excel file")
    parser.add_argument("--max-shots", type=int, default=None, help="Maximum number of shots to process")
    parser.add_argument("--directory", type=str, default="public", help="HDF5 directory (default: public)")
    parser.add_argument("--output", type=str, default=None, help="Output Excel path")
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing Excel and rebuild from scratch")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N processed shots")
    args = parser.parse_args()

    generate_equilibrium_global_history_excel(
        shot_numbers=None,
        max_shots=args.max_shots,
        directory=args.directory,
        output_path=args.output,
        rebuild=args.rebuild,
        save_every=args.save_every,
    )
