"""Canonical VEST triple Langmuir probe builders integrated under machine_mapping.

Raw-signal acquisition, shot-era bias voltage/tip geometry, and IDS
population live here; the backend-independent physics (offset removal,
calibration, Te solve, n_e calculation) live in
:mod:`vaft.process.langmuir`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from vaft.database import raw as raw_db
from vaft.process.langmuir import probe_surface_area, process_triple_probe

from .utils import VestConfigurationError, path_exists, resolve_vest_diagnostic, set_path

logger = logging.getLogger(__name__)

DEFAULT_DT = 4e-5

# The upper assembly (z=0.98 m, raw fields 259/260) was installed starting
# this shot (user-confirmed against the VEST shot log). Before this shot the
# upper assembly is structurally absent -- not merely "unresolved" -- and
# must not be represented or loaded at all, regardless of what the bias
# voltage era table below says for earlier shot ranges.
UPPER_PROBE_FIRST_SHOT = 42675

MID_Z_M = 0.0
UPPER_Z_M = 0.98

# Toroidal angle (phi) is deliberately NOT written to the ODS: issue #152
# flags that an absolute phi value requires "a documented VEST clock-position
# reference and IMAS sign convention" that has not been established. The
# clock positions are recorded here only as provenance for whoever resolves
# that convention later.
MID_CLOCK_POSITION = "11 o'clock"
UPPER_CLOCK_POSITION = "4 o'clock"

ION_MASS_KG = {
    "H": 1.67262192369e-27,
    "D": 3.3435837724e-27,
}

ASSEMBLIES: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "name": "Mid triple Langmuir probe",
        "diagnostic": "langmuir_probe_mid",
        "z": MID_Z_M,
        "position_key": "mid_r",
    },
    {
        "index": 1,
        "name": "Upper triple Langmuir probe",
        "diagnostic": "langmuir_probe_upper",
        "z": UPPER_Z_M,
        "position_key": "upper_r",
    },
)


def _safe_vest_load(
    shot: int,
    field: int,
    raw_source: raw_db.RawSource | None = None,
):
    return raw_db.vest_load(
        shot,
        field,
        sample_opt=False if raw_source is None else raw_source,
    )


def _probe_signal_present(
    shot: int,
    field: int,
    raw_source: raw_db.RawSource | None,
) -> bool:
    """Best-effort presence check that never raises.

    Triple Langmuir probes are not operated every shot, so a missing signal
    is the expected, common case here -- unlike :func:`raw_db.require_signal`,
    which is for data that is supposed to always be present.
    """
    try:
        loaded = _safe_vest_load(shot, field, raw_source)
    except Exception:  # pragma: no cover - defensive against backend errors
        return False
    if loaded is None:
        return False
    try:
        time_values, data_values = loaded
    except (TypeError, ValueError):
        return False
    return np.asarray(time_values).size >= 2 and np.asarray(data_values).size >= 2


def _build_target_time(source_time: np.ndarray, tstart: float, tend: float, dt: float) -> np.ndarray:
    if dt > 0 and source_time.size > 0:
        start = max(tstart, float(source_time[0]))
        end = min(tend, float(source_time[-1]))
        if end > start:
            return np.arange(start, end, dt)
    step = dt if dt > 0 else DEFAULT_DT
    return np.arange(tstart, tend, step)


def _require_era_fields(config: dict[str, Any], *, shot: int, diagnostic: str) -> None:
    missing = [key for key in ("vd3", "tip_length_mm", "tip_radius_mm") if config.get(key) is None]
    if missing:
        raise VestConfigurationError(
            f"langmuir_probes: no bias-voltage/tip-geometry configuration for shot {shot} "
            f"in diagnostic {diagnostic!r} (missing {', '.join(missing)}). This shot falls in "
            "a documented but unresolved era gap and must be verified, not assumed."
        )


def vfit_langmuir_probes_static(ods: object) -> None:
    set_path(
        ods,
        "langmuir_probes.ids_properties.comment",
        "VEST triple Langmuir probe (mid + upper assemblies)",
    )
    # Each assembly may be absent for a given shot (not operated, or -- for
    # the upper assembly -- not yet installed), and independently loads its
    # own voltage/current time coordinate, so there is no IDS-wide shared
    # time node to declare homogeneous. Mirrors barometry's per-gauge time.
    set_path(ods, "langmuir_probes.ids_properties.homogeneous_time", 0)


def vfit_langmuir_probes_dynamic(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
    mid_r: float | None = None,
    upper_r: float | None = None,
) -> None:
    positions = {"mid_r": mid_r, "upper_r": upper_r}

    for assembly in ASSEMBLIES:
        index = assembly["index"]
        if index == 1 and int(shot) < UPPER_PROBE_FIRST_SHOT:
            continue

        config = resolve_vest_diagnostic(shot, assembly["diagnostic"])
        voltage_field = int(config["source"]["voltage_field"])
        current_field = int(config["source"]["current_field"])

        if not _probe_signal_present(shot, voltage_field, raw_source):
            logger.info(
                "%s not operated for shot %s (field %s); skipping.",
                assembly["name"],
                shot,
                voltage_field,
            )
            continue

        _require_era_fields(config, shot=shot, diagnostic=assembly["diagnostic"])

        gas_species = config["ion"]["gas_species"]
        ion_mass_kg = ION_MASS_KG.get(gas_species)
        if ion_mass_kg is None:
            raise VestConfigurationError(
                f"langmuir_probes: unknown gas_species {gas_species!r} for {assembly['diagnostic']!r}"
            )

        source_time_v, source_v_raw = raw_db.require_signal(
            _safe_vest_load(shot, voltage_field, raw_source),
            shot=shot,
            field=voltage_field,
            signal_name=f"{assembly['name']} voltage",
        )
        source_time_i, source_i_raw = raw_db.require_signal(
            _safe_vest_load(shot, current_field, raw_source),
            shot=shot,
            field=current_field,
            signal_name=f"{assembly['name']} current",
        )

        gain = config["gain"]
        processing = config["processing"]
        result = process_triple_probe(
            source_time_v,
            source_v_raw,
            source_time_i,
            source_i_raw,
            float(config["vd3"]),
            tip_radius_m=float(config["tip_radius_mm"]) * 1e-3,
            tip_length_m=float(config["tip_length_mm"]) * 1e-3,
            ion_mass_kg=ion_mass_kg,
            voltage_gain=float(gain["voltage_gain"]),
            current_divisor=float(gain["current_divisor"]),
            n_baseline_samples=int(processing["baseline_samples"]),
            median_kernel=processing.get("median_kernel"),
        )

        time = (
            np.asarray(target_time, dtype=float)
            if target_time is not None
            else _build_target_time(result["time"], tstart, tend, dt)
        )
        n_e_data = np.interp(time, result["time"], result["n_e"])
        te_data = np.interp(time, result["time"], result["te"])
        validity_fraction = np.interp(time, result["time"], result["solver_ok"].astype(float))
        # IMAS validity convention: 0 = valid. Any sample whose interpolation
        # touches a failed/nonphysical solve is marked invalid (-1) rather
        # than silently presented as a plausible value.
        validity = np.where(validity_fraction >= 1.0, 0, -1).astype(int)

        surface_area = probe_surface_area(
            tip_radius_m=float(config["tip_radius_mm"]) * 1e-3,
            tip_length_m=float(config["tip_length_mm"]) * 1e-3,
        )

        prefix = f"langmuir_probes.embedded.{index}"
        set_path(ods, f"{prefix}.identifier", assembly["diagnostic"])
        set_path(ods, f"{prefix}.name", assembly["name"])
        set_path(ods, f"{prefix}.position.z", assembly["z"])
        position_r = positions.get(assembly["position_key"])
        if position_r is not None:
            set_path(ods, f"{prefix}.position.r", float(position_r))
        set_path(ods, f"{prefix}.surface_area", surface_area)
        set_path(ods, f"{prefix}.time", time)
        set_path(ods, f"{prefix}.n_e.data", n_e_data)
        set_path(ods, f"{prefix}.n_e.validity_timed.time", time)
        set_path(ods, f"{prefix}.n_e.validity_timed.data", validity)
        set_path(ods, f"{prefix}.t_e.data", te_data)
        set_path(ods, f"{prefix}.t_e.validity_timed.time", time)
        set_path(ods, f"{prefix}.t_e.validity_timed.data", validity)


def langmuir_probes(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
    mid_r: float | None = None,
    upper_r: float | None = None,
) -> None:
    vfit_langmuir_probes_static(ods)
    vfit_langmuir_probes_dynamic(
        ods,
        shot,
        tstart,
        tend,
        dt,
        raw_source=raw_source,
        target_time=target_time,
        mid_r=mid_r,
        upper_r=upper_r,
    )


def langmuir_probes_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    options = options or {}
    langmuir_probes(
        ods,
        shot,
        tstart,
        tend,
        dt,
        raw_source=options.get("raw_source"),
        mid_r=options.get("mid_r"),
        upper_r=options.get("upper_r"),
    )


_POSITION_CSV_COLUMNS = ("mid TP position[m]", "upper TP position[m]")


def _read_measured_position_row(csv_path: str | Path, shot: int) -> tuple[float | None, float | None] | None:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_shot = int(row["shot"])
            except (KeyError, ValueError):
                continue
            if row_shot != int(shot):
                continue

            def _parse(column: str) -> float | None:
                raw_value = (row.get(column) or "").strip()
                return float(raw_value) if raw_value else None

            return _parse(_POSITION_CSV_COLUMNS[0]), _parse(_POSITION_CSV_COLUMNS[1])
    return None


def apply_langmuir_probe_measured_positions(
    ods: object,
    shot: int,
    *,
    csv_path: str | Path | None = None,
) -> None:
    """Update ``embedded.{0,1}.position.r`` from the measured-position CSV.

    Non-blocking by design: a missing/unreadable CSV or a shot absent from it
    only logs at INFO and returns -- it must never prevent the raw-signal
    path (n_e/t_e) from being processed and stored. Only ``position.r`` is
    touched; ``n_e``/``t_e``/``time`` are never re-derived here.
    """
    if csv_path is None:
        logger.info(
            "No measured-position CSV supplied for shot %s; leaving position.r unset.", shot
        )
        return

    path = Path(csv_path)
    if not path.exists():
        logger.info("Measured-position CSV %s not found for shot %s; leaving position.r unset.", path, shot)
        return

    try:
        row = _read_measured_position_row(path, shot)
    except Exception:
        logger.exception("Failed to read langmuir probe measured-position CSV %s", path)
        return

    if row is None:
        logger.info("No measured position recorded for shot %s in %s", shot, path)
        return

    mid_r, upper_r = row
    if mid_r is not None and path_exists(ods, "langmuir_probes.embedded.0.time"):
        set_path(ods, "langmuir_probes.embedded.0.position.r", float(mid_r))
    if (
        upper_r is not None
        and int(shot) >= UPPER_PROBE_FIRST_SHOT
        and path_exists(ods, "langmuir_probes.embedded.1.time")
    ):
        set_path(ods, "langmuir_probes.embedded.1.position.r", float(upper_r))


__all__ = [
    "UPPER_PROBE_FIRST_SHOT",
    "apply_langmuir_probe_measured_positions",
    "langmuir_probes",
    "langmuir_probes_from_raw_database",
    "vfit_langmuir_probes_dynamic",
    "vfit_langmuir_probes_static",
]
