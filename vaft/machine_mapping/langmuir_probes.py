"""Canonical VEST triple Langmuir probe builders integrated under machine_mapping.

Raw-signal acquisition, shot-era bias voltage/tip geometry, and IDS
population live here; the backend-independent physics (offset removal,
calibration, Te solve, n_e calculation) live in
:mod:`vaft.process.langmuir`. Machine constants live in
``vaft/machine_mapping/vest.yaml`` under ``langmuir_probes``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vaft.database import raw as raw_db
from vaft.process.langmuir import probe_surface_area, process_triple_probe

from .utils import (
    _deep_merge,
    _normalize_shot_key,
    _resolve_info_file_path,
    get_path,
    load_yaml,
    path_exists,
    resolve_data_root,
    set_path,
)

logger = logging.getLogger(__name__)

DEFAULT_DT = 4e-5

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
    {"key": "mid", "name": "Mid triple Langmuir probe", "z": MID_Z_M, "position_key": "mid_r"},
    {"key": "upper", "name": "Upper triple Langmuir probe", "z": UPPER_Z_M, "position_key": "upper_r"},
)


class LangmuirProbeConfigError(ValueError):
    """Raised when the VEST langmuir_probes configuration cannot resolve a shot."""


def _vest_config(info_file: str | None = None) -> Mapping[str, Any]:
    # Deliberately uncached: the mapping file is small, this is not a hot
    # path, and a cache here would serve stale settings to anything that
    # rewrites it (mirrors vaft.machine_mapping.impa._vest_config).
    return load_yaml(_resolve_info_file_path(info_file))


def resolve_langmuir_probe_config(assembly_key: str, shot: int, info_file: str | None = None) -> dict[str, Any]:
    """Return the ``langmuir_probes.<assembly_key>`` block, with shot overrides merged."""
    content = _vest_config(info_file)
    default_block = content.get("0") or content.get(0) or {}
    shot_block = content.get(_normalize_shot_key(shot), {}) or {}
    merged = _deep_merge(default_block, shot_block)
    config = (merged.get("langmuir_probes") or {}).get(assembly_key)
    if not isinstance(config, Mapping):
        raise LangmuirProbeConfigError(
            f"No langmuir_probes.{assembly_key} configuration in the VEST machine mapping"
        )
    return dict(config)


def _resolve_era(config: Mapping[str, Any], shot: int, *, assembly_key: str) -> dict[str, Any]:
    """Return the bias-voltage/tip-geometry era for ``shot``.

    Unlike other VEST shot-era lookups (e.g. IMPA's gain override), there is
    no unconditional default here: a shot outside every declared era has no
    known bias voltage or tip geometry, and that must fail clearly rather
    than silently reuse a neighboring era or an arbitrary default.
    """
    numeric_shot = int(shot)
    matches = []
    for era in config.get("shot_era_overrides") or ():
        min_shot = era.get("min_shot")
        max_shot = era.get("max_shot")
        if min_shot is not None and numeric_shot < int(min_shot):
            continue
        if max_shot is not None and numeric_shot > int(max_shot):
            continue
        matches.append(era)

    if not matches:
        raise LangmuirProbeConfigError(
            f"No bias-voltage/tip-geometry configuration for shot {numeric_shot} in "
            f"langmuir_probes.{assembly_key}. This shot falls in a documented but "
            "unresolved era gap and must be verified, not assumed."
        )
    if len(matches) > 1:
        raise LangmuirProbeConfigError(
            f"Overlapping langmuir_probes.{assembly_key} shot_era_overrides apply to "
            f"shot {numeric_shot}: {matches}"
        )

    era = matches[0]
    for key in ("vd3", "tip_length_mm", "tip_radius_mm"):
        if key not in era:
            raise LangmuirProbeConfigError(
                f"langmuir_probes.{assembly_key} era for shot {numeric_shot} is missing {key!r}"
            )
    return dict(era)


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

    # embedded[] is an IMAS array of structures and must be filled
    # contiguously: a skipped assembly (not installed / not operated for this
    # shot) must not leave a gap, so the slot comes from a running counter
    # rather than the assembly's nominal position.
    next_index = 0
    for assembly in ASSEMBLIES:
        config = resolve_langmuir_probe_config(assembly["key"], shot)

        first_shot = config.get("first_shot")
        if first_shot is not None and int(shot) < int(first_shot):
            continue

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

        era = _resolve_era(config, shot, assembly_key=assembly["key"])

        gas_species = config["ion"]["gas_species"]
        ion_mass_kg = ION_MASS_KG.get(gas_species)
        if ion_mass_kg is None:
            raise LangmuirProbeConfigError(
                f"langmuir_probes.{assembly['key']}: unknown gas_species {gas_species!r}"
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

        calibration = config["calibration"]
        processing = config["processing"]
        result = process_triple_probe(
            source_time_v,
            source_v_raw,
            source_time_i,
            source_i_raw,
            float(era["vd3"]),
            tip_radius_m=float(era["tip_radius_mm"]) * 1e-3,
            tip_length_m=float(era["tip_length_mm"]) * 1e-3,
            ion_mass_kg=ion_mass_kg,
            voltage_gain=float(calibration["voltage_gain"]),
            current_divisor=float(calibration["current_divisor"]),
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
            tip_radius_m=float(era["tip_radius_mm"]) * 1e-3,
            tip_length_m=float(era["tip_length_mm"]) * 1e-3,
        )

        index = next_index
        next_index += 1
        prefix = f"langmuir_probes.embedded.{index}"
        set_path(ods, f"{prefix}.identifier", f"langmuir_probes:{assembly['key']}")
        set_path(ods, f"{prefix}.name", assembly["name"])
        set_path(ods, f"{prefix}.position.z", assembly["z"])
        position_r = positions.get(assembly["position_key"])
        if position_r is not None:
            set_path(ods, f"{prefix}.position.r", float(position_r))
        set_path(ods, f"{prefix}.surface_area", surface_area)
        set_path(ods, f"{prefix}.time", time)
        set_path(ods, f"{prefix}.n_e.data", n_e_data)
        set_path(ods, f"{prefix}.n_e.validity_timed", validity)
        set_path(ods, f"{prefix}.t_e.data", te_data)
        set_path(ods, f"{prefix}.t_e.validity_timed", validity)


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
    position_csv_path: str | Path | None = None,
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
    apply_langmuir_probe_measured_positions(ods, shot, csv_path=position_csv_path)


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
        position_csv_path=options.get("position_csv_path"),
    )


_POSITION_CSV_COLUMNS = ("mid TP position[m]", "upper TP position[m]")

DEFAULT_POSITION_CSV = resolve_data_root() / "legacy" / "langmuir_probe_positions.csv"


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


def _embedded_index_of(ods: object, assembly_key: str) -> int | None:
    """Return the ``langmuir_probes.embedded`` slot holding ``assembly_key``.

    The array is filled contiguously over the assemblies operated for a shot,
    so the slot number is not the assembly's nominal position; the identifier
    written alongside the data is what names the probe.
    """
    identifier = f"langmuir_probes:{assembly_key}"
    for index in range(len(ASSEMBLIES)):
        base = f"langmuir_probes.embedded.{index}"
        if not path_exists(ods, f"{base}.time"):
            continue
        if not path_exists(ods, f"{base}.identifier"):
            continue
        try:
            stored = str(get_path(ods, f"{base}.identifier"))
        except Exception:  # pragma: no cover - defensive against backend errors
            continue
        if stored == identifier:
            return index
    return None


def apply_langmuir_probe_measured_positions(
    ods: object,
    shot: int,
    *,
    csv_path: str | Path | None = None,
) -> None:
    """Update ``embedded.{0,1}.position.r`` from the measured-position CSV.

    Defaults to the bundled ``vaft/data/legacy/langmuir_probe_positions.csv``
    table (per-shot mid/upper probe radial positions from the VEST shot log)
    when ``csv_path`` is not given. Non-blocking by design: a missing/
    unreadable CSV or a shot absent from it only logs at INFO and returns --
    it must never prevent the raw-signal path (n_e/t_e) from being processed
    and stored. Only ``position.r`` is touched; ``n_e``/``t_e``/``time`` are
    never re-derived here.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_POSITION_CSV
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
    # embedded[] is filled contiguously over the assemblies actually operated,
    # so a slot number does not identify a probe: when the mid probe is absent
    # the upper probe occupies embedded[0]. Resolve each assembly by the
    # identifier the mapper wrote, or the measured radii get swapped onto the
    # wrong probe.
    for assembly_key, radius in (("mid", mid_r), ("upper", upper_r)):
        if radius is None:
            continue
        index = _embedded_index_of(ods, assembly_key)
        if index is None:
            continue
        set_path(ods, f"langmuir_probes.embedded.{index}.position.r", float(radius))


__all__ = [
    "LangmuirProbeConfigError",
    "apply_langmuir_probe_measured_positions",
    "langmuir_probes",
    "langmuir_probes_from_raw_database",
    "resolve_langmuir_probe_config",
    "vfit_langmuir_probes_dynamic",
    "vfit_langmuir_probes_static",
]
