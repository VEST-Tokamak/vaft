"""VEST soft X-ray mapping for the OMAS ``soft_x_rays`` IDS.

The IMAS/OMAS soft_x_rays IDS stores detector signals as channel brightness
arrays plus static channel geometry such as line-of-sight endpoints.  VEST SXR
raw digitizer CSV files are voltage traces, so this mapper stores them as a
calibrated brightness proxy unless a physical ``brightness_scale`` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import yaml

from .utils import path_exists, resolve_data_root, set_path

# ``sample_v3.py`` records a waveform relative to its digitizer trigger and
# then decimates it.  The two known digitizers use different decimation.
DEFAULT_SAMPLE_RATES: dict[str, float] = {
    "17592": 125e6 / 32.0,
    "22577": 125e6 / 128.0,
}
# Kept as a compatibility alias for callers that explicitly build a 22577 axis.
DEFAULT_SAMPLE_RATE = DEFAULT_SAMPLE_RATES["22577"]
DEFAULT_TIME_OFFSET = 0.0
DEFAULT_ENERGY_BAND = (0.0, 20_000.0)
PACKAGED_GEOMETRY_TABLE = "geometry/line_of_sight_endpoints.csv"
PACKAGED_TRIGGER_SETTINGS = "legacy/diagnostic-trigger-settings.yaml"
# The 455xx SXR CSV campaign has no SXR entry in the legacy settings table.
# It is allowed to inherit HXR timing because paired SXR/HXR records in the
# later campaign have the same 285 ms trigger start.
HXR_FALLBACK_SHOT_RANGE = range(45531, 45541)


@dataclass(frozen=True)
class SXRArraySpec:
    name: str
    folder: str
    file_template: str
    channels: int
    label: str


@dataclass(frozen=True)
class SXRDigitizerBlock:
    """One contiguous digitizer block and its physical channel semantics."""

    array: str
    start: int
    count: int
    filter_material: str | None = None
    filter_thickness_m: float | None = None
    reverse_spatial_order: bool = False


@dataclass(frozen=True)
class SXRTimeAlignment:
    """Resolved relation between a CSV trigger axis and the shot clock."""

    offset_seconds: float
    source: str
    detail: str


DEFAULT_ARRAY_SPECS: dict[str, SXRArraySpec] = {
    "horizontal": SXRArraySpec(
        name="horizontal",
        folder="horizontal",
        file_template="horizontalLOS_ch_{channel}_.csv",
        channels=20,
        label="Horizontal SXR",
    ),
    "vertical": SXRArraySpec(
        name="vertical",
        folder="vertical",
        file_template="verticalLOS_ch_{channel}_.csv",
        channels=20,
        label="Vertical SXR",
    ),
    "lowermid": SXRArraySpec(
        name="lowermid",
        folder="lowermid",
        file_template="twofilter_horizontalLOS_ch_{channel}_.csv",
        channels=16,
        label="Lower-mid two-filter SXR",
    ),
    "bottom": SXRArraySpec(
        name="bottom",
        folder="bottom",
        file_template="twofilter_verticalLOS_ch_{channel}_.csv",
        channels=16,
        label="Bottom two-filter SXR",
    ),
}

# Canonical wiring from the VEST SXR profiles.  Source columns retain digitizer
# order; ``array_channel`` is the physical order used by the LOS table.
DEFAULT_DIGITIZER_ARRAYS: dict[str, tuple[SXRDigitizerBlock, ...]] = {
    "17592": (
        SXRDigitizerBlock("vertical", 0, 20),
        SXRDigitizerBlock("horizontal", 20, 20),
    ),
    "22577": (
        SXRDigitizerBlock("bottom", 0, 16, "Be", 0.2e-6),
        SXRDigitizerBlock("bottom", 16, 16, "Al", 0.2e-6),
        SXRDigitizerBlock("lowermid", 32, 16, "Be", 0.2e-6, True),
        SXRDigitizerBlock("lowermid", 48, 16, "Al", 0.2e-6, True),
    ),
}


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _resolve_digitizer_file(
    shot: int,
    daq_label: str | int,
    data_root: str | Path | None = None,
    digitizer_file: str | Path | None = None,
) -> Path:
    if digitizer_file is not None:
        path = Path(digitizer_file).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Digitizer CSV file not found: {path}")

    root = resolve_data_root(data_root)
    label = str(daq_label)
    candidates = [
        root / "legacy" / f"digitizer_{label}_{int(shot)}.csv",
        root / f"digitizer_{label}_{int(shot)}.csv",
        root / "soft_x_rays" / f"digitizer_{label}_{int(shot)}.csv",
        root / "raw" / f"digitizer_{label}_{int(shot)}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot find digitizer_{label}_{int(shot)}.csv. Packaged digitizer "
        "samples are not included in the PyPI distribution; provide "
        "digitizer_file/data_root or clone the VAFT GitHub repository. "
        f"Searched: {searched}"
    )


def _digitizer_file_candidates(
    shot: int,
    daq_label: str | int,
    data_root: str | Path | None,
) -> list[Path]:
    """Return candidate locations for one archived SXR digitizer file."""
    root = resolve_data_root(data_root)
    filename = f"digitizer_{str(daq_label)}_{int(shot)}.csv"
    return [
        root / "legacy" / filename,
        root / filename,
        root / "soft_x_rays" / filename,
        root / "raw" / filename,
    ]


def _discover_digitizer_files(shot: int, data_root: str | Path | None = None) -> list[tuple[str, Path]]:
    """Find supported SXR digitizers for a shot in deterministic DAQ order."""
    labels = sorted(
        set(DEFAULT_SAMPLE_RATES) | set(DEFAULT_DIGITIZER_ARRAYS),
        key=lambda value: (int(value), value),
    )
    found: list[tuple[str, Path]] = []
    for label in labels:
        for candidate in _digitizer_file_candidates(shot, label, data_root):
            if candidate.exists():
                found.append((label, candidate))
                break
    return found


def _daq_label_from_filename(path: str | Path, shot: int) -> str | None:
    """Infer a DAQ label from the standard digitizer filename, if present."""
    match = re.fullmatch(rf"digitizer_(.+)_{int(shot)}\.csv", Path(path).name)
    return match.group(1) if match else None


def _packaged_geometry_table_path() -> Path:
    from vaft.data.resources import data_path

    return data_path(PACKAGED_GEOMETRY_TABLE)


def _packaged_trigger_settings_path() -> Path:
    from vaft.data.resources import data_path

    return data_path(PACKAGED_TRIGGER_SETTINGS)


def resolve_sxr_trigger_settings(
    trigger_settings_path: str | Path | None = None,
) -> Path | None:
    """Resolve the packaged or caller-provided diagnostic trigger settings."""
    if trigger_settings_path is not None:
        path = Path(trigger_settings_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SXR trigger settings file not found: {path}")
        return path
    packaged = _packaged_trigger_settings_path()
    return packaged if packaged.exists() else None


@lru_cache(maxsize=4)
def _load_trigger_settings(path: str) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    shots = data.get("shots", {})
    if not isinstance(shots, Mapping):
        raise ValueError("Diagnostic trigger settings must contain a 'shots' mapping.")
    return shots


def resolve_sxr_time_alignment(
    shot: int,
    *,
    trigger_settings_path: str | Path | None = None,
    allow_hxr_fallback: bool = True,
) -> SXRTimeAlignment:
    """Resolve a machine-time offset for a trigger-relative SXR CSV.

    Direct ``SXR`` settings are authoritative.  The 45531--45540 campaign has
    no direct SXR record, but its HXR trigger is an approved 285 ms fallback
    based on the paired SXR/HXR campaign records and legacy SXR notebook.
    Missing settings intentionally leave data trigger-relative and warn rather
    than inventing a machine-time offset.
    """
    path = resolve_sxr_trigger_settings(trigger_settings_path)
    if path is None:
        detail = "No diagnostic trigger-settings file is available; using trigger-relative time."
        warnings.warn(detail, RuntimeWarning, stacklevel=2)
        return SXRTimeAlignment(0.0, "trigger_relative", detail)

    settings = _load_trigger_settings(str(path))
    entry = settings.get(int(shot), settings.get(str(int(shot))))
    if not isinstance(entry, Mapping):
        detail = f"Shot {shot} is absent from trigger settings; using trigger-relative time."
        warnings.warn(detail, RuntimeWarning, stacklevel=2)
        return SXRTimeAlignment(0.0, "trigger_relative", detail)

    sxr = entry.get("SXR")
    if isinstance(sxr, Mapping) and sxr.get("start_time_ms") is not None:
        start_ms = float(sxr["start_time_ms"])
        return SXRTimeAlignment(start_ms * 1e-3, "sxr_trigger", f"SXR start_time_ms={start_ms:g}")

    hxr = entry.get("HXR")
    if (
        allow_hxr_fallback
        and int(shot) in HXR_FALLBACK_SHOT_RANGE
        and isinstance(hxr, Mapping)
        and hxr.get("start_time_ms") is not None
    ):
        start_ms = float(hxr["start_time_ms"])
        detail = (
            f"Inferred SXR start from HXR start_time_ms={start_ms:g} for the "
            "455xx co-trigger campaign."
        )
        return SXRTimeAlignment(start_ms * 1e-3, "hxr_fallback", detail)

    detail = f"Shot {shot} has no usable SXR trigger setting; using trigger-relative time."
    warnings.warn(detail, RuntimeWarning, stacklevel=2)
    return SXRTimeAlignment(0.0, "trigger_relative", detail)


def _sample_rate_for_daq(daq_label: str | int, sample_rate: float | None) -> float:
    if sample_rate is not None:
        return float(sample_rate)
    label = str(daq_label)
    try:
        return DEFAULT_SAMPLE_RATES[label]
    except KeyError as exc:
        raise ValueError(
            f"No archived sample_v3 sampling profile is known for digitizer {label}; "
            "pass sample_rate explicitly."
        ) from exc


def resolve_sxr_geometry_table(geometry_root: str | Path | None = None) -> Path | None:
    """Resolve a single CSV table with SXR LOS endpoint geometry."""
    explicit = _as_path(geometry_root)
    if explicit is not None:
        if explicit.is_file():
            return explicit
        if explicit.is_dir():
            table = explicit / "line_of_sight_endpoints.csv"
            return table if table.exists() else None
        raise FileNotFoundError(f"SXR geometry table/root not found: {explicit}")

    env_table = os.environ.get("VEST_SXR_GEOMETRY_TABLE")
    if env_table:
        candidate = Path(env_table).expanduser()
        if candidate.exists():
            return candidate

    packaged = _packaged_geometry_table_path()
    if packaged.exists():
        return packaged

    env_root = os.environ.get("VEST_SXR_GEOMETRY_DIR")
    if env_root:
        candidate = Path(env_root).expanduser() / "line_of_sight_endpoints.csv"
        if candidate.exists():
            return candidate

    candidates = [
        resolve_data_root() / "geometry" / "line_of_sight_endpoints.csv",
        resolve_data_root() / "geometry" / "soft_x_rays" / "line_of_sight_endpoints.csv",
        resolve_data_root() / "soft_x_rays" / "geometry" / "line_of_sight_endpoints.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_sxr_geometry_root(geometry_root: str | Path | None = None) -> Path | None:
    """Resolve a legacy directory containing full SXR LOS grid CSV files."""
    explicit = _as_path(geometry_root)
    if explicit is not None:
        if explicit.is_dir() and (explicit / "rgrid.csv").exists() and (explicit / "zgrid.csv").exists():
            return explicit
        if explicit.exists():
            return None
        raise FileNotFoundError(f"SXR geometry root not found: {explicit}")

    env_path = os.environ.get("VEST_SXR_GEOMETRY_DIR")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists() and (candidate / "rgrid.csv").exists() and (candidate / "zgrid.csv").exists():
            return candidate

    candidates = [
        resolve_data_root() / "geometry" / "soft_x_rays",
        resolve_data_root() / "soft_x_rays" / "geometry",
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "rgrid.csv").exists() and (candidate / "zgrid.csv").exists():
            return candidate
    return None


@lru_cache(maxsize=8)
def load_sxr_geometry_table(geometry_table: str | Path | None = None) -> dict[tuple[str, int], dict[str, Any]]:
    """Load SXR LOS endpoint geometry keyed by ``(array, channel)``."""
    table_path = resolve_sxr_geometry_table(geometry_table)
    if table_path is None:
        return {}
    frame = pd.read_csv(table_path)
    geometry: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["array"]), int(row["channel"]))
        geometry[key] = {
            "daq_label": str(row.get("daq_label", "")),
            "first_r": float(row["first_r"]),
            "first_z": float(row["first_z"]),
            "second_r": float(row["second_r"]),
            "second_z": float(row["second_z"]),
            "phi": float(row.get("phi", 0.0)),
        }
    return geometry


def load_digitizer_csv(
    filepath: str | Path,
    *,
    channels_as_rows: bool = True,
    nrows: int | None = None,
) -> np.ndarray:
    """Load a VEST SXR digitizer CSV as a time x channel matrix."""
    frame = pd.read_csv(filepath, delimiter=",", header=None, nrows=nrows)
    values = frame.to_numpy(dtype=float)
    if channels_as_rows:
        values = values.T
    return values


def build_time_axis(
    samples: int,
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    time_offset: float = DEFAULT_TIME_OFFSET,
) -> np.ndarray:
    """Return seconds for digitizer samples from the trigger-relative archive axis."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return np.arange(samples, dtype=float) / float(sample_rate) + float(time_offset)


def default_channel_map(daq_label: str | int, n_channels: int) -> list[dict[str, Any]]:
    """Return default VEST SXR digitizer-to-array channel metadata."""
    label = str(daq_label)
    mapping: list[dict[str, Any]] = []
    used: set[int] = set()

    for block in DEFAULT_DIGITIZER_ARRAYS.get(label, ()):  # known DAQs
        spec = DEFAULT_ARRAY_SPECS[block.array]
        for offset in range(block.count):
            source_column = block.start + offset
            if source_column >= n_channels:
                continue
            array_channel = block.count - offset if block.reverse_spatial_order else offset + 1
            filter_label = f" {block.filter_material}" if block.filter_material else ""
            entry: dict[str, Any] = {
                "source_column": source_column,
                "daq_label": label,
                "array": block.array,
                "array_channel": array_channel,
                "name": f"{spec.label}{filter_label} Ch {array_channel}",
                "identifier": f"{label}:{block.array}:{block.filter_material or 'none'}:{array_channel}",
            }
            if block.filter_material is not None:
                entry["filter_material"] = block.filter_material
                entry["filter_thickness_m"] = block.filter_thickness_m
            mapping.append(entry)
            used.add(source_column)

    for source_column in range(n_channels):
        if source_column in used:
            continue
        channel_number = source_column + 1
        mapping.append(
            {
                "source_column": source_column,
                "daq_label": label,
                "array": None,
                "array_channel": None,
                "name": f"Digitizer {label} Ch {channel_number}",
                "identifier": f"{label}:digitizer:{channel_number}",
            }
        )

    return sorted(mapping, key=lambda item: int(item["source_column"]))


def _load_rz_grids(geometry_root: Path) -> tuple[np.ndarray, np.ndarray]:
    rgrid = np.genfromtxt(geometry_root / "rgrid.csv", delimiter=",")
    zgrid = np.genfromtxt(geometry_root / "zgrid.csv", delimiter=",")
    rgrid = np.asarray(rgrid, dtype=float).reshape(-1)
    zgrid = np.asarray(zgrid, dtype=float).reshape(-1)
    if np.nanmax(np.abs(rgrid)) > 10.0:
        rgrid = rgrid / 1000.0
    if np.nanmax(np.abs(zgrid)) > 10.0:
        zgrid = zgrid / 1000.0
    return rgrid, zgrid


def _los_file_for_channel(geometry_root: Path, array_name: str, array_channel: int) -> Path:
    spec = DEFAULT_ARRAY_SPECS[array_name]
    return geometry_root / spec.folder / spec.file_template.format(channel=int(array_channel))


def _line_endpoints_from_los(
    los_matrix: np.ndarray,
    rgrid: np.ndarray,
    zgrid: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    los = np.asarray(los_matrix, dtype=float)
    if los.shape != (zgrid.size, rgrid.size):
        raise ValueError(
            f"LOS shape {los.shape} is inconsistent with z/r grids {(zgrid.size, rgrid.size)}"
        )

    weights = np.abs(los)
    rows, cols = np.nonzero(weights > 0)
    if rows.size < 2:
        raise ValueError("LOS matrix does not contain enough non-zero points.")

    coords = np.column_stack((rgrid[cols], zgrid[rows]))
    selected_weights = weights[rows, cols]
    center = np.average(coords, axis=0, weights=selected_weights)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = centered @ axis
    first = coords[int(np.argmin(projection))]
    second = coords[int(np.argmax(projection))]
    return (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))


def _geometry_for_channel(
    geometry_table: Mapping[tuple[str, int], Mapping[str, Any]],
    geometry_root: Path | None,
    channel_info: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], float | None] | None:
    array_name = channel_info.get("array")
    array_channel = channel_info.get("array_channel")
    if not array_name or not array_channel:
        return None
    if array_name not in DEFAULT_ARRAY_SPECS:
        return None

    key = (str(array_name), int(array_channel))
    table_entry = geometry_table.get(key)
    if table_entry is not None:
        channel_daq_label = channel_info.get("daq_label")
        table_daq_label = table_entry.get("daq_label")
        if channel_daq_label is not None and table_daq_label and str(channel_daq_label) != str(table_daq_label):
            return None
        first = (float(table_entry["first_r"]), float(table_entry["first_z"]))
        second = (float(table_entry["second_r"]), float(table_entry["second_z"]))
        return first, second, float(table_entry.get("phi", 0.0))

    if geometry_root is None:
        return None
    filepath = _los_file_for_channel(geometry_root, str(array_name), int(array_channel))
    if not filepath.exists():
        return None
    rgrid, zgrid = _load_rz_grids(geometry_root)
    # The legacy LOS CSVs are vertically flipped before VFIT mapping; use the same
    # convention so Z coordinates follow the loaded zgrid.
    los = np.flipud(np.genfromtxt(filepath, delimiter=","))
    first, second = _line_endpoints_from_los(los, rgrid, zgrid)
    return first, second, None


def _normalise_channel_map(channel_map: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised = []
    for item in channel_map:
        entry = dict(item)
        if "source_column" not in entry:
            raise ValueError("Each channel_map entry must define source_column.")
        entry["source_column"] = int(entry["source_column"])
        normalised.append(entry)
    return sorted(normalised, key=lambda item: item["source_column"])


def _set_filter_metadata(ods: Any, prefix: str, item: Mapping[str, Any]) -> None:
    """Write only filter facts that are known for the mapped channel."""
    material = item.get("filter_material")
    thickness = item.get("filter_thickness_m")
    if material is None or thickness is None:
        return

    material_name = str(material)
    material_key = material_name.lower()
    if material_key == "be":
        material_index = 10  # standard IMAS materials_identifier: beryllium
        description = "Beryllium filter."
    elif material_key == "al":
        # Aluminium is not available in the IMAS 3.41 standard material list.
        material_index = -1
        description = "Aluminum filter (private VEST material identifier)."
    else:
        raise ValueError(f"Unsupported SXR filter material {material_name!r}.")

    filter_prefix = f"{prefix}.filter_window.0"
    set_path(ods, f"{filter_prefix}.material.index", material_index)
    set_path(ods, f"{filter_prefix}.material.name", material_name)
    set_path(ods, f"{filter_prefix}.material.description", description)
    set_path(ods, f"{filter_prefix}.thickness", float(thickness))


def vfit_soft_x_rays_static(
    ods: Any,
    *,
    channel_map: Sequence[Mapping[str, Any]],
    channel_offset: int = 0,
    homogeneous_time: bool = True,
    geometry_root: str | Path | None = None,
    energy_band: tuple[float, float] = DEFAULT_ENERGY_BAND,
    phi: float = 0.0,
    brightness_scale: float = 1.0,
) -> None:
    """Fill static ``soft_x_rays`` IDS metadata and LOS geometry."""
    del brightness_scale  # documented in dynamic data; kept here for call symmetry
    geometry_table_path = resolve_sxr_geometry_table(geometry_root)
    geometry_table = load_sxr_geometry_table(geometry_table_path) if geometry_table_path else {}
    resolved_geometry = None if geometry_table else resolve_sxr_geometry_root(geometry_root)
    set_path(ods, "soft_x_rays.ids_properties.homogeneous_time", int(homogeneous_time))
    set_path(ods, "soft_x_rays.ids_properties.name", "VEST soft X-ray arrays")
    set_path(
        ods,
        "soft_x_rays.ids_properties.comment",
        "VEST SXR digitizer data; relative calibrated signal proxy, not absolute brightness.",
    )
    set_path(ods, "soft_x_rays.ids_properties.creation_date", datetime.now(timezone.utc).isoformat())

    lower, upper = energy_band
    for idx, item in enumerate(_normalise_channel_map(channel_map), start=channel_offset):
        name = str(item.get("name") or f"SXR Ch {idx + 1}")
        identifier = str(item.get("identifier") or name)
        prefix = f"soft_x_rays.channel.{idx}"
        set_path(ods, f"{prefix}.name", name)
        set_path(ods, f"{prefix}.identifier", identifier)
        set_path(ods, f"{prefix}.energy_band.0.lower_bound", float(lower))
        set_path(ods, f"{prefix}.energy_band.0.upper_bound", float(upper))
        if item.get("etendue") is not None:
            set_path(ods, f"{prefix}.etendue", float(item["etendue"]))
        _set_filter_metadata(ods, prefix, item)

        endpoints = _geometry_for_channel(geometry_table, resolved_geometry, item)
        if endpoints is None:
            continue
        first, second, geometry_phi = endpoints
        point_phi = float(phi if geometry_phi is None else geometry_phi)
        for point_name, point in (("first_point", first), ("second_point", second)):
            set_path(ods, f"{prefix}.line_of_sight.{point_name}.r", point[0])
            set_path(ods, f"{prefix}.line_of_sight.{point_name}.z", point[1])
            set_path(ods, f"{prefix}.line_of_sight.{point_name}.phi", point_phi)


def vfit_soft_x_rays_dynamic(
    ods: Any,
    *,
    data: np.ndarray,
    time: np.ndarray,
    channel_map: Sequence[Mapping[str, Any]],
    channel_offset: int = 0,
    set_global_time: bool = True,
    brightness_scale: float = 1.0,
    baseline_range: tuple[int | None, int | None] | None = None,
    polarity: float = 1.0,
) -> None:
    """Fill dynamic channel brightness traces from a time x channel matrix."""
    values = np.asarray(data, dtype=float)
    time_values = np.asarray(time, dtype=float).reshape(-1)
    if values.ndim != 2:
        raise ValueError("data must be a 2D time x channel matrix.")
    if values.shape[0] != time_values.size:
        raise ValueError("data time dimension must match time length.")

    if set_global_time:
        set_path(ods, "soft_x_rays.time", time_values)
    for idx, item in enumerate(_normalise_channel_map(channel_map), start=channel_offset):
        source_column = int(item["source_column"])
        if source_column < 0 or source_column >= values.shape[1]:
            raise IndexError(f"source_column {source_column} is outside data shape {values.shape}")
        signal = values[:, source_column].astype(float, copy=True)
        if baseline_range is not None:
            start, stop = baseline_range
            baseline_values = signal[slice(start, stop)]
            if baseline_values.size == 0:
                raise ValueError("baseline_range selects no SXR samples.")
            baseline = np.mean(baseline_values)
            signal = signal - baseline
        signal = float(polarity) * float(brightness_scale) * signal
        brightness = signal.reshape(1, -1)
        prefix = f"soft_x_rays.channel.{idx}"
        set_path(ods, f"{prefix}.brightness.time", time_values)
        set_path(ods, f"{prefix}.brightness.data", brightness)
        set_path(ods, f"{prefix}.validity_timed.time", time_values)
        set_path(ods, f"{prefix}.validity_timed.data", np.zeros(time_values.size, dtype=int))


def _existing_sxr_channel_identifiers(ods: Any) -> set[str]:
    """Return channel identifiers already present in an ODS or dict."""
    try:
        channels = (
            ods["soft_x_rays.channel"]
            if not isinstance(ods, dict)
            else ods["soft_x_rays"]["channel"]
        )
        channel_count = len(channels)
    except (KeyError, IndexError, TypeError, ValueError):
        return set()

    identifiers: set[str] = set()
    for index in range(channel_count):
        path = f"soft_x_rays.channel.{index}.identifier"
        if path_exists(ods, path):
            value = (
                ods[path]
                if not isinstance(ods, dict)
                else ods["soft_x_rays"]["channel"][index]["identifier"]
            )
            identifiers.add(str(value))
    return identifiers


def _existing_sxr_channel_count(ods: Any) -> int:
    """Return the current channel-array length without creating ODS branches."""
    try:
        channels = (
            ods["soft_x_rays.channel"]
            if not isinstance(ods, dict)
            else ods["soft_x_rays"]["channel"]
        )
        return len(channels)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def _existing_sxr_source_files(ods: Any) -> list[str]:
    """Read the existing semicolon-delimited IDS source provenance."""
    path = "soft_x_rays.ids_properties.source"
    if not path_exists(ods, path):
        return []
    if isinstance(ods, dict):
        value = ods["soft_x_rays"]["ids_properties"]["source"]
    else:
        value = ods[path]
    return [entry.strip() for entry in str(value).split(";") if entry.strip()]


def _clear_sxr_global_time(ods: Any) -> None:
    """Remove a global time axis when SXR channels have heterogeneous clocks."""
    path = "soft_x_rays.time"
    if not path_exists(ods, path):
        return
    if isinstance(ods, dict):
        ods.get("soft_x_rays", {}).pop("time", None)
    else:
        del ods[path]


def _normalise_source_map(
    daq_label: str,
    data: np.ndarray,
    channel_map: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Fill source metadata and reject ambiguous channels before writing an IDS."""
    mapping = list(channel_map) if channel_map is not None else default_channel_map(daq_label, data.shape[1])
    prepared: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    source_columns: set[int] = set()
    for item in _normalise_channel_map(mapping):
        entry = dict(item)
        entry.setdefault("daq_label", daq_label)
        entry.setdefault(
            "name", f"Digitizer {daq_label} Ch {int(entry['source_column']) + 1}"
        )
        entry.setdefault("identifier", str(entry["name"]))
        identifier = str(entry["identifier"])
        source_column = int(entry["source_column"])
        if identifier in identifiers:
            raise ValueError(
                f"Duplicate SXR channel identifier {identifier!r} for digitizer {daq_label}."
            )
        if source_column in source_columns:
            raise ValueError(f"Duplicate SXR source_column {source_column} for digitizer {daq_label}.")
        identifiers.add(identifier)
        source_columns.add(source_column)
        prepared.append(entry)
    return prepared


def soft_x_rays(
    ods: Any,
    shot: int,
    daq_label: str | int | None = None,
    *,
    data_root: str | Path | None = None,
    digitizer_file: str | Path | None = None,
    geometry_root: str | Path | None = None,
    channel_map: Sequence[Mapping[str, Any]] | None = None,
    channel_maps: Mapping[str | int, Sequence[Mapping[str, Any]]] | None = None,
    sample_rate: float | None = None,
    time_offset: float | None = None,
    time_reference: str = "auto",
    trigger_settings_path: str | Path | None = None,
    energy_band: tuple[float, float] = DEFAULT_ENERGY_BAND,
    brightness_scale: float = 1.0,
    baseline_range: tuple[int | None, int | None] | None = None,
    polarity: float = 1.0,
    channels_as_rows: bool = True,
) -> None:
    """Populate one shot-level SXR IDS from all available digitizer files.

    With no ``daq_label`` this is the canonical API: all supported SXR files
    found for ``shot`` are appended in DAQ-label/source-column order.  Pass a
    ``daq_label`` (or ``digitizer_file`` with a label) to retain the legacy
    single-source behavior.  Each channel retains its native DAQ time axis in
    ``brightness.time``; the IDS is marked non-homogeneous when sources have
    different acquisition axes.

    ``sample_v3.py`` CSVs start at their digitizer trigger.  By default,
    ``time_reference='auto'`` translates that axis to the shot clock using the
    packaged diagnostic trigger settings.  Use ``time_reference='archive'``
    for a zero-origin archive axis, or provide ``time_offset`` to override
    either convention explicitly.
    """
    if digitizer_file is not None:
        inferred_label = _daq_label_from_filename(digitizer_file, shot)
        if daq_label is None:
            if inferred_label is None:
                raise ValueError(
                    "daq_label is required when digitizer_file does not use "
                    "digitizer_{daq}_{shot}.csv naming."
                )
            daq_label = inferred_label
        sources = [
            (str(daq_label), _resolve_digitizer_file(shot, daq_label, data_root, digitizer_file))
        ]
    elif daq_label is not None:
        sources = [(str(daq_label), _resolve_digitizer_file(shot, daq_label, data_root))]
    else:
        sources = _discover_digitizer_files(shot, data_root)
        if not sources:
            raise FileNotFoundError(
                f"No supported SXR digitizer CSV files found for shot {int(shot)}. "
                "Provide data_root, digitizer_file, or daq_label for an explicit source."
            )

    if channel_map is not None and len(sources) > 1:
        raise ValueError(
            "channel_map applies to one DAQ; use channel_maps when mapping multiple digitizers."
        )
    if time_reference not in {"auto", "archive"}:
        raise ValueError("time_reference must be 'auto' or 'archive'.")

    normalised_channel_maps = {
        str(label): mapping for label, mapping in (channel_maps or {}).items()
    }
    existing_identifiers = _existing_sxr_channel_identifiers(ods)
    channel_offset = _existing_sxr_channel_count(ods)
    homogeneous_time = len(sources) == 1 and channel_offset == 0
    existing_sources = _existing_sxr_source_files(ods)
    existing_comment = ""
    comment_path = "soft_x_rays.ids_properties.comment"
    if path_exists(ods, comment_path):
        existing_comment = str(
            ods[comment_path]
            if not isinstance(ods, dict)
            else ods["soft_x_rays"]["ids_properties"]["comment"]
        )
    if not homogeneous_time:
        _clear_sxr_global_time(ods)
    source_details: list[str] = []

    for label, source_file in sources:
        try:
            data = load_digitizer_csv(source_file, channels_as_rows=channels_as_rows)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Malformed SXR digitizer CSV {source_file}: {exc}") from exc
        if data.ndim != 2 or not data.shape[0] or not data.shape[1] or not np.isfinite(data).all():
            raise ValueError(
                f"Malformed SXR digitizer CSV {source_file}: "
                "expected a finite, non-empty 2D array."
            )

        mapping_for_source = channel_map if channel_map is not None else (
            normalised_channel_maps.get(label) if channel_maps is not None else None
        )
        mapping = _normalise_source_map(label, data, mapping_for_source)
        mapping = [item for item in mapping if str(item["identifier"]) not in existing_identifiers]
        if not mapping:
            continue

        resolved_rate = _sample_rate_for_daq(label, sample_rate)
        if time_offset is not None:
            alignment = SXRTimeAlignment(float(time_offset), "explicit", "Caller-provided time_offset.")
        elif time_reference == "auto":
            alignment = resolve_sxr_time_alignment(shot, trigger_settings_path=trigger_settings_path)
        else:
            alignment = SXRTimeAlignment(0.0, "trigger_relative", "Archive trigger-relative time requested.")
        time = build_time_axis(data.shape[0], sample_rate=resolved_rate, time_offset=alignment.offset_seconds)

        vfit_soft_x_rays_static(
            ods,
            channel_map=mapping,
            channel_offset=channel_offset,
            homogeneous_time=homogeneous_time,
            geometry_root=geometry_root,
            energy_band=energy_band,
            brightness_scale=brightness_scale,
        )
        vfit_soft_x_rays_dynamic(
            ods,
            data=data,
            time=time,
            channel_map=mapping,
            channel_offset=channel_offset,
            set_global_time=homogeneous_time,
            brightness_scale=brightness_scale,
            baseline_range=baseline_range,
            polarity=polarity,
        )
        channel_offset += len(mapping)
        existing_identifiers.update(str(item["identifier"]) for item in mapping)
        source_details.append(
            f"{label}:{source_file} sample_rate_hz={resolved_rate:g}; "
            f"time_alignment={alignment.source}"
        )

    if sources and not source_details:
        return
    all_sources = list(dict.fromkeys([*existing_sources, *(str(path) for _, path in sources)]))
    set_path(ods, "soft_x_rays.ids_properties.source", "; ".join(all_sources))
    new_comment = (
        "VEST SXR digitizer data; relative calibrated signal proxy, not absolute brightness. "
        f"Sources: {'; '.join(source_details)}."
    )
    set_path(
        ods,
        "soft_x_rays.ids_properties.comment",
        f"{existing_comment} Sources appended: {'; '.join(source_details)}."
        if existing_comment
        else new_comment,
    )


def soft_x_rays_from_digitizer_csv(
    shot: int,
    daq_label: str | int | None = None,
    *,
    consistency_check: bool = True,
    **kwargs: Any,
):
    """Create a shot-level ODS, discovering all SXR DAQs when no label is given."""
    from omas import ODS

    ods = ODS(consistency_check=consistency_check)
    soft_x_rays(ods, shot, daq_label, **kwargs)
    return ods


def save_soft_x_rays_ods(
    output_path: str | Path,
    shot: int,
    daq_label: str | int | None = None,
    **kwargs: Any,
):
    """Build and save one shot-level soft_x_rays ODS in the selected OMAS format."""
    output = Path(output_path).expanduser()
    consistency_check = kwargs.pop("consistency_check", True)
    ods = soft_x_rays_from_digitizer_csv(
        shot,
        daq_label,
        consistency_check=consistency_check,
        **kwargs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ods.save(str(output))
    return ods


soft_x_rays_from_raw_database = soft_x_rays

__all__ = [
    "DEFAULT_ARRAY_SPECS",
    "DEFAULT_DIGITIZER_ARRAYS",
    "DEFAULT_ENERGY_BAND",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLE_RATES",
    "PACKAGED_GEOMETRY_TABLE",
    "PACKAGED_TRIGGER_SETTINGS",
    "DEFAULT_TIME_OFFSET",
    "SXRDigitizerBlock",
    "SXRTimeAlignment",
    "build_time_axis",
    "default_channel_map",
    "load_digitizer_csv",
    "load_sxr_geometry_table",
    "resolve_sxr_geometry_root",
    "resolve_sxr_geometry_table",
    "resolve_sxr_time_alignment",
    "resolve_sxr_trigger_settings",
    "save_soft_x_rays_ods",
    "soft_x_rays",
    "soft_x_rays_from_digitizer_csv",
    "soft_x_rays_from_raw_database",
    "vfit_soft_x_rays_dynamic",
    "vfit_soft_x_rays_static",
]
