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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import resolve_data_root, set_path

DEFAULT_SAMPLE_RATE = 125e6 / 128.0
DEFAULT_TIME_OFFSET = 0.285
DEFAULT_ENERGY_BAND = (0.0, 20_000.0)
PACKAGED_GEOMETRY_TABLE = "geometry/line_of_sight_endpoints.csv"


@dataclass(frozen=True)
class SXRArraySpec:
    name: str
    folder: str
    file_template: str
    channels: int
    label: str


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

# Conservative defaults matching the tutorial notebooks in VEST_Soft X-ray.
# Unknown/unassigned digitizer channels are still stored as brightness traces,
# but without LOS metadata.
DEFAULT_DIGITIZER_ARRAYS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "22577": (("lowermid", 0, 16), ("bottom", 16, 16)),
    "17592": (("horizontal", 0, 20), ("vertical", 20, 20)),
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


def _packaged_geometry_table_path() -> Path:
    from vaft.data.resources import data_path

    return data_path(PACKAGED_GEOMETRY_TABLE)


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
    """Return seconds for digitizer samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return np.arange(samples, dtype=float) / float(sample_rate) + float(time_offset)


def default_channel_map(daq_label: str | int, n_channels: int) -> list[dict[str, Any]]:
    """Return default VEST SXR digitizer-to-array channel metadata."""
    label = str(daq_label)
    mapping: list[dict[str, Any]] = []
    used: set[int] = set()

    for array_name, start, count in DEFAULT_DIGITIZER_ARRAYS.get(label, ()):  # known DAQs
        spec = DEFAULT_ARRAY_SPECS[array_name]
        for offset in range(count):
            source_column = start + offset
            if source_column >= n_channels:
                continue
            array_channel = offset + 1
            mapping.append(
                {
                    "source_column": source_column,
                    "daq_label": label,
                    "array": array_name,
                    "array_channel": array_channel,
                    "name": f"{spec.label} Ch {array_channel}",
                    "identifier": f"{label}:{array_name}:{array_channel}",
                }
            )
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


def vfit_soft_x_rays_static(
    ods: Any,
    *,
    channel_map: Sequence[Mapping[str, Any]],
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
    set_path(ods, "soft_x_rays.ids_properties.homogeneous_time", 1)
    set_path(ods, "soft_x_rays.ids_properties.name", "VEST soft X-ray arrays")
    set_path(ods, "soft_x_rays.ids_properties.comment", "VEST SXR digitizer data mapped by vaft")
    set_path(ods, "soft_x_rays.ids_properties.creation_date", datetime.now(timezone.utc).isoformat())

    lower, upper = energy_band
    for idx, item in enumerate(_normalise_channel_map(channel_map)):
        name = str(item.get("name") or f"SXR Ch {idx + 1}")
        identifier = str(item.get("identifier") or name)
        prefix = f"soft_x_rays.channel.{idx}"
        set_path(ods, f"{prefix}.name", name)
        set_path(ods, f"{prefix}.identifier", identifier)
        set_path(ods, f"{prefix}.energy_band.0.lower_bound", float(lower))
        set_path(ods, f"{prefix}.energy_band.0.upper_bound", float(upper))
        set_path(ods, f"{prefix}.etendue", float(item.get("etendue", 1.0)))
        set_path(ods, f"{prefix}.etendue_method.index", -1)
        set_path(ods, f"{prefix}.etendue_method.name", "proxy")
        set_path(
            ods,
            f"{prefix}.etendue_method.description",
            "Default unit etendue; digitizer voltage is stored as brightness proxy unless calibrated.",
        )

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

    set_path(ods, "soft_x_rays.time", time_values)
    for idx, item in enumerate(_normalise_channel_map(channel_map)):
        source_column = int(item["source_column"])
        if source_column < 0 or source_column >= values.shape[1]:
            raise IndexError(f"source_column {source_column} is outside data shape {values.shape}")
        signal = values[:, source_column].astype(float, copy=True)
        if baseline_range is not None:
            start, stop = baseline_range
            baseline = np.mean(signal[slice(start, stop)])
            signal = signal - baseline
        signal = float(polarity) * float(brightness_scale) * signal
        brightness = signal.reshape(-1, 1)
        prefix = f"soft_x_rays.channel.{idx}"
        set_path(ods, f"{prefix}.brightness.time", time_values)
        set_path(ods, f"{prefix}.brightness.data", brightness)
        set_path(ods, f"{prefix}.validity_timed.time", time_values)
        set_path(ods, f"{prefix}.validity_timed.data", np.zeros(time_values.size, dtype=int))


def soft_x_rays(
    ods: Any,
    shot: int,
    daq_label: str | int,
    *,
    data_root: str | Path | None = None,
    digitizer_file: str | Path | None = None,
    geometry_root: str | Path | None = None,
    channel_map: Sequence[Mapping[str, Any]] | None = None,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    time_offset: float = DEFAULT_TIME_OFFSET,
    energy_band: tuple[float, float] = DEFAULT_ENERGY_BAND,
    brightness_scale: float = 1.0,
    baseline_range: tuple[int | None, int | None] | None = None,
    polarity: float = 1.0,
    channels_as_rows: bool = True,
) -> None:
    """Populate ``ods`` with VEST SXR data from ``digitizer_{daq_label}_{shot}.csv``."""
    source_file = _resolve_digitizer_file(
        shot=shot,
        daq_label=daq_label,
        data_root=data_root,
        digitizer_file=digitizer_file,
    )
    data = load_digitizer_csv(source_file, channels_as_rows=channels_as_rows)
    time = build_time_axis(data.shape[0], sample_rate=sample_rate, time_offset=time_offset)
    mapping = list(channel_map) if channel_map is not None else default_channel_map(daq_label, data.shape[1])

    vfit_soft_x_rays_static(
        ods,
        channel_map=mapping,
        geometry_root=geometry_root,
        energy_band=energy_band,
        brightness_scale=brightness_scale,
    )
    vfit_soft_x_rays_dynamic(
        ods,
        data=data,
        time=time,
        channel_map=mapping,
        brightness_scale=brightness_scale,
        baseline_range=baseline_range,
        polarity=polarity,
    )
    set_path(ods, "soft_x_rays.ids_properties.source", str(source_file))


def soft_x_rays_from_digitizer_csv(
    shot: int,
    daq_label: str | int,
    **kwargs: Any,
):
    """Create and return an OMAS ODS filled with one VEST SXR digitizer file."""
    from omas import ODS

    ods = ODS()
    soft_x_rays(ods, shot, daq_label, **kwargs)
    return ods


def save_soft_x_rays_ods(
    output_path: str | Path,
    shot: int,
    daq_label: str | int,
    **kwargs: Any,
):
    """Build and save a soft_x_rays ODS. The file extension selects OMAS format."""
    output = Path(output_path).expanduser()
    consistency_check = kwargs.pop("consistency_check", False)
    ods = soft_x_rays_from_digitizer_csv(shot, daq_label, **kwargs)
    ods.consistency_check = consistency_check
    output.parent.mkdir(parents=True, exist_ok=True)
    ods.save(str(output))
    return ods


soft_x_rays_from_raw_database = soft_x_rays

__all__ = [
    "DEFAULT_ARRAY_SPECS",
    "DEFAULT_DIGITIZER_ARRAYS",
    "DEFAULT_ENERGY_BAND",
    "DEFAULT_SAMPLE_RATE",
    "PACKAGED_GEOMETRY_TABLE",
    "DEFAULT_TIME_OFFSET",
    "build_time_axis",
    "default_channel_map",
    "load_digitizer_csv",
    "load_sxr_geometry_table",
    "resolve_sxr_geometry_root",
    "resolve_sxr_geometry_table",
    "save_soft_x_rays_ods",
    "soft_x_rays",
    "soft_x_rays_from_digitizer_csv",
    "soft_x_rays_from_raw_database",
    "vfit_soft_x_rays_dynamic",
    "vfit_soft_x_rays_static",
]
