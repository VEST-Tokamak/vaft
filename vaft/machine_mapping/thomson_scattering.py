"""Canonical thomson_scattering builders integrated under machine_mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .utils import resolve_data_root, set_path

try:
    import uncertainties.unumpy as unumpy
except ImportError:
    unumpy = None


_CHANNEL_META = (
    (0, 0.475, "Polychrometer 1R1", "poly1R1"),
    (1, 0.425, "Polychrometer 2R2", "poly2R2"),
    (2, 0.370, "Polychrometer 3R3", "poly3R3"),
    (3, 0.310, "Polychrometer 4R4", "poly4R4"),
    (4, 0.255, "Polychrometer 5R5", "poly5R5"),
)


def _uarray_or_values(values: Any, errors: Any) -> Any:
    if unumpy is None:
        return values
    return unumpy.uarray(values, errors)


def _as_real_array(values: Any) -> np.ndarray:
    return np.real(np.asarray(values)).astype(float)


def _normalize_thomson_time_to_seconds(time_values: np.ndarray) -> np.ndarray:
    """Return Thomson time in seconds (MAT time is treated as milliseconds)."""
    time = np.asarray(time_values, dtype=float).reshape(-1)
    return time / 1e3


def _candidate_thomson_paths(shotnumber: int, data_root: Path) -> list[Path]:
    return [
        data_root / "thomson_scattering" / f"NeTe_Shot{shotnumber}_v9.mat",
        data_root / "thomson_scattering" / f"NeTe_Shot{shotnumber}_v9_rev.mat",
        data_root / f"NeTe_Shot{shotnumber}_v9.mat",
        data_root / f"NeTe_Shot{shotnumber}_v9_rev.mat",
        data_root / f"{shotnumber}_NeTe.mat",
    ]


def _resolve_thomson_mat_file(
    shotnumber: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> Path:
    """Resolve a Thomson MAT path.

    ``data_root`` may be a directory (default search root) or a legacy positional
    path to a specific ``*.mat`` file (as used by ``thomson_scattering(ods, shot, filepath)``).
    A ``*.mat`` path must not be used as a directory when probing ``NeTe_Shot{shot}_v9`` etc.;
    in that case the parent directory (or package ``vaft/data``) is used for those patterns.
    """
    pkg_data = resolve_data_root(None)
    candidates: list[Path] = []

    if mat_file is not None:
        explicit = Path(mat_file)
        if explicit.is_absolute():
            candidates.append(explicit)
        else:
            if data_root is None:
                base = pkg_data
            else:
                dr = Path(data_root)
                if dr.suffix.lower() == ".mat" or (dr.exists() and dr.is_file()):
                    base = dr.parent if dr.parent.is_dir() else pkg_data
                else:
                    base = dr
            candidates.append(base / explicit)
            candidates.append(explicit)
    else:
        if data_root is None:
            search_root = pkg_data
        else:
            dr = Path(data_root)
            if dr.suffix.lower() == ".mat" or dr.is_file():
                # Legacy: third positional argument is often a full path to a MAT file.
                if dr.is_absolute():
                    candidates.append(dr)
                else:
                    candidates.append(dr)
                    candidates.append(pkg_data / dr)
                search_root = dr.parent if dr.parent.is_dir() else pkg_data
            else:
                search_root = dr

        candidates.extend(_candidate_thomson_paths(shotnumber, search_root))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot find Thomson MAT file for shot {shotnumber}; searched: {searched}"
    )


def _set_dynamic_from_v9(mat_data: dict[str, Any], ods: Any) -> None:
    time = _normalize_thomson_time_to_seconds(_as_real_array(mat_data["time_TS"]))
    set_path(ods, "thomson_scattering.time", time)
    for channel, _, _, tag in _CHANNEL_META:
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.t_e.data",
            _uarray_or_values(
                _as_real_array(mat_data[f"{tag}_Te"]).reshape(-1),
                _as_real_array(mat_data[f"{tag}_sigmaTe"]).reshape(-1),
            ),
        )
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.n_e.data",
            _uarray_or_values(
                _as_real_array(mat_data[f"{tag}_Ne"]).reshape(-1),
                _as_real_array(mat_data[f"{tag}_sigmaNe"]).reshape(-1),
            ),
        )


def _extract_channel_series(matrix: np.ndarray, channel: int, time_len: int) -> np.ndarray:
    if matrix.ndim == 1:
        return matrix.reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {matrix.shape}")
    if matrix.shape[0] == time_len and matrix.shape[1] > channel:
        return matrix[:, channel]
    if matrix.shape[1] == time_len and matrix.shape[0] > channel:
        return matrix[channel, :]
    raise ValueError(
        f"Cannot infer time/channel axes for shape {matrix.shape} and time_len={time_len}"
    )


def _set_dynamic_from_simple(mat_data: dict[str, Any], ods: Any) -> None:
    time = _normalize_thomson_time_to_seconds(_as_real_array(mat_data["time"]))
    te = _as_real_array(mat_data["Te"])
    ne = _as_real_array(mat_data["Ne"])
    sigma_te = _as_real_array(mat_data["sigmaTe"])
    sigma_ne = _as_real_array(mat_data["sigmaNe"])

    set_path(ods, "thomson_scattering.time", time)
    for channel, _, _, _ in _CHANNEL_META:
        te_values = _extract_channel_series(te, channel, time.size)
        ne_values = _extract_channel_series(ne, channel, time.size)
        te_err = _extract_channel_series(sigma_te, channel, time.size)
        ne_err = _extract_channel_series(sigma_ne, channel, time.size)
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.t_e.data",
            _uarray_or_values(te_values, te_err),
        )
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.n_e.data",
            _uarray_or_values(ne_values, ne_err),
        )


def vfit_thomson_scattering_static(ods: Any) -> None:
    set_path(ods, "thomson_scattering.ids_properties.homogeneous_time", 1)
    for channel, r_pos, name, _ in _CHANNEL_META:
        prefix = f"thomson_scattering.channel.{channel}"
        set_path(ods, f"{prefix}.position.r", r_pos)
        set_path(ods, f"{prefix}.position.z", 0)
        set_path(ods, f"{prefix}.name", name)


def vfit_thomson_scattering_dynamic(
    ods: Any,
    shotnumber: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    source_file = _resolve_thomson_mat_file(
        shotnumber=shotnumber,
        data_root=data_root,
        mat_file=mat_file,
    )
    mat_data = loadmat(str(source_file))

    if "time_TS" in mat_data:
        _set_dynamic_from_v9(mat_data, ods)
        return

    simple_keys = {"time", "Te", "sigmaTe", "Ne", "sigmaNe"}
    if simple_keys.issubset(mat_data):
        _set_dynamic_from_simple(mat_data, ods)
        return

    available = sorted(key for key in mat_data.keys() if not key.startswith("__"))
    raise KeyError(
        f"Unsupported Thomson MAT schema in {source_file}. "
        f"Expected v9 keys (time_TS/poly*) or simple keys {sorted(simple_keys)}; "
        f"available keys: {available}"
    )


def thomson_scattering(
    ods: Any,
    shotnumber: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    vfit_thomson_scattering_static(ods)
    vfit_thomson_scattering_dynamic(
        ods,
        shotnumber,
        data_root=data_root,
        mat_file=mat_file,
    )


__all__ = [
    "thomson_scattering",
    "vfit_thomson_scattering_dynamic",
    "vfit_thomson_scattering_static",
]
