"""Canonical thomson_scattering builders integrated under machine_mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def vfit_thomson_scattering_static(ods: Any) -> None:
    set_path(ods, "thomson_scattering.ids_properties.homogeneous_time", 1)
    for channel, r_pos, name, _ in _CHANNEL_META:
        prefix = f"thomson_scattering.channel.{channel}"
        set_path(ods, f"{prefix}.position.r", r_pos)
        set_path(ods, f"{prefix}.position.z", 0)
        set_path(ods, f"{prefix}.name", name)


def vfit_thomson_scattering_dynamic(ods: Any, shotnumber: int, data_root: str | Path | None = None) -> None:
    mat_file = resolve_data_root(data_root) / "thomson_scattering" / f"NeTe_Shot{shotnumber}_v9.mat"
    mat_data = loadmat(str(mat_file))

    set_path(ods, "thomson_scattering.time", mat_data["time_TS"][0])
    for channel, _, _, tag in _CHANNEL_META:
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.t_e.data",
            _uarray_or_values(mat_data[f"{tag}_Te"][0], mat_data[f"{tag}_sigmaTe"][0]),
        )
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.n_e.data",
            _uarray_or_values(mat_data[f"{tag}_Ne"][0], mat_data[f"{tag}_sigmaNe"][0]),
        )


def thomson_scattering(ods: Any, shotnumber: int, data_root: str | Path | None = None) -> None:
    vfit_thomson_scattering_static(ods)
    vfit_thomson_scattering_dynamic(ods, shotnumber, data_root=data_root)


__all__ = [
    "thomson_scattering",
    "vfit_thomson_scattering_dynamic",
    "vfit_thomson_scattering_static",
]
