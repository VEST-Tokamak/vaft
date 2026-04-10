"""Canonical spectrometer_uv builders integrated under machine_mapping."""

from __future__ import annotations

import numpy as np

from vaft.database import raw as raw_db

from .utils import set_path

DEFAULT_DT = 4e-5

CHANNEL_NAMES: dict[int, str] = {
    0: "H alpha Filterscope",
    1: "O-I Filterscope",
    2: "Versatile Filterscope",
}

SIGNALS: list[tuple[int, int, int, str, float]] = [
    (101, 0, 0, "H-alpha_6563", 656.3e-9),
    (214, 1, 0, "OI_7770", 777.0e-9),
    (144, 2, 0, "H-alpha_6563", 656.3e-9),
    (141, 2, 1, "H-beta_4861", 486.1e-9),
    (138, 2, 2, "H-gamma_4340", 434.0e-9),
    (142, 2, 3, "CII_3726", 372.6e-9),
    (140, 2, 4, "CIII_1909", 190.9e-9),
    (139, 2, 5, "OII_3726", 372.6e-9),
    (143, 2, 6, "OV_629", 62.9e-9),
]


def _safe_vest_load(shot: int, field: int):
    if not raw_db.sql_loading_available():
        return None
    return raw_db.vest_load(shot, field)


def _build_time_axis(t_start: float, t_end: float, dt: float) -> np.ndarray:
    start = max(t_start, 0.0)
    end = min(t_end, 1.0)
    step = dt if dt > 0 else DEFAULT_DT
    if end <= start:
        return np.array([start], dtype=float)
    return np.arange(start, end, step)


def _needs_legacy_time_shift(shot: int) -> bool:
    return (41446 <= shot <= 41451) or (shot >= 41660)


def vfit_filterscope(ods: object, shot: int, t_start: float, t_end: float, dt: float) -> None:
    set_path(ods, "spectrometer_uv.ids_properties.comment", "VEST filterscope data")
    set_path(ods, "spectrometer_uv.ids_properties.homogeneous_time", 1)

    for channel, name in CHANNEL_NAMES.items():
        set_path(ods, f"spectrometer_uv.channel.{channel}.name", name)

    for _, channel, line, label, wavelength in SIGNALS:
        set_path(ods, f"spectrometer_uv.channel.{channel}.processed_line.{line}.label", label)
        set_path(
            ods,
            f"spectrometer_uv.channel.{channel}.processed_line.{line}.wavelength_central",
            wavelength,
        )

    time = _build_time_axis(t_start, t_end, dt)
    shift = 0.26 if _needs_legacy_time_shift(shot) else 0.24

    for field, channel, line, _, _ in SIGNALS:
        intensity_key = f"spectrometer_uv.channel.{channel}.processed_line.{line}.intensity.data"
        loaded = _safe_vest_load(shot, field)
        if loaded is None:
            set_path(ods, intensity_key, np.zeros(time.size))
            continue

        source_time, source_data = loaded
        source_time = np.asarray(source_time, dtype=float)
        source_data = np.asarray(source_data, dtype=float)
        if source_time.size <= 1 or source_data.size <= 1:
            set_path(ods, intensity_key, np.zeros(time.size))
            continue

        if source_time[-1] < 0.1:
            source_time = source_time + shift
        set_path(ods, intensity_key, -np.interp(time, source_time, source_data))

    set_path(ods, "spectrometer_uv.time", time)


def spectrometer_uv(ods: object, shot: int, t_start: float, t_end: float, dt: float) -> None:
    """Canonical machine_mapping entry point for the spectrometer_uv IDS."""
    vfit_filterscope(ods, shot, t_start, t_end, dt)


def filterscope_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    del options
    spectrometer_uv(ods, shot, tstart, tend, dt)


__all__ = ["filterscope_from_raw_database", "spectrometer_uv", "vfit_filterscope"]
