"""Canonical spectrometer_uv builders integrated under machine_mapping."""

from __future__ import annotations

import numpy as np

from vaft.database import raw as raw_db
from vaft.process.signal_processing import resample_to_time

from .utils import set_path

DEFAULT_DT = 4e-5

CHANNEL_NAMES: dict[int, str] = {
    0: "H alpha Filterscope",
    1: "O-I Filterscope",
    2: "Versatile Filterscope",
}

#: Native digitizer rate per channel: channels 0 and 1 are on the slow DAQ,
#: the versatile filterscope (channel 2) on the fast DAQ.  Every policy grid
#: is 25 kHz, so channel 2 is resampled (see :func:`vfit_filterscope`).
CHANNEL_CADENCE_HZ: dict[int, float] = {
    0: 25e3,
    1: 25e3,
    2: 250e3,
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


def _build_time_axis(t_start: float, t_end: float, dt: float) -> np.ndarray:
    start = max(t_start, 0.0)
    end = min(t_end, 1.0)
    step = dt if dt > 0 else DEFAULT_DT
    if end <= start:
        return np.array([start], dtype=float)
    return np.arange(start, end, step)


def _needs_legacy_time_shift(shot: int) -> bool:
    return (41446 <= shot <= 41451) or (shot >= 41660)


def legacy_time_shift_s(shot: int) -> float:
    """The offset the mapper adds to a filterscope record's own time axis.

    A fast-DAQ record starts at zero on its own clock and the mapper adds
    this to place it on the discharge clock.  The mapper applies it to any
    record whose axis ends before 0.1 s -- in practice the fast channel; a
    slow-DAQ record spans the full second -- so this is the value a shifted
    record received, not proof that a given record was shifted.
    """
    return 0.26 if _needs_legacy_time_shift(shot) else 0.24


def vfit_filterscope(
    ods: object,
    shot: int,
    t_start: float,
    t_end: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
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

    time = (
        np.asarray(target_time, dtype=float)
        if target_time is not None
        else _build_time_axis(t_start, t_end, dt)
    )
    shift = legacy_time_shift_s(shot)

    loaded_signals = {
        field: raw_db.require_signal(
            _safe_vest_load(shot, field, raw_source),
            shot=shot,
            field=field,
            signal_name=label,
        )
        for field, _, _, label, _ in SIGNALS
    }

    for field, channel, line, _, _ in SIGNALS:
        intensity_key = f"spectrometer_uv.channel.{channel}.processed_line.{line}.intensity.data"
        source_time, source_data = loaded_signals[field]

        if source_time[-1] < 0.1:
            source_time = source_time + shift
        # The versatile filterscope (channel 2) is on the fast DAQ at 250 kHz
        # while every policy grid is 25 kHz, so this is a 10x decimation and a
        # bare np.interp would fold the filterscope's broadband noise into the
        # stored band.  resample_to_time low-passes first where the rate really
        # drops and is bit-for-bit np.interp for the slow-DAQ channels.
        set_path(ods, intensity_key, -resample_to_time(source_time, source_data, time))

    set_path(ods, "spectrometer_uv.time", time)


def spectrometer_uv(
    ods: object,
    shot: int,
    t_start: float,
    t_end: float,
    dt: float,
    *,
    raw_source: raw_db.RawSource | None = None,
    target_time: np.ndarray | None = None,
) -> None:
    """Canonical machine_mapping entry point for the spectrometer_uv IDS."""
    vfit_filterscope(
        ods, shot, t_start, t_end, dt, raw_source=raw_source, target_time=target_time
    )


def filterscope_from_raw_database(
    ods: object,
    shot: int,
    tstart: float,
    tend: float,
    dt: float,
    options: dict | None = None,
) -> None:
    raw_source = options.get("raw_source") if options else None
    spectrometer_uv(ods, shot, tstart, tend, dt, raw_source=raw_source)


__all__ = [
    "CHANNEL_CADENCE_HZ",
    "SIGNALS",
    "filterscope_from_raw_database",
    "legacy_time_shift_s",
    "spectrometer_uv",
    "vfit_filterscope",
]
