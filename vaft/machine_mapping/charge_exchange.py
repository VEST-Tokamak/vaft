"""Canonical charge_exchange builders integrated under machine_mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from .utils import resolve_data_root, set_path

try:
    import uncertainties.unumpy as unumpy
except ImportError:
    unumpy = None


ION_METADATA = {
    "C3+": {"a": 12, "z_ion": 3, "z_n": 6},
    "O2+": {"a": 16, "z_ion": 2, "z_n": 8},
}
MODE_ALIASES = {
    "ces": "ces",
    "charge_exchange": "ces",
    "profile": "ces",  # backward compatibility
    "single": "ces",  # backward compatibility
    "ids": "ids",
    "ion_doppler": "ids",
    "ids_mat": "ids",  # backward compatibility
}


def _uarray_or_values(values: Any, errors: Any) -> Any:
    if unumpy is None:
        return values
    return unumpy.uarray(values, errors)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas and xlrd are required for single-mode charge_exchange imports"
        ) from exc
    return pd


def _set_ion_metadata(ods: Any, prefix: str, label: str) -> None:
    meta = ION_METADATA[label]
    set_path(ods, f"{prefix}.label", label)
    set_path(ods, f"{prefix}.a", meta["a"])
    set_path(ods, f"{prefix}.z_ion", meta["z_ion"])
    set_path(ods, f"{prefix}.z_n", meta["z_n"])


def _resolve_mode(options: str) -> str:
    mode = str(options).strip().lower()
    if mode not in MODE_ALIASES:
        valid = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Unsupported charge_exchange option: {options}. Supported: {valid}")
    return MODE_ALIASES[mode]


def read_doppler_single(ods: Any, shotnumber: int, data_root: str | Path | None = None) -> None:
    pd = _require_pandas()
    filename = resolve_data_root(data_root) / "ion_doppler_spectroscophy" / f"{shotnumber}.csv"

    set_path(ods, "charge_exchange.channel.0.name", "Ion Doppler Spectroscopy")
    _set_ion_metadata(ods, "charge_exchange.channel.0.ion.0", "C3+")
    _set_ion_metadata(ods, "charge_exchange.channel.0.ion.1", "O2+")
    set_path(ods, "charge_exchange.channel.0.position.r.data", 0.39)

    df = pd.read_excel(filename, engine="xlrd")
    set_path(ods, "charge_exchange.time", np.array(df["Time [ms]"]))
    set_path(
        ods,
        "charge_exchange.channel.0.ion.0.intensity.data",
        _uarray_or_values(df.iloc[:, 2], df.iloc[:, 3]),
    )
    set_path(
        ods,
        "charge_exchange.channel.0.ion.0.velocity_tor.data",
        _uarray_or_values(df.iloc[:, 4] * 1000, df.iloc[:, 5] * 1000),
    )
    set_path(
        ods,
        "charge_exchange.channel.0.ion.0.t_i.data",
        _uarray_or_values(df.iloc[:, 6], df.iloc[:, 7]),
    )
    set_path(
        ods,
        "charge_exchange.channel.0.ion.1.intensity.data",
        _uarray_or_values(df.iloc[:, 8], df.iloc[:, 9]),
    )
    set_path(
        ods,
        "charge_exchange.channel.0.ion.1.velocity_tor.data",
        _uarray_or_values(df.iloc[:, 10] * 1000, df.iloc[:, 11] * 1000),
    )
    set_path(
        ods,
        "charge_exchange.channel.0.ion.1.t_i.data",
        _uarray_or_values(df.iloc[:, 12], df.iloc[:, 13]),
    )


def read_doppler_profile(
    ods: Any,
    shotnumber: int,
    line: str = "C3+",
    data_root: str | Path | None = None,
) -> None:
    if line not in ION_METADATA:
        raise ValueError(f"Unsupported ion line: {line}")

    shot_dir = resolve_data_root(data_root) / "ion_doppler_spectroscophy" / str(shotnumber)
    mat_files = sorted(
        shot_dir.glob(f"{shotnumber}_*.mat"),
        key=lambda path: float(path.stem.split("_")[1]),
    )

    set_path(
        ods,
        "charge_exchange.time",
        np.array([float(path.stem.split("_")[1]) for path in mat_files]) / 1000.0,
    )

    results = []
    for mat_file in mat_files:
        mat = scipy.io.loadmat(str(mat_file))
        results.append(mat["result"])

    try:
        result_stack = np.stack(results, axis=2)
    except ValueError:
        print(f"Error: {shotnumber} has no data")
        return

    channel_count = len(result_stack[:, 0, 0])
    for channel in range(channel_count):
        channel_prefix = f"charge_exchange.channel.{channel}"
        set_path(ods, f"{channel_prefix}.position.r.data", np.array(result_stack[channel, 0, :]))
        set_path(
            ods,
            f"{channel_prefix}.ion.0.intensity.data",
            _uarray_or_values(result_stack[channel, 1, :], result_stack[channel, 2, :]),
        )
        set_path(
            ods,
            f"{channel_prefix}.ion.0.velocity_tor.data",
            _uarray_or_values(result_stack[channel, 3, :] * 1000, result_stack[channel, 4, :] * 1000),
        )
        set_path(
            ods,
            f"{channel_prefix}.ion.0.t_i.data",
            _uarray_or_values(result_stack[channel, 5, :], result_stack[channel, 6, :]),
        )
        _set_ion_metadata(ods, f"{channel_prefix}.ion.0", line)


def read_doppler_ids_mat(
    ods: Any,
    shotnumber: int | None = None,
    mat_file: str | Path | None = None,
    data_root: str | Path | None = None,
    line: str = "C3+",
) -> None:
    if line not in ION_METADATA:
        raise ValueError(f"Unsupported ion line: {line}")

    if mat_file is None:
        if shotnumber is None:
            raise ValueError("shotnumber is required when mat_file is not provided")
        source_path = resolve_data_root(data_root) / f"IDS_{int(shotnumber)}.mat"
    else:
        source_path = Path(mat_file)
        if not source_path.is_absolute():
            # Prefer caller-relative path when it exists, then fall back to data_root.
            source_path = source_path if source_path.exists() else resolve_data_root(data_root) / source_path

    mat = scipy.io.loadmat(str(source_path))

    required_keys = (
        "Rposition",
        "time_IDS",
        "temperature",
        "temperature_err",
        "velocity",
        "velocity_err",
        "emissivity",
        "emissivity_err",
    )
    missing = [key for key in required_keys if key not in mat]
    if missing:
        raise KeyError(f"Missing required IDS keys in {source_path}: {missing}")

    r_position = np.asarray(mat["Rposition"], dtype=float).reshape(-1)
    times_ms = np.asarray(mat["time_IDS"], dtype=float).reshape(-1)
    # Keep charge_exchange.time in seconds for process/plot consistency.
    set_path(ods, "charge_exchange.time", times_ms / 1000.0)

    intensity = np.asarray(mat["emissivity"], dtype=float)
    intensity_err = np.asarray(mat["emissivity_err"], dtype=float)
    temperature = np.asarray(mat["temperature"], dtype=float)
    temperature_err = np.asarray(mat["temperature_err"], dtype=float)
    # IDS velocity is stored in km/s in this source; convert to m/s.
    velocity = np.asarray(mat["velocity"], dtype=float) * 1000.0
    velocity_err = np.asarray(mat["velocity_err"], dtype=float) * 1000.0

    n_channels = r_position.size
    n_times = times_ms.size
    times_s = times_ms / 1000.0
    arrays_to_validate = (
        ("emissivity", intensity),
        ("emissivity_err", intensity_err),
        ("temperature", temperature),
        ("temperature_err", temperature_err),
        ("velocity", velocity),
        ("velocity_err", velocity_err),
    )
    for name, array in arrays_to_validate:
        if array.shape != (n_channels, n_times):
            raise ValueError(
                f"Unexpected shape for {name}: {array.shape}; expected {(n_channels, n_times)}"
            )

    for channel in range(n_channels):
        channel_prefix = f"charge_exchange.channel.{channel}"
        set_path(ods, f"{channel_prefix}.name", "Ion Doppler Spectroscopy")
        set_path(ods, f"{channel_prefix}.position.r.time", np.asarray(times_s, dtype=float))
        set_path(
            ods,
            f"{channel_prefix}.position.r.data",
            np.full(n_times, float(r_position[channel]), dtype=float),
        )
        # IDS_47518.mat does not contain channel z-position; assume mid-plane.
        set_path(ods, f"{channel_prefix}.position.z.time", np.asarray(times_s, dtype=float))
        set_path(ods, f"{channel_prefix}.position.z.data", np.zeros(n_times, dtype=float))
        _set_ion_metadata(ods, f"{channel_prefix}.ion.0", line)
        set_path(
            ods,
            f"{channel_prefix}.ion.0.velocity_tor.data",
            _uarray_or_values(velocity[channel, :], velocity_err[channel, :]),
        )
        set_path(
            ods,
            f"{channel_prefix}.ion.0.t_i.data",
            _uarray_or_values(temperature[channel, :], temperature_err[channel, :]),
        )


def read_charge_exchange_ces_mat(
    ods: Any,
    shotnumber: int | None = None,
    mat_file: str | Path | None = None,
    data_root: str | Path | None = None,
    line: str = "C3+",
    default_time_s: float = 0.300,
) -> None:
    if line not in ION_METADATA:
        raise ValueError(f"Unsupported ion line: {line}")

    if mat_file is None:
        if shotnumber is None:
            raise ValueError("shotnumber is required when mat_file is not provided")
        source_path = resolve_data_root(data_root) / f"CES_{int(shotnumber)}.mat"
    else:
        source_path = Path(mat_file)
        if not source_path.is_absolute():
            source_path = source_path if source_path.exists() else resolve_data_root(data_root) / source_path

    mat = scipy.io.loadmat(str(source_path))
    required_keys = ("LOS", "LOS_err", "velocity", "velocity_err", "temperature", "temperature_err")
    missing = [key for key in required_keys if key not in mat]
    if missing:
        raise KeyError(f"Missing required CES keys in {source_path}: {missing}")

    los = np.asarray(mat["LOS"], dtype=float).reshape(-1)
    los_err = np.asarray(mat["LOS_err"], dtype=float).reshape(-1)
    temperature = np.asarray(mat["temperature"], dtype=float).reshape(-1)
    temperature_err = np.asarray(mat["temperature_err"], dtype=float).reshape(-1)
    # CES velocity in this source is km/s; convert to m/s.
    velocity = np.asarray(mat["velocity"], dtype=float).reshape(-1) * 1000.0
    velocity_err = np.asarray(mat["velocity_err"], dtype=float).reshape(-1) * 1000.0

    n_channels = los.size
    lengths = (los_err.size, temperature.size, temperature_err.size, velocity.size, velocity_err.size)
    if any(length != n_channels for length in lengths):
        raise ValueError(
            "CES arrays must have matching channel lengths: "
            f"LOS={n_channels}, LOS_err={los_err.size}, "
            f"T={temperature.size}, T_err={temperature_err.size}, "
            f"V={velocity.size}, V_err={velocity_err.size}"
        )

    times_s = np.array([float(default_time_s)], dtype=float)
    set_path(ods, "charge_exchange.time", times_s)

    for channel in range(n_channels):
        channel_prefix = f"charge_exchange.channel.{channel}"
        set_path(ods, f"{channel_prefix}.name", "Charge Exchange Spectroscopy")
        set_path(ods, f"{channel_prefix}.position.r.time", times_s)
        set_path(
            ods,
            f"{channel_prefix}.position.r.data",
            _uarray_or_values(np.array([los[channel]]), np.array([abs(los_err[channel])])),
        )
        set_path(ods, f"{channel_prefix}.position.z.time", times_s)
        set_path(ods, f"{channel_prefix}.position.z.data", np.array([0.0], dtype=float))
        _set_ion_metadata(ods, f"{channel_prefix}.ion.0", line)
        set_path(
            ods,
            f"{channel_prefix}.ion.0.velocity_tor.data",
            _uarray_or_values(np.array([velocity[channel]]), np.array([abs(velocity_err[channel])])),
        )
        set_path(
            ods,
            f"{channel_prefix}.ion.0.t_i.data",
            _uarray_or_values(np.array([temperature[channel]]), np.array([abs(temperature_err[channel])])),
        )


def vfit_charge_exchange(
    ods: Any,
    shotnumber: int,
    options: str = "ces",
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    mode = _resolve_mode(options)
    set_path(ods, "charge_exchange.ids_properties.homogeneous_time", 1)
    if mode == "ces":
        if mat_file is not None:
            read_charge_exchange_ces_mat(
                ods,
                shotnumber=shotnumber,
                mat_file=mat_file,
                data_root=data_root,
            )
        else:
            read_doppler_profile(ods, shotnumber, data_root=data_root)
    elif mode == "ids":
        read_doppler_ids_mat(
            ods,
            shotnumber=shotnumber,
            mat_file=mat_file,
            data_root=data_root,
        )


def vfit_ion_doppler_spectroscophy(
    ods: Any,
    shotnumber: int,
    options: str = "ces",
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    vfit_charge_exchange(
        ods,
        shotnumber,
        options=options,
        data_root=data_root,
        mat_file=mat_file,
    )


def charge_exchange(
    ods: Any,
    shotnumber: int,
    options: str = "ces",
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    vfit_charge_exchange(
        ods,
        shotnumber,
        options=options,
        data_root=data_root,
        mat_file=mat_file,
    )


__all__ = [
    "charge_exchange",
    "read_charge_exchange_ces_mat",
    "read_doppler_ids_mat",
    "read_doppler_profile",
    "read_doppler_single",
    "vfit_charge_exchange",
    "vfit_ion_doppler_spectroscophy",
]
