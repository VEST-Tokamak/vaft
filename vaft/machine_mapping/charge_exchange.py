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


def vfit_charge_exchange(
    ods: Any,
    shotnumber: int,
    options: str = "single",
    data_root: str | Path | None = None,
) -> None:
    set_path(ods, "charge_exchange.ids_properties.homogeneous_time", 1)
    if options == "single":
        read_doppler_single(ods, shotnumber, data_root=data_root)
    elif options == "profile":
        read_doppler_profile(ods, shotnumber, data_root=data_root)
    else:
        raise ValueError(f"Unsupported charge_exchange option: {options}")


def vfit_ion_doppler_spectroscophy(
    ods: Any,
    shotnumber: int,
    options: str = "single",
    data_root: str | Path | None = None,
) -> None:
    vfit_charge_exchange(ods, shotnumber, options=options, data_root=data_root)


def charge_exchange(
    ods: Any,
    shotnumber: int,
    options: str = "single",
    data_root: str | Path | None = None,
) -> None:
    vfit_charge_exchange(ods, shotnumber, options=options, data_root=data_root)


__all__ = [
    "charge_exchange",
    "read_doppler_profile",
    "read_doppler_single",
    "vfit_charge_exchange",
    "vfit_ion_doppler_spectroscophy",
]
