"""Shared helpers for direct-HSDS lazy OMAS and IMAS adapters."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def decode_hdf5_value(value: Any) -> Any:
    """Restore conventional Python/NumPy values from IMAS HDF5 storage."""
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else item
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    elif array.dtype.kind == "O":
        array = np.asarray(
            [
                item.decode("utf-8")
                if isinstance(item, (bytes, np.bytes_))
                else item
                for item in array.flat
            ],
            dtype=object,
        ).reshape(array.shape)
    # IMAS HDF5 stores primitive array dimensions in Fortran order. AOS axes
    # have already been selected, so reverse only the remaining dimensions.
    return array.transpose() if array.ndim > 1 else array


def normalize_ids(ids: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Normalize an optional IDS restriction while preserving input order."""
    if ids is None:
        return None
    values = (ids,) if isinstance(ids, str) else tuple(ids)
    normalized = tuple(
        dict.fromkeys(str(value).removesuffix(".h5") for value in values)
    )
    if not normalized:
        raise ValueError("ids must contain at least one top-level IDS name")
    return normalized


def discover_hsds_ids(h5pyd_module: Any, directory: str, shot: int) -> tuple[str, ...]:
    """List canonical IDS image domains for a shot."""
    return tuple(
        sorted(
            name[:-3]
            for name in h5pyd_module.Folder(f"/{directory}/{shot}/", mode="r")
            if name.endswith(".h5")
            and name != "master.h5"
            and not name.endswith(".h5image.h5")
        )
    )
