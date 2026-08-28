"""Shared netCDF reading helpers for GPEC-suite solver output.

Every solver in the suite writes complex quantities the same way -- as a real
array with a trailing length-2 ``i`` dimension holding the real and imaginary
parts (``nf90_put_var(ncid,i_id,(/0,1/))`` in ``dcon/dcon_netcdf.f``,
``rdcon/rdcon_netcdf.f`` and ``stride/stride_netcdf.f`` alike) -- so the
decoding lives here once rather than being duplicated per solver module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def complex_var(ds, name: str) -> Optional[np.ndarray]:
    """Read a ``(..., i)`` netCDF variable as a complex array over its leading dims.

    Returns ``None`` when the variable is absent or does not carry the
    real/imaginary ``i`` dimension, so callers can treat "this run did not
    compute that quantity" as a normal, non-fatal case.
    """
    if name not in ds.variables:
        return None
    var = ds[name]
    if "i" not in var.dims:
        return None
    real = var.isel(i=0).values
    imag = var.isel(i=1).values
    return np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)


def complex_scalar_attr(ds, name: str) -> Optional[complex]:
    """Read a complex-valued netCDF *global attribute* (e.g. ``plasma1``/``total1``).

    xarray/netCDF4 surfaces such an attribute as a 2-element real array (or,
    on some builds, as a Python complex directly) -- handle both.
    """
    if name not in ds.attrs:
        return None
    raw = ds.attrs[name]
    if isinstance(raw, complex):
        return raw
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size >= 2:
        return complex(arr[0], arr[1])
    if arr.size == 1:
        return complex(arr[0], 0.0)
    return None
