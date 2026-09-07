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


#: DCON sorts its energy eigenvalues so the least-stable one is labelled 1:
#: ``dcon/free.f``'s ``plasma1``/``vacuum1``/``total1`` are the
#: ``ep(1)``/``ev(1)``/``et(1)`` entries.
LEAST_STABLE_MODE_LABEL = 1


def least_stable_eigenvalue(
    eigenvalues: Optional[np.ndarray], labels: Optional[np.ndarray]
) -> Optional[complex]:
    """The least-stable entry of an energy-eigenvalue array, selected by mode *label*.

    The netCDF writes a ``mode`` coordinate of ``1..mpert``
    (``dcon/dcon_netcdf.f``'s ``nf90_put_var(ncid,mo_id,(/(i,i=1,mpert)/))``),
    and the value we want is the one labelled
    :data:`LEAST_STABLE_MODE_LABEL`. Position 0 is the same entry for every file
    the suite writes today, but selecting by label keeps that an explicit,
    checkable assumption rather than a silent one -- a differently-ordered
    ``mode`` coordinate would otherwise change which eigenvalue is reported with
    no visible signal.

    ``labels`` may be ``None`` for a file that carries no ``mode`` coordinate,
    in which case the leading entry is used.
    """
    if eigenvalues is None:
        return None
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.size == 0:
        return None
    if labels is None:
        return complex(eigenvalues.reshape(-1)[0])
    matches = np.flatnonzero(np.asarray(labels) == LEAST_STABLE_MODE_LABEL)
    if matches.size == 0:
        return None
    return complex(eigenvalues[int(matches[0])])
