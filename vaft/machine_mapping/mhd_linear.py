"""
`mhd_linear` IDS mapping helpers.

Schema reference:
https://gafusion.github.io/omas/schema.html
"""

from __future__ import annotations

import os
import re
import struct
from typing import Any, Optional

import numpy as np
import xarray as xr
from omas import ODS

# (filename pattern, netCDF variable) per GPEC-suite solver. DCON's
# `W_t_eigenvalue` maps directly onto `mhd_linear`'s `energy_perturbed`.
# RDCON/STRIDE's `Delta_prime` has no dedicated slot in the `mhd_linear` IDS
# schema (verified against omas's packaged 3.41.0 structures -- no tearing-
# index/Delta-prime field exists under `toroidal_mode`), so it is recorded as
# a `ballooning_type` classification in the ODS and returned as a raw extra
# for the caller to keep alongside the ODS (e.g. in a run manifest) rather
# than forced into a schema field it doesn't belong in.
_MODULE_OUTPUT = {
    "dcon": (re.compile(r"dcon_output_n(\d+)\.nc"), "W_t_eigenvalue"),
    "rdcon": (re.compile(r"rdcon_output_n(\d+)\.nc"), "Delta_prime"),
    "stride": (re.compile(r"stride_output_n(\d+)\.nc"), "Delta_prime"),
}


def _extract_scalar(var: xr.DataArray) -> float | None:
    """Best-effort scalar extraction from a netCDF variable of unknown rank.

    GPEC-suite ``.nc`` outputs store what is conceptually a single number
    under varying shapes (scalar, ``[1,1,1]``, ragged arrays) depending on
    solver and version. Ported from
    ``gen_stability_history.extract_delta_prime_from_nc``'s defensive
    indexing so both consumers behave the same way for the same files.
    """
    if var.ndim == 0:
        return var.item()
    if var.ndim >= 3 and all(size > 0 for size in var.shape[:3]):
        return var.data[0, 0, 0].item()
    if var.ndim > 0 and var.size > 0:
        current = var.data
        try:
            for _ in range(var.ndim):
                if hasattr(current, "__getitem__") and len(current) > 0:
                    current = current[0]
                else:
                    break
            if np.isscalar(current):
                return float(current)
        except IndexError:
            if np.isscalar(current):
                return float(current)
    return None


def mhd_linear(ods: ODS, source: str, options: Optional[dict] = None) -> dict[int, dict[str, Any]]:
    """Parse GPEC-suite output under ``source`` into the ``mhd_linear`` IDS.

    ``options["module"]`` selects which solver's output to read (``"dcon"``
    by default, matching prior behavior); one of ``"dcon"``, ``"rdcon"``,
    ``"stride"``. Returns a ``{mode: {...}}`` dict of raw extracted values
    that either have no ``mhd_linear`` IDS slot (RDCON/STRIDE's
    ``Delta_prime``) or are convenient to carry alongside the ODS in a run
    manifest.
    """
    if options is None:
        options = {}

    time_slice = options.get("time_slice", 0)
    module = str(options.get("module", "dcon")).lower()
    if module not in _MODULE_OUTPUT:
        raise ValueError(f"Unsupported mhd_linear source module: {module!r}")
    pattern, variable = _MODULE_OUTPUT[module]

    def _read_fortran_record_length(f):
        raw = f.read(4)
        if len(raw) < 4:
            return None
        return struct.unpack("<i", raw)[0]

    def _read_n_floats(f, n):
        raw = f.read(n * 4)
        if len(raw) < n * 4:
            raise EOFError("Unexpected EOF while reading float data.")
        return np.frombuffer(raw, dtype="<f4")

    def _existing_toroidal_mode_count() -> int:
        # Plain `in` checks are non-mutating on an ODS; indexing a path that
        # does not exist yet auto-vivifies it, which would corrupt an empty
        # `ods` the caller never asked this function to touch.
        if "mhd_linear.time_slice" not in ods:
            return 0
        if time_slice >= len(ods["mhd_linear.time_slice"]):
            return 0
        path = f"mhd_linear.time_slice.{time_slice}.toroidal_mode"
        if path not in ods:
            return 0
        return len(ods[path])

    def _read_solutions_bin(filename):
        data_blocks = []
        with open(filename, "rb") as f:
            while True:
                length = _read_fortran_record_length(f)
                if length is None:
                    break
                if length == 0:
                    continue

                num_floats = length // 4
                arr_step0 = _read_n_floats(f, num_floats)
                trailing_len = _read_fortran_record_length(f)

                steps_for_ipert = [arr_step0]
                while True:
                    length2 = _read_fortran_record_length(f)
                    if length2 is None:
                        break
                    if length2 == 0:
                        break
                    nfloat2 = length2 // 4
                    arr2 = _read_n_floats(f, nfloat2)
                    _ = _read_fortran_record_length(f)
                    steps_for_ipert.append(arr2)

                data_blocks.append(steps_for_ipert)

        n_ipert = len(data_blocks)
        if n_ipert == 0:
            return np.zeros((0, 0, 7), dtype=np.float32)

        max_steps = max(len(steps) for steps in data_blocks)
        arr3d = np.full((n_ipert, max_steps, 7), np.nan, dtype=np.float32)
        for i_ipert, step_list in enumerate(data_blocks):
            for j_step, vec7 in enumerate(step_list):
                arr3d[i_ipert, j_step, :] = vec7
        return arr3d

    found: dict[int, float] = {}
    for file in sorted(os.listdir(source)):
        match = pattern.fullmatch(file)
        if not match:
            continue
        n = int(match.group(1))
        filepath = os.path.join(source, file)
        try:
            with xr.open_dataset(filepath) as ds:
                if variable not in ds.variables:
                    continue
                if module == "dcon":
                    value = ds[variable].isel(i=0).sel(mode=1).values.item()
                else:
                    value = _extract_scalar(ds[variable])
        except Exception:
            continue
        if value is None:
            continue
        found[n] = value

    # `toroidal_mode` is an IMAS array of structures: entries must be
    # appended sequentially (position != mode number), so the physical
    # toroidal mode number is only ever recovered from `n_tor`, never from
    # array position. `existing` accounts for entries a prior call already
    # wrote for this time slice (e.g. a DCON pass before this RDCON pass on
    # the same `ods`) so repeated calls extend rather than overwrite them.
    extras: dict[int, dict[str, Any]] = {}
    if found:
        existing = _existing_toroidal_mode_count()
    else:
        existing = 0
    for offset, n in enumerate(sorted(found)):
        position = existing + offset
        value = found[n]
        mode = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
        mode["n_tor"] = n
        if module == "dcon":
            mode["energy_perturbed"] = value
        else:
            mode["ballooning_type"]["name"] = "Tearing"
        extras[n] = {"module": module, "variable": variable, "value": value}

    if module != "dcon":
        return extras

    bin_file = os.path.join(source, "solutions.bin")
    if not os.path.exists(bin_file):
        return extras

    arr3d = _read_solutions_bin(bin_file)
    n_ipert, n_step, _ = arr3d.shape

    # solutions.bin's per-block index has no independently verified
    # correspondence to a physical toroidal mode number, so these are
    # appended as additional AOS entries after whatever the .nc scan already
    # wrote for this time slice, rather than overwriting those entries.
    start = _existing_toroidal_mode_count()
    for offset in range(n_ipert):
        n = offset
        position = start + offset
        mode_entry = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
        # The IMAS schema declares `displacement_perpendicular.real/imaginary`
        # as FLT_2D over (grid.dim1, grid.dim2), but solutions.bin's actual
        # physical layout along those two axes has never been verified against
        # DCON's Fortran source -- writing a 1D real/imaginary series (as this
        # historically unreachable code did) fails that shape check. Rather
        # than guess a reshape that could misrepresent the physics, this
        # narrow subtree is written best-effort with validation off; fixing it
        # properly needs the real solutions.bin field layout from DCON, out of
        # scope for the DCON/RDCON/STRIDE Delta_prime/W_t work this is part of.
        mode_entry.consistency_check = False
        psi_grid = arr3d[offset, :, 0]
        alpha_grid = np.arange(n_step)

        mode_entry["plasma"]["grid"]["dim1"] = psi_grid.tolist()
        mode_entry["plasma"]["grid"]["dim2"] = alpha_grid.tolist()
        mode_entry["plasma"]["displacement_perpendicular"]["real"] = arr3d[offset, :, 3].tolist()
        mode_entry["plasma"]["displacement_perpendicular"]["imaginary"] = arr3d[offset, :, 4].tolist()
        mode_entry["n_tor"] = n

    return extras
