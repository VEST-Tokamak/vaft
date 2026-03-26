"""
`mhd_linear` IDS mapping helpers.

Schema reference:
https://gafusion.github.io/omas/schema.html
"""

from __future__ import annotations

import os
import re
import struct
from typing import Optional

import numpy as np
import xarray as xr
from omas import ODS


def mhd_linear(ods: ODS, source: str, options: Optional[dict] = None) -> None:
    if options is None:
        options = {}

    time_slice = options.get("time_slice", 0)

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

    for file in os.listdir(source):
        match = re.match(r"dcon_output_n(\d+)\.nc", file)
        if not match:
            continue
        n = int(match.group(1))
        filepath = os.path.join(source, file)
        try:
            ds = xr.open_dataset(filepath)
            W_t = ds["W_t_eigenvalue"].isel(i=0).sel(mode=1).values.item()
            mode_index = n
            ods["mhd_linear"][time_slice][mode_index]["n"] = n
            ods["mhd_linear"][time_slice][mode_index]["energy_perturbed"] = W_t
        except Exception:
            continue

    bin_file = os.path.join(source, "solutions.bin")
    if not os.path.exists(bin_file):
        return

    arr3d = _read_solutions_bin(bin_file)
    n_ipert, n_step, _ = arr3d.shape

    while len(ods["mhd_linear"]) <= time_slice:
        ods["mhd_linear"].append([])

    for n in range(n_ipert):
        mode_entry = {"plasma": {}}
        psi_grid = arr3d[n, :, 0]
        alpha_grid = np.arange(n_step)

        mode_entry["plasma"]["grid"] = {"dim1": psi_grid.tolist(), "dim2": alpha_grid.tolist()}
        mode_entry["plasma"]["displacement_perpendicular"] = {
            "real": arr3d[n, :, 3].tolist(),
            "imaginary": arr3d[n, :, 4].tolist(),
        }
        mode_entry["n"] = n
        ods["mhd_linear"][time_slice].append(mode_entry)

