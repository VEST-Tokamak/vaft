"""Memory-mapped native CGYRO fields, audited against pygacode b493397.

No IMAS projection or plotting imports. Axes are (radial slot, saved theta, ky, time).
The first radial slot in ordinary runs is special, not a Fourier coefficient.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .time import time_coordinate_metadata
from typing import Any

import numpy as np

from .outputs import (
    _ascii,
    _monotonic_records,
    _version,
    parse_equilibrium,
    parse_grids,
)

__all__ = ["FieldHistory", "read_field_history"]

_FILES = {"phi": "phi", "a_parallel": "apar", "b_parallel": "bpar"}


@dataclass(frozen=True)
class FieldHistory:
    """Lazy native field with explicit selected raw records and saved theta grid."""

    raw: np.ndarray
    records: np.ndarray
    theta_indices: np.ndarray
    time: np.ndarray
    theta: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    radial_start: int
    metadata: Mapping[str, Any]

    def read(self, time_slice=slice(None), theta_index=0, *, physical_radial=True):
        """Read (kx, ky, selected time) at one saved theta; no interpolated angles."""
        index = int(self.theta_indices[theta_index])
        start = self.radial_start if physical_radial else 0
        return np.asarray(self.raw[start:, index, :, :][..., self.records[time_slice]])


def read_field_history(
    directory, *, field="phi", time_window=None, stride=1, theta_indices=None
) -> FieldHistory:
    """Open a native field without loading flux histories or the whole field.

    Keep the first strictly increasing restart records, as CgyroOutputs does.
    Incomplete trailing bytes are excluded and reported. Missing fields raise
    FileNotFoundError; invalid grids/times are never silently truncated internally.
    """
    directory = Path(directory)
    if field not in _FILES:
        raise ValueError(f"field must be one of {tuple(_FILES)}")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    path = directory / f"bin.cgyro.kxky_{_FILES[field]}"
    if not path.is_file():
        raise FileNotFoundError(f"unavailable native field: {path}")
    grid = parse_grids(directory / "out.cgyro.grids")
    if grid is None:
        raise ValueError("missing/invalid native grid")
    if grid["n_field"] < list(_FILES).index(field) + 1:
        raise ValueError("requested field is not part of the recorded field model")
    nr, nth, nk = (int(grid[k]) for k in ("n_radial", "theta_plot", "n_n"))
    if min(nr, nth, nk) < 1 or len(grid["ky"]) != nk or len(grid["p"]) != nr:
        raise ValueError("incomplete native grid")
    eq = parse_equilibrium(directory / "out.cgyro.equilibrium", grid["n_species"])
    if eq is None or eq.get("hiprec_flag") not in (0, 1):
        raise ValueError("field precision is not recorded")
    dtype = np.dtype(np.complex128 if eq["hiprec_flag"] else np.complex64)
    record_bytes = nr * nth * nk * dtype.itemsize
    steps, tail = divmod(path.stat().st_size, record_bytes)
    table = _ascii(directory / "out.cgyro.time")
    if table is None or table.size < 4 or table.size % 4:
        raise ValueError("missing or incomplete time table")
    times = table.reshape(-1, 4)[:, 0]
    if not np.isfinite(times).all():
        raise ValueError("non-finite simulation time")
    keep = _monotonic_records(times)
    records = np.asarray([i for i in keep if i < steps], dtype=int)
    if time_window is not None:
        lo, hi = map(float, time_window)
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("invalid time window")
        records = records[(times[records] >= lo) & (times[records] <= hi)]
    if records.size:
        last_record = records[-1]
        records = records[::stride]
        if records[-1] != last_record:
            records = np.append(records, last_record)
    if not records.size:
        raise ValueError("no complete field records in selected window")
    ti = (
        np.arange(nth)
        if theta_indices is None
        else np.atleast_1d(theta_indices).astype(int)
    )
    if not ti.size or np.any(ti < 0) or np.any(ti >= nth):
        raise ValueError("theta selection outside saved angles")
    if nth == 1:
        angles = np.zeros(1)
    else:
        if grid["n_theta"] % nth:
            raise ValueError("saved theta layout is ambiguous")
        angles = grid["theta"][:: grid["n_theta"] // nth]
    start = 0 if grid["p"][0] > 0 else 1
    native_length = float(grid["length"])
    length = abs(native_length)
    if length <= 0:
        raise ValueError("nonpositive radial box length")
    raw = np.memmap(path, mode="r", dtype=dtype, shape=(nr, nth, nk, steps), order="F")
    from ..result_identity import digest, read_native_input

    inputs, input_path = read_native_input(directory, "CGYRO")
    input_hash = (
        hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path else None
    )
    record_path = directory / "record.json"
    run_record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    metadata = {
        "source": str(path.resolve()),
        "source_run_id": run_record.get("run_identity")
        or digest(
            {
                "input_sha256": input_hash,
                "version": _version(directory / "out.cgyro.version"),
            }
        ),
        "input_sha256": input_hash,
        "field_output_settings": {
            k: inputs.get(k)
            for k in ("FIELD_PRINT_FLAG", "PRINT_STEP", "THETA_PLOT", "HIPREC_FLAG")
        },
        "field": field,
        "version": _version(directory / "out.cgyro.version"),
        "normalization": "native CGYRO",
        "time_unit": "a/c_s",
        "time_coordinate": "gyrokinetic_normalized_time",
        "time_normalization": time_coordinate_metadata(
            SimpleNamespace(equilibrium=eq, directory=str(directory))
        ),
        "axis_order": ["radial_slot", "theta", "ky", "time"],
        "radial_start": start,
        "length_rho_s": length,
        "native_signed_length": native_length,
        "display_orientation": "pygacode plot_fluct: +p*x - n*y; positive box lengths",
        "partial_tail_bytes": tail,
        "raw_field_records": steps,
        "raw_time_records": len(times),
        "restart_records_excluded": len(times) - len(keep),
        "unpaired_field_records": max(0, steps - len(times)),
        "missing_trailing_field_records": max(0, len(times) - steps),
        "selected_raw_records": records.tolist(),
        "theta_indices": ti.tolist(),
        "fourier_convention": "Re sum phi(p,n) exp(+i kx x -i |ky| y); n=0 half weight (pygacode)",
        "zonal_definition": "ky=0 at a saved theta, not a flux-surface average",
    }
    return FieldHistory(
        raw,
        records,
        ti,
        times[records],
        angles[ti],
        2 * np.pi * np.asarray(grid["p"][start:]) / length,
        np.abs(np.asarray(grid["ky"])),
        start,
        metadata,
    )
