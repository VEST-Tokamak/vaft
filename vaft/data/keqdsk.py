"""Standalone parser for EFIT k-file Fortran namelists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import f90nml
import numpy as np


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key).lower(): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class KEQDSK:
    """Parsed EFIT k-file, retaining every submitted namelist value."""

    namelists: Mapping[str, Mapping[str, Any]]
    source: Path | None = None

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        return self.namelists[key.lower()]

    def to_omas(self, ods: Any = None, *, time_index: int = 0) -> Any:
        """Store the exact submitted namelists as equilibrium code parameters."""
        from omas import ODS

        if ods is None:
            ods = ODS()
        write_namelists_to_ods(ods, self.namelists, time_index=time_index)
        return ods


def write_namelists_to_ods(
    ods: Any, namelists: Mapping[str, Mapping[str, Any]], *, time_index: int = 0
) -> Any:
    """Write parsed Fortran namelists to ``equilibrium.code.parameters``.

    Array-valued entries are stored as NumPy arrays rather than lists: omas'
    code-parameters encoder recurses into a list and then tries to reindex it
    by string key, which raises on save to HDF5 and netCDF. It handles ndarrays
    (see ``omas_utils.recursive_encoder``).
    """
    root = f"equilibrium.code.parameters.time_slice.{time_index}"
    for namelist, values in namelists.items():
        for name, value in values.items():
            ods[f"{root}.{namelist}.{name}"] = _as_stored(value)
    return ods


def _as_stored(value: Any) -> Any:
    """Shape one namelist value for omas.

    Array entries arrive as Python lists, which omas' code-parameters encoder
    cannot serialize -- it recurses into a list and then reindexes it by string
    key. NumPy arrays it handles. Two shapes f90nml can produce resist the
    conversion: a sparse assignment (``A(3) = 1.0``) pads with ``None``, and a
    sparse multi-dimensional one is ragged, which ``np.asarray`` rejects
    outright. Neither is worth losing the rest of the namelist over, so an
    unconvertible value is left exactly as it came.
    """
    if not isinstance(value, (list, tuple)):
        return value
    try:
        array = np.asarray(value)
    except ValueError:
        return value  # ragged; nothing sensible to convert it to
    if array.dtype == object:
        return value
    return array


def read_keqdsk(path: str | Path) -> KEQDSK:
    """Read all namelists from an EFIT k-file using Fortran semantics."""
    source = Path(path).expanduser()
    parsed = f90nml.read(str(source))
    return KEQDSK(_plain(parsed), source=source)


__all__ = ["KEQDSK", "read_keqdsk", "write_namelists_to_ods"]
