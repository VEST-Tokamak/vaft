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
        root = f"equilibrium.code.parameters.time_slice.{time_index}"
        for namelist, values in self.namelists.items():
            for name, value in values.items():
                ods[f"{root}.{namelist}.{name}"] = value
        return ods


def read_keqdsk(path: str | Path) -> KEQDSK:
    """Read all namelists from an EFIT k-file using Fortran semantics."""
    source = Path(path).expanduser()
    parsed = f90nml.read(str(source))
    return KEQDSK(_plain(parsed), source=source)


__all__ = ["KEQDSK", "read_keqdsk"]
