"""Static VEST first-wall mapping."""

from __future__ import annotations

import copy
from pathlib import Path

from omas import ODS

from vaft.data.resources import data_path
from vaft.machine_mapping.static_geometry import load_static_ods


DEFAULT_STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")


def wall(ods: ODS, source: str | Path | None = None) -> None:
    """Populate the VEST limiter wall from the packaged static geometry."""
    path = Path(source or DEFAULT_STATIC_GEOMETRY).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"VEST wall geometry not found: {path}")
    reference = load_static_ods(path)
    if "wall" not in reference:
        raise KeyError(f"Static geometry has no wall IDS: {path}")
    ods["wall"] = copy.deepcopy(reference["wall"])
    if "wall.time" in ods:
        del ods["wall.time"]


__all__ = ["DEFAULT_STATIC_GEOMETRY", "wall"]
