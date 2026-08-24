"""Static VEST first-wall mapping."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from omas import ODS

from vaft.data.resources import data_path
from vaft.machine_mapping.static_geometry import load_static_ods

from .utils import set_path


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

    # `wall` has no dynamic counterpart in VAFT, so per the DD's
    # `homogeneous_time` rule ("if only constant or static nodes are filled,
    # homogeneous_time must be set to 2") this IDS is always independent.
    set_path(ods, "wall.ids_properties.homogeneous_time", 2)

    for description_index in range(len(ods["wall.description_2d"])):
        entry = ods[f"wall.description_2d.{description_index}"]
        # Membership (`in`) is used instead of `path_exists()` here: querying
        # a missing ODS branch by index auto-vivifies an empty placeholder as
        # a side effect, which then fails a consistency-checked reload even
        # though nothing was ever written to it.
        if "vessel" not in entry and "type" not in entry:
            # No vessel description is provided; only limiter geometry is filled.
            entry["type.index"] = 1
            entry["type.name"] = "multiple_units_no_vessel"
            entry["type.description"] = "Limiter geometry only; no vessel description"
        if "limiter" not in entry:
            continue
        for unit_index in range(len(entry["limiter.unit"])):
            unit = entry[f"limiter.unit.{unit_index}"]
            r = np.asarray(unit["outline.r"], dtype=float)
            z = np.asarray(unit["outline.z"], dtype=float)
            unit["closed"] = int(r.size > 1 and r[0] == r[-1] and z[0] == z[-1])
            if "name" not in unit:
                unit["name"] = "VEST first wall"


__all__ = ["DEFAULT_STATIC_GEOMETRY", "wall"]
