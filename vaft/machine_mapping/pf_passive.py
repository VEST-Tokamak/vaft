"""`pf_passive` IDS mapping helpers."""

from __future__ import annotations

import copy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omas import ODS

from vaft.data.resources import data_path
from vaft.machine_mapping.static_geometry import load_static_ods


DEFAULT_STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")
# Kept as an import-level compatibility name for callers that referenced it.
DEFAULT_REFERENCE_ODS = DEFAULT_STATIC_GEOMETRY
#: The fifteen conductors wall 2409 adds, with their coupling rows
#: (`workflow/em_coupling/import_wall_2409.py`, issue #956).
DEFAULT_WALL_2409_ADDITIONS = data_path("geometry/VEST_passive_wall_2409.npz")
#: First shot on wall 2409: 43016 is 2024-07-05 and 43017 is 2024-07-15, the
#: access in which the fifteen SUS316LN elements at Z = -1.164 m went in.
WALL_GEOMETRY_2409_FIRST_SHOT = 43017


def wall_geometry_version_for_shot(shot: int | None) -> str:
    """Return the passive-wall version for *shot*: ``"1512"`` or ``"2409"``.

    Like :func:`vaft.machine_mapping.pf_active.pf_geometry_version_for_shot`,
    the one place the boundary lives, so the passive loops and the coupling
    rows cannot select different walls. ``None`` keeps the historical 1512.
    """
    if shot is not None and int(shot) >= WALL_GEOMETRY_2409_FIRST_SHOT:
        return "2409"
    return "1512"


@lru_cache(maxsize=1)
def load_wall_2409_additions() -> dict[str, Any]:
    """The wall 2409 additions asset: loops, coupling rows and provenance."""
    with np.load(DEFAULT_WALL_2409_ADDITIONS) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    for key, value in arrays.items():
        if value.dtype.kind == "f":
            value.setflags(write=False)
    return {
        "loops": tuple(json.loads(str(arrays["loops"]))),
        "mutual_passive_passive_rows": arrays["mutual_passive_passive_rows"],
        "mutual_passive_active": {
            "1906": arrays["mutual_passive_active_1906"],
            "2507": arrays["mutual_passive_active_2507"],
        },
        "provenance": json.loads(str(arrays["provenance"])),
    }


def append_wall_loops(ods: Any, version: str) -> int:
    """Append *version*'s additional loops to ``ods["pf_passive.loop"]``.

    Returns how many were added: 0 for the base wall, and 0 when *ods* already
    carries them (a wall 2409 product passed back in as a reference).
    """
    if version == "1512":
        return 0
    if version != "2409":
        raise ValueError(f"unknown VEST passive wall version {version!r}")
    start = len(ods["pf_passive.loop"])
    loops = load_wall_2409_additions()["loops"]
    names = {loop["name"] for loop in loops}
    present = sum(str(ods.get(f"pf_passive.loop.{index}.name")) in names for index in range(start))
    if present == len(loops):
        return 0
    if present:
        raise ValueError(
            f"pf_passive carries {present} of the {len(loops)} wall {version} loops; "
            "it is neither the base wall nor wall 2409"
        )
    for offset, loop in enumerate(loops):
        base = f"pf_passive.loop.{start + offset}"
        element = loop["element"][0]
        ods[f"{base}.name"] = loop["name"]
        ods[f"{base}.resistance"] = float(loop["resistance"])
        ods[f"{base}.resistivity"] = float(loop["resistivity"])
        ods[f"{base}.element.0.identifier"] = element["identifier"]
        ods[f"{base}.element.0.area"] = float(element["area"])
        ods[f"{base}.element.0.turns_with_sign"] = float(element["turns_with_sign"])
        ods[f"{base}.element.0.geometry.geometry_type"] = int(element["geometry"]["geometry_type"])
        ods[f"{base}.element.0.geometry.outline.r"] = np.asarray(element["geometry"]["outline"]["r"], dtype=float)
        ods[f"{base}.element.0.geometry.outline.z"] = np.asarray(element["geometry"]["outline"]["z"], dtype=float)
    return len(loops)


def _resolve_reference(source: str | Path | None, options: Optional[dict]) -> Path:
    if options is None:
        options = {}
    candidate = source or options.get("reference_ods") or options.get("source") or DEFAULT_REFERENCE_ODS
    path = Path(candidate).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"pf_passive reference ODS not found: {path}")
    return path


def pf_passive(
    ods: ODS,
    source: str | Path | None = None,
    options: Optional[dict] = None,
    *,
    shot: int | None = None,
) -> None:
    """Populate canonical VEST passive-loop geometry from a static asset.

    *shot* selects the wall (:func:`wall_geometry_version_for_shot`): from
    43017 the fifteen wall 2409 conductors follow the 950 base loops, in the
    row order of :func:`vaft.machine_mapping.em_coupling.em_coupling`.

    An explicitly supplied legacy ODS may contain historical eddy-current time
    traces. Those traces are removed so target-shot currents are computed fresh.
    """
    if options is None:
        options = {}
    if shot is None:
        shot = options.get("shot")
    reference_path = _resolve_reference(source, options)
    reference = load_static_ods(reference_path)
    if "pf_passive" not in reference:
        raise KeyError(f"Reference ODS has no pf_passive IDS: {reference_path}")

    ods["pf_passive"] = copy.deepcopy(reference["pf_passive"])
    try:
        del ods["pf_passive.time"]
    except Exception:
        pass
    for loop_index in range(len(ods["pf_passive.loop"])):
        try:
            del ods[f"pf_passive.loop.{loop_index}.current"]
        except Exception:
            pass
    version = wall_geometry_version_for_shot(shot)
    append_wall_loops(ods, version)
    parameters = str(ods["pf_passive.code.parameters"]) if "pf_passive.code.parameters" in ods else ""
    ods["pf_passive.code.parameters"] = parameters + f"wall_geometry={version}\n"


__all__ = [
    "DEFAULT_REFERENCE_ODS",
    "DEFAULT_STATIC_GEOMETRY",
    "DEFAULT_WALL_2409_ADDITIONS",
    "WALL_GEOMETRY_2409_FIRST_SHOT",
    "append_wall_loops",
    "load_wall_2409_additions",
    "pf_passive",
    "wall_geometry_version_for_shot",
]
