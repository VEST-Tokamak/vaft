"""`pf_passive` IDS mapping helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

from omas import ODS

from vaft.data.resources import data_path
from vaft.machine_mapping.static_geometry import load_static_ods


DEFAULT_STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")
# Kept as an import-level compatibility name for callers that referenced it.
DEFAULT_REFERENCE_ODS = DEFAULT_STATIC_GEOMETRY


def _resolve_reference(source: str | Path | None, options: Optional[dict]) -> Path:
    if options is None:
        options = {}
    candidate = source or options.get("reference_ods") or options.get("source") or DEFAULT_REFERENCE_ODS
    path = Path(candidate).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"pf_passive reference ODS not found: {path}")
    return path


def pf_passive(ods: ODS, source: str | Path | None = None, options: Optional[dict] = None) -> None:
    """Populate canonical VEST passive-loop geometry from a static asset.

    An explicitly supplied legacy ODS may contain historical eddy-current time
    traces. Those traces are removed so target-shot currents are computed fresh.
    """
    if options is None:
        options = {}
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


__all__ = ["pf_passive", "DEFAULT_REFERENCE_ODS", "DEFAULT_STATIC_GEOMETRY"]
