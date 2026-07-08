"""`em_coupling` IDS mapping helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from omas import load_omas_json

from vaft.data.resources import data_path


DEFAULT_REFERENCE_ODS = data_path("39915.json")


def _resolve_reference(source: str | Path | None, options: dict | None) -> Path:
    if options is None:
        options = {}
    candidate = source or options.get("reference_ods") or options.get("source") or DEFAULT_REFERENCE_ODS
    path = Path(candidate).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"em_coupling reference ODS not found: {path}")
    return path


def em_coupling(ods: Any, source: str | Path | None = None, options: dict | None = None) -> None:
    """Populate VEST mutual inductance matrices from a packaged reference ODS."""
    reference_path = _resolve_reference(source, options)
    reference = load_omas_json(str(reference_path), consistency_check=False)
    if "em_coupling" not in reference:
        raise KeyError(f"Reference ODS has no em_coupling IDS: {reference_path}")
    ods["em_coupling"] = copy.deepcopy(reference["em_coupling"])


def calculate_em_coupling_from_raw_database(ods: Any, options: dict | None = None) -> None:
    em_coupling(ods, options=options)


__all__ = ["calculate_em_coupling_from_raw_database", "em_coupling", "DEFAULT_REFERENCE_ODS"]
