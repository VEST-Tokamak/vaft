"""Shared loader for compact VEST static-geometry assets."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from omas import ODS, load_omas_json


def load_static_ods(path: str | Path) -> ODS:
    """Load plain or gzip-compressed OMAS JSON without temporary files."""
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        ods = ODS(consistency_check=False)
        ods.from_structure(payload)
        return ods
    return load_omas_json(str(source), consistency_check=False)


__all__ = ["load_static_ods"]
