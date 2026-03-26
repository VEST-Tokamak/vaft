"""
`summary` IDS mapping helpers.

Schema reference:
https://gafusion.github.io/omas/schema.html
"""

from __future__ import annotations

from typing import Optional

from omas import ODS


def summary(ods: ODS, source: str, options: Optional[dict] = None) -> None:
    """
    Add experiment summary information to the ODS structure.
    """
    if options is None:
        options = {}

    ods["summary.shot"] = int(source) if options.get("source_type") == "shot" else None

    if "start_time" in options:
        ods["summary.time.start"] = options["start_time"]
    if "end_time" in options:
        ods["summary.time.end"] = options["end_time"]

    if "plasma_current" in options:
        ods["summary.plasma_current.maximum"] = options["plasma_current"]
    if "toroidal_field" in options:
        ods["summary.toroidal_field.maximum"] = options["toroidal_field"]

    if "additional_info" in options:
        for key, value in options["additional_info"].items():
            ods[f"summary.{key}"] = value

    ods["summary.status"] = options.get("status", "completed")
    ods["summary.comment"] = options.get("comment", "VEST experiment data")

