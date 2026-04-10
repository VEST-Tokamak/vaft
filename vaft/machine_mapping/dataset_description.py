"""Canonical dataset_description builders integrated under machine_mapping."""

from __future__ import annotations

import os
from typing import Any

from .utils import set_path


def vfit_dataset_description(
    ods: Any,
    shot: int,
    run: int,
    machine: str = "VEST",
    pulse_type: str = "pulse",
    user: str | None = None,
) -> None:
    """Populate canonical dataset metadata for a VEST pulse/run pair."""
    username = user or os.environ.get("USER", "unknown")
    set_path(ods, "dataset_description.ids_properties.comment", "Wall from vfit_machine_description")
    set_path(ods, "dataset_description.ids_properties.homogeneous_time", 2)
    set_path(ods, "dataset_description.data_entry.machine", machine)
    set_path(ods, "dataset_description.data_entry.pulse", shot)
    set_path(ods, "dataset_description.data_entry.pulse_type", pulse_type)
    set_path(ods, "dataset_description.data_entry.run", run)
    set_path(ods, "dataset_description.data_entry.user", username)


def dataset_description(
    ods: Any,
    source: int,
    options: dict | None = None,
) -> None:
    """Compatibility wrapper that maps legacy calls onto the canonical builder."""
    if options is None:
        options = {}

    vfit_dataset_description(
        ods,
        shot=int(source),
        run=int(options.get("run", 0)),
        machine=options.get("machine", "VEST"),
        pulse_type=options.get("pulse_type", "pulse"),
        user=options.get("user"),
    )


def dataset_description_from_raw_database(
    ods: Any,
    shot: int,
    options: dict | None = None,
) -> None:
    if options is None:
        options = {}
    if "description" not in options:
        options["description"] = "VEST dataset imported from raw database"
    options.setdefault("source_type", "shot")
    dataset_description(ods, shot, options)


__all__ = [
    "dataset_description",
    "dataset_description_from_raw_database",
    "vfit_dataset_description",
]
