"""Canonical dataset_description builders integrated under machine_mapping."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from .utils import set_path

logger = logging.getLogger(__name__)


def vfit_dataset_description(
    ods: Any,
    shot: int,
    run: int,
    machine: str = "VEST",
    pulse_type: str = "pulse",
    user: str | None = None,
    description: str | None = None,
    pulse_datetime: datetime | None = None,
) -> None:
    """Populate canonical dataset metadata for a VEST pulse/run pair.

    ``pulse_datetime`` is the pulse's real acquisition timestamp (e.g. from
    the SQL ``shot`` table's ``recordDateTime``, via
    ``vaft.database.raw.date_from_shot``/``list_shots``) and is optional:
    most callers build this ODS from an already-archived source with no live
    database access, and `dataset_description.pulse_time_begin` is left
    unset in that case rather than guessed. Naive datetimes (no tzinfo) are
    stored as-is via ISO 8601 formatting; no epoch-seconds conversion is
    attempted since that would require assuming a timezone the SQL column
    does not record.
    """
    username = user or os.environ.get("USER", "unknown")
    comment = description or "VEST dataset"
    set_path(ods, "dataset_description.ids_properties.comment", comment)
    set_path(ods, "dataset_description.ids_properties.homogeneous_time", 2)
    set_path(ods, "dataset_description.data_entry.machine", machine)
    set_path(ods, "dataset_description.data_entry.pulse", shot)
    set_path(ods, "dataset_description.data_entry.pulse_type", pulse_type)
    set_path(ods, "dataset_description.data_entry.run", run)
    set_path(ods, "dataset_description.data_entry.user", username)
    if pulse_datetime is not None:
        set_path(ods, "dataset_description.pulse_time_begin", pulse_datetime.isoformat())


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
        pulse_type=options.get("pulse_type", options.get("source_type", "pulse")),
        user=options.get("user"),
        description=options.get("description"),
        pulse_datetime=options.get("pulse_datetime"),
    )


def dataset_description_from_raw_database(
    ods: Any,
    shot: int,
    options: dict | None = None,
) -> None:
    """Build ``dataset_description`` and, when live SQL access is available,
    fill ``pulse_time_begin`` from the authoritative ``shot`` table.

    Unlike the generic ``dataset_description()``, this entry point is meant
    for callers with real database access: it looks up the shot's actual
    ``recordDateTime`` via ``vaft.database.raw.date_from_shot`` unless the
    caller already supplied ``options["pulse_datetime"]``. A missing or
    unreachable database is not an error here -- the pulse timestamp is
    simply left unset, same as calling ``dataset_description()`` directly.
    """
    if options is None:
        options = {}
    if "description" not in options:
        options["description"] = "VEST dataset imported from raw database"
    options.setdefault("source_type", "shot")
    if options.get("pulse_datetime") is None:
        try:
            from vaft.database.raw import date_from_shot

            _, pulse_datetime = date_from_shot(int(shot))
        except Exception as exc:
            logger.info("Could not resolve pulse_datetime for shot %s from SQL: %s", shot, exc)
        else:
            if pulse_datetime is not None:
                options["pulse_datetime"] = pulse_datetime
    dataset_description(ods, shot, options)


__all__ = [
    "dataset_description",
    "dataset_description_from_raw_database",
    "vfit_dataset_description",
]
