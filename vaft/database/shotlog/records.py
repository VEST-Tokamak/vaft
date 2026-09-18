"""Per-shot ShotLog records and the effective trigger table.

A session document is organised the way the workbook is -- by day and run
group. Everything downstream asks about one shot, so this module inverts the
sessions into one record per shot, which is what FileDB stores under
``legacy/shotlog/{shot}/metadata/shotlog.json`` and what the ``pulse_schedule``
mapping reads.

Two rules shape the result.

Carry-forward
    Within one session a diagnostic trigger stays in force until a later card
    changes it; a blank trigger cell means "unchanged", not "off". This is the
    rule the packaged trigger table has always been built with, and SXR time
    alignment depends on it. It applies to diagnostic triggers only: heating
    and gas windows are copied onto every card, so they are taken as written.
    Every effective entry names the ``source_shot`` whose card set it, so an
    inherited value is never mistaken for one logged on the shot itself.

One primary occurrence
    A shot can appear in more than one workbook (an in-progress copy and the
    closed month, a sheet duplicated across months). The record's content comes
    from one occurrence -- a workbook whose ``#first-last`` span covers the
    shot, else the earliest -- and every occurrence is listed. When they
    disagree on timing the record says so (``ambiguous``) rather than merging.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

from .batch import SourceWorkbook

RECORD_VERSION = 1

#: The ShotLog's timing controller counts from 200 ms before the DAQ trigger,
#: so a logged ``490-510`` is 290-310 ms on the DAQ clock vaft stores (the
#: ``'daq'`` time convention). Applied exactly once, where a ShotLog time
#: leaves this package; stored records keep the logged numbers.
DAQ_OFFSET_MS = -200

#: Labels the trigger table files under another key. The primary soft X-ray
#: array is logged as ``SXR1`` on the modern card, and
#: ``machine_mapping.soft_x_rays`` reads ``SXR`` -- the packaged table has
#: always carried it under that name. A card that logs a bare ``SXR`` wins.
TABLE_ALIASES = {"SXR1": "SXR"}

#: Systems whose triggers carry forward within a session.
CARRY_FORWARD_SYSTEMS = frozenset({"diagnostic"})


def _explicit(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("planned_configuration", {}).get("timing", []))


def effective_timing_by_session(dataset: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """``{shot: {"timing": [...], "probe_positions": {...}}}`` for one session.

    Timing entries gain ``source_shot`` and ``inherited``; probe positions,
    which the packaged table has always carried forward too, gain
    ``source_shot``.
    """
    active: dict[tuple[str, str], dict[str, Any]] = {}
    positions: dict[str, dict[str, Any]] = {}
    result: dict[int, dict[str, Any]] = {}
    for group in dataset.get("run_groups", []):
        for shot_record in group.get("shots", []):
            shot = shot_record["shot"]
            entries: dict[tuple[str, str], dict[str, Any]] = {}
            for entry in _explicit(shot_record):
                key = (entry["system"], entry["identifier"])
                if entry["system"] in CARRY_FORWARD_SYSTEMS and entry.get("window_ms") is None:
                    # An unreadable value neither sets nor clears the carried one.
                    continue
                stamped = {**deepcopy(entry), "source_shot": shot, "inherited": False}
                entries[key] = stamped
                if entry["system"] in CARRY_FORWARD_SYSTEMS:
                    active[key] = stamped
            for key, carried in active.items():
                if key not in entries:
                    entries[key] = {**deepcopy(carried), "inherited": carried["source_shot"] != shot}
            planned = shot_record.get("planned_configuration", {})
            for label, position in planned.get("probe_positions", {}).items():
                if position.get("positions_m"):
                    positions[label] = {**deepcopy(position), "source_shot": shot}
            result[shot] = {
                "timing": sorted(entries.values(), key=lambda item: (item["system"], item["identifier"])),
                "probe_positions": deepcopy(positions),
            }
    return result


def _occurrence(source: SourceWorkbook, dataset: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    provenance = dataset["provenance"]
    return {
        "session_id": dataset["experiment_session"]["id"],
        "experiment_date": dataset["experiment_session"].get("experiment_date"),
        "workbook": provenance["source_file"],
        "sha256": provenance["source_sha256"],
        "sheet": provenance["sheet_name"],
        "run_group_id": group["id"],
        "covers_shot": None,  # filled per shot
        "_source": source,
    }


def build_shot_records(
    sessions: Iterable[tuple[SourceWorkbook, dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Invert converted sessions into ``{shot: record}``."""
    candidates: dict[int, list[tuple]] = defaultdict(list)
    for source, dataset in sessions:
        effective = effective_timing_by_session(dataset)
        for group in dataset["run_groups"]:
            for shot_record in group["shots"]:
                shot = shot_record["shot"]
                occurrence = _occurrence(source, dataset, group)
                occurrence["covers_shot"] = source.covers(shot)
                candidates[shot].append((occurrence, dataset, group, shot_record,
                                         effective.get(shot, {"timing": [], "probe_positions": {}})))

    records: dict[int, dict[str, Any]] = {}
    for shot, items in candidates.items():
        items.sort(key=lambda item: (
            not item[0]["covers_shot"],
            item[0]["experiment_date"] or "9999",
            item[0]["workbook"],
            item[0]["sheet"],
        ))
        occurrence, dataset, group, shot_record, effective = items[0]
        timings = {_timing_signature(item[3]) for item in items}
        records[shot] = {
            "record_version": RECORD_VERSION,
            "shot": shot,
            "clock": {
                "logged_on": "shotlog_timing_controller",
                "daq_offset_ms": DAQ_OFFSET_MS,
            },
            "schema_version": dataset["schema_version"],
            "schema_revision": dataset.get("schema_revision"),
            "source": {
                "workbook": occurrence["workbook"],
                "sha256": occurrence["sha256"],
                "sheet": occurrence["sheet"],
            },
            "session": deepcopy(dataset["experiment_session"]),
            "run_group": {"id": group["id"], "title": group.get("title", {}).get("raw")},
            "status": deepcopy(shot_record["status"]),
            "planned_configuration": deepcopy(shot_record["planned_configuration"]),
            "observed_outcome": deepcopy(shot_record["observed_outcome"]),
            "effective_timing": effective["timing"],
            "effective_probe_positions": effective["probe_positions"],
            "source_occurrences": deepcopy(shot_record["source_occurrences"]),
            "occurrences": [
                {key: value for key, value in item[0].items() if not key.startswith("_")}
                for item in items
            ],
            "ambiguous": len(timings) > 1,
        }
    return dict(sorted(records.items()))


def _timing_signature(shot_record: dict[str, Any]) -> tuple:
    return tuple(sorted(
        (entry["system"], entry["identifier"], tuple(entry["window_ms"] or ()))
        for entry in _explicit(shot_record)
    ))


def to_daq_ms(value: int | float, offset_ms: int | float = DAQ_OFFSET_MS) -> int | float:
    shifted = value + offset_ms
    return int(shifted) if isinstance(shifted, float) and shifted.is_integer() else shifted


def trigger_table(records: dict[int, dict[str, Any]], offset_ms: int | float = DAQ_OFFSET_MS) -> dict[str, Any]:
    """The packaged ``diagnostic-trigger-settings.yaml`` content.

    ``shots: {shot: {LABEL: {start_time_ms, end_time_ms, source_shot}}}`` on the
    DAQ clock, plus ``TP``/``TP(M/U)`` probe positions -- the shape
    ``machine_mapping.soft_x_rays`` has always read.
    """
    shots: dict[int, dict[str, Any]] = {}
    for shot, record in records.items():
        entry: dict[str, Any] = {}
        # Under an alias two logged labels compete for one key (SXR1 -> SXR).
        # The shot's own value beats an inherited one, then the most recently
        # set wins -- never whichever happened to be iterated last.
        rank: dict[str, tuple[bool, int]] = {}
        for timing in record["effective_timing"]:
            if timing["system"] != "diagnostic" or timing.get("window_ms") is None:
                continue
            start, end = timing["window_ms"]
            label = TABLE_ALIASES.get(timing["identifier"], timing["identifier"])
            priority = (not timing.get("inherited"), int(timing.get("source_shot", shot)))
            if label in rank and rank[label] >= priority:
                continue
            rank[label] = priority
            entry[label] = {
                "start_time_ms": to_daq_ms(start, offset_ms),
                "end_time_ms": to_daq_ms(end, offset_ms),
                "source_shot": timing["source_shot"],
            }
        for label, position in record.get("effective_probe_positions", {}).items():
            entry[label] = {"measured_position_m": position["positions_m"],
                            "source_shot": position["source_shot"]}
        if entry:
            shots[shot] = dict(sorted(entry.items()))
    return {"shots": shots}


__all__ = [
    "CARRY_FORWARD_SYSTEMS",
    "DAQ_OFFSET_MS",
    "RECORD_VERSION",
    "build_shot_records",
    "effective_timing_by_session",
    "to_daq_ms",
    "trigger_table",
]
