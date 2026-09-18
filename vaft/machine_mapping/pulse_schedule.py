"""Map the VEST ShotLog's planned timing into ``pulse_schedule`` (#995).

The ShotLog records, for every discharge since the 2023 template, when each
diagnostic was triggered and when the 2.45 GHz ECH, NBI, helicity-injection
and gas valves were fired. Those are predefined triggers, which is exactly
what ``pulse_schedule.event(:)`` holds. Nothing here is a measurement: an
event is what the operator *set*, and says so in its provider.

Input is the per-shot record FileDB keeps at
``legacy/shotlog/{shot}/metadata/shotlog.json`` (built by
``vaft.database.shotlog``); ``data_root`` is ``legacy/shotlog``, the same
per-diagnostic root the soft X-ray and camera mappings take.

Clock. The ShotLog is logged on the timing controller, which starts 200 ms
before the DAQ trigger. Event times here are on the DAQ clock -- the ``'daq'``
time convention every stored vaft IDS uses -- so ``IF 490-510`` in the
workbook is an event at 0.290 s lasting 0.020 s. The offset applied is
recorded in ``code.parameters``.

Carried-forward triggers. A diagnostic trigger left blank on a card stays in
force from the last card that set it within the same day (the rule the packaged
trigger table has always used). Such an event is still written, and its
provenance names the shot whose card set it and marks it ``inherited``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vaft.database.shotlog.archive import load_record
from vaft.database.shotlog.records import DAQ_OFFSET_MS, to_daq_ms
from vaft.machine_mapping.utils import set_path

CODE_NAME = "vaft.machine_mapping.pulse_schedule"
CODE_VERSION = "1"
PROVIDER = "VEST ShotLog (operator-logged planned trigger)"

#: ``type`` of an event, one per ShotLog system. The DD defines no enumeration
#: for event types, so the indices are private (negative), as it requires.
EVENT_TYPES = {
    "diagnostic": (-1, "diagnostic_trigger", "Diagnostic acquisition trigger"),
    "ec": (-2, "ec", "Electron-cyclotron heating window"),
    "nbi": (-3, "nbi", "Neutral-beam injection trigger"),
    "hi": (-4, "helicity_injection", "Helicity-injection trigger"),
    "gas": (-5, "gas_injection", "Gas valve window"),
}

#: The IDS that holds each triggered system's data, where vaft maps one.
#: A label absent here gets no listener rather than a guessed one.
LISTENERS = {
    "TS": ("thomson_scattering",),
    "IF": ("interferometer",),
    "IF H": ("interferometer",),
    "IF V": ("interferometer",),
    "SXR": ("soft_x_rays",),
    "SXR1": ("soft_x_rays",),
    "SXR2": ("soft_x_rays",),
    "SXR3": ("soft_x_rays",),
    "SXR4": ("soft_x_rays",),
    "HXR": ("hard_x_rays",),
    "CES": ("charge_exchange",),
    "IDS(CES)": ("charge_exchange",),
}
SYSTEM_LISTENERS = {
    "ec": ("ec_launchers",),
    "nbi": ("nbi",),
    "gas": ("gas_injection",),
}


class PulseScheduleUnavailableError(LookupError):
    """The ShotLog has this shot but logged no usable trigger for it."""


def _seconds(milliseconds: float) -> float:
    # Logged to 0.1 ms at best; rounding keeps 350 ms from becoming 0.35000000000000003 s.
    return round(float(milliseconds) * 1e-3, 9)


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Usable timing entries, in a stable order."""
    return [
        entry
        for entry in record.get("effective_timing", [])
        if entry.get("window_ms") is not None and entry.get("system") in EVENT_TYPES
    ]


def _source_text(record: dict[str, Any], entry: dict[str, Any], offset_ms: float) -> str:
    source = record["source"]
    parts = [
        f"{source['workbook']}",
        f"sha256={source['sha256']}",
        f"sheet={source['sheet']}",
        f"cell={entry['cell']}",
        f"raw={entry['raw']!r}",
        f"source_shot={entry.get('source_shot', record['shot'])}",
        f"inherited={str(bool(entry.get('inherited'))).lower()}",
        f"daq_offset_ms={offset_ms:g}",
    ]
    if entry.get("valve_voltage_V") is not None:
        parts.append(f"valve_voltage_V={entry['valve_voltage_V']:g}")
    if entry.get("species"):
        parts.append(f"species={entry['species']}")
    return "; ".join(parts)


def map_pulse_schedule(ods: Any, record: dict[str, Any], *, offset_ms: float | None = None) -> int:
    """Write ``record``'s events into ``ods.pulse_schedule``; return how many."""
    offset = DAQ_OFFSET_MS if offset_ms is None else offset_ms
    entries = _events(record)
    if not entries:
        raise PulseScheduleUnavailableError(
            f"The ShotLog logs no usable trigger for shot {record.get('shot')} "
            f"(schema {record.get('schema_version')})"
        )
    nbi_duration_ms = (
        record.get("planned_configuration", {})
        .get("heating_and_current_drive", {})
        .get("nbi_duration_ms")
    )

    for index, entry in enumerate(entries):
        prefix = f"pulse_schedule.event.{index}"
        system = entry["system"]
        type_index, type_name, type_description = EVENT_TYPES[system]
        start_ms, end_ms = entry["window_ms"]
        duration_ms = end_ms - start_ms
        if system == "nbi" and duration_ms == 0 and nbi_duration_ms is not None:
            duration_ms = nbi_duration_ms
        set_path(ods, f"{prefix}.identifier", f"{system}:{entry['identifier']}")
        set_path(ods, f"{prefix}.type.index", type_index)
        set_path(ods, f"{prefix}.type.name", type_name)
        set_path(ods, f"{prefix}.type.description", type_description)
        set_path(ods, f"{prefix}.provider", PROVIDER)
        set_path(ods, f"{prefix}.time_stamp", _seconds(to_daq_ms(start_ms, offset)))
        set_path(ods, f"{prefix}.duration", _seconds(duration_ms))
        listeners = LISTENERS.get(entry["identifier"]) or SYSTEM_LISTENERS.get(system)
        if listeners:
            set_path(ods, f"{prefix}.listeners", list(listeners))
        set_path(ods, f"pulse_schedule.ids_properties.provenance.node.{index}.path", f"event({index + 1})")
        set_path(
            ods,
            f"pulse_schedule.ids_properties.provenance.node.{index}.sources",
            [_source_text(record, entry, offset)],
        )

    session = record.get("session", {})
    set_path(ods, "pulse_schedule.ids_properties.homogeneous_time", 2)
    set_path(
        ods,
        "pulse_schedule.ids_properties.comment",
        "Planned triggers logged by VEST operators in the ShotLog; times on the "
        "DAQ clock. Not measured.",
    )
    set_path(ods, "pulse_schedule.code.name", CODE_NAME)
    set_path(ods, "pulse_schedule.code.version", CODE_VERSION)
    # JSON rather than XML: it survives a product reload and the Access Layer
    # verbatim (docs/_guide/Data_structures.md, "What survives on code.parameters").
    set_path(
        ods,
        "pulse_schedule.code.parameters",
        json.dumps({
            "daq_offset_ms": offset,
            "record_version": record.get("record_version"),
            "schema_version": record.get("schema_version"),
            "schema_revision": record.get("schema_revision"),
            "session_id": session.get("id"),
            "experiment_date": session.get("experiment_date"),
            "ambiguous": bool(record.get("ambiguous")),
        }, sort_keys=True),
    )
    return len(entries)


def pulse_schedule(ods: Any, shot: int, data_root: str | Path | None = None) -> Any:
    """Fill ``ods.pulse_schedule`` for ``shot`` from its FileDB ShotLog record.

    ``data_root`` is the FileDB ``legacy/shotlog`` directory. Raises
    ``FileNotFoundError`` when the ShotLog does not mention the shot and
    :class:`PulseScheduleUnavailableError` when it does but logs no trigger --
    both mean "unavailable", neither is a fault.
    """
    if data_root is None:
        from vaft.database.filedb import FileDB

        data_root = FileDB.from_config().legacy("shotlog", None, artifact="input").parent
    record = load_record(shot, data_root)
    map_pulse_schedule(ods, record)
    return ods


__all__ = [
    "EVENT_TYPES",
    "LISTENERS",
    "PulseScheduleUnavailableError",
    "map_pulse_schedule",
    "pulse_schedule",
]
