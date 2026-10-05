"""Recorded faults of the VEST diagnostics that have no IMAS validity node (#1543).

The equilibrium magnetics record their broken channels under
``equilibrium_magnetics.processing.known_faults`` and the signal-quality layer
projects them into ``validity``.  The filterscope (``spectrometer_uv``), the
pressure gauge (``barometry``) and the other machine-system signals have no
validity node in the Data Dictionary, so their faults are recorded here -- one
top-level ``diagnostic_faults`` list in ``vest.yaml`` -- and read by whoever
must decide whether to trust a shot (``workflow/magnetics_quality/
class_shot_checklist.py``).  The mapped values are not altered by a record.

Each entry names the sensor three ways (IDS node, label, raw field) and is
checked against the mapper's own definition, so an entry cannot silently move
to another sensor when a channel table changes.
"""

from __future__ import annotations

from typing import Any

from .utils import VestConfigurationError, _policy_document, resolve_vest_diagnostic

__all__ = ["DIAGNOSTIC_FAULTS_KEY", "FAULT_KINDS", "known_diagnostic_faults"]

DIAGNOSTIC_FAULTS_KEY = "diagnostic_faults"
#: What a record may say about a sensor.
FAULT_KINDS = frozenset({"railed", "no_signal", "calibration_unverified"})
_FIELDS = ("ids", "node", "label", "field", "kind", "reason")


def _check_spectrometer_uv(entry: dict[str, Any], context: str) -> None:
    from .spectrometer_uv import SIGNALS

    parts = str(entry["node"]).split(".")
    if len(parts) != 4 or parts[0] != "channel" or parts[2] != "processed_line":
        raise VestConfigurationError(f"{context}: spectrometer_uv node must be channel.<i>.processed_line.<j>")
    channel, line = int(parts[1]), int(parts[3])
    for field, sig_channel, sig_line, label, _wavelength in SIGNALS:
        if (sig_channel, sig_line) == (channel, line):
            if (int(entry["field"]), str(entry["label"])) != (field, label):
                raise VestConfigurationError(
                    f"{context}: {entry['node']} is field {field} {label!r} in the mapper, "
                    f"not field {entry['field']} {entry['label']!r}"
                )
            return
    raise VestConfigurationError(f"{context}: the mapper writes no {entry['node']}")


def _check_barometry(entry: dict[str, Any], context: str) -> None:
    if str(entry["node"]) != "gauge.0":
        raise VestConfigurationError(f"{context}: barometry records only gauge.0 (the main-chamber gauge)")
    shot = int(entry.get("from_shot") or 0)
    field = int(resolve_vest_diagnostic(shot, "barometry_main")["source"]["field"])
    if int(entry["field"]) != field:
        raise VestConfigurationError(f"{context}: barometry gauge.0 reads field {field}, not {entry['field']}")


_CHECKS = {"spectrometer_uv": _check_spectrometer_uv, "barometry": _check_barometry}


def known_diagnostic_faults(shot: int, *, info_file: str | None = None) -> list[dict[str, Any]]:
    """The recorded diagnostic faults in force on *shot*, in file order.

    ``from_shot`` and ``to_shot`` are inclusive; either may be omitted.
    """
    entries = _policy_document(info_file).get(DIAGNOSTIC_FAULTS_KEY) or []
    found: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        context = f"vest.yaml {DIAGNOSTIC_FAULTS_KEY}[{position}]"
        missing = [name for name in _FIELDS if name not in entry]
        if missing:
            raise VestConfigurationError(f"{context}: missing {', '.join(missing)}")
        if entry["kind"] not in FAULT_KINDS:
            raise VestConfigurationError(f"{context}: kind {entry['kind']!r} is not one of {sorted(FAULT_KINDS)}")
        check = _CHECKS.get(str(entry["ids"]))
        if check is None:
            raise VestConfigurationError(f"{context}: no record check for IDS {entry['ids']!r}")
        check(entry, context)
        first, last = entry.get("from_shot"), entry.get("to_shot")
        if (first is None or int(shot) >= int(first)) and (last is None or int(shot) <= int(last)):
            found.append({name: entry[name] for name in _FIELDS} | {"from_shot": first, "to_shot": last})
    return found
