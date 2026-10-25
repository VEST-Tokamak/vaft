"""The VEST equilibrium magnetic sensor registry (issue #2158).

``vest.yaml`` ``0.diagnostics.equilibrium_magnetics.sensors`` is the only
runtime definition of the 75 equilibrium magnetic sensors: 64 poloidal-field
probes then 11 flux loops. This module reads it, validates it, and resolves the
shot's wiring onto it. Every machine-mapping consumer of the sensors (static and
diagnostics ODS, raw-source records, constraint groups, provenance, the wiring
check) goes through :func:`resolve_equilibrium_magnetics_sensors`.

Three identities are kept apart (#2148 audit, section 2):

``id``
    The physical sensor -- a position. It never changes with wiring.
``field_code``
    The raw DAQ field the position reads for the shot. The ``wiring``
    revisions in ``processing`` move fields between positions (39437/39438).
``name``
    The DAQ label (``table.dat``) of that field. ODS ``name``/``identifier``
    carry it, so it follows the wiring exactly as it always has; ``aliases``
    lists the labels a position carries under other eras' wiring.

The record order is the canonical index: ODS ``b_field_pol_probe[i]`` is record
``i`` and ODS ``flux_loop[j]`` is record ``64 + j``, which is also the EFUND and
EFIT order. ``kind_index`` is the index within the record's IDS array.

``wiring`` and ``known_faults`` entries name sensors by ``id``. The legacy
``{index, r, z}`` form is still accepted through :func:`sensor_for_entry`, which
checks the position the entry claims, so an entry can never silently move to a
different sensor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from .utils import (
    VestConfigurationError,
    _resolve_info_file_path,
    load_yaml,
    resolve_shot_revisions_with_provenance,
    resolve_vest_diagnostic,
)

__all__ = [
    "EXPECTED_SENSOR_COUNTS",
    "FLUX_LOOP_KIND",
    "PROBE_KIND",
    "REGISTRY_CONFIG_PATH",
    "ResolvedMagneticsRegistry",
    "SensorRecord",
    "field_labels",
    "registry_fingerprint",
    "registry_source",
    "resolve_equilibrium_magnetics_registry",
    "resolve_equilibrium_magnetics_sensors",
    "sensor_for_entry",
]

PROBE_KIND = "b_field_pol_probe"
FLUX_LOOP_KIND = "flux_loop"
#: The kinds in canonical order: every probe precedes every flux loop, which is
#: what makes the flux-loop IDS index ``canonical index - 64``.
KIND_ORDER = (PROBE_KIND, FLUX_LOOP_KIND)
EXPECTED_SENSOR_COUNTS = MappingProxyType({PROBE_KIND: 64, FLUX_LOOP_KIND: 11})
REGISTRY_CONFIG_PATH = "0.diagnostics.equilibrium_magnetics.sensors"
ID_BASES = ("table", "position")

_REQUIRED_RECORD_KEYS = frozenset(
    {"id", "kind", "index", "name", "aliases", "r", "z", "field_code", "calibration", "id_basis"}
)
_OPTIONAL_RECORD_KEYS = frozenset({"ladder_candidate"})
_OVERRIDE_KEYS = frozenset({"id", "index", "r", "z", "field_code", "calibration"})
_ENTRY_KEYS = frozenset({"id", "index", "r", "z"})

# A label-shaped id is derived from the DAQ label; a position-shaped id encodes
# the probe column and z, e.g. ``bpol_C_z+0.060``.
_PROBE_LABEL = re.compile(r"^MagneticFieldProbe_(?P<stem>[A-Z][A-Z0-9]*-\d+)(?:_Bz)?$")
_LOOP_LABEL = re.compile(r"^Flux Loop - #(?P<number>\d+)$")
_POSITION_ID = re.compile(r"^bpol_(?P<column>[A-Z]+)_z(?P<z>[+-]\d+\.\d+)$")


@dataclass(frozen=True)
class SensorRecord:
    """One equilibrium magnetic sensor, resolved for a shot (or the inventory).

    ``field_code``, ``calibration`` and ``name`` are the shot's effective
    values; ``rewired`` says the shot's wiring revision replaced them.
    """

    id: str
    kind: str
    index: int
    kind_index: int
    name: str
    aliases: tuple[str, ...]
    r: float
    z: float
    field_code: int
    calibration: float
    id_basis: str
    ladder_candidate: str | None = None
    rewired: bool = False

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["aliases"] = list(self.aliases)
        return record


@dataclass(frozen=True)
class ResolvedMagneticsRegistry:
    """The registry resolved for one shot, with the wiring that produced it.

    ``overrides`` keeps the shape ``MagneticsWiring.overrides`` always had:
    ``{index, r, z, field_code, calibration}`` per rewired probe, ``index``
    being the canonical index. ``shot`` is ``None`` for the inventory, which
    applies no wiring revision at all.
    """

    shot: int | None
    records: tuple[SensorRecord, ...]
    layout: str
    overrides: tuple[Mapping[str, Any], ...]
    wiring_provenance: Mapping[str, Any] | None
    fingerprint: str
    source: Mapping[str, Any]


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """A plain ``dict``/``list`` deep copy of a frozen registry value."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _document_key(info_file: str | None) -> tuple[str, int]:
    path = _resolve_info_file_path(info_file)
    return path, os.stat(path).st_mtime_ns


@lru_cache(maxsize=8)
def _load_document(path: str, mtime_ns: int) -> Mapping[str, Any]:
    """The one place this module parses vest.yaml (B2 replaces the loader)."""
    return load_yaml(path)


def _sensor_block(document: Mapping[str, Any]) -> Mapping[str, Any]:
    defaults = document.get("0") or document.get(0) or {}
    try:
        block = defaults["diagnostics"]["equilibrium_magnetics"]["sensors"]
    except (KeyError, TypeError) as exc:
        raise VestConfigurationError(f"vest.yaml has no {REGISTRY_CONFIG_PATH} block") from exc
    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"{REGISTRY_CONFIG_PATH} must be a mapping")
    return block


# ---------------------------------------------------------------------------
# Validation of the base records
# ---------------------------------------------------------------------------


def _number(value: Any, *, context: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VestConfigurationError(f"{context}: {name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise VestConfigurationError(f"{context}: {name} must be finite, got {value!r}")
    return number


def _integer(value: Any, *, context: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VestConfigurationError(f"{context}: {name} must be an integer, got {value!r}")
    return int(value)


def _calibration(value: Any, *, context: str) -> float:
    number = _number(value, context=context, name="calibration")
    if number == 0.0:
        raise VestConfigurationError(f"{context}: calibration must be non-zero")
    return number


def label_shaped_id(kind: str, label: str) -> str | None:
    """The id a DAQ label implies (``bpol_H1-01``, ``fl_03``), or ``None``."""
    if kind == PROBE_KIND:
        match = _PROBE_LABEL.match(label)
        return None if match is None else f"bpol_{match['stem']}"
    match = _LOOP_LABEL.match(label)
    return None if match is None else f"fl_{int(match['number']):02d}"


def _parse_record(raw: Any, position: int, kind_index: int) -> SensorRecord:
    context = f"{REGISTRY_CONFIG_PATH}.records[{position}]"
    if not isinstance(raw, Mapping):
        raise VestConfigurationError(f"{context}: a record must be a mapping")
    keys = set(raw)
    missing = _REQUIRED_RECORD_KEYS - keys
    if missing:
        raise VestConfigurationError(f"{context}: missing {sorted(missing)}")
    unknown = keys - _REQUIRED_RECORD_KEYS - _OPTIONAL_RECORD_KEYS
    if unknown:
        raise VestConfigurationError(f"{context}: unknown keys {sorted(unknown)}")
    sensor_id = raw["id"]
    if not isinstance(sensor_id, str) or not sensor_id:
        raise VestConfigurationError(f"{context}: id must be a non-empty string")
    context = f"{context} ({sensor_id})"
    kind = raw["kind"]
    if kind not in KIND_ORDER:
        raise VestConfigurationError(f"{context}: kind must be one of {KIND_ORDER}, got {kind!r}")
    index = _integer(raw["index"], context=context, name="index")
    if index != position:
        raise VestConfigurationError(
            f"{context}: index {index} is not its list position {position}; the record "
            "order is the canonical index"
        )
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise VestConfigurationError(f"{context}: name must be a non-empty string")
    aliases = raw["aliases"]
    if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)) or not all(
        isinstance(alias, str) and alias for alias in aliases
    ):
        raise VestConfigurationError(f"{context}: aliases must be a list of labels")
    if len(set(aliases)) != len(aliases) or name in aliases:
        raise VestConfigurationError(f"{context}: aliases must be distinct and differ from name")
    id_basis = raw["id_basis"]
    if id_basis not in ID_BASES:
        raise VestConfigurationError(f"{context}: id_basis must be one of {ID_BASES}, got {id_basis!r}")
    ladder = raw.get("ladder_candidate")
    if ladder is not None and (id_basis != "position" or not isinstance(ladder, str) or not ladder):
        raise VestConfigurationError(
            f"{context}: ladder_candidate is a non-empty string, and only for id_basis position"
        )
    r = _number(raw["r"], context=context, name="r")
    z = _number(raw["z"], context=context, name="z")
    if id_basis == "table":
        expected = label_shaped_id(kind, name)
        if expected != sensor_id:
            raise VestConfigurationError(
                f"{context}: id_basis table needs the id its label {name!r} implies "
                f"({expected!r})"
            )
    else:
        match = _POSITION_ID.match(sensor_id) if kind == PROBE_KIND else None
        if match is None or not math.isclose(float(match["z"]), z, abs_tol=5e-5):
            raise VestConfigurationError(
                f"{context}: a position-shaped id is bpol_<column>_z<+-z>, and its z must be "
                f"the record's z={z}"
            )
    return SensorRecord(
        id=sensor_id,
        kind=kind,
        index=index,
        kind_index=kind_index,
        name=name,
        aliases=tuple(aliases),
        r=r,
        z=z,
        field_code=_integer(raw["field_code"], context=context, name="field_code"),
        calibration=_calibration(raw["calibration"], context=context),
        id_basis=id_basis,
        ladder_candidate=ladder,
    )


def _validate_base(records: Sequence[SensorRecord]) -> None:
    counts = {kind: sum(1 for record in records if record.kind == kind) for kind in KIND_ORDER}
    if counts != dict(EXPECTED_SENSOR_COUNTS):
        raise VestConfigurationError(
            f"{REGISTRY_CONFIG_PATH}: expected {dict(EXPECTED_SENSOR_COUNTS)} sensors, got {counts}"
        )
    order = [KIND_ORDER.index(record.kind) for record in records]
    if order != sorted(order):
        raise VestConfigurationError(
            f"{REGISTRY_CONFIG_PATH}: every {PROBE_KIND} must precede every {FLUX_LOOP_KIND}"
        )
    for attribute in ("id", "field_code", "name"):
        values = [getattr(record, attribute) for record in records]
        duplicates = sorted({value for value in values if values.count(value) > 1}, key=str)
        if duplicates:
            raise VestConfigurationError(f"{REGISTRY_CONFIG_PATH}: duplicate {attribute} {duplicates}")
    labels = {record.name for record in records}
    for record in records:
        unknown = [alias for alias in record.aliases if alias not in labels]
        if unknown:
            raise VestConfigurationError(
                f"{REGISTRY_CONFIG_PATH}: {record.id} aliases {unknown} are not the label of any "
                "registry field"
            )
    # A position-shaped id must not be a label-shaped id another position owns:
    # a consumer deriving ids from label suffixes would then cross-wire them.
    label_ids = {
        label_shaped_id(record.kind, label): record.id
        for record in records
        for label in (record.name, *record.aliases)
    }
    for record in records:
        owner = label_ids.get(record.id)
        if owner is not None and owner != record.id:
            raise VestConfigurationError(
                f"{REGISTRY_CONFIG_PATH}: id {record.id} is the label-shaped id of {owner}"
            )


@lru_cache(maxsize=8)
def _base(path: str, mtime_ns: int) -> tuple[tuple[SensorRecord, ...], Mapping[str, Any]]:
    block = _sensor_block(_load_document(path, mtime_ns))
    raw_records = block.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise VestConfigurationError(f"{REGISTRY_CONFIG_PATH}.records must be a list")
    source = block.get("source")
    if not isinstance(source, Mapping):
        raise VestConfigurationError(f"{REGISTRY_CONFIG_PATH}.source must be a mapping")
    records = []
    kind_counts = dict.fromkeys(KIND_ORDER, 0)
    for position, raw in enumerate(raw_records):
        kind = raw.get("kind") if isinstance(raw, Mapping) else None
        kind_index = kind_counts.get(kind, 0)
        records.append(_parse_record(raw, position, kind_index))
        if kind in kind_counts:
            kind_counts[kind] += 1
    _validate_base(records)
    return tuple(records), _frozen(source)


# ---------------------------------------------------------------------------
# Entry adapter (wiring / known_faults)
# ---------------------------------------------------------------------------


def _check_pin(entry: Mapping[str, Any], record: SensorRecord, *, context: str, required: bool) -> None:
    has_r, has_z = "r" in entry, "z" in entry
    if not (has_r or has_z):
        if required:
            raise VestConfigurationError(f"{context}: a legacy index entry must pin r and z")
        return
    if has_r != has_z:
        raise VestConfigurationError(f"{context}: give both r and z, or neither")
    claimed = (float(entry["r"]), float(entry["z"]))
    if not all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip((record.r, record.z), claimed)):
        raise VestConfigurationError(
            f"{context}: {record.id} is at r={record.r}, z={record.z}, not at "
            f"r={claimed[0]}, z={claimed[1]} as the entry names"
        )


def _entry_record(
    records: Sequence[SensorRecord],
    entry: Mapping[str, Any],
    *,
    kind: str,
    context: str,
    index_scope: str,
) -> SensorRecord:
    if not isinstance(entry, Mapping):
        raise VestConfigurationError(f"{context}: an entry must be a mapping")
    if ("id" in entry) == ("index" in entry):
        raise VestConfigurationError(f"{context}: an entry names its sensor by id or by index, exactly one")
    if "id" in entry:
        by_id = {record.id: record for record in records}
        record = by_id.get(str(entry["id"]))
        if record is None:
            raise VestConfigurationError(f"{context}: no sensor with id {entry['id']!r}")
        if record.kind != kind:
            raise VestConfigurationError(f"{context}: {record.id} is a {record.kind}, not a {kind}")
        _check_pin(entry, record, context=context, required=False)
        return record
    # Legacy adapter: the pre-#2158 `{index, r, z}` form, pinned by position.
    index = int(entry["index"])
    if "r" not in entry or "z" not in entry:
        raise VestConfigurationError(f"{context}: a legacy index entry must pin r and z")
    claimed = (float(entry["r"]), float(entry["z"]))
    if index_scope == "kind":
        rows = [record for record in records if record.kind == kind]
        if not 0 <= index < len(rows):
            raise VestConfigurationError(f"{context}: {kind} index {index} is out of range")
        row = rows[index]
        if not all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip((row.r, row.z), claimed)):
            raise VestConfigurationError(
                f"{context}: {kind} index {index} is at r={row.r}, z={row.z}, "
                f"not at r={claimed[0]}, z={claimed[1]} as the entry names"
            )
        return row
    if not 0 <= index < len(records):
        raise VestConfigurationError(f"{context}: probe index {index} is out of range")
    row = records[index]
    if row.kind != kind or not all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip((row.r, row.z), claimed)
    ):
        raise VestConfigurationError(
            f"{context}: index {index} is a {row.kind} at r={row.r}, z={row.z}, "
            f"not the probe at r={claimed[0]}, z={claimed[1]} the entry names"
        )
    return row


def sensor_for_entry(
    entry: Mapping[str, Any],
    *,
    kind: str,
    context: str,
    index_scope: str = "canonical",
    info_file: str | None = None,
) -> SensorRecord:
    """The inventory record a ``wiring``/``known_faults`` entry names.

    ``{id: ...}`` names the sensor directly (optional ``r``/``z`` are checked
    against it). The legacy ``{index, r, z}`` form is the validated adapter:
    ``index`` is the canonical index (``index_scope="canonical"``) or counts
    within *kind* (``"kind"``, how flux-loop entries were written), and ``r``
    and ``z`` must match the record it lands on.
    """
    if index_scope not in {"canonical", "kind"}:
        raise ValueError(f"index_scope must be 'canonical' or 'kind', not {index_scope!r}")
    records, _source = _base(*_document_key(info_file))
    return _entry_record(records, entry, kind=kind, context=context, index_scope=index_scope)


# ---------------------------------------------------------------------------
# Shot resolution
# ---------------------------------------------------------------------------


def _fingerprint(records: Sequence[SensorRecord]) -> str:
    payload = json.dumps(
        [record.as_dict() for record in records],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _wiring_for_shot(shot: int, info_file: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = resolve_vest_diagnostic(int(shot), "equilibrium_magnetics", info_file=info_file)
    item = config["processing"]["wiring"]
    return resolve_shot_revisions_with_provenance(
        {key: value for key, value in item.items() if key != "revisions"},
        item.get("revisions"),
        int(shot),
        context="VEST equilibrium_magnetics wiring",
    )


@lru_cache(maxsize=2048)
def _resolve(path: str, mtime_ns: int, shot: int | None, info_file: str | None) -> ResolvedMagneticsRegistry:
    base, source = _base(path, mtime_ns)
    labels = {record.field_code: record.name for record in base}
    if shot is None:
        return ResolvedMagneticsRegistry(
            shot=None,
            records=base,
            layout=str(source.get("wiring_layout", "")),
            overrides=(),
            wiring_provenance=None,
            fingerprint=_fingerprint(base),
            source=source,
        )
    resolved, provenance = _wiring_for_shot(shot, info_file)
    context = "equilibrium_magnetics wiring"
    records = list(base)
    overrides = []
    rewired: set[str] = set()
    for entry in resolved.get("probe_overrides") or ():
        if isinstance(entry, Mapping):
            unknown = set(entry) - _OVERRIDE_KEYS
            if unknown:
                raise VestConfigurationError(f"{context}: unknown override keys {sorted(unknown)}")
        record = _entry_record(base, entry, kind=PROBE_KIND, context=context, index_scope="canonical")
        if record.id in rewired:
            raise VestConfigurationError(f"{context} for shot {int(shot)}: {record.id} is overridden twice")
        rewired.add(record.id)
        field_code = _integer(entry["field_code"], context=context, name="field_code")
        calibration = _calibration(entry["calibration"], context=context)
        name = labels.get(field_code)
        if name is None:
            raise VestConfigurationError(
                f"{context} for shot {int(shot)}: field {field_code} has no label in the registry"
            )
        if name != record.name and name not in record.aliases:
            raise VestConfigurationError(
                f"{context} for shot {int(shot)}: {record.id} would read {name!r}, which is neither "
                "its name nor one of its aliases"
            )
        records[record.index] = replace(
            record, field_code=field_code, calibration=calibration, name=name, rewired=True
        )
        overrides.append(
            MappingProxyType(
                {
                    "index": record.index,
                    "r": record.r,
                    "z": record.z,
                    "field_code": field_code,
                    "calibration": calibration,
                }
            )
        )
    codes = [record.field_code for record in records]
    if len(set(codes)) != len(codes):
        raise VestConfigurationError(
            f"equilibrium_magnetics wiring for shot {int(shot)} feeds one raw field "
            "to two positions"
        )
    result = tuple(records)
    return ResolvedMagneticsRegistry(
        shot=int(shot),
        records=result,
        layout=str(resolved["layout"]),
        overrides=tuple(overrides),
        wiring_provenance=_frozen(provenance),
        fingerprint=_fingerprint(result),
        source=source,
    )


def resolve_equilibrium_magnetics_registry(
    shot: int | None = None, *, info_file: str | None = None
) -> ResolvedMagneticsRegistry:
    """The validated registry for *shot*, with its wiring and fingerprint.

    ``shot=None`` is the inventory: the records as ``vest.yaml`` lists them
    (the 2409 base wiring), no revision applied. Any integer, ``0`` included,
    applies the wiring revision whose bounds contain it.
    """
    path, mtime_ns = _document_key(info_file)
    return _resolve(path, mtime_ns, None if shot is None else int(shot), info_file)


def resolve_equilibrium_magnetics_sensors(
    shot: int | None = None, *, info_file: str | None = None
) -> tuple[SensorRecord, ...]:
    """The 75 sensor records for *shot*, ordered by canonical index.

    See :func:`resolve_equilibrium_magnetics_registry` for what ``shot=None``
    means. The records are frozen; the tuple is shared between calls.
    """
    return resolve_equilibrium_magnetics_registry(shot, info_file=info_file).records


def registry_fingerprint(shot: int | None = None, *, info_file: str | None = None) -> str:
    """sha256 of the canonical JSON of the resolved records (sorted keys, float repr).

    It identifies the effective sensor definitions, not the bytes of
    ``vest.yaml``: an edit elsewhere in the file leaves it unchanged.
    """
    return resolve_equilibrium_magnetics_registry(shot, info_file=info_file).fingerprint


def registry_source(*, info_file: str | None = None) -> Mapping[str, Any]:
    """The block-level provenance (``sensors.source``) of the registry."""
    return resolve_equilibrium_magnetics_registry(None, info_file=info_file).source


def field_labels(*, info_file: str | None = None) -> Mapping[int, str]:
    """``{field_code: DAQ label}`` for the 75 registry fields (the old name table)."""
    records = resolve_equilibrium_magnetics_sensors(None, info_file=info_file)
    return MappingProxyType({record.field_code: record.name for record in records})
