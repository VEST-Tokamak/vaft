"""The effective VEST configuration for one shot and diagnostic (#2159).

``vest.yaml`` resolves in layers: a diagnostic block's base values, the one
shot-era revision that applies to the shot (at every nesting level that has a
``revisions`` table), and finally an explicit, validated workflow override::

    canonical vest.yaml  ->  shot-era revisions  ->  validated overrides  ->  effective

:func:`resolve_vest_processing_config` performs all three and returns a frozen
:class:`ResolvedVestConfiguration` that says what was resolved, where every
value came from, and a deterministic identity of the result.

Field classes
-------------
Not every value is a tuneable analysis parameter. A dotted key path is
classified by the most restrictive segment it contains:

``machine_fact``
    sensor identity, signal sources and field codes, wiring, DAQ type, mode
    and sample rate, geometry, probe bias and tip dimensions. **Never** changed by an override; restating the
    effective value is inert.
``calibration``
    gains, polarity, calibration factors and circuit constants. Changed only
    with an explicit ``calibration_override_reason``, which is recorded.
``analysis``
    analysis and baseline windows -- overrideable.
``processing``
    everything else (filters, repairs, tolerances) -- overrideable.

Overrides are validated against the effective structure: an unknown key, a
change of type (a number for a string, a list for a scalar, a mapping for a
leaf) or a protected change raises :class:`VestConfigurationError`.

Fingerprint
-----------
``fingerprint`` is the SHA-256 of the canonical JSON of ``effective`` alone --
not of the provenance, the source path, the shot or a timestamp -- so two
shots, or two differently formatted files, that resolve to the same values
have the same fingerprint. The canonical form (:func:`canonical_json`):

* mapping keys become strings (YAML ``0:`` and ``"0":`` are the same key;
  a mapping holding both is refused) and are sorted;
* tuples and lists are JSON arrays, in order;
* numbers are compared by value: an integral finite float is written as an
  integer (``25000`` and ``25000.0`` are the same setting), other floats by
  Python's shortest round-trip ``repr``; booleans stay booleans;
* non-finite floats are written as ``{"$float": "NaN"}`` (``"Infinity"``,
  ``"-Infinity"``), distinct from any string; anything else non-JSON is refused;
* compact separators, UTF-8, no ASCII escaping.

YAML comments, key order, quoting, flow versus block style and number
spelling (``2.5e+3`` versus ``2500.0``) therefore never move the fingerprint.
(PyYAML reads an exponent without a sign, ``2.5e3``, as a *string*; that is a
value change, and the type check refuses it as an override.)
``source_sha256`` is the SHA-256 of the configuration file's bytes, kept
separately: it changes with any edit, including a comment.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .utils import (
    REVISION_BOUND_KEYS,
    VestConfigurationError,
    _deep_merge,
    _policy_document,
    _resolve_info_file_path,
    _shot_number,
    package_data_path,
    resolve_shot_revisions_with_provenance,
    validate_calibration,
)

__all__ = [
    "EFFECTIVE_CONFIG_SCHEMA",
    "FieldClass",
    "MAPPED_DIAGNOSTICS",
    "ResolvedVestConfiguration",
    "canonical_json",
    "classify_config_path",
    "configuration_fingerprint",
    "impa_shot_era",
    "resolve_vest_processing_config",
    "vest_configuration_record",
    "vest_configuration_source",
]

#: Version of the provenance record and of the fingerprint's canonical form.
EFFECTIVE_CONFIG_SCHEMA = "vaft.vest_effective_config/1"


class FieldClass(str, Enum):
    """Override authority of a configuration key (see the module docstring)."""

    MACHINE_FACT = "machine_fact"
    CALIBRATION = "calibration"
    ANALYSIS = "analysis"
    PROCESSING = "processing"


#: Key segments that make a path a machine fact.
MACHINE_FACT_SEGMENTS = frozenset(
    {
        "source", "sources", "source_field", "legacy_field", "field", "fields",
        "voltage_field", "current_field", "tf_field", "channels", "bz_channels",
        "br_channels_retired", "coil_field_codes", "daq_type", "daq_channel",
        "daq_mode", "raw_unit", "wiring", "layout", "probe_overrides",
        "known_faults", "geometry", "port", "launching_position", "direction",
        "polarization", "stations", "first_shot", "sensors", "label", "labels",
        "gauge", "fl10_field", "unavailable_reason", "sample_rate", "fs",
        "first_operational_shot", "reference_radius", "vd3", "tip_length_mm",
        "tip_radius_mm", "gas_species",
    }
)
#: Key segments that make a path a calibration (unless a machine fact).
CALIBRATION_SEGMENTS = frozenset(
    {
        "calibration", "gain", "gains", "coil_gains", "bz_gain", "factor", "operation",
        "sign", "multiply", "voltage_gain", "current_divisor", "flux_gain",
        "gain_numerator", "effective_resistance_ohm", "rogowski_shunt", "tf_circuit",
        "turns", "tf_turns", "inductance", "resistance", "capacitance", "orientation",
        "output_unit",
    }
)
#: Key segments that make a path an analysis selection (else processing).
ANALYSIS_SEGMENTS = frozenset(
    {
        "window", "baseline", "analysis_start", "analysis_end", "lookback",
        "index_start", "index_end", "probe_baseline_end", "flux_baseline_window",
        "flux_baseline_samples", "subtract_window", "calibration_window",
    }
)


def classify_config_path(path: Sequence[Any]) -> FieldClass:
    """The :class:`FieldClass` of a dotted key path (most restrictive segment wins)."""
    segments = {str(part) for part in path}
    if segments & MACHINE_FACT_SEGMENTS:
        return FieldClass.MACHINE_FACT
    if segments & CALIBRATION_SEGMENTS:
        return FieldClass.CALIBRATION
    if segments & ANALYSIS_SEGMENTS:
        return FieldClass.ANALYSIS
    return FieldClass.PROCESSING


# --- canonical serialization and identity ----------------------------------------------


def _canonical(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in out:
                raise VestConfigurationError(
                    f"{path}: keys {key!r} and {name!r} collide in the canonical form"
                )
            out[name] = _canonical(item, path=f"{path}.{name}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, str)):
        return value
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        value = value.item()  # numpy scalar
        if isinstance(value, bool):
            return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "NaN"}
        if math.isinf(value):
            return {"$float": "Infinity" if value > 0 else "-Infinity"}
        if value.is_integer():
            return int(value)
        return value
    raise VestConfigurationError(
        f"{path}: {type(value).__name__} is not a configuration value"
    )


def canonical_json(value: Any) -> str:
    """The canonical JSON form the fingerprint is computed over (module docstring)."""
    return json.dumps(
        _canonical(value, path="<config>"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def configuration_fingerprint(effective: Any) -> str:
    """SHA-256 of :func:`canonical_json` of ``effective``."""
    return hashlib.sha256(canonical_json(effective).encode("utf-8")).hexdigest()


@lru_cache(maxsize=16)
def _file_sha256(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vest_configuration_source(info_file: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Which configuration file a resolution reads: path, packaged flag and SHA-256.

    The packaged file is recorded by its package-relative name, so the record
    carries no machine-specific path.
    """
    path = _resolve_info_file_path(info_file)
    stat = os.stat(path)
    packaged = os.path.samefile(path, package_data_path("vest.yaml"))
    return {
        "path": "vaft/machine_mapping/vest.yaml" if packaged else path,
        "packaged": bool(packaged),
        "sha256": _file_sha256(path, stat.st_mtime_ns, stat.st_size),
    }


def _vaft_version() -> str:
    try:
        from vaft import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - a broken install still resolves
        return "unknown"


# --- frozen views ----------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolvedVestConfiguration:
    """The effective configuration of one diagnostic for one shot.

    ``effective`` and ``provenance`` are read-only views (mappings are
    ``MappingProxyType``, lists are tuples); :meth:`effective_dict` and
    :meth:`as_record` return plain, caller-owned copies.
    """

    shot: int
    diagnostic: str
    effective: Mapping[str, Any]
    provenance: Mapping[str, Any]
    fingerprint: str
    source_sha256: str

    def effective_dict(self) -> dict[str, Any]:
        """A mutable deep copy of ``effective`` with lists restored."""
        return _thaw(self.effective)

    def as_record(self, *, include_keys: bool = False) -> dict[str, Any]:
        """A JSON-ready record for a manifest.

        ``include_keys`` adds the per-key provenance table, which is large; the
        revision list and accepted overrides are always included.
        """
        provenance = _thaw(self.provenance)
        if not include_keys:
            provenance.pop("keys", None)
        return {
            "schema": EFFECTIVE_CONFIG_SCHEMA,
            "shot": self.shot,
            "diagnostic": self.diagnostic,
            "fingerprint": self.fingerprint,
            "source_sha256": self.source_sha256,
            "effective": json.loads(canonical_json(self.effective)),
            "provenance": json.loads(json.dumps(provenance, default=str)),
        }


# --- resolution ------------------------------------------------------------------------


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], Any]]:
    """Dotted leaves of a nested mapping; lists and scalars are leaves."""
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    else:
        yield path, value


def _dotted(path: Sequence[Any]) -> str:
    return ".".join(str(part) for part in path)


def _resolve_tree(
    node: Any,
    shot: int,
    *,
    context: str,
    path: tuple[Any, ...],
    origins: dict[tuple[Any, ...], dict[str, Any]],
    applied: list[dict[str, Any]],
) -> Any:
    """Resolve every ``revisions`` table in ``node``, recording which era set which key."""
    if not isinstance(node, Mapping):
        return node
    if isinstance(node.get("revisions"), list):
        base = {key: value for key, value in node.items() if key != "revisions"}
        resolved, provenance = resolve_shot_revisions_with_provenance(
            base, node["revisions"], shot, context=context
        )
        record = {"path": _dotted(path), **provenance}
        applied.append(record)
        index = provenance["revision_index"]
        if index is not None:
            revision = {
                key: value
                for key, value in node["revisions"][index].items()
                if key not in REVISION_BOUND_KEYS
            }
            # Only the leaves the revision itself sets come from it; a sibling
            # it leaves alone inside the same block is still the base's.
            for leaf, _value in _leaves(revision):
                origins[(*path, *leaf)] = record
        node = resolved
    return {
        key: _resolve_tree(
            value,
            shot,
            context=f"{context}.{key}",
            path=(*path, key),
            origins=origins,
            applied=applied,
        )
        for key, value in node.items()
    }


def _origin_for(path: tuple[Any, ...], origins: Mapping[tuple[Any, ...], Any]) -> Any:
    for depth in range(len(path), -1, -1):
        record = origins.get(path[:depth])
        if record is not None:
            return record
    return None


def _numeric(value: Any) -> float | None:
    """``value`` as a number, accepting PyYAML's numeric strings.

    PyYAML (YAML 1.1) reads an exponent without a sign -- ``1.0e4`` -- as a
    string, and vest.yaml spells the PF coil gains and the TF factor that way;
    the mappers ``float()`` them. An override is compared with such a value
    by number, not by spelling.
    """
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _same_value(left: Any, right: Any) -> bool:
    left_number, right_number = _numeric(left), _numeric(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number or (math.isnan(left_number) and math.isnan(right_number))
    try:
        return canonical_json(left) == canonical_json(right)
    except VestConfigurationError:
        return False


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, numbers.Real):
        return "number"
    if isinstance(value, str):
        return "numeric string" if _numeric(value) is not None else "string"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def _validate_overrides(
    effective: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    context: str,
    calibration_override_reason: str | None,
    path: tuple[Any, ...] = (),
    accepted: list[dict[str, Any]],
) -> dict[Any, Any]:
    """Validate ``overrides`` against ``effective``; return only its real changes.

    Inert restatements are dropped, so they can neither change a value's
    spelling nor the fingerprint without being recorded.
    """
    changes: dict[Any, Any] = {}
    for key, value in overrides.items():
        here = (*path, key)
        dotted = _dotted(here)
        if not isinstance(effective, Mapping) or key not in effective:
            known = ", ".join(sorted(map(str, effective))) if isinstance(effective, Mapping) else ""
            raise VestConfigurationError(
                f"{context}: override {dotted!r} names no configured key"
                + (f"; configured keys here: {known}" if known else "")
            )
        current = effective[key]
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            nested = _validate_overrides(
                current,
                value,
                context=context,
                calibration_override_reason=calibration_override_reason,
                path=here,
                accepted=accepted,
            )
            if nested:
                changes[key] = nested
            continue
        current_kind, value_kind = _kind(current), _kind(value)
        # A configured numeric string (PyYAML's "1.0e4") takes a number; any
        # other change of kind -- a string for a number included -- is refused.
        compatible = current_kind == value_kind or (
            current_kind == "numeric string" and value_kind == "number"
        )
        if current_kind != "null" and value_kind != "null" and not compatible:
            raise VestConfigurationError(
                f"{context}: override {dotted!r} is a {value_kind}, the configured value a {current_kind}"
            )
        if _same_value(current, value):
            continue  # restating the effective value is inert, for every class
        field_class = classify_config_path(here)
        if field_class is FieldClass.MACHINE_FACT:
            raise VestConfigurationError(
                f"{context}: {dotted!r} is a machine fact (sensor, source, wiring, DAQ or geometry) "
                "and cannot be overridden; change vest.yaml with its evidence instead"
            )
        if field_class is FieldClass.CALIBRATION and not calibration_override_reason:
            raise VestConfigurationError(
                f"{context}: {dotted!r} is a calibration; overriding it needs an explicit "
                "calibration_override_reason, which is recorded in the provenance"
            )
        accepted.append(
            {
                "path": dotted,
                "class": field_class.value,
                "configured": json.loads(canonical_json(current)),
                "value": json.loads(canonical_json(value)),
            }
        )
        changes[key] = value
    return changes


def impa_shot_era(
    calibration: Mapping[str, Any], shot: int, *, context: str = "impa.calibration"
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """The IMPA ``shot_era_overrides`` entry for ``shot``, validated (audit gap G6).

    The YAML schema is unchanged (``min_shot``/``max_shot``); this adapter
    normalizes each entry to the inclusive ``from_shot``/``to_shot`` bounds of
    every other revision table and resolves it with the same validator, so
    overlapping entries are an error rather than first-match-wins. A missing
    ``min_shot`` is open-ended (the old ``0`` default meant the same); a
    missing ``max_shot`` is an error -- the old ``0`` default made such an
    entry silently never match. Entries may set only ``gain``, ``channels``
    and ``bz_channels``.

    Returns ``(entry, provenance)``; ``entry`` is ``None`` when no era applies.
    """
    eras = calibration.get("shot_era_overrides") or ()
    if isinstance(eras, (str, bytes)) or not isinstance(eras, Sequence):
        raise VestConfigurationError(f"{context}: shot_era_overrides must be a list")
    normalized = []
    allowed = {"min_shot", "max_shot", "gain", "channels", "bz_channels"}
    for index, era in enumerate(eras):
        era_context = f"{context}.shot_era_overrides[{index}]"
        if not isinstance(era, Mapping):
            raise VestConfigurationError(f"{era_context}: entry must be a mapping")
        unknown = sorted(str(key) for key in era if key not in allowed)
        if unknown:
            raise VestConfigurationError(
                f"{era_context}: unknown keys {', '.join(map(repr, unknown))}; "
                f"an IMPA era may set {', '.join(sorted(allowed - {'min_shot', 'max_shot'}))}"
            )
        if era.get("max_shot") is None:
            raise VestConfigurationError(
                f"{era_context}: max_shot is required (an entry without it never matched)"
            )
        normalized.append(
            {
                "from_shot": era.get("min_shot"),
                "to_shot": era.get("max_shot"),
                **{key: value for key, value in era.items() if key not in {"min_shot", "max_shot"}},
            }
        )
    base = {"gain": None, "channels": None, "bz_channels": None}
    _resolved, provenance = resolve_shot_revisions_with_provenance(
        base, normalized, shot, context=f"{context}.shot_era_overrides"
    )
    index = provenance["revision_index"]
    return (None if index is None else eras[index]), provenance


def _raw_block(document: Mapping[Any, Any], diagnostic: str) -> tuple[Any, str]:
    if diagnostic == "impa":
        return document.get("impa"), "impa"
    defaults = document.get("0") or document.get(0) or {}
    diagnostics = defaults.get("diagnostics", {}) if isinstance(defaults, Mapping) else {}
    if not isinstance(diagnostics, Mapping) or diagnostic not in diagnostics:
        raise VestConfigurationError(f"No canonical VEST diagnostic configuration for {diagnostic!r}")
    return diagnostics[diagnostic], f"VEST diagnostic {diagnostic!r}"


def resolve_vest_processing_config(
    shot: int,
    diagnostic: str,
    *,
    overrides: Mapping[str, Any] | None = None,
    info_file: str | os.PathLike[str] | None = None,
    calibration_override_reason: str | None = None,
    override_source: str = "workflow override",
) -> ResolvedVestConfiguration:
    """Resolve, validate and fingerprint one diagnostic's configuration for ``shot``.

    ``diagnostic`` names a block of ``0.diagnostics`` in vest.yaml, or
    ``"impa"`` for the top-level IMPA section. Every nested ``revisions``
    table is resolved for the shot (the same tables the mappers resolve), so
    ``effective`` is what the processing actually reads. For IMPA the
    ``calibration.shot_era_overrides`` table is validated through
    :func:`impa_shot_era` and left in place; the matched era is recorded in
    the provenance.

    ``overrides`` is a nested mapping in the shape of ``effective``; see the
    module docstring for the rules. ``override_source`` names where it came
    from (for example the workflow config file) in the provenance.
    """
    numeric_shot = _shot_number(shot, name="shot", context=f"VEST {diagnostic!r}")
    source = vest_configuration_source(info_file)
    document = _policy_document(info_file)
    if diagnostic == "impa":
        from .impa import resolve_impa_config

        block, context = resolve_impa_config(numeric_shot, info_file=info_file), "impa"
    else:
        block, context = _raw_block(document, diagnostic)
    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"{context} must be a mapping")

    origins: dict[tuple[Any, ...], dict[str, Any]] = {}
    applied: list[dict[str, Any]] = []
    effective = _resolve_tree(
        block, numeric_shot, context=context, path=(), origins=origins, applied=applied
    )
    if diagnostic == "impa":
        era, era_provenance = impa_shot_era(effective.get("calibration") or {}, numeric_shot)
        era_record = {"path": "calibration.shot_era_overrides", **era_provenance}
        applied.append(era_record)
        if era is not None:
            # Apply the era the way the mapper does, so `effective` -- and the
            # fingerprint -- say which channels and which gain this shot uses.
            if era.get("gain") is not None:
                effective["calibration"]["gain"] = era["gain"]
                origins[("calibration", "gain")] = era_record
            for key in ("channels", "bz_channels"):
                wired = era.get(key)
                if wired:
                    allowed = {int(field) for field in wired}
                    effective[key] = {
                        index: entry
                        for index, entry in (effective.get(key) or {}).items()
                        if int(entry["field"]) in allowed
                    }
                    origins[(key,)] = era_record
    elif effective.get("calibration") is not None:
        validate_calibration(effective["calibration"])

    accepted: list[dict[str, Any]] = []
    if overrides:
        if not isinstance(overrides, Mapping):
            raise VestConfigurationError(f"{context}: overrides must be a mapping")
        if calibration_override_reason is not None and (
            not isinstance(calibration_override_reason, str) or not calibration_override_reason.strip()
        ):
            raise VestConfigurationError(
                f"{context}: calibration_override_reason must be a non-empty string"
            )
        changes = _validate_overrides(
            effective,
            overrides,
            context=context,
            calibration_override_reason=calibration_override_reason,
            accepted=accepted,
        )
        effective = _deep_merge(effective, changes, context=context)
        if diagnostic != "impa" and effective.get("calibration") is not None:
            validate_calibration(effective["calibration"])

    overridden = {entry["path"] for entry in accepted}
    keys: dict[str, Any] = {}
    for path, _value in _leaves(effective):
        dotted = _dotted(path)
        origin = _origin_for(path, origins)
        is_overridden = any(dotted == item or dotted.startswith(item + ".") for item in overridden)
        keys[dotted] = {
            "class": classify_config_path(path).value,
            "source": override_source if is_overridden else source["path"],
            "origin": "override" if is_overridden else ("revision" if origin else "base"),
            "revision_context": None if origin is None else origin["context"],
            "revision_index": None if origin is None else origin["revision_index"],
            "revision_bounds": None if origin is None else origin["revision_bounds"],
            "overridden": is_overridden,
        }

    provenance = {
        "schema": EFFECTIVE_CONFIG_SCHEMA,
        "source": source,
        "revisions": applied,
        "overrides": {
            "source": override_source if accepted else None,
            "accepted": accepted,
            "calibration_override_reason": calibration_override_reason if accepted else None,
        },
        "keys": keys,
        "vaft_version": _vaft_version(),
    }
    return ResolvedVestConfiguration(
        shot=numeric_shot,
        diagnostic=str(diagnostic),
        effective=_freeze(effective),
        provenance=_freeze(provenance),
        fingerprint=configuration_fingerprint(effective),
        source_sha256=source["sha256"],
    )


#: The ``0.diagnostics`` blocks a stage mapper reads through
#: ``resolve_vest_diagnostic`` -- what a manifest records by default. The
#: ``langmuir_probe_mid``/``_upper`` blocks are left out: the Langmuir mapper
#: reads the top-level ``langmuir_probes`` section (its own era table), so
#: recording them would describe a configuration nothing applied.
MAPPED_DIAGNOSTICS = (
    "barometry_main",
    "core_profiles",
    "diamagnetic_flux",
    "ech_6kw_forward",
    "ech_6kw_reflected",
    "equilibrium_magnetics",
    "fluctuation_mirnov",
    "pf_active",
    "plasma_current",
    "tf",
    "toroidal_mirnov_reference",
)


def vest_configuration_record(
    shot: int,
    *,
    diagnostics: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    info_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """The configuration identity a stage manifest records for ``shot`` (#2159).

    Each named diagnostic (default: :data:`MAPPED_DIAGNOSTICS`) contributes
    its effective configuration, fingerprint, applied revisions and accepted
    overrides; a diagnostic whose configuration does not resolve for the shot
    records the :class:`VestConfigurationError` instead of failing the stage
    -- its mapper reports the same failure in the channel status. ``extra`` adds
    already-resolved effective blocks (the magnetics processing dataclass, the
    time policies) as ``{name: effective}``; each is fingerprinted the same
    way. ``fingerprint`` identifies the whole record's effective values;
    ``source.sha256`` and ``vaft_version`` are recorded beside it, not in it.
    """
    numeric_shot = _shot_number(shot, name="shot", context="vest configuration record")
    if diagnostics is None:
        diagnostics = MAPPED_DIAGNOSTICS
    entries: dict[str, Any] = {}
    for name in diagnostics:
        try:
            resolved = resolve_vest_processing_config(numeric_shot, name, info_file=info_file)
        except VestConfigurationError as error:
            entries[str(name)] = {"error": str(error)}
            continue
        record = resolved.as_record()
        entries[str(name)] = {
            "fingerprint": record["fingerprint"],
            "effective": record["effective"],
            "revisions": record["provenance"]["revisions"],
            "overrides": record["provenance"]["overrides"],
        }
    for name, effective in (extra or {}).items():
        entries[str(name)] = {
            "fingerprint": configuration_fingerprint(effective),
            "effective": json.loads(canonical_json(effective)),
        }
    identity = {
        name: entry.get("fingerprint", {"error": entry.get("error")})
        for name, entry in entries.items()
    }
    return {
        "schema": EFFECTIVE_CONFIG_SCHEMA,
        "shot": numeric_shot,
        "source": vest_configuration_source(info_file),
        "vaft_version": _vaft_version(),
        "fingerprint": configuration_fingerprint(identity),
        "components": entries,
    }
