"""Validation and deterministic export for the VEST diagnostic registry.

The registry is deliberately separate from the shot-resolved mapping settings in
``vest.yaml``.  It describes ownership and data availability; it never changes
the runtime configuration returned by :func:`resolve_vest_diagnostic`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from .conventions import port_toroidal_angle, vest_clock_angle
from .utils import package_data_path


REGISTRY_KEY = "diagnostic_registry"
AVAILABILITY_VALUES = frozenset({"Routine", "If-requested", "Retired"})
LIFECYCLE_VALUES = frozenset({"available", "in_maintenance", "retired"})
MAPPING_STATUS_VALUES = frozenset({"implemented", "partial", "not_implemented"})
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


PORT_MAP_KEY = "port_map"
#: ``<clock><location1>[<location2>][<size>]``.  Size is absent on the ports the
#: document lists without one (``2U``, ``6L``), and ``R`` marks a rectangular
#: port.  See the ``port_map`` header in ``vest.yaml`` for the grammar.
_PORT_NAME = re.compile(
    r"^(?P<clock>1[0-2]|[1-9])"
    r"(?P<location1>[MULTB])"
    r"(?P<location2>[MUL])?"
    r"(?P<size>R|4\.5|4|6|10|12)?$"
)
_CHAMBERS = {"main": "M", "upper": "U", "lower": "L", "top": "T", "bottom": "B"}
_TIERS = {"middle": "M", "upper": "U", "lower": "L"}


class DiagnosticRegistryError(ValueError):
    """Raised when the diagnostic registry has an invalid schema."""


class PortMapError(ValueError):
    """Raised when the VEST port table has an invalid schema."""


def registry_path(path: str | Path | None = None) -> Path:
    """Return the registry-bearing VEST YAML file."""
    return Path(path) if path is not None else Path(package_data_path("vest.yaml"))


def load_diagnostic_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate the top-level VEST diagnostic registry."""
    source = registry_path(path)
    content = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    registry = content.get(REGISTRY_KEY)
    if not isinstance(registry, Mapping):
        raise DiagnosticRegistryError(f"{source}: {REGISTRY_KEY} must be a mapping")
    normalized = {str(identifier): dict(record) for identifier, record in registry.items()}
    validate_diagnostic_registry(normalized)
    return normalized


def validate_diagnostic_registry(registry: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate the schema shared by runtime checks and documentation export."""
    seen_names: set[str] = set()
    for identifier, record in registry.items():
        context = f"diagnostic_registry.{identifier}"
        if not _IDENTIFIER.fullmatch(identifier):
            raise DiagnosticRegistryError(f"{context}: identifier must be stable lowercase text")
        if not isinstance(record, Mapping):
            raise DiagnosticRegistryError(f"{context}: record must be a mapping")
        for field in ("name", "family", "category", "ids", "ids_path", "source", "availability",
                      "lifecycle", "mapping_status", "quantities", "responsible"):
            if field not in record:
                raise DiagnosticRegistryError(f"{context}: missing {field}")
        name = record["name"]
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            raise DiagnosticRegistryError(f"{context}: name must be unique non-empty text")
        seen_names.add(name)
        if record["availability"] not in AVAILABILITY_VALUES:
            raise DiagnosticRegistryError(f"{context}: invalid availability")
        if record["lifecycle"] not in LIFECYCLE_VALUES:
            raise DiagnosticRegistryError(f"{context}: invalid lifecycle")
        if record["mapping_status"] not in MAPPING_STATUS_VALUES:
            raise DiagnosticRegistryError(f"{context}: invalid mapping_status")
        quantities = record["quantities"]
        if not isinstance(quantities, Mapping) or set(quantities) != {"static", "measured", "derived"}:
            raise DiagnosticRegistryError(f"{context}: quantities must contain static, measured, derived")
        if any(not isinstance(quantities[key], list) for key in quantities):
            raise DiagnosticRegistryError(f"{context}: quantity values must be lists")
        responsible = record["responsible"]
        if not isinstance(responsible, list):
            raise DiagnosticRegistryError(f"{context}: responsible must be a list")
        for person in responsible:
            if not isinstance(person, Mapping) or not isinstance(person.get("name"), str) or not person["name"].strip():
                raise DiagnosticRegistryError(f"{context}: responsible people require names")
            if "email" in person and not _EMAIL.fullmatch(str(person["email"])):
                raise DiagnosticRegistryError(f"{context}: invalid responsible email")
        source = record["source"]
        if not isinstance(source, Mapping) or not isinstance(source.get("type"), str):
            raise DiagnosticRegistryError(f"{context}: source requires a type")
        if record["mapping_status"] == "implemented":
            module = record.get("mapping")
            if not isinstance(module, Mapping) or not module.get("module") or not module.get("entrypoint"):
                raise DiagnosticRegistryError(f"{context}: implemented mappings require module and entrypoint")
            if source["type"] == "raw_daq" and set(source.get("backends", [])) != {"mysql", "archived_raw_dump"}:
                raise DiagnosticRegistryError(f"{context}: raw_daq requires mysql and archived_raw_dump backends")
            if source["type"] == "file" and (not source.get("formats") or not source.get("patterns")):
                raise DiagnosticRegistryError(f"{context}: file source requires formats and patterns")


@lru_cache(maxsize=4)
def _load_port_map_cached(source: Path) -> tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]:
    """Parse and validate the port table once per file.

    Cached because :func:`port_phi` is called per diagnostic channel, and
    without this every call re-read and re-validated the whole of ``vest.yaml``
    -- about 90 ms each, which a per-channel loop pays once per channel.  The
    same ``lru_cache`` treatment the channel tables in
    :mod:`vaft.machine_mapping.magnetics` get, for the same reason.

    Hashable items are returned so the cache cannot hand out a shared mutable
    dict that a caller could corrupt for everyone else.
    """
    content = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    block = content.get(PORT_MAP_KEY)
    if not isinstance(block, Mapping):
        raise PortMapError(f"{source}: {PORT_MAP_KEY} must be a mapping")
    ports = block.get("ports")
    if not isinstance(ports, list) or not ports:
        raise PortMapError(f"{source}: {PORT_MAP_KEY}.ports must be a non-empty list")

    normalized: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(ports):
        context = f"{source}: {PORT_MAP_KEY}.ports[{index}]"
        if not isinstance(record, Mapping):
            raise PortMapError(f"{context}: entry must be a mapping")
        if "name" not in record:
            raise PortMapError(f"{context}: missing name")
        name = str(record["name"])
        if name in normalized:
            raise PortMapError(f"{context}: duplicate port name {name!r}")
        normalized[name] = dict(record)

    validate_port_map(normalized)
    return tuple((name, tuple(record.items())) for name, record in normalized.items())


def load_port_map(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate the VEST port table, keyed by port name.

    The returned records carry the ``clock`` position only.  A toroidal angle is
    *derived* (:func:`port_phi`), never stored, so the table and the convention
    cannot drift apart.

    A fresh dict is built on each call, so a caller may mutate the result
    without disturbing the cached parse behind it.
    """
    return {
        name: dict(items)
        for name, items in _load_port_map_cached(registry_path(path))
    }


def validate_port_map(ports: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate the port table, including name/field agreement.

    The port name encodes the clock position, chamber and tier, and the record
    repeats them as fields.  Checking that the two agree is what stops a typo in
    either half from silently relocating a diagnostic: a name says where the
    hardware is, and nothing else in the file would notice a mismatch.
    """
    for name, record in ports.items():
        context = f"{PORT_MAP_KEY}.{name}"
        if not isinstance(record, Mapping):
            raise PortMapError(f"{context}: record must be a mapping")
        match = _PORT_NAME.fullmatch(name)
        if match is None:
            raise PortMapError(f"{context}: name does not match the VEST port grammar")
        for field in ("clock", "chamber", "use"):
            if field not in record:
                raise PortMapError(f"{context}: missing {field}")
        clock = record["clock"]
        if not isinstance(clock, (int, float)) or isinstance(clock, bool) or not 1 <= float(clock) <= 12:
            raise PortMapError(f"{context}: clock must be a position in 1..12")
        if float(clock) != float(match.group("clock")):
            raise PortMapError(
                f"{context}: clock {clock} disagrees with the {match.group('clock')} "
                "o'clock encoded in the port name"
            )
        chamber = record["chamber"]
        if chamber not in _CHAMBERS:
            raise PortMapError(f"{context}: unknown chamber {chamber!r}")
        if _CHAMBERS[chamber] != match.group("location1"):
            raise PortMapError(
                f"{context}: chamber {chamber!r} disagrees with {match.group('location1')!r} in the name"
            )
        tier = record.get("tier")
        if tier is not None and chamber != "main":
            raise PortMapError(f"{context}: tier is meaningful only on the main chamber")
        if (_TIERS.get(tier) if tier is not None else None) != match.group("location2"):
            raise PortMapError(f"{context}: tier {tier!r} disagrees with the port name")
        if not isinstance(record["use"], str) or not record["use"].strip():
            raise PortMapError(f"{context}: use must be non-empty text")


def port_clock(name: str, path: str | Path | None = None) -> float:
    """The clock position of a named port.  Clockwise-positive, not IMAS phi."""
    for port_name, items in _load_port_map_cached(registry_path(path)):
        if port_name == name:
            return float(dict(items)["clock"])
    raise PortMapError(f"{PORT_MAP_KEY}: no port named {name!r}")


def port_clock_angle(name: str, path: str | Path | None = None) -> float:
    """The VEST clock angle of a named port, in degrees, clockwise-positive."""
    return vest_clock_angle(port_clock(name, path))


def port_phi(name: str, path: str | Path | None = None) -> float:
    """The IMAS toroidal angle of a named port, in radians.

    This is the one function a mapper should call to place a diagnostic: it
    carries the clockwise-to-counter-clockwise negation that
    :func:`vaft.machine_mapping.conventions.port_toroidal_angle` documents.
    """
    return port_toroidal_angle(port_clock(name, path))


def documentation_snapshot(
    path: str | Path | None = None,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the deterministic, publishable representation of the registry.

    ``provenance`` records which source tree the snapshot describes -- the
    commit and ref the documentation build extracted -- and is omitted entirely
    when it is not supplied, so the default output stays byte-for-byte what it
    has always been.
    """
    source = registry_path(path)
    registry = load_diagnostic_registry(source)
    diagnostics = []
    for identifier in sorted(registry):
        record = registry[identifier]
        diagnostics.append(
            {
                "id": identifier,
                **{field: record[field] for field in (
                    "name", "family", "category", "ids", "ids_path", "responsible", "source",
                    "availability", "lifecycle", "mapping_status",
                )},
            }
        )
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "path": "vaft/machine_mapping/vest.yaml",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "diagnostics": diagnostics,
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def export_documentation_snapshot(
    output: str | Path,
    path: str | Path | None = None,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Write a normalized YAML documentation snapshot and return its path."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            documentation_snapshot(path, provenance),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=True,
        ),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the VEST diagnostic registry for the documentation site."
    )
    parser.add_argument("--output", required=True, help="YAML destination for the generated snapshot")
    parser.add_argument("--registry", help="Override the packaged vest.yaml path")
    parser.add_argument(
        "--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot"
    )
    parser.add_argument(
        "--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot"
    )
    arguments = parser.parse_args(argv)
    provenance = {
        key: value
        for key, value in (
            ("commit", arguments.provenance_commit),
            ("ref", arguments.provenance_ref),
        )
        if value
    }
    export_documentation_snapshot(arguments.output, arguments.registry, provenance or None)


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
