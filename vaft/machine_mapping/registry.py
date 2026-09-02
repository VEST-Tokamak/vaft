"""Validation and deterministic export for the VEST diagnostic registry.

The registry is deliberately separate from the shot-resolved mapping settings in
``vest.yaml``.  It describes ownership and data availability; it never changes
the runtime configuration returned by :func:`resolve_vest_diagnostic`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import package_data_path


REGISTRY_KEY = "diagnostic_registry"
AVAILABILITY_VALUES = frozenset({"Routine", "If-requested", "Retired"})
LIFECYCLE_VALUES = frozenset({"available", "in_maintenance", "retired"})
MAPPING_STATUS_VALUES = frozenset({"implemented", "partial", "not_implemented"})
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class DiagnosticRegistryError(ValueError):
    """Raised when the diagnostic registry has an invalid schema."""


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
