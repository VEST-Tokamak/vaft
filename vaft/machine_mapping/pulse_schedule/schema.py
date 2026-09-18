"""The ShotLog era schemas and their registry.

A schema describes one generation of the workbook template: which headers
identify it and which fields are promoted from raw cells to structured values.
They ship as package data (``vaft/data/shotlog/schemas``) so that what a
conversion recognised is pinned by the vaft version that ran it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REGISTRY_VERSION = 2
SCHEMA_KEYS = ("schema_id", "version", "status", "detection", "fields")
FIELD_KEYS = ("path", "type", "required", "meaning_ko", "aliases", "parsing", "validation")


class SchemaError(ValueError):
    """Raised when a packaged or caller-supplied schema is malformed."""


def _schema_directory() -> Path:
    from vaft.data.resources import data_path

    return data_path("shotlog/schemas")


def _validate(metadata: Any, document: Path) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise SchemaError(f"{document}: a schema must be a mapping")
    for key in SCHEMA_KEYS:
        if key not in metadata:
            raise SchemaError(f"{document}: missing required key {key!r}")
    if not isinstance(metadata["fields"], list):
        raise SchemaError(f"{document}: fields must be a list")
    for field in metadata["fields"]:
        for key in FIELD_KEYS:
            if key not in field:
                raise SchemaError(f"{document}: field {field.get('path')!r} is missing {key!r}")
        # The physical discharge is always `shot` in this data model.
        if "shot_no" in field["path"]:
            raise SchemaError(f"{document}: prohibited key shot_no in {field['path']}")
    return metadata


def build_registry(directory: str | Path | None = None) -> dict[str, Any]:
    """Load every ``*.yaml`` schema in ``directory`` into one registry."""
    root = Path(directory) if directory is not None else _schema_directory()
    documents = sorted(root.glob("*.yaml"))
    if not documents:
        raise SchemaError(f"No ShotLog schema documents found in {root}")
    common: dict[str, Any] | None = None
    schemas: dict[str, Any] = {}
    for document in documents:
        metadata = _validate(yaml.safe_load(document.read_text(encoding="utf-8")), document)
        schema_id = metadata["schema_id"]
        if schema_id == "common":
            common = metadata
        elif schema_id in schemas:
            raise SchemaError(f"Duplicate ShotLog schema id {schema_id!r}")
        else:
            schemas[schema_id] = metadata
    if common is None:
        raise SchemaError("A common ShotLog schema document is required")
    return {"registry_version": REGISTRY_VERSION, "common": common, "schemas": schemas}


@lru_cache(maxsize=1)
def packaged_registry() -> dict[str, Any]:
    """The registry shipped with this vaft version (cached; do not mutate)."""
    return build_registry()


def schema_versions(registry: dict[str, Any]) -> dict[str, str]:
    """``{schema_id: version}``, recorded with every conversion."""
    return {schema_id: schema["version"] for schema_id, schema in registry["schemas"].items()}
