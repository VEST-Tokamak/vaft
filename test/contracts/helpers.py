"""Helpers for validating canonical OMAS contract paths on plain payloads."""

from __future__ import annotations

from typing import Any, Iterable


def get_path(payload: Any, path: str) -> Any:
    """Resolve a dotted path across nested dict/list payloads."""
    current = payload
    for token in path.split("."):
        if isinstance(current, list):
            current = current[int(token)]
            continue
        if isinstance(current, dict):
            current = current[token]
            continue
        raise KeyError(path)
    return current


def path_exists(payload: Any, path: str) -> bool:
    """Return whether a dotted path exists in the nested payload."""
    try:
        get_path(payload, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def first_existing_path(payload: Any, paths: Iterable[str]) -> str | None:
    """Return the first existing path from a list of alternatives."""
    for path in paths:
        if path_exists(payload, path):
            return path
    return None


def sequence_length(value: Any) -> int | None:
    """Return the length of a list-like value, or ``None`` for scalars."""
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def validate_required_paths(payload: Any, required_paths: Iterable[str]) -> list[str]:
    """Collect missing required dotted paths."""
    errors = []
    for path in required_paths:
        if not path_exists(payload, path):
            errors.append(f"missing path: {path}")
    return errors


def validate_expected_values(payload: Any, expected_values: dict[str, Any]) -> list[str]:
    """Collect value mismatches for exact-value contract leaves."""
    errors = []
    for path, expected in expected_values.items():
        if not path_exists(payload, path):
            errors.append(f"missing path: {path}")
            continue
        actual = get_path(payload, path)
        if actual != expected:
            errors.append(f"unexpected value at {path}: expected {expected!r}, got {actual!r}")
    return errors


def validate_series(payload: Any, series_specs: Iterable[dict[str, Any]]) -> list[str]:
    """Collect time-axis and sequence-length mismatches for time series."""
    errors = []
    for spec in series_specs:
        data_path = spec["data_path"]
        time_paths = spec["time_paths"]
        allow_scalar = spec.get("allow_scalar", False)

        if not path_exists(payload, data_path):
            errors.append(f"missing path: {data_path}")
            continue

        time_path = first_existing_path(payload, time_paths)
        if time_path is None:
            joined = ", ".join(time_paths)
            errors.append(f"missing governing time path for {data_path}: one of [{joined}]")
            continue

        data_value = get_path(payload, data_path)
        time_value = get_path(payload, time_path)
        data_length = sequence_length(data_value)
        time_length = sequence_length(time_value)

        if time_length is None:
            errors.append(f"time axis is not sequence-like: {time_path}")
            continue
        if data_length is None:
            if not allow_scalar:
                errors.append(f"data is not sequence-like: {data_path}")
            continue
        if data_length != time_length:
            errors.append(
                f"length mismatch for {data_path}: data={data_length}, time={time_length} via {time_path}"
            )
    return errors


def validate_ids_contract(payload: Any, ids_spec: dict[str, Any], strict_values: bool = True) -> list[str]:
    """Validate one IDS contract spec against a payload."""
    errors = []
    errors.extend(validate_required_paths(payload, ids_spec.get("required_paths", [])))
    if strict_values:
        errors.extend(validate_expected_values(payload, ids_spec.get("expected_values", {})))
    errors.extend(validate_series(payload, ids_spec.get("series", [])))
    return errors


def validate_contract(
    payload: Any,
    specs: dict[str, dict[str, Any]],
    ids_names: Iterable[str] | None = None,
    strict_values: bool = True,
) -> dict[str, list[str]]:
    """Validate multiple IDS specs and return only failing entries."""
    failures: dict[str, list[str]] = {}
    targets = tuple(ids_names) if ids_names is not None else tuple(specs.keys())
    for ids_name in targets:
        errors = validate_ids_contract(payload, specs[ids_name], strict_values=strict_values)
        if errors:
            failures[ids_name] = errors
    return failures


def format_failures(failures: dict[str, list[str]]) -> str:
    """Pretty-print validation failures for assertion messages."""
    if not failures:
        return ""
    lines = []
    for ids_name in sorted(failures):
        lines.append(f"[{ids_name}]")
        for error in failures[ids_name]:
            lines.append(f"- {error}")
    return "\n".join(lines)
