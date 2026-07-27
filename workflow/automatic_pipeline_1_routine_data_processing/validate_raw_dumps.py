#!/usr/bin/env python3
"""Classify raw DAQ archives before launching per-shot processing."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
from typing import Any, Iterable


class RawDumpValidationError(ValueError):
    """Raised when an archive cannot be trusted as a completed raw DB dump."""


_OUTPUT_SHOT = re.compile(r"vest_(?P<shot>\d+)_daq_raw\.json\.gz$")


def _read_dump(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RawDumpValidationError(f"Cannot read raw dump {path}: {exc}") from exc

    try:
        shot = int(payload["shot"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RawDumpValidationError(f"Raw dump {path} has no valid shot number") from exc
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise RawDumpValidationError(f"Raw dump {path} has no fields mapping")
    return shot, fields


def _missing_required_fields(fields: dict[str, Any], required_fields: Iterable[int], min_samples: int) -> list[int]:
    missing: list[int] = []
    for field in required_fields:
        entry = fields.get(str(field))
        data = entry.get("data") if isinstance(entry, dict) else None
        if not isinstance(data, list) or len(data) < min_samples:
            missing.append(int(field))
    return missing


def validate_raw_dumps(
    dump_paths: Iterable[str | Path],
    required_fields: Iterable[int],
    min_samples: int = 2,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return eligible shots and reports for archives lacking required DAQ signals.

    A malformed archive is an acquisition/export failure and raises. A readable
    archive missing required field data is an incomplete shot and is reported as
    excluded without failing the batch.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    required = sorted({int(field) for field in required_fields})
    if not required:
        raise ValueError("At least one required raw field must be configured")

    eligible: list[int] = []
    excluded: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_path in sorted(Path(path) for path in dump_paths):
        shot, fields = _read_dump(raw_path)
        expected = _OUTPUT_SHOT.search(raw_path.name)
        if expected is not None and int(expected.group("shot")) != shot:
            raise RawDumpValidationError(
                f"Raw dump {raw_path} is named for shot {expected.group('shot')} but contains shot {shot}"
            )
        if shot in seen:
            raise RawDumpValidationError(f"Duplicate raw dumps for shot {shot}")
        seen.add(shot)
        missing = _missing_required_fields(fields, required, min_samples)
        if missing:
            excluded.append(
                {
                    "shot": shot,
                    "reason": "missing_required_raw_signal",
                    "missing_field_codes": missing,
                    "raw_dump": str(raw_path),
                }
            )
        else:
            eligible.append(shot)
    return eligible, excluded


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dumps", nargs="+", type=Path)
    parser.add_argument("--required-fields", required=True, help="Comma-separated required raw field codes.")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--eligible-output", required=True, type=Path)
    parser.add_argument("--excluded-output", required=True, type=Path)
    args = parser.parse_args()

    required_fields = [int(field) for field in args.required_fields.split(",") if field.strip()]
    eligible, excluded = validate_raw_dumps(args.raw_dumps, required_fields, args.min_samples)
    _write_json(args.eligible_output, {"eligible_shots": eligible})
    _write_json(args.excluded_output, {"excluded_shots": excluded})
    print(f"Raw preflight: {len(eligible)} eligible, {len(excluded)} excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
