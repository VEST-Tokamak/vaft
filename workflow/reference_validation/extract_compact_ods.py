#!/usr/bin/env python3
"""Extract a deterministic compact ODS fixture from a full legacy artifact."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import tempfile

from omas import ODS, omas_environment

from vaft.omas.reference import load_reference_manifest, sha256_file


def extract_compact_fixture(
    source: Path,
    manifest_path: Path,
    fixture_id: str,
    output: Path | None = None,
) -> Path:
    manifest = load_reference_manifest(manifest_path)
    fixtures = {
        str(item["id"]): item for item in manifest.get("compact_fixtures", [])
    }
    if fixture_id not in fixtures:
        raise ValueError(f"Unknown compact fixture id: {fixture_id}")
    fixture = fixtures[fixture_id]
    expected_source = str(fixture["source_sha256"])
    actual_source = sha256_file(source)
    if actual_source != expected_source:
        raise ValueError(
            f"Source checksum mismatch: expected {expected_source}, got {actual_source}"
        )

    source_ods = ODS(consistency_check=False).load(
        str(source), consistency_check=False
    )
    flat = source_ods.flat()
    selectors = tuple(str(item) for item in fixture["selectors"])
    selected = sorted(
        path
        for path in flat
        if any(_matches(path, selector) for selector in selectors)
    )
    missing = [
        selector
        for selector in selectors
        if not any(_matches(path, selector) for path in flat)
    ]
    if missing:
        raise ValueError("Selectors matched no ODS paths: " + ", ".join(missing))

    compact = ODS(consistency_check=False)
    with omas_environment(compact, dynamic_path_creation="dynamic_array_structures"):
        for path in selected:
            compact[path] = flat[path]

    target = output or manifest_path.parent / str(fixture["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vaft-reference-") as temporary:
        plain = Path(temporary) / "fixture.json"
        compact.save(str(plain))
        with plain.open("rb") as source_handle, target.open("wb") as target_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=target_handle,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                    compressed.write(chunk)
    print(f"Wrote {target}")
    print(f"Selected leaves: {len(selected)}")
    print(f"Size: {target.stat().st_size}")
    print(f"SHA-256: {sha256_file(target)}")
    return target


def _matches(path: str, selector: str) -> bool:
    from fnmatch import fnmatchcase

    return fnmatchcase(path, selector)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Full legacy OMAS JSON artifact")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("test/data/vest_reference/manifest.yaml"),
    )
    parser.add_argument("--fixture-id", default="shot-39915-combined-compact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    extract_compact_fixture(
        args.source.expanduser(),
        args.manifest.expanduser(),
        args.fixture_id,
        args.output.expanduser() if args.output is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
