"""Semantic normalization and verification for paired reference samples."""

from __future__ import annotations

from fnmatch import fnmatchcase
import hashlib
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sample_view(data: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten an ODS and retain representation-independent semantic leaves."""
    if hasattr(data, "flat") and callable(data.flat):
        flat = {str(path): value for path, value in data.flat().items()}
    elif isinstance(data, Mapping):
        flat = {str(path): value for path, value in data.items()}
    else:
        raise TypeError("sample semantic data must be an ODS or flat path mapping")

    policy = manifest.get("semantic_comparison", {})
    include = tuple(policy.get("paths", ["*"]))
    exclude = tuple(policy.get("exclude_paths", []))
    return {
        path: value
        for path, value in flat.items()
        if any(fnmatchcase(path, pattern) for pattern in include)
        and not any(fnmatchcase(path, pattern) for pattern in exclude)
    }


def verify_sample_artifacts(
    sample_root: str | Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Verify all representation checksums and return computed digests."""
    root = Path(sample_root)
    actual: dict[str, str] = {}
    for representation, record in manifest["representations"].items():
        path = root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {representation} sample artifact: {path}")
        digest = sha256_file(path)
        actual[representation] = digest
        if digest != record["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {representation} sample artifact {path}: "
                f"expected {record['sha256']}, got {digest}"
            )
        expected_size = record.get("size")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise ValueError(
                f"Size mismatch for {representation} sample artifact {path}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
    return actual


__all__ = ["semantic_sample_view", "sha256_file", "verify_sample_artifacts"]
