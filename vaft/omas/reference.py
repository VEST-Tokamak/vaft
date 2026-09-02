"""Reference-manifest loading and checksum verification utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ReferenceManifestError(ValueError):
    """Raised when a reference manifest does not satisfy schema version 1."""


@dataclass(frozen=True)
class ArtifactVerification:
    """Checksum verification result for one reference artifact."""

    artifact_id: str
    path: str
    storage: str
    status: str
    expected_size: int | None
    actual_size: int | None
    expected_sha256: str
    actual_sha256: str | None

    @property
    def valid(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"valid": self.valid}


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 checksum for a local artifact."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_manifest(source: str | Path) -> dict[str, Any]:
    """Load and minimally validate a VEST reference manifest."""

    path = Path(source).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, Mapping):
        raise ReferenceManifestError("Reference manifest root must be a mapping")
    if int(manifest.get("schema_version", 0)) != 1:
        raise ReferenceManifestError("Reference manifest schema_version must be 1")
    if not manifest.get("reference_id"):
        raise ReferenceManifestError("Reference manifest must define reference_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ReferenceManifestError("Reference manifest must define artifacts")
    seen: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ReferenceManifestError(f"Artifact {index} must be a mapping")
        missing = [
            name
            for name in ("id", "path", "storage", "sha256")
            if not artifact.get(name)
        ]
        if missing:
            raise ReferenceManifestError(
                f"Artifact {index} is missing required fields: {', '.join(missing)}"
            )
        artifact_id = str(artifact["id"])
        if artifact_id in seen:
            raise ReferenceManifestError(f"Duplicate artifact id: {artifact_id}")
        seen.add(artifact_id)
        checksum = str(artifact["sha256"])
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ReferenceManifestError(
                f"Artifact {artifact_id} has an invalid SHA-256 checksum"
            )
        if artifact["storage"] not in {"repository", "external"}:
            raise ReferenceManifestError(
                f"Artifact {artifact_id} storage must be repository or external"
            )
    return dict(manifest)


def verify_reference_artifacts(
    manifest_source: str | Path,
    *,
    root: str | Path | None = None,
    storage: Iterable[str] = ("repository",),
) -> tuple[ArtifactVerification, ...]:
    """Verify selected manifest artifacts without accessing remote systems.

    Relative artifact paths resolve against ``root`` or, by default, against
    the manifest directory.  External artifacts are skipped unless explicitly
    selected and their path exists locally.
    """

    manifest_path = Path(manifest_source).expanduser()
    manifest = load_reference_manifest(manifest_path)
    base = Path(root).expanduser() if root is not None else manifest_path.parent
    selected_storage = set(storage)
    results: list[ArtifactVerification] = []
    for artifact in manifest["artifacts"]:
        if artifact["storage"] not in selected_storage:
            continue
        declared = Path(str(artifact["path"])).expanduser()
        path = declared if declared.is_absolute() else base / declared
        expected_size = (
            None if artifact.get("size") is None else int(artifact["size"])
        )
        if not path.is_file():
            results.append(
                ArtifactVerification(
                    artifact_id=str(artifact["id"]),
                    path=str(path),
                    storage=str(artifact["storage"]),
                    status="missing",
                    expected_size=expected_size,
                    actual_size=None,
                    expected_sha256=str(artifact["sha256"]),
                    actual_sha256=None,
                )
            )
            continue
        actual_size = path.stat().st_size
        actual_checksum = sha256_file(path)
        status = "verified"
        if expected_size is not None and actual_size != expected_size:
            status = "size_mismatch"
        elif actual_checksum != artifact["sha256"]:
            status = "checksum_mismatch"
        results.append(
            ArtifactVerification(
                artifact_id=str(artifact["id"]),
                path=str(path),
                storage=str(artifact["storage"]),
                status=status,
                expected_size=expected_size,
                actual_size=actual_size,
                expected_sha256=str(artifact["sha256"]),
                actual_sha256=actual_checksum,
            )
        )
    return tuple(results)


__all__ = [
    "ArtifactVerification",
    "ReferenceManifestError",
    "load_reference_manifest",
    "sha256_file",
    "verify_reference_artifacts",
]
