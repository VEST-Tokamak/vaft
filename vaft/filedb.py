"""Canonical local FileDB paths and read-only legacy-layout auditing.

The canonical resolver is deliberately pure: resolving a path never creates a
directory or touches an existing artifact.  The old shot-first layout is
available only through explicitly named read-only APIs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


class FileDBError(ValueError):
    """Base error for invalid FileDB configuration or path requests."""


class FileDBConfigError(FileDBError):
    """Raised when the FileDB storage root cannot be configured."""


class FileDBPathError(FileDBError):
    """Raised when a path request is outside the canonical grammar."""


class FileDBDomain(str, Enum):
    RAW = "raw"
    LEGACY = "legacy"
    OMAS = "omas"
    EFIT = "efit"
    CHEASE = "chease"
    GPEC = "gpec"


class OMASStage(str, Enum):
    STATIC = "static"
    DIAGNOSTICS = "diagnostics"
    EDDY = "eddy"
    EFIT = "efit"
    CHEASE = "chease"


class GPECCode(str, Enum):
    DCON = "dcon"
    RDCON = "rdcon"
    STRIDE = "stride"
    IDEAL_GPEC = "ideal-gpec"


class ArtifactClass(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    LOG = "log"
    PLOT = "plot"
    CONFIG = "config"
    WORK = "work"
    METADATA = "metadata"


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_REFERENCE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
_LEGACY_AREAS = {
    "diagnostics",
    "omas",
    "efit",
    "chease",
    "linear_stability",
    "logs",
}
_DEFAULT_EXPECTED_PRODUCTS = {
    "raw_dump": "diagnostics/vest_{shot}_daq_raw.json.gz",
    "diagnostics_ods": "omas/{shot}_diagnostics.json",
    "eddy_ods": "omas/{shot}_eddy.json",
    "efit_ods": "omas/{shot}_efit.json",
    "chease_ods": "omas/{shot}_chease.json",
}


def _choices(enum: type[Enum]) -> str:
    return ", ".join(member.value for member in enum)


def _enum_value(value: Any, enum: type[Enum], label: str) -> str:
    try:
        return enum(value).value
    except (TypeError, ValueError) as exc:
        raise FileDBPathError(
            f"Invalid {label} {value!r}; expected one of: {_choices(enum)}"
        ) from exc


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise FileDBPathError(f"{label} must be a positive integer, not a boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise FileDBPathError(
            f"{label} must be a positive integer; got {value!r}"
        ) from exc
    if str(value).strip() != str(number) or number <= 0:
        raise FileDBPathError(f"{label} must be a positive integer; got {value!r}")
    return number


def _component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise FileDBPathError(
            f"{label} must be one safe path component containing only letters, "
            f"numbers, '.', '_' or '-'; got {value!r}"
        )
    if value in {".", ".."}:
        raise FileDBPathError(f"{label} cannot be {value!r}")
    return value


def _absent(value: Any, label: str, domain: str) -> None:
    if value is not None:
        raise FileDBPathError(f"{label} is not valid for FileDB domain {domain!r}")


def _expand_environment(value: str, environment: Mapping[str, str]) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        if name not in environment:
            missing.add(name)
            return match.group(0)
        return environment[name]

    expanded = _ENV_REFERENCE.sub(replace, value)
    if missing:
        names = ", ".join(sorted(missing))
        raise FileDBConfigError(
            f"FileDB root references missing environment variable(s): {names}. "
            "Set VAFT_FILEDB_DIR or provide filedb.root explicitly."
        )
    return expanded


@dataclass(frozen=True)
class LegacyResolution:
    """An explicitly read-only path into the former shot-first layout."""

    path: Path
    exists: bool
    read_only: bool = True
    layout: str = "legacy-shot-first"

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __str__(self) -> str:
        return str(self.path)


class FileDB:
    """Resolve canonical OMAS-first FileDB paths without filesystem writes."""

    canonical_environment_variable = "VAFT_FILEDB_DIR"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        raw_root = os.fspath(root)
        if not raw_root or not raw_root.strip():
            raise FileDBConfigError("FileDB root must be a non-empty filesystem path")
        self.root = Path(raw_root).expanduser()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "FileDB":
        """Resolve root precedence: explicit config, canonical environment."""

        config = {} if config is None else config
        environment = os.environ if environment is None else environment
        filedb = config.get("filedb", {})
        if filedb is None:
            filedb = {}
        if not isinstance(filedb, Mapping):
            raise FileDBConfigError("filedb configuration must be a mapping")
        configured = filedb.get("root", config.get("filedb_root"))
        if configured is None:
            configured = environment.get(cls.canonical_environment_variable)
        if configured is None:
            raise FileDBConfigError(
                "FileDB root is not configured. Set filedb.root or "
                f"{cls.canonical_environment_variable}."
            )
        if not isinstance(configured, (str, os.PathLike)):
            raise FileDBConfigError("FileDB root must be a filesystem path")
        expanded = _expand_environment(os.fspath(configured), environment)
        return cls(expanded)

    def resolve(
        self,
        domain: str | FileDBDomain,
        *,
        subdomain: str | OMASStage | None = None,
        shot: int | str | None = None,
        machine_version: str | None = None,
        code: str | GPECCode | None = None,
        mode: int | str | None = None,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        """Return a path from the canonical grammar without creating it."""

        domain_value = _enum_value(domain, FileDBDomain, "domain")
        artifact_value = (
            None
            if artifact is None
            else _enum_value(artifact, ArtifactClass, "artifact class")
        )

        if domain_value == FileDBDomain.RAW.value:
            _absent(subdomain, "subdomain", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            path = self.root / domain_value / str(_positive_integer(shot, "shot"))

        elif domain_value == FileDBDomain.LEGACY.value:
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            diagnostic = _component(subdomain, "legacy diagnostic")
            path = (
                self.root
                / domain_value
                / diagnostic
                / str(_positive_integer(shot, "shot"))
            )

        elif domain_value == FileDBDomain.OMAS.value:
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            stage = _enum_value(subdomain, OMASStage, "OMAS subdomain")
            if stage == OMASStage.STATIC.value:
                _absent(shot, "shot", "omas/static")
                version = _component(machine_version, "machine_version")
                path = self.root / domain_value / stage / version
            else:
                _absent(machine_version, "machine_version", f"omas/{stage}")
                path = (
                    self.root
                    / domain_value
                    / stage
                    / str(_positive_integer(shot, "shot"))
                )

        elif domain_value in {FileDBDomain.EFIT.value, FileDBDomain.CHEASE.value}:
            _absent(subdomain, "subdomain", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            path = self.root / domain_value / str(_positive_integer(shot, "shot"))

        else:
            _absent(subdomain, "subdomain", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            code_value = _enum_value(code, GPECCode, "GPEC code")
            path = (
                self.root
                / domain_value
                / code_value
                / str(_positive_integer(shot, "shot"))
                / f"n={_positive_integer(mode, 'toroidal mode')}"
            )

        return path if artifact_value is None else path / artifact_value

    def raw(
        self, shot: int | str, *, artifact: str | ArtifactClass | None = None
    ) -> Path:
        return self.resolve("raw", shot=shot, artifact=artifact)

    def legacy(
        self,
        diagnostic: str,
        shot: int | str,
        *,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve(
            "legacy", subdomain=diagnostic, shot=shot, artifact=artifact
        )

    def omas(
        self,
        stage: str | OMASStage,
        *,
        shot: int | str | None = None,
        machine_version: str | None = None,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve(
            "omas",
            subdomain=stage,
            shot=shot,
            machine_version=machine_version,
            artifact=artifact,
        )

    def efit(
        self, shot: int | str, *, artifact: str | ArtifactClass | None = None
    ) -> Path:
        return self.resolve("efit", shot=shot, artifact=artifact)

    def chease(
        self, shot: int | str, *, artifact: str | ArtifactClass | None = None
    ) -> Path:
        return self.resolve("chease", shot=shot, artifact=artifact)

    def gpec(
        self,
        code: str | GPECCode,
        shot: int | str,
        mode: int | str,
        *,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve("gpec", code=code, shot=shot, mode=mode, artifact=artifact)

    def resolve_legacy_readonly(
        self,
        shot: int | str,
        area: str,
        *relative: str,
        require_exists: bool = False,
    ) -> LegacyResolution:
        """Resolve the former ``{shot}/{area}`` hierarchy without writes."""

        shot_value = _positive_integer(shot, "shot")
        if area not in _LEGACY_AREAS:
            allowed = ", ".join(sorted(_LEGACY_AREAS))
            raise FileDBPathError(
                f"Invalid legacy area {area!r}; expected one of: {allowed}"
            )
        parts = tuple(
            _component(part, "legacy relative path component") for part in relative
        )
        path = self.root / str(shot_value) / area
        if parts:
            path = path.joinpath(*parts)
        exists = path.exists() or path.is_symlink()
        if require_exists and not exists:
            raise FileNotFoundError(f"Legacy FileDB artifact does not exist: {path}")
        return LegacyResolution(path=path, exists=exists)


@dataclass(frozen=True)
class LegacyAuditEntry:
    source: str
    proposed_target: str | None
    status: str
    reason: str | None = None
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class LegacyCollision:
    proposed_target: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LegacyDuplicate:
    sha256: str
    size: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LegacyMissingProduct:
    shot: int
    product: str
    expected_source: str


@dataclass(frozen=True)
class LegacyAuditReport:
    legacy_root: str
    target_root: str
    entries: tuple[LegacyAuditEntry, ...]
    collisions: tuple[LegacyCollision, ...]
    duplicates: tuple[LegacyDuplicate, ...]
    symlinks: tuple[str, ...]
    missing_products: tuple[LegacyMissingProduct, ...]

    @property
    def unmapped(self) -> tuple[LegacyAuditEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == "unmapped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dry_run": True,
            "legacy_root": self.legacy_root,
            "target_root": self.target_root,
            "summary": {
                "files": len(self.entries),
                "mapped": sum(entry.status == "mapped" for entry in self.entries),
                "unmapped": len(self.unmapped),
                "symlinks": len(self.symlinks),
                "collisions": len(self.collisions),
                "duplicate_groups": len(self.duplicates),
                "missing_products": len(self.missing_products),
            },
            "entries": [asdict(entry) for entry in self.entries],
            "collisions": [asdict(item) for item in self.collisions],
            "duplicates": [asdict(item) for item in self.duplicates],
            "symlinks": list(self.symlinks),
            "missing_products": [asdict(item) for item in self.missing_products],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_code_and_mode(
    parts: tuple[str, ...],
) -> tuple[str, int, tuple[str, ...]] | None:
    code_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part in {item.value for item in GPECCode}
        ),
        None,
    )
    if code_index is None:
        return None
    mode_index = next(
        (
            index
            for index, part in enumerate(parts)
            if re.fullmatch(r"n{1,2}=\d+", part)
        ),
        None,
    )
    if mode_index is None:
        return None
    remaining = tuple(
        part
        for index, part in enumerate(parts)
        if index not in {code_index, mode_index}
    )
    return parts[code_index], int(parts[mode_index].split("=", 1)[1]), remaining


def _propose_mapping(relative: Path, target: FileDB) -> tuple[Path | None, str | None]:
    parts = relative.parts
    if len(parts) < 3 or not parts[0].isdigit():
        return None, "not a recognized shot-first artifact"
    shot = int(parts[0])
    area = parts[1]
    remainder = tuple(parts[2:])
    filename = remainder[-1]

    if area == "diagnostics":
        if filename == f"vest_{shot}_daq_raw.json.gz":
            return target.raw(shot, artifact="output") / filename, None
        return target.legacy("diagnostics", shot, artifact="input").joinpath(
            *remainder
        ), None

    if area == "omas" and len(remainder) == 1:
        suffixes = {
            f"{shot}_diagnostics.json": ("diagnostics", "output"),
            f"{shot}_eddy.json": ("eddy", "output"),
            f"{shot}_constraints.json": ("efit", "work"),
            f"{shot}_efit.json": ("efit", "output"),
            f"{shot}_chease.json": ("chease", "output"),
        }
        stage_artifact = suffixes.get(filename)
        if stage_artifact is None:
            return None, "legacy OMAS product has no canonical stage mapping"
        stage, artifact = stage_artifact
        return target.omas(stage, shot=shot, artifact=artifact) / filename, None

    if area in {"efit", "chease"}:
        first = remainder[0].lower()
        artifact = (
            "input"
            if first in {"input", "kfile"}
            else "output"
            if first in {"output", "gfile", "afile", "mfile"}
            else "log"
            if first in {"log", "logs"}
            else "plot"
            if first in {"plot", "plots"}
            else "config"
            if first in {"config", "configuration"}
            else "work"
        )
        base = (
            target.efit(shot, artifact=artifact)
            if area == "efit"
            else target.chease(shot, artifact=artifact)
        )
        tail = (
            remainder[1:]
            if first
            in {
                "input",
                "output",
                "log",
                "logs",
                "plot",
                "plots",
                "config",
                "configuration",
                "kfile",
                "gfile",
                "afile",
                "mfile",
            }
            else remainder
        )
        return base.joinpath(*tail), None

    if area == "linear_stability":
        parsed = _legacy_code_and_mode(remainder)
        if parsed is None:
            return None, "stability artifact has no recognizable code and mode"
        code, mode, remaining = parsed
        return target.gpec(code, shot, mode, artifact="work").joinpath(*remaining), None

    return None, f"legacy area {area!r} has no canonical mapping"


def audit_legacy_filedb(
    legacy_root: str | os.PathLike[str],
    *,
    target_root: str | os.PathLike[str] | None = None,
    expected_products: Mapping[str, str] | None = None,
) -> LegacyAuditReport:
    """Inventory and propose mappings without modifying either FileDB root."""

    source_root = Path(legacy_root).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Legacy FileDB root is not a directory: {source_root}")
    target = FileDB(source_root if target_root is None else target_root)
    nodes = sorted(source_root.rglob("*"), key=lambda path: path.as_posix())
    symlinks = tuple(
        path.relative_to(source_root).as_posix() for path in nodes if path.is_symlink()
    )
    files = [path for path in nodes if not path.is_symlink() and path.is_file()]

    entries: list[LegacyAuditEntry] = []
    target_sources: dict[str, list[str]] = defaultdict(list)
    duplicate_sources: dict[tuple[int, str], list[str]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(source_root)
        source = relative.as_posix()
        proposed, reason = _propose_mapping(relative, target)
        size = path.stat().st_size
        checksum = _sha256(path)
        duplicate_sources[(size, checksum)].append(source)
        proposed_text = None if proposed is None else str(proposed)
        status = "unmapped" if proposed is None else "mapped"
        entries.append(
            LegacyAuditEntry(
                source=source,
                proposed_target=proposed_text,
                status=status,
                reason=reason,
                size=size,
                sha256=checksum,
            )
        )
        if proposed_text is not None:
            target_sources[proposed_text].append(source)

    collisions = tuple(
        LegacyCollision(target_path, tuple(sorted(sources)))
        for target_path, sources in sorted(target_sources.items())
        if len(sources) > 1
    )
    duplicates = tuple(
        LegacyDuplicate(checksum, size, tuple(sorted(sources)))
        for (size, checksum), sources in sorted(duplicate_sources.items())
        if len(sources) > 1
    )

    products = dict(
        _DEFAULT_EXPECTED_PRODUCTS if expected_products is None else expected_products
    )
    shots = sorted(
        int(path.name)
        for path in source_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.isdigit()
    )
    missing: list[LegacyMissingProduct] = []
    for shot in shots:
        for product, template in sorted(products.items()):
            expected = source_root / str(shot) / template.format(shot=shot)
            if not expected.is_file():
                missing.append(
                    LegacyMissingProduct(
                        shot=shot,
                        product=product,
                        expected_source=str(expected.relative_to(source_root)),
                    )
                )

    return LegacyAuditReport(
        legacy_root=str(source_root),
        target_root=str(target.root),
        entries=tuple(entries),
        collisions=collisions,
        duplicates=duplicates,
        symlinks=symlinks,
        missing_products=tuple(missing),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit", help="run a read-only legacy migration audit"
    )
    audit.add_argument("legacy_root", type=Path)
    audit.add_argument("--target-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = audit_legacy_filedb(args.legacy_root, target_root=args.target_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ArtifactClass",
    "FileDB",
    "FileDBConfigError",
    "FileDBDomain",
    "FileDBError",
    "FileDBPathError",
    "GPECCode",
    "LegacyAuditEntry",
    "LegacyAuditReport",
    "LegacyCollision",
    "LegacyDuplicate",
    "LegacyMissingProduct",
    "LegacyResolution",
    "OMASStage",
    "audit_legacy_filedb",
]
