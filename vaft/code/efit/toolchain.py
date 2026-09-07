"""The EFIT toolchain: two executable roles, one root, one build lineage (issue #194).

EFIT's source tree builds two programs from one repository and one CMake
configuration: ``efit``, the reconstruction, and ``efund``, the Green-table
generator whose output ``efit`` consumes.  A table is only as traceable as
the binary that produced it, so VAFT resolves both roles from the same
``$EFITHOME`` and records for each the path, the checksum and, when the
root sits inside a git checkout, the revision it was built from.

Two layouts are recognised beneath the root:

``bin/efit``, ``bin/efund``
    the installed layout ``install/install_efit.sh`` writes and the one every
    other external code uses;
``efit/efit``, ``green/efund``
    a raw CMake build directory, which is what a hand-configured build looks
    like and what EFIT's own documentation produces.

There is deliberately no separate root variable for ``efund``: a second root
would let the two roles come from two builds, which is exactly what
provenance must rule out.  An explicit executable path still wins for either
role, as it always has for ``efit``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from vaft.compat import is_executable, resolve_executable

from .._executables import missing_home_message

__all__ = [
    "BUILD_TREE_LAYOUT",
    "EFIT_HOME_ENV",
    "EFIT_LEGACY_EXEC_ENV",
    "EFIT_ROLES",
    "INSTALLED_LAYOUT",
    "ExecutableIdentity",
    "executable_identity",
    "resolve_role",
    "resolve_toolchain",
    "toolchain_identities",
    "unconfigured_reason",
]

EFIT_HOME_ENV = "EFITHOME"
#: The historical ``$EFIT`` variable, honoured for the ``efit`` role only.
EFIT_LEGACY_EXEC_ENV = "EFIT"
EFIT_ROLES = ("efit", "efund")
INSTALLED_LAYOUT: Mapping[str, Path] = {"efit": Path("bin/efit"), "efund": Path("bin/efund")}
BUILD_TREE_LAYOUT: Mapping[str, Path] = {"efit": Path("efit/efit"), "efund": Path("green/efund")}


def _check_role(role: str) -> None:
    if role not in EFIT_ROLES:
        raise ValueError(f"unknown EFIT toolchain role {role!r}; expected one of {EFIT_ROLES}")


def resolve_role(
    role: str,
    *,
    explicit: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """One role's executable: explicit path, then ``$EFITHOME`` in either layout.

    An explicit *directory* is taken as a root holding ``<role>`` directly
    (the historical ``EFITConfig.executable`` contract).  A configured
    ``$EFITHOME`` that holds the role in neither layout raises
    ``FileNotFoundError`` naming both expected locations: a half-installed
    root is a misconfiguration, not an absence.  ``None`` when nothing is
    configured.
    """
    _check_role(role)
    if explicit:
        candidate = Path(explicit).expanduser()
        target = candidate / role if candidate.is_dir() else candidate
        return resolve_executable(target) or target
    environment = dict(os.environ if env is None else env)
    home = environment.get(EFIT_HOME_ENV)
    if home and str(home).strip():
        root = Path(home).expanduser()
        for layout in (INSTALLED_LAYOUT, BUILD_TREE_LAYOUT):
            found = resolve_executable(root / layout[role])
            if found is not None:
                if not is_executable(found):
                    raise PermissionError(
                        f"EFIT toolchain executable is not executable for ${EFIT_HOME_ENV}={root}: "
                        f"{found}. Compile or install EFIT correctly and ensure the file is executable."
                    )
                return found
        raise FileNotFoundError(
            f"EFIT executable ({role}) is missing for ${EFIT_HOME_ENV}={root}: expected "
            f"{root / INSTALLED_LAYOUT[role]} (or {root / BUILD_TREE_LAYOUT[role]} in a CMake "
            f"build tree). Compile or install EFIT so that the executable exists at the "
            "documented location; the EFIT installer for this platform builds both "
            "efit and efund under one root."
        )
    if role == "efit":
        legacy = environment.get(EFIT_LEGACY_EXEC_ENV)
        if legacy:
            candidate = Path(legacy).expanduser()
            target = candidate / "efit" if candidate.is_dir() else candidate
            return resolve_executable(target) or target
    return None


def resolve_toolchain(
    *,
    efit_executable: str | os.PathLike[str] | None = None,
    efund_executable: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Path | None]:
    """Both roles, resolved by the same rule; ``None`` for a role nothing configures."""
    return {
        "efit": resolve_role("efit", explicit=efit_executable, env=env),
        "efund": resolve_role("efund", explicit=efund_executable, env=env),
    }


def unconfigured_reason(role: str = "efit") -> str:
    _check_role(role)
    return missing_home_message(
        home_variable=EFIT_HOME_ENV,
        relative_path=INSTALLED_LAYOUT[role],
        code_name=f"EFIT toolchain ({role})",
        compatibility_variables=(EFIT_LEGACY_EXEC_ENV,) if role == "efit" else (),
    )


@dataclass(frozen=True)
class ExecutableIdentity:
    """What a table manifest or a run record says about one binary."""

    role: str
    path: str
    sha256: str
    size: int
    mtime: str
    build_revision: str | None = None
    build_root: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_describe(start: Path, *, max_depth: int = 6) -> tuple[str | None, str | None]:
    """``git describe --always --dirty`` of the checkout an executable lives in, if any."""
    current = start if start.is_dir() else start.parent
    for _ in range(max_depth):
        if (current / ".git").exists():
            try:
                completed = subprocess.run(
                    ["git", "-C", str(current), "describe", "--always", "--dirty"],
                    capture_output=True, text=True, timeout=2.0, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None, str(current)
            described = completed.stdout.strip()
            return (described or None) if completed.returncode == 0 else None, str(current)
        if current.parent == current:
            break
        current = current.parent
    return None, None


def executable_identity(path: str | os.PathLike[str], role: str) -> ExecutableIdentity:
    """Checksum and, where recoverable, the source revision behind an executable."""
    _check_role(role)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} executable does not exist: {resolved}")
    stat = resolved.stat()
    revision, root = _git_describe(resolved.parent)
    return ExecutableIdentity(
        role=role,
        path=str(resolved),
        sha256=_sha256(resolved),
        size=int(stat.st_size),
        mtime=datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        build_revision=revision,
        build_root=root,
    )


def toolchain_identities(resolved: Mapping[str, Path | None]) -> dict[str, dict[str, Any] | None]:
    """Identities for every resolved role, ``None`` where a role is unconfigured."""
    return {
        role: None if path is None else executable_identity(path, role).as_dict()
        for role, path in resolved.items()
    }
