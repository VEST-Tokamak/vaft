"""Shared pieces for the external-code environment checkers.

`check_chease.py` and `check_gpec.py` report the same way `check_vaft_environment.py`
does, so the vocabulary comes from that module rather than being copied: the
`CheckResult` record, the PASS/FAIL/SKIP/WARN statuses and `check_command` are
imported from it unchanged. Only `format_report` is reimplemented here, because
the original hard-codes its own title and rerun line.

Every check answers for one layer -- toolchain, source, build, discovery, smoke
run, workflow, numerical agreement -- and a failure names the layer it belongs
to. A checker that answers a build failure with a page of linker output tells
the reader nothing they can act on.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


INSTALL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = INSTALL_DIR.parent
MANIFEST_NAME = "vaft-external-install.json"


def _load_vaft_checker():
    """Import `check_vaft_environment.py` from this directory, by path."""
    path = INSTALL_DIR / "check_vaft_environment.py"
    spec = importlib.util.spec_from_file_location("_vaft_environment_checker", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_vaft_checker()

CheckResult = checker.CheckResult
PASS = checker.PASS
FAIL = checker.FAIL
SKIP = checker.SKIP
WARN = checker.WARN
check_command = checker.check_command


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(results: Sequence[CheckResult], *, title: str, rerun: str) -> str:
    """Render the human-readable report for one external code."""
    lines = [title, "-" * len(title)]
    for result in results:
        lines.append(f"[{result.status}] {result.name}")
        if result.detail:
            lines.append(f"       {result.detail}")
        if result.status in (FAIL, WARN) and result.remediation:
            lines.append(f"       -> {result.remediation}")
    warnings = [result for result in results if result.status == WARN]
    failures = [result for result in results if result.failed]
    lines.append("")
    if warnings and not failures:
        plural = "check" if len(warnings) == 1 else "checks"
        lines.append(
            f"{len(warnings)} {plural} reported something worth knowing; "
            "read the arrows above."
        )
    if failures:
        lines.append(f"{len(failures)} check(s) failed. Fix the actions above and rerun:")
        lines.append(f"  {rerun}")
    else:
        lines.append("All checks passed.")
    return "\n".join(lines)


def emit(results: Sequence[CheckResult], *, title: str, rerun: str, as_json: bool) -> int:
    """Print the report and return the process exit status."""
    if as_json:
        print(json.dumps({"checks": [result._asdict() for result in results]}, indent=2))
    else:
        print(format_report(results, title=title, rerun=rerun))
    return 1 if any(result.failed for result in results) else 0


# ---------------------------------------------------------------------------
# Layer: the source checkout
# ---------------------------------------------------------------------------


def check_source_checkout(
    source: Optional[str | os.PathLike[str]],
    *,
    project: str,
    markers: Sequence[str],
    remediation: str,
) -> CheckResult:
    """The supplied path is a plausible checkout of the expected project."""
    label = f"{project} source"
    if not source:
        return CheckResult(
            label,
            SKIP,
            "no --source given, so the build inputs cannot be verified",
        )
    root = Path(source).expanduser()
    if not root.is_dir():
        return CheckResult(label, FAIL, f"{root} does not exist", remediation)
    missing = [name for name in markers if not (root / name).exists()]
    if missing:
        return CheckResult(
            label,
            FAIL,
            f"{root} is missing {', '.join(missing)}",
            remediation,
        )
    return CheckResult(label, PASS, str(root))


def source_revision(source: Optional[str | os.PathLike[str]]) -> Optional[dict[str, Any]]:
    """Return the checkout's revision, or None when it is not a repository.

    Read-only: it only asks Git what the tree already is.
    """
    if not source:
        return None
    git = shutil.which("git")
    if git is None:
        return None
    root = Path(source).expanduser()

    def _run(*arguments: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                [git, "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return (completed.stdout or "").strip()

    revision = _run("rev-parse", "--short", "HEAD")
    if not revision:
        return None
    return {
        "revision": revision,
        "described": _run("describe", "--tags", "--always") or revision,
        # Untracked files are excluded: a build writes its own products into
        # the source tree, and those say nothing about which revision was
        # compiled. Only tracked changes make the recorded revision a lie.
        "dirty": bool(_run("status", "--porcelain", "--untracked-files=no")),
    }


def check_source_revision(
    source: Optional[str | os.PathLike[str]], *, project: str
) -> CheckResult:
    """Report the revision the build inputs came from, for provenance."""
    label = f"{project} revision"
    if not source:
        return CheckResult(label, SKIP, "no --source given")
    information = source_revision(source)
    if information is None:
        return CheckResult(
            label,
            SKIP,
            "not a repository, so the built binaries cannot be traced to a revision",
        )
    detail = information["described"]
    if information["described"] != information["revision"]:
        detail = f"{information['described']} ({information['revision']})"
    if information["dirty"]:
        return CheckResult(
            label,
            WARN,
            f"{detail}, with uncommitted changes",
            "The revision recorded for this build does not describe the files that "
            "were compiled. Commit or set aside the changes before a run whose "
            "provenance matters.",
        )
    return CheckResult(label, PASS, detail)


# ---------------------------------------------------------------------------
# Layer: the toolchain
# ---------------------------------------------------------------------------


def find_msys2_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate an MSYS2 installation, mirroring the installer's search."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MSYS2_ROOT"):
        candidates.append(Path(os.environ["MSYS2_ROOT"]))
    candidates.append(Path("C:/msys64"))
    if os.environ.get("SystemDrive"):
        candidates.append(Path(os.environ["SystemDrive"] + "/msys64"))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "msys64")
    for candidate in candidates:
        if (candidate / "usr" / "bin" / "bash.exe").is_file():
            return candidate
    return None


def check_toolchain(*, required: bool, mingw_environment: str = "ucrt64") -> CheckResult:
    """A MinGW-w64 Fortran toolchain is available to rebuild the code."""
    label = "MinGW-w64 toolchain"
    remediation = (
        "Install MSYS2 and the compiler packages, or rerun the installer with "
        "-InstallToolchain. See install/README.md."
    )
    if os.name != "nt":
        return CheckResult(label, SKIP, "not native Windows")
    root = find_msys2_root()
    if root is None:
        status = FAIL if required else SKIP
        return CheckResult(label, status, "MSYS2 was not found", remediation if required else "")
    compiler = root / mingw_environment / "bin" / "gfortran.exe"
    if not compiler.is_file():
        status = FAIL if required else SKIP
        return CheckResult(
            label, status, f"{compiler} is missing", remediation if required else ""
        )
    version = ""
    try:
        completed = subprocess.run(
            [str(compiler), "-dumpversion"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        version = (completed.stdout or "").strip()
    except OSError:
        version = ""
    detail = str(compiler)
    if version:
        detail = f"gfortran {version} ({compiler})"
    return CheckResult(label, PASS, detail)


# ---------------------------------------------------------------------------
# Layer: the build record and the installed executables
# ---------------------------------------------------------------------------


def default_prefix(code: str) -> Optional[Path]:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    return Path(local) / "vaft" / "external" / code


def read_manifest(prefix: Optional[str | os.PathLike[str]]) -> Optional[dict[str, Any]]:
    if not prefix:
        return None
    path = Path(prefix).expanduser() / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        # utf-8-sig, because Windows PowerShell 5.1 always writes a byte-order
        # mark and json.loads treats it as a syntax error.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def check_build_record(
    prefix: Optional[str | os.PathLike[str]],
    source: Optional[str | os.PathLike[str]],
    *,
    project: str,
    remediation: str,
) -> CheckResult:
    """The prefix records a build, and it matches the checkout being verified."""
    label = f"{project} build record"
    manifest = read_manifest(prefix)
    if manifest is None:
        return CheckResult(
            label, SKIP, "no install manifest, so this prefix was not written by the installer"
        )
    recorded = manifest.get("source_revision")
    current = source_revision(source) if source else None
    if recorded and current and recorded != current["revision"]:
        return CheckResult(
            label,
            WARN,
            f"built from {recorded}, but --source is now at {current['revision']}",
            "Rerun the installer so the installed binaries match the checkout you "
            "are verifying against.",
        )
    detail = f"built from {recorded}" if recorded else "recorded"
    return CheckResult(label, PASS, detail)


def _looks_like_a_program(path: Path) -> bool:
    if os.name != "nt":
        return os.access(path, os.X_OK)
    if path.suffix.lower() in (".exe", ".bat", ".cmd"):
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"MZ"
    except OSError:
        return False


def check_executables(
    prefix: Optional[str | os.PathLike[str]],
    names: Sequence[str],
    *,
    project: str,
    remediation: str,
) -> CheckResult:
    """Every executable the workflow needs exists in the prefix and is a program."""
    label = f"{project} executables"
    if not prefix:
        return CheckResult(label, FAIL, "no install prefix to look in", remediation)
    bin_directory = Path(prefix).expanduser() / "bin"
    if not bin_directory.is_dir():
        return CheckResult(label, FAIL, f"{bin_directory} does not exist", remediation)

    problems: list[str] = []
    found: list[str] = []
    for name in names:
        candidates = [bin_directory / (name + ".exe"), bin_directory / name]
        resolved = next((path for path in candidates if path.is_file()), None)
        if resolved is None:
            problems.append(f"missing {name}")
        elif resolved.stat().st_size == 0:
            problems.append(f"{resolved.name} is empty, which is what a failed link leaves behind")
        elif not _looks_like_a_program(resolved):
            problems.append(f"{resolved.name} is not a program this platform can start")
        else:
            found.append(resolved.name)
    if problems:
        return CheckResult(label, FAIL, "; ".join(problems), remediation)
    return CheckResult(label, PASS, f"{len(found)} in {bin_directory}")


def check_executables_load(
    prefix: Optional[str | os.PathLike[str]],
    names: Sequence[str],
    *,
    project: str,
    timeout: int = 30,
) -> CheckResult:
    """Each executable starts with PATH stripped to the system directory.

    A build can succeed and still be unusable because its runtime libraries are
    only reachable from the shell it was built in. Starting each program with a
    bare PATH is what proves the prefix is self-contained.
    """
    label = f"{project} executables load"
    if os.name != "nt" or not prefix:
        return CheckResult(label, SKIP, "native Windows only")
    bin_directory = Path(prefix).expanduser() / "bin"
    environment = {
        "SystemRoot": os.environ.get("SystemRoot", "C:/Windows"),
        "PATH": str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"),
    }
    scratch = tempfile.mkdtemp(prefix="vaft-loadprobe-")
    unloadable: list[str] = []
    try:
        for name in names:
            executable = bin_directory / (name + ".exe")
            if not executable.is_file():
                continue
            try:
                completed = subprocess.run(
                    [str(executable)],
                    cwd=scratch,
                    env=environment,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                # It started and kept running, which is all this layer asks.
                continue
            except OSError as error:
                unloadable.append(f"{name}: {error}")
                continue
            # 0xC0000135 / 0xC0000139: a library or an entry point is missing.
            if completed.returncode in (-1073741515, -1073741701):
                unloadable.append(f"{name}: missing runtime libraries")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if unloadable:
        return CheckResult(
            label,
            FAIL,
            "; ".join(unloadable),
            "Rerun the installer so it copies the runtime libraries next to the "
            "executables. See install/README.md.",
        )
    return CheckResult(label, PASS, "every executable starts with a bare PATH")


# ---------------------------------------------------------------------------
# Layer: numerical agreement
# ---------------------------------------------------------------------------


def compare_scalars(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
    tolerances: Mapping[str, float],
    *,
    default_tolerance: float = 1e-6,
) -> list[str]:
    """Return one message per quantity that falls outside its tolerance."""
    problems: list[str] = []
    for key, reference in expected.items():
        if key not in actual:
            problems.append(f"{key} was not produced")
            continue
        tolerance = float(tolerances.get(key, default_tolerance))
        value = float(actual[key])
        scale = max(abs(float(reference)), 1e-30)
        deviation = abs(value - float(reference)) / scale
        if deviation > tolerance:
            problems.append(
                f"{key}={value:.6g} differs from {float(reference):.6g} "
                f"by {deviation:.3g}, tolerance {tolerance:.3g}"
            )
    return problems


def first_error_line(text: str, *, limit: int = 200) -> str:
    """Return the first line that looks like an error, never the whole log."""
    needles = ("error:", "error #", "undefined reference", "cannot find", "fatal")
    for line in text.splitlines():
        lowered = line.lower()
        if any(needle in lowered for needle in needles):
            return line.strip()[:limit]
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:limit]
    return ""


def scratch_directory(prefix: str) -> str:
    """A working directory outside the repository, for a smoke run."""
    return tempfile.mkdtemp(prefix=prefix)
