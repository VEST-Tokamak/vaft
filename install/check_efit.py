"""Verify an EFIT toolchain installation (efit + efund), layer by layer.

A build that compiles is not yet a toolchain VAFT can trust with a Green
table.  This walks the same path a table regeneration and a reconstruction
take -- toolchain, source checkout and revision, the build record, both
executables, what the build can do (NetCDF, hence m-files), a real EFUND run
on EFIT's own public DIII-D machine file at a small grid, and the resolution
VAFT itself performs -- and names the layer that failed.

    python install/check_efit.py --source ~/git/efit --prefix ~/git/efit/vaft-install
    python install/check_efit.py --source ~/git/efit --build-tree ~/git/efit/build-mac

`--prefix` checks the installed layout (bin/efit, bin/efund) that $EFITHOME
resolves; `--build-tree` checks a raw CMake build directory (efit/efit,
green/efund), which is what a hand-configured build looks like.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

# The checkout this script lives in must answer `import vaft`, not whichever
# editable install happens to be registered: a worktree run would otherwise
# check the main checkout's toolchain support and misreport this one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _external_code_common import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    check_build_record,
    check_source_checkout,
    check_source_revision,
    emit,
    first_error_line,
    read_manifest,
    scratch_directory,
)

TITLE = "EFIT toolchain check"
RERUN = "python install/check_efit.py"
PROJECT = "EFIT"
ROLES = ("efit", "efund")
SOURCE_MARKERS = ("CMakeLists.txt", "efit/efit.F90", "green/efund.f90", "LICENSE.rst")
BUILD_REMEDIATION = (
    "Build the toolchain with:\n"
    "         bash install/install_efit.sh --source <efit source> --accept-efit-users-agreement"
)
INSTALLED_LAYOUT = {"efit": Path("bin/efit"), "efund": Path("bin/efund")}
BUILD_TREE_LAYOUT = {"efit": Path("efit/efit"), "efund": Path("green/efund")}
#: EFIT's own public DIII-D machine file, shipped with the source; the smoke
#: run needs no VAFT geometry and no VEST data.
SMOKE_MHDIN = Path("share/support_files/DIII-D/green/181292/mhdin.dat")
SMOKE_GRID = 33
#: (nsilop, magpri, nfsum, nesum, nvsum) of that file, for the size checks.
SMOKE_COUNTS = (44, 76, 18, 6, 28)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executables(prefix: Optional[str], build_tree: Optional[str]) -> dict[str, Path]:
    if build_tree:
        root = Path(build_tree).expanduser()
        return {role: root / relative for role, relative in BUILD_TREE_LAYOUT.items()}
    if prefix:
        root = Path(prefix).expanduser()
        return {role: root / relative for role, relative in INSTALLED_LAYOUT.items()}
    return {}


def check_posix_toolchain(*, required: bool) -> CheckResult:
    label = "EFIT build toolchain"
    if os.name == "nt":
        return CheckResult(label, SKIP, "POSIX platforms only")
    missing = [tool for tool in ("cmake", "gfortran", "git") if shutil.which(tool) is None]
    if missing:
        status = FAIL if required else WARN
        return CheckResult(label, status, f"missing {', '.join(missing)}", "Install them (macOS: brew install cmake gcc).")
    try:
        version = subprocess.run(["gfortran", "--version"], capture_output=True, text=True, timeout=20, check=False).stdout.splitlines()[0]
    except Exception:
        version = "gfortran"
    return CheckResult(label, PASS, version)


def check_toolchain_executables(executables: dict[str, Path]) -> CheckResult:
    label = "EFIT executables"
    if not executables:
        return CheckResult(label, FAIL, "no --prefix or --build-tree to look in", BUILD_REMEDIATION)
    problems, found = [], []
    for role, path in executables.items():
        if not path.is_file():
            problems.append(f"missing {role} at {path}")
        elif not os.access(path, os.X_OK):
            problems.append(f"{path} is not executable")
        else:
            found.append(f"{role} {_sha256(path)[:12]}")
    if problems:
        return CheckResult(label, FAIL, "; ".join(problems), BUILD_REMEDIATION)
    return CheckResult(label, PASS, ", ".join(found))


def check_capabilities(prefix: Optional[str], build_tree: Optional[str]) -> CheckResult:
    """What the build can do: NetCDF (m-files), build type, compiler."""
    label = "EFIT build capabilities"
    netcdf = None
    build_type = None
    compiler = None
    manifest = read_manifest(prefix) if prefix else None
    if manifest and manifest.get("code") == "efit":
        netcdf = bool(manifest.get("netcdf", {}).get("enabled"))
        build_type = next((a.split("=", 1)[1] for a in manifest.get("cmake_arguments", []) if a.startswith("-DCMAKE_BUILD_TYPE=")), None)
        compiler = manifest.get("compiler", {}).get("version")
    else:
        cache = None
        for root in (build_tree, prefix):
            if root and (Path(root).expanduser() / "CMakeCache.txt").is_file():
                cache = (Path(root).expanduser() / "CMakeCache.txt").read_text(errors="replace")
                break
        if cache is None:
            return CheckResult(label, SKIP, "no build manifest or CMakeCache.txt to read")
        for line in cache.splitlines():
            if line.startswith("ENABLE_NETCDF:"):
                netcdf = line.split("=", 1)[1].strip().upper() == "ON"
            elif line.startswith("CMAKE_BUILD_TYPE:"):
                build_type = line.split("=", 1)[1].strip()
            elif line.startswith("CMAKE_Fortran_COMPILER:"):
                compiler = line.split("=", 1)[1].strip()
    detail = f"build type {build_type or 'unknown'}, compiler {compiler or 'unknown'}, NetCDF {'on' if netcdf else 'off'}"
    if netcdf is False:
        return CheckResult(label, WARN, detail, "Without NetCDF EFIT writes no m-file, so iteration counts and per-slice residuals are unavailable. Rebuild with install/install_efit.sh (NetCDF on by default).")
    return CheckResult(label, PASS, detail)


def _run_with_stack(command: list[str], *, cwd: Path, timeout: int, stack_kb: int = 65536):
    shell = f"ulimit -s {int(stack_kb)} 2>/dev/null; exec \"$@\""
    return subprocess.run(["bash", "-c", shell, "efit-check", *command], cwd=str(cwd), capture_output=True, text=True, errors="replace", timeout=timeout, check=False)


def check_efund_smoke(executables: dict[str, Path], source: Optional[str], *, skip: bool) -> CheckResult:
    """efund generates the expected Green tables from EFIT's own DIII-D file at a small grid."""
    label = "EFUND smoke run"
    if skip:
        return CheckResult(label, SKIP, "requested with --skip-smoke")
    efund = executables.get("efund")
    if efund is None or not efund.is_file():
        return CheckResult(label, SKIP, "no efund to run")
    if not source or not (Path(source).expanduser() / SMOKE_MHDIN).is_file():
        return CheckResult(label, SKIP, f"no --source with {SMOKE_MHDIN}")
    workdir = Path(scratch_directory("vaft-efund-check-"))
    shutil.copy(Path(source).expanduser() / SMOKE_MHDIN, workdir / "mhdin.dat")
    try:
        completed = _run_with_stack([str(efund), str(SMOKE_GRID)], cwd=workdir, timeout=600)
    except subprocess.TimeoutExpired:
        return CheckResult(label, FAIL, "efund did not finish within 600 s", f"The run directory is {workdir}.")
    (workdir / "run_green.out").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        return CheckResult(label, FAIL, first_error_line(completed.stdout + completed.stderr) or f"efund exited {completed.returncode}", f"The run directory is {workdir}.")
    nsilop, magpri, nfsum, nesum, nvsum = SMOKE_COUNTS
    nwnh = SMOKE_GRID * SMOKE_GRID
    expected = {
        "rfcoil.ddd": (nsilop * nfsum + magpri * nfsum) * 8 + 16,
        "brzgfc.dat": 2 * nwnh * nfsum * 8 + 16,
        f"ep{SMOKE_GRID}{SMOKE_GRID}.ddd": (nsilop + magpri) * nwnh * 8 + 16,
        f"ec{SMOKE_GRID}{SMOKE_GRID}.ddd": 2 * 4 + 8 + (SMOKE_GRID * 2) * 8 + 8 + nwnh * nfsum * 8 + 8 + nwnh * SMOKE_GRID * 8 + 8,
        f"re{SMOKE_GRID}{SMOKE_GRID}.ddd": (nsilop * nesum + magpri * nesum + nwnh * nesum) * 8 + 24,
        f"rv{SMOKE_GRID}{SMOKE_GRID}.ddd": (nsilop * nvsum + magpri * nvsum + nwnh * nvsum + nfsum * nvsum + nesum * nvsum + nvsum * nvsum) * 8 + 48,
        "mhdout.dat": None,
    }
    problems = []
    for name, size in expected.items():
        path = workdir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif size is not None and path.stat().st_size != size:
            problems.append(f"{name} is {path.stat().st_size} bytes, expected {size}")
    if problems:
        return CheckResult(label, FAIL, "; ".join(problems), f"The run directory is {workdir}.")
    shutil.rmtree(workdir, ignore_errors=True)
    return CheckResult(label, PASS, f"{len(expected)} tables at {SMOKE_GRID}x{SMOKE_GRID} from {SMOKE_MHDIN.name}, sizes exact")


def check_efit_starts(executables: dict[str, Path]) -> CheckResult:
    """efit starts (its runtime libraries resolve) when asked for a grid it has no tables for."""
    label = "EFIT starts"
    efit = executables.get("efit")
    if efit is None or not efit.is_file():
        return CheckResult(label, SKIP, "no efit to start")
    workdir = Path(scratch_directory("vaft-efit-check-"))
    try:
        completed = subprocess.run([str(efit), "129"], cwd=str(workdir), input="", capture_output=True, text=True, errors="replace", timeout=60, check=False)
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        return CheckResult(label, PASS, "started and waited for input")
    except OSError as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    shutil.rmtree(workdir, ignore_errors=True)
    text = completed.stdout + completed.stderr
    if "dyld" in text or "error while loading shared libraries" in text:
        return CheckResult(label, FAIL, first_error_line(text), "The executable cannot find its runtime libraries; rebuild with install/install_efit.sh.")
    return CheckResult(label, PASS, f"exited {completed.returncode} with no input")


def check_vaft_discovery(prefix: Optional[str], build_tree: Optional[str]) -> CheckResult:
    """VAFT resolves both roles from EFITHOME through its own documented mechanism."""
    label = "VAFT toolchain discovery"
    home = prefix or build_tree
    if not home:
        return CheckResult(label, SKIP, "no prefix or build tree")
    previous = os.environ.get("EFITHOME")
    os.environ["EFITHOME"] = str(Path(home).expanduser())
    try:
        try:
            from vaft.code.efit.toolchain import resolve_toolchain
        except Exception:
            from vaft.code.efit.magnetic import EFITConfig, find_efit_executable

            resolved = find_efit_executable(EFITConfig())
            if resolved is None:
                return CheckResult(label, FAIL, "EFITHOME does not resolve bin/efit", "Point EFITHOME at the prefix install/install_efit.sh wrote.")
            return CheckResult(label, WARN, f"efit {resolved}; this VAFT has no efund role yet")
        resolved = resolve_toolchain()
        import vaft

        vaft_root = Path(vaft.__file__).resolve().parents[1]
    except Exception as error:
        return CheckResult(label, FAIL, f"{type(error).__name__}: {error}", "Point EFITHOME at the prefix install/install_efit.sh wrote, or at a CMake build tree.")
    finally:
        if previous is None:
            os.environ.pop("EFITHOME", None)
        else:
            os.environ["EFITHOME"] = previous
    missing = [role for role in ROLES if resolved.get(role) is None]
    if missing:
        return CheckResult(label, FAIL, f"EFITHOME resolves no {', '.join(missing)}", BUILD_REMEDIATION)
    return CheckResult(
        label, PASS, "; ".join(f"{role} {resolved[role]}" for role in ROLES) + f" (vaft from {vaft_root})"
    )


def run_checks(*, source=None, prefix=None, build_tree=None, skip_smoke=False) -> list[CheckResult]:
    if prefix is None and build_tree is None:
        prefix = os.environ.get("EFITHOME")
    executables = _executables(prefix, build_tree)
    results = [
        check_posix_toolchain(required=bool(source)),
        check_source_checkout(source, project=PROJECT, markers=SOURCE_MARKERS, remediation=BUILD_REMEDIATION),
        check_source_revision(source, project=PROJECT),
        check_build_record(prefix, source, project=PROJECT, remediation=BUILD_REMEDIATION) if prefix else CheckResult("EFIT build record", SKIP, "a raw build tree carries no install manifest"),
        check_toolchain_executables(executables),
        check_capabilities(prefix, build_tree),
        check_efit_starts(executables),
        check_efund_smoke(executables, source, skip=skip_smoke),
        check_vaft_discovery(prefix, build_tree),
    ]
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="EFIT source tree (for revision, markers, and the smoke machine file)")
    parser.add_argument("--prefix", help="installed prefix with bin/efit and bin/efund (default: $EFITHOME)")
    parser.add_argument("--build-tree", help="a raw CMake build directory with efit/efit and green/efund")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = run_checks(source=args.source, prefix=args.prefix, build_tree=args.build_tree, skip_smoke=args.skip_smoke)
    return emit(results, title=TITLE, rerun=RERUN, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
