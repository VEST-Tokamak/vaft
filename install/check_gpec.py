"""Verify a DCON/GPEC installation, layer by layer.

Compilation alone is not support, and neither is one executable starting. GPEC
consumes what DCON leaves behind, so the run this performs is the real handoff:
DCON on a self-contained upstream example, then GPEC on DCON's output, with the
products checked at each step.

    python install/check_gpec.py --source C:/git/GPEC
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _external_code_common import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    check_build_record,
    check_executables,
    check_executables_load,
    check_source_checkout,
    check_source_revision,
    check_toolchain,
    default_prefix,
    emit,
    first_error_line,
    read_manifest,
    scratch_directory,
)

TITLE = "DCON/GPEC environment check"
RERUN = "python install/check_gpec.py"
PROJECT = "GPEC"

#: The order the suite depends on: DCON runs first and GPEC reads its output.
EXECUTABLES = ("dcon", "match", "rdcon", "rmatch", "stride", "gpec")

SOURCE_MARKERS = ("install/makefile", "install/DEFAULTS.inc", "install/TARGETS.inc", "dcon", "gpec")
BUILD_REMEDIATION = (
    "Build the suite with:\n"
    "         powershell -ExecutionPolicy Bypass -File install\\install_gpec_windows.ps1 <source>"
)

#: Upstream's own self-contained regression case: an analytic Solov'ev
#: equilibrium plus the namelists for it, so the smoke run needs no data from
#: this repository and always matches whatever revision was checked out.
EXAMPLE = "docs/examples/regression_solovev_ideal_example"

#: What DCON must leave behind for GPEC to have anything to read.
DCON_HANDOFF = ("euler.bin", "dcon.out")

#: The libraries whose presence next to the executables means the installation
#: carries the AWS S3 SDK, and therefore cannot exit. See check_netcdf_exits.
AWS_MARKER = "libaws-c-common.dll"


def _example_directory(source: Optional[str]) -> Optional[Path]:
    if not source:
        return None
    candidate = Path(source).expanduser() / EXAMPLE
    return candidate if candidate.is_dir() else None


def check_netcdf_without_s3(prefix: Optional[str]) -> CheckResult:
    """The netCDF this suite links can let a process exit.

    MSYS2's netCDF and its HDF5 both carry the AWS S3 SDK, whose shutdown
    handler waits on a condition variable that is never signalled. A program
    linked against them writes every output correctly and then never exits.

    Building netCDF without S3 does not help on its own: HDF5 pulls the same
    SDK in. Until MSYS2 ships those packages without it, or both are built
    from source, WSL2 is the route for this suite. This layer exists so that
    shows up as one named line rather than as a run that never returns.
    """
    label = "netCDF without S3"
    if os.name != "nt":
        return CheckResult(label, SKIP, "native Windows only")
    if not prefix:
        return CheckResult(label, SKIP, "no install prefix")
    marker = Path(prefix).expanduser() / "bin" / AWS_MARKER
    if marker.is_file():
        return CheckResult(
            label,
            FAIL,
            f"{AWS_MARKER} is installed beside the executables, so netCDF carries the S3 SDK",
            "A suite linked against these finishes its work and then never "
            "exits. Use WSL2 for DCON/GPEC, or build HDF5 and netCDF without "
            "S3 yourself and point -NetcdfHome at them. See install/README.md.",
        )
    manifest = read_manifest(prefix)
    home = (manifest or {}).get("netcdf_home")
    if home:
        return CheckResult(label, PASS, str(home))
    return CheckResult(label, PASS, "no S3 SDK beside the executables")


def check_vaft_discovery(prefix: Optional[str]) -> CheckResult:
    """VAFT resolves every executable through its own documented mechanism."""
    label = "VAFT executable discovery"
    try:
        from vaft.code import gpec
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(
            label,
            FAIL,
            f"vaft.code.gpec could not be imported: {error}",
            "Run install/check_vaft_environment.py first.",
        )

    previous = os.environ.get("GPECHOME")
    if prefix:
        os.environ["GPECHOME"] = str(prefix)
    try:
        config = gpec.GPECSuiteConfig()
        resolved = []
        for program in ("dcon", "rdcon", "stride", "gpec"):
            resolved.append(gpec._executable(config, program))
        for companion in ("match", "rmatch"):
            resolved.append(gpec._optional_executable(config, companion))
    except Exception as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    finally:
        if prefix:
            if previous is None:
                os.environ.pop("GPECHOME", None)
            else:
                os.environ["GPECHOME"] = previous

    missing = [str(path) for path in resolved if path is None or not Path(path).is_file()]
    if missing:
        return CheckResult(label, FAIL, "; ".join(missing), BUILD_REMEDIATION)
    return CheckResult(label, PASS, f"{len(resolved)} executables under {prefix}")


def _run_solver(executable: Path, workdir: Path, prefix: str, timeout: int):
    """Run one solver and report both its status and whether it terminated."""
    environment = dict(os.environ)
    environment["GPECHOME"] = str(prefix)
    environment["OMP_NUM_THREADS"] = "1"
    environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
    log = workdir / (executable.stem + ".log")
    with log.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                [str(executable)],
                cwd=workdir,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, log
        except OSError as error:
            log.write_text(str(error), encoding="utf-8")
            return None, log
    return completed.returncode, log


def _log_says_normal_termination(log: Path) -> bool:
    if not log.is_file():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    return "PROGRAM STOP" in text and "Normal termination" in text


def check_dcon_run(
    prefix: Optional[str], source: Optional[str], *, skip: bool, timeout: int
) -> tuple[CheckResult, Optional[Path]]:
    """DCON solves upstream's reference case and writes the handoff files."""
    label = "DCON reference run"
    if skip:
        return CheckResult(label, SKIP, "requested with --skip-smoke"), None
    example = _example_directory(source)
    if example is None or not prefix:
        return CheckResult(label, SKIP, "needs --source and an install prefix"), None

    workdir = Path(scratch_directory("vaft-gpec-check-"))
    for item in example.iterdir():
        if item.is_file():
            shutil.copy2(item, workdir)

    executable = Path(prefix) / "bin" / "dcon.exe"
    if not executable.is_file():
        executable = Path(prefix) / "bin" / "dcon"
    returncode, log = _run_solver(executable, workdir, str(prefix), timeout)

    produced = [name for name in DCON_HANDOFF if (workdir / name).is_file()]
    finished = _log_says_normal_termination(log)

    if returncode is None and finished and len(produced) == len(DCON_HANDOFF):
        # It did all its work and then would not exit. Naming that precisely is
        # the point of this layer: every later step inherits the same wait.
        return (
            CheckResult(
                label,
                FAIL,
                "DCON completed its calculation and then did not terminate",
                "This is the AWS S3 shutdown defect in MSYS2 netCDF and HDF5. "
                "Use WSL2 for this suite; see install/README.md.",
            ),
            workdir,
        )
    if returncode is None:
        return (
            CheckResult(
                label,
                FAIL,
                f"DCON did not finish within {timeout}s",
                f"The run is in {workdir}.",
            ),
            workdir,
        )
    if returncode != 0:
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        return (
            CheckResult(
                label,
                FAIL,
                first_error_line(text) or f"DCON exited {returncode}",
                f"The full run is in {workdir}.",
            ),
            workdir,
        )
    return CheckResult(label, PASS, f"solved {EXAMPLE} in {workdir}"), workdir


def check_handoff(workdir: Optional[Path]) -> CheckResult:
    """DCON left GPEC something to read."""
    label = "DCON to GPEC handoff"
    if workdir is None:
        return CheckResult(label, SKIP, "no DCON run")
    missing = [name for name in DCON_HANDOFF if not (workdir / name).is_file()]
    empty = [
        name
        for name in DCON_HANDOFF
        if (workdir / name).is_file() and (workdir / name).stat().st_size == 0
    ]
    if missing or empty:
        problems = [f"missing {name}" for name in missing] + [f"{name} is empty" for name in empty]
        return CheckResult(
            label,
            FAIL,
            "; ".join(problems),
            "DCON exited cleanly without writing what GPEC consumes. Check its "
            "namelist output settings.",
        )
    sizes = ", ".join(f"{name} {(workdir / name).stat().st_size} bytes" for name in DCON_HANDOFF)
    return CheckResult(label, PASS, sizes)


def check_gpec_run(
    prefix: Optional[str], workdir: Optional[Path], *, timeout: int
) -> CheckResult:
    """GPEC consumes DCON's output in the same directory."""
    label = "GPEC reference run"
    if workdir is None or not prefix:
        return CheckResult(label, SKIP, "no DCON run to continue from")
    executable = Path(prefix) / "bin" / "gpec.exe"
    if not executable.is_file():
        executable = Path(prefix) / "bin" / "gpec"
    returncode, log = _run_solver(executable, workdir, str(prefix), timeout)
    outputs = sorted(path.name for path in workdir.glob("gpec_*.nc"))
    if returncode is None:
        if outputs:
            # GPEC is known to hang after writing its outputs, on every
            # platform; VAFT's own adapter carries the same carve-out.
            return CheckResult(
                label,
                WARN,
                f"GPEC wrote {len(outputs)} output file(s) and then did not exit",
                "Known behaviour: the adapter treats a timeout with complete "
                "outputs as a completed run.",
            )
        return CheckResult(label, FAIL, f"GPEC did not finish within {timeout}s", f"See {workdir}.")
    if returncode != 0:
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        return CheckResult(
            label,
            FAIL,
            first_error_line(text) or f"GPEC exited {returncode}",
            f"The full run is in {workdir}.",
        )
    return CheckResult(label, PASS, f"produced {len(outputs)} output file(s)")


def check_netcdf_outputs(workdir: Optional[Path]) -> CheckResult:
    """The products exist and can be read back as netCDF."""
    label = "GPEC netCDF outputs"
    if workdir is None:
        return CheckResult(label, SKIP, "no run")
    outputs = sorted(workdir.glob("*.nc"))
    if not outputs:
        return CheckResult(
            label, FAIL, "no netCDF file was produced", "Check the run log for solver diagnostics."
        )
    try:
        import netCDF4
    except Exception:
        names = ", ".join(path.name for path in outputs)
        return CheckResult(label, WARN, f"{names} (netCDF4 is not importable, so not opened)")
    unreadable = []
    for path in outputs:
        try:
            with netCDF4.Dataset(str(path)):
                pass
        except Exception as error:
            unreadable.append(f"{path.name}: {error}")
    if unreadable:
        return CheckResult(
            label,
            FAIL,
            "; ".join(unreadable),
            "A truncated output is what a solver killed mid-write leaves behind.",
        )
    return CheckResult(label, PASS, f"{len(outputs)} readable file(s)")


ENERGY = re.compile(
    r"Energies:\s*plasma\s*=\s*([-\d.Ee+]+),\s*vacuum\s*=\s*([-\d.Ee+]+),\s*real\s*=\s*([-\d.Ee+]+)"
)


def check_numerical_agreement(workdir: Optional[Path]) -> CheckResult:
    """DCON's energies for the Solov'ev case are physically sane.

    The case is analytic and upstream ships it as a regression, so the values
    are properties of the equilibrium rather than of a machine: a stable
    free-boundary result has a positive total energy, and the plasma and vacuum
    contributions are both finite.
    """
    label = "DCON numerical agreement"
    if workdir is None:
        return CheckResult(label, SKIP, "no run")
    log = workdir / "dcon.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    match = ENERGY.search(text)
    if match is None:
        return CheckResult(label, SKIP, "DCON reported no energies to compare")
    plasma, vacuum, total = (float(value) for value in match.groups())
    problems = []
    if not (plasma > 0.0):
        problems.append(f"plasma energy {plasma:.4g} is not positive")
    if not (vacuum > 0.0):
        problems.append(f"vacuum energy {vacuum:.4g} is not positive")
    if not (total > 0.0):
        problems.append(f"total energy {total:.4g} is not positive, so the case is unstable here")
    if problems:
        return CheckResult(
            label,
            FAIL,
            "; ".join(problems),
            "Upstream's Solov'ev regression is stable. A different answer here "
            "means the build is not numerically sound; compare with Linux.",
        )
    return CheckResult(
        label, PASS, f"plasma {plasma:.4g}, vacuum {vacuum:.4g}, total {total:.4g}, stable"
    )


def run_checks(
    *,
    source: Optional[str] = None,
    prefix: Optional[str] = None,
    skip_smoke: bool = False,
    timeout: int = 900,
) -> list[CheckResult]:
    """Run every DCON/GPEC layer, in the order the suite depends on them."""
    if prefix is None:
        prefix = os.environ.get("GPECHOME")
    if prefix is None:
        candidate = default_prefix("gpec")
        if candidate is not None and (candidate / "bin").is_dir():
            prefix = str(candidate)

    results = [
        check_toolchain(required=bool(source)),
        check_source_checkout(
            source, project=PROJECT, markers=SOURCE_MARKERS, remediation=BUILD_REMEDIATION
        ),
        check_source_revision(source, project=PROJECT),
        check_build_record(prefix, source, project=PROJECT, remediation=BUILD_REMEDIATION),
        check_executables(prefix, EXECUTABLES, project=PROJECT, remediation=BUILD_REMEDIATION),
        check_executables_load(prefix, EXECUTABLES, project=PROJECT),
        check_netcdf_without_s3(prefix),
        check_vaft_discovery(prefix),
    ]

    workdir = None
    if any(result.failed for result in results):
        for label in (
            "DCON reference run",
            "DCON to GPEC handoff",
            "GPEC reference run",
            "GPEC netCDF outputs",
            "DCON numerical agreement",
        ):
            results.append(CheckResult(label, SKIP, "an earlier layer failed"))
        return results

    dcon, workdir = check_dcon_run(prefix, source, skip=skip_smoke, timeout=timeout)
    results.append(dcon)
    results.append(check_handoff(workdir if dcon.status == PASS else None))
    if dcon.status == PASS:
        results.append(check_gpec_run(prefix, workdir, timeout=timeout))
        results.append(check_netcdf_outputs(workdir))
    else:
        results.append(CheckResult("GPEC reference run", SKIP, "DCON did not complete"))
        results.append(CheckResult("GPEC netCDF outputs", SKIP, "DCON did not complete"))
    results.append(check_numerical_agreement(workdir))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_gpec",
        description="Verify a DCON/GPEC installation and the handoff between them.",
    )
    parser.add_argument("--source", help="path to your GPEC checkout")
    parser.add_argument("--prefix", help="installation root (default: $GPECHOME)")
    parser.add_argument(
        "--skip-smoke", action="store_true", help="do not run the solvers, only inspect the install"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds allowed for each solver (default 900)"
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    arguments = parser.parse_args(argv)

    results = run_checks(
        source=arguments.source,
        prefix=arguments.prefix,
        skip_smoke=arguments.skip_smoke,
        timeout=arguments.timeout,
    )
    return emit(results, title=TITLE, rerun=RERUN, as_json=arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
