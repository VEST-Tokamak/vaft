"""Verify a CHEASE installation, layer by layer.

Compilation alone is not support. This walks the same path a workflow does --
toolchain, source, build, the executable VAFT resolves, a real refinement of a
packaged equilibrium, and the numbers that come out -- and names the layer that
failed rather than printing compiler output at the reader.

    python install/check_chease.py --source C:/git/CHEASE
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
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
    scratch_directory,
)

TITLE = "CHEASE environment check"
RERUN = "python install/check_chease.py"
PROJECT = "CHEASE"
EXECUTABLES = ("chease",)
SOURCE_MARKERS = (
    "src-f90/Makefile",
    "src-f90/Makefile.define_FLAGS",
    "src-f90/chease_prog_effxml.f90",
)
BUILD_REMEDIATION = (
    "Build CHEASE with:\n"
    "         powershell -ExecutionPolicy Bypass -File install\\install_chease_windows.ps1 <source>"
)

#: A packaged equilibrium, so the smoke run needs no data of its own.
SAMPLE = "efit/g039915.00319"

#: `CHEASEConfig.nideal` defaults to 11, which reproduces the VEST `jsk95`
#: workflow against the CHEASE build that group uses. Upstream CHEASE accepts
#: 1 through 10, where 6 is the documented default that writes the EQDSK VAFT
#: reads back. The smoke run reports that difference rather than hiding it.
UPSTREAM_NIDEAL = 6


def check_symlinked_sources(source: Optional[str], *, materialized_hint: str) -> CheckResult:
    """CHEASE's committed symbolic links are real sources, not placeholders.

    CHEASE stores several sources as symbolic links, and one of them is
    compiled into the plain `chease` target. Git for Windows writes a small
    text file naming the target unless symlink support is on, and the build
    then fails with a Fortran syntax error that says nothing about the cause.
    """
    label = "CHEASE symbolic-link placeholders"
    if not source:
        return CheckResult(label, SKIP, "no --source given")
    directory = Path(source).expanduser() / "src-f90"
    if not directory.is_dir():
        return CheckResult(label, SKIP, f"{directory} does not exist")

    placeholders = []
    for candidate in directory.glob("*.f90"):
        try:
            if candidate.stat().st_size > 512:
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text or "\n" in text:
            continue
        if (directory / text).is_file():
            placeholders.append(candidate.name)

    if placeholders:
        return CheckResult(
            label,
            FAIL,
            f"{len(placeholders)} source file(s) are link placeholders: {', '.join(sorted(placeholders)[:4])}",
            materialized_hint,
        )
    return CheckResult(label, PASS, "every source file holds real source")


def check_vaft_discovery(prefix: Optional[str]) -> CheckResult:
    """VAFT resolves the executable through its own documented mechanism.

    This deliberately calls the adapter rather than looking in the prefix
    itself: what matters is that the code the workflow runs finds the same file
    the installer wrote.
    """
    label = "VAFT executable discovery"
    try:
        from vaft.code.chease import CHEASEConfig, find_chease_executable
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(
            label, FAIL, f"vaft.code.chease could not be imported: {error}",
            "Run install/check_vaft_environment.py first.",
        )

    previous = os.environ.get("CHEASEHOME")
    if prefix:
        os.environ["CHEASEHOME"] = str(prefix)
    try:
        resolved = find_chease_executable(CHEASEConfig())
    except Exception as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    finally:
        if prefix:
            if previous is None:
                os.environ.pop("CHEASEHOME", None)
            else:
                os.environ["CHEASEHOME"] = previous

    if resolved is None:
        return CheckResult(
            label,
            FAIL,
            "CHEASEHOME is not configured, so VAFT has nothing to run",
            "Set CHEASEHOME to the installation root, or rerun the installer "
            "without -NoEnvironmentWiring.",
        )
    return CheckResult(label, PASS, str(resolved))


def _run_refinement(prefix: Optional[str], nideal: int, workdir: Path):
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.resources import data_path

    previous = os.environ.get("CHEASEHOME")
    if prefix:
        os.environ["CHEASEHOME"] = str(prefix)
    try:
        # No explicit executable: resolution goes through $CHEASEHOME exactly as
        # it does for a notebook, so this layer exercises the real path.
        config = CHEASEConfig(
            workdir=workdir,
            create_plot=False,
            cleanup=False,
            timeout=900,
            nideal=nideal,
        )
        inputs = prepare_chease_inputs(data_path(SAMPLE), config)
        return run_chease(inputs, config)
    finally:
        if prefix:
            if previous is None:
                os.environ.pop("CHEASEHOME", None)
            else:
                os.environ["CHEASEHOME"] = previous


def check_reference_run(prefix: Optional[str], *, skip: bool) -> tuple[CheckResult, object]:
    """A packaged equilibrium is refined by the installed executable."""
    label = "CHEASE reference run"
    if skip:
        return CheckResult(label, SKIP, "requested with --skip-smoke"), None
    if not prefix:
        return CheckResult(label, SKIP, "no install prefix"), None

    workdir = Path(scratch_directory("vaft-chease-check-"))
    try:
        result = _run_refinement(prefix, 11, workdir)
    except Exception as error:
        return CheckResult(label, FAIL, f"{type(error).__name__}: {error}", BUILD_REMEDIATION), None

    if result.returncode == 0:
        return CheckResult(label, PASS, f"refined {SAMPLE} in {workdir}"), result

    log = workdir / "chease.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if "WRONG VALUE FOR NIDEAL" in text:
        # Not an installation fault: the build is fine and the two codes simply
        # disagree about one namelist value.
        retry = Path(scratch_directory("vaft-chease-check-"))
        try:
            second = _run_refinement(prefix, UPSTREAM_NIDEAL, retry)
        except Exception as error:
            return CheckResult(label, FAIL, f"{type(error).__name__}: {error}", BUILD_REMEDIATION), None
        if second.returncode == 0:
            return (
                CheckResult(
                    label,
                    WARN,
                    f"refined {SAMPLE} only with nideal={UPSTREAM_NIDEAL}; this CHEASE "
                    "rejects the VAFT default of 11",
                    "CHEASEConfig.nideal defaults to 11 to reproduce the VEST jsk95 "
                    "workflow, and this CHEASE accepts 1 to 10. Pass "
                    f"CHEASEConfig(nideal={UPSTREAM_NIDEAL}) with this build, or use the "
                    "CHEASE revision the VEST workflow was written against.",
                ),
                second,
            )
        shutil.rmtree(retry, ignore_errors=True)

    return (
        CheckResult(
            label,
            FAIL,
            first_error_line(text) or f"CHEASE exited {result.returncode}",
            f"The full run is in {workdir}.",
        ),
        None,
    )


def check_expected_outputs(result) -> CheckResult:
    """The refined equilibrium exists and VAFT could read it back."""
    label = "CHEASE expected outputs"
    if result is None:
        return CheckResult(label, SKIP, "no reference run")
    if result.refined_geqdsk is None or not Path(result.refined_geqdsk).is_file():
        return CheckResult(
            label,
            FAIL,
            "no refined equilibrium was produced",
            "CHEASE exited cleanly but wrote no EQDSK. Check its output_flag diagnostics.",
        )
    return CheckResult(label, PASS, str(result.refined_geqdsk))


def check_numerical_agreement(result) -> CheckResult:
    """The refined equilibrium is physically consistent with its input.

    These are the quantities CHEASE must preserve rather than reproduce: the
    boundary it was handed, and global scalars that a converged refinement
    moves only slightly. They are properties of the run itself, so they hold on
    any platform without needing a reference file captured elsewhere.
    """
    label = "CHEASE numerical agreement"
    if result is None or not result.comparison:
        return CheckResult(label, SKIP, "no reference run")

    comparison = dict(result.comparison)
    problems: list[str] = []
    if comparison.get("boundary_points", 0) <= 0:
        problems.append("the refined equilibrium carries no plasma boundary")
    # The adapter restores the input boundary verbatim, so any difference here
    # means the restoration did not happen.
    for key in ("boundary_r_rms", "boundary_z_rms", "boundary_rz_rms"):
        value = float(comparison.get(key, 0.0))
        if value > 1e-9:
            problems.append(f"{key}={value:.3g}, but the input boundary must be preserved exactly")
    # A converged refinement changes the profiles a little, not wildly.
    for key, limit in (("q_rms_rel", 0.25), ("pressure_rms_rel", 0.35), ("current_rel_diff", 0.25)):
        if key not in comparison:
            continue
        value = abs(float(comparison[key]))
        if value > limit:
            problems.append(f"{key}={value:.3g} exceeds {limit}")

    if problems:
        return CheckResult(
            label,
            FAIL,
            "; ".join(problems),
            "The build runs but does not reproduce the input equilibrium. Compare "
            "against a Linux reference before using this installation.",
        )
    summary = ", ".join(
        f"{key}={float(comparison[key]):.3g}"
        for key in ("q_rms_rel", "pressure_rms_rel", "current_rel_diff")
        if key in comparison
    )
    return CheckResult(label, PASS, summary or "boundary preserved")


def run_checks(
    *,
    source: Optional[str] = None,
    prefix: Optional[str] = None,
    skip_smoke: bool = False,
) -> list[CheckResult]:
    """Run every CHEASE layer, in the order a workflow depends on them."""
    if prefix is None:
        prefix = os.environ.get("CHEASEHOME")
    if prefix is None:
        candidate = default_prefix("chease")
        if candidate is not None and (candidate / "bin").is_dir():
            prefix = str(candidate)

    results = [
        check_toolchain(required=bool(source)),
        check_source_checkout(
            source, project=PROJECT, markers=SOURCE_MARKERS, remediation=BUILD_REMEDIATION
        ),
        check_source_revision(source, project=PROJECT),
        check_symlinked_sources(
            source,
            materialized_hint=(
                "Obtain CHEASE again with symlink support enabled, or rerun the "
                "installer with -MaterializeSymlinks. See install/README.md."
            ),
        ),
        check_build_record(prefix, source, project=PROJECT, remediation=BUILD_REMEDIATION),
        check_executables(prefix, EXECUTABLES, project=PROJECT, remediation=BUILD_REMEDIATION),
        check_executables_load(prefix, EXECUTABLES, project=PROJECT),
        check_vaft_discovery(prefix),
    ]
    run_result = None
    if not any(result.failed for result in results):
        reference, run_result = check_reference_run(prefix, skip=skip_smoke)
        results.append(reference)
        results.append(check_expected_outputs(run_result))
        results.append(check_numerical_agreement(run_result))
    else:
        for label in ("CHEASE reference run", "CHEASE expected outputs", "CHEASE numerical agreement"):
            results.append(CheckResult(label, SKIP, "an earlier layer failed"))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_chease",
        description="Verify a CHEASE installation and the VAFT workflow that uses it.",
    )
    parser.add_argument("--source", help="path to your CHEASE checkout")
    parser.add_argument("--prefix", help="installation root (default: $CHEASEHOME)")
    parser.add_argument(
        "--skip-smoke", action="store_true", help="do not run CHEASE, only inspect the installation"
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    arguments = parser.parse_args(argv)

    results = run_checks(
        source=arguments.source, prefix=arguments.prefix, skip_smoke=arguments.skip_smoke
    )
    return emit(results, title=TITLE, rerun=RERUN, as_json=arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
