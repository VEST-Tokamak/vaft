"""Verify a GENRAY installation, layer by layer.

Compilation alone is not support. This walks toolchain, source, build, the
executable VAFT resolves, a real run of upstream's own EC regression case
(``00_Genray_Regression_Tests/ci-tests/test-EC-ITER-Centra-CD``) and the
numbers that come out against upstream's ``gold-genray.nc``, and names the
layer that failed.

    python install/check_genray.py --source ~/git/genray
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _external_code_common import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    WARN,  # noqa: F401  (the shared vocabulary; see check_chease.py)
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

TITLE = "GENRAY environment check"
RERUN = "python install/check_genray.py"
PROJECT = "GENRAY"
EXECUTABLES = ("xgenray",)
REFERENCE_CASE = Path("00_Genray_Regression_Tests/ci-tests/test-EC-ITER-Centra-CD")
SOURCE_MARKERS = ("genray.f", "makefile_gfortran.Ubuntu", str(REFERENCE_CASE / "gold-genray.nc"))
BUILD_REMEDIATION = "Build GENRAY with:\n         bash install/install_genray.sh --source <source>"

#: Agreement demanded of the reference run against upstream's gold output.
#: Relative, on the injected/absorbed power and the driven current; absolute
#: [m] on the ray end point. A different compiler moves the last digits, not
#: these.
RELATIVE_TOLERANCE = 1.0e-3
POSITION_TOLERANCE_M = 1.0e-3
#: Seconds the reference case may take (it runs in seconds on a laptop).
REFERENCE_TIMEOUT = 600


def check_vaft_discovery(prefix: Optional[str]) -> CheckResult:
    """VAFT resolves the executable through its own documented mechanism."""
    label = "VAFT executable discovery"
    try:
        from vaft.code.genray import find_genray_executable
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(
            label, FAIL, f"vaft.code.genray could not be imported: {error}",
            "Run install/check_vaft_environment.py first.",
        )
    previous = os.environ.get("GENRAYHOME")
    if prefix:
        os.environ["GENRAYHOME"] = str(prefix)
    try:
        resolved = find_genray_executable()
    except Exception as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    finally:
        if prefix:
            if previous is None:
                os.environ.pop("GENRAYHOME", None)
            else:
                os.environ["GENRAYHOME"] = previous
    return CheckResult(label, PASS, str(resolved))


def check_reference_run(source: Optional[str], prefix: Optional[str], *, skip: bool) -> tuple[CheckResult, Optional[Path]]:
    """Run upstream's EC ITER regression case in a scratch directory."""
    label = "GENRAY reference run"
    if skip:
        return CheckResult(label, SKIP, "requested with --skip-smoke"), None
    if not source:
        return CheckResult(label, SKIP, "no --source, so the reference case is not available"), None
    if not prefix:
        return CheckResult(label, SKIP, "no install prefix"), None
    case = Path(source).expanduser() / REFERENCE_CASE
    workdir = Path(scratch_directory("vaft-genray-check-"))
    for name in ("genray.dat", "equilib.dat"):
        shutil.copy2(case / name, workdir / name)
    executable = Path(prefix).expanduser() / "bin" / "xgenray"
    try:
        completed = subprocess.run(
            [str(executable)], cwd=workdir, capture_output=True, encoding="utf-8",
            errors="replace", timeout=REFERENCE_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(label, FAIL, f"did not finish in {REFERENCE_TIMEOUT} s ({workdir})"), None
    output = workdir / "genray.nc"
    if completed.returncode != 0 or not output.is_file():
        detail = first_error_line(completed.stdout + "\n" + completed.stderr)
        return CheckResult(
            label, FAIL, f"exit {completed.returncode}, genray.nc {'present' if output.is_file() else 'absent'}: {detail}",
            BUILD_REMEDIATION,
        ), None
    return CheckResult(label, PASS, f"ran {REFERENCE_CASE.name} in {workdir}"), output


def compare_with_gold(produced: Path, gold: Path) -> list[str]:
    """Disagreements between a run's genray.nc and upstream's gold output."""
    import netCDF4
    import numpy as np

    problems: list[str] = []
    with netCDF4.Dataset(str(produced)) as ours, netCDF4.Dataset(str(gold)) as theirs:
        a, b = ours.variables, theirs.variables
        rays_a = np.asarray(a["nrayelt"][:], dtype=int).reshape(-1)
        rays_b = np.asarray(b["nrayelt"][:], dtype=int).reshape(-1)
        if rays_a.size != rays_b.size:
            return [f"{rays_a.size} rays, gold has {rays_b.size}"]
        for name in ("power_inj_total", "power_total", "powtot_e", "toroidal_cur_total"):
            x, y = float(a[name][:]), float(b[name][:])
            scale = max(abs(y), 1e-300)
            if abs(x - y) / scale > RELATIVE_TOLERANCE:
                problems.append(f"{name} {x:.6g} vs gold {y:.6g}")
        for i in range(rays_b.size):
            if rays_a[i] < 1 or rays_b[i] < 1:
                if rays_a[i] != rays_b[i]:
                    problems.append(f"ray {i} traced {rays_a[i]} points, gold {rays_b[i]}")
                continue
            end_a, end_b = rays_a[i] - 1, rays_b[i] - 1
            for name in ("wr", "wz"):
                # cm -> m
                xa = float(a[name][i, end_a]) * 1e-2
                xb = float(b[name][i, end_b]) * 1e-2
                if abs(xa - xb) > POSITION_TOLERANCE_M:
                    problems.append(f"ray {i} end {name} {xa:.5f} m vs gold {xb:.5f} m")
    return problems


def check_numerical_agreement(source: Optional[str], produced: Optional[Path]) -> CheckResult:
    label = "GENRAY numerical agreement"
    if produced is None or not source:
        return CheckResult(label, SKIP, "no reference run")
    gold = Path(source).expanduser() / REFERENCE_CASE / "gold-genray.nc"
    try:
        problems = compare_with_gold(produced, gold)
    except Exception as error:
        return CheckResult(label, FAIL, f"{type(error).__name__}: {error}")
    if problems:
        return CheckResult(
            label, FAIL, "; ".join(problems[:4]),
            "The build runs but does not reproduce upstream's reference output. Check the "
            "compiler flags in the build log before using it.",
        )
    return CheckResult(
        label, PASS,
        f"power, current and ray end points match gold-genray.nc (rel {RELATIVE_TOLERANCE:g}, {POSITION_TOLERANCE_M:g} m)",
    )


def run_checks(*, source: Optional[str] = None, prefix: Optional[str] = None, skip_smoke: bool = False) -> list[CheckResult]:
    """Run every GENRAY layer, in the order a workflow depends on them."""
    if prefix is None:
        prefix = os.environ.get("GENRAYHOME")
    if prefix is None:
        candidate = default_prefix("genray", source)
        if candidate is not None and (candidate / "bin").is_dir():
            prefix = str(candidate)

    results = [
        check_toolchain(required=bool(source)),
        check_source_checkout(source, project=PROJECT, markers=SOURCE_MARKERS, remediation=BUILD_REMEDIATION),
        check_source_revision(source, project=PROJECT),
        check_build_record(prefix, source, project=PROJECT, remediation=BUILD_REMEDIATION),
        check_executables(prefix, EXECUTABLES, project=PROJECT, remediation=BUILD_REMEDIATION),
        check_executables_load(prefix, EXECUTABLES, project=PROJECT),
        check_vaft_discovery(prefix),
    ]
    if not any(result.failed for result in results):
        reference, produced = check_reference_run(source, prefix, skip=skip_smoke)
        results.append(reference)
        results.append(check_numerical_agreement(source, produced))
    else:
        for label in ("GENRAY reference run", "GENRAY numerical agreement"):
            results.append(CheckResult(label, SKIP, "an earlier layer failed"))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_genray", description="Verify a GENRAY installation and the VAFT adapter that uses it."
    )
    parser.add_argument("--source", help="path to your GENRAY checkout")
    parser.add_argument("--prefix", help="installation root (default: $GENRAYHOME)")
    parser.add_argument("--skip-smoke", action="store_true", help="do not run GENRAY, only inspect the installation")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    arguments = parser.parse_args(argv)
    results = run_checks(source=arguments.source, prefix=arguments.prefix, skip_smoke=arguments.skip_smoke)
    return emit(results, title=TITLE, rerun=RERUN, as_json=arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
