"""Verify a NUBEAM installation, layer by layer.

NUBEAM needs more than an executable: it aborts unless PREACTDIR and ADASDIR
both resolve to populated, writable directories, and it composes every filename
in a fixed-width Fortran buffer that a long working directory overruns without
saying so. Both are checked here, because both fail as something else -- a
missing reaction table surfaces as a translate-environment-variable abort, and a
long path surfaces as a file-open failure naming the wrong file.

    python install/check_nubeam.py --source C:/git/NUBEAM
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
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
)

TITLE = "NUBEAM environment check"
RERUN = "python install/check_nubeam.py"
PROJECT = "NUBEAM"

#: What the adapter drives, in the order a case uses them: the Plasma State
#: generator, then NUBEAM itself, then the merge of its state changes.
EXECUTABLES = ("plasma_state_test", "nubeam_comp_exec", "update_state")

#: A NUBEAM source tree, as external/nubeam/macos.sh identifies one.
SOURCE_MARKERS = ("Makefile", "nubeam_comp_exec")

BUILD_REMEDIATION = (
    "Build NUBEAM with:\n"
    "         powershell -ExecutionPolicy Bypass -File external\\nubeam\\windows.ps1 <source> -AcceptNtccTerms"
)


def check_reaction_databases(prefix: Optional[str]) -> CheckResult:
    """PREACT and ADAS are present, populated, and writable.

    ``nubeam_comp_exec`` calls its own bad-exit path when either variable
    resolves to a blank string, reporting only that it failed to translate an
    environment variable. Both are also caches rather than read-only data: the
    table code writes newly computed reaction tables back into them the first
    time a reaction is needed, so a read-only copy fails partway through a run
    rather than at startup.
    """
    label = "Reaction databases"
    if not prefix:
        return CheckResult(label, SKIP, "no install prefix")

    root = Path(prefix).expanduser()
    preact = Path(os.environ.get("PREACTDIR") or root / "share" / "preact")
    adas = Path(os.environ.get("ADASDIR") or root / "share" / "adas")

    remediation = (
        "Rerun the installer, which stages both databases. They are populated "
        "from the PREACT source you downloaded, never redistributed by VAFT."
    )

    problems: list[str] = []
    for name, path in (("PREACTDIR", preact), ("ADASDIR", adas)):
        if not path.is_dir():
            problems.append(f"{name} does not exist: {path}")
    if problems:
        return CheckResult(label, FAIL, "; ".join(problems), remediation)

    # The layout the table code composes paths against.
    if not (preact / "tables").is_dir():
        problems.append(f"PREACTDIR has no tables/: {preact}")
    if not (adas / "data").is_dir():
        problems.append(f"ADASDIR has no data/: {adas}")
    if not (adas / "tables").is_dir():
        problems.append(f"ADASDIR has no tables/: {adas}")

    # Writability, probed rather than asked. os.access reports the read-only
    # attribute on Windows and not the ACL, so it answers yes where a write
    # then fails.
    for name, path in (("PREACTDIR", preact), ("ADASDIR/tables", adas / "tables")):
        if not path.is_dir():
            continue
        probe = path / ".vaft-write-probe"
        try:
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            problems.append(f"{name} is not writable: {error.strerror or error}")

    if problems:
        return CheckResult(label, FAIL, "; ".join(problems), remediation)
    return CheckResult(label, PASS, f"{preact} and {adas}, both writable")


def check_path_budget(prefix: Optional[str]) -> CheckResult:
    """A run directory can be found that NUBEAM's filename buffer survives.

    ``nubeam_comp_exec`` builds every filename in a ``character*140`` buffer.
    Overflow truncates silently, and the run then fails as a file-open error
    naming the input state rather than the path -- so this is checked before a
    run rather than diagnosed after one.
    """
    label = "Path budget"
    try:
        from vaft.code.nubeam.config import NUBEAMConfig
        from vaft.compat import short_temporary_directory
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(label, FAIL, f"vaft.code.nubeam is not importable: {error}",
                           "Run install/check_vaft_environment.py first.")

    budget = NUBEAMConfig().workdir_budget
    try:
        with short_temporary_directory(max_length=budget) as scratch:
            return CheckResult(
                label, PASS, f"{len(str(scratch))} of {budget} characters used by {scratch}"
            )
    except RuntimeError as error:
        return CheckResult(
            label,
            FAIL,
            str(error),
            "NUBEAM cannot run from a directory this deep. Set TMPDIR to a "
            "short path, or pass a shorter workdir to run_nubeam_case.",
        )


def check_vaft_discovery(prefix: Optional[str]) -> CheckResult:
    """VAFT resolves the executables through its own documented mechanism."""
    label = "VAFT executable discovery"
    try:
        from vaft.code import nubeam
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(
            label, FAIL, f"vaft.code.nubeam could not be imported: {error}",
            "Run install/check_vaft_environment.py first.",
        )

    previous = os.environ.get("NUBEAMHOME")
    if prefix:
        os.environ["NUBEAMHOME"] = str(prefix)
    try:
        resolved = nubeam.find_nubeam_executable(nubeam.NUBEAMConfig())
    except Exception as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    finally:
        if prefix:
            if previous is None:
                os.environ.pop("NUBEAMHOME", None)
            else:
                os.environ["NUBEAMHOME"] = previous

    if resolved is None:
        return CheckResult(
            label,
            FAIL,
            "NUBEAMHOME is not configured, so VAFT has nothing to run",
            "Set NUBEAMHOME to the installation root, or rerun the installer "
            "without -NoEnvironmentWiring.",
        )
    return CheckResult(label, PASS, str(resolved))


def check_imas_mapping() -> CheckResult:
    """State plainly which halves of the mapping exist.

    A checker that reports only green invites the reading that every NUBEAM
    result reaches IMAS. Most do: the plasma-side source term and the fast-ion
    population are both mapped. The Monte Carlo marker records are not, and a
    caller expecting to find them in an IDS should be told so here rather than
    by their absence.
    """
    try:
        from vaft.machine_mapping.core_sources import core_sources_from_nubeam  # noqa: F401
        from vaft.machine_mapping.distributions import distributions_from_nubeam  # noqa: F401
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(
            "IMAS mapping",
            FAIL,
            f"the NUBEAM IDS mappings could not be imported: {error}",
            "Run install/check_vaft_environment.py first.",
        )
    return CheckResult(
        "IMAS mapping",
        WARN,
        "profiles map to core_sources and distributions; the birth and "
        "lost-particle markers stay in the native container",
        "Expected. distribution_sources is the remaining half of issue #490 "
        "section 6. Read the markers through "
        "vaft.code.nubeam.collect_nubeam_outputs and vaft.plot.nubeam.",
    )


def run_checks(
    *,
    source: Optional[str] = None,
    prefix: Optional[str] = None,
    skip_smoke: bool = True,
) -> list[CheckResult]:
    """Run every NUBEAM layer, in the order a run depends on them."""
    if prefix is None:
        prefix = os.environ.get("NUBEAMHOME")
    if prefix is None:
        candidate = default_prefix("nubeam")
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
        check_reaction_databases(prefix),
        check_vaft_discovery(prefix),
        check_path_budget(prefix),
        check_imas_mapping(),
    ]
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_nubeam",
        description="Verify a NUBEAM installation and the environment it needs.",
    )
    parser.add_argument("--source", help="path to your NUBEAM source tree")
    parser.add_argument("--prefix", help="installation root (default: $NUBEAMHOME)")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="accepted for symmetry with the other checkers; a NUBEAM run needs a case",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    arguments = parser.parse_args(argv)

    results = run_checks(
        source=arguments.source, prefix=arguments.prefix, skip_smoke=arguments.skip_smoke
    )
    return emit(results, title=TITLE, rerun=RERUN, as_json=arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
