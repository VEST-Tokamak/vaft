"""Verify a GACODE installation, layer by layer.

GACODE differs from the other external codes VAFT drives in three ways that
each fail as something else, so each gets its own check here:

* It **builds in place**. There is no installation prefix: the executables land
  inside the source tree, so ``$GACODEHOME`` is the checkout, and "source" and
  "prefix" are the same path.
* Every suite member has **its own ``bin``** -- ``neo/bin/neo``, not
  ``bin/neo`` -- so the shared ``check_executables`` layout does not apply.
* The launcher shells out to ``neo_parse.py``, which imports ``gacodeinput``
  from ``f2py/pygacode``. When that import fails the launcher carries on and
  NEO aborts on a missing ``input.neo.gen``, blaming the wrong thing entirely.

    python install/check_gacode.py --source ~/git/gacode
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
    check_source_checkout,
    check_source_revision,
    check_toolchain,
    emit,
)

TITLE = "GACODE environment check"
RERUN = "python install/check_gacode.py"
PROJECT = "GACODE"

#: Suite members VAFT can drive today. TGLF and CGYRO are issue #553.
CODES = ("neo",)

#: What a GACODE checkout looks like.
SOURCE_MARKERS = ("Makefile", "shared/bin/gacode_setup", "platform/build", "neo/src")

BUILD_REMEDIATION = (
    "Build GACODE with:\n"
    "         bash external/gacode/macos.sh --gacode-root <source> --check"
)


def _root(prefix: Optional[str]) -> Optional[Path]:
    if not prefix:
        return None
    return Path(prefix).expanduser()


def check_suite_executables(prefix: Optional[str]) -> CheckResult:
    """Each suite member's launcher and its compiled binary both exist.

    The launcher is a shell script that ships with the source, so it is present
    even before a build; only the binary beside it proves the build happened.
    Checking just the launcher would pass on an unbuilt checkout.
    """
    label = f"{PROJECT} executables"
    root = _root(prefix)
    if root is None:
        return CheckResult(label, FAIL, "no installation root to look in", BUILD_REMEDIATION)

    problems: list[str] = []
    found: list[str] = []
    for code in CODES:
        launcher = root / code / "bin" / code
        binary = root / code / "src" / code
        if not launcher.is_file():
            problems.append(f"missing launcher {launcher.relative_to(root)}")
            continue
        if not binary.is_file():
            problems.append(
                f"{code} is not built: {binary.relative_to(root)} does not exist"
            )
            continue
        if binary.stat().st_size == 0:
            problems.append(f"{binary.name} is empty, which is what a failed link leaves")
            continue
        found.append(code)
    if problems:
        return CheckResult(label, FAIL, "; ".join(problems), BUILD_REMEDIATION)
    return CheckResult(label, PASS, f"{', '.join(found)} in {root}")


def check_platform(prefix: Optional[str]) -> CheckResult:
    """A platform tag is set and this installation carries it.

    An unset or wrong ``GACODE_PLATFORM`` fails inside ``neo/bin/neo`` without
    naming the variable, so it is worth failing here instead.
    """
    label = "GACODE platform"
    root = _root(prefix)
    platform = os.environ.get("GACODE_PLATFORM")
    if root is None:
        return CheckResult(label, SKIP, "no installation root")
    build = root / "platform" / "build"
    known = sorted(
        entry.name[len("make.inc."):]
        for entry in build.iterdir()
        if entry.is_file() and entry.name.startswith("make.inc.")
    ) if build.is_dir() else []
    if not platform:
        return CheckResult(
            label,
            FAIL,
            "GACODE_PLATFORM is not set",
            "Set it to the tag you built with, for example "
            "GFORTRAN_OSX_BREW on macOS. It selects platform/exec/exec.$GACODE_PLATFORM, "
            "which the launcher execs.",
        )
    if known and platform not in known:
        return CheckResult(
            label,
            FAIL,
            f"GACODE_PLATFORM={platform} is not one this installation provides",
            f"Available: {', '.join(known)}.",
        )
    return CheckResult(label, PASS, platform)


def check_pygacode(prefix: Optional[str]) -> CheckResult:
    """``gacodeinput`` is importable from the tree, for the launcher's parse step."""
    label = "GACODE input parser"
    root = _root(prefix)
    if root is None:
        return CheckResult(label, SKIP, "no installation root")
    module = root / "f2py" / "pygacode" / "gacodeinput.py"
    if not module.is_file():
        return CheckResult(
            label,
            FAIL,
            f"{module} is missing, so neo_parse.py cannot run",
            "The launcher does not stop when its parse step fails; NEO then aborts on "
            "a missing input.neo.gen instead. Check out the full GACODE tree.",
        )
    return CheckResult(label, PASS, str(module.parent))


def check_vaft_discovery(prefix: Optional[str]) -> CheckResult:
    """VAFT resolves the launcher through its own documented mechanism."""
    label = "VAFT executable discovery"
    try:
        from vaft.code import gacode
    except Exception as error:  # pragma: no cover - import environment problem
        return CheckResult(
            label, FAIL, f"vaft.code.gacode could not be imported: {error}",
            "Run install/check_vaft_environment.py first.",
        )

    previous = os.environ.get("GACODEHOME")
    if prefix:
        os.environ["GACODEHOME"] = str(prefix)
    try:
        resolved = gacode.find_gacode_executable(gacode.GACODEConfig(), "neo")
    except Exception as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    finally:
        if prefix:
            if previous is None:
                os.environ.pop("GACODEHOME", None)
            else:
                os.environ["GACODEHOME"] = previous

    if resolved is None:
        return CheckResult(
            label,
            FAIL,
            "GACODEHOME is not configured, so VAFT has nothing to run",
            "Set GACODEHOME to the GACODE checkout you built.",
        )
    return CheckResult(label, PASS, str(resolved))


def check_regression(prefix: Optional[str], *, skip: bool) -> CheckResult:
    """Run NEO's shipped reg18 case and compare its precision scalar.

    This is the only check that proves the build actually computes, rather than
    merely linking.
    """
    label = "NEO reg18 regression"
    if skip:
        return CheckResult(label, SKIP, "--skip-smoke")
    root = _root(prefix)
    if root is None:
        return CheckResult(label, SKIP, "no installation root")
    case = root / "neo" / "tools" / "input" / "reg18"
    if not case.is_dir():
        return CheckResult(label, SKIP, f"{case} is not in this checkout")

    import shutil
    import tempfile

    try:
        from vaft.code.gacode._input_gacode import read_input_gacode
        from vaft.code.gacode.neo import NEOConfig, run_neo_case
    except Exception as error:  # pragma: no cover
        return CheckResult(label, FAIL, f"the VAFT adapter could not be imported: {error}")

    expected = float((case / "out.neo.prec").read_text().split()[0])
    scratch = tempfile.mkdtemp(prefix="vaft-gacode-reg18-")
    try:
        config = NEOConfig(
            home=str(root),
            platform=os.environ.get("GACODE_PLATFORM"),
            n_species=3,
            rotation_model=2,
        )
        result = run_neo_case(
            read_input_gacode(case / "input.gacode"), Path(scratch) / "reg18", config
        )
        actual = result.outputs_native.precision
    except Exception as error:
        return CheckResult(label, FAIL, str(error), BUILD_REMEDIATION)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if actual is None:
        return CheckResult(label, FAIL, "NEO wrote no out.neo.prec", BUILD_REMEDIATION)
    if abs(actual / expected - 1.0) > 1e-6:
        return CheckResult(
            label,
            FAIL,
            f"reg18 gave {actual:.8g} against the shipped {expected:.8g}",
            "The build links but does not reproduce GACODE's own reference. Check the "
            "platform file's compiler flags.",
        )
    return CheckResult(label, PASS, f"{actual:.8g} matches the shipped reference")


def check_imas_mapping() -> CheckResult:
    """State plainly which half of the picture exists.

    A checker reporting only green would suggest NEO results reach IMAS. They do
    not yet: the native container is complete, and the audit that decides which
    quantities have a defensible IDS home is deliberately still open.
    """
    return CheckResult(
        "IMAS mapping",
        WARN,
        "NEO results stop at the native NeoOutputs container; nothing is written to an IDS",
        "Expected. The core_profiles/core_transport mapping is phase 5 of issue #550 "
        "and is audited by physical definition, not by field name. Read results through "
        "vaft.code.gacode.neo.collect_neo_outputs.",
    )


def run_checks(
    *,
    source: Optional[str] = None,
    prefix: Optional[str] = None,
    skip_smoke: bool = False,
) -> list[CheckResult]:
    """Run every GACODE layer, in the order a run depends on them."""
    # GACODE builds in place, so the source tree is the installation root.
    if prefix is None:
        prefix = source or os.environ.get("GACODEHOME") or os.environ.get("GACODE_ROOT")
    if source is None:
        source = prefix

    return [
        check_toolchain(required=bool(source)),
        check_source_checkout(
            source, project=PROJECT, markers=SOURCE_MARKERS, remediation=BUILD_REMEDIATION
        ),
        check_source_revision(source, project=PROJECT),
        check_suite_executables(prefix),
        check_platform(prefix),
        check_pygacode(prefix),
        check_vaft_discovery(prefix),
        check_regression(prefix, skip=skip_smoke),
        check_imas_mapping(),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_gacode",
        description="Verify a GACODE installation and the environment it needs.",
    )
    parser.add_argument("--source", help="path to your GACODE checkout")
    parser.add_argument(
        "--prefix",
        help="installation root (default: --source, then $GACODEHOME, then $GACODE_ROOT)",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true", help="do not run the reg18 regression case"
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    arguments = parser.parse_args(argv)

    results = run_checks(
        source=arguments.source, prefix=arguments.prefix, skip_smoke=arguments.skip_smoke
    )
    return emit(results, title=TITLE, rerun=RERUN, as_json=arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
