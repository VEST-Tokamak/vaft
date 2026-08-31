#!/usr/bin/env python3
"""Verify that a VAFT student/course environment is usable.

Run from the VAFT checkout::

    python install/check_vaft_environment.py

The default run is completely offline: it never contacts HSDS and never reads a
credential value. Pass ``--include-network`` to additionally probe the HSDS
endpoint using the credentials already configured in ``~/.hscfg``.

Every failing check reports a concrete corrective action rather than only a
traceback. The exit status is 0 only when no check failed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable, Iterable, NamedTuple, Optional, Sequence


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
#: Reported and visible, but does not fail the run. Used for capabilities that
#: are optional for the offline course material, such as HSDS credentials.
WARN = "WARN"

EXPECTED_ENVIRONMENT = "vaft"
KERNEL_NAME = "vaft"
KERNEL_DISPLAY_NAME = "Python (vaft)"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HSCFG_KEYS = ("hs_endpoint", "hs_username", "hs_password", "hs_api_key")


class CheckResult(NamedTuple):
    """Outcome of a single environment check."""

    name: str
    status: str
    detail: str
    remediation: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


# --------------------------------------------------------------------------
# Supported-interpreter metadata
# --------------------------------------------------------------------------


def read_requires_python(pyproject: Optional[Path] = None) -> str:
    """Return the ``requires-python`` specifier declared by the checkout.

    Parsed with ``tomllib`` when available (Python >= 3.11) and with a narrow
    regular expression otherwise, so the checker still runs on an unsupported
    interpreter -- which is exactly the case it has to diagnose.
    """
    path = Path(pyproject) if pyproject is not None else REPOSITORY_ROOT / "pyproject.toml"
    if not path.is_file():
        return ""
    try:
        import tomllib  # noqa: PLC0415 - optional, stdlib only from 3.11
    except ModuleNotFoundError:
        match = re.search(
            r'^\s*requires-python\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M
        )
        return match.group(1) if match else ""
    with path.open("rb") as handle:
        return str(tomllib.load(handle).get("project", {}).get("requires-python", ""))


def parse_version_bounds(specifier: str) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """Return ``(minimum_inclusive, maximum_exclusive)`` for a simple specifier.

    Only the ``>=X.Y`` / ``<X.Y`` forms VAFT actually uses are understood; any
    other clause is ignored rather than guessed at.
    """
    minimum: Optional[tuple[int, int]] = None
    maximum: Optional[tuple[int, int]] = None
    for clause in specifier.split(","):
        clause = clause.strip()
        match = re.fullmatch(r"(>=|<)\s*(\d+)\.(\d+)(?:\.\d+)?", clause)
        if not match:
            continue
        operator, major, minor = match.group(1), int(match.group(2)), int(match.group(3))
        if operator == ">=":
            minimum = (major, minor)
        else:
            maximum = (major, minor)
    return minimum, maximum


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_python_version(
    version: Optional[tuple[int, int]] = None, specifier: Optional[str] = None
) -> CheckResult:
    """The running interpreter is inside the supported range."""
    current = version if version is not None else sys.version_info[:2]
    declared = specifier if specifier is not None else read_requires_python()
    minimum, maximum = parse_version_bounds(declared)
    shown = f"{current[0]}.{current[1]}"
    if minimum is None and maximum is None:
        return CheckResult(
            "supported Python",
            PASS,
            f"Python {shown} (no requires-python declared)",
        )
    if (minimum is not None and current < minimum) or (maximum is not None and current >= maximum):
        return CheckResult(
            "supported Python",
            FAIL,
            f"Python {shown} is outside {declared}",
            "Recreate the environment with the interpreter pinned in environment.yml, "
            "then rerun the platform bootstrap script.",
        )
    return CheckResult("supported Python", PASS, f"Python {shown} satisfies {declared}")


def check_conda_environment(
    environment: Optional[str] = None, prefix: Optional[str] = None
) -> CheckResult:
    """The active interpreter belongs to the expected ``vaft`` environment."""
    name = environment if environment is not None else os.environ.get("CONDA_DEFAULT_ENV", "")
    active_prefix = Path(prefix) if prefix is not None else Path(sys.prefix)
    if name == EXPECTED_ENVIRONMENT or active_prefix.name == EXPECTED_ENVIRONMENT:
        return CheckResult(
            "expected Conda environment", PASS, f"active interpreter: {active_prefix}"
        )
    described = name or "<none>"
    return CheckResult(
        "expected Conda environment",
        FAIL,
        f"expected the `{EXPECTED_ENVIRONMENT}` environment, but the active "
        f"interpreter is {active_prefix} (CONDA_DEFAULT_ENV={described})",
        f"Run `conda activate {EXPECTED_ENVIRONMENT}`, then rerun this check.",
    )


def check_import(
    module: str, label: str, remediation: str, *, importer: Optional[Callable[[str], object]] = None
) -> CheckResult:
    """A required module imports successfully."""
    load = importer or importlib.import_module
    try:
        loaded = load(module)
    except Exception as error:  # noqa: BLE001 - any import failure is reportable
        return CheckResult(label, FAIL, f"{type(error).__name__}: {error}", remediation)
    version = getattr(loaded, "__version__", "")
    return CheckResult(label, PASS, f"{module} {version}".strip())


def check_vaft_location(
    module_file: Optional[Path] = None, repository_root: Optional[Path] = None
) -> CheckResult:
    """``import vaft`` resolves inside this checkout, not an unrelated copy."""
    root = Path(repository_root) if repository_root is not None else REPOSITORY_ROOT
    if module_file is None:
        try:
            spec = importlib.util.find_spec("vaft")
        except Exception as error:  # noqa: BLE001 - a broken package must report, not crash
            return CheckResult(
                "VAFT resolves to this checkout",
                FAIL,
                f"{type(error).__name__}: {error}",
                "Run the platform bootstrap script for your system.",
            )
        if spec is None or spec.origin is None:
            return CheckResult(
                "VAFT resolves to this checkout",
                FAIL,
                "vaft is not importable",
                f"Run `python -m pip install -e .` from {root}.",
            )
        located = Path(spec.origin)
    else:
        located = Path(module_file)
    located = located.resolve()
    root = root.resolve()
    if root in located.parents:
        return CheckResult("VAFT resolves to this checkout", PASS, str(located))
    return CheckResult(
        "VAFT resolves to this checkout",
        FAIL,
        f"vaft resolves to {located}, which is outside {root}",
        f"Run `python -m pip install -e .` from {root} so the cloned source tree "
        "shadows any unrelated installed copy.",
    )


def check_command(
    command: str, label: str, remediation: str, *, which: Optional[Callable[[str], Optional[str]]] = None
) -> CheckResult:
    """An executable required by the workflow is on ``PATH``."""
    locate = which or shutil.which
    found = locate(command)
    if found:
        return CheckResult(label, PASS, str(found))
    return CheckResult(label, FAIL, f"`{command}` is not on PATH", remediation)


def _kernelspec_names(runner: Optional[Callable[[Sequence[str]], str]] = None) -> list[str]:
    """Return the registered Jupyter kernel names.

    ``kernelspecs`` is a mapping keyed by kernel name, so the result never
    contains a repeat.
    """
    def _default(arguments: Sequence[str]) -> str:
        completed = subprocess.run(
            list(arguments), capture_output=True, text=True, check=True, timeout=60
        )
        return completed.stdout

    execute = runner or _default
    payload = execute([sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"])
    return sorted(json.loads(payload).get("kernelspecs", {}))


def check_vaft_kernel(
    names: Optional[Iterable[str]] = None,
    *,
    runner: Optional[Callable[[Sequence[str]], str]] = None,
) -> CheckResult:
    """The ``Python (vaft)`` kernel is registered.

    Duplication is not checked because it cannot occur: Jupyter keys
    kernelspecs by name, and the bootstrap always registers with a fixed
    ``--name vaft``, so a repeated run replaces the spec rather than adding
    one. Presence is therefore the whole question.
    """
    remediation = (
        f'Run `python -m ipykernel install --user --name {KERNEL_NAME} '
        f'--display-name "{KERNEL_DISPLAY_NAME}"`, then rerun this check.'
    )
    if names is None:
        try:
            names = _kernelspec_names(runner)
        except Exception as error:  # noqa: BLE001 - jupyter may be missing entirely
            return CheckResult(
                "Python (vaft) kernel", FAIL, f"{type(error).__name__}: {error}", remediation
            )
    if KERNEL_NAME in set(names):
        return CheckResult("Python (vaft) kernel", PASS, f"kernel `{KERNEL_NAME}` is registered")
    return CheckResult(
        "Python (vaft) kernel", FAIL, f"no `{KERNEL_NAME}` kernel is registered", remediation
    )


def check_hsds_configuration(
    path: Optional[Path] = None, *, required: bool = False
) -> CheckResult:
    """``~/.hscfg`` exists and names the keys h5pyd needs.

    Missing credentials are a warning, not a failure: the whole offline course
    -- including Tutorial 01 -- runs from data packaged in the repository. They
    become a failure only when a network probe was explicitly requested.

    Only key names are reported. Credential values are never read into the
    report, printed, or returned.
    """
    config = Path(path) if path is not None else Path.home() / ".hscfg"
    remediation = "Run `hsconfigure`, then rerun this check."
    missing_status = FAIL if required else WARN
    if not config.is_file():
        return CheckResult(
            "HSDS configuration",
            missing_status,
            f"{config} does not exist; needed only for remote database access",
            remediation,
        )
    present = []
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in HSCFG_KEYS and stripped.split("=", 1)[1].strip():
            present.append(key)
    if "hs_endpoint" not in present:
        return CheckResult(
            "HSDS configuration",
            missing_status,
            f"{config} does not set hs_endpoint",
            remediation,
        )
    return CheckResult(
        "HSDS configuration", PASS, f"{config} sets: {', '.join(sorted(present))}"
    )


def check_hsds_connection(probe: Optional[Callable[[], bool]] = None) -> CheckResult:
    """The configured HSDS endpoint answers and reports READY.

    Delegates to :func:`vaft.database.utils.is_connect`, the helper the rest of
    VAFT already uses, rather than reimplementing the probe.
    """
    remediation = (
        "Check the endpoint and credentials with `hsconfigure`, confirm network "
        "access to the VEST HSDS server, then rerun with --include-network."
    )
    if probe is None:
        try:
            from vaft.database.utils import is_connect  # noqa: PLC0415 - optional, network path
        except Exception as error:  # noqa: BLE001
            return CheckResult(
                "HSDS connection", FAIL, f"{type(error).__name__}: {error}", remediation
            )
        probe = is_connect
    try:
        connected = bool(probe())
    except Exception as error:  # noqa: BLE001 - any transport failure is reportable
        return CheckResult("HSDS connection", FAIL, f"{type(error).__name__}: {error}", remediation)
    if connected:
        return CheckResult("HSDS connection", PASS, "server state is READY")
    return CheckResult("HSDS connection", FAIL, "the server did not report READY", remediation)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_checks(*, include_network: bool = False) -> list[CheckResult]:
    """Run the offline checks, plus the network probe when requested."""
    results = [
        check_python_version(),
        check_conda_environment(),
        check_import(
            "vaft",
            "VAFT import",
            f"Run `python -m pip install -e .` from {REPOSITORY_ROOT}.",
        ),
        check_vaft_location(),
        check_import(
            "h5pyd",
            "h5pyd / HSDS client",
            "Rerun the platform bootstrap script for your system.",
        ),
        check_command(
            "hsconfigure",
            "HSDS command-line tools",
            "The h5pyd command-line tools are missing from PATH. Activate the "
            "`vaft` environment, or rerun the platform bootstrap script.",
        ),
        check_import(
            "jupyterlab",
            "JupyterLab",
            "Run `python -m pip install jupyterlab`, or rerun the platform bootstrap script.",
        ),
        check_import(
            "ipykernel",
            "ipykernel",
            "Run `python -m pip install ipykernel`, or rerun the platform bootstrap script.",
        ),
        check_vaft_kernel(),
        check_command(
            "git",
            "Git",
            "Install Git, then reopen your shell. See install/README.md for the "
            "per-platform instructions.",
        ),
        check_hsds_configuration(required=include_network),
    ]
    if include_network:
        results.append(check_hsds_connection())
    else:
        results.append(
            CheckResult(
                "HSDS connection",
                SKIP,
                "offline run; pass --include-network to probe the server",
            )
        )
    return results


def format_report(results: Sequence[CheckResult]) -> str:
    """Render the human-readable PASS/FAIL report."""
    lines = ["VAFT environment check", "----------------------"]
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
        plural = "capability is" if len(warnings) == 1 else "capabilities are"
        lines.append(
            f"{len(warnings)} optional {plural} unconfigured; "
            "the offline course material works without that."
        )
    if failures:
        lines.append(f"{len(failures)} check(s) failed. Fix the actions above and rerun:")
        lines.append("  python install/check_vaft_environment.py")
    else:
        lines.append("All checks passed. Launch the tutorial with:")
        lines.append("  jupyter lab")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_vaft_environment",
        description="Verify a VAFT course environment. Offline by default.",
    )
    parser.add_argument(
        "--include-network",
        action="store_true",
        help="also probe the configured HSDS endpoint (requires ~/.hscfg)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="emit machine-readable JSON"
    )
    arguments = parser.parse_args(argv)

    results = run_checks(include_network=arguments.include_network)
    if arguments.as_json:
        print(
            json.dumps(
                {
                    "repository_root": str(REPOSITORY_ROOT),
                    "checks": [result._asdict() for result in results],
                },
                indent=2,
            )
        )
    else:
        print(format_report(results))
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
