"""Contract for the cross-platform bootstrap under ``install/`` (issue #225).

These tests keep three promises the student-facing bootstrap makes:

* the checker diagnoses a broken environment instead of only crashing;
* nothing under ``install/`` can destroy a student's local work; and
* ``environment.yml`` and ``pyproject.toml`` cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "install"
CHECKER = INSTALL / "check_vaft_environment.py"

POSIX_SCRIPTS = (
    "linux.sh", "macos.sh", "windows_wsl.sh", "uninstall.sh", "_common.sh",
    "_external_code_common.sh", "install_chease.sh", "install_efit.sh", "install_gpec.sh",
)
PLATFORM_SCRIPTS = ("linux.sh", "macos.sh", "windows_wsl.sh", "windows_native.ps1")
# Removal is identical on every POSIX platform, so it needs one entry point,
# not one per platform.
UNINSTALL_SCRIPTS = ("uninstall.sh", "uninstall_windows_native.ps1")
# The external Fortran codes are their own entry points, deliberately separate
# from the VAFT bootstrap: building them takes tens of minutes and needs a
# compiler toolchain, neither of which belongs in the path a student runs first.
EXTERNAL_CODE_WINDOWS_SCRIPTS = (
    "install_chease_windows.ps1",
    "install_efit_windows.ps1",
    "install_gpec_windows.ps1",
)
#: Their POSIX counterparts. Each code has one script covering Linux and macOS,
#: because the two differ by a compiler prefix rather than by a build model --
#: which is why these carry no platform suffix and the PowerShell ones do.
EXTERNAL_CODE_POSIX_SCRIPTS = (
    "install_chease.sh",
    "install_efit.sh",
    "install_gpec.sh",
)
#: Rules that hold for an external-code installer whatever it is written in.
EXTERNAL_CODE_SCRIPTS = (
    *EXTERNAL_CODE_WINDOWS_SCRIPTS,
    *EXTERNAL_CODE_POSIX_SCRIPTS,
)
EXTERNAL_CODE_CHECKERS = (
    "check_chease.py",
    "check_efit.py",
    "check_gacode.py",
    "check_gpec.py",
    "check_nubeam.py",
)
POWERSHELL_SCRIPTS = (
    "windows_native.ps1",
    "uninstall_windows_native.ps1",
    "_external_code_common.ps1",
    *EXTERNAL_CODE_WINDOWS_SCRIPTS,
)

#: Commands that would obtain or move an external checkout for the operator.
#: Issue #226 keeps source acquisition a separate, explicit act: provenance is
#: something the operator states, not something a script infers.
ACQUISITIVE = (
    re.compile(r"git\s+clone"),
    re.compile(r"git\s+pull"),
    re.compile(r"git\s+fetch"),
    re.compile(r"git\s+submodule"),
)


def _usable_bash() -> str | None:
    """Absolute path to a POSIX bash that actually runs, or None.

    ``shutil.which("bash")`` is not a usable guard on Windows, for two
    independent reasons:

    * It answers a different question than ``subprocess`` asks. ``which``
      searches ``PATH``; ``CreateProcess`` searches the application directory
      and ``System32`` *first*, so a bare ``["bash", ...]`` runs
      ``C:\\Windows\\System32\\bash.exe`` -- the WSL launcher -- no matter what
      ``which`` found.
    * That launcher ships with every modern Windows install, including ones
      with no distribution, where it fails with
      ``Bash/Service/CreateInstance/MountDisk/HCS/ERROR_PATH_NOT_FOUND``.

    Together those turn "bash is available" into a wall of confusing failures
    rather than honest skips. The interpreter is therefore pinned to an
    absolute path and *proven to run* before any test relying on it is enabled.
    """
    candidates = [shutil.which("bash")]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        # The WSL launcher lives in System32 and is never a POSIX bash.
        if os.name == "nt" and Path(candidate).parent.name.lower() == "system32":
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "exit 0"], capture_output=True, timeout=60
            )
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    return None


BASH = _usable_bash()
requires_bash = pytest.mark.skipif(BASH is None, reason="no working POSIX bash")


def _load_checker() -> ModuleType:
    """Import the checker by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("vaft_environment_checker", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


#: The external Fortran codes that keep a directory of their own under
#: `install/`, because a build recipe is more than one file: it carries its
#: own README, reference cases and validation notes. This is not the axis
#: #225's flatness rule is about -- that one forbids splitting the *bootstrap*
#: by platform or by role, which is what a student would have to navigate.
EXTERNAL_CODE_DIRECTORIES = ("gacode", "nubeam")


def test_install_directory_is_flat_and_complete():
    """Issue #225 requires a flat install/ with one entry point per platform.

    Flat means no `install/linux/`, no `install/checkers/`: the bootstrap a
    student runs first is one directory of scripts. A per-code build recipe is
    a different axis and keeps its own directory, listed above.
    """
    assert INSTALL.is_dir()
    expected = (
        *PLATFORM_SCRIPTS,
        *UNINSTALL_SCRIPTS,
        *EXTERNAL_CODE_SCRIPTS,
        *EXTERNAL_CODE_CHECKERS,
        "_common.sh",
        "_external_code_common.ps1",
        "_external_code_common.sh",
        "_external_code_common.py",
        "README.md",
        "check_vaft_environment.py",
    )
    for name in expected:
        assert (INSTALL / name).is_file(), f"install/{name} is missing"
    for code in EXTERNAL_CODE_DIRECTORIES:
        assert (INSTALL / code).is_dir(), f"install/{code}/ is missing"
        assert (INSTALL / code / "README.md").is_file(), (
            f"install/{code}/ must say what it builds and how"
        )
    subdirectories = [
        child.name
        for child in INSTALL.iterdir()
        if child.is_dir()
        and child.name != "__pycache__"
        and child.name not in EXTERNAL_CODE_DIRECTORIES
    ]
    assert not subdirectories, (
        "install/ must stay flat: no platform or checker subdirectories, found "
        f"{subdirectories}. A new external code's recipe goes in "
        "EXTERNAL_CODE_DIRECTORIES."
    )


def test_platform_wrappers_are_thin():
    """Shared logic belongs in _common.sh, not copied into each wrapper."""
    entry_points = {
        "linux.sh": "vaft_bootstrap_main",
        "macos.sh": "vaft_bootstrap_main",
        "windows_wsl.sh": "vaft_bootstrap_main",
        "uninstall.sh": "vaft_uninstall_main",
    }
    for name, entry_point in entry_points.items():
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "_common.sh" in text, f"install/{name} must source install/_common.sh"
        assert entry_point in text
        code_lines = [
            line
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(code_lines) < 25, f"install/{name} is doing too much itself"


# ---------------------------------------------------------------------------
# Checker unit tests
# ---------------------------------------------------------------------------


def test_python_version_bounds_are_parsed():
    assert checker.parse_version_bounds(">=3.10,<3.14") == ((3, 10), (3, 14))
    assert checker.parse_version_bounds("") == (None, None)


def test_requires_python_is_read_from_the_checkout():
    assert checker.read_requires_python() == ">=3.10,<3.14"


@pytest.mark.parametrize("version", [(3, 9), (3, 14)])
def test_unsupported_python_fails_with_remediation(version):
    result = checker.check_python_version(version=version, specifier=">=3.10,<3.14")
    assert result.failed
    assert result.remediation


def test_supported_python_passes():
    result = checker.check_python_version(version=(3, 12), specifier=">=3.10,<3.14")
    assert result.status == checker.PASS


def test_wrong_conda_environment_is_detected():
    """A student running the base environment must be told, not left guessing."""
    result = checker.check_conda_environment(environment="base", prefix="/opt/miniconda3")
    assert result.failed
    assert "vaft" in result.detail
    assert "conda activate vaft" in result.remediation


def test_expected_conda_environment_passes_by_prefix():
    result = checker.check_conda_environment(environment="", prefix="/opt/miniconda3/envs/vaft")
    assert result.status == checker.PASS


def test_vaft_outside_the_checkout_is_detected(tmp_path):
    """An unrelated installed copy shadowing the clone is the classic failure."""
    stray = tmp_path / "site-packages" / "vaft" / "__init__.py"
    stray.parent.mkdir(parents=True)
    stray.touch()
    result = checker.check_vaft_location(module_file=stray, repository_root=ROOT)
    assert result.failed
    assert "pip install -e ." in result.remediation


def test_vaft_inside_the_checkout_passes():
    result = checker.check_vaft_location(
        module_file=ROOT / "vaft" / "__init__.py", repository_root=ROOT
    )
    assert result.status == checker.PASS


def test_missing_module_reports_remediation():
    def explode(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    result = checker.check_import("nope", "Nope", "Install it.", importer=explode)
    assert result.failed
    assert result.remediation == "Install it."


def test_missing_command_reports_remediation():
    result = checker.check_command("nope", "Nope", "Install it.", which=lambda _: None)
    assert result.failed


def test_kernel_names_cannot_repeat():
    """Jupyter keys kernelspecs by name, so duplication is not a reachable state.

    The bootstrap's fixed `--name vaft` is what makes a repeated run replace the
    spec rather than add one; this pins the assumption the checker relies on.
    """
    payload = '{"kernelspecs": {"python3": {}, "vaft": {}}}'
    names = checker._kernelspec_names(lambda _arguments: payload)
    assert names == ["python3", "vaft"]
    assert len(names) == len(set(names))


def test_single_kernel_passes():
    assert checker.check_vaft_kernel(names=["python3", "vaft"]).status == checker.PASS


def test_missing_kernel_reports_the_install_command():
    result = checker.check_vaft_kernel(names=["python3"])
    assert result.failed
    assert "ipykernel install" in result.remediation


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------


def test_missing_hsds_configuration_warns_but_does_not_fail_offline(tmp_path):
    """The offline course works without credentials, so this must not block it."""
    result = checker.check_hsds_configuration(path=tmp_path / ".hscfg")
    assert result.status == checker.WARN
    assert not result.failed
    assert "hsconfigure" in result.remediation


def test_missing_hsds_configuration_fails_when_a_network_probe_was_requested(tmp_path):
    result = checker.check_hsds_configuration(path=tmp_path / ".hscfg", required=True)
    assert result.failed
    assert "hsconfigure" in result.remediation


def test_offline_run_passes_without_any_credentials(monkeypatch, tmp_path):
    """CI has no ~/.hscfg and must still get a clean offline result."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    results = checker.run_checks(include_network=False)
    configuration = next(item for item in results if item.name == "HSDS configuration")
    assert configuration.status == checker.WARN


def test_hsds_configuration_never_reports_credential_values(tmp_path):
    """Key names may be reported. Values must never leave the file."""
    config = tmp_path / ".hscfg"
    config.write_text(
        "hs_endpoint = https://hsds.example\n"
        "hs_username = a_student\n"
        "hs_password = sup3r-s3cret\n",
        encoding="utf-8",
    )
    result = checker.check_hsds_configuration(path=config)
    assert result.status == checker.PASS
    rendered = checker.format_report([result])
    for secret in ("sup3r-s3cret", "a_student", "https://hsds.example"):
        assert secret not in rendered
    assert "hs_password" in result.detail  # the key name is useful; the value is not


def test_offline_run_skips_the_network_probe(monkeypatch):
    """The default run must not touch the network."""
    def forbidden():
        raise AssertionError("the offline checker probed the network")

    monkeypatch.setattr(checker, "check_hsds_connection", lambda *a, **k: forbidden())
    results = checker.run_checks(include_network=False)
    connection = next(item for item in results if item.name == "HSDS connection")
    assert connection.status == checker.SKIP


def test_network_probe_delegates_to_the_shared_helper():
    assert checker.check_hsds_connection(probe=lambda: True).status == checker.PASS
    failed = checker.check_hsds_connection(probe=lambda: False)
    assert failed.failed and "hsconfigure" in failed.remediation


def test_connection_errors_are_reported_not_raised():
    def explode():
        raise OSError("no route to host")

    result = checker.check_hsds_connection(probe=explode)
    assert result.failed
    assert "OSError" in result.detail


# ---------------------------------------------------------------------------
# Non-destructive guarantees
# ---------------------------------------------------------------------------


DESTRUCTIVE = (
    re.compile(r"git\s+reset"),
    re.compile(r"git\s+clean"),
    re.compile(r"git\s+checkout"),
    re.compile(r"git\s+stash"),
    re.compile(r"git\s+restore"),
)


_POWERSHELL_BLOCK_COMMENT = re.compile(r"<#.*?#>", re.S)
_LINE_COMMENT = re.compile(r"(?m)#.*$")


def _executable_source(path: Path) -> str:
    """Return the file with comments stripped.

    The non-destructive promise is *written down* in these files' headers, so a
    naive substring scan would flag the very documentation that makes the
    promise. Only code is interesting here.
    """
    text = path.read_text(encoding="utf-8")
    text = _POWERSHELL_BLOCK_COMMENT.sub("", text)
    return _LINE_COMMENT.sub("", text)


def _install_sources():
    for path in sorted(INSTALL.iterdir()):
        if path.is_file() and path.name != "README.md":
            yield path


def test_bootstrap_tooling_never_runs_destructive_git_commands():
    """Student work must never be stashed, reset, cleaned, or discarded for them."""
    for path in _install_sources():
        text = _executable_source(path)
        for pattern in DESTRUCTIVE:
            assert not pattern.search(text), f"{path.name} contains `{pattern.pattern}`"


def test_bootstrap_tooling_never_accepts_credential_flags():
    for path in _install_sources():
        text = _executable_source(path)
        assert "--password" not in text, f"{path.name} accepts a --password flag"
        assert "--username" not in text, f"{path.name} accepts a --username flag"


def test_kernel_registration_is_pinned_to_one_name():
    """`--name vaft` overwrites in place, which is what prevents duplicates."""
    for name in PLATFORM_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        if "ipykernel install" not in text:
            continue
        assert "--name" in text
        assert "KernelName" in text or "VAFT_KERNEL_NAME" in text


# ---------------------------------------------------------------------------
# Documentation contract
# ---------------------------------------------------------------------------


def test_powershell_helper_is_always_called_with_an_argument_array():
    """A bare `-e` at the call site binds to the *function*, not to conda.

    PowerShell resolves parameter names before the arguments reach the command
    being run, so `Invoke-InVaft python -m pip install -e .` fails with an
    ambiguous-parameter error. Every call must pass one array literal.
    """
    for name in POWERSHELL_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("Invoke-InVaft") and "= Invoke-InVaft" not in stripped:
                continue
            call = stripped.split("Invoke-InVaft", 1)[1].strip()
            assert call.startswith("@("), (
                f"install/{name}: pass an argument array, not bare flags: {stripped}"
            )


def test_readme_documents_the_update_path():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    for fragment in ("git status", "git pull --ff-only", "check_vaft_environment.py"):
        assert fragment in text
    assert "pip install -e ." in text


def test_readme_documents_conflict_recovery_without_destructive_advice():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    assert 'git stash push -m "before VAFT update"' in text
    assert "git stash pop" in text
    assert "can itself produce conflicts" in text
    # The destructive commands appear only inside an explicit warning.
    warning = text.split("**If you hit conflicts, stop.**", 1)
    assert len(warning) == 2
    assert "git reset --hard" in warning[1].split("\n\n", 1)[0]


def test_readme_includes_the_reusable_agent_prompt():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    assert "I updated the VAFT repository and now have Git conflicts." in text
    assert "do not discard or overwrite my work" in text


def test_readme_covers_every_platform_and_the_wsl_limitation():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    for name in PLATFORM_SCRIPTS:
        assert name in text
    assert "WSL2 is never required" in text
    assert "manually" in text  # the WSL2 verification caveat


def test_readme_explains_editable_installation_and_dependency_ownership():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    assert "does not imply that you are\nexpected to develop VAFT itself" in text
    assert "single source of truth" in text


# ---------------------------------------------------------------------------
# environment.yml <-> pyproject.toml
# ---------------------------------------------------------------------------


def _environment_specification() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))


def _conda_package_names(specification: dict) -> set[str]:
    names = set()
    for entry in specification.get("dependencies", []):
        if isinstance(entry, dict):  # the optional `pip:` block
            for requirement in entry.get("pip", []):
                names.add(re.split(r"[<>=!\[ ]", str(requirement), 1)[0].strip().lower())
            continue
        names.add(re.split(r"[<>=! ]", str(entry), 1)[0].strip().lower())
    return names


def test_environment_declares_the_expected_environment_name():
    assert _environment_specification()["name"] == "vaft"


def test_environment_python_pin_satisfies_requires_python():
    specification = _environment_specification()
    pin = next(
        entry
        for entry in specification["dependencies"]
        if isinstance(entry, str) and entry.startswith("python")
    )
    match = re.fullmatch(r"python\s*=\s*(\d+)\.(\d+)", pin)
    assert match, f"pin the course interpreter as `python=X.Y`, got {pin!r}"
    version = (int(match.group(1)), int(match.group(2)))
    minimum, maximum = checker.parse_version_bounds(checker.read_requires_python())
    assert minimum is not None and maximum is not None
    assert minimum <= version < maximum


def test_environment_does_not_duplicate_project_dependencies():
    """pyproject.toml owns every Python dependency; environment.yml must not restate one."""
    tomllib = pytest.importorskip("tomllib")
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    declared = {
        re.split(r"[<>=!;\[ ]", requirement, 1)[0].strip().lower()
        for requirement in project["dependencies"]
    }
    overlap = _conda_package_names(_environment_specification()) & declared
    assert not overlap, (
        f"{sorted(overlap)} appear in both environment.yml and pyproject.toml; "
        "pyproject.toml is the single source of truth"
    )


# ---------------------------------------------------------------------------
# Script syntax
# ---------------------------------------------------------------------------


@requires_bash
@pytest.mark.parametrize("name", POSIX_SCRIPTS)
def test_posix_scripts_parse(name):
    subprocess.run([BASH, "-n", str(INSTALL / name)], check=True, timeout=60)


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell parsing is checked on Windows")
@pytest.mark.parametrize("name", POWERSHELL_SCRIPTS)
def test_powershell_script_parses(name):
    script = INSTALL / name
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$null = [ScriptBlock]::Create((Get-Content -Raw '{script}'))",
        ],
        check=True,
        timeout=120,
    )


def test_windows_recreate_contract_is_scoped_and_detects_python_mismatch():
    text = (INSTALL / "windows_native.ps1").read_text(encoding="utf-8")
    assert "[switch] $Recreate" in text
    assert "conda env remove --name $EnvironmentName --yes" in text
    assert "Get-PinnedPython" in text
    assert "Get-EnvironmentPython" in text
    assert "$pinned -ne $current" in text
    assert "windows_native.ps1 -Recreate" in text


def test_windows_bootstrap_initializes_native_status_and_reports_progress():
    text = (INSTALL / "windows_native.ps1").read_text(encoding="utf-8")
    assert "$global:LASTEXITCODE = 0" in text
    assert "function Write-Step" in text
    for step in ("Creating", "Updating", "Installing", "Registering", "Verifying"):
        assert f'Write-Step "{step}' in text or f"Write-Step '{step}" in text


def test_windows_recreate_is_documented():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    assert "windows_native.ps1 -Recreate" in text
    assert "removes and recreates the `vaft` Conda environment only" in text
    assert "cannot be combined with `-CheckOnly`" in text


def test_checker_help_runs_without_side_effects():
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "--include-network" in completed.stdout


# ---------------------------------------------------------------------------
# End-to-end shell behaviour against a recording `conda` stub
# ---------------------------------------------------------------------------
#
# The real bootstrap builds a Conda environment, which is far too slow and far
# too invasive for a unit test. Substituting a recording stub for `conda` still
# exercises the actual shell control flow: the create/update branch, the step
# ordering, the working directory of the editable install, and the promise that
# no environment other than `vaft` is ever touched.


FAKE_CONDA = r'''#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

log = Path(os.environ["FAKE_CONDA_LOG"])
# Removals persist in a sidecar file rather than in this process, so a later
# `env list` -- in the same run or in a second one -- really does see the
# environment gone. That is what makes an uninstall/reinstall cycle testable.
removed_record = log.with_suffix(".removed")
removed = removed_record.read_text(encoding="utf-8").split() if removed_record.exists() else []
existing = [
    name
    for name in os.environ.get("FAKE_CONDA_ENVS", "").split(",")
    if name and name not in removed
]
arguments = sys.argv[1:]
with log.open("a", encoding="utf-8") as handle:
    handle.write(" ".join(arguments) + "\n")

if arguments[:1] == ["--version"]:
    print("conda 99.9.9")
    raise SystemExit(0)

if arguments[:2] == ["env", "list"]:
    print("# conda environments:")
    print("base                  *  /opt/fake")
    for name in existing:
        print(f"{name}                     /opt/fake/envs/{name}")
    raise SystemExit(0)

if arguments[:2] in (["env", "create"], ["env", "update"]):
    raise SystemExit(0)

if arguments[:2] == ["env", "remove"]:
    name = arguments[arguments.index("--name") + 1] if "--name" in arguments else None
    if name not in existing:
        raise SystemExit(f"conda: environment {name} does not exist")
    with removed_record.open("a", encoding="utf-8") as handle:
        handle.write(name + "\n")
    raise SystemExit(0)

if arguments[:1] == ["run"]:
    rest = arguments[1:]
    while rest and rest[0].startswith("-"):
        if rest[0] in ("--name", "-n"):
            rest = rest[2:]
        else:
            rest = rest[1:]
    command = rest[1:] if rest[:1] == ["python"] else rest
    joined = " ".join(command)
    # Simulate the mutating steps, and any probe whose result depends on what
    # happens to be installed in the ambient interpreter. What remains under
    # test here is the shell control flow, not the probes themselves -- those
    # are unit-tested directly against check_vaft_environment.py.
    if (
        "pip" in command
        or "ipykernel" in joined
        or "kernelspec" in joined
        or "import " in joined
        or "check_vaft_environment" in joined
    ):
        print(f"[stub] {joined}")
        raise SystemExit(0)
    raise SystemExit(subprocess.run([sys.executable, *command]).returncode)

raise SystemExit(f"unexpected conda invocation: {arguments}")
'''


@pytest.fixture
def fake_conda(tmp_path, monkeypatch):
    """Put a recording `conda` stub in front of the real one on PATH."""
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "conda"
    stub.write_text(FAKE_CONDA, encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "conda.log"
    log.touch()
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CONDA_LOG", str(log))
    return log


def _run_script(script: str, *arguments: str, root: Path = ROOT):
    completed = subprocess.run(
        [BASH, str(root / "install" / script), *arguments],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=300,
    )
    return completed


def _run_bootstrap(script: str = "linux.sh", *arguments: str):
    return _run_script(script, *arguments)


def _removal_commands(log: Path) -> list[str]:
    """Every logged conda invocation that would destroy something."""
    return [
        line
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("env remove")
    ]


@pytest.fixture
def sandboxed_home(tmp_path, monkeypatch):
    """Point every user-level Jupyter path at tmp_path.

    The uninstaller sweeps the user kernelspec directories as a fallback. A test
    must never be able to reach the real one -- least of all while a developer
    has an actual `vaft` kernel registered.
    """
    home = tmp_path / "home"
    kernels = home / ".local" / "share" / "jupyter" / "kernels"
    kernels.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(home / ".local" / "share" / "jupyter"))
    return kernels


@requires_bash
def test_bootstrap_creates_a_missing_environment(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    invocations = fake_conda.read_text(encoding="utf-8")
    assert "env create --name vaft" in invocations
    assert "env update" not in invocations
    assert "[PASS] vaft environment" in completed.stdout


@requires_bash
def test_bootstrap_reuses_an_existing_environment(fake_conda, monkeypatch):
    """Rerunning must update in place, never recreate a working environment."""
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    invocations = fake_conda.read_text(encoding="utf-8")
    assert "env update --name vaft" in invocations
    assert "env create" not in invocations


@requires_bash
def test_bootstrap_never_touches_another_environment(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft,someone_elses_project")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    for line in fake_conda.read_text(encoding="utf-8").splitlines():
        if "--name" in line:
            assert "--name vaft" in line, f"conda was pointed at another environment: {line}"
    assert "someone_elses_project" not in fake_conda.read_text(encoding="utf-8")


@requires_bash
def test_bootstrap_performs_the_documented_steps_in_order(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    for fragment in (
        "[PASS] Conda",
        "[PASS] vaft environment",
        "[PASS] Python",
        "[PASS] editable VAFT installation",
        "[PASS] Python (vaft) kernel",
    ):
        assert fragment in stdout, f"missing step: {fragment}"
    positions = [
        stdout.index("[PASS] vaft environment"),
        stdout.index("[PASS] editable VAFT installation"),
        stdout.index("[PASS] Python (vaft) kernel"),
        stdout.index("Verifying the environment"),
    ]
    assert positions == sorted(positions), "bootstrap steps ran out of order"
    assert "pip install -e ." in stdout  # via the stub echo
    assert "check_vaft_environment" in stdout, "the bootstrap must end by verifying"
    assert "hsconfigure" in stdout
    assert "never asks for, stores, or transmits your credentials" in stdout


def test_bootstrap_delegates_verification_to_the_checker():
    """One implementation of each probe, not one per platform script."""
    # The POSIX wrappers delegate to _common.sh, which is where their shared
    # flow -- including the closing verification -- lives.
    for name in ("_common.sh", "windows_native.ps1"):
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "check_vaft_environment.py" in text, (
            f"install/{name} must finish by running the checker"
        )
    for name in PLATFORM_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "vaft.__file__" not in text, (
            f"install/{name} reimplements the checker's import-location probe"
        )
        assert "kernelspec list" not in text, (
            f"install/{name} reimplements the checker's kernel probe"
        )


def test_no_multiline_python_payload_crosses_conda_run():
    """`conda run` on Windows rejects any argument containing a newline.

    It fails with `NotImplementedError: Support for scripts where arguments
    contain newlines not implemented`, so a multi-line `python -c` payload
    silently turns into a false FAIL. Pass a file path instead.
    """
    for name in (*PLATFORM_SCRIPTS, *EXTERNAL_CODE_SCRIPTS):
        text = _executable_source(INSTALL / name)
        for match in re.finditer(r"-c'?,?\s*(['\"])", text):
            quote = match.group(1)
            end = text.find(quote, match.end())
            assert end != -1, f"install/{name}: unterminated -c payload"
            payload = text[match.end():end]
            assert "\n" not in payload, (
                f"install/{name}: a multi-line `python -c` payload cannot cross "
                f"`conda run` on Windows:\n{payload[:200]}"
            )


@requires_bash
def test_bootstrap_leaves_the_checkout_clean(fake_conda, monkeypatch):
    """A bootstrap run must not dirty the repository it was launched from."""
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True, check=True
    ).stdout
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    assert _run_bootstrap().returncode == 0
    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True, check=True
    ).stdout
    assert before == after


@requires_bash
def test_check_only_mode_changes_nothing(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    completed = _run_bootstrap("linux.sh", "--check-only")
    invocations = fake_conda.read_text(encoding="utf-8")
    assert "env create" not in invocations
    assert "env update" not in invocations
    assert "pip install" not in invocations
    assert "check_vaft_environment.py" in completed.stdout


@requires_bash
def test_bootstrap_reports_missing_conda_with_guidance(tmp_path, monkeypatch):
    """Without Conda the script must explain the fix, not traceback."""
    bash = BASH
    assert bash

    # Do not simply prepend an empty directory to PATH: hosted CI runners expose
    # `conda` from the standard system directories, so the script would find one,
    # really build an environment, and hang on an interactive prompt. Instead give
    # it a sandbox containing only the handful of external tools it needs before
    # the Conda check -- and, deliberately, no conda.
    sandbox = tmp_path / "bin"
    sandbox.mkdir()
    if os.name == "nt":
        # The coreutils shipped beside a Windows bash are dynamically linked
        # against msys DLLs in their own directory, so they cannot be copied or
        # linked into an isolated sandbox -- and a link named `dirname` rather
        # than `dirname.exe` is not executable there at all, which silently
        # empties `$(dirname ...)` instead of failing loudly. Use those
        # directories where they live; none of them carries conda, which is the
        # only thing this sandbox has to exclude. The assertion below proves
        # that rather than assuming it.
        interpreter = Path(bash).parent
        search = [
            interpreter,
            interpreter.parent / "usr" / "bin",
            interpreter.parent.parent / "usr" / "bin",
        ]
        path = os.pathsep.join(str(item) for item in search if item.is_dir())
    else:
        for tool in ("dirname", "basename", "uname", "awk", "grep", "cat", "sed", "env"):
            located = shutil.which(tool)
            if located:
                (sandbox / tool).symlink_to(located)
        path = str(sandbox)

    environment = {"PATH": path, "HOME": os.environ.get("HOME", str(tmp_path))}
    reachable = subprocess.run(
        [bash, "-c", "command -v conda"], env=environment, capture_output=True, timeout=60
    )
    assert reachable.returncode != 0, "the sandbox must not expose conda"

    completed = subprocess.run(
        [bash, str(INSTALL / "linux.sh")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=environment,
        timeout=120,
    )
    assert completed.returncode == 1
    assert "Install Miniconda first" in completed.stderr
    assert "does not install Conda for you" in completed.stderr
    assert "env create" not in completed.stdout, "nothing may be built without Conda"


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
#
# The uninstaller exists so the *installer* can be tested. On a machine that
# already has VAFT, a rerun of the bootstrap only ever takes the update branch;
# the create-from-nothing path is reachable again only after a removal. These
# tests therefore care about two things above all: that removal is scoped to
# exactly what the bootstrap created, and that it is safe to repeat.


@pytest.fixture
def fake_checkout(tmp_path):
    """A throwaway repository root, so artifact deletion never touches ours.

    ``VAFT_REPOSITORY_ROOT`` is derived from the script's own location, so
    copying the two scripts elsewhere is what relocates it.
    """
    root = tmp_path / "checkout"
    (root / "install").mkdir(parents=True)
    for name in ("_common.sh", "uninstall.sh"):
        shutil.copy2(INSTALL / name, root / "install" / name)
    return root


@requires_bash
def test_uninstall_dry_run_removes_nothing(fake_conda, sandboxed_home, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    (sandboxed_home / "vaft").mkdir()

    completed = _run_script("uninstall.sh", "--dry-run")

    assert completed.returncode == 0, completed.stderr
    assert "Dry run: nothing was removed." in completed.stdout
    assert _removal_commands(fake_conda) == []
    assert (sandboxed_home / "vaft").is_dir(), "a dry run must not touch the kernelspec"


@requires_bash
def test_uninstall_refuses_to_guess_when_there_is_no_terminal(
    fake_conda, sandboxed_home, monkeypatch
):
    """No TTY and no --yes means stop, not remove, and not hang on a read."""
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    (sandboxed_home / "vaft").mkdir()

    completed = _run_script("uninstall.sh")

    assert completed.returncode == 1
    assert "without confirmation" in completed.stderr
    assert "--yes" in completed.stderr
    assert _removal_commands(fake_conda) == []
    assert (sandboxed_home / "vaft").is_dir()


@requires_bash
def test_uninstall_removes_the_kernel_before_the_environment(
    fake_conda, sandboxed_home, monkeypatch
):
    """`jupyter kernelspec remove` runs through the environment it is deleting.

    Reverse the order and the command has no interpreter left to run in, which
    would leave the kernelspec pointing at an environment that no longer exists.
    """
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    (sandboxed_home / "vaft").mkdir()

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert completed.returncode == 0, completed.stderr
    log = fake_conda.read_text(encoding="utf-8")
    assert "kernelspec remove" in log
    assert "env remove --name vaft" in log
    assert log.index("kernelspec remove") < log.index("env remove")
    assert not (sandboxed_home / "vaft").exists()


@requires_bash
def test_uninstall_never_names_another_environment(fake_conda, sandboxed_home, monkeypatch):
    """`vaft-np2-test` is somebody's work, and its name merely starts with vaft."""
    monkeypatch.setenv(
        "FAKE_CONDA_ENVS", "vaft,vaft-np2-test,vaftlike,someone_elses_project"
    )

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert completed.returncode == 0, completed.stderr
    assert _removal_commands(fake_conda) == ["env remove --name vaft --yes"]


@requires_bash
def test_uninstall_with_nothing_installed_succeeds_quietly(
    fake_conda, sandboxed_home, monkeypatch
):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "")

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert completed.returncode == 0, completed.stderr
    assert "[SKIP] vaft environment" in completed.stdout
    assert "[SKIP] Python (vaft) kernel" in completed.stdout
    assert _removal_commands(fake_conda) == []


@requires_bash
def test_a_second_uninstall_is_a_no_op(fake_conda, sandboxed_home, monkeypatch):
    """Idempotency in the removal direction: the cycle has to survive repeats."""
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    (sandboxed_home / "vaft").mkdir()

    first = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")
    second = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "[PASS] vaft environment" in first.stdout
    assert "[SKIP] vaft environment" in second.stdout
    assert _removal_commands(fake_conda) == ["env remove --name vaft --yes"]


@requires_bash
def test_uninstall_clears_build_artifacts_from_the_checkout(
    fake_conda, sandboxed_home, fake_checkout, monkeypatch
):
    """An editable install leaves these behind, and the next one inherits them."""
    monkeypatch.setenv("FAKE_CONDA_ENVS", "")
    (fake_checkout / "vaft.egg-info").mkdir()
    # `python -m build` writes these (RELEASING.md); the bootstrap never does,
    # so they are somebody's release artifacts, not an install leftover.
    for release_output in ("build", "dist"):
        (fake_checkout / release_output).mkdir()
    keeper = fake_checkout / "vaft"
    keeper.mkdir()

    completed = _run_script("uninstall.sh", "--yes", root=fake_checkout)

    assert completed.returncode == 0, completed.stderr
    assert not (fake_checkout / "vaft.egg-info").exists(), "egg-info survived"
    for release_output in ("build", "dist"):
        assert (fake_checkout / release_output).is_dir(), (
            f"{release_output}/ is a release artifact and must not be removed"
        )
    assert keeper.is_dir(), "only the install leftover may be removed, not source"


@requires_bash
def test_keep_build_artifacts_leaves_them_alone(
    fake_conda, sandboxed_home, fake_checkout, monkeypatch
):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "")
    (fake_checkout / "vaft.egg-info").mkdir()

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts", root=fake_checkout)

    assert completed.returncode == 0, completed.stderr
    assert (fake_checkout / "vaft.egg-info").is_dir()


@requires_bash
def test_uninstall_never_removes_the_hsds_configuration(
    fake_conda, sandboxed_home, monkeypatch, tmp_path
):
    """`~/.hscfg` holds credentials and the bootstrap never wrote it."""
    configuration = tmp_path / "home" / ".hscfg"
    configuration.write_text("hs_endpoint = http://example.invalid\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert completed.returncode == 0, completed.stderr
    assert configuration.is_file(), "the uninstaller deleted HSDS credentials"
    assert ".hscfg" in completed.stdout, "say that credentials were preserved"


@requires_bash
def test_uninstall_leaves_the_checkout_clean(fake_conda, sandboxed_home, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    ).stdout

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    ).stdout
    assert completed.returncode == 0, completed.stderr
    assert before == after


@requires_bash
def test_uninstall_rejects_an_unknown_option(fake_conda, sandboxed_home, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")

    completed = _run_script("uninstall.sh", "--purge-everything")

    assert completed.returncode == 1
    assert "Unknown option" in completed.stderr
    assert _removal_commands(fake_conda) == []


@requires_bash
def test_uninstall_help_documents_every_flag(fake_conda, sandboxed_home, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")

    completed = _run_script("uninstall.sh", "--help")

    assert completed.returncode == 0, completed.stderr
    for flag in ("--yes", "--dry-run", "--keep-build-artifacts"):
        assert flag in completed.stdout
    assert _removal_commands(fake_conda) == []


@requires_bash
def test_uninstall_stops_when_the_environment_is_active(
    fake_conda, sandboxed_home, monkeypatch
):
    """Conda will not delete the environment you are standing in.

    Since the kernelspec has to be removed first -- it needs that environment's
    interpreter -- a refusal part-way would strand a working environment with no
    kernel. Refusing up front leaves the machine exactly as it was.
    """
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "vaft")
    (sandboxed_home / "vaft").mkdir()

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert completed.returncode == 1
    assert "conda deactivate" in completed.stderr
    assert _removal_commands(fake_conda) == []
    assert (sandboxed_home / "vaft").is_dir(), "the kernelspec must survive"


@requires_bash
def test_another_active_environment_does_not_block_the_uninstall(
    fake_conda, sandboxed_home, monkeypatch
):
    """The guard is about `vaft` specifically, not about being in any env."""
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

    completed = _run_script("uninstall.sh", "--yes", "--keep-build-artifacts")

    assert completed.returncode == 0, completed.stderr
    assert _removal_commands(fake_conda) == ["env remove --name vaft --yes"]


def test_uninstall_leaves_release_artifacts_alone():
    """`build/` and `dist/` come from `python -m build`, not from the bootstrap."""
    text = (INSTALL / "_common.sh").read_text(encoding="utf-8")
    assert 'vaft_build_artifacts=("vaft.egg-info")' in text, (
        "only the editable install's own leftover is the uninstaller's business"
    )
    powershell = (INSTALL / "uninstall_windows_native.ps1").read_text(encoding="utf-8")
    assert "$BuildArtifacts = @('vaft.egg-info')" in powershell


def test_readme_documents_uninstalling():
    """The removal contract is pinned here, the way every other one is."""
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    assert "## Uninstalling" in text
    for name in UNINSTALL_SCRIPTS:
        assert name in text, f"install/README.md does not mention {name}"
    for fragment in ("--dry-run", "--keep-build-artifacts", "conda env remove --name vaft"):
        assert fragment in text
    # The two hazards a reader has to be told about.
    assert "conda deactivate" in text
    assert "release artifacts" in text
    # The two promises that make the script safe to hand to a student.
    assert "~/.hscfg" in text
    assert "never in scope" in text


def test_uninstall_reverses_exactly_what_the_bootstrap_creates():
    """The two directions must not drift apart in what they name."""
    text = (INSTALL / "_common.sh").read_text(encoding="utf-8")
    for created, removed in (
        ("conda env create", "conda env remove"),
        ("ipykernel install", "kernelspec remove"),
    ):
        assert created in text and removed in text, f"{created} has no counterpart"


# ---------------------------------------------------------------------------
# External Fortran codes (issue #226)
# ---------------------------------------------------------------------------


EXTERNAL_SOURCES = (
    *EXTERNAL_CODE_SCRIPTS,
    *EXTERNAL_CODE_CHECKERS,
    "_external_code_common.ps1",
    "_external_code_common.py",
)


def _load_external_checker(name):
    module_name = name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, INSTALL / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_external_code_tooling_never_obtains_or_moves_a_checkout():
    """Which revision was built is a statement the operator makes.

    Issue #226 keeps source acquisition separate from building so that
    provenance, access and revision selection stay decisions a person takes.
    A script that fetches or switches revisions on their behalf destroys that.
    """
    for name in EXTERNAL_SOURCES:
        text = _executable_source(INSTALL / name)
        for pattern in ACQUISITIVE:
            assert not pattern.search(text), f"install/{name} runs `{pattern.pattern}`"


def test_external_code_installers_never_guess_where_the_source_is():
    for name in EXTERNAL_CODE_SCRIPTS:
        text = _executable_source(INSTALL / name)
        for guess in ("~/git", "$HOME/git", "USERPROFILE\\git"):
            assert guess not in text, f"install/{name} guesses a source path: {guess}"


def test_external_code_installers_require_an_explicit_source_path():
    for name in EXTERNAL_CODE_WINDOWS_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "[Parameter(Position = 0)] [string] $SourcePath" in text, (
            f"install/{name} must take the source path as its first argument"
        )
        assert "Assert-SourceCheckout" in text, (
            f"install/{name} must validate the path it was given"
        )


def test_posix_external_installers_require_an_explicit_source_path():
    """The POSIX twin of the rule above, in this platform's spelling.

    A named `--source` rather than a positional argument, and a marker check
    against the tree it names, so a mistyped path fails on the path instead of
    part-way through a twenty-minute build.
    """
    for name in EXTERNAL_CODE_POSIX_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "--source" in text, f"install/{name} must take --source"
        assert "--source is required" in text, (
            f"install/{name} must refuse to guess when --source is missing"
        )
        assert "not a" in text and "source tree (missing" in text, (
            f"install/{name} must validate the path it was given against markers"
        )


def test_toolchain_installation_is_opt_in():
    """Installing a compiler system-wide is the operator's decision.

    The bootstrap already treats Git, Conda and WSL2 as things it will not
    install for you. A Fortran toolchain is no different.
    """
    for name in EXTERNAL_CODE_WINDOWS_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "[switch] $InstallToolchain" in text

    shared = _executable_source(INSTALL / "_external_code_common.ps1")
    for command in ("winget install", "pacman -S"):
        assert command in shared, f"{command} belongs in the shared helper"
    # Reachable only through the opt-in function.
    body = shared[shared.index("function Install-Msys2Toolchain"):]
    assert "winget install" in body
    assert "pacman -S" in body
    # And a machine without it is told both ways to fix that.
    assert "Get-ToolchainGuidance" in shared
    assert "-InstallToolchain" in shared


def test_external_code_installs_outside_every_checkout():
    """Nothing may land in the VAFT checkout.

    The bootstrap CI cycle ends by requiring `git status` to come back empty,
    so an artifact inside the repository would fail a job that has nothing to
    do with these scripts.
    """
    shared = (INSTALL / "_external_code_common.ps1").read_text(encoding="utf-8")
    assert "LOCALAPPDATA" in shared
    assert "The install prefix must be outside" in shared
    for name in EXTERNAL_CODE_WINDOWS_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "Resolve-InstallPrefix" in text, f"install/{name} must validate its prefix"


def test_posix_external_installers_never_install_a_toolchain():
    """A package install on Linux needs root, which no installer here takes.

    These scripts name the package in the failure instead -- "apt install
    gfortran" as advice inside a `die`, never as something they run. So the
    assertion is about position, not about the words: a package manager may
    appear inside a message, but never as the command a line executes.
    `brew list` and `brew --prefix` are queries and stay allowed.
    """
    invocation = re.compile(r"^\s*(sudo|apt|apt-get|pacman|dnf|yum)\b|^\s*brew\s+install\b")
    for name in EXTERNAL_CODE_POSIX_SCRIPTS:
        for number, line in enumerate(_executable_source(INSTALL / name).splitlines(), 1):
            assert not invocation.match(line), (
                f"install/{name}:{number} runs a package manager rather than "
                f"naming the package: {line.strip()}"
            )


def test_the_vaft_bootstrap_never_builds_fortran():
    """Bootstrap CI runs windows_native.ps1 three times on a hosted runner.

    A Fortran build wired into that path would add tens of minutes and need a
    toolchain the runner does not have, so the external codes stay separate
    entry points that CI never invokes.
    """
    for name in ("windows_native.ps1", "_common.sh", "uninstall_windows_native.ps1"):
        text = (INSTALL / name).read_text(encoding="utf-8")
        for external in EXTERNAL_CODE_SCRIPTS:
            assert external not in text, f"install/{name} references {external}"


def test_gpec_build_never_requires_x11():
    """Issue #226 excludes xdraw: the Windows target is the CLI workflow."""
    text = _executable_source(INSTALL / "install_gpec_windows.ps1")
    assert "mkbin" in text, "build the executables target rather than everything"
    assert "xdraw" not in text
    assert "make all" not in text


def test_gpec_build_sets_every_variable_the_upstream_makefile_reads():
    """Each of these has its own opaque failure when it is left out.

    An unset CC means GNU make's built-in `cc` reaches a compiler test that
    rejects it. The argument-mismatch flag is mandatory on modern gfortran. The
    library homes are what stop the makefile building its own dependencies from
    submodules inside the operator's tree. And a stray MKLROOT from an
    unrelated toolkit silently changes which math library gets linked.
    """
    text = _executable_source(INSTALL / "install_gpec_windows.ps1")
    for required in (
        "FC=gfortran",
        "CC=gcc",
        "OPENBLASHOME",
        "NETCDF_FORTRAN_HOME",
        "-fallow-argument-mismatch",
        "RECURSFLAG=-frecursive",
    ):
        assert required in text, f"install_gpec_windows.ps1 does not set {required}"
    assert "unset MKLROOT" in text
    for leaked in ("LAPACKHOME", "NETCDFHOME"):
        assert leaked in text, f"{leaked} should be cleared before the build"


def test_gpec_build_refuses_to_let_make_fetch_its_own_dependencies():
    """The submodule path would modify the operator's checkout."""
    text = _executable_source(INSTALL / "install_gpec_windows.ps1")
    assert "make v" in text
    assert "Compiling supporting modules" in text


def test_gpec_build_is_serial_and_records_why():
    """OpenMP cannot be used, and a reader must not have to rediscover that.

    gfortran expresses `!$OMP THREADPRIVATE` on a COMMON block with an
    assembler directive the PE object format has no equivalent for, so an
    OpenMP build of LSODE and ZVODE cannot assemble at all.
    """
    text = (INSTALL / "install_gpec_windows.ps1").read_text(encoding="utf-8")
    assert "OMPFLAG=" in text
    assert "threadprivate" in text.lower()


def test_chease_build_pins_the_machine_and_the_compiler():
    """CHEASE_MACHINE is not cosmetic.

    Only that branch of the upstream flags file sets the double-precision
    options, so the default machine builds a numerically different code that
    still compiles cleanly.
    """
    text = (INSTALL / "install_chease_windows.ps1").read_text(encoding="utf-8")
    assert "CHEASE_F90=gfortran" in text
    assert "CHEASE_MACHINE=linux_nohdf5" in text
    assert "precision" in text


def test_chease_linux_build_pins_the_machine_and_the_compiler():
    """The same rule, on the platform whose name that machine branch carries.

    `src-f90/Makefile.define_FLAGS` matches linux_nohdf5 in one branch and that
    branch is the only one setting -fdefault-real-8 -fdefault-double-8. The
    default from Makefile.define_MACHINE is `none`, which compiles in single
    precision without complaining.
    """
    text = (INSTALL / "install_chease.sh").read_text(encoding="utf-8")
    assert "CHEASE_F90=gfortran" in text
    assert "linux_nohdf5" in text
    assert "precision" in text
    # The goal must be exactly `chease`; `all` pulls in libxml2.
    assert "make -j\"$JOBS\" chease" in text
    assert "make all" not in _executable_source(INSTALL / "install_chease.sh")


def test_gpec_linux_build_sets_every_variable_the_upstream_makefile_reads():
    """DEFAULTS.inc resolves the toolchain from the environment, not from flags.

    Anything left unset is inferred, and on a machine with more than one
    toolchain installed the inference is what goes wrong.
    """
    text = _executable_source(INSTALL / "install_gpec.sh")
    for variable in (
        "FC=gfortran",
        "CC=gcc",
        "LAPACKHOME=",
        "NETCDF_FORTRAN_HOME=",
        "NETCDFINC=",
        "-fallow-argument-mismatch",
        "RECURSFLAG=-frecursive",
    ):
        assert variable in text, f"install/install_gpec.sh must set {variable}"
    assert "unset MKLROOT" in text, "a stray MKLROOT silently changes the math library"


def test_gpec_linux_build_uses_openmp_and_says_why_windows_cannot():
    """The deliberate inverse of test_gpec_build_is_serial_and_records_why.

    The Windows build disables OpenMP because LSODE and ZVODE mark a COMMON
    block threadprivate, which gfortran cannot express in PE object format. ELF
    has no such limit. This test exists so that nobody "harmonises" the two
    scripts and makes the Linux build serial for a reason that is not about it.
    """
    text = (INSTALL / "install_gpec.sh").read_text(encoding="utf-8")
    assert "OMPFLAG=-fopenmp" in text
    # Matched loosely: LDFLAGS also carries the -rpath that pins netCDF, so the
    # assertion is that OpenMP reaches the link, not that it is the only flag.
    ldflags = [l for l in text.splitlines() if "LDFLAGS=" in l]
    assert ldflags and any("-fopenmp" in l for l in ldflags), (
        "OpenMP must reach the link step, not only the compile step"
    )
    assert "threadprivate" in text, "the header must explain why Windows differs"


def test_gpec_linux_build_never_requires_x11():
    """xdraw is a viewer no VAFT workflow uses; building it would need libX11."""
    text = _executable_source(INSTALL / "install_gpec.sh")
    assert "xdraw" not in text
    assert "make all" not in text


def test_gpec_linux_build_refuses_to_let_make_fetch_its_own_dependencies():
    """A non-empty NEEDED_DEPS means upstream is about to build inside the checkout.

    `make v` reports it without building anything, so the refusal costs nothing
    and lands before the twenty minutes rather than after.
    """
    text = _executable_source(INSTALL / "install_gpec.sh")
    assert "make v" in text
    assert "NEEDED_DEPS" in text


def test_gpec_linux_verifies_its_binaries_are_not_empty():
    """Upstream's rules judge themselves by a `cp`.

    A failed link can leave a zero-length file that make then treats as up to
    date, so make's exit status is not the evidence -- the file size is.
    """
    text = _executable_source(INSTALL / "install_gpec.sh")
    assert 'rm -f "$SOURCE/bin/$program"' in text, "clear the targets before building"
    assert '[[ -s "$SOURCE/bin/$program" ]]' in text, "assert on size, not on make"


def test_posix_external_installers_resolve_netcdf_by_compiler_not_by_path():
    """`nf-config` first on PATH is not necessarily the right netCDF-Fortran.

    A netCDF-Fortran built with ifort ships ifort .mod files; linking them into
    a gfortran build fails with errors that never mention a compiler. nf-config
    reports its own --fc, so the right library is identifiable rather than
    guessable -- and on the VAFT reference server the wrong one is first.
    """
    text = _executable_source(INSTALL / "install_gpec.sh")
    assert "--fc" in text, "pick the netCDF-Fortran by the compiler it was built with"
    assert "gfortran*" in text or "gfortran" in text


def test_chease_installer_explains_the_symbolic_link_placeholders():
    """That failure happens at acquisition time, and no compiler flag fixes it."""
    text = (INSTALL / "install_chease_windows.ps1").read_text(encoding="utf-8")
    assert "[switch] $MaterializeSymlinks" in text
    assert "symbolic link" in text.lower()


def test_external_code_checkers_share_the_vaft_vocabulary():
    """One vocabulary, imported -- not a second one that can drift from it."""
    base = _load_checker()
    for name in (*EXTERNAL_CODE_CHECKERS, "_external_code_common.py"):
        source = (INSTALL / name).read_text(encoding="utf-8")
        assert "class CheckResult" not in source, f"install/{name} redefines CheckResult"
        for status in ("PASS", "FAIL", "SKIP", "WARN"):
            assert f'{status} = "' not in source, f"install/{name} redefines {status}"
    for name in EXTERNAL_CODE_CHECKERS:
        module = _load_external_checker(name)
        assert module.CheckResult._fields == base.CheckResult._fields
        for status in ("PASS", "FAIL", "SKIP", "WARN"):
            assert getattr(module, status) == getattr(base, status)


def test_external_code_checkers_report_every_layer():
    """Issue #226 asks each layer to answer for itself, not just the binary."""
    expected = {
        "check_chease.py": ("toolchain", "source", "build record", "executables", "discovery", "run"),
        # EFIT installs two roles from one build, so both are checked; the
        # smoke layer runs EFUND rather than the reconstruction code, because
        # a table is what a reconstruction needs before it can run at all.
        "check_efit.py": (
            "toolchain", "source", "build record", "executables", "discovery",
            "capabilities", "starts", "smoke",
        ),
        # GACODE has no build-record layer: it builds in place and leaves no
        # manifest to read. It has three the others do not -- the platform tag
        # that selects the run-time exec script, the input parser the launcher
        # shells out to, and a deliberate WARN saying which NEO results reach an
        # IDS and which stay native.
        "check_gacode.py": (
            "toolchain", "source", "executables", "platform",
            "input parser", "discovery", "regression", "imas mapping",
        ),
        "check_gpec.py": ("toolchain", "source", "build record", "executables", "discovery", "handoff"),
        # NUBEAM has no smoke run without a case, and two layers the others do
        # not: the reaction databases it aborts without, and the fixed-width
        # filename buffer a deep working directory overruns.
        "check_nubeam.py": (
            "toolchain", "source", "build record", "executables", "discovery",
            "reaction databases", "path budget",
        ),
    }
    for name, layers in expected.items():
        module = _load_external_checker(name)
        results = module.run_checks(source=None, prefix=None, skip_smoke=True)
        reported = " ".join(result.name for result in results).lower()
        for layer in layers:
            assert layer in reported, f"{name} does not report a {layer} layer"


def test_external_code_checkers_name_the_layer_not_the_linker(tmp_path):
    """A page of compiler output tells the reader nothing they can act on."""
    for name in EXTERNAL_CODE_CHECKERS:
        module = _load_external_checker(name)
        results = module.run_checks(source=None, prefix=str(tmp_path), skip_smoke=True)
        for result in results:
            if result.status == module.FAIL:
                assert result.remediation, f"{name}: {result.name} fails with no remediation"
            for noise in ("Traceback", "undefined reference", "collect2"):
                assert noise not in result.detail, f"{name}: {result.name} leaks {noise}"


def test_external_code_checkers_have_a_usable_command_line():
    for name in EXTERNAL_CODE_CHECKERS:
        completed = subprocess.run(
            [sys.executable, str(INSTALL / name), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--source" in completed.stdout
        assert "--prefix" in completed.stdout


def test_readme_documents_the_external_code_path():
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    for fragment in (
        "install_chease_windows.ps1",
        "install_gpec_windows.ps1",
        "install_chease.sh",
        "install_gpec.sh",
        "check_chease.py",
        "check_gpec.py",
        "-InstallToolchain",
        "MSYS2",
        # The POSIX path needs its packages named, because the scripts refuse to
        # install them and a reader who hits that failure needs the line here.
        "libnetcdff-dev",
    ):
        assert fragment in text, f"install/README.md does not mention {fragment}"


def test_readme_routes_every_platform_for_every_external_code():
    """The decision tree used to send Linux and macOS readers to PowerShell.

    Each external code needs a POSIX branch and a Windows branch, in the tree
    itself -- naming the POSIX script five hundred lines further down is not
    routing.
    """
    text = (INSTALL / "README.md").read_text(encoding="utf-8")
    tree = text[text.index("Need CHEASE?"):]
    tree = tree[: tree.index("```")]
    for script in ("install_chease.sh", "install_gpec.sh", "install_efit.sh"):
        assert script in tree, f"the decision tree does not route to {script}"
    assert tree.count("Linux/macOS") >= 3


# ---------------------------------------------------------------------------
# install/nubeam: the Windows NUBEAM recipe
#
# NUBEAM's entry point lives in install/nubeam/ rather than install/, beside
# the macOS recipe it mirrors, because the two share the reference cases and
# the validation scripts. The rules issue #226 sets for install/ apply to it
# all the same, so they are asserted here rather than assumed.
# ---------------------------------------------------------------------------

NUBEAM_DIR = ROOT / "install" / "nubeam"
NUBEAM_SCRIPTS = ("windows.ps1", "windows.sh")
#: Every POSIX recipe and the remover they share.
NUBEAM_POSIX_SCRIPTS = ("linux.sh", "macos.sh", "uninstall.sh")


def test_nubeam_windows_recipe_is_present_beside_the_macos_one():
    for name in (*NUBEAM_SCRIPTS, "macos.sh"):
        assert (NUBEAM_DIR / name).is_file(), f"install/nubeam/{name} is missing"


def test_nubeam_windows_recipe_never_obtains_or_moves_the_source():
    """The NUBEAM tree is the operator's, and its revision is their statement.

    The recipe does download the three NTCC dependency modules, which is a
    different act: those are versionless tarballs from PPPL, gated on an
    explicit acceptance flag, and they land in a vendor directory rather than
    over anything the operator holds.
    """
    for name in NUBEAM_SCRIPTS:
        text = _executable_source(NUBEAM_DIR / name)
        for pattern in ACQUISITIVE:
            assert not pattern.search(text), f"install/nubeam/{name} runs `{pattern.pattern}`"


def test_nubeam_windows_recipe_never_guesses_where_the_source_is():
    for name in NUBEAM_SCRIPTS:
        text = _executable_source(NUBEAM_DIR / name)
        for guess in ("~/git", "$HOME/git", "USERPROFILE\\git"):
            assert guess not in text, f"install/nubeam/{name} guesses a source path: {guess}"


def test_nubeam_windows_wrapper_requires_an_explicit_source_path():
    text = (NUBEAM_DIR / "windows.ps1").read_text(encoding="utf-8")
    assert "[Parameter(Position = 0)] [string] $SourcePath" in text
    assert "Assert-SourceCheckout" in text
    assert "[switch] $InstallToolchain" in text


def test_nubeam_downloads_are_gated_on_explicit_acceptance():
    """NTCC requires each user to accept its licence before downloading.

    Both halves have to enforce it: the wrapper so the refusal is legible
    before an hour of compilation, and the recipe so running it directly is
    not a way around the wrapper.
    """
    wrapper = (NUBEAM_DIR / "windows.ps1").read_text(encoding="utf-8")
    assert "[switch] $AcceptNtccTerms" in wrapper
    assert "downloads.shtml" in wrapper

    recipe = (NUBEAM_DIR / "windows.sh").read_text(encoding="utf-8")
    assert "--accept-ntcc-terms" in recipe
    assert "ACCEPT_NTCC_TERMS" in recipe
    # The flag has to be checked before the first download, not after it.
    gate = recipe.index("((ACCEPT_NTCC_TERMS))")
    assert gate < recipe.index("download_ntcc_module()")


def test_nubeam_generates_only_inside_the_source_tree():
    """Everything generated stays where macos.sh puts it.

    The prefix follows from the source path rather than being a separate
    choice, so the two platforms cannot disagree about it, and -Uninstall has
    an exact list to remove.
    """
    recipe = (NUBEAM_DIR / "windows.sh").read_text(encoding="utf-8")
    assert 'PREFIX="$ROOT_DIR/local"' in recipe
    assert "refusing path outside source tree" in recipe

    wrapper = (NUBEAM_DIR / "windows.ps1").read_text(encoding="utf-8")
    assert "Get-NubeamPrefix" in wrapper
    assert "[switch] $Uninstall" in wrapper


def test_the_vaft_bootstrap_never_builds_nubeam():
    for name in ("windows_native.ps1", "_common.sh", "uninstall_windows_native.ps1"):
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "nubeam" not in text.lower(), f"install/{name} references NUBEAM"


def test_nubeam_windows_recipe_is_valid_shell():
    bash = _usable_bash()
    if bash is None:
        pytest.skip("no usable POSIX bash on this machine")
    completed = subprocess.run(
        [bash, "-n", str(NUBEAM_DIR / "windows.sh")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_nubeam_readme_documents_the_windows_path():
    text = (NUBEAM_DIR / "README.md").read_text(encoding="utf-8")
    for fragment in ("windows.ps1", "windows.sh", "-AcceptNtccTerms"):
        assert fragment in text, f"install/nubeam/README.md does not mention {fragment}"


# ---------------------------------------------------------------------------
# install/gacode: the per-platform GACODE recipes
#
# GACODE is the one external code that builds in place, so it has no prefix and
# no manifest to assert against. What it does have is a platform tag that
# selects both a build file and an exec file, and a build order the per-code
# makefiles depend on -- and both fail as something else when they are wrong.
# ---------------------------------------------------------------------------

GACODE_DIR = ROOT / "install" / "gacode"
GACODE_RECIPES = ("linux.sh", "macos.sh")


def test_gacode_has_a_recipe_for_every_supported_platform():
    for name in GACODE_RECIPES:
        assert (GACODE_DIR / name).is_file(), f"install/gacode/{name} is missing"


@pytest.mark.parametrize("name", GACODE_RECIPES)
def test_gacode_recipe_is_valid_shell(name):
    bash = _usable_bash()
    if bash is None:
        pytest.skip("no usable POSIX bash on this machine")
    completed = subprocess.run(
        [bash, "-n", str(GACODE_DIR / name)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("name", GACODE_RECIPES)
def test_gacode_recipe_never_obtains_or_moves_the_source(name):
    """The tree is the operator's, and its revision is their statement."""
    text = _executable_source(GACODE_DIR / name)
    for pattern in (*ACQUISITIVE, *DESTRUCTIVE):
        assert not pattern.search(text), (
            f"install/gacode/{name} runs `{pattern.pattern}`"
        )


@pytest.mark.parametrize("name", GACODE_RECIPES)
def test_gacode_recipe_never_guesses_where_the_source_is(name):
    text = _executable_source(GACODE_DIR / name)
    for guess in ("~/git", "$HOME/git", "USERPROFILE\\git"):
        assert guess not in text, f"install/gacode/{name} guesses a source path: {guess}"


def test_gacode_linux_recipe_requires_both_halves_of_the_platform_tag():
    """$GACODE_PLATFORM selects a build file *and* an exec file.

    A tag with only the build half compiles and then fails at run time inside
    platform/exec/exec.$GACODE_PLATFORM, which never names the variable that is
    wrong. Checking both at build time is what turns that into a sentence.
    """
    text = _executable_source(GACODE_DIR / "linux.sh")
    assert "platform/build/make.inc." in text
    assert "platform/exec/exec." in text
    assert "Available platforms:" in text, "list the tags when the chosen one is absent"


def test_gacode_linux_recipe_asserts_on_the_compiled_binary_not_the_launcher():
    """`<code>/bin/<code>` is a committed shell script, not a build product.

    It exists in a fresh clone, so asserting on it would pass for a build that
    compiled nothing. The ELF lands at `<code>/src/<code>`.
    """
    text = _executable_source(GACODE_DIR / "linux.sh")
    assert '/src/$code' in text, "assert on the compiled binary"
    assert '/bin/$code' in text, "and confirm the launcher VAFT resolves is there"


def test_gacode_linux_recipe_installs_nothing():
    """An apt install needs root; macos.sh may `brew install`, this may not."""
    invocation = re.compile(r"^\s*(sudo|apt|apt-get|pacman|dnf|yum)\b")
    for number, line in enumerate(
        _executable_source(GACODE_DIR / "linux.sh").splitlines(), 1
    ):
        assert not invocation.match(line), (
            f"install/gacode/linux.sh:{number} installs packages: {line.strip()}"
        )


def test_gacode_linux_recipe_builds_the_shared_libraries_first():
    """The per-code makefiles link shared/*/*.a and f2py/*/*.a as EXTRA_LIBS."""
    text = _executable_source(GACODE_DIR / "linux.sh")
    shared = text.index('make -C "$GACODE_ROOT/shared"')
    f2py = text.index('make -C "$GACODE_ROOT/f2py"')
    per_code = text.index('make -C "$GACODE_ROOT/$code"')
    assert shared < f2py < per_code, "build order is load-bearing, not incidental"


def test_gacode_readme_documents_both_platforms():
    text = (GACODE_DIR / "README.md").read_text(encoding="utf-8")
    for fragment in ("linux.sh", "macos.sh", "TUMBLEWEED", "GACODE_PLATFORM"):
        assert fragment in text, f"install/gacode/README.md does not mention {fragment}"


def test_nubeam_uninstall_script_exists():
    """The POSIX recipes name it in three places, so it has to be there.

    install/nubeam/macos.sh points at `./uninstall.sh` in its usage text, in the
    header it writes into every generated Make.local, and in the error raised
    when it is re-run over an existing installation -- which is the moment an
    operator most needs it to exist.
    """
    assert (NUBEAM_DIR / "uninstall.sh").is_file(), (
        "install/nubeam/macos.sh tells the operator to run ./uninstall.sh"
    )


def test_nubeam_uninstall_is_driven_by_the_manifest():
    """What to remove is recorded by the installer, not listed in the remover.

    A hand-kept list in the uninstaller drifts from what the installer actually
    generated; the manifest cannot, because it is written before anything is.
    """
    text = _executable_source(NUBEAM_DIR / "uninstall.sh")
    assert ".nubeam-install-manifest" in text
    assert "managed_dir" in text
    assert "generated_config" in text


def test_nubeam_uninstall_never_reaches_outside_the_source_tree():
    """The installer generates only inside the tree, so nothing else is ours."""
    text = _executable_source(NUBEAM_DIR / "uninstall.sh")
    assert "refusing path outside source tree" in text
    assert "is_child_of_root" in text


def test_nubeam_uninstall_leaves_a_config_the_operator_has_changed():
    """A generated file stops being the installer's the moment it is edited."""
    text = _executable_source(NUBEAM_DIR / "uninstall.sh")
    assert "Generated by (VAFT (install|external)/nubeam" in text, (
        "recognise the marker, including the pre-move spelling"
    )


def test_nubeam_uninstall_never_touches_the_source_itself():
    text = _executable_source(NUBEAM_DIR / "uninstall.sh")
    for pattern in (*ACQUISITIVE, *DESTRUCTIVE):
        assert not pattern.search(text), (
            f"install/nubeam/uninstall.sh runs `{pattern.pattern}`"
        )


def test_nubeam_uninstall_is_valid_shell():
    bash = _usable_bash()
    if bash is None:
        pytest.skip("no usable POSIX bash on this machine")
    completed = subprocess.run(
        [bash, "-n", str(NUBEAM_DIR / "uninstall.sh")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("name", NUBEAM_POSIX_SCRIPTS)
def test_nubeam_posix_recipe_is_valid_shell(name):
    bash = _usable_bash()
    if bash is None:
        pytest.skip("no usable POSIX bash on this machine")
    completed = subprocess.run(
        [bash, "-n", str(NUBEAM_DIR / name)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("name", NUBEAM_POSIX_SCRIPTS)
def test_nubeam_posix_recipe_never_obtains_or_moves_the_source(name):
    text = _executable_source(NUBEAM_DIR / name)
    for pattern in (*ACQUISITIVE, *DESTRUCTIVE):
        assert not pattern.search(text), (
            f"install/nubeam/{name} runs `{pattern.pattern}`"
        )


@pytest.mark.parametrize("name", NUBEAM_POSIX_SCRIPTS)
def test_nubeam_posix_recipe_never_guesses_where_the_source_is(name):
    text = _executable_source(NUBEAM_DIR / name)
    for guess in ("~/git", "$HOME/git", "USERPROFILE\\git"):
        assert guess not in text, f"install/nubeam/{name} guesses a source path: {guess}"


def test_nubeam_linux_downloads_are_gated_on_explicit_acceptance():
    """NTCC requires each user to accept its licence before downloading."""
    text = _executable_source(NUBEAM_DIR / "linux.sh")
    assert "ACCEPT_NTCC_TERMS" in text
    download = text[text.index("download_ntcc_module() {"):]
    download = download[: download.index("\n}")]
    assert "((ACCEPT_NTCC_TERMS))" in download, "the gate must precede the fetch"
    assert download.index("((ACCEPT_NTCC_TERMS))") < download.index("curl"), (
        "the licence gate must come before the first download"
    )


def test_nubeam_linux_generates_only_inside_the_source_tree():
    text = _executable_source(NUBEAM_DIR / "linux.sh")
    assert 'PREFIX="$ROOT_DIR/local"' in text
    assert "refusing path outside source tree" in text


def test_nubeam_linux_delivers_legacy_flags_through_the_compiler():
    """FFLAGS cannot carry them, and neither can an override.

    The 2021 sources need -std=legacy. A plain `FFLAGS =` in Make.local is
    discarded, because Make.local is read before Make.flags' MKGCC block, which
    assigns FFLAGS itself (Make.flags:333) -- macOS never sees this, since MKGCC
    is Linux-gated, which is why macos.sh can assign plainly.

    An `override` is worse than useless: upstream's convention is that FFLAGS
    ends with `-o`, appended per submodule, and `MFFLAGS = $(FFLAGS)`
    (Make.flags:658) is consumed as `$(FC90) $(MFFLAGS) $@ $<`. Freezing FFLAGS
    stops that append, and gfortran is handed an object path with no -o before
    it -- which fails as "linker input file not found", naming neither FFLAGS
    nor the missing flag.

    So the switches ride on FC instead, which Make.local may set because MKGCC
    only fills FC when its origin is "default".
    """
    text = (NUBEAM_DIR / "linux.sh").read_text(encoding="utf-8")
    assert "override FFLAGS" not in text, (
        "an override freezes FFLAGS and suppresses the per-submodule -o append"
    )
    assert "FC_WRAPPER" in text, "the legacy switches travel with the compiler"
    assert "-std=legacy" in text
    assert "-fallow-argument-mismatch" in text
    # The wrapper is only reached if Make.local actually points FC at it.
    generated = text[text.index("write_make_local() {"):]
    generated = generated[: generated.index("\n}")]
    assert "FC = $FC_WRAPPER" in generated
    assert "FC90 = $FC_WRAPPER" in generated


def test_nubeam_linux_does_not_carry_the_macos_only_workarounds():
    """Most of macos.sh's Make.local defeats branches upstream gates on Linux.

    Setting them here would shadow the values Make.flags derives correctly, so
    their absence is deliberate rather than an omission.
    """
    generated = (NUBEAM_DIR / "linux.sh").read_text(encoding="utf-8")
    generated = generated[generated.index("write_make_local() {"):]
    generated = generated[: generated.index("\n}")]
    for macos_only in ("-D__OSX", "MACHINE = DARWIN", "USEFC = Y", "FORTLIBS ="):
        assert macos_only not in generated, (
            f"install/nubeam/linux.sh writes the macOS-only setting {macos_only}"
        )


def test_nubeam_linux_checks_pspline_symbols_with_elf_spelling():
    """Mach-O prefixes global symbols with an underscore; ELF does not.

    Carried over unchanged, the macOS pattern matches nothing here -- so a
    truncated libpspline.a would pass the very check that exists to catch it.
    """
    text = _executable_source(NUBEAM_DIR / "linux.sh")
    assert ' T $symbol\\$' in text, "match ELF symbols, not Mach-O ones"
    assert ' T _$symbol' not in text


def test_nubeam_validation_runs_its_comparison_rather_than_skipping_it():
    """The comparison is the point of the script, not an optional extra.

    It used to be guarded on `[[ -x compare-plasma-state.py ]]`, and that file
    is committed 100644 -- so on every fresh checkout, on every platform, the
    guard was false, the profile comparison was skipped, and the run still
    reported success. Test for the file and invoke the interpreter, so the
    check does not depend on a mode git records but Windows checkouts drop.
    """
    text = _executable_source(NUBEAM_DIR / "run-local-validation.sh")
    assert '-x "$SCRIPT_DIR/compare-plasma-state.py"' not in text, (
        "the execute bit is not a usable guard for a committed 100644 script"
    )
    assert '-f "$SCRIPT_DIR/compare-plasma-state.py"' in text
    assert "compare-plasma-state.py is missing" in text, (
        "a validation run that cannot compare must fail, not report success"
    )


def test_nubeam_helper_scripts_derive_their_build_directory():
    """The installers write build/darwin-* or build/linux-*, one per platform.

    run-local-validation.sh hardcoded the macOS path, so on Linux it worked in
    a darwin-arm64 directory that no build had ever written to.
    """
    for name in ("run-local-validation.sh", "run-local-vest.sh"):
        text = _executable_source(NUBEAM_DIR / name)
        assert "build/darwin-arm64" not in text, (
            f"install/nubeam/{name} hardcodes the macOS build directory"
        )


def test_nubeam_validation_records_both_platforms():
    text = (NUBEAM_DIR / "VALIDATION.md").read_text(encoding="utf-8")
    for fragment in ("Linux", "noise", "pcx_reco"):
        assert fragment in text, f"VALIDATION.md does not mention {fragment}"


# ---------------------------------------------------------------------------
# Three defects the Linux work surfaced, each of which made a checker or an
# installer disagree with the code it exists to serve.
# ---------------------------------------------------------------------------


def test_posix_installers_pick_netcdf_by_compiler_not_by_path_order():
    """`nf-config` first on PATH need not be the right netCDF-Fortran.

    One built with ifort ships ifort .mod files; linking them into a gfortran
    build fails with errors that never mention a compiler. nf-config reports
    its own --fc, so the right library is identifiable rather than guessable --
    and on the VAFT reference server the wrong one is first on PATH.
    """
    for name in ("install_efit.sh", "install_gpec.sh"):
        text = _executable_source(INSTALL / name)
        assert "--fc" in text, f"install/{name} must ask nf-config which compiler built it"
        assert "nf-config --prefix" not in text.replace('"$candidate" --prefix', ""), (
            f"install/{name} still takes the first nf-config on PATH"
        )


def test_default_prefix_answers_on_posix_too(monkeypatch):
    """Returning None off Windows is what recommended PowerShell to Linux users.

    The POSIX installers put the prefix inside the source tree, which is
    unguessable without --source and derivable with it.

    `default_prefix` chooses by whether `LOCALAPPDATA` is set, not by the
    operating system, so this controls that variable rather than asking which
    platform it runs on. Both branches are then checked on every runner. It
    used to read the real environment, which on Windows took the per-user
    branch and failed every assertion written for the other one.
    """
    module = _load_external_checker("_external_code_common.py")

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert module.default_prefix("chease", "/tmp/chease").as_posix().endswith(
        "/chease/vaft-install"
    )
    # Two codes place it differently, and both say so in their own recipes.
    assert module.default_prefix("nubeam", "/tmp/nubeam").as_posix().endswith("/nubeam/local")
    assert module.default_prefix("gacode", "/tmp/gacode").as_posix().endswith("/gacode")
    # Without a source there is still nothing to derive from.
    assert module.default_prefix("chease") is None

    # And the per-user branch answers from the code name alone, source or not.
    monkeypatch.setenv("LOCALAPPDATA", str(Path("/per-user")))
    expected = (Path("/per-user") / "vaft" / "external" / "chease").as_posix()
    assert module.default_prefix("chease").as_posix() == expected
    assert module.default_prefix("chease", "/tmp/chease").as_posix() == expected


@pytest.mark.parametrize("name", EXTERNAL_CODE_CHECKERS)
def test_checkers_never_recommend_powershell_on_posix(name):
    """A Linux operator told to launch PowerShell has been given a dead end."""
    if os.name == "nt":
        pytest.skip("the PowerShell remediation is correct on Windows")
    source = (INSTALL / name).read_text(encoding="utf-8")
    if "powershell" not in source.lower():
        return
    assert 'os.name == "nt"' in source or "sys.platform" in source, (
        f"install/{name} names a PowerShell script without asking what platform it is on"
    )


def test_efit_checker_resolves_the_layouts_the_runtime_resolves():
    """$EFITHOME may point at an installed prefix or at a CMake build tree.

    vaft/code/efit/toolchain.py tries both under one root. The checker tried
    only the first, so a working build-tree $EFITHOME failed the executables
    layer on the same run whose VAFT-discovery layer found it.
    """
    text = (INSTALL / "check_efit.py").read_text(encoding="utf-8")
    body = text[text.index("def _executables("):]
    body = body[: body.index("\ndef ")]
    assert "BUILD_TREE_LAYOUT" in body and "INSTALLED_LAYOUT" in body, (
        "the prefix branch must try both layouts, as toolchain.py does"
    )


# ---------------------------------------------------------------------------
# What the cold review turned up in the change above.
# ---------------------------------------------------------------------------


def test_nubeam_manifest_records_its_own_root():
    """Inferring the root from the entries is wrong when they share a subtree.

    The longest common prefix of `<root>/local/bin` and `<root>/local/lib` is
    `<root>/local`, one level too deep. Rebasing a relocated manifest against
    that maps recorded paths onto *different* real paths still inside the tree,
    where the outside-the-tree refusal cannot see them -- removing `<root>/bin`
    because the manifest said `<old>/local/bin`.
    """
    installer = _executable_source(NUBEAM_DIR / "linux.sh")
    assert "printf 'root\\t%s\\n'" in installer, "the installer must record the root"

    # Raw text, not _executable_source: the comment stripper cuts at the first
    # `#`, and the guard below is written with ${var##*/}.
    remover = (NUBEAM_DIR / "uninstall.sh").read_text(encoding="utf-8")
    assert '$1 == "root"' in remover, "the remover must prefer the recorded root"
    # And when there is none, the fallback is constrained rather than trusted.
    assert '"${common##*/}" == "${ROOT_DIR##*/}"' in remover, (
        "a guessed root must at least name the same tree"
    )


def test_nubeam_linux_does_not_derive_a_library_dir_from_a_bare_find():
    """`find` exits 0 and prints nothing when nothing matches.

    `dirname "$(find ... || echo /usr/lib)"` therefore yields "." rather than
    the fallback, and the link line silently becomes `-L.`.
    """
    text = _executable_source(NUBEAM_DIR / "linux.sh")
    assert 'dirname "$(find' not in text, (
        "a bare find cannot carry its own fallback; ask the compiler instead"
    )
    assert "-print-file-name=liblapack.so" in text


@pytest.mark.parametrize("name", EXTERNAL_CODE_POSIX_SCRIPTS)
def test_posix_external_installers_install_outside_every_checkout(name):
    """Nothing an installer writes may land inside the VAFT checkout.

    The comparison is made on canonical paths; the behaviour is exercised in
    test_posix_prefix_is_canonicalised_before_the_checkout_comparison.
    """
    text = _executable_source(INSTALL / name)
    assert "VAFT_ROOT=" in text, f"install/{name} must know where the checkout is"
    assert "must be outside the VAFT checkout" in text
    # A relative or non-canonical --prefix would otherwise slip past it.
    assert 'PREFIX="$(vaft_external_canonical_path "$PREFIX")"' in text
    assert 'vaft_external_is_inside "$PREFIX" "$VAFT_ROOT"' in text


def test_efit_checker_resolves_both_layouts_through_the_same_helper():
    """_resolve is what appends .exe on Windows.

    Testing the bare POSIX name inside the layout loop would miss a Windows
    build tree holding efit.exe -- the same checker/runtime contradiction the
    loop was added to remove, just confined to one platform.
    """
    text = (INSTALL / "check_efit.py").read_text(encoding="utf-8")
    body = text[text.index("def _executables("):]
    body = body[: body.index("\ndef ")]
    assert "_resolve(root / layout[role])" in body


@pytest.mark.parametrize(
    "name,variable",
    [("install_chease.sh", "CHEASEHOME"), ("install_gpec.sh", "GPECHOME")],
)
def test_check_only_reports_what_the_install_reports(name, variable):
    """--check-only exists to answer the same question the installer answers.

    The post-install verification passes the home variable; --check-only did
    not, so the discovery layer fell back to the ambient environment and failed
    on an installation that was fine.
    """
    text = _executable_source(INSTALL / name)
    assert f'exec env {variable}="$PREFIX"' in text


def test_both_gacode_recipes_build_what_vaft_drives():
    """A neo-only tree fails its own verification.

    `install/check_gacode.py` requires neo and tglf, and `vaft.code.gacode`
    resolves both, so `--codes` defaulting to `neo` alone on one platform and
    `neo,tglf` on the other made the same command produce a tree that passed on
    Linux and failed on macOS.
    """
    checker = (INSTALL / "check_gacode.py").read_text(encoding="utf-8")
    assert 'CODES = ("neo", "tglf")' in checker
    for name in GACODE_RECIPES:
        text = _executable_source(GACODE_DIR / name)
        assert 'CODES="neo,tglf"' in text, (
            f"install/gacode/{name} must default to the set the checker requires"
        )


def test_windows_nubeam_link_restores_a_space_separated_ifs():
    """`link_libraries` has to reach gfortran as separate arguments.

    The script runs under IFS=$'\\n\\t', where an unquoted command substitution
    does not split on spaces at all -- gfortran would be handed the whole
    `-L... -l... -l...` list as one malformed option. The three other call
    sites embed it in a quoted "VAR=..." string, where no splitting is wanted;
    only the bare one needs the IFS restored.
    """
    text = (NUBEAM_DIR / "windows.sh").read_text(encoding="utf-8")
    # The bare expansion is the plasma_state_test link; the three earlier ones
    # sit inside quoted "VAR=..." strings, so index() would find those first.
    bare = text.index("gfortran -o plasma_state_test.exe")
    preceding = text[:bare]
    assert "IFS=' '" in preceding[-600:], (
        "the bare $(link_libraries) expansion needs a space-separated IFS set "
        "just before it, inside the same subshell"
    )
    # The other call sites are inside quoted "VAR=..." strings and must not be
    # touched; if one of them ever goes bare it needs the same treatment.
    assert text.count("$(link_libraries)") == 4


def test_every_shell_entry_point_is_linted():
    """The CI glob covers the per-code recipes, not just install/*.sh.

    Naming files one at a time meant a new recipe was unlinted until somebody
    remembered to add it.
    """
    workflow = (ROOT / ".github" / "workflows" / "bootstrap-ci.yml").read_text(
        encoding="utf-8"
    )
    assert "shellcheck --severity=warning install/*.sh install/*/*.sh" in workflow


@pytest.mark.parametrize("name", GACODE_RECIPES)
def test_gacode_recipes_reject_an_empty_codes_list(name):
    """An empty --codes builds the suite root, not a suite member.

    `IFS=',' read -r -a CODE_LIST <<< ""` yields one empty element, and
    `[[ -d "$GACODE_ROOT/" ]]` is true for it, so the loop runs
    `make -C "$GACODE_ROOT/"` against the top-level Makefile.
    """
    # Raw text: _executable_source cuts at the first `#`, and the Linux guard
    # is written `(($# >= 2))`. Same trap as the ${var##*/} assertion elsewhere.
    text = (GACODE_DIR / name).read_text(encoding="utf-8")
    codes = text[text.index("--codes)"):]
    codes = codes[: codes.index(";;")]
    assert "needs a" in codes, f"install/gacode/{name} must refuse an empty --codes"


def test_gacode_verification_says_which_members_it_covered():
    """`reg18` exercises NEO alone, whichever members were built.

    The rule this pins has outlived the fact it was first written against.
    The macOS entry once said TGLF had not been built there at all, because
    it predated the `neo,tglf` default; TGLF has since been compiled from
    scratch on macOS, so that sentence would now be false. What still has to
    be said plainly is the narrower gap that remains: GACODE ships no TGLF
    regression the way it ships `reg18` for NEO, so for TGLF "Verified" means
    it builds and VAFT resolves it -- not that a number was reproduced.
    """
    text = (GACODE_DIR / "README.md").read_text(encoding="utf-8")
    verified = text[text.index("## Verified"):]
    macos = verified[verified.index("**macOS/arm64.**"):verified.index("**Linux/x86_64.**")]
    assert "`--codes neo,tglf`" in macos, "the macOS entry must name what it built"
    assert "exercises NEO alone" in macos, (
        "and say plainly that the one regression covers NEO, not TGLF"
    )
    assert "not a number" in macos, (
        "and that for TGLF nothing numerical was reproduced"
    )


def test_gpec_binaries_pin_the_netcdf_they_were_built_against():
    """LD_LIBRARY_PATH must not be able to substitute a different netCDF.

    A machine carrying a second, differently-compiled netCDF-Fortran on
    LD_LIBRARY_PATH -- an ifort build is the ordinary case on a cluster -- links
    correctly and then dies at run time with
    `undefined symbol: __netcdf_MOD_nf90_put_var_*`, naming neither the library
    nor the variable that chose it.

    --disable-new-dtags is the load-bearing half: current binutils emits
    DT_RUNPATH by default, and LD_LIBRARY_PATH takes precedence over RUNPATH,
    so the rpath would be present and ignored. DT_RPATH is searched first.
    """
    text = _executable_source(INSTALL / "install_gpec.sh")
    assert "-Wl,-rpath," in text
    assert "--disable-new-dtags" in text, (
        "without it the rpath is emitted as RUNPATH, which LD_LIBRARY_PATH overrides"
    )
    # And only on Linux: ld64 rejects --disable-new-dtags, Mach-O has no
    # DT_RUNPATH, and this script does not refuse Darwin -- so an unguarded
    # flag would fail the first link on a platform it claims to support.
    assert 'PLATFORM" == linux-*' in text, (
        "the ELF-only linker flags must be guarded by platform"
    )


def test_tokamaker_is_an_optional_extra_not_a_dependency():
    """It is the one external code pip can express, and still optional.

    VAFT drives TokaMaker in-process, so upstream's wheel is installable --
    unlike the Fortran codes behind install/. That does not make it required:
    the wheel carries compiled libraries and every other workflow runs without
    it.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras = text[text.index("[project.optional-dependencies]"):]
    extras = extras[: extras.index("\n[")]
    assert "tokamaker = [" in extras
    assert "openfusiontoolkit" in extras
    # And never among the hard dependencies.
    required = text[text.index("dependencies = ["):]
    required = required[: required.index("]")]
    assert "openfusiontoolkit" not in required


# ---------------------------------------------------------------------------
# 0.7.1: cold review `install`
# ---------------------------------------------------------------------------

_MANIFEST_WRITERS = (
    INSTALL / "install_chease.sh",
    INSTALL / "install_gpec.sh",
    INSTALL / "install_efit.sh",
    INSTALL / "nubeam" / "linux.sh",
)
_PYTHON_HEREDOC = re.compile(r'"\$PYTHON" - [^\n]*<<(\S+)\n(.*?)\nEOF\n', re.S)


@pytest.mark.parametrize("path", _MANIFEST_WRITERS, ids=lambda p: p.name)
def test_manifest_python_never_interpolates_shell_values(path, tmp_path):
    """Cold review install F20: shell text must not become Python source.

    The record was written by an unquoted heredoc holding
    `\"\"\"$DIRTY_FILES\"\"\"`. `git status --porcelain` quotes a path with a
    space in it, so the literal ended in four quotes: a SyntaxError after bin/
    was installed and before the manifest existed, which --uninstall then
    refused. Values travel through the environment instead.
    """
    text = path.read_text(encoding="utf-8")
    blocks = _PYTHON_HEREDOC.findall(text)
    assert blocks, f"{path.name} no longer writes its record through a heredoc"
    for delimiter, body in blocks:
        assert delimiter == "'EOF'", f"{path.name}: the Python heredoc must be quoted"
        assert "$" not in body, f"{path.name}: a shell expansion inside Python source"

    # Run the record writer itself against values that used to break it.
    delimiter, body = blocks[-1]
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    for name in ("chease", "efit", "efund", "dcon"):
        (prefix / "bin" / name).write_text("x", encoding="utf-8")
    hostile = 'a"b \\ $(touch nope) \'c'
    names = re.findall(r"^\s*VAFT_MANIFEST_([A-Z_]+)=", text, re.M)
    assert names
    environment = dict(os.environ)
    environment.update({f"VAFT_MANIFEST_{name}": hostile for name in names})
    environment.update(
        VAFT_MANIFEST_PREFIX=str(prefix),
        VAFT_MANIFEST_DIRTY_FILES=' M "src-f90/my file.f90"',
        VAFT_MANIFEST_PROGRAMS_LINE="dcon",
        VAFT_MANIFEST_NETCDF_ACHIEVED="1",
        VAFT_MANIFEST_WITH_NETCDF="1",
        VAFT_MANIFEST_PREFIX_CREATED="1",
        VAFT_MANIFEST_ACCEPTANCE="passed",
        VAFT_MANIFEST_NUBEAM_CPP_SOURCE=str(prefix / "bin" / "chease"),
    )
    record = tmp_path / "record.json"
    completed = subprocess.run(
        [sys.executable, "-", str(record)],
        input=body, text=True, capture_output=True, env=environment, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    written = json.loads(record.read_text(encoding="utf-8"))
    assert written["source"] == hostile
    if "source_dirty_files" in written:
        assert written["source_dirty_files"] == [' M "src-f90/my file.f90"']


_PREFIX_HELPER = INSTALL / "_external_code_common.sh"
_PREFIX_HARNESS = r"""
set -euo pipefail
IFS=$'\n\t'
die() { printf 'DIE: %s\n' "$*" >&2; exit 1; }
note() { printf '%s\n' "$*"; }
. "$1"; shift
"$@"
if [[ -n "${PREFIX_CREATED+x}" ]]; then printf 'PREFIX_CREATED=%s\n' "$PREFIX_CREATED"; fi
"""


def _prefix_helper(*arguments, cwd=None):
    """Run one function of install/_external_code_common.sh in a subshell.

    The functions touch nothing outside the directory they are handed, so this
    is the installers' ownership logic executed for real, on a temp tree.
    """
    return subprocess.run(
        [BASH, "-c", _PREFIX_HARNESS, "harness", str(_PREFIX_HELPER), *map(str, arguments)],
        capture_output=True, text=True, timeout=60, cwd=cwd,
    )


def _write_manifest(prefix: Path, code: str, **extra) -> None:
    record = {"code": code, "prefix": str(prefix), **extra}
    (prefix / "vaft-external-install.json").write_text(json.dumps(record), encoding="utf-8")


@requires_bash
def test_posix_uninstall_spares_a_prefix_the_installer_did_not_create(tmp_path):
    """Cold review install F5: `--prefix ~/.local` must survive `--uninstall`.

    The only guard was "our manifest is in the prefix", which the installer
    satisfies for any directory by writing it there, and the removal was
    `rm -rf "$PREFIX"`. This is that scenario with a 0.7.0 manifest: what was
    installed goes, everything else -- and the directory -- stays.
    """
    prefix = tmp_path / "dot-local"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "logs").mkdir()
    (prefix / "share").mkdir()
    (prefix / "keep.txt").write_text("mine", encoding="utf-8")
    (prefix / "bin" / "pip").write_text("mine", encoding="utf-8")
    (prefix / "bin" / "chease").write_text("theirs", encoding="utf-8")
    (prefix / "logs" / "chease-build-20260101-000000.log").write_text("log", encoding="utf-8")
    _write_manifest(
        prefix, "chease",
        executables={"chease": {"path": str(prefix / "bin" / "chease")}},
    )
    done = _prefix_helper("vaft_external_uninstall_prefix", prefix, "chease", sys.executable)
    assert done.returncode == 0, done.stderr
    assert (prefix / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert (prefix / "bin" / "pip").is_file()
    assert (prefix / "share").is_dir()
    assert not (prefix / "bin" / "chease").exists()
    assert not (prefix / "logs").exists()
    assert not (prefix / "vaft-external-install.json").exists()


@requires_bash
def test_posix_install_refuses_a_populated_prefix_it_does_not_own(tmp_path):
    """Cold review install F5, install side: ownership is decided before writing."""
    prefix = tmp_path / "opt-fusion"
    prefix.mkdir()
    (prefix / "keep.txt").write_text("mine", encoding="utf-8")
    refused = _prefix_helper("vaft_external_claim_prefix", prefix, "gpec", sys.executable)
    assert refused.returncode == 1
    assert "not created by this script" in refused.stderr
    assert sorted(child.name for child in prefix.iterdir()) == ["keep.txt"]

    # A prefix shared between codes: each keeps its record under one name, so
    # the last install would authorise removing the others.
    _write_manifest(prefix, "chease")
    other = _prefix_helper("vaft_external_claim_prefix", prefix, "gpec", sys.executable)
    assert other.returncode == 1 and "'chease'" in other.stderr
    wrong = _prefix_helper("vaft_external_uninstall_prefix", prefix, "gpec", sys.executable)
    assert wrong.returncode == 1
    assert (prefix / "vaft-external-install.json").is_file()

    # An existing but empty directory is accepted, and is not ours to remove.
    empty = tmp_path / "empty"
    empty.mkdir()
    claimed = _prefix_helper("vaft_external_claim_prefix", empty, "gpec", sys.executable)
    assert claimed.returncode == 0, claimed.stderr
    assert "PREFIX_CREATED=0" in claimed.stdout
    assert _prefix_helper("vaft_external_uninstall_prefix", empty, "gpec", sys.executable).returncode == 0
    assert empty.is_dir() and not any(empty.iterdir())


@requires_bash
def test_posix_install_and_uninstall_round_trip(tmp_path):
    """A prefix the installer created is removed, but only once it is empty."""
    prefix = tmp_path / "source" / "vaft-install"
    (tmp_path / "source").mkdir()
    claimed = _prefix_helper("vaft_external_claim_prefix", prefix, "efit", sys.executable)
    assert claimed.returncode == 0, claimed.stderr
    assert "PREFIX_CREATED=1" in claimed.stdout

    # A build that failed leaves logs and no manifest; the next run must still
    # recognise the prefix as its own, and remember that it created it.
    (prefix / "logs").mkdir()
    (prefix / "logs" / "efit-build-1.log").write_text("log", encoding="utf-8")
    again = _prefix_helper("vaft_external_claim_prefix", prefix, "efit", sys.executable)
    assert again.returncode == 0, again.stderr
    assert "PREFIX_CREATED=1" in again.stdout

    (prefix / "bin").mkdir()
    for name in ("efit", "efund"):
        (prefix / "bin" / name).write_text("x", encoding="utf-8")
    _write_manifest(prefix, "efit", prefix_created=True, installed_files=["bin/efit", "bin/efund"])
    (prefix / "notes.txt").write_text("mine", encoding="utf-8")
    first = _prefix_helper("vaft_external_uninstall_prefix", prefix, "efit", sys.executable)
    assert first.returncode == 0, first.stderr
    assert sorted(child.name for child in prefix.iterdir()) == ["notes.txt"]

    (prefix / "notes.txt").unlink()
    _prefix_helper("vaft_external_claim_prefix", prefix, "efit", sys.executable)
    second = _prefix_helper("vaft_external_uninstall_prefix", prefix, "efit", sys.executable)
    assert second.returncode == 0, second.stderr
    # Re-claimed as an existing empty directory, so no longer this script's.
    assert prefix.is_dir()

    fresh = tmp_path / "fresh"
    _prefix_helper("vaft_external_claim_prefix", fresh, "efit", sys.executable)
    assert _prefix_helper("vaft_external_uninstall_prefix", fresh, "efit", sys.executable).returncode == 0
    assert not fresh.exists()


@requires_bash
def test_posix_uninstall_refuses_a_manifest_entry_outside_the_prefix(tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("mine", encoding="utf-8")
    for entry in ("../victim.txt", str(victim), "bin/../../victim.txt"):
        _write_manifest(prefix, "chease", installed_files=[entry])
        done = _prefix_helper("vaft_external_uninstall_prefix", prefix, "chease", sys.executable)
        assert done.returncode == 1, entry
        assert victim.is_file()


@requires_bash
def test_posix_prefix_is_canonicalised_before_the_checkout_comparison(tmp_path):
    """`/tmp/../<checkout>/x` and a symlink both name a path inside the checkout."""
    checkout = tmp_path / "vaft"
    (checkout / "install").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "link").symlink_to(checkout)
    root = _prefix_helper("vaft_external_canonical_path", checkout).stdout.strip()
    for spelling in (
        f"{tmp_path}/elsewhere/../vaft/x",
        f"{tmp_path}/link/x/y",
        f"{checkout}/./install/new",
        "vaft/x",
    ):
        resolved = _prefix_helper("vaft_external_canonical_path", spelling, cwd=tmp_path)
        assert resolved.returncode == 0, spelling
        inside = _prefix_helper("vaft_external_is_inside", resolved.stdout.strip(), root)
        assert inside.returncode == 0, f"{spelling} -> {resolved.stdout!r} escaped the guard"
    outside = _prefix_helper("vaft_external_canonical_path", f"{tmp_path}/vaft-install/new")
    assert _prefix_helper("vaft_external_is_inside", outside.stdout.strip(), root).returncode == 1
    # A `..` below a directory that does not exist cannot be resolved: refused.
    assert _prefix_helper("vaft_external_canonical_path", f"{tmp_path}/missing/../vaft/x").returncode == 1


@pytest.mark.parametrize("name", EXTERNAL_CODE_POSIX_SCRIPTS)
def test_posix_external_installers_never_remove_the_prefix_wholesale(name):
    text = _executable_source(INSTALL / name)
    assert 'rm -rf "$PREFIX"' not in text
    assert f'vaft_external_uninstall_prefix "$PREFIX" {name[len("install_"):-3]} ' in text
    assert f'vaft_external_claim_prefix "$PREFIX" {name[len("install_"):-3]} ' in text
    # Claimed before the first thing is written into the prefix.
    assert text.index("vaft_external_claim_prefix") < text.index('mkdir -p "$PREFIX')
    assert "rm -rf" not in _executable_source(_PREFIX_HELPER)


def test_windows_uninstall_is_driven_by_an_ownership_record():
    """Cold review install F5, Windows half (read, not run: no pwsh in CI here).

    `-Uninstall -Prefix X` resolved X and removed it recursively with no check
    of any kind.
    """
    shared = _executable_source(INSTALL / "_external_code_common.ps1")
    removal = shared[shared.index("function Remove-InstallPrefix"):]
    removal = removal[: removal.index("\nfunction ", 1)]
    assert "Get-PrefixRecord" in removal
    assert removal.index("Stop-WithGuidance") < removal.index("Remove-Item")
    # Never the prefix recursively: only the owned children are.
    assert "Remove-Item -LiteralPath $Prefix -Recurse" not in removal
    assert "Remove-Item -LiteralPath $Prefix -Force" in removal
    resolve = shared[shared.index("function Resolve-InstallPrefix"):]
    resolve = resolve[: resolve.index("\nfunction ", 1)]
    assert "was not created by this script" in resolve
    assert "prefix_created" in resolve
    for name in EXTERNAL_CODE_WINDOWS_SCRIPTS:
        text = _executable_source(INSTALL / name)
        block = text[text.index("if ($Uninstall) {"):]
        block = block[: block.index("exit 0")]
        assert "Remove-InstallPrefix -Prefix $resolved -CodeName $CodeName" in block
        assert "-Recurse" not in block, f"install/{name} still removes the prefix wholesale"


@requires_bash
@pytest.mark.parametrize("name", EXTERNAL_CODE_POSIX_SCRIPTS)
@pytest.mark.parametrize("checker_status", [0, 1])
def test_posix_installers_exit_with_the_acceptance_status(name, checker_status, tmp_path):
    """Cold review install F8: `check_*.py ... || true` made a failed build exit 0.

    The Windows installers end with the checker's status; the POSIX ones threw
    it away and printed "Point VAFT at this build". This runs the script's own
    acceptance tail with a stand-in for the interpreter.
    """
    text = (INSTALL / name).read_text(encoding="utf-8")
    assert not re.search(r"check_\w+\.py[^\n]*\|\| true", text)
    tail = text[text.index("\nACCEPTANCE_STATUS=0\n"):]
    stub = tmp_path / "python-stub"
    stub.write_text(f"#!/bin/sh\necho '[checker ran]'\nexit {checker_status}\n", encoding="utf-8")
    stub.chmod(0o755)
    preamble = (
        "set -euo pipefail\nnote() { printf '%s\\n' \"$*\"; }\n"
        f"PYTHON='{stub}' SCRIPT_DIR=/nonexistent SOURCE=/src PREFIX=/prefix\n"
        "SKIP_TESTS=0 CTEST_STATUS=passed LOG=/log\n"
    )
    done = subprocess.run([BASH, "-c", preamble + tail], capture_output=True, text=True, timeout=60)
    assert "[checker ran]" in done.stdout
    assert "HOME=/prefix" in done.stdout, "the export hint comes before the verdict"
    assert done.returncode == checker_status
    if checker_status:
        assert "[FAIL]" in done.stderr


@requires_bash
def test_efit_installer_fails_on_a_failed_ctest_and_skip_tests_is_explicit(tmp_path):
    text = (INSTALL / "install_efit.sh").read_text(encoding="utf-8")
    tail = text[text.index("\nACCEPTANCE_STATUS=0\n"):]
    stub = tmp_path / "python-stub"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    preamble = (
        "set -euo pipefail\nnote() { :; }\n"
        f"PYTHON='{stub}' SCRIPT_DIR=/x SOURCE=/src PREFIX=/prefix CTEST_STATUS=failed LOG=/log\n"
    )
    done = subprocess.run([BASH, "-c", preamble + tail], capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and "ctest" in done.stderr

    # --skip-tests: CHEASE and GPEC install unverified and say so.
    for name in ("install_chease.sh", "install_gpec.sh"):
        body = (INSTALL / name).read_text(encoding="utf-8")
        tail = body[body.index("\nACCEPTANCE_STATUS=0\n"):]
        preamble = (
            "set -euo pipefail\nnote() { printf '%s\\n' \"$*\"; }\n"
            "PYTHON=/nonexistent SCRIPT_DIR=/x SOURCE=/src PREFIX=/prefix SKIP_TESTS=1\n"
        )
        done = subprocess.run([BASH, "-c", preamble + tail], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0 and "unverified" in done.stdout


@requires_bash
def test_efit_build_directory_is_absolute_before_it_is_recorded_or_removed(tmp_path):
    """Cold review install F10: `--build-dir build` was recorded verbatim.

    --uninstall then ran `rm -rf build` relative to wherever it was started.
    """
    text = (INSTALL / "install_efit.sh").read_text(encoding="utf-8")
    absolutise = text.index('BUILD_DIR="$(vaft_external_canonical_path "$BUILD_DIR")"')
    for later in ('if ((UNINSTALL)); then', 'rm -rf "$BUILD_DIR"', 'VAFT_MANIFEST_BUILD_DIR='):
        assert absolutise < text.index(later)
    assert 'vaft_external_is_inside "$BUILD_DIR" "$VAFT_ROOT"' in text

    # The uninstall side: a recorded directory is removed only when it proves
    # to be a CMake tree configured from this source, by absolute path.
    start = text.index("efit_build_dir_is_ours() {")
    function = text[start: text.index("\n}\n", start) + 3]
    source = tmp_path / "efit"
    ours = tmp_path / "build-ours"
    other = tmp_path / "build"
    for directory in (source, ours, other):
        directory.mkdir()
    (ours / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n", encoding="utf-8"
    )
    (other / "CMakeCache.txt").write_text(
        "CMAKE_HOME_DIRECTORY:INTERNAL=/somewhere/else\n", encoding="utf-8"
    )

    def judged(directory):
        script = function + 'efit_build_dir_is_ours "$1" "$2"'
        return subprocess.run(
            [BASH, "-c", script, "x", str(directory), str(source)], cwd=tmp_path, timeout=60
        ).returncode

    assert judged(ours) == 0
    assert judged(other) == 1, "somebody else's CMake tree"
    assert judged("build-ours") == 1, "a relative path from a 0.7.0 manifest"
    assert judged(tmp_path / "missing") == 1


def _nubeam_windows_uninstall_block() -> str:
    wrapper = _executable_source(NUBEAM_DIR / "windows.ps1")
    block = wrapper[wrapper.index("if ($Uninstall) {"):]
    return block[: block.index("exit 0")]


def test_nubeam_windows_uninstall_checks_the_tree_and_the_manifest_first():
    """Cold review install F4 (read, not run: there is no pwsh here).

    `windows.ps1 <any path> -Uninstall` removed local\\, build\\windows-x86_64\\
    and vendor\\ntcc\\ under whatever path it was given: the source-tree
    assertion ran only on the build path, and no manifest was consulted.
    """
    block = _nubeam_windows_uninstall_block()
    first_removal = block.index("Remove-Item")
    assert block.index("Assert-SourceCheckout") < first_removal
    refusal = block.index("Nothing was removed")
    assert block.index("$ManifestName") < refusal < first_removal
    # The environment variable is not touched either before ownership is known.
    assert refusal < block.index("Remove-ExternalCodeEnvironment")


def test_nubeam_windows_uninstall_spares_hand_placed_ntcc_sources():
    """vendor\\ntcc is operator-populated whenever the PPPL download fails.

    The recipe uses such a tree as-is, so only modules it recorded downloading
    may be removed -- and the directory itself only once it is empty.
    """
    wrapper = _executable_source(NUBEAM_DIR / "windows.ps1")
    generated = re.search(r"\$GeneratedPaths = @\(([^)]*)\)", wrapper)
    assert generated and "ntcc" not in generated.group(1)
    block = _nubeam_windows_uninstall_block()
    assert "'managed_dir'" in block and "/vendor/ntcc/" in block
    assert "Remove-Item -LiteralPath $target -Force\n" in block  # parents: non-recursive
    parents = block[block.index("foreach ($relative in @('vendor\\ntcc'"):]
    assert "-Recurse" not in parents[: parents.index("foreach ($makeLocal")]

    recipe = (NUBEAM_DIR / "windows.sh").read_text(encoding="utf-8")
    assert ': > "$MANIFEST"' not in recipe, "an empty manifest records nothing to remove"
    assert "printf 'root\\t%s\\n' \"$ROOT_DIR\"" in recipe
    function = recipe[recipe.index("download_ntcc_module() {"):]
    function = function[: function.index("\n}\n")]
    already_there = function.index('[[ -d "$destination" ]] && return 0')
    recorded = function.index("printf 'managed_dir\\t%s\\n' \"$destination\" >> \"$MANIFEST\"")
    assert already_there < recorded < function.index('cp -R "$candidate"')


_ALL_POWERSHELL = (
    *(INSTALL / name for name in POWERSHELL_SCRIPTS),
    NUBEAM_DIR / "windows.ps1",
)
_POWERSHELL_OPERATORS = {
    "join", "f", "eq", "ne", "not", "and", "or", "replace", "split", "match",
    "notmatch", "like", "contains", "notcontains", "gt", "lt", "ge", "le", "in",
}
_POWERSHELL_COMMON_PARAMETERS = {"ErrorAction", "Verbose", "WarningAction", "OutVariable"}


def _common_powershell_signatures() -> dict[str, set[str]]:
    text = _executable_source(INSTALL / "_external_code_common.ps1")
    signatures = {}
    for match in re.finditer(r"(?m)^function ([A-Za-z0-9-]+) \{", text):
        body = text[match.end():]
        following = re.search(r"(?m)^function ", body)
        body = body[: following.start()] if following else body
        block = re.search(r"(?s)\bparam\((.*?)\n    \)", body)
        one_line = re.search(r"\bparam\(([^\n]*)\)", body)
        declared = (block or one_line).group(1) if (block or one_line) else ""
        signatures[match.group(1)] = set(re.findall(r"\$(\w+)", declared))
    return signatures


def test_shared_powershell_helpers_are_called_with_parameters_they_declare():
    """Cold review install F1: a parameter the helper does not have stops the build.

    nubeam\\windows.ps1 called `Write-RevisionResult -SourcePath $source`; the
    helper declares -Project and -Revision. Under $ErrorActionPreference =
    'Stop' that binding error ended every real NUBEAM build on Windows.
    """
    signatures = _common_powershell_signatures()
    assert signatures["Write-RevisionResult"] == {"Project", "Revision"}
    for path in _ALL_POWERSHELL:
        text = _executable_source(path).replace("`\n", " ")
        if "_external_code_common.ps1" not in text and path.name != "_external_code_common.ps1":
            continue  # the VAFT bootstrap has helpers of its own under these names
        for name, declared in signatures.items():
            for call in re.finditer(rf"(?m)(?<![\w-]){re.escape(name)}((?: [^\n|]*)?)$|(?<![\w-]){re.escape(name)} ([^\n|]*)\|", text):
                arguments = call.group(1) or call.group(2) or ""
                if arguments.lstrip().startswith("{"):
                    continue  # the definition itself
                # Message text such as "rerun with -Recreate" is not a parameter.
                arguments = re.sub(r"'[^']*'|\"[^\"]*\"", "''", arguments)
                # Nor is what a nested call in parentheses is given.
                while re.search(r"\([^()]*\)", arguments):
                    arguments = re.sub(r"\([^()]*\)", "''", arguments)
                used = set(re.findall(r"(?<=\s)-([A-Za-z]\w*)", arguments))
                unknown = used - declared - _POWERSHELL_OPERATORS - _POWERSHELL_COMMON_PARAMETERS
                assert not unknown, f"{path.name}: {name} has no parameter {sorted(unknown)}"


def test_nubeam_windows_wrapper_does_not_create_the_directory_its_recipe_refuses():
    """Cold review install F2: the wrapper's log directory was the recipe's BUILD_DIR.

    windows.sh dies on a build/windows-x86_64 that exists without a manifest,
    and the wrapper created exactly that for its log before calling it.
    """
    recipe = (NUBEAM_DIR / "windows.sh").read_text(encoding="utf-8")
    assert 'BUILD_DIR="$ROOT_DIR/build/windows-x86_64"' in recipe
    assert 'elif [[ -e "$BUILD_DIR" ]]; then' in recipe
    wrapper = _executable_source(NUBEAM_DIR / "windows.ps1")
    assert "$logDirectory = Join-Path $source 'build'\n" in wrapper
    created = re.findall(r"New-Item -ItemType Directory -Path (\$\w+)", wrapper)
    assert created == ["$logDirectory"]


def test_toolchain_installation_returns_only_the_msys2_root():
    """Cold review install F7: bare `& winget install` leaked into the return value."""
    text = _executable_source(INSTALL / "_external_code_common.ps1")
    body = text[text.index("function Install-Msys2Toolchain {"):]
    body = body[: body.index("\nfunction ", 1)]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("& ") or stripped.startswith("Invoke-Msys2 "):
            assert re.search(r"\| (Out-Host|Out-Null)$", stripped), stripped


def test_nubeam_windows_recipe_says_it_is_experimental():
    assert "EXPERIMENTAL" in (NUBEAM_DIR / "windows.ps1").read_text(encoding="utf-8")
    readme = (NUBEAM_DIR / "README.md").read_text(encoding="utf-8")
    assert "Experimental and unverified" in readme


def test_nubeam_windows_commands_name_a_path_that_exists():
    """Cold review install F13: nine references to external\\nubeam, which is gone.

    The recipes moved to install/nubeam; the README, Get-Help, the script's own
    guidance and its NEXT block still sent the operator to the old path.
    """
    for path in (NUBEAM_DIR / "README.md", NUBEAM_DIR / "windows.ps1"):
        text = path.read_text(encoding="utf-8")
        assert "external\\nubeam" not in text, path.name
        for script in re.findall(r"-File ([\w\\]+\.ps1)", text):
            assert (ROOT / Path(*script.split("\\"))).is_file(), f"{path.name}: {script}"
    wrapper = (NUBEAM_DIR / "windows.ps1").read_text(encoding="utf-8")
    # The harness needs --nubeam-root and refuses any platform but Darwin/Linux.
    assert "run-local-validation.sh --case" not in wrapper


@requires_bash
def test_nubeam_macos_derives_the_gcc_major_from_the_selected_gfortran(tmp_path):
    """Cold review install F3: gcc-15/g++-15 were literals; Homebrew's gcc is 16.

    Every macOS user with a current Homebrew died at "GCC executables were not
    found" before anything was built.
    """
    text = (NUBEAM_DIR / "macos.sh").read_text(encoding="utf-8")
    assert not re.search(r"(gcc|g\+\+|gfortran)-\d+", _executable_source(NUBEAM_DIR / "macos.sh"))
    start = text.index("select_homebrew_compilers() {")
    function = text[start: text.index("\n}\n", start) + 3]
    keg = tmp_path / "gcc" / "bin"
    keg.mkdir(parents=True)
    for name, body in (("gfortran", "echo 16.2.0"), ("gcc-16", ":"), ("g++-16", ":"), ("gcc-15", ":")):
        (keg / name).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        (keg / name).chmod(0o755)
    script = (
        "set -euo pipefail\ndie() { printf 'DIE: %s\\n' \"$*\" >&2; exit 1; }\n"
        + function
        + 'select_homebrew_compilers "$1"; printf \'%s\\n\' "$FC" "$CC" "$CXX"\n'
    )

    def select():
        return subprocess.run(
            [BASH, "-c", script, "x", str(keg.parent)], capture_output=True, text=True, timeout=60
        )

    chosen = select()
    assert chosen.returncode == 0, chosen.stderr
    assert chosen.stdout.split() == [str(keg / "gfortran"), str(keg / "gcc-16"), str(keg / "g++-16")]

    (keg / "g++-16").unlink()
    refused = select()
    assert refused.returncode == 1
    assert "g++-16" in refused.stderr and "brew reinstall gcc" in refused.stderr


def test_chease_installers_and_checker_agree_on_what_a_checkout_is():
    """Cold review install F6: the Windows installer required a generated file.

    `chease_prog_effxml.f90` is not tracked -- src-f90/Makefile deletes it at
    parse time and the build rewrites it -- so a fresh clone was rejected on
    Windows while the POSIX installer and the checker accepted it.
    """
    spec = importlib.util.spec_from_file_location("check_chease_markers", INSTALL / "check_chease.py")
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(INSTALL))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(INSTALL))
    expected = list(module.SOURCE_MARKERS)

    posix = _executable_source(INSTALL / "install_chease.sh")
    listed = re.search(r"for marker in ([^;]+); do", posix)
    assert listed and listed.group(1).split() == expected

    windows = _executable_source(INSTALL / "install_chease_windows.ps1")
    declared = re.search(r"-ExpectedFiles @\(([^)]*)\)", windows)
    assert declared
    names = [name.replace("\\", "/") for name in re.findall(r"'([^']+)'", declared.group(1))]
    assert names == expected
