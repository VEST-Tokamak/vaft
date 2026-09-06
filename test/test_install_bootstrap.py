"""Contract for the cross-platform bootstrap under ``install/`` (issue #225).

These tests keep three promises the student-facing bootstrap makes:

* the checker diagnoses a broken environment instead of only crashing;
* nothing under ``install/`` can destroy a student's local work; and
* ``environment.yml`` and ``pyproject.toml`` cannot drift apart.
"""

from __future__ import annotations

import importlib.util
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

POSIX_SCRIPTS = ("linux.sh", "macos.sh", "windows_wsl.sh", "uninstall.sh", "_common.sh")
PLATFORM_SCRIPTS = ("linux.sh", "macos.sh", "windows_wsl.sh", "windows_native.ps1")
# Removal is identical on every POSIX platform, so it needs one entry point,
# not one per platform.
UNINSTALL_SCRIPTS = ("uninstall.sh", "uninstall_windows_native.ps1")
# The external Fortran codes are their own entry points, deliberately separate
# from the VAFT bootstrap: building them takes tens of minutes and needs a
# compiler toolchain, neither of which belongs in the path a student runs first.
EXTERNAL_CODE_SCRIPTS = ("install_chease_windows.ps1", "install_gpec_windows.ps1")
EXTERNAL_CODE_CHECKERS = ("check_chease.py", "check_gpec.py")
POWERSHELL_SCRIPTS = (
    "windows_native.ps1",
    "uninstall_windows_native.ps1",
    "_external_code_common.ps1",
    *EXTERNAL_CODE_SCRIPTS,
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


def test_install_directory_is_flat_and_complete():
    """Issue #225 requires a flat install/ with one entry point per platform."""
    assert INSTALL.is_dir()
    expected = (
        *PLATFORM_SCRIPTS,
        *UNINSTALL_SCRIPTS,
        *EXTERNAL_CODE_SCRIPTS,
        *EXTERNAL_CODE_CHECKERS,
        "_common.sh",
        "_external_code_common.ps1",
        "_external_code_common.py",
        "README.md",
        "check_vaft_environment.py",
    )
    for name in expected:
        assert (INSTALL / name).is_file(), f"install/{name} is missing"
    subdirectories = [
        child.name
        for child in INSTALL.iterdir()
        if child.is_dir() and child.name != "__pycache__"
    ]
    assert not subdirectories, (
        f"install/ must stay flat: no platform or checker subdirectories, found {subdirectories}"
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
    for name in EXTERNAL_CODE_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "[Parameter(Position = 0)] [string] $SourcePath" in text, (
            f"install/{name} must take the source path as its first argument"
        )
        assert "Assert-SourceCheckout" in text, (
            f"install/{name} must validate the path it was given"
        )


def test_toolchain_installation_is_opt_in():
    """Installing a compiler system-wide is the operator's decision.

    The bootstrap already treats Git, Conda and WSL2 as things it will not
    install for you. A Fortran toolchain is no different.
    """
    for name in EXTERNAL_CODE_SCRIPTS:
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
    for name in EXTERNAL_CODE_SCRIPTS:
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "Resolve-InstallPrefix" in text, f"install/{name} must validate its prefix"


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
        "check_gpec.py": ("toolchain", "source", "build record", "executables", "discovery", "handoff"),
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
        "check_chease.py",
        "check_gpec.py",
        "-InstallToolchain",
        "MSYS2",
    ):
        assert fragment in text, f"install/README.md does not mention {fragment}"
