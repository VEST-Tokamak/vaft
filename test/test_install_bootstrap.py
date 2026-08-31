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

POSIX_SCRIPTS = ("linux.sh", "macos.sh", "windows_wsl.sh", "_common.sh")
PLATFORM_SCRIPTS = ("linux.sh", "macos.sh", "windows_wsl.sh", "windows_native.ps1")


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
    for name in (*PLATFORM_SCRIPTS, "_common.sh", "README.md", "check_vaft_environment.py"):
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
    for name in ("linux.sh", "macos.sh", "windows_wsl.sh"):
        text = (INSTALL / name).read_text(encoding="utf-8")
        assert "_common.sh" in text, f"install/{name} must source install/_common.sh"
        assert "vaft_bootstrap_main" in text
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
    text = (INSTALL / "windows_native.ps1").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Invoke-InVaft") and "= Invoke-InVaft" not in stripped:
            continue
        call = stripped.split("Invoke-InVaft", 1)[1].strip()
        assert call.startswith("@("), (
            f"pass an argument array, not bare flags: {stripped}"
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
@pytest.mark.parametrize("name", POSIX_SCRIPTS)
def test_posix_scripts_parse(name):
    subprocess.run(["bash", "-n", str(INSTALL / name)], check=True, timeout=60)


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell parsing is checked on Windows")
def test_powershell_script_parses():
    script = INSTALL / "windows_native.ps1"
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
existing = [name for name in os.environ.get("FAKE_CONDA_ENVS", "").split(",") if name]
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


def _run_bootstrap(script: str = "linux.sh", *arguments: str):
    completed = subprocess.run(
        ["bash", str(INSTALL / script), *arguments],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=300,
    )
    return completed


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_bootstrap_creates_a_missing_environment(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    invocations = fake_conda.read_text(encoding="utf-8")
    assert "env create --name vaft" in invocations
    assert "env update" not in invocations
    assert "[PASS] vaft environment" in completed.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_bootstrap_reuses_an_existing_environment(fake_conda, monkeypatch):
    """Rerunning must update in place, never recreate a working environment."""
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    invocations = fake_conda.read_text(encoding="utf-8")
    assert "env update --name vaft" in invocations
    assert "env create" not in invocations


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_bootstrap_never_touches_another_environment(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft,someone_elses_project")
    completed = _run_bootstrap()
    assert completed.returncode == 0, completed.stderr
    for line in fake_conda.read_text(encoding="utf-8").splitlines():
        if "--name" in line:
            assert "--name vaft" in line, f"conda was pointed at another environment: {line}"
    assert "someone_elses_project" not in fake_conda.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
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
    for name in PLATFORM_SCRIPTS:
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_check_only_mode_changes_nothing(fake_conda, monkeypatch):
    monkeypatch.setenv("FAKE_CONDA_ENVS", "vaft")
    completed = _run_bootstrap("linux.sh", "--check-only")
    invocations = fake_conda.read_text(encoding="utf-8")
    assert "env create" not in invocations
    assert "env update" not in invocations
    assert "pip install" not in invocations
    assert "check_vaft_environment.py" in completed.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_bootstrap_reports_missing_conda_with_guidance(tmp_path, monkeypatch):
    """Without Conda the script must explain the fix, not traceback."""
    bash = shutil.which("bash")
    assert bash

    # Do not simply prepend an empty directory to PATH: hosted CI runners expose
    # `conda` from the standard system directories, so the script would find one,
    # really build an environment, and hang on an interactive prompt. Instead give
    # it a sandbox containing only the handful of external tools it needs before
    # the Conda check -- and, deliberately, no conda.
    sandbox = tmp_path / "bin"
    sandbox.mkdir()
    for tool in ("dirname", "basename", "uname", "awk", "grep", "cat", "sed", "env"):
        located = shutil.which(tool)
        if located:
            (sandbox / tool).symlink_to(located)

    environment = {"PATH": str(sandbox), "HOME": os.environ.get("HOME", str(tmp_path))}
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
