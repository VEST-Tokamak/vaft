"""Run the TES (``rtes``) binary on prepared inputs."""

from __future__ import annotations

import os
from pathlib import Path

from ...compat import is_executable, resolve_executable
from .._executables import executable_from_home, missing_home_message
from ..execution import ExecutionRequest, resolve_backend
from .config import TESConfig, TESInputs, TESResult
from .outputs import collect_tes_outputs

TES_HOME_ENV = "TESHOME"
TES_HOME_EXECUTABLE = Path("bin/rtes")
TES_COMPATIBILITY_ENV = "RTES"


def _resolve_executable(config: TESConfig) -> str:
    if config.executable:
        requested = Path(config.executable).expanduser()
        exe = resolve_executable(requested) or requested
    else:
        home_executable = executable_from_home(
            os.environ.get(TES_HOME_ENV),
            home_variable=TES_HOME_ENV,
            relative_path=TES_HOME_EXECUTABLE,
            code_name="TES/RTES",
        )
        exe = home_executable or (
            Path(os.environ[TES_COMPATIBILITY_ENV]).expanduser()
            if os.environ.get(TES_COMPATIBILITY_ENV)
            else None
        )
    if not exe:
        raise ValueError(
            missing_home_message(
                home_variable=TES_HOME_ENV,
                relative_path=TES_HOME_EXECUTABLE,
                code_name="TES/RTES",
                compatibility_variables=(TES_COMPATIBILITY_ENV,),
            )
        )
    if not exe.is_file():
        raise FileNotFoundError(f"rtes binary not found: {exe}")
    if not is_executable(exe):
        raise PermissionError(f"rtes binary is not executable: {exe}")
    return str(exe)


def run_tes(inputs: TESInputs, config: TESConfig) -> TESResult:
    """Execute ``rtes`` with prepared inputs and collect produced outputs.

    ``rtes`` derives its output filenames (g-file, a-file, ``.RESULT`` ...) from
    SHOT/CTIME inside the input file and writes them to the working directory.
    """
    exe = _resolve_executable(config)

    cmd = [exe]
    if config.niter and config.niter > 0:
        cmd.append(f"-f{int(config.niter)}")
    cmd.append(str(inputs.cinput.name))
    if config.restart:
        cmd.append(f"-r{config.restart}")

    execution = resolve_backend(config).run(
        ExecutionRequest(
            command=tuple(cmd),
            workdir=Path(inputs.workdir),
            env=dict(config.env),
            timeout=config.timeout,
            label="rtes",
        )
    )
    if execution.timed_out:
        result = collect_tes_outputs(inputs.workdir, config)
        result.returncode = 124
        timeout_msg = f"rtes timed out after {config.timeout} seconds"
        result.stdout = execution.stdout
        result.stderr = (
            f"{execution.stderr}\n{timeout_msg}" if execution.stderr else timeout_msg
        )
        return result

    result = collect_tes_outputs(inputs.workdir, config)
    result.returncode = execution.returncode
    result.stdout = execution.stdout
    result.stderr = execution.stderr
    return result
