"""Run the TES (``rtes``) binary on prepared inputs."""

from __future__ import annotations

import os
from pathlib import Path

from ...compat import is_executable, resolve_executable
from .._executables import executable_from_home, missing_home_message
from ..execution import ExecutionRequest, resolve_backend, timeout_reason
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
        # returncode=None and runtime_status="timeout" (#1016; it was 124).
        result = collect_tes_outputs(inputs.workdir, config)
        result.returncode = None
        reason = timeout_reason("rtes", execution, config.timeout)
        result.stdout = execution.stdout
        result.stderr = f"{execution.stderr}\n{reason}" if execution.stderr else reason
        result.runtime_status = execution.runtime_status
        result.elapsed_s = execution.elapsed_s
        return result

    result = collect_tes_outputs(inputs.workdir, config)
    result.returncode = execution.returncode
    result.stdout = execution.stdout
    result.stderr = execution.stderr
    result.elapsed_s = execution.elapsed_s
    return result


__all__ = [
    "TES_COMPATIBILITY_ENV",
    "TES_HOME_ENV",
    "TES_HOME_EXECUTABLE",
    "run_tes",
]
