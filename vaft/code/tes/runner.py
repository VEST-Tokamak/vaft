"""Run the TES (``rtes``) binary on prepared inputs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ...compat import is_executable
from .._executables import executable_from_home, missing_home_message
from .config import TESConfig, TESInputs, TESResult
from .outputs import collect_tes_outputs

TES_HOME_ENV = "TESHOME"
TES_HOME_EXECUTABLE = Path("bin/rtes")
TES_COMPATIBILITY_ENV = "RTES"


def _resolve_executable(config: TESConfig) -> str:
    if config.executable:
        exe = Path(config.executable).expanduser()
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

    env = os.environ.copy()
    env.update(dict(config.env))

    completed = subprocess.run(
        cmd,
        cwd=str(inputs.workdir),
        env=env,
        text=True,
        capture_output=True,
        timeout=config.timeout,
        check=False,
    )

    result = collect_tes_outputs(inputs.workdir, config)
    result.returncode = completed.returncode
    result.stdout = completed.stdout
    result.stderr = completed.stderr
    return result
