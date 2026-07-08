"""Run the TES (``rtes``) binary on prepared inputs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import TESConfig, TESInputs, TESResult
from .outputs import collect_tes_outputs


def _resolve_executable(config: TESConfig) -> str:
    exe = config.executable or os.environ.get("RTES")
    if not exe:
        raise ValueError(
            "TESConfig.executable is required to run TES "
            "(or set the RTES environment variable to the rtes binary path)."
        )
    if not Path(exe).expanduser().exists():
        raise FileNotFoundError(f"rtes binary not found: {exe}")
    return str(Path(exe).expanduser())


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
