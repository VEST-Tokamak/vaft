"""Run the TES (``rtes``) binary on prepared inputs."""

from __future__ import annotations

import os
from pathlib import Path

from ...compat import is_executable, resolve_executable
from .._executables import executable_from_home, missing_home_message
from ..execution import ExecutionRequest, resolve_backend, timeout_reason
from .config import TESConfig, TESInputs, TESResult
from .outputs import collect_tes_outputs, snapshot_outputs

TES_HOME_ENV = "TESHOME"
TES_HOME_EXECUTABLE = Path("bin/rtes")
# TES's own Makefile builds rtes inside its source tree, at TES/rtes, so a
# $TESHOME pointing at an unmodified build has no bin/ directory.
TES_SOURCE_TREE_EXECUTABLE = Path("TES/rtes")
TES_COMPATIBILITY_ENV = "RTES"


def _executable_under_tes_home() -> Path | None:
    """rtes under $TESHOME: bin/rtes, else the source-tree build TES/rtes.

    Returns None when $TESHOME is unset or holds neither, so an explicit $RTES
    can still apply; a file present but not executable raises.
    """
    home = os.environ.get(TES_HOME_ENV)
    if not home or not home.strip():
        return None
    root = Path(home).expanduser()
    for relative in (TES_HOME_EXECUTABLE, TES_SOURCE_TREE_EXECUTABLE):
        if resolve_executable(root / relative) is not None:
            return executable_from_home(
                root, home_variable=TES_HOME_ENV, relative_path=relative, code_name="TES/RTES"
            )
    return None


def _resolve_executable(config: TESConfig) -> str:
    """The rtes to launch: config.executable, $TESHOME (bin/rtes or TES/rtes), then $RTES."""
    if config.executable:
        requested = Path(config.executable).expanduser()
        exe = resolve_executable(requested) or requested
    else:
        exe = _executable_under_tes_home()
        if exe is None and os.environ.get(TES_COMPATIBILITY_ENV):
            exe = Path(os.environ[TES_COMPATIBILITY_ENV]).expanduser()
        home = os.environ.get(TES_HOME_ENV)
        if exe is None and home and home.strip():
            root = Path(home).expanduser()
            raise FileNotFoundError(
                f"TES/RTES executable is missing for ${TES_HOME_ENV}={root}: expected "
                f"{root / TES_HOME_EXECUTABLE} or {root / TES_SOURCE_TREE_EXECUTABLE}. "
                "Compile or install TES/RTES so that the executable exists at one of "
                f"these locations, or point ${TES_COMPATIBILITY_ENV} at it."
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

    # rtes writes no g-file when it fails; in a reused directory an earlier
    # run's files must not be reported as this run's equilibrium
    before = snapshot_outputs(inputs.workdir)
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
        result = collect_tes_outputs(inputs.workdir, config, before=before)
        result.returncode = None
        reason = timeout_reason("rtes", execution, config.timeout)
        result.stdout = execution.stdout
        result.stderr = f"{execution.stderr}\n{reason}" if execution.stderr else reason
        result.runtime_status = execution.runtime_status
        result.elapsed_s = execution.elapsed_s
        return result

    result = collect_tes_outputs(inputs.workdir, config, before=before)
    result.returncode = execution.returncode
    result.stdout = execution.stdout
    result.stderr = execution.stderr
    result.elapsed_s = execution.elapsed_s
    if result.returncode == 0 and result.gfile is None:
        # rtes exits 0 when the Picard loop diverges ("(r,z) is out of range")
        # and simply writes no equilibrium; without a g-file there is no result.
        result.returncode = 1
        note = "rtes exited 0 but wrote no g-file (the solve did not converge; see tes.log)"
        result.stderr = f"{result.stderr}\n{note}" if result.stderr else note
    return result


__all__ = [
    "TES_COMPATIBILITY_ENV",
    "TES_HOME_ENV",
    "TES_HOME_EXECUTABLE",
    "run_tes",
]
