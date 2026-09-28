"""Run GENRAY on a prepared case and map its rays into ``waves``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...compat import is_executable, resolve_executable
from .._executables import executable_from_home, missing_home_message
from ..execution import ExecutionRequest, resolve_backend
from .config import GENRAY_HOME_ENV, GENRAY_HOME_EXECUTABLE, GENRAYConfig, GENRAYInputs, GENRAYResult
from .inputs import prepare_genray_inputs
from .outputs import collect_genray_outputs, genray_to_waves, read_genray_netcdf


def find_genray_executable(config: GENRAYConfig | None = None) -> Path:
    """``config.executable`` if given, else ``$GENRAYHOME/bin/xgenray``."""
    if config is not None and config.executable:
        requested = Path(config.executable).expanduser()
        executable = resolve_executable(requested) or requested
        if not executable.is_file():
            raise FileNotFoundError(f"GENRAY executable not found: {executable}")
        if not is_executable(executable):
            raise PermissionError(f"GENRAY executable is not executable: {executable}")
        return executable
    executable = executable_from_home(
        os.environ.get(GENRAY_HOME_ENV),
        home_variable=GENRAY_HOME_ENV,
        relative_path=GENRAY_HOME_EXECUTABLE,
        code_name="GENRAY",
    )
    if executable is None:
        raise FileNotFoundError(
            missing_home_message(
                home_variable=GENRAY_HOME_ENV,
                relative_path=GENRAY_HOME_EXECUTABLE,
                code_name="GENRAY",
            )
            + " Build it with install/install_genray.sh."
        )
    return executable


def run_genray(inputs: GENRAYInputs, config: GENRAYConfig) -> GENRAYResult:
    """Execute GENRAY in ``inputs.workdir`` and parse ``genray.nc`` if it was written.

    GENRAY exits 0 on some input errors, so success is the return code *and*
    a non-empty ``genray.nc``; :attr:`GENRAYResult.ok` checks both.
    """
    executable = find_genray_executable(config)
    stale = collect_genray_outputs(inputs.workdir)
    if stale is not None:
        stale.unlink()  # never report a previous run's rays as this run's
    execution = resolve_backend(config).run(
        ExecutionRequest(
            command=(str(executable),),
            workdir=Path(inputs.workdir),
            env=dict(config.env),
            timeout=config.timeout,
            label="genray",
        )
    )
    returncode = 124 if execution.timed_out else execution.returncode
    netcdf = collect_genray_outputs(inputs.workdir)
    parsed = None
    if netcdf is not None and returncode == 0:
        try:
            parsed = read_genray_netcdf(netcdf)
        except (KeyError, OSError, ValueError, IndexError) as error:
            # A writer that stopped part-way leaves a file without the ray arrays.
            parsed = {"complete": False, "parse_error": f"{type(error).__name__}: {error}"}
    provenance = dict(inputs.provenance)
    provenance["executable"] = str(executable)
    return GENRAYResult(
        returncode=returncode,
        workdir=Path(inputs.workdir),
        stdout=execution.stdout,
        stderr=execution.stderr,
        netcdf=netcdf,
        parsed=parsed,
        provenance=provenance,
    )


def run(ods: Any, config: GENRAYConfig, *, workdir: str | Path | None = None, output: Any = None) -> GENRAYResult:
    """Prepare, run and map one EC case: ``ec_launchers + equilibrium + core_profiles -> waves``.

    The ``waves`` IDS is written into ``output`` (default: ``ods``) only when
    the run succeeded (:attr:`GENRAYResult.ok`); a failed run leaves it
    untouched and the result carries the return code, GENRAY's output and the
    parsed stop reasons for diagnosis. Each launcher beam fills
    ``coherent_wave[beam_index]``, replacing that entry whole.
    """
    inputs = prepare_genray_inputs(ods, config, workdir)
    result = run_genray(inputs, config)
    if result.ok:
        launcher = inputs.provenance["launcher"]
        genray_to_waves(
            result.parsed,
            ods if output is None else output,
            time=float(config.time),
            beam_name=launcher.get("name") or launcher.get("identifier") or "",
            provenance=result.provenance,
            coherent_wave_index=int(config.beam_index),
        )
    return result


__all__ = ["find_genray_executable", "run", "run_genray"]
