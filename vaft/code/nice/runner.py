"""Standalone NICE process execution with reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ...compat import is_executable, resolve_executable
from .._executables import (
    ExecutableNotLaunchable,
    executable_from_home,
    missing_home_message,
)
from ..execution import ExecutionRequest, resolve_backend
from .config import NiceConfig, NiceInputs, NiceResult
from .outputs import collect_nice_outputs


NICE_HOME_ENV = "NICEHOME"
#: Where a NICE CMake build leaves ``nice_recon``, most usual first; the
#: upstream project has no install step, so ``$NICEHOME`` is the source tree.
NICE_HOME_LAYOUTS = (Path("build/nice_recon"), Path("run/nice_recon"), Path("nice_recon"))


def resolve_nice_executable(config: NiceConfig) -> Path:
    if config.executable:
        path = Path(config.executable).expanduser()
        resolved = resolve_executable(path)
        if resolved is None:
            raise FileNotFoundError(f"NICE executable not found: {path}")
        if not is_executable(resolved):
            raise PermissionError(f"NICE executable is not executable: {resolved}")
        return resolved
    root = config.nice_home or {**os.environ, **dict(config.env)}.get(NICE_HOME_ENV)
    if not root or not str(root).strip():
        raise FileNotFoundError(
            missing_home_message(
                home_variable=NICE_HOME_ENV,
                relative_path=NICE_HOME_LAYOUTS[0],
                code_name="NICE",
            )
        )
    root = Path(root).expanduser()
    layout = next(
        (rel for rel in NICE_HOME_LAYOUTS if resolve_executable(root / rel) is not None),
        NICE_HOME_LAYOUTS[0],
    )
    # Raises the same FileNotFoundError/PermissionError every adapter does.
    return executable_from_home(
        root, home_variable=NICE_HOME_ENV, relative_path=layout, code_name="NICE"
    )


def run_nice(inputs: NiceInputs, config: NiceConfig) -> NiceResult:
    """Execute NICE and then collect its native outputs.

    The launch goes through ``config.backend`` (see :mod:`vaft.code.execution`),
    so it can run as a Slurm job as well as locally. Both streams are captured
    and written to ``nice.stdout.log``/``nice.stderr.log`` once NICE exits or
    times out: the logs do not grow while NICE runs, and they hold the output
    decoded as UTF-8 (undecodable bytes replaced) rather than a byte copy.
    """
    families = {d.family for d in inputs.diagnostics if d.enabled}
    if not {"bpol_probe", "flux_loop"}.issubset(families):
        return NiceResult(
            None,
            inputs.workdir,
            converged=False,
            scientifically_usable=False,
            termination_reason="unsupported standalone VacTH family: requires B-pol and flux loops",
            provenance=inputs.manifest,
        )
    exe = resolve_nice_executable(config)
    if not (inputs.input_dir / "param.xml").is_file():
        raise ValueError(
            "NICE execution requires NiceConfig.parameter_file (input/param.xml is absent)"
        )
    backend = resolve_backend(config)
    stdout_file = inputs.workdir / "nice.stdout.log"
    stderr_file = inputs.workdir / "nice.stderr.log"
    # Truncated before the launch, as when the streams were written directly:
    # a run that never starts must not leave an earlier run's logs behind.
    stdout_file.write_text("", encoding="utf-8")
    stderr_file.write_text("", encoding="utf-8")
    try:
        execution = backend.run(
            ExecutionRequest(
                command=(str(exe), *config.arguments),
                workdir=Path(inputs.workdir),
                env=dict(config.env),
                timeout=config.timeout,
                label="nice",
            )
        )
    except ExecutableNotLaunchable as error:
        # An absent or non-program executable keeps its own type.
        if isinstance(error.__cause__, (FileNotFoundError, PermissionError)):
            raise error.__cause__ from None
        raise
    # Written before the timeout is acted on, so a killed run keeps what it
    # printed; the backend decodes NICE's bytes as UTF-8 with replacement.
    stdout, stderr = execution.stdout, execution.stderr
    stdout_file.write_text(stdout, encoding="utf-8")
    stderr_file.write_text(stderr, encoding="utf-8")
    timed_out = execution.timed_out
    returncode = 124 if timed_out else int(execution.returncode)
    manifest = dict(inputs.manifest)
    manifest["nice_executable"] = str(exe)
    manifest["nice_executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    manifest["process_returncode"] = returncode
    manifest["process_timed_out"] = timed_out
    inputs.manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    result = collect_nice_outputs(inputs.workdir, config)
    result.returncode = returncode
    result.process_succeeded = returncode == 0
    result.stdout, result.stderr = stdout, stderr
    if returncode != 0:
        result.converged = False
        result.scientifically_usable = False
        result.termination_reason = (
            "NICE timed out" if timed_out else f"NICE exited with status {returncode}"
        )
    return result
