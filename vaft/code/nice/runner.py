"""Standalone NICE process execution with reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .config import NiceConfig, NiceInputs, NiceResult
from .outputs import collect_nice_outputs


def resolve_nice_executable(config: NiceConfig) -> Path:
    if config.executable:
        path = Path(config.executable).expanduser()
    else:
        root = config.nice_home or os.environ.get("NICEHOME")
        if not root:
            raise ValueError(
                "NICE is not configured: set NiceConfig.executable or $NICEHOME"
            )
        root = Path(root).expanduser()
        candidates = (
            root / "build" / "nice_recon",
            root / "run" / "nice_recon",
            root / "nice_recon",
        )
        path = next((p for p in candidates if p.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"NICE executable not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"NICE executable is not executable: {path}")
    return path


def run_nice(inputs: NiceInputs, config: NiceConfig) -> NiceResult:
    """Execute NICE and then collect its native outputs."""
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
    stdout_file = inputs.workdir / "nice.stdout.log"
    stderr_file = inputs.workdir / "nice.stderr.log"
    timed_out = False
    with stdout_file.open("w", encoding="utf-8") as stdout_handle, stderr_file.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            completed = subprocess.run(
                [str(exe), *config.arguments],
                cwd=inputs.workdir,
                env={**os.environ, **dict(config.env)},
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=config.timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, 124
    stdout = stdout_file.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_file.read_text(encoding="utf-8", errors="replace")
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
