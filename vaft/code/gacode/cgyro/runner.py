"""Execute CGYRO and collect its native result.

VAFT drives ``$GACODEHOME/cgyro/bin/cgyro``, the launcher, not ``cgyro/src/cgyro``: the
launcher runs ``cgyro_parse.py`` (which writes ``input.cgyro.gen``, the complete record of
every key) and stamps ``out.cgyro.version``. Unlike TGLF's, it accepts ``-nomp``.

**A stale restart file silently continues an old run.** CGYRO resumes from
``bin.cgyro.restart`` when it finds one, so a fresh run in a reused directory would
otherwise extend the previous simulation and report it as new. Every CGYRO product is
therefore cleared before a run unless :attr:`CGYROConfig.restart` asks to continue.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional

from .._profiles import GACODEProfile
from .._runtime import (
    gacode_home,
    gacode_platform,
    require_gacode_executable,
    run_gacode,
    stopped_reason,
)
from ._types import CGYROConfig, CGYROResult, formalism
from .inputs import CGYROInputs, input_sha256, prepare_cgyro_case
from .outputs import CgyroOutputs, collect_cgyro_outputs

__all__ = [
    "CGYROExecutionError",
    "gacode_revision",
    "read_cgyro_case",
    "run_cgyro",
    "run_cgyro_case",
]


class CGYROExecutionError(RuntimeError):
    """CGYRO ran and did not produce a usable result."""


def gacode_revision(config: Optional[CGYROConfig] = None) -> Optional[str]:
    """The GACODE source commit, from the installation's own git checkout.

    ``out.cgyro.version`` carries a version tag too, but it is whatever
    ``gacode_getversion`` printed, which is a describe string rather than a commit.
    ``None`` when the installation is not a git checkout.
    """
    home = gacode_home(config)
    if home is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(home), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _clear(workdir: Path, *, keep_restart: bool) -> None:
    """Remove the previous run's products, or nothing at all when restarting.

    CGYRO *appends* to its time records on a restart, so a restart keeps every file:
    deleting ``out.cgyro.time`` or ``bin.cgyro.ky_flux`` would leave only the
    post-restart samples and silently shorten the flux trace a saturation window reads.
    """
    if keep_restart:
        return
    for pattern in ("out.cgyro.*", "bin.cgyro.*"):
        for stale in workdir.glob(pattern):
            if stale.is_file():
                stale.unlink()


def run_cgyro(
    inputs: CGYROInputs,
    config: Optional[CGYROConfig] = None,
    *,
    check: bool = True,
    state_key: Optional[Mapping[str, Any]] = None,
) -> CGYROResult:
    """Run CGYRO on an already-staged case and parse everything it wrote.

    Parameters
    ----------
    inputs
        A staged case from :func:`~vaft.code.gacode.cgyro.inputs.prepare_cgyro_case`.
    config
        Executable, platform, MPI tasks and threads, and the execution backend: a
        :class:`~vaft.code.slurm.SlurmBackend` there runs this as a Slurm job or step,
        and a time limit comes back as ``runtime_status`` rather than an exception.
    check
        Raise :class:`CGYROExecutionError` unless :attr:`CGYROResult.ok`.
    state_key
        Recorded in the provenance; overrides the key given at staging time.
    """
    configuration = config or CGYROConfig()
    executable = require_gacode_executable(configuration, "cgyro")
    platform = gacode_platform(configuration)
    workdir = Path(inputs.workdir)
    _clear(workdir, keep_restart=bool(configuration.restart))

    log = workdir / "cgyro.log"
    run = run_gacode(
        executable,
        ["-e", workdir.name, "-n", str(int(configuration.n_mpi)),
         "-nomp", str(int(configuration.n_omp))],
        cwd=workdir.parent,
        log_path=log,
        config=configuration,
        code="cgyro",
    )
    returncode, log = run

    native = collect_cgyro_outputs(workdir)
    revision = gacode_revision(configuration)
    version = None if native is None else native.version
    staged = dict(inputs.provenance)
    key = state_key if state_key is not None else staged.get("state_key")
    result = CGYROResult(
        returncode=returncode,
        runtime_status=getattr(run, "runtime_status", "completed"),
        elapsed_s=getattr(run, "elapsed_s", None),
        workdir=workdir,
        logs=(log,),
        outputs={
            "native": tuple(sorted(
                p for pattern in ("out.cgyro.*", "bin.cgyro.*")
                for p in workdir.glob(pattern)
            )),
        },
        outputs_native=native,
        provenance={
            "executable": str(executable),
            "platform": platform,
            "gacode_commit": revision,
            "version": version,
            "input_sha256": (
                input_sha256(inputs.input_cgyro) if inputs.input_cgyro else None
            ),
            "state_key": None if key is None else dict(key),
            "resolution": configuration.resolution(),
            "n_mpi": int(configuration.n_mpi),
            "n_omp": int(configuration.n_omp),
            "formalism": formalism(
                configuration,
                solver_version=revision or (None if version is None else version.get("commit")),
            ),
            "parameters": dict(inputs.parameters),
            "inputs": staged,
        },
    )
    if check and not result.ok:
        raise CGYROExecutionError(_failure_message(result, log))
    return result


def _failure_message(result: CGYROResult, log: Path) -> str:
    native: Optional[CgyroOutputs] = result.outputs_native
    if result.timed_out:
        reason = stopped_reason(log, result.runtime_status)
    elif result.returncode != 0:
        reason = f"CGYRO exited with status {result.returncode}"
    elif native is None:
        reason = "CGYRO wrote no out.cgyro.* or bin.cgyro.* files at all"
    elif native.errors:
        reason = "CGYRO logged: " + "; ".join(native.errors)
    elif native.exit_message is None:
        reason = "CGYRO never reached its EXIT line (killed, or the kernel aborted)"
    else:
        reason = f"CGYRO finished ({native.exit_message}) with non-finite results"

    tail = ""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            tail = "\n  " + "\n  ".join(lines[-12:])
    except OSError:  # pragma: no cover - the log is written by run_gacode
        pass
    return f"{reason} (in {result.workdir}).{tail}"


def run_cgyro_case(
    profile: GACODEProfile,
    r_over_a: float,
    workdir: str | Path,
    config: Optional[CGYROConfig] = None,
    *,
    check: bool = True,
    state_key: Optional[Mapping[str, Any]] = None,
) -> CGYROResult:
    """Stage and run one CGYRO case at one flux surface: the one-call path."""
    staged = prepare_cgyro_case(profile, r_over_a, workdir, config, state_key=state_key)
    return run_cgyro(staged, config, check=check)


def read_cgyro_case(workdir: str | Path) -> Optional[CgyroOutputs]:
    """Read a finished run directory without re-running it."""
    return collect_cgyro_outputs(workdir)
