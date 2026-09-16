"""Execute TGLF and collect its native result.

VAFT drives ``$GACODEHOME/tglf/bin/tglf``, the launcher, rather than the binary beneath
it -- the launcher expands ``input.tglf`` into the ``input.tglf.gen`` the binary reads
and stamps ``out.tglf.version``, and calling the binary directly would skip both.

**The launcher takes ``-e`` and ``-n`` and nothing else.** Unlike NEO's it has no
``-nomp``; its argument parser ends in ``*) echo "ERROR: incorrect tglf syntax" ; exit
1``, so passing NEO's flag list fails before TGLF starts. ``n_omp`` on the config is
therefore honoured only through the environment, which is what TGLF's own script does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .._profiles import GACODEProfile
from .._runtime import gacode_platform, require_gacode_executable, run_gacode
from ._types import TGLFConfig, TGLFResult
from .inputs import TGLFInputs, prepare_tglf_case
from .outputs import TglfOutputs, collect_tglf_outputs

__all__ = [
    "TGLFExecutionError",
    "read_tglf_case",
    "run_tglf",
    "run_tglf_case",
]


class TGLFExecutionError(RuntimeError):
    """TGLF ran and did not produce a usable result.

    Separate from the runtime's ``FileNotFoundError`` and ``ExecutableNotLaunchable``,
    which mean it never started.
    """


def run_tglf(
    inputs: TGLFInputs,
    config: Optional[TGLFConfig] = None,
    *,
    check: bool = True,
) -> TGLFResult:
    """Run TGLF on an already-staged case and parse everything it wrote.

    Parameters
    ----------
    inputs
        A staged case from :func:`~vaft.code.gacode.tglf.inputs.prepare_tglf_case`.
    config
        Resolves the executable, the platform and the MPI task count.
    check
        Raise :class:`TGLFExecutionError` when the run did not solve. ``False`` returns
        the result and leaves the judgement to the caller, as NEO's runner does.

    Returns
    -------
    TGLFResult
    """
    configuration = config or TGLFConfig()
    executable = require_gacode_executable(configuration, "tglf")
    platform = gacode_platform(configuration)
    workdir = Path(inputs.workdir)

    # Parsing is by filename, so a rerun that fails would otherwise hand back the
    # previous run's physics.
    for stale in workdir.glob("out.tglf.*"):
        stale.unlink()

    log = workdir / "tglf.log"
    returncode, log = run_gacode(
        executable,
        ["-e", workdir.name, "-n", str(configuration.n_mpi)],
        cwd=workdir.parent,
        log_path=log,
        config=configuration,
        code="tglf",
    )

    native = collect_tglf_outputs(workdir)
    result = TGLFResult(
        returncode=returncode,
        workdir=workdir,
        logs=(log,),
        outputs={"native": tuple(sorted(workdir.glob("out.tglf.*")))},
        outputs_native=native,
        provenance={
            "executable": str(executable),
            "platform": platform,
            "parameters": dict(inputs.parameters),
            "inputs": dict(inputs.provenance),
            "version": None if native is None else native.version,
        },
    )
    if check and not result.ok:
        raise TGLFExecutionError(_failure_message(result, log))
    return result


def _failure_message(result: TGLFResult, log: Path) -> str:
    """Say which of the several ways to fail this was, then show the log tail."""
    native = result.outputs_native
    if result.returncode != 0:
        reason = f"TGLF exited with status {result.returncode}"
    elif native is None:
        reason = "TGLF wrote no out.tglf.* files at all"
    elif native.errors:
        reason = "TGLF logged: " + "; ".join(native.errors)
    elif native.gbflux is None:
        reason = (
            "TGLF exited cleanly but wrote no out.tglf.gbflux, so it produced no fluxes"
        )
    else:
        reason = "TGLF wrote fluxes that are not finite"

    tail = ""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            tail = "\n  " + "\n  ".join(lines[-12:])
    except OSError:  # pragma: no cover - the log is written by run_gacode
        pass
    return f"{reason} (in {result.workdir}).{tail}"


def run_tglf_case(
    profile: GACODEProfile,
    rho: float,
    workdir: str | Path,
    config: Optional[TGLFConfig] = None,
    *,
    check: bool = True,
) -> TGLFResult:
    """Stage and run one TGLF case at one flux surface: the one-call path.

    Equivalent to :func:`~vaft.code.gacode.tglf.inputs.prepare_tglf_case` followed by
    :func:`run_tglf`, and the counterpart of ``run_neo_case``.
    """
    staged = prepare_tglf_case(profile, rho, workdir, config)
    return run_tglf(staged, config, check=check)


def read_tglf_case(workdir: str | Path) -> Optional[TglfOutputs]:
    """Read a finished run directory without re-running it."""
    return collect_tglf_outputs(workdir)
