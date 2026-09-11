"""Execute NEO and collect its native result.

VAFT drives ``<GACODEHOME>/neo/bin/neo``, the launcher, rather than the Fortran
binary underneath it.  The launcher expands ``input.neo`` into the
``input.neo.gen`` the binary actually reads, and stamps ``out.neo.version`` with
the revision, platform and date -- which is where the run's executable identity
comes from.  Calling the binary directly would skip both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .._runtime import gacode_platform, require_gacode_executable
from .._runtime import run_gacode
from .._profiles import GACODEProfile
from ._types import NEOConfig, NEOResult
from .inputs import NEOInputs, prepare_neo_case
from .outputs import NeoOutputs, collect_neo_outputs


class NEOExecutionError(RuntimeError):
    """NEO ran and did not produce a usable result.

    Separate from the runtime's ``FileNotFoundError`` and
    ``ExecutableNotLaunchable``, which mean it never started.
    """


def run_neo(
    inputs: NEOInputs,
    config: Optional[NEOConfig] = None,
    *,
    check: bool = True,
) -> NEOResult:
    """Run NEO on an already-staged case and parse everything it wrote.

    Parameters
    ----------
    inputs
        A staged case from :func:`prepare_neo_case`.
    config
        Runtime settings.  Only the GACODE-side fields matter here; the
        numerical ones were baked into ``input.neo`` at staging.
    check
        Raise :class:`NEOExecutionError` on a failed run.  With ``check=False``
        the failure is returned instead, which is what a scan wants: the log and
        whatever NEO managed to write are still on the result.

    Raises
    ------
    FileNotFoundError
        GACODE is not configured, or the launcher is missing.
    NEOExecutionError
        NEO exited non-zero, or exited zero having written nothing usable.
    """
    configuration = config or NEOConfig()
    executable = require_gacode_executable(configuration, "neo")
    # Resolved before launching: a wrong platform otherwise fails inside a shell
    # script without naming itself.
    platform = gacode_platform(configuration)

    workdir = Path(inputs.workdir)
    # Whatever an earlier run left here is not this run's result. Parsing is by
    # file name, so a rerun that fails early would otherwise return the previous
    # run's physics as its own; only NEO's products are removed.
    for stale in workdir.glob("out.neo.*"):
        if stale.is_file():
            stale.unlink()

    # The launcher joins its -e argument onto $PWD, so it is run from the parent
    # with the case named relatively.
    returncode, log = run_gacode(
        executable,
        ["-e", workdir.name, "-n", str(int(configuration.n_mpi)),
         "-nomp", str(int(configuration.n_omp))],
        cwd=workdir.parent,
        log_path=workdir / "neo.log",
        config=configuration,
        code="neo",
    )
    native = collect_neo_outputs(workdir)
    result = NEOResult(
        returncode=returncode,
        workdir=workdir,
        logs=(log,),
        outputs={"native": tuple(sorted(workdir.glob("out.neo.*")))},
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
        raise NEOExecutionError(_failure_message(result, log))
    return result


def _failure_message(result: NEOResult, log: Path) -> str:
    tail = ""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-12:])
    except OSError:
        pass
    native = result.outputs_native
    if result.returncode != 0:
        reason = f"NEO exited with status {result.returncode}"
    elif native is not None and native.errors:
        reason = "NEO rejected the case: " + "; ".join(native.errors)
    elif native is not None and native.transport is not None:
        reason = (
            "NEO completed but its drift-kinetic current is not finite, which is what "
            "a degenerate geometry produces"
        )
    else:
        reason = (
            "NEO exited cleanly but wrote no readable output, which is what a "
            "failed parse or an aborted solve looks like"
        )
    return f"{reason}. Working directory: {result.workdir}\n{tail}"


def run_neo_case(
    profile: GACODEProfile,
    workdir: str | Path,
    config: Optional[NEOConfig] = None,
    *,
    check: bool = True,
) -> NEOResult:
    """Stage and run one NEO case: the one-call path.

    Equivalent to :func:`prepare_neo_case` followed by :func:`run_neo`, and the
    counterpart of ``vaft.code.chease.refine_equilibrium`` and
    ``vaft.code.nubeam.run_nubeam_case``.
    """
    staged = prepare_neo_case(profile, workdir, config)
    return run_neo(staged, config, check=check)


def read_neo_case(workdir: str | Path) -> Optional[NeoOutputs]:
    """Read a finished run directory without re-running it."""
    return collect_neo_outputs(workdir)
