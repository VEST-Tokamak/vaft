"""Executable discovery and process execution for NUBEAM."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Optional

from vaft.code._executables import executable_from_home, missing_home_message

from .config import (
    NUBEAM_GENERATOR_EXECUTABLE,
    NUBEAM_HOME_ENV,
    NUBEAM_HOME_EXECUTABLE,
    NUBEAM_UPDATE_STATE_EXECUTABLE,
    NUBEAMConfig,
)
from .inputs import NUBEAMInputs, prepare_nubeam_inputs
from .outputs import NUBEAMResult, collect_nubeam_outputs

#: NUBEAM reports success on stdout rather than through its exit status in
#: some paths, so these are checked as well as the return code.
INIT_SUCCESS = "nubeam INIT completed:  normal exit."
STEP_SUCCESS = "nubeam STEP completed:  normal exit."


class NUBEAMExecutionError(RuntimeError):
    """Raised when a NUBEAM stage does not complete normally."""


def nubeam_home() -> Optional[Path]:
    """The configured NUBEAM installation root, if any."""
    raw = os.environ.get(NUBEAM_HOME_ENV)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def _resolve(
    override: Optional[str], relative: Path, code_name: str
) -> Optional[Path]:
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"{code_name} executable not found: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise PermissionError(f"{code_name} executable is not executable: {candidate}")
        return candidate
    return executable_from_home(
        os.environ.get(NUBEAM_HOME_ENV),
        home_variable=NUBEAM_HOME_ENV,
        relative_path=relative,
        code_name=code_name,
    )


def find_nubeam_executable(config: Optional[NUBEAMConfig] = None) -> Optional[Path]:
    """Resolve ``nubeam_comp_exec`` from explicit config or ``$NUBEAMHOME``."""
    config = config or NUBEAMConfig()
    return _resolve(config.executable, NUBEAM_HOME_EXECUTABLE, "NUBEAM")


def find_plasma_state_generator(
    config: Optional[NUBEAMConfig] = None,
) -> Optional[Path]:
    """Resolve the Plasma State generator ``plasma_state_test``."""
    config = config or NUBEAMConfig()
    return _resolve(
        config.generator_executable,
        NUBEAM_GENERATOR_EXECUTABLE,
        "NUBEAM Plasma State generator",
    )


def find_update_state_executable(
    config: Optional[NUBEAMConfig] = None,
) -> Optional[Path]:
    """Resolve ``update_state``, which merges NUBEAM's changes into a state."""
    config = config or NUBEAMConfig()
    return _resolve(
        config.update_state_executable,
        NUBEAM_UPDATE_STATE_EXECUTABLE,
        "NUBEAM update_state",
    )


def _require(executable: Optional[Path], relative: Path, code_name: str) -> Path:
    if executable is None:
        raise FileNotFoundError(
            missing_home_message(
                home_variable=NUBEAM_HOME_ENV,
                relative_path=relative,
                code_name=code_name,
            )
        )
    return executable


def _reaction_database_env(config: NUBEAMConfig) -> dict[str, str]:
    """``PREACTDIR`` and ``ADASDIR``, both of which NUBEAM requires.

    ``nubeam_comp_exec`` calls ``my_bad_exit`` when either resolves to a blank
    string, so an unset variable aborts the run with only
    ``failed to translate PREACTDIR environment variable`` to go on. Resolve
    them here and say what is actually missing.
    """
    home = nubeam_home()
    preact = config.preact_dir or (home / "share" / "preact" if home else None)
    adas = config.adas_dir or (home / "share" / "adas" if home else None)

    missing = [
        name
        for name, value in (("PREACTDIR", preact), ("ADASDIR", adas))
        if value is None
    ]
    if missing:
        raise FileNotFoundError(
            f"NUBEAM requires {' and '.join(missing)} and aborts without them. "
            f"Set ${NUBEAM_HOME_ENV} so they default to its share/ directories, "
            "or set them on NUBEAMConfig."
        )

    for name, value in (("PREACTDIR", preact), ("ADASDIR", adas)):
        path = Path(value).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"{name} does not exist: {path}. Run external/nubeam/macos.sh to "
                "create and populate the reaction databases."
            )
    # Both are caches: the table code writes newly computed reaction tables
    # back into them the first time a reaction is needed.
    if not os.access(Path(preact), os.W_OK):
        raise PermissionError(f"PREACTDIR must be writable: {preact}")

    return {"PREACTDIR": str(Path(preact)), "ADASDIR": str(Path(adas))}


def _run(
    executable: Path,
    *,
    workdir: Path,
    env: dict[str, str],
    log_stem: str,
    config: NUBEAMConfig,
) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    merged.update(env)
    merged.update(dict(config.env))
    completed = subprocess.run(
        [str(executable), *config.args],
        cwd=str(workdir),
        env=merged,
        text=True,
        capture_output=True,
        timeout=config.timeout,
        check=False,
    )
    (workdir / f"{log_stem}.log").write_text(completed.stdout or "", encoding="utf-8")
    (workdir / f"{log_stem}.err").write_text(completed.stderr or "", encoding="utf-8")
    return completed


def generate_plasma_state(
    inputs: NUBEAMInputs, config: Optional[NUBEAMConfig] = None
) -> Path:
    """Run ``plasma_state_test`` to build the Plasma State NUBEAM will read."""
    config = config or NUBEAMConfig()
    executable = _require(
        find_plasma_state_generator(config),
        NUBEAM_GENERATOR_EXECUTABLE,
        "NUBEAM Plasma State generator",
    )

    completed = _run(
        executable,
        workdir=inputs.workdir,
        env={},
        log_stem="plasma_state_test",
        config=config,
    )

    # The generator reports its own errors on stdout and still exits 0 -- a
    # missing G-EQDSK produces "?geq_init: file ... does not exist" and a zero
    # status -- so the output file is the only trustworthy success signal.
    state = inputs.plasma_state
    if state is None or not state.is_file() or state.stat().st_size == 0:
        raise NUBEAMExecutionError(
            f"plasma_state_test did not create {state}; "
            f"exit status was {completed.returncode}. See "
            f"{inputs.workdir / 'plasma_state_test.log'}"
        )
    return state


def run_nubeam(
    inputs: NUBEAMInputs, config: Optional[NUBEAMConfig] = None
) -> NUBEAMResult:
    """Run NUBEAM INIT then STEP against a staged Plasma State."""
    config = config or NUBEAMConfig()
    executable = _require(
        find_nubeam_executable(config), NUBEAM_HOME_EXECUTABLE, "NUBEAM"
    )

    base = _reaction_database_env(config)
    base["NUBEAM_WORKPATH"] = str(inputs.workdir)

    # init_hold, not init: it holds the RNG seed at the namelist's `nseed`
    # instead of reseeding from the system clock. Without it no two runs are
    # comparable, which makes every downstream regression check meaningless.
    init_env = dict(base, NUBEAM_ACTION="init_hold")
    if config.frantic:
        init_env["FRANTIC_INIT"] = str(config.frantic_zones)
    init = _run(
        executable, workdir=inputs.workdir, env=init_env, log_stem="init", config=config
    )
    init_log = (inputs.workdir / "init.log").read_text(encoding="utf-8")
    if init.returncode != 0 or INIT_SUCCESS not in init_log:
        raise NUBEAMExecutionError(
            f"NUBEAM INIT did not complete normally (exit {init.returncode}); "
            f"see {inputs.workdir / 'init.log'}"
        )

    step_env = dict(
        base,
        NUBEAM_ACTION="step",
        NUBEAM_REPEAT_COUNT=config.repeat_count,
        NUBEAM_POSTPROC=config.postproc,
    )
    if config.frantic:
        step_env["FRANTIC_ACTION"] = "execute"
    step = _run(
        executable, workdir=inputs.workdir, env=step_env, log_stem="step", config=config
    )
    step_log = (inputs.workdir / "step.log").read_text(encoding="utf-8")
    if step.returncode != 0 or STEP_SUCCESS not in step_log:
        raise NUBEAMExecutionError(
            f"NUBEAM STEP did not complete normally (exit {step.returncode}); "
            f"see {inputs.workdir / 'step.log'}"
        )

    return collect_nubeam_outputs(inputs.workdir, config=config, returncode=0)


def run_nubeam_case(
    input_dir: str | Path,
    *,
    gfile: str | Path,
    workdir: str | Path,
    config: Optional[NUBEAMConfig] = None,
) -> NUBEAMResult:
    """Stage a case, build its Plasma State, and run NUBEAM over it.

    The one-call path, equivalent to ``vaft.code.chease.refine_equilibrium``.
    *workdir* is required rather than defaulted to a temporary directory: the
    run leaves several hundred megabytes of particle state behind that a
    caller usually wants to inspect. For a throwaway run, wrap this in
    :func:`vaft.compat.short_temporary_directory` with
    ``max_length=config.workdir_budget`` -- which also guarantees the path fits
    NUBEAM's fixed-width buffer.
    """
    config = config or NUBEAMConfig()
    inputs = prepare_nubeam_inputs(
        input_dir, gfile=gfile, workdir=workdir, config=config
    )
    generate_plasma_state(inputs, config)
    return run_nubeam(inputs, config)
