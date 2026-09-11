"""Executable discovery and subprocess environment for the GACODE suite.

Solver-agnostic: everything here is true of NEO, TGLF and CGYRO alike, so a
second backend reuses it unchanged.

Two things about GACODE's launchers are worth stating because both fail as
something else:

* ``<code>/bin/<code>`` is a shell script that shells out to
  ``<code>_parse.py``, which imports ``gacodeinput`` from ``f2py/pygacode``.
  When that import fails the launcher does **not** stop -- it carries on, and
  the Fortran binary then aborts on a missing ``./input.<code>.gen``, which
  points at the wrong thing entirely.  :func:`gacode_environment` therefore
  always puts ``pygacode`` on ``PYTHONPATH``.
* The launcher execs ``platform/exec/exec.$GACODE_PLATFORM``.  An unset or
  wrong ``GACODE_PLATFORM`` produces a shell error naming a path, not the
  variable, so :func:`gacode_platform` resolves it before anything is launched
  and lists the platforms the installation actually carries.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Sequence

from .._executables import (
    ExecutableNotLaunchable,
    executable_from_home,
    missing_home_message,
)
from ._types import (
    GACODE_COMPATIBILITY_ENVS,
    GACODE_HOME_ENV,
    GACODE_PLATFORM_ENV,
    GACODE_ROOT_ENV,
    GACODEConfig,
    SUITE_CODES,
)


def _validated_code(code: str) -> str:
    """Normalise a suite-member name, refusing anything not in the suite."""
    name = str(code).strip().lower()
    if name not in SUITE_CODES:
        raise ValueError(
            f"{code!r} is not a GACODE suite member; expected one of "
            f"{', '.join(SUITE_CODES)}."
        )
    return name


def launcher_relative_path(code: str) -> Path:
    """Where a suite member's launcher sits beneath the installation root.

    Not ``bin/<code>``: GACODE gives every suite member its own ``bin``, so the
    path is ``<code>/bin/<code>``.
    """
    name = _validated_code(code)
    return Path(name) / "bin" / name


def gacode_home(config: GACODEConfig | None = None) -> Path | None:
    """Resolve the installation root, or ``None`` when nothing is configured.

    Order: the config, then ``$GACODEHOME``, then ``$GACODE_ROOT``.  An
    unconfigured installation is not an error here -- it becomes one only when
    something is actually run, which is what keeps ``import vaft.code.gacode``
    working with GACODE absent.
    """
    if config is not None and config.home and str(config.home).strip():
        return Path(str(config.home)).expanduser()
    for variable in (GACODE_HOME_ENV, *GACODE_COMPATIBILITY_ENVS):
        value = os.environ.get(variable)
        if value and value.strip():
            return Path(value).expanduser()
    return None


def available_platforms(home: Path) -> tuple[str, ...]:
    """Platform tags this installation carries, from ``platform/build``."""
    build = Path(home) / "platform" / "build"
    if not build.is_dir():
        return ()
    prefix = "make.inc."
    return tuple(
        sorted(
            entry.name[len(prefix) :]
            for entry in build.iterdir()
            if entry.is_file() and entry.name.startswith(prefix)
        )
    )


def gacode_platform(config: GACODEConfig | None = None, *, home: Path | None = None) -> str:
    """Resolve the platform tag, refusing to guess one.

    Raises
    ------
    ValueError
        Nothing configured it, or it names a platform this installation does
        not carry.  Both messages list what is available.
    """
    root = home if home is not None else gacode_home(config)
    provided = None
    if config is not None and config.platform and str(config.platform).strip():
        provided = str(config.platform).strip()
    else:
        value = os.environ.get(GACODE_PLATFORM_ENV)
        if value and value.strip():
            provided = value.strip()

    known = available_platforms(root) if root is not None else ()
    if provided is None:
        listed = f" This installation provides: {', '.join(known)}." if known else ""
        raise ValueError(
            f"GACODE platform is not configured: set ${GACODE_PLATFORM_ENV}, or "
            f"pass platform= on the config. It selects "
            f"platform/exec/exec.$GACODE_PLATFORM, which the launcher execs."
            f"{listed}"
        )
    if known and provided not in known:
        raise ValueError(
            f"GACODE platform {provided!r} is not built in this installation. "
            f"Available: {', '.join(known)}."
        )
    return provided


def find_gacode_executable(
    config: GACODEConfig | None = None, code: str = "neo"
) -> Path | None:
    """Resolve one suite member's launcher, or ``None`` when unconfigured.

    Raises ``FileNotFoundError`` when the root is set but the launcher is not
    there, and ``PermissionError`` when it is there but is not a program --
    the same three-way split every other adapter in :mod:`vaft.code` makes.
    """
    name = _validated_code(code)
    if config is not None and config.executable and str(config.executable).strip():
        return Path(str(config.executable)).expanduser()
    return executable_from_home(
        gacode_home(config),
        home_variable=GACODE_HOME_ENV,
        relative_path=launcher_relative_path(name),
        code_name=f"GACODE suite ({name})",
    )


def require_gacode_executable(
    config: GACODEConfig | None = None, code: str = "neo"
) -> Path:
    """Resolve one launcher, raising an actionable error when unconfigured."""
    name = _validated_code(code)
    executable = find_gacode_executable(config, name)
    if executable is None:
        raise FileNotFoundError(
            missing_home_message(
                home_variable=GACODE_HOME_ENV,
                relative_path=launcher_relative_path(name),
                code_name=f"GACODE suite ({name})",
                compatibility_variables=GACODE_COMPATIBILITY_ENVS,
            )
        )
    return executable


def gacode_environment(config: GACODEConfig | None = None, code: str = "neo") -> dict[str, str]:
    """Build the environment a GACODE launcher needs.

    ``GACODE_ROOT`` and ``GACODE_PLATFORM`` are *set* from the VAFT-side root
    rather than replaced in meaning, and ``PATH`` and ``PYTHONPATH`` are
    prefixed rather than overwritten, so a caller who already sourced
    ``gacode_setup`` sees no change.  Anything in ``config.env`` wins, last.
    """
    name = _validated_code(code)
    environment = dict(os.environ)
    home = gacode_home(config)
    if home is not None:
        root = str(home)
        environment[GACODE_HOME_ENV] = root
        environment[GACODE_ROOT_ENV] = root
        environment[GACODE_PLATFORM_ENV] = gacode_platform(config, home=home)
        environment["PATH"] = os.pathsep.join(
            [
                str(home / "shared" / "bin"),
                str(home / name / "bin"),
                environment.get("PATH", ""),
            ]
        ).rstrip(os.pathsep)
        # Without this the launcher's parse step fails silently; see the module
        # docstring.
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(home / "f2py"),
                str(home / "f2py" / "pygacode"),
                environment.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)
    if config is not None:
        environment.update({str(k): str(v) for k, v in config.env.items()})
    return environment


def run_gacode(
    executable: Path,
    arguments: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    config: GACODEConfig | None = None,
    code: str = "neo",
) -> tuple[int, Path]:
    """Run a GACODE launcher, capturing merged stdout and stderr to a log.

    Returns the exit status and the log path.  A non-zero status is returned,
    not raised: whether it is fatal is the backend's judgement, and NEO in
    particular writes useful diagnostics alongside a failure.
    """
    command = [str(executable), *[str(argument) for argument in arguments]]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Opened outside the try so a bad log path stays its own error rather than
    # being reported as an unlaunchable solver.
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=gacode_environment(config, code),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=None if config is None else config.timeout,
                check=False,
            )
        except OSError as error:
            raise ExecutableNotLaunchable(
                f"cannot launch {executable}: {error}"
            ) from error
    return int(completed.returncode), log_path
