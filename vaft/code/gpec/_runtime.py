"""Solver-agnostic plumbing: executable resolution, env, namelist patching, subprocess execution.

Nothing in this module knows what DCON, RDCON, STRIDE, or GPEC actually are --
that lives in :mod:`vaft.code.gpec._solvers`. This module only knows how to
find an executable under ``$GPECHOME``, build a subprocess environment, patch
a copied namelist template, and run a subprocess with a timeout.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .._executables import executable_from_home, missing_home_message
from ._types import GPEC_HOME_ENV

if TYPE_CHECKING:
    from ._types import GPECSuiteConfig


def package_vest_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "gpec"


def template_dir(config: "GPECSuiteConfig") -> Path:
    return Path(config.templates_dir).expanduser() if config.templates_dir else package_vest_dir()


def coil_data_dir(config: "GPECSuiteConfig") -> Path:
    return Path(config.coil_data_dir).expanduser() if config.coil_data_dir else package_vest_dir()


def gpec_home(config: "GPECSuiteConfig") -> Path | None:
    """Resolve the GPEC installation root from the config or ``$GPECHOME``."""
    home = config.gpec_home or os.environ.get(GPEC_HOME_ENV)
    return Path(home).expanduser() if home else None


def executable_dir(config: "GPECSuiteConfig") -> Path | None:
    if config.executable_dir:
        return Path(config.executable_dir).expanduser()
    home = gpec_home(config)
    return home / "bin" if home else None


def executable(config: "GPECSuiteConfig", program: str) -> Path | None:
    if config.executable_dir:
        return Path(config.executable_dir).expanduser() / program
    home = gpec_home(config)
    return executable_from_home(
        home,
        home_variable=GPEC_HOME_ENV,
        relative_path=Path("bin") / program,
        code_name=f"GPEC suite ({program})",
    )


def optional_executable(config: "GPECSuiteConfig", program: str) -> Path | None:
    """Return an optional companion executable without making it mandatory."""
    directory = executable_dir(config)
    return directory / program if directory is not None else None


def unconfigured_reason() -> str:
    return missing_home_message(
        home_variable=GPEC_HOME_ENV,
        relative_path="bin/{dcon,match,rdcon,rmatch,stride,gpec}",
        code_name="GPEC suite",
    )


def gpec_env(config: "GPECSuiteConfig") -> dict[str, str]:
    env = dict(os.environ)
    home = gpec_home(config)
    if home is not None:
        env[GPEC_HOME_ENV] = str(home)
    env.update({str(key): str(value) for key, value in config.env.items()})
    return env


def run_policy(config: "GPECSuiteConfig") -> str:
    text = str(config.run_mode).strip().lower()
    if text in {"", "auto", "run_if_available", "if_available"}:
        return "run_if_available"
    if text in {"prepare", "prepare_only", "skip", "false", "0", "no"}:
        return "prepare_only"
    if text in {"strict", "required", "must_run"}:
        return "strict"
    if text in {"true", "1", "yes", "run"}:
        return "run_if_available"
    raise ValueError(f"Unsupported GPEC run_mode: {config.run_mode!r}")


def time_label(time_ms: int | str | None, geqdsk: Path | None = None) -> str:
    """Directory label for one case's time point.

    Callers that only know ``time_ms`` (e.g. building result paths from a run
    manifest, with no GEQDSK on hand) can omit ``geqdsk`` entirely; it is only
    consulted as a fallback when ``time_ms`` is ``None``.
    """
    if time_ms is not None:
        text = str(time_ms)
        return text if len(text) >= 5 or not text.isdigit() else f"{int(text):05d}"
    if geqdsk is None:
        raise ValueError("time_label requires time_ms or geqdsk")
    suffix = geqdsk.name.rsplit(".", maxsplit=1)[-1]
    return suffix if suffix != geqdsk.name else geqdsk.stem


def module_dir(
    workdir: Path,
    time_ms: int | str | None,
    module: str,
    mode: int,
    *,
    geqdsk: Path | None = None,
) -> Path:
    return workdir / time_label(time_ms, geqdsk) / module / f"nn={mode}"


def _quote_namelist_string(value: str) -> str:
    r"""Quote ``value`` the way Fortran list-directed input expects.

    Deliberately not ``json.dumps``: JSON escapes a backslash as ``\\``, and a
    Fortran namelist has no escape sequences at all, so the reader takes the
    doubled separators literally. That is invisible on POSIX -- where a path
    holds no backslashes -- and silently corrupts every Windows path.

    Fortran's only in-string metacharacter is the delimiter itself, escaped by
    doubling it. The double quote is kept as that delimiter because it is what
    the packaged template and the committed shot-48226 reference already use,
    so only the escaping changes here, never the file's style.
    """
    return '"' + value.replace('"', '""') + '"'


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, str):
        return _quote_namelist_string(value)
    if isinstance(value, Path):
        return _quote_namelist_string(str(value))
    return str(value)


def _set_value(text: str, key: str, value: Any) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*)(.*)$")

    def replace(match: re.Match[str]) -> str:
        suffix = match.group(3)
        spacer = " " if suffix.startswith("!") else ""
        return f"{match.group(1)}{_format_value(value)}{spacer}{suffix}"

    new_text, count = pattern.subn(replace, text, count=1)
    if count == 0:
        raise KeyError(f"Cannot find namelist key {key!r}")
    return new_text


def write_template(
    template: Path,
    target: Path,
    replacements: Mapping[str, Any],
) -> Path:
    text = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = _set_value(text, key, value)
    target.write_text(text, encoding="utf-8")
    return target


def _extract_shot_time_from_header(header_line: str) -> tuple[int | None, float | None]:
    match = re.search(r"#\s*(\d+)\s*#\s*(\d+(?:\.\d+)?)\s*ms", header_line)
    if match:
        return int(match.group(1)), float(match.group(2))
    match = re.search(r"\b(\d{4,6})\b.*?(\d+(?:\.\d+)?)\s*ms", header_line)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None, None


def format_gfile_header_for_gpec(
    header_line: str,
    shot: int | None = None,
    time_ms: float | int | str | None = None,
) -> str:
    """Format a GEQDSK header for GPEC's fixed-column EFIT reader."""

    extracted_shot, extracted_time = _extract_shot_time_from_header(header_line)
    shot_num = int(shot if shot is not None else (extracted_shot or 0))
    try:
        time_value = float(time_ms if time_ms is not None else (extracted_time or 0.0))
    except (TypeError, ValueError):
        time_value = 0.0

    nw = 65
    nh = 65
    if len(header_line) >= 60:
        try:
            nw_candidate = header_line[52:56].strip()
            nh_candidate = header_line[56:60].strip()
            if nw_candidate:
                nw = int(nw_candidate)
            if nh_candidate:
                nh = int(nh_candidate)
        except ValueError:
            pass

    prefix = header_line[:26].ljust(26)
    if time_value == int(time_value):
        timestr = f"{int(time_value):>5d}ms"
    else:
        timestr = f"{time_value:>5.1f}ms"
    formatted = f"{prefix}{shot_num:>7d}{timestr[-7:].rjust(7)}{' ' * 12}{nw:>4d}{nh:>4d}"
    return formatted[:61].ljust(61)


def copy_gfile_for_gpec(
    source: Path,
    target: Path,
    *,
    shot: int,
    time_ms: int | str | None,
) -> Path:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"GEQDSK is empty: {source}")
    lines[0] = format_gfile_header_for_gpec(lines[0].rstrip("\n\r"), shot, time_ms) + "\n"
    target.write_text("".join(lines), encoding="utf-8")
    return target


def run_subprocess(
    executable_path: Path,
    cwd: Path,
    log_path: Path,
    *,
    config: "GPECSuiteConfig",
) -> tuple[int, Path]:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(executable_path)],
            cwd=cwd,
            env=gpec_env(config),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=config.timeout,
            check=False,
        )
    return int(result.returncode), log_path
