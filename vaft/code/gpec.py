"""Python-first adapter for the GPEC code suite.

This module treats GPEC as the full VEST linear-stability suite, not only the
``gpec`` executable.  It prepares and optionally runs DCON/MATCH, RDCON,
STRIDE, and GPEC cases from CHEASE/EFIT GEQDSK files using packaged VEST
reference namelists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Optional, Sequence

from ._executables import executable_from_home, missing_home_message
from .base import CodeConfig, CodeInputs, CodeResult, CodeRunner

GPEC_HOME_ENV = "GPECHOME"
DEFAULT_MODULES = ("dcon", "rdcon", "stride", "gpec")
DEFAULT_MODES = (1, 2)
SUPPORTED_MODULES = frozenset(DEFAULT_MODULES)
STABILITY_MODULES = frozenset(("dcon", "rdcon", "stride"))


@dataclass(frozen=True)
class GPECSuiteConfig:
    """Runtime and VEST-default configuration for the GPEC suite.

    ``gpec_home`` defaults to ``None``, in which case the installation root is
    read from ``$GPECHOME``.  Preparing a case never needs it; only running a
    module does, and a missing installation is reported there.
    """

    gpec_home: Path | str | None = None
    executable_dir: Path | str | None = None
    modules: Sequence[str] = DEFAULT_MODULES
    modes: Sequence[int] = DEFAULT_MODES
    run_mode: str = "run_if_available"
    templates_dir: Path | str | None = None
    coil_data_dir: Path | str | None = None
    psilow: float = 1e-2
    psihigh: float = 0.994
    dcon_sas_flag: bool = False
    dcon_qhigh: float = 20.2
    dcon_psiedge: float = 1.0
    timeout: Optional[float] = 1200.0
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass
class GPECCaseInputs:
    """Materialized inputs for one shot/time GPEC-suite case."""

    shot: int
    time_ms: int | str | None
    geqdsk: Path
    workdir: Path
    coil_in: Path | None = None


@dataclass
class GPECModuleRun:
    """Status for one module/mode directory."""

    module: str
    mode: int
    workdir: Path
    returncode: Optional[int] = None
    status: str = "prepared"
    reason: str = ""
    logs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    commands: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0


@dataclass
class GPECSuiteResult:
    """Suite status, records, and collected artifacts."""

    returncode: Optional[int]
    workdir: Path
    shot: int | None = None
    time_ms: int | str | None = None
    records: tuple[GPECModuleRun, ...] = ()
    logs: tuple[Path, ...] = ()
    outputs: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    parsed: Any = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


__all__ = [
    "CodeConfig",
    "CodeInputs",
    "CodeResult",
    "CodeRunner",
    "GPECCaseInputs",
    "GPECModuleRun",
    "GPECSuiteConfig",
    "GPECSuiteResult",
    "collect_gpec_suite_outputs",
    "format_gfile_header_for_gpec",
    "prepare_gpec_suite_case",
    "run_gpec",
    "run_gpec_suite_case",
]


def _package_vest_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "gpec"


def _template_dir(config: GPECSuiteConfig) -> Path:
    return Path(config.templates_dir).expanduser() if config.templates_dir else _package_vest_dir()


def _coil_data_dir(config: GPECSuiteConfig) -> Path:
    return Path(config.coil_data_dir).expanduser() if config.coil_data_dir else _package_vest_dir()


def _gpec_home(config: GPECSuiteConfig) -> Path | None:
    """Resolve the GPEC installation root from the config or ``$GPECHOME``."""
    home = config.gpec_home or os.environ.get(GPEC_HOME_ENV)
    return Path(home).expanduser() if home else None


def _executable_dir(config: GPECSuiteConfig) -> Path | None:
    if config.executable_dir:
        return Path(config.executable_dir).expanduser()
    home = _gpec_home(config)
    return home / "bin" if home else None


def _executable(config: GPECSuiteConfig, program: str) -> Path | None:
    if config.executable_dir:
        return Path(config.executable_dir).expanduser() / program
    home = _gpec_home(config)
    return executable_from_home(
        home,
        home_variable=GPEC_HOME_ENV,
        relative_path=Path("bin") / program,
        code_name=f"GPEC suite ({program})",
    )


def _unconfigured_reason() -> str:
    return missing_home_message(
        home_variable=GPEC_HOME_ENV,
        relative_path="bin/{dcon,match,rdcon,stride,gpec}",
        code_name="GPEC suite",
    )


def _gpec_env(config: GPECSuiteConfig) -> dict[str, str]:
    env = dict(os.environ)
    home = _gpec_home(config)
    if home is not None:
        env[GPEC_HOME_ENV] = str(home)
    env.update({str(key): str(value) for key, value in config.env.items()})
    return env


def _normalized_modules(config: GPECSuiteConfig) -> tuple[str, ...]:
    modules = tuple(str(module).lower() for module in config.modules)
    unknown = sorted(set(modules) - SUPPORTED_MODULES)
    if unknown:
        raise ValueError(f"Unsupported GPEC suite module(s): {', '.join(unknown)}")
    return modules


def _normalized_modes(config: GPECSuiteConfig) -> tuple[int, ...]:
    modes = tuple(int(mode) for mode in config.modes)
    if not modes:
        raise ValueError("At least one toroidal mode is required")
    return modes


def _run_policy(config: GPECSuiteConfig) -> str:
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


def _time_label(inputs: GPECCaseInputs) -> str:
    if inputs.time_ms is not None:
        text = str(inputs.time_ms)
        return text if len(text) >= 5 or not text.isdigit() else f"{int(text):05d}"
    suffix = inputs.geqdsk.name.rsplit(".", maxsplit=1)[-1]
    return suffix if suffix != inputs.geqdsk.name else inputs.geqdsk.stem


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
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


def _write_template(
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


def _copy_gfile_for_gpec(
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


def _module_dir(inputs: GPECCaseInputs, module: str, mode: int) -> Path:
    return inputs.workdir / _time_label(inputs) / module / f"nn={mode}"


def _output_patterns(module: str, mode: int) -> tuple[str, ...]:
    if module == "dcon":
        return (
            "euler.bin",
            "psi_in.bin",
            "vacuum.bin",
            f"dcon_output_n{mode}.nc",
            "dcon.out",
            "match.out",
            "solutions.bin",
        )
    if module == "rdcon":
        return (f"rdcon_output_n{mode}.nc", "delta_gw.out", "dcon.out", "globalsol.bin")
    if module == "stride":
        return (f"stride_output_n{mode}.nc", "stride.out", "delta_prime.out")
    if module == "gpec":
        return (
            f"gpec_control_output_n{mode}.nc",
            f"gpec_profile_output_n{mode}.nc",
            f"gpec_cylindrical_output_n{mode}.nc",
            f"gpec_response_n{mode}.out",
            f"gpec_bnormal_pest_n{mode}.out",
        )
    return ()


def collect_gpec_suite_outputs(workdir: Path | str) -> dict[str, tuple[Path, ...]]:
    """Collect known GPEC-suite outputs under a prepared case directory."""

    root = Path(workdir)
    outputs: dict[str, list[Path]] = {module: [] for module in DEFAULT_MODULES}
    for module in DEFAULT_MODULES:
        for path in sorted(root.glob(f"*/{module}/nn=*/*")):
            if path.is_file():
                outputs[module].append(path)
    return {key: tuple(value) for key, value in outputs.items()}


def _prepared_record(module: str, mode: int, workdir: Path) -> GPECModuleRun:
    return GPECModuleRun(
        module=module,
        mode=mode,
        workdir=workdir,
        status="prepared",
        outputs=tuple(path for pattern in _output_patterns(module, mode) if (path := workdir / pattern).exists()),
    )


def prepare_gpec_suite_case(
    inputs: GPECCaseInputs,
    config: GPECSuiteConfig | None = None,
) -> GPECSuiteResult:
    """Create VEST GPEC-suite input directories for one shot/time GEQDSK."""

    config = config or GPECSuiteConfig()
    template_dir = _template_dir(config)
    coil_data_dir = _coil_data_dir(config)
    modules = _normalized_modules(config)
    modes = _normalized_modes(config)
    geqdsk = Path(inputs.geqdsk).expanduser()
    if not geqdsk.exists():
        raise FileNotFoundError(f"GEQDSK not found: {geqdsk}")
    if not template_dir.is_dir():
        raise FileNotFoundError(f"GPEC template directory not found: {template_dir}")
    if "gpec" in modules and not coil_data_dir.is_dir() and inputs.coil_in is None:
        raise FileNotFoundError(f"GPEC coil data directory not found: {coil_data_dir}")

    inputs.workdir.mkdir(parents=True, exist_ok=True)
    records: list[GPECModuleRun] = []
    eq_filename = geqdsk.name
    for mode in modes:
        for module in modules:
            run_dir = _module_dir(inputs, module, mode)
            run_dir.mkdir(parents=True, exist_ok=True)
            if module in STABILITY_MODULES:
                _copy_gfile_for_gpec(geqdsk, run_dir / eq_filename, shot=inputs.shot, time_ms=inputs.time_ms)
                _write_template(
                    template_dir / "equil.in",
                    run_dir / "equil.in",
                    {"eq_filename": eq_filename, "psilow": config.psilow, "psihigh": config.psihigh},
                )
                shutil.copy2(template_dir / "vac.in", run_dir / "vac.in")
            if module == "dcon":
                _write_template(
                    template_dir / "dcon.in",
                    run_dir / "dcon.in",
                    {
                        "nn": mode,
                        "sas_flag": config.dcon_sas_flag,
                        "qhigh": config.dcon_qhigh,
                        "psiedge": config.dcon_psiedge,
                    },
                )
                shutil.copy2(template_dir / "match.in", run_dir / "match.in")
            elif module == "rdcon":
                _write_template(template_dir / "rdcon.in", run_dir / "rdcon.in", {"nn": mode})
            elif module == "stride":
                _write_template(template_dir / "stride.in", run_dir / "stride.in", {"nn": mode})
            elif module == "gpec":
                dcon_dir = _module_dir(inputs, "dcon", mode).resolve()
                _write_template(
                    template_dir / "gpec.in",
                    run_dir / "gpec.in",
                    {"dcon_dir": str(dcon_dir), "coil_flag": True},
                )
                shutil.copy2(template_dir / "vac.in", run_dir / "vac.in")
                if inputs.coil_in:
                    shutil.copy2(Path(inputs.coil_in).expanduser(), run_dir / "coil.in")
                else:
                    _write_template(
                        template_dir / "coil.in",
                        run_dir / "coil.in",
                        {"data_dir": str(coil_data_dir.resolve()), "machine": "vest", "coil_num": 3},
                    )
            records.append(_prepared_record(module, mode, run_dir))

    outputs = collect_gpec_suite_outputs(inputs.workdir)
    return GPECSuiteResult(
        returncode=0,
        workdir=inputs.workdir,
        shot=inputs.shot,
        time_ms=inputs.time_ms,
        records=tuple(records),
        outputs=outputs,
    )


def _run_subprocess(
    executable: Path,
    cwd: Path,
    log_path: Path,
    *,
    config: GPECSuiteConfig,
) -> tuple[int, Path]:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(executable)],
            cwd=cwd,
            env=_gpec_env(config),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=config.timeout,
            check=False,
        )
    return int(result.returncode), log_path


def _run_module(
    inputs: GPECCaseInputs,
    config: GPECSuiteConfig,
    module: str,
    mode: int,
) -> GPECModuleRun:
    run_dir = _module_dir(inputs, module, mode)
    policy = _run_policy(config)
    if policy == "prepare_only":
        record = _prepared_record(module, mode, run_dir)
        record.status = "skipped"
        record.reason = "run_mode=prepare_only"
        return record

    program = module
    executable = _executable(config, program)
    if executable is None:
        if policy == "strict":
            raise FileNotFoundError(_unconfigured_reason())
        return GPECModuleRun(module, mode, run_dir, status="skipped", reason=_unconfigured_reason())
    if not executable.is_file() or not os.access(executable, os.X_OK):
        reason = f"missing or non-executable {program}: {executable}"
        if policy == "strict":
            raise FileNotFoundError(reason)
        return GPECModuleRun(module, mode, run_dir, status="skipped", reason=reason)

    if module == "gpec":
        dcon_dir = _module_dir(inputs, "dcon", mode)
        required = ("euler.bin", "psi_in.bin")
        missing = [name for name in required if not (dcon_dir / name).exists()]
        if missing:
            return GPECModuleRun(module, mode, run_dir, status="skipped", reason=f"missing DCON outputs: {', '.join(missing)}")

    commands: list[str] = [str(executable)]
    logs: list[Path] = []
    try:
        returncode, log_path = _run_subprocess(executable, run_dir, run_dir / f"{program}.log", config=config)
        logs.append(log_path)
        status = "completed" if returncode == 0 else "failed"
        if module == "dcon" and returncode == 0:
            match_exec = _executable(config, "match")
            if match_exec is not None and match_exec.is_file() and os.access(match_exec, os.X_OK):
                match_rc, match_log = _run_subprocess(match_exec, run_dir, run_dir / "match.log", config=config)
                commands.append(str(match_exec))
                logs.append(match_log)
                if match_rc != 0:
                    returncode = match_rc
                    status = "failed"
            elif policy == "strict":
                raise FileNotFoundError(f"match executable not found: {match_exec}")
        outputs = tuple(path for pattern in _output_patterns(module, mode) if (path := run_dir / pattern).exists())
        return GPECModuleRun(
            module=module,
            mode=mode,
            workdir=run_dir,
            returncode=returncode,
            status=status,
            logs=tuple(logs),
            outputs=outputs,
            commands=tuple(commands),
        )
    except subprocess.TimeoutExpired as exc:
        outputs = tuple(path for pattern in _output_patterns(module, mode) if (path := run_dir / pattern).exists())
        core_gpec_outputs = {
            f"gpec_control_output_n{mode}.nc",
            f"gpec_profile_output_n{mode}.nc",
            f"gpec_cylindrical_output_n{mode}.nc",
        }
        output_names = {path.name for path in outputs}
        if module == "gpec" and core_gpec_outputs.issubset(output_names):
            return GPECModuleRun(
                module=module,
                mode=mode,
                workdir=run_dir,
                returncode=0,
                status="completed",
                reason=f"timeout after outputs materialized ({exc.timeout} seconds)",
                logs=tuple(logs),
                outputs=outputs,
                commands=tuple(commands),
            )
        return GPECModuleRun(
            module=module,
            mode=mode,
            workdir=run_dir,
            returncode=None,
            status="failed",
            reason=f"timeout after {exc.timeout} seconds",
            logs=tuple(logs),
            outputs=outputs,
            commands=tuple(commands),
        )


def run_gpec_suite_case(
    inputs: GPECCaseInputs,
    config: GPECSuiteConfig | None = None,
) -> GPECSuiteResult:
    """Prepare and optionally run all configured VEST GPEC-suite modules."""

    config = config or GPECSuiteConfig()
    prepare_gpec_suite_case(inputs, config)
    records: list[GPECModuleRun] = []
    modules = _normalized_modules(config)
    modes = _normalized_modes(config)
    run_order = tuple(module for module in ("dcon", "rdcon", "stride", "gpec") if module in modules)
    for mode in modes:
        for module in run_order:
            records.append(_run_module(inputs, config, module, mode))

    failures = [record for record in records if record.status == "failed"]
    returncode = 1 if failures else 0
    logs = tuple(path for record in records for path in record.logs)
    outputs = collect_gpec_suite_outputs(inputs.workdir)
    return GPECSuiteResult(
        returncode=returncode,
        workdir=inputs.workdir,
        shot=inputs.shot,
        time_ms=inputs.time_ms,
        records=tuple(records),
        logs=logs,
        outputs=outputs,
    )


def _coerce_inputs(inputs: GPECCaseInputs | CodeInputs) -> GPECCaseInputs:
    if isinstance(inputs, GPECCaseInputs):
        return inputs
    if not inputs.files:
        raise ValueError("CodeInputs.files must contain a GEQDSK path for run_gpec()")
    return GPECCaseInputs(shot=0, time_ms=None, geqdsk=Path(inputs.files[0]), workdir=Path(inputs.workdir))


def _coerce_config(config: GPECSuiteConfig | CodeConfig) -> GPECSuiteConfig:
    if isinstance(config, GPECSuiteConfig):
        return config
    executable_dir = Path(config.executable).expanduser().parent if config.executable else None
    return GPECSuiteConfig(executable_dir=executable_dir, timeout=config.timeout, env=config.env)


def run_gpec(inputs: GPECCaseInputs | CodeInputs, config: GPECSuiteConfig | CodeConfig) -> GPECSuiteResult:
    """Compatibility entry point for the GPEC-suite runner."""

    suite_inputs = _coerce_inputs(inputs)
    suite_config = _coerce_config(config)
    return run_gpec_suite_case(suite_inputs, suite_config)
