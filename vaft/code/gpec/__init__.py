"""Python-first adapter for the GPEC code suite.

This package treats GPEC as the full VEST linear-stability suite, not only the
``gpec`` executable.  It prepares and optionally runs DCON/MATCH, RDCON/RMATCH,
STRIDE, and GPEC cases from CHEASE/EFIT GEQDSK files using packaged VEST
reference namelists.

Solver-agnostic plumbing (subprocess execution, ``$GPECHOME`` resolution,
namelist patching, directory layout) lives in :mod:`._runtime`. What differs
between DCON/RDCON/STRIDE/GPEC (namelist content, companion executables,
expected outputs, and what "succeeded" means) lives in :mod:`._solvers`. This
module wires the two together into the suite-level ``prepare``/``run`` API.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import _runtime as rt
from ._coil_input import (
    CoilInputSpec,
    emit_coil_dat,
    read_coil_in,
    stage_coil_data,
    write_coil_in,
)
from ._dcon_output import DconEigenfunction, DconOutput, read_dcon_output, read_solutions_bin
from ._gpec_output import (
    GpecControlOutput,
    GpecCylindricalOutput,
    GpecIdealResult,
    read_gpec_netcdf,
)
from ._matching_output import Pest3MatchingOutput, read_pest3_matching_output
from ._solvers import SOLVERS, SolverContext, _check_nc_variable
from ._types import (
    DEFAULT_MODES,
    DEFAULT_MODULES,
    GPEC_HOME_ENV,
    STABILITY_MODULES,
    SUPPORTED_MODULES,
    DCONOptions,
    GPECCaseInputs,
    GPECModuleRun,
    GPECSuiteConfig,
    GPECSuiteResult,
    IdealGPECOptions,
    RDCONOptions,
    STRIDEOptions,
)

# Re-exported for tests/callers that reach into adapter internals (e.g.
# `gpec._gpec_home`, `gpec._executable`) -- kept as thin aliases so existing
# call sites do not need to know the plumbing moved to `_runtime`.
_gpec_home = rt.gpec_home
_executable_dir = rt.executable_dir
_executable = rt.executable
_optional_executable = rt.optional_executable
_gpec_env = rt.gpec_env
_unconfigured_reason = rt.unconfigured_reason
_run_policy = rt.run_policy
_time_label = rt.time_label
_module_dir = rt.module_dir
format_gfile_header_for_gpec = rt.format_gfile_header_for_gpec


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
    solver = SOLVERS[module]
    return GPECModuleRun(
        module=module,
        mode=mode,
        workdir=workdir,
        status="prepared",
        outputs=tuple(path for pattern in solver.output_patterns(mode) if (path := workdir / pattern).exists()),
    )


def validate_dcon_result(
    dcon_dir: Path | str,
    mode: int,
    *,
    verify_outputs: bool = False,
) -> list[str]:
    """Describe what makes ``dcon_dir`` unusable as an ideal-GPEC prerequisite.

    A valid ideal-GPEC run consumes the same-mode DCON products ``euler.bin``
    and ``psi_in.bin``.  ``verify_outputs`` additionally checks that
    ``dcon_output_n{mode}.nc`` exists and carries the eigenvalue variable, so
    a truncated or failed DCON solve is caught before GPEC is launched.
    Returns an empty list when the DCON result is usable.
    """
    dcon_dir = Path(dcon_dir)
    problems = [
        f"missing DCON output: {name}"
        for name in ("euler.bin", "psi_in.bin")
        if not (dcon_dir / name).exists()
    ]
    if verify_outputs and not problems:
        ok, reason = _check_nc_variable(
            dcon_dir, f"dcon_output_n{mode}.nc", "W_t_eigenvalue"
        )
        if not ok:
            problems.append(reason)
    return problems


def _dcon_run_dir(inputs: GPECCaseInputs, mode: int, geqdsk: Path) -> Path:
    """Locate same-cell DCON output, possibly in a separate code work tree."""

    root = inputs.dcon_workdir or inputs.workdir
    return rt.module_dir(root, inputs.time_ms, "dcon", mode, geqdsk=geqdsk)


def prepare_gpec_suite_case(
    inputs: GPECCaseInputs,
    config: GPECSuiteConfig | None = None,
) -> GPECSuiteResult:
    """Create VEST GPEC-suite input directories for one shot/time GEQDSK."""

    config = config or GPECSuiteConfig()
    template_dir = rt.template_dir(config)
    coil_data_dir = rt.coil_data_dir(config)
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
            run_dir = rt.module_dir(inputs.workdir, inputs.time_ms, module, mode, geqdsk=geqdsk)
            run_dir.mkdir(parents=True, exist_ok=True)
            if module in STABILITY_MODULES:
                rt.copy_gfile_for_gpec(geqdsk, run_dir / eq_filename, shot=inputs.shot, time_ms=inputs.time_ms)
                rt.write_template(
                    template_dir / "equil.in",
                    run_dir / "equil.in",
                    {"eq_filename": eq_filename, "psilow": config.psilow, "psihigh": config.psihigh},
                )
                shutil.copy2(template_dir / "vac.in", run_dir / "vac.in")
            ctx = SolverContext(
                run_dir=run_dir,
                template_dir=template_dir,
                coil_data_dir=coil_data_dir,
                eq_filename=eq_filename,
                mode=mode,
                inputs=inputs,
                config=config,
                dcon_dir=_dcon_run_dir(inputs, mode, geqdsk).resolve(),
            )
            SOLVERS[module].prepare(ctx)
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


def _run_module(
    inputs: GPECCaseInputs,
    config: GPECSuiteConfig,
    module: str,
    mode: int,
) -> GPECModuleRun:
    run_dir = rt.module_dir(inputs.workdir, inputs.time_ms, module, mode, geqdsk=inputs.geqdsk)
    policy = rt.run_policy(config)
    solver = SOLVERS[module]
    if policy == "prepare_only":
        record = _prepared_record(module, mode, run_dir)
        record.status = "skipped"
        record.reason = "run_mode=prepare_only"
        return record

    program = module
    executable = rt.executable(config, program)
    if executable is None:
        if policy == "strict":
            raise FileNotFoundError(rt.unconfigured_reason())
        return GPECModuleRun(module, mode, run_dir, status="skipped", reason=rt.unconfigured_reason())
    if not executable.is_file() or not os.access(executable, os.X_OK):
        reason = f"missing or non-executable {program}: {executable}"
        if policy == "strict":
            raise FileNotFoundError(reason)
        return GPECModuleRun(module, mode, run_dir, status="skipped", reason=reason)

    if module == "gpec":
        dcon_dir = _dcon_run_dir(inputs, mode, Path(inputs.geqdsk))
        problems = validate_dcon_result(dcon_dir, mode, verify_outputs=config.verify_outputs)
        if problems:
            reason = (
                f"invalid DCON result in {dcon_dir}: {'; '.join(problems)} "
                "-- run the dcon module for the same mode first, or pass "
                "dcon_workdir pointing at a completed DCON work tree"
            )
            if policy == "strict":
                raise RuntimeError(reason)
            return GPECModuleRun(module, mode, run_dir, status="skipped", reason=reason)

    # Resuming a pipeline should not re-run a completed numerical solve just
    # because another time slice in the same code/mode cell failed.  A full
    # solver output set is immutable input for the downstream IDS builder.
    existing_outputs = tuple(path for pattern in solver.output_patterns(mode) if (path := run_dir / pattern).exists())
    if len(existing_outputs) == len(solver.output_patterns(mode)):
        if config.verify_outputs:
            ok, reason = solver.check_success(run_dir, mode)
            if not ok:
                return GPECModuleRun(module, mode, run_dir, status="failed", reason=reason, outputs=existing_outputs)
        return GPECModuleRun(
            module,
            mode,
            run_dir,
            returncode=0,
            status="completed",
            reason="reused existing solver outputs",
            outputs=existing_outputs,
        )

    commands: list[str] = [str(executable)]
    logs: list[Path] = []
    try:
        returncode, log_path = rt.run_subprocess(executable, run_dir, run_dir / f"{program}.log", config=config)
        logs.append(log_path)
        status = "completed" if returncode == 0 else "failed"
        reason = ""
        if returncode == 0:
            for companion in solver.companion_executables():
                companion_exec = rt.optional_executable(config, companion)
                if companion_exec is not None and companion_exec.is_file() and os.access(companion_exec, os.X_OK):
                    companion_rc, companion_log = rt.run_subprocess(
                        companion_exec, run_dir, run_dir / f"{companion}.log", config=config
                    )
                    commands.append(str(companion_exec))
                    logs.append(companion_log)
                    if companion_rc != 0:
                        returncode = companion_rc
                        status = "failed"
                elif policy == "strict":
                    raise FileNotFoundError(f"{companion} executable not found: {companion_exec}")
        if status == "completed" and config.verify_outputs:
            ok, check_reason = solver.check_success(run_dir, mode)
            if not ok:
                status = "failed"
                reason = check_reason
        outputs = tuple(path for pattern in solver.output_patterns(mode) if (path := run_dir / pattern).exists())
        return GPECModuleRun(
            module=module,
            mode=mode,
            workdir=run_dir,
            returncode=returncode,
            status=status,
            reason=reason,
            logs=tuple(logs),
            outputs=outputs,
            commands=tuple(commands),
        )
    except subprocess.TimeoutExpired as exc:
        outputs = tuple(path for pattern in solver.output_patterns(mode) if (path := run_dir / pattern).exists())
        core_gpec_outputs = {
            f"gpec_control_output_n{mode}.nc",
            f"gpec_profile_output_n{mode}.nc",
            f"gpec_cylindrical_output_n{mode}.nc",
        }
        output_names = {path.name for path in outputs}
        if module == "gpec" and core_gpec_outputs.issubset(output_names):
            # A process killed mid-write can leave the core files present but
            # truncated, so the timeout carve-out must verify its outputs
            # before reporting success (release review, 0.6.0). This does not
            # honour `verify_outputs`: that flag trades extra checking against
            # a run that already exited cleanly, whereas calling a *timed-out*
            # run successful is a claim that needs evidence either way.
            ok, check_reason = solver.check_success(run_dir, mode)
            if ok:
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
                reason=(
                    f"timeout after {exc.timeout} seconds; outputs present but "
                    f"failed verification: {check_reason}"
                ),
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


__all__ = [
    "DCONOptions",
    "RDCONOptions",
    "STRIDEOptions",
    "IdealGPECOptions",
    "GPEC_HOME_ENV",
    "GPECCaseInputs",
    "GPECModuleRun",
    "GPECSuiteConfig",
    "GPECSuiteResult",
    "CoilInputSpec",
    "collect_gpec_suite_outputs",
    "emit_coil_dat",
    "format_gfile_header_for_gpec",
    "stage_coil_data",
    "write_coil_in",
    "prepare_gpec_suite_case",
    "read_coil_in",
    "run_gpec_suite_case",
    "validate_dcon_result",
    "DconEigenfunction",
    "DconOutput",
    "GpecControlOutput",
    "GpecCylindricalOutput",
    "GpecIdealResult",
    "read_dcon_output",
    "read_gpec_netcdf",
    "read_solutions_bin",
    "Pest3MatchingOutput",
    "read_pest3_matching_output",
]
