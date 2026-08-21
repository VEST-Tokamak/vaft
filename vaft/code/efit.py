from vaft.formula import green_br_bz, green_r, calculate_distance
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import statistics
import math
from scipy.signal import savgol_filter
from scipy import interpolate,optimize
from omas import *
import os
from typing import Any, Mapping, Optional, Sequence
from vaft.machine_mapping.utils import path_exists

from ._executables import executable_from_home, missing_home_message
from .efit_status import (
    EFITSliceStatus,
    EFITValidationConfig,
    apply_temporal_continuity,
    validate_efit_slice,
)


# Canonical EFIT installation root and its historical executable-oriented name.
EFIT_HOME_ENV = "EFITHOME"
EFIT_HOME_EXECUTABLE = Path("bin/efit")
EFIT_EXEC_ENV = "EFIT"


@dataclass(frozen=True)
class EFITConfig:
    """Python-first EFIT workflow configuration."""

    executable: Optional[str] = None
    workdir: Path | str = Path(".")
    shot: Optional[int] = None
    times: Optional[Sequence[float]] = None
    constraint_options: Mapping[str, Any] = field(default_factory=dict)
    profile_options: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    args: Sequence[str] = ()
    timeout: Optional[float] = None
    npprime: int = 2
    nffprime: int = 2
    stack_size_kb: Optional[int] = 32768


@dataclass
class EFITInputs:
    """Input bundle for an EFIT run."""

    workdir: Path
    ods: Any = None
    geqdsk: Any = None
    kfiles: tuple[Path, ...] = ()
    files: tuple[Path, ...] = ()


@dataclass
class EFITResult:
    """Collected EFIT run status, files, logs, and parsed equilibria."""

    returncode: Optional[int]
    workdir: Path
    gfiles: tuple[Path, ...] = ()
    afiles: tuple[Path, ...] = ()
    mfiles: tuple[Path, ...] = ()
    kfiles: tuple[Path, ...] = ()
    logs: tuple[Path, ...] = ()
    stdout: str = ""
    stderr: str = ""
    geqdsk: tuple[Any, ...] = ()
    parse_errors: tuple[str, ...] = ()
    ods: Any = None
    status: str = "completed"
    reason: str = ""
    slice_statuses: tuple[EFITSliceStatus, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0

    @property
    def usable(self) -> bool:
        """Whether at least one collected slice is scientifically usable."""
        return any(status.usable for status in self.slice_statuses)


def _efit_workdir(config: EFITConfig | None = None, workdir: str | Path | None = None) -> Path:
    if workdir is not None:
        return Path(workdir).expanduser()
    if config is not None:
        return Path(config.workdir).expanduser()
    return Path(".").expanduser()


def _infer_shot(ods: Any = None, config: EFITConfig | None = None) -> int:
    if config is not None and config.shot is not None:
        return int(config.shot)
    if ods is not None:
        for path in (
            "dataset_description.data_entry.pulse",
            "summary.global_quantities.pulse",
        ):
            try:
                return int(ods[path])
            except Exception:
                pass
    raise ValueError("EFIT shot number is required in EFITConfig.shot or ODS metadata")


def _find_outputs(workdir: Path, prefix: str, shot: int | None = None) -> tuple[Path, ...]:
    candidates = []
    search_roots = [workdir / f"{prefix}file", workdir]
    pattern = f"{prefix}0{shot}.*" if shot is not None else f"{prefix}*"
    for root in search_roots:
        if root.exists():
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
    return tuple(sorted(set(candidates)))


def _relative_input_names(workdir: Path, kfiles: Sequence[Path]) -> list[str]:
    names = []
    for kfile in kfiles:
        path = Path(kfile)
        try:
            names.append(str(path.relative_to(workdir)))
        except ValueError:
            names.append(str(path))
    return names


def _efit_stdin(workdir: Path, kfiles: Sequence[Path]) -> str:
    input_names = _relative_input_names(workdir, kfiles)
    if not input_names:
        raise ValueError("At least one EFIT kfile is required")
    return "2\n{}\n{}\n".format(len(input_names), "\n".join(input_names))


def _resolve_efit_executable(config: EFITConfig) -> Path | None:
    """Resolve EFIT from an explicit path, ``$EFITHOME``, or legacy ``$EFIT``.

    The existing explicit adapter option remains authoritative for backward
    compatibility.  A configured ``$EFITHOME`` must contain ``bin/efit`` and
    fails immediately when that installation is incomplete.  The historical
    ``$EFIT`` lookup is used only when ``$EFITHOME`` is absent.
    """
    if config.executable:
        candidate = Path(config.executable).expanduser()
        return candidate / "efit" if candidate.is_dir() else candidate
    environment = {**os.environ, **dict(config.env)}
    home_executable = executable_from_home(
        environment.get(EFIT_HOME_ENV),
        home_variable=EFIT_HOME_ENV,
        relative_path=EFIT_HOME_EXECUTABLE,
        code_name="EFIT",
    )
    if home_executable is not None:
        return home_executable
    env_path = environment.get(EFIT_EXEC_ENV)
    if env_path:
        env_candidate = Path(env_path).expanduser()
        return env_candidate / "efit" if env_candidate.is_dir() else env_candidate
    return None


def _efit_unconfigured_reason() -> str:
    return missing_home_message(
        home_variable=EFIT_HOME_ENV,
        relative_path=EFIT_HOME_EXECUTABLE,
        code_name="EFIT",
        compatibility_variables=(EFIT_EXEC_ENV,),
    )


def find_efit_executable(config: EFITConfig | None = None) -> Path | None:
    """Return EFIT resolved from explicit config, ``$EFITHOME``, or ``$EFIT``."""
    exe = _resolve_efit_executable(config or EFITConfig())
    if exe is not None and exe.exists() and os.access(exe, os.X_OK):
        return exe
    return None


def _efit_command(config: EFITConfig, executable: str | Path | None = None) -> list[str]:
    resolved = executable if executable is not None else config.executable
    if not resolved:
        raise ValueError("EFITConfig.executable is required to run EFIT")
    executable = str(resolved)
    args = [str(arg) for arg in config.args]
    if config.stack_size_kb is None:
        return [executable, *args]
    return [
        "bash",
        "-lc",
        f"ulimit -s {int(config.stack_size_kb)}; exec \"$@\"",
        "efit-runner",
        executable,
        *args,
    ]


def prepare_efit_inputs(ods: Any, config: EFITConfig) -> EFITInputs:
    """Prepare EFIT input files from an ODS and workflow configuration."""
    workdir = _efit_workdir(config)
    workdir.mkdir(parents=True, exist_ok=True)
    shot = _infer_shot(ods, config)

    if config.times is not None:
        try:
            ods["equilibrium.time"] = np.asarray(config.times)
        except Exception:
            pass

    generate_kfile(
        ods,
        shot,
        config.npprime,
        config.nffprime,
        save_dir=str(workdir),
    )
    kfiles = _find_outputs(workdir, "k", shot)
    return EFITInputs(workdir=workdir, ods=ods, kfiles=kfiles, files=kfiles)


def run_efit(inputs: EFITInputs, config: EFITConfig) -> EFITResult:
    """Run EFIT with prepared inputs and collect produced outputs.

    The EFIT binary is resolved from ``EFITConfig.executable``, ``$EFITHOME``,
    or the legacy ``$EFIT`` variable. An incomplete ``$EFITHOME`` installation
    raises immediately; an unconfigured installation degrades to a skipped
    :class:`EFITResult`.
    """
    workdir = _efit_workdir(config, inputs.workdir)
    executable = _resolve_efit_executable(config)
    if executable is None:
        result = collect_efit_outputs(
            workdir,
            config,
            runtime_status="runtime_error",
            runtime_reason=_efit_unconfigured_reason(),
            expected_kfiles=inputs.kfiles,
        )
        result.status = "skipped"
        result.reason = _efit_unconfigured_reason()
        return result
    if not (executable.exists() and os.access(executable, os.X_OK)):
        reason = f"missing executable: {executable}"
        result = collect_efit_outputs(
            workdir,
            config,
            runtime_status="runtime_error",
            runtime_reason=reason,
            executable=executable,
            expected_kfiles=inputs.kfiles,
        )
        result.status = "skipped"
        result.reason = reason
        return result
    command = _efit_command(config, executable)
    stdin_text = _efit_stdin(workdir, inputs.kfiles)
    env = os.environ.copy()
    env.update(dict(config.env))
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            env=env,
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=config.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        (workdir / "run_efit.out").write_text(stdout, encoding="utf-8")
        (workdir / "run_efit.err").write_text(stderr, encoding="utf-8")
        result = collect_efit_outputs(
            workdir,
            config,
            runtime_status="timeout",
            runtime_reason=f"EFIT timed out after {config.timeout} seconds",
            executable=executable,
            expected_kfiles=inputs.kfiles,
        )
        result.status = "failed"
        result.reason = f"EFIT timed out after {config.timeout} seconds"
        result.stdout = stdout
        result.stderr = stderr
        return result
    except OSError as error:
        result = collect_efit_outputs(
            workdir,
            config,
            runtime_status="runtime_error",
            runtime_reason=str(error),
            executable=executable,
            expected_kfiles=inputs.kfiles,
        )
        result.status = "failed"
        result.reason = str(error)
        result.stderr = str(error)
        return result
    (workdir / "run_efit.out").write_text(completed.stdout, encoding="utf-8")
    (workdir / "run_efit.err").write_text(completed.stderr, encoding="utf-8")
    result = collect_efit_outputs(
        workdir,
        config,
        returncode=completed.returncode,
        runtime_status="completed",
        executable=executable,
        expected_kfiles=inputs.kfiles,
    )
    result.status = "completed" if completed.returncode == 0 else "failed"
    result.stdout = completed.stdout
    result.stderr = completed.stderr
    return result


def _efit_case_key(path: Path) -> str:
    """Return the common shot/time portion of a k-, g-, a-, or m-file name."""
    if path.name[:1].lower() in {"k", "g", "a", "m"}:
        return path.name[1:]
    return path.name


def _efit_case_time(case: str) -> float:
    """Decode the conventional millisecond suffix used by VEST EFIT files."""
    try:
        return int(case.rsplit(".", 1)[1]) / 1000.0
    except (IndexError, ValueError):
        return float("nan")


def collect_efit_outputs(
    workdir: str | Path,
    config: EFITConfig | None = None,
    *,
    returncode: int | None = None,
    runtime_status: str = "collected",
    runtime_reason: str = "",
    executable: str | Path | None = None,
    expected_kfiles: Sequence[str | Path] = (),
    validation_config: EFITValidationConfig | None = None,
) -> EFITResult:
    """Collect EFIT files and assign independent status to every attempted slice."""
    base = _efit_workdir(config, workdir)
    shot = config.shot if config is not None else None
    gfiles = _find_outputs(base, "g", shot)
    afiles = _find_outputs(base, "a", shot)
    mfiles = _find_outputs(base, "m", shot)
    kfiles = tuple(
        sorted(
            set(_find_outputs(base, "k", shot))
            | {Path(path) for path in expected_kfiles}
        )
    )
    logs = (
        tuple(
            sorted(
                path
                for path in base.rglob("*")
                if path.is_file()
                and (
                    path.suffix == ".log"
                    or path.name in {"run_efit.out", "run_efit.err"}
                )
            )
        )
        if base.exists()
        else ()
    )

    parsed_by_case = {}
    parse_error_by_case = {}
    for gfile in gfiles:
        case = _efit_case_key(gfile)
        try:
            from vaft.data.eqdsk import read_geqdsk

            parsed_by_case[case] = read_geqdsk(gfile)
        except Exception as exc:
            parse_error_by_case[case] = f"{gfile}: {exc}"
            continue

    parsed_cases = sorted(parsed_by_case, key=_efit_case_time)
    parsed = [parsed_by_case[case] for case in parsed_cases]
    parse_errors = list(parse_error_by_case.values())
    ods = None
    conversion_error = None
    if parsed:
        try:
            for idx, item in enumerate(parsed):
                ods = item.to_omas(ods=ods, time_index=idx)
            times = np.asarray([_efit_case_time(case) for case in parsed_cases])
            ods["equilibrium.time"] = times
            for idx, time_value in enumerate(times):
                ods[f"equilibrium.time_slice.{idx}.time"] = time_value
        except Exception as exc:
            conversion_error = f"to_omas: {exc}"
            parse_errors.append(conversion_error)
            ods = None

    file_maps = {
        "kfile": {_efit_case_key(path): path for path in kfiles},
        "gfile": {_efit_case_key(path): path for path in gfiles},
        "afile": {_efit_case_key(path): path for path in afiles},
        "mfile": {_efit_case_key(path): path for path in mfiles},
    }
    cases = sorted(
        set().union(*(mapping.keys() for mapping in file_maps.values())),
        key=_efit_case_time,
    )
    if config is not None and config.shot is not None and config.times is not None:
        configured_cases = {
            f"0{int(config.shot)}.{int(round(float(time_value) * 1000)):05d}"
            for time_value in config.times
        }
        cases = sorted(set(cases) | configured_cases, key=_efit_case_time)
    status_shot = int(shot) if shot is not None else 0
    statuses = []
    for case in cases:
        statuses.append(
            validate_efit_slice(
                shot=status_shot,
                time=_efit_case_time(case),
                runtime_status=runtime_status,
                returncode=returncode,
                kfile=file_maps["kfile"].get(case),
                gfile=file_maps["gfile"].get(case),
                afile=file_maps["afile"].get(case),
                mfile=file_maps["mfile"].get(case),
                geqdsk=parsed_by_case.get(case),
                parse_error=(
                    parse_error_by_case.get(case)
                    or (conversion_error if case in parsed_by_case else None)
                ),
                provenance={
                    "case": case,
                    "runtime_reason": runtime_reason,
                    "attempt_logs": [str(path) for path in logs],
                    "executable": (
                        str(executable or config.executable)
                        if executable is not None
                        or (config is not None and config.executable)
                        else None
                    ),
                    "configuration": (
                        {
                            "npprime": config.npprime,
                            "nffprime": config.nffprime,
                            "args": [str(value) for value in config.args],
                            "timeout": config.timeout,
                            "stack_size_kb": config.stack_size_kb,
                        }
                        if config is not None
                        else {}
                    ),
                },
                config=validation_config,
            )
        )
    statuses = list(apply_temporal_continuity(statuses, validation_config))

    return EFITResult(
        returncode=returncode,
        workdir=base,
        gfiles=gfiles,
        afiles=afiles,
        mfiles=mfiles,
        kfiles=kfiles,
        logs=logs,
        geqdsk=tuple(parsed),
        parse_errors=tuple(parse_errors),
        ods=ods,
        slice_statuses=tuple(statuses),
    )

def gauss_fit4(coef, x):
    return coef[0] * np.exp(-(x - coef[1]) ** 2 / 2 / coef[2] / coef[2]) + coef[3]

def min_gauss_fit4(coef, x, y):
    res = 0.0
    for i in range(len(x)):
        res = res + (y[i] - gauss_fit4(coef, x[i])) ** 2
    return np.sqrt(res)


def vfit_equilibrium_form_constraints(EQ,PF,MG,TF, times, constraints, average):
    # difference with omas version: pf_current not multiplied by nbturn
    # flux_loop divided by 2Pi

#    EQ=ods['equilibrium']
#    PF=ods['pf_active']
#    MG=ods['magnetics']
#    TF=ods['tf']
    
    EQ['ids_properties.comment'] = 'constraint equilibrium'
    EQ['ids_properties.homogeneous_time'] = 1
    EQ['time'] = times
    nbt = len(times)

    ave_time = []
    for i in range(nbt):
        ave_time.append(np.arange(times[i] - average, times[i] + average, 0.0001))

    if 'pf_current' in constraints:
        for channel in PF['coil']:
            label = PF[f'coil.{channel}.name']
            turns = PF[f'coil.{channel}.element.0.turns_with_sign']
            time = PF[f'coil.{channel}.current.time']
            data = PF[f'coil.{channel}.current.data']
            error = PF[f'coil.{channel}.current.data_error_upper']

            for i in range(nbt):
                const = statistics.mean(np.interp(ave_time[i], time, data))
                const_error = statistics.mean(np.interp(ave_time[i], time, error))

                EQ[f'time_slice.{i}.constraints.pf_current.{channel}.measured'] = const
                EQ[f'time_slice.{i}.constraints.pf_current.{channel}.measured_error_upper'] = const_error
                EQ[f'time_slice.{i}.constraints.pf_current.{channel}.source'] = label

    if 'bpol_probe' in constraints:
        for channel in MG['b_field_pol_probe']:
            if not path_exists(MG, f'b_field_pol_probe.{channel}.field.data'):
                continue
            time = MG['time']
            data = MG[f'b_field_pol_probe.{channel}.field.data']
            error = MG[f'b_field_pol_probe.{channel}.field.data_error_upper']
            label = MG[f'b_field_pol_probe.{channel}.identifier']
            for i in range(nbt):
                const = statistics.mean(np.interp(ave_time[i], time, data))
                const_error = statistics.mean(np.interp(ave_time[i], time, error))
                EQ[f'time_slice.{i}.constraints.bpol_probe.{channel}.measured'] = const
                EQ[f'time_slice.{i}.constraints.bpol_probe.{channel}.measured_error_upper'] = const_error
                EQ[f'time_slice.{i}.constraints.bpol_probe.{channel}.source'] = label

    if 'flux_loop' in constraints:
        for channel in MG['flux_loop']:
            time = MG['time']
            data = MG[f'flux_loop.{channel}.flux.data']
            error = MG[f'flux_loop.{channel}.flux.data_error_upper']
            label = MG[f'flux_loop.{channel}.identifier']
            for i in range(nbt):
                # const=statistics.mean(np.interp(ave_time[i],time,data/2/math.pi)) # Origianl version
                # const_error=statistics.mean(np.interp(ave_time[i],time,error/2/math.pi))
                const = statistics.mean(
                    np.interp(ave_time[i], time, data)
                )  # Modified Version for ods/ids convention (division by 2Pi are conducted in the k-file generation)
                const_error = statistics.mean(np.interp(ave_time[i], time, error))  # Wb
                EQ[f'time_slice.{i}.constraints.flux_loop.{channel}.measured'] = const
                EQ[f'time_slice.{i}.constraints.flux_loop.{channel}.measured_error_upper'] = const_error
                EQ[f'time_slice.{i}.constraints.flux_loop.{channel}.source'] = label

    if 'ip' in constraints:
        time = MG[f'ip.0.time']
        data = MG[f'ip.0.data']
        error = MG[f'ip.0.data_error_upper']

        for i in range(nbt):
            const = statistics.mean(np.interp(ave_time[i], time, data))
            const_error = statistics.mean(np.interp(ave_time[i], time, error))

            EQ[f'time_slice.{i}.constraints.ip.measured'] = const
            EQ[f'time_slice.{i}.constraints.ip.measured_error_upper'] = const_error

    if 'diamagnetic_flux' in constraints:
        time = MG[f'diamagnetic_flux.0.time']
        data = MG[f'diamagnetic_flux.0.data']
        error = MG[f'diamagnetic_flux.0.data_error_upper']

        for i in range(nbt):
            const = statistics.mean(np.interp(ave_time[i], time, data))
            const_error = statistics.mean(np.interp(ave_time[i], time, error))
            #            print(i,const*2*math.pi)

            EQ[f'time_slice.{i}.constraints.diamagnetic_flux.measured'] = const
            EQ[f'time_slice.{i}.constraints.diamagnetic_flux.measured_error_upper'] = const_error

    if 'b_field_tor_vacuum_r' in constraints:
        time = TF[f'b_field_tor_vacuum_r.time']
        data = TF[f'b_field_tor_vacuum_r.data']
        error = TF[f'b_field_tor_vacuum_r.data_error_upper']
        for i in range(nbt):
            const = statistics.mean(np.interp(ave_time[i], time, data))
            const_error = statistics.mean(np.interp(ave_time[i], time, error))

            EQ[f'time_slice.{i}.constraints.b_field_tor_vacuum_r.measured'] = const
            EQ[f'time_slice.{i}.constraints.b_field_tor_vacuum_r.measured_error_upper'] = const_error


def vfit_pf_active_efit26(PF,PF_orig,shot,tstart,tend,dt):
    nbcoil=26 # number of coil
    PFname=['PF1-1','PF1-2','PF1-3','PF1-4','PF1-5','PF1-6','PF1-7','PF1-8','PF2U','PF2L','PF3U','PF3L','PF4U','PF4L','PF5U','PF5L','PF6U','PF6L','PF7U','PF7L','PF8U','PF8L','PF9U','PF9L','PF10U','PF10L']
    Rcoil = 1.68E-8 # Copper
    # For resistance calculation
    if shot <= 45957:
        nbELT=[79,79,79,79,79,79,79,79,250,250,8,8,8,8,12,12,12,12,24,24,24,24,24,24,24,24] # number of turn of each coil
        RPF=[0.053,0.053,0.053,0.053,0.053,0.053,0.053,0.053,0.104,0.104,0.29,0.29,0.57,0.57,0.71,0.71,0.71,0.71,0.71,0.71,0.71,0.71,0.93,0.93,0.93,0.93] # Radius
        ZPF=[0.15,0.45,0.75,1.05,-0.15,-0.45,-0.75,-1.05,0.98,-0.98,1.25,-1.25,1.25,-1.25,1.15,-1.15,0.875,-0.875,0.72,-0.72,0.68,-0.68,0.5223,-0.5223,0.4807,-0.4807]
        dRPF=[0.0172,0.0172,0.0172,0.0172,0.0172,0.0172,0.0172,0.0172,0.04,0.04,0.028,0.028,0.028,0.028,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042]
        dZPF=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.38,0.38,0.0145,0.0145,0.0145,0.0145,0.0145,0.0145,0.0145,0.0145,0.0324,0.0324,0.0324,0.0324,0.0324,0.0324,0.0324,0.0324]
    else:
        nbELT=[79,79,79,79,79,79,79,79,250,250,8,8,8,8,12,12,24,24,12,12,24,24,24,24,24,24] # number of turn of each coil
        RPF=[0.053,0.053,0.053,0.053,0.053,0.053,0.053,0.053,0.104,0.104,0.29,0.29,0.57,0.57,0.71,0.71,0.71,0.71,0.71,0.71,0.71,0.71,0.93,0.93,0.93,0.93] # Radius
        ZPF=[0.15,0.45,0.75,1.05,-0.15,-0.45,-0.75,-1.05,0.98,-0.98,1.25,-1.25,1.25,-1.25,1.15,-1.15,0.8827,-0.8827,0.7119,-0.7119,0.68,-0.68,0.5223,-0.5223,0.4807,-0.4807]
        dRPF=[0.0172,0.0172,0.0172,0.0172,0.0172,0.0172,0.0172,0.0172,0.04,0.04,0.028,0.028,0.028,0.028,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042]
        dZPF=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.38,0.38,0.0145,0.0145,0.0145,0.0145,0.0145,0.0145,0.0308,0.0308,0.0162,0.0162,0.0324,0.0324,0.0324,0.0324,0.0324,0.0324]

#    (time,data)=vest_loadn(shot,'PF1 Current')
    time=PF_orig['time']
    if dt > 0:
        tstart=max(tstart,time[0])
        tend=min(tend,time[-1])
        time_1=np.arange(tstart,tend,dt)
    else:
        time_1=time

    PF['ids_properties.comment'] = 'PF config from vfit_pf_active_efit16'
    PF['ids_properties.homogeneous_time'] = 1

    PF['time']=time_1
    nbt=len(time_1)
    
    for i in range(nbcoil):
        PF['coil.{}.name'.format(i)]=PFname[i]
        PF['coil.{}.identifier'.format(i)]=PFname[i]

    for i in range(nbcoil):
        APF=dRPF[i]*dZPF[i]
        PF['coil.{}.resistance'.format(i)]=2.* math.pi*Rcoil*RPF[i]/APF
        PF['coil.{}.element.0.turns_with_sign'.format(i)]=nbELT[i]
        PF['coil.{}.element.0.geometry.geometry_type'.format(i)]=2
        PF['coil.{}.element.0.geometry.rectangle.r'.format(i)]=RPF[i]
        PF['coil.{}.element.0.geometry.rectangle.z'.format(i)]=ZPF[i]
        PF['coil.{}.element.0.geometry.rectangle.width'.format(i)]=dRPF[i]
        PF['coil.{}.element.0.geometry.rectangle.height'.format(i)]=dZPF[i]
        PF['coil.{}.element.0.area'.format(i)]=APF

    # Take current from VEST DB
    PFdata=[]
    nbcoil2 = len(PF_orig['coil'])
    for i in range(nbcoil2):
        PFdata.append(PF_orig[f'coil.{i}.current.data'])

    conver=[0,0,0,0,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
    for i in range(nbcoil):
        j=conver[i]
        PF[f'coil.{i}.current.data']=np.interp(time_1,time,PFdata[j])
#        PF[f'coil.{i}.current.time']=time_1


def correct_flux_loop(ods):
    """
    In the inboard flux loop near the central solenoid,
    there is an issue where the uncertainty in the vacuum component of the total signal becomes larger than the plasma signal.
    To address this, a scaling factor vector is calculated by fitting the ratio between the measured values and the calculated values before plasma onset,
    aligning the measured data with the calculated values.

    Caclulate
    """
    #    xrange = [0.3, 0.36] # ms
    # Print the code
    print('Run Inboard Flux Loop Correction Script')

    # Find the loop voltage (flux loop) voltage onset time
    MG = ods['magnetics']
    nbflux = len(MG['flux_loop'])
    fl_data = MG[f'flux_loop.{nbflux-1}.flux.data']
    fl_time = MG['time']
    #    (fl_time, fl_data) = vest_load(shotnumber, 26)
    tstart=max(0.24,fl_time[0])
    tend=min(0.305,fl_time[-1])
    (fl_onset, _, _) = vest_signal_onoffsetpeak(fl_time, fl_data, tstart=tstart, tend=tend, threshold=0.01)
    print('Flux Loop Onset Time: ', fl_onset)

    # Find the plasma onset and offset time
    (onset, offset) = vest_Halpha_tstart_tend(ods)
    fl_onset = fl_onset + 0.003
    onset = onset - 0.003

    # Calculate response matrix
    (psi_total, psi_coil, psi_eddy, _, _, _, _, _) = calculate_md_by_ods(ods, method='vectorized')

    # Extract the measured and calculated flux loop data
    measured_flux_loop = MG['flux_loop.:.flux.data']
    calculated_flux_loop_temp = psi_coil + psi_eddy
    calculated_flux_loop = np.zeros((nbflux, len(MG['time'])))
    pf_time = np.asarray(ods['pf_active.time'], dtype=float)
    for i in range(nbflux):
        calculated_flux_loop[i, :] = np.interp(MG['time'], pf_time, calculated_flux_loop_temp[i, :])

    # Filter the data between the flux loop onset time and plasma onset time
    fl_onset_idx = (np.abs(MG['time'] - fl_onset)).argmin()
    plas_onset_idx = (np.abs(MG['time'] - onset)).argmin()
    measured_flux_loop = measured_flux_loop[:, fl_onset_idx:plas_onset_idx]
    calculated_flux_loop = calculated_flux_loop[:, fl_onset_idx:plas_onset_idx]

    # Calculate the scaling factor for each flux loop based on least square fitting
    scaling_factor = np.ones(nbflux)
    for i in range(nbflux):
        measured = measured_flux_loop[i]
        calculated = calculated_flux_loop[i]

        denominator = np.dot(measured, calculated)
        if denominator != 0.0:
            scaling_factor[i] = np.dot(measured, measured) / denominator

    return scaling_factor


def vfit_signal_startend(time, data):
    threshold = 0.06  # minimum value
    nbt = len(time)

    # index of maximum value
    indxm = min(range(len(data)), key=lambda i: abs(data[i] - max(data)))
    indxs = -1
    indxe = -1

    # We are looking for windows that constain continue data above threshold.
    # The window we are looking for, must contain the maximum value
    for i in range(nbt):
        if data[i] >= threshold:
            if indxs == -1:
                indxs = i  # start of the window
        else:
            indxe = i - 1  # end of the window
            if indxs < indxm and indxm < indxe:
                break  # if the window contains the maximum, we stop
    #            indxs=-1
    tstart = time[indxs]
    tend = time[indxe]

    return (tstart, tend)

def smooth(array, span):
    if span % 2 == 0:
        span = span - 1

    nbv = len(array)
    out = np.zeros(nbv)
    span2 = int((span - 1) / 2)
    for i in range(span2):
        div = 2 * i + 1
        win = [j for j in range(div)]
        out[i] = np.sum(array[win]) / div
        win2 = [nbv - 1 - j for j in range(div)]
        out[nbv - 1 - i] = np.sum(array[win2]) / div

    endl = nbv - span2
    for i in range(span2, endl):
        div = span
        win = [i - span2 + j for j in range(div)]
        out[i] = np.sum(array[win]) / div

    return out


def vest_rspv1(ods, plasma, rz):
    PF = ods['pf_active']
    PFp = ods['pf_passive']

    nbcoil = len(PF['coil'])
    nbloop = len(PFp['loop'])
    plasma = plasma if plasma is not None else []
    nbplas = len(plasma)
    tot = len(rz)
    Br = np.zeros((tot, nbcoil+nbloop+nbplas))
    Bz = np.zeros((tot, nbcoil+nbloop+nbplas))
    Psi = np.zeros((tot, nbcoil+nbloop+nbplas))

    shft = 0.01
    for i in range(tot):
        r1 = rz[i][0]
        z1 = rz[i][1]

# From coils
        for ii in range(nbcoil):
            nbelti = len(PF['coil.{}.element'.format(ii)])
            sumr = 0.
            sumz = 0.
            sump = 0.
            for jj in range(nbelti):
                nbturnl=PF['coil.{}.element.{}.turns_with_sign'.format(ii,jj)]
                gtype = PF['coil.{}.element.{}.geometry.geometry_type'.format(ii,jj)]
                if gtype == 1:                    
                    myr=PF['coil.{}.element.{}.geometry.outline.r'.format(ii,jj)]
                    myz=PF['coil.{}.element.{}.geometry.outline.z'.format(ii,jj)]
                    r2=sum(myr)/len(myr)
                    z2=sum(myz)/len(myz)
                elif gtype == 2:
                    r2=PF['coil.{}.element.{}.geometry.rectangle.r'.format(ii,jj)]
                    z2=PF['coil.{}.element.{}.geometry.rectangle.z'.format(ii,jj)]
                elif gtype == 3:
                    r2=PF['coil.{}.element.{}.geometry.oblique.r'.format(ii,jj)]
                    z2=PF['coil.{}.element.{}.geometry.oblique.z'.format(ii,jj)]
                elif gtype==5:
                    r2=PF['coil.{}.element.{}.geometry.annulus.r'.format(ii,jj)]
                    z2=PF['coil.{}.element.{}.geometry.annulus.z'.format(ii,jj)]
                elif gtype ==6:
                    r21=PF['coil.{}.element.{}.geometry.thick_line.first_point.r'.format(ii,jj)]
                    r22=PF['coil.{}.element.{}.geometry.thick_line.second_point.r'.format(ii,jj)]
                    r2=(r21+r22)/2
                    z21=PF['coil.{}.element.{}.geometry.thick_line.first_point.z'.format(ii,jj)]
                    z22=PF['coil.{}.element.{}.geometry.thick_line.second_point.z'.format(ii,jj)]
                    z2=(z21+z22)/2

                    
                if calculate_distance(r1,r2,z1,z2) < shft/3.:
                    print(1)
                    (myBr1,myBz1) = green_br_bz(r1+shft,z1,r2,z2)
                    (myBr2,myBz2) = green_br_bz(r1-shft,z1,r2,z2)
                    myP1=green_r(r1+shft,z1,r2,z2)
                    myP2=green_r(r1-shft,z1,r2,z2)
                    myBr=(myBr1+myBr2)/2.
                    myBz=(myBz1+myBz2)/2.
                    myP=(myP1+myP2)/2.
                else:
                    (myBr,myBz) = green_br_bz(r1,z1,r2,z2)
                    myP =  green_r(r1,z1,r2,z2)
                sumr=sumr+myBr*nbturnl
                sumz=sumz+myBz*nbturnl
                sump=sump+myP*nbturnl

            Br[i][ii] = sumr
            Bz[i][ii] = sumz
            Psi[i][ii] = sump

# From wall
        for ii in range(nbloop):
            nbelti = len(PFp['loop.{}.element'.format(ii)])
            sumr = 0.
            sumz = 0.
            sump = 0.
            for jj in range(nbelti):
                gtype = PFp['loop.{}.element.{}.geometry.geometry_type'.format(ii,jj)]
                if gtype == 1:
                    myr=PFp['loop.{}.element.{}.geometry.outline.r'.format(ii,jj)]
                    myz=PFp['loop.{}.element.{}.geometry.outline.z'.format(ii,jj)]
                    r2=sum(myr)/len(myr)
                    z2=sum(myz)/len(myz)
                elif gtype == 2:
                    r2=PFp['loop.{}.element.{}.geometry.rectangle.r'.format(ii,jj)]
                    z2=PFp['loop.{}.element.{}.geometry.rectangle.z'.format(ii,jj)]
                elif gtype == 3:
                    r2=PFp['loop.{}.element.{}.geometry.oblique.r'.format(ii,jj)]
                    z2=PFp['loop.{}.element.{}.geometry.oblique.z'.format(ii,jj)]
                
                if calculate_distance(r1,r2,z1,z2) < shft/3.:
                    print(2)
                    (myBr1, myBz1) = green_br_bz(r1+shft, z1, r2, z2)
                    (myBr2, myBz2) = green_br_bz(r1-shft, z1, r2, z2)
                    myP1 = green_r(r1+shft, z1, r2, z2)
                    myP2 = green_r(r1-shft, z1, r2, z2)
                    myBr = (myBr1+myBr2)/2.
                    myBz = (myBz1+myBz2)/2.
                    myP = (myP1+myP2)/2.
                else:
                    (myBr, myBz) = green_br_bz(r1, z1, r2, z2)
                    myP = green_r(r1, z1, r2, z2)
                sumr = sumr+myBr
                sumz = sumz+myBz
                sump = sump+myP

            Br[i][nbcoil+ii] = sumr
            Bz[i][nbcoil+ii] = sumz
            Psi[i][nbcoil+ii] = sump

# From plasma (if any)
        for ii in range(nbplas):
            r2 = plasma[ii][0]
            z2 = plasma[ii][1]

            if calculate_distance(r1, r2, z1, z2) < shft/3.:
                print(3)
                (myBr1, myBz1) = green_br_bz(r1+shft, z1, r2, z2)
                (myBr2, myBz2) = green_br_bz(r1-shft, z1, r2, z2)
                myP1 = green_r(r1+shft, z1, r2, z2)
                myP2 = green_r(r1-shft, z1, r2, z2)
                myBr = (myBr1+myBr2)/2.
                myBz = (myBz1+myBz2)/2.
                myP = (myP1+myP2)/2.
            else:
                (myBr, myBz) = green_br_bz(r1, z1, r2, z2)
                myP = green_r(r1, z1, r2, z2)

            Br[i][nbcoil+nbloop+ii] = myBr
            Bz[i][nbcoil+nbloop+ii] = myBz
            Psi[i][nbcoil+nbloop+ii] = myP

    return (Psi, Bz, Br)


def calculate_md_by_ods(ods, filament_position = [], filament_fraction = [], method = 'vecterized'):
    # Method
    ## nested_loop : calculate by nested loop
    ## vectorized : calculate by vectorized method (default, faster)

    # Load the magnetics, pf_active, and pf_passive ODS
    MG = ods['magnetics']
    PFP = ods['pf_passive']
    PF = ods['pf_active']

    # Load the Magnetics Position
    probe_rz = [] # (r, z) for each Bz probe points
    nbprobe = len(MG['b_field_pol_probe'])

    for i in range(nbprobe):    
        r = MG['b_field_pol_probe.{}.position.r'.format(i)]
        z = MG['b_field_pol_probe.{}.position.z'.format(i)]
        probe_rz.append([r, z])

    fl_rz = [] # (r, z) for each flux loop points
    nbfl = len(MG['flux_loop'])

    for i in range(nbfl):
        r = MG['flux_loop.{}.position.0.r'.format(i)]
        z = MG['flux_loop.{}.position.0.z'.format(i)]
        fl_rz.append([r, z])

    # Calculate response matrix
    #    if filament_position != []:
    (cpsi, _, _) = vest_rspv1(ods, filament_position, fl_rz)  # flux loop response matrix
    (_, cbz, _) = vest_rspv1(ods, filament_position, probe_rz) # Bz probe response matrix

    # Load and Interpolate Ip to the time of PF
    if filament_position != []:
        Ip_total=np.interp(PF['time'],MG['ip.0.time'],MG['ip.0.data'])
        Ip = np.array([Ip_total * fraction for fraction in filament_fraction])

        
    # Initialize the variables
    nbtime = len(PF['time'])
    nbcoil = len(PF['coil'])
    nbloop = len(PFP['loop'])
    if filament_position != []:
        nbplas = len(Ip)
    else:
        nbplas = 0

    bz_total = np.zeros((nbprobe, nbtime))
    bz_coil = np.zeros((nbprobe, nbtime))
    bz_eddy = np.zeros((nbprobe, nbtime))
    if filament_position != []:
        bz_plas = np.zeros((nbprobe, nbtime))

    psi_total = np.zeros((nbfl, nbtime))
    psi_coil = np.zeros((nbfl, nbtime))
    psi_eddy = np.zeros((nbfl, nbtime))
    if filament_position != []:
        psi_plas = np.zeros((nbfl, nbtime))

    # Calculate the induced magnetic field and poloidal flux quantities by the PF, eddy currents, and filamentry plasma
    if method == 'nested_loop':

        I_coil = np.zeros(nbcoil+nbloop+nbplas)
        I_plas = np.zeros(nbcoil+nbloop+nbplas)
        I_eddy = np.zeros(nbcoil+nbloop+nbplas)

        for k in range(nbtime):
            for i in range(nbcoil):
                I_coil[i] = PF['coil.{}.current.data'.format(i)][k]
            for i in range(nbloop):
                I_eddy[nbcoil+i] = PFP['loop.{}.current'.format(i)][k]
            for i in range(nbplas):
                I_plas[nbcoil+nbloop+i] = Ip[i][k]

            for i in range(nbprobe):
                bz_coil[i][k] = np.matmul(cbz[i], I_coil)
                bz_eddy[i][k] = np.matmul(cbz[i], I_eddy)
                bz_plas[i][k] = np.matmul(cbz[i], I_plas)
                bz_total[i][k] = bz_coil[i][k] + bz_eddy[i][k] + bz_plas[i][k]
        for i in range(nbfl):
                psi_coil[i][k] = np.matmul(cpsi[i], I_coil)
                psi_eddy[i][k] = np.matmul(cpsi[i], I_eddy)
                psi_plas[i][k] = np.matmul(cpsi[i], I_plas)
                psi_total[i][k] = psi_coil[i][k] + psi_eddy[i][k] + psi_plas[i][k]

    elif method == 'vectorized':
        # Initialize current arrays
        I_coil = np.zeros((nbcoil + nbloop + nbplas, nbtime))
        I_eddy = np.zeros((nbcoil + nbloop + nbplas, nbtime))
        I_plas = np.zeros((nbcoil + nbloop + nbplas, nbtime))


        # Assign current values
        print(nbtime,len(PFP['time']))
        I_coil[:nbcoil] = PF['coil.:.current.data']
        I_eddy[nbcoil:nbcoil+nbloop] = PFP['loop.:.current']
        if filament_position != []:
            I_plas[nbcoil+nbloop:] = Ip
        
        # Compute magnetic field contributions
        bz_coil = np.dot(cbz, I_coil)
        bz_eddy = np.dot(cbz, I_eddy)
        if filament_position != []:
            bz_plas = np.dot(cbz, I_plas)
        else:
            bz_plas=0.
        bz_total = bz_coil + bz_eddy + bz_plas

        # Compute poloidal flux contributions
        psi_coil = np.dot(cpsi, I_coil)
        psi_eddy = np.dot(cpsi, I_eddy)
        if filament_position != []:
            psi_plas = np.dot(cpsi, I_plas)
        else:
            psi_plas=0.
        psi_total = psi_coil + psi_eddy + psi_plas

    return psi_total, psi_coil, psi_eddy, psi_plas, bz_total, bz_coil, bz_eddy, bz_plas


def brokenFinder(ods, option=2):
    # Calculate response matrix
    (psi_total, psi_coil, psi_eddy, psi_plas, bz_total, bz_coil, bz_eddy, bz_plas) = calculate_md_by_ods(ods, method='vectorized')


    # Find the plasma onset and offset time
    (onset, offset) = vest_Halpha_tstart_tend(ods)
    bz_calc_time=ods['pf_active.time']
    bz_exp_time=ods['magnetics.time']

    min_time=max(bz_calc_time[0],bz_exp_time[0])
    max_time=min(bz_calc_time[-1],bz_exp_time[-1])

    # phase1: time before plasma
    time1=np.linspace(min_time,onset-0.015,11)
    # phase2: time after plasma
    time2=np.linspace(offset+0.001,max_time,11)

#    print(time1)
#    print(time2)
    
    # list of probes:
    broken=[]

    if option == 1:
        # brokenThreshold
        brokenThreshold = 0.95
        # Bz probes
        for index in range(64):
            bz_calc=bz_coil[index]+bz_eddy[index]
            bz_exp=ods['magnetics'][f'b_field_pol_probe.{index}.field.data']

            #phase 1:
            bz_calc1=np.interp(time1,bz_calc_time,bz_calc)
            bz_exp1=np.interp(time1,bz_exp_time,bz_exp)

            cxy = np.corrcoef(bz_calc1,bz_exp1)
            if cxy[0][1] < brokenThreshold:
                broken.append(index+1)
#                print('1',index+1,cxy[0][1])

            else:
                #phase 2:
                bz_calc2=np.interp(time2,bz_calc_time,bz_calc)
                bz_exp2=np.interp(time2,bz_exp_time,bz_exp)

                cxy = np.corrcoef(bz_calc2,bz_exp2)
                if cxy[0][1] < brokenThreshold:
                    broken.append(index+1)
#                    print('2',index+1,cxy[0][1])

        # Flux loop
        for index in range(11):
            psi_calc=psi_coil[index]+psi_eddy[index]
            psi_exp=ods['magnetics'][f'flux_loop.{index}.flux.data']
        
            psi_calc1=np.interp(time1,bz_calc_time,psi_calc)
            psi_exp1=np.interp(time1,bz_exp_time,psi_exp)

            cxy = np.corrcoef(psi_calc1,psi_exp1)
            if cxy[0][1] < brokenThreshold:
                broken.append(64+index+1)
#                print('1',index+65,cxy[0][1])

            else:
                #phase 2:
                psi_calc2=np.interp(time2,bz_calc_time,psi_calc)
                psi_exp2=np.interp(time2,bz_exp_time,psi_exp)

                cxy = np.corrcoef(psi_calc2,psi_exp2)
                if cxy[0][1] < brokenThreshold:
                    broken.append(64+index+1)
#                    print('2',index+65,cxy[0][1])

    else:
        # brokenThreshold
        brokenThreshold = 0.005 # mV
        # Bz probes
        for index in range(64):
            if index <27:
                brokenThreshold = 0.009
            elif index <48:
                brokenThreshold = 0.005
            else:
                brokenThreshold = 0.007
                
            bz_calc=bz_coil[index]+bz_eddy[index]
            bz_exp=ods['magnetics'][f'b_field_pol_probe.{index}.field.data']

            #phase 1:
            bz_calc1=np.interp(time1,bz_calc_time,bz_calc)
            bz_exp1=np.interp(time1,bz_exp_time,bz_exp)

            # max1=max(abs(bz_calc1))
            # max2=max(abs(bz_exp1))
            # max3=max(max1,max2)

            cxy=abs(bz_calc1-bz_exp1)
            max4=max(cxy)

            if max4 > brokenThreshold:
                broken.append(index+1)
#                print('1',index+1,max4)

            else:
                #phase 2:
                bz_calc2=np.interp(time2,bz_calc_time,bz_calc)
                bz_exp2=np.interp(time2,bz_exp_time,bz_exp)

                # max1=max(abs(bz_calc2))
                # max2=max(abs(bz_exp2))
                # max3=max(max1,max2)

                cxy=abs(bz_calc2-bz_exp2)
                max4=max(cxy)
                
                if max4 > brokenThreshold:
                    broken.append(index+1)
#                    print('2',index+1,max4)

        # Flux loop
        for index in range(11):
            psi_calc=psi_coil[index]+psi_eddy[index]
            psi_exp=ods['magnetics'][f'flux_loop.{index}.flux.data']
        
            psi_calc1=np.interp(time1,bz_calc_time,psi_calc)
            psi_exp1=np.interp(time1,bz_exp_time,psi_exp)

            # max1=max(abs(psi_calc1))
            # max2=max(abs(psi_exp1))
            # max3=max(max1,max2)

            cxy=abs(psi_calc1-psi_exp1)
            max4=max(cxy)

            if max4 > brokenThreshold:
                broken.append(64+index+1)
#                print('1',index+65,max4)

            else:
                #phase 2:
                psi_calc2=np.interp(time2,bz_calc_time,psi_calc)
                psi_exp2=np.interp(time2,bz_exp_time,psi_exp)

                # max1=max(abs(psi_calc2))
                # max2=max(abs(psi_exp2))
                # max3=max(max1,max2)

                cxy=abs(psi_calc2-psi_exp2)
                max4=max(cxy)
                if max4 > brokenThreshold:
                    broken.append(64+index+1)
#                    print('2',index+65,max4)

                
    return broken


def vest_signal_onoffsetpeak(time, data, tstart, tend, threshold):
    """
    Finds the onset, peak, and offset times of the signal within the specified time range
    by normalizing the signal and identifying the time window where the signal exceeds a threshold.

    Parameters:
        time (array): Array of time values corresponding to the data points.
        data (array): Array of data values to analyze.
        tstart (float): Start time of the interval to analyze.
        tend (float): End time of the interval to analyze.
        threshold (float): Minimum value to consider as signal onset (0 ~ 1).

    Returns:
        t_onset (float): Time of signal onset.
        t_peak (float): Time of signal peak.
        t_offset (float): Time of signal offset.
    """
    # Parameters
    smooth_factor = 50

    # Smooth the data using Savitzky-Golay filter
    data_smoothed = savgol_filter(data, smooth_factor, 3)  # window size 50, polynomial order 3
    nbt = len(time)

    # Find indices for the start and end times
    ind_start = np.argmin(np.abs(time - tstart))
    ind_end = np.argmin(np.abs(time - tend))

    # Ensure the specified range is within the data length
    if ind_start < 0 or ind_end > nbt or ind_end <= ind_start:
        raise ValueError('Specified time range is out of bounds.',ind_start,ind_end)

    mini = np.min(data_smoothed[ind_start:ind_end])
    maxi = np.max(data_smoothed[ind_start:ind_end])

    # Normalize the data using the larger absolute value of the minimum or maximum
    if abs(mini) > abs(maxi):
        data_norm = data_smoothed / mini
    else:
        data_norm = data_smoothed / maxi

    # Find the index of the maximum value within the specified range
    indxm = np.argmax(data_norm[ind_start:ind_end]) + ind_start

    indxs = -1
    indxe = -1

    # Look for windows that contain continuous data above the threshold
    # The window must contain the maximum value
    for i in range(ind_start, ind_end):
        if data_norm[i] >= threshold:
            if indxs == -1:
                indxs = i  # Start of the window
        else:
            if indxs != -1:
                indxe = i - 1  # End of the window
                if indxs < indxm < indxe:
                    break  # Stop if the window contains the maximum
                indxs = -1

    # Determine onset, peak, and offset times
    t_onset = time[indxs] if indxs != -1 else 0
    t_peak = time[indxm]
    t_offset = time[indxe] if indxe != -1 else 0

    return t_onset, t_peak, t_offset


def vest_Halpha_tstart_tend(ods):

    # Load the data
    SP = ods['spectrometer_uv']
    data = SP['channel.0.processed_line.0.intensity.data']
    time = SP['time']
    #    (time, data) = vest_load(int(shot), 101)
    data_smoothed = smooth(data, 10)

    # Look for the minimum (Halpha signal negative) between 0.3 and 0.6 s
    indx1 = min(range(len(time)), key=lambda i: abs(time[i] - 0.3))
    indx2 = min(range(len(time)), key=lambda i: abs(time[i] - 0.36))
    mini = min(data_smoothed[indx1:indx2])

    # Normalize data
    if mini != 0:
        data_smoothed = data_smoothed / mini

    #    plot(time[indx1:indx2], data_smoothed[indx1:indx2])

    # Find start and end time
    (tstart, tend) = vfit_signal_startend(time[indx1:indx2], data_smoothed[indx1:indx2])

    return (tstart, tend)


def set_discharge_index(ods):
    """
    Set the time range higher than 20 kA and between 0.3 ~ 0.36 sec with manual tstep.
    If no Ip > 20 kA is found, return time range and 'vacuum'; otherwise return time range and 'plasma'.
    """

    Ip=ods['magnetics.ip.0.data']
    time=ods['magnetics.ip.0.time']
    
    # Define base time range (0.280.38 s)
    base_index = (time >= 0.28) & (time <= 0.38)

    # Filter by both time and current threshold
    valid_index = base_index & (Ip > 20e3)

    # Check if plasma current > 20 kA exists
    if np.any(valid_index):
        status = "plasma"
        selected_index = valid_index
    else:
        status = "vacuum"
        selected_index = base_index

    print('status',status)
    time = time[selected_index]
    Ip = Ip[selected_index]

    return time



def generate_constraints_ods(ods,shotnumber, save_dir, efit_table_dir, time, uncertainty, weighting, broken=[],fit=0, fl_correct_coeff = None,FFCUR=2,PPCUR=2):
    """
    Generate Constraints ODS file such that save_dir/{shotnumber}_constraints.json is created
    """
    # fit = 0: no fitting, only exp. data
    # fit = 1: broken data replaced by gaussian fitted data
    # fit = 2: all data replaced by gaussian fitted data

    # coilset_opt
    # "16_coils" : PF1 - 8 segments + PF5, 6, 9, 10 Upper and Lower Segments
    # "26_coils" : PF1 - 8 segments + PF2-10 Upper and Lower Segments
    
    # Default constraints which are used routinely in VEST
    constraints = [
        'pf_current',
        'bpol_probe',
        'flux_loop',
        'ip',
        'diamagnetic_flux',
        'b_field_tor_vacuum_r',
    ]
    # For later need to develop option to add other constraints (e.g. internal magnetic probe, thomson scattering, etc.)

    ods_tmp = ODS()
    PF = ods_tmp['pf_active']
    PF_orig = ods['pf_active']

    ## (1) PF coil - # 16 coils. It is used for EFIT input, PF1 (CS) Coil is discriesed by 16 coils in EFIT.
    tstart = 0.26
    #    if shotnumber>=43635:
    #        tstart=0.245
    tend = 0.36
    dt = 4e-5

    vfit_pf_active_efit26(PF,PF_orig,shotnumber,tstart,tend,dt)

    for i, _ in enumerate(PF['coil']):
        PF[f'coil.{i}.current.time']=PF['time']
        PF[f'coil.{i}.current.data_error_upper']=abs(uncertainty[0]*PF[f'coil.{i}.current.data'])    

#    print('___',PF['coil.0.current.data'])
#    print('___',PF['coil.0.current.time'])
        
    ## (2) TF coil
    TF=ods['tf']
    TF['b_field_tor_vacuum_r.time']=TF['time']
    TF['b_field_tor_vacuum_r.data_error_upper']=abs(uncertainty[1]*TF['b_field_tor_vacuum_r.data'])

    ## (3) Ip
    MG = ods['magnetics']
    MG['ip.0.data_error_upper']=abs(uncertainty[2]*MG['ip.0.data'])

    ## (4) Diamagnetic flux
    MG['diamagnetic_flux.0.time']=MG['time']
    MG['diamagnetic_flux.0.data_error_upper']=abs(uncertainty[3]*MG['diamagnetic_flux.0.data'])

    ## (5) Poloidal magnetic probe
    Index_inBz = np.where(MG['b_field_pol_probe.:.position.r']<0.09)
    Index_sideBz = np.where(np.abs(MG['b_field_pol_probe.:.position.z']) > 0.8)
    Index_outBz = np.where(MG['b_field_pol_probe.:.position.r']>0.795)
    # convert tuple to array
    valid_bpol_indices = np.array(
        [
            int(index)
            for index, _ in enumerate(MG['b_field_pol_probe'])
            if path_exists(MG, f'b_field_pol_probe.{index}.field.data')
        ],
        dtype=int,
    )
    Index_inBz = np.intersect1d(Index_inBz[0], valid_bpol_indices)
    Index_sideBz = np.intersect1d(Index_sideBz[0], valid_bpol_indices)
    Index_outBz = np.intersect1d(Index_outBz[0], valid_bpol_indices)

#    print('==!==')
#    print(broken)
#    print(Index_inBz)
#    print(Index_sideBz)
#    print(Index_outBz)
#    print('==!==')

    # Position for In, Out, side
    Bzx=[0.54, 0.5, 0.46, 0.42, 0.38, 0.34, 0.3, 0.26, 0.22, 0.16, 0.12, 0.08, 0.04, 0.0, -0.04, -0.08, -0.12, -0.16, -0.22, -0.26, -0.3, -0.34, -0.38, -0.42 ,-0.46, -0.5, -0.54, 0.42, 0.38, 0.34, 0.3, 0.26, 0.22, 0.18, 0.1, 0.06, 0.02, -0.02, -0.06, -0.1, -0.14, -0.18,-0.22, -0.26, -0.3, -0.34, -0.38, -0.42,0.8328, 0.8728, 0.9128, 0.9528, 0.9928, 1.0328,1.0728, 1.1128, -0.8328, -0.8728, -0.9128, -0.9528, -0.9928, -1.0328, -1.0728,-1.1128]
    
    for i in Index_inBz:
        MG[f'b_field_pol_probe.{i}.field.time']=MG['time']
        MG[f'b_field_pol_probe.{i}.field.data_error_upper']=abs(uncertainty[4]*MG[f'b_field_pol_probe.{i}.field.data'])
    for i in Index_sideBz:
        MG[f'b_field_pol_probe.{i}.field.time']=MG['time']
        MG[f'b_field_pol_probe.{i}.field.data_error_upper']=abs(uncertainty[5]*MG[f'b_field_pol_probe.{i}.field.data'])

    for i in Index_outBz:
        MG[f'b_field_pol_probe.{i}.field.time']=MG['time']
        MG[f'b_field_pol_probe.{i}.field.data_error_upper']=abs(uncertainty[6]*MG[f'b_field_pol_probe.{i}.field.data'])

    ## (6) Flux loops
    Index_inFlux = np.where(MG['flux_loop.:.position.0.r'] < 0.15)
    Index_OutFlux = np.where(MG['flux_loop.:.position.0.r'] > 0.5)

    # conduct ad-hoc correction for the flux loop signal
    if fl_correct_coeff is not None:
        for i in range(len(MG['flux_loop'])):
            MG[f'flux_loop.{i}.flux.data'] = MG[f'flux_loop.{i}.flux.data'] / fl_correct_coeff[i]        

    # convert tuple to array
    Index_inFlux = Index_inFlux[0].astype(int)
    Index_OutFlux = Index_OutFlux[0].astype(int)

    for i in Index_inFlux:
        MG[f'flux_loop.{i}.flux.time']=MG['time']
        MG[f'flux_loop.{i}.flux.data_error_upper']=abs(uncertainty[7]*MG[f'flux_loop.{i}.flux.data'])

    for i in Index_OutFlux:
        MG[f'flux_loop.{i}.flux.time']=MG['time']
        MG[f'flux_loop.{i}.flux.data_error_upper']=abs(uncertainty[8]*MG[f'flux_loop.{i}.flux.data'])

    # Convert diagnostics ODS to equilibrium constraints ODS

    default_average=(time[1]-time[0])/2 # Diagnostics data of each time point in equilibrium constraints ODS is the average of the +- 0.5*timestep
    #    default_average=0.0002
    default_average=0.0005

    EQ = ods['equilibrium']
    EQtime=EQ['time']
    
#    vfit_equilibrium_form_constraints(EQ,PF,MG,TF,EQtime,constraints,default_average)
    vfit_equilibrium_form_constraints(EQ,PF,MG,TF,time,constraints,default_average)

    PFP=ods['pf_passive']
    nbloop=len(PFP['loop'])
    nbprobe=len(valid_bpol_indices)
    PM=ods['equilibrium.code.parameters']

    # Ad hoc diamaagnetic flux constraint sign change 
    EQ['time_slice.0.constraints.diamagnetic_flux.measured'] = abs(EQ['time_slice.0.constraints.diamagnetic_flux.measured'])

    # convert one-based index to zero-based index 
    broken = [i-1 for i in broken]

    # Add weight and validity (not broken) to the constraints - [pf_coil, tf_coil, pf_passive, ip, dia_flux, inboard_bz, side_bz, outboard_bz, inboard_fl, outboard_fl]
    IPLIM=45000
    x0=[0.1,0.,0.2,-0.1]
    x0m=[-0.1,0.,0.2,-0.1]
    x3=[0.1,0.2,-0.1]
    x3m=[-0.1,0.2,-0.1]
    icpt=0
    for i in range(len(EQ['time'])):
        # PF coil
        for j in range(len(EQ[f'time_slice.{i}.constraints.pf_current'])):
            EQ[f'time_slice.{i}.constraints.pf_current.{j}.weight']=weighting[0]
        # Ip
        EQ[f'time_slice.{i}.constraints.ip.weight']=weighting[1]
        IP=EQ[f'time_slice.{i}.constraints.ip.measured']
        # Diamagnetic flux
        EQ[f'time_slice.{i}.constraints.diamagnetic_flux.weight']=weighting[2]

        # Inboard Bz
        print(EQ['time'][i],IP)
        if fit >0 and IP>IPLIM: # gaussian fitting
            x=[]
            y=[]
            for j in Index_inBz:
                if j not in broken:
                    x.append(Bzx[j])
                    y.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
            res=optimize.minimize(min_gauss_fit4,x0,args=(x,y),method='SLSQP',tol=1.e-8,options={'maxiter':1000})
            print('in msg',res.message)
            coef=res.x

            xinBz=[]
            yinBz=[]
            yinBzg=[]
            xinBzb=[]
            yinBzb=[]
            for j in Index_inBz:
                xinBz.append(Bzx[j])
                yinBz.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
                gau=gauss_fit4(coef,Bzx[j])
                yinBzg.append(gau)
                if j in broken:
                    xinBzb.append(Bzx[j])
                    yinBzb.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
                
            if fit ==1: # only broken
                for j in Index_inBz:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[3]
                    if j in broken:
                        gau=gauss_fit4(coef,Bzx[j])
                        EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured']=gau

            else: # all data
                for j in Index_inBz:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[3]
                    gau=gauss_fit4(coef,Bzx[j])
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured']=gau
        else: # no fitting
            for j in Index_inBz:
                if j not in broken:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[3]
                else:               
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=0            
        # Side Bz
        if fit > 0 and IP>IPLIM:
            x=[]
            y=[]
            for j in Index_sideBz:
                if j not in broken:
                    x.append(Bzx[j])
                    y.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
            res=optimize.minimize(min_gauss_fit4,x0m,args=(x,y),method='SLSQP',tol=1.e-8,options={'maxiter':1000})
            coef1=res.x
            res=optimize.minimize(min_gauss_fit4,x0,args=(x,y),method='SLSQP',tol=1.e-8,options={'maxiter':1000})
            coef2=res.x
            if min_gauss_fit4(coef1,x,y) < min_gauss_fit4(coef2,x,y):
                coef=coef1
            else:
                coef=coef2

            xsideBz=[]
            ysideBz=[]
            ysideBzg=[]
            xsideBzb=[]
            ysideBzb=[]
            for j in Index_sideBz:
                xsideBz.append(Bzx[j])
                ysideBz.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
                gau=gauss_fit4(coef,Bzx[j])
                ysideBzg.append(gau)
                if j in broken:
                    xsideBzb.append(Bzx[j])
                    ysideBzb.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])

                
            if fit ==1: # only broken
                for j in Index_sideBz:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[4]
                    if j in broken:
                        gau=gauss_fit4(coef,Bzx[j])
                        EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured']=gau

            else: # all data
                for j in Index_sideBz:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[4]
                    gau=gauss_fit4(coef,Bzx[j])
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured']=gau
        else: # no fitting     
            for j in Index_sideBz:
                if j not in broken:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[4]
                else:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=0

        # Outboard Bz
        if fit > 0 and IP>IPLIM:
            x=[]
            y=[]
            for j in Index_outBz:
                if j not in broken:
                    x.append(Bzx[j])
                    y.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
            res=optimize.minimize(min_gauss_fit4,x0,args=(x,y),method='SLSQP',tol=1.e-8,options={'maxiter':1000})
            coef=res.x

            xoutBz=[]
            youtBz=[]
            youtBzg=[]
            xoutBzb=[]
            youtBzb=[]
            for j in Index_outBz:
                xoutBz.append(Bzx[j])
                youtBz.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])
                gau=gauss_fit4(coef,Bzx[j])
                youtBzg.append(gau)
                if j in broken:
                    xoutBzb.append(Bzx[j])
                    youtBzb.append(EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured'])

            if fit ==1: # only broken
                for j in Index_outBz:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[5]
                    if j in broken:
                        gau=gauss_fit4(coef,Bzx[j])
                        EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured']=gau

            else: # all data
                for j in Index_outBz:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[5]
                    gau=gauss_fit4(coef,Bzx[j])
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.measured']=gau
        else: # no fitting     
            for j in Index_outBz:
                if j not in broken:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=weighting[5]
                else:
                    EQ[f'time_slice.{i}.constraints.bpol_probe.{j}.weight']=0
        
        # Inboard flux loop
        for j in Index_inFlux:
            if (j + nbprobe) not in broken:
                EQ[f'time_slice.{i}.constraints.flux_loop.{j}.weight']=weighting[6]
            else:
                EQ[f'time_slice.{i}.constraints.flux_loop.{j}.weight']=0
        # Outboard flux loop
        for j in Index_OutFlux:
            if (j + nbprobe) not in broken:
                EQ[f'time_slice.{i}.constraints.flux_loop.{j}.weight']=weighting[7]
            else:
                EQ[f'time_slice.{i}.constraints.flux_loop.{j}.weight']=0

        # if fit > 0 and IP>IPLIM: # plot all graphs
        #     fig2, axs = plt.subplots(1, 3, layout="constrained")
        #     axs[ 0].scatter(xinBz, yinBz, label='exp.',c='black')
        #     axs[ 0].scatter(xinBz, yinBzg, label='gauss',c='red')
        #     if len(xinBzb) > 0:
        #         axs[ 0].scatter(xinBzb, yinBzb, label='broken',c='blue')
        #     axs[ 0].set_title("Bzin")
        #     for j, txt in enumerate(Index_inBz):
        #         axs[0].text(xinBz[j], yinBz[j], f'{int(txt)+1}', fontsize=8, color='black')
        #     axs[ 1].scatter(xsideBz, ysideBz,c='black')
        #     axs[ 1].scatter(xsideBz, ysideBzg, c='red')
        #     if len(xsideBzb) > 0:
        #         axs[ 1].scatter(xsideBzb, ysideBzb, c='blue')            
        #     axs[ 1].set_title("Bzside")
        #     loctime=EQ['time'][i]
        #     plt.xlabel(f'{loctime}')

        #     for j, txt in enumerate(Index_sideBz):
        #         axs[1].text(xsideBz[j], ysideBz[j], f'{int(txt)+1}', fontsize=8, color='black')
        #     axs[ 2].scatter(xoutBz, youtBz, c='black')
        #     axs[ 2].scatter(xoutBz, youtBzg, c='red')
        #     if len(xoutBzb) > 0:
        #         axs[ 2].scatter(xoutBzb, youtBzb, c='blue')            
        #     axs[ 2].set_title("Bzout")
        #     for j, txt in enumerate(Index_outBz):
        #         axs[2].text(xoutBz[j], youtBz[j], f'{int(txt)+1}', fontsize=8, color='black')
        #     axs[0].legend()

        #     plt.savefig(os.path.join(save_dir, 'plots', f'{shotnumber}_gfit_{icpt}.png'))
        #     icpt=icpt+1
        #     plt.close(fig2)

#    make_gif(os.path.join(save_dir, 'plots'), f'{shotnumber}_gfit')

    # add namelist parameters in the k-file
    for i in range(len(EQ['time'])):
        mytime=EQ['time'][i]
        print(i,mytime)

        PM[f'time_slice.{i}.IN1.INPUT_DIR']=efit_table_dir
        PM[f'time_slice.{i}.IN1.TABLE_DIR']=efit_table_dir

        PM[f'time_slice.{i}.IN1.IECURR']=0
        PM[f'time_slice.{i}.IN1.KFFCUR']=FFCUR
        PM[f'time_slice.{i}.IN1.KPPCUR']=PPCUR
        PM[f'time_slice.{i}.IN1.KFFFNC']=0
        PM[f'time_slice.{i}.IN1.KPPFNC']=0
        PM[f'time_slice.{i}.IN1.SERROR']=0.05
        PM[f'time_slice.{i}.IN1.IVESEL']=1
        PM[f'time_slice.{i}.IN1.IFITVS']=0
        PM[f'time_slice.{i}.IN1.FCURBD']=1
        PM[f'time_slice.{i}.IN1.PCURBD']=1
        PM[f'time_slice.{i}.IN1.KCALPA']=0
        PM[f'time_slice.{i}.IN1.KCGAMA']=0
        PM[f'time_slice.{i}.IN1.CUTIP']=50000.
        PM[f'time_slice.{i}.IN1.RZERO']=0.4
        PM[f'time_slice.{i}.IN1.RELIP']=0.4
        PM[f'time_slice.{i}.IN1.AELIP']=0.3
        PM[f'time_slice.{i}.IN1.EELIP']=1.6

        # Add wall eddy current to the k-file
        Iwall=[]
        for j in range(nbloop):
            Iwall.append(np.interp(mytime, PFP['time'],PFP[f'loop.{j}.current']))
        PM[f'time_slice.{i}.IN1.VCURRT']=Iwall

        # Add coil geometry to the k-file

        nbcoil1 = 8
        coilset_opt="26_coils"
        if coilset_opt == "16_coils":
            nbcoil = nbcoil1+8
            nbcons= nbcoil1+(nbcoil-nbcoil1)/2
        elif coilset_opt == "26_coils":
            nbcoil = nbcoil1+18
            nbcons= int(nbcoil1+(nbcoil-nbcoil1)/2)
            # add extra constraint if needed: PF2U=0, PF2L=0,PF3U=0, PF3L=0,PF4U=0, PF4L=0,PF7U=0, PF7L=0,PF8U=0, PF8L=0
            add0=[]
            for i in range(nbcoil):
                if statistics.mean(PF[f'coil.{i}.current.data']) == 0.:
                    add0.append(i)
            nbcons= nbcons + int(len(add0)/2)

        PM[f'time_slice.{i}.INWANT.NCCOIL']=0
        PM[f'time_slice.{i}.INWANT.KCCOILS']=nbcons
        PM[f'time_slice.{i}.INWANT.XCOILS']=np.zeros(nbcons)

        CCOILS=np.zeros((nbcoil,nbcons))
        for j in range(nbcoil1-1):
            CCOILS[0][j]=1.
            CCOILS[j+1][j]=-1.
        
        if coilset_opt == "16_coils":
            # Upper and lower is same
            CCOILS[nbcoil1][nbcoil1-1]=1.
            CCOILS[nbcoil1+1][nbcoil1-1]=-1.
            CCOILS[nbcoil1+2][nbcoil1]=1. 
            CCOILS[nbcoil1+3][nbcoil1]=-1. 
            CCOILS[nbcoil1+4][nbcoil1+1]=1. 
            CCOILS[nbcoil1+5][nbcoil1+1]=-1. 
            CCOILS[nbcoil1+6][nbcoil1+2]=1. 
            CCOILS[nbcoil1+7][nbcoil1+2]=-1. 
            # PF 9 = PF 10 is same
            CCOILS[nbcoil1+5][nbcoil1+3]=1. 
            CCOILS[nbcoil1+6][nbcoil1+3]=-1.
        elif coilset_opt == "26_coils":
            # Upper and lower is same
            CCOILS[nbcoil1][nbcoil1-1]=1.
            CCOILS[nbcoil1+1][nbcoil1-1]=-1.
            CCOILS[nbcoil1+2][nbcoil1]=1. 
            CCOILS[nbcoil1+3][nbcoil1]=-1. 
            CCOILS[nbcoil1+4][nbcoil1+1]=1. 
            CCOILS[nbcoil1+5][nbcoil1+1]=-1. 
            CCOILS[nbcoil1+6][nbcoil1+2]=1. 
            CCOILS[nbcoil1+7][nbcoil1+2]=-1. 
            CCOILS[nbcoil1+8][nbcoil1+3]=1.
            CCOILS[nbcoil1+9][nbcoil1+3]=-1.
            CCOILS[nbcoil1+10][nbcoil1+4]=1.
            CCOILS[nbcoil1+11][nbcoil1+4]=-1.
            CCOILS[nbcoil1+12][nbcoil1+5]=1.
            CCOILS[nbcoil1+13][nbcoil1+5]=-1.
            CCOILS[nbcoil1+14][nbcoil1+6]=1.
            CCOILS[nbcoil1+15][nbcoil1+6]=-1.
            CCOILS[nbcoil1+16][nbcoil1+7]=1.
            CCOILS[nbcoil1+17][nbcoil1+7]=-1.
            # PF 9 = PF 10 is same
            CCOILS[nbcoil1+15][nbcoil1+8]=1. 
            CCOILS[nbcoil1+16][nbcoil1+8]=-1.

            # force 0 for unused coils
            for j in range(int(len(add0)/2)):
                CCOILS[add0[2*j]][nbcoil1+9+j] = 1.
        PM[f'time_slice.{i}.INWANT.CCOILS']=CCOILS    

    # Save the constraints ODS
    ods_eq = ODS()
    ods_eq['equilibrium'] = EQ
    fullfilename = os.path.join(save_dir, f'{shotnumber}_constraints.json')
    save_omas_json(ods_eq, fullfilename)


    
def generate_kfile(ods, shotnumber, npprime, nffprime, save_dir='./tmp'):
    """
    Generate k-files under ``save_dir/kfile`` for the requested shot.
    """

    # Load the constraints ODS
    EQ = ods['equilibrium']
    time = EQ['time']
    PM = ods['equilibrium.code.parameters']

    # Define the kfile parameters
    vbit = 10
    shft = 10000.0
    TABLE_DIR = "TABLE_DIR = '{}' \n".format(PM['time_slice.0.IN1.INPUT_DIR'])
    INPUT_DIR = "INPUT_DIR = '{}' \n".format(PM['time_slice.0.IN1.INPUT_DIR'])

    def _machine_count(name, default):
        mhdin = Path(PM['time_slice.0.IN1.INPUT_DIR']).expanduser() / "mhdin.dat"
        if not mhdin.exists():
            return default
        match = re.search(rf"\b{name}\s*=\s*(\d+)", mhdin.read_text(encoding="utf-8", errors="ignore"), re.IGNORECASE)
        return int(match.group(1)) if match else default

    def _namelist_array(name, values, per_line=4, formatter=str):
        chunks = [f"{name}= "]
        for idx, value in enumerate(values, start=1):
            chunks.append(f"{formatter(value)}, ")
            if idx % per_line == 0:
                chunks.append("\n ")
        return "".join(chunks).rstrip(" \n,") + "\n"

    def _efit16_indices(count: int, target: int) -> list[int]:
        if target == 16 and count >= 26:
            return [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 22, 23, 24, 25]
        return list(range(min(count, target)))

    # find the maximum decimal places in the time
    #    for time_idx, _ in enumerate(time):
    #        if time_idx == 0:
    #            digit = get_decimal_places(time[time_idx])
    #        else:
    #            if digit < get_decimal_places(time[time_idx]):
    #                digit = get_decimal_places(time[time_idx])

    for time_idx, _ in enumerate(time):
        # Load the diagnostics constraints data and convert them to kfile parameters
        CSTR = EQ[f'time_slice.{time_idx}.constraints']

        ## (1) PF Coil currents with weight
        nfsum = _machine_count("nfsum", len(CSTR['pf_current']))
        pf_indices = _efit16_indices(len(CSTR['pf_current']), nfsum)
        nbcoil = len(pf_indices)
        COILCURRENT = np.zeros(nbcoil)
        BITCURRENT = np.zeros(nbcoil)
        for i, source_index in enumerate(pf_indices):
            COILCURRENT[i] = CSTR[f'pf_current.{source_index}.measured']  # Coil current in A
            BITCURRENT[i] = CSTR[f'pf_current.{source_index}.measured_error_upper']  # Standard deviation of coil current measurement data in A
        BRSP = _namelist_array('BRSP', COILCURRENT, per_line=3)
        BITFC = _namelist_array('BITFC', BITCURRENT, per_line=3)
        FWTFC = f'FWTFC= 16*{CSTR[f"pf_current.0.weight"]}\n'  # Fitting weight

        ## (2) Wall eddy current
        WALLCURRENT = PM[f'time_slice.{time_idx}.IN1.VCURRT']  # Wall current array in A
        nbloop = len(WALLCURRENT)

        VCURRT = _namelist_array('VCURRT', WALLCURRENT, per_line=4)

        ## (3) Toroidal magnetic field (TF coil)
        RCENTR = 0.4
        BTOR = CSTR['b_field_tor_vacuum_r.measured'] / RCENTR

        ## (4) Plasma current with weight
        PLASMA = f'PLASMA= {CSTR["ip.measured"]}'  # Plasma current in A
        BITIP = f'BITIP= {CSTR["ip.weight"]/vbit*shft}'
        FWTCUR = f'FWTCUR= {CSTR["ip.weight"]}'  # Fitting weight

        ## (5) Diamagnetic flux with weight
        DFLUX = 'DFLUX= '
        VAL = abs(CSTR['diamagnetic_flux.measured'] * 1000)  # diamagnetic flux in mWb
#        VAL = -VAL  # diamagnetic flux is negative to positive
        DFLUX = DFLUX + f'{VAL} \n'

        ## Original SIGDLC is written as the standard deviation but we use the fitting weight instead
        SIGDLC = f'SIGDLC= {CSTR["diamagnetic_flux.weight"]*shft*1000.}'  # Fitting weights of diamagnetic flux measurement data in mWb
        # SIGDLC=f'SIGDLC= {CSTR["diamagnetic_flux.measured_error_upper"]*1000}' # Standard deviation of diamagnetic flux measurement data in mWb
        # SIGDLC=f'SIGDLC= {VAL*CSTR["diamagnetic_flux.weight"]}' # set sigdlc as measured value * weight

        if (
            CSTR['diamagnetic_flux.weight'] == 0
        ):  # if the diamagnetic flux weight is 0, the diamagnetic flux is not considered as a constraint
            FWTDLC = 'FWTDLC= 0'
        else:
            FWTDLC = 'FWTDLC= 1'

        ## (6) Poloidal magnetic probe with weight
        nbprobe = len(CSTR['bpol_probe'])
        EXPMP2 = _namelist_array('EXPMP2', [CSTR[f"bpol_probe.{i}.measured"] for i in range(nbprobe)], per_line=3)
        fwtmp2_values = []
        bitmpi_values = []
        for i in range(nbprobe):
            if CSTR[f'bpol_probe.{i}.weight'] == 0:  # if the probe is broken
                fwtmp2_values.append(0)
                bitmpi_values.append(0.0)
            else:
                poids = CSTR[f'bpol_probe.{i}.weight']
                poids = poids / vbit * shft
                fwtmp2_values.append(1)
                bitmpi_values.append(poids)
        FWTMP2 = _namelist_array('FWTMP2', fwtmp2_values, per_line=32)
        BITMPI = _namelist_array('BITMPI', bitmpi_values, per_line=3, formatter=lambda value: f'{value:.3f}')

        ## (4) Flux loops
        nbfl = len(CSTR['flux_loop'])
        COILS = _namelist_array(
            'COILS',
            [CSTR[f'flux_loop.{i}.measured'] / 2 / np.pi for i in range(len(CSTR['flux_loop']))],
            per_line=3,
        )

        fwtsi_values = []
        psibit_values = []
        for i in range(nbfl):
            if CSTR[f'flux_loop.{i}.weight'] == 0:  # if the flux loop is broken
                fwtsi_values.append(0)
                psibit_values.append(0.0)
            else:
                poids = CSTR[f'flux_loop.{i}.weight']
                poids = poids / vbit * shft
                fwtsi_values.append(1)
                psibit_values.append(poids)
        FWTSI = _namelist_array('FWTSI', fwtsi_values, per_line=32)
        PSIBIT = _namelist_array('PSIBIT', psibit_values, per_line=3, formatter=lambda value: f'{value:.3f}')

        TTIME = str(int(np.round(time[time_idx], 4) * 1e6))
        #        print('TTIME=',TTIME)
        ITIME = TTIME[0:3]
        UTIME = TTIME[3:]
        #        print(ITIME,UTIME)

        filename = f'k0{shotnumber}.00{time[time_idx]*1000.:.0f}'  # 0.305 -> 305
        if UTIME != '000':
            filename = f'k0{shotnumber}.00{ITIME}_{UTIME}'  # 0.3051 -> 305_100

        # Write the kfile
        #        filename=f'k0{shotnumber}.00{time[time_idx]*1e+5:.0f}'
        #        filename=f'k0{shotnumber}.00{time[time_idx]*1e+3:.0f}' # 0.305 -> 305
        # filename=f'k0{shotnumber}.00{time[time_idx]*10**digit:.0f}' # 0.305 -> 305
        print(f'filename: {filename}')
        fullfile = os.path.join(save_dir, 'kfile', filename)
        # make the kfile directory if it does not exist
        if not os.path.exists(os.path.join(save_dir, 'kfile')):
            os.makedirs(os.path.join(save_dir, 'kfile'))
        f = open(fullfile, 'w')
        f.write(' &IN1\n')  # the main namelist in the kfile
        f.write(' IOUT=4\n')  # write one measurement file for each slice in m0sssss.ttttt
        f.write(' AELIP = 0.3\n')  # Minor radius in meters of ellipse for current initialization
        f.write(
            ' CUTIP = 5000.0\n'
        )  # If Ip is less than CUTIP = 5 kA, the plasma is not initialized and conduct only the vacuum calculation
        f.write(' EELIP = 1.6\n')  # Initial elongation of the plasma
        f.write(
            ' FCURBD = 1\n'
        )  # FF' Boundary condition (1 means force FF'=0 at the boundary, if 0, FF' have a finite value at the boundary - H-mode)
        f.write(' IECURR = 0\n')  # 0 means that the Ohmic coil flag is ignored (Not classify 5
        f.write(' IFITVS = 0\n')
        # f.write(' ERRMIN = 1.0e-3\n') # Minimum relative error for the fitting (Default is 1.0e-2 - fitted < 1 min, if 1.0e-3 - fitted < 3 min)
        # f.write(' SAICON = 60.0\n') # Minimum chi2 error for the fitting (Default is 80.0)
        f.write(INPUT_DIR)
        f.write(' IVESEL = 1\n')
        f.write(' KCALPA = 0\n')
        f.write(' KCGAMA = 0\n')
        f.write(f' KFFCUR = {nffprime}\n')
        f.write(' KFFFNC = 0\n')
        f.write(f' KPPCUR = {npprime}\n')
        f.write(' KPPFNC = 0\n')
        f.write(' PCURBD = 1\n')
        f.write(' RELIP = 0.4\n')
        f.write(' RZERO = 0.4\n')
        f.write(' SERROR = 0.0005\n')
        f.write(TABLE_DIR)
        f.write(VCURRT)
        f.write('\n')
        f.write(f' RCENTR = {RCENTR}\n')
        f.write(f' ISHOT = {shotnumber}\n')
        f.write(f' ITIME = {int(round(time[time_idx]*1000.))}\n')
        # if digit > 3: # Add ITIMEU (microsecond) if digit is greater than 3
        #     f.write(f' ITIMEU = {(time[time_idx]*1000-int(time[time_idx]*1000))*1000}\n')
        f.write(BRSP)
        f.write('\n')
        f.write(BITFC)
        f.write('\n')
        f.write(FWTFC)
        f.write(EXPMP2)
        f.write('\n')
        f.write(BITMPI)
        f.write('\n')
        f.write(COILS)
        f.write('\n')
        f.write(PSIBIT)
        f.write('\n')
        f.write(FWTSI)
        f.write('\n')
        f.write(FWTMP2)
        # f.write('\n')
        f.write(PLASMA)
        f.write('\n')
        f.write(BITIP)
        f.write('\n')
        f.write(FWTCUR)
        f.write('\n')
        f.write(DFLUX)
        f.write(SIGDLC)
        f.write('\n')
        f.write(FWTDLC)
        f.write('\n')
        f.write(f' BTOR = {BTOR}\n')

        f.write(' RELAX = 1.\n')  # No relaxation
        # f.write(' RELAX = 0.8\n') # Relaxation (It says it can help the convergence in document, but it seems to not work well, even worse)
        # f.write(' RELAX = 0.5\n') # Backaveraging (It says it can help the convergence in document, but it seems to not work well, even worse)

        f.write(' ERROR = 1e-05\n')
        f.write(' MXITER = -100\n')
        f.write(' NBDRY = 0\n')
        f.write(' /\n')
        f.write(' &INWANT\n')
        f.write(' CCOILS(1,1) = 1.0 -1.0 14*0.0\n')
        f.write(' CCOILS(1,2) = 1.0 0.0 -1.0 13*0.0\n')
        f.write(' CCOILS(1,3) = 1.0 2*0.0 -1.0 12*0.0\n')
        f.write(' CCOILS(1,4) = 1.0 3*0.0 -1.0 11*0.0\n')
        f.write(' CCOILS(1,5) = 1.0 4*0.0 -1.0 10*0.0\n')
        f.write(' CCOILS(1,6) = 1.0 5*0.0 -1.0 9*0.0\n')
        f.write(' CCOILS(1,7) = 1.0 6*0.0 -1.0 8*0.0\n')
        f.write(' CCOILS(1,8) = 8*0.0 1.0 -1.0 6*0.0\n')
        f.write(' CCOILS(1,9) = 10*0.0 1.0 -1.0 4*0.0\n')
        f.write(' CCOILS(1,10) = 12*0.0 1.0 -1.0 2*0.0\n')
        f.write(' CCOILS(1,11) = 14*0.0 1.0 -1.0\n')
        f.write(' CCOILS(1,12) = 13*0.0 1.0 -1.0 0.0\n')
        f.write(' KCCOILS = 12\n')
        f.write(' NCCOIL = 0\n')
        f.write(' XCOILS = 12*0.0\n')
        f.write(' /\n')
        f.write('                                            MAG\n')
        f.close()

    
def gfile_to_omas(self, ods=None, time_index=0, profile_index=0, allow_derived_data=True):
    """
    translate gEQDSK class to OMAS data structure

    :param ods: input ods to which data is added

    :param time_index: time index to which data is added

    :param allow_derived_data: bool
        Populate simple derived equilibrium quantities when available.

    :return: ODS
    """
    from vaft.data.eqdsk import to_omas as _geqdsk_to_omas

    return _geqdsk_to_omas(
        self,
        ods=ods,
        time_index=time_index,
        profile_index=profile_index,
        allow_derived_data=allow_derived_data,
    )
