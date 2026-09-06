"""Magnetic (routine) EFIT adapter: config, prepare/run/collect, gfile conversion.

Moved verbatim out of the former monolithic ``efit.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ...compat import is_executable, resolve_executable
from .._executables import executable_from_home, missing_home_message
from .status import (
    EFITSliceStatus,
    EFITValidationConfig,
    apply_temporal_continuity,
    validate_efit_slice,
)
from .config import (
    EFITConstraintConfig,
    EFITInitializationConfig,
    EFITNumericsConfig,
    EFITProfileConfig,
    EFITScientificConfig,
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
    profile: EFITProfileConfig | None = None
    initialization: EFITInitializationConfig = field(
        default_factory=EFITInitializationConfig
    )
    numerics: EFITNumericsConfig = field(default_factory=EFITNumericsConfig)
    constraints: EFITConstraintConfig = field(default_factory=EFITConstraintConfig)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("npprime", "nffprime"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.profile is not None:
            if self.npprime not in (2, self.profile.kppcur):
                raise ValueError(
                    "npprime conflicts with profile.kppcur; use the typed profile only"
                )
            if self.nffprime not in (2, self.profile.kffcur):
                raise ValueError(
                    "nffprime conflicts with profile.kffcur; use the typed profile only"
                )
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.stack_size_kb is not None and self.stack_size_kb <= 0:
            raise ValueError("stack_size_kb must be greater than zero")
        if self.times is not None and any(
            not math.isfinite(float(value)) for value in self.times
        ):
            raise ValueError("EFIT times must be finite")

    def scientific_config(self) -> EFITScientificConfig:
        """Return the fully resolved scientific configuration.

        The historical ``npprime`` and ``nffprime`` fields are honored when a
        typed profile configuration was not supplied.
        """
        profile = self.profile or EFITProfileConfig(
            kppcur=self.npprime,
            kffcur=self.nffprime,
        )
        return EFITScientificConfig(
            profile=profile,
            initialization=self.initialization,
            numerics=self.numerics,
            constraints=self.constraints,
        )


@dataclass
class EFITInputs:
    """Input bundle for an EFIT run."""

    workdir: Path
    ods: Any = None
    geqdsk: Any = None
    kfiles: tuple[Path, ...] = ()
    files: tuple[Path, ...] = ()
    configuration: Mapping[str, Any] = field(default_factory=dict)
    manifest: Path | None = None


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
    keqdsk: tuple[Any, ...] = ()
    meqdsk: tuple[Any, ...] = ()
    parse_errors: tuple[str, ...] = ()
    mapping_diagnostics: tuple[Mapping[str, Any], ...] = ()
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    ods: Any = None
    status: str = "completed"
    reason: str = ""
    slice_statuses: tuple[EFITSliceStatus, ...] = ()
    configuration: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0

    @property
    def usable(self) -> bool:
        """Whether at least one collected slice is scientifically usable."""
        return any(status.usable for status in self.slice_statuses)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolved_efit_configuration(config: EFITConfig) -> dict[str, Any]:
    """Return the canonical JSON-compatible configuration for an EFIT run."""
    scientific = config.scientific_config()
    return {
        "schema_version": 1,
        "scientific": scientific.to_dict(),
        "scientific_sha256": scientific.sha256,
        "execution": {
            "shot": int(config.shot) if config.shot is not None else None,
            "times": (
                [float(value) for value in config.times]
                if config.times is not None
                else None
            ),
            "args": [str(value) for value in config.args],
            "timeout": float(config.timeout) if config.timeout is not None else None,
            "stack_size_kb": config.stack_size_kb,
            "requested_executable": (
                str(config.executable) if config.executable is not None else None
            ),
        },
        "provenance": _json_value(config.provenance),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vaft_revision() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _write_efit_configuration_manifest(
    config: EFITConfig,
    kfiles: Sequence[Path],
    destination: Path,
) -> Path:
    from vaft.version import __version__

    resolved = resolved_efit_configuration(config)
    payload = {
        "requested": {
            "legacy_profile_orders": {
                "npprime": config.npprime,
                "nffprime": config.nffprime,
            },
            "typed_profile_supplied": config.profile is not None,
            "scientific": resolved["scientific"],
            "execution": resolved["execution"],
            "provenance": resolved["provenance"],
        },
        "resolved": resolved,
        "vaft_version": __version__,
        "vaft_revision": _vaft_revision(),
        "kfiles": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for path in sorted(Path(path) for path in kfiles)
        ],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


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
    # Workflow reruns leave staged copies in ``<prefix>file/`` while EFIT writes
    # a fresh artifact with the same basename in ``workdir``.  Treat those as
    # one artifact (preferring the fresh root copy); genuinely different names
    # for the same case are still rejected later by ``_case_file_map``.
    candidates: dict[str, Path] = {}
    search_roots = [workdir / f"{prefix}file", workdir]
    pattern = f"{prefix}0{shot}.*" if shot is not None else f"{prefix}*"
    for root in search_roots:
        if root.exists():
            candidates.update(
                (path.name, path) for path in root.glob(pattern) if path.is_file()
            )
    return tuple(sorted(candidates.values()))


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
        requested = env_candidate / "efit" if env_candidate.is_dir() else env_candidate
        return resolve_executable(requested) or requested
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
    if exe is not None and is_executable(exe):
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

    # Lazy import: .kfile imports EFITConfig from this module, so a top-level
    # import here would be circular.
    from .kfile import generate_kfile

    generate_kfile(
        ods,
        shot,
        config.npprime,
        config.nffprime,
        save_dir=str(workdir),
        config=config,
    )
    kfiles = _find_outputs(workdir, "k", shot)
    manifest = _write_efit_configuration_manifest(
        config,
        kfiles,
        workdir / "efit_configuration.json",
    )
    return EFITInputs(
        workdir=workdir,
        ods=ods,
        kfiles=kfiles,
        files=(*kfiles, manifest),
        configuration=resolved_efit_configuration(config),
        manifest=manifest,
    )


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
        return _skipped_efit_result(
            inputs,
            config,
            reason=_efit_unconfigured_reason(),
        )
    if not is_executable(executable):
        reason = f"missing executable: {executable}"
        return _skipped_efit_result(
            inputs,
            config,
            reason=reason,
            executable=executable,
        )
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
            # A foreign program's bytes; see the note in vaft.code.chease.
            encoding="utf-8",
            errors="replace",
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
    name = (
        path.name[1:]
        if path.name[:1].lower() in {"k", "g", "a", "m"}
        else path.name
    )
    if name.lower().endswith(".nc"):
        name = name[:-3]
    try:
        shot, suffix = name.rsplit(".", 1)
        return f"{shot}.{int(suffix)}"
    except (ValueError, TypeError):
        return name


def _case_file_map(paths: Sequence[Path], kind: str) -> dict[str, Path]:
    """Build a one-file-per-case map and reject ambiguous artifacts."""
    result: dict[str, Path] = {}
    for path in paths:
        case = _efit_case_key(path)
        if case in result and result[case].resolve() != path.resolve():
            raise ValueError(
                f"Duplicate {kind} artifacts for EFIT case {case}: "
                f"{result[case]} and {path}"
            )
        result[case] = path
    return result


def _constraint_index_for_time(
    ods: Any,
    time_value: float,
    tolerance: float = 5.0e-4,
) -> int | None:
    try:
        times = np.asarray(ods["equilibrium.time"], dtype=float).reshape(-1)
    except Exception:
        return None
    if not times.size:
        return None
    index = int(np.argmin(np.abs(times - time_value)))
    return index if abs(float(times[index]) - time_value) <= tolerance else None


def _merge_input_constraints(
    target: Any,
    source: Any,
    target_index: int,
    time_value: float,
) -> None:
    """Copy submitted constraints and their metadata into a g-file slice."""
    if source is None:
        return
    source_index = _constraint_index_for_time(source, time_value)
    if source_index is None:
        return
    source_path = f"equilibrium.time_slice.{source_index}.constraints"
    try:
        target[f"equilibrium.time_slice.{target_index}.constraints"] = copy.deepcopy(
            source[source_path]
        )
    except Exception:
        pass
    try:
        params = source[f"equilibrium.code.parameters.time_slice.{source_index}"]
        path = (
            f"equilibrium.code.parameters.time_slice.{target_index}.constraints_input"
        )
        target[path] = copy.deepcopy(params)
    except Exception:
        pass


def _constraint_snapshot(ods: Any, index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in (
        "bpol_probe",
        "flux_loop",
        "pf_current",
        "ip",
        "diamagnetic_flux",
    ):
        root = f"equilibrium.time_slice.{index}.constraints.{family}"
        try:
            node = ods[root]
        except Exception:
            continue
        for path in (
            "measured",
            "measured_error_upper",
            "weight",
            "reconstructed",
            "chi_squared",
        ):
            try:
                value = node[path]
                result[f"{family}.{path}"] = np.asarray(value).tolist()
            except Exception:
                try:
                    value = node[f":.{path}"]
                    result[f"{family}.{path}"] = np.asarray(value).tolist()
                except Exception:
                    pass
    return result


def _mapping_differences(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for path in sorted(set(before) & set(after)):
        left = np.asarray(before[path])
        right = np.asarray(after[path])
        if left.shape != right.shape or not np.allclose(left, right, equal_nan=True):
            output.append({"path": path, "input": before[path], "mfile": after[path]})
    return output


def _efit_case_time(case: str) -> float:
    """Decode the conventional millisecond suffix used by VEST EFIT files."""
    try:
        return int(case.rsplit(".", 1)[1]) / 1000.0
    except (IndexError, ValueError):
        return float("nan")


def _skipped_efit_result(
    inputs: EFITInputs,
    config: EFITConfig,
    *,
    reason: str,
    executable: str | Path | None = None,
) -> EFITResult:
    """Build skipped slice statuses without collecting stale workdir outputs."""
    kfiles_by_time = {
        _efit_case_time(_efit_case_key(Path(path))): Path(path)
        for path in inputs.kfiles
    }
    times = set(kfiles_by_time)
    if config.times is not None:
        times.update(round(float(value) * 1000) / 1000 for value in config.times)
    statuses = tuple(
        validate_efit_slice(
            shot=int(config.shot or 0),
            time=time_value,
            runtime_status="runtime_error",
            returncode=None,
            kfile=kfiles_by_time.get(time_value),
            gfile=None,
            provenance={
                "runtime_reason": reason,
                "executable": str(executable) if executable is not None else None,
                "attempt_logs": [],
            },
        )
        for time_value in sorted(times)
    )
    return EFITResult(
        returncode=None,
        workdir=Path(inputs.workdir),
        kfiles=tuple(Path(path) for path in inputs.kfiles),
        status="skipped",
        reason=reason,
        slice_statuses=statuses,
        configuration=resolved_efit_configuration(config),
    )


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
    constraints_ods: Any = None,
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

    kfile_by_case = _case_file_map(kfiles, "k-file")
    gfile_by_case = _case_file_map(gfiles, "g-file")
    afile_by_case = _case_file_map(afiles, "a-file")
    mfile_by_case = _case_file_map(mfiles, "m-file")

    parsed_by_case = {}
    parse_error_by_case = {}
    artifact_parse_error_by_case: dict[str, list[str]] = {}
    for gfile in gfiles:
        case = _efit_case_key(gfile)
        try:
            from vaft.data.eqdsk import read_geqdsk

            parsed_by_case[case] = read_geqdsk(gfile)
        except Exception as exc:
            parse_error_by_case[case] = f"{gfile}: {exc}"
            continue

    parsed_k_by_case = {}
    parsed_m_by_case = {}
    parsed_a_by_case = {}
    for case, afile in afile_by_case.items():
        try:
            from vaft.data.aeqdsk import read_aeqdsk
            parsed_a_by_case[case] = read_aeqdsk(afile)
        except Exception as exc:
            artifact_parse_error_by_case.setdefault(case, []).append(f"{afile}: {exc}")
    for case, kfile in kfile_by_case.items():
        try:
            from vaft.data.keqdsk import read_keqdsk
            parsed_k_by_case[case] = read_keqdsk(kfile)
        except Exception as exc:
            artifact_parse_error_by_case.setdefault(case, []).append(f"{kfile}: {exc}")
    for case, mfile in mfile_by_case.items():
        try:
            from vaft.data.meqdsk import read_meqdsk
            parsed_m_by_case[case] = read_meqdsk(mfile)
            embedded_time = parsed_m_by_case[case].time_seconds()
            case_time = _efit_case_time(case)
            if embedded_time is not None and abs(embedded_time - case_time) > 5.0e-4:
                artifact_parse_error_by_case.setdefault(case, []).append(
                    "m-file embedded time does not match filename: "
                    f"{embedded_time:.9g} s versus {case_time:.9g} s"
                )
        except Exception as exc:
            artifact_parse_error_by_case.setdefault(case, []).append(f"{mfile}: {exc}")

    parsed_cases = sorted(parsed_by_case, key=_efit_case_time)
    parsed = [parsed_by_case[case] for case in parsed_cases]
    parse_errors = list(parse_error_by_case.values()) + [
        message
        for messages in artifact_parse_error_by_case.values()
        for message in messages
    ]
    ods = None
    conversion_error = None
    mapping_diagnostics: list[dict[str, Any]] = []
    if parsed:
        try:
            for idx, (case, item) in enumerate(zip(parsed_cases, parsed)):
                ods = item.to_omas(ods=ods, time_index=idx)
                time_value = _efit_case_time(case)
                _merge_input_constraints(ods, constraints_ods, idx, time_value)
                # EFIT's own convergence verdict and total chi-square. Written
                # before the k-/m-file overlays so their precedence is unchanged;
                # it lands under its own `aeqdsk` parameter block and collides
                # with nothing either of them writes.
                if case in parsed_a_by_case:
                    parsed_a_by_case[case].to_omas(ods, time_index=idx)
                before = _constraint_snapshot(ods, idx)
                if case in parsed_k_by_case:
                    parsed_k_by_case[case].to_omas(ods, time_index=idx)
                if case in parsed_m_by_case:
                    parsed_m_by_case[case].to_omas(ods, time_index=idx)
                after = _constraint_snapshot(ods, idx)
                differences = _mapping_differences(before, after)
                diagnostic = {"case": case, "differences": differences}
                mapping_diagnostics.append(diagnostic)
                diagnostic_path = (
                    f"equilibrium.code.parameters.time_slice.{idx}.mapping_diagnostics"
                )
                ods[diagnostic_path] = diagnostic
                for kind, path in (
                    ("kfile", kfile_by_case.get(case)),
                    ("gfile", gfile_by_case.get(case)),
                    ("mfile", mfile_by_case.get(case)),
                    ("afile", afile_by_case.get(case)),
                ):
                    if path is not None and path.is_file():
                        artifact_root = (
                            f"equilibrium.code.parameters.time_slice.{idx}.artifacts"
                            f".{kind}"
                        )
                        ods[f"{artifact_root}.path"] = str(path)
                        ods[f"{artifact_root}.sha256"] = _file_sha256(path)
            times = np.asarray([_efit_case_time(case) for case in parsed_cases])
            ods["equilibrium.time"] = times
            for idx, time_value in enumerate(times):
                ods[f"equilibrium.time_slice.{idx}.time"] = time_value
        except Exception as exc:
            conversion_error = f"to_omas: {exc}"
            parse_errors.append(conversion_error)
            ods = None

    file_maps = {
        "kfile": kfile_by_case,
        "gfile": gfile_by_case,
        "afile": afile_by_case,
        "mfile": mfile_by_case,
    }
    cases = sorted(
        set().union(*(mapping.keys() for mapping in file_maps.values())),
        key=_efit_case_time,
    )
    if config is not None and config.shot is not None and config.times is not None:
        configured_cases = {
            f"0{int(config.shot)}.{int(round(float(time_value) * 1000))}"
            for time_value in config.times
        }
        cases = sorted(set(cases) | configured_cases, key=_efit_case_time)
    status_shot = int(shot) if shot is not None else 0
    statuses = []
    for case in cases:
        case_kfile = file_maps["kfile"].get(case)
        kfile_sha256 = (
            _file_sha256(case_kfile)
            if case_kfile is not None and case_kfile.is_file()
            else None
        )
        statuses.append(
            validate_efit_slice(
                shot=status_shot,
                time=_efit_case_time(case),
                runtime_status=runtime_status,
                returncode=returncode,
                kfile=case_kfile,
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
                    "kfile_sha256": kfile_sha256,
                    "artifact_parse_errors": artifact_parse_error_by_case.get(case, []),
                    "configuration": (
                        resolved_efit_configuration(config)
                        if config is not None
                        else {}
                    ),
                },
                config=validation_config,
            )
        )
    statuses = list(apply_temporal_continuity(statuses, validation_config))

    artifact_hashes = {
        str(path): _file_sha256(path)
        for paths in (kfiles, gfiles, afiles, mfiles, logs)
        for path in paths
        if path.is_file()
    }
    return EFITResult(
        returncode=returncode,
        workdir=base,
        gfiles=gfiles,
        afiles=afiles,
        mfiles=mfiles,
        kfiles=kfiles,
        logs=logs,
        geqdsk=tuple(parsed),
        keqdsk=tuple(
            parsed_k_by_case[case]
            for case in sorted(parsed_k_by_case, key=_efit_case_time)
        ),
        meqdsk=tuple(
            parsed_m_by_case[case]
            for case in sorted(parsed_m_by_case, key=_efit_case_time)
        ),
        parse_errors=tuple(parse_errors),
        mapping_diagnostics=tuple(mapping_diagnostics),
        artifact_hashes=artifact_hashes,
        ods=ods,
        slice_statuses=tuple(statuses),
        configuration=(
            resolved_efit_configuration(config) if config is not None else {}
        ),
    )


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
