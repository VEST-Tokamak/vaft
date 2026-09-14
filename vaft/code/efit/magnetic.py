"""Magnetic (routine) EFIT adapter: config, prepare/run/collect, gfile conversion.

Moved verbatim out of the former monolithic ``efit.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

from ... import compat
from ...compat import is_executable
from .._executables import missing_home_message
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

if TYPE_CHECKING:
    from .linearization import EFITLinearization


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
    # Execution-only diagnostics/continuation controls.  These deliberately do
    # not participate in EFITScientificConfig (and therefore cannot change the
    # scientific k-file identity).
    export_linearization: bool = False
    direction_constraint: Path | str | None = None
    restart_from: Path | str | None = None
    write_restart: bool = False

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
        for name in ("export_linearization", "write_restart"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("direction_constraint", "restart_from"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())

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
    linearization_files: tuple[Path, ...] = ()
    linearizations: tuple[EFITLinearization, ...] = ()
    restart_file: Path | None = None
    diagnostic_errors: tuple[str, ...] = ()

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
            "export_linearization": config.export_linearization,
            "direction_constraint": (
                str(config.direction_constraint)
                if config.direction_constraint is not None
                else None
            ),
            "direction_constraint_sha256": _optional_file_sha256(
                config.direction_constraint
            ),
            "restart_from": (
                str(config.restart_from) if config.restart_from is not None else None
            ),
            "restart_from_sha256": _optional_file_sha256(config.restart_from),
            "write_restart": config.write_restart,
        },
        "provenance": _json_value(config.provenance),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_fingerprint(path: Path) -> tuple[int, int, int, int, str] | None:
    """Return enough inode metadata to detect an unwritten stale output."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        _file_sha256(path),
    )


def _optional_file_sha256(path: Path | str | None) -> str | None:
    """Hash an optional execution input without making configuration impure."""
    if path is None:
        return None
    candidate = Path(path)
    return _file_sha256(candidate) if candidate.is_file() else None


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


_TABLE_DIR_LINE = re.compile(r"^\s*TABLE_DIR\s*=\s*'([^']*)'", re.IGNORECASE | re.MULTILINE)


def _table_record(kfiles: Sequence[Path]) -> dict[str, Any] | None:
    """What the k-files say EFIT will read its Green tables from, identified.

    EFIT reads every table dimension from ``TABLE_DIR/mhdin.dat`` with no
    consistency check, so the run manifest records the table by its own
    manifest (``vaft.code.efit.efund``) when it has one and by the hash of
    ``mhdin.dat`` otherwise -- and says which of the two it could do.
    """
    from .efund import table_identity

    directories: list[str] = []
    for path in sorted(Path(path) for path in kfiles):
        try:
            match = _TABLE_DIR_LINE.search(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if match and match.group(1) not in directories:
            directories.append(match.group(1))
    if not directories:
        return None
    record = table_identity(directories[0])
    if len(directories) > 1:
        record["other_dirs"] = directories[1:]
    return record


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
        "table": _table_record(kfiles),
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
    # One rule for both toolchain roles (issue #194): explicit path, then
    # $EFITHOME in the installed or the CMake build-tree layout, then the
    # legacy $EFIT for efit only.
    from .toolchain import resolve_role

    environment = {**os.environ, **dict(config.env)}
    return resolve_role("efit", explicit=config.executable or None, env=environment)


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


#: Where the Windows installer leaves the `ls` EFIT shells out for, relative to
#: the installation root. It is kept out of `bin/` deliberately: it belongs on
#: one child process's PATH, never on a user's.
SHIM_DIRECTORY = Path("shim")


def _add_shim_directory(env: dict[str, str], executable: str | Path) -> None:
    """Put the installation's shim directory on the child's PATH, on Windows.

    ``set_table_dir`` picks the Green-table subdirectory for a shot by shelling
    out: ``call system('ls '//table_dir//' > shot_tables.txt')``
    (``efit/tables.F90``). ``cmd.exe`` has no ``ls``, and the failure is silent
    rather than loud -- EFIT falls back to ``<link_efit>/green/`` itself instead
    of ``<link_efit>/green/<shot range>/``, so it reads the wrong tables or none
    at all. The installer writes a small ``ls`` there for exactly this call.

    Only this child's environment is touched. Putting the directory on a user's
    PATH would shadow a real ``ls`` for everything else they run.
    """
    if not compat.IS_WINDOWS:
        return
    # Both layouts resolve_role accepts: <root>/bin/efit.exe from an install,
    # and <build>/efit/efit.exe from a CMake tree whose shim sits one level
    # further up, beside the build directory rather than inside it.
    here = Path(executable).resolve().parent
    for root in (here.parent, here.parent.parent):
        shim = root / SHIM_DIRECTORY
        if shim.is_dir():
            existing = env.get("PATH", "")
            env["PATH"] = f"{shim}{os.pathsep}{existing}" if existing else str(shim)
            return
    # Saying nothing here is what the shim exists to prevent: EFIT would read
    # the wrong Green tables and never mention it. A prefix from an installer
    # older than the shim reaches this too.
    warnings.warn(
        f"No {SHIM_DIRECTORY}/ls.cmd found near {executable}. EFIT picks its "
        "Green-table subdirectory by shelling out to `ls`, which cmd.exe does "
        "not have, and falls back to the wrong directory silently. Reinstall "
        "with install/install_efit_windows.ps1, which writes it.",
        RuntimeWarning,
        stacklevel=3,
    )


def _efit_command(config: EFITConfig, executable: str | Path | None = None) -> list[str]:
    resolved = executable if executable is not None else config.executable
    if not resolved:
        raise ValueError("EFITConfig.executable is required to run EFIT")
    executable = str(resolved)
    args = [str(arg) for arg in config.args]
    if config.stack_size_kb is None:
        return [executable, *args]
    if compat.IS_WINDOWS:
        # A native Windows image takes its stack reserve from the PE header,
        # fixed by the linker, so no wrapper can raise it after the fact --
        # install/install_efit_windows.ps1 passes -Wl,--stack instead, and
        # -StackReserveMB is where this setting goes there. Wrapping anyway
        # would also make every run depend on an MSYS2 bash that the installers
        # deliberately keep off PATH.
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


_NAMELIST_INTEGER = r"(?im)^(?P<prefix>\s*{name}\s*=\s*)(?P<value>[-+]?\d+)"


def _set_namelist_integer(text: str, name: str, value: int) -> str:
    """Replace or add a scalar integer in the first EFIT namelist.

    Continuation controls are execution policy, so they are applied to a
    staged copy instead of mutating the scientifically hashed source k-file.
    """
    pattern = re.compile(_NAMELIST_INTEGER.format(name=re.escape(name)))
    if pattern.search(text):
        return pattern.sub(
            lambda match: f"{match.group('prefix')}{int(value)}", text, count=1
        )
    terminator = re.search(r"(?m)^\s*/\s*$", text)
    if terminator is None:
        raise ValueError(f"EFIT k-file has no IN1 terminator; cannot set {name}")
    return text[: terminator.start()] + f" {name} = {int(value)}\n" + text[terminator.start() :]


def _execution_kfiles(inputs: EFITInputs, config: EFITConfig) -> tuple[Path, ...]:
    """Return k-files used by the child, staging execution-only edits."""
    mutation_controls_requested = (
        config.direction_constraint is not None
        or config.restart_from is not None
        or config.write_restart
    )
    # Native exports may be requested for source k-files outside the run
    # directory.  EFIT's interactive filename buffer is short, so stage those
    # too; otherwise a valid absolute path can be silently truncated.
    staging_requested = mutation_controls_requested or config.export_linearization
    kfiles = tuple(Path(path) for path in inputs.kfiles)
    if not staging_requested:
        return kfiles
    if mutation_controls_requested and len(kfiles) != 1:
        raise ValueError(
            "direction and restart controls require exactly one EFIT k-file"
        )

    staging_parent = Path(inputs.workdir) / ".vaft_efit_execution"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="kfile-", dir=staging_parent))
    destinations = []
    for source in kfiles:
        if not source.is_file():
            raise FileNotFoundError(f"EFIT k-file does not exist: {source}")
        text = source.read_text(encoding="utf-8")
        if config.restart_from is not None:
            text = _set_namelist_integer(text, "ICINIT", -2)
        if config.write_restart:
            match = re.search(_NAMELIST_INTEGER.format(name="IOUT"), text)
            current_iout = int(match.group("value")) if match is not None else 0
            text = _set_namelist_integer(text, "IOUT", current_iout | 16)
        destination = staging / source.name
        if destination.exists():
            raise ValueError(f"duplicate staged EFIT k-file name: {source.name}")
        destination.write_text(text, encoding="utf-8")
        destinations.append(destination)
    return tuple(destinations)


def _restart_control_sha256(
    kfiles: Sequence[str | Path], direction_file: Path | None
) -> str:
    """Bind native restart history to the original scientific controls.

    This is intentionally evaluated before :func:`_execution_kfiles` stages
    ``ICINIT=-2`` or adds the IOUT restart bit.  The framed byte stream makes
    an absent direction file distinct from an empty one and avoids ambiguous
    concatenations.
    """
    sources = tuple(Path(path) for path in kfiles)
    if len(sources) != 1:
        raise ValueError("restart-control identity requires exactly one EFIT k-file")
    source = sources[0]
    if not source.is_file():
        raise FileNotFoundError(f"EFIT k-file does not exist: {source}")

    digest = hashlib.sha256()
    digest.update(b"vaft-efit-restart-control-v1\0")

    def update_framed(label: bytes, payload: bytes) -> None:
        digest.update(label)
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    update_framed(b"kfile\0", source.read_bytes())
    if direction_file is None:
        update_framed(b"direction-null\0", b"")
    else:
        if not direction_file.is_file():
            raise FileNotFoundError(
                f"EFIT direction constraint does not exist: {direction_file}"
            )
        update_framed(b"direction-file\0", direction_file.read_bytes())
    value = digest.hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AssertionError("restart-control SHA-256 is not canonical lowercase hex")
    return value


def _prepare_response_diagnostics_directory(workdir: Path) -> Path:
    parent = workdir / "response_diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=parent)).resolve()


def _validate_execution_file(path: Path | str | None, label: str) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {candidate}")
    return candidate


def _read_direction_control_identity(path: Path) -> dict[str, str]:
    """Validate the execution identity embedded in a native direction file."""
    import xarray as xr

    try:
        with xr.open_dataset(path, decode_cf=False) as dataset:
            attrs = dict(dataset.attrs)
    except Exception as exc:
        raise ValueError(f"cannot read EFIT direction constraint {path}: {exc}") from exc

    def text_attribute(name: str) -> str:
        value = attrs.get(name, "")
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="strict")
        return str(value).strip()

    schema_id = text_attribute("schema_id")
    try:
        schema_version = int(attrs.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("EFIT direction constraint has an invalid schema version") from exc
    if schema_id != "efit_direction_constraint_v1" or schema_version != 1:
        raise ValueError(
            f"unsupported EFIT direction constraint schema {schema_id!r} "
            f"version {schema_version}"
        )
    try:
        writer_complete = int(attrs.get("writer_complete", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "EFIT direction constraint has an invalid completion marker"
        ) from exc
    if writer_complete != 1:
        raise ValueError("EFIT direction constraint is incomplete")
    result = {
        "schema_id": schema_id,
        "schema_version": str(schema_version),
        "source_sidecar_sha256": text_attribute("source_sidecar_sha256"),
        "parameter_order_sha256": text_attribute("parameter_order_sha256"),
    }
    for name in ("source_sidecar_sha256", "parameter_order_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", result[name]) is None:
            raise ValueError(
                f"EFIT direction constraint has an invalid {name.replace('_', ' ')}"
            )
    return result


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
    direction_file = _validate_execution_file(
        config.direction_constraint, "EFIT direction constraint"
    )
    direction_identity = (
        _read_direction_control_identity(direction_file)
        if direction_file is not None
        else None
    )
    restart_source = _validate_execution_file(config.restart_from, "EFIT restart")
    restart_control_sha256 = None
    if config.write_restart or restart_source is not None:
        restart_control_sha256 = _restart_control_sha256(
            inputs.kfiles, direction_file
        )
    execution_kfiles = _execution_kfiles(inputs, config)
    command = _efit_command(config, executable)
    stdin_text = _efit_stdin(workdir, execution_kfiles)
    env = os.environ.copy()
    env.update(dict(config.env))
    env.setdefault("OMP_NUM_THREADS", "1")
    if restart_control_sha256 is not None:
        env["EFIT_RESTART_CONTROL_SHA256"] = restart_control_sha256
    diagnostics_dir = None
    if config.export_linearization:
        diagnostics_dir = _prepare_response_diagnostics_directory(workdir)
        env["EFIT_RESPONSE_DIAGNOSTICS_DIR"] = str(diagnostics_dir)
        # Bind every native response sidecar to the exact executable which
        # produced it.  This is execution provenance, not scientific input,
        # and therefore deliberately remains outside the k-file hash.
        env["EFIT_EXECUTABLE_SHA256"] = _file_sha256(Path(executable))
    if direction_file is not None:
        env["EFIT_DIRECTION_CONSTRAINT_FILE"] = str(direction_file)
        assert direction_identity is not None
        env["EFIT_DIRECTION_SOURCE_SIDECAR_SHA256"] = direction_identity[
            "source_sidecar_sha256"
        ]
    restart_destination = workdir / "esave.dat"
    if restart_source is not None:
        if restart_source != restart_destination.resolve():
            shutil.copy2(restart_source, restart_destination)
    restart_pre_run_fingerprint = (
        _file_stat_fingerprint(restart_destination) if config.write_restart else None
    )
    attempted_case_keys = {_efit_case_key(Path(path)) for path in inputs.kfiles}
    pre_run_output_fingerprints = {
        str(path.resolve()): fingerprint
        for prefix in ("g", "a", "m")
        for path in _find_outputs(workdir, prefix, config.shot)
        if _efit_case_key(path) in attempted_case_keys
        if (fingerprint := _file_stat_fingerprint(path)) is not None
    }
    _add_shim_directory(env, executable)
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
            diagnostics_dir=diagnostics_dir,
            requested_diagnostics=config.export_linearization,
            restart_file=None,
            executed_kfiles=execution_kfiles,
            direction_identity=direction_identity,
            restart_control_sha256=restart_control_sha256,
            pre_run_output_fingerprints=pre_run_output_fingerprints,
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
            diagnostics_dir=diagnostics_dir,
            requested_diagnostics=config.export_linearization,
            restart_file=None,
            executed_kfiles=execution_kfiles,
            direction_identity=direction_identity,
            restart_control_sha256=restart_control_sha256,
            pre_run_output_fingerprints=pre_run_output_fingerprints,
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
        diagnostics_dir=diagnostics_dir,
        requested_diagnostics=config.export_linearization,
        restart_file=(
            restart_destination
            if config.write_restart and completed.returncode == 0
            else None
        ),
        restart_pre_run_fingerprint=restart_pre_run_fingerprint,
        executed_kfiles=execution_kfiles,
        direction_identity=direction_identity,
        restart_control_sha256=restart_control_sha256,
        pre_run_output_fingerprints=pre_run_output_fingerprints,
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
    diagnostics_dir: str | Path | None = None,
    requested_diagnostics: bool = False,
    restart_file: str | Path | None = None,
    restart_pre_run_fingerprint: tuple[int, int, int, int, str] | None = None,
    executed_kfiles: Sequence[str | Path] = (),
    direction_identity: Mapping[str, str] | None = None,
    restart_control_sha256: str | None = None,
    pre_run_output_fingerprints: Mapping[
        str, tuple[int, int, int, int, str]
    ] | None = None,
) -> EFITResult:
    """Collect EFIT files and assign independent status to every attempted slice."""
    base = _efit_workdir(config, workdir)
    shot = config.shot if config is not None else None
    result_configuration = (
        resolved_efit_configuration(config) if config is not None else {}
    )
    if result_configuration and executed_kfiles:
        execution = result_configuration["execution"]
        execution["executed_kfiles"] = [
            {"path": str(path), "sha256": _file_sha256(Path(path))}
            for path in executed_kfiles
            if Path(path).is_file()
        ]
        execution["response_diagnostics_dir"] = (
            str(Path(diagnostics_dir)) if diagnostics_dir is not None else None
        )
        if executable is not None and Path(executable).is_file():
            execution["executable_sha256"] = _file_sha256(Path(executable))
        if direction_identity is not None:
            execution["direction_control_identity"] = dict(direction_identity)
        if restart_control_sha256 is not None:
            if re.fullmatch(r"[0-9a-f]{64}", restart_control_sha256) is None:
                raise ValueError("restart-control SHA-256 must be 64 lowercase hex")
            execution["restart_control_sha256"] = restart_control_sha256
    stale_output_errors: list[str] = []

    def current_outputs(prefix: str) -> tuple[Path, ...]:
        files = _find_outputs(base, prefix, shot)
        if not pre_run_output_fingerprints:
            return files
        current: list[Path] = []
        for path in files:
            old = pre_run_output_fingerprints.get(str(path.resolve()))
            if old is not None and _file_stat_fingerprint(path) == old:
                stale_output_errors.append(
                    f"unchanged stale EFIT {prefix}-file was ignored: {path}"
                )
            else:
                current.append(path)
        return tuple(current)

    gfiles = current_outputs("g")
    afiles = current_outputs("a")
    mfiles = current_outputs("m")
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

    linearization_files: list[Path] = []
    linearizations: list[Any] = []
    diagnostic_errors: list[str] = list(stale_output_errors)
    diagnostic_candidates = (
        tuple(sorted(Path(diagnostics_dir).rglob("*.nc")))
        if diagnostics_dir is not None and Path(diagnostics_dir).is_dir()
        else ()
    )
    if requested_diagnostics and not diagnostic_candidates:
        diagnostic_errors.append("requested EFIT response diagnostics were not produced")
    if diagnostic_candidates:
        from .linearization import read_efit_linearization

        accepted_cases: set[str] = set()
        expected_executable_sha256 = (
            _file_sha256(Path(executable))
            if executable is not None and Path(executable).is_file()
            else None
        )
        for path in diagnostic_candidates:
            try:
                problem = read_efit_linearization(path)
                if (
                    expected_executable_sha256 is not None
                    and problem.executable_sha256 != expected_executable_sha256
                ):
                    raise ValueError(
                        "diagnostic executable SHA-256 does not match the executed binary"
                    )
                matching_cases = [
                    case
                    for case in gfile_by_case
                    if case.startswith(f"0{problem.shot}.")
                    and abs(_efit_case_time(case) - problem.time_seconds) <= 5.0e-4
                ]
                if len(matching_cases) != 1:
                    raise ValueError(
                        "diagnostic does not identify exactly one produced plasma slice"
                    )
                case = matching_cases[0]
                if case in accepted_cases:
                    raise ValueError(f"duplicate completed diagnostic for case {case}")
                accepted_cases.add(case)
                linearization_files.append(path)
                linearizations.append(problem)
            except Exception as exc:
                diagnostic_errors.append(f"{path}: {exc}")

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
                        result_configuration
                    ),
                },
                config=validation_config,
            )
        )
    statuses = list(apply_temporal_continuity(statuses, validation_config))

    completed_restart = Path(restart_file) if restart_file is not None else None
    if completed_restart is not None and not completed_restart.is_file():
        completed_restart = None
        if config is not None and config.write_restart:
            diagnostic_errors.append("requested EFIT restart file was not produced")
    elif completed_restart is not None and (
        restart_pre_run_fingerprint is not None
        and _file_stat_fingerprint(completed_restart) == restart_pre_run_fingerprint
    ):
        completed_restart = None
        diagnostic_errors.append(
            "requested EFIT restart file was not rewritten; refusing stale esave.dat"
        )

    artifact_hashes = {
        str(path): _file_sha256(path)
        for paths in (
            kfiles,
            gfiles,
            afiles,
            mfiles,
            logs,
            tuple(linearization_files),
            ((completed_restart,) if completed_restart is not None else ()),
            tuple(Path(path) for path in executed_kfiles),
        )
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
        configuration=result_configuration,
        linearization_files=tuple(linearization_files),
        linearizations=tuple(linearizations),
        restart_file=completed_restart,
        diagnostic_errors=tuple(diagnostic_errors),
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
