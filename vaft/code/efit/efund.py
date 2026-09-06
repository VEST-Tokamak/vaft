"""EFUND adapter: canonical geometry -> ``mhdin.dat`` -> ``efund`` -> Green tables.

EFUND is the Green-function table generator of the EFIT toolchain.  It reads
``mhdin.dat`` from its working directory, takes the grid size on the command
line, and writes sequential unformatted tables (``ec<nw><nh>.ddd``,
``ep<nw><nh>.ddd``, ``rfcoil.ddd``, ``rv<nw><nh>.ddd``, ...) that EFIT later
reads from ``TABLE_DIR`` with no consistency check of its own.  Everything an
equilibrium's provenance needs to say about its table is therefore recorded
here, at generation time:

* the geometry (:class:`~vaft.machine_mapping.efund_geometry.EFUNDGeometry`)
  and the era and asset hashes it came from,
* the EFUND-only quantities (:class:`EFUNDConfig`: grid, flags, quadrature),
* the executable that ran (:func:`~vaft.code.efit.toolchain.executable_identity`),
* the sha256 of the input and of every output file, checked against the
  record layout EFUND's source implies.

The generated directory is self-describing through
:data:`TABLE_MANIFEST_NAME`; :func:`table_identity` is what an EFIT run
manifest records for the table it consumed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import f90nml
import numpy as np

from vaft.machine_mapping.efund_geometry import EFUNDGeometry, efund_geometry_from_static

from .toolchain import executable_identity, resolve_role, unconfigured_reason

__all__ = [
    "EFUNDConfig",
    "EFUNDInputs",
    "EFUNDResult",
    "MHDIN_NAME",
    "TABLE_MANIFEST_NAME",
    "collect_efund_outputs",
    "expected_table_files",
    "prepare_efund_inputs",
    "fortran_byte_order",
    "read_fortran_arrays",
    "read_fortran_records",
    "read_table_manifest",
    "run_efund",
    "table_identity",
    "write_mhdin",
    "write_table_manifest",
]

MHDIN_NAME = "mhdin.dat"
TABLE_MANIFEST_NAME = "efund_table_manifest.json"
EFUND_STDOUT_NAME = "run_efund.out"
EFUND_STDERR_NAME = "run_efund.err"

#: EFUND declares its name arrays ``character*10``; longer names are cut.
_NAME_LENGTH = 10
#: gfortran sequential unformatted files frame every record with a 4-byte
#: length before and after it.
_RECORD_MARKER = 4
_FLAG_FIELDS = ("igrid", "ifcoil", "ivesel", "iecoil", "iacoil", "islpfc")


@dataclass(frozen=True)
class EFUNDConfig:
    """The EFUND-only quantities, explicit and typed.

    The defaults are the bundled legacy VEST table's, read back from its
    ``mhdin.dat``, with two corrections where that file's own echo disagreed
    with the table beside it: ``ivesel = 1`` (the file says 0, yet
    ``rv129129.ddd`` exists and the k-file needs it) and ``iecoil = 0`` (the
    file says 1 with no E-coil defined).
    """

    workdir: Path | str = Path(".")
    nw: int = 129
    nh: int = 129
    rleft: float = 0.05
    rright: float = 1.2
    zbotto: float = -1.5
    ztop: float = 1.5
    igrid: int = 1
    ifcoil: int = 1
    ivesel: int = 1
    iecoil: int = 0
    iacoil: int = 0
    islpfc: int = 0
    isize: int = 0
    nsmp2: int = 1
    mgaus1: int = 8
    mgaus2: int = 10
    device: str = "VEST"
    executable: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    stack_size_kb: int | str | None = "hard"

    def __post_init__(self) -> None:
        for name in ("nw", "nh", "nsmp2", "mgaus1", "mgaus2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.nw > 9999 or self.nh > 9999:
            raise ValueError("EFUND reads the grid size with an i4 format; nw and nh must be below 10000")
        for name in (*_FLAG_FIELDS, "isize"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if name in _FLAG_FIELDS and value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
            object.__setattr__(self, name, int(value))
        for name in ("rleft", "rright", "zbotto", "ztop"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not self.rleft < self.rright:
            raise ValueError("rleft must be below rright")
        if self.rleft <= 0.0:
            raise ValueError("rleft must be positive: the grid cannot touch the axis")
        if not self.zbotto < self.ztop:
            raise ValueError("zbotto must be below ztop")
        if not str(self.device).strip():
            raise ValueError("device must not be empty")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if isinstance(self.stack_size_kb, str):
            if self.stack_size_kb != "hard":
                raise ValueError('stack_size_kb must be a positive integer, "hard", or None')
        elif self.stack_size_kb is not None and (
            isinstance(self.stack_size_kb, bool) or self.stack_size_kb <= 0
        ):
            raise ValueError("stack_size_kb must be greater than zero")

    @property
    def table_suffix(self) -> str:
        """The ``<nw><nh>`` EFUND appends to grid-sized table names."""
        return f"{self.nw}{self.nh}"

    @property
    def argv(self) -> list[str]:
        """EFUND's command-line arguments: ``nw`` and, when different, ``nh``."""
        return [str(self.nw)] + ([str(self.nh)] if self.nh != self.nw else [])

    def in5(self) -> dict[str, Any]:
        """The ``&in5`` namelist values.

        ``islpfc`` is deliberately not here: EFUND declares it in ``&in3``.
        The bundled legacy ``mhdin.dat`` lists it under ``&in5`` as well, and
        this EFUND revision rejects that group on the unknown name, returns
        before allocating its grid and then crashes -- so that file cannot
        have been consumed by this revision as it stands.
        """
        return {
            "rleft": self.rleft,
            "rright": self.rright,
            "zbotto": self.zbotto,
            "ztop": self.ztop,
            "mgaus1": self.mgaus1,
            "mgaus2": self.mgaus2,
            "nsmp2": self.nsmp2,
            "igrid": self.igrid,
            "ifcoil": self.ifcoil,
            "iecoil": self.iecoil,
            "ivesel": self.ivesel,
            "iacoil": self.iacoil,
            "isize": self.isize,
        }

    def to_dict(self) -> dict[str, Any]:
        """The scientific content only: no paths, executable, or timeouts."""
        return {
            "schema_version": 1,
            "device": str(self.device),
            "grid": {
                "nw": self.nw,
                "nh": self.nh,
                "rleft": self.rleft,
                "rright": self.rright,
                "zbotto": self.zbotto,
                "ztop": self.ztop,
            },
            "flags": {
                "igrid": self.igrid,
                "ifcoil": self.ifcoil,
                "ivesel": self.ivesel,
                "iecoil": self.iecoil,
                "iacoil": self.iacoil,
                "islpfc": self.islpfc,
                "isize": self.isize,
                "nsmp2": self.nsmp2,
            },
            "quadrature": {"mgaus1": self.mgaus1, "mgaus2": self.mgaus2},
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EFUNDInputs:
    """What :func:`run_efund` needs: a working directory holding ``mhdin.dat``."""

    workdir: Path
    mhdin: Path
    mhdin_sha256: str
    geometry: EFUNDGeometry
    counts: dict[str, int]
    header: tuple[str, ...] = ()


@dataclass
class EFUNDResult:
    """Outcome of an EFUND run or of collecting an existing table directory."""

    returncode: Optional[int]
    workdir: Path
    status: str
    reason: str = ""
    files: dict[str, Path] = field(default_factory=dict)
    expected: dict[str, Optional[int]] = field(default_factory=dict)
    problems: tuple[str, ...] = ()
    executable: Optional[Path] = None
    logs: tuple[Path, ...] = ()
    stdout: str = ""
    stderr: str = ""
    manifest: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


# --- input -----------------------------------------------------------------


def _names(values: Sequence[str]) -> list[str]:
    return [str(value)[:_NAME_LENGTH] for value in values]


def _floats(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _ints(values: Any) -> list[int]:
    return [int(value) for value in np.asarray(values).reshape(-1)]


def efund_namelist(geometry: EFUNDGeometry, config: EFUNDConfig) -> f90nml.Namelist:
    """The three namelists EFUND reads, in the order ``machinein``, ``in5``, ``in3``."""
    counts = geometry.counts()
    namelist = f90nml.Namelist()
    namelist["machinein"] = {"device": str(config.device), **counts}
    namelist["in5"] = config.in5()
    namelist["in3"] = {
        "islpfc": config.islpfc,
        "rf": _floats(geometry.fcoil_r),
        "zf": _floats(geometry.fcoil_z),
        "wf": _floats(geometry.fcoil_w),
        "hf": _floats(geometry.fcoil_h),
        "af": _floats(geometry.fcoil_a),
        "af2": _floats(geometry.fcoil_a2),
        "fcid": _ints(geometry.fcoil_group),
        "fcturn": _floats(geometry.fcoil_turns),
        "turnfc": _floats(geometry.group_turns),
        "fcname": _names(geometry.group_names),
        "rvs": _floats(geometry.vessel_r),
        "zvs": _floats(geometry.vessel_z),
        "wvs": _floats(geometry.vessel_w),
        "hvs": _floats(geometry.vessel_h),
        "avs": _floats(geometry.vessel_a),
        "avs2": _floats(geometry.vessel_a2),
        "vsid": _ints(geometry.vessel_group),
        "rsisvs": _floats(geometry.vessel_resistance),
        "vsname": _names(geometry.vessel_names),
        "rsi": _floats(geometry.loop_r),
        "zsi": _floats(geometry.loop_z),
        "lpname": _names(geometry.loop_names),
        "xmp2": _floats(geometry.probe_r),
        "ymp2": _floats(geometry.probe_z),
        "amp2": _floats(geometry.probe_angle_deg),
        "smp2": _floats(geometry.probe_length),
        "mpnam2": _names(geometry.probe_names),
    }
    namelist.column_width = 100
    namelist.float_format = ".17g"
    return namelist


def _header_lines(geometry: EFUNDGeometry, config: EFUNDConfig, extra: Sequence[str] = ()) -> list[str]:
    machine = geometry.machine
    lines = [
        "EFUND input generated by vaft.code.efit.efund from canonical VAFT static geometry",
        f"device {config.device}; machine era {machine.get('era')}; PF geometry {machine.get('pf_geometry')}",
        f"efund configuration sha256 {config.sha256}",
    ]
    for name, record in (machine.get("static_inputs") or {}).items():
        lines.append(f"static input {name}: {record.get('name')} sha256 {record.get('sha256')}")
    lines.append(
        "F-coil groups: " + ", ".join(geometry.group_names)
        + f" ({geometry.nfcoil} elements); vessel {geometry.nvesel} segments in em_coupling order;"
        f" {geometry.nsilop} flux loops; {geometry.magpri} probes"
    )
    lines.extend(str(item) for item in extra)
    return lines


def write_mhdin(
    geometry: EFUNDGeometry,
    config: EFUNDConfig,
    path: str | os.PathLike[str],
    *,
    header: Sequence[str] = (),
) -> Path:
    """Write ``mhdin.dat`` for ``geometry`` and ``config``; returns the path."""
    destination = Path(path)
    buffer = io.StringIO()
    for line in _header_lines(geometry, config, header):
        buffer.write(f"! {line}\n")
    efund_namelist(geometry, config).write(buffer)
    destination.write_text(buffer.getvalue(), encoding="utf-8")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_efund_inputs(
    static_ods: Any,
    config: EFUNDConfig,
    *,
    manifest: Mapping[str, Any] | None = None,
    geometry: EFUNDGeometry | None = None,
    header: Sequence[str] = (),
) -> EFUNDInputs:
    """Project ``static_ods`` and write ``mhdin.dat`` into ``config.workdir``.

    ``manifest`` is the record returned beside the ODS by
    :func:`vaft.omas.vest_upstream.build_static_ods`; it is how the era and
    asset hashes reach the table manifest.  Pass ``geometry`` to reuse a
    projection already made.
    """
    workdir = Path(config.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    if geometry is None:
        geometry = efund_geometry_from_static(static_ods, manifest=manifest)
    lines = _header_lines(geometry, config, header)
    mhdin = write_mhdin(geometry, config, workdir / MHDIN_NAME, header=header)
    return EFUNDInputs(
        workdir=workdir,
        mhdin=mhdin,
        mhdin_sha256=_sha256(mhdin),
        geometry=geometry,
        counts=geometry.counts(),
        header=tuple(lines),
    )


# --- outputs ---------------------------------------------------------------


def expected_table_files(config: EFUNDConfig, counts: Mapping[str, int]) -> dict[str, Optional[int]]:
    """File names EFUND writes for ``config``, with exact sizes where the layout is fixed.

    Sizes follow the ``write`` statements in ``green/efund.f90``: one array
    per record, ``real*8`` values, a 4-byte marker either side of each record.
    ``None`` marks a file whose size depends on things not in the counts.
    """
    nsilop = int(counts["nsilop"])
    magpri = int(counts["magpri"])
    nfsum = int(counts["nfsum"])
    nfcoil = int(counts["nfcoil"])
    nesum = int(counts.get("nesum", 0))
    nvsum = int(counts["nvsum"])
    nw, nh = config.nw, config.nh
    nwnh = nw * nh
    suffix = config.table_suffix
    marker = 2 * _RECORD_MARKER

    def record(elements: int) -> int:
        return elements * 8 + marker

    expected: dict[str, Optional[int]] = {}
    if config.ifcoil == 1:
        expected["rfcoil.ddd"] = record(nsilop * nfsum) + record(magpri * nfsum)
        expected["brzgfc.dat"] = record(nwnh * nfsum) + record(nwnh * nfsum)
        if config.islpfc == 1:
            expected[f"fc{suffix}.ddd"] = record(nfcoil * nfcoil) + record(nwnh * nfcoil)
    if config.igrid == 1:
        expected[f"ec{suffix}.ddd"] = (
            (2 * 4 + marker) + record(nw + nh) + record(nwnh * nfsum) + record(nwnh * nw)
        )
        expected[f"ep{suffix}.ddd"] = record(nsilop * nwnh) + record(magpri * nwnh)
    if config.iecoil == 1:
        expected[f"re{suffix}.ddd"] = record(nsilop * nesum) + record(magpri * nesum) + record(nwnh * nesum)
    if config.ivesel == 1:
        expected[f"rv{suffix}.ddd"] = (
            record(nsilop * nvsum)
            + record(magpri * nvsum)
            + record(nwnh * nvsum)
            + record(nfsum * nvsum)
            + record(nesum * nvsum)
            + record(nvsum * nvsum)
        )
    if config.iacoil == 1:
        expected[f"ra{suffix}.ddd"] = None
    expected["mhdout.dat"] = None
    return expected


def fortran_byte_order(path: str | os.PathLike[str]) -> str:
    """``">"`` or ``"<"``: the byte order of a sequential unformatted file.

    The EFIT build writes its tables big-endian (``-fconvert=big-endian``,
    for compatibility with the tables machines have accumulated), so the
    order cannot be assumed from the host.  It is inferred from the first
    record: the order under which its leading and trailing markers agree.
    """
    data = Path(path).read_bytes()
    for order in (">", "<"):
        if len(data) < 2 * _RECORD_MARKER:
            break
        (length,) = struct.unpack(order + "i", data[:_RECORD_MARKER])
        end = _RECORD_MARKER + length
        if 0 <= length and end + _RECORD_MARKER <= len(data):
            (trailer,) = struct.unpack(order + "i", data[end : end + _RECORD_MARKER])
            if trailer == length:
                return order
    raise ValueError(f"{path}: not a sequential unformatted file with 4-byte record markers")


def read_fortran_records(path: str | os.PathLike[str]) -> list[bytes]:
    """The raw records of a gfortran sequential unformatted file."""
    order = fortran_byte_order(path)
    data = Path(path).read_bytes()
    records: list[bytes] = []
    offset = 0
    while offset < len(data):
        if offset + _RECORD_MARKER > len(data):
            raise ValueError(f"{path}: truncated record marker at byte {offset}")
        (length,) = struct.unpack(order + "i", data[offset : offset + _RECORD_MARKER])
        start = offset + _RECORD_MARKER
        end = start + length
        if length < 0 or end + _RECORD_MARKER > len(data):
            raise ValueError(f"{path}: record at byte {offset} runs past the end of the file")
        (trailer,) = struct.unpack(order + "i", data[end : end + _RECORD_MARKER])
        if trailer != length:
            raise ValueError(f"{path}: record markers disagree at byte {offset} ({length} vs {trailer})")
        records.append(data[start:end])
        offset = end + _RECORD_MARKER
    return records


def read_fortran_arrays(
    path: str | os.PathLike[str], *, int32_records: Sequence[int] = ()
) -> list[np.ndarray]:
    """Every record of a table file as a flat array, in file order.

    Records are Fortran arrays written whole, so a record of ``n*m`` doubles
    reshapes as ``(n, m)`` with ``order="F"``.  Records are ``float64``
    except those listed in ``int32_records``: the first record of an ``ec``
    file holds two ``integer*4`` (``mw, mh``), so read it with
    ``int32_records=(0,)``.
    """
    order = fortran_byte_order(path)
    arrays = []
    for index, record in enumerate(read_fortran_records(path)):
        if index in int32_records or len(record) % 8:
            arrays.append(np.frombuffer(record, dtype=order + "i4"))
        else:
            arrays.append(np.frombuffer(record, dtype=order + "f8"))
    return arrays


def collect_efund_outputs(
    workdir: str | os.PathLike[str],
    config: EFUNDConfig,
    counts: Mapping[str, int],
    *,
    returncode: Optional[int] = None,
    executable: Optional[Path] = None,
    stdout: str = "",
    stderr: str = "",
    reason: str = "",
) -> EFUNDResult:
    """Check an EFUND working directory against the files ``config`` implies."""
    directory = Path(workdir).expanduser()
    expected = expected_table_files(config, counts)
    files: dict[str, Path] = {}
    problems: list[str] = []
    for name, size in expected.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        files[name] = path
        actual = path.stat().st_size
        if size is not None and actual != size:
            problems.append(f"{name} is {actual} bytes, expected {size}")
    logs = tuple(
        path for path in (directory / EFUND_STDOUT_NAME, directory / EFUND_STDERR_NAME) if path.is_file()
    )
    if returncode not in (None, 0):
        problems.insert(0, f"efund exited {returncode}")
    status = "success" if not problems else "failed"
    if not reason and problems:
        reason = "; ".join(problems)
    return EFUNDResult(
        returncode=returncode,
        workdir=directory,
        status=status,
        reason=reason,
        files=files,
        expected=expected,
        problems=tuple(problems),
        executable=executable,
        logs=logs,
        stdout=stdout,
        stderr=stderr,
    )


# --- running ---------------------------------------------------------------


def _resolve_executable(config: EFUNDConfig) -> Optional[Path]:
    environment = {**os.environ, **dict(config.env)}
    return resolve_role("efund", explicit=config.executable or None, env=environment)


def _efund_command(config: EFUNDConfig, executable: Path) -> list[str]:
    argv = [str(executable), *config.argv]
    if config.stack_size_kb is None:
        return argv
    # EFUND keeps several nvsum-by-nvsum work arrays on the stack; at 950
    # vessel segments that is four times 7 MB, far beyond the 8 MB default
    # soft limit.  "hard" raises the child's soft limit to whatever the hard
    # limit allows (just under 64 MB on macOS, usually unlimited on Linux); an
    # integer asks for that many kB and falls back to the hard limit when the
    # request exceeds it.
    if config.stack_size_kb == "hard":
        shell = 'ulimit -s $(ulimit -Hs) 2>/dev/null; exec "$@"'
    else:
        limit = int(config.stack_size_kb)
        shell = f'ulimit -s {limit} 2>/dev/null || ulimit -s $(ulimit -Hs) 2>/dev/null; exec "$@"'
    return ["bash", "-lc", shell, "efund-runner", *argv]


def run_efund(inputs: EFUNDInputs, config: EFUNDConfig) -> EFUNDResult:
    """Run EFUND in ``inputs.workdir`` and collect the tables it wrote.

    The executable comes from ``config.executable`` or ``$EFITHOME`` (the
    same root as ``efit``; see :mod:`vaft.code.efit.toolchain`).  Without one
    the result is ``skipped`` and says how to configure it.
    """
    workdir = Path(inputs.workdir)
    executable = _resolve_executable(config)
    if executable is None:
        return EFUNDResult(
            returncode=None,
            workdir=workdir,
            status="skipped",
            reason=unconfigured_reason("efund"),
            expected=expected_table_files(config, inputs.counts),
        )
    if not os.access(executable, os.X_OK):
        return EFUNDResult(
            returncode=None,
            workdir=workdir,
            status="skipped",
            reason=f"missing executable: {executable}",
            executable=executable,
            expected=expected_table_files(config, inputs.counts),
        )
    if not (workdir / MHDIN_NAME).is_file():
        raise FileNotFoundError(f"{workdir / MHDIN_NAME} does not exist; call prepare_efund_inputs first")
    env = os.environ.copy()
    env.update(dict(config.env))
    env.setdefault("OMP_NUM_THREADS", "1")
    command = _efund_command(config, executable)
    try:
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            env=env,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.timeout,
            check=False,
        )
        returncode: Optional[int] = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        reason = ""
    except subprocess.TimeoutExpired as error:
        returncode = None
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        reason = f"efund timed out after {config.timeout} seconds"
    (workdir / EFUND_STDOUT_NAME).write_text(stdout, encoding="utf-8")
    (workdir / EFUND_STDERR_NAME).write_text(stderr, encoding="utf-8")
    result = collect_efund_outputs(
        workdir,
        config,
        inputs.counts,
        returncode=returncode,
        executable=executable,
        stdout=stdout,
        stderr=stderr,
        reason=reason,
    )
    if reason:
        result.status = "failed"
    return result


# --- manifest --------------------------------------------------------------


def _vaft_identity() -> dict[str, Any]:
    from vaft.version import __version__

    revision = None
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "describe", "--always", "--dirty"],
            text=True,
            capture_output=True,
            timeout=2.0,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            revision = completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"version": __version__, "revision": revision}


def _table_digest(files: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(files.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_table_manifest(
    result: EFUNDResult,
    inputs: EFUNDInputs,
    config: EFUNDConfig,
    *,
    label: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write :data:`TABLE_MANIFEST_NAME` beside the tables and return its path.

    Refuses a result that is not a success: a manifest must describe a
    complete table, not a partial directory.
    """
    if not result.ok:
        raise ValueError(f"cannot write a table manifest for a {result.status} run: {result.reason}")
    hashes = {name: _sha256(path) for name, path in sorted(result.files.items())}
    sizes = {name: path.stat().st_size for name, path in sorted(result.files.items())}
    identity = executable_identity(result.executable, "efund") if result.executable else None
    payload = {
        "schema_version": 1,
        "code": "efund",
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {**inputs.geometry.machine, "counts": dict(inputs.counts)},
        "geometry_groups": inputs.geometry.group_summary(),
        "efund": {
            "executable": asdict(identity) if identity else None,
            "argv": config.argv,
            "stack_size_kb": config.stack_size_kb,
            "input": {"name": MHDIN_NAME, "sha256": inputs.mhdin_sha256},
            "config": config.to_dict(),
            "config_sha256": config.sha256,
        },
        "table": {
            "suffix": config.table_suffix,
            "identity": _table_digest(hashes),
            "files": {
                name: {"sha256": hashes[name], "size": sizes[name], "expected_size": result.expected.get(name)}
                for name in hashes
            },
        },
        "vaft": _vaft_identity(),
    }
    if extra:
        payload["extra"] = dict(extra)
    destination = Path(result.workdir) / TABLE_MANIFEST_NAME
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result.manifest = destination
    return destination


def read_table_manifest(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The manifest of a table directory, or ``None`` when it has none."""
    path = Path(directory).expanduser() / TABLE_MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table_identity(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """What an EFIT run records about the table directory it consumed.

    A manifested directory is identified by its manifest's table digest; a
    legacy directory only by the hash of its ``mhdin.dat``, and is marked as
    having no recorded provenance.
    """
    root = Path(directory).expanduser()
    mhdin = root / MHDIN_NAME
    record: dict[str, Any] = {
        "dir": str(root),
        "mhdin_sha256": _sha256(mhdin) if mhdin.is_file() else None,
        "manifest": None,
        "identity": None,
        "provenance": "unrecorded",
    }
    manifest = read_table_manifest(root)
    if manifest is not None:
        record["manifest"] = str(root / TABLE_MANIFEST_NAME)
        record["identity"] = (manifest.get("table") or {}).get("identity")
        record["label"] = manifest.get("label")
        record["provenance"] = "manifest"
    return record
