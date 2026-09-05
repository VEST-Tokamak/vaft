"""Native NUBEAM output containers and directory collection.

This layer parses NUBEAM's own files and stops there. Nothing here writes an
IDS: following the split that ``vaft/machine_mapping/mhd_linear.py`` documents,
the IDS-populating layer reads a native container owned by ``vaft.code.<code>``
rather than re-parsing solver output itself. The IMAS mapping for NUBEAM is not
implemented yet -- see issue #490 section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from vaft.code.base import CodeResult

from .config import NUBEAMConfig

#: Written by NUBEAM STEP: the changes it computed, as a partial Plasma State.
STATE_CHANGES = "state_changes.cdf"

#: Profiles in ``state_changes.cdf`` whose physical meaning is unambiguous.
#: Recorded here so a caller can discover what a run produced without knowing
#: NUBEAM's naming; this is an index, not a mapping to IMAS.
PROFILE_DESCRIPTIONS: Mapping[str, str] = {
    "pbe": "beam power density to electrons [W/m^3]",
    "pbi": "beam power density to ions [W/m^3]",
    "pbth": "beam power density to thermalization [W/m^3]",
    "nbeami": "fast ion density per beam species [m^-3]",
    "curbeam": "beam-driven current density [A/m^2]",
    "curfusn": "fusion-product-driven current density [A/m^2]",
    "tqbe": "toroidal torque density to electrons [N/m^2]",
    "tqbi": "toroidal torque density to ions [N/m^2]",
    "tqbjxb": "JxB toroidal torque density [N/m^2]",
    "pfuse": "fusion power density to electrons [W/m^3]",
    "pfusi": "fusion power density to ions [W/m^3]",
    "eperp_beami": "fast ion perpendicular energy density [J/m^3]",
    "epll_beami": "fast ion parallel energy density [J/m^3]",
    "sbedep": "beam electron deposition rate [m^-3 s^-1]",
    "sbtherm": "beam ion thermalization rate [m^-3 s^-1]",
}

#: Written by NUBEAM STEP when it has fast ions: the xplasma container that
#: carries, among much else, the lost-particle record.
XPLASMA_OUT_SUFFIX = "_xplasma_out.cdf"

#: Columns of the ``LOST_ORBIT`` record, in the order they are packed.
#:
#: xplasma stores the record as three opaque variables -- ``LOST_ORBIT_type``,
#: ``_iwork`` and ``_r8work`` -- rather than as named arrays, so the column
#: order is not recoverable from the file. It is taken from two independent
#: sources that agree: the reference reader shipped on the VEST cluster
#: (``/home/leecyid/tool/lost_orbit.py``), and NUBEAM's own writer, which names
#: the same quantities in the same order (``lost_orbit_r8get1`` calls in
#: ``nubeam/gfbm_set_track_data.f90:503-543``).
LOST_PARTICLE_FIELDS = (
    "time",    # when the marker was lost [s]
    "beam",    # originating beam index
    "efrac",   # energy component: 1 full, 2 half, 3 third
    "ptcl",    # physical particles represented by the marker
    "rlost",   # major radius of the loss [m]
    "zlost",   # elevation of the loss [m]
    "energy",  # marker energy at loss [keV]
    "vfrac",   # pitch, v_parallel / v
    "lstype",  # loss channel: 1 prompt, >1 orbit
    "spec",    # fast ion species index
)

#: ``lstype`` below this is a prompt loss; at or above it, an orbit loss.
#: The threshold is the reference reader's (``lstype < 1.0001``), kept because
#: the field is stored as a real.
LOST_PROMPT_MAX = 1.0001

#: Marker columns in the birth file, named ``bs_<field>_<species>_MCBEAM``.
BIRTH_MARKER_FIELDS = (
    "r",      # major radius of the deposition point [cm]
    "z",      # elevation of the deposition point [cm]
    "rgc",    # guiding-centre major radius [cm]
    "zgc",    # guiding-centre elevation [cm]
    "xksid",  # pitch, v_parallel / v
    "einj",   # injection energy [eV]
    "wght",   # Monte Carlo marker weight
    "zeta",   # toroidal angle [degrees]
    "time",   # deposition time [s]
    "ib",     # originating beam index
)


@dataclass
class NUBEAMBirthMarkers:
    """Deposition markers from ``<runid>_birth_cpu*.cdf_*``.

    One entry per Monte Carlo deposition track. The columns line up closely
    with what a marker-based source description needs, but this container
    keeps NUBEAM's own names and units; no conversion is applied.
    """

    path: Path
    species: str
    count: int
    columns: Mapping[str, Any] = field(default_factory=dict)


#: Heading of the power-accounting block NUBEAM prints at the end of a step.
POWER_BALANCE_HEADING = "rough power balance"

#: ``1.138D+05`` -- Fortran writes a D exponent that Python will not parse.
_FORTRAN_REAL = re.compile(r"^[-+]?\d*\.?\d+(?:[DdEe][-+]?\d+)?$")

#: ``    -electron heating:     1.138D+05`` and ``->residual:  -1.468D+03``.
#: ``->`` must precede ``-`` in the alternation, or the residual parses as an
#: ordinary negative entry named ">residual".
_BALANCE_ENTRY = re.compile(
    r"^\s*(?P<sign>->|[-+])\s*(?P<name>[^:]+?):\s+(?P<value>\S+)\s*$"
)


@dataclass
class NUBEAMPowerBalance:
    """NUBEAM's own end-of-step power accounting for one fast ion species.

    Every entry is in watts, and the code prints the closure explicitly, so the
    residual is NUBEAM's own statement of how well the budget balances rather
    than something recomputed here.
    """

    species: str
    #: Entry name as NUBEAM prints it -> watts. Sources are positive, sinks
    #: negative, exactly as the log signs them.
    entries: Mapping[str, float] = field(default_factory=dict)
    residual: Optional[float] = None

    @property
    def injected(self) -> Optional[float]:
        """Injected power [W], the quantity everything else is a fraction of."""
        for name, value in self.entries.items():
            if "injected power" in name:
                return value
        return None

    def sinks(self) -> dict[str, float]:
        """Loss and heating channels, as positive watts."""
        return {
            name: -value
            for name, value in self.entries.items()
            if value < 0.0
        }

    def fractions(self) -> dict[str, float]:
        """Each sink as a fraction of injected power, or ``{}`` if unknown."""
        total = self.injected
        if not total:
            return {}
        return {name: value / total for name, value in self.sinks().items()}


def parse_power_balance(text: str) -> tuple[NUBEAMPowerBalance, ...]:
    """Read the power-accounting blocks out of a NUBEAM step log.

    This is the summary the VEST workflow has always read by eye. It is parsed
    rather than recomputed: NUBEAM states the budget and its own residual, and
    a reconstruction from the profiles would be a different number carrying
    different assumptions.
    """
    blocks: list[NUBEAMPowerBalance] = []
    lines = text.splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if POWER_BALANCE_HEADING in line
        )
    except StopIteration:
        return ()

    species = ""
    entries: dict[str, float] = {}
    residual: Optional[float] = None

    def flush() -> None:
        nonlocal species, entries, residual
        if species and entries:
            blocks.append(
                NUBEAMPowerBalance(
                    species=species, entries=dict(entries), residual=residual
                )
            )
        species, entries, residual = "", {}, None

    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        # "H beam ion:" opens a block; a line with no leading sign and a
        # trailing colon is the next heading, so the block is complete.
        match = _BALANCE_ENTRY.match(line)
        if match is None:
            if stripped.endswith(":"):
                flush()
                species = stripped[:-1].strip()
                continue
            if entries:
                break
            continue
        raw = match.group("value")
        if not _FORTRAN_REAL.match(raw):
            continue
        value = float(raw.replace("D", "E").replace("d", "e"))
        name = match.group("name").strip().replace(chr(34), "")
        if match.group("sign") == "->":
            residual = value
        else:
            entries[name] = value if match.group("sign") == "+" else -value
    flush()
    return tuple(blocks)


@dataclass
class NUBEAMLostParticles:
    """Fast ions NUBEAM stopped following, one entry per lost marker.

    Positions are in metres and energies in keV -- xplasma's own units here,
    and *not* the centimetres the birth file uses. No conversion is applied.
    """

    path: Path
    count: int
    columns: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> Any:
        """Boolean mask selecting the prompt-loss markers."""
        import numpy as np

        return np.asarray(self.columns["lstype"]) < LOST_PROMPT_MAX

    def channel_counts(self) -> dict[str, int]:
        """Marker counts per loss channel.

        Worth reading before labelling any figure: NUBEAM's step log calls this
        whole channel "bad orbit loss", but a run can be entirely prompt loss.
        """
        prompt = int(self.prompt.sum())
        return {"prompt": prompt, "orbit": int(self.count - prompt)}


@dataclass
class NUBEAMOutputs:
    """Everything a completed NUBEAM run produced, in NUBEAM's own terms."""

    workdir: Path
    runid: Optional[str] = None
    state_changes: Optional[Path] = None
    plasma_state: Optional[Path] = None
    #: Profile name -> 1-D or 2-D array, straight out of ``state_changes.cdf``.
    profiles: Mapping[str, Any] = field(default_factory=dict)
    #: Scalar diagnostics from ``<runid>_scalars_out.cdf``.
    scalars: Mapping[str, Any] = field(default_factory=dict)
    birth: Optional[NUBEAMBirthMarkers] = None
    lost: Optional[NUBEAMLostParticles] = None
    power_balance: tuple[NUBEAMPowerBalance, ...] = ()
    #: Count of ``xpprof`` out-of-bounds interpolation warnings in step.log.
    interpolation_warnings: int = 0

    def describe(self, name: str) -> str:
        """Documented meaning of a profile, or a note that it is undocumented."""
        return PROFILE_DESCRIPTIONS.get(
            name, f"{name}: native NUBEAM quantity, meaning not catalogued here"
        )


@dataclass
class NUBEAMResult(CodeResult):
    """Result bundle for a NUBEAM run."""

    outputs_native: Optional[NUBEAMOutputs] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.outputs_native is not None


def _open_dataset(path: Path):
    """Open a netCDF file with xarray, matching how the GPEC adapter reads its own.

    ``netCDF4`` is not a VAFT dependency; ``xarray`` is, and
    ``vaft/code/gpec/_netcdf.py`` already reads solver netCDF through it.

    The warning suppressed here is specific and understood. NUBEAM declares
    three square scalar arrays -- ``nbi_outflx``, ``nbi_eescav`` and
    ``nbi_cexflx`` -- with the same dimension twice, ``(dim_00002,
    dim_00002)``. xarray accepts that but warns that most of its functionality
    "is likely to fail silently" on such a variable. Everything here reads
    ``.values`` and nothing else, which is unaffected: the arrays come back
    with the right shape and contents, checked against ``ncdump``. The warning
    is suppressed rather than left to reach the caller, who cannot act on it.
    """
    import warnings

    import xarray as xr

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Duplicate dimension names present"
        )
        return xr.open_dataset(path, decode_times=False)


def _read_variables(path: Path) -> dict[str, Any]:
    import numpy as np

    values: dict[str, Any] = {}
    with _open_dataset(path) as dataset:
        for name, variable in dataset.variables.items():
            data = variable.values
            if data.dtype.kind in "SU":
                # Character arrays are Plasma State bookkeeping, not physics.
                continue
            values[str(name)] = np.asarray(data)
    return values


def _read_birth(path: Path) -> Optional[NUBEAMBirthMarkers]:
    import numpy as np

    with _open_dataset(path) as dataset:
        names = [str(name) for name in dataset.variables]
        # bs_<field>_<species>_MCBEAM -- recover the species from any column.
        prefixes = [n for n in names if n.startswith("bs_") and n.endswith("_MCBEAM")]
        if not prefixes:
            return None
        sample = prefixes[0][len("bs_"): -len("_MCBEAM")]
        species = sample.split("_", 1)[1] if "_" in sample else ""

        columns: dict[str, Any] = {}
        for field_name in BIRTH_MARKER_FIELDS:
            variable = f"bs_{field_name}_{species}_MCBEAM"
            if variable in dataset.variables:
                columns[field_name] = np.asarray(dataset[variable].values)
        if not columns:
            return None
        count = int(len(next(iter(columns.values()))))
    return NUBEAMBirthMarkers(
        path=path, species=species, count=count, columns=columns
    )


def _read_lost_particles(path: Path) -> Optional[NUBEAMLostParticles]:
    """Decode the ``LOST_ORBIT`` record from an xplasma output file.

    The record is three flat variables. ``iwork`` holds a small header:
    ``iwork[1]`` is the number of lost markers, and ``iwork[2:]`` are 1-based
    start offsets into ``r8work``, one per column of
    :data:`LOST_PARTICLE_FIELDS`.

    Each column is read as ``count`` values from its own start offset, rather
    than as the span between consecutive offsets. That is deliberate: the
    reference reader takes the last column as ``r8work[iwork[11]-1:iwork[12]-1]``
    and ``iwork[12]`` is zero -- the offset list ends -- so it silently drops
    the final marker of that one column. Reading a fixed length keeps every
    column the same length, which is what the caller can reason about.

    Only these three variables are touched. The containing file routinely
    exceeds 50 MB, and xarray reads lazily, so nothing else is loaded.
    """
    import numpy as np

    with _open_dataset(path) as dataset:
        if "LOST_ORBIT_iwork" not in dataset.variables:
            return None
        iwork = np.asarray(dataset["LOST_ORBIT_iwork"].values).ravel().astype(int)
        r8work = np.asarray(dataset["LOST_ORBIT_r8work"].values).ravel()

    if iwork.size < 2 + len(LOST_PARTICLE_FIELDS):
        return None
    count = int(iwork[1])
    if count <= 0:
        # A run that lost nothing is a normal result, not a missing record.
        return NUBEAMLostParticles(
            path=path,
            count=0,
            columns={name: np.empty(0) for name in LOST_PARTICLE_FIELDS},
        )

    columns: dict[str, Any] = {}
    for index, name in enumerate(LOST_PARTICLE_FIELDS):
        start = int(iwork[2 + index]) - 1
        if start < 0 or start + count > r8work.size:
            return None
        columns[name] = r8work[start : start + count]
    return NUBEAMLostParticles(path=path, count=count, columns=columns)


def collect_nubeam_outputs(
    workdir: str | Path,
    config: Optional[NUBEAMConfig] = None,
    *,
    returncode: Optional[int] = None,
) -> NUBEAMResult:
    """Read a NUBEAM run directory without re-running anything.

    Every product is optional: a directory from an INIT-only run, or one whose
    step wrote no birth file, is a normal input here rather than an error.
    """
    config = config or NUBEAMConfig()
    directory = Path(workdir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"NUBEAM work directory does not exist: {directory}")

    state_changes = directory / STATE_CHANGES
    profiles = _read_variables(state_changes) if state_changes.is_file() else {}

    runid = config.runid
    runid_file = directory / "nubeam_comp_exec.RUNID"
    if runid_file.is_file():
        recorded = runid_file.read_text(encoding="utf-8", errors="replace").strip()
        if recorded:
            runid = recorded.split()[0]

    scalars: dict[str, Any] = {}
    scalars_file = directory / f"{runid}_scalars_out.cdf"
    if scalars_file.is_file():
        scalars = _read_variables(scalars_file)

    birth = None
    birth_files = sorted(directory.glob(f"{runid}_birth_cpu*"))
    if birth_files:
        birth = _read_birth(birth_files[0])

    lost = None
    xplasma_out = directory / f"{runid}{XPLASMA_OUT_SUFFIX}"
    if xplasma_out.is_file():
        lost = _read_lost_particles(xplasma_out)

    warnings_count = 0
    balance: tuple[NUBEAMPowerBalance, ...] = ()
    step_log = directory / "step.log"
    if step_log.is_file():
        text = step_log.read_text(encoding="utf-8", errors="replace")
        warnings_count = text.count(
            "x arguments for interpolation are out of bounds"
        )
        balance = parse_power_balance(text)

    plasma_state = None
    for candidate in sorted(directory.glob("*.cdf")):
        if candidate.name == STATE_CHANGES:
            continue
        if candidate.stem == runid or candidate.name == f"{runid}.cdf":
            plasma_state = candidate
            break

    native = NUBEAMOutputs(
        workdir=directory,
        runid=runid,
        state_changes=state_changes if state_changes.is_file() else None,
        plasma_state=plasma_state,
        profiles=profiles,
        scalars=scalars,
        birth=birth,
        lost=lost,
        power_balance=balance,
        interpolation_warnings=warnings_count,
    )

    logs = tuple(
        path for path in (directory / "init.log", directory / "step.log") if path.is_file()
    )
    return NUBEAMResult(
        returncode=returncode,
        workdir=directory,
        logs=logs,
        outputs={"state_changes": (state_changes,) if state_changes.is_file() else ()},
        outputs_native=native,
    )
