"""Native TRANSP output containers: a transcript of one ``.CDF``, unconverted.

TRANSP writes its whole run into a single netCDF file -- roughly 1900
variables for a MAST case -- carrying scalars, time series, and radial
profiles on two interleaved grids.  This layer reads it and stops there.
Nothing here writes an IDS and nothing here converts units, following the
split that :mod:`vaft.code.nubeam` and :mod:`vaft.code.gpec` already keep:
the layer above owns the interpretation, and it cannot own it if this one has
already quietly reinterpreted the file.

Three things the file forces, each of which the code this replaces got wrong:

- **A variable's grid comes from its dimension name, never its length.**
  ``X`` holds zone centres and ``XB`` zone outer boundaries, and in a real run
  they are *the same length* -- 20 and 20 for the MAST case -- so a reader
  that compares array lengths silently returns every boundary quantity as a
  centre quantity.  They interleave: ``X = 0.025, 0.075, ...`` against
  ``XB = 0.05, 0.10, ...``, so ``XB[i]`` bounds zone ``i`` from outside, and a
  cumulative sum over zones is therefore an ``XB`` quantity.
- **Units come from the variable's own attribute.**  ``NE`` says
  ``N/CM**3``, ``TQIN`` says ``Nt-M/CM3``, ``PLFLX`` says ``Wb/rad``.  A
  reader that takes them from a flag instead is one wrong argument away from a
  silent factor of a million.
- **The profiles are densities, not per-zone integrals.**  This is worth
  stating because the sibling adapter is the other way round: NUBEAM's Plasma
  State writes per-zone integrals, and assuming the family convention carries
  is an error of order the zone volume.  Checked rather than assumed --
  ``sum(TQIN)`` is 5.6e-7, which matches nothing in the file, while
  ``sum(TQIN * DVOL)`` is -0.544 N m, the order of the run's own newton-metre
  torque scalars (``BPHXB`` = -0.713).

The file is netCDF-3 classic, so it is opened through xarray's scipy backend:
``netCDF4`` is not a VAFT dependency and leaving the engine to auto-detection
would exercise an undeclared backend wherever one happens to be installed.
Reading is lazy -- a production ``.CDF`` runs to hundreds of megabytes and a
caller usually wants a handful of variables at one time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "EMPTY_PLACEHOLDERS",
    "PROFILE_GRIDS",
    "TIME_DIMENSIONS",
    "VARIABLE_DESCRIPTIONS",
    "TranspFormatError",
    "TranspOutput",
    "TranspSlice",
    "TranspVariable",
    "read_transp_output",
]


def _first(value: Any) -> Any:
    """First element of an attribute that may be a scalar or a one-element list.

    netCDF backends differ in whether a one-element attribute surfaces as a
    scalar or as a length-1 array, so every attribute read goes through this
    rather than calling ``str`` on whatever the file happened to carry.
    (:func:`vaft.code.gpec._netcdf.scalar_attr` does the same job for the
    GPEC suite; it is not imported here because importing it would pull the
    whole ``vaft.code.gpec`` package in behind a reader that needs nothing
    from it.)
    """
    if isinstance(value, (list, tuple)):
        return _first(value[0]) if value else None
    array = np.asarray(value)
    return array.reshape(-1)[0] if array.ndim else array.item()

#: Dimension names TRANSP uses for its time axis.  ``TIME`` carries the
#: scalars and ``TIME3`` the profiles; in the runs checked they hold the same
#: 56 values, but they are separate dimensions and a reader must accept both.
TIME_DIMENSIONS: tuple[str, ...] = ("TIME", "TIME3")

#: The two radial grids, and what a point on each one means.  ``X`` and ``XB``
#: have the same length in a real file, which is why nothing here resolves a
#: grid by counting.
PROFILE_GRIDS: Mapping[str, str] = {
    "X": "zone centres, where a zone-averaged quantity belongs",
    "XB": "zone outer boundaries; XB[i] bounds zone i, so a cumulative sum over zones lands here",
}

#: What the variables this adapter names actually are, in the units TRANSP
#: itself declares on them -- read from the ``units`` attribute of a real run,
#: not from prose documentation.  An index for discovery, not a mapping to
#: IMAS.
#:
#: Every profile here is a *density*: a per-zone integral would make
#: ``sum(TQIN)`` a torque, and it is 5.6e-7 against a machine that produces
#: order 0.5 N m.  This is the opposite of the Plasma State convention in
#: :mod:`vaft.code.nubeam`, where the same-looking profiles *are* per-zone
#: integrals; do not carry that reading across.
VARIABLE_DESCRIPTIONS: Mapping[str, str] = {
    "TIME": "time base of the scalars [s]",
    "TIME3": "time base of the profiles [s]",
    "X": "zone-centre radial coordinate, sqrt of normalized toroidal flux [-]",
    "XB": "zone-boundary radial coordinate [-]",
    "DVOL": "volume of each zone [cm^3]",
    "DAREA": "cross-sectional area of each zone [cm^2]",
    "NE": "electron density [cm^-3]",
    "NI": "total ion density [cm^-3]",
    "TE": "electron temperature [eV]",
    "TI": "ion temperature [eV]",
    "OMEGA": "toroidal angular velocity [rad/s]",
    "VRPOT": "radial electrostatic potential, on zone boundaries [V]",
    "PLFLX": "poloidal flux enclosed, measured from the axis, on zone boundaries [Wb/rad]",
    "PLFLX2PI": "the same flux in webers; PLFLX2PI / PLFLX is 2*pi [Wb]",
    "PLFLXA": "poloidal flux enclosed by the boundary; equals PLFLX[-1] [Wb/rad]",
    "TQIN": "total input torque density [N m / cm^3]",
    "TQTOTNB": (
        "an empty placeholder -- a zero-valued scalar carrying no radial or time "
        "axis in the runs checked, though it declares torque-density units "
        "(Nt-M/CM3). The total input torque is TQIN"
    ),
}

#: ``TQTOTNB`` is written but carries nothing; asking for it is a mistake
#: worth catching by name rather than returning a scalar zero.
EMPTY_PLACEHOLDERS: Mapping[str, str] = {
    "TQTOTNB": "TQIN",
}


def _at_time(variable: "TranspVariable", index: int) -> np.ndarray:
    """``variable`` at time ``index``, or whole when it carries no time axis.

    The dimension is checked rather than assumed: ``X`` and ``XB`` are
    ``(TIME3, X)`` in the runs looked at, but TRANSP may write a run's grids
    once as plain ``(X,)``, and indexing axis 0 of *that* silently returns a
    single radial point where a whole grid was asked for.
    """
    if any(dim in TIME_DIMENSIONS for dim in variable.dims):
        return np.asarray(variable.values[index])
    return np.asarray(variable.values)


class TranspFormatError(ValueError):
    """The file is not shaped like a TRANSP output."""


@dataclass(frozen=True)
class TranspVariable:
    """One variable, with the name, dimensions and units TRANSP gave it."""

    name: str
    dims: tuple[str, ...]
    units: str
    long_name: str
    values: np.ndarray

    @property
    def grid(self) -> Optional[str]:
        """``"X"``, ``"XB"`` or ``None`` -- from the dimension name."""
        for dim in self.dims:
            if dim in PROFILE_GRIDS:
                return dim
        return None

    @property
    def is_profile(self) -> bool:
        return self.grid is not None


@dataclass
class TranspSlice:
    """One time of a TRANSP run: its grids and the variables on them.

    Values are the file's own -- ``NE`` is still per cubic centimetre here.
    """

    time_s: float
    time_index: int
    x: np.ndarray
    xb: np.ndarray
    _output: "TranspOutput" = field(repr=False)

    def units(self, name: str) -> str:
        return self._output.units(name)

    def describe(self, name: str) -> str:
        return self._output.describe(name)

    def grid_of(self, name: str) -> Optional[str]:
        return self._output.grid_of(name)

    def variable(self, name: str) -> np.ndarray:
        """One variable at this time, in the file's own units.

        A variable with no time dimension is returned whole.
        """
        return _at_time(self._output.variable(name), self.time_index)

    def _on(self, name: str, grid: str) -> np.ndarray:
        actual = self.grid_of(name)
        if actual is None:
            raise TranspFormatError(
                f"{name} is not a radial profile; its dimensions are "
                f"{self._output.variable(name).dims}"
            )
        if actual != grid:
            raise TranspFormatError(
                f"{name} is on the {actual} grid ({PROFILE_GRIDS[actual]}), not {grid}. "
                "Interpolating between the two is a choice a caller has to make "
                "explicitly -- the grids have the same length, so nothing here "
                "will do it silently"
            )
        return self.variable(name)

    def on_x(self, name: str) -> np.ndarray:
        """A zone-centre variable, refusing a zone-boundary one."""
        return self._on(name, "X")

    def on_xb(self, name: str) -> np.ndarray:
        """A zone-boundary variable, refusing a zone-centre one."""
        return self._on(name, "XB")

    @property
    def psi_norm_xb(self) -> np.ndarray:
        """Normalized poloidal flux on the zone boundaries, ``PLFLX / PLFLXA``.

        ``PLFLX`` is measured from the magnetic axis and ``PLFLXA`` is the flux
        enclosed by the boundary, so their ratio is the normalized flux
        directly.  Normalizing by the *first and last samples* instead --
        ``(P - P[0]) / (P[-1] - P[0])`` -- declares the innermost zone
        boundary to be the axis, which moves the whole grid inward by
        ``PLFLX[0] / PLFLXA`` (0.0049 for the MAST reference run).
        """
        flux = self.on_xb("PLFLX")
        # self.variable has already taken this time's sample, so PLFLXA is a
        # scalar by the time it arrives here -- reading it off the whole
        # series instead would pin psi_norm to the first sample's edge flux,
        # which is 0.0272 Wb/rad against 0.0765 at 750 ms in the MAST
        # reference run.  Hence the size check rather than a bare [0].
        enclosed = np.asarray(self.variable("PLFLXA"), dtype=float)
        if enclosed.size != 1:
            raise TranspFormatError(
                f"PLFLXA came back as {enclosed.shape} rather than this time's "
                "single value; psi_norm would be normalized by the wrong sample"
            )
        edge = float(enclosed.reshape(-1)[0])
        if edge == 0:
            raise TranspFormatError("PLFLXA is zero at this time; psi_norm is undefined")
        return flux / edge


@dataclass
class TranspOutput:
    """A TRANSP run's output file, read lazily.

    Use as a context manager; the underlying dataset stays open so a caller
    can pull a few of the file's ~1900 variables without materializing it.
    Every variable read is then held for the life of the object -- the cache
    does not evict, so a caller that walks all ~1900 ends up holding the
    whole file.  Read what you need, or open the file again.
    """

    path: Path
    _dataset: Any = field(repr=False, default=None)
    _cache: dict = field(repr=False, default_factory=dict)

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "TranspOutput":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    @property
    def _open(self):
        """The open dataset, refusing a closed one by name.

        Without this a closed file half-works: whatever a caller happened to
        read before :meth:`close` still answers out of the cache, and
        anything else fails deep in xarray with an ``AttributeError`` about
        ``NoneType`` that says nothing about the real cause.
        """
        if self._dataset is None:
            raise TranspFormatError(
                f"{self.path.name} is closed; open it again with "
                "read_transp_output to read more of it"
            )
        return self._dataset

    # -- structure -------------------------------------------------------
    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._open.variables)

    @property
    def dims(self) -> Mapping[str, int]:
        return {str(name): int(size) for name, size in self._open.sizes.items()}

    def units(self, name: str) -> str:
        """The file's own ``units`` attribute, stripped; ``""`` when it has none."""
        return str(_first(self._open[name].attrs.get("units", "")) or "").strip()

    def long_name(self, name: str) -> str:
        return str(_first(self._open[name].attrs.get("long_name", "")) or "").strip()

    def describe(self, name: str) -> str:
        """What a variable is, or a note that this adapter does not catalogue it."""
        described = VARIABLE_DESCRIPTIONS.get(name)
        if described is not None:
            return described
        if name not in self._open.variables:
            return f"{name}: not in {self.path.name}"
        long_name = self.long_name(name)
        units = self.units(name)
        if long_name or units:
            return f"{long_name or name} [{units or 'no units declared'}] (TRANSP's own description)"
        return f"{name}: native TRANSP quantity, meaning not catalogued here"

    def grid_of(self, name: str) -> Optional[str]:
        """Which radial grid a variable is on, from its dimension name."""
        return self.variable(name).grid

    def variable(self, name: str) -> TranspVariable:
        """One variable, materialized and cached."""
        if name in self._cache:
            return self._cache[name]
        replacement = EMPTY_PLACEHOLDERS.get(name)
        if replacement is not None:
            raise TranspFormatError(
                f"{name} is an empty placeholder in TRANSP output -- {VARIABLE_DESCRIPTIONS[name]}. "
                f"Use {replacement}"
            )
        if name not in self._open.variables:
            raise KeyError(f"{self.path.name} has no variable {name!r}")
        entry = self._open[name]
        variable = TranspVariable(
            name=name,
            dims=tuple(str(dim) for dim in entry.dims),
            units=self.units(name),
            long_name=self.long_name(name),
            values=np.asarray(entry.values),
        )
        self._cache[name] = variable
        return variable

    # -- time ------------------------------------------------------------
    @property
    def time(self) -> np.ndarray:
        """The scalar time base [s]."""
        return np.asarray(self.variable("TIME").values, dtype=float)

    def time_index(self, time_s: float) -> int:
        """Index of the sample nearest ``time_s``."""
        times = self.time
        if times.size == 0:
            raise TranspFormatError(f"{self.path.name} carries no times")
        return int(np.argmin(np.abs(times - float(time_s))))

    def slice(self, time_s: float) -> TranspSlice:
        """The run at the sample nearest ``time_s``.

        The chosen time is on the result as ``time_s``/``time_index``: a run's
        samples are irregular, so which one was taken is part of the answer.
        """
        index = self.time_index(time_s)
        return TranspSlice(
            time_s=float(self.time[index]),
            time_index=index,
            x=np.asarray(_at_time(self.variable("X"), index), dtype=float),
            xb=np.asarray(_at_time(self.variable("XB"), index), dtype=float),
            _output=self,
        )


def _check_not_truncated(output: TranspOutput) -> None:
    """Refuse a file that is shorter than its own header says it should be.

    A TRANSP run killed mid-write leaves a complete header over incomplete
    data, and a classic netCDF reads the missing tail back as zeros rather
    than raising -- so presence of a variable is not evidence that it is
    whole.  The header states every variable's shape and type, so the file
    has a minimum size that can simply be checked against the one on disk.
    Record variables are interleaved and padded, so the sum is a floor rather
    than an exact size, which is all that is wanted here.  The technique is
    the one :func:`vaft.code.gpec._solvers._check_nc_variable` uses on solver
    output.
    """
    with open(output.path, "rb") as handle:
        if handle.read(3) != b"CDF":
            # Not classic; a truncated HDF5 file raises on read instead, so
            # there is nothing for a size floor to add.
            return
    needed = 0
    for entry in output._open.variables.values():
        # The on-disk type, not the decoded one: xarray promotes a variable
        # carrying a _FillValue to float64, which would inflate the floor and
        # refuse a healthy file.
        dtype = entry.encoding.get("dtype", entry.dtype)
        needed += int(np.prod(entry.shape)) * int(np.dtype(dtype).itemsize)
    actual = output.path.stat().st_size
    if actual < needed:
        raise TranspFormatError(
            f"{output.path.name} is truncated: {actual} bytes on disk against the "
            f"{needed} bytes of variable data its own header declares. A run "
            "killed mid-write reads its missing tail back as zeros rather than "
            "raising, so this is checked rather than discovered later"
        )


def _check_integrity(output: TranspOutput) -> None:
    """Assertions the file itself makes possible, checked once on open."""
    for name in ("TIME", "X", "XB"):
        if name not in output.variables:
            raise TranspFormatError(
                f"{output.path.name} has no {name!r} variable; it does not look like "
                "a TRANSP output file"
            )

    _check_not_truncated(output)

    # TRANSP writes the scalars on TIME and the profiles on TIME3.  They are
    # separate dimensions but the same axis, and this reader picks a sample
    # index from TIME and applies it to profiles on TIME3, so a file where
    # they disagree would hand back a profile from a different instant than
    # the one it reports.  Refuse rather than mislabel.
    if "TIME3" in output.variables:
        times = output.time
        times3 = np.asarray(output.variable("TIME3").values, dtype=float)
        if times.shape != times3.shape or not np.allclose(times, times3, rtol=1e-6, atol=0.0):
            raise TranspFormatError(
                f"TIME is {times.shape} and TIME3 is {times3.shape}, and they do not "
                "hold the same instants; this reader chooses a sample on TIME and "
                "reads profiles on TIME3, so it cannot say which time a profile is from"
            )

    x = np.asarray(output.variable("X").values, dtype=float)
    xb = np.asarray(output.variable("XB").values, dtype=float)
    if x.shape != xb.shape:
        raise TranspFormatError(
            f"X is {x.shape} and XB is {xb.shape}; this reader expects one zone "
            "centre per zone boundary, which is how the runs it was written "
            "against are shaped. A file that also writes the innermost boundary "
            "(nzones + 1 points on XB) is a convention this reader has not been "
            "checked against, not a broken file"
        )
    # Every time, not just the first: the grids are declared time-varying, and
    # a truncated tail read back as zeros is exactly a row that has stopped
    # increasing.  They are a few kilobytes, so this costs nothing.
    rows_x = x.reshape(-1, x.shape[-1])
    rows_xb = xb.reshape(-1, xb.shape[-1])
    for index, (row_x, row_xb) in enumerate(zip(rows_x, rows_xb)):
        if not (np.all(np.diff(row_x) > 0) and np.all(np.diff(row_xb) > 0)):
            raise TranspFormatError(
                f"X and XB must both increase outward; they do not at sample {index}"
            )
        if not np.all(row_x < row_xb):
            raise TranspFormatError(
                "every zone centre must lie inside its own outer boundary; X and XB "
                f"look swapped or misaligned at sample {index}"
            )

    if "PLFLX" in output.variables and "PLFLX2PI" in output.variables:
        flux = np.asarray(output.variable("PLFLX").values, dtype=float)
        webers = np.asarray(output.variable("PLFLX2PI").values, dtype=float)
        finite = np.isfinite(flux) & np.isfinite(webers) & (flux != 0)
        if not np.any(finite):
            raise TranspFormatError(
                "PLFLX holds no non-zero finite value, so the file's own Wb versus "
                "Wb/rad statement cannot be checked and its flux cannot be trusted"
            )
        ratio = float(np.nanmedian(webers[finite] / flux[finite]))
        if not np.isclose(ratio, 2.0 * np.pi, rtol=1e-4):
            raise TranspFormatError(
                f"PLFLX2PI / PLFLX is {ratio:.6f}, not 2*pi: the file's own "
                "Wb versus Wb/rad statement does not hold, so its flux cannot "
                "be trusted"
            )


def read_transp_output(path: str | Path) -> TranspOutput:
    """Open a TRANSP output ``.CDF``.

    Lazy: the dataset stays open and variables materialize on first use.  Use
    it as a context manager, or call :meth:`TranspOutput.close`.
    """
    import xarray as xr

    path = Path(path).expanduser()
    if path.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory; read_transp_output takes one .CDF file "
            "(collect_transp_outputs takes a run directory)"
        )
    # engine="scipy" pins the declared backend and the netCDF-3 classic format
    # TRANSP writes; netCDF4 is not a VAFT dependency, so letting xarray
    # auto-detect would use whichever backend happens to be installed.
    try:
        dataset = xr.open_dataset(path, engine="scipy", decode_times=False)
    except Exception as exc:
        # The backend reads a classic file's non-record variables at open time,
        # so a run killed mid-write usually fails here -- with a message about
        # reshaping an array, which says nothing about the real cause.
        raise TranspFormatError(
            f"could not read {path.name} ({path.stat().st_size} bytes on disk): {exc}. "
            "A TRANSP run killed mid-write leaves a complete header over incomplete "
            "data, which is what this normally is"
        ) from exc
    output = TranspOutput(path=path, _dataset=dataset)
    try:
        _check_integrity(output)
    except Exception:
        output.close()
        raise
    return output
