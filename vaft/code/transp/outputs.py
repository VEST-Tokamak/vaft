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

from vaft.code.gpec._netcdf import float_attr, scalar_attr

__all__ = [
    "PROFILE_GRIDS",
    "TIME_DIMENSIONS",
    "VARIABLE_DESCRIPTIONS",
    "TranspOutput",
    "TranspSlice",
    "TranspVariable",
    "read_transp_output",
]

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
        "an empty placeholder -- a dimensionless scalar of value zero in the runs "
        "checked. The total input torque is TQIN"
    ),
}

#: ``TQTOTNB`` is written but carries nothing; asking for it is a mistake
#: worth catching by name rather than returning a scalar zero.
EMPTY_PLACEHOLDERS: Mapping[str, str] = {
    "TQTOTNB": "TQIN",
}


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
        variable = self._output.variable(name)
        if any(dim in TIME_DIMENSIONS for dim in variable.dims):
            return np.asarray(variable.values[self.time_index])
        return np.asarray(variable.values)

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
        edge = float(np.asarray(self.variable("PLFLXA")).reshape(-1)[0])
        if edge == 0:
            raise TranspFormatError("PLFLXA is zero at this time; psi_norm is undefined")
        return flux / edge


@dataclass
class TranspOutput:
    """A TRANSP run's output file, read lazily.

    Use as a context manager; the underlying dataset stays open so a caller
    can pull a few of the file's ~1900 variables without materializing it.
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

    # -- structure -------------------------------------------------------
    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._dataset.variables)

    @property
    def dims(self) -> Mapping[str, int]:
        return {str(name): int(size) for name, size in self._dataset.sizes.items()}

    def units(self, name: str) -> str:
        """The file's own ``units`` attribute, stripped; ``""`` when it has none."""
        return str(scalar_attr(self._dataset[name].attrs.get("units", "")) or "").strip()

    def long_name(self, name: str) -> str:
        return str(scalar_attr(self._dataset[name].attrs.get("long_name", "")) or "").strip()

    def describe(self, name: str) -> str:
        """What a variable is, or a note that this adapter does not catalogue it."""
        described = VARIABLE_DESCRIPTIONS.get(name)
        if described is not None:
            return described
        if name not in self._dataset.variables:
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
        if name not in self._dataset.variables:
            raise KeyError(f"{self.path.name} has no variable {name!r}")
        entry = self._dataset[name]
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
            x=np.asarray(self.variable("X").values[index], dtype=float),
            xb=np.asarray(self.variable("XB").values[index], dtype=float),
            _output=self,
        )


def _check_integrity(output: TranspOutput) -> None:
    """Assertions the file itself makes possible, checked once on open.

    A TRANSP run killed mid-write leaves a readable header over truncated
    data, and a classic netCDF reads the missing tail back as silent zeros
    rather than raising, so presence of a variable is not evidence it is
    whole.
    """
    for name in ("TIME", "X", "XB"):
        if name not in output.variables:
            raise TranspFormatError(
                f"{output.path.name} has no {name!r} variable; it does not look like "
                "a TRANSP output file"
            )

    x = np.asarray(output.variable("X").values, dtype=float)
    xb = np.asarray(output.variable("XB").values, dtype=float)
    if x.shape != xb.shape:
        raise TranspFormatError(
            f"X is {x.shape} and XB is {xb.shape}; this reader expects one zone "
            "centre per zone boundary"
        )
    first_x, first_xb = x.reshape(-1, x.shape[-1])[0], xb.reshape(-1, xb.shape[-1])[0]
    if not (np.all(np.diff(first_x) > 0) and np.all(np.diff(first_xb) > 0)):
        raise TranspFormatError("X and XB must both increase outward")
    if not np.all(first_x < first_xb):
        raise TranspFormatError(
            "every zone centre must lie inside its own outer boundary; X and XB "
            "look swapped or misaligned"
        )

    if "PLFLX" in output.variables and "PLFLX2PI" in output.variables:
        flux = np.asarray(output.variable("PLFLX").values, dtype=float)
        webers = np.asarray(output.variable("PLFLX2PI").values, dtype=float)
        finite = np.isfinite(flux) & np.isfinite(webers) & (flux != 0)
        if np.any(finite):
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
    dataset = xr.open_dataset(path, engine="scipy", decode_times=False)
    output = TranspOutput(path=path, _dataset=dataset)
    try:
        _check_integrity(output)
    except Exception:
        output.close()
        raise
    return output
