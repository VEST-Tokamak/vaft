"""Kinetic profiles as a file-format container, and GPEC ``.kin`` I/O.

A kinetic profile set is a radial coordinate plus the densities,
temperatures, rotation frequencies and pressures defined on it.  Three file
formats carry them -- GPEC's ``.kin``, the Osborne ``pfile`` and MARS's
``PROF*.IN`` -- and this module is the one container all three read into and
write from.

**The canonical unit set is fixed by the container**, not carried per
instance: densities in m^-3, temperatures in eV, angular frequencies in
rad/s, pressures in Pa, electric fields in V/m.  Those are the units GPEC's
own reader consumes, so the ``.kin`` path needs no conversion at all; the
pfile and MARS readers convert on the way in, through the single table
:data:`KINETIC_UNITS`.

Two things this container is careful about, because the code it replaces was
not:

- **The radial coordinate is never rescaled on read.**  A reader records what
  the file said and how it was normalised (:class:`PsiNormalization`); making
  it span exactly [0, 1] is :func:`normalize_psi`, an explicit operation.
  This matters physically: GPEC's ``read_kin`` re-splines a ``.kin`` onto a
  uniform 101-point [0, 1] grid *with extrapolation* (``nkin = 100``
  intervals, ``inputs.f90:204,226``), so it expects a truncated edge and
  handles it.  Stretching the axis instead moves every
  interior point -- for the reference MAST-U file, which spans
  psi_n = 0.00495 to 1.0, by up to half a percent of the minor radius, and
  permanently once the stretched values are written back.
- **Toroidal rotation and the E×B frequency are different fields.**
  ``omega_tor`` and ``omega_exb`` are never merged, and :func:`write_kin`
  refuses a profile set that has only the former: the ``.kin`` rotation
  column is omega_E, and writing a toroidal rotation into it is silent and
  wrong.  The formats invite the mistake -- MARS writes the two to
  ``PROFROT.IN`` and ``PROFWE.IN``, and a pfile has ten rotation-like
  sections.

Provenance: every field records where it came from (a file section, a code
variable) in :attr:`KineticProfiles.provenance`, so a converted profile set
can say which TRANSP variable or pfile section produced each quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "KINETIC_UNITS",
    "KIN_COLUMNS",
    "KIN_HEADER",
    "PROFILE_FIELDS",
    "KineticProfiles",
    "PsiNormalization",
    "Species",
    "normalize_psi",
    "read_kin",
    "write_kin",
]

#: The six ``.kin`` columns, in file order, as GPEC's ``read_kin`` defines
#: them (``pentrc/inputs.f90``): ``psi_n, n_i, n_e, T_i, T_e, omega_E``.
KIN_COLUMNS: tuple[str, ...] = ("psi_norm", "n_i", "n_e", "T_i", "T_e", "omega_exb")

#: The header line GPEC's own example files carry.  It is a comment as far as
#: the reader is concerned -- what fixes the meaning is the column order.
KIN_HEADER = (
    "             psi         ni(m^-3)         ne(m^-3)"
    "           ti(eV)           te(eV)      wexb(rad/s)"
)

#: The container's units, per field.  One set, fixed here; a reader converts
#: into them and a writer converts out of them.  Exported as
#: ``vaft.data.KINETIC_UNITS``; the pfile and MARS readers convert through
#: this table and no other.
KINETIC_UNITS: Mapping[str, str] = {
    "psi_norm": "-",
    "n_e": "m^-3",
    "n_i": "m^-3",
    "n_z": "m^-3",
    "n_fast": "m^-3",
    "T_e": "eV",
    "T_i": "eV",
    "T_z": "eV",
    "omega_tor": "rad/s",
    "omega_exb": "rad/s",
    "omega_pol": "rad/s",
    "p_total": "Pa",
    "p_fast": "Pa",
    "e_radial": "V/m",
}

#: Profile fields, in the order a reader should prefer to report them.
PROFILE_FIELDS: tuple[str, ...] = tuple(name for name in KINETIC_UNITS if name != "psi_norm")


@dataclass(frozen=True)
class PsiNormalization:
    """How a profile set's radial coordinate came to be what it is.

    ``method`` is ``"as_read"`` for a coordinate taken from a file unchanged,
    or the name of the operation that produced it.  Recording this is the
    point: a coordinate that has been rescaled looks exactly like one that
    has not.
    """

    method: str = "as_read"
    source: str = ""
    axis_value: float | None = None
    edge_value: float | None = None


@dataclass(frozen=True)
class Species:
    """One ion species from a pfile's ``N Z A`` block.

    A ``.kin`` carries no species block, so :func:`read_kin` never builds
    one; it is the type of :attr:`KineticProfiles.species`, which the pfile
    reader populates.
    """

    label: str
    n: float
    z: float
    a: float


def _sealed(values) -> np.ndarray:
    """A read-only *view* of ``values``, leaving the caller's array writable.

    ``frozen=True`` stops the attributes being rebound and nothing else, so an
    array reached through one would still be editable in place -- and a
    provenance record that says "as read" would then be untrue.  A view has
    its own writeable flag, so sealing it here does not reach back into the
    array the caller passed in.
    """
    array = np.asarray(values).view()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class KineticProfiles:
    """Kinetic profiles on one radial coordinate, in the container's units.

    Every profile field is optional because the formats differ in what they
    carry; :meth:`available` says which are present.  Anything a file holds
    that has no field of its own is kept in :attr:`extras` under its own
    name, so nothing is dropped.

    The arrays and mappings are sealed on construction, so a profile set
    cannot be edited in place behind its own provenance record; build a
    changed one with :func:`dataclasses.replace`.  ``eq`` is off because the
    fields are arrays: a generated ``__eq__`` would raise rather than answer.
    """

    psi_norm: np.ndarray
    n_e: np.ndarray | None = None
    n_i: np.ndarray | None = None
    n_z: np.ndarray | None = None
    n_fast: np.ndarray | None = None
    T_e: np.ndarray | None = None
    T_i: np.ndarray | None = None
    T_z: np.ndarray | None = None
    omega_tor: np.ndarray | None = None
    omega_exb: np.ndarray | None = None
    omega_pol: np.ndarray | None = None
    p_total: np.ndarray | None = None
    p_fast: np.ndarray | None = None
    e_radial: np.ndarray | None = None
    normalization: PsiNormalization = field(default_factory=PsiNormalization)
    species: tuple[Species, ...] = ()
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "psi_norm", _sealed(self.psi_norm))
        size = self.psi_norm.size
        for name in PROFILE_FIELDS:
            values = getattr(self, name)
            if values is None:
                continue
            if np.asarray(values).size != size:
                raise ValueError(
                    f"{name} has {np.asarray(values).size} samples but psi_norm has "
                    f"{size}; every profile is on the same radial coordinate"
                )
            object.__setattr__(self, name, _sealed(values))
        object.__setattr__(
            self,
            "extras",
            MappingProxyType({str(k): _sealed(v) for k, v in dict(self.extras).items()}),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def __len__(self) -> int:
        return int(np.asarray(self.psi_norm).size)

    def available(self) -> tuple[str, ...]:
        """Names of the profile fields this set actually carries."""
        return tuple(name for name in PROFILE_FIELDS if getattr(self, name) is not None)

    def field(self, name: str) -> np.ndarray:
        """One profile by name, from a field or from :attr:`extras`."""
        values = getattr(self, name, None) if name in KINETIC_UNITS else None
        if values is None:
            values = self.extras.get(name)
        if values is None:
            raise KeyError(
                f"no profile {name!r}; this set has {list(self.available())} "
                f"and extras {sorted(self.extras)}"
            )
        return np.asarray(values)

    def unit(self, name: str) -> str:
        """The container's unit for ``name``; ``""`` for an extra."""
        return KINETIC_UNITS.get(name, "")


def normalize_psi(
    profiles: KineticProfiles,
    *,
    method: str,
    axis: float | None = None,
    edge: float | None = None,
) -> KineticProfiles:
    """Rescale the radial coordinate, explicitly, recording that it happened.

    ``"min_max"`` maps the coordinate's own range onto [0, 1] -- what the
    legacy readers did silently on every read.  ``"explicit"`` divides by
    ``edge`` after subtracting ``axis``, for a caller that knows the axis and
    edge values from elsewhere.

    ``method`` has no default on purpose.  A caller who passes ``axis`` and
    ``edge`` has said which rescale they mean, and a default of ``"min_max"``
    would ignore both, stretch the coordinate instead, and stamp the result
    with a :class:`PsiNormalization` that certifies the operation they did
    not ask for -- which is the silent rescale this module exists to prevent,
    only now with a provenance record vouching for it.
    """
    psi = np.asarray(profiles.psi_norm, dtype=float)
    if method == "min_max":
        if axis is not None or edge is not None:
            raise ValueError(
                "method='min_max' takes the axis and edge from the coordinate's own "
                "range; pass method='explicit' to use the axis= and edge= given"
            )
        axis_value, edge_value = float(psi.min()), float(psi.max())
    elif method == "explicit":
        if edge is None:
            raise ValueError("method='explicit' needs edge=")
        axis_value, edge_value = float(axis or 0.0), float(edge)
    else:
        raise ValueError(f"method must be 'min_max' or 'explicit', got {method!r}")
    span = edge_value - axis_value
    if span == 0:
        raise ValueError("the radial coordinate has zero range; nothing to normalize")
    return replace(
        profiles,
        psi_norm=(psi - axis_value) / span,
        normalization=PsiNormalization(
            method=method,
            source=profiles.normalization.source,
            axis_value=axis_value,
            edge_value=edge_value,
        ),
    )


def _starts_with_number(line: str) -> bool:
    """GPEC's own test: after leading blanks and up to two signs, a digit.

    Deliberately not ``float(token)``: ``readtable``
    (``pentrc/utilities.f90:436-444``) looks at one character, so ``nan`` and
    ``Infinity`` are header text to GPEC where Python would read them as
    numbers, and ``.5`` is header text too.
    """
    text = line.lstrip()
    index = 0
    for _ in range(2):  # readtable passes over a plus and a minus
        if index < len(text) and text[index] in "+-":
            index += 1
    return index < len(text) and text[index] in "0123456789"


def _data_lines(lines: Sequence[str]) -> list[str]:
    """The first contiguous block of numeric lines, as GPEC's reader takes it.

    ``readtable`` (``pentrc/utilities.f90:428-451``) sets ``startline`` to the
    first line beginning with a number and ``endline`` to the line before the
    *next* line that does not -- and then ignores the rest of the file.  So a
    footer stops the table rather than being skipped over, and a numeric line
    after one is not data.  Collecting every numeric line anywhere instead
    would read a summary row appended below a footer as a fourth data point
    where GPEC reads three, with no error on either side.
    """
    data: list[str] = []
    started = False
    for line in lines:
        if _starts_with_number(line):
            started = True
            data.append(line.strip())
        elif started:
            break
    return data


def _float(token: str) -> float:
    """A number as Fortran writes it, ``1.0D+00`` included.

    GPEC's list-directed ``read`` accepts a ``D`` exponent, so a ``.kin``
    written by a Fortran tool carries them; Python's ``float`` does not, and
    dropping such a row would lose data from a file GPEC reads without
    complaint.
    """
    try:
        return float(token)
    except ValueError:
        return float(token.replace("D", "E").replace("d", "e"))


def read_kin(path: str | Path) -> KineticProfiles:
    """Read a GPEC ``.kin`` file.

    The radial coordinate is taken exactly as written; nothing is rescaled.
    """
    path = Path(path).expanduser()
    # Universal newlines, so a CRLF file reads correctly; write_kin emits LF.
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = _data_lines(lines)
    if not rows:
        raise ValueError(f"{path.name} holds no data rows (no line starts with a number)")

    split = [row.split() for row in rows]
    width = len(split[0])
    # GPEC allocates its table from the first data line's token count and
    # reads the rest into it, so a ragged row is a misparse there and an
    # inhomogeneous-shape error from numpy here -- neither of which names the
    # line.  Say which one it is.
    for offset, tokens in enumerate(split):
        if len(tokens) != width:
            raise ValueError(
                f"{path.name} data row {offset + 1} has {len(tokens)} columns against "
                f"{width} on the first row: {rows[offset]!r}"
            )
    if width < len(KIN_COLUMNS):
        raise ValueError(
            f"{path.name} has {width} columns; a .kin file has "
            f"{len(KIN_COLUMNS)}: {', '.join(KIN_COLUMNS)}"
        )
    table = np.array([[_float(token) for token in tokens] for tokens in split], dtype=float)

    columns = {name: table[:, index] for index, name in enumerate(KIN_COLUMNS)}
    extras = {
        f"column_{index + 1}": table[:, index]
        for index in range(len(KIN_COLUMNS), width)
    }
    # Columns are numbered as a person reading the file would: the first is 1.
    provenance = {
        name: f"{path.name} column {index + 1}" for index, name in enumerate(KIN_COLUMNS)
    }
    provenance.update(
        {name: f"{path.name} column {index + 1}" for index, name in enumerate(extras, len(KIN_COLUMNS))}
    )
    return KineticProfiles(
        **columns,
        normalization=PsiNormalization(method="as_read", source=path.name),
        extras=extras,
        provenance=provenance,
        source=str(path),
    )


def write_kin(
    profiles: KineticProfiles,
    path: str | Path,
    *,
    header: bool = True,
    allow_zero_rotation: bool = False,
) -> Path:
    """Write a GPEC ``.kin`` file.

    Raises :class:`ValueError` rather than write a file GPEC would read as
    something other than what it says, in four cases:

    - only ``omega_tor`` is present.  The ``.kin`` rotation column is
      omega_E, and a toroidal rotation written there is wrong in a way
      nothing downstream can detect.
    - ``omega_exb`` holds a zero.  GPEC substitutes 1e-9 for *each* zero
      element to keep its spline finite (``pentrc/inputs.f90:255-265``), so
      those points do not mean what the file says; ``allow_zero_rotation``
      writes them anyway, which is what a set converted from an all-zero
      ``PROFROT.IN`` needs.
    - any value is not finite.  GPEC's own warning at that substitution says
      a NaN "ruins the whole spline", and nothing downstream recovers.
    - the radial coordinate does not increase.  ``spline_fit``
      (``inputs.f90:219-222``) assumes an increasing abscissa and silently
      produces a garbage spline otherwise.

    Line endings are LF, so a file read from CRLF input is not written back
    byte-identically.
    """
    path = Path(path).expanduser()
    missing = [name for name in KIN_COLUMNS if getattr(profiles, name, None) is None]
    if "omega_exb" in missing and profiles.omega_tor is not None:
        raise ValueError(
            "this profile set has omega_tor but no omega_exb, and a .kin file's "
            "rotation column is omega_E; convert explicitly rather than writing "
            "toroidal rotation into it"
        )
    if missing:
        raise ValueError(f"a .kin file needs {list(KIN_COLUMNS)}; missing {missing}")

    table = np.column_stack(
        [np.asarray(getattr(profiles, name), dtype=float) for name in KIN_COLUMNS]
    )
    if not np.all(np.isfinite(table)):
        bad = {
            name: int(np.count_nonzero(~np.isfinite(table[:, index])))
            for index, name in enumerate(KIN_COLUMNS)
            if not np.all(np.isfinite(table[:, index]))
        }
        raise ValueError(
            f"{bad} hold values that are not finite; GPEC splines the file it reads "
            "and a single NaN ruins the whole spline"
        )

    psi = table[:, 0]
    if not np.all(np.diff(psi) > 0):
        first = int(np.argmin(np.diff(psi) > 0))
        raise ValueError(
            f"the radial coordinate does not increase (row {first + 1} is "
            f"{psi[first]:.8e}, row {first + 2} is {psi[first + 1]:.8e}); GPEC splines "
            "against it and assumes an increasing abscissa"
        )

    zeros = int(np.count_nonzero(table[:, KIN_COLUMNS.index("omega_exb")] == 0.0))
    if zeros and not allow_zero_rotation:
        raise ValueError(
            f"omega_exb is zero at {zeros} of {len(psi)} points; GPEC replaces every "
            "zero with 1e-9 to keep its spline finite, so those points would not mean "
            "what the file says. Pass allow_zero_rotation=True to write it anyway"
        )

    lines = [KIN_HEADER] if header else []
    lines.extend("  " + "   ".join(f"{value:.8e}" for value in row) for row in table)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
