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
pfile and MARS readers convert on the way in, through the single table in
:mod:`vaft.data.kinetic_units`.

Two things this container is careful about, because the code it replaces was
not:

- **The radial coordinate is never rescaled on read.**  A reader records what
  the file said and how it was normalised (:class:`PsiNormalization`); making
  it span exactly [0, 1] is :func:`normalize_psi`, an explicit operation.
  This matters physically: GPEC's ``read_kin`` re-splines a ``.kin`` onto a
  uniform 100-point [0, 1] grid *with extrapolation*, so it expects a
  truncated edge and handles it.  Stretching the axis instead moves every
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
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "KIN_COLUMNS",
    "KIN_HEADER",
    "KineticProfiles",
    "PsiNormalization",
    "Species",
    "UNITS",
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
#: into them and a writer converts out of them.
UNITS: Mapping[str, str] = {
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
PROFILE_FIELDS: tuple[str, ...] = tuple(name for name in UNITS if name != "psi_norm")


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
    """One ion species from a pfile's ``N Z A`` block."""

    label: str
    n: float
    z: float
    a: float


@dataclass(frozen=True)
class KineticProfiles:
    """Kinetic profiles on one radial coordinate, in the container's units.

    Every profile field is optional because the formats differ in what they
    carry; :meth:`available` says which are present.  Anything a file holds
    that has no field of its own is kept in :attr:`extras` under its own
    name, so nothing is dropped.
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
        size = np.asarray(self.psi_norm).size
        for name in PROFILE_FIELDS:
            values = getattr(self, name)
            if values is not None and np.asarray(values).size != size:
                raise ValueError(
                    f"{name} has {np.asarray(values).size} samples but psi_norm has "
                    f"{size}; every profile is on the same radial coordinate"
                )

    def __len__(self) -> int:
        return int(np.asarray(self.psi_norm).size)

    def available(self) -> tuple[str, ...]:
        """Names of the profile fields this set actually carries."""
        return tuple(name for name in PROFILE_FIELDS if getattr(self, name) is not None)

    def field(self, name: str) -> np.ndarray:
        """One profile by name, from a field or from :attr:`extras`."""
        values = getattr(self, name, None) if name in UNITS else None
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
        return UNITS.get(name, "")


def normalize_psi(
    profiles: KineticProfiles,
    *,
    method: str = "min_max",
    axis: float | None = None,
    edge: float | None = None,
) -> KineticProfiles:
    """Rescale the radial coordinate, explicitly, recording that it happened.

    ``"min_max"`` maps the coordinate's own range onto [0, 1] -- what the
    legacy readers did silently on every read.  ``"explicit"`` divides by
    ``edge`` after subtracting ``axis``, for a caller that knows the axis and
    edge values from elsewhere.
    """
    psi = np.asarray(profiles.psi_norm, dtype=float)
    if method == "min_max":
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


def _data_lines(lines: Sequence[str]) -> list[str]:
    """Lines GPEC would read as data: those whose first token is a number.

    ``read_kin`` accepts "(nearly) arbitrary header and/or footer, with the
    exception that no lines start with a number", so this is the file's
    actual contract -- not a fixed header count.
    """
    data = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            float(stripped.split()[0])
        except ValueError:
            continue
        data.append(stripped)
    return data


def read_kin(path: str | Path) -> KineticProfiles:
    """Read a GPEC ``.kin`` file.

    The radial coordinate is taken exactly as written; nothing is rescaled.
    """
    path = Path(path).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = _data_lines(lines)
    if not rows:
        raise ValueError(f"{path.name} holds no data rows (no line starts with a number)")

    table = np.array([[float(token) for token in row.split()] for row in rows], dtype=float)
    if table.shape[1] < len(KIN_COLUMNS):
        raise ValueError(
            f"{path.name} has {table.shape[1]} columns; a .kin file has "
            f"{len(KIN_COLUMNS)}: {', '.join(KIN_COLUMNS)}"
        )
    columns = {name: table[:, index] for index, name in enumerate(KIN_COLUMNS)}
    extras = {
        f"column_{index}": table[:, index]
        for index in range(len(KIN_COLUMNS), table.shape[1])
    }
    return KineticProfiles(
        **columns,
        normalization=PsiNormalization(method="as_read", source=path.name),
        extras=extras,
        provenance={name: f"{path.name} column {index}" for index, name in enumerate(KIN_COLUMNS)},
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

    Refuses a profile set carrying only ``omega_tor``: the ``.kin`` rotation
    column is omega_E, and a toroidal rotation written there is wrong in a
    way nothing downstream can detect.  Refuses an all-zero ``omega_exb`` too
    unless asked, because GPEC silently substitutes 1e-9 for every zero to
    keep its spline from going NaN -- so the file does not mean what it says.
    """
    path = Path(path).expanduser()
    missing = [name for name in KIN_COLUMNS if getattr(profiles, name, None) is None]
    if missing == ["omega_exb"] and profiles.omega_tor is not None:
        raise ValueError(
            "this profile set has omega_tor but no omega_exb, and a .kin file's "
            "rotation column is omega_E; convert explicitly rather than writing "
            "toroidal rotation into it"
        )
    if missing:
        raise ValueError(f"a .kin file needs {list(KIN_COLUMNS)}; missing {missing}")

    omega = np.asarray(profiles.omega_exb, dtype=float)
    if not allow_zero_rotation and np.all(omega == 0.0):
        raise ValueError(
            "omega_exb is identically zero; GPEC replaces every zero with 1e-9 to "
            "keep its spline finite, so the file would not mean what it says. Pass "
            "allow_zero_rotation=True to write it anyway"
        )

    table = np.column_stack([np.asarray(getattr(profiles, name), dtype=float) for name in KIN_COLUMNS])
    lines = [KIN_HEADER] if header else []
    lines.extend("  " + "   ".join(f"{value:.8e}" for value in row) for row in table)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
