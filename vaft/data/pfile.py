"""The Osborne pfile: a native transcript, and its conversion.

A pfile is the profile format the Osborne fitting tools and OMFIT write and
that kinetic EFIT and the GPEC workflows consume.  It is a sequence of
blocks -- an ``N Z A of ION SPECIES`` table, then one block per profile --
each block a header naming the quantity and its unit, then rows of
``psinorm``, the value, and *the quantity's derivative with respect to
psinorm, as the writer computed it*:

.. code-block:: text

    3 N Z A of ION SPECIES
     6.000000   6.000000   12.010700
     1.000000   1.000000   2.000000
     1.000000   1.000000   2.000000
    201 psinorm ne(10^20/m^3) dne/dpsiN
     0.00000000e+00   1.45835195e-01   -1.13443100e-02

Two layers live here, for the reason :mod:`vaft.code.transp` keeps the same
split: :class:`PFile` is the file as written, in the file's own units and
section order, converting nothing, and
:func:`kinetic_profiles_from_pfile` is the conversion into
:class:`~vaft.data.kinetic_profiles.KineticProfiles`.  Keeping them apart is
what lets "this is byte-for-byte the file we read" and "these are the right
units" fail independently of each other.

Three properties of the format decide how this module is written, and each
was measured against the 57 shipped ``p045453.*`` files rather than assumed:

- **The derivative column cannot be recomputed.**  Preserving it, a file
  round-trips byte-identically 57 times out of 57; recomputing it with
  ``np.gradient(..., edge_order=2)`` -- which is what the writer this
  replaces used -- round-trips 0 times out of 57, and only 4 of the 22
  sections agree even to one part in a million.  So the derivative is data,
  read and written like any other column, and :func:`write_pfile` only
  computes one for a section that has none.
- **Every section carries the same psinorm column**, bit for bit, in every
  one of those files.  It is therefore checked, rather than taken from the
  first section and assumed of the rest.
- **A unit can be empty.**  ``omghb`` really does declare ``omghb()``, so an
  absent unit is ``""`` and not a missing one.

Nothing rescales the radial coordinate.  As in :func:`.read_kin`, the
coordinate is recorded as ``"as_read"`` and
:func:`~vaft.data.kinetic_profiles.normalize_psi` is the separate, named
operation that changes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .kinetic_profiles import (
    KINETIC_UNITS,
    KineticProfiles,
    PsiNormalization,
    Species,
    _float,
    _sealed,
)

__all__ = [
    "PFILE_LABELS",
    "PFILE_SECTION_ORDER",
    "PFILE_TO_CONTAINER",
    "PFILE_UNITS",
    "PFILE_UNIT_LADDER",
    "PFile",
    "PFileFormatError",
    "PFileSection",
    "kinetic_profiles_from_pfile",
    "read_pfile",
    "write_pfile",
]

#: The section order the format is written in, following OMFIT's
#: ``OMFITpFile``.  All 57 reference files carry exactly these 22 in exactly
#: this order; ``pplas`` is an optional non-standard extra that belongs after
#: ``ptot`` when a writer includes it.
PFILE_SECTION_ORDER: tuple[str, ...] = (
    "ne", "ni", "nz1", "nb",
    "te", "ti",
    "ptot", "pb",
    "omeg", "omegp", "omgvb", "omgpp", "omgeb",
    "er",
    "ommvb", "ommpp", "omevb", "omepp",
    "kpol", "omghb",
    "vtor1", "vpol1",
)

#: The unit each section declares.  Read from the file rather than taken from
#: here -- this is the default for a section built in memory, and what a
#: reader checks a surprising file against.  Every key's unit is constant
#: across the reference files.
PFILE_UNITS: Mapping[str, str] = {
    "ne": "10^20/m^3", "ni": "10^20/m^3", "nz1": "10^20/m^3", "nb": "10^20/m^3",
    "te": "KeV", "ti": "KeV",
    "ptot": "KPa", "pplas": "KPa", "pb": "KPa",
    "omeg": "kRad/s", "omegp": "kRad/s", "omgvb": "kRad/s", "omgpp": "kRad/s",
    "omgeb": "kRad/s", "ommvb": "kRad/s", "ommpp": "kRad/s", "omevb": "kRad/s",
    "omepp": "kRad/s",
    "er": "kV/m",
    "kpol": "km/s/T",
    "omghb": "",
    "vtor1": "km/s", "vpol1": "km/s",
}

#: What each section is.  Worth stating in full because the rotation family
#: is ten sections whose names differ by two letters and whose meanings do
#: not: reading one for another is the defect this module exists to end.
PFILE_LABELS: Mapping[str, str] = {
    "ne": "electron density",
    "ni": "main ion density",
    "nz1": "impurity density",
    "nb": "fast ion density",
    "te": "electron temperature",
    "ti": "ion temperature",
    "ptot": "total pressure, fast ions included",
    "pplas": "thermal plasma pressure",
    "pb": "fast ion pressure",
    "omeg": "toroidal angular velocity",
    "omegp": "poloidal rotation contribution",
    "omgvb": "impurity VxB (perpendicular) rotation",
    "omgpp": "impurity diamagnetic rotation",
    "omgeb": "ExB rotation, -dPhi/dPsi",
    "er": "radial electric field",
    "ommvb": "main ion VxB (perpendicular) rotation",
    "ommpp": "main ion diamagnetic rotation",
    "omevb": "electron VxB (perpendicular) rotation",
    "omepp": "electron diamagnetic rotation",
    "kpol": "impurity parallel stream function",
    "omghb": "Hahm-Burrell ExB shearing rate",
    "vtor1": "toroidal velocity of the first impurity species",
    "vpol1": "poloidal velocity of the first impurity species",
}

#: Which sections have a field of their own in
#: :class:`~vaft.data.kinetic_profiles.KineticProfiles`.
#:
#: **This table is the fix for the conflation the legacy reader had.**  That
#: reader chose a section by asking whether ``"omeg"`` was *contained in* the
#: header, which matches ``omeg`` and ``omegp`` alike, and assigned both to
#: one rotation key with the later section overwriting the earlier -- so what
#: survived was ``omegp``, the poloidal contribution, under a name meaning
#: E x B, while the actual E x B section ``omgeb`` matched nothing and was
#: dropped.  Three different quantities, one of them silently standing in for
#: another.  Matching is therefore by exact key, and the three land in three
#: different fields.
#:
#: Everything absent from this table keeps its own pfile name in
#: :attr:`KineticProfiles.extras`, in the file's units.  ``T_z`` has no pfile
#: source and stays absent.
PFILE_TO_CONTAINER: Mapping[str, str] = {
    "ne": "n_e",
    "ni": "n_i",
    "nz1": "n_z",
    "nb": "n_fast",
    "te": "T_e",
    "ti": "T_i",
    "ptot": "p_total",
    "pb": "p_fast",
    "omeg": "omega_tor",
    "omgeb": "omega_exb",
    "omegp": "omega_pol",
    "er": "e_radial",
}

#: pfile unit -> (the container's unit, the factor between them).  The one
#: ladder; :data:`~vaft.data.kinetic_profiles.KINETIC_UNITS` fixes the far
#: end of it.  Keys are matched case-insensitively because the format writes
#: ``KeV`` and ``kRad/s`` but nothing in it compels that spelling.
PFILE_UNIT_LADDER: Mapping[str, tuple[str, float]] = {
    "10^20/m^3": ("m^-3", 1.0e20),
    "kev": ("eV", 1.0e3),
    "kpa": ("Pa", 1.0e3),
    "krad/s": ("rad/s", 1.0e3),
    "kv/m": ("V/m", 1.0e3),
}

_SPECIES_HEADER = "N Z A of ION SPECIES"

#: The row of the species block a pfile writes first, then second, then
#: third.  The block itself carries no labels, only numbers, so the position
#: is the only thing that says which species a row is.
SPECIES_ORDER: tuple[str, ...] = ("impurity", "main ion", "fast ion")


class PFileFormatError(ValueError):
    """The file is not shaped like an Osborne pfile."""


def _psinorm_derivative(psi_norm: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``d(values)/d(psi_norm)``, for a section that carries no derivative.

    Second-order one-sided at the ends, which is what the writer this module
    replaces used.  It is **not** how the shipped files' own derivative
    column was produced -- recomputing that column reproduces none of the 57
    reference files, and only four of their 22 sections agree with it to one
    part in a million -- so this is the fallback for a section built in
    memory, never a substitute for a derivative that was read.

    :mod:`vaft.process.numerical`'s ``time_derivative`` is deliberately not
    reused: it is a different stencil, and its own docstring pins its results
    so they do not move.
    """
    return np.gradient(np.asarray(values, dtype=float), np.asarray(psi_norm, dtype=float), edge_order=2)


@dataclass(frozen=True, eq=False)
class PFileSection:
    """One block of a pfile, in the file's own units.

    ``derivative`` is the file's third column, kept as written.  It is
    ``None`` only for a section built in memory.
    """

    key: str
    unit: str
    psi_norm: np.ndarray
    values: np.ndarray
    derivative: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "psi_norm", _sealed(self.psi_norm))
        object.__setattr__(self, "values", _sealed(self.values))
        if self.values.size != self.psi_norm.size:
            raise PFileFormatError(
                f"{self.key} has {self.values.size} values against "
                f"{self.psi_norm.size} coordinate points"
            )
        if self.derivative is not None:
            derivative = _sealed(self.derivative)
            if derivative.size != self.psi_norm.size:
                raise PFileFormatError(
                    f"{self.key} has {derivative.size} derivative points against "
                    f"{self.psi_norm.size} coordinate points"
                )
            object.__setattr__(self, "derivative", derivative)

    @property
    def label(self) -> str:
        """What the quantity is, or ``""`` for a section not catalogued here."""
        return PFILE_LABELS.get(self.key, "")


@dataclass(frozen=True, eq=False)
class PFile:
    """One Osborne pfile as it was written: sections in order, units as declared.

    Nothing here is converted.  ``ne`` is still in ``10^20/m^3`` and ``te``
    still in keV; :func:`kinetic_profiles_from_pfile` is what changes that.
    """

    psi_norm: np.ndarray
    sections: tuple[PFileSection, ...] = ()
    species: tuple[Species, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "psi_norm", _sealed(self.psi_norm))
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "species", tuple(self.species))
        seen: set[str] = set()
        for section in self.sections:
            if section.key in seen:
                raise PFileFormatError(
                    f"{section.key} appears twice; a pfile carries one block per "
                    "quantity, and silently keeping the last is how a section "
                    "comes to stand in for another"
                )
            seen.add(section.key)
            if section.psi_norm.size != self.psi_norm.size:
                raise PFileFormatError(
                    f"{section.key} is on {section.psi_norm.size} points against the "
                    f"file's {self.psi_norm.size}"
                )

    def __len__(self) -> int:
        return int(self.psi_norm.size)

    def keys(self) -> tuple[str, ...]:
        """The section names, in file order."""
        return tuple(section.key for section in self.sections)

    def section(self, key: str) -> PFileSection:
        """One section by its pfile name."""
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(f"no section {key!r}; this file has {list(self.keys())}")

    def unit(self, key: str) -> str:
        """The unit the file declares for ``key``."""
        return self.section(key).unit


# --- reading -----------------------------------------------------------------


def _parse_block_header(line: str) -> tuple[int, str, str] | None:
    """``(count, key, unit)`` for a profile header, or ``None`` if not one.

    A header is ``<count> psinorm <key>(<unit>) d<key>/dpsiN``.  The unit may
    be empty -- ``omghb()`` is written that way -- so an absent unit is
    ``""`` rather than a missing one.
    """
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        count = int(parts[0])
    except ValueError:
        return None
    quantity = parts[2]
    if "(" in quantity and quantity.endswith(")"):
        key, _, rest = quantity.partition("(")
        unit = rest[:-1]
    else:
        key, unit = quantity, ""
    if not key:
        return None
    return count, key, unit


def _rows(lines: Sequence[str], start: int, count: int, what: str, width: int) -> np.ndarray:
    """``count`` rows of ``width`` numbers, refusing anything else.

    The readers this replaces skipped a short or ragged row and carried on,
    so a truncated file became a shorter profile with no diagnostic.
    """
    if start + count > len(lines):
        raise PFileFormatError(
            f"{what} declares {count} rows but the file ends after "
            f"{len(lines) - start}"
        )
    table = np.empty((count, width), dtype=float)
    for offset in range(count):
        tokens = lines[start + offset].split()
        if len(tokens) != width:
            raise PFileFormatError(
                f"{what} row {offset + 1} has {len(tokens)} columns against the "
                f"{width} a {what.split()[0]} row carries: {lines[start + offset]!r}"
            )
        table[offset] = [_float(token) for token in tokens]
    return table


def read_pfile(path: str | Path) -> PFile:
    """Read an Osborne pfile.

    The radial coordinate, the section order, the declared units and the
    derivative column are all taken exactly as written, and a section this
    module does not catalogue is kept under its own name rather than
    dropped.  :func:`write_pfile` writes such a file back byte for byte.
    """
    path = Path(path).expanduser()
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()

    sections: list[PFileSection] = []
    species: list[Species] = []
    psi_norm: np.ndarray | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        if _SPECIES_HEADER in line:
            count = int(line.split()[0])
            table = _rows(lines, index + 1, count, "species block", 3)
            species = [
                Species(
                    label=SPECIES_ORDER[row] if row < len(SPECIES_ORDER) else f"species {row + 1}",
                    n=float(values[0]),
                    z=float(values[1]),
                    a=float(values[2]),
                )
                for row, values in enumerate(table)
            ]
            index += 1 + count
            continue

        header = _parse_block_header(line)
        if header is None:
            index += 1
            continue
        count, key, unit = header
        table = _rows(lines, index + 1, count, f"section {key}", 3)
        if psi_norm is None:
            psi_norm = table[:, 0]
        elif not np.array_equal(table[:, 0], psi_norm):
            # Every section of every reference file carries the same column,
            # so a file where they differ is telling us something, and taking
            # the first one silently would put each profile on a coordinate
            # that is not its own.
            raise PFileFormatError(
                f"section {key} is on a different psinorm column from the first "
                "section of the file; a pfile's blocks share one radial coordinate"
            )
        sections.append(
            PFileSection(
                key=key,
                unit=unit,
                psi_norm=table[:, 0],
                values=table[:, 1],
                derivative=table[:, 2],
            )
        )
        index += 1 + count

    if psi_norm is None:
        raise PFileFormatError(f"{path.name} holds no profile sections")
    return PFile(
        psi_norm=psi_norm,
        sections=tuple(sections),
        species=tuple(species),
        source=str(path),
    )


# --- writing -----------------------------------------------------------------


def _write_order(keys: Sequence[str]) -> list[str]:
    """Catalogued sections in the format's order, then the rest as they came."""
    known = [key for key in PFILE_SECTION_ORDER if key in keys]
    if "pplas" in keys:
        known.insert(known.index("ptot") + 1 if "ptot" in known else 0, "pplas")
    return known + [key for key in keys if key not in known]


def write_pfile(
    pfile: PFile,
    path: str | Path,
    *,
    recompute_derivatives: bool = False,
) -> Path:
    """Write an Osborne pfile.

    The derivative column is the one that was read.  That is not fussiness:
    the shipped reference files round-trip byte-identically 57 times out of
    57 this way and 0 times out of 57 when the column is recomputed, because
    whatever produced them did not use the stencil the previous writer did.
    ``recompute_derivatives=True`` recomputes every section anyway, for a
    caller who has changed the values; a section with no derivative of its
    own is computed either way.

    Refuses a value that is not finite and a coordinate that does not
    increase, as :func:`~vaft.data.kinetic_profiles.write_kin` does, since
    both make a file that consumers spline into nonsense rather than reject.
    """
    path = Path(path).expanduser()
    psi = np.asarray(pfile.psi_norm, dtype=float)
    if not np.all(np.diff(psi) > 0):
        first = int(np.argmin(np.diff(psi) > 0))
        raise PFileFormatError(
            f"psinorm does not increase (row {first + 1} is {psi[first]:.8e}, row "
            f"{first + 2} is {psi[first + 1]:.8e}); consumers spline against it"
        )

    ordered = _write_order(pfile.keys())
    lines: list[str] = []
    if pfile.species:
        lines.append(f"{len(pfile.species)} {_SPECIES_HEADER}")
        lines.extend(
            f" {one.n:.6f}   {one.z:.6f}   {one.a:.6f}" for one in pfile.species
        )

    for key in ordered:
        section = pfile.section(key)
        values = np.asarray(section.values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise PFileFormatError(
                f"section {key} holds {int(np.count_nonzero(~np.isfinite(values)))} "
                "values that are not finite"
            )
        if recompute_derivatives or section.derivative is None:
            derivative = _psinorm_derivative(psi, values)
        else:
            derivative = np.asarray(section.derivative, dtype=float)
        lines.append(f"{values.size} psinorm {key}({section.unit}) d{key}/dpsiN")
        lines.extend(
            f" {a:.8e}   {b:.8e}   {c:.8e}" for a, b, c in zip(psi, values, derivative)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- conversion --------------------------------------------------------------


def _converted(section: PFileSection, target: str) -> np.ndarray:
    """A mapped section in the container's units, refusing an unknown one."""
    unit, factor = PFILE_UNIT_LADDER.get(section.unit.lower(), (None, None))
    if unit is None:
        raise PFileFormatError(
            f"section {section.key} declares the unit {section.unit!r}, which this "
            f"module cannot convert to {KINETIC_UNITS[target]} for {target}. Guessing "
            "a factor here is how a density ends up wrong by a million"
        )
    if unit != KINETIC_UNITS[target]:
        raise PFileFormatError(
            f"section {section.key} is in {section.unit!r}, which converts to {unit}, "
            f"but {target} is carried in {KINETIC_UNITS[target]}"
        )
    return np.asarray(section.values, dtype=float) * factor


def kinetic_profiles_from_pfile(pfile: PFile) -> KineticProfiles:
    """Convert a pfile into the kinetic-profile container.

    The catalogued sections in :data:`PFILE_TO_CONTAINER` become named
    fields, converted through :data:`PFILE_UNIT_LADDER`; the rest -- the
    diamagnetic and perpendicular rotation family, the shearing rate, the
    velocities -- keep their pfile names in :attr:`KineticProfiles.extras`
    in the file's own units, with those units recorded in provenance.  The
    derivative columns do not survive the conversion: the container holds
    profiles, and :class:`PFile` is where the file's own derivatives live.

    The radial coordinate is not touched, so
    ``normalization.method`` is ``"as_read"``.
    """
    source = Path(pfile.source).name if pfile.source else ""
    fields: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}

    for section in pfile.sections:
        target = PFILE_TO_CONTAINER.get(section.key)
        if target is None:
            extras[section.key] = np.asarray(section.values, dtype=float)
            provenance[section.key] = (
                f"{source} section {section.key} "
                f"[{section.unit or 'no unit declared'}], not converted"
            )
            continue
        fields[target] = _converted(section, target)
        provenance[target] = (
            f"{source} section {section.key} [{section.unit}] -> {KINETIC_UNITS[target]}"
        )

    if pfile.species:
        provenance["species"] = f"{source} {_SPECIES_HEADER}"

    return KineticProfiles(
        psi_norm=np.asarray(pfile.psi_norm, dtype=float),
        **fields,
        normalization=PsiNormalization(method="as_read", source=source),
        species=pfile.species,
        extras=extras,
        provenance=provenance,
        source=pfile.source,
    )
