"""MARS profile inputs: one quantity per file, in SI, on MARS's own coordinate.

MARS reads its kinetic profiles as a small deck of two-column ASCII files --
``PROFDEN.IN``, ``PROFTE.IN``, ``PROFTI.IN``, ``PROFROT.IN``, ``PROFWE.IN``.
Each is a row count and a radial-variable key, then that many rows of an
abscissa and a value:

.. code-block:: text

    200 1
    0.000000000000000000e+00 4.233437695718514688e+19
    5.025125628140703592e-03 4.233433978258264064e+19

Two things about that deck are easy to get wrong, and the code this replaces
got both.

**The second header field is not decoration: it names the abscissa.**  MARS
branches on it in every one of its dozen profile readers, and rejects any
other value outright::

    IF (IRAD.EQ.1) THEN
       SR  = CS(1:NRP1)          ! s = sqrt(normalised poloidal flux)
    ELSEIF (IRAD.EQ.2) THEN
       RPRS = RPRS/RPRS(NRAD)
       SR  = RAD                 ! marsq.f:16269 -- "RAD = SQRT(TOROIDAL FLUX)"
    ELSE
       STOP 'NPROFR=4'
    ENDIF

So the first column of a ``key = 1`` file is **s**, the square root of the
normalised poloidal flux -- not the normalised flux itself.  Reading it
straight into a field that means psi_N puts every point at the square root
of where it belongs: a point at 0.5 in the file is psi_N = 0.25.  This
module converts, and records the conversion in
:class:`~vaft.data.kinetic_profiles.PsiNormalization`; a ``key = 2`` deck is
on a toroidal-flux coordinate that cannot reach psi_N without an
equilibrium, so it is refused rather than guessed at.

**The two rotation files are two different quantities.**  MARS's source
settles which is which:

- ``PROFROT.IN`` is the bulk toroidal **fluid** rotation, omega_phi, read
  under ``NPROFR = 4`` into ``ROT``/``ROTM`` with amplitude ``ROTE``; the
  usage manual calls it "experimental fluid rotation profile".
- ``PROFWE.IN`` is the toroidal **E x B** rotation, omega_E, read under
  ``NPROFWE = 4`` into ``ROTWEI``/``ROTWEM``; the source block is headed
  ``TOROIDAL EXB ROTATION FREQUENCY``.
- They differ by the ion diamagnetic frequency.  MARS states the relation in
  the comment on its own analytic branch: ``ROTWE = ROT - OMEGAI*``.

So the two map onto the two fields
:class:`~vaft.data.kinetic_profiles.KineticProfiles` already has, and there
is no precedence rule anywhere in this module: a deck carrying both is a
deck carrying two quantities.

**Values are not scaled**, and that is a finding rather than an assumption,
because "a MARS input is Alfven-normalised" is the obvious wrong guess.  Two
things rule it out.  The committed profiles run 1.4e4 to 1.0e5 rad/s against
an Alfven frequency of order 1.4e6 for MAST, so normalised they would be
0.01 to 0.07 -- which is what MARS's *namelist* amplitudes hold, not what its
profile files hold.  And MARS converts the file itself when told the file
carries absolute values: ``IF (NEXPV.EQ.1) ROTE = ZTEMP*ZTAUA0``, where
``ZTEMP`` is the on-axis file value and ``ZTAUA0`` the Alfven time, which is
dimensionally sensible only if the file is in rad/s.  MARS computes the
Alfven time itself from the equilibrium file; no part of that belongs here.

One consequence worth knowing when reading a deck: under MARS's default
``NEXPV = 0`` only the *shape* of each profile is used, the absolute scale
being replaced by a namelist amplitude.  Whether a deck's numbers reach the
run at all is a property of its ``RUN.IN``, not of these files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .kinetic_profiles import KineticProfiles, PsiNormalization, _sealed

__all__ = [
    "MARS_FIELD_FILES",
    "MARS_PROFILE_FILES",
    "MARS_PROFILE_LABELS",
    "MARS_RADIAL_VARIABLES",
    "MarsProfile",
    "MarsProfileFormatError",
    "read_mars_profile",
    "read_mars_profiles",
    "write_mars_profile",
    "write_mars_profiles",
]

#: Which file holds which quantity, and therefore which container field it
#: fills.  ``PROFROT.IN`` and ``PROFWE.IN`` are **separate** entries because
#: MARS reads them into separate arrays under separate flags; merging them,
#: or preferring one over the other, is the defect this module ends.
#:
#: ``PROFDEN.IN`` fills ``n_e`` and nothing else.  MARS carries one density
#: (``NPROFN``), so equating the ion density with it is a modelling choice a
#: caller makes, not something a reader may invent.
MARS_PROFILE_FILES: Mapping[str, str] = {
    "PROFDEN.IN": "n_e",
    "PROFTE.IN": "T_e",
    "PROFTI.IN": "T_i",
    "PROFROT.IN": "omega_tor",
    "PROFWE.IN": "omega_exb",
}

#: The inverse, for writing.
MARS_FIELD_FILES: Mapping[str, str] = {
    field: name for name, field in MARS_PROFILE_FILES.items()
}

#: What MARS calls each file, for error messages and provenance.
MARS_PROFILE_LABELS: Mapping[str, str] = {
    "PROFDEN.IN": "plasma density (NPROFN)",
    "PROFTE.IN": "electron temperature",
    "PROFTI.IN": "ion temperature",
    "PROFROT.IN": "fluid rotation omega_phi (NPROFR)",
    "PROFWE.IN": "E x B rotation omega_E (NPROFWE)",
}

#: The header's second field, and what it says the abscissa is.  MARS stops
#: on anything else.
MARS_RADIAL_VARIABLES: Mapping[int, str] = {
    1: "s, the square root of the normalised poloidal flux",
    2: "the square root of the normalised toroidal flux",
}

#: The key this module can convert to the container's psi_N, and how.
POLOIDAL_KEY = 1


class MarsProfileFormatError(ValueError):
    """The file is not shaped like a MARS profile input."""


def _profile_stem(filename: str) -> str:
    """``PROFXX.IN`` -> ``XX``, the name an unrecognised profile is kept under.

    Falls back to the whole file name when stripping would leave nothing, so
    a file called ``PROF.IN`` does not become an entry with an empty key.
    """
    stem = filename[: -len(".IN")] if filename.upper().endswith(".IN") else filename
    trimmed = stem[len("PROF"):] if stem.upper().startswith("PROF") else stem
    return trimmed or filename


def _column(values, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise MarsProfileFormatError(
            f"{what} is {array.ndim}-dimensional; a MARS profile column is a 1-D array"
        )
    return array


@dataclass(frozen=True, eq=False)
class MarsProfile:
    """One ``PROF*.IN``: an abscissa and one quantity, as written.

    ``coordinate`` is the file's own first column and is deliberately *not*
    called ``psi_norm``: what it is depends on ``irad``, and for the ``1``
    every committed deck carries it is ``s``, the square root of the
    normalised poloidal flux.  :func:`read_mars_profiles` is what converts.
    """

    coordinate: np.ndarray
    values: np.ndarray
    irad: int = POLOIDAL_KEY
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate", _sealed(_column(self.coordinate, "coordinate")))
        object.__setattr__(self, "values", _sealed(_column(self.values, "values")))
        if self.values.size != self.coordinate.size:
            raise MarsProfileFormatError(
                f"{self.values.size} values against {self.coordinate.size} coordinate "
                "points"
            )
        if self.irad not in MARS_RADIAL_VARIABLES:
            raise MarsProfileFormatError(
                f"{self.irad!r} is not a MARS radial-variable key; it must be one of "
                f"{sorted(MARS_RADIAL_VARIABLES)} "
                f"({'; '.join(f'{k} is {v}' for k, v in MARS_RADIAL_VARIABLES.items())}), "
                "and MARS stops on anything else"
            )

    def __len__(self) -> int:
        return int(self.coordinate.size)

    @property
    def radial_variable(self) -> str:
        """What ``coordinate`` is, in words."""
        return MARS_RADIAL_VARIABLES[self.irad]

    def psi_norm(self) -> np.ndarray:
        """The coordinate as normalised poloidal flux.

        Only a ``irad = 1`` file can answer: its abscissa is ``s``, so
        ``psi_N`` is ``s**2``.  A toroidal-flux abscissa needs an equilibrium
        to become poloidal flux, which a file-format reader does not have.
        """
        if self.irad != POLOIDAL_KEY:
            raise MarsProfileFormatError(
                f"this profile is on {self.radial_variable}; converting it to "
                "normalised poloidal flux needs an equilibrium, which this layer does "
                "not have. Read the coordinate as it stands"
            )
        return np.asarray(self.coordinate, dtype=float) ** 2


def read_mars_profile(path: str | Path) -> MarsProfile:
    """Read one ``PROF*.IN``, in the file's own coordinate and units.

    The declared row count is checked rather than trusted, and a malformed
    row is named.  The readers this replaces skipped an unparseable row and
    carried on, so a damaged file quietly became a shorter profile on a
    coordinate that no longer lined up with the rest of its deck.
    """
    path = Path(path).expanduser()
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MarsProfileFormatError(
            f"{path.name} is not UTF-8 text ({exc}); a MARS profile is plain ASCII"
        ) from exc

    count, irad, rows = _split(path.name, text)
    table = np.empty((count, 2), dtype=float)
    for index, row in enumerate(rows):
        tokens = row.split()
        if len(tokens) != 2:
            raise MarsProfileFormatError(
                f"{path.name} row {index + 1} has {len(tokens)} columns against the 2 "
                f"a MARS profile carries: {row!r}"
            )
        try:
            table[index] = [float(tokens[0].replace("D", "E").replace("d", "e")),
                            float(tokens[1].replace("D", "E").replace("d", "e"))]
        except ValueError as exc:
            raise MarsProfileFormatError(
                f"{path.name} row {index + 1} is not two numbers: {row!r}"
            ) from exc

    return MarsProfile(
        coordinate=table[:, 0], values=table[:, 1], irad=irad, source=str(path)
    )


def _split(name: str, text: str) -> tuple[int, int, list[str]]:
    """``(count, irad, rows)``, refusing a header or a length that does not fit."""
    lines = text.splitlines()
    if not lines:
        raise MarsProfileFormatError(f"{name} is empty")

    header = lines[0].split()
    if len(header) != 2:
        raise MarsProfileFormatError(
            f"{name} starts with {lines[0].strip()!r}; a MARS profile begins with a "
            "row count and a radial-variable key, and nothing else. A header with a "
            "third field is a different MARS format (PROFROTC.IN carries a harmonic "
            "count), not this one"
        )
    try:
        count, irad = int(header[0]), int(header[1])
    except ValueError as exc:
        raise MarsProfileFormatError(
            f"{name} starts with {lines[0].strip()!r}, which is not a row count and a "
            "radial-variable key"
        ) from exc
    if count <= 0:
        raise MarsProfileFormatError(f"{name} declares {count} rows")

    rows = [line for line in lines[1:] if line.strip()]
    if len(rows) != count:
        raise MarsProfileFormatError(
            f"{name} declares {count} rows and carries {len(rows)}; a count that "
            "disagrees with the file is a truncated or appended-to deck, not a "
            "difference to read past"
        )
    return count, irad, rows


def write_mars_profile(profile: MarsProfile, path: str | Path) -> Path:
    """Write one ``PROF*.IN``, in the coordinate the profile already carries.

    ``%.18e``, which is what the decks carry and more than enough to round a
    double trip exactly.  Refuses a column that is not finite or a coordinate
    that does not increase: MARS splines these profiles against that
    coordinate, as GPEC and the pfile consumers do against theirs.
    """
    path = Path(path).expanduser()
    coordinate = np.asarray(profile.coordinate, dtype=float)
    values = np.asarray(profile.values, dtype=float)
    for column, what in ((coordinate, "coordinate"), (values, "value")):
        if not np.all(np.isfinite(column)):
            raise MarsProfileFormatError(
                f"{path.name} would carry "
                f"{int(np.count_nonzero(~np.isfinite(column)))} {what}s that are not "
                "finite; MARS splines what it reads"
            )
    if coordinate.size > 1 and not np.all(np.diff(coordinate) > 0):
        first = int(np.argmin(np.diff(coordinate) > 0))
        raise MarsProfileFormatError(
            f"{path.name}'s coordinate does not increase (row {first + 1} is "
            f"{coordinate[first]:.8e}, row {first + 2} is {coordinate[first + 1]:.8e}); "
            "MARS splines against it"
        )

    lines = [f"{values.size} {int(profile.irad)}"]
    lines.extend(f"{a:.18e} {b:.18e}" for a, b in zip(coordinate, values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _one_coordinate(found: Mapping[str, MarsProfile]) -> MarsProfile:
    """The deck's one abscissa, refusing a deck that carries more than one.

    The reader this replaces kept whichever file it read first and grafted
    every other file's values onto it element by element -- warning when the
    coordinates differed in value, and saying nothing at all when they
    differed in length.
    """
    # Which file is the reference only decides which name an error message
    # leads with; any disagreement is refused whichever way round it is found.
    reference_name, reference = next(iter(found.items()))
    for name, profile in found.items():
        if name == reference_name:
            continue
        if profile.irad != reference.irad:
            raise MarsProfileFormatError(
                f"{name} is on {profile.radial_variable} and {reference_name} on "
                f"{reference.radial_variable}; a deck whose files disagree about their "
                "own radial variable cannot be read onto one coordinate"
            )
        if profile.coordinate.size != reference.coordinate.size:
            raise MarsProfileFormatError(
                f"{name} is on {profile.coordinate.size} points and {reference_name} "
                f"on {reference.coordinate.size}; a MARS deck's files share one "
                "radial coordinate"
            )
        if not np.array_equal(profile.coordinate, reference.coordinate):
            raise MarsProfileFormatError(
                f"{name} is on a different radial coordinate from {reference_name}; a "
                "MARS deck's files share one, and reading one file's values against "
                "another's coordinate would move every point"
            )
    return reference


def read_mars_profiles(directory: str | Path) -> KineticProfiles:
    """Read a directory of ``PROF*.IN`` into the kinetic-profile container.

    ``PROFROT.IN`` becomes ``omega_tor`` and ``PROFWE.IN`` becomes
    ``omega_exb`` -- both, when both are there, with no precedence rule.  A
    two-column ``PROF*.IN`` this module does not recognise is kept in
    :attr:`KineticProfiles.extras` under its own stem; one that is *not* two
    columns belongs to a different MARS format (``PROFPA.IN`` carries one
    column per species, ``PROFROTC.IN`` a harmonic block) and is recorded in
    provenance as present and unread rather than making the whole deck
    unreadable.

    Values are taken exactly as written, in SI.  The coordinate is not: a
    MARS deck's abscissa is ``s``, so ``psi_norm`` is ``s**2`` and
    ``normalization.method`` says so.  A deck on the toroidal-flux key is
    refused, because reaching poloidal flux from it needs an equilibrium.
    """
    workdir = Path(directory).expanduser()
    if not workdir.is_dir():
        raise FileNotFoundError(f"MARS profile directory does not exist: {workdir}")

    paths = sorted(p for p in workdir.glob("PROF*.IN") if p.is_file())
    if not paths:
        raise MarsProfileFormatError(f"{workdir} holds no PROF*.IN files")

    found: dict[str, MarsProfile] = {}
    unread: dict[str, str] = {}
    for path in paths:
        try:
            found[path.name] = read_mars_profile(path)
        except MarsProfileFormatError:
            if path.name in MARS_PROFILE_FILES:
                raise
            unread[path.name] = (
                f"{path.name} is present but not read: it is not the two-column "
                "format this module reads, so it belongs to one of MARS's other "
                "profile layouts"
            )
    if not found:
        raise MarsProfileFormatError(
            f"{workdir} holds no PROF*.IN file in the two-column format: "
            f"{', '.join(sorted(unread))}"
        )

    reference = _one_coordinate(found)
    if reference.irad != POLOIDAL_KEY:
        raise MarsProfileFormatError(
            f"{workdir.name} is on {reference.radial_variable}; converting it to the "
            "normalised poloidal flux this container carries needs an equilibrium, "
            "which a file-format reader does not have"
        )

    rotations = ("PROFROT.IN", "PROFWE.IN")
    copied = all(name in found for name in rotations) and np.array_equal(
        found["PROFROT.IN"].values, found["PROFWE.IN"].values
    )
    # Not an error -- every committed deck is like this -- but it is the
    # signature of a converter that wrote one column into both files, and
    # omega_phi and omega_E differ by the ion diamagnetic frequency.
    copied_note = (
        ". PROFROT.IN and PROFWE.IN hold identical values; MARS reads them as fluid "
        "and E x B rotation, which differ by the ion diamagnetic frequency, so one "
        "was most likely copied from the other"
    )

    fields: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = dict(unread)
    for name, profile in found.items():
        field = MARS_PROFILE_FILES.get(name)
        values = np.asarray(profile.values, dtype=float)
        if field is None:
            extras[_profile_stem(name)] = values
            provenance[_profile_stem(name)] = (
                f"{name}, a MARS profile not catalogued here"
            )
            continue
        fields[field] = values
        provenance[field] = f"{name} -- {MARS_PROFILE_LABELS[name]}" + (
            copied_note if copied and name in rotations else ""
        )

    return KineticProfiles(
        psi_norm=reference.psi_norm(),
        **fields,
        normalization=PsiNormalization(
            method="mars_s_squared",
            source=workdir.name,
            axis_value=float(reference.coordinate[0]),
            edge_value=float(reference.coordinate[-1]),
        ),
        extras=extras,
        provenance=provenance,
        source=str(workdir),
    )


def write_mars_profiles(
    profiles: KineticProfiles,
    directory: str | Path,
    *,
    include_extras: bool = True,
) -> tuple[Path, ...]:
    """Write the ``PROF*.IN`` files a profile set can fill.

    The coordinate is converted back: MARS wants ``s``, so the abscissa is
    ``sqrt(psi_norm)`` and the header key is ``1``, saying so.  Writing
    ``psi_norm`` straight out would tell MARS the column is already ``s`` and
    put every point at the square of where it belongs.

    One file per field the set actually carries, and no others: a set with
    ``omega_exb`` and no ``omega_tor`` writes ``PROFWE.IN`` and **not**
    ``PROFROT.IN``.  Filling the fluid-rotation file from the E x B one is
    the mistake this module exists to end -- MARS reads the two under
    different flags into different arrays, and they differ by the ion
    diamagnetic frequency, so a deck written that way tells MARS the plasma
    rotates at its E x B frequency.

    ``extras`` are written back as ``PROF<name>.IN`` so that a deck read and
    written keeps what it came with; ``include_extras=False`` writes only the
    catalogued five.

    Returns the paths written, in file-name order.
    """
    workdir = Path(directory).expanduser()
    psi = np.asarray(profiles.psi_norm, dtype=float)
    if np.any(psi < 0):
        raise MarsProfileFormatError(
            "psi_norm holds negative values, so the coordinate MARS wants -- its "
            "square root -- does not exist"
        )
    coordinate = np.sqrt(psi)

    columns: dict[str, np.ndarray] = {
        name: np.asarray(getattr(profiles, field), dtype=float)
        for field, name in MARS_FIELD_FILES.items()
        if getattr(profiles, field, None) is not None
    }
    if include_extras:
        for name, values in profiles.extras.items():
            columns.setdefault(f"PROF{name}.IN", np.asarray(values, dtype=float))
    if not columns:
        raise MarsProfileFormatError(
            "this profile set carries none of the quantities a MARS deck holds "
            f"({', '.join(MARS_FIELD_FILES.values())}); nothing to write"
        )

    written = [
        write_mars_profile(
            MarsProfile(coordinate=coordinate, values=values, irad=POLOIDAL_KEY),
            workdir / name,
        )
        for name, values in sorted(columns.items())
    ]
    return tuple(written)
