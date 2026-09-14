"""MARS profile inputs: one quantity per file, in SI, unscaled.

MARS reads its kinetic profiles as a small deck of two-column ASCII files --
``PROFDEN.IN``, ``PROFTE.IN``, ``PROFTI.IN``, ``PROFROT.IN``, ``PROFWE.IN``.
Each is a row count and an ``irad`` flag, then that many rows of a radial
coordinate and a value:

.. code-block:: text

    200 1
    0.000000000000000000e+00 4.233437695718514688e+19
    5.025125628140703592e-03 4.233433978258264064e+19

**The two rotation files are two different quantities**, and this module
exists because the code it replaces treated them as one.  MARS's own source
settles it:

- ``PROFROT.IN`` is the bulk toroidal **fluid** rotation, omega_phi.  MARS
  reads it under ``NPROFR = 4`` into ``ROT``/``ROTM`` with amplitude
  ``ROTE``; the usage manual calls it "experimental fluid rotation profile".
- ``PROFWE.IN`` is the toroidal **E x B** rotation, omega_E.  MARS reads it
  under ``NPROFWE = 4`` into ``ROTWEI``/``ROTWEM``; the source block is
  headed ``TOROIDAL EXB ROTATION FREQUENCY``.
- They differ by the ion diamagnetic frequency.  MARS states the relation in
  the comment on its own analytic branch: ``ROTWE = ROT - OMEGAI*``.

So the two map onto the two fields
:class:`~vaft.data.kinetic_profiles.KineticProfiles` already has, and there
is no precedence rule anywhere in this module: a deck carrying both is a
deck carrying two quantities.

**Nothing here is scaled**, and that is a finding rather than an assumption,
because "a MARS input is Alfven-normalised" is the obvious wrong guess.  Two
things rule it out.  The committed profiles run 1.4e4 to 1.0e5 rad/s against
an Alfven frequency of order 1.4e6 for MAST, so normalised they would be
0.01 to 0.07 -- which is what MARS's *namelist* amplitudes hold, not what its
profile files hold.  And MARS converts the file itself when told the file
carries absolute values: ``IF (NEXPV.EQ.1) ROTE = ZTEMP*ZTAUA0``, where
``ZTEMP`` is the on-axis file value and ``ZTAUA0`` the Alfven time, which is
dimensionally sensible only if the file is in rad/s.  MARS computes the
Alfven time itself from the equilibrium file; no part of that belongs in a
reader.

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

from .kinetic_profiles import KineticProfiles, PsiNormalization, _float, _sealed

__all__ = [
    "MARS_FIELD_FILES",
    "MARS_PROFILE_FILES",
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


def _profile_stem(filename: str) -> str:
    """``PROFXX.IN`` -> ``XX``, the name an unrecognised profile is kept under."""
    stem = filename[: -len(".IN")] if filename.upper().endswith(".IN") else filename
    return stem[len("PROF"):] if stem.upper().startswith("PROF") else stem


class MarsProfileFormatError(ValueError):
    """The file is not shaped like a MARS profile input."""


@dataclass(frozen=True, eq=False)
class MarsProfile:
    """One ``PROF*.IN``: a radial coordinate and one quantity, as written.

    ``irad`` is the header's second field, kept so a file round-trips; it is
    ``1`` in every deck looked at.
    """

    psi_norm: np.ndarray
    values: np.ndarray
    irad: int = 1
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "psi_norm", _sealed(self.psi_norm))
        object.__setattr__(self, "values", _sealed(self.values))
        if self.values.size != self.psi_norm.size:
            raise MarsProfileFormatError(
                f"{self.values.size} values against {self.psi_norm.size} coordinate "
                "points"
            )

    def __len__(self) -> int:
        return int(self.psi_norm.size)


def read_mars_profile(path: str | Path) -> MarsProfile:
    """Read one ``PROF*.IN``.

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

    lines = text.splitlines()
    if not lines:
        raise MarsProfileFormatError(f"{path.name} is empty")

    header = lines[0].split()
    if len(header) < 2:
        raise MarsProfileFormatError(
            f"{path.name} starts with {lines[0].strip()!r}; a MARS profile begins "
            "with a row count and an irad flag"
        )
    try:
        count, irad = int(header[0]), int(header[1])
    except ValueError as exc:
        raise MarsProfileFormatError(
            f"{path.name} starts with {lines[0].strip()!r}, which is not a row count "
            "and an irad flag"
        ) from exc
    if count <= 0:
        raise MarsProfileFormatError(f"{path.name} declares {count} rows")

    rows = [line for line in lines[1:] if line.strip()]
    if len(rows) != count:
        raise MarsProfileFormatError(
            f"{path.name} declares {count} rows and carries {len(rows)}; a count that "
            "disagrees with the file is a truncated or appended-to deck, not a "
            "difference to read past"
        )

    table = np.empty((count, 2), dtype=float)
    for index, row in enumerate(rows):
        tokens = row.split()
        if len(tokens) != 2:
            raise MarsProfileFormatError(
                f"{path.name} row {index + 1} has {len(tokens)} columns against the 2 "
                f"a MARS profile carries: {row!r}"
            )
        table[index] = [_float(tokens[0]), _float(tokens[1])]

    return MarsProfile(
        psi_norm=table[:, 0], values=table[:, 1], irad=irad, source=str(path)
    )


def write_mars_profile(profile: MarsProfile, path: str | Path) -> Path:
    """Write one ``PROF*.IN``.

    ``%.18e``, which is what the decks carry and more than enough to round a
    double trip exactly.  Refuses a value that is not finite: MARS splines
    these profiles, and one NaN spreads through the result.
    """
    path = Path(path).expanduser()
    psi = np.asarray(profile.psi_norm, dtype=float)
    values = np.asarray(profile.values, dtype=float)
    for column, what in ((psi, "coordinate"), (values, "value")):
        if not np.all(np.isfinite(column)):
            raise MarsProfileFormatError(
                f"{path.name} would carry "
                f"{int(np.count_nonzero(~np.isfinite(column)))} {what}s that are not "
                "finite; MARS splines what it reads"
            )

    lines = [f"{values.size} {int(profile.irad)}"]
    lines.extend(f"{a:.18e} {b:.18e}" for a, b in zip(psi, values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _shared_coordinate(found: Mapping[str, MarsProfile]) -> np.ndarray:
    """The deck's one radial coordinate, refusing a deck with more than one.

    The reader this replaces kept whichever file it read first and grafted
    every other file's values onto it element by element -- warning when the
    coordinates differed in value, and saying nothing at all when they
    differed in length.
    """
    reference_name, reference = next(iter(found.items()))
    for name, profile in found.items():
        if profile.psi_norm.size != reference.psi_norm.size:
            raise MarsProfileFormatError(
                f"{name} is on {profile.psi_norm.size} points and {reference_name} on "
                f"{reference.psi_norm.size}; a MARS deck's files share one radial "
                "coordinate"
            )
        if not np.array_equal(profile.psi_norm, reference.psi_norm):
            raise MarsProfileFormatError(
                f"{name} is on a different radial coordinate from {reference_name}; a "
                "MARS deck's files share one, and reading one file's values against "
                "another's coordinate would move every point"
            )
    return np.asarray(reference.psi_norm, dtype=float)


def read_mars_profiles(directory: str | Path) -> KineticProfiles:
    """Read a directory of ``PROF*.IN`` into the kinetic-profile container.

    ``PROFROT.IN`` becomes ``omega_tor`` and ``PROFWE.IN`` becomes
    ``omega_exb`` -- both, when both are there, with no precedence rule.  A
    ``PROF*.IN`` this module does not recognise is kept in
    :attr:`KineticProfiles.extras` under its own stem rather than dropped.

    Values are taken exactly as written, in SI, and so is the coordinate:
    ``normalization.method`` is ``"as_read"``.
    """
    workdir = Path(directory).expanduser()
    if not workdir.is_dir():
        raise FileNotFoundError(f"MARS profile directory does not exist: {workdir}")

    paths = sorted(p for p in workdir.glob("PROF*.IN") if p.is_file())
    if not paths:
        raise MarsProfileFormatError(f"{workdir} holds no PROF*.IN files")

    found = {path.name: read_mars_profile(path) for path in paths}
    psi_norm = _shared_coordinate(found)

    fields: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}
    for name, profile in found.items():
        field = MARS_PROFILE_FILES.get(name)
        values = np.asarray(profile.values, dtype=float)
        if field is None:
            extras[_profile_stem(name)] = values
            provenance[_profile_stem(name)] = f"{name}, a MARS profile not catalogued here"
            continue
        fields[field] = values
        provenance[field] = f"{name} -- {MARS_PROFILE_LABELS[name]}"

    rotation = ("PROFROT.IN", "PROFWE.IN")
    if all(name in found for name in rotation) and np.array_equal(
        found["PROFROT.IN"].values, found["PROFWE.IN"].values
    ):
        # Not an error -- every committed deck is like this -- but it is the
        # signature of a converter that wrote one column into both files, and
        # omega_phi and omega_E differ by the ion diamagnetic frequency.
        note = (
            "PROFROT.IN and PROFWE.IN hold identical values; MARS reads them as "
            "fluid and E x B rotation, which differ by the ion diamagnetic "
            "frequency, so one was most likely copied from the other"
        )
        provenance["omega_tor"] += f". {note}"
        provenance["omega_exb"] += f". {note}"

    return KineticProfiles(
        psi_norm=psi_norm,
        **fields,
        normalization=PsiNormalization(method="as_read", source=workdir.name),
        extras=extras,
        provenance=provenance,
        source=str(workdir),
    )


def write_mars_profiles(
    profiles: KineticProfiles,
    directory: str | Path,
    *,
    irad: int = 1,
) -> tuple[Path, ...]:
    """Write the ``PROF*.IN`` files a profile set can fill.

    One file per field the set actually carries, and no others: a set with
    ``omega_exb`` and no ``omega_tor`` writes ``PROFWE.IN`` and **not**
    ``PROFROT.IN``.  Filling the fluid-rotation file from the E x B one is
    the mistake this module exists to end -- MARS reads the two under
    different flags into different arrays, and they differ by the ion
    diamagnetic frequency, so a deck written that way tells MARS the plasma
    rotates at its E x B frequency.

    Returns the paths written, in file-name order.
    """
    workdir = Path(directory).expanduser()
    written: list[Path] = []
    for field, name in MARS_FIELD_FILES.items():
        values = getattr(profiles, field, None)
        if values is None:
            continue
        written.append(
            write_mars_profile(
                MarsProfile(
                    psi_norm=np.asarray(profiles.psi_norm, dtype=float),
                    values=np.asarray(values, dtype=float),
                    irad=irad,
                ),
                workdir / name,
            )
        )
    if not written:
        raise MarsProfileFormatError(
            "this profile set carries none of the quantities a MARS deck holds "
            f"({', '.join(MARS_FIELD_FILES)}); nothing to write"
        )
    return tuple(sorted(written))
