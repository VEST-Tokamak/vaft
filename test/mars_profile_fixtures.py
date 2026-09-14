"""Synthetic MARS profile decks, shaped like the real ones -- with one change.

The format is what the committed decks carry: a ``<count> <irad>`` header,
then that many rows of two ``%.18e`` columns, one file per quantity, every
file in a deck on the same radial coordinate.

The first column is the file's own abscissa -- ``s``, the square root of the
normalised poloidal flux, for the ``1`` key every committed deck carries --
so the ground truth returns both it and the ``psi_norm = s**2`` a reader
should produce from it.

**The one deliberate divergence: ``PROFROT.IN`` and ``PROFWE.IN`` hold
different profiles here.**  In all 29 committed files they are byte-identical,
because the converter that wrote them put the same ``.kin`` column into both.
A fixture copying that could not tell a reader that keeps fluid rotation and
E x B rotation apart from one that merges them -- which is precisely how the
defect survived in the first place.  So the two differ, and by construction
no reader can pass by accident.
"""

from __future__ import annotations

import numpy as np

#: Filename to the quantity it holds, as MARS reads them.
DECK_FILES: tuple[str, ...] = (
    "PROFDEN.IN",
    "PROFTE.IN",
    "PROFTI.IN",
    "PROFROT.IN",
    "PROFWE.IN",
)


def deck_values(psi: np.ndarray) -> dict[str, np.ndarray]:
    """One array per file. The two rotations are different profiles."""
    edge = 0.5 * (1.0 - np.tanh((psi - 0.9) / 0.05))
    return {
        "PROFDEN.IN": 4.0e19 * (0.25 + 0.75 * edge),
        "PROFTE.IN": 1.3e3 * (0.05 + 0.95 * edge),
        "PROFTI.IN": 1.4e3 * (0.05 + 0.95 * edge),
        # Fluid rotation: positive, falling outward.
        "PROFROT.IN": 5.8e4 - 4.4e4 * psi,
        # E x B rotation: the fluid rotation less a diamagnetic term, so it
        # crosses zero where the real one does. Different everywhere.
        "PROFWE.IN": 5.8e4 - 4.4e4 * psi - 3.0e4 * (0.2 + psi),
    }


def write_mars_deck(
    path,
    *,
    points=41,
    files=DECK_FILES,
    identical_rotations=False,
    mismatched_psi_in=None,
    short_psi_in=None,
    wrong_count_in=None,
    ragged_row_in=None,
    unknown_file=False,
    unreadable_file=False,
    irad=1,
    irad_in=None,
    blank_line_in=None,
):
    """Write a synthetic deck into ``path``; returns the ground truth.

    Each keyword breaks one thing: ``identical_rotations`` reproduces the
    corpus's copied pair, ``mismatched_psi_in`` puts one named file on its own
    coordinate, ``short_psi_in`` gives one file a shorter grid,
    ``wrong_count_in`` makes a header disagree with the rows beneath it,
    ``ragged_row_in`` drops a token from a row, ``unknown_file`` adds a
    two-column ``PROF*.IN`` this module does not recognise,
    ``unreadable_file`` adds one in a different MARS layout altogether,
    ``irad_in`` gives one file its own radial-variable key, and
    ``blank_line_in`` puts a blank line inside a file's rows.
    """
    psi = np.linspace(0.0, 1.0, points)
    values = deck_values(psi)
    if identical_rotations:
        values["PROFWE.IN"] = values["PROFROT.IN"].copy()

    path.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in files:
        column = values[name]
        file_psi = psi
        if mismatched_psi_in == name:
            file_psi = psi + 0.001
        if short_psi_in == name:
            file_psi, column = psi[:-1], column[:-1]
        declared = len(file_psi) + 1 if wrong_count_in == name else len(file_psi)
        key = irad_in[name] if irad_in and name in irad_in else irad
        lines = [f"{declared} {key}"]
        rows = [f"{a:.18e} {b:.18e}" for a, b in zip(file_psi, column)]
        if ragged_row_in == name:
            rows[2] = rows[2].split(" ")[0]
        if blank_line_in == name:
            rows.insert(3, "")
        lines += rows
        (path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[name] = column

    if unknown_file:
        # A real MARS file this module does not catalogue.
        extra = 2.0 + psi
        lines = [f"{points} {irad}"]
        lines += [f"{a:.18e} {b:.18e}" for a, b in zip(psi, extra)]
        (path / "PROFZEFF.IN").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["PROFZEFF.IN"] = extra

    if unreadable_file:
        # PROFPA.IN carries one column per species, not two: a different MARS
        # layout, present in a legitimate deck.
        lines = [f"{points} {irad}"]
        lines += [f"{a:.18e} {a:.18e} {a:.18e}" for a in psi]
        (path / "PROFPA.IN").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "directory": path,
        "coordinate": psi,
        "psi_norm": psi**2,
        "values": written,
        "points": points,
        "irad": irad,
    }
