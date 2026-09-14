"""Synthetic Osborne pfiles, shaped like the real ones.

The layout mirrors the reference corpus (MAST 45453, TRANSP X01, 57 files):
an ``N Z A of ION SPECIES`` block of three rows, then 22 profile sections in
one fixed order, each a header naming the quantity and its unit followed by
rows of ``psinorm``, value and derivative in ``%.8e``.

The awkward details are the ones those files actually have:

* the **derivative column is not the numerical derivative of the values**.
  In the real files it is whatever the original fitting tool computed, and
  recomputing it reproduces none of the 57; so the fixture writes a
  derivative that is deliberately inconsistent with its own values, and a
  reader or writer that quietly recomputes one is caught rather than
  flattered;
* every section carries **the same psinorm column**, which is true bit for
  bit in all 57 files and is therefore an invariant worth breaking on
  purpose;
* ``omghb`` declares an **empty** unit -- ``omghb()`` -- so a reader must
  tell "no unit" from "no parentheses";
* ``omeg``, ``omegp`` and ``omgeb`` hold three **distinct, individually
  recognisable** profiles.  They are three different physical quantities
  whose names differ by two letters, and the reader this replaces merged
  them; a fixture where they looked alike could not tell.

The values are synthetic; nothing here is copied from a real run.
"""

from __future__ import annotations

import numpy as np

#: Section order and declared units, as the reference files carry them.
#: Written out here rather than imported from the module under test: a
#: fixture that takes the format from the code it checks can only compare
#: the code with itself.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("ne", "10^20/m^3"),
    ("ni", "10^20/m^3"),
    ("nz1", "10^20/m^3"),
    ("nb", "10^20/m^3"),
    ("te", "KeV"),
    ("ti", "KeV"),
    ("ptot", "KPa"),
    ("pb", "KPa"),
    ("omeg", "kRad/s"),
    ("omegp", "kRad/s"),
    ("omgvb", "kRad/s"),
    ("omgpp", "kRad/s"),
    ("omgeb", "kRad/s"),
    ("er", "kV/m"),
    ("ommvb", "kRad/s"),
    ("ommpp", "kRad/s"),
    ("omevb", "kRad/s"),
    ("omepp", "kRad/s"),
    ("kpol", "km/s/T"),
    ("omghb", ""),
    ("vtor1", "km/s"),
    ("vpol1", "km/s"),
)

#: Impurity, main ion, fast ion -- carbon and two deuterium rows, the block
#: the reference files carry.
SPECIES_ROWS: tuple[tuple[float, float, float], ...] = (
    (6.0, 6.0, 12.0107),
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
)


def profile_values(psi: np.ndarray) -> dict[str, np.ndarray]:
    """One array per section, with the three rotation quantities distinct."""
    edge = 0.5 * (1.0 - np.tanh((psi - 0.93) / 0.04))
    density = 0.4 * (0.3 + 0.7 * edge)
    return {
        "ne": density,
        "ni": 0.91 * density,
        "nz1": 0.015 * density,
        "nb": 0.005 * density,
        "te": 1.2 * (0.1 + 0.9 * edge),
        "ti": 1.1 * (0.1 + 0.9 * edge),
        "ptot": 15.0 * (0.05 + 0.95 * edge),
        "pb": 1.1 * (0.05 + 0.95 * edge),
        # The three that the legacy reader merged: each one recognisable on
        # sight, and no two equal anywhere.
        "omeg": 10.0 + psi,
        "omegp": -(3.0 + psi),
        "omgeb": 7.0 * psi - 2.0,
        "omgvb": 4.0 - 2.0 * psi,
        "omgpp": -1.5 - psi,
        "er": 1.0 - 2.0 * psi,
        "ommvb": 5.0 + 3.0 * psi,
        "ommpp": -8.0 * psi,
        "omevb": -20.0 * psi,
        "omepp": 30.0 * psi,
        "kpol": 2.0 + 20.0 * psi,
        "omghb": 1.0e-4 * (1.0 - 2.0 * psi),
        "vtor1": 0.5 + 9.0 * psi,
        "vpol1": 0.3 + 2.0 * psi,
    }


def write_pfile_file(
    path,
    *,
    name="p090001.00500",
    points=41,
    species=True,
    mismatched_psi_in=None,
    short_section=None,
    ragged_row=None,
    unknown_unit_in=None,
    unknown_section=False,
    only=None,
    descending_psi=False,
):
    """Write a synthetic pfile; returns the ground truth it wrote.

    Each keyword breaks exactly one thing: ``mismatched_psi_in`` puts one
    named section on its own coordinate, ``short_section`` makes a section's
    declared row count exceed the rows that follow, ``ragged_row`` drops a
    token from one row, ``unknown_unit_in`` relabels a named section's unit,
    and ``unknown_section`` appends a block this module does not catalogue.
    """
    psi = np.linspace(0.0, 1.0, points)
    values = profile_values(psi)
    order = [(key, unit) for key, unit in SECTIONS if only is None or key in only]

    # A derivative that is emphatically not the numerical derivative of the
    # values, so that recomputing one is a visible change rather than a
    # rounding difference.
    derivatives = {key: -0.5 * array - 0.125 for key, array in values.items()}

    written_psi = psi[::-1].copy() if descending_psi else psi
    lines: list[str] = []
    if species:
        lines.append(f"{len(SPECIES_ROWS)} N Z A of ION SPECIES")
        lines += [f" {n:.6f}   {z:.6f}   {a:.6f}" for n, z, a in SPECIES_ROWS]

    for key, unit in order:
        section_psi = written_psi
        if mismatched_psi_in == key:
            section_psi = written_psi + 0.001
        if unknown_unit_in == key:
            unit = "furlongs/fortnight"
        rows = list(zip(section_psi, values[key], derivatives[key]))
        declared = len(rows) + 1 if short_section == key else len(rows)
        lines.append(f"{declared} psinorm {key}({unit}) d{key}/dpsiN")
        body = [f" {a:.8e}   {b:.8e}   {c:.8e}" for a, b, c in rows]
        if ragged_row is not None and key == order[0][0]:
            body[ragged_row] = body[ragged_row].rsplit("   ", 1)[0]
        lines += body

    if unknown_section:
        extra = 1.0 + psi
        lines.append(f"{points} psinorm zeff(-) dzeff/dpsiN")
        lines += [
            f" {a:.8e}   {b:.8e}   {c:.8e}"
            for a, b, c in zip(written_psi, extra, -0.5 * extra - 0.125)
        ]

    target = path / name
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": target,
        "psi_norm": written_psi,
        "values": values,
        "derivatives": derivatives,
        "keys": [key for key, _ in order] + (["zeff"] if unknown_section else []),
        "species": SPECIES_ROWS,
        "points": points,
    }
