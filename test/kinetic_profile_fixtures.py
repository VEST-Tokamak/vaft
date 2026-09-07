"""Synthetic kinetic-profile files, shaped like the real ones.

The ``.kin`` layout is GPEC's own (``pentrc/inputs.f90``): six columns
``psi_n, n_i, n_e, T_i, T_e, omega_E``, a header line that is a comment, and
"(nearly) arbitrary header and/or footer, with the exception that no lines
start with a number".  The awkward details here are the ones a real file has:
a radial coordinate that does *not* span exactly [0, 1], a negative rotation
column, and a footer.
"""

from __future__ import annotations

import numpy as np

from vaft.data.kinetic_profiles import KIN_HEADER

#: A real MAST-U .kin spans this, not [0, 1] -- which is what makes a
#: min-max rescale on read move every interior point.
TRUNCATED_SPAN = (0.00494621873, 1.0)


def profile_columns(points=41, span=TRUNCATED_SPAN):
    """The six ``.kin`` columns as arrays, in file order."""
    psi = np.linspace(span[0], span[1], points)
    edge = 0.5 * (1.0 - np.tanh((psi - 0.93) / 0.04))
    n_e = 4.0e19 * (0.3 + 0.7 * edge)
    return {
        "psi_norm": psi,
        "n_i": 0.91 * n_e,
        "n_e": n_e,
        "T_i": 1.2e3 * (0.1 + 0.9 * edge),
        "T_e": 1.1e3 * (0.1 + 0.9 * edge),
        # Negative, as a real E x B frequency profile is.
        "omega_exb": -1.2e4 * (0.2 + 0.8 * psi),
    }


def write_kin_file(path, *, points=41, span=TRUNCATED_SPAN, header=True, footer=False,
                   extra_column=False, columns=None):
    """Write a synthetic ``.kin``; returns the columns it wrote."""
    data = columns if columns is not None else profile_columns(points, span)
    table = np.column_stack(list(data.values()))
    if extra_column:
        table = np.column_stack([table, np.arange(table.shape[0], dtype=float)])
    lines = [KIN_HEADER] if header else []
    lines.extend("  " + "   ".join(f"{value:.8e}" for value in row) for row in table)
    if footer:
        lines += ["", "written by a synthetic fixture", "provenance: none"]
    target = path / "synthetic.kin"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data
