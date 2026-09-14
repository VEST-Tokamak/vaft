"""Synthetic resonant tables and q profiles, shaped like GPEC's.

A resonant table is one row per rational surface: the coordinate, the mode
numbers, and a column per quantity -- complex for the fluxes, currents and
the resonance parameter, real for the island widths and Chirikov.

The awkward details here are the ones a real run has:

* the surfaces are **few** -- four in the DIII-D n=1 ideal example -- so a
  radial window can legitimately contain none, and a reduction has to say so
  rather than raise;
* they are **not evenly spaced**: 0.59, 0.82, 0.93, 0.99, bunched towards the
  edge, so where a window boundary falls decides how many it catches;
* ``w_isl_v_crit`` is **identically zero** on an ideal run, which is why the
  geometric critical width has to be computed rather than read;
* the complex columns have a phase that varies surface to surface, so a
  reduction that averaged them instead of their magnitudes would give a
  different answer.
"""

from __future__ import annotations

import numpy as np

#: Where the n=1 surfaces sit in the DIII-D ideal example, to four figures.
REFERENCE_SURFACES = (0.5936, 0.8186, 0.9282, 0.9884)


def resonant_table(surfaces=REFERENCE_SURFACES, *, n_tor=1, phase=True):
    """A resonant table with one row per surface."""
    psi = np.asarray(surfaces, dtype=float)
    count = psi.size
    q = np.arange(2, 2 + count, dtype=float)
    # Magnitudes that fall outward, and a phase that turns, so a complex mean
    # and a mean magnitude are different numbers.
    magnitude = 5.0e-4 * np.exp(-2.0 * psi)
    angle = np.linspace(0.0, 2.4, count) if phase else np.zeros(count)
    flux = magnitude * np.exp(1j * angle)
    return {
        "psi_n_rational": psi,
        "q_rational": q,
        "m_rational": q * n_tor,
        "Phi_res": flux,
        "Phi_res_v": 0.6 * magnitude * np.exp(1j * angle * 0.5),
        "I_res": 1.0e3 * magnitude / magnitude[0] * np.exp(-1j * angle),
        "Delta": (1.0 + 0.5 * psi) * np.exp(1j * angle),
        "w_isl": 0.08 * np.exp(-1.5 * psi),
        "w_isl_v": 0.05 * np.exp(-1.5 * psi),
        # Zero on an ideal run, exactly as GPEC writes it.
        "w_isl_v_crit": np.zeros(count),
        "K_isl": np.linspace(0.30, 0.48, count),
        "K_isl_v": np.linspace(0.20, 0.35, count),
    }


def q_profile(points=400, *, q_axis=1.05, q_edge=5.6, reversed_shear=False):
    """A monotonic (or reversed-shear) q profile on a psi_norm grid."""
    psi = np.linspace(0.0, 1.0, points)
    if reversed_shear:
        # Dips below the axis value and comes back, so one m resonates twice.
        return psi, q_axis + 3.0 * (psi - 0.45) ** 2
    return psi, q_axis + (q_edge - q_axis) * psi**2


class PedestalStub:
    """Stands in for `vaft.process.profile.PedestalTop`.

    The real one is built by fitting a profile this module never sees, and
    only four of its attributes are read here.
    """

    def __init__(self, position=0.93, width=0.06, method="eped_fit", reason="",
                 coordinate="psi_norm", inner_edge=...):
        self.position = position
        self.width = width
        self.method = method
        self.reason = reason
        self.coordinate = coordinate
        self.inner_edge = (position - 0.5 * width) if inner_edge is ... else inner_edge
