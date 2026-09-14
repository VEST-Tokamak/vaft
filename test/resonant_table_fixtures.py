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
* ``|Phi_res|`` **spikes at the outermost surface** rather than falling
  outward. That is what the DIII-D ideal run does (3.37e-4, 2.51e-4, 4.87e-5,
  5.74e-4 in tesla), and it is the whole reason the window boundary matters:
  the legacy edge window stops at 0.95 and drops the largest surface in the
  run. A fixture that fell monotonically would move the same number the
  *other* way and make a window bug look harmless;
* a real table carries **twenty-five** columns, not five, and most of them are
  not a response: ``rho_rational`` and ``q1_rational`` are coordinates,
  ``T_e_rational`` and ``n_e_rational`` are the equilibrium sampled at the
  surfaces. Several are identically zero on an ideal run;
* ``w_isl_v_crit``, ``Phi_res_crit`` and the profile columns are
  **identically zero** on an ideal run, which is why the geometric critical
  width has to be computed rather than read;
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
    # Magnitudes shaped like the real run's: falling outward, then spiking at
    # the last surface. A phase that turns, so a complex mean and a mean
    # magnitude are different numbers.
    magnitude = 5.0e-4 * np.exp(-2.0 * psi)
    magnitude[-1] = 1.8 * magnitude[0]
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
        # Zero on an ideal run, as GPEC writes them.
        "Phi_res_crit": np.zeros(count),
        "B_pen": 0.3 * magnitude * np.exp(1j * angle),
        # Not a response: coordinates and the equilibrium at the surfaces.
        # A reduction must not reach these unless it is asked to.
        "area_rational": 2.0 + psi,
        "dqdpsi_n_rational": np.linspace(2.0, 55.0, count),
        "rho_rational": np.sqrt(psi),
        "rho1_rational": np.sqrt(psi) * 0.98,
        "q1_rational": q * 1.01,
        "T_e_rational": np.zeros(count),
        "n_e_rational": np.zeros(count),
        # Signed and changing sign, so rectifying it is visible.
        "omega_E_rational": np.linspace(-3.0e4, 1.0e4, count),
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


def island_chain(*, overlapping_outer=True, break_at=None, count=5, spacing=0.1,
                 start=0.5, width=None):
    """A chain of islands, sorted outward, with a controllable break.

    ``overlapping_outer`` decides whether the outermost pair touches, which
    is what makes an overlap region edge-connected; ``break_at`` opens a gap
    between that pair index and the next, so a chain can overlap inside and
    still not reach the boundary.
    """
    psi = start + spacing * np.arange(count, dtype=float)
    if width is None:
        # Wide enough that neighbours touch: half-widths sum to the spacing.
        width = np.full(count, 1.2 * spacing)
    else:
        width = np.full(count, float(width))
    if not overlapping_outer:
        width = width.copy()
        width[-1] = 0.1 * spacing
        width[-2] = 0.1 * spacing
    if break_at is not None:
        width = width.copy()
        width[break_at] = 0.1 * spacing
        width[break_at + 1] = 0.1 * spacing
    return psi, width
