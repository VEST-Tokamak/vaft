"""Reduced neutral-beam attenuation along a prescribed 1-D path (vaft #1136).

A transparent reference for NBI deposition: given a path coordinate and the
attenuation coefficient along it, the neutral survival, the fast-ion birth
per path cell and the shine-through, and -- with the beam power and energy --
the particle and power bookkeeping. It composes the relations of
:mod:`vaft.formula.nbi` and adds nothing to their physics.

It is deliberately *not* NUBEAM, ASCOT5 or BEAMS3D. What each layer answers:

==============================  ===========  ==============================
quantity                        this layer   full solver (``vaft.code.*``)
==============================  ===========  ==============================
beam particle rate              yes          yes
neutral attenuation on a path   yes          yes, 3-D beamlets and footprint
birth profile along the path    yes          yes, in (R, Z) and velocity
shine-through                   yes          yes, with divergence
beam-stopping atomic data       input        built in
prompt / delayed losses         no           yes
slowing down, heating, drive    no           yes
3-D orbit effects               no           yes
==============================  ===========  ==============================

The "birth" power profile is where neutrals become fast ions, not where the
plasma is heated. Related work: the VEST NBI description (#265) feeds the
path and components; NUBEAM results and their IMAS mapping (#592) are what
this is compared against, never re-derived; generic scales and orbits
belong to #1064 and #1092; the diagrams follow #890/#1090.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from vaft.formula.nbi import (
    beam_particle_rate_from_power_energy,
    neutral_beam_optical_depth,
    neutral_survival_fraction_from_optical_depth,
)

__all__ = ["NeutralBeamAttenuation", "neutral_beam_attenuation_along_path"]


@dataclass(frozen=True)
class NeutralBeamAttenuation:
    """What the reduced attenuation model says about one beam component along its path.

    Node quantities (``s``, ``alpha``, ``optical_depth``, ``survival_fraction``)
    have one value per path point; cell quantities (``cell_centers`` onward)
    one per interval between consecutive points, so that they sum exactly.
    """

    s: np.ndarray
    alpha: np.ndarray
    optical_depth: np.ndarray
    survival_fraction: np.ndarray
    cell_centers: np.ndarray
    birth_fraction_per_cell: np.ndarray
    birth_fraction_density: np.ndarray
    shine_through_fraction: float
    absorbed_neutral_fraction: float
    particle_rate: Optional[float] = None
    birth_rate_density: Optional[np.ndarray] = None
    power_birth_profile: Optional[np.ndarray] = None
    shine_through_power: Optional[float] = None
    absorbed_beam_power: Optional[float] = None


def neutral_beam_attenuation_along_path(
    s: np.ndarray,
    alpha: np.ndarray,
    *,
    beam_power: Optional[float] = None,
    beam_energy_eV: Optional[float] = None,
) -> NeutralBeamAttenuation:
    """Neutral survival, fast-ion birth and shine-through of one beam component on a 1-D path.

    Parameters
    ----------
    s : numpy.ndarray
        Path coordinate from the entry into the attenuating region to the
        exit, strictly increasing, 1-D [m].
    alpha : numpy.ndarray
        Attenuation coefficient $\\sum_j n_j\\sigma_j$ at each ``s``, for the
        beam's species and component energy, non-negative [1/m].
    beam_power : float, optional
        Power of this component at the entry [W].
    beam_energy_eV : float, optional
        Energy per particle of this component; with ``beam_power`` it adds
        the particle and power bookkeeping [eV].

    Returns
    -------
    NeutralBeamAttenuation
        Node quantities ``optical_depth`` [-] and ``survival_fraction`` [-];
        per cell ``birth_fraction_per_cell`` [-] and
        ``birth_fraction_density`` [1/m]; ``shine_through_fraction`` and
        ``absorbed_neutral_fraction`` [-]; with power and energy also
        ``particle_rate`` [1/s], ``birth_rate_density`` [1/(m s)],
        ``power_birth_profile`` [W/m], ``shine_through_power`` and
        ``absorbed_beam_power`` [W] [any].

    Raises
    ------
    ValueError
        ``s`` not strictly increasing or not 1-D, ``alpha`` of another length,
        negative or non-finite; only one of ``beam_power`` and
        ``beam_energy_eV`` given; a negative or non-finite power or a
        non-positive energy.

    Processing steps
    ----------------
    1. Optical depth at the nodes by trapezoidal quadrature,
       :func:`vaft.formula.nbi.neutral_beam_optical_depth`.
    2. Survival $S = e^{-\\tau}$ at the nodes.
    3. Births per cell $S_i - S_{i+1}$ -- the neutrals lost between two
       nodes -- and their density over the cell width; they telescope, so
       $\\sum_i (S_i - S_{i+1}) + S_\\mathrm{exit} = 1$ exactly on any grid.
    4. Shine-through $S_\\mathrm{exit}$; absorbed fraction $1 - S_\\mathrm{exit}$.
    5. With power and energy: $\\dot N_b = P_b/E_b$, then rate and power per
       unit path in each cell, shine-through and absorbed power, which add
       up to $P_b$ exactly.

    Applicability
    -------------
    Machine-independent. One energy component, a straight prescribed path,
    a supplied attenuation coefficient: no beam-stopping atomic data, no
    divergence, footprint, orbits, slowing down or losses after ionisation.
    Compare with, never substitute for, a full NBI code (``vaft.code.nubeam``).

    Limitations
    -----------
    On a coarse grid ($\\alpha\\,\\Delta s$ not small) the cell births remain
    exactly conserved, but the optical depth inherits the trapezoidal error
    of step 1, so the split between cells and the shine-through converge
    only as the grid is refined; the point density
    :func:`vaft.formula.nbi.beam_birth_probability_density` does not
    integrate to the absorbed fraction there.

    Provenance
    ----------
    .. [1136] Issue #1136, the reduced NBI reference layer.
    .. [Wesson] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Sec. 5.4.
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    if np.ndim(s) != 1 or alpha_arr.ndim != 1:
        raise ValueError("s and alpha must be 1-D: one beam component along one path")
    tau = neutral_beam_optical_depth(s, alpha_arr)
    s = np.asarray(s, dtype=float)
    survival = np.asarray(neutral_survival_fraction_from_optical_depth(tau), dtype=float)
    width = np.diff(s)
    per_cell = survival[:-1] - survival[1:]
    density = per_cell / width
    centers = 0.5 * (s[1:] + s[:-1])
    shine = float(survival[-1])
    absorbed = 1.0 - shine
    if (beam_power is None) != (beam_energy_eV is None):
        raise ValueError("pass both beam_power and beam_energy_eV, or neither")
    if beam_power is None:
        return NeutralBeamAttenuation(s, alpha_arr, tau, survival, centers, per_cell, density, shine, absorbed)
    rate = float(beam_particle_rate_from_power_energy(beam_power, beam_energy_eV))
    power = float(beam_power)
    return NeutralBeamAttenuation(
        s, alpha_arr, tau, survival, centers, per_cell, density, shine, absorbed,
        particle_rate=rate,
        birth_rate_density=rate * density,
        power_birth_profile=power * density,
        shine_through_power=power * shine,
        absorbed_beam_power=power * absorbed,
    )
