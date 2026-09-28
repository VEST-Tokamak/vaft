"""Reduced neutral-beam attenuation along a prescribed 1-D path (vaft #1136).

A transparent reference for NBI deposition: given a path coordinate and the
attenuation coefficient along it, the neutral survival, the fast-ion birth
density and the shine-through, and -- with the beam power and energy -- the
particle and power bookkeeping. It composes the relations of
:mod:`vaft.formula.nbi` and adds nothing to their physics.

It is deliberately *not* NUBEAM, ASCOT5 or BEAMS3D: no beam divergence or
footprint, no orbits, no slowing down, no losses after ionisation. The
"birth" power profile is where neutrals become fast ions, not where the
plasma is heated. Its use is sanity checks, limiting cases and the
interpretation of a full code's result, never a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from vaft.formula.nbi import (
    beam_birth_probability_density,
    beam_particle_rate_from_power_energy,
    neutral_beam_optical_depth,
    neutral_survival_fraction_from_optical_depth,
)

__all__ = ["NeutralBeamAttenuation", "neutral_beam_attenuation_along_path"]


@dataclass(frozen=True)
class NeutralBeamAttenuation:
    """What the reduced attenuation model says about one beam component along its path."""

    s: np.ndarray
    alpha: np.ndarray
    optical_depth: np.ndarray
    survival_fraction: np.ndarray
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
        ``s``, ``alpha``, ``optical_depth`` [-], ``survival_fraction`` [-],
        ``birth_fraction_density`` [1/m], ``shine_through_fraction`` [-] and
        ``absorbed_neutral_fraction`` [-]; with power and energy also
        ``particle_rate`` [1/s], ``birth_rate_density`` [1/(m s)],
        ``power_birth_profile`` [W/m], ``shine_through_power`` [W] and
        ``absorbed_beam_power`` [W] [-].

    Raises
    ------
    ValueError
        ``s`` not strictly increasing or not 1-D, ``alpha`` of another length,
        negative or non-finite; only one of ``beam_power`` and
        ``beam_energy_eV`` given; a negative power or a non-positive energy.

    Processing steps
    ----------------
    1. Optical depth by trapezoidal quadrature,
       :func:`vaft.formula.nbi.neutral_beam_optical_depth`.
    2. Survival $S = e^{-\\tau}$ and birth density $b = \\alpha S$.
    3. Shine-through $S(s_\\mathrm{exit})$; absorbed fraction $1 - S(s_\\mathrm{exit})$,
       which equals $\\int b\\,ds$ exactly (not by quadrature).
    4. With power and energy: $\\dot N_b = P_b/E_b$, then rate and power
       densities $\\dot N_b b$ and $P_b b$, shine-through and absorbed power.

    Applicability
    -------------
    Machine-independent. One energy component, a straight prescribed path,
    a supplied attenuation coefficient: no beam-stopping atomic data, no
    divergence, footprint, orbits, slowing down or losses after ionisation.
    Compare with, never substitute for, a full NBI code (``vaft.code.nubeam``).

    Provenance
    ----------
    .. [1136] Issue #1136, the reduced NBI reference layer.
    .. [Wesson] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Sec. 5.4.
    """
    tau = neutral_beam_optical_depth(s, alpha)
    s = np.asarray(s, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if s.ndim != 1 or alpha.ndim != 1:
        raise ValueError("s and alpha must be 1-D: one beam component along one path")
    survival = np.asarray(neutral_survival_fraction_from_optical_depth(tau), dtype=float)
    birth = beam_birth_probability_density(s, alpha)
    shine = float(survival[-1])
    absorbed = 1.0 - shine
    if (beam_power is None) != (beam_energy_eV is None):
        raise ValueError("pass both beam_power and beam_energy_eV, or neither")
    if beam_power is None:
        return NeutralBeamAttenuation(s, alpha, tau, survival, birth, shine, absorbed)
    rate = float(beam_particle_rate_from_power_energy(beam_power, beam_energy_eV))
    return NeutralBeamAttenuation(
        s, alpha, tau, survival, birth, shine, absorbed,
        particle_rate=rate,
        birth_rate_density=rate * birth,
        power_birth_profile=float(beam_power) * birth,
        shine_through_power=float(beam_power) * shine,
        absorbed_beam_power=float(beam_power) * absorbed,
    )
