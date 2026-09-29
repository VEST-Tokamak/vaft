r"""Neutral beam injection: particle rate, neutral attenuation along a path, birth density, shine-through.

The NBI-specific relations of a one-dimensional neutral-attenuation model:
the injected particle rate of a monoenergetic component, the optical depth of
a prescribed attenuation coefficient, the neutral survival fraction, the
fast-ion birth probability density along the path, the shine-through
fraction, and the ideal injected toroidal angular-momentum rate.

The attenuation coefficient $\alpha(s) = \sum_j n_j\sigma_j$ is an *input*:
beam-stopping cross sections and effective stopping coefficients are atomic
data this module does not own, and no universal stopping model is built in.
Generic orbit and characteristic-scale quantities (Larmor radius, magnetic
moment, $P_\phi$) belong to ``vaft.formula.particle``. This is a reference
layer, not NUBEAM, ASCOT5 or BEAMS3D: no orbits, no slowing down, no 3-D
deposition.

Notation
--------
P_b      : beam power of one energy component              [W]
E_b      : energy per beam particle of that component       [eV]
Ndot_b   : injected particle rate                            [1/s]
s        : distance along the beam path from entry           [m]
alpha    : attenuation coefficient sum_j n_j sigma_j          [1/m]
tau      : optical depth                                     [-]
S        : neutral survival fraction                         [-]
b        : birth probability density per unit path           [1/m]

Conventions
-----------
Energies are per particle in electronvolts at the API boundary, the energy
of the component considered (full, half or third). The path coordinate
starts at the beam's entry into the attenuating region and increases along
the beam.

References
----------
.. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Sec. 5.4.
.. [2] R. J. Goldston and P. H. Rutherford, *Introduction to Plasma Physics*,
       IOP (1995), Ch. 14.
"""

import numpy as np

from .constants import QE

__all__ = [
    "beam_particle_rate_from_power_energy",
    "neutral_beam_optical_depth",
    "neutral_survival_fraction_from_optical_depth",
    "beam_birth_probability_density",
    "shine_through_fraction",
    "injected_toroidal_angular_momentum_rate",
]


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def _path(s, alpha):
    s = np.asarray(s, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if s.ndim != 1 or s.size < 2 or alpha.ndim < 1 or alpha.shape[-1] != s.size:
        raise ValueError("s must be 1-D with at least two points, and alpha must end with the same length")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(alpha))):
        raise ValueError("s and alpha must be finite")
    if np.any(np.diff(s) <= 0.0):
        raise ValueError("s must increase strictly along the beam")
    if np.any(alpha < 0.0):
        raise ValueError("alpha must be non-negative: an attenuation coefficient does not create neutrals")
    return s, alpha


def beam_particle_rate_from_power_energy(P_b, E_b_eV):
    r"""Injected particle rate of one monoenergetic beam component.

    $$\dot N_b = \frac{P_b}{E_b}$$

    Parameters
    ----------
    P_b : float or np.ndarray
        Power carried by the component, non-negative [W].
    E_b_eV : float or np.ndarray
        Energy per particle of that component, positive [eV].

    Returns
    -------
    float or np.ndarray
        Particles injected per second [1/s].

    Raises
    ------
    ValueError
        A negative or non-finite power, or a non-positive energy.

    Convention
    ----------
    $E_b$ is the energy *per particle of this component*: a source at
    accelerating voltage $V$ delivers full-, half- and third-energy
    components at $eV$, $eV/2$ and $eV/3$ (from D$^+$, D$_2^+$, D$_3^+$), each
    with its own power, so the rates are computed per component and summed.
    No species is implied; the energy converts from eV with the elementary
    charge.

    Physical interpretation
    -----------------------
    The source term for particle bookkeeping: at fixed power a lower-energy
    component injects more particles.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.
    """
    P_b = np.asarray(P_b, dtype=float)
    E = np.asarray(E_b_eV, dtype=float)
    if np.any(~np.isfinite(P_b)) or np.any(P_b < 0.0):
        raise ValueError("P_b must be non-negative and finite")
    if np.any(~np.isfinite(E)) or np.any(E <= 0.0):
        raise ValueError("E_b_eV must be positive and finite")
    return _out(P_b / (E * QE))


def neutral_beam_optical_depth(s, alpha):
    r"""Optical depth accumulated along the beam path, by the trapezoidal rule.

    $$\tau(s) = \int_0^s \alpha(l)\,dl$$

    Parameters
    ----------
    s : np.ndarray
        Path coordinate from the entry, strictly increasing [m].
    alpha : np.ndarray
        Attenuation coefficient $\sum_j n_j\sigma_j$ at each ``s``,
        non-negative; leading axes are independent beams [1/m].

    Returns
    -------
    np.ndarray
        $\tau$ at each ``s``, zero at ``s[0]`` [-].

    Raises
    ------
    ValueError
        ``s`` not strictly increasing, shapes that differ, a negative or
        non-finite ``alpha``.

    Convention
    ----------
    $\alpha$ is prescribed: the caller composes it from densities and
    beam-stopping cross sections (or effective stopping coefficients) for
    the beam energy and species. Trapezoidal quadrature, so $\tau$ is exact
    for piecewise-linear $\alpha$.

    Physical interpretation
    -----------------------
    The number of e-foldings of the neutral flux: a beam with
    $\tau_\mathrm{exit} \approx 3$ deposits 95 % of its particles.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.
    """
    s, alpha = _path(s, alpha)
    steps = 0.5 * (alpha[..., 1:] + alpha[..., :-1]) * np.diff(s)
    return np.concatenate([np.zeros(alpha.shape[:-1] + (1,)), np.cumsum(steps, axis=-1)], axis=-1)


def neutral_survival_fraction_from_optical_depth(tau):
    r"""Fraction of the injected neutrals not yet ionised.

    $$S = e^{-\tau}$$

    Parameters
    ----------
    tau : float or np.ndarray
        Optical depth, non-negative [-].

    Returns
    -------
    float or np.ndarray
        $S \in (0, 1]$ [-].

    Raises
    ------
    ValueError
        A negative or NaN optical depth.

    Physical interpretation
    -----------------------
    The neutral flux at depth $\tau$ over the injected flux. Every neutral
    removed from it has become a fast ion, by ionisation or by charge
    exchange with a plasma ion; the thermal neutral that charge exchange
    leaves behind is not followed.

    Assumptions
    -----------
    Straight-line neutral flight, no re-neutralisation feeding the beam.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.
    """
    tau = np.asarray(tau, dtype=float)
    if np.any(~np.isfinite(tau)) or np.any(tau < 0.0):
        raise ValueError("tau must be non-negative and finite")
    return _out(np.exp(-tau))


def beam_birth_probability_density(s, alpha):
    r"""Fast-ion birth probability per unit path length along the beam.

    $$b(s) = -\frac{dS}{ds} = \alpha(s)\,e^{-\tau(s)}$$

    Parameters
    ----------
    s : np.ndarray
        Path coordinate from the entry, strictly increasing [m].
    alpha : np.ndarray
        Attenuation coefficient at each ``s``, non-negative [1/m].

    Returns
    -------
    np.ndarray
        $b(s)$: probability that an injected neutral is ionised per unit
        length at $s$ [1/m].

    Raises
    ------
    ValueError
        As ``neutral_beam_optical_depth``.

    Convention
    ----------
    A density along the 1-D path coordinate, not a volumetric deposition:
    $\int b\,ds + S(s_\mathrm{exit}) = 1$ over the whole path. Mapping it to
    flux surfaces or $(R, Z)$ needs the beam geometry and footprint.

    Physical interpretation
    -----------------------
    Largest where $\alpha$ is large *and* the beam has not yet been
    depleted: a dense edge ionises the beam early, which is why a
    high-density plasma deposits NBI off-axis.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.
    """
    s, alpha = _path(s, alpha)
    return alpha * np.exp(-neutral_beam_optical_depth(s, alpha))


def shine_through_fraction(s, alpha):
    r"""Fraction of the injected neutrals that leave the path un-ionised.

    $$f_\mathrm{shine} = S(s_\mathrm{exit}) = e^{-\tau(s_\mathrm{exit})}$$

    Parameters
    ----------
    s : np.ndarray
        Path coordinate from entry to exit, strictly increasing [m].
    alpha : np.ndarray
        Attenuation coefficient at each ``s``, non-negative [1/m].

    Returns
    -------
    float or np.ndarray
        $f_\mathrm{shine} \in (0, 1]$, one per beam [-].

    Raises
    ------
    ValueError
        As ``neutral_beam_optical_depth``.

    Convention
    ----------
    The reduced neutral-attenuation result along the prescribed path, not a
    NUBEAM-equivalent shine-through: no beam divergence, footprint, multiple
    energy components or re-neutralisation.

    Physical interpretation
    -----------------------
    Power that reaches the far wall as neutrals: the low-density limit of
    NBI heating, and a wall-load constraint on low-density operation.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.
    """
    s, alpha = _path(s, alpha)
    return _out(np.exp(-neutral_beam_optical_depth(s, alpha)[..., -1]))


def injected_toroidal_angular_momentum_rate(Ndot_b, m, R_tan, v):
    r"""Toroidal angular momentum carried into the machine per second by the injected particles.

    $$\dot L_\phi = \dot N_b\,m\,R_\mathrm{tan}\,v$$

    Parameters
    ----------
    Ndot_b : float or np.ndarray
        Injected particle rate, non-negative [1/s].
    m : float or np.ndarray
        Beam-particle mass, positive [kg].
    R_tan : float or np.ndarray
        Tangency radius of the beam line, signed: positive for injection
        along $+\phi$ (counter-clockwise from above), negative for the
        opposite direction [m].
    v : float or np.ndarray
        Beam-particle speed, non-negative [m/s].

    Returns
    -------
    float or np.ndarray
        Angular-momentum injection rate about the symmetry axis, positive
        along $+\phi$ [N m].

    Raises
    ------
    ValueError
        A negative rate or speed, a non-positive mass, or a non-finite input.

    Convention
    ----------
    For a straight beam $m\,\mathbf R\times\mathbf v$ about the axis is
    $mR_\mathrm{tan}v$ at every point of the line, so the tangency radius
    carries the geometry and its sign the direction. IMAS ``nbi`` stores an
    unsigned ``beamlets_group.tangency_radius`` and a separate
    ``direction`` ($\pm1$, counter-clockwise from above): pass
    ``direction * tangency_radius``.

    Physical interpretation
    -----------------------
    The ideal rate at which the beam brings angular momentum in -- an upper
    bound, not the torque on the plasma: shine-through, prompt and delayed
    losses, and the $\mathbf J\times\mathbf B$ torque of fast-ion radial
    currents decide how much of it the plasma receives and where.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.
    """
    Ndot_b, m, R_tan, v = (np.asarray(x, dtype=float) for x in (Ndot_b, m, R_tan, v))
    if not all(np.all(np.isfinite(x)) for x in (Ndot_b, m, R_tan, v)):
        raise ValueError("inputs must be finite")
    if np.any(Ndot_b < 0.0) or np.any(v < 0.0) or np.any(m <= 0.0):
        raise ValueError("Ndot_b and v must be non-negative and m positive")
    return _out(Ndot_b * m * R_tan * v)
