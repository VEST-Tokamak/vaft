"""
Plasma-wall interaction: collision kinematics, reflection and recycling definitions, sputtering threshold.

The small, exactly defined relations the plasma-wall vocabulary rests on --
not response data. Reflection and sputtering *coefficients* depend on
projectile, target, energy, angle and surface state, and come from tables or
codes (TRIM/SDTrimSP, Eckstein fits); nothing here supplies them. What is
here keeps the concepts apart: particle versus energy reflection, reflection
versus recycling versus retention, the projectile versus the target species.

Notation
--------
m_1       : projectile mass                                  [kg or u]
m_2       : target-atom mass                                  [kg or u]
R_N       : particle reflection coefficient                   [-]
R_E       : energy reflection coefficient                     [-]
E_s       : surface binding energy of the target              [eV]
Gamma     : particle flux density                             [m^-2 s^-1]

Conventions
-----------
Masses enter only as ratios, so any consistent unit works. Coefficients are
per incident particle (``R_N``) or per unit incident energy (``R_E``), for one
projectile-target pair, energy and angle.

References
----------
.. [1] W. Eckstein, *Computer Simulation of Ion-Solid Interactions*,
       Springer (1991).
.. [2] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
       IOP (2000), Ch. 3.
"""

import numpy as np

__all__ = [
    "binary_collision_energy_transfer_factor",
    "mean_reflected_energy_fraction",
    "recycling_coefficient",
    "sputtering_threshold_bohdansky",
]


def _positive(value, name):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return arr


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def binary_collision_energy_transfer_factor(m_1, m_2):
    r"""Largest fraction of a projectile's energy one elastic collision can give a target atom.

    $$\gamma = \frac{4m_1m_2}{(m_1 + m_2)^2}$$

    Parameters
    ----------
    m_1 : float or np.ndarray
        Projectile mass, positive [kg or u].
    m_2 : float or np.ndarray
        Target-atom mass, positive, same unit [kg or u].

    Returns
    -------
    float or np.ndarray
        $\gamma \in (0, 1]$, one for equal masses [-].

    Raises
    ------
    ValueError
        A mass is not positive.

    Physical interpretation
    -----------------------
    A head-on elastic collision transfers $\gamma E$. A light projectile
    (D on W: $\gamma \approx 0.043$) cannot give a heavy target atom much
    energy in one collision, which is why light ions sputter heavy targets
    only well above the surface binding energy -- the origin of the high
    sputtering threshold of tungsten under hydrogen.

    Assumptions
    -----------
    Two-body elastic collision, target initially at rest; exact kinematics.

    References
    ----------
    .. [1] W. Eckstein, *Computer Simulation of Ion-Solid Interactions*,
           Springer (1991), Ch. 2.
    """
    m_1 = _positive(m_1, "m_1")
    m_2 = _positive(m_2, "m_2")
    return _out(4.0 * m_1 * m_2 / (m_1 + m_2) ** 2)


def mean_reflected_energy_fraction(R_N, R_E):
    r"""Mean energy of a reflected particle, as a fraction of the incident energy.

    $$\frac{\langle E_\mathrm{refl}\rangle}{E_\mathrm{in}} = \frac{R_E}{R_N}$$

    Parameters
    ----------
    R_N : float or np.ndarray
        Particle reflection coefficient, in (0, 1] [-].
    R_E : float or np.ndarray
        Energy reflection coefficient, in [0, R_N] [-].

    Returns
    -------
    float or np.ndarray
        $\langle E_\mathrm{refl}\rangle/E_\mathrm{in}$ [-].

    Raises
    ------
    ValueError
        ``R_N`` outside (0, 1] or ``R_E`` outside [0, ``R_N``].

    Convention
    ----------
    $R_N$ counts reflected particles per incident particle, $R_E$ reflected
    energy per incident energy; they are different quantities with different
    data, and $R_E \le R_N$ because a reflected particle leaves with at most
    its incident energy.

    Physical interpretation
    -----------------------
    Particle balance and energy balance are not the same bookkeeping: a
    surface may return most particles ($R_N$ large) while keeping most of
    their energy ($R_E/R_N$ small), and it is $R_E$, not $R_N$, that enters
    the power the wall absorbs.

    References
    ----------
    .. [1] W. Eckstein, *Computer Simulation of Ion-Solid Interactions*,
           Springer (1991), Ch. 8.
    """
    R_N = np.asarray(R_N, dtype=float)
    R_E = np.asarray(R_E, dtype=float)
    if np.any(~(R_N > 0.0)) or np.any(R_N > 1.0):
        raise ValueError("R_N must lie in (0, 1]")
    if np.any(R_E < 0.0) or np.any(R_E > R_N):
        raise ValueError("R_E must lie in [0, R_N]: a reflected particle cannot leave with more than it brought")
    return _out(R_E / R_N)


def recycling_coefficient(Gamma_reflected, Gamma_reemitted, Gamma_incident):
    r"""Recycling coefficient: particles returned to the plasma per incident particle.

    $$R = \frac{\Gamma_\mathrm{refl} + \Gamma_\mathrm{re\text{-}em}}{\Gamma_\mathrm{in}},\qquad
      1 - R = \frac{\Gamma_\mathrm{retained}}{\Gamma_\mathrm{in}}$$

    Parameters
    ----------
    Gamma_reflected : float or np.ndarray
        Promptly reflected flux (fast atoms), non-negative [m^-2 s^-1].
    Gamma_reemitted : float or np.ndarray
        Flux re-emitted after implantation (thermal atoms and molecules,
        counted as atoms), non-negative [m^-2 s^-1].
    Gamma_incident : float or np.ndarray
        Incident ion plus atom flux, positive [m^-2 s^-1].

    Returns
    -------
    float or np.ndarray
        $R$; below one while the wall retains, above one while it releases a
        previous inventory [-].

    Raises
    ------
    ValueError
        A returned flux is negative or ``Gamma_incident`` is not positive.

    Convention
    ----------
    Counts atoms: a re-emitted D$_2$ molecule is two. Reflection is the
    prompt part only; recycling includes the delayed re-emission; retention
    is what neither returns. $R$ is not bounded by one: an outgassing wall
    returns more than it receives.

    References
    ----------
    .. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Ch. 3.
    """
    Gamma_reflected = np.asarray(Gamma_reflected, dtype=float)
    Gamma_reemitted = np.asarray(Gamma_reemitted, dtype=float)
    if np.any(Gamma_reflected < 0.0) or np.any(Gamma_reemitted < 0.0):
        raise ValueError("returned fluxes must be non-negative")
    Gamma_incident = _positive(Gamma_incident, "Gamma_incident")
    return _out((Gamma_reflected + Gamma_reemitted) / Gamma_incident)


def sputtering_threshold_bohdansky(E_s, m_1, m_2):
    r"""Physical-sputtering threshold energy, Bohdansky's empirical fit.

    $$E_\mathrm{th} = \begin{cases}
      \dfrac{E_s}{\gamma(1 - \gamma)}, & m_1/m_2 \le 0.2 \\[1ex]
      8E_s\left(\dfrac{m_1}{m_2}\right)^{2/5}, & m_1/m_2 > 0.2
      \end{cases},\qquad \gamma = \frac{4m_1m_2}{(m_1 + m_2)^2}$$

    Parameters
    ----------
    E_s : float or np.ndarray
        Surface binding energy of the target (usually its sublimation
        energy), supplied by the caller, positive [eV].
    m_1 : float
        Projectile mass, positive [kg or u].
    m_2 : float
        Target-atom mass, positive, same unit [kg or u].

    Returns
    -------
    float or np.ndarray
        Threshold incident energy, normal incidence [eV].

    Raises
    ------
    ValueError
        A mass or ``E_s`` is not positive.

    Convention
    ----------
    A named empirical model (Bohdansky 1984), not a table: $E_s$ is an
    input and no material data are built in. Fits of Eckstein and others
    differ by tens of per cent near threshold; use tabulated thresholds for
    quantitative work.

    Physical interpretation
    -----------------------
    For light projectiles the first collision can give at most $\gamma E$
    (``binary_collision_energy_transfer_factor``), and the recoil must turn
    back out of the surface: hence $E_s/\gamma(1 - \gamma)$, far above $E_s$
    for D on W.

    References
    ----------
    .. [1] J. Bohdansky, J. Roth and H. L. Bay, J. Appl. Phys. 51 (1980) 2861;
           J. Bohdansky, Nucl. Instrum. Methods B 2 (1984) 587.
    """
    E_s = _positive(E_s, "E_s")
    m_1 = float(_positive(m_1, "m_1"))
    m_2 = float(_positive(m_2, "m_2"))
    ratio = m_1 / m_2
    if ratio <= 0.2:
        gamma = 4.0 * m_1 * m_2 / (m_1 + m_2) ** 2
        return _out(E_s / (gamma * (1.0 - gamma)))
    return _out(8.0 * E_s * ratio**0.4)
