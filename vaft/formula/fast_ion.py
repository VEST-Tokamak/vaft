r"""Classical fast-ion slowing down: critical velocity, slowing-down time, distribution, density and pressure.

A light analytic baseline for beam (or other fast) ions in a Maxwellian plasma
(issue #1606): the critical velocity at which electron and ion drag are equal,
the Spitzer slowing-down time on electrons, the time to slow between two
speeds, the steady classical slowing-down distribution of a monoenergetic
source, and the fast-ion density, energy density and scalar pressure it holds.
It is not a replacement for an orbit-following or Fokker--Planck solver
(NUBEAM, #1216): pitch-angle scattering, energy diffusion, orbits and charge
exchange are absent, so the distribution is isotropic by construction.

Notation
--------
v_b       : birth (injection) speed of the fast ions                  [m/s]
v_c       : critical speed, where electron and ion drag are equal     [m/s]
E_c       : critical energy, m_b v_c^2 / 2                             [eV]
tau_s     : Spitzer slowing-down time on electrons                     [s]
S         : fast-ion birth rate per unit volume                        [m^-3 s^-1]
n_f, W_f  : fast-ion density and energy density                        [m^-3], [J m^-3]
p_f       : scalar fast-ion pressure, 2 W_f / 3                        [Pa]
A, Z      : mass number [u] and charge number of a species             [-]

Conventions
-----------
Temperatures in eV, energies of fast ions in eV, every other quantity SI.
The drag law is the classical one: $dv/dt = -(v/\tau_s)(1 + v_c^3/v^3)$,
valid for $v_{th,i} \ll v \ll v_{th,e}$.  Field-ion species enter the
critical speed through $\sum_j n_j Z_j^2/A_j$ only.

References
----------
.. [1] T. H. Stix, Plasma Phys. 14 (1972) 367 (heating by fast ions; the
       critical energy and the slowing-down distribution).
.. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Sec. 5.4 (neutral-beam heating).
"""

from __future__ import annotations

import numpy as np

from .constants import EPS0, ME, QE

__all__ = [
    "critical_velocity_from_T_e_n_species",
    "critical_energy_from_T_e_A_b_n_species",
    "slowing_down_time_from_T_e_n_e_A_b_Z_b",
    "slowing_down_time_between_speeds",
    "slowing_down_distribution",
    "fast_ion_density_from_source",
    "fast_ion_energy_density_from_source",
    "fast_ion_pressure_from_energy_density",
]

#: Atomic mass unit [kg] (CODATA 2018).
_AMU = 1.66053906660e-27


def _out(value):
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def _positive(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def _non_negative(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return array


def _field_ion_sum(n_e, n_j, Z_j, A_j):
    """sum_j n_j Z_j^2 / A_j / n_e, species on the last axis."""
    ne = _positive(n_e, "n_e")
    n = _non_negative(n_j, "n_j")
    z = _positive(Z_j, "Z_j")
    a = _positive(A_j, "A_j")
    n, z, a = np.broadcast_arrays(n, z, a)
    if n.ndim == 0:
        raise ValueError("n_j, Z_j and A_j need a species axis (the last axis)")
    return np.sum(n * z**2 / a, axis=-1) / ne


def critical_velocity_from_T_e_n_species(T_e, n_e, n_j, Z_j, A_j):
    r"""Critical speed at which a fast ion's drag on electrons equals its drag on the field ions.

    $$v_c^3 = \frac{3\sqrt{\pi}}{4}\,\frac{m_e}{m_u}\,
      \Big(\frac{1}{n_e}\sum_j \frac{n_j Z_j^2}{A_j}\Big)\,v_{te}^3,\qquad
      v_{te} = \sqrt{2 e T_e/m_e}$$

    Parameters
    ----------
    T_e : float or array-like
        Electron temperature, positive [eV].
    n_e : float or array-like
        Electron density, positive [m^-3].
    n_j : array-like
        Density of each field-ion species, species on the last axis,
        non-negative [m^-3].
    Z_j : array-like
        Charge of each field-ion species, positive [-].
    A_j : array-like
        Mass number of each field-ion species, positive [-].

    Returns
    -------
    float or np.ndarray
        $v_c$, independent of the fast ion's own mass and charge [m/s].

    Raises
    ------
    ValueError
        Non-finite input, a non-positive temperature, density, charge or mass,
        or no species axis.

    Physical interpretation
    -----------------------
    Above $v_c$ a fast ion loses energy mainly to electrons, below it mainly
    to ions; $E_c = m_b v_c^2/2$ is where a beam's heating switches from
    electrons to ions.

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367, Eq. (9).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    critical_energy_from_T_e_A_b_n_species
    """
    te = _positive(T_e, "T_e")
    field = _field_ion_sum(n_e, n_j, Z_j, A_j)
    v_te = np.sqrt(2.0 * QE * te / ME)
    return _out((0.75 * np.sqrt(np.pi) * (ME / _AMU) * field) ** (1.0 / 3.0) * v_te)


def critical_energy_from_T_e_A_b_n_species(T_e, A_b, n_e, n_j, Z_j, A_j):
    r"""Critical energy of a fast ion of mass number A_b: E_c = m_b v_c^2 / 2.

    $$E_c = \Big(\frac{3\sqrt\pi}{4}\Big)^{2/3}
      \Big(\frac{m_u}{m_e}\Big)^{1/3} A_b\,T_e
      \Big(\frac{1}{n_e}\sum_j\frac{n_jZ_j^2}{A_j}\Big)^{2/3}
      \approx 14.8\,A_b\,T_e\,\Big(\frac{1}{n_e}\sum_j\frac{n_jZ_j^2}{A_j}\Big)^{2/3}$$

    Parameters
    ----------
    T_e : float or array-like
        Electron temperature, positive [eV].
    A_b : float or array-like
        Mass number of the fast ion, positive [-].
    n_e : float or array-like
        Electron density, positive [m^-3].
    n_j : array-like
        Density of each field-ion species, species on the last axis [m^-3].
    Z_j : array-like
        Charge of each field-ion species, positive [-].
    A_j : array-like
        Mass number of each field-ion species, positive [-].

    Returns
    -------
    float or np.ndarray
        $E_c$ [eV].

    Raises
    ------
    ValueError
        As :func:`critical_velocity_from_T_e_n_species`, or a non-positive A_b.

    Physical interpretation
    -----------------------
    For a pure deuterium plasma and a deuterium beam, $E_c \approx 18.7 T_e$:
    a 30 keV beam into 100 eV electrons is far above it and heats electrons.
    Impurities act only through $\sum_j n_jZ_j^2/A_j$ at fixed $n_e$: fully
    stripped carbon or oxygen in deuterium leave it unchanged ($Z^2/A = Z/2$,
    as for D), while in hydrogen, or partly ionised, they lower $E_c$.

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    critical_velocity_from_T_e_n_species
    """
    a_b = _positive(A_b, "A_b")
    v_c = np.asarray(critical_velocity_from_T_e_n_species(T_e, n_e, n_j, Z_j, A_j))
    return _out(0.5 * a_b * _AMU * v_c**2 / QE)


def slowing_down_time_from_T_e_n_e_A_b_Z_b(T_e, n_e, A_b, Z_b, ln_Lambda):
    r"""Spitzer slowing-down time of a fast ion on electrons.

    $$\tau_s = \frac{3\,(2\pi)^{3/2}\,\varepsilon_0^2\,m_b\,(eT_e)^{3/2}}
      {n_e\,Z_b^2\,e^4\,m_e^{1/2}\,\ln\Lambda}
      \approx 6.27\times10^{14}\,\frac{A_b\,T_e^{3/2}}{Z_b^2\,n_e\,\ln\Lambda}$$

    with $T_e$ in eV and $n_e$ in m^-3 in the numeric form.

    Parameters
    ----------
    T_e : float or array-like
        Electron temperature, positive [eV].
    n_e : float or array-like
        Electron density, positive [m^-3].
    A_b : float or array-like
        Mass number of the fast ion, positive [-].
    Z_b : float or array-like
        Charge of the fast ion, positive [-].
    ln_Lambda : float or array-like
        Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_s$ [s].

    Raises
    ------
    ValueError
        Non-finite or non-positive input.

    Convention
    ----------
    The e-folding time of the fast ion's *momentum* against electron drag
    alone, $dv/dt = -v/\tau_s$ for $v \gg v_c$; the energy e-folds in
    $\tau_s/2$.  The Coulomb logarithm is the caller's: pass the electron one
    (:func:`vaft.formula.equilibrium.coulomb_logarithm_from_n_T`).

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 2.15 and 5.4.

    See Also
    --------
    slowing_down_time_between_speeds
    """
    te = _positive(T_e, "T_e") * QE
    ne = _positive(n_e, "n_e")
    m_b = _positive(A_b, "A_b") * _AMU
    z_b = _positive(Z_b, "Z_b")
    lnl = _positive(ln_Lambda, "ln_Lambda")
    return _out(3.0 * (2.0 * np.pi) ** 1.5 * EPS0**2 * m_b * te**1.5
                / (ne * z_b**2 * QE**4 * np.sqrt(ME) * lnl))


def slowing_down_time_between_speeds(tau_s, v_c, v_from, v_to=0.0):
    r"""Time a fast ion takes to slow from one speed to a lower one under classical drag.

    $$t = \frac{\tau_s}{3}\,\ln\frac{v_{from}^3 + v_c^3}{v_{to}^3 + v_c^3}$$

    Parameters
    ----------
    tau_s : float or array-like
        Spitzer slowing-down time, positive [s].
    v_c : float or array-like
        Critical speed, positive [m/s].
    v_from : float or array-like
        Starting speed, e.g. the birth speed, non-negative [m/s].
    v_to : float or array-like, optional
        Final speed, at most ``v_from``; default 0, the full slowing-down
        (thermalisation) time [m/s].

    Returns
    -------
    float or np.ndarray
        Elapsed time [s].

    Raises
    ------
    ValueError
        Non-finite input, non-positive tau_s or v_c, or v_to > v_from.

    Validity
    --------
    Classical drag only; the last stretch below the ion thermal speed is not
    drag-dominated, so the time to "zero" is the time to merge with the
    thermal ions to within that approximation.

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    slowing_down_time_from_T_e_n_e_A_b_Z_b
    """
    tau = _positive(tau_s, "tau_s")
    vc = _positive(v_c, "v_c")
    v0 = _non_negative(v_from, "v_from")
    v1 = _non_negative(v_to, "v_to")
    if np.any(v1 > v0):
        raise ValueError("v_to must not exceed v_from")
    return _out(tau / 3.0 * np.log((v0**3 + vc**3) / (v1**3 + vc**3)))


def slowing_down_distribution(v, S, tau_s, v_b, v_c):
    r"""Steady isotropic classical slowing-down distribution of a monoenergetic source.

    $$f(v) = \frac{S\,\tau_s}{4\pi}\,\frac{H(v_b - v)}{v^3 + v_c^3}$$

    normalised so that $\int f\,d^3v = n_f$.

    Parameters
    ----------
    v : float or array-like
        Speed, non-negative [m/s].
    S : float or array-like
        Birth rate per unit volume, non-negative [m^-3 s^-1].
    tau_s : float or array-like
        Spitzer slowing-down time, positive [s].
    v_b : float or array-like
        Birth speed, positive [m/s].
    v_c : float or array-like
        Critical speed, positive [m/s].

    Returns
    -------
    float or np.ndarray
        $f(v)$ [s^3 m^-6].

    Raises
    ------
    ValueError
        Non-finite input, a negative speed or source, or a non-positive
        tau_s, v_b or v_c.

    Assumptions
    -----------
    Steady state, a single birth speed, classical drag; no pitch-angle
    scattering is needed for the scalar moments because the distribution is
    taken isotropic.

    Limitations
    -----------
    Isotropic by construction: a real beam is not.  Its anisotropic pressure
    split $p_f = (p_\parallel + 2 p_\perp)/3$ is a later extension.

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    fast_ion_density_from_source
    """
    speed = _non_negative(v, "v")
    source = _non_negative(S, "S")
    tau = _positive(tau_s, "tau_s")
    vb = _positive(v_b, "v_b")
    vc = _positive(v_c, "v_c")
    return _out(np.where(speed <= vb, source * tau / (4.0 * np.pi * (speed**3 + vc**3)), 0.0))


def fast_ion_density_from_source(S, tau_s, v_b, v_c):
    r"""Fast-ion density held by a steady source slowing down classically.

    $$n_f = \frac{S\,\tau_s}{3}\,\ln\!\Big(1 + \frac{v_b^3}{v_c^3}\Big)$$

    Parameters
    ----------
    S : float or array-like
        Birth rate per unit volume, non-negative [m^-3 s^-1].
    tau_s : float or array-like
        Spitzer slowing-down time, positive [s].
    v_b : float or array-like
        Birth speed, positive [m/s].
    v_c : float or array-like
        Critical speed, positive [m/s].

    Returns
    -------
    float or np.ndarray
        $n_f$ [m^-3].

    Raises
    ------
    ValueError
        As :func:`slowing_down_distribution`.

    Physical interpretation
    -----------------------
    The source rate times the time to slow to rest: $n_f = S\,t(v_b \to 0)$.

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    fast_ion_energy_density_from_source
    """
    source = _non_negative(S, "S")
    tau = _positive(tau_s, "tau_s")
    x3 = (_positive(v_b, "v_b") / _positive(v_c, "v_c")) ** 3
    return _out(source * tau / 3.0 * np.log1p(x3))


def _velocity_moment_4(x):
    """int_0^x u^4/(u^3+1) du: closed form, or its series where the closed form cancels."""
    x = np.asarray(x, dtype=float)
    small = x < 0.05
    xs = np.where(small, x, 0.0)
    series = xs**5 / 5.0 - xs**8 / 8.0 + xs**11 / 11.0 - xs**14 / 14.0
    xl = np.where(small, 1.0, x)
    h = (np.log((xl**2 - xl + 1.0) / (xl + 1.0) ** 2) / 6.0
         + (np.arctan((2.0 * xl - 1.0) / np.sqrt(3.0)) + np.pi / 6.0) / np.sqrt(3.0))
    return np.where(small, series, xl**2 / 2.0 - h)


def fast_ion_energy_density_from_source(S, tau_s, A_b, v_b, v_c):
    r"""Kinetic-energy density held by a steady source slowing down classically.

    $$W_f = \frac{m_b}{2}\int_0^{v_b} v^2 f\,4\pi v^2\,dv
      = \frac{S\,\tau_s\,m_b\,v_c^2}{2}\,G\!\Big(\frac{v_b}{v_c}\Big),\qquad
      G(x) = \int_0^x \frac{u^4}{u^3+1}\,du$$

    with $G(x) = x^2/2 - \frac16\ln\frac{x^2-x+1}{(x+1)^2}
    - \frac{1}{\sqrt3}\big[\arctan\frac{2x-1}{\sqrt3} + \frac\pi6\big]$.

    Parameters
    ----------
    S : float or array-like
        Birth rate per unit volume, non-negative [m^-3 s^-1].
    tau_s : float or array-like
        Spitzer slowing-down time, positive [s].
    A_b : float or array-like
        Mass number of the fast ion, positive [-].
    v_b : float or array-like
        Birth speed, positive [m/s].
    v_c : float or array-like
        Critical speed, positive [m/s].

    Returns
    -------
    float or np.ndarray
        $W_f$ [J m^-3].

    Raises
    ------
    ValueError
        As :func:`slowing_down_distribution`, or a non-positive A_b.

    Physical interpretation
    -----------------------
    For $v_b \gg v_c$, $W_f \to S\,\tau_s\,E_b/2$: the source power times
    the energy e-folding time $\tau_s/2$.

    References
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    fast_ion_pressure_from_energy_density
    """
    source = _non_negative(S, "S")
    tau = _positive(tau_s, "tau_s")
    m_b = _positive(A_b, "A_b") * _AMU
    vc = _positive(v_c, "v_c")
    x = _positive(v_b, "v_b") / vc
    return _out(0.5 * source * tau * m_b * vc**2 * _velocity_moment_4(x))


def fast_ion_pressure_from_energy_density(W_f):
    r"""Scalar fast-ion pressure of an isotropic fast population: p_f = 2 W_f / 3.

    $$p_f = \tfrac{2}{3}W_f$$

    Parameters
    ----------
    W_f : float or array-like
        Fast-ion kinetic-energy density, non-negative [J m^-3].

    Returns
    -------
    float or np.ndarray
        $p_f$ [Pa].

    Raises
    ------
    ValueError
        A non-finite or negative energy density.

    Convention
    ----------
    Isotropic: $p_\parallel = p_\perp = p_f$.  An anisotropic population has
    $p_f = (p_\parallel + 2p_\perp)/3$ with $W_f = (p_\parallel + 2p_\perp)/2$,
    which this scalar form keeps; the split itself is not represented.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.4.

    See Also
    --------
    fast_ion_energy_density_from_source
    """
    return _out(2.0 / 3.0 * _non_negative(W_f, "W_f"))
