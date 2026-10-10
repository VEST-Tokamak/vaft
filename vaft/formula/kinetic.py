r"""Collisional closures of a multi-species plasma: collision times, energy exchange and classical diffusivities.

The thermal half of the analytic kinetic closure (issue #1606).  The
composition it needs -- species densities from Z_eff and an impurity model,
main-ion dilution -- is :mod:`vaft.formula.impurity` (#1565); the per-species
thermal pressure is :func:`vaft.formula.equilibrium.ion_pressure`; the
Coulomb logarithm is the caller's
(:func:`vaft.formula.equilibrium.coulomb_logarithm_from_n_T`).  This module
adds the collision times a thermal model of several ion species needs, and the
Braginskii perpendicular heat diffusivities built on them (the classical
transport baseline, #1435/#1899).

Notation
--------
T_e       : electron temperature                                    [eV]
n_j, Z_j  : density [m^-3] and charge of field-ion species j        [-]
A_j       : mass number of species j                                [-]
tau_e     : Braginskii electron collision time                      [s]
tau_i     : Braginskii ion collision time                           [s]
tau_eq    : electron-ion energy-exchange (equilibration) time       [s]
B         : magnetic-field magnitude                                [T]
chi_perp  : perpendicular heat diffusivity, q = n chi (-dT/dr)      [m^2/s]

Conventions
-----------
Species on the last axis; temperatures in eV; everything else SI.  The
electron-ion collision rate sums $n_j Z_j^2$ over the ions, which is
$Z_\mathrm{eff} n_e$ for a quasi-neutral plasma.

References
----------
.. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205.
.. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Sec. 2.15.
"""

from __future__ import annotations

import numpy as np

from .constants import EPS0, ME, QE

__all__ = [
    "braginskii_gamma1_perp_from_Z",
    "classical_electron_heat_diffusivity_from_T_e_B",
    "classical_ion_heat_diffusivity_from_T_i_B",
    "electron_collision_time_from_T_e_n_species",
    "electron_ion_energy_exchange_time_from_T_e_n_species",
    "ion_collision_time_from_T_i_n_species",
    "ion_ion_coulomb_logarithm_from_n_T",
]

#: Braginskii's gamma_1' (electron perpendicular heat conductivity) against Z,
#: Braginskii 1965, Table 2: 4.66, 4.0, 3.7, 3.6, 3.25 for Z = 1, 2, 3, 4, inf.  The
#: Z = inf knot sits at 1e9, so a linear lookup in Z is held at 3.6 for Z > 4.
_GAMMA1_PERP = ((1.0, 4.66), (2.0, 4.0), (3.0, 3.7), (4.0, 3.6), (1e9, 3.25))

#: Atomic mass unit [kg] (CODATA 2018).
_AMU = 1.66053906660e-27


def _out(value):
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def _species(n_j, Z_j, A_j=None):
    n = np.asarray(n_j, dtype=float)
    z = np.asarray(Z_j, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n < 0.0):
        raise ValueError("n_j must be finite and non-negative")
    if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("Z_j must be finite and positive")
    arrays = [n, z]
    if A_j is not None:
        a = np.asarray(A_j, dtype=float)
        if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError("A_j must be finite and positive")
        arrays.append(a)
    arrays = np.broadcast_arrays(*arrays)
    if arrays[0].ndim == 0:
        raise ValueError("species need a species axis (the last axis)")
    return arrays


def _positive(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def electron_collision_time_from_T_e_n_species(T_e, n_j, Z_j, ln_Lambda):
    r"""Braginskii electron collision time against a mixture of ion species.

    $$\tau_e = \frac{6\sqrt2\,\pi^{3/2}\,\varepsilon_0^2\,m_e^{1/2}\,(eT_e)^{3/2}}
      {e^4\,\ln\Lambda\,\sum_j n_j Z_j^2}
      \approx 1.09\times10^{16}\,\frac{T_e[\mathrm{keV}]^{3/2}}{\ln\Lambda\,\sum_j n_jZ_j^2}$$

    Parameters
    ----------
    T_e : float or array-like
        Electron temperature, positive [eV].
    n_j : array-like
        Density of each ion species, species on the last axis, non-negative [m^-3].
    Z_j : array-like
        Charge of each ion species, positive [-].
    ln_Lambda : float or array-like
        Electron Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_e$ [s].

    Raises
    ------
    ValueError
        Non-finite input, non-positive T_e, charges or ln_Lambda, no species
        axis, or a mixture of zero total $\sum n_j Z_j^2$.

    Convention
    ----------
    Braginskii's $\tau_e$, the time in his transport coefficients (and in the
    Spitzer resistivity $\eta = m_e/(n_e e^2\tau_e)\cdot 0.51$ for $Z=1$); the
    ion sum makes it the multi-species form, $\sum_j n_jZ_j^2 = Z_\mathrm{eff}n_e$.

    References
    ----------
    .. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 2.15.

    See Also
    --------
    electron_ion_energy_exchange_time_from_T_e_n_species
    """
    te = _positive(T_e, "T_e") * QE
    lnl = _positive(ln_Lambda, "ln_Lambda")
    n, z = _species(n_j, Z_j)
    nz2 = np.sum(n * z**2, axis=-1)
    if np.any(nz2 <= 0.0):
        raise ValueError("the ion mixture has zero sum n_j Z_j^2")
    return _out(6.0 * np.sqrt(2.0) * np.pi**1.5 * EPS0**2 * np.sqrt(ME) * te**1.5 / (QE**4 * lnl * nz2))


def electron_ion_energy_exchange_time_from_T_e_n_species(T_e, n_j, Z_j, A_j, ln_Lambda):
    r"""Time in which electrons and a mixture of ion species exchange thermal energy.

    $$\frac{dT_e}{dt}\Big|_{ei} = -\frac{T_e - T_i}{\tau_{eq}},\qquad
      \frac{1}{\tau_{eq}} = \sum_j \frac{2\,m_e}{A_j m_u}\,\frac{1}{\tau_{e,j}},
      \qquad \tau_{e,j} = \tau_e\big|_{\text{species } j \text{ alone}}$$

    Parameters
    ----------
    T_e : float or array-like
        Electron temperature, positive [eV].
    n_j : array-like
        Density of each ion species, species on the last axis, non-negative [m^-3].
    Z_j : array-like
        Charge of each ion species, positive [-].
    A_j : array-like
        Mass number of each ion species, positive [-].
    ln_Lambda : float or array-like
        Electron Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_{eq}$, the electron energy e-folding time against all ions [s].

    Raises
    ------
    ValueError
        As :func:`electron_collision_time_from_T_e_n_species`, or a
        non-positive mass number.

    Assumptions
    -----------
    $T_i/m_i \ll T_e/m_e$ (the electron thermal speed dominates the relative
    speed), so the rate is evaluated at $T_e$ alone; each species exchanges
    in proportion to $n_jZ_j^2/A_j$.

    Physical interpretation
    -----------------------
    Heavy impurities raise $Z_\mathrm{eff}$ (shortening $\tau_e$) but carry
    $1/A_j$, so at fixed $Z_\mathrm{eff}$ a carbon/oxygen mixture exchanges
    less energy with electrons per unit $\sum n_jZ_j^2$ than hydrogen does.

    References
    ----------
    .. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 2.15.

    See Also
    --------
    electron_collision_time_from_T_e_n_species
    """
    te = _positive(T_e, "T_e") * QE
    lnl = _positive(ln_Lambda, "ln_Lambda")
    n, z, a = _species(n_j, Z_j, A_j)
    rate_sum = np.sum(n * z**2 / a, axis=-1)
    if np.any(rate_sum <= 0.0):
        raise ValueError("the ion mixture has zero sum n_j Z_j^2 / A_j")
    tau_unit = 6.0 * np.sqrt(2.0) * np.pi**1.5 * EPS0**2 * np.sqrt(ME) * te**1.5 / (QE**4 * lnl)
    return _out(tau_unit / (2.0 * ME / _AMU * rate_sum))


def ion_collision_time_from_T_i_n_species(T_i, A_i, Z_i, n_j, Z_j, ln_Lambda):
    r"""Braginskii collision time of one ion species against a mixture of field ions.

    $$\tau_i = \frac{12\,\pi^{3/2}\,\varepsilon_0^2\,m_i^{1/2}\,(eT_i)^{3/2}}
      {e^4\,\ln\Lambda\,Z_i^2\sum_j n_j Z_j^2}$$

    Parameters
    ----------
    T_i : float or array-like
        Temperature of the test ion species, positive [eV].
    A_i : float or array-like
        Its mass in atomic mass units, positive [-].
    Z_i : float or array-like
        Its charge, positive [-].
    n_j : array-like
        Density of each field-ion species, species on the last axis, non-negative [m^-3].
    Z_j : array-like
        Charge of each field-ion species, positive [-].
    ln_Lambda : float or array-like
        Ion-ion Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_i$ [s].

    Raises
    ------
    ValueError
        Non-finite input, non-positive T_i, A_i, Z_i, charges or ln_Lambda, no
        species axis, or a mixture of zero total $\sum n_j Z_j^2$.

    Convention
    ----------
    Braginskii's $\tau_i$, with $\tau_i/\tau_e = \sqrt{2 m_i/m_e}\,(T_i/T_e)^{3/2}$ at
    equal density and $Z = 1$ (the 12 against $6\sqrt2$ of
    :func:`electron_collision_time_from_T_e_n_species`). Unlike field ions enter in
    the like-particle form, $Z_i^2\sum_j n_jZ_j^2$; for one species this is
    $Z^4 n_i$, the NRL form, whose rounded $2.09\times10^{13}$ (m$^{-3}$, eV,
    $\mu = m_i/m_p$) :func:`vaft.formula.ordering.braginskii_ion_collision_time`
    uses; the exact coefficient here is $2.085\times10^{13}$, 0.24 % lower.

    References
    ----------
    .. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205.
    .. [2] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).

    See Also
    --------
    electron_collision_time_from_T_e_n_species
    """
    ti = _positive(T_i, "T_i") * QE
    mass = _positive(A_i, "A_i") * _AMU
    zi = _positive(Z_i, "Z_i")
    lnl = _positive(ln_Lambda, "ln_Lambda")
    n, z = _species(n_j, Z_j)
    nz2 = np.sum(n * z**2, axis=-1)
    if np.any(nz2 <= 0.0):
        raise ValueError("the ion mixture has zero sum n_j Z_j^2")
    return _out(12.0 * np.pi**1.5 * EPS0**2 * np.sqrt(mass) * ti**1.5 / (QE**4 * lnl * zi**2 * nz2))


def ion_ion_coulomb_logarithm_from_n_T(n_i, T_i, Z_i):
    r"""Coulomb logarithm for collisions between ions of one species (NRL).

    $$\ln\Lambda_{ii} = 23 - \ln\!\left[\frac{Z_i^2}{T_i}\left(\frac{2\,n_i Z_i^2}{T_i}\right)^{1/2}\right]$$

    Parameters
    ----------
    n_i : float or array-like
        Ion density, positive [m^-3].
    T_i : float or array-like
        Ion temperature, positive [eV].
    Z_i : float or array-like
        Ion charge, positive [-].

    Returns
    -------
    float or np.ndarray
        $\ln\Lambda_{ii}$ [-].

    Raises
    ------
    ValueError
        A non-finite or non-positive input.

    Convention
    ----------
    The NRL formulary's like-species ion-ion form, with $n_i$ in cm$^{-3}$ inside the
    logarithm (converted here from m$^{-3}$) and $T_i$ in eV.

    References
    ----------
    .. [1] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019), p. 34.
    """
    n = _positive(n_i, "n_i") * 1e-6
    t = _positive(T_i, "T_i")
    z = _positive(Z_i, "Z_i")
    return _out(23.0 - np.log(z**2 / t * np.sqrt(2.0 * n * z**2 / t)))


def braginskii_gamma1_perp_from_Z(Z):
    r"""Braginskii's electron perpendicular heat-conductivity coefficient $\gamma_1'(Z)$.

    $$\kappa_{\perp e} = \gamma_1'(Z)\,\frac{n_e T_e}{m_e\Omega_e^2\tau_e}$$

    Parameters
    ----------
    Z : float or array-like
        Effective ion charge, positive [-].

    Returns
    -------
    float or np.ndarray
        $\gamma_1'$ [-].

    Raises
    ------
    ValueError
        A non-finite or non-positive Z.

    Convention
    ----------
    Braginskii's Table 2: 4.66, 4.0, 3.7, 3.6 and 3.25 at Z = 1, 2, 3, 4 and infinity,
    linear in Z between the tabulated charges. Above Z = 4 it is held at 3.6 (the
    infinite-Z asymptote cannot be reached by a linear lookup); below Z = 1 at 4.66.

    References
    ----------
    .. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205, Table 2.
    """
    z = _positive(Z, "Z")
    # anti-alias: not a time series and not a downsample; a lookup in Braginskii's
    # gamma_1'(Z) table.
    return _out(np.interp(z, *zip(*_GAMMA1_PERP)))


def classical_electron_heat_diffusivity_from_T_e_B(T_e, B, tau_e, Z_eff):
    r"""Braginskii perpendicular electron heat diffusivity, strongly magnetised limit.

    $$\chi_{\perp e} = \gamma_1'(Z_\mathrm{eff})\,\frac{T_e}{m_e\,\Omega_e^2\,\tau_e},\qquad
      \Omega_e = \frac{eB}{m_e}$$

    Parameters
    ----------
    T_e : float or array-like
        Electron temperature, positive [eV].
    B : float or array-like
        Magnetic-field magnitude, positive [T].
    tau_e : float or array-like
        Electron collision time, e.g.
        :func:`electron_collision_time_from_T_e_n_species`, positive [s].
    Z_eff : float or array-like
        Effective charge for :func:`braginskii_gamma1_perp_from_Z`, positive [-].

    Returns
    -------
    float or np.ndarray
        $\chi_{\perp e}$, with $q_{\perp e} = n_e\chi_{\perp e}(-\mathrm{d}T_e/\mathrm{d}r)$ [m^2/s].

    Raises
    ------
    ValueError
        A non-finite or non-positive input.

    Convention
    ----------
    The conductive heat flux only: no thermal-force or convective terms. It is a
    physical diffusivity of the stated model, not the order-of-magnitude
    $\nu\rho^2$ reference scale.

    References
    ----------
    .. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205.
    """
    t = _positive(T_e, "T_e") * QE
    omega = QE * _positive(B, "B") / ME
    return _out(braginskii_gamma1_perp_from_Z(Z_eff) * t / (ME * omega**2 * _positive(tau_e, "tau_e")))


def classical_ion_heat_diffusivity_from_T_i_B(T_i, B, tau_i, A_i, Z_i):
    r"""Braginskii perpendicular ion heat diffusivity, strongly magnetised limit.

    $$\chi_{\perp i} = 2\,\frac{T_i}{m_i\,\Omega_i^2\,\tau_i},\qquad \Omega_i = \frac{Z_ieB}{m_i}$$

    Parameters
    ----------
    T_i : float or array-like
        Ion temperature, positive [eV].
    B : float or array-like
        Magnetic-field magnitude, positive [T].
    tau_i : float or array-like
        Ion collision time, e.g. :func:`ion_collision_time_from_T_i_n_species`,
        positive [s].
    A_i : float or array-like
        Ion mass in atomic mass units, positive [-].
    Z_i : float or array-like
        Ion charge, positive [-].

    Returns
    -------
    float or np.ndarray
        $\chi_{\perp i}$, with $q_{\perp i} = n_i\chi_{\perp i}(-\mathrm{d}T_i/\mathrm{d}r)$ [m^2/s].

    Raises
    ------
    ValueError
        A non-finite or non-positive input.

    Convention
    ----------
    Braginskii's coefficient 2 for one ion species; conductive flux only.

    References
    ----------
    .. [1] S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205.
    """
    t = _positive(T_i, "T_i") * QE
    mass = _positive(A_i, "A_i") * _AMU
    omega = _positive(Z_i, "Z_i") * QE * _positive(B, "B") / mass
    return _out(2.0 * t / (mass * omega**2 * _positive(tau_i, "tau_i")))
