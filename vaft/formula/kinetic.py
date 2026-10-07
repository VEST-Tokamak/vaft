r"""Collisional closures of a multi-species plasma: electron collision time and electron-ion energy exchange.

The thermal half of the analytic kinetic closure (issue #1606).  The
composition it needs -- species densities from Z_eff and an impurity model,
main-ion dilution -- is :mod:`vaft.formula.impurity` (#1565); the per-species
thermal pressure is :func:`vaft.formula.equilibrium.ion_pressure`; the
Coulomb logarithm is the caller's
(:func:`vaft.formula.equilibrium.coulomb_logarithm_from_n_T`).  This module
adds the two collision times a thermal model of several ion species needs.

Notation
--------
T_e       : electron temperature                                    [eV]
n_j, Z_j  : density [m^-3] and charge of field-ion species j        [-]
A_j       : mass number of species j                                [-]
tau_e     : Braginskii electron collision time                      [s]
tau_eq    : electron-ion energy-exchange (equilibration) time       [s]

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
    "electron_collision_time_from_T_e_n_species",
    "electron_ion_energy_exchange_time_from_T_e_n_species",
]

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
