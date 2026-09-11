r"""Analytic neoclassical transport formulas: Sauter and Redl.

Bootstrap current and parallel conductivity from fitted analytic models, as an
independent reference for the drift-kinetic solvers driven through
:mod:`vaft.code.gacode`.  Two formulations are provided as distinct functions
rather than as one backend with a switch, because they are different physics
models and their disagreement is a result rather than an error: the 1999 Sauter
fit was built for conventional aspect ratio, while the 2021 Redl refit extends
to the trapped fractions a spherical tokamak actually reaches.

Everything here is array/scalar numerics on physical quantities.  Nothing reads
an ODS, and nothing knows a flux-surface geometry: the trapped fraction, the
flux-surface-averaged pressure gradients and $I(\psi)$ are inputs, supplied by
the schema-facing layer.

The parallel current a caller usually wants is the sum of two terms,

$$\langle j_\parallel B\rangle = \sigma_{\mathrm{neo}}\langle E_\parallel B\rangle
  + \langle j_\parallel B\rangle_{\mathrm{bs}}$$

whose pieces are :func:`sauter_neoclassical_conductivity` (or its Redl
counterpart) and :func:`sauter_bootstrap_current` (or its Redl counterpart).
They are kept apart because an inductive electric field is a state a caller may
or may not have.

Notation
--------
n_e : electron density  [m^-3]
n_i : ion density  [m^-3]
T_e : electron temperature  [eV]
T_i : ion temperature  [eV]
f_t : fraction of trapped particles on the surface  [-]
nu_e_star : Sauter electron collisionality, Eq. (18b)  [-]
nu_i_star : Sauter ion collisionality, Eq. (18c)  [-]
Z_eff : effective ion charge  [-]
epsilon : inverse aspect ratio r/R_0  [-]
q : safety factor  [-]
psi : poloidal flux per radian  [Wb/rad]
I_psi : the flux function R B_phi  [T m]
L31, L32, L34 : Sauter transport coefficients  [-]
alpha : Sauter ion-temperature-gradient coefficient  [-]

Conventions
-----------
Temperatures are in electronvolts throughout, matching
:func:`vaft.formula.equilibrium.coulomb_logarithm_from_n_T` and
:func:`vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda`,
not in joules or keV.

Poloidal flux is **per radian**.  An ODS stores ``equilibrium`` psi in full
weber per the IMAS data dictionary, so a caller reading psi from an ODS must
divide by $2\pi$ first; see
:func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor`.

References
----------
.. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834.
.. [2] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 9 (2002) 5140
       (erratum).
.. [3] A. Redl, C. Angioni, E. Belli and O. Sauter, Phys. Plasmas 28 (2021)
       022502.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import numpy as np

from .constants import COLLISIONALITY_COEF

__all__ = [
    "BootstrapCoefficients",
    "coulomb_logarithm_electron_sauter",
    "coulomb_logarithm_ion_sauter",
    "electron_collisionality_sauter",
    "ion_collisionality_sauter",
    "redl_bootstrap_coefficients",
    "redl_bootstrap_current",
    "redl_neoclassical_conductivity",
    "sauter_bootstrap_coefficients",
    "sauter_bootstrap_current",
    "sauter_neoclassical_conductivity",
    "sauter_spitzer_conductivity",
    "trapped_particle_fraction",
]

Numeric = Union[float, np.ndarray]

#: Sauter Eq. (18c) ion-collisionality prefactor, the counterpart of
#: ``COLLISIONALITY_COEF`` for the electron expression.
_ION_COLLISIONALITY_COEF = 4.90e-18

#: Sauter Eq. (12) Spitzer-conductivity prefactor, for T_e in eV [S m^-1 eV^-3/2].
_SPITZER_CONDUCTIVITY_COEF = 1.9012e4


class BootstrapCoefficients(NamedTuple):
    """The four dimensionless coefficients of the Sauter bootstrap expression.

    Both :func:`sauter_bootstrap_coefficients` and
    :func:`redl_bootstrap_coefficients` return this shape, because the two
    models differ only in the fits, not in how the coefficients enter the
    current.  Each field is an array when any input was an array.
    """

    L31: Numeric
    L32: Numeric
    L34: Numeric
    alpha: Numeric


def _maybe_scalar(value: Numeric) -> Numeric:
    """Return Python float for 0-d arrays, otherwise return NumPy array."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


def _validate_positive(name: str, value: Numeric) -> np.ndarray:
    """Validate finite positive scalar/array input and return as float array."""
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite. Got {value!r}")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be > 0. Got {value!r}")
    return array


def _validate_non_negative(name: str, value: Numeric) -> np.ndarray:
    """Validate finite non-negative scalar/array input and return as float array."""
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite. Got {value!r}")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be >= 0. Got {value!r}")
    return array


def _validate_fraction(name: str, value: Numeric) -> np.ndarray:
    """Validate a finite value in [0, 1] and return as float array."""
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite. Got {value!r}")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]. Got {value!r}")
    return array


def _l31_polynomial(x: np.ndarray, Z_eff: np.ndarray) -> np.ndarray:
    """Sauter Eq. (14) quartic in X31.

    Shared with Sauter's L34, which is the same polynomial evaluated at X34.
    Redl refits both the polynomial and the effective trapped fraction, so it
    does not use this.
    """
    return (
        (1.0 + 1.4 / (Z_eff + 1.0)) * x
        - (1.9 / (Z_eff + 1.0)) * x**2
        + (0.3 / (Z_eff + 1.0)) * x**3
        + (0.2 / (Z_eff + 1.0)) * x**4
    )


def trapped_particle_fraction(epsilon: Numeric) -> Numeric:
    r"""Trapped-particle fraction $f_t$ of a circular surface of inverse aspect ratio.

    $$f_t = 1 - \frac{(1-\epsilon)^2}{\sqrt{1-\epsilon^2}\,(1 + 1.46\sqrt{\epsilon})}$$

    Parameters
    ----------
    epsilon : float or np.ndarray
        Inverse aspect ratio $r/R_0$ of the surface, in [0, 1) [-].

    Returns
    -------
    float or np.ndarray
        Fraction of particles trapped on the surface [-].

    Raises
    ------
    ValueError
        For non-finite input, or input outside [0, 1).

    Convention
    ----------
    $f_t$ is the flux-surface quantity that enters every coefficient in this
    module.  This function is the *circular* approximation to it, written in
    terms of the inverse aspect ratio alone.  A shaped equilibrium's trapped
    fraction is an integral over the field-strength distribution on the surface
    and differs from this at the tens-of-percent level at strong elongation, so
    a caller holding a real equilibrium should compute $f_t$ from it and pass
    that instead of calling this.  NEO writes its own value to
    ``out.neo.diagnostic_geo`` as ``f_trap``.

    Physical interpretation
    -----------------------
    The share of the local Maxwellian whose parallel energy is too small to
    cross the magnetic well on the outboard side.  It rises steeply with
    $\epsilon$, which is why bootstrap current matters far more in a spherical
    tokamak than in a conventional one.

    Validity
    --------
    Empirical fit.  Concentric circular surfaces; the coefficient 1.46 is a fit
    to the exact integral rather than a derived value.

    Limitations
    -----------
    At $\epsilon \to 1$ the expression tends to 1 but the underlying expansion
    has long since stopped being controlled.  VEST reaches
    $\epsilon \approx 0.6$, where this returns about 0.91: nearly every
    particle counted as trapped, which is precisely where treating a fitted
    neoclassical coefficient as reliable stops being safe.

    References
    ----------
    .. [1] Y. R. Lin-Liu and R. L. Miller, Phys. Plasmas 2 (1995) 1666.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.9 (trapped particles).
    """
    array = np.asarray(epsilon, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"epsilon must be finite. Got {epsilon!r}")
    if np.any(array < 0.0) or np.any(array >= 1.0):
        raise ValueError(f"epsilon must lie in [0, 1). Got {epsilon!r}")
    numerator = (1.0 - array) ** 2
    denominator = np.sqrt(1.0 - array**2) * (1.0 + 1.46 * np.sqrt(array))
    return _maybe_scalar(1.0 - numerator / denominator)


def coulomb_logarithm_electron_sauter(n_e: Numeric, T_e: Numeric) -> Numeric:
    r"""Electron Coulomb logarithm $\ln\Lambda_e$ in the Sauter convention.

    $$\ln\Lambda_e = 31.3 - \ln\!\left(\frac{\sqrt{n_e}}{T_e}\right)$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density, strictly positive [m^-3].
    T_e : float or np.ndarray
        Electron temperature, strictly positive [eV].

    Returns
    -------
    float or np.ndarray
        Electron Coulomb logarithm [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    This is Sauter Eq. (18d), and it is **not** the NRL expression already in
    this package: :func:`vaft.formula.equilibrium.coulomb_logarithm_from_n_T`
    uses 30.9, this uses 31.3.  The difference is about 1.3 percent of a typical
    value, but the collisionality coefficients here were fitted with this one,
    so mixing them biases $\nu_e^*$ consistently.  Use this function wherever a
    Sauter or Redl coefficient is downstream.

    Validity
    --------
    Thermal electrons well above the ionisation stage; the same
    $T_e \gtrsim 10$ eV floor as the NRL form.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (18d).

    See Also
    --------
    vaft.formula.equilibrium.coulomb_logarithm_from_n_T : the NRL convention.
    """
    n_array = _validate_positive("n_e", n_e)
    t_array = _validate_positive("T_e", T_e)
    return _maybe_scalar(31.3 - np.log(np.sqrt(n_array) / t_array))


def coulomb_logarithm_ion_sauter(n_i: Numeric, T_i: Numeric, Z: Numeric) -> Numeric:
    r"""Ion-ion Coulomb logarithm $\ln\Lambda_{ii}$ in the Sauter convention.

    $$\ln\Lambda_{ii} = 30.0 - \ln\!\left(\frac{Z^3\sqrt{n_i}}{T_i^{3/2}}\right)$$

    Parameters
    ----------
    n_i : float or np.ndarray
        Ion density, strictly positive [m^-3].
    T_i : float or np.ndarray
        Ion temperature, strictly positive [eV].
    Z : float or np.ndarray
        Ion charge number, strictly positive [-].

    Returns
    -------
    float or np.ndarray
        Ion-ion Coulomb logarithm [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    Sauter Eq. (18e).  The charge enters cubed, so an impurity species changes
    this substantially more than it changes $\ln\Lambda_e$.

    Validity
    --------
    Thermal ions of a single charge state.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (18e).
    """
    n_array = _validate_positive("n_i", n_i)
    t_array = _validate_positive("T_i", T_i)
    z_array = _validate_positive("Z", Z)
    return _maybe_scalar(30.0 - np.log(z_array**3 * np.sqrt(n_array) / t_array**1.5))


def electron_collisionality_sauter(
    n_e: Numeric,
    T_e: Numeric,
    q: Numeric,
    R: Numeric,
    epsilon: Numeric,
    Z_eff: Numeric,
    ln_Lambda_e: Numeric | None = None,
) -> Numeric:
    r"""Normalised electron collisionality $\nu_e^*$ in the Sauter convention.

    $$\nu_e^* = 6.921\times10^{-18}\,
      \frac{q R\, n_e\, Z_{\mathrm{eff}} \ln\Lambda_e}{\epsilon^{3/2} T_e^2}$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density, strictly positive [m^-3].
    T_e : float or np.ndarray
        Electron temperature, strictly positive [eV].
    q : float or np.ndarray
        Safety factor; the magnitude is used, so either sign convention is
        accepted [-].
    R : float or np.ndarray
        Major radius of the surface, strictly positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio $r/R$, strictly positive [-].
    Z_eff : float or np.ndarray
        Effective ion charge, strictly positive [-].
    ln_Lambda_e : float or np.ndarray, optional
        Electron Coulomb logarithm; computed from *n_e* and *T_e* with
        :func:`coulomb_logarithm_electron_sauter` when omitted [-].

    Returns
    -------
    float or np.ndarray
        Electron collisionality, the ratio of the effective collision frequency
        to the banana bounce frequency [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    This is Sauter Eq. (18b) and it is one of **four** mutually inconsistent
    collisionality definitions now in this package, a situation tracked in
    issue #353.  The others are
    :func:`vaft.formula.equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I`
    (an engineering form with a $5\times10^{-11}$ prefactor),
    :func:`vaft.formula.equilibrium.normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q`
    (which reduces to Sauter Eq. 18b only when handed Sauter's own $\nu_{ii}$),
    and :func:`vaft.formula.stability.collisionality_from_n_T_B_R`.  Only this
    one may be passed to the Sauter and Redl coefficient functions here; the
    numbers are not interchangeable.

    Physical interpretation
    -----------------------
    Below one the plasma is in the banana regime and trapped orbits complete;
    above one collisions detrap particles first and the bootstrap coefficients
    fall away.

    Validity
    --------
    Positive $\epsilon$; the $\epsilon^{-3/2}$ factor diverges on axis, where
    the trapped fraction vanishes and the expression has no meaning.

    Limitations
    -----------
    The prefactor bundles physical constants evaluated for the Sauter unit
    choice; it is not dimensionally reusable with temperatures in keV.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (18b).
    """
    n_array = _validate_positive("n_e", n_e)
    t_array = _validate_positive("T_e", T_e)
    r_array = _validate_positive("R", R)
    epsilon_array = _validate_positive("epsilon", epsilon)
    z_array = _validate_positive("Z_eff", Z_eff)
    q_array = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q_array)):
        raise ValueError(f"q must be finite. Got {q!r}")
    if ln_Lambda_e is None:
        log_array = np.asarray(
            coulomb_logarithm_electron_sauter(n_array, t_array), dtype=float
        )
    else:
        log_array = _validate_positive("ln_Lambda_e", ln_Lambda_e)
    value = (
        COLLISIONALITY_COEF
        * np.abs(q_array)
        * r_array
        * n_array
        * z_array
        * log_array
        / (epsilon_array**1.5 * t_array**2)
    )
    return _maybe_scalar(value)


def ion_collisionality_sauter(
    n_i: Numeric,
    T_i: Numeric,
    q: Numeric,
    R: Numeric,
    epsilon: Numeric,
    Z: Numeric,
    ln_Lambda_ii: Numeric | None = None,
) -> Numeric:
    r"""Normalised ion collisionality $\nu_i^*$ in the Sauter convention.

    $$\nu_i^* = 4.90\times10^{-18}\,
      \frac{q R\, n_i\, Z^4 \ln\Lambda_{ii}}{\epsilon^{3/2} T_i^2}$$

    Parameters
    ----------
    n_i : float or np.ndarray
        Ion density, strictly positive [m^-3].
    T_i : float or np.ndarray
        Ion temperature, strictly positive [eV].
    q : float or np.ndarray
        Safety factor; the magnitude is used [-].
    R : float or np.ndarray
        Major radius of the surface, strictly positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio $r/R$, strictly positive [-].
    Z : float or np.ndarray
        Ion charge number, strictly positive [-].
    ln_Lambda_ii : float or np.ndarray, optional
        Ion-ion Coulomb logarithm; computed from *n_i*, *T_i* and *Z* with
        :func:`coulomb_logarithm_ion_sauter` when omitted [-].

    Returns
    -------
    float or np.ndarray
        Ion collisionality [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    Sauter Eq. (18c).  With more than one ion species the paper's single-species
    reading is ambiguous, and implementations differ.  NEO resolved it in 2013
    by evaluating this for the main ion and then scaling by the summed ion
    density, $\nu_i^* \to \nu_i^* \sum_s n_s / n_{\mathrm{main}}$, rather than
    by the $Z_i^2 Z_{\mathrm{eff}} n_e$ reading; a caller reproducing NEO must
    apply that scaling to this result.  See issue #353 for the wider
    collisionality-convention problem.

    Validity
    --------
    Positive $\epsilon$, as for the electron expression.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (18c).
    .. [2] E. A. Belli, GACODE ``neo/src/neo_theory.f90``, note of 11 July 2013
           (the multi-species reading NEO adopted).
    """
    n_array = _validate_positive("n_i", n_i)
    t_array = _validate_positive("T_i", T_i)
    r_array = _validate_positive("R", R)
    epsilon_array = _validate_positive("epsilon", epsilon)
    z_array = _validate_positive("Z", Z)
    q_array = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q_array)):
        raise ValueError(f"q must be finite. Got {q!r}")
    if ln_Lambda_ii is None:
        log_array = np.asarray(
            coulomb_logarithm_ion_sauter(n_array, t_array, z_array), dtype=float
        )
    else:
        log_array = _validate_positive("ln_Lambda_ii", ln_Lambda_ii)
    value = (
        _ION_COLLISIONALITY_COEF
        * np.abs(q_array)
        * r_array
        * n_array
        * z_array**4
        * log_array
        / (epsilon_array**1.5 * t_array**2)
    )
    return _maybe_scalar(value)


def sauter_spitzer_conductivity(
    T_e: Numeric, Z_eff: Numeric, ln_Lambda_e: Numeric
) -> Numeric:
    r"""Spitzer parallel conductivity $\sigma_{\mathrm{Sptz}}$ in the Sauter normalisation.

    $$\sigma_{\mathrm{Sptz}} = 1.9012\times10^{4}\,
      \frac{T_e^{3/2}}{Z_{\mathrm{eff}} N_Z(Z_{\mathrm{eff}}) \ln\Lambda_e},
      \qquad N_Z = 0.58 + \frac{0.74}{0.76 + Z_{\mathrm{eff}}}$$

    Parameters
    ----------
    T_e : float or np.ndarray
        Electron temperature, strictly positive [eV].
    Z_eff : float or np.ndarray
        Effective ion charge, strictly positive [-].
    ln_Lambda_e : float or np.ndarray
        Electron Coulomb logarithm, strictly positive; use
        :func:`coulomb_logarithm_electron_sauter` [-].

    Returns
    -------
    float or np.ndarray
        Classical parallel conductivity, without trapped-particle correction
        [S m^-1].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    This is the reference conductivity the neoclassical correction multiplies,
    Sauter Eq. (12), and it is not the same object as
    :func:`vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda`:
    that one applies the charge dependence linearly, this one through the
    fitted $N_Z$, so their reciprocals differ by tens of percent at
    $Z_{\mathrm{eff}} > 1$.  Pair this one with the neoclassical corrections in
    this module.

    Validity
    --------
    Empirical fit.  $N_Z$ is a fit to the Spitzer-Harm charge dependence, valid
    for $1 \le Z_{\mathrm{eff}} \lesssim 5$.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (12).
    .. [2] L. Spitzer and R. Harm, Phys. Rev. 89 (1953) 977.

    See Also
    --------
    vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
    """
    t_array = _validate_positive("T_e", T_e)
    z_array = _validate_positive("Z_eff", Z_eff)
    log_array = _validate_positive("ln_Lambda_e", ln_Lambda_e)
    charge_factor = 0.58 + 0.74 / (0.76 + z_array)
    value = _SPITZER_CONDUCTIVITY_COEF * t_array**1.5 / (
        z_array * charge_factor * log_array
    )
    return _maybe_scalar(value)


def sauter_neoclassical_conductivity(
    sigma_spitzer: Numeric, f_trap: Numeric, nu_e_star: Numeric, Z_eff: Numeric
) -> Numeric:
    r"""Neoclassical parallel conductivity $\sigma_{\mathrm{neo}}$, Sauter 1999.

    $$\frac{\sigma_{\mathrm{neo}}}{\sigma_{\mathrm{Sptz}}}
      = 1 - \left(1 + \frac{0.36}{Z}\right) X_{33}
        + \frac{0.59}{Z} X_{33}^2 - \frac{0.23}{Z} X_{33}^3$$

    with $X_{33} = f_t / \left[1 + (0.55 - 0.1 f_t)\sqrt{\nu_e^*}
    + 0.45(1-f_t)\nu_e^*/Z^{3/2}\right]$.

    Parameters
    ----------
    sigma_spitzer : float or np.ndarray
        Reference Spitzer conductivity from
        :func:`sauter_spitzer_conductivity` [S m^-1].
    f_trap : float or np.ndarray
        Trapped-particle fraction of the surface, in [0, 1] [-].
    nu_e_star : float or np.ndarray
        Electron collisionality from
        :func:`electron_collisionality_sauter`, non-negative [-].
    Z_eff : float or np.ndarray
        Effective ion charge, strictly positive [-].

    Returns
    -------
    float or np.ndarray
        Neoclassical parallel conductivity [S m^-1].

    Raises
    ------
    ValueError
        For non-finite input, *f_trap* outside [0, 1], negative *nu_e_star*, or
        non-positive *Z_eff*.

    Convention
    ----------
    The collisionality must be the Sauter Eq. (18b) one; see
    :func:`electron_collisionality_sauter` and issue #353.  The result is the
    coefficient of $\langle E_\parallel B\rangle$, so the Ohmic contribution to
    the parallel current is this times that field, added to the bootstrap term
    from :func:`sauter_bootstrap_current`.

    Physical interpretation
    -----------------------
    Trapped electrons cannot carry parallel current, so the conductivity falls
    below Spitzer roughly in proportion to $f_t$; collisions restore it, which
    is why the correction weakens as $\nu_e^*$ rises.

    Validity
    --------
    Empirical fit.  Fitted to numerical solutions of the drift-kinetic
    equation at conventional aspect ratio.

    Limitations
    -----------
    At the trapped fractions a spherical tokamak reaches, roughly
    $f_t \gtrsim 0.6$, this fit is outside the range it was built on;
    :func:`redl_neoclassical_conductivity` is the refit that covers it.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (13).
    .. [2] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 9 (2002)
           5140 (erratum).
    """
    sigma_array = _validate_positive("sigma_spitzer", sigma_spitzer)
    trapped = _validate_fraction("f_trap", f_trap)
    collisionality = _validate_non_negative("nu_e_star", nu_e_star)
    z_array = _validate_positive("Z_eff", Z_eff)
    x33 = trapped / (
        1.0
        + (0.55 - 0.1 * trapped) * np.sqrt(collisionality)
        + 0.45 * (1.0 - trapped) * collisionality / z_array**1.5
    )
    ratio = (
        1.0
        - (1.0 + 0.36 / z_array) * x33
        + (0.59 / z_array) * x33**2
        - (0.23 / z_array) * x33**3
    )
    return _maybe_scalar(sigma_array * ratio)


def redl_neoclassical_conductivity(
    sigma_spitzer: Numeric, f_trap: Numeric, nu_e_star: Numeric, Z_eff: Numeric
) -> Numeric:
    r"""Neoclassical parallel conductivity $\sigma_{\mathrm{neo}}$, Redl 2021.

    $$\frac{\sigma_{\mathrm{neo}}}{\sigma_{\mathrm{Sptz}}}
      = 1 - \left(1 + \frac{0.21}{Z}\right) X_{33}
        + \frac{0.54}{Z} X_{33}^2 - \frac{0.33}{Z} X_{33}^3$$

    with $X_{33} = f_t / \left[1 + 0.25(1 - 0.7 f_t)\sqrt{\nu_e^*}
    (1 + 0.45\sqrt{Z-1}) + 0.61(1 - 0.41 f_t)\nu_e^*/\sqrt{Z}\right]$.

    Parameters
    ----------
    sigma_spitzer : float or np.ndarray
        Reference Spitzer conductivity from
        :func:`sauter_spitzer_conductivity` [S m^-1].
    f_trap : float or np.ndarray
        Trapped-particle fraction of the surface, in [0, 1] [-].
    nu_e_star : float or np.ndarray
        Electron collisionality from
        :func:`electron_collisionality_sauter`, non-negative [-].
    Z_eff : float or np.ndarray
        Effective ion charge, at least one [-].

    Returns
    -------
    float or np.ndarray
        Neoclassical parallel conductivity [S m^-1].

    Raises
    ------
    ValueError
        For non-finite input, *f_trap* outside [0, 1], negative *nu_e_star*, or
        *Z_eff* below one.

    Convention
    ----------
    The reference conductivity is still Sauter Eq. (12), so
    :func:`sauter_spitzer_conductivity` is the right input here despite the
    name; Redl refits the correction, not the normalisation.  As in the Sauter
    case the collisionality must be the Eq. (18b) one (issue #353).

    Validity
    --------
    Empirical fit.  Fitted to NEO solutions spanning tight aspect ratio, so it
    stays usable at the trapped fractions where the 1999 fit does not.
    $Z_{\mathrm{eff}} \ge 1$ is required because $\sqrt{Z-1}$ appears.

    References
    ----------
    .. [1] A. Redl, C. Angioni, E. Belli and O. Sauter, Phys. Plasmas 28 (2021)
           022502.
    """
    sigma_array = _validate_positive("sigma_spitzer", sigma_spitzer)
    trapped = _validate_fraction("f_trap", f_trap)
    collisionality = _validate_non_negative("nu_e_star", nu_e_star)
    z_array = np.asarray(Z_eff, dtype=float)
    if np.any(~np.isfinite(z_array)):
        raise ValueError(f"Z_eff must be finite. Got {Z_eff!r}")
    if np.any(z_array < 1.0):
        raise ValueError(f"Z_eff must be >= 1. Got {Z_eff!r}")
    x33 = trapped / (
        1.0
        + 0.25
        * (1.0 - 0.7 * trapped)
        * np.sqrt(collisionality)
        * (1.0 + 0.45 * np.sqrt(z_array - 1.0))
        + 0.61 * (1.0 - 0.41 * trapped) * collisionality / np.sqrt(z_array)
    )
    ratio = (
        1.0
        - (1.0 + 0.21 / z_array) * x33
        + (0.54 / z_array) * x33**2
        - (0.33 / z_array) * x33**3
    )
    return _maybe_scalar(sigma_array * ratio)


def sauter_bootstrap_coefficients(
    f_trap: Numeric, nu_e_star: Numeric, nu_i_star: Numeric, Z_eff: Numeric
) -> BootstrapCoefficients:
    r"""The Sauter 1999 bootstrap coefficients $L_{31}$, $L_{32}$, $L_{34}$, $\alpha$.

    Each coefficient is a rational fit in an effective trapped fraction that
    collisions reduce, for example
    $X_{31} = f_t / [1 + (1 - 0.1 f_t)\sqrt{\nu_e^*}
    + 0.5(1-f_t)\nu_e^*/Z]$, with $L_{31}$ a quartic in $X_{31}$.

    Parameters
    ----------
    f_trap : float or np.ndarray
        Trapped-particle fraction of the surface, in [0, 1] [-].
    nu_e_star : float or np.ndarray
        Electron collisionality from
        :func:`electron_collisionality_sauter`, non-negative [-].
    nu_i_star : float or np.ndarray
        Ion collisionality from :func:`ion_collisionality_sauter`,
        non-negative [-].
    Z_eff : float or np.ndarray
        Effective ion charge, strictly positive [-].

    Returns
    -------
    BootstrapCoefficients
        The four coefficients, each dimensionless and each an array when any
        input was an array [-].

    Raises
    ------
    ValueError
        For non-finite input, *f_trap* outside [0, 1], a negative
        collisionality, or non-positive *Z_eff*.

    Convention
    ----------
    Incorporating the 2002 erratum.  $L_{32}$ is the sum of the two branches
    $F_{32,ee}$ and $F_{32,ei}$, each evaluated at its own effective trapped
    fraction; they are not separately meaningful and are not returned apart.
    Both collisionalities must be the Sauter Eq. (18b) and (18c) ones, and
    *nu_i_star* must already carry the multi-species scaling described in
    :func:`ion_collisionality_sauter` if the caller is reproducing NEO.  See
    issue #353.

    Physical interpretation
    -----------------------
    $L_{31}$ multiplies the total pressure gradient, $L_{32}$ the electron
    temperature gradient, and $L_{34}\alpha$ the ion temperature gradient.
    $\alpha$ is negative in the banana regime, so the ion-temperature term
    opposes the other two.

    Validity
    --------
    Empirical fit.  Quoted by the authors as accurate to a few percent for
    $0 \le \nu^* \le 100$ and $1 \le Z_{\mathrm{eff}} \le 5$ at conventional
    aspect ratio.

    Limitations
    -----------
    The fits were built on surfaces with $f_t$ well below what a spherical
    tokamak reaches; at VEST's $f_t \approx 0.7$ they are extrapolations.
    :func:`redl_bootstrap_coefficients` is the refit that covers the range, and
    the difference between the two is the honest uncertainty band.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eqs. (14)-(17).
    .. [2] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 9 (2002)
           5140 (erratum).
    """
    trapped = _validate_fraction("f_trap", f_trap)
    nu_e = _validate_non_negative("nu_e_star", nu_e_star)
    nu_i = _validate_non_negative("nu_i_star", nu_i_star)
    z_array = _validate_positive("Z_eff", Z_eff)
    root_nu_e = np.sqrt(nu_e)

    x31 = trapped / (
        1.0
        + (1.0 - 0.1 * trapped) * root_nu_e
        + 0.5 * (1.0 - trapped) * nu_e / z_array
    )
    l31 = _l31_polynomial(x31, z_array)

    x32e = trapped / (
        1.0
        + 0.26 * (1.0 - trapped) * root_nu_e
        + 0.18 * (1.0 - 0.37 * trapped) * nu_e / np.sqrt(z_array)
    )
    f32_ee = (
        (0.05 + 0.62 * z_array) / (z_array * (1.0 + 0.44 * z_array))
        * (x32e - x32e**4)
        + 1.0 / (1.0 + 0.22 * z_array)
        * (x32e**2 - x32e**4 - 1.2 * (x32e**3 - x32e**4))
        + 1.2 / (1.0 + 0.5 * z_array) * x32e**4
    )
    x32ei = trapped / (
        1.0
        + (1.0 + 0.6 * trapped) * root_nu_e
        + 0.85 * (1.0 - 0.37 * trapped) * nu_e * (1.0 + z_array)
    )
    f32_ei = (
        -(0.56 + 1.93 * z_array) / (z_array * (1.0 + 0.44 * z_array))
        * (x32ei - x32ei**4)
        + 4.95 / (1.0 + 2.48 * z_array)
        * (x32ei**2 - x32ei**4 - 0.55 * (x32ei**3 - x32ei**4))
        - 1.2 / (1.0 + 0.5 * z_array) * x32ei**4
    )
    l32 = f32_ee + f32_ei

    x34 = trapped / (
        1.0
        + (1.0 - 0.1 * trapped) * root_nu_e
        + 0.5 * (1.0 - 0.5 * trapped) * nu_e / z_array
    )
    l34 = _l31_polynomial(x34, z_array)

    alpha_0 = -1.17 * (1.0 - trapped) / (
        1.0 - 0.22 * trapped - 0.19 * trapped**2
    )
    root_nu_i = np.sqrt(nu_i)
    alpha = (
        (alpha_0 + 0.25 * (1.0 - trapped**2) * root_nu_i) / (1.0 + 0.5 * root_nu_i)
        + 0.315 * nu_i**2 * trapped**6
    ) / (1.0 + 0.15 * nu_i**2 * trapped**6)

    return BootstrapCoefficients(
        L31=_maybe_scalar(l31),
        L32=_maybe_scalar(l32),
        L34=_maybe_scalar(l34),
        alpha=_maybe_scalar(alpha),
    )


def redl_bootstrap_coefficients(
    f_trap: Numeric, nu_e_star: Numeric, nu_i_star: Numeric, Z_eff: Numeric
) -> BootstrapCoefficients:
    r"""The Redl 2021 bootstrap coefficients $L_{31}$, $L_{32}$, $L_{34}$, $\alpha$.

    The same four-coefficient structure as Sauter 1999, refitted against NEO
    over a parameter range that includes tight aspect ratio, for example
    $L_{31} = X_{31} + (0.15 X_{31} - 0.22 X_{31}^2 + 0.01 X_{31}^3
    + 0.06 X_{31}^4)/(Z^{1.2} - 0.71)$.

    Parameters
    ----------
    f_trap : float or np.ndarray
        Trapped-particle fraction of the surface, in [0, 1] [-].
    nu_e_star : float or np.ndarray
        Electron collisionality from
        :func:`electron_collisionality_sauter`, non-negative [-].
    nu_i_star : float or np.ndarray
        Ion collisionality from :func:`ion_collisionality_sauter`,
        non-negative [-].
    Z_eff : float or np.ndarray
        Effective ion charge, at least one [-].

    Returns
    -------
    BootstrapCoefficients
        The four coefficients, each dimensionless [-].

    Raises
    ------
    ValueError
        For non-finite input, *f_trap* outside [0, 1], a negative
        collisionality, or *Z_eff* below one.

    Convention
    ----------
    $L_{34}$ is set equal to $L_{31}$: Redl does not refit it separately, and
    the field is kept only so that the return shape matches
    :func:`sauter_bootstrap_coefficients` and the shared current assembly can
    consume either.  The collisionality convention is unchanged from Sauter
    (issue #353), and $Z_{\mathrm{eff}} \ge 1$ is required because
    $\sqrt{Z-1}$ appears in several denominators.

    Validity
    --------
    Empirical fit.  Refitted against NEO across aspect ratios reaching the
    spherical-tokamak range, which is the reason to prefer it over the 1999 fit
    for VEST.

    Limitations
    -----------
    Agreement with the 1999 fit is close at conventional aspect ratio and
    degrades as $f_t$ rises; treat a large Sauter-Redl gap as a signal that the
    analytic model is being asked for more than it can give, not as an error in
    either.

    References
    ----------
    .. [1] A. Redl, C. Angioni, E. Belli and O. Sauter, Phys. Plasmas 28 (2021)
           022502.
    """
    trapped = _validate_fraction("f_trap", f_trap)
    nu_e = _validate_non_negative("nu_e_star", nu_e_star)
    nu_i = _validate_non_negative("nu_i_star", nu_i_star)
    z_array = np.asarray(Z_eff, dtype=float)
    if np.any(~np.isfinite(z_array)):
        raise ValueError(f"Z_eff must be finite. Got {Z_eff!r}")
    if np.any(z_array < 1.0):
        raise ValueError(f"Z_eff must be >= 1. Got {Z_eff!r}")
    root_nu_e = np.sqrt(nu_e)
    z_minus_one = z_array - 1.0

    x31 = trapped / (
        1.0
        + 0.67 * (1.0 - 0.7 * trapped) * root_nu_e / (0.56 + 0.44 * z_array)
        + (0.52 + 0.086 * root_nu_e)
        * (1.0 + 0.87 * trapped)
        * nu_e
        / (1.0 + 1.13 * np.sqrt(z_minus_one))
    )
    l31 = x31 + (
        0.15 * x31 - 0.22 * x31**2 + 0.01 * x31**3 + 0.06 * x31**4
    ) / (z_array**1.2 - 0.71)

    x32e = trapped / (
        1.0
        + 0.23 * (1.0 - 0.96 * trapped) * np.sqrt(nu_e / z_array)
        + 0.13
        * (1.0 - 0.38 * trapped)
        * nu_e
        / z_array**2
        * (
            np.sqrt(1.0 + 2.0 * np.sqrt(z_minus_one))
            + trapped**2 * np.sqrt(nu_e * (0.075 + 0.25 * z_minus_one**2))
        )
    )
    f32_ee = (
        (0.1 + 0.6 * z_array)
        / (z_array * (0.77 + 0.63 * (1.0 + z_minus_one**1.1)))
        * (x32e - x32e**4)
        + 0.7 / (1.0 + 0.2 * z_array)
        * (x32e**2 - x32e**4 - 1.2 * (x32e**3 - x32e**4))
        + 1.3 / (1.0 + 0.5 * z_array) * x32e**4
    )
    x32ei = trapped / (
        1.0
        + 0.87 * (1.0 + 0.39 * trapped) * root_nu_e / (1.0 + 2.95 * z_minus_one**2)
        + 1.53 * (1.0 - 0.37 * trapped) * nu_e * (2.0 + 0.375 * z_minus_one)
    )
    f32_ei = (
        -(0.4 + 1.93 * z_array) / (z_array * (0.8 + 0.6 * z_array))
        * (x32ei - x32ei**4)
        + 5.5 / (1.5 + 2.0 * z_array)
        * (x32ei**2 - x32ei**4 - 0.8 * (x32ei**3 - x32ei**4))
        - 1.3 / (1.0 + 0.5 * z_array) * x32ei**4
    )
    l32 = f32_ee + f32_ei

    alpha_0 = (
        -(0.62 + 0.055 * z_minus_one)
        / (0.53 + 0.17 * z_minus_one)
        * (1.0 - trapped)
        / (1.0 - trapped * (0.31 - 0.065 * z_minus_one) - 0.25 * trapped**2)
    )
    alpha = (
        (alpha_0 + 0.7 * z_array * np.sqrt(trapped * nu_i))
        / (1.0 + 0.18 * np.sqrt(nu_i))
        - 0.002 * nu_i**2 * trapped**6
    ) / (1.0 + 0.004 * nu_i**2 * trapped**6)

    return BootstrapCoefficients(
        L31=_maybe_scalar(l31),
        L32=_maybe_scalar(l32),
        L34=_maybe_scalar(l31),
        alpha=_maybe_scalar(alpha),
    )


def _assemble_bootstrap_current(
    coefficients: BootstrapCoefficients,
    I_psi: Numeric,
    p_e: Numeric,
    p_i: Numeric,
    dp_dpsi: Numeric,
    dln_Te_dpsi: Numeric,
    dln_Ti_dpsi: Numeric,
) -> Numeric:
    """Combine coefficients and gradients into the bootstrap current.

    Shared by the Sauter and Redl entry points because the assembly is common
    to both models; only the coefficient fits differ.
    """
    i_array = np.asarray(I_psi, dtype=float)
    pe_array = np.asarray(p_e, dtype=float)
    pi_array = np.asarray(p_i, dtype=float)
    dp_array = np.asarray(dp_dpsi, dtype=float)
    dte_array = np.asarray(dln_Te_dpsi, dtype=float)
    dti_array = np.asarray(dln_Ti_dpsi, dtype=float)
    for name, array in (
        ("I_psi", i_array),
        ("p_e", pe_array),
        ("p_i", pi_array),
        ("dp_dpsi", dp_array),
        ("dln_Te_dpsi", dte_array),
        ("dln_Ti_dpsi", dti_array),
    ):
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must be finite.")
    value = -i_array * (
        np.asarray(coefficients.L31, dtype=float) * dp_array
        + np.asarray(coefficients.L32, dtype=float) * pe_array * dte_array
        + np.asarray(coefficients.L34, dtype=float)
        * np.asarray(coefficients.alpha, dtype=float)
        * pi_array
        * dti_array
    )
    return _maybe_scalar(value)


def sauter_bootstrap_current(
    f_trap: Numeric,
    nu_e_star: Numeric,
    nu_i_star: Numeric,
    Z_eff: Numeric,
    I_psi: Numeric,
    p_e: Numeric,
    p_i: Numeric,
    dp_dpsi: Numeric,
    dln_Te_dpsi: Numeric,
    dln_Ti_dpsi: Numeric,
) -> Numeric:
    r"""Flux-surface-averaged bootstrap current $\langle j_\parallel B\rangle$, Sauter 1999.

    $$\langle j_\parallel B\rangle_{\mathrm{bs}} = -I(\psi)\left[
      L_{31}\frac{\partial p}{\partial\psi}
      + L_{32}\,p_e \frac{\partial \ln T_e}{\partial\psi}
      + L_{34}\alpha\,p_i \frac{\partial \ln T_i}{\partial\psi}\right]$$

    Parameters
    ----------
    f_trap : float or np.ndarray
        Trapped-particle fraction of the surface, in [0, 1] [-].
    nu_e_star : float or np.ndarray
        Electron collisionality from
        :func:`electron_collisionality_sauter` [-].
    nu_i_star : float or np.ndarray
        Ion collisionality from :func:`ion_collisionality_sauter` [-].
    Z_eff : float or np.ndarray
        Effective ion charge, strictly positive [-].
    I_psi : float or np.ndarray
        The flux function $I = R B_\phi$ of the surface [T m].
    p_e : float or np.ndarray
        Electron pressure $n_e T_e$ [Pa].
    p_i : float or np.ndarray
        Summed thermal-ion pressure $\sum_{\mathrm{ions}} n_s T_s$ [Pa].
    dp_dpsi : float or np.ndarray
        Derivative of the total thermal pressure, electrons included, with
        respect to poloidal flux per radian [Pa rad Wb^-1].
    dln_Te_dpsi : float or np.ndarray
        Logarithmic derivative of the electron temperature with respect to
        poloidal flux per radian [rad Wb^-1].
    dln_Ti_dpsi : float or np.ndarray
        Logarithmic derivative of the ion temperature with respect to poloidal
        flux per radian [rad Wb^-1].

    Returns
    -------
    float or np.ndarray
        Flux-surface-averaged bootstrap current density times field strength
        [A T m^-2].

    Raises
    ------
    ValueError
        For non-finite input, *f_trap* outside [0, 1], a negative
        collisionality, or non-positive *Z_eff*.

    Convention
    ----------
    Poloidal flux is **per radian**, not the full weber the IMAS data
    dictionary stores; convert an ODS-sourced psi with
    :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor` before
    differentiating.  The sign follows from that choice together with the sign
    of $I(\psi)$, so a COCOS mismatch shows up here as a sign flip rather than
    as a magnitude error.

    This is the bootstrap term alone.  The Ohmic term
    $\sigma_{\mathrm{neo}}\langle E_\parallel B\rangle$, with
    $\sigma_{\mathrm{neo}}$ from :func:`sauter_neoclassical_conductivity`, is
    added by the caller when an inductive field is known.

    The single ion temperature in the last term is the paper's reduction for
    ions that share a temperature; with unlike ion temperatures, use the main
    ion's logarithmic gradient against the summed ion pressure, which is what
    NEO does.

    Physical interpretation
    -----------------------
    Trapped particles on adjacent orbits carry unequal momentum where a
    gradient exists, and the resulting banana current is transferred to the
    passing population by collisions.  It is a pressure-gradient-driven current
    that needs no loop voltage, which is why it dominates the current budget of
    a high-beta spherical tokamak.

    Validity
    --------
    Empirical fit.  Inherits the range of
    :func:`sauter_bootstrap_coefficients`.

    Limitations
    -----------
    At VEST's trapped fraction this is an extrapolation of the 1999 fit;
    compare against :func:`redl_bootstrap_current` rather than trusting either
    alone.  It also assumes the local gradients are resolved: on a reconstructed
    equilibrium the near-axis region often is not.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834, Eq. (11).
    .. [2] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 9 (2002)
           5140 (erratum).

    See Also
    --------
    vaft.formula.equilibrium.bootstrap_current_fraction : the zero-dimensional
        heuristic this supersedes for profile work.
    redl_bootstrap_current : the 2021 refit, preferred at tight aspect ratio.
    """
    coefficients = sauter_bootstrap_coefficients(f_trap, nu_e_star, nu_i_star, Z_eff)
    return _assemble_bootstrap_current(
        coefficients, I_psi, p_e, p_i, dp_dpsi, dln_Te_dpsi, dln_Ti_dpsi
    )


def redl_bootstrap_current(
    f_trap: Numeric,
    nu_e_star: Numeric,
    nu_i_star: Numeric,
    Z_eff: Numeric,
    I_psi: Numeric,
    p_e: Numeric,
    p_i: Numeric,
    dp_dpsi: Numeric,
    dln_Te_dpsi: Numeric,
    dln_Ti_dpsi: Numeric,
) -> Numeric:
    r"""Flux-surface-averaged bootstrap current $\langle j_\parallel B\rangle$, Redl 2021.

    The assembly is identical to :func:`sauter_bootstrap_current`; only the
    coefficients differ, coming from :func:`redl_bootstrap_coefficients`.

    Parameters
    ----------
    f_trap : float or np.ndarray
        Trapped-particle fraction of the surface, in [0, 1] [-].
    nu_e_star : float or np.ndarray
        Electron collisionality from
        :func:`electron_collisionality_sauter` [-].
    nu_i_star : float or np.ndarray
        Ion collisionality from :func:`ion_collisionality_sauter` [-].
    Z_eff : float or np.ndarray
        Effective ion charge, at least one [-].
    I_psi : float or np.ndarray
        The flux function $I = R B_\phi$ of the surface [T m].
    p_e : float or np.ndarray
        Electron pressure $n_e T_e$ [Pa].
    p_i : float or np.ndarray
        Summed thermal-ion pressure $\sum_{\mathrm{ions}} n_s T_s$ [Pa].
    dp_dpsi : float or np.ndarray
        Derivative of the total thermal pressure, electrons included, with
        respect to poloidal flux per radian [Pa rad Wb^-1].
    dln_Te_dpsi : float or np.ndarray
        Logarithmic derivative of the electron temperature with respect to
        poloidal flux per radian [rad Wb^-1].
    dln_Ti_dpsi : float or np.ndarray
        Logarithmic derivative of the ion temperature with respect to poloidal
        flux per radian [rad Wb^-1].

    Returns
    -------
    float or np.ndarray
        Flux-surface-averaged bootstrap current density times field strength
        [A T m^-2].

    Raises
    ------
    ValueError
        For non-finite input, *f_trap* outside [0, 1], a negative
        collisionality, or *Z_eff* below one.

    Convention
    ----------
    Identical to :func:`sauter_bootstrap_current`: poloidal flux per radian,
    sign carried by $I(\psi)$, bootstrap term only.

    Validity
    --------
    Empirical fit.  Inherits the range of
    :func:`redl_bootstrap_coefficients`, which covers tight aspect ratio and is
    therefore the one to prefer for VEST.

    Limitations
    -----------
    Being the better-conditioned fit does not make it exact; a drift-kinetic
    solve through :mod:`vaft.code.gacode` remains the reference.

    References
    ----------
    .. [1] A. Redl, C. Angioni, E. Belli and O. Sauter, Phys. Plasmas 28 (2021)
           022502.

    See Also
    --------
    sauter_bootstrap_current : the 1999 formulation.
    """
    coefficients = redl_bootstrap_coefficients(f_trap, nu_e_star, nu_i_star, Z_eff)
    return _assemble_bootstrap_current(
        coefficients, I_psi, p_e, p_i, dp_dpsi, dln_Te_dpsi, dln_Ti_dpsi
    )
