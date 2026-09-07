"""
Plasma stability, operational limits, and transport calculations.

This module provides functions for calculating various stability parameters
including beta limits, ballooning stability, MHD stability criteria, operational
limits, and transport parameters.

Notation
--------
β_N    : normalized beta                              [-]
β_p    : poloidal beta                                [-]
β_t    : toroidal beta                                [-]
q_95   : safety factor at 95% flux surface            [-]
α      : ballooning parameter                         [-]
s      : magnetic shear                               [-]
n_G    : Greenwald density limit                      [10¹⁹ m⁻³]
P_L    : power limit                                  [W]
ν*     : effective collisionality                     [-]
v_A    : Alfven speed                                [m/s]
c_s    : ion-sound speed                             [m/s]
τ_E    : energy confinement time                     [s]
"""

import numpy as np
from typing import Union, Tuple

from .constants import (
    MU0, QE, ME, MI_P,
    COLLISIONALITY_COEF,
    _SCALING_COEFS
)
from .utils import gradient

#: What ``from vaft.formula.stability import *`` binds, and therefore what
#: reaches ``vaft.formula.__all__``. Stability limits, operational boundaries and transport figures.
#: Declared so the package stops re-exporting this module's own imports --
#: ``np``, ``warnings``, ``Union``, ``curve_fit`` -- as though they were
#: formulas (#368).
__all__ = [
    "ballooning_alpha_from_p_B_R",
    "ballooning_stability_criterion",
    "beta_N_from_beta_a_B0_Ip",
    "beta_pol_from_beta_tor",
    "beta_stability_boundary",
    "beta_tor_from_beta_pol",
    "c_s_from_Te_Ti_mi",
    "collisionality_from_n_T_B_R",
    "empirical_li_qa",
    "greenwald_density",
    "greenwald_fraction",
    "kink_stability_criterion",
    "li_from_qa_empirical",
    "plasma_stability_margins",
    "power_limit_from_beta",
    "power_limit_from_q",
    "rhostar_from_Te_a_Bt",
    "sawtooth_stability_criterion",
    "v_alfven_from_B_n_mi",
]


# ------------------------------------------------------------------
# Beta Calculations
# ------------------------------------------------------------------


def _validate_epsilon(epsilon):
    """The cylindrical beta relations divide by epsilon; refuse a non-positive one."""
    value = np.asarray(epsilon, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(
            f"epsilon must be finite and strictly positive, got {epsilon!r}"
        )
    return value if value.ndim else float(value)


def beta_N_from_beta_a_B0_Ip(beta_percent: float,
                            a: float,
                            B0: float,
                            I_p_MA: float) -> float:
    r"""Normalised beta in the Troyon convention, %·m·T/MA.

    $$\beta_N = \frac{\beta[\%]\;a[\mathrm{m}]\;B_0[\mathrm{T}]}{I_p[\mathrm{MA}]}$$

    Parameters
    ----------
    beta_percent : float
        Toroidal beta **in percent**, not as a fraction [%].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field on axis [T].
    I_p_MA : float
        Plasma current **in megaamperes** [MA].

    Returns
    -------
    float
        Normalised beta, directly comparable with the Troyon limit [%·m·T/MA].

    Convention
    ----------
    Percent and megaamperes, which is how Troyon [1]_ and the ITER Physics
    Basis [2]_ quote $\beta_N$ and the only convention in which the limit
    $\beta_N \lesssim 2.8$ means anything. The parameter names carry their
    units because this function used to take SI -- a fraction and amperes --
    and return $10^{-8}$ times the conventional number, which no rescaling of
    the *output* fixes for a reader comparing against 2.8 (#349).

    Physical interpretation
    -----------------------
    The pressure a tokamak can hold scales with $I_p/(aB_0)$, so dividing it
    out leaves a figure that is comparable across machines; exceeding ~2.8
    means an ideal-MHD beta limit rather than a machine-specific one.

    References
    ----------
    .. [1] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 3,
           Sec. 2.1 (definition of $\beta_N$).
    """
    return beta_percent * a * B0 / I_p_MA


def beta_pol_from_beta_tor(beta_tor: float,
                          q_95: float,
                          epsilon: float) -> float:
    r"""Poloidal beta from toroidal beta, cylindrical relation.

    $$\beta_p = \beta_t\left(\frac{q}{\varepsilon}\right)^2$$

    Parameters
    ----------
    beta_tor : float
        Toroidal beta [-].
    q_95 : float
        Safety factor at the 95% flux surface [-].
    epsilon : float
        Inverse aspect ratio $a/R_0$, strictly positive [-].

    Returns
    -------
    float
        Poloidal beta [-].

    Raises
    ------
    ValueError
        When ``epsilon`` is not strictly positive, since the relation divides
        by it [-].

    Assumptions
    -----------
    Circular, large-aspect-ratio cylinder in which $B_p/B_t = \varepsilon/q$.

    Limitations
    -----------
    Until #363 the $1/\varepsilon^2$ was missing, so the result was right only
    at $\varepsilon = 1$ and low by $\varepsilon^2$ everywhere else -- a factor
    of about 2 at the $\varepsilon \simeq 0.7$ of a spherical tokamak and more
    than 10 at a conventional $\varepsilon \simeq 0.3$. Round-tripping through
    :func:`beta_tor_from_beta_pol` hid it, because both directions were wrong by
    the same factor.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.5 (relation between $\beta$, $\beta_p$ and $q$).
    """
    epsilon = _validate_epsilon(epsilon)
    return beta_tor * (q_95 / epsilon) ** 2


def beta_tor_from_beta_pol(beta_pol: float,
                          q_95: float,
                          epsilon: float) -> float:
    r"""Toroidal beta from poloidal beta, cylindrical relation.

    $$\beta_t = \beta_p\left(\frac{\varepsilon}{q}\right)^2$$

    Parameters
    ----------
    beta_pol : float
        Poloidal beta [-].
    q_95 : float
        Safety factor at the 95% flux surface [-].
    epsilon : float
        Inverse aspect ratio $a/R_0$, strictly positive [-].

    Returns
    -------
    float
        Toroidal beta [-].

    Raises
    ------
    ValueError
        When ``epsilon`` is not strictly positive [-].

    Assumptions
    -----------
    Circular, large-aspect-ratio cylinder; exact inverse of
    :func:`beta_pol_from_beta_tor`.

    Limitations
    -----------
    Carried the same missing $\varepsilon^2$ as its inverse until #363.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 3.5.
    """
    epsilon = _validate_epsilon(epsilon)
    return beta_pol * (epsilon / q_95) ** 2


# ------------------------------------------------------------------
# Empirical Data
# ------------------------------------------------------------------

def empirical_li_qa():
    r"""Surveyed $(q_a, l_i)$ operating points from the JET disruption study.

    Eighteen $(q_a, l_i)$ pairs read from the JET operational diagram: at each
    integer $q_a$ from 2 to 10, the upper and lower $l_i$ of the observed
    operating band.

    Returns
    -------
    qa : np.ndarray
        Edge safety factor of each point [-].
    li : np.ndarray
        Internal inductance of each point [-].

    Physical interpretation
    -----------------------
    Low $q_a$ goes with high $l_i$ (a peaked current profile); above
    $q_a \approx 6$ the lower branch saturates near $l_i \approx 0.3$ (flat
    profile).  The band bounds the region where JET discharges avoided
    disruptions in the $q_a$-$l_i$ plane.

    Validity
    --------
    Empirical fit.  Digitised from the JET survey of Wesson et al. [1]_
    (ohmic and early NBI discharges, circular-to-D-shaped, 1985-1988); the
    values are approximate readings of a published figure, not tabulated data.

    Limitations
    -----------
    JET-specific operating experience; a spherical tokamak reaches lower $q_a$
    and different $l_i$ ranges, so use only as a qualitative boundary.

    References
    ----------
    .. [1] J. A. Wesson et al., "Disruptions in JET", Nucl. Fusion 29 (1989)
           641, Fig. 5 ($l_i$-$q_a$ diagram).
    """
    qa = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                   7, 7, 8, 8, 9, 9, 10, 10])
    li = np.array([0.95, 0.68, 0.93, 0.61, 0.86, 0.5, 0.71, 0.435, 0.7, 0.35,
                   0.67, 0.3, 0.67, 0.3, 0.67, 0.3, 0.67, 0.3])
    return qa, li


def li_from_qa_empirical(qa: np.ndarray) -> np.ndarray:
    r"""Internal inductance at a given $q_a$ by interpolating the JET survey points.

    $$l_i(q_a) = \mathrm{interp}\big(q_a;\ q_a^{(k)}, l_i^{(k)}\big)$$

    Parameters
    ----------
    qa : np.ndarray
        Edge safety factor values [-].

    Returns
    -------
    np.ndarray
        Interpolated internal inductance [-].

    Validity
    --------
    Empirical fit.  Piecewise-linear interpolation through the eighteen
    :func:`empirical_li_qa` points from Wesson et al. [1]_; a sanity check for
    when only $q_a$ is known.

    Limitations
    -----------
    The survey lists two $l_i$ per integer $q_a$ (upper and lower band edge),
    so ``numpy.interp`` over the duplicated abscissae returns a value that
    depends on the point ordering; outside $2 \le q_a \le 10$ the end values are
    held constant (no extrapolation).

    References
    ----------
    .. [1] J. A. Wesson et al., Nucl. Fusion 29 (1989) 641, Fig. 5.
    """
    qa_ref, li_ref = empirical_li_qa()
    return np.interp(qa, qa_ref, li_ref)


# ------------------------------------------------------------------
# Ballooning Stability
# ------------------------------------------------------------------

def ballooning_alpha_from_p_B_R(p: Union[float, np.ndarray],
                            B: Union[float, np.ndarray],
                            R: Union[float, np.ndarray],
                            q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Normalised pressure gradient $\alpha$ of the $s$-$\alpha$ ballooning model.

    $$\alpha = -\frac{2\mu_0 R q^2}{B^2}\,\frac{dp}{dr}$$

    Parameters
    ----------
    p : float or np.ndarray
        Pressure along a radial cut [Pa].
    B : float or np.ndarray
        Magnetic field strength [T].
    R : float or np.ndarray
        Major radius of the samples, monotonic [m].
    q : float or np.ndarray
        Safety factor on the same surfaces [-].

    Returns
    -------
    float or np.ndarray
        Ballooning parameter in the Connor-Hastie-Taylor normalisation [-].

    Convention
    ----------
    Connor, Hastie and Taylor [1]_ define $\alpha$ with the $q^2$ and with $r$
    the minor radius. Until #364 this omitted the $q^2$ entirely, so it returned
    $\alpha/q^2$ -- an order of magnitude at $q \simeq 3$ -- while
    :func:`ballooning_stability_criterion` compared the result against
    $0.6\,s$ as though it were the standard $\alpha$.

    Limitations
    -----------
    The derivative is still taken against the **major** radius $R$, not the
    minor radius $r$ of the definition. For a large-aspect-ratio circular
    plasma on the outboard midplane the two coincide, which is the case this is
    used for; anywhere else the caller must supply a minor-radius abscissa.

    Numerical notes
    ---------------
    ``numpy.gradient`` along the supplied axis (second-order interior, first-order
    ends), sign-sensitive to the direction of ``R``.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40
           (1978) 396, Eq. (2).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.13 (ballooning modes).
    """
    return -2 * MU0 * R * q**2 * gradient(R, p) / B**2


def ballooning_stability_criterion(alpha: Union[float, np.ndarray],
                                 s: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""Distance from the first ballooning stability boundary $\alpha_{crit} \approx 0.6\,s$.

    $$\Delta = \alpha - \alpha_{crit}, \qquad \alpha_{crit} = 0.6\,s$$

    Parameters
    ----------
    alpha : float or np.ndarray
        Ballooning parameter in the Connor-Hastie-Taylor normalisation [-].
    s : float or np.ndarray
        Magnetic shear [-].

    Returns
    -------
    margin : float or np.ndarray
        $\alpha - \alpha_{crit}$; positive means unstable [-].
    alpha_crit : float or np.ndarray
        The threshold $0.6\,s$ [-].

    Validity
    --------
    Empirical fit.  A straight-line approximation of the first-stability
    boundary of the circular $s$-$\alpha$ diagram [1]_ for moderate shear
    ($0.3 \lesssim s \lesssim 1.5$); the true boundary bends over and closes at
    $\alpha \approx 1$, with second stability beyond.

    Limitations
    -----------
    Ignores shaping, which raises the boundary, and the second-stable region;
    uncited coefficient (read from the diagram).

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40
           (1978) 396, Fig. 1.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 6.13.
    """
    alpha_crit = 0.6 * s
    return alpha - alpha_crit, alpha_crit


# ------------------------------------------------------------------
# MHD Stability
# ------------------------------------------------------------------

def kink_stability_criterion(q_95: float,
                           beta_N: float) -> Tuple[float, float]:
    r"""Heuristic kink margin against $\beta_{N,crit} = 2.8\,q_{95}$.

    $$\Delta = \beta_N - \beta_{N,crit}, \qquad \beta_{N,crit} = 2.8\,q_{95}$$

    Parameters
    ----------
    q_95 : float
        Safety factor at the 95% flux surface [-].
    beta_N : float
        Normalised beta in %·m·T/MA [-].

    Returns
    -------
    margin : float
        $\beta_N - \beta_{N,crit}$; positive means the limit is exceeded [-].
    beta_N_crit : float
        The threshold $2.8\,q_{95}$ [-].

    Validity
    --------
    Empirical fit.  The coefficient 2.8 is the Troyon limit $\beta_N \le 2.8$
    [1]_; the multiplication by $q_{95}$ has no source in the literature or the
    VAFT history and makes the limit rise with $q_{95}$, opposite to the
    observed trend.  :func:`beta_stability_boundary` uses the same form with
    0.028, i.e. the fraction rather than percent convention of $\beta_N$.
    Tracked in #350.

    Limitations
    -----------
    Not a stability calculation; use DCON or a Troyon-type $\beta_N \le C\,l_i$
    estimate for a physical limit.

    References
    ----------
    .. [1] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    """
    beta_N_crit = 2.8 * q_95
    return beta_N - beta_N_crit, beta_N_crit


def sawtooth_stability_criterion(q_0: float,
                               beta_pol: float) -> Tuple[float, float]:
    r"""Heuristic sawtooth margin against $\beta_{p,crit} = 0.3\,(1 - q_0)$.

    $$\Delta = \beta_p - \beta_{p,crit}, \qquad \beta_{p,crit} = 0.3\,(1 - q_0)$$

    Parameters
    ----------
    q_0 : float
        Safety factor on axis [-].
    beta_pol : float
        Poloidal beta [-].

    Returns
    -------
    margin : float
        $\beta_p - \beta_{p,crit}$; positive means the threshold is exceeded [-].
    beta_pol_crit : float
        The threshold [-].

    Validity
    --------
    Empirical fit.  Modelled on the Porcelli trigger, in which the internal-kink
    threshold scales with the poloidal beta inside the $q=1$ surface and a
    critical value of order 0.3 [1]_; the linear $(1 - q_0)$ dependence and
    the coefficient are VAFT heuristics without a recorded source.

    Limitations
    -----------
    Uses the global $\beta_p$, not $\beta_{p,1}$ inside $q=1$; returns a
    negative threshold for $q_0 > 1$, where sawteeth do not occur.  Tracked in
    #350.

    References
    ----------
    .. [1] F. Porcelli, D. Boucher and M. N. Rosenbluth, Plasma Phys. Control.
           Fusion 38 (1996) 2163, Sec. 3.
    """
    beta_pol_crit = 0.3 * (1 - q_0)
    return beta_pol - beta_pol_crit, beta_pol_crit


# ------------------------------------------------------------------
# Density Limits
# ------------------------------------------------------------------

def greenwald_density(I_p: float,
                      a: float) -> float:
    r"""Greenwald density limit $n_G$.

    $$n_G\,[10^{20}\,\mathrm{m^{-3}}] = \frac{I_p\,[\mathrm{MA}]}{\pi a^2\,[\mathrm{m^2}]}$$

    returned as $10\,I_p/(\pi a^2)$ in units of $10^{19}\,\mathrm{m^{-3}}$.

    Parameters
    ----------
    I_p : float
        Plasma current [MA].
    a : float
        Minor radius [m].

    Returns
    -------
    float
        Greenwald density limit [1e19 m^-3].

    Convention
    ----------
    Engineering units (MA, m) with the result in $10^{19}\,\mathrm{m^{-3}}$; the
    literature quotes $n_G$ in $10^{20}\,\mathrm{m^{-3}}$.  Pair with the
    line-averaged electron density when forming $f_G$
    (:func:`greenwald_fraction`).

    Physical interpretation
    -----------------------
    Operational density limit above which discharges typically disrupt through
    edge cooling and MARFE formation; not a hard MHD boundary.

    Validity
    --------
    Empirical fit.  Multi-machine ohmic and auxiliary-heated database,
    Greenwald et al. 1988 [1]_; reviewed with H-mode data in [2]_.  Peaked
    profiles can exceed $f_G = 1$.

    Limitations
    -----------
    No dependence on shaping, heating power or fuelling; spherical tokamaks
    routinely exceed it.

    References
    ----------
    .. [1] M. Greenwald et al., Nucl. Fusion 28 (1988) 2199, Eq. (1).
    .. [2] M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27, Sec. 2.
    """
    return 10.0 * I_p / (np.pi * a**2)


def greenwald_fraction(n_e: float,
                        n_G: float) -> float:
    r"""Greenwald fraction $f_G = n_e/n_G$.

    $$f_G = \frac{\bar n_e}{n_G}$$

    Parameters
    ----------
    n_e : float
        Line-averaged electron density [1e19 m^-3].
    n_G : float
        Greenwald density limit in the same unit [1e19 m^-3].

    Returns
    -------
    float
        Greenwald fraction [-].

    Convention
    ----------
    Both inputs in one unit and with the line-averaged density, the definition
    used in the Greenwald database; a volume average gives a systematically
    lower fraction.

    References
    ----------
    .. [1] M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27, Sec. 2.
    """
    return n_e / n_G


# ------------------------------------------------------------------
# Power Limits
# ------------------------------------------------------------------

def power_limit_from_beta(beta_N: float,
                         B0: float,
                         V: float) -> float:
    r"""Energy-like figure $\beta_N B_0^2 V/(2\mu_0)$ labelled a power limit.

    $$P = \beta_N\,\frac{B_0^2}{2\mu_0}\,V$$

    Parameters
    ----------
    beta_N : float
        Normalised beta, any convention [-].
    B0 : float
        Toroidal field on axis [T].
    V : float
        Plasma volume [m^3].

    Returns
    -------
    float
        The expression above, dimensionally an energy [J].

    Limitations
    -----------
    $\beta B_0^2V/2\mu_0$ is the stored energy at beta $\beta$
    (:func:`vaft.formula.equilibrium.stored_energy_from_beta_V`), not a power;
    no time scale enters, and no source records what limit was intended.  Kept
    for compatibility.  Tracked in #362.

    References
    ----------
    .. [1] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209 (the
           $\beta_N$ limit the expression appears to draw on).
    """
    return beta_N * B0**2 * V / (2 * MU0)


def power_limit_from_q(q_95: float,
                      I_p: float,
                      R0: float) -> float:
    r"""Expression $2\pi R_0 I_p/(\mu_0 q_{95})$ labelled a power limit.

    $$P = \frac{2\pi R_0\,I_p}{\mu_0\,q_{95}}$$

    Parameters
    ----------
    q_95 : float
        Safety factor at the 95% flux surface [-].
    I_p : float
        Plasma current [A].
    R0 : float
        Major radius [m].

    Returns
    -------
    float
        The expression above [A^2].

    Limitations
    -----------
    Dimensionally $I_p R/\mu_0 q$ is A$^2$ (current times $R/L$), not a power;
    no derivation or source is recorded.  Kept for compatibility.  Tracked in
    #362.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.4 (cylindrical $q$, the relation this seems to rearrange).
    """
    return 2 * np.pi * R0 * I_p / (MU0 * q_95)


# ------------------------------------------------------------------
# Stability Boundaries
# ------------------------------------------------------------------

def beta_stability_boundary(beta_N: float,
                            q_95: float) -> Tuple[float, float]:
    r"""Heuristic beta margin against $\beta_{N,crit} = 0.028\,q_{95}$.

    $$\Delta = \beta_N - \beta_{N,crit}, \qquad \beta_{N,crit} = 0.028\,q_{95}$$

    Parameters
    ----------
    beta_N : float
        Normalised beta as a fraction (Troyon's 2.8 % is 0.028) [-].
    q_95 : float
        Safety factor at the 95% flux surface [-].

    Returns
    -------
    margin : float
        $\beta_N - \beta_{N,crit}$; positive means the limit is exceeded [-].
    beta_N_crit : float
        The threshold $0.028\,q_{95}$ [-].

    Validity
    --------
    Empirical fit.  The fraction-convention twin of
    :func:`kink_stability_criterion` (2.8 versus 0.028); the Troyon coefficient
    [1]_ is the only sourced part, the $q_{95}$ factor is not.
    :func:`plasma_stability_margins` builds on this function, so it lives in
    the fraction convention.  Tracked in #350.

    References
    ----------
    .. [1] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    """
    beta_N_crit = 0.028 * q_95
    stab_margin = beta_N - beta_N_crit
    return stab_margin, beta_N_crit


def plasma_stability_margins(beta_N: float,
                             q_95: float,
                             n_e: float,
                             n_G: float) -> Tuple[float, float, float]:
    r"""Beta, $q_{95}$ and density margins in one call.

    $$\Delta_\beta = \beta_N - 0.028\,q_{95}, \qquad
      \Delta_q = q_{95} - 2, \qquad
      f_G = \frac{n_e}{n_G}$$

    Parameters
    ----------
    beta_N : float
        Normalised beta as a fraction [-].
    q_95 : float
        Safety factor at the 95% flux surface [-].
    n_e : float
        Line-averaged electron density [1e19 m^-3].
    n_G : float
        Greenwald density limit [1e19 m^-3].

    Returns
    -------
    beta_margin : float
        From :func:`beta_stability_boundary`; positive exceeds the limit [-].
    q_margin : float
        $q_{95} - 2$; negative is below the $q_{95} = 2$ disruption boundary [-].
    density_margin : float
        Greenwald fraction, from :func:`greenwald_fraction`; 1 is the limit [-].

    Convention
    ----------
    The three margins use three different sign conventions: beta and $q$ are
    differences (sign tells the side of the boundary) while density is a ratio.
    $q_{95} = 2$ is the empirical operational boundary [1]_.

    References
    ----------
    .. [1] J. A. Wesson et al., Nucl. Fusion 29 (1989) 641.
    .. [2] M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27.
    """
    beta_margin, _ = beta_stability_boundary(beta_N, q_95)
    q_margin = q_95 - 2.0  # Minimum q_95 for stability
    density_margin = greenwald_fraction(n_e, n_G)
    return beta_margin, q_margin, density_margin


# ------------------------------------------------------------------
# Transport
# ------------------------------------------------------------------

def collisionality_from_n_T_B_R(n_e: float,
                               T_e_keV: float,
                               B_t: float,
                               R0: float) -> float:
    r"""Electron collisionality figure $6.921\times10^{-18}\,n_e R_0/(T_e^2 B_t)$.

    $$\nu_* = 6.921\times10^{-18}\,\frac{n_e\,R_0}{T_e^2\,B_t}$$

    Parameters
    ----------
    n_e : float
        Electron density [1e19 m^-3].
    T_e_keV : float
        Electron temperature [keV].
    B_t : float
        Toroidal field [T].
    R0 : float
        Major radius [m].

    Returns
    -------
    float
        Collisionality figure in this function's own normalisation [-].

    Convention
    ----------
    The prefactor is Sauter's $\nu_{e*} = 6.921\times10^{-18}\,qR\,n_e
    Z_{\mathrm{eff}}\ln\Lambda/(T_e^2\varepsilon^{3/2})$ with $n_e$ in m^-3 and
    $T_e$ in eV [1]_.  This routine drops $q$, $\varepsilon^{-3/2}$,
    $Z_{\mathrm{eff}}$ and $\ln\Lambda$, divides by $B_t$ instead, and takes
    $n_e$ in $10^{19}$ m^-3 and $T_e$ in keV, so it is not Sauter's $\nu_*$ nor
    the IPB98 database $\nu_*$ it was labelled with, and it is not comparable
    to :func:`vaft.formula.equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I`.
    Tracked in #353.

    Limitations
    -----------
    Use only for relative trends within one dataset.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834,
           Eq. (18b).
    """
    return COLLISIONALITY_COEF * n_e * R0 / (T_e_keV**2 * B_t)


def v_alfven_from_B_n_mi(B: float,
                         n: float,
                         m_i: float = MI_P) -> float:
    r"""Alfven speed $v_A = B/\sqrt{\mu_0 n m_i}$.

    $$v_A = \frac{B}{\sqrt{\mu_0\,n\,m_i}}$$

    Parameters
    ----------
    B : float
        Magnetic field strength [T].
    n : float
        Ion number density [m^-3].
    m_i : float, optional
        Ion mass; default the proton mass [kg].

    Returns
    -------
    float
        Alfven speed [m/s].

    Assumptions
    -----------
    Single ion species, $n_i = n$; the mass density is $n m_i$ (electron mass
    neglected).

    References
    ----------
    .. [1] J. P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge
           University Press (2007), Sec. 10.5 (Alfven waves).
    .. [2] NRL Plasma Formulary (2019), p. 29.
    """
    return B / np.sqrt(MU0 * n * m_i)


def c_s_from_Te_Ti_mi(T_e_keV: float,
                      T_i_keV: float,
                      m_i: float = MI_P) -> float:
    r"""Isothermal ion sound speed $c_s = \sqrt{(T_e + T_i)/m_i}$.

    $$c_s = \sqrt{\frac{\gamma_eT_e + \gamma_iT_i}{m_i}}, \qquad \gamma_e = \gamma_i = 1$$

    Parameters
    ----------
    T_e_keV : float
        Electron temperature [keV].
    T_i_keV : float
        Ion temperature [keV].
    m_i : float, optional
        Ion mass; default the proton mass [kg].

    Returns
    -------
    float
        Ion sound speed [m/s].

    Convention
    ----------
    Isothermal closure ($\gamma = 1$ for both species); the adiabatic
    $\gamma_i = 3$ used for ion acoustic waves gives a speed larger by up to
    $\sqrt{(T_e + 3T_i)/(T_e + T_i)}$.

    References
    ----------
    .. [1] J. P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge
           University Press (2007), Sec. 10.4 (sound waves).
    .. [2] NRL Plasma Formulary (2019), p. 29.
    """
    Te_J = T_e_keV * 1e3 * QE
    Ti_J = T_i_keV * 1e3 * QE
    return np.sqrt((Te_J + Ti_J) / m_i)


# ------------------------------------------------------------------
# Operational Parameters
# ------------------------------------------------------------------

def rhostar_from_Te_a_Bt(Te_eV: float,
                         a_minor: float,
                         B_t: float) -> float:
    r"""Normalised electron gyroradius $\rho_e/a$.

    $$\rho_* = \frac{\rho_e}{a}, \qquad
      \rho_e = \frac{\sqrt{2 m_e e T_e}}{e B_t}
              = \sqrt{\frac{2m_e}{e}}\;\frac{\sqrt{T_e}}{B_t}$$

    Parameters
    ----------
    Te_eV : float
        Electron temperature [eV].
    a_minor : float
        Minor radius [m].
    B_t : float
        Toroidal field [T].

    Returns
    -------
    float
        Normalised electron gyroradius [-].

    Convention
    ----------
    Thermal speed $\sqrt{2T/m}$ and the toroidal field, normalised by the minor
    radius: the ITER Physics Basis definition [1]_, and the electron counterpart
    of :func:`vaft.formula.equilibrium.normalized_larmor_radius_from_M_T_a_Bt`,
    which this delegates to so the two cannot drift.

    Until #348 this multiplied by $a$ instead of dividing, omitted the constant
    $\sqrt{2m_e/e} = 3.37\times10^{-6}$ m T eV$^{-1/2}$, and ignored an
    ``m_e`` argument it accepted -- so the result was neither dimensionless nor
    proportional to $\rho_*$ across devices, and no rescaling recovered it. The
    dead ``m_e`` parameter is gone with the defect.

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2,
           Sec. 6 (dimensionless parameters).
    """
    from .equilibrium import normalized_larmor_radius_from_M_T_a_Bt

    return normalized_larmor_radius_from_M_T_a_Bt(ME, Te_eV, a_minor, B_t)
