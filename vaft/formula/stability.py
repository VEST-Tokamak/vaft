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

import warnings

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
    "delta_prime_from_outer_derivatives",
    "empirical_li_qa",
    "ggj_resistive_interchange_index_from_ideal",
    "greenwald_density",
    "greenwald_fraction",
    "helical_harmonic",
    "helical_phase",
    "island_pendulum_hamiltonian",
    "island_separatrix_half_width",
    "island_width_from_resonant_flux",
    "kink_stability_criterion",
    "li_from_qa_empirical",
    "plasma_stability_margins",
    "power_limit_from_beta",
    "power_limit_from_q",
    "resonant_flux_from_delta",
    "s_alpha_ballooning_stable",
    "s_alpha_marginal_alpha",
    "rhostar_from_Te_a_Bt",
    "sawtooth_stability_criterion",
    "slab_perturbed_flux",
    "s_alpha_ballooning_solution",
    "s_alpha_curvature_drive",
    "field_line_label",
    "v_alfven_from_B_n_mi",
    "shear_alfven_frequency",
    "magnetosonic_phase_speeds",
    "kadomtsev_mixing_radius",
    "ballooning_radial_wavenumber",
    "s_alpha_ballooning_eigenmode",
    "suydam_criterion",
    "mercier_criterion_circular",
    "ggj_ideal_interchange_index",
    "ggj_resistive_interchange_index",
    "magnetic_well_from_specific_volume",
    "bussac_poloidal_beta",
    "bussac_internal_kink_energy",
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
    r"""Vertices of the JET empirical $l_i$-$q_\psi$ stability boundary (Wesson 1989, Fig. 6).

    Eighteen $(q, l_i)$ pairs: at each integer $q$ from 2 to 10, the top and
    then the bottom of the vertical edge of one tooth of the saw-tooth
    "empirical stability boundary" -- the *lower* boundary of the JET operating
    space, below which rotating kink and double-tearing modes grow during the
    current rise.

    Returns
    -------
    qa : np.ndarray
        Edge safety factor of each point; the paper's $q_\psi$ [-].
    li : np.ndarray
        Internal inductance of each point, $2\int B_\theta^2 d\tau/(\mu_0^2 R I^2)$ [-].

    Physical interpretation
    -----------------------
    Between integers the boundary rises linearly from one tooth's bottom to
    the next tooth's top; each vertical drop sits at a rational edge $q$.
    Above $q \approx 6$ the bottoms saturate near $l_i \approx 0.3$. The
    *upper* boundary of Fig. 6 (density-limit disruptions) is not in these
    arrays.

    Validity
    --------
    Empirical fit. Approximate readings of a published figure (JET 1985-1988),
    not tabulated data. A 2026 audit (#1422) re-digitized Fig. 6 at 600 dpi:
    these values agree with it to about 0.03 in $l_i$. The registered
    ``vaft.formula.boundaries`` entries ``wesson_1989_jet_li_qpsi_lower`` and
    ``wesson_1989_jet_li_qpsi_upper`` carry the audited boundary with its
    provenance and quantity identity; prefer them.

    Limitations
    -----------
    JET-specific operating experience; a spherical tokamak reaches lower $q_a$
    and different $l_i$ ranges, so use only as a qualitative boundary.

    References
    ----------
    .. [1] J. A. Wesson et al., "Disruptions in JET", Nucl. Fusion 29 (1989)
           641, Fig. 6, p. 645 ($l_i$-$q$ diagram; Fig. 5 is the Hugill diagram).
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
    .. [1] J. A. Wesson et al., Nucl. Fusion 29 (1989) 641, Fig. 6 (lower boundary; see empirical_li_qa).
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

    Reduction
    ---------
    input: profile_1d
    output: profile_1d
    kind: differential
    locality: flux_surface_local
    role: stability_coordinate

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
    ($0.3 \lesssim s \lesssim 1.5$). The boundary itself, computed from the
    ballooning equation, is ``s_alpha_marginal_alpha``: it leaves this line
    at small shear and has a second-stable region beyond $\alpha_2(s)$.

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
    r"""Deprecated: heuristic margin against an unsourced $\beta_{N,crit} = 2.8\,q_{95}$; use the ``troyon`` boundary.

    Neither Troyon et al. [1]_ nor any other recorded source multiplies the
    $\beta_N$ limit by $q_{95}$ (#350). The sourced limit is
    ``vaft.formula.boundaries.get_boundary("troyon")``: Troyon's
    $(\beta A)_{max} \approx 2.2\,I_N$ (p. 214), i.e.
    $\beta_N \le 2.2\,\mu_0\cdot10^6 \approx 2.76$ %·m·T/MA with no $q_{95}$ factor.
    Results are unchanged; a ``DeprecationWarning`` is emitted.

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
    Empirical fit.  The coefficient 2.8 is the rounded Troyon limit
    $2.2\,\mu_0\cdot10^6 \approx 2.76$ [1]_ (p. 214, $(\beta A)_{max} \approx 2.2\,I_N$);
    the multiplication by $q_{95}$ has no source in the literature or the
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
    warnings.warn(
        "`kink_stability_criterion` is deprecated: its q_95 factor has no source (#350). Use "
        "`vaft.formula.boundaries.get_boundary('troyon')` (beta_N <= 2.2 mu0 1e6 ~ 2.76, %·m·T/MA).",
        DeprecationWarning,
        stacklevel=2,
    )
    beta_N_crit = 2.8 * q_95
    return beta_N - beta_N_crit, beta_N_crit


def sawtooth_stability_criterion(q_0: float,
                               beta_pol: float) -> Tuple[float, float]:
    r"""Deprecated: heuristic margin against an unsourced $\beta_{p,crit} = 0.3\,(1 - q_0)$; no replacement.

    Porcelli et al. [1]_ trigger the crash on conditions in the internal-kink
    energy $\delta\hat W$, the fast-ion precession frequency, the ion diamagnetic
    frequency (Eqs. 13-15) and, through Eq. (15a), the shear $s_1$ at $q = 1$, with $\beta_{p1}$
    defined inside the $q = 1$ surface (Eq. 12). None reduces to a threshold
    $0.3\,(1 - q_0)$ on the global $\beta_p$ (#350), so this has no drop-in
    replacement. Results are unchanged; a ``DeprecationWarning`` is emitted.

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
           Fusion 38 (1996) 2163, Sec. 3, Eqs. (12)-(15a).
    """
    warnings.warn(
        "`sawtooth_stability_criterion` is deprecated: 0.3 (1 - q_0) on the global beta_p has no source "
        "(#350); the Porcelli trigger is a set of conditions (PPCF 38 (1996) 2163, Eqs. 12-15), not a beta threshold.",
        DeprecationWarning,
        stacklevel=2,
    )
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

    Semantics
    ---------
    consumes: plasma_current, minor_radius

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

    Reduction
    ---------
    input: scalar_0d
    output: scalar_0d
    kind: dimensionless_normalization
    locality: global
    role: regime_coordinate

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
    $\beta B_0^2V/2\mu_0$ is the pressure volume integral $\langle p\rangle V$
    at beta $\beta$ (:func:`vaft.formula.equilibrium.stored_energy_from_beta_V`),
    two thirds of the thermal energy, and not a power;
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
    r"""Deprecated: fraction-convention twin of :func:`kink_stability_criterion`; use the ``troyon`` boundary.

    The $q_{95}$ factor has no source (#350). Since #349 $\beta_N$ is in
    %·m·T/MA, where the sourced limit is
    ``vaft.formula.boundaries.get_boundary("troyon")`` ($\approx 2.76$).
    Results are unchanged; a ``DeprecationWarning`` is emitted.

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
    warnings.warn(
        "`beta_stability_boundary` is deprecated: its q_95 factor has no source (#350). Use "
        "`vaft.formula.boundaries.get_boundary('troyon')` with beta_N in %·m·T/MA.",
        DeprecationWarning,
        stacklevel=2,
    )
    beta_N_crit = 0.028 * q_95
    stab_margin = beta_N - beta_N_crit
    return stab_margin, beta_N_crit


def plasma_stability_margins(beta_N: float,
                             q_95: float,
                             n_e: float,
                             n_G: float) -> Tuple[float, float, float]:
    r"""Deprecated: beta, $q_{95}$ and density margins built on the unsourced $0.028\,q_{95}$ beta limit.

    The beta margin comes from :func:`beta_stability_boundary`, whose $q_{95}$
    factor has no source (#350). Use ``vaft.formula.boundaries``:
    ``evaluate_boundary`` with ``"troyon"``, ``"low_q"`` and ``"greenwald"``
    gives each margin with its source and one sign convention. Results are
    unchanged; a ``DeprecationWarning`` is emitted.

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
    warnings.warn(
        "`plasma_stability_margins` is deprecated: its beta margin uses the unsourced 0.028 q_95 limit "
        "(#350). Use vaft.formula.boundaries.evaluate_boundary with 'troyon', 'low_q' and 'greenwald'.",
        DeprecationWarning,
        stacklevel=2,
    )
    beta_margin = beta_N - 0.028 * q_95  # beta_stability_boundary, without its own warning
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


def shear_alfven_frequency(k_parallel, v_A):
    r"""Frequency of the ideal shear Alfven wave in a uniform plasma.

    $$\omega^2 = k_\parallel^2v_A^2,\qquad \omega = |k_\parallel|\,v_A$$

    Parameters
    ----------
    k_parallel : float or np.ndarray
        Wavenumber along $\mathbf B_0$, signed [1/m].
    v_A : float or np.ndarray
        Alfven speed $B_0/\sqrt{\mu_0\rho}$ (``v_alfven_from_B_n_mi``), non-negative [m/s].

    Returns
    -------
    float or np.ndarray
        $\omega \ge 0$ [rad/s].

    Raises
    ------
    ValueError
        ``v_A`` is negative.

    Physical interpretation
    -----------------------
    Field-line bending restored by magnetic tension: the displacement and
    $\delta\mathbf B_\perp = -\delta\mathbf v_\perp\,B_0/v_A$ (forward wave)
    are perpendicular to both $\mathbf B_0$ and $\mathbf k$, and $|\mathbf B|$
    is unchanged to first order. $\omega$ depends on $k_\perp$ not at all, so
    energy travels along $\mathbf B_0$ only; in an inhomogeneous plasma this
    gives the Alfven continuum $\omega = k_\parallel(r)v_A(r)$.

    Assumptions
    -----------
    Uniform ideal MHD plasma, linear, $\omega \ll \Omega_i$; no finite
    Larmor radius or electron inertia (kinetic and inertial Alfven waves).

    References
    ----------
    .. [1] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
           Sec. 10.2.
    .. [2] H. Alfven, Nature 150, 405 (1942).
    """
    v_A = np.asarray(v_A, dtype=float)
    if np.any(v_A < 0.0):
        raise ValueError("v_A must be non-negative")
    return np.abs(np.asarray(k_parallel, dtype=float)) * v_A


def magnetosonic_phase_speeds(theta, v_A, c_s):
    r"""Fast and slow magnetosonic phase speeds at angle $\theta$ to $\mathbf B_0$.

    $$v_{f,s}^2 = \tfrac12\left[v_A^2 + c_s^2 \pm
      \sqrt{\left(v_A^2 + c_s^2\right)^2 - 4v_A^2c_s^2\cos^2\theta}\right]$$

    Parameters
    ----------
    theta : float or np.ndarray
        Angle between $\mathbf k$ and $\mathbf B_0$ [rad].
    v_A : float
        Alfven speed, non-negative [m/s].
    c_s : float
        Sound speed $\sqrt{\gamma p/\rho}$, non-negative [m/s].

    Returns
    -------
    tuple of (float or np.ndarray)
        $(v_f, v_s)$, $v_f \ge v_s \ge 0$ [m/s].

    Raises
    ------
    ValueError
        ``v_A`` or ``c_s`` is negative.

    Physical interpretation
    -----------------------
    The two compressive ideal-MHD branches, polarized in the
    $\mathbf k$-$\mathbf B_0$ plane. The fast wave is magnetic and thermal
    pressure acting together; at $\theta = 90^\circ$ it is
    $\sqrt{v_A^2 + c_s^2}$, and for $c_s \ll v_A$ it is the compressional
    Alfven wave $\omega \simeq kv_A$ with $\delta B_\parallel \ne 0$. The
    slow wave has them in antiphase and vanishes at $\theta = 90^\circ$. With
    the shear Alfven speed $v_A|\cos\theta|$ (``shear_alfven_frequency``$/k$)
    they are ordered $v_s \le v_A|\cos\theta| \le v_f$ at every angle -- the
    Friedrichs diagram.

    Assumptions
    -----------
    Uniform ideal MHD plasma, adiabatic, linear. Pass the adiabatic
    $c_s$; ``c_s_from_Te_Ti_mi`` is the isothermal ($\gamma = 1$) one.

    References
    ----------
    .. [1] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
           Sec. 10.2.
    .. [2] T. J. M. Boyd and J. J. Sanderson, *The Physics of Plasmas*,
           Cambridge University Press (2003), Sec. 4.8.
    """
    v_A = float(v_A)
    c_s = float(c_s)
    if v_A < 0.0 or c_s < 0.0:
        raise ValueError(f"v_A and c_s must be non-negative, not {v_A!r} and {c_s!r}")
    cos2 = np.cos(np.asarray(theta, dtype=float)) ** 2
    total = v_A * v_A + c_s * c_s
    root = np.sqrt(np.maximum(total * total - 4.0 * v_A * v_A * c_s * c_s * cos2, 0.0))
    fast = np.sqrt(0.5 * (total + root))
    # v_s^2 = v_A^2 c_s^2 cos^2 / v_f^2: no cancellation where the minus-sign form loses digits
    slow = np.sqrt(np.divide(v_A * v_A * c_s * c_s * cos2, fast * fast, out=np.zeros_like(fast), where=fast > 0.0))
    return fast, slow


def kadomtsev_mixing_radius(r, q):
    r"""Kadomtsev mixing radius: where the $m/n = 1/1$ helical flux returns to its axis value.

    $$\psi_*(r) \propto \int_0^r r'\left(\frac{1}{q(r')} - 1\right)dr',\qquad
      \psi_*(r_\mathrm{mix}) = \psi_*(0),\ r_\mathrm{mix} > r_1$$

    Parameters
    ----------
    r : np.ndarray
        Minor radius (or a radial label used as one), increasing from the
        axis, first point at or near zero [m or -].
    q : np.ndarray
        Safety factor on ``r``, below one on the axis [-].

    Returns
    -------
    float
        $r_\mathrm{mix}$, in the unit of ``r`` [m or -].

    Raises
    ------
    ValueError
        ``r`` and ``q`` differ in length or ``r`` is not increasing,
        $q(0) \ge 1$ (no $q = 1$ surface to reconnect), or $\psi_*$ does not
        return to zero within ``r``.

    Convention
    ----------
    Cylindrical helical flux of the 1/1 harmonic,
    $d\psi_*/dr = rB_z(1/q - 1)/R_0$, up to the constant factor that cancels
    in the root; it rises inside $q = 1$ ($r_1$) and falls outside it.
    The integral starts at ``r[0]``, so pass the axis.

    Physical interpretation
    -----------------------
    Full reconnection pairs each surface inside $r_1$ with the surface
    outside it of equal helical flux; the outermost pair is the axis and
    $r_\mathrm{mix}$. Everything inside $r_\mathrm{mix}$ is mixed and flattened
    and $q$ there is raised to about one. When $1/q - 1$ is parabolic,
    $\propto 1 - r^2/r_1^2$, $r_\mathrm{mix} = \sqrt 2\,r_1$ exactly; a
    parabolic $q$ gives a little more.

    Assumptions
    -----------
    Complete (Kadomtsev) reconnection in a cylinder; large aspect ratio.
    Many sawteeth reconnect only partly, so $r_\mathrm{mix}$ is an upper bound
    on the region a real crash flattens.

    Reduction
    ---------
    input: profile_1d
    output: scalar_0d
    kind: feature_extraction
    locality: global
    role: profile_descriptor

    References
    ----------
    .. [1] B. B. Kadomtsev, Sov. J. Plasma Phys. 1, 389 (1975).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.6.
    """
    r = np.asarray(r, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if r.shape != q.shape or r.size < 3 or np.any(np.diff(r) <= 0.0):
        raise ValueError("r must be increasing and the same length as q (at least 3 points)")
    if q[0] >= 1.0:
        raise ValueError(f"q on the axis is {q[0]:g} >= 1: there is no q = 1 surface to reconnect")
    integrand = r * (1.0 / q - 1.0)
    psi_star = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))])
    past = np.flatnonzero((psi_star[1:] <= 0.0) & (psi_star[:-1] > 0.0))
    if not past.size:
        raise ValueError("the helical flux does not return to its axis value within r: r_mix is beyond the range")
    i = int(past[0])
    return float(r[i] + (r[i + 1] - r[i]) * psi_star[i] / (psi_star[i] - psi_star[i + 1]))


# ------------------------------------------------------------------
# Local interchange criteria and the reduced internal kink (#1635)
# ------------------------------------------------------------------

def _finite(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, not {value!r}")
    return array


def _scalar_or_array(result):
    result = np.asarray(result, dtype=float)
    return float(result) if result.ndim == 0 else result


def suydam_criterion(r, B_z, q, dq_dr, dp_dr):
    r"""Suydam's local interchange criterion of a cylindrical screw pinch.

    $$D_S = \frac{rB_z^2}{8\mu_0}\left(\frac{q'}{q}\right)^2 + p'$$

    Parameters
    ----------
    r : float or np.ndarray
        Minor radius of the surface, positive [m].
    B_z : float or np.ndarray
        Axial field on the surface [T].
    q : float or np.ndarray
        Safety factor on the surface, non-zero [-].
    dq_dr : float or np.ndarray
        Radial derivative of $q$ [1/m].
    dp_dr : float or np.ndarray
        Radial derivative of the pressure [Pa/m].

    Returns
    -------
    float or np.ndarray
        $D_S$: positive where the criterion holds, negative where it is
        violated [Pa/m].

    Raises
    ------
    ValueError
        ``r`` is not positive, ``q`` is zero, or an input is not finite.

    Convention
    ----------
    Sign chosen so that $D_S > 0$ satisfies the criterion. $q$ enters only
    through the shear $q'/q$, so its sign convention does not matter.

    Physical interpretation
    -----------------------
    Field-line bending from magnetic shear (the first term, always
    stabilising) against the pressure gradient falling outward in the bad
    curvature of the azimuthal field ($p' < 0$). With no shear any outward
    pressure fall is unstable to localised interchanges.

    Assumptions
    -----------
    Ideal MHD, a straight cylinder (no toroidal curvature), perturbations
    localised about one surface ($m \to \infty$ with $k_\parallel = 0$ there).
    The criterion is **necessary** for stability: $D_S < 0$ proves a local
    instability, $D_S > 0$ proves nothing about global modes.

    References
    ----------
    .. [1] B. R. Suydam, Proc. 2nd UN Int. Conf. Peaceful Uses of Atomic
           Energy 31 (1958) 157.
    .. [2] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
           Ch. 11.
    """
    r = _finite(r, "r")
    q = _finite(q, "q")
    if np.any(r <= 0.0):
        raise ValueError(f"r must be positive, not {r!r}")
    if np.any(q == 0.0):
        raise ValueError("q must be non-zero")
    B_z, dq_dr, dp_dr = (_finite(v, n) for v, n in ((B_z, "B_z"), (dq_dr, "dq_dr"), (dp_dr, "dp_dr")))
    return _scalar_or_array(r * B_z ** 2 / (8.0 * MU0) * (dq_dr / q) ** 2 + dp_dr)


def mercier_criterion_circular(r, B_phi, q, dq_dr, dp_dr):
    r"""Mercier's interchange criterion for a large-aspect-ratio circular tokamak.

    $$D_M = \frac{rB_\phi^2}{8\mu_0}\left(\frac{q'}{q}\right)^2 + p'\,(1 - q^2)$$

    Parameters
    ----------
    r : float or np.ndarray
        Minor radius of the surface, positive [m].
    B_phi : float or np.ndarray
        Toroidal field on the surface [T].
    q : float or np.ndarray
        Safety factor on the surface, non-zero [-].
    dq_dr : float or np.ndarray
        Radial derivative of $q$ [1/m].
    dp_dr : float or np.ndarray
        Radial derivative of the pressure [Pa/m].

    Returns
    -------
    float or np.ndarray
        $D_M$: positive where the criterion holds, negative where it is
        violated [Pa/m].

    Raises
    ------
    ValueError
        ``r`` is not positive, ``q`` is zero, or an input is not finite.

    Convention
    ----------
    Sign chosen so that $D_M > 0$ satisfies the criterion, as for
    ``suydam_criterion``. It is not the GGJ index $D_I$, which is negative
    when stable (``ggj_ideal_interchange_index``). $D_M = D_S - p'q^2$ with
    $B_z \to B_\phi$.

    Physical interpretation
    -----------------------
    Toroidicity adds the average curvature of the torus to Suydam's
    cylinder. The extra $-p'q^2$ is the toroidal average-curvature
    (magnetic-well) contribution, Pfirsch--Schlüter currents included; it
    makes the pressure factor $1 - q^2$, which changes sign at $q = 1$. For
    $q > 1$ an outward pressure fall *stabilises* interchanges and the
    criterion holds even without shear.

    Assumptions
    -----------
    Ideal MHD; large aspect ratio $\epsilon = r/R_0 \ll 1$; circular,
    unshifted surfaces to the order kept; low $\beta$. Localised
    interchanges only: a **necessary** condition, not a global stability
    proof. For shaped or tight-aspect-ratio equilibria the general Mercier
    index from flux-surface averages is needed; DCON reports it as $D_I$.

    References
    ----------
    .. [1] C. Mercier, Nucl. Fusion 1 (1960) 47.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Ch. 6.
    """
    r = _finite(r, "r")
    q = _finite(q, "q")
    if np.any(r <= 0.0):
        raise ValueError(f"r must be positive, not {r!r}")
    if np.any(q == 0.0):
        raise ValueError("q must be non-zero")
    B_phi, dq_dr, dp_dr = (_finite(v, n) for v, n in ((B_phi, "B_phi"), (dq_dr, "dq_dr"), (dp_dr, "dp_dr")))
    return _scalar_or_array(r * B_phi ** 2 / (8.0 * MU0) * (dq_dr / q) ** 2 + dp_dr * (1.0 - q ** 2))


def ggj_ideal_interchange_index(E, F, H):
    r"""Glasser–Greene–Johnson ideal-interchange (Mercier) index $D_I$ from the layer coefficients.

    $$D_I = E + F + H - \tfrac{1}{4}$$

    Parameters
    ----------
    E : float or np.ndarray
        GGJ coefficient $E$ of the resonant surface [-].
    F : float or np.ndarray
        GGJ coefficient $F$ [-].
    H : float or np.ndarray
        GGJ coefficient $H$ [-].

    Returns
    -------
    float or np.ndarray
        $D_I$; negative where the Mercier criterion holds [-].

    Raises
    ------
    ValueError
        An input is not finite.

    Convention
    ----------
    GGJ sign: $D_I < 0$ is ideal-interchange stable, the opposite sign to
    ``mercier_criterion_circular``. The Mercier small-solution exponents
    are $-\tfrac12 \pm \sqrt{-D_I}$, real when $D_I < 0$. This is the
    convention of the ``di`` column DCON and RDCON write.

    Physical interpretation
    -----------------------
    The local ideal-interchange drive at a surface: pressure against
    average curvature ($E$, $F$) and the geodesic coupling $H$, measured
    against the shear, which the $-\tfrac14$ represents.

    Assumptions
    -----------
    Toroidal ideal MHD; $E$, $F$, $H$ are flux-surface quantities of the
    equilibrium at the surface. Local criterion only.

    References
    ----------
    .. [1] A. H. Glasser, J. M. Greene and J. L. Johnson, Phys. Fluids 18
           (1975) 875.
    """
    E, F, H = (_finite(v, n) for v, n in ((E, "E"), (F, "F"), (H, "H")))
    return _scalar_or_array(E + F + H - 0.25)


def ggj_resistive_interchange_index(E, F, H):
    r"""Glasser–Greene–Johnson resistive-interchange index $D_R$ from the layer coefficients.

    $$D_R = E + F + H^2$$

    Parameters
    ----------
    E : float or np.ndarray
        GGJ coefficient $E$ of the resonant surface [-].
    F : float or np.ndarray
        GGJ coefficient $F$ [-].
    H : float or np.ndarray
        GGJ coefficient $H$ [-].

    Returns
    -------
    float or np.ndarray
        $D_R$; negative where resistive interchanges are stable [-].

    Raises
    ------
    ValueError
        An input is not finite.

    Convention
    ----------
    $D_R = D_I + (H - \tfrac12)^2$ with $D_I$ from
    ``ggj_ideal_interchange_index``, so $D_R \ge D_I$: a surface can be
    Mercier stable ($D_I < 0$) and resistive-interchange unstable
    ($D_R > 0$). Notation varies between references; this is GGJ's, used by
    the ``dr`` column of DCON and RDCON.

    Physical interpretation
    -----------------------
    Resistivity lets the layer interchange without bending field lines,
    so the geodesic term $H$ enters squared instead of linearly and the
    shear no longer helps.

    Assumptions
    -----------
    Resistive MHD in the GGJ inner layer; the same flux-surface
    coefficients as $D_I$.

    References
    ----------
    .. [1] A. H. Glasser, J. M. Greene and J. L. Johnson, Phys. Fluids 18
           (1975) 875.
    """
    E, F, H = (_finite(v, n) for v, n in ((E, "E"), (F, "F"), (H, "H")))
    return _scalar_or_array(E + F + H ** 2)


def magnetic_well_from_specific_volume(flux, dV_dflux):
    r"""Normalised magnetic well from the specific volume $V'$ against toroidal flux.

    $$W = -\frac{\Phi}{V'}\,\frac{dV'}{d\Phi},\qquad V' = \frac{dV}{d\Phi}$$

    Parameters
    ----------
    flux : np.ndarray
        Toroidal flux $\Phi$ enclosed by each surface, monotonic outward,
        at least 3 points [Wb].
    dV_dflux : np.ndarray
        Specific volume $V' = dV/d\Phi$ on ``flux``, of one sign [m^3/Wb].

    Returns
    -------
    np.ndarray
        $W$: positive in a magnetic well, negative on a magnetic hill [-].

    Raises
    ------
    ValueError
        The arrays differ in length or have fewer than 3 points, ``flux`` is
        not monotonic, ``dV_dflux`` changes sign or is zero, ``flux`` and
        ``dV_dflux`` disagree in sign of travel, or an input is not finite.

    Convention
    ----------
    The flux label is the **toroidal** flux; with the poloidal flux the
    sign of $V''$ also carries the shear and is a different quantity.
    $W > 0$ ($V'' < 0$): $|B|$ rises outward on average, a well. A straight
    cylinder of uniform $B_z$ has $V' = 2\pi R_0/B_z$ constant and $W = 0$.
    $W$ is unchanged by $\Phi \to -\Phi$ (both $V'$ and $d\Phi$ flip), so a
    toroidal flux that a COCOS makes decrease outward, with $V' < 0$, is
    accepted as it is. The derivative is second-order finite differences,
    one-sided second order at the ends.

    Physical interpretation
    -----------------------
    Whether a flux tube that moves outward finds a stronger average field,
    which opposes interchange (a well), or a weaker one (a hill). It is the
    average-curvature ingredient of the Mercier criterion, a geometric
    diagnostic, not a stability boundary by itself.

    Assumptions
    -----------
    Nested flux surfaces; ``dV_dflux`` from the equilibrium's surface
    volumes. Only the sign and the normalisation are fixed here; the
    stabilising contribution in the Mercier criterion also depends on
    pressure and shear.

    References
    ----------
    .. [1] J. M. Greene, Comments Plasma Phys. Control. Fusion 17 (1997) 389.
    .. [2] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
           Ch. 12.
    """
    flux = _finite(flux, "flux").ravel()
    v_prime = _finite(dV_dflux, "dV_dflux").ravel()
    if flux.shape != v_prime.shape or flux.size < 3:
        raise ValueError("flux and dV_dflux must have the same length, at least 3 points")
    if np.all(v_prime < 0.0):  # a sign-flipped toroidal flux: W is invariant under Phi -> -Phi
        flux, v_prime = -flux, -v_prime
    if np.any(v_prime <= 0.0):
        raise ValueError("dV_dflux must be of one sign and non-zero")
    if np.any(np.diff(flux) <= 0.0):
        raise ValueError("flux must be monotonic, travelling outward the way dV_dflux says")
    return -flux / v_prime * np.gradient(v_prime, flux, edge_order=2)


def bussac_poloidal_beta(p_mean_inside, p_at_r1, B_theta_at_r1):
    r"""Bussac's poloidal beta inside the $q = 1$ surface.

    $$\beta_{p1} = \frac{2\mu_0}{B_{\theta 1}^2}\left(\langle p\rangle_1 - p(r_1)\right),
      \qquad \langle p\rangle_1 = \frac{2}{r_1^2}\int_0^{r_1} p\,r\,dr$$

    Parameters
    ----------
    p_mean_inside : float or np.ndarray
        Area-averaged pressure inside $r_1$, $\langle p\rangle_1$ [Pa].
    p_at_r1 : float or np.ndarray
        Pressure at the $q = 1$ surface [Pa].
    B_theta_at_r1 : float or np.ndarray
        Poloidal field at $r_1$, non-zero [T].

    Returns
    -------
    float or np.ndarray
        $\beta_{p1}$ [-].

    Raises
    ------
    ValueError
        ``B_theta_at_r1`` is zero or an input is not finite.

    Convention
    ----------
    The pressure *difference* enters, so a pedestal outside $r_1$ does not;
    the average is over the circular cross-section inside $r_1$.

    Physical interpretation
    -----------------------
    The pressure drive of the toroidal $m = n = 1$ internal kink: how far
    the core inside $q = 1$ is peaked above its edge value, against the
    poloidal field there. It is the variable of ``bussac_internal_kink_energy``.

    Assumptions
    -----------
    Large aspect ratio, circular cross-section.

    References
    ----------
    .. [1] M. N. Bussac, R. Pellat, D. Edery and J. L. Soulé, Phys. Rev.
           Lett. 35 (1975) 1638.
    """
    p_mean_inside, p_at_r1, B_theta_at_r1 = (
        _finite(v, n) for v, n in ((p_mean_inside, "p_mean_inside"), (p_at_r1, "p_at_r1"),
                                   (B_theta_at_r1, "B_theta_at_r1")))
    if np.any(B_theta_at_r1 == 0.0):
        raise ValueError("B_theta_at_r1 must be non-zero")
    return _scalar_or_array(2.0 * MU0 * (p_mean_inside - p_at_r1) / B_theta_at_r1 ** 2)


def bussac_internal_kink_energy(beta_p1, q0):
    r"""Reduced toroidal $m = n = 1$ internal-kink energy of Bussac et al., up to its positive prefactor.

    $$\delta\hat W_T = (1 - q_0)\left(\frac{13}{144} - \beta_{p1}^2\right)$$

    Parameters
    ----------
    beta_p1 : float or np.ndarray
        Poloidal beta inside $q = 1$, from ``bussac_poloidal_beta`` [-].
    q0 : float or np.ndarray
        Safety factor on the axis, a positive magnitude below one [-].

    Returns
    -------
    float or np.ndarray
        $\delta\hat W_T$: positive is ideal-stable, negative unstable [-].

    Raises
    ------
    ValueError
        ``q0`` is not in $(0, 1)$ or an input is not finite.

    Convention
    ----------
    $q_0$ is the magnitude: a COCOS that signs $q$ must be undone first, or
    $1 - q_0$ would be wrong. Only the sign and the $q_0$, $\beta_{p1}$ dependence are kept: the
    omitted prefactor ($\propto \epsilon_1^4\xi_0^2 B_\phi^2R_0/\mu_0$, with
    $\epsilon_1 = r_1/R_0$) is positive, so the sign is the verdict. The
    marginal value is $\beta_{p1} = \sqrt{13/144} \approx 0.30$.

    Physical interpretation
    -----------------------
    In a cylinder the $m = 1$ internal kink is marginal at leading order;
    toroidicity decides it. A weakly peaked core is stabilised by the
    $13/144$ term and a core with $\beta_{p1}$ above about 0.3 is
    destabilised.

    Assumptions
    -----------
    Ideal MHD; large aspect ratio, circular cross-section; the parabolic
    $q(r) = q_0 + (1 - q_0)r^2/r_1^2$ inside $r_1$ with $1 - q_0 \ll 1$, for
    which $13/144$ is evaluated. Other $q$ profiles change that constant;
    shaping and kinetic effects (Porcelli's $\delta W$ terms) are not
    included.

    References
    ----------
    .. [1] M. N. Bussac, R. Pellat, D. Edery and J. L. Soulé, Phys. Rev.
           Lett. 35 (1975) 1638.
    .. [2] F. Porcelli, D. Boucher and M. N. Rosenbluth, Plasma Phys.
           Control. Fusion 38 (1996) 2163.
    """
    beta_p1 = _finite(beta_p1, "beta_p1")
    q0 = _finite(q0, "q0")
    if np.any((q0 <= 0.0) | (q0 >= 1.0)):
        raise ValueError(f"q0 must be a positive magnitude below one (a q = 1 surface must exist), not {q0!r}; "
                         "pass |q| if a COCOS gives q a sign")
    return _scalar_or_array((1.0 - q0) * (13.0 / 144.0 - beta_p1 ** 2))


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

    Reduction
    ---------
    input: scalar_0d
    output: scalar_0d
    kind: dimensionless_normalization
    locality: global
    role: similarity_coordinate

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2,
           Sec. 6 (dimensionless parameters).
    """
    from .equilibrium import normalized_larmor_radius_from_M_T_a_Bt

    return normalized_larmor_radius_from_M_T_a_Bt(ME, Te_eV, a_minor, B_t)


# ------------------------------------------------------------------
# Resonant response at a rational surface
# ------------------------------------------------------------------


def island_width_from_resonant_flux(Phi_res,
                                    area,
                                    q,
                                    dq_dpsi_norm,
                                    m_pol,
                                    chi1: float):
    r"""Full magnetic island width from the pitch-resonant flux.

    $$w = 2\sqrt{\left|\frac{4\,\Phi_\mathrm{res}\,A}{2\pi\,s\,q\,\chi_1}\right|},
    \qquad s = \frac{m\,q'}{q^{2}}$$

    Parameters
    ----------
    Phi_res : complex or np.ndarray
        Pitch-resonant flux, normalised by the surface area [T].
    area : float or np.ndarray
        Area of the rational surface [m^2].
    q : float or np.ndarray
        Safety factor at the surface [-].
    dq_dpsi_norm : float or np.ndarray
        $\mathrm{d}q/\mathrm{d}\psi_N$ at the surface [-].
    m_pol : int or np.ndarray
        Poloidal mode number, $m = nq$ [-].
    chi1 : float
        $\mathrm{d}\chi/\mathrm{d}\psi_N$, the poloidal flux normalisation the
        equilibrium was solved with [-].

    Returns
    -------
    float or np.ndarray
        Full island width, in normalised poloidal flux [-].

    Convention
    ----------
    A **full** width, not a half width, and in $\psi_N$ rather than in metres
    -- matching what GPEC writes as ``w_isl`` (its ``units`` attribute reads
    ``psi_n``). Converting to metres needs the equilibrium's
    $\mathrm{d}r/\mathrm{d}\psi_N$ and is not done here.

    Physical interpretation
    -----------------------
    The island a resonant flux would open if it were not shielded. In an ideal
    solution it is not opened: the resonant component of the perturbed field
    is screened to zero at the surface, and $\Phi_\mathrm{res}$ is what the
    singular current would have to admit for the island to form.

    Assumptions
    -----------
    Constant-$\psi$, a single helicity, and a shear evaluated at the surface
    rather than across the island. The island is taken to be symmetric about
    the rational surface.

    Validity
    --------
    A rational surface with non-zero shear. At $q' \to 0$ the width diverges,
    which is the formula failing rather than the island growing.

    Numerical notes
    --------------
    The magnitude is taken before the square root, so a negative or complex
    argument does not propagate a ``nan``: the sign carries the island's
    phase, which this width does not report.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.4.
    .. [2] ``GPEC/gpec/gpout.f:1709-1711``, which this reproduces exactly on
           the DIII-D 147131 reference -- 16 rational surfaces over n = 1 and
           n = 3, ratio 1.000000 on every one.
    """
    shear = np.asarray(m_pol) * np.asarray(dq_dpsi_norm) / np.asarray(q) ** 2
    return 2.0 * np.sqrt(
        np.abs(4.0 * np.asarray(Phi_res) * np.asarray(area)
               / (2.0 * np.pi * shear * np.asarray(q) * chi1))
    )


def resonant_flux_from_delta(delta, geometric_factor, n_tor: int):
    r"""Pitch-resonant flux from the resonance parameter and the surface geometry.

    $$\Phi_\mathrm{res} = -\frac{G(\psi_\mathrm{res})}{n}\,\Delta$$

    Parameters
    ----------
    delta : complex or np.ndarray
        Unitless resonance parameter $\Delta$, the jump in the resonant
        field's $\psi$ derivative across the surface [-].
    geometric_factor : float or np.ndarray
        $G(\psi_\mathrm{res})$, the surface's own factor [T].
    n_tor : int
        Toroidal mode number [-].

    Returns
    -------
    complex or np.ndarray
        Pitch-resonant flux, normalised by the surface area [T].

    Raises
    ------
    ValueError
        ``n_tor`` is not a positive mode number.

    Convention
    ----------
    $G$ absorbs two equilibrium-only quantities GPEC forms separately: the
    surface integral $j_c$ of $B^{2}/|\nabla\psi|^{3}$, and the diagonal of
    the vacuum surface inductance that ``gpvacuum_flxsurf`` builds by calling
    the VACUUM code. Both depend on the flux surface and not on the
    perturbation, so $G$ is a property of the equilibrium and is measured
    once for it rather than rebuilt per run.

    Physical interpretation
    -----------------------
    $\Delta$ says how much the resonant field is discontinuous across the
    surface; $G$ converts that discontinuity into the flux the singular
    current drives. The minus sign is GPEC's: measured, the phase of
    $n\,\Phi_\mathrm{res}/\Delta$ is exactly 180 degrees.

    Assumptions
    -----------
    That $G$ was measured on the same equilibrium. It is not transferable
    between equilibria, and nothing here can detect a mismatch.

    Validity
    --------
    Measured on the DIII-D 147131 reference, $n\,\Phi_\mathrm{res}/\Delta$
    agrees across n = 1, 2 and 3 at every shared rational surface to within
    0.6 per cent, and the values the three modes report at *different*
    surfaces fall on one smooth curve in $q$ -- which is what makes $G$
    interpolable in $\psi$ from the surfaces any single run reaches.

    Limitations
    -----------
    The 0.6 per cent spread is systematic rather than noise: it grows
    monotonically from n = 1 to n = 3, which is the vacuum Green's function's
    own weak dependence on the toroidal mode number. A study needing better
    than that has to build the inductance rather than measure $G$.

    References
    ----------
    .. [1] ``GPEC/gpec/gpout.f:1690-1705`` for the chain
           $\Delta \to I_\mathrm{res} \to \Phi_\mathrm{res}$, and
           ``GPEC/gpec/gpvacuum.f:236-342`` for the inductance $G$ absorbs.
    """
    if int(n_tor) <= 0:
        raise ValueError(f"n_tor must be a positive mode number, not {n_tor!r}")
    return -np.asarray(geometric_factor) * np.asarray(delta) / float(n_tor)


# ------------------------------------------------------------------
# Local magnetic-island topology
# ------------------------------------------------------------------


def helical_phase(theta, phi, m_pol, n_tor, phase=0.0):
    r"""Helical phase of an $m/n$ perturbation at a point on a flux surface.

    $$\xi = m\,\theta - n\,\phi - \phi_0$$

    Parameters
    ----------
    theta : float or np.ndarray
        Poloidal angle [rad].
    phi : float or np.ndarray
        Toroidal angle [rad].
    m_pol : int
        Poloidal mode number [-].
    n_tor : int
        Toroidal mode number [-].
    phase : float or np.ndarray
        Phase offset $\phi_0$ of the perturbation [rad].

    Returns
    -------
    float or np.ndarray
        Helical phase $\xi$, not wrapped [rad].

    Raises
    ------
    ValueError
        ``m_pol`` or ``n_tor`` is not a positive mode number.

    Convention
    ----------
    $\theta$ is a straight-field-line poloidal angle (see
    ``vaft.formula.equilibrium.straight_field_line_angle``), increasing from
    the outboard midplane towards the top of the cross-section, and $\phi$
    increases counter-clockwise seen from above, so a field line of
    $q = m/n$ keeps $\xi$ constant. In a geometric poloidal angle it does
    not, even on circular surfaces. **Which** straight-field-line angle is
    not free once $\phi$ is the machine angle: Hamada, PEST and Boozer all
    make field lines straight, but each pairs its poloidal angle with its own
    toroidal angle, and PEST's is the machine $\phi$. So this expression is
    the helical phase in the PEST angle and in no other; a mesh in another
    one is relabelled first (see
    ``vaft.process.equilibrium.pest_angle_from_jacobian_angle``). At the
    resonant surface, where $m = nq$, the phase itself is the same in every
    straight-field-line system -- which is why a code may report a resonant
    harmonic's phase without naming its jacobian -- but the *position* that
    phase corresponds to at fixed machine $\phi$ is not. Both mode numbers are
    positive and the sign of the helicity sits in the minus sign; a
    perturbation of the opposite helicity is $n \to -n$ in this expression,
    not a negative argument. This $\phi$ is the IMAS one; only VEST's port
    clock numbering runs the other way (see
    ``vaft.machine_mapping.conventions``), which matters when a measured
    phase is converted, not for this definition.

    **$\phi_0$ from a resonant flux harmonic.** For GPEC's ``Phi_res`` (and
    its equal-argument ``I_res``) the O-point of the island sits at
    $\xi = 0$ with $\phi_0 = \arg\Phi_\mathrm{res} - \pi/2$, not at
    $\phi_0 = -\arg\Phi_\mathrm{res}$. ``Phi_res`` is a *normal-field*
    harmonic in tesla, while the field-line Hamiltonian's potential is the
    flux function, and harmonic $k$ of the two differ by $i\,k\,m$: the
    modulus is already inside the island width, the $1/i$ is this quarter
    period. Equivalently, the potential's own phase is
    $\delta = \pi/2 - \arg\Phi_\mathrm{res}$ and
    $V = -(w/4)^{2}\cos(\xi + \delta)$ puts its O-point at $\xi = -\delta$;
    a potential phase and an O-point location are the same statement with
    opposite sign, and quoting one as the other inverts the law.

    Measured rather than derived, on a FLARE Poincare trace of the DIII-D GPEC
    example at $q = 2$ on the vacuum $n = 1$ field: the traced O-point is
    2.1 degrees from where that law puts it and 128 degrees from where
    $\phi_0 = -\arg\Phi_\mathrm{res}$ does. The conjugation's sense follows
    the orientation of the code's angles against the machine helicity and was
    traced at $\mathrm{helicity} = -1$ only, so it is not a law to carry to
    the other sign without a trace there. Provenance (D-13, 2026-09-28):
    DIII-D 147131 @ 2300 ms, GPEC ``v1.5.5-378-gf06e6ab``, equilibrium
    ``g147131.02300_DIIID_KEFIT`` (SHA-256 ``35bf902f...bfe579``) and vacuum
    field ``gpec_cbrzphi_n1.out`` (SHA-256 ``718da784...05a8fa``), traced with
    FLARE ``7ad6d2dc``'s ``poincare_map_psiN`` over $\psi_N$ in
    [0.533644, 0.653644] -- the $q = 2$ surface plus and minus a 0.06 span --
    45 field lines $\times$ 300 punctures, ``nsym = 1``, on a single ``Gpec``
    element carrying that vacuum field at $|c_1| = 1$ with no $n = 3$ element.
    The full hashes and the fit that reads $\xi_O$ off the punctures are in
    ``test/test_gpec_island_geometry.py::R01_PROVENANCE``.

    Physical interpretation
    -----------------------
    The one coordinate a single-helicity perturbation depends on: every point
    with the same $\xi$ on the resonant surface sees the same perturbed field.
    O-points sit at $\xi = 0$ and X-points at $\xi = \pi$ (see
    ``island_pendulum_hamiltonian``), so a fixed-$\phi$ section shows $m$ of
    each and a fixed-$\theta$ trace shows $n$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.2.
    .. [2] R. Fitzpatrick, *Plasma Physics: An Introduction*, CRC Press
           (2014), Ch. 7 (magnetic islands).
    """
    if int(m_pol) <= 0 or int(m_pol) != m_pol:
        raise ValueError(f"m_pol must be a positive mode number, not {m_pol!r}")
    if int(n_tor) <= 0 or int(n_tor) != n_tor:
        raise ValueError(f"n_tor must be a positive mode number, not {n_tor!r}")
    return (int(m_pol) * np.asarray(theta, dtype=float)
            - int(n_tor) * np.asarray(phi, dtype=float)
            - np.asarray(phase, dtype=float))


def helical_harmonic(b_hat, theta, phi, m_pol, n_tor):
    r"""The real perturbation carried by one complex $m/n$ Fourier coefficient.

    $$\delta b = \mathrm{Re}\left[\hat b\,e^{i(m\theta - n\phi)}\right]
      = b_R\cos\xi - b_I\sin\xi$$

    Parameters
    ----------
    b_hat : complex or np.ndarray
        Complex harmonic coefficient $\hat b = b_R + i\,b_I$ [B].
    theta : float or np.ndarray
        Poloidal angle [rad].
    phi : float or np.ndarray
        Toroidal angle [rad].
    m_pol : int
        Poloidal mode number [-].
    n_tor : int
        Toroidal mode number [-].

    Returns
    -------
    float or np.ndarray
        The physical, real perturbation at $(\theta, \phi)$, in the unit of
        ``b_hat`` [B].

    Raises
    ------
    ValueError
        ``m_pol`` or ``n_tor`` is not a positive mode number.

    Convention
    ----------
    The phase is ``helical_phase`` with $\phi_0 = 0$, $\xi = m\theta - n\phi$:
    both mode numbers positive, the helicity in the minus sign, and
    $\hat b$ carrying the amplitude $|\hat b|$ and the phase
    $\arg\hat b$ of the pattern. A crest ($\delta b = |\hat b|$) sits where
    $\xi = -\arg\hat b$. A coefficient taken with the kernel
    $e^{-in\phi}$ over a real pattern -- ``toroidal_mode_decomposition``'s
    $C_n$, for which $A\cos(n\phi + \delta)$ gives $(A/2)e^{+i\delta}$ --
    is the conjugate of this one: $\hat b = 2\,\overline{C_n}$. GPEC's
    spectral output matches $e^{-in\phi}$ as written here, while its
    real-space $\theta$-functions (``*_fun``) are stored as
    $(\mathrm{Re}, -h\,\mathrm{Im})$ with the helicity $h$ (see
    ``vaft.machine_mapping.conventions``). A stored pair is converted by the
    rule of its source, never reinterpreted.

    Physical interpretation
    -----------------------
    A magnetic perturbation is real. Its complex coefficient is bookkeeping:
    $b_R$ and $b_I$ are the cosine and sine quadratures of one pattern, not
    two fields, and only $|\hat b|$ and phases relative to a stated
    origin are independent of the reference.

    Assumptions
    -----------
    One harmonic; a field with several is the sum of this over $m$ (and $n$).

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.2.
    .. [2] J.-K. Park and N. C. Logan, Phys. Plasmas 24 (2017) 032505
           (GPEC).
    """
    xi = helical_phase(theta, phi, m_pol, n_tor)
    result = np.real(np.asarray(b_hat, dtype=complex) * np.exp(1j * xi))
    return float(result) if np.ndim(result) == 0 else result


def island_pendulum_hamiltonian(x, xi, width):
    r"""Local flux function of a constant-$\psi$ magnetic island.

    $$H(x, \xi) = \tfrac{1}{2}x^{2} - \left(\frac{w}{4}\right)^{2}\cos\xi$$

    Parameters
    ----------
    x : float or np.ndarray
        Radial distance from the rational surface, $r - r_s$ [L].
    xi : float or np.ndarray
        Helical phase, as ``helical_phase`` returns it [rad].
    width : float
        Full island width at the O-point [L].

    Returns
    -------
    float or np.ndarray
        Helical flux function, in the square of the unit of ``x`` [L^2].

    Raises
    ------
    ValueError
        ``width`` is not positive.

    Convention
    ----------
    Normalised so that ``width`` is the **full** radial width at the O-point
    and the separatrix is the level $H = (w/4)^{2}$. The minimum
    $H = -(w/4)^{2}$ at $x = 0$, $\xi = 0$ is the O-point; the saddle at
    $x = 0$, $\xi = \pi$ is the X-point. $x$ and $w$ share one length unit,
    which the function never converts.

    Physical interpretation
    -----------------------
    The helical flux a single resonant harmonic leaves near its rational
    surface: a linear shear of the helical field plus a $\cos\xi$ ripple.
    Its level sets are the perturbed flux surfaces -- closed around the
    O-points inside the separatrix, open and rippled outside it. It is the
    phase portrait of a pendulum, which is where the name comes from.

    Assumptions
    -----------
    Constant-$\psi$ and a single helicity, with the shear taken constant
    across the island, which is what makes the island symmetric about the
    rational surface.

    Validity
    --------
    An island narrow compared with the distance to the neighbouring rational
    surfaces and to the plasma boundary. Overlapping islands (Chirikov
    parameter near one) are outside it.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.4.
    .. [2] R. Fitzpatrick, *Plasma Physics: An Introduction*, CRC Press
           (2014), Ch. 7 (magnetic islands).
    """
    width = float(width)
    if not width > 0.0:
        raise ValueError(f"width must be positive, not {width!r}")
    x = np.asarray(x, dtype=float)
    return 0.5 * x ** 2 - (width / 4.0) ** 2 * np.cos(np.asarray(xi, dtype=float))


def island_separatrix_half_width(xi, width):
    r"""Radial half-width of the island separatrix at a given helical phase.

    $$x_\mathrm{sep}(\xi) = \frac{w}{2}\,\left|\cos\frac{\xi}{2}\right|$$

    Parameters
    ----------
    xi : float or np.ndarray
        Helical phase, as ``helical_phase`` returns it [rad].
    width : float
        Full island width at the O-point [L].

    Returns
    -------
    float or np.ndarray
        Distance from the rational surface to the separatrix, the same on
        both sides [L].

    Raises
    ------
    ValueError
        ``width`` is not positive.

    Convention
    ----------
    The separatrix of ``island_pendulum_hamiltonian``: the level
    $H = (w/4)^{2}$ solved for $x$. It is $w/2$ at the O-point ($\xi = 0$),
    so the full width there is ``width``, and zero at the X-point
    ($\xi = \pi$), where the two branches cross.

    Physical interpretation
    -----------------------
    The boundary between field lines trapped in the island and the ones that
    pass it. The separatrix is $2\pi$-periodic in $\xi$, so a fixed-$\phi$
    section closes into $m$ lobes.

    Assumptions
    -----------
    The same constant-$\psi$, single-helicity island as
    ``island_pendulum_hamiltonian``.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.4.
    .. [2] R. Fitzpatrick, *Plasma Physics: An Introduction*, CRC Press
           (2014), Ch. 7 (magnetic islands).
    """
    width = float(width)
    if not width > 0.0:
        raise ValueError(f"width must be positive, not {width!r}")
    return 0.5 * width * np.abs(np.cos(0.5 * np.asarray(xi, dtype=float)))



def delta_prime_from_outer_derivatives(psi_s, dpsi_dr_minus, dpsi_dr_plus):
    r"""Tearing stability index from the outer solution's derivatives at the rational surface.

    $$\Delta' = \frac{1}{\tilde\psi(r_s)}\left(\left.\frac{d\tilde\psi}{dr}\right|_{r_s^+}
      - \left.\frac{d\tilde\psi}{dr}\right|_{r_s^-}\right)$$

    Parameters
    ----------
    psi_s : float or np.ndarray
        Perturbed poloidal flux of the outer solution at the rational
        surface, where both sides meet [Wb].
    dpsi_dr_minus : float or np.ndarray
        Radial derivative of the inner-side outer solution as $r \to r_s^-$ [Wb/m].
    dpsi_dr_plus : float or np.ndarray
        Radial derivative of the outer-side outer solution as $r \to r_s^+$ [Wb/m].

    Returns
    -------
    float or np.ndarray
        $\Delta'$, the jump in the logarithmic derivative [1/m].

    Raises
    ------
    ValueError
        ``psi_s`` is zero or not finite.

    Convention
    ----------
    $r$ increases outward, so the jump is the outer-side derivative minus
    the inner-side one. $\Delta' > 0$ is the classical tearing drive; the
    index is unchanged by the normalisation of $\tilde\psi$, and $r\Delta'$
    is its dimensionless form. Any flux unit works if the derivatives share
    it.

    Physical interpretation
    -----------------------
    The free energy the ideal outer region offers a reconnecting layer at
    $r_s$: the two outer solutions are continuous there but their slopes
    are not, and only non-ideal physics in a thin layer can bridge the jump.

    Assumptions
    -----------
    $\tilde\psi$ is continuous across $r_s$ with a jump only in its
    derivative, the outer solutions being those of ideal, marginally stable
    MHD. Nothing else is needed for the definition, which solves no outer
    equation. What makes $\Delta'$ the quantity a layer matches is separate:
    a layer thin compared with $r_s$ and, for the constant-$\psi$ regime,
    $\Delta'\delta \ll 1$ across its width $\delta$.

    References
    ----------
    .. [1] H. P. Furth, J. Killeen and M. N. Rosenbluth, Phys. Fluids 6
           (1963) 459.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.8.
    """
    psi_s = np.asarray(psi_s, dtype=float)
    if not (np.all(np.isfinite(psi_s)) and np.all(psi_s != 0.0)):
        raise ValueError(f"psi_s must be finite and non-zero, not {psi_s!r}")
    jump = np.asarray(dpsi_dr_plus, dtype=float) - np.asarray(dpsi_dr_minus, dtype=float)
    result = jump / psi_s
    return float(result) if result.ndim == 0 else result


# ------------------------------------------------------------------
# s-alpha ballooning stability
# ------------------------------------------------------------------

#: Newcomb integration of the s-alpha equation: range and fixed RK4 step [rad].
#: Doubling the range or halving the step moves the boundaries by < 0.01.
_S_ALPHA_THETA_MAX = 40.0 * np.pi
_S_ALPHA_STEP = 0.02


def slab_perturbed_flux(x, y, shear, amplitude, k_y, parity="tearing"):
    r"""Helical flux of a sheared slab with a tearing- or twisting-parity perturbation.

    $$\Psi_T = \frac{B_s'}{2}x^2 + \psi_0\cos k_y y,\qquad
      \Psi_W = \frac{B_s'}{2}x^2 + \psi_1\,x\cos k_y y$$

    Parameters
    ----------
    x : float or np.ndarray
        Distance from the rational surface [m].
    y : float or np.ndarray
        Binormal coordinate [m].
    shear : float
        $B_s' = dB_y/dx$ at the rational surface, non-zero [T/m].
    amplitude : float
        $\psi_0$ (tearing) or $\psi_1$ (twisting); the units follow $\Psi$ [T m or T].
    k_y : float
        Binormal wavenumber $m/r_s$ [1/m].
    parity : str
        ``"tearing"`` (even $\tilde\psi$) or ``"twisting"`` (odd $\tilde\psi$) [-].

    Returns
    -------
    float or np.ndarray
        $\Psi$; field lines lie on its contours [T m].

    Raises
    ------
    ValueError
        ``shear`` is zero or ``parity`` is unknown.

    Convention
    ----------
    $\mathbf B_\perp = \hat{\mathbf z}\times\nabla\Psi$ in the slab frame of
    ``sheared_slab_field`` (with $B_y = B_s'x$), so $\delta B_x = -\partial_y\tilde\psi$.
    Tearing parity has $\tilde\psi(-x) = \tilde\psi(x)$ and $\delta B_x(0) \ne 0$;
    twisting parity has $\tilde\psi(-x) = -\tilde\psi(x)$ and $\delta B_x(0) = 0$.
    $\Psi_T/B_s'$ is ``island_pendulum_hamiltonian`` with $\xi = k_yy + \pi$ and
    full width $w = 4\sqrt{|\psi_0/B_s'|}$; the O-points sit where
    $\cos k_yy = -\mathrm{sgn}(\psi_0/B_s')$. ``shear`` may be negative -- the
    slab of ``local_slab_from_cylinder`` has $L_s < 0$ for positive shear.

    Physical interpretation
    -----------------------
    Tearing parity reconnects flux across the rational surface and opens a
    magnetic island (O- and X-points, separatrix); its displacement
    $\xi_x = -\tilde\psi/(B_s'x)$ is odd in $x$. Twisting parity has no normal
    field on the rational surface ($k_\parallel = 0$ there), no reconnection,
    and an even displacement $\xi_x = -\psi_1\cos k_yy/B_s'$: the surfaces on
    both sides, and the rational surface with them, move together.

    Assumptions
    -----------
    Constant shear across the layer, a single helicity, the perturbation's
    radial structure taken as its leading term at $x = 0$ ($\psi_0$, or $\psi_1x$).
    Linear in the amplitude: contours of $\Psi_W$ close to $x = 0$ form thin
    cells of width $O(\psi_1/B_s')$ that are an artefact of dropping the
    $O(\psi_1^2)$ term; $\tfrac12B_s'(x + \psi_1\cos k_yy/B_s')^2$ completes it.

    References
    ----------
    .. [1] H. P. Furth, J. Killeen and M. N. Rosenbluth, Phys. Fluids 6
           (1963) 459.
    .. [2] R. Fitzpatrick, *Plasma Physics: An Introduction*, CRC Press
           (2014), Ch. 7.
    """
    shear = float(shear)
    if shear == 0.0 or not np.isfinite(shear):
        raise ValueError(f"shear must be finite and non-zero, not {shear!r}")
    if parity not in ("tearing", "twisting"):
        raise ValueError(f"parity must be 'tearing' or 'twisting', not {parity!r}")
    x = np.asarray(x, dtype=float)
    wave = float(amplitude) * np.cos(float(k_y) * np.asarray(y, dtype=float))
    result = 0.5 * shear * x * x + (wave if parity == "tearing" else x * wave)
    return float(result) if np.ndim(result) == 0 else result


def s_alpha_ballooning_stable(s, alpha, theta_max=_S_ALPHA_THETA_MAX, step=_S_ALPHA_STEP):
    r"""Ideal ballooning stability of the circular $s$-$\alpha$ model.

    $$\frac{\mathrm{d}}{\mathrm{d}\theta}\left[(1+\Lambda^{2})\frac{\mathrm{d}F}{\mathrm{d}\theta}\right]
    + \alpha\,(\cos\theta + \Lambda\sin\theta)\,F = 0,
    \qquad \Lambda = s\theta - \alpha\sin\theta$$

    Parameters
    ----------
    s : float or np.ndarray
        Magnetic shear $r q'/q$ [-].
    alpha : float or np.ndarray
        Normalised pressure gradient $-2\mu_0 R q^{2} p'/B^{2}$ [-].
    theta_max : float
        Extent of the ballooning-angle integration [rad].
    step : float
        Fixed fourth-order Runge-Kutta step [rad].

    Returns
    -------
    bool or np.ndarray
        True where the marginal ballooning equation is stable, broadcast over
        ``s`` and ``alpha`` [-].

    Raises
    ------
    ValueError
        ``theta_max`` or ``step`` is not positive.

    Convention
    ----------
    Connor-Hastie-Taylor normalisation: $\alpha > 0$ for pressure falling
    outwards (``ballooning_alpha_from_p_B_R``), $s > 0$ for $q$ rising
    outwards. Newcomb's criterion on the even solution, $F(0) = 1$,
    $F'(0) = 0$: the point is unstable if $F$ crosses zero anywhere on
    $0 < \theta \le \theta_\mathrm{max}$.

    Physical interpretation
    -----------------------
    $\alpha\cos\theta$ is the bad-curvature drive on the outboard side and
    $\Lambda$ the local shear that bends field lines against it. Raising
    $\alpha$ first destabilises (first stability boundary) and then, through
    the $-\alpha\sin\theta$ term that makes the local shear vanish on the
    outboard side and reverse elsewhere, restabilises (second stability).

    Assumptions
    -----------
    Large aspect ratio, circular shifted flux surfaces, infinite toroidal
    mode number, marginal stability ($\omega^{2} = 0$).

    Validity
    --------
    $s \gtrsim 0.05$. Towards zero shear the solutions decay only
    algebraically and a finite ``theta_max`` resolves the boundary
    progressively worse; at $s \le 0$ the model is stable.

    Numerical notes
    ---------------
    A fixed-step integrator, vectorised over the broadcast inputs, so the
    result is deterministic and a whole $(s, \alpha)$ grid costs one pass.
    With the defaults, halving ``step`` does not move the boundary at the
    1e-4 level; extending ``theta_max`` to $2560\pi$ moves $\alpha_1$ by
    about 0.012 at $s = 0.1$ and 0.002 at $s = 1$.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40,
           396 (1978).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.13.
    """
    if not theta_max > 0.0 or not step > 0.0:
        raise ValueError("theta_max and step must be positive")
    s_arr, a_arr = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(alpha, dtype=float))
    F = np.ones(s_arr.shape)
    G = np.zeros(s_arr.shape)  # (1 + Lambda^2) dF/dtheta
    stable = np.ones(s_arr.shape, dtype=bool)

    def rhs(theta, F, G):
        lam = s_arr * theta - a_arr * np.sin(theta)
        return G / (1.0 + lam * lam), -a_arr * (np.cos(theta) + lam * np.sin(theta)) * F

    h = float(step)
    theta = 0.0
    for _ in range(int(np.ceil(theta_max / h))):
        k1F, k1G = rhs(theta, F, G)
        k2F, k2G = rhs(theta + h / 2, F + h / 2 * k1F, G + h / 2 * k1G)
        k3F, k3G = rhs(theta + h / 2, F + h / 2 * k2F, G + h / 2 * k2G)
        k4F, k4G = rhs(theta + h, F + h * k3F, G + h * k3G)
        F = F + h / 6 * (k1F + 2 * k2F + 2 * k3F + k4F)
        G = G + h / 6 * (k1G + 2 * k2G + 2 * k3G + k4G)
        theta += h
        stable &= F > 0.0
    return stable if stable.ndim else bool(stable)


def field_line_label(phi, theta, q):
    r"""Clebsch field-line label on a flux surface in straight-field-line coordinates.

    $$\alpha = \phi - q\,\theta$$

    Parameters
    ----------
    phi : float or np.ndarray
        Toroidal angle [rad].
    theta : float or np.ndarray
        Straight-field-line poloidal angle, e.g. PEST $\theta^*$ [rad].
    q : float or np.ndarray
        Safety factor of the surface, a positive magnitude [-].

    Returns
    -------
    float or np.ndarray
        $\alpha$, constant along each field line; not wrapped [rad].

    Convention
    ----------
    Along a field line $d\phi/d\theta = q$, so $\alpha$ is constant. With
    ``helical_phase``'s angles ($\theta$ counter-clockwise from the outboard
    midplane, $\phi$ counter-clockwise from above: $(\psi, \theta, \phi)$
    left-handed), $\psi$ rising outward and $\mathbf B$ along $+\phi$,
    $+\theta$, the Clebsch form is $\mathbf B \propto \nabla\psi\times\nabla\alpha$;
    in right-handed coordinates (e.g. $\theta$ clockwise) it is
    $\nabla\alpha\times\nabla\psi$, the Connor--Hastie--Taylor form.
    For a rational $q = m/n$ it relates to ``helical_phase`` by
    $\alpha = -\xi/n$ ($\phi_0 = 0$): lines of one $\alpha$ are lines of one
    helical phase.

    Physical interpretation
    -----------------------
    Together with $\psi$ it names a field line: the line is the intersection
    of the surfaces $\psi = $ const and $\alpha = $ const. It is the natural
    binormal coordinate of field-aligned, ballooning and flux-tube models.

    Assumptions
    -----------
    Nested flux surfaces with a straight-field-line angle; undefined at
    separatrices and in islands or stochastic regions.

    References
    ----------
    .. [1] W. D. D'haeseleer, W. N. G. Hitchon, J. D. Callen and
           J. L. Shohet, *Flux Coordinates and Magnetic Field Structure*,
           Springer (1991), Ch. 4 and 6.
    """
    result = np.asarray(phi, dtype=float) - np.asarray(q, dtype=float) * np.asarray(theta, dtype=float)
    return float(result) if np.ndim(result) == 0 else result


def s_alpha_curvature_drive(theta, s, alpha, theta0=0.0):
    r"""Normal-curvature drive of the $s$-$\alpha$ ballooning equation along the extended angle.

    $$K(\theta) = \cos\theta + \Lambda\sin\theta,\qquad
      \Lambda = s(\theta - \theta_0) - \alpha(\sin\theta - \sin\theta_0)$$

    Parameters
    ----------
    theta : float or np.ndarray
        Extended ballooning angle, not restricted to one period [rad].
    s : float
        Magnetic shear [-].
    alpha : float
        Normalised pressure gradient [-].
    theta0 : float
        Ballooning angle $\theta_0$ (radial-wavenumber parameter) [rad].

    Returns
    -------
    float or np.ndarray
        $K$; the drive $\alpha K F$ is destabilising where $K > 0$ [-].

    Convention
    ----------
    The coefficient of $\alpha F$ in the equation of
    ``s_alpha_ballooning_stable`` (there $\theta_0 = 0$). $\cos\theta$ is the
    normal curvature -- bad (positive) on the outboard side -- and
    $\Lambda\sin\theta$ the geodesic curvature weighted by the local shear.

    Physical interpretation
    -----------------------
    Where the mode sits along the field line decides whether pressure
    drives it: it balloons where $K > 0$, on the outboard, bad-curvature side,
    and is stabilised by field-line bending elsewhere. $\theta_0$ slides the
    point of zero local shear along the line.

    Assumptions
    -----------
    As ``s_alpha_ballooning_stable``: large aspect ratio, circular shifted
    surfaces, infinite toroidal mode number.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40,
           396 (1978).
    """
    theta = np.asarray(theta, dtype=float)
    lam = float(s) * (theta - float(theta0)) - float(alpha) * (np.sin(theta) - np.sin(float(theta0)))
    result = np.cos(theta) + lam * np.sin(theta)
    return float(result) if np.ndim(result) == 0 else result


def s_alpha_ballooning_solution(s, alpha, theta_max=8.0 * np.pi, step=_S_ALPHA_STEP):
    r"""The even solution $F(\theta)$ of the marginal $s$-$\alpha$ ballooning equation.

    $$\frac{\mathrm{d}}{\mathrm{d}\theta}\left[(1+\Lambda^{2})\frac{\mathrm{d}F}{\mathrm{d}\theta}\right]
    + \alpha\,(\cos\theta + \Lambda\sin\theta)\,F = 0,\qquad F(0) = 1,\ F'(0) = 0$$

    Parameters
    ----------
    s : float
        Magnetic shear [-].
    alpha : float
        Normalised pressure gradient [-].
    theta_max : float
        End of the extended-angle interval [rad].
    step : float
        Fixed fourth-order Runge-Kutta step [rad].

    Returns
    -------
    theta : np.ndarray
        Extended angle from 0 in steps of ``step``, to the first step at or
        beyond ``theta_max`` [rad].
    F : np.ndarray
        The solution; it is even, so $F(-\theta) = F(\theta)$ [-].

    Raises
    ------
    ValueError
        ``theta_max`` or ``step`` is not positive.

    Convention
    ----------
    The same equation, normalisation and integrator as
    ``s_alpha_ballooning_stable``, returning the path instead of the verdict:
    by Newcomb's criterion the surface is unstable exactly when this $F$
    crosses zero.

    Physical interpretation
    -----------------------
    The marginal ($\omega^2 = 0$) solution along the field line on the
    extended angle -- not a localised eigenfunction: a stable $F$ grows
    without decaying. A zero crossing means a localised perturbation can
    release energy.

    Assumptions
    -----------
    As ``s_alpha_ballooning_stable``.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40,
           396 (1978).
    """
    if not theta_max > 0.0 or not step > 0.0:
        raise ValueError("theta_max and step must be positive")
    s, a = float(s), float(alpha)

    def rhs(theta, F, G):
        lam = s * theta - a * np.sin(theta)
        return G / (1.0 + lam * lam), -a * (np.cos(theta) + lam * np.sin(theta)) * F

    h = float(step)
    n = int(np.ceil(theta_max / h))
    thetas = np.arange(n + 1) * h
    F, G = 1.0, 0.0
    out = [F]
    for i in range(n):
        theta = thetas[i]
        k1F, k1G = rhs(theta, F, G)
        k2F, k2G = rhs(theta + h / 2, F + h / 2 * k1F, G + h / 2 * k1G)
        k3F, k3G = rhs(theta + h / 2, F + h / 2 * k2F, G + h / 2 * k2G)
        k4F, k4G = rhs(theta + h, F + h * k3F, G + h * k3G)
        F = F + h / 6 * (k1F + 2 * k2F + 2 * k3F + k4F)
        G = G + h / 6 * (k1G + 2 * k2G + 2 * k3G + k4G)
        out.append(F)
    return thetas, np.array(out)



def ballooning_radial_wavenumber(k_y, s, theta, alpha=0.0, theta0=0.0):
    r"""Radial wavenumber of a ballooning mode along the field line, in the $s$-$\alpha$ local frame.

    $$k_x = k_y\,\Lambda,\qquad \Lambda = s(\theta - \theta_0) - \alpha(\sin\theta - \sin\theta_0)$$

    Parameters
    ----------
    k_y : float or np.ndarray
        Binormal wavenumber, set by the toroidal mode number [1/m].
    s : float or np.ndarray
        Magnetic shear [-].
    theta : float or np.ndarray
        Extended poloidal angle along the field line, the parallel
        coordinate $z$ [rad].
    alpha : float or np.ndarray
        Normalised pressure gradient; zero gives the sheared-slab relation [-].
    theta0 : float or np.ndarray
        Ballooning angle, where $k_x$ vanishes [rad].

    Returns
    -------
    float or np.ndarray
        $k_x$, same units as ``k_y`` [1/m].

    Raises
    ------
    ValueError
        A non-finite input.

    Convention
    ----------
    The $\Lambda$ of ``s_alpha_curvature_drive`` and
    ``s_alpha_ballooning_solution``: $k_\perp^2 = k_y^2(1 + \Lambda^2)$. With
    $\alpha = 0$ it is the sheared slab's $k_x(z) = k_{x0} + k_y\hat s z$ with
    $z = \theta$ and $k_{x0} = -k_y\hat s\theta_0$; in physical length
    $z_\mathrm{phys} = qR\theta$ and $\hat s = qR/L_s$. Constant $k_y$ along
    the line, as in the field-aligned $(x, y, z)$ frame. Codes differ in the
    sign of $\theta_0$: gyrokinetic flux-tube codes (GS2, GENE) define it
    through $k_{x0}/(\hat s k_y)$, which is $-\theta_0$ here.

    Physical interpretation
    -----------------------
    Magnetic shear tilts the phase fronts of a mode that is aligned with the
    field: moving along the line, neighbouring field lines slide past each
    other, so a fixed binormal structure acquires a growing radial
    wavenumber -- the link between ballooning geometry and sheared-slab models.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Proc. R. Soc. A 365,
           1 (1979).
    """
    vals = [np.asarray(v, dtype=float) for v in (k_y, s, theta, alpha, theta0)]
    if not all(np.all(np.isfinite(v)) for v in vals):
        raise ValueError("inputs must be finite")
    k_y, s, theta, alpha, theta0 = vals
    result = k_y * (s * (theta - theta0) - alpha * (np.sin(theta) - np.sin(theta0)))
    return float(result) if np.ndim(result) == 0 else result


def s_alpha_ballooning_eigenmode(s, alpha, theta_max=6.0 * np.pi, n_points=1201):
    r"""The most unstable localised eigenmode of the $s$-$\alpha$ ballooning equation with inertia.

    $$\frac{\mathrm{d}}{\mathrm{d}\theta}\left[(1+\Lambda^{2})\frac{\mathrm{d}F}{\mathrm{d}\theta}\right]
    + \alpha\,(\cos\theta + \Lambda\sin\theta)\,F = \hat\gamma^2\,(1+\Lambda^{2})\,F,\qquad
    F(\pm\theta_\mathrm{max}) = 0$$

    Parameters
    ----------
    s : float
        Magnetic shear [-].
    alpha : float
        Normalised pressure gradient [-].
    theta_max : float
        Half-length of the extended-angle interval, positive [rad].
    n_points : int
        Grid points on $[-\theta_\mathrm{max}, \theta_\mathrm{max}]$, at least 101 [-].

    Returns
    -------
    growth_rate_squared : float
        $\hat\gamma^2 = \gamma^2 q^2R^2/v_A^2$ of the most unstable mode;
        positive is unstable [-].
    theta : np.ndarray
        Extended angle [rad].
    F : np.ndarray
        The eigenfunction, even, normalised to $\max|F| = 1$ with $F(0) > 0$ [-].

    Raises
    ------
    ValueError
        A non-positive ``theta_max`` or too few points.

    Convention
    ----------
    $\Lambda = s\theta - \alpha\sin\theta$ and the operator of
    ``s_alpha_ballooning_solution``; the inertia term $(1+\Lambda^2)$ is
    $k_\perp^2$ along the line and the growth rate is in Alfvén units
    $v_A/(qR)$, the circular large-aspect-ratio normalisation. Second-order
    finite differences, a symmetric generalised eigenproblem, Dirichlet ends:
    for a stable surface the largest eigenvalue is near zero and negative
    (the discretised continuum), for an unstable one it is positive and the
    mode decays well inside the interval. The matrix is scaled by
    $M^{-1/2}$ to a symmetric tridiagonal one, so the cost is linear in
    ``n_points``.

    Physical interpretation
    -----------------------
    The mode balloons: largest at $\theta = 0$, the outboard midplane where
    the curvature is bad, and decaying over a few poloidal transits of the
    extended angle -- the "ballooning" the name refers to.

    Assumptions
    -----------
    As ``s_alpha_ballooning_stable``; ideal MHD, incompressible, $n \to \infty$.

    Validity
    --------
    The unstable mode decays over $\sim 1/\hat\gamma$ in $\theta$, so the
    Dirichlet box needs $\theta_\mathrm{max} \gg 1/\hat\gamma$; near a
    stability boundary $\hat\gamma \to 0$ and the box stabilises the marginal
    mode. With the default $6\pi$ a verdict within about $0.015$ of the
    boundary in $\alpha$ is unreliable (at $s = 1$ the modes for
    $\alpha \in (0.61, 0.625)$ come out stable while
    ``s_alpha_ballooning_stable`` finds them unstable); use
    ``s_alpha_ballooning_stable`` for the verdict and this function for the
    mode structure and growth rate away from the boundary. Like that
    function it needs $s \gtrsim 0.05$ to resolve the envelope.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40,
           396 (1978).
    .. [2] J. W. Connor, R. J. Hastie and J. B. Taylor, Proc. R. Soc. A 365,
           1 (1979).
    """
    from scipy.linalg import eigh_tridiagonal

    if not theta_max > 0.0:
        raise ValueError("theta_max must be positive")
    n_points = int(n_points)
    if n_points < 101:
        raise ValueError("n_points must be at least 101")
    s, a = float(s), float(alpha)
    theta = np.linspace(-theta_max, theta_max, n_points)
    h = theta[1] - theta[0]
    lam = lambda t: s * t - a * np.sin(t)
    p = 1.0 + lam(theta) ** 2
    half = 1.0 + lam(0.5 * (theta[1:] + theta[:-1])) ** 2
    drive = a * (np.cos(theta) + lam(theta) * np.sin(theta))
    inner = slice(1, n_points - 1)
    diag = -(half[:-1] + half[1:]) / h**2 + drive[inner]
    off = half[1:-1] / h**2
    # A F = w M F with M = diag(p) -> the symmetric tridiagonal M^{-1/2} A M^{-1/2} u = w u, F = M^{-1/2} u
    root = np.sqrt(p[inner])
    k = n_points - 3
    w, v = eigh_tridiagonal(diag / p[inner], off / (root[:-1] * root[1:]), select="i", select_range=(k, k))
    F = np.concatenate([[0.0], v[:, 0] / root, [0.0]])
    F = F / F[np.argmax(np.abs(F))]
    if F[n_points // 2] < 0:
        F = -F
    return float(w[0]), theta, F


def s_alpha_marginal_alpha(s, alpha_max=6.0, resolution=1e-3):
    r"""First and second ballooning stability boundaries of the $s$-$\alpha$ model.

    $$\alpha_1(s) = \min\{\alpha : \text{unstable}\}, \qquad
    \alpha_2(s) = \max\{\alpha : \text{unstable}\}$$

    Parameters
    ----------
    s : float or np.ndarray
        Magnetic shear $r q'/q$ [-].
    alpha_max : float
        Upper end of the searched $\alpha$ range [-].
    resolution : float
        Bisection tolerance on each boundary [-].

    Returns
    -------
    alpha_first : float or np.ndarray
        Lowest unstable $\alpha$: the first stability boundary; ``nan`` where
        no instability is found [-].
    alpha_second : float or np.ndarray
        Highest unstable $\alpha$: the second stability boundary; ``nan``
        where none is found or the unstable band reaches ``alpha_max`` [-].

    Raises
    ------
    ValueError
        ``s`` is not finite, ``alpha_max`` is below 0.1 or ``resolution`` is
        not positive.

    Convention
    ----------
    The boundaries of ``s_alpha_ballooning_stable`` (same normalisation and
    Newcomb criterion). The unstable set at fixed $s$ is taken to be the
    single band $[\alpha_1, \alpha_2]$ it is in this model.

    Physical interpretation
    -----------------------
    Below $\alpha_1$ the plasma is first-stable; above $\alpha_2$ it has
    reached second stability. $\alpha_1 \approx 0.6\,s$ at moderate shear,
    the line ``ballooning_stability_criterion`` uses.

    Assumptions
    -----------
    As for ``s_alpha_ballooning_stable``.

    Numerical notes
    ---------------
    A scan on a 0.05-spaced $\alpha$ grid brackets each boundary, then one
    vectorised bisection refines both to ``resolution``. A band narrower than
    the scan spacing (very small $s$) can be missed.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40,
           396 (1978).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.13.
    """
    if not alpha_max >= 0.1 or not resolution > 0.0:
        raise ValueError("alpha_max must be at least 0.1 and resolution positive")
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("s must be finite")
    spacing = 0.05
    grid = np.arange(spacing, alpha_max + 1e-12, spacing)
    unstable = ~s_alpha_ballooning_stable(s_arr[:, None], grid[None, :])
    found = unstable.any(axis=1)
    first_i = np.where(found, unstable.argmax(axis=1), 0)
    last_i = np.where(found, unstable.shape[1] - 1 - unstable[:, ::-1].argmax(axis=1), 0)
    at_top = last_i == len(grid) - 1

    # Both edges in one batch: each bracket has its stable end at ``lo``.
    lo = np.concatenate([np.where(first_i > 0, grid[np.maximum(first_i - 1, 0)], 0.0),
                         grid[np.minimum(last_i + 1, len(grid) - 1)] + at_top * spacing])
    hi = np.concatenate([grid[first_i], grid[last_i]])
    shear = np.concatenate([s_arr, s_arr])
    while np.max(np.abs(hi - lo)) > resolution:
        mid = 0.5 * (lo + hi)
        ok = s_alpha_ballooning_stable(shear, mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    edge = 0.5 * (lo + hi)
    first = np.where(found, edge[: len(s_arr)], np.nan)
    second = np.where(found & ~at_top, edge[len(s_arr):], np.nan)
    if np.ndim(s) == 0:
        return float(first[0]), float(second[0])
    return first, second


def ggj_resistive_interchange_index_from_ideal(D_I, H):
    r"""GGJ resistive-interchange index $D_R$ from the ideal index $D_I$ and $H$.

    $$D_R = D_I + \left(H - \tfrac{1}{2}\right)^{2}$$

    Parameters
    ----------
    D_I : float or np.ndarray
        GGJ ideal-interchange (Mercier) index on a flux surface [-].
    H : float or np.ndarray
        GGJ coefficient $H$ on the same surface [-].

    Returns
    -------
    float or np.ndarray
        Resistive-interchange index $D_R$ [-].

    Convention
    ----------
    GGJ's signs, as ``ggj_ideal_interchange_index`` and
    ``ggj_resistive_interchange_index`` define them and RDCON and DCON write
    them: $D_I > 0$ is Mercier unstable, $D_R > 0$ resistive-interchange
    unstable. It is the same $D_R$ as ``ggj_resistive_interchange_index``,
    eliminating $E + F$ between $D_I = E + F + H - 1/4$ and
    $D_R = E + F + H^{2}$; this form exists because RDCON writes $D_I$ and $H$
    but not $E$ and $F$. $D_R$ is a local criterion, not a Rutherford
    $\Delta'$ contribution, and with any sign of $\Delta'$ it is not a
    tearing verdict, which needs an inner-layer solution.

    Physical interpretation
    -----------------------
    The offset $(H - 1/2)^{2} \ge 0$ is why $D_R \ge D_I$: a surface can be
    Mercier stable yet resistively unstable, never the reverse.

    Assumptions
    -----------
    $D_I$ and $H$ from the same equilibrium, on the same surface.

    References
    ----------
    .. [1] A. H. Glasser, J. M. Greene and J. L. Johnson, "Resistive
           instabilities in general toroidal plasma configurations",
           Phys. Fluids 18, 875 (1975).
    .. [2] A. H. Glasser, Z. R. Wang and J.-K. Park, "Computation of resistive
           instabilities by matched asymptotic expansions", Phys. Plasmas 23,
           112506 (2016).
    """
    return np.asarray(D_I, dtype=float) + (np.asarray(H, dtype=float) - 0.5) ** 2
