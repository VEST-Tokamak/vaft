"""Virial closures: global quantities from the Shafranov boundary integrals.

An equilibrium's poloidal beta and internal inductance can be read off three
boundary integrals and a diamagnetic term, without integrating the interior.
That is what the virial relations give, and this module holds them: the
integral combinations, the closures that solve them in pairs, and the
diamagnetic parameter the third relation needs.

Split out of :mod:`vaft.formula.equilibrium` (#711), where they were 32 of its
110 functions and a third of its length while coupling to nothing else in it.
Every name remains importable from that module.

Notation
--------
S_1, S_2, S_3 : Shafranov boundary integrals, normalised by B_pa    [-]
alpha         : closure coefficient multiplying l_i                 [-]
mu_i          : diamagnetic parameter, volume convention            [-]
mu_i_hat      : the same in the flux convention, its negative       [-]
R_T           : the virial radius                                   [m]
B_pa          : boundary-average poloidal field                     [T]

Conventions
-----------
**Two sign conventions for the diamagnetic parameter coexist, and they are
negatives of each other.** The volume form is what the three relations use --
the volume average of ``B_tv^2 - B_t^2`` over ``B_pa^2``, positive for a
diamagnetic plasma -- and every closure here takes it. The flux form, produced
by :func:`virial_muihat_from_Bt_R0_dphi`, is what an EFIT-style diamagnetic
flux gives; :func:`virial_beta_pd_from_S_mu_rt` is the one consumer written for
it. Feeding a closure the wrong one shifts the result by ``2*mu_i`` with
nothing in the number to show it (#546).

A closure whose denominator approaches zero returns NaN rather than a large
finite number, at the threshold :data:`VIRIAL_SINGULAR_EPS`.

Provenance
----------
.. [1] V. D. Shafranov, *Plasma Physics* 13 (1971) 757, for the virial
   relations these solve.
.. [2] L. L. Lao et al., Nucl. Fusion 25 (1985) 1611, for the large-aspect-ratio
   closure and the EFIT normalisation by the boundary-average poloidal field.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np

from ._exports import public_names
from .constants import MU0


# ------------------------------------------------------------------
# Virial Theorem
# ------------------------------------------------------------------

def virial_magnetic_energy(B: np.ndarray,
                          V: float) -> float:
    r"""Magnetic energy of a sampled field, $W_{mag} = \sum B^2\,V/(2\mu_0)$.

    $$W_{\mathrm{mag}} = \int\frac{B^2}{2\mu_0}\,dV \approx \frac{V}{2\mu_0}\sum_i B_i^2$$

    Parameters
    ----------
    B : np.ndarray
        Field magnitude at each sample [T].
    V : float
        Volume attributed to *each* sample [m^3].

    Returns
    -------
    float
        Magnetic energy [J].

    Assumptions
    -----------
    Equal-volume samples: ``V`` multiplies the plain sum, so it is the cell
    volume, not the total.  Pass ``V_total / B.size`` for a uniform grid.

    Numerical notes
    ---------------
    A Riemann sum, first order in the cell size.

    References
    ----------
    .. [1] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2, Consultants
           Bureau (1966), p. 103 (virial theorem for a confined plasma).
    """
    return np.sum(B**2) * V / (2 * MU0)
def virial_kinetic_energy(n: np.ndarray,
                         v: np.ndarray,
                         m: float,
                         V: float) -> float:
    r"""Bulk kinetic energy of a sampled flow, $W_{kin} = \tfrac{1}{2}\sum n m v^2\,V$.

    $$W_{\mathrm{kin}} = \int\frac{1}{2}\,n\,m\,v^2\,dV$$

    Parameters
    ----------
    n : np.ndarray
        Number density at each sample [m^-3].
    v : np.ndarray
        Flow speed at each sample [m/s].
    m : float
        Particle mass [kg].
    V : float
        Volume attributed to each sample [m^3].

    Returns
    -------
    float
        Kinetic energy of the flow [J].

    Assumptions
    -----------
    Equal-volume samples (``V`` is the cell volume); the flow is a bulk velocity,
    not a thermal speed.

    References
    ----------
    .. [1] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2, Consultants
           Bureau (1966), p. 103.
    """
    return 0.5 * np.sum(n * m * v**2) * V
def virial_thermal_energy(n: np.ndarray,
                         T: np.ndarray,
                         V: float) -> float:
    r"""Thermal energy of a sampled plasma, $W_{th} = \tfrac{3}{2}\sum n T\,V$.

    $$W_{\mathrm{th}} = \int\frac{3}{2}\,n\,T\,dV$$

    Parameters
    ----------
    n : np.ndarray
        Number density at each sample [m^-3].
    T : np.ndarray
        Temperature at each sample, in energy units [J].
    V : float
        Volume attributed to each sample [m^3].

    Returns
    -------
    float
        Thermal energy [J].

    Convention
    ----------
    ``T`` must be in joules ($k_B T$); a temperature in eV needs the factor
    ``QE``.  One species only: sum electron and ion calls for the total.

    Assumptions
    -----------
    Equal-volume samples (``V`` is the cell volume); three degrees of freedom.

    References
    ----------
    .. [1] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2, Consultants
           Bureau (1966), p. 103.
    """
    return 1.5 * np.sum(n * T) * V
def virial_theorem(W_mag: float,
                  W_kin: float,
                  W_th: float) -> Tuple[float, float]:
    r"""Total energy and virial ratio of magnetic, kinetic and thermal contributions.

    $$W_{\mathrm{total}} = W_{\mathrm{mag}} + W_{\mathrm{kin}} + W_{\mathrm{th}},
      \qquad r_v = \frac{W_{\mathrm{kin}} + W_{\mathrm{th}}}{W_{\mathrm{mag}}}$$

    Parameters
    ----------
    W_mag : float
        Magnetic energy [J].
    W_kin : float
        Bulk kinetic energy [J].
    W_th : float
        Thermal energy [J].

    Returns
    -------
    W_total : float
        Sum of the three energies [J].
    virial_ratio : float
        Ratio of material to magnetic energy [-].

    Physical interpretation
    -----------------------
    The scalar virial theorem forbids a plasma confined by its own fields alone:
    a positive-definite $W_{\mathrm{mag}}$ must be balanced by external
    (coil) fields, and the ratio measures how far the material energy is from
    that balance.

    References
    ----------
    .. [1] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2, Consultants
           Bureau (1966), p. 103.
    .. [2] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
           Sec. 3.6 (virial theorem).
    """
    W_total = W_mag + W_kin + W_th
    virial_ratio = (W_kin + W_th) / W_mag
    return W_total, virial_ratio
def virial_stability_criterion(W_mag: float,
                             W_kin: float,
                             W_th: float) -> Tuple[float, float]:
    r"""Virial-ratio margin against the heuristic threshold $r_v = 0.5$.

    $$\Delta = r_v - 0.5, \qquad
      r_v = \frac{W_{\mathrm{kin}} + W_{\mathrm{th}}}{W_{\mathrm{mag}}}$$

    Parameters
    ----------
    W_mag : float
        Magnetic energy [J].
    W_kin : float
        Bulk kinetic energy [J].
    W_th : float
        Thermal energy [J].

    Returns
    -------
    margin : float
        $r_v - 0.5$ [-].
    critical_ratio : float
        The threshold 0.5 [-].

    Limitations
    -----------
    The threshold 0.5 is labelled "theoretical value for stability" in the
    original VAFT source but no derivation or reference for it was recorded;
    the scalar virial theorem constrains equilibrium, not stability.  Treat the
    margin as a bookkeeping diagnostic.  Tracked in #366.

    References
    ----------
    .. [1] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2, Consultants
           Bureau (1966), p. 103.
    """
    W_total, virial_ratio = virial_theorem(W_mag, W_kin, W_th)
    critical_ratio = 0.5  # Theoretical value for stability
    return virial_ratio - critical_ratio, critical_ratio
def virial_beta_p_from_volume(p: np.ndarray,
                              dV: np.ndarray,
                              B_pa: float,
                              Omega: float,
                              mu0: float = None) -> float:
    r"""Poloidal beta from a volume integral of pressure.

    $$\beta_p = \frac{2\mu_0}{B_{pa}^2\,\Omega}\int_\Omega p\,dV$$

    Parameters
    ----------
    p : np.ndarray
        Pressure at each cell [Pa].
    dV : np.ndarray
        Volume of each cell [m^3].
    B_pa : float
        Boundary-averaged poloidal field, $\mu_0 I_p/L_p$ [T].
    Omega : float
        Plasma volume $\Omega = \sum dV$ [m^3].
    mu0 : float, optional
        Vacuum permeability; default ``MU0`` [H/m].

    Returns
    -------
    float
        Poloidal beta [-].

    Convention
    ----------
    The EFIT/Lao definition normalised by $B_{pa} = \mu_0 I_p/L_p$ with $L_p$ the
    boundary contour length, as used by the virial closures below.  Other codes
    normalise by $B_p$ at the boundary or by $\mu_0 I_p/(2\pi a)$; the values
    differ by shape-dependent factors.

    Numerical notes
    ---------------
    Plain weighted sum over the supplied cells.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 2.
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    if mu0 is None:
        mu0 = MU0
    return (2 * mu0 / (B_pa**2 * Omega)) * np.sum(p * dV)
def virial_li_from_volume(B_p: np.ndarray,
                         dV: np.ndarray,
                         B_pa: float,
                         Omega: float) -> float:
    r"""Internal inductance from a volume integral of $B_p^2$.

    $$l_i = \frac{1}{B_{pa}^2\,\Omega}\int_\Omega B_p^2\,dV$$

    Parameters
    ----------
    B_p : np.ndarray
        Poloidal field magnitude at each cell [T].
    dV : np.ndarray
        Volume of each cell [m^3].
    B_pa : float
        Boundary-averaged poloidal field, $\mu_0 I_p/L_p$ [T].
    Omega : float
        Plasma volume [m^3].

    Returns
    -------
    float
        Internal inductance [-].

    Convention
    ----------
    Lao/EFIT normalisation by $B_{pa}$; not the IMAS $l_{i,3}$ (normalised by
    $(\mu_0 I_p)^2 R_0/2$) nor the cylindrical $l_i$.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 2.
    """
    return (1.0 / (B_pa**2 * Omega)) * np.sum(B_p**2 * dV)
def virial_muihat_from_Bt_R0_dphi(B_t: float,
                                 R0: float,
                                 dphi: float,
                                 B_pa: float,
                                 Omega: float) -> float:
    r"""Diamagnetic parameter $\hat\mu_i$ from the measured diamagnetic flux.

    $$\hat\mu_i \approx \frac{4\pi\,B_t\,R_0\,\Delta\phi}{B_{pa}^2\,\Omega}$$

    Parameters
    ----------
    B_t : float
        Vacuum toroidal field at ``R0`` [T].
    R0 : float
        Major radius at which ``B_t`` is quoted [m].
    dphi : float
        Diamagnetic flux $\Delta\phi$ (plasma-induced change of toroidal flux) [Wb].
    B_pa : float
        Boundary-averaged poloidal field [T].
    Omega : float
        Plasma volume [m^3].

    Returns
    -------
    float
        $\hat\mu_i$ [-].

    Convention
    ----------
    Sign follows ``dphi``: a paramagnetic (low-$\beta_p$) plasma increases the
    toroidal flux and gives $\hat\mu_i > 0$ in the sign convention of the
    diamagnetic loop; check the loop's polarity before comparing with a virial
    closure.  $\Delta\phi$ is a full-weber toroidal flux.

    Assumptions
    -----------
    Large-aspect-ratio expansion of the toroidal-field energy term, $B_t R_0$
    constant over the cross-section.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 2 ($\mu_i$ definition).
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    return (4 * np.pi * B_t * R0 * dphi) / (B_pa**2 * Omega)
def approximated_diamagnetism_from_B_pa_B_tv_R0_delta_phi(B_pa: float,
                                                         B_tv: float,
                                                         R0: float,
                                                         delta_phi: float,
                                                         V_p: float) -> float:
    r"""Diamagnetic parameter from the vacuum toroidal field and flux change.

    $$\hat\mu_i \approx \frac{1}{B_{pa}^2 V_p}\int_0^{2\pi}d\varphi\,R_0\,(2B_{tv}\Delta\phi)
      = \frac{4\pi\,B_{tv}\,R_0\,\Delta\phi}{B_{pa}^2\,V_p}$$

    Parameters
    ----------
    B_pa : float
        Boundary-averaged poloidal field [T].
    B_tv : float
        Vacuum toroidal field at ``R0`` [T].
    R0 : float
        Major radius [m].
    delta_phi : float
        Diamagnetic flux $\Delta\phi$ [Wb].
    V_p : float
        Plasma volume [m^3].

    Returns
    -------
    float
        $\hat\mu_i$ [-].

    Convention
    ----------
    Numerically identical to :func:`virial_muihat_from_Bt_R0_dphi`; kept under
    the name used by the VFIT-era analysis.  Same sign caveat on ``delta_phi``.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 2.
    """
    return (4 * np.pi * B_tv * R0 * delta_phi) / (B_pa**2 * V_p)
#: Denominator magnitude below which a virial closure is reported as
#: indeterminate rather than as a large finite number.  The Shafranov integrals
#: and $\alpha$ are all $O(1)$, so an absolute floor is the meaningful one.
VIRIAL_SINGULAR_EPS = 1e-9
def _virial_ratio(numerator: float, denominator: float, eps: float) -> float:
    """Divide, or return NaN when the denominator is within ``eps`` of zero."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= eps:
        return float("nan")
    return float(numerator) / float(denominator)
def virial_beta_p_from_S_alpha_mu(S1: float,
                                  S2: float,
                                  S3: float,
                                  alpha: float,
                                  mui_hat: float) -> float:
    r"""Poloidal beta from the Shafranov integrals, low-aspect-ratio closure.

    $$\beta_p = \frac{(S_1 + S_2)(\alpha - 1) + \alpha\mu_i + S_3}{3(\alpha-1) + 1}$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third virial relation [-].
    mui_hat : float
        Diamagnetic parameter [-].

    Returns
    -------
    float
        Poloidal beta [-].

    Convention
    ----------
    Despite the parameter name, this takes $\mu_i$ in the **volume** sign the
    three virial relations use -- $\langle B_{tv}^2-B_t^2\rangle/B_{pa}^2$,
    positive when diamagnetic -- not the flux-sign $\hat\mu_i$ that
    :func:`virial_muihat_from_Bt_R0_dphi` produces. The two are negatives of
    each other, so the wrong one returns $\beta_p \mp 2\mu_i$ with nothing in
    the number to show it. $S_1$-$S_3$ are in the Lao/EFIT normalisation by
    $B_{pa}$ (:func:`virial_beta_p_from_volume`). The closure retains the
    diamagnetic term, so it holds at low aspect ratio where the Lao form does
    not.

    References
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), low-aspect-ratio
           virial closure (journal page not recorded in the VAFT source).
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    num = (S1 + S2) * (alpha - 1) + alpha * mui_hat + S3
    den = 3 * (alpha - 1) + 1
    return num / den
def virial_li_from_S_alpha_mu(S1: float,
                             S2: float,
                             S3: float,
                             alpha: float,
                             mui_hat: float) -> float:
    r"""Internal inductance from the Shafranov integrals, low-aspect-ratio closure.

    $$l_i = \frac{S_1 + S_2 - 2\mu_i - 3S_3}{3\alpha - 2}$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third virial relation [-].
    mui_hat : float
        Diamagnetic parameter $\mu_i$, volume convention -- see Convention [-].

    Returns
    -------
    float
        Internal inductance [-].

    Convention
    ----------
    Despite the parameter name, this takes $\mu_i$ in the **volume** sign the
    three virial relations use -- $\langle B_{tv}^2-B_t^2\rangle/B_{pa}^2$,
    positive when diamagnetic -- not the flux-sign $\hat\mu_i$ that
    :func:`virial_muihat_from_Bt_R0_dphi` produces. The two are negatives of
    each other, so the wrong one returns a result off by $2\mu_i$ with nothing
    in the number to show it.

    Companion of :func:`virial_beta_p_from_S_alpha_mu`, same normalisation.

    Limitations
    -----------
    Ill-conditioned as $\alpha\to2/3$; no guard.

    References
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), low-aspect-ratio
           virial closure (journal page not recorded in the VAFT source).
    """
    num = S1 + S2 - 2 * mui_hat - 3 * S3
    den = 3 * alpha - 2
    return num / den
def virial_beta_p_lao_from_S_mu_rt(
    S1: float,
    S2: float,
    mui: float,
    RT_over_R0: float,
) -> float:
    r"""Poloidal beta, Lao large-aspect-ratio virial closure.

    $$\beta_p = \frac{S_1}{2} + \frac{S_2}{2}\left(1 - \frac{R_T}{R_0}\right) + \mu_i$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    mui : float
        Diamagnetic parameter $\mu_i$ [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius, $R_T/R_0$ [-].

    Returns
    -------
    float
        Poloidal beta [-].

    Convention
    ----------
    The $R_T/R_0$ term carries a **minus** sign: this is $\tfrac{1}{2}(E_1-E_2)$
    of the three virial relations solved by
    :func:`virial_bp_li_lihat_from_S123`, which fixes $\beta_p - \mu_i$ from
    $E_1$ and $E_2$ alone. A plus sign belongs to the other combination,
    $E_1+E_2$, which eliminates $\mu_i$ in favour of $l_i$ and is what
    :func:`virial_beta_p_from_S_li` evaluates at $R_T/R_0 = 1$. The two are not
    interchangeable, and confusing them is what this function did until #546.

    Validity
    --------
    Large aspect ratio; at VEST aspect ratio the neglected $\epsilon$ terms
    reach tens of percent, which is why the Bongard closure exists.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    """
    return 0.5 * S1 + 0.5 * S2 * (1.0 - RT_over_R0) + mui
def virial_li_from_S_alpha_rt(
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    RT_over_R0: float,
    eps: float = 1e-12,
) -> float:
    r"""Internal inductance, Lao large-aspect-ratio virial closure.

    $$l_i^{\mathrm{vir}} = \frac{\tfrac{S_1}{2} + \tfrac{S_2}{2}\left(1 - \tfrac{R_T}{R_0}\right) - S_3}{\alpha - 1}$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third virial relation [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius [-].
    eps : float, optional
        Tolerance below which $|\alpha-1|$ is rejected; default 1e-12 [-].

    Returns
    -------
    float
        Internal inductance [-].

    Raises
    ------
    ValueError
        When ``alpha`` is within ``eps`` of 1 (the closure is singular there).

    Validity
    --------
    Large aspect ratio, as :func:`virial_beta_p_lao_from_S_mu_rt`.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    """
    den = alpha - 1.0
    if abs(den) <= eps:
        raise ValueError("alpha is too close to 1; li_vir is ill-conditioned.")
    num = 0.5 * S1 + 0.5 * S2 * (1.0 - RT_over_R0) - S3
    return num / den
def virial_beta_p_from_S_li(
    S1: float,
    S2: float,
    li: float,
) -> float:
    r"""Poloidal beta from $S_1$, $S_2$ and a known internal inductance.

    $$\beta_p^{\mathrm{vir}} = \frac{S_1}{4} + \frac{S_2}{2} - \frac{l_i}{2}$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    li : float
        Internal inductance [-].

    Returns
    -------
    float
        Poloidal beta [-].

    Physical interpretation
    -----------------------
    Solves the first virial relation $S_1 + S_2 = 3\beta_p + l_i - \hat l_i$ for
    $\beta_p$ after dropping $\hat l_i$ and rescaling, giving the classic
    "$\beta_p + l_i/2$ from magnetics" separation when $l_i$ is known
    independently.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    """
    return 0.25 * S1 + 0.5 * S2 - 0.5 * li
def virial_beta_pd_from_S_mu_rt(
    S1: float,
    S2: float,
    mui: float,
    RT_over_R0: float,
) -> float:
    r"""Diamagnetic poloidal beta $\beta_{p,d}$ from $S_1$, $S_2$ and $\mu_i$.

    $$\beta_{p,d}^{\mathrm{vir}} = \frac{S_1}{2} - \mu_i + \frac{S_2}{2}\left(1 - \frac{R_T}{R_0}\right)$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    mui : float
        Diamagnetic parameter $\mu_i$ [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius [-].

    Returns
    -------
    float
        Diamagnetic poloidal beta [-].

    Convention
    ----------
    **The one consumer in this module that takes the flux-sign $\hat\mu_i$**,
    the quantity :func:`virial_muihat_from_Bt_R0_dphi` and
    :func:`vaft.process.equilibrium.computed_diamagnetism_from_phi` produce --
    not the volume $\mu_i$ every closure here takes. The two are negatives of
    each other, so the wrong one returns $\beta_p \mp 2\mu_i$ with nothing in
    the number to show it. The parameter is named ``mui`` for history; read it
    as $\hat\mu_i$.

    Validity
    --------
    Large aspect ratio (Lao closure).

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    """
    return 0.5 * S1 - mui + 0.5 * S2 * (1.0 - RT_over_R0)
def virial_beta_p_li_from_S_alpha_mu_rt(
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    mui: float,
    RT_over_R0: float,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    r"""Lao closure bundle: $\beta_p$, $l_i$ and $\beta_{p,d}$ in one call.

    Evaluates :func:`virial_beta_p_lao_from_S_mu_rt`,
    :func:`virial_li_from_S_alpha_rt` and :func:`virial_beta_pd_from_S_mu_rt`
    on the same inputs.

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third virial relation [-].
    mui : float
        Diamagnetic parameter $\mu_i$ [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius [-].
    eps : float, optional
        Singularity tolerance on $|\alpha - 1|$; default 1e-12 [-].

    Returns
    -------
    beta_p_lao : float
        Poloidal beta [-].
    li_lao : float
        Internal inductance [-].
    beta_pd_vir : float
        Diamagnetic poloidal beta [-].

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    """
    li_lao = virial_li_from_S_alpha_rt(S1, S2, S3, alpha, RT_over_R0, eps=eps)
    beta_p_lao = virial_beta_p_lao_from_S_mu_rt(S1, S2, mui, RT_over_R0)
    beta_pd_vir = virial_beta_pd_from_S_mu_rt(S1, S2, mui, RT_over_R0)
    return beta_p_lao, li_lao, beta_pd_vir
def virial_lao_from_S_alpha_mu_rt(
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    mui: float,
    RT_over_R0: float,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    r"""Large-aspect-ratio virial closure (Lao 1985): $\beta_p$ and $l_i$.

    Evaluates :func:`virial_beta_p_lao_from_S_mu_rt` and
    :func:`virial_li_from_S_alpha_rt`.

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third virial relation [-].
    mui : float
        Diamagnetic parameter $\mu_i$ [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius [-].
    eps : float, optional
        Singularity tolerance on $|\alpha - 1|$; default 1e-12 [-].

    Returns
    -------
    beta_p_lao : float
        Poloidal beta [-].
    li_lao : float
        Internal inductance [-].

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    """
    beta_p_lao = virial_beta_p_lao_from_S_mu_rt(S1, S2, mui, RT_over_R0)
    li_lao = virial_li_from_S_alpha_rt(S1, S2, S3, alpha, RT_over_R0, eps=eps)
    return beta_p_lao, li_lao
def virial_bongard_from_S_alpha_mu(
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    mui: float,
    eps: float = VIRIAL_SINGULAR_EPS,
) -> Tuple[float, float]:
    r"""Low-aspect-ratio virial closure (Bongard 2016): $\beta_p$ and $l_i$.

    The historical name for the $E_1$/$E_3$ pairwise closure; delegates to
    :func:`virial_pair_13_from_S_alpha_mu`, which is the same algebra. The two
    are identical by construction, not by approximation.

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third virial relation [-].
    mui : float
        Diamagnetic parameter $\mu_i$, volume convention -- see Convention [-].
    eps : float, optional
        Denominator magnitude below which the result is NaN, forwarded to the
        closure; default :data:`VIRIAL_SINGULAR_EPS` [-].

    Returns
    -------
    beta_p_bongard : float
        Poloidal beta [-].
    li_bongard : float
        Internal inductance [-].

    References
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), low-aspect-ratio
           virial closure (journal page not recorded in the VAFT source).
    """
    return virial_pair_13_from_S_alpha_mu(S1, S2, S3, alpha, mui, eps=eps)
def virial_identity_residuals(
    beta_p: float,
    li: float,
    mu_i: float,
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    RT_over_R0: float,
) -> Tuple[float, float, float]:
    r"""Residuals of the three virial identities at a given $(\beta_p, l_i, \mu_i)$.

    $$R_1 = 3\beta_p + l_i - \mu_i - (S_1+S_2), \qquad
      R_2 = \beta_p + l_i + \mu_i - \frac{R_T}{R_0}S_2, \qquad
      R_3 = \beta_p - (\alpha-1)l_i - \mu_i - S_3$$

    Parameters
    ----------
    beta_p : float
        Poloidal beta at which the identities are evaluated [-].
    li : float
        Internal inductance at which the identities are evaluated [-].
    mu_i : float
        Diamagnetic parameter at which the identities are evaluated [-].
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third relation [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius, $R_T/R_0$ [-].

    Returns
    -------
    e1 : float
        Residual of the first identity [-].
    e2 : float
        Residual of the second identity [-].
    e3 : float
        Residual of the third identity [-].

    Physical interpretation
    -----------------------
    Zero residuals say the triple closes the three global force-balance
    relations; a residual localises *which* relation an inconsistent
    equilibrium violates, which a single $\beta_p$ estimator cannot.

    Convention
    ----------
    Signs follow the relations as written above, which are the ones
    :func:`virial_full_123_from_S_alpha_rt` solves. Every closure and residual
    in this module is derived from these three and no other statement of them.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eqs. (1)-(3).
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    e1 = 3.0 * beta_p + li - mu_i - (S1 + S2)
    e2 = beta_p + li + mu_i - RT_over_R0 * S2
    e3 = beta_p - (alpha - 1.0) * li - mu_i - S3
    return float(e1), float(e2), float(e3)
def virial_pair_12_from_S_mu_rt(
    S1: float,
    S2: float,
    mu_i: float,
    RT_over_R0: float,
) -> Tuple[float, float]:
    r"""Virial closure on $E_1$ and $E_2$, leaving $E_3$ free.

    $$\beta_p = \frac{S_1}{2} + \frac{S_2}{2}\left(1-\frac{R_T}{R_0}\right) + \mu_i,
      \qquad
      l_i = -\frac{S_1}{2} + \frac{S_2}{2}\left(3\frac{R_T}{R_0}-1\right) - 2\mu_i$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    mu_i : float
        Diamagnetic parameter $\mu_i$, supplied independently [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius, $R_T/R_0$ [-].

    Returns
    -------
    beta_p : float
        Poloidal beta [-].
    li : float
        Internal inductance [-].

    Physical interpretation
    -----------------------
    Both unknowns come from the two relations that do not involve $\alpha$, so
    this closure is insensitive to the boundary field's poloidal anisotropy and
    has no singular denominator. It is the only one of the three pairs that
    cannot become ill-conditioned.

    Convention
    ----------
    The $\beta_p$ here is the conventional Lao one and agrees with
    :func:`virial_beta_p_lao_from_S_mu_rt`. The $l_i$ does **not**: the
    historical Lao $l_i$ takes $\beta_p-\mu_i$ from this same pair and then
    substitutes it into $E_3$, so it is the $l_i$ of the full three-relation
    solve, not of this closure. See
    :func:`virial_li_from_S_alpha_rt`.

    Validity
    --------
    Depends entirely on $R_T/R_0$, which is poorly conditioned whenever the
    volume integral defining $R_T$ has a near-vanishing denominator. Compare
    against :func:`virial_pair_13_from_S_alpha_mu`, which does not use it.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 3.
    .. [2] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eqs. (1)-(3).
    """
    beta_p = 0.5 * S1 + 0.5 * (1.0 - RT_over_R0) * S2 + mu_i
    li = -0.5 * S1 + 0.5 * (3.0 * RT_over_R0 - 1.0) * S2 - 2.0 * mu_i
    return float(beta_p), float(li)
def virial_pair_13_from_S_alpha_mu(
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    mu_i: float,
    eps: float = VIRIAL_SINGULAR_EPS,
) -> Tuple[float, float]:
    r"""Virial closure on $E_1$ and $E_3$, leaving $E_2$ free.

    $$\beta_p = \frac{(\alpha-1)(S_1+S_2)+S_3+\alpha\mu_i}{3\alpha-2},
      \qquad
      l_i = \frac{S_1+S_2-3S_3-2\mu_i}{3\alpha-2}$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third relation [-].
    mu_i : float
        Diamagnetic parameter $\mu_i$, supplied independently [-].
    eps : float, optional
        Denominator magnitude below which the result is NaN; default
        :data:`VIRIAL_SINGULAR_EPS` [-].

    Returns
    -------
    beta_p : float
        Poloidal beta, NaN when $3\alpha-2$ is within ``eps`` of zero [-].
    li : float
        Internal inductance, NaN under the same condition [-].

    Physical interpretation
    -----------------------
    The only closure that never mentions $R_T/R_0$. At low aspect ratio, where
    the current centroid is both large and poorly determined, comparing this
    against the other two pairs isolates how much of a $\beta_p$ or $l_i$
    disagreement is $R_T$ sensitivity rather than physics.

    Convention
    ----------
    This is the Bongard low-aspect-ratio closure;
    :func:`virial_bongard_from_S_alpha_mu` is the same numbers under the
    historical name.

    Numerical notes
    ---------------
    Singular at $\alpha = 2/3$. Returns NaN there rather than a large finite
    value, so a near-singular closure is not mistaken for a physical failure.

    References
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), low-aspect-ratio
           virial closure (journal page not recorded in the VAFT source).
    .. [2] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eqs. (1)-(3).
    """
    den = 3.0 * alpha - 2.0
    beta_p = _virial_ratio((alpha - 1.0) * (S1 + S2) + S3 + alpha * mu_i, den, eps)
    li = _virial_ratio(S1 + S2 - 3.0 * S3 - 2.0 * mu_i, den, eps)
    return beta_p, li
def virial_pair_23_from_S_alpha_mu_rt(
    S2: float,
    S3: float,
    alpha: float,
    mu_i: float,
    RT_over_R0: float,
    eps: float = VIRIAL_SINGULAR_EPS,
) -> Tuple[float, float]:
    r"""Virial closure on $E_2$ and $E_3$, leaving $E_1$ free.

    $$\beta_p = \frac{(\alpha-1)\tfrac{R_T}{R_0}S_2+S_3+(2-\alpha)\mu_i}{\alpha},
      \qquad
      l_i = \frac{\tfrac{R_T}{R_0}S_2-S_3-2\mu_i}{\alpha}$$

    Parameters
    ----------
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third relation [-].
    mu_i : float
        Diamagnetic parameter $\mu_i$, supplied independently [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius, $R_T/R_0$ [-].
    eps : float, optional
        Denominator magnitude below which the result is NaN; default
        :data:`VIRIAL_SINGULAR_EPS` [-].

    Returns
    -------
    beta_p : float
        Poloidal beta, NaN when $\alpha$ is within ``eps`` of zero [-].
    li : float
        Internal inductance, NaN under the same condition [-].

    Physical interpretation
    -----------------------
    Drops $E_1$, the relation carrying $S_1$, which is the integral least
    sensitive to boundary shape and therefore the best determined of the three.
    Read this closure as a probe of $E_1$ rather than as a preferred estimator:
    its residual on the omitted identity is the informative output.

    Numerical notes
    ---------------
    Singular at $\alpha = 0$, and returns NaN there rather than a large finite
    value.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eqs. (1)-(3).
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    beta_p = _virial_ratio(
        (alpha - 1.0) * RT_over_R0 * S2 + S3 + (2.0 - alpha) * mu_i, alpha, eps
    )
    li = _virial_ratio(RT_over_R0 * S2 - S3 - 2.0 * mu_i, alpha, eps)
    return beta_p, li
def virial_full_123_from_S_alpha_rt(
    S1: float,
    S2: float,
    S3: float,
    alpha: float,
    RT_over_R0: float,
    eps: float = VIRIAL_SINGULAR_EPS,
) -> Tuple[float, float, float]:
    r"""Solve all three virial relations for $\beta_p$, $l_i$ and $\mu_i$.

    $$\begin{pmatrix} 3 & 1 & -1 \\ 1 & 1 & 1 \\ 1 & -(\alpha-1) & -1
      \end{pmatrix}
      \begin{pmatrix}\beta_p \\ l_i \\ \mu_i\end{pmatrix}
      = \begin{pmatrix} S_1+S_2 \\ \tfrac{R_T}{R_0}S_2 \\ S_3\end{pmatrix}$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    alpha : float
        Closure coefficient multiplying $l_i$ in the third relation [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius, $R_T/R_0$ [-].
    eps : float, optional
        Determinant magnitude below which the result is NaN; default
        :data:`VIRIAL_SINGULAR_EPS` [-].

    Returns
    -------
    beta_p : float
        Poloidal beta, NaN when $4(\alpha-1)$ is within ``eps`` of zero [-].
    li : float
        Internal inductance, NaN under the same condition [-].
    mu_i : float
        Diamagnetic parameter, NaN under the same condition [-].

    Physical interpretation
    -----------------------
    A different inverse problem from the pairwise closures: they take $\mu_i$ as
    known and solve for two unknowns, while this takes none of the three as
    known. Its $\mu_i$ is therefore an equilibrium-derived prediction that a
    measured diamagnetic flux can be compared against, rather than an input.

    Convention
    ----------
    Its $l_i$ is identically the historical Lao $l_i$, because $E_1$ and $E_2$
    already fix $\beta_p-\mu_i$ and only $E_3$ separates $l_i$ from it. That is
    an algebraic identity, not an approximation.

    Numerical notes
    ---------------
    Determinant $4(\alpha-1)$: singular at $\alpha = 1$, the same limit that
    makes the historical Lao $l_i$ diverge. Solved in closed form rather than
    with ``numpy.linalg.solve`` so the singularity is detected instead of
    raising or returning a large finite value.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eqs. (1)-(3).
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    # beta_p - mu_i follows from (E1 - E2)/2 alone; E3 then gives li. The guard
    # is on the determinant 4*(alpha-1) that the docstring and
    # virial_closure_denominators both name, not on alpha-1, so a caller tuning
    # eps against those numbers gets the band they asked for.
    if not np.isfinite(alpha) or abs(4.0 * (alpha - 1.0)) <= eps:
        return float("nan"), float("nan"), float("nan")
    half_diff = 0.5 * S1 + 0.5 * (1.0 - RT_over_R0) * S2
    li = _virial_ratio(half_diff - S3, alpha - 1.0, 0.0)
    if not np.isfinite(li):
        return float("nan"), float("nan"), float("nan")
    # E2 supplies beta_p + mu_i once li is known.
    sum_ = RT_over_R0 * S2 - li
    beta_p = 0.5 * (sum_ + half_diff)
    mu_i = 0.5 * (sum_ - half_diff)
    return float(beta_p), float(li), float(mu_i)
def virial_closure_denominators(alpha: float) -> Tuple[float, float, float, float]:
    r"""The denominator each virial closure divides by, for conditioning checks.

    $$3\alpha-2,\qquad \alpha,\qquad \alpha-1,\qquad 4(\alpha-1)$$

    Parameters
    ----------
    alpha : float
        Closure coefficient multiplying $l_i$ in the third relation [-].

    Returns
    -------
    pair_13 : float
        Denominator of the $E_1$/$E_3$ closure, $3\alpha-2$ [-].
    pair_23 : float
        Denominator of the $E_2$/$E_3$ closure, $\alpha$ [-].
    lao_li : float
        Denominator of the historical Lao $l_i$, $\alpha-1$ [-].
    full_123 : float
        Determinant of the three-relation system, $4(\alpha-1)$ [-].

    Physical interpretation
    -----------------------
    The identities themselves have no singularities; only the inversions do.
    Reporting the denominators separates "this equilibrium violates force
    balance" from "this particular closure cannot be inverted here", which a
    bare $\beta_p$ cannot distinguish.

    Notes
    -----
    The $E_1$/$E_2$ closure is absent because it has no denominator; it is the
    one pair that cannot become singular.
    """
    return (
        float(3.0 * alpha - 2.0),
        float(alpha),
        float(alpha - 1.0),
        float(4.0 * (alpha - 1.0)),
    )
def virial_normalized_residual(residual: float, lhs: float, rhs: float) -> float:
    r"""Scale a virial identity residual so shots can be compared.

    $$\epsilon = \frac{R}{\max\left(1, |\mathrm{LHS}|, |\mathrm{RHS}|\right)}$$

    Parameters
    ----------
    residual : float
        Raw residual $\mathrm{LHS}-\mathrm{RHS}$ of one identity [-].
    lhs : float
        Left-hand side of that identity [-].
    rhs : float
        Right-hand side of that identity [-].

    Returns
    -------
    float
        Dimensionless residual [-].

    Physical interpretation
    -----------------------
    Symmetric in the two sides, and the floor of 1 keeps the scale from
    collapsing when both sides pass through zero, which would turn a small
    absolute residual into a large apparent one.

    Limitations
    -----------
    A placeholder scale, not an uncertainty. It is deliberately replaceable: an
    uncertainty-propagated normalisation would be strictly better and would
    change the numbers, so the thresholds built on this one stay report-only
    until a representative equilibrium population qualifies them.
    """
    lhs, rhs = float(lhs), float(rhs)
    # max() keeps its first argument against a NaN, so guard explicitly rather
    # than letting an unknown side quietly become a scale of 1.
    if not (np.isfinite(lhs) and np.isfinite(rhs)):
        return float("nan")
    scale = max(1.0, abs(lhs), abs(rhs))
    return float(residual) / scale
def virial_residual_rms(e1: float, e2: float, e3: float) -> float:
    r"""Aggregate the three normalised virial residuals into one number.

    $$E_{virial} = \sqrt{\frac{\epsilon_1^2+\epsilon_2^2+\epsilon_3^2}{3}}$$

    Parameters
    ----------
    e1 : float
        Normalised residual of the first identity [-].
    e2 : float
        Normalised residual of the second identity [-].
    e3 : float
        Normalised residual of the third identity [-].

    Returns
    -------
    float
        Root-mean-square residual, NaN if any input is [-].

    Limitations
    -----------
    A summary, not a verdict: it says how badly the three relations fail to
    close together but not which one failed. Read it beside the three
    residuals, never instead of them.
    """
    values = np.asarray([e1, e2, e3], dtype=float)
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))
def virial_alpha_approx_from_kappa(kappa: float) -> float:
    r"""Virial closure coefficient from the boundary elongation (Bongard $\hat\alpha_1$).

    $$\hat\alpha_1 = \frac{2\kappa^2}{1 + \kappa^2}$$

    Parameters
    ----------
    kappa : float
        Boundary elongation $\kappa$ [-].

    Returns
    -------
    float
        Closure coefficient $\hat\alpha_1$ [-].

    Convention
    ----------
    $\alpha$ here is the coefficient multiplying $l_i$ in the third virial
    relation, defined by the volume ratio
    $\alpha = 2\langle R B_Z^2\rangle / \langle R B_p^2\rangle$ over the plasma,
    so that $\alpha\to1$ for an up-down and in-out symmetric circular
    cross-section and $\alpha\to2$ in the infinitely elongated limit.  The
    elongation must come from the same convention throughout: VAFT's
    :func:`elongation_from_RZ_boundary` and the IMAS ``boundary.elongation``
    are both the bounding-box $(Z_{max}-Z_{min})/(2a)$, while a Miller fit
    $\kappa$ is a different measure of the same contour and shifts
    $\hat\alpha_1$ by a few percent.

    Physical interpretation
    -----------------------
    Replaces the volume integral over the plasma interior by a function of the
    boundary shape alone, so $\alpha$ can be obtained from a reconstruction
    that determines the boundary but not the internal poloidal field --
    a filament or current-element model, for instance.

    Validity
    --------
    Bongard et al. report roughly 10% over the aspect-ratio range they tested,
    with slightly less spread than their annulus estimate.  Measured here
    against the volume integral of an analytic Solov'ev equilibrium over
    $\kappa\in[1,2]$ and $\epsilon\in[0.4,0.8]$, the ordering comes out the
    other way: this form has bias $-3.6\%$, scatter $2.7\%$ and worst case
    $11.4\%$, against $-1.0\%$, $1.0\%$ and $2.8\%$ for
    :func:`vaft.process.equilibrium.virial_alpha_conformal_annulus`.  Pick
    between them on evidence from the geometry at hand, not on either claim.

    The error is systematically negative and largest for a round, fat plasma:
    $\hat\alpha_1$ depends only on $\kappa$, so it cannot represent the
    aspect-ratio contribution that holds the true $\alpha$ above 1 even at
    $\kappa=1$ (where it runs 1.03 to 1.14 across that $\epsilon$ range while
    this returns 1.00).  $\kappa=1$ is therefore where the approximation is
    weakest, not where it is exact.

    The Bongard closure divides by $3\alpha-2$, so an $\alpha$ error grows on
    its way into $l_i$: near $\alpha=1.4$ the logarithmic sensitivity
    $-3\alpha/(3\alpha-2)$ is $-1.9$, and a finite 10% error costs about 16%.
    See :func:`virial_li_from_S_alpha_mu`.

    References
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), low-aspect-ratio
           virial closure (journal page not recorded in the VAFT source).
    """
    kappa = float(kappa)
    den = 1.0 + kappa**2
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return 2.0 * kappa**2 / den
def virial_alpha_from_R_Bz_Bp_dl(R: np.ndarray,
                                 B_Z: np.ndarray,
                                 B_p: np.ndarray,
                                 dl: np.ndarray,
                                 weight: np.ndarray = None) -> float:
    r"""Virial closure coefficient from the boundary field (thin-annulus $\hat\alpha_2$).

    $$\hat\alpha_2 = \frac{2\oint R\,B_Z^2\,w\,dl}{\oint R\,B_p^2\,w\,dl}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of each boundary segment [m].
    B_Z : np.ndarray
        Vertical field component on each segment [T].
    B_p : np.ndarray
        Poloidal field magnitude on each segment [T].
    dl : np.ndarray
        Arc length of each segment [m].
    weight : np.ndarray, optional
        Annulus width per unit arc length $w$, in the same segment order;
        default ones, the uniform-offset annulus [m].

    Returns
    -------
    float
        Closure coefficient $\hat\alpha_2$ in the thin-annulus limit [-].

    Convention
    ----------
    Same normalisation as :func:`virial_alpha_approx_from_kappa`: the volume
    ratio $2\langle R B_Z^2\rangle/\langle R B_p^2\rangle$, which is 1 for a
    symmetric circular cross-section.  Every array is indexed by boundary
    segment and must share one ordering; ``B_Z`` and ``B_p`` are components of
    the same field, so $|B_Z| \le B_p$ segment by segment.

    Physical interpretation
    -----------------------
    A thin annulus of local width $t\,w$ has area element $dA = t\,w\,dl$, and
    the constant $t$ cancels between numerator and denominator, so this is the
    $t\to0$ limit of an annulus estimate.  **The weight is what makes the limit
    exact, and it depends on how the inner contour was constructed.**  For a
    uniform-offset annulus $w=1$.  For an annulus conformal to the boundary --
    the inner contour a copy scaled by $1-t$ about a centre $c$, as
    :func:`vaft.process.equilibrium.virial_alpha_conformal_annulus` builds it --
    the width varies around the contour and
    $w = (\mathbf{x} - \mathbf{c})\cdot\hat{\mathbf{n}}$, the support function of
    the boundary about $c$.  Passing $w=1$ against a conformal annulus does not
    converge to it: on a VEST-like Solov'ev boundary the two limits differ by
    about 5%.  :func:`vaft.process.equilibrium.virial_alpha_thin_annulus`
    assembles the geometry and calls this function with the right weight.

    Validity
    --------
    The limiting form of an estimate reported accurate to roughly 10% by
    Bongard et al.  How fast a given equilibrium approaches the limit is a
    property of that equilibrium, not of this quadrature: a boundary whose
    field varies strongly across the annulus approaches it more slowly.

    Numerical notes
    ---------------
    Plain weighted sum over the supplied segments; the caller chooses the
    quadrature by choosing the segment values.  Returns ``nan`` rather than
    raising when the denominator is zero or non-finite, matching
    :func:`vaft.process.equilibrium.efit_virial_volume_integrals`.

    References
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), low-aspect-ratio
           virial closure (journal page not recorded in the VAFT source).
    """
    R = np.asarray(R, dtype=float)
    B_Z = np.asarray(B_Z, dtype=float)
    B_p = np.asarray(B_p, dtype=float)
    dl = np.asarray(dl, dtype=float)
    w = np.ones_like(dl) if weight is None else np.asarray(weight, dtype=float)
    num = float(np.sum(R * B_Z**2 * w * dl))
    den = float(np.sum(R * B_p**2 * w * dl))
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return float("nan")
    return 2.0 * num / den
def virial_S1_approx() -> float:
    r"""Leading-order value of the first Shafranov integral, $S_1 = 2$.

    $$S_1 = 2 + O(\epsilon, D_0, \delta)$$

    Returns
    -------
    float
        The constant 2 [-].

    Convention
    ----------
    Lao/EFIT normalisation of the surface integrals by $B_{pa}$ and the plasma
    volume, in which $S_1\to2$ for a circular, unshifted, large-aspect-ratio
    boundary.

    Validity
    --------
    Valid to first order in inverse aspect ratio, Shafranov shift $D_0$ and
    triangularity; a placeholder when the surface integral itself is unavailable.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Sec. III.
    .. [2] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421.
    """
    return 2.0
def virial_S2_approx_from_D0_a_R0(eK: float,
                                  D0: float,
                                  a_minor: float,
                                  R0: float) -> float:
    r"""Analytic approximation of the second Shafranov integral for an elongated boundary.

    $$S_2 = -\frac{2a}{R_0}\,(D_0 + 1)\left(1 + \frac{e_K}{2}\right)$$

    Parameters
    ----------
    eK : float
        Elongation parameter $(\kappa^2-1)/(\kappa^2+1)$ [-].
    D0 : float
        Normalised Shafranov shift of the boundary [-].
    a_minor : float
        Minor radius [m].
    R0 : float
        Reference major radius [m].

    Returns
    -------
    float
        $S_2$ [-].

    Convention
    ----------
    Lao/EFIT normalisation of the surface integrals; $D_0$ as defined by
    Martynov and Pustovitov (shift over minor radius).

    Validity
    --------
    First order in $a/R_0$ and in the shift; elongation enters only through
    $e_K$.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eq. (21).
    """
    return -(2 * a_minor / R0) * (D0 + 1) * (1 + eK / 2)
def virial_S3_approx_from_eK_d(eK: float,
                              d_param: float) -> float:
    r"""Analytic approximation of the third Shafranov integral for an elongated boundary.

    $$S_3 = 1 - \frac{e_K}{2} - \delta\left(1 - \frac{e_K^2}{2}\right)$$

    Parameters
    ----------
    eK : float
        Elongation parameter $(\kappa^2-1)/(\kappa^2+1)$ [-].
    d_param : float
        Triangularity-like shape parameter $\delta$ of the approximation [-].

    Returns
    -------
    float
        $S_3$ [-].

    Convention
    ----------
    Lao/EFIT normalisation of the surface integrals.

    Validity
    --------
    First order in the shape parameters.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eq. (22).
    """
    return 1 - 0.5 * eK - d_param * (1 - 0.5 * eK**2)
def virial_bp_li_lihat_from_S123(S1: float,
                                 S2: float,
                                 S3: float,
                                 a_param: float,
                                 RT_over_R0: float) -> Tuple[float, float, float]:
    r"""Deprecated historical name for the three-relation virial solve.

    Use :func:`virial_full_123_from_S_alpha_rt`, which this delegates to. The
    third return is renamed there from $\hat l_i$ to $\mu_i$: it is the
    diamagnetic parameter the pairwise closures take as an input, and the hat
    notation read as a second internal inductance.  Emits a
    ``DeprecationWarning``.

    $$3\beta_p + l_i - \hat l_i = S_1 + S_2, \qquad
      \beta_p + l_i + \hat l_i = \frac{R_T}{R_0}S_2, \qquad
      \beta_p - (\alpha-1)\,l_i - \hat l_i = S_3$$

    Parameters
    ----------
    S1 : float
        First Shafranov surface integral [-].
    S2 : float
        Second Shafranov surface integral [-].
    S3 : float
        Third Shafranov surface integral [-].
    a_param : float
        Closure coefficient $\alpha$ of the third relation [-].
    RT_over_R0 : float
        Current-centroid radius over reference radius [-].

    Returns
    -------
    beta_p : float
        Poloidal beta [-].
    li_int : float
        Internal inductance [-].
    li_hat : float
        Toroidal-field contribution $\hat l_i$ [-].

    Numerical notes
    ---------------
    Closed-form solve, delegated to :func:`virial_full_123_from_S_alpha_rt`.
    The determinant is $4(\alpha-1)$, so the system is singular at
    $\alpha = 1$ -- the same limit that makes the historical Lao $l_i$ blow up,
    because $E_1$ and $E_2$ fix $\beta_p - \mu_i$ and only $E_3$ separates $l_i$
    from it.  There all three returns are ``nan``, the framework's contract for
    a singular closure; until 0.7.0 this name used ``numpy.linalg.solve`` and
    raised ``LinAlgError`` instead.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eqs. (1)-(3).
    .. [2] V. D. Shafranov, Plasma Phys. 13 (1971) 757.
    """
    warnings.warn(
        "`virial_bp_li_lihat_from_S123` is deprecated; use "
        "`virial_full_123_from_S_alpha_rt` (same numbers, third return named mu_i). "
        "It returns (nan, nan, nan) at alpha = 1 rather than raising LinAlgError.",
        DeprecationWarning,
        stacklevel=2,
    )
    return virial_full_123_from_S_alpha_rt(S1, S2, S3, a_param, RT_over_R0)
def virial_D0_boundary_from_bp_li_eK(beta_p: float,
                                    li_int: float,
                                    eK: float,
                                    b_minor: float,
                                    R_plasma: float) -> float:
    r"""Boundary Shafranov shift $D_0(b)$ from $\beta_p$, $l_i$ and elongation.

    $$D_0(b) = -\frac{b}{2R_{\mathrm{plasma}}}\;
      \frac{2\beta_p + l_i + \tfrac{1}{2}e_K}{1 + \tfrac{1}{2}e_K}$$

    Parameters
    ----------
    beta_p : float
        Poloidal beta [-].
    li_int : float
        Internal inductance [-].
    eK : float
        Elongation parameter $(\kappa^2-1)/(\kappa^2+1)$ [-].
    b_minor : float
        Minor radius at which the shift is evaluated [m].
    R_plasma : float
        Plasma major radius [m].

    Returns
    -------
    float
        Normalised shift $D_0$ at radius ``b_minor`` [-].

    Convention
    ----------
    Negative sign convention of Martynov and Pustovitov: the magnetic axis moves
    outboard, so $D_0 < 0$ for positive $\beta_p + l_i/2$.

    Validity
    --------
    First order in $b/R$; elongation through $e_K$ only.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", Eq. (37).
    """
    numerator = 2 * beta_p + li_int + 0.5 * eK
    denominator = 1 + 0.5 * eK
    return - (b_minor / (2 * R_plasma)) * numerator / denominator


#: Derived rather than listed: see :mod:`vaft.formula._exports`.
__all__ = public_names(globals(), constants=("VIRIAL_SINGULAR_EPS",))
