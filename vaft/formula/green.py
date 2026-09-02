"""
Green's function calculations for plasma physics.

This module provides functions for calculating various Green's function integrals
used in plasma physics calculations.

Notation
--------
G      : Green's function                              [-]
K      : complete elliptic integral of first kind      [-]
E      : complete elliptic integral of second kind     [-]
"""

import numpy as np
from scipy.special import ellipe, ellipk
from typing import Union, Tuple
from vaft.compat import trapz_compat


def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    r"""Trapezoidal integral $\int y\,dx$, local copy for the Green kernels.

    $$\int y\,dx \approx \sum_i \frac{y_i + y_{i+1}}{2}\,(x_{i+1} - x_i)$$

    Parameters
    ----------
    x : np.ndarray
        Sample abscissae, monotonic [any].
    y : np.ndarray
        Integrand at the samples [any].

    Returns
    -------
    float
        Integral over the sampled range, in units of ``y`` times ``x`` [any].

    Numerical notes
    ---------------
    Identical to :func:`vaft.formula.utils.trapz_integral` (``numpy.trapezoid``
    via :func:`vaft.compat.trapz_compat`) but defined here so the Green's
    function module does not import the sklearn-heavy utilities; this copy is
    the one ``vaft.formula.trapz_integral`` resolves to.
    """
    return float(trapz_compat(y, x=x))



def calculate_distance(r1: Union[np.ndarray, float], r2: Union[np.ndarray, float], 
                      z1: Union[np.ndarray, float], z2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    r"""Euclidean distance between $(r_1, z_1)$ and $(r_2, z_2)$ in the poloidal plane.

    $$d = \sqrt{(r_2 - r_1)^2 + (z_2 - z_1)^2}$$

    Parameters
    ----------
    r1 : float or np.ndarray
        Radius of the first point(s) [m].
    r2 : float or np.ndarray
        Radius of the second point(s) [m].
    z1 : float or np.ndarray
        Height of the first point(s) [m].
    z2 : float or np.ndarray
        Height of the second point(s) [m].

    Returns
    -------
    float or np.ndarray
        Distance, broadcast over the inputs [m].

    Notes
    -----
    A distance in the $(R, Z)$ half-plane, not the 3-D distance between points
    on two current loops.
    """
    return np.sqrt((r2 - r1) ** 2 + (z2 - z1) ** 2)


# ------------------------------------------------------------------
# Elliptic Integrals
# ------------------------------------------------------------------

def complete_elliptic_integral_k(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Complete elliptic integral of the first kind $K(m)$.

    $$K(m) = \int_0^{\pi/2}\frac{d\theta}{\sqrt{1 - m\sin^2\theta}}$$

    Parameters
    ----------
    m : float or np.ndarray
        Parameter $m = k^2$, in $[0, 1)$ [-].

    Returns
    -------
    float or np.ndarray
        $K(m)$ [-].

    Convention
    ----------
    Takes the *parameter* $m = k^2$, as ``scipy.special.ellipk`` does, not the
    modulus $k$; $K \to \infty$ logarithmically as $m \to 1$.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Sec. 19.2(ii),
           https://dlmf.nist.gov/19.2.
    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*,
           Dover (1972), Sec. 17.3.
    """
    return ellipk(m)


def complete_elliptic_integral_e(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Complete elliptic integral of the second kind $E(m)$.

    $$E(m) = \int_0^{\pi/2}\sqrt{1 - m\sin^2\theta}\,d\theta$$

    Parameters
    ----------
    m : float or np.ndarray
        Parameter $m = k^2$, in $[0, 1]$ [-].

    Returns
    -------
    float or np.ndarray
        $E(m)$, from $\pi/2$ at $m = 0$ to 1 at $m = 1$ [-].

    Convention
    ----------
    Takes the parameter $m = k^2$, as ``scipy.special.ellipe`` does.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Sec. 19.2(ii),
           https://dlmf.nist.gov/19.2.
    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*,
           Dover (1972), Sec. 17.3.
    """
    return ellipe(m)


# ------------------------------------------------------------------
# Green's Functions
# ------------------------------------------------------------------

def greens_function_2d(R: np.ndarray,
                      Z: np.ndarray,
                      R0: float,
                      Z0: float) -> np.ndarray:
    r"""Axisymmetric kernel $\sqrt{RR_0}\,K(m)$ between a field point and a ring source.

    $$G_{2D} = \sqrt{R R_0}\;K(m), \qquad
      m = \frac{4RR_0}{(R + R_0)^2 + (Z - Z_0)^2}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the field points [m].
    Z : np.ndarray
        Height of the field points [m].
    R0 : float
        Major radius of the ring source [m].
    Z0 : float
        Height of the ring source [m].

    Returns
    -------
    np.ndarray
        Kernel value at each field point [m].

    Convention
    ----------
    This is the leading term of the ring-current flux function, *not* the flux
    Green's function itself: the poloidal flux per unit current is
    $\mu_0\sqrt{RR_0}\,[(2 - m)K(m) - 2E(m)]/\sqrt{m}$
    (:func:`greens_function_exact`, :func:`green_psi_exact`).  Use those for
    any physical field; this kernel is kept for the legacy integrals below.

    Limitations
    -----------
    Diverges logarithmically at the source point; no guard.

    References
    ----------
    .. [1] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999),
           Sec. 5.5 (vector potential of a circular loop).
    """
    k2 = 4 * R * R0 / ((R + R0)**2 + (Z - Z0)**2)
    return np.sqrt(R * R0) * complete_elliptic_integral_k(k2)


def greens_function_3d(R: np.ndarray,
                      Z: np.ndarray,
                      phi: np.ndarray,
                      R0: float,
                      Z0: float,
                      phi0: float) -> np.ndarray:
    r"""Toroidal-angle-resolved variant of the $\sqrt{RR_0}\,K(m)$ kernel.

    $$G_{3D} = \sqrt{R R_0}\;K(m), \qquad
      m = \frac{4RR_0}{(R + R_0)^2 + (Z - Z_0)^2 + 4RR_0\sin^2\!\big(\tfrac{\varphi - \varphi_0}{2}\big)}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the field points [m].
    Z : np.ndarray
        Height of the field points [m].
    phi : np.ndarray
        Toroidal angle of the field points [rad].
    R0 : float
        Major radius of the source [m].
    Z0 : float
        Height of the source [m].
    phi0 : float
        Toroidal angle of the source [rad].

    Returns
    -------
    np.ndarray
        Kernel value at each field point [m].

    Limitations
    -----------
    The $\sin^2$ term augments the denominator with the 3-D chord between the
    two toroidal angles, but the resulting expression is a heuristic
    generalisation of :func:`greens_function_2d` without a recorded derivation
    or source; it is not the Biot-Savart kernel of a point source.  Tracked in
    #356.

    References
    ----------
    .. [1] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999),
           Sec. 5.5.
    """
    k2 = 4 * R * R0 / ((R + R0)**2 + (Z - Z0)**2 + 4 * R * R0 * np.sin((phi - phi0)/2)**2)
    return np.sqrt(R * R0) * complete_elliptic_integral_k(k2)


# ------------------------------------------------------------------
# Green's Function Integrals
# ------------------------------------------------------------------

def greens_integral_2d(R: np.ndarray,
                      Z: np.ndarray,
                      R0: float,
                      Z0: float,
                      f: np.ndarray) -> float:
    r"""Line integral along $R$ of the 2-D kernel times a source density.

    $$I = \int G_{2D}(R, Z; R_0, Z_0)\,f(R)\,dR$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the samples, monotonic [m].
    Z : np.ndarray
        Height of the samples [m].
    R0 : float
        Major radius of the source [m].
    Z0 : float
        Height of the source [m].
    f : np.ndarray
        Source density at the samples [any].

    Returns
    -------
    float
        Integral in units of ``f`` times m^2 [any].

    Limitations
    -----------
    Integrates along the single supplied ``R`` array (a line, not an area) with
    the kernel of :func:`greens_function_2d`; not a flux or field.

    Numerical notes
    ---------------
    Trapezoidal rule.
    """
    G = greens_function_2d(R, Z, R0, Z0)
    return trapz_integral(R, G * f)


def greens_integral_3d(R: np.ndarray,
                      Z: np.ndarray,
                      phi: np.ndarray,
                      R0: float,
                      Z0: float,
                      phi0: float,
                      f: np.ndarray) -> float:
    r"""Line integral along $R$ of the 3-D kernel times a source density.

    $$I = \int G_{3D}(R, Z, \varphi; R_0, Z_0, \varphi_0)\,f(R)\,dR$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the samples, monotonic [m].
    Z : np.ndarray
        Height of the samples [m].
    phi : np.ndarray
        Toroidal angle of the samples [rad].
    R0 : float
        Major radius of the source [m].
    Z0 : float
        Height of the source [m].
    phi0 : float
        Toroidal angle of the source [rad].
    f : np.ndarray
        Source density at the samples [any].

    Returns
    -------
    float
        Integral in units of ``f`` times m^2 [any].

    Limitations
    -----------
    Same caveats as :func:`greens_integral_2d` and :func:`greens_function_3d`.

    Numerical notes
    ---------------
    Trapezoidal rule along ``R`` only.
    """
    G = greens_function_3d(R, Z, phi, R0, Z0, phi0)
    return trapz_integral(R, G * f)


def elliptic_integral(r_obs: np.ndarray, z_obs: np.ndarray, r_src: float, z_src: float) -> tuple:
    r"""Polynomial approximations of $K$ and $E$ for a ring source and observer points.

    $$m = \frac{4r_{obs}r_{src}}{(r_{obs} + r_{src})^2 + (z_{obs} - z_{src})^2}, \qquad
      m_1 = 1 - m$$

    then the Hastings polynomials
    $K \approx \sum a_km_1^k + \ln(1/m_1)\sum b_km_1^k$ and
    $E \approx \sum c_km_1^k + \ln(1/m_1)\sum d_km_1^k$ to fourth order.

    Parameters
    ----------
    r_obs : np.ndarray
        Radius of the observation points [m].
    z_obs : np.ndarray
        Height of the observation points [m].
    r_src : float
        Radius of the ring source [m].
    z_src : float
        Height of the ring source [m].

    Returns
    -------
    ek : np.ndarray
        Approximate $K(m)$ [-].
    ee : np.ndarray
        Approximate $E(m)$ [-].

    Validity
    --------
    Abramowitz and Stegun 17.3.34 and 17.3.36, absolute error below
    $2\times10^{-8}$ for $0 \le m < 1$; the legacy path of :func:`green_r` and
    :func:`green_br_bz`, retained so their numbers do not change.

    Limitations
    -----------
    $\ln(1/m_1)$ diverges at a coincident observer and source and a warning is
    emitted with a bare ``print`` rather than ``warnings.warn`` (tracked in
    #356); :func:`greens_function_exact` uses scipy's exact integrals instead.

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*,
           Dover (1972), Eqs. 17.3.34 and 17.3.36.
    .. [2] C. Hastings, *Approximations for Digital Computers*, Princeton
           University Press (1955).
    """
    ak0 = 1.386294361120
    ak1 = 0.096663442590
    ak2 = 0.035900923830
    ak3 = 0.037425637130
    ak4 = 0.014511962120

    bk0 = 0.500000000000
    bk1 = 0.124985935970
    bk2 = 0.068802485760
    bk3 = 0.033283553460
    bk4 = 0.004417870120

    ae0 = 1.000000000000
    ae1 = 0.443251414630
    ae2 = 0.062606012200
    ae3 = 0.047573835460
    ae4 = 0.017365064510

    be0 = 0.000000000000
    be1 = 0.249983683100
    be2 = 0.092001800370
    be3 = 0.040696975260
    be4 = 0.005264496390

    z_val = z_obs - z_src # z_obs is array, z_src is scalar -> z_val is array
    zsq = z_val * z_val   # array
    s = r_obs + r_src     # r_obs is array, r_src is scalar -> s is array
    s2 = s * s            # array
    # a2 must be calculated carefully for broadcasting if r_obs is an array
    # a2 = 4.0 * r_obs * r_src # array * scalar -> array

    # k2 calculation:
    # denom_k2 = s2 + zsq (array)
    # num_k2 = 4.0 * r_obs * r_src (array)
    k2 = (4.0 * r_obs * r_src) / (s2 + zsq) # array
    
    kp2 = 1.0 - k2 # array
    
    # Handle potential division by zero or log of non-positive if kp2 is very small or zero.
    # For simplicity, we'll rely on numpy's handling (e.g., log(0) -> -inf, log(negative) -> nan)
    # but in a robust implementation, one might add checks or epsilons.
    # A small epsilon can be added to kp2 to avoid log(0) if necessary,
    # or use np.errstate to manage warnings/errors.
    # For now, let's assume kp2 will be positive.
    
    # Check for kp2 being too close to zero, which can cause issues with log.
    # This warning will now print for each element where condition is met.
    # Consider if a vectorized warning is needed or if individual warnings are acceptable.
    if np.any(np.abs(kp2) < 1e-15):
        # This is a simplified warning for demonstration.
        # In practice, you might want to log specific indices or handle differently.
        print(f"Warning: kp2 ~ 0 for some r_obs/z_obs points with r_src={r_src}, z_src={z_src}")


    # Approximate logs
    kln = -np.log(kp2) # array

    # Elliptic integral of the first kind
    ek = (
        ak0
        + kp2 * (ak1 + kp2 * (ak2 + kp2 * (ak3 + kp2 * ak4)))
        + kln
        * (
            bk0
            + kp2 * (bk1 + kp2 * (bk2 + kp2 * (bk3 + kp2 * bk4)))
        )
    ) # array

    # Elliptic integral of the second kind
    ee = (
        ae0
        + kp2 * (ae1 + kp2 * (ae2 + kp2 * (ae3 + kp2 * ae4)))
        + kln
        * (
            be0
            + kp2 * (be1 + kp2 * (be2 + kp2 * (be3 + kp2 * be4)))
        )
    ) # array
    # Corrected typo for 'ee' calculation:
    # ee = (
    #     ae0
    #     + kp2 * (ae1 + kp2 * (ae2 + kp2 * (ae3 + kp2 * ae4)))
    #     + kln
    #     * (
    #         be0
    #         + kp2 * (be1 + kp2 * (be2 + kp2 * (be3 + kp2 * be4)))
    #     )
    # )


    return ek, ee


def green_br_bz(r_obs: np.ndarray, z_obs: np.ndarray, r_src: float, z_src: float) -> tuple:
    r"""Field of a unit-current ring, $(B_r, B_z)$ at observer points (legacy elliptic path).

    $$B_r = \frac{\mu_0}{2\pi r}\,\frac{z - z_0}{\sqrt{(r + r_0)^2 + (z - z_0)^2}}
      \left[\frac{r^2 + r_0^2 + (z - z_0)^2}{(r - r_0)^2 + (z - z_0)^2}E - K\right], \quad
      B_z = \frac{\mu_0}{2\pi}\,\frac{1}{\sqrt{(r + r_0)^2 + (z - z_0)^2}}
      \left[K - \frac{r^2 - r_0^2 + (z - z_0)^2}{(r - r_0)^2 + (z - z_0)^2}E\right]$$

    Parameters
    ----------
    r_obs : np.ndarray
        Radius of the observation points [m].
    z_obs : np.ndarray
        Height of the observation points [m].
    r_src : float
        Radius of the current ring [m].
    z_src : float
        Height of the current ring [m].

    Returns
    -------
    br : np.ndarray
        Radial field per ampere [T/A].
    bz : np.ndarray
        Vertical field per ampere [T/A].

    Convention
    ----------
    Right-handed $(r, \varphi, z)$ with the current flowing in $+\varphi$; $B_z$
    is positive inside the ring.  SI throughout; no $2\pi$ flux ambiguity
    arises for fields.  Uses the approximate :func:`elliptic_integral`; the
    exact, broadcasting counterpart is :func:`green_br_bz_exact`.

    Limitations
    -----------
    Divides by $(r - r_0)^2 + (z - z_0)^2$ and by $r_{obs}$ without guards, so
    a coincident point or an on-axis observer gives ``inf``/``nan`` (the
    callers' shifted-evaluation scheme avoids exact coincidence).  Tracked in
    #356.

    References
    ----------
    .. [1] W. R. Smythe, *Static and Dynamic Electricity*, 3rd ed., McGraw-Hill
           (1968), Sec. 7.10 (field of a circular loop).
    .. [2] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999),
           Sec. 5.5.
    """
    mu0 = 4.0 * np.pi * 1.0e-7 # Use np.pi
    z_diff = z_obs - z_src # array

    # Elliptic part - r_obs, z_obs are arrays, r_src, z_src are scalars
    ek, ee = elliptic_integral(r_obs, z_obs, r_src, z_src) # ek, ee are arrays

    denom_sqrt = np.sqrt((r_obs + r_src) ** 2 + z_diff ** 2) # array
    
    # Br
    # Denominator for the second term of Br and Bz factor
    # This term can be zero if r_obs = r_src and z_obs = z_src.
    # ((r_obs - r_src) ** 2 + z_diff ** 2)
    # Add a small epsilon to avoid division by zero, or handle this case specifically.
    # For simplicity in this step, let's assume it's not exactly zero,
    # or that the calling function (compute_br_bz_phi) handles singularities.
    
    br_denom_factor = (r_obs - r_src) ** 2 + z_diff ** 2 # array
    # To prevent division by zero, ensure br_denom_factor is not zero.
    # A common approach is to add a small epsilon, or use np.where.
    # For now, let's assume the shift mechanism in compute_br_bz_phi handles exact singularities.
    
    br_num = z_diff / denom_sqrt # array
    br_factor = (((r_obs * r_obs + r_src * r_src + z_diff * z_diff) / br_denom_factor) * ee - ek) # array
    br = br_num * br_factor * mu0 / (2.0 * np.pi * r_obs) # array
    # Note: Division by r_obs can be problematic if r_obs contains zero.
    # This needs to be handled, e.g. by setting Br to 0 or another appropriate value at r_obs=0.
    # For now, assuming r_obs will be non-zero in typical use cases for Br.

    # Bz
    bz_num = 1.0 / denom_sqrt # array
    bz_factor = (ek - ee * (r_obs * r_obs - r_src * r_src + z_diff * z_diff) / br_denom_factor) # array
    bz = bz_num * bz_factor * mu0 / (2.0 * np.pi) # array

    return br, bz


# ------------------------------------------------------------------
# Exact Green's functions (scipy elliptic integrals) and coil inductances
#
# Additive API (see issue #219): the approximate `elliptic_integral` /
# `green_r` / `green_br_bz` above are kept unchanged for backward
# compatibility; the functions below use scipy's exact K/E and support
# full NumPy broadcasting over observation and source coordinates.
#
# Conventions (unit source current):
#   G(r, z; r0, z0) = sqrt(r*r0)/k * [(2 - k^2) K(k) - 2 E(k)],
#   k^2 = m = 4 r r0 / [(r + r0)^2 + (z - z0)^2]
#   psi [Wb]  = mu0 * G                (same convention as `green_r`)
#   Bz  [T]   = +mu0/(2 pi r) dG/dr
#   Br  [T]   = -mu0/(2 pi r) dG/dz
# ------------------------------------------------------------------

from vaft.formula.constants import MU0

GREEN_EXACT_MODES = ("psi", "dpsi_dr", "dpsi_dz", "d2psi_drdz", "d2psi_dr2", "K", "E")

# Largest double strictly below 1 — keeps K(m), E(m) and 1/(1-m) finite at
# the coincident-point singularity instead of returning inf/NaN.
_M_MAX = np.nextafter(1.0, 0.0)


def greens_function_exact(r, z, r0, z0, mode: str = "psi"):
    r"""Exact free-space axisymmetric Green's function and its derivatives.

    $$G(r, z; r_0, z_0) = \frac{\sqrt{rr_0}}{k}\left[(2 - k^2)K(k^2) - 2E(k^2)\right], \qquad
      k^2 = m = \frac{4rr_0}{(r + r_0)^2 + (z - z_0)^2}$$

    so that $\psi = \mu_0 I\,G$ is the poloidal flux of a ring current $I$;
    ``mode`` selects $G$, $\partial G/\partial r$, $\partial G/\partial z$,
    $\partial^2G/\partial r\partial z$, $\partial^2G/\partial r^2$, or the raw
    $K$ and $E$.

    Parameters
    ----------
    r : array-like
        Observation major radius [m].
    z : array-like
        Observation height [m].
    r0 : array-like
        Source major radius [m].
    z0 : array-like
        Source height [m].
    mode : str, optional
        One of ``GREEN_EXACT_MODES``; default ``"psi"`` [str].

    Returns
    -------
    np.ndarray
        Requested quantity, broadcast over all four coordinates [any].
        $G$ is in m, its derivatives in m per m, $K$ and $E$ dimensionless.

    Raises
    ------
    ValueError
        Unknown ``mode`` or coordinates giving $m$ outside $[0, 1]$.

    Convention
    ----------
    Full weber: $\psi = \mu_0 G$ per ampere (same as :func:`green_r`), and
    $B_z = +\mu_0/(2\pi r)\,\partial G/\partial r$, $B_r = -\mu_0/(2\pi r)\,
    \partial G/\partial z$.  The equilibrium helpers assume flux per radian by
    default, so divide by $2\pi$ (or pass ``cocos``) before feeding this flux to
    :func:`vaft.formula.equilibrium.vertical_magnetic_field_from_psi`.

    Limitations
    -----------
    Points with $rr_0 = 0$ return their analytic limits (0 for every mode except
    ``d2psi_dr2``, whose on-axis limit is $\pi r_0^2/(r_0^2 + (z - z_0)^2)^{3/2}$).
    The elliptic parameter is clamped just below 1, so a coincident observer and
    source return finite numbers, but those are artifacts of the clamp, not
    physical limits (the ideal-filament self term diverges): handle genuinely
    coincident pairs yourself, e.g. with :func:`self_inductance` or a
    shifted-evaluation scheme such as ``compute_br_bz_phi``.

    Numerical notes
    ---------------
    ``scipy.special.ellipk``/``ellipe`` on the parameter $m$; every $1/k$,
    $1/m$ and $1/r$ is guarded and overwritten on axis.  The ``d2psi_dr2``
    branch corrects a sign error in the legacy VFIT ``getGreenFunction.m``
    case 7 ($-2(r + r_0)A$, from $d(1/D)/dr = -D'/D^2$) and is validated
    against finite differences of ``dpsi_dr``.

    References
    ----------
    .. [1] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999),
           Sec. 5.5, Eq. (5.37).
    .. [2] Legacy VFIT ``getGreenFunction.m`` (modes 1-7), the reference
           algorithm.
    """
    if mode not in GREEN_EXACT_MODES:
        raise ValueError(f"mode must be one of {GREEN_EXACT_MODES}, got {mode!r}")

    r, z, r0, z0 = np.broadcast_arrays(
        np.asarray(r, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(r0, dtype=float),
        np.asarray(z0, dtype=float),
    )

    dz2 = (z - z0) ** 2
    denom = (r + r0) ** 2 + dz2
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.divide(4.0 * r * r0, denom, out=np.zeros_like(denom), where=denom != 0)

    if np.any(m > 1.0 + 1e-9) or np.any(m < -1e-9):
        raise ValueError("elliptic parameter m outside [0, 1]: invalid coordinates")
    m = np.clip(m, 0.0, _M_MAX)

    KK = ellipk(m)
    EE = ellipe(m)
    if mode == "K":
        return KK
    if mode == "E":
        return EE

    on_axis = m == 0.0
    # Guard every 1/k, 1/m, 1/r factor; on-axis entries are overwritten below.
    m_safe = np.where(on_axis, 1.0, m)
    k = np.sqrt(m_safe)
    r_safe = np.where(r == 0.0, 1.0, r)
    r0_safe = np.where(r0 == 0.0, 1.0, r0)
    sqrt_rr0 = np.sqrt(r_safe * r0_safe)
    one_m = 1.0 - m  # clamped above, so strictly positive

    if mode == "psi":
        out = sqrt_rr0 / k * ((2.0 - m_safe) * KK - 2.0 * EE)
        return np.where(on_axis, 0.0, out)

    # Shared building blocks for the derivative modes (getGreenFunction.m).
    Gk = -sqrt_rr0 / m_safe * (2.0 * KK - (1.0 + 1.0 / one_m) * EE)
    kr = k / (2.0 * r_safe) * (r0**2 - r**2 + dz2) / denom
    kz = -(k**3) * (z - z0) / (4.0 * r0_safe * r_safe)

    if mode == "dpsi_dr":
        Gr = (0.5 * r0) / (k * sqrt_rr0) * ((2.0 - m_safe) * KK - 2.0 * EE)
        out = Gr + Gk * kr
        return np.where(on_axis, 0.0, out)

    if mode == "dpsi_dz":
        return np.where(on_axis, 0.0, Gk * kz)

    Gkk = (-5.0 + 1.0 / one_m) * KK + (m_safe**2 - 7.0 * m_safe + 4.0) / one_m**2 * EE
    Gkk = -Gkk * sqrt_rr0 / (m_safe * k)
    Gkr = Gk / (2.0 * r_safe)

    if mode == "d2psi_drdz":
        Gk_dr = Gkr + Gkk * kr
        kzr = (z - z0) * m_safe * k / (4.0 * r0_safe * r_safe**2)
        kzk = (-3.0 * m_safe) * (z - z0) / (4.0 * r0_safe * r_safe)
        kz_dr = kzr + kzk * kr
        out = Gk_dr * kz + kz_dr * Gk
        return np.where(on_axis, 0.0, out)

    # mode == "d2psi_dr2"
    # krr = d(kr)/dr. NOTE: legacy getGreenFunction.m case 7 carries a sign
    # error here (+2(r+r0)A instead of -2(r+r0)A in the last numerator term,
    # from d/dr[1/D] = -D'/D^2); the corrected form below is validated
    # against finite differences of dpsi_dr.
    krr = (
        kr / (2.0 * r_safe) * (r0**2 - r**2 + dz2) / denom
        - k / (2.0 * r_safe**2) * (r0**2 - r**2 + dz2) / denom
        + k
        / (2.0 * r_safe)
        * (-2.0 * r * denom - 2.0 * (r + r0) * (r0**2 - r**2 + dz2))
        / denom**2
    )
    Gr = (0.5 * r0) / (k * sqrt_rr0) * ((2.0 - m_safe) * KK - 2.0 * EE)
    Grr = -Gr / (2.0 * r_safe)
    out = Grr + 2.0 * Gkr * kr + Gkk * kr**2 + krr * Gk
    # On-axis limit is nonzero for this mode: near the axis
    # G ~ r^2 * pi r0^2 / (2 (r0^2 + dz^2)^{3/2}), so d2G/dr2 -> the
    # curvature of that parabola (0 when the *source* is on axis).
    with np.errstate(divide="ignore", invalid="ignore"):
        axis_limit = np.where(
            r0 == 0.0, 0.0, np.pi * r0**2 / (r0**2 + dz2) ** 1.5
        )
    return np.where(on_axis, axis_limit, out)


def green_psi_exact(r_obs, z_obs, r_src, z_src) -> np.ndarray:
    r"""Poloidal flux per unit ring current, exact elliptic integrals.

    $$\psi = \mu_0\,G(r, z; r_0, z_0)$$

    with $G$ from :func:`greens_function_exact`.

    Parameters
    ----------
    r_obs : array-like
        Observation major radius [m].
    z_obs : array-like
        Observation height [m].
    r_src : array-like
        Source major radius [m].
    z_src : array-like
        Source height [m].

    Returns
    -------
    np.ndarray
        Flux per ampere, broadcast over observation and source arrays [Wb/A].

    Convention
    ----------
    Full weber, the IMAS Data Dictionary storage (COCOS 11-18), same as
    :func:`green_r`.  The per-radian flux of an EFIT g-file or of
    :func:`vaft.formula.equilibrium.poloidal_field_factor` with ``cocos=None``
    is this divided by $2\pi$; mixing the two silently mis-scales fields by
    $2\pi$.  Tracked in #354.

    References
    ----------
    .. [1] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999),
           Sec. 5.5.
    """
    return MU0 * greens_function_exact(r_obs, z_obs, r_src, z_src, "psi")


def green_br_bz_exact(r_obs, z_obs, r_src, z_src) -> Tuple[np.ndarray, np.ndarray]:
    r"""$(B_r, B_z)$ per unit ring current, exact elliptic integrals, broadcasting.

    $$B_r = -\frac{\mu_0}{2\pi r}\,\frac{\partial G}{\partial z}, \qquad
      B_z = +\frac{\mu_0}{2\pi r}\,\frac{\partial G}{\partial r}$$

    Parameters
    ----------
    r_obs : array-like
        Observation major radius [m].
    z_obs : array-like
        Observation height [m].
    r_src : array-like
        Source major radius [m].
    z_src : array-like
        Source height [m].

    Returns
    -------
    br : np.ndarray
        Radial field per ampere [T/A].
    bz : np.ndarray
        Vertical field per ampere [T/A].

    Convention
    ----------
    Right-handed $(r, \varphi, z)$, current in $+\varphi$; consistent with
    :func:`green_psi_exact` through $B_z = (1/r)\,\partial\psi/\partial r
    \times 1/2\pi$ for a full-weber $\psi$.

    Limitations
    -----------
    Observation points on the geometric axis ($r_{obs} = 0$) return the analytic
    limits $B_r = 0$, $B_z = \mu_0 r_0^2/(2(r_0^2 + (z - z_0)^2)^{3/2})$; the
    coincident-point caveat of :func:`greens_function_exact` applies.

    References
    ----------
    .. [1] W. R. Smythe, *Static and Dynamic Electricity*, 3rd ed., McGraw-Hill
           (1968), Sec. 7.10.
    """
    r_obs, z_obs, r_src, z_src = np.broadcast_arrays(
        np.asarray(r_obs, dtype=float),
        np.asarray(z_obs, dtype=float),
        np.asarray(r_src, dtype=float),
        np.asarray(z_src, dtype=float),
    )
    on_axis = r_obs == 0.0
    r_safe = np.where(on_axis, 1.0, r_obs)
    coeff = MU0 / (2.0 * np.pi * r_safe)
    br = -coeff * greens_function_exact(r_obs, z_obs, r_src, z_src, "dpsi_dz")
    bz = coeff * greens_function_exact(r_obs, z_obs, r_src, z_src, "dpsi_dr")
    if np.any(on_axis):
        dz2 = (z_obs - z_src) ** 2
        bz_axis = MU0 * r_src**2 / (2.0 * (r_src**2 + dz2) ** 1.5)
        bz = np.where(on_axis, bz_axis, bz)
        br = np.where(on_axis, 0.0, br)
    return br, bz


def _rect_coil_midpoints(
    rc: float,
    zc: float,
    dr: float,
    dz: float,
    tilt: float,
    dl: float,
    curvature_correction: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Midpoint subdivision of a rectangular cross-section coil.

    Ports the subdivision scheme of ``getMutualInductanceCoil.m``:
    cell size chosen from *dl*, midpoint filaments laid out along the
    (possibly tilted) cross-section, with the toroidal curvature
    correction r -> r * (1 + (cos(tilt) * dr_cell / r)^2 / 24).
    """
    nr = max(int(np.floor(dr / dl + 0.5)), 1)
    nz = max(int(np.floor(dz / dl + 0.5)), 1)
    ddr = dr / nr
    ddz = dz / nz
    ir = np.arange(1, nr + 1, dtype=float)
    iz = np.arange(1, nz + 1, dtype=float)
    IR, IZ = np.meshgrid(ir, iz, indexing="ij")
    ct, st = np.cos(tilt), np.sin(tilt)
    r = rc - 0.5 * (dr * ct - dz * st) - (IZ - 0.5) * ddz * st + (IR - 0.5) * ddr * ct
    if curvature_correction:
        r = r * (1.0 + (ct * ddr / r) ** 2 / 24.0)
    zz = zc - 0.5 * (dr * st + dz * ct) + (IZ - 0.5) * ddz * ct + (IR - 0.5) * ddr * st
    return r.ravel(), zz.ravel()


def mutual_inductance(
    r1: float,
    z1: float,
    dr1: float,
    dz1: float,
    r2: float,
    z2: float,
    dr2: float,
    dz2: float,
    *,
    tilt1: float = 0.0,
    tilt2: float = 0.0,
    turns1: float = 1.0,
    turns2: float = 1.0,
    mu_r: float = 1.0,
    n_div: int = 5,
) -> float:
    r"""Mutual inductance between two rectangular-cross-section ring coils.

    $$M = N_1N_2\,\mu_r\mu_0\,\big\langle G(r_i, z_i; r_j, z_j)\big\rangle_{i \in 1,\ j \in 2}$$

    the Neumann formula evaluated as the mean flux linkage between midpoint
    filaments laid over each cross-section.

    Parameters
    ----------
    r1 : float
        Centre radius of coil 1 [m].
    z1 : float
        Centre height of coil 1 [m].
    dr1 : float
        Radial width of coil 1; 0 degenerates to a filament [m].
    dz1 : float
        Height of coil 1; 0 degenerates to a filament [m].
    r2 : float
        Centre radius of coil 2 [m].
    z2 : float
        Centre height of coil 2 [m].
    dr2 : float
        Radial width of coil 2 [m].
    dz2 : float
        Height of coil 2 [m].
    tilt1 : float, optional
        Tilt of cross-section 1 about its centre; default 0 [rad].
    tilt2 : float, optional
        Tilt of cross-section 2; default 0 [rad].
    turns1 : float, optional
        Turns of coil 1; default 1 [-].
    turns2 : float, optional
        Turns of coil 2; default 1 [-].
    mu_r : float, optional
        Relative permeability of the medium, e.g. 1.04 for SUS304; default 1 [-].
    n_div : int, optional
        Subdivision refinement; default 5 [-].
        The cell size is half the pair's characteristic scale over ``n_div``.

    Returns
    -------
    float
        Mutual inductance [H].

    Assumptions
    -----------
    Axisymmetric coils with uniform current density over the cross-section;
    the toroidal curvature correction $r \to r\,(1 + (\cos\theta\,\Delta r/r)^2/24)$
    is applied to each cell.

    Numerical notes
    ---------------
    Midpoint rule over both cross-sections (reference: legacy VFIT
    ``getMutualInductanceCoil.m``).  The cell size uses the $z$-*separation*
    $\sqrt{(r_1 + r_2)^2 + (z_1 - z_2)^2}$; the legacy code used $z_1 + z_2$,
    which is not invariant under a rigid vertical translation.  Legacy
    material codes are intentionally not reproduced: pass ``mu_r`` explicitly.

    References
    ----------
    .. [1] W. R. Smythe, *Static and Dynamic Electricity*, 3rd ed., McGraw-Hill
           (1968), Sec. 8.06 (mutual inductance of coaxial circles).
    .. [2] Legacy VFIT ``getMutualInductanceCoil.m``.
    """
    # Subdivision cell size from the pair's characteristic scale. NOTE:
    # legacy getMutualInductanceCoil.m used hypot(r1+r2, z1+z2), which is
    # not invariant under a rigid z-translation of the coil pair; the
    # z-separation form below is.
    dl = 0.5 * np.hypot(r1 + r2, z1 - z2) / n_div

    if dr1 != 0.0 and dz1 != 0.0:
        p1r, p1z = _rect_coil_midpoints(r1, z1, dr1, dz1, tilt1, dl)
    else:
        p1r, p1z = np.array([r1]), np.array([z1])

    if dr2 != 0.0 and dz2 != 0.0:
        p2r, p2z = _rect_coil_midpoints(r2, z2, dr2, dz2, tilt2, dl)
    else:
        p2r, p2z = np.array([r2]), np.array([z2])

    G = greens_function_exact(p1r[:, None], p1z[:, None], p2r[None, :], p2z[None, :], "psi")
    return float(turns1 * turns2 * mu_r * MU0 * G.mean())


def self_inductance(
    r: float,
    dr: float,
    dz: float,
    *,
    tilt: float = 0.0,
    turns: float = 1.0,
    mu_r: float = 1.0,
    n_div: int = 5,
) -> float:
    r"""Self-inductance of a rectangular-cross-section ring coil.

    $$L = \frac{N^2}{n^2}\left[\sum_{i \ne j}\mu_r\mu_0\,G(r_i, z_i; r_j, z_j)
      + \sum_i \mu_r\mu_0\,r_i\left(\ln\frac{8r_i}{s} - \frac{7}{4}\right)\right],
      \qquad s = \sqrt{\Delta r\,\Delta z/\pi}$$

    midpoint subdivision into $n$ cells with the analytic self-term of a
    circular filament of equivalent radius $s$ on the diagonal.

    Parameters
    ----------
    r : float
        Centre radius, positive [m].
    dr : float
        Radial width, positive [m].
    dz : float
        Height, positive [m].
    tilt : float, optional
        Tilt of the cross-section; default 0 [rad].
    turns : float, optional
        Number of turns; default 1 [-].
    mu_r : float, optional
        Relative permeability; default 1 [-].
    n_div : int, optional
        Cells per centre radius; default 5 [-].

    Returns
    -------
    float
        Self-inductance [H].

    Raises
    ------
    ValueError
        For non-positive ``r``, ``dr`` or ``dz``.

    Assumptions
    -----------
    Uniform current density; the self-cell term is the low-frequency
    inductance of a thin ring of wire radius $s$ (internal inductance
    included, hence $7/4$ rather than 2).

    Numerical notes
    ---------------
    Reference: legacy VFIT ``getSelfInductanceCoil.m``; ``nz`` is forced even.
    Convergence in ``n_div`` is first order because of the self-cell
    approximation.

    References
    ----------
    .. [1] W. R. Smythe, *Static and Dynamic Electricity*, 3rd ed., McGraw-Hill
           (1968), Sec. 8.10 (self-inductance of a circular ring).
    .. [2] Legacy VFIT ``getSelfInductanceCoil.m``.
    """
    if r <= 0.0 or dr <= 0.0 or dz <= 0.0:
        raise ValueError("self_inductance requires r, dr, dz > 0")

    cell = r / n_div
    nr = max(int(np.floor(dr / cell + 0.5)), 1)
    nz = max(2 * int(np.floor(0.5 * dz / cell + 0.5)), 2)
    ddr = dr / nr
    ddz = dz / nz
    sr = np.sqrt(ddr * ddz / np.pi)

    ir = np.arange(1, nr + 1, dtype=float)
    iz = np.arange(1, nz + 1, dtype=float)
    IR, IZ = np.meshgrid(ir, iz, indexing="ij")
    ct, st = np.cos(tilt), np.sin(tilt)
    rp = r - 0.5 * (dr * ct - dz * st) - (IZ - 0.5) * ddz * st + (IR - 0.5) * ddr * ct
    zp = -0.5 * (dr * st + dz * ct) + (IZ - 0.5) * ddz * ct + (IR - 0.5) * ddr * st
    rp, zp = rp.ravel(), zp.ravel()

    flux = mu_r * MU0 * greens_function_exact(
        rp[:, None], zp[:, None], rp[None, :], zp[None, :], "psi"
    )
    np.fill_diagonal(flux, mu_r * MU0 * rp * (np.log(8.0 * rp / sr) - 1.75))
    n_cells = rp.size
    return float(turns**2 * flux.sum() / n_cells**2)


def green_r(r_obs: np.ndarray, z_obs: np.ndarray, r_src: float, z_src: float) -> np.ndarray:
    r"""Poloidal flux per unit ring current at observer points (legacy elliptic path).

    $$\psi = 2\mu_0\,\frac{\sqrt{r r_0}}{k}\left[\left(1 - \frac{k^2}{2}\right)K - E\right],
      \qquad k^2 = \frac{4rr_0}{(r + r_0)^2 + (z - z_0)^2}$$

    Parameters
    ----------
    r_obs : np.ndarray
        Radius of the observation points [m].
    z_obs : np.ndarray
        Height of the observation points [m].
    r_src : float
        Radius of the current ring [m].
    z_src : float
        Height of the current ring [m].

    Returns
    -------
    np.ndarray
        Flux per ampere at each observation point [Wb/A].

    Convention
    ----------
    Full weber ($\psi = \mu_0 G$, identical to :func:`green_psi_exact` up to
    the elliptic-integral approximation); divide by $2\pi$ before combining
    with the per-radian equilibrium helpers.  Tracked in #354.

    Limitations
    -----------
    Uses the Hastings polynomials of :func:`elliptic_integral`; returns 0 on
    the axis ($r_{obs} = 0$ or $r_{src} = 0$) and clips $k^2$ into $[0, 1]$.

    References
    ----------
    .. [1] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999),
           Sec. 5.5, Eq. (5.37).
    """
    mu0 = 4.0 * np.pi * 1.0e-7 # Use np.pi
    z_diff = z_obs - z_src # array
    
    denom_k_calc = (r_obs + r_src) ** 2 + z_diff ** 2 # array
    # Avoid division by zero if denom_k_calc can be zero (e.g. r_obs = -r_src and z_diff = 0, though r usually positive)
    # k2 = 4.0 * r_obs * r_src / denom_k_calc # array
    
    # Ensure r_obs and r_src are non-negative as typically expected for radii.
    # k calculation can be sensitive.
    # Original code: k = np.sqrt(k2)
    # If k2 can be negative due to r_obs or r_src being negative (not typical for physical radii),
    # np.sqrt will produce NaNs. Assume r_obs, r_src >= 0.
    
    # Numerator for k2
    num_k2 = 4.0 * r_obs * r_src # array
    # k2 = num_k2 / denom_k_calc (array)
    # Handle cases where denom_k_calc might be zero or very small.
    # If r_obs = 0 and r_src = 0 and z_diff = 0, then denom_k_calc is 0.
    # If r_obs and r_src are always positive, denom_k_calc should be positive.
    k2 = np.divide(num_k2, denom_k_calc, out=np.zeros_like(num_k2), where=denom_k_calc!=0)


    # k = np.sqrt(k2) # array. This might be problematic if k2 is negative due to float precision for k2 > 1.
    # k^2 = 4 R R_s / ((R+R_s)^2 + (Z-Z_s)^2). For R, R_s > 0, k^2 should be <= 1.
    # However, floating point issues might make k2 slightly > 1.
    # We can clip k2 to be at most 1.
    k2_clipped = np.clip(k2, 0, 1.0) # Clip k2 to be in [0, 1]
    k = np.sqrt(k2_clipped) # array

    ek, ee = elliptic_integral(r_obs, z_obs, r_src, z_src) # ek, ee are arrays
    
    sqrt_rr_src = np.sqrt(r_obs * r_src) # array
    
    # Original formula: res = sqrt_rr1 * 2.0 * mu0 / k * ((1.0 - k2 / 2.0) * ek - ee)
    # Division by k can be problematic if k is zero.
    # k is zero if r_obs = 0 or r_src = 0.
    # If r_obs = 0: sqrt_rr_src is 0. Then result is 0 * (inf or nan) if k is 0.
    # If r_src = 0: sqrt_rr_src is 0.
    # Psi should be 0 if r_obs=0 or r_src=0 (axis).
    
    term_in_parenthesis = (1.0 - k2 / 2.0) * ek - ee # array
    
    # Handle division by k = 0
    # Where k is zero, the flux should be zero (e.g., on axis if r_obs=0 or r_src=0).
    # The term sqrt_rr_src will also be zero in these cases.
    # So, 0 * (something / 0) -> nan. We need to ensure result is 0.
    
    res = np.zeros_like(r_obs) # Initialize result array with zeros
    
    # Calculate only where k is not zero and sqrt_rr_src is not zero
    # (which implies r_obs > 0 and r_src > 0 for k to be non-zero and sqrt_rr_src non-zero)
    # A simpler way: if k is zero, sqrt_rr_src is also zero, making the numerator zero.
    # So, if k is zero, the expression 0/0 might arise if not careful.
    # Let's compute the main term and then set to zero where appropriate.
    
    main_term = sqrt_rr_src * 2.0 * mu0 # array
    
    # Calculate factor = main_term / k
    factor = np.divide(main_term, k, out=np.zeros_like(main_term), where=k!=0) # array
    
    res = factor * term_in_parenthesis # array
    
    # Ensure psi is 0 if r_obs = 0 (on axis) or if r_src = 0
    res = np.where((r_obs == 0) | (r_src == 0), 0.0, res)

    return res
