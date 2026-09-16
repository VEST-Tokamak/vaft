"""
Plasma equilibrium, current, energy, and geometry calculations.

This module provides functions for calculating various plasma equilibrium parameters
including poloidal flux, toroidal flux, safety factor, current, energy, and geometry.

Notation
--------
ψ      : poloidal magnetic flux                     [Wb] or [Wb/rad], per COCOS
ψ_a    : ψ at magnetic axis                         (same convention as ψ)
ψ_b    : ψ at plasma boundary                       (same convention as ψ)
Φ(ψ)   : toroidal flux through surface C(ψ)         [Wb]
Φ_b    : Φ(ψ_b)                                     [Wb]
ρ_N    : normalised minor-radius (0 at axis, 1 at edge)
q      : safety factor                              [-]
j      : current density                            [A/m²]
I_p    : plasma current                             [A]
W      : stored energy                              [J]
V      : plasma volume                              [m³]
κ      : elongation                                 [-]
δ      : triangularity                              [-]
α      : virial closure coefficient, 2⟨R B_Z²⟩/⟨R B_p²⟩  [-]
"""

import warnings
from typing import Union, Tuple, Optional
import numpy as np

from ._exports import public_names
from .constants import (
    MU0, QE, ME, MI_P, EPS0, C_LIGHT,
    E_ALPHA, SIGMA_V_COEF,
    SPITZER_RESISTIVITY_COEF,
    C_B, K_B_COEF,
    _SCALING_COEFS
)
from scipy.integrate import cumulative_trapezoid

from .utils import (
    _guarded_ratio,
    calculate_peaking_factor,
    gradient,
    trapz_integral,
    normalize_profile,
    calculate_poloidal_flux,
    calculate_toroidal_flux,
    calculate_volume_weighted_average
)

# Backward compatibility: the virial closures moved to `.virial` (#711) and are
# re-bound here so every `from vaft.formula.equilibrium import virial_...` keeps
# working. Deliberately absent from this module's `__all__`, so the catalog and
# `vaft.formula`'s own resolution attribute them to the module that defines
# them; an explicit import does not consult `__all__`, which is why both hold.
# Spelled out rather than `import *`. A star import makes ruff abandon F821
# (undefined-name) for the *whole* module, which is how the synchrotron function
# went on referencing names #368 had deleted without any check noticing (#753).
# This module is 4000 lines of pure formulas; that is the last place to give up
# undefined-name detection for a compatibility shim.
#
# The list is `virial.__all__`, which is what the star import bound. It has to be
# kept in step by hand, and
# `test_the_virial_compatibility_shim_still_re_exports_without_a_star_import`
# iterates `virial.__all__` and fails if a name is missing here -- so a closure
# added there without being added here breaks a test rather than a caller.
from .virial import (  # noqa: F401
    VIRIAL_SINGULAR_EPS,
    approximated_diamagnetism_from_B_pa_B_tv_R0_delta_phi,
    virial_D0_boundary_from_bp_li_eK,
    virial_S1_approx,
    virial_S2_approx_from_D0_a_R0,
    virial_S3_approx_from_eK_d,
    virial_alpha_approx_from_kappa,
    virial_alpha_from_R_Bz_Bp_dl,
    virial_beta_p_from_S_alpha_mu,
    virial_beta_p_from_S_li,
    virial_beta_p_from_volume,
    virial_beta_p_lao_from_S_mu_rt,
    virial_beta_p_li_from_S_alpha_mu_rt,
    virial_beta_pd_from_S_mu_rt,
    virial_bongard_from_S_alpha_mu,
    virial_bp_li_lihat_from_S123,
    virial_closure_denominators,
    virial_full_123_from_S_alpha_rt,
    virial_identity_residuals,
    virial_kinetic_energy,
    virial_lao_from_S_alpha_mu_rt,
    virial_li_from_S_alpha_mu,
    virial_li_from_S_alpha_rt,
    virial_li_from_volume,
    virial_magnetic_energy,
    virial_muihat_from_Bt_R0_dphi,
    virial_normalized_residual,
    virial_pair_12_from_S_mu_rt,
    virial_pair_13_from_S_alpha_mu,
    virial_pair_23_from_S_alpha_mu_rt,
    virial_residual_rms,
    virial_stability_criterion,
    virial_theorem,
    virial_thermal_energy,
)


#: What ``from vaft.formula.equilibrium import *`` binds, and therefore what
#: reaches ``vaft.formula.__all__``. Equilibrium, current, energy and geometry kernels.
#: Declared so the package stops re-exporting this module's own imports --
#: ``np``, ``warnings``, ``Union``, ``curve_fit`` -- as though they were
#: formulas (#368).


# ------------------------------------------------------------------
# Poloidal Flux Calculations
# ------------------------------------------------------------------

def psi_from_RBtheta(R: np.ndarray,
                     B_theta: np.ndarray,
                     l: np.ndarray,
                     psi_axis: float = 0.0) -> float:
    r"""Poloidal flux $\psi$ from a line integral of $R B_\theta$ across flux surfaces.

    $$\psi(l) = \int_0^{l} R\,B_\theta\,dl' + \psi_a$$

    along a path $l$ that crosses the flux surfaces (the outboard midplane, say),
    where $B_\theta$ is the poloidal field component normal to the path.  This is
    the inverse of $B_p = |\nabla\psi|/R$, so the result is the flux per radian.

    Parameters
    ----------
    R : np.ndarray
        Major radius along the integration path [m].
    B_theta : np.ndarray
        Poloidal magnetic field normal to the path, same shape as ``R`` [T].
    l : np.ndarray
        Path coordinate, monotonic, same shape as ``R`` [m].
    psi_axis : float, optional
        Flux at the start of the path, added as an offset; default 0 [Wb/rad].

    Returns
    -------
    np.ndarray
        Poloidal flux at every point of the path [Wb/rad].

    Convention
    ----------
    Returns flux per radian, $\psi = \int R B_\theta\,dl$, the COCOS 1-8 storage
    of an EFIT g-file or of VFIT.  Multiply by $2\pi$ for the IMAS Data
    Dictionary's full-weber ``equilibrium.*.psi`` (COCOS 11-18).  The sign follows
    the sign of ``B_theta`` and the direction of ``l``; nothing is re-oriented.

    Assumptions
    -----------
    Axisymmetry, and that ``B_theta`` is the component perpendicular to the path
    so that $R B_\theta\,dl$ is exactly $d\psi$.

    Numerical notes
    ---------------
    Trapezoidal rule (``numpy.trapezoid``) on the supplied samples; the result is
    returned only at the sample points and is second-order accurate in the
    spacing of ``l``.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.2 (flux functions).
    .. [2] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Sec. 2 and Table I (per-radian versus full-weber flux).
    """
    return calculate_poloidal_flux(R, B_theta, l, psi_axis)


def psi_normalised(psi: Union[np.ndarray, float],
                   psi_axis: float,
                  psi_boundary: float) -> Union[np.ndarray, float]:
    r"""Normalised poloidal flux $\psi_N$.

    $$\psi_N = \frac{\psi - \psi_a}{\psi_b - \psi_a}$$

    Parameters
    ----------
    psi : float or np.ndarray
        Poloidal flux [Wb/rad or Wb].
    psi_axis : float
        Flux at the magnetic axis, same unit as ``psi`` [Wb/rad or Wb].
    psi_boundary : float
        Flux at the plasma boundary, same unit as ``psi`` [Wb/rad or Wb].

    Returns
    -------
    float or np.ndarray
        Normalised flux, 0 on axis and 1 at the boundary [-].

    Convention
    ----------
    Independent of the $2\pi$ storage convention and of the COCOS sign, because
    both cancel in the ratio, provided all three inputs share one convention.
    Equals the IMAS ``profiles_1d.psi_norm`` label.

    Numerical notes
    ---------------
    A degenerate equilibrium with equal axis and boundary flux warns and
    returns ``nan`` rather than ``inf``.

    See Also
    --------
    vaft.formula.utils.normalize_profile
    """
    return normalize_profile(psi, psi_axis, psi_boundary)


# Backwards compatibility alias
def normalize_psi(*args, **kw):  # noqa: N802
    r"""Deprecated: use :func:`psi_normalised`.

    Kept for backwards compatibility; emits a ``DeprecationWarning`` and forwards
    every argument unchanged.

    See Also
    --------
    psi_normalised
    """
    warnings.warn("`normalize_psi` is deprecated → use `psi_normalised`",
                 DeprecationWarning, stacklevel=2)
    return psi_normalised(*args, **kw)


# ------------------------------------------------------------------
# Toroidal Flux Calculations
# ------------------------------------------------------------------

def phi_from_Bphi(B_phi: np.ndarray,
                  dA: np.ndarray) -> float:
    r"""Toroidal flux $\Phi$ through a poloidal cross-section.

    $$\Phi = \int_{S} B_\varphi\,dA \approx \sum_i B_{\varphi,i}\,\Delta A_i$$

    Parameters
    ----------
    B_phi : np.ndarray
        Toroidal magnetic field on the area elements [T].
    dA : np.ndarray
        Poloidal-plane area element of each sample, same shape as ``B_phi`` [m^2].

    Returns
    -------
    float
        Toroidal flux through the surface [Wb].

    Convention
    ----------
    Full weber, never per radian: toroidal flux carries no $2\pi$ ambiguity.  The
    sign is that of $B_\varphi$, i.e. $\sigma_{B_\varphi}$ of the COCOS in use
    (COCOS 1-8 and 11-18 differ only in the poloidal flux).

    Assumptions
    -----------
    ``dA`` are true area elements of the cross-section bounded by the flux surface
    of interest; the routine does not construct them.

    Numerical notes
    ---------------
    A Riemann sum, not a quadrature rule: accuracy is first order in the cell size
    of the caller's grid.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.4 (toroidal flux and the safety factor).
    """
    return calculate_toroidal_flux(B_phi, dA)


def rhoN_from_phi(phi: Union[np.ndarray, float],
                  phi_boundary: float) -> Union[np.ndarray, float]:
    r"""Normalised toroidal-flux radius $\rho_N$.

    $$\rho_N = \sqrt{\frac{\Phi}{\Phi_b}}$$

    Parameters
    ----------
    phi : float or np.ndarray
        Toroidal flux enclosed by the surface [Wb].
    phi_boundary : float
        Toroidal flux enclosed by the plasma boundary [Wb].

    Returns
    -------
    float or np.ndarray
        Toroidal-flux label, 0 on axis and 1 at the boundary [-].

    Convention
    ----------
    This is the IMAS ``rho_tor_norm`` label (the square root of normalised
    *toroidal* flux), not the poloidal-flux label $\sqrt{\psi_N}$ nor a geometric
    minor radius.  ``phi`` and ``phi_boundary`` must carry the same sign; a
    COCOS-sign mismatch between them produces ``nan`` from the square root.

    Physical interpretation
    -----------------------
    $\rho_N$ is the minor radius of the circular cylinder that would enclose the
    same toroidal flux at the same $B_0$, scaled to the boundary value.

    References
    ----------
    .. [1] F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239,
           Sec. II.B (flux-surface coordinates).
    .. [2] IMAS Data Dictionary, ``equilibrium.time_slice[:].profiles_1d.rho_tor_norm``.
    """
    return np.sqrt(phi / phi_boundary)


def toroidal_flux_from_q_psi(q: np.ndarray,
                             psi_wb: np.ndarray) -> np.ndarray:
    r"""Cumulative toroidal flux $\Phi(\psi)$ from the $q$ profile on a full-weber grid.

    $$\Phi(\psi) = \int_{\psi_a}^{\psi} q\,d\psi', \qquad \psi\ \text{in Wb}$$

    Parameters
    ----------
    q : np.ndarray
        Safety factor on the flux grid [-].
    psi_wb : np.ndarray
        Poloidal flux of the same surfaces, monotonic, full weber [Wb].

    Returns
    -------
    np.ndarray
        Toroidal flux enclosed by each surface, starting at zero [Wb].

    Convention
    ----------
    ``psi_wb`` is the IMAS Data Dictionary flux (COCOS 11-18).  No $2\pi$
    appears because $d\Phi/d\psi_{rad} = 2\pi q$ and $\psi_{wb} = 2\pi
    \psi_{rad}$ cancel; pass ``2*np.pi*psi_rad`` for an EFIT or VFIT
    per-radian profile (:func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor`
    settles which family an ODS holds).  The orientation sign of $q$ and
    $\psi$ is not applied: a COCOS with $\sigma_{\rho\theta\varphi} = -1$
    gives a negative $\Phi$.

    Numerical notes
    ---------------
    Cumulative trapezoidal rule (``scipy.integrate.cumulative_trapezoid`` via
    :func:`vaft.compat.cumtrapz_compat`); second order in the grid spacing,
    and the result is the running integral, so its first element is zero.

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (17) and Table I.
    .. [2] IMAS Data Dictionary, ``equilibrium.time_slice[:].profiles_1d.phi``.
    """
    from vaft.compat import cumtrapz_compat

    q = np.asarray(q, dtype=float).reshape(-1)
    psi_wb = np.asarray(psi_wb, dtype=float).reshape(-1)
    return np.asarray(cumtrapz_compat(q, x=psi_wb), dtype=float)


def rho_tor_from_phi(phi: Union[np.ndarray, float],
                     B0: float) -> Union[np.ndarray, float]:
    r"""Dimensional toroidal-flux radius $\rho_{tor} = \sqrt{|\Phi|/(\pi|B_0|)}$.

    $$\rho_{tor} = \sqrt{\frac{|\Phi|}{\pi\,|B_0|}}$$

    Parameters
    ----------
    phi : float or np.ndarray
        Toroidal flux enclosed by the surface [Wb].
    B0 : float
        Vacuum toroidal field at the reference major radius [T].

    Returns
    -------
    float or np.ndarray
        Toroidal-flux radius [m].

    Convention
    ----------
    The IMAS ``rho_tor`` coordinate: the minor radius of the circle that
    would carry the same toroidal flux in a uniform field $B_0$, with $B_0 =$
    ``vacuum_toroidal_field.b0`` at ``r0``.  Absolute values are taken, so the
    result is independent of the COCOS signs of $\Phi$ and $B_0$; divide by
    the boundary value for :func:`rhoN_from_phi`'s ``rho_tor_norm``.

    Physical interpretation
    -----------------------
    A length-like flux label that reduces to the geometric minor radius for a
    circular, large-aspect-ratio plasma with uniform $B_0$.

    References
    ----------
    .. [1] IMAS Data Dictionary, ``equilibrium.time_slice[:].profiles_1d.rho_tor``.
    .. [2] F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239, Sec. II.B.
    """
    return np.sqrt(np.abs(phi) / (np.pi * abs(float(B0))))


# ------------------------------------------------------------------
# Safety Factor Calculations
# ------------------------------------------------------------------

def q_from_flux_surface_averages(
    gm1: Union[np.ndarray, float],
    dvolume_dpsi: Union[np.ndarray, float],
    f: Union[np.ndarray, float],
    *,
    cocos: int | None = None,
    psi_per_radian: bool | None = None,
    sigma_ip: int = 1,
    sigma_b0: int = 1,
) -> Union[np.ndarray, float]:
    r"""Safety factor $q$ from flux-surface geometric averages and poloidal current.

    $$q(\psi) = \sigma \cdot \frac{F(\psi)}{2\pi} \oint \frac{dl_p}{R^2 B_p}
              = \sigma \cdot (2\pi)^{e_{B_p} - 2} \cdot F(\psi) \cdot \left\langle \frac{1}{R^2} \right\rangle \cdot \frac{dV}{d|\psi|}$$

    Parameters
    ----------
    gm1 : float or np.ndarray
        Geometric flux-surface average $\langle 1/R^2 \rangle$ [m^-2].
    dvolume_dpsi : float or np.ndarray
        Differential volume element $dV/d|\psi|$ [m^3/(Wb/rad) if $e_{B_p}=0$ or m^3/Wb if $e_{B_p}=1$].
    f : float or np.ndarray
        Poloidal current function $F = R B_\varphi$ on the flux surface [T m].
    cocos : int or None, optional
        COCOS coordinate convention index (1-8, 11-18) [-].
    psi_per_radian : bool or None, optional
        Storage family of the flux when ``cocos`` is None [bool].
        ``False`` assumes full-weber flux ($e_{B_p} = 1$); ``True`` and ``None``
        keep the per-radian assumption ($e_{B_p} = 0$).
    sigma_ip : int, optional
        Sign of plasma current (+1 or -1) in the equilibrium coordinate system [-].
    sigma_b0 : int, optional
        Sign of toroidal field (+1 or -1) in the equilibrium coordinate system [-].

    Returns
    -------
    float or np.ndarray
        Safety factor profile $q$ on the corresponding flux surfaces [-].

    Convention
    ----------
    In Sauter and Medvedev (2013), $B_p = |\nabla\psi| / (R (2\pi)^{e_{B_p}})$.
    The safety factor contour integral is:
    $$q = \frac{F}{2\pi} \oint \frac{dl_p}{R^2 B_p} = (2\pi)^{e_{B_p}-1} \frac{F}{2\pi} \oint \frac{dl_p}{R |\nabla\psi|}$$
    Since $dV/d\psi = 2\pi \oint \frac{R dl_p}{|\nabla\psi|}$ and
    $\langle 1/R^2 \rangle = \frac{\oint dl_p / (R |\nabla\psi|)}{\oint R dl_p / |\nabla\psi|}$,
    this gives:
    $$q = \sigma \cdot (2\pi)^{e_{B_p}-2} \cdot |F| \cdot \langle 1/R^2 \rangle \cdot \frac{dV}{d|\psi|}$$
    For $e_{B_p} = 0$ (COCOS 1-8, Wb/rad), $(2\pi)^{0-2} = 1/(4\pi^2)$.
    For $e_{B_p} = 1$ (COCOS 11-18, full Wb), $(2\pi)^{1-2} = 1/(2\pi)$.
    The sign $\sigma$ is determined by Sauter Eq. 23: $\sigma_q = \sigma_{Ip}\sigma_{B0}\sigma_{\rho\theta\varphi}$.

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293.
    """
    gm1_arr = np.asarray(gm1, dtype=float)
    dv_arr = np.asarray(dvolume_dpsi, dtype=float)
    f_arr = np.asarray(f, dtype=float)

    if cocos is not None:
        from vaft.data.cocos import cocos_spec

        spec = cocos_spec(cocos)
        exp_bp = spec.exp_bp
        target_sign = spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    else:
        exp_bp = 0 if (psi_per_radian is None or psi_per_radian) else 1
        target_sign = 1 if (sigma_ip * sigma_b0) >= 0 else -1

    factor = (2.0 * np.pi) ** (exp_bp - 2)
    q_mag = factor * np.abs(f_arr) * np.abs(gm1_arr) * np.abs(dv_arr)
    q = target_sign * q_mag
    if np.ndim(gm1) == 0 and np.ndim(dvolume_dpsi) == 0 and np.ndim(f) == 0:
        return float(q)
    return q




def q_from_phi(psi: np.ndarray,
               phi: np.ndarray) -> np.ndarray:
    r"""Safety factor $q$ as the flux derivative $d\Phi/d\psi$.

    $$q = \frac{d\Phi}{d\psi}$$

    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux profile, monotonic [Wb/rad].
    phi : np.ndarray
        Toroidal flux enclosed by the same surfaces [Wb].

    Returns
    -------
    np.ndarray
        Safety factor on the input surfaces [-].

    Convention
    ----------
    Sauter and Medvedev define $q = \sigma_{\rho\theta\varphi}\sigma_{B_p}
    (2\pi)^{-e_{B_p}}\,d\Phi/d\psi$.  This routine applies neither sign nor
    $2\pi$: it is exact for ``psi`` in Wb/rad (COCOS 1-8, $e_{B_p}=0$) up to the
    orientation sign, and returns $2\pi q$ when ``psi`` is the IMAS full-weber
    flux.  Convert with :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor`
    first, or take ``profiles_1d.q`` from the equilibrium directly.  Tracked in
    #354.

    Numerical notes
    ---------------
    ``numpy.gradient``: second-order central differences in the interior,
    first-order one-sided at the two ends, noise-amplifying; needs at least two
    samples and a strictly monotonic ``psi``.

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (17) and Table I.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 3.4.
    """
    return gradient(psi, phi)


def q_from_rhoN(psiN: np.ndarray,
                rhoN: np.ndarray,
                C: float = 1.0) -> np.ndarray:
    r"""Safety factor from the toroidal-flux label $\rho_N(\psi_N)$.

    $$q = C\,\rho_N\,\frac{d\rho_N}{d\psi_N}, \qquad
      C = \frac{2\,\Phi_b}{\psi_b - \psi_a}$$

    follows from $\Phi = \Phi_b\rho_N^2$ and $q = d\Phi/d\psi$.

    Parameters
    ----------
    psiN : np.ndarray
        Normalised poloidal flux, monotonic [-].
    rhoN : np.ndarray
        Normalised toroidal-flux radius on the same surfaces [-].
    C : float, optional
        Prefactor $2\Phi_b/(\psi_b-\psi_a)$ with $\psi$ in Wb/rad; default 1 [-].

    Returns
    -------
    np.ndarray
        Safety factor, or with the default ``C`` only its shape [-].

    Convention
    ----------
    With ``C=1`` the result is $q$ up to the constant $2\Phi_b/(\psi_b-\psi_a)$
    and is only proportional to the true profile.  Supply ``C`` from the
    equilibrium (with $\psi$ per radian, or $2\pi$ smaller with full-weber
    $\psi$) for absolute values; the orientation sign is not applied.

    Numerical notes
    ---------------
    ``numpy.gradient`` of ``rhoN`` against ``psiN`` (second-order interior,
    first-order ends); the on-axis value is dominated by the one-sided difference
    and by $\rho_N\to0$.

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (17).
    .. [2] F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239, Sec. II.B.
    """
    drhoN_dpsiN = gradient(psiN, rhoN)
    return C * rhoN * drhoN_dpsiN


def rhoN_from_qpsiN(psiN: np.ndarray,
                    qpsiN: np.ndarray) -> np.ndarray:
    r"""Normalised toroidal-flux radius from the $q$ profile.

    $$\rho_N = \sqrt{\frac{\int_0^{\psi_N} q\,d\psi_N'}{\int_0^{1} q\,d\psi_N'}}$$

    which is $\sqrt{\Phi/\Phi_b}$ since $d\Phi = q\,d\psi$ and the constants
    cancel in the ratio.

    Parameters
    ----------
    psiN : np.ndarray
        Normalised poloidal flux, increasing from 0 [-].
    qpsiN : np.ndarray
        Safety factor on the same surfaces [-].

    Returns
    -------
    np.ndarray
        Normalised toroidal-flux radius on the input surfaces [-].

    Convention
    ----------
    Independent of the $\psi$ unit and of the COCOS sign as long as ``qpsiN``
    does not change sign; a signed $q$ (COCOS with $\sigma_{\rho\theta\varphi}=-1$)
    must be passed as $|q|$ or the square root returns ``nan``.

    Assumptions
    -----------
    ``psiN`` starts at the magnetic axis; the integral is taken from the first
    sample, so a profile that starts inside the plasma is mis-normalised.

    Numerical notes
    ---------------
    One vectorised cumulative trapezoid
    (``scipy.integrate.cumulative_trapezoid``), not a rebuild per sample. A
    vanishing or non-finite total integral warns and yields ``nan`` rather than
    ``inf``. A uniformly signed $q$ is fine -- numerator and denominator flip
    together -- but one that changes sign makes the cumulative ratio negative
    on some samples, and those warn rather than becoming a silent ``nan``.

    References
    ----------
    .. [1] F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239, Sec. II.B.
    """
    psiN = np.asarray(psiN, dtype=float)
    qpsiN = np.asarray(qpsiN, dtype=float)
    # cumulative_trapezoid with initial=0 is the same quantity the old
    # per-sample rebuild produced, in one pass instead of O(N^2).
    num = cumulative_trapezoid(qpsiN, psiN, initial=0.0)
    den = trapz_integral(psiN, qpsiN)
    ratio = _guarded_ratio(
        num, den, what="rhoN_from_qpsiN", because="the total integral of q dpsi_N"
    )
    negative = np.asarray(ratio) < 0.0
    if np.any(negative):
        warnings.warn(
            "rhoN_from_qpsiN: the cumulative flux ratio is negative on "
            f"{int(np.count_nonzero(negative))} sample(s), which means q "
            "changes sign over the profile; the square root is undefined "
            "there. Returning nan on those samples.",
            RuntimeWarning,
            stacklevel=2,
        )
        ratio = np.where(negative, np.nan, ratio)
    return np.sqrt(ratio)


# ------------------------------------------------------------------
# Magnetic Shear
# ------------------------------------------------------------------

def shear_from_r_q(r: np.ndarray,
                   q: np.ndarray) -> np.ndarray:
    r"""Magnetic shear $s$ of the safety-factor profile.

    $$s = \frac{r}{q}\,\frac{dq}{dr}$$

    Parameters
    ----------
    r : np.ndarray
        Flux-surface radius label, monotonic; minor radius or $\rho_N$ [m or -].
    q : np.ndarray
        Safety factor on the same surfaces [-].

    Returns
    -------
    np.ndarray
        Local magnetic shear [-].

    Convention
    ----------
    The logarithmic derivative $d\ln q/d\ln r$, the definition used in the
    $s$-$\alpha$ ballooning diagram; any monotonic radius label gives the same
    number up to the choice of $r$ (minor radius versus $\rho_N$ differ in the
    Shafranov-shifted region).  Sign is that of $dq/dr$, which is independent of
    COCOS.

    Physical interpretation
    -----------------------
    Rate at which field-line pitch changes across surfaces; positive shear
    stabilises ballooning modes at low $\alpha$ and localises resonant
    perturbations.

    Numerical notes
    ---------------
    ``numpy.gradient`` (second-order interior, first-order ends), then division by
    ``q`` and multiplication by ``r``: the axis value is exactly 0 when ``r``
    starts at 0 and undefined where $q$ crosses zero.

    References
    ----------
    .. [1] J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40 (1978)
           396, definition of $s$.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 6.13.
    """
    dqdr = gradient(r, q)
    return (r / q) * dqdr


# Alias for backwards compatibility
magnetic_shear = shear_from_r_q  # noqa: E305


# ------------------------------------------------------------------
# Current Density
# ------------------------------------------------------------------

def current_density_from_B(B: Union[float, np.ndarray],
                          R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Toroidal current density from the radial derivative of a poloidal field.

    $$j_\varphi \approx \frac{1}{\mu_0}\,\frac{dB}{dR}$$

    the slab form of Ampere's law $\mu_0 j_\varphi = \partial B_Z/\partial R -
    \partial B_R/\partial Z$ with the second term dropped.

    Parameters
    ----------
    B : float or np.ndarray
        Poloidal field component (normally $B_Z$) along a 1-D radial cut [T].
    R : float or np.ndarray
        Major radius of the samples, monotonic [m].

    Returns
    -------
    float or np.ndarray
        Current density along the cut [A/m^2].

    Convention
    ----------
    Sign follows the COCOS $\sigma_{B_p}$ of the supplied component: with the
    midplane $B_Z$ of a standard equilibrium the result has the sign of the
    plasma current.  The $\partial B_R/\partial Z$ contribution is neglected, so
    the answer is exact only on the midplane of an up-down symmetric equilibrium.

    Numerical notes
    ---------------
    ``numpy.gradient`` along the single supplied axis (second-order interior,
    first-order ends); pass a 1-D slice, not a 2-D map.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1 (Ampere's law in the tokamak).
    """
    return gradient(R, B) / MU0


def current_density_from_psi(psi: Union[float, np.ndarray],
                           R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Deprecated: this is $-B_Z/\mu_0$, not a current density.

    $$j = -\frac{1}{\mu_0 R}\,\frac{d\psi}{dR} = -\frac{B_Z}{\mu_0}
      \ \ [\mathrm{A/m}]$$

    The value is unchanged; only the name and the guidance are.  Emits a
    ``DeprecationWarning``.  The old spelling is removed in 0.8.0.

    Parameters
    ----------
    psi : float or np.ndarray
        Poloidal flux along a 1-D radial cut [Wb/rad].
    R : float or np.ndarray
        Major radius of the samples, monotonic [m].

    Returns
    -------
    float or np.ndarray
        The quantity $-(\mu_0 R)^{-1}\,d\psi/dR$ [A/m].

    Convention
    ----------
    **Mind the sign when migrating.**  This module's default convention is
    $k = -1$ (weber-per-radian, COCOS 2/3/6/7), where
    :func:`vertical_magnetic_field_from_psi` returns
    $B_Z = -k/R\;d\psi/dR = +(1/R)\,d\psi/dR$.  This function is therefore
    $-B_Z/\mu_0$, not $+B_Z/\mu_0$ as its ``Limitations`` section claimed until
    #355; the replacement expression needs the minus sign or the result flips.

    The $k$ was written inline rather than taken from
    :func:`poloidal_field_factor`, so this function cannot be told its COCOS.
    That is the second reason not to keep it: the replacement can.

    Limitations
    -----------
    A current per unit length [A/m], not the toroidal current density
    $j_\varphi = -\Delta^*\psi/(\mu_0 R)$ [A/m^2].  A single first derivative
    along one radial cut cannot produce $j_\varphi$, which needs second
    derivatives in both $R$ and $Z$.

    See Also
    --------
    vertical_magnetic_field_from_psi : for the quantity this actually returns,
        as ``-vertical_magnetic_field_from_psi(psi, R, Z, cocos=...) / MU0``,
        which unlike this function can be told its COCOS.
    vaft.process.equilibrium.grad_shafranov_operator : $\Delta^*\psi$ on a 2-D
        map, for a real $j_\varphi = -\Delta^*\psi/(\mu_0 R)$.
    vaft.omas.update.update_equilibrium_profiles_2d_j_tor : writes
        ``profiles_2d.j_tor``, which is where an ODS already carries it.

    Numerical notes
    ---------------
    ``numpy.gradient`` along the single supplied axis; pass a 1-D slice.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.3 (Grad-Shafranov equation, $\mu_0 R j_\varphi = -\Delta^*\psi$).
    """
    warnings.warn(
        "`current_density_from_psi` is deprecated -- it returns -B_Z/mu0 [A/m], "
        "not a current density. For that quantity use "
        "`-vertical_magnetic_field_from_psi(psi, R, Z, cocos=...) / MU0` (mind "
        "the sign); for a real toroidal current density use "
        "`vaft.process.equilibrium.grad_shafranov_operator` or the ODS's own "
        "`profiles_2d.j_tor`. Removed in 0.8.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return -gradient(R, psi) / (MU0 * R)


# ------------------------------------------------------------------
# Current Drive
# ------------------------------------------------------------------

def current_drive_efficiency(n_e: float,
                           T_e_keV: float,
                           Z_eff: float = 1.0) -> float:
    r"""Heuristic lower-hybrid current-drive efficiency $\eta_{CD}$.

    $$\eta_{CD} = 0.3\,\sqrt{\frac{n_e\,T_e}{Z_{\mathrm{eff}}}}$$

    Parameters
    ----------
    n_e : float
        Electron density in the unit the coefficient was fitted for [any].
        The original source does not record which.
    T_e_keV : float
        Electron temperature [keV].
    Z_eff : float, optional
        Effective charge; default 1 [-].

    Returns
    -------
    float
        Efficiency figure in the coefficient's own normalisation [-].

    Physical interpretation
    -----------------------
    Lower-hybrid current drive becomes more efficient at higher temperature and
    lower $Z_{\mathrm{eff}}$ because the wave-driven fast electrons slow down on
    a hotter, cleaner background; the density factor here is unusual (the
    standard figure of merit $\eta = n_e I_{CD} R / P$ *divides* by density).

    Validity
    --------
    Empirical fit.  Labelled "ITER scaling for lower hybrid current drive" in the
    original VAFT source; no publication, dataset or unit system for the
    coefficient 0.3 was recorded, so the number is a placeholder.  The theory of
    the efficiency and its $T_e/Z_{\mathrm{eff}}$ dependence is Fisch [1]_.

    Limitations
    -----------
    Unsourced coefficient and unstated density unit; treat the result as
    qualitative.  Tracked in #361.

    References
    ----------
    .. [1] N. J. Fisch, Rev. Mod. Phys. 59 (1987) 175, Sec. VI (lower-hybrid
           current-drive efficiency).
    """
    return 0.3 * (n_e * T_e_keV / Z_eff)**0.5


def bootstrap_current_fraction(n_e: float,
                             T_e_keV: float,
                             R0: float,
                             a: float,
                             q_95: float) -> float:
    r"""Heuristic bootstrap-current fraction $f_{BS}$.

    $$f_{BS} = 0.3\sqrt{\beta_p}, \qquad
      \beta_p = \frac{0.4\,n_e\,T_e\,a}{R_0\,q_{95}^2}$$

    Parameters
    ----------
    n_e : float
        Electron density in the unit the coefficient was fitted for [any].
        The original source does not record which.
    T_e_keV : float
        Electron temperature [keV].
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    q_95 : float
        Safety factor at the 95% flux surface [-].

    Returns
    -------
    float
        Bootstrap fraction of the plasma current [-].

    Physical interpretation
    -----------------------
    Neoclassical bootstrap current scales as $\epsilon^{1/2}\beta_p$; the inner
    expression is a crude $\beta_p$ estimate from an $n T$ pressure and the
    cylindrical $q$-$I_p$ relation, and the outer square root is a fit.

    Validity
    --------
    Empirical fit.  Labelled "ITER scaling for bootstrap current" in the original
    VAFT source without a publication or unit system for the coefficients 0.3
    and 0.4.  The physics it approximates is the $\sqrt{\epsilon}\,\beta_p$
    scaling of Peeters [1]_ and Wesson [2]_.

    Limitations
    -----------
    Unsourced coefficients and unstated density unit; the $\beta_p$ estimate
    ignores profile shape and the ion pressure.  Tracked in #361.

    References
    ----------
    .. [1] A. G. Peeters, Plasma Phys. Control. Fusion 42 (2000) B231, Sec. 2.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.9 (bootstrap current).
    """
    beta_p = 0.4 * n_e * T_e_keV * a / (R0 * q_95**2)
    return 0.3 * np.sqrt(beta_p)

# ------------------------------------------------------------------
# Magnetic Field $B$
# ------------------------------------------------------------------


def poloidal_field_magnitude(b_r: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    r"""Poloidal field strength $|B_p| = \sqrt{B_R^2 + B_Z^2}$.

    Parameters
    ----------
    b_r : array_like
        Radial field component [T].
    b_z : array_like
        Vertical field component [T].

    Returns
    -------
    np.ndarray
        Poloidal field magnitude, elementwise [T].

    Convention
    ----------
    A magnitude, so it carries no COCOS sign: the orientation conventions cancel
    in the quadrature.  Both inputs must already be in tesla and in the same
    convention as each other, which they are when they come from one call to
    :func:`vaft.formula.green.green_br_bz_exact` or from one response matrix.

    Validity
    --------
    Machine-independent.
    """
    return np.hypot(np.asarray(b_r, dtype=float), np.asarray(b_z, dtype=float))


def decay_index_from_bz(
    r: np.ndarray, b_z: np.ndarray, *, axis: int = -1
) -> np.ndarray:
    r"""Field decay index $n = -\dfrac{R}{B_Z}\dfrac{\partial B_Z}{\partial R}$.

    How fast the vertical field falls off with major radius, which decides
    whether the radial force balance holding a current ring is *stable*.

    Parameters
    ----------
    r : array_like
        Major radius, monotonic, along ``axis`` [m].
    b_z : array_like
        Vertical field sampled on ``r``; may carry extra leading axes [T].
    axis : int, optional
        Axis of ``b_z`` along which ``r`` varies [-].

    Returns
    -------
    np.ndarray
        Decay index, ``nan`` where $B_Z$ vanishes [-].

    Convention
    ----------
    $0 < n < 1.5$ is the passively stable window: below zero the vertical field
    does not restore a radial displacement, above 1.5 the ring is unstable to
    vertical motion.

    Limitations
    -----------
    Returns ``nan`` where $B_Z$ crosses zero rather than a large number.  The
    index is genuinely undefined on that surface, and a spike there is an
    artefact of the division, not a physical instability -- which is what a
    reader would otherwise take from it.

    Validity
    --------
    Machine-independent.  The stable window quoted above is the rigid-ring
    result: it assumes a thin current ring and no conducting wall, so a real
    vessel widens it.

    References
    ----------
    .. [1] V. S. Mukhovatov and V. D. Shafranov, Nucl. Fusion 11 (1971) 605,
           Sec. 2 (equilibrium of a current ring in a vertical field).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.7 (vertical field and positional stability).
    """
    r_arr = np.asarray(r, dtype=float)
    b_arr = np.asarray(b_z, dtype=float)
    gradient_bz = np.gradient(b_arr, r_arr, axis=axis, edge_order=2)
    shape = [1] * b_arr.ndim
    shape[axis] = r_arr.size
    with np.errstate(divide="ignore", invalid="ignore"):
        index = -r_arr.reshape(shape) * gradient_bz / b_arr
    return np.where(np.isfinite(index), index, np.nan)


def toroidal_electric_field(r: np.ndarray, dpsi_dt: np.ndarray) -> np.ndarray:
    r"""Toroidal electric field $E_\varphi = -\dfrac{1}{2\pi R}\dfrac{\partial\psi}{\partial t}$.

    The inductive drive a startup has to work with: the loop voltage
    $-\partial\psi/\partial t$ spread around the torus at each major radius.

    Parameters
    ----------
    r : array_like
        Major radius [m].
    dpsi_dt : array_like
        Time derivative of the **full-weber** poloidal flux, broadcastable
        against ``r`` [Wb/s].

    Returns
    -------
    np.ndarray
        Toroidal electric field [V/m].

    Convention
    ----------
    ``dpsi_dt`` is in weber, not weber per radian: the $2\pi$ here is the one
    that turns a flux into a loop voltage, so passing a per-radian flux gives an
    answer $2\pi$ too small.  Green's-function flux
    (:func:`vaft.formula.green.green_psi_exact`) is already full weber and needs
    no conversion.  The sign is the one this expression gives;
    :func:`loop_voltage_from_total_flux` disagrees with it on both sign and flux
    normalisation, which is `#354 <https://github.com/VEST-Tokamak/vaft/issues/354>`_
    and still open.  The start-up chain in :mod:`vaft.formula.startup` takes the
    *magnitude* of this field -- an avalanche has no preferred direction -- so it
    does not inherit the disagreement and must not be used to settle it.

    Validity
    --------
    Machine-independent.  Faraday's law in axisymmetry, so it holds wherever the
    flux does; it says nothing on its own about whether that field will break
    the gas down, which also needs the connection length and the fill pressure.

    References
    ----------
    .. [1] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2 (the toroidal
           electric field required for tokamak start-up).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 11.1 (start-up and breakdown).

    See Also
    --------
    vaft.formula.startup.lloyd_breakdown_field
    vaft.formula.startup.breakdown_margin
    """
    r_arr = np.asarray(r, dtype=float)
    return -np.asarray(dpsi_dt, dtype=float) / (2.0 * np.pi * r_arr)


def poloidal_field_factor(
    cocos: int | None, *, psi_per_radian: bool | None = None,
) -> float:
    r"""Sauter Eq. 20 prefactor $k = \sigma_{R\varphi Z}\,\sigma_{B_p}/(2\pi)^{e_{B_p}}$.

    $$B_R = \frac{k}{R}\,\frac{\partial\psi}{\partial Z}, \qquad
      B_Z = -\frac{k}{R}\,\frac{\partial\psi}{\partial R}$$

    The factor carries both the $2\pi$ normalisation *and* the orientation sign,
    so applying only the former leaves the field inverted for half the
    conventions.

    Parameters
    ----------
    cocos : int or None
        COCOS index (1-8, 11-18); ``None`` selects the historical behaviour [-].
    psi_per_radian : bool or None, optional
        Storage family of the flux when ``cocos`` is ``None`` [bool].
        ``False`` removes the $2\pi$ of a full-weber flux while keeping the $-1$
        orientation; ``True`` and ``None`` keep the per-radian assumption.

    Returns
    -------
    float
        Prefactor $k$ multiplying $\nabla\psi/R$ [-].

    Convention
    ----------
    ``cocos=None`` keeps the weber-per-radian, $k=-1$ behaviour that the rest of
    this module assumed before conventions were explicit: the COCOS 2/3/6/7 form.
    Pass an index to get any other; it is resolved through
    :func:`vaft.data.cocos.cocos_spec`, the single source of truth for the COCOS
    model in VAFT.  The two halves are established by different evidence, so they
    can be known separately: ``psi_per_radian`` supplies the $2\pi$ half on its
    own for a caller that settled the storage family without pinning the index
    (an ODS whose flux scale is unambiguous while ``clockwise_phi`` leaves the
    index open).  It is consulted only when ``cocos`` is ``None``, where the
    orientation still falls back to $-1$; an index carries both halves and wins
    outright.

    See Also
    --------
    vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (20) and Table I.
    """
    if cocos is None:
        # False is the only value that changes anything: True and None both mean
        # "no 2*pi to remove", which is the historical assumption.
        return -1.0 if psi_per_radian in (None, True) else -1.0 / (2.0 * np.pi)
    from vaft.data.cocos import cocos_spec

    return cocos_spec(int(cocos)).bp_factor


def radial_magnetic_field_from_psi(psi: np.ndarray,
                                   R: np.ndarray,
                                   Z: np.ndarray,
                                   cocos: int | None = None) -> np.ndarray:
    r"""Radial magnetic field $B_R$ from the poloidal flux map.

    $$B_R = \frac{k}{R}\,\frac{\partial\psi}{\partial Z}, \qquad
      k = \frac{\sigma_{R\varphi Z}\,\sigma_{B_p}}{(2\pi)^{e_{B_p}}}$$

    with $k$ from :func:`poloidal_field_factor` (Sauter Eq. 20).

    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux along the vertical direction; a 1-D cut in $Z$ [Wb/rad or Wb].
    R : np.ndarray or float
        Major radius of the samples [m].
    Z : np.ndarray
        Vertical coordinate of the samples, monotonic [m].
    cocos : int or None, optional
        COCOS index fixing sign and $2\pi$; ``None`` = per-radian, $k=-1$ [-].

    Returns
    -------
    np.ndarray
        Radial field on the samples [T].

    Convention
    ----------
    ``cocos=None`` assumes ``psi`` in Wb/rad with $\sigma_{B_p}\sigma_{R\varphi Z}
    =-1$ (COCOS 2/3/6/7).  A full-weber map (IMAS Data Dictionary,
    :func:`vaft.formula.green.green_psi_exact`) passed without ``cocos``
    overestimates $|B_R|$ by $2\pi$; convert with
    :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor` or pass the index.

    Assumptions
    -----------
    Axisymmetry.  The derivative is taken along the *first* axis of ``psi``
    against ``Z``, so the map must be indexed ``[Z]`` or ``[Z, R]``; a
    ``[R, Z]``-ordered 2-D array differentiates along the wrong axis.

    Numerical notes
    ---------------
    ``numpy.gradient`` (second-order interior, first-order one-sided ends).

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (20) and Table I.
    """

    return poloidal_field_factor(cocos)/R * gradient(Z, psi)

def vertical_magnetic_field_from_psi(psi: np.ndarray,
                                   R: np.ndarray,
                                   Z: np.ndarray,
                                   cocos: int | None = None) -> np.ndarray:
    r"""Vertical magnetic field $B_Z$ from the poloidal flux map.

    $$B_Z = -\frac{k}{R}\,\frac{\partial\psi}{\partial R}, \qquad
      k = \frac{\sigma_{R\varphi Z}\,\sigma_{B_p}}{(2\pi)^{e_{B_p}}}$$

    with $k$ from :func:`poloidal_field_factor` (Sauter Eq. 20).

    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux along the radial direction; a 1-D cut in $R$ [Wb/rad or Wb].
    R : np.ndarray
        Major radius of the samples, monotonic [m].
    Z : np.ndarray
        Vertical coordinate of the samples, unused by the derivative [m].
    cocos : int or None, optional
        COCOS index fixing sign and $2\pi$; ``None`` = per-radian, $k=-1$ [-].

    Returns
    -------
    np.ndarray
        Vertical field on the samples [T].

    Convention
    ----------
    ``cocos=None`` assumes ``psi`` in Wb/rad with $\sigma_{B_p}\sigma_{R\varphi Z}
    =-1$ (COCOS 2/3/6/7).  A full-weber map (IMAS Data Dictionary,
    :func:`vaft.formula.green.green_psi_exact`) passed without ``cocos``
    overestimates $|B_Z|$ by $2\pi$; convert with
    :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor` or pass the index.

    Assumptions
    -----------
    Axisymmetry.  The derivative is taken along the *first* axis of ``psi``
    against ``R``, so the map must be indexed ``[R]`` or ``[R, Z]``.

    Numerical notes
    ---------------
    ``numpy.gradient`` (second-order interior, first-order one-sided ends).

    References
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (20) and Table I.
    """
    return -poloidal_field_factor(cocos)/R * gradient(R, psi)





def beta_toroidal_from_p_B0(p_average: float,
                            B0: float) -> float:
    r"""Toroidal beta from the volume-averaged pressure and the vacuum field.
    
    $$\beta_t = \frac{2\mu_0 \langle p \rangle_V}{B_0^2}$$
    
    Parameters
    ----------
    p_average : float
        Volume-averaged plasma pressure [Pa].
    B0 : float
        Vacuum toroidal field at ``r0`` (``equilibrium.vacuum_toroidal_field.b0``),
        as the DD requires [T].
    
    Returns
    -------
    float
        Toroidal beta [-].
    
    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Ch. 3, Equilibrium (definition of beta).
    .. [2] IMAS Data Dictionary, ``equilibrium.time_slice[:].global_quantities.beta_tor``.
    """
    return 2 * MU0 * float(p_average) / float(B0) ** 2


def beta_poloidal_from_pressure_integral(pressure_integral: float,
                                         R0: float,
                                         Ip: float) -> float:
    r"""Poloidal beta in the IMAS definition, from the pressure volume integral.
    
    $$\beta_p = \frac{4 \int p \, dV}{R_0 \mu_0 I_p^2}$$
    
    Parameters
    ----------
    pressure_integral : float
        Plasma pressure integrated over the plasma volume [Pa m^3].
    R0 : float
        Reference major radius the DD normalizes by [m].
    Ip : float
        Plasma current [A].
    
    Returns
    -------
    float
        Poloidal beta [-].
    
    Convention
    ----------
    This is the DD-normative ``beta_pol``, normalized by ``R_0 mu_0 Ip^2``.
    The EFIT/OMFIT circumference form is a different definition; see
    :func:`beta_poloidal_from_circumference`.
    
    References
    ----------
    .. [1] IMAS Data Dictionary, ``equilibrium.time_slice[:].global_quantities.beta_pol``.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Ch. 3, Equilibrium (definition of beta).
    """
    return 4 * float(pressure_integral) / (float(R0) * MU0 * float(Ip) ** 2)


def beta_normal_from_beta_tor(beta_tor: float,
                              a: float,
                              B0: float,
                              Ip: float) -> float:
    r"""Normalized beta (Troyon) from toroidal beta, minor radius, field and current.
    
    $$\beta_N = 100\,\beta_t \frac{a |B_0|}{|I_p[\mathrm{MA}]|}$$
    
    Parameters
    ----------
    beta_tor : float
        Toroidal beta [-].
    a : float
        Minor radius [m].
    B0 : float
        Vacuum toroidal field at ``r0`` [T].
    Ip : float
        Plasma current; converted to MA internally [A].
    
    Returns
    -------
    float
        Normalized beta [% m T/MA].
    
    References
    ----------
    .. [1] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    .. [2] IMAS Data Dictionary, ``equilibrium.time_slice[:].global_quantities.beta_normal``.
    """
    return 100 * float(beta_tor) * float(a) * abs(float(B0)) / abs(float(Ip) / 1e6)


def li_3_from_Bp2_volume_integral(Bp2_dV: float,
                                  Ip: float,
                                  R0: float) -> float:
    r"""Internal inductance in the IMAS ``li_3`` definition, from the poloidal-field energy.
    
    $$l_{i3} = \frac{2 \int B_p^2 \, dV}{\mu_0^2 I_p^2 R_0}$$
    
    The same quantity OMFIT reports as ``li_(3)_IMAS``.
    
    Parameters
    ----------
    Bp2_dV : float
        Poloidal field squared integrated over the plasma volume [T^2 m^3].
    Ip : float
        Plasma current [A].
    R0 : float
        Reference major radius the DD normalizes by [m].
    
    Returns
    -------
    float
        Internal inductance, ``li_3`` definition [-].
    
    References
    ----------
    .. [1] IMAS Data Dictionary, ``equilibrium.time_slice[:].global_quantities.li_3``.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Ch. 3, Equilibrium (internal inductance).
    """
    return 2 * float(Bp2_dV) / (MU0**2 * float(Ip) ** 2 * float(R0))


def beta_poloidal_from_circumference(p_average: float,
                                     Ip: float,
                                     length_pol: float) -> float:
    r"""Poloidal beta in the EFIT/OMFIT convention, normalized by the LCFS circumference.
    
    $$\beta_{p,\mathrm{circ}} = \frac{2\mu_0 \langle p \rangle_V}{B_{pa}^2}, \qquad
      B_{pa} = \frac{\mu_0 I_p}{L_{pol}}$$
    
    Parameters
    ----------
    p_average : float
        Volume-averaged plasma pressure [Pa].
    Ip : float
        Plasma current [A].
    length_pol : float
        Poloidal circumference of the last closed flux surface [m].
    
    Returns
    -------
    float
        Poloidal beta, circumference convention [-].
    
    Convention
    ----------
    **Not** the IMAS DD's ``beta_pol``, and not an estimate of it: this
    normalizes the volume-averaged pressure by the poloidal field implied by
    the LCFS *circumference* rather than by ``R_0 mu_0 Ip^2``.  The two differ
    by the geometric factor ``R_0 L_pol^2 / (2 V)`` -- 26% on the packaged
    kineticEfit reference, where this form reproduces the stored value to
    0.1% and the DD form does not.  It exists so the database summary can keep
    reporting what OMFIT reported; ``global_quantities.beta_pol`` stays
    DD-normative (see :func:`beta_poloidal_from_pressure_integral` and issue
    #318, which owns the sensitivity study of the two).
    
    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh, A. G. Kellman and
           W. Pfeiffer, Nucl. Fusion 25 (1985) 1611 (EFIT definitions).
    """
    b_pa = MU0 * abs(float(Ip)) / float(length_pol)
    return 2 * MU0 * float(p_average) / b_pa**2


# ------------------------------------------------------------------
# Current Limits
# ------------------------------------------------------------------

def current_limit_from_q(q_95: float,
                        a: float,
                        B0: float) -> float:
    r"""Plasma current at a prescribed edge safety factor, cylindrical approximation.

    $$I_p = \frac{2\pi a^2 B_0}{\mu_0\,q_{95}}$$

    the inversion of $q_{cyl} = 2\pi a^2 B_0/(\mu_0 R\,I_p)\times R/R$ for a
    circular cross-section written per unit major radius.

    Parameters
    ----------
    q_95 : float
        Target safety factor at the 95% flux surface [-].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field on axis [T].

    Returns
    -------
    float
        Plasma current reaching ``q_95`` [A m].

    Assumptions
    -----------
    Circular, large-aspect-ratio cylinder with $q_{95}$ standing in for the
    cylindrical $q_a$.

    Limitations
    -----------
    As written the expression lacks the $1/R$ of the cylindrical safety factor
    $q = 2\pi a^2 B_0/(\mu_0 R I_p)$ and therefore returns $I_p R$ [A m], not a
    current; divide by $R$ for amperes.  Elongation is ignored.  Tracked in #362.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.4 (cylindrical safety factor).
    """
    return 2 * np.pi * a**2 * B0 / (MU0 * q_95)


def current_limit_from_beta(beta_N: float,
                          a: float,
                          B0: float) -> float:
    r"""Current figure obtained by substituting $\beta_N$ for $q$ in the cylindrical relation.

    $$I_p = \frac{2\pi a^2 B_0}{\mu_0\,\beta_N}$$

    Parameters
    ----------
    beta_N : float
        Normalised beta in whatever convention the caller uses [-].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field on axis [T].

    Returns
    -------
    float
        The expression above [A m].

    Limitations
    -----------
    Dimensionally the cylindrical-$q$ formula with $\beta_N$ in the place of $q$;
    no derivation or source records what this is meant to bound, and the result
    depends on the units chosen for $\beta_N$.  Kept for compatibility; prefer
    :func:`vaft.formula.stability.beta_N_from_beta_a_B0_Ip` and the Troyon limit.
    Tracked in #362.

    References
    ----------
    .. [1] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209 (the
           $\beta_N$ limit this appears to invert).
    """
    return 2 * np.pi * a**2 * B0 / (MU0 * beta_N)


# ------------------------------------------------------------------
# Stored Energy
# ------------------------------------------------------------------

def stored_energy_from_p_V(p: Union[float, np.ndarray],
                          V: float) -> Union[float, np.ndarray]:
    r"""Stored energy as pressure times volume, $W = pV$.

    $$W = \int p\,dV \approx p\,V$$

    Parameters
    ----------
    p : float or np.ndarray
        Pressure; a volume-averaged value gives the total energy [Pa].
    V : float
        Plasma volume [m^3].

    Returns
    -------
    float or np.ndarray
        Energy $pV$ [J].

    Convention
    ----------
    $pV$ is the *magnetic-like* energy normalisation; the thermal energy of an
    ideal gas is $W_{th} = \tfrac{3}{2}\int p\,dV$, so multiply by 1.5 for the
    IMAS ``energy_thermal`` convention.

    Assumptions
    -----------
    ``p`` is the volume average (or the profile is flat) when ``V`` is the total
    volume.
    """
    return p * V


def stored_energy_from_beta_V(beta: float,
                            B0: float,
                            V: float) -> float:
    r"""Stored energy from toroidal beta, $W = \beta B_0^2 V/(2\mu_0)$.

    $$W = \beta\,\frac{B_0^2}{2\mu_0}\,V$$

    Parameters
    ----------
    beta : float
        Toroidal beta as a fraction (not percent), $\langle p\rangle/(B_0^2/2\mu_0)$ [-].
    B0 : float
        Toroidal field on axis [T].
    V : float
        Plasma volume [m^3].

    Returns
    -------
    float
        Energy $\langle p\rangle V$ [J].

    Convention
    ----------
    Uses the fraction form of $\beta_t$; a percentage input is 100 times too
    large.  As for :func:`stored_energy_from_p_V`, the result is $\langle p\rangle
    V$, so the thermal energy is 1.5 times it.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.5 (definition of beta).
    """
    return beta * B0**2 * V / (2 * MU0)

# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def volume_from_RZ_boundary(R: np.ndarray,
                           Z: np.ndarray) -> float:
    r"""Plasma volume from a boundary polygon, mean-radius approximation.

    $$V = 2\pi\oint R\,Z\,dR \approx 2\pi\,A_{\mathrm{poly}}\,\bar R$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the boundary vertices, closed or open polygon [m].
    Z : np.ndarray
        Height of the boundary vertices [m].

    Returns
    -------
    float
        Volume of the solid of revolution [m^3].

    Assumptions
    -----------
    $\bar R$ is the *arithmetic* mean of the vertex radii, not the area centroid
    demanded by Pappus' theorem; exact only for a boundary symmetric about
    $\bar R$.

    Limitations
    -----------
    On VEST flux surfaces the mean-radius factorisation differs from the exact
    contour integral by up to ~6 % at the edge; use
    :func:`exact_volume_from_RZ_contour` for a reported volume.

    Numerical notes
    ---------------
    Shoelace formula for the polygon area (exact for straight edges); the polygon
    is implicitly closed by ``numpy.roll``.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1 (plasma volume of a toroidal cross-section).
    """
    # Calculate polygon area
    area = 0.5 * np.abs(np.dot(R, np.roll(Z, 1)) - np.dot(Z, np.roll(R, 1)))
    # R̄: area-weighted mean radius (approximation)
    R_bar = np.mean(R)
    return 2 * np.pi * area * R_bar


def exact_volume_from_RZ_contour(R: np.ndarray,
                                 Z: np.ndarray) -> float:
    r"""Plasma volume from a closed $(R, Z)$ contour by Green's theorem.

    $$V = \pi\oint R^2\,dZ$$

    exact for the solid of revolution swept by a closed poloidal contour, with
    no mean-radius approximation.

    Parameters
    ----------
    R : np.ndarray
        Major radius of the contour vertices, at least 3 [m].
    Z : np.ndarray
        Height of the contour vertices [m].

    Returns
    -------
    float
        Volume of the solid of revolution [m^3].

    Convention
    ----------
    Orientation-agnostic: the sign of the contour integral follows the traversal
    direction and the magnitude is returned.  The contour is closed automatically
    when its first and last points differ.

    Limitations
    -----------
    Unlike :func:`volume_from_RZ_boundary`, which factors the integral as
    $2\pi A_{\mathrm{poly}}\bar R$ with $\bar R = \mathrm{mean}(R)$, this
    evaluates the contour integral itself.  On VEST flux surfaces the two differ
    by up to ~6 % at the plasma edge, where the $\bar R$ factorisation is
    weakest.  Use this one whenever the volume is a reported quantity rather than
    an intermediate.

    Numerical notes
    ---------------
    Trapezoidal rule on $R^2$ between successive vertices, second-order in the
    vertex spacing.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.
    """
    R = np.asarray(R, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    if R.size != Z.size:
        raise ValueError("R and Z must have the same length")
    if R.size < 3:
        raise ValueError("a contour needs at least 3 points")
    if R[0] != R[-1] or Z[0] != Z[-1]:
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
    # Trapezoidal ∮ R² dZ; the sign follows the traversal direction, so take
    # the magnitude and let the caller stay orientation-agnostic.
    return float(abs(np.pi * np.sum(0.5 * (R[:-1] ** 2 + R[1:] ** 2) * np.diff(Z))))


def elongation_from_RZ_boundary(R: np.ndarray,
                               Z: np.ndarray) -> float:
    r"""Boundary elongation $\kappa$ from the extremal points of a contour.

    $$\kappa = \frac{Z_{\max} - Z_{\min}}{2a}, \qquad a = \frac{R_{\max} - R_{\min}}{2}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the boundary points [m].
    Z : np.ndarray
        Height of the boundary points [m].

    Returns
    -------
    float
        Elongation [-].

    Convention
    ----------
    The IMAS ``boundary.elongation`` definition (half-height over half-width of
    the bounding box), not an area-based or flux-surface-averaged elongation.

    Limitations
    -----------
    A single-point or degenerate contour divides by zero; no validation.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.
    .. [2] IMAS Data Dictionary, ``equilibrium.time_slice[:].boundary.elongation``.
    """
    a = (R.max() - R.min()) / 2
    return (Z.max() - Z.min()) / (2 * a)


def r_at_z_extremum_from_RZ_contour(R: np.ndarray,
                                   Z: np.ndarray,
                                   *,
                                   upper: bool) -> float:
    r"""Major radius where a closed contour reaches its highest or lowest point.

    $$R_{Z_{\mathrm{ext}}} = R_i + |s|\,(R_{i\pm1} - R_i), \qquad
      s = \frac{1}{2}\,\frac{Z_{i-1} - Z_{i+1}}{Z_{i-1} - 2Z_i + Z_{i+1}}$$

    with $i$ the index of the extreme sample and $s$ the vertex of the parabola
    through its two neighbours.

    Parameters
    ----------
    R : np.ndarray
        Major radius of the contour points [m].
    Z : np.ndarray
        Height of the contour points, same length as ``R`` [m].
    upper : bool
        ``True`` for the highest point, ``False`` for the lowest [-].

    Returns
    -------
    float
        Major radius at that extremum [m].

    Convention
    ----------
    Sub-vertex, not nearest-vertex.  Reading the major radius off the sampled
    vertex of extreme height is wrong by several percent in triangularity
    because the true extremum falls between vertices, so a parabola is fitted to
    height over the three points around the extreme sample and the major radius
    is interpolated there.  Indices wrap, so the contour is treated as closed.
    A repeated final point is dropped first: without that, an extremum landing on
    the seam takes its own duplicate as a neighbour, the parabola degenerates and
    the fit silently falls back to the vertex.

    Limitations
    -----------
    Falls back to the extreme vertex for a contour of fewer than three distinct
    points, and for a flat neighbourhood where the parabola is degenerate.  The
    parabola is local, so a contour too coarsely sampled to resolve its own
    curvature near the extremum is still limited by that sampling.

    See Also
    --------
    triangularity_upper_from_RZ_boundary
    vaft.process.equilibrium.r_at_z_extremum

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.
    """
    R = np.asarray(R, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    # Marching-squares contours and g-file boundaries usually repeat the first
    # point to close the loop.  Left in place it becomes the wrap-around
    # neighbour of an extremum at index 0, which makes z_prev == z_here and
    # collapses the parabola to the vertex it was meant to improve on.
    if Z.size > 1 and R[0] == R[-1] and Z[0] == Z[-1]:
        R, Z = R[:-1], Z[:-1]
    index = int(np.argmax(Z) if upper else np.argmin(Z))
    size = Z.size
    if size < 3:
        return float(R[index])
    prev, nxt = (index - 1) % size, (index + 1) % size
    z_prev, z_here, z_next = float(Z[prev]), float(Z[index]), float(Z[nxt])
    denominator = z_prev - 2.0 * z_here + z_next
    if denominator == 0.0:
        return float(R[index])
    # Vertex of the parabola through (-1, z_prev), (0, z_here), (1, z_next).
    shift = 0.5 * (z_prev - z_next) / denominator
    # |shift| <= 1/2 whenever the middle sample really is the extremum, so the
    # magnitude test only guards against a non-finite Z; it is not a case the
    # geometry can reach.
    if not np.isfinite(shift) or abs(shift) > 1.0:
        return float(R[index])
    r_here = float(R[index])
    neighbour = float(R[nxt] if shift > 0 else R[prev])
    return r_here + abs(shift) * (neighbour - r_here)


def triangularity_upper_from_RZ_boundary(R: np.ndarray,
                                        Z: np.ndarray,
                                        R0: float) -> float:
    r"""Upper boundary triangularity $\delta_u$ from the contour's highest point.

    $$\delta_u = \frac{R_0 - R_{Z_{\max}}}{a}, \qquad a = \frac{R_{\max} - R_{\min}}{2}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the boundary points [m].
    Z : np.ndarray
        Height of the boundary points [m].
    R0 : float
        Reference major radius; pass the geometric centre
        $(R_{\max} + R_{\min})/2$ for the IMAS definition [m].

    Returns
    -------
    float
        Upper triangularity [-].

    Convention
    ----------
    The IMAS ``boundary.triangularity_upper`` definition, measured at the
    vertically extremal point rather than at a midplane crossing.  Positive
    means the top of the plasma sits inboard of ``R0``.  ``R0`` is the caller's
    to choose because a boundary offset from its own bounding box has no single
    right centre; the IMAS value is the geometric centre, which is what
    :func:`vaft.process.equilibrium.contour_shape_parameters` uses.

    Limitations
    -----------
    A degenerate contour of zero width divides by zero; no validation.  The
    extremum is located to sub-vertex accuracy, so the result is only as good as
    the contour's sampling near the top.

    See Also
    --------
    triangularity_lower_from_RZ_boundary
    triangularity_from_RZ_boundary
    r_at_z_extremum_from_RZ_contour

    References
    ----------
    .. [1] IMAS Data Dictionary,
           ``equilibrium.time_slice[:].boundary.triangularity_upper``.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.
    """
    R = np.asarray(R, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    a = (R.max() - R.min()) / 2
    return (R0 - r_at_z_extremum_from_RZ_contour(R, Z, upper=True)) / a


def triangularity_lower_from_RZ_boundary(R: np.ndarray,
                                        Z: np.ndarray,
                                        R0: float) -> float:
    r"""Lower boundary triangularity $\delta_l$ from the contour's lowest point.

    $$\delta_l = \frac{R_0 - R_{Z_{\min}}}{a}, \qquad a = \frac{R_{\max} - R_{\min}}{2}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the boundary points [m].
    Z : np.ndarray
        Height of the boundary points [m].
    R0 : float
        Reference major radius; pass the geometric centre
        $(R_{\max} + R_{\min})/2$ for the IMAS definition [m].

    Returns
    -------
    float
        Lower triangularity [-].

    Convention
    ----------
    The IMAS ``boundary.triangularity_lower`` definition, measured at the
    vertically extremal point rather than at a midplane crossing.  Positive
    means the bottom of the plasma sits inboard of ``R0``.  An up-down symmetric
    boundary has $\delta_l = \delta_u$; the two differ only for an asymmetric
    one, which is why IMAS stores both.

    Limitations
    -----------
    A degenerate contour of zero width divides by zero; no validation.  The
    extremum is located to sub-vertex accuracy, so the result is only as good as
    the contour's sampling near the bottom.

    See Also
    --------
    triangularity_upper_from_RZ_boundary
    triangularity_from_RZ_boundary

    References
    ----------
    .. [1] IMAS Data Dictionary,
           ``equilibrium.time_slice[:].boundary.triangularity_lower``.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.
    """
    R = np.asarray(R, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    a = (R.max() - R.min()) / 2
    return (R0 - r_at_z_extremum_from_RZ_contour(R, Z, upper=False)) / a


def triangularity_from_RZ_boundary(R: np.ndarray,
                                  Z: np.ndarray,
                                  R0: float) -> float:
    r"""Boundary triangularity $\delta$, the mean of the upper and lower values.

    $$\delta = \tfrac{1}{2}(\delta_u + \delta_l), \qquad
      \delta_{u,l} = \frac{R_0 - R_{Z_{\max,\min}}}{a}$$

    Parameters
    ----------
    R : np.ndarray
        Major radius of the boundary points [m].
    Z : np.ndarray
        Height of the boundary points [m].
    R0 : float
        Reference major radius; pass the geometric centre
        $(R_{\max} + R_{\min})/2$ for the IMAS definition [m].

    Returns
    -------
    float
        Mean triangularity [-].

    Convention
    ----------
    The IMAS ``boundary.triangularity``, which is defined as the mean of the two
    extremity values and not as a separate measurement.  Measured at the two
    vertically extremal points.  Until #365 this function
    read the boundary sample nearest $Z=0$ instead, which returned exactly
    $-1$ for every boundary whose parameterisation starts at the outboard
    midplane -- that is, for every standard one -- so no caller can have
    depended on the old number.  Positive means the extremities sit inboard of
    ``R0``.

    Limitations
    -----------
    The mean is the right summary only for a boundary that is close to up-down
    symmetric; report $\delta_u$ and $\delta_l$ separately for a strongly
    asymmetric one.

    See Also
    --------
    triangularity_upper_from_RZ_boundary
    triangularity_lower_from_RZ_boundary
    vaft.process.equilibrium.contour_shape_parameters

    References
    ----------
    .. [1] IMAS Data Dictionary,
           ``equilibrium.time_slice[:].boundary.triangularity``.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.
    """
    return 0.5 * (
        triangularity_upper_from_RZ_boundary(R, Z, R0)
        + triangularity_lower_from_RZ_boundary(R, Z, R0)
    )


def eK_from_K(K: float) -> float:
    r"""Elongation parameter $e_K$ of the virial relations.

    $$e_K = \frac{K^2 - 1}{K^2 + 1}$$

    Parameters
    ----------
    K : float
        Elongation $\kappa$ [-].

    Returns
    -------
    float
        $e_K$, 0 for a circle and $\to1$ for infinite elongation [-].

    Physical interpretation
    -----------------------
    The ellipticity measure in which the Martynov-Pustovitov virial
    approximations are linear.

    References
    ----------
    .. [1] A. A. Martynov and V. D. Pustovitov, Phys. Plasmas 31 (2024), "Virial
           relations for elongated plasmas in tokamaks", definition preceding
           Eq. (21).
    """
    return (K**2 - 1) / (K**2 + 1)


def peaking_factor(central: float,
                   volume_avg: float) -> float:
    r"""Profile peaking factor, central value over volume average.

    $$\mathrm{PF} = \frac{X(0)}{\langle X\rangle}$$

    Parameters
    ----------
    central : float
        Value on the magnetic axis [any].
    volume_avg : float
        Volume average of the same quantity, same unit [any].

    Returns
    -------
    float
        Peaking factor [-].

    Numerical notes
    ---------------
    A zero volume average warns and returns ``nan``.

    See Also
    --------
    vaft.formula.utils.calculate_peaking_factor
    """
    return calculate_peaking_factor(central, volume_avg)

# ------------------------------------------------------------------
# Plasma Resistance
# ------------------------------------------------------------------

def spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(T_e: float,
                                                 Z_eff: float = 2.0,
                                                 ln_Lambda: float = 17.0) -> float:
    r"""Spitzer parallel resistivity $\eta_\parallel$.

    $$\eta = 5.2\times10^{-5}\,\frac{Z_{\mathrm{eff}}\,\ln\Lambda}{T_e^{3/2}}
      \quad[\Omega\,\mathrm{m}],\ T_e\ \text{in eV}$$

    Parameters
    ----------
    T_e : float
        Electron temperature [eV].
    Z_eff : float, optional
        Effective ion charge; default 2 [-].
    ln_Lambda : float, optional
        Coulomb logarithm; default 17 [-].

    Returns
    -------
    float
        Parallel resistivity [Ohm m].

    Convention
    ----------
    The NRL Formulary value $\eta_\parallel = 1.65\times10^{-9}\,Z\ln\Lambda\,
    T_{\mathrm{keV}}^{-3/2}\ \Omega$ m rewritten for $T_e$ in eV; the
    $Z_{\mathrm{eff}}$ factor is applied linearly (the Spitzer-Harm
    $Z$-dependence is weaker than linear for $Z>1$).

    Assumptions
    -----------
    Classical collisional plasma, no neoclassical trapped-particle correction.

    Validity
    --------
    Core tokamak plasmas well above the ionisation stage; the defaults
    $Z_{\mathrm{eff}}=2$ and $\ln\Lambda=17$ are typical rather than derived,
    use :func:`coulomb_logarithm_from_n_T` for a self-consistent value.

    Limitations
    -----------
    Neoclassical resistivity in a spherical tokamak exceeds this by the
    trapped-fraction factor (up to ~2 at VEST aspect ratio).

    References
    ----------
    .. [1] NRL Plasma Formulary (2019), p. 29 (Spitzer resistivity).
    .. [2] L. Spitzer and R. Harm, Phys. Rev. 89 (1953) 977.
    .. [3] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 2.16 (resistivity).
    """
    return SPITZER_RESISTIVITY_COEF * Z_eff * ln_Lambda / T_e**1.5


# ------------------------------------------------------------------
# Normalized Plasma Current
# ------------------------------------------------------------------

def normalized_plasma_current(Ip: Union[float, np.ndarray],
                            R: Union[float, np.ndarray],
                            a: Union[float, np.ndarray],
                            Bt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Normalised plasma current $I_N = I_p/(a B_t)$ in MA/(m T).

    $$I_N = \frac{I_p\,[\mathrm{MA}]}{a\,[\mathrm{m}]\,B_t\,[\mathrm{T}]}$$

    Parameters
    ----------
    Ip : float or np.ndarray
        Plasma current, converted to MA internally [A].
    R : float or np.ndarray
        Major radius; accepted for signature symmetry, unused [m].
    a : float or np.ndarray
        Minor radius [m].
    Bt : float or np.ndarray
        Toroidal field [T].

    Returns
    -------
    float or np.ndarray
        Normalised current [MA/(m T)].

    Convention
    ----------
    SI current in, engineering-unit ratio out: the same $I_N$ that normalises
    $\beta_N = \beta_t[\%]/I_N$ and that the ST beta-limit literature plots
    against.

    References
    ----------
    .. [1] J. E. Menard et al., Phys. Plasmas 23 (2016) 072508,
           https://doi.org/10.1063/1.4959808, Sec. II.
    .. [2] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    """
    Ip = Ip / 1e6  # Convert A to MA
    return Ip / (a * Bt)


def kink_safety_factor(R: Union[float, np.ndarray],
                      a: Union[float, np.ndarray],
                      kappa: Union[float, np.ndarray],
                      Ip: Union[float, np.ndarray],
                      Bt: Union[float, np.ndarray],
                      type_: str) -> Tuple[Union[float, np.ndarray], ...]:
    r"""Kink safety factor $q_*$ with the Freidberg beta and current limits.

    $$q_{\mathrm{kink}} = \frac{2\pi a^2 B_t}{\mu_0 I_p R}\;g(\kappa)$$

    with $g = 1$ (``'circular'``), $g = 1 + \tfrac{4}{\pi^2}(\kappa^2-1)$
    (``'conventional'``) or $g = \tfrac{1}{2}(1+\kappa^2)$ (``'ST'``), plus the
    matching Troyon-type $\beta$ limits and the current at $q_{\mathrm{kink}}
    = q_{\min}$.

    Parameters
    ----------
    R : float or np.ndarray
        Major radius [m].
    a : float or np.ndarray
        Minor radius [m].
    kappa : float or np.ndarray
        Elongation [-].
    Ip : float or np.ndarray
        Plasma current [A].
    Bt : float or np.ndarray
        Toroidal field [T].
    type_ : str
        ``'circular'``, ``'conventional'`` or ``'ST'``; anything else raises [str].

    Returns
    -------
    q_kink : float or np.ndarray
        Kink safety factor [-].
    q_min : float or np.ndarray
        Minimum stable kink safety factor $1 + \kappa/2$ [-].
    beta_max : float or np.ndarray or None
        Beta limit (fraction); ``None`` for ``'circular'`` [-].
    beta_crit : float or np.ndarray or None
        Critical beta (fraction); ``None`` for ``'circular'`` [-].
    ip_max : float or np.ndarray
        Current at which $q_{\mathrm{kink}}$ reaches $q_{\min}$ [A].

    Convention
    ----------
    $\beta$ limits are fractions, not percent, and are given in Freidberg's
    $\epsilon$-scaled form ($\beta_{\max} = 0.072\,\tfrac{1+\kappa^2}{2}\,
    \epsilon$ for the ST branch, $\pi^2\kappa\epsilon/16q^2$ for the
    conventional one).  $\mu_0$ is hard-coded as $4\pi\times10^{-7}$.

    Validity
    --------
    Freidberg's reduced-MHD estimates for external kink and pressure-driven
    limits; order-of-magnitude design numbers, not a stability code result.

    References
    ----------
    .. [1] J. P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge
           University Press (2007), Ch. 13, Eq. (13.158) and the surrounding
           kink and Troyon limit discussion.
    .. [2] F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    """
    mu0 = 4 * np.pi * 1e-7
    epsilon = a / R

    if type_ == 'circular':
        q_kink = 2 * np.pi * a**2 * Bt / (mu0 * Ip * R)
        beta_max = None
        beta_crit = None
    elif type_ == 'conventional':
        q_kink = 2 * np.pi * a**2 * kappa * Bt / (mu0 * Ip * R)
        g_factor = 1 / kappa * (1 + 4 / np.pi**2 * (kappa**2 - 1))
        q_kink *= g_factor
        beta_max = np.pi**2 / 16 * kappa * epsilon / q_kink**2
        beta_crit = 0.14 * epsilon * kappa / q_kink
    elif type_ == 'ST':
        q_kink = 2 * np.pi * a**2 * Bt / (mu0 * Ip * R) * (1 + kappa**2 / 2)
        beta_max = 0.072 * (1 + kappa**2) / 2 * epsilon
        betaN_braket = 0.03 * (q_kink - 1) / ((3/4)**4 + (q_kink - 1)**4)**(1/4)
        beta_crit = 5 * betaN_braket * (1 + kappa**2) / 2 * epsilon / q_kink
    else:
        raise ValueError("Invalid type specified. Must be 'circular', 'conventional', or 'ST'")

    q_min = 1 + kappa / 2
    ip_max = q_kink * Ip * 2 / (1 + kappa)

    return q_kink, q_min, beta_max, beta_crit, ip_max


# ------------------------------------------------------------------
# Plasma beta / energy
# ------------------------------------------------------------------
# W_K = (3/2) * (1/(2*mu0) * beta_p * B_pa^2 * V_p)
# W_M = (1/(2*mu0)) * li * B_pa^2 * V_p
def kinetic_energy_from_beta_p_B_pa_V_p(beta_p: float,
                                       B_pa: float,
                                       V_p: float) -> float:
    r"""Thermal energy from poloidal beta, $W_K = \tfrac{3}{2}\,\beta_p B_{pa}^2 V_p/(2\mu_0)$.

    $$W_K = \frac{3}{2}\left(\frac{\beta_p\,B_{pa}^2}{2\mu_0}\,V_p\right)$$

    Parameters
    ----------
    beta_p : float
        Poloidal beta, $2\mu_0\langle p\rangle/B_{pa}^2$ [-].
    B_pa : float
        Boundary-averaged poloidal field, $\mu_0 I_p/L_p$ [T].
    V_p : float
        Plasma volume [m^3].

    Returns
    -------
    float
        Thermal (kinetic) energy [J].

    Convention
    ----------
    $B_{pa}$ is the poloidal field averaged over the boundary contour of length
    $L_p$, the EFIT/Lao normalisation of $\beta_p$; the $3/2$ converts $pV$ to
    the ideal-gas thermal energy.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 2 (definitions of $\beta_p$ and $B_{pa}$).
    """
    return 1.5 * (1 / (2 * MU0) * beta_p * B_pa**2 * V_p)


def magnetic_energy_from_li_B_pa_V_p(li: float,
                                     B_pa: float,
                                     V_p: float) -> float:
    r"""Poloidal magnetic energy from the internal inductance, $W_M = l_i B_{pa}^2 V_p/(2\mu_0)$.

    $$W_M = \frac{l_i\,B_{pa}^2}{2\mu_0}\,V_p$$

    Parameters
    ----------
    li : float
        Internal inductance $\langle B_p^2\rangle/B_{pa}^2$ [-].
    B_pa : float
        Boundary-averaged poloidal field, $\mu_0 I_p/L_p$ [T].
    V_p : float
        Plasma volume [m^3].

    Returns
    -------
    float
        Poloidal field energy inside the plasma [J].

    Convention
    ----------
    The Lao/EFIT $l_i$ (volume average of $B_p^2$ over $B_{pa}^2$), which is
    neither the IMAS $l_{i,3}$ nor the large-aspect-ratio $l_i = \langle B_p^2
    \rangle/B_p(a)^2$; each differs by its normalising field.

    References
    ----------
    .. [1] L. L. Lao, H. St. John, R. D. Stambaugh and W. Pfeiffer, Nucl. Fusion
           25 (1985) 1421, Sec. 2.
    """
    return 1 / (2 * MU0) * li * B_pa**2 * V_p







































































# ------------------------------------------------------------------
# Power Density $S$
# ------------------------------------------------------------------


def bremsstrahlung_power_density_from_T_e_p_Z_eff(
    T_e: float,
    p: float,
    Z_eff: float = 2.0
) -> float:
    r"""Bremsstrahlung power density in the pressure form.

    $$S_B = Z_{\mathrm{eff}}\,K_B\,\frac{p_{\mathrm{bar}}^2}{T_{\mathrm{keV}}^{3/2}}
      \ [\mathrm{MW/m^3}], \qquad K_B = 0.052$$

    which is the NRL $P_{br} = 1.69\times10^{-38}\,Z_{\mathrm{eff}}n_e^2\sqrt{T_e}$
    rewritten with $n_e = p/(2T)$ (equal electron and ion pressure).

    Parameters
    ----------
    T_e : float
        Electron temperature [eV].
    p : float
        Total plasma pressure [Pa].
    Z_eff : float, optional
        Effective charge; default 2 [-].

    Returns
    -------
    float
        Radiated power density [W/m^3].

    Convention
    ----------
    Pressure is converted to bar ($10^5$ Pa) and temperature to keV internally;
    the prefactor 0.052 MW/m^3 follows exactly from the NRL coefficient and
    $p = 2n_eT$, so a pressure that already includes fast ions or unequal
    $T_i$ overestimates $n_e$.

    Assumptions
    -----------
    Maxwellian electrons, Gaunt factor 1, $n_i = n_e$ and $T_i = T_e$; no
    recombination or line radiation.

    References
    ----------
    .. [1] NRL Plasma Formulary (2019), p. 58 (bremsstrahlung).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Ch. 4 (radiation losses).
    """
    # convert pressure to 1e5 Pa units
    p_bar = p / 1e5
    # convert temperature to keV
    T_k_keV = T_e * 1e-3

    S_B_MW_m3 = K_B_COEF * (p_bar ** 2) / (T_k_keV ** 1.5)

    return Z_eff * S_B_MW_m3 * 1e6  # W/m^3



def bremsstrahlung_power_density_from_n_e_T_e_Z_eff(
    n_e_m3: float,
    T_e_eV: float,
    *,
    Z_eff: float = 2.0
) -> float:
    r"""Maxwellian free-free (bremsstrahlung) power density from first principles.

    $$S_B = \frac{2^{1/2}}{3\pi^{5/2}}\,\frac{e^6}{\varepsilon_0^3c^3h\,m_e^{3/2}}\;
      Z_{\mathrm{eff}}\,n_e^2\,\sqrt{k_BT_e}$$

    Parameters
    ----------
    n_e_m3 : float
        Electron density [m^-3].
    T_e_eV : float
        Electron temperature, converted to joules internally [eV].
    Z_eff : float, optional
        Effective charge, keyword-only; default 2 [-].

    Returns
    -------
    float
        Radiated power density [W/m^3].

    Convention
    ----------
    The prefactor evaluates to $4.2\times10^{-29}$ W m^3 J^-1/2, equal to the NRL
    $1.69\times10^{-38}\,n_e^2\sqrt{T_{\mathrm{eV}}}$ used by
    :func:`bremsstrahlung_radiation_power_from_z_eff_n_e_t_e`; the two agree to
    0.2 %, which is the rounding in NRL's published coefficient.

    ``Z_eff`` is keyword-only, unlike its two siblings.  The NRL form takes
    $(Z_{\mathrm{eff}}, n_e, T_e)$ and this one takes $(n_e, T_e)$, so a caller
    moving between them by name used to swap the arguments silently and get a
    number wrong by 27 orders of magnitude; there is no argument order that can
    now do that without raising.  Tracked in #760.

    Assumptions
    -----------
    Maxwellian, non-relativistic electrons; Gaunt factor 1; $Z_{\mathrm{eff}}$
    absorbs $\sum_Z Z^2 n_Z / n_e$.

    Validity
    --------
    $T_e \ll m_ec^2$; relativistic corrections exceed 10 % above ~50 keV.

    See Also
    --------
    bremsstrahlung_radiation_power_from_z_eff_n_e_t_e
    bremsstrahlung_power_density_from_T_e_p_Z_eff

    References
    ----------
    .. [1] G. B. Rybicki and A. P. Lightman, *Radiative Processes in
           Astrophysics*, Wiley (1979), Eq. (5.15b).
    .. [2] I. H. Hutchinson, *Principles of Plasma Diagnostics*, 2nd ed.,
           Cambridge University Press (2002), Sec. 5.3.
    """

    T_J = T_e_eV * QE
    return C_B * Z_eff * n_e_m3**2 * np.sqrt(T_J)


def bremsstrahlung_power_density_from_Z_eff_n_e_T_e(
    n_e_m3: float,
    T_e_eV: float,
    Z_eff: float = 2.0
) -> float:
    r"""Deprecated: use :func:`bremsstrahlung_power_density_from_n_e_T_e_Z_eff`.

    The name states the argument order everywhere else in this module, and this
    one stated it wrongly: it promised $(Z_{\mathrm{eff}}, n_e, T_e)$ while the
    signature was $(n_e, T_e, Z_{\mathrm{eff}})$.  Every argument is a float and
    $Z_{\mathrm{eff}}$ carried a default, so a call in the documented order was
    accepted and returned a value 27 orders of magnitude wrong.

    The shim keeps the old signature exactly and forwards unchanged, so a caller
    written against the *signature* keeps its answer and a caller written
    against the *name* keeps its wrong one -- a compatibility shim cannot tell
    the two apart.  What it adds is the warning naming the replacement, whose
    keyword-only ``Z_eff`` makes the mistake impossible to repeat.  Tracked in
    #760.

    See Also
    --------
    bremsstrahlung_power_density_from_n_e_T_e_Z_eff
    """
    warnings.warn(
        "`bremsstrahlung_power_density_from_Z_eff_n_e_T_e` is deprecated -- its "
        "name contradicts its argument order; use "
        "`bremsstrahlung_power_density_from_n_e_T_e_Z_eff(n_e, T_e, Z_eff=...)`. "
        "The old spelling is removed in 0.8.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bremsstrahlung_power_density_from_n_e_T_e_Z_eff(
        n_e_m3, T_e_eV, Z_eff=Z_eff
    )


# ------------------------------------------------------------------
# Flux Consumption
# ------------------------------------------------------------------
def surface_poloidal_flux_from_psi_boundary(psi_boundary: np.ndarray) -> float:
    r"""Total poloidal flux at the plasma surface, $\Phi_{surface} = 2\pi\psi_b$.

    $$\Phi_{\mathrm{surface}} = 2\pi\,\psi_b$$

    Parameters
    ----------
    psi_boundary : np.ndarray or float
        Poloidal flux at the plasma boundary [Wb/rad].

    Returns
    -------
    np.ndarray or float
        Surface flux in full weber [Wb].

    Convention
    ----------
    Converts a per-radian boundary flux (COCOS 1-8, EFIT g-file, VFIT) to the full
    flux that flux-consumption bookkeeping uses.  An IMAS full-weber
    ``global_quantities.psi_boundary`` (COCOS 11-18) must not be passed: the
    result would be $2\pi$ too large.  Tracked in #354.

    Physical interpretation
    -----------------------
    The poloidal flux linked by the plasma boundary, whose time derivative is the
    surface loop voltage in the Ejima flux-consumption balance.

    References
    ----------
    .. [1] S. Ejima et al., Nucl. Fusion 22 (1982) 1313, Sec. 2.
    .. [2] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Table I.
    """
    return psi_boundary * 2 * np.pi

def loop_voltage_from_total_flux(time_slice: np.ndarray, psi_boundary: np.ndarray) -> float:
    r"""Surface loop voltage from the time series of boundary flux.

    $$V_{\mathrm{loop}} = \frac{d\Phi_{\mathrm{surface}}}{dt} = 2\pi\,\frac{d\psi_b}{dt}$$

    Parameters
    ----------
    time_slice : np.ndarray
        Time of each sample, monotonic [s].
    psi_boundary : np.ndarray
        Boundary poloidal flux at each time [Wb/rad].

    Returns
    -------
    np.ndarray
        Loop voltage at the plasma surface at each time [V].

    Convention
    ----------
    Assumes ``psi_boundary`` per radian, as :func:`surface_poloidal_flux_from_psi_boundary`
    does; a full-weber IMAS flux gives a voltage $2\pi$ too large.  The sign is
    that of $d\psi_b/dt$ in the supplied COCOS, so a discharge with positive
    current and the usual $\sigma_{B_p}$ shows negative $V_{loop}$ during ramp-up.
    Tracked in `#354 <https://github.com/VEST-Tokamak/vaft/issues/354>`_, which is
    the disagreement with :func:`toroidal_electric_field` over both sign and flux
    normalisation.  The start-up chain in :mod:`vaft.formula.startup` sidesteps it
    by taking a field magnitude; that choice is not a resolution of #354.

    Physical interpretation
    -----------------------
    Sum of resistive and inductive voltage at the last closed flux surface, the
    quantity a flux loop on the boundary would read.

    Numerical notes
    ---------------
    ``numpy.gradient`` in time (second-order interior, first-order ends); noisy
    $\psi_b$ reconstructions need smoothing first.

    References
    ----------
    .. [1] S. Ejima et al., Nucl. Fusion 22 (1982) 1313, Sec. 2.

    See Also
    --------
    toroidal_electric_field
    vaft.formula.startup.breakdown_margin
    """
    return gradient(time_slice, psi_boundary) * 2 * np.pi

def inductive_voltage_from_dW_magdt_I_p(dW_magdt: float, I_p: float) -> float:
    r"""Inductive voltage from the rate of change of magnetic energy.

    $$V_{\mathrm{ind}} = \frac{1}{I_p}\,\frac{dW_{\mathrm{mag}}}{dt}$$

    Parameters
    ----------
    dW_magdt : float
        Rate of change of the poloidal magnetic energy [W].
    I_p : float
        Plasma current [A].

    Returns
    -------
    float
        Inductive voltage [V].

    Assumptions
    -----------
    Exact for $W = \tfrac{1}{2}LI^2$ with constant $L$; when the inductance
    changes (current-profile evolution) the full inductive voltage is
    $d(LI)/dt$ and this expression captures only part of it.
    """
    return dW_magdt / I_p


# ------------------------------------------------------------------
# Power Balance
# ------------------------------------------------------------------

def ohmic_heating_power_from_I_p_V_res(I_p: float,
                                        V_res: float) -> float:
    r"""Ohmic heating power $P_{ohm} = I_pV_{res}$.

    $$P_{\mathrm{ohm}} = I_p\,V_{\mathrm{res}}$$

    Parameters
    ----------
    I_p : float
        Plasma current [A].
    V_res : float
        Resistive part of the loop voltage [V].

    Returns
    -------
    float
        Ohmic heating power [W].

    Convention
    ----------
    With the *surface* loop voltage in place of $V_{res}$ the product also
    counts the inductive power $L\,dI_p/dt$ and the change of internal
    inductance, so it is a resistive-heating estimate only when $dI_p/dt \approx 0$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 5.1 (ohmic heating).
    """
    return I_p * V_res


def alpha_heating_power_from_n_D_n_T_T_keV_V(
    n_D_1e19: float, n_T_1e19: float, T_keV: float, V_m3: float
) -> float:
    r"""D-T alpha heating power with the rough $\langle\sigma v\rangle \propto T^2$ fit.

    $$P_\alpha = n_D\,n_T\,\langle\sigma v\rangle\,E_\alpha\,V, \qquad
      \langle\sigma v\rangle \approx 1.1\times10^{-24}\,T_{\mathrm{keV}}^2\ \mathrm{m^3/s}$$

    Parameters
    ----------
    n_D_1e19 : float
        Deuterium density [1e19 m^-3].
    n_T_1e19 : float
        Tritium density [1e19 m^-3].
    T_keV : float
        Ion temperature [keV].
    V_m3 : float
        Plasma volume [m^3].

    Returns
    -------
    float
        Alpha heating power [W].

    Assumptions
    -----------
    Flat profiles (the product of densities is taken at one temperature over
    the whole volume); all alpha energy deposited in the plasma.

    Validity
    --------
    Empirical fit.  The quadratic $\langle\sigma v\rangle$ is Wesson's
    interpolation of the D-T reactivity, accurate to ~10 % for $10 < T <
    20$ keV [1]_; outside that window use the Bosch-Hale parametrisation [2]_.

    Limitations
    -----------
    Irrelevant for a hydrogen device such as VEST, which runs hydrogen and
    occasionally helium and so sustains no D-T reaction; kept for power-balance
    completeness.  Tracked in #360 (Bosch-Hale replacement).

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 1.3.
    .. [2] H.-S. Bosch and G. M. Hale, Nucl. Fusion 32 (1992) 611, Table VII.
    """
    n_D = n_D_1e19 * 1e19  # m^-3
    n_T = n_T_1e19 * 1e19  # m^-3
    sigma_v = SIGMA_V_COEF * T_keV**2  # m^3/s (rough fit)
    return n_D * n_T * sigma_v * E_ALPHA * V_m3  # W


alpha_heating_power = alpha_heating_power_from_n_D_n_T_T_keV_V

def nbi_heating_power_from_I_nbi_V_nbi(I_nbi: float,
                                      V_nbi: float) -> float:
    r"""Neutral-beam injected power $P_{nbi} = I_{nbi}V_{nbi}$.

    $$P_{\mathrm{nbi}} = I_{\mathrm{nbi}}\,V_{\mathrm{nbi}}$$

    Parameters
    ----------
    I_nbi : float
        Beam current [A].
    V_nbi : float
        Acceleration voltage [V].

    Returns
    -------
    float
        Beam power leaving the injector [W].

    Limitations
    -----------
    Injected, not absorbed: neutralisation efficiency, duct losses and shine-through
    are not subtracted.
    """
    return I_nbi * V_nbi

def ec_heating_power_from_I_ec_V_ec(I_ec: float,
                                    V_ec: float) -> float:
    r"""Electron-cyclotron launched power $P_{ec} = I_{ec}V_{ec}$.

    $$P_{\mathrm{ec}} = I_{\mathrm{ec}}\,V_{\mathrm{ec}}$$

    Parameters
    ----------
    I_ec : float
        Gyrotron beam current [A].
    V_ec : float
        Gyrotron beam voltage [V].

    Returns
    -------
    float
        Electrical beam power of the source [W].

    Limitations
    -----------
    Gyrotron electrical power, not RF power (efficiency ~30-50 %) and not the
    power absorbed by the plasma.
    """
    return I_ec * V_ec


def auxiliary_heating_power(P_aux: float,
                          eta_CD: float) -> Tuple[float, float]:
    r"""Split auxiliary power into heating and current-drive parts.

    $$P_{CD} = \frac{P_{aux}}{1 + \eta_{CD}}, \qquad P_{heat} = P_{aux} - P_{CD}$$

    Parameters
    ----------
    P_aux : float
        Total auxiliary power [W].
    eta_CD : float
        Current-drive efficiency figure from :func:`current_drive_efficiency` [-].

    Returns
    -------
    P_heat : float
        Part of the power counted as heating [W].
    P_CD : float
        Part of the power counted as current drive [W].

    Limitations
    -----------
    A bookkeeping split with the same unsourced normalisation as
    ``eta_CD``; the two parts always sum to ``P_aux``.
    """
    P_CD = P_aux / (1 + eta_CD)
    P_heat = P_aux - P_CD
    return P_heat, P_CD


def heating_power_from_p_ohm_p_aux(P_ohm: float, P_aux: float) -> float:
    r"""Total heating power $P_{heat} = P_{ohm} + P_{aux}$.

    $$P_{\mathrm{heat}} = P_{\mathrm{ohm}} + P_{\mathrm{aux}}$$

    Parameters
    ----------
    P_ohm : float
        Ohmic heating power [W].
    P_aux : float
        Absorbed auxiliary heating power [W].

    Returns
    -------
    float
        Total heating power [W].
    """
    return P_ohm + P_aux


def bremsstrahlung_radiation_power_from_z_eff_n_e_t_e(Z_eff: float,
                                        n_e: float,
                                        T_e_eV: float) -> float:
    r"""Bremsstrahlung power density, NRL engineering form.

    $$p_{br} = 1.69\times10^{-38}\,Z_{\mathrm{eff}}\,n_e^2\,\sqrt{T_e\,[\mathrm{eV}]}
      \ [\mathrm{W/m^3}]$$

    Parameters
    ----------
    Z_eff : float
        Effective charge [-].
    n_e : float
        Electron density [m^-3].
    T_e_eV : float
        Electron temperature [eV].

    Returns
    -------
    float
        Radiated power density [W/m^3].

    Convention
    ----------
    The NRL Formulary $1.69\times10^{-32}\,n_e T_e^{1/2}\sum Z^2n_Z$ W/cm^3 with
    cm^-3 densities, converted to SI, and $\sum Z^2 n_Z = Z_{\mathrm{eff}}n_e$.

    Assumptions
    -----------
    Maxwellian electrons, Gaunt factor 1.

    References
    ----------
    .. [1] NRL Plasma Formulary (2019), p. 58.
    """
    return 1.69e-38 * Z_eff * n_e**2 * np.sqrt(T_e_eV)


def cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e(
    n_e_m3: float,
    B_t_T: float,
    T_e_eV: float,
) -> float:
    r"""Classical electron-cyclotron emission power density, no reabsorption.

    $$p_{\mathrm{cyc}} = \frac{e^4}{3\pi\varepsilon_0 m_e^3c^3}\,n_e\,B_t^2\,k_BT_e
      \approx 6.2\times10^{-17}\,B_t^2\,n_e\,T_{\mathrm{keV}}\ \mathrm{W/m^3}$$

    Parameters
    ----------
    n_e_m3 : float
        Electron density [m^-3].
    B_t_T : float
        Magnetic field [T].
    T_e_eV : float
        Electron temperature, converted to joules internally [eV].

    Returns
    -------
    float
        Emitted cyclotron power density [W/m^3].

    Physical interpretation
    -----------------------
    Total single-particle cyclotron radiation of a non-relativistic Maxwellian;
    the plasma is optically thick at the low harmonics, so the net loss is a
    small fraction of this, set by wall reflectivity and $\beta$ (Trubnikov).

    Validity
    --------
    Non-relativistic electrons; an upper bound on the loss, intended as a
    start-up loss-channel estimate rather than a radiation-transport result.

    Limitations
    -----------
    Ignores reabsorption and wall reflection, which reduce the net loss by one
    to two orders of magnitude in a tokamak.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Ch. 4 (cyclotron radiation).
    .. [2] B. A. Trubnikov, in *Reviews of Plasma Physics*, Vol. 7, Consultants
           Bureau (1979), p. 345.
    """
    # QE/EPS0/ME/C_LIGHT, not the lower-case `e`/`epsilon_0`/`m_e`/`c` this
    # module used to define: #368 moved those to `constants.py` because the
    # lower-case spellings leaked into `vaft.formula.__all__`, and this function
    # was the one caller left behind, so it has raised NameError ever since
    # (#753).
    coeff = QE**4 / (3.0 * np.pi * EPS0 * ME**3 * C_LIGHT**3)
    return coeff * n_e_m3 * B_t_T**2 * (T_e_eV * QE)

def loss_power_from_p_heat_dWdt_p_rad(P_heat: float, dWdt: float, p_rad: float) -> float:
    r"""Loss power $P_{loss} = P_{heat} - dW/dt - P_{rad}$.

    $$P_{\mathrm{loss}} = P_{\mathrm{heat}} - \frac{dW}{dt} - P_{\mathrm{rad}}$$

    Parameters
    ----------
    P_heat : float
        Total heating power [W].
    dWdt : float
        Rate of change of stored energy [W].
    p_rad : float
        Radiated power to subtract; 0 keeps radiation inside the loss [W].

    Returns
    -------
    float
        Loss power [W].

    Convention
    ----------
    With ``p_rad = 0`` this is the ITER-database $P_L$ that the confinement
    scalings are fitted to (radiation counted as a loss); with the core
    radiation subtracted it is the conducted-plus-convected loss to the
    boundary.  Use the same choice as the scaling being compared against.

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2, Sec. 3.
    """
    return P_heat - dWdt - p_rad

# ------------------------------------------------------------------
# Dimensionless Parameters
# ------------------------------------------------------------------

def inverse_aspect_ratio_from_a_R(a: float, R: float) -> float:
    r"""Inverse aspect ratio $\varepsilon = a/R$.

    $$\varepsilon = \frac{a}{R}$$

    Parameters
    ----------
    a : float
        Minor radius [m].
    R : float
        Major radius [m].

    Returns
    -------
    float
        Inverse aspect ratio [-].

    See Also
    --------
    calc_inverse_aspect_ratio : the same ratio with positivity validation.
    """
    return a / R

def aspect_ratio_from_a_R(a: float, R: float) -> float:
    r"""Aspect ratio $A = R/a = 1/\varepsilon$.

    $$A = \frac{R}{a}$$

    Parameters
    ----------
    a : float
        Minor radius [m].
    R : float
        Major radius [m].

    Returns
    -------
    float
        Aspect ratio [-].

    Notes
    -----
    Not to be confused with the elongation $\kappa$.
    """
    return R / a



def normalized_larmor_radius_from_M_T_a_Bt(M: float,
                               T: float,
                               a: float,
                               Bt: float) -> float:
    r"""Normalised ion gyroradius $\rho_* = \rho_i/a$ in SI inputs.

    $$\rho_* = \frac{\rho_i}{a}, \qquad
      \rho_i = \frac{m_i v_{th}}{eB_T} = \frac{\sqrt{2\,m_i\,eT_i}}{e\,B_T}$$

    with $v_{th} = \sqrt{2T_i/m_i}$ and $T_i$ in eV.

    Parameters
    ----------
    M : float
        Ion mass [kg].
    T : float
        Ion temperature [eV].
    a : float
        Minor radius [m].
    Bt : float
        Toroidal field [T].

    Returns
    -------
    float
        Normalised gyroradius [-].

    Convention
    ----------
    Thermal speed $\sqrt{2T/m}$ and the toroidal field, normalised by the minor
    radius: the ITER Physics Basis definition.  Differs from
    :func:`rho_star_from_M_T_B_R_epsilon` (mass in amu, normalised by $R\varepsilon$
    with a rounded prefactor) only in input units.  It agreed with neither
    :func:`vaft.formula.stability.rhostar_from_Te_a_Bt` until #364 gave that
    one the same definition; the two now return the same number for electrons,
    and the remaining spread is tracked in #353.

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2,
           Sec. 6 (dimensionless parameters).
    """
    # ρ* ∝ √(M T) / (a B_T)  (image scaling); constants kept explicitly
    return np.sqrt(2.0 * M * QE * T) / (QE * Bt * a)

def normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q(nu_ii: float,
                                     T_i_eV: float,
                                     M_i: float,
                                     R: float,
                                     a: float,
                                     q: float) -> float:
    r"""Ion collisionality $\nu_*$ from a collision frequency.

    $$\nu_* = \nu_{ii}\left(\frac{M_i}{eT_i}\right)^{1/2}\left(\frac{R}{a}\right)^{3/2} qR
      = \frac{\nu_{ii}\,qR}{\varepsilon^{3/2}\,v_{th,i}}$$

    the ratio of the effective detrapping frequency to the bounce frequency,
    with $v_{th,i} = \sqrt{eT_i/M_i}$.

    Parameters
    ----------
    nu_ii : float
        Ion-ion collision frequency [1/s].
    T_i_eV : float
        Ion temperature [eV].
    M_i : float
        Ion mass [kg].
    R : float
        Major radius [m].
    a : float
        Minor radius [m].
    q : float
        Safety factor [-].

    Returns
    -------
    float
        Normalised collisionality [-].

    Convention
    ----------
    Thermal speed $\sqrt{T/m}$ (no factor 2) and the collision frequency
    supplied by the caller; with Sauter's $\nu_{ii}$ this is Sauter Eq. (18b)
    without its numeric prefactor.  Two other $\nu_*$ definitions live in this
    package (:func:`normalized_collisionality_from_a_n_q_epsilon_T` and
    :func:`nu_star_from_n_T_B_R_epsilon_kappa_I`); tracked in #353.

    Physical interpretation
    -----------------------
    $\nu_* \ll 1$ is the banana (collisionless) regime, $\nu_* \gg
    \varepsilon^{-3/2}$ the Pfirsch-Schluter regime.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834,
           Eq. (18b).
    .. [2] F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239,
           Sec. IV (collisionality regimes).
    """
    # Note: ν* is dimensionless; this form matches the common scaling ν* ~ ν_ii qR / v_th · (R/a)^{3/2}
    return nu_ii * np.sqrt(M_i / (QE * T_i_eV)) * ((R / a)**1.5) * q * R

def normalized_collisionality_from_a_n_q_epsilon_T(a: float,
                                                   n: float,
                                                   q: float,
                                                   epsilon: float,
                                                   T_eV: float,
                                                   C: float = 1.0) -> float:
    r"""Collisionality scaling form $\nu_* \propto a\,n\,q/(\varepsilon^{5/2}T^2)$.

    $$\nu_* = C\,\frac{a\,n\,q}{\varepsilon^{5/2}\,T^2}$$

    which is the Sauter form $\nu_* = 6.921\times10^{-18}\,qRn\ln\Lambda/
    (T^2\varepsilon^{3/2})$ with $R = a/\varepsilon$ and $C = 6.921\times10^{-18}
    \ln\Lambda$ (electrons; times $Z^4$ for ions).

    Parameters
    ----------
    a : float
        Minor radius [m].
    n : float
        Density [m^-3].
    q : float
        Safety factor [-].
    epsilon : float
        Inverse aspect ratio [-].
    T_eV : float
        Temperature [eV].
    C : float, optional
        Proportionality constant; default 1 gives only the scaling [-].

    Returns
    -------
    float
        Collisionality, or with the default ``C`` only its scaling [-].

    Raises
    ------
    ValueError
        For non-positive ``epsilon`` or ``T_eV``.

    Convention
    ----------
    With ``C=1`` the number is not $\nu_*$ but proportional to it; supply
    $C = 6.921\times10^{-18}\ln\Lambda$ (with $n$ in m^-3 and $T$ in eV) for
    Sauter's electron collisionality.  Tracked with the other two definitions in
    #353.

    References
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834,
           Eq. (18b).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if T_eV <= 0:
        raise ValueError("T_eV must be > 0")
    return C * (a * n * q) / (epsilon**2.5 * T_eV**2)

def cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta(R: float,
                                           B: float,
                                           epsilon: float,
                                           I: float,
                                           f_kappa_delta: float) -> float:
    r"""Cylindrical safety factor with a shape function.

    $$q_{cyl} = \frac{2\pi\,\varepsilon^2 R\,B_T}{\mu_0\,I_p\,f(\kappa,\delta)}
      = \frac{2\pi a^2 B_T}{\mu_0 R I_p}\,\frac{1}{f(\kappa,\delta)}$$

    Parameters
    ----------
    R : float
        Major radius [m].
    B : float
        Toroidal field [T].
    epsilon : float
        Inverse aspect ratio $a/R$ [-].
    I : float
        Plasma current [A].
    f_kappa_delta : float
        Shape function; $1/\kappa$ reproduces the ITER $q_{cyl} = 5a^2\kappa B/(RI_{MA})$ [-].

    Returns
    -------
    float
        Cylindrical safety factor [-].

    Convention
    ----------
    $2\pi/\mu_0 = 5\times10^6$, so with SI current this is the ITER Physics
    Basis $q_{cyl}$ once $f = 1/\kappa$; the shape function is left to the
    caller because the ITER-89P and IPB98 databases used different $\kappa$
    definitions.  Unlike the flux-derivative $q$ it never changes sign.

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2,
           Sec. 3 (definition of $q_{cyl}$).
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 3.4.
    """
    return (2.0 * np.pi * epsilon**2 * R * B) / (MU0 * I * f_kappa_delta)


def _maybe_scalar(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return Python float for 0-d arrays, otherwise return NumPy array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _validate_positive(name: str, value: Union[float, np.ndarray]) -> np.ndarray:
    """
    Validate finite positive scalar/array input and return as float array.

    Parameters
    ----------
    name : str
        Input variable name used in error messages.
    value : float or np.ndarray
        Input value(s) to validate.
    """
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite. Got {value!r}")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be > 0. Got {value!r}")
    return arr


def coulomb_logarithm_from_n_T(
    n_m3: Union[float, np.ndarray],
    T_eV: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Coulomb logarithm $\ln\Lambda$ for electron collisions above 10 eV.

    $$\ln\Lambda = 30.9 - \ln\!\left(\frac{\sqrt{n_e\,[\mathrm{m^{-3}}]}}{T_e\,[\mathrm{eV}]}\right)$$

    Parameters
    ----------
    n_m3 : float or np.ndarray
        Electron density, strictly positive [m^-3].
    T_eV : float or np.ndarray
        Electron temperature, strictly positive [eV].

    Returns
    -------
    float or np.ndarray
        Coulomb logarithm [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    The NRL Formulary electron-electron/electron-ion form $24 - \ln(n_e^{1/2}
    T_e^{-1})$ with $n_e$ in cm^-3, converted to m^-3 ($24 + \ln 10^3 = 30.9$);
    also the convention of the Verdoolaege confinement-database analysis.

    Validity
    --------
    $T_e > 10$ eV; below that the NRL low-temperature branch applies.

    References
    ----------
    .. [1] NRL Plasma Formulary (2019), p. 34 (Coulomb logarithm).
    .. [2] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    """
    n_arr = _validate_positive("n_m3", n_m3)
    t_arr = _validate_positive("T_eV", T_eV)
    ln_lambda = 30.9 - np.log(np.sqrt(n_arr) / t_arr)
    return _maybe_scalar(ln_lambda)


def line_to_volume_avg_density(
    n_line_m3: Union[float, np.ndarray],
    factor: Union[float, np.ndarray] = 0.88,
) -> Union[float, np.ndarray]:
    r"""Volume-averaged density from a line average with a fixed profile factor.

    $$\langle n\rangle_V = f\,\bar n_l, \qquad f = 0.88\ \text{by default}$$

    Parameters
    ----------
    n_line_m3 : float or np.ndarray
        Line-averaged density, strictly positive [m^-3].
    factor : float or np.ndarray, optional
        Profile factor, strictly positive; default 0.88 [-].

    Returns
    -------
    float or np.ndarray
        Volume-averaged density [m^-3].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Assumptions
    -----------
    A moderately peaked density profile; 0.88 is the ITPA workflow value for
    H-mode-like profiles and is not a measurement.

    References
    ----------
    .. [1] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    """
    n_line_arr = _validate_positive("n_line_m3", n_line_m3)
    factor_arr = _validate_positive("factor", factor)
    return _maybe_scalar(n_line_arr * factor_arr)


def calc_inverse_aspect_ratio(
    a_m: Union[float, np.ndarray],
    R_geo_m: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Inverse aspect ratio $\varepsilon = a/R_{geo}$ with input validation.

    $$\varepsilon = \frac{a}{R_{geo}}$$

    Parameters
    ----------
    a_m : float or np.ndarray
        Minor radius, strictly positive [m].
    R_geo_m : float or np.ndarray
        Geometric major radius, strictly positive [m].

    Returns
    -------
    float or np.ndarray
        Inverse aspect ratio [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    See Also
    --------
    inverse_aspect_ratio_from_a_R : the unvalidated form.
    """
    a_arr = _validate_positive("a_m", a_m)
    r_arr = _validate_positive("R_geo_m", R_geo_m)
    epsilon = inverse_aspect_ratio_from_a_R(a_arr, r_arr)
    return _maybe_scalar(epsilon)


def rho_star_from_M_T_B_R_epsilon(
    M_eff_amu: Union[float, np.ndarray],
    T_eV: Union[float, np.ndarray],
    B_t_T: Union[float, np.ndarray],
    R_geo_m: Union[float, np.ndarray],
    epsilon: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Normalised ion gyroradius $\rho_*$ in the Verdoolaege engineering form.

    $$\rho_* = 1.44\times10^{-4}\,\frac{\sqrt{M_{eff}\,[\mathrm{amu}]\;T\,[\mathrm{eV}]}}
      {B_t\,[\mathrm{T}]\,R_{geo}\,[\mathrm{m}]\,\varepsilon}$$

    Parameters
    ----------
    M_eff_amu : float or np.ndarray
        Effective ion mass, strictly positive [amu].
    T_eV : float or np.ndarray
        Temperature, strictly positive [eV].
    B_t_T : float or np.ndarray
        Toroidal field, strictly positive [T].
    R_geo_m : float or np.ndarray
        Geometric major radius, strictly positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio, strictly positive [-].

    Returns
    -------
    float or np.ndarray
        Normalised gyroradius [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    $1.44\times10^{-4} = \sqrt{2m_p/e}$, i.e. $\rho_i = \sqrt{2m_iT}/(eB)$
    normalised by $a = R_{geo}\varepsilon$: the same physics as
    :func:`normalized_larmor_radius_from_M_T_a_Bt` in database units.  Tracked
    with the other $\rho_*$ definitions in #353.

    References
    ----------
    .. [1] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    """
    m_arr = _validate_positive("M_eff_amu", M_eff_amu)
    t_arr = _validate_positive("T_eV", T_eV)
    b_arr = _validate_positive("B_t_T", B_t_T)
    r_arr = _validate_positive("R_geo_m", R_geo_m)
    eps_arr = _validate_positive("epsilon", epsilon)
    rho_star = 1.44e-4 * np.sqrt(m_arr * t_arr) / (b_arr * r_arr * eps_arr)
    return _maybe_scalar(rho_star)


def beta_t_from_n_T_B(
    n_m3: Union[float, np.ndarray],
    T_eV: Union[float, np.ndarray],
    B_t_T: Union[float, np.ndarray],
    output: str = "percent",
) -> Union[float, np.ndarray]:
    r"""Toroidal beta from density, temperature and field.

    $$\beta_t\,[\%] = 8.05\times10^{-23}\,\frac{n\,[\mathrm{m^{-3}}]\;T\,[\mathrm{eV}]}{B_t^2\,[\mathrm{T^2}]}$$

    Parameters
    ----------
    n_m3 : float or np.ndarray
        Electron density, strictly positive [m^-3].
    T_eV : float or np.ndarray
        Temperature, strictly positive [eV].
    B_t_T : float or np.ndarray
        Toroidal field, strictly positive [T].
    output : str, optional
        ``"percent"`` (default) or ``"fraction"`` [str].

    Returns
    -------
    float or np.ndarray
        Toroidal beta in percent, or as a fraction [%].

    Raises
    ------
    ValueError
        For non-finite or non-positive input, or an unknown ``output``.

    Convention
    ----------
    $8.05\times10^{-23} = 100\times2\mu_0\times2e$: the total pressure is
    taken as $2n_eT$ (equal electron and ion temperatures, $n_i = n_e$), and
    the default output is a percentage.  Verdoolaege's ITER example is
    reproduced by construction.

    References
    ----------
    .. [1] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 3.5.
    """
    n_arr = _validate_positive("n_m3", n_m3)
    t_arr = _validate_positive("T_eV", T_eV)
    b_arr = _validate_positive("B_t_T", B_t_T)
    beta_percent = 8.05e-23 * n_arr * t_arr / (b_arr**2)

    normalized = output.strip().lower()
    if normalized == "percent":
        return _maybe_scalar(beta_percent)
    if normalized == "fraction":
        return _maybe_scalar(beta_percent / 100.0)
    raise ValueError("output must be either 'percent' or 'fraction'")


def q_cyl_from_B_R_epsilon_kappa_I(
    B_t_T: Union[float, np.ndarray],
    R_geo_m: Union[float, np.ndarray],
    epsilon: Union[float, np.ndarray],
    kappa_a: Union[float, np.ndarray],
    I_p_A: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Cylindrical safety factor in the Verdoolaege convention.

    $$q_{cyl} = 5\times10^{6}\,\frac{B_t\,R_{geo}\,\varepsilon^2\,\kappa_a}{I_p\,[\mathrm{A}]}$$

    Parameters
    ----------
    B_t_T : float or np.ndarray
        Toroidal field, strictly positive [T].
    R_geo_m : float or np.ndarray
        Geometric major radius, strictly positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio, strictly positive [-].
    kappa_a : float or np.ndarray
        Area elongation, strictly positive [-].
    I_p_A : float or np.ndarray
        Plasma current, strictly positive [A].

    Returns
    -------
    float or np.ndarray
        Cylindrical safety factor [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    :func:`cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta` with
    $f = 1/\kappa_a$ and $2\pi/\mu_0 = 5\times10^6$; $\kappa_a$ is the *area*
    elongation $S/(\pi a^2)$ of the confinement databases, not the boundary
    elongation.

    References
    ----------
    .. [1] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2, Sec. 3.
    """
    b_arr = _validate_positive("B_t_T", B_t_T)
    r_arr = _validate_positive("R_geo_m", R_geo_m)
    eps_arr = _validate_positive("epsilon", epsilon)
    kappa_arr = _validate_positive("kappa_a", kappa_a)
    i_arr = _validate_positive("I_p_A", I_p_A)
    q_cyl = cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta(
        R=r_arr,
        B=b_arr,
        epsilon=eps_arr,
        I=i_arr,
        f_kappa_delta=1.0 / kappa_arr,
    )
    return _maybe_scalar(q_cyl)


def nu_star_from_n_T_B_R_epsilon_kappa_I(
    n_m3: Union[float, np.ndarray],
    T_eV: Union[float, np.ndarray],
    B_t_T: Union[float, np.ndarray],
    R_geo_m: Union[float, np.ndarray],
    epsilon: Union[float, np.ndarray],
    kappa_a: Union[float, np.ndarray],
    I_p_A: Union[float, np.ndarray],
    ln_lambda: Union[float, np.ndarray, None] = None,
) -> Union[float, np.ndarray]:
    r"""Normalised collisionality $\nu_*$ in the Verdoolaege engineering form.

    $$\nu_* = 5\times10^{-11}\,\ln\Lambda\;\frac{n\,B_t\,R_{geo}^2\,\sqrt{\varepsilon}\,\kappa_a}
      {I_p\,T^2}$$

    Parameters
    ----------
    n_m3 : float or np.ndarray
        Electron density, strictly positive [m^-3].
    T_eV : float or np.ndarray
        Temperature, strictly positive [eV].
    B_t_T : float or np.ndarray
        Toroidal field, strictly positive [T].
    R_geo_m : float or np.ndarray
        Geometric major radius, strictly positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio, strictly positive [-].
    kappa_a : float or np.ndarray
        Area elongation, strictly positive [-].
    I_p_A : float or np.ndarray
        Plasma current, strictly positive [A].
    ln_lambda : float or np.ndarray or None, optional
        Coulomb logarithm; ``None`` computes :func:`coulomb_logarithm_from_n_T` [-].

    Returns
    -------
    float or np.ndarray
        Normalised collisionality [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    Sauter's $\nu_* = 6.921\times10^{-18}\,qRn\ln\Lambda/(T^2\varepsilon^{3/2})$
    with $q = q_{cyl}$ substituted gives $3.46\times10^{-11}$ in front; the
    $5\times10^{-11}$ used here is Verdoolaege's database convention and is 1.45
    times larger, so values are comparable only within one convention.  Tracked
    with the other $\nu_*$ definitions in #353.

    References
    ----------
    .. [1] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    .. [2] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834,
           Eq. (18b).
    """
    n_arr = _validate_positive("n_m3", n_m3)
    t_arr = _validate_positive("T_eV", T_eV)
    b_arr = _validate_positive("B_t_T", B_t_T)
    r_arr = _validate_positive("R_geo_m", R_geo_m)
    eps_arr = _validate_positive("epsilon", epsilon)
    kappa_arr = _validate_positive("kappa_a", kappa_a)
    i_arr = _validate_positive("I_p_A", I_p_A)

    if ln_lambda is None:
        ln_arr = np.asarray(coulomb_logarithm_from_n_T(n_arr, t_arr), dtype=float)
    else:
        ln_arr = _validate_positive("ln_lambda", ln_lambda)

    nu_star = (
        5.0e-11
        * ln_arr
        * n_arr
        * b_arr
        * (r_arr**2)
        * np.sqrt(eps_arr)
        * kappa_arr
        / (i_arr * (t_arr**2))
    )
    return _maybe_scalar(nu_star)


def omega_i_tau_E_from_B_tau_E_M(
    B_t_T: Union[float, np.ndarray],
    tau_E_s: Union[float, np.ndarray],
    M_eff_amu: Union[float, np.ndarray],
    Z_i: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    r"""Ion-cyclotron-normalised confinement time $\Omega_i\tau_E$.

    $$\Omega_i\tau_E = \frac{Z_i e B_t}{M_{eff}\,m_p}\,\tau_E$$

    Parameters
    ----------
    B_t_T : float or np.ndarray
        Toroidal field, strictly positive [T].
    tau_E_s : float or np.ndarray
        Energy confinement time, strictly positive [s].
    M_eff_amu : float or np.ndarray
        Effective ion mass, strictly positive [amu].
    Z_i : float or np.ndarray, optional
        Ion charge state, strictly positive; default 1 [-].

    Returns
    -------
    float or np.ndarray
        Normalised confinement time [-].

    Raises
    ------
    ValueError
        For non-finite or non-positive input.

    Convention
    ----------
    Exact SI angular cyclotron frequency with the proton mass and elementary
    charge from :mod:`vaft.formula.constants`; not a fitted prefactor.  The
    dependent variable of dimensionless confinement scalings.

    References
    ----------
    .. [1] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, Sec. 2.
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2, Sec. 6.
    """
    b_arr = _validate_positive("B_t_T", B_t_T)
    tau_arr = _validate_positive("tau_E_s", tau_E_s)
    m_eff_arr = _validate_positive("M_eff_amu", M_eff_amu)
    z_arr = _validate_positive("Z_i", Z_i)
    m_i = m_eff_arr * MI_P
    omega_i = z_arr * QE * b_arr / m_i
    return _maybe_scalar(omega_i * tau_arr)


def kadomtsev_constraint_from_engineering_exponents(
    alpha_I: Union[float, np.ndarray],
    alpha_B: Union[float, np.ndarray],
    alpha_P: Union[float, np.ndarray],
    alpha_n: Union[float, np.ndarray],
    alpha_R: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Residual of the Kadomtsev high-beta constraint on engineering exponents.

    $$\alpha_K = 4\alpha_R - 8\alpha_n - \alpha_I - 3\alpha_P - 5\alpha_B - 5$$

    for $\tau_E \propto I^{\alpha_I}B^{\alpha_B}P^{\alpha_P}n^{\alpha_n}R^{\alpha_R}$;
    $\alpha_K = 0$ when the scaling is expressible in the three dimensionless
    parameters $\rho_*$, $\beta$, $\nu_*$ alone.

    Parameters
    ----------
    alpha_I : float or np.ndarray
        Exponent of the plasma current [-].
    alpha_B : float or np.ndarray
        Exponent of the toroidal field [-].
    alpha_P : float or np.ndarray
        Exponent of the heating power [-].
    alpha_n : float or np.ndarray
        Exponent of the density [-].
    alpha_R : float or np.ndarray
        Exponent of the major radius [-].

    Returns
    -------
    float or np.ndarray
        Constraint residual, 0 for exact satisfaction [-].

    Raises
    ------
    ValueError
        For non-finite input.

    Physical interpretation
    -----------------------
    Dimensional analysis of the Vlasov-Maxwell system: only three of the
    engineering variables are independent once $\rho_*$, $\beta$ and $\nu_*$ are
    fixed at constant geometry.  ITER89P gives $\alpha_K = -0.15$ and IPB98(y,2)
    gives $-0.01$.

    References
    ----------
    .. [1] B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295.
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2,
           Sec. 6.2 (Kadomtsev constraint).
    .. [3] G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006.
    """
    a_i = np.asarray(alpha_I, dtype=float)
    a_b = np.asarray(alpha_B, dtype=float)
    a_p = np.asarray(alpha_P, dtype=float)
    a_n = np.asarray(alpha_n, dtype=float)
    a_r = np.asarray(alpha_R, dtype=float)
    for name, arr in (
        ("alpha_I", a_i),
        ("alpha_B", a_b),
        ("alpha_P", a_p),
        ("alpha_n", a_n),
        ("alpha_R", a_r),
    ):
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must be finite. Got {arr!r}")
    alpha_k = 4.0 * a_r - 8.0 * a_n - a_i - 3.0 * a_p - 5.0 * a_b - 5.0
    return _maybe_scalar(alpha_k)


def check_kadomtsev_constraint(
    alpha_I: Union[float, np.ndarray],
    alpha_B: Union[float, np.ndarray],
    alpha_P: Union[float, np.ndarray],
    alpha_n: Union[float, np.ndarray],
    alpha_R: Union[float, np.ndarray],
    tol: float = 1e-6,
) -> Union[bool, np.ndarray]:
    r"""Whether engineering exponents satisfy the Kadomtsev constraint within a tolerance.

    $$|\alpha_K| \le \mathrm{tol}$$

    with $\alpha_K$ from :func:`kadomtsev_constraint_from_engineering_exponents`.

    Parameters
    ----------
    alpha_I : float or np.ndarray
        Exponent of the plasma current [-].
    alpha_B : float or np.ndarray
        Exponent of the toroidal field [-].
    alpha_P : float or np.ndarray
        Exponent of the heating power [-].
    alpha_n : float or np.ndarray
        Exponent of the density [-].
    alpha_R : float or np.ndarray
        Exponent of the major radius [-].
    tol : float, optional
        Absolute tolerance on $|\alpha_K|$; default 1e-6 [-].

    Returns
    -------
    bool or np.ndarray
        ``True`` where the constraint holds [bool].

    Limitations
    -----------
    The default tolerance is far tighter than any published fit satisfies
    (ITER89P misses by 0.15); pass a physically motivated ``tol``.
    :func:`verify_kadomtsev_constraint` evaluates a different expression from
    the dimensionless indices; the two are tracked in #351.

    References
    ----------
    .. [1] B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295.
    """
    tol_arr = _validate_positive("tol", tol)
    alpha_k = np.asarray(
        kadomtsev_constraint_from_engineering_exponents(
            alpha_I=alpha_I,
            alpha_B=alpha_B,
            alpha_P=alpha_P,
            alpha_n=alpha_n,
            alpha_R=alpha_R,
        ),
        dtype=float,
    )
    satisfied = np.abs(alpha_k) <= tol_arr
    if np.asarray(satisfied).ndim == 0:
        return bool(satisfied)
    return satisfied


# Paper-convention discoverability aliases
coulomb_logarithm = coulomb_logarithm_from_n_T
calc_rho_star = rho_star_from_M_T_B_R_epsilon
calc_beta_t = beta_t_from_n_T_B
calc_q_cyl = q_cyl_from_B_R_epsilon_kappa_I
calc_nu_star = nu_star_from_n_T_B_R_epsilon_kappa_I
calc_omega_i_tau_E = omega_i_tau_E_from_B_tau_E_M


# ------------------------------------------------------------------
# Confinement Time
# ------------------------------------------------------------------
def confinement_time_from_P_loss_W_th(P_loss: float, W_th: float) -> float:
    r"""Energy confinement time as stored energy over loss power.

    $$\tau_E = \frac{W_{th}}{P_{loss}}$$

    Parameters
    ----------
    P_loss : float
        Loss power [W].
    W_th : float
        Thermal stored energy [J].

    Returns
    -------
    float
        Energy confinement time [s].

    Convention
    ----------
    Thermal energy over *loss* power ($P_{heat} - dW/dt$, with or without
    radiation subtracted depending on the database); the ITER definition of
    $\tau_{E,th}$ needs $P_{loss}$ from :func:`loss_power_from_p_heat_dWdt_p_rad`.

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2, Sec. 3.
    """
    return W_th / P_loss

def confinement_time_from_engineering_parameters(
    I_p: float,
    B_t: float,
    P_loss: float,
    n_e: float,
    M: float,
    R: float,
    epsilon: float,
    kappa: float,
    scaling: str = "ITER89P",
    input_density_definition: str = "line_avg",
    line_to_volume_factor: Optional[float] = None,
) -> float:
    r"""Thermal energy confinement time from an engineering-parameter scaling law.

    $$\tau_{E,th} = C\prod_i x_i^{\alpha_i}$$

    with the product running only over the variables the selected scaling
    declares, in the engineering units MA, T, MW, $10^{19}$ m^-3, amu and m.

    Parameters
    ----------
    I_p : float
        Plasma current, converted to MA internally [A].
    B_t : float
        Toroidal field [T].
    P_loss : float
        Loss power, converted to MW internally [W].
    n_e : float
        Electron density, converted to $10^{19}$ m^-3 internally [m^-3].
    M : float
        Average ion mass [amu].
    R : float
        Major radius [m].
    epsilon : float
        Inverse aspect ratio [-].
    kappa : float
        Elongation [-].
    scaling : str, optional
        Scaling-law name; default ``"ITER89P"`` [str].
        One of ``"ITER89P"``, ``"H98y2"``, ``"NSTX2006H"``, ``"NSTX2006L"``,
        ``"Kurskiev2022"``.
    input_density_definition : str, optional
        What ``n_e`` is: ``"line_avg"`` (default) or ``"volume_avg"`` [str].
    line_to_volume_factor : float or None, optional
        Volume-to-line density ratio, default ``None`` [-].
        Used only when the input and the scaling's density definitions differ,
        and required then.

    Returns
    -------
    float
        Thermal energy confinement time [s].

    Raises
    ------
    ValueError
        Unknown scaling, non-positive input, or a density-definition mismatch
        without ``line_to_volume_factor``.

    Convention
    ----------
    Strict SI in; the SI-to-engineering conversions ($\times10^{-6}$ for
    current and power, $\times10^{-19}$ for density) happen inside, so
    pre-scaled inputs are wrong by orders of magnitude.  Every prefactor $C$ is
    tied to that unit convention: the NSTX fits were converted from the papers'
    SI-like form, and the density definition each scaling expects is declared
    in ``_SCALING_COEFS``.  Variables a scaling does not use are neither
    range-checked nor raised to any power.

    Validity
    --------
    Empirical fit.  Multi-machine regressions of the ITER L-mode (ITER89P [1]_)
    and ELMy H-mode (IPB98(y,2) [2]_) databases, the NSTX H- and L-mode fits of
    Kaye [3]_, and the spherical-tokamak multi-machine H-mode fit of Kurskiev
    [4]_; each is valid over its database's parameter range and the ST fits are
    the only ones that include low-aspect-ratio data.

    Limitations
    -----------
    Extrapolation to VEST (small size, low field) lies outside every database
    range except in part the ST fit; the Kurskiev regression's absorbed-power
    dependence is mapped onto ``P_loss`` as supplied.

    References
    ----------
    .. [1] P. N. Yushmanov et al., Nucl. Fusion 30 (1990) 1999 (ITER89P).
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2,
           Eq. (20) (IPB98(y,2)).
    .. [3] S. M. Kaye et al., Nucl. Fusion 46 (2006) 848, Table 2.
    .. [4] G. S. Kurskiev et al., Nucl. Fusion 62 (2022) 016011.
    """
    def _normalise_density_definition(label: str) -> str:
        mapping = {
            "line_avg": "line_avg",
            "line-average": "line_avg",
            "line_averaged": "line_avg",
            "line-averaged": "line_avg",
            "line": "line_avg",
            "volume_avg": "volume_avg",
            "volume-average": "volume_avg",
            "volume_averaged": "volume_avg",
            "volume-averaged": "volume_avg",
            "volume": "volume_avg",
        }
        key = str(label).strip().lower()
        if key not in mapping:
            raise ValueError(
                f"Invalid density definition '{label}'. "
                "Supported values: 'line_avg', 'volume_avg'."
            )
        return mapping[key]

    if scaling not in _SCALING_COEFS:
        raise ValueError(f"Unknown scaling '{scaling}'. Available: {list(_SCALING_COEFS.keys())}")

    coefs = _SCALING_COEFS[scaling]
    C = float(coefs["C"])
    if not np.isfinite(C) or C <= 0.0:
        raise ValueError(f"Scaling constant C must be finite and > 0 for '{scaling}'. Got {C!r}")

    # Backward compatibility: accept both the new nested `exponents` schema and
    # historical flat entries.
    if "exponents" in coefs:
        exponents = dict(coefs["exponents"])
    else:
        exponent_keys = ("Ip_MA", "Bt", "P_MW", "n_19", "Mi", "R", "epsilon", "kappa")
        exponents = {key: coefs[key] for key in exponent_keys if key in coefs}

    if len(exponents) == 0:
        raise ValueError(f"No exponents defined for scaling '{scaling}'.")

    input_density_def = _normalise_density_definition(input_density_definition)
    target_density_def = _normalise_density_definition(
        coefs.get("density_definition", "line_avg")
    )

    # Unit conversions: SI → scaling law units
    I_p_MA = I_p * 1e-6          # A → MA
    P_loss_MW = P_loss * 1e-6    # W → MW

    # Density conversion is explicit and only applied when required.
    n_e_input = _validate_positive("n_e", n_e)
    if input_density_def == target_density_def:
        n_e_target = n_e_input
    else:
        if line_to_volume_factor is None:
            raise ValueError(
                f"Scaling '{scaling}' expects '{target_density_def}' density, "
                f"but input_density_definition is '{input_density_def}'. "
                "Provide line_to_volume_factor explicitly to convert densities."
            )
        factor = _validate_positive("line_to_volume_factor", line_to_volume_factor)
        if input_density_def == "line_avg" and target_density_def == "volume_avg":
            n_e_target = n_e_input * factor
        elif input_density_def == "volume_avg" and target_density_def == "line_avg":
            n_e_target = n_e_input / factor
        else:
            raise ValueError(
                f"Unsupported density conversion: {input_density_def} -> {target_density_def}"
            )

    n_e_19 = n_e_target * 1e-19  # m^-3 → 10^19 m^-3

    variable_values = {
        "Ip_MA": I_p_MA,
        "Bt": B_t,
        "P_MW": P_loss_MW,
        "n_19": n_e_19,
        "Mi": M,
        "R": R,
        "epsilon": epsilon,
        "kappa": kappa,
    }

    result = C
    used_variables = []
    for variable_name, alpha in exponents.items():
        if variable_name not in variable_values:
            raise ValueError(
                f"Unsupported exponent variable '{variable_name}' in scaling '{scaling}'."
            )
        alpha_float = float(alpha)
        if not np.isfinite(alpha_float):
            raise ValueError(
                f"Non-finite exponent for variable '{variable_name}' in scaling '{scaling}': {alpha!r}"
            )
        value = _validate_positive(variable_name, variable_values[variable_name])
        result = result * value ** alpha_float
        used_variables.append(variable_name)

    # Handle complex numbers (should not happen with valid inputs)
    if np.iscomplexobj(result):
        result_complex = np.asarray(result)
        if np.any(result_complex.imag != 0):
            raise ValueError(
                f"Confinement time calculation resulted in complex number for scaling '{scaling}'. "
                f"Used variables: {used_variables}. "
                f"Inputs: I_p={I_p}, B_t={B_t}, P_loss={P_loss}, n_e={n_e}, M={M}, R={R}, "
                f"epsilon={epsilon}, kappa={kappa}"
            )
        result = np.real(result_complex)

    # Ensure result is finite
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(
            f"Invalid confinement time result: {result}. "
            f"Used variables: {used_variables}. "
            f"Inputs: I_p={I_p}, B_t={B_t}, P_loss={P_loss}, n_e={n_e}, M={M}, R={R}, "
            f"epsilon={epsilon}, kappa={kappa}, scaling={scaling}"
        )

    return float(result)


def confinement_factor_ITER89P(tau_E_exp: float, tau_E_ITER89P: float) -> float:

    r"""Confinement enhancement factor $H_{89}$ relative to ITER89P.

    $$H_{89} = \frac{\tau_{E,\mathrm{exp}}}{\tau_{E,\mathrm{ITER89P}}}$$

    Parameters
    ----------
    tau_E_exp : float
        Measured energy confinement time [s].
    tau_E_ITER89P : float
        ITER89P prediction for the same discharge [s].

    Returns
    -------
    float
        $H_{89}$; values above 1 beat the L-mode scaling [-].

    References
    ----------
    .. [1] P. N. Yushmanov et al., Nucl. Fusion 30 (1990) 1999.
    """

    return tau_E_exp / tau_E_ITER89P

def dimensionless_scaling_coeffs_from_engineering_scaling_coeffs(
    a_I: float,
    a_B: float,
    a_P: float,
    a_n: float,
    a_M: float,
    a_R: float,
    a_eps: float,
    a_kappa: float,
) -> Tuple[float, float, float, float, float]:
    r"""Dimensionless scaling indices $(\mu_\rho, \mu_\beta, \mu_\nu)$ from engineering exponents.

    $$\Omega_i\tau_E \propto \rho_*^{\mu_\rho}\,\beta^{\mu_\beta}\,\nu_*^{\mu_\nu}$$

    with, for $\alpha_L = \alpha_R + \alpha_I$, $\alpha_B^* = \alpha_B + \alpha_I$
    and $D = 1 + \alpha_P$,

    $$\mu_\rho = \frac{3\alpha_L + \alpha_B^* + \alpha_n - 2\alpha_P - 5}{D}, \quad
      \mu_\beta = \frac{-\alpha_L - 2\alpha_n - \alpha_B^* + 3\alpha_P + 3}{D}, \quad
      \mu_\nu = \frac{\alpha_L + 3\alpha_n + \alpha_B^* - 2\alpha_P - 4}{2D}$$

    Parameters
    ----------
    a_I : float
        Exponent of the plasma current [-].
    a_B : float
        Exponent of the toroidal field [-].
    a_P : float
        Exponent of the heating power [-].
    a_n : float
        Exponent of the density [-].
    a_M : float
        Exponent of the ion mass, passed through [-].
    a_R : float
        Exponent of the major radius [-].
    a_eps : float
        Exponent of the inverse aspect ratio, unused [-].
    a_kappa : float
        Exponent of the elongation, passed through [-].

    Returns
    -------
    mu_rho : float
        Gyroradius index; $-3$ is gyro-Bohm, $-2$ Bohm [-].
    mu_beta : float
        Beta index [-].
    mu_nu : float
        Collisionality index [-].
    mu_M : float
        Mass index, equal to ``a_M`` [-].
    mu_kappa : float
        Elongation index, equal to ``a_kappa`` [-].

    Raises
    ------
    ValueError
        When $1 + \alpha_P$ vanishes, which is the exact-power-degradation case
        $\alpha_P = -1$: every index divides by it, so the transformation has no
        value there rather than a special one [-].

    Assumptions
    -----------
    Constant safety factor, $I_p \propto a^2B/R$, so current is absorbed into
    the size and field indices; temperature eliminated through $P = W/\tau_E$.

    Limitations
    -----------
    ``a_eps`` is accepted and unused; ``a_M`` and ``a_kappa`` are returned
    unchanged, so the transformation is a no-op for those two axes.

    References
    ----------
    .. [1] T. C. Luce, C. C. Petty and J. G. Cordey, Plasma Phys. Control.
           Fusion 50 (2008) 043001, Sec. 3 (engineering to dimensionless
           exponent transformation).
    .. [2] B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295.
    """
    
    # 1. Basis Transformation (Consolidating to fundamental dimensions: L and B)
    # Using the relation I_p ∝ R * eps^2 * B (since a = R * eps)
    a_L = a_R + a_I        # Combined length (L) scaling index
    a_B_star = a_B + a_I   # Combined magnetic field (B) scaling index
    # Note: a_n (density) and a_P (power) remain as primary engineering inputs.

    # 2. Derive Dimensionless Indices (mu) via Power Balance
    # Normalized confinement time follows: Ω_c * τ_E ∝ (ρ*)^μ_ρ * β^μ_β * (ν*)^μ_ν
    # The derivation eliminates Temperature (T) using P = W / τ_E.
    
    denom = 1 + a_P
    if abs(denom) < 1e-9:
        # Returning None here made every caller fail on the unpack instead, with
        # a TypeError naming the call site rather than the degenerate exponent.
        raise ValueError(
            "a_P = -1 leaves 1 + a_P = 0, and every dimensionless index divides "
            f"by it; the transformation is undefined there. Got a_P={a_P!r}."
        )

    # Mapping based on Gyro-kinetic transport theory and Kadomtsev's similarity principles
    # mu_rho: Characterizes size scaling (e.g., -3 for Gyro-Bohm, -2 for Bohm)
    mu_rho = (3 * a_L + a_B_star + a_n - 2 * a_P - 5) / denom
    
    # mu_beta: Characterizes plasma pressure scaling
    mu_beta = (-a_L - 2 * a_n - a_B_star + 3 * a_P + 3) / denom
    
    # mu_nu: Characterizes collisionality scaling
    mu_nu = (a_L + 3 * a_n + a_B_star - 2 * a_P - 4) / (2 * denom)

    # Deliberately unrounded: three decimals is a presentation choice, and a
    # kernel that bakes one in cannot be used for anything needing more.
    return mu_rho, mu_beta, mu_nu, a_M, a_kappa

def verify_kadomtsev_constraint(mu_rho, mu_beta, mu_nu, a_P):
    r"""Reconstruct the Kadomtsev constraint value from dimensionless indices.

    $$x = 5 + \mu_\rho(1 + \alpha_P) - \frac{3}{2}\left(\mu_\rho + 2\mu_\beta - 4\mu_\nu - 2\right)$$

    which should return 5 when the dimensionless mapping of
    :func:`dimensionless_scaling_coeffs_from_engineering_scaling_coeffs`
    preserves the identity $\alpha_L + 2\alpha_n + \alpha_B^* - 3\alpha_P = 5$.

    Parameters
    ----------
    mu_rho : float
        Gyroradius index [-].
    mu_beta : float
        Beta index [-].
    mu_nu : float
        Collisionality index [-].
    a_P : float
        Engineering exponent of the heating power [-].

    Returns
    -------
    float
        Reconstructed constraint value, 5 for a consistent mapping [-].

    Limitations
    -----------
    Evaluates a different expression from
    :func:`check_kadomtsev_constraint` (which uses the engineering exponents
    directly) and the two need not agree; tracked in #351.

    References
    ----------
    .. [1] B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295.
    .. [2] T. C. Luce, C. C. Petty and J. G. Cordey, Plasma Phys. Control.
           Fusion 50 (2008) 043001.
    """
     
    # Reconstructing the constraint value x from the dimensionless indices
    # For a purely physical model, calculated_x should converge to 5.0.
    x_val = 5.0 + (mu_rho * (1 + a_P) - (3 * (mu_rho + 2 * mu_beta - 4 * mu_nu - 2) / 2))
    
    return x_val


#: Derived rather than listed, so a function added here cannot silently
#: leave the package surface the way seven did in #711.  The virial names
#: re-bound above belong to `.virial` and are excluded by construction.
__all__ = public_names(globals())
