"""
Vertical displacement events: local motion, wall response and halo-current descriptors.

Generic, model-neutral reference quantities for a VDE -- how fast the plasma
moves, how fast a conducting wall lets it, and how much and how unevenly the
current returns through the wall as halo current. None of them is a VDE
model: the hot-VDE edge-current loss, the cold-VDE equilibrium-branch loss and
analytic halo-current models are named literature models that belong in their
own functions once one is selected (#1042). The disruption chain these
couple to -- thermal and current quench, runaways -- is
``vaft.formula.disruption``; the plain circuit time $L/R$ of a wall element is
``lr_time_from_L_R``.

Notation
--------
t       : time                                            [s]
Z       : vertical position of the current centroid       [m]
Z0      : reference vertical position                      [m]
sigma   : wall electrical conductivity                     [S/m]
d       : wall thickness                                   [m]
b       : wall minor radius                                [m]
m       : poloidal mode number of the wall eddy pattern    [-]
I_halo  : halo current                                     [A]
I_p0    : pre-disruption plasma current                    [A]

Conventions
-----------
Growth rates are local: $\\gamma = d\\ln|Z - Z_0|/dt$ on whatever interval the
caller passes, with no claim that the whole event is exponential. Halo
fractions use the peak halo current over the pre-disruption plasma current,
the ITER Physics Basis convention.

References
----------
.. [1] T. C. Hender et al., Nucl. Fusion 47 (2007) S128 (ITER Physics Basis,
       Chapter 3: MHD stability, operational limits and disruption management).
.. [2] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
       the resistive-wall chapter.
"""

import numpy as np

from .constants import MU0

__all__ = [
    "vertical_velocity",
    "vde_growth_rate",
    "thin_wall_time",
    "wall_mode_decay_time",
    "halo_current_fraction",
    "toroidal_peaking_factor",
]


def _series(t, y, name):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or t.shape != y.shape or t.size < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"t must be increasing and the same length as {name} (at least 3 samples)")
    return t, y


def vertical_velocity(t, Z):
    r"""Vertical velocity of the current centroid, $dZ/dt$.

    $$v_Z = \frac{dZ}{dt}$$

    Parameters
    ----------
    t : np.ndarray
        Time, increasing [s].
    Z : np.ndarray
        Vertical centroid position on ``t`` [m].

    Returns
    -------
    np.ndarray
        $v_Z$ on ``t`` [m/s].

    Raises
    ------
    ValueError
        ``t`` is not increasing or differs in length from ``Z``.

    Numerical notes
    ---------------
    Second-order central differences (``numpy.gradient``), one-sided at the
    ends; smooth or window a noisy $Z$ first.

    References
    ----------
    .. [1] T. C. Hender et al., Nucl. Fusion 47 (2007) S128.
    """
    t, Z = _series(t, Z, "Z")
    return np.gradient(Z, t)


def vde_growth_rate(t, Z, Z0):
    r"""Local growth rate of a vertical displacement, $d\ln|Z - Z_0|/dt$.

    $$\gamma_\mathrm{VDE}(t) = \frac{d}{dt}\ln|\Delta Z|,\qquad \Delta Z = Z - Z_0$$

    Parameters
    ----------
    t : np.ndarray
        Time, increasing [s].
    Z : np.ndarray
        Vertical centroid position on ``t`` [m].
    Z0 : float
        Reference position the displacement is measured from [m].

    Returns
    -------
    np.ndarray
        $\gamma_\mathrm{VDE}$ on ``t``; NaN where $\Delta Z = 0$ and at its
        neighbouring samples, which the central difference reaches [1/s].

    Raises
    ------
    ValueError
        ``t`` is not increasing or differs in length from ``Z``.

    Convention
    ----------
    A local characterisation: constant only over an interval where the motion
    really is exponential ($\Delta Z = \Delta Z_0e^{\gamma t}$). A VDE is in
    general not one exponential -- wall-limited growth, then an accelerating
    cold-VDE drift as the current decays, then wall contact.

    Physical interpretation
    -----------------------
    With a conducting wall the growth is slowed from the Alfvenic ideal rate
    to a rate set by the wall eddy time (``wall_mode_decay_time``, $m = 1$)
    divided by the stability margin; $1/\gamma$ read off a measured
    displacement says which regime an interval is in.

    References
    ----------
    .. [1] T. C. Hender et al., Nucl. Fusion 47 (2007) S128.
    """
    t, Z = _series(t, Z, "Z")
    dZ = np.abs(Z - float(Z0))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dz = np.where(dZ > 0.0, np.log(np.where(dZ > 0.0, dZ, 1.0)), np.nan)
    return np.gradient(log_dz, t)


def thin_wall_time(sigma, d, b):
    r"""Resistive time of a thin cylindrical wall, $\tau_w = \mu_0\sigma db$.

    $$\tau_w = \mu_0\,\sigma\,d\,b$$

    Parameters
    ----------
    sigma : float or np.ndarray
        Wall conductivity, positive [S/m].
    d : float or np.ndarray
        Wall thickness, positive and much less than ``b`` [m].
    b : float or np.ndarray
        Wall minor radius, positive [m].

    Returns
    -------
    float or np.ndarray
        $\tau_w$ [s].

    Raises
    ------
    ValueError
        An argument is not positive.

    Convention
    ----------
    The thin-wall time of the resistive-wall-mode literature (Freidberg). The
    eddy pattern of a poloidal harmonic $m$ decays in $\tau_w/(2m)$
    (``wall_mode_decay_time``); the vertical ($m = 1$) displacement sees
    $\tau_w/2$ as its eddy time -- its growth time is that divided by the
    stability margin, not equal to it. Some resistive-wall-mode papers call
    $\mu_0\sigma db/2$ "$\tau_w$": the definition here is explicit. A real vessel with ports, gaps and several shells has a
    spectrum of times; its circuit elements' $L/R$ are ``lr_time_from_L_R``.

    Assumptions
    -----------
    Axisymmetric, continuous cylindrical shell, $d \ll b$ (thin wall),
    uniform conductivity.

    References
    ----------
    .. [1] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014), the
           resistive-wall chapter.
    """
    arrays = [np.asarray(v, dtype=float) for v in (sigma, d, b)]
    if any(np.any(~np.isfinite(a)) or np.any(a <= 0.0) for a in arrays):
        raise ValueError("sigma, d and b must be positive and finite")
    out = MU0 * arrays[0] * arrays[1] * arrays[2]
    return float(out) if np.ndim(out) == 0 else out


def wall_mode_decay_time(tau_w, m):
    r"""Decay time of a thin wall's poloidal-harmonic-$m$ eddy current.

    $$\tau_m = \frac{\tau_w}{2m}$$

    Parameters
    ----------
    tau_w : float or np.ndarray
        Thin-wall time (``thin_wall_time``), positive [s].
    m : int
        Poloidal harmonic, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_m$ [s].

    Raises
    ------
    ValueError
        ``tau_w`` is not positive or ``m`` is not a positive integer.

    Physical interpretation
    -----------------------
    Higher harmonics have shorter current paths per unit flux and decay
    faster; a vertical displacement is the $m = 1$ pattern.

    References
    ----------
    .. [1] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014), the
           resistive-wall chapter.
    """
    if isinstance(m, bool) or int(m) != m or int(m) < 1:
        raise ValueError(f"m must be a positive integer, not {m!r}")
    tau_w = np.asarray(tau_w, dtype=float)
    if np.any(tau_w <= 0.0):
        raise ValueError("tau_w must be positive")
    out = tau_w / (2.0 * int(m))
    return float(out) if np.ndim(out) == 0 else out


def halo_current_fraction(I_halo_peak, I_p0):
    r"""Halo-current fraction: peak poloidal halo current over the pre-disruption plasma current.

    $$f_\mathrm{halo} = \frac{I_\mathrm{halo}^\mathrm{peak}}{I_{p,0}}$$

    Parameters
    ----------
    I_halo_peak : float or np.ndarray
        Peak (toroidally integrated, poloidal) halo current, non-negative [A].
    I_p0 : float
        Plasma current before the disruption, non-zero [A].

    Returns
    -------
    float or np.ndarray
        $f_\mathrm{halo}$ [-].

    Raises
    ------
    ValueError
        ``I_halo_peak`` is negative or ``I_p0`` is zero.

    Convention
    ----------
    Magnitudes; the ITER Physics Basis convention. A characterisation metric,
    not a halo-current model: it says nothing of the current path, the
    toroidal asymmetry (``toroidal_peaking_factor``) or the forces.

    References
    ----------
    .. [1] T. C. Hender et al., Nucl. Fusion 47 (2007) S128.
    """
    I_halo_peak = np.asarray(I_halo_peak, dtype=float)
    if np.any(I_halo_peak < 0.0):
        raise ValueError("I_halo_peak must be non-negative (a magnitude)")
    if np.ndim(I_p0) != 0:
        raise ValueError("I_p0 must be a scalar current")
    I_p0 = float(I_p0)
    if I_p0 == 0.0 or not np.isfinite(I_p0):
        raise ValueError("I_p0 must be finite and non-zero")
    if np.any(~np.isfinite(I_halo_peak)):
        raise ValueError("I_halo_peak must be finite")
    out = I_halo_peak / abs(I_p0)
    return float(out) if np.ndim(out) == 0 else out


def toroidal_peaking_factor(j_halo_phi):
    r"""Toroidal peaking factor of the halo current: maximum over the toroidal mean.

    $$\mathrm{TPF} = \frac{\max_\phi j_\mathrm{halo}}{\langle j_\mathrm{halo}\rangle_\phi}$$

    Parameters
    ----------
    j_halo_phi : np.ndarray
        Halo current (or current density) at toroidal positions spaced
        uniformly around the machine, last axis toroidal, non-negative [A or A/m^2].

    Returns
    -------
    float or np.ndarray
        TPF, at least one; one for an axisymmetric halo [-].

    Raises
    ------
    ValueError
        Fewer than two toroidal samples, a negative sample, or a zero mean.

    Convention
    ----------
    The samples must be uniform in $\phi$ for the mean to be the toroidal
    average; with a few discrete sensors TPF is a lower bound on the true
    peak. The product $f_\mathrm{halo}\cdot\mathrm{TPF}$ is the usual
    design-load figure.

    References
    ----------
    .. [1] T. C. Hender et al., Nucl. Fusion 47 (2007) S128.
    """
    j = np.asarray(j_halo_phi, dtype=float)
    if j.shape[-1:] == () or j.shape[-1] < 2:
        raise ValueError("need at least two toroidal samples on the last axis")
    if np.any(j < 0.0):
        raise ValueError("halo samples must be non-negative (magnitudes)")
    mean = j.mean(axis=-1)
    if np.any(mean == 0.0):
        raise ValueError("the toroidal mean is zero: no halo current")
    out = j.max(axis=-1) / mean
    return float(out) if np.ndim(out) == 0 else out
