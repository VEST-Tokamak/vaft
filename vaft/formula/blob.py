r"""SOL blobs and filaments: reference scales, closure-limited velocities and the regime scalings.

A blob is a localized positive density (pressure) perturbation in the
scrape-off layer; a hole is a negative one; the filament is the field-aligned
structure of which a blob is the perpendicular cross-section. Curvature and
$\nabla B$ polarize it into a dipole, and the $E\times B$ drift moves it
radially outward on the low-field side. How fast depends on how the
polarization current closes: through the sheaths at the ends of the field
line (sheath-connected, $v \propto \delta^{-2}$), across the field by ion
polarization (inertial, $v \propto \delta^{1/2}$), or, with resistivity and
X-point fanning, in between.

Every relation here follows one convention, D'Ippolito, Myra and Zweben
(2011): a Gaussian blob $n \propto \exp(-r^2/2\delta^2)$ of radius $\delta$,
coordinates $x$ radial, $y$ binormal, $z$ (``s``) parallel. The $O(1)$
prefactors of other papers (Krasheninnikov 2001, Theiler et al. 2011) belong
to their own size definitions and are not interchangeable with these.

Notation
--------
c_s      : ion sound speed sqrt(T_e/m_i)                          [m/s]
rho_s    : ion sound gyroradius c_s/Omega_i                         [m]
delta    : blob radius (Gaussian e^{-r^2/2 delta^2})                 [m]
L_par    : parallel connection length, sheath to sheath              [m]
R        : major radius (radius of curvature)                        [m]
delta_*  : reference size rho_s^{4/5} L_par^{2/5} / R^{1/5}           [m]
v_*      : reference velocity c_s (delta_*/R)^{1/2}                  [m/s]
Lambda   : collisionality nu_ei L_par / (Omega_e rho_s)             [-]
epsilon_x: X-point fanning parameter, << 1 in diverted geometry     [-]

Conventions
-----------
Velocities are radial, positive outward along $-\nabla B$; a hole moves the
other way. $\hat\delta = \delta/\delta_*$ and $\hat v = v/v_*$.

References
----------
.. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
       (2011) 060501.
.. [2] S. I. Krasheninnikov, Phys. Lett. A 283 (2001) 368.
.. [3] J. R. Myra et al., Phys. Plasmas 13 (2006) 092509.
.. [4] C. Theiler et al., Phys. Plasmas 18 (2011) 055901.
"""

from typing import NamedTuple

import numpy as np

__all__ = [
    "BlobRegimeVelocities",
    "blob_reference_size",
    "blob_reference_velocity",
    "blob_collisionality",
    "sheath_connected_blob_velocity",
    "inertial_blob_velocity",
    "interpolated_blob_velocity",
    "blob_regime_velocities",
]


class BlobRegimeVelocities(NamedTuple):
    """Normalized radial velocities $\\hat v$ of the four regimes of the two-region model [-]."""

    resistive_ballooning: np.ndarray
    resistive_x_point: np.ndarray
    sheath_connected: np.ndarray
    connected_ideal_interchange: np.ndarray


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def _positive(value, name):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return arr


def _amplitude(value):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0) or np.any(arr > 1.0):
        raise ValueError("relative_amplitude (delta n / n) must lie in (0, 1]")
    return arr


def blob_reference_size(rho_s, L_par, R):
    r"""The blob size at which sheath and polarization closure balance.

    $$\delta_* = \frac{\rho_s^{4/5}L_\parallel^{2/5}}{R^{1/5}}$$

    Parameters
    ----------
    rho_s : float or np.ndarray
        Ion sound gyroradius, positive [m].
    L_par : float or np.ndarray
        Parallel connection length, positive [m].
    R : float or np.ndarray
        Major radius, positive [m].

    Returns
    -------
    float or np.ndarray
        $\delta_*$ [m].

    Raises
    ------
    ValueError
        A non-positive input.

    Convention
    ----------
    D'Ippolito, Myra and Zweben (2011) Eq. (8), identical to Myra et al.
    (2006) Eq. (2)'s $a_*$. Theiler et al.'s $a^*$ is $4^{1/5}$ times this.

    Physical interpretation
    -----------------------
    Smaller blobs are limited by ion polarization (inertial), larger ones by
    the sheaths; blobs near $\delta_*$ are the most coherent (Kelvin--Helmholtz
    below, Rayleigh--Taylor break-up above), so it is also the characteristic
    observed size.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, Eq. (8).
    """
    rho_s, L_par, R = _positive(rho_s, "rho_s"), _positive(L_par, "L_par"), _positive(R, "R")
    return _out(rho_s**0.8 * L_par**0.4 / R**0.2)


def blob_reference_velocity(c_s, delta_star, R):
    r"""The blob velocity scale at the reference size.

    $$v_* = c_s\left(\frac{\delta_*}{R}\right)^{1/2}$$

    Parameters
    ----------
    c_s : float or np.ndarray
        Ion sound speed, positive [m/s].
    delta_star : float or np.ndarray
        Reference size (``blob_reference_size``), positive [m].
    R : float or np.ndarray
        Major radius, positive [m].

    Returns
    -------
    float or np.ndarray
        $v_*$ [m/s].

    Raises
    ------
    ValueError
        A non-positive input.

    Convention
    ----------
    D'Ippolito, Myra and Zweben (2011) Eq. (8); Myra et al. (2006) Eq. (3).
    The sheath and inertial limits meet at $\hat\delta = 1$, $\hat v = 1$.

    Physical interpretation
    -----------------------
    A few per cent of $c_s$ in tokamak SOLs (about 2 km/s for NSTX and
    C-Mod parameters, Myra et al. 2006), the scale of measured blob speeds.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, Eq. (8).
    """
    c_s, delta_star, R = _positive(c_s, "c_s"), _positive(delta_star, "delta_star"), _positive(R, "R")
    return _out(c_s * np.sqrt(delta_star / R))


def blob_collisionality(nu_ei, L_par, Omega_e, rho_s):
    r"""The collisionality that decides whether a filament stays connected to the sheaths.

    $$\Lambda = \frac{\nu_{ei}L_\parallel}{\Omega_e\rho_s}$$

    Parameters
    ----------
    nu_ei : float or np.ndarray
        Electron--ion collision frequency, non-negative [1/s].
    L_par : float or np.ndarray
        Parallel connection length (in the X-point region for the
        two-region model), positive [m].
    Omega_e : float or np.ndarray
        Electron gyrofrequency, magnitude, positive [rad/s].
    rho_s : float or np.ndarray
        Ion sound gyroradius, positive [m].

    Returns
    -------
    float or np.ndarray
        $\Lambda$ [-].

    Raises
    ------
    ValueError
        A negative collision frequency or a non-positive length or frequency.

    Convention
    ----------
    Myra et al. (2006) Eq. (1); equivalently $(m_e/m_i)^{1/2}L_\parallel/
    \lambda_{ei}$ (D'Ippolito, Myra and Zweben 2011, p. 060501-23). A SOL
    collisionality in its own right, not a core $\nu_*$.

    Physical interpretation
    -----------------------
    Above one, parallel resistivity cuts the filament off from the sheath
    and the blob moves faster (resistive regimes); below, it stays
    sheath-connected.

    References
    ----------
    .. [1] J. R. Myra et al., Phys. Plasmas 13 (2006) 092509, Eq. (1).
    """
    nu = np.asarray(nu_ei, dtype=float)
    if np.any(~np.isfinite(nu)) or np.any(nu < 0.0):
        raise ValueError("nu_ei must be non-negative and finite")
    L_par, Omega_e, rho_s = _positive(L_par, "L_par"), _positive(Omega_e, "Omega_e"), _positive(rho_s, "rho_s")
    return _out(nu * L_par / (Omega_e * rho_s))


def sheath_connected_blob_velocity(c_s, rho_s, delta, L_par, R):
    r"""Radial velocity of a blob whose polarization current closes through the sheaths.

    $$v_x = c_s\,\frac{L_\parallel}{R}\left(\frac{\rho_s}{\delta}\right)^2$$

    Parameters
    ----------
    c_s : float or np.ndarray
        Ion sound speed, positive [m/s].
    rho_s : float or np.ndarray
        Ion sound gyroradius, positive [m].
    delta : float or np.ndarray
        Blob radius, Gaussian $e^{-r^2/2\delta^2}$, positive [m].
    L_par : float or np.ndarray
        Sheath-to-sheath parallel connection length, positive [m].
    R : float or np.ndarray
        Radius of curvature, positive [m].

    Returns
    -------
    float or np.ndarray
        $v_x$, outward [m/s].

    Raises
    ------
    ValueError
        A non-positive input.

    Convention
    ----------
    D'Ippolito, Myra and Zweben (2011) Eq. (3): an isolated blob in vacuum
    ($\delta n/n = 1$), linearized sheath closure $J_\parallel = ne^2c_s\Phi/T_e$,
    an exact nonlinear solution that convects without distortion.
    Krasheninnikov's (2001) Eq. (6) has the same scaling with his
    $e^{-y^2/\delta^2}$ width and $n_b/n_t$; Theiler et al. have a factor 2
    with a HWHM size.

    Physical interpretation
    -----------------------
    The sheath is the least resistive path, so the dipole potential and the
    $E\times B$ speed are smallest: $v \propto \delta^{-2}$, large blobs slow.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, Eq. (3).
    .. [2] S. I. Krasheninnikov, Phys. Lett. A 283 (2001) 368, Eq. (6).
    """
    c_s, rho_s, delta = _positive(c_s, "c_s"), _positive(rho_s, "rho_s"), _positive(delta, "delta")
    L_par, R = _positive(L_par, "L_par"), _positive(R, "R")
    return _out(c_s * L_par / R * (rho_s / delta) ** 2)


def inertial_blob_velocity(c_s, delta, R, *, relative_amplitude=1.0):
    r"""Radial velocity of a blob whose polarization current closes across the field (inertia).

    $$v_x = c_s\left(\frac{\delta n}{n}\right)^{1/2}\left(\frac{\delta}{R}\right)^{1/2}$$

    Parameters
    ----------
    c_s : float or np.ndarray
        Ion sound speed, positive [m/s].
    delta : float or np.ndarray
        Blob radius, positive [m].
    R : float or np.ndarray
        Radius of curvature, positive [m].
    relative_amplitude : float or np.ndarray
        Blob amplitude over the total density, $\delta n/n$, in (0, 1] [-].

    Returns
    -------
    float or np.ndarray
        $v_x$, outward [m/s].

    Raises
    ------
    ValueError
        A non-positive input or an amplitude outside (0, 1].

    Convention
    ----------
    D'Ippolito, Myra and Zweben (2011) p. 060501-25: $\hat v = (\delta n/n)^{1/2}
    \hat\delta^{1/2}$, here in dimensional form ($\hat v v_*$). Myra et al.
    (2006) Eq. (11) is the same with $f_b$ for $\delta n/n$; Theiler et al.'s
    $\sqrt{2a/R}\,c_s$ has their HWHM size.

    Physical interpretation
    -----------------------
    The resistive-ballooning (fully disconnected) limit: curvature drive
    against ion inertia at the midplane, the fastest a blob can go;
    $v \propto \delta^{1/2}$, large blobs fast.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, p. 060501-25.
    .. [2] J. R. Myra et al., Phys. Plasmas 13 (2006) 092509, Eq. (11).
    """
    c_s, delta, R = _positive(c_s, "c_s"), _positive(delta, "delta"), _positive(R, "R")
    return _out(c_s * np.sqrt(_amplitude(relative_amplitude) * delta / R))


def interpolated_blob_velocity(delta_hat, *, relative_amplitude=1.0):
    r"""Normalized blob velocity bridging the inertial and sheath-connected limits.

    $$\hat v = \frac{(\delta n/n)\,\hat\delta^{1/2}}{(\delta n/n)^{1/2} + \hat\delta^{5/2}}$$

    Parameters
    ----------
    delta_hat : float or np.ndarray
        Blob size over the reference size, $\delta/\delta_*$, positive [-].
    relative_amplitude : float or np.ndarray
        $\delta n/n$, in (0, 1] [-].

    Returns
    -------
    float or np.ndarray
        $\hat v = v_x/v_*$ [-].

    Raises
    ------
    ValueError
        A non-positive size or an amplitude outside (0, 1].

    Convention
    ----------
    D'Ippolito, Myra and Zweben (2011) Eq. (9), from $1/\hat v = 1/\hat v_1
    + 1/\hat v_2$ of the two limits: $(\delta n/n)^{1/2}\hat\delta^{1/2}$
    for small blobs and $(\delta n/n)/\hat\delta^2$ for large ones. An
    interpolation valid in the limits only.

    Physical interpretation
    -----------------------
    The measured blob speeds of nine tokamaks lie between the two limits
    (their Fig. 27), peaking near $\hat\delta \approx 1$.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, Eq. (9).
    """
    d = _positive(delta_hat, "delta_hat")
    f = _amplitude(relative_amplitude)
    return _out(f * np.sqrt(d) / (np.sqrt(f) + d**2.5))


def blob_regime_velocities(delta_hat, Lambda, epsilon_x):
    r"""The normalized velocity of each regime of the two-region (midplane plus X-point) model.

    $$\hat v_\mathrm{RB} = \hat\delta^{1/2},\quad \hat v_\mathrm{RX} = \frac{\Lambda}{\hat\delta^2},\quad
      \hat v_{C_s} = \frac{1}{\hat\delta^2},\quad \hat v_{C_i} = \varepsilon_x\hat\delta^{1/2}$$

    Parameters
    ----------
    delta_hat : float or np.ndarray
        $\delta/\delta_*$, positive [-].
    Lambda : float or np.ndarray
        Collisionality (``blob_collisionality``), positive [-].
    epsilon_x : float or np.ndarray
        X-point fanning parameter, in (0, 1) [-].

    Returns
    -------
    BlobRegimeVelocities
        $\hat v$ of the resistive-ballooning, resistive X-point,
        sheath-connected and connected ideal-interchange regimes [-].

    Raises
    ------
    ValueError
        A non-positive size or collisionality, or $\varepsilon_x$ outside (0, 1).

    Convention
    ----------
    D'Ippolito, Myra and Zweben (2011) Fig. 23, after Myra, Russell and
    D'Ippolito (2006), with $\delta n/n = 1$. The regime boundaries are where
    neighbouring scalings meet: $\Lambda = \Theta$ (RB|RX),
    $\Lambda = \varepsilon_x\Theta$ (RX|$C_i$), $\Lambda = 1$ (RX|$C_s$) and
    $\Theta = 1/\varepsilon_x$ ($C_i$|$C_s$), with $\Theta = \hat\delta^{5/2}$.

    Physical interpretation
    -----------------------
    Collisionality disconnects the filament from the sheath (faster, RX then
    RB); X-point fanning makes cross-field closure easy near the target
    ($C_i$). In the RX regime transport rises with collisionality.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, Fig. 23 and pp. 060501-23 to -27.
    """
    d = _positive(delta_hat, "delta_hat")
    lam = _positive(Lambda, "Lambda")
    eps = np.asarray(epsilon_x, dtype=float)
    if np.any(~np.isfinite(eps)) or np.any(eps <= 0.0) or np.any(eps >= 1.0):
        raise ValueError("epsilon_x must lie in (0, 1)")
    d, lam, eps = np.broadcast_arrays(d, lam, eps)
    return BlobRegimeVelocities(_out(np.sqrt(d)), _out(lam / d**2), _out(1.0 / d**2), _out(eps * np.sqrt(d)))
