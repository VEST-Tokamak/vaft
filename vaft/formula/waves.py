r"""Cold-plasma waves: characteristic frequencies, Stix parameters, refractive indices, cutoffs and resonances.

The canonical cold-plasma dielectric model and what follows from it
directly: the Stix parameters $R, L, S, D, P$, the Appleton--Hartree-type
roots $n^2$ of the oblique dispersion relation $An^4 - Bn^2 + C = 0$, the
perpendicular O and X modes, the CMA coordinates $(X, Y)$, and a sign
classification of $n^2$. Warm-plasma, kinetic and damping effects, ray
tracing and full-wave solutions are out of scope; they build on this layer.

Notation
--------
omega     : wave angular frequency, positive                  [rad/s]
omega_p   : plasma frequency of a species                     [rad/s]
Omega     : signed cyclotron frequency q B / m                 [rad/s]
n, q, m   : species density, signed charge, mass              [m^-3, C, kg]
B         : magnetic-field magnitude                          [T]
R, L, S, D, P : Stix parameters                               [-]
theta     : angle between k and B                             [rad]
n2        : refractive index squared, c^2 k^2 / omega^2        [-]
X, Y      : omega_pe^2 / omega^2 and |Omega_e| / omega         [-]

Conventions
-----------
Cyclotron frequencies are *signed* (``vaft.formula.particle.gyrofrequency``):
$\Omega_e < 0$. Stix's $R$ and $L$ are then the right- and left-hand
circularly polarised waves with respect to $\mathbf B_0 \parallel \hat z$,
and $R$ has the electron cyclotron resonance. The CMA coordinate $Y$ is the
positive magnitude $|\Omega_e|/\omega$. Species arrays carry the species
along axis 0; frequencies and field broadcast against the rest.

References
----------
.. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Ch. 1--2.
.. [2] D. G. Swanson, *Plasma Waves*, 2nd ed., IOP (2003), Ch. 2.
"""

from typing import NamedTuple

import numpy as np

from .constants import EPS0, ME, QE
from .particle import gyrofrequency

__all__ = [
    "StixParameters",
    "plasma_frequency",
    "stix_parameters",
    "cold_plasma_refractive_index_squared",
    "perpendicular_refractive_index_squared",
    "cma_coordinates",
    "propagation_regime",
]


class StixParameters(NamedTuple):
    """The five Stix parameters of the cold-plasma dielectric tensor [-]."""

    R: np.ndarray
    L: np.ndarray
    S: np.ndarray
    D: np.ndarray
    P: np.ndarray


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def plasma_frequency(n, q, m):
    r"""Plasma frequency of one species.

    $$\omega_{ps} = \sqrt{\frac{n_sq_s^2}{\epsilon_0m_s}}$$

    Parameters
    ----------
    n : float or np.ndarray
        Density, non-negative [m^-3].
    q : float or np.ndarray
        Charge, either sign [C].
    m : float or np.ndarray
        Mass, positive [kg].

    Returns
    -------
    float or np.ndarray
        $\omega_{ps}$, non-negative [rad/s].

    Raises
    ------
    ValueError
        A negative density, a non-positive mass, or a non-finite input.

    Convention
    ----------
    An angular frequency: divide by $2\pi$ for hertz ($f_{pe} \approx
    8.98\sqrt{n_e}$ Hz with $n_e$ in m$^{-3}$). The charge enters squared,
    so its sign does not matter here; it does in ``stix_parameters``.

    Physical interpretation
    -----------------------
    The frequency at which the species oscillates about a neutralising
    background when displaced; waves below $\omega_{pe}$ without a magnetic
    field cannot propagate.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-2.
    """
    n = np.asarray(n, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=float)
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(q)) and np.all(np.isfinite(m))):
        raise ValueError("n, q and m must be finite")
    if np.any(n < 0.0) or np.any(m <= 0.0):
        raise ValueError("n must be non-negative and m positive")
    return _out(np.sqrt(n * q**2 / (EPS0 * m)))


def stix_parameters(omega, n, q, m, B):
    r"""Stix parameters of a cold, multi-species, magnetised plasma.

    $$R = 1 - \sum_s\frac{\omega_{ps}^2}{\omega(\omega + \Omega_s)},\quad
      L = 1 - \sum_s\frac{\omega_{ps}^2}{\omega(\omega - \Omega_s)},\quad
      P = 1 - \sum_s\frac{\omega_{ps}^2}{\omega^2},\quad
      S = \frac{R + L}{2},\quad D = \frac{R - L}{2}$$

    Parameters
    ----------
    omega : float or np.ndarray
        Wave angular frequency, positive, broadcast against the trailing axes
        [rad/s].
    n : array_like
        Species densities along axis 0, non-negative [m^-3].
    q : array_like
        Signed species charges, shape ``(S,)`` [C].
    m : array_like
        Species masses, shape ``(S,)``, positive [kg].
    B : float or np.ndarray
        Magnetic-field magnitude, broadcast against the trailing axes [T].

    Returns
    -------
    StixParameters
        ``(R, L, S, D, P)``; infinite at a cyclotron resonance
        $\omega = |\Omega_s|$ of a species with non-zero density [-].

    Raises
    ------
    ValueError
        A non-positive ``omega``, species arrays of different length, or an
        invalid density or mass.

    Convention
    ----------
    $\Omega_s = q_sB/m_s$ is signed (``gyrofrequency``), $\Omega_e < 0$, with
    $\mathbf B_0 \parallel \hat z$. The dielectric tensor is
    $\begin{pmatrix} S & -iD & 0\\ iD & S & 0\\ 0 & 0 & P\end{pmatrix}$ for
    fields $\propto e^{i(\mathbf k\cdot\mathbf x - \omega t)}$. With the
    signed $\Omega_e$, $R$ carries the electron cyclotron resonance and $L$
    the ion ones. $P$ does not depend on $B$.

    Physical interpretation
    -----------------------
    $R$ and $L$ are the dielectric responses to right- and left-hand
    circular polarisation, $P$ to a field along $\mathbf B_0$, $S$ and $D$
    their mean and half-difference; every cold-plasma cutoff is a zero of
    $P$, $R$ or $L$, and every resonance a pole of $R$ or $L$ or a zero of
    $S$ (at perpendicular propagation).

    Assumptions
    -----------
    Cold species (thermal speeds negligible against $\omega/k$),
    collisionless, uniform on the wavelength, no equilibrium flows.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-2.
    """
    omega = np.asarray(omega, dtype=float)
    if np.any(~(omega > 0.0)) or np.any(~np.isfinite(omega)):
        raise ValueError("omega must be positive and finite")
    q = np.atleast_1d(np.asarray(q, dtype=float))
    m = np.atleast_1d(np.asarray(m, dtype=float))
    n = np.asarray(n, dtype=float)
    if n.ndim == 0:
        n = n[None]
    if q.ndim != 1 or q.shape != m.shape or n.shape[0] != q.shape[0]:
        raise ValueError("q and m must be 1-D with one entry per species along axis 0 of n")
    B = np.asarray(B, dtype=float)
    if np.any(~np.isfinite(B)):
        raise ValueError("B must be finite")
    extra = (1,) * max(n.ndim - 1, np.ndim(B), np.ndim(omega))
    wp2 = plasma_frequency(n, q.reshape(q.shape + (1,) * (n.ndim - 1)), m.reshape(m.shape + (1,) * (n.ndim - 1)))
    wp2 = np.asarray(wp2, dtype=float) ** 2
    wp2 = wp2.reshape(wp2.shape + (1,) * (len(extra) - (wp2.ndim - 1)))
    Omega = gyrofrequency(q.reshape(q.shape + extra), m.reshape(m.shape + extra), B)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = 1.0 - np.sum(wp2 / (omega * (omega + Omega)), axis=0)
        L = 1.0 - np.sum(wp2 / (omega * (omega - Omega)), axis=0)
    P = 1.0 - np.sum(wp2 / omega**2 * np.ones_like(Omega), axis=0)
    S, D = (R + L) / 2.0, (R - L) / 2.0
    return StixParameters(*(_out(v) for v in (R, L, S, D, P)))


def cold_plasma_refractive_index_squared(R, L, P, theta):
    r"""The two roots $n^2$ of the cold-plasma dispersion relation at angle $\theta$ to $\mathbf B_0$.

    $$An^4 - Bn^2 + C = 0,\qquad n^2_\pm = \frac{B \pm F}{2A}$$
    $$A = S\sin^2\theta + P\cos^2\theta,\quad B = RL\sin^2\theta + PS(1 + \cos^2\theta),\quad C = PRL,$$
    $$F^2 = (RL - PS)^2\sin^4\theta + 4P^2D^2\cos^2\theta$$

    Parameters
    ----------
    R : float or np.ndarray
        Stix $R$ [-].
    L : float or np.ndarray
        Stix $L$ [-].
    P : float or np.ndarray
        Stix $P$ [-].
    theta : float or np.ndarray
        Angle between $\mathbf k$ and $\mathbf B_0$ [rad].

    Returns
    -------
    tuple of float or np.ndarray
        ``(n2_plus, n2_minus)``; infinite where $A = 0$ (a resonance cone)
        [-].

    Raises
    ------
    ValueError
        A non-finite ``theta``.

    Convention
    ----------
    $F$ is written in Stix's non-negative form, so $F^2 = B^2 - 4AC \ge 0$
    and both roots are real for real Stix parameters. The $\pm$ labels are
    the algebraic branches, not O/X or R/L: which physical mode a branch is
    changes across $A = 0$ and $\theta$, so name modes from a tracked root
    (``perpendicular_refractive_index_squared`` at $\theta = \pi/2$; $R$
    and $L$ at $\theta = 0$).

    Physical interpretation
    -----------------------
    $n^2 > 0$ propagates, $n^2 < 0$ is evanescent, $n^2 = 0$ is a cutoff
    (reflection) and $n^2 \to \infty$ a resonance (absorption, or mode
    conversion) -- ``propagation_regime``.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-3.
    """
    theta = np.asarray(theta, dtype=float)
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta must be finite")
    R, L, P = (np.asarray(v, dtype=float) for v in (R, L, P))
    S, D = (R + L) / 2.0, (R - L) / 2.0
    s2, c2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    A = S * s2 + P * c2
    Bq = R * L * s2 + P * S * (1.0 + c2)
    F = np.sqrt((R * L - P * S) ** 2 * s2**2 + 4.0 * P**2 * D**2 * c2)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = np.where(A != 0.0, (Bq + F) / (2.0 * A), np.inf)
        minus = np.where(A != 0.0, (Bq - F) / (2.0 * A), np.inf)
    return _out(plus), _out(minus)


def perpendicular_refractive_index_squared(R, L, P):
    r"""Ordinary and extraordinary modes at perpendicular propagation.

    $$n_O^2 = P,\qquad n_X^2 = \frac{RL}{S}$$

    Parameters
    ----------
    R : float or np.ndarray
        Stix $R$ [-].
    L : float or np.ndarray
        Stix $L$ [-].
    P : float or np.ndarray
        Stix $P$ [-].

    Returns
    -------
    tuple of float or np.ndarray
        ``(n2_O, n2_X)``; $n_X^2$ is infinite at $S = 0$, the hybrid
        resonances [-].

    Convention
    ----------
    $\theta = \pi/2$: the O mode has $\mathbf E \parallel \mathbf B_0$ and
    feels only $P$; the X mode has $\mathbf E \perp \mathbf B_0$. These are
    the two roots of ``cold_plasma_refractive_index_squared`` at
    $\theta = \pi/2$, named by polarisation.

    Physical interpretation
    -----------------------
    O-mode cutoff: $P = 0$, $\omega = \omega_{pe}$ (with cold ions). X-mode
    cutoffs: $R = 0$ and $L = 0$; its resonance $S = 0$ is the upper (and
    lower) hybrid layer, reached from the high-field side or by tunnelling.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-4.
    """
    R, L, P = (np.asarray(v, dtype=float) for v in (R, L, P))
    S = (R + L) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        n2_X = np.where(S != 0.0, R * L / S, np.inf)
    return _out(P), _out(n2_X)


def cma_coordinates(omega, n_e, B):
    r"""CMA-diagram coordinates of a cold electron plasma.

    $$X = \frac{\omega_{pe}^2}{\omega^2},\qquad Y = \frac{|\Omega_e|}{\omega}$$

    Parameters
    ----------
    omega : float or np.ndarray
        Wave angular frequency, positive [rad/s].
    n_e : float or np.ndarray
        Electron density, non-negative [m^-3].
    B : float or np.ndarray
        Magnetic-field magnitude [T].

    Returns
    -------
    tuple of float or np.ndarray
        ``(X, Y)``, both non-negative [-].

    Raises
    ------
    ValueError
        A non-positive ``omega`` or a negative density.

    Convention
    ----------
    $Y$ is the positive magnitude $|\Omega_e|/\omega$, the usual CMA axis;
    ``stix_parameters`` uses the signed $\Omega_e$. For electrons only,
    $P = 1 - X$, $R = 1 - X/(1 - Y)$, $L = 1 - X/(1 + Y)$ and
    $S = 1 - X/(1 - Y^2)$.

    Physical interpretation
    -----------------------
    A point $(X, Y)$ is a local plasma at one frequency: moving along a
    profile at fixed $\omega$ traces a curve in the CMA plane, and each
    boundary it crosses is a cutoff or resonance of some mode.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 2-2.
    """
    omega = np.asarray(omega, dtype=float)
    if np.any(~(omega > 0.0)):
        raise ValueError("omega must be positive")
    X = np.asarray(plasma_frequency(n_e, -QE, ME), dtype=float) ** 2 / omega**2
    Y = np.abs(np.asarray(gyrofrequency(-QE, ME, B), dtype=float)) / omega
    return _out(X), _out(Y)


def propagation_regime(n2, *, atol=0.0):
    r"""Classify a refractive index squared: propagating, evanescent, cutoff or resonance.

    $$n^2 > 0: \text{propagating},\quad n^2 < 0: \text{evanescent},\quad
      n^2 = 0: \text{cutoff},\quad |n^2| \to \infty: \text{resonance}$$

    Parameters
    ----------
    n2 : float or np.ndarray
        Refractive index squared [-].
    atol : float
        Half-width of the band $|n^2| \le$ ``atol`` counted as a cutoff;
        zero by default [-].

    Returns
    -------
    str or np.ndarray
        ``"propagating"``, ``"evanescent"``, ``"cutoff"`` or ``"resonance"``
        per element [-].

    Raises
    ------
    ValueError
        A negative ``atol``, or a NaN in ``n2``.

    Convention
    ----------
    A resonance is $\pm\infty$ (as ``cold_plasma_refractive_index_squared``
    returns at $A = 0$); a large finite $n^2$ near it is still
    "propagating" or "evanescent". Cutoff and resonance are layers of zero
    width, so on a sampled profile they appear only through ``atol`` or as
    sign changes between samples.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-3.
    """
    if not atol >= 0.0:
        raise ValueError("atol must be non-negative")
    n2 = np.asarray(n2, dtype=float)
    if np.any(np.isnan(n2)):
        raise ValueError("n2 must not be NaN")
    result = np.where(np.isinf(n2), "resonance",
                      np.where(np.abs(n2) <= atol, "cutoff", np.where(n2 > 0.0, "propagating", "evanescent")))
    return str(result) if result.ndim == 0 else result
