r"""Two-variable turbulence--zonal-flow predator--prey model (Lotka--Volterra form).

The transparent reduced description of turbulence self-regulation by zonal
flows (issue #1820): fluctuation intensity $N$ (the prey) grows at an
effective linear rate and is sheared apart by the zonal flow; zonal-flow
intensity $E$ (the predator) is driven by the turbulence through the
Reynolds stress and decays at an effective damping rate.  The functions here
are the model's analytic properties only -- right-hand side, fixed point,
conserved quantity, small-amplitude period and response lag.  Integrating
orbits, extracting $N$ and $E$ from gyrokinetic fields and fitting
the coefficients belong to :mod:`vaft.process`.

The model is a reference against which a gyrokinetic time series can be tested,
not a claim that every nonlinear state obeys it: a bursty trace may equally come
from tertiary instability, profile relaxation or noise.

Notation
--------
N           : turbulence (non-zonal fluctuation) intensity          [arbitrary, > 0]
E           : zonal-flow intensity                                    [arbitrary, > 0]
gamma_eff   : effective turbulence growth rate                        [1/time]
gamma_Z     : effective zonal-flow damping rate                       [1/time]
c1          : suppression of turbulence by the zonal flow             [1/(time * E)]
c2          : drive of the zonal flow by the turbulence               [1/(time * N)]

Conventions
-----------
$$\dot N = \gamma_\mathrm{eff} N - c_1 N E, \qquad \dot E = c_2 N E - \gamma_Z E$$

Any consistent time unit is accepted (s, $a/c_s$, ...): the rates set it,
and every returned time is in the same unit.  $N$ and $E$ carry no
fixed normalisation.  Rescaling $N \to \lambda N$ maps
$c_2 \to c_2/\lambda$, and $E \to \mu E$ maps
$c_1 \to c_1/\mu$; the rates and the dynamics are unchanged.  A fitted
$c_1, c_2$ therefore means nothing without the definition of the
intensities it was fitted to.

References
----------
.. [1] P. H. Diamond, Y.-M. Liang, B. A. Carreras and P. W. Terry,
       Phys. Rev. Lett. 72 (1994) 2565.
.. [2] P. H. Diamond, S.-I. Itoh, K. Itoh and T. S. Hahm,
       Plasma Phys. Control. Fusion 47 (2005) R35.
.. [3] S. Kobayashi, O. D. Gurcan and P. H. Diamond, Phys. Plasmas 22 (2015) 090702.
.. [4] M. Leconte, A. Masson and L. Qi, Phys. Plasmas 29 (2022) 022302.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "predator_prey_fixed_point",
    "predator_prey_invariant",
    "predator_prey_period",
    "predator_prey_response_lag",
    "predator_prey_rhs",
]

_APPROXIMATIONS = ("linearized",)


def _out(value):
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def _rate(value, name, *, strict: bool):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0) or (strict and np.any(array == 0.0)):
        bound = "positive" if strict else "non-negative"
        raise ValueError(f"{name} must be {bound} in the classical predator-prey model")
    return array


def _intensity(value, name, *, strict: bool):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if strict and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if not strict and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _approximation(approximation: str) -> None:
    if approximation not in _APPROXIMATIONS:
        raise ValueError(f"approximation must be one of {_APPROXIMATIONS}, not {approximation!r}")


def predator_prey_rhs(turbulence, zonal, gamma_eff, coupling_suppression, coupling_drive, gamma_zonal):
    r"""Time derivatives of the turbulence--zonal-flow predator--prey model.

    $$\dot N = \gamma_\mathrm{eff}N - c_1 N E, \qquad \dot E = c_2 N E - \gamma_Z E$$

    Parameters
    ----------
    turbulence : float or array-like
        Turbulence intensity $N$, non-negative [arbitrary].
    zonal : float or array-like
        Zonal-flow intensity $E$, non-negative [arbitrary].
    gamma_eff : float or array-like
        Effective turbulence growth rate $\gamma_\mathrm{eff}$, non-negative [1/time].
    coupling_suppression : float or array-like
        Suppression coefficient $c_1$, non-negative [1/(time E)].
    coupling_drive : float or array-like
        Drive coefficient $c_2$, non-negative [1/(time N)].
    gamma_zonal : float or array-like
        Effective zonal-flow damping rate $\gamma_Z$, non-negative [1/time].

    Returns
    -------
    d_turbulence_dt : float or np.ndarray
        $\dot N$ [N/time].
    d_zonal_dt : float or np.ndarray
        $\dot E$ [E/time].

    Raises
    ------
    ValueError
        Non-finite input, negative intensity, or a negative rate or coupling.

    Convention
    ----------
    Arguments broadcast with NumPy rules; the time unit is the one the rates
    are given in.

    Physical interpretation
    -----------------------
    $\gamma_\mathrm{eff}N$ is linear drive; $-c_1NE$ is shear decorrelation of
    the turbulence by the zonal flow; $c_2NE$ is Reynolds-stress drive of the
    zonal flow; $-\gamma_ZE$ is its collisional (or other) damping.

    Assumptions
    -----------
    Two scalar intensities stand for the whole spectrum; the couplings are
    constants; no saturation of $N$ in the absence of zonal flow and no
    zonal-flow self-interaction.

    Validity
    --------
    A reduced model for weakly to moderately driven turbulence near marginal
    stability, where zonal flows dominate saturation.  Away from that regime
    (strong drive, tertiary instability, profile relaxation) it is a
    reference, not a prediction.

    References
    ----------
    .. [1] P. H. Diamond, Y.-M. Liang, B. A. Carreras and P. W. Terry,
           Phys. Rev. Lett. 72 (1994) 2565.
    .. [2] S. Kobayashi, O. D. Gurcan and P. H. Diamond, Phys. Plasmas 22 (2015) 090702.
    """
    n = _intensity(turbulence, "turbulence", strict=False)
    e = _intensity(zonal, "zonal", strict=False)
    g = _rate(gamma_eff, "gamma_eff", strict=False)
    c1 = _rate(coupling_suppression, "coupling_suppression", strict=False)
    c2 = _rate(coupling_drive, "coupling_drive", strict=False)
    gz = _rate(gamma_zonal, "gamma_zonal", strict=False)
    n, e, g, c1, c2, gz = np.broadcast_arrays(n, e, g, c1, c2, gz)
    return _out(g * n - c1 * n * e), _out(c2 * n * e - gz * e)


def predator_prey_fixed_point(gamma_eff, coupling_suppression, coupling_drive, gamma_zonal):
    r"""Positive (coexistence) fixed point of the predator--prey model.

    $$N_* = \frac{\gamma_Z}{c_2}, \qquad E_* = \frac{\gamma_\mathrm{eff}}{c_1}$$

    Parameters
    ----------
    gamma_eff : float or array-like
        Effective turbulence growth rate, positive [1/time].
    coupling_suppression : float or array-like
        Suppression coefficient $c_1$, positive [1/(time E)].
    coupling_drive : float or array-like
        Drive coefficient $c_2$, positive [1/(time N)].
    gamma_zonal : float or array-like
        Effective zonal-flow damping rate, positive [1/time].

    Returns
    -------
    turbulence_star : float or np.ndarray
        $N_*$ [N].
    zonal_star : float or np.ndarray
        $E_*$ [E].

    Raises
    ------
    ValueError
        Non-finite input or a non-positive rate or coupling (the positive
        fixed point does not exist then).

    Convention
    ----------
    The model has two equilibria: the trivial $(0, 0)$ and this positive one.
    Only the positive one is returned.

    Physical interpretation
    -----------------------
    The turbulence level at which zonal-flow drive balances its damping, and
    the zonal-flow level at which shear suppression balances the linear drive.
    It is the time average of $N$ and $E$ over any closed orbit.

    Assumptions
    -----------
    The classical model of :func:`predator_prey_rhs`.

    Validity
    --------
    The positive fixed point of the classical model is a centre: neutrally
    stable, surrounded by closed orbits, and *not* asymptotically stable.  A
    trajectory does not relax to it; damping or saturation terms (not in this
    model) are needed for that.

    References
    ----------
    .. [1] P. H. Diamond, S.-I. Itoh, K. Itoh and T. S. Hahm,
           Plasma Phys. Control. Fusion 47 (2005) R35.
    """
    g = _rate(gamma_eff, "gamma_eff", strict=True)
    c1 = _rate(coupling_suppression, "coupling_suppression", strict=True)
    c2 = _rate(coupling_drive, "coupling_drive", strict=True)
    gz = _rate(gamma_zonal, "gamma_zonal", strict=True)
    g, c1, c2, gz = np.broadcast_arrays(g, c1, c2, gz)
    return _out(gz / c2), _out(g / c1)


def predator_prey_invariant(turbulence, zonal, gamma_eff, coupling_suppression, coupling_drive, gamma_zonal):
    r"""Conserved quantity of the classical predator--prey model.

    $$H(N, E) = c_2 N - \gamma_Z \ln N + c_1 E - \gamma_\mathrm{eff} \ln E$$

    Parameters
    ----------
    turbulence : float or array-like
        Turbulence intensity $N$, strictly positive [arbitrary].
    zonal : float or array-like
        Zonal-flow intensity $E$, strictly positive [arbitrary].
    gamma_eff : float or array-like
        Effective turbulence growth rate, positive [1/time].
    coupling_suppression : float or array-like
        Suppression coefficient $c_1$, positive [1/(time E)].
    coupling_drive : float or array-like
        Drive coefficient $c_2$, positive [1/(time N)].
    gamma_zonal : float or array-like
        Effective zonal-flow damping rate, positive [1/time].

    Returns
    -------
    float or np.ndarray
        $H$ [1/time].

    Raises
    ------
    ValueError
        Non-finite input, non-positive intensity, or a non-positive rate or
        coupling.

    Convention
    ----------
    The logarithms take $N$ and $E$ in whatever unit they are given, so $H$ is
    defined up to an additive constant; only differences of $H$ along or
    between orbits are meaningful.

    Physical interpretation
    -----------------------
    Each closed orbit is a level set of $H$; its minimum
    $H(N_*, E_*)$ marks the fixed point.  The amount by which a measured
    trajectory changes $H$ measures how far it departs from classical
    predator--prey dynamics.

    Assumptions
    -----------
    The classical model of :func:`predator_prey_rhs`; $dH/dt = 0$ along its
    exact trajectories.

    Validity
    --------
    Conserved only for the undamped two-variable model; any saturation,
    noise or third variable makes $H$ drift.

    References
    ----------
    .. [1] M. Leconte, A. Masson and L. Qi, Phys. Plasmas 29 (2022) 022302.
    """
    n = _intensity(turbulence, "turbulence", strict=True)
    e = _intensity(zonal, "zonal", strict=True)
    g = _rate(gamma_eff, "gamma_eff", strict=True)
    c1 = _rate(coupling_suppression, "coupling_suppression", strict=True)
    c2 = _rate(coupling_drive, "coupling_drive", strict=True)
    gz = _rate(gamma_zonal, "gamma_zonal", strict=True)
    return _out(c2 * n - gz * np.log(n) + c1 * e - g * np.log(e))


def predator_prey_period(gamma_eff, gamma_zonal, *, approximation="linearized"):
    r"""Small-amplitude oscillation period about the positive fixed point.

    $$\omega_\mathrm{PP} = \sqrt{\gamma_\mathrm{eff}\gamma_Z}, \qquad
      T_\mathrm{lin} = \frac{2\pi}{\sqrt{\gamma_\mathrm{eff}\gamma_Z}}$$

    Parameters
    ----------
    gamma_eff : float or array-like
        Effective turbulence growth rate, positive [1/time].
    gamma_zonal : float or array-like
        Effective zonal-flow damping rate, positive [1/time].
    approximation : {"linearized"}, optional
        Only the linearized period is implemented [-].

    Returns
    -------
    float or np.ndarray
        $T_\mathrm{lin}$ [time].

    Raises
    ------
    ValueError
        Non-finite input, a non-positive rate, or an unknown approximation.

    Convention
    ----------
    The same time unit as the rates.  The couplings $c_1, c_2$ drop out of the
    linearized period.

    Physical interpretation
    -----------------------
    The time for one turbulence burst -- zonal-flow growth -- turbulence
    quench -- zonal-flow decay cycle for a small excursion about the fixed
    point.

    Assumptions
    -----------
    Linearization of :func:`predator_prey_rhs` about $(N_*, E_*)$.

    Validity
    --------
    Exact only in the small-amplitude limit.  Finite-amplitude
    Lotka--Volterra orbits have amplitude-dependent periods that grow with
    the orbit size; this is not the period of a general orbit.

    References
    ----------
    .. [1] M. Leconte, A. Masson and L. Qi, Phys. Plasmas 29 (2022) 022302.
    """
    _approximation(approximation)
    g = _rate(gamma_eff, "gamma_eff", strict=True)
    gz = _rate(gamma_zonal, "gamma_zonal", strict=True)
    return _out(2.0 * np.pi / np.sqrt(g * gz))


def predator_prey_response_lag(gamma_eff, gamma_zonal, *, approximation="linearized"):
    r"""Small-amplitude delay of the zonal-flow peak after the turbulence peak.

    $$\tau_\mathrm{lag} = \frac{\pi}{2\sqrt{\gamma_\mathrm{eff}\gamma_Z}} = \frac{T_\mathrm{lin}}{4}$$

    Parameters
    ----------
    gamma_eff : float or array-like
        Effective turbulence growth rate, positive [1/time].
    gamma_zonal : float or array-like
        Effective zonal-flow damping rate, positive [1/time].
    approximation : {"linearized"}, optional
        Only the linearized lag is implemented [-].

    Returns
    -------
    float or np.ndarray
        $\tau_\mathrm{lag}$ [time].

    Raises
    ------
    ValueError
        Non-finite input, a non-positive rate, or an unknown approximation.

    Convention
    ----------
    Positive when the zonal flow (predator) peaks after the turbulence (prey);
    the same time unit as the rates.

    Physical interpretation
    -----------------------
    Linearizing about the fixed point gives $\delta\dot N = -c_1N_*\delta E$
    and $\delta\dot E = c_2E_*\delta N$, so $\delta E$ is $\delta N$ shifted by
    a quarter period: the zonal flow builds up while the turbulence is high
    and peaks when the turbulence, already falling, crosses its mean level
    and is collapsing fastest.

    Assumptions
    -----------
    Linearization of :func:`predator_prey_rhs` about $(N_*, E_*)$.

    Validity
    --------
    $T/4$ holds in the small-amplitude limit only; finite-amplitude orbits
    have a different, amplitude-dependent delay.  A delay measured from data
    belongs in :mod:`vaft.process`.

    References
    ----------
    .. [1] M. Leconte, A. Masson and L. Qi, Phys. Plasmas 29 (2022) 022302.
    """
    _approximation(approximation)
    return _out(0.25 * np.asarray(predator_prey_period(gamma_eff, gamma_zonal), dtype=float))
