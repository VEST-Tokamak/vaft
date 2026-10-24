r"""Turbulence reference relations: fluctuation-driven fluxes, drift-wave scales, shear, zonal flows.

Two groups (issues #1112 and #1820). The first are definitions and reduced
estimates of fluctuation-driven transport -- the diamagnetic velocity and
frequency, the E-cross-B velocity of a potential fluctuation, turbulent
particle and heat fluxes as correlations, the cross-phase flux of one mode,
the mixing-length diffusivity, the E-cross-B shearing rate, the Reynolds
stress and the zonal-flow drive. None is a turbulence closure, a stability
classifier or a transport prediction; geometry is slab unless stated.

The second is the two-variable turbulence--zonal-flow predator--prey model
(Lotka--Volterra form).

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
    "coherent_particle_flux_from_amplitudes_phase",
    "diamagnetic_frequency_from_k_v_star",
    "diamagnetic_velocity_from_grad_p_q_n_B",
    "exb_shearing_rate_from_velocity_gradient",
    "exb_velocity_from_potential_gradient",
    "shear_to_growth_rate_ratio",
    "mixing_length_diffusivity_from_gamma_k_perp",
    "predator_prey_fixed_point",
    "predator_prey_invariant",
    "predator_prey_period",
    "predator_prey_response_lag",
    "predator_prey_rhs",
    "reynolds_stress_from_velocity_fluctuations",
    "turbulent_heat_flux_from_fluctuations",
    "turbulent_particle_flux_from_fluctuations",
    "zonal_flow_drive_from_reynolds_stress_gradient",
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


# ---------------------------------------------------------------- fluctuation-driven transport (#1112)


def _vector3(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have three Cartesian components on its last axis")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive(value, name):
    array = _finite(value, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def _ensemble_mean(a, b, axis, names):
    a = _finite(a, names[0])
    b = _finite(b, names[1])
    if a.shape != b.shape:  # broadcasting (N,) against (N, 1) would average an outer product
        raise ValueError(f"{names[0]} and {names[1]} must have the same shape, not {a.shape} and {b.shape}")
    if a.ndim == 0 or not -a.ndim <= axis < a.ndim or a.shape[axis] < 1:
        raise ValueError("the averaging axis must exist and hold at least one sample")
    return _out(np.mean(a * b, axis=axis))


def diamagnetic_velocity_from_grad_p_q_n_B(grad_p, q, n, B):
    r"""Diamagnetic drift velocity of a species, from its pressure gradient.

    $$\mathbf v_{*s} = \frac{\mathbf B\times\nabla p_s}{q_s n_s B^2}$$

    Parameters
    ----------
    grad_p : array-like
        Pressure gradient of the species, last axis the three Cartesian
        components [Pa/m].
    q : float or array-like
        Species charge, signed (negative for electrons), non-zero [C].
    n : float or array-like
        Species density, positive [m^-3].
    B : array-like
        Magnetic field, last axis the three Cartesian components, non-zero [T].

    Returns
    -------
    np.ndarray
        $\mathbf v_{*s}$, last axis the three Cartesian components [m/s].

    Raises
    ------
    ValueError
        A vector without three components, non-finite input, a zero charge or
        field, or a non-positive density.

    Convention
    ----------
    Right-handed Cartesian components. Ions and electrons drift in opposite
    directions through the sign of $q_s$; $\mathbf v_{*s}$ is a fluid
    velocity, not a guiding-centre drift.

    Physical interpretation
    -----------------------
    More gyrating particles on the high-pressure side than on the low one
    leave a net fluid flow perpendicular to $\mathbf B$ and $\nabla p$, with no
    guiding centre moving.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled
           Fusion*, 3rd ed., Springer (2016), Sec. 3.5.
    .. [2] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    gp = _vector3(grad_p, "grad_p")
    b = _vector3(B, "B")
    b2 = np.sum(b * b, axis=-1)
    if np.any(b2 == 0.0):
        raise ValueError("B must be non-zero")
    charge = _finite(q, "q")
    if np.any(charge == 0.0):
        raise ValueError("q must be non-zero")
    density = _positive(n, "n")
    lead = np.broadcast_shapes(gp.shape[:-1], b.shape[:-1])
    try:
        scale = np.broadcast_to(charge * density, lead)
    except ValueError as error:
        raise ValueError(f"q and n must broadcast to the vectors' leading shape {lead}") from error
    return np.cross(b, gp) / (scale * np.broadcast_to(b2, lead))[..., None]


def diamagnetic_frequency_from_k_v_star(k, v_star):
    r"""Diamagnetic frequency of one Fourier mode: the Doppler shift of the diamagnetic flow.

    $$\omega_{*s} = \mathbf k\cdot\mathbf v_{*s}$$

    Parameters
    ----------
    k : array-like
        Wave vector, last axis the three Cartesian components [1/m].
    v_star : array-like
        Diamagnetic velocity of the species, last axis the three Cartesian
        components, as from :func:`diamagnetic_velocity_from_grad_p_q_n_B` [m/s].

    Returns
    -------
    float or np.ndarray
        $\omega_{*s}$ [rad/s].

    Raises
    ------
    ValueError
        A vector without three components, or non-finite input.

    Convention
    ----------
    The sign follows the species charge through $\mathbf v_{*s}$ and the
    direction of $\mathbf k$; only the perpendicular part of $\mathbf k$
    contributes, since $\mathbf v_{*s}\perp\mathbf B$. A reduced form in
    $k_y\rho_s$ is not given here: it needs the binormal direction and the
    reference gyroradius stated, which belong to the code normalisation.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    kk = _vector3(k, "k")
    v = _vector3(v_star, "v_star")
    return _out(np.sum(kk * v, axis=-1))


def exb_velocity_from_potential_gradient(grad_phi, B):
    r"""Electrostatic E-cross-B velocity from a potential gradient.

    $$\mathbf v_E = \frac{\mathbf B\times\nabla\phi}{B^2}$$

    Parameters
    ----------
    grad_phi : array-like
        Gradient of the electrostatic potential (mean or fluctuating), last
        axis the three Cartesian components [V/m].
    B : array-like
        Magnetic field, last axis the three Cartesian components [T].

    Returns
    -------
    np.ndarray
        $\mathbf v_E$, last axis the three Cartesian components [m/s].

    Raises
    ------
    ValueError
        A vector without three components, or ``B`` vanishes.

    Convention
    ----------
    $\mathbf E = -\nabla\phi$, so this is
    :func:`vaft.formula.particle.exb_drift_velocity` evaluated at
    $-\nabla\phi$; that function remains the owner of
    $\mathbf E\times\mathbf B/B^2$ and is called here.

    Physical interpretation
    -----------------------
    The fluid velocity of a potential fluctuation: equipotentials are
    streamlines, so a potential eddy is a convective cell.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    from .particle import exb_drift_velocity

    return exb_drift_velocity(-_vector3(grad_phi, "grad_phi"), _vector3(B, "B"))


def turbulent_particle_flux_from_fluctuations(n_fluct, v_r_fluct, axis=-1):
    r"""Turbulent radial particle flux: the correlation of density and radial-velocity fluctuations.

    $$\Gamma_s = \langle\tilde n_s\,\tilde v_r\rangle$$

    Parameters
    ----------
    n_fluct : array-like
        Density fluctuation $\tilde n_s$ samples [m^-3].
    v_r_fluct : array-like
        Radial velocity fluctuation $\tilde v_r$ at the same samples, positive
        outward [m/s].
    axis : int
        Axis that ``<.>`` averages over (time, ensemble or flux surface) [-].

    Returns
    -------
    float or np.ndarray
        $\Gamma_s$, positive outward [m^-2 s^-1].

    Raises
    ------
    ValueError
        Non-finite input, inputs of different shapes, or a missing or empty averaging axis.

    Convention
    ----------
    The inputs are the fluctuating parts: removing the mean (or the
    equilibrium) is a procedure the caller does before this, and a
    non-zero mean in both inputs adds $\bar n\bar v_r$ to the result. The
    average is the arithmetic mean along ``axis``.

    Physical interpretation
    -----------------------
    A flux needs density and radial velocity to be correlated: two fluctuations
    of any amplitude carry nothing if they are out of phase by $\pi/2$
    (see :func:`coherent_particle_flux_from_amplitudes_phase`).

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    .. [2] A. J. Wootton et al., Phys. Fluids B 2 (1990) 2879.
    """
    return _ensemble_mean(n_fluct, v_r_fluct, axis, ("n_fluct", "v_r_fluct"))


def turbulent_heat_flux_from_fluctuations(p_fluct, v_r_fluct, axis=-1):
    r"""Turbulent radial heat flux carried by pressure fluctuations: $\tfrac32\langle\tilde p\,\tilde v_r\rangle$.

    $$Q_s = \frac{3}{2}\langle\tilde p_s\,\tilde v_r\rangle$$

    Parameters
    ----------
    p_fluct : array-like
        Pressure fluctuation $\tilde p_s = \widetilde{n_sT_s}$ samples [Pa].
    v_r_fluct : array-like
        Radial velocity fluctuation $\tilde v_r$ at the same samples, positive
        outward [m/s].
    axis : int
        Axis that ``<.>`` averages over [-].

    Returns
    -------
    float or np.ndarray
        $Q_s$, positive outward [W/m^2].

    Raises
    ------
    ValueError
        Non-finite input, inputs of different shapes, or a missing or empty averaging axis.

    Convention
    ----------
    The flux of the fluctuating thermal energy $\tfrac32\tilde p_s$ by the
    radial velocity -- the energy-flux definition of the electrostatic
    gyrokinetic codes. It is the total (convective plus conductive) energy
    flux; the conductive part alone is
    $Q_s - \tfrac32T_s\Gamma_s$, and some texts use $\tfrac52$ in the
    convective term. ``p_fluct`` must be the fluctuation of $n_sT_s$, not
    $T_s\tilde n_s$ or $n_s\tilde T_s$ alone.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    return _out(1.5 * np.asarray(_ensemble_mean(p_fluct, v_r_fluct, axis, ("p_fluct", "v_r_fluct"))))


def coherent_particle_flux_from_amplitudes_phase(n_amplitude, phi_amplitude, k_y, B, phase):
    r"""E-cross-B particle flux of one Fourier mode from its amplitudes and cross-phase.

    $$\Gamma_k = \frac{k_y\,|\tilde n_k|\,|\tilde\phi_k|}{2B}\,\sin\theta_{n\phi}$$

    Parameters
    ----------
    n_amplitude : float or array-like
        Peak density amplitude $|\tilde n_k|$, non-negative [m^-3].
    phi_amplitude : float or array-like
        Peak potential amplitude $|\tilde\phi_k|$, non-negative [V].
    k_y : float or array-like
        Wavenumber along the binormal direction $\hat y$ [1/m].
    B : float or array-like
        Field strength, positive [T].
    phase : float or array-like
        Cross-phase $\theta_{n\phi} = \alpha_\phi - \alpha_n$: the phase by which
        the potential leads the density [rad].

    Returns
    -------
    float or np.ndarray
        $\Gamma_k$ along $+\hat x$ (outward), averaged over a period [m^-2 s^-1].

    Raises
    ------
    ValueError
        Non-finite input, a negative amplitude, or a non-positive field.

    Convention
    ----------
    Slab geometry: $\hat x$ radial (outward), $\hat y$ binormal and
    $\mathbf B = B\hat z$, right-handed. Real fields
    $\tilde n = |\tilde n_k|\cos(k_yy - \omega t + \alpha_n)$ and
    $\tilde\phi = |\tilde\phi_k|\cos(k_yy - \omega t + \alpha_\phi)$, so
    $\tilde v_x = -\partial_y\tilde\phi/B$ from
    :func:`exb_velocity_from_potential_gradient`; the factor $1/2$ is the
    period average of $\cos\cdot\sin$. Another Fourier sign convention
    ($e^{-i\omega t}$ against $e^{i\omega t}$), a left-handed frame or a
    reversed field flips the sign.

    Physical interpretation
    -----------------------
    In phase ($\theta_{n\phi} = 0$, the adiabatic electron response) a drift
    wave carries no flux; the flux grows with the phase shift that a
    non-adiabatic response opens and reverses with its sign.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    .. [2] E. J. Powers, Nucl. Fusion 14 (1974) 749.
    """
    na = _finite(n_amplitude, "n_amplitude")
    pa = _finite(phi_amplitude, "phi_amplitude")
    if np.any(na < 0.0) or np.any(pa < 0.0):
        raise ValueError("amplitudes must be non-negative")
    ky = _finite(k_y, "k_y")
    b = _positive(B, "B")
    theta = _finite(phase, "phase")
    return _out(ky * na * pa / (2.0 * b) * np.sin(theta))


def mixing_length_diffusivity_from_gamma_k_perp(gamma, k_perp):
    r"""Mixing-length diffusivity estimate: growth rate over the squared perpendicular wavenumber.

    $$D_\mathrm{ML} \sim \frac{\gamma}{k_\perp^2}$$

    Parameters
    ----------
    gamma : float or array-like
        Linear growth rate of the mode, non-negative [1/s].
    k_perp : float or array-like
        Perpendicular wavenumber of the mode, non-zero [1/m].

    Returns
    -------
    float or np.ndarray
        $\gamma/k_\perp^2$ [m^2/s].

    Raises
    ------
    ValueError
        Non-finite input, a negative growth rate or a zero wavenumber.

    Convention
    ----------
    No order-unity prefactor. In gyro-Bohm units the same expression is
    $(\gamma\,a/c_s)/(k_\perp\rho_s)^2$ and returns $D/\chi_\mathrm{gB}$; which
    $k_\perp$ (often $k_y$ alone, at $k_x = 0$) is the caller's statement.

    Physical interpretation
    -----------------------
    Saturation when the eddy grows to the size of its own wavelength:
    $\tilde n/n \sim 1/(k_\perp L_n)$ and a step $1/k_\perp$ per growth time
    $1/\gamma$ give the random-walk diffusivity $\gamma/k_\perp^2$.

    Limitations
    -----------
    A reduced saturation estimate, not a transport closure: it ignores
    zonal flows, nonlinear spectral transfer and the mode spectrum, and is no
    prediction of ITG, TEM or ETG transport.

    References
    ----------
    .. [1] B. B. Kadomtsev, *Plasma Turbulence*, Academic Press (1965).
    .. [2] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    g = _finite(gamma, "gamma")
    if np.any(g < 0.0):
        raise ValueError("gamma must be non-negative")
    k = _finite(k_perp, "k_perp")
    if np.any(k == 0.0):
        raise ValueError("k_perp must be non-zero")
    return _out(g / k ** 2)


def exb_shearing_rate_from_velocity_gradient(dv_E_dr):
    r"""Local E-cross-B shearing rate: the magnitude of the radial derivative of the flow.

    $$\gamma_E = \left|\frac{\partial v_E}{\partial r}\right|$$

    Parameters
    ----------
    dv_E_dr : float or array-like
        Radial derivative of the perpendicular (binormal) $E\times B$ velocity,
        taken along the radial coordinate $r$ in metres [1/s].

    Returns
    -------
    float or np.ndarray
        $\gamma_E$ [1/s].

    Raises
    ------
    ValueError
        Non-finite input.

    Convention
    ----------
    The simplest slab definition. The derivative is with respect to the
    coordinate the caller states, in metres; differentiating a profile and
    converting from a flux label ($\rho$, $\psi$) is a procedure. The
    Hahm--Burrell rate $(RB_\theta)^2/B\,\partial_\psi(E_r/RB_\theta)$ and
    code-specific normalised rates are different quantities and need their
    own identities.

    Limitations
    -----------
    No geometry factor and no sign; it does not decide whether turbulence is
    suppressed (see :func:`shear_to_growth_rate_ratio`).

    References
    ----------
    .. [1] H. Biglari, P. H. Diamond and P. W. Terry, Phys. Fluids B 2 (1990) 1.
    .. [2] T. S. Hahm and K. H. Burrell, Phys. Plasmas 2 (1995) 1648.
    """
    return _out(np.abs(_finite(dv_E_dr, "dv_E_dr")))


def shear_to_growth_rate_ratio(gamma_E, gamma_lin):
    r"""Ratio of the E-cross-B shearing rate to a linear growth rate.

    $$\frac{\gamma_E}{\gamma_\mathrm{lin}}$$

    Parameters
    ----------
    gamma_E : float or array-like
        Shearing rate, non-negative [1/s].
    gamma_lin : float or array-like
        Linear growth rate, positive [1/s].

    Returns
    -------
    float or np.ndarray
        $\gamma_E/\gamma_\mathrm{lin}$ [-].

    Raises
    ------
    ValueError
        Non-finite input, a negative shearing rate or a non-positive growth rate.

    Convention
    ----------
    Both rates in the same unit and taken at the same location; the rate
    definitions (which $\gamma_E$, the maximum or a mode's $\gamma$) are the
    caller's.

    Limitations
    -----------
    A ratio, not a classifier. The rule of thumb that turbulence is suppressed
    near $\gamma_E/\gamma_\mathrm{lin} \gtrsim 1$ depends on the definitions
    and the geometry, and is not encoded here.

    References
    ----------
    .. [1] R. E. Waltz, G. D. Kerbel and J. Milovich, Phys. Plasmas 1 (1994) 2229.
    """
    ge = _finite(gamma_E, "gamma_E")
    if np.any(ge < 0.0):
        raise ValueError("gamma_E must be non-negative")
    return _out(ge / _positive(gamma_lin, "gamma_lin"))


def reynolds_stress_from_velocity_fluctuations(v_r_fluct, v_theta_fluct, axis=-1):
    r"""Turbulent Reynolds stress: the correlation of radial and poloidal velocity fluctuations.

    $$R_{r\theta} = \langle\tilde v_r\,\tilde v_\theta\rangle$$

    Parameters
    ----------
    v_r_fluct : array-like
        Radial velocity fluctuation, positive outward [m/s].
    v_theta_fluct : array-like
        Poloidal (binormal) velocity fluctuation at the same samples [m/s].
    axis : int
        Axis that ``<.>`` averages over [-].

    Returns
    -------
    float or np.ndarray
        $R_{r\theta}$, the radial flux of poloidal momentum per unit mass density [m^2/s^2].

    Raises
    ------
    ValueError
        Non-finite input, inputs of different shapes, or a missing or empty averaging axis.

    Convention
    ----------
    Kinematic stress (per unit mass density). Positive $R_{r\theta}$ carries
    positive $\theta$-momentum outward; the direction of $\hat\theta$ is the
    caller's, and reversing it reverses the sign. As for the particle flux,
    the inputs are fluctuating parts.

    Physical interpretation
    -----------------------
    Tilted eddies have correlated $\tilde v_r$ and $\tilde v_\theta$; their
    stress, and its radial gradient, is how turbulence drives a sheared
    zonal flow.

    References
    ----------
    .. [1] P. H. Diamond and Y.-B. Kim, Phys. Fluids B 3 (1991) 1626.
    .. [2] P. H. Diamond, S.-I. Itoh, K. Itoh and T. S. Hahm,
           Plasma Phys. Control. Fusion 47 (2005) R35.
    """
    return _ensemble_mean(v_r_fluct, v_theta_fluct, axis, ("v_r_fluct", "v_theta_fluct"))


def zonal_flow_drive_from_reynolds_stress_gradient(dR_dr, zonal_velocity=0.0, damping_rate=0.0):
    r"""Rate of change of a zonal flow driven by the Reynolds-stress divergence.

    $$\frac{\partial V_\mathrm{ZF}}{\partial t} = -\frac{\partial R_{r\theta}}{\partial r} - \mu\,V_\mathrm{ZF}$$

    Parameters
    ----------
    dR_dr : float or array-like
        Radial derivative of the Reynolds stress $R_{r\theta}$, along $r$ in
        metres [m/s^2].
    zonal_velocity : float or array-like
        Zonal-flow velocity $V_\mathrm{ZF}$; enters only through the damping [m/s].
    damping_rate : float or array-like
        Damping rate $\mu$, non-negative; 0 (the default) leaves the drive
        undamped [1/s].

    Returns
    -------
    float or np.ndarray
        $\partial V_\mathrm{ZF}/\partial t$ [m/s^2].

    Raises
    ------
    ValueError
        Non-finite input or a negative damping rate.

    Convention
    ----------
    Slab form of the flux-surface-averaged poloidal momentum balance; the
    damping is a model choice the caller makes explicit (collisional or
    neoclassical damping, or none). This is not the predator--prey model
    above, which describes intensities, not the flow itself.

    References
    ----------
    .. [1] P. H. Diamond and Y.-B. Kim, Phys. Fluids B 3 (1991) 1626.
    .. [2] P. H. Diamond, S.-I. Itoh, K. Itoh and T. S. Hahm,
           Plasma Phys. Control. Fusion 47 (2005) R35.
    """
    mu = _finite(damping_rate, "damping_rate")
    if np.any(mu < 0.0):
        raise ValueError("damping_rate must be non-negative")
    return _out(-_finite(dR_dr, "dR_dr") - mu * _finite(zonal_velocity, "zonal_velocity"))
