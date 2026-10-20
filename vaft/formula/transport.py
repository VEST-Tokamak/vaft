r"""Canonical transport reference relations: diffusion, constitutive fluxes, Bohm and gyro-Bohm scales.

Definitions only (issue #1112). A diffusivity here is either the definition of
one -- from a mean-square displacement or a velocity autocorrelation -- or a
reference *scale* against which a measured or simulated transport coefficient
is compared; none is a prediction of the anomalous transport of a plasma.
Constitutive fluxes go forward, from coefficients to a flux: a flux alone does
not determine a unique pair $(D, V)$, and nothing here inverts one. The
complete Braginskii heat diffusivities are
:func:`vaft.formula.kinetic.classical_electron_heat_diffusivity_from_T_e_B`
and its ion counterpart; the gyroradius, sound speed and $\rho_*$ have their
own owners in :mod:`vaft.formula.ordering`, :mod:`vaft.formula.sol` and
:mod:`vaft.formula.equilibrium` (``rho_star_from_M_T_B_R_epsilon``),
and are inputs here.

Notation
--------
D           : particle diffusivity                                     [m^2/s]
V           : convection (pinch) velocity                              [m/s]
chi         : heat diffusivity                                         [m^2/s]
n           : density                                                  [m^-3]
T           : temperature                                              [eV]
nu          : collision frequency                                      [1/s]
rho         : gyroradius (the step of a classical random walk)         [m]
rho_s       : ion-sound gyroradius                                     [m]
c_s         : ion sound speed                                          [m/s]
L_ref       : normalising length of the gyro-Bohm scale                [m]
Gamma       : particle flux                                            [m^-2 s^-1]
q, Q        : heat (energy) flux                                       [W/m^2]

Conventions
-----------
$$\Gamma = -D\,\partial_r n + nV, \qquad q = -n\chi\,\partial_r T$$

$r$ is the radial coordinate the gradient is taken along, in metres, and an
outward flux is positive; a positive $V$ is outward convection, so an inward
pinch has $V < 0$. Converting a gradient in a flux label ($\rho$,
$\psi_N$) to metres is the caller's job (#551 / #563). Temperatures are in
eV; a flux carries the factor $e$ that turns eV into joules.

References
----------
.. [1] G. I. Taylor, Proc. London Math. Soc. s2-20 (1922) 196.
.. [2] D. Bohm, in *The Characteristics of Electrical Discharges in Magnetic
       Fields*, ed. A. Guthrie and R. K. Wakerling, McGraw-Hill (1949), Ch. 2.
.. [3] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
.. [4] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Ch. 4.
"""

from __future__ import annotations

import numpy as np

from .constants import QE
from .utils import trapz_integral

__all__ = [
    "bohm_diffusivity_from_T_B",
    "classical_diffusivity_from_nu_rho",
    "conductive_heat_flux_from_n_chi_grad_T",
    "diffusivity_from_mean_square_displacement",
    "diffusivity_from_velocity_autocorrelation",
    "gyrobohm_diffusivity_from_rho_s_c_s_L_ref",
    "gyrobohm_energy_flux_from_n_T_c_s_rho_s_L_ref",
    "gyrobohm_particle_flux_from_n_c_s_rho_s_L_ref",
    "particle_flux_from_D_grad_n_V_n",
]


def _out(value):
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


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


def _non_negative(value, name):
    array = _finite(value, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _nonzero(value, name):
    array = _finite(value, name)
    if np.any(array == 0.0):
        raise ValueError(f"{name} must be non-zero")
    return array


def diffusivity_from_mean_square_displacement(msd_rate, dimensions=1):
    r"""Diffusivity from the growth rate of the mean-square displacement (Einstein relation).

    $$D = \frac{1}{2d}\,\frac{d\langle(\Delta r)^2\rangle}{dt}$$

    Parameters
    ----------
    msd_rate : float or array-like
        Rate of growth of the mean-square displacement $d\langle(\Delta r)^2\rangle/dt$,
        non-negative [m^2/s].
    dimensions : int
        Number $d$ of independent directions the displacement is summed over:
        1 for a radial displacement alone [-].

    Returns
    -------
    float or np.ndarray
        $D$ [m^2/s].

    Raises
    ------
    ValueError
        A non-finite or negative rate, or ``dimensions`` not a positive integer.

    Convention
    ----------
    $(\Delta r)^2$ is summed over the $d$ directions, so a radial-only
    displacement takes $d = 1$; the factor $1/(2d)$ is explicit, never hidden
    in the rate. The rate is the long-time (diffusive) slope; estimating it
    from a trajectory ensemble is a procedure, not part of this definition.

    Physical interpretation
    -----------------------
    For a random walk of step $\Delta$ every $\tau$, $\langle(\Delta r)^2\rangle
    = (\Delta^2/\tau)\,t$ in one dimension, so $D = \Delta^2/(2\tau)$: the
    reference behind every $D \sim (\text{step})^2/(\text{time})$ estimate.

    Validity
    --------
    Only where the mean-square displacement grows linearly in time. Ballistic
    ($\propto t^2$) or sub-/super-diffusive growth has no single $D$.

    References
    ----------
    .. [1] G. I. Taylor, Proc. London Math. Soc. s2-20 (1922) 196.
    """
    rate = _non_negative(msd_rate, "msd_rate")
    if not isinstance(dimensions, (int, np.integer)) or isinstance(dimensions, bool) or dimensions < 1:
        raise ValueError("dimensions must be a positive integer")
    return _out(rate / (2.0 * int(dimensions)))


def diffusivity_from_velocity_autocorrelation(lag, autocorrelation):
    r"""Diffusivity as the time integral of the velocity autocorrelation (Taylor--Green--Kubo).

    $$D = \int_0^\infty \langle v_r(0)\,v_r(\tau)\rangle\,d\tau$$

    Parameters
    ----------
    lag : array-like
        Lags $\tau$, starting at 0 and increasing [s].
    autocorrelation : array-like
        Velocity autocorrelation $\langle v_r(0)v_r(\tau)\rangle$ at the lags,
        not normalised [m^2/s^2].

    Returns
    -------
    float
        $D$ over the sampled lags [m^2/s].

    Raises
    ------
    ValueError
        Non-finite input, mismatched lengths, fewer than two lags, a first lag
        that is not zero, or lags that do not increase.

    Convention
    ----------
    One velocity component, so the result is the diffusivity along it with no
    dimensional factor. The upper limit is the last lag supplied: the
    integral is truncated there, by the trapezoidal rule.

    Physical interpretation
    -----------------------
    For $\langle v_r(0)v_r(\tau)\rangle = \langle v_r^2\rangle e^{-\tau/\tau_c}$,
    $D = \langle v_r^2\rangle\tau_c$: the velocity variance times the
    correlation time.

    Validity
    --------
    Stationary, homogeneous fluctuations whose autocorrelation decays well
    within the sampled lags; a slowly decaying or oscillating tail makes the
    truncated integral depend on the upper limit.

    Numerical notes
    ---------------
    :func:`vaft.formula.utils.trapz_integral`, second-order in the lag spacing.

    References
    ----------
    .. [1] G. I. Taylor, Proc. London Math. Soc. s2-20 (1922) 196.
    .. [2] R. Kubo, J. Phys. Soc. Jpn. 12 (1957) 570.
    """
    tau = _finite(lag, "lag")
    c = _finite(autocorrelation, "autocorrelation")
    if tau.ndim != 1 or tau.shape != c.shape or tau.size < 2:
        raise ValueError("lag and autocorrelation must be 1-D, of equal length, with at least two samples")
    if tau[0] != 0.0 or np.any(np.diff(tau) <= 0.0):
        raise ValueError("lag must start at 0 and increase")
    return float(trapz_integral(tau, c))


def particle_flux_from_D_grad_n_V_n(D, grad_n, n, V):
    r"""Radial particle flux from a diffusivity and a convection velocity.

    $$\Gamma = -D\,\partial_r n + nV$$

    Parameters
    ----------
    D : float or array-like
        Diffusivity, non-negative [m^2/s].
    grad_n : float or array-like
        Radial density gradient $\partial_r n$ [m^-4].
    n : float or array-like
        Density, non-negative [m^-3].
    V : float or array-like
        Convection velocity, positive outward [m/s].

    Returns
    -------
    float or np.ndarray
        $\Gamma$, positive outward [m^-2 s^-1].

    Raises
    ------
    ValueError
        Non-finite input, or a negative diffusivity or density.

    Convention
    ----------
    $r$ increases outward and the gradient is taken along it, in metres. A
    peaked profile ($\partial_r n < 0$) with $D > 0$ diffuses outward; an
    inward pinch is $V < 0$.

    Physical interpretation
    -----------------------
    A forward definition of the flux. The inverse is not unique: one flux at
    one gradient fixes only $-D\,\partial_r n + nV$, not $D$ and $V$ apart;
    separating them needs transient or modulated data, which is a procedure
    and not this formula.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Ch. 4.
    """
    d = _non_negative(D, "D")
    g = _finite(grad_n, "grad_n")
    density = _non_negative(n, "n")
    v = _finite(V, "V")
    return _out(-d * g + density * v)


def conductive_heat_flux_from_n_chi_grad_T(n, chi, grad_T):
    r"""Radial conductive heat flux from a heat diffusivity (Fourier's law).

    $$q_\mathrm{cond} = -n\chi\,\partial_r T$$

    Parameters
    ----------
    n : float or array-like
        Density of the species carrying the heat, non-negative [m^-3].
    chi : float or array-like
        Heat diffusivity, non-negative [m^2/s].
    grad_T : float or array-like
        Radial temperature gradient $\partial_r T$ [eV/m].

    Returns
    -------
    float or np.ndarray
        $q_\mathrm{cond}$, positive outward [W/m^2].

    Raises
    ------
    ValueError
        Non-finite input, or a negative density or diffusivity.

    Convention
    ----------
    Conduction only: the convective part $\tfrac{5}{2}T\Gamma$ (or
    $\tfrac{3}{2}T\Gamma$, depending on the convention of the total energy
    flux) is not included, and neither are off-diagonal terms of the
    transport matrix. The factor $e$ converts eV to joules.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Ch. 4.
    """
    density = _non_negative(n, "n")
    x = _non_negative(chi, "chi")
    g = _finite(grad_T, "grad_T")
    return _out(-density * x * g * QE)


def classical_diffusivity_from_nu_rho(nu, rho):
    r"""Classical random-walk reference diffusivity: one gyroradius per collision.

    $$D_\mathrm{cl} \sim \nu\rho^2$$

    Parameters
    ----------
    nu : float or array-like
        Collision frequency that decorrelates the gyro-orbit, non-negative [1/s].
    rho : float or array-like
        Gyroradius, the random-walk step, positive [m].

    Returns
    -------
    float or np.ndarray
        $\nu\rho^2$ [m^2/s].

    Raises
    ------
    ValueError
        Non-finite input, a negative frequency or a non-positive gyroradius.

    Convention
    ----------
    The reference scale without an order-unity prefactor. Which collision
    frequency and which species' gyroradius enter is the caller's choice and
    sets the meaning: electron collisions with the electron gyroradius give
    the classical particle diffusivity, ion--ion collisions with the ion
    gyroradius the classical ion heat diffusivity scale.

    Physical interpretation
    -----------------------
    The random walk of :func:`diffusivity_from_mean_square_displacement` with
    step $\rho$ and step time $1/\nu$.

    Limitations
    -----------
    An order-of-magnitude scale, not a classical transport model: the full
    Braginskii coefficients, with their numerical factors and cross terms,
    are :func:`vaft.formula.kinetic.classical_electron_heat_diffusivity_from_T_e_B`
    and :func:`vaft.formula.kinetic.classical_ion_heat_diffusivity_from_T_i_B`.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Ch. 4.
    """
    return _out(_non_negative(nu, "nu") * _positive(rho, "rho") ** 2)


def bohm_diffusivity_from_T_B(T, B, Z=1.0, coefficient=1.0 / 16.0):
    r"""Bohm diffusivity: the reference scale of strongly anomalous transport.

    $$D_B = C_B\,\frac{T}{|q|B}$$

    Parameters
    ----------
    T : float or array-like
        Temperature, positive [eV].
    B : float or array-like
        Field strength, positive [T].
    Z : float or array-like
        Charge number of the species, $|q| = Ze$, positive [-].
    coefficient : float
        $C_B$, positive; Bohm's value $1/16$ by default [-].

    Returns
    -------
    float or np.ndarray
        $D_B$ [m^2/s].

    Raises
    ------
    ValueError
        A non-positive or non-finite input.

    Convention
    ----------
    With $T$ in eV and $|q| = Ze$, $T/(|q|B) = T_\mathrm{eV}/(ZB)$ in
    m^2/s. The coefficient is explicit: $C_B = 1/16$ is Bohm's empirical
    value, and comparisons that use $C_B = 1$ must pass it.

    Validity
    --------
    Empirical fit. Bohm's 1949 arc-discharge observation, kept as a reference
    scale: $D_B \propto T/B$ has no gyroradius dependence, which is what
    separates Bohm from gyro-Bohm scaling.

    References
    ----------
    .. [1] D. Bohm, in *The Characteristics of Electrical Discharges in Magnetic
           Fields*, ed. A. Guthrie and R. K. Wakerling, McGraw-Hill (1949), Ch. 2.
    """
    c = _positive(coefficient, "coefficient")
    return _out(c * _positive(T, "T") / (_positive(Z, "Z") * _positive(B, "B")))


def gyrobohm_diffusivity_from_rho_s_c_s_L_ref(rho_s, c_s, L_ref):
    r"""Gyro-Bohm diffusivity: the Bohm scale reduced by one power of $\rho_s/L_\mathrm{ref}$.

    $$\chi_\mathrm{gB} = \frac{\rho_s^2 c_s}{L_\mathrm{ref}}$$

    Parameters
    ----------
    rho_s : float or array-like
        Ion-sound gyroradius, non-zero; its sign (that of the field) does not
        matter [m].
    c_s : float or array-like
        Ion sound speed, positive [m/s].
    L_ref : float or array-like
        Normalising length, positive [m].

    Returns
    -------
    float or np.ndarray
        $\chi_\mathrm{gB}$ [m^2/s].

    Raises
    ------
    ValueError
        Non-finite input, a zero gyroradius, or a non-positive speed or length.

    Convention
    ----------
    The value depends on which $\rho_s$, $c_s$ and $L_\mathrm{ref}$ are
    passed; GACODE uses $c_s = \sqrt{T_e/m_D}$, $\rho_s = c_s/(eB_\mathrm{unit}/m_D)$
    and $L_\mathrm{ref} = a$ (see
    :class:`vaft.code.gacode.tglf.inputs.TGLFNormalisation`). The reusable
    owners are :func:`vaft.formula.ordering.sound_gyroradius` and
    :func:`vaft.formula.sol.ion_sound_speed`.

    Physical interpretation
    -----------------------
    $\chi_\mathrm{gB} = (\rho_s/L_\mathrm{ref})\,\rho_s c_s$: transport by
    eddies of size $\rho_s$ decorrelating at $c_s/L_\mathrm{ref}$, smaller
    than the Bohm scale $\rho_s c_s$ by $\rho_* = \rho_s/L_\mathrm{ref}$.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    r = _nonzero(rho_s, "rho_s")
    return _out(r ** 2 * _positive(c_s, "c_s") / _positive(L_ref, "L_ref"))


def gyrobohm_particle_flux_from_n_c_s_rho_s_L_ref(n, c_s, rho_s, L_ref):
    r"""Gyro-Bohm particle-flux unit.

    $$\Gamma_\mathrm{gB} = n\,c_s\left(\frac{\rho_s}{L_\mathrm{ref}}\right)^2$$

    Parameters
    ----------
    n : float or array-like
        Reference density, positive [m^-3].
    c_s : float or array-like
        Ion sound speed, positive [m/s].
    rho_s : float or array-like
        Ion-sound gyroradius, non-zero [m].
    L_ref : float or array-like
        Normalising length, positive [m].

    Returns
    -------
    float or np.ndarray
        $\Gamma_\mathrm{gB}$ [m^-2 s^-1].

    Raises
    ------
    ValueError
        Non-finite input, a zero gyroradius, or a non-positive density, speed or length.

    Convention
    ----------
    GACODE's ``gamma_gb`` with $n = n_e$ and $L_\mathrm{ref} = a$; equal to
    ``TGLFNormalisation.particle_flux`` under the same inputs. A flux in
    gyro-Bohm units is the SI flux divided by this.

    References
    ----------
    .. [1] J. Candy, E. A. Belli and R. V. Bravenec, J. Comput. Phys. 324 (2016) 73.
    """
    ratio = _nonzero(rho_s, "rho_s") / _positive(L_ref, "L_ref")
    return _out(_positive(n, "n") * _positive(c_s, "c_s") * ratio ** 2)


def gyrobohm_energy_flux_from_n_T_c_s_rho_s_L_ref(n, T, c_s, rho_s, L_ref):
    r"""Gyro-Bohm energy-flux unit.

    $$Q_\mathrm{gB} = n\,T\,c_s\left(\frac{\rho_s}{L_\mathrm{ref}}\right)^2$$

    Parameters
    ----------
    n : float or array-like
        Reference density, positive [m^-3].
    T : float or array-like
        Reference temperature, positive [eV].
    c_s : float or array-like
        Ion sound speed, positive [m/s].
    rho_s : float or array-like
        Ion-sound gyroradius, non-zero [m].
    L_ref : float or array-like
        Normalising length, positive [m].

    Returns
    -------
    float or np.ndarray
        $Q_\mathrm{gB}$ [W/m^2].

    Raises
    ------
    ValueError
        Non-finite input, a zero gyroradius, or a non-positive density,
        temperature, speed or length.

    Convention
    ----------
    GACODE's ``q_gb`` with $n = n_e$, $T = T_e$ and $L_\mathrm{ref} = a$;
    equal to ``TGLFNormalisation.energy_flux`` under the same inputs (that
    class holds $T_e$ in joules, this function takes eV).

    References
    ----------
    .. [1] J. Candy, E. A. Belli and R. V. Bravenec, J. Comput. Phys. 324 (2016) 73.
    """
    gamma = gyrobohm_particle_flux_from_n_c_s_rho_s_L_ref(n, c_s, rho_s, L_ref)
    return _out(gamma * _positive(T, "T") * QE)
