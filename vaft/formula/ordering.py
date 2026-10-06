r"""Asymptotic ordering parameters: the scale separations reduced models assume.

A Lundquist number, an ion skin depth over a length, a Knudsen number or a
magnetization is not a performance figure: it is the small or large
parameter a model's derivation expands in. These kernels compute them from
explicit physical inputs -- the characteristic length, the resistivity, the
collision time are always arguments, never defaults -- so a caller cannot
silently substitute a global scale for a layer scale (#1627). Whether an
ordering is satisfied is not decided here; that is the ordering contracts'
job.

Notation
--------
L        : characteristic length the ordering is taken against   [m]
v_A      : Alfven speed                                            [m/s]
eta      : resistivity                                             [Ohm m]
tau_A    : Alfven time L / v_A                                     [s]
tau_R    : resistive diffusion time mu_0 L^2 / eta                 [s]
S        : Lundquist number tau_R / tau_A                          [-]
d_s      : inertial (skin) length c / omega_ps of species s        [m]
rho_s    : ion-sound gyroradius c_s / Omega_i                      [m]
v_t      : thermal speed sqrt(T / m)                               [m/s]
tau_e, tau_i : Braginskii electron and ion collision times         [s]
lambda   : mean free path v_t tau                                  [m]
Kn       : Knudsen number lambda / L                               [-]

Conventions
-----------
Temperatures are in eV. The thermal speed is $\sqrt{T/m}$ (NRL formulary);
some texts use $\sqrt{2T/m}$, which multiplies every mean free path and
Knudsen number by $\sqrt 2$. The collision times are Braginskii's, as the NRL
formulary gives them, with the Coulomb logarithm an explicit argument.

References
----------
.. [1] S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1,
       Consultants Bureau (1965), p. 205.
.. [2] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
.. [3] D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge University
       Press (2000), Ch. 1.
"""

import numpy as np

from .constants import C_LIGHT, MU0, QE
from .waves import plasma_frequency

__all__ = [
    "alfven_time",
    "resistive_diffusion_time",
    "lundquist_number",
    "magnetic_reynolds_number",
    "inertial_length",
    "sound_gyroradius",
    "thermal_speed",
    "braginskii_electron_collision_time",
    "braginskii_ion_collision_time",
    "mean_free_path",
    "knudsen_number",
    "magnetization",
    "evolution_time",
]


def _array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, not {value!r}")
    return array


def _positive(value, name: str) -> np.ndarray:
    array = _array(value, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive, not {value!r}")
    return array


def _out(result):
    result = np.asarray(result, dtype=float)
    return float(result) if result.ndim == 0 else result


def alfven_time(L, v_A):
    r"""Alfven transit time across a characteristic length.

    $$\tau_A = \frac{L}{v_A}$$

    Parameters
    ----------
    L : float or np.ndarray
        Characteristic length, positive [m].
    v_A : float or np.ndarray
        Alfven speed, positive (``vaft.formula.stability.v_alfven_from_B_n_mi``) [m/s].

    Returns
    -------
    float or np.ndarray
        $\tau_A$ [s].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    $L$ is the caller's choice and names the ordering: the minor radius for a
    global ordering, a layer width for a tearing-layer one. Some texts use
    $qR_0/v_A$ (the shear-Alfven transit along the field); pass $L = qR_0$
    for that.

    Physical interpretation
    -----------------------
    The time on which ideal-MHD forces rebalance across $L$: equilibrium is a
    sequence of states only if the plasma evolves much more slowly than this.

    References
    ----------
    .. [1] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014), Ch. 2.
    """
    return _out(_positive(L, "L") / _positive(v_A, "v_A"))


def resistive_diffusion_time(L, eta):
    r"""Resistive diffusion time of magnetic field across a characteristic length.

    $$\tau_R = \frac{\mu_0L^2}{\eta}$$

    Parameters
    ----------
    L : float or np.ndarray
        Characteristic length, positive [m].
    eta : float or np.ndarray
        Resistivity, positive, e.g. ``spitzer_resistivity_from_T_e_Z_eff_ln_Lambda`` [Ohm m].

    Returns
    -------
    float or np.ndarray
        $\tau_R$ [s].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    No geometric factor: a cylindrical current profile relaxes on
    $\mu_0a^2/(\pi^2\eta)$-like times, so treat $\tau_R$ as an ordering scale,
    not a prediction of the relaxation time.

    Physical interpretation
    -----------------------
    How long resistivity needs to diffuse field and current across $L$: a
    current profile younger than this has not relaxed resistively.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 2.11.
    """
    L = _positive(L, "L")
    return _out(MU0 * L ** 2 / _positive(eta, "eta"))


def lundquist_number(L, v_A, eta):
    r"""Lundquist number: resistive diffusion time over Alfven time.

    $$S = \frac{\tau_R}{\tau_A} = \frac{\mu_0Lv_A}{\eta}$$

    Parameters
    ----------
    L : float or np.ndarray
        Characteristic length, positive [m].
    v_A : float or np.ndarray
        Alfven speed, positive [m/s].
    eta : float or np.ndarray
        Resistivity, positive [Ohm m].

    Returns
    -------
    float or np.ndarray
        $S$ [-].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    $S$ scales with $L$: a global $S$ (with $L = a$) and a sheet or layer $S$
    (with its own thickness or length) are different numbers. Name the
    length when reporting it.

    Physical interpretation
    -----------------------
    $S \gg 1$ separates fast Alfvenic dynamics from slow resistive diffusion
    at that scale: it supports ideal MHD **globally**, and says nothing about
    thin resonant layers, where resistivity matters however large $S$ is.

    References
    ----------
    .. [1] D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge University
           Press (2000), Ch. 1.
    """
    L = _positive(L, "L")
    return _out(MU0 * L * _positive(v_A, "v_A") / _positive(eta, "eta"))


def magnetic_reynolds_number(V, L, eta):
    r"""Magnetic Reynolds number: flux advection against resistive diffusion.

    $$R_m = \frac{\mu_0VL}{\eta}$$

    Parameters
    ----------
    V : float or np.ndarray
        Characteristic flow speed, non-negative [m/s].
    L : float or np.ndarray
        Characteristic length, positive [m].
    eta : float or np.ndarray
        Resistivity, positive [Ohm m].

    Returns
    -------
    float or np.ndarray
        $R_m$ [-].

    Raises
    ------
    ValueError
        ``V`` is negative, ``L`` or ``eta`` is not positive, or an input is
        not finite.

    Convention
    ----------
    $V$ is a *measured or modelled* flow; with $V = v_A$ this is the
    Lundquist number. Do not substitute an arbitrary speed to fill a table.

    Physical interpretation
    -----------------------
    $R_m \gg 1$: the field is frozen into the flow at that scale.

    References
    ----------
    .. [1] D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge University
           Press (2000), Ch. 1.
    """
    V = _array(V, "V")
    if np.any(V < 0.0):
        raise ValueError("V must be non-negative")
    return _out(MU0 * V * _positive(L, "L") / _positive(eta, "eta"))


def inertial_length(n, m, q=QE):
    r"""Inertial (skin) length of a species: the light speed over its plasma frequency.

    $$d_s = \frac{c}{\omega_{ps}} = \sqrt{\frac{m_s}{\mu_0n_sq_s^2}}$$

    Parameters
    ----------
    n : float or np.ndarray
        Species density, positive [m^-3].
    m : float or np.ndarray
        Species mass, positive [kg].
    q : float or np.ndarray
        Species charge, non-zero; the elementary charge by default [C].

    Returns
    -------
    float or np.ndarray
        $d_s$ [m].

    Raises
    ------
    ValueError
        ``n`` or ``m`` is not positive, ``q`` is zero, or an input is not
        finite.

    Convention
    ----------
    $d_i$ with the ion mass and density, $d_e$ with the electron's. The
    ordering parameter is $d_s/L$: a global $d_i/a$ and a layer $d_i/\delta$
    are different statements.

    Physical interpretation
    -----------------------
    Below $d_i$ ions decouple from the field and Hall physics enters; below
    $d_e$ electron inertia does. $d_i/L \ll 1$ is the scale separation
    single-fluid MHD needs.

    References
    ----------
    .. [1] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
    """
    n = _positive(n, "n")
    m = _positive(m, "m")
    q = _array(q, "q")
    if np.any(q == 0.0):
        raise ValueError("q must be non-zero")
    return _out(C_LIGHT / plasma_frequency(n, q, m))


def sound_gyroradius(T_e, m_i, B, Z=1.0):
    r"""Ion-sound gyroradius: the ion sound speed over the ion gyrofrequency.

    $$\rho_s = \frac{c_s}{\Omega_i} = \frac{\sqrt{m_iT_e/Z}}{eB}$$

    Parameters
    ----------
    T_e : float or np.ndarray
        Electron temperature, positive [eV].
    m_i : float or np.ndarray
        Ion mass, positive [kg].
    B : float or np.ndarray
        Field strength, positive [T].
    Z : float or np.ndarray
        Ion charge number, positive [-].

    Returns
    -------
    float or np.ndarray
        $\rho_s$ [m].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    Cold ions: $c_s = \sqrt{ZT_e/m_i}$ and $\Omega_i = ZeB/m_i$. With a finite
    ion temperature some texts put $T_e + T_i$ under the root.

    Physical interpretation
    -----------------------
    The drift-wave and FLR scale at the electron temperature; $\rho_s/L_T$ is
    the FLR ordering parameter of fluid and gyrofluid models.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735.
    """
    T = _positive(T_e, "T_e") * QE
    return _out(np.sqrt(_positive(m_i, "m_i") * T / _positive(Z, "Z")) / (QE * _positive(B, "B")))


def thermal_speed(T, m):
    r"""Thermal speed of a species.

    $$v_t = \sqrt{\frac{T}{m}}$$

    Parameters
    ----------
    T : float or np.ndarray
        Temperature, positive [eV].
    m : float or np.ndarray
        Mass, positive [kg].

    Returns
    -------
    float or np.ndarray
        $v_t$ [m/s].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    The NRL formulary's $\sqrt{T/m}$. The most-probable speed $\sqrt{2T/m}$ is
    $\sqrt2$ larger, and so is every mean free path built on it.

    Physical interpretation
    -----------------------
    The speed that, with a collision time, makes a mean free path.

    References
    ----------
    .. [1] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
    """
    return _out(np.sqrt(_positive(T, "T") * QE / _positive(m, "m")))


def braginskii_electron_collision_time(n_e, T_e, ln_Lambda, Z=1.0):
    r"""Braginskii electron collision time.

    $$\tau_e = 3.44\times10^{11}\,\frac{T_e^{3/2}}{Z\,n_e\ln\Lambda}$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density, positive [m^-3].
    T_e : float or np.ndarray
        Electron temperature, positive [eV].
    ln_Lambda : float or np.ndarray
        Coulomb logarithm, positive [-].
    Z : float or np.ndarray
        Ion charge number, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_e$ [s].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    NRL's $3.44\times10^5\,T_e^{3/2}/(n\ln\Lambda)$ with $n$ in cm$^{-3}$,
    converted to m$^{-3}$; ions of charge $Z$ with $n_iZ^2 = n_eZ$ give the
    $1/Z$. The Coulomb logarithm is an argument so its convention stays
    visible (``coulomb_logarithm_electron_sauter`` or
    ``coulomb_logarithm_from_n_T``).

    Physical interpretation
    -----------------------
    The time between momentum-changing electron collisions: with the thermal
    speed it sets the electron mean free path, with the gyrofrequency the
    electron magnetization.

    Assumptions
    -----------
    Maxwellian electrons, $\ln\Lambda \gg 1$.

    References
    ----------
    .. [1] S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1,
           Consultants Bureau (1965), p. 205.
    .. [2] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
    """
    T = _positive(T_e, "T_e")
    return _out(3.44e11 * T ** 1.5 / (_positive(Z, "Z") * _positive(n_e, "n_e") * _positive(ln_Lambda, "ln_Lambda")))


def braginskii_ion_collision_time(n_i, T_i, ln_Lambda, mass_number=1.0, Z=1.0):
    r"""Braginskii ion collision time.

    $$\tau_i = 2.09\times10^{13}\,\frac{\mu^{1/2}\,T_i^{3/2}}{Z^4n_i\ln\Lambda}$$

    Parameters
    ----------
    n_i : float or np.ndarray
        Ion density, positive [m^-3].
    T_i : float or np.ndarray
        Ion temperature, positive [eV].
    ln_Lambda : float or np.ndarray
        Ion Coulomb logarithm, positive [-].
    mass_number : float or np.ndarray
        Ion mass in proton masses, $\mu = m_i/m_p$, positive [-].
    Z : float or np.ndarray
        Ion charge number, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_i$ [s].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    NRL's $2.09\times10^7\,T_i^{3/2}\mu^{1/2}/(n\ln\Lambda)$ with $n$ in
    cm$^{-3}$, converted to m$^{-3}$, with Braginskii's $Z^4$ for a single
    ion species of charge $Z$.

    Physical interpretation
    -----------------------
    The ion-ion collision time: with the ion thermal speed it sets the ion
    mean free path, with the ion gyrofrequency the ion magnetization.

    Assumptions
    -----------
    One Maxwellian ion species, $\ln\Lambda \gg 1$.

    References
    ----------
    .. [1] S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1,
           Consultants Bureau (1965), p. 205.
    .. [2] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
    """
    T = _positive(T_i, "T_i")
    Z = _positive(Z, "Z")
    return _out(2.09e13 * np.sqrt(_positive(mass_number, "mass_number")) * T ** 1.5
                / (Z ** 4 * _positive(n_i, "n_i") * _positive(ln_Lambda, "ln_Lambda")))


def mean_free_path(v_t, tau):
    r"""Mean free path: thermal speed times collision time.

    $$\lambda = v_t\tau$$

    Parameters
    ----------
    v_t : float or np.ndarray
        Thermal speed, positive (``thermal_speed``) [m/s].
    tau : float or np.ndarray
        Collision time, positive [s].

    Returns
    -------
    float or np.ndarray
        $\lambda$ [m].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    Inherits the thermal-speed convention: with $\sqrt{2T/m}$ it is $\sqrt2$
    longer.

    Physical interpretation
    -----------------------
    How far a particle travels between collisions; against a gradient length
    it decides whether a local fluid closure holds.

    References
    ----------
    .. [1] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
    """
    return _out(_positive(v_t, "v_t") * _positive(tau, "tau"))


def knudsen_number(mean_free_path, L):
    r"""Knudsen number: mean free path over a characteristic length.

    $$Kn = \frac{\lambda}{L}$$

    Parameters
    ----------
    mean_free_path : float or np.ndarray
        Mean free path, positive [m].
    L : float or np.ndarray
        Characteristic length, positive: a gradient length $L_T$, $L_n$, or a
        parallel connection length [m].

    Returns
    -------
    float or np.ndarray
        $Kn$ [-].

    Raises
    ------
    ValueError
        An input is not positive and finite.

    Convention
    ----------
    Species- and scale-specific: $Kn_{e,T} = \lambda_e/L_{T_e}$,
    $Kn_{\parallel} = \lambda_e/L_\parallel$. The same mean free path gives a
    small $Kn$ against a long parallel length and a large one against a
    steep perpendicular gradient.

    Physical interpretation
    -----------------------
    $Kn \ll 1$ supports a local (Braginskii-type) collisional closure; as it
    grows, transport becomes non-local and kinetic.

    References
    ----------
    .. [1] S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1,
           Consultants Bureau (1965), p. 205.
    """
    return _out(_positive(mean_free_path, "mean_free_path") / _positive(L, "L"))


def magnetization(gyrofrequency, collision_time):
    r"""Magnetization of a species: gyro-orbits per collision time.

    $$\chi_s = |\Omega_s|\,\tau_s$$

    Parameters
    ----------
    gyrofrequency : float or np.ndarray
        Gyrofrequency, either sign, non-zero (``vaft.formula.particle.gyrofrequency``) [rad/s].
    collision_time : float or np.ndarray
        Collision time of the same species, positive [s].

    Returns
    -------
    float or np.ndarray
        $\chi_s = \Omega_s/\nu_s$ [-].

    Raises
    ------
    ValueError
        ``gyrofrequency`` is zero, ``collision_time`` is not positive, or an
        input is not finite.

    Convention
    ----------
    Signed gyrofrequencies are taken in magnitude. Braginskii's transport
    coefficients are written in this $\Omega\tau$.

    Physical interpretation
    -----------------------
    $\chi \gg 1$: particles gyrate many times between collisions, the
    strongly magnetized ordering behind anisotropic transport. Independent
    of the Knudsen number: a plasma can be strongly magnetized and
    collisionless along the field at once.

    References
    ----------
    .. [1] S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1,
           Consultants Bureau (1965), p. 205.
    """
    omega = _array(gyrofrequency, "gyrofrequency")
    if np.any(omega == 0.0):
        raise ValueError("gyrofrequency must be non-zero")
    return _out(np.abs(omega) * _positive(collision_time, "collision_time"))


def evolution_time(X, dX_dt):
    r"""Evolution time of a state variable from its rate of change.

    $$\tau_{evol} = \left|\frac{X}{dX/dt}\right|$$

    Parameters
    ----------
    X : float or np.ndarray
        State variable: $I_p$, stored energy, axis position, ... [any].
    dX_dt : float or np.ndarray
        Its time derivative, in the unit of ``X`` per second [any/s].

    Returns
    -------
    float or np.ndarray
        $\tau_{evol}$; infinite where $X$ is stationary [s].

    Raises
    ------
    ValueError
        An input is not finite.

    Convention
    ----------
    Positive by construction. A position $X$ needs a reference: use a
    displacement over a length, not a coordinate that passes through zero.

    Physical interpretation
    -----------------------
    $\tau_{evol}/\tau_A \gg 1$ is the quasi-static ordering behind treating a
    discharge as a sequence of equilibria -- a physical statement, separate
    from whether a reconstruction converged.

    References
    ----------
    .. [1] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014), Ch. 2.
    """
    X = _array(X, "X")
    rate = _array(dX_dt, "dX_dt")
    with np.errstate(divide="ignore"):
        result = np.where(rate == 0.0, np.inf, np.abs(X / np.where(rate == 0.0, 1.0, rate)))
    return _out(result)
