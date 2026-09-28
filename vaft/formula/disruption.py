"""
Disruption reference relations: thermal and current quench, induced field, runaway electrons.

The semi-analytic middle layer between detecting a disruption in the data
(``vaft.process.transients``) and simulating one (DREAM-like kinetic or
integrated codes): transparent reference models whose assumptions are easy
to state. The chain they describe is::

    thermal quench (T_e drops)  ->  resistivity rises  ->  current quench (L/R)
    ->  induced E_par  ->  runaway seed (Dreicer)  ->  avalanche  ->  RE current

Resistivity is ``spitzer_resistivity_from_T_e_Z_eff_ln_Lambda``, the plasma
resistance, inductance and L/R time are the ``startup`` lumped-circuit
relations; they are reused here, not restated.

Notation
--------
n_e       : electron density                                    [m^-3]
T_e       : electron temperature                                [eV]
E         : parallel electric field, magnitude                  [V/m]
E_c       : Connor-Hastie critical field                        [V/m]
E_D       : Dreicer field                                       [V/m]
Z_eff     : effective ion charge                                [-]
ln_Lambda : Coulomb logarithm (relativistic for E_c, thermal for E_D) [-]
L_p       : plasma self-inductance                              [H]
I_p       : plasma current                                      [A]
R0        : major radius                                        [m]
p_c       : critical momentum, in units of m_e c                [-]

Conventions
-----------
SI throughout, temperatures in eV. Fields are magnitudes along the current;
the induced field has the sign that opposes the current's decay, so a
falling current drives electrons that carry it. Every Coulomb logarithm is an
explicit argument -- the relativistic one ($\\approx 15$-$20$) belongs in
$E_c$ and the avalanche rate, the thermal one in $E_D$ and the Dreicer rate;
no default is hidden (#1188).

References
----------
.. [1] J. W. Connor and R. J. Hastie, Nucl. Fusion 15 (1975) 415.
.. [2] H. Dreicer, Phys. Rev. 115 (1959) 238.
.. [3] M. N. Rosenbluth and S. V. Putvinski, Nucl. Fusion 37 (1997) 1355.
.. [4] B. N. Breizman, P. Aleynikov, E. M. Hollmann and M. Lehnen,
       Nucl. Fusion 59 (2019) 083001 (review).
"""

import numpy as np

from .constants import C_LIGHT, EPS0, ME, QE

__all__ = [
    "thermal_quench_temperature",
    "current_quench_current",
    "inductive_parallel_electric_field",
    "connor_hastie_critical_field",
    "dreicer_field",
    "runaway_critical_momentum",
    "relativistic_collision_time",
    "dreicer_generation_rate",
    "avalanche_growth_rate",
    "avalanche_efolds_from_current_drop",
    "runaway_current_from_density",
]


def _positive(value, name: str):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return arr


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def thermal_quench_temperature(t, T_0, T_final, tau_TQ):
    r"""Prescribed exponential thermal quench of the electron temperature.

    $$T_e(t) = T_\mathrm{f} + (T_0 - T_\mathrm{f})\,e^{-t/\tau_\mathrm{TQ}},\quad t \ge 0;\qquad T_e = T_0,\ t < 0$$

    Parameters
    ----------
    t : float or np.ndarray
        Time from the start of the quench [s].
    T_0 : float
        Pre-disruption temperature [eV].
    T_final : float
        Post-quench temperature, positive and below ``T_0`` [eV].
    tau_TQ : float
        Quench e-folding time, positive [s].

    Returns
    -------
    float or np.ndarray
        $T_e(t)$ [eV].

    Raises
    ------
    ValueError
        A temperature or ``tau_TQ`` is not positive, or ``T_final > T_0``.

    Assumptions
    -----------
    A prescribed shape, not a transport or radiation calculation: real
    quenches are often two-stage (a fast conductive drop after stochastisation,
    then radiative cooling to a few eV) and not exponential.

    References
    ----------
    .. [1] T. C. Hender et al., Nucl. Fusion 47 (2007) S128, Sec. 3.
    """
    T_0 = float(_positive(T_0, "T_0"))
    T_final = float(_positive(T_final, "T_final"))
    tau_TQ = float(_positive(tau_TQ, "tau_TQ"))
    if T_final > T_0:
        raise ValueError("T_final must not exceed T_0")
    t = np.asarray(t, dtype=float)
    return _out(np.where(t < 0.0, T_0, T_final + (T_0 - T_final) * np.exp(-np.maximum(t, 0.0) / tau_TQ)))


def current_quench_current(t, I_0, tau_CQ):
    r"""Exponential (L/R) current quench of the plasma current.

    $$I_p(t) = I_0\,e^{-t/\tau_\mathrm{CQ}},\quad t \ge 0,\qquad \tau_\mathrm{CQ} = L_p/R_p$$

    Parameters
    ----------
    t : float or np.ndarray
        Time from the start of the current quench [s].
    I_0 : float
        Current at the start [A].
    tau_CQ : float
        L/R time, positive (``lr_time_from_L_R``) [s].

    Returns
    -------
    float or np.ndarray
        $I_p(t)$; $I_0$ for $t < 0$ [A].

    Raises
    ------
    ValueError
        ``tau_CQ`` is not positive.

    Assumptions
    -----------
    Constant $L_p$ and $R_p$ -- a cold, fixed-temperature plasma with no
    runaway current and no coupling to the vessel. With $R_p$ rising as the
    plasma cools, or runaways taking over the current, the decay is not a
    single exponential; the linear-decay fit used for CQ rates is another
    reduction.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.10.
    """
    tau_CQ = float(_positive(tau_CQ, "tau_CQ"))
    t = np.asarray(t, dtype=float)
    return _out(np.where(t < 0.0, I_0, I_0 * np.exp(-np.maximum(t, 0.0) / tau_CQ)))


def inductive_parallel_electric_field(L_p, dI_p_dt, R0):
    r"""Parallel electric field a changing plasma current induces in the plasma.

    $$E_\parallel = -\frac{L_p}{2\pi R_0}\,\frac{dI_p}{dt}$$

    Parameters
    ----------
    L_p : float
        Plasma self-inductance, positive [H].
    dI_p_dt : float or np.ndarray
        Rate of change of the plasma current [A/s].
    R0 : float
        Major radius, positive [m].

    Returns
    -------
    float or np.ndarray
        $E_\parallel$, positive along the current when it decays [V/m].

    Raises
    ------
    ValueError
        ``L_p`` or ``R0`` is not positive.

    Convention
    ----------
    Signed along the current: a decaying current ($dI_p/dt < 0$) gives
    $E_\parallel > 0$, the field that keeps pushing the current-carrying
    electrons. The flux linked with the vessel and coils is ignored, so this
    is the upper estimate an isolated plasma would see.

    Physical interpretation
    -----------------------
    The magnetic energy $\tfrac12 L_pI_p^2$ cannot vanish instantly: a fast
    current quench turns it into a loop voltage $L_p|dI_p/dt|$, which spread
    over the circumference $2\pi R_0$ is the field that can exceed $E_c$
    by orders of magnitude.

    Assumptions
    -----------
    Uniform field over the cross-section; lumped (0-D) inductance.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 7.10.
    """
    L_p = float(_positive(L_p, "L_p"))
    R0 = float(_positive(R0, "R0"))
    return _out(-L_p * np.asarray(dI_p_dt, dtype=float) / (2.0 * np.pi * R0))


def connor_hastie_critical_field(n_e, ln_Lambda):
    r"""Connor--Hastie critical field below which no electron runs away.

    $$E_c = \frac{n_e e^3\ln\Lambda}{4\pi\varepsilon_0^2 m_e c^2}$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density (free plus bound for a partially ionised plasma), positive [m^-3].
    ln_Lambda : float
        Relativistic Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $E_c$ [V/m].

    Raises
    ------
    ValueError
        ``n_e`` or ``ln_Lambda`` is not positive.

    Physical interpretation
    -----------------------
    The minimum of the collisional drag on an electron, reached at
    relativistic speed: below $E_c$ even the fastest electrons slow down. It
    is the threshold of the avalanche; synchrotron and bremsstrahlung losses
    and partial screening raise the effective threshold above it.

    References
    ----------
    .. [1] J. W. Connor and R. J. Hastie, Nucl. Fusion 15 (1975) 415.
    """
    n_e = _positive(n_e, "n_e")
    ln_Lambda = _positive(ln_Lambda, "ln_Lambda")
    return _out(n_e * QE**3 * ln_Lambda / (4.0 * np.pi * EPS0**2 * ME * C_LIGHT**2))


def dreicer_field(n_e, T_e, ln_Lambda):
    r"""Dreicer field: the drag on a thermal electron.

    $$E_D = \frac{n_e e^3\ln\Lambda}{4\pi\varepsilon_0^2 T_e} = E_c\,\frac{m_ec^2}{T_e}$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density, positive [m^-3].
    T_e : float or np.ndarray
        Electron temperature, positive [eV].
    ln_Lambda : float
        Thermal Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $E_D$ [V/m].

    Raises
    ------
    ValueError
        An argument is not positive.

    Convention
    ----------
    $T_e$ enters as an energy $eT_e$; some texts put $T_e/m_e$ as $v_{te}^2$
    with $v_{te} = \sqrt{2T_e/m_e}$, which changes the prefactor by two. This
    is the form of Connor & Hastie, consistent with $E_D/E_c = m_ec^2/T_e$.

    Physical interpretation
    -----------------------
    A field $E \gtrsim E_D$ accelerates the bulk; well below it only the
    tail beyond the critical velocity runs away, at the exponentially small
    Dreicer rate. At a few eV after a thermal quench $E_D$ is enormous, which
    is why hot-tail and avalanche usually matter more than Dreicer there.

    References
    ----------
    .. [1] H. Dreicer, Phys. Rev. 115 (1959) 238.
    .. [2] J. W. Connor and R. J. Hastie, Nucl. Fusion 15 (1975) 415.
    """
    n_e = _positive(n_e, "n_e")
    T_e = _positive(T_e, "T_e")
    ln_Lambda = _positive(ln_Lambda, "ln_Lambda")
    return _out(n_e * QE**3 * ln_Lambda / (4.0 * np.pi * EPS0**2 * QE * T_e))


def runaway_critical_momentum(E, E_c):
    r"""Critical momentum above which an electron runs away, in units of $m_ec$.

    $$p_c = \frac{1}{\sqrt{E/E_c - 1}},\qquad E > E_c$$

    Parameters
    ----------
    E : float or np.ndarray
        Parallel electric field, magnitude [V/m].
    E_c : float or np.ndarray
        Critical field (``connor_hastie_critical_field``), positive [V/m].

    Returns
    -------
    float or np.ndarray
        $p_c$; ``inf`` where $E \le E_c$ [-].

    Raises
    ------
    ValueError
        ``E_c`` is not positive.

    Assumptions
    -----------
    A test electron moving along the field with the relativistic drag
    $\propto 1 + 1/p^2$; pitch-angle scattering (the $Z_\mathrm{eff}$ factor)
    and radiation raise $p_c$.

    References
    ----------
    .. [1] B. N. Breizman et al., Nucl. Fusion 59 (2019) 083001, Sec. 2.
    """
    E_c = _positive(E_c, "E_c")
    ratio = np.asarray(E, dtype=float) / E_c
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(ratio > 1.0, 1.0 / np.sqrt(np.maximum(ratio - 1.0, 1e-300)), np.inf)
    return _out(p)


def relativistic_collision_time(n_e, ln_Lambda):
    r"""Collision time of a relativistic electron, $\tau_c = m_ec/(eE_c)$.

    $$\tau_c = \frac{4\pi\varepsilon_0^2m_e^2c^3}{n_ee^4\ln\Lambda}$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density, positive [m^-3].
    ln_Lambda : float
        Relativistic Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $\tau_c$ [s].

    Raises
    ------
    ValueError
        An argument is not positive.

    Physical interpretation
    -----------------------
    The time the critical field takes to give an electron momentum $m_ec$;
    the avalanche grows on $\tau_c\ln\Lambda$.

    References
    ----------
    .. [1] M. N. Rosenbluth and S. V. Putvinski, Nucl. Fusion 37 (1997) 1355.
    """
    return _out(ME * C_LIGHT / (QE * connor_hastie_critical_field(n_e, ln_Lambda)))


def dreicer_generation_rate(n_e, T_e, E, Z_eff, ln_Lambda, *, prefactor=1.0):
    r"""Dreicer (primary) runaway generation rate, Connor--Hastie asymptotic form.

    $$\frac{dn_\mathrm{RE}}{dt} = C\,n_e\nu_{ee}\left(\frac{E_D}{E}\right)^{\frac{3(1+Z)}{16}}
      \exp\left(-\frac{E_D}{4E} - \sqrt{\frac{(1+Z)E_D}{E}}\right),\qquad
      \nu_{ee} = \frac{n_ee^4\ln\Lambda}{4\pi\varepsilon_0^2m_e^2v_{te}^3}$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Electron density, positive [m^-3].
    T_e : float or np.ndarray
        Electron temperature, positive [eV].
    E : float or np.ndarray
        Parallel electric field, magnitude [V/m].
    Z_eff : float
        Effective ion charge, positive [-].
    ln_Lambda : float
        Thermal Coulomb logarithm, positive [-].
    prefactor : float, optional
        The order-unity $C$ of the asymptotic formula (fits to kinetic
        solutions put it between about 0.3 and 1) [-].

    Returns
    -------
    float or np.ndarray
        Primary generation rate; zero where $E \le 0$ [m^-3 s^-1].

    Raises
    ------
    ValueError
        A density, temperature, ``Z_eff`` or ``ln_Lambda`` is not positive.

    Convention
    ----------
    $v_{te} = \sqrt{2eT_e/m_e}$; $E_D$ is ``dreicer_field``. The prefactor is
    an explicit argument because published fits differ by a factor of a few.

    Assumptions
    -----------
    Steady state, $E \ll E_D$ (the asymptotic regime), non-relativistic
    thermal electrons, no hot tail; unreliable above $E/E_D \approx 0.1$.

    References
    ----------
    .. [1] J. W. Connor and R. J. Hastie, Nucl. Fusion 15 (1975) 415.
    """
    n_e = _positive(n_e, "n_e")
    T_e = _positive(T_e, "T_e")
    Z = float(_positive(Z_eff, "Z_eff"))
    ln_Lambda = _positive(ln_Lambda, "ln_Lambda")
    E = np.asarray(E, dtype=float)
    E_D = dreicer_field(n_e, T_e, ln_Lambda)
    v_te = np.sqrt(2.0 * QE * T_e / ME)
    nu_ee = n_e * QE**4 * ln_Lambda / (4.0 * np.pi * EPS0**2 * ME**2 * v_te**3)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x = np.where(E > 0.0, E_D / np.where(E > 0.0, E, 1.0), np.inf)
        rate = prefactor * n_e * nu_ee * x ** (3.0 * (1.0 + Z) / 16.0) * np.exp(-x / 4.0 - np.sqrt((1.0 + Z) * x))
    return _out(np.where(np.isfinite(rate), rate, 0.0))


def avalanche_growth_rate(E, E_c, Z_eff, tau_c, ln_Lambda):
    r"""Rosenbluth--Putvinski avalanche (secondary) growth rate of the runaway density.

    $$\gamma_\mathrm{av} = \frac{E/E_c - 1}{\tau_c\ln\Lambda}\sqrt{\frac{\pi}{3(Z_\mathrm{eff} + 5)}},\qquad
      \frac{dn_\mathrm{RE}}{dt} = \gamma_\mathrm{av}\,n_\mathrm{RE}$$

    Parameters
    ----------
    E : float or np.ndarray
        Parallel electric field, magnitude [V/m].
    E_c : float
        Critical field, positive [V/m].
    Z_eff : float
        Effective ion charge, positive [-].
    tau_c : float
        Relativistic collision time (``relativistic_collision_time``), positive [s].
    ln_Lambda : float
        Relativistic Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        $\gamma_\mathrm{av}$; zero where $E \le E_c$ [1/s].

    Raises
    ------
    ValueError
        ``E_c``, ``Z_eff``, ``tau_c`` or ``ln_Lambda`` is not positive.

    Physical interpretation
    -----------------------
    Close collisions of existing runaways knock thermal electrons above
    $p_c$: exponential growth from any seed, independent of $T_e$. Over a
    whole current quench it multiplies a seed by
    ``avalanche_efolds_from_current_drop`` e-folds, which is why a tiny
    Dreicer or hot-tail seed can carry most of the current afterwards.

    Assumptions
    -----------
    $E \gg E_c$ asymptote of Rosenbluth & Putvinski with a complete-screening
    $Z_\mathrm{eff}$; partial screening of impurities and radiation are not
    included. The rate is clipped at zero below $E_c$.

    References
    ----------
    .. [1] M. N. Rosenbluth and S. V. Putvinski, Nucl. Fusion 37 (1997) 1355.
    """
    E_c = _positive(E_c, "E_c")
    Z = _positive(Z_eff, "Z_eff")
    tau_c = _positive(tau_c, "tau_c")
    ln_Lambda = _positive(ln_Lambda, "ln_Lambda")
    excess = np.maximum(np.asarray(E, dtype=float) / E_c - 1.0, 0.0)
    return _out(excess / (tau_c * ln_Lambda) * np.sqrt(np.pi / (3.0 * (Z + 5.0))))


def avalanche_efolds_from_current_drop(delta_I_p, L_p, R0, Z_eff, ln_Lambda):
    r"""Number of avalanche e-folds a current quench provides, $E \gg E_c$.

    $$N = \int\gamma_\mathrm{av}\,dt \simeq \frac{e\,L_p\,\Delta I_p}{2\pi R_0\,m_ec\,\ln\Lambda}
      \sqrt{\frac{\pi}{3(Z_\mathrm{eff} + 5)}}$$

    Parameters
    ----------
    delta_I_p : float or np.ndarray
        Current lost in the quench, non-negative [A].
    L_p : float
        Plasma self-inductance, positive [H].
    R0 : float
        Major radius, positive [m].
    Z_eff : float
        Effective ion charge, positive [-].
    ln_Lambda : float
        Relativistic Coulomb logarithm, positive [-].

    Returns
    -------
    float or np.ndarray
        Number of e-folds; the seed is multiplied by $e^N$ [-].

    Raises
    ------
    ValueError
        ``delta_I_p`` is negative or another argument is not positive.

    Physical interpretation
    -----------------------
    $\int E\,dt = L_p\Delta I_p/(2\pi R_0)$ by ``inductive_parallel_electric_field``,
    and $E_c\tau_c = m_ec/e$, so the $E_c$ dependence cancels: the gain
    depends on the current and the inductance, not the density. With
    $L_p \approx \mu_0R_0(\ln(8R_0/a) - 2 + l_i/2)$ this is the familiar
    $N \propto I_p/(I_A\ln\Lambda)$, $I_A = 4\pi\varepsilon_0m_ec^3/e \approx 17$ kA:
    a few MA gives tens of e-folds.

    Assumptions
    -----------
    $E \gg E_c$ throughout (the $-1$ in $E/E_c - 1$ dropped), so this is an
    upper estimate; no runaway current feeding back on $E$; no vessel
    coupling.

    References
    ----------
    .. [1] M. N. Rosenbluth and S. V. Putvinski, Nucl. Fusion 37 (1997) 1355.
    """
    delta_I_p = np.asarray(delta_I_p, dtype=float)
    if np.any(delta_I_p < 0.0):
        raise ValueError("delta_I_p must be non-negative (the current lost)")
    L_p = float(_positive(L_p, "L_p"))
    R0 = float(_positive(R0, "R0"))
    Z = float(_positive(Z_eff, "Z_eff"))
    ln_Lambda = float(_positive(ln_Lambda, "ln_Lambda"))
    return _out(QE * L_p * delta_I_p / (2.0 * np.pi * R0 * ME * C_LIGHT * ln_Lambda)
                * np.sqrt(np.pi / (3.0 * (Z + 5.0))))


def runaway_current_from_density(n_RE, area):
    r"""Current carried by runaway electrons moving at the speed of light.

    $$I_\mathrm{RE} = e\,c\,n_\mathrm{RE}\,A$$

    Parameters
    ----------
    n_RE : float or np.ndarray
        Runaway-electron density, non-negative [m^-3].
    area : float
        Poloidal cross-section they occupy, positive [m^2].

    Returns
    -------
    float or np.ndarray
        $I_\mathrm{RE}$ [A].

    Raises
    ------
    ValueError
        ``n_RE`` is negative or ``area`` is not positive.

    Assumptions
    -----------
    Every runaway moves along $\mathbf B$ at $v \simeq c$, uniform over the
    area; pitch-angle spread lowers the parallel velocity somewhat.

    References
    ----------
    .. [1] B. N. Breizman et al., Nucl. Fusion 59 (2019) 083001, Sec. 1.
    """
    n_RE = np.asarray(n_RE, dtype=float)
    if np.any(n_RE < 0.0):
        raise ValueError("n_RE must be non-negative")
    area = float(_positive(area, "area"))
    return _out(QE * C_LIGHT * n_RE * area)
