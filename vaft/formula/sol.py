r"""Scrape-off layer: sound speed, sheath fluxes, parallel conduction and the target heat-flux profile.

The small, independently testable relations of the open-field-line edge
(issue #951, first slice): the ion sound speed with its closure explicit,
the sheath-limited particle flux and ion saturation current, the sheath heat
transmission, Spitzer--Harm parallel electron conduction and its integrated
two-point form, and the Eich target heat-flux profile with its integral width.
Reduced models built on them (the two-point model with loss factors,
equilibrium-derived connection length and flux expansion, named lambda_q
scalings) are later slices of #951; SOLPS-ITER and UEDGE are not replaced.

Notation
--------
T_e, T_i   : electron and ion temperature                      [eV]
n          : density at the location named (target t, upstream u) [m^-3]
m_i        : ion mass                                          [kg]
Z          : ion charge number                                 [-]
c_s        : ion sound speed                                   [m/s]
gamma      : sheath heat-transmission coefficient              [-]
q_par      : parallel heat-flux density                        [W/m^2]
L_par      : parallel connection length between named endpoints [m]
kappa_0    : Spitzer--Harm coefficient, kappa = kappa_0 T_e^{5/2} [W/(m eV^{7/2})]
lambda_q   : heat-flux decay length at the outer midplane (Eich fit); lambda_q f_x at the target [m]
S          : Gaussian spreading width at the target, not mapped [m]
lambda_int : integral power width, mapped to the outer midplane [m]

Conventions
-----------
Temperatures in electronvolts; "upstream" is the outer-midplane SOL unless a
function says otherwise, and is not the separatrix. The parallel coordinate
$s$ runs from upstream to the target. Every coefficient that is a model
choice -- $\gamma$, $\gamma_i$, $\kappa_0$ -- is an input or a documented,
overridable keyword, never a hidden constant.

References
----------
.. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
       IOP (2000), Ch. 2, 4 and 5.
.. [2] T. Eich et al., Phys. Rev. Lett. 107 (2011) 215001.
"""

import numpy as np

from .constants import QE

__all__ = [
    "ion_sound_speed",
    "sheath_particle_flux",
    "ion_saturation_current_density",
    "sheath_heat_flux",
    "spitzer_harm_parallel_heat_flux",
    "two_point_upstream_temperature",
    "eich_target_heat_flux_profile",
    "eich_integral_width",
    "radiative_condensation_growth_rate",
    "radiative_thermal_instability_growth_rate",
]

#: Spitzer--Harm electron conduction coefficient for Z = 1, kappa_0 in W/(m eV^{7/2}): Stangeby's working value;
#: 3.16 n T tau_e / m_e with the NRL tau_e and ln Lambda = 15 gives 2040
KAPPA_0E = 2000.0


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def _positive(value, name):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return arr


def _non_negative(value, name):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative and finite")
    return arr


def ion_sound_speed(T_e, T_i, m_i, *, Z=1.0, gamma_i=1.0):
    r"""Ion sound speed with the charge and the ion closure explicit.

    $$c_s = \sqrt{\frac{e\,(Z T_e + \gamma_i T_i)}{m_i}}$$

    Parameters
    ----------
    T_e : float or np.ndarray
        Electron temperature, non-negative [eV].
    T_i : float or np.ndarray
        Ion temperature, non-negative [eV].
    m_i : float or np.ndarray
        Ion mass, positive [kg].
    Z : float
        Ion charge number, positive [-].
    gamma_i : float
        Ion adiabatic index: 1 isothermal, 3 one-dimensional adiabatic, 5/3
        three-dimensional [-].

    Returns
    -------
    float or np.ndarray
        $c_s$ [m/s].

    Raises
    ------
    ValueError
        A negative temperature, a non-positive mass, charge or index.

    Convention
    ----------
    Electrons isothermal. Stangeby's working choice is $\gamma_i = 1$
    (isothermal ions); $\gamma_i = 3$ is the one-dimensional adiabatic,
    collisionless case. The choice moves $c_s$ by up to $\sqrt2$ at
    $T_i = T_e$, so it is a keyword, not a constant. ``vaft.formula.stability.c_s_from_Te_Ti_mi`` is the case
    $Z = \gamma_i = 1$ with temperatures in keV.

    Physical interpretation
    -----------------------
    The speed at which plasma flows into the sheath (Bohm): it sets the
    target particle flux and, with the sheath, the heat flux.

    References
    ----------
    .. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Sec. 2.2.
    """
    T_e = _non_negative(T_e, "T_e")
    T_i = _non_negative(T_i, "T_i")
    m_i = _positive(m_i, "m_i")
    _positive(Z, "Z")
    _positive(gamma_i, "gamma_i")
    return _out(np.sqrt(QE * (Z * T_e + gamma_i * T_i) / m_i))


def sheath_particle_flux(n_t, c_s, *, mach=1.0):
    r"""Particle flux density into the target sheath.

    $$\Gamma_t = M\,n_t\,c_s$$

    Parameters
    ----------
    n_t : float or np.ndarray
        Ion density at the sheath edge, non-negative [m^-3].
    c_s : float or np.ndarray
        Ion sound speed at the sheath edge (``ion_sound_speed``) [m/s].
    mach : float
        Parallel Mach number at the sheath edge; one is the Bohm criterion
        satisfied marginally [-].

    Returns
    -------
    float or np.ndarray
        $\Gamma_t$, along the field [m^-2 s^-1].

    Raises
    ------
    ValueError
        A negative density, a non-positive sound speed or Mach number.

    Convention
    ----------
    Parallel to $\mathbf B$ at the sheath *edge*; the flux onto the target
    surface is this times $\sin$ of the field-line incidence angle. $n_t$
    is the sheath-edge density, not the upstream one.

    Physical interpretation
    -----------------------
    The sheath removes ions at the rate the presheath accelerates them to
    the sound speed: the target is a particle sink at fixed velocity.

    References
    ----------
    .. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Sec. 2.3.
    """
    n_t = _non_negative(n_t, "n_t")
    c_s = _positive(c_s, "c_s")
    _positive(mach, "mach")
    return _out(mach * n_t * c_s)


def ion_saturation_current_density(n_i, c_s, *, Z=1.0):
    r"""Ion saturation current density collected by a negatively biased surface.

    $$j_\mathrm{sat} = Z e\,n_i\,c_s$$

    Parameters
    ----------
    n_i : float or np.ndarray
        Ion density at the sheath edge, non-negative [m^-3].
    c_s : float or np.ndarray
        Ion sound speed at the sheath edge [m/s].
    Z : float
        Ion charge number, positive [-].

    Returns
    -------
    float or np.ndarray
        $j_\mathrm{sat}$ [A/m^2].

    Raises
    ------
    ValueError
        A negative density, a non-positive sound speed or charge.

    Convention
    ----------
    Sheath-edge density with Mach one. From the unperturbed density
    $n_\infty$ the presheath drop is $\tfrac12$ in Stangeby's isothermal
    fluid model and $e^{-1/2} \approx 0.61$ in the Boltzmann form.
    ``vaft.process.langmuir`` uses $e^{-1/2}$ with $T_i = 0$: pass
    ``n_inf * np.exp(-0.5)`` and ``ion_sound_speed(T_e, 0, m_i)`` to match it.

    Physical interpretation
    -----------------------
    What a Langmuir probe or divertor tile measures directly; with $T_e$
    from the I--V characteristic it gives the density.

    References
    ----------
    .. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Sec. 2.6.
    """
    _positive(Z, "Z")
    return _out(Z * QE * np.asarray(sheath_particle_flux(n_i, c_s), dtype=float))


def sheath_heat_flux(gamma, n_t, T_t, c_s):
    r"""Parallel heat-flux density into the target sheath.

    $$q_{\parallel,t} = \gamma\,n_t\,e T_t\,c_s$$

    Parameters
    ----------
    gamma : float or np.ndarray
        Sheath heat-transmission coefficient, a model input (typically 7--8
        for a floating D target with $T_i = T_e$), positive [-].
    n_t : float or np.ndarray
        Sheath-edge density, non-negative [m^-3].
    T_t : float or np.ndarray
        Sheath-edge electron temperature, non-negative [eV].
    c_s : float or np.ndarray
        Sheath-edge sound speed [m/s].

    Returns
    -------
    float or np.ndarray
        $q_{\parallel,t}$ [W/m^2].

    Raises
    ------
    ValueError
        A non-positive ``gamma`` or ``c_s``, a negative density or temperature.

    Convention
    ----------
    $\gamma$ is per electron temperature: it absorbs $T_i/T_e$, secondary
    emission and the sheath potential, so it is supplied, never a universal
    constant. Surface recombination energy is not included.

    Physical interpretation
    -----------------------
    The sheath lets through a fixed energy per particle, a few $T_e$: with
    the particle flux it closes the heat balance at the target.

    References
    ----------
    .. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Sec. 2.8.
    """
    gamma = _positive(gamma, "gamma")
    T_t = _non_negative(T_t, "T_t")
    return _out(gamma * QE * T_t * np.asarray(sheath_particle_flux(n_t, c_s), dtype=float))


def spitzer_harm_parallel_heat_flux(T_e, dT_ds, *, kappa_0=KAPPA_0E):
    r"""Classical parallel electron heat conduction.

    $$q_{\parallel,e} = -\kappa_0\,T_e^{5/2}\,\frac{dT_e}{ds}$$

    Parameters
    ----------
    T_e : float or np.ndarray
        Electron temperature, non-negative [eV].
    dT_ds : float or np.ndarray
        Temperature gradient along $s$ [eV/m].
    kappa_0 : float
        Conduction coefficient; the default is Stangeby's $Z = 1$ working
        value 2000 [W/(m eV^{7/2})].

    Returns
    -------
    float or np.ndarray
        $q_{\parallel,e}$, positive along $+s$ [W/m^2].

    Raises
    ------
    ValueError
        A negative temperature or a non-positive ``kappa_0``.

    Convention
    ----------
    $\kappa_0$ depends on $Z_\mathrm{eff}$ and on the Coulomb logarithm:
    $3.16\,nT\tau_e/m_e$ gives 2040 at $Z = 1$, $\ln\Lambda = 15$, and the
    default 2000 is Stangeby's rounded working value; pass another for
    impure plasmas or another $\ln\Lambda$. The flux is along $+s$, upstream to target, when the
    temperature falls towards the target.

    Physical interpretation
    -----------------------
    $\kappa \propto T_e^{5/2}$ makes a hot SOL nearly isothermal along the
    field and concentrates the temperature drop near the target.

    Assumptions
    -----------
    Collisional (the electron mean free path short against the gradient
    length); flux limiting is not applied.

    References
    ----------
    .. [1] L. Spitzer and R. Harm, Phys. Rev. 89 (1953) 977.
    .. [2] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Sec. 4.10.
    """
    T_e = _non_negative(T_e, "T_e")
    _positive(kappa_0, "kappa_0")
    return _out(-kappa_0 * T_e**2.5 * np.asarray(dT_ds, dtype=float))


def two_point_upstream_temperature(T_t, q_par, L_par, *, kappa_0=KAPPA_0E):
    r"""Upstream temperature from conduction at constant parallel heat flux.

    $$T_u = \left(T_t^{7/2} + \frac{7}{2}\,\frac{q_\parallel L_\parallel}{\kappa_0}\right)^{2/7}$$

    Parameters
    ----------
    T_t : float or np.ndarray
        Target (sheath-edge) electron temperature, non-negative [eV].
    q_par : float or np.ndarray
        Parallel heat-flux density, constant along the flux tube,
        non-negative [W/m^2].
    L_par : float or np.ndarray
        Parallel connection length from upstream to the target, positive [m].
    kappa_0 : float
        Conduction coefficient [W/(m eV^{7/2})].

    Returns
    -------
    float or np.ndarray
        $T_u$ [eV].

    Raises
    ------
    ValueError
        A negative temperature or heat flux, a non-positive length or
        ``kappa_0``.

    Convention
    ----------
    The integral of ``spitzer_harm_parallel_heat_flux`` at constant
    $q_\parallel$ from the target ($s = L_\parallel$) back to upstream; the
    endpoints are the declared upstream point and the target. With a
    uniform source from the stagnation point ($q = 0$) to the target,
    $q_\parallel$ the target value and $L_\parallel$ measured from the
    stagnation point, the $\tfrac72$ becomes $\tfrac74$.

    Physical interpretation
    -----------------------
    The two-point model's conduction leg: $T_u$ depends on $q_\parallel L$
    only to the $2/7$ power, so upstream temperatures are robust while the
    target temperature is not.

    Assumptions
    -----------
    Conduction only, no convection or volumetric losses between the two
    points, $q_\parallel$ constant along the tube.

    References
    ----------
    .. [1] P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*,
           IOP (2000), Sec. 5.2.
    """
    T_t = _non_negative(T_t, "T_t")
    q_par = _non_negative(q_par, "q_par")
    L_par = _positive(L_par, "L_par")
    _positive(kappa_0, "kappa_0")
    return _out((T_t**3.5 + 3.5 * q_par * L_par / kappa_0) ** (2.0 / 7.0))


def eich_target_heat_flux_profile(s, q0, lambda_q, S, *, s0=0.0, q_bg=0.0, flux_expansion=1.0):
    r"""Target heat-flux profile: an exponential SOL decay convolved with Gaussian spreading.

    $$q(\bar s) = \frac{q_0}{2}\exp\!\left[\left(\frac{S}{2\lambda_q f_x}\right)^2
      - \frac{\bar s}{\lambda_q f_x}\right]\mathrm{erfc}\!\left(\frac{S}{2\lambda_q f_x}
      - \frac{\bar s}{S}\right) + q_\mathrm{bg},\qquad \bar s = s - s_0$$

    Parameters
    ----------
    s : float or np.ndarray
        Coordinate along the target, increasing into the SOL [m].
    q0 : float or np.ndarray
        Peak of the unspread exponential at the strike point [W/m^2].
    lambda_q : float or np.ndarray
        Heat-flux decay length at the outer midplane, positive [m].
    S : float or np.ndarray
        Gaussian spreading width at the target, positive [m].
    s0 : float
        Strike-point position [m].
    q_bg : float
        Background heat flux [W/m^2].
    flux_expansion : float
        Total expansion from the outer midplane to the target along ``s``,
        positive [-].

    Returns
    -------
    float or np.ndarray
        Heat-flux density at the target [W/m^2].

    Raises
    ------
    ValueError
        A non-positive ``lambda_q``, ``S`` or ``flux_expansion``.

    Convention
    ----------
    $\lambda_q$ is the *midplane* value and $f_x$ maps it to the target;
    $S$ is measured at the target. The profile is the exponential
    $q_0 e^{-\bar s/\lambda_q f_x}$ for $\bar s > 0$ convolved with a
    Gaussian of width $S$: private-flux spreading, not a second exponential.

    Physical interpretation
    -----------------------
    What IR thermography and target probes fit: $\lambda_q$ is set upstream,
    $S$ by diffusion in the divertor leg, and the peak heat flux falls as
    $S$ grows at fixed $\lambda_q$.

    References
    ----------
    .. [1] T. Eich et al., Phys. Rev. Lett. 107 (2011) 215001.
    .. [2] T. Eich et al., Nucl. Fusion 53 (2013) 093031.
    """
    from scipy.special import erfc, erfcx

    lam = _positive(lambda_q, "lambda_q") * _positive(flux_expansion, "flux_expansion")
    S = _positive(S, "S")
    x = np.asarray(s, dtype=float) - s0
    u = S / (2.0 * lam) - x / S
    # exp(a) erfc(u) with a = (S/2lam)^2 - x/lam = u^2 - (x/S)^2: for u >= 0 as exp(-(x/S)^2) erfcx(u), which
    # cannot overflow in the private flux region; for u < 0 directly, where a < 0 and erfc(u) <= 2
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        shape = np.where(u >= 0.0, np.exp(-(x / S) ** 2) * erfcx(np.maximum(u, 0.0)),
                         np.exp(np.minimum(u**2 - (x / S) ** 2, 0.0)) * erfc(np.minimum(u, 0.0)))
    q = 0.5 * np.asarray(q0, dtype=float) * shape
    return _out(q + q_bg)


def eich_integral_width(lambda_q, S, *, flux_expansion=1.0):
    r"""Integral power width of the Eich profile, mapped to the outer midplane.

    $$\lambda_\mathrm{int} \simeq \lambda_q + 1.64\,S/f_x$$

    Parameters
    ----------
    lambda_q : float or np.ndarray
        Heat-flux decay length at the outer midplane, positive [m].
    S : float or np.ndarray
        Gaussian spreading width at the target, positive [m].
    flux_expansion : float
        Total expansion from the outer midplane to the target, positive [-].

    Returns
    -------
    float or np.ndarray
        $\lambda_\mathrm{int}$ at the outer midplane [m].

    Raises
    ------
    ValueError
        A non-positive input.

    Convention
    ----------
    Integral width $\int(q - q_\mathrm{bg})\,ds / (q_\mathrm{peak} f_x)$ with
    $s$ along the target, i.e. mapped to the midplane; 1.64 is
    Makowski's fit to the exact integral of the Eich profile, accurate to a
    few per cent over the fitted range, so this is an approximation, not a
    definition.

    Physical interpretation
    -----------------------
    The width over which the target actually receives the power: the
    engineering quantity for peak loading, $q_\mathrm{peak} \approx
    P/(2\pi R\,\lambda_\mathrm{int} f_x)$.

    Validity
    --------
    Empirical fit. Makowski et al. (2012), over the $S/\lambda_q$ range of
    the multi-machine database.

    References
    ----------
    .. [1] M. A. Makowski et al., Phys. Plasmas 19 (2012) 056122.
    .. [2] T. Eich et al., Nucl. Fusion 53 (2013) 093031.
    """
    lam = _positive(lambda_q, "lambda_q")
    S = _positive(S, "S")
    fx = _positive(flux_expansion, "flux_expansion")
    return _out(lam + 1.64 * S / fx)


def radiative_condensation_growth_rate(n, T, L, dL_dT, k_parallel, kappa_parallel):
    r"""Growth rate of the thermal-condensation (MARFE) instability at constant pressure.

    $$\gamma = \frac{2}{5\,n e}\left(\frac{2L}{T} - \frac{\partial L}{\partial T}
      - k_\parallel^2\kappa_\parallel\right)$$

    Parameters
    ----------
    n : float or np.ndarray
        Electron density, positive [m^-3].
    T : float or np.ndarray
        Temperature, positive [eV].
    L : float or np.ndarray
        Radiated power density, $L \propto n^2$, non-negative [W/m^3].
    dL_dT : float or np.ndarray
        $\partial L/\partial T$ at fixed density [W/(m^3 eV)].
    k_parallel : float or np.ndarray
        Parallel wavenumber of the perturbation, $m/(qR)$ for a poloidal
        harmonic $m$ [1/m].
    kappa_parallel : float or np.ndarray
        Parallel conductivity, $\kappa_0 T^{5/2}$ in ``spitzer_harm_parallel_heat_flux``'s
        units, non-negative [W/(m eV)].

    Returns
    -------
    float or np.ndarray
        $\gamma$; positive is unstable [1/s].

    Raises
    ------
    ValueError
        A non-positive density or temperature, a negative $L$ or $\kappa_\parallel$.

    Convention
    ----------
    Drake's Eq. (2): the perturbation is slow against sound, $\gamma \ll
    k_\parallel c_s$, so pressure stays constant and $\tilde n/n = -\tilde T/T$
    (his Eq. 17); with $L \propto n^2$ the density rise feeds the $2L/T$
    term. Temperatures in eV and $\kappa_\parallel$ per eV, as in this module,
    so the $e$ converts the heat capacity $\tfrac52 n$ to joules. Instability
    needs $2L/T - \partial L/\partial T > k_\parallel^2\kappa_\parallel$, which
    equals $-dL/dT$ at constant pressure: it does **not** need
    $\partial L/\partial T < 0$. Perpendicular conduction (Lipschultz's
    $K_\perp/\Delta^2$) is neglected, as Drake and Lipschultz find it small.

    Physical interpretation
    -----------------------
    A cooled spot on a flux surface radiates more, is compressed by the
    surrounding pressure, radiates more still and condenses: the MARFE.
    Parallel conduction refills it and stabilises short parallel wavelengths;
    the $m = 1$ harmonic with the longest connection length goes first.

    References
    ----------
    .. [1] J. F. Drake, Phys. Fluids 30 (1987) 2429, Eqs. (2), (16)-(18).
    .. [2] B. Lipschultz, J. Nucl. Mater. 145-147 (1987) 15, Eq. (11).
    """
    n = _positive(n, "n")
    T = _positive(T, "T")
    L = _non_negative(L, "L")
    kappa = _non_negative(kappa_parallel, "kappa_parallel")
    drive = 2.0 * L / T - np.asarray(dL_dT, dtype=float)
    return _out(2.0 / (5.0 * n * QE) * (drive - np.asarray(k_parallel, dtype=float) ** 2 * kappa))


def radiative_thermal_instability_growth_rate(n, dL_dT):
    r"""Growth rate of the radiative thermal instability of a flute perturbation at constant density.

    $$\gamma = -\frac{2}{3\,n e}\frac{\partial L}{\partial T}$$

    Parameters
    ----------
    n : float or np.ndarray
        Electron density, positive [m^-3].
    dL_dT : float or np.ndarray
        $\partial L/\partial T$ at fixed density [W/(m^3 eV)].

    Returns
    -------
    float or np.ndarray
        $\gamma$; positive is unstable [1/s].

    Raises
    ------
    ValueError
        A non-positive density.

    Convention
    ----------
    Drake's Eq. (1): $k_\parallel = 0$, so no sound wave equalises pressure
    and the density does not change; only a falling radiation curve,
    $\partial L/\partial T < 0$, is unstable. Compare
    ``radiative_condensation_growth_rate``, where the constant-pressure
    density rise adds the $2L/T$ drive.

    Physical interpretation
    -----------------------
    The axisymmetric counterpart of the MARFE: a whole flux surface cooling
    where the radiation curve falls with temperature -- the poloidally
    symmetric radiation collapse behind detachment and the density limit.

    References
    ----------
    .. [1] J. F. Drake, Phys. Fluids 30 (1987) 2429, Eq. (1).
    """
    n = _positive(n, "n")
    return _out(-2.0 / (3.0 * n * QE) * np.asarray(dL_dT, dtype=float))

