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
lambda_q   : upstream heat-flux decay length, mapped to the target [m]
S          : Gaussian spreading width at the target            [m]

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
    Electrons isothermal. The Bohm criterion at the sheath edge is usually
    written with $\gamma_i = 3$ (Stangeby) or 1 (isothermal ions); the choice
    moves $c_s$ by up to $\sqrt2$ at $T_i = T_e$, so it is a keyword, not a
    constant. ``vaft.formula.stability.c_s_from_Te_Ti_mi`` is the case
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
    Sheath-edge density with Mach one. Langmuir-probe analyses often write
    $j_\mathrm{sat} = \tfrac12 e n_\infty c_s$ with the *unperturbed* density
    $n_\infty$ and the presheath factor $\tfrac12$ folded in; pass
    $n_\infty/2$ for that convention (``vaft.process.langmuir``).

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
    endpoints are the declared upstream point and the target. With
    $q_\parallel$ deposited along the tube instead (uniform source), the
    $\tfrac72$ becomes $\tfrac74$.

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
    from scipy.special import erfc

    lam = _positive(lambda_q, "lambda_q") * _positive(flux_expansion, "flux_expansion")
    S = _positive(S, "S")
    x = np.asarray(s, dtype=float) - s0
    q = 0.5 * np.asarray(q0, dtype=float) * np.exp((S / (2.0 * lam)) ** 2 - x / lam) * erfc(S / (2.0 * lam) - x / S)
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
    Integral width $\int(q - q_\mathrm{bg})\,ds / q_\mathrm{peak}$; 1.64 is
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
