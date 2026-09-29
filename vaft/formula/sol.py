r"""Scrape-off layer: sound speed, sheath fluxes, parallel conduction and the target heat-flux profile.

The small, independently testable relations of the open-field-line edge
(issue #951, first slice): the ion sound speed with its closure explicit,
the sheath-limited particle flux and ion saturation current, the sheath heat
transmission, Spitzer--Harm parallel electron conduction and its integrated
two-point form, and the Eich target heat-flux profile with its integral width.
SOL blobs and filaments (#1211) follow D'Ippolito, Myra and Zweben (2011):
a Gaussian of radius delta, cold ions, x radial, y binormal. Reduced models built on them (the two-point model with loss factors,
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

from typing import NamedTuple

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
    "BlobRegimeVelocities",
    "blob_reference_size",
    "blob_reference_velocity",
    "blob_collisionality",
    "sheath_connected_blob_velocity",
    "inertial_blob_velocity",
    "interpolated_blob_velocity",
    "blob_regime_velocities",
    "blob_density_perturbation",
    "blob_crossover_size",
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


class BlobRegimeVelocities(NamedTuple):
    """Normalized radial velocities $\\hat v$ of the four regimes of the two-region model [-]."""

    resistive_ballooning: np.ndarray
    resistive_x_point: np.ndarray
    sheath_connected: np.ndarray
    connected_ideal_interchange: np.ndarray


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
        Ion sound gyroradius $c_s/\Omega_i$ with the cold-ion $c_s$, positive [m].
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
        Cold-ion sound speed $(T_e/m_i)^{1/2}$, positive [m/s].
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


def sheath_connected_blob_velocity(c_s, rho_s, delta, L_par, R, *, relative_amplitude=1.0):
    r"""Radial velocity of a blob whose polarization current closes through the sheaths.

    $$v_x = c_s\,\frac{\delta n}{n}\,\frac{L_\parallel}{R}\left(\frac{\rho_s}{\delta}\right)^2$$

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
    D'Ippolito, Myra and Zweben (2011) Eq. (3) for an isolated blob in vacuum
    ($\delta n/n = 1$), and the linear $\delta n/n$ of their normalized form
    $\hat v = (\delta n/n)/\hat\delta^2$ (p. 060501-25) with a background;
    cold ions, $c_s = (T_e/m_i)^{1/2}$. Linearized sheath closure $J_\parallel = ne^2c_s\Phi/T_e$,
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
    return _out(_amplitude(relative_amplitude) * c_s * L_par / R * (rho_s / delta) ** 2)


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
    (2006) Eq. (11) is linear in their $f_b$, $c_sf_b(a_b/R)^{1/2}$; their
    Eq. (A2) is the square-root form with background. Theiler et al.'s
    $\sqrt{2a/R}\,c_s$ has their HWHM size. Cold ions: $c_s = (T_e/m_i)^{1/2}$,
    i.e. ``ion_sound_speed(T_e, 0, m_i)``.

    Physical interpretation
    -----------------------
    The resistive-ballooning (fully disconnected) limit: curvature drive
    against ion inertia at the midplane, the fastest a blob can go;
    $v \propto \delta^{1/2}$, large blobs fast.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, p. 060501-25.
    .. [2] J. R. Myra et al., Phys. Plasmas 13 (2006) 092509, Eqs. (11), (A2).
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
    (their Fig. 27). The limits cross at $\hat\delta = (\delta n/n)^{1/5}$
    (``blob_crossover_size``); the interpolation itself peaks lower, at
    $0.574\,(\delta n/n)^{1/5}$. Coherence, not speed, is what the review
    ties to $\hat\delta \sim 1$.

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


def blob_density_perturbation(r, n_background, delta_n, delta):
    r"""The prescribed filament state: a Gaussian density perturbation on a background.

    $$n(r) = n_0 + \Delta n\,\exp\!\left(-\frac{r^2}{2\delta^2}\right)$$

    Parameters
    ----------
    r : float or np.ndarray
        Distance from the filament axis in the perpendicular plane, non-negative [m].
    n_background : float or np.ndarray
        Background density $n_0$, positive [m^-3].
    delta_n : float or np.ndarray
        Amplitude: positive for a blob, negative for a hole, above $-n_0$ [m^-3].
    delta : float or np.ndarray
        Radius, positive [m].

    Returns
    -------
    float or np.ndarray
        $n(r)$ [m^-3].

    Raises
    ------
    ValueError
        A negative radius, a non-positive background or radius, or a hole
        deeper than the background.

    Convention
    ----------
    The size convention every blob relation of this module uses: the radius
    at which the perturbation has fallen to $e^{-1/2}$ (D'Ippolito, Myra and
    Zweben 2011, Eq. 3). Krasheninnikov's $e^{-y^2/\delta^2}$ width is
    $\sqrt2$ times this; a HWHM is $1.18$ times it. The relative amplitude of
    the velocity relations is $\Delta n/(n_0 + \Delta n)$ at the peak.

    Physical interpretation
    -----------------------
    A blob ($\Delta n > 0$) polarizes into a dipole that drives it outward on
    the low-field side; a hole ($\Delta n < 0$) polarizes the other way and
    moves inward.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, Eq. (3) and p. 060501-6.
    """
    r = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("r must be non-negative and finite")
    n0, delta = _positive(n_background, "n_background"), _positive(delta, "delta")
    dn = np.asarray(delta_n, dtype=float)
    if np.any(~np.isfinite(dn)) or np.any(n0 + dn <= 0.0):
        raise ValueError("delta_n must be finite, and a hole cannot be deeper than the background")
    return _out(n0 + dn * np.exp(-(r**2) / (2.0 * delta**2)))


def blob_crossover_size(*, relative_amplitude=1.0):
    r"""Normalized size at which the inertial and sheath-connected limits give the same velocity.

    $$\hat\delta_c = \left(\frac{\delta n}{n}\right)^{1/5}$$

    Parameters
    ----------
    relative_amplitude : float or np.ndarray
        $\delta n/n$, in (0, 1] [-].

    Returns
    -------
    float or np.ndarray
        $\hat\delta_c = \delta_c/\delta_*$ [-].

    Raises
    ------
    ValueError
        An amplitude outside (0, 1].

    Convention
    ----------
    Where $(\delta n/n)^{1/2}\hat\delta^{1/2} = (\delta n/n)/\hat\delta^2$, the
    two limits of D'Ippolito, Myra and Zweben (2011) p. 060501-25; one at
    $\delta n/n = 1$. The interpolation of ``interpolated_blob_velocity``
    peaks at $0.574\,\hat\delta_c$, not at the crossing.

    Physical interpretation
    -----------------------
    Smaller blobs are inertia-limited, larger ones sheath-limited; a weak
    blob changes regime at a smaller size.

    References
    ----------
    .. [1] D. A. D'Ippolito, J. R. Myra and S. J. Zweben, Phys. Plasmas 18
           (2011) 060501, p. 060501-25.
    """
    return _out(_amplitude(relative_amplitude) ** 0.2)
