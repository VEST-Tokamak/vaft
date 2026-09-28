r"""Neoclassical toroidal viscosity: the analytic relations of transport in broken toroidal symmetry.

A non-axisymmetric $\delta B$ makes the radial particle flux of each species
non-ambipolar. Quasineutrality forces a return (polarisation) current that
cancels the net radial current, and its $\mathbf J\times\mathbf B$ is the
toroidal torque on the plasma -- NTV, equal and opposite to the
$\mathbf J\times\mathbf B$ of the non-ambipolar current itself. The *size* of
the flux needs the bounce-averaged drift-kinetic response (PENT/PENTRC, NEO,
kinetic GPEC; ``vaft.code``). What is here is the part that is exact and
convention-bearing: the precession frequency whose zero is the
superbanana-plateau resonance, and the flux-force relation that turns
non-ambipolar fluxes into torque.

Shaing's asymptotic regimes lie on two branches, in order of decreasing
collisionality:

- non-resonant ($\omega_d$ bounded away from zero): $1/\nu$ while
  $\nu_\mathrm{eff}$ exceeds the precession, flux $\propto 1/\nu$; then the
  $\nu$--$\sqrt\nu$ regime, a collisional boundary layer at the
  trapped-passing boundary, whose $\sqrt\nu$ part dominates as $\nu \to 0$;
- resonant ($\omega_d \to 0$ for some trapped particles): $1/\nu$, then the
  superbanana plateau, flux independent of $\nu$, then the superbanana
  regime, flux $\propto\nu$, at the lowest collisionality.

Shaing states the $1/\nu$ boundary as $\nu/\epsilon$ against $|q\omega_E|$
with the magnetic precession neglected; with it kept the comparison is with
$|\omega_d|$. The connected formula that joins the regimes (Shaing, Sun and
Sabbagh 2010) is not implemented: its kernels need the bounce-averaged
$\delta B$ spectrum and source-specific normalisations. The diagrams draw the
exponents only.

Notation
--------
omega_exb      : toroidal E x B rotation frequency, $-d\Phi/d\psi_\mathrm{out}$  [rad/s]
omega_magnetic : bounce-averaged toroidal magnetic-drift precession  [rad/s]
omega_d        : bounce-averaged toroidal precession  [rad/s]
Z_s            : charge number of species s  [-]
Gamma_s        : $\langle\Gamma_s\cdot\nabla V\rangle$, particles of species s crossing the surface  [1/s]
psi            : $RA_\phi$, poloidal flux per radian, IMAS $\phi$  [Wb/rad]
V              : volume enclosed by the surface  [m^3]

Conventions
-----------
Precession frequencies are toroidal and positive along the plasma current,
as VAFT's kinetic-profile ``omega_exb`` is ($-d\Phi/d\psi_\mathrm{out}$ with
$\psi_\mathrm{out}$ the poloidal flux per radian increasing outward, e.g.
TRANSP ``PLFLX``); ``omega_exb`` is never the toroidal fluid rotation
``omega_tor``. The torque is positive along the IMAS $\phi$ (counter-clockwise
from above), with $\psi = RA_\phi$ as
``vaft.formula.particle.psi_per_radian_from_cocos`` returns it.

References
----------
.. [1] K. C. Shaing, Phys. Plasmas 10 (2003) 1443.
.. [2] K. C. Shaing, Y. Sun and S. A. Sabbagh, Plasma Phys. Control. Fusion
       52 (2010) 025005.
.. [3] J.-K. Park, A. H. Boozer and J. E. Menard, Phys. Rev. Lett. 102
       (2009) 065002.
"""

import numpy as np

from .constants import QE

__all__ = [
    "ntv_precession_frequency",
    "nonambipolar_torque_density",
]


def _out(result):
    return float(result) if np.ndim(result) == 0 else result


def _finite(value, name):
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def ntv_precession_frequency(omega_exb, omega_magnetic):
    r"""Bounce-averaged toroidal precession of trapped particles, whose zero is the superbanana-plateau resonance.

    $$\omega_d = \omega_E + \omega_B \qquad (= q\,\omega_E^\mathrm{Shaing} + \omega_B)$$

    Parameters
    ----------
    omega_exb : float or np.ndarray
        Toroidal E x B rotation frequency $\omega_E = -d\Phi/d\psi_\mathrm{out}$,
        $\psi_\mathrm{out}$ the poloidal flux per radian increasing outward --
        VAFT's kinetic-profile ``omega_exb``, positive along the plasma
        current, not the toroidal fluid rotation [rad/s].
    omega_magnetic : float or np.ndarray
        Bounce-averaged toroidal precession by the grad-B and curvature
        drifts, for the species and energy considered, positive along the
        plasma current [rad/s].

    Returns
    -------
    float or np.ndarray
        $\omega_d$, positive along the plasma current; zero on the
        superbanana-plateau resonance [rad/s].

    Raises
    ------
    ValueError
        A non-finite input.

    Convention
    ----------
    Both inputs must be measured in the same toroidal orientation; VAFT's
    is along the plasma current, the orientation of ``omega_exb``. A rigid
    rotation $\omega\hat\phi$ has $\mathbf E = -\omega\nabla(RA_\phi)$, so
    $-d\Phi/d\psi_\mathrm{out}$ is the rotation along the current whichever
    way the current flows. Shaing writes $q\omega_E + \omega_B$ with his
    $\omega_E$ taken against the *toroidal* flux; since $d\psi_t = q\,d\psi$,
    his $q\omega_E$ is this ``omega_exb``. The toroidal rotation
    $\omega_\phi$ differs from $\omega_E$ by the diamagnetic and
    poloidal-flow terms of radial force balance, and substituting it moves
    the resonance. $\omega_B \propto$ energy over charge, so each species and
    energy has its own $\omega_d$. This is the $\ell = 0$ case of
    ``vaft.formula.particle.bounce_harmonic_detuning`` with $n = -1$.

    Physical interpretation
    -----------------------
    A trapped orbit precessing at $\omega_d$ sees a static $\delta B$ at a
    Doppler-shifted frequency; where $\omega_d \to 0$ it sits in a fixed
    phase of the perturbation and its radial step accumulates -- the
    superbanana plateau, whose flux no longer depends on collisions.
    Away from it, $\nu_\mathrm{eff}$ against $|\omega_d|$ separates the
    $1/\nu$ regime from the $\nu$--$\sqrt\nu$ regime.

    Assumptions
    -----------
    Bounce-averaged guiding-centre motion; $\omega_E$ constant across the
    banana width.

    References
    ----------
    .. [1] K. C. Shaing, Phys. Plasmas 10 (2003) 1443.
    .. [2] J.-K. Park, A. H. Boozer and J. E. Menard, Phys. Rev. Lett. 102
           (2009) 065002.
    """
    return _out(_finite(omega_exb, "omega_exb") + _finite(omega_magnetic, "omega_magnetic"))


def nonambipolar_torque_density(charge_numbers, particle_flux, dpsi_dV):
    r"""Flux-surface-averaged toroidal torque density on the plasma from non-ambipolar radial fluxes (NTV).

    $$T_\phi = \frac{d\psi}{dV}\sum_s Z_s e\,\langle\boldsymbol\Gamma_s\cdot\nabla V\rangle$$

    Parameters
    ----------
    charge_numbers : array_like
        $Z_s$ of each species, electrons $-1$, shape ``(S,)`` [-].
    particle_flux : array_like
        $\langle\boldsymbol\Gamma_s\cdot\nabla V\rangle$ of each species along
        axis 0, shape ``(S, ...)``: particles crossing the surface outward
        per second [1/s].
    dpsi_dV : float or np.ndarray
        $d\psi/dV$ of $\psi = RA_\phi$, broadcast against the trailing axes
        of ``particle_flux``; negative for a plasma current along $+\phi$
        [Wb/(rad m^3)].

    Returns
    -------
    float or np.ndarray
        Torque density on the plasma about the symmetry axis, positive
        along the IMAS $\phi$, averaged over the surface [N m/m^3].

    Raises
    ------
    ValueError
        Species counts differ, or an input is not finite.

    Convention
    ----------
    With $\mathbf B = \nabla\psi\times\nabla\phi + F\nabla\phi$ and
    $\psi = RA_\phi$ per radian (``psi_per_radian_from_cocos``; IMAS COCOS 11
    stores $-2\pi\psi$), $R\hat\phi\cdot(\mathbf J\times\mathbf B)
    = -\mathbf J\cdot\nabla\psi$ exactly. The non-ambipolar current
    $\mathbf J_\mathrm{na} = \sum_s Z_se\boldsymbol\Gamma_s$ is cancelled by a
    return current $-\mathbf J_\mathrm{na}$, whose $\mathbf J\times\mathbf B$
    -- equivalently, the sum of each species' toroidal viscous force -- is
    the torque on the plasma: $T_\phi = +\langle\mathbf J_\mathrm{na}\cdot
    \nabla\psi\rangle = (d\psi/dV)\sum_s Z_se\langle\boldsymbol\Gamma_s\cdot
    \nabla V\rangle$. In Shaing's $\psi = -RA_\phi$ this reads
    $-(d\psi/dV)\sum_s Z_se\Gamma_s$. A torque *density*; integrate over $V$
    for the torque. Ambipolar fluxes give zero.

    Physical interpretation
    -----------------------
    An outward non-ambipolar ion flux in a plasma with current along $+\phi$
    ($d\psi/dV < 0$) drives rotation against the current -- the
    counter-current torque of fast-ion ripple loss and of ion-root NTV.

    Assumptions
    -----------
    Steady state on the transport time scale, quasineutrality; the fluxes
    are the non-ambipolar parts from a kinetic calculation this function
    does not make. No phenomenological $-\nu_\mathrm{NTV}(\Omega - \Omega_0)$
    damping model is implied.

    References
    ----------
    .. [1] K. C. Shaing, Phys. Plasmas 10 (2003) 1443.
    .. [2] P. Helander and D. J. Sigmar, *Collisional Transport in Magnetized
           Plasmas*, Cambridge University Press (2002), Ch. 8.
    """
    Z = _finite(charge_numbers, "charge_numbers")
    flux = _finite(particle_flux, "particle_flux")
    if Z.ndim != 1 or flux.ndim < 1 or flux.shape[0] != Z.shape[0]:
        raise ValueError("charge_numbers must be 1-D with one entry per species along axis 0 of particle_flux")
    current = QE * np.tensordot(Z, flux, axes=(0, 0))
    return _out(_finite(dpsi_dV, "dpsi_dV") * current)
