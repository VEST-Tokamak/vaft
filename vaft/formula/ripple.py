"""
Toroidal-field ripple from a finite number of TF coils, and its orbit consequences.

Two kinds of relation live here and stay apart: the *field* -- the smooth
$1/R$ variation (``vaft.formula.vacuum_toroidal_field``) plus an
$N_\\mathrm{TF}$-periodic ripple -- and the *orbit consequences* a trapped
particle feels in it: local ripple wells, the ripple-trapped pitch range and
the Goldston--White--Boozer stochasticity threshold. All are the standard
large-aspect-ratio, small-ripple estimates; none replaces a 3-D field map or
orbit following.

Notation
--------
B0      : field at the magnetic axis                           [T]
epsilon : inverse aspect ratio of the surface, r/R0            [-]
theta   : poloidal angle from the outboard midplane            [rad]
phi     : toroidal angle                                       [rad]
delta   : ripple amplitude (Bmax - Bmin)/(Bmax + Bmin)         [-]
n_tf    : number of toroidal-field coils                       [-]
q       : safety factor                                        [-]
dq_dr   : radial derivative of q in physical minor radius       [1/m]
rho     : Larmor radius of the particle                        [m]

Conventions
-----------
$B = B_0(1 - \\epsilon\\cos\\theta)[1 - \\delta\\cos(N_\\mathrm{TF}\\phi + \\phi_0)]$,
so $\\delta$ is the local amplitude of ``ripple_amplitude`` exactly (the
additive $B_0[1 - \\epsilon\\cos\\theta - \\delta\\cos(\\ldots)]$ agrees to first
order). The ripple is a minimum where $N_\\mathrm{TF}\\phi + \\phi_0 = 0$, i.e. between coils
for $\\phi_0 = 0$ when a coil sits at $\\phi = \\pi/N_\\mathrm{TF}$. $\\delta$ is the
local amplitude on the surface, not the machine's maximum.

References
----------
.. [1] P. N. Yushmanov, Rev. Plasma Phys. 16 (1990) 117 (review of ripple
       transport).
.. [2] R. J. Goldston, R. B. White and A. H. Boozer, Phys. Rev. Lett. 47
       (1981) 647.
"""

import numpy as np

__all__ = [
    "toroidal_ripple_field",
    "ripple_amplitude",
    "ripple_well_parameter",
    "ripple_trapping_pitch",
    "gwb_stochastic_threshold",
    "gwb_stochasticity_parameter",
]


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, not {value!r}")
    return int(value)


def _scalar_or_array(result):
    return float(result) if np.ndim(result) == 0 else result


def toroidal_ripple_field(B0, epsilon, theta, delta, n_tf, phi, phase=0.0):
    r"""Field strength on a flux surface with toroidicity and TF-coil ripple.

    $$B = B_0\left(1 - \epsilon\cos\theta\right)\left[1 - \delta\cos(N_\mathrm{TF}\phi + \phi_0)\right]$$

    Parameters
    ----------
    B0 : float
        Field at the magnetic axis [T].
    epsilon : float or np.ndarray
        Inverse aspect ratio $r/R_0$ of the surface [-].
    theta : float or np.ndarray
        Poloidal angle from the outboard midplane [rad].
    delta : float or np.ndarray
        Ripple amplitude on the surface [-].
    n_tf : int
        Number of TF coils [-].
    phi : float or np.ndarray
        Toroidal angle [rad].
    phase : float
        Toroidal phase $\phi_0$ of the ripple [rad].

    Returns
    -------
    float or np.ndarray
        $|B|$ [T].

    Raises
    ------
    ValueError
        ``n_tf`` is not a positive integer, or ``delta`` or ``epsilon`` is negative.

    Convention
    ----------
    $\theta = 0$ outboard, so the $1/R$ term makes the outboard midplane the
    field minimum; the ripple factor is a minimum where
    $N_\mathrm{TF}\phi + \phi_0 = 0$. Multiplicative, so $\delta$ is the local
    ``ripple_amplitude`` at every $\theta$; to first order it equals the
    additive $B_0[1 - \epsilon\cos\theta - \delta\cos(\ldots)]$. With
    $\delta = 0$ it is the first-order expansion of ``vacuum_toroidal_field``.

    Physical interpretation
    -----------------------
    The smooth $1/R$ mirror that traps bananas, corrugated $N_\mathrm{TF}$ times
    around the torus by the gaps between discrete coils.

    Assumptions
    -----------
    Large aspect ratio ($\epsilon \ll 1$), small ripple ($\delta \ll \epsilon$
    typically), a single $N_\mathrm{TF}$ harmonic; $\delta$ is taken as given
    on the surface (it grows steeply towards the outboard edge).

    See Also
    --------
    vaft.diagram.toroidal_field_ripple : the canonical diagram of this relation.

    References
    ----------
    .. [1] P. N. Yushmanov, Rev. Plasma Phys. 16 (1990) 117, Sec. 1.
    """
    n = _positive_int(n_tf, "n_tf")
    delta = np.asarray(delta, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    if np.any(delta < 0.0) or np.any(epsilon < 0.0):
        raise ValueError("delta and epsilon must be non-negative")
    result = (float(B0) * (1.0 - epsilon * np.cos(np.asarray(theta, dtype=float)))
              * (1.0 - delta * np.cos(n * np.asarray(phi, dtype=float) + float(phase))))
    return _scalar_or_array(result)


def ripple_amplitude(B_max, B_min):
    r"""Ripple amplitude from the extreme field strengths along the toroidal angle.

    $$\delta = \frac{B_\mathrm{max} - B_\mathrm{min}}{B_\mathrm{max} + B_\mathrm{min}}$$

    Parameters
    ----------
    B_max : float or np.ndarray
        Largest $|B|$ along $\phi$ at fixed $(R, Z)$, under a coil [T].
    B_min : float or np.ndarray
        Smallest $|B|$ along $\phi$ at the same $(R, Z)$, between coils [T].

    Returns
    -------
    float or np.ndarray
        $\delta$, in $[0, 1)$ [-].

    Raises
    ------
    ValueError
        A field is not positive, or ``B_min`` exceeds ``B_max``.

    Convention
    ----------
    The half peak-to-peak variation over the mean, so that
    ``toroidal_ripple_field`` has $B_\mathrm{max,min} = \bar B(1 \pm \delta)$.
    Some machines quote the full peak-to-peak $(B_\mathrm{max} - B_\mathrm{min})/\bar B = 2\delta$.

    Physical interpretation
    -----------------------
    How deep the toroidal corrugation is at one point; it rises steeply with
    $R$ towards the coil legs.

    See Also
    --------
    vaft.diagram.toroidal_field_ripple : the canonical diagram of this relation.

    References
    ----------
    .. [1] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2471, Ch. 5
           (ripple definition).
    """
    B_max = np.asarray(B_max, dtype=float)
    B_min = np.asarray(B_min, dtype=float)
    if np.any(B_min <= 0.0) or np.any(B_max < B_min):
        raise ValueError("fields must be positive with B_max >= B_min")
    return _scalar_or_array((B_max - B_min) / (B_max + B_min))


def ripple_well_parameter(epsilon, theta, q, delta, n_tf):
    r"""Whether ripple makes local wells along a field line: $\alpha^* < 1$.

    $$\alpha^* = \frac{\epsilon\,|\sin\theta|}{N_\mathrm{TF}\,q\,\delta}$$

    Parameters
    ----------
    epsilon : float or np.ndarray
        Inverse aspect ratio of the surface [-].
    theta : float or np.ndarray
        Poloidal angle from the outboard midplane [rad].
    q : float or np.ndarray
        Safety factor, a positive magnitude [-].
    delta : float or np.ndarray
        Ripple amplitude on the surface [-].
    n_tf : int
        Number of TF coils [-].

    Returns
    -------
    float or np.ndarray
        $\alpha^*$; ripple wells exist where it is below 1 [-].

    Raises
    ------
    ValueError
        ``n_tf`` is not a positive integer, or ``q`` or ``delta`` is not positive.

    Convention
    ----------
    Along a field line $\phi = q\theta$, so the ripple term of
    ``toroidal_ripple_field`` oscillates with $dB/d\theta \sim N_\mathrm{TF}q\delta B_0$
    while the toroidal term's slope is $\epsilon B_0\sin\theta$; $\alpha^*$ is
    their ratio. It is zero at the midplanes, where wells always form for any
    ripple.

    Physical interpretation
    -----------------------
    Where the fine ripple modulation is steeper than the smooth $1/R$ slope,
    the field along the line has local minima that can trap particles with
    small $v_\parallel$ -- ripple trapping.

    Assumptions
    -----------
    Large aspect ratio, circular surfaces, $\phi = q\theta$ along the line;
    a local criterion, not a 3-D field analysis. It is first order in
    $\epsilon$: for the multiplicative ``toroidal_ripple_field`` the ripple
    slope carries a factor $1 - \epsilon\cos\theta$, and wells form where
    $\alpha^* < 1 - \epsilon\cos\theta$.

    See Also
    --------
    vaft.diagram.ripple_well_formation : the canonical diagram of this relation.

    References
    ----------
    .. [1] P. N. Yushmanov, Rev. Plasma Phys. 16 (1990) 117, Sec. 2.
    """
    n = _positive_int(n_tf, "n_tf")
    q = np.asarray(q, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if np.any(q <= 0.0) or np.any(delta <= 0.0):
        raise ValueError("q and delta must be positive")
    result = np.asarray(epsilon, dtype=float) * np.abs(np.sin(np.asarray(theta, dtype=float))) / (n * q * delta)
    return _scalar_or_array(result)


def ripple_trapping_pitch(delta):
    r"""Largest pitch $|v_\parallel/v|$ trapped in a ripple well of depth $\delta$.

    $$\left|\frac{v_\parallel}{v}\right|_\mathrm{trap} = \sqrt{\frac{2\delta}{1 + \delta}} \simeq \sqrt{2\delta}$$

    Parameters
    ----------
    delta : float or np.ndarray
        Ripple amplitude, in $[0, 1)$ [-].

    Returns
    -------
    float or np.ndarray
        Pitch $\xi = v_\parallel/v$ at the well bottom below which a particle
        cannot climb over the next ripple maximum [-].

    Raises
    ------
    ValueError
        ``delta`` lies outside $[0, 1)$.

    Convention
    ----------
    Measured at the well bottom $B = \bar B(1 - \delta)$, with the barrier
    $\bar B(1 + \delta)$; conservation of $\mu$ and energy gives
    $\xi^2 < 1 - B_\mathrm{min}/B_\mathrm{max}$. Ignores the $1/R$ tilt of the well,
    which only makes wells shallower (see ``ripple_well_parameter``).

    Physical interpretation
    -----------------------
    Only nearly perpendicular particles -- a pitch cone of half-width
    $\sim\sqrt{2\delta}$, a few per cent -- are ripple-trapped; they then drift
    vertically out of the plasma unless collisions or the well's end detrap them.

    Assumptions
    -----------
    A well of full depth $2\delta$; $\mu$ and energy conserved.

    References
    ----------
    .. [1] P. N. Yushmanov, Rev. Plasma Phys. 16 (1990) 117, Sec. 2.
    """
    delta = np.asarray(delta, dtype=float)
    if np.any(delta < 0.0) or np.any(delta >= 1.0):
        raise ValueError("delta must lie in [0, 1)")
    return _scalar_or_array(np.sqrt(2.0 * delta / (1.0 + delta)))


def gwb_stochastic_threshold(epsilon, q, dq_dr, rho, n_tf):
    r"""Goldston--White--Boozer ripple above which banana-tip motion is stochastic.

    $$\delta_\mathrm{GWB} = \left(\frac{\epsilon}{\pi N_\mathrm{TF}\,q}\right)^{3/2}\frac{1}{\rho\,q'}$$

    Parameters
    ----------
    epsilon : float or np.ndarray
        Inverse aspect ratio $r/R_0$ of the surface [-].
    q : float or np.ndarray
        Safety factor, a positive magnitude [-].
    dq_dr : float or np.ndarray
        $dq/dr$ in the physical minor radius, positive magnitude [1/m].
    rho : float or np.ndarray
        Larmor radius $v/\Omega$ with the full speed and the field at the
        surface, e.g. ``larmor_radius(q, m, v, B)`` [m].
    n_tf : int
        Number of TF coils [-].

    Returns
    -------
    float or np.ndarray
        Threshold ripple $\delta_\mathrm{GWB}$ [-].

    Raises
    ------
    ValueError
        ``n_tf`` is not a positive integer, or ``q``, ``dq_dr`` or ``rho`` is not positive.

    Convention
    ----------
    The ITER Physics Basis form of the GWB criterion: $\rho$ is the gyroradius
    with the total speed, $q' = dq/dr$ in metres of minor radius, and the
    coefficient is exactly the one written (no hidden $O(1)$ factor). Other
    papers fold factors of order unity into $\rho$ (toroidal vs total field,
    $v$ vs $v_\perp$); compare thresholds only within one convention.

    Physical interpretation
    -----------------------
    Each bounce the ripple kicks the banana tip radially by an amount $\propto\delta$;
    when the kick shifts the tip's toroidal precession phase by more than about
    a radian between bounces, successive kicks decorrelate and the tips random-walk
    out. Larger $\rho$ (faster particles), stronger shear $q'$ and higher $q$
    lower the threshold; $\epsilon$ alone raises it, and outer surfaces still go
    stochastic first because $\delta$ and $q$ grow outward much faster.

    Assumptions
    -----------
    Large aspect ratio, deeply trapped particles, a single $N_\mathrm{TF}$ harmonic,
    no collisions; an order-of-magnitude threshold, not a loss rate.

    Validity
    --------
    Orbit-following comparisons find stochastic losses near this threshold to
    within a factor of order one; use it to flag regimes, not to predict losses.

    See Also
    --------
    vaft.diagram.stochastic_ripple_orbit : the canonical diagram of this relation.

    References
    ----------
    .. [1] R. J. Goldston, R. B. White and A. H. Boozer, Phys. Rev. Lett. 47
           (1981) 647.
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2471, Ch. 5.
    """
    n = _positive_int(n_tf, "n_tf")
    q = np.asarray(q, dtype=float)
    dq_dr = np.asarray(dq_dr, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if np.any(q <= 0.0) or np.any(dq_dr <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("q, dq_dr and rho must be positive")
    result = (np.asarray(epsilon, dtype=float) / (np.pi * n * q)) ** 1.5 / (rho * dq_dr)
    return _scalar_or_array(result)


def gwb_stochasticity_parameter(delta, epsilon, q, dq_dr, rho, n_tf):
    r"""Ripple over its GWB threshold: stochastic banana tips where it exceeds 1.

    $$S_\mathrm{GWB} = \frac{\delta}{\delta_\mathrm{GWB}}$$

    Parameters
    ----------
    delta : float or np.ndarray
        Ripple amplitude on the surface [-].
    epsilon : float or np.ndarray
        Inverse aspect ratio $r/R_0$ [-].
    q : float or np.ndarray
        Safety factor [-].
    dq_dr : float or np.ndarray
        $dq/dr$ in physical minor radius [1/m].
    rho : float or np.ndarray
        Larmor radius, as in ``gwb_stochastic_threshold`` [m].
    n_tf : int
        Number of TF coils [-].

    Returns
    -------
    float or np.ndarray
        $S_\mathrm{GWB}$ [-].

    Raises
    ------
    ValueError
        As ``gwb_stochastic_threshold``, or ``delta`` is negative.

    Convention
    ----------
    The threshold of ``gwb_stochastic_threshold`` with its stated $\rho$ and
    $q'$ conventions.

    Physical interpretation
    -----------------------
    A regime flag: $S_\mathrm{GWB} > 1$ marks where trapped fast particles of
    that gyroradius lose their orbits to ripple stochasticity.

    See Also
    --------
    vaft.diagram.stochastic_ripple_orbit : the canonical diagram of this relation.

    References
    ----------
    .. [1] R. J. Goldston, R. B. White and A. H. Boozer, Phys. Rev. Lett. 47
           (1981) 647.
    """
    delta = np.asarray(delta, dtype=float)
    if np.any(delta < 0.0):
        raise ValueError("delta must be non-negative")
    return _scalar_or_array(delta / gwb_stochastic_threshold(epsilon, q, dq_dr, rho, n_tf))
