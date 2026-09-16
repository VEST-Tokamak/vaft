r"""Start-up physics from the gas fill to the avalanche.

Prefill, the Townsend coefficient and the Lloyd breakdown threshold: the
reduced kernels that decide whether a gas fill will break down under a given
toroidal electric field.  This is the pre-equilibrium phase of a discharge,
before any closed flux surface exists, so nothing here reads a flux map, knows
a COCOS, or substitutes for a field-line trace.

The module stops at the avalanche on purpose.  Burn-through and the
radiation-ionisation barrier are a later slice of the same work; plasma
inductance and the current ramp belong with the transformer dynamics.

Notation
--------
p           : neutral fill pressure                                  [Pa]
T_gas       : fill-gas kinetic temperature, at the wall              [K]
n           : neutral number density                                 [m^-3]
E_parallel  : electric field along the open field line, magnitude    [V/m]
L           : connection length of an open field line                [m]
alpha       : Townsend first ionisation coefficient                  [m^-1]
A, B        : Townsend similarity coefficients   [m^-1 Pa^-1], [V m^-1 Pa^-1]
E_BD        : Lloyd breakdown threshold field                        [V/m]
M_BD        : breakdown margin, E_parallel / E_BD                    [-]

Conventions
-----------
**Pressure is pascal.**  The Townsend and Lloyd coefficients are published per
torr; they are pinned here in that published form and a pascal input is
converted with :data:`vaft.formula.constants.PA_PER_TORR`.  Near the domain
edge the logarithm is small, so a rounded SI prefactor costs half a percent
exactly where a spherical tokamak's prefill sits.  VEST's barometry is stored
in pascal; the torr on a plot axis is a display choice.

**Gas temperature is kelvin, not electronvolts.**  This is the one place in
:mod:`vaft.formula` where that is so: a fill gas sits at wall temperature,
where $k_B$ is the natural constant, and it is not a plasma species.

**Electric fields here are magnitudes.**  The two flux-to-voltage kernels in
:mod:`vaft.formula.equilibrium` disagree on both flux normalisation and sign --
:func:`vaft.formula.equilibrium.loop_voltage_from_total_flux` is
$+2\pi\,\mathrm{d}\psi_b/\mathrm{d}t$ on a per-radian flux, while
:func:`vaft.formula.equilibrium.toroidal_electric_field` is
$-\dot\psi/2\pi R$ on a full-weber flux, tracked in #354.  An avalanche does
not care which way the field points, so nothing here propagates that
disagreement: every function below takes $|E_\parallel|$.

References
----------
.. [1] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.
.. [2] D. Mueller, Phys. Plasmas 20 (2013) 058101, Sec. II.
.. [3] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 4.2.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._exports import public_names
from .constants import K_BOLTZMANN, MU0, PA_PER_TORR

#: Lloyd's hydrogen/deuterium coefficients, in the torr units they are
#: published in.  These are the Townsend $A$ and $B$ for hydrogen: setting
#: $\alpha L = 1$ in the Townsend expression and solving for $E$ gives Lloyd's
#: threshold exactly, which is what ties the two functions below together.
_LLOYD_A_PER_M_TORR = 510.0
_LLOYD_B_V_PER_M_TORR = 1.25e4


def _maybe_scalar(value, *inputs):
    """Return a float when every input was scalar, mirroring ``neoclassical``."""
    if all(np.isscalar(item) or np.ndim(item) == 0 for item in inputs):
        return float(value)
    return value


def _require_positive(name, value):
    """Reject a non-finite or non-positive physical input by name."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(
            f"{name} must be finite and positive; got {value!r}"
        )
    return array


def neutral_density_from_pressure(p_Pa, T_gas_K=300.0):
    r"""Number density of a fill gas at a measured pressure and temperature.

    $$n = \frac{p}{k_B T_{\mathrm{gas}}}$$

    Parameters
    ----------
    p_Pa : float or np.ndarray
        Neutral fill pressure, finite and positive [Pa].
    T_gas_K : float or np.ndarray, optional
        Fill-gas kinetic temperature; default 300 K, room temperature [K].

    Returns
    -------
    float or np.ndarray
        Number density of the gas the gauge weighs [m^-3].

    Raises
    ------
    ValueError
        Non-finite, zero or negative pressure or temperature.

    Convention
    ----------
    Pressure in pascal and temperature in **kelvin** -- the one place in
    :mod:`vaft.formula` where a temperature is not in electronvolts, because
    this is a wall-temperature fill gas rather than a plasma species.  The
    300 K default is 0.02585 eV.  The result is the density of the particles
    the gauge weighs: for a $D_2$ fill that is $n_{D_2}$, the *molecular*
    density, and the deuteron inventory is twice it -- see
    :func:`atomic_inventory_from_molecular_gas`.  Nothing here reads a species
    label, so nothing here can infer that factor for you.

    Assumptions
    -----------
    Ideal gas in thermal equilibrium with the wall, uniform over the vessel,
    at the moment the gauge is read.

    Validity
    --------
    Machine-independent.  A tokamak prefill is $10^{-3}$ to $10^{-1}$ Pa at a
    few hundred kelvin, far inside the ideal-gas regime.  Nothing here survives
    ionisation: once the plasma starts consuming the fill, the neutral
    inventory is set by a particle balance rather than by a gauge reading.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, Sec. 2 (the DYON prefill inventory).

    See Also
    --------
    atomic_inventory_from_molecular_gas
    """
    pressure = _require_positive("p_Pa", p_Pa)
    temperature = _require_positive("T_gas_K", T_gas_K)
    return _maybe_scalar(pressure / (K_BOLTZMANN * temperature), p_Pa, T_gas_K)


def atomic_inventory_from_molecular_gas(n_molecular_m3, atoms_per_molecule=2):
    r"""Atom inventory of a molecular gas fill.

    $$n^{0} = a\,n_{\mathrm{mol}}$$

    Parameters
    ----------
    n_molecular_m3 : float or np.ndarray
        Molecular number density, finite and positive [m^-3].
    atoms_per_molecule : float or np.ndarray, optional
        Atoms each molecule contains; default 2 for a diatomic fill [-].

    Returns
    -------
    float or np.ndarray
        Atom number density of the same fill [m^-3].

    Raises
    ------
    ValueError
        Non-finite, zero or negative density or atom count.

    Convention
    ----------
    ``atoms_per_molecule`` is an explicit input and is never inferred: 2 for
    $H_2$, $D_2$ and $T_2$, 1 for a noble gas, 5 for $CH_4$.  The default is 2
    because a tokamak prefill is a diatomic hydrogen isotope, but a caller
    filling with helium must say 1.  Written together with
    :func:`neutral_density_from_pressure` this is DYON's
    $n_D^0(0) = 2p_0 / (k_B T_n)$.

    Physical interpretation
    -----------------------
    How many atoms the fill contains, which is what an ionisation balance
    counts, as against how many particles the pressure gauge weighs.

    Limitations
    -----------
    Bookkeeping, not chemistry: it says how many atoms the molecules contain,
    not how many are dissociated.  A cold prefill is fully molecular, and the
    energy cost of dissociating it belongs to the burn-through balance rather
    than to this count.  Getting the factor wrong is a factor of four in the
    analytic radiation-ionisation barrier, which goes as the square of this
    density.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, Sec. 2.
    """
    density = _require_positive("n_molecular_m3", n_molecular_m3)
    atoms = _require_positive("atoms_per_molecule", atoms_per_molecule)
    return _maybe_scalar(density * atoms, n_molecular_m3, atoms_per_molecule)


def townsend_ionization_coefficient(E_parallel, p_Pa, A, B):
    r"""Townsend first ionisation coefficient of a gas in a field.

    $$\alpha = A\,p\,\exp\!\left(-\frac{B\,p}{|E_\parallel|}\right)$$

    Parameters
    ----------
    E_parallel : float or np.ndarray
        Electric field along the field line; used as a magnitude [V/m].
    p_Pa : float or np.ndarray
        Neutral fill pressure, finite and positive [Pa].
    A : float or np.ndarray
        Townsend similarity coefficient of the gas [m^-1 Pa^-1].
    B : float or np.ndarray
        Townsend similarity coefficient of the gas [V m^-1 Pa^-1].

    Returns
    -------
    float or np.ndarray
        Ionisations per metre of electron drift [m^-1].

    Raises
    ------
    ValueError
        Non-finite or zero field, or non-finite or non-positive pressure.

    Convention
    ----------
    $A$ and $B$ are in SI-pressure units.  The literature almost always
    publishes them per centimetre and per torr: divide a published
    $A\,[\mathrm{cm^{-1}Torr^{-1}}]$ by $10^{-2}\times$
    :data:`vaft.formula.constants.PA_PER_TORR`, and a published
    $B\,[\mathrm{V\,cm^{-1}Torr^{-1}}]$ by the same factor.  The hydrogen
    values :func:`lloyd_breakdown_field` pins are
    $A = 510\ \mathrm{m^{-1}Torr^{-1}}$ and
    $B = 1.25\times10^{4}\ \mathrm{V\,m^{-1}Torr^{-1}}$, which are
    $3.8253\ \mathrm{m^{-1}Pa^{-1}}$ and $93.758\ \mathrm{V\,m^{-1}Pa^{-1}}$.
    $E_\parallel$ is used as a magnitude: an avalanche runs in whichever
    direction the field points.

    Physical interpretation
    -----------------------
    The similarity law: at fixed $E/p$ an electron sees the same number of
    ionising collisions per mean free path, so $\alpha/p$ is a function of
    $E/p$ alone.

    Validity
    --------
    Holds while an electron gains its ionisation energy in much less than a
    mean free path and loses nothing to attachment.  The exponential is a
    two-parameter *fit* over a limited band of $E/p$, and it has no upper
    saturation: the ionisation cross-section peaks and falls, so at high $E/p$
    the fit over-predicts $\alpha$, while at low $E/p$ it underestimates the
    contribution of the tail of the electron distribution.  Convert an
    operating point with :data:`vaft.formula.constants.PA_PER_TORR` before
    comparing it against a published band -- a tokamak prefill start-up sits
    near $10^{3}\ \mathrm{V\,cm^{-1}Torr^{-1}}$, at the top of where hydrogen
    coefficients are normally quoted, so the threshold it feeds is an
    extrapolation rather than an interpolation.  The coefficients are the
    caller's: this function fixes the functional form, not the gas.  Supplying
    values fitted outside the range they were fitted in is the usual way to get
    a confident wrong answer.

    References
    ----------
    .. [1] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 4.2.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.

    See Also
    --------
    lloyd_breakdown_field
    """
    field = np.abs(np.asarray(E_parallel, dtype=float))
    if not np.all(np.isfinite(field)) or np.any(field == 0.0):
        raise ValueError(
            f"E_parallel must be finite and non-zero; got {E_parallel!r}"
        )
    pressure = _require_positive("p_Pa", p_Pa)
    a_coef = np.asarray(A, dtype=float)
    b_coef = np.asarray(B, dtype=float)
    alpha = a_coef * pressure * np.exp(-b_coef * pressure / field)
    return _maybe_scalar(alpha, E_parallel, p_Pa, A, B)


def townsend_breakdown_field(p, connection_length_m, A, B):
    r"""Breakdown threshold field of a gas, by inverting the Townsend closure.

    $$E_{BD} = \frac{B\,p}{\ln\!\left(A\,p\,L\right)}$$

    which is :func:`townsend_ionization_coefficient` solved for the field at
    which exactly one avalanche length fits inside the connection length,
    $\alpha L = 1$.

    Parameters
    ----------
    p : float or np.ndarray
        Neutral fill pressure, finite and positive, **in whatever unit ``A``
        and ``B`` are published per** [Pa or Torr].
    connection_length_m : float or np.ndarray
        Connection length of an open field line, finite and positive [m].
    A : float
        Townsend similarity coefficient, finite and positive
        [m^-1 per unit of ``p``].
    B : float
        Townsend similarity coefficient, finite and positive
        [V m^-1 per unit of ``p``].

    Returns
    -------
    float or np.ndarray
        Threshold field, ``nan`` where $A\,p\,L \le 1$ [V/m].

    Raises
    ------
    ValueError
        Non-finite, zero or negative pressure, connection length or
        coefficient.

    Convention
    ----------
    **This function does not know what unit the pressure is in.**  ``A`` and
    ``B`` carry it, so the caller must pass ``p`` in the unit the coefficients
    were published per, and the returned field is in volts per metre either
    way because ``B`` supplies that.  The hydrogenic wrapper
    :func:`lloyd_breakdown_field` takes pascal and converts, which is the
    behaviour to copy for any other gas rather than re-deriving it here.

    The secondary-emission closure is absorbed into $\alpha L = 1$, as Lloyd
    does; a full Townsend/Paschen treatment keeping $\gamma_{se}$ explicit
    solves $\gamma_{se}(e^{\alpha L} - 1) = 1$ instead and is a different
    function, not a keyword on this one.

    Validity
    --------
    Empirical similarity fit.  ``A`` and ``B`` hold over the $E/p$ range the
    source measured them across, and nothing in this signature records that
    range -- the gas model that supplies them must.

    Limitations
    -----------
    Says nothing about pre-ionisation: ECRH assist lowers the effective
    threshold by seeding electrons, and this expression has no term for it.
    Says nothing about burn-through, which is a separate barrier a plasma that
    has broken down can still fail to cross.

    Numerical notes
    ---------------
    $A\,p\,L \le 1$ warns and returns ``nan`` elementwise rather than raising:
    the degenerate case is real rather than hypothetical and sits under a
    plotting path, so it blanks a point instead of taking down a figure.
    Exactly at $A\,p\,L = 1$ the expression has a pole, and ``nan`` is returned
    there too, because an infinite threshold is a feature of the fit and not a
    physical field.

    References
    ----------
    .. [1] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 4.2.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.

    See Also
    --------
    lloyd_breakdown_field
    townsend_ionization_coefficient
    """
    pressure = _require_positive("p", p)
    length = _require_positive("connection_length_m", connection_length_m)
    coeff_a = _require_positive("A", A)
    coeff_b = _require_positive("B", B)
    argument = coeff_a * pressure * length
    degenerate = argument <= 1.0
    if np.any(degenerate):
        warnings.warn(
            "the Townsend avalanche cannot close over this connection length "
            "(A p L <= 1), so no breakdown threshold exists; returning nan",
            RuntimeWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        field = coeff_b * pressure / np.log(argument)
    field = np.where(degenerate, np.nan, field)
    return _maybe_scalar(field, p, connection_length_m)


def lloyd_breakdown_field(p_Pa, connection_length_m):
    r"""Lloyd threshold toroidal field for hydrogenic tokamak breakdown.

    $$E_{BD} = \frac{B\,p}{\ln\!\left(A\,p\,L\right)},\qquad
      A = 510\ \mathrm{m^{-1}Torr^{-1}},\quad
      B = 1.25\times10^{4}\ \mathrm{V\,m^{-1}Torr^{-1}}$$

    the field at which one avalanche length fits inside the connection length:
    :func:`townsend_ionization_coefficient` evaluated here returns exactly
    $\alpha = 1/L$.

    Parameters
    ----------
    p_Pa : float or np.ndarray
        Neutral fill pressure, finite and positive [Pa].
    connection_length_m : float or np.ndarray
        Connection length of an open field line, finite and positive [m].

    Returns
    -------
    float or np.ndarray
        Threshold toroidal electric field, ``nan`` where $A\,p\,L \le 1$ [V/m].

    Raises
    ------
    ValueError
        Non-finite, zero or negative pressure or connection length.

    Convention
    ----------
    Pressure in **pascal**.  The coefficients are pinned in the torr units
    Lloyd publishes them in and the input is converted with
    :data:`vaft.formula.constants.PA_PER_TORR`, rather than hard-coding the
    rounded SI values 93.76 and 3.825: near the domain edge the logarithm is
    small and that rounding is worth half a percent, which is exactly where a
    spherical tokamak's prefill sits.  ``connection_length_m`` is the *actual*
    connection length, from a field-line trace against the wall; a scaling
    such as $a B_T / B_\perp$ is a teaching estimate of it, not a computation
    of it.

    Physical interpretation
    -----------------------
    The Paschen curve of a tokamak, with the electrode gap replaced by the
    connection length of an open field line.  At fixed $L$ it has a minimum in
    pressure at $A\,p\,L = e$: below that the gas is too thin to multiply,
    above it the electrons collide before they reach ionising energy.

    Validity
    --------
    Empirical fit.  Hydrogen and deuterium, with the secondary-emission
    closure absorbed into $\alpha L = 1$; Lloyd's own data are $10^{-5}$ to
    $10^{-3}$ torr over connection lengths of tens to hundreds of metres.
    Outside a hydrogen isotope the coefficients are simply wrong, and no
    argument of this function says so.

    Limitations
    -----------
    Returns ``nan`` below $A\,p\,L = 1$ rather than a number: there the
    avalanche cannot close at any field, so no threshold exists.  Says nothing
    about pre-ionisation -- ECRH assist lowers the effective threshold by
    seeding electrons, which this expression has no term for -- and nothing
    about burn-through, which is a separate barrier a plasma that has broken
    down can still fail to cross.

    Numerical notes
    ---------------
    $A\,p\,L \le 1$ warns and returns ``nan`` elementwise rather than raising:
    the degenerate case is real rather than hypothetical and sits under a
    plotting path, so it blanks a point instead of taking down a figure.
    Exactly at $A\,p\,L = 1$ the expression has a pole, and ``nan`` is returned
    there too, because an infinite threshold is a feature of the fit and not a
    physical field.

    References
    ----------
    .. [1] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.
    .. [2] D. Mueller, Phys. Plasmas 20 (2013) 058101, Sec. II.

    See Also
    --------
    townsend_ionization_coefficient
    breakdown_margin
    vaft.formula.equilibrium.toroidal_electric_field
    """
    p_torr = _require_positive("p_Pa", p_Pa) / PA_PER_TORR
    field = townsend_breakdown_field(
        p_torr,
        connection_length_m,
        _LLOYD_A_PER_M_TORR,
        _LLOYD_B_V_PER_M_TORR,
    )
    return _maybe_scalar(field, p_Pa, connection_length_m)


def breakdown_margin(E_parallel, E_breakdown):
    r"""Available drive relative to the Lloyd breakdown threshold.

    $$M_{BD} = \frac{|E_\parallel|}{E_{BD}}$$

    Parameters
    ----------
    E_parallel : float or np.ndarray
        Electric field along the field line; used as a magnitude [V/m].
    E_breakdown : float or np.ndarray
        Threshold field, typically from
        :func:`lloyd_breakdown_field`; ``nan`` propagates [V/m].

    Returns
    -------
    float or np.ndarray
        Ratio of available drive to threshold [-].

    Convention
    ----------
    $E_\parallel$ is used as a **magnitude**.  The two flux-to-voltage kernels
    in :mod:`vaft.formula.equilibrium` disagree on sign as well as on flux
    normalisation, tracked in #354, so a signed field arriving here cannot be
    trusted to carry the meaning its sign implies.  An avalanche runs in
    whichever direction the field points, so the absolute value is the physics
    rather than a patch over the disagreement.  It does mean this function
    cannot tell a caller that their flux convention is wrong.

    Physical interpretation
    -----------------------
    How much inductive drive is available relative to the least that could
    start an avalanche, under the reduced Lloyd model.

    Limitations
    -----------
    Not a success classifier.  $M_{BD} > 1$ is necessary within this model and
    is not sufficient in a machine: it says nothing about pre-ionisation, about
    impurity or wall conditioning, about whether the null persists long enough,
    or about burn-through, which is a second barrier past this one.  Discharges
    with $M_{BD} > 1$ fail routinely, and discharges below it break down with
    ECRH assist.  Read it as an ordering of operating points, not as a
    prediction.  ``nan`` wherever the threshold is ``nan``, which is how the
    no-threshold region of :func:`lloyd_breakdown_field` propagates.

    References
    ----------
    .. [1] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.
    .. [2] D. Mueller, Phys. Plasmas 20 (2013) 058101, Sec. II.

    See Also
    --------
    lloyd_breakdown_field
    """
    field = np.abs(np.asarray(E_parallel, dtype=float))
    threshold = np.asarray(E_breakdown, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = field / threshold
    return _maybe_scalar(margin, E_parallel, E_breakdown)



# ------------------------------------------------------------------
# After the avalanche: a current channel, and whether it can close
# ------------------------------------------------------------------

def startup_geometry_from_limiter_radii(R_inboard_m, R_outboard_m):
    r"""Geometric centre and minor radius of the limiter aperture.

    $$R_0 = \frac{R_{\mathrm{in}} + R_{\mathrm{out}}}{2},\qquad
      a = \frac{R_{\mathrm{out}} - R_{\mathrm{in}}}{2}$$

    Parameters
    ----------
    R_inboard_m : float or np.ndarray
        Inboard limiter radius at the midplane, finite and positive [m].
    R_outboard_m : float or np.ndarray
        Outboard limiter radius at the midplane, finite and positive and
        greater than ``R_inboard_m`` [m].

    Returns
    -------
    tuple of (float or np.ndarray, float or np.ndarray)
        Geometric major radius and minor radius of the aperture [m].

    Raises
    ------
    ValueError
        Non-finite or non-positive radii, or an outboard radius that does not
        exceed the inboard one.

    Convention
    ----------
    **A machine aperture, not a plasma.**  The other reduced start-up kernels
    need an $R_0$ and an $a$ before a magnetic axis or an LCFS exists, and this
    supplies them from the vessel alone: the midplane limiter intersections.
    The moment a qualified equilibrium exists, its own boundary supersedes
    this -- use :func:`vaft.formula.equilibrium.elongation_from_RZ_boundary`
    and the triangularity helpers on the real contour instead.

    Limitations
    -----------
    Ignores elongation entirely: it is a midplane chord, so a tall vessel and
    a round one with the same midplane give the same answer.  It also assumes
    the plasma fills the aperture, which during start-up is exactly what is
    not yet true.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 1.4.

    See Also
    --------
    plasma_self_field_from_I_p_a
    vertical_field_from_I_p_R0_a_beta_p_li
    """
    inboard = _require_positive("R_inboard_m", R_inboard_m)
    outboard = _require_positive("R_outboard_m", R_outboard_m)
    if np.any(outboard <= inboard):
        raise ValueError(
            "R_outboard_m must exceed R_inboard_m; got "
            f"{R_outboard_m!r} and {R_inboard_m!r}"
        )
    major = 0.5 * (inboard + outboard)
    minor = 0.5 * (outboard - inboard)
    return (
        _maybe_scalar(major, R_inboard_m, R_outboard_m),
        _maybe_scalar(minor, R_inboard_m, R_outboard_m),
    )


def plasma_self_field_from_I_p_a(I_p_A, a_m):
    r"""Poloidal field a circular current channel makes at its own edge.

    $$B_{p,\mathrm{self}}(a) = \frac{\mu_0 I_p}{2\pi a}$$

    Parameters
    ----------
    I_p_A : float or np.ndarray
        Plasma current, magnitude, finite and positive [A].
    a_m : float or np.ndarray
        Minor radius of the current channel, finite and positive [m].

    Returns
    -------
    float or np.ndarray
        Poloidal field magnitude at the channel edge [T].

    Raises
    ------
    ValueError
        Non-finite, zero or negative current or minor radius.

    Convention
    ----------
    A magnitude, like everything else in this module: Ampere's law around a
    straight wire, with no sign and no COCOS.  During start-up the quantity it
    is compared against is a *stray* field whose orientation is a property of
    the coil set, so propagating a sign through this would imply a shared
    convention that does not exist yet.

    Assumptions
    -----------
    Straight, circular, uniform current channel; the toroidal correction is
    $O(a/R)$ and is deliberately absent, because the comparison this feeds is
    an order-of-magnitude one.

    Limitations
    -----------
    Says nothing about where the field points, so it cannot by itself say
    whether the plasma field *cancels* or *reinforces* a stray field.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.1.

    See Also
    --------
    flux_closure_margin_from_E_t_a_B_stray_eta
    """
    current = _require_positive("I_p_A", I_p_A)
    minor = _require_positive("a_m", a_m)
    field = MU0 * current / (2.0 * np.pi * minor)
    return _maybe_scalar(field, I_p_A, a_m)


def vertical_field_from_I_p_R0_a_beta_p_li(I_p_A, R0_m, a_m, beta_p, li, kappa=1.0):
    r"""Vertical field that holds a circular plasma in radial equilibrium.

    $$B_v = \frac{\mu_0 I_p}{4\pi R_0}
      \left[\ln\!\left(\frac{8R_0}{a\,l_\kappa}\right)
      + \beta_p + \frac{l_i}{2} - \frac{3}{2}\right],\qquad
      l_\kappa = \sqrt{\frac{1 + \kappa^2}{2}}$$

    Parameters
    ----------
    I_p_A : float or np.ndarray
        Plasma current, magnitude, finite and positive [A].
    R0_m : float or np.ndarray
        Major radius of the current channel, finite and positive [m].
    a_m : float or np.ndarray
        Minor radius, finite and positive and smaller than ``R0_m`` [m].
    beta_p : float or np.ndarray
        Poloidal beta [-].
    li : float or np.ndarray
        Normalised internal inductance, the same normalisation the
        equilibrium layer reports [-].
    kappa : float or np.ndarray, optional
        Elongation; default 1, a circular cross-section [-].

    Returns
    -------
    float or np.ndarray
        Required vertical field magnitude [T].

    Raises
    ------
    ValueError
        Non-finite or non-positive current, major radius, minor radius or
        elongation, or a minor radius that is not smaller than the major one.

    Convention
    ----------
    **A magnitude, and the orientation is settled rather than open.**  Mitarai
    writes Eq. (1.3) signed, $B_{VE} = -(\mu_0 I_p/4\pi R)[\cdots]$: the
    equilibrium field opposes $I_p$, because it has to push the ring back
    against its own outward hoop force.  So the field this returns points
    *anti-parallel* to the field a positive $I_p$ would make on the axis, and a
    caller turning it into a coil current takes the sign from that statement
    plus the machine description, not from a COCOS -- there is no flux map here
    for a COCOS to describe.

    The magnitude is returned rather than the signed value because the
    comparison this feeds during start-up is against a stray field whose
    orientation belongs to the coil set, and carrying a sign through would
    imply a shared frame that does not exist before an equilibrium does.

    ``li`` is whichever normalisation the caller's equilibrium reports.  The
    bracket is $O(1)$ and the $l_i/2$ term is a fraction of it, so the choice
    between $l_i$ and $l_{i3}$ moves $B_v$ by a few percent -- small enough to
    ignore in a start-up estimate, large enough that a reported number should
    say which was used.

    Elongation enters only through $l_\kappa = \sqrt{(1+\kappa^2)/2}$ inside the
    logarithm, which is Mitarai's form: the shape shortens the effective minor
    radius rather than changing the bracket.  At $\kappa = 1$ it is exactly the
    circular Shafranov result, and at the $\kappa = 1.7$ of an ITER-FEAT-like
    case it lowers $B_v$ by about 7 %.

    Assumptions
    -----------
    High aspect ratio: the Shafranov result, with shaping carried only by
    $l_\kappa$.  A spherical tokamak violates the aspect-ratio expansion, and
    the Hirshman external inductance is the route to that case --
    see :func:`plasma_external_inductance_hirshman_from_R_eps_kappa` and
    :func:`d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li`.

    Validity
    --------
    $a/R_0 \ll 1$.  At the $\varepsilon \approx 0.7$ of a spherical tokamak the
    expression is being used far outside the expansion it came from, and its
    value there is indicative rather than quantitative.

    Limitations
    -----------
    Reduced-order.  Once a qualified free-boundary equilibrium exists, it
    supersedes this entirely; this expression exists for the window in which a
    current channel has formed but an equilibrium reconstruction has not.

    References
    ----------
    .. [1] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2,
           Consultants Bureau (1966), p. 103.
    .. [2] O. Mitarai, R. Yoshino and K. Ushigusa, Nucl. Fusion 42 (2002) 1257,
           Eq. (1.3); the elongation factor is Eq. (1.2).
    .. [3] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.2.

    See Also
    --------
    startup_geometry_from_limiter_radii
    plasma_self_field_from_I_p_a
    plasma_external_inductance_hirshman_from_R_eps_kappa
    """
    current = _require_positive("I_p_A", I_p_A)
    major = _require_positive("R0_m", R0_m)
    minor = _require_positive("a_m", a_m)
    if np.any(minor >= major):
        raise ValueError(
            f"a_m must be smaller than R0_m; got {a_m!r} and {R0_m!r}"
        )
    shape = _require_positive("kappa", kappa)
    beta = np.asarray(beta_p, dtype=float)
    inductance = np.asarray(li, dtype=float)
    l_kappa = np.sqrt(0.5 * (1.0 + shape**2))
    bracket = (
        np.log(8.0 * major / (minor * l_kappa)) + beta + 0.5 * inductance - 1.5
    )
    field = MU0 * current * bracket / (4.0 * np.pi * major)
    return _maybe_scalar(field, I_p_A, R0_m, a_m, beta_p, li, kappa)


def flux_closure_margin_from_E_t_a_B_stray_eta(E_t, a_m, B_stray_T, eta_ohm_m):
    r"""Order-of-magnitude margin for a current channel to overcome stray field.

    $$M_{\mathrm{FC}}
      = \frac{\mu_0}{2}\,\frac{E_t\,a}{B_{\mathrm{stray}}\,\eta}$$

    the ratio $B_{p,\mathrm{self}}/B_{\mathrm{stray}}$ after substituting the
    reduced Ohmic current $I_p \simeq \pi a^2 E_t/\eta$, so that $M_{FC} > 1$
    is the same statement as $E_t a/(B_{\mathrm{stray}}\eta) > 2/\mu_0$.

    Parameters
    ----------
    E_t : float or np.ndarray
        Toroidal electric field driving the channel, magnitude, finite and
        positive [V/m].
    a_m : float or np.ndarray
        Minor radius of the channel, finite and positive [m].
    B_stray_T : float or np.ndarray
        Residual stray **poloidal-field magnitude**, finite and positive [T].
    eta_ohm_m : float or np.ndarray
        Plasma resistivity, finite and positive [Ohm m].

    Returns
    -------
    float or np.ndarray
        Ratio of the channel's own edge field to the stray field [-].

    Raises
    ------
    ValueError
        Non-finite, zero or negative field, radius, stray field or resistivity.

    Convention
    ----------
    ``B_stray_T`` is the **magnitude of the residual poloidal field**, not its
    vertical component alone: during start-up the radial component is of the
    same order, and taking only $B_z$ flatters the margin.  Compose it with
    :func:`vaft.formula.equilibrium.poloidal_field_magnitude`.

    The legacy gauss form of this criterion,
    $E_t a/(B_{\mathrm{stray}}[\mathrm{G}]\,\eta) > 1.6\times10^2$, is the same
    inequality with $2/\mu_0 = 1.59\times10^6$ carrying the $10^{-4}$ of the
    unit change; it is quoted here only so a reader meeting it in the
    literature can recognise it, and this function takes tesla.

    Assumptions
    -----------
    Uniform current density over a circular channel, a purely Ohmic current
    with no inductive or bootstrap contribution, and a resistivity that is a
    single number rather than a profile.  Each of those is wrong in detail
    during start-up; together they make this a scaling, not a prediction.

    Limitations
    -----------
    **Not a closed-flux-surface detector.**  Flux closure is a statement about
    the topology of the 2-D poloidal flux map, and no scalar ratio decides it.
    A margin above one means the plasma's own field has become comparable to
    what is trying to prevent closure, which is a necessary condition and not
    a sufficient one.

    References
    ----------
    .. [1] D. Mueller, Phys. Plasmas 20 (2013) 058101, Sec. II.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 4.

    See Also
    --------
    plasma_self_field_from_I_p_a
    vaft.formula.equilibrium.poloidal_field_magnitude
    """
    field = _require_positive("E_t", E_t)
    minor = _require_positive("a_m", a_m)
    stray = _require_positive("B_stray_T", B_stray_T)
    resistivity = _require_positive("eta_ohm_m", eta_ohm_m)
    margin = 0.5 * MU0 * field * minor / (stray * resistivity)
    return _maybe_scalar(margin, E_t, a_m, B_stray_T, eta_ohm_m)



#: Hirshman and Neilson's fit coefficients for the external inductance of a
#: shaped plasma, in the order they appear in $a(\epsilon)$ and $b(\epsilon)$.
#: Pinned here rather than inlined because the two derivative helpers below
#: differentiate the same fit and must not drift from it.
_HIRSHMAN_A = (1.81, 2.05, 9.25, 1.21)
_HIRSHMAN_B = (0.73, 2.0, 6.0, 3.7)

#: How the minor radius is held while the external inductance is
#: differentiated with respect to the major radius.  Each entry is
#: $R\,\mathrm{d}\epsilon/\mathrm{d}R$, which is the only way the choice enters.
_MINOR_RADIUS_CONVENTIONS = {
    "fixed": lambda eps: -eps,
    "inboard": lambda eps: 1.0 - eps,
    "outboard": lambda eps: -(1.0 + eps),
}


def plasma_external_inductance_hirshman_from_R_eps_kappa(R_m, epsilon, kappa):
    r"""External inductance of a shaped plasma, Hirshman and Neilson's fit.

    $$L_e = \mu_0 R\,\frac{a(\epsilon)\,(1 - \epsilon)}
      {(1 - \epsilon) + b(\epsilon)\,\kappa}$$

    $$a(\epsilon) = \left(1 + 1.81\sqrt\epsilon + 2.05\epsilon\right)
      \ln\!\frac{8}{\epsilon} - \left(2.0 + 9.25\sqrt\epsilon
      - 1.21\epsilon\right)$$

    $$b(\epsilon) = 0.73\sqrt\epsilon\left(1 + 2\epsilon^4 - 6\epsilon^5
      + 3.7\epsilon^6\right)$$

    Parameters
    ----------
    R_m : float or np.ndarray
        Major radius, finite and positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio $a/R$, finite and in $(0, 1)$ [-].
    kappa : float or np.ndarray
        Elongation, finite and positive [-].

    Returns
    -------
    float or np.ndarray
        External inductance [H].

    Raises
    ------
    ValueError
        Non-finite or non-positive input, or an inverse aspect ratio that is
        not below one.

    Convention
    ----------
    **External only.**  The plasma's own internal inductance is the separate
    $\mu_0 R\,l_i/2$ that :func:`plasma_inductance_hirshman_from_R_eps_kappa_li`
    adds; splitting them is what lets the internal term carry whichever $l_i$
    normalisation the caller's equilibrium reports.

    This is the fit the low-aspect-ratio start-up literature reaches for when
    the circular $\ln(8R/a) - 2$ form runs out.  It is often met under
    Mitarai's name, because his vertical-field expression is where it is
    usually substituted, but the inductance itself is Hirshman and Neilson's;
    Mitarai's own Eq. (1.2) is the circular form with the
    $l_\kappa = \sqrt{(1+\kappa^2)/2}$ correction that
    :func:`vertical_field_from_I_p_R0_a_beta_p_li` carries.

    Validity
    --------
    An empirical fit over $\epsilon$ and $\kappa$, unlike the circular form,
    which is an expansion.  It therefore stays usable at the
    $\epsilon \approx 0.7$ of a spherical tokamak, which is the reason to
    prefer it there.

    Limitations
    -----------
    Shaping enters only through $\kappa$: no triangularity, no squareness.  The
    fit says nothing about where the plasma sits, only about the inductance of
    a boundary of that shape.

    References
    ----------
    .. [1] S. P. Hirshman and G. H. Neilson, Phys. Fluids 29 (1986) 790.
    .. [2] O. Mitarai, R. Yoshino and K. Ushigusa, Nucl. Fusion 42 (2002) 1257.

    See Also
    --------
    plasma_inductance_hirshman_from_R_eps_kappa_li
    d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li
    """
    major = _require_positive("R_m", R_m)
    eps = _require_positive("epsilon", epsilon)
    shape = _require_positive("kappa", kappa)
    if np.any(eps >= 1.0):
        raise ValueError(f"epsilon must be below 1; got {epsilon!r}")
    numerator = _hirshman_a(eps) * (1.0 - eps)
    denominator = (1.0 - eps) + _hirshman_b(eps) * shape
    return _maybe_scalar(MU0 * major * numerator / denominator, R_m, epsilon, kappa)


def plasma_inductance_hirshman_from_R_eps_kappa_li(R_m, epsilon, kappa, li):
    r"""Total plasma inductance, Hirshman external plus the internal term.

    $$L_p = \mu_0 R\left[\frac{a(\epsilon)(1 - \epsilon)}
      {(1 - \epsilon) + b(\epsilon)\kappa} + \frac{l_i}{2}\right]$$

    Parameters
    ----------
    R_m : float or np.ndarray
        Major radius, finite and positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio $a/R$, finite and in $(0, 1)$ [-].
    kappa : float or np.ndarray
        Elongation, finite and positive [-].
    li : float or np.ndarray
        Normalised internal inductance [-].

    Returns
    -------
    float or np.ndarray
        Total plasma inductance [H].

    Raises
    ------
    ValueError
        Non-finite or non-positive radius, inverse aspect ratio or elongation,
        or an inverse aspect ratio that is not below one.

    Convention
    ----------
    $l_i$ is whichever normalisation the caller's equilibrium reports, and it
    enters as the dimensional $\mu_0 R\,l_i/2$.  That is the Romero/ITER
    convention $l_i = 2L_i/(\mu_0 R)$ read backwards, so a caller holding a
    dimensional $L_i$ should divide rather than pass it here.

    Limitations
    -----------
    A lumped inductance for a circuit model.  It is not the flux-surface
    quantity an equilibrium code reports, and the two coincide only to the
    accuracy of the fit.

    References
    ----------
    .. [1] S. P. Hirshman and G. H. Neilson, Phys. Fluids 29 (1986) 790.
    .. [2] O. Mitarai, R. Yoshino and K. Ushigusa, Nucl. Fusion 42 (2002) 1257,
           Eq. (1.2).

    See Also
    --------
    plasma_external_inductance_hirshman_from_R_eps_kappa
    d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li
    """
    external = plasma_external_inductance_hirshman_from_R_eps_kappa(
        R_m, epsilon, kappa
    )
    major = np.asarray(R_m, dtype=float)
    internal = MU0 * major * 0.5 * np.asarray(li, dtype=float)
    return _maybe_scalar(external + internal, R_m, epsilon, kappa, li)


def d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li(
    R_m, epsilon, kappa, li, minor_radius="fixed"
):
    r"""Radial derivative of the Hirshman plasma inductance.

    $$\frac{\partial L_p}{\partial R} = \mu_0\frac{N}{D}
      + \mu_0\left(R\frac{\mathrm{d}\epsilon}{\mathrm{d}R}\right)
        \left[\frac{N'}{D} - \frac{N D'}{D^2}\right]
      + \mu_0\frac{l_i}{2}$$

    with $N = a(\epsilon)(1-\epsilon)$ and $D = (1-\epsilon) + b(\epsilon)\kappa$.

    Parameters
    ----------
    R_m : float or np.ndarray
        Major radius, finite and positive [m].
    epsilon : float or np.ndarray
        Inverse aspect ratio $a/R$, finite and in $(0, 1)$ [-].
    kappa : float or np.ndarray
        Elongation, finite and positive [-].
    li : float or np.ndarray
        Normalised internal inductance, held fixed by the derivative [-].
    minor_radius : str, optional
        What is held while $R$ varies: ``'fixed'`` (default) keeps $a$,
        ``'inboard'`` keeps the inboard limiter so $a = R - R_{\min}$, and
        ``'outboard'`` keeps the outboard limiter so $a = R_{\max} - R$ [str].

    Returns
    -------
    float or np.ndarray
        $\partial L_p/\partial R$ [H/m].

    Raises
    ------
    ValueError
        Non-finite or non-positive input, an inverse aspect ratio that is not
        below one, or an unknown ``minor_radius``.

    Convention
    ----------
    **The minor-radius convention changes the sign, not just the size.**  It
    enters only through $R\,\mathrm{d}\epsilon/\mathrm{d}R$, which is
    $-\epsilon$ for a fixed minor radius, $1-\epsilon$ for an inboard-limited
    plasma and $-(1+\epsilon)$ for an outboard-limited one.  At
    $\epsilon = 0.3$, $\kappa = 1$, $l_i = 0$ the three give $+2.89$, $-1.80$
    and $+7.58\ \mathrm{\mu H/m}$: a radial force balance solved with the wrong
    one does not merely misestimate, it pushes the plasma the other way.

    ``'fixed'`` is the default because it is what reproduces the reduced
    vertical-field expressions: substituting the circular inductance into
    $B_{VE} = -(\mu_0 I_p/4\pi R)[\mu_0^{-1}\partial L_p/\partial R + \beta_p
    - 1/2]$ with $a$ held gives Mitarai's Eq. (1.3) exactly.  A solver that
    moves the plasma against a limiter wants one of the other two.

    $l_i$ is held fixed, so the $\mu_0 l_i/2$ term is the derivative of the
    internal part alone.  A current profile that redistributes while the plasma
    moves contributes a $\mu_0 R\,\dot l_i/2$ this does not carry.

    Limitations
    -----------
    Differentiates the fit, not the plasma: it is only as good as Hirshman's
    $a$ and $b$, and inherits their shaping limits.

    Numerical notes
    ---------------
    Analytic, not a finite difference: $a'$ and $b'$ are differentiated in
    closed form, so the result is exact to machine precision rather than
    limited by a step size.

    References
    ----------
    .. [1] S. P. Hirshman and G. H. Neilson, Phys. Fluids 29 (1986) 790.
    .. [2] O. Mitarai, R. Yoshino and K. Ushigusa, Nucl. Fusion 42 (2002) 1257,
           Eq. (1.2) and Eq. (1.3).

    See Also
    --------
    plasma_inductance_hirshman_from_R_eps_kappa_li
    vertical_field_from_I_p_R0_a_beta_p_li
    """
    _require_positive("R_m", R_m)
    eps = _require_positive("epsilon", epsilon)
    shape = _require_positive("kappa", kappa)
    if np.any(eps >= 1.0):
        raise ValueError(f"epsilon must be below 1; got {epsilon!r}")
    if minor_radius not in _MINOR_RADIUS_CONVENTIONS:
        raise ValueError(
            f"unknown minor_radius {minor_radius!r}; choose one of "
            f"{sorted(_MINOR_RADIUS_CONVENTIONS)}"
        )
    numerator = _hirshman_a(eps) * (1.0 - eps)
    denominator = (1.0 - eps) + _hirshman_b(eps) * shape
    d_numerator = _hirshman_da(eps) * (1.0 - eps) - _hirshman_a(eps)
    d_denominator = -1.0 + _hirshman_db(eps) * shape
    core = d_numerator / denominator - numerator * d_denominator / denominator**2
    r_deps_dr = _MINOR_RADIUS_CONVENTIONS[minor_radius](eps)
    derivative = MU0 * (
        numerator / denominator
        + r_deps_dr * core
        + 0.5 * np.asarray(li, dtype=float)
    )
    return _maybe_scalar(derivative, R_m, epsilon, kappa, li)


def _hirshman_a(eps):
    """Hirshman's $a(\\epsilon)$."""
    a1, a2, a3, a4 = _HIRSHMAN_A
    root = np.sqrt(eps)
    return (1.0 + a1 * root + a2 * eps) * np.log(8.0 / eps) - (
        2.0 + a3 * root - a4 * eps
    )


def _hirshman_da(eps):
    r"""$\\mathrm{d}a/\\mathrm{d}\\epsilon$, in closed form.

    Two coefficients here are worth stating because they are easy to get
    wrong and the error is invisible: the $1/\\sqrt\\epsilon$ term is
    $a_1 + a_3/2 = 6.435$, and the constant is $a_2 - a_4 = 0.84$.  Both are
    checked against a finite difference of $a(\\epsilon)$ in the tests, which
    is the only way to be sure of them -- reading the algebra is not.
    """
    a1, a2, a3, a4 = _HIRSHMAN_A
    root = np.sqrt(eps)
    return (0.5 * a1 / root + a2) * np.log(8.0 / eps) - (
        (a2 - a4) + 1.0 / eps + (a1 + 0.5 * a3) / root
    )


def _hirshman_b(eps):
    """Hirshman's $b(\\epsilon)$."""
    b1, b2, b3, b4 = _HIRSHMAN_B
    return b1 * np.sqrt(eps) * (1.0 + b2 * eps**4 - b3 * eps**5 + b4 * eps**6)


def _hirshman_db(eps):
    r"""$\\mathrm{d}b/\\mathrm{d}\\epsilon$, in closed form."""
    b1, b2, b3, b4 = _HIRSHMAN_B
    return (b1 / np.sqrt(eps)) * (
        0.5 + 4.5 * b2 * eps**4 - 5.5 * b3 * eps**5 + 6.5 * b4 * eps**6
    )



# ------------------------------------------------------------------
# Before the avalanche: can an EC-born electron stay?
# ------------------------------------------------------------------

def ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max(R_S_m, R_LIN_m, R_C_m, Z_max_m):
    r"""Slope of the Ejiri confined-orbit boundary in velocity space.

    $$\alpha = \max\!\left[
      \sqrt{\frac{R_{\mathrm{LIN}}}{R_S - R_{\mathrm{LIN}}}},\;
      \sqrt{\frac{2R_CR_S - Z_{\max}^2}{Z_{\max}^2}}\right]$$

    the boundary being $v_\perp = \alpha|v_\parallel|$: an electron born above
    that line is mirror-trapped, one below it is lost.

    Parameters
    ----------
    R_S_m : float or np.ndarray
        Major radius the particle starts at, finite and positive [m].
    R_LIN_m : float or np.ndarray
        Inboard limiter radius at the midplane, finite and positive and
        smaller than ``R_S_m`` [m].
    R_C_m : float or np.ndarray
        Radius of curvature of the poloidal field line on the midplane, finite
        and positive [m].
    Z_max_m : float or np.ndarray
        Effective vertical extent available for mirror confinement, finite and
        positive [m].

    Returns
    -------
    float or np.ndarray
        Slope of the confined-orbit boundary [-].

    Raises
    ------
    ValueError
        Non-finite or non-positive input, or a starting radius that does not
        exceed the inboard limiter radius.

    Convention
    ----------
    The two branches are two distinct loss channels and the *larger* wins,
    because a particle must survive both: the first is loss to the inboard
    limiter, the second loss along a field line whose curvature carries it past
    the vertical extent.  Taking the maximum therefore makes $\alpha$ the
    stricter of the two conditions, not an average of them.

    Assumptions
    -----------
    Low-energy, low-temperature single particles; the field line is
    approximated near the midplane by $R(Z) \simeq R_S - Z^2/(2R_C)$, which is
    what makes $R_C$ the only curvature input.

    Validity
    --------
    Ejiri's derivation is for the pre-breakdown or very early start-up
    configuration, where the field is essentially the vacuum field.  It says
    nothing once a plasma current shapes the field it is quoting.

    Limitations
    -----------
    Geometry only: no EC resonance, no power deposition, no collisions, so
    this cannot predict breakdown success.  It answers the narrower question of
    whether the configuration *could* hold an electron that an EC wave
    produced.  The effective $Z_{\max}$ that reproduces the paper's numerical
    orbit boundary can differ from the literal limiter coordinate, because the
    analytic model uses the parabolic approximation above -- so this is an
    Ejiri-inspired geometric proxy rather than a reproduction of the full
    orbit calculation.

    References
    ----------
    .. [1] A. Ejiri and Y. Takase, Nucl. Fusion 47 (2007) 403, Sec. 3.

    See Also
    --------
    ejiri_f3_from_alpha
    """
    start = _require_positive("R_S_m", R_S_m)
    limiter = _require_positive("R_LIN_m", R_LIN_m)
    curvature = _require_positive("R_C_m", R_C_m)
    extent = _require_positive("Z_max_m", Z_max_m)
    if np.any(limiter >= start):
        raise ValueError(
            "R_LIN_m must be smaller than R_S_m; got "
            f"{R_LIN_m!r} and {R_S_m!r}"
        )
    inboard_branch = np.sqrt(limiter / (start - limiter))
    curvature_branch = np.sqrt(
        np.maximum(2.0 * curvature * start - extent**2, 0.0) / extent**2
    )
    alpha = np.maximum(inboard_branch, curvature_branch)
    return _maybe_scalar(alpha, R_S_m, R_LIN_m, R_C_m, Z_max_m)


def ejiri_f3_from_alpha(alpha):
    r"""Low-temperature geometry factor of the Ejiri mirror proxy.

    $$F_3(\alpha) = \frac{2 + 3\alpha^2}{2\left(1 + \alpha^2\right)^{3/2}}$$

    the confined fraction of an isotropic low-temperature population under the
    boundary $v_\perp = \alpha|v_\parallel|$.

    Parameters
    ----------
    alpha : float or np.ndarray
        Slope of the confined-orbit boundary, finite and non-negative [-].

    Returns
    -------
    float or np.ndarray
        Geometry factor [-].

    Raises
    ------
    ValueError
        Non-finite or negative slope.

    Convention
    ----------
    $F_3(0) = 1$: a zero slope confines everything, because the boundary has
    collapsed onto the parallel axis.  $F_3$ falls monotonically from there and
    tends to $3/(2\alpha)$ for large $\alpha$, so a steeper boundary confines
    less -- a larger $\alpha$ is a *worse* configuration, which is the opposite
    of what the symbol suggests at a glance.

    Assumptions
    -----------
    Isotropic, low-temperature velocity distribution; the same single-particle
    picture the slope came from.

    Limitations
    -----------
    A relative figure of merit for comparing configurations, not an absolute
    confined fraction of a real EC-heated population, whose distribution is
    neither isotropic nor low-temperature.

    References
    ----------
    .. [1] A. Ejiri and Y. Takase, Nucl. Fusion 47 (2007) 403, Sec. 3.

    See Also
    --------
    ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max
    """
    slope = np.asarray(alpha, dtype=float)
    if not np.all(np.isfinite(slope)) or np.any(slope < 0.0):
        raise ValueError(f"alpha must be finite and non-negative; got {alpha!r}")
    factor = (2.0 + 3.0 * slope**2) / (2.0 * (1.0 + slope**2) ** 1.5)
    return _maybe_scalar(factor, alpha)


__all__ = public_names(globals())
