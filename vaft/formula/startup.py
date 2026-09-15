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
from .constants import K_BOLTZMANN, PA_PER_TORR

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
    Holds where an electron gains its ionisation energy in much less than a
    mean free path and loses nothing to attachment, roughly
    $100 \lesssim E/p \lesssim 10^{4}\ \mathrm{V\,m^{-1}Pa^{-1}}$ for hydrogen.
    The coefficients are the caller's: this function fixes the functional form,
    not the gas.  Supplying values fitted outside the $E/p$ range they were
    fitted in is the usual way to get a confident wrong answer.

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
    pressure = _require_positive("p_Pa", p_Pa)
    length = _require_positive("connection_length_m", connection_length_m)
    p_torr = pressure / PA_PER_TORR
    argument = _LLOYD_A_PER_M_TORR * p_torr * length
    degenerate = argument <= 1.0
    if np.any(degenerate):
        warnings.warn(
            "the Townsend avalanche cannot close over this connection length "
            "(A p L <= 1), so no breakdown threshold exists; returning nan",
            RuntimeWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        field = _LLOYD_B_V_PER_M_TORR * p_torr / np.log(argument)
    field = np.where(degenerate, np.nan, field)
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


__all__ = public_names(globals())
