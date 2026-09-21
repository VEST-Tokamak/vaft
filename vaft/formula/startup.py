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
f           : microwave source frequency                             [Hz]
B_ECR       : electron-cyclotron resonant field magnitude            [T]
R_ECR       : major radius of the resonance in a vacuum 1/R field    [m]

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
.. [4] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-2.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._exports import public_names
from .constants import K_BOLTZMANN, ME, MU0, PA_PER_TORR, QE

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


def _require_positive_or_blank(name, value):
    """Like :func:`_require_positive`, but a ``nan`` passes through and propagates.

    Only for a quantity that arrives as a *masked map*.  A connection-length map
    is ``nan`` outside the wall, and a caller feeding that map to a threshold is
    doing the ordinary thing rather than making a mistake -- so the blank carries
    through to the threshold instead of raising.  An infinity or a non-positive
    value still is a mistake and still raises.  Every other input in this module
    keeps the strict guard: it is scalar or physically total, and a ``nan`` there
    is far more likely to be an uninitialised array than a deliberate blank.
    """
    array = np.asarray(value, dtype=float)
    if np.any(np.isinf(array)) or np.any(array <= 0.0):
        raise ValueError(
            f"{name} must be positive, or nan where it has no value; got {value!r}"
        )
    return array


def _breakdown_field(p, connection_length_m, A, B):
    """Townsend closure inverted for the threshold field, without the wrapping.

    Both public spellings call this, so the degenerate-case warning is always
    raised two frames below the caller and blames the call site rather than a
    line inside this module.
    """
    pressure = _require_positive("p", p)
    length = _require_positive_or_blank("connection_length_m", connection_length_m)
    coeff_a = _require_positive("A", A)
    coeff_b = _require_positive("B", B)
    argument = coeff_a * pressure * length
    degenerate = argument <= 1.0
    if np.any(degenerate):
        warnings.warn(
            "the Townsend avalanche cannot close over this connection length "
            "(A p L <= 1), so no breakdown threshold exists; returning nan",
            RuntimeWarning,
            stacklevel=3,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        field = coeff_b * pressure / np.log(argument)
    return np.where(degenerate, np.nan, field)


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
        Non-finite (``nan`` included), zero or negative pressure or
        temperature. Mask a gauge trace's gaps before calling.

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
        Non-finite (``nan`` included), zero or negative density or atom
        count.

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
        Non-finite, zero or negative pressure or coefficient; an infinite, zero
        or negative connection length.  A ``nan`` connection length passes
        through as a missing value, which is how a connection-length map carries
        the points outside the wall.

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
    physical field.  Both this function and :func:`lloyd_breakdown_field` are
    thin wrappers over one private kernel, so the warning is raised exactly two
    frames below the caller either way and blames the call site rather than a
    line in this module.

    References
    ----------
    .. [1] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 4.2.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.

    See Also
    --------
    lloyd_breakdown_field
    townsend_ionization_coefficient
    """
    field = _breakdown_field(p, connection_length_m, A, B)
    return _maybe_scalar(field, p, connection_length_m, A, B)


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
        Non-finite, zero or negative pressure; an infinite, zero or negative
        connection length.  A ``nan`` connection length passes through as a
        missing value, which is how a connection-length map carries the points
        outside the wall.

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
    field = _breakdown_field(
        p_torr,
        connection_length_m,
        _LLOYD_A_PER_M_TORR,
        _LLOYD_B_V_PER_M_TORR,
    )
    return _maybe_scalar(field, p_Pa, connection_length_m)


#: Townsend coefficients by gas, per metre and per torr: the published
#: per-centimetre pair times 100, kept in torr for the reason the module
#: docstring gives.  Hydrogen is
#: Lloyd's pair so the gas-keyed threshold for ``"H2"`` *is* Lloyd's threshold,
#: not a second opinion about it.  Deuterium is deliberately absent: its
#: measured coefficients differ from hydrogen's (Rose 1956), and no single
#: pair was found that a source states for tokamak-relevant $E/p$.
_TOWNSEND_GASES = {
    "H2": (_LLOYD_A_PER_M_TORR, _LLOYD_B_V_PER_M_TORR),
    "He": (300.0, 3400.0),
    "Ar": (1200.0, 18000.0),
}


def _townsend_gas(gas):
    """Look a gas up by its exact key, or say which keys exist."""
    try:
        return _TOWNSEND_GASES[gas]
    except (KeyError, TypeError):
        raise ValueError(
            f"no Townsend coefficients for gas {gas!r}; known gases are "
            f"{sorted(_TOWNSEND_GASES)}.  Pass A and B to "
            "townsend_breakdown_field for any other gas."
        ) from None


def townsend_coefficients_for_gas(gas):
    r"""Townsend similarity coefficients of a catalogued fill gas, in SI.

    $$A_{SI} = \frac{100\,A_{\mathrm{cm\,Torr}}}{P_T},\qquad
      B_{SI} = \frac{100\,B_{\mathrm{cm\,Torr}}}{P_T}$$

    with $P_T$ = :data:`vaft.formula.constants.PA_PER_TORR` pascal per torr,
    ready to pass to :func:`townsend_ionization_coefficient`.

    ========  ==========================  ==============  ========================
    ``gas``   $A$, $B$ [cm^-1 Torr^-1,   fitted $E/p$    source
              V cm^-1 Torr^-1]            [V/(cm Torr)]
    ========  ==========================  ==============  ========================
    ``"H2"``  5.1, 125                    tokamak fits    Lloyd et al. 1991
    ``"He"``  3, 34                       20-150          Raizer, Table 4.1
    ``"Ar"``  12, 180                     100-600         Raizer, Table 4.1
    ========  ==========================  ==============  ========================

    Parameters
    ----------
    gas : str
        Exact catalogue key: ``"H2"``, ``"He"`` or ``"Ar"`` [-].

    Returns
    -------
    A : float
        Townsend similarity coefficient [m^-1 Pa^-1].
    B : float
        Townsend similarity coefficient [V m^-1 Pa^-1].

    Raises
    ------
    ValueError
        A gas that is not in the catalogue, including ``"D2"``.

    Convention
    ----------
    Keys are chemical formulas matched exactly, so ``"h2"`` and
    ``"hydrogen"`` are refused rather than guessed at.  The table stores the
    published per-torr pair and converts on the way out, for the reason the
    module docstring gives: a rounded SI pair costs half a percent where a
    spherical tokamak's prefill sits.

    Validity
    --------
    Empirical fit.  Each pair is a single-exponential fit over a limited
    $E/p$ range, and the literature for one gas scatters by up to a factor of
    two in $B$ depending on that range -- Massarczyk et al. collect 34 to 60
    V/(cm Torr) for helium and 133 to 320 for argon.  Refitting argon's own
    data over a different $pd$ range gives $A = 3.6$, $B = 52$ (Norman et
    al.).  A pair is a representative value, not a measurement of VEST's gas.

    **A tokamak threshold mostly sits below those ranges.**  Over 100 m of
    connection length the threshold's own $E/p$ is 98 V/(cm Torr) for argon
    at 7 mPa and 55 at 30 mPa, under Raizer's 100-600; helium stays inside
    its 20-150 only from about 7 to 25 mPa.  Raizer's hydrogen pair, 5 and
    130 over 150-600, is not the one catalogued: Lloyd's 5.1 and 125 were
    fitted to tokamak breakdown itself, which is the regime this is for.
    Evaluate $E_{BD}/p$ and compare before trusting a noble-gas threshold.

    Limitations
    -----------
    Deuterium is not catalogued: its measured coefficients are not
    hydrogen's (Rose 1956), and Lloyd applies his hydrogen pair to both
    isotopes by approximation.  Use :func:`lloyd_breakdown_field` for that
    approximation deliberately, or pass a measured pair to
    :func:`townsend_breakdown_field`.

    References
    ----------
    .. [1] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 4.1.5,
           Table 4.1.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.
    .. [3] A. M. Howatson, *An Introduction to Gas Discharges*, Pergamon, as cited by [5].
    .. [4] R. Massarczyk et al., arXiv:1612.07170 (2016), Table I.
    .. [5] R. Norman et al., arXiv:2107.07521 (2021), Table 1.
    .. [6] D. J. Rose, Phys. Rev. 104 (1956) 273.

    See Also
    --------
    townsend_ionization_coefficient
    townsend_breakdown_field_for_gas
    """
    A_per_m_torr, B_per_m_torr = _townsend_gas(gas)
    return A_per_m_torr / PA_PER_TORR, B_per_m_torr / PA_PER_TORR


def townsend_breakdown_field_for_gas(p_Pa, connection_length_m, gas):
    r"""Breakdown threshold field of a catalogued fill gas, pressure in pascal.

    $$E_{BD} = \frac{B\,p}{\ln\!\left(A\,p\,L\right)}$$

    :func:`townsend_breakdown_field` with the $A$, $B$ pair of
    :func:`townsend_coefficients_for_gas`.  For ``gas="H2"`` this is
    :func:`lloyd_breakdown_field`, bit for bit.

    Parameters
    ----------
    p_Pa : float or np.ndarray
        Neutral fill pressure, finite and positive [Pa].
    connection_length_m : float or np.ndarray
        Connection length of an open field line, finite and positive [m].
    gas : str
        Exact catalogue key: ``"H2"``, ``"He"`` or ``"Ar"`` [-].

    Returns
    -------
    float or np.ndarray
        Threshold field, ``nan`` where $A\,p\,L \le 1$ [V/m].

    Raises
    ------
    ValueError
        A gas that is not in the catalogue; a non-finite, zero or negative
        pressure; an infinite, zero or negative connection length.  A ``nan``
        connection length passes through as a missing value.

    Convention
    ----------
    Pressure in **pascal**, converted to torr and evaluated against the
    per-torr pair, exactly as :func:`lloyd_breakdown_field` does, so the
    hydrogen entry reproduces it to the last bit rather than to rounding.

    Physical interpretation
    -----------------------
    No gas is easiest everywhere: the ordering depends on where the fill
    sits on its Paschen curve.  Over 100 m of connection length argon's
    larger $A$ puts its threshold below hydrogen's at 3 mPa, and its larger
    $B$ puts it above hydrogen's by 30 mPa; helium's small $A$ gives it no
    threshold at all at 3 mPa and the lowest one of the three from 7 mPa
    up.

    Validity
    --------
    Empirical fit.  Inherits the catalogue's scatter and fitted $E/p$
    ranges: see :func:`townsend_coefficients_for_gas`.  The returned field
    itself gives the operating $E/p$, and at a tokamak fill it is usually
    below the range the noble-gas pairs were fitted over.  Pure gases only; a mixture is not a weighted average of
    these pairs.

    Limitations
    -----------
    Deuterium is not catalogued.  Says nothing about pre-ionisation or
    burn-through, as for :func:`townsend_breakdown_field`.

    Numerical notes
    ---------------
    $A\,p\,L \le 1$ warns and returns ``nan`` elementwise, from the same
    private kernel as the other two spellings, so the warning blames the
    caller's line.

    References
    ----------
    .. [1] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 4.1.5,
           Table 4.1.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.

    See Also
    --------
    townsend_coefficients_for_gas
    townsend_breakdown_field
    lloyd_breakdown_field
    """
    A_per_m_torr, B_per_m_torr = _townsend_gas(gas)
    p_torr = _require_positive("p_Pa", p_Pa) / PA_PER_TORR
    field = _breakdown_field(p_torr, connection_length_m, A_per_m_torr, B_per_m_torr)
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


#: Empirical threshold on the Lloyd figure of merit $E_\phi B_\phi / B_p$ above
#: which a purely Ohmic breakdown is expected [V/m].  VEST's VFIT
#: ``Plot_Vacuum_2D.m`` draws it as "Lloyd Condition BtEt/Bp > 1000V/m
#: (Ohmic)", following Lloyd et al., Nucl. Fusion 31 (1991) 2031.
LLOYD_FIGURE_OF_MERIT_OHMIC_V_PER_M = 1000.0

#: The same threshold with ECH pre-ionisation, an order of magnitude lower
#: [V/m]; the "> 100V/m (ECH)" line of the same VFIT figure.
LLOYD_FIGURE_OF_MERIT_ECH_V_PER_M = 100.0


def lloyd_figure_of_merit(E_phi, B_phi, B_p):
    r"""Lloyd's breakdown figure of merit: toroidal drive times field-line pitch.

    $$F_{BD} = \frac{|E_\phi|\,|B_\phi|}{|B_p|}$$

    The toroidal electric field projected along a field line whose length
    grows as $B_\phi / B_p$; the quantity the empirical Ohmic and
    ECH-assisted breakdown thresholds
    (:data:`LLOYD_FIGURE_OF_MERIT_OHMIC_V_PER_M`,
    :data:`LLOYD_FIGURE_OF_MERIT_ECH_V_PER_M`) are stated against.

    Parameters
    ----------
    E_phi : float or np.ndarray
        Toroidal electric field; used as a magnitude [V/m].
    B_phi : float or np.ndarray
        Toroidal field; used as a magnitude [T].
    B_p : float or np.ndarray
        Poloidal field magnitude; used as a magnitude [T].

    Returns
    -------
    float or np.ndarray
        Figure of merit; ``nan`` where ``B_p`` is zero or any input is
        non-finite [V/m].

    Convention
    ----------
    All three inputs are used as **magnitudes**: the sign of $E_\phi$ depends
    on the flux kernel (tracked in #354) and the signs of $B_\phi$ and $B_p$ on
    COCOS and coil polarity, none of which changes whether an avalanche can
    form.  A null ($B_p = 0$) has no finite figure and is returned as ``nan``
    rather than ``inf``, so a map of it keeps a finite colour range.

    Physical interpretation
    -----------------------
    $E_\phi B_\phi / B_p$ is the parallel field an electron would see if the
    connection length were set by the pitch alone.  A strong toroidal field
    and a weak poloidal one lengthen the path over which the drive acts, which
    is why a startup looks for a null.

    Limitations
    -----------
    A proxy, not a threshold: it ignores the fill pressure and the real
    connection length to the wall, both of which
    :func:`lloyd_breakdown_field` and :func:`breakdown_margin` take into
    account.  The 1000 V/m and 100 V/m thresholds are machine experience, not
    derived.

    References
    ----------
    .. [1] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2.

    See Also
    --------
    breakdown_margin
    """
    e_phi = np.abs(np.asarray(E_phi, dtype=float))
    b_phi = np.abs(np.asarray(B_phi, dtype=float))
    b_p = np.abs(np.asarray(B_p, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        figure = e_phi * b_phi / b_p
    figure = np.where(np.isfinite(figure), figure, np.nan)
    return _maybe_scalar(figure, E_phi, B_phi, B_p)



# ------------------------------------------------------------------
# Pre-ionisation: where a microwave source resonates with the electrons
# ------------------------------------------------------------------

def _require_harmonic(harmonic):
    """Reject anything but a positive integer harmonic number."""
    if (
        isinstance(harmonic, (bool, np.bool_))
        or not isinstance(harmonic, (int, np.integer))
        or harmonic < 1
    ):
        raise ValueError(f"harmonic must be a positive integer; got {harmonic!r}")
    return int(harmonic)


def electron_cyclotron_resonance_field(frequency_Hz, harmonic=1):
    r"""Field magnitude at which electrons gyrate in step with a microwave source.

    $$B_{ECR} = \frac{2\pi\,m_e\,f}{h\,e}$$

    the cold, non-relativistic electron-cyclotron resonance at harmonic $h$.
    A 2.45 GHz magnetron resonates at 87.5 mT on the fundamental.

    Parameters
    ----------
    frequency_Hz : float or np.ndarray
        Source frequency, finite and positive [Hz].
    harmonic : int, optional
        Positive cyclotron harmonic number; the fundamental is ``1`` [-].

    Returns
    -------
    float or np.ndarray
        Resonant field magnitude [T].

    Raises
    ------
    ValueError
        A non-finite, zero or negative frequency; a harmonic that is not a
        positive integer.

    Physical interpretation
    -----------------------
    Where $|B| = B_{ECR}$ an electron sees the wave's electric field rotate
    with it and gains energy every gyration; everywhere else the phase slips
    and the gain averages away.  That surface is where microwave
    pre-ionisation seeds the free electrons an avalanche starts from.

    Limitations
    -----------
    The cold resonance only.  The relativistic mass increase and the Doppler
    shift of an oblique launch move the absorption away from this field, and
    whether the wave reaches the layer at all -- the cutoff density -- is a
    separate question this expression does not ask.  Says nothing about how
    much power is absorbed.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-2.

    See Also
    --------
    electron_cyclotron_resonance_radius
    """
    frequency = _require_positive("frequency_Hz", frequency_Hz)
    h = _require_harmonic(harmonic)
    field = 2.0 * np.pi * ME * frequency / (h * QE)
    return _maybe_scalar(field, frequency_Hz)


def electron_cyclotron_resonance_radius(B_T_R_Tm, frequency_Hz, harmonic=1):
    r"""Major radius of the electron-cyclotron resonance in a vacuum toroidal field.

    $$R_{ECR} = \frac{|B_T R|}{B_{ECR}(f, h)}$$

    because a vacuum toroidal field falls as $1/R$, so the product $B_T R$ is
    a constant of the coil current and the resonance is a vertical line at one
    major radius.

    Parameters
    ----------
    B_T_R_Tm : float or np.ndarray
        Vacuum toroidal field times major radius, as stored in
        ``tf.b_field_tor_vacuum_r``; used as a magnitude [T m].
    frequency_Hz : float or np.ndarray
        Source frequency, finite and positive [Hz].
    harmonic : int, optional
        Positive cyclotron harmonic number; the fundamental is ``1`` [-].

    Returns
    -------
    float or np.ndarray
        Resonance major radius; ``0`` while the toroidal field is off [m].

    Raises
    ------
    ValueError
        A non-finite $B_T R$; a non-finite, zero or negative frequency; a
        harmonic that is not a positive integer.

    Convention
    ----------
    $B_T R$ is taken as a **magnitude**: the resonance depends on $|B|$, and
    the sign of ``b_field_tor_vacuum_r`` is a COCOS and coil-polarity choice.
    The input is the product, not $B_T$ at some radius, so no reference
    radius enters -- VEST's $R_0 = 0.4$ m is where a plot quotes $B_T$, not
    part of this relation.

    Physical interpretation
    -----------------------
    As the TF current ramps, $B_T R$ grows and the resonance sweeps outward
    from the centre stack.  Pre-ionisation happens where this line crosses
    the vessel; compare it with the field null to see whether the seed
    electrons are born where the avalanche can use them.

    Limitations
    -----------
    Vacuum toroidal field only.  The poloidal field adds to $|B|$ and bends
    the resonance near a coil; at a startup null it is a few gauss against
    tens of millitesla and the vertical line is accurate, far from one it is
    not.  Plasma diamagnetism is ignored, which is exact before breakdown.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas*, AIP (1992), Sec. 1-2.

    See Also
    --------
    electron_cyclotron_resonance_field
    """
    product = np.asarray(B_T_R_Tm, dtype=float)
    if not np.all(np.isfinite(product)):
        raise ValueError(f"B_T_R_Tm must be finite; got {B_T_R_Tm!r}")
    radius = np.abs(product) / electron_cyclotron_resonance_field(
        np.asarray(frequency_Hz, dtype=float), harmonic
    )
    return _maybe_scalar(radius, B_T_R_Tm, frequency_Hz)


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
        Required vertical field, carrying the sign of the Shafranov bracket
        [T].

    Raises
    ------
    ValueError
        Non-finite or non-positive current, major radius, minor radius or
        elongation, or a minor radius that is not smaller than the major one.

    Convention
    ----------
    **The orientation is settled, and the bracket's sign is kept.**  Mitarai
    writes Eq. (1.3) signed, $B_{VE} = -(\mu_0 I_p/4\pi R)[\cdots]$: the
    equilibrium field opposes $I_p$, because it has to push the ring back
    against its own outward hoop force.  What is returned here is the bracket
    without that leading minus, so for the usual positive bracket it is the
    size of a field pointing *anti-parallel* to the one a positive $I_p$ makes
    on the axis.  A caller turning it into a coil current takes the sense from
    that statement plus the machine description, not from a COCOS -- there is
    no flux map here for a COCOS to describe.

    **The bracket can go negative, and then so does the result.**  At
    $\kappa = 4$, $\beta_p = 0$, $l_i = 0$, $R_0 = 0.5$, $a = 0.35$ it is,
    because $l_\kappa$ has shrunk the logarithm below $3/2$.  That is a real
    reversal of the required field rather than a domain error, so no absolute
    value is taken; a caller comparing this against a stray-field magnitude
    must take ``abs`` itself.

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
    adds; that term needs the IMAS ``li_3`` normalisation, see
    :func:`vaft.formula.equilibrium.internal_inductance_from_li_3_R0`.

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
        Normalised internal inductance in the IMAS ``li_3`` definition [-].

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
    **``li`` must be $l_{i3}$**, the IMAS ``global_quantities.li_3``.  It
    enters as the dimensional $\mu_0 R\,l_i/2$, which is the internal
    inductance $L_i = 2W_{p,\mathrm{int}}/I_p^2$ only for
    $l_{i3} = 2L_i/(\mu_0 R)$, and only when $R$ is the radius the
    equilibrium normalised by.  $l_{i1}$ normalises by the edge poloidal field
    instead: $l_{i1}/l_{i3} = L_{pol}^2 R/(2V)$ whatever the current profile,
    which for a large-aspect-ratio ellipse is 1.085 at $\kappa = 1.6$ and 1.19
    at $\kappa = 2$, so passing $l_{i1}$ overstates $L_i$ by that much.  A caller
    holding a dimensional $L_i$ should convert with
    :func:`vaft.formula.equilibrium.li_3_from_internal_inductance_R0` rather
    than pass it here.

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
        Normalised internal inductance in the IMAS ``li_3`` definition, held
        fixed by the derivative [-].
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
    **``R_m`` does not enter the value.**  Expressed in $\epsilon$ the
    derivative is $\mu_0$ times a dimensionless function, so it is the same at
    every major radius; the argument is kept for symmetry with the two sibling
    functions and to decide whether a scalar or an array comes back.  Pass the
    radius you mean anyway -- a future shaping term would use it.

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
# The lumped plasma circuit: resistivity, resistance, inductance, ramp
# ------------------------------------------------------------------

def resistivity_from_n_e_nu_e(n_e_m3, nu_e_s):
    r"""Resistivity from the electron momentum-transfer collision frequency.

    $$\eta = \frac{m_e\,\nu_e}{n_e e^2}$$

    Parameters
    ----------
    n_e_m3 : float or np.ndarray
        Electron density, finite and positive [m^-3].
    nu_e_s : float or np.ndarray
        Total electron momentum-transfer collision frequency, finite and
        non-negative [s^-1].

    Returns
    -------
    float or np.ndarray
        Resistivity [Ohm m].

    Raises
    ------
    ValueError
        Non-finite or non-positive density, or a non-finite or negative
        collision frequency.

    Convention
    ----------
    **Pass the sum of every electron drag, and this is the partially ionised
    resistivity.**  During start-up $\nu_e = \nu_{ei} + \nu_{en}$: electron-
    neutral drag is not a correction there, it can dominate until burn-through.
    Both rates come from the caller, because $\nu_{en}$ needs a cross-section
    this module does not own.

    **This is the Lorentz resistivity, not the parallel Spitzer one.**  Fed the
    NRL electron-ion rate $\nu_{ei} = 2.91\times10^{-12}\,n_e\ln\Lambda\,
    T_e^{-3/2}$ (SI, $T_e$ in eV) it returns $1.03\times10^{-4}\ln\Lambda\,
    T_e^{-3/2}\ \Omega$ m -- NRL's $\eta_\perp$.  Current along the field, which
    is the toroidal plasma current, sees the Spitzer-Harm $\eta_\parallel$,
    smaller by the factor $0.51$ at $Z = 1$ that electron-electron collisions
    buy; that is
    :func:`vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda`.
    Using this value for the toroidal circuit doubles the plasma resistance.

    Limitations
    -----------
    A single scalar: no profile, no trapped-particle (neoclassical) correction,
    and no velocity dependence of the cross-sections, which the effective
    frequency is assumed to have absorbed.

    References
    ----------
    .. [1] NRL Plasma Formulary (2019), p. 28 (collision rates) and p. 29
           (transverse and parallel Spitzer resistivity).
    .. [2] Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991), Sec. 2.3.

    See Also
    --------
    vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
    plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa
    """
    density = _require_positive("n_e_m3", n_e_m3)
    rate = np.asarray(nu_e_s, dtype=float)
    if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError(
            f"nu_e_s must be finite and non-negative; got {nu_e_s!r}"
        )
    return _maybe_scalar(ME * rate / (density * QE**2), n_e_m3, nu_e_s)


def plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa(eta_ohm_m, R0_m, a_m, kappa=1.0):
    r"""Loop resistance of a uniform-current elliptical plasma ring.

    $$R_p = \eta\,\frac{2\pi R_0}{\pi a^2\kappa} = \frac{2\,\eta R_0}{a^2\kappa}$$

    Parameters
    ----------
    eta_ohm_m : float or np.ndarray
        Resistivity along the current, finite and positive [Ohm m].
    R0_m : float or np.ndarray
        Major radius, finite and positive [m].
    a_m : float or np.ndarray
        Minor radius, finite and positive [m].
    kappa : float or np.ndarray, optional
        Elongation; default 1, a circular cross-section [-].

    Returns
    -------
    float or np.ndarray
        Loop resistance [Ohm].

    Raises
    ------
    ValueError
        Non-finite or non-positive resistivity, radius or elongation.

    Convention
    ----------
    The path length is the magnetic-axis circumference $2\pi R_0$ and the
    cross-section the ellipse $\pi a^2\kappa$: a wire, not a torus.  The
    resistivity must be the one the toroidal current sees, which is
    $\eta_\parallel$ -- see :func:`resistivity_from_n_e_nu_e` for why the
    Lorentz value would double this.

    Assumptions
    -----------
    Uniform current density and uniform resistivity.  A peaked current profile
    in a hotter core carries more of the current through less resistance, so
    this overestimates $R_p$ once a temperature profile has formed.

    Limitations
    -----------
    Order-$a/R_0$ toroidal corrections to the path length are absent.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 2.16.

    See Also
    --------
    resistivity_from_n_e_nu_e
    plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p
    vaft.formula.equilibrium.ohmic_heating_power_from_I_p_V_res
    """
    resistivity = _require_positive("eta_ohm_m", eta_ohm_m)
    major = _require_positive("R0_m", R0_m)
    minor = _require_positive("a_m", a_m)
    shape = _require_positive("kappa", kappa)
    resistance = 2.0 * resistivity * major / (minor**2 * shape)
    return _maybe_scalar(resistance, eta_ohm_m, R0_m, a_m, kappa)


def plasma_inductance_circular_from_R0_a_li(R0_m, a_m, li, kappa=1.0):
    r"""Self-inductance of a high-aspect-ratio plasma ring, Mitarai's form.

    $$L_p = \mu_0 R_0\left[\ln\!\left(\frac{8R_0}{a\,l_\kappa}\right)
      + \frac{l_i}{2} - 2\right],\qquad l_\kappa = \sqrt{\frac{1+\kappa^2}{2}}$$

    Parameters
    ----------
    R0_m : float or np.ndarray
        Major radius, finite and positive [m].
    a_m : float or np.ndarray
        Minor radius, finite and positive and smaller than ``R0_m`` [m].
    li : float or np.ndarray
        Normalised internal inductance in the IMAS ``li_3`` definition [-].
    kappa : float or np.ndarray, optional
        Elongation; default 1, a circular cross-section [-].

    Returns
    -------
    float or np.ndarray
        Total plasma self-inductance, ``nan`` where the expansion gives zero or
        less [H].

    Raises
    ------
    ValueError
        Non-finite or non-positive radius or elongation, or a minor radius that
        is not smaller than the major one.

    Convention
    ----------
    The same expression :func:`vertical_field_from_I_p_R0_a_beta_p_li` is built
    on: holding $a$ fixed, $\partial L_p/\partial R$ substituted into
    $B_{VE} = (\mu_0 I_p/4\pi R)[\mu_0^{-1}\partial L_p/\partial R + \beta_p - 1/2]$
    returns that function exactly.  $l_i$ enters as the dimensional
    $\mu_0 R_0 l_i/2$, which is the internal inductance only for ``li_3``
    normalised by $R_0$; see
    :func:`plasma_inductance_hirshman_from_R_eps_kappa_li` for what passing
    ``li_1`` costs.

    Validity
    --------
    $a/R_0 \ll 1$.  At a spherical tokamak's aspect ratio the expansion is out
    of its range and the Hirshman fit,
    :func:`plasma_inductance_hirshman_from_R_eps_kappa_li`, is the one to use.
    The error there is not one-signed: at $\epsilon = 0.75$ this form is 51 %
    high at $\kappa = 1$ and $l_i = 0.3$, 4.5 times low at $\kappa = 2$, and
    not positive from $\kappa \approx 2.15$, because $l_\kappa$ shrinks the
    logarithm below $2 - l_i/2$.
    The two do converge as $a/R_0 \to 0$, but slowly and not monotonically:
    Hirshman's correction terms go as $\sqrt{\epsilon}$, so at $\kappa = 1$
    they differ by 6.7 % at $\epsilon = 0.3$, cross near 0.05, still differ by
    1.6 % at 0.01 and by 0.07 % only at $10^{-6}$.  Agreement at one aspect
    ratio is therefore no evidence of agreement at another.

    Limitations
    -----------
    Shaping enters only through $l_\kappa$ inside the logarithm.

    Numerical notes
    ---------------
    A non-positive result is outside the expansion rather than a physical
    inductance -- a self-inductance cannot be negative -- so it warns and
    returns ``nan`` elementwise, as the Townsend kernels do below
    $A\,p\,L = 1$.  Passed on, a negative value would only fail later in
    :func:`plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p` with a
    message naming the wrong function.

    References
    ----------
    .. [1] O. Mitarai, R. Yoshino and K. Ushigusa, Nucl. Fusion 42 (2002) 1257,
           Eq. (1.2).
    .. [2] V. D. Shafranov, in *Reviews of Plasma Physics*, Vol. 2,
           Consultants Bureau (1966), p. 103.

    See Also
    --------
    plasma_inductance_hirshman_from_R_eps_kappa_li
    vertical_field_from_I_p_R0_a_beta_p_li
    plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p
    """
    major = _require_positive("R0_m", R0_m)
    minor = _require_positive("a_m", a_m)
    shape = _require_positive("kappa", kappa)
    if np.any(minor >= major):
        raise ValueError(
            f"a_m must be smaller than R0_m; got {a_m!r} and {R0_m!r}"
        )
    l_kappa = np.sqrt(0.5 * (1.0 + shape**2))
    inductance = MU0 * major * (
        np.log(8.0 * major / (minor * l_kappa)) + 0.5 * np.asarray(li, dtype=float) - 2.0
    )
    outside = inductance <= 0.0
    if np.any(outside):
        warnings.warn(
            "the high-aspect-ratio inductance is not positive here -- the "
            "expansion is outside its range at this aspect ratio and elongation; "
            "use plasma_inductance_hirshman_from_R_eps_kappa_li; returning nan",
            RuntimeWarning,
            stacklevel=2,
        )
        inductance = np.where(outside, np.nan, inductance)
    return _maybe_scalar(inductance, R0_m, a_m, li, kappa)


def plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p(
    V_loop_V, R_p_ohm, I_p_A, L_p_H, dL_p_dt_H_s=0.0
):
    r"""Rate of change of plasma current in a lumped single-loop circuit.

    $$\dot I_p = \frac{V_{\mathrm{loop}} - R_p I_p - I_p \dot L_p}{L_p}$$

    from $V_{\mathrm{loop}} = R_p I_p + \mathrm{d}(L_p I_p)/\mathrm{d}t$.

    Parameters
    ----------
    V_loop_V : float or np.ndarray
        Loop voltage driving the plasma, finite [V].
    R_p_ohm : float or np.ndarray
        Plasma loop resistance, finite and non-negative [Ohm].
    I_p_A : float or np.ndarray
        Plasma current, finite [A].
    L_p_H : float or np.ndarray
        Plasma self-inductance, finite and positive [H].
    dL_p_dt_H_s : float or np.ndarray, optional
        Rate of change of the inductance; default 0, fixed geometry [H/s].

    Returns
    -------
    float or np.ndarray
        $\mathrm{d}I_p/\mathrm{d}t$ [A/s].

    Raises
    ------
    ValueError
        A non-finite input, a negative resistance, or a non-positive
        inductance.

    Convention
    ----------
    **Signed, unlike the rest of this module.**  $V_{\mathrm{loop}}$ and $I_p$
    must be measured in the same sense around the torus, so a loop voltage that
    drives the existing current is positive; a sign mismatch turns a ramp-up
    into a ramp-down.  $I_p\dot L_p$ is kept because a plasma that grows or
    moves changes its inductance, and during start-up that term is not small.

    **With $L_p = L_e + L_i$, pass** ``dL_p_dt_H_s`` **$= \dot L_e +
    \tfrac12\dot L_i$, not $\dot L_e + \dot L_i$.**  The external inductance
    is a flux linkage and takes the full rate; the internal one is defined by
    the energy $\tfrac12 L_i I_p^2$ and takes half (Romero, eq. 23), so the
    naive sum overstates the profile term by $\tfrac12 I_p\dot L_i$.
    :func:`internal_inductive_voltage_terms_from_L_i_I_p` and
    :func:`boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j` keep the two
    apart.

    The power this balances is not the Ohmic power.
    $V_{\mathrm{loop}} I_p = R_p I_p^2 + \mathrm{d}(\tfrac12 L_p I_p^2)/
    \mathrm{d}t + \tfrac12 I_p^2 \dot L_p$: part of the transformer's work goes
    into magnetic energy, which is why
    :func:`vaft.formula.equilibrium.ohmic_heating_power_from_I_p_V_res` takes
    the resistive voltage and not the loop voltage.

    Limitations
    -----------
    One loop.  Mutual coupling to the vessel and the coils -- the eddy currents
    that dominate a VEST start-up -- is the caller's to add to
    $V_{\mathrm{loop}}$; this does not see them.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.7.
    .. [2] O. Mitarai, R. Yoshino and K. Ushigusa, Nucl. Fusion 42 (2002) 1257,
           Eq. (2.1).
    .. [3] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eqs. (23) and (31).

    See Also
    --------
    lr_time_from_L_R
    internal_inductive_voltage_terms_from_L_i_I_p
    plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa
    plasma_inductance_circular_from_R0_a_li
    """
    voltage = np.asarray(V_loop_V, dtype=float)
    resistance = np.asarray(R_p_ohm, dtype=float)
    current = np.asarray(I_p_A, dtype=float)
    inductance = _require_positive("L_p_H", L_p_H)
    rate = np.asarray(dL_p_dt_H_s, dtype=float)
    for name, value in (("V_loop_V", voltage), ("R_p_ohm", resistance),
                        ("I_p_A", current), ("dL_p_dt_H_s", rate)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    if np.any(resistance < 0.0):
        raise ValueError(f"R_p_ohm must be non-negative; got {R_p_ohm!r}")
    derivative = (voltage - resistance * current - current * rate) / inductance
    return _maybe_scalar(derivative, V_loop_V, R_p_ohm, I_p_A, L_p_H, dL_p_dt_H_s)


def lr_time_from_L_R(L_H, R_ohm):
    r"""Characteristic time of an inductive-resistive circuit.

    $$\tau_{L/R} = \frac{L}{R}$$

    Parameters
    ----------
    L_H : float or np.ndarray
        Inductance, finite and positive [H].
    R_ohm : float or np.ndarray
        Resistance, finite and positive [Ohm].

    Returns
    -------
    float or np.ndarray
        Time constant [s].

    Raises
    ------
    ValueError
        Non-finite or non-positive inductance or resistance.

    Convention
    ----------
    For the plasma loop this is the time a fixed loop voltage takes to bring
    $I_p$ within $1/e$ of its resistive limit $V/R_p$ at fixed inductance -- the
    linear rate of :func:`plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p`
    about that fixed point.  For a vessel element it is the eddy-current decay
    time.  It is not the current-profile diffusion time, which needs a length
    scale this does not have.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.7.

    See Also
    --------
    plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p
    """
    inductance = _require_positive("L_H", L_H)
    resistance = _require_positive("R_ohm", R_ohm)
    return _maybe_scalar(inductance / resistance, L_H, R_ohm)


def _require_finite(name, value):
    """Reject a non-finite signed input by name."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return array


def boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(
    L_e_H,
    I_p_A,
    dI_p_dt_A_s,
    dL_e_dt_H_s=0.0,
    M_pj_H=None,
    I_j_A=None,
    dI_j_dt_A_s=None,
    dM_pj_dt_H_s=None,
):
    r"""Boundary loop voltage split into the four ways the boundary flux can change.

    $$V_B = -\frac{\mathrm{d}\psi_B}{\mathrm{d}t},\qquad
      \psi_B = L_e I_p + \sum_j M_{pj} I_j$$

    $$V_B = \underbrace{-L_e \dot I_p}_{\text{ramp}}
            \underbrace{- I_p \dot L_e}_{\text{shape}}
            \underbrace{- \textstyle\sum_j M_{pj} \dot I_j}_{\text{drive}}
            \underbrace{- \textstyle\sum_j I_j \dot M_{pj}}_{\text{geometry}}$$

    Parameters
    ----------
    L_e_H : float or np.ndarray
        Plasma external inductance, finite and positive [H].
    I_p_A : float or np.ndarray
        Plasma current, finite [A].
    dI_p_dt_A_s : float or np.ndarray
        Rate of change of the plasma current, finite [A/s].
    dL_e_dt_H_s : float or np.ndarray, optional
        Rate of change of the external inductance; default 0, fixed boundary
        [H/s].
    M_pj_H : array_like, optional
        Plasma-coil mutual inductances, coils along the last axis, finite [H].
    I_j_A : array_like, optional
        Coil circuit currents, broadcastable against ``M_pj_H`` -- a fixed
        ``(n_coils,)`` mutual with ``(n_times, n_coils)`` currents is the
        usual call -- finite [A].
    dI_j_dt_A_s : array_like, optional
        Rates of change of the coil currents, broadcastable likewise, finite
        [A/s].
    dM_pj_dt_H_s : array_like, optional
        Rates of change of the mutual inductances, broadcastable likewise;
        default 0, fixed geometry [H/s].

    Returns
    -------
    V_ramp : float or np.ndarray
        $-L_e \dot I_p$, the plasma current changing at fixed boundary [V].
    V_shape : float or np.ndarray
        $-I_p \dot L_e$, the boundary moving or reshaping [V].
    V_drive : float or np.ndarray
        $-\sum_j M_{pj}\dot I_j$, the external circuits driving [V].
    V_geometry : float or np.ndarray
        $-\sum_j I_j \dot M_{pj}$, the plasma moving relative to the coils [V].

    Raises
    ------
    ValueError
        A non-finite input, a non-positive external inductance, coil arrays
        that do not broadcast, or only some of ``M_pj_H``, ``I_j_A`` and
        ``dI_j_dt_A_s`` given.

    Convention
    ----------
    **Romero's signs: $\psi$ is the full flux through the toroidal circle, in
    weber, not per radian, and $V_B = -\dot\psi_B$.**  $I_p$, the $I_j$ and
    the $M_{pj}$ are all referred to one toroidal direction, so a coil wound
    against the plasma current has a negative $M_{pj}$.  In this convention a
    positive $V_B$ sustains a positive $I_p$, the same sense as the
    ``V_loop_V`` of
    :func:`plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p`.  VAFT's
    two flux-to-voltage kernels disagree on both points (#354), so a $V_B$
    from a flux map must be brought to this convention before it is compared
    with the sum of these terms.

    $V_B$ is the sum of the four; they are returned apart because #782 asks
    for the ramp, the shape change and the drive to be inspectable rather than
    folded into one inductive voltage.  The drive and geometry terms are zero
    when no coils are given.  $M_{pj}$ multiplies the *circuit* current, so a
    multi-turn coil's turns belong in $M_{pj}$.

    Physical interpretation
    -----------------------
    Only the external inductance appears: $\psi_B$ is the flux *at* the
    boundary, and the internal inductance enters the balance on the other
    side, $V_B = R_p I_p + V_{\mathrm{ind}}$ with $V_{\mathrm{ind}}$ from
    :func:`internal_inductive_voltage_terms_from_L_i_I_p`.

    Limitations
    -----------
    Needs $L_e$, $M_{pj}$ and their rates from somewhere else: the Hirshman
    fit for $L_e$ ignores triangularity, and nothing here computes $M_{pj}$
    from a boundary.  Vessel eddy currents are coils in this sum, with their
    own $M_{pj}$ and $I_j$; leaving them out is the usual reason a VEST
    start-up balance does not close.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eqs. (12) and (28).

    See Also
    --------
    internal_inductive_voltage_terms_from_L_i_I_p
    plasma_external_inductance_hirshman_from_R_eps_kappa
    """
    external = _require_positive("L_e_H", L_e_H)
    current = _require_finite("I_p_A", I_p_A)
    ramp_rate = _require_finite("dI_p_dt_A_s", dI_p_dt_A_s)
    shape_rate = _require_finite("dL_e_dt_H_s", dL_e_dt_H_s)
    v_ramp = -external * ramp_rate
    v_shape = -current * shape_rate

    coils = (M_pj_H, I_j_A, dI_j_dt_A_s)
    given = [item is not None for item in coils]
    if any(given) and not all(given):
        raise ValueError(
            "M_pj_H, I_j_A and dI_j_dt_A_s must be given together, or none of them"
        )
    if all(given):
        mutual = _require_finite("M_pj_H", M_pj_H)
        coil_current = _require_finite("I_j_A", I_j_A)
        coil_rate = _require_finite("dI_j_dt_A_s", dI_j_dt_A_s)
        mutual_rate = (
            np.zeros_like(mutual)
            if dM_pj_dt_H_s is None
            else _require_finite("dM_pj_dt_H_s", dM_pj_dt_H_s)
        )
        try:
            mutual, coil_current, coil_rate, mutual_rate = np.broadcast_arrays(
                mutual, coil_current, coil_rate, mutual_rate
            )
        except ValueError:
            raise ValueError(
                "coil arrays do not broadcast against each other; got shapes "
                f"{[np.shape(a) for a in (M_pj_H, I_j_A, dI_j_dt_A_s, dM_pj_dt_H_s)]}"
            ) from None
        v_drive = -np.sum(mutual * coil_rate, axis=-1)
        v_geometry = -np.sum(coil_current * mutual_rate, axis=-1)
    elif dM_pj_dt_H_s is not None:
        raise ValueError("dM_pj_dt_H_s was given without the coils it belongs to")
    else:
        v_drive = v_geometry = np.zeros(())

    terms = np.broadcast_arrays(v_ramp, v_shape, v_drive, v_geometry)
    if terms[0].ndim == 0:
        return tuple(float(term) for term in terms)
    return tuple(np.array(term) for term in terms)


def internal_inductive_voltage_terms_from_L_i_I_p(L_i_H, I_p_A, dI_p_dt_A_s, dL_i_dt_H_s=0.0):
    r"""Inductive voltage of the internal inductance, split into ramp and profile terms.

    $$V_{\mathrm{ind}} = \frac{1}{I_p}\frac{\mathrm{d}}{\mathrm{d}t}
      \left(\tfrac12 L_i I_p^2\right)
      = \underbrace{L_i \dot I_p}_{\text{ramp}}
      + \underbrace{\tfrac12 I_p \dot L_i}_{\text{profile}}$$

    Parameters
    ----------
    L_i_H : float or np.ndarray
        Dimensional internal inductance, finite and non-negative [H].
    I_p_A : float or np.ndarray
        Plasma current, finite [A].
    dI_p_dt_A_s : float or np.ndarray
        Rate of change of the plasma current, finite [A/s].
    dL_i_dt_H_s : float or np.ndarray, optional
        Rate of change of the internal inductance; default 0, a frozen current
        profile [H/s].

    Returns
    -------
    V_ramp : float or np.ndarray
        $L_i \dot I_p$ [V].
    V_profile : float or np.ndarray
        $\tfrac12 I_p \dot L_i$, the current profile redistributing [V].

    Raises
    ------
    ValueError
        A non-finite input or a negative internal inductance.

    Convention
    ----------
    **The profile term carries one half, not one.**  $L_i I_p$ is not a flux
    linked by a single loop, so the internal term is fixed by the energy
    $W_{p,\mathrm{int}} = \tfrac12 L_i I_p^2$ and $I_p V_{\mathrm{ind}} =
    \dot W_{p,\mathrm{int}}$ exactly; the external inductance is a flux
    linkage, and there the full $I_p\dot L_e$ is right.  So a lumped circuit
    with $L_p = L_e + L_i$ closes Romero's balance only with
    $\dot L_p = \dot L_e + \tfrac12\dot L_i$.

    Signed in the sense of $I_p$, as for
    :func:`boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j`: Romero's
    balance is $V_B = R_p I_p + V_{\mathrm{ind}}$, with any non-inductive
    current drive inside the resistive term.

    Physical interpretation
    -----------------------
    A peaking current profile raises $L_i$ and costs flux even at constant
    $I_p$; a broadening one returns it.  That profile term is what separates
    a start-up with an evolving $l_i$ from one with a frozen profile.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eqs. (22)-(24).

    See Also
    --------
    boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j
    vaft.formula.equilibrium.internal_inductance_from_W_int_Ip
    """
    internal = np.asarray(L_i_H, dtype=float)
    if not np.all(np.isfinite(internal)) or np.any(internal < 0.0):
        raise ValueError(f"L_i_H must be finite and non-negative; got {L_i_H!r}")
    current = _require_finite("I_p_A", I_p_A)
    ramp_rate = _require_finite("dI_p_dt_A_s", dI_p_dt_A_s)
    profile_rate = _require_finite("dL_i_dt_H_s", dL_i_dt_H_s)
    inputs = (L_i_H, I_p_A, dI_p_dt_A_s, dL_i_dt_H_s)
    return (
        _maybe_scalar(internal * ramp_rate, *inputs),
        _maybe_scalar(0.5 * current * profile_rate, *inputs),
    )


# ------------------------------------------------------------------
# Burn-through: the radiation-ionisation barrier of a depleting fill
# ------------------------------------------------------------------

def neutral_density_after_ionization_from_n_0_n_e_V_p_V_V(n_0_m3, n_e_m3, V_p_m3, V_V_m3):
    r"""Neutral atom density left in the vessel after some has been ionised.

    $$n_D^0 = n_0 - \frac{V_p}{V_V}\,n_e$$

    Parameters
    ----------
    n_0_m3 : float or np.ndarray
        Initial neutral atom density filling the vessel, finite and positive
        [m^-3].
    n_e_m3 : float or np.ndarray
        Electron density in the plasma, finite and non-negative [m^-3].
    V_p_m3 : float or np.ndarray
        Plasma volume, finite and positive [m^3].
    V_V_m3 : float or np.ndarray
        Vessel volume the neutrals fill, finite and positive [m^3].

    Returns
    -------
    float or np.ndarray
        Remaining neutral atom density, ``nan`` where the plasma would hold
        more electrons than the fill had atoms [m^-3].

    Raises
    ------
    ValueError
        Non-finite or non-positive inventory or volume, or a non-finite or
        negative electron density.

    Convention
    ----------
    **Atoms, not molecules.**  $n_0$ is the atomic-equivalent inventory,
    :func:`atomic_inventory_from_molecular_gas` of the prefill, because each
    ion of a pure hydrogenic plasma consumes one atom.  Conservation is
    $n_D^0 V_V + n_e V_p = n_0 V_V$: the neutrals fill the whole vessel while
    the plasma occupies $V_p$ of it.

    Limitations
    -----------
    Pure hydrogenic plasma, no wall recycling or fuelling, no impurities.
    Recycling feeds neutrals back and is what makes a real burn-through slower
    than this closed-box inventory suggests.

    Numerical notes
    ---------------
    An electron density above $n_0 V_V/V_p$ needs more atoms than the fill
    had.  That is an inconsistent input rather than a state, so it warns and
    returns ``nan`` elementwise rather than a negative density.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, for the DYON burn-through model this closed-box balance
           reduces.

    See Also
    --------
    atomic_inventory_from_molecular_gas
    ionization_fraction_from_n_e_n_D0
    """
    inventory = _require_positive("n_0_m3", n_0_m3)
    electrons = np.asarray(n_e_m3, dtype=float)
    if not np.all(np.isfinite(electrons)) or np.any(electrons < 0.0):
        raise ValueError(f"n_e_m3 must be finite and non-negative; got {n_e_m3!r}")
    plasma = _require_positive("V_p_m3", V_p_m3)
    vessel = _require_positive("V_V_m3", V_V_m3)
    neutrals = inventory - (plasma / vessel) * electrons
    exhausted = neutrals < 0.0
    if np.any(exhausted):
        warnings.warn(
            "the electron density needs more atoms than the fill had "
            "(n_e > n_0 V_V / V_p); returning nan",
            RuntimeWarning,
            stacklevel=2,
        )
        neutrals = np.where(exhausted, np.nan, neutrals)
    return _maybe_scalar(neutrals, n_0_m3, n_e_m3, V_p_m3, V_V_m3)


def ionization_fraction_from_n_e_n_D0(n_e_m3, n_D0_m3):
    r"""Ionisation fraction of a partially ionised hydrogenic plasma.

    $$\gamma_{iz} = \frac{n_e}{n_e + n_D^0}$$

    Parameters
    ----------
    n_e_m3 : float or np.ndarray
        Electron density, finite and non-negative [m^-3].
    n_D0_m3 : float or np.ndarray
        Neutral atom density, finite and non-negative [m^-3].

    Returns
    -------
    float or np.ndarray
        Ionisation fraction, in $[0, 1]$ [-].

    Raises
    ------
    ValueError
        A non-finite or negative density, or both zero at once.

    Convention
    ----------
    Local, in the plasma: both densities are the ones inside $V_p$.  Mixing the
    plasma's electron density with a vessel-averaged neutral density gives a
    number that is neither.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, for the DYON burn-through model this reduces.

    See Also
    --------
    critical_ionization_fraction_from_V_p_V_V
    """
    electrons = np.asarray(n_e_m3, dtype=float)
    neutrals = np.asarray(n_D0_m3, dtype=float)
    for name, value, raw in (("n_e_m3", electrons, n_e_m3), ("n_D0_m3", neutrals, n_D0_m3)):
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError(f"{name} must be finite and non-negative; got {raw!r}")
    total = electrons + neutrals
    if np.any(total <= 0.0):
        raise ValueError("n_e_m3 and n_D0_m3 are both zero; there is nothing to ionise")
    return _maybe_scalar(electrons / total, n_e_m3, n_D0_m3)


def critical_ionization_fraction_from_V_p_V_V(V_p_m3, V_V_m3):
    r"""Ionisation fraction at the top of the radiation-ionisation barrier.

    $$\gamma_{iz,\mathrm{RIB}} = \frac{V_V}{V_V + V_p}$$

    Parameters
    ----------
    V_p_m3 : float or np.ndarray
        Plasma volume, finite and positive [m^3].
    V_V_m3 : float or np.ndarray
        Vessel volume, finite and positive [m^3].

    Returns
    -------
    float or np.ndarray
        Critical ionisation fraction [-].

    Raises
    ------
    ValueError
        Non-finite or non-positive volume.

    Convention
    ----------
    **Where the loss peaks, not where burn-through is complete.**  With the
    closed-box inventory of
    :func:`neutral_density_after_ionization_from_n_0_n_e_V_p_V_V`, the
    radiation-ionisation power $V_p P_{RI} n_e n_D^0$ is largest at
    $n_e = n_0 V_V/(2V_p)$, where the ionisation fraction is exactly this.
    Past it the loss falls as the fill runs out, which is why crossing it is
    the burn-through condition.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, for the DYON burn-through model this reduces; the
           closed-box maximisation above is derived here, not quoted.

    See Also
    --------
    radiation_ionization_barrier_from_P_RI_n_0_V_V
    ionization_fraction_from_n_e_n_D0
    """
    plasma = _require_positive("V_p_m3", V_p_m3)
    vessel = _require_positive("V_V_m3", V_V_m3)
    return _maybe_scalar(vessel / (vessel + plasma), V_p_m3, V_V_m3)


def radiation_ionization_power_from_P_RI_n_e_n_D0_V_p(P_RI_W_m3, n_e_m3, n_D0_m3, V_p_m3):
    r"""Radiation and ionisation power lost by a partially ionised plasma.

    $$P_{\mathrm{rad+iz}} = V_p\,P_{RI}(T_e)\,n_e\,n_D^0$$

    Parameters
    ----------
    P_RI_W_m3 : float or np.ndarray
        Combined radiation-plus-ionisation power coefficient per electron and
        per neutral atom, finite and non-negative [W m^3].
    n_e_m3 : float or np.ndarray
        Electron density, finite and non-negative [m^-3].
    n_D0_m3 : float or np.ndarray
        Neutral atom density, finite and non-negative [m^-3].
    V_p_m3 : float or np.ndarray
        Plasma volume, finite and positive [m^3].

    Returns
    -------
    float or np.ndarray
        Power lost [W].

    Raises
    ------
    ValueError
        A non-finite or negative coefficient or density, or a non-positive
        volume.

    Convention
    ----------
    **The coefficient is the caller's.**  $P_{RI}(T_e)$ carries the atomic
    physics -- excitation radiation plus the ionisation energy per event times
    the ionisation rate -- and this module does not invent a fit for it.  The
    radiation half is what
    :func:`vaft.formula.atomic.line_cooling_coefficient` returns for an
    identified atomic source; the ionisation half is the rate times the
    ionisation energy.

    Limitations
    -----------
    Pure hydrogen with no impurity line radiation, which in a real start-up is
    often what actually sets the barrier.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, for the DYON burn-through model this reduces; the
           closed-box maximisation above is derived here, not quoted.

    See Also
    --------
    radiation_ionization_barrier_from_P_RI_n_0_V_V
    vaft.formula.atomic.line_cooling_coefficient
    """
    coefficient = np.asarray(P_RI_W_m3, dtype=float)
    electrons = np.asarray(n_e_m3, dtype=float)
    neutrals = np.asarray(n_D0_m3, dtype=float)
    for name, value, raw in (("P_RI_W_m3", coefficient, P_RI_W_m3),
                             ("n_e_m3", electrons, n_e_m3), ("n_D0_m3", neutrals, n_D0_m3)):
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError(f"{name} must be finite and non-negative; got {raw!r}")
    plasma = _require_positive("V_p_m3", V_p_m3)
    power = plasma * coefficient * electrons * neutrals
    return _maybe_scalar(power, P_RI_W_m3, n_e_m3, n_D0_m3, V_p_m3)


def radiation_ionization_barrier_from_P_RI_n_0_V_V(P_RI_W_m3, n_0_m3, V_V_m3):
    r"""Height of the radiation-ionisation barrier a burn-through must cross.

    $$P_{\mathrm{RIB}} = \frac{V_V}{4}\,P_{RI}(T_e)\,n_0^2$$

    Parameters
    ----------
    P_RI_W_m3 : float or np.ndarray
        Radiation-plus-ionisation power coefficient, as for
        :func:`radiation_ionization_power_from_P_RI_n_e_n_D0_V_p`, finite and
        non-negative [W m^3].
    n_0_m3 : float or np.ndarray
        Initial neutral atom inventory, finite and positive [m^-3].
    V_V_m3 : float or np.ndarray
        Vessel volume, finite and positive [m^3].

    Returns
    -------
    float or np.ndarray
        Peak radiation-plus-ionisation power over the burn-through [W].

    Raises
    ------
    ValueError
        A non-finite or negative coefficient, or a non-positive inventory or
        volume.

    Convention
    ----------
    **The maximum of the loss, not a threshold fitted to data.**  Substituting
    the closed-box inventory into $V_p P_{RI} n_e n_D^0$ and maximising over
    $n_e$ gives this, at $n_e = n_0 V_V/(2V_p)$ where the ionisation fraction
    is :func:`critical_ionization_fraction_from_V_p_V_V`.  It does not depend
    on $V_p$ at all, and it goes as $p_0^2$ at fixed temperature, which is why
    a lower prefill eases burn-through.  Heating that exceeds it everywhere on
    the way -- $P_\Omega + P_{\mathrm{aux}} > P_{\mathrm{RIB}}$ -- is the
    reduced accessibility condition; it is the caller's comparison to make,
    because $P_{RI}$ varies with $T_e$ along the way.

    Limitations
    -----------
    Holds $P_{RI}$ fixed at one $T_e$ while $n_e$ varies, which is the reduced
    model's simplification; DYON evolves them together.

    References
    ----------
    .. [1] H.-T. Kim, W. Fundamenski and A. C. C. Sips, Nucl. Fusion 52 (2012)
           103016, for the DYON burn-through model this reduces; the
           closed-box maximisation above is derived here, not quoted.

    See Also
    --------
    radiation_ionization_power_from_P_RI_n_e_n_D0_V_p
    critical_ionization_fraction_from_V_p_V_V
    atomic_inventory_from_molecular_gas
    """
    coefficient = np.asarray(P_RI_W_m3, dtype=float)
    if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError(f"P_RI_W_m3 must be finite and non-negative; got {P_RI_W_m3!r}")
    inventory = _require_positive("n_0_m3", n_0_m3)
    vessel = _require_positive("V_V_m3", V_V_m3)
    barrier = 0.25 * vessel * coefficient * inventory**2
    return _maybe_scalar(barrier, P_RI_W_m3, n_0_m3, V_V_m3)


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


__all__ = public_names(
    globals(),
    constants=("LLOYD_FIGURE_OF_MERIT_OHMIC_V_PER_M", "LLOYD_FIGURE_OF_MERIT_ECH_V_PER_M"),
)
