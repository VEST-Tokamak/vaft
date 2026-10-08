"""Asymptotic ordering quantities of measured plasma states (#1627 §2, for the #1629 atlas).

The ordering contracts of :mod:`vaft.validation.orderings` read a state as a
row keyed by the names in
:data:`~vaft.validation.orderings.ORDERING_QUANTITIES`. This module builds
those rows from what a reconstructed VEST state actually carries -- global
equilibrium scalars, Thomson profiles, the plasma-current history -- with the
:mod:`vaft.formula` kernels the quantity registry names, and with nothing
else.

Three builders, one per scope, because a global ``d_i/a`` and a profile
``rho_s/L_Te`` are different quantities on different scales (#1627):

* :func:`global_ordering_quantities` -- one number per state, on ``a`` or
  ``R0``;
* :func:`profile_ordering_quantities` -- one number per flux surface, on that
  surface's own gradient length or connection length;
* :func:`time_history_ordering_quantities` -- one number per state from its
  time history (evolution rate, age since onset).

:func:`ordering_table` applies the global and time-history builders row by row
to a table of states; :func:`ordering_coverage` reports which quantities a
table could assess, by the data-availability tiers of #1627.

Notation
--------
a, R0      : minor and major radius                                 [m]
B0         : toroidal field on axis                                 [T]
n_e        : electron density                                       [m^-3]
T_e, T_i   : electron and ion temperatures                          [eV]
Z_eff      : effective charge; the main ion is singly charged       [-]
lnLambda   : Coulomb logarithm                                      [-]
eta        : Spitzer resistivity (parallel)                         [Ohm m]
v_A        : Alfven speed in the ion mass density                   [m/s]
L_T        : temperature gradient length |T / (dT/dr)|              [m]

Conventions
-----------
**Nothing is imputed.** A quantity whose inputs are missing, non-finite or
non-positive is NaN, so the contract evaluation reports it ``UNASSESSED``; an
ion temperature is never set equal to ``T_e`` here, and a density is never
filled in from another diagnostic. A caller who adopts such an assumption does
it explicitly and records it.

**One ion species.** The ion mass density is ``n_i m_i`` with ``n_i = n_e``
(quasineutral, singly charged main ion); ``Z_eff`` enters only the
resistivity and the electron collisions. Thermal speeds are
``sqrt(T/m)`` (NRL formulary), as :mod:`vaft.formula.ordering` uses.

**Gradient lengths** are ``|T/(dT/dr)|`` on the minor-radius coordinate ``r``
in metres, from :func:`vaft.formula.utils.normalized_gradient_scale_length`;
a flat or non-monotonic stretch gives a long ``L`` and a small ordering, never
a sign.

Provenance
----------
.. [1] Issue #1627 §2 (summary layer) and its quantity registry
   :data:`vaft.validation.orderings.ORDERING_QUANTITIES`; the ordering kernels
   follow J. D. Huba, *NRL Plasma Formulary* (2019) and S. I. Braginskii,
   *Reviews of Plasma Physics* Vol. 1 (1965), p. 205.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

__all__ = [
    "COVERAGE_TIERS",
    "DEFAULT_COLUMNS",
    "contract_population",
    "ordering_margins",
    "global_ordering_quantities",
    "ordering_coverage",
    "ordering_table",
    "profile_ordering_quantities",
    "time_history_ordering_quantities",
]

#: Which ordering groups each #1627 data-availability tier can fill, and from what.
#: Flow quantities sit in the ``profile`` group but need a rotation measurement,
#: which #1627 counts as tier 3.
COVERAGE_TIERS: Mapping[str, tuple[str, ...]] = {
    "tier1_equilibrium_and_profiles": ("foundational", "global", "profile"),
    "tier2_time_history": ("time_history",),
    "tier3_flow": ("sonic_mach_number", "alfven_mach_number"),
    "tier4_mode_and_layer": ("perturbation", "layer"),
}

#: Default column names of :func:`ordering_table`'s input.
DEFAULT_COLUMNS: Mapping[str, str] = {
    "minor_radius": "a_m",
    "major_radius": "r_geo_m",
    "b0": "b_t_T",
    "n_e": "n_e_m3",
    "t_e": "t_e_eV",
    "z_eff": "z_eff",
    "ln_lambda": "ln_lambda",
    "beta": "beta_t",
    "plasma_current": "i_p_A",
    "current_rate": "dip_dt_A_s",
    "plasma_age": "plasma_age_s",
}


def _finite_positive(*values: Any) -> bool:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(number) and number > 0.0):
            return False
    return True


def _alfven_speed(b0, n_e, ion_mass_amu):
    from vaft.formula.constants import MI_P
    from vaft.formula.stability import v_alfven_from_B_n_mi

    return float(v_alfven_from_B_n_mi(float(b0), float(n_e), float(ion_mass_amu) * MI_P))


def _resistivity(t_e, z_eff, ln_lambda):
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda

    return float(spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(float(t_e), float(z_eff), float(ln_lambda)))


def global_ordering_quantities(
    *,
    minor_radius,
    major_radius,
    b0,
    n_e=None,
    t_e=None,
    z_eff=None,
    ln_lambda=None,
    ion_mass_amu=1.0,
    beta=None,
):
    r"""The global ordering quantities of one plasma state, NaN where an input is missing.

    $$S = \frac{\mu_0 a v_A}{\eta},\quad \frac{d_i}{a},\quad \epsilon = \frac{a}{R_0},\quad
      \beta,\quad \frac{\beta}{\epsilon}$$

    Parameters
    ----------
    minor_radius : float
        Minor radius ``a`` [m].
    major_radius : float
        Major radius ``R0`` the inverse aspect ratio is taken against [m].
    b0 : float
        Toroidal field on axis [T].
    n_e : float, optional
        Electron density; a volume or line average, whichever the state has,
        recorded by the caller [m^-3].
    t_e : float, optional
        Electron temperature for the resistivity [eV].
    z_eff : float, optional
        Effective charge for the resistivity; never defaulted here [-].
    ln_lambda : float, optional
        Coulomb logarithm for the resistivity; never defaulted here [-].
    ion_mass_amu : float, optional
        Main-ion mass number, 1 for hydrogen [-].
    beta : float, optional
        Toroidal beta of the state, as a fraction (not percent) [-].

    Returns
    -------
    dict
        ``lundquist_number``, ``ion_skin_depth_over_a``,
        ``inverse_aspect_ratio``, ``beta``, ``beta_over_inverse_aspect_ratio``,
        each a float or NaN [-].

    Processing steps
    ----------------
    1. ``epsilon = a / R0``.
    2. With ``n_e``: ``d_i = c / omega_pi`` at ``n_i = n_e`` and ``v_A`` from
       ``B0`` and ``n_i m_i``.
    3. With ``T_e``, ``Z_eff`` and ``lnLambda`` as well: Spitzer ``eta`` and
       ``S = mu0 a v_A / eta``.
    4. ``beta`` passes through, and ``beta / epsilon`` follows.

    Convention
    ----------
    ``S`` is taken on the minor radius ``a``, the scale of
    :data:`~vaft.validation.orderings.ORDERING_QUANTITIES`; the ion density
    is ``n_e`` (singly charged main ion).

    Limitations
    -----------
    A single global ``T_e`` stands for the whole plasma in ``eta``; the
    resistivity of a peaked profile varies by an order of magnitude from core
    to edge, so ``S`` is an order-of-magnitude statement.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #1627 §1 and §4; the Lundquist number as in D. Biskamp,
       *Magnetic Reconnection in Plasmas*, Cambridge (2000), Ch. 1.
    """
    from vaft.formula.constants import MI_P
    from vaft.formula.equilibrium import inverse_aspect_ratio_from_a_R
    from vaft.formula.ordering import inertial_length, lundquist_number

    out = dict.fromkeys(("lundquist_number", "ion_skin_depth_over_a", "inverse_aspect_ratio", "beta",
                         "beta_over_inverse_aspect_ratio"), math.nan)
    if _finite_positive(minor_radius, major_radius):
        out["inverse_aspect_ratio"] = float(inverse_aspect_ratio_from_a_R(float(minor_radius), float(major_radius)))
    if _finite_positive(minor_radius, n_e, ion_mass_amu):
        out["ion_skin_depth_over_a"] = float(
            inertial_length(float(n_e), float(ion_mass_amu) * MI_P)) / float(minor_radius)
    if _finite_positive(minor_radius, b0, n_e, ion_mass_amu, t_e, z_eff, ln_lambda):
        out["lundquist_number"] = float(lundquist_number(
            float(minor_radius), _alfven_speed(b0, n_e, ion_mass_amu), _resistivity(t_e, z_eff, ln_lambda)))
    if _finite_positive(beta):
        out["beta"] = float(beta)
        if math.isfinite(out["inverse_aspect_ratio"]):
            out["beta_over_inverse_aspect_ratio"] = float(beta) / out["inverse_aspect_ratio"]
    return out


def time_history_ordering_quantities(
    *,
    minor_radius,
    b0,
    n_e=None,
    t_e=None,
    z_eff=None,
    ln_lambda=None,
    ion_mass_amu=1.0,
    plasma_current=None,
    current_rate=None,
    plasma_age=None,
):
    r"""How slowly a state evolves against Alfvenic and resistive times, NaN where an input is missing.

    $$\frac{\tau_{evol}}{\tau_A} = \frac{|I_p / \dot I_p|}{a / v_A},\qquad
      \frac{\tau_{age}}{\tau_R} = \frac{t - t_{onset}}{\mu_0 a^2/\eta}$$

    Parameters
    ----------
    minor_radius : float
        Minor radius ``a`` [m].
    b0 : float
        Toroidal field on axis [T].
    n_e : float, optional
        Electron density for ``v_A`` [m^-3].
    t_e : float, optional
        Electron temperature for ``eta`` [eV].
    z_eff : float, optional
        Effective charge for ``eta`` [-].
    ln_lambda : float, optional
        Coulomb logarithm for ``eta`` [-].
    ion_mass_amu : float, optional
        Main-ion mass number [-].
    plasma_current : float, optional
        Plasma current at the state [A].
    current_rate : float, optional
        ``dI_p/dt`` at the state, from the same current history [A/s].
    plasma_age : float, optional
        Time since plasma onset at the state [s].

    Returns
    -------
    dict
        ``tau_evolution_over_tau_alfven`` and ``tau_age_over_tau_resistive``,
        each a float or NaN [-].

    Processing steps
    ----------------
    1. ``tau_A = a / v_A`` from ``B0`` and ``n_i m_i`` (``n_i = n_e``).
    2. ``tau_evol = |I_p / (dI_p/dt)|``; a zero rate leaves it NaN rather than
       infinite.
    3. ``tau_R = mu0 a^2 / eta`` with Spitzer ``eta``, and the age over it.

    Convention
    ----------
    The evolution time is the plasma current's; another state variable
    (stored energy, axis position) gives another ``tau_evol`` and is the
    caller's to pass. Both ratios use the minor radius ``a``.

    Limitations
    -----------
    ``tau_R`` uses one global ``T_e``, as in
    :func:`global_ordering_quantities`; the current-relaxation statement is
    order-of-magnitude.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #1627 §9 (quasi-static equilibrium) and §10 (resistive
       current relaxation).
    """
    from vaft.formula.ordering import alfven_time, evolution_time, resistive_diffusion_time

    out = {"tau_evolution_over_tau_alfven": math.nan, "tau_age_over_tau_resistive": math.nan}
    if _finite_positive(minor_radius, b0, n_e, ion_mass_amu) and plasma_current is not None \
            and current_rate is not None:
        current, rate = float(plasma_current), float(current_rate)
        if math.isfinite(current) and math.isfinite(rate) and current != 0.0 and rate != 0.0:
            tau_a = float(alfven_time(float(minor_radius), _alfven_speed(b0, n_e, ion_mass_amu)))
            out["tau_evolution_over_tau_alfven"] = float(evolution_time(abs(current), abs(rate))) / tau_a
    if _finite_positive(minor_radius, t_e, z_eff, ln_lambda, plasma_age):
        tau_r = float(resistive_diffusion_time(float(minor_radius), _resistivity(t_e, z_eff, ln_lambda)))
        out["tau_age_over_tau_resistive"] = float(plasma_age) / tau_r
    return out


def _inverse_gradient_length(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``|dln y / dr|`` on the finite, positive samples of ``y``; NaN elsewhere.

    A profile known only on part of the grid (an ion temperature inferred
    inside the Thomson span) keeps its gradient where it is known instead of
    losing it everywhere.
    """
    from vaft.formula.utils import normalized_gradient_scale_length

    out = np.full(r.shape, np.nan)
    ok = np.isfinite(r) & np.isfinite(y) & (y > 0)
    if ok.sum() >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.abs(normalized_gradient_scale_length(r[ok], y[ok], 1.0))
    return out


def _masked(valid: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(valid.shape, np.nan)
    out[valid] = values[valid]
    return out


def profile_ordering_quantities(
    *,
    minor_radius_coordinate,
    n_e,
    t_e,
    magnetic_field,
    safety_factor,
    major_radius,
    z_eff,
    ln_lambda,
    ion_mass_amu=1.0,
    t_i=None,
):
    r"""The profile ordering quantities on each flux surface, NaN where an input is missing.

    $$\frac{\rho_s}{L_{T_e}},\ \frac{\rho_i}{L_{T_i}},\ Kn_s = \frac{v_{ts}\tau_s}{qR},\
      \Omega_{cs}\tau_s,\ \nu_{*s},\ \frac{\lambda_D}{L_{T_e}}$$

    Parameters
    ----------
    minor_radius_coordinate : array-like
        Minor-radius coordinate ``r`` of each sample, increasing outward [m].
    n_e : array-like
        Electron density on ``r`` [m^-3].
    t_e : array-like
        Electron temperature on ``r`` [eV].
    magnetic_field : float or array-like
        Field strength at each sample, e.g. the vacuum ``B0 R0 / R`` [T].
    safety_factor : array-like
        ``|q|`` on ``r`` from the equilibrium [-].
    major_radius : float
        Major radius for the connection length ``qR`` and ``epsilon = r/R`` [m].
    z_eff : float
        Effective charge for the electron collisions [-].
    ln_lambda : float
        Coulomb logarithm [-].
    ion_mass_amu : float, optional
        Main-ion mass number [-].
    t_i : array-like, optional
        Ion temperature on ``r``; without it every ion quantity is NaN [eV].

    Returns
    -------
    dict
        ``rho_s_over_LTe``, ``rho_i_over_LTi``,
        ``electron_parallel_knudsen_number``, ``ion_parallel_knudsen_number``,
        ``electron_magnetization``, ``ion_magnetization``,
        ``electron_collisionality``, ``ion_collisionality``,
        ``debye_length_over_L``: arrays on ``r`` [-].

    Raises
    ------
    ValueError
        The profile arrays do not share one length, or ``r`` is not increasing.

    Processing steps
    ----------------
    1. Gradient lengths ``L_T = |T / (dT/dr)|`` per species on ``r``, on the
       finite positive samples of that species' temperature only.
    2. ``rho_s`` from ``T_e`` and ``rho_i = v_ti / Omega_ci`` with
       ``v_ti = sqrt(T_i/m_i)``; each over its own ``L_T``.
    3. Braginskii collision times, mean free paths ``v_t tau`` and the
       parallel Knudsen numbers on ``qR``; magnetizations ``|Omega_c| tau``.
    4. Sauter collisionalities with the local ``epsilon = r / R``.
    5. ``lambda_D / L_Te``.

    Convention
    ----------
    ``rho_i`` uses ``sqrt(T_i/m_i)`` (NRL), the convention of
    :mod:`vaft.formula.ordering`; ``sqrt(2T/m)`` would raise it by ``sqrt 2``.
    The Debye length is taken against ``L_Te``, the profile's own scale.

    Limitations
    -----------
    Gradient lengths of sparse Thomson channels are finite differences of a
    few points; a profile fit should be passed when the raw channels are
    noisy. One main-ion species with ``n_i = n_e`` (no impurity dilution);
    impurities enter only through ``Z_eff`` in the electron collisions. One
    Coulomb logarithm serves both species, so the ion collision time and
    Sauter's ``nu*_i`` use the caller's (electron) ``lnLambda`` rather than
    Sauter's own ion value -- a 10-20 % effect on the ion quantities.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #1627 §6-§8 and the registry
       :data:`vaft.validation.orderings.ORDERING_QUANTITIES`; collisionality as
       in O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6, 2834 (1999).
    """
    from vaft.formula.constants import ME, MI_P, QE
    from vaft.formula.neoclassical import electron_collisionality_sauter, ion_collisionality_sauter
    from vaft.formula.ordering import (
        braginskii_electron_collision_time,
        braginskii_ion_collision_time,
        debye_length,
        knudsen_number,
        magnetization,
        mean_free_path,
        sound_gyroradius,
        thermal_speed,
    )
    from vaft.formula.particle import gyrofrequency, larmor_radius

    r = np.asarray(minor_radius_coordinate, dtype=float)
    ne = np.asarray(n_e, dtype=float)
    te = np.asarray(t_e, dtype=float)
    q = np.abs(np.asarray(safety_factor, dtype=float))
    b = np.broadcast_to(np.asarray(magnetic_field, dtype=float), r.shape).astype(float)
    ti = None if t_i is None else np.asarray(t_i, dtype=float)
    for name, array in (("n_e", ne), ("t_e", te), ("safety_factor", q)) + ((("t_i", ti),) if ti is not None else ()):
        if array.shape != r.shape:
            raise ValueError(f"{name} has shape {array.shape}, the radial coordinate {r.shape}")
    if r.size > 1 and np.any(np.diff(r) <= 0):
        raise ValueError("minor_radius_coordinate must increase outward")
    m_i = float(ion_mass_amu) * MI_P
    major = float(major_radius)
    names = ("rho_s_over_LTe", "rho_i_over_LTi", "electron_parallel_knudsen_number", "ion_parallel_knudsen_number",
             "electron_magnetization", "ion_magnetization", "electron_collisionality", "ion_collisionality",
             "debye_length_over_L")
    out = {name: np.full(r.shape, np.nan) for name in names}
    if not _finite_positive(major, z_eff, ln_lambda, ion_mass_amu) or r.size < 2:
        return out

    def ok(*arrays):
        mask = np.ones(r.shape, dtype=bool)
        for array in arrays:
            mask &= np.isfinite(array) & (array > 0)
        return mask

    inv_lte = _inverse_gradient_length(r, te)
    one = np.ones(r.shape)
    # Each quantity is gated only by what it uses: a missing q removes the
    # Knudsen numbers and collisionalities, never rho_s or the magnetization.
    safe = {name: np.where(ok(x), x, 1.0) for name, x in (("ne", ne), ("te", te), ("b", b), ("q", q), ("r", r))}
    e_base = ok(ne, te, b, r)
    e_q = e_base & ok(q)
    if e_base.any():
        rho_s = sound_gyroradius(safe["te"], m_i, safe["b"])
        out["rho_s_over_LTe"] = _masked(ok(te, b, r) & np.isfinite(inv_lte), rho_s * inv_lte)
        tau_e = braginskii_electron_collision_time(safe["ne"], safe["te"], float(ln_lambda), float(z_eff))
        out["electron_magnetization"] = _masked(e_base, magnetization(np.abs(gyrofrequency(-QE, ME, safe["b"])), tau_e))
        out["debye_length_over_L"] = _masked(ok(ne, te, r) & np.isfinite(inv_lte),
                                             debye_length(safe["ne"], safe["te"]) * inv_lte)
        if e_q.any():
            lam_e = mean_free_path(thermal_speed(safe["te"], ME), tau_e)
            out["electron_parallel_knudsen_number"] = _masked(e_q, knudsen_number(lam_e, safe["q"] * major))
            out["electron_collisionality"] = _masked(e_q, np.asarray(electron_collisionality_sauter(
                safe["ne"], safe["te"], safe["q"], major, safe["r"] / major, float(z_eff), float(ln_lambda)),
                dtype=float) * one)
    if ti is not None:
        inv_lti = _inverse_gradient_length(r, ti)
        safe_ti = np.where(ok(ti), ti, 1.0)
        i_base = ok(ne, ti, b, r)
        i_q = i_base & ok(q)
        if i_base.any():
            v_ti = thermal_speed(safe_ti, m_i)
            rho_i = larmor_radius(QE, m_i, v_ti, safe["b"])
            out["rho_i_over_LTi"] = _masked(ok(ti, b, r) & np.isfinite(inv_lti), np.abs(rho_i) * inv_lti)
            tau_i = braginskii_ion_collision_time(safe["ne"], safe_ti, float(ln_lambda), float(ion_mass_amu), 1.0)
            out["ion_magnetization"] = _masked(i_base, magnetization(np.abs(gyrofrequency(QE, m_i, safe["b"])), tau_i))
            if i_q.any():
                out["ion_parallel_knudsen_number"] = _masked(
                    i_q, knudsen_number(mean_free_path(v_ti, tau_i), safe["q"] * major))
                out["ion_collisionality"] = _masked(i_q, np.asarray(ion_collisionality_sauter(
                    safe["ne"], safe_ti, safe["q"], major, safe["r"] / major, 1.0, float(ln_lambda)),
                    dtype=float) * one)
    return out


def ordering_table(states, *, columns: Mapping[str, str] | None = None, ion_mass_amu=1.0):
    r"""The global and time-history ordering quantities of every state in a table.

    $$\text{row}_k \mapsto \{S, d_i/a, \epsilon, \beta, \beta/\epsilon,
      \tau_{evol}/\tau_A, \tau_{age}/\tau_R\}_k$$

    Parameters
    ----------
    states : pandas.DataFrame
        One row per plasma state; the inputs are read from the columns named
        in ``columns`` and a missing column means a missing input [-].
    columns : Mapping, optional
        Input name -> column name, overriding :data:`DEFAULT_COLUMNS`
        (``minor_radius``, ``major_radius``, ``b0``, ``n_e``, ``t_e``,
        ``z_eff``, ``ln_lambda``, ``beta``, ``plasma_current``,
        ``current_rate``, ``plasma_age``) [-].
    ion_mass_amu : float, optional
        Main-ion mass number [-].

    Returns
    -------
    pandas.DataFrame
        The input index, the seven ordering columns, and ``attrs['columns']``
        recording which input column fed each argument [-].

    Processing steps
    ----------------
    1. Resolve the input columns; an absent one is a missing input.
    2. :func:`global_ordering_quantities` and
       :func:`time_history_ordering_quantities` on every row.

    Limitations
    -----------
    Profile quantities need radial profiles and are built per state with
    :func:`profile_ordering_quantities`, not here.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #1627 §2 (the summary layer) and #1629 (the VEST atlas).
    """
    import pandas as pd

    mapping = {**DEFAULT_COLUMNS, **dict(columns or {})}
    rows = []
    for _, row in states.iterrows():
        get = {name: (row[column] if column in states.columns else None) for name, column in mapping.items()}
        common = dict(minor_radius=get["minor_radius"], b0=get["b0"], n_e=get["n_e"], t_e=get["t_e"],
                      z_eff=get["z_eff"], ln_lambda=get["ln_lambda"], ion_mass_amu=ion_mass_amu)
        values = global_ordering_quantities(major_radius=get["major_radius"], beta=get["beta"], **common)
        values.update(time_history_ordering_quantities(plasma_current=get["plasma_current"],
                                                       current_rate=get["current_rate"],
                                                       plasma_age=get["plasma_age"], **common))
        rows.append(values)
    table = pd.DataFrame(rows, index=states.index)
    table.attrs["columns"] = {name: column for name, column in mapping.items() if column in states.columns}
    table.attrs["ion_mass_amu"] = float(ion_mass_amu)
    return table


#: Ordering groups defined once per state; every other group is per flux surface or mode.
_STATE_GROUPS = ("global", "time_history")


def _state_quantity(name: str) -> bool:
    from vaft.validation.orderings import ORDERING_QUANTITIES

    return ORDERING_QUANTITIES[name].group in _STATE_GROUPS


def ordering_coverage(states, points=None):
    r"""The fraction of states (or profile points) with a finite value, per ordering quantity, by #1627 tier.

    $$f_q = \frac{\#\{k : q_k\ \text{finite}\}}{N}$$

    Parameters
    ----------
    states : pandas.DataFrame
        One row per state; global and time-history quantities are counted
        here [-].
    points : pandas.DataFrame, optional
        One row per state and flux surface; profile, foundational and mode
        quantities are counted here when given, else in ``states`` [-].

    Returns
    -------
    pandas.DataFrame
        One row per registered quantity: ``group``, ``tier``, ``of`` (states or
        points), ``assessed`` (count) and ``fraction``; a quantity the table
        lacks has fraction 0 [-].

    Limitations
    -----------
    Coverage counts finite values only; it says nothing about how well a
    quantity is determined.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #1627 "Data-availability tiers": the summary reports coverage
       by tier.
    """
    import pandas as pd

    from vaft.validation.orderings import ORDERING_QUANTITIES

    tier_of = {group: tier for tier, groups in COVERAGE_TIERS.items() for group in groups}
    flow = set(COVERAGE_TIERS["tier3_flow"])
    rows = []
    for name, quantity in ORDERING_QUANTITIES.items():
        table = states if (points is None or quantity.group in _STATE_GROUPS) else points
        if name in table.columns:
            values = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
            assessed = int(np.isfinite(values).sum())
        else:
            assessed = 0
        rows.append({"quantity": name, "group": quantity.group,
                     "tier": "tier3_flow" if name in flow else tier_of[quantity.group],
                     "of": "states" if table is states else "points",
                     "assessed": assessed, "fraction": assessed / len(table) if len(table) else math.nan})
    return pd.DataFrame(rows).set_index("quantity")


def ordering_margins(contracts, states, points=None):
    r"""Median ordering margin and the fraction on the permitted side, per assessable ordering.

    $$m = \log_{10}(x_0/x)\ (\text{small}),\qquad m = \log_{10}(x/x_0)\ (\text{large})$$

    Parameters
    ----------
    contracts : Mapping or iterable of ApproximationContract
        The contracts whose orderings are summarised; an ordering listed by
        several is reported once [-].
    states : pandas.DataFrame
        Global and time-history quantities, one row per state [-].
    points : pandas.DataFrame, optional
        Profile quantities, one row per flux surface; ``states`` is used when
        absent [-].

    Returns
    -------
    pandas.DataFrame
        Indexed by ``"<quantity> (<ordering>)"``: ``per`` (state or point), ``n``,
        ``median_margin`` [decades] and ``permitted`` (fraction with ``m > 0``),
        sorted from the least to the most room [-].

    Processing steps
    ----------------
    1. Collect each distinct ``(quantity, ordering, threshold)`` once.
    2. Take a global or time-history quantity from ``states`` (one value per
       state, never repeated per profile point) and any other from ``points``.
    3. Margins with :func:`vaft.validation.applicability.ordering_margin`.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issues #1627 (margins first) and #1629 §5.
    """
    import pandas as pd

    from vaft.validation.applicability import ordering_margin

    items = contracts.values() if hasattr(contracts, "values") else contracts
    seen, rows = set(), []
    for contract in items:
        for a in contract.assumptions:
            key = (a.quantity, a.ordering, a.threshold)
            if key in seen:
                continue
            seen.add(key)
            table = states if (points is None or _state_quantity(a.quantity)) else points
            if a.quantity not in table:
                continue
            values = pd.to_numeric(table[a.quantity], errors="coerce").dropna()
            if values.empty:
                continue
            m = values.map(lambda v: ordering_margin(v, a.ordering, a.threshold)).dropna()
            if m.empty:
                continue
            rows.append({"ordering": f"{a.quantity} ({a.ordering})",
                         "per": "state" if table is states else "point", "n": int(len(m)),
                         "median_margin": float(m.median()), "permitted": float((m > 0).mean())})
    frame = pd.DataFrame(rows, columns=["ordering", "per", "n", "median_margin", "permitted"])
    return frame.set_index("ordering").sort_values("median_margin")


def contract_population(contract, states, points=None):
    r"""Evaluate one contract on the table its orderings live on.

    $$\text{table} = \begin{cases}\text{states} & \text{every ordering global or time-history}\\
      \text{points} & \text{otherwise}\end{cases}$$

    Parameters
    ----------
    contract : ApproximationContract
        A #1627 ordering contract [-].
    states : pandas.DataFrame
        One row per state [-].
    points : pandas.DataFrame, optional
        One row per state and flux surface, carrying the state's global and
        time-history quantities as well; required for a contract with a
        profile ordering [-].

    Returns
    -------
    pandas.DataFrame
        :func:`vaft.validation.applicability.evaluate_population` output, with
        ``attrs['evaluated_on']`` ``"states"`` or ``"points"`` [-].

    Limitations
    -----------
    A contract mixing global and profile orderings is evaluated per flux
    surface, so a state contributes one row per surface; a state without
    profiles is then absent from it.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #1629 §8-§9: per-state evaluation of state-level contracts,
       per-surface of local ones.
    """
    from vaft.validation.applicability import evaluate_population

    state_level = all(_state_quantity(a.quantity) for a in contract.assumptions)
    table = states if (state_level or points is None) else points
    evaluated = evaluate_population(contract, table)
    evaluated.attrs["evaluated_on"] = "states" if table is states else "points"
    return evaluated
