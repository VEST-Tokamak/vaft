"""Analytic L-mode, H-mode and ITB plasma states on a fixed geometry (#1045).

A lightweight preset and composition layer: prescribed kinetic profiles built
from the #552 profile kernels, the pressures derived from them, and the
projection of any of them onto the ``(R, Z)`` grid of an existing equilibrium
through its normalized flux.  Nothing here solves an equilibrium, a transport
problem or a pedestal model, and nothing makes a state consistent with the
geometry it is projected onto.  :mod:`vaft.process.profile` is the public
import location.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from vaft.data.analytic_plasma_state import AnalyticPlasmaState, AnalyticProfile, BarrierStep
from vaft.formula.atomic import impurity_fraction_from_effective_charge
from vaft.formula.constants import QE
from vaft.formula.equilibrium import (
    generalized_parabolic_profile,
    generalized_parabolic_profile_derivative,
    modified_tanh_profile,
    modified_tanh_profile_derivative,
)

__all__ = [
    "analytic_hmode_itb_state",
    "analytic_hmode_state",
    "analytic_itb_state",
    "analytic_lmode_state",
    "compose_analytic_profile",
    "compose_plasma_state",
    "evaluate_analytic_profile",
    "evaluate_plasma_state",
    "project_flux_function",
    "project_plasma_state",
]

#: The prescribed channels of a state and their units.
_CHANNELS = {"n_e": "m^-3", "T_e": "eV", "T_i": "eV"}

#: Every quantity :func:`evaluate_plasma_state` answers for.
_STATE_QUANTITIES = (
    "n_e", "n_i", "n_impurity", "T_e", "T_i", "p_e", "p_i", "p_total",
    "dp_e_dpsi_norm", "dp_i_dpsi_norm", "dp_total_dpsi_norm",
)

#: Samples used to check a composed profile for positivity; refined around
#: each barrier by :func:`_check_grid`.
_CHECK_GRID = np.linspace(0.0, 1.0, 2001)

#: Narrowest and widest barrier layer, full width in psi_norm.  Below the
#: minimum the step is a discontinuity on any practical grid; above the maximum
#: it is no longer a localized layer but a second core shape.
MIN_BARRIER_WIDTH = 1e-3
MAX_BARRIER_WIDTH = 0.5

#: Smallest ``S(knee) - C(knee)`` the pedestal solve accepts: below it the
#: pedestal step and the core are indistinguishable at the knee.
_MIN_PEDESTAL_CONDITIONING = 1e-3

#: How far below zero psi_norm may fall, without an LCFS outline, and still be
#: taken as the axis side of the plasma.  With an outline the axis side is
#: decided by containment alone.
_AXIS_TOLERANCE = 0.05

#: Default grid of a state: uniform in psi_norm, axis and separatrix included.
_DEFAULT_POINTS = 201

#: Illustrative small-tokamak values the presets default to; assumed, not a
#: machine setting.  Densities in m^-3, temperatures in eV.
_DEFAULTS = {
    "ne_axis": 2.0e19, "ne_ped": 1.2e19, "ne_sep": 2.0e18,
    "te_axis": 250.0, "te_ped": 80.0, "te_sep": 10.0,
    "ti_axis": 150.0, "ti_ped": 60.0, "ti_sep": 10.0,
    "ne_itb_height": 4.0e18, "te_itb_height": 120.0, "ti_itb_height": 60.0,
}


# --- the normalized barrier step ----------------------------------------------


def _step_scale(position: float, width: float) -> tuple[float, float]:
    """``S(1)`` and ``S(0) - S(1)`` of the unit Groebner step."""
    ends = modified_tanh_profile(np.array([0.0, 1.0]), pedestal_height=1.0,
                                 pedestal_position=position, pedestal_width=width)
    return float(ends[1]), float(ends[0] - ends[1])


def _step(x, step: BarrierStep, *, derivative: bool = False):
    """``height * (S(x) - S(1)) / (S(0) - S(1))``, or its psi_norm derivative."""
    tail, span = _step_scale(step.position, step.width)
    if derivative:
        return step.height * modified_tanh_profile_derivative(
            x, pedestal_height=1.0, pedestal_position=step.position, pedestal_width=step.width) / span
    unit = modified_tanh_profile(x, pedestal_height=1.0, pedestal_position=step.position,
                                 pedestal_width=step.width)
    return step.height * (unit - tail) / span


def _core(x, alpha: float, beta: float, *, derivative: bool = False):
    kernel = generalized_parabolic_profile_derivative if derivative else generalized_parabolic_profile
    return kernel(x, core_value=1.0, edge_value=0.0, alpha=alpha, beta=beta)


def _check_grid(barriers) -> np.ndarray:
    """The uniform check grid plus 401 points within three widths of each barrier centre."""
    parts = [_CHECK_GRID]
    for step in barriers:
        parts.append(np.clip(np.linspace(step.position - 3 * step.width, step.position + 3 * step.width, 401),
                             0.0, 1.0))
    return np.unique(np.concatenate(parts))


def _barrier_width(state: str, name: str, value) -> float:
    width = _finite(name, value)
    if not MIN_BARRIER_WIDTH <= width <= MAX_BARRIER_WIDTH:
        raise ValueError(
            f"{state}: {name} = {width!r} is outside [{MIN_BARRIER_WIDTH}, {MAX_BARRIER_WIDTH}]; "
            "narrower is a discontinuity, wider is not a localized barrier"
        )
    return width


def _finite(name: str, value) -> float:
    if np.ndim(value) != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    return float(value)


# --- one profile ---------------------------------------------------------------


def compose_analytic_profile(
    quantity: str,
    *,
    axis_value: float,
    separatrix_value: float,
    core_alpha: float = 1.0,
    core_beta: float = 1.5,
    pedestal_top_value: float | None = None,
    pedestal_position: float | None = None,
    pedestal_width: float | None = None,
    itb_height: float = 0.0,
    itb_position: float | None = None,
    itb_width: float | None = None,
    unit: str | None = None,
    positive: bool = True,
) -> AnalyticProfile:
    r"""Compose one kinetic profile from a smooth core, an optional edge pedestal and an optional ITB.

    $$f(\psi_N) = f_\mathrm{sep} + A\,C(\psi_N) + H_\mathrm{ped}\,\tilde S_\mathrm{ped}(\psi_N) + H_\mathrm{itb}\,\tilde S_\mathrm{itb}(\psi_N)$$

    with $C = (1-\psi_N^{\alpha})^{\beta}$ the generalized parabolic core and
    $\tilde S = (S - S(1))/(S(0) - S(1))$ a Groebner tanh step normalized to
    exactly one on axis and zero at the separatrix.  $A$ and $H_\mathrm{ped}$
    are solved so the requested values hold exactly.

    Parameters
    ----------
    quantity : str
        Name of the profile, e.g. ``"n_e"``, ``"T_e"`` or ``"T_i"`` [-].
    axis_value : float
        Value on the magnetic axis, ``psi_norm = 0``; holds exactly [any].
    separatrix_value : float
        Value at ``psi_norm = 1``; holds exactly [any].
    core_alpha : float, optional
        Radial exponent of the core shape, at least one [-].
    core_beta : float, optional
        Peaking exponent of the core shape, at least one [-].
    pedestal_top_value : float, optional
        Value at the pedestal knee, ``pedestal_position - pedestal_width/2``;
        holds exactly.  ``None`` for no edge pedestal [any].
    pedestal_position : float, optional
        Centre of the pedestal's steep-gradient layer in ``psi_norm``; required
        with *pedestal_top_value* [-].
    pedestal_width : float, optional
        Full width of the pedestal layer in ``psi_norm``, positive [-].
    itb_height : float, optional
        Whole amplitude of the internal-barrier step, from its separatrix side
        to its axis side (the knee-to-foot change is about ``tanh(1) = 0.76``
        of it); zero for no ITB [any].
    itb_position : float, optional
        Centre of the ITB layer in ``psi_norm``, strictly inside ``(0, 1)``;
        required when *itb_height* is non-zero [-].
    itb_width : float, optional
        Full width of the ITB layer in ``psi_norm``, positive [-].
    unit : str, optional
        Unit label recorded on the profile; ``m^-3`` for ``n_*`` and ``eV``
        for ``T_*`` when omitted [-].
    positive : bool, optional
        Refuse a profile that is not strictly positive on ``[0, 1]`` [-].

    Returns
    -------
    AnalyticProfile
        The requested values, the barrier steps and the solved core amplitude
        and pedestal height; evaluate it with :func:`evaluate_analytic_profile` [-].

    Raises
    ------
    ValueError
        A non-finite value, an exponent below one, a pedestal given in part,
        a width outside ``[1e-3, 0.5]``, a pedestal knee outside ``(0, 1)``, an
        ITB layer not strictly inside ``(0, 1)`` or overlapping the pedestal
        layer, a pedestal knee where the core and the step cannot be told apart,
        or a profile that is not positive (checked on a grid refined around
        each barrier) when *positive* is set.

    Convention
    ----------
    The coordinate is the normalized poloidal flux $\psi_N$, 0 on axis and 1
    at the separatrix, so the profile is independent of the flux sign and of
    Wb against Wb/rad storage.  Barrier widths are *full* widths
    (Groebner-Carlstrom): the steep layer runs from ``position - width/2``
    (the knee) to ``position + width/2`` (the foot), and the steepest gradient
    of an isolated step, $-H/(\Delta\,(S(0)-S(1)))$, sits at ``position``.
    ``itb_height`` is the whole step amplitude (axis side minus separatrix
    side), not the knee-to-foot change; the pedestal is specified by its top
    value instead.  Widths must lie in ``[1e-3, 0.5]``; an ITB layer must lie
    inside ``(0, 1)`` and end before the pedestal layer begins.

    Assumptions
    -----------
    The axis value is held fixed: adding an ITB or a pedestal redistributes
    the profile under it rather than raising the axis.  A caller who wants a
    higher core with an ITB raises *axis_value* as well.

    Applicability
    -------------
    Machine-independent.  Synthetic, prescribed profiles for demonstrations,
    tests and fixtures; not a transport or pedestal model.

    Limitations
    -----------
    No pedestal width or height is predicted (EPED), and the ITB carries no
    physics of its formation.  A large ITB or pedestal with a fixed axis value
    can make the core amplitude negative, i.e. a hollow core; that is returned
    as asked, and only non-positivity is refused.  The steps are rescaled to
    vanish exactly at ``psi_norm = 1``, so a layer centred at or beyond the
    separatrix is compressed rather than cut.

    Provenance
    ----------
    .. [552] :func:`vaft.formula.equilibrium.generalized_parabolic_profile` and
       :func:`vaft.formula.equilibrium.modified_tanh_profile` (#552), the core
       shape and the step reused unchanged.
    .. [GC98] R. J. Groebner and T. N. Carlstrom, Plasma Phys. Control. Fusion
       40, 673 (1998), for the tanh step and its full-width convention.
    .. [1045] VAFT issue #1045, which fixes the composition: exact axis,
       separatrix and pedestal-top values, independent ITB per channel.
    """
    axis = _finite("axis_value", axis_value)
    sep = _finite("separatrix_value", separatrix_value)
    alpha = _finite("core_alpha", core_alpha)
    beta = _finite("core_beta", core_beta)
    if alpha < 1.0 or beta < 1.0:
        raise ValueError(
            f"core_alpha and core_beta must be at least one, got {alpha!r} and {beta!r}; "
            "below one the core gradient is infinite at an endpoint"
        )
    itb = None
    h_itb = _finite("itb_height", itb_height)
    if h_itb != 0.0:
        if itb_position is None or itb_width is None:
            raise ValueError("a non-zero itb_height needs itb_position and itb_width")
        position = _finite("itb_position", itb_position)
        if not 0.0 < position < 1.0:
            raise ValueError(f"{quantity} ITB: itb_position must lie strictly inside (0, 1), got {position!r}")
        width = _barrier_width(f"{quantity} ITB", "itb_width", itb_width)
        itb = BarrierStep(position=position, width=width, height=h_itb)
        if not (0.0 < itb.knee and itb.foot < 1.0):
            raise ValueError(
                f"{quantity} ITB: the layer [{itb.knee:.4g}, {itb.foot:.4g}] must lie strictly inside "
                "(0, 1); an ITB reaching the separatrix is a pedestal"
            )

    pedestal_given = [v is not None for v in (pedestal_top_value, pedestal_position, pedestal_width)]
    pedestal = None
    top = None
    if any(pedestal_given):
        if not all(pedestal_given):
            raise ValueError("a pedestal needs pedestal_top_value, pedestal_position and pedestal_width")
        top = _finite("pedestal_top_value", pedestal_top_value)
        position = _finite("pedestal_position", pedestal_position)
        width = _barrier_width(f"{quantity} pedestal", "pedestal_width", pedestal_width)
        knee = position - 0.5 * width
        if not 0.0 < knee < 1.0:
            raise ValueError(f"{quantity} pedestal: the knee {knee!r} must lie strictly inside (0, 1)")
        if itb is not None and itb.foot >= knee:
            raise ValueError(
                f"{quantity}: the ITB layer [{itb.knee:.4g}, {itb.foot:.4g}] overlaps the pedestal layer "
                f"starting at its knee {knee:.4g}; move the ITB inward or narrow one of them"
            )
        # f(0) = axis and f(knee) = top, linear in (A, H_ped).
        unit_step = BarrierStep(position=position, width=width, height=1.0)
        s_k = float(_step(knee, unit_step))
        c_k = float(_core(knee, alpha, beta))
        t_k = float(_step(knee, itb)) if itb is not None else 0.0
        if s_k - c_k < _MIN_PEDESTAL_CONDITIONING:
            raise ValueError(
                f"{quantity} pedestal: at the knee {knee:.4g} the core shape ({c_k:.4g}) and the step "
                f"({s_k:.4g}) are indistinguishable, so the top value cannot be set; move the pedestal "
                "outward or peak the core (larger core_alpha or core_beta)"
            )
        height = (top - sep - (axis - sep - h_itb) * c_k - t_k) / (s_k - c_k)
        pedestal = BarrierStep(position=position, width=width, height=float(height))
    amplitude = axis - sep - h_itb - (pedestal.height if pedestal is not None else 0.0)

    if unit is None:
        unit = "m^-3" if quantity.startswith("n") else "eV" if quantity.startswith("T") else ""
    profile = AnalyticProfile(
        quantity=quantity, unit=unit, axis_value=axis, separatrix_value=sep,
        core_alpha=alpha, core_beta=beta, core_amplitude=float(amplitude),
        pedestal=pedestal, pedestal_top_value=top, itb=itb,
    )
    if positive:
        grid = _check_grid(profile.barriers)
        values = evaluate_analytic_profile(profile, grid)
        if not np.all(values > 0.0):
            where = float(grid[np.argmin(values)])
            raise ValueError(
                f"{quantity} is not positive on [0, 1] (minimum {values.min():.4g} at "
                f"psi_norm = {where:.3f}); check the requested values"
            )
    return profile


def evaluate_analytic_profile(profile: AnalyticProfile, psi_norm, *, derivative: bool = False):
    r"""Evaluate an analytic profile, or its gradient against normalized flux, anywhere in ``[0, 1]``.

    Parameters
    ----------
    profile : AnalyticProfile
        From :func:`compose_analytic_profile` or a state's ``profiles`` [-].
    psi_norm : float or array_like
        Normalized poloidal flux in ``[0, 1]`` [-].
    derivative : bool, optional
        Return $df/d\psi_N$ instead of $f$ [-].

    Returns
    -------
    float or np.ndarray
        The profile, or its analytic derivative, in the profile's unit (per
        unit ``psi_norm`` for the derivative), with the shape of *psi_norm* [any].

    Raises
    ------
    ValueError
        *psi_norm* is not finite or lies outside ``[0, 1]``.

    Convention
    ----------
    The derivative is against $\psi_N$, not $\psi$; divide by
    $\psi_\mathrm{boundary}-\psi_\mathrm{axis}$ of a chosen equilibrium, in its
    own flux unit and sign, for $df/d\psi$.  It is the sum of the kernels'
    analytic derivatives, not a finite difference.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [552] :func:`vaft.formula.equilibrium.generalized_parabolic_profile_derivative`
       and :func:`vaft.formula.equilibrium.modified_tanh_profile_derivative` (#552).
    """
    x = np.asarray(psi_norm, dtype=float)
    value = _core(x, profile.core_alpha, profile.core_beta, derivative=derivative) * profile.core_amplitude
    if not derivative:
        value = value + profile.separatrix_value
    for step in profile.barriers:
        value = value + _step(x, step, derivative=derivative)
    return float(value) if np.ndim(value) == 0 else value


# --- a state ---------------------------------------------------------------------


def _composition(z_eff: float, impurity_charge: float) -> tuple[float, float]:
    """``n_impurity/n_e`` and ``(n_i + n_impurity)/n_e`` for a hydrogenic main ion."""
    fraction = impurity_fraction_from_effective_charge(z_eff, impurity_charge)
    return fraction, 1.0 - (impurity_charge - 1.0) * fraction


def evaluate_plasma_state(state: AnalyticPlasmaState, psi_norm, quantity: str = "p_total"):
    r"""Evaluate one quantity of an analytic plasma state at any normalized flux.

    Parameters
    ----------
    state : AnalyticPlasmaState
        From :func:`compose_plasma_state` or a preset [-].
    psi_norm : float or array_like
        Normalized poloidal flux in ``[0, 1]`` [-].
    quantity : str, optional
        One of ``n_e``, ``n_i``, ``n_impurity``, ``T_e``, ``T_i``, ``p_e``,
        ``p_i``, ``p_total``, ``dp_e_dpsi_norm``, ``dp_i_dpsi_norm``,
        ``dp_total_dpsi_norm`` [-].

    Returns
    -------
    float or np.ndarray
        The quantity in the unit of :data:`vaft.data.ANALYTIC_STATE_UNITS`
        (m^-3, eV, Pa, or Pa per unit ``psi_norm``) [any].

    Raises
    ------
    ValueError
        An unknown quantity, or *psi_norm* outside ``[0, 1]``.

    Convention
    ----------
    ``p = e n T`` with *T* in eV and ``e`` the elementary charge, so pressure
    is in Pa; ``p_i`` counts the main ion and the impurity, both at ``T_i``.
    Gradients are against ``psi_norm`` by the product rule on the channels'
    analytic derivatives.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [W11] J. Wesson, *Tokamaks*, 4th ed. (2011), Ch. 2, for quasi-neutrality
       and the effective charge that fix the ion densities (through
       :func:`vaft.formula.atomic.impurity_fraction_from_effective_charge`).
    """
    if quantity not in _STATE_QUANTITIES:
        raise ValueError(f"quantity must be one of {_STATE_QUANTITIES}, got {quantity!r}")
    x = np.asarray(psi_norm, dtype=float)
    impurity, ions = _composition(state.z_eff, state.impurity_charge)
    prof = state.profiles

    def f(name, d=False):
        return np.asarray(evaluate_analytic_profile(prof[name], x, derivative=d), dtype=float)

    if quantity in ("n_e", "T_e", "T_i"):
        out = f(quantity)
    elif quantity == "n_i":
        out = f("n_e") * (1.0 - state.impurity_charge * impurity)
    elif quantity == "n_impurity":
        out = f("n_e") * impurity
    else:
        n_e, dn_e = f("n_e"), f("n_e", True)
        p_e = QE * n_e * f("T_e")
        p_i = QE * ions * n_e * f("T_i")
        dp_e = QE * (dn_e * f("T_e") + n_e * f("T_e", True))
        dp_i = QE * ions * (dn_e * f("T_i") + n_e * f("T_i", True))
        out = {"p_e": p_e, "p_i": p_i, "p_total": p_e + p_i, "dp_e_dpsi_norm": dp_e,
               "dp_i_dpsi_norm": dp_i, "dp_total_dpsi_norm": dp_e + dp_i}[quantity]
    return float(out) if out.ndim == 0 else out


def compose_plasma_state(
    n_e: AnalyticProfile,
    T_e: AnalyticProfile,
    T_i: AnalyticProfile,
    *,
    z_eff: float = 1.0,
    impurity_charge: float = 6.0,
    psi_norm=None,
    label: str = "analytic",
) -> AnalyticPlasmaState:
    r"""Assemble prescribed ``n_e``, ``T_e`` and ``T_i`` profiles into one kinetic state with derived pressure.

    Parameters
    ----------
    n_e : AnalyticProfile
        Electron density profile [m^-3].
    T_e : AnalyticProfile
        Electron temperature profile [eV].
    T_i : AnalyticProfile
        Ion temperature profile, shared by the main ion and the impurity [eV].
    z_eff : float, optional
        Effective charge, uniform in radius, in ``[1, impurity_charge]`` [-].
    impurity_charge : float, optional
        Charge of the single fully stripped impurity, greater than one [-].
    psi_norm : array_like, optional
        Grid to sample on, inside ``[0, 1]``; 201 uniform points when omitted [-].
    label : str, optional
        Name recorded on the state, e.g. ``"H-mode"`` [-].

    Returns
    -------
    AnalyticPlasmaState
        The three channels, the ion and impurity densities, the electron, ion
        and total pressures and their analytic gradients against ``psi_norm``,
        sampled on *psi_norm*, with the channel definitions for exact
        evaluation anywhere [-].

    Raises
    ------
    ValueError
        A profile whose quantity or unit does not match its slot, *z_eff*
        outside ``[1, impurity_charge]``, or a grid that is not 1-D, strictly
        increasing and inside ``[0, 1]``.

    Output semantics
    ----------------
    Synthetic: prescribed kinetic profiles and quantities derived from them.
    No measurement, fit or equilibrium is involved.

    Defaults
    --------
    ``z_eff = 1`` (assumed value): a pure hydrogenic plasma, so ``n_i = n_e``.
    ``impurity_charge = 6`` (assumed value, carbon) only matters once
    ``z_eff > 1``.  The 201-point grid is a numerical convenience.

    Convention
    ----------
    Coordinate ``psi_norm``; ``rho_pol_norm = sqrt(psi_norm)`` is exposed by
    the record, never a toroidal radius.  Quasi-neutrality
    ``n_e = n_i + Z_I n_impurity`` with a hydrogenic main ion and
    ``Z_eff n_e = n_i + Z_I^2 n_impurity``; pressure ``p = e n T`` in Pa with
    ``T`` in eV, ions at ``T_i``.  The pressure is derived, never prescribed,
    so it cannot disagree with the stored densities and temperatures.

    Assumptions
    -----------
    Uniform ``Z_eff``; one impurity charge state; thermal particles only (no
    fast-ion pressure).

    Applicability
    -------------
    Machine-independent.  A lightweight preset layer for demonstrations,
    tests and synthetic-diagnostic fixtures; the general synthetic kinetic
    framework is #122.

    Limitations
    -----------
    The state is not Grad-Shafranov consistent with any geometry; projecting
    it onto one (:func:`project_plasma_state`) does not change that.

    Provenance
    ----------
    .. [W11] J. Wesson, *Tokamaks*, 4th ed. (2011), Ch. 2, for quasi-neutrality
       and the effective charge.
    .. [1045] VAFT issue #1045: pressure derived from the stored kinetic state.
    """
    for name, profile in (("n_e", n_e), ("T_e", T_e), ("T_i", T_i)):
        if not isinstance(profile, AnalyticProfile):
            raise ValueError(f"{name} must be an AnalyticProfile, got {type(profile).__name__}")
        if profile.quantity != name or profile.unit != _CHANNELS[name]:
            raise ValueError(
                f"the {name} slot got a {profile.quantity!r} profile in {profile.unit!r}; "
                f"expected {name!r} in {_CHANNELS[name]!r}"
            )
    z_eff = _finite("z_eff", z_eff)
    impurity_charge = _finite("impurity_charge", impurity_charge)
    _composition(z_eff, impurity_charge)  # validates the pair
    grid = np.linspace(0.0, 1.0, _DEFAULT_POINTS) if psi_norm is None else np.asarray(psi_norm, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("psi_norm must be a non-empty 1-D grid")
    skeleton = AnalyticPlasmaState(
        label=label, psi_norm=grid, **{name: np.zeros_like(grid) for name in _STATE_QUANTITIES},
        profiles={"n_e": n_e, "T_e": T_e, "T_i": T_i},
        z_eff=z_eff, impurity_charge=impurity_charge,
    )
    arrays = {name: np.asarray(evaluate_plasma_state(skeleton, grid, name), dtype=float).reshape(grid.shape)
              for name in _STATE_QUANTITIES}
    return AnalyticPlasmaState(
        label=label, psi_norm=grid, **arrays, profiles=skeleton.profiles,
        z_eff=z_eff, impurity_charge=impurity_charge,
        metadata={"source": "vaft.process.profile.compose_plasma_state", "issue": 1045,
                  "grad_shafranov_consistent": False},
    )


# --- presets ---------------------------------------------------------------------


def _per_channel(value, channel: str, name: str):
    """A scalar shared by every channel, or a mapping keyed by channel."""
    if isinstance(value, Mapping):
        if channel not in value:
            raise ValueError(f"{name} is a mapping without an entry for {channel!r}")
        return value[channel]
    return value


def _preset(label, values, *, pedestal, itb, core_alpha, core_beta, z_eff, impurity_charge, psi_norm,
            pedestal_position=None, pedestal_width=None, itb_position=None, itb_width=None):
    profiles = {}
    for channel, prefix in (("n_e", "ne"), ("T_e", "te"), ("T_i", "ti")):
        kwargs = dict(axis_value=values[f"{prefix}_axis"], separatrix_value=values[f"{prefix}_sep"],
                      core_alpha=_per_channel(core_alpha, channel, "core_alpha"),
                      core_beta=_per_channel(core_beta, channel, "core_beta"))
        if pedestal:
            kwargs.update(pedestal_top_value=values[f"{prefix}_ped"],
                          pedestal_position=_per_channel(pedestal_position, channel, "pedestal_position"),
                          pedestal_width=_per_channel(pedestal_width, channel, "pedestal_width"))
        if itb:
            kwargs.update(itb_height=values[f"{prefix}_itb_height"],
                          itb_position=_per_channel(itb_position, channel, "itb_position"),
                          itb_width=_per_channel(itb_width, channel, "itb_width"))
        profiles[channel] = compose_analytic_profile(channel, **kwargs)
    return compose_plasma_state(profiles["n_e"], profiles["T_e"], profiles["T_i"], z_eff=z_eff,
                                impurity_charge=impurity_charge, psi_norm=psi_norm, label=label)


def analytic_lmode_state(
    *,
    ne_axis: float = _DEFAULTS["ne_axis"],
    ne_sep: float = _DEFAULTS["ne_sep"],
    te_axis: float = _DEFAULTS["te_axis"],
    te_sep: float = _DEFAULTS["te_sep"],
    ti_axis: float = _DEFAULTS["ti_axis"],
    ti_sep: float = _DEFAULTS["ti_sep"],
    core_alpha=1.0,
    core_beta=1.5,
    z_eff: float = 1.0,
    impurity_charge: float = 6.0,
    psi_norm=None,
) -> AnalyticPlasmaState:
    r"""Smooth L-mode-like kinetic state: generalized parabolic profiles with no barrier.

    Parameters
    ----------
    ne_axis : float, optional
        Electron density on axis [m^-3].
    ne_sep : float, optional
        Electron density at the separatrix [m^-3].
    te_axis : float, optional
        Electron temperature on axis [eV].
    te_sep : float, optional
        Electron temperature at the separatrix [eV].
    ti_axis : float, optional
        Ion temperature on axis [eV].
    ti_sep : float, optional
        Ion temperature at the separatrix [eV].
    core_alpha : float or mapping, optional
        Core radial exponent, shared or keyed by ``n_e``/``T_e``/``T_i`` [-].
    core_beta : float or mapping, optional
        Core peaking exponent, shared or keyed by channel [-].
    z_eff : float, optional
        Uniform effective charge [-].
    impurity_charge : float, optional
        Charge of the single impurity [-].
    psi_norm : array_like, optional
        Sampling grid in ``[0, 1]`` [-].

    Returns
    -------
    AnalyticPlasmaState
        Labelled ``"L-mode"``; see :func:`compose_plasma_state` [-].

    Raises
    ------
    ValueError
        As for :func:`compose_analytic_profile` and :func:`compose_plasma_state`.

    Defaults
    --------
    Densities and temperatures are illustrative small-tokamak values (assumed
    value, not a machine setting): ``n_e`` 2e19 to 2e18 m^-3, ``T_e`` 250 to
    10 eV, ``T_i`` 150 to 10 eV; ``core_alpha = 1``, ``core_beta = 1.5``
    (assumed value).

    Convention
    ----------
    Every position and profile is in normalized poloidal flux ``psi_norm``.

    Applicability
    -------------
    Machine-independent.  The baseline against which the barrier presets are
    compared on one fixed geometry.

    Limitations
    -----------
    A prescribed shape, not a confinement prediction, and not Grad-Shafranov
    consistent with any geometry.

    Provenance
    ----------
    .. [1045] VAFT issue #1045, canonical confinement-regime presets.
    .. [552] :func:`vaft.formula.equilibrium.generalized_parabolic_profile` (#552).
    """
    values = dict(ne_axis=ne_axis, ne_sep=ne_sep, te_axis=te_axis, te_sep=te_sep,
                  ti_axis=ti_axis, ti_sep=ti_sep)
    return _preset("L-mode", values, pedestal=False, itb=False, core_alpha=core_alpha,
                   core_beta=core_beta, z_eff=z_eff, impurity_charge=impurity_charge, psi_norm=psi_norm)


def analytic_hmode_state(
    *,
    ne_axis: float = _DEFAULTS["ne_axis"],
    ne_ped: float = _DEFAULTS["ne_ped"],
    ne_sep: float = _DEFAULTS["ne_sep"],
    te_axis: float = _DEFAULTS["te_axis"],
    te_ped: float = _DEFAULTS["te_ped"],
    te_sep: float = _DEFAULTS["te_sep"],
    ti_axis: float = _DEFAULTS["ti_axis"],
    ti_ped: float = _DEFAULTS["ti_ped"],
    ti_sep: float = _DEFAULTS["ti_sep"],
    pedestal_position=0.93,
    pedestal_width=0.05,
    core_alpha=1.0,
    core_beta=1.5,
    z_eff: float = 1.0,
    impurity_charge: float = 6.0,
    psi_norm=None,
) -> AnalyticPlasmaState:
    r"""H-mode-like kinetic state: a smooth core on an edge-localized pedestal.

    Parameters
    ----------
    ne_axis : float, optional
        Electron density on axis [m^-3].
    ne_ped : float, optional
        Electron density at the pedestal knee [m^-3].
    ne_sep : float, optional
        Electron density at the separatrix [m^-3].
    te_axis : float, optional
        Electron temperature on axis [eV].
    te_ped : float, optional
        Electron temperature at the pedestal knee [eV].
    te_sep : float, optional
        Electron temperature at the separatrix [eV].
    ti_axis : float, optional
        Ion temperature on axis [eV].
    ti_ped : float, optional
        Ion temperature at the pedestal knee [eV].
    ti_sep : float, optional
        Ion temperature at the separatrix [eV].
    pedestal_position : float or mapping, optional
        Centre of the pedestal layer in ``psi_norm``, shared or keyed by
        ``n_e``/``T_e``/``T_i`` [-].
    pedestal_width : float or mapping, optional
        Full pedestal width in ``psi_norm``, shared or keyed by channel [-].
    core_alpha : float or mapping, optional
        Core radial exponent [-].
    core_beta : float or mapping, optional
        Core peaking exponent [-].
    z_eff : float, optional
        Uniform effective charge [-].
    impurity_charge : float, optional
        Charge of the single impurity [-].
    psi_norm : array_like, optional
        Sampling grid in ``[0, 1]`` [-].

    Returns
    -------
    AnalyticPlasmaState
        Labelled ``"H-mode"``; axis, knee and separatrix values hold exactly [-].

    Raises
    ------
    ValueError
        As for :func:`compose_analytic_profile` and :func:`compose_plasma_state`.

    Defaults
    --------
    Illustrative values (assumed value, not a machine setting): pedestal tops
    ``n_e`` 1.2e19 m^-3, ``T_e`` 80 eV, ``T_i`` 60 eV, the axis and separatrix
    values of :func:`analytic_lmode_state`, centre 0.93 and full width 0.05 in
    ``psi_norm``.

    Convention
    ----------
    The pedestal top is the knee, ``pedestal_position - pedestal_width/2``,
    in the full-width Groebner-Carlstrom convention; ``pedestal_position`` is
    the centre of the steep layer, not the top.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The pedestal is prescribed: no EPED width or height, no peeling-ballooning
    limit, and no bootstrap current (#550/#559).  The state is not
    Grad-Shafranov consistent with any geometry.

    Provenance
    ----------
    .. [1045] VAFT issue #1045, canonical confinement-regime presets.
    .. [GC98] R. J. Groebner and T. N. Carlstrom, Plasma Phys. Control. Fusion
       40, 673 (1998), through :func:`vaft.formula.equilibrium.modified_tanh_profile`.
    """
    values = dict(ne_axis=ne_axis, ne_ped=ne_ped, ne_sep=ne_sep, te_axis=te_axis, te_ped=te_ped,
                  te_sep=te_sep, ti_axis=ti_axis, ti_ped=ti_ped, ti_sep=ti_sep)
    return _preset("H-mode", values, pedestal=True, itb=False, core_alpha=core_alpha,
                   core_beta=core_beta, z_eff=z_eff, impurity_charge=impurity_charge, psi_norm=psi_norm,
                   pedestal_position=pedestal_position, pedestal_width=pedestal_width)


def analytic_itb_state(
    *,
    ne_axis: float = _DEFAULTS["ne_axis"],
    ne_sep: float = _DEFAULTS["ne_sep"],
    te_axis: float = _DEFAULTS["te_axis"],
    te_sep: float = _DEFAULTS["te_sep"],
    ti_axis: float = _DEFAULTS["ti_axis"],
    ti_sep: float = _DEFAULTS["ti_sep"],
    ne_itb_height: float = _DEFAULTS["ne_itb_height"],
    te_itb_height: float = _DEFAULTS["te_itb_height"],
    ti_itb_height: float = _DEFAULTS["ti_itb_height"],
    itb_position=0.3,
    itb_width=0.08,
    core_alpha=1.0,
    core_beta=1.5,
    z_eff: float = 1.0,
    impurity_charge: float = 6.0,
    psi_norm=None,
) -> AnalyticPlasmaState:
    r"""ITB-like kinetic state: a smooth profile with an internal transport barrier per channel.

    Parameters
    ----------
    ne_axis : float, optional
        Electron density on axis [m^-3].
    ne_sep : float, optional
        Electron density at the separatrix [m^-3].
    te_axis : float, optional
        Electron temperature on axis [eV].
    te_sep : float, optional
        Electron temperature at the separatrix [eV].
    ti_axis : float, optional
        Ion temperature on axis [eV].
    ti_sep : float, optional
        Ion temperature at the separatrix [eV].
    ne_itb_height : float, optional
        Density amplitude of the ITB step, axis side to separatrix side; zero switches the ITB off in ``n_e`` [m^-3].
    te_itb_height : float, optional
        Electron temperature amplitude of the ITB step, axis side to separatrix side; zero for none [eV].
    ti_itb_height : float, optional
        Ion temperature amplitude of the ITB step, axis side to separatrix side; zero for none [eV].
    itb_position : float or mapping, optional
        ITB centre in ``psi_norm``, shared or keyed by ``n_e``/``T_e``/``T_i``,
        so each channel can have its own [-].
    itb_width : float or mapping, optional
        Full ITB width in ``psi_norm``, shared or keyed by channel [-].
    core_alpha : float or mapping, optional
        Core radial exponent [-].
    core_beta : float or mapping, optional
        Core peaking exponent [-].
    z_eff : float, optional
        Uniform effective charge [-].
    impurity_charge : float, optional
        Charge of the single impurity [-].
    psi_norm : array_like, optional
        Sampling grid in ``[0, 1]`` [-].

    Returns
    -------
    AnalyticPlasmaState
        Labelled ``"ITB"``; axis and separatrix values hold exactly [-].

    Raises
    ------
    ValueError
        As for :func:`compose_analytic_profile` and :func:`compose_plasma_state`.

    Defaults
    --------
    Illustrative values (assumed value, not a machine setting): step amplitudes of
    4e18 m^-3, 120 eV and 60 eV at ``psi_norm = 0.3`` with full width 0.08,
    on the axis and separatrix values of :func:`analytic_lmode_state`.

    Convention
    ----------
    Positions and widths in normalized poloidal flux, full-width convention.
    The axis value is held, so the barrier steepens the profile at its centre
    and flattens it elsewhere; raise the axis value to raise the core.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Nothing about ITB formation (magnetic shear, E x B shear) is modelled, and
    the state is not Grad-Shafranov consistent with any geometry.

    Provenance
    ----------
    .. [1045] VAFT issue #1045: independent ITB behaviour per channel.
    .. [GC98] R. J. Groebner and T. N. Carlstrom, Plasma Phys. Control. Fusion
       40, 673 (1998), for the tanh step reused as an interior barrier.
    """
    values = dict(ne_axis=ne_axis, ne_sep=ne_sep, te_axis=te_axis, te_sep=te_sep, ti_axis=ti_axis,
                  ti_sep=ti_sep, ne_itb_height=ne_itb_height, te_itb_height=te_itb_height,
                  ti_itb_height=ti_itb_height)
    return _preset("ITB", values, pedestal=False, itb=True, core_alpha=core_alpha, core_beta=core_beta,
                   z_eff=z_eff, impurity_charge=impurity_charge, psi_norm=psi_norm,
                   itb_position=itb_position, itb_width=itb_width)


def analytic_hmode_itb_state(
    *,
    ne_axis: float = _DEFAULTS["ne_axis"],
    ne_ped: float = _DEFAULTS["ne_ped"],
    ne_sep: float = _DEFAULTS["ne_sep"],
    te_axis: float = _DEFAULTS["te_axis"],
    te_ped: float = _DEFAULTS["te_ped"],
    te_sep: float = _DEFAULTS["te_sep"],
    ti_axis: float = _DEFAULTS["ti_axis"],
    ti_ped: float = _DEFAULTS["ti_ped"],
    ti_sep: float = _DEFAULTS["ti_sep"],
    ne_itb_height: float = _DEFAULTS["ne_itb_height"],
    te_itb_height: float = _DEFAULTS["te_itb_height"],
    ti_itb_height: float = _DEFAULTS["ti_itb_height"],
    pedestal_position=0.93,
    pedestal_width=0.05,
    itb_position=0.3,
    itb_width=0.08,
    core_alpha=1.0,
    core_beta=1.5,
    z_eff: float = 1.0,
    impurity_charge: float = 6.0,
    psi_norm=None,
) -> AnalyticPlasmaState:
    r"""H-mode + ITB kinetic state: smooth core, internal barrier and edge pedestal composed.

    Parameters
    ----------
    ne_axis : float, optional
        Electron density on axis [m^-3].
    ne_ped : float, optional
        Electron density at the pedestal knee [m^-3].
    ne_sep : float, optional
        Electron density at the separatrix [m^-3].
    te_axis : float, optional
        Electron temperature on axis [eV].
    te_ped : float, optional
        Electron temperature at the pedestal knee [eV].
    te_sep : float, optional
        Electron temperature at the separatrix [eV].
    ti_axis : float, optional
        Ion temperature on axis [eV].
    ti_ped : float, optional
        Ion temperature at the pedestal knee [eV].
    ti_sep : float, optional
        Ion temperature at the separatrix [eV].
    ne_itb_height : float, optional
        Density amplitude of the ITB step, axis side to separatrix side [m^-3].
    te_itb_height : float, optional
        Electron temperature amplitude of the ITB step, axis side to separatrix side [eV].
    ti_itb_height : float, optional
        Ion temperature amplitude of the ITB step, axis side to separatrix side [eV].
    pedestal_position : float or mapping, optional
        Pedestal centre in ``psi_norm``, shared or per channel [-].
    pedestal_width : float or mapping, optional
        Full pedestal width in ``psi_norm``, shared or per channel [-].
    itb_position : float or mapping, optional
        ITB centre in ``psi_norm``, shared or per channel [-].
    itb_width : float or mapping, optional
        Full ITB width in ``psi_norm``, shared or per channel [-].
    core_alpha : float or mapping, optional
        Core radial exponent [-].
    core_beta : float or mapping, optional
        Core peaking exponent [-].
    z_eff : float, optional
        Uniform effective charge [-].
    impurity_charge : float, optional
        Charge of the single impurity [-].
    psi_norm : array_like, optional
        Sampling grid in ``[0, 1]`` [-].

    Returns
    -------
    AnalyticPlasmaState
        Labelled ``"H-mode + ITB"``; axis, pedestal-knee and separatrix values
        hold exactly [-].

    Raises
    ------
    ValueError
        As for :func:`compose_analytic_profile` and :func:`compose_plasma_state`.

    Defaults
    --------
    The illustrative values (assumed value) of :func:`analytic_hmode_state`
    and :func:`analytic_itb_state` together.

    Convention
    ----------
    Normalized poloidal flux, full-width barrier convention; the pedestal top
    is the knee.  The same composition as the other presets, with both steps.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    As for :func:`analytic_hmode_state` and :func:`analytic_itb_state`.

    Provenance
    ----------
    .. [1045] VAFT issue #1045: composition of an internal barrier and an edge
       pedestal in one state.
    """
    values = dict(ne_axis=ne_axis, ne_ped=ne_ped, ne_sep=ne_sep, te_axis=te_axis, te_ped=te_ped,
                  te_sep=te_sep, ti_axis=ti_axis, ti_ped=ti_ped, ti_sep=ti_sep,
                  ne_itb_height=ne_itb_height, te_itb_height=te_itb_height, ti_itb_height=ti_itb_height)
    return _preset("H-mode + ITB", values, pedestal=True, itb=True, core_alpha=core_alpha,
                   core_beta=core_beta, z_eff=z_eff, impurity_charge=impurity_charge, psi_norm=psi_norm,
                   pedestal_position=pedestal_position, pedestal_width=pedestal_width,
                   itb_position=itb_position, itb_width=itb_width)


# --- projection onto (R, Z) -----------------------------------------------------


def _inside_boundary(equilibrium, psi_n: np.ndarray) -> np.ndarray:
    # psi_n < 0 next to the axis is a stored psi_axis slightly shallower than the
    # grid's own extremum (a coarse or interpolated axis, a rescaled file).  It is
    # the plasma centre, not the outside, so only the boundary side is thresholded.
    inside = np.isfinite(psi_n) & (psi_n <= 1.0)
    lcfs = getattr(equilibrium, "lcfs", None)
    if lcfs is None or np.asarray(lcfs.r).size < 3:
        return inside & (psi_n >= -_AXIS_TOLERANCE)
    from matplotlib.path import Path

    rr, zz = np.meshgrid(equilibrium.r, equilibrium.z, indexing="ij")
    points = np.column_stack([rr.ravel(), zz.ravel()])
    outline = Path(np.column_stack([lcfs.r, lcfs.z]))
    # A hair of tolerance either way keeps grid points lying on the outline itself.
    contained = outline.contains_points(points, radius=1e-9) | outline.contains_points(points, radius=-1e-9)
    return inside & contained.reshape(psi_n.shape)


def project_flux_function(equilibrium, function: Callable, *, outside: str = "nan") -> np.ndarray:
    r"""Map a flux function ``f(psi_norm)`` onto an equilibrium's ``(R, Z)`` grid.

    Parameters
    ----------
    equilibrium : EquilibriumData
        Geometry with ``r``, ``z``, ``psi`` indexed ``(R, Z)``, ``psi_axis``,
        ``psi_boundary`` and, preferably, ``lcfs`` [-].
    function : callable
        ``f(psi_norm)`` accepting an array in ``[0, 1]`` [-].
    outside : str, optional
        ``"nan"`` to leave points outside the plasma as NaN, ``"edge"`` to
        fill them with ``f(1)``, the separatrix value [-].

    Returns
    -------
    np.ndarray
        ``f`` on the grid, shape ``(r.size, z.size)``, in the function's unit [any].

    Raises
    ------
    ValueError
        No flux map, equal axis and boundary flux, or an unknown *outside*.

    Convention
    ----------
    ``psi_norm = (psi - psi_axis)/(psi_boundary - psi_axis)``
    (:func:`vaft.formula.equilibrium.psi_normalised`), which is independent of
    the COCOS sign and of Wb against Wb/rad because both cancel in the ratio;
    no conversion is applied or needed.  A point is inside when
    ``psi_norm <= 1`` *and* it lies within the ``lcfs`` outline, which
    excludes the private-flux region below an X-point and any exterior region
    where the flux turns back below one.  Inside points with ``psi_norm < 0``
    -- a stored ``psi_axis`` slightly shallower than the grid's own extremum --
    are the plasma centre and are evaluated at ``psi_norm = 0``; without an
    outline, ``psi_norm >= -0.05`` stands in for containment on that side.

    Applicability
    -------------
    Machine-independent.  Any equilibrium record with a gridded flux map: an
    analytic Solov'ev or Guazzotto-Freidberg geometry, a GEQDSK, an ODS slice
    through :func:`vaft.process.equilibrium.as_equilibrium`.

    Limitations
    -----------
    Without an ``lcfs`` outline only the flux threshold is used, which can
    admit exterior points where the flux turns over near coils.  A flux
    function is constant on flux surfaces by construction; the projection
    adds no poloidal variation.

    Provenance
    ----------
    .. [1045] VAFT issue #1045: the explicit ``f(psi_N) -> f(R, Z)`` mapping.
    """
    if outside not in ("nan", "edge"):
        raise ValueError(f"outside must be 'nan' or 'edge', got {outside!r}")
    if getattr(equilibrium, "psi", None) is None or equilibrium.r is None or equilibrium.z is None:
        raise ValueError("the equilibrium carries no gridded flux map")
    psi_axis = float(equilibrium.psi_axis)
    psi_boundary = float(equilibrium.psi_boundary)
    if not np.isfinite(psi_axis) or not np.isfinite(psi_boundary) or psi_axis == psi_boundary:
        raise ValueError("psi_axis and psi_boundary must be finite and distinct")
    from vaft.formula.equilibrium import psi_normalised

    psi_n = np.asarray(psi_normalised(np.asarray(equilibrium.psi, dtype=float), psi_axis, psi_boundary))
    inside = _inside_boundary(equilibrium, psi_n)
    values = np.asarray(function(np.clip(np.where(inside, psi_n, 1.0), 0.0, 1.0)), dtype=float)
    if outside == "nan":
        return np.where(inside, values, np.nan)
    return np.where(inside, values, float(np.asarray(function(np.array([1.0])), dtype=float)[0]))


def project_plasma_state(state: AnalyticPlasmaState, equilibrium, quantity: str = "p_total", *,
                         outside: str = "nan") -> np.ndarray:
    r"""Project one quantity of an analytic plasma state onto an equilibrium's ``(R, Z)`` grid.

    Parameters
    ----------
    state : AnalyticPlasmaState
        From :func:`compose_plasma_state` or a preset [-].
    equilibrium : EquilibriumData
        The geometry to project onto; it is read, never changed [-].
    quantity : str, optional
        Any quantity :func:`evaluate_plasma_state` accepts [-].
    outside : str, optional
        ``"nan"`` or ``"edge"``, as in :func:`project_flux_function` [-].

    Returns
    -------
    np.ndarray
        The quantity on the grid, shape ``(r.size, z.size)``, in the unit of
        :data:`vaft.data.ANALYTIC_STATE_UNITS` [any].

    Raises
    ------
    ValueError
        As for :func:`project_flux_function` and :func:`evaluate_plasma_state`.

    Convention
    ----------
    Evaluated exactly from the state's analytic definition at each point's
    ``psi_norm``, not interpolated from the state's sampling grid.  The
    profile keeps its 1-D definition whatever geometry it is projected onto.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The projected pressure is *not* the equilibrium's own pressure: the state
    was prescribed, and the geometry was solved for different sources.  Making
    them agree is a new equilibrium solve (CHEASE refinement, #123).

    Provenance
    ----------
    .. [1045] VAFT issue #1045: 2-D projection of a prescribed plasma state.
    """
    if quantity not in _STATE_QUANTITIES:
        raise ValueError(f"quantity must be one of {_STATE_QUANTITIES}, got {quantity!r}")
    return project_flux_function(
        equilibrium, lambda x: evaluate_plasma_state(state, x, quantity), outside=outside
    )
