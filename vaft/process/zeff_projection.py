"""Resistive projection of an effective-charge profile: Z_eff(rho) -> R_p -> Z_eff^res,equiv.

The composition model (#1565) gives a local profile ``Z_eff(rho)``; the
Romero transformer balance (#1214, Lane Z) gives one scalar ``Z_eff^res``,
the constant a conductivity model needs to reproduce the observed plasma
resistance.  They are different quantities and are compared only through an
explicit forward projection (issue #1566): the scalar that makes the *same*
conductivity model, on the *same* flux-surface states, produce the *same*
resistance -- or, over a window, the same resistive voltage under the *same*
objective -- as the profile does.

The chain::

    Z_eff(psi_N) on a FluxSurfaceState      [composition, #1565]
    -> sigma_par(psi_N; Z_eff(psi_N))       [Lane Z's model, surface by surface]
    -> R_p^profile = P_Ohm / I_p^2          [vaft.process.resistive_zeff.model_resistance]
    -> Z* with R_p^model(Z*) = R_p^profile  [one slice: a bounded root]
    -> or argmin_Z sum w (V_R^profile - V_R^model(Z))^2 over a window
                                            [vaft.process.resistive_zeff.infer_resistive_zeff]

Nothing here writes ``core_profiles.zeff``: the projection is a model-space
scalar, never a profile.  A flat profile built from a resistive scalar needs
the explicit assumption of :func:`flat_profile_from_resistive`.

Notation
--------
Z_eff(psi_N)      : effective-charge profile on the state's surfaces         [-]
R_p               : plasma resistance, P_Ohm / I_p^2                        [Ohm]
Z_eff^res,equiv   : constant Z_eff giving the profile's R_p (or V_R(t))     [-]
Z_eff^res,obs     : Lane Z's scalar inferred from the observed V_R          [-]

Conventions
-----------
**Same model, same states, same objective.**  The profile's conductivity is
Lane Z's :func:`~vaft.process.resistive_zeff.parallel_conductivity`
evaluated surface by surface at that surface's Z_eff, so the profile and the
scalar differ only in Z_eff; the window scalar is Lane Z's own
:func:`~vaft.process.resistive_zeff.infer_resistive_zeff` fed the profile's
predicted voltage, so the projection is defined in the observable space the
experimental scalar lives in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = [
    "ResistiveZeffProjection",
    "flat_profile_from_resistive",
    "profile_conductivity_model",
    "project_window_to_resistive_scalar",
    "project_zeff_profile_to_resistive_scalar",
    "spitzer_resistive_equivalent_zeff",
    "zeff_profile_for_state",
]


@dataclass(frozen=True)
class ResistiveZeffProjection:
    """The resistively equivalent scalar of a Z_eff profile, and how it was obtained."""

    zeff_equivalent: float
    model: str
    profile_source: str
    resistance_profile: np.ndarray
    resistance_flat: np.ndarray
    time_window: tuple[float, float]
    bounds: tuple[float, float]
    convergence: Mapping[str, Any]
    profile_volume_mean: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "zeff_res_equiv": self.zeff_equivalent, "model": self.model,
            "profile_source": self.profile_source,
            "t_start_s": self.time_window[0], "t_end_s": self.time_window[1],
            "zeff_profile_volume_mean": self.profile_volume_mean,
            "r_p_profile_ohm_mean": float(np.mean(self.resistance_profile)),
            "status": self.convergence.get("status"),
        }


def _profile(state, zeff) -> np.ndarray:
    z = np.asarray(zeff, dtype=float)
    if z.ndim == 0:
        z = np.full(np.shape(state.psi_norm), float(z))
    if z.shape != np.shape(state.psi_norm):
        raise ValueError(f"the Z_eff profile has {z.shape}, the state's surfaces {np.shape(state.psi_norm)}")
    if not np.all(np.isfinite(z)) or np.any(z < 1.0):
        raise ValueError("a Z_eff profile must be finite and at least 1 on every surface")
    return z


def _volume_mean(state, z) -> float:
    from scipy.integrate import trapezoid

    volume = np.asarray(state.volume, dtype=float)
    return float(trapezoid(z, volume) / (volume[-1] - volume[0]))


def profile_conductivity_model(
    zeff_profile: Any,
    *,
    base_model: str = "redl",
    ln_lambda: Union[float, str] = "sauter",
) -> Callable:
    """A conductivity-model callable that evaluates Lane Z's model at a radially varying Z_eff.

    Parameters
    ----------
    zeff_profile : array-like
        Z_eff on the state's ``psi_norm`` surfaces, at least 1 [-].
    base_model : str, optional
        One of :data:`vaft.process.resistive_zeff.CONDUCTIVITY_MODELS` [-].
    ln_lambda : float or str, optional
        Coulomb logarithm as :func:`~vaft.process.resistive_zeff.parallel_conductivity`
        takes it [-].

    Returns
    -------
    callable
        ``model(state, z, lnl) -> sigma(psi_N)`` for
        :func:`~vaft.process.resistive_zeff.model_resistance`; the scalar
        ``z`` it is handed is ignored [any].

    Raises
    ------
    ValueError
        At call time: a profile that does not match the state's surfaces, or
        that is non-finite or below 1.

    Convention
    ----------
    Surface ``i`` gets ``parallel_conductivity(state, Z_eff[i])[i]``: the
    unmodified scalar model, so a flat profile reproduces it exactly and no
    kernel is copied.  ``model_resistance`` records ``z_eff = 1`` for this
    route; the profile is the record of what was used.  The ``ln_lambda`` it
    is built with must be the one ``model_resistance`` is called with; a
    mismatch is refused, not silently overridden.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1566 Sec. 3 (the Redl/Sauter projection through Lane Z's
               conductivity models), #1214 (the models), #1486 (Lane Z).
    """
    from vaft.process.resistive_zeff import _ln_lambda_profile, parallel_conductivity

    profile = np.asarray(zeff_profile, dtype=float)

    def model(state, _z, lnl):
        z = _profile(state, profile)
        own, _ = _ln_lambda_profile(state, ln_lambda)
        if not np.allclose(np.asarray(lnl, dtype=float), own, rtol=1e-12, atol=0.0):
            raise ValueError(
                "model_resistance was given a different ln_lambda from the one this profile model was "
                f"built with ({ln_lambda!r}); build it with the same ln_lambda"
            )
        sigma = np.empty(z.shape)
        for value in np.unique(z):
            at = z == value
            sigma[at] = parallel_conductivity(state, float(value), model=base_model, ln_lambda=ln_lambda)[at]
        return sigma

    model.zeff_profile = profile
    model.base_model = base_model
    model.__name__ = f"profile_{base_model}"
    return model


def spitzer_resistive_equivalent_zeff(state: Any, zeff_profile: Any, *, ln_lambda: Union[float, str] = "sauter") -> float:
    """Analytic resistive equivalent of a Z_eff profile under Spitzer (NRL) resistivity.

    Parameters
    ----------
    state : FluxSurfaceState
        One slice (``bootstrap = none``) [any].
    zeff_profile : array-like
        Z_eff on the state's ``psi_norm`` surfaces [-].
    ln_lambda : float or str, optional
        Coulomb logarithm [-].

    Returns
    -------
    float
        ``int Z w dV / int w dV`` with ``w = eta(Z=1) <J.B>^2/<B^2>``, the
        Ohmic-dissipation weight [-].

    Raises
    ------
    ValueError
        A profile off the state's surfaces, a state carrying a bootstrap
        current (the weight is then not linear in Z), or a zero weight.

    Convention
    ----------
    NRL Spitzer resistivity is linear in Z_eff, so ``P_Ohm`` is a weighted
    integral of Z_eff and the equivalent scalar is that weighted mean -- the
    weight is the dissipation ``eta_1 j^2``, not the volume or the density.
    Integrated with the same trapezoid rule on the enclosed volume as
    :func:`~vaft.process.resistive_zeff.model_resistance`.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1566 Sec. 2 (the Spitzer reference projection).
    """
    from scipy.integrate import trapezoid

    from vaft.process.resistive_zeff import parallel_conductivity

    if getattr(state, "j_bootstrap_dot_b", None) is not None:
        raise ValueError("the analytic Spitzer projection needs a state without bootstrap current")
    z = _profile(state, zeff_profile)
    sigma_1 = parallel_conductivity(state, 1.0, model="spitzer_nrl", ln_lambda=ln_lambda)
    jb = np.asarray(state.j_dot_b, dtype=float)
    weight = jb**2 / np.asarray(state.b2_average, dtype=float) / sigma_1
    volume = np.asarray(state.volume, dtype=float)
    denominator = trapezoid(weight, volume)
    if not denominator > 0.0:
        raise ValueError("the dissipation weight integrates to zero")
    return float(trapezoid(weight * z, volume) / denominator)


def project_zeff_profile_to_resistive_scalar(
    state: Any,
    zeff_profile: Any,
    *,
    model: str = "redl",
    ln_lambda: Union[float, str] = "sauter",
    bounds: tuple[float, float] = (1.0, 10.0),
    profile_source: str = "caller",
) -> ResistiveZeffProjection:
    """The constant Z_eff whose model resistance equals that of a Z_eff profile, on one slice.

    Parameters
    ----------
    state : FluxSurfaceState
        One slice [any].
    zeff_profile : array-like
        Z_eff on the state's ``psi_norm`` surfaces [-].
    model : str, optional
        Conductivity model, as Lane Z names them [-].
    ln_lambda : float or str, optional
        Coulomb logarithm [-].
    bounds : tuple of float, optional
        Search interval for the scalar [-].
    profile_source : str, optional
        Where the profile came from, for provenance [-].

    Returns
    -------
    ResistiveZeffProjection
        The scalar, both resistances, the bracket and status [any].

    Raises
    ------
    ValueError
        A profile off the state's surfaces, or no root inside ``bounds``
        (the profile's resistance is outside what a constant Z_eff in
        ``bounds`` can give).

    Processing steps
    ----------------
    1. ``R_p^profile`` from ``model_resistance`` with
       :func:`profile_conductivity_model`.
    2. A bounded root of ``R_p^model(Z) - R_p^profile`` (``brentq``).

    Convention
    ----------
    The projection is defined in resistance space with the same states and
    model on both sides (#1566 Sec. 1); a flat profile returns its own value.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1566 Sec. 1 and 3.
    """
    from scipy.optimize import brentq

    from vaft.process.resistive_zeff import model_resistance

    z = _profile(state, zeff_profile)
    closure = profile_conductivity_model(z, base_model=model, ln_lambda=ln_lambda)
    r_profile = model_resistance(state, 1.0, model=closure, ln_lambda=ln_lambda).R_p

    def mismatch(value):
        return model_resistance(state, value, model=model, ln_lambda=ln_lambda).R_p - r_profile

    lo, hi = float(bounds[0]), float(bounds[1])
    f_lo, f_hi = mismatch(lo), mismatch(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(f"no constant Z_eff in [{lo:g}, {hi:g}] reproduces the profile's resistance")
    root = brentq(mismatch, lo, hi, xtol=1e-10)
    return ResistiveZeffProjection(
        zeff_equivalent=float(root), model=model, profile_source=profile_source,
        resistance_profile=np.array([r_profile]),
        resistance_flat=np.array([model_resistance(state, root, model=model, ln_lambda=ln_lambda).R_p]),
        time_window=(float(state.time), float(state.time)), bounds=(lo, hi),
        convergence={"status": "ok", "method": "brentq"}, profile_volume_mean=_volume_mean(state, z),
        provenance={"ln_lambda": str(ln_lambda), "objective": "R_p(Z) = R_p(Z_eff(psi_N)), one slice"},
    )


def project_window_to_resistive_scalar(
    states: Sequence[Any],
    zeff_profiles: Sequence[Any],
    *,
    model: str = "redl",
    ln_lambda: Union[float, str] = "sauter",
    bounds: tuple[float, float] = (1.0, 10.0),
    weights: Union[str, Sequence[float]] = "uniform",
    profile_source: str = "caller",
) -> ResistiveZeffProjection:
    """The window scalar Lane Z's estimator would infer if the profile were the truth.

    Parameters
    ----------
    states : sequence of FluxSurfaceState
        The window's slices [any].
    zeff_profiles : sequence of array-like
        One Z_eff profile per state, on its surfaces [-].
    model : str, optional
        Conductivity model [-].
    ln_lambda : float or str, optional
        Coulomb logarithm [-].
    bounds : tuple of float, optional
        Search interval [-].
    weights : str or sequence of float, optional
        As :func:`~vaft.process.resistive_zeff.infer_resistive_zeff` [-].
    profile_source : str, optional
        Where the profiles came from [-].

    Returns
    -------
    ResistiveZeffProjection
        The scalar, the per-slice resistances of the profile and of the
        scalar, the window and Lane Z's status [any].

    Raises
    ------
    ValueError
        Unequal numbers of states and profiles, two states at one time, or a
        profile off its state's surfaces.

    Processing steps
    ----------------
    1. Per slice, ``R_p^profile(t)`` as in
       :func:`project_zeff_profile_to_resistive_scalar`.
    2. An ``ObservedResistance`` whose ``V_R = R_p^profile (I_p - I_ni)``
       with ``I_ni = 0`` -- the profile's voltage, as if observed.
    3. Lane Z's ``infer_resistive_zeff`` on it: the same objective
       ``sum w (V_R - R_p^model(Z) I_p)^2`` and bounds as the experimental scalar.

    Convention
    ----------
    #1566 Sec. 7: a spatio-temporal model-equivalent scalar, comparable with
    ``Z_eff^res,obs`` because the objective is literally the same function.
    The drive is ``|I_p|``: ``R_p`` is a dissipation and does not depend on
    the current's sign convention.  Near Z_eff = 1 the estimator reports
    ``bound_hit`` (a ~0.3 % bias at the lower bound), exactly as it does for
    the observed scalar.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1566 Sec. 7, #1214 Sec. 10.
    """
    from vaft.process.resistive_zeff import ObservedResistance, infer_resistive_zeff, model_resistance

    states = list(states)
    profiles = list(zeff_profiles)
    if len(states) != len(profiles) or not states:
        raise ValueError("one Z_eff profile per state, and at least one state")
    r_profile = np.array([
        model_resistance(s, 1.0, model=profile_conductivity_model(_profile(s, z), base_model=model,
                                                                    ln_lambda=ln_lambda),
                         ln_lambda=ln_lambda).R_p
        for s, z in zip(states, profiles)
    ])
    time = np.array([float(s.time) for s in states])
    if np.unique(time).size != time.size:
        raise ValueError("two states share a time; Lane Z's estimator would silently use only the first")
    # the current's sign is a COCOS choice; the dissipation is not -- drive with |I_p|
    ip = np.abs(np.array([float(s.I_p) for s in states]))
    n = time.size
    nan = np.full(n, np.nan)
    observed = ObservedResistance(
        time=time, I_p=ip, I_ni=np.zeros(n), L_i=nan, li_3=nan, dI_p_dt=nan, dL_i_dt=nan,
        V_B=nan, V_I=nan, V_R=r_profile * ip, R_p=r_profile, inductive_fraction=np.zeros(n),
        flags=tuple(() for _ in range(n)),
        provenance={"current_source_assumption": "I_ni = 0 (projection of a Z_eff profile)",
                    "source": "vaft.process.zeff_projection"},
    )
    inference = infer_resistive_zeff(observed, states, model=model, ln_lambda=ln_lambda,
                                     bounds=bounds, weights=weights)
    zeff = inference.zeff
    flat = (np.array([model_resistance(s, zeff, model=model, ln_lambda=ln_lambda).R_p for s in states])
            if zeff is not None else np.full(n, np.nan))
    return ResistiveZeffProjection(
        zeff_equivalent=float("nan") if zeff is None else float(zeff), model=model,
        profile_source=profile_source, resistance_profile=r_profile, resistance_flat=flat,
        time_window=(float(time.min()), float(time.max())), bounds=tuple(map(float, bounds)),
        convergence={"status": inference.status, "reason": inference.reason},
        profile_volume_mean=float(np.mean([_volume_mean(s, _profile(s, z)) for s, z in zip(states, profiles)])),
        provenance={"ln_lambda": str(ln_lambda), "weights": weights if isinstance(weights, str) else "array",
                    "objective": "Lane Z infer_resistive_zeff on the profile's V_R"},
    )


def flat_profile_from_resistive(zeff_resistive: float, state: Any, *, assumption: str) -> np.ndarray:
    """A flat Z_eff profile from a resistive scalar -- only under the stated flat assumption.

    Parameters
    ----------
    zeff_resistive : float
        A resistive scalar (Lane Z), at least 1 [-].
    state : FluxSurfaceState
        The surfaces to put it on [any].
    assumption : str
        Must be ``"flat"``: the caller states the modelling assumption [-].

    Returns
    -------
    np.ndarray
        ``Z_eff(psi_N) = zeff_resistive`` on every surface [-].

    Raises
    ------
    ValueError
        Any assumption other than ``"flat"``, or a scalar below 1.

    Convention
    ----------
    #1566 Sec. 6: a modelling assumption, never a measurement or an
    inference of the profile; nothing here writes it to ``core_profiles``.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1566 Sec. 6.
    """
    if assumption != "flat":
        raise ValueError("a profile from a resistive scalar needs assumption='flat', stated explicitly")
    value = float(zeff_resistive)
    if not value >= 1.0:
        raise ValueError(f"zeff_resistive must be at least 1, got {zeff_resistive!r}")
    return np.full(np.shape(state.psi_norm), value)


def zeff_profile_for_state(state: Any, elemental_weights: Mapping[str, float], **options: Any):
    """The atomic-data Z_eff profile on a flux-surface state's own surfaces.

    Parameters
    ----------
    state : FluxSurfaceState
        One slice; its ``T_e``, ``n_e`` and enclosed volume are used [any].
    elemental_weights : mapping
        Relative elemental densities, e.g. ``{"C": 1, "O": 1}``; further
        keyword arguments go to
        :func:`vaft.process.impurity.resolve_radial_composition`
        (``normalization``, ``target_zeff``, ``ionization``, ``plasma_age_s``,
        ``projection``/``resistive_target``, ``tables``...) [-].

    Returns
    -------
    RadialImpurityComposition
        On ``psi_norm``, with ``dV`` from the state's volume [any].

    Raises
    ------
    ValueError
        As ``resolve_radial_composition``.

    Convention
    ----------
    The profile is built from the very T_e and n_e the conductivity model
    uses, so composition and resistance share one state.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1566 Sec. 5 (connect #1565 profiles to #1214).
    """
    from vaft.process.impurity import resolve_radial_composition

    volume = np.asarray(state.volume, dtype=float)
    edges = np.concatenate(([volume[0]], 0.5 * (volume[1:] + volume[:-1]), [volume[-1]]))
    return resolve_radial_composition(state.T_e, state.n_e, state.psi_norm, elemental_weights,
                                      volume_weights=np.diff(edges), time=float(state.time), **options)
