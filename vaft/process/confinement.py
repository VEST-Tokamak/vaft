"""Power balance and slice qualification for energy-confinement scaling (vaft #548).

The first layer of the #548 hierarchy: from time series of plasma current,
resistive loop voltage and stored energy to the loss power and confinement
time of each time slice, and the evidence that decides whether a slice enters
a scaling fit. It composes the definitions of :mod:`vaft.formula.equilibrium`
and adds no physics of its own.

Two loss powers are kept apart, because databases do not agree on one:

=================  ===========================================  =====================
name               definition                                   compares with
=================  ===========================================  =====================
``p_net``          $P_\\mathrm{OH} - dW/dt$                       ITPA DB5 ``PLTH``
``p_transport``    $P_\\mathrm{OH} - dW/dt - P_\\mathrm{rad}$      transport analyses
=================  ===========================================  =====================

``p_transport`` is NaN wherever the radiated power was not measured; a
modelled $P_\\mathrm{rad}$ is a sensitivity study, never the default.

Slice qualification keeps the *evidence* (rates and ratios per slice) apart
from the *decision* (thresholds), as #253 asks: thresholds are arguments,
never constants copied from another database, and
:func:`confinement_exclusion_table` reports how many slices each rule
removes so a threshold sweep can be read off directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from vaft.formula.equilibrium import (
    confinement_time_from_P_loss_W_th,
    loss_power_from_p_heat_dWdt_p_rad,
    ohmic_heating_power_from_I_p_V_res,
)
from vaft.formula.constants import MU0
from vaft.process.numerical import time_derivative

__all__ = [
    "ConfinementPowerBalance",
    "ConfinementSliceEvidence",
    "smoothed_time_derivative",
    "ResistiveLoopVoltage",
    "resistive_loop_voltage",
    "confinement_power_balance",
    "confinement_slice_evidence",
    "confinement_slice_decision",
    "confinement_exclusion_table",
    "ConfinementScalingFit",
    "fit_confinement_scaling",
    "ObservedConfinement",
    "resolve_observed_confinement",
    "assess_predictor_identifiability",
    "predictor_principal_directions",
    "bootstrap_confinement_scaling",
    "leave_one_group_out_scaling",
    "fit_confinement_scaling_odr",
    "infer_kadomtsev_size_exponent",
    "dimensionless_confinement_indices",
    "closure_constraint",
    "fit_constrained_confinement_scaling",
]

def _series(name: str, value, size: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if size is not None and arr.size != size:
        raise ValueError(f"{name} has {arr.size} samples, expected {size}")
    return arr


def smoothed_time_derivative(
    time: np.ndarray,
    data: np.ndarray,
    *,
    window_s: Optional[float] = None,
    polyorder: int = 2,
    at: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Time derivative of a sampled quantity by a local polynomial fit, on any time grid.

    Parameters
    ----------
    time : numpy.ndarray
        Sample times, strictly increasing, 1-D; gaps are allowed [s].
    data : numpy.ndarray
        The sampled quantity, same length as ``time`` [any].
    window_s : float, optional
        Full width of the fitting window centred on each evaluation time;
        ``None`` takes the unsmoothed weighted central difference of
        :func:`vaft.process.numerical.time_derivative` at the samples [s].
    polyorder : int, optional
        Degree of the local polynomial [-].
    at : numpy.ndarray, optional
        Times to evaluate the derivative at; default the samples themselves.
        Needs ``window_s`` [s].

    Returns
    -------
    numpy.ndarray
        ``d(data)/dt`` at ``at`` (or at each sample); NaN where the window
        holds no more than ``polyorder`` finite samples [any/s].

    Raises
    ------
    ValueError
        Arrays of different length, fewer than two samples, a non-increasing
        time axis, a non-positive window, ``polyorder`` below 1 with a window,
        or ``at`` without ``window_s``.

    Processing steps
    ----------------
    1. Without ``window_s``: the weighted central difference at the samples
       (one-sided at the ends).
    2. With ``window_s``: at each evaluation time $t_0$, the finite samples
       with $|t - t_0| \\le$ ``window_s``/2 are fitted by a degree-
       ``polyorder`` polynomial in $t - t_0$ (least squares, unweighted);
       the derivative is its linear coefficient. On a uniform grid away
       from the ends this is the Savitzky-Golay first derivative; at the
       ends and across gaps the window is simply one-sided or sparser.

    Defaults
    --------
    ``polyorder = 2``: numerical convenience, the lowest degree whose
    derivative is unbiased for a quantity with constant curvature.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Smoothing biases a derivative that changes within the window; the window
    width belongs in the provenance of every quantity built on it. Near the
    ends the window is one-sided and the derivative noisier.

    Provenance
    ----------
    .. [SG] A. Savitzky and M. J. E. Golay, *Anal. Chem.* **36** (1964) 1627:
       the local least-squares polynomial derivative.
    """
    t = _series("time", time)
    y = _series("data", data, t.size)
    if t.size < 2:
        raise ValueError("a derivative needs at least two samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    if window_s is None:
        if at is not None:
            raise ValueError("evaluating away from the samples needs window_s")
        return np.asarray(time_derivative(t, y), dtype=float)
    if int(polyorder) < 1:
        raise ValueError(f"polyorder must be at least 1 to carry a slope, got {polyorder!r}")
    half = 0.5 * float(window_s)
    if not half > 0:
        raise ValueError(f"window_s must be positive, got {window_s!r}")
    targets = t if at is None else np.atleast_1d(np.asarray(at, dtype=float))
    good = np.isfinite(y)
    out = np.full(targets.shape, np.nan)
    lo = np.searchsorted(t, targets - half, side="left")
    hi = np.searchsorted(t, targets + half, side="right")
    for k, (t0, i, j) in enumerate(zip(targets, lo, hi)):
        sel = slice(i, j)
        keep = good[sel]
        if np.count_nonzero(keep) <= polyorder:
            continue
        dt = t[sel][keep] - t0
        coef = np.polynomial.polynomial.polyfit(dt, y[sel][keep], polyorder)
        out[k] = coef[1]
    return out


@dataclass(frozen=True)
class ResistiveLoopVoltage:
    """Loop voltage split into its internal-inductive and resistive parts.

    The single home of $V_R$ and $R_p$ in VAFT (#548 Lane D, shared with the
    resistive $Z_\\mathrm{eff}$ inference of #1214): ``v_ind`` and ``v_res``
    [V], ``r_p = v_res / i_p`` [Ohm], ``l_int`` [H], one value per sample.
    """

    v_ind: np.ndarray
    v_res: np.ndarray
    r_p: np.ndarray
    l_int: np.ndarray


def resistive_loop_voltage(
    v_loop: np.ndarray,
    i_p: np.ndarray,
    dip_dt: np.ndarray,
    li_3: np.ndarray,
    r0: float,
    *,
    dli3_dt: Optional[np.ndarray] = None,
) -> "ResistiveLoopVoltage":
    """Inductive and resistive parts of a loop voltage, and the plasma resistance, from the internal inductance.

    Parameters
    ----------
    v_loop : numpy.ndarray
        Loop voltage, in the orientation of ``i_p`` [V].
    i_p : numpy.ndarray
        Plasma current [A].
    dip_dt : numpy.ndarray
        Its time derivative [A/s].
    li_3 : numpy.ndarray
        Normalised internal inductance $l_{i,3}$ [-].
    r0 : float
        Major radius normalising $l_{i,3}$ [m].
    dli3_dt : numpy.ndarray, optional
        Time derivative of $l_{i,3}$; ``None`` holds the inductance fixed, so
        only $L_i\\,dI_p/dt$ is removed [1/s].

    Returns
    -------
    ResistiveLoopVoltage
        ``v_ind``, the inductive voltage $(1/I_p)\\,dW_\\mathrm{int}/dt$, and
        ``v_res = v_loop - v_ind`` [V]; ``r_p = v_res / i_p``, the plasma
        resistance seen by the total current (NaN where $I_p = 0$) [Ohm];
        ``l_int``, the internal inductance $\\mu_0 R_0 l_{i,3}/2$ [H] [any].

    Raises
    ------
    ValueError
        Arrays of different length or a non-positive ``r0``.

    Processing steps
    ----------------
    1. $L_i = \\mu_0 R_0 l_{i,3}/2$, so $W_\\mathrm{int} = L_i I_p^2/2$.
    2. $V_\\mathrm{ind} = (1/I_p)\\,dW_\\mathrm{int}/dt
       = L_i\\,dI_p/dt + \\tfrac14\\mu_0 R_0 I_p\\,dl_{i,3}/dt$, the second
       term only when ``dli3_dt`` is given.
    3. $V_\\mathrm{res} = V_\\mathrm{loop} - V_\\mathrm{ind}$ and
       $R_p = V_\\mathrm{res}/I_p$.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Exact (Romero's eq. 24) only for the loop voltage *at the plasma
    boundary*. A flux loop away from the boundary also sees the external
    inductive flux between loop and boundary, which this does not remove.

    Provenance
    ----------
    .. [Romero] J. A. Romero et al., *Nucl. Fusion* **50** (2010) 115002,
       eq. 24: the inductive voltage of the internal poloidal-field energy.
    .. [548] Issue #548 Sec. 1: P_OH from the loop voltage with the inductive
       part removed, the Spitzer path only for comparison (#1188).
    """
    v = _series("v_loop", v_loop)
    n = v.size
    ip = _series("i_p", i_p, n)
    didt = _series("dip_dt", dip_dt, n)
    li = _series("li_3", li_3, n)
    if not float(r0) > 0:
        raise ValueError(f"r0 must be positive, got {r0!r}")
    l_int = 0.5 * MU0 * float(r0) * li
    v_ind = l_int * didt
    if dli3_dt is not None:
        v_ind = v_ind + 0.25 * MU0 * float(r0) * ip * _series("dli3_dt", dli3_dt, n)
    v_res = v - v_ind
    with np.errstate(divide="ignore", invalid="ignore"):
        r_p = np.where(ip != 0.0, v_res / ip, np.nan)
    return ResistiveLoopVoltage(v_ind=v_ind, v_res=v_res, r_p=r_p, l_int=l_int)


@dataclass(frozen=True)
class ConfinementPowerBalance:
    """Ohmic power balance on one time base, with both loss-power definitions.

    Every array has one value per sample of ``time``. ``p_transport`` and
    ``tau_e_transport`` are NaN where no measured radiated power was given.
    """

    time: np.ndarray
    p_ohmic: np.ndarray
    dwdt: np.ndarray
    p_net: np.ndarray
    p_transport: np.ndarray
    tau_e_net: np.ndarray
    tau_e_transport: np.ndarray


def confinement_power_balance(
    time: np.ndarray,
    i_p: np.ndarray,
    v_res: np.ndarray,
    w_th: np.ndarray,
    *,
    p_rad: Optional[np.ndarray] = None,
    dwdt_window_s: Optional[float] = None,
    dwdt_polyorder: int = 2,
) -> ConfinementPowerBalance:
    """Ohmic heating, stored-energy change, loss powers and confinement times of a discharge.

    Parameters
    ----------
    time : numpy.ndarray
        Sample times, strictly increasing, 1-D [s].
    i_p : numpy.ndarray
        Plasma current, in the orientation of ``v_res`` [A].
    v_res : numpy.ndarray
        Resistive loop voltage, the loop voltage with the inductive part
        removed [V].
    w_th : numpy.ndarray
        Stored (thermal) energy whose change and confinement are wanted [J].
    p_rad : numpy.ndarray, optional
        Measured radiated power; ``None`` (or NaN samples) leaves the
        transport loss power undefined there [W].
    dwdt_window_s : float, optional
        Window of the local-polynomial $dW/dt$ derivative, see
        :func:`smoothed_time_derivative` [s].
    dwdt_polyorder : int, optional
        Polynomial order of that derivative [-].

    Returns
    -------
    ConfinementPowerBalance
        ``p_ohmic``, ``dwdt``, ``p_net`` and ``p_transport`` [W];
        ``tau_e_net`` and ``tau_e_transport`` [s]; ``time`` [s] [any].

    Raises
    ------
    ValueError
        Arrays of different length, or what :func:`smoothed_time_derivative`
        rejects.

    Processing steps
    ----------------
    1. $P_\\mathrm{OH} = I_p V_\\mathrm{res}$
       (:func:`vaft.formula.equilibrium.ohmic_heating_power_from_I_p_V_res`).
    2. $dW/dt$ by :func:`smoothed_time_derivative` on ``w_th``.
    3. $P_\\mathrm{net} = P_\\mathrm{OH} - dW/dt$ and, where ``p_rad`` is
       finite, $P_\\mathrm{transport} = P_\\mathrm{net} - P_\\mathrm{rad}$
       (:func:`vaft.formula.equilibrium.loss_power_from_p_heat_dWdt_p_rad`).
    4. $\\tau_E = W/P$ for each loss power, NaN where that power is not
       positive.

    Applicability
    -------------
    Machine-independent. Ohmic plasmas: no auxiliary heating term.

    Limitations
    -----------
    ``v_res`` carries every assumption of how the inductive voltage was
    removed; with a loop voltage taken away from the plasma boundary the
    flux between loop and boundary is still inductive. The transport loss
    power needs a *measured* radiated power, which VEST does not map
    (no bolometer), so on VEST it is NaN.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 1: P_net and P_transport as separate, named
       quantities; W = (3/2) int p dV fixed by #1282/#1410.
    .. [ITER99] ITER Physics Expert Groups, *Nucl. Fusion* **39** (1999) 2175,
       Ch. 2: the loss power $P_L = P_\\mathrm{heat} - dW/dt$ of the
       confinement database.
    """
    t = _series("time", time)
    ip = _series("i_p", i_p, t.size)
    vr = _series("v_res", v_res, t.size)
    w = _series("w_th", w_th, t.size)
    rad = np.full(t.size, np.nan) if p_rad is None else _series("p_rad", p_rad, t.size)

    p_ohmic = np.asarray(ohmic_heating_power_from_I_p_V_res(ip, vr), dtype=float)
    dwdt = smoothed_time_derivative(t, w, window_s=dwdt_window_s, polyorder=dwdt_polyorder)
    p_net = np.asarray(loss_power_from_p_heat_dWdt_p_rad(p_ohmic, dwdt, 0.0), dtype=float)
    p_transport = np.asarray(loss_power_from_p_heat_dWdt_p_rad(p_ohmic, dwdt, rad), dtype=float)

    def _tau(power):
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.asarray(confinement_time_from_P_loss_W_th(power, w), dtype=float)
        return np.where(np.isfinite(power) & (power > 0), tau, np.nan)

    return ConfinementPowerBalance(
        time=t, p_ohmic=p_ohmic, dwdt=dwdt, p_net=p_net, p_transport=p_transport,
        tau_e_net=_tau(p_net), tau_e_transport=_tau(p_transport),
    )


@dataclass(frozen=True)
class ConfinementSliceEvidence:
    """Per-slice measurements a confinement slice is judged on; no thresholds.

    ``ip_rate`` is $|dI_p/dt|/|I_p|$ [1/s]; ``ip_change_per_tau`` is
    ``ip_rate * tau_e`` [-], the fractional current change within one
    confinement time; ``dwdt_fraction`` is $|dW/dt|/P_\\mathrm{OH}$ [-];
    ``finite`` says W, P and $\\tau_E$ are finite and positive.
    """

    ip_abs: np.ndarray
    ip_rate: np.ndarray
    ip_change_per_tau: np.ndarray
    dwdt_fraction: np.ndarray
    finite: np.ndarray


def confinement_slice_evidence(
    i_p: np.ndarray,
    dip_dt: np.ndarray,
    w_th: np.ndarray,
    p_ohmic: np.ndarray,
    dwdt: np.ndarray,
    tau_e: np.ndarray,
) -> ConfinementSliceEvidence:
    """Rates and ratios that decide whether a time slice is quasi-stationary enough to fit.

    Parameters
    ----------
    i_p : numpy.ndarray
        Plasma current at the slices [A].
    dip_dt : numpy.ndarray
        Its time derivative at the slices [A/s].
    w_th : numpy.ndarray
        Stored energy at the slices [J].
    p_ohmic : numpy.ndarray
        Ohmic heating power at the slices [W].
    dwdt : numpy.ndarray
        Stored-energy change at the slices [W].
    tau_e : numpy.ndarray
        Confinement time at the slices [s].

    Returns
    -------
    ConfinementSliceEvidence
        ``ip_abs`` [A], ``ip_rate`` [1/s], ``ip_change_per_tau`` and
        ``dwdt_fraction`` [-], ``finite`` [bool] [any].

    Raises
    ------
    ValueError
        Arrays of different length.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 5: the slice-quality evidence, kept apart from
       the acceptance decision as #253 asks.
    """
    ip = _series("i_p", i_p)
    n = ip.size
    didt = _series("dip_dt", dip_dt, n)
    w = _series("w_th", w_th, n)
    p = _series("p_ohmic", p_ohmic, n)
    dw = _series("dwdt", dwdt, n)
    tau = _series("tau_e", tau_e, n)
    ip_abs = np.abs(ip)
    with np.errstate(divide="ignore", invalid="ignore"):
        ip_rate = np.abs(didt) / ip_abs
        dw_frac = np.abs(dw) / p
    dw_frac = np.where(p > 0, dw_frac, np.nan)
    finite = (
        np.isfinite(w) & (w > 0) & np.isfinite(p) & (p > 0) & np.isfinite(tau) & (tau > 0)
    )
    return ConfinementSliceEvidence(
        ip_abs=ip_abs, ip_rate=ip_rate, ip_change_per_tau=ip_rate * tau,
        dwdt_fraction=dw_frac, finite=finite,
    )


def confinement_slice_decision(
    evidence: ConfinementSliceEvidence,
    *,
    ip_min: float,
    max_dwdt_fraction: float,
    max_ip_change_per_tau: float,
) -> dict[str, np.ndarray]:
    """Pass/fail of each slice-quality rule, and their conjunction, for given thresholds.

    Parameters
    ----------
    evidence : ConfinementSliceEvidence
        Output of :func:`confinement_slice_evidence` [any].
    ip_min : float
        Smallest accepted $|I_p|$ [A].
    max_dwdt_fraction : float
        Largest accepted $|dW/dt|/P_\\mathrm{OH}$ [-].
    max_ip_change_per_tau : float
        Largest accepted fractional current change within one confinement
        time, $\\tau_E |dI_p/dt|/|I_p|$ [-].

    Returns
    -------
    dict
        Boolean arrays keyed ``finite``, ``ip_min``, ``dwdt_fraction``,
        ``ip_change_per_tau`` and ``accepted`` (all rules); a NaN evidence
        value fails its rule [bool].

    Applicability
    -------------
    Machine-independent. The thresholds are the caller's: none is a
    database or machine default.

    Limitations
    -----------
    Ramp phases are judged by rate against the confinement time, not by a
    phase label: a short discharge with no flat top (VEST) has no plateau to
    select, and a rate rule says how far from stationary each slice is.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 5: thresholds as arguments, studied by sweep
       before any is adopted.
    """
    ev = evidence
    rules = {
        "finite": np.asarray(ev.finite, dtype=bool),
        "ip_min": np.nan_to_num(ev.ip_abs, nan=-np.inf) >= float(ip_min),
        "dwdt_fraction": np.nan_to_num(ev.dwdt_fraction, nan=np.inf) <= float(max_dwdt_fraction),
        "ip_change_per_tau": (
            np.nan_to_num(ev.ip_change_per_tau, nan=np.inf) <= float(max_ip_change_per_tau)
        ),
    }
    accepted = np.ones_like(rules["finite"])
    for passed in rules.values():
        accepted = accepted & passed
    rules["accepted"] = accepted
    return rules


def confinement_exclusion_table(rules: Mapping[str, np.ndarray]) -> list[dict]:
    """How many slices each quality rule removes, alone and in sequence.

    Parameters
    ----------
    rules : Mapping
        Boolean pass arrays keyed by rule name, in the order to apply them,
        as :func:`confinement_slice_decision` returns; an ``accepted`` key
        is ignored [bool].

    Returns
    -------
    list of dict
        One row per rule: ``rule``, ``failed`` (slices failing it),
        ``failed_only_this`` (failing this rule and no other),
        ``removed_in_sequence`` (newly removed when the rules are applied in
        order) and ``remaining`` (after it) [-].

    Raises
    ------
    ValueError
        Pass arrays of different length.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 5: the rules and the number each excludes are
       part of the result.
    """
    names = [name for name in rules if name != "accepted"]
    arrays = [np.asarray(rules[name], dtype=bool) for name in names]
    if len({a.size for a in arrays}) > 1:
        raise ValueError("every rule must judge the same slices")
    if not arrays:
        return []
    passing = np.ones(arrays[0].size, dtype=bool)
    stack = np.vstack(arrays)
    table = []
    for k, (name, passed) in enumerate(zip(names, arrays)):
        others = np.delete(stack, k, axis=0)
        others_pass = others.all(axis=0) if others.size else np.ones_like(passed)
        removed = passing & ~passed
        passing = passing & passed
        table.append({
            "rule": name,
            "failed": int(np.sum(~passed)),
            "failed_only_this": int(np.sum(~passed & others_pass)),
            "removed_in_sequence": int(np.sum(removed)),
            "remaining": int(np.sum(passing)),
        })
    return table


# --- Regression and identifiability (#548 Sec. 3, 4, 6) -------------------------


def _log_design(response, predictors: Mapping[str, np.ndarray], groups):
    """Log response, log design with intercept, groups and the kept-row mask."""
    names = list(predictors)
    if not names:
        raise ValueError("at least one predictor is needed")
    y = _series("response", response)
    columns = [_series(name, predictors[name], y.size) for name in names]
    g = np.asarray(groups)
    if g.shape != y.shape:
        raise ValueError(f"groups has shape {g.shape}, expected {y.shape}")
    stack = np.vstack([y, *columns])
    keep = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.log(y[keep])
        x = np.column_stack([np.ones(keep.sum())] + [np.log(c[keep]) for c in columns])
    return names, log_y, x, g[keep], keep


#: Relative spread of a log predictor below which it counts as constant.
_SPREAD_RTOL = 1e-9


def _constant_columns(names, logs) -> list:
    """Names of log-predictor columns (rows of ``logs``) that do not vary."""
    out = []
    for name, col in zip(names, logs):
        scale = max(1.0, float(np.max(np.abs(col)))) if col.size else 1.0
        if col.size == 0 or float(np.ptp(col)) <= _SPREAD_RTOL * scale:
            out.append(name)
    return out


@dataclass(frozen=True)
class ConfinementScalingFit:
    """A log-linear confinement scaling fit with group-aware uncertainty.

    ``names`` starts with ``"log_C"`` (the intercept) followed by the
    predictors; ``coef``, ``stderr`` and ``ci95`` follow that order.
    ``stderr``/``cov``/``ci95`` are cluster-robust by group (CR1, t with
    G - 1 degrees of freedom); ``stderr_iid`` is the same estimator's error
    treating every row as independent (OLS, or the Huber fit's own), kept to
    show how much the grouping matters. ``mask`` marks the input rows the fit used.
    """

    names: tuple
    coef: np.ndarray
    stderr: np.ndarray
    cov: np.ndarray
    ci95: np.ndarray
    stderr_iid: np.ndarray
    n: int
    n_groups: int
    r2: float
    rmse_log: float
    residuals: np.ndarray
    leverage: np.ndarray
    cooks_distance: np.ndarray
    mask: np.ndarray
    robust: bool

    def exponents(self) -> dict:
        """Predictor exponents keyed by name, without the intercept."""
        return dict(zip(self.names[1:], self.coef[1:]))


def fit_confinement_scaling(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
    groups: np.ndarray,
    *,
    robust: bool = False,
) -> ConfinementScalingFit:
    """Log-linear power-law fit $y = C \\prod_j x_j^{\\alpha_j}$ with errors clustered by group.

    Parameters
    ----------
    response : numpy.ndarray
        The fitted quantity, usually $\\tau_E$ or $W$ [any].
    predictors : Mapping
        Predictor arrays keyed by name, same length as ``response`` [any].
    groups : numpy.ndarray
        Group label per row, normally the shot: rows of one group are not
        independent samples [-].
    robust : bool, optional
        Huber M-estimation instead of least squares for the coefficients
        [bool].

    Returns
    -------
    ConfinementScalingFit
        Coefficients in log space (exponents, and $\\ln C$ first) [-], their
        cluster-robust and naive errors, 95 % intervals, $R^2$ and the RMS
        log residual [-], leverage and Cook's distance per kept row [-] [any].

    Raises
    ------
    ValueError
        No predictor, mismatched lengths, fewer kept rows than coefficients
        plus one, fewer than two groups, a predictor that does not vary over
        the kept rows, or an exactly collinear design: a least-squares solver
        would otherwise return an arbitrary exponent with a confident error.

    Processing steps
    ----------------
    1. Keep rows where the response and every predictor are finite and
       positive; take logs and prepend an intercept column.
    2. Fit by ordinary least squares (or Huber IRLS with ``robust``).
    3. Covariance clustered by ``groups`` (statsmodels ``cov_type="cluster"``,
       CR1 small-sample correction); 95 % intervals from Student's t with
       $G - 1$ degrees of freedom.
    4. Leverage and Cook's distance from the least-squares influence of the
       same design (also for a robust fit, as a diagnostic).

    With ``robust`` the cluster covariance is that of a weighted fit with the
    converged Huber weights held fixed, which slightly understates the error
    when rows are down-weighted; ``stderr_iid`` is then the Huber fit's own.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    With few groups the cluster-robust error is itself noisy (it has
    $G - 1$ degrees of freedom); read it together with the group bootstrap
    and leave-one-group-out results. Errors in the predictors are ignored,
    so exponents are attenuated where a predictor is noisy; with $\\tau_E =
    W/P$ and $P$ a predictor, the $P$ exponent is biased towards $-1$ by any
    error in $P$.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 6: covariance, intervals, influence and robust
       regression with grouped samples.
    .. [CGM] A. C. Cameron and D. L. Miller, *J. Human Resources* **50**
       (2015) 317: cluster-robust inference and the few-clusters problem.
    """
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.outliers_influence import OLSInfluence

    names, log_y, x, g, keep = _log_design(response, predictors, groups)
    n, k = x.shape
    n_groups = int(np.unique(g).size)
    if n <= k:
        raise ValueError(f"{n} usable rows cannot fit {k} coefficients")
    if n_groups < 2:
        raise ValueError("cluster-robust errors need at least two groups")
    constant = _constant_columns(names, x[:, 1:].T)
    if constant:
        raise ValueError(f"predictor(s) {constant} do not vary over the kept rows; "
                         "their exponents are not identifiable")
    if np.linalg.matrix_rank(x) < k:
        raise ValueError("the log design is rank deficient: some predictors are exactly "
                         "collinear and their exponents are not separately identifiable")
    codes = np.unique(g, return_inverse=True)[1]

    ols = sm.OLS(log_y, x).fit()
    clustered = sm.OLS(log_y, x).fit(cov_type="cluster", cov_kwds={"groups": codes})
    if robust:
        # Huber IRLS, then a WLS refit with its converged weights so the
        # cluster covariance applies to the robust coefficients.
        rlm = sm.RLM(log_y, x, M=sm.robust.norms.HuberT()).fit()
        clustered = sm.WLS(log_y, x, weights=rlm.weights).fit(
            cov_type="cluster", cov_kwds={"groups": codes})
        coef = np.asarray(clustered.params, dtype=float)
        stderr_iid = np.asarray(rlm.bse, dtype=float)
    else:
        stderr_iid = np.asarray(ols.bse, dtype=float)
    if not robust:
        coef = np.asarray(ols.params, dtype=float)
    cov = np.asarray(clustered.cov_params(), dtype=float)
    stderr = np.sqrt(np.diag(cov))
    t = stats.t.ppf(0.975, n_groups - 1)
    ci95 = np.column_stack([coef - t * stderr, coef + t * stderr])
    residuals = log_y - x @ coef
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    influence = OLSInfluence(ols)
    return ConfinementScalingFit(
        names=("log_C", *names), coef=coef, stderr=stderr, cov=cov, ci95=ci95,
        stderr_iid=stderr_iid, n=n, n_groups=n_groups,
        r2=1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else np.nan,
        rmse_log=float(np.sqrt(np.mean(residuals**2))), residuals=residuals,
        leverage=np.asarray(influence.hat_matrix_diag, dtype=float),
        cooks_distance=np.asarray(influence.cooks_distance[0], dtype=float),
        mask=keep, robust=bool(robust),
    )


@dataclass(frozen=True)
class ObservedConfinement:
    """The observed confinement time a scaling is compared with, and how it was obtained.

    ``tau`` [s] per row (NaN where unavailable); ``energy_basis_required`` is the
    scaling's; ``energy_basis_used`` per row (``"thermal"``, ``"global"`` or
    ``""`` where NaN); ``approximation`` names the assumption when a thermal
    time stood in for a global one, else ``None``; ``reason`` says why rows are
    NaN, else ``None``.
    """

    tau: np.ndarray
    energy_basis_required: str
    energy_basis_used: np.ndarray
    approximation: Optional[str]
    reason: Optional[str]


#: The assumption recorded when a thermal confinement time stands in for a global one.
THERMAL_AS_GLOBAL = "W_global ~ W_th: negligible non-thermal (fast-ion) stored energy"


def resolve_observed_confinement(
    tau_thermal,
    energy_basis: str,
    *,
    tau_global=None,
    thermal_as_global: bool = False,
) -> ObservedConfinement:
    """Observed confinement time with the energy basis a scaling was fitted on (issue #1713).

    Parameters
    ----------
    tau_thermal : array-like
        Thermal confinement time $W_{th}/P$ per row, NaN where unknown [s].
    energy_basis : str
        The scaling's basis, from
        :func:`vaft.formula.equilibrium.confinement_scaling_basis`:
        ``"thermal"``, ``"global"`` or ``"unaudited"`` [str].
    tau_global : array-like, optional
        Global confinement time $W/P$ per row, fast ions included; default
        none observed [s].
    thermal_as_global : bool, optional
        Use the thermal time where a global (or unaudited) basis has no global
        observation, recording :data:`THERMAL_AS_GLOBAL`; default False [-].

    Returns
    -------
    ObservedConfinement
        The resolved times and their provenance [any].

    Raises
    ------
    ValueError
        An unknown ``energy_basis``, or arrays of different lengths.

    Processing steps
    ----------------
    1. A thermal scaling takes ``tau_thermal``; nothing else stands in for it.
    2. A global scaling takes ``tau_global`` where it is finite. Elsewhere it
       is NaN (strict), or the thermal time with :data:`THERMAL_AS_GLOBAL`
       recorded when ``thermal_as_global`` is set.
    3. An unaudited scaling has no known basis: NaN unless
       ``thermal_as_global``, when the thermal time is used and recorded.

    Applicability
    -------------
    Machine-independent. The thermal-for-global substitution is defensible for
    ohmic plasmas without fast ions (VEST Tier A) and wrong for beam- or
    RF-heated plasmas with a fast-ion population.

    Limitations
    -----------
    Resolves which time to use; it does not construct $W_{global}$ or
    $W_{th}$ from a reconstruction. ``w_mhd_J`` from a magnetics EFIT is total
    kinetic pressure and is not by itself a thermal energy.

    Provenance
    ----------
    .. [1713] Issue #1713: thermal and global energy confinement must not be
       silently interchanged in scaling comparisons.
    """
    if energy_basis not in ("thermal", "global", "unaudited"):
        raise ValueError(f"energy_basis must be thermal, global or unaudited; got {energy_basis!r}")
    th = np.asarray(tau_thermal, dtype=float)
    gl = np.full(th.shape, np.nan) if tau_global is None else np.asarray(tau_global, dtype=float)
    if gl.shape != th.shape:
        raise ValueError(f"tau_global has shape {gl.shape}, tau_thermal {th.shape}")
    used = np.full(th.shape, "", dtype=object)
    if energy_basis == "thermal":
        ok = np.isfinite(th)
        used[ok] = "thermal"
        return ObservedConfinement(np.where(ok, th, np.nan), energy_basis, used, None,
                                   None if ok.all() else "no thermal confinement time on some rows")
    tau = np.full(th.shape, np.nan)
    direct = np.isfinite(gl) if energy_basis == "global" else np.zeros(th.shape, bool)
    tau[direct] = gl[direct]
    used[direct] = "global"
    approximation = None
    if thermal_as_global:
        fill = ~direct & np.isfinite(th)
        tau[fill] = th[fill]
        used[fill] = "thermal"
        if fill.any():
            approximation = THERMAL_AS_GLOBAL + ("" if energy_basis == "global" else "; scaling basis unaudited")
    missing = ~np.isfinite(tau)
    reason = None
    if missing.any():
        reason = ("energy basis of the scaling is unaudited" if energy_basis == "unaudited"
                  else "no global confinement time observed") + (
            "" if thermal_as_global else " (pass thermal_as_global=True to use the thermal time, recorded)")
    return ObservedConfinement(tau, energy_basis, used, approximation, reason)


def assess_predictor_identifiability(predictors: Mapping[str, np.ndarray]) -> dict:
    """Collinearity of log predictors: correlation, VIF, condition number and rank.

    Parameters
    ----------
    predictors : Mapping
        Predictor arrays keyed by name; rows with any non-finite or
        non-positive value are dropped [any].

    Returns
    -------
    dict
        ``n`` rows used [-]; ``correlation``, the Pearson matrix of the log
        predictors as nested dicts [-]; ``vif``, the variance inflation factor
        per predictor [-]; ``condition_number`` of the column-equilibrated log
        design with intercept [-]; ``rank`` of that design [-]; ``constant``,
        the predictors that do not vary (VIF infinite, correlation NaN) [-]
        [any].

    Raises
    ------
    ValueError
        No predictor, or mismatched lengths.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Says whether the sample *can* separate the exponents, not whether a fit
    did; a predictor that barely varies has a large VIF only if it is also
    correlated with another, so read the spread of each log predictor too.
    The condition number follows Belsley (columns scaled to unit length,
    intercept kept); values above about 30 signal harmful collinearity.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 3 and 6: B_T stays a fitted variable only if its
       covariance with I_p and P is quantified.
    .. [BKW] D. A. Belsley, E. Kuh and R. E. Welsch, *Regression Diagnostics*,
       Wiley (1980), Ch. 3.
    """
    names = list(predictors)
    if not names:
        raise ValueError("at least one predictor is needed")
    first = _series(names[0], predictors[names[0]])
    columns = [_series(name, predictors[name], first.size) for name in names]
    stack = np.vstack(columns)
    keep = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    logs = np.log(stack[:, keep])
    n = int(keep.sum())
    constant = _constant_columns(names, logs)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.atleast_2d(np.corrcoef(logs) if len(names) > 1 else np.ones((1, 1)))
    for j, name in enumerate(names):
        if name in constant:
            corr[j, :] = corr[:, j] = np.nan
    vif = {}
    for j, name in enumerate(names):
        others = np.delete(logs, j, axis=0)
        if name in constant:
            vif[name] = float("inf")
            continue
        if others.size == 0:
            vif[name] = 1.0
            continue
        design = np.column_stack([np.ones(n), others.T])
        beta, *_ = np.linalg.lstsq(design, logs[j], rcond=None)
        resid = logs[j] - design @ beta
        ss_tot = float(np.sum((logs[j] - logs[j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 1.0
        vif[name] = float(1.0 / (1.0 - r2)) if r2 < 1.0 - 1e-12 else float("inf")
    design = np.column_stack([np.ones(n), logs.T])
    scaled = design / np.linalg.norm(design, axis=0)
    return {
        "n": n,
        "correlation": {a: {b: float(corr[i, j]) for j, b in enumerate(names)}
                        for i, a in enumerate(names)},
        "vif": vif,
        "condition_number": float(np.linalg.cond(scaled)),
        "rank": int(np.linalg.matrix_rank(design)),
        "constant": constant,
    }


def predictor_principal_directions(predictors: Mapping[str, np.ndarray], *, groups=None) -> dict:
    """Principal directions of the log predictors: which exponent combinations the data constrain.

    Parameters
    ----------
    predictors : Mapping
        Predictor arrays keyed by name; rows with any non-finite or
        non-positive value are dropped [any].
    groups : array-like, optional
        Group label per row (e.g. shot); when given, the analysis is repeated
        on the shot means (between-group) and on the within-group deviations
        [any].

    Returns
    -------
    dict
        ``names`` [-]; ``n`` rows used [-]; ``scale``, the standard deviation
        of each log predictor [-]; ``singular_values`` of the centred,
        standardised log design, largest first [-]; ``directions``, the unit
        right-singular vectors as rows, one per singular value, in that
        design's coordinates [-]; ``variance_fraction`` per direction [-];
        ``effective_rank``, the number of directions whose singular value is
        at least ``0.1`` of the largest [-]; and, with ``groups``, ``between``
        and ``within``: the spread of the group means and of the within-group
        deviations *along the same directions*, in the same standardised
        units, as root-sum-square projections over the rows, so that
        ``between**2 + within**2`` is each singular value squared [-] [any].

    Raises
    ------
    ValueError
        No predictor, mismatched lengths (``groups`` included), fewer than
        three usable rows, or a predictor that does not vary.

    Processing steps
    ----------------
    1. Take logs, centre and divide each column by its standard deviation.
    2. Singular-value decompose; a direction with a small singular value is a
       combination of predictors the sample barely varies, so the matching
       combination of exponents is poorly determined (PCR diagnostic). With
       fewer rows than predictors the missing directions are returned with
       singular value zero: they are the unconstrained ones.
    3. With ``groups``, split each direction's spread into the part carried by
       the group means (each row taking its group's mean) and the part carried
       by the within-group deviations, both with the pooled centring and scale.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Directions are in standardised log coordinates; a direction mixes
    exponents in units of each predictor's spread, so convert with ``scale``
    before reading it as a physical scan. It diagnoses the design, not a fit:
    it says which scan would add information, not which exponent is right.

    Provenance
    ----------
    .. [1621] Issue #1621 Secs. 8 and 12: PCR-style identifiability and the
       weak singular directions that point to the next scan.
    .. [BKW] D. A. Belsley, E. Kuh and R. E. Welsch, *Regression Diagnostics*,
       Wiley (1980), Ch. 3.
    """
    names = list(predictors)
    if not names:
        raise ValueError("at least one predictor is needed")
    first = _series(names[0], predictors[names[0]])
    stack = np.vstack([_series(name, predictors[name], first.size) for name in names])
    if groups is not None and np.asarray(groups).shape != (first.size,):
        raise ValueError(f"groups has shape {np.asarray(groups).shape}, expected ({first.size},)")
    keep = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    logs = np.log(stack[:, keep]).T
    if logs.shape[0] < 3:
        raise ValueError("fewer than three usable rows")
    centre = logs.mean(axis=0)
    scale = logs.std(axis=0, ddof=1)
    if np.any(~(scale > 0)):
        raise ValueError("a predictor does not vary: " + ", ".join(n for n, s in zip(names, scale) if not s > 0))
    z = (logs - centre) / scale
    _, sv, vt = np.linalg.svd(z, full_matrices=True)
    sv = np.concatenate([sv, np.zeros(len(names) - sv.size)])  # fewer rows than predictors
    vt = vt * np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])[:, None]
    out = {"names": names, "n": int(logs.shape[0]), "scale": dict(zip(names, map(float, scale))),
           "singular_values": sv.tolist(), "directions": vt.tolist(),
           "variance_fraction": (sv**2 / np.sum(sv**2)).tolist(),
           "effective_rank": int(np.sum(sv >= 0.1 * sv[0]))}
    if groups is not None:
        labels = np.asarray(groups)[keep]
        uniq, inverse = np.unique(labels, return_inverse=True)
        means = np.vstack([z[inverse == k].mean(axis=0) for k in range(uniq.size)])
        within = z - means[inverse]
        # Each row carries its group's mean, so between^2 + within^2 = singular value^2.
        out["between"] = {"n": int(uniq.size),
                          "projected": np.sqrt(((means[inverse] @ vt.T) ** 2).sum(axis=0)).tolist()}
        out["within"] = {"n": int(z.shape[0]), "projected": np.sqrt(((within @ vt.T) ** 2).sum(axis=0)).tolist()}
    return out


def bootstrap_confinement_scaling(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Group (shot) bootstrap of the log-linear scaling coefficients.

    Parameters
    ----------
    response : numpy.ndarray
        The fitted quantity [any].
    predictors : Mapping
        Predictor arrays keyed by name [any].
    groups : numpy.ndarray
        Group label per row; whole groups are resampled [-].
    n_boot : int, optional
        Number of resamples [-].
    seed : int, optional
        Seed of the random generator [-].

    Returns
    -------
    dict
        ``names`` (``log_C`` first) [-]; ``samples``, coefficients of every
        resample whose design had full rank, shape ``(m, k)`` [-];
        ``rejected``, the number of rank-deficient resamples [-];
        ``percentile95``, the 2.5 and 97.5 percentiles per coefficient [-]
        [any].

    Raises
    ------
    ValueError
        As :func:`fit_confinement_scaling` for the full sample.

    Defaults
    --------
    ``n_boot = 2000`` and ``seed = 0``: numerical convenience; 2000 resamples
    put the percentile endpoints within about a percent of their limit.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    With a handful of groups the bootstrap distribution is lumpy and its
    intervals undercover; ``rejected`` counts resamples that could not
    identify every exponent (e.g. a single field value drawn), which is
    itself identifiability evidence.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 6: slices of one shot are not independent, so
       resampling is by shot.
    .. [Efron] B. Efron and R. J. Tibshirani, *An Introduction to the
       Bootstrap*, Chapman & Hall (1993), Ch. 13.
    """
    names, log_y, x, g, _ = _log_design(response, predictors, groups)
    labels, codes = np.unique(g, return_inverse=True)
    if labels.size < 2:
        raise ValueError("a group bootstrap needs at least two groups")
    members = [np.flatnonzero(codes == i) for i in range(labels.size)]
    rng = np.random.default_rng(seed)
    k = x.shape[1]
    samples, rejected = [], 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, labels.size, labels.size)
        rows = np.concatenate([members[i] for i in draw])
        xs = x[rows]
        if rows.size <= k or np.linalg.matrix_rank(xs) < k:
            rejected += 1
            continue
        beta, *_ = np.linalg.lstsq(xs, log_y[rows], rcond=None)
        samples.append(beta)
    samples = np.asarray(samples, dtype=float).reshape(-1, k)
    pct = (np.percentile(samples, [2.5, 97.5], axis=0).T if len(samples)
           else np.full((k, 2), np.nan))
    return {"names": ("log_C", *names), "samples": samples, "rejected": rejected,
            "percentile95": pct}


def leave_one_group_out_scaling(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
    groups: np.ndarray,
) -> dict:
    """Leave-one-group-out refits: coefficient stability and out-of-group prediction error.

    Parameters
    ----------
    response : numpy.ndarray
        The fitted quantity [any].
    predictors : Mapping
        Predictor arrays keyed by name [any].
    groups : numpy.ndarray
        Group label per row; each group is held out in turn [-].

    Returns
    -------
    dict
        ``names`` (``log_C`` first) [-]; ``groups``, the held-out labels [-];
        ``coef``, the refit coefficients per held-out group, NaN where the
        rest could not identify them, shape ``(G, k)`` [-]; ``log_error``, the
        log prediction error of every kept row when its group was held out
        [-]; ``rmse_log``, its RMS over rows [-] [any].

    Raises
    ------
    ValueError
        As :func:`fit_confinement_scaling` for the full sample.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A held-out shot whose predictor values lie outside the others' range
    tests extrapolation, not interpolation; the prediction error of such a
    group dominates ``rmse_log``.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 6 and 8: leave-one-shot-out error as a model
       comparison criterion.
    """
    names, log_y, x, g, _ = _log_design(response, predictors, groups)
    labels, codes = np.unique(g, return_inverse=True)
    k = x.shape[1]
    coef = np.full((labels.size, k), np.nan)
    error = np.full(log_y.size, np.nan)
    for i in range(labels.size):
        held = codes == i
        xs, ys = x[~held], log_y[~held]
        if xs.shape[0] <= k or np.linalg.matrix_rank(xs) < k:
            continue
        beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        coef[i] = beta
        error[held] = log_y[held] - x[held] @ beta
    finite = np.isfinite(error)
    return {
        "names": ("log_C", *names), "groups": labels, "coef": coef, "log_error": error,
        "rmse_log": float(np.sqrt(np.mean(error[finite] ** 2))) if finite.any() else np.nan,
    }


# --- Errors in variables, Kadomtsev completion and closures (#548 Sec. 6-8) ------


def fit_confinement_scaling_odr(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
    *,
    sigma_log_response,
    sigma_log_predictors: Mapping[str, float],
) -> dict:
    """Errors-in-variables (orthogonal distance) power-law fit in log space.

    Parameters
    ----------
    response : numpy.ndarray
        The fitted quantity. For the $\\tau_E = W/P$ coupling, fit $W$, not
        $\\tau_E$, so its error is independent of the power's [any].
    predictors : Mapping
        Predictor arrays keyed by name [any].
    sigma_log_response : float or array-like
        Standard deviation of the *equation* error of $\\ln y$: measurement
        error plus the scaling's intrinsic scatter, not measurement error
        alone. One number for every row, or one per input row (e.g. a
        reconstruction's model-form spread, #579); a row whose value is not
        finite and positive is dropped [-].
    sigma_log_predictors : Mapping
        Standard deviation of the measurement error of $\\ln x_j$ per
        predictor; zero marks a predictor as exact [-].

    Returns
    -------
    dict
        ``names`` (``log_C`` first) [-]; ``coef``, the errors-in-variables
        coefficients [-]; ``n`` rows used [-]; ``mask`` of the rows used
        [bool]; ``ols_coef``, the least-squares coefficients of the same data
        [-]; ``method``, ``"partial_tls"`` for one response error or
        ``"weighted_odr"`` for per-row errors [-] [any].

    Raises
    ------
    ValueError
        No predictor, mismatched lengths, a non-positive response error, a
        negative or non-finite or missing predictor error, no noisy
        predictor, or fewer rows than coefficients plus one.

    Processing steps
    ----------------
    1. Keep rows where everything is finite and positive; take logs.
    2. Partial the exact predictors (zero error) and the intercept out of the
       response and the noisy predictors by least squares.
    3. Scale each residual column by its error and take the right singular
       vector $v$ of the smallest singular value (total least squares); the
       noisy slopes are $b_j = -(v_j/\\sigma_j)/(v_y/\\sigma_y)$.
    4. The exact predictors' slopes and the intercept by least squares of
       $y - \\sum_{j\\in\\mathrm{noisy}} b_j x_j$ on them.
    5. With per-row response errors steps 2-4 have no closed form: the same
       linear model is solved as weighted orthogonal distance regression: the
       maximum-likelihood criterion of a linear model with independent Gaussian
       errors, $\\sum_i (y_i - \\beta\\cdot x_i)^2 / (\\sigma_{y,i}^2 +
       \\sum_j \\beta_j^2\\sigma_{x,j}^2)$, minimised from the least-squares
       start (exact predictors carry $\\sigma = 0$). For a constant per-row
       error it reproduces steps 2-4.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The answer depends only on the assumed error *ratios*, so report it as a
    function of them. Understating the equation error of the response
    over-corrects the attenuation and inflates the noisy slopes, the classic
    pitfall of errors-in-variables confinement fits. Errors are assumed independent between variables.
    Fitting $\\tau_E = W/P$ against $P$ violates that, which is why the
    response should be $W$ (its $P$ exponent is $1 + \\alpha_P$). No standard
    error is returned: rows of one shot are not independent, so use a shot
    bootstrap around this function.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 6: errors-in-variables regression, and the
       coupling from $\\tau_E = W/P$ with $P$ a predictor.
    .. [VHV] S. Van Huffel and J. Vandewalle, *The Total Least Squares
       Problem*, SIAM (1991), Ch. 3: mixed exact/noisy columns (partial TLS);
       for a linear model with known error ratios it is the orthogonal
       distance (maximum-likelihood) solution.
    """
    per_row = np.ndim(sigma_log_response) > 0
    if per_row:
        sy_rows = np.asarray(sigma_log_response, dtype=float)
        if sy_rows.shape != (np.size(response),):
            raise ValueError(f"sigma_log_response has shape {sy_rows.shape}, expected ({np.size(response)},)")
        ok_sy = np.isfinite(sy_rows) & (sy_rows > 0)
        response = np.where(ok_sy, np.asarray(response, dtype=float), np.nan)
    names, log_y, x, _, keep = _log_design(response, predictors, np.zeros(np.size(response)))
    n, k = x.shape
    if n <= k:
        raise ValueError(f"{n} usable rows cannot fit {k} coefficients")
    if per_row:
        sy = sy_rows[keep]
    else:
        sy = float(sigma_log_response)
        if not (np.isfinite(sy) and sy > 0):
            raise ValueError(f"sigma_log_response must be finite and > 0, got {sigma_log_response!r}")
    missing = [name for name in names if name not in sigma_log_predictors]
    if missing:
        raise ValueError(f"no log error given for predictor(s) {missing}")
    sx = np.array([float(sigma_log_predictors[name]) for name in names])
    if np.any(~np.isfinite(sx)) or np.any(sx < 0):
        raise ValueError(f"predictor log errors must be finite and >= 0, got {sx!r}")
    noisy = np.flatnonzero(sx > 0)
    exact = np.flatnonzero(sx == 0)
    if noisy.size == 0:
        raise ValueError("no predictor has an error: use fit_confinement_scaling")
    ols, *_ = np.linalg.lstsq(x, log_y, rcond=None)
    if per_row:
        from scipy.optimize import least_squares

        # For a linear model with independent Gaussian errors the orthogonal-distance
        # (maximum-likelihood) estimate minimises sum_i r_i^2 with the effective
        # variance sigma_y,i^2 + sum_j b_j^2 sigma_x,j^2 (exact predictors: sigma 0).
        def residuals(beta):
            scale = np.sqrt(sy**2 + np.sum((beta[1:] * sx) ** 2))
            return (log_y - x @ beta) / scale

        fit = least_squares(residuals, ols, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=20000)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ValueError(f"weighted ODR did not converge: {fit.message}")
        return {"names": ("log_C", *names), "coef": np.asarray(fit.x, dtype=float), "n": n, "mask": keep,
                "ols_coef": ols, "method": "weighted_odr"}

    base = np.column_stack([np.ones(n)] + [x[:, 1 + j] for j in exact])
    proj = base @ np.linalg.pinv(base)
    resid = lambda v: v - proj @ v  # noqa: E731
    z = np.column_stack([resid(x[:, 1 + j]) / sx[j] for j in noisy] + [resid(log_y) / sy])
    v = np.linalg.svd(z, full_matrices=False)[2][-1]
    if abs(v[-1]) < 1e-12:
        raise ValueError("the errors-in-variables solution is vertical: no finite slope")
    b_noisy = -(v[:-1] / sx[noisy]) / (v[-1] / sy)
    rest = log_y - x[:, 1 + noisy] @ b_noisy
    b_base, *_ = np.linalg.lstsq(base, rest, rcond=None)
    coef = np.zeros(k)
    coef[0] = b_base[0]
    coef[1 + exact] = b_base[1:]
    coef[1 + noisy] = b_noisy
    return {"names": ("log_C", *names), "coef": coef, "n": n, "mask": keep, "ols_coef": ols,
            "method": "partial_tls"}


def infer_kadomtsev_size_exponent(
    alpha: Mapping[str, float],
    cov: np.ndarray,
) -> tuple[float, float]:
    """Size exponent the Kadomtsev constraint implies for fitted I, B, P (and n) exponents.

    Parameters
    ----------
    alpha : Mapping
        Fitted exponents keyed ``i_p``, ``b_t``, ``p_net`` and, optionally,
        ``n_e`` (absent means zero by model definition, not measured) [-].
    cov : numpy.ndarray
        Their covariance, rows and columns in the order of ``alpha`` [-].

    Returns
    -------
    alpha_R : float
        $\\alpha_R^{\\mathrm{KCT}} = \\tfrac14\\alpha_I + \\tfrac54\\alpha_B
        + \\tfrac34\\alpha_P + 2\\alpha_n + \\tfrac54$ [-].
    stderr : float
        Its standard error, $\\sqrt{g^\\top C g}$ with $g$ the coefficients
        above [-].

    Raises
    ------
    ValueError
        A missing or unknown exponent name, or a covariance of the wrong
        shape.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    An *assumed* size exponent, not a measured one: one machine's shot-to-shot
    radius changes are not a size scan, and a Kadomtsev-completed scaling
    satisfies the constraint by construction, so a zero residual is no
    evidence for it. Label every use accordingly (#548 Sec. 7).

    Provenance
    ----------
    .. [548] Issue #548 Sec. 7: the Connor-Taylor/Kadomtsev completion and its
       covariance propagation.
    .. [Kadomtsev] B. B. Kadomtsev, *Sov. J. Plasma Phys.* **1** (1975) 295.
    """
    weights = {"i_p": 0.25, "b_t": 1.25, "p_net": 0.75, "n_e": 2.0}
    names = list(alpha)
    unknown = [name for name in names if name not in weights]
    if unknown or not {"i_p", "b_t", "p_net"} <= set(names):
        raise ValueError(f"exponents must be i_p, b_t, p_net and optionally n_e; got {names}")
    c = np.atleast_2d(np.asarray(cov, dtype=float))
    if c.shape != (len(names), len(names)):
        raise ValueError(f"cov must be {len(names)}x{len(names)}, got {c.shape}")
    g = np.array([weights[name] for name in names])
    value = float(g @ np.array([float(alpha[name]) for name in names]) + 1.25)
    return value, float(np.sqrt(max(g @ c @ g, 0.0)))


def dimensionless_confinement_indices(
    alpha: Mapping[str, float],
    cov: np.ndarray,
) -> dict:
    """Kadomtsev-completed dimensionless indices $(\\mu_\\rho, \\mu_\\beta, \\mu_\\nu, \\mu_q)$ with their covariance.

    Parameters
    ----------
    alpha : Mapping
        Fitted exponents keyed ``i_p``, ``b_t``, ``p_net`` and optionally
        ``n_e`` [-].
    cov : numpy.ndarray
        Their covariance in the order of ``alpha`` [-].

    Returns
    -------
    dict
        ``alpha_R`` and ``alpha_R_stderr``, the completed size exponent [-];
        ``names`` (``mu_rho``, ``mu_beta``, ``mu_nu``, ``mu_q``) [-];
        ``value`` and ``cov`` of the indices [-]; ``stderr`` [-] [any].

    Raises
    ------
    ValueError
        As :func:`infer_kadomtsev_size_exponent`, or $\\alpha_P = -1$.

    Processing steps
    ----------------
    1. Complete the size exponent with :func:`infer_kadomtsev_size_exponent`.
    2. Map the completed engineering exponents to dimensionless indices with
       :func:`vaft.formula.equilibrium.dimensionless_scaling_coeffs_from_engineering_scaling_coeffs`
       (exact here: a completed scaling satisfies the constraint), and
       $\\mu_q = -\\alpha_I/(1+\\alpha_P)$.
    3. Propagate ``cov`` through the map's Jacobian (central differences,
       the size exponent completed at every evaluation).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Inherits the *assumed* size exponent: the indices are those of the
    Kadomtsev-completed scaling, not measured dimensionless dependences. The
    linear propagation is poor when $1 + \\alpha_P$ is within a few standard
    errors of zero, where every index diverges.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 7 (layer E): dimensionless conversion with
       uncertainties and covariances.
    .. [Luce] T. C. Luce, C. C. Petty and J. G. Cordey, *Plasma Phys. Control.
       Fusion* **50** (2008) 043001.
    """
    from vaft.formula.equilibrium import (
        dimensionless_scaling_coeffs_from_engineering_scaling_coeffs as _to_dimensionless,
    )

    names = list(alpha)
    a0 = np.array([float(alpha[name]) for name in names])
    alpha_r, alpha_r_se = infer_kadomtsev_size_exponent(alpha, cov)

    def indices(vec):
        a = dict(zip(names, vec))
        a_r, _ = infer_kadomtsev_size_exponent(a, np.zeros((len(names), len(names))))
        mu = _to_dimensionless(a["i_p"], a["b_t"], a["p_net"], a.get("n_e", 0.0), 0.0, a_r, 0.0, 0.0)
        return np.array([mu[0], mu[1], mu[2], -a["i_p"] / (1.0 + a["p_net"])])

    value = indices(a0)
    jac = np.zeros((4, len(names)))
    for j in range(len(names)):
        h = 1e-6 * max(1.0, abs(a0[j]))
        up, down = a0.copy(), a0.copy()
        up[j] += h
        down[j] -= h
        jac[:, j] = (indices(up) - indices(down)) / (2 * h)
    c = jac @ np.atleast_2d(np.asarray(cov, dtype=float)) @ jac.T
    return {"alpha_R": alpha_r, "alpha_R_stderr": alpha_r_se,
            "names": ("mu_rho", "mu_beta", "mu_nu", "mu_q"), "value": value, "cov": c,
            "stderr": np.sqrt(np.clip(np.diag(c), 0.0, None))}


def closure_constraint(mu_rho: float, names) -> tuple[dict, float]:
    """Linear constraint on I, B, P (and n) exponents fixing the Kadomtsev-completed $\\mu_\\rho$.

    Parameters
    ----------
    mu_rho : float
        Gyroradius index of the closure: $-2$ Bohm, $-3$ gyro-Bohm, or any
        value between [-].
    names : sequence of str
        The fit's predictor names: ``i_p``, ``b_t``, ``p_net`` and optionally
        ``n_e`` [-].

    Returns
    -------
    coef : dict
        Constraint coefficient per predictor name [-].
    rhs : float
        Right-hand side, so that ``sum(coef[k] * alpha[k]) == rhs`` [-].

    Raises
    ------
    ValueError
        Unknown or missing predictor names.

    Processing steps
    ----------------
    1. Complete the size exponent with the Kadomtsev constraint,
       $\\alpha_R = (\\alpha_I + 5\\alpha_B + 3\\alpha_P + 8\\alpha_n + 5)/4$.
    2. Substitute into
       $\\mu_\\rho(1+\\alpha_P) = \\alpha_B - \\alpha_I - 2\\alpha_R + 2\\alpha_n
       - 3\\alpha_P + 1$, which leaves
       $\\tfrac32\\alpha_I + \\tfrac32\\alpha_B + (\\mu_\\rho + \\tfrac92)\\alpha_P
       + 2\\alpha_n = -(\\mu_\\rho + \\tfrac32)$.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The closures are alternative hypotheses, not simultaneous constraints;
    the intermediate $\\mu_\\rho = -2.5$ is a single index, not an additive
    Bohm plus gyro-Bohm model (#548 Sec. 8). A density-free fit has
    $\\alpha_n = 0$ by definition, which is part of the hypothesis tested.

    Provenance
    ----------
    .. [548] Issue #548 Sec. 8: Bohm, intermediate and gyro-Bohm closures as
       one parameterized constraint.
    """
    allowed = {"i_p": 1.5, "b_t": 1.5, "p_net": float(mu_rho) + 4.5, "n_e": 2.0}
    names = list(names)
    unknown = [name for name in names if name not in allowed]
    if unknown or not {"i_p", "b_t", "p_net"} <= set(names):
        raise ValueError(f"predictors must be i_p, b_t, p_net and optionally n_e; got {names}")
    return {name: allowed[name] for name in names}, -(float(mu_rho) + 1.5)


def fit_constrained_confinement_scaling(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
    groups: np.ndarray,
    constraint: Mapping[str, float],
    rhs: float,
) -> ConfinementScalingFit:
    """Log-linear power-law fit under one linear equality constraint on the exponents.

    Parameters
    ----------
    response : numpy.ndarray
        The fitted quantity, usually $\\tau_E$ [any].
    predictors : Mapping
        Predictor arrays keyed by name [any].
    groups : numpy.ndarray
        Group label per row (the shot) [-].
    constraint : Mapping
        Coefficient per predictor name; omitted predictors (and the
        intercept) enter with zero [-].
    rhs : float
        Right-hand side of ``sum(constraint[k] * alpha[k]) == rhs`` [-].

    Returns
    -------
    ConfinementScalingFit
        As :func:`fit_confinement_scaling`, with the constrained coefficients,
        the cluster covariance projected onto the constraint (singular along
        it) and Student-t intervals with $G - 1$ degrees of freedom [any].

    Raises
    ------
    ValueError
        As :func:`fit_confinement_scaling`, or a constraint naming an unknown
        predictor or with all-zero coefficients.

    Processing steps
    ----------------
    1. The unconstrained least-squares fit and its clustered covariance $V$
       (:func:`fit_confinement_scaling`).
    2. Restricted least squares,
       $\\beta_c = \\beta - (X^\\top X)^{-1}c^\\top(c(X^\\top X)^{-1}c^\\top)^{-1}(c\\beta - r)$.
    3. Covariance $MVM^\\top$ with
       $M = I - (X^\\top X)^{-1}c^\\top(c(X^\\top X)^{-1}c^\\top)^{-1}c$.
    4. Residuals, $R^2$, the restricted leverage
       $H - XAc^\\top(cAc^\\top)^{-1}cAX^\\top$ ($A = (X^\\top X)^{-1}$) and Cook's
       distance with $k - 1$ free coefficients; ``stderr_iid`` from
       $s^2 MAM^\\top$ with the constrained residual variance.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Least squares only (no robust option).

    Provenance
    ----------
    .. [548] Issue #548 Sec. 8: closures fitted as constraints and compared.
    .. [Greene] W. H. Greene, *Econometric Analysis*, 7th ed., Pearson (2012),
       Sec. 5.5: restricted least squares.
    """
    from scipy import stats

    base = fit_confinement_scaling(response, predictors, groups)
    names = list(base.names)
    unknown = [name for name in constraint if name not in names[1:]]
    if unknown:
        raise ValueError(f"constraint names unknown predictor(s) {unknown}")
    c = np.array([0.0] + [float(constraint.get(name, 0.0)) for name in names[1:]])
    if not np.any(c):
        raise ValueError("the constraint has no non-zero coefficient")
    _, log_y, x, g, keep = _log_design(response, predictors, groups)
    xtx_inv = np.linalg.inv(x.T @ x)
    gain = xtx_inv @ c / float(c @ xtx_inv @ c)
    coef = base.coef - gain * (float(c @ base.coef) - float(rhs))
    m = np.eye(c.size) - np.outer(gain, c)
    cov = m @ base.cov @ m.T
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t = stats.t.ppf(0.975, base.n_groups - 1)
    residuals = log_y - x @ coef
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    k_free = c.size - 1
    s2 = float(np.sum(residuals**2)) / max(base.n - k_free, 1)
    iid = m @ (s2 * xtx_inv) @ m.T
    # Restricted hat matrix H_c = H - X A c'(c A c')^-1 c A X', A = (X'X)^-1.
    xa_c = x @ (xtx_inv @ c)
    lev = base.leverage - xa_c**2 / float(c @ xtx_inv @ c)
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = residuals**2 / (k_free * s2) * lev / (1.0 - lev) ** 2
    return ConfinementScalingFit(
        names=base.names, coef=coef, stderr=stderr, cov=cov,
        ci95=np.column_stack([coef - t * stderr, coef + t * stderr]),
        stderr_iid=np.sqrt(np.clip(np.diag(iid), 0.0, None)), n=base.n, n_groups=base.n_groups,
        r2=1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else np.nan,
        rmse_log=float(np.sqrt(np.mean(residuals**2))), residuals=residuals,
        leverage=lev, cooks_distance=cooks, mask=keep, robust=False,
    )
