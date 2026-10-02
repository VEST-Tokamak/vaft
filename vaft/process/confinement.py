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
    "assess_predictor_identifiability",
    "bootstrap_confinement_scaling",
    "leave_one_group_out_scaling",
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


@dataclass(frozen=True)
class ConfinementScalingFit:
    """A log-linear confinement scaling fit with group-aware uncertainty.

    ``names`` starts with ``"log_C"`` (the intercept) followed by the
    predictors; ``coef``, ``stderr`` and ``ci95`` follow that order.
    ``stderr``/``cov``/``ci95`` are cluster-robust by group (CR1, t with
    G - 1 degrees of freedom); ``stderr_iid`` is the textbook OLS error that
    treats every row as independent, kept to show how much the grouping
    matters. ``mask`` marks the input rows the fit used.
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
        plus one, or fewer than two groups.

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
    else:
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
        stderr_iid=np.asarray(ols.bse, dtype=float), n=n, n_groups=n_groups,
        r2=1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else np.nan,
        rmse_log=float(np.sqrt(np.mean(residuals**2))), residuals=residuals,
        leverage=np.asarray(influence.hat_matrix_diag, dtype=float),
        cooks_distance=np.asarray(influence.cooks_distance[0], dtype=float),
        mask=keep, robust=bool(robust),
    )


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
        design with intercept [-]; ``rank`` of that design [-] [any].

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
    corr = np.corrcoef(logs) if len(names) > 1 else np.ones((1, 1))
    corr = np.atleast_2d(corr)
    vif = {}
    for j, name in enumerate(names):
        others = np.delete(logs, j, axis=0)
        if others.size == 0:
            vif[name] = 1.0
            continue
        design = np.column_stack([np.ones(n), others.T])
        beta, *_ = np.linalg.lstsq(design, logs[j], rcond=None)
        resid = logs[j] - design @ beta
        ss_tot = float(np.sum((logs[j] - logs[j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 1.0
        vif[name] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")
    design = np.column_stack([np.ones(n), logs.T])
    scaled = design / np.linalg.norm(design, axis=0)
    return {
        "n": n,
        "correlation": {a: {b: float(corr[i, j]) for j, b in enumerate(names)}
                        for i, a in enumerate(names)},
        "vif": vif,
        "condition_number": float(np.linalg.cond(scaled)),
        "rank": int(np.linalg.matrix_rank(design)),
    }


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
