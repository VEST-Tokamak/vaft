r"""Turbulence--zonal-flow diagnostics of gyrokinetic time series and the predator--prey fit (issue #1820).

Turns the potential of a nonlinear gyrokinetic run into the two intensities the
reduced model of :mod:`vaft.formula.turbulence` speaks about, characterises the
cycles they go through, and fits the two-variable Lotka--Volterra model to them.
Every routine takes plain NumPy arrays, so any code that writes
$\phi_{k_x,k_y}(t)$ can be analysed; nothing here runs, steers or reads a
solver.

The chain::

    phi(kx, ky, t)
        -> zonal_turbulence_intensity : N(t) = sum_{ky!=0} |phi|^2,  E(t) = sum_{ky=0} |phi|^2
        -> zonal_shear_proxy          : sum kx^2 |phi_{kx,0}|^2, sum kx^4 |phi_{kx,0}|^2
    (N, E, optional transport)
        -> predator_prey_cycle_metrics : peaks, periods, peak lag, correlation lag
        -> fit_predator_prey_model     : gamma_eff, c1, c2, gamma_Z by trajectory fit

The output is a consistency test, not a mechanism claim.  A good fit says the
traces are compatible with predator--prey dynamics under the stated
intensity definitions.  It does not establish Dimits-shift physics, and a poor
fit does not rule zonal flows out: tertiary instability, profile evolution and
noise also make traces bursty.

Notation
--------
phi           : fluctuating electrostatic potential, Fourier components    [solver units]
N, E          : turbulence (k_y != 0) and zonal (k_y = 0) potential intensity  [phi^2]
gamma_eff, gamma_Z : effective growth and zonal damping rates             [1/time]
c1, c2        : suppression and drive couplings                           [1/(time E)], [1/(time N)]

Conventions
-----------
**Intensities are proxies.**  Unweighted $\sum|\phi|^2$ is a
potential-intensity proxy, not an energy; a fit's $c_1, c_2$ are
meaningful only together with the definition of $N$ and $E$, which
every result records (:attr:`PredatorPreyFit.normalization`).  Rescaling
$N\to aN$, $E\to bE$ maps $c_1\to c_1/b$,
$c_2\to c_2/a$; the rates are unchanged.

**Time is the caller's.**  Any consistent unit (s, $a/c_s$); every
returned rate or lag is in that unit.

**Positive lag = zonal flow after turbulence**, for both the peak lag and the
cross-correlation lag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "PredatorPreyCycleMetrics",
    "PredatorPreyFit",
    "ZonalShearProxy",
    "ZonalTurbulenceIntensity",
    "fit_predator_prey_model",
    "predator_prey_cycle_metrics",
    "zonal_shear_proxy",
    "zonal_turbulence_intensity",
]

#: Keys every fit's normalisation record should carry (#1820 Sec. 10).
NORMALIZATION_KEYS = (
    "turbulence_definition",
    "zonal_definition",
    "potential_normalization",
    "time_normalization",
    "spectral_coordinates",
)


@dataclass(frozen=True)
class ZonalTurbulenceIntensity:
    """Zonal and non-zonal potential intensity per time sample."""

    turbulence: np.ndarray
    zonal: np.ndarray
    ratio: np.ndarray
    weighting: str


@dataclass(frozen=True)
class ZonalShearProxy:
    """Spectral zonal-flow and zonal-shear proxies per time sample."""

    flow: np.ndarray
    shear: np.ndarray
    normalization: str


@dataclass(frozen=True)
class PredatorPreyCycleMetrics:
    """Cycle-level observables of a turbulence / zonal-flow pair of traces."""

    n_cycles: int
    turbulence_peak_times: np.ndarray
    zonal_peak_times: np.ndarray
    periods: np.ndarray
    peak_response_lags: np.ndarray
    period_mean: float
    period_std: float
    peak_response_lag_mean: float
    peak_response_lag_std: float
    correlation_lag: float
    correlation_lags: np.ndarray
    cross_correlation: np.ndarray
    qualified: bool
    reasons: tuple[str, ...]
    transport_cycles: Optional[dict] = None


@dataclass(frozen=True)
class PredatorPreyFit:
    """Lotka--Volterra coefficients fitted to a turbulence / zonal-flow pair."""

    gamma_eff: float
    coupling_suppression: float
    coupling_drive: float
    gamma_zonal: float
    turbulence_fit: np.ndarray
    zonal_fit: np.ndarray
    rms_turbulence: float
    rms_zonal: float
    success: bool
    reasons: tuple[str, ...]
    time_window: tuple[float, float]
    normalization: dict = field(default_factory=dict)
    initial_state: tuple[float, float] = (float("nan"), float("nan"))
    method: str = "trajectory"


# -- helpers ----------------------------------------------------------------------


def _spectral_cube(phi, name="phi") -> np.ndarray:
    cube = np.asarray(phi)
    if cube.ndim != 3:
        raise ValueError(f"{name} must be (n_kx, n_ky, n_time); got shape {cube.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"{name} must be finite")
    return cube


def _ky_masks(ky, n_ky: int, tolerance: float):
    k = np.asarray(ky, dtype=float).ravel()
    if k.size != n_ky:
        raise ValueError(f"ky has {k.size} entries but phi has {n_ky} along its ky axis")
    zonal = np.abs(k) <= tolerance
    if not np.any(zonal):
        raise ValueError("no ky = 0 component within ky_tolerance: the zonal part is undefined")
    return zonal, ~zonal


def _traces(time, turbulence, zonal):
    t = np.asarray(time, dtype=float).ravel()
    n = np.asarray(turbulence, dtype=float).ravel()
    e = np.asarray(zonal, dtype=float).ravel()
    if not (t.size == n.size == e.size):
        raise ValueError("time, turbulence and zonal must have the same length")
    if t.size < 4:
        raise ValueError("at least four samples are needed")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(n)) and np.all(np.isfinite(e))):
        raise ValueError("time, turbulence and zonal must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    return t, n, e


def _peaks(t, y, prominence_fraction):
    from scipy.signal import find_peaks

    span = float(np.nanmax(y) - np.nanmin(y))
    if span <= 0:
        return np.array([], dtype=int)
    index, _ = find_peaks(y, prominence=prominence_fraction * span)
    return index


# -- intensities ------------------------------------------------------------------


def zonal_turbulence_intensity(phi, ky, *, weights=None, ky_tolerance=1e-12, weighting=None):
    r"""Split fluctuation intensity into its zonal ($k_y = 0$) and non-zonal parts.

    $I_\mathrm{turb}(t) = \sum_{k_x, k_y\ne0} w\,|\phi_{k_x,k_y}(t)|^2$,
    $I_\mathrm{zonal}(t) = \sum_{k_x} w\,|\phi_{k_x,0}(t)|^2$ and
    $R_{ZT} = I_\mathrm{zonal}/I_\mathrm{turb}$.

    Parameters
    ----------
    phi : array-like
        Potential Fourier components, shape ``(n_kx, n_ky, n_time)``, real or
        complex; average or select any poloidal coordinate beforehand [solver units].
    ky : array-like
        Binormal wavenumber of each ``ky`` column; the zonal column is the one
        with ``|ky| <= ky_tolerance`` [1/length].
    weights : array-like, optional
        Spectral weight broadcastable to ``(n_kx, n_ky)``; ``None`` for 1 [caller units].
    ky_tolerance : float, optional
        Largest ``|ky|`` counted as zonal [1/length].
    weighting : str, optional
        Name of the weighting, recorded in the result [-].

    Returns
    -------
    ZonalTurbulenceIntensity
        ``turbulence``, ``zonal`` and ``ratio`` per time sample (``ratio`` is
        NaN where the turbulence intensity is zero) and the ``weighting`` label
        [weight * phi^2; ratio -].

    Raises
    ------
    ValueError
        ``phi`` is not 3-D or not finite, ``ky`` does not match its ky axis, no
        ``ky`` is zonal within the tolerance, or a weight is negative.

    Defaults
    --------
    ``ky_tolerance=1e-12`` is a numerical convenience: solvers store the zonal
    wavenumber as exact zero.  Unit weights (``weights=None``) are the
    conventional potential-intensity proxy.

    Convention
    ----------
    Unweighted $|\phi|^2$ is a potential-intensity proxy, labelled as such; it
    becomes an energy only with the metric weights the caller supplies (e.g.
    $k_\perp^2$ for an $E\times B$ kinetic-energy proxy).  Solvers that store
    only $k_y \ge 0$ count each non-zonal mode once.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] S. Kobayashi, O. D. Gurcan and P. H. Diamond, Phys. Plasmas 22 (2015) 090702.
    .. [issue] #1820 Sec. 6.
    """
    cube = _spectral_cube(phi)
    zonal_mask, turb_mask = _ky_masks(ky, cube.shape[1], ky_tolerance)
    power = np.abs(cube) ** 2
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        power = power * np.broadcast_to(w, cube.shape[:2])[..., None]
    turbulence = power[:, turb_mask].sum(axis=(0, 1))
    zonal = power[:, zonal_mask].sum(axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(turbulence > 0, zonal / np.where(turbulence > 0, turbulence, 1.0), np.nan)
    label = weighting or ("unweighted |phi|^2 (potential-intensity proxy)" if weights is None
                          else "caller-weighted |phi|^2")
    return ZonalTurbulenceIntensity(turbulence=turbulence, zonal=zonal, ratio=ratio, weighting=label)


def zonal_shear_proxy(phi, kx, ky, *, ky_tolerance=1e-12):
    r"""Spectral proxies for the zonal-flow energy and the zonal shear.

    $E_Z \propto \sum_{k_x} k_x^2|\phi_{k_x,0}|^2$ and
    $S_Z^2 \propto \sum_{k_x} k_x^4|\phi_{k_x,0}|^2$, from
    $v_{E,Z} \propto k_x\phi_Z$ and $\partial_x v_{E,Z} \propto k_x^2\phi_Z$.

    Parameters
    ----------
    phi : array-like
        Potential Fourier components, shape ``(n_kx, n_ky, n_time)`` [solver units].
    kx : array-like
        Radial wavenumber of each ``kx`` row, length ``n_kx`` [1/length].
    ky : array-like
        Binormal wavenumber of each ``ky`` column, length ``n_ky`` [1/length].
    ky_tolerance : float, optional
        Largest ``|ky|`` counted as zonal [1/length].

    Returns
    -------
    ZonalShearProxy
        ``flow`` and ``shear`` per time sample and the ``normalization`` label
        [kx^2 phi^2, kx^4 phi^2].

    Raises
    ------
    ValueError
        Shapes do not match, inputs are not finite, or no zonal ``ky`` exists.

    Defaults
    --------
    ``ky_tolerance=1e-12`` is a numerical convenience (solvers store the zonal
    wavenumber as exact zero).

    Convention
    ----------
    Spectral proxies in the units of $\phi$ and $k_x$: physical flow energy
    and shearing rate need the solver's potential and perpendicular-length
    normalisation and geometric factors, which this routine does not apply.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] P. H. Diamond, S.-I. Itoh, K. Itoh and T. S. Hahm,
           Plasma Phys. Control. Fusion 47 (2005) R35.
    .. [issue] #1820 Sec. 7.
    """
    cube = _spectral_cube(phi)
    zonal_mask, _ = _ky_masks(ky, cube.shape[1], ky_tolerance)
    k = np.asarray(kx, dtype=float).ravel()
    if k.size != cube.shape[0] or not np.all(np.isfinite(k)):
        raise ValueError(f"kx must be finite with {cube.shape[0]} entries")
    power = (np.abs(cube[:, zonal_mask]) ** 2).sum(axis=1)       # (n_kx, n_time)
    flow = (k[:, None] ** 2 * power).sum(axis=0)
    shear = (k[:, None] ** 4 * power).sum(axis=0)
    return ZonalShearProxy(flow=flow, shear=shear,
                           normalization="spectral proxy: sum kx^p |phi_{kx,0}|^2 in solver units")


# -- cycles -----------------------------------------------------------------------


def predator_prey_cycle_metrics(time, turbulence, zonal, *, transport=None, min_cycles=2,
                                prominence=0.2, scale="log"):
    r"""Cycle observables of a turbulence / zonal-flow pair: periods, peak lag, correlation lag.

    Parameters
    ----------
    time : array-like
        Strictly increasing sample times [time].
    turbulence : array-like
        Turbulence intensity $N(t)$ [N].
    zonal : array-like
        Zonal-flow intensity $E(t)$ [E].
    transport : array-like, optional
        A transport trace on the same times (a flux); summarised per cycle,
        never used in place of $N$ [flux].
    min_cycles : int, optional
        Complete turbulence cycles needed for ``qualified`` [-].
    prominence : float, optional
        Peak prominence as a fraction of each trace's range [-].
    scale : {"log", "linear"}, optional
        Detect peaks and correlate on $\ln N, \ln E$ or on $N, E$ [-].

    Returns
    -------
    PredatorPreyCycleMetrics
        Peak times of both traces, cycle periods (turbulence peak to peak) and
        their mean and spread, the per-cycle peak response lag (first zonal
        peak after each turbulence peak and before the next) with mean and
        spread, the cross-correlation curve and the lag of its maximum,
        ``qualified`` with the ``reasons`` it is not, and optional
        ``transport_cycles`` [times and lags in the unit of ``time``].

    Raises
    ------
    ValueError
        Mismatched or non-finite traces, non-increasing time, fewer than four
        samples, or ``min_cycles < 1``.

    Processing steps
    ----------------
    1. Take $\ln N, \ln E$ (``scale="log"``) or $N, E$; find peaks of each with
       a prominence of ``prominence`` times that trace's range.
    2. Periods are the spacings of successive $N$ peaks; ``n_cycles`` counts them.
    3. For each $N$ peak, the first $E$ peak before the next $N$ peak gives the
       peak response lag.
    4. On a uniform resampling of both traces (median spacing), standardise and
       cross-correlate; the lag of the maximum within $\pm$ half a mean period
       (a quarter of the record without cycles) is ``correlation_lag``.
    5. Qualify: at least ``min_cycles`` cycles, a lag for most of them, and a
       positive mean peak lag.

    Defaults
    --------
    ``min_cycles=2`` is the conventional minimum for a period estimate with a
    spread; ``prominence=0.2`` is a numerical convenience that ignores
    sub-burst wiggles and should be lowered for weak, regular oscillations.
    ``scale="log"`` is a numerical convenience for bursty gyrokinetic traces
    that span decades, where a linear-range prominence sees only the largest
    burst; peak *times* do not depend on the scale, only which peaks pass.

    Convention
    ----------
    Lags are positive when the zonal flow follows the turbulence.  Peak lag and
    cross-correlation lag are different estimators and are returned
    separately; for a small-amplitude Lotka--Volterra orbit both equal $T/4$
    (:func:`vaft.formula.turbulence.predator_prey_response_lag`).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Irregular bursts give a period distribution, not a period; the spread is
    reported, not hidden.  Peaks closer than the sampling interval are not
    resolved.

    Provenance
    ----------
    .. [1] M. Leconte, A. Masson and L. Qi, Phys. Plasmas 29 (2022) 022302.
    .. [issue] #1820 Sec. 8.
    """
    t, n, e = _traces(time, turbulence, zonal)
    if int(min_cycles) < 1:
        raise ValueError("min_cycles must be at least 1")
    if scale == "log":
        if np.any(n <= 0) or np.any(e <= 0):
            raise ValueError('scale="log" needs strictly positive intensities')
        n, e = np.log(n), np.log(e)
    elif scale != "linear":
        raise ValueError(f"scale must be 'log' or 'linear', not {scale!r}")
    reasons: list[str] = []
    n_peaks = _peaks(t, n, prominence)
    e_peaks = _peaks(t, e, prominence)
    tn, te = t[n_peaks], t[e_peaks]
    periods = np.diff(tn)
    lags = []
    for k, start in enumerate(tn):
        stop = tn[k + 1] if k + 1 < tn.size else np.inf
        following = te[(te > start) & (te < stop)]
        if following.size:
            lags.append(following[0] - start)
    lags = np.asarray(lags, dtype=float)

    period_mean = float(periods.mean()) if periods.size else float("nan")
    period_std = float(periods.std(ddof=1)) if periods.size > 1 else float("nan")
    lag_mean = float(lags.mean()) if lags.size else float("nan")
    lag_std = float(lags.std(ddof=1)) if lags.size > 1 else float("nan")

    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    nu = np.interp(grid, t, n)
    eu = np.interp(grid, t, e)
    nu = (nu - nu.mean()) / (nu.std() or 1.0)
    eu = (eu - eu.mean()) / (eu.std() or 1.0)
    # half a period each way: a periodic correlation repeats every period, so a
    # wider search would find the alias at T/4 - T as easily as T/4
    span = 0.5 * period_mean if np.isfinite(period_mean) else 0.25 * (t[-1] - t[0])
    max_shift = max(1, min(int(round(span / dt)), grid.size - 2))
    shifts = np.arange(-max_shift, max_shift + 1)
    cc = np.empty(shifts.size)
    for i, s in enumerate(shifts):           # corr(N(t), E(t + s dt))
        if s >= 0:
            a, b = nu[: grid.size - s], eu[s:]
        else:
            a, b = nu[-s:], eu[: grid.size + s]
        cc[i] = float(np.mean(a * b))
    correlation_lags = shifts * dt
    correlation_lag = float(correlation_lags[int(np.argmax(cc))])

    n_cycles = int(periods.size)
    if n_cycles < int(min_cycles):
        reasons.append(f"{n_cycles} complete cycles, fewer than min_cycles={int(min_cycles)}")
    if tn.size and lags.size < max(1, tn.size - 1):
        reasons.append(f"a zonal peak follows only {lags.size} of {tn.size} turbulence peaks")
    if lags.size and lag_mean <= 0:
        reasons.append("the mean peak lag is not positive")

    transport_cycles = None
    if transport is not None:
        q = np.asarray(transport, dtype=float).ravel()
        if q.size != t.size or not np.all(np.isfinite(q)):
            raise ValueError("transport must be finite with one value per time sample")
        cycles = []
        for a, b in zip(tn[:-1], tn[1:]):
            inside = (t >= a) & (t <= b)
            if np.count_nonzero(inside) >= 2:
                integ = getattr(np, "trapezoid", None) or np.trapz
                cycles.append({"start": float(a), "stop": float(b),
                               "mean": float(integ(q[inside], t[inside]) / (b - a)),
                               "max": float(q[inside].max()), "min": float(q[inside].min()),
                               "peak_time": float(t[inside][int(np.argmax(q[inside]))])})
        transport_cycles = {"cycles": cycles}

    return PredatorPreyCycleMetrics(
        n_cycles=n_cycles, turbulence_peak_times=tn, zonal_peak_times=te, periods=periods,
        peak_response_lags=lags, period_mean=period_mean, period_std=period_std,
        peak_response_lag_mean=lag_mean, peak_response_lag_std=lag_std,
        correlation_lag=correlation_lag, correlation_lags=correlation_lags, cross_correlation=cc,
        qualified=not reasons, reasons=tuple(reasons), transport_cycles=transport_cycles,
    )


# -- fit --------------------------------------------------------------------------


def _per_capita_initial(t, n, e):
    """gamma_eff, c1, c2, gamma_Z from d ln N/dt = g - c1 E, d ln E/dt = c2 N - gZ."""
    dlnn = np.gradient(np.log(n), t)
    dlne = np.gradient(np.log(e), t)
    a1 = np.column_stack([np.ones_like(e), -e])
    a2 = np.column_stack([n, -np.ones_like(n)])
    (g, c1), *_ = np.linalg.lstsq(a1, dlnn, rcond=None)
    (c2, gz), *_ = np.linalg.lstsq(a2, dlne, rcond=None)
    return np.array([g, c1, c2, gz], dtype=float)


def _fixed_point_initial(t, n, e):
    """Coefficients that put the fixed point at the trace means with a period of
    the record over its number of N peaks (or a quarter record)."""
    peaks = _peaks(t, n, 0.2)
    period = (t[-1] - t[0]) / max(len(peaks), 4)
    omega = 2 * np.pi / period
    n_star, e_star = float(np.mean(n)), float(np.mean(e))
    return np.array([omega, omega / e_star, omega / n_star, omega], dtype=float)


def fit_predator_prey_model(time, turbulence, zonal, *, method="trajectory", initial_parameters=None,
                            normalization=None, window=None, fit_initial_state=True):
    r"""Fit the two-variable Lotka--Volterra model to turbulence and zonal-flow traces.

    Finds $\gamma_\mathrm{eff}, c_1, c_2, \gamma_Z$ (and, by default, the
    initial state) whose trajectory of
    :func:`vaft.formula.turbulence.predator_prey_rhs` best matches the data.

    Parameters
    ----------
    time : array-like
        Strictly increasing sample times [time].
    turbulence : array-like
        Turbulence intensity $N(t)$, positive [N].
    zonal : array-like
        Zonal-flow intensity $E(t)$, positive [E].
    method : {"trajectory"}, optional
        Only the trajectory fit is implemented [-].
    initial_parameters : sequence of 4 floats, optional
        Starting $(\gamma_\mathrm{eff}, c_1, c_2, \gamma_Z)$; by default a
        per-capita regression, or the fixed-point guess when that gives a
        non-positive coefficient [1/time, 1/(time E), 1/(time N), 1/time].
    normalization : mapping, optional
        How $N$, $E$ and time were constructed; stored with the result
        (keys in ``NORMALIZATION_KEYS``) [-].
    window : (float, float), optional
        Fit only samples with ``window[0] <= t <= window[1]`` [time].
    fit_initial_state : bool, optional
        Fit $N(t_0), E(t_0)$ as well instead of taking the first samples [-].

    Returns
    -------
    PredatorPreyFit
        The four coefficients, the model traces on the data times, the RMS
        of the relative residuals of each trace, ``success`` with ``reasons``,
        the time window, the normalisation record, the initial state used and
        the method [rates 1/time; c1 1/(time E); c2 1/(time N); RMS -].

    Raises
    ------
    ValueError
        Mismatched or non-finite traces, non-positive intensities, fewer than
        four samples in the window, non-increasing time, an unknown method, or
        non-positive initial parameters.

    Processing steps
    ----------------
    1. Restrict to ``window``; take $(N, E)$ at its first sample as the initial
       state guess.
    2. Start from ``initial_parameters``, else the per-capita regression of
       $\dot N/N = \gamma_\mathrm{eff} - c_1E$, $\dot E/E = c_2N - \gamma_Z$,
       else the fixed-point guess.
    3. Integrate :func:`~vaft.formula.turbulence.predator_prey_rhs` from the
       initial state and sample it at the data times.
    4. Minimise the residual $\ln N_\mathrm{model} - \ln N$ and
       $\ln E_\mathrm{model} - \ln E$ over log-coefficients (positivity by
       construction) with a bounded least-squares solver.
    5. Report the model traces, relative RMS per trace and the normalisation.

    Defaults
    --------
    The trajectory method is the conventional choice (#1820 Sec. 9): it uses no
    numerical derivative.  Log residuals are a numerical convenience for traces
    spanning decades.  ``fit_initial_state=True`` is a numerical convenience:
    a noisy first sample otherwise biases the whole orbit.

    Convention
    ----------
    $c_1, c_2$ carry the units of $1/E$, $1/N$ and are comparable across fits
    only when ``normalization`` matches; the rates are normalisation-free.

    Assumptions
    -----------
    The traces follow one closed orbit of the classical model over the window;
    bursts of changing amplitude (orbits with different invariant) are fitted
    by a single compromise orbit.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A local optimum is possible; ``success`` reflects the optimiser and a
    relative RMS below one, not a global search.  Irregular bursty traces fit
    poorly by construction, and that misfit is the information.

    Provenance
    ----------
    .. [1] S. Kobayashi, O. D. Gurcan and P. H. Diamond, Phys. Plasmas 22 (2015) 090702.
    .. [issue] #1820 Sec. 9 and 10.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares

    from vaft.formula.turbulence import predator_prey_rhs

    if method != "trajectory":
        raise ValueError(f"method must be 'trajectory', not {method!r}")
    t, n, e = _traces(time, turbulence, zonal)
    if np.any(n <= 0) or np.any(e <= 0):
        raise ValueError("turbulence and zonal must be strictly positive")
    if window is not None:
        keep = (t >= window[0]) & (t <= window[1])
        t, n, e = t[keep], n[keep], e[keep]
        if t.size < 4:
            raise ValueError(f"window {window} holds fewer than four samples")
    reasons: list[str] = []

    if initial_parameters is not None:
        p0 = np.asarray(initial_parameters, dtype=float)
        if p0.shape != (4,) or not np.all(np.isfinite(p0)) or np.any(p0 <= 0):
            raise ValueError("initial_parameters must be four finite positive numbers")
    else:
        p0 = _per_capita_initial(t, n, e)
        if not np.all(np.isfinite(p0)) or np.any(p0 <= 0):
            p0 = _fixed_point_initial(t, n, e)
    x0 = np.log(np.concatenate([p0, [n[0], e[0]]]))

    def model(x):
        g, c1, c2, gz, n0, e0 = np.exp(x)
        if not fit_initial_state:
            n0, e0 = n[0], e[0]

        def rhs(_, y):
            return predator_prey_rhs(max(y[0], 0.0), max(y[1], 0.0), g, c1, c2, gz)

        sol = solve_ivp(rhs, (t[0], t[-1]), [n0, e0], t_eval=t, method="LSODA",
                        rtol=1e-7, atol=1e-12 * max(n.max(), e.max()))
        if not sol.success or sol.y.shape[1] != t.size:
            return None
        return sol.y

    def residual(x):
        y = model(x)
        if y is None or np.any(y <= 0):
            return np.full(2 * t.size, 1e3)
        return np.concatenate([np.log(y[0]) - np.log(n), np.log(y[1]) - np.log(e)])

    result = least_squares(residual, x0, method="trf", x_scale="jac", max_nfev=2000)
    coeffs = np.exp(result.x)
    traces = model(result.x)
    if traces is None:
        traces = np.full((2, t.size), np.nan)
        reasons.append("the fitted coefficients do not integrate over the window")
    rms_n = float(np.sqrt(np.mean(((traces[0] - n) / np.mean(n)) ** 2)))
    rms_e = float(np.sqrt(np.mean(((traces[1] - e) / np.mean(e)) ** 2)))
    if not result.success:
        reasons.append(f"optimiser: {result.message}")
    if not (rms_n < 1.0 and rms_e < 1.0):
        reasons.append(f"relative RMS misfit N {rms_n:.2f}, E {rms_e:.2f} (>= 1)")
    record = dict(normalization or {})
    missing = [key for key in NORMALIZATION_KEYS if key not in record]
    if missing:
        reasons.append(f"normalisation not recorded for {', '.join(missing)}; c1, c2 are not comparable")
    initial = (float(coeffs[4]), float(coeffs[5])) if fit_initial_state else (float(n[0]), float(e[0]))
    return PredatorPreyFit(
        gamma_eff=float(coeffs[0]), coupling_suppression=float(coeffs[1]),
        coupling_drive=float(coeffs[2]), gamma_zonal=float(coeffs[3]),
        turbulence_fit=traces[0], zonal_fit=traces[1], rms_turbulence=rms_n, rms_zonal=rms_e,
        success=bool(result.success) and traces is not None and rms_n < 1.0 and rms_e < 1.0,
        reasons=tuple(reasons), time_window=(float(t[0]), float(t[-1])),
        normalization=record, initial_state=initial, method=method,
    )
