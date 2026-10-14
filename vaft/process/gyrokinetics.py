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
    "ComparisonAssessment", "assess_comparison", "select_time",
    "prepare_diagnostic_traces", "local_result_signature",
    "time_window_average",
    "saturated_intensity", "radial_correlation_from_spectrum",
    "SpatialField", "reconstruct_field", "zonal_fraction", "spectral_integral",
    "rebin_contributions", "field_transport_contributions", "reduce_time_series",
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

    @property
    def fraction(self):
        total = self.zonal + self.turbulence
        return np.divide(self.zonal, total, out=np.full_like(total, np.nan), where=total > 0)


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
    r2_turbulence: float = float("nan")
    r2_zonal: float = float("nan")
    warnings: tuple[str, ...] = ()


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
    5. Qualify: at least ``min_cycles`` cycles, a following zonal peak for every
       turbulence peak but at most one (the last may have none), and a
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
    # anti-alias: not a downsample -- the series is put on a uniform grid at its own
    # median spacing so the lagged products line up; the sample rate is unchanged.
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
                            normalization=None, window=None, fit_initial_state=True, min_r2=0.5):
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
    min_r2 : float, optional
        Coefficient of determination each trace must reach for ``success`` [-].

    Returns
    -------
    PredatorPreyFit
        The four coefficients, the model traces on the data times, the RMS
        misfit of each trace divided by its mean (``rms_*``) and its
        coefficient of determination $R^2 = 1 - \sum r^2/\sum(y-\bar y)^2$
        (``r2_*``, 0 for a flat line at the mean), ``success`` with the
        ``reasons`` that block it, non-blocking ``warnings``, the time window,
        the normalisation record, the initial state used and the method
        [rates 1/time; c1 1/(time E); c2 1/(time N); RMS and R^2 -].

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
    5. Report the model traces, $R^2$ and the mean-normalised RMS per trace;
       ``success`` needs the optimiser to converge and $R^2 \ge$ ``min_r2`` on
       both traces.

    Defaults
    --------
    The trajectory method is the conventional choice (#1820 Sec. 9): it uses no
    numerical derivative.  Log residuals are a numerical convenience for traces
    spanning decades.  ``fit_initial_state=True`` is a numerical convenience:
    a noisy first sample otherwise biases the whole orbit.  ``min_r2=0.5`` is
    a conventional threshold: the model must explain at least half of each
    trace's variance.

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
    A local optimum is possible; ``success`` reflects the optimiser and the
    $R^2$ threshold, not a global search.  Irregular bursty traces fit
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
    x0 = np.log(np.concatenate([p0, [n[0], e[0]]]) if fit_initial_state else p0)

    def model(x):
        values = np.exp(x)
        g, c1, c2, gz = values[:4]
        n0, e0 = (values[4], values[5]) if fit_initial_state else (n[0], e[0])

        def rhs(_, y):
            return predator_prey_rhs(max(y[0], 0.0), max(y[1], 0.0), g, c1, c2, gz)

        sol = solve_ivp(
            rhs,
            (t[0], t[-1]),
            [n0, e0],
            t_eval=t,
            method="LSODA",
            rtol=1e-7,
            atol=1e-12 * max(n.max(), e.max()),
        )
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
    integrated = traces is not None
    if not integrated:
        traces = np.full((2, t.size), np.nan)
        reasons.append("the fitted coefficients do not integrate over the window")

    def r2(model_trace, data):
        return float(
            1.0 - np.sum((model_trace - data) ** 2) / np.sum((data - data.mean()) ** 2)
        )

    rms_n = float(np.sqrt(np.mean(((traces[0] - n) / np.mean(n)) ** 2)))
    rms_e = float(np.sqrt(np.mean(((traces[1] - e) / np.mean(e)) ** 2)))
    r2_n, r2_e = r2(traces[0], n), r2(traces[1], e)
    if not result.success:
        reasons.append(f"optimiser: {result.message}")
    if integrated and not (r2_n >= min_r2 and r2_e >= min_r2):
        reasons.append(f"R^2 N {r2_n:.2f}, E {r2_e:.2f} below {min_r2}")
    warnings: list[str] = []
    record = dict(normalization or {})
    missing = [key for key in NORMALIZATION_KEYS if key not in record]
    if missing:
        warnings.append(
            f"normalisation not recorded for {', '.join(missing)}; c1, c2 are not comparable"
        )
    initial = (
        (float(coeffs[4]), float(coeffs[5]))
        if fit_initial_state
        else (float(n[0]), float(e[0]))
    )
    return PredatorPreyFit(
        gamma_eff=float(coeffs[0]),
        coupling_suppression=float(coeffs[1]),
        coupling_drive=float(coeffs[2]),
        gamma_zonal=float(coeffs[3]),
        turbulence_fit=traces[0],
        zonal_fit=traces[1],
        rms_turbulence=rms_n,
        rms_zonal=rms_e,
        success=not reasons,
        reasons=tuple(reasons),
        time_window=(float(t[0]), float(t[-1])),
        normalization=record,
        initial_state=initial,
        method=method,
        r2_turbulence=r2_n,
        r2_zonal=r2_e,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class SpatialField:
    """One local saved-theta plane; values are native CGYRO potential units."""

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    metadata: Mapping[str, Any]


def reconstruct_field(
    coefficients, kx, ky, *, length_x, component="total", nx=None, ny=None
):
    """Reconstruct CGYRO's half-sum Re phi exp(+ikx*x-i*ky*y).

    Parameters
    ----------
    coefficients : array-like
        Complex Fourier coefficients (kx, ky), excluding native special slots [native potential].
    kx : array-like
        Radial Fourier wavenumbers [inverse perpendicular length].
    ky : array-like
        Binormal wavenumbers, including zero [inverse perpendicular length].
    length_x : float
        Positive periodic radial box length [perpendicular length].
    component : str
        Total, zonal (ky=0), or nonzonal field selection [-].
    nx : int, optional
        Radial sample count; default resolves retained Fourier harmonics [-].
    ny : int, optional
        Binormal sample count; default resolves retained Fourier harmonics [-].

    Returns
    -------
    SpatialField
        Periodic coordinates, real potential and explicit half-sum convention [native potential].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    Reconstruct CGYRO's half-sum Re phi exp(+ikx*x-i*ky*y).

    Matches pygacode b493397 plot_fluct.maptoreal, including half weight
    for ky=0. The caller must remove CGYRO's special first radial slot.
    This is a saved-theta local plane, not a global or full 3D field.

    Provenance
    ----------
    .. [1] GACODE b493397 pygacode/cgyro/plot_fluct.py, maptoreal and maptoreal_fft; #2019.
    """
    c = np.asarray(coefficients, dtype=complex)
    kx, ky = np.asarray(kx, float), np.asarray(ky, float)
    if c.shape != (kx.size, ky.size) or not np.isfinite(c).all():
        raise ValueError("finite coefficients must have shape (kx, ky)")
    if component not in ("total", "zonal", "nonzonal"):
        raise ValueError("component must be total, zonal or nonzonal")
    if (
        not np.isfinite(length_x)
        or length_x <= 0
        or not np.isfinite(kx).all()
        or not np.isfinite(ky).all()
    ):
        raise ValueError("finite wavenumbers and positive radial length required")
    if np.any(ky < 0) or not np.isclose(ky[0], 0) or ky.size < 2:
        raise ValueError("reconstruction requires ky=0 and a one-sided multi-mode grid")
    dky = ky[1]
    if dky <= 0 or not np.allclose(ky, np.arange(ky.size) * dky):
        raise ValueError("nonuniform ky grid is unsupported")
    p = kx * length_x / (2 * np.pi)
    if not np.allclose(p, np.rint(p)) or len(np.unique(np.rint(p))) != len(p):
        raise ValueError("kx must be unique Fourier harmonics of the radial box")
    # The ky=0 plane is its own Hermitian partner; validate symmetric radial grids.
    opposite = np.array([np.argmin(abs(p + value)) for value in p])
    if np.allclose(p[opposite], -p) and not np.allclose(
        c[:, 0], c[opposite, 0].conj(), rtol=1e-6, atol=1e-12
    ):
        raise ValueError("ky=0 coefficients violate radial Hermitian symmetry")
    nx = int(nx or (2 * int(np.max(np.abs(p))) + 1))
    ny = int(ny or (2 * ky.size - 1))
    if nx <= 2 * np.max(np.abs(p)) or ny < 2 * ky.size - 1:
        raise ValueError("real-space grid would alias saved harmonics")
    x, y = np.arange(nx) * length_x / nx, np.arange(ny) * (2 * np.pi / dky) / ny
    weights = np.ones(ky.size)
    weights[0] = 0.5
    if component == "zonal":
        weights[1:] = 0
    if component == "nonzonal":
        weights[0] = 0
    values = np.real(
        np.exp(1j * np.outer(x, kx)) @ (c * weights) @ np.exp(-1j * np.outer(ky, y))
    )
    return SpatialField(
        x,
        y,
        values,
        {
            "component": component,
            "length_unit": "rho_s",
            "normalization": "native CGYRO",
            "fourier_sign": "+kx*x-ky*y",
            "ky_zero_weight": 0.5,
            "domain": "local saved-theta flux-tube plane",
        },
    )


def zonal_fraction(phi, ky, *, weights=None, ky_tolerance=1e-12, weighting=None):
    """I_Z/(I_Z+I_T); all-zero intensity is undefined, not zero.

    Parameters
    ----------
    phi : array-like
        Complex potential coefficients (kx, ky, time) [native potential].
    ky : array-like
        Binormal wavenumbers, including zero [inverse perpendicular length].
    weights : array-like, optional
        Nonnegative intensity weights broadcast over kx and ky [-].
    ky_tolerance : float, optional
        Numerical threshold identifying the zero binormal wavenumber [ky unit].
    weighting : str, optional
        Explicit interpretation of the supplied intensity weights [-].

    Returns
    -------
    numpy.ndarray
        Zonal fraction at each time; all-zero intensity is NaN [-].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    I_Z/(I_Z+I_T); all-zero intensity is undefined, not zero.

    Provenance
    ----------
    .. [1] Canonical mathematical definition in #2028; uses the retained-source and weighting contract of #2025.
    """
    result = zonal_turbulence_intensity(
        phi, ky, weights=weights, ky_tolerance=ky_tolerance, weighting=weighting
    )
    return result.fraction


def spectral_integral(values, *, representation="contribution", edges=None, axis=-1):
    """Integrate signed bin contributions or densities with explicit bin edges.

    Parameters
    ----------
    values : array-like
        Signed spectral contributions or aligned time samples [quantity].
    representation : str
        Contribution means bin-integrated values; density requires edges [-].
    edges : array-like, optional
        Audited increasing spectral bin edges [wavenumber].
    axis : int, optional
        Axis containing spectral bins or additive field contributions [-].

    Returns
    -------
    numpy.ndarray
        Signed integral with the selected spectral axis removed [quantity].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    Integrate signed bin contributions or densities with explicit bin edges.

    Provenance
    ----------
    .. [1] Canonical mathematical definition in #2028; uses the retained-source and weighting contract of #2025.
    """
    values = np.asarray(values, float)
    if representation == "contribution":
        return np.sum(values, axis=axis)
    if representation != "density" or edges is None:
        raise ValueError("density requires audited bin edges")
    edges = np.asarray(edges, float)
    if (
        edges.size != values.shape[axis] + 1
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("increasing finite edges must bracket every bin")
    shape = [1] * values.ndim
    shape[axis] = len(edges) - 1
    return np.sum(values * np.diff(edges).reshape(shape), axis=axis)


def rebin_contributions(values, edges, new_edges):
    """Conserve signed flux using piecewise-uniform density within audited bins.

    Parameters
    ----------
    values : array-like
        Signed spectral contributions or aligned time samples [quantity].
    edges : array-like, optional
        Audited increasing spectral bin edges [wavenumber].
    new_edges : array-like
        Destination bin edges spanning exactly the source interval [wavenumber].

    Returns
    -------
    numpy.ndarray
        Conservative destination-bin contributions under piecewise-uniform density [quantity].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    Conserve signed flux using piecewise-uniform density within audited bins.

    The destination must cover exactly the same interval; no extrapolation.

    Provenance
    ----------
    .. [1] Canonical mathematical definition in #2028; uses the retained-source and weighting contract of #2025.
    """
    values, edges, new = map(lambda x: np.asarray(x, float), (values, edges, new_edges))
    if (
        values.shape[-1] + 1 != len(edges)
        or len(new) < 2
        or not np.isfinite(edges).all()
        or not np.isfinite(new).all()
        or np.any(np.diff(edges) <= 0)
        or np.any(np.diff(new) <= 0)
        or not np.allclose([edges[0], edges[-1]], [new[0], new[-1]], rtol=0, atol=1e-12)
    ):
        raise ValueError("audited increasing edges with identical coverage required")
    overlap = np.maximum(
        0,
        np.minimum(new[1:, None], edges[None, 1:])
        - np.maximum(new[:-1, None], edges[None, :-1]),
    )
    return (values / np.diff(edges)) @ overlap.T


def field_transport_contributions(values, *, axis=0, cancellation_rtol=1e-8):
    """Signed field totals and fractions; cancellation is explicit, never clipped.

    Parameters
    ----------
    values : array-like
        Signed spectral contributions or aligned time samples [quantity].
    axis : int, optional
        Axis containing spectral bins or additive field contributions [-].
    cancellation_rtol : float, optional
        Numerical threshold for total versus absolute summed magnitude [-].

    Returns
    -------
    dict
        Signed total, unbounded signed fractions and cancellation mask [quantity; fractions -].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    Signed field totals and fractions; cancellation is explicit, never clipped.

    Provenance
    ----------
    .. [1] Canonical mathematical definition in #2028; uses the retained-source and weighting contract of #2025.
    """
    values = np.asarray(values, float)
    total = np.sum(values, axis=axis)
    magnitude = np.sum(np.abs(values), axis=axis)
    cancelled = np.abs(total) <= cancellation_rtol * magnitude
    good = np.isfinite(total) & (total != 0) & ~cancelled
    fractions = np.divide(values, np.expand_dims(np.where(good, total, np.nan), axis))
    return {
        "total": total,
        "fractions": fractions,
        "cancellation": cancelled,
        "definition": "signed contribution / signed total",
    }


def reduce_time_series(
    time, values, *, window, blocks=4, max_gap=None, time_unit="a/c_s"
):
    """Time-weighted window reduction with explicit coverage and independent evidence.

    Parameters
    ----------
    time : array-like
        Finite strictly increasing simulation times [time_unit].
    values : array-like
        Signed spectral contributions or aligned time samples [quantity].
    window : tuple of float
        Selected closed integration interval; no extrapolation [time_unit].
    blocks : int, optional
        Equal-duration blocks; at least two observed samples per block [-].
    max_gap : float, optional
        Largest integrated interval; default three median sample spacings [time_unit].
    time_unit : str, optional
        Time-coordinate unit recorded with the result; default a/c_s [-].

    Returns
    -------
    dict
        Mean, coverage, block evidence, drift and independent qualifications [quantity; time_unit].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    Time-weighted window reduction with explicit coverage and independent evidence.

    Integrates linear segments only between finite adjacent observed samples.
    Without an explicit max_gap, gaps above 3*median cadence are excluded.
    No extrapolation or interpolation across missing samples. Blocks tile time,
    and their standard error is conditional on independence, never a confidence interval.

    Provenance
    ----------
    .. [1] Existing nonlinear workflow batch-means implementation (#1484/#1820); piecewise-linear quadrature with explicit gaps; #2028.
    """
    t, y = np.asarray(time, float), np.asarray(values, float)
    lo, hi = map(float, window)
    if (
        t.ndim != 1
        or y.shape != t.shape
        or len(t) < 2
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or not np.isfinite([lo, hi]).all()
        or hi <= lo
        or not isinstance(blocks, int)
        or blocks < 2
    ):
        raise ValueError(
            "increasing time, aligned trace, positive window and >=2 blocks required"
        )
    gap = 3 * float(np.median(np.diff(t))) if max_gap is None else float(max_gap)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("max_gap must be positive")
    ok = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (np.diff(t) <= gap)

    def integrate(a, b):
        left, right = np.maximum(t[:-1], a), np.minimum(t[1:], b)
        valid = ok & (right > left)
        dt = np.diff(t)[valid]
        l = left[valid]
        r = right[valid]
        slope = (y[1:][valid] - y[:-1][valid]) / dt
        yl = y[:-1][valid] + slope * (l - t[:-1][valid])
        yr = y[:-1][valid] + slope * (r - t[:-1][valid])
        duration = float(np.sum(r - l))
        return (
            float(np.sum((yl + yr) * 0.5 * (r - l)) / duration) if duration else None
        ), duration

    mean, duration = integrate(lo, hi)
    edges = np.linspace(lo, hi, blocks + 1)
    bm = [integrate(a, b) for a, b in zip(edges[:-1], edges[1:])]
    block_samples = [
        int(np.count_nonzero((t >= a) & (t <= b) & np.isfinite(y)))
        for a, b in zip(edges[:-1], edges[1:])
    ]
    n_samples = int(np.count_nonzero((t >= lo) & (t <= hi) & np.isfinite(y)))
    complete = all(
        m is not None and count >= 2 and np.isclose(d, (hi - lo) / blocks)
        for (m, d), count in zip(bm, block_samples)
    )
    stderr = (
        float(np.std([m for m, d in bm], ddof=1) / np.sqrt(blocks))
        if complete
        else None
    )
    halves = [integrate(lo, (lo + hi) / 2)[0], integrate((lo + hi) / 2, hi)[0]]
    drift = None if None in halves else halves[1] - halves[0]
    scale = float(np.nanmax(np.abs(y))) if np.isfinite(y).any() else 0.0
    relative = (
        None
        if mean is None or abs(mean) <= 1e-8 * scale or mean == 0
        else abs(drift) / abs(mean)
        if drift is not None
        else None
    )
    return {
        "mean": mean,
        "window": [lo, hi],
        "time_unit": time_unit,
        "coverage": duration / (hi - lo),
        "max_gap": gap,
        "gap_policy": "exclude segments above max_gap and missing endpoints",
        "n_samples": n_samples,
        "blocks": blocks,
        "block_length": (hi - lo) / blocks,
        "block_samples": block_samples,
        "block_means": [m for m, d in bm],
        "block_coverage": [d / ((hi - lo) / blocks) for m, d in bm],
        "standard_error": stderr,
        "half_window_drift": drift,
        "relative_drift": relative,
        "qualification": {
            "data_sufficiency": "sufficient"
            if duration > 0 and n_samples >= 4
            else "insufficient_data",
            "stationarity": "screen_passed"
            if n_samples >= 4
            and duration >= hi - lo - 1e-10
            and relative is not None
            and relative < 0.2
            else "unverified",
            "uncertainty": "estimated_with_block_independence_assumption"
            if stderr is not None
            else "insufficient_data",
            "numerical_convergence": "unverified",
            "physical_applicability": "unverified",
        },
        "method": "piecewise-linear time integral; equal-duration batch means; relative drift screen <0.2 (heuristic)",
    }


def saturated_intensity(ql_weights, potential_intensity):
    """TGLF moment intensity = QL intensity weight * saturated phi intensity.

    Parameters
    ----------
    ql_weights : array-like
        Native TGLF quasilinear moment weights (species, ky, mode, moment) [native weight].
    potential_intensity : array-like
        Saturated potential intensity (ky, mode) [native phi squared].

    Returns
    -------
    numpy.ndarray
        Saturated moment intensity on the original four axes [native moment intensity].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    TGLF moment intensity = QL intensity weight * saturated phi intensity.

    Weights: (species, ky, mode, moment); phi: (ky, mode). This is not a
    transport-flux reconstruction: transport also needs its audited quadrature.

    Provenance
    ----------
    .. [1] GACODE b493397 tglf_inout.f90 field/intensity writers; native SAT0/SAT2 fixture identities; #2025/#2028.
    """
    q, p = np.asarray(ql_weights, float), np.asarray(potential_intensity, float)
    if q.ndim != 4 or p.shape != q.shape[1:3]:
        raise ValueError("QL weights and saturated phi intensity axes disagree")
    return q * p[None, :, :, None]


def radial_correlation_from_spectrum(kx, spectrum, *, box_length, samples=513):
    """Cosine autocorrelation 1/e crossing and integral scale; unresolved is a bound.

    Parameters
    ----------
    kx : array-like
        Radial Fourier wavenumbers [inverse perpendicular length].
    spectrum : array-like
        Nonnegative radial power spectrum on kx [intensity].
    box_length : float
        Positive periodic radial box extent [perpendicular length].
    samples : int, optional
        Number of separation samples across half the box; default 513 [-].

    Returns
    -------
    dict
        1/e crossing, lower bound if unresolved, and first-zero integral scale [perpendicular length].

    Applicability
    -------------
    Machine-independent. Requires the declared spectral convention, retained axes and units.
    These are numerical reductions of solver data, not independent evidence of physical validity.

    Notes
    -----
    Cosine autocorrelation 1/e crossing and integral scale; unresolved is a bound.

    Array-only counterpart of the CGYRO locality reader, with its existing
    half-box search and first-zero integral definition preserved.

    Provenance
    ----------
    .. [1] Existing CGYRO locality algorithm (#1354/#1484), spectrum-based cosine autocorrelation; #2028.
    """
    kx, power = np.asarray(kx, float), np.asarray(spectrum, float)
    length = float(box_length)
    if (
        kx.ndim != 1
        or power.shape != kx.shape
        or not np.isfinite(kx).all()
        or not np.isfinite(power).all()
        or np.any(power < 0)
        or not np.isfinite(length)
        or length <= 0
        or samples < 2
    ):
        raise ValueError("finite kx/nonnegative spectrum and positive box required")
    if not np.any(power > 0):
        return {"reason": "zero turbulent amplitude in the window"}
    dx = np.linspace(0, 0.5 * length, samples)
    correlation = (power[None, :] * np.cos(np.outer(dx, kx))).sum(axis=1)
    correlation /= correlation[0]
    below = np.flatnonzero(correlation <= np.exp(-1.0))
    crossing = None
    if below.size:
        j = int(below[0])
        x0, x1, c0, c1 = dx[j - 1], dx[j], correlation[j - 1], correlation[j]
        crossing = float(x0 + (np.exp(-1.0) - c0) * (x1 - x0) / (c1 - c0))
    zero = np.flatnonzero(correlation <= 0)
    stop = int(zero[0]) if zero.size else dx.size
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return {
        "l_corr": crossing,
        "l_corr_lower_bound": None if crossing is not None else 0.5 * length,
        "l_integral": float(integrate(correlation[:stop], dx[:stop])),
    }


def time_window_average(time, values, *, window, max_gap=None):
    """Average aligned scalar or multi-channel histories with the shared gap policy.

    Parameters
    ----------
    time : array-like
        Strictly increasing sample times [time].
    values : array-like
        Histories with time on the final axis [quantity].
    window : tuple of float
        Selected integration bounds in the input time unit [time].
    max_gap : float, optional
        Largest integrated interval; default three median sample spacings [time].

    Returns
    -------
    numpy.ndarray
        Means with the time axis removed; unavailable integrals are NaN [quantity].

    Applicability
    -------------
    Machine-independent. Uses observed adjacent samples only, with the same
    no-extrapolation and missing-interval policy as reduce_time_series.

    Provenance
    ----------
    .. [1] Shared piecewise-linear reduction contract in #2028.
    """
    t, y = np.asarray(time, float), np.asarray(values, float)
    if y.ndim < 1 or y.shape[-1] != t.size:
        raise ValueError("histories must end in the simulation time axis")
    means = [
        reduce_time_series(t, row, window=window, max_gap=max_gap, blocks=2)["mean"]
        for row in y.reshape(-1, len(t))
    ]
    return np.asarray([np.nan if v is None else v for v in means]).reshape(y.shape[:-1])


# Same public scientific family; the small implementation module prevents plot
# recipes from owning scientific compatibility checks. No native/plot imports.
from .gyrokinetic_comparison import (ComparisonAssessment, assess_comparison,
    select_time, prepare_diagnostic_traces, local_result_signature)
