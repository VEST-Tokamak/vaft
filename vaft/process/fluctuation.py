"""Diagnostic-independent spectral analysis of scalar fluctuation time series.

Every routine here takes a plain time-and-data pair and knows nothing about the
diagnostic that produced it.  The same functions serve magnetic pickup coils,
interferometry, soft X-ray channels, Langmuir probes and optical intensity, so
mapping modules must not grow their own spectral implementations.

Notation
--------
t        : time                                              [s]
f        : frequency                                        [Hz]
S(f)     : one-sided power spectral density         [signal^2/Hz]
alpha    : power-law spectral index, S ~ f^alpha              [-]
f_break  : frequency separating two power-law regimes       [Hz]
S_xy(f)  : one-sided cross-spectral density of x and y  [x*y/Hz]
gamma^2  : magnitude-squared coherence |S_xy|^2/(S_xx S_yy)   [-]

Conventions
-----------
**Transfer functions stay outside this module.**  These routines analyse
whatever quantity they are handed and never correct for a diagnostic's
transfer function.  That matters most for a magnetic pickup coil, which
measures a time derivative::

    V(t) proportional to dB/dt   =>   S_dBdt(f) = (2 pi f)**2 * S_B(f)

so a field spectrum going as ``f**alpha`` gives a derivative-signal spectrum
going as ``f**(alpha + 2)``.  **A spectral index fitted to raw pickup voltage
is not the magnetic-field spectral index.**  Integrate and calibrate first --
:func:`vaft.process.magnetics.b_field_pol_probe_field` is the canonical VEST
path -- and pass the resulting field in.  Nothing here will do it for you, by
design.

**No physical interpretation.**  Nothing classifies a fitted slope, names a
spectral regime, or attaches meaning to a break frequency.  Reference slopes
and characteristic frequencies are the caller's, supplied at the plotting
layer; this module ships no slope constants of its own.

**Nothing is inferred that the caller could state.**  A fit range is always
explicit, so a reported index always belongs to a band someone chose; and the
two-regime mode is selected by which argument is supplied, never guessed.

Provenance
----------
.. [1] Welch's method and the short-time Fourier transform as implemented by
   :mod:`scipy.signal`, which every routine here delegates to.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from scipy import signal as scipy_signal

__all__ = [
    "CrossSpectrum",
    "FluctuationSpectrogram",
    "FluctuationSpectrum",
    "FrequencyTrack",
    "SpectralBreak",
    "SpectralFit",
    "analyze_fluctuation_spectrum",
    "compute_band_power",
    "compute_psd",
    "compute_spectrogram",
    "cross_spectrum",
    "find_spectral_break",
    "fit_power_law_spectrum",
    "track_dominant_frequency",
]

#: Fractional sample-spacing scatter tolerated before a time axis is rejected as
#: materially nonuniform.  ``max|dt - median(dt)| / median(dt)`` must stay below
#: this value for Welch/STFT to be meaningful.
NONUNIFORM_TOLERANCE = 1e-3

#: Fewest points a power-law fit needs: two to define a line, one to make
#: ``r_squared`` and the residuals meaningful.
MIN_FIT_POINTS = 3


@dataclass(frozen=True)
class SpectralFit:
    """One power-law fit ``S(f) = 10**intercept * f**alpha`` over a frequency range."""

    alpha: float
    intercept: float
    r_squared: float
    frequency_range: tuple[float, float]
    n_points: int
    stderr: float
    residuals: np.ndarray


@dataclass(frozen=True)
class SpectralBreak:
    """Two-regime power-law fit with a break frequency separating the regimes.

    ``mode`` records how the break was obtained: ``"imposed"`` when the caller
    supplied the boundary, ``"search"`` when it was estimated from the data over
    an explicit search interval.  A detected break is a numerical feature and is
    never identified here with any plasma scale.
    """

    alpha_low: float
    alpha_high: float
    break_frequency: float
    r_squared: float
    mode: str
    low_fit: SpectralFit
    high_fit: SpectralFit


@dataclass(frozen=True)
class FluctuationSpectrum:
    """A PSD plus whatever fits, break and band powers the caller asked for."""

    frequency: np.ndarray
    psd: np.ndarray
    sample_rate: float
    method: str = "welch"
    units: str = "signal**2/Hz"
    fits: tuple[SpectralFit, ...] = ()
    spectral_break: SpectralBreak | None = None
    band_power: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FluctuationSpectrogram:
    """Time-frequency magnitude map.

    Field names match :class:`vaft.process.magnetics.MirnovSpectrogramResult`, so
    ``vaft.plot.models.Spectrogram.from_result`` accepts this directly.
    """

    time: np.ndarray
    frequency: np.ndarray
    magnitude: np.ndarray


def _validate_time_axis(time: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(time, data, sample_rate)`` after rejecting unusable time axes.

    Raises rather than guessing: a non-monotonic axis, a length mismatch, or
    materially nonuniform sampling all make a Welch PSD meaningless, and silently
    assuming uniform spacing would report a confidently wrong frequency axis.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(data, dtype=float)
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError(
            f"time and data must be 1D; got shapes {time.shape} and {values.shape}"
        )
    if time.size != values.size:
        raise ValueError(
            f"time and data must have equal length; got {time.size} and {values.size}"
        )
    if time.size < 2:
        raise ValueError("time must contain at least two samples to define a sample rate.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time must be finite; got NaN or inf entries.")

    steps = np.diff(time)
    if np.any(steps <= 0):
        raise ValueError(
            "time must be strictly increasing; got a non-monotonic or repeated axis."
        )

    median_step = float(np.median(steps))
    deviation = float(np.max(np.abs(steps - median_step))) / median_step
    if deviation > NONUNIFORM_TOLERANCE:
        raise ValueError(
            "time is materially nonuniform: sample spacing varies by "
            f"{deviation:.3%} of the median step, above the {NONUNIFORM_TOLERANCE:.3%} "
            "tolerance. Resample onto a uniform grid before spectral analysis; these "
            "estimators do not interpolate for you."
        )
    return time, values, 1.0 / median_step


def _resolve_sample_rate(
    time: np.ndarray, data: np.ndarray, sample_rate: float | None
) -> tuple[np.ndarray, float]:
    """Validate the axis and honour an explicit ``sample_rate`` override."""
    _, values, derived = _validate_time_axis(time, data)
    if sample_rate is None:
        return values, derived
    sample_rate = float(sample_rate)
    if sample_rate <= 0 or not np.isfinite(sample_rate):
        raise ValueError(f"sample_rate must be a positive finite value; got {sample_rate!r}")
    return values, sample_rate


def compute_psd(
    time,
    data,
    *,
    sample_rate: float | None = None,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | bool = "constant",
    units: str = "signal**2/Hz",
) -> FluctuationSpectrum:
    """One-sided power spectral density by Welch's method.

    Parameters
    ----------
    time : array_like
        Strictly increasing, uniformly sampled time axis [s].
    data : array_like
        Signal samples, same length [any].
    sample_rate : float, optional
        Overrides the rate derived from the time axis [Hz].
    window : str, optional
        Any window name SciPy accepts [-].
    nperseg : int, optional
        Segment length in samples [-].
    noverlap : int, optional
        Overlap in samples [-].
    detrend : str or bool, optional
        Passed through to the underlying estimator [-].
    units : str, optional
        Free-text unit label recorded on the result [-].

    Returns
    -------
    FluctuationSpectrum
        The frequency axis in hertz, the density in the signal's own squared unit
        per hertz, and the parameters needed to reproduce them [-].

    Raises
    ------
    ValueError
        The time axis is not uniform within the module's tolerance.

    Convention
    ----------
    One-sided, so the power at each frequency already accounts for its negative
    counterpart. The input is whatever quantity the caller wants a spectrum of and
    **no transfer-function correction is applied**; see the module's conventions
    for what that means for a pickup coil.

    Defaults
    --------
    The segment length and overlap are numerical conveniences, deferred to SciPy's
    own choices rather than fixed here, so a caller who has not thought about
    resolution gets that library's behaviour rather than one invented for VAFT.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Welch's method assumes a uniformly sampled, stationary signal. A materially
    nonuniform time axis is rejected rather than interpolated, because
    interpolating one silently changes the spectrum it is being asked about.

    Provenance
    ----------
    .. [1] Welch's method as implemented by :func:`scipy.signal.welch`.
    """
    values, fs = _resolve_sample_rate(time, data, sample_rate)
    kwargs = {"fs": fs, "window": window, "detrend": detrend}
    if nperseg is not None:
        kwargs["nperseg"] = int(nperseg)
    if noverlap is not None:
        kwargs["noverlap"] = int(noverlap)
    frequency, psd = scipy_signal.welch(values, **kwargs)
    return FluctuationSpectrum(
        frequency=frequency,
        psd=psd,
        sample_rate=fs,
        method="welch",
        units=units,
    )


def _fit_range_mask(frequency: np.ndarray, f_range: Sequence[float]) -> np.ndarray:
    """Select strictly positive frequencies inside the closed interval ``f_range``."""
    low, high = (float(f_range[0]), float(f_range[1]))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"f_range bounds must be finite; got ({low!r}, {high!r})")
    if low >= high:
        raise ValueError(f"f_range must be increasing; got ({low}, {high})")
    # f=0 and non-positive PSD samples have no logarithm, so a log-log fit
    # cannot use them regardless of the requested interval.
    return (frequency >= low) & (frequency <= high) & (frequency > 0)


def fit_power_law_spectrum(frequency, psd, *, f_range: Sequence[float]) -> SpectralFit:
    """Fit a power law to a spectrum over an explicitly chosen band.

    Parameters
    ----------
    frequency : array_like
        Frequency axis [Hz].
    psd : array_like
        Power spectral density on that axis [any].
    f_range : sequence of float
        The band to fit over, as ``(f_low, f_high)`` [Hz].

    Returns
    -------
    SpectralFit
        The index, the amplitude, the goodness of fit and the band used [-].

    Raises
    ------
    ValueError
        Fewer than the minimum number of usable points fall inside the band.

    Convention
    ----------
    Ordinary least squares in log-log space, so the fit is on the exponent
    directly. **There is no default range and no automatic range selection**: a
    reported index therefore always belongs to a band the caller chose
    deliberately, rather than to whatever the algorithm happened to like.

    Nothing here names the regime that index belongs to.

    Defaults
    --------
    The minimum point count is a numerical convenience: two points define a line,
    and the third is what makes the goodness of fit and the residuals mean
    anything.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Samples with a non-positive density are excluded, since their logarithm does
    not exist. A log-log fit weights decades equally, so a band spanning many
    decades is dominated by the sparse high-frequency end.

    Provenance
    ----------
    .. [1] Ordinary least squares in log-log space; the module's conventions on
       why the band is never chosen for the caller.
    """
    frequency = np.asarray(frequency, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if frequency.shape != psd.shape:
        raise ValueError(
            f"frequency and psd must have the same shape; got {frequency.shape} and {psd.shape}"
        )

    mask = _fit_range_mask(frequency, f_range) & (psd > 0) & np.isfinite(psd)
    n_points = int(np.count_nonzero(mask))
    if n_points < MIN_FIT_POINTS:
        raise ValueError(
            f"insufficient fit range: {n_points} usable point(s) with positive PSD in "
            f"{tuple(float(v) for v in f_range)} Hz, need at least {MIN_FIT_POINTS}. "
            "Widen f_range, lengthen the signal, or reduce nperseg for finer "
            "frequency resolution."
        )

    log_f = np.log10(frequency[mask])
    log_s = np.log10(psd[mask])
    alpha, intercept = np.polyfit(log_f, log_s, 1)
    model = alpha * log_f + intercept
    residuals = log_s - model

    ss_residual = float(np.sum(residuals**2))
    ss_total = float(np.sum((log_s - log_s.mean()) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0

    # Standard error of the slope from the usual OLS expression; degenerate when
    # every retained point sits at one frequency, which the span check catches.
    span = float(np.sum((log_f - log_f.mean()) ** 2))
    if n_points > 2 and span > 0:
        stderr = float(np.sqrt(ss_residual / (n_points - 2) / span))
    else:
        stderr = float("nan")

    return SpectralFit(
        alpha=float(alpha),
        intercept=float(intercept),
        r_squared=float(r_squared),
        frequency_range=(float(f_range[0]), float(f_range[1])),
        n_points=n_points,
        stderr=stderr,
        residuals=residuals,
    )


def _joint_r_squared(low: SpectralFit, high: SpectralFit) -> float:
    """Combine two segment fits into one point-weighted coefficient of determination."""
    total = low.n_points + high.n_points
    if total == 0:
        return 0.0
    return (low.r_squared * low.n_points + high.r_squared * high.n_points) / total


def _joint_residual(low: SpectralFit, high: SpectralFit) -> float:
    """Total squared log-space residual of a two-segment fit.

    Every candidate break partitions the same set of points, so this total is
    directly comparable across candidates.  A point-weighted ``r_squared`` is
    not: moving the break changes how many points each segment holds, which
    biases the score toward splits that hand most points to whichever segment
    fits best, and in practice pins the answer to the edge of the search range.
    """
    return float(np.sum(low.residuals**2) + np.sum(high.residuals**2))


def find_spectral_break(
    frequency,
    psd,
    *,
    fit_range: Sequence[float],
    break_frequency: float | None = None,
    search_range: Sequence[float] | None = None,
    n_candidates: int = 64,
) -> SpectralBreak:
    """Fit a two-regime power law, below and above a break frequency.

    Parameters
    ----------
    frequency : array_like
        Frequency axis [Hz].
    psd : array_like
        Power spectral density [any].
    fit_range : sequence of float
        Outer band spanning both regimes [Hz].
    break_frequency : float, optional
        An imposed boundary, for the physics-informed mode [Hz].
    search_range : sequence of float, optional
        Interval to scan, for the data-driven mode [Hz].
    n_candidates : int, optional
        Log-spaced candidates scanned in search mode [-].

    Returns
    -------
    SpectralBreak
        The break, the two fits either side, the mode used, and the score [-].

    Raises
    ------
    ValueError
        Neither or both of the two mode arguments were supplied, or the band holds
        too few points.

    Convention
    ----------
    **The mode is chosen by which argument is supplied, never inferred.** Passing
    a break frequency imposes it and the caller owns the boundary; passing a search
    range scans for one. Supplying neither, or both, is an error rather than a
    default, because a break that looks physics-informed and was in fact fitted is
    the failure this refuses to permit.

    Candidates are scored by **total squared residual in log space**, not by a
    point-weighted goodness of fit. The latter biases the score toward splits that
    hand most points to whichever segment fits best, and in practice pins the
    answer to the edge of the search range.

    A break found this way is a numerical feature of the spectrum. **It is not
    identified with any plasma scale here.**

    Defaults
    --------
    The candidate count is a numerical convenience, dense enough that the scan's
    resolution is not the limiting error.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Two straight lines are fitted to something that may not have two regimes; a
    best break is always returned in search mode, whether or not one exists.
    Inherits the log-space caveats of the single-regime fit.

    Provenance
    ----------
    .. [1] :func:`fit_power_law_spectrum`, applied either side of the break.
    """
    if (break_frequency is None) == (search_range is None):
        raise ValueError(
            "pass exactly one of break_frequency= (physics-informed) or "
            "search_range= (data-driven); the mode is never chosen for you."
        )

    frequency = np.asarray(frequency, dtype=float)
    psd = np.asarray(psd, dtype=float)
    low_edge, high_edge = (float(fit_range[0]), float(fit_range[1]))

    def _fit_pair(boundary: float) -> tuple[SpectralFit, SpectralFit]:
        return (
            fit_power_law_spectrum(frequency, psd, f_range=(low_edge, boundary)),
            fit_power_law_spectrum(frequency, psd, f_range=(boundary, high_edge)),
        )

    if break_frequency is not None:
        boundary = float(break_frequency)
        if not low_edge < boundary < high_edge:
            raise ValueError(
                f"break_frequency {boundary} Hz must lie strictly inside fit_range "
                f"({low_edge}, {high_edge})."
            )
        low_fit, high_fit = _fit_pair(boundary)
        mode = "imposed"
    else:
        search_low, search_high = (float(search_range[0]), float(search_range[1]))
        if not low_edge <= search_low < search_high <= high_edge:
            raise ValueError(
                f"search_range ({search_low}, {search_high}) must lie within fit_range "
                f"({low_edge}, {high_edge})."
            )
        candidates = np.logspace(
            np.log10(search_low), np.log10(search_high), int(n_candidates)
        )
        best: tuple[float, SpectralFit, SpectralFit] | None = None
        failures: list[str] = []
        for candidate in candidates:
            try:
                low_fit, high_fit = _fit_pair(float(candidate))
            except ValueError as error:  # too few points on one side of this split
                failures.append(str(error))
                continue
            score = _joint_residual(low_fit, high_fit)
            if best is None or score < best[0]:
                best = (score, low_fit, high_fit)
        if best is None:
            raise ValueError(
                "no candidate break in search_range left enough points on both sides "
                f"to fit. Last reason: {failures[-1] if failures else 'none tried'}"
            )
        _, low_fit, high_fit = best
        boundary = low_fit.frequency_range[1]
        mode = "search"

    return SpectralBreak(
        alpha_low=low_fit.alpha,
        alpha_high=high_fit.alpha,
        break_frequency=float(boundary),
        r_squared=_joint_r_squared(low_fit, high_fit),
        mode=mode,
        low_fit=low_fit,
        high_fit=high_fit,
    )


def compute_band_power(
    frequency,
    psd,
    bands: Mapping[str, Sequence[float]],
    *,
    ratios: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Integrate spectral power over named frequency bands, and optionally their ratios.

    Parameters
    ----------
    frequency : array_like
        Frequency axis [Hz].
    psd : array_like
        Power spectral density [any].
    bands : mapping of str to sequence of float
        Named ``(f_low, f_high)`` pairs; the names are the caller's [Hz].
    ratios : mapping, optional
        Named pairs of band names whose power ratio to report [-].

    Returns
    -------
    dict
        Power per named band, in the signal's own squared unit, and any requested
        ratios, dimensionless [-].

    Convention
    ----------
    **Band edges are closed**, so two adjacent bands sharing an edge both include
    the sample sitting on it. A band containing fewer than two samples integrates
    to zero rather than raising, and a ratio with a zero denominator is NaN rather
    than infinite, so a caller sweeping many channels gets a comparable result
    from every one.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Integration is over the samples present, so a band narrower than the frequency
    resolution reports zero rather than an error. Nothing normalizes by bandwidth;
    these are powers, not densities.

    Provenance
    ----------
    .. [1] Numerical integration of the density returned by :func:`compute_psd`.
    """
    frequency = np.asarray(frequency, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if frequency.shape != psd.shape:
        raise ValueError(
            f"frequency and psd must have the same shape; got {frequency.shape} and {psd.shape}"
        )

    powers: dict[str, float] = {}
    for name, edges in bands.items():
        low, high = (float(edges[0]), float(edges[1]))
        if low >= high:
            raise ValueError(f"band {name!r} must be increasing; got ({low}, {high})")
        mask = (frequency >= low) & (frequency <= high)
        if np.count_nonzero(mask) < 2:
            powers[str(name)] = 0.0
            continue
        powers[str(name)] = float(np.trapezoid(psd[mask], frequency[mask]))

    for name, pair in (ratios or {}).items():
        numerator, denominator = (str(pair[0]), str(pair[1]))
        for key in (numerator, denominator):
            if key not in powers:
                raise KeyError(
                    f"ratio {name!r} refers to unknown band {key!r}; "
                    f"defined bands are {sorted(powers)}"
                )
        below = powers[denominator]
        powers[str(name)] = powers[numerator] / below if below else float("nan")

    return powers


def compute_spectrogram(
    time,
    data,
    *,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    window_duration: float | None = None,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> FluctuationSpectrogram:
    """Short-time Fourier magnitude map of a fluctuation signal.

    Parameters
    ----------
    time : array_like
        Uniformly sampled time axis [s].
    data : array_like
        Signal samples [any].
    sample_rate : float, optional
        Overrides the rate derived from the time axis [Hz].
    nperseg : int, optional
        Window length in samples [-].
    window_duration : float, optional
        Window length as a duration, converted to samples [s].
    overlap : float, optional
        Fractional overlap between windows [-].
    window : str, optional
        Window name [-].
    detrend : str or bool, optional
        Passed through to the underlying transform [-].

    Returns
    -------
    FluctuationSpectrogram
        The time and frequency axes and the magnitude map [-].

    Convention
    ----------
    **The time axis is returned in the caller's own absolute base**, offset back
    from the transform's window-relative one, so a feature can be read against the
    shot clock without correction.

    The field names match the magnetics module's own spectrogram result, so the
    plotting layer accepts either without knowing which produced it.

    Defaults
    --------
    The overlap is a numerical convenience. The window may be given in samples or
    as a duration; the duration form is the portable one, since it means the same
    thing at a different sample rate.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The window sets the trade-off between time and frequency resolution and
    nothing here chooses it for the caller. A window longer than the record
    returns an empty result rather than raising, so callers sweeping many channels
    get deterministic behaviour at the edges.

    Provenance
    ----------
    .. [1] The short-time Fourier transform as implemented by
       :func:`scipy.signal.spectrogram`.
    """
    if nperseg is not None and window_duration is not None:
        raise ValueError("pass either nperseg= or window_duration=, not both.")
    values, fs = _resolve_sample_rate(time, data, sample_rate)
    start = float(np.asarray(time, dtype=float)[0])

    if window_duration is not None:
        if float(window_duration) <= 0:
            raise ValueError(f"window_duration must be positive; got {window_duration!r}")
        segment = int(round(float(window_duration) * fs))
        if segment < 2:
            raise ValueError(
                f"window_duration {window_duration} s spans {segment} sample(s) at "
                f"{fs:.6g} Hz; it must cover at least two."
            )
    else:
        segment = int(nperseg) if nperseg is not None else min(256, values.size)
        if segment < 2:
            raise ValueError(f"nperseg must be at least 2; got {segment}")

    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError(f"overlap must be a fraction in [0, 1); got {overlap!r}")

    if values.size < segment:
        frequencies = np.fft.rfftfreq(segment, d=1.0 / fs)
        return FluctuationSpectrogram(
            time=np.empty(0, dtype=float),
            frequency=frequencies,
            magnitude=np.empty((frequencies.size, 0), dtype=float),
        )

    frequencies, times, magnitude = scipy_signal.spectrogram(
        values,
        fs=fs,
        window=window,
        nperseg=segment,
        noverlap=int(round(segment * float(overlap))),
        detrend=detrend,
        mode="magnitude",
    )
    return FluctuationSpectrogram(
        time=times + start,
        frequency=frequencies,
        magnitude=magnitude,
    )


def analyze_fluctuation_spectrum(
    time,
    data,
    *,
    fit_ranges: Sequence[Sequence[float]] = (),
    bands: Mapping[str, Sequence[float]] | None = None,
    ratios: Mapping[str, Sequence[str]] | None = None,
    break_frequency: float | None = None,
    search_range: Sequence[float] | None = None,
    break_fit_range: Sequence[float] | None = None,
    **psd_options,
) -> FluctuationSpectrum:
    """Run the spectral chain on one signal: density, then whatever else was asked for.

    Parameters
    ----------
    time : array_like
        Uniformly sampled time axis [s].
    data : array_like
        Signal samples [any].
    fit_ranges : sequence, optional
        Bands to fit a power law over, each ``(f_low, f_high)`` [Hz].
    bands : mapping, optional
        Named bands whose power to integrate [Hz].
    ratios : mapping, optional
        Named band-power ratios to report [-].
    break_frequency : float, optional
        An imposed two-regime boundary [Hz].
    search_range : sequence of float, optional
        Interval to scan for a break instead [Hz].
    break_fit_range : sequence of float, optional
        Outer band for the two-regime fit.  Further keyword arguments are
        passed through to the density estimate [Hz].

    Returns
    -------
    dict
        The spectrum, and an entry for each optional analysis that was requested
        [-].

    Processing steps
    ----------------
    1. Estimate the power spectral density.
    2. Fit a power law over each requested band.
    3. Fit a two-regime break, if either mode argument was given.
    4. Integrate the requested band powers and ratios.

    Convention
    ----------
    **Everything past the density is opt-in.** Nothing is computed because it
    might be interesting, so the returned mapping holds exactly what was asked
    for. The same rules apply as to the functions this calls: fit ranges are
    explicit, the break mode is never inferred, and no result is given a physical
    name.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A convenience wrapper only. It adds no analysis of its own and inherits every
    limitation of the steps it runs.

    Provenance
    ----------
    .. [1] The functions it orchestrates, each documented on its own.
    """
    spectrum = compute_psd(time, data, **psd_options)

    fits = tuple(
        fit_power_law_spectrum(spectrum.frequency, spectrum.psd, f_range=f_range)
        for f_range in fit_ranges
    )

    spectral_break = None
    if break_frequency is not None or search_range is not None:
        if break_fit_range is None:
            raise ValueError(
                "break analysis needs break_fit_range=(f_low, f_high) spanning both regimes."
            )
        spectral_break = find_spectral_break(
            spectrum.frequency,
            spectrum.psd,
            fit_range=break_fit_range,
            break_frequency=break_frequency,
            search_range=search_range,
        )

    band_power = (
        compute_band_power(spectrum.frequency, spectrum.psd, bands, ratios=ratios)
        if bands
        else {}
    )

    return FluctuationSpectrum(
        frequency=spectrum.frequency,
        psd=spectrum.psd,
        sample_rate=spectrum.sample_rate,
        method=spectrum.method,
        units=spectrum.units,
        fits=fits,
        spectral_break=spectral_break,
        band_power=band_power,
    )


# ---------------------------------------------------------------------------
# Two records: cross-spectrum, coherence and phase (issue #1005)
# ---------------------------------------------------------------------------

#: Significance level of :attr:`CrossSpectrum.significance_95`.
COHERENCE_SIGNIFICANCE_ALPHA = 0.05


@dataclass(frozen=True)
class CrossSpectrum:
    """Cross-spectral density, coherence and phase of two records on one grid.

    ``phase`` is ``arg(csd)``, the phase of ``y`` relative to ``x``; ``coherence``
    is the magnitude-squared coherence.  ``significance_95`` is the level
    independent noise exceeds with probability 0.05 for ``n_segments`` averages.
    ``time_range``, ``sample_rate`` and ``resampled`` record the common grid the
    two records were compared on, so the overlap cut is never a silent one.
    """

    frequency: np.ndarray
    csd: np.ndarray
    coherence: np.ndarray
    phase: np.ndarray
    n_segments: int
    significance_95: float
    sample_rate: float
    time_range: tuple[float, float]
    nperseg: int
    noverlap: int
    resampled: tuple[str, ...] = ()


def _uniform_record(time, data, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate one record's axis, naming the record in the error."""
    try:
        return _validate_time_axis(time, data)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None


def _common_grid(
    time_x: np.ndarray,
    time_y: np.ndarray,
    fs_x: float,
    fs_y: float,
    sample_rate: float | None,
    common_time,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The grid both records are compared on, and which of them must be resampled.

    Identical axes are used as they are.  Otherwise the grid is ``common_time``
    when given, a uniform grid at ``sample_rate`` over the overlap when that is
    given, and otherwise the slower record's own samples inside the overlap, so
    the slower record is never interpolated and the faster one is anti-aliased
    down to it.
    """
    if common_time is not None:
        grid = np.asarray(common_time, dtype=float).reshape(-1)
        _uniform_record(grid, grid, "common_time")
        return grid, ("x", "y")

    same = time_x.size == time_y.size and np.allclose(
        time_x, time_y, rtol=0.0, atol=1e-3 / max(fs_x, fs_y)
    )
    if same and sample_rate is None:
        return time_x, ()

    start = max(float(time_x[0]), float(time_y[0]))
    stop = min(float(time_x[-1]), float(time_y[-1]))
    if not start < stop:
        raise ValueError(
            f"the two records do not overlap in time: x spans [{time_x[0]:.6g}, "
            f"{time_x[-1]:.6g}] s and y spans [{time_y[0]:.6g}, {time_y[-1]:.6g}] s"
        )
    if sample_rate is not None:
        step = 1.0 / float(sample_rate)
        grid = start + step * np.arange(int(np.floor((stop - start) / step + 1e-9)) + 1)
        # Both records are interpolated onto it without extrapolation, so a last
        # point a rounding error past ``stop`` is dropped, not clamped.
        return grid[grid <= stop], ("x", "y")

    # The slower record keeps its own samples; ties keep x's.  The crop is strict:
    # a kept sample even a rounding error outside the other record would ask the
    # resampler to extrapolate that record, which it refuses.
    keep_x = fs_x <= fs_y
    base = time_x if keep_x else time_y
    inside = (base >= start) & (base <= stop)
    return base[inside], (("y",) if keep_x else ("x",))


def cross_spectrum(
    time_x,
    x,
    time_y,
    y,
    *,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    common_time=None,
) -> CrossSpectrum:
    """Cross-spectral density, coherence and relative phase of two fluctuation records.

    Parameters
    ----------
    time_x : array_like
        Uniformly sampled time axis of ``x`` [s].
    x : array_like
        First record, the phase reference [any].
    time_y : array_like
        Uniformly sampled time axis of ``y`` [s].
    y : array_like
        Second record [any].
    sample_rate : float, optional
        Rate of a uniform common grid laid over the overlap; both records are
        resampled onto it [Hz].
    nperseg : int, optional
        Welch segment length on the common grid [-].
    noverlap : int, optional
        Segment overlap on the common grid [-].
    window : str, optional
        Any window name SciPy accepts [-].
    detrend : str or bool, optional
        Passed through to the underlying estimator [-].
    common_time : array_like, optional
        An explicit uniform grid to compare on, overriding ``sample_rate`` [s].

    Returns
    -------
    CrossSpectrum
        Frequency axis in hertz, the complex cross-spectral density in the product
        of the two signals' units per hertz, the dimensionless coherence and its
        95 % significance level, the phase in radians, and the grid used [-].

    Raises
    ------
    ValueError
        Either axis is not uniform, or the two records do not overlap in time.

    Processing steps
    ----------------
    1. Validate both time axes as uniform and increasing.
    2. Choose the common grid: the shared axis when the two are identical;
       otherwise ``common_time``, else a uniform grid at ``sample_rate`` over the
       overlap, else the slower record's own samples inside the overlap.
    3. Put each record that is not already on that grid onto it with
       :func:`vaft.process.signal_processing.resample_to_time`, which low-passes
       first when that is a rate reduction.
    4. Estimate the two auto-spectra and the cross-spectrum by Welch's method
       with one set of segments, and form coherence and phase from them.
    5. Count the segments and set the 95 % significance level of the coherence.

    Convention
    ----------
    **The phase is that of ``y`` relative to ``x``**: ``phase = arg(S_xy)`` with
    ``S_xy = conj(X) Y``, SciPy's convention, so ``y`` lagging ``x`` by ``dt``
    reads ``-2 pi f dt`` at frequency ``f``.  Swapping the arguments flips the
    sign.  The phase is wrapped to ``(-pi, pi]``.

    Coherence is the *magnitude-squared* coherence ``|S_xy|^2 / (S_xx S_yy)``.
    Its 95 % level for ``n`` averaged segments is ``1 - 0.05**(1/(n-1))``, the
    level two independent Gaussian records exceed with probability 0.05 at any
    one frequency.

    **Different timebases are never compared silently.** The comparison is
    restricted to the overlap of the two records, and the grid, its rate, the
    overlap and which record was resampled are all returned with the result.
    Whenever the slower record keeps its own samples, only the faster one is
    changed, and it is anti-aliased before it is decimated.

    Defaults
    --------
    A numerical convenience, SciPy's own: without ``nperseg`` the segment is 256
    samples, or the whole record when that is shorter, and the overlap half a
    segment, stated here so the segment count can be reproduced.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The significance formula assumes independent segments; with the default half
    overlap of a Hann window neighbouring segments are correlated, so the true
    95 % level is somewhat higher than the one reported. A single segment has no
    significance level and reports NaN. Coherence between two diagnostics with
    different transfer functions is unaffected by either transfer function, but
    the phase is not: a pickup coil's derivative adds ``+pi/2`` to its phase.

    Provenance
    ----------
    .. [1] Welch's method for auto- and cross-spectra as implemented by
       :func:`scipy.signal.welch` and :func:`scipy.signal.csd`.
    .. [2] J. S. Bendat and A. G. Piersol, *Random Data: Analysis and
       Measurement Procedures*, 4th ed. (Wiley, 2010), section 9.2.3: the
       coherence significance level for averaged segments.
    """
    time_x, values_x, fs_x = _uniform_record(time_x, x, "x")
    time_y, values_y, fs_y = _uniform_record(time_y, y, "y")
    if sample_rate is not None and not (np.isfinite(sample_rate) and float(sample_rate) > 0):
        raise ValueError(f"sample_rate must be a positive finite value; got {sample_rate!r}")

    grid, resampled = _common_grid(time_x, time_y, fs_x, fs_y, sample_rate, common_time)
    from vaft.process.signal_processing import resample_to_time

    for name in ("x", "y"):
        source_time, values = (time_x, values_x) if name == "x" else (time_y, values_y)
        if name in resampled:
            values = resample_to_time(source_time, values, grid, extrapolate="error")
        elif source_time.size != grid.size:
            # The record that keeps its own samples is cut to the overlap, the
            # cut recorded in ``time_range``, never extended.
            tolerance = 1e-6 / max(fs_x, fs_y)
            values = values[(source_time >= grid[0] - tolerance) & (source_time <= grid[-1] + tolerance)]
        if name == "x":
            values_x = values
        else:
            values_y = values
    if grid.size < 2:
        raise ValueError("the common grid holds fewer than two samples")
    fs = 1.0 / float(np.median(np.diff(grid)))

    segment = min(int(nperseg) if nperseg is not None else 256, grid.size)
    if segment < 2:
        raise ValueError(f"nperseg must be at least 2; got {segment}")
    overlap = int(noverlap) if noverlap is not None else segment // 2
    if not 0 <= overlap < segment:
        raise ValueError(f"noverlap must lie in [0, nperseg); got {overlap} for {segment}")

    kwargs = dict(fs=fs, window=window, nperseg=segment, noverlap=overlap, detrend=detrend)
    frequency, sxx = scipy_signal.welch(values_x, **kwargs)
    _, syy = scipy_signal.welch(values_y, **kwargs)
    _, sxy = scipy_signal.csd(values_x, values_y, **kwargs)
    denominator = sxx * syy
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(denominator > 0, np.abs(sxy) ** 2 / denominator, 0.0)

    n_segments = (grid.size - overlap) // (segment - overlap)
    if n_segments > 1:
        significance = 1.0 - COHERENCE_SIGNIFICANCE_ALPHA ** (1.0 / (n_segments - 1))
    else:
        significance = float("nan")

    return CrossSpectrum(
        frequency=frequency,
        csd=sxy,
        coherence=np.clip(coherence, 0.0, 1.0),
        phase=np.angle(sxy),
        n_segments=int(n_segments),
        significance_95=float(significance),
        sample_rate=float(fs),
        time_range=(float(grid[0]), float(grid[-1])),
        nperseg=int(segment),
        noverlap=int(overlap),
        resampled=tuple(resampled),
    )


# ---------------------------------------------------------------------------
# One ridge through a time-frequency map (issue #1005)
# ---------------------------------------------------------------------------

#: Fraction of the map's in-range peak a window's ridge must exceed to count.
DEFAULT_RIDGE_FLOOR_RATIO = 0.05


class FrequencyTrack(NamedTuple):
    """A tracked ridge: one frequency and magnitude per window, NaN where none."""

    time: np.ndarray
    frequency: np.ndarray
    power: np.ndarray


def _spectrogram_arrays(spectrogram: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(time, frequency, |magnitude|)`` from a result object or a triple."""
    if isinstance(spectrogram, (tuple, list)):
        if len(spectrogram) != 3:
            raise ValueError("a spectrogram triple must be (time, frequency, magnitude)")
        time, frequency, magnitude = spectrogram
    else:
        frequency, magnitude = spectrogram.frequency, spectrogram.magnitude
        time = getattr(spectrogram, "time", None)
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    magnitude = np.abs(np.asarray(magnitude, dtype=float))
    if magnitude.ndim != 2 or magnitude.shape[0] != frequency.size:
        raise ValueError(
            "magnitude must be (frequency, time) matching the frequency axis; got "
            f"{magnitude.shape} for {frequency.size} frequencies"
        )
    if time is None:
        time = np.arange(magnitude.shape[1], dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size != magnitude.shape[1]:
        raise ValueError(
            f"time has {time.size} entries for {magnitude.shape[1]} spectrogram windows"
        )
    magnitude = np.where(np.isfinite(magnitude), magnitude, 0.0)
    return time, frequency, magnitude


def _best_path(block: np.ndarray, reach: int) -> np.ndarray:
    """Row index per column of the path maximising summed magnitude, steps <= ``reach`` rows."""
    rows, columns = block.shape
    score = block[:, 0].copy()
    back = np.zeros((rows, columns), dtype=int)
    index = np.arange(rows)
    for column in range(1, columns):
        best = np.full(rows, -np.inf)
        origin = np.zeros(rows, dtype=int)
        for shift in range(-reach, reach + 1):
            source = index - shift
            valid = (source >= 0) & (source < rows)
            candidate = np.full(rows, -np.inf)
            candidate[valid] = score[source[valid]]
            better = candidate > best
            best[better] = candidate[better]
            origin[better] = source[better]
        score = best + block[:, column]
        back[:, column] = origin
    path = np.zeros(columns, dtype=int)
    path[-1] = int(np.argmax(score))
    for column in range(columns - 1, 0, -1):
        path[column - 1] = back[path[column], column]
    return path


def track_dominant_frequency(
    spectrogram,
    *,
    search_range: Sequence[float],
    max_jump: float | None = None,
    floor_ratio: float = DEFAULT_RIDGE_FLOOR_RATIO,
) -> FrequencyTrack:
    """Follow the dominant spectral ridge of a time-frequency map, window by window.

    Parameters
    ----------
    spectrogram : Any
        An object with ``time``, ``frequency`` and a ``(frequency, time)``
        ``magnitude`` -- :class:`FluctuationSpectrogram` or a magnetics
        spectrogram result -- or the triple ``(time, frequency, magnitude)`` [-].
    search_range : sequence of float
        Lowest and highest frequency the ridge may be found at [Hz].
    max_jump : float, optional
        Largest frequency change allowed between neighbouring windows; ``None``
        takes each window's own maximum independently [Hz].
    floor_ratio : float, optional
        Fraction of the map's in-range peak a window's ridge must exceed [-].

    Returns
    -------
    FrequencyTrack
        ``(time, frequency, power)``: the window times in seconds, the ridge
        frequency in hertz and its magnitude in the map's own unit, each NaN in a
        window where no ridge clears the floor [-].

    Raises
    ------
    ValueError
        The magnitude does not match its axes, ``search_range`` is not
        increasing, or an argument is negative.

    Processing steps
    ----------------
    1. Keep the frequency bins inside ``search_range``.
    2. Set the floor at ``floor_ratio`` times the largest in-range magnitude of the
       whole map; a window whose in-range maximum does not exceed it has no ridge.
    3. Split the remaining windows into runs of consecutive windows.
    4. Within each run, choose one bin per window: the window's own maximum when
       ``max_jump`` is ``None``, otherwise the path of largest summed magnitude
       whose step between neighbouring windows never exceeds ``max_jump``.
    5. Report NaN for a window whose chosen bin does not exceed the floor.

    Convention
    ----------
    **A silent window is NaN, never its loudest noise bin.** The floor is relative
    to the map, not absolute, so it means the same thing for a voltage, a
    brightness or a camera intensity. A gap breaks continuity: the run after it
    is found afresh rather than held to where the last run ended.

    The continuity path maximises summed magnitude over the whole run rather than
    starting from the loudest pixel, so a brief loud burst far from the ridge
    cannot capture it.

    Defaults
    --------
    ``floor_ratio = 0.05`` is a numerical convenience: well above the tail of a
    Hann window's sidelobes relative to a clear mode, well below any mode worth
    tracking. Set it to zero to report every window that has any content at all.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The ridge is resolved to one frequency bin; nothing interpolates between bins.
    One ridge only: two comparable modes are not separated, and with no
    ``max_jump`` the track may alternate between them. The floor is relative, so a
    map that is noise throughout still reports its noise as a ridge.

    Provenance
    ----------
    .. [1] Ridge extraction as a maximum-energy continuous path through the
       time-frequency plane, R. Carmona, W. L. Hwang and B. Torresani,
       "Characterization of signals by the ridges of their wavelet transforms",
       *IEEE Trans. Signal Process.* **45** (1997) 2586, solved here by dynamic
       programming over discrete bins.
    .. [2] The single-argmax reference-frequency tracker of the legacy FAST-camera
       analysis, which :func:`vaft.process.camera_fluctuation.track_reference_frequency`
       now calls with ``floor_ratio=0``.
    """
    time, frequency, magnitude = _spectrogram_arrays(spectrogram)
    low, high = (float(search_range[0]), float(search_range[1]))
    if low >= high:
        raise ValueError(f"search_range must be increasing; got {search_range!r}")
    if floor_ratio < 0:
        raise ValueError(f"floor_ratio must be non-negative; got {floor_ratio!r}")
    if max_jump is not None and max_jump < 0:
        raise ValueError(f"max_jump must be non-negative; got {max_jump!r}")

    windows = magnitude.shape[1]
    tracked = np.full(windows, np.nan)
    power = np.full(windows, np.nan)
    candidates = np.nonzero((frequency >= low) & (frequency <= high))[0]
    if candidates.size == 0 or windows == 0:
        return FrequencyTrack(time, tracked, power)

    block = magnitude[candidates, :]
    floor = float(floor_ratio) * float(block.max())
    peaks = block.max(axis=0)
    active = peaks > floor

    reach = None
    if max_jump is not None and candidates.size > 1:
        step = float(np.median(np.diff(frequency[candidates])))
        reach = int(np.floor(float(max_jump) / step + 1e-9))

    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    for start, stop in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        run = block[:, start:stop]
        if reach is None:
            rows = np.argmax(run, axis=0)
        else:
            rows = _best_path(run, reach)
        columns = np.arange(start, stop)
        chosen = run[rows, np.arange(run.shape[1])]
        keep = chosen > floor
        tracked[columns[keep]] = frequency[candidates[rows[keep]]]
        power[columns[keep]] = chosen[keep]
    return FrequencyTrack(time, tracked, power)
