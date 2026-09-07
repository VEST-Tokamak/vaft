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
from typing import Mapping, Sequence

import numpy as np
from scipy import signal as scipy_signal

__all__ = [
    "FluctuationSpectrogram",
    "FluctuationSpectrum",
    "SpectralBreak",
    "SpectralFit",
    "analyze_fluctuation_spectrum",
    "compute_band_power",
    "compute_psd",
    "compute_spectrogram",
    "find_spectral_break",
    "fit_power_law_spectrum",
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
