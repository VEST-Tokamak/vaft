"""Pure statistical kernels for residual and convergence diagnostics (issue #186).

This module is the single source of truth for the mathematics used to judge how
well a reconstruction reproduces its own inputs: residual magnitude, residual
bias, residual structure, goodness of fit against declared uncertainties, and
the rate at which an iterative solver approaches its answer.

Every function here takes plain arrays or scalars and returns plain floats.
Nothing in this module knows what an ODS, an EFIT run, a diagnostic channel or a
validation policy is -- deciding *which* physical arrays enter a formula belongs
to the OMAS layer, and deciding what value is acceptable belongs to the
validation layer.  Keeping the mathematics separate means a definition can be
tested against an analytic reference without constructing a tokamak.

Conventions
-----------
* Non-finite samples are ignored rather than propagated, so a single dead
  channel does not erase a whole family's statistic.
* A statistic that is undefined for the input given -- too few samples, zero
  variance, a zero denominator -- returns ``nan`` rather than raising, because
  these arise routinely from real diagnostics and every caller aggregates over
  many channels or slices.
* No numpy warning is emitted for an empty or all-non-finite input; the
  degenerate case is guarded explicitly.

None of these statistics is a pass/fail rule on its own.  Each is a diagnostic
indicator whose interpretation depends on assumptions -- stated per function --
that the caller is responsible for defending.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _finite(values: np.ndarray | Iterable[float]) -> np.ndarray:
    """The finite entries of ``values`` as a 1-D float array."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


# ---------------------------------------------------------------------------
# Residual magnitude
# ---------------------------------------------------------------------------

def rms(values: np.ndarray | Iterable[float]) -> float:
    """Root mean square, ``sqrt(mean(x_i**2))``, over the finite entries.

    Statistical meaning
        The quadratic mean: the scale of the sample about zero, not about its
        own mean.  For a zero-mean sample it coincides with the population
        standard deviation; for a biased one it exceeds it, because a systematic
        offset contributes on equal footing with scatter.

    Assumptions
        That zero is the meaningful reference.  Applied to a residual this holds
        by construction; applied to a signal it does not, and
        :func:`residual_bias` should be inspected alongside it.  Samples are
        weighted equally, so a channel array mixing units or uncertainties must
        be normalized first (see :func:`normalized_residual`).

    Interpretation
        The single number for "how far off is this reconstruction", in the units
        of the residual.  It is dominated by the worst channels, so a small RMS
        with a large :func:`outlier_fraction` is contradictory and means the
        residual distribution is not what the average suggests.

    Returns ``nan`` when no finite sample remains.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def fractional_rms_improvement(
    baseline_residual: np.ndarray | Iterable[float],
    residual: np.ndarray | Iterable[float],
) -> float:
    """``1 - RMS(residual) / RMS(baseline_residual)``.

    Statistical meaning
        The fraction of the baseline residual amplitude removed by whatever
        distinguishes the two models.  It is a relative, dimensionless skill
        score, not a significance test.

    Assumptions
        The two residuals come from the same samples and the same measurement,
        differing only in the model subtracted.  The baseline must be non-zero
        and finite; otherwise the ratio is meaningless and ``nan`` is returned.

    Interpretation
        ``1.0`` is a perfect reconstruction, ``0.0`` means the extra model term
        contributed nothing, and a negative value means it made the agreement
        worse -- which is informative, since a physically motivated term that
        degrades the fit points at a sign error or a bad response matrix rather
        than at noise.
    """
    baseline = rms(baseline_residual)
    if not np.isfinite(baseline) or baseline == 0.0:
        return float("nan")
    return float(1.0 - rms(residual) / baseline)


# ---------------------------------------------------------------------------
# Goodness of fit against declared uncertainties
# ---------------------------------------------------------------------------

def normalized_residual(
    residual: np.ndarray | Iterable[float],
    weight: np.ndarray | Iterable[float],
    k: float,
) -> np.ndarray:
    """Residuals in units of the fitted uncertainty: ``z = residual * weight / k``.

    Statistical meaning
        A weighted least-squares fit assigns each channel an effective
        uncertainty ``sigma = k / weight``; dividing the residual by it puts
        every channel on the same dimensionless scale, so a flux loop and a
        B-probe residual become directly comparable.

    Assumptions
        That ``weight`` really is the inverse uncertainty up to the single
        common factor ``k`` -- recover ``k`` from the fit's own chi-square with
        :func:`sigma_unit_factor` rather than assuming it.  If ``k`` is not
        finite or is zero the normalization is undefined and an all-``nan``
        array of the residual's shape is returned.

    Interpretation
        ``|z| ~ 1`` means a channel is reproduced to within the uncertainty it
        was given.  Because the uncertainties are an input, a uniformly small
        ``|z|`` can equally mean the uncertainties were overstated.
    """
    residual_array = np.asarray(residual, dtype=float)
    if not np.isfinite(k) or k == 0.0:
        return np.full(residual_array.shape, np.nan)
    return residual_array * np.asarray(weight, dtype=float) / k


def chi_squared(
    residual: np.ndarray | Iterable[float],
    sigma: np.ndarray | Iterable[float],
) -> float:
    """``sum((residual_i / sigma_i)**2)`` over the finite terms.

    Statistical meaning
        The weighted sum of squared residuals -- the quantity a Gaussian
        maximum-likelihood fit minimizes.  Under the null hypothesis that the
        model is correct and the errors are independent, zero-mean and Gaussian
        with the stated ``sigma``, it follows a chi-square distribution.

    Assumptions
        Independent, unbiased, correctly scaled Gaussian errors.  Correlated
        channels -- a common situation for magnetic diagnostics sharing an
        integrator or a calibration -- inflate or deflate this sum without any
        model being wrong.

    Interpretation
        Rarely read raw, since its expected size grows with the number of
        channels; divide by the degrees of freedom with
        :func:`reduced_chi_squared`.  Its useful raw form is the *share* each
        diagnostic family contributes, which localizes a bad fit.

    Terms where ``sigma`` is zero or either input is non-finite are dropped.
    """
    residual_array = np.asarray(residual, dtype=float).ravel()
    sigma_array = np.asarray(sigma, dtype=float).ravel()
    usable = (
        np.isfinite(residual_array)
        & np.isfinite(sigma_array)
        & (sigma_array != 0.0)
    )
    if not usable.any():
        return float("nan")
    return float(np.sum((residual_array[usable] / sigma_array[usable]) ** 2))


def reduced_chi_squared(chi_squared_total: float, degrees_of_freedom: float) -> float:
    """``chi2 / dof`` -- chi-square per degree of freedom.

    Statistical meaning
        The mean squared normalized residual per free parameter's worth of
        freedom.  Its expectation is 1 when the model is correct and the
        uncertainties are right, since each independent constraint contributes
        one unit of chi-square on average.

    Assumptions
        All three of these must be defensible before the value means anything:
        the supplied uncertainties are the true measurement uncertainties, the
        residuals are independent, and ``dof`` is the *effective* number of
        degrees of freedom rather than a nominal channel count.  Regularized or
        constrained fits routinely violate the last of these, and correlated
        magnetics violate the second.

    Interpretation
        Read it as a diagnostic indicator, never as a pass/fail gate.  A value
        well above 1 means the model does not explain the data *given the stated
        uncertainties* -- which may be a bad model, understated uncertainties, or
        a mis-counted ``dof``.  A value well below 1 usually means overstated
        uncertainties or over-fitting, not an unusually good reconstruction.

    Returns ``nan`` unless ``degrees_of_freedom`` is finite and positive.
    """
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        return float("nan")
    return float(chi_squared_total / degrees_of_freedom)


def sigma_unit_factor(
    residual: np.ndarray | Iterable[float],
    weight: np.ndarray | Iterable[float],
    chi_squared_per_channel: np.ndarray | Iterable[float],
) -> tuple[float, float]:
    """Recover the units-of-fit factor ``k`` from a fit's own chi-square.

    Given a stored per-channel chi-square consistent with the identity
    ``chi2_i = (residual_i * weight_i / k)**2``, the factor is

    .. code-block:: text

        k = median_i( |residual_i * weight_i| / sqrt(chi2_i) )

    Returns ``(k, spread)`` where ``spread = ptp(ratios) / k`` is the
    channel-to-channel scatter of the individual ratios.

    Statistical meaning
        A robust (median) estimate of a single multiplicative constant that
        reconciles the units three stored arrays were written in.  The median
        rather than the mean, so one corrupted channel cannot move the estimate.

    Assumptions
        That *one* constant explains all channels.  ``spread`` is the test of
        that assumption, and it is what makes the estimate self-validating: it
        must be small.  A large spread means the stored chi-square and the
        stored residual no longer describe the same fit, and every normalized
        residual built on ``k`` is then suspect.

    Interpretation
        Recovering ``k`` rather than hard-coding a unit convention means a future
        convention change surfaces as a spread warning instead of as silently
        wrong normalized residuals.  Note that ``spread`` here is a
        peak-to-peak-over-median ratio and is deliberately *not* the same
        statistic as :func:`relative_spread`.

    Channels are used only where the weight is positive and both the chi-square
    and the residual are finite with a positive chi-square; ``(nan, nan)`` comes
    back when none qualify.
    """
    residual_array = np.asarray(residual, dtype=float)
    weight_array = np.asarray(weight, dtype=float)
    chi_array = np.asarray(chi_squared_per_channel, dtype=float)
    usable = (
        (weight_array > 0)
        & np.isfinite(chi_array)
        & (chi_array > 0)
        & np.isfinite(residual_array)
    )
    if not usable.any():
        return float("nan"), float("nan")
    ratios = np.abs(residual_array[usable] * weight_array[usable]) / np.sqrt(
        chi_array[usable]
    )
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if not ratios.size:
        return float("nan"), float("nan")
    k = float(np.median(ratios))
    spread = float(np.ptp(ratios) / k) if k else float("nan")
    return k, spread


# ---------------------------------------------------------------------------
# Residual bias and structure
# ---------------------------------------------------------------------------

def residual_bias(values: np.ndarray | Iterable[float]) -> float:
    """Mean of the finite entries -- the systematic offset of a residual sample.

    Statistical meaning
        The first moment.  For normalized residuals its sampling distribution
        under the null hypothesis has mean 0 and standard deviation
        ``1/sqrt(n)`` (see :func:`bias_standard_error`), so the mean is directly
        comparable against that scale.

    Assumptions
        Independent samples drawn from one distribution.  Channels that share a
        systematic -- a common calibration, a common integrator drift -- are not
        independent, and the bias is then easier to trigger than the nominal
        standard error suggests.

    Interpretation
        Random measurement noise averages to zero; a bias that does not is
        evidence of something the model is not representing, such as an
        unmodelled field component or a geometry offset, rather than of noise.
        Unlike :func:`rms`, it is signed, so cancelling positive and negative
        residuals give a small bias with a large RMS -- which is itself
        diagnostic of structure rather than offset.

    Returns ``nan`` when no finite sample remains.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.mean(finite))


def bias_standard_error(count: int) -> float:
    """``1 / sqrt(n)`` -- the standard error of the mean of unit-variance samples.

    Statistical meaning
        The standard deviation of the sample mean when each sample has unit
        variance, which is the case for correctly normalized residuals by
        construction.  It is the yardstick a measured :func:`residual_bias` is
        held against.

    Assumptions
        Unit-variance, independent samples.  Do **not** use this for a raw
        residual in physical units -- there the sample standard deviation must
        be estimated from the data instead.  Correlation between channels makes
        the true standard error larger than this value.

    Interpretation
        A bias exceeding roughly twice this is unlikely to be a fluctuation, and
        so points at a systematic.  Because it shrinks only as ``1/sqrt(n)``, a
        family with few channels tolerates a visibly large bias before it
        becomes significant.

    Returns ``nan`` for ``count <= 0``.
    """
    if count <= 0:
        return float("nan")
    return 1.0 / math.sqrt(count)


def outlier_fraction(values: np.ndarray | Iterable[float], level: float) -> float:
    """Fraction of finite entries with ``|x| > level``.

    Statistical meaning
        The empirical tail mass beyond a threshold.  For standard normal samples
        the expected fractions are about 4.6% beyond 2 and 0.27% beyond 3.

    Assumptions
        That ``values`` are normalized so ``level`` is a number of standard
        deviations -- for raw residuals the threshold has no such meaning.  With
        few channels the estimate is coarse: one outlier in ten channels is 10%,
        which is not by itself evidence of anything.

    Interpretation
        Distinguishes a fit that is uniformly mediocre from one that is good
        everywhere except a handful of channels.  The second, an excess tail
        with an acceptable RMS, usually means a specific broken diagnostic
        rather than a bad reconstruction.

    Returns ``nan`` when no finite sample remains.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.mean(np.abs(finite) > level))


def lag1_autocorrelation(values: np.ndarray | Iterable[float]) -> float:
    """Lag-1 autocorrelation of the finite entries, in the order given.

    .. code-block:: text

        r1 = sum_i (x_i - xbar)(x_{i+1} - xbar) / sum_i (x_i - xbar)**2

    Statistical meaning
        The normalized correlation between each sample and its neighbour in
        sequence, bounded roughly to ``[-1, 1]``.  Under independence its
        expectation is approximately ``-1/n``, i.e. zero for practical purposes.

    Assumptions
        That the ordering of the array carries meaning -- adjacency in time, in
        space, or in a physically ordered channel index.  Applied to an
        arbitrarily ordered array the statistic is meaningless.  The sample must
        have at least three finite entries and non-zero variance.

    Interpretation
        A strongly positive value means neighbouring residuals move together,
        which for a spatially ordered channel array is the signature of a smooth
        unmodelled field rather than of independent measurement noise.  A
        strongly negative value means alternation, which points at a wiring or
        sign convention error on alternate channels.
    """
    finite = _finite(values)
    if finite.size < 3:
        return float("nan")
    centered = finite - finite.mean()
    denominator = float(np.sum(centered * centered))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(centered[1:] * centered[:-1]) / denominator)


def runs_test_z(values: np.ndarray | Iterable[float]) -> float:
    """Wald-Wolfowitz runs-test z-score for the sign sequence of ``values``.

    With ``n_+`` positive and ``n_-`` negative entries out of ``n``, and ``R``
    the number of maximal same-sign runs,

    .. code-block:: text

        E[R]   = 2 n_+ n_- / n + 1
        Var[R] = 2 n_+ n_- (2 n_+ n_- - n) / (n**2 (n - 1))
        z      = (R - E[R]) / sqrt(Var[R])

    Statistical meaning
        A distribution-free test of whether a binary sequence is randomly
        ordered.  It uses only the signs, so it is insensitive to outlier
        magnitude -- which makes it complementary to :func:`rms` and
        :func:`lag1_autocorrelation` rather than redundant with them.

    Assumptions
        Exchangeable samples under the null, a meaningful ordering, and enough
        of both signs for the normal approximation: this returns ``nan`` below
        three non-zero samples or when one sign is absent entirely.  Zero-valued
        entries carry no sign and are dropped.

    Interpretation
        ``|z| > 2`` means the sign pattern is unlikely under independence.  Too
        few runs (negative ``z``) is clustering -- contiguous stretches of
        one-signed residual, the fingerprint of an unmodelled coherent field
        component.  Too many (positive ``z``) is alternation, which points at an
        indexing or polarity error rather than at physics.
    """
    array = np.asarray(values, dtype=float).ravel()
    signs = np.sign(array[np.isfinite(array) & (array != 0.0)])
    n = signs.size
    if n < 3:
        return float("nan")
    positive = int(np.sum(signs > 0))
    negative = n - positive
    if positive == 0 or negative == 0:
        return float("nan")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 2.0 * positive * negative / n + 1.0
    variance = (
        2.0 * positive * negative * (2.0 * positive * negative - n)
        / (n * n * (n - 1.0))
    )
    if variance <= 0.0:
        return float("nan")
    return float((runs - expected) / math.sqrt(variance))


def relative_spread(values: Iterable[float]) -> float:
    """``(max - min) / max|x|`` over the finite entries.

    Statistical meaning
        A dimensionless, non-parametric measure of disagreement within a small
        set: the full range normalized by the largest magnitude present.  It is
        deliberately range-based rather than variance-based, because it is meant
        for a handful of values where a standard deviation would be noise.

    Assumptions
        The values are independent estimates of the *same* quantity, so that any
        spread between them is inconsistency rather than physics.  Being
        range-based it is maximally sensitive to a single bad estimate, which is
        the intent when used as a consistency check.

    Interpretation
        A self-consistency metric: several routes to one quantity should agree,
        and the spread is how far they do not.  Returns ``0.0`` -- not ``nan`` --
        when every value is exactly zero, since that is perfect agreement, and
        ``nan`` when fewer than two finite values are available to compare.
    """
    finite = [value for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    scale = max(abs(value) for value in finite)
    if scale == 0.0:
        return 0.0
    return float((max(finite) - min(finite)) / scale)


# ---------------------------------------------------------------------------
# Threshold crossing
# ---------------------------------------------------------------------------

def noise_band(values: np.ndarray | Iterable[float]) -> tuple[float, float]:
    """``(mean, standard deviation)`` of the finite entries of a reference sample.

    Statistical meaning
        The first two moments of a stretch of signal taken to contain no effect,
        estimating the offset and the amplitude of the measurement noise.  The
        standard deviation is the population (``ddof = 0``) one.

    Assumptions
        The sample is genuinely effect-free and stationary over its extent.  A
        drift, a switching transient, or a leaked part of the effect inside the
        reference window inflates the estimated noise and makes any threshold
        built on it too permissive.

    Interpretation
        Deriving the threshold from the channel's own measured quiet stretch,
        rather than from a global constant, means a noisy channel is judged by
        its own noise -- essential when channels differ in gain, integrator
        drift and cabling.  Returns ``(nan, nan)`` for an empty or wholly
        non-finite sample.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def sigma_threshold_crossing(
    time: np.ndarray | Iterable[float],
    values: np.ndarray | Iterable[float],
    reference_mask: np.ndarray | Iterable[bool],
    *,
    sigma: float,
) -> float:
    """First time outside ``reference_mask`` where ``values`` leave their noise band.

    The band is ``mean ± sigma * std`` of the samples *inside* ``reference_mask``
    (see :func:`noise_band`); the returned time is the first sample outside the
    mask with ``|x - mean| > sigma * std``.

    Statistical meaning
        A fixed-threshold first-passage detector with the threshold expressed in
        units of the sample's own noise, so the nominal false-alarm probability
        per sample is set by ``sigma`` alone rather than by the channel's gain.

    Assumptions
        A well-defined quiet reference stretch of at least two finite samples,
        stationary noise across the whole record, and independent samples --
        oversampled or filtered data has fewer independent samples than points,
        so the effective false-alarm rate is lower than a naive count suggests.
        With ``n`` samples searched, the expected number of noise-only crossings
        is roughly ``n`` times the per-sample tail probability, which is why a
        large ``sigma`` is used in practice.

    Interpretation
        Onset detection: the first moment a channel sees something its own
        pre-event noise cannot explain.  Comparing onsets across channels is the
        real test -- a physical event appears everywhere at once, while a
        forward-model or response-matrix artifact does not.  Returns ``nan``
        when the reference is too short, its noise is zero or non-finite, or the
        signal never emerges.
    """
    time_array = np.asarray(time, dtype=float)
    value_array = np.asarray(values, dtype=float)
    mask = np.asarray(reference_mask, dtype=bool)
    reference = value_array[mask]
    if reference.size < 2:
        return float("nan")
    baseline, noise = noise_band(reference)
    if not np.isfinite(noise) or noise == 0.0:
        return float("nan")
    emerged = np.flatnonzero(~mask & (np.abs(value_array - baseline) > sigma * noise))
    if emerged.size == 0:
        return float("nan")
    return float(time_array[emerged[0]])


# ---------------------------------------------------------------------------
# Iterative-solver convergence history
# ---------------------------------------------------------------------------

def monotonic_fraction(values: np.ndarray | Iterable[float]) -> float:
    """Fraction of consecutive steps in which ``values`` decreased.

    Statistical meaning
        The empirical rate of the event "the next sample is smaller than this
        one" over the ``n - 1`` adjacent pairs.  Comparisons involving a
        non-finite entry count as non-decreases, so a broken history is
        penalized rather than silently shortened.

    Assumptions
        The sequence is an ordered iteration history in which decrease is the
        desired direction.  A short history makes the fraction coarse -- with
        four iterations it can only take five values.

    Interpretation
        For a solver error history, ``1.0`` is a cleanly contracting iteration.
        Values near ``0.5`` mean the error is bouncing, which indicates a step
        that is too aggressive or a solution oscillating between two branches --
        a distinct failure from steady but slow progress, which
        :func:`log10_decay_rate` detects instead.

    Returns ``nan`` for fewer than two samples.
    """
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        decreases = int(np.sum(array[1:] < array[:-1]))
    return float(decreases / (array.size - 1))


def log10_decay_rate(
    values: np.ndarray | Iterable[float], *, tail: int = 5
) -> float:
    """Least-squares slope of ``log10(x)`` against iteration index, over the tail.

    Positive, finite entries are selected, the last ``tail`` of them are taken,
    and a straight line is fitted to their base-10 logarithm against
    ``0, 1, 2, ...``.  The slope is *decades per iteration*.

    Statistical meaning
        The exponential rate constant of the sequence, estimated in log space so
        that a geometric decay becomes a straight-line fit.  Ordinary least
        squares in log space weights relative rather than absolute errors, which
        is the right choice for a quantity spanning orders of magnitude.

    Assumptions
        At least three positive finite samples in the tail; non-positive entries
        cannot be logged and are dropped, which silently shortens the window.
        The fit assumes a single exponential regime, so a history that changes
        behaviour mid-run gives a slope describing neither phase.  Only the tail
        is used, deliberately: the early transient of an iterative solve says
        nothing about whether it is converging now.

    Interpretation
        A slope of ``-1`` means the error falls by a decade per iteration.  A
        value near zero means the iteration has stagnated -- it is still running
        but no longer improving, which is a different and more troubling outcome
        than hitting an iteration limit while still descending.  What counts as
        "near zero" is a solver policy decision and belongs to the caller.

    Returns ``nan`` when fewer than three usable samples remain.
    """
    array = np.asarray(values, dtype=float).ravel()
    positive = array[np.isfinite(array) & (array > 0)]
    if positive.size == 0 or tail <= 0:
        return float("nan")
    window = positive[-min(positive.size, tail):]
    if window.size < 3:
        return float("nan")
    return float(
        np.polyfit(np.arange(window.size, dtype=float), np.log10(window), 1)[0]
    )


__all__ = [
    "bias_standard_error",
    "chi_squared",
    "fractional_rms_improvement",
    "lag1_autocorrelation",
    "log10_decay_rate",
    "monotonic_fraction",
    "noise_band",
    "normalized_residual",
    "outlier_fraction",
    "reduced_chi_squared",
    "relative_spread",
    "residual_bias",
    "rms",
    "runs_test_z",
    "sigma_threshold_crossing",
    "sigma_unit_factor",
]
