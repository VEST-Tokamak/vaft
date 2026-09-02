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
    r"""Root mean square, $\sqrt{\mathrm{mean}(x_i^2)}$, over the finite entries.

    $$\mathrm{RMS} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$$

    Parameters
    ----------
    values : array-like
        Sample; non-finite entries are ignored [any].

    Returns
    -------
    float
        Quadratic mean in the unit of the sample; ``nan`` for an empty one [any].

    Physical interpretation
    -----------------------
    The quadratic mean: the scale of the sample about zero, not about its own
    mean.  For a zero-mean sample it coincides with the population standard
    deviation; for a biased one it exceeds it, because a systematic offset
    contributes on equal footing with scatter.

    Assumptions
    -----------
    That zero is the meaningful reference.  Applied to a residual this holds by
    construction; applied to a signal it does not, and :func:`residual_bias`
    should be inspected alongside it.  Samples are weighted equally, so a
    channel array mixing units or uncertainties must be normalized first (see
    :func:`normalized_residual`).

    Limitations
    -----------
    Returns ``nan`` when no finite sample remains.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 1.3.

    Notes
    -----
    The single number for "how far off is this reconstruction", in the units of
    the residual.  It is dominated by the worst channels, so a small RMS with a
    large :func:`outlier_fraction` is contradictory and means the residual
    distribution is not what the average suggests.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def fractional_rms_improvement(
    baseline_residual: np.ndarray | Iterable[float],
    residual: np.ndarray | Iterable[float],
) -> float:
    r"""Fractional RMS improvement, $1 - \mathrm{RMS}(r)/\mathrm{RMS}(r_0)$.

    $$\eta = 1 - \frac{\mathrm{RMS}(\mathrm{residual})}{\mathrm{RMS}(\mathrm{baseline})}$$

    Parameters
    ----------
    baseline_residual : array-like
        Residual of the reference model [any].
    residual : array-like
        Residual of the model under test, same samples and unit [any].

    Returns
    -------
    float
        Skill score; ``nan`` when the baseline RMS is zero or non-finite [-].

    Physical interpretation
    -----------------------
    The fraction of the baseline residual amplitude removed by whatever
    distinguishes the two models.  It is a relative, dimensionless skill score,
    not a significance test.

    Assumptions
    -----------
    The two residuals come from the same samples and the same measurement,
    differing only in the model subtracted.  The baseline must be non-zero and
    finite; otherwise the ratio is meaningless and ``nan`` is returned.

    References
    ----------
    .. [1] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 4.4.4 (model validation).

    Notes
    -----
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
    r"""Residuals in units of the fitted uncertainty, $z = r\,w/k$.

    $$z_i = \frac{r_i\,w_i}{k}, \qquad \sigma_i = \frac{k}{w_i}$$

    Parameters
    ----------
    residual : array-like
        Residuals of a weighted least-squares fit [any].
    weight : array-like
        Inverse-uncertainty weights of the same channels [any].
    k : float
        Common units-of-fit factor, see :func:`sigma_unit_factor` [any].

    Returns
    -------
    np.ndarray
        Normalized residuals, shape of ``residual``; all ``nan`` for ``k`` zero or
        non-finite [-].

    Physical interpretation
    -----------------------
    A weighted least-squares fit assigns each channel an effective uncertainty
    ``sigma = k / weight``; dividing the residual by it puts every channel on
    the same dimensionless scale, so a flux loop and a B-probe residual become
    directly comparable.

    Assumptions
    -----------
    That ``weight`` really is the inverse uncertainty up to the single common
    factor ``k`` -- recover ``k`` from the fit's own chi-square with
    :func:`sigma_unit_factor` rather than assuming it.  If ``k`` is not finite
    or is zero the normalization is undefined and an all-``nan`` array of the
    residual's shape is returned.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 4.1
           (weighted means and standardized residuals).

    Notes
    -----
    ``|z| ~ 1`` means a channel is reproduced to within the uncertainty it was
    given.  Because the uncertainties are an input, a uniformly small ``|z|``
    can equally mean the uncertainties were overstated.
    """
    residual_array = np.asarray(residual, dtype=float)
    if not np.isfinite(k) or k == 0.0:
        return np.full(residual_array.shape, np.nan)
    return residual_array * np.asarray(weight, dtype=float) / k


def chi_squared(
    residual: np.ndarray | Iterable[float],
    sigma: np.ndarray | Iterable[float],
) -> float:
    r"""Chi-square, $\sum_i (r_i/\sigma_i)^2$ over the finite terms.

    $$\chi^2 = \sum_i\left(\frac{r_i}{\sigma_i}\right)^2$$

    Parameters
    ----------
    residual : array-like
        Residuals [any].
    sigma : array-like
        One-sigma uncertainty of each residual, same unit [any].

    Returns
    -------
    float
        Chi-square; ``nan`` when no usable term remains [-].

    Physical interpretation
    -----------------------
    The weighted sum of squared residuals -- the quantity a Gaussian
    maximum-likelihood fit minimizes.  Under the null hypothesis that the model
    is correct and the errors are independent, zero-mean and Gaussian with the
    stated ``sigma``, it follows a chi-square distribution.

    Assumptions
    -----------
    Independent, unbiased, correctly scaled Gaussian errors.  Correlated
    channels -- a common situation for magnetic diagnostics sharing an
    integrator or a calibration -- inflate or deflate this sum without any
    model being wrong.

    Limitations
    -----------
    Terms where ``sigma`` is zero or either input is non-finite are dropped.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 4.3
           and Ch. 11 (chi-square test of goodness of fit).

    Notes
    -----
    Rarely read raw, since its expected size grows with the number of channels;
    divide by the degrees of freedom with :func:`reduced_chi_squared`.  Its
    useful raw form is the *share* each diagnostic family contributes, which
    localizes a bad fit.
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
    r"""Reduced chi-square, $\chi^2/\nu$ per degree of freedom.

    $$\chi^2_\nu = \frac{\chi^2}{\nu}$$

    Parameters
    ----------
    chi_squared_total : float
        Total chi-square [-].
    degrees_of_freedom : float
        Effective degrees of freedom $\nu$, positive [-].

    Returns
    -------
    float
        Chi-square per degree of freedom; ``nan`` unless $\nu$ is finite and positive [-].

    Physical interpretation
    -----------------------
    The mean squared normalized residual per free parameter's worth of freedom.
    Its expectation is 1 when the model is correct and the uncertainties are
    right, since each independent constraint contributes one unit of chi-square
    on average.

    Assumptions
    -----------
    All three of these must be defensible before the value means anything: the
    supplied uncertainties are the true measurement uncertainties, the residuals
    are independent, and ``dof`` is the *effective* number of degrees of freedom
    rather than a nominal channel count.  Regularized or constrained fits
    routinely violate the last of these, and correlated magnetics violate the
    second.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 11.1.

    Notes
    -----
    Read it as a diagnostic indicator, never as a pass/fail gate.  A value well
    above 1 means the model does not explain the data *given the stated
    uncertainties* -- which may be a bad model, understated uncertainties, or a
    mis-counted ``dof``.  A value well below 1 usually means overstated
    uncertainties or over-fitting, not an unusually good reconstruction.
    """
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        return float("nan")
    return float(chi_squared_total / degrees_of_freedom)


def sigma_unit_factor(
    residual: np.ndarray | Iterable[float],
    weight: np.ndarray | Iterable[float],
    chi_squared_per_channel: np.ndarray | Iterable[float],
) -> tuple[float, float]:
    r"""Recover the units-of-fit factor $k$ from a fit's own per-channel chi-square.

    $$k = \operatorname{median}_i\frac{|r_i\,w_i|}{\sqrt{\chi^2_i}}, \qquad
      \mathrm{spread} = \frac{\max_i - \min_i}{k}$$

    from the identity $\chi^2_i = (r_i w_i/k)^2$.

    Parameters
    ----------
    residual : array-like
        Stored residual of each channel [any].
    weight : array-like
        Stored inverse-uncertainty weight of each channel [any].
    chi_squared_per_channel : array-like
        Stored per-channel chi-square [-].

    Returns
    -------
    k : float
        Median units-of-fit factor; ``nan`` when no channel qualifies [any].
    spread : float
        Peak-to-peak scatter of the per-channel ratios over ``k`` [-].

    Physical interpretation
    -----------------------
    A robust (median) estimate of a single multiplicative constant that
    reconciles the units three stored arrays were written in.  The median
    rather than the mean, so one corrupted channel cannot move the estimate.

    Assumptions
    -----------
    That *one* constant explains all channels.  ``spread`` is the test of that
    assumption, and it is what makes the estimate self-validating: it must be
    small.  A large spread means the stored chi-square and the stored residual
    no longer describe the same fit, and every normalized residual built on
    ``k`` is then suspect.

    Limitations
    -----------
    Channels are used only where the weight is positive and both the chi-square
    and the residual are finite with a positive chi-square; ``(nan, nan)`` comes
    back when none qualify.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 4.1.

    Notes
    -----
    Recovering ``k`` rather than hard-coding a unit convention means a future
    convention change surfaces as a spread warning instead of as silently wrong
    normalized residuals.  Note that ``spread`` here is a peak-to-peak-over-median
    ratio and is deliberately *not* the same statistic as :func:`relative_spread`.
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
    r"""Mean of the finite entries, the systematic offset of a residual sample.

    $$\bar r = \frac{1}{n}\sum_{i=1}^{n} r_i$$

    Parameters
    ----------
    values : array-like
        Residual sample; non-finite entries are ignored [any].

    Returns
    -------
    float
        Sample mean in the unit of the residual; ``nan`` for an empty sample [any].

    Physical interpretation
    -----------------------
    The first moment.  For normalized residuals its sampling distribution under
    the null hypothesis has mean 0 and standard deviation ``1/sqrt(n)`` (see
    :func:`bias_standard_error`), so the mean is directly comparable against
    that scale.

    Assumptions
    -----------
    Independent samples drawn from one distribution.  Channels that share a
    systematic -- a common calibration, a common integrator drift -- are not
    independent, and the bias is then easier to trigger than the nominal
    standard error suggests.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 1.2.

    Notes
    -----
    Random measurement noise averages to zero; a bias that does not is evidence
    of something the model is not representing, such as an unmodelled field
    component or a geometry offset, rather than of noise.  Unlike :func:`rms`,
    it is signed, so cancelling positive and negative residuals give a small
    bias with a large RMS -- which is itself diagnostic of structure rather than
    offset.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.mean(finite))


def bias_standard_error(count: int) -> float:
    r"""Standard error of the mean of $n$ unit-variance samples, $1/\sqrt{n}$.

    $$\sigma_{\bar x} = \frac{1}{\sqrt{n}}$$

    Parameters
    ----------
    count : int
        Number of samples [-].

    Returns
    -------
    float
        Standard error; ``nan`` for ``count <= 0`` [-].

    Physical interpretation
    -----------------------
    The standard deviation of the sample mean when each sample has unit
    variance, which is the case for correctly normalized residuals by
    construction.  It is the yardstick a measured :func:`residual_bias` is held
    against.

    Assumptions
    -----------
    Unit-variance, independent samples.  Do **not** use this for a raw residual
    in physical units -- there the sample standard deviation must be estimated
    from the data instead.  Correlation between channels makes the true
    standard error larger than this value.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 4.2
           (standard deviation of the mean).

    Notes
    -----
    A bias exceeding roughly twice this is unlikely to be a fluctuation, and so
    points at a systematic.  Because it shrinks only as ``1/sqrt(n)``, a family
    with few channels tolerates a visibly large bias before it becomes
    significant.
    """
    if count <= 0:
        return float("nan")
    return 1.0 / math.sqrt(count)


def outlier_fraction(values: np.ndarray | Iterable[float], level: float) -> float:
    r"""Fraction of finite entries with $|x_i| > \mathrm{level}$.

    $$f = \frac{1}{n}\,\operatorname{count}\big(|x_i| > \mathrm{level}\big)$$

    Parameters
    ----------
    values : array-like
        Normalized sample; non-finite entries are ignored [-].
    level : float
        Threshold in the sample's own units [-].

    Returns
    -------
    float
        Tail fraction; ``nan`` for an empty sample [-].

    Physical interpretation
    -----------------------
    The empirical tail mass beyond a threshold.  For standard normal samples
    the expected fractions are about 4.6% beyond 2 and 0.27% beyond 3.

    Assumptions
    -----------
    That ``values`` are normalized so ``level`` is a number of standard
    deviations -- for raw residuals the threshold has no such meaning.  With
    few channels the estimate is coarse: one outlier in ten channels is 10%,
    which is not by itself evidence of anything.

    References
    ----------
    .. [1] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 1.3.5.17 (detection of
           outliers).

    Notes
    -----
    Distinguishes a fit that is uniformly mediocre from one that is good
    everywhere except a handful of channels.  The second, an excess tail with
    an acceptable RMS, usually means a specific broken diagnostic rather than a
    bad reconstruction.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.mean(np.abs(finite) > level))


def lag1_autocorrelation(values: np.ndarray | Iterable[float]) -> float:
    r"""Lag-1 autocorrelation of the finite entries, in the order given.

    $$r_1 = \frac{\sum_i (x_i - \bar x)(x_{i+1} - \bar x)}{\sum_i (x_i - \bar x)^2}$$

    Parameters
    ----------
    values : array-like
        Ordered sample; non-finite entries are dropped before pairing [any].

    Returns
    -------
    float
        Lag-1 autocorrelation in roughly $[-1, 1]$; ``nan`` below three samples
        or for zero variance [-].

    Physical interpretation
    -----------------------
    The normalized correlation between each sample and its neighbour in
    sequence.  Under independence its expectation is approximately ``-1/n``,
    i.e. zero for practical purposes.

    Assumptions
    -----------
    That the ordering of the array carries meaning -- adjacency in time, in
    space, or in a physically ordered channel index.  Applied to an arbitrarily
    ordered array the statistic is meaningless.  The sample must have at least
    three finite entries and non-zero variance.

    References
    ----------
    .. [1] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 1.3.5.12 (autocorrelation).
    .. [2] G. E. P. Box, G. M. Jenkins and G. C. Reinsel, *Time Series
           Analysis*, 4th ed., Wiley (2008), Sec. 2.1.

    Notes
    -----
    A strongly positive value means neighbouring residuals move together, which
    for a spatially ordered channel array is the signature of a smooth
    unmodelled field rather than of independent measurement noise.  A strongly
    negative value means alternation, which points at a wiring or sign
    convention error on alternate channels.
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
    r"""Wald-Wolfowitz runs-test z-score for the sign sequence of a sample.

    With $n_+$ positive and $n_-$ negative entries out of $n$, and $R$ the
    number of maximal same-sign runs,

    $$E[R] = \frac{2n_+n_-}{n} + 1, \qquad
      \mathrm{Var}[R] = \frac{2n_+n_-(2n_+n_- - n)}{n^2(n - 1)}, \qquad
      z = \frac{R - E[R]}{\sqrt{\mathrm{Var}[R]}}$$

    Parameters
    ----------
    values : array-like
        Ordered sample; zeros and non-finite entries carry no sign and are
        dropped [any].

    Returns
    -------
    float
        z-score; ``nan`` below three signed samples or when one sign is absent [-].

    Physical interpretation
    -----------------------
    A distribution-free test of whether a binary sequence is randomly ordered.
    It uses only the signs, so it is insensitive to outlier magnitude -- which
    makes it complementary to :func:`rms` and :func:`lag1_autocorrelation`
    rather than redundant with them.

    Assumptions
    -----------
    Exchangeable samples under the null, a meaningful ordering, and enough of
    both signs for the normal approximation: this returns ``nan`` below three
    non-zero samples or when one sign is absent entirely.

    References
    ----------
    .. [1] A. Wald and J. Wolfowitz, Ann. Math. Statist. 11 (1940) 147.
    .. [2] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 1.3.5.13 (runs test).

    Notes
    -----
    ``|z| > 2`` means the sign pattern is unlikely under independence.  Too few
    runs (negative ``z``) is clustering -- contiguous stretches of one-signed
    residual, the fingerprint of an unmodelled coherent field component.  Too
    many (positive ``z``) is alternation, which points at an indexing or
    polarity error rather than at physics.
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
    r"""Relative spread, $(\max - \min)/\max|x|$ over the finite entries.

    $$s = \frac{\max_i x_i - \min_i x_i}{\max_i |x_i|}$$

    Parameters
    ----------
    values : iterable of float
        Independent estimates of one quantity [any].

    Returns
    -------
    float
        Relative spread; 0 for all-zero values, ``nan`` below two finite values [-].

    Physical interpretation
    -----------------------
    A dimensionless, non-parametric measure of disagreement within a small
    set: the full range normalized by the largest magnitude present.  It is
    deliberately range-based rather than variance-based, because it is meant
    for a handful of values where a standard deviation would be noise.

    Assumptions
    -----------
    The values are independent estimates of the *same* quantity, so that any
    spread between them is inconsistency rather than physics.  Being
    range-based it is maximally sensitive to a single bad estimate, which is
    the intent when used as a consistency check.

    References
    ----------
    .. [1] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 1.3.5.6 (measures of scale).

    Notes
    -----
    A self-consistency metric: several routes to one quantity should agree, and
    the spread is how far they do not.  Returns ``0.0`` -- not ``nan`` -- when
    every value is exactly zero, since that is perfect agreement, and ``nan``
    when fewer than two finite values are available to compare.
    """
    finite = [value for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    scale = max(abs(value) for value in finite)
    if scale == 0.0:
        return 0.0
    return float((max(finite) - min(finite)) / scale)


# ---------------------------------------------------------------------------
# Robust scale, trend and agreement
# ---------------------------------------------------------------------------

def median_absolute_deviation(values: np.ndarray | Iterable[float]) -> float:
    r"""Median absolute deviation, $\mathrm{median}(|x - \mathrm{median}(x)|)$.

    $$\mathrm{MAD} = \operatorname{median}_i\big|x_i - \operatorname{median}(x)\big|$$

    Parameters
    ----------
    values : array-like
        Sample; non-finite entries are ignored [any].

    Returns
    -------
    float
        Raw MAD in the unit of the sample, not rescaled to a Gaussian sigma;
        ``nan`` for an empty sample [any].

    Physical interpretation
    -----------------------
    A scale estimator with a 50% breakdown point: half the samples may be
    arbitrarily corrupted before it moves.  The standard deviation, by
    contrast, has a breakdown point of zero -- one bad sample moves it without
    limit.

    Assumptions
    -----------
    None beyond the samples sharing a scale.  In particular no distribution is
    assumed; the value returned is the raw median deviation, *not* rescaled to
    a Gaussian-equivalent sigma.  :func:`robust_z_scores` applies the 1.4826
    consistency factor where that interpretation is wanted, so the two uses
    cannot drift apart.

    References
    ----------
    .. [1] P. J. Rousseeuw and C. Croux, J. Am. Stat. Assoc. 88 (1993) 1273,
           Sec. 1.

    Notes
    -----
    For a signal being screened for spikes or dropouts, this is the width of
    the bulk of the samples -- the noise level the anomalies must be measured
    against, rather than a noise level the anomalies have already inflated.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.median(np.abs(finite - np.median(finite))))


def robust_z_scores(values: np.ndarray | Iterable[float]) -> np.ndarray:
    r"""Deviation from the median in robust sigma units, sample by sample.

    $$z_i = \frac{x_i - \operatorname{median}(x)}{1.4826\,\mathrm{MAD}}$$

    where 1.4826 makes the denominator a consistent estimator of the standard
    deviation for Gaussian samples, so a threshold in these units carries its
    usual meaning.

    Parameters
    ----------
    values : array-like
        Sample [any].

    Returns
    -------
    np.ndarray
        Robust z-scores with the input's shape, ``nan`` where the input was
        non-finite, signed ``inf`` off a zero-MAD bulk [-].

    Physical interpretation
    -----------------------
    A z-score built on estimators that the anomalies being looked for cannot
    themselves corrupt.  A single large spike inflates the mean and the
    standard deviation enough to hide itself from a conventional z-score; it
    moves neither the median nor the MAD.

    Assumptions
    -----------
    A unimodal bulk.  These scores are meaningless for a genuinely bimodal
    sample, where the median falls between the modes and every sample scores
    as an outlier.

    Limitations
    -----------
    A zero MAD means the bulk of the samples are identical -- a perfectly
    linear ramp differenced, or a flatlined channel.  Any departure from that
    constant is then infinitely many sigmas out, and is reported as signed
    ``inf`` rather than as ``nan``, because such a sample is the clearest
    possible anomaly and not an undefined one.  Returns an all-``nan`` array
    when no finite sample remains.

    References
    ----------
    .. [1] P. J. Rousseeuw and C. Croux, J. Am. Stat. Assoc. 88 (1993) 1273.
    .. [2] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 1.3.5.17.

    Notes
    -----
    Unlike the scalar reductions elsewhere in this module, the result is
    positional: it has the shape of the input, so a caller can locate *which*
    samples deviate and not merely how many.
    """
    array = np.asarray(values, dtype=float).ravel()
    scores = np.full(array.shape, float("nan"))
    finite = np.isfinite(array)
    if not finite.any():
        return scores
    centre = float(np.median(array[finite]))
    deviation = array[finite] - centre
    scale = 1.4826 * float(np.median(np.abs(deviation)))
    if scale == 0.0:
        # `np.where` evaluates both branches, so `sign(0) * inf` would raise an
        # invalid-value warning for every unremarkable sample.  `copysign` is
        # total on the whole input and never does.
        scores[finite] = np.where(deviation == 0.0, 0.0, np.copysign(np.inf, deviation))
        return scores
    scores[finite] = deviation / scale
    return scores


def linear_trend(
    x: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]
) -> float:
    r"""Least-squares slope of $y$ against $x$, in units of $y$ per unit $x$.

    $$\hat b = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}$$

    Parameters
    ----------
    x : array-like
        Abscissae [any].
    y : array-like
        Ordinates, same length [any].

    Returns
    -------
    float
        Slope; ``nan`` below two paired finite samples or for constant ``x`` [any].

    Raises
    ------
    ValueError
        Mismatched lengths.

    Physical interpretation
    -----------------------
    The first-order coefficient of an ordinary least-squares fit, over the
    samples where both series are finite.

    Assumptions
    -----------
    That a straight line is a meaningful summary.  It is the right question for
    integrator drift or a slowly walking baseline, and the wrong one for a
    signal whose shape is genuinely curved -- the slope will report something
    in that case too, and it will not mean drift.

    Numerical notes
    ---------------
    ``numpy.polyfit`` of degree 1 (unweighted).

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 6.3.

    Notes
    -----
    Signed, and in physical units, so it is compared against the signal's own
    scale rather than against a universal number: a drift of 1e-4 T/s matters
    for a probe whose dynamic range is 1e-3 T and does not for one spanning
    1 T.  Unlike :func:`residual_bias` it is insensitive to a constant offset,
    which is what separates "this channel is offset" from "this channel is
    walking".
    """
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()
    if x_array.size != y_array.size:
        raise ValueError(
            f"x and y must have the same length, got {x_array.size} and {y_array.size}"
        )
    paired = np.isfinite(x_array) & np.isfinite(y_array)
    if paired.sum() < 2:
        return float("nan")
    x_finite, y_finite = x_array[paired], y_array[paired]
    if float(np.ptp(x_finite)) == 0.0:
        return float("nan")
    return float(np.polyfit(x_finite, y_finite, 1)[0])


def pearson_correlation(
    a: np.ndarray | Iterable[float], b: np.ndarray | Iterable[float]
) -> float:
    r"""Pearson correlation of two series over their pairwise-finite samples.

    $$\rho = \frac{\sum_i (a_i - \bar a)(b_i - \bar b)}
      {\sqrt{\sum_i (a_i - \bar a)^2}\sqrt{\sum_i (b_i - \bar b)^2}}$$

    Parameters
    ----------
    a : array-like
        First series [any].
    b : array-like
        Second series, same length [any].

    Returns
    -------
    float
        Correlation in $[-1, 1]$; ``nan`` below two pairs or for zero variance [-].

    Raises
    ------
    ValueError
        Mismatched lengths.

    Physical interpretation
    -----------------------
    Covariance normalized by both standard deviations: the cosine of the angle
    between the two mean-centred series.

    Assumptions
    -----------
    A linear relationship.  Two series related by a strong but curved mapping
    correlate poorly, and the low value says the relationship is not linear
    rather than that it is absent.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 11.2
           (linear-correlation coefficient).

    Notes
    -----
    For a measured signal against a forward model it separates *shape*
    agreement from *amplitude* agreement, which an RMS residual conflates.  A
    correlation near 1 with a large residual says the model has the dynamics
    right and the gain wrong -- a calibration question.  A small correlation
    with a small residual says neither signal is doing much, and the comparison
    is uninformative rather than successful.
    """
    a_array = np.asarray(a, dtype=float).ravel()
    b_array = np.asarray(b, dtype=float).ravel()
    if a_array.size != b_array.size:
        raise ValueError(
            f"series must have the same length, got {a_array.size} and {b_array.size}"
        )
    paired = np.isfinite(a_array) & np.isfinite(b_array)
    if paired.sum() < 2:
        return float("nan")
    left, right = a_array[paired], b_array[paired]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def dynamic_range(values: np.ndarray | Iterable[float]) -> float:
    r"""Sample range, $\max - \min$ over the finite entries.

    $$D = \max_i x_i - \min_i x_i$$

    Parameters
    ----------
    values : array-like
        Signal samples; non-finite entries are ignored [any].

    Returns
    -------
    float
        Range in the unit of the signal; 0 for a constant signal, ``nan`` for an
        empty one [any].

    Physical interpretation
    -----------------------
    The span the signal actually used, in its own physical units.  Unlike
    :func:`relative_spread` it is not normalized, so it stays meaningful for a
    signal whose values straddle zero, where dividing by ``max|x|`` is
    dominated by whichever excursion happened to be larger.

    Assumptions
    -----------
    None, but note that the range is the least robust of the scale estimators
    here: it is defined entirely by the two most extreme samples, so one spike
    sets it.  Screen for spikes (see :func:`robust_z_scores`) before reading it
    as the signal's working span.

    References
    ----------
    .. [1] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 1.3.5.6.

    Notes
    -----
    The denominator that turns an absolute residual into a fraction of what the
    channel was actually doing.  A 1 mT residual on a probe that swung 10 mT is
    a 10% model error; the same residual on a probe that swung 1 mT is a
    failure of the model.
    """
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(np.ptp(finite))


# ---------------------------------------------------------------------------
# Threshold crossing
# ---------------------------------------------------------------------------

def noise_band(values: np.ndarray | Iterable[float]) -> tuple[float, float]:
    r"""Mean and population standard deviation of a reference (effect-free) sample.

    $$\mu = \frac{1}{n}\sum_i x_i, \qquad \sigma = \sqrt{\frac{1}{n}\sum_i (x_i - \mu)^2}$$

    Parameters
    ----------
    values : array-like
        Reference stretch of signal; non-finite entries are ignored [any].

    Returns
    -------
    mean : float
        Offset of the reference; ``nan`` for an empty sample [any].
    std : float
        Population (``ddof = 0``) standard deviation; ``nan`` for an empty sample [any].

    Physical interpretation
    -----------------------
    The first two moments of a stretch of signal taken to contain no effect,
    estimating the offset and the amplitude of the measurement noise.

    Assumptions
    -----------
    The sample is genuinely effect-free and stationary over its extent.  A
    drift, a switching transient, or a leaked part of the effect inside the
    reference window inflates the estimated noise and makes any threshold
    built on it too permissive.

    References
    ----------
    .. [1] P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis
           for the Physical Sciences*, 3rd ed., McGraw-Hill (2003), Sec. 1.3.

    Notes
    -----
    Deriving the threshold from the channel's own measured quiet stretch,
    rather than from a global constant, means a noisy channel is judged by its
    own noise -- essential when channels differ in gain, integrator drift and
    cabling.
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
    r"""First time outside the reference window where the signal leaves its noise band.

    $$t^* = \min\big(t_i : i \notin \mathrm{ref},\ |x_i - \mu| > \sigma_{\mathrm{thr}}\,\sigma\big)$$

    with $(\mu, \sigma)$ from :func:`noise_band` over the reference samples.

    Parameters
    ----------
    time : array-like
        Sample times [any].
    values : array-like
        Signal samples [any].
    reference_mask : array-like of bool
        ``True`` on the effect-free reference samples [bool].
    sigma : float
        Threshold in units of the reference noise [-].

    Returns
    -------
    float
        Crossing time in the unit of ``time``; ``nan`` when the reference is too
        short, its noise zero or non-finite, or the signal never emerges [any].

    Physical interpretation
    -----------------------
    A fixed-threshold first-passage detector with the threshold expressed in
    units of the sample's own noise, so the nominal false-alarm probability per
    sample is set by ``sigma`` alone rather than by the channel's gain.

    Assumptions
    -----------
    A well-defined quiet reference stretch of at least two finite samples,
    stationary noise across the whole record, and independent samples --
    oversampled or filtered data has fewer independent samples than points, so
    the effective false-alarm rate is lower than a naive count suggests.  With
    ``n`` samples searched, the expected number of noise-only crossings is
    roughly ``n`` times the per-sample tail probability, which is why a large
    ``sigma`` is used in practice.

    References
    ----------
    .. [1] NIST/SEMATECH *e-Handbook of Statistical Methods* (2012),
           https://doi.org/10.18434/M32189, Sec. 6.3.2 (Shewhart control
           limits).

    Notes
    -----
    Onset detection: the first moment a channel sees something its own
    pre-event noise cannot explain.  Comparing onsets across channels is the
    real test -- a physical event appears everywhere at once, while a
    forward-model or response-matrix artifact does not.
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
    r"""Fraction of consecutive steps in which the sequence decreased.

    $$f = \frac{1}{n - 1}\,\operatorname{count}\big(x_{i+1} < x_i\big)$$

    Parameters
    ----------
    values : array-like
        Ordered iteration history [any].

    Returns
    -------
    float
        Decreasing-step fraction; ``nan`` below two samples [-].

    Physical interpretation
    -----------------------
    The empirical rate of the event "the next sample is smaller than this one"
    over the ``n - 1`` adjacent pairs.  Comparisons involving a non-finite entry
    count as non-decreases, so a broken history is penalized rather than
    silently shortened.

    Assumptions
    -----------
    The sequence is an ordered iteration history in which decrease is the
    desired direction.  A short history makes the fraction coarse -- with four
    iterations it can only take five values.

    References
    ----------
    .. [1] C. T. Kelley, *Iterative Methods for Linear and Nonlinear Equations*,
           SIAM (1995), Sec. 1.2 (convergence histories).

    Notes
    -----
    For a solver error history, ``1.0`` is a cleanly contracting iteration.
    Values near ``0.5`` mean the error is bouncing, which indicates a step that
    is too aggressive or a solution oscillating between two branches -- a
    distinct failure from steady but slow progress, which
    :func:`log10_decay_rate` detects instead.
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
    r"""Least-squares slope of $\log_{10}x$ against iteration index over the tail.

    $$\hat b = \operatorname{slope}\big(\log_{10}x_{n-m+1..n}\ \text{vs}\ 0, 1, \dots, m-1\big)$$

    Parameters
    ----------
    values : array-like
        Iteration history; non-positive and non-finite entries are dropped [any].
    tail : int, optional
        Number of trailing samples fitted; default 5 [-].

    Returns
    -------
    float
        Decades per iteration; ``nan`` below three usable samples [-].

    Physical interpretation
    -----------------------
    The exponential rate constant of the sequence, estimated in log space so
    that a geometric decay becomes a straight-line fit.  Ordinary least squares
    in log space weights relative rather than absolute errors, which is the
    right choice for a quantity spanning orders of magnitude.

    Assumptions
    -----------
    At least three positive finite samples in the tail; non-positive entries
    cannot be logged and are dropped, which silently shortens the window.  The
    fit assumes a single exponential regime, so a history that changes
    behaviour mid-run gives a slope describing neither phase.  Only the tail is
    used, deliberately: the early transient of an iterative solve says nothing
    about whether it is converging now.

    Numerical notes
    ---------------
    ``numpy.polyfit`` of degree 1 on ``log10`` of the tail.

    References
    ----------
    .. [1] C. T. Kelley, *Iterative Methods for Linear and Nonlinear Equations*,
           SIAM (1995), Sec. 1.2 (linear and q-linear convergence).

    Notes
    -----
    A slope of ``-1`` means the error falls by a decade per iteration.  A value
    near zero means the iteration has stagnated -- it is still running but no
    longer improving, which is a different and more troubling outcome than
    hitting an iteration limit while still descending.  What counts as "near
    zero" is a solver policy decision and belongs to the caller.
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
    "dynamic_range",
    "fractional_rms_improvement",
    "lag1_autocorrelation",
    "linear_trend",
    "log10_decay_rate",
    "median_absolute_deviation",
    "monotonic_fraction",
    "noise_band",
    "normalized_residual",
    "outlier_fraction",
    "pearson_correlation",
    "reduced_chi_squared",
    "relative_spread",
    "residual_bias",
    "rms",
    "robust_z_scores",
    "runs_test_z",
    "sigma_threshold_crossing",
    "sigma_unit_factor",
]
