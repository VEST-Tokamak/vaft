"""Pedestal characterization: canonical geometry, amplitudes, uncertainty and detection status.

The EPED-style tanh fit of :func:`vaft.formula.utils.fit_eped_tanh` reduced
to the descriptors a downstream consumer needs -- pedestal top, centre, foot
and full width in the Groebner convention, evaluated values against the
model's asymptotic levels, the steepest gradient, their covariance-propagated
uncertainties -- together with the evidence behind them and a detection
status.  :mod:`vaft.process.profile` is the public import location;
:func:`vaft.process.profile.pedestal_top` is the backward-compatible wrapper
that reduces a characterization to one boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from vaft.formula.utils import _eped_tanh_model, fit_eped_tanh, tanh_pedestal_geometry

__all__ = [
    "characterize_pedestal",
]

#: Widest native tanh scale ``w``, in the fitted coordinate, that still
#: describes a pedestal.  The model's own box allows 0.3, and a fit that comes
#: back near it is a ramp across most of the plasma rather than an edge
#: feature: over the 171 fits of the reference MAST campaign the widths
#: separate into a physical group below 0.11 and 25 fits between 0.20 and 0.29
#: whose "pedestal" spans half the minor radius.  The legacy fitter bounded
#: width at 0.2 for the same reason.  (The canonical full width is twice this.)
PEDESTAL_MAX_WIDTH = 0.15

#: How far the fitted edge feature must drop, from its top to its foot,
#: above the local scatter of the samples around it before a pedestal counts
#: as resolved.  The factor is the one the original whole-window test used:
#: on a pure-noise profile the fitted variation over residual ratio was about
#: 1.3 and on a real pedestal a few hundred, so three sat comfortably between
#: them.  Since #2335 it is applied to the *localized* top-to-foot drop and the
#: scatter within one full width of the feature, not to the curve's range over
#: the whole window, which also counted core variation as pedestal evidence.
PEDESTAL_RESOLUTION_FACTOR = 3.0

#: Smallest ratio of the top-to-foot drop to its own covariance-propagated
#: uncertainty that counts as a detected step: the conventional three-sigma
#: significance.
PEDESTAL_SIGNIFICANCE = 3.0

#: Largest ``|correlation|`` between two pedestal coefficients (``f_ped``,
#: ``f_sep``, ``x_ped``, ``width``) at which the fit still singles out the
#: pedestal; above it the coefficients trade off against each other and the
#: linear covariance no longer describes them.
PEDESTAL_MAX_CORRELATION = 0.995

#: Largest gap between consecutive distinct samples inside ``[0.8, 1.0]`` at
#: which the absence of a resolved step is evidence that there is none: a
#: pedestal with a full width above it would have had a sample inside its
#: gradient region.  Coarser data leave a missing pedestal ``unresolved``.
PEDESTAL_ABSENCE_MAX_GAP = 0.05

#: Outermost sample needed before absence can be claimed: the data must reach
#: the LCFS region where an edge pedestal would sit.
PEDESTAL_ABSENCE_EDGE_COVERAGE = 0.98

#: Largest local residual scatter, as a fraction of the median magnitude of
#: the samples in the window, at which a missing step counts as absent: with
#: it, a step of three times the scatter -- the resolution test -- is at most
#: 30 % of the profile, so a physically relevant pedestal would have been
#: seen.  Noisier data leave a missing pedestal ``unresolved``.
PEDESTAL_ABSENCE_MAX_RELATIVE_SCATTER = 0.1

#: Smallest ratio of the steepest edge gradient to the mean gradient of the
#: fitted profile between the inner edge of the data and the pedestal top.  A
#: pedestal is a localized *steepening*; a smooth peaked profile can still be
#: fitted with a small tanh step near its edge, but its steepest gradient is
#: then no steeper than its core.  On synthetic profiles smooth shapes
#: ((1 - x^2)^2, (1 - x/1.1)^3, with and without 1 % noise) gave at most
#: 1.43 and tanh pedestals (widths 0.03-0.08, pedestal/axis 0.17-0.4) at
#: least 2.1; 1.75 sits between them.  An empirical estimate, not a
#: physical constant.
PEDESTAL_MIN_GRADIENT_CONTRAST = 1.75

#: The detection statuses a characterization can carry.
PEDESTAL_STATUSES = ("present", "absent", "unresolved", "fit_failed", "not_available")

#: What the width numbers on a characterization mean.
WIDTH_DEFINITION = (
    "native_width is the tanh scale w of (1 - tanh((x - center)/w))/2; "
    "full_width = 2 w is the Groebner full width, top = center - full_width/2, "
    "foot = center + full_width/2"
)

_DESCRIPTORS = (
    "top", "center", "foot", "full_width", "native_width",
    "value_top", "value_foot", "value_separatrix", "drop", "height",
    "gradient_max", "gradient_position",
)


@dataclass(frozen=True, eq=False)
class PedestalCharacterization:
    """Canonical descriptors of one fitted pedestal, their uncertainties, evidence and status.

    Geometry is in the fitted ``coordinate``: ``center`` is the tanh's
    symmetry point, ``native_width`` its native scale ``w``, ``full_width``
    the Groebner full width ``2 w``, ``top = center - w`` and
    ``foot = center + w``.  ``value_top``, ``value_foot`` and
    ``value_separatrix`` are the fitted profile *evaluated* at the top, the
    foot and the LCFS (coordinate one); ``height = f_ped - f_sep`` is the
    tanh's *asymptotic* amplitude and ``drop = value_top - value_foot`` the
    actual change across the pedestal, about ``tanh(1) = 0.76`` of the height
    for a pure tanh.  ``gradient_max`` is the steepest slope ``df/dx`` of the
    fitted profile near the pedestal, signed (negative for a profile falling
    outward), at ``gradient_position``.

    ``uncertainties`` holds one-sigma values from ``J C J^T`` of the fit
    covariance, ``nan`` where it is unavailable; ``descriptor_covariance`` is
    the full matrix in the order of ``descriptor_names``.  Every number is
    ``nan`` when the fit failed or was never attempted.  ``status`` is one of
    :data:`PEDESTAL_STATUSES` and ``reason`` explains any status other than
    ``present``.
    """

    quantity: str
    coordinate: str
    status: str
    reason: str = ""
    top: float = float("nan")
    center: float = float("nan")
    foot: float = float("nan")
    full_width: float = float("nan")
    native_width: float = float("nan")
    value_top: float = float("nan")
    value_foot: float = float("nan")
    value_separatrix: float = float("nan")
    drop: float = float("nan")
    height: float = float("nan")
    gradient_max: float = float("nan")
    gradient_position: float = float("nan")
    foot_location: str = "unknown"
    uncertainties: dict = field(default_factory=dict)
    descriptor_covariance: np.ndarray | None = None
    descriptor_names: tuple = _DESCRIPTORS
    evidence: dict = field(default_factory=dict)
    fit: object = None
    span: tuple | None = None
    width_definition: str = WIDTH_DEFINITION

    @property
    def present(self) -> bool:
        """Whether a pedestal was detected and resolved."""
        return self.status == "present"

    def interval(self, name, k=1.0):
        """``(value - k sigma, value + k sigma)`` of one descriptor; ``nan`` bounds without a sigma [same as the descriptor]."""
        value = float(getattr(self, name))
        sigma = float(self.uncertainties.get(name, float("nan")))
        return value - k * sigma, value + k * sigma


def _eped_tanh_derivative(x, f0, f_ped, f_sep, x_ped, width, alpha, beta):
    """Analytic ``df/dx`` of :func:`vaft.formula.utils._eped_tanh_model`."""
    x = np.asarray(x, dtype=float)
    z = (x - x_ped) / width
    out = -0.5 * (f_ped - f_sep) / width * (1.0 - np.tanh(z) ** 2)
    axis_pedestal = f_sep + 0.5 * (f_ped - f_sep) * (1.0 + np.tanh(x_ped / width))
    inside = (x < x_ped) & (x > 0.0)
    if np.any(inside):
        s = x[inside] / x_ped
        base = np.clip(1.0 - s**alpha, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            core = beta * base ** (beta - 1.0) * (-alpha * s ** (alpha - 1.0) / x_ped)
        out[inside] = out[inside] + np.nan_to_num(core) * (f0 - axis_pedestal)
    return out


def _descriptors(theta):
    """The :data:`_DESCRIPTORS` vector of one coefficient set."""
    f0, f_ped, f_sep, x_ped, width, alpha, beta = theta
    top, foot, full_width, *_ = tanh_pedestal_geometry(x_ped, width)
    value_top, value_foot, value_sep = _eped_tanh_model(np.array([top, foot, 1.0]), *theta)
    drop = value_top - value_foot
    # Steepest slope in the direction of the drop, near the pedestal.
    direction = 1.0 if drop >= 0.0 else -1.0
    low, high = max(top - full_width, 1e-9), foot + full_width
    grid = np.linspace(low, high, 801)
    slope = direction * _eped_tanh_derivative(grid, *theta)
    i = int(np.argmin(slope))
    bracket = (grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)])
    refined = minimize_scalar(
        lambda u: direction * float(_eped_tanh_derivative(np.array([u]), *theta)[0]),
        bounds=bracket, method="bounded", options={"xatol": 1e-12},
    )
    position = float(refined.x) if refined.success else float(grid[i])
    gradient = float(_eped_tanh_derivative(np.array([position]), *theta)[0])
    return np.array([
        top, x_ped, foot, full_width, width,
        value_top, value_foot, value_sep, drop, f_ped - f_sep,
        gradient, position,
    ], dtype=float)


#: Positions closer than this, in the fitted coordinate, count as one
#: independent observation: repeated or densely resampled channels do not
#: add spatial information.  Numerical convenience; the diagnostic's own
#: spatial response is not modelled.
PEDESTAL_DISTINCT_SPACING = 1e-3


def _unique_count(values, tolerance=PEDESTAL_DISTINCT_SPACING):
    """Number of distinct positions: dense resampling is not independent data."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(values) > tolerance))


def _absence_evidence(x_used, scatter, magnitude):
    """``(adequate, why_not)``: whether the samples could have shown a pedestal."""
    if x_used.size == 0:
        return False, "no samples"
    if not (np.isfinite(scatter) and magnitude > 0
            and scatter <= PEDESTAL_ABSENCE_MAX_RELATIVE_SCATTER * magnitude):
        return False, (
            f"the local scatter {scatter:.3g} exceeds {PEDESTAL_ABSENCE_MAX_RELATIVE_SCATTER} "
            f"of the profile's median magnitude {magnitude:.3g}, so a relevant pedestal "
            "could hide in it"
        )
    reach = float(np.max(x_used))
    if reach < PEDESTAL_ABSENCE_EDGE_COVERAGE:
        return False, (
            f"the samples end at {reach:.3g}, short of {PEDESTAL_ABSENCE_EDGE_COVERAGE}, "
            "so an edge pedestal could sit outside them"
        )
    edge = np.unique(x_used[(x_used >= 0.8) & (x_used <= 1.0)])
    edge = np.concatenate(([0.8], edge, [min(reach, 1.0)]))
    gap = float(np.max(np.diff(edge))) if edge.size > 1 else 1.0
    if gap > PEDESTAL_ABSENCE_MAX_GAP:
        return False, (
            f"the largest gap between samples in [0.8, 1.0] is {gap:.3g}, wider than "
            f"{PEDESTAL_ABSENCE_MAX_GAP}, so a pedestal could fall between them"
        )
    return True, ""


def characterize_pedestal(
    x,
    values,
    *,
    quantity,
    value_std=None,
    coordinate="psi_norm",
    window=(0.4, 1.05),
    min_points=20,
    max_width=PEDESTAL_MAX_WIDTH,
    min_gradient_points=2,
    min_top_points=2,
    n_starts=4,
):
    """Characterize the edge pedestal of one profile: geometry, amplitudes, gradient, uncertainty and status.

    Fits the EPED-style tanh model (:func:`vaft.formula.utils.fit_eped_tanh`)
    over ``window`` and reduces it to canonical descriptors with
    covariance-propagated uncertainties, the evidence behind them and a
    detection status: ``present``, ``absent``, ``unresolved``, ``fit_failed``
    or ``not_available``.

    Parameters
    ----------
    x : array_like
        Radial coordinate of the profile, in ``coordinate``; LCFS at one [-].
    values : array_like
        Profile samples: any one quantity with a pedestal, e.g. ``n_e``,
        ``T_e``, ``T_i``, ``p_e`` or ``p_total`` [any].
    quantity : str
        What ``values`` is, recorded on the result; required, because the
        pedestals of different quantities need not coincide [n/a].
    value_std : array_like, optional
        One-sigma uncertainty per sample; samples with a non-finite or
        non-positive one are dropped before any count is taken [any].
    coordinate : str, optional
        Which radial coordinate ``x`` is, one of ``rho_tor_norm``,
        ``rho_pol_norm``, ``psi_norm``; default ``psi_norm`` [n/a].
    window : tuple of float, optional
        ``(low, high)`` slice of ``x`` the fit sees [-].
    min_points : int, optional
        Fewest valid samples in ``window``, after every mask, that can
        support the seven-parameter fit [-].
    max_width : float, optional
        Widest native tanh scale ``w`` still counted as a pedestal, in
        ``coordinate`` [-].
    min_gradient_points : int, optional
        Fewest distinct samples between the top and the foot for the width to
        count as resolved [-].
    min_top_points : int, optional
        Fewest distinct samples within one full width inside the top for the
        top to count as supported [-].
    n_starts : int, optional
        Starting points of the multi-start fit [-].

    Returns
    -------
    PedestalCharacterization
        Descriptors, their uncertainties and covariance, evidence, the fit
        and the detection status [-].

    Raises
    ------
    ValueError
        ``x``, ``values`` or ``value_std`` differ in length, or
        ``coordinate`` is not a known radial coordinate.

    Processing steps
    ----------------
    1. Drop samples that are non-finite, outside ``window`` or carry an
       invalid ``value_std``; ``not_available`` if fewer than ``min_points``
       remain.
    2. Fit the tanh model by multi-start least squares, keeping its
       covariance; ``fit_failed`` if no start converges or the coefficients
       are not finite.
    3. Reduce the coefficients to the canonical descriptors and propagate the
       covariance to them as ``J C J^T``.
    4. Qualify the localized feature: the top-to-foot drop against
       :data:`PEDESTAL_RESOLUTION_FACTOR` times the scatter within one full
       width of it and against :data:`PEDESTAL_SIGNIFICANCE` times its own
       uncertainty, and the steepest gradient against
       :data:`PEDESTAL_MIN_GRADIENT_CONTRAST` times the mean gradient inside
       the top; a ``width`` above ``max_width``, a centre outside the data
       or a pedestal coefficient on a bound of the box also fail it.  A
       failure is ``absent`` when the samples could have shown a pedestal
       (they reach :data:`PEDESTAL_ABSENCE_EDGE_COVERAGE` with no gap above
       :data:`PEDESTAL_ABSENCE_MAX_GAP` in ``[0.8, 1]``, and their local
       scatter is below :data:`PEDESTAL_ABSENCE_MAX_RELATIVE_SCATTER` of the
       profile) *and* the fit excludes a relevant step (``|drop| + 2 sigma``
       below three times that scatter fraction of the profile, or for the
       gradient test ``contrast + 2 sigma`` below its threshold); otherwise
       ``unresolved``.
    5. A qualified feature is still ``unresolved`` when the width is not
       spatially resolved (fewer than ``min_gradient_points`` distinct samples
       between top and foot, or ``min_top_points`` inside the top), when the
       covariance is unavailable, when two pedestal coefficients correlate
       above :data:`PEDESTAL_MAX_CORRELATION`, or when another start reached
       an equally good fit elsewhere; otherwise it is ``present``.

    Defaults
    --------
    ``window = (0.4, 1.05)``, ``min_points = 20``, ``min_gradient_points = 2``,
    ``min_top_points = 2`` and ``n_starts = 4`` are numerical convenience: the
    edge with enough samples for seven parameters, and the fewest samples that
    can place a width and a top.  ``max_width = 0.15`` is an empirical
    estimate from the MAST campaign widths documented at
    :data:`PEDESTAL_MAX_WIDTH`.

    Convention
    ----------
    Groebner full width: ``full_width = 2 w``, ``top = center - w``,
    ``foot = center + w`` (:func:`vaft.formula.utils.tanh_pedestal_geometry`),
    the convention of :func:`vaft.formula.equilibrium.modified_tanh_profile`
    and ``BarrierStep``.  ``center`` is not the top.  Positions are in
    ``coordinate`` and never converted; ``value_separatrix`` and
    ``foot_location`` (``inside``, ``at`` or ``outside`` the LCFS within the
    foot's own uncertainty, floored at 1e-3) take the LCFS at coordinate one.
    ``gradient_max`` is ``df/dx`` in the unit of ``values`` per unit
    ``coordinate``, signed.

    Applicability
    -------------
    Machine-independent.  Any one-dimensional profile with a single edge
    pedestal on a normalized radial coordinate: a kinetic-profile file, a
    fitted diagnostic profile or a transport-code output.

    Limitations
    -----------
    One edge pedestal; an internal transport barrier or a second step is not
    modelled, and no smooth-versus-pedestal model comparison is made yet.
    Distinct samples are counted at :data:`PEDESTAL_DISTINCT_SPACING`; the
    diagnostic's spatial response is not compared with the width.
    Uncertainties are linearized; with a weakly identified fit the
    correlation diagnostic flags it, but profile-likelihood or bootstrap
    intervals are not computed.  Mapping and separatrix-position uncertainty
    are not included.

    Provenance
    ----------
    .. [G98] R. J. Groebner and T. N. Carlstrom, *Critical edge parameters for
       H-mode transition in DIII-D*, Plasma Phys. Control. Fusion 40, 673
       (1998): the full-width tanh convention.
    .. [2335] VAFT issue #2335: canonical pedestal conventions, covariance,
       localized resolution test and detection statuses.
    .. [D05] Migration decision D-05: the EPED-style fit and its parameter box.
    """
    from vaft.process.profile import COORDINATES

    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {COORDINATES}, got {coordinate!r}")
    x = np.asarray(x, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if x.shape != values.shape:
        raise ValueError(
            f"x and values must have the same length, got {x.size} and {values.size}"
        )
    sigma = None
    if value_std is not None:
        sigma = np.asarray(value_std, dtype=float).ravel()
        if sigma.shape != x.shape:
            raise ValueError(
                f"value_std has {sigma.size} samples but the profile has {x.size}"
            )

    low, high = window
    finite = np.isfinite(x) & np.isfinite(values)
    inside = finite & (x >= low) & (x <= high)
    valid_sigma = np.ones_like(inside) if sigma is None else (np.isfinite(sigma) & (sigma > 0))
    used = inside & valid_sigma
    evidence = {
        "n_samples": int(x.size),
        "n_nonfinite": int(np.count_nonzero(~finite)),
        "n_outside_window": int(np.count_nonzero(finite & ~inside)),
        "n_invalid_sigma": int(np.count_nonzero(inside & ~valid_sigma)),
        "n_used": int(np.count_nonzero(used)),
        "window": (float(low), float(high)),
        "coordinate": coordinate,
        "width_definition": WIDTH_DEFINITION,
        "weighted": sigma is not None,
    }
    common = dict(quantity=quantity, coordinate=coordinate, evidence=evidence)

    n_used = evidence["n_used"]
    if n_used < min_points:
        return PedestalCharacterization(
            status="not_available",
            reason=(
                f"{n_used} valid samples in {window} after masking, fewer than "
                f"min_points={min_points}"
            ),
            **common,
        )
    x_fit, y_fit = x[used], values[used]
    std_fit = None if sigma is None else sigma[used]
    span = (float(x_fit.min()), float(x_fit.max()))
    evidence["span"] = span
    evidence["n_unique"] = _unique_count(x_fit)

    try:
        fit = fit_eped_tanh(x_fit, y_fit, std_fit, n_starts=n_starts)
    except (RuntimeError, ValueError) as error:
        return PedestalCharacterization(
            status="fit_failed", reason=f"eped_tanh fit did not converge: {error}",
            span=span, **common,
        )
    evidence.update(
        chi_squared=fit.chi_squared, degrees_of_freedom=fit.degrees_of_freedom,
        reduced_chi_squared=fit.reduced_chi_squared, fit_message=fit.message,
        n_starts=fit.n_starts, n_converged=fit.n_converged,
        competing_minima=fit.competing_minima, bound_saturated=fit.bound_saturated,
        covariance_available=fit.covariance_available,
        pedestal_max_correlation=fit.pedestal_max_correlation,
        pedestal_condition_number=fit.pedestal_condition_number,
    )
    theta = np.asarray(fit.coefficients, float)
    if not np.all(np.isfinite(theta)):
        return PedestalCharacterization(
            status="fit_failed", reason="the fit returned a non-finite x_ped or width",
            fit=fit, span=span, **common,
        )

    values_vec = _descriptors(theta)
    covariance = fit.propagate(_descriptors)
    var = np.diag(covariance)
    with np.errstate(invalid="ignore"):
        sigmas = np.where(np.isfinite(var), np.sqrt(np.clip(var, 0.0, None)), np.nan)
    # top/foot through the closed form, which is exact, rather than the
    # finite-difference Jacobian.
    _, _, _, s_top, s_foot, s_full = tanh_pedestal_geometry(
        theta[3], theta[4], fit.covariance[np.ix_((3, 4), (3, 4))] if fit.covariance_available else None,
    )
    descriptor = dict(zip(_DESCRIPTORS, (float(v) for v in values_vec)))
    uncertainties = dict(zip(_DESCRIPTORS, (float(s) for s in sigmas)))
    uncertainties.update(top=s_top, foot=s_foot, full_width=s_full)

    top, foot, full_width = descriptor["top"], descriptor["foot"], descriptor["full_width"]
    tolerance = uncertainties["foot"] if np.isfinite(uncertainties["foot"]) else 0.0
    tolerance = max(tolerance, 1e-3)
    foot_location = (
        "at" if abs(foot - 1.0) <= tolerance else ("inside" if foot < 1.0 else "outside")
    )

    # Localized evidence: distinct samples around the top, across the
    # gradient region and beyond the foot.
    evidence["n_top_region"] = _unique_count(x_fit[(x_fit >= top - full_width) & (x_fit < top)])
    evidence["n_gradient_region"] = _unique_count(x_fit[(x_fit >= top) & (x_fit <= foot)])
    evidence["n_foot_region"] = _unique_count(x_fit[(x_fit > foot) & (x_fit <= foot + full_width)])
    evidence["extrapolated"] = bool(foot > span[1] or top < span[0])
    local = (x_fit >= top - full_width) & (x_fit <= foot + full_width)
    residual = y_fit - fit(x_fit)
    pool = residual[local] if np.count_nonzero(local) >= 3 else residual
    # The residual scatter, as the original test used -- inverse-variance
    # weighted when the fit was, so samples the fit was told to distrust do
    # not count as scatter.  The sigmas' absolute size enters through the
    # significance test, which weighs the drop against its propagated
    # uncertainty.
    if std_fit is None:
        local_scatter = float(np.sqrt(np.mean(pool**2)))
        all_weights = np.ones_like(residual)
    else:
        weights = 1.0 / (std_fit[local] if np.count_nonzero(local) >= 3 else std_fit) ** 2
        local_scatter = float(np.sqrt(np.sum(weights * pool**2) / np.sum(weights)))
        all_weights = 1.0 / std_fit**2
    # A seven-parameter fit absorbs part of the noise where few samples sit,
    # so the local RMS alone underestimates it; the whole-window residual,
    # corrected for the fitted parameters, floors it.
    dof = max(residual.size - 7, 1)
    global_scatter = float(np.sqrt(
        np.sum(all_weights * residual**2) / np.sum(all_weights) * residual.size / dof
    ))
    scatter = max(local_scatter, global_scatter)
    inner = span[0]

    def _contrast(t):
        d = _descriptors(t)
        top_t, value_top_t, gradient_t = d[0], d[5], d[10]
        if not top_t > inner:
            return np.array([np.nan])
        core = abs(float(_eped_tanh_model(np.array([inner]), *t)[0]) - value_top_t) / (top_t - inner)
        return np.array([abs(gradient_t) / core if core > 0 else np.inf])

    contrast = float(_contrast(theta)[0])
    with np.errstate(invalid="ignore"):
        s_contrast = (
            float(np.sqrt(max(fit.propagate(_contrast)[0, 0], 0.0)))
            if np.isfinite(contrast) and fit.covariance_available else float("nan")
        )
    evidence["gradient_contrast"] = contrast
    evidence["gradient_contrast_std"] = s_contrast
    drop, s_drop = descriptor["drop"], uncertainties["drop"]
    significance = abs(drop) / s_drop if np.isfinite(s_drop) and s_drop > 0 else float("nan")
    evidence.update(local_scatter=local_scatter, global_scatter=global_scatter,
                    scatter=scatter, drop_significance=significance,
                    drop_over_scatter=abs(drop) / scatter if scatter > 0 else float("inf"))

    result = dict(
        **descriptor, foot_location=foot_location, uncertainties=uncertainties,
        descriptor_covariance=covariance, fit=fit, span=span, **common,
    )
    magnitude = float(np.median(np.abs(y_fit)))
    adequate, why_not = _absence_evidence(x_fit, scatter, magnitude)
    evidence["absence_evidence_adequate"] = adequate
    # The smallest drop counted as a physically relevant pedestal: the step
    # the resolution test can see at the largest scatter absence allows.
    relevant_drop = PEDESTAL_RESOLUTION_FACTOR * PEDESTAL_ABSENCE_MAX_RELATIVE_SCATTER * magnitude
    drop_excluded = bool(np.isfinite(s_drop) and abs(drop) + 2.0 * s_drop <= relevant_drop)
    evidence["relevant_drop"] = relevant_drop
    evidence["relevant_drop_excluded"] = drop_excluded

    def _missing(reason, excluded):
        """``absent`` only when the data could show a pedestal *and* the fit excludes one."""
        if adequate and excluded:
            return PedestalCharacterization(status="absent", reason=reason, **result)
        why = why_not if not adequate else (
            "the fit's uncertainty does not exclude a relevant pedestal"
        )
        return PedestalCharacterization(
            status="unresolved", reason=f"{reason}; absence not established: {why}", **result,
        )

    x_ped, width = theta[3], theta[4]
    if not span[0] <= x_ped <= span[1]:
        return PedestalCharacterization(
            status="unresolved",
            reason=(
                f"x_ped={x_ped:.4g} lies outside the fitted data, which spans "
                f"[{span[0]:.4g}, {span[1]:.4g}]"
            ),
            **result,
        )
    resolved_drop = abs(drop) > PEDESTAL_RESOLUTION_FACTOR * scatter
    significant = np.isfinite(significance) and significance >= PEDESTAL_SIGNIFICANCE
    if width > max_width:
        return _missing(
            f"the fitted tanh is {width:.4g} wide, more than max_width={max_width}: "
            "that is a ramp across the profile, not a pedestal",
            fit.covariance_available,
        )
    saturated = [name for name in ("f_ped", "f_sep", "x_ped", "width") if name in fit.bound_saturated]
    if saturated:
        reason = f"{saturated[0]} rests on a bound of the model's parameter box"
        if not (resolved_drop and significant):
            return _missing(reason + ", and no localized step stands above the scatter",
                            drop_excluded)
        return PedestalCharacterization(status="unresolved", reason=reason, **result)
    if not resolved_drop:
        return _missing(
            f"the fitted edge feature varies by {abs(drop):.4g} from top to foot, not "
            f"{PEDESTAL_RESOLUTION_FACTOR}x the local scatter {scatter:.4g}: "
            "no pedestal resolved",
            drop_excluded,
        )
    if np.isfinite(contrast) and contrast < PEDESTAL_MIN_GRADIENT_CONTRAST:
        return _missing(
            f"the steepest edge gradient is only {contrast:.3g} times the mean gradient "
            f"inside the top (need {PEDESTAL_MIN_GRADIENT_CONTRAST}): a smooth profile, "
            "no localized steepening",
            bool(np.isfinite(s_contrast)
                 and contrast + 2.0 * s_contrast < PEDESTAL_MIN_GRADIENT_CONTRAST),
        )
    if not fit.covariance_available:
        return PedestalCharacterization(
            status="unresolved",
            reason="the fit covariance could not be estimated, so the step's significance is unknown",
            **result,
        )
    if not significant:
        return _missing(
            f"the fitted edge feature varies by {abs(drop):.4g} from top to foot, only "
            f"{significance:.3g} times its uncertainty {s_drop:.3g} (need "
            f"{PEDESTAL_SIGNIFICANCE}): no pedestal resolved",
            drop_excluded,
        )
    if evidence["n_gradient_region"] < min_gradient_points:
        return PedestalCharacterization(
            status="unresolved",
            reason=(
                f"a step is present but only {evidence['n_gradient_region']} distinct samples lie "
                f"between its top and foot (need {min_gradient_points}): the width is not resolved"
            ),
            **result,
        )
    if evidence["n_top_region"] < min_top_points:
        return PedestalCharacterization(
            status="unresolved",
            reason=(
                f"a step is present but only {evidence['n_top_region']} distinct samples lie within "
                f"one full width inside its top (need {min_top_points}): the top is not supported"
            ),
            **result,
        )
    correlation = fit.pedestal_max_correlation
    if not np.isfinite(correlation) or correlation > PEDESTAL_MAX_CORRELATION:
        return PedestalCharacterization(
            status="unresolved",
            reason=(
                f"two pedestal coefficients correlate at |rho|={correlation:.4g} > "
                f"{PEDESTAL_MAX_CORRELATION}: the fit does not identify the pedestal"
            ),
            **result,
        )
    if fit.competing_minima:
        return PedestalCharacterization(
            status="unresolved",
            reason=(
                f"{fit.competing_minima} other start(s) fit equally well with the pedestal "
                "more than one width away: the data do not single out one pedestal"
            ),
            **result,
        )
    return PedestalCharacterization(status="present", **result)
