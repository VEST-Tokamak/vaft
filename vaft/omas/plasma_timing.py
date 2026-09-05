"""The plasma window of a discharge, from an ODS, with its provenance (issue #409).

The generic detectors in :mod:`vaft.process.onset` answer *when does this
waveform become active* about one array.  This module decides *which* of the
recorded signals is allowed to answer for the plasma, and says so:

* the H-alpha line found **by label** on the slow filterscope is authoritative
  for the onset and the offset -- it is optical, so the coil-firing pickup
  every magnetic diagnostic carries cannot trigger it;
* a fast-filterscope H-alpha line is the first fallback, accepted only after
  the same usability checks (label, finite, not railed, live baseline, not
  invalidated);
* the plasma-current principal pulse is the final fallback and is always
  computed as the cross-check, gated by the same validity floor;
* a raw magnetic first crossing is never used.

Every rule the detectors run with -- the shared plasma-analysis range, the
baseline before it, the H-alpha and Ip recipes, the usability
floors and the cross-check tolerances -- comes from the ``plasma_timing``
block of ``vest.yaml`` through
:func:`vaft.machine_mapping.utils.resolve_plasma_timing_policy`.  Nothing
here writes the ODS.  Both detectors measure a *positive* excess over the
baseline: the stored H-alpha intensity is positive-going (the mapper negates
the raw record) and the stored plasma current is positive-going under the
present VEST convention; a sign migration of either node (#275) must be
matched here.

A missing H-alpha channel is a normal operating state, recorded as the
``fallback_reason`` of a timing found from the current; a disagreement between
the light and the current is a flag, never a silent ``min``/``max``.  When
neither source shows a plasma the result says so with ``found = False`` -- a
window is never assumed from the analysis range.  :class:`PlasmaTiming` is
timing and provenance only: representative peaks and other derived features
are a separate concept (``PlasmaFeatures``, later) that consumes this window,
and actuator onsets (``DischargeTiming``) are independent events that never
enter the hierarchy above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vaft.formula.statistics import median_absolute_deviation
from vaft.machine_mapping.spectrometer_uv import (
    CHANNEL_CADENCE_HZ,
    SIGNALS,
    legacy_time_shift_s,
)
from vaft.machine_mapping.utils import (
    MIN_BASELINE_S,
    CroppedRecord,
    PlasmaTimingPolicy,
    crop_to_span,
    resolve_plasma_timing_policy,
)
from vaft.ods_access import path_value
from vaft.process.onset import PulseWindow, robust_baseline
from vaft.validation.imas import (
    VALIDITY_SUSPECT,
    is_condemned_channel,
    read_validity,
    read_validity_timed,
    resolve_signal_time,
    valid_fraction,
)

__all__ = [
    "AGREEMENT_CONSISTENT",
    "AGREEMENT_HALPHA_LEADS_IP_LARGE",
    "AGREEMENT_HALPHA_ONLY",
    "AGREEMENT_IP_BEFORE_HALPHA",
    "AGREEMENT_IP_ONLY",
    "AGREEMENT_NONE",
    "HALPHA_LABEL",
    "AnalysisSpan",
    "HalphaSource",
    "HalphaUsability",
    "IP_BASE",
    "IP_CHECKS",
    "MIN_BASELINE_S",
    "PlasmaTiming",
    "PlasmaTimingError",
    "SOURCE_H_FAST",
    "SOURCE_H_PRIMARY",
    "SOURCE_H_SECONDARY",
    "SOURCE_IP",
    "USABILITY_CHECKS",
    "WAVELENGTH_TOLERANCE_M",
    "analysis_span",
    "halpha_sources",
    "halpha_usability",
    "halpha_window",
    "ip_window",
    "plasma_timing",
]

HALPHA_LABEL = "H-alpha_6563"
WAVELENGTH_TOLERANCE_M = 1.0e-9
IP_BASE = "magnetics.ip.0"

SOURCE_H_PRIMARY = "h_alpha_primary"
SOURCE_H_FAST = "h_alpha_fast"
SOURCE_H_SECONDARY = "h_alpha_secondary"
SOURCE_IP = "ip_principal"

AGREEMENT_CONSISTENT = "consistent"
AGREEMENT_IP_BEFORE_HALPHA = "ip_before_halpha"
AGREEMENT_HALPHA_LEADS_IP_LARGE = "halpha_leads_ip_large"
AGREEMENT_HALPHA_ONLY = "halpha_only"
AGREEMENT_IP_ONLY = "ip_only"
AGREEMENT_NONE = "none"

#: The H-alpha usability checks, in the order they are applied; the first
#: failure is the ``reason`` and the later checks are not evaluated.
USABILITY_CHECKS = ("present", "label", "finite", "not_railed", "baseline_live", "validity")
#: The plasma-current checks; a failure makes the current unusable as a
#: source and as a cross-check, recorded in ``fallback_reason``.
IP_CHECKS = ("present", "finite", "validity")


class PlasmaTimingError(ValueError):
    """Raised when the ODS lacks what any plasma timing needs."""


# ---------------------------------------------------------------------------
# The analysis span
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSpan:
    """Where the plasma is searched for and where its baseline is measured.

    The search covers ``[tstart, tend)`` -- the shared ``plasma_analysis``
    window -- and the baseline the stretch ``[baseline_start, tstart)``
    before it, which on the analysis grid is the vacuum stretch every product
    carries.  A record is cropped to ``[baseline_start, tend)`` before any
    detector sees it, so a full-discharge product behaves exactly like a
    product built on the analysis window.
    """

    tstart: float
    tend: float
    baseline_start: float
    window: str

    @property
    def baseline_lead_s(self) -> float:
        return float(self.tstart - self.baseline_start)

    def record(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "tstart": self.tstart,
            "tend": self.tend,
            "baseline_start": self.baseline_start,
        }


def _resolved(
    policy: PlasmaTimingPolicy | None, span: AnalysisSpan | None = None
) -> tuple[PlasmaTimingPolicy, AnalysisSpan]:
    if policy is None:
        policy = resolve_plasma_timing_policy()
    if span is None:
        span = analysis_span(policy)
    return policy, span


def analysis_span(policy: PlasmaTimingPolicy | None = None) -> AnalysisSpan:
    """The :class:`AnalysisSpan` the configured policy prescribes."""
    if policy is None:
        policy = resolve_plasma_timing_policy()
    window = policy.window
    return AnalysisSpan(
        tstart=float(window.tstart),
        tend=float(window.tend),
        baseline_start=policy.baseline_start,
        window=str(window.name),
    )


def _crop(time: Any, values: Any, span: AnalysisSpan) -> CroppedRecord:
    """Crop a record to the span with the shared rule (:func:`crop_to_span`)."""
    try:
        return crop_to_span(
            time, values, baseline_start=span.baseline_start, tstart=span.tstart, tend=span.tend
        )
    except ValueError as exc:
        raise PlasmaTimingError(str(exc)) from exc


# ---------------------------------------------------------------------------
# H-alpha sources and their usability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HalphaSource:
    """One configured H-alpha line: where it is stored and what role it plays."""

    channel: int
    line: int
    label: str
    cadence_hz: float
    role: str

    @property
    def line_base(self) -> str:
        return f"spectrometer_uv.channel.{self.channel}.processed_line.{self.line}"

    @property
    def base(self) -> str:
        return f"{self.line_base}.intensity"

    def record(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "line": self.line,
            "label": self.label,
            "cadence_hz": self.cadence_hz,
            "role": self.role,
            "base": self.base,
        }


def halpha_sources(*, label: str = HALPHA_LABEL) -> tuple[HalphaSource, ...]:
    """The configured lines carrying ``label``, slowest digitizer first.

    The order is the hierarchy: the slow-DAQ line is ``h_alpha_primary``, a
    line on a faster digitizer ``h_alpha_fast`` (its stored samples are
    resampled from the native rate), and another line at the primary's own
    rate ``h_alpha_secondary``.  Nothing here indexes a channel by position --
    the label is what identifies H-alpha -- and a configured channel without
    a digitizer rate is a configuration error, not a silent demotion.
    """
    entries = []
    for _, channel, line, declared, _ in SIGNALS:
        if declared != label:
            continue
        if channel not in CHANNEL_CADENCE_HZ:
            raise PlasmaTimingError(
                f"spectrometer_uv channel {channel} carries {label!r} but has no entry in "
                "CHANNEL_CADENCE_HZ; the source hierarchy is ordered by digitizer rate"
            )
        entries.append((float(CHANNEL_CADENCE_HZ[channel]), int(channel), int(line), str(declared)))
    entries.sort()
    sources: list[HalphaSource] = []
    for index, (cadence, channel, line, declared) in enumerate(entries):
        if index == 0:
            role = SOURCE_H_PRIMARY
        elif cadence > entries[0][0]:
            role = SOURCE_H_FAST
        else:
            role = SOURCE_H_SECONDARY
        sources.append(HalphaSource(channel, line, declared, cadence, role))
    return tuple(sources)


@dataclass(frozen=True)
class HalphaUsability:
    """Whether one H-alpha line may answer for the plasma, check by check.

    ``checks`` holds every check that was evaluated, in
    :data:`USABILITY_CHECKS` order, and stops at the first failure, which is
    the ``reason``.  ``metrics`` are the numbers the checks were made with;
    the peak-over-noise figure among them is recorded and is *not* a check --
    a healthy channel that saw no light is evidence about the shot, not about
    the channel.
    """

    source: HalphaSource
    usable: bool
    checks: Mapping[str, bool] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "source": self.source.record(),
            "usable": self.usable,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "notes": dict(self.notes),
            "reason": self.reason,
        }


def _declared_wavelength(source: HalphaSource) -> float | None:
    for _, channel, line, _, wavelength in SIGNALS:
        if channel == source.channel and line == source.line:
            return float(wavelength)
    return None


def halpha_usability(
    ods: Any,
    source: HalphaSource,
    *,
    span: AnalysisSpan | None = None,
    policy: PlasmaTimingPolicy | None = None,
) -> HalphaUsability:
    """Apply the configured usability checks to one H-alpha line.

    The checks, in order: ``present`` (data and a time coordinate of equal
    length, with at least ``min_samples`` inside the span); ``label`` (the
    stored label is the configured one and, when a wavelength is stored, it
    is H-alpha's); ``finite``; ``not_railed`` (a fast digitizer rails at
    ``rail_level``; a clipped peak moves a fraction-of-peak onset -- judged
    on the search stretch, where the light is); ``baseline_live`` (a
    quantized, flat baseline cannot carry a noise-relative threshold);
    ``validity`` (the node is not marked invalid, and enough of its timed
    validity inside the span is at least suspect).  Notes record what the
    mapper did to a fast-DAQ line: it is resampled, and its own time axis
    was shifted onto the discharge clock by ``time_shift_s``.
    """
    policy, span = _resolved(policy, span)
    limits = policy.usability
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    notes: dict[str, Any] = {}

    def verdict(reason: str | None) -> HalphaUsability:
        return HalphaUsability(source, reason is None, checks, metrics, notes, reason)

    base = source.base
    data = path_value(ods, f"{base}.data")
    time = resolve_signal_time(ods, base)
    if data is None or time is None:
        checks["present"] = False
        notes["absent"] = True
        return verdict("present")
    y_all = np.asarray(data, dtype=float).reshape(-1)
    t_all = np.asarray(time, dtype=float).reshape(-1)
    if y_all.size != t_all.size or y_all.size == 0:
        checks["present"] = False
        metrics["n_samples"] = float(y_all.size)
        return verdict("present")
    cropped = _crop(t_all, y_all, span)
    t, y, search = cropped.t, cropped.y, cropped.search
    metrics["n_samples"] = float(t.size)
    notes["baseline_inside_search"] = "baseline_inside_search" in cropped.flags
    checks["present"] = t.size >= int(limits["min_samples"])
    if not checks["present"]:
        return verdict("present")

    stored = path_value(ods, f"{source.line_base}.label")
    checks["label"] = stored is not None and str(stored) == source.label
    wavelength = path_value(ods, f"{source.line_base}.wavelength_central")
    declared = _declared_wavelength(source)
    if checks["label"] and wavelength is not None and declared is not None:
        checks["label"] = bool(abs(float(wavelength) - declared) <= WAVELENGTH_TOLERANCE_M)
    notes["stored_label"] = None if stored is None else str(stored)
    if not checks["label"]:
        return verdict("label")

    finite = np.isfinite(y)
    metrics["finite_fraction"] = float(finite.mean())
    checks["finite"] = metrics["finite_fraction"] >= float(limits["min_finite_fraction"])
    if not checks["finite"]:
        return verdict("finite")

    rail = float(limits["rail_fraction"]) * float(limits["rail_level"])
    lit = search & finite
    metrics["railed_fraction"] = float(np.mean(np.abs(y[lit]) >= rail)) if lit.any() else 0.0
    checks["not_railed"] = metrics["railed_fraction"] < float(limits["max_railed_fraction"])
    if not checks["not_railed"]:
        return verdict("not_railed")

    if cropped.baseline_mask is not None:
        baseline = cropped.baseline_mask & finite
    else:
        baseline = (np.arange(t.size) < max(2, t.size // 5)) & finite
    ref_values = y[baseline]
    if ref_values.size >= 2:
        median, sigma = robust_baseline(ref_values)
        mad = float(median_absolute_deviation(ref_values))
        live = bool(ref_values.min() != ref_values.max())
    else:
        median, sigma, mad, live = float("nan"), 0.0, 0.0, False
    metrics["baseline_median"] = float(median)
    metrics["baseline_mad"] = mad
    peak_excess = float(np.nanmax(y[search])) - float(median) if search.any() and np.isfinite(median) else float("nan")
    metrics["peak_excess"] = peak_excess
    metrics["peak_over_sigma"] = peak_excess / sigma if sigma > 0.0 and np.isfinite(peak_excess) else float("nan")
    checks["baseline_live"] = live and mad >= float(limits["min_baseline_mad"])
    if not checks["baseline_live"]:
        return verdict("baseline_live")

    scalar = read_validity(ods, base)
    notes["validity"] = scalar
    fraction = 1.0
    if read_validity_timed(ods, base) is not None:
        inside = (t_all >= span.baseline_start) & (t_all < span.tend)
        fraction = valid_fraction(ods, base, times=t_all, window=inside, min_validity=VALIDITY_SUSPECT)
        fraction = 0.0 if not np.isfinite(fraction) else float(fraction)
    metrics["valid_fraction"] = fraction
    # The scalar is the worst state reached; a channel condemned outright has
    # no usable sample anywhere, and one that was good early and bad late is
    # judged by how much of the span its timed validity leaves (#424).
    checks["validity"] = not is_condemned_channel(ods, base, min_validity=VALIDITY_SUSPECT) and (
        fraction >= float(limits["min_valid_fraction"])
    )
    if not checks["validity"]:
        return verdict("validity")

    grid_dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    resampled = bool(np.isfinite(grid_dt) and grid_dt * source.cadence_hz > 1.5)
    notes["stored_dt"] = grid_dt
    notes["native_rate_hz"] = source.cadence_hz
    notes["resampled"] = resampled
    if resampled:
        shot = path_value(ods, "dataset_description.data_entry.pulse")
        notes["time_shift_s"] = None if shot is None else legacy_time_shift_s(int(shot))
    return verdict(None)


# ---------------------------------------------------------------------------
# Windows per source
# ---------------------------------------------------------------------------


def _halpha_cropped(ods: Any, source: HalphaSource, span: AnalysisSpan) -> CroppedRecord:
    data = path_value(ods, f"{source.base}.data")
    time = resolve_signal_time(ods, source.base)
    if data is None or time is None:
        raise PlasmaTimingError(f"{source.base} carries no data")
    return _crop(time, data, span)


def halpha_window(
    ods: Any,
    source: HalphaSource,
    *,
    span: AnalysisSpan | None = None,
    policy: PlasmaTimingPolicy | None = None,
) -> PulseWindow:
    """The envelope of H-alpha activity inside the span, with the ``h_alpha`` rule.

    The envelope of every segment, not the principal one: the pre-ionization
    light some campaigns show before the current is plasma light, and a
    window that skipped it would start late.  The caller sees it as
    ``multiple_segments``.
    """
    policy, span = _resolved(policy, span)
    return _halpha_cropped(ods, source, span).detect(policy.h_alpha)


def _ip_cropped(ods: Any, span: AnalysisSpan) -> tuple[CroppedRecord, dict[str, bool]]:
    """The plasma current inside the span and its :data:`IP_CHECKS`."""
    data = path_value(ods, f"{IP_BASE}.data")
    time = resolve_signal_time(ods, IP_BASE)
    if data is None or time is None:
        raise PlasmaTimingError(
            f"{IP_BASE} carries no plasma current; every plasma timing needs it as the cross-check"
        )
    y = np.asarray(data, dtype=float).reshape(-1)
    t = np.asarray(time, dtype=float).reshape(-1)
    if y.size == 0 or y.size != t.size:
        raise PlasmaTimingError(f"{IP_BASE}: {y.size} samples against {t.size} time instants")
    cropped = _crop(t, y, span)
    checks = {
        "present": bool(cropped.t.size >= 2),
        "finite": bool(np.isfinite(cropped.y).all()) if cropped.t.size else False,
        "validity": not is_condemned_channel(ods, IP_BASE, min_validity=VALIDITY_SUSPECT),
    }
    return cropped, checks


def ip_window(
    ods: Any,
    *,
    span: AnalysisSpan | None = None,
    policy: PlasmaTimingPolicy | None = None,
) -> PulseWindow:
    """The principal plasma-current pulse inside the span, with the ``ip`` rule.

    Raises :class:`PlasmaTimingError` when the current is absent or has no
    samples inside the span; :func:`plasma_timing` additionally refuses a
    current that fails :data:`IP_CHECKS` as a source.
    """
    policy, span = _resolved(policy, span)
    cropped, checks = _ip_cropped(ods, span)
    if not checks["present"]:
        raise PlasmaTimingError(f"{IP_BASE} has no samples inside {span.baseline_start}-{span.tend} s")
    return cropped.detect(policy.ip)


# ---------------------------------------------------------------------------
# The plasma timing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlasmaTiming:
    """The plasma window and how it was found.

    ``onset``/``offset`` are the window the consumer should use; both come
    from ``source`` (the light when it is usable, the current otherwise --
    never one edge from each).  ``optical`` and ``ip`` are the windows each
    source produced, ``candidates`` the usability verdict of every H-alpha
    line examined, ``ip_checks`` the current's, and ``agreement`` the
    relation between the light and the current.  ``found`` is ``False`` when
    neither source shows a plasma; ``window`` is then ``None`` and no consumer
    may substitute the analysis range for it without saying so.
    """

    onset: float | None
    offset: float | None
    source: str | None
    optical: PulseWindow | None
    optical_source: HalphaSource | None
    ip: PulseWindow | None
    agreement: str
    onset_delta_s: float | None
    offset_delta_s: float | None
    candidates: tuple[HalphaUsability, ...]
    ip_checks: Mapping[str, bool]
    fallback_reason: str | None
    flags: tuple[str, ...]
    span: AnalysisSpan

    @property
    def onset_source(self) -> str | None:
        return self.source

    @property
    def offset_source(self) -> str | None:
        return self.source

    @property
    def found(self) -> bool:
        return self.onset is not None and self.offset is not None

    @property
    def window(self) -> tuple[float, float] | None:
        if not self.found:
            return None
        return (float(self.onset), float(self.offset))

    def summary(self) -> dict[str, Any]:
        """What a metrics record or a manifest carries: the verdict and its provenance."""
        return {
            "onset": self.onset,
            "offset": self.offset,
            "source": self.source,
            "agreement": self.agreement,
            "onset_delta_s": self.onset_delta_s,
            "offset_delta_s": self.offset_delta_s,
            "fallback_reason": self.fallback_reason,
            "flags": list(self.flags),
            "ip_window": None if self.ip is None or not self.ip.found else [self.ip.start, self.ip.end],
        }

    def record(self) -> dict[str, Any]:
        """Everything, including both detector windows and every candidate's usability."""
        return {
            **{key: value for key, value in self.summary().items() if key != "ip_window"},
            "span": self.span.record(),
            "optical_source": None if self.optical_source is None else self.optical_source.record(),
            "optical": None if self.optical is None else self.optical.as_dict(),
            "ip": None if self.ip is None else self.ip.as_dict(),
            "ip_checks": dict(self.ip_checks),
            "candidates": [c.record() for c in self.candidates],
        }


def _light_outcome(candidate: HalphaUsability, window: PulseWindow | None) -> str:
    """One phrase per H-alpha candidate for ``fallback_reason``."""
    if not candidate.usable:
        return f"{candidate.source.role}: {candidate.reason}"
    if window is None:
        return f"{candidate.source.role}: usable, not consulted"
    return f"{candidate.source.role}: usable but no light ({', '.join(window.onset.flags) or 'no window'})"


def _agreement(onset_delta: float, tolerances: Mapping[str, float]) -> str:
    if onset_delta < -float(tolerances["onset_tolerance_s"]):
        return AGREEMENT_IP_BEFORE_HALPHA
    if onset_delta > float(tolerances["lag_tolerance_s"]):
        return AGREEMENT_HALPHA_LEADS_IP_LARGE
    return AGREEMENT_CONSISTENT


def plasma_timing(
    ods: Any,
    *,
    policy: PlasmaTimingPolicy | None = None,
) -> PlasmaTiming:
    """Find the plasma window with the configured source hierarchy.

    H-alpha lines are examined in :func:`halpha_sources` order; the first
    usable one answers for the light, whether or not it saw any -- a usable
    slow channel that stayed dark is not overruled by a fast one, it is
    evidence.  The plasma-current window is computed whenever the current
    passes :data:`IP_CHECKS`.  The onset and the offset then come from the
    light when it found a window and from the current otherwise, and the two
    are compared with the configured tolerances:

    ==============  ==============  ====================  =======================
    light           current         window from           agreement
    ==============  ==============  ====================  =======================
    found           found, agrees   light                 ``consistent``
    found           earlier         light                 ``ip_before_halpha``
    found           much later      light                 ``halpha_leads_ip_large``
    found           none            light                 ``halpha_only``
    dark / unusable found           current               ``ip_only``
    dark / unusable none            none (``found=False``) ``none``
    ==============  ==============  ====================  =======================

    ``fallback_reason`` explains why the light did not answer, check by check,
    and, when the current did not either, why not.  Only a product without
    ``magnetics.ip`` raises.
    """
    policy, span = _resolved(policy)
    flags: list[str] = []

    candidates: list[HalphaUsability] = []
    optical: PulseWindow | None = None
    optical_source: HalphaSource | None = None
    for source in halpha_sources():
        usability = halpha_usability(ods, source, span=span, policy=policy)
        candidates.append(usability)
        if usability.usable and optical is None:
            cropped = _halpha_cropped(ods, source, span)
            optical, optical_source = cropped.detect(policy.h_alpha), source
            flags.extend(cropped.flags)

    ip_cropped, ip_checks = _ip_cropped(ods, span)
    ip_usable = all(ip_checks.values())
    ip = ip_cropped.detect(policy.ip) if ip_usable else None
    if ip is not None:
        flags.extend(ip_cropped.flags)
    else:
        flags.append("ip_unusable")

    light_found = optical is not None and optical.found
    ip_found = ip is not None and ip.found
    onset_delta = offset_delta = None
    if light_found and ip_found:
        onset_delta = float(ip.start - optical.start)
        offset_delta = float(ip.end - optical.end)
        if abs(offset_delta) > float(policy.agreement["offset_tolerance_s"]):
            flags.append("offset_disagreement")

    light_reasons = [
        _light_outcome(c, optical if c.source is optical_source else None) for c in candidates
    ] or ["no H-alpha line is configured"]
    if ip is None:
        failed = ", ".join(name for name, ok in ip_checks.items() if not ok)
        ip_reason = f"{SOURCE_IP}: {failed}"
    else:
        ip_reason = f"{SOURCE_IP}: {', '.join(ip.onset.flags) or 'no window'}"

    fallback_reason: str | None = None
    if light_found:
        chosen, source_name = optical, optical_source.role
        if optical_source.role != SOURCE_H_PRIMARY:
            flags.append("optical_fallback_fast" if optical_source.role == SOURCE_H_FAST else "optical_fallback_secondary")
            fallback_reason = "; ".join(r for c, r in zip(candidates, light_reasons) if not c.usable) or None
        if ip_found:
            agreement = _agreement(onset_delta, policy.agreement)
            if agreement != AGREEMENT_CONSISTENT:
                flags.append(agreement)
        else:
            agreement = AGREEMENT_HALPHA_ONLY
            flags.append("ip_no_pulse" if ip is not None else "ip_unusable")
    elif ip_found:
        chosen, source_name = ip, SOURCE_IP
        agreement = AGREEMENT_IP_ONLY
        if optical is not None:
            flags.append("halpha_dark_with_ip_pulse")
        fallback_reason = "; ".join(light_reasons)
    else:
        chosen, source_name = None, None
        agreement = AGREEMENT_NONE
        flags.append("no_plasma_timing")
        fallback_reason = "; ".join(light_reasons + [ip_reason])

    onset = offset = None
    if chosen is not None:
        flags.extend(chosen.flags)
        onset, offset = float(chosen.start), float(chosen.end)

    return PlasmaTiming(
        onset=onset,
        offset=offset,
        source=source_name,
        optical=optical,
        optical_source=optical_source,
        ip=ip,
        agreement=agreement,
        onset_delta_s=onset_delta,
        offset_delta_s=offset_delta,
        candidates=tuple(candidates),
        ip_checks=ip_checks,
        fallback_reason=fallback_reason,
        flags=tuple(dict.fromkeys(flags)),
        span=span,
    )
