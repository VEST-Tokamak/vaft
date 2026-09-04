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
  computed as the cross-check;
* a raw magnetic first crossing is never used.

Every rule the detectors run with -- the shared plasma-analysis range, the
baseline reference before it, the H-alpha and Ip recipes, the usability
floors and the cross-check tolerances -- comes from the ``plasma_timing``
block of ``vest.yaml`` through
:func:`vaft.machine_mapping.utils.resolve_plasma_timing_policy`.  Nothing
here is a magic number, and nothing here writes the ODS.

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

from vaft.machine_mapping.spectrometer_uv import (
    CHANNEL_CADENCE_HZ,
    SIGNALS,
    legacy_time_shift_s,
)
from vaft.machine_mapping.utils import PlasmaTimingPolicy, resolve_plasma_timing_policy
from vaft.ods_access import path_exists, path_value
from vaft.process.onset import MAD_TO_SIGMA, PulseWindow, active_window, robust_baseline
from vaft.validation.imas import (
    VALIDITY_INVALID,
    VALIDITY_SUSPECT,
    read_validity,
    read_validity_timed,
    resolve_signal_time,
    validity_mask,
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
    "MIN_REFERENCE_S",
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
#: A baseline reference shorter than this (a product built from the plasma
#: range onward) is not trusted on its own; the detectors then use their own
#: leading fraction of the record and the timing is flagged.
MIN_REFERENCE_S = 5.0e-3
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

#: The usability checks, in the order they are applied; the first failure is
#: the ``reason`` and the later checks are not evaluated.
USABILITY_CHECKS = ("present", "label", "finite", "not_railed", "baseline_live", "validity")

_SPECTROMETER_TIME = "spectrometer_uv.time"


class PlasmaTimingError(ValueError):
    """Raised when the ODS lacks what any plasma timing needs."""


# ---------------------------------------------------------------------------
# The analysis span
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSpan:
    """Where the plasma is searched for and where its baseline is measured.

    The search covers ``[tstart, tend)`` -- the shared ``plasma_analysis``
    window -- and the reference the stretch ``[reference_start, tstart)``
    before it, which on the analysis grid is the vacuum stretch every product
    carries.  A record is cropped to ``[reference_start, tend)`` before any
    detector sees it, so a full-discharge product behaves exactly like a
    product built on the analysis window.
    """

    tstart: float
    tend: float
    reference_start: float
    window: str

    @property
    def reference_lead_s(self) -> float:
        return float(self.tstart - self.reference_start)

    def record(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "tstart": self.tstart,
            "tend": self.tend,
            "reference_start": self.reference_start,
        }


def analysis_span(policy: PlasmaTimingPolicy | None = None) -> AnalysisSpan:
    """The :class:`AnalysisSpan` the configured policy prescribes."""
    if policy is None:
        policy = resolve_plasma_timing_policy()
    window = policy.window
    return AnalysisSpan(
        tstart=float(window.tstart),
        tend=float(window.tend),
        reference_start=float(window.tstart - policy.reference_lead_s),
        window=str(window.name),
    )


def _crop(time: Any, values: Any, span: AnalysisSpan):
    """Crop a record to the span; return ``(t, y, reference_mask, search_mask, flags)``.

    ``reference_mask`` is ``None`` when the record carries less than
    :data:`MIN_REFERENCE_S` before ``tstart``: the detector then falls back to
    its own leading fraction and the caller records ``reference_inside_search``.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise PlasmaTimingError(f"time has {t.size} samples but the signal has {y.size}")
    keep = (t >= span.reference_start) & (t < span.tend)
    t, y = t[keep], y[keep]
    reference = t < span.tstart
    search = ~reference
    flags: tuple[str, ...] = ()
    reference_mask: np.ndarray | None = reference
    if t.size and reference.any():
        covered = float(t[reference][-1] - t[reference][0]) if reference.sum() > 1 else 0.0
    else:
        covered = 0.0
    if covered < MIN_REFERENCE_S:
        reference_mask = None
        flags = ("reference_inside_search",)
    return t, y, reference_mask, search, flags


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
    def base(self) -> str:
        return f"spectrometer_uv.channel.{self.channel}.processed_line.{self.line}.intensity"

    @property
    def line_base(self) -> str:
        return f"spectrometer_uv.channel.{self.channel}.processed_line.{self.line}"

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
    the label is what identifies H-alpha.
    """
    entries = [
        (CHANNEL_CADENCE_HZ.get(channel, float("inf")), channel, line, declared)
        for _, channel, line, declared, _ in SIGNALS
        if declared == label
    ]
    entries.sort()
    sources: list[HalphaSource] = []
    for index, (cadence, channel, line, declared) in enumerate(entries):
        if index == 0:
            role = SOURCE_H_PRIMARY
        elif cadence > entries[0][0]:
            role = SOURCE_H_FAST
        else:
            role = SOURCE_H_SECONDARY
        sources.append(HalphaSource(int(channel), int(line), str(declared), float(cadence), role))
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


def _finite_fraction(y: np.ndarray) -> float:
    return float(np.isfinite(y).mean()) if y.size else 0.0


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
    ``rail_level``; a clipped peak moves a fraction-of-peak onset);
    ``baseline_live`` (a quantized, flat baseline cannot carry a
    noise-relative threshold); ``validity`` (the node is not marked invalid,
    and enough of its timed validity inside the span is at least suspect).
    """
    if policy is None:
        policy = resolve_plasma_timing_policy()
    if span is None:
        span = analysis_span(policy)
    limits = policy.usability
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    notes: dict[str, Any] = {}

    def verdict(reason: str | None) -> HalphaUsability:
        return HalphaUsability(
            source=source,
            usable=reason is None,
            checks=checks,
            metrics=metrics,
            notes=notes,
            reason=reason,
        )

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
    t, y, reference_mask, search, flags = _crop(t_all, y_all, span)
    metrics["n_samples"] = float(t.size)
    notes["reference_inside_search"] = bool(flags)
    checks["present"] = t.size >= int(limits["min_samples"])
    if not checks["present"]:
        return verdict("present")

    stored = path_value(ods, f"{source.line_base}.label")
    checks["label"] = stored is not None and str(stored) == source.label
    wavelength = path_value(ods, f"{source.line_base}.wavelength_central")
    if checks["label"] and wavelength is not None:
        declared = next(
            (w for _, c, l, _, w in SIGNALS if c == source.channel and l == source.line), None
        )
        if declared is not None:
            checks["label"] = bool(abs(float(wavelength) - float(declared)) <= WAVELENGTH_TOLERANCE_M)
    notes["stored_label"] = None if stored is None else str(stored)
    if not checks["label"]:
        return verdict("label")

    metrics["finite_fraction"] = _finite_fraction(y)
    checks["finite"] = metrics["finite_fraction"] >= float(limits["min_finite_fraction"])
    if not checks["finite"]:
        return verdict("finite")
    finite = np.isfinite(y)

    rail = float(limits["rail_fraction"]) * float(limits["rail_level"])
    metrics["railed_fraction"] = float(np.mean(np.abs(y[finite]) >= rail)) if finite.any() else 0.0
    checks["not_railed"] = metrics["railed_fraction"] < float(limits["max_railed_fraction"])
    if not checks["not_railed"]:
        return verdict("not_railed")

    reference = reference_mask if reference_mask is not None else (np.arange(t.size) < max(2, t.size // 5))
    ref_values = y[reference & finite]
    if ref_values.size >= 2:
        median, spread = robust_baseline(ref_values)
        mad = float(spread) / MAD_TO_SIGMA
        distinct = int(np.unique(ref_values).size)
    else:
        median, mad, distinct = float("nan"), 0.0, ref_values.size
    metrics["baseline_median"] = float(median)
    metrics["baseline_mad"] = float(mad)
    peak_excess = float(np.nanmax(y[search])) - float(median) if search.any() and np.isfinite(median) else float("nan")
    metrics["peak_excess"] = peak_excess
    metrics["peak_over_sigma"] = (
        peak_excess / (mad * MAD_TO_SIGMA) if mad > 0.0 and np.isfinite(peak_excess) else float("nan")
    )
    checks["baseline_live"] = distinct >= 2 and mad >= float(limits["min_baseline_mad"])
    if not checks["baseline_live"]:
        return verdict("baseline_live")

    scalar = read_validity(ods, base)
    notes["validity"] = scalar
    valid_fraction = 1.0
    if read_validity_timed(ods, base) is not None:
        usable = validity_mask(ods, base, times=t_all, min_validity=VALIDITY_SUSPECT)
        inside = (t_all >= span.reference_start) & (t_all < span.tend)
        valid_fraction = float(usable[inside].mean()) if inside.any() else 0.0
    metrics["valid_fraction"] = valid_fraction
    checks["validity"] = (scalar is None or scalar > VALIDITY_INVALID) and (
        valid_fraction >= float(limits["min_valid_fraction"])
    )
    if not checks["validity"]:
        return verdict("validity")

    grid_dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    native_dt = 1.0 / source.cadence_hz if source.cadence_hz > 0 else float("nan")
    resampled = bool(np.isfinite(grid_dt) and grid_dt > 1.5 * native_dt)
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
    if policy is None:
        policy = resolve_plasma_timing_policy()
    if span is None:
        span = analysis_span(policy)
    data = path_value(ods, f"{source.base}.data")
    time = resolve_signal_time(ods, source.base)
    if data is None or time is None:
        raise PlasmaTimingError(f"{source.base} carries no data")
    t, y, reference_mask, search, _ = _crop(time, data, span)
    return active_window(t, y, reference_mask=reference_mask, search_mask=search, **policy.h_alpha)


def _ip_arrays(ods: Any) -> tuple[np.ndarray, np.ndarray]:
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
    return t, y


def ip_window(
    ods: Any,
    *,
    span: AnalysisSpan | None = None,
    policy: PlasmaTimingPolicy | None = None,
) -> PulseWindow:
    """The principal plasma-current pulse inside the span, with the ``ip`` rule."""
    if policy is None:
        policy = resolve_plasma_timing_policy()
    if span is None:
        span = analysis_span(policy)
    t_all, y_all = _ip_arrays(ods)
    t, y, reference_mask, search, _ = _crop(t_all, y_all, span)
    if t.size < 2:
        raise PlasmaTimingError(f"{IP_BASE} has no samples inside {span.reference_start}-{span.tend} s")
    fs = 1.0 / float(np.median(np.diff(t)))
    rule = dict(policy.ip)
    if rule.get("cutoff_hz") is not None:
        rule.setdefault("fs", fs)
    return active_window(t, y, reference_mask=reference_mask, search_mask=search, **rule)


def _ip_checks(ods: Any, span: AnalysisSpan) -> dict[str, bool]:
    t_all, y_all = _ip_arrays(ods)
    inside = (t_all >= span.reference_start) & (t_all < span.tend)
    scalar = read_validity(ods, IP_BASE)
    return {
        "present": bool(inside.sum() >= 2),
        "finite": bool(np.isfinite(y_all[inside]).all()) if inside.any() else False,
        "validity": scalar is None or scalar > VALIDITY_INVALID,
    }


# ---------------------------------------------------------------------------
# The plasma timing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlasmaTiming:
    """The plasma window and how it was found.

    ``onset``/``offset`` are the window the consumer should use; both come
    from ``onset_source`` (the light when it is usable, the current
    otherwise -- never one edge from each).  ``optical`` and ``ip`` are the
    windows each source produced, ``candidates`` the usability verdict of
    every H-alpha line examined, and ``agreement`` the relation between the
    light and the current.  ``found`` is ``False`` when neither source shows
    a plasma; ``window`` is then ``None`` and no consumer may substitute the
    analysis range for it without saying so.
    """

    onset: float | None
    offset: float | None
    onset_source: str | None
    offset_source: str | None
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
    def found(self) -> bool:
        return self.onset is not None and self.offset is not None

    @property
    def window(self) -> tuple[float, float] | None:
        if not self.found:
            return None
        return (float(self.onset), float(self.offset))

    def record(self) -> dict[str, Any]:
        return {
            "onset": self.onset,
            "offset": self.offset,
            "onset_source": self.onset_source,
            "offset_source": self.offset_source,
            "agreement": self.agreement,
            "onset_delta_s": self.onset_delta_s,
            "offset_delta_s": self.offset_delta_s,
            "fallback_reason": self.fallback_reason,
            "flags": list(self.flags),
            "span": self.span.record(),
            "optical_source": None if self.optical_source is None else self.optical_source.record(),
            "optical": None if self.optical is None else self.optical.as_dict(),
            "ip": None if self.ip is None else self.ip.as_dict(),
            "ip_checks": dict(self.ip_checks),
            "candidates": [c.record() for c in self.candidates],
        }


def _describe(window: PulseWindow | None) -> str:
    if window is None:
        return "not evaluated"
    if window.found:
        return "found"
    flags = ", ".join(window.onset.flags) or "no window"
    return f"no light ({flags})" if flags else "no window"


def plasma_timing(
    ods: Any,
    *,
    policy: PlasmaTimingPolicy | None = None,
) -> PlasmaTiming:
    """Find the plasma window with the configured source hierarchy.

    H-alpha lines are examined in :func:`halpha_sources` order; the first
    usable one answers for the light, whether or not it saw any -- a usable
    slow channel that stayed dark is not overruled by a fast one, it is
    evidence.  The plasma-current window is always computed.  The onset and
    the offset then come from the light when it found a window and from the
    current otherwise, and the two are compared with the configured
    tolerances:

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

    ``fallback_reason`` explains why the light did not answer, check by check.
    Only a product without ``magnetics.ip`` raises.
    """
    if policy is None:
        policy = resolve_plasma_timing_policy()
    span = analysis_span(policy)
    tolerances = policy.agreement
    flags: list[str] = []

    candidates: list[HalphaUsability] = []
    optical: PulseWindow | None = None
    optical_source: HalphaSource | None = None
    for source in halpha_sources():
        usability = halpha_usability(ods, source, span=span, policy=policy)
        candidates.append(usability)
        if usability.usable and optical is None:
            optical = halpha_window(ods, source, span=span, policy=policy)
            optical_source = source
            if usability.notes.get("reference_inside_search"):
                flags.append("reference_inside_search")

    ip_checks = _ip_checks(ods, span)
    ip = ip_window(ods, span=span, policy=policy)

    light_found = optical is not None and optical.found
    ip_found = ip.found
    onset_delta = offset_delta = None
    if light_found and ip_found:
        onset_delta = float(ip.start - optical.start)
        offset_delta = float(ip.end - optical.end)
        if abs(offset_delta) > float(tolerances["offset_tolerance_s"]):
            flags.append("offset_disagreement")

    fallback_reason: str | None = None
    if light_found:
        chosen, source_name = optical, optical_source.role
        if optical_source.role != SOURCE_H_PRIMARY:
            flags.append("optical_fallback_fast" if optical_source.role == SOURCE_H_FAST else "optical_fallback_secondary")
            fallback_reason = "; ".join(
                f"{c.source.role}: {c.reason}" for c in candidates if not c.usable
            ) or None
        if ip_found:
            if onset_delta < -float(tolerances["onset_tolerance_s"]):
                agreement = AGREEMENT_IP_BEFORE_HALPHA
                flags.append("ip_before_halpha")
            elif onset_delta > float(tolerances["lag_tolerance_s"]):
                agreement = AGREEMENT_HALPHA_LEADS_IP_LARGE
                flags.append("halpha_leads_ip_large")
            else:
                agreement = AGREEMENT_CONSISTENT
        else:
            agreement = AGREEMENT_HALPHA_ONLY
            flags.append("ip_no_pulse")
    else:
        parts = []
        for c in candidates:
            if c.usable:
                parts.append(f"{c.source.role}: usable but {_describe(optical if c.source is optical_source else None)}")
            else:
                parts.append(f"{c.source.role}: {c.reason}")
        if not candidates:
            parts.append("no H-alpha line is configured")
        if ip_found:
            chosen, source_name = ip, SOURCE_IP
            agreement = AGREEMENT_IP_ONLY
            if optical is not None:
                flags.append("halpha_dark_with_ip_pulse")
            fallback_reason = "; ".join(parts)
        else:
            chosen, source_name = None, None
            agreement = AGREEMENT_NONE
            flags.append("no_plasma_timing")
            parts.append(f"{SOURCE_IP}: {', '.join(ip.onset.flags) or 'no window'}")
            fallback_reason = "; ".join(parts)

    if chosen is not None:
        flags.extend(f for f in chosen.flags if f not in flags)
        onset, offset = chosen.start, chosen.end
    else:
        onset = offset = None

    return PlasmaTiming(
        onset=None if onset is None else float(onset),
        offset=None if offset is None else float(offset),
        onset_source=source_name,
        offset_source=source_name,
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
