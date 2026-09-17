"""Representative peaks of the plasma phase, from an ODS, with their provenance (issue #409).

:mod:`vaft.omas.plasma_timing` says *when* there is a plasma;
:mod:`vaft.omas.discharge_timing` says when the machine acted.  This module
answers the third, separate question -- *how large did the plasma get* --
with one number and one time per signal: the plasma current, the H-alpha
line that answered for the plasma, each configured impurity line, and the
diamagnetic flux.  Every peak is the *representative* one
(:func:`vaft.process.onset.robust_peak`: the largest sustained excursion,
never a coil-firing impulse or an isolated optical spike), measured
strictly inside the window the timing found, on the product's own clock.

When the timing found no plasma nothing is measured.  The record then says
``computed = False`` with the timing's ``fallback_reason`` and every
feature is ``not_computed`` -- the analysis range is never reinterpreted
as a plasma window.  A signal a product does not carry is ``absent``; a line
that is not configured simply does not appear.  The rules come from the
``plasma_features`` block of ``vest.yaml``
(:func:`vaft.machine_mapping.utils.resolve_plasma_features_policy`), keyed
by signal: impurity lines never borrow the H-alpha values.  The diamagnetic
value is signed (VEST's stored flux is negative-going); ``ip_ramp_end`` is
reserved and never required.  Nothing here writes the ODS.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from vaft.machine_mapping.spectrometer_uv import CHANNEL_CADENCE_HZ, CHANNEL_RAIL_LEVEL
from vaft.machine_mapping.utils import (
    PLASMA_FEATURES_HALPHA_LABEL,
    PlasmaFeaturesPolicy,
    PlasmaTimingPolicy,
    resolve_plasma_features_policy,
    resolve_plasma_timing_policy,
)
from vaft.ods_access import path_count, path_value
from vaft.process.onset import PeakRecord, robust_peak
from vaft.validation.imas import resolve_signal_waveform

from .plasma_timing import (
    HALPHA_LABEL,
    IP_BASE,
    SOURCE_H_FAST,
    SOURCE_H_PRIMARY,
    AnalysisSpan,
    HalphaUsability,
    PlasmaTiming,
    _crop,
    channel_notes,
    plasma_timing,
)

assert PLASMA_FEATURES_HALPHA_LABEL == HALPHA_LABEL, "the policy and the timing disagree on the H-alpha label"

__all__ = [
    "ABSENT",
    "DIAMAGNETIC_BASE",
    "FEATURE_DIAMAGNETIC",
    "FEATURE_H_ALPHA",
    "FEATURE_IP",
    "NOT_COMPUTED",
    "SPECTROMETER_CHANNELS",
    "Feature",
    "PlasmaFeatures",
    "ip_peak",
    "line_by_label",
    "plasma_features",
]

DIAMAGNETIC_BASE = "magnetics.diamagnetic_flux.0"
SPECTROMETER_CHANNELS = "spectrometer_uv.channel"

FEATURE_IP = "ip"
FEATURE_H_ALPHA = "h_alpha"
FEATURE_DIAMAGNETIC = "diamagnetic"

NOT_COMPUTED = "not_computed"
ABSENT = "absent"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    """One signal's representative peak inside the plasma window, and why.

    ``peak`` is the detector's record (``None`` when nothing was measured);
    ``notes`` what the reader knew about the signal (the H-alpha role, a
    resampled fast line and its time shift, a railed level, the diamagnetic
    mapper's method); ``reason`` why there is no peak.
    """

    name: str
    base: str | None
    label: str | None
    peak: PeakRecord | None
    flags: tuple[str, ...]
    notes: Mapping[str, Any]
    reason: str | None

    @property
    def value(self) -> float | None:
        return None if self.peak is None else self.peak.value

    @property
    def time(self) -> float | None:
        return None if self.peak is None else self.peak.time

    @property
    def found(self) -> bool:
        return self.peak is not None and self.peak.found

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base,
            "label": self.label,
            "value": self.value,
            "time": self.time,
            "flags": list(self.flags),
            "reason": self.reason,
        }

    def record(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "notes": dict(self.notes),
            "peak": None if self.peak is None else self.peak.as_dict(),
        }


@dataclass(frozen=True)
class PlasmaFeatures:
    """The representative peaks of one shot, on the timing that bounds them.

    ``computed`` is ``False`` when the timing found no plasma; ``reason`` is
    then the timing's ``fallback_reason`` and every feature is
    ``not_computed``.  ``lines`` is keyed by the configured label and holds
    every configured line, ``absent`` when the product lacks it.
    """

    timing: PlasmaTiming
    computed: bool
    reason: str | None
    ip: Feature
    h_alpha: Feature
    lines: Mapping[str, Feature]
    diamagnetic: Feature
    span: AnalysisSpan
    flags: tuple[str, ...]

    @property
    def window(self) -> tuple[float, float] | None:
        return self.timing.window

    def features(self) -> dict[str, Feature]:
        return {
            FEATURE_IP: self.ip,
            FEATURE_H_ALPHA: self.h_alpha,
            **{f"line:{label}": feature for label, feature in self.lines.items()},
            FEATURE_DIAMAGNETIC: self.diamagnetic,
        }

    def summary(self) -> dict[str, Any]:
        """What a metrics record or a manifest carries: every peak and its provenance."""
        return {
            "computed": self.computed,
            "reason": self.reason,
            "window": None if self.window is None else list(self.window),
            "ip": self.ip.summary(),
            "h_alpha": self.h_alpha.summary(),
            "lines": {label: feature.summary() for label, feature in self.lines.items()},
            "diamagnetic": self.diamagnetic.summary(),
            "flags": list(self.flags),
            "timing": self.timing.summary(),
        }

    def record(self) -> dict[str, Any]:
        """Everything, including every detector record and the full timing."""
        return {
            "computed": self.computed,
            "reason": self.reason,
            "window": None if self.window is None else list(self.window),
            "span": self.span.record(),
            "ip": self.ip.record(),
            "h_alpha": self.h_alpha.record(),
            "lines": {label: feature.record() for label, feature in self.lines.items()},
            "diamagnetic": self.diamagnetic.record(),
            "flags": list(self.flags),
            "timing": self.timing.record(),
        }


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _not_measured(name: str, flag: str, reason: str, *, base: str | None = None,
                  label: str | None = None, notes: Mapping[str, Any] | None = None) -> Feature:
    return Feature(name=name, base=base, label=label, peak=None, flags=(flag,),
                   notes=dict(notes or {}), reason=reason)


def _peak_in_window(
    ods: Any,
    base: str,
    timing: PlasmaTiming,
    rule: Mapping[str, Any],
    *,
    name: str,
    label: str | None = None,
    notes: Mapping[str, Any] | None = None,
) -> Feature:
    """The representative peak of the signal at ``base`` inside the timing's window."""
    notes = dict(notes or {})
    waveform = resolve_signal_waveform(ods, base)
    if waveform is None:
        return _not_measured(name, ABSENT, f"{base} carries no waveform", base=base, label=label, notes=notes)
    cropped = _crop(*waveform, timing.span)
    inside = (cropped.t >= float(timing.onset)) & (cropped.t <= float(timing.offset))
    if not inside.any():
        return _not_measured(
            name, "no_samples_in_window",
            f"{base} has no samples inside {timing.onset:.4f}-{timing.offset:.4f} s",
            base=base, label=label, notes=notes,
        )
    peak = robust_peak(
        cropped.t, cropped.y, reference_mask=cropped.baseline_mask, search_mask=inside, **rule
    )
    notes.setdefault("stored_dt", float(np.median(np.diff(cropped.t))) if cropped.t.size > 1 else float("nan"))
    flags = tuple(dict.fromkeys((*cropped.flags, *peak.flags)))
    reason = None if peak.found else ", ".join(f for f in peak.flags if f != "no_peak") or "no peak"
    return Feature(name=name, base=base, label=label, peak=peak, flags=flags, notes=notes, reason=reason)


def _flagged(feature: Feature, *flags: str, **notes: Any) -> Feature:
    """``feature`` with ``flags`` appended (deduplicated) and ``notes`` merged."""
    return replace(feature, flags=tuple(dict.fromkeys((*feature.flags, *flags))),
                   notes={**feature.notes, **notes})


def _railed(feature: Feature, channel: int | None, usability: Mapping[str, float]) -> Feature:
    """``feature`` with the ``railed`` note (and flag) when its channel's digitizer clipped inside the window.

    A rail is a property of the digitizer, not of the line:
    :data:`vaft.machine_mapping.spectrometer_uv.CHANNEL_RAIL_LEVEL` names the
    channels whose rail is known, and a line on a channel without one is not
    judged (``railed`` is ``None``).  Judged on the raw maximum the detector
    saw: a rail held for less than the peak's minimum width is refused as a
    spike and the reported peak is below it, yet the line did clip.
    """
    level = None if channel is None else CHANNEL_RAIL_LEVEL.get(int(channel))
    raw_max = None if feature.peak is None else feature.peak.evidence.get("raw_max")
    if level is None or raw_max is None:
        return _flagged(feature, railed=None)
    railed = bool(abs(raw_max) >= float(usability.get("rail_fraction", 1.0)) * float(level))
    return _flagged(feature, *(("railed",) if railed else ()), railed=railed)


def _line_notes(ods: Any, channel: int, feature: Feature) -> dict[str, Any]:
    """The mapper's handling of the line's channel (resampling, time shift), as the timing records it."""
    cadence = CHANNEL_CADENCE_HZ.get(int(channel))
    if cadence is None or feature.peak is None:
        return {}
    return channel_notes(ods, cadence, float(feature.notes.get("stored_dt", float("nan"))))


def line_by_label(ods: Any, label: str) -> tuple[str, int, int] | None:
    """``(base, channel, line)`` of the first ``processed_line`` whose stored label is ``label``.

    The stored ``label`` is authoritative; the mapper's positional table is
    not consulted.  ``None`` when no line carries it.
    """
    for channel in range(path_count(ods, SPECTROMETER_CHANNELS)):
        lines = f"{SPECTROMETER_CHANNELS}.{channel}.processed_line"
        for line in range(path_count(ods, lines)):
            stored = path_value(ods, f"{lines}.{line}.label")
            if isinstance(stored, str) and stored.strip() == label:
                return f"{lines}.{line}.intensity", channel, line
    return None


def _answering_halpha(timing: PlasmaTiming) -> HalphaUsability | None:
    """The H-alpha line that answers for the plasma: the timing's optical source, else its first usable candidate."""
    if timing.optical_source is not None:
        for candidate in timing.candidates:
            if candidate.source is timing.optical_source:
                return candidate
    for candidate in timing.candidates:
        if candidate.usable:
            return candidate
    return None


def _halpha_feature(ods: Any, timing: PlasmaTiming, rule: Mapping[str, Any],
                    usability: Mapping[str, float]) -> Feature:
    candidate = _answering_halpha(timing)
    if candidate is None:
        reasons = "; ".join(f"{c.source.role}: {c.reason}" for c in timing.candidates) or "no H-alpha line configured"
        return _not_measured(FEATURE_H_ALPHA, "no_usable_h_alpha", reasons, label=HALPHA_LABEL)
    source = candidate.source
    notes = {
        "role": source.role,
        "channel": source.channel,
        "line": source.line,
        **{key: candidate.notes[key] for key in ("resampled", "native_rate_hz", "time_shift_s", "stored_dt")
           if key in candidate.notes},
    }
    feature = _peak_in_window(ods, source.base, timing, rule, name=FEATURE_H_ALPHA,
                              label=source.label, notes=notes)
    if source.role != SOURCE_H_PRIMARY:
        feature = _flagged(feature, "optical_fallback_fast" if source.role == SOURCE_H_FAST
                           else "optical_fallback_secondary")
    return _railed(feature, source.channel, usability)


def _line_feature(ods: Any, label: str, timing: PlasmaTiming, rule: Mapping[str, Any],
                  usability: Mapping[str, float]) -> Feature:
    name = f"line:{label}"
    found = line_by_label(ods, label)
    if found is None:
        return _not_measured(name, ABSENT, f"no processed_line carries label {label!r}", label=label)
    base, channel, line = found
    feature = _peak_in_window(ods, base, timing, rule, name=name, label=label,
                              notes={"channel": channel, "line": line})
    feature = _flagged(feature, **_line_notes(ods, channel, feature))
    return _railed(feature, channel, usability)


def _diamagnetic_feature(ods: Any, timing: PlasmaTiming, rule: Mapping[str, Any]) -> Feature:
    method = path_value(ods, f"{DIAMAGNETIC_BASE}.method_name")
    method = str(method) if isinstance(method, str) and method.strip() else None
    notes = {"method_name": method, "saturated": bool(method and "acquisition limit" in method)}
    feature = _peak_in_window(ods, DIAMAGNETIC_BASE, timing, rule, name=FEATURE_DIAMAGNETIC, notes=notes)
    if feature.peak is not None and "record_flat" in feature.peak.flags:
        feature = _flagged(feature, "diamagnetic_flat")
    return feature


def _ip_feature(ods: Any, timing: PlasmaTiming, rule: Mapping[str, Any]) -> Feature:
    failed = [name for name, ok in timing.ip_checks.items() if not ok]
    if failed:
        return _not_measured(FEATURE_IP, "ip_unusable", ", ".join(failed), base=IP_BASE)
    feature = _peak_in_window(ods, IP_BASE, timing, rule, name=FEATURE_IP,
                              notes={"timing_source": timing.source})
    if timing.ip is None or not timing.ip.found:
        feature = _flagged(feature, "ip_no_pulse")
    return feature


def ip_peak(
    ods: Any,
    *,
    timing: PlasmaTiming | None = None,
    policy: PlasmaFeaturesPolicy | None = None,
) -> Feature:
    """The plasma-current feature alone, for a caller that needs one number.

    The same measurement :func:`plasma_features` makes for ``ip``, without
    the optical and diamagnetic peaks; ``not_computed`` when the timing found
    no plasma.
    """
    if timing is None:
        timing = plasma_timing(ods)
    if policy is None:
        policy = resolve_plasma_features_policy()
    if not timing.found:
        return _not_measured(FEATURE_IP, NOT_COMPUTED, timing.fallback_reason or "no plasma timing", base=IP_BASE)
    return _ip_feature(ods, timing, policy.ip)


# ---------------------------------------------------------------------------
# The plasma features
# ---------------------------------------------------------------------------


def plasma_features(
    ods: Any,
    *,
    timing: PlasmaTiming | None = None,
    policy: PlasmaFeaturesPolicy | None = None,
    timing_policy: PlasmaTimingPolicy | None = None,
) -> PlasmaFeatures:
    """Every configured representative peak inside the plasma window.

    ``timing`` may be handed in when the caller already has it; otherwise
    :func:`vaft.omas.plasma_timing.plasma_timing` runs first, and a product
    without a plasma current raises its :class:`PlasmaTimingError` exactly as
    the timing does.  No window means no measurement (``computed = False``).
    """
    if timing_policy is None:
        timing_policy = resolve_plasma_timing_policy()
    if timing is None:
        timing = plasma_timing(ods, policy=timing_policy)
    if policy is None:
        policy = resolve_plasma_features_policy()
    usability = timing_policy.usability

    if not timing.found:
        reason = timing.fallback_reason or "no plasma timing"
        return PlasmaFeatures(
            timing=timing, computed=False, reason=reason,
            ip=_not_measured(FEATURE_IP, NOT_COMPUTED, reason, base=IP_BASE),
            h_alpha=_not_measured(FEATURE_H_ALPHA, NOT_COMPUTED, reason, label=HALPHA_LABEL),
            lines={label: _not_measured(f"line:{label}", NOT_COMPUTED, reason, label=label)
                   for label in policy.lines},
            diamagnetic=_not_measured(FEATURE_DIAMAGNETIC, NOT_COMPUTED, reason, base=DIAMAGNETIC_BASE),
            span=timing.span, flags=("no_plasma_timing",),
        )

    ip = _ip_feature(ods, timing, policy.ip)
    h_alpha = _halpha_feature(ods, timing, policy.h_alpha, usability)
    lines = {label: _line_feature(ods, label, timing, rule, usability) for label, rule in policy.lines.items()}
    diamagnetic = _diamagnetic_feature(ods, timing, policy.diamagnetic)

    # record-level outcomes: an outcome that already names its feature
    # (ip_unusable, no_usable_h_alpha) is repeated as is, the generic ones
    # (absent, no_samples_in_window) are prefixed with the feature
    flags: list[str] = []
    for feature in (ip, h_alpha, diamagnetic):
        for outcome in ("ip_unusable", "no_usable_h_alpha", ABSENT, "no_samples_in_window"):
            if outcome in feature.flags:
                flags.append(outcome if feature.name in outcome else f"{feature.name}_{outcome}")
    for label, feature in lines.items():
        if ABSENT in feature.flags:
            flags.append(f"line_absent:{label}")
    return PlasmaFeatures(
        timing=timing, computed=True, reason=None, ip=ip, h_alpha=h_alpha, lines=lines,
        diamagnetic=diamagnetic, span=timing.span, flags=tuple(dict.fromkeys(flags)),
    )
