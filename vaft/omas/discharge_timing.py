"""Discharge-timing events of a shot, from an ODS, with their provenance (issue #409).

Where :mod:`vaft.omas.plasma_timing` answers *when is there a plasma*, this
module answers *when did the machine act*: the onset of every poloidal-field
coil current, the onset of the ohmic (solenoid) drive among them, and the
loop-voltage event that drive produces -- the first sign change of the
inboard-midplane loop voltage after the extremum of the solenoid-driven
excursion, which is the ``'vloop'`` time convention of
:func:`vaft.omas.general.change_time_convention`.

These are independent, *measured* events.  None of them is a fallback for a
plasma onset, and none is a trigger: a command timestamp is a different
quantity that would come from the shot log or the control system, and no
such source is mapped today.  Actuators that have no mapped signal (EC
power, gas injection) are reported as ``not_present`` with the reason, not
guessed from a response signal.

Every rule comes from the ``discharge_timing`` block of ``vest.yaml`` through
:func:`vaft.machine_mapping.utils.resolve_discharge_timing_policy`: the same
analysis span and baseline as the plasma policy (on the product's own clock,
see :func:`vaft.omas.plasma_timing.clock_offset`), the coil-onset rule
(:func:`vaft.process.onset.active_window` on ``|I - baseline|`` so an idle
coil is flat and a bipolar one is judged on its magnitude), the loop
selection, and the zero-crossing rule
(:func:`vaft.process.onset.zero_crossing_after_excursion`).  The excursion
is anchored on the ohmic onset: on VEST the loop voltage is loudest in the
plasma phase (41524: +12.9 V at 335.7 ms against the −6.5 V solenoid swing),
so the drive's excursion is judged where the drive starts, not where the
record peaks.  Nothing here writes the ODS.  Missing data is a ``None`` with
a flag -- the record is buildable on a vacuum shot; only a configuration
problem raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

import numpy as np

from vaft.machine_mapping.utils import (
    CroppedRecord,
    DischargeTimingPolicy,
    resolve_discharge_timing_policy,
)
from vaft.ods_access import path_count
from vaft.process.onset import (
    OnsetRecord,
    active_window,
    robust_baseline,
    zero_crossing_after_excursion,
)
from vaft.validation.imas import resolve_signal_waveform, signal_label

from .plasma_timing import AnalysisSpan, _crop, analysis_span, clock_offset
from .vacuum_magnetics import FLUX_LOOP, _position, probe_family

__all__ = [
    "COIL_BASE",
    "FLUX_LOOP_BASE",
    "NOT_PRESENT",
    "VOLTAGE_DERIVED",
    "VOLTAGE_MEASURED",
    "CoilOnset",
    "DischargeTiming",
    "DischargeTimingError",
    "LoopVoltageEvent",
    "coil_onsets",
    "discharge_timing",
    "inboard_midplane_loop",
    "loop_voltage",
    "loop_voltage_event",
    "oh_coil_onset",
]

COIL_BASE = "pf_active.coil"
FLUX_LOOP_BASE = "magnetics.flux_loop"

VOLTAGE_MEASURED = "voltage"
VOLTAGE_DERIVED = "dflux_dt"

#: The leading fraction of a record that stands in for the baseline when the
#: crop carries no lead stretch (the detectors' own ``reference_fraction``).
LEADING_REFERENCE_FRACTION = 0.2

#: Actuators the discharge timing would report but for which no signal is
#: mapped; the reason is the registry state, not a guess from a response.
NOT_PRESENT: Mapping[str, str] = {
    "ec": "no actuator signal mapped (registry ech: not_implemented)",
    "gas": (
        "no actuator signal mapped (registry gas_injection: not_implemented; "
        "barometry pressure is a response, not the actuator)"
    ),
}


class DischargeTimingError(ValueError):
    """A configuration problem that prevents any discharge timing at all."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoilOnset:
    """When one poloidal-field coil started to carry current, and on what evidence.

    ``onset`` is the detector's record on ``|I - baseline|``; ``polarity`` is
    the sign of the current at the pulse peak relative to the baseline, or
    ``None`` when the coil did not fire (an idle coil is ``reference_flat``).
    """

    index: int
    name: str
    onset: OnsetRecord
    polarity: str | None

    @property
    def time(self) -> float | None:
        return self.onset.time

    @property
    def found(self) -> bool:
        return self.onset.found

    @property
    def flags(self) -> tuple[str, ...]:
        return self.onset.flags

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "time": self.time,
            "polarity": self.polarity,
            "flags": list(self.flags),
            "onset": self.onset.as_dict(),
        }


@dataclass(frozen=True)
class LoopVoltageEvent:
    """The solenoid-driven loop-voltage excursion and its zero crossing.

    ``event`` is the detector's record (``None`` when no detection ran);
    ``zero_crossing`` is its time, ``excursion_time``/``excursion_value``
    locate the excursion's extremum (baseline-relative, signed), and
    ``approach_min``/``approach_time`` are set when the decay came within the
    configured fraction of zero and climbed back before crossing
    (``approached_without_crossing``) -- all read from the event's evidence.
    ``voltage_source`` says whether a stored ``voltage.data`` or
    ``-d(flux)/dt`` was read; ``loop_index`` is ``-1`` when no loop was found.
    """

    loop_index: int
    loop_name: str
    flags: tuple[str, ...]
    position: tuple[float, float] | None = None
    voltage_source: str | None = None
    anchor_time: float | None = None
    event: OnsetRecord | None = None

    @property
    def found(self) -> bool:
        return self.event is not None and self.event.found

    @property
    def zero_crossing(self) -> float | None:
        return None if self.event is None else self.event.time

    def _evidence(self, key: str) -> float | None:
        return None if self.event is None else self.event.evidence.get(key)

    @property
    def excursion_time(self) -> float | None:
        return self._evidence("extremum_time")

    @property
    def excursion_value(self) -> float | None:
        return self._evidence("extremum")

    @property
    def approach_min(self) -> float | None:
        return self._evidence("approach_min")

    @property
    def approach_time(self) -> float | None:
        return self._evidence("approach_time")

    @property
    def base(self) -> str | None:
        return None if self.loop_index < 0 else f"{FLUX_LOOP_BASE}.{self.loop_index}"

    def summary(self) -> dict[str, Any]:
        return {
            "loop_index": self.loop_index,
            "loop_name": self.loop_name,
            "position": None if self.position is None else list(self.position),
            "voltage_source": self.voltage_source,
            "anchor_time": self.anchor_time,
            "excursion_time": self.excursion_time,
            "excursion_value": self.excursion_value,
            "zero_crossing": self.zero_crossing,
            "approach_min": self.approach_min,
            "approach_time": self.approach_time,
            "flags": list(self.flags),
        }

    def record(self) -> dict[str, Any]:
        return {**self.summary(), "event": None if self.event is None else self.event.as_dict()}


@dataclass(frozen=True)
class DischargeTiming:
    """The actuator events of one shot.

    ``pf_onsets`` has one entry per ``pf_active`` coil in coil order; ``oh``
    is the entry named by the policy's ``ohmic_coil`` (``None`` when no coil
    carries that name); ``vloop`` the loop-voltage event anchored on it.
    ``ec`` and ``gas`` are ``None`` and ``not_present`` says why.  ``flags``
    collects the record-level outcomes (``oh_coil_not_found``,
    ``oh_not_fired``, ``vloop_not_found``, ``no_pf_active``) with the loop
    event's own.  Times are on the product's own clock (``span``).
    """

    oh: CoilOnset | None
    oh_coil: str
    vloop: LoopVoltageEvent
    pf_onsets: tuple[CoilOnset, ...]
    not_present: Mapping[str, str]
    flags: tuple[str, ...]
    span: AnalysisSpan
    ec: None = field(default=None, init=False)
    gas: None = field(default=None, init=False)

    @property
    def oh_onset(self) -> float | None:
        return None if self.oh is None else self.oh.time

    @property
    def vloop_time(self) -> float | None:
        return self.vloop.zero_crossing

    def summary(self) -> dict[str, Any]:
        """What a manifest or a metrics record carries: the events and their provenance."""
        return {
            "oh_coil": self.oh_coil,
            "oh_onset": self.oh_onset,
            "oh_polarity": None if self.oh is None else self.oh.polarity,
            "vloop": self.vloop.summary(),
            "pf_onsets": {coil.name: coil.time for coil in self.pf_onsets},
            "not_present": dict(self.not_present),
            "flags": list(self.flags),
        }

    def record(self) -> dict[str, Any]:
        """Everything, including every detector record."""
        return {
            **{key: value for key, value in self.summary().items() if key not in ("vloop", "pf_onsets")},
            "span": self.span.record(),
            "vloop": self.vloop.record(),
            "pf_onsets": [coil.record() for coil in self.pf_onsets],
        }


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _resolved(
    policy: DischargeTimingPolicy | None,
    span: AnalysisSpan | None = None,
    *,
    ods: Any = None,
) -> tuple[DischargeTimingPolicy, AnalysisSpan]:
    if policy is None:
        policy = resolve_discharge_timing_policy()
    if span is None:
        span = analysis_span(policy)
        if ods is not None:
            span = span.shifted(clock_offset(ods))
    return policy, span


def _cropped(time: Any, values: Any, span: AnalysisSpan) -> CroppedRecord:
    return _crop(time, values, span, error=DischargeTimingError)


def _reference_mask(cropped: CroppedRecord) -> np.ndarray:
    """The samples the baseline is measured on.

    The crop's lead stretch when the record carries one; otherwise the
    record's leading fraction -- the same stand-in the detectors use -- so a
    product built on the analysis window alone is not baselined on the
    median of a record that is mostly pulse.
    """
    if cropped.baseline_mask is not None:
        return cropped.baseline_mask
    mask = np.zeros(cropped.t.size, dtype=bool)
    mask[: max(2, int(round(LEADING_REFERENCE_FRACTION * cropped.t.size)))] = True
    return mask


def _absent(method: str, reason: str) -> OnsetRecord:
    return OnsetRecord(time=None, index=None, method=method, evidence={"reason": reason},
                       flags=("no_onset", "absent"))


def coil_onsets(
    ods: Any,
    *,
    span: AnalysisSpan | None = None,
    policy: DischargeTimingPolicy | None = None,
    names: Iterable[str] | None = None,
) -> tuple[CoilOnset, ...]:
    """The onset of every ``pf_active`` coil current, in coil order.

    Each current is cropped to the span, its baseline taken over the lead
    stretch (or the record's leading fraction when there is none), and the
    coil rule run on ``|I - baseline|``.  A coil without a waveform is
    recorded as ``absent``; an idle coil comes back with the detector's
    ``reference_flat`` and no time.  ``names`` restricts the work to the
    coils so named (case-insensitive), for a caller that needs one drive.
    """
    policy, span = _resolved(policy, span, ods=ods)
    wanted = None if names is None else {name.casefold() for name in names}
    out: list[CoilOnset] = []
    for index in range(path_count(ods, COIL_BASE)):
        name = signal_label(ods, f"{COIL_BASE}.{index}", f"coil {index}")
        if wanted is not None and name.casefold() not in wanted:
            continue
        waveform = resolve_signal_waveform(ods, f"{COIL_BASE}.{index}.current")
        if waveform is None:
            out.append(CoilOnset(index, name, _absent("coil_onset", "no current waveform"), None))
            continue
        cropped = _cropped(*waveform, span)
        if cropped.t.size < 2:
            out.append(CoilOnset(index, name, _absent("coil_onset", "no samples inside the span"), None))
            continue
        reference = _reference_mask(cropped)
        baseline, _ = robust_baseline(cropped.y, reference)
        if not np.isfinite(baseline):
            baseline = 0.0
        window = active_window(
            cropped.t, np.abs(cropped.y - baseline), reference_mask=reference,
            search_mask=cropped.search, **policy.coil,
        )
        onset = window.onset
        if cropped.flags:
            onset = replace(onset, flags=tuple(dict.fromkeys((*onset.flags, *cropped.flags))))
        polarity = None
        if onset.found and onset.accepted is not None:
            peak_index = int(np.argmin(np.abs(cropped.t - onset.accepted.peak_time)))
            polarity = "positive" if cropped.y[peak_index] - baseline >= 0.0 else "negative"
        out.append(CoilOnset(index, name, onset, polarity))
    return tuple(out)


def oh_coil_onset(
    ods: Any,
    *,
    span: AnalysisSpan | None = None,
    policy: DischargeTimingPolicy | None = None,
    onsets: tuple[CoilOnset, ...] | None = None,
) -> CoilOnset | None:
    """The :class:`CoilOnset` of the policy's ``ohmic_coil``, or ``None`` when no coil has that name.

    Reads only that coil unless ``onsets`` (a full :func:`coil_onsets`) is handed in.
    """
    policy, span = _resolved(policy, span, ods=ods)
    if onsets is None:
        onsets = coil_onsets(ods, span=span, policy=policy, names=(policy.ohmic_coil,))
    wanted = policy.ohmic_coil.casefold()
    for coil in onsets:
        if coil.name.casefold() == wanted:
            return coil
    return None


def _finite_waveform(ods: Any, base: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The waveform at ``base`` when it carries at least one finite sample."""
    waveform = resolve_signal_waveform(ods, base)
    if waveform is None or not np.isfinite(waveform[1]).any():
        return None
    return waveform


def inboard_midplane_loop(ods: Any) -> int | None:
    """The flux loop that measures the inboard-midplane loop voltage.

    Among the loops whose position puts them in the ``inboard_flux_loop``
    family (:func:`vaft.omas.vacuum_magnetics.probe_family`) and that carry
    a flux or voltage record with finite samples, the one nearest the
    midplane (smallest ``|z|``, lowest index on a tie).  ``None`` when there
    is no such loop.  The plot layer's inboard-midplane selection delegates
    here, so the loop a figure labels is the loop the timing read.
    """
    candidates: list[tuple[float, int]] = []
    for index in range(path_count(ods, FLUX_LOOP_BASE)):
        position = _position(ods, f"{FLUX_LOOP_BASE}.{index}.position.0")
        if position is None or not np.isfinite(position).all():
            continue
        if probe_family(FLUX_LOOP, *position) == "inboard_flux_loop":
            candidates.append((abs(position[1]), index))
    for _, index in sorted(candidates):
        if any(_finite_waveform(ods, f"{FLUX_LOOP_BASE}.{index}.{node}") is not None
               for node in ("voltage", "flux")):
            return index
    return None


def loop_voltage(
    ods: Any, index: int, *, prefer_measured: bool = True
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """``(time, voltage, source)`` of flux loop ``index``.

    A stored ``voltage.data`` and ``-d(flux)/dt`` on the flux record's own
    grid are tried in the order ``prefer_measured`` sets; a record with no
    finite sample (a placeholder) does not win.  ``None`` when the loop
    carries neither.
    """
    base = f"{FLUX_LOOP_BASE}.{index}"
    order = ("voltage", "flux") if prefer_measured else ("flux", "voltage")
    for node in order:
        waveform = _finite_waveform(ods, f"{base}.{node}")
        if waveform is None:
            continue
        t, y = waveform
        if node == "voltage":
            return t, y, VOLTAGE_MEASURED
        return t, -np.gradient(y, t), VOLTAGE_DERIVED
    return None


def loop_voltage_event(
    ods: Any,
    *,
    anchor: CoilOnset | None,
    span: AnalysisSpan | None = None,
    policy: DischargeTimingPolicy | None = None,
) -> LoopVoltageEvent:
    """The zero crossing of the inboard-midplane loop voltage after the excursion ``anchor`` drives.

    Always returns a record: ``loop_not_found`` when no inboard loop carries
    a waveform, ``no_oh_anchor`` when the anchoring coil onset is missing,
    otherwise the outcome of
    :func:`vaft.process.onset.zero_crossing_after_excursion` with the
    ``vloop`` rule on the span's search stretch -- the anchoring, including
    its tolerance, is the detector's.
    """
    policy, span = _resolved(policy, span, ods=ods)
    index = inboard_midplane_loop(ods)
    if index is None:
        return LoopVoltageEvent(loop_index=-1, loop_name="", flags=("loop_not_found",))
    name = signal_label(ods, f"{FLUX_LOOP_BASE}.{index}", f"flux loop {index}")
    position = _position(ods, f"{FLUX_LOOP_BASE}.{index}.position.0")
    read = loop_voltage(ods, index, prefer_measured=bool(policy.loop_voltage["prefer_measured_voltage"]))
    if read is None:  # pragma: no cover - the selector already required a record
        return LoopVoltageEvent(loop_index=index, loop_name=name, position=position, flags=("loop_not_found",))
    t, v, source = read
    flags: list[str] = ["voltage_derived"] if source == VOLTAGE_DERIVED else []
    if anchor is None or not anchor.found:
        return LoopVoltageEvent(loop_index=index, loop_name=name, position=position,
                                voltage_source=source, flags=(*flags, "no_oh_anchor"))
    cropped = _cropped(t, v, span)
    flags.extend(cropped.flags)
    if cropped.t.size < 2:
        return LoopVoltageEvent(loop_index=index, loop_name=name, position=position,
                                voltage_source=source, anchor_time=float(anchor.time),
                                flags=(*flags, "loop_not_found"))
    event = zero_crossing_after_excursion(
        cropped.t, cropped.y, anchor_time=float(anchor.time),
        reference_mask=cropped.baseline_mask, search_mask=cropped.search, **policy.vloop,
    )
    event_flags = ["no_oh_excursion" if f == "no_excursion_at_anchor" else f for f in event.flags
                   if f != "no_onset"]
    return LoopVoltageEvent(
        loop_index=index, loop_name=name, position=position, voltage_source=source,
        anchor_time=float(anchor.time), event=event,
        flags=tuple(dict.fromkeys((*flags, *event_flags))),
    )


# ---------------------------------------------------------------------------
# The discharge timing
# ---------------------------------------------------------------------------


def discharge_timing(
    ods: Any,
    *,
    policy: DischargeTimingPolicy | None = None,
) -> DischargeTiming:
    """Every actuator event the products carry, with provenance.

    Per-coil onsets, the ohmic onset among them, and the loop-voltage zero
    crossing anchored on it.  Raises :class:`DischargeTimingError` only for a
    record that cannot be cropped (mismatched lengths); every missing datum is
    a ``None`` with a flag.
    """
    policy, span = _resolved(policy, ods=ods)
    onsets = coil_onsets(ods, span=span, policy=policy)
    oh = oh_coil_onset(ods, span=span, policy=policy, onsets=onsets)
    vloop = loop_voltage_event(ods, anchor=oh, span=span, policy=policy)
    flags: list[str] = []
    if not onsets:
        flags.append("no_pf_active")
    if oh is None:
        flags.append("oh_coil_not_found")
    elif not oh.found:
        flags.append("oh_not_fired")
    if not vloop.found:
        flags.append("vloop_not_found")
    flags.extend(vloop.flags)
    return DischargeTiming(
        oh=oh,
        oh_coil=policy.ohmic_coil,
        vloop=vloop,
        pf_onsets=onsets,
        not_present=dict(NOT_PRESENT),
        flags=tuple(dict.fromkeys(flags)),
        span=span,
    )
