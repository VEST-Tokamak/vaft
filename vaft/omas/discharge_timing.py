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
analysis span and baseline as the plasma policy, the coil-onset rule
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

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vaft.machine_mapping.utils import (
    CroppedRecord,
    DischargeTimingPolicy,
    crop_to_span,
    resolve_discharge_timing_policy,
)
from vaft.ods_access import path_count, path_value
from vaft.process.onset import (
    OnsetRecord,
    active_window,
    robust_baseline,
    zero_crossing_after_excursion,
)
from vaft.validation.imas import resolve_signal_time

from .plasma_timing import AnalysisSpan, analysis_span
from .vacuum_magnetics import FLUX_LOOP, probe_family

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

    ``zero_crossing`` is the event (``None`` when not found); ``excursion_time``
    and ``excursion_value`` locate the excursion's extremum (baseline-relative,
    signed); ``approach_min``/``approach_time`` are set when the decay came
    within the configured fraction of zero and climbed back before crossing
    (``approached_without_crossing``).  ``voltage_source`` says whether a
    stored ``voltage.data`` or ``-d(flux)/dt`` was read.
    """

    loop_index: int
    loop_name: str
    position: tuple[float, float] | None
    voltage_source: str | None
    anchor_time: float | None
    excursion_time: float | None
    excursion_value: float | None
    zero_crossing: float | None
    approach_min: float | None
    approach_time: float | None
    event: OnsetRecord | None
    flags: tuple[str, ...]

    @property
    def found(self) -> bool:
        return self.zero_crossing is not None

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
    event's own.
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
    policy: DischargeTimingPolicy | None, span: AnalysisSpan | None = None
) -> tuple[DischargeTimingPolicy, AnalysisSpan]:
    if policy is None:
        policy = resolve_discharge_timing_policy()
    if span is None:
        span = analysis_span(policy)
    return policy, span


def _crop(time: Any, values: Any, span: AnalysisSpan) -> CroppedRecord:
    try:
        return crop_to_span(
            time, values, baseline_start=span.baseline_start, tstart=span.tstart, tend=span.tend
        )
    except ValueError as exc:
        raise DischargeTimingError(str(exc)) from exc


def _waveform(ods: Any, base: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(time, data)`` of the signal node at ``base``, or ``None`` when absent or inconsistent."""
    data = path_value(ods, f"{base}.data")
    time = resolve_signal_time(ods, base)
    if data is None or time is None:
        return None
    y = np.asarray(data, dtype=float).reshape(-1)
    t = np.asarray(time, dtype=float).reshape(-1)
    if y.size < 2 or y.size != t.size:
        return None
    return t, y


def _coil_name(ods: Any, index: int) -> str:
    for key in ("name", "identifier"):
        value = path_value(ods, f"{COIL_BASE}.{index}.{key}")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"coil {index}"


def _loop_name(ods: Any, index: int) -> str:
    for key in ("name", "identifier"):
        value = path_value(ods, f"{FLUX_LOOP_BASE}.{index}.{key}")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"flux loop {index}"


def _loop_position(ods: Any, index: int) -> tuple[float, float] | None:
    r = path_value(ods, f"{FLUX_LOOP_BASE}.{index}.position.0.r")
    z = path_value(ods, f"{FLUX_LOOP_BASE}.{index}.position.0.z")
    if r is None or z is None:
        return None
    try:
        r, z = float(np.asarray(r).reshape(-1)[0]), float(np.asarray(z).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(r) and np.isfinite(z)):
        return None
    return r, z


def _absent(method: str, reason: str) -> OnsetRecord:
    return OnsetRecord(time=None, index=None, method=method, evidence={"reason": reason},
                       flags=("no_onset", "absent"))


def coil_onsets(
    ods: Any,
    *,
    span: AnalysisSpan | None = None,
    policy: DischargeTimingPolicy | None = None,
) -> tuple[CoilOnset, ...]:
    """The onset of every ``pf_active`` coil current, in coil order.

    Each current is cropped to the span, its baseline taken over the lead
    stretch, and the coil rule run on ``|I - baseline|``.  A coil without a
    waveform is recorded as ``absent``; an idle coil comes back with the
    detector's ``reference_flat`` and no time.
    """
    policy, span = _resolved(policy, span)
    out: list[CoilOnset] = []
    for index in range(path_count(ods, COIL_BASE)):
        name = _coil_name(ods, index)
        waveform = _waveform(ods, f"{COIL_BASE}.{index}.current")
        if waveform is None:
            out.append(CoilOnset(index, name, _absent("coil_onset", "no current waveform"), None))
            continue
        cropped = _crop(*waveform, span)
        if cropped.t.size < 2:
            out.append(CoilOnset(index, name, _absent("coil_onset", "no samples inside the span"), None))
            continue
        baseline, _ = robust_baseline(cropped.y, cropped.baseline_mask)
        if not np.isfinite(baseline):
            baseline = 0.0
        magnitude = np.abs(cropped.y - baseline)
        window = active_window(
            cropped.t, magnitude, reference_mask=cropped.baseline_mask, search_mask=cropped.search,
            **policy.coil,
        )
        onset = window.onset
        if cropped.flags:
            onset = OnsetRecord(
                time=onset.time, index=onset.index, method=onset.method, evidence=onset.evidence,
                flags=tuple(dict.fromkeys((*onset.flags, *cropped.flags))),
                rejected=onset.rejected, accepted=onset.accepted,
            )
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
    """The :class:`CoilOnset` of the policy's ``ohmic_coil``, or ``None`` when no coil has that name."""
    policy, span = _resolved(policy, span)
    if onsets is None:
        onsets = coil_onsets(ods, span=span, policy=policy)
    wanted = policy.ohmic_coil.casefold()
    for coil in onsets:
        if coil.name.casefold() == wanted:
            return coil
    return None


def inboard_midplane_loop(ods: Any) -> int | None:
    """The flux loop that measures the inboard-midplane loop voltage.

    Among the loops whose position puts them in the ``inboard_flux_loop``
    family (:func:`vaft.omas.vacuum_magnetics.probe_family`) and that carry
    a finite flux or voltage record, the one nearest the midplane (smallest
    ``|z|``, lowest index on a tie).  ``None`` when there is no such loop.
    """
    best: tuple[float, int] | None = None
    for index in range(path_count(ods, FLUX_LOOP_BASE)):
        position = _loop_position(ods, index)
        if position is None or probe_family(FLUX_LOOP, *position) != "inboard_flux_loop":
            continue
        has_record = False
        for node in ("voltage", "flux"):
            waveform = _waveform(ods, f"{FLUX_LOOP_BASE}.{index}.{node}")
            if waveform is not None and np.isfinite(waveform[1]).any():
                has_record = True
                break
        if not has_record:
            continue
        key = (abs(position[1]), index)
        if best is None or key < best:
            best = key
    return None if best is None else best[1]


def loop_voltage(
    ods: Any, index: int, *, prefer_measured: bool = True
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """``(time, voltage, source)`` of flux loop ``index``.

    A stored ``voltage.data`` is read when present and preferred; otherwise
    the voltage is ``-d(flux)/dt`` on the flux record's own grid
    (``source = "dflux_dt"``).  ``None`` when the loop carries neither.
    """
    base = f"{FLUX_LOOP_BASE}.{index}"
    measured = _waveform(ods, f"{base}.voltage") if prefer_measured else None
    if measured is not None:
        return measured[0], measured[1], VOLTAGE_MEASURED
    flux = _waveform(ods, f"{base}.flux")
    if flux is not None:
        t, phi = flux
        return t, -np.gradient(phi, t), VOLTAGE_DERIVED
    if not prefer_measured:
        measured = _waveform(ods, f"{base}.voltage")
        if measured is not None:
            return measured[0], measured[1], VOLTAGE_MEASURED
    return None


def _no_event(index: int, name: str, position, source, anchor, flags) -> LoopVoltageEvent:
    return LoopVoltageEvent(
        loop_index=index, loop_name=name, position=position, voltage_source=source,
        anchor_time=anchor, excursion_time=None, excursion_value=None, zero_crossing=None,
        approach_min=None, approach_time=None, event=None, flags=tuple(dict.fromkeys(flags)),
    )


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
    ``vloop`` rule, searched from the anchor (less its tolerance) onward.
    """
    policy, span = _resolved(policy, span)
    index = inboard_midplane_loop(ods)
    if index is None:
        return _no_event(-1, "", None, None, None, ["loop_not_found"])
    name, position = _loop_name(ods, index), _loop_position(ods, index)
    read = loop_voltage(ods, index, prefer_measured=bool(policy.loop_voltage["prefer_measured_voltage"]))
    if read is None:
        return _no_event(index, name, position, None, None, ["loop_not_found"])
    t, v, source = read
    flags: list[str] = ["voltage_derived"] if source == VOLTAGE_DERIVED else []
    if anchor is None or not anchor.found:
        return _no_event(index, name, position, source, None, [*flags, "no_oh_anchor"])
    cropped = _crop(t, v, span)
    flags.extend(cropped.flags)
    if cropped.t.size < 2:
        return _no_event(index, name, position, source, anchor.time, [*flags, "loop_not_found"])
    tolerance = float(policy.vloop.get("anchor_tolerance_s", 0.0))
    search = cropped.search & (cropped.t >= float(anchor.time) - tolerance)
    event = zero_crossing_after_excursion(
        cropped.t, cropped.y, anchor_time=float(anchor.time),
        reference_mask=cropped.baseline_mask, search_mask=search, **policy.vloop,
    )
    evidence = event.evidence
    event_flags = ["no_oh_excursion" if f == "no_excursion_at_anchor" else f for f in event.flags
                   if f != "no_onset"]
    return LoopVoltageEvent(
        loop_index=index, loop_name=name, position=position, voltage_source=source,
        anchor_time=float(anchor.time),
        excursion_time=evidence.get("extremum_time"),
        excursion_value=evidence.get("extremum"),
        zero_crossing=event.time,
        approach_min=evidence.get("approach_min"),
        approach_time=evidence.get("approach_time"),
        event=event,
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
    policy, span = _resolved(policy)
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
