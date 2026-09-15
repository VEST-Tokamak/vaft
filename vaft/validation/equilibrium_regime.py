"""Which operating regime each reconstructed slice belongs to (issue #76).

Reconstruction outcomes are only interpretable against the state the plasma
was in.  Claims of the shape "high-current shots do not converge" or "ramp-up
equilibria have unreliable pressure" are claims about cohorts, and a cohort
that is not defined by a stated rule cannot be checked, only asserted.

This applies the rule; :mod:`vaft.machine_mapping.equilibrium_regime` states
it.  The split is the usual one: the thresholds are a choice about VEST, the
arithmetic is not about any machine.

Two things the labels deliberately do **not** do:

* ``unknown`` is never a negative answer.  A slice whose current cannot be
  read is ``unknown``, not ``vacuum``; a boundary EFIT did not name is
  ``unknown``, not ``limited``.  #76 asks for exactly this separation, and it
  is the difference between "we looked and there was no X-point" and "we never
  looked".
* Nothing is discarded.  Every label carries the numbers it was derived from
  -- the current, the rate, the fraction of peak, the raw ``limloc`` -- so a
  cohort can be re-cut under a different threshold without re-running EFIT.

Where the topology can and cannot be read
-----------------------------------------

``limloc`` is the only place EFIT says where the boundary was defined, and it
arrives in the a-file parser cache under ``code.parameters``.  That cache is a
nested sub-path of a ``STR_0D`` and so **stops at the local product**: a
reconstruction that has round-tripped through the Access Layer no longer
carries it (#380, #642).

The consequence is deliberate and visible rather than silent.  Such a slice
classifies as topology ``unknown`` with reason ``limloc_absent``, never as
``limited``, so a cohort built from a replicated product reports that it does
not know the topology instead of reporting the wrong one.  A phase label is
unaffected -- it is derived from the plasma current, which survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vaft.machine_mapping.equilibrium_regime import (
    EquilibriumRegimePolicy,
    vest_equilibrium_regime_policy,
)
from vaft.ods_access import path_value

__all__ = [
    "PHASES",
    "RegimeLabels",
    "classify_equilibrium_regime",
    "classify_equilibrium_regimes",
]

#: Every phase a slice can be given.  ``unknown`` is a real answer.
PHASES = ("vacuum", "ramp_up", "flat", "ramp_down", "unknown")

_IP = "equilibrium.time_slice.{index}.constraints.ip.measured"
_LIMLOC = "equilibrium.code.parameters.time_slice.{index}.aeqdsk.limloc"


@dataclass(frozen=True)
class RegimeLabels:
    """One slice's cohort membership, and the evidence that put it there."""

    time: float
    phase: str
    topology: str
    #: Signed measured plasma current [A], or ``None`` when unreadable.
    current: float | None
    #: ``dIp/dt`` [A/s] on the reconstruction abscissa, or ``None``.
    current_rate: float | None
    #: ``|Ip|`` as a fraction of this discharge's peak, or ``None``.
    fraction_of_peak: float | None
    #: EFIT's own ``limloc`` string, kept verbatim beside its mapped name.
    limloc: str | None
    rule: str
    #: Why a label is ``unknown``, when it is.
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe record; the inputs travel with the label."""
        return {
            "time": float(self.time),
            "phase": self.phase,
            "topology": self.topology,
            "current": None if self.current is None else float(self.current),
            "current_rate": None if self.current_rate is None else float(self.current_rate),
            "fraction_of_peak": (
                None if self.fraction_of_peak is None else float(self.fraction_of_peak)
            ),
            "limloc": self.limloc,
            "rule": self.rule,
            "reasons": list(self.reasons),
        }


def _slice_count(ods: Any) -> int:
    try:
        return len(ods["equilibrium.time_slice"])
    except Exception:
        return 0


def _times(ods: Any, count: int) -> np.ndarray:
    times = path_value(ods, "equilibrium.time", None)
    if times is not None:
        values = np.asarray(times, dtype=float).reshape(-1)
        if values.size == count:
            return values
    # Fall back to the per-slice time, which a hand-built product may carry
    # instead of the top-level array.
    out = []
    for index in range(count):
        value = path_value(ods, f"equilibrium.time_slice.{index}.time", None)
        out.append(float("nan") if value is None else float(value))
    return np.asarray(out, dtype=float)


def _currents(ods: Any, count: int) -> np.ndarray:
    out = []
    for index in range(count):
        value = path_value(ods, _IP.format(index=index), None)
        try:
            out.append(float(np.asarray(value).reshape(-1)[0]))
        except Exception:
            out.append(float("nan"))
    return np.asarray(out, dtype=float)


def _rates(times: np.ndarray, currents: np.ndarray) -> np.ndarray:
    usable = np.isfinite(times) & np.isfinite(currents)
    rates = np.full(currents.shape, np.nan)
    if usable.sum() >= 2:
        rates[usable] = np.gradient(currents[usable], times[usable])
    return rates


def classify_equilibrium_regimes(
    ods: Any, *, policy: EquilibriumRegimePolicy | None = None
) -> tuple[RegimeLabels, ...]:
    """Label every slice of one reconstruction.

    The whole discharge at once, because the level rule needs this
    discharge's peak current -- see the policy module for why that
    non-causality was accepted.
    """
    resolved = policy or vest_equilibrium_regime_policy()
    count = _slice_count(ods)
    if count == 0:
        return ()
    times = _times(ods, count)
    currents = _currents(ods, count)
    rates = _rates(times, currents)

    plasma = np.isfinite(currents) & (np.abs(currents) >= resolved.vacuum_current_amperes)
    peak = float(np.max(np.abs(currents[plasma]))) if plasma.any() else 0.0
    peak_index = int(np.argmax(np.where(plasma, np.abs(currents), -np.inf))) if plasma.any() else -1
    peak_rate = float(np.nanmax(np.abs(rates))) if np.isfinite(rates).any() else 0.0

    labels = []
    for index in range(count):
        labels.append(
            _label_one(
                index,
                times[index],
                currents[index],
                rates[index],
                peak=peak,
                peak_index=peak_index,
                peak_rate=peak_rate,
                limloc=path_value(ods, _LIMLOC.format(index=index), None),
                policy=resolved,
            )
        )
    return tuple(labels)


def classify_equilibrium_regime(
    ods: Any, *, time_index: int, policy: EquilibriumRegimePolicy | None = None
) -> RegimeLabels:
    """Label one slice, by classifying the discharge and taking that slice.

    The level rule is a statement about where a slice sits in *its own*
    discharge, so a single slice cannot be labelled on its own. Classifying
    the whole reconstruction and selecting is the honest spelling of that.
    """
    labels = classify_equilibrium_regimes(ods, policy=policy)
    if not labels:
        raise ValueError("the product carries no equilibrium time slices to classify")
    if not 0 <= int(time_index) < len(labels):
        raise IndexError(
            f"time_index {time_index} is outside the {len(labels)} slices present"
        )
    return labels[int(time_index)]


def _label_one(
    index: int,
    time: float,
    current: float,
    rate: float,
    *,
    peak: float,
    peak_index: int,
    peak_rate: float,
    limloc: Any,
    policy: EquilibriumRegimePolicy,
) -> RegimeLabels:
    reasons: list[str] = []
    raw_limloc = None if limloc is None else str(limloc).strip()
    topology = policy.topology(limloc)
    if topology == "unknown":
        reasons.append("limloc_absent" if not raw_limloc else "limloc_unrecognised")

    fraction = None
    if np.isfinite(current) and peak > 0.0:
        fraction = abs(float(current)) / peak

    if not np.isfinite(current):
        return RegimeLabels(
            time=float(time) if np.isfinite(time) else float("nan"),
            phase="unknown",
            topology=topology,
            current=None,
            current_rate=None,
            fraction_of_peak=None,
            limloc=raw_limloc or None,
            rule=policy.phase_rule,
            reasons=tuple([*reasons, "plasma_current_unreadable"]),
        )

    if abs(current) < policy.vacuum_current_amperes:
        phase = "vacuum"
    elif policy.phase_rule == "level":
        if fraction is not None and fraction >= policy.flat_fraction:
            phase = "flat"
        else:
            phase = "ramp_up" if index < peak_index else "ramp_down"
    else:  # "slope"
        if not np.isfinite(rate):
            phase = "unknown"
            reasons.append("current_rate_unreadable")
        elif abs(rate) <= policy.slope_fraction * peak_rate:
            phase = "flat"
        else:
            phase = "ramp_up" if rate > 0 else "ramp_down"

    return RegimeLabels(
        time=float(time) if np.isfinite(time) else float("nan"),
        phase=phase,
        topology=topology,
        current=float(current),
        current_rate=float(rate) if np.isfinite(rate) else None,
        fraction_of_peak=fraction,
        limloc=raw_limloc or None,
        rule=policy.phase_rule,
        reasons=tuple(reasons),
    )
