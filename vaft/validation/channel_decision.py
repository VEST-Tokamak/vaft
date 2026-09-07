"""The per-channel, per-slice decision an equilibrium code consumes (issue #296).

An EFIT constraint is a number, an uncertainty and a weight.  Whether a
magnetic channel may supply that number at a given reconstruction time is a
*decision* that belongs upstream -- to the signal-quality layer (#189), the
acceptance policy (#295) and any recovery backend (#297) -- and the adapter
that writes constraints should only translate it.  Before this contract the
adapter re-derived the decision from three unrelated inputs (a caller list,
the projected validity, and a Gaussian fit of its own), and the product could
express only "enabled / disabled / missing" through two numbers.

A :class:`ChannelDecision` says, for one channel on the consumer's time grid:

``usable``
    the measurement stands at the family's nominal weight;
``suspect``
    the measurement stands at an explicitly configured reduced weight (or at
    nominal when the policy is report-only);
``rejected``
    weight zero -- the measurement is not a constraint at this slice;
``missing``
    no measurement exists; the adapter keeps its deterministic zero-weight
    placeholder (#145) and writes nothing;
``recovered``
    a backend supplied a value, with its own uncertainty and weight factor
    and a provenance naming the backend.

States are per slice, so a channel that rails at 0.31 s stays usable at the
slices before it.  The mapping from state to weight is stated once, in
:func:`weight_factor_for`, and is deterministic.  This module imports only
NumPy: it holds decisions, it does not make them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "MISSING",
    "PROVENANCE_MEASURED",
    "REASON_CONDEMNED",
    "REASON_MANUAL_LIST",
    "REASON_NO_SIGNAL",
    "REASON_PROJECTED_VALIDITY",
    "RECOVERED",
    "REJECTED",
    "STATE_NAMES",
    "SUSPECT",
    "USABLE",
    "ChannelDecision",
    "ChannelDecisions",
    "decision_from_codes",
    "missing_decision",
    "rejected_decision",
    "state_name",
    "usable_decision",
    "weight_factor_for",
]

USABLE = 0
SUSPECT = 1
REJECTED = 2
MISSING = 3
RECOVERED = 4
STATE_NAMES = ("usable", "suspect", "rejected", "missing", "recovered")

#: Provenance of a decision whose value is the adapter's own measurement.
PROVENANCE_MEASURED = "measured"

#: The reason vocabulary.  Free text may follow, but these prefixes are what
#: a consumer can rely on.
REASON_NO_SIGNAL = "no_signal"
REASON_CONDEMNED = "condemned_whole_record"
REASON_PROJECTED_VALIDITY = "projected_validity"
REASON_MANUAL_LIST = "manual_exclusion_list"


def state_name(code: int) -> str:
    return STATE_NAMES[int(code)]


def weight_factor_for(state: np.ndarray, *, suspect_weight_factor: float = 1.0) -> np.ndarray:
    """The deterministic state -> weight-factor mapping.

    ``usable`` -> 1, ``suspect`` -> ``suspect_weight_factor`` (1 means the
    suspicion is report-only), ``rejected`` and ``missing`` -> 0.  A
    ``recovered`` state has no default: the backend that recovered the value
    owns its weight, so asking here is an error.
    """
    codes = np.asarray(state, dtype=np.int8).reshape(-1)
    if np.any(codes == RECOVERED):
        raise ValueError("a recovered state carries the backend's own weight factor")
    factor = np.ones(codes.size, dtype=float)
    factor[codes == SUSPECT] = float(suspect_weight_factor)
    factor[(codes == REJECTED) | (codes == MISSING)] = 0.0
    return factor


@dataclass(frozen=True, eq=False)
class ChannelDecision:
    """One channel's decision on the consumer's time grid."""

    kind: str
    index: int
    name: str
    state: np.ndarray
    weight_factor: np.ndarray
    value: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    provenance: str = PROVENANCE_MEASURED
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.int8).reshape(-1)
        if state.size == 0:
            raise ValueError(f"{self.kind}[{self.index}]: a decision needs at least one slice")
        if np.any(state < USABLE) or np.any(state > RECOVERED):
            raise ValueError(f"{self.kind}[{self.index}]: unknown state code in {state.tolist()}")
        factor = np.asarray(self.weight_factor, dtype=float).reshape(-1)
        if factor.shape != state.shape:
            raise ValueError(f"{self.kind}[{self.index}]: weight_factor shape {factor.shape} != state {state.shape}")
        if not np.all(np.isfinite(factor)) or np.any(factor < 0.0):
            raise ValueError(f"{self.kind}[{self.index}]: weight factors must be finite and non-negative")
        dead = (state == REJECTED) | (state == MISSING)
        if np.any(factor[dead] != 0.0):
            raise ValueError(f"{self.kind}[{self.index}]: a rejected or missing slice cannot carry weight")
        recovered = state == RECOVERED
        value = None if self.value is None else np.asarray(self.value, dtype=float).reshape(-1)
        if value is not None and value.shape != state.shape:
            raise ValueError(f"{self.kind}[{self.index}]: value shape {value.shape} != state {state.shape}")
        if np.any(recovered):
            if value is None or not np.all(np.isfinite(value[recovered])):
                raise ValueError(f"{self.kind}[{self.index}]: a recovered slice needs a finite value")
            if self.provenance == PROVENANCE_MEASURED:
                raise ValueError(f"{self.kind}[{self.index}]: a recovered value must name its backend")
        elif self.provenance != PROVENANCE_MEASURED:
            raise ValueError(f"{self.kind}[{self.index}]: provenance {self.provenance!r} without a recovered slice")
        uncertainty = None if self.uncertainty is None else np.asarray(self.uncertainty, dtype=float).reshape(-1)
        if uncertainty is not None and uncertainty.shape != state.shape:
            raise ValueError(f"{self.kind}[{self.index}]: uncertainty shape {uncertainty.shape} != state {state.shape}")
        for attribute, array in (("state", state), ("weight_factor", factor), ("value", value), ("uncertainty", uncertainty)):
            if array is not None:
                array.setflags(write=False)
            object.__setattr__(self, attribute, array)
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))

    @property
    def n_slices(self) -> int:
        return int(self.state.size)

    def state_at(self, slice_index: int) -> int:
        return int(self.state[int(slice_index)])

    def recovered_mask(self) -> np.ndarray:
        return self.state == RECOVERED

    @property
    def all_usable(self) -> bool:
        return bool(np.all(self.state == USABLE))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "index": self.index,
            "name": self.name,
            "state": [STATE_NAMES[code] for code in self.state.tolist()],
            "weight_factor": self.weight_factor.tolist(),
            "value": None if self.value is None else self.value.tolist(),
            "uncertainty": None if self.uncertainty is None else self.uncertainty.tolist(),
            "provenance": self.provenance,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, eq=False)
class ChannelDecisions:
    """Every submitted channel's decision on one time grid.

    A channel absent from ``entries`` was not submitted at all (a probe
    outside the equilibrium code's geometry, say); that is different from
    ``missing``, which is a submitted channel with no measurement.
    """

    times: np.ndarray
    entries: Mapping[tuple[str, int], ChannelDecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        times.setflags(write=False)
        object.__setattr__(self, "times", times)
        entries = dict(self.entries)
        for key, decision in entries.items():
            if key != (decision.kind, decision.index):
                raise ValueError(f"entry {key} holds a decision for {(decision.kind, decision.index)}")
            if decision.n_slices != times.size:
                raise ValueError(
                    f"{decision.kind}[{decision.index}] decides {decision.n_slices} slices on a grid of {times.size}"
                )
        object.__setattr__(self, "entries", entries)

    def get(self, kind: str, index: int) -> ChannelDecision | None:
        return self.entries.get((str(kind), int(index)))

    def indices(self, kind: str) -> tuple[int, ...]:
        return tuple(sorted(index for (entry_kind, index) in self.entries if entry_kind == kind))

    def state_at(self, kind: str, index: int, slice_index: int) -> int:
        return self.entries[(str(kind), int(index))].state_at(slice_index)

    def replaced(self, *decisions: ChannelDecision) -> "ChannelDecisions":
        """A new container with these decisions swapped in (same grid)."""
        entries = dict(self.entries)
        for decision in decisions:
            entries[(decision.kind, decision.index)] = decision
        return ChannelDecisions(times=self.times, entries=entries)

    def summary(self) -> dict[str, dict[str, int]]:
        """Channel-slice counts per state, per kind."""
        out: dict[str, dict[str, int]] = {}
        for (kind, _index), decision in self.entries.items():
            bucket = out.setdefault(kind, {name: 0 for name in STATE_NAMES})
            for code in decision.state.tolist():
                bucket[STATE_NAMES[code]] += 1
        return out

    def as_dict(self, *, only_notable: bool = False) -> dict[str, Any]:
        entries = [
            decision.as_dict()
            for _key, decision in sorted(self.entries.items())
            if not (only_notable and decision.all_usable)
        ]
        return {"schema_version": 1, "times": self.times.tolist(), "entries": entries}


def usable_decision(kind: str, index: int, name: str, n_slices: int, *, reasons: Iterable[str] = ()) -> ChannelDecision:
    state = np.full(int(n_slices), USABLE, dtype=np.int8)
    return ChannelDecision(kind, index, name, state, weight_factor_for(state), reasons=tuple(reasons))


def missing_decision(kind: str, index: int, name: str, n_slices: int) -> ChannelDecision:
    state = np.full(int(n_slices), MISSING, dtype=np.int8)
    return ChannelDecision(kind, index, name, state, np.zeros(int(n_slices)), reasons=(REASON_NO_SIGNAL,))


def rejected_decision(kind: str, index: int, name: str, n_slices: int, *, reason: str | Iterable[str]) -> ChannelDecision:
    state = np.full(int(n_slices), REJECTED, dtype=np.int8)
    reasons = (reason,) if isinstance(reason, str) else tuple(reason)
    return ChannelDecision(kind, index, name, state, np.zeros(int(n_slices)), reasons=reasons)


def decision_from_codes(
    kind: str,
    index: int,
    name: str,
    codes: np.ndarray | None,
    n_slices: int,
    *,
    min_validity: int,
    suspect_code: int,
    suspect_weight_factor: float | None = None,
    reasons: Iterable[str] = (),
) -> ChannelDecision:
    """A decision from per-slice validity codes.

    ``codes`` are the Data Dictionary validity codes resampled onto the
    consumer's grid of ``n_slices`` instants (``None`` when the datum was
    never assessed: everything usable, the #424 default).  A code at or above
    ``min_validity`` is usable; ``suspect_code`` below the floor becomes
    ``suspect`` only when a ``suspect_weight_factor`` is configured, else it
    is rejected like any other code below the floor.
    """
    if codes is None:
        return usable_decision(kind, index, name, n_slices, reasons=reasons)
    values = np.asarray(codes, dtype=int).reshape(-1)
    if values.size != int(n_slices):
        raise ValueError(f"{kind}[{index}]: {values.size} codes for {n_slices} slices")
    state = np.full(values.size, REJECTED, dtype=np.int8)
    state[values >= int(min_validity)] = USABLE
    if suspect_weight_factor is not None:
        state[(values == int(suspect_code)) & (values < int(min_validity))] = SUSPECT
    factor = weight_factor_for(state, suspect_weight_factor=1.0 if suspect_weight_factor is None else suspect_weight_factor)
    noted = tuple(reasons)
    if not np.all(state == USABLE) and REASON_PROJECTED_VALIDITY not in noted:
        noted = noted + (REASON_PROJECTED_VALIDITY,)
    return ChannelDecision(kind, index, name, state, factor, reasons=noted)
