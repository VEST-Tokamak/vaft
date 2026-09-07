"""The acceptance policy that decides which magnetic channels EFIT may use (issue #296).

This is the one place the chain

    mapped magnetics -> signal quality (#189) -> acceptance policy -> constraints

turns evidence into a :class:`~vaft.validation.channel_decision.ChannelDecision`
per channel and reconstruction slice.  It reads the validity the diagnostics
stage projected and nothing else: no detector runs here, no vacuum model is
consulted, and no machine-specific list lives here.  The routine manual
exclusion list is accepted as an *input*, recorded with its own reason, so
that removing it (#295 §4) is a change to a caller's configuration and not
to this module.

The rules, in the order they are applied to each channel:

1. a probe beyond the count EFIT's geometry represents is not submitted at
   all -- no decision is formed;
2. a channel with no waveform on the diagnostics grid is ``missing``;
3. a channel the quality layer condemned outright is ``rejected`` everywhere;
4. a channel on the manual list is ``rejected`` everywhere;
5. otherwise the projected per-sample validity, resampled to the slices,
   decides slice by slice; a whole-window ``suspects`` mapping (for example
   from :func:`vaft.validation.flux_loop_assessment.assess_flux_loops`) can
   mark usable slices ``suspect``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from vaft.validation.channel_decision import (
    REASON_CONDEMNED,
    REASON_MANUAL_LIST,
    SUSPECT,
    USABLE,
    ChannelDecision,
    ChannelDecisions,
    decision_from_codes,
    missing_decision,
    rejected_decision,
    weight_factor_for,
)
from vaft.validation.imas import (
    VALIDITY_SUSPECT,
    VALIDITY_VALID,
    is_condemned_channel,
    read_validity_record,
    signal_label,
    signal_matches_root_time,
)
from vaft.validation.validity import codes_at

__all__ = [
    "EFIT_KINDS",
    "EFIT_QUANTITY",
    "condemned_channels",
    "decide_efit_channels",
    "efit_probe_count",
    "split_manual_rejections",
]

#: The magnetics families EFIT takes constraints from, and the processed
#: quantity each family's validity is written on.
EFIT_KINDS = ("b_field_pol_probe", "flux_loop")
EFIT_QUANTITY = {"b_field_pol_probe": "field", "flux_loop": "flux"}


def _count(source: Any, kind: str) -> int:
    return len(source[f"magnetics.{kind}"]) if f"magnetics.{kind}" in source else 0


def _name(source: Any, kind: str, index: int) -> str:
    return signal_label(source, f"magnetics.{kind}.{index}", f"{kind}[{index}]")


def efit_probe_count(source: Any) -> int:
    """The number of B-pol probes EFIT's geometry represents.

    ``min(present, defined)``: the magnetics IDS also carries trailing
    toroidal-Mirnov channels that are not in EFIT's ``dprobe.dat`` and must
    not become constraints.  This is also the offset the routine manual list
    uses for flux loops, so every consumer must take it from here.
    """
    from vaft.machine_mapping.magnetics import vest_equilibrium_magnetics_channel_definitions

    defined = sum(
        1 for entry in vest_equilibrium_magnetics_channel_definitions() if entry.get("kind") == "b_field_pol_probe"
    )
    present = _count(source, "b_field_pol_probe")
    return min(present, defined) if defined else present


def split_manual_rejections(manual: Iterable[int], *, nbprobe: int) -> dict[tuple[str, int], int]:
    """Combined one-based indexes -> ``(kind, index)``.

    The inverse of :func:`vaft.validation.flux_loop_assessment.manual_exclusion_index`:
    ``1..nbprobe`` are probes, above that flux loops at ``index + nbprobe``.
    """
    out: dict[tuple[str, int], int] = {}
    for item in manual:
        combined = int(item)
        if combined <= 0:
            raise ValueError(f"manual rejection index {combined} is not one-based")
        if combined <= int(nbprobe):
            out[("b_field_pol_probe", combined - 1)] = combined
        else:
            out[("flux_loop", combined - 1 - int(nbprobe))] = combined
    return out


def condemned_channels(source: Any, *, nbprobe: int) -> set[int]:
    """Legacy-style combined zero-based indexes of channels condemned outright.

    Judged on the time-resolved validity through
    :func:`vaft.validation.imas.is_condemned_channel`: a channel that holds
    its last value after the diagnostics window is not condemned, one the
    quality layer rejected in its entirety is.  Probes map to their own
    index, flux loops to ``index + nbprobe``.
    """
    condemned: set[int] = set()
    for kind, offset in (("b_field_pol_probe", 0), ("flux_loop", int(nbprobe))):
        quantity = EFIT_QUANTITY[kind]
        for index in range(_count(source, kind)):
            if is_condemned_channel(source, f"magnetics.{kind}.{index}.{quantity}"):
                condemned.add(index + offset)
    return condemned


def decide_efit_channels(
    source: Any,
    times: Any,
    *,
    nbprobe: int | None = None,
    manual_rejections: Iterable[int] = (),
    min_validity: int = VALIDITY_VALID,
    suspect_weight_factor: float | None = None,
    suspects: Mapping[tuple[str, int], str] | None = None,
    kinds: Iterable[str] = EFIT_KINDS,
) -> ChannelDecisions:
    """Decide every EFIT-facing magnetics channel at every slice of ``times``.

    See the module docstring for the rules.  ``manual_rejections`` are the
    routine configuration's combined one-based indexes; ``suspects`` maps
    ``(kind, index)`` to a reason and marks that channel's usable slices
    suspect (weight factor ``suspect_weight_factor``, or report-only at 1
    when that is ``None``).
    """
    grid = np.asarray(times, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("no reconstruction times to decide for")
    probe_count = efit_probe_count(source) if nbprobe is None else int(nbprobe)
    manual = split_manual_rejections(manual_rejections, nbprobe=probe_count)
    flagged = dict(suspects or {})
    factor_for_suspect = 1.0 if suspect_weight_factor is None else float(suspect_weight_factor)

    entries: dict[tuple[str, int], ChannelDecision] = {}
    for kind in kinds:
        quantity = EFIT_QUANTITY[kind]
        for index in range(_count(source, kind)):
            if kind == "b_field_pol_probe" and index >= probe_count:
                continue
            name = _name(source, kind, index)
            base = f"magnetics.{kind}.{index}.{quantity}"
            key = (kind, index)
            if not signal_matches_root_time(source, base):
                entries[key] = missing_decision(kind, index, name, grid.size)
                continue
            reasons: list[str] = []
            if is_condemned_channel(source, base, min_validity=min_validity):
                reasons.append(REASON_CONDEMNED)
            if key in manual:
                reasons.append(f"{REASON_MANUAL_LIST}:{manual[key]}")
            if reasons:
                entries[key] = rejected_decision(kind, index, name, grid.size, reason=reasons)
                continue
            record = read_validity_record(source, base)
            codes = codes_at(record, times=grid) if record.assessed else None
            decision = decision_from_codes(
                kind,
                index,
                name,
                codes,
                grid.size,
                min_validity=min_validity,
                suspect_code=VALIDITY_SUSPECT,
                suspect_weight_factor=suspect_weight_factor,
            )
            if key in flagged:
                state = decision.state.copy()
                state[state == USABLE] = SUSPECT
                decision = ChannelDecision(
                    kind, index, name, state,
                    weight_factor_for(state, suspect_weight_factor=factor_for_suspect),
                    reasons=decision.reasons + (f"suspect:{flagged[key]}",),
                )
            entries[key] = decision
    return ChannelDecisions(times=grid, entries=entries)
