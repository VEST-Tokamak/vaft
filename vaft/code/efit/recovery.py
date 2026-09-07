"""The Gaussian-profile recovery of rejected B-probe constraints, as a backend (issue #296).

The legacy constraint builder fitted a four-parameter Gaussian through each
poloidal-probe family (inboard, side, outboard) at every reconstruction slice
and replaced the readings of "broken" probes with the fit.  That is a
*recovery policy*: it decides what value a rejected channel contributes and
with what confidence, which is not the constraint adapter's business (#296).
It lives here as a compatibility backend with the same numbers and the same
optimizer calls, consumed through the decision contract:

    decisions -> gaussian_probe_recovery(EQ, decisions, ...) -> decisions'

Two rules are made explicit that the legacy code only implied:

* a rejected or missing probe is never a fit point -- zeroing its weight
  afterwards does not undo the pull it exerted on every neighbour's fitted
  value;
* a recovered value never re-enables a channel the quality layer rejected.
  It replaces the number; the weight stays zero unless the only reason for
  the rejection was the routine manual list, which is a configuration
  choice rather than a finding about the signal.

A GP or Bayesian backend (#297) plugs in the same way: take decisions, return
decisions with recovered slices carrying value, uncertainty, weight
factor and provenance.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np
from scipy import optimize

from vaft.machine_mapping.magnetics import (
    INBOARD_PROBE_MAX_R,
    OUTBOARD_PROBE_MIN_R,
    SIDE_PROBE_MIN_ABS_Z,
)
from vaft.validation.channel_decision import (
    MISSING,
    REASON_MANUAL_LIST,
    RECOVERED,
    REJECTED,
    SUSPECT,
    USABLE,
    ChannelDecision,
    ChannelDecisions,
)

from .legacy import gauss_fit4, min_gauss_fit4

__all__ = [
    "GAUSSIAN_PROVENANCE",
    "IP_FIT_FLOOR",
    "PROBE_Z_TABLE",
    "ProbeFamilies",
    "family_of",
    "gaussian_probe_recovery",
    "probe_families",
]

#: The fit abscissa per EFIT probe index: the legacy Bzx table, moved
#: verbatim from the constraint builder.  Inboard probes are indexed by their
#: vertical position, side probes by their radial position along the top and
#: bottom, outboard probes by their vertical position again.  Positional by
#: EFIT probe index; it is not read from the ODS.
PROBE_Z_TABLE: tuple[float, ...] = (
    0.54, 0.5, 0.46, 0.42, 0.38, 0.34, 0.3, 0.26,
    0.22, 0.16, 0.12, 0.08, 0.04, 0, -0.04, -0.08,
    -0.12, -0.16, -0.22, -0.26, -0.3, -0.34, -0.38, -0.42,
    -0.46, -0.5, -0.54, 0.42, 0.38, 0.34, 0.3, 0.26,
    0.22, 0.18, 0.1, 0.06, 0.02, -0.02, -0.06, -0.1,
    -0.14, -0.18, -0.22, -0.26, -0.3, -0.34, -0.38, -0.42,
    0.8328, 0.8728, 0.9128, 0.9528, 0.9928, 1.0328, 1.0728, 1.1128,
    -0.8328, -0.8728, -0.9128, -0.9528, -0.9928, -1.0328, -1.0728, -1.1128,
)

#: Below this plasma current no fit is attempted (the legacy IPLIM).
IP_FIT_FLOOR = 45000.0
GAUSSIAN_PROVENANCE = "gaussian_fit4"
_START = (0.1, 0.0, 0.2, -0.1)
_START_MIRRORED = (-0.1, 0.0, 0.2, -0.1)
PROBE_KIND = "b_field_pol_probe"


class ProbeFamilies(NamedTuple):
    """Zero-based probe indexes per geometric family, ascending."""

    inboard: np.ndarray
    side: np.ndarray
    outboard: np.ndarray


def probe_families(magnetics: Any, *, count: int) -> ProbeFamilies:
    """The legacy family membership, from probe positions, clamped to count."""
    r = np.asarray(magnetics["b_field_pol_probe.:.position.r"], dtype=float).reshape(-1)
    z = np.asarray(magnetics["b_field_pol_probe.:.position.z"], dtype=float).reshape(-1)
    limit = min(int(count), r.size, z.size)
    indexes = np.arange(limit)
    return ProbeFamilies(
        inboard=indexes[r[:limit] < INBOARD_PROBE_MAX_R],
        side=indexes[np.abs(z[:limit]) > SIDE_PROBE_MIN_ABS_Z],
        outboard=indexes[r[:limit] > OUTBOARD_PROBE_MIN_R],
    )


def family_of(index: int, families: ProbeFamilies) -> str | None:
    """Which family a probe belongs to; the last family wins, as the legacy
    builder's sequential assignment did (on VEST the families are disjoint)."""
    found = None
    for name in ("inboard", "side", "outboard"):
        if int(index) in getattr(families, name):
            found = name
    return found


def _fit(x: np.ndarray, y: np.ndarray, starts: Sequence[Sequence[float]]) -> np.ndarray:
    best = None
    for start in starts:
        result = optimize.minimize(
            min_gauss_fit4, list(start), args=(x, y), method="SLSQP", tol=1.0e-8, options={"maxiter": 1000}
        )
        if best is None or min_gauss_fit4(result.x, x, y) < min_gauss_fit4(best, x, y):
            best = result.x
    return np.asarray(best, dtype=float)


def gaussian_probe_recovery(
    EQ: Any,
    decisions: ChannelDecisions,
    *,
    mode: int,
    families: ProbeFamilies,
    geometry: Sequence[float] = PROBE_Z_TABLE,
    ip_floor: float = IP_FIT_FLOOR,
    uncertainty: str = "residual_rms",
) -> ChannelDecisions:
    """Replace rejected (mode=1) or every (mode=2) probe reading by a family Gaussian fit.

    EQ is the constraint equilibrium the adapter has already formed: the
    fit reads time_slice.i.constraints.bpol_probe.j.measured (window
    averaged) and constraints.ip.measured (the floor), and never writes
    to it.  Per slice above the floor and per family, the fit points are the
    members whose decision is usable or suspect at that slice; the targets
    are the rejected members (mode 1) or every member (mode 2); missing
    members are never targets.

    uncertainty="residual_rms" gives each recovered slice the fit's
    residual RMS over its family; "legacy" leaves the adapter's own
    measured_error_upper in place, which is what the routine pipeline
    used before this backend existed.
    """
    if int(mode) not in (1, 2):
        raise ValueError(f"mode must be 1 (rejected probes) or 2 (every probe), got {mode!r}")
    if uncertainty not in ("residual_rms", "legacy"):
        raise ValueError(f"uncertainty must be 'residual_rms' or 'legacy', got {uncertainty!r}")
    n_slices = decisions.times.size
    members = {
        "inboard": (families.inboard, (_START,)),
        "side": (families.side, (_START_MIRRORED, _START)),
        "outboard": (families.outboard, (_START,)),
    }
    recovered_value: dict[int, np.ndarray] = {}
    recovered_error: dict[int, np.ndarray] = {}
    for i in range(n_slices):
        ip_path = f"time_slice.{i}.constraints.ip.measured"
        if ip_path not in EQ or not float(EQ[ip_path]) > float(ip_floor):
            continue
        for _family, (indexes, starts) in members.items():
            points_x, points_y, targets = [], [], []
            for j in (int(k) for k in indexes):
                decision = decisions.get(PROBE_KIND, j)
                path = f"time_slice.{i}.constraints.bpol_probe.{j}.measured"
                if decision is None or path not in EQ:
                    continue
                state = decision.state_at(i)
                if state in (USABLE, SUSPECT):
                    points_x.append(float(geometry[j]))
                    points_y.append(float(EQ[path]))
                    if int(mode) == 2:
                        targets.append(j)
                elif state == REJECTED:
                    targets.append(j)
            if not targets or not points_x:
                continue
            x, y = np.asarray(points_x), np.asarray(points_y)
            coef = _fit(x, y, starts)
            error = float(min_gauss_fit4(coef, x, y)) / np.sqrt(x.size)
            for j in targets:
                recovered_value.setdefault(j, np.full(n_slices, np.nan))[i] = float(gauss_fit4(coef, geometry[j]))
                recovered_error.setdefault(j, np.full(n_slices, np.nan))[i] = error

    replaced: list[ChannelDecision] = []
    for j, values in recovered_value.items():
        decision = decisions.get(PROBE_KIND, j)
        assert decision is not None
        recovered = np.isfinite(values)
        state = decision.state.copy()
        state[recovered] = RECOVERED
        factor = decision.weight_factor.copy()
        manual_only = all(reason.startswith(REASON_MANUAL_LIST) for reason in decision.reasons) and bool(decision.reasons)
        was_rejected = decision.state == REJECTED
        # A value can be recovered for a rejected channel, but only the
        # manual list's rejection is undone by it: a channel the quality
        # layer rejected keeps weight zero.
        factor[recovered & was_rejected] = 1.0 if manual_only else 0.0
        value = np.where(recovered, values, np.nan) if decision.value is None else np.where(recovered, values, decision.value)
        error = None
        if uncertainty == "residual_rms":
            error = np.where(recovered, recovered_error[j], np.nan) if decision.uncertainty is None else np.where(
                recovered, recovered_error[j], decision.uncertainty
            )
        replaced.append(
            ChannelDecision(
                decision.kind, decision.index, decision.name, state, factor,
                value=value, uncertainty=error, provenance=GAUSSIAN_PROVENANCE,
                reasons=decision.reasons + (f"recovered_by_{GAUSSIAN_PROVENANCE}",),
            )
        )
    return decisions.replaced(*replaced)
