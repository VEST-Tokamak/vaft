"""Is the plasma-current Rogowski recording a current at all? (#1373, #1543)

A Rogowski coil that has no plasma, no coil drive and no induced structure
current inside its contour must read zero.  VEST's discharge, its PF swing and
the induced currents that follow it all live inside a window of a few tens of
milliseconds around 0.3 s, so the stretches of the record well before and well
after that window are a *categorical* test of the sensor: hundreds of kA there
cannot be a plasma, and no baseline choice can turn such a record into one.

Shots 45780-45796 are the example that motivated this: field 109 toggles
between about +100 and -100 kA as a square wave for the whole second, and the
linear baseline then extrapolates that into a 1.7 MA "plasma" (#1373).

Only that categorical evidence becomes validity here.  Softer symptoms -- a
post-discharge tail, a slope in the baseline window -- are physically possible
(the inner Rogowski links the induced current of the structure around the CS),
so they are reported as metrics and never fail the channel (#189).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .validity import VALIDITY_INVALID, VALIDITY_VALID

__all__ = [
    "PlasmaCurrentQuality",
    "PlasmaCurrentQualityConfig",
    "assess_plasma_rogowski",
]


@dataclass(frozen=True)
class PlasmaCurrentQualityConfig:
    """Where the record must be quiet, and how quiet.

    ``quiet_windows`` are on the acquisition's own clock (seconds).  The
    defaults bracket the VEST discharge with a margin: the PF swing starts
    after 0.27 s and every induced current has decayed long before 0.45 s.

    ``max_off_discharge_p2p`` is the robust (0.5-99.5 percentile) peak-to-peak
    the quiet stretches may show, in amperes.  Healthy shots reach about
    50 kA there (spikes and the slow tail of the induced current); the square-
    wave fault reaches about 200 kA.  The value is set from the population
    scan recorded in #1543 and #1373.
    """

    quiet_windows: tuple[tuple[float, float], ...] = ((0.0, 0.20), (0.45, 1.0))
    reference_window: tuple[float, float] = (0.20, 0.24)
    max_off_discharge_p2p: float = 120e3
    min_quiet_samples: int = 1000


@dataclass(frozen=True)
class PlasmaCurrentQuality:
    validity: int
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"validity": self.validity, "reason": self.reason, "metrics": dict(self.metrics)}


def _p2p(values: np.ndarray) -> float:
    return float(np.percentile(values, 99.5) - np.percentile(values, 0.5))


def assess_plasma_rogowski(
    time: np.ndarray,
    current: np.ndarray,
    config: PlasmaCurrentQualityConfig | None = None,
) -> PlasmaCurrentQuality:
    """Judge a calibrated plasma-current Rogowski record over its full span.

    ``current`` is the sensor current before baseline removal (amperes), as
    :func:`vaft.machine_mapping.magnetics.vest_plasma_rogowski_current`
    returns it.  The result is a whole-record verdict: ``-2`` when the quiet
    stretches carry more than ``max_off_discharge_p2p``, else ``0``.  A record
    too short to contain the quiet stretches cannot be judged and is ``0`` with
    that said in ``reason``; absence of evidence is not a fault.
    """
    config = config or PlasmaCurrentQualityConfig()
    t = np.asarray(time, dtype=float).reshape(-1)
    x = np.asarray(current, dtype=float).reshape(-1)
    finite = np.isfinite(t) & np.isfinite(x)
    t, x = t[finite], x[finite]
    reference = (t >= config.reference_window[0]) & (t <= config.reference_window[1])
    quiet = np.zeros(t.size, dtype=bool)
    for start, end in config.quiet_windows:
        quiet |= (t >= start) & (t <= end)
    if quiet.sum() < config.min_quiet_samples or not reference.any():
        return PlasmaCurrentQuality(
            VALIDITY_VALID,
            "record does not span the quiet windows; not judged",
            {"quiet_samples": float(quiet.sum())},
        )
    level = float(np.median(x[reference]))
    p2p = _p2p(x[quiet] - level)
    metrics = {
        "off_discharge_p2p": p2p,
        "off_discharge_absmax": float(np.max(np.abs(x[quiet] - level))),
        "reference_level": level,
        "quiet_samples": float(quiet.sum()),
    }
    if p2p > config.max_off_discharge_p2p:
        return PlasmaCurrentQuality(
            VALIDITY_INVALID,
            f"{p2p / 1e3:.0f} kA peak-to-peak outside the discharge window, where no current can flow "
            f"(limit {config.max_off_discharge_p2p / 1e3:.0f} kA): the sensor is not recording a current",
            metrics,
        )
    return PlasmaCurrentQuality(VALIDITY_VALID, "", metrics)
