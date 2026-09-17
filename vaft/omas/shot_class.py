"""What kind of shot a product records, decided from the shared timing (issue #409).

Three classes, decided in this order and with the deciding check on record:

* ``Plasma`` -- :func:`vaft.omas.plasma_timing.plasma_timing` found a
  plasma-current pulse (whichever source answered for the window); when the
  current could not be judged at all (``ip_unusable``: a condemned or
  absent-inside-the-span channel) the light stands in, and a window it saw
  is ``Plasma`` too, flagged so;
* ``BD failure`` -- a usable current shows no pulse, but the shot was
  attempted: the barometry pressure responded to the gas puff, or the light
  saw a window (a flash without current);
* ``Vacuum`` -- none of those.

The pressure response is judged over the whole record (the puff precedes the
analysis window) with :func:`vaft.process.signal_processing.is_signal_active`;
a product without barometry says so (``barometry_absent``) and is judged on
the light alone.  A product without a plasma current cannot be classified
either way and raises the timing's :class:`PlasmaTimingError` -- the old
classifier printed an error and answered ``Vacuum``.  Nothing here writes
the ODS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vaft.machine_mapping.utils import PlasmaTimingPolicy
from vaft.ods_access import path_value
from vaft.process.signal_processing import is_signal_active

from .plasma_timing import PlasmaTiming, plasma_timing

__all__ = [
    "CLASS_BD_FAILURE",
    "CLASS_PLASMA",
    "CLASS_VACUUM",
    "DECIDED_IP_PULSE",
    "DECIDED_NONE",
    "DECIDED_OPTICAL_WINDOW",
    "DECIDED_PRESSURE",
    "PRESSURE_BASE",
    "ShotClass",
    "pressure_response",
    "shot_class",
]

CLASS_PLASMA = "Plasma"
CLASS_BD_FAILURE = "BD failure"
CLASS_VACUUM = "Vacuum"

DECIDED_IP_PULSE = "ip_pulse"
DECIDED_PRESSURE = "pressure_response"
DECIDED_OPTICAL_WINDOW = "optical_window"
DECIDED_NONE = "none"

PRESSURE_BASE = "barometry.gauge.0.pressure"

#: The timing flags a class record repeats, so a reader need not open the timing.
_TIMING_FLAGS = ("ip_no_pulse", "ip_unusable", "no_plasma_timing", "halpha_dark_with_ip_pulse")


@dataclass(frozen=True)
class ShotClass:
    """The class of one shot and the check that decided it."""

    label: str
    decided_by: str
    ip_pulse: bool
    optical_window: bool
    pressure_active: bool | None
    timing: PlasmaTiming
    flags: tuple[str, ...]
    reason: str

    def __str__(self) -> str:
        return self.label

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "decided_by": self.decided_by,
            "ip_pulse": self.ip_pulse,
            "optical_window": self.optical_window,
            "pressure_active": self.pressure_active,
            "agreement": self.timing.agreement,
            "reason": self.reason,
            "flags": list(self.flags),
            "timing": self.timing.summary(),
        }

    def record(self) -> dict[str, Any]:
        return {**{k: v for k, v in self.summary().items() if k != "timing"}, "timing": self.timing.record()}


def pressure_response(ods: Any, *, var_ratio_thresh: float = 1e-2) -> bool | None:
    """Whether the barometry pressure responded, or ``None`` when the product carries none.

    ``var_ratio_thresh`` is handed to both of :func:`is_signal_active`'s
    ratio thresholds, so a trace is flat only when its variance and its
    sample-to-sample change are both below it relative to its own level.
    """
    data = path_value(ods, f"{PRESSURE_BASE}.data")
    if data is None:
        return None
    values = np.asarray(data, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]   # a NaN would make the activity ratios NaN and read as active
    if values.size < 2:
        return None
    # both ratios relative to the trace's own level, as the classifier always passed them
    threshold = float(var_ratio_thresh)
    return bool(is_signal_active(values, var_ratio_thresh=threshold, change_ratio_thresh=threshold))


def shot_class(
    ods: Any,
    *,
    timing: PlasmaTiming | None = None,
    policy: PlasmaTimingPolicy | None = None,
    pressure_threshold: float = 1e-2,
) -> ShotClass:
    """Classify a shot from the shared timing and the gas response.

    ``timing`` may be handed in when the caller already has it.  ``Plasma``
    when a plasma-current pulse was found; otherwise ``BD failure`` when the
    pressure responded or the light saw a window; otherwise ``Vacuum``.
    """
    if timing is None:
        timing = plasma_timing(ods, policy=policy)
    ip_pulse = timing.ip is not None and timing.ip.found
    ip_unusable = timing.ip is None
    optical_window = timing.optical is not None and timing.optical.found
    pressure_active = pressure_response(ods, var_ratio_thresh=pressure_threshold)
    flags = [flag for flag in timing.flags if flag in _TIMING_FLAGS]
    if pressure_active is None:
        flags.append("barometry_absent")

    if ip_pulse:
        label, decided, reason = CLASS_PLASMA, DECIDED_IP_PULSE, (
            f"a plasma-current pulse was found ({timing.source} window {timing.onset:.4f}-{timing.offset:.4f} s)"
        )
    elif ip_unusable and optical_window:
        label, decided, reason = CLASS_PLASMA, DECIDED_OPTICAL_WINDOW, (
            f"the plasma current could not be judged (ip_unusable) and the light saw a window "
            f"({timing.optical.start:.4f}-{timing.optical.end:.4f} s)"
        )
    elif pressure_active:
        label, decided, reason = CLASS_BD_FAILURE, DECIDED_PRESSURE, (
            "no plasma-current pulse, but the barometry pressure responded to the gas puff"
        )
    elif optical_window:
        label, decided, reason = CLASS_BD_FAILURE, DECIDED_OPTICAL_WINDOW, (
            f"no plasma-current pulse, but the light saw a window ({timing.optical.start:.4f}-{timing.optical.end:.4f} s)"
        )
    else:
        label, decided, reason = CLASS_VACUUM, DECIDED_NONE, (
            "no plasma-current pulse, no pressure response"
            + (" (barometry absent)" if pressure_active is None else "")
            + ", no light"
        )
    return ShotClass(
        label=label, decided_by=decided, ip_pulse=bool(ip_pulse), optical_window=bool(optical_window),
        pressure_active=pressure_active, timing=timing, flags=tuple(dict.fromkeys(flags)), reason=reason,
    )
