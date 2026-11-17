"""IMPA source validity: is a processed internal-probe-array shot credible?

One question, asked of a shot :func:`vaft.process.impa.process_impa` has
already transformed: **can its measurements be trusted, and if not, why?**
That is source validity -- the same question :mod:`vaft.validation.magnetics`
answers for the magnetic probes -- so it lives here and speaks
:class:`~vaft.validation.model.ValidationStatus`.  The fitting and
compensation kernels it grades stay in :mod:`vaft.process.impa`; this module
computes no field, radius or coupling of its own.

Three layers, kept apart:

===========================  ===============================================
shot verdict                 :func:`validate_impa` -> :class:`ImpaAssessment`
per-channel DD validity      :func:`channel_validity`, :func:`bz_channel_validity`
stage-manifest spelling      :func:`manifest_label`, :func:`manifest_checks`
===========================  ===============================================

**The verdict is the worst check.**  One failing check fails the shot however
many pass, and reasons accumulate rather than short-circuit, so a caller sees
everything wrong at once.  ``INDETERMINATE`` and ``NOT_AVAILABLE`` are never
returned: every check here is decidable from the processed shot it is handed.

**Per-channel validity codes come from** :mod:`vaft.validation.imas`, the one
place that owns the Data Dictionary's validity vocabulary.  The mapping keeps
the codes IMPA products have always carried: a channel whose raw signal never
arrived is ``-2``; one that could not be calibrated, or belongs to a rejected
shot, is ``-1``; a shot accepted with warnings is ``1``; a clean one ``0``.
That ``1`` reads as *certified* in the Data Dictionary, which
:mod:`vaft.validation.magnetics` documents and deliberately caps in its own
scalar; changing it here would change every stored IMPA product, so it is
kept.

**The manifest keeps its spelling.**  Stage manifests have always recorded the
calibration verdict as ``valid`` / ``warning`` / ``invalid``, and the
replication gate reads ``invalid`` as *rejected*.  Those strings are
a serialization of :class:`ValidationStatus` produced only by
:func:`manifest_label`, not a vocabulary any caller reasons in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from vaft.validation.imas import (
    VALIDITY_CERTIFIED,
    VALIDITY_INVALID,
    VALIDITY_SUSPECT,
    VALIDITY_VALID,
)
from vaft.validation.model import ValidationStatus

if TYPE_CHECKING:  # the kernels' result types; importing them at run time would cycle
    from vaft.process.impa import (
        ImpaCouplingFit,
        ImpaCrosstalkFit,
        ImpaGeometryFit,
        TfCalibrationWindow,
    )

__all__ = [
    "MANIFEST_LABELS",
    "ImpaAssessment",
    "bz_channel_validity",
    "channel_validity",
    "manifest_checks",
    "manifest_label",
    "validate_impa",
]

_PASS = ValidationStatus.PASS
_WARN = ValidationStatus.WARN
_FAIL = ValidationStatus.FAIL

#: The array's channel count, equal to :data:`vaft.process.impa.IMPA_CHANNEL_COUNT`
#: (asserted by the tests); restated so this module never imports the kernels.
_EXPECTED_CHANNELS = 8

#: How a stage manifest spells each verdict.  Only the three states
#: :func:`validate_impa` can reach are listed.
MANIFEST_LABELS: Mapping[ValidationStatus, str] = {
    _PASS: "valid",
    _WARN: "warning",
    _FAIL: "invalid",
}

#: The legacy ``ImpaQuality`` strings, accepted on construction so a caller
#: that still builds an assessment from them keeps working.
_FROM_LABEL = {label: status for status, label in MANIFEST_LABELS.items()}

_SEVERITY = {_PASS: 0, _WARN: 1, _FAIL: 2}


def _coerce(state: ValidationStatus | str) -> ValidationStatus:
    """One of the three states a shot grading reaches, from either spelling.

    ``INDETERMINATE`` and ``NOT_AVAILABLE`` are refused: every check here is
    decidable, and a hand-built undecided assessment would otherwise project
    to a ``valid`` channel code.
    """
    if not isinstance(state, ValidationStatus):
        state = _FROM_LABEL[state] if state in _FROM_LABEL else ValidationStatus(state)
    if state not in _SEVERITY:
        raise ValueError(f"an IMPA assessment is PASS, WARN or FAIL, not {state.name}")
    return state


def _worst(*states: ValidationStatus) -> ValidationStatus:
    return max(states, key=lambda state: _SEVERITY[state])


@dataclass(frozen=True)
class ImpaAssessment:
    """One IMPA shot's source-validity verdict, its per-check states and reasons.

    ``status`` and every value of ``checks`` are :class:`ValidationStatus`.
    The legacy strings ``valid`` / ``warning`` / ``invalid`` are accepted on
    construction and converted, which is what keeps the deprecated
    ``vaft.process.impa.ImpaQuality`` name a thin alias of this class.
    """

    status: ValidationStatus
    checks: Mapping[str, ValidationStatus]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce(self.status))
        object.__setattr__(
            self, "checks", {name: _coerce(state) for name, state in dict(self.checks).items()}
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def is_usable(self) -> bool:
        """Whether the shot may be used at all: it passed, or passed with warnings."""
        return self.status in (_PASS, _WARN)


def manifest_label(status: ValidationStatus | str) -> str:
    """The stage-manifest spelling of a verdict: ``valid``, ``warning`` or ``invalid``."""
    return MANIFEST_LABELS[_coerce(status)]


def manifest_checks(assessment: ImpaAssessment) -> dict[str, str]:
    """``assessment.checks`` in the stage-manifest spelling."""
    return {name: manifest_label(state) for name, state in assessment.checks.items()}


def channel_validity(
    assessment: ImpaAssessment, *, wired: bool, coupled: bool, finite: bool
) -> tuple[int, str]:
    """The DD validity code and a short reason for one IMPA Hall channel.

    ``wired``: the raw signal was read.  ``coupled``: its TF coupling was
    fitted to a finite value.  ``finite``: its compensated field has at least
    one finite sample.  Channel-level failures win over the shot verdict,
    because they say more precisely what is wrong.
    """
    if not wired:
        return VALIDITY_INVALID, "raw signal unavailable"
    if not coupled:
        return VALIDITY_SUSPECT, "TF coupling could not be fitted"
    if not finite:
        return VALIDITY_SUSPECT, "compensated Bz is not finite"
    if assessment.status is _FAIL:
        return VALIDITY_SUSPECT, "shot-level IMPA calibration was rejected"
    if assessment.status is _WARN:
        return VALIDITY_CERTIFIED, "calibration accepted with warnings"
    return VALIDITY_VALID, "valid"


def bz_channel_validity(
    assessment: ImpaAssessment, *, wired: bool, calibrated: bool
) -> tuple[int, str]:
    """The DD validity code and a short reason for one vertical-field sensor.

    ``wired``: the raw signal was read.  ``calibrated``: its crosstalk angle
    was fitted to a finite value.  The verdict that applies is the
    ``bz_sensors`` check, not the shot's: a Hall-geometry warning says nothing
    about whether a vertical sensor is alive.
    """
    if not wired:
        return VALIDITY_INVALID, "raw signal unavailable"
    if not calibrated:
        return VALIDITY_SUSPECT, "crosstalk calibration could not be fitted"
    state = assessment.checks.get("bz_sensors")
    if state is _FAIL:
        return VALIDITY_SUSPECT, "vertical-field sensor calibration was rejected"
    if state is _WARN:
        return VALIDITY_CERTIFIED, "calibration accepted with warnings; sensor may be inactive"
    return VALIDITY_VALID, "valid"


def validate_impa(
    time: np.ndarray,
    raw: np.ndarray,
    b_measured: np.ndarray,
    b_z: np.ndarray,
    channel_valid: np.ndarray,
    window: TfCalibrationWindow | None,
    geometry: ImpaGeometryFit,
    coupling: ImpaCouplingFit | None,
    *,
    expected_channels: int = _EXPECTED_CHANNELS,
    max_normalized_rmse: float = 0.15,
    r_bounds: Sequence[float] = (0.1, 0.9),
    pitch_tolerance: float = 0.01,
    window_reasons: Sequence[str] = (),
    orientation: str = "toroidal",
    alpha_tolerance: float = 0.15,
    crosstalk: ImpaCrosstalkFit | None = None,
    min_crosstalk_r_squared: float = 0.8,
) -> ImpaAssessment:
    """Grade whether an IMPA array's processed measurements can be trusted, and say why not.

    Parameters
    ----------
    time : array_like
        Sample times [s].
    raw : array_like
        Raw probe records [V].
    b_measured : array_like
        Conditioned probe records [T].
    b_z : array_like
        Compensated vertical field [T].
    channel_valid : array_like
        Which channels to consider [-].
    window : TfCalibrationWindow
        The calibration interval that was used [-].
    geometry : ImpaGeometryFit
        The fitted geometry [-].
    coupling : ImpaCouplingFit
        The fitted coupling [-].
    expected_channels : int, optional
        How many channels the array should have [-].
    max_normalized_rmse : float, optional
        Largest geometry residual still acceptable [-].
    r_bounds : tuple of float, optional
        Radial bounds the fit should sit inside [m].
    pitch_tolerance : float, optional
        Largest departure of fitted from as-built spacing [m].
    window_reasons : sequence of str, optional
        Reasons carried from the window selection [-].
    orientation : str, optional
        Which field the array is taken to face [-].
    alpha_tolerance : float, optional
        Largest departure of the coupling from unity [-].
    crosstalk : ImpaCrosstalkFit, optional
        The crosstalk fit, when one was made [-].
    min_crosstalk_r_squared : float, optional
        Smallest acceptable crosstalk goodness of fit [-].

    Returns
    -------
    ImpaAssessment
        ``PASS``, ``WARN`` or ``FAIL``, each check's state, and the reasons [-].

    Processing steps
    ----------------
    1. Check the channels are present and the time axis usable.
    2. Check each channel's own signal health, including railing.
    3. Check the calibration window, then the geometry, then the coupling.
    4. Check the compensated signal and the vertical sensors.
    5. Combine by taking the worst verdict.

    Defaults
    --------
    The residual, spacing, coupling and goodness-of-fit thresholds are
    hard-coded.  ``max_normalized_rmse`` matches ``vest.yaml``
    ``impa.quality`` (0.15); ``pitch_tolerance`` does not (0.01 m here, 0.015 m
    there), and the VEST mapper does not pass the configured value, so 0.01 m is
    what every shot is graded with. The channel count and the radial bounds are
    machine-specific.

    Limitations
    -----------
    Grades the fits it is handed; it cannot tell a well-fitted wrong model from a
    right one. A shot whose array faces the toroidal field can pass the geometry
    checks and still have no usable vertical measurement.

    Provenance
    ----------
    Moved from ``vaft.process.impa.grade_impa_quality`` (formerly
    ``validate_impa``), verdicts unchanged; ``vest.yaml``
    ``impa.quality`` carries the configured thresholds.
    """
    checks: dict[str, ValidationStatus] = {}
    reasons: list[str] = list(window_reasons)
    time = np.asarray(time, dtype=float)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    channel_valid = np.asarray(channel_valid, dtype=bool)

    present = int(np.count_nonzero(channel_valid))
    if present == expected_channels:
        checks["channels_present"] = _PASS
    elif present == 0:
        checks["channels_present"] = _FAIL
        reasons.append("no IMPA raw channel could be read")
    else:
        checks["channels_present"] = _WARN
        reasons.append(f"{expected_channels - present} of {expected_channels} IMPA channels are unavailable")

    if time.size >= 2 and np.all(np.isfinite(time)) and np.all(np.diff(time) > 0):
        checks["time_axis"] = _PASS
    else:
        checks["time_axis"] = _FAIL
        reasons.append("IMPA time axis is empty, non-finite or not monotonic")

    channel_states: list[ValidationStatus] = []
    for channel in range(raw.shape[0]):
        if not channel_valid[channel]:
            continue
        values = raw[channel]
        if values.size == 0 or not np.any(np.isfinite(values)):
            channel_states.append(_FAIL)
            reasons.append(f"channel {channel} carries no finite raw sample")
        elif np.allclose(values, values.flat[0]):
            channel_states.append(_FAIL)
            reasons.append(f"channel {channel} is constant (dead or railed)")
        else:
            # A railed channel repeats one *exact* digitiser level; a genuine
            # analogue plateau still carries noise, so near-equality would
            # flag every flat TF top as clipping.
            railed = np.count_nonzero(values == np.max(values)) + np.count_nonzero(
                values == np.min(values)
            )
            if railed > 0.01 * values.size:
                channel_states.append(_WARN)
                reasons.append(f"channel {channel} holds one exact extreme level (probable clipping)")
            else:
                channel_states.append(_PASS)
    checks["channel_signals"] = _worst(_PASS, *channel_states) if channel_states else _FAIL

    if window is not None:
        checks["calibration_window"] = _PASS
    elif coupling is not None:
        # A calibration carried in from a reference shot is a legitimate
        # substitute for this shot's own clean interval.
        checks["calibration_window"] = _WARN
        reasons.append("no clean TF window in this shot; calibration taken from a reference shot")
    else:
        checks["calibration_window"] = _FAIL
        reasons.append("no clean TF calibration window was found in this shot")

    lower, upper = float(min(r_bounds)), float(max(r_bounds))
    geometry_state = _PASS
    if not geometry.monotonic:
        geometry_state = _FAIL
        reasons.append("fitted IMPA radii are not monotonically increasing")
    # Only positions that carry a wired channel are measurements; a nominal
    # radius for an unwired position says nothing about the hardware.
    wired = geometry.r[channel_valid] if channel_valid.size == geometry.r.size else geometry.r
    if not geometry.within_bounds or (wired.size and (np.any(wired < lower) or np.any(wired > upper))):
        geometry_state = _FAIL
        reasons.append(f"fitted IMPA radii fall outside the physical bounds [{lower}, {upper}] m")
    if geometry.bound_hit:
        geometry_state = _worst(geometry_state, _FAIL)
        reasons.append("the geometry fit saturated at a parameter bound")
    if geometry.pitch is not None and wired.size > 1:
        spacing = np.diff(wired)
        if np.any(np.abs(spacing - geometry.pitch) > pitch_tolerance):
            geometry_state = _worst(geometry_state, _WARN)
            reasons.append("channel spacing deviates from the configured radial pitch")
    if geometry.nrmse is not None and np.isfinite(geometry.nrmse) and geometry.nrmse > max_normalized_rmse:
        geometry_state = _worst(geometry_state, _WARN)
        reasons.append(
            f"geometry fit residual is {geometry.nrmse * 100:.1f}% of the signal spread, above the "
            f"{max_normalized_rmse * 100:.1f}% tolerance"
        )
    checks["geometry"] = geometry_state

    if coupling is None:
        checks["tf_coupling"] = _FAIL
        reasons.append("TF coupling could not be fitted")
    else:
        coupling_state = _PASS
        if not np.any(np.isfinite(coupling.alpha)):
            coupling_state = _FAIL
            reasons.append("no channel produced a finite TF coupling")
        if orientation == "toroidal":
            # A toroidally mounted probe should see essentially the whole
            # toroidal field; departures mean the array is not where or how
            # the configuration says it is.
            finite_alpha = np.isfinite(coupling.alpha)
            deviation = np.abs(np.abs(coupling.alpha[finite_alpha]) - 1.0)
            if deviation.size and np.max(deviation) > alpha_tolerance:
                coupling_state = _worst(coupling_state, _FAIL)
                stray = np.flatnonzero(
                    finite_alpha & (np.abs(np.abs(coupling.alpha) - 1.0) > alpha_tolerance)
                ).tolist()
                reasons.append(
                    f"TF coupling departs from a toroidally aligned probe (|alpha| - 1 > "
                    f"{alpha_tolerance}) on channels {stray}"
                )
        elif np.any(coupling.bound_hit):
            coupling_state = _worst(coupling_state, _FAIL)
            hits = np.flatnonzero(coupling.bound_hit).tolist()
            reasons.append(f"TF coupling exceeds the configured tilt bounds on channels {hits}")
        finite = np.isfinite(coupling.nrmse)
        if finite.any() and np.nanmax(coupling.nrmse[finite]) > max_normalized_rmse:
            coupling_state = _worst(coupling_state, _WARN)
            reasons.append("TF coupling residuals exceed the configured tolerance on at least one channel")
        checks["tf_coupling"] = coupling_state

    b_z = np.atleast_2d(np.asarray(b_z, dtype=float))
    usable = b_z[channel_valid] if channel_valid.size == b_z.shape[0] else b_z
    if usable.size and np.all(np.isfinite(usable)):
        checks["compensated_signal"] = _PASS
    elif usable.size and np.any(np.isfinite(usable)):
        checks["compensated_signal"] = _WARN
        reasons.append("some compensated Bz samples are not finite")
    else:
        checks["compensated_signal"] = _FAIL
        reasons.append("compensated Bz is not finite on any usable channel")

    if crosstalk is not None:
        bz_state = _PASS
        finite = np.isfinite(crosstalk.slope_v_per_t)
        if not finite.any():
            bz_state = _FAIL
            reasons.append("no Bz sensor produced a finite crosstalk calibration")
        else:
            # A misalignment-angle bound can only be checked once the Bz
            # sensor's own gain is known; bound_hit is all-False without one.
            if np.any(crosstalk.bound_hit[finite]):
                bz_state = _worst(bz_state, _FAIL)
                hits = np.flatnonzero(finite & crosstalk.bound_hit).tolist()
                reasons.append(f"Bz sensor incident angle exceeds the configured limit on channels {hits}")
            weak = finite & (crosstalk.r_squared < min_crosstalk_r_squared)
            if np.any(weak):
                bz_state = _worst(bz_state, _WARN)
                reasons.append(
                    "the toroidal field does not explain the Bz sensor signal on channels "
                    f"{np.flatnonzero(weak).tolist()}; those sensors may be inactive"
                )
        checks["bz_sensors"] = bz_state

    return ImpaAssessment(status=_worst(*checks.values()), checks=checks, reasons=tuple(reasons))
