"""Generic algorithms for the VEST Internal Magnetic Probe Array.

The array is an insertable midplane probe carrying, at each of eight radial
positions, a toroidal-field Hall sensor and a vertical-field sensor.  The
radial-field sensors are retired: the original design used them only to
confirm the probe sat at the midplane, where the radial field is negligible.

The two sets play different roles [1]_:

* the Hall channels establish *where the array is*.  The toroidal field goes
  as one over the major radius, so measuring it against the known coil
  current gives each sensor's radius, and the fitted spacing against the
  rigid as-built pitch gives the probe's incident angle.
* they also establish *how each vertical sensor is aligned*.  On an interval
  driven by the toroidal field alone the true vertical field is negligible,
  so whatever the vertical sensor reads there is crosstalk, and its ratio to
  the measured toroidal field is that sensor's misalignment.
* the vertical sensors, once corrected for that crosstalk, carry the physics
  measurement: the toroidal current density from the radial gradient of the
  vertical field at the midplane.

Every routine here works on plain arrays, so the machine-specific parts --
raw field codes, shot-era geometry and thresholds -- stay in
:mod:`vaft.machine_mapping.impa` and ``vest.yaml``.

Notation
--------
B_meas   : field a probe actually measures                          [T]
Bt       : vacuum toroidal field at a radius, mu0 N I / (2 pi R)    [T]
Bz       : vertical field, after compensation                       [T]
alpha    : probe coupling to the toroidal field, sin of its tilt    [-]
beta     : per-probe offset                                         [T]
kappa    : the observable ratio alpha / R                         [1/m]
R0       : major radius of the innermost channel                    [m]
I_TF     : toroidal field coil current                              [A]

Conventions
-----------
**The measurement model.**  Each probe sees a mixture of the local poloidal
field and the toroidal field::

    B_meas_i(t) = alpha_i * Bt(R_i, t) + Bz_i(t) * sqrt(1 - alpha_i**2) + beta_i

Inverting it gives the compensated internal field::

    Bz_i = (B_meas_i - alpha_i * Bt(R_i)) / sqrt(1 - alpha_i**2)

**Sign.**  VAFT takes the coil current as the raw field times ``-3e4``, as
:mod:`vaft.machine_mapping.tf` already does, together with a Hall gain of
``-2/15`` T/V.  That is the "Data gain" on the array's own wiring datasheet,
the authoritative hardware configuration, and it matches neither legacy
script exactly [2]_.  All three pairings give the same ratio of measured to
toroidal field up to an overall sign, so **the fits here are sign-agnostic**:
they detect the working polarity from the data rather than assuming a
positive coupling, and only the magnitude of the coupling carries physical
meaning.  The shipped configuration default disagrees with that gain's sign,
which is tracked in #624.

The datasheet leaves the vertical sensors' own volts-to-tesla gain
unspecified, so those waveforms stay in native volts throughout; see
:class:`ImpaCrosstalkFit` for what that means for the crosstalk angle.

**Geometry and coupling are degenerate.**  During a toroidal-field-only
interval that field is the only driver, so a probe measures
``alpha_i * mu0 * N * I_TF / (2 * pi * R_i)`` and only the ratio
``kappa_i = alpha_i / R_i`` is observable.  Assuming unit coupling yields the
radial position, which is the legacy method; a known radius yields the
coupling instead.  The two cannot be recovered together from that data alone.

**Geometry is per shot.**  The array is insertable and its radial position is
set per shot, so self-calibration from the shot's own clean window is the
primary geometry source, not a fallback, and comparing fitted positions
between shots is meaningless because the probe really did move.  A configured
radius in ``vest.yaml`` applies only to an era where the array was surveyed
or left fixed, and then takes precedence.  A reference shot taken with the
array in the same position can supply the calibration for another, which is
the legacy two-shot arrangement; the single-shot path never requires it.

Notes
-----
**What the reference shots show.**  The 2022-04-23 alignment block, shots
35376 to 35379 and 35385, is the verified reference condition.  With that
campaign's cabling the array behaves exactly as a rigid seven-channel probe
at the as-built pitch facing the toroidal field: the fitted coupling is 1.01
to 1.045 on every channel, meaning the probes measure the toroidal field
itself rather than a small tilt pickup; the rigid fit lands at a major radius
of 0.486 to 0.501 m across the block with a 9 to 12 percent normalized
residual; and one field is not wired in this campaign, so the array runs
seven channels.

Because those probes face the toroidal field they have essentially no
sensitivity left for the poloidal component.  Such a probe cannot measure the
vertical field, and the measurement belongs with the toroidal probes instead.
The legacy ten-degree tilt model assumed a near-vertical probe; on shot 39204
it saturates at its bound on all eight channels and returns a compensated
vertical field of order 0.1 T, two orders of magnitude above VEST's real one.

Shots failing the reference condition are rejected with reasons: 39204 and
39923 leave a 34 to 35 percent rigid-array residual and carry channels far
from unit coupling, and 35325, a reference taken with the array withdrawn,
shows a coupling of about 0.004.

Provenance
----------
.. [1] Yang et al., Rev. Sci. Instrum. 85, 11D809 (2014), which establishes
   the roles of the two sensor sets.
.. [2] The legacy MATLAB scripts this module ports and deliberately differs
   from: ``VEST_IMPAProcessing.m`` used ``-2/15`` with ``+3e4`` and
   ``vest_impa_position.m`` used ``+2/15`` with ``-3e4``.  Both are
   reproduced faithfully by :func:`legacy_impa_compensation` and
   :func:`legacy_impa_position` respectively.
.. [3] The array's own DAQ wiring datasheet, the authoritative source for the
   Hall gain and the reason the vertical sensors stay in volts.
.. [4] ``vest.yaml`` ``impa``, which carries the shot-era geometry,
   calibration and thresholds this module takes as arguments.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage, optimize, signal

from vaft.formula.statistics import rms

__all__ = [
    "IMPA_CHANNEL_COUNT",
    "ImpaCouplingFit",
    "ImpaCrosstalkFit",
    "ImpaGeometryFit",
    "ImpaProcessingConfig",
    "ImpaQuality",
    "ImpaResult",
    "TfCalibrationWindow",
    "TfWindowCriteria",
    "find_tf_calibration_window",
    "fit_impa_crosstalk",
    "fit_impa_geometry",
    "fit_impa_tf_coupling",
    "impa_calibrate_signals",
    "impa_lowpass",
    "legacy_impa_compensation",
    "legacy_impa_position",
    "process_impa",
    "remove_bz_crosstalk",
    "remove_tf_pickup",
    "toroidal_field",
    "grade_impa_quality",
    "validate_impa",
]

MU0 = 4.0e-7 * math.pi
IMPA_CHANNEL_COUNT = 8

#: Legacy ``vest_filter(fs, fc, 1)`` is ``designfilt('lowpassfir',
#: 'FilterOrder', round(fs*1e-3), ...)`` -- a Hamming-window FIR of order 25,
#: i.e. 26 taps, applied with ``filtfilt``.
_FIR_TAPS = 26

_VALID_BASELINES = ("first_sample", "mean_first_samples", "none")
_VALID_ORIENTATIONS = ("toroidal", "poloidal")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ImpaProcessingConfig:
    """Signal-conditioning and TF-compensation knobs for the IMPA."""

    sample_rate: float = 25_000.0
    #: Cut-off for the physics waveform (legacy ``VEST_IMPAProcessing``).
    signal_lowpass_hz: float = 250.0
    #: Cut-off used by the legacy position fit (``vest_impa_position``).  Kept
    #: as a separate stage because the two legacy routines genuinely differ.
    position_lowpass_hz: float = 2_500.0
    #: Hall calibration factor [T/V] in the canonical VAFT sign convention.
    gain: float = 2.0 / 15.0
    baseline: str = "first_sample"
    baseline_samples: int = 2_500
    tf_turns: int = 24
    tilt_bounds_deg: tuple[float, float] = (-10.0, 10.0)
    #: How the array is mounted.  ``toroidal`` means the probes face the
    #: toroidal field (``alpha ~ 1``) and measure it directly, which is what
    #: the 2022-04-23 alignment shots show; ``poloidal`` is the legacy
    #: near-vertical model, where a small tilt admits a TF pickup that has to
    #: be removed to recover Bz.
    orientation: str = "toroidal"
    #: Tolerance on ``|alpha| - 1`` for a toroidally aligned array.
    alpha_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if self.orientation not in _VALID_ORIENTATIONS:
            raise ValueError(
                f"Unsupported IMPA orientation {self.orientation!r}; expected one of {_VALID_ORIENTATIONS}"
            )
        if self.baseline not in _VALID_BASELINES:
            raise ValueError(
                f"Unsupported IMPA baseline {self.baseline!r}; expected one of {_VALID_BASELINES}"
            )
        if self.sample_rate <= 0:
            raise ValueError("IMPA sample_rate must be positive")


@dataclass(frozen=True)
class TfWindowCriteria:
    """Signal-based conditions a clean TF calibration interval must satisfy."""

    #: Absolute floor: the TF must actually be energised.
    tf_current_min: float = 500.0
    #: ...and reach this fraction of the shot's own |I_TF| peak.  VEST runs
    #: legitimate low-TF shots (39923 peaks at 1.3 kA against 39204's 12.7 kA),
    #: so a fixed ampere threshold tuned on one shot silently rejects them;
    #: what matters is that the probes see a strong TF drive for that shot.
    tf_current_min_fraction: float = 0.5
    ip_max: float = 3_000.0
    pf_current_max: float = 500.0
    min_duration: float = 5.0e-3
    #: Relative TF spread ``(max-min)/max|I_TF|`` below which the interval is
    #: usable but reported as poorly conditioned for a slope fit.
    tf_dynamic_range_min: float = 0.02
    #: Per-channel sample-to-sample noise relative to the in-window spread.
    max_relative_noise: float = 0.5
    #: Median-filter width applied to |Ip| and the PF peak before thresholding.
    #: "No plasma" is a sustained condition, so isolated noise spikes must not
    #: fragment an otherwise clean interval.  25 samples is 1 ms at 25 kHz.
    smoothing_samples: int = 25


# --------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TfCalibrationWindow:
    """A contiguous same-shot interval judged to be TF-dominated."""

    start_time: float
    end_time: float
    indices: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(np.size(self.indices))

    @property
    def duration(self) -> float:
        return float(self.end_time - self.start_time)


@dataclass(frozen=True)
class ImpaCouplingFit:
    """Per-channel TF coupling from a multi-sample regression."""

    #: ``B_meas = alpha * Bt(R) + beta`` for the resolved radii.
    alpha: np.ndarray
    beta: np.ndarray
    tilt_deg: np.ndarray
    #: ``alpha / R`` -- the only quantity a TF-only window truly constrains.
    coupling_ratio: np.ndarray
    rmse: np.ndarray
    nrmse: np.ndarray
    r_squared: np.ndarray
    residual_trend: np.ndarray
    n_samples: int
    bound_hit: np.ndarray
    method: str = "linear_regression"


@dataclass(frozen=True)
class ImpaGeometryFit:
    """Resolved probe positions and how they were obtained."""

    r: np.ndarray
    z: np.ndarray
    method: str
    r0: float | None = None
    pitch: float | None = None
    #: The rigid, as-built channel spacing this array is known to have.
    nominal_pitch: float | None = None
    #: Insertion angle away from a purely radial midplane path, in degrees,
    #: implied by the fitted pitch falling short of the nominal one.
    incident_angle_deg: float | None = None
    rmse: float | None = None
    nrmse: float | None = None
    monotonic: bool = True
    within_bounds: bool = True
    bound_hit: bool = False
    n_samples: int = 0
    #: Channels that actually contributed; an era may not wire all of them.
    n_channels_fitted: int = 0


@dataclass(frozen=True)
class ImpaQuality:
    """Structured verdict; ``status`` is ``valid``/``warning``/``invalid``."""

    status: str
    checks: Mapping[str, str]
    reasons: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.status in ("valid", "warning")


@dataclass(frozen=True)
class ImpaResult:
    """Everything one IMPA shot produces, including its provenance."""

    time: np.ndarray
    b_measured: np.ndarray
    tf_pickup: np.ndarray
    #: For a poloidal array, the compensated vertical field.  For a toroidal
    #: array, the residual left after removing the modelled TF contribution --
    #: not a calibrated Bz, since such a probe cannot measure one.
    b_z: np.ndarray
    channel_valid: np.ndarray
    geometry: ImpaGeometryFit
    coupling: ImpaCouplingFit | None
    window: TfCalibrationWindow | None
    quality: ImpaQuality
    #: Bz sensor positions: Hall geometry.r plus the fixed hardware offset --
    #: the Bz sensor in one probe housing is not co-located with its Hall
    #: neighbour.
    bz_r: np.ndarray | None = None
    #: Raw vertical-field waveforms from the dedicated Bz sensors, if wired.
    b_z_raw: np.ndarray | None = None
    #: Those waveforms with the toroidal-field crosstalk removed.
    b_z_corrected: np.ndarray | None = None
    crosstalk: ImpaCrosstalkFit | None = None
    bz_channel_valid: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# signal conditioning
# --------------------------------------------------------------------------
def impa_lowpass(values: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Zero-phase low pass reproducing the legacy filter design.

    Parameters
    ----------
    values : array_like
        Waveform or stack of them; the sample axis is the last [any].
    cutoff_hz : float
        Cut-off frequency [Hz].
    sample_rate : float
        Sample rate [Hz].

    Returns
    -------
    np.ndarray
        The filtered waveform, same shape [any].

    Convention
    ----------
    **Zero phase where it can be.**  The forward-backward filter moves no feature
    in time, which matters because the array's channels are compared against each
    other and against the coil current.  A record too short for the padding that
    needs falls back to a single forward pass, which is causal and therefore
    delays every feature; the convention silently changes with record length, and
    only tiny synthetic records are short enough for it.

    Defaults
    --------
    The tap count is a legacy compatibility value: the donor design is a
    Hamming-window filter whose order follows from the sample rate, giving the
    same 26 taps this uses.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The fallback above is the one case where the phase convention is not what the
    summary says.

    Provenance
    ----------
    .. [1] The legacy ``vest_filter`` design, whose window, order and
       forward-backward application this reproduces.
    """
    values = np.asarray(values, dtype=float)
    taps = signal.firwin(_FIR_TAPS, cutoff_hz, pass_zero="lowpass", fs=sample_rate)
    if values.shape[-1] > 3 * (taps.size - 1):
        return signal.filtfilt(taps, 1, values, axis=-1)
    return signal.lfilter(taps, 1, values, axis=-1)


def _apply_baseline(values: np.ndarray, baseline: str, baseline_samples: int) -> np.ndarray:
    if baseline == "none":
        return values
    if baseline == "first_sample":
        return values - values[..., :1]
    count = max(1, min(int(baseline_samples), values.shape[-1]))
    return values - np.mean(values[..., :count], axis=-1, keepdims=True)


def impa_calibrate_signals(
    raw: np.ndarray,
    *,
    gain: float,
    cutoff_hz: float,
    sample_rate: float,
    baseline: str = "first_sample",
    baseline_samples: int = 2_500,
) -> np.ndarray:
    """Filter, scale and baseline raw probe records into field units.

    Parameters
    ----------
    raw : array_like
        Raw digitizer records as ``(n_channels, n_samples)`` [V].
    gain : float
        Hall calibration factor [T/V].
    cutoff_hz : float
        Conditioning low-pass cut-off [Hz].
    sample_rate : float
        Sample rate [Hz].
    baseline : str, optional
        Which baseline to remove [-].
    baseline_samples : int, optional
        Leading samples used when the baseline is a mean [-].

    Returns
    -------
    np.ndarray
        Conditioned records in field units [T].

    Processing steps
    ----------------
    1. Low-pass each channel.
    2. Multiply by the calibration factor.
    3. Remove the baseline.

    Convention
    ----------
    **The stage order is the donor's**, low pass then gain then baseline, and it
    is not interchangeable: removing a baseline before filtering leaves the
    filter's transient riding on a different level.

    The sign of the calibration factor is the caller's, and the fits downstream
    are deliberately sign-agnostic; see the module's conventions and #624.

    Defaults
    --------
    ``baseline_samples = 2500`` is an acquisition-era value, the leading interval
    that precedes the discharge at the array's own sample rate. It is a sample
    count, so it means a different duration at a different rate.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes the leading window is field-free. Inherits the record-length caveat
    of :func:`impa_lowpass`.

    Provenance
    ----------
    .. [1] ``VEST_IMPAProcessing.m``, whose stage order this follows.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    filtered = impa_lowpass(raw, cutoff_hz, sample_rate) * float(gain)
    return _apply_baseline(filtered, baseline, baseline_samples)


def toroidal_field(r: np.ndarray | float, i_tf: np.ndarray | float, turns: int = 24) -> np.ndarray:
    """Vacuum toroidal field at a radius, from the coil current.

    Parameters
    ----------
    r : array_like
        Major radius, strictly positive [m].
    i_tf : array_like
        Toroidal field coil current [A].
    turns : int, optional
        Coil turns [-].

    Returns
    -------
    np.ndarray
        The vacuum toroidal field [T].

    Convention
    ----------
    ``mu0 * N * I / (2 * pi * R)``, the vacuum field of an ideal toroidal
    solenoid, so it carries the sign of the current as given. It is the field the
    array is calibrated against, not a measurement.

    Defaults
    --------
    ``turns = 24`` is machine-specific, the VEST toroidal field coil's turn count.
    A caller on another machine passes its own.

    Applicability
    -------------
    Machine-independent.  Only the default turn count is a VEST number.

    Limitations
    -----------
    Vacuum field only: it ignores the plasma's own diamagnetic or paramagnetic
    contribution, which is the assumption that makes a coil-driven interval usable
    for calibration in the first place.

    Provenance
    ----------
    .. [1] The ideal toroidal solenoid field; the same relation
       :mod:`vaft.machine_mapping.tf` uses for the coil current.
    """
    r_array = np.asarray(r, dtype=float)
    if np.any(r_array <= 0):
        raise ValueError("IMPA probe radii must be positive")
    return MU0 * int(turns) * np.asarray(i_tf, dtype=float) / (2.0 * math.pi * r_array)


# --------------------------------------------------------------------------
# clean-TF window selection
# --------------------------------------------------------------------------
def _bound_edge(lower: float, upper: float) -> float:
    """Tolerance for calling a bounded fit result "saturated at the bound"."""
    return max((upper - lower) * 1e-4, 1e-9)


def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1))


def _sustained(values: np.ndarray, width: int) -> np.ndarray:
    """Median-filter a magnitude so isolated spikes do not veto an interval."""
    width = int(max(1, width))
    if width <= 1 or values.size <= width:
        return values
    return ndimage.median_filter(values, size=width, mode="nearest")


def find_tf_calibration_window(
    time: np.ndarray,
    i_tf: np.ndarray,
    ip: np.ndarray | None = None,
    pf_currents: np.ndarray | None = None,
    *,
    criteria: TfWindowCriteria | None = None,
    b_measured: np.ndarray | None = None,
) -> tuple[TfCalibrationWindow | None, tuple[str, ...]]:
    """Find the interval where the toroidal field alone drives the probes.

    Calibration needs a stretch of the same shot in which the coil is energised
    and nothing else is: no plasma current, no poloidal field. That interval is
    what makes the geometry and coupling fits meaningful, so this reports why it
    chose the one it did.

    Parameters
    ----------
    time : array_like
        Sample times [s].
    i_tf : array_like
        Toroidal field coil current [A].
    ip : array_like, optional
        Plasma current, used to exclude the discharge [A].
    pf_currents : array_like, optional
        Poloidal field coil currents as ``(n_coils, n_samples)`` [A].
    criteria : TfWindowCriteria, optional
        Thresholds the interval must satisfy [-].
    b_measured : array_like, optional
        Conditioned probe records, used to require finite channels [T].

    Returns
    -------
    tuple
        The chosen :class:`TfCalibrationWindow` with its metrics, and a tuple of
        reason strings recording what was applied and why [-].

    Processing steps
    ----------------
    1. Validate the time axis.
    2. Require the coil current above an absolute floor and above a fraction of
       the shot's own peak.
    3. Exclude samples where the plasma current or any poloidal coil exceeds its
       limit, on median-filtered traces.
    4. Require every probe channel finite.
    5. Group what survives into contiguous runs and drop the too-short ones.
    6. Choose among the survivors by **conditioning, not length**.

    Convention
    ----------
    The chosen interval is the best-conditioned one, meaning the widest spread in
    coil current, rather than the longest. A long interval at nearly constant
    current constrains the fit poorly however many samples it holds, because the
    fit is a regression against that current.

    Both an absolute floor and a fraction of the shot's own peak are applied,
    because a shot can be energised weakly overall while still having a usable
    interval, and a fixed floor alone would either reject it or admit noise on a
    strong shot.

    Defaults
    --------
    The thresholds are hard-coded: an absolute current floor, a fraction of peak,
    plasma and poloidal current limits, a minimum duration and a dynamic-range
    floor. The fraction of peak is a validated-workflow value distinguishing the
    reference shots from a weakly energised one; the rest have no recorded
    derivation. The median-filter width is an acquisition-era value, one
    millisecond at the array's own rate. One configured threshold, the relative
    noise limit, is read by nothing at all, which is tracked in #625.

    Applicability
    -------------
    Machine-independent.  The thresholds arrive as an argument.

    Limitations
    -----------
    Reasons are returned rather than raised, so a caller must read them: a window
    can be chosen and still be poor. Without the optional currents the
    corresponding exclusions simply do not happen, and an interval containing
    plasma can be selected.

    Provenance
    ----------
    .. [1] ``vest.yaml`` ``impa.calibration_window``, which carries the VEST
       thresholds and the evidence for the fraction-of-peak value.
    """
    criteria = criteria or TfWindowCriteria()
    time = np.asarray(time, dtype=float)
    i_tf = np.asarray(i_tf, dtype=float)
    reasons: list[str] = []

    if time.size < 2 or time.size != i_tf.size:
        return None, ("time and TF current axes are missing or mismatched",)
    if not np.all(np.diff(time) > 0):
        return None, ("time axis is not strictly increasing",)

    tf_peak = float(np.nanmax(np.abs(i_tf))) if i_tf.size else 0.0
    tf_threshold = max(criteria.tf_current_min, criteria.tf_current_min_fraction * tf_peak)
    mask = np.isfinite(i_tf) & (np.abs(i_tf) >= tf_threshold)
    if not mask.any():
        reasons.append(
            f"no sample reaches |I_TF| >= {tf_threshold:g} A (peak {tf_peak:.0f} A)"
        )
        return None, tuple(reasons)

    if ip is not None:
        ip = np.asarray(ip, dtype=float)
        if ip.size == time.size:
            ip_ok = np.isfinite(ip) & (
                _sustained(np.abs(ip), criteria.smoothing_samples) <= criteria.ip_max
            )
            if not (mask & ip_ok).any():
                reasons.append(f"|Ip| never falls below {criteria.ip_max:g} A while the TF is on")
            mask &= ip_ok
        else:
            reasons.append("plasma-current axis length does not match the IMPA time axis; Ip criterion skipped")

    if pf_currents is not None:
        pf_currents = np.atleast_2d(np.asarray(pf_currents, dtype=float))
        if pf_currents.shape[-1] == time.size:
            pf_peak = np.nanmax(np.abs(pf_currents), axis=0)
            pf_ok = np.isfinite(pf_peak) & (
                _sustained(pf_peak, criteria.smoothing_samples) <= criteria.pf_current_max
            )
            if not (mask & pf_ok).any():
                reasons.append(f"PF currents never fall below {criteria.pf_current_max:g} A while the TF is on")
            mask &= pf_ok
        else:
            reasons.append("PF current axis length does not match the IMPA time axis; PF criterion skipped")

    if b_measured is not None:
        finite_channels = np.all(np.isfinite(np.atleast_2d(b_measured)), axis=0)
        if finite_channels.size == time.size:
            mask &= finite_channels

    runs = _contiguous_runs(mask)
    if not runs:
        reasons.append("no contiguous interval satisfies the TF / Ip / PF criteria simultaneously")
        return None, tuple(reasons)

    longest = float(time[max(runs, key=len)[-1]] - time[max(runs, key=len)[0]])
    candidates = [run for run in runs if float(time[run[-1]] - time[run[0]]) >= criteria.min_duration]
    if not candidates:
        reasons.append(
            f"longest clean interval is {longest * 1e3:.2f} ms, shorter than the "
            f"required {criteria.min_duration * 1e3:.2f} ms"
        )
        return None, tuple(reasons)

    # Separating a probe's TF coupling from its offset needs the TF to actually
    # move, so prefer the best-conditioned interval rather than the longest one.
    def conditioning(run: np.ndarray) -> tuple[float, int]:
        values = i_tf[run]
        peak = float(np.max(np.abs(values)))
        spread = float(np.max(values) - np.min(values)) / peak if peak else 0.0
        return spread, int(run.size)

    run = max(candidates, key=conditioning)
    duration = float(time[run[-1]] - time[run[0]])

    tf_window = i_tf[run]
    peak = float(np.max(np.abs(tf_window)))
    dynamic_range = float((np.max(tf_window) - np.min(tf_window)) / peak) if peak else 0.0
    metrics = {
        "candidate_intervals": float(len(candidates)),
        "tf_current_threshold": float(tf_threshold),
        "tf_current_mean": float(np.mean(tf_window)),
        "tf_current_peak": peak,
        "tf_dynamic_range": dynamic_range,
        "duration": duration,
    }
    if ip is not None and np.size(ip) == time.size:
        metrics["ip_peak"] = float(np.max(np.abs(ip[run])))
    if pf_currents is not None and np.shape(pf_currents)[-1] == time.size:
        metrics["pf_current_peak"] = float(np.max(np.abs(pf_currents[:, run])))
    if dynamic_range < criteria.tf_dynamic_range_min:
        reasons.append(
            f"TF dynamic range in the window is {dynamic_range * 100:.1f}%, below the "
            f"{criteria.tf_dynamic_range_min * 100:.1f}% preferred for a slope fit"
        )

    window = TfCalibrationWindow(
        start_time=float(time[run[0]]),
        end_time=float(time[run[-1]]),
        indices=run,
        metrics=metrics,
    )
    return window, tuple(reasons)


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------
def fit_impa_tf_coupling(
    b_measured: np.ndarray,
    i_tf: np.ndarray,
    r: np.ndarray,
    window: TfCalibrationWindow,
    *,
    turns: int = 24,
    tilt_bounds_deg: Sequence[float] = (-10.0, 10.0),
) -> ImpaCouplingFit:
    """Fit each probe's coupling to the toroidal field, at known radii.

    Parameters
    ----------
    b_measured : array_like
        Conditioned probe records as ``(n_channels, n_samples)`` [T].
    i_tf : array_like
        Toroidal field coil current [A].
    r : array_like
        Major radius of each probe [m].
    window : TfCalibrationWindow
        The interval to regress over [-].
    turns : int, optional
        Coil turns [-].
    tilt_bounds_deg : tuple of float, optional
        Bounds on the probe tilt [deg].

    Returns
    -------
    ImpaCouplingFit
        Per-channel coupling and offset with their residual statistics [-].

    Convention
    ----------
    Regresses the measured field on the modelled toroidal field at the known
    radius, giving the coupling as slope and the offset as intercept. This is the
    half of the degeneracy that a **known radius** resolves; see
    :func:`fit_impa_geometry` for the other half. Only the magnitude of the
    coupling carries meaning, since its sign follows the calibration polarity.

    Defaults
    --------
    ``turns = 24`` is machine-specific. The tilt bounds are a legacy compatibility
    value from a model that assumed a near-vertical probe, and they saturate on an
    array that faces the toroidal field, which is the reference condition here.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Needs the radii, which on this array are usually themselves fitted, so the two
    must not be circular. The tilt bounds saturating is a signal that the model
    does not fit the array, not a result to use.

    Provenance
    ----------
    .. [1] The measurement model in this module's conventions.
    """
    b_measured = np.atleast_2d(np.asarray(b_measured, dtype=float))
    i_tf = np.asarray(i_tf, dtype=float)
    r = np.asarray(r, dtype=float)
    idx = np.asarray(window.indices, dtype=int)
    n_channels = b_measured.shape[0]

    alpha = np.full(n_channels, np.nan)
    beta = np.full(n_channels, np.nan)
    rmse = np.full(n_channels, np.nan)
    nrmse = np.full(n_channels, np.nan)
    r_squared = np.full(n_channels, np.nan)
    trend = np.full(n_channels, np.nan)

    lower, upper = float(min(tilt_bounds_deg)), float(max(tilt_bounds_deg))
    alpha_bounds = (math.sin(math.radians(lower)), math.sin(math.radians(upper)))

    for channel in range(n_channels):
        y = b_measured[channel, idx]
        x = toroidal_field(r[channel], i_tf[idx], turns)
        if y.size < 2 or not np.all(np.isfinite(y)) or np.allclose(x, x[0]):
            # A constant TF drive cannot separate slope from offset.
            continue
        design = np.vstack([x, np.ones_like(x)]).T
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        alpha[channel], beta[channel] = float(solution[0]), float(solution[1])
        residual = y - design @ solution
        rmse[channel] = rms(residual)
        # Normalized by the spread of the measurement, not by its RMS: the
        # offset beta is fitted, so only the varying part of y was ever signal
        # the model had to explain.
        spread = float(np.std(y))
        nrmse[channel] = rmse[channel] / spread if spread > 0 else np.inf
        total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared[channel] = 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan
        # A sloped residual means the linear TF model is missing something.
        trend[channel] = float(np.polyfit(np.arange(residual.size), residual, 1)[0] * residual.size)

    with np.errstate(invalid="ignore"):
        tilt = np.degrees(np.arcsin(np.clip(alpha, -1.0, 1.0)))
        coupling_ratio = alpha / r
        bound_hit = (alpha < alpha_bounds[0]) | (alpha > alpha_bounds[1])

    return ImpaCouplingFit(
        alpha=alpha,
        beta=beta,
        tilt_deg=tilt,
        coupling_ratio=coupling_ratio,
        rmse=rmse,
        nrmse=nrmse,
        r_squared=r_squared,
        residual_trend=trend,
        n_samples=int(idx.size),
        bound_hit=bound_hit,
    )


def fit_impa_geometry(
    b_measured: np.ndarray,
    i_tf: np.ndarray,
    window: TfCalibrationWindow,
    *,
    pitch: float = 0.05,
    z: float = 0.0,
    turns: int = 24,
    r0_initial: float = 0.4,
    r_bounds: Sequence[float] = (0.1, 0.9),
    fit_pitch: bool = False,
) -> ImpaGeometryFit:
    """Self-calibrate the array's radial position from the toroidal-field profile.

    Parameters
    ----------
    b_measured : array_like
        Conditioned probe records as ``(n_channels, n_samples)`` [T].
    i_tf : array_like
        Toroidal field coil current [A].
    window : TfCalibrationWindow
        The interval to fit over [-].
    pitch : float, optional
        As-built spacing between channels [m].
    z : float, optional
        Height of the array [m].
    turns : int, optional
        Coil turns [-].
    r0_initial : float, optional
        Starting guess for the innermost channel's radius [m].
    r_bounds : tuple of float, optional
        Bounds on that radius [m].
    fit_pitch : bool, optional
        Whether to fit the spacing as well as the position [-].

    Returns
    -------
    ImpaGeometryFit
        The fitted radii, the spacing, the incident angle and the residual
        statistics [-].

    Processing steps
    ----------------
    1. Keep the channels finite across the window.
    2. Detect the working polarity from the data.
    3. Solve a bounded least squares for the innermost radius, and the spacing
       when asked.
    4. Form the residual and its normalized form.
    5. Recover the incident angle from the ratio of fitted to as-built spacing.

    Convention
    ----------
    This is the half of the degeneracy that **assuming unit coupling** resolves,
    which is the legacy method: a rigid array of known spacing is slid along the
    one-over-radius profile until it matches. The other half is
    :func:`fit_impa_tf_coupling`.

    **The coupling sign is detected, not assumed.**  The measured sign follows the
    configured Hall gain's polarity, which this function has no reason to know.
    An incident angle of zero is a purely radial insertion in the midplane.

    Defaults
    --------
    The spacing is machine-specific, the array's rigid as-built value. The
    starting guess and the radial bounds are machine-specific too, spanning the
    vessel. ``turns = 24`` is the VEST coil.

    Applicability
    -------------
    Machine-independent.  Every machine number is an argument.

    Limitations
    -----------
    Assumes the array is rigid and the coupling uniform, so a fitted position is
    meaningful only where the residual is small; the reference shots sit at 9 to
    12 percent and the rejected ones at 34 to 35. **Positions are not comparable
    between shots**, because the array is insertable and really is repositioned.

    Provenance
    ----------
    .. [1] The ``vest_impa_position`` model: uniform channel spacing and unit
       coupling, reproduced faithfully by :func:`legacy_impa_position`.
    """
    b_measured = np.atleast_2d(np.asarray(b_measured, dtype=float))
    i_tf = np.asarray(i_tf, dtype=float)
    idx = np.asarray(window.indices, dtype=int)
    n_channels = b_measured.shape[0]
    offsets = np.arange(n_channels) * float(pitch)
    observed = b_measured[:, idx]
    lower, upper = float(min(r_bounds)), float(max(r_bounds))

    # A channel that is unwired for this shot era, or otherwise unavailable,
    # carries NaN.  Fit the channels that are present and still report a radius
    # for every position, since the array is rigid and the pitch is known.
    usable = np.all(np.isfinite(observed), axis=1)
    if not usable.any():
        raise ValueError("No IMPA channel has finite samples inside the calibration window")
    fitted = observed[usable]
    fitted_offsets = offsets[usable]

    # The array is rigid, so the physical spacing is fixed; a probe inserted at
    # an angle to the midplane projects it onto a shorter radial step, and the
    # ratio of the fitted to the nominal pitch recovers that incident angle.
    positions = np.arange(n_channels, dtype=float)
    fitted_positions = positions[usable]

    # This fit assumes every probe is fully aligned with the toroidal field
    # (|alpha| = 1), matching the legacy vest_impa_position.m model, but the
    # measured *sign* depends on the configured Hall gain's polarity, which
    # this function has no reason to know.  Detect it once from the data
    # instead of hard-coding a positive coupling.
    nominal_radii = float(r0_initial) + fitted_positions * float(pitch)
    nominal_model = toroidal_field(nominal_radii[:, None], i_tf[None, idx], turns)
    sign = 1.0 if float(np.sum(fitted * nominal_model)) >= 0 else -1.0

    def residual(params: np.ndarray) -> np.ndarray:
        step = params[1] if fit_pitch else float(pitch)
        radii = params[0] + fitted_positions * step
        model = toroidal_field(radii[:, None], i_tf[None, idx], turns)
        return (fitted - sign * model).ravel()

    if fit_pitch:
        solution = optimize.least_squares(
            residual,
            [float(np.clip(r0_initial, lower, upper)), float(pitch)],
            bounds=([lower, 1e-3], [upper, float(pitch)]),
        )
        fitted_pitch = float(solution.x[1])
    else:
        solution = optimize.least_squares(
            residual,
            [float(np.clip(r0_initial, lower, upper)), float(pitch)],
            bounds=([lower, float(pitch) - 1e-9], [upper, float(pitch) + 1e-9]),
        )
        fitted_pitch = float(pitch)
    r0 = float(solution.x[0])
    final = residual(solution.x)
    rmse = rms(final)
    spread = float(np.std(fitted))
    radii = r0 + positions * fitted_pitch
    # arccos of the projected-to-physical pitch ratio; 0 deg is a purely radial
    # insertion in the midplane.
    ratio = float(np.clip(fitted_pitch / float(pitch), -1.0, 1.0)) if pitch else 1.0
    incident_angle = float(math.degrees(math.acos(ratio)))

    return ImpaGeometryFit(
        r=radii,
        z=np.full(n_channels, float(z)),
        method="tf_profile_fit",
        r0=r0,
        pitch=fitted_pitch,
        nominal_pitch=float(pitch),
        incident_angle_deg=incident_angle,
        rmse=rmse,
        nrmse=rmse / spread if spread > 0 else float("inf"),
        monotonic=bool(np.all(np.diff(radii) > 0)),
        within_bounds=bool(lower <= r0 <= upper),
        bound_hit=bool((r0 - lower) <= _bound_edge(lower, upper) or (upper - r0) <= _bound_edge(lower, upper)),
        n_samples=int(idx.size),
        n_channels_fitted=int(np.count_nonzero(usable)),
    )


@dataclass(frozen=True)
class ImpaCrosstalkFit:
    """Toroidal-field bleed into each vertical-field sensor.

    Following Yang et al., Rev. Sci. Instrum. 85, 11D809 (2014): during a
    TF-only interval the true vertical field is negligible, so whatever the Bz
    sensor reads there is crosstalk from the toroidal field.

    ``b_z_raw`` is native Hall-sensor volts and ``b_toroidal`` is tesla, so the
    fitted slope is in **volts per tesla**, not a dimensionless projection --
    turning it into ``sin(angle)`` needs the Bz sensor's own volts-to-tesla
    gain, which the DAQ wiring datasheet does not specify.  ``sin_angle`` and
    ``angle_deg`` are populated only when a ``bz_gain`` is supplied; otherwise
    they are NaN rather than silently wrong.  Crosstalk *removal* needs no
    gain at all -- it subtracts ``slope * b_toroidal`` in the same raw volts
    the sensor already reports.
    """

    #: ``B_z_raw[V] = slope_v_per_t[V/T] * B_toroidal[T] + offset[V]``.
    slope_v_per_t: np.ndarray
    offset: np.ndarray
    r_squared: np.ndarray
    nrmse: np.ndarray
    n_samples: int
    #: Only finite where ``bz_gain`` was supplied to :func:`fit_impa_crosstalk`.
    sin_angle: np.ndarray
    angle_deg: np.ndarray
    bound_hit: np.ndarray


def fit_impa_crosstalk(
    b_toroidal: np.ndarray,
    b_z_raw: np.ndarray,
    window: TfCalibrationWindow,
    *,
    max_angle_deg: float = 30.0,
    bz_gain: float | None = None,
) -> ImpaCrosstalkFit:
    """Fit the toroidal-field bleed into each vertical sensor.

    On an interval driven by the toroidal field alone the true vertical field is
    negligible, so whatever a vertical sensor reads there is crosstalk. Its ratio
    to the measured toroidal field is that sensor's misalignment.

    Parameters
    ----------
    b_toroidal : array_like
        Toroidal field at each sensor [T].
    b_z_raw : array_like
        Raw vertical-sensor records, in native units [V].
    window : TfCalibrationWindow
        The toroidal-field-only interval [-].
    max_angle_deg : float, optional
        Largest misalignment still treated as physical [deg].
    bz_gain : float, optional
        Volts-to-tesla gain for the vertical sensors, if known [T/V].

    Returns
    -------
    ImpaCrosstalkFit
        Per-sensor slope, offset and goodness of fit; the misalignment angle only
        when a gain was supplied [-].

    Convention
    ----------
    **The slope is volts per tesla, not a dimensionless projection**, because the
    wiring datasheet leaves the vertical sensors' gain unspecified and those
    waveforms stay in volts. Without a gain the angle cannot be formed and is
    returned as NaN **rather than silently wrong**.

    Defaults
    --------
    The maximum angle is hard-coded with no recorded derivation.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Rests on the vertical field being negligible over the window, which is what
    makes the window selection load-bearing. A linear fit only: a sensor
    responding non-linearly keeps a residual.

    Provenance
    ----------
    .. [1] Yang et al., Rev. Sci. Instrum. 85, 11D809 (2014), for the alignment
       role of the two sensor sets; ``vest.yaml`` ``impa.crosstalk`` for the
       configured thresholds.
    """
    b_toroidal = np.atleast_2d(np.asarray(b_toroidal, dtype=float))
    b_z_raw = np.atleast_2d(np.asarray(b_z_raw, dtype=float))
    idx = np.asarray(window.indices, dtype=int)
    n_channels = b_z_raw.shape[0]

    slope = np.full(n_channels, np.nan)
    offset = np.full(n_channels, np.nan)
    r_squared = np.full(n_channels, np.nan)
    nrmse = np.full(n_channels, np.nan)

    for channel in range(n_channels):
        if channel >= b_toroidal.shape[0]:
            break
        x = b_toroidal[channel, idx]
        y = b_z_raw[channel, idx]
        if y.size < 2 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)) or np.allclose(x, x[0]):
            continue
        design = np.vstack([x, np.ones_like(x)]).T
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        slope[channel], offset[channel] = float(solution[0]), float(solution[1])
        residual = y - design @ solution
        # Spread-normalized, as in fit_impa_tf_coupling: the fitted offset
        # means only the varying part of y was ever signal.
        spread = float(np.std(y))
        nrmse[channel] = rms(residual) / spread if spread > 0 else np.inf
        total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared[channel] = 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan

    if bz_gain is not None:
        with np.errstate(invalid="ignore"):
            sin_angle = np.clip(slope * float(bz_gain), -1.0, 1.0)
            angle = np.degrees(np.arcsin(sin_angle))
            bound_hit = np.abs(angle) > float(max_angle_deg)
    else:
        sin_angle = np.full(n_channels, np.nan)
        angle = np.full(n_channels, np.nan)
        # No angle bound is checkable without a gain; a sensor that shows no
        # crosstalk at all is still caught by the r-squared threshold.
        bound_hit = np.zeros(n_channels, dtype=bool)

    return ImpaCrosstalkFit(
        slope_v_per_t=slope,
        offset=offset,
        r_squared=r_squared,
        nrmse=nrmse,
        n_samples=int(idx.size),
        sin_angle=sin_angle,
        angle_deg=angle,
        bound_hit=bound_hit,
    )


def remove_bz_crosstalk(
    b_z_raw: np.ndarray,
    b_toroidal: np.ndarray,
    crosstalk: ImpaCrosstalkFit,
) -> np.ndarray:
    """Subtract a vertical sensor's toroidal crosstalk.

    Parameters
    ----------
    b_z_raw : array_like
        Raw vertical-sensor records, in the sensor's native units [V].
    b_toroidal : array_like
        The toroidal field at each sensor [T].
    crosstalk : ImpaCrosstalkFit
        The fitted per-sensor slope and offset [-].

    Returns
    -------
    np.ndarray
        The records with the crosstalk removed, still in native units [V].

    Convention
    ----------
    **The output stays in volts.**  The wiring datasheet leaves the vertical
    sensors' volts-to-tesla gain unspecified, so nothing here converts them; the
    fitted slope is therefore volts per tesla rather than a dimensionless
    projection.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Removes only the linear crosstalk the fit describes. A sensor whose response
    to the toroidal field is not linear keeps its residual.

    Provenance
    ----------
    .. [1] :func:`fit_impa_crosstalk`, which supplies the coefficients, and the
       wiring datasheet that leaves the vertical gain unspecified.
    """
    b_z_raw = np.atleast_2d(np.asarray(b_z_raw, dtype=float))
    b_toroidal = np.atleast_2d(np.asarray(b_toroidal, dtype=float))
    channels = min(b_z_raw.shape[0], b_toroidal.shape[0])
    corrected = np.full_like(b_z_raw, np.nan)
    for channel in range(channels):
        slope = crosstalk.slope_v_per_t[channel]
        if not np.isfinite(slope):
            continue
        corrected[channel] = (
            b_z_raw[channel] - slope * b_toroidal[channel] - crosstalk.offset[channel]
        )
    return corrected


def remove_tf_pickup(b_measured: np.ndarray, tf_pickup: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Recover the vertical field by removing a probe's toroidal pickup.

    Parameters
    ----------
    b_measured : array_like
        What the probe measured [T].
    tf_pickup : array_like
        The toroidal field at that probe [T].
    alpha : array_like
        The probe's coupling to the toroidal field [-].

    Returns
    -------
    np.ndarray
        The compensated vertical field, NaN where the probe is toroidally
        aligned [T].

    Convention
    ----------
    Inverts the measurement model: subtract the coupled toroidal part, then
    divide by the remaining poloidal sensitivity. A probe with unit coupling
    faces the toroidal field and has **no** sensitivity left for the vertical
    one, so the result is NaN rather than a division blowing up. That is not an
    edge case on this array: the reference shots put the coupling at 1.01 to
    1.045, which is exactly that condition.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The compensated field is only as good as the coupling it was given, and the
    division amplifies its error as the coupling approaches one. Trust the
    result only where the array is genuinely near-vertical.

    Provenance
    ----------
    .. [1] The measurement model in this module's conventions, of which this is
       the inversion.
    """
    b_measured = np.atleast_2d(np.asarray(b_measured, dtype=float))
    tf_pickup = np.atleast_2d(np.asarray(tf_pickup, dtype=float))
    alpha = np.asarray(alpha, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        projection = np.sqrt(1.0 - np.clip(alpha, -1.0, 1.0) ** 2)
        projection = np.where(projection > 1e-3, projection, np.nan)
        return (b_measured - tf_pickup) / projection[:, None]


# --------------------------------------------------------------------------
# legacy ports (parity references, single hard-coded time sample)
# --------------------------------------------------------------------------
def legacy_impa_position(
    time: np.ndarray,
    raw: np.ndarray,
    tf_raw: np.ndarray,
    *,
    target_time: float = 0.30,
    sample_rate: float = 25_000.0,
    cutoff_hz: float = 2_500.0,
    gain: float = 2.0 / 15.0,
    tf_gain: float = -3.0e4,
    turns: int = 24,
    pitch: float = 0.05,
    r0_initial: float = 0.4,
    r_bounds: Sequence[float] = (0.1, 0.9),
    baseline_samples: int = 2_500,
) -> dict[str, Any]:
    """Faithful port of the legacy position script, at one time sample.

    Kept so the current chain can be compared against what the donor actually
    did, not so it can be used in preference; :func:`fit_impa_geometry` is the
    maintained path.

    Parameters
    ----------
    time : array_like
        Sample times [s].
    raw : array_like
        Raw probe records as ``(n_channels, n_samples)`` [V].
    tf_raw : array_like
        Raw toroidal-field coil trace [V].
    target_time : float, optional
        The single sample the fit is evaluated at [s].
    sample_rate : float, optional
        Sample rate [Hz].
    cutoff_hz : float, optional
        Conditioning low-pass cut-off [Hz].
    gain : float, optional
        Hall calibration factor [T/V].
    tf_gain : float, optional
        Raw-to-amperes factor for the coil trace [A/V].
    turns : int, optional
        Coil turns [-].
    pitch : float, optional
        As-built channel spacing [m].
    r0_initial : float, optional
        Starting guess for the innermost radius [m].
    r_bounds : tuple of float, optional
        Bounds on that radius [m].
    baseline_samples : int, optional
        Leading samples forming the baseline [-].

    Returns
    -------
    ImpaGeometryFit
        The fitted radii and spacing at that one sample [-].

    Processing steps
    ----------------
    1. Subtract each probe's first sample, apply the gain, low-pass, then
       subtract the mean of the leading samples.
    2. Condition the coil trace with gain and low pass, **without** removing any
       offset.
    3. Take the sample nearest the target time.
    4. Solve the bounded least squares for the innermost radius.

    Convention
    ----------
    **The donor's order, kept deliberately**, including the asymmetry that the
    coil trace is not offset-corrected while the probes are. The sign pairing is
    the donor's too, a positive gain with a negative coil factor, which differs
    from both the datasheet and the other legacy script; see the module's
    conventions.

    Defaults
    --------
    Every default here is a legacy compatibility value reproducing the donor
    script: its single evaluation time, its sample rate, its cut-off, its two
    gains and its baseline length.

    Applicability
    -------------
    VEST-specific.  A port of a VEST script, carrying that machine's coil factor,
    sample rate and array spacing as its defaults; it reproduces one historical
    routine rather than describing a general method.

    Limitations
    -----------
    One time sample, not a time series, and no residual quality reported. It
    assumes unit coupling, which the reference shots show is nearly true and the
    rejected shots show is not.

    Provenance
    ----------
    .. [1] ``vest_impa_position.m``, ported faithfully including its stage order
       and its sign pairing.
    """
    time = np.asarray(time, dtype=float)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    hall = (raw - raw[:, :1]) * float(gain)
    hall = impa_lowpass(hall, cutoff_hz, sample_rate)
    hall = hall - np.mean(hall[:, : max(1, min(baseline_samples, hall.shape[1]))], axis=1, keepdims=True)

    i_tf = impa_lowpass(np.asarray(tf_raw, dtype=float) * float(tf_gain), cutoff_hz, sample_rate)

    index = int(np.argmin(np.abs(time - float(target_time))))
    measured = hall[:, index]
    offsets = np.arange(hall.shape[0]) * float(pitch)
    lower, upper = float(min(r_bounds)), float(max(r_bounds))

    def residual(params: np.ndarray) -> np.ndarray:
        return measured - toroidal_field(params[0] + offsets, i_tf[index], turns)

    solution = optimize.least_squares(
        residual, [float(np.clip(r0_initial, lower, upper))], bounds=([lower], [upper])
    )
    r0 = float(solution.x[0])
    final = residual(solution.x)
    edge = max((upper - lower) * 1e-4, 1e-9)
    return {
        "r0": r0,
        "r": r0 + offsets,
        "index": index,
        "time": float(time[index]),
        "tf_current": float(i_tf[index]),
        "measured": measured,
        "residual": final,
        "rmse": rms(final),
        "bound_hit": bool((r0 - lower) <= edge or (upper - r0) <= edge),
    }


def legacy_impa_compensation(
    time: np.ndarray,
    raw: np.ndarray,
    tf_raw: np.ndarray,
    r: np.ndarray,
    *,
    target_time: float = 0.29,
    sample_rate: float = 25_000.0,
    cutoff_hz: float = 250.0,
    gain: float = -2.0 / 15.0,
    tf_gain: float = 3.0e4,
    turns: int = 24,
    tilt_bounds_deg: Sequence[float] = (-10.0, 10.0),
) -> dict[str, Any]:
    """Faithful port of the legacy compensation script, at one time sample.

    Kept for comparison against the donor, not as the maintained path; the
    current chain is :func:`fit_impa_tf_coupling` with :func:`remove_tf_pickup`.

    Parameters
    ----------
    time : array_like
        Sample times [s].
    raw : array_like
        Raw probe records as ``(n_channels, n_samples)`` [V].
    tf_raw : array_like
        Raw toroidal-field coil trace [V].
    r : array_like
        Major radius of each probe [m].
    target_time : float, optional
        The single sample the fit is evaluated at [s].
    sample_rate : float, optional
        Sample rate [Hz].
    cutoff_hz : float, optional
        Conditioning low-pass cut-off [Hz].
    gain : float, optional
        Hall calibration factor [T/V].
    tf_gain : float, optional
        Raw-to-amperes factor for the coil trace [A/V].
    turns : int, optional
        Coil turns [-].
    tilt_bounds_deg : tuple of float, optional
        Bounds on the probe tilt [deg].

    Returns
    -------
    ImpaCouplingFit
        The per-channel coupling and the compensated field at that sample [-].

    Processing steps
    ----------------
    1. Low-pass each probe, apply the gain, subtract the first sample.
    2. Low-pass the coil trace and subtract its first sample.
    3. Take the sample nearest the target time.
    4. Solve a bounded tilt per channel, build the pickup, and divide it out.

    Convention
    ----------
    **The donor's order, which differs from the position script's**: here the low
    pass comes before the gain, and both traces are offset-corrected. The sign
    pairing is also the donor's, a negative gain with a positive coil factor,
    the mirror of the other script. Both are reproduced rather than harmonized.

    Defaults
    --------
    Every default is a legacy compatibility value from the donor script,
    including its evaluation time, which differs from the position script's.

    Applicability
    -------------
    VEST-specific.  A port of a VEST script carrying that machine's coil factor,
    sample rate and calibration as defaults, reproducing one historical routine
    rather than a general method.

    Limitations
    -----------
    One time sample only. The donor also processed a reference shot through the
    same path; that branch is deliberately absent here, so this is the
    single-shot half of the original. The tilt model assumes a near-vertical
    probe and saturates on an array that faces the toroidal field.

    Provenance
    ----------
    .. [1] ``VEST_IMPAProcessing.m``, ported faithfully including its stage order
       and its sign pairing; the omitted reference-shot branch is noted above.
    """
    time = np.asarray(time, dtype=float)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    r = np.asarray(r, dtype=float)

    measured = impa_lowpass(raw, cutoff_hz, sample_rate) * float(gain)
    measured = measured - measured[:, :1]
    i_tf = impa_lowpass(np.asarray(tf_raw, dtype=float) * float(tf_gain), cutoff_hz, sample_rate)
    i_tf = i_tf - i_tf[0]

    index = int(np.argmin(np.abs(time - float(target_time))))
    lower, upper = float(min(tilt_bounds_deg)), float(max(tilt_bounds_deg))

    tilt = np.empty(measured.shape[0])
    bound_hit = np.zeros(measured.shape[0], dtype=bool)
    compensated = np.empty_like(measured)
    pickup = np.empty_like(measured)
    for channel in range(measured.shape[0]):
        bt_target = toroidal_field(r[channel], i_tf[index], turns)

        def residual(angle: np.ndarray, _bt: float = float(bt_target), _ch: int = channel) -> np.ndarray:
            return np.atleast_1d(_bt * math.sin(math.radians(float(angle[0]))) - measured[_ch, index])

        solution = optimize.least_squares(residual, [0.0], bounds=([lower], [upper]))
        angle = float(solution.x[0])
        tilt[channel] = angle
        # A bounded optimiser stops just short of the bound, so compare with a
        # tolerance scaled to the allowed range rather than exact equality.
        edge = max((upper - lower) * 1e-4, 1e-9)
        bound_hit[channel] = (angle - lower) <= edge or (upper - angle) <= edge
        pickup[channel] = toroidal_field(r[channel], i_tf, turns) * math.sin(math.radians(angle))
        compensated[channel] = (measured[channel] - pickup[channel]) / math.cos(math.radians(angle))

    return {
        "index": index,
        "time": float(time[index]),
        "b_measured": measured,
        "tf_pickup": pickup,
        "b_z": compensated,
        "tilt_deg": tilt,
        "bound_hit": bound_hit,
    }


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _worst(*states: str) -> str:
    order = {"valid": 0, "warning": 1, "invalid": 2}
    return max(states, key=lambda state: order[state])


def grade_impa_quality(
    time: np.ndarray,
    raw: np.ndarray,
    b_measured: np.ndarray,
    b_z: np.ndarray,
    channel_valid: np.ndarray,
    window: TfCalibrationWindow | None,
    geometry: ImpaGeometryFit,
    coupling: ImpaCouplingFit | None,
    *,
    expected_channels: int = IMPA_CHANNEL_COUNT,
    max_normalized_rmse: float = 0.1,
    r_bounds: Sequence[float] = (0.1, 0.9),
    pitch_tolerance: float = 0.01,
    window_reasons: Sequence[str] = (),
    orientation: str = "toroidal",
    alpha_tolerance: float = 0.15,
    crosstalk: ImpaCrosstalkFit | None = None,
    min_crosstalk_r_squared: float = 0.8,
) -> ImpaQuality:
    """Grade whether an array's measurements can be trusted, and say why not.

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
    ImpaQuality
        A verdict of valid, warning or invalid, with the reasons behind it [-].

    Processing steps
    ----------------
    1. Check the channels are present and the time axis usable.
    2. Check each channel's own signal health, including railing.
    3. Check the calibration window, then the geometry, then the coupling.
    4. Check the compensated signal and the vertical sensors.
    5. Combine by taking the worst verdict.

    Convention
    ----------
    The verdict is the **worst** of the checks, not an average, so one invalid
    check makes the result invalid however many pass. Reasons accumulate rather
    than short-circuiting, so a caller sees everything wrong at once.

    Renamed from ``validate_impa`` under issue #337: this layer transforms and
    infers, and grading sits uneasily here even so. The grading itself moves to
    the validation layer under issue #339.

    Defaults
    --------
    The residual, spacing, coupling and goodness-of-fit thresholds are
    hard-coded, and two of them disagree with the values ``vest.yaml`` carries,
    which is tracked in #625. The channel count and the radial bounds are
    machine-specific.

    Applicability
    -------------
    Machine-independent.  Every threshold arrives as an argument.

    Limitations
    -----------
    Grades the fits it is handed; it cannot tell a well-fitted wrong model from a
    right one. A shot whose array faces the toroidal field can pass the geometry
    checks and still have no usable vertical measurement.

    Provenance
    ----------
    .. [1] Issues #337 and #339 for the naming and the eventual move; ``vest.yaml``
       ``impa.quality`` for the configured thresholds.
    """
    checks: dict[str, str] = {}
    reasons: list[str] = list(window_reasons)
    time = np.asarray(time, dtype=float)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    channel_valid = np.asarray(channel_valid, dtype=bool)

    present = int(np.count_nonzero(channel_valid))
    if present == expected_channels:
        checks["channels_present"] = "valid"
    elif present == 0:
        checks["channels_present"] = "invalid"
        reasons.append("no IMPA raw channel could be read")
    else:
        checks["channels_present"] = "warning"
        reasons.append(f"{expected_channels - present} of {expected_channels} IMPA channels are unavailable")

    if time.size >= 2 and np.all(np.isfinite(time)) and np.all(np.diff(time) > 0):
        checks["time_axis"] = "valid"
    else:
        checks["time_axis"] = "invalid"
        reasons.append("IMPA time axis is empty, non-finite or not monotonic")

    channel_states: list[str] = []
    for channel in range(raw.shape[0]):
        if not channel_valid[channel]:
            continue
        values = raw[channel]
        if values.size == 0 or not np.any(np.isfinite(values)):
            channel_states.append("invalid")
            reasons.append(f"channel {channel} carries no finite raw sample")
        elif np.allclose(values, values.flat[0]):
            channel_states.append("invalid")
            reasons.append(f"channel {channel} is constant (dead or railed)")
        else:
            # A railed channel repeats one *exact* digitiser level; a genuine
            # analogue plateau still carries noise, so near-equality would
            # flag every flat TF top as clipping.
            railed = np.count_nonzero(values == np.max(values)) + np.count_nonzero(
                values == np.min(values)
            )
            if railed > 0.01 * values.size:
                channel_states.append("warning")
                reasons.append(f"channel {channel} holds one exact extreme level (probable clipping)")
            else:
                channel_states.append("valid")
    checks["channel_signals"] = _worst("valid", *channel_states) if channel_states else "invalid"

    if window is not None:
        checks["calibration_window"] = "valid"
    elif coupling is not None:
        # A calibration carried in from a reference shot is a legitimate
        # substitute for this shot's own clean interval.
        checks["calibration_window"] = "warning"
        reasons.append("no clean TF window in this shot; calibration taken from a reference shot")
    else:
        checks["calibration_window"] = "invalid"
        reasons.append("no clean TF calibration window was found in this shot")

    lower, upper = float(min(r_bounds)), float(max(r_bounds))
    geometry_state = "valid"
    if not geometry.monotonic:
        geometry_state = "invalid"
        reasons.append("fitted IMPA radii are not monotonically increasing")
    # Only positions that carry a wired channel are measurements; a nominal
    # radius for an unwired position says nothing about the hardware.
    wired = geometry.r[channel_valid] if channel_valid.size == geometry.r.size else geometry.r
    if not geometry.within_bounds or (wired.size and (np.any(wired < lower) or np.any(wired > upper))):
        geometry_state = "invalid"
        reasons.append(f"fitted IMPA radii fall outside the physical bounds [{lower}, {upper}] m")
    if geometry.bound_hit:
        geometry_state = _worst(geometry_state, "invalid")
        reasons.append("the geometry fit saturated at a parameter bound")
    if geometry.pitch is not None and wired.size > 1:
        spacing = np.diff(wired)
        if np.any(np.abs(spacing - geometry.pitch) > pitch_tolerance):
            geometry_state = _worst(geometry_state, "warning")
            reasons.append("channel spacing deviates from the configured radial pitch")
    if geometry.nrmse is not None and np.isfinite(geometry.nrmse) and geometry.nrmse > max_normalized_rmse:
        geometry_state = _worst(geometry_state, "warning")
        reasons.append(
            f"geometry fit residual is {geometry.nrmse * 100:.1f}% of the signal spread, above the "
            f"{max_normalized_rmse * 100:.1f}% tolerance"
        )
    checks["geometry"] = geometry_state

    if coupling is None:
        checks["tf_coupling"] = "invalid"
        reasons.append("TF coupling could not be fitted")
    else:
        coupling_state = "valid"
        if not np.any(np.isfinite(coupling.alpha)):
            coupling_state = "invalid"
            reasons.append("no channel produced a finite TF coupling")
        if orientation == "toroidal":
            # A toroidally mounted probe should see essentially the whole
            # toroidal field; departures mean the array is not where or how
            # the configuration says it is.
            finite_alpha = np.isfinite(coupling.alpha)
            deviation = np.abs(np.abs(coupling.alpha[finite_alpha]) - 1.0)
            if deviation.size and np.max(deviation) > alpha_tolerance:
                coupling_state = _worst(coupling_state, "invalid")
                stray = np.flatnonzero(
                    finite_alpha & (np.abs(np.abs(coupling.alpha) - 1.0) > alpha_tolerance)
                ).tolist()
                reasons.append(
                    f"TF coupling departs from a toroidally aligned probe (|alpha| - 1 > "
                    f"{alpha_tolerance}) on channels {stray}"
                )
        elif np.any(coupling.bound_hit):
            coupling_state = _worst(coupling_state, "invalid")
            hits = np.flatnonzero(coupling.bound_hit).tolist()
            reasons.append(f"TF coupling exceeds the configured tilt bounds on channels {hits}")
        finite = np.isfinite(coupling.nrmse)
        if finite.any() and np.nanmax(coupling.nrmse[finite]) > max_normalized_rmse:
            coupling_state = _worst(coupling_state, "warning")
            reasons.append("TF coupling residuals exceed the configured tolerance on at least one channel")
        checks["tf_coupling"] = coupling_state

    b_z = np.atleast_2d(np.asarray(b_z, dtype=float))
    usable = b_z[channel_valid] if channel_valid.size == b_z.shape[0] else b_z
    if usable.size and np.all(np.isfinite(usable)):
        checks["compensated_signal"] = "valid"
    elif usable.size and np.any(np.isfinite(usable)):
        checks["compensated_signal"] = "warning"
        reasons.append("some compensated Bz samples are not finite")
    else:
        checks["compensated_signal"] = "invalid"
        reasons.append("compensated Bz is not finite on any usable channel")

    if crosstalk is not None:
        bz_state = "valid"
        finite = np.isfinite(crosstalk.slope_v_per_t)
        if not finite.any():
            bz_state = "invalid"
            reasons.append("no Bz sensor produced a finite crosstalk calibration")
        else:
            # A misalignment-angle bound can only be checked once the Bz
            # sensor's own gain is known; bound_hit is all-False without one.
            if np.any(crosstalk.bound_hit[finite]):
                bz_state = _worst(bz_state, "invalid")
                hits = np.flatnonzero(finite & crosstalk.bound_hit).tolist()
                reasons.append(f"Bz sensor incident angle exceeds the configured limit on channels {hits}")
            weak = finite & (crosstalk.r_squared < min_crosstalk_r_squared)
            if np.any(weak):
                bz_state = _worst(bz_state, "warning")
                reasons.append(
                    "the toroidal field does not explain the Bz sensor signal on channels "
                    f"{np.flatnonzero(weak).tolist()}; those sensors may be inactive"
                )
        checks["bz_sensors"] = bz_state

    return ImpaQuality(status=_worst(*checks.values()), checks=checks, reasons=tuple(reasons))


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def validate_impa(*args: Any, **kwargs: Any) -> ImpaQuality:
    """Deprecated compatibility wrapper for :func:`grade_impa_quality`."""
    warnings.warn(
        "vaft.process.validate_impa() is deprecated; use grade_impa_quality(). "
        "vaft.process does not reach verdicts (issues #253, #337); the grading "
        "itself moves to vaft.validation.impa under #339.",
        DeprecationWarning,
        stacklevel=2,
    )
    return grade_impa_quality(*args, **kwargs)


def process_impa(
    time: np.ndarray,
    raw: np.ndarray,
    i_tf: np.ndarray,
    *,
    config: ImpaProcessingConfig | None = None,
    criteria: TfWindowCriteria | None = None,
    ip: np.ndarray | None = None,
    pf_currents: np.ndarray | None = None,
    channel_valid: np.ndarray | None = None,
    r: np.ndarray | None = None,
    z: np.ndarray | float = 0.0,
    pitch: float = 0.05,
    r_bounds: Sequence[float] = (0.1, 0.9),
    r0_initial: float = 0.4,
    max_normalized_rmse: float = 0.1,
    reference: ImpaResult | None = None,
    b_z_raw: np.ndarray | None = None,
    bz_channel_valid: np.ndarray | None = None,
    fit_pitch: bool = False,
    max_crosstalk_angle_deg: float = 30.0,
    min_crosstalk_r_squared: float = 0.8,
    bz_gain: float | None = None,
    bz_radial_offset: float = 0.0,
) -> ImpaResult:
    """Run the whole array chain for one shot, from raw records to a graded result.

    Parameters
    ----------
    time : array_like
        Sample times [s].
    raw : array_like
        Raw probe records as ``(n_channels, n_samples)`` [V].
    i_tf : array_like
        Toroidal field coil current [A].
    config : ImpaProcessingConfig, optional
        Conditioning settings [-].
    criteria : TfWindowCriteria, optional
        Window-selection thresholds [-].
    ip : array_like, optional
        Plasma current [A].
    pf_currents : array_like, optional
        Poloidal field coil currents [A].
    channel_valid : array_like, optional
        Which probe channels to use [-].
    r : array_like, optional
        Configured radii, which take precedence when supplied [m].
    z : float, optional
        Height of the array [m].
    pitch : float, optional
        As-built channel spacing [m].
    r_bounds : tuple of float, optional
        Bounds on the fitted radius [m].
    r0_initial : float, optional
        Starting guess for it [m].
    max_normalized_rmse : float, optional
        Largest acceptable geometry residual [-].
    reference : Mapping, optional
        A reference shot supplying the calibration instead of this shot's own [-].
    b_z_raw : array_like, optional
        Raw vertical-sensor records [V].
    bz_channel_valid : array_like, optional
        Which vertical sensors to use [-].
    fit_pitch : bool, optional
        Whether to fit the spacing too [-].
    max_crosstalk_angle_deg : float, optional
        Largest misalignment treated as physical [deg].
    min_crosstalk_r_squared : float, optional
        Smallest acceptable crosstalk goodness of fit [-].
    bz_gain : float, optional
        Volts-to-tesla gain for the vertical sensors [T/V].
    bz_radial_offset : float, optional
        Radial offset of the vertical sensors from their Hall neighbours [m].

    Returns
    -------
    ImpaResult
        The conditioned fields, the geometry and coupling, the compensated
        vertical field, the quality verdict, and a provenance record [-].

    Processing steps
    ----------------
    1. Condition the raw records into field units.
    2. Mask the invalid channels.
    3. Find the interval driven by the toroidal field alone.
    4. Resolve the geometry by precedence: a reference shot, then a configured
       radius, then a fit on this shot's own window, then the nominal
       uncalibrated positions.
    5. Resolve the coupling, from the reference if there is one, else by fitting.
    6. Remove the toroidal pickup, branching on the array's orientation.
    7. Optionally fit and remove the vertical sensors' crosstalk.
    8. Grade the result and record what was used.

    Convention
    ----------
    **Geometry precedence is deliberate and ordered.**  Self-calibration from the
    shot's own window is the primary source, not a fallback, because the array is
    insertable and its position is a per-shot quantity. A configured radius wins
    only where an era was surveyed or left fixed.

    The vertical sensors are **not** co-located with the Hall channel of the same
    index; the offset is a hardware fact supplied by the caller.

    Defaults
    --------
    The spacing, bounds, starting guess and channel count are machine-specific
    VEST values. The residual threshold and the radial offset disagree with what
    ``vest.yaml`` carries, which is tracked in #625; the VEST pipeline passes the
    configured values, so the disagreement bites only a direct caller.

    Applicability
    -------------
    Machine-independent.  Every machine number is an argument, and the VEST
    pipeline supplies them from configuration.

    Limitations
    -----------
    Inherits every assumption of the steps it orchestrates, above all that the
    selected window is genuinely driven by the toroidal field alone. A shot whose
    array faces that field yields a compensated vertical field that is not a
    measurement, which the grading flags rather than the chain refusing.

    Provenance
    ----------
    .. [1] The stages this orchestrates, each documented on its own function;
       ``vest.yaml`` ``impa`` for the configuration the VEST pipeline passes in.
    """
    config = config or ImpaProcessingConfig()
    time = np.asarray(time, dtype=float)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    i_tf = np.asarray(i_tf, dtype=float)
    n_channels = raw.shape[0]
    if channel_valid is None:
        channel_valid = np.ones(n_channels, dtype=bool)
    channel_valid = np.asarray(channel_valid, dtype=bool)

    b_measured = impa_calibrate_signals(
        raw,
        gain=config.gain,
        cutoff_hz=config.signal_lowpass_hz,
        sample_rate=config.sample_rate,
        baseline=config.baseline,
        baseline_samples=config.baseline_samples,
    )
    b_measured[~channel_valid] = np.nan

    window, window_reasons = find_tf_calibration_window(
        time, i_tf, ip, pf_currents, criteria=criteria or TfWindowCriteria()
    )

    z_values = np.full(n_channels, float(z)) if np.isscalar(z) else np.asarray(z, dtype=float)

    if reference is not None:
        reference_size = int(np.size(reference.geometry.r))
        if reference_size != n_channels:
            raise ValueError(
                f"IMPA reference calibration has {reference_size} channels but this "
                f"shot has {n_channels}; align the reference onto this shot's channel "
                "layout by field code before calling process_impa (see "
                "vaft.machine_mapping.impa.process_impa_shot), or supply a reference "
                "with a matching channel count."
            )
        geometry = ImpaGeometryFit(
            r=np.asarray(reference.geometry.r, dtype=float),
            z=np.asarray(reference.geometry.z, dtype=float),
            method=f"reference_shot:{reference.provenance.get('shot', 'unknown')}",
            r0=reference.geometry.r0,
            pitch=reference.geometry.pitch,
            rmse=reference.geometry.rmse,
            nrmse=reference.geometry.nrmse,
            monotonic=reference.geometry.monotonic,
            within_bounds=reference.geometry.within_bounds,
            n_channels_fitted=reference.geometry.n_channels_fitted,
        )
    elif r is not None:
        geometry = ImpaGeometryFit(
            r=np.asarray(r, dtype=float),
            z=z_values,
            method="configured",
            pitch=float(pitch),
            monotonic=bool(np.all(np.diff(np.asarray(r, dtype=float)) > 0)),
            within_bounds=bool(
                np.all(np.asarray(r, dtype=float) >= min(r_bounds))
                and np.all(np.asarray(r, dtype=float) <= max(r_bounds))
            ),
        )
    elif window is not None:
        geometry = fit_impa_geometry(
            b_measured,
            i_tf,
            window,
            pitch=pitch,
            z=float(np.mean(z_values)),
            turns=config.tf_turns,
            r0_initial=r0_initial,
            r_bounds=r_bounds,
            fit_pitch=fit_pitch,
        )
    else:
        # Without a window there is nothing to calibrate against; record the
        # nominal layout so downstream code still has shapes, and let the
        # quality verdict carry the failure.
        geometry = ImpaGeometryFit(
            r=float(r0_initial) + np.arange(n_channels) * float(pitch),
            z=z_values,
            method="nominal_uncalibrated",
            r0=float(r0_initial),
            pitch=float(pitch),
        )

    coupling = None
    tf_pickup = np.full_like(b_measured, np.nan)
    b_z = np.full_like(b_measured, np.nan)
    if reference is not None and reference.coupling is not None:
        # A reference taken with the array in the same position carries the
        # calibration; this shot only supplies the waveform to correct.
        coupling = reference.coupling
    elif window is not None:
        coupling = fit_impa_tf_coupling(
            b_measured,
            i_tf,
            geometry.r,
            window,
            turns=config.tf_turns,
            tilt_bounds_deg=config.tilt_bounds_deg,
        )
    if coupling is not None:
        bt = toroidal_field(geometry.r[:, None], i_tf[None, :], config.tf_turns)
        tf_pickup = coupling.alpha[:, None] * bt
        if config.orientation == "toroidal":
            # A probe facing the toroidal field has essentially no sensitivity
            # left for the poloidal component, so there is no Bz to recover.
            # What the measurement yields is the toroidal field itself; the
            # residual after removing the modelled TF is reported separately
            # and carries no calibrated projection.
            b_z = b_measured - tf_pickup
        else:
            b_z = remove_tf_pickup(b_measured, tf_pickup, coupling.alpha)

    # The Hall channels have now established where each position is and how
    # the toroidal field appears there; that is what makes the co-located Bz
    # sensors interpretable.
    crosstalk = None
    b_z_corrected = None
    if b_z_raw is not None:
        b_z_raw = np.atleast_2d(np.asarray(b_z_raw, dtype=float))
        if bz_channel_valid is None:
            bz_channel_valid = np.all(np.isfinite(b_z_raw), axis=1)
        bz_channel_valid = np.asarray(bz_channel_valid, dtype=bool)
        if window is not None:
            crosstalk = fit_impa_crosstalk(
                b_measured,
                b_z_raw,
                window,
                max_angle_deg=max_crosstalk_angle_deg,
                bz_gain=bz_gain,
            )
            b_z_corrected = remove_bz_crosstalk(b_z_raw, b_measured, crosstalk)

    quality = grade_impa_quality(
        time,
        raw,
        b_measured,
        b_z,
        channel_valid,
        window,
        geometry,
        coupling,
        expected_channels=n_channels,
        max_normalized_rmse=max_normalized_rmse,
        r_bounds=r_bounds,
        window_reasons=window_reasons,
        orientation=config.orientation,
        alpha_tolerance=config.alpha_tolerance,
        crosstalk=crosstalk,
        min_crosstalk_r_squared=min_crosstalk_r_squared,
    )

    provenance: dict[str, Any] = {
        "gain": float(config.gain),
        "signal_lowpass_hz": float(config.signal_lowpass_hz),
        "sample_rate": float(config.sample_rate),
        "baseline": config.baseline,
        "tf_turns": int(config.tf_turns),
        "geometry_method": geometry.method,
        "orientation": config.orientation,
        "incident_angle_deg": geometry.incident_angle_deg,
        "sign_convention": "I_TF = raw * -3e4; IMPA gain from configuration",
        "reference_shot_used": reference is not None,
    }
    if window is not None:
        provenance["calibration_window"] = {
            "start_time": window.start_time,
            "end_time": window.end_time,
            "n_samples": window.n_samples,
            **{key: float(value) for key, value in window.metrics.items()},
        }

    return ImpaResult(
        time=time,
        b_measured=b_measured,
        tf_pickup=tf_pickup,
        b_z=b_z,
        channel_valid=channel_valid,
        geometry=geometry,
        coupling=coupling,
        window=window,
        quality=quality,
        bz_r=(geometry.r + float(bz_radial_offset)) if b_z_raw is not None else None,
        b_z_raw=b_z_raw,
        b_z_corrected=b_z_corrected,
        crosstalk=crosstalk,
        bz_channel_valid=bz_channel_valid,
        provenance=provenance,
    )
