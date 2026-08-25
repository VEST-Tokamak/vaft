"""Generic algorithms for the VEST Internal Magnetic Probe Array (IMPA).

The IMPA is an eight-channel Hall-probe array on the VEST midplane.  Every
routine here works on plain arrays so the machine-specific parts -- raw field
codes, shot-era geometry and thresholds -- stay in
``vaft.machine_mapping.impa`` and ``vaft/machine_mapping/vest.yaml``.

Measurement model
-----------------
Each probe sees a mixture of the local poloidal field and the toroidal field::

    B_meas_i(t) = alpha_i * Bt(R_i, t) + Bz_i(t) * sqrt(1 - alpha_i**2) + beta_i

with ``Bt(R, t) = mu0 * N_TF * I_TF(t) / (2 * pi * R)`` and ``alpha_i = sin(theta_i)``
for a probe tilted by ``theta_i`` away from the vertical.  Inverting it gives the
compensated internal field::

    Bz_i = (B_meas_i - alpha_i * Bt(R_i)) / sqrt(1 - alpha_i**2)

Sign convention
---------------
VAFT uses ``I_TF = raw_field_1 * -3e4`` (as :mod:`vaft.machine_mapping.tf`
already does) together with an IMPA gain of ``+2/15`` T/V.  The legacy MATLAB
``VEST_IMPAProcessing.m`` used ``-2/15`` with ``+3e4``; both give an identical
``B_meas / Bt`` ratio, so the two conventions differ only in the overall
polarity of the reported ``Bz``.  ``vest_impa_position.m`` already used the
convention adopted here.

Geometry / coupling degeneracy
------------------------------
During a TF-only interval the toroidal field is the *only* driver, so a probe
measures ``alpha_i * mu0 * N * I_TF / (2 * pi * R_i)``.  Only the ratio
``kappa_i = alpha_i / R_i`` is observable: assuming ``alpha_i = 1`` yields the
radial position (the legacy ``vest_impa_position`` method), while a known
``R_i`` yields the coupling.  The two cannot be recovered simultaneously from
TF data alone, which is why configured shot-era geometry always takes
precedence and every self-calibrated result carries its fit quality.

What shot 39204 shows
---------------------
On the reference shot the fitted coupling reaches ``alpha ~ 1`` on most
channels: the probes measure essentially the whole toroidal field rather than a
small tilt pickup.  Consequences, all reproduced by the tests:

* the legacy ``+/-10 degree`` tilt fit saturates at its bound on every channel
  and returns a "compensated Bz" of order 0.1 T, two orders of magnitude above
  VEST's real vertical field;
* the legacy uniform 5 cm pitch model leaves a ~30% residual, and no static
  geometry -- not even a free radius per channel -- explains the measurement to
  better than ~20% across the TF ramp;
* projecting such a probe back onto the vertical axis is unbounded, so this
  implementation reports the shot as ``invalid`` instead of emitting a
  plausible-looking number.

Shot 39923 is a second reference, at a tenth of the TF drive (1.3 kA against
12.7 kA).  It reproduces the channel ordering and, to within ~12%, the shape of
the radial profile -- but its response per unit TF current is smaller by a
factor of about 1.45, and the fitted ``R0`` lands 27 cm away.  Probe geometry
cannot move that far between shots, so the two shots do not agree on a
calibration.

Extracting a vertical field from this array therefore needs an independent
geometry/orientation survey recorded in ``vest.yaml``; the self-calibration
path exists to validate such a survey, not to replace it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage, optimize, signal

__all__ = [
    "IMPA_CHANNEL_COUNT",
    "ImpaCouplingFit",
    "ImpaGeometryFit",
    "ImpaProcessingConfig",
    "ImpaQuality",
    "ImpaResult",
    "TfCalibrationWindow",
    "TfWindowCriteria",
    "find_tf_calibration_window",
    "fit_impa_geometry",
    "fit_impa_tf_coupling",
    "impa_calibrate_signals",
    "impa_lowpass",
    "legacy_impa_compensation",
    "legacy_impa_position",
    "process_impa",
    "remove_tf_pickup",
    "toroidal_field",
    "validate_impa",
]

MU0 = 4.0e-7 * math.pi
IMPA_CHANNEL_COUNT = 8

#: Legacy ``vest_filter(fs, fc, 1)`` is ``designfilt('lowpassfir',
#: 'FilterOrder', round(fs*1e-3), ...)`` -- a Hamming-window FIR of order 25,
#: i.e. 26 taps, applied with ``filtfilt``.
_FIR_TAPS = 26

_VALID_BASELINES = ("first_sample", "mean_first_samples", "none")


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

    def __post_init__(self) -> None:
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
    rmse: float | None = None
    nrmse: float | None = None
    monotonic: bool = True
    within_bounds: bool = True
    bound_hit: bool = False
    n_samples: int = 0


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
    b_z: np.ndarray
    channel_valid: np.ndarray
    geometry: ImpaGeometryFit
    coupling: ImpaCouplingFit | None
    window: TfCalibrationWindow | None
    quality: ImpaQuality
    provenance: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# signal conditioning
# --------------------------------------------------------------------------
def impa_lowpass(values: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Zero-phase FIR low pass matching the legacy ``vest_filter`` design.

    ``filtfilt`` needs more samples than the filter's padding length; tiny
    synthetic dumps fall back to a forward filter, the same accommodation
    :mod:`vaft.machine_mapping.pf_active` already makes.
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
    """Filter, gain-calibrate and baseline-correct raw IMPA voltages.

    The stage order follows ``VEST_IMPAProcessing.m``: low pass, then gain,
    then baseline removal.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    filtered = impa_lowpass(raw, cutoff_hz, sample_rate) * float(gain)
    return _apply_baseline(filtered, baseline, baseline_samples)


def toroidal_field(r: np.ndarray | float, i_tf: np.ndarray | float, turns: int = 24) -> np.ndarray:
    """Vacuum toroidal field ``mu0 * N * I_TF / (2 * pi * R)``."""
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
    """Find the longest same-shot interval where the TF alone drives the probes.

    Returns ``(window, reasons)``.  ``window`` is ``None`` when no interval
    satisfies every criterion -- an arbitrary fallback sample is never chosen.
    ``reasons`` always explains which criteria eliminated candidates.
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
    """Regress ``B_meas_i = alpha_i * Bt(R_i) + beta_i`` over a whole window.

    Fitting a full interval rather than the single legacy time sample gives
    residual statistics that make a bad calibration visible.
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
        rmse[channel] = float(np.sqrt(np.mean(residual**2)))
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
    r_bounds: Sequence[float] = (0.1, 0.8),
) -> ImpaGeometryFit:
    """Self-calibrate ``R_0`` from the TF ``1/R`` profile over a full window.

    This is the ``vest_impa_position`` model -- uniform channel pitch and unit
    TF coupling -- fitted across every window sample instead of one hard-coded
    time.  It is a fallback for shots without configured geometry, and its
    residuals must be checked before the result is trusted.
    """
    b_measured = np.atleast_2d(np.asarray(b_measured, dtype=float))
    i_tf = np.asarray(i_tf, dtype=float)
    idx = np.asarray(window.indices, dtype=int)
    n_channels = b_measured.shape[0]
    offsets = np.arange(n_channels) * float(pitch)
    observed = b_measured[:, idx]
    lower, upper = float(min(r_bounds)), float(max(r_bounds))

    def residual(params: np.ndarray) -> np.ndarray:
        radii = params[0] + offsets
        model = toroidal_field(radii[:, None], i_tf[None, idx], turns)
        return (observed - model).ravel()

    solution = optimize.least_squares(
        residual,
        [float(np.clip(r0_initial, lower, upper))],
        bounds=([lower], [upper]),
    )
    r0 = float(solution.x[0])
    final = residual(solution.x)
    rmse = float(np.sqrt(np.mean(final**2)))
    spread = float(np.std(observed))
    radii = r0 + offsets

    return ImpaGeometryFit(
        r=radii,
        z=np.full(n_channels, float(z)),
        method="tf_profile_fit",
        r0=r0,
        pitch=float(pitch),
        rmse=rmse,
        nrmse=rmse / spread if spread > 0 else float("inf"),
        monotonic=bool(np.all(np.diff(radii) > 0)),
        within_bounds=bool(lower <= r0 <= upper),
        bound_hit=bool((r0 - lower) <= _bound_edge(lower, upper) or (upper - r0) <= _bound_edge(lower, upper)),
        n_samples=int(idx.size),
    )


def remove_tf_pickup(b_measured: np.ndarray, tf_pickup: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return ``Bz = (B_meas - alpha * Bt) / sqrt(1 - alpha**2)``.

    ``tf_pickup`` is the already-scaled ``alpha * Bt`` contribution.  Channels
    whose coupling approaches unity are toroidally aligned; the projection back
    onto the vertical axis is then unbounded, so they yield NaN rather than a
    plausible-looking but meaningless number.
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
    r_bounds: Sequence[float] = (0.1, 0.8),
    baseline_samples: int = 2_500,
) -> dict[str, Any]:
    """Faithful port of ``vest_impa_position.m`` (single time sample).

    Kept for numerical parity checks against the legacy pipeline.  Its stage
    order differs from :func:`impa_calibrate_signals` on purpose: offset first,
    then gain, then the 2.5 kHz filter, then a 2500-sample mean removal.  The
    TF trace is deliberately *not* offset-corrected, exactly as in MATLAB.
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
        "rmse": float(np.sqrt(np.mean(final**2))),
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
    """Faithful port of ``VEST_IMPAProcessing.m`` for a single shot.

    The legacy routine also processed a reference shot through this same path;
    that branch is intentionally absent because calibration never needed it --
    the reference waveform only mattered for the later equilibrium residual.
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
    expected_channels: int = IMPA_CHANNEL_COUNT,
    max_normalized_rmse: float = 0.1,
    r_bounds: Sequence[float] = (0.1, 0.8),
    pitch_tolerance: float = 0.01,
    window_reasons: Sequence[str] = (),
) -> ImpaQuality:
    """Grade one processed IMPA shot as ``valid``/``warning``/``invalid``."""
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
            extremes = np.count_nonzero(np.isclose(values, np.max(values))) + np.count_nonzero(
                np.isclose(values, np.min(values))
            )
            if extremes > 0.01 * values.size:
                channel_states.append("warning")
                reasons.append(f"channel {channel} spends >1% of the shot at a rail (possible clipping)")
            else:
                channel_states.append("valid")
    checks["channel_signals"] = _worst("valid", *channel_states) if channel_states else "invalid"

    if window is None:
        checks["calibration_window"] = "invalid"
        reasons.append("no clean TF calibration window was found in this shot")
    else:
        checks["calibration_window"] = "valid"

    lower, upper = float(min(r_bounds)), float(max(r_bounds))
    geometry_state = "valid"
    if not geometry.monotonic:
        geometry_state = "invalid"
        reasons.append("fitted IMPA radii are not monotonically increasing")
    if not geometry.within_bounds or np.any(geometry.r < lower) or np.any(geometry.r > upper):
        geometry_state = "invalid"
        reasons.append(f"fitted IMPA radii fall outside the physical bounds [{lower}, {upper}] m")
    if geometry.bound_hit:
        geometry_state = _worst(geometry_state, "invalid")
        reasons.append("the geometry fit saturated at a parameter bound")
    if geometry.pitch is not None and geometry.r.size > 1:
        spacing = np.diff(geometry.r)
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
        if np.any(coupling.bound_hit):
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

    return ImpaQuality(status=_worst(*checks.values()), checks=checks, reasons=tuple(reasons))


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
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
    r_bounds: Sequence[float] = (0.1, 0.8),
    r0_initial: float = 0.4,
    max_normalized_rmse: float = 0.1,
) -> ImpaResult:
    """Run the full single-shot IMPA pipeline.

    Pass ``r`` to use configured shot-era geometry (always preferred); leave it
    ``None`` to self-calibrate the radial positions from the clean TF window
    with the legacy ``1/R`` model.  No reference or vacuum shot is involved at
    any stage.
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

    if r is not None:
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
    if window is not None:
        coupling = fit_impa_tf_coupling(
            b_measured,
            i_tf,
            geometry.r,
            window,
            turns=config.tf_turns,
            tilt_bounds_deg=config.tilt_bounds_deg,
        )
        bt = toroidal_field(geometry.r[:, None], i_tf[None, :], config.tf_turns)
        tf_pickup = coupling.alpha[:, None] * bt
        b_z = remove_tf_pickup(b_measured, tf_pickup, coupling.alpha)

    quality = validate_impa(
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
    )

    provenance: dict[str, Any] = {
        "gain": float(config.gain),
        "signal_lowpass_hz": float(config.signal_lowpass_hz),
        "sample_rate": float(config.sample_rate),
        "baseline": config.baseline,
        "tf_turns": int(config.tf_turns),
        "geometry_method": geometry.method,
        "sign_convention": "I_TF = raw * -3e4 with IMPA gain +2/15 T/V",
        "reference_shot_used": False,
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
        provenance=provenance,
    )
